"""Amazon S3 storage service for document management."""

import functools
import io
from contextlib import AsyncExitStack
from typing import Any, BinaryIO

from botocore.exceptions import ClientError

from app.config import settings
from app.core.aws import get_aioboto3_session
from app.core.exceptions import NotFoundError, ServiceUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)


class StorageService:
    """Service for S3 file operations with a long-lived client."""

    def __init__(self) -> None:
        self._session = get_aioboto3_session()
        self._bucket = settings.s3_bucket_name
        self._s3: Any = None
        self._exit_stack: AsyncExitStack | None = None

    async def connect(self) -> None:
        """Open a long-lived S3 client."""
        if self._s3 is not None:
            return  # Already connected
        self._exit_stack = AsyncExitStack()
        self._s3 = await self._exit_stack.enter_async_context(
            self._session.client(
                "s3",
                region_name=settings.aws_region,
                endpoint_url=settings.s3_endpoint_url,
            )
        )
        logger.info("S3 client opened", bucket=self._bucket)

    async def close(self) -> None:
        """Tear down the S3 client."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._s3 = None
            logger.info("S3 client closed")

    @property
    def s3(self) -> Any:
        """Return the long-lived S3 client."""
        if self._s3 is None:
            raise RuntimeError("StorageService not connected — call connect() first")
        return self._s3

    async def upload_file(
        self,
        file_data: BinaryIO | bytes,
        s3_key: str,
        content_type: str = "application/pdf",
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload a file to S3."""
        try:
            if isinstance(file_data, bytes):
                file_data = io.BytesIO(file_data)

            extra_args = {"ContentType": content_type}
            if metadata:
                extra_args["Metadata"] = metadata

            await self.s3.upload_fileobj(
                file_data,
                self._bucket,
                s3_key,
                ExtraArgs=extra_args,
            )

            logger.info("Uploaded file to S3", s3_key=s3_key, bucket=self._bucket)
            return s3_key

        except ClientError as e:
            logger.error("S3 upload failed", error=str(e), s3_key=s3_key)
            raise ServiceUnavailableError(
                "Failed to upload file to storage",
                service="s3",
            ) from e

    async def download_file(self, s3_key: str) -> bytes:
        """Download a file from S3."""
        try:
            response = await self.s3.get_object(Bucket=self._bucket, Key=s3_key)
            content = await response["Body"].read()
            logger.debug("Downloaded file from S3", s3_key=s3_key, size=len(content))
            return content

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "NoSuchKey":
                raise NotFoundError(
                    f"File not found: {s3_key}",
                    resource_type="file",
                    resource_id=s3_key,
                ) from e

            logger.error("S3 download failed", error=str(e), s3_key=s3_key)
            raise ServiceUnavailableError(
                "Failed to download file from storage",
                service="s3",
            ) from e

    async def generate_presigned_url(
        self,
        s3_key: str,
        expiry_seconds: int | None = None,
        for_upload: bool = False,
        content_type: str | None = None,
    ) -> str:
        """Generate a presigned URL for direct access."""
        expiry = expiry_seconds or settings.s3_presigned_url_expiry

        try:
            if for_upload:
                params: dict[str, str] = {
                    "Bucket": self._bucket,
                    "Key": s3_key,
                }
                if content_type:
                    params["ContentType"] = content_type

                url = await self.s3.generate_presigned_url(
                    "put_object",
                    Params=params,
                    ExpiresIn=expiry,
                )
            else:
                url = await self.s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self._bucket, "Key": s3_key},
                    ExpiresIn=expiry,
                )

            logger.debug(
                "Generated presigned URL",
                s3_key=s3_key,
                for_upload=for_upload,
                expiry=expiry,
            )
            return url

        except ClientError as e:
            logger.error("Failed to generate presigned URL", error=str(e), s3_key=s3_key)
            raise ServiceUnavailableError(
                "Failed to generate access URL",
                service="s3",
            ) from e

    async def delete_file(self, s3_key: str) -> bool:
        """Delete a file from S3."""
        try:
            await self.s3.delete_object(Bucket=self._bucket, Key=s3_key)
            logger.info("Deleted file from S3", s3_key=s3_key)
            return True

        except ClientError as e:
            logger.error("S3 delete failed", error=str(e), s3_key=s3_key)
            return False

    async def delete_files_by_prefix(self, prefix: str) -> int:
        """Delete all files with a given prefix."""
        deleted_count = 0

        try:
            paginator = self.s3.get_paginator("list_objects_v2")

            async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                contents = page.get("Contents", [])
                if not contents:
                    continue

                objects = [{"Key": obj["Key"]} for obj in contents]
                await self.s3.delete_objects(
                    Bucket=self._bucket,
                    Delete={"Objects": objects},
                )
                deleted_count += len(objects)

            logger.info(
                "Deleted files by prefix",
                prefix=prefix,
                count=deleted_count,
            )
            return deleted_count

        except ClientError as e:
            logger.error("S3 batch delete failed", error=str(e), prefix=prefix)
            return deleted_count

    async def file_exists(self, s3_key: str) -> bool:
        """Check if a file exists in S3."""
        try:
            await self.s3.head_object(Bucket=self._bucket, Key=s3_key)
            return True
        except ClientError:
            return False

    async def get_file_metadata(self, s3_key: str) -> dict[str, str] | None:
        """Get file metadata from S3."""
        try:
            response = await self.s3.head_object(Bucket=self._bucket, Key=s3_key)
            return {
                "content_type": response.get("ContentType", ""),
                "content_length": str(response.get("ContentLength", 0)),
                "last_modified": response.get("LastModified", "").isoformat()
                if response.get("LastModified")
                else "",
                **response.get("Metadata", {}),
            }
        except ClientError:
            return None


@functools.lru_cache
def get_storage_service() -> StorageService:
    """Get or create the storage service singleton."""
    return StorageService()
