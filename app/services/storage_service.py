"""Amazon S3 storage service for document management."""

import io
from typing import BinaryIO

from botocore.exceptions import ClientError

from app.config import settings
from app.core.aws import get_aioboto3_session
from app.core.exceptions import NotFoundError, ServiceUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)


class StorageService:
    """Service for S3 file operations."""

    def __init__(self) -> None:
        self._session = get_aioboto3_session()
        self._bucket = settings.s3_bucket_name

    async def upload_file(
        self,
        file_data: BinaryIO | bytes,
        s3_key: str,
        content_type: str = "application/pdf",
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload a file to S3.

        Args:
            file_data: File content as bytes or file-like object.
            s3_key: S3 object key.
            content_type: MIME type of the file.
            metadata: Optional metadata to attach.

        Returns:
            The S3 key of the uploaded file.

        Raises:
            ServiceUnavailableError: If upload fails.
        """
        try:
            async with self._session.client(
                "s3",
                region_name=settings.aws_region,
                endpoint_url=settings.s3_endpoint_url,
            ) as s3:
                # Convert bytes to BytesIO if needed
                if isinstance(file_data, bytes):
                    file_data = io.BytesIO(file_data)

                extra_args = {"ContentType": content_type}
                if metadata:
                    extra_args["Metadata"] = metadata

                await s3.upload_fileobj(
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
        """Download a file from S3.

        Args:
            s3_key: S3 object key.

        Returns:
            File content as bytes.

        Raises:
            NotFoundError: If file doesn't exist.
            ServiceUnavailableError: If download fails.
        """
        try:
            async with self._session.client(
                "s3",
                region_name=settings.aws_region,
                endpoint_url=settings.s3_endpoint_url,
            ) as s3:
                response = await s3.get_object(Bucket=self._bucket, Key=s3_key)
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
        """Generate a presigned URL for direct access.

        Args:
            s3_key: S3 object key.
            expiry_seconds: URL expiry time in seconds.
            for_upload: If True, generate upload URL; otherwise download.
            content_type: Content type for upload URLs.

        Returns:
            Presigned URL string.
        """
        expiry = expiry_seconds or settings.s3_presigned_url_expiry

        try:
            async with self._session.client(
                "s3",
                region_name=settings.aws_region,
                endpoint_url=settings.s3_endpoint_url,
            ) as s3:
                if for_upload:
                    params = {
                        "Bucket": self._bucket,
                        "Key": s3_key,
                    }
                    if content_type:
                        params["ContentType"] = content_type

                    url = await s3.generate_presigned_url(
                        "put_object",
                        Params=params,
                        ExpiresIn=expiry,
                    )
                else:
                    url = await s3.generate_presigned_url(
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
        """Delete a file from S3.

        Args:
            s3_key: S3 object key.

        Returns:
            True if deleted successfully.
        """
        try:
            async with self._session.client(
                "s3",
                region_name=settings.aws_region,
                endpoint_url=settings.s3_endpoint_url,
            ) as s3:
                await s3.delete_object(Bucket=self._bucket, Key=s3_key)
                logger.info("Deleted file from S3", s3_key=s3_key)
                return True

        except ClientError as e:
            logger.error("S3 delete failed", error=str(e), s3_key=s3_key)
            return False

    async def delete_files_by_prefix(self, prefix: str) -> int:
        """Delete all files with a given prefix.

        Args:
            prefix: S3 key prefix.

        Returns:
            Number of files deleted.
        """
        deleted_count = 0

        try:
            async with self._session.client(
                "s3",
                region_name=settings.aws_region,
                endpoint_url=settings.s3_endpoint_url,
            ) as s3:
                paginator = s3.get_paginator("list_objects_v2")

                async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                    contents = page.get("Contents", [])
                    if not contents:
                        continue

                    # Delete in batches of 1000 (S3 limit)
                    objects = [{"Key": obj["Key"]} for obj in contents]
                    await s3.delete_objects(
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
        """Check if a file exists in S3.

        Args:
            s3_key: S3 object key.

        Returns:
            True if file exists.
        """
        try:
            async with self._session.client(
                "s3",
                region_name=settings.aws_region,
                endpoint_url=settings.s3_endpoint_url,
            ) as s3:
                await s3.head_object(Bucket=self._bucket, Key=s3_key)
                return True
        except ClientError:
            return False

    async def get_file_metadata(self, s3_key: str) -> dict[str, str] | None:
        """Get file metadata from S3.

        Args:
            s3_key: S3 object key.

        Returns:
            Metadata dict or None if file doesn't exist.
        """
        try:
            async with self._session.client(
                "s3",
                region_name=settings.aws_region,
                endpoint_url=settings.s3_endpoint_url,
            ) as s3:
                response = await s3.head_object(Bucket=self._bucket, Key=s3_key)
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


# Singleton instance
_storage_service: StorageService | None = None


def get_storage_service() -> StorageService:
    """Get or create the storage service singleton."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
