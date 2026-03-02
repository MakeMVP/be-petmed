"""Lambda invocation helpers — used by the API to enqueue background tasks."""

import asyncio
import json
from typing import Any

from app.config import settings
from app.core.aws import get_boto3_client
from app.core.logging import get_logger

logger = get_logger(__name__)


def _invoke_lambda(payload: dict[str, Any]) -> None:
    """Invoke the document worker Lambda asynchronously."""
    client = get_boto3_client("lambda")
    response = client.invoke(
        FunctionName=settings.lambda_function_name,
        InvocationType="Event",
        Payload=json.dumps(payload),
    )
    logger.info(
        "Lambda invoked",
        action=payload.get("action"),
        doc_id=payload.get("doc_id"),
        status_code=response.get("StatusCode"),
    )


async def enqueue_document_processing(
    doc_id: str,
    user_id: str,
    s3_key: str,
    filename: str,
) -> str:
    """Enqueue a document processing task via Lambda."""
    await asyncio.to_thread(_invoke_lambda, {
        "action": "process",
        "doc_id": doc_id,
        "user_id": user_id,
        "s3_key": s3_key,
        "filename": filename,
    })
    return doc_id


async def enqueue_document_deletion(
    doc_id: str,
    user_id: str,
    s3_key: str,
) -> str:
    """Enqueue a document deletion task via Lambda."""
    await asyncio.to_thread(_invoke_lambda, {
        "action": "delete",
        "doc_id": doc_id,
        "user_id": user_id,
        "s3_key": s3_key,
    })
    return doc_id
