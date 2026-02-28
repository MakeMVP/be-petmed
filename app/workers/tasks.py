"""Lambda invocation helpers — used by the API to enqueue background tasks."""

import json
from typing import Any

import boto3

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _get_lambda_client():
    """Get boto3 Lambda client with explicit credentials."""
    return boto3.client(
        "lambda",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )


def _invoke_lambda(payload: dict[str, Any]) -> None:
    """Invoke the document worker Lambda asynchronously."""
    client = _get_lambda_client()
    client.invoke(
        FunctionName=settings.lambda_function_name,
        InvocationType="Event",
        Payload=json.dumps(payload),
    )
    logger.info(
        "Lambda invoked",
        action=payload.get("action"),
        doc_id=payload.get("doc_id"),
    )


async def enqueue_document_processing(
    doc_id: str,
    user_id: str,
    s3_key: str,
    filename: str,
) -> str:
    """Enqueue a document processing task via Lambda."""
    _invoke_lambda({
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
    _invoke_lambda({
        "action": "delete",
        "doc_id": doc_id,
        "user_id": user_id,
        "s3_key": s3_key,
    })
    return doc_id
