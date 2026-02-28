"""AWS Lambda handler for document processing."""

import asyncio
from typing import Any

from app.core.logging import get_logger, setup_logging
from app.workers.processing import delete_document_data, process_document, reprocess_document

setup_logging()
logger = get_logger(__name__)


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda entry point for document processing tasks.

    Event format:
        {
            "action": "process" | "delete" | "reprocess",
            "doc_id": "...",
            "user_id": "...",
            "s3_key": "...",
            "filename": "..."  # required for process/reprocess
        }
    """
    logger.info("Lambda invoked", action=event.get("action"), doc_id=event.get("doc_id"))

    action = event.get("action")
    if not action:
        return {"status": "error", "error": "Missing 'action' in event"}

    try:
        result = asyncio.run(_dispatch(event))
        logger.info("Lambda completed", action=action, result=result)
        return result
    except Exception as e:
        logger.error("Lambda failed", action=action, error=str(e))
        return {"status": "error", "error": str(e)}


async def _dispatch(event: dict[str, Any]) -> dict[str, Any]:
    """Route to the appropriate task function."""
    action = event["action"]
    doc_id = event["doc_id"]
    user_id = event["user_id"]
    s3_key = event["s3_key"]

    if action == "process":
        return await process_document(
            doc_id=doc_id,
            user_id=user_id,
            s3_key=s3_key,
            filename=event["filename"],
        )
    elif action == "delete":
        return await delete_document_data(
            doc_id=doc_id,
            user_id=user_id,
            s3_key=s3_key,
        )
    elif action == "reprocess":
        return await reprocess_document(
            doc_id=doc_id,
            user_id=user_id,
            s3_key=s3_key,
            filename=event["filename"],
        )
    else:
        return {"status": "error", "error": f"Unknown action: {action}"}
