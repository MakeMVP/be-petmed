"""Background task definitions and Lambda invocation helpers."""

import json
from typing import Any

import boto3

from app.config import settings
from app.core.logging import get_logger, setup_logging
from app.db.dynamodb import get_dynamodb_client
from app.models.entities import Chunk, DocumentStatus
from app.services.embedding_service import get_embedding_service
from app.services.pdf_service import get_pdf_service
from app.services.pinecone_service import get_pinecone_service
from app.services.storage_service import get_storage_service

logger = get_logger(__name__)


async def process_document(
    doc_id: str,
    user_id: str,
    s3_key: str,
    filename: str,
) -> dict[str, Any]:
    """Process an uploaded PDF document.

    1. Downloads the PDF from S3
    2. Extracts text from all pages
    3. Chunks the text for RAG
    4. Generates embeddings for each chunk
    5. Stores chunks in DynamoDB
    6. Upserts embeddings to Pinecone
    """
    setup_logging()
    logger.info("Starting document processing", doc_id=doc_id, filename=filename)

    db = get_dynamodb_client()
    storage = get_storage_service()
    pdf_service = get_pdf_service()
    embedding_service = get_embedding_service()
    pinecone = get_pinecone_service()

    try:
        await db.update_item(
            pk=f"USER#{user_id}",
            sk=f"DOC#{doc_id}",
            updates={"status": DocumentStatus.PROCESSING.value},
        )

        logger.debug("Downloading PDF from S3", s3_key=s3_key)
        pdf_data = await storage.download_file(s3_key)

        logger.debug("Extracting text from PDF", doc_id=doc_id)
        extracted = await pdf_service.extract_text(pdf_data, filename)

        logger.debug("Chunking document", doc_id=doc_id, pages=extracted.page_count)
        chunks = await pdf_service.chunk_document(extracted)

        if not chunks:
            raise ValueError("No text could be extracted from the document")

        logger.debug("Generating embeddings", doc_id=doc_id, chunks=len(chunks))
        chunk_texts = [c.content for c in chunks]
        embeddings = await embedding_service.embed_documents(chunk_texts)

        doc_item = await db.get_item(
            pk=f"USER#{user_id}",
            sk=f"DOC#{doc_id}",
        )
        document_title = doc_item.get("title", filename) if doc_item else filename

        chunk_items = []
        vector_items = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_obj = Chunk(
                doc_id=doc_id,
                user_id=user_id,
                content=chunk.content,
                page_number=chunk.page_number,
                chunk_index=i,
                token_count=chunk.token_count,
                metadata=chunk.metadata,
            )

            chunk_items.append(chunk_obj.to_dynamodb_item())

            vector_items.append({
                "id": chunk_obj.chunk_id,
                "values": embedding,
                "metadata": {
                    "doc_id": doc_id,
                    "user_id": user_id,
                    "chunk_index": i,
                    "page_number": chunk.page_number,
                    "content": chunk.content[:1000],
                    "document_title": document_title,
                },
            })

            chunk_obj.embedding_id = chunk_obj.chunk_id

        logger.debug("Storing chunks in DynamoDB", count=len(chunk_items))
        await db.batch_write(chunk_items)

        logger.debug("Upserting vectors to Pinecone", count=len(vector_items))
        await pinecone.upsert_vectors(vector_items)

        await db.update_item(
            pk=f"USER#{user_id}",
            sk=f"DOC#{doc_id}",
            updates={
                "status": DocumentStatus.COMPLETED.value,
                "page_count": extracted.page_count,
                "chunk_count": len(chunks),
                "metadata": {
                    **extracted.metadata,
                    "avg_quality_score": sum(p.quality_score for p in extracted.pages)
                    / len(extracted.pages),
                },
            },
        )

        logger.info(
            "Document processing completed",
            doc_id=doc_id,
            pages=extracted.page_count,
            chunks=len(chunks),
        )

        return {
            "doc_id": doc_id,
            "status": "completed",
            "page_count": extracted.page_count,
            "chunk_count": len(chunks),
        }

    except Exception as e:
        logger.error("Document processing failed", doc_id=doc_id, error=str(e))
        await db.update_item(
            pk=f"USER#{user_id}",
            sk=f"DOC#{doc_id}",
            updates={
                "status": DocumentStatus.FAILED.value,
                "error_message": str(e),
            },
        )
        return {
            "doc_id": doc_id,
            "status": "failed",
            "error": str(e),
        }


async def delete_document_data(
    doc_id: str,
    user_id: str,
    s3_key: str,
) -> dict[str, Any]:
    """Delete all data associated with a document."""
    setup_logging()
    logger.info("Starting document deletion", doc_id=doc_id)

    db = get_dynamodb_client()
    storage = get_storage_service()
    pinecone = get_pinecone_service()

    try:
        await storage.delete_file(s3_key)

        chunks, _ = await db.query(
            pk=f"DOC#{doc_id}",
            sk_begins_with="CHUNK#",
        )

        if chunks:
            chunk_keys = [(c["PK"], c["SK"]) for c in chunks]
            await db.batch_delete(chunk_keys)

            chunk_ids = [c["chunk_id"] for c in chunks]
            await pinecone.delete_vectors(ids=chunk_ids)

        await db.delete_item(
            pk=f"USER#{user_id}",
            sk=f"DOC#{doc_id}",
        )

        logger.info("Document deletion completed", doc_id=doc_id)
        return {
            "doc_id": doc_id,
            "status": "deleted",
            "chunks_deleted": len(chunks) if chunks else 0,
        }

    except Exception as e:
        logger.error("Document deletion failed", doc_id=doc_id, error=str(e))
        return {
            "doc_id": doc_id,
            "status": "failed",
            "error": str(e),
        }


async def reprocess_document(
    doc_id: str,
    user_id: str,
    s3_key: str,
    filename: str,
) -> dict[str, Any]:
    """Reprocess an existing document — deletes existing chunks/vectors, then reprocesses."""
    setup_logging()
    logger.info("Starting document reprocessing", doc_id=doc_id)

    db = get_dynamodb_client()
    pinecone = get_pinecone_service()

    try:
        chunks, _ = await db.query(
            pk=f"DOC#{doc_id}",
            sk_begins_with="CHUNK#",
        )

        if chunks:
            chunk_keys = [(c["PK"], c["SK"]) for c in chunks]
            await db.batch_delete(chunk_keys)

            chunk_ids = [c["chunk_id"] for c in chunks]
            await pinecone.delete_vectors(ids=chunk_ids)

        return await process_document(doc_id, user_id, s3_key, filename)

    except Exception as e:
        logger.error("Document reprocessing failed", doc_id=doc_id, error=str(e))
        return {
            "doc_id": doc_id,
            "status": "failed",
            "error": str(e),
        }


# --- Lambda invocation helpers ---

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
        InvocationType="Event",  # async — returns immediately
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
