"""ARQ background task definitions."""

import asyncio
from typing import Any

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from app.config import settings
from app.core.logging import get_logger, setup_logging
from app.db.dynamodb import get_dynamodb_client
from app.models.entities import Chunk, Document, DocumentStatus
from app.services.embedding_service import get_embedding_service
from app.services.pdf_service import get_pdf_service
from app.services.pinecone_service import get_pinecone_service
from app.services.storage_service import get_storage_service

logger = get_logger(__name__)


async def process_document(
    ctx: dict[str, Any],
    doc_id: str,
    user_id: str,
    s3_key: str,
    filename: str,
) -> dict[str, Any]:
    """Process an uploaded PDF document.

    This task:
    1. Downloads the PDF from S3
    2. Extracts text from all pages
    3. Chunks the text for RAG
    4. Generates embeddings for each chunk
    5. Stores chunks in DynamoDB
    6. Upserts embeddings to Pinecone

    Args:
        ctx: ARQ context.
        doc_id: Document ID.
        user_id: Owner user ID.
        s3_key: S3 object key.
        filename: Original filename.

    Returns:
        Processing result with chunk count.
    """
    setup_logging()
    logger.info("Starting document processing", doc_id=doc_id, filename=filename)

    db = get_dynamodb_client()
    storage = get_storage_service()
    pdf_service = get_pdf_service()
    embedding_service = get_embedding_service()
    pinecone = get_pinecone_service()

    try:
        # Update status to processing
        await db.update_item(
            pk=f"USER#{user_id}",
            sk=f"DOC#{doc_id}",
            updates={"status": DocumentStatus.PROCESSING.value},
        )

        # Step 1: Download PDF from S3
        logger.debug("Downloading PDF from S3", s3_key=s3_key)
        pdf_data = await storage.download_file(s3_key)

        # Step 2: Extract text
        logger.debug("Extracting text from PDF", doc_id=doc_id)
        extracted = await pdf_service.extract_text(pdf_data, filename)

        # Step 3: Chunk the document
        logger.debug("Chunking document", doc_id=doc_id, pages=extracted.page_count)
        chunks = await pdf_service.chunk_document(extracted)

        if not chunks:
            raise ValueError("No text could be extracted from the document")

        # Step 4: Generate embeddings
        logger.debug("Generating embeddings", doc_id=doc_id, chunks=len(chunks))
        chunk_texts = [c.content for c in chunks]
        embeddings = await embedding_service.embed_documents(chunk_texts)

        # Step 5: Get document title for metadata
        doc_item = await db.get_item(
            pk=f"USER#{user_id}",
            sk=f"DOC#{doc_id}",
        )
        document_title = doc_item.get("title", filename) if doc_item else filename

        # Step 6: Prepare data for storage
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
                    "content": chunk.content[:1000],  # Truncate for metadata limit
                    "document_title": document_title,
                },
            })

            # Update chunk with embedding ID
            chunk_obj.embedding_id = chunk_obj.chunk_id

        # Step 7: Batch write chunks to DynamoDB
        logger.debug("Storing chunks in DynamoDB", count=len(chunk_items))
        await db.batch_write(chunk_items)

        # Step 8: Upsert vectors to Pinecone
        logger.debug("Upserting vectors to Pinecone", count=len(vector_items))
        await pinecone.upsert_vectors(vector_items)

        # Step 9: Update document status
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

        # Update document status to failed
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
    ctx: dict[str, Any],
    doc_id: str,
    user_id: str,
    s3_key: str,
) -> dict[str, Any]:
    """Delete all data associated with a document.

    This task:
    1. Deletes the PDF from S3
    2. Deletes all chunks from DynamoDB
    3. Deletes all vectors from Pinecone

    Args:
        ctx: ARQ context.
        doc_id: Document ID.
        user_id: Owner user ID.
        s3_key: S3 object key.

    Returns:
        Deletion result.
    """
    setup_logging()
    logger.info("Starting document deletion", doc_id=doc_id)

    db = get_dynamodb_client()
    storage = get_storage_service()
    pinecone = get_pinecone_service()

    try:
        # Step 1: Delete from S3
        await storage.delete_file(s3_key)

        # Step 2: Get and delete all chunks from DynamoDB
        chunks, _ = await db.query(
            pk=f"DOC#{doc_id}",
            sk_begins_with="CHUNK#",
        )

        if chunks:
            chunk_keys = [
                (c["PK"], c["SK"])
                for c in chunks
            ]
            await db.batch_delete(chunk_keys)

            # Step 3: Delete vectors from Pinecone
            chunk_ids = [c["chunk_id"] for c in chunks]
            await pinecone.delete_vectors(ids=chunk_ids)

        # Step 4: Delete document record
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
    ctx: dict[str, Any],
    doc_id: str,
    user_id: str,
    s3_key: str,
    filename: str,
) -> dict[str, Any]:
    """Reprocess an existing document.

    Deletes existing chunks and vectors, then reprocesses.

    Args:
        ctx: ARQ context.
        doc_id: Document ID.
        user_id: Owner user ID.
        s3_key: S3 object key.
        filename: Original filename.

    Returns:
        Processing result.
    """
    setup_logging()
    logger.info("Starting document reprocessing", doc_id=doc_id)

    db = get_dynamodb_client()
    pinecone = get_pinecone_service()

    try:
        # Delete existing chunks
        chunks, _ = await db.query(
            pk=f"DOC#{doc_id}",
            sk_begins_with="CHUNK#",
        )

        if chunks:
            chunk_keys = [(c["PK"], c["SK"]) for c in chunks]
            await db.batch_delete(chunk_keys)

            chunk_ids = [c["chunk_id"] for c in chunks]
            await pinecone.delete_vectors(ids=chunk_ids)

        # Reprocess
        return await process_document(ctx, doc_id, user_id, s3_key, filename)

    except Exception as e:
        logger.error("Document reprocessing failed", doc_id=doc_id, error=str(e))
        return {
            "doc_id": doc_id,
            "status": "failed",
            "error": str(e),
        }


async def startup(ctx: dict[str, Any]) -> None:
    """Worker startup hook."""
    setup_logging()
    logger.info("ARQ worker starting up")


async def shutdown(ctx: dict[str, Any]) -> None:
    """Worker shutdown hook."""
    logger.info("ARQ worker shutting down")


# Worker settings for ARQ
class WorkerSettings:
    """ARQ worker settings."""

    functions = [
        process_document,
        delete_document_data,
        reprocess_document,
    ]

    on_startup = startup
    on_shutdown = shutdown

    redis_settings = RedisSettings.from_dsn(settings.redis_url)

    max_jobs = 10
    job_timeout = 600  # 10 minutes
    keep_result = 3600  # 1 hour
    retry_jobs = True
    max_tries = 3


# Helper to enqueue tasks
async def get_redis_pool() -> ArqRedis:
    """Get ARQ Redis connection pool."""
    return await create_pool(
        RedisSettings.from_dsn(settings.redis_url)
    )


async def enqueue_document_processing(
    doc_id: str,
    user_id: str,
    s3_key: str,
    filename: str,
) -> str:
    """Enqueue a document processing task.

    Args:
        doc_id: Document ID.
        user_id: Owner user ID.
        s3_key: S3 object key.
        filename: Original filename.

    Returns:
        Job ID.
    """
    redis = await get_redis_pool()
    job = await redis.enqueue_job(
        "process_document",
        doc_id,
        user_id,
        s3_key,
        filename,
    )
    await redis.close()
    return job.job_id


async def enqueue_document_deletion(
    doc_id: str,
    user_id: str,
    s3_key: str,
) -> str:
    """Enqueue a document deletion task.

    Args:
        doc_id: Document ID.
        user_id: Owner user ID.
        s3_key: S3 object key.

    Returns:
        Job ID.
    """
    redis = await get_redis_pool()
    job = await redis.enqueue_job(
        "delete_document_data",
        doc_id,
        user_id,
        s3_key,
    )
    await redis.close()
    return job.job_id
