"""Document upload and management endpoints."""

from urllib.parse import quote

from fastapi import APIRouter, File, Form, UploadFile

from app.api.v1.schemas.common import DeleteResponse, ERROR_RESPONSES
from app.api.v1.schemas.documents import (
    ChunkListResponse,
    ChunkResponse,
    DocumentListResponse,
    DocumentProcessingStatus,
    DocumentResponse,
    DocumentUpdateRequest,
    DocumentUploadResponse,
)
from app.config import settings
from app.core.exceptions import BadRequestError, NotFoundError
from app.core.logging import get_logger
from app.db.pagination import decode_cursor, encode_cursor
from app.dependencies import CurrentUser, DynamoDB, Storage
from app.models.entities import Document, DocumentStatus
from app.workers.tasks import enqueue_document_deletion, enqueue_document_processing

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = get_logger(__name__)


async def _get_download_url(doc: dict, storage: Storage) -> str | None:
    """Generate presigned download URL if document is completed."""
    if doc.get("status") != DocumentStatus.COMPLETED.value:
        return None
    return await storage.generate_presigned_url(
        s3_key=doc["s3_key"],
        for_upload=False,
    )


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=201,
    summary="Upload document",
    description="Upload a PDF document for processing. Returns immediately with job ID.",
    responses={
        400: ERROR_RESPONSES[400],
        401: ERROR_RESPONSES[401],
    },
)
async def upload_document(
    current_user: CurrentUser,
    db: DynamoDB,
    storage: Storage,
    file: UploadFile = File(..., description="PDF file to upload"),
    title: str = Form(None, description="Document title (defaults to filename)"),
) -> DocumentUploadResponse:
    """Upload a PDF document for processing."""
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise BadRequestError("Only PDF files are supported")

    if file.content_type and file.content_type != "application/pdf":
        raise BadRequestError("Only PDF files are supported")

    # Read file content
    content = await file.read()
    file_size = len(content)

    # Validate size
    if file_size > settings.pdf_max_size_bytes:
        raise BadRequestError(
            f"File exceeds maximum size of {settings.pdf_max_size_mb}MB"
        )

    if file_size == 0:
        raise BadRequestError("File is empty")

    # Create document record
    doc = Document(
        user_id=current_user.user_id,
        title=title or file.filename or "Untitled Document",
        filename=file.filename or "document.pdf",
        file_size=file_size,
        s3_key="",  # Will be updated after upload
    )

    # Generate S3 key (use doc_id for safe ASCII key)
    s3_key = f"documents/{current_user.user_id}/{doc.doc_id}/{doc.doc_id}.pdf"
    doc.s3_key = s3_key

    # Upload to S3
    await storage.upload_file(
        file_data=content,
        s3_key=s3_key,
        content_type="application/pdf",
        metadata={
            "doc_id": doc.doc_id,
            "user_id": current_user.user_id,
            "original_filename": quote(doc.filename, safe=""),
        },
    )

    # Atomically write document + increment counters in one transaction
    user_pk = f"USER#{current_user.user_id}"
    await db.transact_put_and_increment(
        item=doc.to_dynamodb_item(),
        counter_pk=user_pk,
        counter_sk=user_pk,
        counters={"document_count": 1, "storage_used_bytes": file_size},
    )

    # Enqueue processing task
    job_id = await enqueue_document_processing(
        doc_id=doc.doc_id,
        user_id=current_user.user_id,
        s3_key=s3_key,
        filename=doc.filename,
    )

    logger.info(
        "Document upload started",
        doc_id=doc.doc_id,
        filename=doc.filename,
        size=file_size,
    )

    return DocumentUploadResponse(
        doc_id=doc.doc_id,
        job_id=job_id,
        status="pending",
        message="Document upload started. Processing will begin shortly.",
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List documents",
    description="List all documents for the current user.",
    responses={
        401: ERROR_RESPONSES[401],
    },
)
async def list_documents(
    current_user: CurrentUser,
    db: DynamoDB,
    storage: Storage,
    limit: int = 20,
    cursor: str | None = None,
) -> DocumentListResponse:
    """List user's documents."""
    exclusive_start_key = decode_cursor(cursor)

    # Query documents
    docs, last_key = await db.query(
        pk=f"USER#{current_user.user_id}",
        sk_begins_with="DOC#",
        limit=limit,
        exclusive_start_key=exclusive_start_key,
        scan_forward=False,  # Most recent first
    )

    has_more = last_key is not None
    next_cursor = encode_cursor(last_key)

    # Generate download URLs
    items = []
    for doc in docs:
        download_url = await _get_download_url(doc, storage)

        items.append(
            DocumentResponse(
                doc_id=doc["doc_id"],
                title=doc["title"],
                filename=doc["filename"],
                file_size=doc["file_size"],
                mime_type=doc.get("mime_type", "application/pdf"),
                status=DocumentStatus(doc["status"]),
                page_count=doc.get("page_count"),
                chunk_count=doc.get("chunk_count"),
                error_message=doc.get("error_message"),
                download_url=download_url,
                metadata=doc.get("metadata", {}),
                created_at=doc["created_at"],
                updated_at=doc["updated_at"],
            )
        )

    return DocumentListResponse(
        items=items,
        next_cursor=next_cursor,
        has_more=has_more,
    )


@router.get(
    "/{doc_id}",
    response_model=DocumentResponse,
    summary="Get document",
    description="Get details of a specific document.",
    responses={
        401: ERROR_RESPONSES[401],
        404: ERROR_RESPONSES[404],
    },
)
async def get_document(
    doc_id: str,
    current_user: CurrentUser,
    db: DynamoDB,
    storage: Storage,
) -> DocumentResponse:
    """Get document details."""
    doc = await db.get_item(
        pk=f"USER#{current_user.user_id}",
        sk=f"DOC#{doc_id}",
    )

    if not doc:
        raise NotFoundError(
            f"Document not found: {doc_id}",
            resource_type="document",
            resource_id=doc_id,
        )

    download_url = await _get_download_url(doc, storage)

    return DocumentResponse(
        doc_id=doc["doc_id"],
        title=doc["title"],
        filename=doc["filename"],
        file_size=doc["file_size"],
        mime_type=doc.get("mime_type", "application/pdf"),
        status=DocumentStatus(doc["status"]),
        page_count=doc.get("page_count"),
        chunk_count=doc.get("chunk_count"),
        error_message=doc.get("error_message"),
        download_url=download_url,
        metadata=doc.get("metadata", {}),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
    )


@router.get(
    "/{doc_id}/processing-status",
    response_model=DocumentProcessingStatus,
    summary="Get processing status",
    description="Get the processing status of a document. Use for polling.",
    responses={
        401: ERROR_RESPONSES[401],
        404: ERROR_RESPONSES[404],
    },
)
async def get_processing_status(
    doc_id: str,
    current_user: CurrentUser,
    db: DynamoDB,
) -> DocumentProcessingStatus:
    """Get document processing status."""
    doc = await db.get_item(
        pk=f"USER#{current_user.user_id}",
        sk=f"DOC#{doc_id}",
    )

    if not doc:
        raise NotFoundError(
            f"Document not found: {doc_id}",
            resource_type="document",
            resource_id=doc_id,
        )

    # Calculate progress based on status
    status = DocumentStatus(doc["status"])
    progress = {
        DocumentStatus.PENDING: 0.0,
        DocumentStatus.PROCESSING: 50.0,
        DocumentStatus.COMPLETED: 100.0,
        DocumentStatus.FAILED: 0.0,
    }.get(status, 0.0)

    return DocumentProcessingStatus(
        doc_id=doc_id,
        status=status,
        progress=progress,
        page_count=doc.get("page_count"),
        chunk_count=doc.get("chunk_count"),
        error_message=doc.get("error_message"),
    )


@router.get(
    "/{doc_id}/chunks",
    response_model=ChunkListResponse,
    summary="Get document chunks",
    description="Get the text chunks extracted from a document.",
    responses={
        401: ERROR_RESPONSES[401],
        404: ERROR_RESPONSES[404],
    },
)
async def get_document_chunks(
    doc_id: str,
    current_user: CurrentUser,
    db: DynamoDB,
    limit: int = 20,
    cursor: str | None = None,
) -> ChunkListResponse:
    """Get document chunks."""
    # Verify document exists and belongs to user
    doc = await db.get_item(
        pk=f"USER#{current_user.user_id}",
        sk=f"DOC#{doc_id}",
    )

    if not doc:
        raise NotFoundError(
            f"Document not found: {doc_id}",
            resource_type="document",
            resource_id=doc_id,
        )

    exclusive_start_key = decode_cursor(cursor)

    # Query chunks
    chunks, last_key = await db.query(
        pk=f"DOC#{doc_id}",
        sk_begins_with="CHUNK#",
        limit=limit,
        exclusive_start_key=exclusive_start_key,
    )

    has_more = last_key is not None
    next_cursor = encode_cursor(last_key)

    items = [
        ChunkResponse(
            chunk_id=c["chunk_id"],
            content=c["content"],
            page_number=c.get("page_number"),
            chunk_index=c["chunk_index"],
            token_count=c.get("token_count"),
        )
        for c in chunks
    ]

    return ChunkListResponse(
        items=items,
        next_cursor=next_cursor,
        has_more=has_more,
    )


@router.patch(
    "/{doc_id}",
    response_model=DocumentResponse,
    summary="Update document",
    description="Update document metadata (title).",
    responses={
        401: ERROR_RESPONSES[401],
        404: ERROR_RESPONSES[404],
    },
)
async def update_document(
    doc_id: str,
    request: DocumentUpdateRequest,
    current_user: CurrentUser,
    db: DynamoDB,
    storage: Storage,
) -> DocumentResponse:
    """Update document metadata."""
    # Verify document exists
    doc = await db.get_item(
        pk=f"USER#{current_user.user_id}",
        sk=f"DOC#{doc_id}",
    )

    if not doc:
        raise NotFoundError(
            f"Document not found: {doc_id}",
            resource_type="document",
            resource_id=doc_id,
        )

    updates = {}
    if request.title:
        updates["title"] = request.title

    if updates:
        doc = await db.update_item(
            pk=f"USER#{current_user.user_id}",
            sk=f"DOC#{doc_id}",
            updates=updates,
        )

    download_url = await _get_download_url(doc, storage)

    return DocumentResponse(
        doc_id=doc["doc_id"],
        title=doc["title"],
        filename=doc["filename"],
        file_size=doc["file_size"],
        mime_type=doc.get("mime_type", "application/pdf"),
        status=DocumentStatus(doc["status"]),
        page_count=doc.get("page_count"),
        chunk_count=doc.get("chunk_count"),
        error_message=doc.get("error_message"),
        download_url=download_url,
        metadata=doc.get("metadata", {}),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
    )


@router.delete(
    "/{doc_id}",
    response_model=DeleteResponse,
    summary="Delete document",
    description="Delete a document and all associated data.",
    responses={
        401: ERROR_RESPONSES[401],
        404: ERROR_RESPONSES[404],
    },
)
async def delete_document(
    doc_id: str,
    current_user: CurrentUser,
    db: DynamoDB,
) -> DeleteResponse:
    """Delete a document."""
    # Get document
    doc = await db.get_item(
        pk=f"USER#{current_user.user_id}",
        sk=f"DOC#{doc_id}",
    )

    if not doc:
        raise NotFoundError(
            f"Document not found: {doc_id}",
            resource_type="document",
            resource_id=doc_id,
        )

    # Enqueue deletion task
    await enqueue_document_deletion(
        doc_id=doc_id,
        user_id=current_user.user_id,
        s3_key=doc["s3_key"],
    )

    # Atomically decrement user's denormalized document count and storage
    user_pk = f"USER#{current_user.user_id}"
    file_size = doc.get("file_size", 0)
    await db.increment_counters(
        pk=user_pk,
        sk=user_pk,
        counters={"document_count": -1, "storage_used_bytes": -file_size},
    )

    logger.info("Document deletion started", doc_id=doc_id)

    return DeleteResponse(
        success=True,
        deleted_id=doc_id,
    )
