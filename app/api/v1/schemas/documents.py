"""Document-related schemas."""

from datetime import datetime
from typing import Any

from pydantic import Field

from app.api.v1.schemas.common import BaseSchema, PaginatedResponse, TimestampMixin
from app.models.entities import DocumentStatus


class DocumentUploadResponse(BaseSchema):
    """Response after document upload."""

    doc_id: str = Field(description="Document ID")
    job_id: str = Field(description="Processing job ID")
    status: str = Field(default="pending", description="Initial status")
    message: str = Field(default="Document upload started")


class DocumentResponse(BaseSchema, TimestampMixin):
    """Document details response."""

    doc_id: str = Field(description="Document ID")
    title: str = Field(description="Document title")
    filename: str = Field(description="Original filename")
    file_size: int = Field(description="File size in bytes")
    mime_type: str = Field(description="MIME type")
    status: DocumentStatus = Field(description="Processing status")
    page_count: int | None = Field(default=None, description="Number of pages")
    chunk_count: int | None = Field(default=None, description="Number of chunks")
    error_message: str | None = Field(default=None, description="Error if failed")
    download_url: str | None = Field(default=None, description="Presigned download URL")
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(PaginatedResponse[DocumentResponse]):
    """Paginated list of documents."""

    pass


class DocumentProcessingStatus(BaseSchema):
    """Document processing status response."""

    doc_id: str = Field(description="Document ID")
    status: DocumentStatus = Field(description="Current status")
    progress: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Progress percentage"
    )
    page_count: int | None = Field(default=None, description="Pages if known")
    chunk_count: int | None = Field(default=None, description="Chunks if completed")
    error_message: str | None = Field(default=None, description="Error if failed")


class ChunkResponse(BaseSchema):
    """Document chunk response."""

    chunk_id: str = Field(description="Chunk ID")
    content: str = Field(description="Chunk text content")
    page_number: int | None = Field(default=None, description="Source page")
    chunk_index: int = Field(description="Chunk order index")
    token_count: int | None = Field(default=None)


class ChunkListResponse(PaginatedResponse[ChunkResponse]):
    """Paginated list of chunks."""

    pass


class DocumentUpdateRequest(BaseSchema):
    """Request to update document metadata."""

    title: str | None = Field(default=None, min_length=1, max_length=255)
