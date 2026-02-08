"""Query-related schemas."""

from datetime import datetime
from typing import Any

from pydantic import Field

from app.api.v1.schemas.common import BaseSchema, TimestampMixin
from app.models.entities import QueryStatus


class QueryRequest(BaseSchema):
    """RAG query request."""

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="The question to ask",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Conversation ID for context. Creates new if not provided.",
    )
    document_ids: list[str] | None = Field(
        default=None,
        description="Specific documents to query. Queries all if not provided.",
    )


class SourceResponse(BaseSchema):
    """Source chunk used in response."""

    chunk_id: str = Field(description="Chunk ID")
    doc_id: str = Field(description="Document ID")
    document_title: str | None = Field(default=None, description="Document title")
    page_number: int | None = Field(default=None, description="Source page")
    score: float = Field(description="Relevance score")
    content_preview: str | None = Field(
        default=None, description="Content preview (truncated)"
    )


class QueryResponse(BaseSchema, TimestampMixin):
    """RAG query response."""

    query_id: str = Field(description="Query ID")
    conversation_id: str = Field(description="Conversation ID")
    question: str = Field(description="Original question")
    answer: str = Field(description="Generated answer")
    sources: list[SourceResponse] = Field(
        default_factory=list, description="Source chunks used"
    )
    model_used: str | None = Field(default=None, description="LLM model used")
    latency_ms: int | None = Field(default=None, description="Processing time")


class QueryStatusResponse(BaseSchema):
    """Query processing status."""

    query_id: str = Field(description="Query ID")
    status: QueryStatus = Field(description="Current status")
    answer: str | None = Field(default=None, description="Answer if completed")
    error_message: str | None = Field(default=None, description="Error if failed")


class FeedbackRequest(BaseSchema):
    """Query feedback request."""

    rating: int = Field(ge=1, le=5, description="Rating 1-5")
    comment: str | None = Field(
        default=None, max_length=1000, description="Optional feedback comment"
    )


class FeedbackResponse(BaseSchema):
    """Feedback submission response."""

    success: bool = True
    message: str = "Feedback submitted successfully"


class StreamChunk(BaseSchema):
    """SSE stream chunk for streaming responses."""

    type: str = Field(description="Chunk type: text, source, done, error")
    content: str | None = Field(default=None, description="Text content for type=text")
    source: SourceResponse | None = Field(
        default=None, description="Source for type=source"
    )
    query_id: str | None = Field(
        default=None, description="Query ID for type=done"
    )
    error: str | None = Field(default=None, description="Error for type=error")
