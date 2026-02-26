"""Admin-specific schemas."""

from datetime import datetime
from typing import Any

from pydantic import Field

from app.api.v1.schemas.common import BaseSchema, PaginatedResponse, TimestampMixin
from app.models.entities import DocumentStatus, MessageRole


# --- User schemas ---


class AdminUserResponse(BaseSchema, TimestampMixin):
    """Admin view of a user."""

    user_id: str = Field(description="User ID")
    email: str = Field(description="Email address")
    name: str | None = Field(default=None, description="Display name")
    avatar_url: str | None = Field(default=None, description="Avatar URL")
    is_active: bool = Field(default=True, description="Whether user is active")
    role: str = Field(default="user", description="User role")
    message_limit: int = Field(default=50, description="Daily message limit")
    document_count: int = Field(default=0, description="Number of documents")
    conversation_count: int = Field(default=0, description="Number of conversations")


class AdminUserListResponse(PaginatedResponse[AdminUserResponse]):
    """Paginated list of users for admin."""

    pass


class AdminUpdateUserRequest(BaseSchema):
    """Admin request to update a user."""

    is_active: bool | None = Field(default=None, description="Active status")
    message_limit: int | None = Field(
        default=None, ge=0, le=10000, description="Daily message limit"
    )
    role: str | None = Field(default=None, description="User role (user or admin)")


# --- Document schemas ---


class AdminDocumentResponse(BaseSchema, TimestampMixin):
    """Admin view of a document (includes owner info)."""

    doc_id: str = Field(description="Document ID")
    user_id: str = Field(description="Owner user ID")
    user_email: str | None = Field(default=None, description="Owner email")
    title: str = Field(description="Document title")
    filename: str = Field(description="Original filename")
    file_size: int = Field(description="File size in bytes")
    mime_type: str = Field(description="MIME type")
    status: DocumentStatus = Field(description="Processing status")
    page_count: int | None = Field(default=None, description="Number of pages")
    chunk_count: int | None = Field(default=None, description="Number of chunks")
    error_message: str | None = Field(default=None, description="Error if failed")
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdminDocumentListResponse(PaginatedResponse[AdminDocumentResponse]):
    """Paginated list of documents for admin."""

    pass


# --- Conversation schemas ---


class AdminConversationResponse(BaseSchema, TimestampMixin):
    """Admin view of a conversation."""

    conv_id: str = Field(description="Conversation ID")
    user_id: str = Field(description="Owner user ID")
    title: str = Field(description="Conversation title")
    message_count: int = Field(default=0, description="Number of messages")
    last_message_at: datetime | None = Field(default=None)


class AdminConversationListResponse(PaginatedResponse[AdminConversationResponse]):
    """Paginated list of conversations for admin."""

    pass


class AdminMessageResponse(BaseSchema, TimestampMixin):
    """Admin view of a message."""

    message_id: str = Field(description="Message ID")
    role: MessageRole = Field(description="Message role")
    content: str = Field(description="Message content")
    query_id: str | None = Field(default=None)


class AdminConversationDetailResponse(AdminConversationResponse):
    """Conversation with messages for admin."""

    messages: list[AdminMessageResponse] = Field(default_factory=list)


# --- Stats schemas ---


class SystemStatsResponse(BaseSchema):
    """System-wide statistics."""

    total_users: int = Field(description="Total registered users")
    total_documents: int = Field(description="Total documents")
    total_queries: int = Field(description="Total RAG queries")
    avg_rating: float | None = Field(default=None, description="Average feedback rating")
    documents_by_status: dict[str, int] = Field(
        default_factory=dict, description="Document count by status"
    )
