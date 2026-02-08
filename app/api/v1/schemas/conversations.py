"""Conversation-related schemas."""

from datetime import datetime
from typing import Any

from pydantic import Field

from app.api.v1.schemas.common import BaseSchema, PaginatedResponse, TimestampMixin
from app.models.entities import MessageRole


class CreateConversationRequest(BaseSchema):
    """Request to create a new conversation."""

    title: str = Field(
        default="New Conversation",
        min_length=1,
        max_length=255,
        description="Conversation title",
    )


class ConversationResponse(BaseSchema, TimestampMixin):
    """Conversation details response."""

    conv_id: str = Field(description="Conversation ID")
    title: str = Field(description="Conversation title")
    message_count: int = Field(default=0, description="Number of messages")
    last_message_at: datetime | None = Field(
        default=None, description="Last message timestamp"
    )


class ConversationListResponse(PaginatedResponse[ConversationResponse]):
    """Paginated list of conversations."""

    pass


class UpdateConversationRequest(BaseSchema):
    """Request to update conversation."""

    title: str = Field(min_length=1, max_length=255, description="New title")


class MessageResponse(BaseSchema, TimestampMixin):
    """Chat message response."""

    message_id: str = Field(description="Message ID")
    role: MessageRole = Field(description="Message role")
    content: str = Field(description="Message content")
    query_id: str | None = Field(
        default=None, description="Associated query ID if RAG query"
    )


class MessageListResponse(PaginatedResponse[MessageResponse]):
    """Paginated list of messages."""

    pass


class ConversationDetailResponse(ConversationResponse):
    """Conversation with recent messages."""

    messages: list[MessageResponse] = Field(
        default_factory=list, description="Recent messages"
    )
