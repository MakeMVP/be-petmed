"""User-related schemas."""

from datetime import datetime
from typing import Any

from pydantic import EmailStr, Field

from app.api.v1.schemas.common import BaseSchema, TimestampMixin


class UserProfileResponse(BaseSchema, TimestampMixin):
    """User profile response."""

    user_id: str = Field(description="User ID")
    email: str = Field(description="Email address")
    name: str | None = Field(default=None, description="Display name")
    avatar_url: str | None = Field(default=None, description="Profile avatar URL")
    settings: dict[str, Any] = Field(default_factory=dict, description="User settings")


class UpdateProfileRequest(BaseSchema):
    """Request to update user profile."""

    name: str | None = Field(
        default=None, min_length=1, max_length=100, description="Display name"
    )
    avatar_url: str | None = Field(
        default=None, max_length=500, description="Avatar URL"
    )
    settings: dict[str, Any] | None = Field(
        default=None, description="User settings to update"
    )


class UserStatsResponse(BaseSchema):
    """User statistics response."""

    document_count: int = Field(description="Number of documents")
    conversation_count: int = Field(description="Number of conversations")
    query_count: int = Field(description="Total queries made")
    storage_used_bytes: int = Field(description="Storage used in bytes")
