"""Auth-related schemas."""

from datetime import datetime

from pydantic import Field

from app.api.v1.schemas.common import BaseSchema


class UserResponse(BaseSchema):
    """Current user response."""

    user_id: str = Field(description="User ID from Cognito")
    email: str = Field(description="User email address")
    email_verified: bool = Field(description="Whether email is verified")
    name: str | None = Field(default=None, description="Display name")
    avatar_url: str | None = Field(default=None, description="Profile avatar URL")
    created_at: datetime | None = Field(default=None, description="Account creation time")


class TokenRefreshRequest(BaseSchema):
    """Token refresh request."""

    refresh_token: str = Field(description="Cognito refresh token")


class TokenResponse(BaseSchema):
    """Token response."""

    access_token: str = Field(description="New access token")
    id_token: str = Field(description="New ID token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: int = Field(description="Token expiry in seconds")
