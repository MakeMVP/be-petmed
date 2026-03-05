"""Common schemas shared across API endpoints."""

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class BaseSchema(BaseModel):
    """Base schema with common configuration."""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""

    created_at: datetime
    updated_at: datetime


class PaginationParams(BaseModel):
    """Pagination query parameters."""

    limit: int = Field(default=20, ge=1, le=100, description="Number of items per page")
    cursor: str | None = Field(default=None, description="Cursor for next page")


class PaginatedResponse(BaseSchema, Generic[T]):
    """Paginated response wrapper."""

    items: list[T]
    next_cursor: str | None = Field(
        default=None, description="Cursor for next page, null if no more items"
    )
    has_more: bool = Field(description="Whether there are more items")


class SuccessResponse(BaseSchema):
    """Generic success response."""

    success: bool = True
    message: str = "Operation completed successfully"


class DeleteResponse(BaseSchema):
    """Response for delete operations."""

    success: bool = True
    deleted_id: str


class JobResponse(BaseSchema):
    """Response for async job submission."""

    job_id: str = Field(description="Job ID for tracking progress")
    status: str = Field(default="pending", description="Initial job status")
    message: str = Field(default="Job submitted successfully")


class ProcessingStatus(BaseSchema):
    """Processing status for async operations."""

    status: str = Field(description="Current processing status")
    progress: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Progress percentage"
    )
    message: str | None = Field(default=None, description="Status message")
    error: str | None = Field(default=None, description="Error message if failed")
    result: dict[str, Any] | None = Field(
        default=None, description="Result data if completed"
    )


class ProblemDetailResponse(BaseSchema):
    """RFC 9457 Problem Details response schema for OpenAPI documentation."""

    type: str = Field(
        default="about:blank", description="URI reference identifying the problem type"
    )
    title: str = Field(description="Short, human-readable summary of the problem")
    status: int = Field(description="HTTP status code")
    detail: str = Field(description="Human-readable explanation of the problem")
    instance: str | None = Field(
        default=None, description="URI reference identifying the specific occurrence"
    )
    errors: list[dict[str, Any]] | None = Field(
        default=None, description="List of validation errors"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "/problems/validation-error",
                "title": "Validation Error",
                "status": 422,
                "detail": "The request data failed validation.",
                "instance": "/v1/documents",
                "errors": [
                    {
                        "field": "title",
                        "message": "Field is required",
                    }
                ],
            }
        }
    )


# Common response models for OpenAPI documentation
ERROR_RESPONSES = {
    400: {"model": ProblemDetailResponse, "description": "Bad Request"},
    401: {"model": ProblemDetailResponse, "description": "Unauthorized"},
    403: {"model": ProblemDetailResponse, "description": "Forbidden"},
    404: {"model": ProblemDetailResponse, "description": "Not Found"},
    409: {"model": ProblemDetailResponse, "description": "Conflict"},
    422: {"model": ProblemDetailResponse, "description": "Validation Error"},
    429: {"model": ProblemDetailResponse, "description": "Too Many Requests"},
    500: {"model": ProblemDetailResponse, "description": "Internal Server Error"},
}
