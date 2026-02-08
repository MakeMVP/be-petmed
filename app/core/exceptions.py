"""RFC 9457 Problem Details exception handling."""

from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ProblemDetail(BaseModel):
    """RFC 9457 Problem Details response model."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str
    instance: str | None = None
    errors: list[dict[str, Any]] | None = None

    model_config = {"extra": "allow"}


class AppException(Exception):
    """Base application exception with RFC 9457 Problem Details support."""

    def __init__(
        self,
        status_code: int,
        title: str,
        detail: str,
        type_uri: str = "about:blank",
        errors: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        self.status_code = status_code
        self.title = title
        self.detail = detail
        self.type_uri = type_uri
        self.errors = errors
        self.extra = kwargs
        super().__init__(detail)

    def to_problem_detail(self, instance: str | None = None) -> ProblemDetail:
        """Convert exception to RFC 9457 Problem Detail."""
        return ProblemDetail(
            type=self.type_uri,
            title=self.title,
            status=self.status_code,
            detail=self.detail,
            instance=instance,
            errors=self.errors,
            **self.extra,
        )


class BadRequestError(AppException):
    """400 Bad Request error."""

    def __init__(
        self,
        detail: str = "The request was malformed or invalid.",
        errors: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            status_code=400,
            title="Bad Request",
            detail=detail,
            type_uri="/problems/bad-request",
            errors=errors,
            **kwargs,
        )


class UnauthorizedError(AppException):
    """401 Unauthorized error."""

    def __init__(
        self,
        detail: str = "Authentication is required to access this resource.",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            status_code=401,
            title="Unauthorized",
            detail=detail,
            type_uri="/problems/unauthorized",
            **kwargs,
        )


class ForbiddenError(AppException):
    """403 Forbidden error."""

    def __init__(
        self,
        detail: str = "You do not have permission to access this resource.",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            status_code=403,
            title="Forbidden",
            detail=detail,
            type_uri="/problems/forbidden",
            **kwargs,
        )


class NotFoundError(AppException):
    """404 Not Found error."""

    def __init__(
        self,
        detail: str = "The requested resource was not found.",
        resource_type: str | None = None,
        resource_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        extra = kwargs
        if resource_type:
            extra["resource_type"] = resource_type
        if resource_id:
            extra["resource_id"] = resource_id
        super().__init__(
            status_code=404,
            title="Not Found",
            detail=detail,
            type_uri="/problems/not-found",
            **extra,
        )


class ConflictError(AppException):
    """409 Conflict error."""

    def __init__(
        self,
        detail: str = "The request conflicts with the current state of the resource.",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            status_code=409,
            title="Conflict",
            detail=detail,
            type_uri="/problems/conflict",
            **kwargs,
        )


class ValidationError(AppException):
    """422 Validation error."""

    def __init__(
        self,
        detail: str = "The request data failed validation.",
        errors: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            status_code=422,
            title="Validation Error",
            detail=detail,
            type_uri="/problems/validation-error",
            errors=errors,
            **kwargs,
        )


class RateLimitError(AppException):
    """429 Too Many Requests error."""

    def __init__(
        self,
        detail: str = "Too many requests. Please try again later.",
        retry_after: int | None = None,
        **kwargs: Any,
    ) -> None:
        extra = kwargs
        if retry_after:
            extra["retry_after"] = retry_after
        super().__init__(
            status_code=429,
            title="Too Many Requests",
            detail=detail,
            type_uri="/problems/rate-limit-exceeded",
            **extra,
        )


class ServiceUnavailableError(AppException):
    """503 Service Unavailable error."""

    def __init__(
        self,
        detail: str = "The service is temporarily unavailable. Please try again later.",
        service: str | None = None,
        **kwargs: Any,
    ) -> None:
        extra = kwargs
        if service:
            extra["service"] = service
        super().__init__(
            status_code=503,
            title="Service Unavailable",
            detail=detail,
            type_uri="/problems/service-unavailable",
            **extra,
        )


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle AppException and return RFC 9457 Problem Details response."""
    problem = exc.to_problem_detail(instance=str(request.url))
    return JSONResponse(
        status_code=exc.status_code,
        content=problem.model_dump(exclude_none=True),
        media_type="application/problem+json",
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTPException and return RFC 9457 Problem Details response."""
    problem = ProblemDetail(
        type="/problems/http-error",
        title=exc.detail if isinstance(exc.detail, str) else "HTTP Error",
        status=exc.status_code,
        detail=exc.detail if isinstance(exc.detail, str) else str(exc.detail),
        instance=str(request.url),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=problem.model_dump(exclude_none=True),
        media_type="application/problem+json",
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unhandled exceptions and return RFC 9457 Problem Details response."""
    from app.core.logging import get_logger

    logger = get_logger(__name__)
    logger.exception("Unhandled exception", exc_info=exc, path=str(request.url))

    problem = ProblemDetail(
        type="/problems/internal-error",
        title="Internal Server Error",
        status=500,
        detail="An unexpected error occurred. Please try again later.",
        instance=str(request.url),
    )
    return JSONResponse(
        status_code=500,
        content=problem.model_dump(exclude_none=True),
        media_type="application/problem+json",
    )
