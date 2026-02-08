"""Tests for exception handling."""

import pytest

from app.core.exceptions import (
    AppException,
    BadRequestError,
    ForbiddenError,
    NotFoundError,
    UnauthorizedError,
    ValidationError,
)


def test_app_exception_basic():
    """Test basic AppException creation."""
    exc = AppException(
        status_code=400,
        title="Test Error",
        detail="This is a test error",
    )

    assert exc.status_code == 400
    assert exc.title == "Test Error"
    assert exc.detail == "This is a test error"
    assert exc.type_uri == "about:blank"


def test_app_exception_to_problem_detail():
    """Test conversion to Problem Detail."""
    exc = AppException(
        status_code=400,
        title="Test Error",
        detail="This is a test error",
        type_uri="/problems/test",
    )

    problem = exc.to_problem_detail(instance="/test/endpoint")

    assert problem.type == "/problems/test"
    assert problem.title == "Test Error"
    assert problem.status == 400
    assert problem.detail == "This is a test error"
    assert problem.instance == "/test/endpoint"


def test_bad_request_error():
    """Test BadRequestError."""
    exc = BadRequestError("Invalid input")

    assert exc.status_code == 400
    assert exc.title == "Bad Request"
    assert exc.detail == "Invalid input"


def test_unauthorized_error():
    """Test UnauthorizedError."""
    exc = UnauthorizedError()

    assert exc.status_code == 401
    assert exc.title == "Unauthorized"


def test_forbidden_error():
    """Test ForbiddenError."""
    exc = ForbiddenError()

    assert exc.status_code == 403
    assert exc.title == "Forbidden"


def test_not_found_error():
    """Test NotFoundError with resource info."""
    exc = NotFoundError(
        detail="Document not found",
        resource_type="document",
        resource_id="doc-123",
    )

    assert exc.status_code == 404
    assert exc.title == "Not Found"
    assert exc.extra.get("resource_type") == "document"
    assert exc.extra.get("resource_id") == "doc-123"


def test_validation_error_with_errors():
    """Test ValidationError with field errors."""
    exc = ValidationError(
        detail="Validation failed",
        errors=[
            {"field": "email", "message": "Invalid email format"},
            {"field": "name", "message": "Name is required"},
        ],
    )

    assert exc.status_code == 422
    assert exc.errors is not None
    assert len(exc.errors) == 2
