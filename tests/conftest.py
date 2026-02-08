"""Pytest fixtures and configuration."""

import asyncio
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

# Set test environment variables before importing app
os.environ.update({
    "ENVIRONMENT": "development",
    "DEBUG": "true",
    "AWS_REGION": "us-east-1",
    "COGNITO_USER_POOL_ID": "us-east-1_test",
    "COGNITO_CLIENT_ID": "test-client-id",
    "DYNAMODB_TABLE_NAME": "petmed-test",
    "S3_BUCKET_NAME": "petmed-test-bucket",
    "REDIS_URL": "redis://localhost:6379",
    "GOOGLE_CLOUD_PROJECT": "test-project",
    "PINECONE_API_KEY": "test-api-key",
    "PINECONE_INDEX_NAME": "test-index",
})


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def app():
    """Create test FastAPI application."""
    from app.main import create_app
    return create_app()


@pytest.fixture
def client(app) -> Generator[TestClient, None, None]:
    """Create synchronous test client."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.fixture
def mock_cognito_user() -> dict[str, Any]:
    """Mock Cognito user data."""
    return {
        "user_id": "test-user-123",
        "email": "test@example.com",
        "email_verified": True,
        "token_use": "id",
        "auth_time": 1700000000,
        "exp": 1700003600,
        "iat": 1700000000,
        "raw_claims": {},
    }


@pytest.fixture
def auth_headers(mock_cognito_user) -> dict[str, str]:
    """Create mock authentication headers."""
    return {"Authorization": "Bearer mock-token"}


@pytest.fixture
def mock_dynamodb():
    """Mock DynamoDB client."""
    mock = AsyncMock()
    mock.get_item = AsyncMock(return_value=None)
    mock.put_item = AsyncMock(return_value={})
    mock.update_item = AsyncMock(return_value={})
    mock.delete_item = AsyncMock(return_value=True)
    mock.query = AsyncMock(return_value=([], None))
    mock.batch_write = AsyncMock()
    mock.batch_delete = AsyncMock()
    return mock


@pytest.fixture
def mock_storage():
    """Mock S3 storage service."""
    mock = AsyncMock()
    mock.upload_file = AsyncMock(return_value="test-key")
    mock.download_file = AsyncMock(return_value=b"test-content")
    mock.generate_presigned_url = AsyncMock(return_value="https://s3.example.com/test")
    mock.delete_file = AsyncMock(return_value=True)
    mock.file_exists = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_gemini():
    """Mock Gemini service."""
    mock = AsyncMock()
    mock.generate = AsyncMock(return_value=(
        "This is a test response.",
        {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    ))

    async def mock_stream(*args, **kwargs):
        for chunk in ["This ", "is ", "a ", "test."]:
            yield chunk

    mock.generate_stream = mock_stream
    mock.analyze_image = AsyncMock(return_value="Image analysis result")
    return mock


@pytest.fixture
def mock_embeddings():
    """Mock embedding service."""
    mock = AsyncMock()
    mock.embed_text = AsyncMock(return_value=[0.1] * 768)
    mock.embed_texts = AsyncMock(return_value=[[0.1] * 768])
    mock.embed_query = AsyncMock(return_value=[0.1] * 768)
    mock.embed_documents = AsyncMock(return_value=[[0.1] * 768])
    mock.dimensions = 768
    return mock


@pytest.fixture
def mock_pinecone():
    """Mock Pinecone service."""
    mock = AsyncMock()
    mock.upsert_vectors = AsyncMock(return_value=1)
    mock.query = AsyncMock(return_value=[
        {
            "id": "chunk-1",
            "score": 0.95,
            "metadata": {
                "doc_id": "doc-1",
                "content": "Test content",
                "page_number": 1,
            },
        }
    ])
    mock.delete_vectors = AsyncMock()
    mock.fetch_vectors = AsyncMock(return_value={})
    mock.get_stats = AsyncMock(return_value={"total_vector_count": 100})
    return mock


@pytest.fixture
def mock_redis():
    """Mock Redis connection."""
    mock = AsyncMock()
    mock.ping = AsyncMock(return_value=True)
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=1)
    mock.close = AsyncMock()
    mock.aclose = AsyncMock()
    return mock


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Sample PDF content for testing."""
    # Minimal valid PDF
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Test PDF) Tj ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000206 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
300
%%EOF"""


@pytest.fixture
def sample_document() -> dict[str, Any]:
    """Sample document data."""
    return {
        "doc_id": "test-doc-123",
        "user_id": "test-user-123",
        "title": "Test Document",
        "filename": "test.pdf",
        "file_size": 1024,
        "mime_type": "application/pdf",
        "s3_key": "documents/test-user-123/test-doc-123/test.pdf",
        "status": "completed",
        "page_count": 10,
        "chunk_count": 25,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }


@pytest.fixture
def sample_conversation() -> dict[str, Any]:
    """Sample conversation data."""
    return {
        "conv_id": "test-conv-123",
        "user_id": "test-user-123",
        "title": "Test Conversation",
        "message_count": 2,
        "last_message_at": "2024-01-01T00:00:00Z",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }


@pytest.fixture
def sample_query() -> dict[str, Any]:
    """Sample query data."""
    return {
        "query_id": "test-query-123",
        "conv_id": "test-conv-123",
        "user_id": "test-user-123",
        "question": "What are the symptoms of canine distemper?",
        "answer": "Canine distemper symptoms include...",
        "status": "completed",
        "sources": [
            {
                "chunk_id": "chunk-1",
                "doc_id": "doc-1",
                "page_number": 5,
                "score": 0.95,
            }
        ],
        "model_used": "gemini-2.5-pro",
        "latency_ms": 1500,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }


@pytest.fixture(autouse=True)
def mock_auth(mock_cognito_user):
    """Auto-mock authentication for all tests."""
    from app.core.cognito import CognitoUser

    async def mock_get_current_user(*args, **kwargs):
        return CognitoUser(**mock_cognito_user)

    with patch("app.core.cognito.get_current_user", mock_get_current_user):
        with patch("app.dependencies.get_current_user", mock_get_current_user):
            yield
