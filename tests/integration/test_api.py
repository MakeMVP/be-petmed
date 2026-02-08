"""Integration tests for API endpoints."""

import pytest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


@pytest.fixture
def mock_services(mock_dynamodb, mock_storage, mock_gemini, mock_embeddings, mock_pinecone):
    """Mock all external services."""
    with patch("app.dependencies.get_dynamodb_client", return_value=mock_dynamodb):
        with patch("app.dependencies.get_storage_service", return_value=mock_storage):
            with patch("app.dependencies.get_gemini_service", return_value=mock_gemini):
                with patch("app.dependencies.get_embedding_service", return_value=mock_embeddings):
                    with patch("app.dependencies.get_pinecone_service", return_value=mock_pinecone):
                        yield


def test_root_endpoint(client: TestClient):
    """Test root endpoint."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "PetMed" in data["message"]


def test_health_endpoint(client: TestClient):
    """Test health endpoint."""
    response = client.get("/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_auth_me_creates_user(client: TestClient, mock_services, mock_dynamodb, mock_cognito_user):
    """Test that /auth/me creates user on first login."""
    # User doesn't exist yet
    mock_dynamodb.get_item.return_value = None

    response = client.post(
        "/v1/auth/me",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == mock_cognito_user["user_id"]
    assert data["email"] == mock_cognito_user["email"]

    # Verify user was created
    mock_dynamodb.put_item.assert_called_once()


def test_auth_me_returns_existing_user(client: TestClient, mock_services, mock_dynamodb, mock_cognito_user):
    """Test that /auth/me returns existing user."""
    # User already exists
    mock_dynamodb.get_item.return_value = {
        "user_id": mock_cognito_user["user_id"],
        "email": mock_cognito_user["email"],
        "name": "Existing User",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }

    response = client.post(
        "/v1/auth/me",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Existing User"

    # Verify user was not re-created
    mock_dynamodb.put_item.assert_not_called()


def test_list_documents_empty(client: TestClient, mock_services, mock_dynamodb):
    """Test listing documents when none exist."""
    mock_dynamodb.query.return_value = ([], None)

    response = client.get(
        "/v1/documents",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["items"] == []
    assert data["has_more"] is False


def test_list_conversations_empty(client: TestClient, mock_services, mock_dynamodb):
    """Test listing conversations when none exist."""
    mock_dynamodb.query.return_value = ([], None)

    response = client.get(
        "/v1/conversations",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["items"] == []
    assert data["has_more"] is False


def test_create_conversation(client: TestClient, mock_services, mock_dynamodb):
    """Test creating a new conversation."""
    response = client.post(
        "/v1/conversations",
        headers={"Authorization": "Bearer test-token"},
        json={"title": "Test Conversation"},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "Test Conversation"
    assert "conv_id" in data

    # Verify conversation was created
    mock_dynamodb.put_item.assert_called_once()


def test_get_nonexistent_document(client: TestClient, mock_services, mock_dynamodb):
    """Test getting a document that doesn't exist."""
    mock_dynamodb.get_item.return_value = None

    response = client.get(
        "/v1/documents/nonexistent-id",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 404
    data = response.json()
    assert data["type"] == "/problems/not-found"


def test_get_nonexistent_conversation(client: TestClient, mock_services, mock_dynamodb):
    """Test getting a conversation that doesn't exist."""
    mock_dynamodb.get_item.return_value = None

    response = client.get(
        "/v1/conversations/nonexistent-id",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 404
