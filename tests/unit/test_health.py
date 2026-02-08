"""Tests for health check endpoints."""

import pytest
from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """Test basic health check endpoint."""
    response = client.get("/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "environment" in data
    assert "timestamp" in data


def test_health_check_returns_correct_structure(client: TestClient):
    """Test health check response structure."""
    response = client.get("/v1/health")

    assert response.status_code == 200
    data = response.json()

    # Verify all required fields
    required_fields = ["status", "version", "environment", "timestamp"]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"
