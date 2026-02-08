"""Tests for configuration module."""

import pytest

from app.config import Settings, get_settings


def test_settings_defaults():
    """Test default settings values."""
    settings = Settings()

    assert settings.app_name == "PetMed API"
    assert settings.environment == "development"
    assert settings.api_v1_prefix == "/v1"


def test_settings_computed_fields():
    """Test computed settings fields."""
    settings = Settings(
        aws_region="us-east-1",
        cognito_user_pool_id="us-east-1_test123",
    )

    assert "cognito-idp.us-east-1.amazonaws.com" in settings.cognito_issuer
    assert settings.cognito_user_pool_id in settings.cognito_issuer
    assert ".well-known/jwks.json" in settings.cognito_jwks_url


def test_settings_is_production():
    """Test is_production computed field."""
    dev_settings = Settings(environment="development")
    assert not dev_settings.is_production

    prod_settings = Settings(environment="production")
    assert prod_settings.is_production


def test_settings_pdf_max_size():
    """Test PDF max size calculation."""
    settings = Settings(pdf_max_size_mb=10)
    assert settings.pdf_max_size_bytes == 10 * 1024 * 1024


def test_get_settings_cached():
    """Test that get_settings returns cached instance."""
    settings1 = get_settings()
    settings2 = get_settings()

    # Should be the same cached instance
    assert settings1 is settings2
