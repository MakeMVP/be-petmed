"""Central configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "PetMed API"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # API
    api_v1_prefix: str = "/v1"
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # AWS General
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # AWS Cognito
    cognito_user_pool_id: str = ""
    cognito_client_id: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cognito_issuer(self) -> str:
        """Cognito token issuer URL."""
        return f"https://cognito-idp.{self.aws_region}.amazonaws.com/{self.cognito_user_pool_id}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cognito_jwks_url(self) -> str:
        """Cognito JWKS URL for token verification."""
        return f"{self.cognito_issuer}/.well-known/jwks.json"

    # AWS DynamoDB
    dynamodb_table_name: str = "petmed"
    dynamodb_endpoint_url: str | None = None  # For local development

    # AWS S3
    s3_bucket_name: str = "petmed-documents"
    s3_endpoint_url: str | None = None  # For local development
    s3_presigned_url_expiry: int = 3600  # seconds

    # Redis (optional — only needed if rate limiting enabled)
    redis_url: str | None = None

    # Lambda
    lambda_function_name: str = "petmed-document-worker"

    # Google Cloud / Vertex AI
    google_cloud_project: str = ""
    google_cloud_location: str = "us-central1"
    gemini_model: str = "gemini-2.5-pro"
    gemini_flash_model: str = "gemini-2.5-flash"
    embedding_model: str = "text-embedding-004"
    embedding_dimensions: int = 768

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "petmed-knowledge"
    pinecone_namespace: str = "documents"

    # RAG Configuration
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.7

    # PDF Processing
    pdf_max_size_mb: int = 50
    pdf_max_pages: int = 500

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pdf_max_size_bytes(self) -> int:
        """Maximum PDF size in bytes."""
        return self.pdf_max_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
