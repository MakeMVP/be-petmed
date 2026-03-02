"""Dependency injection container and service initialization."""

from typing import Annotated

from fastapi import Depends

from app.core.cognito import CurrentUser, OptionalUser, VerifiedUser
from app.core.logging import get_logger
from app.db.dynamodb import DynamoDBClient, get_dynamodb_client
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.gemini_service import GeminiService, get_gemini_service
from app.services.pinecone_service import PineconeService, get_pinecone_service
from app.services.storage_service import StorageService, get_storage_service

# Re-export auth dependencies
__all__ = [
    "CurrentUser",
    "OptionalUser",
    "VerifiedUser",
    "DynamoDB",
    "Storage",
    "Gemini",
    "Embeddings",
    "VectorDB",
    "get_db",
    "get_storage",
    "get_gemini",
    "get_embeddings",
    "get_vector_db",
]


# Database dependency
def get_db() -> DynamoDBClient:
    """Get DynamoDB client dependency."""
    return get_dynamodb_client()


DynamoDB = Annotated[DynamoDBClient, Depends(get_db)]


# Storage dependency
def get_storage() -> StorageService:
    """Get storage service dependency."""
    return get_storage_service()


Storage = Annotated[StorageService, Depends(get_storage)]


# Gemini LLM dependency
def get_gemini() -> GeminiService:
    """Get Gemini service dependency."""
    return get_gemini_service()


Gemini = Annotated[GeminiService, Depends(get_gemini)]


# Embedding dependency
def get_embeddings() -> EmbeddingService:
    """Get embedding service dependency."""
    return get_embedding_service()


Embeddings = Annotated[EmbeddingService, Depends(get_embeddings)]


# Vector DB dependency
def get_vector_db() -> PineconeService:
    """Get Pinecone service dependency."""
    return get_pinecone_service()


VectorDB = Annotated[PineconeService, Depends(get_vector_db)]


logger = get_logger(__name__)


# Service container for initialization
class ServiceContainer:
    """Container for managing service lifecycle."""

    _initialized: bool = False

    @classmethod
    async def initialize(cls) -> None:
        """Initialize all services on startup."""
        if cls._initialized:
            return

        # Initialize singletons
        get_dynamodb_client()
        get_storage_service()
        try:
            get_gemini_service()
        except Exception as e:
            logger.warning("Gemini service init failed (chat will be unavailable)", error=str(e))
        try:
            get_embedding_service()
        except Exception as e:
            logger.warning("Embedding service init failed", error=str(e))
        get_pinecone_service()

        cls._initialized = True

    @classmethod
    async def shutdown(cls) -> None:
        """Cleanup services on shutdown."""
        # Services use connection pooling managed by their clients
        # No explicit cleanup needed for current implementation
        cls._initialized = False
