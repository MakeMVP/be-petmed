"""Gemini embedding service for vector generation."""

import asyncio
from typing import Any

from google import genai
from google.genai import types

from app.config import settings
from app.core.exceptions import ServiceUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using Gemini."""

    def __init__(self) -> None:
        self._client = genai.Client(
            vertexai=True,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )
        self._model = settings.embedding_model
        self._dimensions = settings.embedding_dimensions

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.

        Raises:
            ServiceUnavailableError: If embedding fails.
        """
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            ServiceUnavailableError: If embedding fails.
        """
        if not texts:
            return []

        try:
            # Batch texts in groups of 100 (API limit)
            batch_size = 100
            all_embeddings: list[list[float]] = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                response = await asyncio.to_thread(
                    self._client.models.embed_content,
                    model=self._model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self._dimensions,
                    ),
                )

                # Extract embeddings from response
                for embedding in response.embeddings:
                    all_embeddings.append(list(embedding.values))

            logger.debug(
                "Generated embeddings",
                count=len(texts),
                dimensions=self._dimensions,
            )

            return all_embeddings

        except Exception as e:
            logger.error("Embedding generation failed", error=str(e))
            raise ServiceUnavailableError(
                f"Failed to generate embeddings: {e}",
                service="gemini-embedding",
            ) from e

    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query.

        This uses the same embedding model but is semantically
        distinct to allow for future query-specific optimizations.

        Args:
            query: Search query to embed.

        Returns:
            Embedding vector as list of floats.
        """
        return await self.embed_text(query)

    async def embed_document(self, document: str) -> list[float]:
        """Generate embedding for a document chunk.

        Args:
            document: Document text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        return await self.embed_text(document)

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple document chunks.

        Args:
            documents: List of document texts to embed.

        Returns:
            List of embedding vectors.
        """
        return await self.embed_texts(documents)

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        return self._dimensions


# Singleton instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
