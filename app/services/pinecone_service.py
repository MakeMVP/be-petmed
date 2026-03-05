"""Pinecone vector database service."""

import asyncio
import functools
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from app.config import settings
from app.core.exceptions import ServiceUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)


class PineconeService:
    """Service for Pinecone vector database operations."""

    def __init__(self) -> None:
        self._client = Pinecone(api_key=settings.pinecone_api_key)
        self._index_name = settings.pinecone_index_name
        self._namespace = settings.pinecone_namespace
        self._index = None

    def _get_index(self):
        """Get or create the Pinecone index."""
        if self._index is None:
            self._index = self._client.Index(self._index_name)
        return self._index

    async def upsert_vectors(
        self,
        vectors: list[dict[str, Any]],
        namespace: str | None = None,
    ) -> int:
        """Upsert vectors to Pinecone.

        Args:
            vectors: List of vector dicts with 'id', 'values', and optional 'metadata'.
            namespace: Optional namespace override.

        Returns:
            Number of vectors upserted.

        Raises:
            ServiceUnavailableError: If upsert fails.
        """
        if not vectors:
            return 0

        ns = namespace or self._namespace

        try:
            index = self._get_index()

            # Batch upserts in groups of 100
            batch_size = 100
            total_upserted = 0

            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]

                # Format for Pinecone
                formatted = [
                    {
                        "id": v["id"],
                        "values": v["values"],
                        "metadata": v.get("metadata", {}),
                    }
                    for v in batch
                ]

                response = await asyncio.to_thread(
                    index.upsert,
                    vectors=formatted,
                    namespace=ns,
                )

                total_upserted += response.upserted_count

            logger.info(
                "Upserted vectors to Pinecone",
                count=total_upserted,
                namespace=ns,
            )

            return total_upserted

        except Exception as e:
            logger.error("Pinecone upsert failed", error=str(e))
            raise ServiceUnavailableError(
                f"Failed to upsert vectors: {e}",
                service="pinecone",
            ) from e

    async def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter_dict: dict[str, Any] | None = None,
        namespace: str | None = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> list[dict[str, Any]]:
        """Query for similar vectors.

        Args:
            vector: Query vector.
            top_k: Number of results to return.
            filter_dict: Optional metadata filter.
            namespace: Optional namespace override.
            include_metadata: Include metadata in results.
            include_values: Include vector values in results.

        Returns:
            List of matches with id, score, and optionally metadata/values.

        Raises:
            ServiceUnavailableError: If query fails.
        """
        ns = namespace or self._namespace

        try:
            index = self._get_index()

            response = await asyncio.to_thread(
                index.query,
                vector=vector,
                top_k=top_k,
                filter=filter_dict,
                namespace=ns,
                include_metadata=include_metadata,
                include_values=include_values,
            )

            matches = []
            for match in response.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                }
                if include_metadata and match.metadata:
                    result["metadata"] = dict(match.metadata)
                if include_values and match.values:
                    result["values"] = list(match.values)
                matches.append(result)

            logger.debug(
                "Queried Pinecone",
                top_k=top_k,
                matches=len(matches),
                namespace=ns,
            )

            return matches

        except Exception as e:
            logger.error("Pinecone query failed", error=str(e))
            raise ServiceUnavailableError(
                f"Failed to query vectors: {e}",
                service="pinecone",
            ) from e

    async def delete_vectors(
        self,
        ids: list[str] | None = None,
        filter_dict: dict[str, Any] | None = None,
        namespace: str | None = None,
        delete_all: bool = False,
    ) -> None:
        """Delete vectors from Pinecone.

        Args:
            ids: List of vector IDs to delete.
            filter_dict: Metadata filter for deletion.
            namespace: Optional namespace override.
            delete_all: Delete all vectors in namespace.

        Raises:
            ServiceUnavailableError: If deletion fails.
        """
        ns = namespace or self._namespace

        try:
            index = self._get_index()

            if delete_all:
                await asyncio.to_thread(
                    index.delete,
                    delete_all=True,
                    namespace=ns,
                )
                logger.info("Deleted all vectors", namespace=ns)
            elif ids:
                # Batch deletes in groups of 1000
                batch_size = 1000
                for i in range(0, len(ids), batch_size):
                    batch = ids[i : i + batch_size]
                    await asyncio.to_thread(
                        index.delete,
                        ids=batch,
                        namespace=ns,
                    )
                logger.info("Deleted vectors by ID", count=len(ids), namespace=ns)
            elif filter_dict:
                await asyncio.to_thread(
                    index.delete,
                    filter=filter_dict,
                    namespace=ns,
                )
                logger.info("Deleted vectors by filter", filter=filter_dict, namespace=ns)

        except Exception as e:
            logger.error("Pinecone delete failed", error=str(e))
            raise ServiceUnavailableError(
                f"Failed to delete vectors: {e}",
                service="pinecone",
            ) from e

    async def fetch_vectors(
        self,
        ids: list[str],
        namespace: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Fetch vectors by ID.

        Args:
            ids: List of vector IDs to fetch.
            namespace: Optional namespace override.

        Returns:
            Dict mapping ID to vector data.

        Raises:
            ServiceUnavailableError: If fetch fails.
        """
        ns = namespace or self._namespace

        try:
            index = self._get_index()

            response = await asyncio.to_thread(
                index.fetch,
                ids=ids,
                namespace=ns,
            )

            result = {}
            for id_, vector in response.vectors.items():
                result[id_] = {
                    "id": id_,
                    "values": list(vector.values) if vector.values else [],
                    "metadata": dict(vector.metadata) if vector.metadata else {},
                }

            return result

        except Exception as e:
            logger.error("Pinecone fetch failed", error=str(e))
            raise ServiceUnavailableError(
                f"Failed to fetch vectors: {e}",
                service="pinecone",
            ) from e

    async def get_stats(self, namespace: str | None = None) -> dict[str, Any]:
        """Get index statistics.

        Args:
            namespace: Optional namespace to get stats for.

        Returns:
            Index statistics.
        """
        try:
            index = self._get_index()

            response = await asyncio.to_thread(index.describe_index_stats)

            stats = {
                "dimension": response.dimension,
                "total_vector_count": response.total_vector_count,
                "namespaces": {},
            }

            if response.namespaces:
                for ns, ns_stats in response.namespaces.items():
                    stats["namespaces"][ns] = {
                        "vector_count": ns_stats.vector_count,
                    }

            return stats

        except Exception as e:
            logger.error("Pinecone stats failed", error=str(e))
            raise ServiceUnavailableError(
                f"Failed to get index stats: {e}",
                service="pinecone",
            ) from e


@functools.lru_cache
def get_pinecone_service() -> PineconeService:
    """Get or create the Pinecone service singleton."""
    return PineconeService()
