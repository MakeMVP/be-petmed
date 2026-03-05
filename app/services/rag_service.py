"""RAG (Retrieval-Augmented Generation) pipeline service."""

import functools
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.core.exceptions import NotFoundError, ServiceUnavailableError
from app.core.logging import get_logger
from app.db.dynamodb import get_dynamodb_client
from app.models.entities import Chunk, Document, DocumentStatus, Query, QueryStatus
from app.services.embedding_service import get_embedding_service
from app.services.gemini_service import get_gemini_service
from app.services.pinecone_service import get_pinecone_service

logger = get_logger(__name__)


@dataclass
class RAGSource:
    """Represents a source chunk used in RAG response."""

    chunk_id: str
    doc_id: str
    content: str
    page_number: int | None
    score: float
    document_title: str | None = None


@dataclass
class RAGResponse:
    """Complete RAG response with sources."""

    answer: str
    sources: list[RAGSource]
    query_id: str
    model_used: str
    token_usage: dict[str, int]
    latency_ms: int


@dataclass
class _PreparedQuery:
    """Internal result of the retrieval phase."""

    query_obj: Query
    prompt: str
    sources: list[RAGSource]
    source_dicts: list[dict[str, Any]] = field(default_factory=list)


# System prompt for veterinary diagnosis assistance
VETERINARY_SYSTEM_PROMPT = """You are a veterinary diagnosis and treatment support assistant.
Your role is to help veterinary professionals by providing accurate, evidence-based information
from veterinary medical literature and knowledge bases.

Important guidelines:
1. Always cite your sources when providing information
2. Clearly distinguish between established facts and clinical opinions
3. Recommend consulting with specialists when cases require expertise beyond general practice
4. Never provide definitive diagnoses - only differential diagnoses and considerations
5. Always emphasize the importance of physical examination and diagnostic testing
6. For medication dosages, always verify with current pharmacological references
7. If information is not available in the provided context, clearly state this

When responding:
- Structure your answers clearly with relevant sections
- Use medical terminology appropriately but explain when needed
- Provide practical, actionable information
- Include relevant warnings or contraindications
- Reference specific documents/sources when available"""


class RAGService:
    """Service for RAG pipeline operations."""

    def __init__(self) -> None:
        self._top_k = settings.rag_top_k
        self._similarity_threshold = settings.rag_similarity_threshold

    async def _prepare_query(
        self,
        question: str,
        user_id: str,
        conv_id: str,
        document_ids: list[str] | None = None,
    ) -> _PreparedQuery:
        """Shared retrieval phase: create record, embed, search, build prompt."""
        db = get_dynamodb_client()
        query_obj = Query(
            conv_id=conv_id,
            user_id=user_id,
            question=question,
            document_ids=document_ids,
            status=QueryStatus.PROCESSING,
        )
        await db.put_item(query_obj.to_dynamodb_item())
        await db.put_item({
            "PK": f"QUERY#{query_obj.query_id}",
            "SK": f"QUERY#{query_obj.query_id}",
            "conv_id": conv_id,
            "user_id": user_id,
            "entity_type": "QUERY_INDEX",
        })

        embedding_service = get_embedding_service()
        query_embedding = await embedding_service.embed_query(question)

        filter_dict: dict[str, Any] = {"user_id": user_id}
        if document_ids:
            filter_dict["doc_id"] = {"$in": document_ids}

        pinecone = get_pinecone_service()
        matches = await pinecone.query(
            vector=query_embedding,
            top_k=self._top_k,
            filter_dict=filter_dict,
            include_metadata=True,
        )

        sources: list[RAGSource] = []
        context_chunks: list[str] = []
        for match in matches:
            if match["score"] < self._similarity_threshold:
                continue
            metadata = match.get("metadata", {})
            sources.append(
                RAGSource(
                    chunk_id=match["id"],
                    doc_id=metadata.get("doc_id", ""),
                    content=metadata.get("content", ""),
                    page_number=metadata.get("page_number"),
                    score=match["score"],
                    document_title=metadata.get("document_title"),
                )
            )
            context_chunks.append(metadata.get("content", ""))

        prompt = self._build_prompt(question, context_chunks, sources)

        source_dicts = [
            {
                "chunk_id": s.chunk_id,
                "doc_id": s.doc_id,
                "page_number": s.page_number,
                "score": s.score,
                "document_title": s.document_title,
            }
            for s in sources
        ]

        return _PreparedQuery(
            query_obj=query_obj,
            prompt=prompt,
            sources=sources,
            source_dicts=source_dicts,
        )

    async def _finalize_query(
        self,
        conv_id: str,
        query_id: str,
        answer: str,
        source_dicts: list[dict[str, Any]],
        latency_ms: int,
        token_usage: dict[str, int] | None = None,
    ) -> None:
        """Update the query record after successful generation."""
        db = get_dynamodb_client()
        updates: dict[str, Any] = {
            "status": QueryStatus.COMPLETED.value,
            "answer": answer,
            "sources": source_dicts,
            "model_used": settings.gemini_model,
            "latency_ms": latency_ms,
        }
        if token_usage is not None:
            updates["token_usage"] = token_usage
        await db.update_item(
            pk=f"CONV#{conv_id}",
            sk=f"QUERY#{query_id}",
            updates=updates,
        )

    async def _fail_query(self, conv_id: str, query_id: str, error: Exception) -> None:
        """Mark a query as failed."""
        db = get_dynamodb_client()
        await db.update_item(
            pk=f"CONV#{conv_id}",
            sk=f"QUERY#{query_id}",
            updates={
                "status": QueryStatus.FAILED.value,
                "error_message": "Query processing failed. Please try again.",
            },
        )
        logger.error("RAG query failed", error=str(error), query_id=query_id, exc_info=True)

    async def query(
        self,
        question: str,
        user_id: str,
        conv_id: str,
        document_ids: list[str] | None = None,
        conversation_context: list[dict[str, Any]] | None = None,
        use_streaming: bool = False,
    ) -> RAGResponse:
        """Execute a RAG query.

        Returns:
            RAGResponse with answer and sources.

        Raises:
            ServiceUnavailableError: If query pipeline fails.
        """
        start_time = time.perf_counter()

        prepared = await self._prepare_query(question, user_id, conv_id, document_ids)

        try:
            gemini = get_gemini_service()

            if use_streaming:
                full_answer = ""
                async for chunk in gemini.generate_stream(
                    prompt=prepared.prompt,
                    system_instruction=VETERINARY_SYSTEM_PROMPT,
                    context=conversation_context,
                ):
                    full_answer += chunk
                answer = full_answer
                token_usage: dict[str, int] = {}
            else:
                answer, token_usage = await gemini.generate(
                    prompt=prepared.prompt,
                    system_instruction=VETERINARY_SYSTEM_PROMPT,
                    context=conversation_context,
                )

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            await self._finalize_query(
                conv_id, prepared.query_obj.query_id, answer,
                prepared.source_dicts, latency_ms, token_usage,
            )

            logger.info(
                "RAG query completed",
                query_id=prepared.query_obj.query_id,
                sources=len(prepared.sources),
                latency_ms=latency_ms,
            )

            return RAGResponse(
                answer=answer,
                sources=prepared.sources,
                query_id=prepared.query_obj.query_id,
                model_used=settings.gemini_model,
                token_usage=token_usage,
                latency_ms=latency_ms,
            )

        except Exception as e:
            await self._fail_query(conv_id, prepared.query_obj.query_id, e)
            raise ServiceUnavailableError(
                "Query processing failed. Please try again.",
                service="rag",
            ) from e

    async def query_stream(
        self,
        question: str,
        user_id: str,
        conv_id: str,
        document_ids: list[str] | None = None,
        conversation_context: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Execute a RAG query with streaming response.

        Yields:
            Text chunks as they are generated.
        """
        start_time = time.perf_counter()

        prepared = await self._prepare_query(question, user_id, conv_id, document_ids)
        full_answer = ""

        try:
            gemini = get_gemini_service()

            async for chunk in gemini.generate_stream(
                prompt=prepared.prompt,
                system_instruction=VETERINARY_SYSTEM_PROMPT,
                context=conversation_context,
            ):
                full_answer += chunk
                yield chunk

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            await self._finalize_query(
                conv_id, prepared.query_obj.query_id, full_answer,
                prepared.source_dicts, latency_ms,
            )

        except Exception as e:
            await self._fail_query(conv_id, prepared.query_obj.query_id, e)
            raise

    def _build_prompt(
        self,
        question: str,
        context_chunks: list[str],
        sources: list[RAGSource],
    ) -> str:
        """Build the RAG prompt with context.

        Args:
            question: User's question.
            context_chunks: Retrieved context chunks.
            sources: Source metadata for citations.

        Returns:
            Formatted prompt string.
        """
        if not context_chunks:
            return f"""Question: {question}

Note: No relevant documents were found in your knowledge base for this query.
Please provide a response based on general veterinary knowledge, and clearly
indicate that this response is not based on uploaded documents."""

        # Build context section with source references
        context_parts = []
        for i, (chunk, source) in enumerate(zip(context_chunks, sources), 1):
            doc_ref = source.document_title or f"Document {source.doc_id[:8]}"
            page_ref = f", Page {source.page_number}" if source.page_number else ""
            context_parts.append(f"[Source {i}: {doc_ref}{page_ref}]\n{chunk}")

        context_text = "\n\n---\n\n".join(context_parts)

        return f"""Based on the following context from veterinary documents, please answer the question.
Always cite your sources using [Source N] format when using information from the context.

CONTEXT:
{context_text}

QUESTION: {question}

Please provide a comprehensive, well-structured answer based on the context above.
If the context doesn't contain sufficient information to fully answer the question,
acknowledge this and provide what information is available."""

    async def submit_feedback(
        self,
        query_id: str,
        conv_id: str,
        rating: int,
        comment: str | None = None,
    ) -> None:
        """Submit feedback for a query.

        Args:
            query_id: Query ID.
            conv_id: Conversation ID.
            rating: Rating 1-5.
            comment: Optional feedback comment.
        """
        db = get_dynamodb_client()

        updates = {"feedback_rating": rating}
        if comment:
            updates["feedback_comment"] = comment

        await db.update_item(
            pk=f"CONV#{conv_id}",
            sk=f"QUERY#{query_id}",
            updates=updates,
        )

        logger.info("Submitted query feedback", query_id=query_id, rating=rating)


@functools.lru_cache
def get_rag_service() -> RAGService:
    """Get or create the RAG service singleton."""
    return RAGService()
