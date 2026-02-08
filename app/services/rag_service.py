"""RAG (Retrieval-Augmented Generation) pipeline service."""

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
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

        Args:
            question: User's question.
            user_id: User ID for filtering documents.
            conv_id: Conversation ID for storing query.
            document_ids: Optional specific documents to query.
            conversation_context: Optional previous conversation messages.
            use_streaming: Whether to use streaming response.

        Returns:
            RAGResponse with answer and sources.

        Raises:
            ServiceUnavailableError: If query pipeline fails.
        """
        start_time = time.perf_counter()

        # Create query record
        db = get_dynamodb_client()
        query_obj = Query(
            conv_id=conv_id,
            user_id=user_id,
            question=question,
            document_ids=document_ids,
            status=QueryStatus.PROCESSING,
        )
        await db.put_item(query_obj.to_dynamodb_item())

        try:
            # Step 1: Generate query embedding
            embedding_service = get_embedding_service()
            query_embedding = await embedding_service.embed_query(question)

            # Step 2: Build filter for Pinecone query
            filter_dict: dict[str, Any] = {"user_id": user_id}
            if document_ids:
                filter_dict["doc_id"] = {"$in": document_ids}

            # Step 3: Query Pinecone for relevant chunks
            pinecone = get_pinecone_service()
            matches = await pinecone.query(
                vector=query_embedding,
                top_k=self._top_k,
                filter_dict=filter_dict,
                include_metadata=True,
            )

            # Step 4: Filter by similarity threshold and build sources
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

            # Step 5: Build prompt with context
            prompt = self._build_prompt(question, context_chunks, sources)

            # Step 6: Generate response with Gemini
            gemini = get_gemini_service()

            if use_streaming:
                # For streaming, we'll collect the full response
                full_answer = ""
                async for chunk in gemini.generate_stream(
                    prompt=prompt,
                    system_instruction=VETERINARY_SYSTEM_PROMPT,
                    context=conversation_context,
                ):
                    full_answer += chunk

                answer = full_answer
                token_usage = {
                    "prompt_tokens": 0,  # Not available in streaming
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            else:
                answer, token_usage = await gemini.generate(
                    prompt=prompt,
                    system_instruction=VETERINARY_SYSTEM_PROMPT,
                    context=conversation_context,
                )

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Step 7: Update query record
            await db.update_item(
                pk=f"CONV#{conv_id}",
                sk=f"QUERY#{query_obj.query_id}",
                updates={
                    "status": QueryStatus.COMPLETED.value,
                    "answer": answer,
                    "sources": [
                        {
                            "chunk_id": s.chunk_id,
                            "doc_id": s.doc_id,
                            "page_number": s.page_number,
                            "score": s.score,
                            "document_title": s.document_title,
                        }
                        for s in sources
                    ],
                    "model_used": settings.gemini_model,
                    "token_usage": token_usage,
                    "latency_ms": latency_ms,
                },
            )

            logger.info(
                "RAG query completed",
                query_id=query_obj.query_id,
                sources=len(sources),
                latency_ms=latency_ms,
            )

            return RAGResponse(
                answer=answer,
                sources=sources,
                query_id=query_obj.query_id,
                model_used=settings.gemini_model,
                token_usage=token_usage,
                latency_ms=latency_ms,
            )

        except Exception as e:
            # Update query as failed
            await db.update_item(
                pk=f"CONV#{conv_id}",
                sk=f"QUERY#{query_obj.query_id}",
                updates={
                    "status": QueryStatus.FAILED.value,
                    "error_message": str(e),
                },
            )
            logger.error("RAG query failed", error=str(e), query_id=query_obj.query_id)
            raise ServiceUnavailableError(
                f"Query processing failed: {e}",
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

        Args:
            question: User's question.
            user_id: User ID for filtering documents.
            conv_id: Conversation ID for storing query.
            document_ids: Optional specific documents to query.
            conversation_context: Optional previous conversation messages.

        Yields:
            Text chunks as they are generated.
        """
        start_time = time.perf_counter()

        # Create query record
        db = get_dynamodb_client()
        query_obj = Query(
            conv_id=conv_id,
            user_id=user_id,
            question=question,
            document_ids=document_ids,
            status=QueryStatus.PROCESSING,
        )
        await db.put_item(query_obj.to_dynamodb_item())

        full_answer = ""

        try:
            # Step 1: Generate query embedding
            embedding_service = get_embedding_service()
            query_embedding = await embedding_service.embed_query(question)

            # Step 2: Build filter for Pinecone query
            filter_dict: dict[str, Any] = {"user_id": user_id}
            if document_ids:
                filter_dict["doc_id"] = {"$in": document_ids}

            # Step 3: Query Pinecone for relevant chunks
            pinecone = get_pinecone_service()
            matches = await pinecone.query(
                vector=query_embedding,
                top_k=self._top_k,
                filter_dict=filter_dict,
                include_metadata=True,
            )

            # Step 4: Build sources and context
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

            # Step 5: Build prompt with context
            prompt = self._build_prompt(question, context_chunks, sources)

            # Step 6: Stream response from Gemini
            gemini = get_gemini_service()

            async for chunk in gemini.generate_stream(
                prompt=prompt,
                system_instruction=VETERINARY_SYSTEM_PROMPT,
                context=conversation_context,
            ):
                full_answer += chunk
                yield chunk

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Step 7: Update query record
            await db.update_item(
                pk=f"CONV#{conv_id}",
                sk=f"QUERY#{query_obj.query_id}",
                updates={
                    "status": QueryStatus.COMPLETED.value,
                    "answer": full_answer,
                    "sources": [
                        {
                            "chunk_id": s.chunk_id,
                            "doc_id": s.doc_id,
                            "page_number": s.page_number,
                            "score": s.score,
                            "document_title": s.document_title,
                        }
                        for s in sources
                    ],
                    "model_used": settings.gemini_model,
                    "latency_ms": latency_ms,
                },
            )

        except Exception as e:
            await db.update_item(
                pk=f"CONV#{conv_id}",
                sk=f"QUERY#{query_obj.query_id}",
                updates={
                    "status": QueryStatus.FAILED.value,
                    "error_message": str(e),
                },
            )
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


# Singleton instance
_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """Get or create the RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
