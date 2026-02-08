"""RAG query endpoints."""

import json
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.v1.schemas.common import ERROR_RESPONSES
from app.api.v1.schemas.queries import (
    FeedbackRequest,
    FeedbackResponse,
    QueryRequest,
    QueryResponse,
    QueryStatusResponse,
    SourceResponse,
)
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.dependencies import CurrentUser, DynamoDB
from app.models.entities import Conversation, Message, MessageRole, QueryStatus
from app.services.rag_service import get_rag_service

router = APIRouter(prefix="/queries", tags=["Queries"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=QueryResponse,
    summary="Submit RAG query",
    description="Submit a question and get an answer based on uploaded documents.",
    responses={
        401: ERROR_RESPONSES[401],
        503: ERROR_RESPONSES[500],
    },
)
async def submit_query(
    request: QueryRequest,
    current_user: CurrentUser,
    db: DynamoDB,
) -> QueryResponse:
    """Submit a RAG query and get response."""
    # Get or create conversation
    conv_id = request.conversation_id

    if conv_id:
        # Verify conversation exists and belongs to user
        conv = await db.get_item(
            pk=f"USER#{current_user.user_id}",
            sk=f"CONV#{conv_id}",
        )
        if not conv:
            raise NotFoundError(
                f"Conversation not found: {conv_id}",
                resource_type="conversation",
                resource_id=conv_id,
            )
    else:
        # Create new conversation
        conv = Conversation(
            user_id=current_user.user_id,
            title=request.question[:50] + "..." if len(request.question) > 50 else request.question,
        )
        await db.put_item(conv.to_dynamodb_item())
        conv_id = conv.conv_id
        logger.debug("Created new conversation", conv_id=conv_id)

    # Get conversation context if exists
    context = None
    if request.conversation_id:
        messages, _ = await db.query(
            pk=f"CONV#{conv_id}",
            sk_begins_with="MSG#",
            limit=10,
            scan_forward=False,  # Most recent first
        )
        if messages:
            # Reverse to chronological order
            context = [
                {"role": m["role"], "content": m["content"]}
                for m in reversed(messages)
            ]

    # Execute RAG query
    rag = get_rag_service()
    response = await rag.query(
        question=request.question,
        user_id=current_user.user_id,
        conv_id=conv_id,
        document_ids=request.document_ids,
        conversation_context=context,
    )

    # Store user message
    user_msg = Message(
        conv_id=conv_id,
        role=MessageRole.USER,
        content=request.question,
    )
    await db.put_item(user_msg.to_dynamodb_item())

    # Store assistant message
    assistant_msg = Message(
        conv_id=conv_id,
        role=MessageRole.ASSISTANT,
        content=response.answer,
        query_id=response.query_id,
    )
    await db.put_item(assistant_msg.to_dynamodb_item())

    # Update conversation
    await db.update_item(
        pk=f"USER#{current_user.user_id}",
        sk=f"CONV#{conv_id}",
        updates={
            "message_count": (conv.get("message_count", 0) if isinstance(conv, dict) else conv.message_count) + 2,
            "last_message_at": assistant_msg.created_at.isoformat(),
        },
    )

    # Build response
    sources = [
        SourceResponse(
            chunk_id=s.chunk_id,
            doc_id=s.doc_id,
            document_title=s.document_title,
            page_number=s.page_number,
            score=s.score,
            content_preview=s.content[:200] + "..." if len(s.content) > 200 else s.content,
        )
        for s in response.sources
    ]

    return QueryResponse(
        query_id=response.query_id,
        conversation_id=conv_id,
        question=request.question,
        answer=response.answer,
        sources=sources,
        model_used=response.model_used,
        latency_ms=response.latency_ms,
        created_at=user_msg.created_at,
        updated_at=assistant_msg.created_at,
    )


@router.post(
    "/stream",
    summary="Submit streaming RAG query",
    description="Submit a question and get a streaming response via SSE.",
    responses={
        401: ERROR_RESPONSES[401],
        503: ERROR_RESPONSES[500],
    },
)
async def submit_streaming_query(
    request: QueryRequest,
    current_user: CurrentUser,
    db: DynamoDB,
) -> StreamingResponse:
    """Submit a RAG query with streaming response."""
    # Get or create conversation
    conv_id = request.conversation_id

    if conv_id:
        conv = await db.get_item(
            pk=f"USER#{current_user.user_id}",
            sk=f"CONV#{conv_id}",
        )
        if not conv:
            raise NotFoundError(
                f"Conversation not found: {conv_id}",
                resource_type="conversation",
                resource_id=conv_id,
            )
    else:
        conv = Conversation(
            user_id=current_user.user_id,
            title=request.question[:50] + "..." if len(request.question) > 50 else request.question,
        )
        await db.put_item(conv.to_dynamodb_item())
        conv_id = conv.conv_id

    # Get conversation context
    context = None
    if request.conversation_id:
        messages, _ = await db.query(
            pk=f"CONV#{conv_id}",
            sk_begins_with="MSG#",
            limit=10,
            scan_forward=False,
        )
        if messages:
            context = [
                {"role": m["role"], "content": m["content"]}
                for m in reversed(messages)
            ]

    # Store user message
    user_msg = Message(
        conv_id=conv_id,
        role=MessageRole.USER,
        content=request.question,
    )
    await db.put_item(user_msg.to_dynamodb_item())

    async def generate_stream():
        """Generate SSE stream."""
        rag = get_rag_service()
        full_answer = ""

        try:
            async for chunk in rag.query_stream(
                question=request.question,
                user_id=current_user.user_id,
                conv_id=conv_id,
                document_ids=request.document_ids,
                conversation_context=context,
            ):
                full_answer += chunk
                event = {"type": "text", "content": chunk}
                yield f"data: {json.dumps(event)}\n\n"

            # Store assistant message
            assistant_msg = Message(
                conv_id=conv_id,
                role=MessageRole.ASSISTANT,
                content=full_answer,
            )
            await db.put_item(assistant_msg.to_dynamodb_item())

            # Send done event
            event = {"type": "done", "conversation_id": conv_id}
            yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            logger.error("Streaming query failed", error=str(e))
            event = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/{query_id}",
    response_model=QueryResponse,
    summary="Get query result",
    description="Get the result of a previous query.",
    responses={
        401: ERROR_RESPONSES[401],
        404: ERROR_RESPONSES[404],
    },
)
async def get_query(
    query_id: str,
    current_user: CurrentUser,
    db: DynamoDB,
) -> QueryResponse:
    """Get query result by ID."""
    # Query uses GSI1 with USER#{user_id} as PK
    queries, _ = await db.query(
        pk=f"USER#{current_user.user_id}",
        sk_begins_with=f"QUERY#",
        index_name="GSI1",
    )

    # Find the specific query
    query = None
    for q in queries:
        if q.get("query_id") == query_id:
            query = q
            break

    if not query:
        raise NotFoundError(
            f"Query not found: {query_id}",
            resource_type="query",
            resource_id=query_id,
        )

    sources = [
        SourceResponse(
            chunk_id=s["chunk_id"],
            doc_id=s["doc_id"],
            document_title=s.get("document_title"),
            page_number=s.get("page_number"),
            score=s["score"],
        )
        for s in query.get("sources", [])
    ]

    return QueryResponse(
        query_id=query["query_id"],
        conversation_id=query["conv_id"],
        question=query["question"],
        answer=query.get("answer", ""),
        sources=sources,
        model_used=query.get("model_used"),
        latency_ms=query.get("latency_ms"),
        created_at=query["created_at"],
        updated_at=query["updated_at"],
    )


@router.post(
    "/{query_id}/feedback",
    response_model=FeedbackResponse,
    summary="Submit feedback",
    description="Submit feedback for a query response.",
    responses={
        401: ERROR_RESPONSES[401],
        404: ERROR_RESPONSES[404],
    },
)
async def submit_feedback(
    query_id: str,
    request: FeedbackRequest,
    current_user: CurrentUser,
    db: DynamoDB,
) -> FeedbackResponse:
    """Submit feedback for a query."""
    # Find the query (need to search since we don't know conv_id)
    queries, _ = await db.query(
        pk=f"USER#{current_user.user_id}",
        sk_begins_with="QUERY#",
        index_name="GSI1",
    )

    query = None
    for q in queries:
        if q.get("query_id") == query_id:
            query = q
            break

    if not query:
        raise NotFoundError(
            f"Query not found: {query_id}",
            resource_type="query",
            resource_id=query_id,
        )

    # Update with feedback
    rag = get_rag_service()
    await rag.submit_feedback(
        query_id=query_id,
        conv_id=query["conv_id"],
        rating=request.rating,
        comment=request.comment,
    )

    logger.info("Query feedback submitted", query_id=query_id, rating=request.rating)

    return FeedbackResponse()
