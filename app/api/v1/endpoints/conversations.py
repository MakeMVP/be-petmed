"""Conversation management endpoints."""

from fastapi import APIRouter

from app.api.v1.schemas.common import DeleteResponse, ERROR_RESPONSES
from app.api.v1.schemas.conversations import (
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationResponse,
    CreateConversationRequest,
    MessageListResponse,
    MessageResponse,
    UpdateConversationRequest,
)
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.db.pagination import decode_cursor, encode_cursor
from app.dependencies import CurrentUser, DynamoDB
from app.models.entities import Conversation, MessageRole

router = APIRouter(prefix="/conversations", tags=["Conversations"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=201,
    summary="Create conversation",
    description="Create a new conversation.",
    responses={
        401: ERROR_RESPONSES[401],
    },
)
async def create_conversation(
    request: CreateConversationRequest,
    current_user: CurrentUser,
    db: DynamoDB,
) -> ConversationResponse:
    """Create a new conversation."""
    conv = Conversation(
        user_id=current_user.user_id,
        title=request.title,
    )

    await db.put_item(conv.to_dynamodb_item())

    # Increment user's denormalized conversation count
    user_pk = f"USER#{current_user.user_id}"
    await db.increment_counter(pk=user_pk, sk=user_pk, counter_attr="conversation_count")

    logger.info("Created conversation", conv_id=conv.conv_id)

    return ConversationResponse(
        conv_id=conv.conv_id,
        title=conv.title,
        message_count=0,
        last_message_at=None,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


@router.get(
    "",
    response_model=ConversationListResponse,
    summary="List conversations",
    description="List all conversations for the current user.",
    responses={
        401: ERROR_RESPONSES[401],
    },
)
async def list_conversations(
    current_user: CurrentUser,
    db: DynamoDB,
    limit: int = 20,
    cursor: str | None = None,
) -> ConversationListResponse:
    """List user's conversations."""
    exclusive_start_key = decode_cursor(cursor)

    convs, last_key = await db.query(
        pk=f"USER#{current_user.user_id}",
        sk_begins_with="CONV#",
        limit=limit,
        exclusive_start_key=exclusive_start_key,
        scan_forward=False,  # Most recent first
    )

    has_more = last_key is not None
    next_cursor = encode_cursor(last_key)

    items = [
        ConversationResponse(
            conv_id=c["conv_id"],
            title=c["title"],
            message_count=c.get("message_count", 0),
            last_message_at=c.get("last_message_at"),
            created_at=c["created_at"],
            updated_at=c["updated_at"],
        )
        for c in convs
    ]

    return ConversationListResponse(
        items=items,
        next_cursor=next_cursor,
        has_more=has_more,
    )


@router.get(
    "/{conv_id}",
    response_model=ConversationDetailResponse,
    summary="Get conversation",
    description="Get conversation details with recent messages.",
    responses={
        401: ERROR_RESPONSES[401],
        404: ERROR_RESPONSES[404],
    },
)
async def get_conversation(
    conv_id: str,
    current_user: CurrentUser,
    db: DynamoDB,
) -> ConversationDetailResponse:
    """Get conversation with messages."""
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

    # Get recent messages
    messages, _ = await db.query(
        pk=f"CONV#{conv_id}",
        sk_begins_with="MSG#",
        limit=50,
        scan_forward=True,  # Chronological order
    )

    message_items = [
        MessageResponse(
            message_id=m["message_id"],
            role=MessageRole(m["role"]),
            content=m["content"],
            query_id=m.get("query_id"),
            created_at=m["created_at"],
            updated_at=m["updated_at"],
        )
        for m in messages
    ]

    return ConversationDetailResponse(
        conv_id=conv["conv_id"],
        title=conv["title"],
        message_count=conv.get("message_count", 0),
        last_message_at=conv.get("last_message_at"),
        created_at=conv["created_at"],
        updated_at=conv["updated_at"],
        messages=message_items,
    )


@router.get(
    "/{conv_id}/messages",
    response_model=MessageListResponse,
    summary="Get messages",
    description="Get paginated messages from a conversation.",
    responses={
        401: ERROR_RESPONSES[401],
        404: ERROR_RESPONSES[404],
    },
)
async def get_messages(
    conv_id: str,
    current_user: CurrentUser,
    db: DynamoDB,
    limit: int = 50,
    cursor: str | None = None,
) -> MessageListResponse:
    """Get paginated conversation messages."""
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

    exclusive_start_key = decode_cursor(cursor)

    messages, last_key = await db.query(
        pk=f"CONV#{conv_id}",
        sk_begins_with="MSG#",
        limit=limit,
        exclusive_start_key=exclusive_start_key,
        scan_forward=True,
    )

    has_more = last_key is not None
    next_cursor = encode_cursor(last_key)

    items = [
        MessageResponse(
            message_id=m["message_id"],
            role=MessageRole(m["role"]),
            content=m["content"],
            query_id=m.get("query_id"),
            created_at=m["created_at"],
            updated_at=m["updated_at"],
        )
        for m in messages
    ]

    return MessageListResponse(
        items=items,
        next_cursor=next_cursor,
        has_more=has_more,
    )


@router.patch(
    "/{conv_id}",
    response_model=ConversationResponse,
    summary="Update conversation",
    description="Update conversation title.",
    responses={
        401: ERROR_RESPONSES[401],
        404: ERROR_RESPONSES[404],
    },
)
async def update_conversation(
    conv_id: str,
    request: UpdateConversationRequest,
    current_user: CurrentUser,
    db: DynamoDB,
) -> ConversationResponse:
    """Update conversation title."""
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

    updated = await db.update_item(
        pk=f"USER#{current_user.user_id}",
        sk=f"CONV#{conv_id}",
        updates={"title": request.title},
    )

    logger.info("Updated conversation", conv_id=conv_id)

    return ConversationResponse(
        conv_id=updated["conv_id"],
        title=updated["title"],
        message_count=updated.get("message_count", 0),
        last_message_at=updated.get("last_message_at"),
        created_at=updated["created_at"],
        updated_at=updated["updated_at"],
    )


@router.delete(
    "/{conv_id}",
    response_model=DeleteResponse,
    summary="Delete conversation",
    description="Delete a conversation and all its messages.",
    responses={
        401: ERROR_RESPONSES[401],
        404: ERROR_RESPONSES[404],
    },
)
async def delete_conversation(
    conv_id: str,
    current_user: CurrentUser,
    db: DynamoDB,
) -> DeleteResponse:
    """Delete a conversation."""
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

    # Get all message and query keys (paginated to handle >1MB)
    msg_keys = await db.query_all_keys(
        pk=f"CONV#{conv_id}", sk_begins_with="MSG#"
    )
    query_keys = await db.query_all_keys(
        pk=f"CONV#{conv_id}", sk_begins_with="QUERY#"
    )

    # Collect all items to delete
    items_to_delete: list[tuple[str, str]] = [
        (f"USER#{current_user.user_id}", f"CONV#{conv_id}"),
    ]
    items_to_delete.extend(msg_keys)
    items_to_delete.extend(query_keys)

    # Clean up query index items (PK=QUERY#{query_id})
    for _pk, sk in query_keys:
        query_id = sk.removeprefix("QUERY#")
        items_to_delete.append((f"QUERY#{query_id}", f"QUERY#{query_id}"))

    # Batch delete
    await db.batch_delete(items_to_delete)

    # Decrement user's denormalized conversation count
    user_pk = f"USER#{current_user.user_id}"
    await db.increment_counter(
        pk=user_pk, sk=user_pk, counter_attr="conversation_count", increment=-1
    )

    logger.info(
        "Deleted conversation",
        conv_id=conv_id,
        messages=len(msg_keys),
        queries=len(query_keys),
    )

    return DeleteResponse(
        success=True,
        deleted_id=conv_id,
    )
