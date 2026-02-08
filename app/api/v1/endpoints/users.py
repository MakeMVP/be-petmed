"""User management endpoints."""

from fastapi import APIRouter

from app.api.v1.schemas.common import DeleteResponse, ERROR_RESPONSES
from app.api.v1.schemas.users import (
    UpdateProfileRequest,
    UserProfileResponse,
    UserStatsResponse,
)
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.dependencies import CurrentUser, DynamoDB, Storage, VectorDB

router = APIRouter(prefix="/users", tags=["Users"])
logger = get_logger(__name__)


@router.get(
    "/me",
    response_model=UserProfileResponse,
    summary="Get user profile",
    description="Get the current user's profile.",
    responses={
        401: ERROR_RESPONSES[401],
        404: ERROR_RESPONSES[404],
    },
)
async def get_profile(
    current_user: CurrentUser,
    db: DynamoDB,
) -> UserProfileResponse:
    """Get current user's profile."""
    pk = f"USER#{current_user.user_id}"
    user = await db.get_item(pk=pk, sk=pk)

    if not user:
        raise NotFoundError(
            "User profile not found",
            resource_type="user",
            resource_id=current_user.user_id,
        )

    return UserProfileResponse(
        user_id=user["user_id"],
        email=user["email"],
        name=user.get("name"),
        avatar_url=user.get("avatar_url"),
        settings=user.get("settings", {}),
        created_at=user["created_at"],
        updated_at=user["updated_at"],
    )


@router.patch(
    "/me",
    response_model=UserProfileResponse,
    summary="Update user profile",
    description="Update the current user's profile.",
    responses={
        401: ERROR_RESPONSES[401],
        404: ERROR_RESPONSES[404],
    },
)
async def update_profile(
    request: UpdateProfileRequest,
    current_user: CurrentUser,
    db: DynamoDB,
) -> UserProfileResponse:
    """Update current user's profile."""
    pk = f"USER#{current_user.user_id}"

    # Build update dict
    updates = {}
    if request.name is not None:
        updates["name"] = request.name
    if request.avatar_url is not None:
        updates["avatar_url"] = request.avatar_url
    if request.settings is not None:
        # Merge settings
        existing = await db.get_item(pk=pk, sk=pk)
        if existing:
            existing_settings = existing.get("settings", {})
            existing_settings.update(request.settings)
            updates["settings"] = existing_settings
        else:
            updates["settings"] = request.settings

    if not updates:
        # Nothing to update, return current
        return await get_profile(current_user, db)

    updated = await db.update_item(pk=pk, sk=pk, updates=updates)

    logger.info("Updated user profile", user_id=current_user.user_id)

    return UserProfileResponse(
        user_id=updated["user_id"],
        email=updated["email"],
        name=updated.get("name"),
        avatar_url=updated.get("avatar_url"),
        settings=updated.get("settings", {}),
        created_at=updated["created_at"],
        updated_at=updated["updated_at"],
    )


@router.get(
    "/me/stats",
    response_model=UserStatsResponse,
    summary="Get user statistics",
    description="Get statistics about the current user's usage.",
    responses={
        401: ERROR_RESPONSES[401],
    },
)
async def get_user_stats(
    current_user: CurrentUser,
    db: DynamoDB,
) -> UserStatsResponse:
    """Get user usage statistics."""
    user_pk = f"USER#{current_user.user_id}"

    # Count documents
    docs, _ = await db.query(pk=user_pk, sk_begins_with="DOC#")
    doc_count = len(docs)
    storage_used = sum(d.get("file_size", 0) for d in docs)

    # Count conversations
    convs, _ = await db.query(pk=user_pk, sk_begins_with="CONV#")
    conv_count = len(convs)

    # Count queries (via GSI)
    queries, _ = await db.query(
        pk=user_pk,
        sk_begins_with="QUERY#",
        index_name="GSI1",
    )
    query_count = len(queries)

    return UserStatsResponse(
        document_count=doc_count,
        conversation_count=conv_count,
        query_count=query_count,
        storage_used_bytes=storage_used,
    )


@router.delete(
    "/me",
    response_model=DeleteResponse,
    summary="Delete user account",
    description="Delete the current user's account and all associated data.",
    responses={
        401: ERROR_RESPONSES[401],
    },
)
async def delete_account(
    current_user: CurrentUser,
    db: DynamoDB,
    storage: Storage,
    vector_db: VectorDB,
) -> DeleteResponse:
    """Delete user account and all associated data."""
    user_pk = f"USER#{current_user.user_id}"

    # Get all documents
    docs, _ = await db.query(pk=user_pk, sk_begins_with="DOC#")

    # Delete S3 files and collect doc IDs
    for doc in docs:
        s3_key = doc.get("s3_key")
        if s3_key:
            await storage.delete_file(s3_key)

    # Delete all vectors for this user
    await vector_db.delete_vectors(
        filter_dict={"user_id": current_user.user_id}
    )

    # Get all conversations
    convs, _ = await db.query(pk=user_pk, sk_begins_with="CONV#")

    # Collect all items to delete
    items_to_delete = [(user_pk, user_pk)]  # User record

    # Add documents
    for doc in docs:
        items_to_delete.append((doc["PK"], doc["SK"]))
        # Get chunks for this doc
        doc_id = doc["doc_id"]
        chunks, _ = await db.query(pk=f"DOC#{doc_id}", sk_begins_with="CHUNK#")
        for chunk in chunks:
            items_to_delete.append((chunk["PK"], chunk["SK"]))

    # Add conversations and messages
    for conv in convs:
        items_to_delete.append((conv["PK"], conv["SK"]))
        conv_id = conv["conv_id"]
        # Get messages
        messages, _ = await db.query(pk=f"CONV#{conv_id}", sk_begins_with="MSG#")
        for msg in messages:
            items_to_delete.append((msg["PK"], msg["SK"]))
        # Get queries
        queries, _ = await db.query(pk=f"CONV#{conv_id}", sk_begins_with="QUERY#")
        for query in queries:
            items_to_delete.append((query["PK"], query["SK"]))

    # Batch delete all items
    await db.batch_delete(items_to_delete)

    logger.info(
        "Deleted user account",
        user_id=current_user.user_id,
        items_deleted=len(items_to_delete),
    )

    return DeleteResponse(
        success=True,
        deleted_id=current_user.user_id,
    )
