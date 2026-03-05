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
from app.services.user_service import delete_user_data

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
    """Get user usage statistics from denormalized counts on User entity."""
    user_pk = f"USER#{current_user.user_id}"
    user = await db.get_item(pk=user_pk, sk=user_pk)

    if not user:
        return UserStatsResponse(
            document_count=0,
            conversation_count=0,
            query_count=0,
            storage_used_bytes=0,
        )

    return UserStatsResponse(
        document_count=user.get("document_count", 0),
        conversation_count=user.get("conversation_count", 0),
        query_count=user.get("query_count", 0),
        storage_used_bytes=user.get("storage_used_bytes", 0),
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
    """Delete user account and all associated data via shared service."""
    items_deleted = await delete_user_data(
        current_user.user_id, db, storage, vector_db
    )

    logger.info(
        "Deleted user account",
        user_id=current_user.user_id,
        items_deleted=items_deleted,
    )

    return DeleteResponse(
        success=True,
        deleted_id=current_user.user_id,
    )
