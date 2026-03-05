"""Admin management endpoints."""

import asyncio
from datetime import UTC, datetime, timedelta
from urllib.parse import quote

from fastapi import APIRouter, File, Form, UploadFile

from app.api.v1.schemas.admin import (
    AdminConversationDetailResponse,
    AdminConversationListResponse,
    AdminConversationResponse,
    AdminDocumentListResponse,
    AdminDocumentResponse,
    AdminMessageResponse,
    AdminUpdateUserRequest,
    AdminUserListResponse,
    AdminUserResponse,
    SystemStatsResponse,
)
from app.api.v1.schemas.common import DeleteResponse, ERROR_RESPONSES
from app.api.v1.schemas.documents import DocumentUploadResponse
from app.config import settings
from app.core.admin_auth import AdminUser
from app.core.exceptions import BadRequestError, NotFoundError
from app.core.logging import get_logger
from app.db.pagination import decode_cursor, encode_cursor
from app.dependencies import DynamoDB, Storage, VectorDB
from app.models.entities import DOC_GSI2_PKS, Document, DocumentStatus, MessageRole
from app.services.user_service import delete_user_data
from app.workers.tasks import enqueue_document_processing

router = APIRouter(prefix="/admin", tags=["Admin"])
logger = get_logger(__name__)


# --- User endpoints ---


@router.get(
    "/users",
    response_model=AdminUserListResponse,
    summary="List all users",
    description="List all users in the system (admin only).",
    responses={
        401: ERROR_RESPONSES[401],
        403: ERROR_RESPONSES[403],
    },
)
async def list_users(
    admin: AdminUser,
    db: DynamoDB,
    limit: int = 20,
    cursor: str | None = None,
) -> AdminUserListResponse:
    """List all users via GSI1 — uses denormalized counts instead of N+1 queries."""
    exclusive_start_key = decode_cursor(cursor)

    users, last_key = await db.query(
        pk="USER#EMAIL",
        index_name="GSI1",
        limit=limit,
        exclusive_start_key=exclusive_start_key,
        scan_forward=True,
    )

    has_more = last_key is not None
    next_cursor = encode_cursor(last_key)

    return AdminUserListResponse(
        items=[_build_admin_user_response(u) for u in users],
        next_cursor=next_cursor,
        has_more=has_more,
    )


def _build_admin_user_response(user: dict) -> AdminUserResponse:
    """Build AdminUserResponse from a DynamoDB user item using denormalized counts."""
    user_settings = user.get("settings", {})
    return AdminUserResponse(
        user_id=user["user_id"],
        email=user["email"],
        name=user.get("name"),
        avatar_url=user.get("avatar_url"),
        is_active=user_settings.get("is_active", True),
        role=user_settings.get("role", "user"),
        message_limit=user_settings.get("message_limit", 50),
        document_count=user.get("document_count", 0),
        conversation_count=user.get("conversation_count", 0),
        created_at=user["created_at"],
        updated_at=user["updated_at"],
    )


@router.get(
    "/users/{user_id}",
    response_model=AdminUserResponse,
    summary="Get user details",
    description="Get details of any user (admin only).",
    responses={
        401: ERROR_RESPONSES[401],
        403: ERROR_RESPONSES[403],
        404: ERROR_RESPONSES[404],
    },
)
async def get_user(
    user_id: str,
    admin: AdminUser,
    db: DynamoDB,
) -> AdminUserResponse:
    """Get any user's details — uses denormalized counts."""
    pk = f"USER#{user_id}"
    user = await db.get_item(pk=pk, sk=pk)

    if not user:
        raise NotFoundError(
            "User not found",
            resource_type="user",
            resource_id=user_id,
        )

    return _build_admin_user_response(user)


@router.patch(
    "/users/{user_id}",
    response_model=AdminUserResponse,
    summary="Update user",
    description="Update user settings (admin only).",
    responses={
        401: ERROR_RESPONSES[401],
        403: ERROR_RESPONSES[403],
        404: ERROR_RESPONSES[404],
    },
)
async def update_user(
    user_id: str,
    request: AdminUpdateUserRequest,
    admin: AdminUser,
    db: DynamoDB,
) -> AdminUserResponse:
    """Update user settings as admin."""
    pk = f"USER#{user_id}"
    user = await db.get_item(pk=pk, sk=pk)

    if not user:
        raise NotFoundError(
            "User not found",
            resource_type="user",
            resource_id=user_id,
        )

    # Merge settings
    existing_settings = user.get("settings", {})
    if request.is_active is not None:
        existing_settings["is_active"] = request.is_active
    if request.message_limit is not None:
        existing_settings["message_limit"] = request.message_limit
    if request.role is not None:
        if request.role not in ("user", "admin"):
            raise BadRequestError("Role must be 'user' or 'admin'")
        existing_settings["role"] = request.role

    updated = await db.update_item(
        pk=pk, sk=pk, updates={"settings": existing_settings}
    )

    logger.info("Admin updated user", user_id=user_id, admin_id=admin.user_id)

    return _build_admin_user_response(updated)


@router.delete(
    "/users/{user_id}",
    response_model=DeleteResponse,
    summary="Delete user",
    description="Delete a user and all their data (admin only).",
    responses={
        401: ERROR_RESPONSES[401],
        403: ERROR_RESPONSES[403],
        404: ERROR_RESPONSES[404],
    },
)
async def delete_user(
    user_id: str,
    admin: AdminUser,
    db: DynamoDB,
    storage: Storage,
    vector_db: VectorDB,
) -> DeleteResponse:
    """Delete a user and all associated data via shared service."""
    user_pk = f"USER#{user_id}"

    user = await db.get_item(pk=user_pk, sk=user_pk)
    if not user:
        raise NotFoundError(
            "User not found",
            resource_type="user",
            resource_id=user_id,
        )

    items_deleted = await delete_user_data(user_id, db, storage, vector_db)

    logger.info(
        "Admin deleted user",
        user_id=user_id,
        admin_id=admin.user_id,
        items_deleted=items_deleted,
    )

    return DeleteResponse(success=True, deleted_id=user_id)


# --- User conversations ---


@router.get(
    "/users/{user_id}/conversations",
    response_model=AdminConversationListResponse,
    summary="List user conversations",
    description="List conversations for a specific user (admin only).",
    responses={
        401: ERROR_RESPONSES[401],
        403: ERROR_RESPONSES[403],
        404: ERROR_RESPONSES[404],
    },
)
async def list_user_conversations(
    user_id: str,
    admin: AdminUser,
    db: DynamoDB,
    limit: int = 20,
    cursor: str | None = None,
) -> AdminConversationListResponse:
    """List a user's conversations."""
    # Verify user exists
    user_pk = f"USER#{user_id}"
    user = await db.get_item(pk=user_pk, sk=user_pk)
    if not user:
        raise NotFoundError(
            "User not found",
            resource_type="user",
            resource_id=user_id,
        )

    exclusive_start_key = decode_cursor(cursor)

    convs, last_key = await db.query(
        pk=user_pk,
        sk_begins_with="CONV#",
        limit=limit,
        exclusive_start_key=exclusive_start_key,
        scan_forward=False,
    )

    has_more = last_key is not None
    next_cursor = encode_cursor(last_key)

    items = [
        AdminConversationResponse(
            conv_id=c["conv_id"],
            user_id=user_id,
            title=c["title"],
            message_count=c.get("message_count", 0),
            last_message_at=c.get("last_message_at"),
            created_at=c["created_at"],
            updated_at=c["updated_at"],
        )
        for c in convs
    ]

    return AdminConversationListResponse(
        items=items,
        next_cursor=next_cursor,
        has_more=has_more,
    )


@router.get(
    "/users/{user_id}/conversations/{conv_id}",
    response_model=AdminConversationDetailResponse,
    summary="Get conversation detail",
    description="Get conversation with messages (admin only).",
    responses={
        401: ERROR_RESPONSES[401],
        403: ERROR_RESPONSES[403],
        404: ERROR_RESPONSES[404],
    },
)
async def get_user_conversation(
    user_id: str,
    conv_id: str,
    admin: AdminUser,
    db: DynamoDB,
) -> AdminConversationDetailResponse:
    """Get a user's conversation with messages."""
    conv = await db.get_item(
        pk=f"USER#{user_id}",
        sk=f"CONV#{conv_id}",
    )

    if not conv:
        raise NotFoundError(
            f"Conversation not found: {conv_id}",
            resource_type="conversation",
            resource_id=conv_id,
        )

    messages, _ = await db.query(
        pk=f"CONV#{conv_id}",
        sk_begins_with="MSG#",
        limit=100,
        scan_forward=True,
    )

    message_items = [
        AdminMessageResponse(
            message_id=m["message_id"],
            role=MessageRole(m["role"]),
            content=m["content"],
            query_id=m.get("query_id"),
            created_at=m["created_at"],
            updated_at=m["updated_at"],
        )
        for m in messages
    ]

    return AdminConversationDetailResponse(
        conv_id=conv["conv_id"],
        user_id=user_id,
        title=conv["title"],
        message_count=conv.get("message_count", 0),
        last_message_at=conv.get("last_message_at"),
        created_at=conv["created_at"],
        updated_at=conv["updated_at"],
        messages=message_items,
    )


# --- Document endpoints ---


@router.get(
    "/documents",
    response_model=AdminDocumentListResponse,
    summary="List all documents",
    description="List all documents in the system (admin only).",
    responses={
        401: ERROR_RESPONSES[401],
        403: ERROR_RESPONSES[403],
    },
)
async def list_all_documents(
    admin: AdminUser,
    db: DynamoDB,
    limit: int = 20,
    cursor: str | None = None,
) -> AdminDocumentListResponse:
    """List all documents via sharded GSI2 (avoids hot partition)."""
    raw_cursor = decode_cursor(cursor)
    shard_cursor = raw_cursor if raw_cursor and "s" in raw_cursor else None

    docs, next_shard_state = await db.query_across_shards(
        shard_pks=DOC_GSI2_PKS,
        index_name="GSI2",
        limit=limit,
        cursor=shard_cursor,
        scan_forward=False,  # Most recent first
    )

    has_more = next_shard_state is not None
    next_cursor = encode_cursor(next_shard_state)

    # Batch fetch owner emails instead of N+1 get_item calls
    unique_user_ids = list({doc["user_id"] for doc in docs})
    user_keys = [(f"USER#{uid}", f"USER#{uid}") for uid in unique_user_ids]
    user_items = await db.batch_get_items(user_keys)
    user_email_map: dict[str, str | None] = {
        u["user_id"]: u.get("email") for u in user_items
    }

    items = [
        AdminDocumentResponse(
            doc_id=doc["doc_id"],
            user_id=doc["user_id"],
            user_email=user_email_map.get(doc["user_id"]),
            title=doc["title"],
            filename=doc["filename"],
            file_size=doc["file_size"],
            mime_type=doc.get("mime_type", "application/pdf"),
            status=DocumentStatus(doc["status"]),
            page_count=doc.get("page_count"),
            chunk_count=doc.get("chunk_count"),
            error_message=doc.get("error_message"),
            metadata=doc.get("metadata", {}),
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
        )
        for doc in docs
    ]

    return AdminDocumentListResponse(
        items=items,
        next_cursor=next_cursor,
        has_more=has_more,
    )


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=201,
    summary="Upload admin document",
    description="Upload a PDF to the knowledge base as admin.",
    responses={
        400: ERROR_RESPONSES[400],
        401: ERROR_RESPONSES[401],
        403: ERROR_RESPONSES[403],
    },
)
async def upload_admin_document(
    admin: AdminUser,
    db: DynamoDB,
    storage: Storage,
    file: UploadFile = File(..., description="PDF file to upload"),
    title: str = Form(None, description="Document title"),
) -> DocumentUploadResponse:
    """Upload a PDF document to the knowledge base as admin."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise BadRequestError("Only PDF files are supported")

    content = await file.read()
    file_size = len(content)

    if file_size > settings.pdf_max_size_bytes:
        raise BadRequestError(
            f"File exceeds maximum size of {settings.pdf_max_size_mb}MB"
        )

    if file_size == 0:
        raise BadRequestError("File is empty")

    doc = Document(
        user_id=admin.user_id,
        title=title or file.filename or "Untitled Document",
        filename=file.filename or "document.pdf",
        file_size=file_size,
        s3_key="",
        metadata={"uploaded_by_admin": True},
    )

    s3_key = f"documents/{admin.user_id}/{doc.doc_id}/{doc.doc_id}.pdf"
    doc.s3_key = s3_key

    await storage.upload_file(
        file_data=content,
        s3_key=s3_key,
        content_type="application/pdf",
        metadata={
            "doc_id": doc.doc_id,
            "user_id": admin.user_id,
            "original_filename": quote(doc.filename, safe=""),
            "admin_upload": "true",
        },
    )

    # Atomically write document + increment counters in one transaction
    user_pk = f"USER#{admin.user_id}"
    await db.transact_put_and_increment(
        item=doc.to_dynamodb_item(),
        counter_pk=user_pk,
        counter_sk=user_pk,
        counters={"document_count": 1, "storage_used_bytes": file_size},
    )

    job_id = await enqueue_document_processing(
        doc_id=doc.doc_id,
        user_id=admin.user_id,
        s3_key=s3_key,
        filename=doc.filename,
    )

    logger.info(
        "Admin document upload started",
        doc_id=doc.doc_id,
        admin_id=admin.user_id,
        filename=doc.filename,
        size=file_size,
    )

    return DocumentUploadResponse(
        doc_id=doc.doc_id,
        job_id=job_id,
        status="pending",
        message="Document upload started. Processing will begin shortly.",
    )


# --- Stats endpoint ---


@router.get(
    "/stats",
    response_model=SystemStatsResponse,
    summary="Get system statistics",
    description="Get system-wide statistics (admin only).",
    responses={
        401: ERROR_RESPONSES[401],
        403: ERROR_RESPONSES[403],
    },
)
async def get_system_stats(
    admin: AdminUser,
    db: DynamoDB,
) -> SystemStatsResponse:
    """Get system-wide statistics using efficient count queries."""
    # Check for pre-computed stats entity first (cached for 5 minutes)
    stats_item = await db.get_item(pk="STATS#SYSTEM", sk="STATS#SYSTEM")
    if stats_item:
        computed_at = stats_item.get("computed_at")
        if computed_at:
            age = datetime.now(UTC) - datetime.fromisoformat(computed_at)
            if age < timedelta(minutes=5):
                return SystemStatsResponse(
                    total_users=stats_item.get("total_users", 0),
                    total_documents=stats_item.get("total_documents", 0),
                    total_queries=stats_item.get("total_queries", 0),
                    avg_rating=stats_item.get("avg_rating"),
                    documents_by_status=stats_item.get("documents_by_status", {}),
                )

    # Compute stats: count users and documents across all GSI2 shards in parallel
    doc_count_tasks = [
        db.query_count(pk=shard_pk, index_name="GSI2") for shard_pk in DOC_GSI2_PKS
    ]
    user_count_task = db.query_count(pk="USER#EMAIL", index_name="GSI1")
    doc_counts, total_users = await asyncio.gather(
        asyncio.gather(*doc_count_tasks), user_count_task
    )
    total_documents = sum(doc_counts)

    # Paginated loop across all shards for documents_by_status
    async def _count_statuses(shard_pk: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        shard_cursor: dict | None = None
        while True:
            docs, shard_cursor = await db.query(
                pk=shard_pk,
                index_name="GSI2",
                limit=500,
                exclusive_start_key=shard_cursor,
                projection_expression="#s",
                expression_attribute_names={"#s": "status"},
            )
            for doc in docs:
                status = doc.get("status", "unknown")
                counts[status] = counts.get(status, 0) + 1
            if not shard_cursor:
                break
        return counts

    status_results = await asyncio.gather(
        *[_count_statuses(pk) for pk in DOC_GSI2_PKS]
    )
    documents_by_status: dict[str, int] = {}
    for shard_counts in status_results:
        for status, count in shard_counts.items():
            documents_by_status[status] = documents_by_status.get(status, 0) + count

    # Paginated loop for total_queries — sum query_count across users
    total_queries = 0
    user_cursor: dict | None = None
    while True:
        users, user_cursor = await db.query(
            pk="USER#EMAIL",
            index_name="GSI1",
            limit=500,
            exclusive_start_key=user_cursor,
            projection_expression="query_count",
        )
        total_queries += sum(u.get("query_count", 0) for u in users)
        if not user_cursor:
            break

    # Cache computed stats for 5 minutes
    now = datetime.now(UTC)
    await db.put_item({
        "PK": "STATS#SYSTEM",
        "SK": "STATS#SYSTEM",
        "entity_type": "STATS",
        "total_users": total_users,
        "total_documents": total_documents,
        "total_queries": total_queries,
        "avg_rating": None,
        "documents_by_status": documents_by_status,
        "computed_at": now.isoformat(),
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    })

    return SystemStatsResponse(
        total_users=total_users,
        total_documents=total_documents,
        total_queries=total_queries,
        avg_rating=None,
        documents_by_status=documents_by_status,
    )
