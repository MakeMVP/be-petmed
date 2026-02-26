"""Admin management endpoints."""

import base64
import json

from boto3.dynamodb.conditions import Attr
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
from app.dependencies import DynamoDB, Storage, VectorDB
from app.models.entities import Document, DocumentStatus, MessageRole
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
    """List all users via GSI1."""
    exclusive_start_key = None
    if cursor:
        try:
            exclusive_start_key = json.loads(base64.b64decode(cursor))
        except Exception:
            raise BadRequestError("Invalid cursor")

    users, last_key = await db.query(
        pk="USER#EMAIL",
        index_name="GSI1",
        limit=limit + 1,
        exclusive_start_key=exclusive_start_key,
        scan_forward=True,
    )

    has_more = len(users) > limit
    if has_more:
        users = users[:limit]

    next_cursor = None
    if last_key and has_more:
        next_cursor = base64.b64encode(json.dumps(last_key).encode()).decode()

    items = []
    for u in users:
        user_settings = u.get("settings", {})
        user_pk = f"USER#{u['user_id']}"

        # Count documents and conversations
        docs, _ = await db.query(pk=user_pk, sk_begins_with="DOC#")
        convs, _ = await db.query(pk=user_pk, sk_begins_with="CONV#")

        items.append(
            AdminUserResponse(
                user_id=u["user_id"],
                email=u["email"],
                name=u.get("name"),
                avatar_url=u.get("avatar_url"),
                is_active=user_settings.get("is_active", True),
                role=user_settings.get("role", "user"),
                message_limit=user_settings.get("message_limit", 50),
                document_count=len(docs),
                conversation_count=len(convs),
                created_at=u["created_at"],
                updated_at=u["updated_at"],
            )
        )

    return AdminUserListResponse(
        items=items,
        next_cursor=next_cursor,
        has_more=has_more,
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
    """Get any user's details."""
    pk = f"USER#{user_id}"
    user = await db.get_item(pk=pk, sk=pk)

    if not user:
        raise NotFoundError(
            "User not found",
            resource_type="user",
            resource_id=user_id,
        )

    user_settings = user.get("settings", {})

    # Count documents and conversations
    docs, _ = await db.query(pk=pk, sk_begins_with="DOC#")
    convs, _ = await db.query(pk=pk, sk_begins_with="CONV#")

    return AdminUserResponse(
        user_id=user["user_id"],
        email=user["email"],
        name=user.get("name"),
        avatar_url=user.get("avatar_url"),
        is_active=user_settings.get("is_active", True),
        role=user_settings.get("role", "user"),
        message_limit=user_settings.get("message_limit", 50),
        document_count=len(docs),
        conversation_count=len(convs),
        created_at=user["created_at"],
        updated_at=user["updated_at"],
    )


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

    updated_settings = updated.get("settings", {})
    docs, _ = await db.query(pk=pk, sk_begins_with="DOC#")
    convs, _ = await db.query(pk=pk, sk_begins_with="CONV#")

    return AdminUserResponse(
        user_id=updated["user_id"],
        email=updated["email"],
        name=updated.get("name"),
        avatar_url=updated.get("avatar_url"),
        is_active=updated_settings.get("is_active", True),
        role=updated_settings.get("role", "user"),
        message_limit=updated_settings.get("message_limit", 50),
        document_count=len(docs),
        conversation_count=len(convs),
        created_at=updated["created_at"],
        updated_at=updated["updated_at"],
    )


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
    """Delete a user and all associated data."""
    user_pk = f"USER#{user_id}"

    user = await db.get_item(pk=user_pk, sk=user_pk)
    if not user:
        raise NotFoundError(
            "User not found",
            resource_type="user",
            resource_id=user_id,
        )

    # Get all documents
    docs, _ = await db.query(pk=user_pk, sk_begins_with="DOC#")

    # Delete S3 files
    for doc in docs:
        s3_key = doc.get("s3_key")
        if s3_key:
            await storage.delete_file(s3_key)

    # Delete all vectors for this user
    await vector_db.delete_vectors(filter_dict={"user_id": user_id})

    # Get all conversations
    convs, _ = await db.query(pk=user_pk, sk_begins_with="CONV#")

    # Collect all items to delete
    items_to_delete = [(user_pk, user_pk)]

    for doc in docs:
        items_to_delete.append((doc["PK"], doc["SK"]))
        doc_id = doc["doc_id"]
        chunks, _ = await db.query(pk=f"DOC#{doc_id}", sk_begins_with="CHUNK#")
        for chunk in chunks:
            items_to_delete.append((chunk["PK"], chunk["SK"]))

    for conv in convs:
        items_to_delete.append((conv["PK"], conv["SK"]))
        conv_id = conv["conv_id"]
        messages, _ = await db.query(pk=f"CONV#{conv_id}", sk_begins_with="MSG#")
        for msg in messages:
            items_to_delete.append((msg["PK"], msg["SK"]))
        queries, _ = await db.query(pk=f"CONV#{conv_id}", sk_begins_with="QUERY#")
        for query in queries:
            items_to_delete.append((query["PK"], query["SK"]))

    await db.batch_delete(items_to_delete)

    logger.info(
        "Admin deleted user",
        user_id=user_id,
        admin_id=admin.user_id,
        items_deleted=len(items_to_delete),
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

    exclusive_start_key = None
    if cursor:
        try:
            exclusive_start_key = json.loads(base64.b64decode(cursor))
        except Exception:
            raise BadRequestError("Invalid cursor")

    convs, last_key = await db.query(
        pk=user_pk,
        sk_begins_with="CONV#",
        limit=limit + 1,
        exclusive_start_key=exclusive_start_key,
        scan_forward=False,
    )

    has_more = len(convs) > limit
    if has_more:
        convs = convs[:limit]

    next_cursor = None
    if last_key and has_more:
        next_cursor = base64.b64encode(json.dumps(last_key).encode()).decode()

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
    """List all documents via scan with entity_type filter."""
    exclusive_start_key = None
    if cursor:
        try:
            exclusive_start_key = json.loads(base64.b64decode(cursor))
        except Exception:
            raise BadRequestError("Invalid cursor")

    docs, last_key = await db.scan(
        filter_expression=Attr("entity_type").eq("DOC"),
        limit=limit + 1,
        exclusive_start_key=exclusive_start_key,
    )

    has_more = len(docs) > limit
    if has_more:
        docs = docs[:limit]

    next_cursor = None
    if last_key and has_more:
        next_cursor = base64.b64encode(json.dumps(last_key).encode()).decode()

    # Fetch owner emails
    user_cache: dict[str, str | None] = {}
    items = []
    for doc in docs:
        uid = doc["user_id"]
        if uid not in user_cache:
            u = await db.get_item(pk=f"USER#{uid}", sk=f"USER#{uid}")
            user_cache[uid] = u["email"] if u else None

        items.append(
            AdminDocumentResponse(
                doc_id=doc["doc_id"],
                user_id=uid,
                user_email=user_cache[uid],
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
        )

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

    from urllib.parse import quote

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

    await db.put_item(doc.to_dynamodb_item())

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
    """Get system-wide statistics."""
    # Count users
    users, _ = await db.query(
        pk="USER#EMAIL",
        index_name="GSI1",
    )
    total_users = len(users)

    # Count documents via scan
    docs, _ = await db.scan(
        filter_expression=Attr("entity_type").eq("DOC"),
    )
    total_documents = len(docs)

    # Count by status
    documents_by_status: dict[str, int] = {}
    for doc in docs:
        status = doc.get("status", "unknown")
        documents_by_status[status] = documents_by_status.get(status, 0) + 1

    # Count queries and compute avg rating
    queries, _ = await db.scan(
        filter_expression=Attr("entity_type").eq("QUERY"),
    )
    total_queries = len(queries)

    ratings = [
        q["feedback_rating"]
        for q in queries
        if q.get("feedback_rating") is not None
    ]
    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else None

    return SystemStatsResponse(
        total_users=total_users,
        total_documents=total_documents,
        total_queries=total_queries,
        avg_rating=avg_rating,
        documents_by_status=documents_by_status,
    )
