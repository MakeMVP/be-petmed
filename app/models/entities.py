"""Pydantic models for DynamoDB entities."""

import hashlib
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from ulid import ULID


def generate_ulid() -> str:
    """Generate a new ULID string."""
    return str(ULID())


def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(UTC)


class EntityType(str, Enum):
    """Entity type enumeration."""

    USER = "USER"
    DOCUMENT = "DOC"
    CHUNK = "CHUNK"
    CONVERSATION = "CONV"
    MESSAGE = "MSG"
    QUERY = "QUERY"


# --- GSI2 write sharding for documents ---
GSI2_DOC_SHARDS = 10


def doc_gsi2_pk(doc_id: str) -> str:
    """Compute the sharded GSI2 partition key for a document.

    Uses SHA-256 for deterministic distribution (Python's hash() is randomized).
    """
    digest = hashlib.sha256(doc_id.encode()).digest()
    shard = digest[0] % GSI2_DOC_SHARDS
    return f"DOC#S{shard}"


# All shard PKs to query (includes legacy unsharded "DOC" for pre-migration items)
DOC_GSI2_PKS = ["DOC"] + [f"DOC#S{i}" for i in range(GSI2_DOC_SHARDS)]


def _sanitize_floats(value: Any) -> Any:
    """Recursively convert float values to Decimal for DynamoDB compatibility."""
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: _sanitize_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_floats(v) for v in value]
    return value


class BaseEntity(BaseModel):
    """Base entity with common fields."""

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def to_dynamodb_item(self) -> dict[str, Any]:
        """Convert to DynamoDB item format."""
        item = self.model_dump(mode="json")
        # Remove None values and convert floats to Decimal (DynamoDB requirement)
        return {k: _sanitize_floats(v) for k, v in item.items() if v is not None}

    @classmethod
    def from_dynamodb_item(cls, item: dict[str, Any]) -> "BaseEntity":
        """Create entity from DynamoDB item."""
        return cls.model_validate(item)


class DocumentStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class User(BaseEntity):
    """User entity.

    PK: USER#{user_id}
    SK: USER#{user_id}
    GSI1PK: USER#EMAIL
    GSI1SK: {email}
    """

    user_id: str = Field(description="Cognito user ID (sub)")
    email: str = Field(description="User email address")
    name: str | None = Field(default=None, description="Display name")
    avatar_url: str | None = Field(default=None, description="Profile avatar URL")
    settings: dict[str, Any] = Field(
        default_factory=dict, description="User preferences"
    )
    document_count: int = Field(default=0, description="Denormalized document count")
    conversation_count: int = Field(default=0, description="Denormalized conversation count")
    query_count: int = Field(default=0, description="Denormalized query count")
    storage_used_bytes: int = Field(default=0, description="Denormalized storage used")

    def to_dynamodb_item(self) -> dict[str, Any]:
        """Convert to DynamoDB item with keys."""
        item = super().to_dynamodb_item()
        item["PK"] = f"USER#{self.user_id}"
        item["SK"] = f"USER#{self.user_id}"
        item["GSI1PK"] = "USER#EMAIL"
        item["GSI1SK"] = self.email
        item["entity_type"] = EntityType.USER.value
        return item


class Document(BaseEntity):
    """Document entity.

    PK: USER#{user_id}
    SK: DOC#{doc_id}
    GSI1PK: DOC#{doc_id}
    GSI1SK: DOC#{doc_id}
    """

    doc_id: str = Field(default_factory=generate_ulid, description="Document ULID")
    user_id: str = Field(description="Owner user ID")
    title: str = Field(description="Document title")
    filename: str = Field(description="Original filename")
    file_size: int = Field(description="File size in bytes")
    mime_type: str = Field(default="application/pdf", description="MIME type")
    s3_key: str = Field(description="S3 object key")
    status: DocumentStatus = Field(
        default=DocumentStatus.PENDING, description="Processing status"
    )
    page_count: int | None = Field(default=None, description="Number of pages")
    chunk_count: int | None = Field(default=None, description="Number of chunks")
    error_message: str | None = Field(
        default=None, description="Error message if failed"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def to_dynamodb_item(self) -> dict[str, Any]:
        """Convert to DynamoDB item with keys."""
        item = super().to_dynamodb_item()
        item["PK"] = f"USER#{self.user_id}"
        item["SK"] = f"DOC#{self.doc_id}"
        item["GSI1PK"] = f"DOC#{self.doc_id}"
        item["GSI1SK"] = f"DOC#{self.doc_id}"
        # GSI2 for admin list-all-documents queries (sharded to avoid hot partition)
        item["GSI2PK"] = doc_gsi2_pk(self.doc_id)
        item["GSI2SK"] = self.created_at.isoformat()
        item["entity_type"] = EntityType.DOCUMENT.value
        return item


class Chunk(BaseEntity):
    """Document chunk entity for RAG.

    PK: DOC#{doc_id}
    SK: CHUNK#{chunk_id}
    """

    chunk_id: str = Field(default_factory=generate_ulid, description="Chunk ULID")
    doc_id: str = Field(description="Parent document ID")
    user_id: str = Field(description="Owner user ID")
    content: str = Field(description="Chunk text content")
    page_number: int | None = Field(default=None, description="Source page number")
    chunk_index: int = Field(description="Chunk order index")
    token_count: int | None = Field(default=None, description="Token count")
    embedding_id: str | None = Field(default=None, description="Pinecone vector ID")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def to_dynamodb_item(self) -> dict[str, Any]:
        """Convert to DynamoDB item with keys."""
        item = super().to_dynamodb_item()
        item["PK"] = f"DOC#{self.doc_id}"
        item["SK"] = f"CHUNK#{self.chunk_id}"
        item["entity_type"] = EntityType.CHUNK.value
        return item


class Conversation(BaseEntity):
    """Conversation entity for chat history.

    PK: USER#{user_id}
    SK: CONV#{conv_id}
    GSI1PK: CONV#{conv_id}
    GSI1SK: CONV#{conv_id}
    """

    conv_id: str = Field(
        default_factory=generate_ulid, description="Conversation ULID"
    )
    user_id: str = Field(description="Owner user ID")
    title: str = Field(default="New Conversation", description="Conversation title")
    message_count: int = Field(default=0, description="Number of messages")
    last_message_at: datetime | None = Field(
        default=None, description="Last message timestamp"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def to_dynamodb_item(self) -> dict[str, Any]:
        """Convert to DynamoDB item with keys."""
        item = super().to_dynamodb_item()
        item["PK"] = f"USER#{self.user_id}"
        item["SK"] = f"CONV#{self.conv_id}"
        item["GSI1PK"] = f"CONV#{self.conv_id}"
        item["GSI1SK"] = f"CONV#{self.conv_id}"
        item["entity_type"] = EntityType.CONVERSATION.value
        return item


class MessageRole(str, Enum):
    """Message role enumeration."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseEntity):
    """Chat message entity.

    PK: CONV#{conv_id}
    SK: MSG#{ulid}  (ULID ensures chronological ordering)
    """

    message_id: str = Field(default_factory=generate_ulid, description="Message ULID")
    conv_id: str = Field(description="Parent conversation ID")
    role: MessageRole = Field(description="Message role")
    content: str = Field(description="Message content")
    query_id: str | None = Field(
        default=None, description="Associated query ID if RAG query"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def to_dynamodb_item(self) -> dict[str, Any]:
        """Convert to DynamoDB item with keys."""
        item = super().to_dynamodb_item()
        item["PK"] = f"CONV#{self.conv_id}"
        item["SK"] = f"MSG#{self.message_id}"
        item["entity_type"] = EntityType.MESSAGE.value
        return item


class QueryStatus(str, Enum):
    """Query processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Query(BaseEntity):
    """RAG query entity.

    PK: CONV#{conv_id}
    SK: QUERY#{query_id}
    GSI1PK: USER#{user_id}
    GSI1SK: QUERY#{timestamp}
    """

    query_id: str = Field(default_factory=generate_ulid, description="Query ULID")
    conv_id: str = Field(description="Parent conversation ID")
    user_id: str = Field(description="Owner user ID")
    question: str = Field(description="User question")
    answer: str | None = Field(default=None, description="Generated answer")
    status: QueryStatus = Field(
        default=QueryStatus.PENDING, description="Processing status"
    )
    sources: list[dict[str, Any]] = Field(
        default_factory=list, description="Source chunks used"
    )
    document_ids: list[str] | None = Field(
        default=None, description="Specific documents to query"
    )
    model_used: str | None = Field(default=None, description="LLM model used")
    token_usage: dict[str, int] | None = Field(
        default=None, description="Token usage stats"
    )
    latency_ms: int | None = Field(default=None, description="Processing latency")
    feedback_rating: int | None = Field(
        default=None, ge=1, le=5, description="User rating 1-5"
    )
    feedback_comment: str | None = Field(
        default=None, description="User feedback comment"
    )
    error_message: str | None = Field(
        default=None, description="Error message if failed"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def to_dynamodb_item(self) -> dict[str, Any]:
        """Convert to DynamoDB item with keys."""
        item = super().to_dynamodb_item()
        item["PK"] = f"CONV#{self.conv_id}"
        item["SK"] = f"QUERY#{self.query_id}"
        item["GSI1PK"] = f"USER#{self.user_id}"
        item["GSI1SK"] = f"QUERY#{self.created_at.isoformat()}"
        item["entity_type"] = EntityType.QUERY.value
        return item
