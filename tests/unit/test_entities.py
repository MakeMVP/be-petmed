"""Tests for entity models."""

import pytest

from app.models.entities import (
    Chunk,
    Conversation,
    Document,
    DocumentStatus,
    Message,
    MessageRole,
    Query,
    QueryStatus,
    User,
    generate_ulid,
)


def test_generate_ulid():
    """Test ULID generation."""
    ulid1 = generate_ulid()
    ulid2 = generate_ulid()

    assert len(ulid1) == 26  # ULID is 26 characters
    assert ulid1 != ulid2  # Should be unique


def test_user_entity():
    """Test User entity creation and DynamoDB item conversion."""
    user = User(
        user_id="user-123",
        email="test@example.com",
        name="Test User",
    )

    item = user.to_dynamodb_item()

    assert item["PK"] == "USER#user-123"
    assert item["SK"] == "USER#user-123"
    assert item["GSI1PK"] == "USER#EMAIL"
    assert item["GSI1SK"] == "test@example.com"
    assert item["entity_type"] == "USER"
    assert item["email"] == "test@example.com"


def test_document_entity():
    """Test Document entity creation."""
    doc = Document(
        user_id="user-123",
        title="Test Document",
        filename="test.pdf",
        file_size=1024,
        s3_key="documents/user-123/doc-123/test.pdf",
    )

    assert doc.status == DocumentStatus.PENDING
    assert doc.doc_id is not None

    item = doc.to_dynamodb_item()

    assert item["PK"] == f"USER#user-123"
    assert item["SK"].startswith("DOC#")
    assert item["entity_type"] == "DOC"


def test_document_status_enum():
    """Test DocumentStatus enum values."""
    assert DocumentStatus.PENDING.value == "pending"
    assert DocumentStatus.PROCESSING.value == "processing"
    assert DocumentStatus.COMPLETED.value == "completed"
    assert DocumentStatus.FAILED.value == "failed"


def test_chunk_entity():
    """Test Chunk entity creation."""
    chunk = Chunk(
        doc_id="doc-123",
        user_id="user-123",
        content="This is test content for the chunk.",
        page_number=1,
        chunk_index=0,
    )

    item = chunk.to_dynamodb_item()

    assert item["PK"] == "DOC#doc-123"
    assert item["SK"].startswith("CHUNK#")
    assert item["content"] == "This is test content for the chunk."


def test_conversation_entity():
    """Test Conversation entity creation."""
    conv = Conversation(
        user_id="user-123",
        title="Test Conversation",
    )

    assert conv.message_count == 0
    assert conv.last_message_at is None

    item = conv.to_dynamodb_item()

    assert item["PK"] == "USER#user-123"
    assert item["SK"].startswith("CONV#")


def test_message_entity():
    """Test Message entity creation."""
    msg = Message(
        conv_id="conv-123",
        role=MessageRole.USER,
        content="Hello, how can you help me?",
    )

    item = msg.to_dynamodb_item()

    assert item["PK"] == "CONV#conv-123"
    assert item["SK"].startswith("MSG#")
    assert item["role"] == "user"


def test_message_role_enum():
    """Test MessageRole enum values."""
    assert MessageRole.USER.value == "user"
    assert MessageRole.ASSISTANT.value == "assistant"
    assert MessageRole.SYSTEM.value == "system"


def test_query_entity():
    """Test Query entity creation."""
    query = Query(
        conv_id="conv-123",
        user_id="user-123",
        question="What are the symptoms of distemper?",
    )

    assert query.status == QueryStatus.PENDING
    assert query.answer is None
    assert query.sources == []

    item = query.to_dynamodb_item()

    assert item["PK"] == "CONV#conv-123"
    assert item["SK"].startswith("QUERY#")
    assert item["GSI1PK"] == "USER#user-123"
