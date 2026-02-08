"""Tests for PDF service."""

import pytest
from unittest.mock import AsyncMock, patch

from app.services.pdf_service import PDFService, ExtractedDocument, TextChunk


@pytest.fixture
def pdf_service():
    """Create PDF service instance."""
    return PDFService()


def test_pdf_service_init(pdf_service):
    """Test PDF service initialization."""
    assert pdf_service._chunk_size > 0
    assert pdf_service._chunk_overlap > 0
    assert pdf_service._chunk_overlap < pdf_service._chunk_size


def test_assess_quality_empty_text(pdf_service):
    """Test quality assessment for empty text."""
    assert pdf_service._assess_quality("") == 0.0
    assert pdf_service._assess_quality("   ") == 0.0


def test_assess_quality_short_text(pdf_service):
    """Test quality assessment for short text."""
    score = pdf_service._assess_quality("Hi")
    assert score < 0.5


def test_assess_quality_good_text(pdf_service):
    """Test quality assessment for well-formed text."""
    text = """
    This is a well-formed paragraph with proper sentences. It contains
    multiple words and proper punctuation. The text flows naturally and
    represents typical document content. This should score highly.
    """
    score = pdf_service._assess_quality(text)
    assert score > 0.5


def test_assess_quality_garbled_text(pdf_service):
    """Test quality assessment for garbled text."""
    text = "!@#$%^&*()_+{}|:<>?~`-=[];',./\\"
    score = pdf_service._assess_quality(text)
    assert score < 0.5


@pytest.mark.asyncio
async def test_chunk_text(pdf_service):
    """Test text chunking."""
    text = "This is a test. " * 100  # Create reasonably long text

    chunks = await pdf_service.chunk_text(text, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 0
    assert all(isinstance(c, TextChunk) for c in chunks)
    assert all(len(c.content) <= 250 for c in chunks)  # Allow some overflow for word boundaries


@pytest.mark.asyncio
async def test_chunk_text_empty(pdf_service):
    """Test chunking empty text."""
    chunks = await pdf_service.chunk_text("")
    assert chunks == []


@pytest.mark.asyncio
async def test_chunk_text_preserves_content(pdf_service):
    """Test that chunking preserves all content."""
    text = "Word " * 50

    chunks = await pdf_service.chunk_text(text, chunk_size=100, chunk_overlap=20)

    # All original content should be present (accounting for overlap)
    combined = " ".join(c.content for c in chunks)
    word_count = combined.count("Word")
    assert word_count >= 50  # At least all original words


@pytest.mark.asyncio
async def test_chunk_document(pdf_service):
    """Test document chunking."""
    from app.services.pdf_service import ExtractedDocument, ExtractedPage

    doc = ExtractedDocument(
        filename="test.pdf",
        page_count=1,
        pages=[
            ExtractedPage(
                page_number=1,
                text="This is test content. " * 50,
                extraction_method="pymupdf",
                quality_score=0.9,
                metadata={},
            )
        ],
        full_text="This is test content. " * 50,
        metadata={},
    )

    chunks = await pdf_service.chunk_document(doc, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 0
    assert all(c.page_number == 1 for c in chunks)
    assert all(c.chunk_index >= 0 for c in chunks)
