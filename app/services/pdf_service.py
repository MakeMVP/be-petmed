"""PDF extraction service with Unstructured and Gemini Vision fallback."""

import asyncio
import io
import re
from dataclasses import dataclass
from typing import Any

import fitz  # PyMuPDF

from app.config import settings
from app.core.exceptions import BadRequestError, ServiceUnavailableError
from app.core.logging import get_logger
from app.services.gemini_service import get_gemini_service

logger = get_logger(__name__)


@dataclass
class ExtractedPage:
    """Represents extracted content from a PDF page."""

    page_number: int
    text: str
    extraction_method: str
    quality_score: float
    metadata: dict[str, Any]


@dataclass
class ExtractedDocument:
    """Represents fully extracted PDF document."""

    filename: str
    page_count: int
    pages: list[ExtractedPage]
    full_text: str
    metadata: dict[str, Any]


@dataclass
class TextChunk:
    """Represents a text chunk for RAG."""

    content: str
    page_number: int | None
    chunk_index: int
    token_count: int
    metadata: dict[str, Any]


class PDFService:
    """Service for PDF extraction and chunking."""

    def __init__(self) -> None:
        self._chunk_size = settings.rag_chunk_size
        self._chunk_overlap = settings.rag_chunk_overlap
        self._max_pages = settings.pdf_max_pages
        self._max_size = settings.pdf_max_size_bytes

    async def extract_text(
        self,
        pdf_data: bytes,
        filename: str = "document.pdf",
        use_vision_fallback: bool = True,
    ) -> ExtractedDocument:
        """Extract text from a PDF document.

        Args:
            pdf_data: Raw PDF bytes.
            filename: Original filename.
            use_vision_fallback: Use Gemini Vision for low-quality pages.

        Returns:
            ExtractedDocument with all page content.

        Raises:
            BadRequestError: If PDF is invalid or too large.
            ServiceUnavailableError: If extraction fails.
        """
        # Validate size
        if len(pdf_data) > self._max_size:
            raise BadRequestError(
                f"PDF exceeds maximum size of {settings.pdf_max_size_mb}MB"
            )

        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(stream=pdf_data, filetype="pdf")

            # Validate page count
            if doc.page_count > self._max_pages:
                doc.close()
                raise BadRequestError(
                    f"PDF exceeds maximum page count of {self._max_pages}"
                )

            pages: list[ExtractedPage] = []
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
            }

            # Extract each page
            for page_num in range(doc.page_count):
                page = doc[page_num]
                extracted = await self._extract_page(
                    page,
                    page_num + 1,  # 1-indexed
                    pdf_data if use_vision_fallback else None,
                )
                pages.append(extracted)

            doc.close()

            # Combine full text
            full_text = "\n\n".join(p.text for p in pages if p.text.strip())

            return ExtractedDocument(
                filename=filename,
                page_count=len(pages),
                pages=pages,
                full_text=full_text,
                metadata=metadata,
            )

        except BadRequestError:
            raise
        except Exception as e:
            logger.error("PDF extraction failed", error=str(e), filename=filename)
            raise ServiceUnavailableError(
                f"Failed to extract PDF: {e}",
                service="pdf-extraction",
            ) from e

    async def _extract_page(
        self,
        page: fitz.Page,
        page_number: int,
        pdf_data: bytes | None = None,
    ) -> ExtractedPage:
        """Extract text from a single page.

        Args:
            page: PyMuPDF page object.
            page_number: 1-indexed page number.
            pdf_data: Original PDF data for vision fallback.

        Returns:
            ExtractedPage with text content.
        """
        # Try PyMuPDF text extraction first
        text = page.get_text("text")
        quality_score = self._assess_quality(text)

        # If quality is low and we have PDF data, try Gemini Vision
        if quality_score < 0.5 and pdf_data:
            try:
                vision_text = await self._extract_with_vision(pdf_data, page_number)
                vision_quality = self._assess_quality(vision_text)

                if vision_quality > quality_score:
                    return ExtractedPage(
                        page_number=page_number,
                        text=vision_text,
                        extraction_method="gemini-vision",
                        quality_score=vision_quality,
                        metadata={"fallback_used": True},
                    )
            except Exception as e:
                logger.warning(
                    "Vision fallback failed",
                    page=page_number,
                    error=str(e),
                )

        return ExtractedPage(
            page_number=page_number,
            text=text,
            extraction_method="pymupdf",
            quality_score=quality_score,
            metadata={"fallback_used": False},
        )

    async def _extract_with_vision(
        self,
        pdf_data: bytes,
        page_number: int,
    ) -> str:
        """Extract text from a page using Gemini Vision.

        Args:
            pdf_data: Full PDF data.
            page_number: 1-indexed page number to extract.

        Returns:
            Extracted text from the page.
        """
        gemini = get_gemini_service()
        return await gemini.extract_pdf_page(pdf_data, page_number)

    def _assess_quality(self, text: str) -> float:
        """Assess the quality of extracted text.

        Args:
            text: Extracted text content.

        Returns:
            Quality score between 0 and 1.
        """
        if not text or not text.strip():
            return 0.0

        # Calculate various quality metrics
        char_count = len(text)
        word_count = len(text.split())

        # Very short text is low quality
        if char_count < 50:
            return 0.2

        # Check for garbled characters (high ratio of non-alphanumeric)
        alnum_count = sum(1 for c in text if c.isalnum() or c.isspace())
        alnum_ratio = alnum_count / char_count if char_count > 0 else 0

        # Check for proper word structure
        avg_word_len = char_count / word_count if word_count > 0 else 0
        word_structure_score = 1.0 if 3 < avg_word_len < 15 else 0.5

        # Check for sentence structure
        sentence_pattern = re.compile(r"[.!?]\s+[A-Z]")
        sentences = len(sentence_pattern.findall(text))
        sentence_score = min(sentences / 10, 1.0)

        # Combine scores
        quality = (
            alnum_ratio * 0.4
            + word_structure_score * 0.3
            + sentence_score * 0.3
        )

        return min(max(quality, 0.0), 1.0)

    async def chunk_document(
        self,
        document: ExtractedDocument,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[TextChunk]:
        """Chunk a document for RAG.

        Args:
            document: Extracted document to chunk.
            chunk_size: Characters per chunk (default from settings).
            chunk_overlap: Overlap between chunks (default from settings).

        Returns:
            List of TextChunk objects.
        """
        size = chunk_size or self._chunk_size
        overlap = chunk_overlap or self._chunk_overlap

        chunks: list[TextChunk] = []
        chunk_index = 0

        for page in document.pages:
            if not page.text.strip():
                continue

            # Split page text into chunks
            text = page.text
            start = 0

            while start < len(text):
                # Find end of chunk
                end = start + size

                # Try to break at a sentence or word boundary
                if end < len(text):
                    # Look for sentence boundary
                    sentence_break = text.rfind(". ", start, end + 50)
                    if sentence_break > start + size // 2:
                        end = sentence_break + 1
                    else:
                        # Fall back to word boundary
                        word_break = text.rfind(" ", start, end + 20)
                        if word_break > start + size // 2:
                            end = word_break

                chunk_text = text[start:end].strip()

                if chunk_text:
                    # Estimate token count (rough approximation)
                    token_count = len(chunk_text.split()) * 1.3

                    chunks.append(
                        TextChunk(
                            content=chunk_text,
                            page_number=page.page_number,
                            chunk_index=chunk_index,
                            token_count=int(token_count),
                            metadata={
                                "extraction_method": page.extraction_method,
                                "quality_score": page.quality_score,
                            },
                        )
                    )
                    chunk_index += 1

                # Move start with overlap
                start = end - overlap if end < len(text) else end

        logger.debug(
            "Chunked document",
            filename=document.filename,
            pages=document.page_count,
            chunks=len(chunks),
        )

        return chunks

    async def chunk_text(
        self,
        text: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[TextChunk]:
        """Chunk raw text for RAG.

        Args:
            text: Text content to chunk.
            chunk_size: Characters per chunk (default from settings).
            chunk_overlap: Overlap between chunks (default from settings).

        Returns:
            List of TextChunk objects.
        """
        # Create a simple document wrapper
        doc = ExtractedDocument(
            filename="text",
            page_count=1,
            pages=[
                ExtractedPage(
                    page_number=1,
                    text=text,
                    extraction_method="raw",
                    quality_score=1.0,
                    metadata={},
                )
            ],
            full_text=text,
            metadata={},
        )

        return await self.chunk_document(doc, chunk_size, chunk_overlap)


# Singleton instance
_pdf_service: PDFService | None = None


def get_pdf_service() -> PDFService:
    """Get or create the PDF service singleton."""
    global _pdf_service
    if _pdf_service is None:
        _pdf_service = PDFService()
    return _pdf_service
