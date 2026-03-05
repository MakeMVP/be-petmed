"""Google Gemini service for LLM operations."""

import asyncio
import functools
import queue
import threading
from collections.abc import AsyncIterator
from typing import Any

from google.genai import types

from app.config import settings
from app.core.exceptions import ServiceUnavailableError
from app.core.google import get_genai_client
from app.core.logging import get_logger

logger = get_logger(__name__)


class GeminiService:
    """Service for Gemini LLM operations."""

    def __init__(self) -> None:
        self._client = get_genai_client()
        self._model = settings.gemini_model
        self._flash_model = settings.gemini_flash_model

    async def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        context: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        use_flash: bool = False,
    ) -> tuple[str, dict[str, int]]:
        """Generate text response from Gemini.

        Args:
            prompt: The user prompt.
            system_instruction: Optional system instruction.
            context: Optional conversation context.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            use_flash: Use Flash model for faster/cheaper responses.

        Returns:
            Tuple of (response text, token usage dict).

        Raises:
            ServiceUnavailableError: If generation fails.
        """
        model_id = self._flash_model if use_flash else self._model

        try:
            # Build contents
            contents = []

            if context:
                for msg in context:
                    role = "user" if msg.get("role") == "user" else "model"
                    contents.append(
                        types.Content(
                            role=role,
                            parts=[types.Part.from_text(text=msg.get("content", ""))],
                        )
                    )

            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            )

            # Build config
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction,
            )

            # Generate response
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=model_id,
                contents=contents,
                config=config,
            )

            # Extract text
            text = response.text or ""

            # Extract token usage
            usage = {}
            if response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count or 0,
                    "total_tokens": response.usage_metadata.total_token_count or 0,
                }

            logger.debug(
                "Generated response",
                model=model_id,
                prompt_len=len(prompt),
                response_len=len(text),
                usage=usage,
            )

            return text, usage

        except Exception as e:
            logger.error("Gemini generation failed", error=str(e), model=model_id)
            raise ServiceUnavailableError(
                f"Failed to generate response: {e}",
                service="gemini",
            ) from e

    async def generate_stream(
        self,
        prompt: str,
        system_instruction: str | None = None,
        context: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        use_flash: bool = False,
    ) -> AsyncIterator[str]:
        """Stream text response from Gemini.

        Args:
            prompt: The user prompt.
            system_instruction: Optional system instruction.
            context: Optional conversation context.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            use_flash: Use Flash model for faster/cheaper responses.

        Yields:
            Text chunks as they are generated.

        Raises:
            ServiceUnavailableError: If generation fails.
        """
        model_id = self._flash_model if use_flash else self._model

        try:
            # Build contents
            contents = []

            if context:
                for msg in context:
                    role = "user" if msg.get("role") == "user" else "model"
                    contents.append(
                        types.Content(
                            role=role,
                            parts=[types.Part.from_text(text=msg.get("content", ""))],
                        )
                    )

            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            )

            # Build config
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction,
            )

            # Stream chunks via a thread to avoid blocking the event loop
            q: queue.Queue[str | None] = queue.Queue()
            cancel = threading.Event()

            def _produce():
                try:
                    stream = self._client.models.generate_content_stream(
                        model=model_id,
                        contents=contents,
                        config=config,
                    )
                    for chunk in stream:
                        if cancel.is_set():
                            break
                        if chunk.text:
                            q.put(chunk.text)
                except Exception as exc:
                    q.put(exc)
                finally:
                    q.put(None)  # sentinel

            thread = threading.Thread(target=_produce, daemon=True)
            thread.start()

            try:
                while True:
                    item = await asyncio.to_thread(q.get)
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                cancel.set()

        except Exception as e:
            logger.error("Gemini streaming failed", error=str(e), model=model_id)
            raise ServiceUnavailableError(
                f"Failed to generate streaming response: {e}",
                service="gemini",
            ) from e

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
        mime_type: str = "image/png",
    ) -> str:
        """Analyze an image using Gemini Vision.

        Args:
            image_data: Raw image bytes.
            prompt: Analysis prompt.
            mime_type: Image MIME type.

        Returns:
            Analysis text.

        Raises:
            ServiceUnavailableError: If analysis fails.
        """
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_data, mime_type=mime_type),
                        types.Part.from_text(text=prompt),
                    ],
                )
            ]

            config = types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=4096,
            )

            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._flash_model,  # Use Flash for vision tasks
                contents=contents,
                config=config,
            )

            return response.text or ""

        except Exception as e:
            logger.error("Gemini vision analysis failed", error=str(e))
            raise ServiceUnavailableError(
                f"Failed to analyze image: {e}",
                service="gemini",
            ) from e

    async def extract_pdf_page(
        self,
        pdf_data: bytes,
        page_number: int,
        extraction_prompt: str | None = None,
    ) -> str:
        """Extract text from a PDF page using Gemini Vision.

        Args:
            pdf_data: Raw PDF bytes.
            page_number: Page number to extract (1-indexed).
            extraction_prompt: Custom extraction prompt.

        Returns:
            Extracted text from the page.
        """
        default_prompt = """Extract all text content from this PDF page.
Preserve the structure and formatting as much as possible.
Include headers, paragraphs, lists, and tables.
For tables, use markdown table format."""

        prompt = extraction_prompt or default_prompt

        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=pdf_data, mime_type="application/pdf"),
                        types.Part.from_text(text=f"Page {page_number}: {prompt}"),
                    ],
                )
            ]

            config = types.GenerateContentConfig(
                temperature=0.1,  # Low temperature for accurate extraction
                max_output_tokens=8192,
            )

            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._flash_model,
                contents=contents,
                config=config,
            )

            return response.text or ""

        except Exception as e:
            logger.error(
                "PDF page extraction failed",
                error=str(e),
                page=page_number,
            )
            raise ServiceUnavailableError(
                f"Failed to extract PDF page: {e}",
                service="gemini",
            ) from e


@functools.lru_cache
def get_gemini_service() -> GeminiService:
    """Get or create the Gemini service singleton."""
    return GeminiService()
