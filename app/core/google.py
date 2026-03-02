"""Google Gemini client factory.

Uses API key auth when GEMINI_API_KEY is set, otherwise falls back to
Vertex AI with GCP project credentials.
"""

from google import genai

from app.config import settings


def get_genai_client() -> genai.Client:
    """Create a Gemini client with resolved authentication."""
    if settings.gemini_api_key:
        return genai.Client(api_key=settings.gemini_api_key)
    return genai.Client(
        vertexai=True,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
    )
