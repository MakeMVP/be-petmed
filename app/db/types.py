"""DynamoDB type utilities."""

from decimal import Decimal
from typing import Any


def sanitize_floats(value: Any) -> Any:
    """Recursively convert float → Decimal (DynamoDB rejects floats)."""
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: sanitize_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_floats(v) for v in value]
    return value
