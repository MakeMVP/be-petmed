"""Pagination cursor utilities for DynamoDB."""

import base64
import json
from typing import Any

from app.core.exceptions import BadRequestError


def decode_cursor(cursor: str | None) -> dict[str, Any] | None:
    """Decode a base64-encoded pagination cursor into a DynamoDB ExclusiveStartKey.

    Args:
        cursor: Base64-encoded cursor string, or None.

    Returns:
        Decoded DynamoDB ExclusiveStartKey dict, or None.

    Raises:
        BadRequestError: If the cursor is malformed.
    """
    if not cursor:
        return None
    try:
        return json.loads(base64.b64decode(cursor))
    except Exception as e:
        raise BadRequestError("Invalid cursor") from e


def encode_cursor(last_key: dict[str, Any] | None) -> str | None:
    """Encode a DynamoDB LastEvaluatedKey into a base64 pagination cursor.

    The presence of a last_key from DynamoDB IS the has_more signal.

    Args:
        last_key: DynamoDB LastEvaluatedKey, or None.

    Returns:
        Base64-encoded cursor string, or None if no more results.
    """
    if not last_key:
        return None
    return base64.b64encode(json.dumps(last_key).encode()).decode()
