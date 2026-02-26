"""Admin authentication guard."""

from typing import Annotated

from fastapi import Depends

from app.core.cognito import CognitoUser, CurrentUser
from app.core.exceptions import ForbiddenError, NotFoundError
from app.core.logging import get_logger
from app.db.dynamodb import get_dynamodb_client

logger = get_logger(__name__)


async def require_admin(current_user: CurrentUser) -> CognitoUser:
    """Require that the authenticated user has admin role.

    Checks the user's settings.role field in DynamoDB.

    Args:
        current_user: The authenticated Cognito user.

    Returns:
        The user if they are an admin.

    Raises:
        ForbiddenError: If user is not an admin.
        NotFoundError: If user record not found.
    """
    db = get_dynamodb_client()
    pk = f"USER#{current_user.user_id}"
    user = await db.get_item(pk=pk, sk=pk)

    if not user:
        raise NotFoundError(
            "User not found",
            resource_type="user",
            resource_id=current_user.user_id,
        )

    role = user.get("settings", {}).get("role", "user")
    if role != "admin":
        logger.warning(
            "Non-admin user attempted admin access",
            user_id=current_user.user_id,
            role=role,
        )
        raise ForbiddenError("Admin access required")

    return current_user


AdminUser = Annotated[CognitoUser, Depends(require_admin)]
