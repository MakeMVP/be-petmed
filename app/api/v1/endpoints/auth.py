"""Authentication endpoints."""

import boto3
from botocore.exceptions import ClientError
from fastapi import APIRouter

from app.api.v1.schemas.auth import TokenRefreshRequest, TokenResponse, UserResponse
from app.api.v1.schemas.common import ERROR_RESPONSES
from app.config import settings
from app.core.cognito import CurrentUser
from app.core.exceptions import BadRequestError, UnauthorizedError
from app.core.logging import get_logger
from app.db.dynamodb import get_dynamodb_client
from app.models.entities import User

router = APIRouter(prefix="/auth", tags=["Authentication"])
logger = get_logger(__name__)


@router.post(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="Get the currently authenticated user's profile from JWT. Creates user in DB if first login.",
    responses={
        401: ERROR_RESPONSES[401],
    },
)
async def get_current_user_profile(current_user: CurrentUser) -> UserResponse:
    """Get or create the current user's profile."""
    db = get_dynamodb_client()

    # Try to get existing user
    pk = f"USER#{current_user.user_id}"
    existing = await db.get_item(pk=pk, sk=pk)

    if existing:
        return UserResponse(
            user_id=existing["user_id"],
            email=existing["email"],
            email_verified=current_user.email_verified,
            name=existing.get("name"),
            avatar_url=existing.get("avatar_url"),
            created_at=existing.get("created_at"),
        )

    # First login - create user
    user = User(
        user_id=current_user.user_id,
        email=current_user.email,
    )

    await db.put_item(user.to_dynamodb_item())
    logger.info("Created new user", user_id=current_user.user_id)

    return UserResponse(
        user_id=user.user_id,
        email=user.email,
        email_verified=current_user.email_verified,
        name=user.name,
        avatar_url=user.avatar_url,
        created_at=user.created_at,
    )


@router.post(
    "/token/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
    description="Exchange a refresh token for new access and ID tokens.",
    responses={
        400: ERROR_RESPONSES[400],
        401: ERROR_RESPONSES[401],
    },
)
async def refresh_token(request: TokenRefreshRequest) -> TokenResponse:
    """Refresh access token using Cognito refresh token."""
    try:
        client = boto3.client("cognito-idp", region_name=settings.aws_region)

        response = client.initiate_auth(
            ClientId=settings.cognito_client_id,
            AuthFlow="REFRESH_TOKEN_AUTH",
            AuthParameters={
                "REFRESH_TOKEN": request.refresh_token,
            },
        )

        auth_result = response.get("AuthenticationResult", {})

        if not auth_result.get("AccessToken"):
            raise UnauthorizedError("Failed to refresh token")

        return TokenResponse(
            access_token=auth_result["AccessToken"],
            id_token=auth_result["IdToken"],
            expires_in=auth_result.get("ExpiresIn", 3600),
        )

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        error_message = e.response.get("Error", {}).get("Message", "")

        logger.warning(
            "Token refresh failed",
            error_code=error_code,
            error_message=error_message,
        )

        if error_code in ("NotAuthorizedException", "InvalidParameterException"):
            raise UnauthorizedError("Invalid or expired refresh token") from e

        raise BadRequestError(f"Token refresh failed: {error_message}") from e
