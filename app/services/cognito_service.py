"""Cognito user management service."""

from botocore.exceptions import ClientError

from app.config import settings
from app.core.aws import get_aioboto3_session
from app.core.exceptions import BadRequestError
from app.core.logging import get_logger

logger = get_logger(__name__)


async def create_cognito_user(email: str, name: str | None = None) -> str:
    """Create a user in Cognito via AdminCreateUser and return the Cognito sub.

    Cognito sends an invite email with a temporary password automatically.
    """
    user_attributes = [
        {"Name": "email", "Value": email},
        {"Name": "email_verified", "Value": "true"},
    ]
    if name:
        user_attributes.append({"Name": "name", "Value": name})

    session = get_aioboto3_session()
    async with session.client("cognito-idp", region_name=settings.aws_region) as client:
        try:
            response = await client.admin_create_user(
                UserPoolId=settings.cognito_user_pool_id,
                Username=email,
                UserAttributes=user_attributes,
                DesiredDeliveryMediums=["EMAIL"],
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "UsernameExistsException":
                raise BadRequestError("A user with this email already exists") from e
            logger.error("Cognito AdminCreateUser failed", error_code=error_code, email=email)
            raise

    # Extract sub from the created user's attributes
    attributes = response["User"]["Attributes"]
    sub = next(attr["Value"] for attr in attributes if attr["Name"] == "sub")

    logger.info("Created Cognito user", email=email, user_id=sub)
    return sub


async def delete_cognito_user(email: str) -> None:
    """Delete a user from Cognito (compensating action for failed invite)."""
    session = get_aioboto3_session()
    async with session.client("cognito-idp", region_name=settings.aws_region) as client:
        try:
            await client.admin_delete_user(
                UserPoolId=settings.cognito_user_pool_id,
                Username=email,
            )
            logger.info("Rolled back Cognito user", email=email)
        except ClientError:
            logger.error("Failed to roll back Cognito user", email=email)
