"""AWS Cognito JWT validation and user authentication."""

import time
from typing import Annotated, Any

import httpx
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwk, jwt
from pydantic import BaseModel

from app.config import settings
from app.core.exceptions import ForbiddenError, UnauthorizedError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)


class CognitoUser(BaseModel):
    """Authenticated user from Cognito JWT."""

    user_id: str
    email: str
    email_verified: bool = False
    token_use: str
    auth_time: int
    exp: int
    iat: int
    raw_claims: dict[str, Any]


class JWKSCache:
    """Cache for Cognito JWKS keys with automatic refresh."""

    def __init__(self, ttl_seconds: int = 3600):
        self._keys: dict[str, Any] = {}
        self._last_fetch: float = 0
        self._ttl = ttl_seconds

    async def get_key(self, kid: str) -> dict[str, Any] | None:
        """Get a signing key by key ID, refreshing cache if needed."""
        if self._should_refresh():
            await self._refresh_keys()

        return self._keys.get(kid)

    def _should_refresh(self) -> bool:
        """Check if cache should be refreshed."""
        return time.time() - self._last_fetch > self._ttl or not self._keys

    async def _refresh_keys(self) -> None:
        """Fetch JWKS from Cognito."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(settings.cognito_jwks_url, timeout=10.0)
                response.raise_for_status()
                jwks_data = response.json()

            self._keys = {key["kid"]: key for key in jwks_data.get("keys", [])}
            self._last_fetch = time.time()
            logger.debug("Refreshed JWKS cache", key_count=len(self._keys))
        except Exception as e:
            logger.error("Failed to refresh JWKS", error=str(e))
            if not self._keys:
                raise UnauthorizedError("Unable to validate authentication") from e


# Global JWKS cache
_jwks_cache = JWKSCache()


async def decode_and_validate_token(token: str) -> dict[str, Any]:
    """Decode and validate a Cognito JWT token.

    Args:
        token: The JWT token to validate.

    Returns:
        The decoded token claims.

    Raises:
        UnauthorizedError: If token is invalid or expired.
    """
    try:
        # Decode header to get key ID
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        if not kid:
            raise UnauthorizedError("Invalid token: missing key ID")

        # Get the signing key
        key_data = await _jwks_cache.get_key(kid)
        if not key_data:
            # Try refreshing once more in case key was rotated
            await _jwks_cache._refresh_keys()
            key_data = await _jwks_cache.get_key(kid)

        if not key_data:
            raise UnauthorizedError("Invalid token: unknown signing key")

        # Construct the public key
        public_key = jwk.construct(key_data)

        # Decode and validate the token
        claims = jwt.decode(
            token,
            public_key.to_pem().decode("utf-8"),
            algorithms=["RS256"],
            audience=settings.cognito_client_id,
            issuer=settings.cognito_issuer,
            options={
                "verify_aud": True,
                "verify_iss": True,
                "verify_exp": True,
            },
        )

        # Validate token_use claim
        token_use = claims.get("token_use")
        if token_use not in ("id", "access"):
            raise UnauthorizedError("Invalid token: wrong token type")

        return claims

    except JWTError as e:
        logger.warning("JWT validation failed", error=str(e))
        raise UnauthorizedError(f"Invalid token: {e}") from e
    except Exception as e:
        if isinstance(e, UnauthorizedError):
            raise
        logger.error("Unexpected error during token validation", error=str(e))
        raise UnauthorizedError("Unable to validate authentication") from e


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> CognitoUser:
    """Extract and validate the current user from the request.

    Args:
        request: The FastAPI request object.
        credentials: The bearer token credentials.

    Returns:
        The authenticated CognitoUser.

    Raises:
        UnauthorizedError: If no token or invalid token.
    """
    if not credentials:
        raise UnauthorizedError("Authentication required")

    token = credentials.credentials
    claims = await decode_and_validate_token(token)

    # Extract user info - Cognito ID tokens have 'sub' and 'email'
    user_id = claims.get("sub")
    if not user_id:
        raise UnauthorizedError("Invalid token: missing user ID")

    # For access tokens, email might not be present
    email = claims.get("email", "")

    return CognitoUser(
        user_id=user_id,
        email=email,
        email_verified=claims.get("email_verified", False),
        token_use=claims.get("token_use", ""),
        auth_time=claims.get("auth_time", 0),
        exp=claims.get("exp", 0),
        iat=claims.get("iat", 0),
        raw_claims=claims,
    )


async def get_optional_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> CognitoUser | None:
    """Get current user if authenticated, None otherwise.

    Args:
        request: The FastAPI request object.
        credentials: The bearer token credentials.

    Returns:
        The authenticated CognitoUser or None.
    """
    if not credentials:
        return None

    try:
        return await get_current_user(request, credentials)
    except UnauthorizedError:
        return None


# Type aliases for dependency injection
CurrentUser = Annotated[CognitoUser, Depends(get_current_user)]
OptionalUser = Annotated[CognitoUser | None, Depends(get_optional_user)]


def require_verified_email(user: CurrentUser) -> CognitoUser:
    """Require that the user's email is verified.

    Args:
        user: The authenticated user.

    Returns:
        The user if email is verified.

    Raises:
        ForbiddenError: If email is not verified.
    """
    if not user.email_verified:
        raise ForbiddenError("Email verification required")
    return user


VerifiedUser = Annotated[CognitoUser, Depends(require_verified_email)]
