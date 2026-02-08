"""Health check endpoints."""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Response, status

from app.api.v1.schemas.common import BaseSchema
from app.config import settings
from app.core.logging import get_logger

router = APIRouter(prefix="/health", tags=["Health"])
logger = get_logger(__name__)


class HealthResponse(BaseSchema):
    """Basic health check response."""

    status: str
    version: str
    environment: str
    timestamp: datetime


class DependencyHealth(BaseSchema):
    """Health status of a dependency."""

    name: str
    status: str
    latency_ms: float | None = None
    error: str | None = None


class ReadinessResponse(BaseSchema):
    """Readiness check response with dependency status."""

    status: str
    version: str
    environment: str
    timestamp: datetime
    dependencies: list[DependencyHealth]


@router.get(
    "",
    response_model=HealthResponse,
    summary="Basic health check",
    description="Returns basic health status of the API. Use for liveness probes.",
)
async def health_check() -> HealthResponse:
    """Basic health check endpoint for liveness probes."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment,
        timestamp=datetime.now(UTC),
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Returns readiness status with dependency health. Use for readiness probes.",
    responses={
        503: {"description": "Service Unavailable - One or more dependencies unhealthy"},
    },
)
async def readiness_check(response: Response) -> ReadinessResponse:
    """Readiness check with dependency health verification."""
    dependencies: list[DependencyHealth] = []
    all_healthy = True

    # Check Redis
    redis_health = await _check_redis()
    dependencies.append(redis_health)
    if redis_health.status != "healthy":
        all_healthy = False

    # Check DynamoDB
    dynamodb_health = await _check_dynamodb()
    dependencies.append(dynamodb_health)
    if dynamodb_health.status != "healthy":
        all_healthy = False

    # Check Pinecone
    pinecone_health = await _check_pinecone()
    dependencies.append(pinecone_health)
    if pinecone_health.status != "healthy":
        all_healthy = False

    # Set response status code
    if not all_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="healthy" if all_healthy else "unhealthy",
        version=settings.app_version,
        environment=settings.environment,
        timestamp=datetime.now(UTC),
        dependencies=dependencies,
    )


async def _check_redis() -> DependencyHealth:
    """Check Redis connectivity."""
    import time

    try:
        import redis.asyncio as redis

        start = time.perf_counter()
        client = redis.from_url(settings.redis_url, decode_responses=True)
        await client.ping()
        await client.aclose()
        latency = (time.perf_counter() - start) * 1000

        return DependencyHealth(
            name="redis",
            status="healthy",
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        logger.warning("Redis health check failed", error=str(e))
        return DependencyHealth(
            name="redis",
            status="unhealthy",
            error=str(e),
        )


async def _check_dynamodb() -> DependencyHealth:
    """Check DynamoDB connectivity."""
    import time

    try:
        import aioboto3

        start = time.perf_counter()
        session = aioboto3.Session()

        async with session.resource(
            "dynamodb",
            region_name=settings.aws_region,
            endpoint_url=settings.dynamodb_endpoint_url,
        ) as dynamodb:
            table = await dynamodb.Table(settings.dynamodb_table_name)
            await table.table_status
            latency = (time.perf_counter() - start) * 1000

        return DependencyHealth(
            name="dynamodb",
            status="healthy",
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        logger.warning("DynamoDB health check failed", error=str(e))
        return DependencyHealth(
            name="dynamodb",
            status="unhealthy",
            error=str(e),
        )


async def _check_pinecone() -> DependencyHealth:
    """Check Pinecone connectivity."""
    import time

    try:
        from pinecone import Pinecone

        if not settings.pinecone_api_key:
            return DependencyHealth(
                name="pinecone",
                status="unconfigured",
                error="Pinecone API key not configured",
            )

        start = time.perf_counter()
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index_name)
        index.describe_index_stats()
        latency = (time.perf_counter() - start) * 1000

        return DependencyHealth(
            name="pinecone",
            status="healthy",
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        logger.warning("Pinecone health check failed", error=str(e))
        return DependencyHealth(
            name="pinecone",
            status="unhealthy",
            error=str(e),
        )
