"""AWS client factories with automatic credential resolution.

When running in Lambda (AWS_SESSION_TOKEN is set), credentials come from the
IAM execution role. Otherwise, explicit credentials from settings are used.
"""

import os

import aioboto3
import boto3

from app.config import settings


def _use_explicit_credentials() -> bool:
    """Use explicit credentials only when not in an AWS execution environment."""
    return bool(settings.aws_access_key_id and not os.environ.get("AWS_SESSION_TOKEN"))


def get_session_kwargs() -> dict:
    """Build kwargs for aioboto3.Session / boto3 clients."""
    kwargs: dict = {"region_name": settings.aws_region}
    if _use_explicit_credentials():
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    return kwargs


def get_aioboto3_session() -> aioboto3.Session:
    """Create an aioboto3 session with resolved credentials."""
    return aioboto3.Session(**get_session_kwargs())


def get_boto3_client(service: str, **extra):
    """Create a boto3 client with resolved credentials."""
    return boto3.client(service, **get_session_kwargs(), **extra)
