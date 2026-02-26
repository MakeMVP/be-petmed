"""API v1 router aggregator."""

from fastapi import APIRouter

from app.api.v1.endpoints import admin, auth, conversations, documents, health, queries, users

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(documents.router)
api_router.include_router(queries.router)
api_router.include_router(conversations.router)
api_router.include_router(admin.router)
