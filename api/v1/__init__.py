from fastapi import APIRouter

from api.v1.routes import inputs, sessions, system

api_router = APIRouter()
api_router.include_router(inputs.router)
api_router.include_router(sessions.router)
api_router.include_router(system.router)

__all__ = ["api_router"]
