"""System-level endpoints: version, stats."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_manager, get_sessions
from config.settings import get_settings
from managers.input_manager import MultiModalInputManager
from managers.session_manager import SessionManager

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/version", summary="Service version")
async def version():
    settings = get_settings()
    return {
        "service": settings.service_name,
        "version": settings.version,
        "env": settings.env.value,
    }


@router.get("/stats", summary="Runtime stats")
async def stats(
    manager: MultiModalInputManager = Depends(get_manager),
    sessions: SessionManager = Depends(get_sessions),
):
    return {
        "active_sessions": await sessions.active_count(),
        "processors": list(p.value for p in manager.processors),
    }
