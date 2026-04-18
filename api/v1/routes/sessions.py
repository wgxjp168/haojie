"""Session management endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_current_user, get_sessions
from entities.user import User
from managers.session_manager import SessionManager

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/{session_id}", summary="Get a session")
async def get_session(
    session_id: str,
    user: User = Depends(get_current_user),
    sessions: SessionManager = Depends(get_sessions),
):
    session = await sessions.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "session not found"},
        )
    if session.user_id != user.user_id and user.user_id != "anonymous":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "FORBIDDEN", "message": "session belongs to another user"},
        )
    return session.to_dict()


@router.delete("/{session_id}", summary="End a session")
async def end_session(
    session_id: str,
    user: User = Depends(get_current_user),
    sessions: SessionManager = Depends(get_sessions),
):
    session = await sessions.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "session not found"},
        )
    if session.user_id != user.user_id and user.user_id != "anonymous":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "FORBIDDEN", "message": "session belongs to another user"},
        )
    await sessions.end_session(session_id, reason="user_end")
    return {"status": "ended", "session_id": session_id}


@router.get("/", summary="List current user's active sessions")
async def list_sessions(
    user: User = Depends(get_current_user),
    sessions: SessionManager = Depends(get_sessions),
):
    result = await sessions.list_user_sessions(user.user_id)
    return {"count": len(result), "sessions": [s.to_dict() for s in result]}
