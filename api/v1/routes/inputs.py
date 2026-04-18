"""Input processing endpoints."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from api.dependencies import get_current_user, get_manager
from config.settings import get_settings
from core.exceptions import InputProcessorError
from core.logging import get_logger
from entities.enums import InputType, PriorityLevel, ProcessingMode
from entities.user import User
from managers.input_manager import (
    InputRequest,
    MultiModalInputManager,
    ProcessingConfig,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/inputs", tags=["inputs"])


# --- request/response models ---


class TextInputBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    language: Optional[str] = Field(default=None, max_length=16)
    session_id: Optional[str] = Field(default=None, max_length=64)


class LinkInputBody(BaseModel):
    url: str = Field(..., min_length=1, max_length=2_048)
    session_id: Optional[str] = Field(default=None, max_length=64)


class BatchItem(BaseModel):
    type: InputType
    data: Union[str, Dict[str, Any]]


class BatchInputBody(BaseModel):
    inputs: List[BatchItem] = Field(..., min_length=1, max_length=100)
    session_id: Optional[str] = None


class ProcessResponse(BaseModel):
    success: bool
    input_id: str
    session_id: str
    input_type: str
    started_at: str
    completed_at: str
    processing_time_ms: float
    data: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    cached: bool = False
    storage_key: Optional[str] = None
    summary: Optional[str] = None


def _handle_error(exc: InputProcessorError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.to_dict())


# --- helper ---


async def _run(
    manager: MultiModalInputManager,
    user: User,
    input_type: InputType,
    data: Any,
    session_id: Optional[str],
    language: Optional[str] = None,
) -> Dict[str, Any]:
    req = InputRequest(
        input_data=data,
        input_type=input_type,
        user_id=user.user_id,
        session_id=session_id,
        config=ProcessingConfig(mode=ProcessingMode.SYNC, language=language),
    )
    try:
        result = await manager.process(req)
    except InputProcessorError as e:
        raise _handle_error(e)
    return result.to_dict()


# --- endpoints ---


@router.post("/text", response_model=ProcessResponse, summary="Process text input")
async def process_text(
    body: TextInputBody,
    user: User = Depends(get_current_user),
    manager: MultiModalInputManager = Depends(get_manager),
):
    return await _run(
        manager, user, InputType.TEXT, body.text, body.session_id, body.language
    )


@router.post("/link", response_model=ProcessResponse, summary="Process link input")
async def process_link(
    body: LinkInputBody,
    user: User = Depends(get_current_user),
    manager: MultiModalInputManager = Depends(get_manager),
):
    return await _run(manager, user, InputType.LINK, body.url, body.session_id)


@router.post("/image", response_model=ProcessResponse, summary="Process image upload")
async def process_image(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
    user: User = Depends(get_current_user),
    manager: MultiModalInputManager = Depends(get_manager),
):
    settings = get_settings()
    max_bytes = settings.image_max_size_mb * 1024 * 1024
    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "INPUT_TOO_LARGE",
                "message": f"image exceeds {settings.image_max_size_mb} MiB",
            },
        )
    return await _run(manager, user, InputType.IMAGE, data, session_id)


@router.post("/voice", response_model=ProcessResponse, summary="Process voice upload")
async def process_voice(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
    language: Optional[str] = Form(default=None),
    user: User = Depends(get_current_user),
    manager: MultiModalInputManager = Depends(get_manager),
):
    settings = get_settings()
    max_bytes = settings.voice_max_size_mb * 1024 * 1024
    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "INPUT_TOO_LARGE",
                "message": f"audio exceeds {settings.voice_max_size_mb} MiB",
            },
        )
    return await _run(
        manager, user, InputType.VOICE, data, session_id, language=language
    )


@router.post("/batch", summary="Batch-process mixed inputs")
async def process_batch(
    body: BatchInputBody,
    user: User = Depends(get_current_user),
    manager: MultiModalInputManager = Depends(get_manager),
):
    requests: List[InputRequest] = []
    for item in body.inputs:
        data = item.data
        if item.type in (InputType.IMAGE, InputType.VOICE):
            # For batch image/voice accept only base64-encoded strings or raw bytes.
            if isinstance(data, str):
                import base64
                try:
                    data = base64.b64decode(data, validate=True)
                except Exception:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "code": "INVALID_INPUT",
                            "message": f"{item.type.value} data must be base64",
                        },
                    )
        requests.append(
            InputRequest(
                input_data=data,
                input_type=item.type,
                user_id=user.user_id,
                session_id=body.session_id,
                config=ProcessingConfig(mode=ProcessingMode.BATCH),
            )
        )
    results = await manager.process_batch(requests)
    return {
        "total": len(results),
        "successful": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "results": [r.to_dict() for r in results],
    }


@router.get("/supported-formats", summary="List supported formats")
async def supported_formats(
    manager: MultiModalInputManager = Depends(get_manager),
):
    return {
        itype.value: proc.supported_formats or []
        for itype, proc in manager.processors.items()
    }
