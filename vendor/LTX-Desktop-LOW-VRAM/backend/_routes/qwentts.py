"""Route handlers for /api/qwentts/* endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import QwenTTSGenerateRequest, QwenTTSProgressResponse
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api/qwentts", tags=["qwentts"])


@router.post("/generate", response_model=QwenTTSProgressResponse)
def route_qwentts_generate(
    req: QwenTTSGenerateRequest,
    handler: AppHandler = Depends(get_state_service),
) -> QwenTTSProgressResponse:
    return handler.qwentts.generate(req)


@router.get("/progress", response_model=QwenTTSProgressResponse)
def route_qwentts_progress(
    handler: AppHandler = Depends(get_state_service),
) -> QwenTTSProgressResponse:
    return handler.qwentts.get_progress()


@router.post("/cancel")
def route_qwentts_cancel(
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, str]:
    handler.qwentts.cancel()
    return {"status": "cancelled"}


@router.post("/unload")
def route_qwentts_unload(
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, str]:
    handler.qwentts.unload()
    return {"status": "unloaded"}


@router.get("/voices")
def route_qwentts_voices(
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, object]:
    return {
        "speakers": handler.qwentts.get_speakers(),
        "languages": handler.qwentts.get_languages(),
    }
