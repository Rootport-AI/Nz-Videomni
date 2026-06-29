"""Route handlers for /api/prismaudio/* endpoints (PrismAudio / ThinkSound)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import PrismAudioGenerateRequest, PrismAudioProgressResponse
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api/prismaudio", tags=["prismaudio"])


@router.post("/generate", response_model=PrismAudioProgressResponse)
def route_prismaudio_generate(
    req: PrismAudioGenerateRequest,
    handler: AppHandler = Depends(get_state_service),
) -> PrismAudioProgressResponse:
    return handler.prismaudio.generate(req)


@router.get("/progress", response_model=PrismAudioProgressResponse)
def route_prismaudio_progress(
    handler: AppHandler = Depends(get_state_service),
) -> PrismAudioProgressResponse:
    return handler.prismaudio.get_progress()


@router.post("/cancel")
def route_prismaudio_cancel(
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, str]:
    handler.prismaudio.cancel()
    return {"status": "cancelled"}
