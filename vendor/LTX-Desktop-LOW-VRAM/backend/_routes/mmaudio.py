"""Route handlers for /api/mmaudio/* endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import MMAudioGenerateRequest, MMAudioProgressResponse
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api/mmaudio", tags=["mmaudio"])


@router.post("/generate", response_model=MMAudioProgressResponse)
def route_mmaudio_generate(
    req: MMAudioGenerateRequest,
    handler: AppHandler = Depends(get_state_service),
) -> MMAudioProgressResponse:
    return handler.mmaudio.generate(req)


@router.get("/progress", response_model=MMAudioProgressResponse)
def route_mmaudio_progress(
    handler: AppHandler = Depends(get_state_service),
) -> MMAudioProgressResponse:
    return handler.mmaudio.get_progress()


@router.post("/cancel")
def route_mmaudio_cancel(
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, str]:
    handler.mmaudio.cancel()
    return {"status": "cancelled"}
