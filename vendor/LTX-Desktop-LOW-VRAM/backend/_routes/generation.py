"""Route handlers for /api/generate, /api/generate/cancel, /api/generation/progress."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import (
    CancelResponse,
    GenerateVideoRequest,
    GenerateVideoResponse,
    GenerationProgressResponse,
)
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api", tags=["generation"])


@router.post("/generate", response_model=GenerateVideoResponse)
def route_generate(
    req: GenerateVideoRequest,
    handler: AppHandler = Depends(get_state_service),
) -> GenerateVideoResponse:
    """POST /api/generate — video generation from JSON body."""
    return handler.video_generation.generate(req)


@router.post("/generate/cancel", response_model=CancelResponse)
def route_generate_cancel(handler: AppHandler = Depends(get_state_service)) -> CancelResponse:
    """POST /api/generate/cancel."""
    return handler.generation.cancel_generation()


@router.get("/generation/progress", response_model=GenerationProgressResponse)
def route_generation_progress(handler: AppHandler = Depends(get_state_service)) -> GenerationProgressResponse:
    """GET /api/generation/progress."""
    return handler.generation.get_generation_progress()
