"""Route handlers for /api/magi/* endpoints (daVinci-MagiHuman)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import MagiGenerateRequest, MagiProgressResponse
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api/magi", tags=["magi"])


@router.post("/generate", response_model=MagiProgressResponse)
def route_magi_generate(
    req: MagiGenerateRequest,
    handler: AppHandler = Depends(get_state_service),
) -> MagiProgressResponse:
    return handler.magi.generate(req)


@router.get("/progress", response_model=MagiProgressResponse)
def route_magi_progress(
    handler: AppHandler = Depends(get_state_service),
) -> MagiProgressResponse:
    return handler.magi.get_progress()


@router.post("/cancel")
def route_magi_cancel(
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, str]:
    handler.magi.cancel()
    return {"status": "cancelled"}
