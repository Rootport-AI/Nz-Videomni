"""Route handler for POST /api/retake."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import RetakeRequest, RetakeResponse
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api", tags=["retake"])


@router.post("/retake", response_model=RetakeResponse)
def route_retake(req: RetakeRequest, handler: AppHandler = Depends(get_state_service)) -> RetakeResponse:
    return handler.retake.run(req)
