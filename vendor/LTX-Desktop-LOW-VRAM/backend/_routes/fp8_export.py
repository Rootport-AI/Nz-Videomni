"""Routes for /api/models/export-fp8."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api/models", tags=["fp8-export"])


class Fp8ExportStatusResponse(BaseModel):
    status: str          # idle | running | done | error
    progress: float      # 0.0 – 1.0
    error: str | None
    file_exists: bool
    file_path: str


class Fp8ExportStartResponse(BaseModel):
    started: bool
    message: str


@router.post("/export-fp8", response_model=Fp8ExportStartResponse)
def route_export_fp8_start(
    handler: AppHandler = Depends(get_state_service),
) -> Fp8ExportStartResponse:
    """POST /api/models/export-fp8 — begin the one-time fp8 export (background)."""
    started = handler.fp8_export.start_export()
    return Fp8ExportStartResponse(
        started=started,
        message="Export started" if started else "Export already running",
    )


@router.get("/export-fp8/status", response_model=Fp8ExportStatusResponse)
def route_export_fp8_status(
    handler: AppHandler = Depends(get_state_service),
) -> Fp8ExportStatusResponse:
    """GET /api/models/export-fp8/status — poll export progress."""
    state = handler.fp8_export.get_status()
    return Fp8ExportStatusResponse(
        status=state.status.value,
        progress=state.progress,
        error=state.error,
        file_exists=handler.fp8_export.fp8_file_exists(),
        file_path=handler.fp8_export.fp8_file_path(),
    )
