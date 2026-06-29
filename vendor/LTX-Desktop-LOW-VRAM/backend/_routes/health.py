"""Route handlers for /health and /api/gpu-info."""

from __future__ import annotations

import os
import signal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from api_types import GpuInfoResponse, HealthResponse
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def route_health(handler: AppHandler = Depends(get_state_service)) -> HealthResponse:
    return handler.health.get_health()


@router.get("/api/gpu-info", response_model=GpuInfoResponse)
def route_gpu_info(handler: AppHandler = Depends(get_state_service)) -> GpuInfoResponse:
    return handler.health.get_gpu_info()


@router.post("/api/system/clear-vram")
def route_clear_vram(request: Request, handler: AppHandler = Depends(get_state_service)) -> dict:
    """POST /api/system/clear-vram — flush PyTorch CUDA cache and run GC.

    Returns VRAM stats before and after so the caller can show the delta.
    """
    import torch

    client_host = request.client.host if request.client else None
    if client_host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=403, detail="Forbidden")

    before_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    before_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0

    handler.gpu_cleaner.cleanup()

    after_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    after_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0

    return {
        "before_allocated_mb": round(before_allocated / 1024 / 1024, 1),
        "before_reserved_mb": round(before_reserved / 1024 / 1024, 1),
        "after_allocated_mb": round(after_allocated / 1024 / 1024, 1),
        "after_reserved_mb": round(after_reserved / 1024 / 1024, 1),
        "freed_mb": round((before_reserved - after_reserved) / 1024 / 1024, 1),
    }


def _shutdown_process() -> None:
    os.kill(os.getpid(), signal.SIGTERM)


@router.post("/api/system/shutdown")
def route_shutdown(background_tasks: BackgroundTasks, request: Request) -> dict[str, str]:
    client_host = request.client.host if request.client else None
    if client_host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=403, detail="Forbidden")

    background_tasks.add_task(_shutdown_process)
    return {"status": "shutting_down"}
