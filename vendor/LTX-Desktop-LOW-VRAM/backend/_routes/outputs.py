"""Route handlers for /api/outputs — list generated videos with metadata."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api", tags=["outputs"])


class OutputEntry(BaseModel):
    filename: str
    path: str
    size_bytes: int
    modified_at: float  # Unix timestamp
    # Sidecar metadata (None if no .json sidecar exists)
    prompt: str | None = None
    negative_prompt: str | None = None
    model: str | None = None
    resolution: str | None = None
    width: int | None = None
    height: int | None = None
    num_frames: int | None = None
    duration_seconds: int | None = None
    fps: int | None = None
    seed: int | None = None
    aspect_ratio: str | None = None
    camera_motion: str | None = None
    timestamp: str | None = None
    loras: list[dict] | None = None
    render_time_seconds: float | None = None


class OutputsResponse(BaseModel):
    entries: list[OutputEntry]
    total: int
    page: int
    page_size: int


@router.get("/outputs", response_model=OutputsResponse)
def route_list_outputs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=24, ge=1, le=100),
    handler: AppHandler = Depends(get_state_service),
) -> OutputsResponse:
    """GET /api/outputs — list generated .mp4 files, newest first."""
    outputs_dir: Path = handler.config.outputs_dir

    mp4_files = sorted(
        [f for f in outputs_dir.glob("*.mp4") if f.is_file()],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    total = len(mp4_files)
    start = (page - 1) * page_size
    page_files = mp4_files[start : start + page_size]

    entries: list[OutputEntry] = []
    for mp4 in page_files:
        stat = mp4.stat()
        entry = OutputEntry(
            filename=mp4.name,
            path=str(mp4),
            size_bytes=stat.st_size,
            modified_at=stat.st_mtime,
        )

        sidecar = mp4.with_suffix(".json")
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
                entry.prompt = meta.get("prompt")
                entry.negative_prompt = meta.get("negative_prompt")
                entry.model = meta.get("model")
                entry.resolution = meta.get("resolution")
                entry.width = meta.get("width")
                entry.height = meta.get("height")
                entry.num_frames = meta.get("num_frames")
                entry.duration_seconds = meta.get("duration_seconds")
                entry.fps = meta.get("fps")
                entry.seed = meta.get("seed")
                entry.aspect_ratio = meta.get("aspect_ratio")
                entry.camera_motion = meta.get("camera_motion")
                entry.timestamp = meta.get("timestamp")
                entry.loras = meta.get("loras") or None
                entry.render_time_seconds = meta.get("render_time_seconds")
            except Exception:
                pass

        entries.append(entry)

    return OutputsResponse(entries=entries, total=total, page=page, page_size=page_size)
