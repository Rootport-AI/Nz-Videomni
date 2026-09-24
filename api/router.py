"""Aggregate API router mounted at /api/v1 (spec 5.2)."""

from __future__ import annotations

from fastapi import APIRouter

from api import (
    generate,
    generate_chain,
    jobs,
    loras,
    media_info,
    models_registry,
    pipeline,
    status,
    tracking,
    uploads,
)

api_router = APIRouter()
api_router.include_router(status.router, tags=["status"])
api_router.include_router(models_registry.router, tags=["models"])
api_router.include_router(loras.router, tags=["loras"])
api_router.include_router(pipeline.router, tags=["pipeline"])
api_router.include_router(uploads.router, tags=["upload"])
api_router.include_router(generate.router, tags=["generate"])
api_router.include_router(generate_chain.router, tags=["generate"])
api_router.include_router(jobs.router, tags=["jobs"])
# Object tracking (§3-54). Its own tag and its own namespace: it creates no job
# and shares nothing with the generation endpoints above.
api_router.include_router(tracking.router, tags=["tracking"])
# mp4 recipe read-out (台帳 §3-164). Loopback-only; reads a local file's
# container ``comment`` tag via ffprobe.
api_router.include_router(media_info.router, tags=["utils"])
