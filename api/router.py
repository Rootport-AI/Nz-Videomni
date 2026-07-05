"""Aggregate API router mounted at /api/v1 (spec 5.2)."""

from __future__ import annotations

from fastapi import APIRouter

from api import generate, generate_chain, jobs, models_registry, pipeline, status, uploads

api_router = APIRouter()
api_router.include_router(status.router, tags=["status"])
api_router.include_router(models_registry.router, tags=["models"])
api_router.include_router(pipeline.router, tags=["pipeline"])
api_router.include_router(uploads.router, tags=["upload"])
api_router.include_router(generate.router, tags=["generate"])
api_router.include_router(generate_chain.router, tags=["generate"])
api_router.include_router(jobs.router, tags=["jobs"])
