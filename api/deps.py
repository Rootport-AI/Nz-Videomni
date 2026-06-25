"""FastAPI dependencies: shared context + optional bearer auth."""

from __future__ import annotations

from fastapi import Depends, Header, Request

from api.context import AppContext
from api.errors import APIError


def get_context(request: Request) -> AppContext:
    return request.app.state.context


def require_auth(
    context: AppContext = Depends(get_context),
    authorization: str | None = Header(default=None),
) -> None:
    """Enforce ``Authorization: Bearer <api-key>`` when an api_key is configured."""
    api_key = context.runtime.api_key
    if not api_key:
        return
    expected = f"Bearer {api_key}"
    if authorization != expected:
        raise APIError("UNAUTHORIZED", "Invalid or missing API key", 401)
