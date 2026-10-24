"""Prompt token counter: ``POST /utils/prompt-tokens``.

    POST /utils/prompt-tokens
        {"prompt": str, "negative_prompt"?: str, "clips"?: [str, ...]}
    -> 200 {"prompt": int | null, "negative_prompt": int | null, "clips": [int, ...] | null}

Counts each text with the tokenizer of the ACTIVE base model (and, for LTX 2.5,
the selected text encoder) the way its engine sees it. ``prompt`` is the shared
prompt as sent (the operator panel strips the ``<lora:>`` tags first); each
``clips`` item is a chain clip's own prompt, counted after joining it to
``prompt`` by the server's rule (:func:`api.models.compose_clip_prompt`).
``negative_prompt`` / ``clips`` come back ``null`` when not sent. The limit to
compare with is the base model's ``constraints.prompt_token_limit`` in
``GET /models``.

ALWAYS 200. Counting is a courtesy that must never get in the way of
generating: when the counts cannot be had (no counter for the family, a
missing tokenizer file, the ``tokenizers`` package absent, ...) every field is
``null`` and the screen shows no counter. The failure is logged once per
distinct message, at this one boundary.

The request model lives here rather than in ``api/models.py`` (the generation
vocabulary), following ``api/media_info.py``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from api.context import AppContext
from api.deps import get_context, require_auth
from api.models import compose_clip_prompt
from services import engines
from services.model_registry import DEFAULT_NAME

logger = logging.getLogger("ltx.prompt_tokens")

router = APIRouter()

#: Failure messages already logged (each is logged once per process).
_logged_failures: set[str] = set()


class PromptTokensRequest(BaseModel):
    """Body of ``POST /utils/prompt-tokens``."""

    prompt: str
    negative_prompt: str | None = None
    clips: list[str] | None = None


class PromptTokensResponse(BaseModel):
    """Token counts; ``null`` where not asked for or not countable."""

    prompt: int | None
    negative_prompt: int | None
    clips: list[int] | None


@router.post(
    "/utils/prompt-tokens",
    response_model=PromptTokensResponse,
    dependencies=[Depends(require_auth)],
)
async def prompt_tokens(
    body: PromptTokensRequest, context: AppContext = Depends(get_context)
) -> PromptTokensResponse:
    """Count the prompts' tokens for the active base model (always 200)."""
    return await run_in_threadpool(_count, body, context)


def _count(body: PromptTokensRequest, context: AppContext) -> PromptTokensResponse:
    """Tokenizer loading and encoding, both blocking, run off the event loop."""
    texts = [body.prompt]
    if body.negative_prompt is not None:
        texts.append(body.negative_prompt)
    clips = [compose_clip_prompt(body.prompt, own) for own in body.clips or []]
    texts.extend(clips)

    registry = context.model_registry
    base_id = context.pipeline_manager.active_base_model
    text_encoder = context.pipeline_manager.active_models.get("text_encoder", DEFAULT_NAME)
    try:
        counts = engines.count_prompt_tokens(
            registry.descriptor(base_id).engine_family,
            texts,
            assets=lambda key: registry.asset_path(key, base_model=base_id),
            text_encoder=lambda: registry.resolve("text_encoder", text_encoder, base_model=base_id),
        )
    except Exception as exc:  # noqa: BLE001 - a counter failure must never be an error
        _log_failure(exc)
        counts = None
    if counts is None:
        return PromptTokensResponse(prompt=None, negative_prompt=None, clips=None)

    rest = iter(counts[1:])
    return PromptTokensResponse(
        prompt=counts[0],
        negative_prompt=next(rest) if body.negative_prompt is not None else None,
        clips=list(rest) if body.clips is not None else None,
    )


def _log_failure(exc: Exception) -> None:
    message = f"{type(exc).__name__}: {exc}"
    detail = getattr(exc, "detail", None)  # an APIError's specifics
    if detail:
        message += f" ({detail})"
    if message in _logged_failures:
        return
    _logged_failures.add(message)
    logger.warning("プロンプトのトークン数を数えられません（カウンターは出しません）: %s", message)
