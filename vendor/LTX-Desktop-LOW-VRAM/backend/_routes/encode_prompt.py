"""Route handlers for /api/encode-prompt."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api", tags=["encode-prompt"])


class EncodePromptRequest(BaseModel):
    prompt: str


class EncodePromptResponse(BaseModel):
    status: str
    encoded_prompt: str


class EnhancePromptRequest(BaseModel):
    prompt: str


class EnhancePromptResponse(BaseModel):
    status: str
    enhanced_prompt: str


class EncodePromptStatusResponse(BaseModel):
    single_gpu_local_mode: bool
    encoded_prompt: str | None


@router.post("/encode-prompt", response_model=EncodePromptResponse)
def route_encode_prompt(
    req: EncodePromptRequest,
    handler: AppHandler = Depends(get_state_service),
) -> EncodePromptResponse:
    """POST /api/encode-prompt — GPU-encode the prompt; ejects video model first."""
    handler.encode_prompt.encode_prompt(req.prompt)
    return EncodePromptResponse(status="ok", encoded_prompt=req.prompt.strip())


@router.post("/enhance-prompt", response_model=EnhancePromptResponse)
def route_enhance_prompt(
    req: EnhancePromptRequest,
    handler: AppHandler = Depends(get_state_service),
) -> EnhancePromptResponse:
    """POST /api/enhance-prompt — run Gemma enhance_t2v on the prompt text only."""
    enhanced = handler.encode_prompt.enhance_prompt(req.prompt)
    return EnhancePromptResponse(status="ok", enhanced_prompt=enhanced)


@router.get("/encode-prompt/status", response_model=EncodePromptStatusResponse)
def route_encode_prompt_status(
    handler: AppHandler = Depends(get_state_service),
) -> EncodePromptStatusResponse:
    """GET /api/encode-prompt/status — returns whether manual encoding is relevant."""
    return EncodePromptStatusResponse(
        single_gpu_local_mode=handler.encode_prompt.is_single_gpu_local_mode(),
        encoded_prompt=handler.encode_prompt.get_encoded_prompt(),
    )
