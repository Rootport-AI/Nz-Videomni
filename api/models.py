"""Pydantic schemas — the external API contract (spec ch.6).

These are FROZEN as the final-form API for Phase 1 so that future frontends
(AviUtl2, DaVinci Resolve) and later phases do not break. Do not relax the
validators here without revisiting the spec.

Resolution note: ``width``/``height`` are the *generation* size and must be a
multiple of **64** — the two-stage distilled pipeline generates stage-1 at half
resolution then 2x-upsamples, so the requested size must be divisible by 64
(``ltx_pipelines`` ``assert_resolution(is_two_stage=True)``). This is a
deliberate tightening from the earlier 32 rule (the mock never enforced it). Any
non-64 final display size (e.g. 960x540) is obtained via ``crop_output``.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class CropOutput(BaseModel):
    width: int = Field(..., ge=32)
    height: int = Field(..., ge=32)


class ConditioningImage(BaseModel):
    image_id: str
    frame_idx: int = 0
    strength: float = Field(0.8, ge=0.0, le=1.0)
    crf: int | None = None


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = ""

    # 生成サイズ。必ず64の倍数（two-stage distilled）。最終表示サイズは crop_output で。
    width: int = Field(512, ge=256, le=4096)
    height: int = Field(320, ge=128, le=4096)

    # 最終MP4のクロップサイズ。None ならクロップしない。
    crop_output: CropOutput | None = None

    # 尺 cap は 20s(481f=8×60+1)@24fps まで許容。溢れ/低速/非実用は
    # クライアント UI 警告に委ねる（解像度別 spill-free は /config の
    # limits.spill_free_frames、実測根拠は RESOLUTION_DURATION_CAPABILITY.md §8.4/§8.6）。
    num_frames: int = Field(49, ge=9, le=481)
    frame_rate: float = Field(24.0, ge=1.0, le=60.0)
    num_inference_steps: int = Field(8, ge=1, le=100)
    guidance_scale: float = Field(1.0, ge=0.0, le=20.0)
    seed: int = -1
    pipeline: Literal["distilled", "two_stage_hq"] = "distilled"

    # 空配列なら T2V。1件なら Phase 1 最小 I2V。
    conditioning_images: list[ConditioningImage] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ltx_constraints(self) -> "GenerateRequest":
        if self.width % 64 != 0:
            raise ValueError("width must be a multiple of 64")
        if self.height % 64 != 0:
            raise ValueError("height must be a multiple of 64")
        if (self.num_frames - 1) % 8 != 0:
            raise ValueError("num_frames must be 8n+1")
        if self.crop_output is not None:
            if self.crop_output.width > self.width:
                raise ValueError("crop_output.width must be <= width")
            if self.crop_output.height > self.height:
                raise ValueError("crop_output.height must be <= height")
        if self.pipeline == "distilled":
            if self.num_inference_steps != 8:
                raise ValueError(
                    "distilled pipeline requires num_inference_steps=8 in Phase 1"
                )
            if self.guidance_scale != 1.0:
                raise ValueError(
                    "distilled pipeline requires guidance_scale=1.0 in Phase 1"
                )

        # Phase 1 最小 I2V 制約。
        if len(self.conditioning_images) > 1:
            raise ValueError("Phase 1 supports at most one conditioning image")
        if self.conditioning_images:
            image = self.conditioning_images[0]
            if image.frame_idx != 0:
                raise ValueError("Phase 1 supports only frame_idx=0 for I2V")
        return self

    @property
    def generation_mode(self) -> Literal["t2v", "i2v"]:
        return "i2v" if self.conditioning_images else "t2v"


class UploadImageResponse(BaseModel):
    image_id: str
    original_filename: str
    stored_path: str
    width: int
    height: int
    content_type: str


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class GenerateResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str


class JobResult(BaseModel):
    video_url: str
    duration_seconds: float
    resolution: str
    file_size_bytes: int
    generation_time_seconds: float
    seed_used: int
    output_path: str
    metadata_path: str


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float
    current_step: int | None
    total_steps: int | None
    created_at: str
    started_at: str | None
    completed_at: str | None
    error: str | None
    request: GenerateRequest
    result: JobResult | None
