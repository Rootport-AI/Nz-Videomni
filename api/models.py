"""Pydantic schemas — the external API contract (spec ch.6).

Conditioning is now Phase-3 UNFROZEN: multiple keyframes (cap 5), arbitrary
``frame_idx`` (snapped server-side to the official ``0``-or-``8n+1`` latent grid
and clamped into range), and per-item ``strength``. ``num_pixel_frames`` and reference-video conditioning
remain out of scope. The OTHER constraints stay FROZEN as the final-form API so
that future frontends (AviUtl2, DaVinci Resolve) and later phases do not break:
÷64 generation resolution, 8n+1 frame counts, and the distilled 8-step / CFG=1.0
requirement. Do not relax those validators without revisiting the spec.

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
    frame_idx: int = Field(0, ge=0)
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

    # 空配列なら T2V。1件以上なら I2V（マルチキーフレーム対応、cap 5）。
    # 各 frame_idx は validator で 0-or-8n+1 グリッドへスナップ＋範囲クランプされる。
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

        # Conditioning (Phase 3): multi-keyframe I2V, cap 5.
        if len(self.conditioning_images) > 5:
            raise ValueError("at most 5 conditioning images are supported")
        # frame_idx is fed to the engine's guide path (VideoConditionByKeyframeIndex)
        # as a RAW PIXEL RoPE offset (positions[:,0] += frame_idx, no ÷8), so it must
        # be a latent-aligned pixel. Two on-grid cases, per the official LTX-2 /
        # ComfyUI (LTXVAddGuide) convention:
        #   * frame_idx == 0  -> start frame, routed to the latent-replace path;
        #     left byte-identical (the engine also special-cases idx==0 causal_fix).
        #   * frame_idx  > 0  -> a keyframe/guide that must sit on a latent-frame
        #     START pixel = the 8n+1 grid. Snap via (f-1)//8*8+1 (ComfyUI
        #     get_latent_index) and clamp to [1, num_frames-8] — for num_frames=8m+1
        #     the last latent-frame start is num_frames-8 (e.g. 49 -> 41).
        # Snapping is a safety net — the UI may still send natural values. The ÷64
        # rule is spatial-only and unrelated.
        for image in self.conditioning_images:
            if image.frame_idx == 0:
                continue  # latent-replace path (start frame); byte-identical to today
            snapped = (image.frame_idx - 1) // 8 * 8 + 1
            image.frame_idx = max(1, min(snapped, self.num_frames - 8))
        return self

    @property
    def generation_mode(self) -> Literal["t2v", "i2v"]:
        return "i2v" if self.conditioning_images else "t2v"


# Total-timeline pixel-frame cap. The masked AV-latent chain is decoded ONCE
# with an always-tiled stage-2, so it is NOT bound by the single-clip 481f cap
# (the tiling keeps VRAM flat at any length). The cap here is a sanity ceiling
# = 8 clips × 481f (the max clip count × the frozen per-clip cap), documented so
# a UI cannot request an unbounded timeline.
MAX_CHAIN_TOTAL_PIXEL_FRAMES = 8 * 481  # 3848


class ChainClip(BaseModel):
    """One clip in a generate-chain request.

    ``prompt`` is optional: when absent the chain's base ``prompt`` is used
    (prompt propagation). ``num_frames`` must be 8n+1. Only clip 0 may carry
    conditioning images (minimal I2V start / keyframes) — clips 1..N are the
    later segments of one continuous masked AV-latent timeline (they inherit
    continuity from the previous segment's frozen overlap latents, not images).
    """

    prompt: str | None = Field(None, max_length=2000)
    num_frames: int = Field(49, ge=9, le=481)
    conditioning_images: list[ConditioningImage] = Field(default_factory=list)


class GenerateChainRequest(BaseModel):
    """A chain of clips assembled into ONE continuous masked AV-latent timeline.

    Additive to the frozen single-``/generate`` contract. Shares
    width/height/seed/frame_rate/pipeline across clips; each clip's effective
    prompt is its override if present else the global ``prompt``. Reuses the
    same FROZEN validators (÷64 resolution, 8n+1 frames, distilled 8-step /
    CFG=1.0) as :class:`GenerateRequest`.

    ARCHITECTURE (Phase 3 WP4): every clip is a stage-1 SEGMENT of one timeline;
    the segments are stage-1 generated with a video+audio latent tail carry over
    a ``overlap_frames`` (= K_v LATENT-frame) overlap, crossfaded into one latent,
    then refined in temporal tiles and decoded ONCE. There is no per-clip mp4 and
    no pixel-domain concat/trim — boundaries live inside the single decode, so
    the seams are continuous.
    """

    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = ""

    width: int = Field(512, ge=256, le=4096)
    height: int = Field(320, ge=128, le=4096)
    crop_output: CropOutput | None = None

    frame_rate: float = Field(24.0, ge=1.0, le=60.0)
    num_inference_steps: int = Field(8, ge=1, le=100)
    guidance_scale: float = Field(1.0, ge=0.0, le=20.0)
    seed: int = -1
    pipeline: Literal["distilled", "two_stage_hq"] = "distilled"

    # Continuity (Phase 3 WP4): overlap = K_v LATENT frames shared between
    # consecutive stage-1 segments (the previous segment's tail is copied into
    # the next segment's head and frozen at ``overlap_strength``). K_v=3 is the
    # spike-validated default; must be < every clip's stage-1 latent-frame count.
    overlap_frames: int = Field(3, ge=1, le=8)
    overlap_strength: float = Field(0.5, ge=0.0, le=1.0)

    # 2..8 clips: at least 2 (a single clip is just /generate); capped so the
    # total timeline stays within MAX_CHAIN_TOTAL_PIXEL_FRAMES.
    clips: list[ChainClip] = Field(..., min_length=2, max_length=8)

    @model_validator(mode="after")
    def validate_chain_constraints(self) -> "GenerateChainRequest":
        if self.width % 64 != 0:
            raise ValueError("width must be a multiple of 64")
        if self.height % 64 != 0:
            raise ValueError("height must be a multiple of 64")
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

        for i, clip in enumerate(self.clips):
            if (clip.num_frames - 1) % 8 != 0:
                raise ValueError(f"clips[{i}].num_frames must be 8n+1")
            # overlap_frames (K_v LATENT) must fit inside EVERY clip's stage-1
            # latent-frame count = (num_frames - 1)//8 + 1 (each segment either
            # provides or receives the K_v-frame overlap).
            stage1_frames = (clip.num_frames - 1) // 8 + 1
            if self.overlap_frames >= stage1_frames:
                raise ValueError(
                    f"overlap_frames ({self.overlap_frames}) must be < "
                    f"clips[{i}] stage-1 latent frames ({stage1_frames})"
                )
            # Only clip 0 may carry conditioning images.
            if i > 0 and clip.conditioning_images:
                raise ValueError(
                    "only clip 0 may carry conditioning_images (later clips are "
                    "later segments of one continuous timeline)"
                )
            if len(clip.conditioning_images) > 5:
                raise ValueError(
                    f"clips[{i}]: at most 5 conditioning images are supported"
                )
            for image in clip.conditioning_images:
                if image.frame_idx == 0:
                    continue
                snapped = (image.frame_idx - 1) // 8 * 8 + 1
                image.frame_idx = max(1, min(snapped, clip.num_frames - 8))

        # Total-timeline geometry: sum of pixel frames minus the shared overlaps.
        # Delegated to the shared pure-Python chain_math so the validator, the
        # engine and the metadata agree; it also raises on a degenerate audio
        # overlap (clips too short for a continuous crossfade).
        import chain_math
        try:
            layout = chain_math.compute_chain_layout(
                [c.num_frames for c in self.clips], self.frame_rate,
                kv=self.overlap_frames,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        if layout.total_px > MAX_CHAIN_TOTAL_PIXEL_FRAMES:
            raise ValueError(
                f"chain total timeline {layout.total_px} pixel frames exceeds the "
                f"cap {MAX_CHAIN_TOTAL_PIXEL_FRAMES} (reduce clip count or lengths)"
            )
        return self

    def clip_prompt(self, index: int) -> str:
        """Effective prompt for clip ``index`` (override else global base)."""
        override = self.clips[index].prompt
        return override if override else self.prompt

    def to_clip_request(self, index: int) -> "GenerateRequest":
        """Build the per-clip :class:`GenerateRequest` (clip 0 keeps its images)."""
        clip = self.clips[index]
        return GenerateRequest(
            prompt=self.clip_prompt(index),
            negative_prompt=self.negative_prompt,
            width=self.width,
            height=self.height,
            crop_output=None,  # crop is applied once, on the final concat.
            num_frames=clip.num_frames,
            frame_rate=self.frame_rate,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            seed=self.seed,
            pipeline=self.pipeline,
            conditioning_images=clip.conditioning_images if index == 0 else [],
        )


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


class GenerateChainResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    num_clips: int


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
