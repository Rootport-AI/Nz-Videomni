"""LTX pipeline adapter — the ONLY file that knows about LTX internals (spec 9.4).

Phase 1 ships a MOCK implementation so the whole backend (API, jobs, Gradio,
tests) can run end-to-end with no GPU and no model weights. The mock renders a
short synthetic clip and encodes it to ``output.mp4`` exactly like the real
pipeline would.

Step 7 (real LTX): replace ``load`` / ``unload`` / ``_render_frames`` with calls
to the official ``DistilledPipeline`` (FP8, CPU offload, VAE tiling). Everything
outside this file — request schema, job layer, output layout — stays unchanged.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

from api.models import GenerateRequest
from config import AppConfig
from services import gpu_info, video_io
from services.low_vram import LowVramSettings, safe_memory_cleanup

logger = logging.getLogger("ltx.runner")

# (current_step, total_steps, progress 0..1)
ProgressCallback = Callable[[int | None, int | None, float], None]

# Phase 1 mock identifier surfaced in logs / status.
MOCK_BACKEND = "mock"


@dataclass
class GenerationOutcome:
    output_path: Path
    seed_used: int
    peak_vram_mb: int | None
    generation_mode: str  # "t2v" | "i2v"
    backend: str = MOCK_BACKEND


class LTXRunner:
    """Adapter around the official LTX pipeline (mocked in Phase 1)."""

    def __init__(self, config: AppConfig, low_vram: LowVramSettings):
        self.config = config
        self.low_vram = low_vram
        self.pipeline = None  # real DistilledPipeline instance in Step 7
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def pipeline_type(self) -> str:
        return self.config.model.pipeline_type

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        """Load the pipeline. Mock: flips a flag and logs the intended config.

        Real (Step 7)::

            from ltx_pipelines.distilled_pipeline import DistilledPipeline
            from ltx_core.quantization.policy import QuantizationPolicy
            quant = QuantizationPolicy.fp8_cast() if self.low_vram.fp8_transformer else None
            self.pipeline = DistilledPipeline.from_config(config_path, quantization=quant)
        """
        if self._loaded:
            return
        logger.info(
            "Loading pipeline (MOCK). checkpoint=%s low_vram=%s fp8=%s cpu_offload_te=%s vae_tiling=%s",
            self.config.model.checkpoint_name,
            self.low_vram.low_vram_mode,
            self.low_vram.fp8_transformer,
            self.low_vram.cpu_offload_text_encoder,
            self.low_vram.vae_tiling,
        )
        self.pipeline = None
        self._loaded = True

    def unload(self) -> None:
        if not self._loaded:
            return
        logger.info("Unloading pipeline (MOCK).")
        self.pipeline = None
        self._loaded = False
        safe_memory_cleanup()

    # -------------------------------------------------------------- generate

    def generate(
        self,
        request: GenerateRequest,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
        conditioning_image_paths: list[Path] | None = None,
    ) -> GenerationOutcome:
        """Generate a video and return the outcome (output.mp4 + metrics).

        ``conditioning_images`` empty -> T2V; one entry -> minimal I2V using the
        resolved image path as the start frame (frame_idx=0, Phase 1).
        """
        if not self._loaded:
            self.load()

        seed = request.seed if request.seed >= 0 else random.randint(0, 2**31 - 1)
        mode = request.generation_mode
        conditioning_image_paths = conditioning_image_paths or []

        gpu_info.reset_peak_vram()
        if progress_callback:
            progress_callback(0, request.num_inference_steps, 0.05)

        start_image: Image.Image | None = None
        if mode == "i2v" and conditioning_image_paths:
            start_image = Image.open(conditioning_image_paths[0]).convert("RGB")
            start_image = start_image.resize((request.width, request.height))

        frames = self._render_frames(
            request=request,
            seed=seed,
            start_image=start_image,
            progress_callback=progress_callback,
        )

        if progress_callback:
            progress_callback(request.num_inference_steps, request.num_inference_steps, 0.90)

        crop = None
        if request.crop_output is not None:
            crop = (request.crop_output.width, request.crop_output.height)

        output_path = output_dir / "output.mp4"
        video_io.encode_frames_to_mp4(
            frames,
            output_path,
            frame_rate=request.frame_rate,
            crop=crop,
            keep_raw=self.config.output.keep_raw_frames,
            raw_dir=output_dir / "raw",
        )

        peak = gpu_info.peak_vram_mb()
        if progress_callback:
            progress_callback(request.num_inference_steps, request.num_inference_steps, 1.0)
        safe_memory_cleanup()

        return GenerationOutcome(
            output_path=output_path,
            seed_used=seed,
            peak_vram_mb=peak,
            generation_mode=mode,
        )

    # --------------------------------------------------------- mock renderer

    def _render_frames(
        self,
        request: GenerateRequest,
        seed: int,
        start_image: Image.Image | None,
        progress_callback: ProgressCallback | None,
    ) -> list[Image.Image]:
        """Render a synthetic clip. Replaced by real LTX inference in Step 7."""
        rng = random.Random(seed)
        w, h = request.width, request.height
        n = request.num_frames
        steps = max(1, request.num_inference_steps)

        base_hue = rng.randint(0, 359)
        # a moving accent so the mock clip is visibly a video, not a still
        ball_color = (rng.randint(120, 255), rng.randint(120, 255), rng.randint(120, 255))

        frames: list[Image.Image] = []
        for i in range(n):
            t = i / max(1, n - 1)
            if start_image is not None:
                # I2V: start from the uploaded frame and add a subtle pan/zoom.
                frame = start_image.copy()
                zoom = 1.0 + 0.06 * t
                zw, zh = int(w * zoom), int(h * zoom)
                frame = frame.resize((zw, zh))
                left = int((zw - w) * (0.5 + 0.1 * math.sin(t * math.pi)))
                top = int((zh - h) * 0.5)
                frame = frame.crop((left, top, left + w, top + h))
            else:
                # T2V: a slowly shifting gradient background.
                hue = (base_hue + int(t * 60)) % 360
                frame = _hue_gradient(w, h, hue)

            draw = ImageDraw.Draw(frame)
            cx = int(w * (0.15 + 0.7 * t))
            cy = int(h * (0.5 + 0.3 * math.sin(t * 2 * math.pi)))
            r = max(6, min(w, h) // 12)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ball_color)
            frames.append(frame)

            # emulate diffusion step progress (0.10 -> 0.90 across steps)
            if progress_callback and n > 1:
                step = 1 + int(t * (steps - 1))
                progress = 0.10 + 0.80 * t
                progress_callback(step, steps, round(progress, 3))

        return frames


def _hue_gradient(w: int, h: int, hue: int) -> Image.Image:
    """A cheap vertical gradient in a given hue (HSV-ish), as an RGB image."""
    from colorsys import hsv_to_rgb

    img = Image.new("RGB", (w, h))
    px = img.load()
    top = hsv_to_rgb(hue / 360.0, 0.55, 0.85)
    bot = hsv_to_rgb(hue / 360.0, 0.75, 0.35)
    for y in range(h):
        f = y / max(1, h - 1)
        r = int(255 * (top[0] * (1 - f) + bot[0] * f))
        g = int(255 * (top[1] * (1 - f) + bot[1] * f))
        b = int(255 * (top[2] * (1 - f) + bot[2] * f))
        for x in range(w):
            px[x, y] = (r, g, b)
    return img
