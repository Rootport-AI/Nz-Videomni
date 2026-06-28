"""LTX pipeline adapter — the ONLY file that knows about LTX internals (spec 9.4).

Two backends live behind one facade (:class:`LTXRunner`):

* ``_MockBackend`` — renders a short synthetic clip with PIL and encodes it to
  ``output.mp4`` with no GPU and no model weights. Used for tests and GPU-less
  development. This is the original Phase-1 implementation, kept intact.
* ``_RealBackend`` — drives the official two-stage ``DistilledPipeline`` (FP8
  cast, CPU offload, VAE tiling) and encodes the returned tensor iterator via
  the official ``encode_video``.

Backend selection (``LTXRunner.load``):

* ``config.model.backend == "mock"`` -> always mock.
* ``config.model.backend == "real"`` -> always real (RuntimeError if the real
  stack / model weights are unavailable).
* ``config.model.backend == "auto"`` (default) -> real when the model paths
  exist *and* torch+CUDA+``ltx_pipelines`` import successfully, else mock.

IMPORTANT: torch / ltx_pipelines / ltx_core are imported lazily *inside methods*
only. ``import services.ltx_runner`` must keep working in the app ``.venv`` that
has no torch installed (the mock test path depends on this).

Everything outside this file — request schema, job layer, output layout — is
unchanged. Only ``GenerationOutcome.backend`` differs between backends.
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

# Mock backend identifier surfaced in logs / status / metadata.
MOCK_BACKEND = "mock"
# Real backend identifier surfaced in metadata.
REAL_BACKEND = "ltx-distilled"


@dataclass
class GenerationOutcome:
    output_path: Path
    seed_used: int
    peak_vram_mb: int | None
    generation_mode: str  # "t2v" | "i2v"
    backend: str = MOCK_BACKEND


class LTXRunner:
    """Facade around the LTX pipeline; delegates to a mock or real backend.

    Public contract (unchanged): ``LTXRunner(config, low_vram)``, properties
    ``loaded`` / ``pipeline_type``, methods ``load`` / ``unload`` /
    ``generate(...) -> GenerationOutcome``.
    """

    def __init__(self, config: AppConfig, low_vram: LowVramSettings):
        self.config = config
        self.low_vram = low_vram
        self._backend: _MockBackend | _RealBackend | None = None

    @property
    def loaded(self) -> bool:
        return self._backend is not None and self._backend.loaded

    @property
    def pipeline_type(self) -> str:
        return self.config.model.pipeline_type

    # Back-compat: pipeline_manager / tests never read this, but the original
    # attribute existed. The active backend's handle (None for mock) is exposed.
    @property
    def pipeline(self):
        return getattr(self._backend, "pipeline", None)

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        if self._backend is not None and self._backend.loaded:
            return
        if self._backend is None:
            self._backend = self._select_backend()
        self._backend.load()

    def unload(self) -> None:
        if self._backend is None:
            return
        self._backend.unload()

    # -------------------------------------------------------------- generate

    def generate(
        self,
        request: GenerateRequest,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
        conditioning_image_paths: list[Path] | None = None,
    ) -> GenerationOutcome:
        if self._backend is None or not self._backend.loaded:
            self.load()
        assert self._backend is not None
        return self._backend.generate(
            request,
            output_dir=output_dir,
            progress_callback=progress_callback,
            conditioning_image_paths=conditioning_image_paths,
        )

    # ----------------------------------------------------- backend selection

    def _select_backend(self) -> _MockBackend | _RealBackend:
        choice = (self.config.model.backend or "auto").strip().lower()
        if choice == "mock":
            logger.info("Backend forced to MOCK (model.backend=mock).")
            return _MockBackend(self.config, self.low_vram)
        if choice == "real":
            if not self._real_available():
                raise RuntimeError(
                    "model.backend='real' but the real LTX stack is unavailable "
                    "(check torch+CUDA, ltx_pipelines install, and model paths)."
                )
            logger.info("Backend forced to REAL (model.backend=real).")
            return _RealBackend(self.config, self.low_vram)
        # auto
        if self._real_available():
            logger.info("Backend auto-selected: REAL.")
            return _RealBackend(self.config, self.low_vram)
        logger.info("Backend auto-selected: MOCK (real stack/model unavailable).")
        return _MockBackend(self.config, self.low_vram)

    def _real_available(self) -> bool:
        """True only if model weights exist AND torch+CUDA+ltx_pipelines import.

        Any failure is swallowed -> False (so 'auto' falls back to mock and
        ``import services.ltx_runner`` stays safe in a torch-less venv).
        """
        model = self.config.model
        try:
            paths = [
                model.checkpoint_path,
                model.spatial_upsampler_path,
                model.gemma_root,
            ]
            if any(not p for p in paths):
                return False
            for p in paths:
                if not self.config._abs(p).exists():
                    return False
            import torch  # type: ignore  # noqa: PLC0415

            if not torch.cuda.is_available():
                return False
            import ltx_pipelines  # type: ignore  # noqa: F401,PLC0415

            return True
        except Exception:
            return False


class _MockBackend:
    """Synthetic-clip backend (no GPU, no weights). Original Phase-1 logic."""

    def __init__(self, config: AppConfig, low_vram: LowVramSettings):
        self.config = config
        self.low_vram = low_vram
        self.pipeline = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
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

    def generate(
        self,
        request: GenerateRequest,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
        conditioning_image_paths: list[Path] | None = None,
    ) -> GenerationOutcome:
        """Generate a synthetic video and return the outcome (output.mp4 + metrics).

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
            backend=MOCK_BACKEND,
        )

    # --------------------------------------------------------- mock renderer

    def _render_frames(
        self,
        request: GenerateRequest,
        seed: int,
        start_image: Image.Image | None,
        progress_callback: ProgressCallback | None,
    ) -> list[Image.Image]:
        """Render a synthetic clip."""
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


class _RealBackend:
    """Official two-stage ``DistilledPipeline`` backend.

    All torch / ltx imports are local to methods so this class can be defined and
    referenced in a torch-less environment; it is only instantiated when the real
    stack is actually available.
    """

    def __init__(self, config: AppConfig, low_vram: LowVramSettings):
        self.config = config
        self.low_vram = low_vram
        self.pipeline = None  # DistilledPipeline instance once loaded

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    def load(self) -> None:
        if self.pipeline is not None:
            return

        from ltx_pipelines.distilled import DistilledPipeline  # noqa: PLC0415
        from ltx_pipelines.utils.types import OffloadMode  # noqa: PLC0415
        from ltx_pipelines.utils.quantization_factory import QuantizationKind  # noqa: PLC0415

        model = self.config.model
        ckpt = self._require_path(model.checkpoint_path, "checkpoint_path")
        upsampler = self._require_path(model.spatial_upsampler_path, "spatial_upsampler_path")
        gemma = self._require_path(model.gemma_root, "gemma_root")

        offload_mode = OffloadMode.CPU if self.low_vram.low_vram_mode else OffloadMode.NONE

        quant = None
        if self.low_vram.fp8_transformer and model.quantization not in (None, "", "none"):
            quant = QuantizationKind(model.quantization).to_policy(checkpoint_path=ckpt)

        logger.info(
            "Loading pipeline (REAL DistilledPipeline). offload=%s quant=%s ckpt=%s",
            offload_mode,
            model.quantization if quant is not None else None,
            ckpt,
        )

        self.pipeline = DistilledPipeline(
            distilled_checkpoint_path=ckpt,
            gemma_root=gemma,
            spatial_upsampler_path=upsampler,
            loras=[],
            quantization=quant,
            compilation_config=None,
            offload_mode=offload_mode,
        )

    def unload(self) -> None:
        if self.pipeline is None:
            return
        logger.info("Unloading pipeline (REAL).")
        self.pipeline = None
        safe_memory_cleanup()

    def generate(
        self,
        request: GenerateRequest,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
        conditioning_image_paths: list[Path] | None = None,
    ) -> GenerationOutcome:
        import random as _random  # noqa: PLC0415
        import torch  # noqa: PLC0415

        from ltx_pipelines.utils.args import ImageConditioningInput  # noqa: PLC0415
        from ltx_core.model.video_vae import (  # noqa: PLC0415
            TilingConfig,
            get_video_chunks_number,
        )
        from ltx_pipelines.utils.media_io import encode_video  # noqa: PLC0415

        if self.pipeline is None:
            self.load()

        conditioning_image_paths = conditioning_image_paths or []
        mode = request.generation_mode

        with torch.inference_mode():
            seed = request.seed if request.seed >= 0 else _random.randint(0, 2**31 - 1)

            gpu_info.reset_peak_vram()
            if progress_callback:
                progress_callback(None, None, 0.05)

            # Image conditioning: minimal Phase-1 I2V (one image, frame_idx=0).
            images: list = []
            if mode == "i2v" and conditioning_image_paths:
                ci = request.conditioning_images[0]
                images = [
                    ImageConditioningInput(
                        path=str(conditioning_image_paths[0]),
                        frame_idx=0,
                        strength=ci.strength,
                        crf=(ci.crf if ci.crf is not None else 33),
                    )
                ]

            tiling = TilingConfig.default() if self.low_vram.vae_tiling else None

            video_iter, _audio = self.pipeline(
                prompt=request.prompt,
                seed=seed,
                height=request.height,
                width=request.width,
                num_frames=request.num_frames,
                frame_rate=request.frame_rate,
                images=images,
                tiling_config=tiling,
                enhance_prompt=False,
            )

            if progress_callback:
                progress_callback(None, None, 0.90)

            output_path = output_dir / "output.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            chunks = get_video_chunks_number(request.num_frames, tiling)

            if request.crop_output is not None:
                # Encode at generation size, then center-crop to the final size.
                tmp = output_dir / "_full.mp4"
                encode_video(
                    video=video_iter,
                    fps=int(request.frame_rate),
                    audio=None,
                    output_path=str(tmp),
                    video_chunks_number=chunks,
                )
                video_io.crop_mp4(
                    tmp,
                    output_path,
                    request.crop_output.width,
                    request.crop_output.height,
                )
                tmp.unlink(missing_ok=True)
            else:
                encode_video(
                    video=video_iter,
                    fps=int(request.frame_rate),
                    audio=None,
                    output_path=str(output_path),
                    video_chunks_number=chunks,
                )

            peak = gpu_info.peak_vram_mb()
            if progress_callback:
                progress_callback(None, None, 1.0)

        safe_memory_cleanup()

        return GenerationOutcome(
            output_path=output_path,
            seed_used=seed,
            peak_vram_mb=peak,
            generation_mode=mode,
            backend=REAL_BACKEND,
        )

    # --------------------------------------------------------------- helpers

    def _require_path(self, value: str | None, label: str) -> str:
        if not value:
            raise RuntimeError(f"model.{label} is not configured (required for the real backend).")
        resolved = self.config._abs(value)
        if not resolved.exists():
            raise RuntimeError(f"model.{label} not found: {resolved}")
        return str(resolved)


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
