"""LTX pipeline adapter — the ONLY file that knows about LTX internals (spec 9.4).

Two backends live behind one facade (:class:`LTXRunner`):

* ``_MockBackend`` — renders a short synthetic clip with PIL and encodes it to
  ``output.mp4`` with no GPU and no model weights. Used for tests and GPU-less
  development. This is the original Phase-1 implementation, kept intact.
* ``_RealBackend`` — Phase 5 (Approach W): manages a persistent subprocess
  worker (``engine.worker``, launched as ``python -m engine.worker``) that runs
  inside the engine venv where the proven GGUF low-VRAM engine lives. It builds
  the model once, then serves jobs over a small JSON-lines protocol; the engine
  writes ``output.mp4`` directly to the shared output dir. This backend NEVER
  imports torch / ltx_* itself — those packages exist only in the engine venv,
  not the app venv.

Backend selection (``LTXRunner.load``):

* ``config.model.backend == "mock"`` -> always mock.
* ``config.model.backend == "real"`` -> always real (RuntimeError if the fork
  python / worker / model weights are unavailable).
* ``config.model.backend == "auto"`` (default) -> real when the fork python,
  worker script and all model paths exist, else mock.

IMPORTANT: torch / ltx_pipelines / ltx_core are NEVER imported in this file.
``import services.ltx_runner`` must keep working in the app ``.venv`` that has no
torch installed (the mock test path depends on this); the real backend defers
all engine work to the subprocess worker.

Everything outside this file — request schema, job layer, output layout — is
unchanged. Only ``GenerationOutcome.backend`` differs between backends.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

import chain_math
from api.errors import lora_preprocess_conflict
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


def _resolve_reference_preprocess(lora_paths: list[tuple[Path, float, str]]) -> str:
    """Phase C: derive the single control-preprocess kind for the one reference
    video from the resolved loras of a job.

    ``lora_paths`` entries are ``(path, strength, preprocess)`` (see
    ``services.lora_registry.LoraRegistry.resolve``). All-``"none"`` (Phase B
    reference-only adapters, or no loras) -> ``"none"``. Exactly one non-``"none"``
    kind -> that kind. More than one distinct kind is a conflict: a single
    uploaded reference video can only be converted into ONE control signal, so
    this raises ``LORA_PREPROCESS_CONFLICT`` (400) -- the same check the API
    layer (``api/generate.py``) already performs up front; this is the
    defensive re-check at the runner hop.
    """
    kinds = {preprocess for _, _, preprocess in lora_paths if preprocess != "none"}
    if len(kinds) > 1:
        raise lora_preprocess_conflict(sorted(kinds))
    return next(iter(kinds)) if kinds else "none"


@dataclass
class GenerationOutcome:
    output_path: Path
    seed_used: int
    peak_vram_mb: int | None
    generation_mode: str  # "t2v" | "i2v" | "chain"
    backend: str = MOCK_BACKEND
    # Phase 3 WP4 masked AV-latent chain: junction pixel-frame indices + full
    # geometry (from chain_math / the engine). None for single-clip generate.
    chain_metadata: dict | None = None


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
        lora_paths: list[tuple[Path, float, str]] | None = None,
        reference_video_path: Path | None = None,
    ) -> GenerationOutcome:
        if self._backend is None or not self._backend.loaded:
            self.load()
        assert self._backend is not None
        return self._backend.generate(
            request,
            output_dir=output_dir,
            progress_callback=progress_callback,
            conditioning_image_paths=conditioning_image_paths,
            lora_paths=lora_paths,
            reference_video_path=reference_video_path,
        )

    def generate_chain(
        self,
        chain_request,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
        clip0_conditioning_paths: list[Path] | None = None,
    ) -> GenerationOutcome:
        """Masked AV-latent clip chain -> ONE continuous output.mp4 (Phase 3 WP4)."""
        if self._backend is None or not self._backend.loaded:
            self.load()
        assert self._backend is not None
        return self._backend.generate_chain(
            chain_request,
            output_dir=output_dir,
            progress_callback=progress_callback,
            clip0_conditioning_paths=clip0_conditioning_paths,
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
        """True only if the engine python, worker script and every file the real
        GGUF + component-file path actually loads are present.

        The GGUF + component-file recipe never opens the 43GB monolith
        (``checkpoint_path``): it is passed to the worker as a reference-only
        payload field (the wheel's lazy builders receive it but the GGUF/component
        installs replace every loader), so it is deliberately NOT gated here. The
        (tokenizer-only ~40MB) ``gemma_root`` IS gated: DistilledPipeline is built
        with gemma_root=None so the wheel's weight glob is bypassed,
        but the engine still loads the tokenizer/processor module_ops from this dir,
        so a missing dir must fail fast in the app layer rather than crash deep in
        the encode path. The load-bearing files are the tokenizer gemma_root, the
        GGUF transformer/Gemma, the spatial upsampler, and the 3 standalone component
        files (use_component_files is fixed True in config.yaml).

        Deliberately does NOT import torch / ltx_* (those live only in the engine
        venv, not the app venv). Any failure/missing is swallowed -> False (so
        'auto' falls back to mock and ``import services.ltx_runner`` stays safe
        in the torch-less app venv).
        """
        model = self.config.model
        try:
            required = [
                model.engine_python,
                model.gemma_root,
                model.spatial_upsampler_path,
                model.gguf_transformer_path,
                model.gguf_gemma_path,
                model.component_video_vae_path,
                model.component_audio_vae_path,
                model.component_text_projection_path,
            ]
            if any(not p for p in required):
                return False
            for p in required:
                if not self.config._abs(p).exists():
                    return False
            if not model.engine_dir:
                return False
            worker = self.config._abs(model.engine_dir) / "worker.py"
            if not worker.exists():
                return False
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
        lora_paths: list[tuple[Path, float, str]] | None = None,
        reference_video_path: Path | None = None,
    ) -> GenerationOutcome:
        """Generate a synthetic video and return the outcome (output.mp4 + metrics).

        ``conditioning_images`` empty -> T2V; one entry -> minimal I2V using the
        resolved image path as the start frame (frame_idx=0, Phase 1).

        ``lora_paths`` (now ``(path, strength, preprocess)`` triples, Phase C) /
        ``reference_video_path`` are the Phase B/C IC-LoRA inputs; the mock
        backend accepts (and ignores) them so the full route completes GPU-free —
        the real weight patch (and the preprocess -> control-signal conversion)
        lives in the engine worker.
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

    def generate_chain(
        self,
        chain_request,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
        clip0_conditioning_paths: list[Path] | None = None,
    ) -> GenerationOutcome:
        """Simulate a masked AV-latent chain: ONE synthetic mp4 of the full
        timeline length + junction metadata (from :mod:`chain_math`). GPU-free;
        exercises the app-side orchestrator/metadata without model weights.
        """
        if not self._loaded:
            self.load()

        chain = chain_request
        seed = chain.seed if chain.seed >= 0 else random.randint(0, 2**31 - 1)
        layout = chain_math.compute_chain_layout(
            [c.num_frames for c in chain.clips], chain.frame_rate,
            kv=chain.overlap_frames,
        )
        gpu_info.reset_peak_vram()
        if progress_callback:
            progress_callback(None, None, 0.05)

        start_image: Image.Image | None = None
        clip0_conditioning_paths = clip0_conditioning_paths or []
        if chain.clips[0].conditioning_images and clip0_conditioning_paths:
            start_image = Image.open(clip0_conditioning_paths[0]).convert("RGB")
            start_image = start_image.resize((chain.width, chain.height))

        frames = self._render_chain_frames(
            width=chain.width, height=chain.height, n=layout.total_px,
            seed=seed, start_image=start_image, progress_callback=progress_callback,
        )
        if progress_callback:
            progress_callback(None, None, 0.90)

        crop = None
        if chain.crop_output is not None:
            crop = (chain.crop_output.width, chain.crop_output.height)

        output_path = output_dir / "output.mp4"
        output_dir.mkdir(parents=True, exist_ok=True)
        video_io.encode_frames_to_mp4(
            frames, output_path, frame_rate=chain.frame_rate, crop=crop,
            keep_raw=self.config.output.keep_raw_frames, raw_dir=output_dir / "raw",
        )
        peak = gpu_info.peak_vram_mb()
        if progress_callback:
            progress_callback(None, None, 1.0)
        safe_memory_cleanup()

        return GenerationOutcome(
            output_path=output_path,
            seed_used=seed,
            peak_vram_mb=peak,
            generation_mode="chain",
            backend=MOCK_BACKEND,
            chain_metadata=layout.to_dict(),
        )

    def _render_chain_frames(
        self, *, width: int, height: int, n: int, seed: int,
        start_image: Image.Image | None, progress_callback: ProgressCallback | None,
    ) -> list[Image.Image]:
        """Cheap synthetic full-timeline clip (shifting gradient + moving ball)."""
        rng = random.Random(seed)
        base_hue = rng.randint(0, 359)
        ball_color = (rng.randint(120, 255), rng.randint(120, 255), rng.randint(120, 255))
        frames: list[Image.Image] = []
        for i in range(n):
            t = i / max(1, n - 1)
            if start_image is not None:
                frame = start_image.copy()
                zoom = 1.0 + 0.06 * t
                zw, zh = int(width * zoom), int(height * zoom)
                frame = frame.resize((zw, zh))
                left = int((zw - width) * (0.5 + 0.1 * math.sin(t * math.pi)))
                top = int((zh - height) * 0.5)
                frame = frame.crop((left, top, left + width, top + height))
            else:
                hue = (base_hue + int(t * 90)) % 360
                frame = _hue_gradient(width, height, hue)
            draw = ImageDraw.Draw(frame)
            cx = int(width * (0.15 + 0.7 * ((i * 5 % max(1, n)) / max(1, n))))
            cy = int(height * (0.5 + 0.3 * math.sin(t * 6 * math.pi)))
            r = max(6, min(width, height) // 12)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ball_color)
            frames.append(frame)
        return frames

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
    """Real GGUF low-VRAM backend via a persistent subprocess worker (Phase 5).

    This class NEVER imports torch / ltx_* (they live only in the engine venv).
    It spawns ``engine.worker`` (``python -m engine.worker``) in the engine venv,
    loads the model once, and serves jobs over a JSON-lines protocol framed by
    the ``@@LTX@@`` prefix. The worker
    writes ``output.mp4`` directly to the shared output dir; only small control
    JSON crosses the pipe. ``self.pipeline`` is retained (always None) only for
    the back-compat ``LTXRunner.pipeline`` attribute.
    """

    # Protocol frame prefix; must match engine.worker.PREFIX.
    _PREFIX = "@@LTX@@"
    _LOAD_TIMEOUT_S = 600.0
    _SHUTDOWN_TIMEOUT_S = 30.0

    def __init__(self, config: AppConfig, low_vram: LowVramSettings):
        self.config = config
        self.low_vram = low_vram
        self.pipeline = None  # back-compat attribute; always None for this backend
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._log_path: Path | None = None

    @property
    def loaded(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # --------------------------------------------------------------- helpers

    def _require_path(self, value: str | None, label: str) -> str:
        if not value:
            raise RuntimeError(f"model.{label} is not configured (required for the real backend).")
        resolved = self.config._abs(value)
        if not resolved.exists():
            raise RuntimeError(f"model.{label} not found: {resolved}")
        return str(resolved)

    def _stderr_tail(self, n: int = 2000) -> str:
        """Best-effort tail of the worker stderr log (for error messages)."""
        if self._log_path is None:
            return ""
        try:
            data = self._log_path.read_text(encoding="utf-8", errors="replace")
            return data[-n:]
        except Exception:
            return ""

    def _send(self, msg: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
        self._proc.stdin.flush()

    def _read_event(self, timeout: float | None = None) -> dict:
        """Block-read worker stdout until a framed ``@@LTX@@`` JSON line.

        Non-prefixed lines (library / tqdm noise that leaked to stdout) are
        ignored. EOF / dead process raises RuntimeError with the stderr tail.
        A timeout (when given) is enforced via a watchdog thread that kills the
        worker so the blocked readline returns.
        """
        assert self._proc is not None and self._proc.stdout is not None
        timer: threading.Timer | None = None
        timed_out = {"v": False}
        if timeout is not None:
            def _kill_on_timeout() -> None:
                timed_out["v"] = True
                self._kill()
            timer = threading.Timer(timeout, _kill_on_timeout)
            timer.daemon = True
            timer.start()
        try:
            for raw in self._proc.stdout:
                line = raw.strip()
                if not line.startswith(self._PREFIX):
                    continue  # ignore library/tqdm stdout noise
                try:
                    return json.loads(line[len(self._PREFIX):])
                except Exception:
                    continue
            # EOF on stdout -> process ended without a terminal event.
            if timed_out["v"]:
                raise RuntimeError(
                    f"LTX worker timed out after {timeout:.0f}s: {self._stderr_tail()}"
                )
            raise RuntimeError("LTX worker died: " + self._stderr_tail())
        finally:
            if timer is not None:
                timer.cancel()

    def _kill(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            proc.kill()
        except Exception:
            pass

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        if self.loaded:
            return

        model = self.config.model

        # Resolve + validate the engine python, worker script, and 5 model paths.
        engine_python = self._require_path(model.engine_python, "engine_python")
        if not model.engine_dir:
            raise RuntimeError("model.engine_dir is not configured (required for the real backend).")
        engine_dir = self.config._abs(model.engine_dir)
        if not engine_dir.exists():
            raise RuntimeError(f"model.engine_dir not found: {engine_dir}")
        worker = engine_dir / "worker.py"
        if not worker.exists():
            raise RuntimeError(f"LTX worker script not found: {worker}")
        # The worker is launched as `python -m engine.worker`, so its imports
        # (`engine.*`, `ltx_core`, `ltx_pipelines`) resolve from the project root.
        project_root = self.config._abs(".")

        # checkpoint_path (43GB monolith) is reference-only: the GGUF + component-file
        # path never opens it. It is still forwarded to the worker as a payload field
        # (the wheel's lazy builders expect it), so resolve to a project-rooted
        # absolute WITHOUT an existence check — it may be physically absent while the
        # real path still works.
        checkpoint_path = str(self.config._abs(model.checkpoint_path)) if model.checkpoint_path else ""
        # gemma_root (tokenizer-only ~40MB) IS load-bearing: DistilledPipeline is
        # built with gemma_root=None so the wheel's weight glob (model*.safetensors)
        # is bypassed, but the engine loads the tokenizer/processor
        # module_ops from this dir (tokenizer.model + preprocessor_config.json), so a
        # missing dir must fail fast here rather than crash deep in the encode path.
        # Forwarded to the worker as a payload field exactly as before.
        gemma_root = self._require_path(model.gemma_root, "gemma_root")

        upsampler_path = self._require_path(model.spatial_upsampler_path, "spatial_upsampler_path")
        gguf_transformer_path = self._require_path(model.gguf_transformer_path, "gguf_transformer_path")
        gguf_gemma_path = self._require_path(model.gguf_gemma_path, "gguf_gemma_path")

        # Component-file re-sourcing. Fixed on in config.yaml; the 3 standalone
        # files replace the monolith for VAE/audio (+ text projection connectors),
        # so they are load-bearing and always validated for existence.
        use_component_files = bool(self.config.vram.use_component_files)
        component_video_vae_path = self._require_path(
            model.component_video_vae_path, "component_video_vae_path"
        )
        component_audio_vae_path = self._require_path(
            model.component_audio_vae_path, "component_audio_vae_path"
        )
        component_text_projection_path = self._require_path(
            model.component_text_projection_path, "component_text_projection_path"
        )

        # Child env: inherit, force the 16GB-load-bearing CUDA + compile knobs,
        # unbuffered IO, and set PYTHONPATH to the project root so the worker's
        # `engine.*` package (and the venv-installed ltx_core/ltx_pipelines)
        # resolve when launched as `python -m engine.worker`.
        env = dict(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["TORCH_COMPILE_DISABLE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("PYTHONPATH", None)
        env["PYTHONPATH"] = str(project_root)
        # Phase 1 gate: the worker reads LTX_COMPONENT_FILES (mirrors LTX_KEEP_RESIDENT).
        env["LTX_COMPONENT_FILES"] = "1" if use_component_files else "0"
        # Default keep-resident-weights OFF. At 720p the keep-resident path builds
        # Gemma on CPU then does an out-of-place .to(cuda) move (momentary
        # double-residence) that overruns the 16GB card and hard-crashes the
        # worker (native, no traceback) during text-encode. This single-user /
        # single-job local server does not need cross-job weight reuse, so default
        # to 0; an explicit LTX_KEEP_RESIDENT in the environment still wins.
        env.setdefault("LTX_KEEP_RESIDENT", "0")
        # Sequential per-layer CPU offload of the GGUF Gemma during text-encode
        # (caps the ~15GB encode peak). On by default; the worker reads this and
        # keeps the 48 Gemma decoder layers CPU-resident, streaming them to GPU
        # per layer. Compute stays on GPU (only PCIe transfer overhead).
        env["LTX_TE_OFFLOAD"] = "1" if self.low_vram.te_offload_text_encoder else "0"
        # Build the DiT (transformer) on CPU and move only non-block submodules to
        # GPU, removing the ~16.9GB load-time GPU spike. On by default; the worker
        # reads this and keeps the blocks CPU-resident for block-swap streaming.
        env["LTX_DIT_CPU_LOAD"] = "1" if self.low_vram.dit_cpu_load else "0"

        # stderr -> a log file (NOT a pipe; piping stderr risks a deadlock when
        # the worker emits lots of tqdm/log output while we block on stdout).
        log_dir = self.config.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / "ltx_worker.log"

        knobs = {
            "gguf_per_layer_quant": bool(model.gguf_per_layer_quant),
            "block_swap_blocks_on_gpu": self.low_vram.block_swap_blocks_on_gpu or 8,
            "vae_spatial_tile_size": int(self.low_vram.vae_spatial_tile_size),
            "vae_temporal_tile_size": int(self.low_vram.vae_temporal_tile_size),
        }
        logger.info(
            "Loading pipeline (REAL worker). python=%s engine_dir=%s block_swap=%s ckpt=%s",
            engine_python,
            engine_dir,
            knobs["block_swap_blocks_on_gpu"],
            checkpoint_path,
        )

        log_fh = open(self._log_path, "a", encoding="utf-8")
        try:
            self._proc = subprocess.Popen(
                [engine_python, "-u", "-m", "engine.worker"],
                cwd=str(project_root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log_fh,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=env,
            )
        except Exception as exc:
            log_fh.close()
            self._proc = None
            raise RuntimeError(f"failed to launch LTX worker: {exc!r}") from exc

        try:
            self._send(
                {
                    "op": "load",
                    "checkpoint_path": checkpoint_path,
                    "gemma_root": gemma_root,
                    "upsampler_path": upsampler_path,
                    "gguf_transformer_path": gguf_transformer_path,
                    "gguf_gemma_path": gguf_gemma_path,
                    # Phase 1 component-file paths (gate via LTX_COMPONENT_FILES env).
                    "component_video_vae_path": component_video_vae_path,
                    "component_audio_vae_path": component_audio_vae_path,
                    "component_text_projection_path": component_text_projection_path,
                    **knobs,
                }
            )
            event = self._read_event(timeout=self._LOAD_TIMEOUT_S)
        except Exception:
            self._kill()
            self._proc = None
            raise

        kind = event.get("event")
        if kind == "ready":
            logger.info("LTX worker ready.")
            return
        if kind == "error":
            detail = event.get("detail", "")
            self._kill()
            self._proc = None
            raise RuntimeError(f"LTX worker failed to load: {detail}")
        # Unexpected terminal event.
        self._kill()
        self._proc = None
        raise RuntimeError(f"LTX worker returned unexpected event during load: {event!r}")

    def unload(self) -> None:
        proc = self._proc
        if proc is None:
            return
        logger.info("Unloading pipeline (REAL worker).")
        try:
            if proc.poll() is None and proc.stdin is not None:
                try:
                    self._send({"op": "shutdown"})
                except Exception:
                    pass
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            try:
                proc.wait(timeout=self._SHUTDOWN_TIMEOUT_S)
            except Exception:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    self._kill()
        finally:
            self._proc = None
            self.pipeline = None
            safe_memory_cleanup()

    # -------------------------------------------------------------- generate

    def generate(
        self,
        request: GenerateRequest,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
        conditioning_image_paths: list[Path] | None = None,
        lora_paths: list[tuple[Path, float, str]] | None = None,
        reference_video_path: Path | None = None,
    ) -> GenerationOutcome:
        if not self.loaded:
            self.load()

        conditioning_image_paths = conditioning_image_paths or []
        lora_paths = lora_paths or []
        mode = request.generation_mode

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "output.mp4"

        # Resolve the seed IN THE PARENT so seed_used is deterministic regardless
        # of the worker.
        seed = request.seed if request.seed >= 0 else random.randint(0, 2**31 - 1)

        # Image conditioning: multi-keyframe I2V. frame_idx is already snapped to
        # a multiple of 8 and clamped in the validator (api/models.py). cond_paths
        # are built by pipeline_manager in the same order as conditioning_images.
        # The engine's ImageConditioningInput has NO crf -> drop it.
        images: list[dict] = []
        if mode == "i2v" and conditioning_image_paths:
            images = [
                {"path": str(path), "frame_idx": ci.frame_idx, "strength": ci.strength}
                for ci, path in zip(request.conditioning_images, conditioning_image_paths)
            ]

        # crop_output: have the worker write the full-size mp4 to a temp file,
        # then center-crop into output.mp4 with the existing ffmpeg helper.
        if request.crop_output is not None:
            target = output_dir / "_full.mp4"
        else:
            target = output_path

        if progress_callback:
            progress_callback(None, None, 0.05)

        # Phase B/C IC-LoRA (forward-time weight patch). ``loras`` is the list of
        # (adapter safetensors path, strength, preprocess) resolved by the
        # registry; empty list -> the worker passes ic_loras=[] (explicit clean
        # detach per Stage 1 semantics). ``reference_video`` is the raw reference
        # (Pixel-Spatial-Upscaler: used as-is / Union-Control: converted to a
        # control signal by the worker per ``preprocess``), applied at a fixed
        # strength of 1.0; None when no loras. ``preprocess`` is derived from the
        # job's loras -- a conflict (>1 distinct kind) is rejected up front by
        # api/generate.py already, this is the defensive re-check at the runner.
        loras_payload = [{"path": str(p), "strength": float(s)} for p, s, _pp in lora_paths]
        preprocess = _resolve_reference_preprocess(lora_paths)
        reference_payload = (
            {"path": str(reference_video_path), "strength": 1.0, "preprocess": preprocess}
            if reference_video_path is not None
            else None
        )

        payload: dict = {
            "op": "generate",
            "prompt": request.prompt,
            "seed": seed,
            "height": request.height,
            "width": request.width,
            "num_frames": request.num_frames,
            "frame_rate": request.frame_rate,
            "num_steps": request.num_inference_steps,
            "images": images,
            "loras": loras_payload,
            "reference_video": reference_payload,
            "output_path": str(target),
        }

        # Serialize the stdin/stdout exchange (single-job server, but be safe).
        with self._lock:
            try:
                self._send(payload)
            except Exception as exc:
                raise RuntimeError("LTX worker died: " + self._stderr_tail()) from exc
            event = self._read_event()

        kind = event.get("event")
        if kind == "error":
            raise RuntimeError(event.get("detail", "LTX worker generation failed"))
        if kind != "done":
            raise RuntimeError(f"LTX worker returned unexpected event: {event!r}")

        if request.crop_output is not None:
            video_io.crop_mp4(
                target,
                output_path,
                request.crop_output.width,
                request.crop_output.height,
            )
            target.unlink(missing_ok=True)

        if not output_path.exists() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"LTX worker produced no/empty output: {output_path}")

        if progress_callback:
            progress_callback(None, None, 0.90)
            progress_callback(None, None, 1.0)

        seed_used = event.get("seed_used", seed)
        peak_vram_mb = event.get("peak_vram_mb")

        return GenerationOutcome(
            output_path=output_path,
            seed_used=seed_used,
            peak_vram_mb=peak_vram_mb,
            generation_mode=mode,
            backend=REAL_BACKEND,
        )

    def generate_chain(
        self,
        chain_request,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
        clip0_conditioning_paths: list[Path] | None = None,
    ) -> GenerationOutcome:
        """Masked AV-latent clip chain via the worker's ``generate_chain`` op.

        ONE worker invocation runs the whole chain (latents resident across
        segments) and writes ONE mp4. Progress events (per stage-1 segment, per
        stage-2 tile, decode) are streamed to ``progress_callback``; the terminal
        ``done`` carries peak VRAM + junction metadata.
        """
        if not self.loaded:
            self.load()

        chain = chain_request
        clip0_conditioning_paths = clip0_conditioning_paths or []
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "output.mp4"

        seed = chain.seed if chain.seed >= 0 else random.randint(0, 2**31 - 1)

        # Only clip 0 may carry conditioning images (validator enforces this).
        clip0_images: list[dict] = []
        if chain.clips[0].conditioning_images and clip0_conditioning_paths:
            clip0_images = [
                {"path": str(path), "frame_idx": ci.frame_idx, "strength": ci.strength}
                for ci, path in zip(chain.clips[0].conditioning_images, clip0_conditioning_paths)
            ]
        clips_payload = [
            {
                "prompt": chain.clip_prompt(i),
                "num_frames": chain.clips[i].num_frames,
                "images": clip0_images if i == 0 else [],
            }
            for i in range(len(chain.clips))
        ]

        target = (output_dir / "_full.mp4") if chain.crop_output is not None else output_path

        if progress_callback:
            progress_callback(None, None, 0.03)

        payload = {
            "op": "generate_chain",
            "width": chain.width,
            "height": chain.height,
            "frame_rate": chain.frame_rate,
            "num_steps": chain.num_inference_steps,
            "seed": seed,
            "overlap_frames": int(chain.overlap_frames),
            "overlap_strength": float(chain.overlap_strength),
            "output_path": str(target),
            "clips": clips_payload,
        }

        with self._lock:
            try:
                self._send(payload)
            except Exception as exc:
                raise RuntimeError("LTX worker died: " + self._stderr_tail()) from exc
            event = self._read_chain_events(progress_callback)

        kind = event.get("event")
        if kind == "error":
            raise RuntimeError(event.get("detail", "LTX worker chain generation failed"))
        if kind != "done":
            raise RuntimeError(f"LTX worker returned unexpected event: {event!r}")

        if chain.crop_output is not None:
            video_io.crop_mp4(target, output_path, chain.crop_output.width, chain.crop_output.height)
            target.unlink(missing_ok=True)

        if not output_path.exists() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"LTX worker produced no/empty chain output: {output_path}")

        if progress_callback:
            progress_callback(None, None, 1.0)

        return GenerationOutcome(
            output_path=output_path,
            seed_used=event.get("seed_used", seed),
            peak_vram_mb=event.get("peak_vram_mb"),
            generation_mode="chain",
            backend=REAL_BACKEND,
            chain_metadata=event.get("chain"),
        )

    def _read_chain_events(self, progress_callback: ProgressCallback | None) -> dict:
        """Read framed events until a terminal ``done``/``error``; forward
        ``progress`` events to ``progress_callback`` as a coarse 0..1 fraction."""
        while True:
            event = self._read_event()
            kind = event.get("event")
            if kind == "progress":
                if progress_callback:
                    stage = event.get("stage")
                    idx = int(event.get("index", 0))
                    total = max(1, int(event.get("total", 1)))
                    step = (idx + 1) / total
                    if stage == "stage1":
                        frac = 0.05 + 0.45 * step
                    elif stage == "tile":
                        frac = 0.50 + 0.40 * step
                    else:  # decode
                        frac = 0.95
                    progress_callback(None, None, round(min(1.0, frac), 3))
                continue
            return event


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
