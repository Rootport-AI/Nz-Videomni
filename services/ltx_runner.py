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
import time
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


def resolve_seed(requested: int) -> int:
    """Resolve a request seed to the concrete value actually used.

    ``requested >= 0`` is used verbatim; a negative (``-1`` random) request draws
    a fresh 31-bit seed. This is the SINGLE resolution point shared by the
    backends and :class:`services.pipeline_manager.PipelineManager`, so the seed
    logged at job start is byte-identical to the seed the backend runs with.
    """
    return requested if requested >= 0 else random.randint(0, 2**31 - 1)

# S2: friendly labels for the coarse chain-progress stages the engine emits
# (worker.py _progress -> chain_pipeline progress("stage1"|"tile"|"decode")).
# ``index`` is a COUNT of completed units (segments / tiles), not a denoise step,
# so the reported rate is honestly "units/s" for that stage, not raw it/s.
_CHAIN_STAGE_LABELS: dict[str, tuple[str, str]] = {
    "stage1": ("stage-1 denoise", "segment"),
    "tile": ("stage-2 tiled upsample", "tile"),
    "decode": ("VAE decode", "step"),
    # F2 additions: chain text-encode marker + per-step denoise stages emitted
    # by the worker's tqdm shim (engine/progress_shim.py).
    "encode": ("text encode", "step"),
    "stage1_denoise": ("stage-1 denoise", "step"),
    "stage2_denoise": ("stage-2 denoise", "step"),
    "denoise": ("denoise", "step"),
}

# Per-step denoise stages (F2): their ``index`` is the 1-based count of
# COMPLETED steps (the tqdm shim emits AFTER each step), unlike the coarse
# chain stages whose 0-based ``index`` means "unit index+1 is now complete".
# Their step/total pass straight through to the ProgressCallback.
_STEP_STAGES = frozenset({"stage1_denoise", "stage2_denoise", "denoise"})

# Don't log every event — a denoise runs steps quickly and the stage-2 tile
# count can be large. Emit the first event of a stage, its final event, and at
# most one line per this interval in between.
_STAGE_LOG_MIN_INTERVAL_S = 2.0


def _log_stage_progress(
    prefix: str,
    label: str,
    unit: str,
    done: int,
    total: int,
    state: dict[str, dict],
    key: str,
    it_s: float | None = None,
) -> None:
    """Emit a rate-limited INFO line for one progress event.

    ``done`` is the number of COMPLETED units (the caller normalizes coarse
    0-based indices to ``idx + 1`` and per-step 1-based counts as-is).
    ``state`` is caller-owned scratch (one dict per event-read loop) holding
    per-``key`` timing so throughput can be measured across events without a
    class attribute; per-step stages key per segment/tile so their timing (and
    the started line) resets each outer unit. ``it_s`` (worker-measured
    steps/s, when present) wins over the locally computed arrival rate. Pure
    logging — never touches the numeric/progress path.
    """
    now = time.monotonic()
    st = state.get(key)
    if st is None:
        state[key] = {"t0": now, "last_t": now, "last_done": done}
        if total <= 1:
            logger.info("%s %s started", prefix, label)
        else:
            logger.info("%s %s started (%d %ss)", prefix, label, total, unit)
        return
    is_last = done >= total
    if not is_last and (now - st["last_t"]) < _STAGE_LOG_MIN_INTERVAL_S:
        return
    if it_s is not None:
        rate = float(it_s)
    else:
        dt = now - st["last_t"]
        dd = done - st["last_done"]
        rate = dd / dt if dt > 0 else 0.0
    elapsed = now - st["t0"]
    logger.info(
        "%s %s %d/%d %ss (%.2f %s/s, %.1fs elapsed)",
        prefix, label, done, total, unit, rate, unit, elapsed,
    )
    st["last_t"] = now
    st["last_done"] = done


def _progress_frac(
    stage: str | None,
    done: int,
    total: int,
    outer_index: int | None,
    outer_total: int | None,
    *,
    chain: bool,
) -> float | None:
    """Map one worker progress event to the coarse job fraction (0..1).

    Chain milestones keep their historical values (0.05..0.50 stage 1,
    0.50..0.90 stage 2, 0.95 decode); per-step events interpolate WITHIN those
    bands using the segment/tile position (``outer_index``/``outer_total``)
    the shim attaches, so the fraction now moves every denoise step instead of
    once per 100+ seconds. Returns None for unknown stages (the caller keeps
    the last fraction — an unknown stage must never yank the bar around).
    """
    step = done / total if total > 0 else 0.0
    if outer_total:
        try:
            outer_step = (int(outer_index or 0) + step) / int(outer_total)
        except (TypeError, ValueError):
            outer_step = step
    else:
        outer_step = step
    if chain:
        if stage == "encode":
            return 0.04
        if stage == "stage1":
            return 0.05 + 0.45 * step
        if stage == "stage1_denoise":
            return 0.05 + 0.45 * outer_step
        if stage == "tile":
            return 0.50 + 0.40 * step
        if stage == "stage2_denoise":
            return 0.50 + 0.40 * outer_step
        if stage == "decode":
            return 0.95
        return None
    # Single generate: 0.05 is emitted before dispatch and 0.90/1.0 after the
    # terminal done (unchanged); the denoise steps fill the space between.
    if stage == "encode":
        return 0.06
    if stage == "stage1_denoise":
        return 0.06 + 0.44 * step
    if stage == "stage2_denoise":
        return 0.50 + 0.35 * step
    return None


# (current_step, total_steps, progress 0..1[, stage]). ``stage`` (F2, additive)
# names the pipeline phase of per-step events ("stage1_denoise" /
# "stage2_denoise" / "encode" / coarse chain stages); callbacks MUST declare it
# with a None default — mock-backend milestone calls pass only 3 args.
# Chain stage-1 events ADDITIONALLY pass ``clip=``/``clip_count=`` keywords
# (1-based segment position / segment total, from the worker's outer_index /
# outer_total); they are only passed when known, so callbacks MUST declare them
# with None defaults too (all other events keep the exact pre-existing calls).
ProgressCallback = Callable[..., None]

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

    def load(self, selection: dict[str, str] | None = None) -> None:
        """Load the pipeline. ``selection`` (model management, additive) maps a
        category (services.model_registry.CATEGORIES) to an ABSOLUTE weight
        path; absent categories / a None selection use the config defaults, so
        the legacy no-argument call is byte-identical to before."""
        if self._backend is not None and self._backend.loaded:
            return
        if self._backend is None:
            self._backend = self._select_backend()
        self._backend.load(selection=selection)

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
        seed: int | None = None,
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
            seed=seed,
        )

    def generate_chain(
        self,
        chain_request,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
        clip0_conditioning_paths: list[Path] | None = None,
        source_tail_path: Path | None = None,
        source_context_frames: int | None = None,
        source_audio_path: Path | None = None,
        lora_paths: list[tuple[Path, float, str]] | None = None,
        seed: int | None = None,
    ) -> GenerationOutcome:
        """Masked AV-latent clip chain -> ONE continuous output.mp4 (Phase 3 WP4).

        ``source_tail_path`` / ``source_context_frames`` (V2V continuation,
        additive): when set, the fps-correct source tail is frozen as clip-0's
        head and the delivered mp4 is the NEW part only (both backends).

        ``source_audio_path`` (A2V, additive): when set, the uploaded audio is
        frozen as the chain's audio latent and its original waveform is muxed onto
        the output; the terminal ``chain.a2v`` sub-dict pins the contract. Mutually
        exclusive with ``source_tail_path`` (enforced at the API layer).

        ``lora_paths`` (style/character IC-LoRA, additive): resolved
        ``(path, strength, preprocess)`` triples applied uniformly across the whole
        chain (every clip / stage). Empty/None -> no loras (byte-identical default);
        the mock ignores them, the real backend forwards them to the worker.
        """
        if self._backend is None or not self._backend.loaded:
            self.load()
        assert self._backend is not None
        return self._backend.generate_chain(
            chain_request,
            output_dir=output_dir,
            progress_callback=progress_callback,
            clip0_conditioning_paths=clip0_conditioning_paths,
            source_tail_path=source_tail_path,
            source_context_frames=source_context_frames,
            source_audio_path=source_audio_path,
            lora_paths=lora_paths,
            seed=seed,
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
        # Model management (tests/observability): the selection passed to the
        # last actual load, and a load counter (a swap = unload + load bumps
        # it; a no-op does not). The mock has no weights to swap — it only
        # records that the selection plumbing reached the backend.
        self.last_selection: dict[str, str] | None = None
        self.load_calls = 0

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self, selection: dict[str, str] | None = None) -> None:
        if self._loaded:
            return
        self.last_selection = dict(selection) if selection else None
        self.load_calls += 1
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
        seed: int | None = None,
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

        seed = int(seed) if seed is not None else resolve_seed(request.seed)
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
        source_tail_path: Path | None = None,
        source_context_frames: int | None = None,
        source_audio_path: Path | None = None,
        lora_paths: list[tuple[Path, float, str]] | None = None,
        seed: int | None = None,
    ) -> GenerationOutcome:
        """Simulate a masked AV-latent chain: ONE synthetic mp4 of the full
        timeline length + junction metadata (from :mod:`chain_math`). GPU-free;
        exercises the app-side orchestrator/metadata without model weights.

        ``lora_paths`` (style/character IC-LoRA, additive) is accepted and ignored
        — the mock has no weights to patch; the real forward-time patch lives in
        the engine worker (mirrors :meth:`generate`).

        V2V continuation (``source_tail_path`` / ``source_context_frames``): mirror
        the engine geometry via ``compute_chain_layout(source_context_px=...)`` —
        the mock mp4 holds ``new_frames_px`` frames (the NEW part only, matching the
        engine's context trim) and ``chain.v2v`` carries the same key set the real
        worker emits so pytest can pin the contract without a GPU. The mock has no
        audio pipeline, so the audio-sample numerics are reported as 0.
        """
        if not self._loaded:
            self.load()

        chain = chain_request
        seed = int(seed) if seed is not None else resolve_seed(chain.seed)
        layout = chain_math.compute_chain_layout(
            [c.num_frames for c in chain.clips], chain.frame_rate,
            kv=chain.overlap_frames,
            source_context_px=source_context_frames,
        )
        gpu_info.reset_peak_vram()
        if progress_callback:
            progress_callback(None, None, 0.05)

        start_image: Image.Image | None = None
        clip0_conditioning_paths = clip0_conditioning_paths or []
        if chain.clips[0].conditioning_images and clip0_conditioning_paths:
            start_image = Image.open(clip0_conditioning_paths[0]).convert("RGB")
            start_image = start_image.resize((chain.width, chain.height))

        # V2V: the delivered mp4 is the NEW part only (context trimmed off front).
        n_out = layout.new_frames_px if source_context_frames is not None else layout.total_px
        frames = self._render_chain_frames(
            width=chain.width, height=chain.height, n=n_out,
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

        chain_metadata = layout.to_dict()
        if source_context_frames is not None:
            source_had_audio = bool(
                source_tail_path is not None and video_io.has_audio_stream(source_tail_path)
            )
            # Mirror the worker's merged chain.v2v key set (engine done.chain.v2v):
            # geometry from ChainLayout + runtime fields. The mock has no audio
            # decode, so trimmed_audio_samples / audio_fade_in_samples are 0.
            # Placeholder audio-handle sidecar (synthetic SILENCE — the mock has no
            # audio pipeline). Mirrors the real engine's sidecar so pytest can pin
            # the contract (existence + metadata keys) without a GPU: the wav holds
            # `decoded_frames_px/frame_rate` seconds of 48kHz mono int16 zeros, with
            # the notional junction at `handle_context_seconds` = trim_px/frame_rate.
            handle_context_seconds = float(layout.trim_px) / float(chain.frame_rate)
            audio_handle_filename = self._write_placeholder_handle_wav(
                output_dir / "output_audio_handle.wav",
                total_frames=int(layout.total_px),
                frame_rate=float(chain.frame_rate),
            )
            v2v = dict(chain_metadata.get("v2v", {}))
            v2v.update({
                "context_frames": int(source_context_frames),
                "n_ctx_v": int(layout.n_ctx_v),
                "n_ctx_a": int(layout.n_ctx_a),
                "freeze_ka": int(layout.n_ctx_a) if source_had_audio else 0,
                "trimmed_px": int(layout.trim_px),
                "trimmed_audio_samples": 0,
                "audio_fade_in_samples": 0,
                "source_had_audio": source_had_audio,
                # Mock has no partial-availability audio decode: it either freezes
                # the full n_ctx_a (source has audio) or nothing (it doesn't), so
                # audio_head_frozen == source_had_audio here — but the key is kept
                # distinct to mirror the real engine's done-dict shape exactly.
                "audio_head_frozen": source_had_audio,
                "new_frames_px": int(layout.new_frames_px),
                "decoded_frames_px": int(layout.total_px),
                "v2v_context_junction_px": layout.v2v_context_junction_px,
                "audio_handle_filename": audio_handle_filename,
                "handle_context_seconds": round(handle_context_seconds, 6),
            })
            chain_metadata["v2v"] = v2v

        # A2V: mirror the engine's chain.a2v sub-dict (geometry from ChainLayout +
        # a ffprobe of the uploaded audio). The mock does NOT decode/mux audio, so
        # the output mp4 has no audio — but the metadata contract (key set +
        # geometry) is pinned so pytest can assert it without a GPU. Source-less
        # and V2V paths never set it, so their metas are unchanged.
        if source_audio_path is not None:
            a_total = int(layout.a_total)
            sr, channels = video_io.probe_audio_stream(source_audio_path)
            duration = video_io.probe_duration(source_audio_path)
            sr = sr or 16000
            channels = channels or 2
            available = (
                round(duration * chain_math.AUDIO_LATENTS_PER_SEC)
                if duration is not None
                else a_total
            )
            n_mux = int(round(layout.total_px / float(chain.frame_rate) * sr))
            chain_metadata["a2v"] = {
                "source_audio_path": str(source_audio_path),
                "a_total": a_total,
                "encoded_audio_frames_available": int(available),
                "muxed_original_waveform": True,
                "vocoder_skipped": True,
                "muxed_audio_samples": n_mux,
                "audio_sampling_rate": int(sr),
                "audio_channels": int(channels),
            }

        return GenerationOutcome(
            output_path=output_path,
            seed_used=seed,
            peak_vram_mb=peak,
            generation_mode="chain",
            backend=MOCK_BACKEND,
            chain_metadata=chain_metadata,
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

    @staticmethod
    def _write_placeholder_handle_wav(
        path: Path, *, total_frames: int, frame_rate: float, sr: int = 48000
    ) -> str:
        """Write a synthetic-SILENCE 48kHz mono int16 wav standing in for the real
        engine's audio-handle sidecar (the mock has no audio decode). Length =
        ``total_frames / frame_rate`` seconds so the sample geometry (junction at
        trim_px/frame_rate) is plausible. Returns the basename for metadata."""
        import wave

        n_samples = int(round(total_frames / float(frame_rate) * sr))
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(sr)
            wf.writeframes(b"\x00\x00" * n_samples)
        return path.name

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
                    # Non-protocol stdout (a stray library/tqdm print that escaped
                    # the worker's STDERR routing). Previously dropped silently;
                    # now surfaced at DEBUG so it is recoverable when diagnosing a
                    # wedged worker, without flooding the default INFO console.
                    # Lazy %-formatting means no cost unless DEBUG is enabled, so
                    # even a tqdm bar spamming stdout stays cheap and quiet here.
                    if line:
                        logger.debug("ignored non-protocol worker stdout: %s", line)
                    continue

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

    # Model-management selection: category -> the worker-payload field it
    # overrides. The four swappable categories (Docs/MODEL_MANAGEMENT_DESIGN.md
    # §0); every other payload field always comes from config.
    _SELECTION_FIELDS = {
        "transformer": "gguf_transformer_path",
        "text_encoder": "gguf_gemma_path",
        "video_vae": "component_video_vae_path",
        "audio": "component_audio_vae_path",
    }

    def _build_load_payload(self, selection: dict[str, str] | None = None) -> dict:
        """Build the ``{"op": "load"}`` worker payload (pure — no process I/O).

        ``selection`` maps a model-management category to an ABSOLUTE weight
        path, already resolved + prechecked by the API layer. Categories absent
        from ``selection`` (or a ``None``/empty selection) resolve from the
        config default fields exactly as before, so the no-selection payload is
        BYTE-IDENTICAL to the pre-model-management payload. Key set AND
        insertion order are part of that contract, pinned by the golden
        snapshot in tests/test_model_swap_load.py — do not reorder.
        """
        selection = selection or {}
        model = self.config.model

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

        def _swappable(category: str, field: str) -> str:
            override = selection.get(category)
            if override:
                return str(override)
            return self._require_path(getattr(model, field), field)

        gguf_transformer_path = _swappable("transformer", "gguf_transformer_path")
        gguf_gemma_path = _swappable("text_encoder", "gguf_gemma_path")

        # Component-file re-sourcing. Fixed on in config.yaml; the 3 standalone
        # files replace the monolith for VAE/audio (+ text projection connectors),
        # so they are load-bearing and always validated for existence.
        component_video_vae_path = _swappable("video_vae", "component_video_vae_path")
        component_audio_vae_path = _swappable("audio", "component_audio_vae_path")
        component_text_projection_path = self._require_path(
            model.component_text_projection_path, "component_text_projection_path"
        )

        return {
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
            "gguf_per_layer_quant": bool(model.gguf_per_layer_quant),
            "block_swap_blocks_on_gpu": self.low_vram.block_swap_blocks_on_gpu or 8,
            "vae_spatial_tile_size": int(self.low_vram.vae_spatial_tile_size),
            "vae_temporal_tile_size": int(self.low_vram.vae_temporal_tile_size),
        }

    def load(self, selection: dict[str, str] | None = None) -> None:
        if self.loaded:
            return

        model = self.config.model

        # Resolve + validate the engine python and worker script.
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

        # Full worker payload: validates the model paths and applies any
        # model-management selection overrides (byte-identical when absent).
        payload = self._build_load_payload(selection)

        use_component_files = bool(self.config.vram.use_component_files)

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

        logger.info(
            "Loading pipeline (REAL worker). python=%s engine_dir=%s block_swap=%s ckpt=%s",
            engine_python,
            engine_dir,
            payload["block_swap_blocks_on_gpu"],
            payload["checkpoint_path"],
        )
        if selection:
            logger.info("Model-management overrides: %s", selection)

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
            self._send(payload)
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
        seed: int | None = None,
    ) -> GenerationOutcome:
        if not self.loaded:
            self.load()

        conditioning_image_paths = conditioning_image_paths or []
        lora_paths = lora_paths or []
        mode = request.generation_mode

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "output.mp4"

        # Resolve the seed IN THE PARENT so seed_used is deterministic regardless
        # of the worker. When the caller (pipeline_manager) already resolved it —
        # so it could log the real value at job start — that value is used as-is;
        # a direct caller that passes nothing resolves here exactly as before.
        seed = int(seed) if seed is not None else resolve_seed(request.seed)

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
        # control signal by the worker per ``preprocess``). The reference
        # conditioning ``strength`` defaults to 1.0 (official guidance) but is
        # overridable per request via ``reference_video_strength``; None when no
        # loras. ``preprocess`` is derived from the job's loras -- a conflict (>1
        # distinct kind) is rejected up front by api/generate.py already, this is
        # the defensive re-check at the runner. When the request carries a
        # ``conditioning_attention_strength`` an ``attention_strength`` key is
        # spliced in (control-adherence override); it is entirely absent
        # otherwise so an omitted-field job's payload stays byte-identical.
        loras_payload = [{"path": str(p), "strength": float(s)} for p, s, _pp in lora_paths]
        preprocess = _resolve_reference_preprocess(lora_paths)
        if reference_video_path is not None:
            ref_strength = (
                1.0
                if request.reference_video_strength is None
                else float(request.reference_video_strength)
            )
            reference_payload = {
                "path": str(reference_video_path),
                "strength": ref_strength,
                "preprocess": preprocess,
            }
            if request.conditioning_attention_strength is not None:
                reference_payload["attention_strength"] = float(
                    request.conditioning_attention_strength
                )
        else:
            reference_payload = None

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
        # F2: the worker now streams per-step ``progress`` events during a
        # single generate too, so read through them (same receipt loop as the
        # chain) instead of the old one-shot _read_event() — which turned the
        # first progress event into "unexpected event" and failed the job.
        with self._lock:
            try:
                self._send(payload)
            except Exception as exc:
                raise RuntimeError("LTX worker died: " + self._stderr_tail()) from exc
            event = self._read_worker_events(progress_callback, chain=False, prefix="generate")

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
        source_tail_path: Path | None = None,
        source_context_frames: int | None = None,
        source_audio_path: Path | None = None,
        lora_paths: list[tuple[Path, float, str]] | None = None,
        seed: int | None = None,
    ) -> GenerationOutcome:
        """Masked AV-latent clip chain via the worker's ``generate_chain`` op.

        ONE worker invocation runs the whole chain (latents resident across
        segments) and writes ONE mp4. Progress events (per stage-1 segment, per
        stage-2 tile, decode) are streamed to ``progress_callback``; the terminal
        ``done`` carries peak VRAM + junction metadata.

        V2V continuation: when ``source_tail_path`` is set, an additive ``source``
        block ({path, context_frames}) is added to the worker payload (the app has
        already cut the fps-correct tail). The worker's ``done.chain`` then carries
        the ``v2v`` sub-dict, returned as-is in ``chain_metadata``.

        Style/character IC-LoRA: when ``lora_paths`` is non-empty an additive
        ``loras`` block ([{path, strength}, ...]) is added to the worker payload
        (mirrors the single-generate ``loras_payload``). The strengths apply
        uniformly to every clip/stage. Absent for a no-lora chain (payload
        byte-identical to before); the worker clears any stale LoRA regardless.
        """
        if not self.loaded:
            self.load()

        chain = chain_request
        clip0_conditioning_paths = clip0_conditioning_paths or []
        lora_paths = lora_paths or []
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "output.mp4"

        seed = int(seed) if seed is not None else resolve_seed(chain.seed)

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
        # V2V continuation (additive): the app-cut fps-correct source tail. Absent
        # for a normal chain (payload byte-identical to before).
        if source_tail_path is not None and source_context_frames is not None:
            payload["source"] = {
                "path": str(source_tail_path),
                "context_frames": int(source_context_frames),
            }
        # A2V continuation (additive): the uploaded audio path, passed as-is (the
        # engine truncates to the timeline). Absent for a normal / V2V chain.
        if source_audio_path is not None:
            payload["audio_source"] = {"path": str(source_audio_path)}
        # Style/character IC-LoRA (additive): (path, strength) per adapter, applied
        # uniformly across the chain. Only added when non-empty so a no-lora chain
        # payload is byte-identical to before (the worker parses msg.get("loras",
        # []) and clears stale LoRA either way). preprocess is dropped — control
        # adapters are rejected at the API layer, so every entry here is style.
        if lora_paths:
            payload["loras"] = [
                {"path": str(p), "strength": float(s)} for p, s, _pp in lora_paths
            ]

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
        ``progress`` events to ``progress_callback`` and emit rate-limited INFO
        lines per stage (S2 console progress). See :meth:`_read_worker_events`."""
        return self._read_worker_events(progress_callback, chain=True, prefix="chain")

    def _read_worker_events(
        self,
        progress_callback: ProgressCallback | None,
        *,
        chain: bool,
        prefix: str,
    ) -> dict:
        """Shared event-receipt loop for ``generate`` and ``generate_chain``.

        Reads framed worker events until the first non-``progress`` event
        (``done``/``error``/anything unexpected) and returns it — the caller
        keeps its existing terminal-event validation. Each ``progress`` event:

        * per-step denoise stages (F2 tqdm shim): step/total pass through to
          ``progress_callback(current_step, total_steps, frac, stage)``;
        * coarse stages (chain segment/tile/decode, encode): forwarded as
          ``(None, None, frac, stage)`` — same fractions as before F2;
        * chain stage-1 events (per-step ``stage1_denoise`` with a segment
          position, and the coarse per-segment ``stage1``) additionally pass
          ``clip=``/``clip_count=`` keywords (1-based clip being denoised /
          clip total) so the job store can surface "clip n/N" to the GUI. The
          keywords are only added when the position is known — every other
          event keeps its exact pre-existing call shape;
        * the fraction is monotone non-decreasing across the whole read (clamped
          against the last emitted value), and unknown stages never move it;
        * a rate-limited INFO line per stage keeps the console readable
          (worker-measured it/s preferred when present).
        """
        stage_state: dict[str, dict] = {}
        last_frac = 0.0
        while True:
            event = self._read_event()
            if event.get("event") != "progress":
                return event
            stage = event.get("stage")
            idx = int(event.get("index", 0))
            total = max(1, int(event.get("total", 1)))
            outer_index = event.get("outer_index")
            outer_total = event.get("outer_total")
            it_s = event.get("it_s")
            is_step = stage in _STEP_STAGES
            done = idx if is_step else idx + 1

            label, unit = _CHAIN_STAGE_LABELS.get(stage or "", (stage or "?", "unit"))
            key = stage or ""
            if is_step and outer_index is not None and outer_total:
                # Per-segment/tile timing + "started" line reset each outer unit.
                key = f"{stage}:{outer_index}"
                label = f"{label} [{int(outer_index) + 1}/{int(outer_total)}]"
            _log_stage_progress(prefix, label, unit, done, total, stage_state, key, it_s=it_s)

            frac = _progress_frac(stage, done, total, outer_index, outer_total, chain=chain)
            if frac is None:
                frac = last_frac
            frac = max(last_frac, min(1.0, frac))
            last_frac = frac

            # Chain clip position (ADDITIVE): stage-1 events carry which clip
            # (= stage-1 segment) is being worked on. Per-step events name it
            # via the shim's outer position; the coarse per-segment event fires
            # when segment idx+1 has just completed. Keywords are only passed
            # when known, so non-stage-1 events keep their exact old call shape
            # (callbacks declare clip/clip_count with None defaults).
            clip_kwargs: dict[str, int] = {}
            if chain:
                if stage == "stage1_denoise" and outer_total:
                    try:
                        clip_kwargs = {
                            "clip": int(outer_index or 0) + 1,
                            "clip_count": int(outer_total),
                        }
                    except (TypeError, ValueError):
                        clip_kwargs = {}
                elif stage == "stage1":
                    clip_kwargs = {"clip": min(idx + 1, total), "clip_count": total}

            if progress_callback:
                if is_step:
                    progress_callback(idx, total, round(frac, 3), stage, **clip_kwargs)
                else:
                    progress_callback(None, None, round(frac, 3), stage, **clip_kwargs)


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
