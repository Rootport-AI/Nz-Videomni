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
from api.errors import lora_preprocess_conflict, model_incompatible
from api.models import GenerateRequest
from config import AppConfig
from services import gpu_info, video_io
from services.base_models import BaseModelDescriptor, load_base_models
from services.lora_registry import ResolvedLora
from services.low_vram import LowVramSettings, safe_memory_cleanup

logger = logging.getLogger("ltx.runner")

#: Model-management category -> the worker load-payload field it overrides.
#: THE mapping, not a copy of one: ``_RealBackend._build_load_payload`` reads
#: this dict to place both the descriptor's ``default_file`` and any selection
#: override, so a category that is not here cannot reach the worker at all.
#: Pinned to the base-model descriptor by a contract test (the descriptor's
#: category set must equal this key set), which is what keeps a new category in
#: a manifest from silently doing nothing.
SELECTION_FIELDS: dict[str, str] = {
    "transformer": "gguf_transformer_path",
    "text_encoder": "gguf_gemma_path",
    "video_vae": "component_video_vae_path",
    "audio": "component_audio_vae_path",
}

#: Fixed (non-selectable) descriptor assets this engine needs, in payload order.
#: ``component_video_vae_pruned_path`` is deliberately NOT here — it is optional
#: (see ``_build_load_payload``) while these three must exist.
REQUIRED_ASSETS: tuple[str, ...] = (
    "gemma_root",
    "spatial_upsampler_path",
    "component_text_projection_path",
)

#: LTX generation this adapter can actually run, as the first two segments of
#: the transformer GGUF's ``model_version`` KV ("2.3.0" -> "2.3"). The LTX 2.5
#: inference path is the NEXT stage (PENDING_TASKS §3-98); until it exists, a
#: 2.5 weight file must fail loud at load time instead of being handed to a
#: worker that would mis-run it.
SUPPORTED_MODEL_VERSIONS: frozenset[str] = frozenset({"2.3"})

#: ``general.architecture`` value of every LTX weight file (2.3 and 2.5 alike).
LTX_ARCHITECTURE = "ltxv"


def _minor_version(version: str) -> str:
    """``"2.5.0"`` -> ``"2.5"`` (the patch segment never selects an engine)."""
    return ".".join(version.strip().split(".")[:2])


def check_kv(category: str, name: str, kv: dict[str, str]) -> None:
    """Rule on the GGUF KV metadata of a model about to be loaded (§2.2).

    The two-step contract, judged ONLY for the ``transformer`` category (the
    file that defines the generation; VAEs and text encoders carry no such
    stamp and are not gated here):

    1. ``general.architecture`` — the engine FAMILY. Anything other than
       ``ltxv`` is a different model lineage entirely and is refused (422).
    2. ``model_version`` — the LTX generation. Only
       :data:`SUPPORTED_MODEL_VERSIONS` can be run today; a newer one is
       refused with a message that names the next stage rather than pretending
       the file is broken.

    A MISSING key is a WARNING, not a refusal: both keys are present in every
    file the project's own converter produces, but a hand-made or third-party
    GGUF may lack them, and rejecting all of those would be a bigger regression
    than letting the engine's own loader have the last word (design §2.5).

    ``kv`` comes from ``services.model_registry.precheck_model_file`` — the
    header was already read there, so this function does no file I/O.
    """
    if category != "transformer":
        return
    architecture = (kv.get("general.architecture") or "").strip()
    if not architecture:
        logger.warning(
            "model '%s' declares no general.architecture in its GGUF header; "
            "loading it anyway (the engine's own loader has the last word).",
            name,
        )
    elif architecture != LTX_ARCHITECTURE:
        raise model_incompatible(
            category,
            name,
            detail=(
                f"'{architecture}'系のモデルです。LTXエンジンは"
                f"'{LTX_ARCHITECTURE}'のみ扱えます。"
            ),
        )
    version = (kv.get("model_version") or "").strip()
    if not version:
        logger.warning(
            "model '%s' declares no model_version in its GGUF header; loading "
            "it anyway (assuming it matches this engine's LTX generation).",
            name,
        )
        return
    if _minor_version(version) not in SUPPORTED_MODEL_VERSIONS:
        raise model_incompatible(
            category,
            name,
            detail=(
                f"このtransformerはltxv {version}です。LTX 2.5エンジンは次段階"
                "(PENDING_TASKS §3-98)で実装予定のため、まだ読み込めません。"
            ),
        )


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


def _resolve_reference_preprocess(lora_paths: list[ResolvedLora]) -> str:
    """Phase C: derive the single control-preprocess kind for the one reference
    video from the resolved loras of a job.

    ``lora_paths`` entries are ``ResolvedLora`` (``path``, ``strength``,
    ``preprocess``, ``audio_strength``; see
    ``services.lora_registry.LoraRegistry.resolve``) — index access (``lp[2]``)
    so plain 3-tuples from legacy/test call sites are still accepted. All-
    ``"none"`` (Phase B reference-only adapters, or no loras) -> ``"none"``.
    Exactly one non-``"none"`` kind -> that kind. More than one distinct kind is
    a conflict: a single uploaded reference video can only be converted into ONE
    control signal, so this raises ``LORA_PREPROCESS_CONFLICT`` (400) -- the
    same check the API layer (``api/generate.py``) already performs up front;
    this is the defensive re-check at the runner hop.
    """
    kinds = {lp[2] for lp in lora_paths if lp[2] != "none"}
    if len(kinds) > 1:
        raise lora_preprocess_conflict(sorted(kinds))
    return next(iter(kinds)) if kinds else "none"


def _lora_payload_entry(lp) -> dict:
    """One worker-payload lora dict: ``{"path", "strength"}`` plus
    ``"audio_strength"`` when the resolved entry carries one.

    Index access (``lp[0]``/``lp[1]``) + ``getattr(lp, "audio_strength", None)``
    so a plain 3-tuple (legacy/test call sites, no ``audio_strength`` field at
    all) still works — only a real ``ResolvedLora`` with a non-None
    ``audio_strength`` adds the key, keeping a no-audio job's payload
    byte-identical to before.
    """
    entry = {"path": str(lp[0]), "strength": float(lp[1])}
    audio_strength = getattr(lp, "audio_strength", None)
    if audio_strength is not None:
        entry["audio_strength"] = float(audio_strength)
    return entry


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
    # Acceleration: the attention backend the engine ACTUALLY ran with
    # ("sdpa" | "sage" | "sage->sdpa" when it fell back). Reported by the
    # worker's terminal ``done`` event and carried to metadata.json exactly like
    # ``seed_used``, so "the request said sage" and "sage actually ran" can never
    # silently diverge (the fp8 "displayed but not applied" trap). None on the
    # mock backend and on any worker that predates the field.
    attention_used: str | None = None
    # Acceleration: whether block-swap prefetch ACTUALLY ran ("off" | "on" |
    # "on->off" when it fell back to the synchronous path). Reported by the
    # worker's terminal ``done`` event and carried to metadata.json exactly like
    # ``attention_used``. None on the mock backend and on any worker that
    # predates the field.
    block_swap_prefetch_used: str | None = None
    # Acceleration: whether the cross-job CPU-skeleton cache ACTUALLY stayed
    # resident for this job ("off" | "on" | "on->off" when a worker-side guard
    # auto-downgraded it — see engine/worker.py's ``_resolve_keep_resident``).
    # Same relay discipline as ``block_swap_prefetch_used``: the worker's
    # terminal ``done`` event carries it into metadata.json, which is the ONLY
    # way to tell "the request asked for it" from "it actually ran" without
    # reading worker logs. None on the mock backend and on any worker that
    # predates the field.
    keep_resident_used: str | None = None
    # Acceleration: whether the fused Triton GGUF dequantization kernel ACTUALLY
    # ran for this job ("off" | "on" | "on->off" when it was requested but never
    # applied — Triton unavailable, a kernel exception latched the fallback, the
    # first-call self-check mismatched, or no eligible tensor existed). Same
    # relay discipline as ``block_swap_prefetch_used``. None on the mock backend
    # and on any worker that predates the field.
    fused_gguf_dequant_kernel_used: str | None = None
    # Acceleration: which video VAE decoder ACTUALLY ran for this job ("off" =
    # the stock decoder, "on" = the pruned PrunaVAED one, "on->off" when it was
    # requested but its weight file was absent so the stock decoder ran). Same
    # relay discipline as ``block_swap_prefetch_used``, but note this is the one
    # acceleration field whose "on" CHANGES THE PIXELS — which is exactly why
    # recording what actually ran matters here more than anywhere else. None on
    # the mock backend and on any worker that predates the field.
    vae_mode_used: str | None = None
    # Acceleration: torch.cuda.max_memory_reserved() in MB, reported alongside
    # peak_vram_mb (which is max_memory_allocated-based and cannot see
    # allocator-reserved-but-unallocated growth from stream-separate pools).
    # Additive — does not replace peak_vram_mb. None on the mock backend and on
    # any worker that predates the field.
    peak_vram_reserved_mb: int | None = None


class LTXRunner:
    """Facade around the LTX pipeline; delegates to a mock or real backend.

    Public contract (unchanged): ``LTXRunner(config, low_vram)``, properties
    ``loaded`` / ``pipeline_type``, methods ``load`` / ``unload`` /
    ``generate(...) -> GenerationOutcome``.

    ``descriptor`` (keyword, additive) is the BASE MODEL this runner serves —
    where every fixed weight path now comes from (§3-97 P3b). The app injects
    the one it loaded at startup (``AppContext.base_models``); omitted, it is
    read lazily from ``config.manifest_dir`` so the standalone constructions
    (tests, outputs/ drivers) keep working unchanged. Switching base model at
    runtime is the API axis's job (P6), not this constructor's.
    """

    # ------------------------------------------------------------------ seams
    # §3-98 P3a. The nine (plus supporting) attributes below are THE override
    # points a sibling engine family subclasses this facade through; every one
    # of them holds the LTX 2.3 value here, so this class behaves exactly as it
    # did before they existed. Bound after both backend classes are defined
    # (they are declared further down this module) — see the assignment block at
    # the end of the file.

    #: The real backend class this family spawns. Also the holder of
    #: :data:`REQUIRED_ASSETS` and of the engine-dir / engine-python resolution
    #: this facade's availability probe consults, so a family declares those
    #: facts ONCE, on its backend.
    _REAL_BACKEND_CLS: type
    #: The GPU-less backend class. LTX 2.5 deliberately REUSES 2.3's — a second
    #: synthetic-clip renderer would be a copy with nothing to say (§3-98 plan,
    #: "やらない"). Only the label below differs.
    _MOCK_BACKEND_CLS: type
    #: ``GenerationOutcome.backend`` the mock reports. The one thing that must
    #: differ per family, so a metadata.json says WHICH engine's mock ran.
    _MOCK_BACKEND_LABEL: str = MOCK_BACKEND

    def __init__(
        self,
        config: AppConfig,
        low_vram: LowVramSettings,
        descriptor: BaseModelDescriptor | None = None,
    ):
        self.config = config
        self.low_vram = low_vram
        self._descriptor = descriptor
        self._backend: _MockBackend | _RealBackend | None = None
        # Acceleration capability probe result (see ``sage_available``); None
        # until the first read, then cached for the process lifetime.
        self._sage_probe_cache: bool | None = None

    @property
    def descriptor(self) -> BaseModelDescriptor:
        """The base-model descriptor backing this runner.

        Resolved lazily (never in ``__init__``) so merely constructing a runner
        — which several capability probes do — reads no files: a caller that
        only asks ``sage_available`` must not fail because the manifests are
        unreadable. The lazy default is the FIRST declared descriptor, matching
        ``ModelRegistry.default_base_model``.
        """
        if self._descriptor is None:
            self._descriptor = next(iter(load_base_models(self.config.manifest_dir).values()))
        return self._descriptor

    def set_descriptor(self, descriptor: BaseModelDescriptor) -> None:
        """Point this runner at ANOTHER base model (§3-97 P6).

        The backend captured its descriptor when it was constructed
        (``_RealBackend.__init__``) and builds every weight path in its load
        payload from it, so re-pointing the runner has to DISCARD the backend,
        not just swap a field — otherwise the next load would send the old base
        model's paths. Dropping it also re-runs :meth:`_select_backend`, which
        is correct: whether the REAL stack is available is a question about the
        new base model's files, not the old one's.

        Any live pipeline is unloaded first (a base-model change is by
        definition a worker rebuild). Re-pointing at the base model already in
        effect is a cheap no-op — it must not tear down a loaded worker.
        """
        if self._descriptor is not None and self._descriptor.id == descriptor.id:
            self._descriptor = descriptor
            return
        self.unload()
        self._descriptor = descriptor
        self._backend = None

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
        lora_paths: list[ResolvedLora] | None = None,
        reference_video_path: Path | None = None,
        seed: int | None = None,
        outpaint_source_path: Path | None = None,
    ) -> GenerationOutcome:
        """``outpaint_source_path`` (§1-13, additive): the ORIGINAL uploaded video
        for an outpainting job. ``reference_video_path`` already points at the
        green-padded canvas pipeline_manager built from it; this second path is
        what the engine reads the frozen-guidance AUDIO from, because the canvas
        is deliberately written video-only (see ``video_io.pad_green_mp4``).
        ``None`` for every non-outpaint job."""
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
            outpaint_source_path=outpaint_source_path,
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
        retake_window_path: Path | None = None,
        end_source_path: Path | None = None,
        end_source_context_frames: int | None = None,
        end_source_strength: float | None = None,
        lora_paths: list[ResolvedLora] | None = None,
        reference_video_path: Path | None = None,
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

        ``retake_window_path`` (retake / temporal inpainting, additive): the
        app-cut window mp4 (frame-exact, CFR, at the request fps — the engine
        never cuts or resamples). The glue-band sizes and the regenerate_audio
        flag are NOT separate arguments: they ride on ``chain_request.retake``,
        the same convention ``stage2_window`` uses. Mutually exclusive with both
        ``source_tail_path`` and ``source_audio_path`` (enforced at the API
        layer). None -> byte-identical to before, payload key set included.

        ``end_source_path`` / ``end_source_context_frames`` (end source,
        additive): the app-prepared ``_end_source.mp4`` (a cut video or a looped
        still — the engine only ever sees a video) plus the length of the tail
        band frozen from it. The file holds ``context_frames + 1`` frames: the
        extra leading frame is the causal VAE's primer and never reaches the
        output. The delivered length is UNCHANGED (unlike the V2V head, nothing
        is trimmed). Mutually exclusive with ``retake_window_path`` and
        ``source_audio_path``, combinable with ``source_tail_path`` (enforced at
        the API layer). None -> byte-identical to before, payload key set
        included.

        ``end_source_strength`` (additive, 0.0..1.0): softens ONLY stage 1's
        freeze of the band (stage 2 always hard-freezes regardless). None ->
        treated as 1.0, a hard freeze byte-identical to before this field
        existed.

        ``lora_paths`` (style/character IC-LoRA, additive): resolved
        ``ResolvedLora`` (``path``, ``strength``, ``preprocess``, ``audio_strength``)
        entries applied uniformly across the whole chain (every clip / stage).
        Empty/None -> no loras (byte-identical default); the mock ignores them,
        the real backend forwards them to the worker.

        ``reference_video_path`` (Phase C reference-video CONTROL IC-LoRA, ALPHA
        scope — clips=1 only, enforced by the schema/endpoint): mirrors
        :meth:`generate`'s ``reference_video_path``. None -> no reference (byte-
        identical default); the mock ignores it, the real backend forwards it to
        the worker.
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
            retake_window_path=retake_window_path,
            end_source_path=end_source_path,
            end_source_context_frames=end_source_context_frames,
            end_source_strength=end_source_strength,
            lora_paths=lora_paths,
            reference_video_path=reference_video_path,
            seed=seed,
        )

    # ----------------------------------------------------- backend selection

    def _select_backend(self) -> _MockBackend | _RealBackend:
        choice = (self.config.model.backend or "auto").strip().lower()
        if choice == "mock":
            logger.info("Backend forced to MOCK (model.backend=mock).")
            return self._new_mock()
        if choice == "real":
            if not self._real_available():
                raise RuntimeError(
                    "model.backend='real' but the real LTX stack is unavailable "
                    "(check torch+CUDA, ltx_pipelines install, and model paths)."
                )
            logger.info("Backend forced to REAL (model.backend=real).")
            return self._REAL_BACKEND_CLS(self.config, self.low_vram, self.descriptor)
        # auto
        if self._real_available():
            logger.info("Backend auto-selected: REAL.")
            return self._REAL_BACKEND_CLS(self.config, self.low_vram, self.descriptor)
        logger.info("Backend auto-selected: MOCK (real stack/model unavailable).")
        return self._new_mock()

    def _new_mock(self) -> _MockBackend:
        """The mock backend instance, labelled for THIS engine family."""
        return self._MOCK_BACKEND_CLS(
            self.config, self.low_vram, backend_label=self._MOCK_BACKEND_LABEL
        )

    def _real_available(self) -> bool:
        """True only if the engine python, worker script and every file the real
        GGUF + component-file path actually loads are present.

        WHICH files those are comes from the base-model descriptor: the four
        categories' ``default_file`` plus the three required ``assets``
        (:data:`REQUIRED_ASSETS`). ``component_video_vae_pruned_path`` is
        deliberately NOT gated — a job that asks for the pruned decoder
        downgrades to the stock one, so its absence must not demote the whole
        server to mock.

        The GGUF + component-file recipe never opens the 43GB monolith. The
        worker payload's ``checkpoint_path`` field is a hardcoded ``""`` (see
        ``_build_load_payload``) — the wheel's lazy builders receive it but the
        GGUF/component installs replace every loader — so there is no path here
        to gate. The (tokenizer-only ~40MB) ``gemma_root`` IS gated:
        DistilledPipeline is built with gemma_root=None so the wheel's weight
        glob is bypassed, but the engine still loads the tokenizer/processor
        module_ops from this dir, so a missing dir must fail fast in the app
        layer rather than crash deep in the encode path.

        A False answer is LOGGED WITH THE MISSING PATHS. This probe is the
        silent-mock-demotion trap: 'auto' falls back to the mock, generation
        keeps "working", and without this list nothing on screen says which
        file was the reason.

        Deliberately does NOT import torch / ltx_* (those live only in the engine
        venv, not the app venv). Any failure/missing is swallowed -> False (so
        'auto' falls back to mock and ``import services.ltx_runner`` stays safe
        in the torch-less app venv).
        """
        backend_cls = self._REAL_BACKEND_CLS
        try:
            descriptor = self.descriptor
            required: dict[str, str | None] = {
                backend_cls._ENGINE_PYTHON_LABEL: backend_cls._engine_python_value(self.config),
            }
            for category, spec in descriptor.categories.items():
                required[category] = (
                    self._models_path(spec.default_file) if spec.default_file else None
                )
            for asset in backend_cls.REQUIRED_ASSETS:
                value = descriptor.assets.get(asset)
                required[asset] = self._models_path(value) if value else None

            missing = [
                f"{label} (not declared by base model '{descriptor.id}')"
                if not path
                else f"{label}: {self.config._abs(path)}"
                for label, path in required.items()
                if not path or not self.config._abs(path).exists()
            ]
            engine_dir = backend_cls._engine_dir_value(self.config)
            if not engine_dir:
                missing.append("engine_dir (not configured)")
            else:
                worker = self.config._abs(engine_dir) / "worker.py"
                if not worker.exists():
                    missing.append(f"engine worker: {worker}")
            if missing:
                logger.warning(
                    "real backend unavailable — %d required file(s) missing: %s",
                    len(missing),
                    ", ".join(missing),
                )
                return False
            return True
        except Exception:
            logger.warning("real backend availability probe failed", exc_info=True)
            return False

    def _models_path(self, rel: str) -> str:
        """Descriptor-relative path -> a path ``config._abs`` can resolve.

        Same normalization as ``ModelRegistry._store_path``: descriptor paths
        are relative to ``model.models_dir``, never to the project root.
        """
        return (Path(self.config.model.models_dir) / rel).as_posix()

    # ------------------------------------------- acceleration (SageAttention)

    @property
    def sage_available(self) -> bool:
        """Pre-load capability probe: is SageAttention installed in the ENGINE venv?

        Same discipline as :meth:`_real_available` — a pure FILE-EXISTENCE check
        that NEVER imports anything. sageattention/triton live only in the engine
        venv (they pull in torch), so importing them here would break the
        torch-free app venv; and a failed import is not cached by Python, so the
        import route would also re-pay the cost on every ``GET /status`` poll.

        WINDOWS VENV LAYOUT is assumed, matching ``model.engine_python``'s own
        default (``./.venv-engine/Scripts/python.exe``): the interpreter's
        grandparent is the venv root and its packages live in
        ``<venv>/Lib/site-packages``. Both ``sageattention/`` and ``triton/`` are
        required — the sage kernels are Triton-backed, so sageattention alone is
        not usable.

        This answers "COULD sage run" before a worker exists. Once the worker is
        up, its own import-time probe (reported on the ``ready`` event) is
        authoritative — see ``_RealBackend.sage_available`` and
        ``PipelineManager.acceleration_status_block``.

        Evaluated ONCE and cached for the process lifetime: GET /status polls
        every 10s, and installing sageattention into the engine venv requires a
        server restart to take effect anyway.
        """
        if self._sage_probe_cache is None:
            self._sage_probe_cache = self._probe_sage_files()
        return self._sage_probe_cache

    def _probe_sage_files(self) -> bool:
        try:
            engine_python = self._REAL_BACKEND_CLS._engine_python_value(self.config)
            if not engine_python:
                return False
            venv_root = self.config._abs(engine_python).parent.parent
            site_packages = venv_root / "Lib" / "site-packages"
            return (
                (site_packages / "sageattention").is_dir()
                and (site_packages / "triton").is_dir()
            )
        except Exception:
            return False

    @property
    def worker_sage_available(self) -> bool | None:
        """The LOADED worker's OWN sage probe result, or None when unknown.

        None means "no loaded worker has told us anything" — no backend, an
        unloaded/dead backend, the mock (which has no engine), or a worker that
        predates the ``ready.sage_available`` field. Callers fall back to the
        file-existence :attr:`sage_available` in that case.
        """
        backend = self._backend
        if backend is None or not backend.loaded:
            return None
        value = getattr(backend, "sage_available", None)
        return None if value is None else bool(value)

    @property
    def is_mock(self) -> bool:
        """True when the mock backend is (or, before any load, would be) active.

        The mock has no engine at all, so it can never run sage regardless of
        what the engine venv contains. Before the first ``load()`` there is no
        backend instance yet, so the CONFIGURED choice decides; ``auto`` is
        deliberately not resolved here (resolving it means the full
        ``_real_available`` file sweep) — an auto install without the real stack
        also has no engine venv, so the file probe reports False anyway.
        """
        if isinstance(self._backend, self._MOCK_BACKEND_CLS):
            return True
        if self._backend is not None:
            return False
        return (self.config.model.backend or "auto").strip().lower() == "mock"


class _MockBackend:
    """Synthetic-clip backend (no GPU, no weights). Original Phase-1 logic.

    SHARED BY EVERY ENGINE FAMILY (§3-98 P3b). A synthetic gradient clip says
    nothing about which engine would have rendered it, so a second copy of this
    class for LTX 2.5 would be a copy with no content of its own. The ONE fact
    that must still differ is what ``GenerationOutcome.backend`` reports, so the
    label is a constructor argument (``backend_label``) rather than a hardcoded
    constant — an omitted argument keeps the historical ``"mock"``.
    """

    #: Class-level default for :attr:`backend_label`, so an instance built
    #: WITHOUT ``__init__`` (tests drive ``generate_chain`` on a hand-assembled
    #: ``__new__`` object) still reports the historical label instead of
    #: raising. ``__init__`` shadows it per instance.
    backend_label: str = MOCK_BACKEND

    def __init__(
        self,
        config: AppConfig,
        low_vram: LowVramSettings,
        *,
        backend_label: str = MOCK_BACKEND,
    ):
        self.config = config
        self.low_vram = low_vram
        #: ``GenerationOutcome.backend`` of every clip this instance renders.
        self.backend_label = backend_label
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
        lora_paths: list[ResolvedLora] | None = None,
        reference_video_path: Path | None = None,
        seed: int | None = None,
        outpaint_source_path: Path | None = None,
    ) -> GenerationOutcome:
        """Generate a synthetic video and return the outcome (output.mp4 + metrics).

        ``conditioning_images`` empty -> T2V; one entry -> minimal I2V using the
        resolved image path as the start frame (frame_idx=0, Phase 1).

        ``lora_paths`` (now ``ResolvedLora`` entries — ``path``, ``strength``,
        ``preprocess``, ``audio_strength``, Phase C/S1) /
        ``reference_video_path`` are the Phase B/C IC-LoRA inputs; the mock
        backend accepts (and ignores) them so the full route completes GPU-free —
        the real weight patch (and the preprocess -> control-signal conversion)
        lives in the engine worker.

        ``outpaint_source_path`` (§1-13) is accepted and ignored for the same
        reason: the mock never opens a video. It does honour the outpaint
        GEOMETRY though — ``request.width``/``height`` are already the canvas, so
        the placeholder comes out at the extended size and ``_render_frames``
        outlines where the source footage would have gone.
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
            backend=self.backend_label,
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
        retake_window_path: Path | None = None,
        end_source_path: Path | None = None,
        end_source_context_frames: int | None = None,
        end_source_strength: float | None = None,
        lora_paths: list[ResolvedLora] | None = None,
        reference_video_path: Path | None = None,
        seed: int | None = None,
    ) -> GenerationOutcome:
        """Simulate a masked AV-latent chain: ONE synthetic mp4 of the full
        timeline length + junction metadata (from :mod:`chain_math`). GPU-free;
        exercises the app-side orchestrator/metadata without model weights.

        ``lora_paths`` (style/character IC-LoRA, additive) is accepted and ignored
        — the mock has no weights to patch; the real forward-time patch lives in
        the engine worker (mirrors :meth:`generate`).

        ``reference_video_path`` (Phase C reference-video CONTROL IC-LoRA,
        additive) is likewise accepted and ignored — the mock has no weights to
        patch against the reference either.

        THE MOCK MP4'S LENGTH IS ALWAYS ``layout.new_frames_px`` (== ``total_px -
        trim_px``), for every chain shape — see the comment at the call site. So a
        V2V continuation is the NEW part only (matching the engine's context trim)
        and an end source's mp4 is LONGER than the clips by exactly the frozen
        band (the band is an internal segment appended after the clips).

        V2V continuation (``source_tail_path`` / ``source_context_frames``): mirror
        the engine geometry via ``compute_chain_layout(source_context_px=...)`` —
        ``chain.v2v`` carries the same key set the real worker emits so pytest can
        pin the contract without a GPU. The mock has no audio pipeline, so the
        audio-sample numerics are reported as 0.
        """
        if not self._loaded:
            self.load()

        chain = chain_request
        seed = int(seed) if seed is not None else resolve_seed(chain.seed)
        # Stage-2 window preset: resolved HERE too, not just in the real
        # backend. The mock's junction metadata comes from the same
        # ``compute_chain_layout``, so omitting this would make every mock-backed
        # test pass with standard-window geometry no matter what the request
        # asked for — a silent false green on the whole feature.
        v_tile, v_adv = chain_math.resolve_stage2_window(
            getattr(chain, "stage2_window", None)
        )
        # Retake: resolved HERE too, for the same reason as the stage-2 window
        # above — the mock's geometry metadata comes from the same
        # compute_chain_layout, so omitting it would let every mock-backed test
        # pass with plain-chain geometry no matter what the request asked for.
        retake = getattr(chain, "retake", None)
        # End source: likewise resolved HERE — the frozen tail band adds a whole
        # internal segment to the layout (and with it a junction, a stage-2 tile
        # boundary and the timeline's own length), so a mock that skipped it
        # would report a different geometry AND a shorter mp4 than the engine.
        end_source = getattr(chain, "end_source", None)
        layout = chain_math.compute_chain_layout(
            [c.num_frames for c in chain.clips], chain.frame_rate,
            kv=chain.overlap_frames,
            v_tile=v_tile, v_adv=v_adv,
            source_context_px=source_context_frames,
            retake_glue_px=(
                None if retake is None else (int(retake.head_px), int(retake.tail_px))
            ),
            end_context_px=end_source_context_frames,
        )
        gpu_info.reset_peak_vram()
        if progress_callback:
            progress_callback(None, None, 0.05)

        start_image: Image.Image | None = None
        clip0_conditioning_paths = clip0_conditioning_paths or []
        if chain.clips[0].conditioning_images and clip0_conditioning_paths:
            start_image = Image.open(clip0_conditioning_paths[0]).convert("RGB")
            start_image = start_image.resize((chain.width, chain.height))

        # The delivered length, for EVERY chain shape, is one expression:
        # ``total_px - trim_px``. ``chain_math`` already folds each feature into
        # one of those two terms, so there is nothing left to branch on here:
        #
        # * plain chain / retake — trim_px is 0 and total_px IS the deliverable
        #   (a retake's deliverable is its whole window);
        # * V2V — the frozen source head is trimmed off the FRONT, which is
        #   exactly trim_px;
        # * end source, "in_window" mode (ONE clip) and "reverse" mode (2+ clips)
        #   — the band is the LAST clip's own tail, so total_px IS the clips'
        #   total and the mp4 is exactly as long as the user asked for;
        # * end source, "internal_segment" mode (API-unreachable) — the band is a
        #   segment APPENDED after the user's clips, so total_px is already clips
        #   + band and the mp4 is correspondingly LONGER than the clips asked for.
        #
        # None of those is branched on here: chain_math folds the mode into
        # total_px, so the mock follows all of them automatically. That is also
        # the mock's own regression check — if it ever needs an ``if mode ==``,
        # the geometry has stopped being the single source of truth.
        #
        # ``layout.new_frames_px`` is that same subtraction, computed once in
        # chain_math so the validator, the mock and the engine cannot disagree.
        n_out = layout.new_frames_px
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

        # Retake: mirror the engine's chain.retake sub-dict. The GEOMETRY half is
        # already there (ChainLayout.to_dict()); this adds the runtime half with
        # the SAME key set the real engine emits — with ONE deliberate exception:
        # ``freeze_proof`` is absent. That number can only be produced by looking
        # at real latents, and the mock has none; emitting 0.0 would fabricate a
        # passing proof of the one thing this feature actually has to prove. Tests
        # assert the difference in both directions.
        if retake is not None:
            window_has_audio = bool(
                retake_window_path is not None
                and video_io.has_audio_stream(retake_window_path)
            )
            rt = dict(chain_metadata.get("retake", {}))
            rt.update({
                "regenerate_audio": bool(retake.regenerate_audio),
                "source_had_audio": window_has_audio,
                # The mock freezes nothing, but it reports what the engine WOULD
                # have frozen, so the contract shape stays checkable: a band only
                # exists when there is audio to put in it.
                "audio_frozen": bool(
                    window_has_audio
                    and (rt.get("n_head_a", 0) > 0 or rt.get("n_tail_a", 0) > 0)
                ),
                "muxed_original_waveform": bool(
                    not retake.regenerate_audio and window_has_audio
                ),
                "decoded_frames_px": int(layout.total_px),
            })
            chain_metadata["retake"] = rt

        # End source: mirror the engine's chain.end_source sub-dict. The GEOMETRY
        # half is already there (ChainLayout.to_dict()); this adds the runtime
        # half — with the SAME deliberate exception retake makes: ``freeze_proof``
        # is absent. That number can only be produced by comparing real latents
        # before and after the denoise, and the mock has none; emitting 0.0 would
        # fabricate a passing proof of the one thing this feature actually has to
        # prove. Tests assert the difference in both directions.
        if end_source_context_frames is not None:
            es = dict(chain_metadata.get("end_source", {}))
            es.update({
                "kind": (
                    "image"
                    if end_source is not None and end_source.image_id is not None
                    else "video"
                ),
                "cut_path": str(end_source_path) if end_source_path else None,
                "decoded_frames_px": int(layout.total_px),
                # Contract passthrough only — the mock has no latents to soften,
                # so it reports what was asked for rather than any observed
                # effect (freeze_proof stays absent, same reasoning as above).
                "strength": (
                    1.0 if end_source_strength is None else float(end_source_strength)
                ),
            })
            chain_metadata["end_source"] = es

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
            backend=self.backend_label,
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
            # Outpainting (§1-13): the clip is already the CANVAS size (width /
            # height ARE the canvas), so the only thing the placeholder has to
            # add is where the original footage would have sat — otherwise a
            # mock outpaint run is indistinguishable from a plain one and the
            # geometry could be wrong all the way to the real backend. The keep
            # rectangle is outlined in the sentinel green so the marker names
            # the feature it belongs to.
            if request.outpaint is not None:
                op = request.outpaint
                draw.rectangle(
                    [op.pad_left, op.pad_top, w - op.pad_right - 1, h - op.pad_bottom - 1],
                    outline=(102, 255, 0),
                    width=3,
                )
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

    # ------------------------------------------------------------------ seams
    # §3-98 P3a. Everything about this class that is a fact about the LTX 2.3
    # ENGINE rather than about "how to talk to a worker subprocess" is named
    # here, so a sibling family (engine25) inherits the process plumbing —
    # spawn, frame, read, unload — and restates only these. Every value below
    # is the 2.3 one, so this class behaves exactly as it did before the seams.

    #: Model-management category -> worker load-payload field. See the
    #: module-level :data:`SELECTION_FIELDS`, which this is THE binding of;
    #: ``_build_load_payload`` reads it through ``self`` so a subclass's table
    #: reaches the payload without re-implementing the builder.
    SELECTION_FIELDS: dict[str, str] = SELECTION_FIELDS
    #: Fixed (non-selectable) descriptor assets this engine requires. Read
    #: through the backend CLASS by ``LTXRunner._real_available`` too, so a
    #: family declares its required assets exactly once.
    REQUIRED_ASSETS: tuple[str, ...] = REQUIRED_ASSETS
    #: ``python -m <this>`` — the worker entry point inside the engine venv.
    _WORKER_MODULE: str = "engine.worker"
    #: Worker stderr log filename under ``config.log_dir``. Distinct per family
    #: on purpose: after a 2.3<->2.5 swap BOTH logs must survive for the
    #: round-trip gate to be checkable.
    _LOG_NAME: str = "ltx_worker.log"
    #: Fixed engine package directory for this family, or None to take
    #: ``config.model.engine_dir`` (2.3 keeps its configurable one).
    _ENGINE_DIR_VALUE: str | None = None
    #: The ``config.model`` key that names this family's interpreter — used
    #: verbatim in the "not configured / not found" messages, so an operator is
    #: told which key to fix.
    _ENGINE_PYTHON_LABEL: str = "engine_python"

    @classmethod
    def _engine_python_value(cls, config: AppConfig) -> str | None:
        """The interpreter that runs THIS family's worker."""
        return config.model.engine_python

    @classmethod
    def _engine_dir_value(cls, config: AppConfig) -> str | None:
        """The engine package directory of THIS family."""
        return cls._ENGINE_DIR_VALUE or config.model.engine_dir

    def __init__(
        self,
        config: AppConfig,
        low_vram: LowVramSettings,
        descriptor: BaseModelDescriptor,
    ):
        self.config = config
        self.low_vram = low_vram
        #: The base model whose ``default_file``s and ``assets`` this worker is
        #: built from (§3-97 P3b) — the only source of fixed weight paths.
        self.descriptor = descriptor
        self.pipeline = None  # back-compat attribute; always None for this backend
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._log_path: Path | None = None
        # Acceleration: the worker's OWN import-time SageAttention probe, taken
        # from the ``ready`` event. None until a worker reports it (and again
        # after unload) — see LTXRunner.worker_sage_available.
        self.sage_available: bool | None = None

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

    def _models_path(self, rel: str) -> str:
        """Descriptor-relative path -> a path ``config._abs`` can resolve
        (mirrors ``ModelRegistry._store_path``: relative to ``models_dir``)."""
        return (Path(self.config.model.models_dir) / rel).as_posix()

    def _require_models_file(self, rel: str | None, label: str) -> str:
        """Absolute path of a descriptor-declared file, which MUST exist.

        Same fail-fast contract as :meth:`_require_path` (the config-sourced
        sibling), phrased against the base model: an undeclared or absent file
        stops the load here, in the app layer, instead of crashing the worker's
        native loader minutes later.
        """
        if not rel:
            raise RuntimeError(
                f"base model '{self.descriptor.id}' declares no {label} "
                "(required for the real backend)."
            )
        resolved = self.config._abs(self._models_path(rel))
        if not resolved.exists():
            raise RuntimeError(f"base model '{self.descriptor.id}' {label} not found: {resolved}")
        return str(resolved)

    def _require_asset(self, key: str) -> str:
        """Absolute path of a fixed (non-selectable) descriptor asset."""
        return self._require_models_file(self.descriptor.assets.get(key), f"asset '{key}'")

    def _require_default_file(self, category: str) -> str:
        """Absolute path of a category's descriptor ``default_file``."""
        spec = self.descriptor.categories.get(category)
        return self._require_models_file(
            spec.default_file if spec else None, f"default file for category '{category}'"
        )

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

    def _build_load_payload(self, selection: dict[str, str] | None = None) -> dict:
        """Build the ``{"op": "load"}`` worker payload (pure — no process I/O).

        ``selection`` maps a model-management category to an ABSOLUTE weight
        path, already resolved + prechecked by the API layer. Categories absent
        from ``selection`` (or a ``None``/empty selection) resolve from the base
        model's ``default_file`` / ``assets`` (§3-97 P3b — they used to be fixed
        ``config.model`` fields holding the very same paths), so the
        no-selection payload is BYTE-IDENTICAL to the pre-model-management
        payload. Key set AND insertion order are part of that contract, pinned
        by the golden snapshot in tests/test_model_swap_load.py — do not
        reorder.

        Which category feeds which payload field is :data:`SELECTION_FIELDS`,
        read here rather than transcribed — there is no second table to drift.
        """
        selection = selection or {}
        model = self.config.model

        # checkpoint_path: hardcoded "" (2026-07-28, PENDING_TASKS.md 3-26; the
        # ModelConfig field this used to read no longer exists — see config.py's
        # NOTE on checkpoint_path for the full evidence chain). The GGUF +
        # component-file path never opens this string; it is forwarded only
        # because DistilledPipeline requires a non-None str so
        # ModelLedger.build_model_builders() populates the lazy builder objects
        # that the GGUF/component re-sourcing later overwrites via
        # dataclasses.replace(). "" satisfies that "not None" requirement and
        # keeps this payload byte-identical to every prior config state (no
        # config.yaml value ever set this to anything else in practice).
        checkpoint_path = ""
        # gemma_root (tokenizer-only ~40MB) IS load-bearing: DistilledPipeline is
        # built with gemma_root=None so the wheel's weight glob (model*.safetensors)
        # is bypassed, but the engine loads the tokenizer/processor
        # module_ops from this dir (tokenizer.model + preprocessor_config.json), so a
        # missing dir must fail fast here rather than crash deep in the encode path.
        # Forwarded to the worker as a payload field exactly as before.
        gemma_root = self._require_asset("gemma_root")

        upsampler_path = self._require_asset("spatial_upsampler_path")

        # Every SELECTABLE path in one pass, keyed by the payload field
        # SELECTION_FIELDS assigns to that category: an explicit selection wins,
        # otherwise the base model's own default_file. Reading the mapping here
        # (instead of transcribing it) is what makes the contract test between
        # SELECTION_FIELDS and the descriptor's category set meaningful — a
        # category missing from the mapping cannot reach the worker at all.
        # Component-file re-sourcing (video/audio VAE) rides the same route:
        # fixed on in config.yaml, the standalone files replace the monolith,
        # so they are load-bearing and always validated for existence.
        swapped = {
            field: (
                str(selection[category])
                if selection.get(category)
                else self._require_default_file(category)
            )
            for category, field in self.SELECTION_FIELDS.items()
        }
        component_text_projection_path = self._require_asset("component_text_projection_path")
        # PrunaVAED (pruned video VAE decoder, ~690MB, decoder half only):
        # resolved but NOT required to exist — deliberately not via
        # _require_models_file and not in the real-backend availability probe.
        # It is read only by a job that asks for vae_mode="prune_vaed", and a
        # missing file downgrades THAT job to the stock decoder
        # (vae_mode_used="on->off") instead of preventing the whole server from
        # loading.
        #
        # It is also deliberately NOT a registry category: it sits in a
        # SUBDIRECTORY (VAE/prunavaed/) and its filename contains neither
        # "video" nor "audio", which is a double defence against the video_vae
        # name_hint scan (services/model_registry.py) — a decoder-only file
        # offered as the server-wide video VAE would break the encoder-side
        # builder, and the scan is non-recursive so the subdirectory is out of
        # reach anyway.
        pruned_rel = self.descriptor.assets.get("component_video_vae_pruned_path")
        component_video_vae_pruned_path = (
            str(self.config._abs(self._models_path(pruned_rel))) if pruned_rel else ""
        )

        return {
            "op": "load",
            "checkpoint_path": checkpoint_path,
            "gemma_root": gemma_root,
            "upsampler_path": upsampler_path,
            "gguf_transformer_path": swapped["gguf_transformer_path"],
            "gguf_gemma_path": swapped["gguf_gemma_path"],
            # Phase 1 component-file paths (gate via LTX_COMPONENT_FILES env).
            "component_video_vae_path": swapped["component_video_vae_path"],
            "component_audio_vae_path": swapped["component_audio_vae_path"],
            "component_text_projection_path": component_text_projection_path,
            "component_video_vae_pruned_path": component_video_vae_pruned_path,
            "gguf_per_layer_quant": bool(model.gguf_per_layer_quant),
            "block_swap_blocks_on_gpu": self.low_vram.block_swap_blocks_on_gpu or 8,
            "vae_spatial_tile_size": int(self.low_vram.vae_spatial_tile_size),
            "vae_temporal_tile_size": int(self.low_vram.vae_temporal_tile_size),
        }

    def _build_child_env(self, project_root: Path) -> dict[str, str]:
        """The worker subprocess's environment (§3-98 P3a seam).

        Inherit, force the compile knob, unbuffered IO, and set PYTHONPATH to
        the project root so the worker's ``engine.*`` package (and the
        venv-installed ltx_core/ltx_pipelines) resolve when launched as
        ``python -m engine.worker``.

        A sibling engine family overrides this WHOLE method rather than editing
        the dict afterwards: every ``LTX_*`` variable below is a 2.3 knob read
        by 2.3's worker, and inheriting them into another engine's process
        would be exactly the borrowed-assumption bug the separate engine exists
        to avoid.
        """
        env = dict(os.environ)
        # INERT on this platform, kept only because LTX 2.3 is frozen. torch
        # refuses expandable_segments on Windows ("expandable_segments not
        # supported on this platform") and keeps the segmented caching allocator,
        # so this line has never changed anything here; torch 2.9 also deprecates
        # the variable's name in favour of PYTORCH_ALLOC_CONF. Measured
        # 2026-08-24, outputs/b4-vram-diag/probe_expandable.py. The comment above
        # used to call this a "16GB-load-bearing CUDA knob" — it is not, and the
        # 2.5 adapter has dropped its copy of the line. Removing it here too would
        # be equally behaviour-neutral; it stays because touching a frozen engine
        # for a no-op is not worth the regression surface.
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["TORCH_COMPILE_DISABLE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("PYTHONPATH", None)
        env["PYTHONPATH"] = str(project_root)
        # Phase 1 gate: the worker reads LTX_COMPONENT_FILES.
        env["LTX_COMPONENT_FILES"] = "1" if bool(self.config.vram.use_component_files) else "0"
        # NOTE (§48): the old ``LTX_KEEP_RESIDENT`` env var is GONE. Keep-resident
        # weights are now a PER-JOB request field (``GenerateRequest.keep_resident``)
        # carried on the generate payload, so there is exactly one source of truth
        # and the setting can be flipped without a 60-90s worker reload. The worker
        # creates the pipeline with keep_resident_weights=False unconditionally and
        # arms/disarms the registry per job. An LTX_KEEP_RESIDENT left over in
        # someone's environment is now inert (it is neither set nor read) — the
        # reproduction steps in §10.2/§46/§47 that export it are historical.
        # Sequential per-layer CPU offload of the GGUF Gemma during text-encode
        # (caps the ~15GB encode peak). On by default; the worker reads this and
        # keeps the 48 Gemma decoder layers CPU-resident, streaming them to GPU
        # per layer. Compute stays on GPU (only PCIe transfer overhead).
        env["LTX_TE_OFFLOAD"] = "1" if self.low_vram.te_offload_text_encoder else "0"
        # Build the DiT (transformer) on CPU and move only non-block submodules to
        # GPU, removing the ~16.9GB load-time GPU spike. On by default; the worker
        # reads this and keeps the blocks CPU-resident for block-swap streaming.
        env["LTX_DIT_CPU_LOAD"] = "1" if self.low_vram.dit_cpu_load else "0"
        return env

    def _log_load_start(self, engine_python: str, engine_dir: Path, payload: dict) -> None:
        """The one INFO line that opens a worker launch (§3-98 P3a seam).

        It quotes payload FIELDS, and a payload's fields are family-specific —
        which is why this is an override point rather than an inline call.
        """
        logger.info(
            "Loading pipeline (REAL worker). python=%s engine_dir=%s block_swap=%s ckpt=%s",
            engine_python,
            engine_dir,
            payload["block_swap_blocks_on_gpu"],
            payload["checkpoint_path"],
        )

    def load(self, selection: dict[str, str] | None = None) -> None:
        if self.loaded:
            return

        # Resolve + validate the engine python and worker script.
        engine_python = self._require_path(
            self._engine_python_value(self.config), self._ENGINE_PYTHON_LABEL
        )
        engine_dir_rel = self._engine_dir_value(self.config)
        if not engine_dir_rel:
            raise RuntimeError("model.engine_dir is not configured (required for the real backend).")
        engine_dir = self.config._abs(engine_dir_rel)
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

        env = self._build_child_env(project_root)

        # stderr -> a log file (NOT a pipe; piping stderr risks a deadlock when
        # the worker emits lots of tqdm/log output while we block on stdout).
        log_dir = self.config.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / self._LOG_NAME

        self._log_load_start(engine_python, engine_dir, payload)
        if selection:
            logger.info("Model-management overrides: %s", selection)

        log_fh = open(self._log_path, "a", encoding="utf-8")
        try:
            self._proc = subprocess.Popen(
                [engine_python, "-u", "-m", self._WORKER_MODULE],
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
            # Acceleration (additive): the worker's own SageAttention probe,
            # run inside the engine venv where the import can actually be
            # attempted. Absent on a pre-acceleration worker -> stays None and
            # the app falls back to LTXRunner's file-existence probe.
            raw_sage = event.get("sage_available")
            self.sage_available = None if raw_sage is None else bool(raw_sage)
            logger.info("LTX worker ready. sage_available=%s", self.sage_available)
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
            # The worker-reported capability dies with the worker: a later
            # /status must fall back to the file probe rather than keep quoting
            # a dead process (the engine venv may have changed meanwhile).
            self.sage_available = None
            safe_memory_cleanup()

    # -------------------------------------------------------------- generate

    def generate(
        self,
        request: GenerateRequest,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
        conditioning_image_paths: list[Path] | None = None,
        lora_paths: list[ResolvedLora] | None = None,
        reference_video_path: Path | None = None,
        seed: int | None = None,
        outpaint_source_path: Path | None = None,
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
        # ResolvedLora (adapter safetensors path, strength, preprocess,
        # audio_strength) resolved by the registry; empty list -> the worker
        # passes ic_loras=[] (explicit clean detach per Stage 1 semantics).
        # ``reference_video`` is the raw reference
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
        loras_payload = [_lora_payload_entry(lp) for lp in lora_paths]
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
        # NAG (additive): only present when enabled, so a non-NAG job's payload
        # stays byte-identical to pre-NAG (regression contract, mirrors the
        # reference_video/loras additive style above).
        if request.nag_enabled:
            payload["nag"] = {
                "negative_prompt": request.negative_prompt,
                "scale": float(request.nag_scale),
                "tau": float(request.nag_tau),
                "alpha": float(request.nag_alpha),
            }
            # VSF (additive, method switch): worker key is "method" (not
            # "neg_method") — scale/tau/alpha above stay unconditional since the
            # engine only reads them when method=="nag".
            payload["nag"]["method"] = request.neg_method
            payload["nag"]["vsf_scale"] = request.vsf_scale

        # Acceleration (additive): the attention backend is sent ONLY when it is
        # not the default, so a default job's payload stays byte-identical to
        # pre-acceleration (regression contract, same style as nag above). The
        # worker fails loud on an unknown value and degrades sage -> sdpa when
        # the import is unavailable.
        if request.attention_backend != "sdpa":
            payload["attention_backend"] = request.attention_backend
        # block_swap_prefetch: same additive contract as attention_backend above
        # — sent only when True, so a default job's payload stays byte-identical
        # to pre-acceleration.
        if request.block_swap_prefetch:
            payload["block_swap_prefetch"] = True
        # keep_resident: same additive contract, but the DEFAULT IS OFF here —
        # so "sent only when True" is also "sent only when it differs from the
        # default", and an omitted key on the worker side means off (which is
        # additionally the explicit trigger that FREES the cache). A default
        # job's payload therefore stays byte-identical to pre-keep_resident.
        if request.keep_resident:
            payload["keep_resident"] = True
        # fused_gguf_dequant_kernel: same additive contract, but as of
        # 2026-08-04 the DEFAULT IS ON (§51: gates G1-G8 passed, owner approved
        # the flip) — so, exactly like block_swap_prefetch above, the key rides
        # on a DEFAULT job too and only disappears when the caller explicitly
        # turns it off (the frozen default-key-set test lists it for that
        # reason). An omitted key still means off on the worker side.
        if request.fused_gguf_dequant_kernel:
            payload["fused_gguf_dequant_kernel"] = True
        # vae_mode: same additive contract, default "default" — so the key rides
        # ONLY on a job that explicitly asks for the pruned decoder, and a
        # default job's payload stays byte-identical to pre-PrunaVAED. Until
        # 2026-08-05 this field was a MOCK that was deliberately never put on
        # the wire; it is now consumed by the engine (§3-50). An environment
        # without the weight file still completes the job — the worker reports
        # vae_mode_used="on->off".
        if request.vae_mode != "default":
            payload["vae_mode"] = request.vae_mode
        # Outpainting (§1-13): same additive contract — the key is absent from
        # every non-outpaint job, so their payloads stay byte-identical. Its
        # presence is ALSO the switch that routes the worker to
        # ``generate_outpaint`` instead of ``generate``, so it carries the full
        # geometry (the engine must rebuild the blend mask) rather than a flag.
        # ``reference_video.path`` above is already the green canvas; the
        # ``source_path`` here is the original upload, read only for audio.
        if request.outpaint is not None:
            op = request.outpaint
            payload["outpaint"] = {
                "source_path": str(outpaint_source_path) if outpaint_source_path else None,
                "canvas_width": request.width,
                "canvas_height": request.height,
                "pad_left": op.pad_left,
                "pad_right": op.pad_right,
                "pad_top": op.pad_top,
                "pad_bottom": op.pad_bottom,
                "blend_dilation_stage1": op.blend_dilation_stage1,
                "blend_dilation_stage2": op.blend_dilation_stage2,
                "freeze_source_audio": op.freeze_source_audio,
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
            # Acceleration: what the engine ACTUALLY ran with (same relay as
            # seed_used). None on a worker that predates the field.
            attention_used=event.get("attention_used"),
            block_swap_prefetch_used=event.get("block_swap_prefetch_used"),
            keep_resident_used=event.get("keep_resident_used"),
            fused_gguf_dequant_kernel_used=event.get(
                "fused_gguf_dequant_kernel_used"
            ),
            vae_mode_used=event.get("vae_mode_used"),
            peak_vram_reserved_mb=event.get("peak_vram_reserved_mb"),
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
        retake_window_path: Path | None = None,
        end_source_path: Path | None = None,
        end_source_context_frames: int | None = None,
        end_source_strength: float | None = None,
        lora_paths: list[ResolvedLora] | None = None,
        reference_video_path: Path | None = None,
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

        End source: when ``end_source_path`` is set, an additive ``end_source``
        block ({path, context_frames}) is added to the worker payload (the app has
        already cut or synthesised the ``context_frames + 1``-frame material). The
        worker's ``done.chain`` then carries the ``end_source`` sub-dict —
        including the ``freeze_proof`` only real latents can produce — returned
        as-is in ``chain_metadata``. ``end_source_strength`` rides on that same
        block ({path, context_frames, strength}); None -> 1.0, a hard stage-1
        freeze byte-identical to before this field existed.

        Style/character IC-LoRA: when ``lora_paths`` is non-empty an additive
        ``loras`` block ([{path, strength[, audio_strength]}, ...]) is added to
        the worker payload (mirrors the single-generate ``loras_payload``). The
        strengths apply uniformly to every clip/stage. Absent for a no-lora
        chain (payload byte-identical to before); the worker clears any stale
        LoRA regardless.

        Reference-video CONTROL IC-LoRA (Phase C; 1..24 clips — owner decision
        2026-08-11 lifted the old clips=1 ALPHA scope, except a depth-preprocess
        adapter, still rejected on >1 clip at the API layer): when
        ``reference_video_path`` is set an additive ``reference_video`` block
        ({path, strength, preprocess[, attention_strength]}) is added to the
        worker payload, mirroring the single-generate ``reference_payload`` (see
        :meth:`_RealBackend.generate`). ONE path is sent regardless of clip
        count -- the engine slices the single long reference into each stage-1
        segment's own window (chain_math.video_segment_windows); a reference
        shorter than the timeline just runs out (later segments generate
        without one). Absent when no reference video was requested, so the
        payload stays byte-identical to before that case.

        NAG (Normalized Attention Guidance, ADDITIVE/optional): when
        ``chain.nag_enabled`` an additive ``nag`` block ({negative_prompt, scale,
        tau, alpha}) is added to the worker payload, applying uniformly across
        every clip/stage. Absent for a non-NAG chain (payload byte-identical to
        before).
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
            "chunked_upsample": bool(chain.chunked_upsample),
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
        # Retake (additive): the app-cut window plus the glue geometry. The
        # ``is not None`` guard on BOTH the path and the request block is what
        # keeps a non-retake chain's payload key set byte-identical
        # (tests/test_ltx_runner_payload.py pins that set in two places).
        if retake_window_path is not None and getattr(chain, "retake", None) is not None:
            payload["retake"] = {
                "path": str(retake_window_path),
                "head_px": int(chain.retake.head_px),
                "tail_px": int(chain.retake.tail_px),
                "regenerate_audio": bool(chain.retake.regenerate_audio),
            }
        # End source (additive): the app-prepared tail material (a cut video or a
        # looped still — the engine sees only a video) plus the band length. The
        # ``is not None`` guard on BOTH values is what keeps a chain without an
        # end source byte-identical, payload key set included
        # (tests/test_ltx_runner_payload.py pins that set in two places).
        if end_source_path is not None and end_source_context_frames is not None:
            payload["end_source"] = {
                "path": str(end_source_path),
                "context_frames": int(end_source_context_frames),
                "strength": (
                    1.0 if end_source_strength is None else float(end_source_strength)
                ),
            }
        # Style/character AND control IC-LoRA (additive): (path, strength[,
        # audio_strength]) per adapter, applied uniformly across the chain. Only
        # added when non-empty so a no-lora chain payload is byte-identical to
        # before (the worker parses msg.get("loras", []) and clears stale LoRA
        # either way). ``preprocess`` is dropped here -- it is derived separately
        # below (via ``_resolve_reference_preprocess``) and only matters when a
        # reference video is also present, since a control adapter without one is
        # already rejected at the API layer (LORA_REQUIRES_REFERENCE).
        if lora_paths:
            payload["loras"] = [_lora_payload_entry(lp) for lp in lora_paths]
        # Reference-video CONTROL IC-LoRA (additive; 1..24 clips — owner decision
        # 2026-08-11, except a depth-preprocess adapter which is still API-layer
        # rejected on >1 clip): mirrors the single-generate ``reference_payload``
        # (see :meth:`generate` above). Only added when a reference video was
        # requested, so a chain without one keeps a byte-identical payload.
        if reference_video_path is not None:
            ref_strength = (
                1.0
                if chain.reference_video_strength is None
                else float(chain.reference_video_strength)
            )
            reference_payload = {
                "path": str(reference_video_path),
                "strength": ref_strength,
                "preprocess": _resolve_reference_preprocess(lora_paths),
            }
            if chain.conditioning_attention_strength is not None:
                reference_payload["attention_strength"] = float(
                    chain.conditioning_attention_strength
                )
            payload["reference_video"] = reference_payload

        # NAG (additive): only present when enabled, so a non-NAG chain's payload
        # stays byte-identical to pre-NAG (regression contract).
        if chain.nag_enabled:
            payload["nag"] = {
                "negative_prompt": chain.negative_prompt,
                "scale": float(chain.nag_scale),
                "tau": float(chain.nag_tau),
                "alpha": float(chain.nag_alpha),
            }
            # VSF (additive, method switch): worker key is "method" (not
            # "neg_method") — scale/tau/alpha above stay unconditional since the
            # engine only reads them when method=="nag".
            payload["nag"]["method"] = chain.neg_method
            payload["nag"]["vsf_scale"] = chain.vsf_scale

        # Acceleration (additive): mirrors the single-generate block in
        # :meth:`generate` — sent only when non-default (byte-identical default
        # payload). See there for the full rationale.
        if chain.attention_backend != "sdpa":
            payload["attention_backend"] = chain.attention_backend
        # block_swap_prefetch: same additive contract as attention_backend above.
        if chain.block_swap_prefetch:
            payload["block_swap_prefetch"] = True
        # keep_resident: mirrors the single-generate block (default OFF, so the
        # key is sent only when True and an omission means off/free-the-cache).
        if chain.keep_resident:
            payload["keep_resident"] = True
        # fused_gguf_dequant_kernel: mirrors the single-generate block (default
        # ON since 2026-08-04, so the key rides on a default chain job too).
        if chain.fused_gguf_dequant_kernel:
            payload["fused_gguf_dequant_kernel"] = True
        # vae_mode: mirrors the single-generate block (default "default", so the
        # key is sent only when the pruned decoder is explicitly asked for). One
        # value covers every clip and every stage of the chain.
        if chain.vae_mode != "default":
            payload["vae_mode"] = chain.vae_mode
        # stage2_window: same additive contract as the acceleration keys above —
        # sent ONLY when the request opted off "standard", so a default chain's
        # worker payload stays byte-identical to before this knob existed
        # (tests/test_ltx_runner_payload.py pins the exact key set).
        if chain.stage2_window != chain_math.STAGE2_WINDOW_DEFAULT:
            payload["stage2_window"] = chain.stage2_window

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
            # Acceleration: same relay as the single-generate path above.
            attention_used=event.get("attention_used"),
            block_swap_prefetch_used=event.get("block_swap_prefetch_used"),
            keep_resident_used=event.get("keep_resident_used"),
            fused_gguf_dequant_kernel_used=event.get(
                "fused_gguf_dequant_kernel_used"
            ),
            vae_mode_used=event.get("vae_mode_used"),
            peak_vram_reserved_mb=event.get("peak_vram_reserved_mb"),
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


# The facade's backend-class seams, bound now that both classes exist. Kept
# here rather than inside the class body because ``LTXRunner`` is declared
# FIRST (it is the file's public face) and Python evaluates a class body at
# definition time — a forward reference in the body would be a NameError.
# Declared (annotation-only) up in the class so a family that forgets to bind
# them fails with a clear AttributeError instead of silently inheriting 2.3's.
LTXRunner._REAL_BACKEND_CLS = _RealBackend
LTXRunner._MOCK_BACKEND_CLS = _MockBackend


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
