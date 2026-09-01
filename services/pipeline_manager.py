"""Pipeline manager (spec ch.9).

Owns pipeline load/unload lifecycle, auto-load on first generation, OOM cleanup,
and runs a job to completion on a background thread: updates the JobRecord,
encodes the video (via the runner), and writes ``metadata.json`` (spec 10.2).
"""

from __future__ import annotations

import logging
import platform
import sys
import threading
import time
from pathlib import Path

import chain_math
from api.errors import (
    end_source_too_short,
    gpu_oom,
    generation_failed,
    pipeline_load_failed,
    pipeline_loading,
    retake_window_out_of_range,
    source_audio_too_short,
    source_video_too_short,
)
from api.models import (
    EndSourceSpec,
    JobResult,
    JobStatus,
    RetakeSpec,
    SourceAudioSpec,
    SourceVideoSpec,
)
from config import AppConfig
from services import gpu_info, video_io
from services.audio_upload_store import AudioUploadStore
from services.base_models import BaseModelDescriptor
from services.engines import runner_class_for
from services.job_store import JobRecord, JobStore, now_iso
from services.low_vram import build_low_vram_settings, safe_memory_cleanup
from services.lora_registry import LoraRegistry
from services.ltx_runner import LTXRunner, resolve_seed
from services.model_registry import CATEGORIES, DEFAULT_NAME, ModelRegistry
from services.runtime_state import RuntimeState
from services.upload_store import UploadStore
from services.video_upload_store import VideoUploadStore

logger = logging.getLogger("ltx.pipeline")


def _peak_vram_suffix(peak_vram_mb) -> str:
    """`` peak_vram=NNNNMB`` for a truthy peak, else ``""`` (mock reports 0/None)."""
    try:
        peak = int(peak_vram_mb)
    except (TypeError, ValueError):
        return ""
    return f" peak_vram={peak}MB" if peak > 0 else ""


def _is_oom(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return "outofmemory" in name or "out of memory" in msg or "cuda oom" in msg


# Console job-info logging (owner requirement): the operator watches the uvicorn
# console and needs to see, per job, the real seed, the base weight file, the
# LoRAs, and the prompt. These helpers build that one INFO line safely.
_PROMPT_LOG_MAX = 200


def _prompt_for_log(prompt: str, limit: int = _PROMPT_LOG_MAX) -> str:
    """One-line, length-capped rendering of a prompt for the console.

    The prompt is raw USER input, so newlines/tabs are escaped (log-injection
    safe — a crafted prompt can't forge extra log lines) and the text is capped
    at ``limit`` chars with an ellipsis so a very long prompt can't flood the
    console. ASCII-only punctuation is used so the line stays encodable on a
    legacy (cp932) console even when the prompt body itself is non-ASCII.
    """
    text = (
        (prompt or "")
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    if len(text) > limit:
        text = text[:limit] + "..."
    return text


def _loras_for_log(specs, resolved) -> str:
    """``name(strength=R[, effective=E][, audio=A[, effective=E]])`` per adapter,
    or ``none``.

    ``specs`` are the request ``LoraSpec``s (friendly NAME + the REQUESTED
    strength); ``resolved`` are the registry ``ResolvedLora``s in the SAME order
    (see ``LoraRegistry.resolve`` — ``effective`` folds in the alpha/rank
    convolution). ``effective`` is only shown when it actually differs from the
    requested strength, so the common scale==1.0 case stays terse. The
    ``audio=`` segment is appended only when the resolved audio_strength is not
    None (video-axis-only jobs keep the exact prior rendering).
    """
    if not specs:
        return "none"
    parts: list[str] = []
    for i, spec in enumerate(specs):
        eff = resolved[i][1] if i < len(resolved) else None
        if eff is not None and abs(float(eff) - float(spec.strength)) > 1e-6:
            part = f"{spec.name}(strength={spec.strength:g}, effective={float(eff):g})"
        else:
            part = f"{spec.name}(strength={spec.strength:g})"
        audio_eff = resolved[i][3] if i < len(resolved) and len(resolved[i]) > 3 else None
        if audio_eff is not None:
            requested_audio = getattr(spec, "audio_strength", None)
            if requested_audio is not None and abs(
                float(audio_eff) - float(requested_audio)
            ) > 1e-6:
                part += f", audio={requested_audio:g}, effective={float(audio_eff):g}"
            else:
                part += f", audio={float(audio_eff):g}"
        parts.append(part)
    return ", ".join(parts)


def _lora_metadata_entry(spec, registry: LoraRegistry) -> dict:
    """One ``ic_lora.loras[]`` row: ``name``/``strength``/``preprocess`` (Phase
    B/C, unchanged) plus ``audio_strength`` — added only when the spec carries
    one (``None`` before WP4 adds the field to ``LoraSpec``), so a video-axis-
    only job's metadata keeps its exact prior key set.
    """
    entry = {
        "name": spec.name,
        "strength": spec.strength,
        "preprocess": registry.preprocess_for(spec.name),
    }
    audio_strength = getattr(spec, "audio_strength", None)
    if audio_strength is not None:
        entry["audio_strength"] = audio_strength
    return entry


class PipelineManager:
    STATE_UNLOADED = "unloaded"
    STATE_LOADING = "loading"
    STATE_READY = "ready"
    STATE_RUNNING = "running"
    STATE_ERROR = "error"

    def __init__(
        self,
        config: AppConfig,
        job_store: JobStore,
        upload_store: UploadStore,
        video_upload_store: VideoUploadStore | None = None,
        lora_registry: LoraRegistry | None = None,
        audio_upload_store: AudioUploadStore | None = None,
        descriptor: BaseModelDescriptor | None = None,
        runtime_state: RuntimeState | None = None,
        active_base_model: str | None = None,
        active_models: dict[str, str] | None = None,
        active_selection_paths: dict[str, str] | None = None,
        model_registry: ModelRegistry | None = None,
    ):
        self.config = config
        self.job_store = job_store
        self.upload_store = upload_store
        # Phase B: reference-video store + IC-LoRA name registry. Defaulted so
        # existing constructions (tests) still work; the app always injects them.
        self.video_upload_store = video_upload_store or VideoUploadStore(config)
        # A2V: source-audio store. Defaulted like the video store so existing
        # constructions keep working; the app always injects it.
        self.audio_upload_store = audio_upload_store or AudioUploadStore(config)
        self.lora_registry = lora_registry or LoraRegistry(config)
        self.low_vram = build_low_vram_settings(config)
        # ``descriptor``: the base model this pipeline serves (§3-97 P3b). The
        # app injects the first descriptor it loaded at startup; None lets the
        # runner resolve it lazily from config.manifest_dir (tests/tools). Read
        # back through ``self.runner.descriptor`` so there is ONE resolution
        # rule, not two.
        #
        # WHICH RUNNER CLASS depends on the descriptor's engine family (§3-98
        # P3c). A None descriptor keeps ``LTXRunner``: resolving the family
        # would mean loading the manifests here, and constructing a manager is
        # deliberately free of file reads — the lazy default is the first
        # declared base model, which is LTX 2.3.
        runner_cls = LTXRunner if descriptor is None else runner_class_for(descriptor.engine_family)
        self.runner = runner_cls(config, self.low_vram, descriptor)
        # Server runtime state (§3-97 P5): the file that remembers the last
        # active base model + selection across restarts. Defaulted so existing
        # constructions (tests, tools) keep working; the app always injects the
        # one it read at startup. Written on a successful load/reload ONLY —
        # see :meth:`_remember`.
        self.runtime_state = runtime_state or RuntimeState(config.state_path)
        # The model registry, needed for exactly two things: looking up the
        # descriptor of a base model a load asks to switch TO, and publishing
        # the new active base back to it afterwards. None for the standalone
        # constructions (tests, outputs/ drivers) that never pass a
        # ``base_model`` — see :meth:`_apply_base_model`.
        self.model_registry = model_registry
        # The base model this pipeline currently serves. Injected by the app
        # from the runtime state; None here means "whatever the runner's
        # descriptor turns out to be", resolved LAZILY (see the property) so
        # constructing a manager still reads no manifest file. The
        # request-level axis that CHANGES it is POST /pipeline/load's
        # ``base_model`` (P6).
        self._active_base_model: str | None = active_base_model
        self.state = self.STATE_UNLOADED
        self._lock = threading.Lock()
        # Model management: the category NAME used by the last successful load
        # ("default" until an explicit selection succeeds). Read by GET /models;
        # retained while the worker is unloaded — load-state questions belong to
        # ``pipeline_loaded`` (design ruling §9-6). Never updated on a failed
        # swap-load (the previous successful selection stays authoritative).
        # ``active_models`` (P5): seeded from the runtime state when the app
        # injects one, so a restart resumes the operator's last combination
        # instead of reverting to the shipped defaults. Always spans the full
        # CATEGORIES set — a remembered name for a category this build does not
        # know is dropped, and a category the state never mentioned falls back
        # to "default", so the map's SHAPE is the same on every boot.
        self.active_models: dict[str, str] = {
            c: (active_models or {}).get(c, DEFAULT_NAME) for c in CATEGORIES
        }
        # The resolved ABSOLUTE paths behind active_models' non-default names
        # (empty while everything is default). A selection-less load() reuses
        # these, so auto-load-on-generate after an unload keeps the active
        # (possibly swapped) combination instead of silently reverting.
        #
        # SEEDED TOGETHER WITH ``active_models`` OR NOT AT ALL. The two are one
        # fact in two halves — the names a client sees and the files the worker
        # gets — and the app resolves the restored names to paths before
        # construction (AppContext._restore_selection) precisely so a restart
        # can never leave this half empty while the other half claims a
        # non-default selection.
        self._active_selection_paths: dict[str, str] = dict(active_selection_paths or {})

    # --------------------------------------------------------------- status

    @property
    def loaded(self) -> bool:
        return self.runner.loaded

    @property
    def active_base_model(self) -> str:
        """Id of the base model this pipeline serves.

        Read-only from the outside: it changes only through a SUCCESSFUL
        ``load``/``reload`` that named a different base model. Retained across
        an unload, exactly like ``active_models`` — "what is selected" and
        "what is loaded right now" are separate questions (design ruling §9-6).
        """
        if self._active_base_model is None:
            self._active_base_model = self.runner.descriptor.id
        return self._active_base_model

    @property
    def active_engine_family(self) -> str:
        """Engine family that would run a job submitted right now (§3-98 P5).

        Read from the RUNNER's descriptor, not from ``active_base_model``: the
        runner object is what actually holds the worker (``_point_runner_at``
        replaces it when a switch crosses families), so its descriptor is the
        engine that a job reaches — while ``active_base_model`` is a NAME the
        client asked for, which can legitimately be a step ahead of the runner
        during a load that has not committed yet.
        """
        return self.runner.descriptor.engine_family

    @property
    def pipeline_type(self) -> str:
        return self.runner.pipeline_type

    def vram_status_block(self) -> dict:
        return self.low_vram.status_block(
            low_vram_disabled_required=self.config.limits.low_vram_disabled_required
        )

    # Acceleration backends advertised by GET /status. Only ``attention_backend``
    # is a real implementation choice; the MOCK request field (vae_mode) is
    # deliberately NOT advertised here — /status describes what the server can
    # actually DO.
    ATTENTION_BACKENDS = ["sdpa", "sage"]

    def acceleration_status_block(self) -> dict:
        """The ``acceleration`` block for GET /status.

        Sibling of :meth:`vram_status_block` (the FROZEN ``vram_optimization``
        block is untouched by this feature). This method is the single place the
        sage-availability TRUTH TABLE lives:

        =========================  ==========================================
        server state               ``sage_available``
        =========================  ==========================================
        mock backend               ``False`` (no engine exists at all)
        pipeline not loaded        engine-venv FILE probe
        pipeline loaded            the WORKER's own import probe (authoritative)
        load failed / unloaded     engine-venv FILE probe (no live worker)
        =========================  ==========================================

        The two probe routes are intentionally different answers to different
        questions: the file probe says "sage COULD be importable", the worker's
        says "sage IS importable in the process that would use it" (it catches
        a DLL/ABI failure the file probe cannot see). ``pipeline_loaded`` in the
        same /status payload already tells a client which route produced the
        value, so no extra ``sage_source`` field is exposed.
        """
        return {
            "attention_backends": list(self.ATTENTION_BACKENDS),
            "sage_available": self._sage_available(),
            # Block-swap prefetch (backend §44). Judged by the SAME formula as
            # the real gate (services/ltx_runner.py's block-swap-blocks-on-GPU
            # expression), NOT the display-only ``low_vram.block_swap`` bool —
            # that bool is never read on the real path and defaults to False,
            # which would make this field lie in the default configuration.
            "block_swap_prefetch_available": self._block_swap_prefetch_available(),
        }

    def _sage_available(self) -> bool:
        if self.runner.is_mock:
            return False
        worker_value = self.runner.worker_sage_available
        if worker_value is not None:
            return worker_value
        return self.runner.sage_available

    def _block_swap_prefetch_available(self) -> bool:
        """True iff block swap is actually active on the real worker.

        Mirrors ``services/ltx_runner.py``'s
        ``self.low_vram.block_swap_blocks_on_gpu or 8`` expression exactly (the
        ``load`` payload's ``block_swap_blocks_on_gpu``), so this can never
        disagree with what the worker was actually told to do.
        """
        if self.runner.is_mock:
            return False
        return int(self.low_vram.block_swap_blocks_on_gpu or 8) > 0

    def _base_model_name(self) -> str:
        """Filename of the transformer weight (GGUF) that would actually load.

        A model-management swap records the selected transformer path in
        ``_active_selection_paths``; otherwise the BASE MODEL's own
        ``default_file`` applies (§3-97 P3b — this used to read a config field
        holding the same path). Only the basename is surfaced (no path leak) —
        e.g. ``LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf`` — so the operator can
        tell from the console which base weight a job ran on. Falls back to the
        configured ``checkpoint_name`` when the base model declares no
        transformer default.
        """
        path = self._active_selection_paths.get("transformer") or self._default_file("transformer")
        if path:
            return Path(path).name
        return self.config.model.checkpoint_name or "unknown"

    def _default_file(self, category: str) -> str | None:
        """The active base model's ``default_file`` for ``category`` (or None).

        ``models_dir``-relative, and used only for its BASENAME here — the
        console line and metadata.json want the file's name, never its path.
        """
        spec = self.runner.descriptor.categories.get(category)
        return spec.default_file if spec else None

    def _models_metadata_block(self) -> dict:
        """The ``models`` block of metadata.json (§1-23 + §3-97 P3b).

        ``base_model`` is the descriptor id of the base model that ran (P3b:
        always the active one — the request-level base-model axis arrives with
        the API phase). ``selection`` is per category: the NAME the user sees
        plus the FILE that name resolved to. A category still on ``"default"``
        now records the base model's ``default_file`` basename instead of null,
        which is what makes an old output reproducible after the default
        changes.
        """
        return {
            "base_model": self.runner.descriptor.id,
            "selection": {
                cat: {
                    "name": name,
                    "file": (
                        Path(p).name
                        if (p := self._active_selection_paths.get(cat) or self._default_file(cat))
                        else None
                    ),
                }
                for cat, name in self.active_models.items()
            },
        }

    # ------------------------------------------------------------ lifecycle

    def load(
        self,
        selection: dict[str, str] | None = None,
        active_names: dict[str, str] | None = None,
        base_model: str | None = None,
    ) -> None:
        """Load the pipeline (model management: optionally with overrides).

        ``selection`` maps a category to a resolved ABSOLUTE weight path;
        ``active_names`` is the matching category->NAME map recorded (on
        success only) for GET /models. Both default to None: a plain ``load()``
        reuses the last successful selection (all-default on boot), keeping the
        legacy call — and auto-load-on-generate — behaviorally unchanged while
        honoring a previous swap.

        ``base_model`` (§3-97 P6) switches the BASE MODEL this pipeline serves.
        None — every pre-P6 caller — keeps the current one, so nothing about
        the legacy paths changes.
        """
        with self._lock:
            self._reject_while_loading()
            if self.runner.loaded:
                self.state = self.STATE_READY
                return
            self.state = self.STATE_LOADING
        if selection is None:
            selection = dict(self._active_selection_paths)
            active_names = dict(self.active_models)
        rollback = None
        try:
            rollback = self._apply_base_model(base_model)
            self.runner.load(selection=selection or None)
            self.state = self.STATE_READY
        except Exception as exc:
            self._restore_base_model(rollback)
            self.state = self.STATE_ERROR
            self._cleanup_after_error()
            logger.exception("Pipeline load failed")
            raise pipeline_load_failed(detail=str(exc)) from exc
        if active_names:
            self.active_models = dict(active_names)
        self._active_selection_paths = dict(selection)
        self._remember()

    def reload(
        self,
        selection: dict[str, str],
        active_names: dict[str, str],
        base_model: str | None = None,
    ) -> None:
        """Swap-load (model management): force a worker rebuild with a new
        model selection.

        The worker builds its pipeline exactly once (engine/worker.py) and
        ``load()`` early-returns while loaded, so a swap is unload (kill the
        subprocess) + load — never an in-place re-load op. On failure there is
        NO automatic fallback (design ruling §9-1): the pipeline stays
        unloaded, ``active_models`` keeps the previous successful selection,
        and the error tells the user how to recover.

        ``base_model`` (§3-97 P6): see :meth:`load`. A base-model change is
        always a rebuild, so it necessarily comes through here or through a
        ``load`` of an unloaded pipeline.
        """
        with self._lock:
            self._reject_while_loading()
            self.runner.unload()
            self.state = self.STATE_LOADING
        rollback = None
        try:
            rollback = self._apply_base_model(base_model)
            self.runner.load(selection=selection or None)
        except Exception as exc:
            self._restore_base_model(rollback)
            self.state = self.STATE_ERROR
            self._cleanup_after_error()
            logger.exception("Pipeline swap-load failed")
            raise pipeline_load_failed(
                detail=(
                    f"{exc} — the previous pipeline was unloaded and no fallback "
                    "was attempted; select the 'default' models and Load again "
                    "to recover."
                )
            ) from exc
        self.active_models = dict(active_names)
        self._active_selection_paths = dict(selection)
        self.state = self.STATE_READY
        self._remember()

    def unload(self) -> None:
        # NO ``_reject_while_loading()`` HERE, DELIBERATELY (§3-97 P6). A load
        # that dies in a way ``except Exception`` cannot see — a BaseException,
        # a killed thread — leaves ``state`` stuck at "loading", and the 409
        # guard would then reject every load/reload forever. Unload is the one
        # door kept unlocked so the operator can always get back to a clean
        # ``unloaded`` state without restarting the server. Unloading during a
        # real in-flight load is harmless anyway: it takes the same lock, so it
        # can only run between the load's own locked sections.
        with self._lock:
            self.runner.unload()
            self.state = self.STATE_UNLOADED

    def reject_if_loading(self) -> None:
        """409 if a load is already in flight (API-layer entry point).

        The judgment and the message stay owned by ``_reject_while_loading``;
        this method only does the locking. The load itself runs OUTSIDE the
        lock (see ``_reject_while_loading``), so this wait is trivial.
        """
        with self._lock:
            self._reject_while_loading()

    def _reject_while_loading(self) -> None:
        """409 if a load is already in flight. CALLED ONLY UNDER ``_lock``.

        A model load takes minutes and runs OUTSIDE the lock (holding it for
        the whole load would block GET /status), so the in-flight window is
        wide and a second load arriving inside it is ordinary — a double-click
        on the frontend's Load button. Letting it through would start a second
        worker build on top of the first.
        """
        if self.state == self.STATE_LOADING:
            raise pipeline_loading(
                detail="現在モデルを読み込んでいます。完了までお待ちください。"
            )

    def _apply_base_model(self, base_model: str | None):
        """Point the runner at ``base_model``; return the descriptor to roll
        back to, or None when nothing changed.

        Done BEFORE the load (the runner builds its payload from the
        descriptor) but rolled back if the load fails, so a failed switch does
        not leave the manager claiming a base model it never managed to load —
        the same discipline ``active_models`` follows (design ruling §9-1).
        """
        if base_model is None or base_model == self.active_base_model:
            return None
        if self.model_registry is None:
            raise RuntimeError(
                "a base-model switch needs the model registry; this "
                "PipelineManager was built without one"
            )
        previous = self.runner.descriptor
        self._point_runner_at(self.model_registry.descriptor(base_model))
        self._active_base_model = base_model
        logger.info("Base model switched to '%s'", base_model)
        return previous

    def _restore_base_model(self, rollback) -> None:
        if rollback is None:
            return
        self._point_runner_at(rollback)
        self._active_base_model = rollback.id

    def _point_runner_at(self, descriptor: BaseModelDescriptor) -> None:
        """Aim ``self.runner`` at ``descriptor`` — replacing the runner OBJECT
        when the engine family changes (§3-98 P3c).

        WITHIN one family, ``set_descriptor`` is the whole story: it discards
        the backend (so the next load builds its payload from the new base
        model's paths) and unloads any live worker.

        ACROSS families it cannot be, because the runner class itself is the
        thing that differs — a different worker module, a different venv, a
        different load payload. Re-pointing an ``LTXRunner`` at an LTX 2.5
        descriptor would spawn the 2.3 worker on 2.5 weights.

        THE OLD WORKER IS SHUT DOWN FIRST, and this is the ONE place that can
        do it: the moment the old runner object is dropped, nothing holds a
        handle on its subprocess any more, and a stranded engine process keeps
        its VRAM. ``unload`` sends ``shutdown``, waits, then terminates and
        kills — so by the time it returns the process is gone whether or not it
        cooperated. Both the apply and the rollback path come through here, so
        a failed switch cannot leave two workers behind either.
        """
        target_cls = runner_class_for(descriptor.engine_family)
        # Exact type, not isinstance: LTX25Runner SUBCLASSES LTXRunner, so an
        # isinstance test would silently keep the 2.3 runner for a 2.5 switch.
        if type(self.runner) is target_cls:
            self.runner.set_descriptor(descriptor)
            return
        logger.info(
            "Engine family change: %s -> %s (unloading the previous worker first)",
            type(self.runner).__name__,
            target_cls.__name__,
        )
        self.runner.unload()
        if self.runner.loaded:  # pragma: no cover - unload's finally always clears it
            logger.error(
                "previous %s still reports loaded after unload; the old worker "
                "process may have survived the switch",
                type(self.runner).__name__,
            )
        self.runner = target_cls(self.config, self.low_vram, descriptor)

    def _remember(self) -> None:
        """Publish the combination that just loaded (§3-97 P5/P6).

        Two destinations: the model REGISTRY (so ``GET /models`` and every
        base-less resolve follow the active base model from the next request
        on) and the STATE FILE (so the next start does).

        Called from the SUCCESS path of ``load``/``reload`` and nowhere else:

        * a FAILED load must not overwrite the last combination that worked —
          the whole point of the file is to come back to something loadable;
        * an UNLOAD must not either. Unloading is "free the VRAM", not "forget
          my choice" (``active_models`` survives it for the same reason), so a
          restart after an unload still resumes the last selection.

        :meth:`RuntimeState.save` never raises — a state file that cannot be
        written is a WARNING, not a failed load.
        """
        if self.model_registry is not None:
            self.model_registry.set_active_base_model(self.active_base_model)
        self.runtime_state.save(self.active_base_model, self.active_models)

    def _cleanup_after_error(self) -> None:
        try:
            self.runner.unload()
        except Exception:
            pass
        safe_memory_cleanup()
        self.state = self.STATE_UNLOADED

    # ------------------------------------------------------------- run job

    def run_job(self, job: JobRecord) -> None:
        """Run one generation job to terminal state. Intended for a worker thread."""
        # Race guard: promote queued -> running as a compare-and-set under the
        # store lock, mutually exclusive with DELETE's queued -> cancelled
        # transition (JobStore.cancel_if_queued). Exactly one side wins — a job
        # DELETE cancelled while queued can never be resurrected here, and once
        # this promotion lands DELETE falls back to best-effort cancel_requested.
        if not self.job_store.start_job(job):
            # Lost to a concurrent cancel (or a cancel_requested flag set while
            # still queued): finalize as cancelled if DELETE hasn't already.
            self.job_store.cancel_if_queued(job)
            self.state = self.STATE_READY
            logger.info("Job %s cancelled before dispatch", job.job_id)
            return

        job.progress = 0.05
        started = time.time()
        output_dir = self.config.output_dir / job.job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve the seed HERE (single source of truth) so the console shows the
        # value actually used even for a -1 (random) request; the same value is
        # handed to the runner below so the log and the render never disagree.
        seed = resolve_seed(job.request.seed)

        logger.info(
            "Job %s start mode=%s %dx%d frames=%d steps=%d fps=%g seed=%d",
            job.job_id,
            job.request.generation_mode,
            job.request.width,
            job.request.height,
            job.request.num_frames,
            job.request.num_inference_steps,
            job.request.frame_rate,
            seed,
        )

        try:
            if not self.runner.loaded:
                if not self.config.model.auto_load_on_generate:
                    raise pipeline_load_failed(detail="auto_load_on_generate is disabled")
                self.load()

            self.state = self.STATE_RUNNING

            cond_paths = [
                self.upload_store.path_for(ci.image_id)
                for ci in job.request.conditioning_images
            ]

            # Phase B/C IC-LoRA: resolve adapter names -> (path, strength,
            # preprocess) via the registry and reference_video_id -> path via the
            # video store. The API layer already validated existence + preprocess-
            # kind conflicts (mirroring conditioning images), so these re-resolve
            # the same objects for the runner hop.
            lora_paths = [
                self.lora_registry.resolve(
                    spec.name, spec.strength, getattr(spec, "audio_strength", None)
                )
                for spec in job.request.loras
            ]
            reference_video_path = (
                self.video_upload_store.path_for(job.request.reference_video_id)
                if job.request.reference_video_id
                else None
            )

            # Outpainting (Docs/PENDING_TASKS_CLOSED.md §3-70, filed as §1-13 at
            # the time): build the green-padded canvas and hand THAT to
            # the runner as the reference video. Substituting the path here is
            # what keeps the whole IC-LoRA chain below untouched — the engine's
            # ``_resolve_ic_reference`` and ``_reference_conditioning_for_stage``
            # never learn that outpainting exists, they just encode whatever
            # reference they were given. The canvas lands next to the output (the
            # same place ``control_<preprocess>.mp4`` goes) rather than in
            # uploads/, which STORAGE_POLICY.md reserves for material the user
            # may delete at any time.
            outpaint_source_path = None
            if job.request.outpaint is not None:
                op = job.request.outpaint
                outpaint_source_path = reference_video_path
                canvas_path = output_dir / "outpaint_canvas.mp4"
                video_io.pad_green_mp4(
                    reference_video_path,
                    canvas_path,
                    canvas_width=job.request.width,
                    canvas_height=job.request.height,
                    pad_left=op.pad_left,
                    pad_top=op.pad_top,
                    frame_rate=job.request.frame_rate,
                    num_frames=job.request.num_frames,
                )
                reference_video_path = canvas_path
                logger.info(
                    "Job %s outpaint canvas %dx%d pads l/r/t/b=%d/%d/%d/%d -> %s",
                    job.job_id, job.request.width, job.request.height,
                    op.pad_left, op.pad_right, op.pad_top, op.pad_bottom,
                    canvas_path.name,
                )

            # Console job-info line (owner requirement): base weight file + LoRAs
            # (name/requested/effective strength) + prompt, so LoRA application is
            # visible from the uvicorn console (the worker's per-adapter attach
            # line only reaches logs/ltx_worker.log). Additive — the start line
            # above keeps its exact format.
            logger.info(
                'Job %s base=%s loras=%s prompt="%s"',
                job.job_id,
                self._base_model_name(),
                _loras_for_log(job.request.loras, lora_paths),
                _prompt_for_log(job.request.prompt),
            )

            def on_progress(step, total, progress, stage=None, clip=None, clip_count=None):
                # clip/clip_count are part of the ProgressCallback contract but
                # only chain stage-1 events ever pass them — a single generate
                # has no clips, so they are accepted and ignored here.
                job.current_step = step
                job.total_steps = total
                job.stage = stage
                job.progress = progress

            outcome = self.runner.generate(
                job.request,
                output_dir=output_dir,
                progress_callback=on_progress,
                conditioning_image_paths=cond_paths,
                lora_paths=lora_paths,
                reference_video_path=reference_video_path,
                seed=seed,
                outpaint_source_path=outpaint_source_path,
            )

            elapsed = time.time() - started
            result = self._finalize(job, outcome, output_dir, elapsed)

            if job.cancel_requested:
                job.status = JobStatus.cancelled
            else:
                job.status = JobStatus.completed
                job.result = result
            job.progress = 1.0
            job.completed_at = now_iso()
            self.state = self.STATE_READY
            logger.info(
                "Job %s done in %.1fs%s -> %s",
                job.job_id, elapsed, _peak_vram_suffix(outcome.peak_vram_mb), job.status.value,
            )

        except Exception as exc:
            job.completed_at = now_iso()
            job.status = JobStatus.failed
            if _is_oom(exc):
                err = gpu_oom(job_id=job.job_id, detail=str(exc))
                logger.error("Job %s OOM: %s", job.job_id, exc)
                self._cleanup_after_error()
            else:
                err = generation_failed(job_id=job.job_id, detail=str(exc))
                logger.exception("Job %s failed", job.job_id)
                self.state = self.STATE_READY if self.runner.loaded else self.STATE_UNLOADED
            job.error = f"{err.code}: {err.message} ({err.detail})" if err.detail else f"{err.code}: {err.message}"
        finally:
            safe_memory_cleanup()

    # --------------------------------------------------- V2V source preflight

    def preflight_source_video(
        self, source_video: SourceVideoSpec, request_frame_rate: float
    ) -> None:
        """Validate an uploaded V2V continuation source BEFORE a job is created.

        ffprobes the stored source for fps + frame count and rejects (422
        SOURCE_VIDEO_TOO_SHORT) when it cannot supply the requested
        ``context_frames`` tail after resampling to ``request_frame_rate``. The
        video_id is assumed already resolved (the endpoint 404s first). Geometry
        bounds (8n+1, [25,145], context < clip-0) are enforced by the schema.
        """
        src_path = self.video_upload_store.path_for(source_video.video_id)
        n_src = video_io.frame_count(src_path)
        src_fps = video_io.probe_fps(src_path)
        # Frame count after resampling to the request fps. Exact for the no-resample
        # case; a duration-based estimate for the resample case (cut_tail_mp4 is the
        # frame-exact backstop, which raises if the resampled source is still short).
        if src_fps and abs(src_fps - float(request_frame_rate)) > 1e-3:
            effective = int(round(n_src * float(request_frame_rate) / src_fps))
        else:
            effective = n_src
        if effective < source_video.context_frames:
            raise source_video_too_short(
                detail=(
                    f"source has {n_src} frames @ {src_fps} fps "
                    f"(~{effective} @ {request_frame_rate} fps) < "
                    f"context_frames={source_video.context_frames}"
                )
            )

    # ----------------------------------------------------- end source preflight

    def preflight_end_source(
        self, end_source: EndSourceSpec, request_frame_rate: float
    ) -> None:
        """Validate an uploaded end source BEFORE a job is created.

        The mirror of :meth:`preflight_source_video`, with one difference that is
        the whole point of the feature's +1 primer: the material must supply
        ``context_frames + 1`` frames, not ``context_frames``. The causal video
        VAE spends the FIRST frame on its lone keyframe latent, which is not part
        of the frozen tail band, so a file exactly ``context_frames`` long would
        cut one frame short of a full band (422 END_SOURCE_TOO_SHORT).

        An IMAGE end source returns immediately: a still is looped to whatever
        length the band asks for, so there is nothing about its length to reject.
        The endpoint only calls this for a video anyway; the guard is here so the
        method is safe to call with either kind. The id is assumed already
        resolved (the endpoint 404s first), and the band's geometry (multiple of
        8, within [8,136], overlap >= 2) is enforced by the schema and
        ``chain_math.compute_chain_layout``.

        THE ASYMMETRY WITH THE APP IS DELIBERATE. This check resamples with
        ``round`` and asks for ``context_frames + 1``; the app derives the band
        length from the server-measured frame count with a ``floor`` and one
        frame of slack, i.e. it is strictly MORE conservative. So on an
        app-driven request this never fires — it exists for the frame-exact case
        and for callers that pick ``context_frames`` themselves (the Gradio UI,
        scripts, a stale client). Do NOT "fix" the two into agreement: matching
        the app's rounding here would only move the boundary, and matching this
        one there would let the app propose bands the material cannot fill.
        """
        if end_source.video_id is None:
            return
        src_path = self.video_upload_store.path_for(end_source.video_id)
        n_src = video_io.frame_count(src_path)
        src_fps = video_io.probe_fps(src_path)
        # Frame count after resampling to the request fps — same estimate
        # preflight_source_video uses; video_io.cut_window_mp4's MEASURED count is
        # the frame-exact backstop.
        if src_fps and abs(src_fps - float(request_frame_rate)) > 1e-3:
            effective = int(round(n_src * float(request_frame_rate) / src_fps))
        else:
            effective = n_src
        required = int(end_source.context_frames) + 1
        if effective < required:
            raise end_source_too_short(
                detail=(
                    f"end source has {n_src} frames @ {src_fps} fps "
                    f"(~{effective} @ {request_frame_rate} fps) < "
                    f"context_frames={end_source.context_frames} + 1 primer frame "
                    f"= {required}"
                )
            )

    # ------------------------------------------------- retake window preflight

    def preflight_retake_window(
        self, retake: RetakeSpec, window_frames: int, request_frame_rate: float
    ) -> None:
        """Validate an uploaded retake source BEFORE a job is created.

        Two checks, both about the MATERIAL rather than the geometry (the window
        length, the 8n+1 grid and the glue-band grids were already settled by the
        schema and ``chain_math.compute_chain_layout``):

        1. the window ``[window_start_sec, +window_frames)`` must fit inside the
           upload once resampled to ``request_frame_rate`` — same effective-frame
           estimate :meth:`preflight_source_video` uses, with
           ``video_io.cut_window_mp4``'s measured frame count as the frame-exact
           backstop;
        2. ``regenerate_audio=False`` asks to keep the window's own audio, which
           a silent upload cannot supply.

        The video_id is assumed already resolved (the endpoint 404s first).
        """
        src_path = self.video_upload_store.path_for(retake.video_id)
        n_src = video_io.frame_count(src_path)
        src_fps = video_io.probe_fps(src_path)
        if src_fps and abs(src_fps - float(request_frame_rate)) > 1e-3:
            effective = int(round(n_src * float(request_frame_rate) / src_fps))
        else:
            effective = n_src
        start_frame = round(float(retake.window_start_sec) * float(request_frame_rate))
        if start_frame + window_frames > effective:
            raise retake_window_out_of_range(
                detail=(
                    f"window starts at frame {start_frame} and needs "
                    f"{window_frames} frames, but the source has {n_src} frames "
                    f"@ {src_fps} fps (~{effective} @ {request_frame_rate} fps)"
                )
            )
        if not retake.regenerate_audio and not video_io.has_audio_stream(src_path):
            raise retake_window_out_of_range(
                detail=(
                    "regenerate_audio=false keeps the window's ORIGINAL audio, "
                    f"but upload {retake.video_id} has no audio stream"
                )
            )

    # --------------------------------------------------- A2V source preflight

    def preflight_source_audio(
        self,
        source_audio: SourceAudioSpec,
        clip_frames: list[int],
        frame_rate: float,
        overlap_frames: int,
    ) -> None:
        """Validate an uploaded A2V source audio BEFORE a job is created.

        Resolves the ``audio_id`` (the endpoint 404s first) and ffprobes the
        stored file for an audio stream + duration. The chain timeline needs
        ``chain_math.audio_latents_required(...)`` audio-latent frames (25/sec —
        :data:`chain_math.AUDIO_LATENTS_PER_SEC`, the SINGLE SOURCE OF TRUTH the
        engine also encodes against); an upload whose duration VAE-encodes to
        fewer than that is rejected (422 SOURCE_AUDIO_TOO_SHORT). Video length is
        authoritative — audio is truncated, never padded (matches upstream a2vid
        and the engine's ``a2v_avail < a_total`` guard). No fps resample is done.
        """
        src_path = self.audio_upload_store.path_for(source_audio.audio_id)
        required = chain_math.audio_latents_required(
            clip_frames, frame_rate, kv=overlap_frames
        )
        if not video_io.has_audio_stream(src_path):
            raise source_audio_too_short(
                detail=f"no decodable audio stream in upload {source_audio.audio_id}"
            )
        duration = video_io.probe_duration(src_path)
        if duration is None:
            # ffprobe unavailable / unreadable duration: cannot verify length here.
            # The engine's a2v_avail < a_total guard is the frame-exact backstop.
            return
        available = round(duration * chain_math.AUDIO_LATENTS_PER_SEC)
        if available < required:
            raise source_audio_too_short(
                detail=(
                    f"audio is {duration:.3f}s (~{available} audio-latent frames) "
                    f"< required {required} frames for the {sum(clip_frames)}-frame "
                    f"timeline @ {frame_rate} fps"
                )
            )

    # -------------------------------------------------------- run chain job

    def run_chain_job(self, job: JobRecord) -> None:
        """Run a multi-clip chain to ONE continuous output.mp4 (Phase 3 WP4).

        Masked AV-latent concatenation: the whole chain runs INSIDE ONE worker
        invocation (latents resident across segments) — per-segment stage-1 with
        AV carry+freeze, crossfade assembly, always-tiled stage-2, ONE VAE decode.
        Replaces the old per-clip generate + carry.pt + ffmpeg-concat + trim path
        (which cut hard at every boundary). Junction pixel-frame indices (segment
        seams AND stage-2 tile seams) are recorded in metadata for the review
        harness. The single-clip :meth:`run_job` is untouched.

        Cancellation: the chain is now one atomic worker op, so cancel is honored
        at the job boundary (before dispatch) — matching that a single generate is
        also not interruptible mid-run.
        """
        chain = job.chain_request
        assert chain is not None, "run_chain_job requires job.chain_request"

        # Same lock-guarded queued -> running compare-and-set as run_job: DELETE's
        # queued -> cancelled transition and this promotion are mutually exclusive,
        # so a cancelled chain can never be resurrected into a running one.
        if not self.job_store.start_job(job):
            self.job_store.cancel_if_queued(job)
            self.state = self.STATE_READY
            logger.info("Chain job %s cancelled before dispatch", job.job_id)
            return

        job.progress = 0.02
        started = time.time()
        output_dir = self.config.output_dir / job.job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        n = len(chain.clips)

        # Single seed resolution point (see run_job) — logged and handed to the
        # runner so the console value matches the render even for a -1 request.
        seed = resolve_seed(chain.seed)

        logger.info(
            "Chain job %s start clips=%d %dx%d steps=%d fps=%g overlap=%d/%.2f seed=%d",
            job.job_id, n, chain.width, chain.height,
            chain.num_inference_steps, chain.frame_rate,
            chain.overlap_frames, chain.overlap_strength, seed,
        )

        try:
            if job.cancel_requested:
                job.status = JobStatus.cancelled
                job.progress = 1.0
                job.completed_at = now_iso()
                self.state = self.STATE_READY
                logger.info("Chain job %s cancelled before dispatch", job.job_id)
                return

            if not self.runner.loaded:
                if not self.config.model.auto_load_on_generate:
                    raise pipeline_load_failed(detail="auto_load_on_generate is disabled")
                self.load()
            self.state = self.STATE_RUNNING

            # Only clip 0 may carry conditioning images (validator enforces this).
            clip0_cond_paths = [
                self.upload_store.path_for(ci.image_id)
                for ci in chain.clips[0].conditioning_images
            ]

            # Style/character IC-LoRA: resolve adapter names -> (path, strength,
            # preprocess) via the registry (mirrors run_job:235-238). The endpoint
            # already validated existence + rejected control adapters, so this
            # re-resolves the same style objects for the runner hop. Empty list
            # when the chain requested no loras (byte-identical default path).
            lora_paths = [
                self.lora_registry.resolve(
                    spec.name, spec.strength, getattr(spec, "audio_strength", None)
                )
                for spec in chain.loras
            ]

            # Reference-video CONTROL IC-LoRA (ALPHA, clips=1 only): resolve
            # reference_video_id -> path via the video store, mirroring run_job
            # (see above). The endpoint already validated existence + the
            # clips=1/preprocess-kind constraints, so this re-resolves the same
            # path for the runner hop. None when the chain requested no reference
            # video (byte-identical default path).
            reference_video_path = (
                self.video_upload_store.path_for(chain.reference_video_id)
                if chain.reference_video_id else None
            )
            # §1-15 metadata provenance: the reference upload's actual measured
            # frame count, best-effort (an ffprobe failure here must not turn a
            # completed job into a 500 — the job already ran successfully with
            # whatever the engine itself decoded). Absent entirely when there is
            # no reference, so a non-reference chain's metadata is unaffected.
            reference_provenance = None
            if reference_video_path is not None:
                reference_provenance = {}
                try:
                    reference_provenance["reference_frames_available"] = (
                        video_io.frame_count(reference_video_path)
                    )
                except (video_io.FFmpegError, OSError):
                    pass

            # Console job-info line (owner requirement): base weight + LoRAs +
            # base prompt (clip overrides propagate from it). Mirrors run_job so
            # LoRA application is visible from the uvicorn console.
            logger.info(
                'Chain job %s base=%s loras=%s prompt="%s"',
                job.job_id,
                self._base_model_name(),
                _loras_for_log(chain.loras, lora_paths),
                _prompt_for_log(chain.prompt),
            )

            # V2V continuation: cut the fps-correct source tail the engine needs
            # (last context_frames frames at the request fps; resampled if the
            # source fps differs). The engine does NOT resample. Provenance
            # (source_fps, resampled) is recorded in the v2v metadata block.
            source_tail_path = None
            source_context_frames = None
            v2v_provenance = None
            if chain.source_video is not None:
                src_path = self.video_upload_store.path_for(chain.source_video.video_id)
                source_context_frames = chain.source_video.context_frames
                source_tail_path = output_dir / "_source_tail.mp4"
                cut = video_io.cut_tail_mp4(
                    src_path, source_tail_path, source_context_frames, chain.frame_rate,
                )
                v2v_provenance = {
                    "source_video_id": chain.source_video.video_id,
                    "source_fps": cut["source_fps"],
                    "resampled": cut["resampled"],
                }

            # A2V: resolve the uploaded audio path (byte-passed to the engine as-is
            # — no cut/resample; the engine truncates to the timeline). Mutually
            # exclusive with source_video (validator enforces this), so only one of
            # source_tail_path / source_audio_path is ever set.
            source_audio_path = None
            a2v_provenance = None
            if chain.source_audio is not None:
                source_audio_path = self.audio_upload_store.path_for(
                    chain.source_audio.audio_id
                )
                a2v_provenance = {"source_audio_id": chain.source_audio.audio_id}

            # Retake: cut the window the engine consumes (frame-exact, CFR, at
            # the request fps; resampled if the upload's cadence differs). The
            # engine never cuts or resamples — same division of labour as the
            # V2V tail above. Provenance records WHERE the window came from so a
            # result can be traced back to the frames it was made of.
            retake_window_path = None
            retake_provenance = None
            if chain.retake is not None:
                rt_src = self.video_upload_store.path_for(chain.retake.video_id)
                retake_window_path = output_dir / "_retake_window.mp4"
                cut = video_io.cut_window_mp4(
                    rt_src, retake_window_path,
                    float(chain.retake.window_start_sec),
                    int(chain.clips[0].num_frames),
                    chain.frame_rate,
                )
                retake_provenance = {
                    "retake_video_id": chain.retake.video_id,
                    "source_fps": cut["source_fps"],
                    "resampled": cut["resampled"],
                    "window_start_sec": float(chain.retake.window_start_sec),
                    "window_start_frame": cut["start_frame"],
                    "written_frames": cut["written_frames"],
                    "upload_has_audio": cut["has_audio"],
                }

            # End source: prepare the ONE mp4 the engine freezes as the chain's
            # tail. Both kinds converge on the same file so the engine never
            # learns about images: a video is cut from its FRONT (the first
            # context_frames+1 frames — the mirror of the V2V tail cut, and the
            # "keep the head, drop the rest" truncation the owner specified for
            # over-long uploads), a still is looped to the same length. The +1 is
            # the causal VAE's primer frame: it is consumed as the lone keyframe
            # latent and never appears in the output, so the delivered mp4 ends
            # with the material's frames 1..context_frames.
            end_source_path = None
            end_source_context_frames = None
            end_source_provenance = None
            end_source_strength = None
            if chain.end_source is not None:
                end_source_context_frames = int(chain.end_source.context_frames)
                end_source_strength = float(chain.end_source.strength)
                cut_frames = end_source_context_frames + 1
                end_source_path = output_dir / "_end_source.mp4"
                if chain.end_source.video_id is not None:
                    es_src = self.video_upload_store.path_for(chain.end_source.video_id)
                    cut = video_io.cut_window_mp4(
                        es_src, end_source_path, 0.0, cut_frames, chain.frame_rate,
                    )
                    end_source_provenance = {
                        "kind": "video",
                        "end_source_video_id": chain.end_source.video_id,
                        "source_fps": cut["source_fps"],
                        "resampled": cut["resampled"],
                        "written_frames": cut["written_frames"],
                        "upload_has_audio": cut["has_audio"],
                    }
                else:
                    es_src = self.upload_store.path_for(chain.end_source.image_id)
                    still = video_io.still_image_mp4(
                        es_src, end_source_path, cut_frames, chain.frame_rate,
                    )
                    end_source_provenance = {
                        "kind": "image",
                        "end_source_image_id": chain.end_source.image_id,
                        "written_frames": still["written_frames"],
                        # Stated rather than omitted: the video branch above
                        # always publishes this key, so a consumer that reads it
                        # to decide "is there material audio to freeze?" would
                        # otherwise have to treat "absent" as an answer. A still
                        # has no audio, and :func:`video_io.still_image_mp4`
                        # writes none. The authoritative record of what the
                        # ENGINE did with it is ``end_source.audio_frozen``.
                        "upload_has_audio": False,
                    }

            def on_progress(step, total, progress, stage=None, clip=None, clip_count=None):
                job.current_step = step
                job.total_steps = total
                job.stage = stage
                # Clip position arrives only with chain stage-1 events; keep the
                # last known value through stage-2/decode so "clip N/N" persists
                # as "all clips are through stage 1".
                if clip is not None:
                    job.clip = clip
                    job.clip_count = clip_count
                job.progress = round(max(0.0, min(1.0, progress)), 3)

            outcome = self.runner.generate_chain(
                chain,
                output_dir=output_dir,
                progress_callback=on_progress,
                clip0_conditioning_paths=clip0_cond_paths,
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

            elapsed = time.time() - started
            meta = outcome.chain_metadata or {}
            total_frames = int(meta.get("total_px", 0))
            duration = round(total_frames / chain.frame_rate, 3) if total_frames else 0.0
            if chain.crop_output is not None:
                resolution = f"{chain.crop_output.width}x{chain.crop_output.height}"
            else:
                resolution = f"{chain.width}x{chain.height}"
            output_path = outcome.output_path
            file_size = output_path.stat().st_size if output_path.exists() else 0

            metadata_path = output_dir / "metadata.json"
            if self.config.output.save_metadata_json:
                self._write_chain_metadata(
                    job=job, chain=chain, metadata_path=metadata_path,
                    resolution=resolution, duration=duration, file_size=file_size,
                    elapsed=elapsed, seed_used=outcome.seed_used,
                    backend=outcome.backend or "mock",
                    peak_vram_mb=outcome.peak_vram_mb, total_frames=total_frames,
                    chain_meta=meta, v2v_provenance=v2v_provenance,
                    a2v_provenance=a2v_provenance,
                    retake_provenance=retake_provenance,
                    end_source_provenance=end_source_provenance,
                    reference_provenance=reference_provenance,
                    attention_used=outcome.attention_used,
                    block_swap_prefetch_used=outcome.block_swap_prefetch_used,
                    keep_resident_used=outcome.keep_resident_used,
                    fused_gguf_dequant_kernel_used=(
                        outcome.fused_gguf_dequant_kernel_used
                    ),
                    vae_mode_used=outcome.vae_mode_used,
                    peak_vram_reserved_mb=outcome.peak_vram_reserved_mb,
                )

            result = JobResult(
                video_url=f"/api/v1/jobs/{job.job_id}/video",
                duration_seconds=duration,
                resolution=resolution,
                file_size_bytes=file_size,
                generation_time_seconds=round(elapsed, 2),
                seed_used=outcome.seed_used,
                output_path=f"outputs/{job.job_id}/output.mp4",
                metadata_path=f"outputs/{job.job_id}/metadata.json",
            )

            job.status = JobStatus.completed
            job.result = result
            job.progress = 1.0
            job.completed_at = now_iso()
            self.state = self.STATE_READY
            logger.info(
                "Chain job %s done in %.1fs%s -> %s (frames=%d)",
                job.job_id, elapsed, _peak_vram_suffix(outcome.peak_vram_mb),
                job.status.value, total_frames,
            )

        except Exception as exc:
            job.completed_at = now_iso()
            job.status = JobStatus.failed
            if _is_oom(exc):
                err = gpu_oom(job_id=job.job_id, detail=str(exc))
                logger.error("Chain job %s OOM: %s", job.job_id, exc)
                self._cleanup_after_error()
            else:
                err = generation_failed(job_id=job.job_id, detail=str(exc))
                logger.exception("Chain job %s failed", job.job_id)
                self.state = self.STATE_READY if self.runner.loaded else self.STATE_UNLOADED
            job.error = (
                f"{err.code}: {err.message} ({err.detail})" if err.detail
                else f"{err.code}: {err.message}"
            )
        finally:
            safe_memory_cleanup()

    def _write_chain_metadata(
        self, *, job, chain, metadata_path, resolution, duration, file_size,
        elapsed, seed_used, backend, peak_vram_mb, total_frames, chain_meta,
        v2v_provenance=None, a2v_provenance=None, retake_provenance=None,
        end_source_provenance=None,
        reference_provenance=None,
        attention_used=None,
        block_swap_prefetch_used=None, keep_resident_used=None,
        fused_gguf_dequant_kernel_used=None, vae_mode_used=None,
        peak_vram_reserved_mb=None,
    ) -> None:
        cm = chain_meta or {}
        metadata = {
            "job_id": job.job_id,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": now_iso(),
            "status": "completed",
            "kind": "chain",
            "request": chain.model_dump(),
            "generation_mode": "chain",
            "seed_used": seed_used,
            # Acceleration: see the same keys in :meth:`_write_metadata`.
            "attention_used": attention_used,
            "block_swap_prefetch_used": block_swap_prefetch_used,
            "keep_resident_used": keep_resident_used,
            "fused_gguf_dequant_kernel_used": fused_gguf_dequant_kernel_used,
            "vae_mode_used": vae_mode_used,
            "peak_vram_reserved_mb": peak_vram_reserved_mb,
            "generation_time_seconds": round(elapsed, 2),
            "backend": backend,
            "chain": {
                "num_clips": len(chain.clips),
                "architecture": "masked_av_latent_concat",
                "overlap_frames": chain.overlap_frames,
                "overlap_strength": chain.overlap_strength,
                "total_frames": total_frames,
                "clip_num_frames": [c.num_frames for c in chain.clips],
                # Junction pixel-frame indices (0-based last-frame-of-segment; the
                # boundary is J / J+1) for the review harness — segment seams AND
                # stage-2 tile seams, plus the ±1 spread that was actually probed.
                "segment_seam_junctions": cm.get("segment_seam_junctions", []),
                "tile_seam_junctions": cm.get("tile_seam_junctions", []),
                "all_junctions": cm.get("all_junctions", []),
                "video_tiles": cm.get("video_tiles", []),
                "n_tiles": cm.get("n_tiles"),
                # Stage-2 window: the preset NAME the request asked for plus the
                # geometry the engine actually laid out with (both backends
                # return chain_math.ChainLayout.to_dict(), which already carries
                # v_tile/kt_v). Recording both is what makes a GPU gate able to
                # prove the opt-in reached the engine.
                "stage2_window": getattr(chain, "stage2_window", "standard"),
                "v_tile": cm.get("v_tile"),
                "kt_v": cm.get("kt_v"),
            },
            "output": {
                "path": f"outputs/{job.job_id}/output.mp4",
                "resolution": resolution,
                "duration_seconds": duration,
                "frame_rate": chain.frame_rate,
                "file_size_bytes": file_size,
            },
            "vram_optimization": self.low_vram.metadata_block(peak_vram_mb=peak_vram_mb),
            # §1-23 (PENDING_TASKS.md): which base model and which model file
            # actually backed this generation, per category. See
            # ``_models_metadata_block``.
            "models": self._models_metadata_block(),
            "environment": self._environment_block(),
        }
        # V2V continuation (additive): only present when a source_video was used,
        # so a normal chain's metadata key set is byte-unchanged. The engine's
        # (or mock's) chain.v2v sub-dict + the app-side provenance (source_video_id,
        # source_fps, resampled).
        v2v = cm.get("v2v")
        if v2v is not None:
            metadata["v2v"] = {**v2v, **(v2v_provenance or {})}
        # A2V continuation (additive): only present when a source_audio was used,
        # so a normal chain's metadata key set is byte-unchanged. The engine's (or
        # mock's) chain.a2v sub-dict + the app-side provenance (source_audio_id).
        a2v = cm.get("a2v")
        if a2v is not None:
            metadata["a2v"] = {**a2v, **(a2v_provenance or {})}
        # Retake (additive): only present when a retake was requested, so a
        # normal chain's metadata key set is byte-unchanged. The engine's (or
        # mock's) chain.retake sub-dict — geometry + runtime + freeze_proof —
        # plus the app-side provenance (which upload, which frames, resampled?).
        retake = cm.get("retake")
        if retake is not None:
            metadata["retake"] = {**retake, **(retake_provenance or {})}
        # End source (additive): only present when an end_source was requested, so
        # a normal chain's metadata key set is byte-unchanged. The engine's (or
        # mock's) chain.end_source sub-dict — geometry + runtime (+ freeze_proof
        # from the real engine, which the mock deliberately never fabricates) —
        # plus the app-side provenance (which upload, which kind, how many frames
        # were actually cut/synthesised).
        end_source = cm.get("end_source")
        if end_source is not None:
            metadata["end_source"] = {**end_source, **(end_source_provenance or {})}
        # §1-15 (clip-wise IC-LoRA reference, additive): only present when a
        # reference video was actually used, so a normal (or style-loras-only)
        # chain's metadata key set is byte-unchanged. Mirrors _write_metadata's
        # single-generate ``ic_lora`` block (same ``loras``/``reference_video_id``/
        # conditioning_attention_strength/reference_video_strength shape), plus
        # the geometry that block has no equivalent for: WHICH slice of the one
        # long reference lands on each stage-1 segment. The mock backend ignores
        # the reference entirely, so this is the only way a real-device gate can
        # confirm "clip i actually got the right window" without eyeballing the
        # video (§1-15 plan, B9).
        if chain.reference_video_id:
            ic_lora_block: dict = {
                "loras": [
                    _lora_metadata_entry(spec, self.lora_registry) for spec in chain.loras
                ],
                "reference_video_id": chain.reference_video_id,
            }
            if chain.conditioning_attention_strength is not None:
                ic_lora_block["conditioning_attention_strength"] = (
                    chain.conditioning_attention_strength
                )
            if chain.reference_video_strength is not None:
                ic_lora_block["reference_video_strength"] = chain.reference_video_strength
            # Per-clip reference windows: chain_math.video_segment_windows is a
            # pure function of clip_frames/fps/kv, all echoed verbatim on the
            # request, so recomputing it here (rather than threading the
            # engine's internal ChainLayout through the runner boundary) is
            # guaranteed to match what chain_pipeline.py actually sliced with.
            # Reference is mutually exclusive with source_video/retake at the
            # API layer (api/models.py), so the plain layout (no
            # source_context_px/retake_glue_px) is always the right one here.
            try:
                layout = chain_math.compute_chain_layout(
                    [c.num_frames for c in chain.clips], chain.frame_rate,
                    kv=chain.overlap_frames,
                )
            except ValueError:
                # A geometrically impossible request would have 422'd at the API
                # layer before generation ever ran, so this recomputation cannot
                # actually fail for a job that got this far -- guarded anyway so
                # a metadata write is never what turns a finished job into a 500.
                pass
            else:
                windows = chain_math.video_segment_windows(layout)
                ic_lora_block["reference_segment_windows"] = [list(w) for w in windows]
                # The reference frame count the windows need in full (the last
                # window's end + 1 — see video_segment_windows's own docstring).
                # Compared against reference_provenance's measured
                # reference_frames_available, this is what tells a reviewer
                # whether every clip got its reference or the upload ran out
                # partway (owner-confirmed behaviour: not an error, later
                # segments just generate unconditioned).
                ic_lora_block["reference_frames_needed"] = layout.total_px
            if reference_provenance:
                ic_lora_block.update(reference_provenance)
            metadata["ic_lora"] = ic_lora_block
        video_io.save_metadata(metadata_path, metadata)

    # ------------------------------------------------------------ finalize

    def _finalize(self, job: JobRecord, outcome, output_dir: Path, elapsed: float) -> JobResult:
        req = job.request
        if req.crop_output is not None:
            res_w, res_h = req.crop_output.width, req.crop_output.height
        else:
            res_w, res_h = req.width, req.height
        resolution = f"{res_w}x{res_h}"
        duration = round(req.num_frames / req.frame_rate, 3)
        file_size = outcome.output_path.stat().st_size if outcome.output_path.exists() else 0

        metadata_path = output_dir / "metadata.json"
        if self.config.output.save_metadata_json:
            self._write_metadata(
                job=job,
                outcome=outcome,
                metadata_path=metadata_path,
                resolution=resolution,
                duration=duration,
                file_size=file_size,
                elapsed=elapsed,
            )

        return JobResult(
            video_url=f"/api/v1/jobs/{job.job_id}/video",
            duration_seconds=duration,
            resolution=resolution,
            file_size_bytes=file_size,
            generation_time_seconds=round(elapsed, 2),
            seed_used=outcome.seed_used,
            output_path=f"outputs/{job.job_id}/output.mp4",
            metadata_path=f"outputs/{job.job_id}/metadata.json",
        )

    def _write_metadata(self, *, job, outcome, metadata_path, resolution, duration, file_size, elapsed) -> None:
        req = job.request
        metadata = {
            "job_id": job.job_id,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": now_iso(),
            "status": "completed",
            "request": req.model_dump(),
            "generation_mode": outcome.generation_mode,
            "seed_used": outcome.seed_used,
            # Acceleration: the attention backend the engine ACTUALLY ran with
            # ("sdpa" | "sage" | "sage->sdpa"), reported by the worker's done
            # event. Unconditional like seed_used — an always-present key is the
            # point: it makes "the request asked for sage but sdpa ran" visible
            # instead of inferable only from logs. None on the mock backend.
            "attention_used": outcome.attention_used,
            # Acceleration: whether block-swap prefetch ACTUALLY ran ("off" |
            # "on" | "on->off"), same relay discipline as attention_used above.
            "block_swap_prefetch_used": outcome.block_swap_prefetch_used,
            # Acceleration: whether the cross-job CPU-skeleton cache actually
            # stayed resident ("off" | "on" | "on->off"), same relay discipline
            # again. This is the ONLY machine-readable place a worker-side
            # auto-downgrade (see engine/worker._resolve_keep_resident) becomes
            # visible — the real-device gate judges on this field, not on logs.
            "keep_resident_used": outcome.keep_resident_used,
            # Acceleration: whether the fused Triton GGUF dequantization kernel
            # actually ran ("off" | "on" | "on->off"), same relay discipline
            # again. "on->off" means the job asked for it but it never applied
            # (Triton missing, kernel exception latched, self-check mismatch, or
            # no eligible tensor) — the real-device gate judges on this field.
            "fused_gguf_dequant_kernel_used": (
                outcome.fused_gguf_dequant_kernel_used
            ),
            # Acceleration: which video VAE decoder actually ran ("off" = stock,
            # "on" = pruned PrunaVAED, "on->off" = asked for but the weight file
            # was missing). Same relay discipline again — and the one field here
            # that also documents WHY two runs with the same seed can differ in
            # fine detail, since the pruned decoder is not bit-identical.
            "vae_mode_used": outcome.vae_mode_used,
            # torch.cuda.max_memory_reserved()-based, additive alongside the
            # vram_optimization block's peak_vram_mb (max_memory_allocated-
            # based) — this feature's VRAM-risk signal (§44).
            "peak_vram_reserved_mb": outcome.peak_vram_reserved_mb,
            "generation_time_seconds": round(elapsed, 2),
            "backend": outcome.backend,
            "output": {
                "path": f"outputs/{job.job_id}/output.mp4",
                "resolution": resolution,
                "duration_seconds": duration,
                "frame_rate": req.frame_rate,
                "file_size_bytes": file_size,
            },
            "vram_optimization": self.low_vram.metadata_block(peak_vram_mb=outcome.peak_vram_mb),
            # §1-23 (PENDING_TASKS.md): the same block as _write_chain_metadata
            # writes — see ``_models_metadata_block``.
            "models": self._models_metadata_block(),
            "environment": self._environment_block(),
        }
        # Phase B IC-LoRA: additive block, only present for lora jobs so non-lora
        # metadata keeps its exact prior key set. (The two new GenerateRequest
        # fields also appear inside the frozen-additive ``request`` dump.)
        if req.loras:
            metadata["ic_lora"] = {
                # Phase C: additive ``preprocess`` field (control-signal kind per
                # adapter). Existing ``name``/``strength``/``reference_video_id``
                # keys are unchanged so Phase B metadata parsers keep working.
                "loras": [_lora_metadata_entry(spec, self.lora_registry) for spec in req.loras],
                "reference_video_id": req.reference_video_id,
            }
            # Control-adjustability overrides: record only when meaningful
            # (mirrors the source_audio precedent), so an omitted-field lora job's
            # ic_lora block stays byte-identical to before.
            if req.conditioning_attention_strength is not None:
                metadata["ic_lora"]["conditioning_attention_strength"] = (
                    req.conditioning_attention_strength
                )
            if req.reference_video_strength is not None:
                metadata["ic_lora"]["reference_video_strength"] = (
                    req.reference_video_strength
                )
        video_io.save_metadata(metadata_path, metadata)

    def _environment_block(self) -> dict:
        torch_version = None
        cuda_version = None
        try:
            import torch  # type: ignore

            torch_version = torch.__version__
            cuda_version = getattr(torch.version, "cuda", None)
        except Exception:
            pass
        gpu = gpu_info.get_gpu_info()
        return {
            "python": platform.python_version(),
            "torch": torch_version,
            "cuda": cuda_version,
            "gpu": gpu.get("name"),
            "platform": sys.platform,
        }
