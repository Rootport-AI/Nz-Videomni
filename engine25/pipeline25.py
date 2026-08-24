"""The LTX 2.5 two-stage generation pipeline, assembled from GGUF weights (§3-98 Phase 2d).

What this module is
-------------------
Phases 2b and 2c built the two halves that cannot come from the official
packages on a 16 GB card -- a transformer stage that loads a Q4 GGUF, keeps it
on the CPU and pages 48 blocks through a small GPU window, and a Gemma 4 text
encoder that does the same for its 48 layers. This module is the assembly: it
constructs the OFFICIAL :class:`DistilledPipeline`, substitutes those two
components, and exposes one ``generate`` call that writes an mp4.

Everything the official code can still do, it still does. The video VAE, the
audio VAE, the spatial upsampler and the image conditioner are plain
``safetensors`` and are built by the stock loaders with no involvement from
engine25 at all. The denoising loops, the conditioning maths, the two-stage
schedule, the mp4 muxing -- all official, unchanged.

The three substitutions
-----------------------
1. ``pipeline.stage`` -> :class:`Ltx25ProgressStage` (a
   :class:`~engine25.gguf_transformer.Ltx25DiffusionStage` that also reports
   per-step progress). The stock stage builds the whole 14.7 GB checkpoint
   straight onto the GPU; ours does not.
2. ``pipeline.prompt_encoder`` -> :class:`Ltx25PromptEncoder`. This is a
   SUBCLASS of the official ``PromptEncoder`` constructed through its public
   ``text_encoder_builder=`` parameter, so the encode/enhance/process
   choreography is the official one. Only the two builders behind it are ours.
3. ``pipeline.use_ancestral_sampler = True`` (fact B). ``DistilledPipeline``
   resolves that flag by reading ``model_version`` out of a *safetensors*
   header; handed a ``.gguf`` it logs a warning, returns ``()``, and silently
   selects the deterministic Euler sampler -- i.e. a different generation than
   the checkpoint was distilled for. The transformer GGUF does carry
   ``model_version=2.5.0`` in its KV block, so the correct answer is known; it
   is asserted here rather than hoped for, and republished on the ``ready``
   event so the app can see which sampler the process actually holds.

Why the prompt encoder needed only a subclass
---------------------------------------------
Because of the assets-only ``.safetensors`` from Phase 2c (fact F2). The
official ``PromptEncoder.__init__`` reads the tokenizer/config/processor
sidecars from ``model_paths.text_encoder()`` before any builder is consulted;
pointing that at the assets-only file satisfies all of it. What remains is
purely "where do the weights come from", and the official class already has a
``text_encoder_builder=`` parameter for exactly that. The one thing it has no
parameter for is the EmbeddingsProcessor builder, which is why
:class:`Ltx25PromptEncoder` assigns ``_embeddings_processor_builder`` -- one
private attribute, checked for existence first so a rename fails loudly instead
of leaving the official (safetensors-reading) builder quietly in place.

The v1 generation contract
--------------------------
:meth:`Ltx25Pipeline.generate` accepts what LTX 2.5 v1 supports and nothing
else: prompt, width/height (multiples of 64), ``num_frames`` (8n+1), frame
rate, seed, and zero or more conditioning images (T2V / I2V). Everything the
app may still send -- ``negative_prompt``, ``guidance_scale``,
``num_inference_steps``, the acceleration knobs this engine has no path for --
is IGNORED WITH A LOG LINE, never silently: the distilled 2.5 model runs a fixed
8 + 3 sigma schedule with no classifier-free guidance, so a step count or a CFG
scale has nothing to attach to. ``crop_output`` does not appear here at all, by
design: it is an ffmpeg post-process the app already performs on the finished
mp4 (see ``services/engines/ltx/adapter.py``), engine-independent in both
engines.

The acceleration knobs that DO apply are not generation parameters and are not
on ``generate``'s signature: they are per-job state on a resident worker
process, armed by :meth:`Ltx25Pipeline.set_acceleration_job` before the job and
disarmed in its ``finally``. So far that is the fused Triton GGUF
dequantization kernels, which work here because this engine's transformer and
text encoder both dequantize through 2.3's ``engine.gguf.quant_service``; block
swap prefetch is accepted by the same call and does nothing yet.

Determinism
-----------
Same seed, same file. The video half gets there on its own -- two runs at one
seed produce a bit-identical video stream -- but the audio vocoder does not: its
transposed convolutions reduce with atomics, so three decodes of ONE fixed
latent give three different waveforms. :func:`enable_deterministic_convolutions`
(on by default, one constructor flag away from off) pins cuDNN and closes that
gap. See its docstring for the measurement and the cost caveat.

fps rounding (fact M)
---------------------
``frame_rate`` stays a float everywhere the model sees it -- RoPE positions and
the audio/video latent grids are built from it, and rounding there is what
makes long clips drift out of sync. It is rounded exactly once, at the mp4
encode, because the container wants an integer rate.
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from engine.gguf import dequant_triton
from engine25 import assets_export
from engine25.gguf_gemma4 import (
    build_embeddings_processor_builder,
    build_text_encoder_builder,
)
from engine25.gguf_transformer import (
    Ltx25DiffusionStage,
    _move_module_tree,
    _rss_bytes,
    _Vram,
)
from engine25.ltxcore_compat import (
    AUTO_TILING,
    DISTILLED_SIGMAS,
    STAGE_2_DISTILLED_SIGMAS,
    AllocatorTrimStrategy,
    AudioConditioner,
    AutoTiling,
    DistilledPipeline,
    ImageConditioningInput,
    ModelPaths,
    OffloadMode,
    PromptEncoder,
    TilingConfig,
    VideoPixelShape,
    cleanup_memory,
    encode_video,
    ensure_tiling_config,
    get_video_chunks_number,
    tiling_scale_factors_for_vae,
    verify,
)
from engine25.reference25 import (
    reference_patch,
    resolve_reference_downscale_factor,
)

logger = logging.getLogger(__name__)

#: The sampler this engine reports on ``ready``. Not a preference: 2.5 distilled
#: was trained with the ancestral (SDE) Euler stage-1 sampler, and
#: ``ANCESTRAL_SAMPLER_SINCE_VERSION`` in the official pipeline is ``(2, 5)``.
SAMPLER_NAME = "euler_ancestral"

#: Blocks of the 48 that stay resident on the GPU. 8 is the plan's starting
#: point; the documented 16 GB fallback ladder is 8 -> 6 -> 4 -> explicit VAE
#: tiling, and every rung is a constructor argument rather than an edit.
DEFAULT_BLOCKS_ON_GPU = 8

#: Gemma layers kept resident. 0 = every layer is streamed. The text encoder
#: runs to completion before the first transformer build, so its residency does
#: not compete with the diffusion peak -- but it does compete with nothing being
#: gained, either: at 0 the encode still takes seconds, so 0 is simply free.
DEFAULT_TE_LAYERS_ON_GPU = 0

#: Resolution divisor the two-stage pipeline requires (stage 1 runs at half).
RESOLUTION_DIVISOR = 64

#: Temporal grid of the causal video VAE: frame counts must be ``8n + 1``.
FRAME_GRID = 8

#: Progress stage names. Deliberately the SAME strings the 2.3 worker emits, so
#: the app-side receipt loop (``_CHAIN_STAGE_LABELS`` / ``_progress_frac`` in
#: services/engines/ltx/adapter.py) maps them without a new branch.
STAGE_ENCODE = "encode"
STAGE_1_DENOISE = "stage1_denoise"
STAGE_2_DENOISE = "stage2_denoise"
STAGE_DECODE = "decode"

#: ``(stage_index -> progress stage name)`` for the denoise loops. The stage
#: object is called twice per job by ``DistilledPipeline.__call__`` and is given
#: no other way to know which call it is on, so the invocation counter (reset
#: per job) is what names them -- the same inference the 2.3 worker's shim makes.
_DENOISE_STAGE_NAMES = {1: STAGE_1_DENOISE, 2: STAGE_2_DENOISE}

#: ``progress(stage, index, total, *, outer_index=None, outer_total=None)``.
#: ``index`` is the number of COMPLETED units for coarse stages and completed
#: steps for the denoise stages, matching what the 2.3 adapter already assumes
#: for each name.
#:
#: The two keyword-only arguments are the chain's position (which clip, which
#: tile) and are the SAME pair the 2.3 worker emits from
#: ``engine/worker.py``'s progress shim. They are optional, and every caller
#: that does not need them is called with THREE POSITIONAL ARGUMENTS exactly as
#: before -- a single-generation job never passes them, so its receipts and any
#: three-argument receiver (``engine25/worker.py:_emit_progress``) are unchanged.
#: Without them the app's chain receipt loop cannot tell step 3-of-8 of clip 1
#: from step 3-of-8 of clip 2, and the job fraction rewinds at every clip.
ProgressCallback = Callable[..., None]


class Ltx25PipelineError(RuntimeError):
    """The LTX 2.5 pipeline could not be assembled or run."""


def enable_deterministic_convolutions() -> dict[str, bool]:
    """Pin cuDNN to deterministic convolution algorithms. Returns the previous state.

    MEASURED, not precautionary. With the stock settings, decoding one fixed
    audio latent three times in a single process yields three different
    waveforms: the audio VAE's vocoder runs ``F.conv_transpose1d`` and
    ``nn.ConvTranspose1d``, whose cuDNN algorithms reduce with atomics, so the
    summation order varies run to run. The video half is unaffected -- same-seed
    runs already produce a bit-identical video stream -- which is exactly why
    this is worth naming precisely: without it the mp4 differs on every run
    *because of the audio track alone*, and the whole same-seed-same-file
    contract the app is built around (job SHAs, regression gates) would be false
    for the 2.5 engine while looking like a model problem.

    Turning ``benchmark`` off as well is part of the same statement: benchmark
    mode picks an algorithm by timing it, so it can select a different one on a
    warm machine than on a cold one.

    Cost: cuDNN loses the freedom to pick the fastest algorithm for the VAE
    convolutions. At 320x192/25f the whole decode+encode phase is 0.2 s, so the
    cost is unmeasurable there; it has NOT been measured at production
    resolution, which is why this is a constructor flag rather than a constant.
    """
    previous = {
        "deterministic": bool(torch.backends.cudnn.deterministic),
        "benchmark": bool(torch.backends.cudnn.benchmark),
    }
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return previous


# ---------------------------------------------------------------------------
# Request / result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelFiles:
    """The five files one loaded LTX 2.5 model is made of.

    ``text_encoder`` is the text-encoder **GGUF**, not the assets-only export:
    the export is derived from it (and regenerated when stale) inside
    :meth:`Ltx25Pipeline.__init__`, so callers hand over the weights and the
    engine owns the derived file. ``text_encoder_assets`` overrides that
    location for a deployment that keeps the two apart.
    """

    transformer: str
    text_encoder: str
    video_vae: str
    audio_vae: str
    spatial_upsampler: str
    text_encoder_assets: str | None = None

    def missing(self) -> list[str]:
        """Names of the required files that are not on disk."""
        return [
            name
            for name in ("transformer", "text_encoder", "video_vae", "audio_vae", "spatial_upsampler")
            if not Path(getattr(self, name)).is_file()
        ]


@dataclass(frozen=True)
class GenerationResult:
    """What one ``generate`` produced, plus the measurements the gates want."""

    output_path: str
    seed: int
    width: int
    height: int
    num_frames: int
    frame_rate: float
    encode_fps: int
    num_images: int
    size_bytes: int
    seconds: float
    phases: dict[str, dict[str, Any]] = field(default_factory=dict)
    peak_allocated_gib: float | None = None
    peak_reserved_gib: float | None = None
    rss_peak_gib: float | None = None
    video_chunks: int = 1
    tiling: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "frame_rate": self.frame_rate,
            "encode_fps": self.encode_fps,
            "num_images": self.num_images,
            "size_bytes": self.size_bytes,
            "seconds": round(self.seconds, 2),
            "video_chunks": self.video_chunks,
            "tiling": self.tiling,
            "peak_allocated_gib": self.peak_allocated_gib,
            "peak_reserved_gib": self.peak_reserved_gib,
            "rss_peak_gib": self.rss_peak_gib,
            "phases": self.phases,
        }


# ---------------------------------------------------------------------------
# Component substitutions
# ---------------------------------------------------------------------------


class _GpuPlacedBuilder:
    """Wraps an engine25 CPU builder so the built model ends up on *device*.

    :class:`~engine25.gguf_transformer.Ltx25CpuModelBuilder` deliberately ignores
    the requested device: for the 14.7 GB transformer, placement is a selective
    operation the stage performs. The EmbeddingsProcessor is the opposite case --
    a ~1.5 GB model that has to sit wholly on the GPU because the hidden states
    it consumes are already there -- so this puts the move back.

    A wrapper rather than a subclass because the builder is produced by
    :func:`engine25.gguf_gemma4.build_embeddings_processor_builder`, which owns
    the (four-way) configuration of that builder; re-deriving it here to change
    one line would duplicate the part most likely to drift. Only ``build`` is
    intercepted; everything else is delegated, so ``with_*``/``model_config``
    still work if a later phase needs them.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def build(self, device: torch.device | None = None, dtype: torch.dtype | None = None, **kwargs: Any) -> Any:
        model = self._inner.build(device=device, dtype=dtype, **kwargs)
        if device is not None:
            moved = _move_module_tree(model, device, skip=set())
            logger.info("EmbeddingsProcessor: moved %d tensors to %s", moved, device)
        return model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _CountingDenoiser:
    """Passes a denoiser through, counting the steps that flow past it.

    Both official denoising loops -- ``euler_denoising_loop`` and the ancestral
    driver -- call ``denoiser(transformer, video_state, audio_state, sigmas,
    step_idx)`` exactly once per step and drive their progress bar off a plain
    ``tqdm``. Wrapping the denoiser is therefore an exact step counter that
    needs no patching of tqdm, no global state, and no assumption about the
    loop's internals beyond its documented call contract. (This is why §3-98
    rules out porting the 2.3 ``progress_shim``: there is a legitimate seam here
    and the 2.3 shim's tqdm swap exists only because the 2.3 wheel has none.)
    """

    def __init__(self, inner: Any, total: int, report: Callable[[int, int], None]) -> None:
        self._inner = inner
        self._total = max(1, int(total))
        self._report = report
        self._done = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self._inner(*args, **kwargs)
        self._done += 1
        try:
            self._report(self._done, self._total)
        except Exception as exc:  # noqa: BLE001 -- progress must never fail a job
            logger.warning("progress report failed (ignored): %r", exc)
        return result


class Ltx25ProgressStage(Ltx25DiffusionStage):
    """The GGUF diffusion stage, plus per-step progress and per-stage timing.

    Overrides ``__call__`` only to wrap the ``denoiser`` argument and to time the
    call; the whole body is then ``super().__call__``, i.e. the official
    build-denoise-free sequence with engine25's builder underneath it.

    ``begin_job()`` resets the invocation counter. The counter is what turns two
    identical calls into ``stage1_denoise`` and ``stage2_denoise``: the stage is
    not told which one it is on, and inferring it from the sigma count would
    couple the naming to a schedule length that is a property of the checkpoint.

    A CHAIN calls this stage many more than twice -- once per stage-1 clip and
    once per stage-2 tile -- so the invocation counter cannot name those calls.
    :meth:`announce` is the seam: the caller says what the next call is, and the
    counter is used only when it has not. Everything about the single-generation
    path is therefore unchanged, down to the phase strings: ``begin_job()``
    clears any announcement, so a job that never announces gets exactly the
    ``stage1_denoise`` / ``stage2_denoise`` + ``21_`` / ``22_`` naming it had.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.progress: ProgressCallback | None = None
        self.vram: _Vram | None = None
        self._invocation = 0
        self._announced: tuple[str | None, str | None, int | None, int | None] = (None, None, None, None)

    def begin_job(self) -> None:
        """Reset the invocation counter AND any announcement. Call once per job.

        ``super()`` first, always: the base stage's ``begin_job`` releases the
        previous job's IC-LoRA attachment, and a path that reset only the
        progress counters would leave that teardown to chance.
        """
        super().begin_job()
        self._invocation = 0
        self._announced = (None, None, None, None)

    def announce(
        self,
        stage_name: str | None = None,
        *,
        vram_phase: str | None = None,
        outer_index: int | None = None,
        outer_total: int | None = None,
    ) -> None:
        """Name the NEXT ``__call__`` explicitly (chain only; consumed once).

        ``stage_name`` is the progress stage the per-step events carry,
        ``vram_phase`` the key the per-phase peak is recorded under, and
        ``outer_index`` / ``outer_total`` the chain position forwarded to the
        progress callback as keyword arguments. Consumed by the next call and
        reset afterwards, so a stray announcement cannot leak into a later job's
        naming -- the failure mode would be silent, and a mislabelled phase in
        ``done`` is exactly the sort of number a gate would then read wrong.
        """
        self._announced = (stage_name, vram_phase, outer_index, outer_total)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._invocation += 1
        stage_name, vram_phase, outer_index, outer_total = self._announced
        self._announced = (None, None, None, None)
        name = stage_name or _DENOISE_STAGE_NAMES.get(self._invocation, "denoise")
        phase = vram_phase or f"2{self._invocation}_{name}"

        # ``denoiser`` and ``sigmas`` are the first two parameters, so a caller
        # may pass either by position or by keyword; both are handled rather
        # than assumed, since the official pipeline's call style is not part of
        # any contract engine25 can hold it to.
        denoiser = kwargs["denoiser"] if "denoiser" in kwargs else (args[0] if args else None)
        sigmas = kwargs["sigmas"] if "sigmas" in kwargs else (args[1] if len(args) > 1 else None)
        if self.progress is not None and denoiser is not None and sigmas is not None:
            progress = self.progress

            def report(
                done: int,
                total: int,
                _name: str = name,
                _outer_index: int | None = outer_index,
                _outer_total: int | None = outer_total,
            ) -> None:
                # THREE POSITIONAL ARGUMENTS when there is no chain position:
                # the single-generation receipt path (and any three-argument
                # receiver such as engine25/worker.py's ``_emit_progress``) must
                # keep seeing the call it has always seen.
                if _outer_index is None and _outer_total is None:
                    progress(_name, done, total)
                else:
                    progress(_name, done, total, outer_index=_outer_index, outer_total=_outer_total)

            counting = _CountingDenoiser(denoiser, len(sigmas) - 1, report)
            if "denoiser" in kwargs:
                kwargs["denoiser"] = counting
            else:
                args = (counting, *args[1:])

        if self.vram is not None:
            self.vram.reset()
        started = time.perf_counter()
        try:
            return super().__call__(*args, **kwargs)
        finally:
            if self.vram is not None:
                self.vram.record(phase, time.perf_counter() - started)


class Ltx25PromptEncoder(PromptEncoder):
    """The official prompt encoder, fed by engine25's two GGUF builders.

    The text encoder goes in through the public ``text_encoder_builder=``
    parameter. The EmbeddingsProcessor has no such parameter, so its builder is
    assigned onto ``_embeddings_processor_builder`` -- verified to exist first,
    because a silent rename upstream would leave the official (safetensors)
    builder in place and the failure would surface much later as a
    FileNotFoundError on the assets-only file, which has no weights in it.
    """

    def __init__(
        self,
        model_paths: ModelPaths,
        dtype: torch.dtype,
        device: torch.device,
        *,
        text_encoder_builder: Any,
        embeddings_processor_builder: Any,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ) -> None:
        super().__init__(
            model_paths,
            dtype,
            device,
            text_encoder_builder=text_encoder_builder,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        if not hasattr(self, "_embeddings_processor_builder"):
            raise Ltx25PipelineError(
                "PromptEncoder no longer has an `_embeddings_processor_builder` attribute; "
                "engine25 cannot route the EmbeddingsProcessor at the GGUF pair without it."
            )
        self._embeddings_processor_builder = embeddings_processor_builder
        self.progress: ProgressCallback | None = None
        self.vram: _Vram | None = None

    def __call__(self, prompts: list[str], **kwargs: Any) -> Any:
        if self.progress is not None:
            self.progress(STAGE_ENCODE, 0, 1)
        if self.vram is not None:
            self.vram.reset()
        started = time.perf_counter()
        try:
            return super().__call__(prompts, **kwargs)
        finally:
            if self.vram is not None:
                self.vram.record("10_prompt_encode", time.perf_counter() - started)
            if self.progress is not None:
                self.progress(STAGE_ENCODE, 1, 1)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_geometry(width: int, height: int, num_frames: int) -> None:
    """Reject a geometry the two-stage pipeline cannot run.

    The app validates the same three rules at the API boundary
    (``api/models.py``), so this is the worker-side backstop for a payload that
    reached the engine another way -- and, on the selftest CLI, the only check
    there is. Stated as three separate branches rather than one compound
    condition so the message names exactly what is wrong.
    """
    if width % RESOLUTION_DIVISOR or height % RESOLUTION_DIVISOR:
        raise Ltx25PipelineError(
            f"{width}x{height} is not a multiple of {RESOLUTION_DIVISOR}. The two-stage pipeline "
            f"runs stage 1 at half resolution, so both sides must divide by {RESOLUTION_DIVISOR}."
        )
    if width <= 0 or height <= 0:
        raise Ltx25PipelineError(f"width and height must be positive, got {width}x{height}")
    if num_frames < 1 or (num_frames - 1) % FRAME_GRID:
        raise Ltx25PipelineError(
            f"num_frames={num_frames} is not on the causal VAE's temporal grid; it must be "
            f"{FRAME_GRID}n + 1 (9, 17, 25, ... 121, ...)."
        )


#: Request fields v1 accepts but does not act on, and why. Logged (once per
#: generate, only for the ones actually present) rather than rejected: the app
#: sends a full GenerateRequest and rejecting on arrival would make every
#: default-valued field a hard error. The API layer is where non-default values
#: of the *unsupported* fields become a 422 (Phase 5); this list is the set that
#: is safe to drop on the floor whatever the value.
IGNORED_FIELDS: dict[str, str] = {
    "negative_prompt": "2.5 distilled runs without classifier-free guidance",
    "guidance_scale": "2.5 distilled runs without classifier-free guidance",
    "num_steps": "the distilled schedule is fixed at 8 + 3 sigmas",
    "num_inference_steps": "the distilled schedule is fixed at 8 + 3 sigmas",
    "neg_method": "no negative-prompt mechanism in v1",
    "vsf_scale": "no negative-prompt mechanism in v1",
    "block_swap_prefetch": "2.5 uses engine25's own block-swap window",
    "attention_backend": "v1 is SDPA-only",
    "keep_resident": "2.5 keeps its weights in the registry instead",
    "vae_mode": "v1 uses the Conv VAE only",
}


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


class Ltx25Pipeline:
    """One loaded LTX 2.5 model, ready to generate.

    Construction is cheap and does no I/O worth naming: the assets-only
    text-encoder file is refreshed if stale (~30 MB), the official pipeline
    object is built -- every one of its component builders is lazy, so no
    checkpoint is opened -- and the two engine25 components are substituted in.
    Measured at 3.9 s for the whole of it. Weights arrive on the first
    :meth:`generate`, and every model except the retained transformer state dict
    is freed again before that call returns.
    """

    def __init__(
        self,
        files: ModelFiles,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        blocks_on_gpu: int = DEFAULT_BLOCKS_ON_GPU,
        te_layers_on_gpu: int = DEFAULT_TE_LAYERS_ON_GPU,
        cache_weights: bool = True,
        deterministic: bool = True,
        progress: ProgressCallback | None = None,
    ) -> None:
        verify()

        missing = files.missing()
        if missing:
            raise FileNotFoundError(
                "LTX 2.5 model files are missing: "
                + ", ".join(f"{name}={getattr(files, name)}" for name in missing)
            )

        self.files = files
        self.device = device or _default_device()
        self.dtype = dtype
        self.blocks_on_gpu = int(blocks_on_gpu)
        self.te_layers_on_gpu = int(te_layers_on_gpu)
        self.cache_weights = bool(cache_weights)
        self.deterministic = bool(deterministic)
        self.progress = progress
        self.vram = _Vram(self.device)
        self.build_report: dict[str, Any] = {}

        # -- per-job acceleration state ---------------------------------------
        # Armed by :meth:`set_acceleration_job` at the worker's entry point and
        # cleared by :meth:`reset_acceleration_job` in its ``finally``; the
        # ``generate`` / ``run_chain`` signatures carry neither knob, so a caller
        # that never arms anything (the spike scripts under ``outputs/``, a
        # direct library user) gets both features OFF, which is the safe default.
        #
        # The two halves live in different places, exactly as they do in 2.3:
        #
        # * the fused GGUF dequantization kernels have NO state here at all --
        #   it is MODULE globals in ``engine/gguf/dequant_triton``, because the
        #   dequantization call sites are plain functions deep inside the GGUF
        #   loaders with no pipeline handle to reach. This class only forwards
        #   arm/reset and reads the finished job's verdict back out.
        # * block-swap prefetch keeps the request here because the flag has to
        #   be re-applied to the block-swap service on every transformer build.
        #   In C1 it is REQUEST-ONLY: nothing reads it yet (see
        #   :meth:`block_swap_prefetch_used`).
        self._block_swap_prefetch_requested = False

        started = time.perf_counter()
        self.vram.reset()

        if self.deterministic:
            previous = enable_deterministic_convolutions()
            logger.info("cuDNN pinned to deterministic convolutions (was %s)", previous)
        self.build_report["deterministic"] = self.deterministic

        # -- F2: the assets-only text encoder the official code reads ---------
        export = assets_export.ensure_assets_only(files.text_encoder, files.text_encoder_assets)
        self.assets_path = str(export.path)
        self.build_report["assets_export"] = export.as_dict()

        # -- the official pipeline --------------------------------------------
        # duration_head_path stays None: the split 2.5 pack has no DurationHead
        # file, so `DurationPredictor.from_checkpoint(None, ...)` returns None
        # and `require_num_frames_source` insists on an explicit num_frames --
        # which the v1 contract always supplies. AutoDuration is out of scope.
        model_paths = ModelPaths.from_split(
            transformer_path=files.transformer,
            text_encoder_path=self.assets_path,
            video_vae_path=files.video_vae,
            audio_vae_path=files.audio_vae,
            duration_head_path=None,
        )
        # `loras=[]` is passed explicitly because it is a REQUIRED positional
        # parameter with no default -- and because an explicit empty list is the
        # statement that this engine runs no LoRA, which v1 rejects at the API.
        # offload_mode=NONE: the official OffloadMode paths are a *different*
        # streaming implementation (StreamingModelBuilder) that would fight
        # engine25's block-swap window for the same GPU budget.
        pipeline = DistilledPipeline(
            model_paths=model_paths,
            spatial_upsampler_path=files.spatial_upsampler,
            loras=[],
            device=self.device,
            registry=None,
            offload_mode=OffloadMode.NONE,
        )

        # -- fact B: the ancestral sampler ------------------------------------
        # Resolved (wrongly) in __init__ from a safetensors header read that a
        # GGUF path cannot satisfy; corrected here, then asserted, because a
        # silent downgrade to deterministic Euler is a different generation and
        # would be invisible in the output.
        detected = bool(pipeline.use_ancestral_sampler)
        pipeline.use_ancestral_sampler = True
        if not pipeline.use_ancestral_sampler:  # pragma: no cover -- property-shadowing guard
            raise Ltx25PipelineError(
                "use_ancestral_sampler did not stick after assignment; DistilledPipeline turned it "
                "into a read-only property and engine25 can no longer select the 2.5 sampler."
            )
        logger.info(
            "use_ancestral_sampler: detected=%s -> forced True (sampler=%s). "
            "The GGUF checkpoint declares model_version=2.5.0 in its KV block; the official "
            "detector only reads safetensors headers.",
            detected, SAMPLER_NAME,
        )
        self.build_report["use_ancestral_sampler"] = {"detected": detected, "forced": True}

        # -- substitution 1: the diffusion stage -------------------------------
        stage = Ltx25ProgressStage.from_gguf(
            files.transformer,
            device=self.device,
            dtype=self.dtype,
            blocks_on_gpu=self.blocks_on_gpu,
            cache_weights=self.cache_weights,
        )
        stage.vram = self.vram
        pipeline.stage = stage
        self.stage = stage

        # -- substitution 2: the prompt encoder --------------------------------
        # The transformer GGUF leads the EmbeddingsProcessor's path list: the
        # official configurator reads `config.transformer` and
        # `gemma_source_checkpoint` from path[0], and only the four
        # `text_embedding_projection.*` tensors come from the TE side.
        text_encoder_builder = build_text_encoder_builder(
            files.text_encoder,
            device=self.device,
            assets_path=self.assets_path,
            layers_on_gpu=self.te_layers_on_gpu,
        )
        embeddings_builder = _GpuPlacedBuilder(
            build_embeddings_processor_builder(
                files.text_encoder,
                files.transformer,
                assets_path=self.assets_path,
            )
        )
        prompt_encoder = Ltx25PromptEncoder(
            model_paths,
            self.dtype,
            self.device,
            text_encoder_builder=text_encoder_builder,
            embeddings_processor_builder=embeddings_builder,
        )
        prompt_encoder.vram = self.vram
        pipeline.prompt_encoder = prompt_encoder
        self.prompt_encoder = prompt_encoder

        # -- addition: the audio ENCODER's lifecycle block (§3-102 C1) ---------
        # ``DistilledPipeline`` has an image conditioner (video encoder) and an
        # audio DECODER, but no audio encoder: nothing in a plain generation ever
        # turns a waveform into a latent. V2V and A2V both do, so the chain needs
        # the block the official A2V pipeline uses -- same class, same arguments
        # ``DistilledPipeline`` would have passed (``registry=None`` gives it the
        # private ``cache_models=True, cache_weights=False`` registry the other
        # conditioners get).
        #
        # Constructed unconditionally and eagerly, with no lazy wrapper: the
        # constructor only builds a ``Builder``, opens no file and touches no
        # GPU (``ltxcore_compat.verify`` pins that signature). The ~46MB encoder
        # itself is built and freed inside ``audio_conditioner(fn)``, so a chain
        # with neither a source video nor a source audio never loads it -- which
        # is what a lazy mechanism would have bought, for the price of a
        # mechanism.
        self.audio_conditioner = AudioConditioner(
            files.audio_vae, self.dtype, self.device, registry=None
        )

        self.pipeline = pipeline
        self.vram.record("00_pipeline_build", time.perf_counter() - started)
        self.build_report["device"] = str(self.device)
        self.build_report["blocks_on_gpu"] = self.blocks_on_gpu
        self.build_report["te_layers_on_gpu"] = self.te_layers_on_gpu
        self.build_report["cache_weights"] = self.cache_weights
        self.build_report["sampler"] = SAMPLER_NAME
        self.build_report["stage_1_steps"] = int(DISTILLED_SIGMAS.numel()) - 1
        self.build_report["stage_2_steps"] = int(STAGE_2_DISTILLED_SIGMAS.numel()) - 1
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            self.build_report["gpu"] = {"name": props.name, "total_gib": round(props.total_memory / 2**30, 2)}
        logger.info(
            "LTX 2.5 pipeline ready on %s (sampler=%s, %d + %d steps, blocks_on_gpu=%d)",
            self.device, SAMPLER_NAME,
            self.build_report["stage_1_steps"], self.build_report["stage_2_steps"], self.blocks_on_gpu,
        )

    # -- generation ----------------------------------------------------------

    @property
    def sampler(self) -> str:
        return SAMPLER_NAME

    # -- acceleration knobs (per job) ----------------------------------------

    def set_acceleration_job(
        self,
        *,
        block_swap_prefetch: bool,
        fused_gguf_dequant_kernel: bool,
    ) -> None:
        """Arm this job's two acceleration knobs.

        Called by the worker BEFORE the try block that runs the job, and paired
        with :meth:`reset_acceleration_job` in that block's ``finally``. Both
        halves are per-JOB state on a resident worker process, so an arm without
        a matching reset would leak one job's request into the next one.

        The ordering constraint is on the fused half and it is not negotiable:
        the flag has to be armed **before the transformer is built**, because the
        build is where the GGUF weights are dequantized. Arming it after
        ``generate`` had started would leave the whole first build on the eager
        path and only catch a later rebuild. Same argument in 2.3
        (``engine/pipeline/fast_video_pipeline.py``'s entry points), for the same
        reason; ``set_job`` itself only assigns module globals -- no import, no
        CUDA -- so there is nothing here that can fail, and the call ORDER is
        what makes the arrangement safe rather than any exception handling.

        The prefetch half is request-only in C1: the flag is stored, and nothing
        reads it. C2 wires it to engine25's block-swap window.
        """
        self._block_swap_prefetch_requested = bool(block_swap_prefetch)
        dequant_triton.set_job(bool(fused_gguf_dequant_kernel))

    def reset_acceleration_job(self) -> None:
        """End-of-job counterpart: freeze both verdicts and disarm.

        **This method never raises.** It runs in the worker's ``finally``, so an
        exception here would replace the job's real error with this one -- and on
        a failed job it is precisely the paths that already went wrong that this
        has to clean up. The two halves therefore get their own try/except: a
        failure to tear down prefetch must not leave the fused kernels armed for
        the next job, and vice versa.

        ``self.stage is None`` (i.e. after :meth:`close`) is absorbed by the same
        handlers rather than by a guard of its own: a reset arriving after the
        pipeline was closed is a shutdown race, not a bug worth failing on.
        """
        try:
            # C1: request-only, so "tear down" is just forgetting the request.
            # C2 replaces this with the block-swap service teardown, which is
            # also where ``self.stage is None`` becomes reachable.
            self._block_swap_prefetch_requested = False
        except Exception:  # pragma: no cover -- never-raise discipline
            logger.exception("block-swap prefetch reset failed; continuing")
        try:
            # The verdict ("off" / "on" / "on->off") is computed INSIDE
            # ``reset_job`` from the request, the exception latch and the number
            # of tensors actually dequantized on Triton -- a job that asked for
            # the kernels but dequantized nothing eligible is a degradation, same
            # as a latch. Which is why the verdict has to be read AFTER this.
            dequant_triton.reset_job()
        except Exception:  # pragma: no cover -- never-raise discipline
            logger.exception("fused GGUF dequant reset failed; continuing")

    def block_swap_prefetch_used(self) -> str:
        """What the last finished job's block swap actually did: "off", "on", or
        "on->off" (asked for, but degraded to the synchronous path).

        C1 ALWAYS RETURNS "off": engine25's block-swap window does not implement
        prefetching yet, so no job can have used it, and reporting anything else
        would be a claim the engine cannot back. C2 replaces the body with the
        real per-build tally; the method exists now so the worker's ``done``
        event carries both echo keys from the same commit and the app-side
        contract does not change shape twice.
        """
        return "off"

    def fused_gguf_dequant_kernel_used(self) -> str:
        """What the last finished job's GGUF dequantization actually did: "off",
        "on", or "on->off" (asked for, but fell back to the eager PyTorch path).

        This is the snapshot taken by :meth:`reset_acceleration_job`, not live
        state, so it is only meaningful after a job has finished.
        """
        return dequant_triton.last_used()

    def generate(  # noqa: PLR0913
        self,
        *,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        num_frames: int,
        frame_rate: float,
        output_path: str,
        images: Sequence[ImageConditioningInput] = (),
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        ic_loras: Sequence[tuple] | None = None,
        ic_reference: tuple[str, float] | None = None,
        ic_attention_strength: float = 1.0,
        ignored: dict[str, Any] | None = None,
    ) -> GenerationResult:
        """Run one two-stage generation and write ``output_path`` as an mp4.

        ``images`` empty -> T2V; one or more entries -> I2V (the official image
        conditioner re-compresses each at the CRF the checkpoint declares and
        encodes it into the stage-1 and stage-2 latents).

        ``ic_loras`` is the job's adapters as ``(path, strength[, audio_strength])``
        -- Style LoRAs and IC-LoRAs alike, since nothing about the attach
        distinguishes them. It is passed on EVERY call, including as ``None``,
        because that is what clears the previous job's adapters (see
        ``Ltx25DiffusionStage.set_loras``).

        ``ic_reference`` is the IC-LoRA's reference video as ``(path, strength)``.
        It requires ``ic_loras``: the reference's spatial downscale factor lives
        in the LoRA's own metadata and nowhere else. The reference is appended to
        STAGE 1 only. ``ic_attention_strength`` (0..1, default 1.0) relaxes how
        strongly the reference tokens drive self-attention; at 1.0 no wrapper is
        applied at all.

        ``ignored`` is the subset of the request this engine drops; it is logged
        here so a job's log names every knob that had no effect, rather than
        leaving the caller to infer it from the absence of a difference.
        """
        validate_geometry(width, height, num_frames)
        _log_ignored(ignored)

        images = list(images)
        encode_fps = int(round(frame_rate))
        if encode_fps < 1:
            raise Ltx25PipelineError(f"frame_rate={frame_rate} rounds to {encode_fps} fps")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        lora_entries = list(ic_loras or [])
        # The downscale factor is resolved BEFORE any model is built: it is a
        # header read, and getting "this LoRA declares no reference factor" as a
        # failure two minutes into a job would be a poor trade.
        reference_factor = resolve_reference_downscale_factor(lora_entries, ic_reference)

        logger.info(
            "generate %dx%d / %d frames @ %.3f fps (encode %d fps) seed=%d images=%d "
            "loras=%d reference=%s -> %s",
            width, height, num_frames, frame_rate, encode_fps, seed, len(images),
            len(lora_entries),
            "no" if ic_reference is None
            else f"{Path(ic_reference[0]).name} strength={ic_reference[1]:.3f} "
                 f"factor={reference_factor} attn={float(ic_attention_strength):.3f}",
            out,
        )

        # set_loras BEFORE begin_job, and unconditionally: begin_job releases the
        # PREVIOUS job's attachment, and a job that asks for no adapter must clear
        # rather than inherit one.
        self.stage.set_loras(lora_entries)
        self.stage.begin_job()
        self.stage.progress = self.progress
        self.prompt_encoder.progress = self.progress
        self.vram.phases.clear()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        # The reference patch receives only images/height/width/video_encoder/
        # dtype/device/color_space, so the tiling config the factor-1 encode needs
        # has to be resolved here and closed over. Same call the pipeline makes
        # internally (distilled.py) -- FULL resolution and the whole frame count,
        # because it is the DECODE's chunking, reused.
        resolved_reference_tiling = None
        if ic_reference is not None:
            resolved_reference_tiling = ensure_tiling_config(
                tiling_config,
                scale_factors=tiling_scale_factors_for_vae(self.pipeline.video_decoder.checkpoint_path),
                video_shape=VideoPixelShape(1, num_frames, height, width, frame_rate),
                vae_checkpoint_path=self.pipeline.video_decoder.checkpoint_path,
                diffvae_optimization=self.pipeline.video_decoder.diffvae_optimization,
                device=self.device,
            )

        started = time.perf_counter()
        with torch.no_grad(), reference_patch(
            ic_reference=ic_reference,
            factor=reference_factor,
            attention_strength=float(ic_attention_strength),
            full_height=height,
            num_frames=num_frames,
            tiling_config=resolved_reference_tiling,
            vram=self.vram,
        ) as reference_receipts:
            video, audio, resolved_frames, resolved_tiling = self.pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                # float on purpose (fact M): every latent grid and RoPE position
                # downstream is built from this value.
                frame_rate=frame_rate,
                images=images,
                num_frames=num_frames,
                tiling_config=tiling_config,
                stage_1_sigmas=DISTILLED_SIGMAS,
                stage_2_sigmas=STAGE_2_DISTILLED_SIGMAS,
            )

            chunks = get_video_chunks_number(resolved_frames, resolved_tiling)
            self.vram.reset()
            encode_started = time.perf_counter()
            encode_video(
                video=self._with_decode_progress(video, chunks),
                fps=encode_fps,
                audio=audio,
                output_path=str(out),
                video_chunks_number=chunks,
            )
            self.vram.record("30_decode_encode", time.perf_counter() - encode_started)
        if reference_receipts:
            logger.info("IC-LoRA reference receipts: %s", reference_receipts)

        seconds = time.perf_counter() - started

        if not out.is_file() or out.stat().st_size <= 0:
            raise Ltx25PipelineError(f"the pipeline produced no/empty output: {out}")

        # The transformer's retained state dict survives on purpose
        # (cache_weights); everything else this job allocated does not. The
        # explicit gc pass is needed because the block-swap closures form
        # reference cycles that plain refcounting cannot reclaim.
        del video, audio
        gc.collect()
        cleanup_memory()

        result = GenerationResult(
            output_path=str(out),
            seed=seed,
            width=width,
            height=height,
            num_frames=resolved_frames,
            frame_rate=frame_rate,
            encode_fps=encode_fps,
            num_images=len(images),
            size_bytes=out.stat().st_size,
            seconds=seconds,
            phases=dict(self.vram.phases),
            video_chunks=chunks,
            tiling=None if resolved_tiling is None else repr(resolved_tiling),
            **_peaks(self.vram, self.device),
        )
        logger.info(
            "GENERATED_OK %.1fs peak_allocated=%sGiB peak_reserved=%sGiB rss_peak=%sGiB -> %s",
            result.seconds, result.peak_allocated_gib, result.peak_reserved_gib,
            result.rss_peak_gib, result.output_path,
        )
        return result

    def _with_decode_progress(self, video: Iterator[torch.Tensor], chunks: int) -> Iterator[torch.Tensor]:
        """Report VAE-decode progress as the encoder pulls chunks off the iterator.

        The decode is lazy -- ``VideoDecoder.__call__`` returns a generator and
        the model is only freed once it is exhausted -- so counting here is
        counting the decode itself, not a re-walk of finished work.
        """
        if self.progress is None:
            return video

        def counted() -> Iterator[torch.Tensor]:
            done = 0
            for chunk in video:
                yield chunk
                done += 1
                assert self.progress is not None
                self.progress(STAGE_DECODE, done - 1, max(1, chunks))

        return counted()

    # -- teardown ------------------------------------------------------------

    def close(self) -> None:
        """Drop the retained weights and free the CUDA cache. Idempotent."""
        self.pipeline = None  # type: ignore[assignment]
        self.stage = None  # type: ignore[assignment]
        self.prompt_encoder = None  # type: ignore[assignment]
        gc.collect()
        try:
            cleanup_memory()
        except Exception:  # pragma: no cover -- teardown is best effort
            pass


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _default_device() -> torch.device:
    if not torch.cuda.is_available():
        raise Ltx25PipelineError(
            "no CUDA device is available. The LTX 2.5 engine has no CPU path: a 22B "
            "transformer at 8 sigma steps would not finish."
        )
    return torch.device("cuda", torch.cuda.current_device())


def _log_ignored(ignored: dict[str, Any] | None) -> None:
    """Name every field this job carried that the engine did not act on."""
    if not ignored:
        return
    for name, value in sorted(ignored.items()):
        reason = IGNORED_FIELDS.get(name, "not supported by the LTX 2.5 v1 contract")
        logger.info("ignoring %s=%r -- %s", name, value, reason)


def _peaks(vram: _Vram, device: torch.device) -> dict[str, float | None]:
    """Highest per-phase peaks of a finished job.

    Per-phase maxima rather than a single global reading: the phases are reset
    around each other precisely so a 14.7 GB transformer build and a 1.5 GB
    aggregate projection can be told apart, and a global peak would report only
    the larger and hide which phase owns it.
    """
    rss_values = [entry.get("rss_gib") or 0.0 for entry in vram.phases.values()]
    rss_now = _rss_bytes()
    if rss_now is not None:
        rss_values.append(rss_now / 2**30)
    out: dict[str, float | None] = {
        "rss_peak_gib": round(max(rss_values), 2) if rss_values else None,
        "peak_allocated_gib": None,
        "peak_reserved_gib": None,
    }
    if device.type == "cuda" and vram.phases:
        out["peak_allocated_gib"] = max(
            entry.get("peak_allocated_gib", 0.0) for entry in vram.phases.values()
        )
        out["peak_reserved_gib"] = max(
            entry.get("peak_reserved_gib", 0.0) for entry in vram.phases.values()
        )
    return out


def image_conditionings(entries: Sequence[dict]) -> list[ImageConditioningInput]:
    """Turn the worker payload's ``images`` list into official conditioning inputs.

    ``crf`` is left unset (None) unless the caller names it: the official
    ``ImageConditioner.resolve_crf`` then fills in the value THIS checkpoint was
    trained against, which is a per-generation property and not something the
    app should be guessing.
    """
    conditionings = []
    for entry in entries:
        path = str(entry["path"])
        if not Path(path).is_file():
            raise FileNotFoundError(f"conditioning image not found: {path}")
        crf = entry.get("crf")
        conditionings.append(
            ImageConditioningInput(
                path=path,
                frame_idx=int(entry.get("frame_idx", 0)),
                strength=float(entry.get("strength", 1.0)),
                crf=None if crf is None else int(crf),
            )
        )
    return conditionings
