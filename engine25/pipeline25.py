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
rate, seed, and zero or more conditioning images (T2V / I2V). What the app may
still send and this engine does not act on -- ``guidance_scale``,
``num_inference_steps`` -- is IGNORED WITH A LOG LINE, never silently: the
distilled 2.5 model runs a fixed 8 + 3 sigma schedule with no classifier-free
guidance, so a step count or a CFG scale has nothing to attach to.
``negative_prompt`` USED TO BE ON THAT LIST AND NO LONGER IS: a CFG-free model
cannot be pushed away from an unconditional prediction, but it can be argued
with inside the single pass it does run, which is what NAG and VSF do (see
:meth:`Ltx25Pipeline.set_nag_job` and ``engine25/neg_prompt25.py``).
``crop_output`` does not appear here at all, by
design: it is an ffmpeg post-process the app already performs on the finished
mp4 (see ``services/engines/ltx/adapter.py``), engine-independent in both
engines.

The acceleration knobs that DO apply are not generation parameters and are not
on ``generate``'s signature: they are per-job state on a resident worker
process, armed by :meth:`Ltx25Pipeline.set_acceleration_job` before the job and
disarmed in its ``finally``. All FOUR are live: the fused Triton GGUF
dequantization kernels (which work here because this engine's transformer and
text encoder both dequantize through 2.3's ``engine.gguf.quant_service``),
asynchronous block-swap prefetching (which the diffusion stage re-arms on every
transformer build -- see ``Ltx25DiffusionStage.set_block_swap_prefetch``),
``keep_resident``, which retains the Gemma 4 text encoder's 7.7 GiB state dict
between jobs instead of re-reading it from the GGUF every time, and
``attention_backend``, which swaps every transformer block's attention kernel
for SageAttention (2.3's ``engine.transformer.sage_attention_service``,
imported unchanged -- the two engines' attention contracts are identical).
``keep_resident`` is the odd one out twice over: it is opt-in rather than on by
default (it costs resident RAM), and its state deliberately OUTLIVES the job
that armed it -- see :func:`_swap_keep_resident`. ``attention_backend`` is the
only one of the four that changes the OUTPUT: sage is a quantized kernel, so a
sage job and an sdpa job at one seed differ in fine detail, which is why it too
defaults to off and why the job echoes back what it really ran on
(:meth:`Ltx25Pipeline.attention_used`).

A FIFTH per-job knob sits beside those four and is a different kind of thing:
:meth:`Ltx25Pipeline.set_nag_job` arms the job's non-CFG negative prompt (NAG or
VSF). It is armed and reset in the same place and by the same discipline, but it
is not an acceleration knob -- it changes what the model computes, on purpose,
which is why it has its own method rather than a fifth argument to
``set_acceleration_job``.

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
# SHARED WITH LTX 2.3, imported rather than reimplemented: 2.5's attention
# contract is 2.3's to the letter (flat (B, S, H*D) q/k/v, head dims 128/64, six
# attention modules per block), so a second copy could only drift. The module's
# own imports are stdlib + torch, so this costs nothing at import time and works
# in a venv with no sageattention wheel -- the wheel is imported lazily, on the
# first job that actually asks for sage.
from engine.transformer.sage_attention_service import SageAttentionService, SageState
from engine25 import assets_export
# The non-CFG negative-prompt patch (NAG / VSF). ``NagState``/``NagParams``/
# ``VsfParams`` are 2.3's own classes, re-exported by engine25's module -- see
# its docstring on why the algebra is shared and only the replacement
# ``Attention.forward`` is 2.5's own.
from engine25.neg_prompt25 import NagParams, NagState, NegPromptService, VsfParams
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
    as_path_list,
    cleanup_memory,
    encode_video,
    ensure_tiling_config,
    get_video_chunks_number,
    is_diffusion_video_vae,
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
            # Denoise is over and the official ``gpu_model`` contract has already
            # disposed the transformer, so give the prefetch arena back NOW rather
            # than at the end of the job. The window is not cyclic: it drains to a
            # single block, and that last block's arena -- 207.9 MB on this
            # checkpoint -- is otherwise held by the engine's own ``_state`` while
            # the spatial upsampler and the VAE decode run. ``dispose()`` cannot
            # reach it; only this call can. MEASURED at 320x192x25: 229.1 MB still
            # allocated at the end of denoise without this line, 21.2 MB with it.
            #
            # Safe against the disposed model, but not silently so, and the
            # difference is worth knowing: ``PrefetchEngine._release`` re-points
            # module slots at the CPU masters, and its first parameter slot raises
            # ``set_data ... incompatible tensor type`` because ``dispose()`` left
            # that parameter on ``device="meta"``. ``teardown()`` catches it, logs
            # "release of block N failed", and clears ``_state`` anyway -- which is
            # what actually frees the arena, so the common case is fully covered.
            # The ONE case it does not cover: IC-LoRA / Style-LoRA A/B buffers are
            # ``persistent=False``, so ``dispose()`` does not meta them either, and
            # with the restore loop aborted they keep viewing the arena until the
            # next build's ``detach_ic_loras``. A LoRA job therefore still carries
            # ~208 MB into the upsampler (measured: 236.1 MB with and without this
            # line). Fixing that means making ``_release`` restore slot by slot,
            # which is a change to the shared 2.3 module and is deliberately not
            # made here.
            self.teardown_block_swap_prefetch()
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
        neg_state_provider: Callable[[], NagState] | None = None,
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
        #: The job's NAG/VSF state, read through a CLOSURE on every call rather
        #: than captured by value -- one long-lived encoder, many jobs, and the
        #: same pattern the sage service uses on this engine. ``None`` is a real
        #: and supported state (a direct library user, the selftest): it means
        #: "never encode a negative prompt", which is what :meth:`__call__`
        #: falls back to.
        self._neg_state_provider = neg_state_provider
        #: The tokenizer the last :meth:`_build_text_encoder` produced, kept
        #: because the VSF slice needs it AFTER the encoder itself has been
        #: freed -- see :meth:`_build_text_encoder`.
        self._tokenizer: Any = None

    # -- sub-phase timers ----------------------------------------------------
    #
    # ``10_prompt_encode`` is one number for three very different pieces of
    # work: building the 9.2 GB Gemma 4 text encoder, running the encode, and
    # building the embeddings processor. These two overrides split the two
    # BUILDS out of it so the split is measured instead of assumed -- the
    # encode's own cost is then the parent minus these two.
    #
    # Both official methods take no arguments (asserted by
    # ``ltxcore_compat.verify``), so wrapping them is a plain ``super()`` call
    # with a clock around it and changes nothing about what gets built. They
    # deliberately do NOT call ``self.vram.reset()``: a reset here would zero
    # CUDA's peak counters inside the parent's window and make
    # ``10_prompt_encode`` understate the peak it exists to report. The
    # sub-phases therefore read the same running peak the parent will, which is
    # what "the peak so far, at this point in the window" means.

    def _build_text_encoder(self) -> Any:
        """Official build, timed as ``10a_te_build`` (inside ``10_prompt_encode``).

        ALSO WHERE THE TOKENIZER IS STASHED, and this is the only place it can
        be: the official ``__call__`` builds the encoder inside a ``with`` block
        and frees it before returning, so by the time the VSF slice needs the
        prompt's real token count the encoder object is gone. What is kept is
        the tokenizer alone -- vocabulary, no weights -- and it is REPLACED on
        every build, so it can never go stale: the encoder is rebuilt once per
        job, and the tokenizer that comes with it is the one that produced this
        job's embeddings.
        """
        started = time.perf_counter()
        try:
            encoder = super()._build_text_encoder()
            self._tokenizer = getattr(encoder, "tokenizer", None)
            return encoder
        finally:
            if self.vram is not None:
                self.vram.record("10a_te_build", time.perf_counter() - started)

    def _build_embeddings_processor(self) -> Any:
        """Official build, timed as ``10b_ep_build`` (inside ``10_prompt_encode``)."""
        started = time.perf_counter()
        try:
            return super()._build_embeddings_processor()
        finally:
            if self.vram is not None:
                self.vram.record("10b_ep_build", time.perf_counter() - started)

    def __call__(self, prompts: list[str], **kwargs: Any) -> Any:
        """The official encode, plus this job's negative prompt when it has one.

        THE BRANCH IS THE POINT. A job that did not ask for a negative prompt
        takes the ``super().__call__(prompts, **kwargs)`` line below -- the same
        call, with the same arguments, that stood here before this feature
        existed -- so its encode is not merely equivalent to what it was, it is
        the same code path. That is the structural half of "a non-NAG job is
        unchanged"; routing every job through the negative-prompt helper and
        having it no-op would not have been.
        """
        if self.progress is not None:
            self.progress(STAGE_ENCODE, 0, 1)
        if self.vram is not None:
            self.vram.reset()
        started = time.perf_counter()
        try:
            state = None if self._neg_state_provider is None else self._neg_state_provider()
            if state is None or not state.requested:
                return super().__call__(prompts, **kwargs)
            return self._call_with_negative(state, prompts, **kwargs)
        finally:
            if self.vram is not None:
                self.vram.record("10_prompt_encode", time.perf_counter() - started)
            if self.progress is not None:
                self.progress(STAGE_ENCODE, 1, 1)

    def _call_with_negative(self, state: NagState, prompts: list[str], **kwargs: Any) -> Any:
        """Encode the negative prompt in the SAME Gemma pass, then slice it off.

        ONE EXTRA LIST ENTRY, AT THE END. The official encoder tokenizes every
        prompt to the same fixed 1024 length and stacks them into a single ``[N,
        1024]`` batch, so appending costs one more row through Gemma and one
        more row through the connectors -- not a second model load, which is the
        expensive part. Appending at the END rather than the front matters for
        one reason: ``enhance_first_prompt`` rewrites ``prompts[0]``, and the
        negative prompt must never be the one that gets enhanced. (2.5 never
        enables it, but the ordering costs nothing to get right.)

        THE TAIL IS SLICED OFF BEFORE RETURNING, and a caller that somehow got
        the unsliced list would not limp along: ``DistilledPipeline.__call__``
        unpacks the result as ``(ctx_p,) = ...`` and ``chain25`` zips it against
        its prompt list with ``strict=True``. Both fail immediately and loudly.

        The reshape below is ``TransformerArgsPreprocessor._prepare_context``'s,
        and deliberately only its tail: that method is ``caption_projection``
        (None on this engine's 22B checkpoints -- the projection lives in the
        text encoder) followed by ``context.view(batch, -1, x.shape[-1])``. So
        the negative context reaches attn2 at exactly the representation stage
        the positive one does, and ``NegPromptService.install`` refuses the job
        outright if a future checkpoint ever brings the projection back.
        """
        params = state.params
        assert params is not None  # implied by state.requested at the call site
        outputs = list(super().__call__([*prompts, params.negative_prompt], **kwargs))
        if len(outputs) != len(prompts) + 1:
            raise Ltx25PipelineError(
                f"the prompt encoder returned {len(outputs)} outputs for "
                f"{len(prompts) + 1} prompts, so the negative prompt cannot be "
                "separated from the positive ones."
            )
        negative = outputs.pop()

        video_ctx = negative.video_encoding
        audio_ctx = negative.audio_encoding
        if video_ctx is None or audio_ctx is None:
            raise Ltx25PipelineError(
                "the negative prompt encoded to a missing video or audio "
                "context; this engine's AV transformer needs both."
            )
        video_ctx = video_ctx.view(video_ctx.shape[0], -1, video_ctx.shape[-1])
        audio_ctx = audio_ctx.view(audio_ctx.shape[0], -1, audio_ctx.shape[-1])

        if isinstance(params, VsfParams):
            n_real = self._real_token_count(params.negative_prompt, video_ctx, audio_ctx)
            video_ctx = video_ctx[:, :n_real, :]
            audio_ctx = audio_ctx[:, :n_real, :]

        state.set_contexts(video_ctx, audio_ctx)
        logger.info(
            "negative prompt encoded alongside %d positive prompt(s): "
            "video=%s audio=%s (method=%s)",
            len(prompts),
            tuple(int(d) for d in video_ctx.shape),
            tuple(int(d) for d in audio_ctx.shape),
            "vsf" if isinstance(params, VsfParams) else "nag",
        )
        return outputs

    def _real_token_count(
        self, prompt: str, video_ctx: torch.Tensor, audio_ctx: torch.Tensor
    ) -> int:
        """How many of the encoded tokens are the prompt's OWN. VSF only.

        WHY THE FRONT OF THE SEQUENCE IS THE RIGHT SLICE ON THIS ENGINE:
        ``EmbeddingsProcessor.create_embeddings`` runs every feature tensor
        through ``_compute_right_pad_order`` before the connectors see it -- a
        STABLE descending sort of the binary mask, i.e. "valid tokens first,
        pads after, relative order preserved". The real tokens are therefore at
        the FRONT by construction. 2.3 gets the same guarantee from a re-pack
        INSIDE the connector; on 2.5 it happens one layer up, which is why this
        docstring cites the 2.5 source rather than restating 2.3's.

        WHY VSF NEEDS IT AND NAG DOES NOT: VSF concatenates the negative
        keys/values into ONE shared softmax and NEGATES the negative values. The
        tail of the encoded sequence is not padding -- it is the connector's
        learned register embeddings, real trained data -- so sign-flipping it
        would inject a large, prompt-independent repulsion. NAG combines two
        SEPARATE attention outputs and is unharmed, which is why it keeps the
        full context.

        Fail-loud rather than guess: an absent tokenizer, or a count that does
        not fit the encoded length, means the encode and the count came from
        different places.
        """
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise Ltx25PipelineError(
                "VSF needs the negative prompt's real token count, but no "
                "tokenizer was captured from the text-encoder build. Slicing is "
                "a correctness requirement for VSF (see this method's "
                "docstring), so this fails rather than silently sign-flipping "
                "learned register embeddings."
            )
        pairs = tokenizer.tokenize_with_weights(prompt)["gemma"]
        n_real = int(sum(int(weight) for _token, weight in pairs))
        seq_len = int(min(video_ctx.shape[1], audio_ctx.shape[1]))
        if n_real <= 0 or n_real > seq_len:
            raise Ltx25PipelineError(
                f"the tokenizer reports {n_real} real tokens for a negative "
                f"context of length {seq_len} -- expected 0 < N <= seq_len. "
                "Refusing to guess."
            )
        return n_real


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
    # ``negative_prompt`` / ``neg_method`` / ``vsf_scale`` LEFT THIS LIST with
    # the NAG/VSF commit. They used to sit here as "no negative-prompt
    # mechanism in v1", which was true until this engine got one: the request's
    # negative prompt is now encoded alongside the positive one and its
    # cross-attention contribution is real (see :meth:`Ltx25Pipeline.set_nag_job`
    # and ``engine25/neg_prompt25.py``). Leaving them here would make the worker
    # log "ignored" for the three fields the feature is made of.
    "guidance_scale": "2.5 distilled runs without classifier-free guidance",
    "num_steps": "the distilled schedule is fixed at 8 + 3 sigmas",
    "num_inference_steps": "the distilled schedule is fixed at 8 + 3 sigmas",
    # ``attention_backend`` LEFT WITH THE SAGE COMMIT. It used to sit here as
    # "v1 is SDPA-only", which was true until this engine got a
    # ``SageAttentionService``; it is now ACTED ON (see
    # :meth:`Ltx25Pipeline.set_acceleration_job`) and echoed back on ``done`` as
    # ``attention_used``, so leaving it on this list would make the worker log
    # "ignored" for the one field whose whole point is that it is obeyed.
    "vae_mode": "v1 uses the Conv VAE only",
}


# ---------------------------------------------------------------------------
# keep_resident: the text encoder's state dict between jobs
# ---------------------------------------------------------------------------
#
# ``keep_resident`` used to sit in IGNORED_FIELDS above ("2.5 keeps its weights
# in the registry instead"), which was true of the TRANSFORMER and only of the
# transformer. The Gemma 4 text encoder's own registry is built with
# ``cache_weights=False`` (``gguf_gemma4.build_text_encoder_builder``), so its
# 7.7 GiB state dict is re-read from the GGUF on every single job. This function
# is the switch for that, and the reason the field is now acted on.
#
# NAME SHARED WITH 2.3, IMPLEMENTATION NOT. 2.3's ``keep_resident`` retains the
# skeletons of every sub-model behind a two-argument call that returns a tuple
# and carries three internal degradation guards (LoRA in-place mutation, and two
# more). This is one registry holding one state dict, with no degradation path
# at all -- 2.5's text encoder takes no LoRA and is loaded with ``assign=True``,
# so nothing ever mutates the retained tensors in place. The CONTRACT is 2.3's
# verbatim (absent key means off, no end-of-job reset, an echo on ``done``); the
# code behind it is unrelated.


def _swap_keep_resident(registry: Any, builder: Any, enabled: bool) -> int | None:
    """Turn the text encoder's weight cache on or off. Returns bytes released.

    **Never raises.** It is called from :meth:`Ltx25Pipeline.set_acceleration_job`,
    which sits OUTSIDE the worker's try/finally so that nothing between the arm
    and the reset can throw -- and unlike the other two knobs this one is not a
    pure assignment: the OFF path drops a 7.7 GiB state dict, which is I/O-shaped
    work (frees, and a collector pass). An exception here would skip the reset
    and leak the job's request into the next one on a resident worker, so it is
    logged and swallowed instead. A failed swap costs speed or RAM, never
    correctness: both settings produce the same weights and the same video.

    ``registry.add`` is the ONLY method that reads ``_cache_weights``, so turning
    the flag ON takes effect at the next build with nothing else to do. Turning
    it OFF does NOT: ``get`` never looks at the flag, so an already-cached state
    dict would keep being served (and keep being held) forever. The OFF path
    therefore has to ``pop`` the entry out by hand, with the same key ``add``
    used -- which is why the builder is needed here and not just the registry.

    Returns the number of bytes released (0 when nothing was cached), or None if
    the swap did not run.
    """
    if registry is None or builder is None:
        # Before the pipeline finished building, or after :meth:`close`.
        return None
    try:
        registry._cache_weights = enabled
        if enabled:
            logger.info(
                "keep-resident ON: the text encoder's state dict will be retained between jobs "
                "(~7.7 GiB of resident RAM; the next job skips its rebuild)"
            )
            return 0
        state_dict = registry.pop(as_path_list(builder.model_path), builder.model_sd_ops)
        # ``del`` is what frees it: this is the last reference to the state dict
        # once the registry has let go, so the tensors are gone at this line.
        # ``gc.collect()`` is insurance for reference CYCLES only (a released
        # StateDict participates in none today) -- it is not the mechanism.
        # Both run BEFORE the log line so that the line is a statement about
        # memory already given back, not about memory that is about to be.
        size = state_dict.size if state_dict is not None else 0
        del state_dict
        gc.collect()
        logger.info(
            "keep-resident OFF: released %.2f GiB of retained text-encoder weights "
            "(the next job rebuilds them from the GGUF)",
            size / 2**30,
        )
        return size
    except Exception:  # pragma: no cover -- never-raise discipline
        logger.exception("keep-resident swap to %s failed; continuing", "on" if enabled else "off")
        return None


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
        #   be re-applied to the block-swap service on every transformer build,
        #   which is the diffusion stage's job -- so this class holds the request
        #   and the finished job's verdict, and the stage holds the tally.
        # * keep_resident is a THIRD shape again: its state is a flag on a
        #   registry object that lives inside the text-encoder builder, so this
        #   class holds the two handles it needs to reach it (set just after the
        #   builder is constructed, below), the CURRENT setting -- which, alone
        #   among the three knobs, SURVIVES the end of the job because the
        #   retained weights are the feature -- and the requested/used pair the
        #   echo is folded from.
        self._block_swap_prefetch_requested = False
        self._block_swap_prefetch_used = "off"
        self._te_builder: Any = None
        self._te_registry: Any = None
        self._keep_resident_enabled = False
        self._keep_resident_requested = False
        self._keep_resident_used = "off"
        # * SageAttention is a FOURTH shape: the request, the per-call kernel
        #   latch and the finished job's echo all live in ONE object shared with
        #   2.3 (``SageState``), which the diffusion stage's service reads
        #   through a closure on every transformer build. This class owns the
        #   object and its arm/reset; the stage owns the wrapping. Held here
        #   rather than on the stage because the stage is rebuilt-into many
        #   times per job and the kernel latch has to outlive every one of those
        #   builds -- a chain that lost its latch between segments would retry a
        #   kernel already known to be broken, once per segment.
        self._sage = SageState()
        # * the non-CFG negative prompt (NAG / VSF) is a FIFTH shape, and the
        #   only per-job knob that changes what the model COMPUTES rather than
        #   how fast it computes it. Its state is one ``NagState`` -- 2.3's own
        #   class -- read by TWO closures: the prompt encoder's, which fills in
        #   the encoded negative contexts, and the diffusion stage's service,
        #   which patches the 96 text cross-attention forwards on every build.
        #   Held here, like the sage state, because the stage is rebuilt-into
        #   many times per job and the encoded contexts have to outlive every
        #   one of those builds.
        self._nag = NagState()

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
        # Attached AFTER construction rather than passed to ``from_gguf``: the
        # service needs a handle on the per-job ``SageState`` this class owns,
        # and the state has to be reachable through a CLOSURE (not captured by
        # value) so one long-lived service always sees the CURRENT job -- the
        # same ``lambda: self._sage`` the 2.3 pipeline uses. Post-construction
        # assignment is safe on this stage: ``with_*`` clones it with
        # ``copy.copy`` (the attribute rides along), chain25 drives this very
        # instance, and a stage built any other way simply has ``None`` there.
        stage._sage_service = SageAttentionService(lambda: self._sage)
        # Same post-construction attachment, same closure, same reason: one
        # long-lived service that always sees the CURRENT job's request. The
        # stage strips and re-installs on every transformer build (see
        # ``Ltx25DiffusionStage._ensure_neg_installed``), which is what a reused
        # model shell requires.
        stage._neg_service = NegPromptService(lambda: self._nag)

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
        # The two handles ``keep_resident`` needs, taken HERE rather than fished
        # out of ``self.prompt_encoder`` later: the builder's private registry is
        # created inside the factory above (``cache_weights=False``, which is the
        # OFF this class starts in), the builder is its only user, and the pair
        # is what ``_swap_keep_resident`` keys the state dict with. Both are
        # dropped in :meth:`close` -- see the note there.
        self._te_builder = text_encoder_builder
        self._te_registry = text_encoder_builder.registry
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
            # The other half of the negative-prompt wiring: the encoder fills the
            # state the stage's service reads. Passed as a closure for the same
            # reason the service gets one -- this object outlives every job.
            neg_state_provider=lambda: self._nag,
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
        # 台帳 §3-131: which VAE decoder this checkpoint builds -- "diff" (the
        # DiT-based diffusion decoder) or "conv" (the plain convolutional one).
        # ``VideoDecoder.__init__`` calls the same function on the same path to
        # pick its own internals, but does not keep the answer anywhere reachable
        # afterwards. It is a load-time fact of the checkpoint, not a per-job
        # decision, so it is read once here and stored as an attribute -- not
        # recomputed as a method on every job.
        self.video_vae_kind = "diff" if is_diffusion_video_vae(self.files.video_vae) else "conv"
        self.build_report["video_vae_kind"] = self.video_vae_kind
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
        keep_resident: bool,
        attention_backend: str,
    ) -> None:
        """Arm this job's four acceleration knobs.

        Called by the worker BEFORE the try block that runs the job, and paired
        with :meth:`reset_acceleration_job` in that block's ``finally``. The
        first two are per-JOB state on a resident worker process, so an arm
        without a matching reset would leak one job's request into the next one.

        ``keep_resident`` is the ASYMMETRIC one and deliberately so: the reset
        does not turn it off, because the retained 7.7 GiB of text-encoder weights
        ARE the feature -- they have to outlive the job that asked for them or
        there is nothing for the next job to hit. What the reset does is freeze
        the echo. The setting therefore changes only here, and only when the new
        request differs from what is already in force (2.3's rule verbatim: a
        job that does not send the key is asking for the weights to be RELEASED,
        which is why a missing key is a real instruction and not a no-op).

        The ordering constraint is on the fused half and it is not negotiable:
        the flag has to be armed **before the transformer is built**, because the
        build is where the GGUF weights are dequantized. Arming it after
        ``generate`` had started would leave the whole first build on the eager
        path and only catch a later rebuild. Same argument in 2.3
        (``engine/pipeline/fast_video_pipeline.py``'s entry points), for the same
        reason; ``set_job`` itself only assigns module globals -- no import, no
        CUDA -- so THAT half cannot fail, and the call ORDER is what makes the
        arrangement safe rather than any exception handling.

        The prefetch half has the SAME ordering constraint, for the same reason:
        the block-swap service reads its flag inside ``install()``, and install
        happens during the build. Handing it to the stage is likewise nothing but
        assignments -- the resources are created later, by the build -- so it
        cannot fail either.

        **The keep-resident half CAN**, which is the one place this method
        departs from "it is all assignments". Switching it off pops a 7.7 GiB
        state dict out of the registry and drops it: real work, not an
        assignment. :func:`_swap_keep_resident` is therefore never-raise (it
        logs and swallows), which restores the property the arm/reset pairing
        outside this class depends on -- **this method still cannot fail** --
        without pretending that what it does is trivial. Its ordering
        constraint runs the same direction as the other two, for a smaller
        reason: the swap happens before the text encoder is built, so a job
        that asked for the release runs on the freed footprint instead of
        paying for it only at the end.

        ``attention_backend`` ("sdpa" / "sage") has NO DEFAULT on purpose: every
        caller states its choice, so a new entry point cannot silently inherit
        one. Its ordering constraint is the strictest of the four -- the wrappers
        are installed during the transformer build, and ``install()`` reads
        ``state.requested`` at that moment, so an arm after the build would
        produce a job that asked for sage and ran entirely on SDPA while
        reporting "sage". Like the two flag knobs it cannot fail:
        ``SageState.set_backend`` degrades an unrecognised value to "sdpa"
        rather than raising, because the fail-loud gate for unknown values
        belongs at the protocol edge (``engine25.worker._resolve_attention``),
        where refusing the job is still possible; raising HERE would skip the
        matching reset and leak the request into the next job.
        """
        # FIRST, and before anything that could conceivably fail: everything
        # below this line is an assignment or a never-raise call, and the sage
        # request is the one whose arm/reset pairing spans the whole job.
        self._sage.set_backend(attention_backend)
        self._block_swap_prefetch_requested = bool(block_swap_prefetch)
        stage = getattr(self, "stage", None)
        if stage is not None:
            stage.set_block_swap_prefetch(self._block_swap_prefetch_requested)
        dequant_triton.set_job(bool(fused_gguf_dequant_kernel))

        keep_resident = bool(keep_resident)
        self._keep_resident_requested = keep_resident
        # The no-op guard is what makes this cheap to call on every job: the
        # common case is a run of identical requests, and re-arming the setting
        # that is already in force must NOT pop (and so destroy) the cache the
        # last job just filled. Only a CHANGE touches the registry.
        if keep_resident != self._keep_resident_enabled:
            _swap_keep_resident(self._te_registry, self._te_builder, keep_resident)
            self._keep_resident_enabled = keep_resident

    def set_nag_job(self, nag: "NagParams | VsfParams | None") -> None:
        """Arm (or clear, with ``None``) this job's non-CFG negative prompt.

        ONE ASSIGNMENT, and deliberately a SEPARATE method from
        :meth:`set_acceleration_job` rather than a sixth argument to it. That
        method's whole contract is "these knobs do not change the output"
        (``attention_backend`` is the acknowledged exception and says so); a
        negative prompt changes it on purpose, and folding the two together
        would make one docstring have to say both things.

        CALLED ON EVERY JOB, ``None`` INCLUDED. That is what gives stale-clear
        semantics on a resident worker: ``NagState.set_params`` drops any
        previously encoded contexts along with the params, so a NAG job followed
        by a plain one cannot leak the prior negative prompt -- and a caller
        that armed params but never encoded them hits ``install``'s
        "requested but not ready" ``RuntimeError`` instead of silently reusing a
        stale encoding.

        THE ORDERING CONSTRAINT is the strictest in this class, and it is the
        reason this sits OUTSIDE the worker's try block like the acceleration
        arm does: the negative prompt has to be encoded during the prompt
        encode, which happens before the first transformer build, and the
        service reads ``state.requested`` at build time. Arming after
        ``generate`` had started would produce a job that asked for a negative
        prompt, ran without one, and said nothing.

        **This method cannot fail** -- it is a single attribute write through
        ``NagState.set_params`` -- which is what keeps the arm/reset pairing
        outside this class safe. The fail-loud gate for an unknown METHOD name
        lives at the protocol edge (``engine25.worker._resolve_nag``), where
        refusing the job is still possible.
        """
        self._nag.set_params(nag)

    def reset_nag_job(self) -> None:
        """End-of-job counterpart: drop the request and the encoded contexts.

        **Never raises** (three attribute writes inside ``NagState.reset``), for
        the same reason :meth:`reset_acceleration_job` must not: it runs in the
        worker's ``finally``, where an exception would replace the job's real
        error with this one.

        SYMMETRIC, unlike ``keep_resident``: nothing here is worth keeping. The
        encoded contexts are two tensors sized to ONE job's negative prompt, and
        holding them past the job would both leak VRAM on a resident worker and
        risk the next job's build finding a populated state it never asked for.
        """
        self._nag.reset()

    def reset_acceleration_job(self) -> None:
        """End-of-job counterpart: freeze the verdicts and disarm.

        **This method never raises.** It runs in the worker's ``finally``, so an
        exception here would replace the job's real error with this one -- and on
        a failed job it is precisely the paths that already went wrong that this
        has to clean up. The two halves therefore get their own try/except: a
        failure to tear down prefetch must not leave the fused kernels armed for
        the next job, and vice versa.

        ``self.stage is None`` (i.e. after :meth:`close`) is absorbed by the same
        handlers rather than by a guard of its own: a reset arriving after the
        pipeline was closed is a shutdown race, not a bug worth failing on.

        The keep-resident third is DELIBERATELY ASYMMETRIC and needs no handler
        of its own, because it undoes nothing: the two plain lines below freeze
        the echo and clear the request, and the retained weights stay exactly
        where they are. Releasing them here would destroy the feature -- the
        point of keep-resident is that the NEXT job finds them still there --
        and the release instead happens at the next :meth:`set_acceleration_job`
        that asks for it. Nothing here can raise, so the never-raise contract
        above is untouched.

        A job that failed BEFORE the text encoder was ever built still echoes
        "on" if it asked for it (the request is what the echo reports, exactly
        as in 2.3) -- but a failed job emits no ``done`` at all, so that echo
        never leaves the process.

        The sage third is the one that has to run FIRST, before anything that
        could raise or return early: ``SageState.reset`` snapshots
        ``attention_used`` and only then clears the request, so it is both the
        leak guard for the next job and the ONLY record of what the finished job
        ran on. It cannot raise (four attribute writes), which is why it needs
        no handler of its own.
        """
        self._sage.reset()
        self._keep_resident_used = "on" if self._keep_resident_requested else "off"
        self._keep_resident_requested = False
        try:
            # Verdict FIRST, teardown second: the tally the verdict reads is
            # per-job state on the stage, and disarming clears it.
            stage = getattr(self, "stage", None)
            if stage is None:
                # After :meth:`close`, or before the pipeline finished building.
                # A job that asked for prefetching and never reached a build got
                # a degradation, not a clean "off" -- 2.3's rule verbatim
                # (``engine/pipeline/fast_video_pipeline.py``).
                self._block_swap_prefetch_used = (
                    "on->off" if self._block_swap_prefetch_requested else "off"
                )
            else:
                self._block_swap_prefetch_used = stage.block_swap_prefetch_verdict()
                # The job owns the arenas, the CPU masters and the events;
                # releasing them here rather than at the next build is what keeps
                # the finished transformer from being pinned alive between jobs.
                stage.teardown_block_swap_prefetch()
                stage.set_block_swap_prefetch(False)
        except Exception:  # pragma: no cover -- never-raise discipline
            logger.exception("block-swap prefetch reset failed; continuing")
        finally:
            self._block_swap_prefetch_requested = False
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

        Like :meth:`fused_gguf_dequant_kernel_used`, this is the snapshot taken
        by :meth:`reset_acceleration_job` -- folded there from the diffusion
        stage's per-build tally, because a job builds the transformer many times
        and "on" has to mean every one of those builds ran accelerated.
        """
        return self._block_swap_prefetch_used

    def fused_gguf_dequant_kernel_used(self) -> str:
        """What the last finished job's GGUF dequantization actually did: "off",
        "on", or "on->off" (asked for, but fell back to the eager PyTorch path).

        This is the snapshot taken by :meth:`reset_acceleration_job`, not live
        state, so it is only meaningful after a job has finished.
        """
        return dequant_triton.last_used()

    def keep_resident_used(self) -> str:
        """What the last finished job's text-encoder residency actually did:
        "off" or "on".

        The echo CONTRACT is 2.3's three-valued one ("off" / "on" / "on->off"),
        and the app relays whatever arrives; this engine simply has no third
        value to emit. 2.3's "on->off" is its automatic degradation -- a LoRA
        that would mutate retained weights in place, and two more guards -- and
        none of those mechanisms exist here: 2.5's text encoder takes no LoRA
        and is loaded with ``assign=True``, so there is nothing that could force
        a retained state dict to be dropped mid-job. The value is the REQUEST,
        frozen at :meth:`reset_acceleration_job`, because with no degradation
        path the request is also the outcome.
        """
        return self._keep_resident_used

    def attention_used(self) -> str:
        """What the last finished job's attention actually ran on: "sdpa",
        "sage", or "sage->sdpa" (sage was requested, but a kernel call raised
        and latched the rest of the job onto SDPA).

        Read by the worker AFTER the job returns, which is why it reports the
        snapshot ``SageState.reset()`` took rather than the live state --
        :meth:`reset_acceleration_job` has already cleared that. Same name, same
        three values and the same read-after-reset discipline as 2.3's
        ``LTXFastVideoPipeline.attention_used``, so the app relays one field from
        one code path per engine.

        Unlike :meth:`keep_resident_used` above, this engine really can emit all
        three values: the degradation here is the sage kernel raising mid-job,
        which is a property of the kernel and the tensors rather than of either
        engine's plumbing.
        """
        return self._sage.last_attention_used

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

        # The job's RESTING footprint: host RSS after the collector has run and
        # before anything of the next job is built. It is a marker, not an
        # interval -- nothing is being timed, so the seconds are 0.0 and the
        # entry exists purely for its ``rss_gib``. Recorded here rather than
        # after ``GenerationResult`` so it lands in ``phases`` and travels on
        # the ``done`` event with the rest.
        self.vram.record("40_job_end", 0.0)

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
        # The keep-resident handles have to go too, or a close() taken while the
        # feature was ON would leave 7.7 GiB of text-encoder weights alive for as
        # long as this object is: the three lines above drop every path to the
        # builder EXCEPT these, and the registry holds the state dict directly.
        # Dropping the handles is enough -- no pop is needed and none is wanted,
        # because the collector below frees the registry itself. Setting the flag
        # back to False keeps the state honest if the object is somehow reused.
        self._te_builder = None
        self._te_registry = None
        self._keep_resident_enabled = False
        # The negative prompt's encoded contexts are two live tensors on the
        # GPU. Nothing above reaches them -- the state is owned by this object,
        # not by the stage or the encoder -- so the collector below would keep
        # them alive for as long as this object is. Never raises, so it is safe
        # in a teardown that is otherwise best-effort.
        self.reset_nag_job()
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
