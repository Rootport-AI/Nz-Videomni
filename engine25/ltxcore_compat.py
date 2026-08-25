"""The ONE import site for official ltx_core / ltx_pipelines symbols (§3-98 Phase 2a).

Why this module exists
----------------------
The LTX 2.5 engine has to reach past the official packages' public surface: a
16GB box cannot use ``DiffusionStage.from_checkpoint`` as shipped (it builds
straight onto the GPU from a safetensors file), so engine25 subclasses the
stage, supplies its own builder, and reuses two module-private helpers from
``single_gpu_model_builder``. Those private reaches are a real upgrade hazard --
a patch release can rename ``_load_model_weights`` and every call site breaks at
a different, later, harder-to-read place.

So the rule for engine25 is: **nothing under engine25/ imports ``ltx_core`` or
``ltx_pipelines`` directly. Everything comes through this module, and every
worker start calls :func:`verify`.** There is deliberately no "we import N
symbols" claim anywhere -- such a count rots the moment a phase adds one. The
contract is the two halves above: one entry point, and one assertion pass over
it that runs before any model is built, so a version drift is reported as
"engine25/ltxcore_compat.py: <symbol> changed" at load time instead of as a
mystery TypeError two minutes into a generation.

What :func:`verify` checks
--------------------------
1. Every re-exported name is bound (an ``ImportError`` at module import already
   covers absence, but a shadowed/None binding would not raise there).
2. The signature *points engine25 actually depends on* -- not full signatures,
   which would fail on cosmetic annotation churn:
     * ``DiffusionStage._build_transformer`` is ``(self, *, device=None, **kwargs)``
       -- keyword-only ``device`` plus a ``**kwargs`` sink, because the stage
       forwards ``video_tools=`` into it and our override must accept that.
     * ``SingleGPUModelBuilder.build`` takes ``device`` / ``dtype``.
     * ``_load_model_weights`` still takes ``model_sd_ops`` / ``fuse_rule`` /
       ``lora_load_device`` (engine25 calls it by keyword).
     * ``ModelRegistry`` still takes keyword-only ``cache_weights`` /
       ``cache_models`` (the H-revision default ``cache_weights=True`` rides on it).
     * ``QuantizationPolicy`` still has exactly the four fields the policy is
       built from.
     * ``SDOps`` still has ``allowed_keys`` + ``with_additional_allowed_keys``
       (the connector drop depends on it) and ``ModuleOps`` its 3 fields.
     * ``create_meta_model`` still takes ``(configurator, metadata, module_ops)``.
     * ``DistilledPipeline.__init__`` still has a REQUIRED positional ``loras``,
       still assigns ``self.stage`` / ``self.prompt_encoder`` /
       ``self.use_ancestral_sampler`` (the three engine25 substitutes or asserts
       after construction), and ``__call__`` still takes the generation
       parameters the v1 contract maps onto.
     * ``PromptEncoder.__init__`` still takes ``text_encoder_builder`` -- the
       public injection point that lets GGUF weights sit behind the official
       prompt encoder without forking it.
     * ``ModelPaths.from_split`` still takes ``text_encoder_path``.
     * ``DISTILLED_SIGMAS`` / ``STAGE_2_DISTILLED_SIGMAS`` are still 8 + 3 steps,
       and ``encode_video`` / ``get_video_chunks_number`` /
       ``ImageConditioningInput`` still have the shape the mp4 write depends on.
     * ``Disposable`` still exposes ``dispose``.
3. For §3-102 (Chained), a handful of BODIES as well as signatures. The chain
   does not run ``DistilledPipeline.__call__``; it drives the same blocks itself
   with a frozen carry band in front of every segment and tile, so three facts
   that the pipeline would otherwise have guaranteed are asserted from source:
     * ``euler_ancestral_denoising_loop`` still gates noise on
       ``draw_noise=stepper.eta > 0``, and the driver still re-pins the
       conditioned tokens (``post_process_latent``) on exactly that branch --
       at ``eta == 0`` an ancestral chain's frozen band would drift with nothing
       raising, which is why chain25 also asserts ``stepper.eta > 0``.
     * ``create_noised_state`` still runs initial state -> conditionings ->
       noiser IN THAT ORDER, and ``create_initial_state`` still clones the
       initial latent into ``clean_latent``. That order is the whole reason
       2.5's ``VideoConditionByMask`` reproduces 2.3's hand-edited mask.
     * ``VideoConditionByMask.apply_to`` still computes
       ``clean*inv + tokens*m`` / ``denoise_mask*inv + (1-strength)*m`` --
       chain25's ``AudioHeadBandMask`` is a line-for-line audio twin of it.
   Plus the surface chain25 drives directly: both denoising loops' keyword
   names, ``ModalitySpec``'s fields, the ``replacing``/``guiding`` image pair
   with ``resolve_crf``, ``ensure_tiling_config``'s three keyword-only
   arguments, ``LatentState.keyframes_mask`` + the helper that marks the first
   latent frame, ``LatentUpsampler``'s block count (from which halo=18 is
   derived, not remembered), and 25 audio latents per second.
4. For §3-102's second stage (V2V + A2V), the MATERIAL-INGEST surface: the audio
   encoder's lifecycle block (``AudioConditioner``), the waveform-to-latent
   encoder (re-exported as ``vae_encode_audio``, because upstream has a second,
   unrelated ``encode_audio`` that writes wav files), the two file decoders and
   the three per-frame preprocessing ops chain25 assembles source pixels from on
   CPU, ``VideoEncoder.tiled_encode``'s silent 8k+1 crop, the 80/24 decode
   chunking a V2V trim has to be spliced across, and -- the one genuinely new
   MECHANISM -- ``ModalitySpec(frozen=True)``: both the branch that zeroes the
   whole denoise mask and the one that zeroes the scalar ``sigma`` with it, which
   is what makes A2V's frozen audio different in kind from the carry band.

The F1 canary (``Disposable.dispose`` metas storage via
``torch.empty_like(..., device="meta")``) is checked but only LOGGED, never
asserted: engine25's tensor subclass is written so that a future upstream that
stops doing this stays correct. It is recorded so the day the workaround
becomes unnecessary is visible in the log rather than guessed at.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
from typing import Any

# ---------------------------------------------------------------------------
# ltx_core -- loader
# ---------------------------------------------------------------------------
from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
# `validate_audio_waveform` is the stereo rule the mp4 mux enforces. It is NOT
# re-exported: nothing outside this file calls it, and chain25 satisfies the rule
# by construction (every waveform it muxes is duplicated to stereo on load). It
# is imported so `verify` can pin the rule itself -- a future upstream that
# accepted mono would make chain25's duplication silently unnecessary rather than
# wrong, but one that demanded a different shape would break the mux at the very
# end of a job.
from ltx_core.color.audio_mux import validate_audio_waveform
from ltx_core.components.diffusion_steps import EulerAncestralDiffusionStep, EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.conditioning.item import ConditioningItem
from ltx_core.conditioning.types.attention_strength_wrapper import (
    ConditioningItemAttentionStrengthWrapper,
)
from ltx_core.conditioning.types.mask_cond import VideoConditionByMask
from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core.loader.fuse_loras import FuseRule, bf16_fuse_rule
from ltx_core.loader.helpers import (
    as_path_list,
    create_meta_model,
    load_state_dict,
    read_model_metadata,
)
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import (
    LoraPathStrengthAndSDOps,
    ModelBuilderProtocol,
    StateDict,
    StateDictLoader,
)
from ltx_core.loader.registry import ModelRegistry, Registry, module_registry_key
from ltx_core.loader.sd_ops import ContentMatching, ContentReplacement, SDOps

# `_load_model_weights` / `_check_uninitialized` are module-private on purpose
# upstream. engine25 reuses them rather than re-implementing the LoRA-fusion +
# assign(strict=False) dance, which is subtle and would silently drift.
from ltx_core.loader.single_gpu_model_builder import (
    SingleGPUModelBuilder,
    _check_uninitialized,
    _load_model_weights,
)

# ---------------------------------------------------------------------------
# ltx_core -- model
# ---------------------------------------------------------------------------
# `encode_audio` is an AMBIGUOUS name upstream: this one is the audio VAE's
# waveform -> latent encoder, and `ltx_pipelines.utils.media_io.encode_audio` is
# a wav WRITER. They are re-exported under different names on purpose -- 2.3
# aliased the VAE one to `vae_encode_audio` at every call site for exactly this
# reason, and engine25 keeps that name so a reader never has to check which one
# a line means. `AudioProcessor` is verify-only (the mel front end
# `vae_encode_audio` builds when the caller passes `audio_processor=None`).
from ltx_core.model.audio_vae import AudioProcessor
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.disposable import Disposable, DisposableProtocol
from ltx_core.model.model_protocol import LTXModelProtocol, ModelConfigurator
from ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModel,
    LTXModelConfigurator,
    X0Model,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.upsampler import LatentUpsampler, upsample_video
from ltx_core.model.video_vae import (
    AUTO_TILING,
    AutoTiling,
    TileSizeConfig,
    TilingConfig,
    VideoEncoder,
    get_video_chunks_number,
)
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tools import AudioLatentTools, LatentTools, VideoLatentTools
from ltx_core.types import (
    Audio,
    AudioLatentShape,
    LatentState,
    VideoLatentShape,
    VideoPixelShape,
)

# ---------------------------------------------------------------------------
# ltx_core -- text encoders (Phase 2c consumes these; imported here so the ONE
# entry-point rule holds for the whole engine, not just the transformer half)
# ---------------------------------------------------------------------------
from ltx_core.text_encoders.gemma import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    EmbeddingsProcessorConfigurator,
    GemmaAssets,
    GemmaTextEncoderConfigurator,
    gemma_model_type,
    get_gemma_ops,
    resolve_gemma_weight_paths,
)
from ltx_core.text_encoders.gemma.encoders.encoder_configurator import (
    _build_gemma4_unified_llm_key_ops,
)

# ---------------------------------------------------------------------------
# ltx_pipelines
# ---------------------------------------------------------------------------
import ltx_pipelines.distilled as ltx_distilled
from ltx_pipelines.distilled import (
    DistilledPipeline,
    # THE SAME function object `DistilledPipeline.__call__` looks up by module
    # global on each stage. It is re-exported so the reference-conditioning
    # monkeypatch (engine25/reference25.py) has ONE name to save and restore,
    # and so `verify` can pin that the distilled namespace still owns it -- an
    # upstream that stopped name-importing it would make the patch a silent
    # no-op rather than an error.
    combined_image_conditionings,
    should_use_ancestral_sampler,
)
# The IC-LoRA metadata reader. Public in 1.2.0 (2.3 had to reach for
# `ic_lora._read_lora_reference_downscale_factor`); it returns 1 BOTH for a
# declared 1 and for an absent key, which is why the factor VOTE in
# reference25.py tests key presence with `safe_open` before letting a LoRA vote.
from ltx_pipelines.iclora_utils import read_lora_reference_downscale_factor
from ltx_pipelines.utils.args import ImageConditioningInput
# `_build_state` is module-private upstream and is NOT re-exported: chain25 never
# calls it. It is imported only so `verify` can read the branch that turns
# `ModalitySpec(frozen=True)` into an all-zero denoise mask -- the mechanism A2V's
# whole-timeline audio freeze rests on, and one that lives in an `if`, not in a
# signature.
from ltx_pipelines.utils.blocks import (
    AudioConditioner,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoUpsampler,
    _build_state,
    # Verify-only, and NOT re-exported: engine25 never calls either. They are
    # imported so `verify` can pin the fact the Single reference conditioning
    # rests on -- the `num_frames` its monkeypatch closes over is the value the
    # caller asked for, because a concrete int short-circuits `resolve_num_frames`
    # and `require_num_frames_source` is the guard that makes an AutoDuration
    # request (which engine25 never sends) fail before any work.
    require_num_frames_source,
    resolve_num_frames,
)
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.gpu_model import gpu_model
from ltx_pipelines.utils.helpers import (
    cleanup_memory,
    create_noised_state,
    ensure_tiling_config,
    image_conditionings_by_adding_guiding_latent,
    image_conditionings_by_replacing_latent,
    modality_from_latent_state,
    post_process_latent,
    state_with_conditionings,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.media_io import (
    decode_audio_from_file,
    decode_video_by_frame,
    encode_video,
    get_videostream_fps,
    normalize_images,
    resize_and_center_crop,
)
from ltx_pipelines.utils.model_paths import ModelPaths
# `_ancestral_euler_denoising_loop` is module-private upstream and is NOT
# re-exported: nothing outside this file calls it. It is imported only so
# `verify` can read the body that decides whether the conditioned tokens are
# re-pinned after each ancestral step -- the single fact the chain's frozen
# carry band depends on, and one that lives in an `if`, not in a signature.
from ltx_pipelines.utils.samplers import (
    _ancestral_euler_denoising_loop,
    euler_ancestral_denoising_loop,
    euler_denoising_loop,
)
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

logger = logging.getLogger(__name__)

__all__ = [
    "AUTO_TILING",
    "DISTILLED_SIGMAS",
    "EMBEDDINGS_PROCESSOR_KEY_OPS",
    "LTXV_MODEL_COMFY_RENAMING_MAP",
    "STAGE_2_DISTILLED_SIGMAS",
    "AllocatorTrimStrategy",
    "Audio",
    "AudioConditioner",
    "AudioLatentShape",
    "AudioLatentTools",
    "AudioPatchifier",
    "AutoTiling",
    "ConditioningItem",
    "ConditioningItemAttentionStrengthWrapper",
    "ContentMatching",
    "ContentReplacement",
    "DiffusionStage",
    "Disposable",
    "DistilledPipeline",
    "DisposableProtocol",
    "EmbeddingsProcessorConfigurator",
    "EulerAncestralDiffusionStep",
    "EulerDiffusionStep",
    "FuseRule",
    "GaussianNoiser",
    "GemmaAssets",
    "GemmaTextEncoderConfigurator",
    "ImageConditioner",
    "ImageConditioningInput",
    "LTXModel",
    "LTXModelConfigurator",
    "LTXModelProtocol",
    "LatentState",
    "LatentTools",
    "LatentUpsampler",
    "LoraPathStrengthAndSDOps",
    "ModalitySpec",
    "ModelBuilderProtocol",
    "ModelConfigurator",
    "ModelPaths",
    "ModelRegistry",
    "Modality",
    "ModuleOps",
    "OffloadMode",
    "PromptEncoder",
    "QuantizationPolicy",
    "Registry",
    "SDOps",
    "SimpleDenoiser",
    "SingleGPUModelBuilder",
    "StateDict",
    "StateDictLoader",
    "TileSizeConfig",
    "TilingConfig",
    "VideoConditionByMask",
    "VideoConditionByReferenceLatent",
    "VideoEncoder",
    "VideoLatentPatchifier",
    "VideoLatentShape",
    "VideoLatentTools",
    "VideoPixelShape",
    "VideoUpsampler",
    "X0Model",
    "_build_gemma4_unified_llm_key_ops",
    "_check_uninitialized",
    "_load_model_weights",
    "as_path_list",
    "bf16_fuse_rule",
    "cleanup_memory",
    "combined_image_conditionings",
    "create_meta_model",
    "decode_audio_from_file",
    "decode_video_by_frame",
    "create_noised_state",
    "encode_video",
    "ensure_tiling_config",
    "euler_ancestral_denoising_loop",
    "euler_denoising_loop",
    "gemma_model_type",
    "get_gemma_ops",
    "get_video_chunks_number",
    "get_videostream_fps",
    "gpu_model",
    "image_conditionings_by_adding_guiding_latent",
    "image_conditionings_by_replacing_latent",
    "load_state_dict",
    "ltx_distilled",
    "module_registry_key",
    "normalize_images",
    "post_process_latent",
    "read_lora_reference_downscale_factor",
    "read_model_metadata",
    "resize_and_center_crop",
    "resolve_gemma_weight_paths",
    "should_use_ancestral_sampler",
    "state_with_conditionings",
    "tiling_scale_factors_for_vae",
    "upsample_video",
    "upsampler_builders",
    "vae_encode_audio",
    "verify",
]


def upsampler_builders(upsampler: VideoUpsampler) -> tuple[Any, Any]:
    """The ``(encoder_builder, upsampler_builder)`` behind a :class:`VideoUpsampler`.

    The ONE reach into ``VideoUpsampler``'s privates, kept here for the same
    reason ``_load_model_weights`` is: the chunked upsample (§3-102) must build
    the video encoder and the latent upsampler **once** and drive them over ~N
    temporal chunks, while the public ``VideoUpsampler.__call__`` builds both,
    upsamples one tensor, and frees them again. Its ``registry`` is constructed
    with ``cache_weights=False``, so calling it per chunk would re-read the
    checkpoints from disk every time.

    Named as a function rather than re-exported as two attribute strings so the
    reach is one grep and :func:`verify` can assert the attribute names exist
    before any weights are touched.
    """
    return upsampler._encoder_builder, upsampler._upsampler_builder


class CompatError(RuntimeError):
    """The installed ltx_core / ltx_pipelines does not match what engine25 expects."""


def _fail(what: str, detail: str) -> None:
    raise CompatError(
        f"engine25/ltxcore_compat.py: {what} -- {detail}. The installed official "
        f"packages differ from the LTX-2 v1.2.0 (d151147) surface engine25 was "
        f"written against; engine25 must be updated before it can run."
    )


def _params(func: Any) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(func).parameters)


def _require_params(func: Any, label: str, *names: str) -> None:
    have = _params(func)
    missing = [n for n in names if n not in have]
    if missing:
        _fail(label, f"missing parameter(s) {missing}; got {sorted(have)}")


def _require_keyword_only(func: Any, label: str, *names: str) -> None:
    have = _params(func)
    for name in names:
        param = have.get(name)
        if param is None:
            _fail(label, f"parameter {name!r} is gone; got {sorted(have)}")
        if param.kind is not inspect.Parameter.KEYWORD_ONLY:
            _fail(label, f"parameter {name!r} is {param.kind.name}, expected KEYWORD_ONLY")


def _require_var_keyword(func: Any, label: str) -> None:
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in _params(func).values()):
        _fail(label, "no **kwargs sink; engine25 forwards extra build kwargs through it")


def _source_of(obj: Any, label: str) -> str:
    """``inspect.getsource`` with a CompatError instead of an OSError.

    Source checks are for the handful of places where the thing engine25 depends
    on is an EXPRESSION rather than a name -- the ancestral loop's re-pin
    condition, the order of the three statements in ``create_noised_state``, the
    two lines of arithmetic in ``VideoConditionByMask``. Each of those is a
    silent-wrong-output hazard if it drifts (see the callers), and none of them
    is reachable by ``inspect.signature``.
    """
    try:
        return inspect.getsource(obj)
    except OSError:  # pragma: no cover -- source unavailable (zipped install)
        _fail(label, "source is unavailable, so its body cannot be verified")
        return ""  # unreachable; _fail raises


def _require_in_source(source: str, label: str, *needles: str) -> None:
    missing = [n for n in needles if n not in source]
    if missing:
        _fail(label, f"body no longer contains {missing!r}")


def _require_source_order(source: str, label: str, *needles: str) -> None:
    """Assert *needles* appear in the given order, each exactly once-or-more."""
    position = -1
    for needle in needles:
        found = source.find(needle, position + 1)
        if found < 0:
            _fail(label, f"body no longer contains {needle!r} after the preceding step")
        position = found


def _require_dataclass_fields(cls: type, label: str, expected: tuple[str, ...]) -> None:
    if not dataclasses.is_dataclass(cls):
        _fail(label, "is no longer a dataclass")
    have = tuple(f.name for f in dataclasses.fields(cls))
    if have != expected:
        _fail(label, f"fields are {have}, expected {expected}")


def verify() -> None:
    """Assert the official surface engine25 depends on. Call at worker start.

    Raises :class:`CompatError` with the offending symbol named. Cheap: pure
    ``inspect`` over already-imported objects, no model or file touched, so it
    is affordable on every process start and belongs BEFORE the first build.
    """
    # 1. Every re-export is bound to something.
    globals_ = globals()
    for name in __all__:
        if name == "verify":
            continue
        if globals_.get(name) is None:
            _fail(name, "is not bound (None) after import")

    # 2. Signature points engine25 actually depends on.

    # The override contract for Ltx25DiffusionStage: keyword-only `device` and a
    # **kwargs sink (DiffusionStage.__call__ forwards `video_tools=`).
    _require_keyword_only(DiffusionStage._build_transformer, "DiffusionStage._build_transformer", "device")
    _require_var_keyword(DiffusionStage._build_transformer, "DiffusionStage._build_transformer")
    _require_params(
        DiffusionStage.__init__,
        "DiffusionStage.__init__",
        "transformer_builder",
        "dtype",
        "device",
        "quantization",
        "alloc_trim_strategy",
    )

    # The builder engine25 subclasses, and the two private helpers it reuses.
    _require_params(SingleGPUModelBuilder.build, "SingleGPUModelBuilder.build", "device", "dtype")
    _require_params(
        SingleGPUModelBuilder.__init__,
        "SingleGPUModelBuilder.__init__",
        "model_class_configurator",
        "model_path",
        "model_sd_ops",
        "module_ops",
        "loras",
        "model_loader",
        "registry",
        "fuse_rule",
    )
    for attr in ("model_sd_ops", "module_ops", "loras", "registry", "model_loader", "fuse_rule",
                 "model_path", "checkpoint", "lora_load_device"):
        if not isinstance(getattr(SingleGPUModelBuilder, attr, None), property):
            _fail("SingleGPUModelBuilder", f"read-only property {attr!r} is gone")
    for method in ("with_sd_ops", "with_module_ops", "with_loras", "with_registry", "with_fuse_rule",
                   "meta_model", "model_metadata", "model_config"):
        if not callable(getattr(SingleGPUModelBuilder, method, None)):
            _fail("SingleGPUModelBuilder", f"method {method!r} is gone")
    _require_params(
        _load_model_weights,
        "single_gpu_model_builder._load_model_weights",
        "meta_model",
        "model_path",
        "loras",
        "loader",
        "registry",
        "device",
        "dtype",
        "model_sd_ops",
        "lora_load_device",
        "fuse_rule",
    )
    _require_params(_check_uninitialized, "single_gpu_model_builder._check_uninitialized", "model")

    # Weight/shell caching -- the cache_weights=True default (plan H revision).
    _require_keyword_only(ModelRegistry.__init__, "ModelRegistry.__init__", "cache_weights", "cache_models")
    for method in ("add", "get", "pop", "clear", "get_model", "add_model", "pop_model"):
        if not callable(getattr(ModelRegistry, method, None)):
            _fail("ModelRegistry", f"method {method!r} is gone")
    # BEHAVIOURAL pin, not a name pin: ``keep_resident`` works by writing
    # ``_cache_weights`` on a live registry and popping the entry back out by
    # key (``pipeline25._swap_keep_resident``). A signature check cannot see any
    # of that -- a wheel that renamed the private attribute, or started reading
    # the flag in ``get`` as well, would leave every signature intact and turn
    # the feature into a silent 7.7 GiB leak. So the round trip is driven here,
    # on a THROWAWAY registry with a fake path list: no I/O, no GPU, no tensors.
    probe = ModelRegistry(cache_weights=False, cache_models=True)
    if not isinstance(getattr(probe, "_cache_weights", None), bool):
        _fail("ModelRegistry._cache_weights", "is no longer a bool attribute keep_resident can flip")
    probe._cache_weights = True
    probe.add(["<compat-probe>"], None, "sentinel")  # type: ignore[arg-type]
    if probe.get(["<compat-probe>"], None) is None:
        _fail("ModelRegistry.add", "no longer honours _cache_weights=True written after construction")
    if probe.pop(["<compat-probe>"], None) is None:
        _fail("ModelRegistry.pop", "did not return the entry keep_resident releases by key")
    if probe.get(["<compat-probe>"], None) is not None:
        _fail("ModelRegistry.pop", "left the entry in place; keep_resident's OFF path would leak it")

    # Quantization policy + state-dict ops.
    _require_dataclass_fields(
        QuantizationPolicy, "QuantizationPolicy", ("sd_ops", "module_ops", "model_configurator", "fuse_rule")
    )
    _require_dataclass_fields(SDOps, "SDOps", ("name", "mapping", "allowed_keys"))
    for method in ("with_matching", "with_replacement", "with_additional_allowed_keys",
                   "apply_to_key", "apply_to_key_value"):
        if not callable(getattr(SDOps, method, None)):
            _fail("SDOps", f"method {method!r} is gone")
    if tuple(getattr(ModuleOps, "_fields", ())) != ("name", "matcher", "mutator"):
        _fail("ModuleOps", f"fields are {getattr(ModuleOps, '_fields', None)}, expected (name, matcher, mutator)")

    # Meta-model construction + state-dict load path.
    _require_params(create_meta_model, "loader.helpers.create_meta_model", "configurator", "metadata", "module_ops")
    _require_params(load_state_dict, "loader.helpers.load_state_dict", "paths", "loader", "registry", "device", "sd_ops")
    _require_params(read_model_metadata, "loader.helpers.read_model_metadata", "model_path", "loader")

    # StateDict is what a custom loader must return.
    _require_dataclass_fields(StateDict, "StateDict", ("sd", "device", "size", "dtype"))

    # Model / stage plumbing.
    if not callable(getattr(Disposable, "dispose", None)):
        _fail("Disposable.dispose", "is gone")
    if not isinstance(getattr(LTXModel, "num_blocks", None), property):
        _fail("LTXModel.num_blocks", "is no longer a property")
    _require_params(X0Model.__init__, "X0Model.__init__", "velocity_model")
    if not dataclasses.is_dataclass(Modality):
        _fail("Modality", "is no longer a dataclass")
    _require_params(LTXModelConfigurator.from_metadata, "LTXModelConfigurator.from_metadata", "metadata")

    # Phase 2c/2d entry points (checked now so a drift is reported once, early).
    loras_param = _params(DistilledPipeline.__init__).get("loras")
    if loras_param is None:
        _fail("DistilledPipeline.__init__", "the `loras` parameter is gone")
    if loras_param.default is not inspect.Parameter.empty:
        _fail("DistilledPipeline.__init__", "`loras` gained a default; engine25 passes it explicitly by contract")
    _require_params(
        ModelPaths.from_split,
        "ModelPaths.from_split",
        "transformer_path",
        "text_encoder_path",
        "video_vae_path",
        "audio_vae_path",
        "duration_head_path",
    )
    # `text_encoder_builder` is the PUBLIC injection point engine25 assembles the
    # pipeline through: without it the only way to put GGUF weights behind the
    # official prompt encoder would be to fork the class.
    _require_params(
        PromptEncoder.__init__,
        "PromptEncoder.__init__",
        "model_paths",
        "dtype",
        "device",
        "text_encoder_builder",
        "alloc_trim_strategy",
    )
    # The two no-argument build methods ``Ltx25PromptEncoder`` overrides to time
    # them (``10a_te_build`` / ``10b_ep_build``). If either is renamed upstream
    # the override stops intercepting anything: the official build still runs,
    # so nothing fails -- the phase simply vanishes from the report, which is a
    # silent loss of measurement rather than an error. Pinned here so the rename
    # is reported at worker start instead.
    for method in ("_build_text_encoder", "_build_embeddings_processor"):
        if not callable(getattr(PromptEncoder, method, None)):
            _fail("PromptEncoder", f"method {method!r} is gone; engine25's phase timers no longer intercept it")
    _require_params(gpu_model, "gpu_model", "model", "alloc_trim_strategy")

    # --- Phase 2d: assembly surface -----------------------------------------
    _require_params(
        DistilledPipeline.__init__,
        "DistilledPipeline.__init__",
        "model_paths",
        "spatial_upsampler_path",
        "device",
        "registry",
        "offload_mode",
    )
    _require_params(
        DistilledPipeline.__call__,
        "DistilledPipeline.__call__",
        "prompt",
        "seed",
        "height",
        "width",
        "frame_rate",
        "images",
        "num_frames",
        "tiling_config",
        "stage_1_sigmas",
        "stage_2_sigmas",
    )
    # The three attributes engine25 substitutes / asserts after construction.
    # They are set in ``__init__``'s body, so the only cheap pre-build check is
    # that the names still appear there; a rename would otherwise surface as a
    # silently ignored assignment (Python creates the attribute either way).
    try:
        pipeline_init_source = inspect.getsource(DistilledPipeline.__init__)
    except OSError:  # pragma: no cover -- source unavailable (zipped install)
        pipeline_init_source = ""
    if pipeline_init_source:
        for attr in ("self.stage", "self.prompt_encoder", "self.use_ancestral_sampler"):
            if attr not in pipeline_init_source:
                _fail("DistilledPipeline.__init__", f"no longer assigns {attr!r}")
    # The stage call contract Ltx25DiffusionStage's progress override forwards.
    _require_params(
        DiffusionStage.__call__,
        "DiffusionStage.__call__",
        "denoiser",
        "sigmas",
        "noiser",
        "width",
        "height",
        "frames",
        "fps",
        "video",
        "audio",
        "stepper",
        "loop",
    )
    # The distilled schedules: 8 stage-1 steps + 3 stage-2 steps. A schedule
    # change is a generation change, not a refactor, so it is asserted.
    if tuple(DISTILLED_SIGMAS.shape) != (9,):
        _fail("DISTILLED_SIGMAS", f"has {tuple(DISTILLED_SIGMAS.shape)} entries, expected (9,) = 8 steps")
    if tuple(STAGE_2_DISTILLED_SIGMAS.shape) != (4,):
        _fail(
            "STAGE_2_DISTILLED_SIGMAS",
            f"has {tuple(STAGE_2_DISTILLED_SIGMAS.shape)} entries, expected (4,) = 3 steps",
        )
    # Output side.
    if tuple(getattr(ImageConditioningInput, "_fields", ())) != ("path", "frame_idx", "strength", "crf"):
        _fail(
            "ImageConditioningInput",
            f"fields are {getattr(ImageConditioningInput, '_fields', None)}, "
            f"expected ('path', 'frame_idx', 'strength', 'crf')",
        )
    _require_params(
        encode_video,
        "media_io.encode_video",
        "video",
        "fps",
        "audio",
        "output_path",
        "video_chunks_number",
    )
    _require_params(get_video_chunks_number, "get_video_chunks_number", "num_frames", "tiling_config")
    _require_params(should_use_ancestral_sampler, "should_use_ancestral_sampler", "transformer_path")

    # 3. §3-102 (Chained): the surface engine25/chain25.py drives directly.
    #
    # The chain does not go through ``DistilledPipeline.__call__``. It calls the
    # same blocks the pipeline calls, in its own order, with a frozen carry band
    # in front of every segment and tile -- so the pieces that pipeline's own
    # code would have guaranteed have to be asserted here instead.

    # (1) The two denoising loops, called by KEYWORD from ``DiffusionStage`` and
    #     re-bound by chain25 (ancestral needs ``noise_seed``). A renamed
    #     parameter would be a TypeError at the first denoise; naming it here
    #     makes it a named symbol at process start.
    for loop, label in (
        (euler_denoising_loop, "samplers.euler_denoising_loop"),
        (euler_ancestral_denoising_loop, "samplers.euler_ancestral_denoising_loop"),
    ):
        _require_params(loop, label, "sigmas", "video_state", "audio_state", "stepper", "transformer", "denoiser")
    _require_params(
        euler_ancestral_denoising_loop,
        "samplers.euler_ancestral_denoising_loop",
        "noise_seed",
        "new_noise_fn",
        "model_dtype",
    )

    # (2) THE ancestral fact the frozen carry band rests on: the loop re-pins
    #     the conditioned tokens (``post_process_latent``) only on the branch it
    #     takes when it draws noise, and it draws noise iff ``stepper.eta > 0``.
    #     At eta == 0 an ancestral run would leave the band drifting, silently --
    #     which is why chain25 asserts ``stepper.eta > 0`` at every ancestral
    #     call site and why the CONDITION ITSELF is pinned here.
    _require_in_source(
        _source_of(euler_ancestral_denoising_loop, "samplers.euler_ancestral_denoising_loop"),
        "samplers.euler_ancestral_denoising_loop",
        "draw_noise=stepper.eta > 0",
    )
    _require_source_order(
        _source_of(_ancestral_euler_denoising_loop, "samplers._ancestral_euler_denoising_loop"),
        "samplers._ancestral_euler_denoising_loop",
        "if draw_noise:",
        "post_process_latent(x_next, step.state.denoise_mask, step.state.clean_latent)",
    )
    _require_params(
        post_process_latent, "helpers.post_process_latent", "denoised", "denoise_mask", "clean"
    )

    # (3) THE ORDER the band's bit-exactness rests on. 2.3 froze its carry by
    #     editing ``denoise_mask`` between ``create_initial_state`` and the
    #     noiser; 2.5's ``VideoConditionByMask`` reaches the same numbers only
    #     because ``create_noised_state`` runs initial -> conditionings ->
    #     noiser in exactly that order. If a release ever noised first, the band
    #     would be noise instead of carry-over and nothing would raise.
    _require_source_order(
        _source_of(create_noised_state, "helpers.create_noised_state"),
        "helpers.create_noised_state",
        "tools.create_initial_state(",
        "state_with_conditionings(",
        "noiser(state, noise_scale)",
    )
    _require_params(
        create_noised_state,
        "helpers.create_noised_state",
        "tools",
        "conditionings",
        "noiser",
        "noise_scale",
        "initial_latent",
    )
    _require_params(
        state_with_conditionings, "helpers.state_with_conditionings", "latent_state", "conditioning_items"
    )
    # ``clean_latent`` is the carry-over itself: chain25 hands the previous
    # segment's tail in as ``initial_latent`` and relies on the clone landing in
    # ``clean_latent``, because that is what the mask pins back every step.
    _require_in_source(
        _source_of(VideoLatentTools.create_initial_state, "VideoLatentTools.create_initial_state"),
        "VideoLatentTools.create_initial_state",
        "clean_latent = initial_latent.clone()",
    )
    _require_in_source(
        _source_of(AudioLatentTools.create_initial_state, "AudioLatentTools.create_initial_state"),
        "AudioLatentTools.create_initial_state",
        "clean_latent = initial_latent.clone()",
    )

    # (4) The band item's own arithmetic. ``AudioHeadBandMask`` in chain25 is a
    #     line-for-line audio twin of these two expressions; pinning them is what
    #     lets that twin be called "the same freeze" rather than "a similar one".
    _require_in_source(
        _source_of(VideoConditionByMask.apply_to, "VideoConditionByMask.apply_to"),
        "VideoConditionByMask.apply_to",
        "clean_latent=latent_state.clean_latent * inv + tokens * m",
        "denoise_mask=latent_state.denoise_mask * inv + (1.0 - self.strength) * m",
        "inv = 1 - m",
    )
    _require_params(VideoConditionByMask.__init__, "VideoConditionByMask.__init__", "latent", "mask", "strength")
    if not callable(getattr(ConditioningItem, "apply_to", None)):
        _fail("ConditioningItem", "the `apply_to` protocol method is gone")

    # (5) ``keyframes_mask`` (new in 1.2.0). ``create_initial_state`` marks the
    #     first latent frame UNCONDITIONALLY; for a chain segment/tile i >= 1
    #     that frame is carried-over content, not a keyframe, so chain25 clears
    #     the marker. Both the field and the marking helper are pinned: were the
    #     marking to disappear, the clear would become a silent no-op rather
    #     than an error.
    if "keyframes_mask" not in {f.name for f in dataclasses.fields(LatentState)}:
        _fail("LatentState", "no longer has a `keyframes_mask` field")
    if not callable(getattr(VideoLatentTools, "_first_frame_keyframes_mask", None)):
        _fail("VideoLatentTools._first_frame_keyframes_mask", "is gone")
    if not isinstance(getattr(DiffusionStage, "supports_generated_keyframes", None), property):
        _fail("DiffusionStage.supports_generated_keyframes", "is no longer a property")

    # (6) Stage plumbing chain25 drives by hand.
    _require_dataclass_fields(
        ModalitySpec, "ModalitySpec", ("context", "conditionings", "noise_scale", "frozen", "initial_latent")
    )
    _require_params(SimpleDenoiser.__init__, "SimpleDenoiser.__init__", "v_context", "a_context")
    _require_params(GaussianNoiser.__init__, "GaussianNoiser.__init__", "generator")
    for name in ("eta", "s_noise"):
        if name not in _params(EulerAncestralDiffusionStep.__init__):
            _fail("EulerAncestralDiffusionStep.__init__", f"parameter {name!r} is gone")
    _require_params(EulerDiffusionStep.step, "EulerDiffusionStep.step", "sample", "denoised_sample", "sigmas")
    if not hasattr(DiffusionStage, "video_scale_factors") and "self.video_scale_factors" not in _source_of(
        DiffusionStage.__init__, "DiffusionStage.__init__"
    ):
        _fail("DiffusionStage", "no longer publishes `video_scale_factors`")

    # (7) Geometry. chain25 rebuilds the stage's own latent shapes to size the
    #     carry tensors, so its arithmetic must be the stage's arithmetic.
    _require_params(
        VideoLatentShape.from_pixel_shape, "VideoLatentShape.from_pixel_shape", "shape", "scale_factors"
    )
    _require_params(
        AudioLatentShape.from_video_pixel_shape, "AudioLatentShape.from_video_pixel_shape", "shape"
    )
    if tuple(VideoPixelShape._fields) != ("batch", "frames", "height", "width", "fps"):
        _fail("VideoPixelShape", f"fields are {VideoPixelShape._fields}, expected (batch, frames, height, width, fps)")
    # 25 audio latents per second -- the constant ``chain_math``'s whole audio
    # side (ka_list, a_tiles, a_total) is computed from. A different rate would
    # misalign every audio carry band without changing a single shape.
    one_second = AudioLatentShape.from_video_pixel_shape(VideoPixelShape(1, 24, 64, 64, 24.0))
    if one_second.frames != 25:
        _fail(
            "AudioLatentShape.from_video_pixel_shape",
            f"resolves 1 second to {one_second.frames} latent frames, expected 25 "
            f"(chain_math.AUDIO_LATENTS_PER_SEC)",
        )
    if AudioPatchifier(patch_size=1).get_token_count(one_second) != one_second.frames:
        _fail("AudioPatchifier", "one token per audio latent frame no longer holds at patch_size=1")

    # (8) Image conditioning -- the REPLACING / GUIDING pair 2.3 routes by
    #     frame_idx, plus the CRF fill-in that must run before either.
    for fn, label in (
        (image_conditionings_by_replacing_latent, "helpers.image_conditionings_by_replacing_latent"),
        (image_conditionings_by_adding_guiding_latent, "helpers.image_conditionings_by_adding_guiding_latent"),
    ):
        _require_params(fn, label, "images", "height", "width", "video_encoder", "dtype", "device")
    _require_params(ImageConditioner.resolve_crf, "ImageConditioner.resolve_crf", "images")
    _require_params(ImageConditioner.__call__, "ImageConditioner.__call__", "fn")

    # (9) The chunked upsample. ``upsampler_builders`` is engine25's one reach
    #     into VideoUpsampler's privates (see that function); the layer count
    #     below is what makes ``chain_math.UPSAMPLE_HALO_FRAMES == 18`` a
    #     derived number rather than a remembered one -- LatentUpsampler's
    #     temporal receptive field is one Conv3d(k=3) radius per Conv3d layer,
    #     and at ``num_blocks_per_stage=4`` there are 1 + 2*4 + 2*4 + 1 = 18 of
    #     them (the spatial upsampler itself is Conv2d, so it adds none).
    upsampler_source = _source_of(VideoUpsampler.__init__, "VideoUpsampler.__init__")
    _require_in_source(
        upsampler_source, "VideoUpsampler.__init__", "self._encoder_builder", "self._upsampler_builder"
    )
    _require_params(upsample_video, "upsampler.upsample_video", "latent", "video_encoder", "upsampler")
    _require_params(
        LatentUpsampler.__init__,
        "LatentUpsampler.__init__",
        "num_blocks_per_stage",
        "dims",
        "spatial_upsample",
        "temporal_upsample",
    )
    _require_in_source(
        _source_of(upsample_video, "upsampler.upsample_video"),
        "upsampler.upsample_video",
        "video_encoder.per_channel_statistics",
    )

    # (10) Decode side. The three keyword-only arguments are the ones a
    #      positional call would silently mis-bind.
    _require_keyword_only(
        ensure_tiling_config,
        "helpers.ensure_tiling_config",
        "scale_factors",
        "video_shape",
        "vae_checkpoint_path",
    )
    _require_params(tiling_scale_factors_for_vae, "helpers.tiling_scale_factors_for_vae", "vae_checkpoint_path")

    # (11) §3-102 second stage (V2V + A2V): the material-ingest surface.
    #
    # Everything below is about turning an UPLOADED file into a latent chain25
    # can freeze. None of it is reached by a plain T2V/I2V chain, so a drift here
    # would first show up as a wrong-looking continuation rather than a crash --
    # which is why the shapes and the two `if` branches are pinned by name.

    # (11a) The audio encoder's lifecycle block, the audio twin of
    #       ImageConditioner. `Ltx25Pipeline` constructs it with
    #       (checkpoint_path, dtype, device, registry=None) and drives it as
    #       `audio_conditioner(fn)` exactly like the image one.
    _require_params(
        AudioConditioner.__init__, "blocks.AudioConditioner.__init__",
        "checkpoint_path", "dtype", "device", "registry", "alloc_trim_strategy",
    )
    _require_params(AudioConditioner.__call__, "blocks.AudioConditioner.__call__", "fn")

    # (11b) The waveform -> latent encoder, and the mel front end it builds when
    #       the caller passes `audio_processor=None` (chain25 always does, so the
    #       four constructor arguments below are read off the encoder every time).
    #       NOT `media_io.encode_audio`, which is a wav writer -- see the import.
    _require_params(vae_encode_audio, "audio_vae.encode_audio", "audio", "audio_encoder", "audio_processor")
    _require_params(
        AudioProcessor.__init__, "audio_vae.AudioProcessor.__init__",
        "target_sample_rate", "mel_bins", "mel_hop_length", "n_fft",
    )
    # The encoder takes the waveform's dtype through to the mel transform, so
    # chain25 casts to bf16 BEFORE the call rather than after. `Audio.to` is the
    # only cast the encoder itself performs, and it is device-only.
    _require_in_source(
        _source_of(vae_encode_audio, "audio_vae.encode_audio"),
        "audio_vae.encode_audio",
        "audio_processor.waveform_to_mel(audio.to(device=device))",
    )

    # (11c) The decoders chain25 ingests material through. `decode_audio_from_file`
    #       returning a 3-D (1, channels, samples) waveform is what makes
    #       "channels is dim 1" -- and therefore the mono-to-stereo duplication --
    #       correct; `decode_video_by_frame` is the ONLY 1.2.0 video decoder with
    #       a `frame_cap`, which is how a tail context is read without decoding
    #       the whole upload.
    _require_params(
        decode_audio_from_file, "media_io.decode_audio_from_file",
        "path", "device", "start_time", "max_duration",
    )
    _require_in_source(
        _source_of(decode_audio_from_file, "media_io.decode_audio_from_file"),
        "media_io.decode_audio_from_file",
        "torch.from_numpy(audio).to(device).unsqueeze(0)",
    )
    _require_params(
        decode_video_by_frame, "media_io.decode_video_by_frame",
        "path", "device", "starting_frame", "frame_cap",
    )
    _require_params(resize_and_center_crop, "media_io.resize_and_center_crop", "tensor", "height", "width")
    _require_params(normalize_images, "media_io.normalize_images", "images", "device", "dtype")
    # chain25 assembles the source pixels on CPU instead of calling
    # `video_preprocess` (which cats on the GPU, O(F^2) in allocated volume). The
    # per-frame maths must stay the official one for that to be a memory
    # optimisation rather than a different preprocessing.
    _require_in_source(
        _source_of(normalize_images, "media_io.normalize_images"),
        "media_io.normalize_images",
        "(images / 127.5 - 1.0).to(device=device, dtype=dtype)",
    )
    _require_params(get_videostream_fps, "media_io.get_videostream_fps", "path")

    # (11d) The VAE encode chain25 drives directly. `tiled_encode` accepts a CPU
    #       tensor (that is the whole point of the CPU assembly) and CROPS a
    #       frame count that is not 8k+1 with only a warning -- chain25 asserts
    #       the count itself, so the pin here is on the crop still being the
    #       silent behaviour it guards against.
    _require_params(VideoEncoder.tiled_encode, "video_vae.VideoEncoder.tiled_encode", "video", "tiling_config")
    _require_in_source(
        _source_of(VideoEncoder.tiled_encode, "video_vae.VideoEncoder.tiled_encode"),
        "video_vae.VideoEncoder.tiled_encode",
        "frames_to_crop = (frames - 1) % self.video_scale_factors.time",
        "video = video[:, :, :-frames_to_crop, ...]",
    )
    # The decode chunking chain25 shares with the encode. 80/24 is a stride of 56
    # frames, which is what makes a V2V trim of 25..145 pixel frames land ACROSS
    # chunk boundaries -- the reason `_drop_leading_frames` exists at all.
    default_tiles = TileSizeConfig.default()
    if (default_tiles.frames.tile_size, default_tiles.frames.overlap) != (80, 24):
        _fail(
            "TileSizeConfig.default",
            f"temporal tiling is {default_tiles.frames.tile_size}/{default_tiles.frames.overlap}, "
            f"expected 80/24 (chain25's leading-frame trim is written against that stride)",
        )

    # (11e) The FROZEN modality -- A2V's whole-timeline audio freeze. This is a
    #       different mechanism from the band (`AudioHeadBandMask`), and
    #       deliberately so: the band leaves `sigma` alone, while `frozen` zeroes
    #       the scalar too. Both halves are pinned because A2V's correctness is
    #       "the audio the model attends to is EXACTLY the upload", and a lost
    #       `frozen` branch would degrade that to "mostly".
    _require_in_source(
        _source_of(_build_state, "blocks._build_state"),
        "blocks._build_state",
        "if spec.frozen:",
        "denoise_mask=torch.zeros_like(state.denoise_mask)",
        "frozen=True",
    )
    _require_in_source(
        _source_of(modality_from_latent_state, "helpers.modality_from_latent_state"),
        "helpers.modality_from_latent_state",
        "if state.frozen:",
        "sigma = torch.zeros_like(sigma)",
    )
    if "frozen" not in {f.name for f in dataclasses.fields(LatentState)}:
        _fail("LatentState", "no longer has a `frozen` field; ModalitySpec(frozen=True) cannot reach the model")
    # `create_initial_state` asserts the SHAPE of an initial latent and does NOT
    # convert its dtype -- which is why every window slice chain25 hands in is
    # cast explicitly rather than left to the stage.
    _require_in_source(
        _source_of(AudioLatentTools.create_initial_state, "AudioLatentTools.create_initial_state"),
        "AudioLatentTools.create_initial_state",
        "assert initial_latent.shape == self.target_shape.to_torch_shape()",
    )

    # (11f) The mux's stereo rule. chain25 muxes the ORIGINAL upload for A2V and
    #       a trimmed vocoder render for V2V; both are duplicated to stereo on
    #       load, which is only meaningful while this rule stands.
    _require_in_source(
        _source_of(validate_audio_waveform, "audio_mux.validate_audio_waveform"),
        "audio_mux.validate_audio_waveform",
        "if samples.ndim != 2 or 2 not in samples.shape:",
    )

    # (12) §3-102 third stage (Style LoRA + IC-LoRA): the reference-conditioning
    #      surface. Everything below is what engine25/reference25.py either
    #      MONKEYPATCHES or constructs by hand, so a drift here would show up as
    #      "the reference had no effect" -- a silent wrong picture, not a crash.

    # (12a) The patch seam. `DistilledPipeline.__call__` looks `combined_image_conditionings`
    #       up as a MODULE GLOBAL of `ltx_pipelines.distilled` once per stage, which
    #       is the only reason a rebinding on that module intercepts both calls. An
    #       upstream that imported it as `helpers.combined_image_conditionings` at
    #       the call site, or inlined it, would leave the patch installed and inert.
    if getattr(ltx_distilled, "combined_image_conditionings", None) is not combined_image_conditionings:
        _fail(
            "ltx_pipelines.distilled.combined_image_conditionings",
            "is no longer the name-imported function object engine25 patches; the "
            "reference conditioning would be silently dropped",
        )
    _require_params(
        combined_image_conditionings,
        "helpers.combined_image_conditionings",
        "images", "height", "width", "video_encoder", "dtype", "device", "color_space",
    )

    # (12b) The STAGE DISCRIMINATOR. `height` is the only per-stage-differing
    #       argument the conditioning function receives (no stage index, no
    #       `num_frames`, no `tiling_config`), so `height == full_height // 2`
    #       IS the stage-1 test -- and it is only valid while stage 1 is built at
    #       half height. 2.3 leans on the same signal for the same reason.
    distilled_call_source = _source_of(DistilledPipeline.__call__, "DistilledPipeline.__call__")
    _require_in_source(
        distilled_call_source,
        "DistilledPipeline.__call__",
        "stage_1_w, stage_1_h = width // 2, height // 2",
    )
    # ...and that BOTH stages go through the patched name, stage 1 first. If a
    # release reordered them the patch would attach the reference to stage 2.
    _require_source_order(
        distilled_call_source,
        "DistilledPipeline.__call__",
        "stage_1_conditionings = self.image_conditioner(",
        "combined_image_conditionings(",
        "stage_2_conditionings = self.image_conditioner(",
        "combined_image_conditionings(",
    )

    # (12c) `num_frames` in the closure. The patch closes over the frame count the
    #       CALLER asked for, while the conditioning function is called with the
    #       count the pipeline RESOLVED -- the two are the same number only because
    #       a concrete int short-circuits `resolve_num_frames`, and an AutoDuration
    #       request (which engine25 never sends, and this checkpoint's absent
    #       DurationHead could not serve anyway) is refused up front.
    _require_in_source(
        distilled_call_source,
        "DistilledPipeline.__call__",
        "require_num_frames_source(num_frames, self.duration_predictor)",
    )
    _require_in_source(
        _source_of(resolve_num_frames, "blocks.resolve_num_frames"),
        "blocks.resolve_num_frames",
        "if not isinstance(num_frames, AutoDuration):",
        "return num_frames",
    )
    _require_params(
        require_num_frames_source,
        "blocks.require_num_frames_source",
        "num_frames", "duration_predictor",
    )

    # (12d) The reference conditioning item engine25 constructs by hand rather
    #       than through `append_ic_lora_reference_video_conditionings` (which
    #       decodes with a GPU-side cat, O(F^2) in allocated volume, and cannot be
    #       fed the per-segment pixel windows a long chain needs).
    _require_params(
        VideoConditionByReferenceLatent.__init__,
        "VideoConditionByReferenceLatent.__init__",
        "latent", "downscale_factor", "temporal_scale_factor", "strength",
    )
    #       The three body facts that make "strength=1.0 keeps the reference
    #       clean" true, and that separate 2.5's item from 2.3's:
    #         * the appended denoise mask is `1 - strength` (so 1.0 == clean);
    #         * the NOISY side gets `zeros_like`, i.e. placeholder zeros rather
    #           than the reference tokens themselves;
    #         * the appended tokens are explicitly NOT keyframes -- a
    #           position-derived marker would otherwise claim the reference's own
    #           first latent frame and change what the model attends to.
    _require_in_source(
        _source_of(VideoConditionByReferenceLatent.apply_to, "VideoConditionByReferenceLatent.apply_to"),
        "VideoConditionByReferenceLatent.apply_to",
        "fill_value=1.0 - self.strength",
        "torch.zeros_like(tokens)",
        "marked=False",
    )
    # (12e) The attention-strength wrapper, applied ONLY below 1.0 (at 1.0 the
    #       bare item is passed through, which is what keeps an ordinary
    #       reference job structurally identical to the upstream one).
    _require_params(
        ConditioningItemAttentionStrengthWrapper.__init__,
        "ConditioningItemAttentionStrengthWrapper.__init__",
        "conditioning", "attention_mask",
    )
    # (12f) The metadata reader behind the downscale-factor VOTE. It DEFAULTS to
    #       1 on an absent key, which is indistinguishable from a declared 1 --
    #       hence reference25's separate `safe_open` presence test. Pinned so the
    #       day upstream starts raising instead, the vote is revisited.
    _require_params(
        read_lora_reference_downscale_factor,
        "iclora_utils.read_lora_reference_downscale_factor",
        "lora_path",
    )
    _require_in_source(
        _source_of(read_lora_reference_downscale_factor, "iclora_utils.read_lora_reference_downscale_factor"),
        "iclora_utils.read_lora_reference_downscale_factor",
        'metadata.get("reference_downscale_factor", 1)',
    )

    # 4. F1 canary -- logged, never asserted (see module docstring).
    try:
        source = inspect.getsource(Disposable.dispose)
        metas_storage = 'device="meta"' in source or "device='meta'" in source
    except OSError:  # pragma: no cover -- source unavailable (zipped install)
        metas_storage = None
    logger.info(
        "ltxcore_compat.verify OK (Disposable.dispose metas storage via empty_like: %s)",
        metas_storage,
    )
