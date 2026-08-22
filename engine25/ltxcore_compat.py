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
from ltx_core.model.disposable import Disposable, DisposableProtocol
from ltx_core.model.model_protocol import LTXModelProtocol, ModelConfigurator
from ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModel,
    LTXModelConfigurator,
    X0Model,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.video_vae import (
    AUTO_TILING,
    AutoTiling,
    TilingConfig,
    get_video_chunks_number,
)
from ltx_core.quantization import QuantizationPolicy

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
from ltx_pipelines.distilled import DistilledPipeline, should_use_ancestral_sampler
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.blocks import DiffusionStage, PromptEncoder
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.gpu_model import gpu_model
from ltx_pipelines.utils.helpers import cleanup_memory
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import OffloadMode

logger = logging.getLogger(__name__)

__all__ = [
    "AUTO_TILING",
    "DISTILLED_SIGMAS",
    "EMBEDDINGS_PROCESSOR_KEY_OPS",
    "LTXV_MODEL_COMFY_RENAMING_MAP",
    "STAGE_2_DISTILLED_SIGMAS",
    "AllocatorTrimStrategy",
    "AutoTiling",
    "ContentMatching",
    "ContentReplacement",
    "DiffusionStage",
    "Disposable",
    "DistilledPipeline",
    "DisposableProtocol",
    "EmbeddingsProcessorConfigurator",
    "FuseRule",
    "GemmaAssets",
    "GemmaTextEncoderConfigurator",
    "ImageConditioningInput",
    "LTXModel",
    "LTXModelConfigurator",
    "LTXModelProtocol",
    "LoraPathStrengthAndSDOps",
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
    "SingleGPUModelBuilder",
    "StateDict",
    "StateDictLoader",
    "TilingConfig",
    "X0Model",
    "_build_gemma4_unified_llm_key_ops",
    "_check_uninitialized",
    "_load_model_weights",
    "as_path_list",
    "bf16_fuse_rule",
    "cleanup_memory",
    "create_meta_model",
    "encode_video",
    "gemma_model_type",
    "get_gemma_ops",
    "get_video_chunks_number",
    "gpu_model",
    "load_state_dict",
    "module_registry_key",
    "read_model_metadata",
    "resolve_gemma_weight_paths",
    "should_use_ancestral_sampler",
    "verify",
]


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

    # 3. F1 canary -- logged, never asserted (see module docstring).
    try:
        source = inspect.getsource(Disposable.dispose)
        metas_storage = 'device="meta"' in source or "device='meta'" in source
    except OSError:  # pragma: no cover -- source unavailable (zipped install)
        metas_storage = None
    logger.info(
        "ltxcore_compat.verify OK (Disposable.dispose metas storage via empty_like: %s)",
        metas_storage,
    )
