"""LTX 2.5 transformer: GGUF load path, per-layer dequant, block swap (§3-98 Phase 2b).

This is the whole 14.7GB-transformer-on-a-16GB-card story for the 2.5 engine.
The official ``DiffusionStage.from_checkpoint`` cannot be used as shipped: it
reads a bf16 safetensors monolith straight onto the GPU. So this module supplies
the four replacement parts and re-assembles the stage around them.

1. :class:`Ltx25GGMLTensor` -- the F1 fix
   Official 1.2.0 added ``Disposable.dispose()``, which releases storage by
   replacing every parameter and persistent buffer with
   ``torch.empty_like(x, device="meta")``. The two-stage distilled pipeline
   disposes the transformer between stage 1 and stage 2 *within a single job*,
   so this runs on every generation, not just between jobs.

   ``GGMLQuantizedTensor`` (2.3's, reused here) keeps its quantisation metadata
   -- ``_ggml_type`` and ``_float_shape`` -- as plain Python attributes, and
   reports the *dequantised* shape from a ``shape`` property that reads
   ``_float_shape``. ``empty_like`` returns a fresh instance of the subclass
   with those attributes ABSENT, so the very next ``.shape`` access raises
   AttributeError. Reproduced in the real venv before this module was written:
   without the fix the first job dies at the stage-1 -> stage-2 handover.

   The fix is a ``__torch_function__`` that re-attaches the metadata to any
   result of the subclass type. It covers ``empty_like`` / ``empty``
   specifically (the dispose call site) and every other op generically, which
   also closes the same hole in ``detach()`` -- reached from
   ``dispose()``'s own ``self.state_dict()`` call one line earlier.

2. :class:`Ltx25GgufStateDictLoader` -- GGUF in place of safetensors
   Keys are used as-is: the converter already stripped the
   ``model.diffusion_model.`` prefix. Metadata is rebuilt from the GGUF KV block
   in **post-parse form** (``config`` a dict, ``model_version`` a str,
   ``gemma_source_checkpoint`` a dict), matching what
   ``SafetensorsModelStateDictLoader.metadata`` produces from the official bf16
   header -- ``_check_gemma_version`` reads ``gemma_source_checkpoint.gemma_version``
   as an attribute-ish dict lookup, and a raw JSON *string* there fails.
   Tensor dtypes are preserved exactly (F32 stays F32), because the official
   standard build path never passes ``dtype=`` to ``build()`` either.

3. SDOps + allowed_keys -- dropping the 258 connector tensors
   The GGUF carries the whole checkpoint, including the 129+129 video/audio
   ``*_embeddings_connector.*`` tensors that belong to the text-encoder-side
   EmbeddingsProcessor, not to ``LTXModel``. They are filtered at load time so
   they never enter the cached state dict. The filter is an ``allowed_keys``
   set on an otherwise all-pass SDOps, which is also why the quantization policy
   below MUST keep ``sd_ops=None``: ``_chain_quantization`` rebuilds the SDOps
   from ``name`` + ``mapping`` only and would silently drop ``allowed_keys``.

4. :class:`Ltx25CpuModelBuilder` + :class:`Ltx25DiffusionStage` -- placement
   The builder loads onto CPU and does NOT call ``.to(device)``; the stage's
   ``_build_transformer`` override does not either. Both of those official
   ``.to(device)`` calls would put all 14.7GB on the GPU at once. Instead the
   stage moves everything EXCEPT ``transformer_blocks`` to the GPU (a few
   hundred MB) and hands the blocks to the block-swap service, which streams a
   window of N blocks in and out during the forward pass.
   Meta residue is a hard error here, not the upstream WARNING: with quantised
   weights arriving as buffers, a key that failed to land is a silent
   wrong-numbers bug rather than a crash.

Weight caching (plan H revision): the registry defaults to
``cache_weights=True``. ``dispose()`` metas the model's storage, so stage 2 would
otherwise re-read 14.7GB from disk mid-job. The cost is ~14.7GB of resident RAM
on top of the working copy; ``cache_weights=False`` is available for RAM-tight
machines.

Reuse from the 2.3 engine (explicitly sanctioned; those files are not edited):
``engine.gguf.quant_service`` for the tensor subclass and the per-layer dequant
module op, ``engine.transformer.block_swap_service`` for the swap itself.
``engine.gemma.gguf_quant_service`` is deliberately NOT used -- it is Gemma3
specific and folds RMSNorm.

Selftest (gate G2)::

    python -m engine25.gguf_transformer --selftest <path-to-transformer.gguf>

builds the transformer, runs a one-step forward on dummy input, disposes, and
repeats -- proving the dispose -> rebuild cycle and reporting per-phase VRAM
peaks and timings as JSON.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch

# --- 2.3 engine reuse (import only; those modules are never edited) ----------
from engine.gguf.quant_service import (
    GGMLQuantizedTensor,
    _patch_model_for_ggml_dequant,
)
from engine.transformer.block_swap_service import (
    _BLOCK_SWAP_ATTR,
    BlockSwapService,
)

# --- official surface: ONE entry point (see engine25/ltxcore_compat.py) ------
from engine25.ltxcore_compat import (
    DiffusionStage,
    LTXModel,
    LTXModelConfigurator,
    Modality,
    ModelRegistry,
    ModuleOps,
    QuantizationPolicy,
    Registry,
    SDOps,
    SingleGPUModelBuilder,
    StateDict,
    X0Model,
    _check_uninitialized,
    _load_model_weights,
    bf16_fuse_rule,
    cleanup_memory,
)

logger = logging.getLogger(__name__)

_CPU = torch.device("cpu")

# GGML type ids for the three float encodings. Re-declared rather than imported
# from quant_service (whose copies are module-private) -- three integers fixed
# by the GGUF spec are not worth a private reach.
_GGML_F32 = 0
_GGML_F16 = 1
_GGML_BF16 = 30

_GGML_FLOAT_DTYPES = {
    _GGML_F32: torch.float32,
    _GGML_F16: torch.float16,
    _GGML_BF16: torch.bfloat16,
}

#: Name of the all-pass transformer SDOps. Registry cache identity is keyed on
#: ``(paths, sd_ops.name)``, so this string is part of the cache key -- changing
#: it invalidates cached weights, and reusing it for a DIFFERENT filter would
#: alias two different state dicts onto one entry.
LTX25_GGUF_PASSTHRU = "LTX25_GGUF_PASSTHRU"

#: Prefixes of the tensors that belong to the EmbeddingsProcessor rather than to
#: ``LTXModel``: the two connectors (129 tensors each) and the text-embedding
#: projection. Dropped from the transformer's state dict.
_EMBEDDINGS_PREFIXES = (
    "video_embeddings_connector.",
    "audio_embeddings_connector.",
    "text_embedding_projection.",
)

#: Renaming for the EmbeddingsProcessor half of the same GGUF (Phase 2c consumes
#: this). Exactly four rules -- the official ``EMBEDDINGS_PROCESSOR_KEY_OPS``
#: rules minus the ``model.diffusion_model.`` prefix the converter already
#: stripped, and minus the V1 single-``aggregate_embed`` rule that a 2.5
#: checkpoint never has.
LTX25_EMBEDDINGS_PROCESSOR_KEY_OPS = (
    SDOps("LTX25_EMBEDDINGS_PROCESSOR_KEY_OPS")
    .with_matching(prefix="text_embedding_projection.video_aggregate_embed.")
    .with_replacement("text_embedding_projection.video_aggregate_embed.", "feature_extractor.video_aggregate_embed.")
    .with_matching(prefix="text_embedding_projection.audio_aggregate_embed.")
    .with_replacement("text_embedding_projection.audio_aggregate_embed.", "feature_extractor.audio_aggregate_embed.")
    .with_matching(prefix="video_embeddings_connector.")
    .with_replacement("video_embeddings_connector.", "video_connector.")
    .with_matching(prefix="audio_embeddings_connector.")
    .with_replacement("audio_embeddings_connector.", "audio_connector.")
)


class Ltx25BuildError(RuntimeError):
    """A transformer build could not be completed correctly."""


# ---------------------------------------------------------------------------
# 1. Ltx25GGMLTensor -- survives dispose() (F1)
# ---------------------------------------------------------------------------


class Ltx25GGMLTensor(GGMLQuantizedTensor):
    """``GGMLQuantizedTensor`` whose quantisation metadata survives torch ops.

    The base class stores ``_ggml_type`` / ``_float_shape`` as Python attributes
    on the instance and reads ``_float_shape`` from its ``shape`` / ``size()`` /
    ``dim()`` / ``numel()`` overrides. Any torch operation that produces a new
    tensor of the subclass type -- ``empty_like``, ``detach``, ``clone``,
    ``to`` -- gets those overrides but not the attributes they read, so the
    result is a live grenade: it looks like a quantised tensor and raises
    AttributeError the first time anything asks for its shape.

    ``Disposable.dispose()`` walks straight into that twice::

        persistent = set(self.state_dict().keys())            # -> buf.detach()
        ...
        parent.register_buffer(attr, torch.empty_like(buf, device="meta"), ...)

    so the fix has to be generic, not a special case for ``empty_like``.
    ``__torch_function__`` post-processes every result: any returned instance of
    this class that is missing the metadata inherits it from the first operand
    that has it. The dispose'd meta buffer therefore keeps reporting the correct
    float shape, and the rebuild's ``load_state_dict`` shape check passes.

    Cost: ``__torch_function__`` fires for ops that touch this subclass. The hot
    path does not -- ``ggml_linear_forward`` drops to a plain tensor with
    ``as_subclass(torch.Tensor)`` before it does any arithmetic -- so what is
    left is the handful of lifecycle ops per tensor per build.

    ``torch.empty`` is covered by the same generic path; with no quantised
    operand there is nothing to propagate, which is correct (a freshly allocated
    tensor is not somebody else's quantised weight).
    """

    _QUANT_ATTRS = ("_ggml_type", "_float_shape")

    @classmethod
    def __torch_function__(cls, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        kwargs = {} if kwargs is None else kwargs
        result = super().__torch_function__(func, types, args, kwargs)
        # Explicit None tests, never truthiness: `a or b` on tensors calls
        # `bool()` and raises "Boolean value of Tensor ... is ambiguous".
        source = cls._find_quant_source(args)
        if source is None:
            source = cls._find_quant_source(tuple(kwargs.values()))
        if source is not None:
            cls._propagate(result, source)
        return result

    @classmethod
    def _has_quant_attrs(cls, obj: Any) -> bool:
        return isinstance(obj, GGMLQuantizedTensor) and all(
            hasattr(obj, name) for name in cls._QUANT_ATTRS
        )

    @classmethod
    def _find_quant_source(cls, obj: Any, _depth: int = 0) -> "GGMLQuantizedTensor | None":
        """First operand carrying usable quantisation metadata (depth-limited)."""
        if cls._has_quant_attrs(obj):
            return obj
        if _depth >= 2:
            return None
        if isinstance(obj, (tuple, list)):
            for item in obj:
                found = cls._find_quant_source(item, _depth + 1)
                if found is not None:
                    return found
        return None

    @classmethod
    def _propagate(cls, obj: Any, source: "GGMLQuantizedTensor", _depth: int = 0) -> None:
        """Re-attach metadata to any subclass instance in *obj* that lacks it."""
        if isinstance(obj, GGMLQuantizedTensor):
            for name in cls._QUANT_ATTRS:
                if not hasattr(obj, name):
                    setattr(obj, name, getattr(source, name))
            return
        if _depth >= 2:
            return
        if isinstance(obj, (tuple, list)):
            for item in obj:
                cls._propagate(item, source, _depth + 1)


# ---------------------------------------------------------------------------
# 2. GGUF state-dict loader
# ---------------------------------------------------------------------------


def _gguf_reader(gguf_path: str):
    import gguf as gguf_lib

    return gguf_lib.GGUFReader(gguf_path, mode="r")


def read_gguf_metadata(gguf_path: str) -> dict:
    """Rebuild the safetensors-style metadata dict from the GGUF KV block.

    Post-parse form, matching ``SafetensorsModelStateDictLoader.metadata``: each
    value is ``json.loads``-ed when it parses and left as a raw string when it
    does not. So ``config`` and ``gemma_source_checkpoint`` come back as dicts,
    ``model_version`` and ``license`` as strings -- which is what the official
    consumers expect (``_check_gemma_version`` subscripts
    ``gemma_source_checkpoint``; a JSON string there would raise).

    ``GGUF.*`` (reader-synthesised structure counts) and ``general.*`` (GGUF's
    own conventions: architecture / file_type / quantization_version) are
    excluded -- they have no counterpart in the official header and would change
    the model-shell registry key for no reason.
    """
    reader = _gguf_reader(gguf_path)
    try:
        metadata: dict[str, Any] = {}
        for name, field in reader.fields.items():
            if name.startswith("GGUF.") or name.startswith("general."):
                continue
            value = field.contents()
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            metadata[name] = value
        if "config" not in metadata:
            raise Ltx25BuildError(
                f"{Path(gguf_path).name}: no 'config' key in the GGUF metadata; this file was not "
                f"produced by the Nz GGUF converter's LTX-2.5 transformer path."
            )
        return metadata
    finally:
        del reader


def read_gguf_tensor_names(gguf_path: str) -> list[str]:
    reader = _gguf_reader(gguf_path)
    try:
        return [tensor.name for tensor in reader.tensors]
    finally:
        del reader


def build_transformer_sd_ops(gguf_path: str) -> SDOps:
    """All-pass SDOps that drops the EmbeddingsProcessor tensors.

    ``with_matching()`` with no prefix/suffix matches every key (the keys are
    already stripped, so there is nothing to rename), and ``allowed_keys``
    carries the surviving set. Filtering here rather than after the load keeps
    the 258 connector tensors out of the cached state dict entirely.

    Also note ``model_sd_ops`` may not be ``None``: ``_chain_quantization``
    dereferences ``sd_ops.name`` whenever a policy contributes its own sd_ops,
    and an all-pass named SDOps is the cheapest thing that keeps that safe.
    """
    names = read_gguf_tensor_names(gguf_path)
    allowed = frozenset(name for name in names if not name.startswith(_EMBEDDINGS_PREFIXES))
    dropped = len(names) - len(allowed)
    logger.info(
        "LTX25 transformer SDOps: %d tensors total, %d dropped as EmbeddingsProcessor, %d kept",
        len(names), dropped, len(allowed),
    )
    return SDOps(LTX25_GGUF_PASSTHRU).with_matching().with_additional_allowed_keys(allowed)


class Ltx25GgufStateDictLoader:
    """``StateDictLoader`` reading the LTX 2.5 transformer GGUF.

    Quantised tensors become :class:`Ltx25GGMLTensor` wrapping the packed bytes;
    float tensors keep their GGUF dtype exactly (F32 stays F32). No cast: the
    official non-streaming build path calls ``build()`` without ``dtype=`` too,
    so the checkpoint's own dtypes are what the reference run uses.
    """

    def __init__(self, gguf_path: str) -> None:
        self.gguf_path = str(gguf_path)

    def metadata(self, path: str | None = None) -> dict:
        return read_gguf_metadata(path or self.gguf_path)

    def load(
        self,
        path: str | list[str],
        sd_ops: SDOps | None = None,
        device: torch.device | None = None,
    ) -> StateDict:
        import numpy as np

        target = device or _CPU
        started = time.perf_counter()
        reader = _gguf_reader(self.gguf_path)

        state_dict: dict[str, torch.Tensor] = {}
        dtypes: set[torch.dtype] = set()
        raw_bytes = 0
        n_float = 0
        n_quant = 0
        n_dropped = 0

        for tensor in reader.tensors:
            key = tensor.name if sd_ops is None else sd_ops.apply_to_key(tensor.name)
            if key is None:
                n_dropped += 1
                continue

            ggml_type = tensor.tensor_type.value
            # GGUF stores dims fastest-varying first; torch wants the reverse.
            float_shape = tuple(reversed(tensor.shape.tolist()))

            # Owned copy off the GGUF memmap. copy=True is load-bearing: an
            # aliased np.memmap outlives the reader and has bitten ComfyUI-GGUF
            # (#444 family) with silently corrupted weights.
            flat = torch.from_numpy(np.array(tensor.data, copy=True)).reshape(-1)
            raw_bytes += flat.numel()

            float_dtype = _GGML_FLOAT_DTYPES.get(ggml_type)
            if float_dtype is not None:
                value: torch.Tensor = flat.view(float_dtype).view(float_shape)
                if target.type != "cpu":
                    value = value.to(target, non_blocking=True)
                dtypes.add(float_dtype)
                n_float += 1
            else:
                value = Ltx25GGMLTensor(flat, ggml_type, float_shape)
                if target.type != "cpu":
                    value = value.to(target, non_blocking=True)
                dtypes.add(torch.uint8)
                n_quant += 1

            pairs = ((key, value),) if sd_ops is None else sd_ops.apply_to_key_value(key, value)
            for out_key, out_value in pairs:
                state_dict[out_key] = out_value

        del reader
        logger.info(
            "GGUF load %s -> %s: %d float + %d quantised tensors (%d dropped by sd_ops), "
            "%.2f GiB raw, %.1fs",
            Path(self.gguf_path).name, target, n_float, n_quant, n_dropped,
            raw_bytes / 2**30, time.perf_counter() - started,
        )
        return StateDict(sd=state_dict, device=target, size=raw_bytes, dtype=dtypes)


# ---------------------------------------------------------------------------
# 3. Quantization policy
# ---------------------------------------------------------------------------


def build_quantization_policy() -> QuantizationPolicy:
    """Policy that installs 2.3's per-layer dequant module op.

    ``sd_ops`` MUST stay ``None``. ``_chain_quantization`` merges a policy's
    sd_ops with the builder's by constructing ``SDOps(name=..., mapping=...)``
    -- it copies neither ``allowed_keys`` nor anything else, so contributing
    sd_ops here would silently re-admit the 258 EmbeddingsProcessor tensors.

    ``model_configurator`` is pinned so the policy, not the caller, decides the
    transformer class -- matching how the official fp8/nvfp4 policies behave.
    """
    ggml_op = ModuleOps(
        name="ltx25_ggml_per_layer_dequant",
        matcher=lambda model: isinstance(model, LTXModel),
        mutator=_patch_model_for_ggml_dequant,
    )
    return QuantizationPolicy(
        sd_ops=None,
        module_ops=(ggml_op,),
        model_configurator=LTXModelConfigurator,
        fuse_rule=bf16_fuse_rule,
    )


# ---------------------------------------------------------------------------
# 4. CPU builder
# ---------------------------------------------------------------------------


class Ltx25CpuModelBuilder(SingleGPUModelBuilder):
    """``SingleGPUModelBuilder`` that materialises on CPU and never casts dtype.

    Two deliberate departures from ``SingleGPUModelBuilder.build``:

    * The requested ``device`` is used for nothing. Weights land on CPU and the
      final ``meta_model.to(device)`` is skipped, because it would move all
      14.7GB onto a 16GB card in one go. Placement is the stage's job (see
      :meth:`Ltx25DiffusionStage._place_transformer`).
    * Meta residue raises instead of warning-and-returning the half-built model.
      Upstream's warning is survivable for a float checkpoint; here a key that
      failed to land leaves a meta *buffer* where a quantised weight should be,
      and the per-layer dequant forward would read garbage rather than crash.
    """

    def build(
        self,
        device: torch.device | None = None,  # noqa: ARG002 -- placement is the stage's job
        dtype: torch.dtype | None = None,  # noqa: ARG002 -- GGUF dtypes are authoritative
        **kwargs: object,  # noqa: ARG002 -- video_tools etc. from DiffusionStage.__call__
    ) -> Any:
        metadata = self.model_metadata()
        meta_model = self.meta_model(metadata, self.module_ops)

        _load_model_weights(
            meta_model=meta_model,
            model_path=self.model_path,
            loras=self.loras,
            loader=self.model_loader,
            registry=self.registry,
            device=_CPU,
            dtype=None,
            model_sd_ops=self.model_sd_ops,
            lora_load_device=self.lora_load_device,
            fuse_rule=self.fuse_rule,
        )

        uninitialized = _check_uninitialized(meta_model)
        if uninitialized:
            head = uninitialized[:12]
            raise Ltx25BuildError(
                f"{len(uninitialized)} parameter(s)/buffer(s) were left on the meta device after "
                f"loading {self.model_path}: {head}"
                f"{' ...' if len(uninitialized) > len(head) else ''}. The GGUF is missing keys the "
                f"model expects (or the sd_ops filter dropped too much)."
            )
        return meta_model


# ---------------------------------------------------------------------------
# 5. Stage
# ---------------------------------------------------------------------------


def transformer_blocks_of(model: torch.nn.Module) -> list[torch.nn.Module]:
    """The 48 ``BasicAVTransformerBlock``s, through the ``X0Model`` wrapper."""
    inner = getattr(model, "velocity_model", model)
    blocks = getattr(inner, "transformer_blocks", None)
    if blocks is None or len(blocks) == 0:
        raise Ltx25BuildError(
            f"{type(model).__name__} has no transformer_blocks; block swap cannot be installed."
        )
    return list(blocks)


def _move_module_tree(root: torch.nn.Module, device: torch.device, skip: set[int]) -> int:
    """Move every parameter/buffer of *root* to *device*, skipping module ids in *skip*.

    Written by hand rather than as ``root.to(device)`` because the whole point
    is to leave one subtree (``transformer_blocks``) behind on CPU. Buffers go
    through their own ``.to()`` so ``Ltx25GGMLTensor`` keeps its identity and
    metadata; parameters are rebound as fresh ``nn.Parameter``s, which is what
    ``Module._apply`` does for a cross-device move anyway -- and it leaves the
    registry's cached CPU tensors untouched.
    """
    moved = 0
    for module in root.modules():
        if id(module) in skip:
            continue
        for name, param in list(module._parameters.items()):
            if param is None or param.device == device:
                continue
            if param.is_meta:
                raise Ltx25BuildError(f"parameter {name!r} of {type(module).__name__} is still on meta")
            module._parameters[name] = torch.nn.Parameter(
                param.data.to(device), requires_grad=param.requires_grad
            )
            moved += 1
        for name, buf in list(module._buffers.items()):
            if buf is None or buf.device == device:
                continue
            if buf.is_meta:
                raise Ltx25BuildError(f"buffer {name!r} of {type(module).__name__} is still on meta")
            module._buffers[name] = buf.to(device)
            moved += 1
    return moved


class Ltx25DiffusionStage(DiffusionStage):
    """``DiffusionStage`` that builds from a GGUF and places weights itself.

    Overrides exactly one method, :meth:`_build_transformer`, so everything else
    -- the denoising loop, conditioning checks, ``with_attention`` /
    ``with_loras``, the ``gpu_model`` dispose-on-exit contract -- is the
    official code path unchanged.

    Block swap is installed once per model shell and is idempotent: with
    ``cache_models=True`` the registry hands back the SAME ``LTXModel``
    instance on every build, so the second build finds the blocks already
    patched and leaves them alone. Re-wrapping would stack two swap windows on
    one block and drive twice the traffic.
    """

    def __init__(
        self,
        transformer_builder: Any,
        dtype: torch.dtype,
        device: torch.device,
        *,
        blocks_on_gpu: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(transformer_builder, dtype, device, **kwargs)
        self.blocks_on_gpu = int(blocks_on_gpu)
        self._swap_service: BlockSwapService | None = (
            BlockSwapService(blocks_on_gpu=self.blocks_on_gpu, device=device)
            if self.blocks_on_gpu > 0
            else None
        )

    # -- construction --------------------------------------------------------

    @classmethod
    def from_gguf(
        cls,
        gguf_path: str,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        blocks_on_gpu: int = 8,
        cache_weights: bool = True,
        registry: Registry | None = None,
        **kwargs: Any,
    ) -> "Ltx25DiffusionStage":
        """Assemble the stage from a transformer GGUF.

        ``cache_weights=True`` is the default on purpose (plan H revision): the
        pipeline disposes the transformer between stage 1 and stage 2, so
        without a retained state dict every job pays a second 14.7GB disk read
        mid-generation. The price is that much resident RAM; pass
        ``cache_weights=False`` on a RAM-tight machine and take the re-read.
        """
        gguf_path = str(gguf_path)
        if not Path(gguf_path).exists():
            raise FileNotFoundError(f"transformer GGUF not found: {gguf_path}")

        registry = registry or ModelRegistry(cache_models=True, cache_weights=cache_weights)
        quantization = build_quantization_policy()
        builder = Ltx25CpuModelBuilder(
            model_class_configurator=LTXModelConfigurator,
            model_path=gguf_path,
            model_sd_ops=build_transformer_sd_ops(gguf_path),
            model_loader=Ltx25GgufStateDictLoader(gguf_path),
            registry=registry,
        )
        logger.info(
            "Ltx25DiffusionStage.from_gguf(%s): device=%s dtype=%s blocks_on_gpu=%d cache_weights=%s",
            Path(gguf_path).name, device, dtype, blocks_on_gpu, cache_weights,
        )
        return cls(
            builder,
            dtype,
            device,
            blocks_on_gpu=blocks_on_gpu,
            quantization=quantization,
            **kwargs,
        )

    # -- build ---------------------------------------------------------------

    def _build_transformer(self, *, device: torch.device | None = None, **kwargs: object) -> X0Model:
        """Official override point. Same contract, without the two ``.to(device)`` calls.

        Upstream is ``X0Model(builder.build(device=target, **kwargs)).to(target)``:
        the builder puts the whole checkpoint on the GPU and the stage then moves
        it there again. Both are fatal at 14.7GB on 16GB, so this builds on CPU
        and places selectively.
        """
        target = device or self._device
        started = time.perf_counter()
        velocity_model = self._prepared_builder().build(device=target, **kwargs)
        model = X0Model(velocity_model).eval()
        self._place_transformer(model, target)
        logger.info("Transformer ready on %s in %.1fs", target, time.perf_counter() - started)
        return model

    def _place_transformer(self, model: X0Model, device: torch.device) -> None:
        """Put the model where it has to be: blocks streamed, everything else resident."""
        blocks = transformer_blocks_of(model)
        total = len(blocks)

        if self._swap_service is None or self.blocks_on_gpu >= total:
            if self._swap_service is not None:
                logger.info(
                    "BlockSwap not needed (blocks_on_gpu=%d >= %d blocks) -- full residency",
                    self.blocks_on_gpu, total,
                )
            _move_module_tree(model, device, skip=set())
            return

        skip = {id(module) for block in blocks for module in block.modules()}
        moved = _move_module_tree(model, device, skip=skip)
        logger.info(
            "Moved %d non-block tensors to %s; %d blocks stay on CPU for the swap window",
            moved, device, total,
        )
        self.ensure_block_swap_installed(model)

    # -- block swap ----------------------------------------------------------

    def ensure_block_swap_installed(self, model: torch.nn.Module) -> bool:
        """Install the swap hooks unless they are already there. Returns True if installed.

        Idempotency is keyed on ``_BLOCK_SWAP_ATTR``, the marker the service
        itself sets. It matters because the shell registry reuses one
        ``LTXModel`` across builds: the synchronous ``_patch_block`` captures
        ``block.forward`` as it finds it, so a second install would wrap the
        first wrapper and every block would be paged in twice per step.
        """
        if self._swap_service is None:
            return False
        blocks = transformer_blocks_of(model)
        marked = [block for block in blocks if hasattr(block, _BLOCK_SWAP_ATTR)]
        if marked:
            if len(marked) != len(blocks):
                raise Ltx25BuildError(
                    f"block swap is half-installed: {len(marked)}/{len(blocks)} blocks carry "
                    f"{_BLOCK_SWAP_ATTR}. Uninstall before rebuilding."
                )
            logger.info("BlockSwap already installed on %d blocks -- skipped (idempotent)", len(blocks))
            return False
        self._swap_service.install(model)
        return True

    def uninstall_block_swap(self, model: torch.nn.Module) -> int:
        """Restore the original forwards and leave the blocks on CPU. Returns blocks cleaned.

        Deliberately NOT ``BlockSwapService.uninstall``: that one ends with
        ``block.to(self.device)`` for all 48 blocks, i.e. exactly the 14.7GB
        GPU residency this engine exists to avoid. Everything else it does --
        restore ``forward``, drop the marker, release the prefetch pool -- is
        reproduced here.
        """
        cleaned = 0
        for block in transformer_blocks_of(model):
            original = getattr(block, _BLOCK_SWAP_ATTR, None)
            if original is None:
                continue
            block.forward = original  # type: ignore[method-assign]
            delattr(block, _BLOCK_SWAP_ATTR)
            block.to(_CPU)
            cleaned += 1
        if self._swap_service is not None:
            self._swap_service.teardown_prefetch()
        logger.info("BlockSwap uninstalled from %d blocks (left on CPU)", cleaned)
        return cleaned


# ---------------------------------------------------------------------------
# Selftest (gate G2)
# ---------------------------------------------------------------------------


def _rss_bytes() -> int | None:
    """Process working set, best effort (no psutil in this venv)."""
    try:
        import ctypes
        import ctypes.wintypes as wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.windll.kernel32
        # restype MUST be c_void_p: GetCurrentProcess returns the pseudo-handle
        # (HANDLE)-1, and ctypes' default int restype truncates it to 32 bits,
        # after which GetProcessMemoryInfo just fails and reports zeroes.
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD]
        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize)
    except Exception:  # pragma: no cover -- diagnostics only
        return None


class _Vram:
    """Per-phase CUDA peak recorder."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.phases: dict[str, dict[str, float]] = {}

    def reset(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def record(self, phase: str, seconds: float) -> None:
        entry = {"seconds": round(seconds, 2), "rss_gib": None}
        rss = _rss_bytes()
        if rss is not None:
            entry["rss_gib"] = round(rss / 2**30, 2)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            entry.update(
                peak_allocated_gib=round(torch.cuda.max_memory_allocated(self.device) / 2**30, 3),
                peak_reserved_gib=round(torch.cuda.max_memory_reserved(self.device) / 2**30, 3),
                current_allocated_gib=round(torch.cuda.memory_allocated(self.device) / 2**30, 3),
            )
        self.phases[phase] = entry
        logger.info("PHASE %-24s %s", phase, entry)


def _dummy_video_modality(
    config: dict,
    *,
    device: torch.device,
    dtype: torch.dtype,
    width: int,
    height: int,
    frames: int,
    seed: int = 0,
) -> Modality:
    """One deterministic video-only ``Modality``, sized from the checkpoint config.

    Video-only is legitimate for an audio-video model: ``LTXModel.forward``
    only rejects a modality the model was not built for, and every block's
    audio and A/V-cross branches are guarded on ``audio is not None``.
    """
    transformer = config["transformer"]
    in_channels = int(transformer.get("in_channels", 128))
    cross_dim = int(transformer.get("cross_attention_dim", 4096))
    scale_t, scale_hw = 8, 32

    latent_frames = (frames - 1) // scale_t + 1
    latent_h, latent_w = height // scale_hw, width // scale_hw
    tokens = latent_frames * latent_h * latent_w
    if tokens == 0:
        raise ValueError(f"{width}x{height}x{frames} yields zero latent tokens")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn(1, tokens, in_channels, generator=generator, dtype=torch.float32)
    context = torch.randn(1, 128, cross_dim, generator=generator, dtype=torch.float32)

    # (B, 3, T, 2): [start, end) bounds per patch on (time, height, width).
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(latent_frames),
            torch.arange(latent_h),
            torch.arange(latent_w),
            indexing="ij",
        ),
        dim=0,
    ).reshape(3, tokens)
    positions = torch.stack([grid, grid + 1], dim=-1).unsqueeze(0).to(torch.float32)

    # timesteps is (B, T, 1), not (B, T): it is built upstream as
    # ``denoise_mask * sigma`` and ``to_denoised`` broadcasts it against the
    # (B, T, D) latent. A (B, T) tensor would fail to broadcast.
    return Modality(
        latent=latent.to(device=device, dtype=dtype),
        sigma=torch.full((1,), 1.0, device=device, dtype=torch.float32),
        timesteps=torch.full((1, tokens, 1), 1.0, device=device, dtype=torch.float32),
        positions=positions.to(device=device),
        context=context.to(device=device, dtype=dtype),
        context_mask=torch.ones(1, 128, device=device, dtype=torch.int64),
    )


def _selftest(  # noqa: PLR0913, PLR0915
    gguf_path: str,
    *,
    blocks_on_gpu: int,
    width: int,
    height: int,
    frames: int,
    rounds: int,
    cache_weights: bool,
) -> dict:
    from engine25 import ltxcore_compat

    ltxcore_compat.verify()

    # Indexed device on purpose: `torch.device("cuda") != torch.device("cuda:0")`,
    # and the placement helper compares devices to decide what still needs moving.
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    dtype = torch.bfloat16
    vram = _Vram(device)
    report: dict[str, Any] = {
        "gguf": gguf_path,
        "device": str(device),
        "blocks_on_gpu": blocks_on_gpu,
        "cache_weights": cache_weights,
        "shape": {"width": width, "height": height, "frames": frames},
        "rounds": [],
        "phases": vram.phases,
        "checks": {},
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        report["gpu"] = {"name": props.name, "total_gib": round(props.total_memory / 2**30, 2)}

    vram.reset()
    started = time.perf_counter()
    stage = Ltx25DiffusionStage.from_gguf(
        gguf_path,
        device=device,
        dtype=dtype,
        blocks_on_gpu=blocks_on_gpu,
        cache_weights=cache_weights,
    )
    config = read_gguf_metadata(gguf_path)["config"]
    vram.record("00_stage_constructed", time.perf_counter() - started)

    modality = _dummy_video_modality(
        config, device=device, dtype=dtype, width=width, height=height, frames=frames
    )
    report["tokens"] = int(modality.latent.shape[1])

    for index in range(1, rounds + 1):
        round_report: dict[str, Any] = {"round": index}

        vram.reset()
        started = time.perf_counter()
        transformer = stage._build_transformer()
        build_seconds = time.perf_counter() - started
        vram.record(f"{index:02d}a_build", build_seconds)
        round_report["build_seconds"] = round(build_seconds, 2)
        round_report["num_blocks"] = int(transformer.num_blocks)

        # Idempotency + uninstall are checked on the LAST round only: they need a
        # live model, and the uninstall leaves the blocks unusable for a forward.
        if index == rounds:
            blocks = transformer_blocks_of(transformer)
            before = [id(block.forward) for block in blocks]
            installed_again = stage.ensure_block_swap_installed(transformer)
            after = [id(block.forward) for block in blocks]
            report["checks"]["swap_second_install_was_noop"] = (not installed_again) and before == after
            report["checks"]["swap_marked_blocks"] = sum(
                1 for block in blocks if hasattr(block, _BLOCK_SWAP_ATTR)
            )

        vram.reset()
        started = time.perf_counter()
        with torch.no_grad():
            video_out, audio_out = transformer(modality, None, None)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - started
        vram.record(f"{index:02d}b_forward", forward_seconds)
        round_report.update(
            forward_seconds=round(forward_seconds, 2),
            output_shape=list(video_out.shape),
            output_dtype=str(video_out.dtype),
            finite=bool(torch.isfinite(video_out.float()).all().item()),
            abs_mean=float(video_out.float().abs().mean().item()),
            audio_out_is_none=audio_out is None,
        )

        if index == rounds and stage._swap_service is not None:
            blocks = transformer_blocks_of(transformer)
            originals = [getattr(block, _BLOCK_SWAP_ATTR, None) for block in blocks]
            cleaned = stage.uninstall_block_swap(transformer)
            report["checks"]["swap_uninstalled_blocks"] = cleaned
            report["checks"]["swap_markers_left"] = sum(
                1 for block in blocks if hasattr(block, _BLOCK_SWAP_ATTR)
            )
            report["checks"]["swap_forwards_restored"] = all(
                original is None or block.forward is original
                for block, original in zip(blocks, originals)
            )
            report["checks"]["swap_blocks_on_cpu_after_uninstall"] = all(
                all(param.device.type == "cpu" for param in block.parameters())
                for block in blocks
            )

        vram.reset()
        started = time.perf_counter()
        transformer.dispose()
        del transformer, video_out, audio_out
        gc.collect()
        cleanup_memory()
        vram.record(f"{index:02d}c_dispose", time.perf_counter() - started)
        report["rounds"].append(round_report)

    builds = [entry["build_seconds"] for entry in report["rounds"]]
    report["checks"]["all_rounds_finite"] = all(entry["finite"] for entry in report["rounds"])
    report["checks"]["rebuild_faster_than_first_build"] = (
        len(builds) < 2 or max(builds[1:]) < builds[0]
    )
    if device.type == "cuda":
        peaks = [
            entry.get("peak_allocated_gib", 0.0)
            for entry in vram.phases.values()
        ]
        report["peak_allocated_gib"] = max(peaks)
        report["fits_in_16gib"] = report["peak_allocated_gib"] < 16.0
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="engine25.gguf_transformer")
    parser.add_argument("--selftest", metavar="GGUF", required=True, help="transformer GGUF to build from")
    parser.add_argument("--blocks-on-gpu", type=int, default=8)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--rounds", type=int, default=3, help="build/forward/dispose cycles")
    parser.add_argument("--no-cache-weights", action="store_true")
    parser.add_argument("--json-out", default=None, help="also write the report to this path")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="[ltx25_gguf] %(asctime)s %(name)s: %(message)s",
    )

    report = _selftest(
        args.selftest,
        blocks_on_gpu=args.blocks_on_gpu,
        width=args.width,
        height=args.height,
        frames=args.frames,
        rounds=args.rounds,
        cache_weights=not args.no_cache_weights,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")

    failed = [name for name, ok in report["checks"].items() if ok is False]
    if failed:
        print(f"SELFTEST FAILED: {failed}", file=sys.stderr)
        return 1
    print("SELFTEST OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    # `python engine25/gguf_transformer.py` as well as `python -m engine25...`.
    if __package__ in (None, ""):  # pragma: no cover
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
