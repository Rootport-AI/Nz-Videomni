"""LTX 2.5 text encoder: Gemma 4 unified from GGUF, with layer offload (§3-98 Phase 2c).

The 2.5 text encoder is a 12B Gemma 4 "unified" model plus the two aggregate
projections that turn its 49 hidden-state layers into the transformer's video and
audio conditioning. In bf16 that file is 26.3 GB; the converter's Q4_K_M GGUF is
9.2 GB. This module is what makes the GGUF usable by the *unmodified* official
pipeline.

Four parts, in the order they matter:

1. Assets, not weights (F2) -- :mod:`engine25.assets_export`
   ``PromptEncoder`` reads the tokenizer / processor / HF config from
   ``text_encoder_path`` through ``GemmaAssets.load``, which only accepts a
   directory or a ``.safetensors``. So engine25 exports a ~31 MB assets-only
   ``.safetensors`` from the GGUF's own sidecars and points the official code at
   that. Nothing in ``ltx_core`` is patched to make the assets work.

2. Key mapping (E, verified against the real model)
   The converter wrote the TE in ComfyUI's flattened layout (``model.layers.*``,
   ``vision_model.*``, ``multi_modal_projector.*``, ``audio_projector.*``). The
   official probe that picks between HF and Comfy layouts,
   ``_gemma4_unified_is_comfy_flat``, inspects the *weight* file -- which for us
   is the assets-only file with no weights in it, so it would answer "HF" and
   every key would miss. engine25 therefore names the layout directly:
   ``_build_gemma4_unified_llm_key_ops(comfy_flat=True)``. Measured on the real
   686-tensor GGUF against a meta-device build of the real model: 678 mapped keys
   vs 678 state-dict keys, zero missing, zero extra (the 678th is ``lm_head``,
   emitted by the official tied-embedding rule).

3. Per-layer dequantisation + layer offload
   Weights stay Q4_K in memory and each ``nn.Linear`` dequantises its own weight
   inside ``forward`` -- the 2.3 engine's ``engine.gguf.quant_service`` patch,
   reused verbatim. On top of that, :class:`Ltx25GemmaLayerOffloader` keeps the
   embedding table and the small towers resident on the GPU and streams the 48
   decoder layers through it, so TE peak VRAM is a tunable knob rather than "all
   of it".

   ``engine.gemma.gguf_quant_service`` (2.3's *Gemma 3* loader) is deliberately
   NOT used and must never be imported here. It folds the ``1 + w`` term of
   Gemma's RMSNorm into the stored weights, which is correct for the checkpoint
   layout 2.3 consumes and *wrong* for this one: the 2.5 converter writes the
   raw HF weights (proven by byte-level key/shape parity with the official bf16
   file), and transformers' own RMSNorm applies ``1 + w`` at runtime. Folding
   twice would silently shift every normalisation in the model.

4. The aggregate projections (L)
   ``FeatureExtractorV2`` holds two enormous ``nn.Linear``s -- 4096x188160 and
   2048x188160, i.e. 1.54 GiB and 0.77 GiB *dequantised*. Naive per-layer dequant
   materialises the whole weight, and the eager Q6_K kernel needs roughly a dozen
   bytes of int32/float32 scratch per output element while it does so, which is
   ~10 GiB of transient for the video projection alone. So these two get a
   chunked forward: dequantise a slice of output rows, matmul, discard, move on.
   Measured on a 16 GiB card: 24.2 GiB peak and 11.9 s unchunked (it survives only
   because WDDM spills to host memory) against 1.8 GiB of transient and 1.7 s
   chunked. Only Linears above the threshold are chunked; everything else keeps
   the plain 2.3 path.

   Chunking is *mathematically* identical -- no reduction ever crosses a chunk
   boundary -- but it is NOT bit-identical to the unchunked matmul, because
   cuBLAS picks a different GEMM for a 170-column output than for a 4096-column
   one and the bf16 accumulation order goes with it. Measured: the Gemma hidden
   states match to the bit across every configuration, the projected embeddings
   do not. So :data:`AGGREGATE_CHUNK_ELEMENTS` is part of the reproducibility
   contract -- a fixed constant, not a per-machine tuning knob. Runs at one chunk
   size are bit-identical to each other, across both re-encode and full
   dispose/rebuild cycles.

Notes for Phase 2d
------------------
* :func:`build_text_encoder_builder` returns something ``PromptEncoder`` accepts
  as ``text_encoder_builder=``; it builds on CPU and places itself, because
  ``gpu_model`` only disposes, it never moves anything to the GPU.
* ``model_config()`` on that builder reports ``model_type="gemma4_unified"``
  straight out of the GGUF KV, which is what ``PromptEncoder`` checks first.
* The ``ProcessorLoad`` module op is **off by default**. Building a
  ``Gemma4UnifiedProcessor`` imports ``torchvision``, which this venv does not
  have; the processor is only used by prompt *enhancement*, which is out of v1
  scope. Pass ``include_processor=True`` (and install torchvision) if enhance is
  ever brought in.

Selftest (gate G3)::

    python -m engine25.gguf_gemma4 --selftest <te.gguf> \
        --transformer-gguf <transformer.gguf> --official-te <official bf16 .safetensors>

encodes three prompts (short / long / Japanese-mixed), repeats one to prove
bit-identical output, reconciles the GGUF key set against a meta-device build of
the real model, asserts the 48-layer / 5-sliding-1-full attention pattern, and
reports per-phase VRAM and RSS peaks as JSON.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch

# --- 2.3 engine reuse (import only; those modules are never edited) ----------
# NOTE: `engine.gemma.gguf_quant_service` is the Gemma *3* loader and folds
# RMSNorm's `1 + w`. It must not appear in this file -- see part 4 above.
from engine.gguf.quant_service import (
    GGMLQuantizedTensor,
    _patch_linear_for_ggml_dequant,
    _patch_model_for_ggml_dequant,
    dequantize_ggml_tensor,
)

# --- engine25 siblings -------------------------------------------------------
# `_move_module_tree` and the two selftest helpers are module-private to
# gguf_transformer, not package-private: reusing them here is the alternative to
# a second copy of "walk the tree and move exactly these tensors", which is
# precisely the kind of duplicate that drifts.
from engine25 import assets_export
from engine25.gguf_transformer import (
    LTX25_EMBEDDINGS_PROCESSOR_KEY_OPS,
    Ltx25CpuModelBuilder,
    Ltx25GgufStateDictLoader,
    _move_module_tree,
    _rss_bytes,
    _Vram,
    read_gguf_metadata,
    read_gguf_tensor_names,
)

# --- official surface: ONE entry point (see engine25/ltxcore_compat.py) ------
from engine25.ltxcore_compat import (
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoderConfigurator,
    ModelRegistry,
    ModuleOps,
    Registry,
    SDOps,
    StateDict,
    _build_gemma4_unified_llm_key_ops,
    cleanup_memory,
    create_meta_model,
    get_gemma_ops,
)

logger = logging.getLogger(__name__)

_CPU = torch.device("cpu")

#: Name of the official ``ModuleOps`` that builds the HF processor. Skipped by
#: default (torchvision; enhance is out of v1 scope) -- see the module docstring.
PROCESSOR_MODULE_OP = "ProcessorLoad"

#: Dotted path from ``LTXGemmaTextEncoder`` down to the decoder-layer list. Also
#: the state-dict prefix, which is why the official ``StreamingModelBuilder``
#: spells the same string twice.
LAYERS_ATTR_PATH = ("model", "model", "language_model", "layers")

#: Gemma 4 unified interleaves five sliding-window layers with one full-attention
#: layer. Asserted rather than assumed: getting this wrong produces plausible but
#: wrong conditioning instead of a crash.
FULL_ATTENTION_PERIOD = 6

#: Output elements per chunk of a chunked aggregate projection. ~32M elements is
#: a few hundred MB of dequant scratch -- small enough to be invisible next to the
#: activations, large enough that the chunk loop is not the bottleneck.
#:
#: Treat this as a pinned constant, not a tuning knob: the chunk size selects the
#: cuBLAS GEMM shape, so changing it changes the projected embeddings in the last
#: bits and breaks same-seed reproducibility against earlier runs.
AGGREGATE_CHUNK_ELEMENTS = 32_000_000

#: Weights bigger than this (in elements) get the chunked forward. Sized so the
#: two aggregate projections (770M / 385M elements) qualify and nothing in the
#: Gemma backbone or the connectors does.
CHUNKED_FORWARD_THRESHOLD = 64_000_000

#: Marker for an installed layer-stream wrapper; holds the original ``forward``.
_OFFLOAD_ATTR = "_ltx25_te_offload_original_forward"


class Ltx25GemmaError(RuntimeError):
    """The text encoder could not be built or placed correctly."""


# ---------------------------------------------------------------------------
# 1. State-dict loading across the two GGUFs
# ---------------------------------------------------------------------------


class Ltx25MultiGgufStateDictLoader:
    """``StateDictLoader`` that merges several GGUFs into one state dict.

    The EmbeddingsProcessor is the only consumer that needs it, and it needs it
    because its weights are genuinely split: the two aggregate projections live
    in the text-encoder GGUF, the 258 connector tensors in the transformer GGUF.
    That is the same split the official code handles with
    ``ModelPaths.embeddings_weight_paths``.

    Metadata comes from the FIRST path, matching ``read_model_metadata``. Order
    therefore matters: the transformer GGUF must come first, because
    ``EmbeddingsProcessorConfigurator`` reads ``config.transformer`` and
    ``gemma_source_checkpoint`` from it.

    Per-file reading is delegated to :class:`Ltx25GgufStateDictLoader` rather than
    reimplemented, so the dtype handling, the ``copy=True`` defence against
    aliasing a closed memmap, and the ``Ltx25GGMLTensor`` wrapping all stay in
    one place.
    """

    def __init__(self, paths: tuple[str, ...] | list[str]) -> None:
        self.paths = tuple(str(path) for path in paths)
        if not self.paths:
            raise Ltx25GemmaError("Ltx25MultiGgufStateDictLoader needs at least one GGUF path")
        self._loaders = tuple(Ltx25GgufStateDictLoader(path) for path in self.paths)

    def metadata(self, path: str | None = None) -> dict:
        return read_gguf_metadata(path or self.paths[0])

    def load(
        self,
        path: str | list[str],
        sd_ops: SDOps | None = None,
        device: torch.device | None = None,
    ) -> StateDict:
        target = device or _CPU
        merged: dict[str, torch.Tensor] = {}
        dtypes: set[torch.dtype] = set()
        size = 0
        for loader in self._loaders:
            part = loader.load(path, sd_ops=sd_ops, device=target)
            clash = sorted(merged.keys() & part.sd.keys())
            if clash:
                raise Ltx25GemmaError(
                    f"{Path(loader.gguf_path).name} redefines {len(clash)} key(s) already supplied by an "
                    f"earlier GGUF ({clash[:5]}). The two files overlap; the sd_ops filter is wrong."
                )
            merged.update(part.sd)
            dtypes |= set(part.dtype)
            size += part.size
        logger.info(
            "Merged %d GGUF(s) into %d tensors (%.2f GiB raw) for %s",
            len(self._loaders), len(merged), size / 2**30, target,
        )
        return StateDict(sd=merged, device=target, size=size, dtype=dtypes)


def build_text_encoder_sd_ops() -> SDOps:
    """The official Gemma 4 unified key map, pinned to the Comfy-flat layout (E).

    ``get_gemma_ops`` would pick the layout by probing the weight file, and the
    weight file it probes is the assets-only export -- which has no weights, so
    the probe answers "HF" and every single key misses. The layout is a property
    of the converter, known at write time, so it is stated rather than guessed.

    This SDOps is also the filter that keeps the five sidecar payloads and the
    four ``text_embedding_projection.*`` tensors out of the Gemma state dict: its
    prefix matchers admit only ``model.`` / ``vision_model.`` /
    ``multi_modal_projector.`` / ``audio_projector.``.
    """
    return _build_gemma4_unified_llm_key_ops(comfy_flat=True)


# ---------------------------------------------------------------------------
# 2. Module ops
# ---------------------------------------------------------------------------


def _ggml_dequant_mutator(model: torch.nn.Module) -> torch.nn.Module:
    """2.3's Linear->buffer + per-layer-dequant patch, with a loud floor."""
    model = _patch_model_for_ggml_dequant(model)
    patched = sum(
        1
        for module in model.modules()
        if isinstance(module, torch.nn.Linear) and "weight" in module._buffers
    )
    if patched == 0:
        raise Ltx25GemmaError(
            f"the GGML dequant patch found no nn.Linear in {type(model).__name__}; quantised weights "
            f"would be loaded into float parameters. The official model structure has changed."
        )
    logger.info("GGML per-layer dequant patched %d Linear layer(s) of %s", patched, type(model).__name__)
    return model


def _install_chunked_forward(module: torch.nn.Linear, chunk_elements: int) -> None:
    """Replace *module*'s dequant forward with a row-chunked one.

    Applied after :func:`_patch_linear_for_ggml_dequant` has already turned the
    weight into a buffer, so this only swaps the forward. Chunking is along the
    output features: ``y[..., a:b] = x @ W[a:b].T``, which needs only the rows of
    the packed weight belonging to that slice, and never reduces across chunks --
    every output element is still one complete dot product.

    That makes it mathematically equivalent but *not* bit-identical to the
    unchunked matmul (cuBLAS chooses its GEMM by shape). It is exactly
    reproducible at a fixed chunk size, which is the property that matters; see
    the module docstring, part 4.

    Falls back to the unchunked forward at call time if the packed bytes do not
    divide evenly by output row (they do whenever ``in_features`` is a multiple of
    the 256-element K-quant block, which holds for both aggregate projections:
    188160 = 735 x 256).
    """
    unchunked = module.forward
    rows_per_chunk = max(1, chunk_elements // max(1, module.in_features))

    def chunked_forward(x: torch.Tensor) -> torch.Tensor:
        weight = module.weight
        if not isinstance(weight, GGMLQuantizedTensor):
            return unchunked(x)
        out_features, in_features = weight._float_shape
        raw = weight.as_subclass(torch.Tensor).view(torch.uint8)
        if raw.numel() % out_features:
            return unchunked(x)
        row_bytes = raw.numel() // out_features

        pieces = []
        for start in range(0, out_features, rows_per_chunk):
            stop = min(start + rows_per_chunk, out_features)
            block = dequantize_ggml_tensor(
                raw[start * row_bytes : stop * row_bytes],
                weight._ggml_type,
                (stop - start, in_features),
                x.dtype,
            )
            pieces.append(torch.nn.functional.linear(x, block, None))
            del block
        result = torch.cat(pieces, dim=-1)
        del pieces
        if module.bias is not None:
            result = result + module.bias
        return result

    module.forward = chunked_forward  # type: ignore[method-assign]
    logger.info(
        "chunked dequant forward on %dx%d Linear (%d rows/chunk, %d chunks)",
        module.out_features, module.in_features, rows_per_chunk,
        -(-module.out_features // rows_per_chunk),
    )


def _embeddings_dequant_mutator(
    model: torch.nn.Module,
    *,
    threshold: int,
    chunk_elements: int,
) -> torch.nn.Module:
    """Per-layer dequant for the EmbeddingsProcessor, chunked where it has to be."""
    model = _ggml_dequant_mutator(model)
    if chunk_elements <= 0:
        logger.warning("chunked aggregate projection DISABLED; the Q6_K dequant will be materialised whole")
        return model
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.out_features * module.in_features > threshold:
            _install_chunked_forward(module, chunk_elements)
    return model


def build_text_encoder_module_ops(
    assets_path: str | Path,
    *,
    include_processor: bool = False,
) -> tuple[ModuleOps, ...]:
    """Official Gemma module ops, then engine25's dequant patch -- in that order.

    Order is the contract: the official ops create the model's non-persistent
    state (rotary inverse frequencies, the embedding scale) and attach the
    tokenizer, and the dequant patch must run last because it rewrites every
    ``nn.Linear.weight`` from parameter to meta buffer -- the shape the GGUF's
    quantised tensors are then assigned into.

    Only ``get_gemma_ops``' *module* ops are taken; its SDOps is discarded in
    favour of :func:`build_text_encoder_sd_ops` (see there for why).
    """
    _discarded_sd_ops, official_ops = get_gemma_ops(str(assets_path))
    kept = tuple(op for op in official_ops if include_processor or op.name != PROCESSOR_MODULE_OP)
    if len(kept) != len(official_ops):
        logger.info(
            "skipping the %s module op: the HF processor needs torchvision and is only used by "
            "prompt enhancement, which is out of the v1 scope",
            PROCESSOR_MODULE_OP,
        )
    if not kept:
        raise Ltx25GemmaError("get_gemma_ops returned no usable module ops for the text encoder")
    ggml_op = ModuleOps(
        name="ltx25_te_ggml_per_layer_dequant",
        # Unconditional on purpose. `create_meta_model` offers module ops the ONE
        # root model the configurator just produced, and this builder pairs the op
        # with GemmaTextEncoderConfigurator itself -- so there is nothing else it
        # could match. An isinstance() guard here could only ever fail *silently*,
        # leaving quantised bytes to be assigned into float parameters.
        matcher=lambda module: True,
        mutator=_ggml_dequant_mutator,
    )
    return (*kept, ggml_op)


def build_embeddings_processor_module_ops(
    *,
    chunk_elements: int = AGGREGATE_CHUNK_ELEMENTS,
    threshold: int = CHUNKED_FORWARD_THRESHOLD,
) -> tuple[ModuleOps, ...]:
    """The dequant patch for the EmbeddingsProcessor, with chunked aggregates (L)."""
    return (
        ModuleOps(
            name="ltx25_embeddings_ggml_per_layer_dequant",
            matcher=lambda module: True,  # see build_text_encoder_module_ops
            mutator=lambda model: _embeddings_dequant_mutator(
                model, threshold=threshold, chunk_elements=chunk_elements
            ),
        ),
    )


# ---------------------------------------------------------------------------
# 3. Layer offload
# ---------------------------------------------------------------------------


def language_model_layers_of(model: torch.nn.Module) -> torch.nn.ModuleList:
    """The 48 Gemma decoder layers, reached through the LTX wrapper."""
    node: Any = model
    for attr in LAYERS_ATTR_PATH:
        node = getattr(node, attr, None)
        if node is None:
            raise Ltx25GemmaError(
                f"{type(model).__name__} has no {'.'.join(LAYERS_ATTR_PATH)}; the official Gemma module "
                f"tree has changed and the layer offload cannot find the decoder layers."
            )
    return node


def assert_layer_pattern(model: torch.nn.Module) -> dict[str, Any]:
    """Assert 48 layers and the 5-sliding / 1-full interleave, and report it.

    ``full_attention`` layers use a wider head dimension (``global_head_dim``
    512 vs 256) and "proportional" RoPE instead of "default", so a shifted
    pattern is a numerically wrong encoder that still runs and still produces
    plausible-looking embeddings. The indices are ``5, 11, ... 47``: every sixth
    layer, counting the full one last.
    """
    layers = language_model_layers_of(model)
    config = model.model.config.text_config
    layer_types = list(config.layer_types)
    declared = int(config.num_hidden_layers)

    if not (len(layers) == len(layer_types) == declared):
        raise Ltx25GemmaError(
            f"layer count disagreement: {len(layers)} modules, {len(layer_types)} layer_types, "
            f"num_hidden_layers={declared}."
        )
    full = [index for index, kind in enumerate(layer_types) if kind == "full_attention"]
    expected = list(range(FULL_ATTENTION_PERIOD - 1, declared, FULL_ATTENTION_PERIOD))
    if full != expected:
        raise Ltx25GemmaError(
            f"full-attention layers are at {full}, expected {expected} (every {FULL_ATTENTION_PERIOD}th "
            f"layer, full last). The checkpoint is not the Gemma 4 unified 12B this engine targets."
        )
    unexpected = sorted(set(layer_types) - {"full_attention", "sliding_attention"})
    if unexpected:
        raise Ltx25GemmaError(f"unknown layer_types {unexpected}")
    return {
        "num_layers": declared,
        "full_attention_indices": full,
        "sliding_attention_layers": len(layer_types) - len(full),
        "pattern": f"{FULL_ATTENTION_PERIOD - 1} sliding : 1 full",
    }


class Ltx25GemmaLayerOffloader:
    """Keeps the embedding table resident and streams the decoder layers.

    Placement, once per build:

    * everything outside the decoder layers goes to the GPU -- the 2.0 GiB bf16
      embedding table, ``model.norm``, the small vision/audio embedders, and the
      non-persistent rotary buffers;
    * ``lm_head`` stays on the CPU. It shares storage with ``embed_tokens`` (the
      official tied-embedding rule assigns the same tensor to both keys), so
      moving it would put a SECOND 2.0 GiB copy on the card -- and ``encode()``
      calls the inner model, which never reaches ``lm_head``, so it is dead
      weight during encoding;
    * the first ``layers_on_gpu`` decoder layers move to the GPU and stay there;
    * the rest get a forward wrapper that moves the layer in, runs it, and moves
      it back out.

    ``layers_on_gpu=0`` is the low-water mark: only one layer's weights (~130 MB)
    are on the card at a time. ``layers_on_gpu >= 48`` disables streaming.

    Install is idempotent, keyed on ``_OFFLOAD_ATTR``: with ``cache_models=True``
    the registry hands back the same module tree on every build, and a second
    wrapper around the first would move each layer twice per pass.
    """

    def __init__(self, device: torch.device, layers_on_gpu: int = 0) -> None:
        self.device = device
        self.layers_on_gpu = int(layers_on_gpu)

    def place(self, model: torch.nn.Module, device: torch.device | None = None) -> dict[str, Any]:
        device = device or self.device
        if device.type == "cpu":
            logger.info("text encoder stays on CPU (device=%s); no offload installed", device)
            return {"device": str(device), "streaming": False, "resident_layers": 0}

        layers = language_model_layers_of(model)
        total = len(layers)
        resident = total if self.layers_on_gpu >= total else max(0, self.layers_on_gpu)

        skip = {id(module) for layer in layers for module in layer.modules()}
        lm_head = getattr(model.model, "lm_head", None)
        if lm_head is not None:
            skip |= {id(module) for module in lm_head.modules()}

        started = time.perf_counter()
        moved = _move_module_tree(model, device, skip=skip)
        for layer in list(layers)[:resident]:
            _move_module_tree(layer, device, skip=set())

        streaming = resident < total
        if streaming:
            self._install_stream(layers, resident, device)

        # ``LTXGemmaTextEncoder.encode`` puts its input ids on ``self.model.device``,
        # which transformers derives from the FIRST parameter it finds. That has to
        # be the (resident) embedding table -- if a streamed layer or lm_head came
        # first, the whole encode would silently run against a CPU device and fail
        # inside the embedding lookup instead of here.
        model_device = getattr(model.model, "device", None)
        if model_device is not None and model_device.type != device.type:
            raise Ltx25GemmaError(
                f"after placement the text encoder reports device={model_device} but was placed on "
                f"{device}: the first parameter of the official module tree is no longer the resident "
                f"embedding table, so `encode()` would build its inputs on the wrong device."
            )
        logger.info(
            "Text encoder placed on %s in %.1fs: %d non-layer tensors moved, %d/%d layers resident, "
            "%s, lm_head left on CPU (shares storage with embed_tokens)",
            device, time.perf_counter() - started, moved, resident, total,
            "streaming the rest" if streaming else "no streaming",
        )
        return {
            "device": str(device),
            "streaming": streaming,
            "resident_layers": resident,
            "total_layers": total,
            "moved_non_layer_tensors": moved,
        }

    def _install_stream(self, layers: torch.nn.ModuleList, resident: int, device: torch.device) -> None:
        installed = 0
        for index, layer in enumerate(layers):
            if index < resident or hasattr(layer, _OFFLOAD_ATTR):
                continue
            setattr(layer, _OFFLOAD_ATTR, layer.forward)
            layer.forward = self._streamed(layer, device)  # type: ignore[method-assign]
            installed += 1
        already = len(layers) - resident - installed
        logger.info(
            "layer stream: wrapped %d layer(s)%s",
            installed, f" ({already} already wrapped -- idempotent)" if already else "",
        )

    @staticmethod
    def _streamed(layer: torch.nn.Module, device: torch.device) -> Any:
        original = getattr(layer, _OFFLOAD_ATTR)

        def streamed(*args: Any, **kwargs: Any) -> Any:
            _move_module_tree(layer, device, skip=set())
            try:
                return original(*args, **kwargs)
            finally:
                _move_module_tree(layer, _CPU, skip=set())

        return streamed

    def uninstall(self, model: torch.nn.Module) -> int:
        """Restore the original forwards and leave the streamed layers on CPU."""
        cleaned = 0
        for layer in language_model_layers_of(model):
            original = getattr(layer, _OFFLOAD_ATTR, None)
            if original is None:
                continue
            layer.forward = original  # type: ignore[method-assign]
            delattr(layer, _OFFLOAD_ATTR)
            _move_module_tree(layer, _CPU, skip=set())
            cleaned += 1
        logger.info("layer stream uninstalled from %d layer(s)", cleaned)
        return cleaned


# ---------------------------------------------------------------------------
# 4. Builders
# ---------------------------------------------------------------------------


class Ltx25GemmaBuilder(Ltx25CpuModelBuilder):
    """CPU builder that places the model before handing it back.

    ``PromptEncoder`` wraps the built encoder in ``gpu_model(...)``, and
    ``gpu_model`` only *disposes* on exit -- it never moves anything onto the
    GPU. So placement has to happen inside ``build()``, or the model would run
    entirely on the CPU.
    """

    def __init__(self, *args: Any, offloader: Ltx25GemmaLayerOffloader | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # `with_*()` clones via copy.copy, which carries this through.
        self._offloader = offloader
        self.placement: dict[str, Any] | None = None

    def build(self, device: torch.device | None = None, dtype: torch.dtype | None = None, **kwargs: Any) -> Any:
        model = super().build(device=device, dtype=dtype, **kwargs)
        if self._offloader is not None:
            self.placement = self._offloader.place(model, device)
        return model


def resolve_assets_path(te_gguf: str | Path, assets_path: str | Path | None = None) -> Path:
    """The assets-only file for *te_gguf*, exported if missing or stale (F2)."""
    if assets_path is not None:
        target = Path(assets_path)
        if not target.is_file():
            raise FileNotFoundError(f"assets-only text encoder not found: {target}")
        return target
    return assets_export.ensure_assets_only(te_gguf).path


def build_text_encoder_builder(
    te_gguf: str | Path,
    *,
    device: torch.device,
    assets_path: str | Path | None = None,
    layers_on_gpu: int = 0,
    registry: Registry | None = None,
    cache_weights: bool = False,
    include_processor: bool = False,
) -> Ltx25GemmaBuilder:
    """A ``text_encoder_builder`` for ``PromptEncoder``, backed by the TE GGUF.

    ``cache_weights`` defaults to False, unlike the transformer's. The text
    encoder is built once per job and disposed before the diffusion stages start,
    so there is no mid-job rebuild to protect against -- and retaining its state
    dict would add 9.2 GB of resident RAM on top of the transformer's 14.7 GB.
    """
    te_gguf = str(te_gguf)
    if not Path(te_gguf).is_file():
        raise FileNotFoundError(f"text-encoder GGUF not found: {te_gguf}")
    assets = resolve_assets_path(te_gguf, assets_path)

    registry = registry or ModelRegistry(cache_models=True, cache_weights=cache_weights)
    logger.info(
        "text encoder builder: %s (assets=%s, layers_on_gpu=%d, cache_weights=%s)",
        Path(te_gguf).name, assets.name, layers_on_gpu, cache_weights,
    )
    return Ltx25GemmaBuilder(
        model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(str(assets)),
        model_path=te_gguf,
        model_sd_ops=build_text_encoder_sd_ops(),
        module_ops=build_text_encoder_module_ops(assets, include_processor=include_processor),
        model_loader=Ltx25GgufStateDictLoader(te_gguf),
        registry=registry,
        offloader=Ltx25GemmaLayerOffloader(device, layers_on_gpu),
    )


def build_embeddings_processor_builder(
    te_gguf: str | Path,
    transformer_gguf: str | Path,
    *,
    assets_path: str | Path | None = None,
    registry: Registry | None = None,
    cache_weights: bool = False,
    chunk_elements: int = AGGREGATE_CHUNK_ELEMENTS,
) -> Ltx25CpuModelBuilder:
    """A builder for the ``EmbeddingsProcessor`` spanning both GGUFs.

    The transformer GGUF comes first: it is the file the official
    ``EmbeddingsProcessorConfigurator`` reads ``config.transformer`` and
    ``gemma_source_checkpoint`` from, and ``read_model_metadata`` takes the first
    path. The text-encoder GGUF supplies only the four
    ``text_embedding_projection.*`` tensors; everything else in it is filtered out
    by ``LTX25_EMBEDDINGS_PROCESSOR_KEY_OPS``, and because that filter runs before
    any tensor data is touched, the 9.2 GB of Gemma weights are never read.
    """
    te_gguf, transformer_gguf = str(te_gguf), str(transformer_gguf)
    for path in (te_gguf, transformer_gguf):
        if not Path(path).is_file():
            raise FileNotFoundError(f"GGUF not found: {path}")
    assets = resolve_assets_path(te_gguf, assets_path)
    paths = (transformer_gguf, te_gguf)

    registry = registry or ModelRegistry(cache_models=True, cache_weights=cache_weights)
    return Ltx25CpuModelBuilder(
        model_class_configurator=EmbeddingsProcessorConfigurator.with_gemma_model_path(str(assets)),
        model_path=paths,
        model_sd_ops=LTX25_EMBEDDINGS_PROCESSOR_KEY_OPS,
        module_ops=build_embeddings_processor_module_ops(chunk_elements=chunk_elements),
        model_loader=Ltx25MultiGgufStateDictLoader(paths),
        registry=registry,
    )


# ---------------------------------------------------------------------------
# 5. Key reconciliation (gate G3.2)
# ---------------------------------------------------------------------------


def map_gguf_keys(gguf_path: str | Path, sd_ops: SDOps) -> tuple[set[str], list[str]]:
    """``(keys the loader will produce, tensor names the sd_ops drops)``.

    Runs the same two calls the loader makes, in the same order -- rename first,
    then the key/value expansion, because the official tied-``lm_head`` rule
    matches on the POST-rename key.
    """
    probe = torch.zeros(1)
    mapped: set[str] = set()
    dropped: list[str] = []
    for name in read_gguf_tensor_names(str(gguf_path)):
        key = sd_ops.apply_to_key(name)
        if key is None:
            dropped.append(name)
            continue
        for out_key, _ in sd_ops.apply_to_key_value(key, probe):
            mapped.add(out_key)
    return mapped, dropped


def reconcile_text_encoder_keys(
    te_gguf: str | Path,
    assets_path: str | Path,
    *,
    include_processor: bool = False,
) -> dict[str, Any]:
    """Build the real model on ``meta`` and check the GGUF covers its state dict.

    This is the gate that says "Gemma 4 is really being loaded", not "something
    loaded without raising". Cheap (~2 s, no weights read) and total: every key
    the model will ask ``load_state_dict`` for must be produced by the GGUF plus
    the sd_ops, or that tensor stays on meta and the forward reads garbage.
    """
    sd_ops = build_text_encoder_sd_ops()
    mapped, dropped = map_gguf_keys(te_gguf, sd_ops)

    metadata = read_gguf_metadata(str(te_gguf))
    model = create_meta_model(
        GemmaTextEncoderConfigurator.with_gemma_model_path(str(assets_path)),
        metadata,
        build_text_encoder_module_ops(assets_path, include_processor=include_processor),
    )
    expected = set(model.state_dict().keys())
    missing = sorted(expected - mapped)
    extra = sorted(mapped - expected)
    pattern = assert_layer_pattern(model)
    del model
    gc.collect()

    return {
        # `mapped_keys` can exceed `gguf_tensors - dropped`: the official tied-
        # embedding rule emits `model.lm_head.weight` from the embed_tokens tensor.
        "gguf_tensors": len(read_gguf_tensor_names(str(te_gguf))),
        "mapped_keys": len(mapped),
        "dropped_by_sd_ops": sorted(dropped),
        "model_state_dict_keys": len(expected),
        "missing_from_gguf": missing,
        "extra_in_gguf": extra,
        "covers_model": not missing,
        "layer_pattern": pattern,
    }


def compare_with_official_safetensors(gguf_path: str | Path, official_path: str | Path) -> dict[str, Any]:
    """Header-only diff of the GGUF against the official bf16 TE file.

    Reads just the safetensors header (a few hundred KB of a 26.3 GB file), so
    this is affordable as a routine check rather than a one-off audit.
    """
    import struct

    with open(official_path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    official_meta = header.pop("__metadata__", {})
    official_names = set(header)
    gguf_names = set(read_gguf_tensor_names(str(gguf_path)))
    gguf_config = read_gguf_metadata(str(gguf_path)).get("config")
    official_config = json.loads(official_meta.get(assets_export.GEMMA_CONFIG_METADATA_KEY, "null"))
    return {
        "official_file": Path(official_path).name,
        "official_tensors": len(official_names),
        "gguf_tensors": len(gguf_names),
        "only_in_official": sorted(official_names - gguf_names),
        "only_in_gguf": sorted(gguf_names - official_names),
        "tensor_names_identical": official_names == gguf_names,
        "config_identical": gguf_config == official_config,
    }


# ---------------------------------------------------------------------------
# 6. Selftest (gate G3)
# ---------------------------------------------------------------------------

#: Short / long / Japanese-mixed. The third is the one that matters most: the
#: tokenizer is the only piece rebuilt from a sidecar rather than a real file, so
#: a multi-byte prompt is what proves the 32 MB tokenizer_json survived the round
#: trip through the GGUF and back out of the assets export.
SELFTEST_PROMPTS = (
    "A red balloon.",
    (
        "A slow cinematic dolly shot through a rain-soaked neon alley at night, "
        "steam rising from a grate, reflections rippling across wet asphalt, a "
        "lone figure in a long coat walking away from camera while distant "
        "traffic hums and a shop sign flickers erratically overhead."
    ),
    "夕暮れの砂浜を歩く少女、a gentle breeze、波の音、35mm film grain。",
)


def _digest(tensors: Any) -> str:
    """SHA-256 over tensor bytes -- the cheap way to prove two runs are identical.

    Comparing hashes rather than keeping the tensors around matters here: one
    prompt's hidden states are 49 x 1024 x 3840 bf16 = 385 MiB, and the bit-exact
    check needs a reference for every prompt.
    """
    digest = hashlib.sha256()
    stack = [tensors]
    while stack:
        item = stack.pop(0)
        if isinstance(item, torch.Tensor):
            array = item.detach().to("cpu").contiguous().view(torch.uint8).numpy()
            digest.update(array.tobytes())
        elif isinstance(item, (tuple, list)):
            stack = list(item) + stack
        else:
            raise TypeError(f"cannot digest {type(item).__name__}")
    return digest.hexdigest()


def _encode(model: torch.nn.Module, prompts: tuple[str, ...]) -> list[Any]:
    with torch.no_grad():
        return model.encode(list(prompts))


def _selftest(  # noqa: PLR0913, PLR0915
    te_gguf: str,
    *,
    transformer_gguf: str | None,
    official_te: str | None,
    layers_on_gpu: int,
    chunk_elements: int,
    rounds: int,
) -> dict:
    from engine25 import ltxcore_compat

    ltxcore_compat.verify()

    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    vram = _Vram(device)
    report: dict[str, Any] = {
        "te_gguf": te_gguf,
        "transformer_gguf": transformer_gguf,
        "device": str(device),
        "layers_on_gpu": layers_on_gpu,
        "aggregate_chunk_elements": chunk_elements,
        "prompts": [len(p) for p in SELFTEST_PROMPTS],
        "rounds": [],
        "phases": vram.phases,
        "checks": {},
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        report["gpu"] = {"name": props.name, "total_gib": round(props.total_memory / 2**30, 2)}
    report["rss_start_gib"] = round((_rss_bytes() or 0) / 2**30, 2)

    # RMSNorm folding is a *contract*, so assert the provenance of the patch that
    # is actually installed instead of trusting the import list to stay clean.
    report["checks"]["dequant_patch_is_transformer_side"] = (
        _patch_model_for_ggml_dequant.__module__ == "engine.gguf.quant_service"
        and _patch_linear_for_ggml_dequant.__module__ == "engine.gguf.quant_service"
    )
    report["checks"]["no_gemma3_rmsnorm_module_imported"] = (
        "engine.gemma.gguf_quant_service" not in sys.modules
    )

    # -- assets export (F2) --------------------------------------------------
    vram.reset()
    started = time.perf_counter()
    export = assets_export.ensure_assets_only(te_gguf)
    report["assets_export"] = export.as_dict()
    report["assets_export"]["official_loader"] = assets_export.verify_with_official_loader(export.path)
    vram.record("00_assets_export", time.perf_counter() - started)

    # -- key reconciliation (G3.2) ------------------------------------------
    started = time.perf_counter()
    reconcile = reconcile_text_encoder_keys(te_gguf, export.path)
    report["reconcile"] = reconcile
    report["checks"]["gguf_covers_model_state_dict"] = reconcile["covers_model"]
    report["checks"]["no_unmapped_gguf_tensors"] = not reconcile["extra_in_gguf"]
    vram.record("01_key_reconcile", time.perf_counter() - started)

    if official_te:
        report["official_compare"] = compare_with_official_safetensors(te_gguf, official_te)
        report["checks"]["matches_official_te_header"] = (
            report["official_compare"]["tensor_names_identical"]
            and report["official_compare"]["config_identical"]
        )

    # -- build ---------------------------------------------------------------
    builder = build_text_encoder_builder(te_gguf, device=device, assets_path=export.path, layers_on_gpu=layers_on_gpu)
    embeddings_builder = (
        build_embeddings_processor_builder(
            te_gguf, transformer_gguf, assets_path=export.path, chunk_elements=chunk_elements
        )
        if transformer_gguf
        else None
    )

    reference: dict[str, str] | None = None
    for index in range(1, rounds + 1):
        round_report: dict[str, Any] = {"round": index}

        vram.reset()
        started = time.perf_counter()
        encoder = builder.build(device=device, dtype=torch.bfloat16).eval()
        round_report["build_seconds"] = round(time.perf_counter() - started, 2)
        round_report["placement"] = builder.placement
        vram.record(f"{index:02d}a_te_build", time.perf_counter() - started)

        if index == 1:
            round_report["layer_pattern"] = assert_layer_pattern(encoder)
            # A second place() must be a no-op: the shell is reused across builds,
            # and a wrapper around the wrapper would move every layer twice.
            # Read the installed override out of __dict__ rather than via
            # `layer.forward`: an *unwrapped* layer returns a freshly built bound
            # method on every access, so comparing ids would report a spurious
            # difference for the no-streaming configuration.
            def _overrides() -> list[Any]:
                return [layer.__dict__.get("forward") for layer in language_model_layers_of(encoder)]

            before = _overrides()
            builder._offloader.place(encoder, device)  # noqa: SLF001 -- our own attribute
            report["checks"]["offload_install_is_idempotent"] = all(
                a is b for a, b in zip(before, _overrides(), strict=True)
            )

        vram.reset()
        started = time.perf_counter()
        raw_outputs = _encode(encoder, SELFTEST_PROMPTS)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        encode_seconds = time.perf_counter() - started
        vram.record(f"{index:02d}b_te_encode", encode_seconds)
        round_report["encode_seconds"] = round(encode_seconds, 2)

        hidden_digests = [_digest(hidden) for hidden, _mask in raw_outputs]
        round_report["hidden_state_layers"] = len(raw_outputs[0][0])
        round_report["hidden_state_shape"] = list(raw_outputs[0][0][0].shape)
        round_report["hidden_state_dtype"] = str(raw_outputs[0][0][0].dtype)
        round_report["hidden_finite"] = all(
            bool(torch.isfinite(h.float()).all().item()) for hidden, _ in raw_outputs for h in hidden
        )
        round_report["hidden_digests"] = hidden_digests

        # Second encode with the SAME model: the plain determinism question.
        started = time.perf_counter()
        repeat = _encode(encoder, SELFTEST_PROMPTS)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        vram.record(f"{index:02d}c_te_encode_repeat", time.perf_counter() - started)
        round_report["repeat_bit_identical"] = [_digest(h) for h, _ in repeat] == hidden_digests
        del repeat

        # Dispose the encoder BEFORE the embeddings processor is built -- that is
        # the order `PromptEncoder.__call__` uses, and the hidden states survive it
        # because they are plain tensors the model does not own. Measuring in any
        # other order would report a peak the real pipeline never reaches.
        vram.reset()
        started = time.perf_counter()
        encoder.dispose()
        del encoder
        gc.collect()
        cleanup_memory()
        vram.record(f"{index:02d}d_te_dispose", time.perf_counter() - started)

        # -- EmbeddingsProcessor (aggregate projections; gate L) -------------
        if embeddings_builder is not None:
            vram.reset()
            started = time.perf_counter()
            processor = embeddings_builder.build(device=_CPU, dtype=torch.bfloat16).eval()
            _move_module_tree(processor, device, skip=set())
            build_seconds = time.perf_counter() - started
            vram.record(f"{index:02d}e_embeddings_build", build_seconds)
            round_report["embeddings_build_seconds"] = round(build_seconds, 2)

            vram.reset()
            started = time.perf_counter()
            with torch.no_grad():
                results = [processor.process_hidden_states(h, m) for h, m in raw_outputs]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            process_seconds = time.perf_counter() - started
            vram.record(f"{index:02d}f_embeddings_process", process_seconds)
            round_report.update(
                embeddings_seconds=round(process_seconds, 2),
                video_encoding_shape=list(results[0].video_encoding.shape),
                audio_encoding_shape=(
                    list(results[0].audio_encoding.shape) if results[0].audio_encoding is not None else None
                ),
                embeddings_finite=all(
                    bool(torch.isfinite(r.video_encoding.float()).all().item())
                    and (r.audio_encoding is None or bool(torch.isfinite(r.audio_encoding.float()).all().item()))
                    for r in results
                ),
            )
            embedding_digests = [
                _digest([r.video_encoding, r.audio_encoding] if r.audio_encoding is not None else [r.video_encoding])
                for r in results
            ]
            round_report["embedding_digests"] = embedding_digests

            with torch.no_grad():
                repeat_results = [processor.process_hidden_states(h, m) for h, m in raw_outputs]
            round_report["embeddings_repeat_bit_identical"] = [
                _digest([r.video_encoding, r.audio_encoding] if r.audio_encoding is not None else [r.video_encoding])
                for r in repeat_results
            ] == embedding_digests
            del repeat_results, results
            vram.reset()
            started = time.perf_counter()
            processor.dispose()
            del processor
            gc.collect()
            cleanup_memory()
            vram.record(f"{index:02d}g_embeddings_dispose", time.perf_counter() - started)
        else:
            embedding_digests = None

        del raw_outputs
        gc.collect()
        cleanup_memory()

        digests = {"hidden": hidden_digests, "embeddings": embedding_digests}
        if reference is None:
            reference = digests
        else:
            round_report["identical_to_round_1"] = digests == reference
        report["rounds"].append(round_report)

    rounds_report = report["rounds"]
    report["checks"]["all_hidden_states_finite"] = all(entry["hidden_finite"] for entry in rounds_report)
    report["checks"]["repeat_encode_bit_identical"] = all(
        entry["repeat_bit_identical"] for entry in rounds_report
    )
    if transformer_gguf:
        report["checks"]["all_embeddings_finite"] = all(
            entry["embeddings_finite"] for entry in rounds_report
        )
        report["checks"]["repeat_embeddings_bit_identical"] = all(
            entry["embeddings_repeat_bit_identical"] for entry in rounds_report
        )
    if len(rounds_report) > 1:
        report["checks"]["rebuild_bit_identical"] = all(
            entry.get("identical_to_round_1") for entry in rounds_report[1:]
        )

    # Highest per-phase working set, not the working set at the end: RSS falls
    # back once the state dict is released, so a final reading understates the run.
    report["rss_peak_gib"] = max(
        [entry.get("rss_gib") or 0.0 for entry in vram.phases.values()] + [report["rss_start_gib"]]
    )
    report["rss_end_gib"] = round((_rss_bytes() or 0) / 2**30, 2)
    if device.type == "cuda":
        peaks = [entry.get("peak_allocated_gib", 0.0) for entry in vram.phases.values()]
        reserved = [entry.get("peak_reserved_gib", 0.0) for entry in vram.phases.values()]
        report["peak_allocated_gib"] = max(peaks)
        report["peak_reserved_gib"] = max(reserved)
        report["fits_in_16gib"] = report["peak_reserved_gib"] < 16.0
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="engine25.gguf_gemma4")
    parser.add_argument("--selftest", metavar="GGUF", required=True, help="text-encoder GGUF")
    parser.add_argument("--transformer-gguf", default=None, help="also build the EmbeddingsProcessor from this")
    parser.add_argument("--official-te", default=None, help="official bf16 TE safetensors, for a header diff")
    parser.add_argument("--layers-on-gpu", type=int, default=0, help="decoder layers kept resident (0 = stream all)")
    parser.add_argument(
        "--aggregate-chunk-elements",
        type=int,
        default=AGGREGATE_CHUNK_ELEMENTS,
        help="chunk size for the aggregate projections; 0 disables chunking (expect a ~10GiB transient)",
    )
    parser.add_argument("--rounds", type=int, default=2, help="build/encode/dispose cycles")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="[ltx25_gemma4] %(asctime)s %(name)s: %(message)s",
    )

    report = _selftest(
        args.selftest,
        transformer_gguf=args.transformer_gguf,
        official_te=args.official_te,
        layers_on_gpu=args.layers_on_gpu,
        chunk_elements=args.aggregate_chunk_elements,
        rounds=args.rounds,
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
    if __package__ in (None, ""):  # pragma: no cover
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
