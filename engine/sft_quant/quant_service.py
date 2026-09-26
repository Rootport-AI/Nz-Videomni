"""Quantized safetensors transformer: loader, module op and install (§3-167 B-1, B-2, §3-168).

The GGUF per-layer service's twin (``engine/gguf/quant_service.py``) for a
quantized (fp8 / int8) safetensors transformer that ``sft_quant_format.inspect``
has accepted:

  * :class:`SftQuantStateDictLoader` reads the file one tensor at a time
    (``engine.sft_quant.sft_reader``: seek + readinto, never mmap) and keeps the
    quantized Linear weights AS stored (fp8 / int8) — the whole model in bf16
    would be ~37 GB of CPU RAM.
  * the ``sft_quant_linear`` module op replaces every Linear's forward with
    :func:`_quant_linear_forward`, which dequantizes the weight per call
    (``engine.sft_quant.dequant.dequantize``, by the layer's scheme) and adds
    the IC-LoRA / LoRA delta out of place, so the stored weight — shared with
    the keep_resident cache — is never written to.
  * :class:`SftQuantLoaderService.install` wires both into the ledger in the same
    three steps as the GGUF service (loader, policy, transformer() wrapper).
  * :func:`sft_transformer_sd_ops` and :func:`load_connector_bf16` are the
    pieces shared with the LTX 2.5 engine (engine25): the key ops for the
    detected prefix, and the text encoder side's connectors in bf16.

Each quantized layer's weight is re-made on the meta skeleton as a
``requires_grad=False`` Parameter of the stored shape (an int8 Parameter cannot
require grad), and its auxiliary tensors (``weight_scale`` ...) as persistent
meta buffers of ``sft_quant_format.aux_specs``' dtype and shape, so
``load_state_dict(strict=False, assign=True)`` fills them from the file and
they then ride block swap / prefetch / the CPU-resident build / the
keep_resident cache like any other state-dict tensor.
"""

from __future__ import annotations

import logging
import types
from collections import Counter
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from engine.sft_quant.dequant import dequantize, normalize_aux
from engine.sft_quant.sft_reader import TORCH_DTYPES, read_tensors
from engine.gguf.ic_lora_common import IC_LORA_SPECS_ATTR

logger = logging.getLogger(__name__)

_TEXT_PROJ_HEAD = "text_embedding_projection."
_CONNECTOR_MARK = "_embeddings_connector."
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
#: Stored dtypes only a patched Linear knows how to dequantize (GGUF's uint8
#: buffers are not among them).
_QUANT_DTYPES = (*_FP8_DTYPES, torch.int8)
#: Non-quantized float dtypes brought to bf16 — the one rule, loader and connector.
_TO_BF16_DTYPES = (torch.float16, torch.float32)
#: Python attribute the module op puts on each quantized Linear: its scheme.
_SCHEME_ATTR = "_sft_scheme"


def _aux_target(scheme: str | None, key: str, leaf: str, header: Any) -> tuple[int, int]:
    """``(o, i)`` of the layer the auxiliary tensor ``key`` belongs to, from the
    stored weight beside it. A leaf the layer's scheme does not take (or a
    layer that is not quantized) is a surplus key: RuntimeError."""
    import sft_quant_format

    if scheme is None or leaf not in sft_quant_format.aux_specs(scheme):
        raise RuntimeError(
            f"quantized safetensors: '{key}' is an auxiliary tensor that no quantized "
            f"layer takes (scheme {scheme!r})"
        )
    weight = header.tensors.get(key[: -len(leaf)] + "weight")
    if weight is None or len(weight.shape) != 2:
        raise RuntimeError(f"quantized safetensors: '{key}' has no 2-D weight beside it")
    o, stored_i = weight.shape
    i = stored_i * sft_quant_format.SCHEME_TABLE[scheme].packing
    return int(o), int(i)


# ──────────────────────────────────────────────────────────────────────────────
# State-dict loader
# ──────────────────────────────────────────────────────────────────────────────


class SftQuantStateDictLoader:
    """``StateDictLoader`` over a quantized safetensors transformer.

    Same shape as the wheel's ``SafetensorsStateDictLoader`` (sd_ops applied per
    key, ``None`` keys skipped, ``apply_to_key_value`` on each value), but the
    bytes come from :func:`read_tensors` instead of ``safe_open``.

    ``path`` arguments are ignored in favour of the file this loader was built
    for: the builder hands in ``checkpoint_path`` (``""`` in this product), the
    same situation the GGUF loader handles the same way.
    """

    def __init__(self, path: str, layout: Any) -> None:
        self.path = path
        self.layout = layout

    def metadata(self, path: str) -> dict:
        # The model config comes from ONE place: the Layout inspect built from
        # __metadata__.config (required by inspect).
        return self.layout.config

    def load(
        self,
        path: str | list[str],
        sd_ops: Any = None,
        device: torch.device | None = None,
    ) -> Any:
        import sft_quant_format
        from ltx_core.loader.primitives import StateDict

        device = device or torch.device("cpu")
        header = sft_quant_format.read_header(self.path)
        connectors = set(self.layout.connector_keys)
        prefix = self.layout.prefix
        layers = self.layout.layers
        aux_leaves = sft_quant_format.AUX_NAMES

        # key in the file -> key after sd_ops. Not read: comfy_quant (only the
        # scheme's label, already checked by inspect), input_scale (activation
        # quantization; this engine computes in bf16 and the skeleton has no
        # such key) and the connectors and text_embedding_projection (the text
        # encoder's; engine/gemma reads them itself — the latter matters only
        # for a bare-named file, where the identity sd_ops would keep it).
        text_proj = prefix + _TEXT_PROJ_HEAD
        wanted: dict[str, str] = {}
        # auxiliary key -> (scheme, leaf, o, i), resolved BEFORE any read so a
        # surplus key fails the load without touching the weights.
        aux_of: dict[str, tuple[str, str, int, int]] = {}
        for key in header.tensors:
            if (
                key.endswith(sft_quant_format.SKIPPED_SUFFIXES)
                or key in connectors
                or key.startswith(text_proj)
            ):
                continue
            expected = key if sd_ops is None else sd_ops.apply_to_key(key)
            if expected is None:
                continue
            wanted[key] = expected
            layer, _, leaf = key.rpartition(".")
            if leaf in aux_leaves:
                scheme = layers.get(layer[len(prefix):]) if layer.startswith(prefix) else None
                aux_of[key] = (scheme, leaf, *_aux_target(scheme, key, leaf, header))

        sd: dict[str, torch.Tensor] = {}
        size = 0
        dtypes: set[torch.dtype] = set()
        for key, value in read_tensors(self.path, wanted, header):
            # With a quantization policy the wheel does no dtype cast, so F32
            # norms / scale_shift_table would stay F32 unless unified here (the
            # GGUF loader's rule; F16 likewise). Auxiliary tensors keep their
            # dtype and take the one shape per scheme the module op registered
            # (normalize_aux); quantized weights (fp8 / int8) stay as stored.
            aux = aux_of.get(key)
            if aux is not None:
                scheme, leaf, o, i = aux
                value = normalize_aux(scheme, leaf, value, o, i)
            elif value.dtype in _TO_BF16_DTYPES:
                value = value.to(torch.bfloat16)
            if device.type != "cpu":
                value = value.to(device)  # the CPU copy is dropped with the name
            expected = wanted[key]
            pairs = (
                ((expected, value),)
                if sd_ops is None
                else sd_ops.apply_to_key_value(expected, value)
            )
            for k, v in pairs:
                size += v.nbytes
                dtypes.add(v.dtype)
                sd[k] = v

        logger.info(
            "quantized safetensors load: %d tensors from %s -> %s (schemes=%s)",
            len(sd), Path(self.path).name, device, dict(Counter(layers.values())),
        )
        return StateDict(sd=sd, device=device, size=size, dtype=dtypes)


def sft_transformer_sd_ops(prefix: str):
    """Transformer key ops for the prefix ``inspect`` detected.

    ``"model.diffusion_model."`` -> the wheel's own ``LTXV_MODEL_COMFY_RENAMING_MAP``
    (what ``ModelLedger`` already puts on the transformer builder, so the 2.3
    path is unchanged). ``""`` (bare names) -> identity: ``with_matching()``
    with no prefix/suffix matches every key. A named SDOps WITHOUT a matcher
    would drop every key instead.
    """
    from ltx_core.loader.sd_ops import SDOps
    from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP

    if prefix == "model.diffusion_model.":
        return LTXV_MODEL_COMFY_RENAMING_MAP
    if prefix == "":
        return SDOps("FP8_BARE_PASSTHRU").with_matching()
    raise ValueError(f"quantized safetensors transformer: unknown key prefix {prefix!r}")


def load_connector_bf16(path: str) -> dict[str, torch.Tensor]:
    """The embeddings connectors of a quantized safetensors transformer, in bf16.

    Keys come back WITHOUT the file's prefix (``video_embeddings_connector.*`` /
    ``audio_embeddings_connector.*``). Read one tensor at a time with
    :func:`read_tensors` (never mmap), in two passes: first every auxiliary
    tensor (normalized), then the tensors one by one, each quantized weight
    dequantized on the CPU by its layer's scheme
    (``sft_quant_format.layer_schemes``) straight to bf16 — the raw weights are
    never held together. An fp8 bias is a plain cast (the scale belongs to the
    weight); F16 / F32 -> bf16; BF16 as is. Auxiliary tensors, ``input_scale``
    and ``comfy_quant`` are not returned.
    """
    import sft_quant_format

    header = sft_quant_format.read_header(path)
    prefix = sft_quant_format.detect_prefix(header)
    schemes = sft_quant_format.layer_schemes(path, header, prefix)
    aux_leaves = sft_quant_format.AUX_NAMES
    keys = [
        k
        for k in header.tensors
        if k.startswith(prefix)
        and _CONNECTOR_MARK in k
        and not k.endswith(sft_quant_format.SKIPPED_SUFFIXES)
    ]

    # Pass 1: every auxiliary tensor, "<layer>" -> {leaf: normalized tensor}.
    aux_of: dict[str, tuple[str, str, str, int, int]] = {}
    wanted: list[str] = []
    for key in keys:
        layer, _, leaf = key.rpartition(".")
        if leaf in aux_leaves:
            scheme = schemes.get(layer)
            aux_of[key] = (layer, scheme, leaf, *_aux_target(scheme, key, leaf, header))
        else:
            wanted.append(key)
    aux: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in read_tensors(path, aux_of, header):
        layer, scheme, leaf, o, i = aux_of[key]
        aux.setdefault(layer, {})[leaf] = normalize_aux(scheme, leaf, value, o, i)

    # Pass 2: one tensor at a time.
    out: dict[str, torch.Tensor] = {}
    for key, value in read_tensors(path, wanted, header):
        layer, _, leaf = key.rpartition(".")
        scheme = schemes.get(layer) if leaf == "weight" else None
        if scheme is not None:
            value = dequantize(scheme, value, aux.get(layer, {}), torch.bfloat16)
        elif value.dtype in _FP8_DTYPES:
            value = value.to(torch.bfloat16)  # an fp8 bias: no scale
        elif value.dtype in _TO_BF16_DTYPES:
            value = value.to(torch.bfloat16)
        elif value.dtype != torch.bfloat16:
            raise RuntimeError(
                f"quantized safetensors connector: '{key}' is {value.dtype} — expected "
                f"BF16, F16, F32, fp8 or a quantized weight"
            )
        out[key[len(prefix):]] = value
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Module op: quantization-aware Linear forward
# ──────────────────────────────────────────────────────────────────────────────


def _quant_linear_forward(self: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
    w = self.weight
    scheme = getattr(self, _SCHEME_ATTR, None)
    specs = getattr(self, IC_LORA_SPECS_ATTR, None)
    b = self.bias if self.bias is None or self.bias.dtype == x.dtype else self.bias.to(x.dtype)
    if w.dtype == x.dtype and scheme is None and not specs:
        return F.linear(x, w, b)  # bf16 layer: byte-identical to a plain Linear
    wd = dequantize(scheme, w, self._buffers, x.dtype)
    if specs:
        # Same per-spec formula as ggml_linear_forward's float branch, except the
        # cast goes to x.dtype: casting to w.dtype would round the delta to fp8.
        acc = None
        for a_name, b_name, strength in specs:
            delta = torch.matmul(
                getattr(self, b_name).to(torch.float32) * strength,
                getattr(self, a_name).to(torch.float32),
            ).to(x.dtype)
            acc = delta if acc is None else acc + delta
        wd = wd + acc  # out of place: the stored weight is never written
    return F.linear(x, wd, b)


def _patch_model_for_quant(
    model: torch.nn.Module, layers: dict[str, str]
) -> torch.nn.Module:
    """Patch every Linear's forward, and re-make each quantized layer's slots.

    ``layers`` is ``Layout.layers`` (prefix-less Linear name -> scheme). For each
    one — fp8 included, one rule — the scheme goes on the module as
    ``_sft_scheme``, the weight becomes a ``requires_grad=False`` meta Parameter
    of the stored shape (``load_state_dict(assign=True)`` keeps the flag, and an
    int8 Parameter cannot require grad), and every auxiliary tensor of the
    scheme a persistent meta buffer of its normalized dtype and shape (a key
    the skeleton lacks would be dropped silently). ``get_submodule`` raises for
    a layer with no module.
    """
    import sft_quant_format

    count = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            m.forward = types.MethodType(_quant_linear_forward, m)
            count += 1
    n_aux = 0
    for name, scheme in layers.items():
        m = model.get_submodule(name)
        o, i = m.out_features, m.in_features
        setattr(m, _SCHEME_ATTR, scheme)
        m.weight = torch.nn.Parameter(
            torch.empty(
                sft_quant_format.weight_shape(scheme, o, i),
                dtype=m.weight.dtype,
                device="meta",
            ),
            requires_grad=False,
        )
        for leaf, (dtype_name, rule) in sft_quant_format.aux_specs(scheme).items():
            m.register_buffer(
                leaf,
                torch.empty(
                    sft_quant_format.aux_shape(rule, o, i),
                    dtype=TORCH_DTYPES[dtype_name],
                    device="meta",
                ),
                persistent=True,
            )
            n_aux += 1
    logger.debug(
        "sft_quant module_ops: patched %d Linear forwards, %d quantized layers %s, "
        "%d auxiliary buffers",
        count, len(layers), dict(Counter(layers.values())), n_aux,
    )
    return model


def _make_quant_module_ops(layers: dict[str, str]):
    from ltx_core.loader.module_ops import ModuleOps
    from ltx_core.model.transformer.model import LTXModel

    return ModuleOps(
        name="sft_quant_linear",
        matcher=lambda model: isinstance(model, LTXModel),
        mutator=lambda model: _patch_model_for_quant(model, layers),
    )


def _assert_quant_only_in_linears(model: torch.nn.Module) -> None:
    """Only a patched Linear knows how to dequantize fp8 / int8; anywhere else it
    is a bug."""
    stray = []
    for mod_name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            continue
        for t_name, t in (*mod._parameters.items(), *mod._buffers.items()):
            if t is not None and t.dtype in _QUANT_DTYPES:
                stray.append(f"{mod_name}.{t_name}" if mod_name else t_name)
    if stray:
        raise RuntimeError(
            f"quantized safetensors transformer: {len(stray)} fp8/int8 tensor(s) "
            f"outside Linear layers (nothing would dequantize them): {stray[:5]}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Service — same install() shape as GGUFQuantLoaderService
# ──────────────────────────────────────────────────────────────────────────────


class SftQuantLoaderService:
    """Install a quantized safetensors transformer into a ``ModelLedger``."""

    def __init__(
        self,
        path: str,
        *,
        dit_cpu_load: bool,
        ic_loras_provider: Any = None,
    ) -> None:
        self.path = path
        self._dit_cpu_load = dit_cpu_load
        # Callable returning the CURRENT job's IC-LoRA entries (as in the GGUF
        # service): read on every transformer build.
        self._ic_loras_provider = ic_loras_provider
        self.layout: Any = None

    def install(self, model_ledger: Any) -> None:
        if not self._dit_cpu_load:
            raise RuntimeError(
                "a quantized safetensors transformer must be built on the CPU side: "
                "building it on the GPU runs out of VRAM. Enable dit_cpu_load "
                "(LTX_DIT_CPU_LOAD=1) to use a quantized safetensors transformer."
            )

        import sft_quant_format
        from ltx_core.quantization import QuantizationPolicy

        # The ONE acceptance check on the engine side; its Layout feeds both the
        # loader and the module op.
        layout = sft_quant_format.inspect(self.path)
        self.layout = layout

        # 1. Replace the transformer builder's loader and key ops (for a
        #    prefixed file the key ops are the ones the ledger already had).
        model_ledger.transformer_builder = dc_replace(
            model_ledger.transformer_builder,
            model_loader=SftQuantStateDictLoader(self.path, layout),
            model_sd_ops=sft_transformer_sd_ops(layout.prefix),
        )

        # 2. Replace the policy WHOLESALE. fp8_cast's sd_ops
        #    (TRANSFORMER_LINEAR_DOWNCAST_MAP) would push biases and the bf16
        #    blocks down to fp8, and its only module op (UPCAST_DURING_INFERENCE)
        #    is superseded by sft_quant_linear. Not None: without a policy the ledger
        #    casts the whole state dict to bf16 (~37 GB).
        model_ledger.quantization = QuantizationPolicy(
            sd_ops=None,
            module_ops=(_make_quant_module_ops(layout.layers),),
        )

        # 3. Wrap transformer(): IC-LoRA attach/detach (the GGUF service's copy),
        #    then check that no fp8 / int8 tensor ended up outside a Linear.
        original_transformer_fn = model_ledger.transformer.__func__

        def patched_transformer(self_ledger: Any) -> Any:
            result = original_transformer_fn(self_ledger)
            if self._ic_loras_provider is not None:
                ic_loras = list(self._ic_loras_provider() or [])
                if ic_loras:
                    from engine.gguf.ic_lora_common import attach_ic_loras
                    attach_ic_loras(result, ic_loras)
                else:
                    from engine.gguf.ic_lora_common import detach_ic_loras
                    detach_ic_loras(result)
            _assert_quant_only_in_linears(result)
            return result

        model_ledger.transformer = types.MethodType(patched_transformer, model_ledger)

        logger.info(
            "SftQuantLoaderService installed: %s (schemes=%s)",
            Path(self.path).name, dict(Counter(layout.layers.values())),
        )
