"""fp8 safetensors transformer: loader, module op and install (§3-167 B-1, B-2).

The GGUF per-layer service's twin (``engine/gguf/quant_service.py``) for an
fp8 safetensors transformer that ``sft_quant_format.inspect`` has accepted:

  * :class:`SftQuantStateDictLoader` reads the file one tensor at a time
    (``engine.sft_quant.sft_reader``: seek + readinto, never mmap) and keeps the fp8
    Linear weights AS fp8 — the whole model in bf16 would be ~37 GB of CPU RAM.
  * the ``sft_quant_linear`` module op replaces every Linear's forward with
    :func:`_quant_linear_forward`, which upcasts the weight per call (times the
    scalar ``weight_scale`` for the "scaled" flavor) and adds the IC-LoRA /
    LoRA delta out of place, so the stored weight — shared with the
    keep_resident cache — is never written to.
  * :class:`SftQuantLoaderService.install` wires both into the ledger in the same
    three steps as the GGUF service (loader, policy, transformer() wrapper).
  * :func:`sft_transformer_sd_ops` and :func:`load_connector_bf16` are the
    pieces shared with the LTX 2.5 engine (engine25): the key ops for the
    detected prefix, and the text encoder side's connectors in bf16.

``weight_scale`` is a persistent 0-dim f32 buffer registered on the meta
skeleton, so ``load_state_dict(strict=False, assign=True)`` fills it from the
file and it then rides block swap / prefetch / the CPU-resident build / the
keep_resident cache like any other state-dict tensor.
"""

from __future__ import annotations

import logging
import types
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from engine.sft_quant.sft_reader import read_tensors
from engine.gguf.ic_lora_common import IC_LORA_SPECS_ATTR

logger = logging.getLogger(__name__)

_SCALE_SUFFIX = ".weight_scale"
_SKIPPED_SUFFIXES = (".comfy_quant", ".input_scale")
_TEXT_PROJ_HEAD = "text_embedding_projection."
_CONNECTOR_MARK = "_embeddings_connector."
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


# ──────────────────────────────────────────────────────────────────────────────
# State-dict loader
# ──────────────────────────────────────────────────────────────────────────────


class SftQuantStateDictLoader:
    """``StateDictLoader`` over an fp8 safetensors transformer.

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

        # key in the file -> key after sd_ops. Not read: comfy_quant (only the
        # scheme's label, already checked by inspect), input_scale (activation
        # quantization; this engine computes in bf16 and the skeleton has no
        # such key) and the connectors and text_embedding_projection (the text
        # encoder's; engine/gemma reads them itself — the latter matters only
        # for a bare-named file, where the identity sd_ops would keep it).
        text_proj = self.layout.prefix + _TEXT_PROJ_HEAD
        wanted: dict[str, str] = {}
        for key in header.tensors:
            if key.endswith(_SKIPPED_SUFFIXES) or key in connectors or key.startswith(text_proj):
                continue
            expected = key if sd_ops is None else sd_ops.apply_to_key(key)
            if expected is None:
                continue
            wanted[key] = expected

        sd: dict[str, torch.Tensor] = {}
        size = 0
        dtypes: set[torch.dtype] = set()
        for key, value in read_tensors(self.path, wanted, header):
            # With a quantization policy the wheel does no dtype cast, so F32
            # norms / scale_shift_table would stay F32 unless unified here (the
            # GGUF loader's rule). weight_scale stays F32: the forward multiplies
            # in f32; shape [1] is brought to 0-dim, the module op's buffer.
            if key.endswith(_SCALE_SUFFIX):
                value = value.reshape(())
            elif value.dtype == torch.float32:
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
            "fp8 safetensors load: %d tensors from %s -> %s (flavor=%s)",
            len(sd), Path(self.path).name, device, self.layout.flavor,
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
    raise ValueError(f"fp8 transformer: unknown key prefix {prefix!r}")


def load_connector_bf16(path: str) -> dict[str, torch.Tensor]:
    """The embeddings connectors of an fp8 safetensors transformer, in bf16.

    Keys come back WITHOUT the file's prefix (``video_embeddings_connector.*`` /
    ``audio_embeddings_connector.*``). Read one tensor at a time with
    :func:`read_tensors` (never mmap). fp8 -> ``float32 * <layer>.weight_scale``
    -> bf16 when the layer has a scale, else a plain cast; F32 -> bf16; BF16 as
    is. ``weight_scale`` / ``input_scale`` / ``comfy_quant`` are not returned.
    """
    import sft_quant_format

    header = sft_quant_format.read_header(path)
    prefix = sft_quant_format.detect_prefix(header)
    keys = [k for k in header.tensors if k.startswith(prefix) and _CONNECTOR_MARK in k]
    scale_keys = [k for k in keys if k.endswith(_SCALE_SUFFIX)]
    # "<layer>.weight" -> its 0-dim f32 scale
    scales = {
        k[: -len(_SCALE_SUFFIX)] + ".weight": v.reshape(())
        for k, v in read_tensors(path, scale_keys, header)
    }
    wanted = [k for k in keys if not k.endswith((_SCALE_SUFFIX, *_SKIPPED_SUFFIXES))]

    out: dict[str, torch.Tensor] = {}
    for key, value in read_tensors(path, wanted, header):
        if value.dtype in _FP8_DTYPES:
            scale = scales.get(key)
            if scale is None:
                value = value.to(torch.bfloat16)
            else:
                value = (value.to(torch.float32) * scale).to(torch.bfloat16)
        elif value.dtype == torch.float32:
            value = value.to(torch.bfloat16)
        elif value.dtype != torch.bfloat16:
            raise RuntimeError(
                f"fp8 connector: '{key}' is {value.dtype} — expected BF16, F32 or fp8"
            )
        out[key[len(prefix):]] = value
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Module op: fp8-aware Linear forward
# ──────────────────────────────────────────────────────────────────────────────


def _quant_linear_forward(self: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
    w = self.weight
    s = self._buffers.get("weight_scale")
    specs = getattr(self, IC_LORA_SPECS_ATTR, None)
    b = self.bias if self.bias is None or self.bias.dtype == x.dtype else self.bias.to(x.dtype)
    if w.dtype == x.dtype and s is None and not specs:
        return F.linear(x, w, b)  # bf16 layer: byte-identical to a plain Linear
    wd = w.to(x.dtype) if s is None else (w.to(torch.float32) * s).to(x.dtype)
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
    model: torch.nn.Module, scaled_layers: frozenset[str]
) -> torch.nn.Module:
    count = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            m.forward = types.MethodType(_quant_linear_forward, m)
            count += 1
    for name in scaled_layers:
        # get_submodule raises when a scaled layer has no module (its scale
        # would otherwise be dropped silently as an unexpected key).
        model.get_submodule(name).register_buffer(
            "weight_scale",
            torch.empty((), dtype=torch.float32, device="meta"),
            persistent=True,
        )
    logger.debug(
        "fp8 module_ops: patched %d Linear forwards, %d weight_scale buffers",
        count, len(scaled_layers),
    )
    return model


def _make_quant_module_ops(scaled_layers: frozenset[str]):
    from ltx_core.loader.module_ops import ModuleOps
    from ltx_core.model.transformer.model import LTXModel

    return ModuleOps(
        name="sft_quant_linear",
        matcher=lambda model: isinstance(model, LTXModel),
        mutator=lambda model: _patch_model_for_quant(model, scaled_layers),
    )


def _assert_quant_only_in_linears(model: torch.nn.Module) -> None:
    """Only a patched Linear knows how to upcast fp8; anywhere else it is a bug."""
    stray = []
    for mod_name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            continue
        for t_name, t in (*mod._parameters.items(), *mod._buffers.items()):
            if t is not None and t.dtype in _FP8_DTYPES:
                stray.append(f"{mod_name}.{t_name}" if mod_name else t_name)
    if stray:
        raise RuntimeError(
            f"fp8 transformer: {len(stray)} fp8 tensor(s) outside Linear layers "
            f"(nothing would upcast them): {stray[:5]}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Service — same install() shape as GGUFQuantLoaderService
# ──────────────────────────────────────────────────────────────────────────────


class SftQuantLoaderService:
    """Install an fp8 safetensors transformer into a ``ModelLedger``."""

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
                "fp8 transformer must be built on the CPU side: building it on "
                "the GPU runs out of VRAM. Enable dit_cpu_load (LTX_DIT_CPU_LOAD=1) "
                "to use an fp8 safetensors transformer."
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
            module_ops=(_make_quant_module_ops(layout.scaled_layers),),
        )

        # 3. Wrap transformer(): IC-LoRA attach/detach (the GGUF service's copy),
        #    then check that no fp8 tensor ended up outside a Linear.
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
            "SftQuantLoaderService installed: %s (flavor=%s, %d scaled layers)",
            Path(self.path).name, layout.flavor, len(layout.scaled_layers),
        )
