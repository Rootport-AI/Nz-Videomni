"""fp8 safetensors transformer: loader, module op and install (§3-167 B-1).

The GGUF per-layer service's twin (``engine/gguf/quant_service.py``) for an
fp8 safetensors transformer that ``sft_fp8_format.inspect`` has accepted:

  * :class:`Fp8StateDictLoader` reads the file one tensor at a time
    (``engine.fp8.sft_reader``: seek + readinto, never mmap) and keeps the fp8
    Linear weights AS fp8 — the whole model in bf16 would be ~37 GB of CPU RAM.
  * the ``fp8_linear`` module op replaces every Linear's forward with
    :func:`_fp8_linear_forward`, which upcasts the weight per call (times the
    scalar ``weight_scale`` for the "scaled" flavor) and adds the IC-LoRA /
    LoRA delta out of place, so the stored weight — shared with the
    keep_resident cache — is never written to.
  * :class:`Fp8LoaderService.install` wires both into the ledger in the same
    three steps as the GGUF service (loader, policy, transformer() wrapper).

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

from engine.fp8.sft_reader import read_tensors
from engine.gguf.ic_lora_common import IC_LORA_SPECS_ATTR

logger = logging.getLogger(__name__)

_SCALE_SUFFIX = ".weight_scale"
_SKIPPED_SUFFIXES = (".comfy_quant", ".input_scale")
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


# ──────────────────────────────────────────────────────────────────────────────
# State-dict loader
# ──────────────────────────────────────────────────────────────────────────────


class Fp8StateDictLoader:
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
        import sft_fp8_format
        from ltx_core.loader.primitives import StateDict

        device = device or torch.device("cpu")
        header = sft_fp8_format.read_header(self.path)
        connectors = set(self.layout.connector_keys)

        # key in the file -> key after sd_ops. Not read: comfy_quant (only the
        # scheme's label, already checked by inspect), input_scale (activation
        # quantization; this engine computes in bf16 and the skeleton has no
        # such key) and the connectors (the text encoder's; engine/gemma reads
        # them itself).
        wanted: dict[str, str] = {}
        for key in header.tensors:
            if key.endswith(_SKIPPED_SUFFIXES) or key in connectors:
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
            # in f32.
            if value.dtype == torch.float32 and not key.endswith(_SCALE_SUFFIX):
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


# ──────────────────────────────────────────────────────────────────────────────
# Module op: fp8-aware Linear forward
# ──────────────────────────────────────────────────────────────────────────────


def _fp8_linear_forward(self: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
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


def _patch_model_for_fp8(
    model: torch.nn.Module, scaled_layers: frozenset[str]
) -> torch.nn.Module:
    count = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            m.forward = types.MethodType(_fp8_linear_forward, m)
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


def _make_fp8_module_ops(scaled_layers: frozenset[str]):
    from ltx_core.loader.module_ops import ModuleOps
    from ltx_core.model.transformer.model import LTXModel

    return ModuleOps(
        name="fp8_linear",
        matcher=lambda model: isinstance(model, LTXModel),
        mutator=lambda model: _patch_model_for_fp8(model, scaled_layers),
    )


def _assert_fp8_only_in_linears(model: torch.nn.Module) -> None:
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


class Fp8LoaderService:
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

        import sft_fp8_format
        from ltx_core.quantization import QuantizationPolicy

        # The ONE acceptance check on the engine side; its Layout feeds both the
        # loader and the module op.
        layout = sft_fp8_format.inspect(self.path)
        self.layout = layout

        # 1. Replace the transformer builder's loader.
        model_ledger.transformer_builder = dc_replace(
            model_ledger.transformer_builder,
            model_loader=Fp8StateDictLoader(self.path, layout),
        )

        # 2. Replace the policy WHOLESALE. fp8_cast's sd_ops
        #    (TRANSFORMER_LINEAR_DOWNCAST_MAP) would push biases and the bf16
        #    blocks down to fp8, and its only module op (UPCAST_DURING_INFERENCE)
        #    is superseded by fp8_linear. Not None: without a policy the ledger
        #    casts the whole state dict to bf16 (~37 GB).
        model_ledger.quantization = QuantizationPolicy(
            sd_ops=None,
            module_ops=(_make_fp8_module_ops(layout.scaled_layers),),
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
            _assert_fp8_only_in_linears(result)
            return result

        model_ledger.transformer = types.MethodType(patched_transformer, model_ledger)

        logger.info(
            "Fp8LoaderService installed: %s (flavor=%s, %d scaled layers)",
            Path(self.path).name, layout.flavor, len(layout.scaled_layers),
        )
