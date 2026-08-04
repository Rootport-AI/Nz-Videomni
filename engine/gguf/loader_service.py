"""GGUF transformer loader service for LTX-2.

Implements the StateDictLoader protocol so that a GGUF-quantized
transformer can be dropped into the ModelLedger pipeline in place of
the default safetensors loader.

The LTX-2 GGUF situation:
- The GGUF file contains ONLY the transformer (DiT) weights, quantized.
- VAE, audio VAE, vocoder, text encoder are still loaded from the
  original safetensors checkpoint separately.
- The GGUF file metadata must contain a 'config' key with the model
  config JSON (same format as the safetensors metadata).

Usage:
    service = GGUFLoaderService(
        gguf_path="/path/to/ltx2_transformer_Q4_K_M.gguf",
        safetensors_checkpoint="/path/to/ltxv2.safetensors",
    )
    service.install(model_ledger)   # replaces transformer_builder loader
    service.uninstall(model_ledger) # restores safetensors loader
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from engine.gguf.ic_lora_common import IcLoraEntry

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# GGUF tensor dequantization                                          #
# ------------------------------------------------------------------ #
#
# NOTE (a-0 fix): the bf16-path dequant now delegates to the FAITHFUL,
# numerically-validated kernels in quant_service.dequantize_ggml_tensor
# (VERIFICATION_LOG §1.3). The former hand-rolled Q4_K/Q6_K kernels here were
# admittedly "simplified" (wrong nibble-interleave and 6-bit scale/min
# unpacking) and produced noise-level weights on this bf16 GGUFStateDictLoader
# path. quant_service imports no ltx_core at module top (only stdlib + torch),
# so this top-level import cannot form a circular import with loader_service.
from engine.gguf.quant_service import dequantize_ggml_tensor


def _dequantize_tensor(
    data: torch.Tensor,
    ggml_type: int,
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a raw GGUF tensor to a floating point torch tensor.

    Thin wrapper over the faithful per-tensor kernel in quant_service. Float
    types (F32/F16/BF16) reinterpret+reshape+cast identically to before; the
    quantized types (Q8_0/Q4_K/Q6_K/…) now use the validated kernels instead of
    the old simplified ones. Output dtype behaviour is unchanged (bf16 default).
    """
    # dequantize_ggml_tensor expects a FLAT (1-D) input — its production caller
    # in quant_service flattens explicitly via reshape(-1). GGUFReader's
    # tensor.data can be MULTI-DIM for quantised types (e.g. (nrows, row_bytes)),
    # which would break the per-block reshape inside the kernels. The caller
    # hands us an owned .copy() (contiguous), so reshape(-1) is safe. For float
    # types the flatten is value-neutral: 1-D view(float_dtype) → view(shape)
    # reinterprets the same bytes (identity when data already carries that
    # dtype; byte-reinterpretation when it arrives as uint8 — both correct).
    return dequantize_ggml_tensor(data.reshape(-1), ggml_type, shape, dtype)


# ------------------------------------------------------------------ #
# StateDictLoader implementation                                      #
# ------------------------------------------------------------------ #

class GGUFStateDictLoader:
    """Implements the ltx-core StateDictLoader protocol for GGUF files.

    Reads transformer weights from a GGUF file, dequantizes them to
    bfloat16, and returns a StateDict compatible with SingleGPUModelBuilder.
    """

    def __init__(
        self,
        gguf_path: str,
        target_dtype: torch.dtype = torch.bfloat16,
        ic_loras: list[IcLoraEntry] | None = None,
    ) -> None:
        self.gguf_path = gguf_path
        self.target_dtype = target_dtype
        # IC-LoRA spike: (path, strength, audio_strength) entries fused in-place
        # into the base state-dict just before it is returned from load(). Empty
        # by default → load() is byte-identical to the historical behaviour.
        self.ic_loras: list[IcLoraEntry] = list(ic_loras or [])

    def metadata(self, path: str) -> dict:
        """Extract model config from GGUF metadata."""
        import gguf as gguf_lib
        reader = gguf_lib.GGUFReader(self.gguf_path, mode="r")

        # Look for config JSON in GGUF metadata fields.
        for field in reader.fields.values():
            name = field.name
            if name in ("config", "ltx.config", "general.config"):
                try:
                    raw = bytes(field.parts[-1])
                    return json.loads(raw.decode("utf-8"))
                except Exception:
                    continue

        # Fallback: try reading config from the safetensors checkpoint.
        logger.warning(
            "No config found in GGUF metadata for %s — "
            "falling back to safetensors checkpoint config",
            self.gguf_path,
        )
        try:
            import safetensors
            with safetensors.safe_open(path, framework="pt") as f:
                meta = f.metadata()
                if meta and "config" in meta:
                    return json.loads(meta["config"])
        except Exception as exc:
            logger.warning("Safetensors config fallback failed: %s", exc)

        raise RuntimeError(
            f"Could not find model config in GGUF file {self.gguf_path} "
            f"or safetensors checkpoint {path}. "
            "Ensure the GGUF was exported with embedded config metadata."
        )

    def load(
        self,
        path: str | list[str],
        sd_ops: Any = None,
        device: torch.device | None = None,
    ) -> Any:
        """Load and dequantize transformer weights from GGUF file."""
        from ltx_core.loader.single_gpu_model_builder import StateDict

        device = device or torch.device("cpu")
        logger.info("Loading GGUF transformer from %s", self.gguf_path)

        import gguf as gguf_lib
        reader = gguf_lib.GGUFReader(self.gguf_path, mode="r")

        state_dict: dict[str, torch.Tensor] = {}
        total_params = 0

        for tensor in reader.tensors:
            name = tensor.name
            shape = tuple(reversed(tensor.shape.tolist()))
            ggml_type = tensor.tensor_type.value

            raw_data = torch.from_numpy(tensor.data.copy())

            try:
                weight = _dequantize_tensor(raw_data, ggml_type, shape, self.target_dtype)
            except Exception as exc:
                logger.warning("Failed to dequantize %s (%s) — skipping", name, exc)
                continue

            # Move to target device if not CPU.
            if device.type != "cpu":
                weight = weight.to(device)

            state_dict[name] = weight
            total_params += weight.numel()

        # Apply sd_ops key remapping if provided.
        if sd_ops is not None:
            try:
                wrapped = StateDict(
                    sd=state_dict,
                    device=device,
                    size=sum(t.numel() * t.element_size() for t in state_dict.values()),
                    dtype={t.dtype for t in state_dict.values()},
                )
                from ltx_core.loader.sd_ops import apply_sd_ops
                wrapped = apply_sd_ops(wrapped, sd_ops)
                state_dict = wrapped.sd
            except Exception as exc:
                logger.warning("sd_ops application failed: %s — using raw keys", exc)

        logger.info(
            "GGUF load complete: %d tensors, %.1fM params from %s",
            len(state_dict),
            total_params / 1e6,
            Path(self.gguf_path).name,
        )

        base_sd = StateDict(
            sd=state_dict,
            device=device,
            size=sum(t.numel() * t.element_size() for t in state_dict.values()),
            dtype={t.dtype for t in state_dict.values()},
        )

        # IC-LoRA in-place fuse (spike). No-op when self.ic_loras is empty →
        # byte-identical to the historical return above.
        if self.ic_loras:
            base_sd = self._fuse_ic_loras(base_sd)

        return base_sd

    def _fuse_ic_loras(self, base_sd: Any) -> Any:
        """Load each configured LoRA safetensors and fuse it into ``base_sd`` in
        place (RAM cost = one per-key fp32 delta transient, not a second full copy).

        The wheel's apply_loras is deliberately NOT used: it matmuls the LoRA
        delta in bf16 on CPU (fuse_loras.py _prepare_deltas), and on CPUs without
        AVX512-BF16/AMX torch's bf16 matmul falls into a ~54x-slower-than-fp32
        path (measured: ~19 min per fuse vs 68 s no-LoRA load). This loop is
        mathematically IDENTICAL to the wheel's bf16 route
        (_prepare_deltas + _fuse_delta_with_bfloat16): both compute
        W + (B*strength) @ A per key; the only difference is where the single
        bf16 rounding happens — the wheel rounds the matmul products in-loop
        (bf16 matmul), we matmul in fp32 and round ONCE when casting the delta
        to the weight dtype before the in-place add.
        """
        # Shared front half (load + LTXV_LORA_COMFY_RENAMING_MAP rename +
        # lora_A/lora_B pairing) with the forward-time path; only the in-place
        # fp32 fuse below is bf16-path specific and stays UNCHANGED.
        from engine.gguf.ic_lora_common import load_ic_lora_pairs, strength_for_prefix

        total_lora_keys = 0
        total_delta_keys = 0

        for path, strength, audio_strength, pairs in load_ic_lora_pairs(self.ic_loras):
            n_keys = 2 * len(pairs)  # each pair = one lora_A + one lora_B key
            n_delta = 0
            n_resolved = 0

            # Per-key fp32 fuse. Multiple LoRAs hitting the same key accumulate
            # sequentially (same net result as the wheel's summed-deltas path).
            # Transient memory = ONE fp32 delta at a time (largest LTX-2.3 layer
            # is a few hundred MB in fp32), freed right after the in-place add.
            for prefix, lora_a, lora_b in pairs:
                weight_key = prefix + ".weight"
                weight = base_sd.sd.get(weight_key)
                if weight is None:
                    continue  # counted via n_resolved; 0 total → loud WARN below
                n_resolved += 1
                # audio_strength=0 on an audio-axis key → zero delta; skip the
                # matmul entirely (mathematically identical to add_(0)).
                eff = strength_for_prefix(prefix, strength, audio_strength)
                if eff == 0.0:
                    continue
                if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    raise RuntimeError(
                        f"IC-LoRA fuse: unsupported model weight dtype {weight.dtype} "
                        f"for {weight_key} (expected bf16/f16/f32)"
                    )
                # B:(out,r) @ A:(r,in) → delta:(out,in), fp32 matmul (fast CPU path).
                delta = torch.matmul(
                    lora_b.to(torch.float32) * eff, lora_a.to(torch.float32)
                )
                if delta.shape != weight.shape:
                    raise RuntimeError(
                        f"IC-LoRA fuse: delta shape {tuple(delta.shape)} != weight "
                        f"shape {tuple(weight.shape)} for {weight_key} "
                        f"(A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)})"
                    )
                weight.add_(delta.to(weight.dtype))
                del delta
                n_delta += 1

            total_lora_keys += n_keys
            total_delta_keys += n_delta
            extra = ""
            if audio_strength is not None:
                extra = (
                    f", audio_strength={audio_strength:.3f}, "
                    f"muted={n_resolved - n_delta} keys"
                )
            logger.info(
                "IC-LoRA %s: %d keys, %d model weights fused (strength=%.3f%s)",
                Path(path).name, n_keys, n_delta, strength, extra,
            )
            if n_resolved == 0:
                logger.warning(
                    "IC-LoRA %s matched 0 model weights — LoRA/model KEY-FORMAT "
                    "MISMATCH; the fuse was a no-op. Check the renaming map.",
                    Path(path).name,
                )

        logger.info(
            "IC-LoRA fuse complete: %d LoRA(s), %d total lora keys, %d total model deltas",
            len(self.ic_loras), total_lora_keys, total_delta_keys,
        )
        return base_sd


# ------------------------------------------------------------------ #
# Service                                                             #
# ------------------------------------------------------------------ #

class GGUFLoaderService:
    """Installs a GGUF-based transformer loader into a ModelLedger.

    Replaces the transformer_builder's model_loader with a GGUF-aware
    loader while leaving all other builders (VAE, text encoder etc.)
    untouched — they still load from the original safetensors checkpoint.
    """

    def __init__(
        self, gguf_path: str, ic_loras: list[IcLoraEntry] | None = None
    ) -> None:
        self.gguf_path = gguf_path
        self._original_loader: Any = None
        # IC-LoRA (path, strength, audio_strength) entries forwarded to the
        # GGUFStateDictLoader for in-place fuse. Empty by default → historical
        # behaviour unchanged.
        self.ic_loras: list[IcLoraEntry] = list(ic_loras or [])

    def install(self, model_ledger: Any) -> None:
        """Replace transformer_builder loader with GGUF loader."""
        if not Path(self.gguf_path).exists():
            raise FileNotFoundError(f"GGUF file not found: {self.gguf_path}")

        if not hasattr(model_ledger, "transformer_builder"):
            logger.warning("ModelLedger has no transformer_builder — GGUF install skipped")
            return

        from dataclasses import replace as dc_replace

        builder = model_ledger.transformer_builder
        self._original_loader = builder.model_loader

        gguf_loader = GGUFStateDictLoader(
            gguf_path=self.gguf_path,
            target_dtype=model_ledger.dtype,
            ic_loras=self.ic_loras,
        )

        # SingleGPUModelBuilder is frozen so we use replace().
        new_builder = dc_replace(builder, model_loader=gguf_loader)
        model_ledger.transformer_builder = new_builder

        logger.info(
            "GGUFLoaderService installed: transformer will load from %s",
            Path(self.gguf_path).name,
        )

    def uninstall(self, model_ledger: Any) -> None:
        """Restore original safetensors loader."""
        if self._original_loader is None:
            return
        if not hasattr(model_ledger, "transformer_builder"):
            return

        from dataclasses import replace as dc_replace
        builder = model_ledger.transformer_builder
        new_builder = dc_replace(builder, model_loader=self._original_loader)
        model_ledger.transformer_builder = new_builder
        self._original_loader = None
        logger.info("GGUFLoaderService uninstalled")


def build_gguf_loader_service(gguf_path: str) -> GGUFLoaderService | None:
    """Factory: returns None if no GGUF path configured."""
    if not gguf_path:
        return None
    return GGUFLoaderService(gguf_path=gguf_path)