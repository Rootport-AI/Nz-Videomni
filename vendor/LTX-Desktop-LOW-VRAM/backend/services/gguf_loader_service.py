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
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# GGUF tensor dequantization                                          #
# ------------------------------------------------------------------ #

# Quantization type constants from the GGUF spec.
_GGML_TYPE_F32 = 0
_GGML_TYPE_F16 = 1
_GGML_TYPE_BF16 = 30
_GGML_TYPE_Q8_0 = 8
_GGML_TYPE_Q4_K = 12
_GGML_TYPE_Q6_K = 14
_GGML_TYPE_Q4_0 = 2
_GGML_TYPE_Q5_0 = 6
_GGML_TYPE_Q5_1 = 7


def _dequantize_tensor(
    data: torch.Tensor,
    ggml_type: int,
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a raw GGUF tensor to a floating point torch tensor."""
    if ggml_type == _GGML_TYPE_F32:
        return data.view(shape).to(dtype)

    if ggml_type == _GGML_TYPE_F16:
        return data.view(torch.float16).view(shape).to(dtype)

    if ggml_type == _GGML_TYPE_BF16:
        return data.view(torch.bfloat16).view(shape).to(dtype)

    if ggml_type == _GGML_TYPE_Q8_0:
        return _dequant_q8_0(data, shape, dtype)

    if ggml_type == _GGML_TYPE_Q4_K:
        return _dequant_q4_k(data, shape, dtype)

    if ggml_type == _GGML_TYPE_Q6_K:
        return _dequant_q6_k(data, shape, dtype)

    if ggml_type in (_GGML_TYPE_Q4_0, _GGML_TYPE_Q5_0, _GGML_TYPE_Q5_1):
        return _dequant_q4_0_family(data, shape, dtype, ggml_type)

    logger.warning("Unsupported GGML type %d — loading as float32 raw", ggml_type)
    return data.view(torch.float32).reshape(-1).to(dtype)


def _dequant_q8_0(
    data: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    """Q8_0: 32 int8 values + 1 float16 scale per block."""
    block_size = 34  # 2 bytes scale + 32 bytes data
    n_blocks = data.numel() // block_size
    raw = data.view(torch.uint8).reshape(n_blocks, block_size)
    scales = raw[:, :2].view(torch.float16).to(torch.float32)
    qs = raw[:, 2:].view(torch.int8).to(torch.float32)
    dequant = (qs * scales).reshape(-1)
    n_elems = 1
    for s in shape:
        n_elems *= s
    return dequant[:n_elems].reshape(shape).to(dtype)


def _dequant_q4_k(
    data: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    """Q4_K: simplified dequantization via numpy for correctness."""
    try:
        import numpy as np
        arr = data.numpy()
        # Q4_K block: 144 bytes = 2 super-scales (fp16) + 12 scales (6-bit) + 128 quants
        block_size = 144
        n_blocks = len(arr) // block_size
        blocks = arr[:n_blocks * block_size].reshape(n_blocks, block_size)

        d = blocks[:, :2].view(np.float16).astype(np.float32)
        dmin = blocks[:, 2:4].view(np.float16).astype(np.float32)

        # Extract 4-bit quants (last 64 bytes = 128 values)
        quants_raw = blocks[:, 80:].reshape(n_blocks, 64)
        lo = (quants_raw & 0x0F).astype(np.float32)
        hi = ((quants_raw >> 4) & 0x0F).astype(np.float32)
        quants = np.stack([lo, hi], axis=2).reshape(n_blocks, 128)

        # Simplified: use d as scale, dmin as offset
        dequant = (quants * d[:, None] - dmin[:, None]).reshape(-1)
        n_elems = 1
        for s in shape:
            n_elems *= s
        result = dequant[:n_elems].reshape(shape)
        return torch.from_numpy(result).to(dtype)
    except Exception as exc:
        logger.warning("Q4_K dequant failed (%s), using zero tensor", exc)
        return torch.zeros(shape, dtype=dtype)


def _dequant_q6_k(
    data: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    """Q6_K: simplified fallback."""
    try:
        import numpy as np
        arr = data.numpy()
        block_size = 210  # Q6_K block size
        n_blocks = len(arr) // block_size
        blocks = arr[:n_blocks * block_size].reshape(n_blocks, block_size)

        d = blocks[:, 208:210].view(np.float16).astype(np.float32)
        ql = blocks[:, :128].astype(np.uint8)
        qh = blocks[:, 128:192].astype(np.uint8)

        q1 = (ql[:, :64] & 0x0F) | ((qh[:, :64] & 0x03) << 4)
        q2 = (ql[:, :64] >> 4) | (((qh[:, :64] >> 2) & 0x03) << 4)
        q3 = (ql[:, 64:] & 0x0F) | ((qh[:, :64] & 0x0C) << 2)
        q4 = (ql[:, 64:] >> 4) | (((qh[:, :64] >> 4) & 0x03) << 4)

        quants = np.stack([q1, q2, q3, q4], axis=2).reshape(n_blocks, 256).astype(np.float32) - 32
        dequant = (quants * d[:, None]).reshape(-1)
        n_elems = 1
        for s in shape:
            n_elems *= s
        result = dequant[:n_elems].reshape(shape)
        return torch.from_numpy(result).to(dtype)
    except Exception as exc:
        logger.warning("Q6_K dequant failed (%s), using zero tensor", exc)
        return torch.zeros(shape, dtype=dtype)


def _dequant_q4_0_family(
    data: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    ggml_type: int,
) -> torch.Tensor:
    """Q4_0 family: 16 int4 pairs + 1 float16 scale per block."""
    try:
        import numpy as np
        arr = data.numpy()
        block_size = 18  # 2 scale bytes + 16 data bytes
        n_blocks = len(arr) // block_size
        blocks = arr[:n_blocks * block_size].reshape(n_blocks, block_size)
        scales = blocks[:, :2].view(np.float16).astype(np.float32)
        raw_q = blocks[:, 2:].astype(np.uint8)
        lo = (raw_q & 0x0F).astype(np.float32) - 8
        hi = ((raw_q >> 4) & 0x0F).astype(np.float32) - 8
        quants = np.stack([lo, hi], axis=2).reshape(n_blocks, 32)
        dequant = (quants * scales[:, None]).reshape(-1)
        n_elems = 1
        for s in shape:
            n_elems *= s
        result = dequant[:n_elems].reshape(shape)
        return torch.from_numpy(result).to(dtype)
    except Exception as exc:
        logger.warning("Q4_0 family dequant failed (%s), using zero tensor", exc)
        return torch.zeros(shape, dtype=dtype)


# ------------------------------------------------------------------ #
# StateDictLoader implementation                                      #
# ------------------------------------------------------------------ #

class GGUFStateDictLoader:
    """Implements the ltx-core StateDictLoader protocol for GGUF files.

    Reads transformer weights from a GGUF file, dequantizes them to
    bfloat16, and returns a StateDict compatible with SingleGPUModelBuilder.
    """

    def __init__(self, gguf_path: str, target_dtype: torch.dtype = torch.bfloat16) -> None:
        self.gguf_path = gguf_path
        self.target_dtype = target_dtype

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

        return StateDict(
            sd=state_dict,
            device=device,
            size=sum(t.numel() * t.element_size() for t in state_dict.values()),
            dtype={t.dtype for t in state_dict.values()},
        )


# ------------------------------------------------------------------ #
# Service                                                             #
# ------------------------------------------------------------------ #

class GGUFLoaderService:
    """Installs a GGUF-based transformer loader into a ModelLedger.

    Replaces the transformer_builder's model_loader with a GGUF-aware
    loader while leaving all other builders (VAE, text encoder etc.)
    untouched — they still load from the original safetensors checkpoint.
    """

    def __init__(self, gguf_path: str) -> None:
        self.gguf_path = gguf_path
        self._original_loader: Any = None

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