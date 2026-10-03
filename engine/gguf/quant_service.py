"""Per-layer GGUF quantization service.

This service keeps transformer weights compressed in VRAM (never dequantized whole at
load time) and dequantizes one layer at a time during the forward pass.

VRAM comparison for LTX-2.3 (22B parameters):
  BF16 (2 bytes/param):  ~44 GB VRAM
  Q8_0 (1.06 bytes):     ~23 GB VRAM
  Q6_K (0.82 bytes):     ~18 GB VRAM
  Q4_K_M (0.56 bytes):   ~12 GB VRAM

Supported tensor types: ``SUPPORTED_GGML_TYPES`` (F32/F16/BF16 and Q8_0, Q4_K,
Q5_K, Q6_K). A GGUF holding any other type is refused with a ValueError when its
state dict is read, before the transformer is built.

At inference: +1 layer BF16 (~100-200 MB peak overhead, freed after each matmul).

Integration: the pipeline's ``_install_gguf`` builds a GGUFQuantLoaderService and
calls install(model_ledger); per-layer dequant is then enabled automatically.

Mechanism:
1. module_ops mutator converts nn.Linear.weight from parameter to meta buffer.
   This sidesteps PyTorch's "float-only parameter" restriction for quantized tensors.
2. GGUFQuantStateDictLoader.load():
   - Float-type GGUF tensors (BF16/F16/F32): converted to bfloat16 with correct shape.
   - Quantized tensors (Q8_0/Q4_K/Q5_K/Q6_K): wrapped in GGMLQuantizedTensor which stores raw
     uint8 bytes but reports the float shape via @property shape override. This passes
     load_state_dict's shape check (shape matches the meta buffer) and assigns the
     GGMLQuantizedTensor directly as the buffer value.
3. load_state_dict(strict=False, assign=True) populates all buffers. Float tensors load
   normally. Quantized buffers receive GGMLQuantizedTensor instances.
4. model.to(device) moves GGMLQuantizedTensors to GPU. The overridden .to() method
   preserves the GGMLQuantizedTensor subclass and its metadata through device moves.
5. ggml_linear_forward() checks isinstance(weight, GGMLQuantizedTensor), extracts raw
   uint8 bytes via weight.as_subclass(torch.Tensor).view(torch.uint8), dequantizes
   on-the-fly, runs F.linear(), frees the dequantised weight.
"""

from __future__ import annotations

import logging
import math
import types
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import torch

from engine.gguf import dequant_triton
from engine.gguf.ic_lora_common import IC_LORA_SPECS_ATTR as _IC_LORA_SPECS_ATTR

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# GGML quantisation type constants (from GGUF spec)
# ──────────────────────────────────────────────────────────────────────────────

_GGML_F32   = 0
_GGML_F16   = 1
_GGML_Q8_0  = 8
_GGML_Q4_K  = 12
_GGML_Q5_K  = 13
_GGML_Q6_K  = 14
_GGML_BF16  = 30

# The ONE list of tensor types this engine can read; every other type is refused
# (``dequantize_ggml_tensor`` and ``GGUFQuantStateDictLoader.load`` both read it).
SUPPORTED_GGML_TYPES = {
    _GGML_F32: "F32",
    _GGML_F16: "F16",
    _GGML_BF16: "BF16",
    _GGML_Q8_0: "Q8_0",
    _GGML_Q4_K: "Q4_K",
    _GGML_Q5_K: "Q5_K",
    _GGML_Q6_K: "Q6_K",
}


def _unsupported_type_message(ggml_type: int) -> str:
    supported = ", ".join(f"{name}({t})" for t, name in SUPPORTED_GGML_TYPES.items())
    return f"unsupported GGML tensor type {ggml_type}; supported types: {supported}"

# The three K-quant types the fused Triton kernels cover. Q8_0 keeps the eager
# path unconditionally — Q4_K/Q5_K/Q6_K are 100% of the quantised tensors in the
# shipped DiT and Gemma GGUFs, so Q8_0 is not worth a kernel.
_TRITON_TYPES = (_GGML_Q4_K, _GGML_Q5_K, _GGML_Q6_K)

# ──────────────────────────────────────────────────────────────────────────────
# Per-type dequantisation (pure PyTorch, runs on CPU or CUDA; Q4_K/Q5_K/Q6_K
# (``_TRITON_TYPES``) can be routed to the fused Triton kernels in dequant_triton).
# Only the types in ``SUPPORTED_GGML_TYPES``; each matches gguf-py's
# ``gguf.quants`` exactly (tests/test_gguf_dequant_reference.py). Any other type
# raises ValueError.
# Based on city96/ComfyUI-GGUF/dequant.py — standalone, no C++ kernels
# ──────────────────────────────────────────────────────────────────────────────

def dequantize_ggml_tensor(
    raw: torch.Tensor,
    ggml_type: int,
    original_shape: tuple[int, ...],
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a raw uint8 GGUF tensor to a floating-point tensor.

    raw: 1-D uint8 tensor (packed bytes, on any device).
    ggml_type: GGML quantisation type constant.
    original_shape: (out_features, in_features) — the unquantised shape.
    out_dtype: target dtype for the returned tensor.

    Raises ValueError for a type outside ``SUPPORTED_GGML_TYPES``.
    """
    if ggml_type == _GGML_F32:
        return raw.view(torch.float32).view(original_shape).to(out_dtype)
    if ggml_type == _GGML_F16:
        return raw.view(torch.float16).view(original_shape).to(out_dtype)
    if ggml_type == _GGML_BF16:
        return raw.view(torch.bfloat16).view(original_shape).to(out_dtype)
    if ggml_type == _GGML_Q8_0:
        return _dequant_q8_0(raw, original_shape, out_dtype)
    # Fused Triton dequant — opt-in per job, and a pure accelerator: it returns
    # None for every failure mode (no Triton, kernel raised, first-call
    # bit-exactness check failed) after latching itself off for the rest of the
    # job, and the eager kernels below run instead. Guards, in order: only the
    # three K-quant types have kernels; `raw.is_cuda` because Gemma's embedding
    # dequantises on CPU and a Triton launch there would raise once per tensor
    # just to be caught; only bf16 output is wired up; and `enabled()` is False
    # unless this job asked for it.
    if (
        ggml_type in _TRITON_TYPES
        and raw.is_cuda
        and out_dtype is torch.bfloat16
        and dequant_triton.enabled()
    ):
        fused = dequant_triton.dequant(raw, ggml_type, original_shape, out_dtype)
        if fused is not None:
            return fused

    if ggml_type == _GGML_Q4_K:
        return _dequant_q4_k(raw, original_shape, out_dtype)
    if ggml_type == _GGML_Q5_K:
        return _dequant_q5_k(raw, original_shape, out_dtype)
    if ggml_type == _GGML_Q6_K:
        return _dequant_q6_k(raw, original_shape, out_dtype)

    raise ValueError(_unsupported_type_message(ggml_type))


def _n_elems(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _dequant_q8_0(raw: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Q8_0: blocks of 34 bytes — 2-byte fp16 scale + 32 int8 values."""
    BLOCK = 34
    data = raw.view(torch.uint8)
    n_blocks = data.numel() // BLOCK
    blocks = data.reshape(n_blocks, BLOCK)
    scale = blocks[:, :2].reshape(-1, 2).view(torch.float16).to(torch.float32)  # (n_blocks, 1)
    qs = blocks[:, 2:].view(torch.int8).to(torch.float32)                       # (n_blocks, 32)
    out = (qs * scale).reshape(-1)
    ne = _n_elems(shape)
    return out[:ne].reshape(shape).to(dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Q_K dequant kernels — faithful torch/GPU port of gguf.quants (Q4_K/Q5_K/Q6_K).
#
# Ported element-for-element from `gguf.quants.{Q4_K,Q5_K,Q6_K}.dequantize_blocks`
# (the reference the numerical verification in VERIFICATION_LOG §1.1 compared
# against), so results match the reference by construction. The nibble
# interleave order (the reference's (n,-1,1,32) >> [0,4], not a flat
# cat([lo,hi])) and the 6-bit scale/min unpacking both follow the reference; a
# hand-derived layout produces noise-level outputs (VERIFICATION_LOG §1.1,
# bug #4). Everything below runs on the input tensor's device (the GPU in the
# per-layer forward).
# ──────────────────────────────────────────────────────────────────────────────

_QK_K = 256
_K_SCALE_SIZE = 12


def _q_get_scale_min(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Port of Q4_K.get_scale_min. scales: (n,12) uint8 → (sc(n,8), min(n,8))."""
    n = scales.shape[0]
    s = scales.to(torch.int32).reshape(n, 3, 4)
    d   = s[:, 0:1, :]   # (n,1,4)
    m   = s[:, 1:2, :]
    m_d = s[:, 2:3, :]
    sc  = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)   # (n,1,8)
    mn  = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc.reshape(n, 8).to(torch.float32), mn.reshape(n, 8).to(torch.float32)


def _dequant_q4_k(raw: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Q4_K: 144-byte blocks, 256 weights. Port of gguf.quants.Q4_K."""
    BLOCK = 144
    data = raw.view(torch.uint8)
    n = data.numel() // BLOCK
    blocks = data.reshape(n, BLOCK)

    d    = blocks[:, 0:2].reshape(n, 2).view(torch.float16).to(torch.float32).reshape(n, 1)
    dmin = blocks[:, 2:4].reshape(n, 2).view(torch.float16).to(torch.float32).reshape(n, 1)
    scales = blocks[:, 4:4 + _K_SCALE_SIZE]                      # (n,12)
    qs = blocks[:, 4 + _K_SCALE_SIZE:].to(torch.int32)           # (n,128)

    sc, mn = _q_get_scale_min(scales)                           # (n,8),(n,8)
    dd = (d * sc).reshape(n, -1, 1)                              # (n,8,1)
    dm = (dmin * mn).reshape(n, -1, 1)                           # (n,8,1)

    # (n,-1,1,32) >> [0,4]  → interleave low/high nibble of each byte
    shifts = torch.tensor([0, 4], dtype=torch.int32, device=raw.device).reshape(1, 1, 2, 1)
    q = (qs.reshape(n, -1, 1, 32) >> shifts) & 0x0F             # (n,4,2,32)
    q = q.reshape(n, -1, 32).to(torch.float32)                  # (n,8,32)

    out = (dd * q - dm).reshape(n, _QK_K)
    ne = _n_elems(shape)
    return out.reshape(-1)[:ne].reshape(shape).to(dtype)


def _dequant_q5_k(raw: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Q5_K: 176-byte blocks, 256 weights. Port of gguf.quants.Q5_K."""
    BLOCK = 176
    data = raw.view(torch.uint8)
    n = data.numel() // BLOCK
    blocks = data.reshape(n, BLOCK)

    d    = blocks[:, 0:2].reshape(n, 2).view(torch.float16).to(torch.float32).reshape(n, 1)
    dmin = blocks[:, 2:4].reshape(n, 2).view(torch.float16).to(torch.float32).reshape(n, 1)
    scales = blocks[:, 4:4 + _K_SCALE_SIZE]
    rest = blocks[:, 4 + _K_SCALE_SIZE:]
    qh = rest[:, :_QK_K // 8].to(torch.int32)                   # (n,32)
    qs = rest[:, _QK_K // 8:].to(torch.int32)                   # (n,128)

    sc, mn = _q_get_scale_min(scales)
    dd = (d * sc).reshape(n, -1, 1)
    dm = (dmin * mn).reshape(n, -1, 1)

    ql_shifts = torch.tensor([0, 4], dtype=torch.int32, device=raw.device).reshape(1, 1, 2, 1)
    ql = (qs.reshape(n, -1, 1, 32) >> ql_shifts) & 0x0F         # (n,4,2,32)
    ql = ql.reshape(n, -1, 32)                                  # (n,8,32)

    qh_shifts = torch.arange(8, dtype=torch.int32, device=raw.device).reshape(1, 1, 8, 1)
    qhb = (qh.reshape(n, -1, 1, 32) >> qh_shifts) & 0x01        # (n,1,8,32)
    qhb = qhb.reshape(n, -1, 32)                                # (n,8,32)

    q = (ql | (qhb << 4)).to(torch.float32)
    out = (dd * q - dm).reshape(n, _QK_K)
    ne = _n_elems(shape)
    return out.reshape(-1)[:ne].reshape(shape).to(dtype)


def _dequant_q6_k(raw: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Q6_K: 210-byte blocks, 256 weights. Port of gguf.quants.Q6_K."""
    BLOCK = 210
    data = raw.view(torch.uint8)
    n = data.numel() // BLOCK
    blocks = data.reshape(n, BLOCK)

    ql = blocks[:, :_QK_K // 2].to(torch.int32)                 # (n,128)
    qh = blocks[:, _QK_K // 2:_QK_K // 2 + _QK_K // 4].to(torch.int32)  # (n,64)
    scales = blocks[:, _QK_K // 2 + _QK_K // 4:
                       _QK_K // 2 + _QK_K // 4 + _QK_K // 16]   # (n,16) int8
    d = blocks[:, -2:].reshape(n, 2).view(torch.float16).to(torch.float32).reshape(n, 1)

    scales = scales.view(torch.int8).to(torch.float32)          # (n,16)
    dd = (d * scales).reshape(n, _QK_K // 16, 1)                # (n,16,1)

    ql_shifts = torch.tensor([0, 4], dtype=torch.int32, device=raw.device).reshape(1, 1, 2, 1)
    qlv = (ql.reshape(n, -1, 1, 64) >> ql_shifts) & 0x0F        # (n,2,2,64)
    qlv = qlv.reshape(n, -1, 32)                                # (n,8,32)

    qh_shifts = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=raw.device).reshape(1, 1, 4, 1)
    qhv = (qh.reshape(n, -1, 1, 32) >> qh_shifts) & 0x03        # (n,2,4,32)
    qhv = qhv.reshape(n, -1, 32)                                # (n,8,32)

    q = ((qlv | (qhv << 4)).to(torch.int8) - 32).to(torch.int8)
    q = q.reshape(n, _QK_K // 16, -1).to(torch.float32)         # (n,16,16)
    out = (dd * q).reshape(n, _QK_K)
    ne = _n_elems(shape)
    return out.reshape(-1)[:ne].reshape(shape).to(dtype)


# ──────────────────────────────────────────────────────────────────────────────
# GGMLQuantizedTensor — tensor subclass that stores raw uint8 GGUF bytes but
# reports the dequantised float shape via @property override.
#
# This solves two problems:
#   1. load_state_dict shape check: sees float_shape (e.g. [4096, 4096]) not the
#      raw byte count, so the shape matches the model's meta buffer.
#   2. nn.Parameter float-only restriction: weight is registered as a buffer
#      (not a parameter) by the module_ops mutator, so uint8 dtype is accepted.
# ──────────────────────────────────────────────────────────────────────────────

class GGMLQuantizedTensor(torch.Tensor):
    """Tensor subclass wrapping raw GGUF uint8 bytes with a float-shaped interface.

    The underlying storage is 1D uint8 (the packed quantised bytes).
    `shape`, `size()`, `dim()`, `numel()` all return values consistent with the
    original dequantised float tensor shape.

    After model.to(device), the .to() override recreates the subclass wrapper
    so that GGMLQuantizedTensor identity is preserved on the GPU.
    """

    @staticmethod
    def __new__(
        cls,
        raw_bytes: torch.Tensor,       # 1D uint8, flat packed bytes
        ggml_type: int,
        float_shape: tuple[int, ...],
    ) -> "GGMLQuantizedTensor":
        instance = torch.Tensor._make_subclass(cls, raw_bytes)
        instance._ggml_type = ggml_type
        instance._float_shape = float_shape
        return instance

    @property
    def shape(self) -> torch.Size:          # type: ignore[override]
        return torch.Size(self._float_shape)

    def size(self, dim: int | None = None) -> torch.Size | int:  # type: ignore[override]
        s = torch.Size(self._float_shape)
        return s if dim is None else s[dim]

    def dim(self) -> int:                   # type: ignore[override]
        return len(self._float_shape)

    def numel(self) -> int:                 # type: ignore[override]
        return math.prod(self._float_shape)

    def to(self, *args: Any, **kwargs: Any) -> "GGMLQuantizedTensor":
        moved = super().to(*args, **kwargs)
        if isinstance(moved, GGMLQuantizedTensor):
            # torch may preserve the subclass type but strip the Python-level
            # attributes (_ggml_type / _float_shape). Re-attach them before
            # returning, otherwise forward()'s w._ggml_type access raises
            # AttributeError.
            moved._ggml_type = self._ggml_type
            moved._float_shape = self._float_shape
            return moved
        # Reconstruct after device/dtype move — preserve quant metadata
        return GGMLQuantizedTensor.__new__(
            GGMLQuantizedTensor,
            moved.view(torch.uint8),
            self._ggml_type,
            self._float_shape,
        )


# ──────────────────────────────────────────────────────────────────────────────
# State dict loader — returns properly-typed tensors for load_state_dict
# ──────────────────────────────────────────────────────────────────────────────

class GGUFQuantStateDictLoader:
    """Loads GGUF tensors for per-layer dequantisation.

    Float-type tensors (BF16/F16/F32): converted to bfloat16 with the correct
    float shape — these load via load_state_dict normally.

    Quantised tensors (Q8_0/Q4_K/Q5_K/Q6_K): wrapped in GGMLQuantizedTensor which
    stores raw uint8 bytes but reports the float shape. The module_ops mutator
    registers Linear.weight as a buffer (not a parameter) so that the uint8
    dtype is accepted by load_state_dict(assign=True).
    """

    def __init__(self, gguf_path: str) -> None:
        self.gguf_path = gguf_path

    def metadata(self, path: str) -> dict:
        """Extract model config from GGUF metadata (or fall back to safetensors)."""
        import json
        import gguf as gguf_lib
        reader = gguf_lib.GGUFReader(self.gguf_path, mode="r")
        for field in reader.fields.values():
            if field.name in ("config", "ltx.config", "general.config"):
                try:
                    raw = bytes(field.parts[-1])
                    return json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
        # Fallback: safetensors checkpoint
        logger.warning("No config in GGUF %s — trying safetensors fallback", self.gguf_path)
        try:
            import safetensors
            with safetensors.safe_open(path, framework="pt") as f:
                meta = f.metadata()
                if meta and "config" in meta:
                    return json.loads(meta["config"])
        except Exception as exc:
            logger.warning("Safetensors config fallback failed: %s", exc)
        raise RuntimeError(
            f"Could not find model config in GGUF file {self.gguf_path}. "
            "Ensure the GGUF was exported with embedded config metadata."
        )

    def load(self, path: str | list[str], sd_ops: Any = None, device: torch.device | None = None) -> Any:
        """Load GGUF and return tensors ready for load_state_dict.

        Float-type entries → bfloat16 tensors with correct float shape.
        Quantised entries → GGMLQuantizedTensor (uint8 bytes, float shape).
        A tensor of a type outside ``SUPPORTED_GGML_TYPES`` → ValueError.
        """
        from ltx_core.loader.single_gpu_model_builder import StateDict
        import gguf as gguf_lib
        import numpy as np

        target_device = device or torch.device("cpu")
        logger.info("GGUF quant-load from %s → %s", Path(self.gguf_path).name, target_device)

        reader = gguf_lib.GGUFReader(self.gguf_path, mode="r")

        state_dict: dict[str, torch.Tensor] = {}
        n_float = 0
        n_quant = 0

        _FLOAT_TYPES = {_GGML_F32, _GGML_F16, _GGML_BF16}

        for tensor in reader.tensors:
            name = tensor.name
            ggml_type = tensor.tensor_type.value
            # Refuse an unsupported type here, before the bytes are copied and
            # the (heavy) transformer build runs — not at the first forward.
            if ggml_type not in SUPPORTED_GGML_TYPES:
                raise ValueError(
                    f"{self.gguf_path}: tensor {name!r}: "
                    f"{_unsupported_type_message(ggml_type)}"
                )
            # GGUF stores shape in reversed order (column-major)
            float_shape = tuple(reversed(tensor.shape.tolist()))

            # raw_np: numpy uint8 array of the packed bytes (may be 2D due to mmap layout).
            # Explicit owned copy off the GGUF memmap — breaks the np.memmap alias
            # entirely (city96 ComfyUI-GGUF #444 family). copy=True guarantees
            # raw_np never aliases the mmap, so the torch tensor below owns its bytes.
            raw_np = np.array(tensor.data, copy=True)
            raw_flat = torch.from_numpy(raw_np).reshape(-1)  # 1D uint8, owns memory

            if ggml_type in _FLOAT_TYPES:
                # Convert bytes directly to the proper float dtype, then to bfloat16
                if ggml_type == _GGML_F32:
                    t = raw_flat.view(torch.float32).view(float_shape).to(torch.bfloat16)
                elif ggml_type == _GGML_F16:
                    t = raw_flat.view(torch.float16).view(float_shape).to(torch.bfloat16)
                else:  # BF16
                    t = raw_flat.view(torch.bfloat16).view(float_shape)
                if target_device.type != "cpu":
                    t = t.to(target_device, non_blocking=True)
                state_dict[name] = t
                n_float += 1
            else:
                # Quantised: wrap raw bytes in GGMLQuantizedTensor
                qt = GGMLQuantizedTensor(raw_flat, ggml_type, float_shape)
                if target_device.type != "cpu":
                    qt = qt.to(target_device, non_blocking=True)
                state_dict[name] = qt
                n_quant += 1

        # NOTE: key remapping is intentionally not applied (``sd_ops`` is
        # ignored): the pinned ``ltx_core`` has no
        # ``ltx_core.loader.sd_ops.apply_sd_ops``, and the GGUF raw keys match
        # the model keys exactly (no remap needed), so we use the raw-key
        # state_dict directly.

        logger.info(
            "GGUF quant-load complete: %d float tensors (→ bf16), %d quantised tensors (GGMLQuantizedTensor)",
            n_float, n_quant,
        )
        return StateDict(
            sd=state_dict,
            device=target_device,
            size=sum(t.numel() for t in state_dict.values()),
            dtype={torch.uint8, torch.bfloat16},
        )


# ──────────────────────────────────────────────────────────────────────────────
# ModuleOps: convert Linear weights to buffers + patch forward() for dequant
# ──────────────────────────────────────────────────────────────────────────────

def _patch_linear_for_ggml_dequant(m: torch.nn.Linear) -> None:
    """Convert m.weight from parameter to meta buffer, then patch forward()."""
    # Remove from parameters (meta, no real allocation)
    m._parameters.pop("weight", None)
    # Register as meta float16 buffer of the correct shape — float16 is a placeholder
    # dtype only; the actual buffer will be replaced with a GGMLQuantizedTensor by
    # load_state_dict(assign=True) once the GGUF state dict is loaded.
    m.register_buffer(
        "weight",
        torch.empty(m.out_features, m.in_features, dtype=torch.float16, device="meta"),
        persistent=True,
    )

    def ggml_linear_forward(self: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        # IC-LoRA: forward-time weight patch. When no factors are attached
        # (`specs` is None/empty) the forward is the plain no-LoRA path (dequant
        # + linear, or linear on a float weight); the only extra cost is this
        # attribute read and a skipped branch (VERIFICATION_LOG §21.3). When
        # attached, the delta (strength * B @ A) is computed in fp32 and cast ONCE
        # onto the per-call dequant tensor (VERIFICATION_LOG §21.4).
        specs = getattr(self, _IC_LORA_SPECS_ATTR, None)
        if isinstance(w, GGMLQuantizedTensor):
            # Raw uint8 bytes (1D flat) live in the underlying storage.
            # Drop the subclass identity FIRST: .view() preserves the
            # GGMLQuantizedTensor type but strips the Python attrs
            # (_ggml_type/_float_shape), and the overridden numel()/size()/
            # shape() then read _float_shape and raise AttributeError inside
            # the _dequant_* kernels. as_subclass(torch.Tensor) yields a plain
            # tensor over the same storage, so the kernels see standard
            # numel()/reshape() semantics. The quant metadata is passed
            # explicitly below from w itself (which still carries the attrs).
            raw = w.as_subclass(torch.Tensor).view(torch.uint8)
            bf16 = dequantize_ggml_tensor(raw, w._ggml_type, w._float_shape, x.dtype)
            if specs:
                # In-place add onto the FRESH per-call dequant tensor (transient,
                # freed below) — the compressed GGMLQuantizedTensor bytes are
                # never mutated, so the StateDictRegistry cache stays pristine.
                for a_name, b_name, strength in specs:
                    a = getattr(self, a_name)
                    b = getattr(self, b_name)
                    delta = torch.matmul(
                        b.to(torch.float32) * strength, a.to(torch.float32)
                    )
                    bf16 += delta.to(bf16.dtype)
            result = torch.nn.functional.linear(x, bf16, self.bias)
            del bf16  # free immediately after matmul
            return result
        # Float buffer (BF16 from non-quantised GGUF layers) or standard path.
        if specs:
            # Plain float weight: do NOT mutate the stored buffer (cache-shared /
            # persisted). Build the delta out-of-place and add it into a
            # throwaway weight for this call only.
            acc = None
            for a_name, b_name, strength in specs:
                a = getattr(self, a_name)
                b = getattr(self, b_name)
                delta = torch.matmul(
                    b.to(torch.float32) * strength, a.to(torch.float32)
                ).to(w.dtype)
                acc = delta if acc is None else acc + delta
            return torch.nn.functional.linear(x, w + acc, self.bias)
        return torch.nn.functional.linear(x, w, self.bias)

    m.forward = types.MethodType(ggml_linear_forward, m)


def _patch_model_for_ggml_dequant(model: torch.nn.Module) -> torch.nn.Module:
    count = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            _patch_linear_for_ggml_dequant(m)
            count += 1
    logger.debug("GGUF module_ops: converted %d Linear layers to buffers + patched forward()", count)
    return model


def _make_ggml_quant_module_ops():
    from ltx_core.loader.module_ops import ModuleOps
    from ltx_core.model.transformer.model import LTXModel
    return ModuleOps(
        name="ggml_per_layer_dequant",
        matcher=lambda model: isinstance(model, LTXModel),
        mutator=_patch_model_for_ggml_dequant,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Service — installs the per-layer loader into the ModelLedger
# ──────────────────────────────────────────────────────────────────────────────

class GGUFQuantLoaderService:
    """Per-layer GGUF dequantisation service.

    Keeps transformer weights compressed in VRAM (uint8 bytes in GGMLQuantizedTensor).
    Each Linear layer dequantises its weight at forward() time and immediately frees
    the temporary BF16 tensor after the matmul.

    Usage:
        service = GGUFQuantLoaderService(gguf_path)
        service.install(model_ledger)
    """

    def __init__(
        self,
        gguf_path: str,
        ic_loras_provider: Any = None,
    ) -> None:
        self.gguf_path = gguf_path
        # Callable[[], list[IcLoraEntry]] (``(path, strength, audio_strength)``
        # tuples) returning the CURRENT job's IC-LoRA adapters. Read fresh on
        # every transformer build so the worker can toggle LoRAs per generate(),
        # with or without keep_resident. None attaches nothing; a call returning
        # [] detaches, so the forward takes its no-LoRA branch.
        self._ic_loras_provider = ic_loras_provider

    def install(self, model_ledger: Any) -> None:
        if not Path(self.gguf_path).exists():
            raise FileNotFoundError(f"GGUF file not found: {self.gguf_path}")

        if not hasattr(model_ledger, "transformer_builder"):
            logger.warning("ModelLedger has no transformer_builder — GGUF quant install skipped")
            return

        gguf_loader = GGUFQuantStateDictLoader(self.gguf_path)
        ggml_module_ops = _make_ggml_quant_module_ops()

        # 1. Replace transformer_builder's loader with our GGUF loader
        builder = model_ledger.transformer_builder
        new_builder = dc_replace(builder, model_loader=gguf_loader)
        model_ledger.transformer_builder = new_builder

        # 2. Set QuantizationPolicy to:
        #    a) add our module_ops (converts Linear weights to buffers + patches forward)
        #    b) trigger ltx_core's quantisation build path (build() called WITHOUT dtype
        #       → skips the {k: v.to(dtype)} cast that would corrupt raw bytes)
        from ltx_core.quantization import QuantizationPolicy
        existing_policy = getattr(model_ledger, "quantization", None)
        if existing_policy is not None:
            model_ledger.quantization = QuantizationPolicy(
                sd_ops=existing_policy.sd_ops,
                module_ops=(*existing_policy.module_ops, ggml_module_ops),
            )
        else:
            model_ledger.quantization = QuantizationPolicy(
                sd_ops=None,
                module_ops=(ggml_module_ops,),
            )

        # 3. Wrap transformer() to attach/detach the job's IC-LoRA adapters and
        #    log the quantised buffer count; GGMLQuantizedTensor is self-describing
        #    so no post-build parameter tagging is needed.
        original_transformer_fn = model_ledger.transformer.__func__

        def patched_transformer(self_ledger: Any) -> Any:
            result = original_transformer_fn(self_ledger)
            # IC-LoRA: attach the CURRENT job's adapters to the freshly
            # built transformer BEFORE block-swap moves blocks to CPU. attach
            # registers A/B as non-persistent buffers on each target Linear, so
            # they move with their block during the swap and the forward patch
            # adds their delta onto the per-call dequant tensor. A None
            # provider attaches nothing; an empty list detaches
            # (`detach_ic_loras`), so the build carries no LoRA buffers.
            if self._ic_loras_provider is not None:
                ic_loras = list(self._ic_loras_provider() or [])
                if ic_loras:
                    from engine.gguf.ic_lora_common import attach_ic_loras
                    attach_ic_loras(result, ic_loras)
                else:
                    from engine.gguf.ic_lora_common import detach_ic_loras
                    detach_ic_loras(result)
            ltx_model = getattr(result, "model", result)
            n_quant = sum(
                1
                for buf in ltx_model.buffers()
                if isinstance(buf, GGMLQuantizedTensor)
            )
            if n_quant:
                logger.info(
                    "GGUF per-layer quant active: %d quantised buffers on %s",
                    n_quant,
                    next(ltx_model.buffers(), torch.empty(0)).device,
                )
            else:
                logger.warning(
                    "GGUF quant: no GGMLQuantizedTensor buffers found after build — "
                    "per-layer dequant inactive. Check GGUF key mapping."
                )
            return result

        model_ledger.transformer = types.MethodType(patched_transformer, model_ledger)

        logger.info(
            "GGUFQuantLoaderService installed: %s — weights stay compressed "
            "(in VRAM, or on the CPU side under block swap)",
            Path(self.gguf_path).name,
        )
