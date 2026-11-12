"""fp8 (E4M3, per-tensor scale) Linear for MiniMax H3 (台帳 §1-283, 裁定 101).

ComfyUI's ``TensorCoreFP8Layout`` path, run on plain tensors
------------------------------------------------------------
The file's fp8 layers hold an F8_E4M3 ``[N, K]`` weight, a float32 0-dim
``weight_scale`` and, usually, a float32 0-dim ``input_scale``. ComfyUI wraps
them in comfy-kitchen's ``QuantizedTensor``; for the reasons given in
``engine_h3.quant_linear`` (block swap packs plain tensors) this module keeps
them as an ordinary Parameter and buffers and calls the SAME public
comfy-kitchen functions with the SAME arguments the wrapper's handlers call
(the fidelity principle of 裁定 101; nothing of their arithmetic is redone
here, nothing is requantized):

* ``scaled`` (compute capability 8.9 and up, decided once at load:
  ``quant_linear.fp8_mode_for``): ``quantize_per_tensor_fp8(x, input_scale)``,
  then ``comfy_kitchen.scaled_mm_v2.scaled_mm_v2(xq, weight.t(),
  scale_a=input_scale, scale_b=weight_scale, bias=None, out_dtype=bf16)`` --
  ``tensor/fp8.py`` ``_handle_fp8_linear`` / ``_fp8_scaled_mm``;
* ``dequant`` (below 8.9, and ALWAYS for a layer marked
  ``full_precision_matrix_mult``): ``dequantize_per_tensor_fp8(weight,
  weight_scale, bf16)``, then ``F.linear`` -- ComfyUI's
  ``weight.dequantize()`` + ``F.linear``.

A layer without ``input_scale`` on the ``scaled`` path quantizes its input with
scale 1.0, a float32 0-dim tensor made at every call on the input's device, as
ComfyUI does (裁定 101 ⑩); the load logs how many layers that is.

VRAM 16GB mode
--------------
``forward_rows`` cuts weight rows ``[n0, n1)`` at every call (block swap
rebinds ``.data``; a kept slice would point at the previous placement); both
ends must be multiples of 16 (``F.scaled_mm``'s constraint on the weight
operand; the head groups of 1,024 rows meet it). On the ``scaled`` path the qkv
input is quantized once per block (:meth:`H3Fp8Linear.quantize_input` +
:meth:`H3Fp8Linear.forward_rows_q`): the quantization is per tensor with a
fixed scale, so a slice multiplies exactly the bytes a full call would.
On the ``dequant`` path the MLP's fc1 / fc2 are called once per row chunk, so
:meth:`H3Fp8Linear.dequant_cache` keeps the bf16 weight for the length of
that loop only (``lowvram.block_forward``); outside it nothing is kept.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine_h3 import quant_linear
from engine_h3.quant_linear import FP8_MODES, H3QuantLinear, _as_2d, add_lora
from h3_format import FP8, QuantSpec

#: ``F.scaled_mm`` wants both dimensions of the weight operand in multiples of 16.
ROW_MULTIPLE = 16


def _scaled_mm_v2():
    """``comfy_kitchen.scaled_mm_v2.scaled_mm_v2``, imported on first use (tests fake this accessor)."""
    from comfy_kitchen.scaled_mm_v2 import scaled_mm_v2  # noqa: PLC0415

    return scaled_mm_v2


class H3Fp8Linear(H3QuantLinear):
    """Linear whose weight is a raw F8_E4M3 Parameter with float32 0-dim scales.

    Build it inside ``with torch.device("meta"):``; the loader fills it with
    ``load_state_dict(assign=True)``. State-dict keys: ``weight`` (F8_E4M3
    [N, K]), ``weight_scale`` (F32 [], persistent), ``input_scale`` (F32 [],
    only when the file has it). No bias. ``mode`` is ``"scaled"`` or
    ``"dequant"`` (module docstring); a ``full_precision_matrix_mult`` layer is
    ``"dequant"`` whatever is passed.
    """

    def __init__(self, in_features: int, out_features: int, *, spec: QuantSpec, mode: str) -> None:
        super().__init__()
        if spec.format != FP8:
            raise ValueError(f"H3Fp8Linear needs an {FP8} spec, got {spec.format!r}")
        if mode not in FP8_MODES:
            raise ValueError(f"fp8 mode must be one of {FP8_MODES}, got {mode!r}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.register_buffer("weight_scale", torch.empty((), dtype=torch.float32), persistent=True)
        if spec.has_input_scale:
            self.register_buffer("input_scale", torch.empty((), dtype=torch.float32), persistent=True)
        else:
            self.register_buffer("input_scale", None)
        self.register_parameter("bias", None)
        self.spec = spec
        self.mode = "dequant" if spec.full_precision else mode
        # dequant_cache(): (weight.data_ptr(), bf16 weight) while the context is open, else None.
        self._caching = False
        self._cached: tuple[int, torch.Tensor] | None = None

    # -- the two comfy-kitchen calls and the matmul --------------------------------

    def _input_scale_for(self, x2d: torch.Tensor) -> torch.Tensor:
        if self.input_scale is not None:
            return self.input_scale
        return torch.ones((), dtype=torch.float32, device=x2d.device)

    def _quantize(self, x2d: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return quant_linear._ck().quantize_per_tensor_fp8(x2d, scale)

    def _mm(self, xq: torch.Tensor, w_rows: torch.Tensor, scale_a: torch.Tensor) -> torch.Tensor:
        """``tensor/fp8.py`` ``_fp8_scaled_mm``: ``scaled_mm_v2(input.contiguous(), weight.t(), ...)``."""
        return _scaled_mm_v2()(
            xq.contiguous(), w_rows.t(), scale_a=scale_a, scale_b=self.weight_scale,
            bias=None, out_dtype=torch.bfloat16,
        )

    def _dequant(self, w: torch.Tensor) -> torch.Tensor:
        return quant_linear._ck().dequantize_per_tensor_fp8(w, self.weight_scale, torch.bfloat16)

    def _weight_bf16(self) -> torch.Tensor:
        """The whole weight in bf16; kept for the length of :meth:`dequant_cache` only."""
        if not self._caching:
            return self._dequant(self.weight)
        ptr = self.weight.data_ptr()
        if self._cached is not None:
            if self._cached[0] != ptr:
                raise RuntimeError(
                    f"H3Fp8Linear: the weight moved inside dequant_cache() ({self._cached[0]:#x} -> {ptr:#x})"
                )
            return self._cached[1]
        w = self._dequant(self.weight)
        self._cached = (ptr, w)
        return w

    def _check_rows(self, n0: int, n1: int) -> None:
        if n0 % ROW_MULTIPLE or (n1 - n0) % ROW_MULTIPLE:
            raise ValueError(
                f"H3Fp8Linear: rows [{n0}, {n1}) must start and span multiples of {ROW_MULTIPLE}"
            )

    # -- the Linear ------------------------------------------------------------------

    @contextlib.contextmanager
    def dequant_cache(self):
        """Keep the bf16 weight across the calls made inside (``dequant`` path only).

        Entered by ``lowvram.block_forward`` around the MLP's row-chunk loop.
        Dropped on exit; a weight that moves while open raises.
        """
        if self.mode != "dequant" or self._caching:
            yield
            return
        self._caching, self._cached = True, None
        try:
            yield
        finally:
            self._caching, self._cached = False, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead, x2d = _as_2d(x)
        if self.mode == "scaled":
            scale = self._input_scale_for(x2d)
            out = self._mm(self._quantize(x2d, scale), self.weight, scale)
        else:
            out = F.linear(x2d, self._weight_bf16())
        return out.reshape(*lead, self.out_features)

    def forward_rows(self, x: torch.Tensor, n0: int, n1: int, lora_xa=None) -> torch.Tensor:
        """Output features ``[n0, n1)`` only (VRAM 16GB mode's qkv; module docstring).

        This call does not go through ``Module.__call__``, so an attached LoRA
        is added here, rows ``[n0, n1)`` of its B, as ``H3Int8Linear`` does.
        """
        self._check_rows(n0, n1)
        lead, x2d = _as_2d(x)
        if self.mode == "scaled":
            scale = self._input_scale_for(x2d)
            out = self._mm(self._quantize(x2d, scale), self.weight[n0:n1], scale)
        else:
            out = F.linear(x2d, self._dequant(self.weight[n0:n1]))
        out = out.reshape(*lead, n1 - n0)
        return add_lora(self, x, out, n0, n1, lora_xa)

    def shares_input(self, x: torch.Tensor) -> bool:
        """True on the ``scaled`` path with the weight and its scale on ``x``'s device:
        then :meth:`quantize_input` + :meth:`forward_rows_q` stand in for :meth:`forward_rows`."""
        return (
            self.mode == "scaled"
            and self.weight.device == x.device
            and self.weight_scale.device == x.device
        )

    def quantize_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(x quantized to fp8 [M, K], its float32 0-dim scale)``, as :meth:`forward_rows` quantizes it."""
        _lead, x2d = _as_2d(x)
        scale = self._input_scale_for(x2d)
        return self._quantize(x2d, scale), scale

    def forward_rows_q(
        self, xq: torch.Tensor, xs: torch.Tensor, n0: int, n1: int, lora_xa=None
    ) -> torch.Tensor:
        """:meth:`forward_rows` on an input from :meth:`quantize_input`; returns ``[M, n1 - n0]``.

        The raw input is gone, so an attached LoRA gets ``x=None`` and needs ``lora_xa``.
        """
        self._check_rows(n0, n1)
        out = self._mm(xq, self.weight[n0:n1], xs)
        return add_lora(self, None, out, n0, n1, lora_xa)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias=False, "
            f"format={self.spec.format}, mode={self.mode}, full_precision={self.spec.full_precision}, "
            f"input_scale={self.input_scale is not None}"
        )
