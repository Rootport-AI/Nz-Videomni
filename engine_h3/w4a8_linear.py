"""Grouped W4A8 Linear for MiniMax H3 (``asym_w4a8_int8``; 台帳 §1-283, 裁定 101).

ComfyUI's ``AsymW4A8Int8Layout`` path, run on plain tensors
------------------------------------------------------------
The file's w4a8 layers hold the packed 4-bit weight as I8 ``[N, K/2]``, a
per-group relative scale ``weight_s_rel`` F8_E4M3 ``[N, K/G]``, a per-row
``weight_s_channel`` F32 ``[N]`` and, optionally, a 16-entry ``weight_codebook``
F32. ComfyUI wraps them in comfy-kitchen's ``QuantizedTensor``; this module
keeps them as an ordinary Parameter and buffers (block swap packs plain
tensors, ``engine_h3.quant_linear``) and calls the SAME public function with
the SAME arguments the wrapper's handler (``tensor/w4a8_int8.py``
``_w4a8_int8_forward``) calls: ``comfy_kitchen.w4a8_int8_linear(x, qdata,
s_rel, s_channel, codebook=..., correction=None, bias=None, group_size=G,
convrot_groupsize=256, out_dtype=bf16)``. The kernel rotates and quantizes the
input itself (``QUANTIZES_INPUT = False`` of the layout), so nothing is
quantized here and :meth:`H3W4A8Linear.shares_input` is False: VRAM 16GB mode
calls :meth:`H3W4A8Linear.forward_rows` once per qkv head-group slice. The
CUDA kernels need compute capability 8.0 (``quant_linear.require_w4a8_capability``).

``forward_rows`` cuts rows ``[n0, n1)`` of the weight and of both scales at
every call (block swap rebinds ``.data``); the output rows are the rows a
full call computes, because each output row only reads its own weight row and
scales (S1 ② of the spike, 記録簿 §210).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from engine_h3 import quant_linear
from engine_h3.quant_linear import H3QuantLinear, _as_2d, add_lora
from h3_format import W4A8, QuantSpec


class H3W4A8Linear(H3QuantLinear):
    """Linear whose packed 4-bit weight is a raw I8 ``[N, K/2]`` Parameter.

    Build it inside ``with torch.device("meta"):``; the loader fills it with
    ``load_state_dict(assign=True)``. State-dict keys: ``weight`` (I8 [N, K/2]),
    ``weight_s_rel`` (F8_E4M3 [N, K/G]), ``weight_s_channel`` (F32 [N]),
    ``weight_codebook`` (F32 [16], only when the file has it). No bias.
    """

    def __init__(self, in_features: int, out_features: int, *, spec: QuantSpec) -> None:
        super().__init__()
        if spec.format != W4A8 or spec.group_size is None:
            raise ValueError(f"H3W4A8Linear needs an {W4A8} spec with a group size, got {spec!r}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        n, k, g = self.out_features, self.in_features, int(spec.group_size)
        self.weight = nn.Parameter(torch.empty(n, k // 2, dtype=torch.int8), requires_grad=False)
        self.register_buffer("weight_s_rel", torch.empty(n, k // g, dtype=torch.float8_e4m3fn), persistent=True)
        self.register_buffer("weight_s_channel", torch.empty(n, dtype=torch.float32), persistent=True)
        if spec.has_codebook:
            self.register_buffer("weight_codebook", torch.empty(16, dtype=torch.float32), persistent=True)
        else:
            self.register_buffer("weight_codebook", None)
        self.register_parameter("bias", None)
        self.spec = spec

    def _linear(self, x2d: torch.Tensor, qdata: torch.Tensor, s_rel: torch.Tensor, s_channel: torch.Tensor):
        return quant_linear._ck().w4a8_int8_linear(
            x2d,
            qdata,
            s_rel,
            s_channel,
            codebook=self.weight_codebook,
            correction=None,
            bias=None,
            group_size=int(self.spec.group_size),
            convrot_groupsize=int(self.spec.convrot_groupsize),
            out_dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead, x2d = _as_2d(x)
        out = self._linear(x2d, self.weight, self.weight_s_rel, self.weight_s_channel)
        return out.reshape(*lead, self.out_features)

    def forward_rows(self, x: torch.Tensor, n0: int, n1: int, lora_xa=None) -> torch.Tensor:
        """Output features ``[n0, n1)`` only (VRAM 16GB mode's qkv; module docstring).

        This call does not go through ``Module.__call__``, so an attached LoRA
        is added here, rows ``[n0, n1)`` of its B, as ``H3Int8Linear`` does.
        """
        lead, x2d = _as_2d(x)
        out = self._linear(x2d, self.weight[n0:n1], self.weight_s_rel[n0:n1], self.weight_s_channel[n0:n1])
        out = out.reshape(*lead, n1 - n0)
        return add_lora(self, x, out, n0, n1, lora_xa)

    def shares_input(self, x: torch.Tensor) -> bool:
        """Always False: the kernel quantizes its input itself (module docstring)."""
        return False

    def quantize_input(self, x: torch.Tensor):
        raise RuntimeError("H3W4A8Linear.quantize_input: the w4a8 input is never shared (shares_input is False)")

    def forward_rows_q(self, xq, xs, n0: int, n1: int, lora_xa=None):
        raise RuntimeError("H3W4A8Linear.forward_rows_q: the w4a8 input is never shared (shares_input is False)")

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias=False, "
            f"format={self.spec.format}, group_size={self.spec.group_size}, "
            f"convrot_groupsize={self.spec.convrot_groupsize}, codebook={self.weight_codebook is not None}"
        )
