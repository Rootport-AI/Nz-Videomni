"""int8 ConvRot Linear for MiniMax H3, executed by ``comfy_kitchen.int8_linear``.

Why a module of our own instead of comfy-kitchen's ``QuantizedTensor``
---------------------------------------------------------------------
``QuantizedTensor`` is a wrapper tensor subclass whose visible dtype/shape are
the ORIGINAL ones (bf16, N x K). The block-swap prefetch packs each block into
one uint8 arena through ``.reshape(-1).view(torch.uint8)`` / ``data_ptr()`` /
``numel() * element_size()``; on the wrapper those either dequantize (a bf16
copy, twice the bytes) or disagree with the int8 payload. A plain int8
``nn.Parameter`` plus a float32 ``weight_scale`` buffer is just two ordinary
tensors, so ``engine/transformer/block_swap_service.py`` and its prefetch carry
it unchanged, and the forward hands the same raw tensors to the public
function ``comfy_kitchen.int8_linear`` — the kernel the wrapper would reach
anyway (R1 "2-2 追加" (a)(b); Docs/Outputs-archive/h3-stage1-reports-2026-10-17/R1_stack.md).

Fail loud, never slow
---------------------
``require_cuda_int8`` disables comfy-kitchen's ``triton`` backend (issue #136
lives on that path) and its ``eager`` backend. With both gone, a call the CUDA
backend refuses (weight still on the CPU, wrong dtype, ...) raises
``NoCapableBackendError`` instead of silently running ``torch._int_mm`` with a
per-call host-to-device copy of the weight — eager accepts such calls and would
only show up as a mysteriously slow job.

The stored weight is ``W @ H^T`` per 256-column group (ConvRot), quantized per
output row; ``weight_scale`` is that row scale, float32 ``[N, 1]`` exactly as
the file holds it. ``int8_linear(convrot=True)`` rotates the activation by
``H`` before quantizing it, so ``(x H)(W H^T)^T = x W^T`` without touching the
weight.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from h3_format import QuantSpec

_ck_module = None


def _ck():
    """comfy_kitchen, imported on first use.

    Not at module top: the CPU tests build H3 modules (meta / plain Linear)
    without ever calling the int8 kernel, and comfy-kitchen's import probes the
    CUDA extension, which is noise there.
    """
    global _ck_module
    if _ck_module is None:
        import comfy_kitchen

        _ck_module = comfy_kitchen
    return _ck_module


class H3Int8Linear(nn.Module):
    """Linear whose weight is a raw int8 Parameter and whose scale is a float32
    ``[N, 1]`` buffer; the forward is ``comfy_kitchen.int8_linear``.

    Build it inside ``with torch.device("meta"):``; the loader fills it with
    ``load_state_dict(assign=True)``. State-dict keys: ``weight`` (I8 [N,K]),
    ``weight_scale`` (F32 [N,1], persistent), ``bias`` (optional). The quant
    spec is a plain attribute, not a tensor, so it never enters the state dict
    or the block-swap arena.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        spec: QuantSpec,
        bias_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        # An int8 Parameter cannot require grad; load_state_dict(assign=True)
        # keeps the module's requires_grad flag, so it stays False after load.
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=torch.int8),
            requires_grad=False,
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(self.out_features, 1, dtype=torch.float32),
            persistent=True,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features, dtype=bias_dtype), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)
        self.spec = spec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x2d = x.reshape(-1, x.shape[-1])
        if x2d.dtype != torch.bfloat16:
            # The kernel accepts f32/f16/bf16; bf16 is the block stack's
            # precision and the only one the gates were measured with.
            x2d = x2d.to(torch.bfloat16)
        x2d = x2d.contiguous()
        bias = self.bias
        if bias is not None and bias.dtype != torch.bfloat16:
            bias = bias.to(torch.bfloat16)
        out = _ck().int8_linear(
            x2d,
            self.weight,
            self.weight_scale,
            bias,
            out_dtype=torch.bfloat16,
            convrot=bool(self.spec.convrot),
            convrot_groupsize=int(self.spec.convrot_groupsize),
        )
        return out.reshape(*lead, self.out_features)

    def forward_rows(self, x: torch.Tensor, n0: int, n1: int) -> torch.Tensor:
        """Output features ``[n0, n1)`` only: :meth:`forward` on rows ``n0:n1`` of the weight.

        For VRAM 16GB mode (``engine_h3.lowvram``: q, k and v of one head group
        from the stacked qkv weight). The weight, scale and bias rows are cut
        from ``self.weight`` / ``self.weight_scale`` / ``self.bias`` at EVERY
        call and never stored: block swap rebinds those tensors' ``.data``, so
        a kept slice would point at the previous placement. A row slice of a
        contiguous [N, K] (and [N, 1]) tensor is a contiguous view; the
        activation is quantized per row, so each output column is the one
        :meth:`forward` computes (the GPU tests of tests/test_h3_engine.py, gate A-G2).
        """
        lead = x.shape[:-1]
        x2d = x.reshape(-1, x.shape[-1])
        if x2d.dtype != torch.bfloat16:
            x2d = x2d.to(torch.bfloat16)
        x2d = x2d.contiguous()
        bias = self.bias
        if bias is not None:
            bias = bias[n0:n1]
            if bias.dtype != torch.bfloat16:
                bias = bias.to(torch.bfloat16)
        out = _ck().int8_linear(
            x2d,
            self.weight[n0:n1],
            self.weight_scale[n0:n1],
            bias,
            out_dtype=torch.bfloat16,
            convrot=bool(self.spec.convrot),
            convrot_groupsize=int(self.spec.convrot_groupsize),
        )
        return out.reshape(*lead, n1 - n0)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, format={self.spec.format}, "
            f"convrot={self.spec.convrot}, groupsize={self.spec.convrot_groupsize}"
        )


def require_cuda_int8() -> dict:
    """Make sure comfy-kitchen's CUDA backend can run ``int8_linear``, then pin
    every int8 call to it. Returns ``list_backends()`` (after the disables) for
    the load log and the ``done`` event.

    ``available`` alone is not enough: the CUDA extension can load while
    cuBLASLt (``cublasLt64_13.dll``) is missing, and then ``int8_linear`` is
    simply not registered on it (R1 2-2). So the capability list is the test.
    """
    ck = _ck()
    backends = ck.list_backends()
    cuda = backends.get("cuda")
    if cuda is None:
        raise RuntimeError(
            "comfy-kitchen に CUDA バックエンドが登録されていません。"
            "MiniMax H3 の int8 推論には CUDA 版の comfy-kitchen（CUDA 13 の torch・ドライバ R580 以上）が必要です。"
        )
    if not cuda.get("available"):
        raise RuntimeError(
            "comfy-kitchen の CUDA バックエンドが使えません（理由: "
            f"{cuda.get('unavailable_reason')}）。CUDA 13 の torch とドライバ R580 以上が必要です。"
        )
    if cuda.get("disabled"):
        raise RuntimeError(
            "comfy-kitchen の CUDA バックエンドが無効にされています。"
            "MiniMax H3 の int8 推論は CUDA 経路でしか動かしません。"
        )
    if "int8_linear" not in (cuda.get("capabilities") or []):
        raise RuntimeError(
            "comfy-kitchen の CUDA バックエンドに int8_linear がありません"
            "（cuBLASLt＝cublasLt64_13.dll が見つからない可能性があります。"
            f"unavailable_reason: {cuda.get('unavailable_reason')}）。"
        )
    # triton: issue #136. eager: would accept every call the CUDA backend
    # refuses and run it slowly; with it gone such a call raises instead.
    ck.disable_backend("triton")
    ck.disable_backend("eager")
    return ck.list_backends()


def swap_in_int8_linears(
    root: nn.Module,
    quant: dict[str, QuantSpec],
    *,
    bias_dtype_of: Callable[[str], torch.dtype],
) -> int:
    """Replace each ``nn.Linear`` named in ``quant`` (dotted, relative to
    ``root``, e.g. ``"blocks.0.attn.qkv_proj"``) by an ``H3Int8Linear`` built on
    the meta device. Returns how many were replaced.

    A name with no module, or a module that is not an ``nn.Linear``, raises:
    a quantized tensor the skeleton cannot hold would otherwise be dropped by
    ``load_state_dict(strict=False)`` and leave a bf16 meta weight behind.
    """
    count = 0
    for name in sorted(quant):
        spec = quant[name]
        parent_name, _, leaf = name.rpartition(".")
        try:
            parent = root.get_submodule(parent_name) if parent_name else root
        except AttributeError as exc:
            raise RuntimeError(f"int8 層 '{name}' の親モジュールがモデルにありません: {exc}") from exc
        old = getattr(parent, leaf, None)
        if old is None or not isinstance(old, nn.Module):
            raise RuntimeError(f"int8 層 '{name}' がモデルにありません")
        if not isinstance(old, nn.Linear):
            raise RuntimeError(
                f"int8 層 '{name}' は nn.Linear ではありません（{type(old).__name__}）"
            )
        with torch.device("meta"):
            new = H3Int8Linear(
                old.in_features,
                old.out_features,
                bias=old.bias is not None,
                spec=spec,
                bias_dtype=bias_dtype_of(name),
            )
        setattr(parent, leaf, new)
        count += 1
    return count
