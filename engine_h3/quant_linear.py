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

The one exception: VRAM 16GB mode's qkv (台帳 §1-105 ①)
---------------------------------------------------------
``engine_h3.lowvram`` projects q, k and v per head group from the stacked qkv
weight: 3 x 7 ``forward_rows`` calls per block on the SAME input, and each
``int8_linear`` call quantizes that input again. For that one caller the call
is split in two (design doc §5, exception): :meth:`H3Int8Linear.quantize_input`
runs comfy-kitchen's ``backends.cuda.quantize_int8_rowwise_convrot64`` once per
block (the kernel ``int8_linear`` itself runs, same arguments; it is
deterministic), and :meth:`H3Int8Linear.forward_rows_q` hands the result to
``_int8_gemm_prequantized``, a copy of ``int8_linear``'s code after the
quantization (the same GEMM kernels, chosen the same way). Both are
comfy-kitchen 0.2.37 internals, not its API: ``require_cuda_int8`` ends with
``_probe_shared_quant``, which checks the version, the symbols and their
argument lists, and on any mismatch leaves the sharing off (logged; the load
goes on) so every call takes ``forward_rows`` as before. That path bypasses the
backend registry and its constraints, so to stay fail loud
:meth:`H3Int8Linear.shares_input` sends a call whose weight is not on the
input's device back to ``forward_rows``, where ``int8_linear`` refuses it.

The stored weight is ``W @ H^T`` per 256-column group (ConvRot), quantized per
output row; ``weight_scale`` is that row scale, float32 ``[N, 1]`` exactly as
the file holds it. ``int8_linear(convrot=True)`` rotates the activation by
``H`` before quantizing it, so ``(x H)(W H^T)^T = x W^T`` without touching the
weight.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import torch
import torch.nn as nn

from h3_format import QuantSpec

_ck_module = None

# VRAM 16GB mode's shared qkv quantization (台帳 §1-105 ①; module docstring).
# _SHARED_CB: comfy_kitchen.backends.cuda once _probe_shared_quant passed it, else None.
# _SHARED_REASON: why it is None (None once it passed).
_SHARED_CB = None
_SHARED_REASON = "not probed"

_SHARED_CK_VERSION = "0.2.37"
_SHARED_CB_NAMES = (
    "quantize_int8_rowwise_convrot64",
    "_CONVROT_FUSED_MAX_K",
    "_convrot_fused_shared_memory_fits",
    "_cuda_device_supports_cutlass_int8_dequant",
    "_prefer_cublas_int8_fallback",
    "_DISABLE_CUTLASS_INT8",
    "_int8_weight_scale_arg",
    "_gemm_vector_arg",
    "_empty_cuda_tensor",
    "_wrap_for_dlpack",
    "DTYPE_TO_CODE",
    "get_cublas_workspace",
    "_EXT_AVAILABLE",
    "_C",
)
# The Python functions' argument names, in order (inspect.signature).
_SHARED_CB_SIGNATURES = {
    "quantize_int8_rowwise_convrot64": (
        "weight_2d", "group_size", "stochastic_rounding", "input_act", "input_act_weight", "input_act_eps",
    ),
    "_convrot_fused_shared_memory_fits": ("x", "k", "group_size"),
    "_cuda_device_supports_cutlass_int8_dequant": ("tensor",),
    "_prefer_cublas_int8_fallback": ("m", "n", "k", "device_index"),
    "_int8_weight_scale_arg": ("weight_scale", "device"),
    "_gemm_vector_arg": ("v", "device", "dtype"),
    "_empty_cuda_tensor": ("device", "dtype"),
    "_wrap_for_dlpack": ("tensor",),
    "get_cublas_workspace": (),
}
# The extension's functions: the first line of the nanobind docstring, as is.
_SHARED_C_DOCS = {
    "quantize_int8_rowwise_convrot64": (
        "quantize_int8_rowwise_convrot64(input: ndarray[shape=(*, *), device='cuda'], "
        "output: ndarray[dtype=int8, shape=(*, *), device='cuda'], "
        "scales: ndarray[dtype=float32, shape=(*, *), device='cuda'], group_size: int, stochastic: bool, "
        "act_code: int, seed: int, act_weight: ndarray[device='cuda'], act_eps: float, stream_ptr: int) -> None"
    ),
    "cutlass_int8_dequant": (
        "cutlass_int8_dequant(a: ndarray[dtype=int8, shape=(*, *), device='cuda'], "
        "b: ndarray[dtype=int8, shape=(*, *), device='cuda'], xs: ndarray[dtype=float32, device='cuda'], "
        "ws: ndarray[dtype=float32, device='cuda'], bias: ndarray[device='cuda'], "
        "d: ndarray[shape=(*, *), device='cuda'], out_dtype_code: int, stream_ptr: int) -> bool"
    ),
    "cublas_gemm_int8": (
        "cublas_gemm_int8(a: ndarray[dtype=int8, shape=(*, *), device='cuda'], "
        "b: ndarray[dtype=int8, shape=(*, *), device='cuda'], c: ndarray[dtype=int32, shape=(*, *), device='cuda'], "
        "workspace: ndarray[device='cuda'], stream_ptr: int) -> None"
    ),
    "dequantize_int8_linear": (
        "dequantize_int8_linear(input: ndarray[dtype=int32, shape=(*, *), device='cuda'], "
        "x_scales: ndarray[dtype=float32, shape=(*, *), device='cuda'], "
        "weight_scales: ndarray[dtype=float32, device='cuda'], bias: ndarray[device='cuda'], "
        "output: ndarray[shape=(*, *), device='cuda'], output_dtype_code: int, stream_ptr: int) -> None"
    ),
}


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

    def forward_rows(self, x: torch.Tensor, n0: int, n1: int, lora_xa=None) -> torch.Tensor:
        """Output features ``[n0, n1)`` only: :meth:`forward` on rows ``n0:n1`` of the weight.

        For VRAM 16GB mode (``engine_h3.lowvram``: q, k and v of one head group
        from the stacked qkv weight). The weight, scale and bias rows are cut
        from ``self.weight`` / ``self.weight_scale`` / ``self.bias`` at EVERY
        call and never stored: block swap rebinds those tensors' ``.data``, so
        a kept slice would point at the previous placement. A row slice of a
        contiguous [N, K] (and [N, 1]) tensor is a contiguous view; the
        activation is quantized per row, so each output column is the one
        :meth:`forward` computes (the GPU tests of tests/test_h3_engine.py, gate A-G2).

        This call does not go through ``Module.__call__``, so an attached LoRA
        (``engine_h3.lora``) is added here, rows ``[n0, n1)`` of its B;
        ``lora_xa`` is ``lora.xa_for(self, x)`` computed once by the caller.
        Without LoRA this is one attribute lookup.
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
        out = out.reshape(*lead, n1 - n0)
        if getattr(self, "_lora_specs", None):
            from engine_h3 import lora  # noqa: PLC0415 -- only a LoRA job gets here

            out = lora.apply(self, x, out, n0, n1, xa=lora_xa)
        return out

    def shares_input(self, x: torch.Tensor) -> bool:
        """Whether :meth:`quantize_input` + :meth:`forward_rows_q` may stand in for
        :meth:`forward_rows` on ``x`` (VRAM 16GB mode's qkv; module docstring).

        True only where ``int8_linear`` would quantize ``x`` with the fused ConvRot
        kernel and multiply on the generic GEMM path that ``_int8_gemm_prequantized``
        copies: the m = 1 / m <= 2 kernels and the Turing kernels are never reached.
        The weight and its scale must already be on ``x``'s device: this path skips
        the backend registry, so such a call goes to ``forward_rows`` and fails
        there. The bias's device is not checked (a CPU bias would be copied over by
        ``_gemm_vector_arg``, slow but not refused); the qkv this serves has no bias.
        """
        cb = _SHARED_CB
        if cb is None:
            return False
        if not self.spec.convrot or int(self.spec.convrot_groupsize) != 256:
            return False
        x2d = x.reshape(-1, x.shape[-1])
        k = x2d.shape[1]
        if k % 256 != 0 or not 256 <= k <= cb._CONVROT_FUSED_MAX_K:          # int8_linear 2062-2065
            return False
        if not cb._convrot_fused_shared_memory_fits(x2d, k, 256):            # int8_linear 2066
            return False
        if x2d.shape[0] <= 2:                                                # int8_linear 2091, 2169
            return False
        if not cb._cuda_device_supports_cutlass_int8_dequant(x2d):           # major >= 8: not Turing
            return False
        return self.weight.device == x.device and self.weight_scale.device == x.device

    def quantize_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` quantized as ``int8_linear`` quantizes it: ``(int8 [M, K], float32 [M, 1])``.

        Only after :meth:`shares_input` said True. Same cast as :meth:`forward_rows`.
        """
        x2d = x.reshape(-1, x.shape[-1])
        if x2d.dtype != torch.bfloat16:
            x2d = x2d.to(torch.bfloat16)
        x2d = x2d.contiguous()
        return _SHARED_CB.quantize_int8_rowwise_convrot64(x2d, int(self.spec.convrot_groupsize))

    def forward_rows_q(
        self, xq: torch.Tensor, xs: torch.Tensor, n0: int, n1: int, lora_xa=None
    ) -> torch.Tensor:
        """:meth:`forward_rows` on an input already quantized by :meth:`quantize_input`.

        Returns ``[M, n1 - n0]``. The weight, scale and bias rows are cut at every
        call, never stored, for the same reason as in :meth:`forward_rows`. The raw
        input is gone, so an attached LoRA gets ``x=None`` and needs ``lora_xa``
        (``lora.xa_for`` of the raw input, computed by the caller before quantizing).
        """
        bias = self.bias
        if bias is not None:
            bias = bias[n0:n1]
            if bias.dtype != torch.bfloat16:
                bias = bias.to(torch.bfloat16)
        out = _int8_gemm_prequantized(
            xq, xs, self.weight[n0:n1], self.weight_scale[n0:n1], bias, torch.bfloat16
        )
        if getattr(self, "_lora_specs", None):
            from engine_h3 import lora  # noqa: PLC0415 -- only a LoRA job gets here

            out = lora.apply(self, None, out, n0, n1, xa=lora_xa)
        return out

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
    _probe_shared_quant()
    return ck.list_backends()


def _shared_quant_missing(cb) -> str | None:
    """What keeps ``cb`` (``comfy_kitchen.backends.cuda``) from serving the shared
    qkv quantization, as a short sentence; None when nothing does. No CUDA call.

    The version first: the copy in ``_int8_gemm_prequantized`` is of 0.2.37's
    ``int8_linear``, and a newer one could branch differently with the same names.
    """
    from importlib import metadata  # noqa: PLC0415

    try:
        version = metadata.version("comfy-kitchen")
    except metadata.PackageNotFoundError:
        return "the comfy-kitchen distribution is not installed"
    if version != _SHARED_CK_VERSION:
        return f"comfy-kitchen {version} != {_SHARED_CK_VERSION}"
    for name in _SHARED_CB_NAMES:
        if not hasattr(cb, name):
            return f"comfy_kitchen.backends.cuda.{name} is missing"
    if not cb._EXT_AVAILABLE or cb._C is None:
        return "comfy_kitchen.backends.cuda._C is not loaded"
    for name, want in _SHARED_C_DOCS.items():
        fn = getattr(cb._C, name, None)
        if fn is None:
            return f"comfy_kitchen.backends.cuda._C.{name} is missing"
        got = (fn.__doc__ or "").split("\n", 1)[0]
        if got != want:
            return f"signature of _C.{name} is {got!r}, expected {want!r}"
    for name, want in _SHARED_CB_SIGNATURES.items():
        got = tuple(inspect.signature(getattr(cb, name)).parameters)
        if got != want:
            return f"signature of {name} is {got}, expected {want}"
    return None


def _probe_shared_quant() -> None:
    """Set ``_SHARED_CB`` / ``_SHARED_REASON`` (module docstring). Never raises:
    a failed probe only means the qkv input is quantized per call, as before."""
    global _SHARED_CB, _SHARED_REASON
    _SHARED_CB, _SHARED_REASON = None, "not probed"
    try:
        import comfy_kitchen.backends.cuda as cb  # noqa: PLC0415

        reason = _shared_quant_missing(cb)
        if reason is None:
            major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
            if major < 8:
                reason = f"compute capability {major}.{minor} < 8.0"
    except Exception as exc:  # noqa: BLE001 -- the probe never fails the load
        reason = f"the probe failed: {exc!r}"
    if reason is None:
        _SHARED_CB, _SHARED_REASON = cb, None
    else:
        _SHARED_REASON = reason


def shared_quant_status() -> tuple[bool, str | None]:
    """``(shared, reason)``: whether the last probe enabled the shared qkv
    quantization, and why not when it did not."""
    return _SHARED_CB is not None, _SHARED_REASON


def _int8_gemm_prequantized(
    xq: torch.Tensor,
    xs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``comfy_kitchen.backends.cuda.int8_linear`` (0.2.37) after its quantization:
    ``xq`` / ``xs`` are what its fused ConvRot quantizer would have produced.

    The numbers in the comments are the lines of ``int8_linear`` in
    ``comfy_kitchen/backends/cuda/__init__.py`` (0.2.37) each line copies;
    compare them when the version moves. Not copied, unreachable under
    :meth:`H3Int8Linear.shares_input`: the m = 1 kernel (2091-2126), the m <= 2
    GEMV (2169-2188), the Turing kernels and padding (2195-2209, 2258-2263) and
    the residual (2079-2089, 2224-2244; none is passed, the output is 2-D).
    ``_DISABLE_CUTLASS_INT8`` is read from the module at every call, as there.
    """
    cb = _SHARED_CB
    ext = cb._C
    wrap = cb._wrap_for_dlpack
    dev = xq.device
    if not weight.is_contiguous():                                                    # 2053
        weight = weight.contiguous()                                                  # 2054
    stream_ptr = torch.cuda.current_stream(dev).cuda_stream                           # 2055
    m, k = xq.shape                                                                   # 2072-2073
    n, k_w = weight.shape                                                             # 2074
    assert k == k_w, "Input and weight inner dimensions must match"                   # 2075
    output_dtype_code = cb.DTYPE_TO_CODE[out_dtype]                                   # 2078
    out = torch.empty((m, n), dtype=out_dtype, device=dev)                            # 2190
    weight_scale = cb._int8_weight_scale_arg(weight_scale, dev)                       # 2191
    bias_arg = (cb._gemm_vector_arg(bias, dev, out_dtype) if bias is not None         # 2192
                else cb._empty_cuda_tensor(dev, out_dtype))                           # 2193
    used_cutlass = False                                                              # 2211
    prefer_cublas_fallback = cb._prefer_cublas_int8_fallback(m, n, k, xq.get_device())  # 2212-2217
    if (
        not prefer_cublas_fallback                                                    # 2219
        and not cb._DISABLE_CUTLASS_INT8                                              # 2220
        and cb._cuda_device_supports_cutlass_int8_dequant(xq)                         # 2221
    ):
        ws_cutlass = weight_scale if weight_scale.numel() == n else weight_scale.expand(n).contiguous()  # 2223
        used_cutlass = ext.cutlass_int8_dequant(                                      # 2246
            wrap(xq),
            wrap(weight),
            wrap(xs),
            wrap(ws_cutlass),
            wrap(bias_arg),
            wrap(out),
            output_dtype_code,
            stream_ptr,
        )                                                                             # 2255
    if not used_cutlass:                                                              # 2256
        out_int32 = torch.empty((m, n), dtype=torch.int32, device=dev)                # 2265-2269 (padded_n = n)
        ext.cublas_gemm_int8(                                                         # 2270
            wrap(xq),
            wrap(weight),
            wrap(out_int32),
            wrap(cb.get_cublas_workspace()),
            stream_ptr,
        )                                                                             # 2276
        ext.dequantize_int8_linear(                                                   # 2279
            wrap(out_int32),
            wrap(xs),
            wrap(weight_scale),
            wrap(bias_arg),
            wrap(out),
            output_dtype_code,
            stream_ptr,
        )                                                                             # 2287
    return out                                                                        # 2289


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
