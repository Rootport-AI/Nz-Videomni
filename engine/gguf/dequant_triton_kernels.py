"""Triton kernels for GGUF K-quant dequantisation (Q4_K / Q5_K / Q6_K).

This is the ONLY module in the package that imports ``triton``. It is imported
lazily by ``dequant_triton._kernels()`` so that ``quant_service`` keeps its
"stdlib + torch only" import invariant: a machine without a working Triton
never even reaches this file, and the job falls back to the eager kernels in
``quant_service`` instead of failing.

What these kernels replace
--------------------------
``quant_service._dequant_q4_k`` / ``_q5_k`` / ``_q6_k`` are element-for-element
ports of ``gguf.quants.{Q4_K,Q5_K,Q6_K}.dequantize_blocks``. They are correct,
but they materialise ~20-30x the minimum bytes through DRAM as int32/fp32
intermediates and issue 18-33 CUDA kernels per call, which costs ~1.96s of GPU
time per DiT forward pass (1632 quantised linears, VERIFICATION_LOG section 50).
Each kernel below does the whole thing in ONE pass: raw bytes in, bf16 out.

Bit-exactness is a hard requirement (the ON and OFF paths must produce the same
video), so the numerics follow three rules, all of which are load-bearing:

  1. **Type discipline.** Bytes are loaded from a ``uint8`` pointer and widened
     to ``int32`` (zero extension, verified on this Triton build). The ONLY
     signed bytes in the whole format are Q6_K's 16 scale bytes, which the eager
     path reads via ``view(torch.int8)`` - reproduced here as ``x - 256 if
     x > 127``. The fp16 ``d``/``dmin`` scalars are assembled from their two
     bytes only AFTER both have been widened past 8 bits (``lo | (hi << 8)``
     on ``uint8`` operands would shift the high byte clean out of the type and
     silently yield a wrong-but-plausible scale), narrowed to ``uint16``, then
     bitcast to fp16.
  2. **fp32 everywhere in the middle**, with exactly one ``.to(bfloat16)``
     immediately before the store. Triton and PyTorch both round-to-nearest-even
     on that conversion, so one conversion at the same place gives the same bits.
  3. **Same multiplication order as the eager code**: ``d*sc`` and ``dmin*mn``
     are formed first, then ``(d*sc)*q - (dmin*mn)``. Both products are exact in
     fp32 by construction (fp16 significand 11 bits x 6-bit scale x 5-bit quant
     <= 22 bits < 24), so the single rounding of the subtraction is the only
     rounding in the expression and FMA contraction cannot change it. We still
     pass ``enable_fp_fusion=False`` as cheap insurance.

Shape independence
------------------
``n_blocks`` is a runtime argument and ``BLOCKS_PER_PROG`` a ``constexpr``, so
each kernel compiles exactly ONCE per process regardless of how many distinct
tensor shapes the model has (the model has 16 distinct element counts; a
shape-specialised design would pay 16 JIT compilations on the first job).
``do_not_specialize=["n_blocks"]`` is what makes "once" literally true: without
it Triton auto-specialises integer arguments on ``== 1`` and ``% 16 == 0``, so
the 16 shapes would still have produced three compilations per kernel. The
divisibility hint buys nothing here because ``n_blocks`` only ever appears in a
bounds comparison. ``triton.autotune`` is deliberately not used: it would
re-benchmark per shape and reintroduce exactly the cost this design removes.

"Once per process" is about SHAPES, and there is one axis it does not cover:
Triton also specialises on each TENSOR argument's ``data_ptr() % 16 == 0``, and
that is out of this module's hands. A caller that slices ``raw`` at an offset
which is not a multiple of 16 gets a second compiled variant of the same kernel
- the LTX 2.5 text encoder's chunked linear does exactly that, because Q6_K's
row stride of 154350 bytes is not a multiple of 16. So a real job may end with
more than three entries in ``jit_cache_size()``, up to six, and that is expected
rather than a shape leak. Every variant is bit-verified before it is trusted:
``dequant_triton._VERIFIED`` keys on the alignment class for this reason. The
selfcheck only ever feeds aligned payloads (its arena case uses a 512B offset),
which is why C8 still sees exactly three.

Index algebra
-------------
Every kernel walks the same coordinates: ``j`` = 0..255 is the weight index
inside a 256-weight super-block, ``s = j >> 5`` its 32-weight sub-block and
``t = j & 31`` the offset within that sub-block. The per-format mapping from
(s, t) back to bytes is documented at each kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Weights per GGUF super-block, and the packed byte size of one super-block.
QK_K = 256
Q4_K_BLOCK_BYTES = 144
Q5_K_BLOCK_BYTES = 176
Q6_K_BLOCK_BYTES = 210

# The same four numbers again, as Triton constexprs. A @triton.jit body may only
# read a module global that is literally a ``tl.constexpr`` object, so the plain
# ints above (used by the Python wrappers) cannot be referenced from a kernel.
_QK_K = tl.constexpr(QK_K)
_Q4_K_BYTES = tl.constexpr(Q4_K_BLOCK_BYTES)
_Q5_K_BYTES = tl.constexpr(Q5_K_BLOCK_BYTES)
_Q6_K_BYTES = tl.constexpr(Q6_K_BLOCK_BYTES)

# Super-blocks handled by one Triton program. Frozen at 8 after benchmarking
# 4/8/16/32 over all 16 shapes of the real model, weighted by their per-pass call
# counts (STEP 1, RTX 4070 Ti SUPER): 97.9 / 96.6 / 103.9 / 113.9 ms per DiT
# forward pass. 8 wins outright, 4 is within 1.3%, and the two larger settings
# lose because 16-32 super-blocks per program is 4096-8192 elements over 4 warps
# and the register pressure costs occupancy on the big Q6_K shapes.
# constexpr, so BPP * 256 is a compile-time power of two and ``tl.arange`` is
# legal.
BLOCKS_PER_PROG = 8

# One warp per 512 elements at BPP=8. Pinned rather than left to Triton's
# heuristic so the compiled kernel does not silently change shape between
# Triton releases.
NUM_WARPS = 4


@triton.jit
def _load_f16(blk, off):
    """Read the 2-byte little-endian fp16 at ``blk + off`` and widen to fp32.

    ``blk`` is a per-element pointer into the uint8 payload. The widening to
    int32 BEFORE the shift is the whole point of this helper - see rule 1 in the
    module docstring.
    """
    lo = tl.load(blk + off).to(tl.int32)
    hi = tl.load(blk + off + 1).to(tl.int32)
    bits = (lo | (hi << 8)).to(tl.uint16)
    return bits.to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _scale_min_k(blk, s):
    """Q4_K/Q5_K 6-bit scale+min unpack. Port of ``quant_service._q_get_scale_min``.

    The 12 scale bytes at ``blk + 4`` hold eight 6-bit scales and eight 6-bit
    mins. For the first four sub-blocks both live in their own byte; for the
    last four they are split across three bytes. ``i = s & 3`` indexes the same
    triple in either case, so three scalar loads plus one ``where`` cover both
    halves without ever materialising the awkward 12-element vector (which
    ``tl.arange`` could not address anyway - 12 is not a power of two).
    """
    i = s & 3
    b0 = tl.load(blk + 4 + i).to(tl.int32)        # scales[i]
    b1 = tl.load(blk + 4 + 4 + i).to(tl.int32)    # scales[4 + i]
    b2 = tl.load(blk + 4 + 8 + i).to(tl.int32)    # scales[8 + i]
    low = s < 4
    sc = tl.where(low, b0 & 0x3F, (b2 & 0x0F) | ((b0 >> 2) & 0x30))
    mn = tl.where(low, b1 & 0x3F, (b2 >> 4) | ((b1 >> 2) & 0x30))
    return sc, mn


@triton.jit(do_not_specialize=["n_blocks"])
def _q4_k_kernel(raw_ptr, out_ptr, n_blocks, BPP: tl.constexpr):
    """Q4_K: 144 bytes -> 256 weights.

    Layout: d (fp16) | dmin (fp16) | scales (12 B) | qs (128 B).
    ``qs`` is read by the eager path as ``(n, 4, 1, 32) >> [0, 4] -> (n, 8, 32)``,
    i.e. sub-block ``s`` is byte group ``g = s >> 1`` and nibble ``h = s & 1``.
    """
    pid = tl.program_id(0)
    idx = tl.arange(0, BPP * _QK_K)
    b = pid * BPP + (idx >> 8)
    j = idx & 255
    keep = b < n_blocks
    # Clamp the block index instead of masking every load: an out-of-range
    # program then reads block 0's bytes (harmless) and the single masked store
    # at the end throws the result away. int64 so that the byte offset is safe
    # for tensors past 2^31 bytes (Gemma's token_embd is 566 MB of Q4_K).
    b_ok = tl.where(keep, b, 0).to(tl.int64)
    blk = raw_ptr + b_ok * _Q4_K_BYTES

    d = _load_f16(blk, 0)
    dmin = _load_f16(blk, 2)

    s = j >> 5
    t = j & 31
    sc, mn = _scale_min_k(blk, s)

    g = s >> 1
    h = s & 1
    qb = tl.load(blk + 4 + 12 + g * 32 + t).to(tl.int32)
    q = (qb >> (4 * h)) & 0x0F

    val = (d * sc.to(tl.float32)) * q.to(tl.float32) - (dmin * mn.to(tl.float32))
    tl.store(out_ptr + b_ok * _QK_K + j, val.to(tl.bfloat16), mask=keep)


@triton.jit(do_not_specialize=["n_blocks"])
def _q5_k_kernel(raw_ptr, out_ptr, n_blocks, BPP: tl.constexpr):
    """Q5_K: 176 bytes -> 256 weights.

    Layout: d | dmin | scales (12 B) | qh (32 B) | qs (128 B).
    Same low nibbles as Q4_K; the 5th bit comes from ``qh`` read as
    ``(n, 1, 1, 32) >> arange(8) -> (n, 8, 32)``, i.e. byte ``t`` bit ``s``.
    """
    pid = tl.program_id(0)
    idx = tl.arange(0, BPP * _QK_K)
    b = pid * BPP + (idx >> 8)
    j = idx & 255
    keep = b < n_blocks
    b_ok = tl.where(keep, b, 0).to(tl.int64)
    blk = raw_ptr + b_ok * _Q5_K_BYTES

    d = _load_f16(blk, 0)
    dmin = _load_f16(blk, 2)

    s = j >> 5
    t = j & 31
    sc, mn = _scale_min_k(blk, s)

    g = s >> 1
    h = s & 1
    qb = tl.load(blk + 4 + 12 + 32 + g * 32 + t).to(tl.int32)
    ql = (qb >> (4 * h)) & 0x0F
    hb = (tl.load(blk + 4 + 12 + t).to(tl.int32) >> s) & 0x01
    q = ql | (hb << 4)

    val = (d * sc.to(tl.float32)) * q.to(tl.float32) - (dmin * mn.to(tl.float32))
    tl.store(out_ptr + b_ok * _QK_K + j, val.to(tl.bfloat16), mask=keep)


@triton.jit(do_not_specialize=["n_blocks"])
def _q6_k_kernel(raw_ptr, out_ptr, n_blocks, BPP: tl.constexpr):
    """Q6_K: 210 bytes -> 256 weights.

    Layout: ql (128 B) | qh (64 B) | scales (16 B, SIGNED) | d (fp16).
    ``ql`` is read as ``(n, 2, 1, 64) >> [0, 4] -> (n, 8, 32)``: sub-block ``s``
    is group ``g = s >> 2``, nibble ``h = (s >> 1) & 1``, byte ``((s & 1) << 5) + t``.
    ``qh`` is ``(n, 2, 1, 32) >> [0, 2, 4, 6] -> (n, 8, 32)``: group ``s >> 2``,
    bit pair ``s & 3``, byte ``t``. There is no ``dmin``; the -32 bias applies to
    the 6-bit quant itself, in int32 (values 0..63, so no wrap-around).
    """
    pid = tl.program_id(0)
    idx = tl.arange(0, BPP * _QK_K)
    b = pid * BPP + (idx >> 8)
    j = idx & 255
    keep = b < n_blocks
    b_ok = tl.where(keep, b, 0).to(tl.int64)
    blk = raw_ptr + b_ok * _Q6_K_BYTES

    d = _load_f16(blk, 208)

    s = j >> 5
    t = j & 31
    g = s >> 2
    h = (s >> 1) & 1
    u = ((s & 1) << 5) + t
    k = s & 3

    lo = (tl.load(blk + g * 64 + u).to(tl.int32) >> (4 * h)) & 0x0F
    hi = (tl.load(blk + 128 + g * 32 + t).to(tl.int32) >> (2 * k)) & 0x03
    q = (lo | (hi << 4)) - 32

    # The only signed bytes in the format: eager reads them via view(int8).
    sb = tl.load(blk + 192 + (j >> 4)).to(tl.int32)
    sc8 = tl.where(sb > 127, sb - 256, sb)

    val = (d * sc8.to(tl.float32)) * q.to(tl.float32)
    tl.store(out_ptr + b_ok * _QK_K + j, val.to(tl.bfloat16), mask=keep)


_KERNELS = {
    "q4_k": (_q4_k_kernel, Q4_K_BLOCK_BYTES),
    "q5_k": (_q5_k_kernel, Q5_K_BLOCK_BYTES),
    "q6_k": (_q6_k_kernel, Q6_K_BLOCK_BYTES),
}


def _n_elems(shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _run(name: str, raw: torch.Tensor, shape) -> torch.Tensor:
    """Launch one kernel over ``raw`` and shape the result like the eager path.

    The eager kernels end with ``out.reshape(-1)[:ne].reshape(shape)``, i.e. the
    tail of the last super-block is simply dropped. Reproduced here by writing
    the full ``n_blocks * 256`` output and slicing, so a tensor whose element
    count is not a multiple of 256 comes back with the same values AND the same
    contiguous layout (the IC-LoRA forward adds into this tensor in place).
    """
    kernel, block_bytes = _KERNELS[name]
    data = raw.view(torch.uint8)
    n_blocks = data.numel() // block_bytes
    ne = _n_elems(shape)
    if n_blocks == 0:
        # Fewer bytes than one super-block: Triton cannot launch an empty grid.
        # Mirror the eager tail literally, so this branch cannot invent a
        # behaviour difference - including how it fails when ne > 0.
        return torch.empty(0, dtype=torch.bfloat16, device=data.device)[:ne].reshape(shape)

    out = torch.empty(n_blocks * QK_K, dtype=torch.bfloat16, device=data.device)
    grid = (triton.cdiv(n_blocks, BLOCKS_PER_PROG),)
    kernel[grid](
        data,
        out,
        n_blocks,
        BPP=BLOCKS_PER_PROG,
        num_warps=NUM_WARPS,
        enable_fp_fusion=False,
    )
    if ne < out.numel():
        out = out[:ne]
    return out.reshape(shape)


def dequant_q4_k(raw: torch.Tensor, shape) -> torch.Tensor:
    return _run("q4_k", raw, shape)


def dequant_q5_k(raw: torch.Tensor, shape) -> torch.Tensor:
    return _run("q5_k", raw, shape)


def dequant_q6_k(raw: torch.Tensor, shape) -> torch.Tensor:
    return _run("q6_k", raw, shape)


def jit_cache_size() -> int:
    """Total number of compiled variants across the three kernels.

    Used by the selfcheck (C8) to prove that a whole job's worth of shapes does
    not trigger a single new JIT compilation after the first call of each type.
    ``JITFunction.device_caches[dev]`` is a tuple whose first element is the
    compiled-kernel dict (Triton 3.5).
    """
    total = 0
    for kernel, _ in _KERNELS.values():
        for entry in getattr(kernel, "device_caches", {}).values():
            total += len(entry[0])
    return total
