"""Standalone self-check for the fused GGUF dequantisation kernels.

Run with the ENGINE venv (needs torch, triton and a CUDA GPU; the app venv has
none of them, and .venv-engine has no pytest, which is why this is a script and
not a test module):

    .venv-engine\\Scripts\\python.exe -m engine.gguf.dequant_triton_selfcheck

Same conventions as sage_selfcheck / block_swap_prefetch_selfcheck: NOT a pytest
module, every check either PASSes or FAILs loudly, nothing is ever skipped, and
the exit code is 0 only when all of them pass.

What the ten checks prove, in one line each:

  C1  Every (quant type, shape) combination that actually occurs in the two
      shipped GGUFs - enumerated from the files, not from a hand-written list -
      is bit-identical between Triton and eager.
  C2  Hand-built boundary blocks are bit-identical: the 6-bit scale/min packing
      at 0x00/0x0F/0x3F/0x80/0xFF, every qh bit position, Q6_K's signed scale
      byte across the 0x7F/0x80 boundary, and fp16 d/dmin at zero, subnormal,
      minimum normal and maximum normal.
  C3  A CPU tensor never reaches Triton (Gemma's embedding dequantises on CPU)
      and comes back from the eager path unchanged.
  C4  The IC-LoRA forward's in-place ``bf16 += delta.to(bf16.dtype)`` gives
      identical bits ON and OFF, including on a shape whose element count is
      not a multiple of 256 (the returned tensor is a slice of a larger buffer).
  C5  An exception from the kernel module degrades every one of ten calls to
      eager, latches once, and warns once.
  C6  A job that asks for the feature but dequantises nothing echoes "on->off".
  C7  A latch does not survive into the next job.
  C8  A whole job's worth of shapes triggers zero new JIT compilations.
  C9  A raw tensor carved out of a bigger buffer at a 512B offset (the
      block-swap prefetch arena's layout) behaves exactly like C1.
  C10 Control run with ``enable_fp_fusion=True``, recorded for the log.

Test payloads are fixed-seed random bytes with VALID fp16 scale fields, at the
real block counts. Fully random scale bytes decode to inf/NaN, which would make
every comparison vacuous.
"""

from __future__ import annotations

import gc
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import torch

from engine.gguf import dequant_triton
from engine.gguf import dequant_triton_kernels as kernels
from engine.gguf.quant_service import (
    _GGML_Q4_K,
    _GGML_Q5_K,
    _GGML_Q6_K,
    _dequant_q4_k,
    _dequant_q5_k,
    _dequant_q6_k,
    dequantize_ggml_tensor,
)

_DEVICE = torch.device("cuda:0")
_QK_K = 256

_BLOCK_BYTES = {_GGML_Q4_K: 144, _GGML_Q5_K: 176, _GGML_Q6_K: 210}
_TYPE_NAME = {_GGML_Q4_K: "Q4_K", _GGML_Q5_K: "Q5_K", _GGML_Q6_K: "Q6_K"}
# The reference is always the eager kernel called DIRECTLY. Going through
# dequantize_ggml_tensor would route the "reference" through Triton too whenever
# the feature is armed, which is exactly when every comparison here happens.
_EAGER = {_GGML_Q4_K: _dequant_q4_k, _GGML_Q5_K: _dequant_q5_k, _GGML_Q6_K: _dequant_q6_k}
# Byte spans of the fp16 scalars inside one super-block.
_SCALE_FIELDS = {_GGML_Q4_K: [(0, 2), (2, 4)], _GGML_Q5_K: [(0, 2), (2, 4)],
                 _GGML_Q6_K: [(208, 210)]}

_MODELS = [
    Path("models/ltx-2.3-gguf/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf"),
    Path("models/gemma-3-12b-it-gguf/gemma-3-12b-it-Q4_K_M.gguf"),
]

# The eager reference is computed in slices of at most this many elements. It
# materialises ~25x its input as int32/fp32 intermediates, so running it in one
# go on Gemma's 1.0-billion-element token embedding would need more VRAM than
# the card has. The Triton call is always made on the FULL real shape (that is
# the thing under test); only the reference is chunked, which is exact because
# every kernel here is per-super-block.
_EAGER_CHUNK_ELEMS = 16 * 1024 * 1024

_RESULTS: list[tuple[str, bool, str]] = []


def _record(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")
    _RESULTS.append((name, passed, detail))


def _run_check(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        _record(name, True)
    except Exception as exc:  # noqa: BLE001 - a broken check must FAIL loudly, never skip
        _record(name, False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        dequant_triton.reset_job()
        gc.collect()
        torch.cuda.empty_cache()


class _LogCapture(logging.Handler):
    """Collect dequant_triton's own warnings for the duration of a block."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []
        self._saved_level = logging.NOTSET

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    def __enter__(self) -> "_LogCapture":
        self._saved_level = dequant_triton.logger.level
        dequant_triton.logger.setLevel(logging.DEBUG)
        dequant_triton.logger.addHandler(self)
        return self

    def __exit__(self, *exc_info: object) -> bool:
        dequant_triton.logger.removeHandler(self)
        dequant_triton.logger.setLevel(self._saved_level)
        return False


# --------------------------------------------------------------------------- #
# Payload construction                                                         #
# --------------------------------------------------------------------------- #


def _make_raw(
    n_blocks: int,
    ggml_type: int,
    seed: int,
    offset_bytes: int = 0,
    device: torch.device = _DEVICE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed payload for ``n_blocks`` super-blocks, with valid fp16 scales.

    Returns ``(raw, owner)``. When ``offset_bytes`` is non-zero the payload is
    carved out of a bigger buffer so ``raw.storage_offset()`` is non-zero, which
    is what the block-swap prefetch arena produces (C9); ``owner`` must be kept
    alive for as long as ``raw`` is used.
    """
    bb = _BLOCK_BYTES[ggml_type]
    g = torch.Generator(device=device).manual_seed(seed)
    blocks = torch.randint(0, 256, (n_blocks, bb), dtype=torch.uint8,
                           device=device, generator=g)
    for lo, hi in _SCALE_FIELDS[ggml_type]:
        vals = torch.rand((n_blocks,), device=device, generator=g) * 0.05 + 1e-4
        blocks[:, lo:hi] = vals.to(torch.float16).view(torch.uint8).reshape(n_blocks, 2)
    flat = blocks.reshape(-1)
    if offset_bytes == 0:
        return flat, flat
    owner = torch.empty(offset_bytes + flat.numel(), dtype=torch.uint8, device=device)
    owner[offset_bytes:] = flat
    raw = owner[offset_bytes:]
    assert raw.storage_offset() == offset_bytes, raw.storage_offset()
    del flat, blocks
    return raw, owner


def _n_elems(shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _eager_matches(raw: torch.Tensor, ggml_type: int, got: torch.Tensor) -> tuple[int, float, int, int]:
    """Compare ``got`` against the eager kernel, computed in VRAM-sized slices.

    Returns (differing bf16 bit patterns, max finite abs diff, NaNs in got,
    NaNs in the reference).
    """
    bb = _BLOCK_BYTES[ggml_type]
    n_blocks = raw.numel() // bb
    flat = got.reshape(-1)
    chunk_blocks = max(1, _EAGER_CHUNK_ELEMS // _QK_K)

    n_diff = 0
    max_abs = 0.0
    n_nan_got = 0
    n_nan_ref = 0
    for b0 in range(0, n_blocks, chunk_blocks):
        b1 = min(b0 + chunk_blocks, n_blocks)
        n_here = (b1 - b0) * _QK_K
        piece = flat[b0 * _QK_K: b0 * _QK_K + n_here]
        if piece.numel() == 0:      # trailing partial block dropped by both paths
            continue
        ref = _EAGER[ggml_type](raw[b0 * bb: b1 * bb], (n_here,), torch.bfloat16)
        ref = ref[: piece.numel()]
        pi = piece.reshape(-1).contiguous().view(torch.int16)
        ri = ref.reshape(-1).contiguous().view(torch.int16)
        n_diff += int((pi != ri).sum())
        pf = piece.float()
        rf = ref.float()
        n_nan_got += int(torch.isnan(pf).sum())
        n_nan_ref += int(torch.isnan(rf).sum())
        finite = torch.isfinite(pf) & torch.isfinite(rf)
        if bool(finite.any()):
            max_abs = max(max_abs, float((pf[finite] - rf[finite]).abs().max()))
        del ref, pf, rf, finite, pi, ri
    return n_diff, max_abs, n_nan_got, n_nan_ref


def _enumerate_real_combos() -> list[tuple[int, tuple[int, ...]]]:
    """Every distinct (quant type, shape) in the two shipped GGUFs.

    Read out of the files rather than transcribed, so a model swap cannot leave
    the check testing shapes that no longer exist.
    """
    from gguf import GGUFReader

    combos: set[tuple[int, tuple[int, ...]]] = set()
    for rel in _MODELS:
        path = Path(__file__).resolve().parents[2] / rel
        if not path.exists():
            raise FileNotFoundError(f"GGUF required by C1 is missing: {path}")
        reader = GGUFReader(str(path), mode="r")
        for tensor in reader.tensors:
            ggml_type = int(tensor.tensor_type)
            if ggml_type not in _BLOCK_BYTES:
                continue
            # GGUF stores ne in column-major order, like the loader does.
            shape = tuple(int(x) for x in reversed(tensor.shape.tolist()))
            combos.add((ggml_type, shape))
        del reader
    # Ascending element count: the first call of each type triggers the
    # production first-call self-verification, which runs eager on the WHOLE
    # tensor. Smallest first keeps that affordable.
    return sorted(combos, key=lambda c: (_n_elems(c[1]), c[0]))


def _sweep(combos: list[tuple[int, tuple[int, ...]]], offset_bytes: int, label: str) -> None:
    """Drive every combo through the production call site and compare to eager."""
    dequant_triton.set_job(True)
    calls_before = dequant_triton._CALLS
    t0 = time.perf_counter()
    for ggml_type, shape in combos:
        ne = _n_elems(shape)
        n_blocks = (ne + _QK_K - 1) // _QK_K
        raw, owner = _make_raw(n_blocks, ggml_type, seed=1234 + ggml_type, offset_bytes=offset_bytes)
        got = dequantize_ggml_tensor(raw, ggml_type, shape, torch.bfloat16)
        if tuple(got.shape) != shape:
            raise AssertionError(f"{_TYPE_NAME[ggml_type]}{shape}: got shape {tuple(got.shape)}")
        n_diff, max_abs, nan_got, nan_ref = _eager_matches(raw, ggml_type, got)
        if n_diff or max_abs != 0.0 or nan_got or nan_ref:
            raise AssertionError(
                f"{_TYPE_NAME[ggml_type]}{shape}: {n_diff} differing bf16 bit patterns, "
                f"max_abs={max_abs}, nan(got/ref)={nan_got}/{nan_ref}"
            )
        del raw, owner, got
        gc.collect()
        torch.cuda.empty_cache()
    used = dequant_triton._CALLS - calls_before
    if used != len(combos):
        raise AssertionError(f"only {used}/{len(combos)} combos went through Triton")
    dequant_triton.reset_job()
    if dequant_triton.last_used() != "on":
        raise AssertionError(f'echo was "{dequant_triton.last_used()}", expected "on"')
    print(f"       {label}: {len(combos)} distinct (type, shape) combos, "
          f"0 differing bits, {time.perf_counter() - t0:.1f}s")


# --------------------------------------------------------------------------- #
# C1 / C9 - real GGUF shapes                                                   #
# --------------------------------------------------------------------------- #

_COMBOS: list[tuple[int, tuple[int, ...]]] = []


def check_c1_real_shapes() -> None:
    global _COMBOS
    _COMBOS = _enumerate_real_combos()
    by_type: dict[int, int] = {}
    for ggml_type, _ in _COMBOS:
        by_type[ggml_type] = by_type.get(ggml_type, 0) + 1
    print("       enumerated from the shipped GGUFs: "
          + ", ".join(f"{_TYPE_NAME[t]} x{n}" for t, n in sorted(by_type.items())))
    biggest = max(_COMBOS, key=lambda c: _n_elems(c[1]))
    print(f"       largest combo: {_TYPE_NAME[biggest[0]]}{biggest[1]} "
          f"= {_n_elems(biggest[1]):,} elements")
    _sweep(_COMBOS, offset_bytes=0, label="contiguous")


def check_c9_offset_arena() -> None:
    if not _COMBOS:
        raise AssertionError("C1 did not run, so there is nothing to re-run")
    _sweep(_COMBOS, offset_bytes=512, label="512B storage_offset")


# --------------------------------------------------------------------------- #
# C2 - hand-built boundary blocks                                              #
# --------------------------------------------------------------------------- #

# fp16 bit patterns for d / dmin: +0, -0, smallest subnormal, largest
# subnormal, smallest normal, 1.0, -1.0, largest normal, and two arbitrary
# mid-range values. inf/NaN are deliberately absent: they are not valid GGUF
# scales, and a NaN would only test how two libraries spell NaN.
_F16_PATTERNS = [0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0xBC00,
                 0x7BFF, 0xFBFF, 0x1234, 0x5678, 0x0002]

# Scale-byte fills called out by the plan. 0x80 and 0xFF matter twice over: they
# are the sign boundary of Q6_K's int8 scales AND the all-ones case of the 6-bit
# scale/min packing.
_SCALE_FILLS = [0x00, 0x0F, 0x3F, 0x80, 0xFF, 0x40, 0x7F, 0xC0]


def _boundary_blocks(ggml_type: int) -> torch.Tensor:
    """Hand-built super-blocks that sweep every payload byte and every fp16 case.

    Block ``b`` sets payload byte ``p`` to ``(b + p) & 0xFF``, so across 256
    blocks every payload position takes every one of the 256 possible values -
    which covers every qh bit position, every nibble pair and (for Q6_K) the
    whole signed scale range including 0x7F/0x80/0xFF. The first blocks are then
    overwritten with uniform fills so the "all scale bytes equal X" cases from
    the plan are present explicitly rather than only implicitly.
    """
    bb = _BLOCK_BYTES[ggml_type]
    n = 256 + len(_SCALE_FILLS) + 8
    b = torch.arange(n, dtype=torch.int32).reshape(n, 1)
    p = torch.arange(bb, dtype=torch.int32).reshape(1, bb)
    blocks = ((b + p) & 0xFF).to(torch.uint8)

    # Explicit uniform fills of the scale region.
    if ggml_type == _GGML_Q6_K:
        sc_lo, sc_hi = 192, 208
    else:
        sc_lo, sc_hi = 4, 16
    for i, fill in enumerate(_SCALE_FILLS):
        blocks[256 + i, sc_lo:sc_hi] = fill

    # Q5_K: eight blocks whose qh region carries exactly one bit each, so every
    # 5th-bit position is exercised in isolation rather than only in company.
    if ggml_type == _GGML_Q5_K:
        for bit in range(8):
            blocks[256 + len(_SCALE_FILLS) + bit, 16:48] = 1 << bit
    # Q6_K: eight blocks whose qh region is a single bit pair.
    if ggml_type == _GGML_Q6_K:
        for bit in range(8):
            blocks[256 + len(_SCALE_FILLS) + bit, 128:192] = 1 << bit
    # Q4_K has no qh; give it eight all-nibble blocks for symmetry.
    if ggml_type == _GGML_Q4_K:
        for i, fill in enumerate([0x00, 0x0F, 0xF0, 0xFF, 0x10, 0x01, 0x88, 0x77]):
            blocks[256 + len(_SCALE_FILLS) + i, 16:144] = fill

    # fp16 scalars cycle through the interesting patterns.
    for lo, hi in _SCALE_FIELDS[ggml_type]:
        pat = torch.tensor(
            [_F16_PATTERNS[(k + (0 if lo == 0 else 5)) % len(_F16_PATTERNS)] for k in range(n)],
            dtype=torch.int32,
        )
        blocks[:, lo] = (pat & 0xFF).to(torch.uint8)
        blocks[:, hi - 1] = ((pat >> 8) & 0xFF).to(torch.uint8)

    return blocks.reshape(-1).to(_DEVICE)


def check_c2_boundary_blocks() -> None:
    dequant_triton.set_job(True)
    for ggml_type in (_GGML_Q4_K, _GGML_Q5_K, _GGML_Q6_K):
        raw = _boundary_blocks(ggml_type)
        n_blocks = raw.numel() // _BLOCK_BYTES[ggml_type]
        shape = (n_blocks * _QK_K,)
        calls_before = dequant_triton._CALLS
        got = dequantize_ggml_tensor(raw, ggml_type, shape, torch.bfloat16)
        # Without this the check is vacuous: if the kernel were wrong, the
        # production first-call verification would latch, `got` would BE the
        # eager result, and the comparison below would trivially pass.
        if dequant_triton._CALLS != calls_before + 1:
            raise AssertionError(
                f"{_TYPE_NAME[ggml_type]}: the result did not come from Triton "
                "(latched, so the comparison would compare eager against itself)"
            )
        n_diff, max_abs, nan_got, nan_ref = _eager_matches(raw, ggml_type, got)
        if n_diff or max_abs != 0.0 or nan_got != nan_ref:
            raise AssertionError(
                f"{_TYPE_NAME[ggml_type]}: {n_diff} differing bit patterns, "
                f"max_abs={max_abs}, nan(got/ref)={nan_got}/{nan_ref}"
            )
        print(f"       {_TYPE_NAME[ggml_type]}: {n_blocks} boundary blocks, "
              f"{n_blocks * _QK_K:,} values, 0 differing bits (NaN {nan_got})")
        del raw, got
    dequant_triton.reset_job()


# --------------------------------------------------------------------------- #
# C3 - the CPU guard                                                           #
# --------------------------------------------------------------------------- #


def check_c3_cpu_guard() -> None:
    dequant_triton.set_job(True)
    calls_before = dequant_triton._CALLS
    raw_cpu, _ = _make_raw(64, _GGML_Q4_K, seed=99, device=torch.device("cpu"))
    shape = (64 * _QK_K,)
    got = dequantize_ggml_tensor(raw_cpu, _GGML_Q4_K, shape, torch.bfloat16)
    if got.device.type != "cpu":
        raise AssertionError(f"a CPU input came back on {got.device}")
    if dequant_triton._CALLS != calls_before:
        raise AssertionError("a CPU tensor reached the Triton path")

    # Same bytes on the GPU DO go through Triton, which is what makes the check
    # above a statement about the device guard and not about the job flag.
    raw_gpu = raw_cpu.to(_DEVICE)
    got_gpu = dequantize_ggml_tensor(raw_gpu, _GGML_Q4_K, shape, torch.bfloat16)
    if dequant_triton._CALLS != calls_before + 1:
        raise AssertionError("the GPU control call did not reach Triton")
    a = got.reshape(-1).view(torch.int16)
    b = got_gpu.cpu().reshape(-1).view(torch.int16)
    if not torch.equal(a, b):
        raise AssertionError(f"{int((a != b).sum())} values differ between the CPU and GPU paths")
    print("       CPU input stayed on eager (_CALLS unchanged); the same bytes on "
          "GPU went through Triton and matched bit for bit")
    dequant_triton.reset_job()


# --------------------------------------------------------------------------- #
# C4 - IC-LoRA in-place add, including a non-multiple-of-256 shape             #
# --------------------------------------------------------------------------- #


def _ic_lora_result(raw: torch.Tensor, ggml_type: int, shape, a: torch.Tensor,
                    b: torch.Tensor, strength: float) -> torch.Tensor:
    """The exact expression from ``quant_service.ggml_linear_forward``."""
    bf16 = dequantize_ggml_tensor(raw, ggml_type, shape, torch.bfloat16)
    delta = torch.matmul(b.to(torch.float32) * strength, a.to(torch.float32))
    bf16 += delta.to(bf16.dtype)
    return bf16


def check_c4_ic_lora_inplace() -> None:
    cases = [
        (_GGML_Q4_K, (512, 256)),      # ne = 131072, an exact multiple of 256
        (_GGML_Q6_K, (100, 100)),      # ne = 10000 -> 40 blocks, 240 values dropped
        (_GGML_Q5_K, (37, 91)),        # ne = 3367 -> 14 blocks, 217 values dropped
    ]
    for ggml_type, shape in cases:
        ne = _n_elems(shape)
        n_blocks = (ne + _QK_K - 1) // _QK_K
        raw, _owner = _make_raw(n_blocks, ggml_type, seed=555)
        rank = 4
        a = torch.randn(rank, shape[1], device=_DEVICE, dtype=torch.bfloat16)
        b = torch.randn(shape[0], rank, device=_DEVICE, dtype=torch.bfloat16)

        dequant_triton.set_job(False)
        off = _ic_lora_result(raw, ggml_type, shape, a, b, 0.85)
        dequant_triton.reset_job()

        dequant_triton.set_job(True)
        calls_before = dequant_triton._CALLS
        on = _ic_lora_result(raw, ggml_type, shape, a, b, 0.85)
        if dequant_triton._CALLS != calls_before + 1:
            raise AssertionError("the ON case did not go through Triton")
        dequant_triton.reset_job()

        oi = off.reshape(-1).contiguous().view(torch.int16)
        ni = on.reshape(-1).contiguous().view(torch.int16)
        if not torch.equal(oi, ni):
            raise AssertionError(
                f"{_TYPE_NAME[ggml_type]}{shape}: {int((oi != ni).sum())} values differ "
                "after the in-place IC-LoRA add"
            )
        dropped = n_blocks * _QK_K - ne
        print(f"       {_TYPE_NAME[ggml_type]}{shape}: ne={ne:,}, {dropped} tail values "
              f"dropped, in-place add identical")
        del raw, _owner, a, b, off, on


# --------------------------------------------------------------------------- #
# C5 / C6 / C7 - degradation and job scoping                                   #
# --------------------------------------------------------------------------- #


def check_c5_exception_fallback() -> None:
    ggml_type = _GGML_Q4_K
    raw, _owner = _make_raw(64, ggml_type, seed=42)
    shape = (64 * _QK_K,)

    dequant_triton.set_job(False)
    ref = dequantize_ggml_tensor(raw, ggml_type, shape, torch.bfloat16)
    dequant_triton.reset_job()

    original = dequant_triton._kernels
    boom = 0

    def exploding_kernels():
        nonlocal boom
        boom += 1
        raise RuntimeError("injected: Triton is on fire")

    dequant_triton._kernels = exploding_kernels
    try:
        dequant_triton.set_job(True)
        with _LogCapture() as logs:
            for i in range(10):
                got = dequantize_ggml_tensor(raw, ggml_type, shape, torch.bfloat16)
                gi = got.reshape(-1).view(torch.int16)
                ri = ref.reshape(-1).view(torch.int16)
                if not torch.equal(gi, ri):
                    raise AssertionError(f"call {i} did not match the eager reference")
        if dequant_triton._CALLS != 0:
            raise AssertionError("a call was counted as having used Triton")
        if boom != 1:
            raise AssertionError(f"_kernels() was entered {boom} times, expected 1 (the latch)")
        if len(logs.messages) != 1:
            raise AssertionError(f"expected exactly 1 warning, got {len(logs.messages)}: {logs.messages}")
        if "injected" not in logs.messages[0]:
            raise AssertionError(f"the warning did not name the cause: {logs.messages[0]}")
        dequant_triton.reset_job()
        if dequant_triton.last_used() != "on->off":
            raise AssertionError(f'echo was "{dequant_triton.last_used()}", expected "on->off"')
    finally:
        dequant_triton._kernels = original
    print("       10/10 calls matched eager, _kernels() entered once (latched), "
          '1 warning, echo "on->off"')


def check_c6_zero_calls_echo() -> None:
    dequant_triton.set_job(True)
    if not dequant_triton.enabled():
        raise AssertionError("set_job(True) did not enable the feature")
    dequant_triton.reset_job()
    if dequant_triton.last_used() != "on->off":
        raise AssertionError(f'echo was "{dequant_triton.last_used()}", expected "on->off"')

    dequant_triton.set_job(False)
    dequant_triton.reset_job()
    if dequant_triton.last_used() != "off":
        raise AssertionError(f'echo was "{dequant_triton.last_used()}", expected "off"')
    print('       asked-but-never-called -> "on->off"; never asked -> "off"')


def check_c7_latch_cleared_next_job() -> None:
    raw, _owner = _make_raw(16, _GGML_Q4_K, seed=7)
    shape = (16 * _QK_K,)

    original = dequant_triton._kernels
    dequant_triton._kernels = lambda: (_ for _ in ()).throw(RuntimeError("injected"))
    try:
        dequant_triton.set_job(True)
        with _LogCapture():
            dequantize_ggml_tensor(raw, _GGML_Q4_K, shape, torch.bfloat16)
        if dequant_triton.enabled():
            raise AssertionError("job 1 did not latch")
        dequant_triton.reset_job()
    finally:
        dequant_triton._kernels = original

    dequant_triton.set_job(True)
    if not dequant_triton.enabled():
        raise AssertionError("job 2 inherited job 1's latch")
    got = dequantize_ggml_tensor(raw, _GGML_Q4_K, shape, torch.bfloat16)
    if dequant_triton._CALLS != 1:
        raise AssertionError("job 2 did not actually use Triton")
    n_diff, _, _, _ = _eager_matches(raw, _GGML_Q4_K, got)
    if n_diff:
        raise AssertionError(f"job 2 produced {n_diff} differing values")
    dequant_triton.reset_job()
    if dequant_triton.last_used() != "on":
        raise AssertionError(f'job 2 echo was "{dequant_triton.last_used()}"')
    print('       job 1 latched -> "on->off"; job 2 ran on Triton -> "on"')


# --------------------------------------------------------------------------- #
# C8 - no JIT compilation mid-job                                              #
# --------------------------------------------------------------------------- #


def check_c8_no_recompiles() -> None:
    if not _COMBOS:
        raise AssertionError("C1 did not run, so there are no shapes to replay")
    before = kernels.jit_cache_size()
    if before != 3:
        raise AssertionError(
            f"expected exactly 3 compiled kernels after C1, found {before} - "
            "something is specialising per shape"
        )
    dequant_triton.set_job(True)
    # Replay every shape once more, in the order a forward pass would see them.
    for ggml_type, shape in _COMBOS:
        ne = _n_elems(shape)
        if ne > 32 * 1024 * 1024:
            # The point is the launch, not the bandwidth: the biggest shapes are
            # already covered by C1 and re-running them here would only re-burn
            # seconds of DRAM traffic. Their block counts are still represented
            # by the mid-size shapes, and n_blocks is do_not_specialize anyway.
            continue
        n_blocks = (ne + _QK_K - 1) // _QK_K
        raw, _owner = _make_raw(n_blocks, ggml_type, seed=3)
        dequantize_ggml_tensor(raw, ggml_type, shape, torch.bfloat16)
        del raw, _owner
    dequant_triton.reset_job()
    after = kernels.jit_cache_size()
    if after != before:
        raise AssertionError(f"the JIT cache grew from {before} to {after} mid-job")
    print(f"       {before} compiled kernels before and after replaying the shape set")


# --------------------------------------------------------------------------- #
# C10 - enable_fp_fusion=True control                                          #
# --------------------------------------------------------------------------- #


def check_c10_fp_fusion_control() -> None:
    """Record, do not judge: run the kernels with FMA contraction allowed.

    The production launch passes ``enable_fp_fusion=False``. The argument that
    it is unnecessary is that both products are exact in fp32, so contraction
    cannot move a rounding. This check prints whether that argument holds on the
    real hardware; it passes either way, because the production setting does not
    depend on the answer.
    """
    import triton

    results = []
    for ggml_type, kernel in (
        (_GGML_Q4_K, kernels._q4_k_kernel),
        (_GGML_Q5_K, kernels._q5_k_kernel),
        (_GGML_Q6_K, kernels._q6_k_kernel),
    ):
        n_blocks = 4096
        raw, _owner = _make_raw(n_blocks, ggml_type, seed=808)
        out = torch.empty(n_blocks * _QK_K, dtype=torch.bfloat16, device=_DEVICE)
        grid = (triton.cdiv(n_blocks, kernels.BLOCKS_PER_PROG),)
        kernel[grid](raw, out, n_blocks, BPP=kernels.BLOCKS_PER_PROG,
                     num_warps=kernels.NUM_WARPS, enable_fp_fusion=True)
        n_diff, max_abs, _, _ = _eager_matches(raw, ggml_type, out)
        results.append((ggml_type, n_diff, max_abs))
        print(f"       {_TYPE_NAME[ggml_type]} with enable_fp_fusion=True: "
              f"{n_diff} differing bf16 bit patterns, max_abs={max_abs}")
        del raw, _owner, out
    total = sum(r[1] for r in results)
    print("       -> FMA contraction "
          + ("does not change any result (the fp32-exactness argument holds); "
             "enable_fp_fusion=False is belt-and-braces"
             if total == 0 else
             f"changes {total} values; enable_fp_fusion=False is load-bearing"))


# --------------------------------------------------------------------------- #


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    if not torch.cuda.is_available():
        print("[FAIL] no CUDA device: the fused dequant kernels cannot be exercised")
        return 1

    print(f"[env] {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    try:
        import triton
        print(f"[env] triton {triton.__version__}  BLOCKS_PER_PROG={kernels.BLOCKS_PER_PROG} "
              f"num_warps={kernels.NUM_WARPS}")
    except ImportError:
        print("[FAIL] triton is not importable in this venv")
        return 1

    checks: list[tuple[str, Callable[[], None]]] = [
        ("C1  every (type, shape) in the shipped GGUFs is bit-identical", check_c1_real_shapes),
        ("C2  hand-built boundary blocks are bit-identical", check_c2_boundary_blocks),
        ("C3  a CPU tensor never reaches Triton", check_c3_cpu_guard),
        ("C4  the IC-LoRA in-place add is identical ON and OFF", check_c4_ic_lora_inplace),
        ("C8  replaying a job's shapes triggers no new JIT compilation", check_c8_no_recompiles),
        ("C9  a 512B-offset arena slice behaves exactly like C1", check_c9_offset_arena),
        ("C10 enable_fp_fusion=True control run (recorded, not judged)", check_c10_fp_fusion_control),
        ("C5  an exception degrades all 10 calls, latches once, warns once", check_c5_exception_fallback),
        ("C6  asked-but-never-called echoes on->off", check_c6_zero_calls_echo),
        ("C7  a latch does not survive into the next job", check_c7_latch_cleared_next_job),
    ]

    for name, fn in checks:
        _run_check(name, fn)

    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_total = len(_RESULTS)
    print(f"\n{n_pass}/{n_total} checks passed")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
