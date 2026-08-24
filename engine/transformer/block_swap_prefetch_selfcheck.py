"""Standalone self-check for engine.transformer.block_swap_prefetch.

Run with the ENGINE venv (needs torch + a CUDA GPU — the app venv has neither,
and .venv-engine has no pytest, which is why this is a script and not a test
module):

    .venv-engine\\Scripts\\python.exe -m engine.transformer.block_swap_prefetch_selfcheck

Same conventions as nag_selfcheck / vsf_selfcheck / sage_selfcheck: NOT a pytest
module, every check either PASSes or FAILs loudly (nothing is skipped), exit code
0 only when all of them pass.

The model under test is a synthetic 12-block transformer whose blocks carry one
of every tensor kind a real LTX block has — a GGUF-compressed
``GGMLQuantizedTensor`` buffer behind a dequantising Linear, a bias Parameter,
LayerNorm weight/bias, a block-level ``scale_shift_table`` Parameter, IC-LoRA
``persistent=False`` buffers wired through the real specs mechanism, plus an
odd-sized buffer and a 0-dim buffer that exercise the arena's alignment padding
and the ``reshape(-1)`` guard. Everything runs inside ``torch.inference_mode()``
because that is where the production install() runs.

What the 15 checks prove, in one line each:

  C1  ON and OFF produce BIT-identical output (the only thing that changed is
      how the bytes travel, so anything less is a bug).
  C2  The CPU masters are never written to.
  C3  Residency never exceeds blocks_on_gpu + 2, and a pass ends with exactly 1.
  C4  After the structural cold-start miss at the head of a pass, no block is
      ever waited for un-issued.
  C5  The GPU-side GGMLQuantizedTensor keeps its subclass, quant metadata and
      float-shape masquerade, and its bytes match the CPU master exactly.
  C6  A machine that cannot pin memory degrades to the synchronous path.
  C7  blocks_on_gpu >= total short-circuits before anything is allocated.
  C8  install -> passes -> teardown, three times, leaks no VRAM and no pinned
      memory (the E9 finally path).
  C9  20 passes of heavy compute stay bit-identical, and S1b is a real
      cross-stream dependency (proven by removing it and watching the copy
      complete early).
  C10 An exception mid-pass leaves nothing behind; the next install completes.
  C11 Arena layout: 512B alignment, no overlap, view(dtype) legality.
  C12 Slot enumeration finds nested, non-persistent and module-level tensors.
  C13 The window function's boundaries.
  C14 Weight tying is detected and warned about.
  C15 A SUBCLASS of GGMLQuantizedTensor is rebuilt as that same subclass, with
      its quant metadata intact (engine25's Ltx25GGMLTensor depends on it).
"""

from __future__ import annotations

import gc
import logging
import sys
import time
import traceback
from typing import Callable

import torch
import torch.nn as nn

from engine.gguf.ic_lora_common import IC_LORA_SPECS_ATTR
from engine.gguf.quant_service import (
    _GGML_Q4_K,
    GGMLQuantizedTensor,
    _patch_linear_for_ggml_dequant,
)
from engine.transformer import block_swap_prefetch as pf
from engine.transformer.block_swap_prefetch import (
    PinnedStagingPool,
    PrefetchEngine,
    _window,
)
from engine.transformer.block_swap_service import BlockSwapService

_DEVICE = torch.device("cuda:0")

# Q4_K packs 256 weights into 144 bytes.
_QK_K = 256
_Q4_K_BLOCK_BYTES = 144

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
    except Exception as exc:  # noqa: BLE001 — a broken check must FAIL loudly, never skip
        _record(name, False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        gc.collect()
        torch.cuda.empty_cache()


class _LogCapture(logging.Handler):
    """Collect the prefetch module's own log records for the duration of a block."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []
        self._saved_level = logging.NOTSET

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    def __enter__(self) -> "_LogCapture":
        self._saved_level = pf.logger.level
        pf.logger.setLevel(logging.DEBUG)
        pf.logger.addHandler(self)
        return self

    def __exit__(self, *exc_info: object) -> bool:
        pf.logger.removeHandler(self)
        pf.logger.setLevel(self._saved_level)
        return False


# --------------------------------------------------------------------------- #
# The synthetic model                                                          #
# --------------------------------------------------------------------------- #


def _random_q4_k_bytes(n_weights: int) -> torch.Tensor:
    """Plausible Q4_K payload for ``n_weights`` values.

    Random bytes would do for the transfer itself, but bytes 0:4 of every block
    are the fp16 ``d``/``dmin`` scales and random ones decode to inf/NaN often
    enough that the dequantised weights (and therefore every output comparison)
    would be NaN — and NaN != NaN, so the parity checks would fail for a reason
    that has nothing to do with prefetching. Only those 4 bytes are constrained;
    the 6-bit scale block and the packed nibbles stay random.
    """
    n_blocks = n_weights // _QK_K
    raw = torch.randint(0, 256, (n_blocks, _Q4_K_BLOCK_BYTES), dtype=torch.uint8)
    d = (torch.rand(n_blocks) * 0.002 + 0.0005).to(torch.float16)
    dmin = (torch.rand(n_blocks) * 0.001).to(torch.float16)
    raw[:, 0:2] = d.view(torch.uint8).reshape(n_blocks, 2)
    raw[:, 2:4] = dmin.view(torch.uint8).reshape(n_blocks, 2)
    return raw.reshape(-1).contiguous()


class _DummyBlock(nn.Module):
    """One block carrying every tensor kind a production block has.

    ``repeat`` re-runs the quantised Linear to make the block expensive enough
    that the transfers actually have compute to hide behind (C9).
    """

    def __init__(self, dim: int, rank: int = 8, repeat: int = 1, tie_with: "_DummyBlock | None" = None) -> None:
        super().__init__()
        self.repeat = repeat

        self.lin = nn.Linear(dim, dim, bias=True).to(torch.bfloat16)
        _patch_linear_for_ggml_dequant(self.lin)     # weight -> meta buffer + dequant forward
        if tie_with is None:
            self.lin._buffers["weight"] = GGMLQuantizedTensor(
                _random_q4_k_bytes(dim * dim), _GGML_Q4_K, (dim, dim)
            )
        else:
            # Weight tying (C14): the SAME storage under two owners.
            self.lin._buffers["weight"] = tie_with.lin._buffers["weight"]

        # IC-LoRA phase-B style: non-persistent buffers + the real specs attr,
        # so the LoRA delta actually participates in the output.
        self.lin.register_buffer(
            "_ic_lora_A_0", (torch.randn(rank, dim) * 0.01).to(torch.bfloat16), persistent=False
        )
        self.lin.register_buffer(
            "_ic_lora_B_0", (torch.randn(dim, rank) * 0.01).to(torch.bfloat16), persistent=False
        )
        setattr(self.lin, IC_LORA_SPECS_ATTR, [("_ic_lora_A_0", "_ic_lora_B_0", 0.7)])

        self.norm = nn.LayerNorm(dim).to(torch.bfloat16)
        # Module-level parameter, like scale_shift_table / audio_scale_shift_table.
        self.scale_shift_table = nn.Parameter(torch.randn(2, dim).to(torch.bfloat16))
        # Alignment stressors: 7 bf16 values = 14 bytes (needs padding), and a
        # 0-dim buffer (view(dtype) would raise on it without the reshape(-1)).
        self.register_buffer("odd_bytes", torch.randn(7).to(torch.bfloat16))
        self.register_buffer("zero_dim", torch.tensor(0.25, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        for _ in range(self.repeat - 1):
            h = self.lin(x) + h * 0.0
        h = self.norm(h)
        h = h * self.scale_shift_table[0] + self.scale_shift_table[1]
        return h + self.odd_bytes.sum() * self.zero_dim


class _DummyTransformer(nn.Module):
    """`_get_blocks` finds ``transformer_blocks`` by name, like the real model."""

    def __init__(self, blocks: list[nn.Module]) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            x = block(x)
        return x


def _build_model(n_blocks: int = 12, dim: int = 256, repeat: int = 1, seed: int = 7) -> _DummyTransformer:
    torch.manual_seed(seed)
    return _DummyTransformer([_DummyBlock(dim, repeat=repeat) for _ in range(n_blocks)])


def _make_input(tokens: int = 64, dim: int = 256, seed: int = 11) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(1, tokens, dim, dtype=torch.bfloat16, device=_DEVICE)


def _run(
    service: BlockSwapService,
    model: _DummyTransformer,
    x: torch.Tensor,
    passes: int,
    prefetch: bool,
    on_entry: Callable[[int], None] | None = None,
    after_pass: Callable[[int], None] | None = None,
) -> list[torch.Tensor]:
    """install() + N forward passes, exactly as a job would."""
    service.prefetch_requested = prefetch
    service.last_prefetch_used = None
    outs: list[torch.Tensor] = []
    with torch.inference_mode():
        service.install(model)
        engine = service._prefetch_engine
        if engine is not None and on_entry is not None:
            original = engine.on_block_forward

            def instrumented(idx: int, _orig=original, _cb=on_entry) -> None:
                _cb(idx)
                _orig(idx)

            engine.on_block_forward = instrumented   # instance attr shadows the method
        for p in range(passes):
            outs.append(model(x).clone())
            if after_pass is not None:
                after_pass(p)
    return outs


def _cpu_snapshot(model: nn.Module) -> list[torch.Tensor]:
    """Byte-level clone of every CPU tensor in the model (masters included)."""
    out = []
    for mod in model.modules():
        for _name, p in mod._parameters.items():
            if p is not None:
                out.append(p.data.detach().cpu().clone())
        for _name, b in mod._buffers.items():
            if isinstance(b, GGMLQuantizedTensor):
                out.append(b.as_subclass(torch.Tensor).reshape(-1).cpu().clone())
            elif isinstance(b, torch.Tensor):
                out.append(b.detach().cpu().clone())
    return out


def _service(blocks_on_gpu: int) -> BlockSwapService:
    return BlockSwapService(blocks_on_gpu=blocks_on_gpu, device=_DEVICE)


# --------------------------------------------------------------------------- #
# C1 / C2: output parity and CPU-master invariance                             #
# --------------------------------------------------------------------------- #


def check_c1_output_parity() -> None:
    # Two independently built models rather than one model installed twice:
    # production hands install() a freshly built transformer every job, and this
    # keeps the OFF run from being contaminated by the ON run's residue.
    model_off, model_on = _build_model(), _build_model()
    if not all(
        torch.equal(a, b) for a, b in zip(_cpu_snapshot(model_off), _cpu_snapshot(model_on))
    ):
        raise AssertionError("the two model builds differ — the comparison would be meaningless")
    x = _make_input()
    service = _service(3)

    off = _run(service, model_off, x, passes=3, prefetch=False)
    if service.last_prefetch_used != "off":
        raise AssertionError(f"OFF job reported last_prefetch_used={service.last_prefetch_used!r}")

    on = _run(service, model_on, x, passes=3, prefetch=True)
    if service.last_prefetch_used != "on":
        raise AssertionError(
            f"prefetch did not engage: last_prefetch_used={service.last_prefetch_used!r}"
        )
    service.teardown_prefetch()

    for i, (a, b) in enumerate(zip(off, on)):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(f"pass {i}: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
        if not torch.equal(a, b):
            diff = (a.float() - b.float()).abs().max().item()
            raise AssertionError(f"pass {i}: outputs differ (max abs diff {diff:g})")
    print(f"       3 passes x 12 blocks, |out|={off[0].shape}, bit-identical ON vs OFF")


def check_c2_master_invariance() -> None:
    model = _build_model()
    x = _make_input()
    before = _cpu_snapshot(model)

    service = _service(3)
    _run(service, model, x, passes=3, prefetch=True)
    if service.last_prefetch_used != "on":
        raise AssertionError("prefetch did not engage")
    service.teardown_prefetch()

    after = _cpu_snapshot(model)
    if len(before) != len(after):
        raise AssertionError(f"tensor count changed: {len(before)} -> {len(after)}")
    for i, (a, b) in enumerate(zip(before, after)):
        if a.dtype != b.dtype or a.shape != b.shape:
            raise AssertionError(f"slot {i}: {a.dtype}{a.shape} -> {b.dtype}{b.shape}")
        if not torch.equal(a, b):
            raise AssertionError(f"slot {i}: the CPU master was modified")
    print(f"       {len(before)} CPU tensors unchanged after 3 prefetched passes")


# --------------------------------------------------------------------------- #
# C3 / C4: residency bound and issue schedule                                  #
# --------------------------------------------------------------------------- #


def check_c3_c4_residency_and_schedule() -> None:
    # Two shapes: a small one, and the production one (48 blocks, 8 resident).
    for n_blocks, bs, passes in ((12, 3, 4), (48, 8, 3)):
        model = _build_model(n_blocks=n_blocks)
        x = _make_input()
        service = _service(bs)

        peak = {"n": 0}
        misses_by_pass: list[list[int]] = []

        def sample(where: str, engine) -> None:
            n = len(engine._state)
            peak["n"] = max(peak["n"], n)
            if n > bs + 2:
                raise AssertionError(
                    f"residency {n} exceeds blocks_on_gpu+2={bs + 2} ({where})"
                )

        def after_pass(_p: int) -> None:
            engine = service._prefetch_engine
            misses_by_pass.append(list(engine._stats["misses"]))
            if len(engine._state) != 1:
                raise AssertionError(
                    f"end of pass: {len(engine._state)} blocks resident, expected exactly 1"
                )

        def on_entry(idx: int) -> None:
            engine = service._prefetch_engine
            sample(f"entering block {idx}", engine)
            if not getattr(engine, "_issue_wrapped", False):
                # The true high-water mark is immediately after an issue, before
                # the outgoing block is released — sampling only at forward entry
                # would under-count by one.
                original_issue = engine._issue

                def instrumented_issue(j: int, _orig=original_issue, _eng=engine) -> None:
                    _orig(j)
                    sample(f"after issuing block {j}", _eng)

                engine._issue = instrumented_issue
                engine._issue_wrapped = True

        _run(service, model, x, passes=passes, prefetch=True,
             on_entry=on_entry, after_pass=after_pass)
        service.teardown_prefetch()

        if peak["n"] != bs + 2:
            raise AssertionError(
                f"peak residency {peak['n']}, expected exactly blocks_on_gpu+2="
                f"{bs + 2} (window + the block being released + last pass's tail)"
            )
        for p, misses in enumerate(misses_by_pass):
            late = [m for m in misses if m != 0]
            if late:
                raise AssertionError(f"pass {p}: blocks {late} were consumed before being issued")
            if p >= 1 and misses != [0]:
                raise AssertionError(
                    f"pass {p}: expected exactly one structural miss on block 0, got {misses}"
                )
        print(
            f"       {n_blocks} blocks / window {bs}: peak residency {peak['n']} "
            f"(limit {bs + 2}), misses per pass {misses_by_pass}"
        )


# --------------------------------------------------------------------------- #
# C5: the GGML tensor survives the trip                                        #
# --------------------------------------------------------------------------- #


def check_c5_ggml_metadata() -> None:
    model = _build_model()
    x = _make_input()
    service = _service(3)
    blocks = list(model.transformer_blocks)
    probe_idx = 5
    seen = {"n": 0}

    def on_entry(idx: int) -> None:
        if idx != probe_idx or seen["n"]:
            return
        engine = service._prefetch_engine
        if idx not in engine._state:
            return          # a cold miss; the next pass will have it prefetched
        seen["n"] += 1
        # The views are installed at ISSUE time, before the copy lands — this
        # probe reads them from the host, so wait for the transfer first (the
        # forward itself gets the same guarantee from S2).
        torch.cuda.synchronize()
        w = blocks[idx].lin._buffers["weight"]
        if not isinstance(w, GGMLQuantizedTensor):
            raise AssertionError(f"device weight is {type(w).__name__}, not GGMLQuantizedTensor")
        if w.device.type != "cuda":
            raise AssertionError(f"device weight is on {w.device}")
        if w._ggml_type != _GGML_Q4_K:
            raise AssertionError(f"_ggml_type={w._ggml_type}, expected {_GGML_Q4_K}")
        if tuple(w.shape) != (256, 256):
            raise AssertionError(f".shape={tuple(w.shape)} — the float-shape masquerade is gone")
        raw_gpu = w.as_subclass(torch.Tensor).reshape(-1).view(torch.uint8)
        raw_cpu = engine._master_u8[idx][
            [n for (_m, n, _p) in engine._layout[idx].slots].index("weight")
        ]
        if raw_gpu.numel() != raw_cpu.numel():
            raise AssertionError(f"byte count {raw_gpu.numel()} != master {raw_cpu.numel()}")
        if not torch.equal(raw_gpu.cpu(), raw_cpu):
            raise AssertionError("the transferred quantised bytes differ from the CPU master")

    _run(service, model, x, passes=2, prefetch=True, on_entry=on_entry)
    service.teardown_prefetch()
    if not seen["n"]:
        raise AssertionError("the probe never ran")
    print("       Q4_K buffer: subclass, _ggml_type, float shape and all bytes intact on GPU")


# --------------------------------------------------------------------------- #
# C6 / C7: degradation paths                                                   #
# --------------------------------------------------------------------------- #


def check_c6_pinned_failure_fallback() -> None:
    model_ref, model = _build_model(), _build_model()
    x = _make_input()
    reference = _run(_service(3), model_ref, x, passes=2, prefetch=False)

    service = _service(3)     # fresh: its pinned pool has not been created yet
    saved = pf._alloc_pinned

    def _refuse(_nbytes: int) -> torch.Tensor:
        raise RuntimeError("simulated cudaHostAlloc failure")

    try:
        pf._alloc_pinned = _refuse
        got = _run(service, model, x, passes=2, prefetch=True)
    finally:
        pf._alloc_pinned = saved

    if service._prefetch_engine is not None:
        raise AssertionError("a prefetch engine survived a pinned-allocation failure")
    if service.last_prefetch_used != "on->off":
        raise AssertionError(
            f"degradation reported as {service.last_prefetch_used!r}, expected 'on->off'"
        )
    for i, (a, b) in enumerate(zip(reference, got)):
        if not torch.equal(a, b):
            raise AssertionError(f"pass {i}: the fallback output differs from the sync path")
    print("       cudaHostAlloc refused -> synchronous path, output unchanged, 'on->off' recorded")


def check_c7_no_swap_needed() -> None:
    model = _build_model(n_blocks=12)
    service = _service(12)                 # blocks_on_gpu >= total
    service.prefetch_requested = True
    with torch.inference_mode():
        service.install(model)
    # No forward here on purpose: install() returned before patching anything,
    # so the blocks are still wherever the loader left them (in production
    # DitCpuLoadService puts the whole model on GPU in exactly this case).
    for block in model.transformer_blocks:
        if getattr(block, "_block_swap_original_forward", None) is not None:
            raise AssertionError("blocks were patched even though no swapping is needed")
    if service._prefetch_engine is not None:
        raise AssertionError("an engine was built even though no swapping is needed")
    if service._pinned_pool is not None:
        raise AssertionError("pinned memory was allocated even though no swapping is needed")
    if service.last_prefetch_used != "off":
        raise AssertionError(f"last_prefetch_used={service.last_prefetch_used!r}, expected 'off'")

    service0 = _service(0)                 # blocks_on_gpu == 0
    service0.prefetch_requested = True
    with torch.inference_mode():
        service0.install(model)
    if service0._prefetch_engine is not None or service0._pinned_pool is not None:
        raise AssertionError("blocks_on_gpu=0 still set prefetching up")
    if service0.last_prefetch_used != "off":
        raise AssertionError(f"last_prefetch_used={service0.last_prefetch_used!r}, expected 'off'")
    print("       blocks_on_gpu >= total and == 0 both short-circuit before any allocation")


# --------------------------------------------------------------------------- #
# C8: no leak across jobs                                                      #
# --------------------------------------------------------------------------- #


def check_c8_no_leak_across_jobs() -> None:
    x = _make_input()
    service = _service(3)
    marks: list[int] = []
    pinned_bytes: list[int] = []
    streams: list[int] = []

    for job in range(3):
        model = _build_model(seed=7)       # a fresh transformer per job, like production
        outs = _run(service, model, x, passes=3, prefetch=True)
        if service.last_prefetch_used != "on":
            raise AssertionError(f"job {job}: prefetch did not engage")
        engine = service._prefetch_engine
        service.teardown_prefetch()        # what the pipeline's finally does
        if engine._state or engine._master or engine._layout:
            raise AssertionError(f"job {job}: teardown left engine state behind")
        if service._prefetch_engine is not None:
            raise AssertionError(f"job {job}: the service still references the engine")
        for block in model.transformer_blocks:
            for _n, p in block.lin._parameters.items():
                if p is not None and p.device.type != "cpu":
                    raise AssertionError(f"job {job}: a parameter was left on the GPU")
            if block.lin._buffers["weight"].device.type != "cpu":
                raise AssertionError(f"job {job}: a GGML buffer was left on the GPU")
        del outs, model, engine
        gc.collect()
        torch.cuda.empty_cache()
        marks.append(torch.cuda.memory_allocated())
        pinned_bytes.append(service._pinned_pool.nbytes)
        streams.append(id(service._xfer_stream))

    drift = abs(marks[-1] - marks[0])
    if drift > 1024 * 1024:
        raise AssertionError(f"VRAM drift across 3 jobs: {drift / 1024:.0f} KB (limit 1 MB)")
    if len(set(pinned_bytes)) != 1:
        raise AssertionError(f"the pinned pool grew across jobs: {pinned_bytes}")
    if len(set(streams)) != 1:
        raise AssertionError("a new transfer stream was created per job")
    print(
        f"       allocated after each job: {marks} bytes (drift {drift}), "
        f"pinned {pinned_bytes[0]} B reused, 1 stream"
    )


# --------------------------------------------------------------------------- #
# C9: stream safety under load, and the S1b dependency itself                  #
# --------------------------------------------------------------------------- #


def check_c9a_stress_bit_identical() -> None:
    n_blocks, bs, passes = 24, 4, 20
    dim, tokens = 512, 2048
    model_off = _build_model(n_blocks=n_blocks, dim=dim, repeat=6)
    model_on = _build_model(n_blocks=n_blocks, dim=dim, repeat=6)
    x = _make_input(tokens=tokens, dim=dim)
    service = _service(bs)

    off = _run(service, model_off, x, passes=1, prefetch=False)[0]
    on = _run(service, model_on, x, passes=passes, prefetch=True)
    service.teardown_prefetch()
    for i, got in enumerate(on):
        if not torch.equal(off, got):
            diff = (off.float() - got.float()).abs().max().item()
            raise AssertionError(f"pass {i} of 20 diverged (max abs diff {diff:g})")
    print(f"       {passes} heavy passes x {n_blocks} blocks: every pass bit-identical to sync")


def _busy_compute(seconds_hint: int = 40) -> torch.Tensor:
    """Queue a few hundred ms of fp32 matmul on the current (compute) stream."""
    a = torch.randn(4096, 4096, device=_DEVICE)
    b = torch.randn(4096, 4096, device=_DEVICE)
    for _ in range(seconds_hint):
        a = a @ b
    return a


def check_c9b_alloc_event_dependency() -> None:
    """S1b must be a real edge: without it the H2D completes while compute runs.

    Proving the WAR hazard by observing corruption would be a race with a
    non-deterministic outcome. What IS deterministic is the ordering: with
    ``xfer.wait_event(alloc_evt)`` the copy cannot finish before the compute
    stream drains, and with the wait removed it finishes immediately.
    """
    poll_window = 0.05

    def measure(skip: bool) -> bool:
        model = _build_model(n_blocks=4)
        service = _service(2)
        service.prefetch_requested = True
        saved = pf._SKIP_ALLOC_EVENT_WAIT
        try:
            pf._SKIP_ALLOC_EVENT_WAIT = skip
            with torch.inference_mode():
                service.install(model)
                engine = service._prefetch_engine
                if engine is None:
                    raise AssertionError("prefetch did not engage")
                torch.cuda.synchronize()
                keep = _busy_compute()          # a few hundred ms on the compute stream
                t0 = time.perf_counter()
                engine._issue(0)
                deadline = time.perf_counter() + poll_window
                evt = engine._state[0].event
                done_early = False
                while time.perf_counter() < deadline:
                    if evt.query():
                        done_early = True
                        break
                torch.cuda.synchronize()
                busy = time.perf_counter() - t0
                del keep
        finally:
            pf._SKIP_ALLOC_EVENT_WAIT = saved
            service.teardown_prefetch()
        if busy < 2 * poll_window:
            raise AssertionError(
                f"the compute burst only lasted {busy * 1e3:.0f} ms — too short to "
                f"distinguish a waiting copy from a free-running one"
            )
        return done_early

    if measure(skip=True) is not True:
        raise AssertionError(
            "with S1b removed the H2D still did not complete during the compute "
            "burst — the experiment cannot distinguish the two cases"
        )
    if measure(skip=False) is not False:
        raise AssertionError(
            "with S1b in place the H2D completed while the compute stream was "
            "still busy: xfer.wait_event(alloc_evt) is not ordering anything"
        )
    print("       S1b removed -> copy lands early; S1b present -> copy waits for the arena")


# --------------------------------------------------------------------------- #
# C10: exception mid-pass                                                      #
# --------------------------------------------------------------------------- #


def check_c10_teardown_after_exception() -> None:
    model_ref, model = _build_model(), _build_model()
    x = _make_input()
    service = _service(3)
    reference = _run(_service(3), model_ref, x, passes=1, prefetch=False)[0]
    del model_ref

    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()

    blocks = list(model.transformer_blocks)
    boom = blocks[7]
    saved_forward = boom.forward

    def explode(*_args, **_kwargs):
        raise RuntimeError("simulated mid-pass failure")

    raised = False
    try:
        with torch.inference_mode():
            service.prefetch_requested = True
            service.install(model)
            boom.forward = explode        # after install, so the patch is in place
            model(x)
    except RuntimeError as exc:
        raised = "simulated mid-pass failure" in str(exc)
    finally:
        boom.forward = saved_forward
    if not raised:
        raise AssertionError("the injected failure did not propagate")

    engine = service._prefetch_engine
    service.teardown_prefetch()           # the job's finally
    if engine._state or engine._master:
        raise AssertionError("teardown after an exception left state behind")
    gc.collect()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated()
    if after - baseline > 1024 * 1024:
        raise AssertionError(f"{(after - baseline) / 1024:.0f} KB of VRAM survived the failure")
    for block in blocks:
        if block.lin._buffers["weight"].device.type != "cpu":
            raise AssertionError("a block was left pointing at GPU memory")

    got = _run(service, model, x, passes=1, prefetch=True)[0]
    service.teardown_prefetch()
    if not torch.equal(reference, got):
        raise AssertionError("the re-installed job's output differs from the sync reference")
    print(f"       exception -> teardown freed everything ({after - baseline} B kept), "
          "re-install completed with identical output")


# --------------------------------------------------------------------------- #
# C11 / C12 / C13: the pure pieces                                             #
# --------------------------------------------------------------------------- #


def _prepared_engine(model: _DummyTransformer, bs: int = 3) -> PrefetchEngine:
    blocks = list(model.transformer_blocks)
    for block in blocks:
        block.to("cpu")
    engine = PrefetchEngine(
        blocks, _DEVICE, bs, PinnedStagingPool(), torch.cuda.Stream(device=_DEVICE)
    )
    with torch.inference_mode():
        engine.prepare()
    return engine


def check_c11_layout() -> None:
    model = _build_model(n_blocks=3)
    engine = _prepared_engine(model)
    try:
        n_zero_dim = 0
        for lay in engine._layout:
            prev_end = 0
            for (mod, name, _is_param), kind, off, nbytes in zip(
                lay.slots, lay.kinds, lay.offsets, lay.sizes
            ):
                if off % pf._ALIGN != 0:
                    raise AssertionError(f"{name}: offset {off} is not {pf._ALIGN}B aligned")
                if off < prev_end:
                    raise AssertionError(f"{name}: region [{off},{off + nbytes}) overlaps the previous one")
                prev_end = off + nbytes
                if kind[0] == "plain":
                    elem = torch.empty(0, dtype=kind[1]).element_size()
                    if nbytes % elem:
                        raise AssertionError(f"{name}: {nbytes} bytes is not a multiple of {elem}")
                    if off % elem:
                        raise AssertionError(f"{name}: view(dtype) offset rule violated")
                    if len(kind[2]) == 0:
                        n_zero_dim += 1
            if lay.total_bytes < sum(lay.sizes):
                raise AssertionError("the arena is smaller than the sum of its tensors")
            if lay.total_bytes % pf._ALIGN:
                raise AssertionError(f"arena size {lay.total_bytes} is not {pf._ALIGN}B aligned")
        if n_zero_dim != 3:
            raise AssertionError(f"expected one 0-dim buffer per block, found {n_zero_dim}")
        # And the reconstruction those offsets imply actually works.
        with torch.inference_mode():
            arena = torch.empty(engine._layout[0].total_bytes, dtype=torch.uint8, device=_DEVICE)
            views = engine._build_views(0, arena)
        if len(views) != len(engine._layout[0].slots):
            raise AssertionError("view count does not match the slot count")
        print(f"       {len(engine._layout[0].slots)} slots/block, "
              f"arena {engine._layout[0].total_bytes} B, 0-dim + odd-size handled")
    finally:
        engine.teardown()


def check_c12_slot_enumeration() -> None:
    class _Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.child = nn.Linear(4, 4, bias=True)
            self.child.register_buffer("_ic_lora_A_0", torch.zeros(2, 4), persistent=False)
            self.own_param = nn.Parameter(torch.zeros(3))
            self.register_buffer("own_buf", torch.zeros(()))
            self.register_buffer("empty_slot", None)

    toy = _Toy()
    got = {(name, is_param, mod is toy) for mod, name, is_param, _t in pf._enumerate_slots(toy)}
    want = {
        ("own_param", True, True),
        ("own_buf", False, True),
        ("weight", True, False),
        ("bias", True, False),
        ("_ic_lora_A_0", False, False),
    }
    if got != want:
        raise AssertionError(f"enumerated {sorted(got)}, expected {sorted(want)}")
    print("       nested params, non-persistent buffers, 0-dim and module-level tensors all found; "
          "None buffer skipped")


def check_c13_window() -> None:
    total, bs = 48, 8
    cases = {
        0: list(range(0, 8)),
        1: list(range(1, 9)),
        total - bs: list(range(40, 48)),
        total - 1: [47],
        total - 3: [45, 46, 47],
    }
    for idx, want in cases.items():
        got = list(_window(idx, bs, total))
        if got != want:
            raise AssertionError(f"_window({idx},{bs},{total}) = {got}, expected {want}")
    if list(_window(0, 1, 48)) != [0]:
        raise AssertionError("blocks_on_gpu=1 must yield a single-block window")
    print("       head, steady state, tail and blocks_on_gpu=1 all match the synchronous window")


def check_c14_weight_tying_warning() -> None:
    torch.manual_seed(3)
    first = _DummyBlock(256)
    second = _DummyBlock(256, tie_with=first)
    model = _DummyTransformer([first, second])
    with _LogCapture() as logs:
        engine = _prepared_engine(model, bs=1)
        engine.teardown()
    tied = [m for m in logs.messages if "tied tensor" in m]
    if len(tied) != 1:
        raise AssertionError(f"expected exactly 1 tying warning, got {len(tied)}: {tied}")
    if "1 tied tensor(s)" not in tied[0]:
        raise AssertionError(f"the warning did not count the tie: {tied[0]}")

    clean = _build_model(n_blocks=2)
    with _LogCapture() as logs2:
        engine2 = _prepared_engine(clean, bs=1)
        engine2.teardown()
    if [m for m in logs2.messages if "tied tensor" in m]:
        raise AssertionError("an untied model produced a tying warning")
    print("       tying detected once, and not reported for an untied model")


# --------------------------------------------------------------------------- #
# C15: the quantised subclass survives the arena round trip                    #
# --------------------------------------------------------------------------- #


def check_c15_ggml_subclass_preserved() -> None:
    """A GGMLQuantizedTensor SUBCLASS must come back out of the arena as itself.

    engine25 does not use the base class: `Ltx25GGMLTensor` overrides
    `__torch_function__` so the quant metadata survives `Disposable.dispose()`'s
    `torch.empty_like(..., device="meta")`. The non-cyclic window leaves the last
    block of every pass resident, so a GPU view rebuilt as the BASE class would
    still be attached to the model when `dispose()` runs and would resurrect
    exactly the F1 AttributeError the subclass exists to prevent. The subclass is
    declared here rather than imported so this check stays a property of the
    shared module and does not drag engine25 into the 2.3 venv.
    """

    class _SubGGML(GGMLQuantizedTensor):
        """Stand-in for engine25's Ltx25GGMLTensor: same three-argument __new__."""

        marker = "c15"

    torch.manual_seed(5)
    block = _DummyBlock(256)
    base_weight = block.lin._buffers["weight"]
    block.lin._buffers["weight"] = _SubGGML(
        base_weight.as_subclass(torch.Tensor).reshape(-1).clone(),
        base_weight._ggml_type,
        tuple(base_weight._float_shape),
    )
    model = _DummyTransformer([block, _DummyBlock(256)])

    engine = _prepared_engine(model, bs=1)
    try:
        kinds = engine._layout[0].kinds
        slots = engine._layout[0].slots
        # BY SLOT INDEX, not by name: ``norm`` has a ``weight`` too, and looking
        # the quantised one up by name would silently grade the LayerNorm.
        idx = next(
            i for i, (mod, name, _p) in enumerate(slots)
            if name == "weight" and mod is block.lin
        )
        if kinds[idx][0] != "ggml":
            raise AssertionError(f"the weight slot was planned as {kinds[idx][0]!r}, not 'ggml'")
        if len(kinds[idx]) != 4 or kinds[idx][3] is not _SubGGML:
            raise AssertionError(f"the plan did not record the concrete class: {kinds[idx]!r}")

        with torch.inference_mode():
            arena = torch.empty(
                engine._layout[0].total_bytes, dtype=torch.uint8, device=_DEVICE
            )
            views = engine._build_views(0, arena)
        rebuilt = views[idx][3]
        if type(rebuilt) is not _SubGGML:
            raise AssertionError(
                f"rebuilt weight is {type(rebuilt).__name__}, expected _SubGGML"
            )
        if rebuilt._ggml_type != base_weight._ggml_type:
            raise AssertionError(f"_ggml_type={rebuilt._ggml_type} after the round trip")
        if tuple(rebuilt.shape) != (256, 256):
            raise AssertionError(f".shape={tuple(rebuilt.shape)} — the float-shape masquerade is gone")
        # And the plain slots of the same block are untouched by the asymmetry.
        for (_m, name, _p, t), kind in zip(views, kinds):
            if kind[0] == "plain" and (t.dtype != kind[1] or tuple(t.shape) != kind[2]):
                raise AssertionError(f"plain slot {name} came back as {t.dtype}{tuple(t.shape)}")
        del views, rebuilt, arena
        print("       subclass identity, _ggml_type and the float-shape masquerade all survive")
    finally:
        engine.teardown()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    if not torch.cuda.is_available():
        print("[FAIL] no CUDA device: block-swap prefetching cannot be exercised")
        return 1

    torch.manual_seed(0)
    checks: list[tuple[str, Callable[[], None]]] = [
        ("C1  ON output is bit-identical to OFF (3 passes, 12 blocks)", check_c1_output_parity),
        ("C2  the CPU masters are never mutated", check_c2_master_invariance),
        ("C3/C4  residency <= blocks_on_gpu+2, 1 at pass end, no late issues",
         check_c3_c4_residency_and_schedule),
        ("C5  GGMLQuantizedTensor metadata and bytes survive the transfer", check_c5_ggml_metadata),
        ("C6  pinned-allocation failure degrades to the synchronous path", check_c6_pinned_failure_fallback),
        ("C7  blocks_on_gpu >= total / == 0 allocate nothing", check_c7_no_swap_needed),
        ("C8  three install/teardown cycles leak no VRAM or pinned memory", check_c8_no_leak_across_jobs),
        ("C9a  20 heavy passes stay bit-identical", check_c9a_stress_bit_identical),
        ("C9b  S1b (xfer waits for the arena allocation) is a real edge", check_c9b_alloc_event_dependency),
        ("C10  an exception mid-pass leaves nothing behind and re-installs cleanly",
         check_c10_teardown_after_exception),
        ("C11  arena layout: alignment, no overlap, view(dtype) legality, 0-dim", check_c11_layout),
        ("C12  slot enumeration covers nested / non-persistent / module-level", check_c12_slot_enumeration),
        ("C13  window boundaries match the synchronous path", check_c13_window),
        ("C14  weight tying is detected and warned about once", check_c14_weight_tying_warning),
        ("C15  a GGMLQuantizedTensor subclass is rebuilt as that subclass",
         check_c15_ggml_subclass_preserved),
    ]

    for name, fn in checks:
        _run_check(name, fn)

    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_total = len(_RESULTS)
    print(f"\n{n_pass}/{n_total} checks passed")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
