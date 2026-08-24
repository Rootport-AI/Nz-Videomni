"""Async block-swap prefetching — hides the CPU->GPU weight transfer behind compute.

The synchronous swap in ``block_swap_service._patch_block`` costs ~2.9s per
forward pass on the production model (47 blocks x ~340MB, pageable H2D at
14.0GB/s plus a D2H eviction copy at 9.1GB/s). This module removes both halves:

  * The **D2H eviction is deleted outright.** Weights are immutable during
    inference, so the CPU copy taken at install() stays authoritative and the
    GPU side is simply dropped (``_release`` re-points the module at the CPU
    master; no transfer at all).
  * The **H2D is issued ahead of time** on a dedicated CUDA stream through
    pinned staging memory (23.4GB/s), so by the time a block's forward runs its
    weights have already landed. The compute stream only waits on an event.

Everything a block owns (GGUF-compressed ``GGMLQuantizedTensor`` buffers, Linear
biases, norm weights, block-level ``scale_shift_table`` parameters, IC-LoRA
``persistent=False`` buffers) is packed into ONE contiguous uint8 arena per
block, so a block is exactly one host memcpy plus one ``cudaMemcpyAsync``.

Stream discipline — the whole design is four synchronisation points:

    S1   ``slot_event[k].synchronize()``   before overwriting a pinned slot, so
         the host memcpy cannot outrun the DMA still reading that slot (WAR
         across host/device).
    S1b  ``xfer.wait_event(alloc_evt)``    before the copy is enqueued, so the
         transfer stream never writes into an arena whose allocation (on the
         COMPUTE stream) has not completed (WAR across streams).
    S2   ``compute.wait_event(copy_evt)``  before the block's forward runs (RAW).
    S4   ``xfer.synchronize()``            in ``teardown()``, so no in-flight
         transfer survives the job that issued it.

``record_stream`` is deliberately NOT used. The arena is allocated on the
*compute* stream (``torch.empty`` outside the ``with torch.cuda.stream(...)``
block), which makes compute its owning stream: freeing it returns the block to
the compute pool, where the next allocation is safe by stream ordering. The
rejected alternative — allocating on the transfer stream and calling
``record_stream`` — is correct for the read hazard but strands ~3GB in the
transfer stream's allocator pool (``BlockComparator`` keys on stream), which
would be reserved-but-unusable right when the spatial upsampler needs it.

The resident window is NOT cyclic: it mirrors the synchronous path's
``W(idx) = {j | idx <= j < min(idx + blocks_on_gpu, total)}`` exactly, so it
drains to a single block at the end of every pass and the VRAM profile at the
stage1->stage2 boundary is unchanged. Steady-state residency is at most
``blocks_on_gpu + 2`` (the window, plus the block being released, plus the last
block of the previous pass which nothing ever releases). If cyclic prefetch is
ever tried (``_CYCLIC``), a ``release_all()`` hook at the end of denoise becomes
mandatory — no such hook exists today.

This module is only ever reached when a job explicitly asks for prefetching;
with the feature off, ``block_swap_service`` does not even import it.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator

import torch
import torch.nn as nn

from engine.gguf.quant_service import GGMLQuantizedTensor

logger = logging.getLogger(__name__)

# Arena offsets are 512B-aligned: that is the CUDA caching allocator's own
# guarantee, and it makes every view(dtype) below legal unconditionally (the
# storage_offset divisibility rule can never bite for element sizes <= 512).
_ALIGN = 512

# Two pinned staging slots is enough to keep the copy engine busy: a slot is
# reused only after its previous H2D completed, and at steady state the issue
# point runs blocks_on_gpu-1 blocks (~3s of compute) ahead of consumption.
_NUM_STAGING_SLOTS = 2

# New transfers started per forward step. Caps the cold-start burst (the first
# block of a pass would otherwise issue the whole window synchronously on the
# host) while still filling an 8-block window within four steps.
_MAX_ISSUES_PER_STEP = 2

# Validate the view reconstruction for every block instead of just the first.
# All 48 blocks share one class and one layout routine, so block 0 is
# representative; the full sweep allocates/frees ~16GB for no new information.
_VALIDATE_ALL = False

# Documented switch, not a supported mode — see the module docstring.
_CYCLIC = False

# Mutation switch for the selfcheck's negative case ONLY: skipping S1b must be
# observable (the transfer must then complete while the compute stream is still
# busy), which is what proves the wait_event edge is real and not decorative.
_SKIP_ALLOC_EVENT_WAIT = False


def _alloc_pinned(nbytes: int) -> torch.Tensor:
    """Single allocation site for pinned staging memory.

    Factored out so the selfcheck can make cudaHostAlloc fail on demand (C6):
    a machine that cannot pin 2x the largest block must degrade to the
    synchronous path, not fail the job.
    """
    return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)


def _window(idx: int, blocks_on_gpu: int, total: int) -> range:
    """Blocks that must be resident while block ``idx`` runs.

    Identical to the synchronous path's window (``block_swap_service`` :168-170)
    — same start, same clamp, no wrap-around.
    """
    return range(idx, min(idx + blocks_on_gpu, total))


class _BlockLayout:
    """Where every tensor of one block lives inside that block's arena."""

    __slots__ = ("slots", "kinds", "offsets", "sizes", "total_bytes")

    def __init__(
        self,
        slots: list[tuple[nn.Module, str, bool]],
        kinds: list[tuple],
        offsets: list[int],
        sizes: list[int],
        total_bytes: int,
    ) -> None:
        self.slots = slots          # (owning module, attribute name, is_parameter)
        # ("ggml", ggml_type, float_shape, tensor_class) | ("plain", dtype, shape)
        # The ggml entry carries the CONCRETE class of the CPU master as a 4th
        # element so `_build_views` can rebuild the GPU view as that same class
        # rather than as the base `GGMLQuantizedTensor`. engine25 subclasses it
        # (`Ltx25GGMLTensor`) purely to keep the quant metadata alive across
        # `dispose()`'s `empty_like`; a base-class view left resident at the end
        # of a pass would bring that bug straight back. The asymmetry (ggml 4
        # elements, plain 3) is safe: all three consumers of `kinds` branch on
        # `kind[0]` before reading anything past index 2.
        self.kinds = kinds
        self.offsets = offsets      # byte offset into the arena (512B aligned)
        self.sizes = sizes          # exact byte count of each tensor
        self.total_bytes = total_bytes


class _BlockState:
    """One block's in-flight/resident GPU residency record."""

    __slots__ = ("arena", "views", "event", "slot", "acquired")

    def __init__(
        self,
        arena: torch.Tensor,
        views: list[tuple[nn.Module, str, bool, torch.Tensor]],
        event: "torch.cuda.Event",
        slot: int,
    ) -> None:
        self.arena = arena          # 1D uint8 CUDA buffer, OWNED BY THE COMPUTE STREAM
        self.views = views          # device tensors carved out of the arena
        self.event = event          # H2D completion, recorded on the transfer stream
        self.slot = slot            # pinned staging slot this block was copied through
        self.acquired = False       # has the compute stream waited on `event` yet?


class PinnedStagingPool:
    """Grow-only pinned staging buffers, owned by the resident BlockSwapService.

    Deliberately NOT per-job: cudaHostAlloc of ~370MB takes ~100ms and gets less
    reliable as the process fragments host memory, so re-allocating every job is
    how "prefetch mysteriously stopped working at job 7" happens. The cost is
    ~740MB of RAM held for the process lifetime (documented in the README).
    """

    def __init__(self, num_slots: int = _NUM_STAGING_SLOTS) -> None:
        self.num_slots = num_slots
        self._buffers: list[torch.Tensor] = []
        self._nbytes = 0

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def buffers(self) -> list[torch.Tensor]:
        return self._buffers

    def ensure(self, nbytes: int) -> list[torch.Tensor]:
        """Return ``num_slots`` pinned buffers of at least ``nbytes`` each.

        Raises whatever the allocator raises — the caller (install) turns that
        into a fallback onto the synchronous path.
        """
        if self._buffers and self._nbytes >= nbytes:
            return self._buffers
        # Drop the old buffers BEFORE allocating the bigger ones: holding both
        # would double the pinned high-water mark for no reason. Safe because a
        # grow only ever happens inside install(), after the previous job's
        # teardown already synchronised the transfer stream.
        self._buffers = []
        self._nbytes = 0
        buffers = [_alloc_pinned(nbytes) for _ in range(self.num_slots)]
        self._buffers = buffers
        self._nbytes = nbytes
        logger.info(
            "BlockSwap prefetch: pinned staging pool = %d x %.1f MB",
            self.num_slots, nbytes / (1024 * 1024),
        )
        return buffers

    def release(self) -> None:
        self._buffers = []
        self._nbytes = 0


def _enumerate_slots(block: nn.Module) -> Iterator[tuple[nn.Module, str, bool, torch.Tensor]]:
    """Every tensor the block owns, with the module and attribute that own it.

    Per-module ``_parameters``/``_buffers`` walking rather than
    ``named_parameters()``: the latter de-duplicates shared tensors by default
    (a slot we failed to restore would silently keep pointing at freed GPU
    memory) and does not hand back the owning module, which restoration needs.
    Same traversal as ``dit_cpu_load_service._move_non_block_tensors_to_gpu``.
    ``block.modules()`` includes the block itself, so module-level parameters
    like ``scale_shift_table`` are picked up too.
    """
    for mod in block.modules():
        for name, p in list(mod._parameters.items()):
            if p is not None:
                yield mod, name, True, p
        for name, b in list(mod._buffers.items()):
            if b is not None and isinstance(b, torch.Tensor):
                yield mod, name, False, b


class PrefetchEngine:
    """Per-job prefetching state for one transformer's block list.

    Created by ``BlockSwapService.install()`` when the job asked for it, thrown
    away by ``BlockSwapService.teardown_prefetch()`` in the job's ``finally``.
    The pinned pool and the transfer stream outlive it (they belong to the
    service); everything else — CPU masters, arenas, events — dies with the job
    so nothing keeps the previous transformer alive.
    """

    def __init__(
        self,
        blocks: list[nn.Module],
        device: torch.device,
        blocks_on_gpu: int,
        pinned_pool: PinnedStagingPool,
        xfer_stream: "torch.cuda.Stream",
    ) -> None:
        self._blocks = blocks
        self.device = device
        self.blocks_on_gpu = blocks_on_gpu
        self._pool = pinned_pool
        self._xfer = xfer_stream

        self._layout: list[_BlockLayout] = []
        # CPU masters: the authoritative weights for the whole job. `_master`
        # holds the tensors handed back to the modules on release; `_master_u8`
        # holds the same storage viewed as flat uint8 (cached so the per-step
        # host memcpy is dtype-agnostic and allocation-free).
        self._master: list[list[torch.Tensor]] = []
        self._master_u8: list[list[torch.Tensor]] = []

        self._state: dict[int, _BlockState] = {}
        self._pinned: list[torch.Tensor] = []
        self._slot_event: list[Any] = [None] * _NUM_STAGING_SLOTS
        self._next_slot = 0
        self._stats = {"issued": 0, "misses": [], "slot_wait_ms": 0.0, "pass": 0}

    # ------------------------------------------------------------------ #
    # Install-time setup                                                   #
    # ------------------------------------------------------------------ #

    def prepare(self) -> None:
        """Snapshot the CPU masters, plan the arenas, secure pinned memory.

        Runs once per install (i.e. once per job), with every block already on
        CPU. Any failure here propagates to install(), which falls back to the
        synchronous path for the whole job.
        """
        if self.device.type != "cuda":
            raise RuntimeError(f"prefetch requires a CUDA device, got {self.device}")

        seen_ids: dict[int, int] = {}
        n_shared = 0
        n_contiguified = 0
        for block in self._blocks:
            layout, master, master_u8, shared, contiguified = self._plan(block, seen_ids)
            self._layout.append(layout)
            self._master.append(master)
            self._master_u8.append(master_u8)
            n_shared += shared
            n_contiguified += contiguified

        if n_shared:
            # Weight tying: both slots get their own arena region, so the output
            # is unaffected and only VRAM is wasted. LTX blocks have none, so
            # this firing at all is worth knowing about.
            logger.warning(
                "BlockSwap prefetch: %d tied tensor(s) detected — each copy gets "
                "its own arena region (correct, but wastes VRAM)", n_shared,
            )
        if n_contiguified:
            logger.info(
                "BlockSwap prefetch: %d non-contiguous tensor(s) were made "
                "contiguous once at install and are now the CPU master",
                n_contiguified,
            )

        max_bytes = max((lay.total_bytes for lay in self._layout), default=0)
        self._pinned = self._pool.ensure(max_bytes)
        self._validate()

        total_mb = sum(lay.total_bytes for lay in self._layout) / (1024 * 1024)
        logger.info(
            "BlockSwap prefetch ready: %d blocks, %.0f MB of CPU masters, "
            "largest block %.1f MB, window %d",
            len(self._blocks), total_mb, max_bytes / (1024 * 1024), self.blocks_on_gpu,
        )

    def _plan(
        self, block: nn.Module, seen_ids: dict[int, int],
    ) -> tuple[_BlockLayout, list[torch.Tensor], list[torch.Tensor], int, int]:
        slots: list[tuple[nn.Module, str, bool]] = []
        kinds: list[tuple] = []
        offsets: list[int] = []
        sizes: list[int] = []
        master: list[torch.Tensor] = []
        master_u8: list[torch.Tensor] = []
        shared = 0
        contiguified = 0
        off = 0

        for mod, name, is_param, tensor in _enumerate_slots(block):
            src = tensor.data if is_param else tensor
            if src.device.type != "cpu":
                raise RuntimeError(
                    f"prepare() requires every block tensor on CPU; {name} is on {src.device}"
                )
            if isinstance(src, GGMLQuantizedTensor):
                # shape/numel/size lie (they report the dequantised float shape,
                # quant_service.py:459-471). The real byte count only exists
                # under the subclass, reached the same way the forward does
                # (quant_service.py:645). reshape(-1) precedes view(uint8) so a
                # 0-dim tensor cannot hit view(dtype)'s hard error.
                raw = src.as_subclass(torch.Tensor).reshape(-1).view(torch.uint8)
                nbytes = raw.numel()
                # `type(src)`, not the base class: see `_BlockLayout.kinds`.
                kind: tuple = ("ggml", src._ggml_type, tuple(src._float_shape), type(src))
                src_u8 = raw
            else:
                if not src.is_contiguous():
                    src = src.contiguous()
                    contiguified += 1
                nbytes = src.numel() * src.element_size()
                kind = ("plain", src.dtype, tuple(src.shape))
                src_u8 = src.reshape(-1).view(torch.uint8)

            # Tying check by storage address. Skipped for empty tensors, whose
            # data_ptr is not a unique identity (it can be 0 for all of them).
            if nbytes:
                key = src_u8.data_ptr()
                if key in seen_ids:
                    shared += 1
                else:
                    seen_ids[key] = 1

            slots.append((mod, name, is_param))
            kinds.append(kind)
            offsets.append(off)
            sizes.append(nbytes)
            master.append(src)
            master_u8.append(src_u8)
            off = (off + nbytes + _ALIGN - 1) // _ALIGN * _ALIGN

        return _BlockLayout(slots, kinds, offsets, sizes, off), master, master_u8, shared, contiguified

    def _validate(self) -> None:
        """Rebuild one block's views over a dummy arena before anything ships.

        A view reconstruction that is subtly wrong (dtype, shape, lost quant
        metadata) would otherwise surface as garbage pixels, not as an error.
        Deliberately no ``empty_cache()`` here: the ON path must not perturb the
        allocator in ways the OFF path does not, or the VRAM gate compares two
        different things.
        """
        targets = self._layout if _VALIDATE_ALL else self._layout[:1]
        for idx, lay in enumerate(targets):
            dummy = torch.empty(lay.total_bytes, dtype=torch.uint8, device=self.device)
            views = self._build_views(idx, dummy)
            for (_mod, _name, _is_param, dev_t), kind in zip(views, lay.kinds):
                if kind[0] == "ggml":
                    if not isinstance(dev_t, GGMLQuantizedTensor):
                        raise RuntimeError("rebuilt tensor lost the GGMLQuantizedTensor identity")
                    if type(dev_t) is not kind[3]:
                        raise RuntimeError(
                            f"rebuilt tensor is {type(dev_t).__name__}, expected "
                            f"{kind[3].__name__} — the subclass did not survive the round trip"
                        )
                    if dev_t._ggml_type != kind[1] or tuple(dev_t._float_shape) != kind[2]:
                        raise RuntimeError("rebuilt tensor lost its quant metadata")
                else:
                    if dev_t.dtype != kind[1] or tuple(dev_t.shape) != kind[2]:
                        raise RuntimeError(
                            f"rebuilt tensor is {dev_t.dtype}{tuple(dev_t.shape)}, "
                            f"expected {kind[1]}{kind[2]}"
                        )
            del views, dummy

    # ------------------------------------------------------------------ #
    # Per-forward entry point                                              #
    # ------------------------------------------------------------------ #

    def on_block_forward(self, idx: int) -> None:
        """Make block ``idx`` safe to read on the compute stream, and look ahead.

        Called at the top of the patched forward. On return every tensor of
        block ``idx`` is a GPU view whose H2D the compute stream has waited on.
        """
        total = len(self._blocks)
        bs = self.blocks_on_gpu

        # (0) One log line per pass. At the pass HEAD rather than after step (1)
        #     as the design sketch had it, so that this pass's own cold-start
        #     miss is attributed to this pass instead of being reset away.
        if idx == 0:
            self._log_and_reset_stats()

        # (1) Not issued yet (cold start, or a window too small to look ahead):
        #     issue it now and eat the transfer synchronously.
        if idx not in self._state:
            self._stats["misses"].append(idx)
            self._issue(idx)

        # (2) Fill the tail of the window BEFORE the wait below, so the copy
        #     engine has work queued while the compute stream blocks.
        issued = 0
        for j in _window(idx, bs, total):
            if j <= idx or j in self._state:
                continue
            self._issue(j)
            issued += 1
            if issued >= _MAX_ISSUES_PER_STEP:
                break

        # (3) S2: the compute stream waits for this block's H2D (RAW).
        st = self._state[idx]
        if not st.acquired:
            torch.cuda.current_stream(self.device).wait_event(st.event)
            st.acquired = True

        # (4) Release the block that just left the window. No D2H: the CPU
        #     master has been authoritative the whole time.
        if idx - 1 >= 0:
            self._release(idx - 1)

    def _issue(self, idx: int) -> None:
        """CPU master -> pinned slot -> (transfer stream) -> GPU arena.

        The host memcpy is synchronous (~14ms for a 370MB block); the device
        copy is not. The arena is allocated on the COMPUTE stream on purpose —
        see the module docstring.
        """
        lay = self._layout[idx]
        slot = self._acquire_slot()                       # S1
        pinned = self._pinned[slot]

        for src_u8, off, nbytes in zip(self._master_u8[idx], lay.offsets, lay.sizes):
            pinned[off:off + nbytes].copy_(src_u8)

        compute = torch.cuda.current_stream(self.device)
        arena = torch.empty(lay.total_bytes, dtype=torch.uint8, device=self.device)
        alloc_evt = torch.cuda.Event()
        alloc_evt.record(compute)

        if not _SKIP_ALLOC_EVENT_WAIT:
            self._xfer.wait_event(alloc_evt)              # S1b
        with torch.cuda.stream(self._xfer):
            arena.copy_(pinned[:lay.total_bytes], non_blocking=True)
        copy_evt = torch.cuda.Event()
        copy_evt.record(self._xfer)

        views = self._build_views(idx, arena)
        # Swapped in BEFORE the data lands. The arena holds garbage right now,
        # but nothing reads a block's weights except its own forward, which is
        # gated by S2 above; NAG/VSF/sage only ever touch forward callables.
        self._install_views(views)
        self._state[idx] = _BlockState(arena, views, copy_evt, slot)
        self._slot_event[slot] = copy_evt
        self._stats["issued"] += 1

    def _acquire_slot(self) -> int:
        """Round-robin a pinned slot, blocking the host until its last H2D read finished."""
        slot = self._next_slot
        self._next_slot = (self._next_slot + 1) % len(self._pinned)
        evt = self._slot_event[slot]
        if evt is not None and not evt.query():
            t0 = time.perf_counter()
            evt.synchronize()
            self._stats["slot_wait_ms"] += (time.perf_counter() - t0) * 1e3
        return slot

    def _build_views(
        self, idx: int, arena: torch.Tensor,
    ) -> list[tuple[nn.Module, str, bool, torch.Tensor]]:
        lay = self._layout[idx]
        out: list[tuple[nn.Module, str, bool, torch.Tensor]] = []
        for (mod, name, is_param), kind, off, nbytes in zip(
            lay.slots, lay.kinds, lay.offsets, lay.sizes
        ):
            raw = arena[off:off + nbytes]                 # 1D uint8, offset is 512-aligned
            if kind[0] == "ggml":
                # Same construction as the GGUF loader (quant_service.py:586) —
                # verified to work under inference_mode (WORKORDER S0 spike).
                # `kind[3]` is the CPU master's own class, so a subclass round
                # trips as itself (three-argument __new__ is the shared shape).
                dev_t: torch.Tensor = kind[3](raw, kind[1], kind[2])
            else:
                dev_t = raw.view(kind[1]).view(kind[2])
            out.append((mod, name, is_param, dev_t))
        return out

    def _install_views(self, views: list[tuple[nn.Module, str, bool, torch.Tensor]]) -> None:
        for mod, name, is_param, dev_t in views:
            if is_param:
                mod._parameters[name].data = dev_t        # keeps the Parameter object's identity
            else:
                mod._buffers[name] = dev_t                # dit_cpu_load_service.py:77's idiom

    def _release(self, idx: int) -> None:
        """Point the module back at its CPU master. Zero transfers."""
        st = self._state.pop(idx, None)
        if st is None:
            return
        for (mod, name, is_param), cpu_t in zip(self._layout[idx].slots, self._master[idx]):
            if is_param:
                mod._parameters[name].data = cpu_t
            else:
                mod._buffers[name] = cpu_t
        # Dropping the last reference returns the arena to the allocator. It is
        # owned by the compute stream, so any kernel still reading it is ordered
        # ahead of whatever allocation reuses the block.
        del st

    # ------------------------------------------------------------------ #
    # Teardown / diagnostics                                               #
    # ------------------------------------------------------------------ #

    def teardown(self) -> None:
        """End-of-job cleanup. Idempotent, and never raises.

        Called from the pipeline's ``finally`` (and defensively at the next
        install), so it also runs after an exception mid-pass: S4 drains the
        transfer stream first, then every arena, CPU master and event reference
        is dropped so the finished transformer can be collected.
        """
        try:
            if self._xfer is not None:
                self._xfer.synchronize()                  # S4
        except Exception as exc:  # noqa: BLE001 — cleanup must not mask the real error
            logger.warning("BlockSwap prefetch: transfer stream sync failed (%s)", exc)
        for idx in list(self._state.keys()):
            try:
                self._release(idx)
            except Exception as exc:  # noqa: BLE001
                logger.warning("BlockSwap prefetch: release of block %d failed (%s)", idx, exc)
        self._state.clear()
        self._slot_event = [None] * _NUM_STAGING_SLOTS
        self._layout = []
        self._master = []
        self._master_u8 = []
        self._pinned = []                                 # the pool keeps the buffers alive
        self._blocks = []

    def _log_and_reset_stats(self) -> None:
        s = self._stats
        if s["pass"] > 0:
            misses = s["misses"]
            # A miss on block 0 is structural (a pass always starts with an
            # empty window, ~46ms); misses anywhere else mean the look-ahead is
            # not keeping up — the expected cause is blocks_on_gpu == 1.
            late = [m for m in misses if m != 0]
            logger.info(
                "BlockSwap prefetch pass %d: %d issued, %d sync miss(es)%s, "
                "%.1f ms waiting on pinned slots",
                s["pass"], s["issued"], len(misses),
                f" (late: {late[:8]})" if late else "", s["slot_wait_ms"],
            )
        s["pass"] += 1
        s["issued"] = 0
        s["misses"] = []
        s["slot_wait_ms"] = 0.0
