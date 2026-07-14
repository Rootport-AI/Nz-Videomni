"""Pure-Python geometry tests for the chunked spatial-upsample layout.

``plan_upsample_chunks`` is the SINGLE SOURCE OF TRUTH for how the optional
``chunked_upsample`` path splits the assembled stage-1 latent into halo-padded
temporal chunks so a long 768p chain can be upsampled without materialising the
whole timeline on the GPU. These tests pin the geometry — no torch, no GPU,
matching the ``test_chain_math_a2v.py`` style — so the engine's chunk loop and
the one-shot pass stay byte-compatible on the convolution interior.
"""

from __future__ import annotations

import chain_math
from chain_math import (
    UPSAMPLE_CHUNK_FRAMES,
    UPSAMPLE_HALO_FRAMES,
    plan_upsample_chunks,
)

# f_total values spanning: degenerate (<= chunk), exactly one chunk, one-over,
# a couple of whole chunks, and long realistic 768p chain lengths.
_F_TOTALS = [1, 31, 32, 33, 96, 467, 1395]


def test_keep_ranges_cover_timeline_without_gap_or_overlap():
    """(a) Every chunk's keep range [out_start, out_start+keep_len) must tile
    [0, f_total) with no overlap and no gap, for a wide span of f_total."""
    for f_total in _F_TOTALS:
        plan = plan_upsample_chunks(f_total)
        cursor = 0
        for ch in plan:
            assert ch.out_start == cursor, (f_total, ch)
            assert ch.keep_len >= 1, (f_total, ch)
            cursor += ch.keep_len
        assert cursor == f_total, (f_total, cursor)


def test_interior_chunks_keep_lo_equals_halo():
    """(b) Any chunk whose core does not touch either physical end keeps the halo
    on its low side, so keep_lo == UPSAMPLE_HALO_FRAMES."""
    for f_total in _F_TOTALS:
        plan = plan_upsample_chunks(f_total)
        for k, ch in enumerate(plan):
            if k > 0:  # not the head chunk -> low-side halo is present in full
                assert ch.keep_lo == UPSAMPLE_HALO_FRAMES, (f_total, k, ch)


def test_head_keep_lo_zero_and_tail_reaches_f_total():
    """(c) The first chunk has keep_lo == 0 (no low-side halo to drop) and the
    last chunk reads right up to the physical end (in_start+in_len == f_total)."""
    for f_total in _F_TOTALS:
        plan = plan_upsample_chunks(f_total)
        assert plan[0].keep_lo == 0, (f_total, plan[0])
        last = plan[-1]
        assert last.in_start + last.in_len == f_total, (f_total, last)


def test_degenerate_single_chunk_is_whole_input():
    """(d) When f_total <= chunk the plan is a single chunk spanning the entire
    input: in_start == 0, in_len == f_total, keep covers the whole timeline."""
    for f_total in [1, 16, 31, UPSAMPLE_CHUNK_FRAMES]:
        plan = plan_upsample_chunks(f_total)
        assert len(plan) == 1, (f_total, plan)
        ch = plan[0]
        assert ch.in_start == 0, (f_total, ch)
        assert ch.in_len == f_total, (f_total, ch)
        assert ch.keep_lo == 0, (f_total, ch)
        assert ch.keep_len == f_total, (f_total, ch)
        assert ch.out_start == 0, (f_total, ch)


def test_ragged_last_core_is_shorter():
    """(e) A non-multiple f_total gives a shorter final core (keep_len < chunk),
    while all preceding cores are exactly ``chunk`` long."""
    for f_total in [33, 96, 467, 1395]:
        plan = plan_upsample_chunks(f_total)
        rem = f_total % UPSAMPLE_CHUNK_FRAMES
        if rem == 0:
            continue  # 96 divides evenly -> covered by the divisor branch below
        for ch in plan[:-1]:
            assert ch.keep_len == UPSAMPLE_CHUNK_FRAMES, (f_total, ch)
        assert plan[-1].keep_len == rem, (f_total, plan[-1])


def test_divisor_f_total_has_uniform_cores():
    """(e') When f_total is a multiple of chunk, every core is exactly chunk."""
    for f_total in [32, 96]:
        plan = plan_upsample_chunks(f_total)
        for ch in plan:
            assert ch.keep_len == UPSAMPLE_CHUNK_FRAMES, (f_total, ch)


def test_in_len_bounded_by_chunk_plus_two_halos():
    """(f) No chunk ever reads more than chunk + 2*halo frames (the GPU working
    set per chunk is bounded independent of f_total)."""
    bound = UPSAMPLE_CHUNK_FRAMES + 2 * UPSAMPLE_HALO_FRAMES
    for f_total in _F_TOTALS:
        for ch in plan_upsample_chunks(f_total):
            assert ch.in_len <= bound, (f_total, ch)


def test_f_total_non_positive_raises():
    """f_total <= 0 is geometrically impossible -> ValueError (matches the
    chain_math validation style: guard + raise)."""
    for bad in [0, -1, -32]:
        try:
            plan_upsample_chunks(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for f_total={bad}")


def test_default_chunk_and_halo_constants():
    """The recipe constants are frozen from the GPU experiment (chunk 32, halo
    18); a regression here would silently change every plan."""
    assert UPSAMPLE_CHUNK_FRAMES == 32
    assert UPSAMPLE_HALO_FRAMES == 18
    assert chain_math.UPSAMPLE_CHUNK_FRAMES == 32
    assert chain_math.UPSAMPLE_HALO_FRAMES == 18
