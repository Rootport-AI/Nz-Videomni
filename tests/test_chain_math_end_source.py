"""End source (tail freeze) geometry — the pure ``chain_math`` layer.

Pins the numbers the app validator, the engine and the mock ALL resolve from
``compute_chain_layout(end_context_px=...)``. An end source appends an INTERNAL
band segment after the user's clips and hard-freezes its ``end_context_px``
pixel frames onto uploaded material, so if any expectation here moves the freeze
band lands on the wrong latents — a silent quality failure, not a crash, exactly
like retake (see tests/test_retake_math.py and VERIFICATION_LOG §55).

The three pillars this file exists to defend:

  1. OUTPUT = CLIPS + BAND. The band is its own segment, so the user's clip
     lengths keep meaning "new material" and the delivered length grows by
     exactly the band (``total_px == clips_total_px + end_context_px``).
  2. THE BAND MAY STRADDLE STAGE-2 TILES. ``end_tile_bands`` is the single
     source of truth for what each tile freezes, clamps included.
  3. NOTHING IS REJECTED FOR "not fitting" ANY MORE. The v1 rejections (final
     clip too short / final tile too short / per-window ceiling) are gone; the
     one NEW rejection is ``kv >= 2``.
"""

from __future__ import annotations

import itertools

import pytest

import chain_math as cm

FPS = 24.0

# The owner-approved default: 72 pixel frames == 3.0s at 24fps.
DEFAULT_END_PX = 72

# The BOX-STOCK chain: two 49-frame clips, the webui default. Under v1 this
# combination was REJECTED with an end source ("the final clip must be at least
# 97 pixel frames"); with the internal segment it is the ordinary case.
DEFAULT_CHAIN = [49, 49]

# A chain that needs several stage-2 tiles.
MULTI_TILE_CHAIN = [161, 169]

# Frame rates the app can generate at (the webui's whole list).
ALL_FPS = (23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0, 59.94, 60.0)

# Representative clip layouts: one clip, the default pair, long pairs, a ragged
# triple, the 8-clip and 4-clip shapes long chains use.
CLIP_SETS = (
    [49],
    [49, 49],
    [169],
    [105, 113],
    [257, 257],
    [121, 121, 121],
    [73, 89, 97],
    [481, 481],
    [49] * 8,
    [257] * 4,
)

BOTH_WINDOWS = ("standard", "high_resolution")

# The stage-2 audio tiling has a PRE-EXISTING quirk at the fractional and 50fps
# rates (it rejects some clip/window combinations with or without an end
# source). Those messages are tolerated by the sweeps below; a "degenerate audio
# overlap" is NOT — that is the one an end source could newly provoke, and the
# kv >= 2 rule exists to keep it unreachable.
KNOWN_AUDIO_TILING_MESSAGES = (
    "audio reassembly",
    "audio tile",
    "audio overlap mismatch",
)


def _clips_only_total_px(clips: list[int], fps: float, kv: int) -> int:
    """What the clips alone assemble to — the independent half of pillar 1."""
    return cm.compute_chain_layout(clips, fps, kv=kv).total_px


# ── the deleted ceiling helper ───────────────────────────────────────────────
def test_stage2_max_end_context_px_is_gone():
    # It only ever existed to express "the band must fit in the last tile", a
    # rule the straddling freeze deleted. Anything still importing it must fail
    # loudly rather than silently keep an obsolete ceiling alive.
    assert not hasattr(cm, "stage2_max_end_context_px")
    # Its HEAD-side sibling stays: a V2V context really is bound by tile 0.
    assert cm.stage2_max_context_px(22) == 161


# ── additive contract ────────────────────────────────────────────────────────
def test_no_end_source_key_when_end_context_is_none():
    # A chain without an end source must be byte-unchanged: no extra segment,
    # no freeze plan, no sub-dict.
    plain = cm.compute_chain_layout(DEFAULT_CHAIN, FPS)
    assert "end_source" not in plain.to_dict()
    assert plain.end_context_px is None
    assert plain.n_end_v == 0
    assert plain.end_source_junction_px is None
    assert plain.seg_frames == DEFAULT_CHAIN        # segments == clips exactly
    assert plain.seg_latent == [7, 7]
    assert plain.end_segment_px == 0
    assert plain.end_segment_latent == 0
    assert plain.end_tile_bands == []


def test_top_level_keys_are_frozen_and_only_end_source_is_added():
    with_end = cm.compute_chain_layout(DEFAULT_CHAIN, FPS, end_context_px=DEFAULT_END_PX)
    without = cm.compute_chain_layout(DEFAULT_CHAIN, FPS)
    a, b = with_end.to_dict(), without.to_dict()
    assert set(a) - set(b) == {"end_source"}
    assert set(b) == {
        "clip_frames", "fps", "kv", "v_tile", "v_adv", "kt_v",
        "seg_latent", "seg_audio", "f_total_latent", "total_px", "a_total",
        "ka_list", "video_tiles", "audio_tiles", "audio_adv", "kt_a", "n_tiles",
        "segment_seam_junctions", "tile_seam_junctions", "all_junctions",
        "duration_sec",
    }
    # ``seg_frames`` is deliberately NOT a new top-level key: seg_latent /
    # seg_audio already report the segment view, and the metadata contract is
    # frozen. The clip echo is untouched by the internal segment.
    assert with_end.clip_frames == DEFAULT_CHAIN
    assert a["clip_frames"] == DEFAULT_CHAIN
    # The end source DOES change the rest of the layout now (that is the point):
    # the timeline is longer, so the tiles and junctions move.
    assert a["total_px"] == b["total_px"] + DEFAULT_END_PX


def test_end_source_sub_dict_shape():
    layout = cm.compute_chain_layout(DEFAULT_CHAIN, FPS, end_context_px=DEFAULT_END_PX)
    assert layout.to_dict()["end_source"] == {
        "end_context_px": 72,
        "n_end_v": 9,                    # 72 // 8 (tail grid)
        "end_source_junction_px": 80,    # 153 - 72 - 1
        "cut_frames": 73,                # the causal VAE's +1 primer
        "end_segment_px": 89,            # px_from_v_latent(3 + 9)
        "end_segment_latent": 12,        # kv + n_end_v
        "end_tile_bands": [[9, 9]],      # one tile, band at its tail
        "clips_total_px": 81,            # what [49, 49] alone assembles to
    }


# ── pillar 1: output length identity ─────────────────────────────────────────
@pytest.mark.parametrize("clips", CLIP_SETS)
@pytest.mark.parametrize("end_px", [8, 24, 72, 136])
def test_output_length_is_clips_plus_band(clips, end_px):
    for fps, kv, window in itertools.product(ALL_FPS, (2, 3, 5), BOTH_WINDOWS):
        if any(kv >= cm.v_latent_frames(c) for c in clips):
            continue
        v_tile, v_adv = cm.resolve_stage2_window(window)
        try:
            layout = cm.compute_chain_layout(
                clips, fps, kv=kv, v_tile=v_tile, v_adv=v_adv,
                end_context_px=end_px,
            )
        except ValueError as exc:          # pre-existing audio-tiling quirk
            assert any(m in str(exc) for m in KNOWN_AUDIO_TILING_MESSAGES), exc
            continue
        try:
            clips_only = cm.compute_chain_layout(
                clips, fps, kv=kv, v_tile=v_tile, v_adv=v_adv
            )
        except ValueError as exc:  # same quirk, on the shorter timeline
            assert any(m in str(exc) for m in KNOWN_AUDIO_TILING_MESSAGES), exc
            continue
        assert layout.total_px == clips_only.total_px + end_px
        assert layout.f_total == clips_only.f_total + end_px // 8
        # ...and the number published to clients agrees with the independent
        # clips-only layout, so the app's "of which N frames are the source"
        # breakdown can never drift from the geometry.
        assert (
            layout.to_dict()["end_source"]["clips_total_px"] == clips_only.total_px
        )


def test_default_chain_pins_every_derived_number():
    layout = cm.compute_chain_layout(DEFAULT_CHAIN, FPS, end_context_px=DEFAULT_END_PX)
    assert layout.seg_frames == [49, 49, 89]
    assert layout.seg_latent == [7, 7, 12]
    assert layout.f_total == 20                      # 11 (clips) + 9 (band)
    assert layout.total_px == 153                    # 81 + 72
    assert layout.n_end_v == 9
    assert layout.end_source_junction_px == 80
    assert layout.duration_sec == round(153 / 24.0, 3)


def test_headline_default_chain_is_the_gate_h1_number():
    # H1 in the real-hardware gate: the box-stock [257, 257] chain plus a
    # 72-frame band is 569 delivered frames, of which the last 72 are the
    # material. This is the number the app shows in its breakdown.
    layout = cm.compute_chain_layout([257, 257], FPS, end_context_px=72)
    assert layout.total_px == 569
    assert layout.to_dict()["end_source"]["clips_total_px"] == 497
    assert 497 + 72 == 569


# ── pillar 1: the internal segment's shape ───────────────────────────────────
@pytest.mark.parametrize("kv", [2, 3, 4, 8])
@pytest.mark.parametrize("end_px", [8, 72, 136])
def test_internal_segment_is_carry_plus_band(kv, end_px):
    clips = [169, 177]
    layout = cm.compute_chain_layout(clips, FPS, kv=kv, end_context_px=end_px)
    n_end_v = end_px // 8
    # One extra segment, appended AFTER the clips, which are echoed untouched.
    assert layout.clip_frames == clips
    assert layout.seg_frames[: len(clips)] == clips
    assert len(layout.seg_frames) == len(clips) + 1
    # Its length: kv latents of carry-over from the last clip + the band.
    assert layout.seg_latent[-1] == kv + n_end_v
    assert layout.end_segment_latent == kv + n_end_v
    assert layout.end_segment_px == cm.px_from_v_latent(kv + n_end_v)
    assert layout.seg_frames[-1] == layout.end_segment_px
    # seg_audio follows seg_frames, so the ka budget covers the extra join too.
    assert layout.seg_audio == [cm.a_frames_for_px(f, FPS) for f in layout.seg_frames]
    assert len(layout.ka_list) == len(layout.seg_frames) - 1


def test_internal_segment_stays_far_below_the_per_clip_ceiling():
    # The reason the design changed: a band can no longer push a stage-1 pass
    # past the 481-frame per-clip ceiling, because it arrives as its own SHORT
    # segment. At the published maximum (kv=3, band 136) the extra segment is 20
    # latent frames == 153 pixel frames.
    layout = cm.compute_chain_layout([481, 481], FPS, end_context_px=136)
    assert layout.end_segment_latent == 20
    assert layout.end_segment_px == 153
    assert max(layout.seg_frames) == 481


# ── pillar 2: end_tile_bands ─────────────────────────────────────────────────
def _naive_bands(layout: cm.ChainLayout) -> list[tuple[int, int]]:
    """Independent re-derivation from the tile geometry alone.

    Deliberately written the long way (intersect each tile with the band
    interval) so it shares no algebra with the implementation.
    """
    band_lo = layout.f_total - layout.n_end_v
    band_hi = layout.f_total
    out: list[tuple[int, int]] = []
    for vs, vlen in layout.v_tiles:
        lo, hi = max(vs, band_lo), min(vs + vlen, band_hi)
        t = max(0, hi - lo)
        out.append((t, vs + vlen - band_lo))
    return out


@pytest.mark.parametrize("clips", CLIP_SETS)
@pytest.mark.parametrize("end_px", [8, 40, 72, 136])
@pytest.mark.parametrize("window", BOTH_WINDOWS)
def test_end_tile_bands_match_a_naive_intersection_and_cover_the_band(
    clips, end_px, window
):
    v_tile, v_adv = cm.resolve_stage2_window(window)
    try:
        layout = cm.compute_chain_layout(
            clips, FPS, v_tile=v_tile, v_adv=v_adv, end_context_px=end_px
        )
    except ValueError as exc:
        assert any(m in str(exc) for m in KNOWN_AUDIO_TILING_MESSAGES), exc
        return
    assert len(layout.end_tile_bands) == layout.n_tiles
    assert layout.end_tile_bands == _naive_bands(layout)

    # Every write stays inside the band, inside its own tile, and the writes
    # together cover the WHOLE band — the property the freeze depends on.
    covered: set[int] = set()
    for (t, off), (vs, vlen) in zip(layout.end_tile_bands, layout.v_tiles):
        assert 0 <= t <= vlen                      # the min(vlen, ...) clamp
        if t == 0:
            continue
        assert 0 <= off - t < off <= layout.n_end_v
        covered |= set(range(off - t, off))
    assert covered == set(range(layout.n_end_v))
    # The last tile always ends on the timeline's end, so it always holds the
    # band's last latent.
    assert layout.end_tile_bands[-1][1] == layout.n_end_v
    assert layout.end_tile_bands[-1][0] > 0


def test_two_tile_straddle_is_pinned():
    # The gate-H2 shape: the default 2x49 chain with the longest published band.
    # f_total 28 -> tiles (0,22) and (18,10); the band's 17 latents start at 11,
    # so tile 0 ends on 11 of them and tile 1 holds the remaining 6 plus the 4
    # it re-freezes as its own carry-over overlap.
    layout = cm.compute_chain_layout(DEFAULT_CHAIN, FPS, end_context_px=136)
    assert layout.f_total == 28 and layout.n_end_v == 17
    assert layout.v_tiles == [(0, 22), (18, 10)]
    assert layout.end_tile_bands == [(11, 11), (10, 17)]
    # Read as engine writes: tile 0 writes band[0:11] into its last 11 latents,
    # tile 1 writes band[7:17] into its last 10.
    assert layout.end_tile_bands[0][1] - layout.end_tile_bands[0][0] == 0
    assert layout.end_tile_bands[1][1] - layout.end_tile_bands[1][0] == 7


def test_three_tile_straddle_is_pinned():
    # A 160-frame band on a single 169-frame clip: 20 band latents spread over
    # tiles 1 and 2 while tile 0 never reaches them.
    layout = cm.compute_chain_layout([169], FPS, end_context_px=160)
    assert layout.f_total == 42 and layout.n_end_v == 20
    assert layout.v_tiles == [(0, 22), (18, 22), (36, 6)]
    assert layout.end_tile_bands == [(0, 0), (18, 18), (6, 20)]


def test_tiles_before_the_band_carry_a_zero_length_write():
    # Only the ``t`` half is meaningful for those tiles; ``off`` goes negative
    # and the engine ignores it because ``t == 0``.
    layout = cm.compute_chain_layout([257, 257], FPS, end_context_px=72)
    assert layout.v_tiles == [(0, 22), (18, 22), (36, 22), (54, 18)]
    assert layout.end_tile_bands == [(0, -41), (0, -23), (0, -5), (9, 9)]
    assert [t for t, _ in layout.end_tile_bands] == [0, 0, 0, 9]


def test_a_fully_frozen_tile_is_admissible():
    # H3: the band can swallow a whole short final tile (t == vlen). That is
    # wasteful but valid — the tile's audio is still refined — and must NOT be
    # rejected the way v1 rejected it ("nothing to generate").
    layout = cm.compute_chain_layout(DEFAULT_CHAIN, FPS, end_context_px=96)
    assert layout.v_tiles == [(0, 22), (18, 5)]
    assert layout.end_tile_bands == [(11, 11), (5, 12)]
    t_last, _ = layout.end_tile_bands[-1]
    assert t_last == layout.v_tiles[-1][1]      # the whole tile is frozen


# ── junctions ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("clips", CLIP_SETS)
@pytest.mark.parametrize("end_px", [8, 72, 136])
def test_junction_is_the_last_new_frame_and_the_internal_segment_seam(clips, end_px):
    try:
        layout = cm.compute_chain_layout(clips, FPS, end_context_px=end_px)
    except ValueError as exc:
        assert any(m in str(exc) for m in KNOWN_AUDIO_TILING_MESSAGES), exc
        return
    # Stated from the tail...
    assert layout.end_source_junction_px == layout.total_px - end_px - 1
    # ...and from the segment list: the internal segment's own seam. Two
    # derivations, one number.
    assert layout.segment_seam_junctions[-1] == layout.end_source_junction_px
    assert len(layout.segment_seam_junctions) == len(clips)
    # J is free, J+1 is the first frozen frame, and the band runs to the end.
    assert layout.total_px - 1 - layout.end_source_junction_px == end_px
    # The junction sits exactly where the clips end.
    assert (
        layout.end_source_junction_px
        == layout.to_dict()["end_source"]["clips_total_px"] - 1
    )


def test_cut_frames_is_one_more_than_the_band():
    # The causal VAE spends the upload's frame 0 as the lone keyframe latent, so
    # the caller must always cut/synthesise one frame more than it freezes.
    for end_px in (8, 24, 72, 136):
        d = cm.compute_chain_layout(
            MULTI_TILE_CHAIN, FPS, end_context_px=end_px
        ).to_dict()["end_source"]
        assert d["cut_frames"] == end_px + 1
        assert d["n_end_v"] == end_px // 8


def test_junction_on_the_untrimmed_timeline_when_a_start_source_trims_the_front():
    # Deliberately UNTRIMMED-basis: the delivered mp4 starts trim_px later, and
    # that index is derivable, so no second field is stored.
    layout = cm.compute_chain_layout(
        [169], FPS, source_context_px=73, end_context_px=72
    )
    assert layout.total_px == 169 + 72
    assert layout.trim_px == 73
    assert layout.end_source_junction_px == 168          # untrimmed basis
    assert layout.end_source_junction_px - layout.trim_px == 95   # delivered basis


# ── grid rejections (unchanged from v1) ──────────────────────────────────────
@pytest.mark.parametrize("bad", [71, 73, 75, 4])
def test_off_grid_end_context_is_rejected(bad):
    # A TAIL band is whole groups of 8 counted back from the end — the head
    # grid's 8n+1 (73 among them) is exactly the wrong answer here.
    with pytest.raises(ValueError, match="multiple of 8"):
        cm.compute_chain_layout(MULTI_TILE_CHAIN, FPS, end_context_px=bad)


@pytest.mark.parametrize("bad", [0, 4])
def test_end_context_below_one_latent_is_rejected(bad):
    with pytest.raises(ValueError):
        cm.compute_chain_layout(MULTI_TILE_CHAIN, FPS, end_context_px=bad)


def test_zero_is_rejected_as_too_short_not_as_off_grid():
    # 0 IS a multiple of 8; it fails because it freezes nothing.
    with pytest.raises(ValueError, match=r"must be >= 8"):
        cm.compute_chain_layout(MULTI_TILE_CHAIN, FPS, end_context_px=0)


def test_end_source_and_retake_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        cm.compute_chain_layout(
            [169], FPS, retake_glue_px=(25, 24), end_context_px=72
        )


# ── the ONE new rejection: kv >= 2 ───────────────────────────────────────────
def test_overlap_one_is_rejected_with_an_actionable_message():
    with pytest.raises(ValueError) as exc:
        cm.compute_chain_layout(DEFAULT_CHAIN, FPS, kv=1, end_context_px=72)
    msg = str(exc.value)
    assert "overlap_frames" in msg
    assert ">= 2" in msg
    # It must name the CAUSE (the audio overlap budget), not just the rule —
    # this text is what the app turns into "raise のりしろ to 2".
    assert "audio" in msg
    # And it must be THIS check that fires, not the raw "degenerate audio
    # overlap" one it pre-empts (which says nothing about what to do).
    assert msg.startswith("end_context_px")


def test_overlap_one_is_still_fine_without_an_end_source():
    # The rule is scoped to end source only; a kv=1 chain on its own is
    # untouched (it remains the pre-existing thin-budget territory it was).
    assert cm.compute_chain_layout(DEFAULT_CHAIN, FPS, kv=1).total_px == 97


def test_degenerate_audio_overlap_names_the_kv_one_cause_and_the_fix():
    # The raw "degenerate audio overlap" rejection the kv >= 2 rule pre-empts is
    # still reachable WITHOUT an end source — and only ever at kv=1 (the
    # exhaustive sweep above is what establishes "only"). Its message must
    # therefore carry the same guidance the end-source check does, or the user
    # is told a number is degenerate with no way to act on it.
    with pytest.raises(ValueError) as exc:
        cm.compute_chain_layout([257, 257], 30.0, kv=1)
    msg = str(exc.value)
    assert msg.startswith("degenerate audio overlap")
    # The cause: overlap_frames = 1 exhausted the audio budget at this rate.
    assert "overlap_frames" in msg
    assert "audio overlap budget" in msg
    # The fix, worded as in the end-source kv >= 2 rejection.
    assert "Raise overlap_frames to 2 or more." in msg


@pytest.mark.parametrize("end_px", [8, 72, 136])
def test_kv_sweep_never_degenerates_the_audio_overlap(end_px):
    # The exhaustive claim the kv >= 2 rule rests on: with an end source and
    # kv >= 2 the "degenerate audio overlap" rejection is unreachable at every
    # frame rate the app offers, on every window. (Pre-existing stage-2 audio
    # TILING rejections are a different, end-source-independent family and are
    # tolerated here — see KNOWN_AUDIO_TILING_MESSAGES.)
    checked = 0
    for kv, fps, clips, window in itertools.product(
        range(2, 9), ALL_FPS, CLIP_SETS, BOTH_WINDOWS
    ):
        if any(kv >= cm.v_latent_frames(c) for c in clips):
            continue
        v_tile, v_adv = cm.resolve_stage2_window(window)
        checked += 1
        try:
            layout = cm.compute_chain_layout(
                clips, fps, kv=kv, v_tile=v_tile, v_adv=v_adv,
                end_context_px=end_px,
            )
        except ValueError as exc:
            assert "degenerate audio overlap" not in str(exc), (
                kv, fps, clips, window, str(exc)
            )
            assert any(m in str(exc) for m in KNOWN_AUDIO_TILING_MESSAGES), exc
            continue
        assert min(layout.ka_list, default=1) >= 1
    assert checked > 500

    # At 24fps — the rate the app actually generates at — nothing is rejected at
    # all across the same sweep.
    for kv, clips, window in itertools.product(range(2, 9), CLIP_SETS, BOTH_WINDOWS):
        if any(kv >= cm.v_latent_frames(c) for c in clips):
            continue
        v_tile, v_adv = cm.resolve_stage2_window(window)
        cm.compute_chain_layout(
            clips, FPS, kv=kv, v_tile=v_tile, v_adv=v_adv, end_context_px=end_px
        )


# ── the v1 rejections that are GONE (inverted into "this passes") ────────────
def test_the_box_stock_chain_accepts_the_default_band():
    # v1: "the final clip must be at least 97 pixel frames long for this end
    # source". The band no longer lives in the final clip, so 2x49 is fine.
    layout = cm.compute_chain_layout([49, 49], FPS, end_context_px=72)
    assert layout.n_end_v == 9 and layout.total_px == 153


def test_a_short_final_tile_accepts_the_longest_band():
    # v1: "leaves the last stage-2 tile nothing to generate" for [161, 161] +
    # 136. Straddling freeze makes the last tile's length irrelevant.
    layout = cm.compute_chain_layout([161, 161], FPS, end_context_px=136)
    assert layout.n_end_v == 17
    assert layout.v_tiles[-1] == (36, 20)
    assert layout.end_tile_bands == [(0, -17), (1, 1), (17, 17)]


def test_the_narrow_window_accepts_the_full_136_band():
    # v1 rejected anything over 88 on "high_resolution" (its v_adv - 1 ceiling).
    # There is no per-window ceiling any more — the band is not tile-bound.
    v_tile, v_adv = cm.resolve_stage2_window("high_resolution")
    layout = cm.compute_chain_layout(
        [161, 169], FPS, v_tile=v_tile, v_adv=v_adv, end_context_px=136
    )
    assert layout.n_end_v == 17
    assert layout.total_px == cm.compute_chain_layout(
        [161, 169], FPS, v_tile=v_tile, v_adv=v_adv
    ).total_px + 136
    assert sum(t for t, _ in layout.end_tile_bands) >= 17


def test_start_and_end_source_that_used_to_meet_in_the_middle_now_fit():
    # v1: "no free stage-1 latents" — the head and the band shared clip 0's
    # latents. Now the band has its own segment, so only the V2V head has to fit
    # inside the clip (that check is unchanged).
    layout = cm.compute_chain_layout([169], FPS, source_context_px=105, end_context_px=72)
    assert layout.n_ctx_v == 14 and layout.n_end_v == 9
    assert layout.total_px == 169 + 72
    assert layout.seg_frames == [169, 89]
    # ...while a head that fills its clip completely is still rejected.
    with pytest.raises(ValueError, match="must be < clip 0"):
        cm.compute_chain_layout([169], FPS, source_context_px=169, end_context_px=72)


def test_start_and_end_source_fit_together_on_one_long_clip():
    layout = cm.compute_chain_layout(
        [169], FPS, source_context_px=73, end_context_px=72
    )
    assert layout.n_ctx_v == 10 and layout.n_end_v == 9
    assert layout.to_dict()["v2v"]["source_context_px"] == 73
    assert layout.to_dict()["end_source"]["end_context_px"] == 72


def test_multi_tile_and_multi_clip_chains_accept_an_end_source():
    # Retake caps its window at one stage-2 tile; an end source must not.
    layout = cm.compute_chain_layout([161, 161, 169], FPS, end_context_px=72)
    assert layout.n_tiles == 4
    assert layout.total_px == 457 + 72
    assert layout.seg_frames == [161, 161, 169, 89]


# ── new_frames_px: the delivered length ──────────────────────────────────────
def test_new_frames_px_is_total_minus_trim_on_a_multi_clip_continuation():
    # The v1 bug: ``clip_frames[0] - source_context_px`` reported only clip 0's
    # new tail, so a 3-clip continuation's mock mp4 came out far too short and
    # disagreed with the app's own prediction.
    clips = [169, 257, 257]
    layout = cm.compute_chain_layout(clips, FPS, source_context_px=73)
    assert layout.trim_px == 73
    assert layout.new_frames_px == layout.total_px - 73
    assert layout.new_frames_px != clips[0] - 73        # what v1 answered
    assert layout.to_dict()["v2v"]["new_frames_px"] == layout.total_px - 73


def test_new_frames_px_still_matches_the_one_clip_continuation():
    # The single-clip case the old formula got right must not move.
    layout = cm.compute_chain_layout([49], FPS, source_context_px=25)
    assert layout.new_frames_px == 24 == layout.total_px - layout.trim_px


@pytest.mark.parametrize(
    "clips,ctx",
    [
        ([49], None), ([49], 25),
        ([49, 49], None), ([49, 49], 25),
        ([257, 257], None), ([257, 257], 73), ([257, 257], 145),
        ([169, 257, 257], 73),
    ],
)
@pytest.mark.parametrize("end_px", (None, 72))
def test_new_frames_px_identity_holds_everywhere(clips, ctx, end_px):
    layout = cm.compute_chain_layout(
        clips, FPS, source_context_px=ctx, end_context_px=end_px
    )
    assert layout.new_frames_px == layout.total_px - layout.trim_px
    if end_px is not None:
        # The band IS delivered — the name is narrower than the meaning, which
        # is why the docstring says so.
        assert layout.new_frames_px >= end_px


# ── geometry facts the total-length cap rests on (Phase B's groundwork) ──────
def test_total_length_cap_geometry_is_pinned():
    # api/models.py caps a chain at MAX_CHAIN_TOTAL_PIXEL_FRAMES = 24 * 481 =
    # 11544. These are the extreme configurations that cap has to keep
    # accepting once it is applied to the CLIPS-ONLY total instead of total_px.
    max_clips = [481] * 24
    # The current maximum accepted chain: 24 clips, のりしろ 1.
    assert cm.compute_chain_layout(max_clips, FPS, kv=1).total_px == 11521
    assert 11521 <= 24 * 481
    # With an end source のりしろ must be >= 2, which by itself shortens the
    # assembled clips (each join gives one more latent back).
    biggest = cm.compute_chain_layout(max_clips, FPS, kv=2, end_context_px=136)
    assert biggest.to_dict()["end_source"]["clips_total_px"] == 11337
    assert biggest.total_px == 11337 + 136 == 11473
    # ...so even band-inclusive the largest end-source chain stays under the
    # cap; applying the cap to the clips-only total can therefore not shrink the
    # accepted envelope, it only stops the band from eating into it.
    assert biggest.total_px <= 24 * 481
    assert biggest.f_total == cm.v_latent_frames(11473)


def test_band_never_shortens_the_clips_the_user_asked_for():
    # The one-line statement of pillar 1, checked against the request echo: no
    # clip length is ever adjusted, at any band size.
    for end_px in (8, 72, 136):
        layout = cm.compute_chain_layout([257, 257], FPS, end_context_px=end_px)
        assert layout.clip_frames == [257, 257]
        assert layout.seg_frames[:2] == [257, 257]
