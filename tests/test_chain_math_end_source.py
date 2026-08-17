"""End source (tail freeze) geometry — the pure ``chain_math`` layer.

Pins the numbers the app validator, the engine and the mock ALL resolve from
``compute_chain_layout(end_context_px=...)``. An end source hard-freezes
``end_context_px`` pixel frames of uploaded material at the end of the timeline,
so if any expectation here moves the freeze band lands on the wrong latents — a
silent quality failure, not a crash, exactly like retake (see
tests/test_retake_math.py and VERIFICATION_LOG §55).

THERE ARE TWO MODES, decided by the CLIP COUNT alone and reported as
``end_source_mode``. Almost every test below is therefore parametrised over one
of the two clip-set constants rather than a mixed one:

  * ``MULTI_CLIP_SETS`` -> ``"internal_segment"``: the band gets a stage-1
    segment of its own appended after the clips, so OUTPUT = CLIPS + BAND. The
    older of the two designs, preserved unchanged.
  * ``SINGLE_CLIP_SETS`` -> ``"in_window"``: the band is the one clip's own
    tail, so OUTPUT = THE CLIP, unchanged, and nothing is appended
    (``seg_frames == clip_frames``, ``end_segment_latent == 0``). Stage 1
    denoises that clip as a single window, so the band is inside the window the
    generated latents attend over — the reason the mode exists.

The pillars this file defends, across both modes:

  1. THE OUTPUT-LENGTH IDENTITY OF WHICHEVER MODE IS IN PLAY, and that
     ``clips_total_px`` states it from the other side.
  2. THE BAND MAY STRADDLE STAGE-2 TILES. ``end_tile_bands`` is the single
     source of truth for what each tile freezes, clamps included.
  3. NOTHING IS REJECTED FOR "not fitting" A TILE OR A CLIP in
     ``internal_segment`` mode (the v1 rejections are gone; the one rejection
     there is ``kv >= 2``). ``in_window`` mode adds exactly ONE rejection of its
     own — the band plus a V2V head must leave the clip a free latent — because
     without it the engine crashes at 500 instead of answering 422.
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

# Representative clip layouts, SPLIT BY MODE rather than tested through a
# condition: the two modes have different output-length identities, so a shared
# constant would force every sweep body to branch. The union is the historical
# CLIP_SETS, so nothing dropped out of the coverage when they were separated.
#
# The default pair, long pairs, a ragged triple, the 8-clip and 4-clip shapes
# long chains use -> "internal_segment".
MULTI_CLIP_SETS = (
    [49, 49],
    [105, 113],
    [257, 257],
    [121, 121, 121],
    [73, 89, 97],
    [481, 481],
    [49] * 8,
    [257] * 4,
)

# The shortest legal clip and the one-latent-clip length the real-hardware
# experiment uses -> "in_window".
SINGLE_CLIP_SETS = (
    [49],
    [169],
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

# The ONE rejection "in_window" mode adds: with a single clip the band is carved
# out of that clip's own latents, so a short clip and a long band can leave
# nothing to generate. The single-clip sweeps below tolerate it the same way
# they tolerate the audio-tiling quirk — the sweeps are about the geometry of
# the accepted cases, and which combinations are accepted is pinned separately
# by the boundary test.
NO_FREE_LATENTS_MESSAGE = "leaves the clip no free video latents"


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
        "mode": "internal_segment",      # two clips
        "end_source_junction_px": 80,    # 153 - 72 - 1
        "cut_frames": 73,                # the causal VAE's +1 primer
        "end_segment_px": 89,            # px_from_v_latent(3 + 9)
        "end_segment_latent": 12,        # kv + n_end_v
        "end_tile_bands": [[9, 9]],      # one tile, band at its tail
        "clips_total_px": 81,            # what [49, 49] alone assembles to
    }


def test_end_source_sub_dict_shape_in_window():
    # The same dict for the OTHER mode: same keys, and the three that state
    # "where the band lives" all say "inside the clip" instead.
    layout = cm.compute_chain_layout([169], FPS, end_context_px=24)
    assert layout.to_dict()["end_source"] == {
        "end_context_px": 24,
        "n_end_v": 3,                    # 24 // 8
        "mode": "in_window",             # one clip
        "end_source_junction_px": 144,   # 169 - 24 - 1
        "cut_frames": 25,
        "end_segment_px": 0,             # no appended segment...
        "end_segment_latent": 0,         # ...at all
        "end_tile_bands": [[3, 3]],
        "clips_total_px": 169,           # == total_px: the band is IN the clip
    }


# ── pillar 1: output length identity ─────────────────────────────────────────
@pytest.mark.parametrize("clips", MULTI_CLIP_SETS)
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
        assert layout.end_source_mode == "internal_segment"


@pytest.mark.parametrize("clips", SINGLE_CLIP_SETS)
@pytest.mark.parametrize("end_px", [8, 24, 72, 136])
def test_output_length_is_the_clip_itself_in_window(clips, end_px):
    """The OTHER half of pillar 1: with one clip the band is the clip's own tail,
    so the output length is the clip length and nothing is appended."""
    for fps, kv, window in itertools.product(ALL_FPS, (2, 3, 5), BOTH_WINDOWS):
        if any(kv >= cm.v_latent_frames(c) for c in clips):
            continue
        v_tile, v_adv = cm.resolve_stage2_window(window)
        try:
            layout = cm.compute_chain_layout(
                clips, fps, kv=kv, v_tile=v_tile, v_adv=v_adv,
                end_context_px=end_px,
            )
        except ValueError as exc:   # audio-tiling quirk, or the band is too long
            assert (
                NO_FREE_LATENTS_MESSAGE in str(exc)
                or any(m in str(exc) for m in KNOWN_AUDIO_TILING_MESSAGES)
            ), exc
            continue
        clips_only = cm.compute_chain_layout(
            clips, fps, kv=kv, v_tile=v_tile, v_adv=v_adv
        )
        assert layout.end_source_mode == "in_window"
        assert layout.total_px == clips_only.total_px == clips[0]
        assert layout.f_total == clips_only.f_total
        assert layout.seg_frames == clips
        assert layout.end_segment_px == layout.end_segment_latent == 0
        # ...and the number published to clients is the whole timeline, because
        # the band is not an addition to it.
        assert layout.to_dict()["end_source"]["clips_total_px"] == layout.total_px


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


@pytest.mark.parametrize("clips", MULTI_CLIP_SETS + SINGLE_CLIP_SETS)
@pytest.mark.parametrize("end_px", [8, 40, 72, 136])
@pytest.mark.parametrize("window", BOTH_WINDOWS)
def test_end_tile_bands_match_a_naive_intersection_and_cover_the_band(
    clips, end_px, window
):
    # Pillar 2 is the one thing the two modes share verbatim — the band sits at
    # the end of the timeline either way — so this sweep runs over BOTH clip-set
    # constants, tolerating each mode's own rejections.
    v_tile, v_adv = cm.resolve_stage2_window(window)
    try:
        layout = cm.compute_chain_layout(
            clips, FPS, v_tile=v_tile, v_adv=v_adv, end_context_px=end_px
        )
    except ValueError as exc:
        assert (
            NO_FREE_LATENTS_MESSAGE in str(exc)
            or any(m in str(exc) for m in KNOWN_AUDIO_TILING_MESSAGES)
        ), exc
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
    # A 160-frame band on clips that assemble to 169 pixel frames: 20 band
    # latents spread over tiles 1 and 2 while tile 0 never reaches them.
    # (``internal_segment`` mode, hence two clips — [97, 89] assembles to the
    # same 22 latents a lone 169-frame clip would, so the tile geometry is
    # exactly the one this test has always pinned.)
    layout = cm.compute_chain_layout([97, 89], FPS, end_context_px=160)
    assert layout.to_dict()["end_source"]["clips_total_px"] == 169
    assert layout.f_total == 42 and layout.n_end_v == 20
    assert layout.v_tiles == [(0, 22), (18, 22), (36, 6)]
    assert layout.end_tile_bands == [(0, 0), (18, 18), (6, 20)]


def test_three_tile_straddle_is_pinned_in_window():
    # The same property in ``in_window`` mode, where the timeline is the clip:
    # a 337-frame clip is three tiles and a 136-frame band reaches the last two.
    layout = cm.compute_chain_layout([337], FPS, end_context_px=136)
    assert layout.end_source_mode == "in_window"
    assert layout.total_px == 337
    assert layout.f_total == 43 and layout.n_end_v == 17
    assert layout.v_tiles == [(0, 22), (18, 22), (36, 7)]
    assert layout.end_tile_bands == [(0, -4), (14, 14), (7, 17)]


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
@pytest.mark.parametrize("clips", MULTI_CLIP_SETS)
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


@pytest.mark.parametrize("clips", SINGLE_CLIP_SETS)
@pytest.mark.parametrize("end_px", [8, 72, 136])
def test_junction_in_window_is_an_interior_index_with_no_seam(clips, end_px):
    """In ``in_window`` mode the junction is where the material STARTS inside the
    one continuous window — NOT a seam. The segment list must stay seamless:
    creating a join there is exactly what this mode exists to avoid."""
    try:
        layout = cm.compute_chain_layout(clips, FPS, end_context_px=end_px)
    except ValueError as exc:
        assert NO_FREE_LATENTS_MESSAGE in str(exc), exc
        return
    assert layout.segment_seam_junctions == []
    assert layout.end_source_junction_px == layout.total_px - end_px - 1
    assert layout.end_source_junction_px == clips[0] - end_px - 1
    assert layout.total_px - 1 - layout.end_source_junction_px == end_px
    # The clips' own end IS the timeline's end here, so the junction is strictly
    # inside it rather than sitting at clips_total_px - 1.
    assert 0 <= layout.end_source_junction_px < layout.total_px - 1
    assert layout.to_dict()["end_source"]["clips_total_px"] == layout.total_px


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
    # that index is derivable, so no second field is stored. One clip, so this is
    # ``in_window`` mode: the 169 frames hold the frozen head, the free middle
    # and the frozen band all at once, and the timeline does not grow.
    layout = cm.compute_chain_layout(
        [169], FPS, source_context_px=73, end_context_px=72
    )
    assert layout.end_source_mode == "in_window"
    assert layout.total_px == 169
    assert layout.trim_px == 73
    assert layout.end_source_junction_px == 96           # untrimmed basis
    assert layout.end_source_junction_px - layout.trim_px == 23   # delivered basis
    # The delivered mp4 is the 96 frames after the trim, of which the last 72 are
    # the material — i.e. 24 newly generated frames reach the viewer.
    assert layout.new_frames_px == 169 - 73 == 96


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
    all_sets = MULTI_CLIP_SETS + SINGLE_CLIP_SETS
    checked = 0
    for kv, fps, clips, window in itertools.product(
        range(2, 9), ALL_FPS, all_sets, BOTH_WINDOWS
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
            assert (
                NO_FREE_LATENTS_MESSAGE in str(exc)
                or any(m in str(exc) for m in KNOWN_AUDIO_TILING_MESSAGES)
            ), exc
            continue
        assert min(layout.ka_list, default=1) >= 1
    assert checked > 500

    # At 24fps — the rate the app actually generates at — nothing is rejected at
    # all across the same sweep, apart from single clips too short for the band.
    for kv, clips, window in itertools.product(range(2, 9), all_sets, BOTH_WINDOWS):
        if any(kv >= cm.v_latent_frames(c) for c in clips):
            continue
        v_tile, v_adv = cm.resolve_stage2_window(window)
        try:
            cm.compute_chain_layout(
                clips, FPS, kv=kv, v_tile=v_tile, v_adv=v_adv, end_context_px=end_px
            )
        except ValueError as exc:
            assert len(clips) == 1 and NO_FREE_LATENTS_MESSAGE in str(exc), exc


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


def test_start_and_end_source_that_meet_in_the_middle_fit_with_two_clips():
    # v1: "no free stage-1 latents" — the head and the band shared clip 0's
    # latents. In ``internal_segment`` mode the band has its own segment, so only
    # the V2V head has to fit inside clip 0 (that check is unchanged).
    layout = cm.compute_chain_layout(
        [169, 169], FPS, source_context_px=105, end_context_px=72
    )
    assert layout.end_source_mode == "internal_segment"
    assert layout.n_ctx_v == 14 and layout.n_end_v == 9
    assert layout.seg_frames == [169, 169, 89]
    assert layout.total_px == layout.to_dict()["end_source"]["clips_total_px"] + 72
    # ...while a head that fills its clip completely is still rejected.
    with pytest.raises(ValueError, match="must be < clip 0"):
        cm.compute_chain_layout(
            [169, 169], FPS, source_context_px=169, end_context_px=72
        )


def test_start_and_end_source_that_meet_in_the_middle_are_rejected_in_window():
    # The SAME request with one clip is the inverse: ``in_window`` mode carves
    # the band out of the clip the V2V head is already frozen into, and 14 + 9
    # fills all 22 of the clip's latents. That must be a 422 here rather than a
    # 500 from the engine's tail-token range.
    with pytest.raises(ValueError) as exc:
        cm.compute_chain_layout([169], FPS, source_context_px=105, end_context_px=72)
    msg = str(exc.value)
    assert NO_FREE_LATENTS_MESSAGE in msg
    assert "n_ctx_v=14" in msg and "n_end_v=9" in msg
    # A shorter head leaves room, and then the timeline is still just the clip.
    layout = cm.compute_chain_layout([169], FPS, source_context_px=73, end_context_px=72)
    assert layout.n_ctx_v == 10 and layout.n_end_v == 9
    assert layout.total_px == 169
    assert layout.seg_frames == [169]


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
        # The single-clip rows are 169 rather than 49 frames: with one clip the
        # band is frozen INSIDE the clip, and a 49-frame clip is 7 stage-1
        # latents — too few to hold a 72-frame band at all (that rejection is
        # pinned by its own boundary test). The identity under test is unrelated
        # to the clip length. A 49-frame single-clip continuation without an end
        # source is still pinned by the test just above.
        ([169], None), ([169], 25),
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


# ── the mode switch itself ───────────────────────────────────────────────────
def test_the_clip_count_alone_picks_the_mode_and_the_segment_list():
    """One clip -> ``in_window`` and NOTHING is appended; two -> the historical
    ``internal_segment``. Nothing else about the request participates, and a
    chain without an end source has no mode at all."""
    one = cm.compute_chain_layout([169], FPS, end_context_px=72)
    assert one.end_source_mode == "in_window"
    assert one.seg_frames == [169] == one.clip_frames
    assert one.seg_latent == [22]
    assert one.end_segment_px == 0 and one.end_segment_latent == 0
    assert one.ka_list == []            # one segment -> no join at all

    two = cm.compute_chain_layout([169, 169], FPS, end_context_px=72)
    assert two.end_source_mode == "internal_segment"
    assert two.seg_frames == [169, 169, 89]
    assert two.clip_frames == [169, 169]
    assert two.end_segment_latent == 3 + 9

    # The same clip counts, band aside, must be seg_frames-identical to a chain
    # with no end source — i.e. ``in_window`` really appends nothing.
    plain_one = cm.compute_chain_layout([169], FPS)
    assert plain_one.seg_frames == one.seg_frames
    assert plain_one.total_px == one.total_px
    assert plain_one.v_tiles == one.v_tiles
    assert plain_one.end_source_mode is None

    # ...and the switch is insensitive to everything but the clip count.
    for kwargs in ({"kv": 4}, {"source_context_px": 25}, {"fps": 30.0}):
        fps = kwargs.pop("fps", FPS)
        assert cm.compute_chain_layout(
            [169], fps, end_context_px=72, **kwargs
        ).end_source_mode == "in_window"


# ── the ONE rejection "in_window" mode adds ──────────────────────────────────
def test_in_window_band_that_fills_the_clip_is_rejected_at_the_boundary():
    """A 49-frame clip is 7 stage-1 latents. A 48-frame band takes 6 of them and
    leaves exactly one free -> accepted; 56 takes all 7 -> rejected, because the
    denoiser would have nothing to generate (and the engine would raise a 500
    from ``retake_tail_token_range`` instead of the API answering 422)."""
    ok = cm.compute_chain_layout([49], FPS, end_context_px=48)
    assert ok.f_total == 7 and ok.n_end_v == 6
    assert ok.total_px == 49
    assert ok.end_tile_bands == [(6, 6)]

    with pytest.raises(ValueError) as exc:
        cm.compute_chain_layout([49], FPS, end_context_px=56)
    msg = str(exc.value)
    # (a) what ran out, in the layout's own terms
    assert NO_FREE_LATENTS_MESSAGE in msg
    assert "n_end_v=7" in msg and "n_ctx_v=0" in msg and "f_total=7" in msg
    # (b) the concrete clip length that WOULD work: 8 free-ish latents == 57 px
    assert "57 pixel frames" in msg
    assert cm.compute_chain_layout([57], FPS, end_context_px=56).total_px == 57
    # (c) all three ways out, so the app can quote the message verbatim
    assert "Lengthen the clip" in msg
    assert "shorten context_frames" in msg
    assert "two or more clips" in msg
    # ...and the same band is fine the moment there are two clips (that mode
    # appends a segment for it instead of carving it out).
    assert cm.compute_chain_layout([49, 49], FPS, end_context_px=56).total_px == 81 + 56


def test_in_window_rejection_does_not_leak_into_the_other_mode():
    # The check is scoped to one clip. A multi-clip chain whose CLIPS are just as
    # short keeps working, because the band never touches them.
    layout = cm.compute_chain_layout([49, 49], FPS, end_context_px=136)
    assert layout.end_source_mode == "internal_segment"
    assert layout.total_px == 81 + 136


# ── the experiment geometry, pinned ──────────────────────────────────────────
@pytest.mark.parametrize("clip_px", [169, 337, 481])
@pytest.mark.parametrize("end_px", [8, 16, 24])
def test_in_window_output_length_never_moves(clip_px, end_px):
    """The headline promise of the mode: ask for N frames, get N frames, whatever
    the band. (The real-hardware experiment sweeps exactly this grid.)"""
    layout = cm.compute_chain_layout([clip_px], FPS, end_context_px=end_px)
    assert layout.total_px == clip_px
    assert layout.new_frames_px == clip_px          # nothing is trimmed either
    assert layout.to_dict()["end_source"]["clips_total_px"] == clip_px
    assert layout.end_source_junction_px == clip_px - end_px - 1
    assert layout.duration_sec == round(clip_px / FPS, 3)


def test_the_real_hardware_experiment_geometry_is_pinned():
    """The exact numbers the MCP experiment's machine check compares against, so
    a geometry drift is caught here rather than after 30 minutes of GPU time.
    Experiment A is one 169-frame clip at three band widths; experiment B keeps
    the band at 24 and grows the clip across 3 and 4 stage-2 tiles."""
    # A-1 / A-2 / A-3: one stage-2 tile, band at its tail.
    for end_px, n_end_v in ((8, 1), (16, 2), (24, 3)):
        a = cm.compute_chain_layout([169], FPS, end_context_px=end_px)
        assert a.f_total == 22
        assert a.v_tiles == [(0, 22)]
        assert a.end_tile_bands == [(n_end_v, n_end_v)]
        assert a.n_tiles == 1
    a3 = cm.compute_chain_layout([169], FPS, end_context_px=24)
    assert a3.end_source_junction_px == 144
    assert a3.segment_seam_junctions == []

    # B-2: 337 frames == 3 stage-2 tiles; the band lands on the last one only.
    b2 = cm.compute_chain_layout([337], FPS, end_context_px=24)
    assert b2.total_px == 337 and b2.f_total == 43
    assert b2.end_tile_bands == [(0, -18), (0, 0), (3, 3)]
    assert b2.end_source_junction_px == 312

    # B-3: 481 frames == 4 tiles, same story.
    b3 = cm.compute_chain_layout([481], FPS, end_context_px=24)
    assert b3.total_px == 481 and b3.f_total == 61
    assert b3.end_tile_bands == [(0, -36), (0, -18), (0, 0), (3, 3)]
    assert b3.end_source_junction_px == 456

    # A-4 reuses the archived v2 job's parameters (169 + 72) — the direct
    # old/new comparison — and is still one tile.
    a4 = cm.compute_chain_layout([169], FPS, end_context_px=72)
    assert a4.total_px == 169 and a4.end_tile_bands == [(9, 9)]


# ── both ends frozen in the same window ──────────────────────────────────────
def test_a_head_and_a_band_may_share_one_window_until_they_meet():
    """``source_video`` + ``end_source`` on one clip is retake's both-ends shape:
    legal while a free latent remains, rejected the moment it does not."""
    ok = cm.compute_chain_layout([169], FPS, source_context_px=73, end_context_px=72)
    assert ok.n_ctx_v == 10 and ok.n_end_v == 9 and ok.f_total == 22
    assert ok.n_ctx_v + ok.n_end_v < ok.f_total     # 19 < 22
    assert ok.total_px == 169
    assert ok.trim_px == 73
    # The frozen head is at the FRONT and the band at the BACK of one window,
    # and the junction sits after the trim (what the layout asserts internally).
    assert ok.end_source_junction_px == 96 >= ok.trim_px

    with pytest.raises(ValueError, match=NO_FREE_LATENTS_MESSAGE):
        cm.compute_chain_layout([169], FPS, source_context_px=105, end_context_px=72)


def test_overlap_one_is_rejected_in_window_mode_too():
    """The ``kv >= 2`` rule is kept in BOTH modes even though a one-clip chain has
    no join for the audio budget to be spent on. Deliberately conservative: the
    accepted range does not widen, and the frontend keeps ONE rule for the whole
    feature. Pinned so relaxing it is a decision, not an accident."""
    with pytest.raises(ValueError) as exc:
        cm.compute_chain_layout([169], FPS, kv=1, end_context_px=24)
    msg = str(exc.value)
    assert msg.startswith("end_context_px")
    assert ">= 2" in msg
    # kv=2 is the first accepted value, and it is a genuinely join-free layout.
    layout = cm.compute_chain_layout([169], FPS, kv=2, end_context_px=24)
    assert layout.ka_list == [] and layout.total_px == 169


def test_band_never_shortens_the_clips_the_user_asked_for():
    # The one-line statement of pillar 1, checked against the request echo: no
    # clip length is ever adjusted, at any band size.
    for end_px in (8, 72, 136):
        layout = cm.compute_chain_layout([257, 257], FPS, end_context_px=end_px)
        assert layout.clip_frames == [257, 257]
        assert layout.seg_frames[:2] == [257, 257]
