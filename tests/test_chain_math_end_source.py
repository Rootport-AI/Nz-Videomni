"""End source (tail freeze) geometry — the pure ``chain_math`` layer.

Pins the numbers the app validator, the engine and the mock ALL resolve from
``compute_chain_layout(end_context_px=...)``. An end source hard-freezes
``end_context_px`` pixel frames of uploaded material at the end of the timeline,
so if any expectation here moves the freeze band lands on the wrong latents — a
silent quality failure, not a crash, exactly like retake (see
tests/test_retake_math.py and VERIFICATION_LOG §55).

THE MODE IS DECIDED BY THE CLIP COUNT ALONE, and there are two reachable
answers. Almost every test below is therefore parametrised over one of the two
clip-set constants rather than a mixed one:

  * ``SINGLE_CLIP_SETS`` -> ``"in_window"``: the band is the one clip's own
    tail, so OUTPUT = THE CLIP, unchanged, and nothing is appended
    (``seg_frames == clip_frames``, ``end_segment_latent == 0``). Stage 1
    denoises that clip as a single window, so the band is inside the window the
    generated latents attend over — the reason the mode exists.
  * ``MULTI_CLIP_SETS`` -> ``"reverse"``: the SAME promise on a chain. The band
    is the LAST clip's own tail, nothing is appended, and OUTPUT = THE CLIPS'
    OWN TOTAL. What the mode adds is a stage-1 SCHEDULE — the three tables
    ``seg_generation_order`` / ``seg_head_source`` / ``seg_tail_source`` — that
    generates the clips last-to-first, each freezing the next one's head as its
    own tail.

A THIRD MODE, ``"internal_segment"``, IS THE HISTORICAL TWO-OR-MORE-CLIPS
DESIGN: the band got a stage-1 segment of its own appended after the clips, so
OUTPUT = CLIPS + BAND. It is no longer reachable from the API, and its tests are
kept here — reached through ``end_source_mode_override``, which exists for
exactly this and for the rollback — so the geometry cannot rot unnoticed and the
rollback stays a one-line change. They use ``LEGACY_MULTI_CLIP_SETS``, the clip
sets that mode was pinned on.

The pillars this file defends, across every mode:

  1. THE OUTPUT-LENGTH IDENTITY OF WHICHEVER MODE IS IN PLAY, and that
     ``clips_total_px`` states it from the other side.
  2. THE BAND MAY STRADDLE STAGE-2 TILES. ``end_tile_bands`` is the single
     source of truth for what each tile freezes, clamps included.
  3. THE THREE SCHEDULE TABLES ARE INERT OUTSIDE ``reverse``. On a plain chain,
     V2V, retake and the legacy mode they are exactly what the engine's old
     ``for i in range(n_seg)`` loop did, which is what makes their introduction a
     no-op on every previously validated path.
  4. WHAT EACH MODE REJECTS. ``in_window`` needs a clip with free latents left;
     ``reverse`` needs a LAST clip that holds the band plus the のり代, and
     refuses a start source outright; the legacy mode rejects neither but needs
     ``kv >= 2``, which ``reverse`` is exempt from.
"""

from __future__ import annotations

import itertools

import pytest

import chain_math as cm

FPS = 24.0

# The owner-approved default: 72 pixel frames == 3.0s at 24fps.
DEFAULT_END_PX = 72

# The BOX-STOCK chain: two 49-frame clips, the webui default. Short clips like
# these can hold no band of their own, so with an end source they belong to the
# legacy mode; ``REVERSE_CHAIN`` is the two-clip workhorse for the current one.
DEFAULT_CHAIN = [49, 49]
REVERSE_CHAIN = [169, 169]

# A chain that needs several stage-2 tiles.
MULTI_TILE_CHAIN = [161, 169]

# Frame rates the app can generate at (the webui's whole list).
ALL_FPS = (23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0, 59.94, 60.0)

# Representative clip layouts, SPLIT BY MODE rather than tested through a
# condition: the modes have different output-length identities and different
# rejections, so a shared constant would force every sweep body to branch.
#
# Two-or-more-clip sets whose LAST clip can hold the band plus the のり代 ->
# "reverse". Pairs, ragged triples and the 4-clip shape long chains use.
MULTI_CLIP_SETS = (
    [169, 169],
    [257, 257],
    [169, 177, 185],
    [161, 161, 169],
    [481, 481],
    [257] * 4,
)

# The sets the RETIRED internal-segment mode was pinned on, kept verbatim: that
# mode never required a clip to be long enough for the band, so its sets include
# ones ``reverse`` refuses. Reached only through the override.
LEGACY_MULTI_CLIP_SETS = (
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
# overlap" is NOT — that is the one an end source could newly provoke.
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

# ``reverse`` mode's own version of the same idea, one clip further along: the
# LAST clip must hold the band AND the のり代 the clip before it takes from its
# head. Tolerated in the sweeps for the same reason, pinned by its own boundary
# test below.
NOTHING_TO_CARRY_MESSAGE = "nothing to carry backwards"


def _expected_tables(n_seg: int, *, reverse: bool):
    """The three schedule tables, re-derived the long way.

    Deliberately written as two explicit literals rather than as the
    implementation's expression, so a change to that expression has to be
    restated here rather than being agreed with automatically.
    """
    if reverse:
        order = list(reversed(range(n_seg)))
        head: list[int | None] = [None for _ in range(n_seg)]
        tail: list[int | None] = [
            *(j for j in range(1, n_seg)), None,
        ]
        return order, head, tail
    order = list(range(n_seg))
    head = [None, *range(n_seg - 1)]
    tail = [None for _ in range(n_seg)]
    return order, head, tail


def _assert_tables(layout: cm.ChainLayout, *, reverse: bool) -> None:
    n_seg = len(layout.seg_frames)
    order, head, tail = _expected_tables(n_seg, reverse=reverse)
    assert layout.seg_generation_order == order
    assert layout.seg_head_source == head
    assert layout.seg_tail_source == tail


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
    # no freeze plan, no sub-dict — and the schedule is the plain ascending one.
    plain = cm.compute_chain_layout(DEFAULT_CHAIN, FPS)
    assert "end_source" not in plain.to_dict()
    assert plain.end_context_px is None
    assert plain.n_end_v == 0
    assert plain.n_end_a == 0
    assert plain.end_source_junction_px is None
    assert plain.seg_frames == DEFAULT_CHAIN        # segments == clips exactly
    assert plain.seg_latent == [7, 7]
    assert plain.end_segment_px == 0
    assert plain.end_segment_latent == 0
    assert plain.end_tile_bands == []
    assert plain.end_tile_bands_a == []
    _assert_tables(plain, reverse=False)


def test_top_level_keys_are_frozen_and_only_end_source_is_added():
    with_end = cm.compute_chain_layout(REVERSE_CHAIN, FPS, end_context_px=DEFAULT_END_PX)
    without = cm.compute_chain_layout(REVERSE_CHAIN, FPS)
    a, b = with_end.to_dict(), without.to_dict()
    assert set(a) - set(b) == {"end_source"}
    assert set(b) == {
        "clip_frames", "fps", "kv", "v_tile", "v_adv", "kt_v",
        "seg_latent", "seg_audio", "f_total_latent", "total_px", "a_total",
        "ka_list", "video_tiles", "audio_tiles", "audio_adv", "kt_a", "n_tiles",
        "segment_seam_junctions", "tile_seam_junctions", "all_junctions",
        "duration_sec",
    }
    # ``seg_frames`` and the three schedule tables are deliberately NOT new
    # top-level keys: seg_latent / seg_audio already report the segment view, the
    # schedule is published under ``end_source`` (the only mode that changes it),
    # and the metadata contract is frozen.
    assert with_end.clip_frames == REVERSE_CHAIN
    assert a["clip_frames"] == REVERSE_CHAIN
    # In ``reverse`` mode the band is the last clip's own tail, so the timeline
    # is the SAME LENGTH it would be with no end source — and therefore the tiles
    # and junctions do not move either.
    assert a["total_px"] == b["total_px"]
    assert a["video_tiles"] == b["video_tiles"]
    assert a["all_junctions"] == b["all_junctions"]


def test_end_source_sub_dict_shape():
    layout = cm.compute_chain_layout(REVERSE_CHAIN, FPS, end_context_px=DEFAULT_END_PX)
    assert layout.to_dict()["end_source"] == {
        "end_context_px": 72,
        "n_end_v": 9,                    # 72 // 8 (tail grid)
        "n_end_a": 73,                   # the causal scan, NOT round(72/24*25)
        "mode": "reverse",               # two clips
        "generation_order": [1, 0],      # last clip first
        "end_source_junction_px": 248,   # 321 - 72 - 1
        "cut_frames": 73,                # the causal VAE's +1 primer
        "end_segment_px": 0,             # nothing is appended...
        "end_segment_latent": 0,         # ...at all
        "end_tile_bands": [[0, -10], [8, 8], [5, 9]],
        "end_tile_bands_a": [[0, -85], [65, 65], [34, 73]],
        "clips_total_px": 321,           # == total_px: the band is IN the clips
    }


def test_end_source_sub_dict_shape_in_window():
    # The same dict for the one-clip mode: same keys, and the schedule is the
    # degenerate single-entry one.
    layout = cm.compute_chain_layout([169], FPS, end_context_px=24)
    assert layout.to_dict()["end_source"] == {
        "end_context_px": 24,
        "n_end_v": 3,                    # 24 // 8
        "n_end_a": 24,                   # the causal scan, NOT round(24/24*25)
        "mode": "in_window",             # one clip
        "generation_order": [0],
        "end_source_junction_px": 144,   # 169 - 24 - 1
        "cut_frames": 25,
        "end_segment_px": 0,             # no appended segment...
        "end_segment_latent": 0,         # ...at all
        "end_tile_bands": [[3, 3]],
        "end_tile_bands_a": [[24, 24]],
        "clips_total_px": 169,           # == total_px: the band is IN the clip
    }


def test_end_source_sub_dict_shape_internal_segment():
    # The RETIRED mode's dict, unchanged apart from the new schedule key (which
    # is the plain ascending one there — the legacy design generated forwards).
    layout = cm.compute_chain_layout(
        DEFAULT_CHAIN, FPS, end_context_px=DEFAULT_END_PX,
        end_source_mode_override="internal_segment",
    )
    assert layout.to_dict()["end_source"] == {
        "end_context_px": 72,
        "n_end_v": 9,
        "n_end_a": 73,
        "mode": "internal_segment",
        "generation_order": [0, 1, 2],   # clips, then the appended band segment
        "end_source_junction_px": 80,    # 153 - 72 - 1
        "cut_frames": 73,
        "end_segment_px": 89,            # px_from_v_latent(3 + 9)
        "end_segment_latent": 12,        # kv + n_end_v
        "end_tile_bands": [[9, 9]],      # one tile, band at its tail
        "end_tile_bands_a": [[73, 73]],  # one tile, audio band at its tail
        "clips_total_px": 81,            # what [49, 49] alone assembles to
    }


# ── pillar 1: output length identity ─────────────────────────────────────────
@pytest.mark.parametrize("clips", MULTI_CLIP_SETS)
@pytest.mark.parametrize("end_px", [8, 24, 72, 136])
def test_output_length_is_the_clips_total_in_reverse(clips, end_px):
    """The headline promise of the mode, on a chain: ask for N frames of clips,
    get exactly those N frames. The band is the last clip's own tail, so a chain
    with an end source is the SAME LENGTH as the same chain without one."""
    for fps, kv, window in itertools.product(ALL_FPS, (1, 2, 3, 5), BOTH_WINDOWS):
        if any(kv >= cm.v_latent_frames(c) for c in clips):
            continue
        v_tile, v_adv = cm.resolve_stage2_window(window)
        try:
            layout = cm.compute_chain_layout(
                clips, fps, kv=kv, v_tile=v_tile, v_adv=v_adv,
                end_context_px=end_px,
            )
        except ValueError as exc:
            assert (
                NOTHING_TO_CARRY_MESSAGE in str(exc)
                or "degenerate audio overlap" in str(exc)
                or any(m in str(exc) for m in KNOWN_AUDIO_TILING_MESSAGES)
            ), exc
            continue
        clips_only = cm.compute_chain_layout(
            clips, fps, kv=kv, v_tile=v_tile, v_adv=v_adv
        )
        assert layout.end_source_mode == "reverse"
        assert layout.total_px == clips_only.total_px
        assert layout.f_total == clips_only.f_total
        assert layout.seg_frames == clips
        assert layout.end_segment_px == layout.end_segment_latent == 0
        # ...and the number published to clients agrees with the independent
        # clips-only layout, so the app's "of which N frames are the source"
        # breakdown can never drift from the geometry.
        assert layout.to_dict()["end_source"]["clips_total_px"] == clips_only.total_px


@pytest.mark.parametrize("clips", LEGACY_MULTI_CLIP_SETS)
@pytest.mark.parametrize("end_px", [8, 24, 72, 136])
def test_output_length_is_clips_plus_band_in_the_legacy_mode(clips, end_px):
    """The RETIRED identity, pinned unchanged: there the band EXTENDED the
    timeline the clips described. This is what makes the delivered frame count a
    decisive new/old discriminator — the same request is ``end_px`` frames longer
    under the old geometry."""
    for fps, kv, window in itertools.product(ALL_FPS, (2, 3, 5), BOTH_WINDOWS):
        if any(kv >= cm.v_latent_frames(c) for c in clips):
            continue
        v_tile, v_adv = cm.resolve_stage2_window(window)
        try:
            layout = cm.compute_chain_layout(
                clips, fps, kv=kv, v_tile=v_tile, v_adv=v_adv,
                end_context_px=end_px,
                end_source_mode_override="internal_segment",
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
        assert (
            layout.to_dict()["end_source"]["clips_total_px"] == clips_only.total_px
        )
        assert layout.end_source_mode == "internal_segment"


@pytest.mark.parametrize("clips", SINGLE_CLIP_SETS)
@pytest.mark.parametrize("end_px", [8, 24, 72, 136])
def test_output_length_is_the_clip_itself_in_window(clips, end_px):
    """The one-clip half of pillar 1: with one clip the band is the clip's own
    tail, so the output length is the clip length and nothing is appended."""
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


def test_reverse_chain_pins_every_derived_number():
    layout = cm.compute_chain_layout(REVERSE_CHAIN, FPS, end_context_px=DEFAULT_END_PX)
    assert layout.seg_frames == [169, 169]
    assert layout.seg_latent == [22, 22]
    assert layout.f_total == 41                      # 22 + 22 - 3
    assert layout.total_px == 321
    assert layout.n_end_v == 9
    assert layout.end_source_junction_px == 248
    assert layout.duration_sec == round(321 / 24.0, 3)
    # ...and the same clips with no end source at all give the same numbers.
    plain = cm.compute_chain_layout(REVERSE_CHAIN, FPS)
    assert (plain.f_total, plain.total_px) == (41, 321)


def test_legacy_default_chain_pins_every_derived_number():
    layout = cm.compute_chain_layout(
        DEFAULT_CHAIN, FPS, end_context_px=DEFAULT_END_PX,
        end_source_mode_override="internal_segment",
    )
    assert layout.seg_frames == [49, 49, 89]
    assert layout.seg_latent == [7, 7, 12]
    assert layout.f_total == 20                      # 11 (clips) + 9 (band)
    assert layout.total_px == 153                    # 81 + 72
    assert layout.n_end_v == 9
    assert layout.end_source_junction_px == 80
    assert layout.duration_sec == round(153 / 24.0, 3)


def test_headline_chain_length_no_longer_grows_by_the_band():
    # H1 in the v2 real-hardware gate: the box-stock [257, 257] chain plus a
    # 72-frame band delivered 569 frames, of which the last 72 were the material.
    # THAT NUMBER IS RETIRED. The same request now delivers the clips' own 497,
    # of which the last 72 are the material — which is the number the app shows.
    layout = cm.compute_chain_layout([257, 257], FPS, end_context_px=72)
    assert layout.total_px == 497
    assert layout.to_dict()["end_source"]["clips_total_px"] == 497
    legacy = cm.compute_chain_layout(
        [257, 257], FPS, end_context_px=72,
        end_source_mode_override="internal_segment",
    )
    assert legacy.total_px == 569 == 497 + 72


# ── pillar 1: the legacy mode's internal segment ─────────────────────────────
@pytest.mark.parametrize("kv", [2, 3, 4, 8])
@pytest.mark.parametrize("end_px", [8, 72, 136])
def test_internal_segment_is_carry_plus_band(kv, end_px):
    clips = [169, 177]
    layout = cm.compute_chain_layout(
        clips, FPS, kv=kv, end_context_px=end_px,
        end_source_mode_override="internal_segment",
    )
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
    # The schedule is the plain forward one — the legacy mode never reversed.
    _assert_tables(layout, reverse=False)


def test_internal_segment_stays_far_below_the_per_clip_ceiling():
    # Why that design was chosen: a band could not push a stage-1 pass past the
    # 481-frame per-clip ceiling, because it arrived as its own SHORT segment. At
    # the published maximum (kv=3, band 136) that segment is 20 latent frames ==
    # 153 pixel frames.
    layout = cm.compute_chain_layout(
        [481, 481], FPS, end_context_px=136,
        end_source_mode_override="internal_segment",
    )
    assert layout.end_segment_latent == 20
    assert layout.end_segment_px == 153
    assert max(layout.seg_frames) == 481


def test_reverse_has_the_same_ceiling_property_for_free():
    # ``reverse`` never lengthens a stage-1 pass either, for a stronger reason:
    # it appends nothing at all, so every segment is a clip the caller already
    # had to keep under the 481-frame ceiling.
    layout = cm.compute_chain_layout([481, 481], FPS, end_context_px=136)
    assert layout.seg_frames == [481, 481]
    assert layout.end_segment_latent == layout.end_segment_px == 0
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
    # Pillar 2 is the one thing every mode shares verbatim — the band sits at the
    # end of the timeline either way — so this sweep runs over BOTH clip-set
    # constants, tolerating each mode's own rejections.
    v_tile, v_adv = cm.resolve_stage2_window(window)
    try:
        layout = cm.compute_chain_layout(
            clips, FPS, v_tile=v_tile, v_adv=v_adv, end_context_px=end_px
        )
    except ValueError as exc:
        assert (
            NO_FREE_LATENTS_MESSAGE in str(exc)
            or NOTHING_TO_CARRY_MESSAGE in str(exc)
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
    # Two clips that assemble to 305 frames with the longest published band:
    # f_total 39 -> tiles (0,22) and (18,21); the band's 17 latents start at 22,
    # so tile 0 ends exactly on the boundary and tile 1 holds all 17.
    layout = cm.compute_chain_layout([161, 161], FPS, end_context_px=136)
    assert layout.f_total == 39 and layout.n_end_v == 17
    assert layout.v_tiles == [(0, 22), (18, 21)]
    assert layout.end_tile_bands == [(0, 0), (17, 17)]


def test_two_tile_straddle_is_pinned_in_the_legacy_mode():
    # The gate-H2 shape, unchanged: the 2x49 chain with the longest published
    # band. f_total 28 -> tiles (0,22) and (18,10); the band's 17 latents start at
    # 11, so tile 0 ends on 11 of them and tile 1 holds the remaining 6 plus the 4
    # it re-freezes as its own carry-over overlap.
    layout = cm.compute_chain_layout(
        DEFAULT_CHAIN, FPS, end_context_px=136,
        end_source_mode_override="internal_segment",
    )
    assert layout.f_total == 28 and layout.n_end_v == 17
    assert layout.v_tiles == [(0, 22), (18, 10)]
    assert layout.end_tile_bands == [(11, 11), (10, 17)]
    # Read as engine writes: tile 0 writes band[0:11] into its last 11 latents,
    # tile 1 writes band[7:17] into its last 10.
    assert layout.end_tile_bands[0][1] - layout.end_tile_bands[0][0] == 0
    assert layout.end_tile_bands[1][1] - layout.end_tile_bands[1][0] == 7


def test_three_tile_straddle_is_pinned_in_the_legacy_mode():
    # A 160-frame band on clips that assemble to 169 pixel frames: 20 band
    # latents spread over tiles 1 and 2 while tile 0 never reaches them. Only the
    # legacy mode can hang a band that long on clips that short.
    layout = cm.compute_chain_layout(
        [97, 89], FPS, end_context_px=160,
        end_source_mode_override="internal_segment",
    )
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
    assert layout.v_tiles == [(0, 22), (18, 22), (36, 22), (54, 9)]
    assert layout.end_tile_bands == [(0, -32), (0, -14), (4, 4), (9, 9)]
    assert [t for t, _ in layout.end_tile_bands] == [0, 0, 4, 9]


def test_a_fully_frozen_tile_is_admissible():
    # H3: the band can swallow a whole short final tile (t == vlen). That is
    # wasteful but valid — the tile's audio is still refined — and must NOT be
    # rejected the way v1 rejected it ("nothing to generate").
    layout = cm.compute_chain_layout(
        DEFAULT_CHAIN, FPS, end_context_px=96,
        end_source_mode_override="internal_segment",
    )
    assert layout.v_tiles == [(0, 22), (18, 5)]
    assert layout.end_tile_bands == [(11, 11), (5, 12)]
    t_last, _ = layout.end_tile_bands[-1]
    assert t_last == layout.v_tiles[-1][1]      # the whole tile is frozen


# ── junctions ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("clips", MULTI_CLIP_SETS)
@pytest.mark.parametrize("end_px", [8, 72, 136])
def test_junction_in_reverse_is_inside_the_last_clip_and_after_every_seam(
    clips, end_px
):
    """In ``reverse`` mode the band is the LAST CLIP's own tail, so the junction
    is an interior index of the timeline that sits strictly AFTER the last
    segment seam. Creating a seam there is exactly what the mode avoids."""
    try:
        layout = cm.compute_chain_layout(clips, FPS, end_context_px=end_px)
    except ValueError as exc:
        assert (
            NOTHING_TO_CARRY_MESSAGE in str(exc)
            or any(m in str(exc) for m in KNOWN_AUDIO_TILING_MESSAGES)
        ), exc
        return
    # Stated from the tail...
    assert layout.end_source_junction_px == layout.total_px - end_px - 1
    # ...and the band is INSIDE the last clip, i.e. past the final seam.
    assert len(layout.segment_seam_junctions) == len(clips) - 1
    assert layout.segment_seam_junctions[-1] < layout.end_source_junction_px
    assert layout.end_source_junction_px < layout.total_px - 1
    # J is free, J+1 is the first frozen frame, and the band runs to the end.
    assert layout.total_px - 1 - layout.end_source_junction_px == end_px
    # The clips' own end IS the timeline's end here.
    assert layout.to_dict()["end_source"]["clips_total_px"] == layout.total_px


@pytest.mark.parametrize("clips", LEGACY_MULTI_CLIP_SETS)
@pytest.mark.parametrize("end_px", [8, 72, 136])
def test_junction_is_the_internal_segment_seam_in_the_legacy_mode(clips, end_px):
    try:
        layout = cm.compute_chain_layout(
            clips, FPS, end_context_px=end_px,
            end_source_mode_override="internal_segment",
        )
    except ValueError as exc:
        assert any(m in str(exc) for m in KNOWN_AUDIO_TILING_MESSAGES), exc
        return
    # Stated from the tail...
    assert layout.end_source_junction_px == layout.total_px - end_px - 1
    # ...and from the segment list: the internal segment's own seam. Two
    # derivations, one number.
    assert layout.segment_seam_junctions[-1] == layout.end_source_junction_px
    assert len(layout.segment_seam_junctions) == len(clips)
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
    one continuous window — NOT a seam. The segment list must stay seamless."""
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
    # that index is derivable, so no second field is stored. ONE clip, which is
    # the only shape a start source and an end source may share.
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


# ── the kv >= 2 rule, and reverse's exemption from it ────────────────────────
def test_overlap_one_is_rejected_in_window_mode_with_an_actionable_message():
    with pytest.raises(ValueError) as exc:
        cm.compute_chain_layout([169], FPS, kv=1, end_context_px=24)
    msg = str(exc.value)
    assert "overlap_frames" in msg
    assert ">= 2" in msg
    # It must name the CAUSE (the audio overlap budget), not just the rule —
    # this text is what the app turns into "raise のりしろ to 2".
    assert "audio" in msg
    # And it must be THIS check that fires, not the raw "degenerate audio
    # overlap" one it pre-empts (which says nothing about what to do).
    assert msg.startswith("end_context_px")
    # kv=2 is the first accepted value, and it is a genuinely join-free layout.
    layout = cm.compute_chain_layout([169], FPS, kv=2, end_context_px=24)
    assert layout.ka_list == [] and layout.total_px == 169


def test_overlap_one_is_accepted_in_reverse_mode():
    """``reverse`` appends no segment, so it spends exactly the audio budget an
    end-source-less chain spends — and kv=1 is in fact its INTENDED value (one
    shared latent per reverse seam). The exemption is pinned against the
    in-window rule above so neither can drift into the other."""
    layout = cm.compute_chain_layout([169] * 3, FPS, kv=1, end_context_px=8)
    assert layout.end_source_mode == "reverse"
    assert layout.total_px == 505
    assert layout.ka_list == [1, 1]
    assert layout.seg_generation_order == [2, 1, 0]
    # The exemption is NOT a hole: the raw budget rejection still guards it, and
    # this layout passes it with exactly nothing to spare (sum_ka == n_join).
    assert sum(layout.ka_list) == len(layout.ka_list)
    # ...and the same chain WITH the legacy geometry would still be refused,
    # because there the appended segment really does cost one more join.
    with pytest.raises(ValueError, match=">= 2"):
        cm.compute_chain_layout(
            [169] * 3, FPS, kv=1, end_context_px=8,
            end_source_mode_override="internal_segment",
        )


def test_reverse_still_hits_the_raw_audio_budget_rejection_when_it_must():
    # The safety net the exemption leans on, shown firing: at 30fps a kv=1
    # two-clip chain has no audio budget left, with or without an end source.
    with pytest.raises(ValueError, match="degenerate audio overlap"):
        cm.compute_chain_layout([257, 257], 30.0, kv=1, end_context_px=8)


def test_overlap_one_is_still_fine_without_an_end_source():
    # The rule is scoped to end source only; a kv=1 chain on its own is
    # untouched (it remains the pre-existing thin-budget territory it was).
    assert cm.compute_chain_layout(DEFAULT_CHAIN, FPS, kv=1).total_px == 97


def test_degenerate_audio_overlap_names_the_kv_one_cause_and_the_fix():
    # The raw "degenerate audio overlap" rejection the kv >= 2 rule pre-empts is
    # still reachable WITHOUT an end source — and only ever at kv=1 (the
    # exhaustive sweep below is what establishes "only"). Its message must
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
                or NOTHING_TO_CARRY_MESSAGE in str(exc)
                or any(m in str(exc) for m in KNOWN_AUDIO_TILING_MESSAGES)
            ), exc
            continue
        assert min(layout.ka_list, default=1) >= 1
    assert checked > 500

    # At 24fps — the rate the app actually generates at — nothing is rejected at
    # all across the same sweep, apart from clips too short for the band.
    for kv, clips, window in itertools.product(range(2, 9), all_sets, BOTH_WINDOWS):
        if any(kv >= cm.v_latent_frames(c) for c in clips):
            continue
        v_tile, v_adv = cm.resolve_stage2_window(window)
        try:
            cm.compute_chain_layout(
                clips, FPS, kv=kv, v_tile=v_tile, v_adv=v_adv, end_context_px=end_px
            )
        except ValueError as exc:
            assert (
                NO_FREE_LATENTS_MESSAGE in str(exc)
                or NOTHING_TO_CARRY_MESSAGE in str(exc)
            ), exc


# ── the v1 rejections that are GONE (inverted into "this passes") ────────────
def test_the_box_stock_chain_accepts_the_default_band_in_the_legacy_mode():
    # v1: "the final clip must be at least 97 pixel frames long for this end
    # source". With the band in its own segment 2x49 was fine.
    layout = cm.compute_chain_layout(
        [49, 49], FPS, end_context_px=72,
        end_source_mode_override="internal_segment",
    )
    assert layout.n_end_v == 9 and layout.total_px == 153


def test_a_short_final_tile_accepts_the_longest_band():
    # v1: "leaves the last stage-2 tile nothing to generate". The straddling
    # freeze makes the last TILE's length irrelevant in every mode — what
    # ``reverse`` requires is a long enough last CLIP, which is a different rule.
    layout = cm.compute_chain_layout([161, 161], FPS, end_context_px=136)
    assert layout.n_end_v == 17
    assert layout.v_tiles[-1] == (18, 21)
    assert layout.end_tile_bands == [(0, 0), (17, 17)]


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
    ).total_px
    assert sum(t for t, _ in layout.end_tile_bands) >= 17


def test_multi_tile_and_multi_clip_chains_accept_an_end_source():
    # Retake caps its window at one stage-2 tile; an end source must not.
    layout = cm.compute_chain_layout([161, 161, 169], FPS, end_context_px=72)
    assert layout.n_tiles == 3
    assert layout.total_px == 457
    assert layout.seg_frames == [161, 161, 169]
    assert layout.seg_generation_order == [2, 1, 0]


# ── the two rejections "reverse" mode adds ───────────────────────────────────
def test_a_last_clip_too_short_for_the_band_is_rejected_at_the_boundary():
    """The band plus the のり代 the previous clip takes from this clip's head must
    leave the LAST clip something to generate. With K_v=3 and a 72-frame band
    (9 latents) that clip needs 13 latents == 97 pixel frames: 89 is refused."""
    ok = cm.compute_chain_layout([49, 97], FPS, end_context_px=72)
    assert ok.seg_latent[-1] == 13 and ok.n_end_v == 9
    assert ok.total_px == ok.to_dict()["end_source"]["clips_total_px"]

    with pytest.raises(ValueError) as exc:
        cm.compute_chain_layout([49, 89], FPS, end_context_px=72)
    msg = str(exc.value)
    # (a) what ran out, in the layout's own terms
    assert NOTHING_TO_CARRY_MESSAGE in msg
    assert "n_end_v=9" in msg and "K_v=3" in msg
    # (b) the concrete clip length that WOULD work
    assert "97 pixel frames" in msg
    # (c) all three ways out, so the app can quote the message verbatim
    assert "Lengthen the last clip" in msg
    assert "shorten context_frames" in msg
    assert "lower overlap_frames" in msg
    # ...and lowering the のり代 really is a way out, exactly as the message says.
    assert cm.compute_chain_layout([49, 89], FPS, kv=2, end_context_px=72).total_px > 0
    # The rule is about the LAST clip alone: the ones before it may be as short
    # as the plain K_v floor allows.
    assert cm.compute_chain_layout([25, 25, 169], FPS, end_context_px=72).total_px > 0


def test_a_start_source_is_rejected_with_two_or_more_clips():
    """Start + end on ONE clip is the interpolation case and stays accepted. On a
    chain it would freeze clip 0 at BOTH ends (head from the source video, tail
    from the reverse のり代), which nothing has generated — refused rather than
    shipped untested."""
    with pytest.raises(ValueError) as exc:
        cm.compute_chain_layout(
            [169, 169], FPS, source_context_px=73, end_context_px=72
        )
    msg = str(exc.value)
    assert "cannot be combined" in msg
    assert "2 clips" in msg
    assert "ONE clip" in msg

    # The one-clip pair is untouched...
    ok = cm.compute_chain_layout([169], FPS, source_context_px=73, end_context_px=72)
    assert ok.end_source_mode == "in_window"
    # ...and so is a start source on a chain with NO end source.
    assert cm.compute_chain_layout(
        [169, 169], FPS, source_context_px=73
    ).trim_px == 73
    # The legacy mode accepted the pair on a chain; kept under test so the
    # rollback restores a known geometry rather than an untested one.
    legacy = cm.compute_chain_layout(
        [169, 169], FPS, source_context_px=105, end_context_px=72,
        end_source_mode_override="internal_segment",
    )
    assert legacy.n_ctx_v == 14 and legacy.n_end_v == 9
    assert legacy.seg_frames == [169, 169, 89]


def test_start_and_end_source_that_meet_in_the_middle_are_rejected_in_window():
    # ``in_window`` mode carves the band out of the clip the V2V head is already
    # frozen into, and 14 + 9 fills all 22 of the clip's latents. That must be a
    # 422 here rather than a 500 from the engine's tail-token range.
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
    "clips,ctx,end_px",
    [
        # A start source and an end source may share a timeline only on ONE clip
        # (that pairing has its own rejection test above), so the multi-clip rows
        # carry one or the other, never both. The single-clip rows are 169 rather
        # than 49 frames: with one clip the band is frozen INSIDE the clip, and a
        # 49-frame clip is 7 stage-1 latents — too few to hold a 72-frame band at
        # all (pinned by its own boundary test).
        ([169], None, None), ([169], 25, None),
        ([169], None, 72), ([169], 25, 72),
        ([49, 49], None, None), ([49, 49], 25, None),
        ([257, 257], None, None), ([257, 257], 73, None), ([257, 257], 145, None),
        ([257, 257], None, 72), ([169, 257, 257], None, 72),
        ([169, 257, 257], 73, None),
    ],
)
def test_new_frames_px_identity_holds_everywhere(clips, ctx, end_px):
    layout = cm.compute_chain_layout(
        clips, FPS, source_context_px=ctx, end_context_px=end_px
    )
    assert layout.new_frames_px == layout.total_px - layout.trim_px
    if end_px is not None:
        # The band IS delivered — the name is narrower than the meaning, which
        # is why the docstring says so.
        assert layout.new_frames_px >= end_px


# ── geometry facts the total-length cap rests on ─────────────────────────────
def test_total_length_cap_geometry_is_pinned():
    # api/models.py caps a chain at MAX_CHAIN_TOTAL_PIXEL_FRAMES = 24 * 481 =
    # 11544. These are the extreme configurations that cap has to keep accepting
    # once it is applied to the CLIPS-ONLY total instead of total_px.
    max_clips = [481] * 24
    # The current maximum accepted chain: 24 clips, のりしろ 1.
    assert cm.compute_chain_layout(max_clips, FPS, kv=1).total_px == 11521
    assert 11521 <= 24 * 481
    # With an end source in ``reverse`` mode nothing is appended, so the band
    # costs the delivered length NOTHING and のりしろ 1 stays legal.
    biggest = cm.compute_chain_layout(max_clips, FPS, kv=1, end_context_px=136)
    assert biggest.to_dict()["end_source"]["clips_total_px"] == 11521
    assert biggest.total_px == 11521
    assert biggest.total_px <= 24 * 481
    assert biggest.f_total == cm.v_latent_frames(11521)
    # The legacy mode is where the two numbers differed, and it still does —
    # which is why the cap is charged on ``clips_total_px`` rather than total_px.
    legacy = cm.compute_chain_layout(
        max_clips, FPS, kv=2, end_context_px=136,
        end_source_mode_override="internal_segment",
    )
    assert legacy.to_dict()["end_source"]["clips_total_px"] == 11337
    assert legacy.total_px == 11337 + 136 == 11473


# ── the mode switch itself ───────────────────────────────────────────────────
def test_the_clip_count_alone_picks_the_mode_and_the_segment_list():
    """One clip -> ``in_window``, two or more -> ``reverse``, and NOTHING is
    appended either way. Nothing else about the request participates, and a chain
    without an end source has no mode at all."""
    one = cm.compute_chain_layout([169], FPS, end_context_px=72)
    assert one.end_source_mode == "in_window"
    assert one.seg_frames == [169] == one.clip_frames
    assert one.seg_latent == [22]
    assert one.end_segment_px == 0 and one.end_segment_latent == 0
    assert one.ka_list == []            # one segment -> no join at all

    two = cm.compute_chain_layout([169, 169], FPS, end_context_px=72)
    assert two.end_source_mode == "reverse"
    assert two.seg_frames == [169, 169] == two.clip_frames
    assert two.end_segment_px == 0 and two.end_segment_latent == 0

    three = cm.compute_chain_layout([169] * 3, FPS, end_context_px=72)
    assert three.end_source_mode == "reverse"
    assert three.seg_generation_order == [2, 1, 0]

    # Both modes must be seg_frames-identical to the same clips with no end
    # source — i.e. neither appends anything.
    for clips in ([169], [169, 169]):
        plain = cm.compute_chain_layout(clips, FPS)
        withes = cm.compute_chain_layout(clips, FPS, end_context_px=72)
        assert plain.seg_frames == withes.seg_frames
        assert plain.total_px == withes.total_px
        assert plain.v_tiles == withes.v_tiles
        assert plain.end_source_mode is None

    # ...and the switch is insensitive to everything but the clip count.
    for kwargs in ({"kv": 4}, {"source_context_px": 25}, {"fps": 30.0}):
        fps = kwargs.pop("fps", FPS)
        assert cm.compute_chain_layout(
            [169], fps, end_context_px=72, **kwargs
        ).end_source_mode == "in_window"
    for kwargs in ({"kv": 4}, {"fps": 30.0}, {"v_tile": 19, "v_adv": 12}):
        fps = kwargs.pop("fps", FPS)
        assert cm.compute_chain_layout(
            [169, 169], fps, end_context_px=72, **kwargs
        ).end_source_mode == "reverse"


def test_the_mode_override_is_the_only_way_to_reach_the_legacy_geometry():
    # It is not derivable from any request shape...
    for clips in ([169], [169, 169], [169] * 4):
        assert cm.compute_chain_layout(
            clips, FPS, end_context_px=72
        ).end_source_mode != "internal_segment"
    # ...and a typo in the override fails loudly rather than silently falling
    # back to a derived mode. Checked on EVERY call, end source or not.
    with pytest.raises(ValueError, match="unknown end_source_mode_override"):
        cm.compute_chain_layout([169], FPS, end_source_mode_override="internal")
    with pytest.raises(ValueError, match="unknown end_source_mode_override"):
        cm.compute_chain_layout([169, 169], FPS, end_source_mode_override="reversed")


# ── pillar 3: the three schedule tables ──────────────────────────────────────
@pytest.mark.parametrize(
    "kwargs,reverse",
    [
        # plain chain / V2V / retake / in-window / the legacy mode -> the plain
        # ascending schedule the engine's loop has always run.
        ({}, False),
        ({"source_context_px": 73}, False),
        ({"end_context_px": 72, "end_source_mode_override": "internal_segment"}, False),
        # ...and only the reverse mode changes it.
        ({"end_context_px": 72}, True),
    ],
)
@pytest.mark.parametrize("clips", [[169, 169], [169] * 3, [169] * 8])
def test_the_schedule_tables_are_inert_outside_reverse(clips, kwargs, reverse):
    layout = cm.compute_chain_layout(clips, FPS, **kwargs)
    _assert_tables(layout, reverse=reverse)


def test_the_schedule_tables_on_one_segment_layouts():
    # One clip: every mode degenerates to the same single-entry schedule.
    for kwargs in ({}, {"end_context_px": 24}, {"retake_glue_px": (25, 24)}):
        layout = cm.compute_chain_layout([169], FPS, **kwargs)
        assert layout.seg_generation_order == [0]
        assert layout.seg_head_source == [None]
        assert layout.seg_tail_source == [None]


def test_the_reverse_schedule_is_a_valid_dependency_order():
    """The invariants the engine's loop leans on, restated independently: the
    order is a permutation, every dependency is generated first, and the anchor's
    tail is never also a のり代 target."""
    for clips in ([169, 169], [169] * 3, [257] * 4, [169] * 24):
        layout = cm.compute_chain_layout(clips, FPS, kv=1, end_context_px=8)
        n_seg = len(layout.seg_frames)
        order = layout.seg_generation_order
        assert sorted(order) == list(range(n_seg))
        pos = {seg: p for p, seg in enumerate(order)}
        for i in range(n_seg):
            for src in (layout.seg_head_source[i], layout.seg_tail_source[i]):
                assert src is None or pos[src] < pos[i], (i, src, order)
        # The anchor lands on the TIMELINE's last segment, which is generated
        # FIRST and borrows from nothing.
        assert order[0] == n_seg - 1
        assert layout.seg_tail_source[n_seg - 1] is None
        assert layout.seg_head_source[n_seg - 1] is None


def test_the_reverse_schedule_carries_every_seam_exactly_once():
    """Each seam is used by exactly one direction: in ``reverse`` every join is a
    tail carry, in every other mode every join is a head carry. The count of
    non-None entries must equal the number of joins either way."""
    clips = [169] * 4
    rev = cm.compute_chain_layout(clips, FPS, kv=1, end_context_px=8)
    fwd = cm.compute_chain_layout(clips, FPS, kv=1)
    n_join = len(clips) - 1
    assert sum(x is not None for x in rev.seg_tail_source) == n_join
    assert sum(x is not None for x in rev.seg_head_source) == 0
    assert sum(x is not None for x in fwd.seg_head_source) == n_join
    assert sum(x is not None for x in fwd.seg_tail_source) == 0
    # ...and both spend the SAME audio overlap budget, since neither appends a
    # segment. That equality is the whole justification for the kv >= 2 exemption.
    assert rev.ka_list == fwd.ka_list
    assert sum(rev.ka_list) == sum(fwd.ka_list) == len(rev.ka_list)


def test_generation_order_is_published_only_under_the_end_source_key():
    # A chain without an end source must not grow the key (the metadata contract
    # is frozen), even though the table itself exists on every layout.
    plain = cm.compute_chain_layout([169, 169], FPS).to_dict()
    assert "end_source" not in plain
    assert "generation_order" not in plain
    withes = cm.compute_chain_layout([169, 169], FPS, end_context_px=72).to_dict()
    assert withes["end_source"]["generation_order"] == [1, 0]


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


def test_each_mode_rejects_on_its_own_terms_not_the_others():
    # ``in_window``'s check is scoped to one clip and ``reverse``'s to the LAST
    # clip; the legacy mode has neither. The same short-clip request therefore
    # gets three different answers, which is what stops one rule leaking.
    with pytest.raises(ValueError, match=NO_FREE_LATENTS_MESSAGE):
        cm.compute_chain_layout([49], FPS, end_context_px=136)
    with pytest.raises(ValueError, match=NOTHING_TO_CARRY_MESSAGE):
        cm.compute_chain_layout([49, 49], FPS, end_context_px=136)
    legacy = cm.compute_chain_layout(
        [49, 49], FPS, end_context_px=136,
        end_source_mode_override="internal_segment",
    )
    assert legacy.total_px == 81 + 136


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


def test_the_reverse_worked_example_is_pinned():
    """The three-clip example the design note works through end to end: 169f x 3
    at 24fps with のりしろ 1 and an 8-frame anchor. Every number the real-hardware
    gate reads out of metadata.json is here."""
    layout = cm.compute_chain_layout([169] * 3, FPS, kv=1, end_context_px=8)
    assert layout.seg_latent == [22, 22, 22]
    assert layout.f_total == 64
    assert layout.total_px == 505
    assert layout.a_total == 526
    assert layout.ka_list == [1, 1]
    assert layout.n_end_v == 1
    assert layout.v_tiles == [(0, 22), (18, 22), (36, 22), (54, 10)]
    assert layout.end_source_junction_px == 496
    # The band is inside the LAST clip: past the final segment seam (336) and
    # before the timeline's end.
    assert layout.segment_seam_junctions == [168, 336]
    assert 336 < 496 < 504
    assert layout.seg_generation_order == [2, 1, 0]
    assert layout.seg_tail_source == [1, 2, None]
    assert layout.to_dict()["end_source"]["clips_total_px"] == 505


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


def test_band_never_shortens_the_clips_the_user_asked_for():
    # The one-line statement of pillar 1, checked against the request echo: no
    # clip length is ever adjusted, at any band size.
    for end_px in (8, 72, 136):
        layout = cm.compute_chain_layout([257, 257], FPS, end_context_px=end_px)
        assert layout.clip_frames == [257, 257]
        assert layout.seg_frames == [257, 257]


# ------------------------------------------------- (j) strength (batch 1, riding along)


@pytest.mark.parametrize("strength", [0.0, 0.25, 0.5, 1.0])
def test_freeze_mask_values_mirrors_end_source_strength(strength):
    """``chain_pipeline`` resolves an end source's stage-1 tail mask as
    ``1.0 - strength`` and passes it as ``tail_mask_value``, leaving the
    ordinary carry-over ``mask_value`` in charge of the head. All four
    resolved values are pinned here (video/audio x head/tail), pure and
    torch-free, at the strength range's two ends and its midpoints."""
    mask_value = 0.5  # an ordinary carry-over seam (overlap_strength default)
    tail_mask_value = 1.0 - strength
    v_head, v_tail, a_head, a_tail = cm.freeze_mask_values(
        mask_value, tail_mask_value=tail_mask_value
    )
    assert v_head == mask_value
    assert v_tail == tail_mask_value == 1.0 - strength
    assert a_head == mask_value
    assert a_tail == tail_mask_value


def test_the_reverse_carry_holds_both_ends_at_the_seam_strength():
    """The reverse のり代 passes NO tail override, which is what makes it a mirror
    of the forward one: all four bands — video AND audio, head AND tail — land on
    ``1 - overlap_strength``. That equality is the mode's "the same seam knob
    works in both directions" promise, stated where it is actually resolved."""
    for overlap_strength in (0.0, 0.5, 1.0):
        mask_value = 1.0 - overlap_strength
        assert cm.freeze_mask_values(mask_value) == (
            mask_value, mask_value, mask_value, mask_value
        )
    # It holds only while the AUDIO override is also absent — A2V is what would
    # set it, and A2V is API-exclusive with an end source.
    v_head, v_tail, a_head, a_tail = cm.freeze_mask_values(0.5, None, 0.0)
    assert (v_head, v_tail) == (0.5, 0.5)
    assert (a_head, a_tail) == (0.0, 0.0)


# ── pillar 5: the band's AUDIO (n_end_a / end_tile_bands_a) ──────────────────
# The material's own audio is frozen over the same band its video is, so the
# geometry grows an audio twin of every number pillar 2 pins. What makes it a
# pillar of its own rather than a footnote is that the two grids do NOT move
# together: video latents advance in groups of 8 pixel frames, audio latents at
# 25 per second, so the audio band's width is a SCAN over the causal patch grid
# and its per-tile plan can be nonzero where the video's is zero.


def _naive_bands_a(layout: cm.ChainLayout) -> list[tuple[int, int]]:
    """The audio counterpart of :func:`_naive_bands`, written the same long way."""
    band_lo = layout.a_total - layout.n_end_a
    band_hi = layout.a_total
    out: list[tuple[int, int]] = []
    for as_, alen in layout.a_tiles:
        lo, hi = max(as_, band_lo), min(as_ + alen, band_hi)
        out.append((max(0, hi - lo), as_ + alen - band_lo))
    return out


@pytest.mark.parametrize(
    "clips, fps, kv, end_px, mode, n_end_v, a_total, n_end_a",
    [
        ([169], 24.0, 3, 72, "in_window", 9, 176, 74),
        ([169, 169], 24.0, 1, 8, "reverse", 1, 351, 7),
        ([169, 169, 169], 24.0, 1, 72, "reverse", 9, 526, 74),
        ([121], 30.0, 2, 48, "in_window", 6, 101, 39),
        ([241], 60.0, 2, 136, "in_window", 17, 100, 55),
    ],
)
def test_n_end_a_pinned_values(clips, fps, kv, end_px, mode, n_end_v, a_total, n_end_a):
    """The five worked examples the design was settled on. If any of these moves,
    the frozen audio lands on different latents than the ones the design and the
    real-hardware gate were written against."""
    layout = cm.compute_chain_layout(clips, fps, kv=kv, end_context_px=end_px)
    assert layout.end_source_mode == mode
    assert (layout.n_end_v, layout.a_total, layout.n_end_a) == (n_end_v, a_total, n_end_a)


def test_n_end_a_is_the_causal_scan_and_not_a_rounding():
    """``round(end_context_px / fps * 25)`` is a DIFFERENT number, and using it
    would push the frozen band into freely generated material. Stated on a case
    where the two disagree, so the scan cannot be quietly replaced."""
    layout = cm.compute_chain_layout([169], FPS, end_context_px=72)
    naive = round(72 / FPS * cm.AUDIO_LATENTS_PER_SEC)
    assert naive == 75 and layout.n_end_a == 74
    # Every frozen latent's support starts at or after the band's start time —
    # the property the scan buys and the rounding loses.
    band_start_s = (layout.total_px - 72) / FPS
    first = layout.a_total - layout.n_end_a
    assert cm.audio_latent_support_sec(first)[0] >= band_start_s - 1e-9
    assert cm.audio_latent_support_sec(first - 1)[0] < band_start_s - 1e-9
    assert cm.audio_latent_support_sec(layout.a_total - naive)[0] < band_start_s - 1e-9


@pytest.mark.parametrize("clips", MULTI_CLIP_SETS + SINGLE_CLIP_SETS)
@pytest.mark.parametrize("end_px", [8, 40, 72, 136])
def test_n_end_a_boundaries_hold_across_the_grid(clips, end_px):
    """The three things the engine's writes depend on, over every fps, K_v and
    stage-2 window the app can ask for: the band is non-empty and fits the
    timeline, it leaves the last segment room for the のり代 that segment's
    neighbour takes out of it, and no frozen latent reaches back before the
    band's start time."""
    checked = 0
    for fps, kv, window in itertools.product(ALL_FPS, (1, 2, 3, 5), BOTH_WINDOWS):
        if any(kv >= cm.v_latent_frames(c) for c in clips):
            continue
        v_tile, v_adv = cm.resolve_stage2_window(window)
        try:
            layout = cm.compute_chain_layout(
                clips, fps, kv=kv, v_tile=v_tile, v_adv=v_adv, end_context_px=end_px
            )
        except ValueError as exc:
            assert (
                NO_FREE_LATENTS_MESSAGE in str(exc)
                or NOTHING_TO_CARRY_MESSAGE in str(exc)
                or "degenerate audio overlap" in str(exc)
                or any(m in str(exc) for m in KNOWN_AUDIO_TILING_MESSAGES)
            ), exc
            continue
        checked += 1
        assert 0 < layout.n_end_a <= layout.a_total
        # Slack against the のり代 taken out of the last segment's head. Two
        # latents is the measured minimum over the whole reachable grid; pinning
        # the bound (not merely "they do not overlap") is what would catch the
        # geometry drifting towards the degenerate case.
        ka_last = layout.ka_list[-1] if layout.ka_list else 0
        assert layout.n_end_a + ka_last + 2 <= layout.seg_audio[-1]
        # A start source's frozen audio head and this tail share one window in
        # the in-window mode; they must not meet there either.
        assert layout.n_ctx_a + layout.n_end_a < layout.a_total
        band_start_s = (layout.total_px - end_px) / float(fps)
        first = layout.a_total - layout.n_end_a
        assert cm.audio_latent_support_sec(first)[0] >= band_start_s - 1e-9
    # A parametrisation where NOTHING was accepted proves nothing, and the only
    # legitimate reason for that is a band at least as long as the clip it would
    # have to fit inside.
    assert checked or end_px >= min(clips)


@pytest.mark.parametrize("clips", MULTI_CLIP_SETS + SINGLE_CLIP_SETS)
@pytest.mark.parametrize("end_px", [8, 40, 72, 136])
@pytest.mark.parametrize("window", BOTH_WINDOWS)
def test_end_tile_bands_a_match_a_naive_intersection_and_cover_the_band(
    clips, end_px, window
):
    """Pillar 2's sweep, on the audio grid: one entry per tile, both clamps
    honoured, the writes covering the WHOLE band, and the last tile holding its
    end — the properties the stage-2 audio freeze depends on."""
    v_tile, v_adv = cm.resolve_stage2_window(window)
    try:
        layout = cm.compute_chain_layout(
            clips, FPS, v_tile=v_tile, v_adv=v_adv, end_context_px=end_px
        )
    except ValueError as exc:
        assert (
            NO_FREE_LATENTS_MESSAGE in str(exc)
            or NOTHING_TO_CARRY_MESSAGE in str(exc)
            or any(m in str(exc) for m in KNOWN_AUDIO_TILING_MESSAGES)
        ), exc
        return
    assert len(layout.end_tile_bands_a) == layout.n_tiles
    assert layout.end_tile_bands_a == _naive_bands_a(layout)

    covered: set[int] = set()
    for (t, off), (as_, alen) in zip(layout.end_tile_bands_a, layout.a_tiles):
        assert 0 <= t <= alen                      # the min(alen, ...) clamp
        if t == 0:
            continue
        assert 0 <= off - t < off <= layout.n_end_a
        covered |= set(range(off - t, off))
    assert covered == set(range(layout.n_end_a))
    assert layout.end_tile_bands_a[-1][1] == layout.n_end_a
    assert layout.end_tile_bands_a[-1][0] > 0


@pytest.mark.parametrize("clips", MULTI_CLIP_SETS + SINGLE_CLIP_SETS)
@pytest.mark.parametrize("end_px", [8, 72, 136])
def test_tail_tile_bands_reproduces_the_video_plan_it_replaced(clips, end_px):
    """``end_tile_bands`` is now the shared :func:`chain_math.tail_tile_bands`
    applied to the video grid. Calling it directly must give the stored list back
    — that equality is what makes the extraction a no-op for every video path
    validated before it."""
    for window in BOTH_WINDOWS:
        v_tile, v_adv = cm.resolve_stage2_window(window)
        try:
            layout = cm.compute_chain_layout(
                clips, FPS, v_tile=v_tile, v_adv=v_adv, end_context_px=end_px
            )
        except ValueError:
            continue
        assert layout.end_tile_bands == cm.tail_tile_bands(
            layout.v_tiles, layout.f_total, layout.n_end_v
        )
        assert layout.end_tile_bands_a == cm.tail_tile_bands(
            layout.a_tiles, layout.a_total, layout.n_end_a
        )


def test_a_short_encode_recuts_the_audio_plan_to_what_was_frozen():
    """What the engine does when the material yields fewer audio latents than the
    band covers: it re-cuts the per-tile plan to the width it actually froze, with
    the SAME pure function. The recut plan must still be a valid tail plan — every
    write inside the narrower band, the whole of it covered, the last tile holding
    its end."""
    layout = cm.compute_chain_layout([169, 169, 169], FPS, kv=1, end_context_px=72)
    assert layout.n_tiles > 1
    assert cm.tail_tile_bands(
        layout.a_tiles, layout.a_total, layout.n_end_a
    ) == layout.end_tile_bands_a
    for frozen in (1, 7, layout.n_end_a - 1):
        recut = cm.tail_tile_bands(layout.a_tiles, layout.a_total, frozen)
        assert len(recut) == layout.n_tiles
        covered: set[int] = set()
        for (t, off), (as_, alen) in zip(recut, layout.a_tiles):
            assert 0 <= t <= alen
            if t == 0:
                continue
            assert 0 <= off - t < off <= frozen
            covered |= set(range(off - t, off))
        assert covered == set(range(frozen))
        assert recut[-1][1] == frozen and recut[-1][0] > 0


def test_the_video_and_audio_tile_plans_are_computed_independently():
    """Neither list is derivable from the other: a tile's audio count is not its
    video count, nor a fixed multiple of it, and a tile the video band misses
    entirely still gets its own audio entry. This is why the engine reads each
    from its own list."""
    layout = cm.compute_chain_layout([169, 169], FPS, end_context_px=72)
    tv = [t for t, _ in layout.end_tile_bands]
    ta = [t for t, _ in layout.end_tile_bands_a]
    assert tv == [0, 8, 5] and ta == [0, 65, 34]
    assert len(tv) == len(ta) == layout.n_tiles
    # The offsets differ too — each one indexes into ITS OWN band tensor.
    assert [off for _, off in layout.end_tile_bands] != [
        off for _, off in layout.end_tile_bands_a
    ]


def test_the_tile_plan_encodes_no_coupling_between_the_two_grids():
    """``t == 0`` on one grid does NOT imply ``t == 0`` on the other. No reachable
    layout is known to show it (the audio band's start time never precedes the
    video band's, so a tile that misses the video band misses the audio one too),
    but nothing in the geometry PROMISES that — and a promise is what an engine
    that nested the audio write inside ``if t > 0:`` would be relying on. Stated
    against the pure function directly, on the two grids' shapes."""
    video_tiles = [(0, 10), (6, 10)]      # 16 latents, band = the last 4
    audio_tiles = [(0, 40), (24, 40)]     # 64 latents, band = the last 40
    assert cm.tail_tile_bands(video_tiles, 16, 4) == [(0, -2), (4, 4)]
    assert cm.tail_tile_bands(audio_tiles, 64, 40) == [(16, 16), (40, 40)]


# ── the audio tail's mask value: hard at every strength ──────────────────────
@pytest.mark.parametrize("strength", [0.0, 0.5, 0.9, 1.0])
def test_the_audio_tail_is_hard_frozen_whatever_the_strength_is(strength):
    """``end_source.strength`` is a VIDEO knob. The engine passes ``1 - strength``
    as the video tail and 0.0 as the audio tail, and this is where those two are
    resolved — the pairing no combination of the older overrides could express."""
    mask_value = 0.5  # an ordinary carry-over seam (overlap_strength default)
    v_head, v_tail, a_head, a_tail = cm.freeze_mask_values(
        mask_value,
        tail_mask_value=1.0 - strength,
        audio_mask_value=None,
        audio_tail_mask_value=0.0,
    )
    assert v_head == mask_value
    assert v_tail == 1.0 - strength
    assert a_head == mask_value          # the head is still an ordinary seam
    assert a_tail == 0.0                 # ...and the band is still hard-frozen


@pytest.mark.parametrize("mask_value", [0.0, 0.3, 0.5, 1.0])
@pytest.mark.parametrize("tail", [None, 0.0, 0.25, 1.0])
@pytest.mark.parametrize("audio", [None, 0.0, 0.75])
def test_the_fourth_argument_is_inert_when_absent(mask_value, tail, audio):
    """Every pre-end-source-audio call site must be bit-identical: passing None
    (or nothing at all) for the fourth argument reproduces the three-argument
    result exactly, over the whole cross product of the older two."""
    old = cm.freeze_mask_values(mask_value, tail, audio)
    assert cm.freeze_mask_values(mask_value, tail, audio, None) == old
    assert cm.freeze_mask_values(
        mask_value, tail, audio, audio_tail_mask_value=None
    ) == old
    # ...and when it IS given it changes the audio tail and NOTHING else.
    v_head, v_tail, a_head, a_tail = cm.freeze_mask_values(mask_value, tail, audio, 0.0)
    assert (v_head, v_tail, a_head) == old[:3]
    assert a_tail == 0.0
