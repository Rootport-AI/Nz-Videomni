"""Pure-Python geometry tests for the clip-wise IC-LoRA reference helpers (§1-15).

``video_segment_windows`` is the SINGLE SOURCE OF TRUTH for "which pixel-frame
slice of the one long uploaded reference belongs to stage-1 segment i", shared by
the engine (which slices the decoded reference) and the app (preflight / metadata).
``chain_stage1_tokens`` + ``CHAIN_STAGE1_COMFORT_TOKEN_BUDGET`` are the stage-1
load estimate the frontend mirrors in ``webui/src/shell/tokenBudget.ts``.

These tests pin both to the geometry ``compute_chain_layout`` resolves — no
torch, no GPU — matching the style of ``tests/test_chain_math_a2v.py``.
"""

from __future__ import annotations

import chain_math


FPS = 24.0


# ── video_segment_windows: single clip / clip 0 ──────────────────────────────
def test_video_segment_windows_single_clip_is_whole_clip():
    """n=1 degenerates to ``[(0, clip_frames[0])]`` — byte-identical framing to
    the single-clip ``/generate`` reference path."""
    for clip_frames in ([121], [481], [249], [9]):
        layout = chain_math.compute_chain_layout(clip_frames, FPS, kv=1)
        assert chain_math.video_segment_windows(layout) == [(0, clip_frames[0])]


def test_video_segment_windows_clip0_matches_single_clip_path():
    """Clip 0's window is ``(0, clip_frames[0])`` no matter how many clips follow
    — the head of a chain reads the reference exactly like a single clip does."""
    for clip_frames in ([121, 121], [481, 257, 121], [73] * 24):
        layout = chain_math.compute_chain_layout(clip_frames, FPS)
        assert chain_math.video_segment_windows(layout)[0] == (0, clip_frames[0])


# ── video_segment_windows: overlap / coverage ────────────────────────────────
def test_video_segment_windows_overlap_is_8kv_minus_7():
    """Adjacent windows overlap by exactly ``8 * K_v - 7`` pixel frames — the
    pixel span the stage-1 latent carry-over covers, which is what makes both
    neighbours see the same reference footage across a seam."""
    for clip_frames, kv in [
        ([121, 121], 3),
        ([481, 257, 121], 3),
        ([145, 73], 2),
        ([121] * 4, 5),
    ]:
        layout = chain_math.compute_chain_layout(clip_frames, FPS, kv=kv)
        windows = chain_math.video_segment_windows(layout)
        for j in range(len(windows) - 1):
            prev_end = windows[j][0] + windows[j][1]
            nxt_start = windows[j + 1][0]
            assert prev_end - nxt_start == 8 * kv - 7, (clip_frames, kv, j)


def test_video_segment_windows_last_window_ends_at_total_px():
    """The last window's final frame is ``total_px - 1``: the windows need
    EXACTLY ``total_px`` reference frames, no more and no less."""
    for clip_frames, kv in [
        ([121], 3),
        ([121, 121], 3),
        ([481, 257, 121], 3),
        ([145, 73, 97, 121], 2),
        ([73] * 24, 3),
    ]:
        layout = chain_math.compute_chain_layout(clip_frames, FPS, kv=kv)
        start, length = chain_math.video_segment_windows(layout)[-1]
        assert start + length - 1 == layout.total_px - 1, (clip_frames, kv)


def test_video_segment_windows_uneven_clips():
    """Clips of wildly different lengths: the windows are pinned literally so a
    regression in the ``sum(seg_latent[:i]) - i*kv`` accumulation is visible."""
    layout = chain_math.compute_chain_layout([481, 257, 9], FPS, kv=1)
    # s = [0, 61, 61+33-1 = 93] latent -> 8*s = [0, 488-8=480, 744-8=736] px.
    assert chain_math.video_segment_windows(layout) == [
        (0, 481), (480, 257), (736, 9)
    ]
    assert layout.total_px == 745

    layout = chain_math.compute_chain_layout([481, 257, 25], FPS, kv=3)
    assert chain_math.video_segment_windows(layout) == [
        (0, 481), (464, 257), (704, 25)
    ]
    assert layout.total_px == 729


# ── video_segment_windows: property sweep ────────────────────────────────────
def test_video_segment_windows_properties_over_kv_and_clip_counts():
    """K_v 1..8 x clip counts {1, 2, 3, 24}: starts strictly increase, every
    window is its own clip's full length, every window stays inside
    ``[0, total_px)``, and every overlap equals ``8*K_v - 7``."""
    for kv in range(1, 9):
        for n_clips in (1, 2, 3, 24):
            clip_frames = [121] * n_clips
            layout = chain_math.compute_chain_layout(clip_frames, FPS, kv=kv)
            windows = chain_math.video_segment_windows(layout)
            ctx = (kv, n_clips)

            assert len(windows) == n_clips, ctx
            assert [wl for _, wl in windows] == clip_frames, ctx
            assert windows[0][0] == 0, ctx
            assert windows[-1][0] + windows[-1][1] == layout.total_px, ctx

            for j in range(len(windows) - 1):
                assert windows[j + 1][0] > windows[j][0], (ctx, j)
                prev_end = windows[j][0] + windows[j][1]
                assert prev_end - windows[j + 1][0] == 8 * kv - 7, (ctx, j)

            for ws, wl in windows:
                assert 0 <= ws, ctx
                assert ws + wl <= layout.total_px, ctx


def test_video_segment_windows_starts_match_seam_junctions():
    """Cross-check against the already-trusted ``segment_seam_junctions``: those
    report segment i's first NEW latent (``s_i + K_v``) as a pixel index, so
    ``window[i].start == px_from_v_latent(junction[i-1] latent) - 8*kv`` — i.e.
    the window opens exactly K_v latents (8*K_v px) before the seam's new
    content, which is the carry band."""
    for clip_frames, kv in [([121, 121], 3), ([481, 257, 121], 3), ([145, 73], 2)]:
        layout = chain_math.compute_chain_layout(clip_frames, FPS, kv=kv)
        windows = chain_math.video_segment_windows(layout)
        for i in range(1, len(clip_frames)):
            # segment_seam_junctions[i-1] = px_from_v_latent(s_i + kv) - 1
            #                             = 8*(s_i + kv - 1) + 1 - 1 = 8*s_i + 8*kv - 8
            expected_start = layout.segment_seam_junctions[i - 1] - 8 * kv + 8
            assert windows[i][0] == expected_start, (clip_frames, kv, i)


# ── chain_stage1_tokens ──────────────────────────────────────────────────────
def test_chain_stage1_tokens_no_reference():
    """Without a reference: ``(w//2//32) * (h//2//32) * v_latent``.
    1152x1536 -> (576//32) * (768//32) = 18*24 = 432 spatial patches."""
    assert chain_math.chain_stage1_tokens(1152, 1536, 46) == 432 * 46 == 19_872
    assert chain_math.chain_stage1_tokens(1152, 1536, 46, None) == 19_872
    assert chain_math.chain_stage1_tokens(1152, 1536, 61) == 432 * 61 == 26_352


def test_chain_stage1_tokens_control_reference_scale2():
    """scale=2 (union-control) adds a QUARTER of the spatial patches:
    ((576//2)//32) * ((768//2)//32) = 9*12 = 108 = 432/4, so x1.25 overall.
    These are the two research-note anchor points (§6): 361 px frames ->
    v_latent 46 -> 24,840, and 481 px frames -> v_latent 61 -> 32,940."""
    assert chain_math.v_latent_frames(361) == 46
    assert chain_math.v_latent_frames(481) == 61
    assert chain_math.chain_stage1_tokens(1152, 1536, 46, 2) == 24_840
    assert chain_math.chain_stage1_tokens(1152, 1536, 61, 2) == 32_940
    # x1.25 of the un-referenced cost, exactly.
    assert chain_math.chain_stage1_tokens(1152, 1536, 61, 2) == int(432 * 61 * 1.25)


def test_chain_stage1_tokens_deblur_reference_scale1_doubles():
    """scale=1 (deblur) patchifies the reference at the SAME size as the clip,
    so it exactly doubles the segment's tokens."""
    for width, height, v_latent in [(1152, 1536, 46), (768, 512, 22), (1024, 1024, 61)]:
        bare = chain_math.chain_stage1_tokens(width, height, v_latent)
        assert chain_math.chain_stage1_tokens(width, height, v_latent, 1) == 2 * bare


def test_chain_stage1_tokens_division_order_is_floor_at_each_step():
    """The floors are applied in order (half, then //32, then //ref_scale, then
    //32) and NOT simplified on a "multiple of 128" assumption — an off-grid
    resolution must truncate, not round."""
    # 1150//2 = 575; 575//32 = 17 (not 1150/64 = 17.97 rounded).
    assert chain_math.chain_stage1_tokens(1150, 1536, 10) == 17 * 24 * 10
    # ref: (575//2)//32 = 287//32 = 8, ((768//2))//32 = 12.
    assert chain_math.chain_stage1_tokens(1150, 1536, 10, 2) == (17 * 24 + 8 * 12) * 10


def test_chain_stage1_comfort_budget_sits_between_the_two_anchors():
    """The provisional 25,000 budget brackets the research note's two measured
    statements: the 361-frame ceiling is just under it and the 328-patch /
    481-frame configuration is just over it."""
    assert chain_math.CHAIN_STAGE1_COMFORT_TOKEN_BUDGET == 25_000
    assert chain_math.chain_stage1_tokens(1152, 1536, 46, 2) < 25_000
    # 328 stage-1 spatial patches x 61 latent frames x 1.25 = 25,010.
    assert int(328 * 61 * 1.25) > chain_math.CHAIN_STAGE1_COMFORT_TOKEN_BUDGET


# ── reference_encode_tokens / REFERENCE_ENCODE_TILE_TOKEN_BUDGET (§3-76) ─────
def test_reference_encode_tokens_matches_stage1_scale2_identity_at_pct128():
    """At %128 OUTPUT resolutions (both the single-shot and chain reference
    APIs force multiples of 128), ``chain_stage1_tokens(w, h, v, ref_scale=2)``
    is EXACTLY 5x the reference's own ``reference_encode_tokens(w//4, h//4, F)``
    — the two integer-division chains (half, //32, //2, //32) collapse to the
    same spatial patch count when w/h are multiples of 128. That is why
    ``REFERENCE_ENCODE_TILE_TOKEN_BUDGET = 25_000 // 5 = 5_000`` cuts the SAME
    job set as the Chained screen's stage-1 comfort banner: both budgets agree
    on which side of the line each (w, h, pixel_frames) triple falls."""
    for width, height, pixel_frames in [
        (1152, 1536, 361),
        (1152, 1536, 481),
        (768, 512, 121),
        (1024, 1024, 249),
        (1920, 1152, 481),
    ]:
        v = chain_math.v_latent_frames(pixel_frames)
        stage1 = chain_math.chain_stage1_tokens(width, height, v, 2)
        ref = chain_math.reference_encode_tokens(width // 4, height // 4, pixel_frames)
        assert stage1 == 5 * ref, (width, height, pixel_frames)
        assert (
            stage1 < chain_math.CHAIN_STAGE1_COMFORT_TOKEN_BUDGET
        ) == (
            ref < chain_math.REFERENCE_ENCODE_TILE_TOKEN_BUDGET
        ), (width, height, pixel_frames)

    # Non-%128 counter-example (1152x704 -- 704 is not a multiple of 128): the
    # identity is NOT claimed off-grid and does in fact break, because the
    # half/half-of-half division chains stop collapsing to the same patch
    # count once a dimension falls off the 128 grid.
    v = chain_math.v_latent_frames(361)
    assert chain_math.chain_stage1_tokens(1152, 704, v, 2) != 5 * chain_math.reference_encode_tokens(
        1152 // 4, 704 // 4, 361
    )


def test_reference_encode_tile_token_budget_knee_values():
    """The knee itself: 361 pixel frames at 288x384 (the reference's own
    resolution after ``resize_and_center_crop``) sits just under 5,000 -- one
    shot, byte-identical to already-shipped output; 369 sits just over --
    tiled. 241/481 are extra anchor points from the real-hardware gate."""
    assert chain_math.REFERENCE_ENCODE_TILE_TOKEN_BUDGET == 5_000
    assert chain_math.reference_encode_tokens(288, 384, 361) == 4_968
    assert chain_math.reference_encode_tokens(288, 384, 369) == 5_076
    assert chain_math.reference_encode_tokens(288, 384, 241) == 3_348
    assert chain_math.reference_encode_tokens(288, 384, 481) == 6_588


def test_reference_encode_tokens_short_window_floors_not_truncates_to_zero():
    """A short reference window still contributes its full spatial patch
    count -- ``v_latent_frames`` floors the LATENT count, never the spatial
    one, so even a single pixel frame (v_latent == 1) yields 108 tokens, not
    0."""
    assert chain_math.reference_encode_tokens(288, 384, 360) == 108 * 45
    assert chain_math.reference_encode_tokens(288, 384, 1) == 108
