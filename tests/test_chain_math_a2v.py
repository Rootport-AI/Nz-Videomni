"""Pure-Python geometry tests for the audio-to-video (A2V) chain_math helpers.

``audio_latents_required`` and ``audio_segment_windows`` are the SINGLE SOURCE OF
TRUTH shared by the app (preflight: is the uploaded audio long enough?) and the
engine (how many audio-latent frames to slice off the VAE-encoded upload, and
which global window each stage-1 segment hard-freezes). These tests pin them to
the same geometry ``compute_chain_layout`` resolves — no torch, no GPU, matching
the ``test_chain_carry_always_fits_current_segment`` style in ``test_chain.py``.
"""

from __future__ import annotations

import chain_math


# ── audio_latents_required ───────────────────────────────────────────────────
def test_audio_latents_required_equals_layout_a_total():
    """The SoT must equal ``compute_chain_layout(...).a_total`` (the value the
    engine actually slices off the encoded upload) for n=1, n=2, and n=3 and a
    couple of fps, so app-side preflight and the engine agree exactly."""
    configs = [
        ([121], 24.0),
        ([121, 121], 24.0),
        ([145, 73], 24.0),
        ([25, 25, 25], 24.0),
        ([121], 30.0),
        ([73, 121, 97], 60.0),
    ]
    for clip_frames, fps in configs:
        layout = chain_math.compute_chain_layout(clip_frames, fps)
        assert chain_math.audio_latents_required(clip_frames, fps) == layout.a_total, (
            clip_frames, fps
        )


def test_audio_latents_required_matches_a_frames_for_px_total():
    """It is exactly ``a_frames_for_px(total_px)`` — the audio-latent count for
    the assembled pixel timeline (overlaps already folded into total_px)."""
    for clip_frames, fps in [([121], 24.0), ([145, 73], 24.0), ([25, 25, 25], 30.0)]:
        layout = chain_math.compute_chain_layout(clip_frames, fps)
        assert chain_math.audio_latents_required(clip_frames, fps) == (
            chain_math.a_frames_for_px(layout.total_px, fps)
        )


def test_audio_latents_required_honours_kv():
    """Overlap K_v shortens the assembled timeline -> fewer required audio
    latents; the SoT tracks the same kv the engine uses."""
    clip_frames, fps = [121, 121], 24.0
    a1 = chain_math.audio_latents_required(clip_frames, fps, kv=1)
    a3 = chain_math.audio_latents_required(clip_frames, fps, kv=3)
    assert a3 < a1
    assert a3 == chain_math.compute_chain_layout(clip_frames, fps, kv=3).a_total


def test_audio_latents_required_fractional_fps():
    """Fractional fps (23.976 / NTSC 30000/1001) rounds via a_frames_for_px with
    no drift vs the resolved layout."""
    for fps in (23.976, 30000.0 / 1001.0, 29.97):
        clip_frames = [121]
        layout = chain_math.compute_chain_layout(clip_frames, fps)
        assert chain_math.audio_latents_required(clip_frames, fps) == layout.a_total


# ── audio_segment_windows ────────────────────────────────────────────────────
def test_audio_segment_windows_single_clip_is_full_timeline():
    """n=1 -> one window covering the whole audio timeline (design: [(0, a_total)])."""
    for clip_frames, fps in [([121], 24.0), ([249], 30.0), ([121], 23.976)]:
        layout = chain_math.compute_chain_layout(clip_frames, fps)
        windows = chain_math.audio_segment_windows(layout)
        assert windows == [(0, layout.a_total)]


def test_audio_segment_windows_lengths_match_seg_audio():
    """Every window's length is that segment's stage-1 audio-latent count."""
    for clip_frames, fps in [([121, 121], 24.0), ([145, 73], 24.0), ([25, 25, 25], 30.0)]:
        layout = chain_math.compute_chain_layout(clip_frames, fps)
        windows = chain_math.audio_segment_windows(layout)
        assert [wl for _, wl in windows] == layout.seg_audio


def test_audio_segment_windows_overlaps_and_coverage():
    """Windows start monotonically, adjacent windows overlap by exactly
    ``ka_list[j]`` (the per-join audio crossfade), the first starts at 0, and the
    last ends exactly at a_total — the whole timeline is covered."""
    for clip_frames, fps in [
        ([121, 121], 24.0),
        ([145, 73], 24.0),
        ([73, 145], 24.0),
        ([25, 25, 25], 24.0),
        ([73, 121, 97], 60.0),
    ]:
        layout = chain_math.compute_chain_layout(clip_frames, fps)
        windows = chain_math.audio_segment_windows(layout)
        assert windows[0][0] == 0
        assert windows[-1][0] + windows[-1][1] == layout.a_total
        for j in range(len(windows) - 1):
            prev_end = windows[j][0] + windows[j][1]
            nxt_start = windows[j + 1][0]
            assert nxt_start < prev_end, (clip_frames, fps, j)  # genuine overlap
            assert prev_end - nxt_start == layout.ka_list[j], (clip_frames, fps, j)


def test_audio_segment_windows_stay_in_bounds():
    """No window runs off either end of the [0, a_total) audio timeline."""
    for clip_frames, fps in [([145, 73], 24.0), ([25, 33, 33, 33], 60.0)]:
        layout = chain_math.compute_chain_layout(clip_frames, fps)
        for ws, wl in chain_math.audio_segment_windows(layout):
            assert 0 <= ws
            assert ws + wl <= layout.a_total
