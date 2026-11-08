"""h3_chain_math: the MiniMax H3 chain's geometry (台帳 §1-106, Docs/MINIMAX_H3_DESIGN.md 裁定 84).

torch-free: runs in the app venv (``pytest tests/test_h3_chain_math.py``) and in
``.venv-engine-h3`` through ``scripts/run_h3_pytest.py``. The expected values
are VERIFICATION_LOG §183.8 (the measured chain) and hand-computed rows; the
sweeps port the measurement script's self-check 1-2
(Docs/Outputs-archive/h3-chain-2026-10-29/c_chain.py ``selfcheck``) to the
product's fixed overlap of 22 frames and to clips of different lengths.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

import h3_chain_math as M
from h3_chain_math import compute_h3_chain_layout

GRID = list(range(22, 482, 17))   # every 17n+5 count the API admits (22..481)


def _latent_frames(f: int) -> int:
    return (f - 5) // 17 * 5 + 2


def _round53(n: int) -> int:
    return round(Fraction(5 * n, 3))


# --------------------------------------------------------------------------- #
# expected values
# --------------------------------------------------------------------------- #

#: (clip frames, S, J, ka, F_total, T'_total, concatenated A, round(F_total*5/3), excess, signed shift)
EXPECTED = [
    # VERIFICATION_LOG §183.8, conditions C0-C3 (overlap 22, 124 x 3)
    ([124, 124, 124], [102, 204], [124, 226], [37, 37], 328, 97, 547, 547, 0, [0.0, 0.0]),
    # the shortest chain: clip 1 = 22 (all of it is the next clip's head), then 39s
    ([22, 39, 39], [0, 17], [22, 39], [37, 37], 56, 17, 93, 93, 0, [0.0, -1 / 3]),
    # the audio ends one latent SHORT of round(F_total*5/3) (excess -1)
    ([141, 107], [119], [141], [37], 226, 67, 376, 377, -1, [-1 / 3]),
]


@pytest.mark.parametrize("frames, S, J, ka, f_total, t_total, a_cat, a_exp, excess, shift", EXPECTED)
def test_expected_rows(frames, S, J, ka, f_total, t_total, a_cat, a_exp, excess, shift):
    lay = compute_h3_chain_layout(frames)
    assert lay.clip_frames == frames
    assert lay.clip_latent_frames == [_latent_frames(f) for f in frames]
    assert lay.clip_audio_latents == [_round53(f) for f in frames]
    assert (lay.overlap_frames, lay.overlap_latents, lay.overlap_strength) == (22, 7, 1.0)
    assert lay.seam_start_frames == S and lay.seam_new_frames == J and lay.audio_overlap_latents == ka
    assert (lay.total_frames, lay.total_latent_frames) == (f_total, t_total)
    assert (lay.total_audio_latents, lay.audio_latents_expected, lay.audio_latent_excess) == (a_cat, a_exp, excess)
    assert all(abs(x - y) < 1e-12 for x, y in zip(lay.audio_head_shift_latents, shift, strict=True))
    assert lay.audio_place_error_latents == [abs(x) for x in lay.audio_head_shift_latents]
    assert lay.audio_drift_flags == [False] + [x != 0.0 for x in shift]
    assert lay.segment_seam_junctions == [j - 1 for j in J]


def test_constants():
    assert M.H3_CHAIN_OVERLAP_FRAMES == 22 and M.H3_CHAIN_OVERLAP_LATENTS == 7
    assert M.H3_CHAIN_OVERLAP_STRENGTH == 1.0
    assert M.H3_CHAIN_MIN_NEXT_FRAMES == 39
    assert (M.H3_CHAIN_MIN_CLIPS, M.H3_CHAIN_MAX_CLIPS) == (2, 24)
    assert M.H3_CHAIN_MAX_TOTAL_FRAMES == 11544


def test_total_cap_is_the_ltx_chain_cap():
    models = pytest.importorskip("api.models")
    assert M.H3_CHAIN_MAX_TOTAL_FRAMES == models.MAX_CHAIN_TOTAL_PIXEL_FRAMES


def test_audio_rounding_is_the_engine_constants():
    constants = pytest.importorskip("engine_h3.constants")
    assert all(M._audio_latents(n) == constants.audio_latents(n) for n in range(0, 3000))
    assert all(M._latent_frames(f) == constants.latent_frames(f) for f in range(5, 3000, 17))
    assert M.H3_CHAIN_OVERLAP_LATENTS == constants.latent_frames(22)


def test_to_dict_carries_the_ltx_metadata_keys():
    lay = compute_h3_chain_layout([22, 39, 39])
    d = lay.to_dict()
    assert d["total_px"] == 56
    assert d["segment_seam_junctions"] == [21, 38]
    assert d["all_junctions"] == [20, 21, 22, 37, 38, 39]
    assert d["tile_seam_junctions"] == [] and d["n_tiles"] == 0
    assert d["overlap_frames"] == 22 and d["overlap_strength"] == 1.0
    assert d["audio_drift_flags"] == [False, False, True]
    assert d["audio_latent_excess"] == 0
    assert set(d) >= {
        "clip_frames", "clip_latent_frames", "clip_audio_latents", "overlap_latents", "seam_start_frames",
        "seam_new_frames", "audio_overlap_latents", "total_latent_frames", "total_audio_latents",
        "audio_latents_expected", "audio_place_error_latents", "audio_head_shift_latents",
    }
    import json

    assert json.loads(json.dumps(d)) == d     # JSON-safe as it is


def test_all_junctions_stay_inside_the_timeline():
    # The last junction's +1 neighbour may not reach the last frame (chain_math's rule).
    lay = compute_h3_chain_layout([124, 39])
    assert lay.total_frames == 141 and lay.segment_seam_junctions == [123]
    assert lay.all_junctions == [122, 123, 124]
    assert all(0 <= j < lay.total_frames - 1 for j in lay.all_junctions)


# --------------------------------------------------------------------------- #
# sweeps (c_chain.py selfcheck 1-2, at the product's overlap)
# --------------------------------------------------------------------------- #

def _reference(frames: list[int]) -> dict:
    """An independent restatement: S_i = sum(F[:i]) - 22 i; the audio concatenated latent by latent."""
    a_lat = [_round53(f) for f in frames]
    starts = [sum(frames[:i]) - 22 * i for i in range(1, len(frames))]
    concat = a_lat[0]
    ka = []
    for i, s in enumerate(starts, start=1):
        ka.append(concat - _round53(s))
        concat += a_lat[i] - ka[-1]
    return {"S": starts, "ka": ka, "concat": concat}


def _invariants(frames: list[int]) -> list[str]:
    lay = compute_h3_chain_layout(frames)
    ref = _reference(frames)
    bad = []
    total = sum(frames) - 22 * (len(frames) - 1)
    checks = {
        "S": lay.seam_start_frames == ref["S"],
        "ka": lay.audio_overlap_latents == ref["ka"],
        "concat": lay.total_audio_latents == ref["concat"],
        "J-S": all(j - s == 22 for s, j in zip(lay.seam_start_frames, lay.seam_new_frames, strict=True)),
        "total": lay.total_frames == total,
        "T'": lay.total_latent_frames == _latent_frames(total),
        "ka range": all(0 <= x <= a for x, a in zip(lay.audio_overlap_latents, lay.clip_audio_latents[1:])),
        "excess": abs(lay.audio_latent_excess) <= 1,
        "shift": all(abs(x) <= 1 / 3 + 1e-12 for x in lay.audio_head_shift_latents),
        "flags": lay.audio_drift_flags == [False] + [s % 3 != 0 for s in lay.seam_start_frames],
        "concat sum": lay.total_audio_latents
        == lay.clip_audio_latents[0] + sum(a - x for a, x in zip(lay.clip_audio_latents[1:], lay.audio_overlap_latents)),
    }
    bad += [name for name, good in checks.items() if not good]
    return bad


def test_sweep_equal_lengths():
    failures, excess = [], {}
    for f in GRID:
        for n in range(2, 7):
            if f < 39:
                with pytest.raises(ValueError):
                    compute_h3_chain_layout([f] * n)
                continue
            bad = _invariants([f] * n)
            if bad:
                failures.append((f, n, bad))
            e = compute_h3_chain_layout([f] * n).audio_latent_excess
            excess[e] = excess.get(e, 0) + 1
    assert not failures, failures[:5]
    assert set(excess) <= {-1, 0, 1}


def test_sweep_mixed_lengths():
    failures, excess = [], set()
    for f0 in GRID:
        for f1 in GRID[1:]:
            for frames in ([f0, f1], [f0, f1, 39], [f0, 39, f1]):
                bad = _invariants(frames)
                if bad:
                    failures.append((frames, bad))
                excess.add(compute_h3_chain_layout(frames).audio_latent_excess)
    assert not failures, failures[:5]
    assert excess == {-1, 0, 1}       # both rounding directions occur and are accepted


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("frames, match", [
    ([124], r"^chain needs 2\.\.24 clips \(got 1\)$"),
    ([124] * 25, r"^chain needs 2\.\.24 clips \(got 25\)$"),
    ([], r"^chain needs 2\.\.24 clips \(got 0\)$"),
    ([123, 124], r"^clips\[0\]: num_frames must be 17n\+5$"),
    ([124, 125], r"^clips\[1\]: num_frames must be 17n\+5$"),
    ([5, 124], r"^clips\[0\]: num_frames must be at least 22$"),
    ([124, 22], r"^clips\[1\]: num_frames must be at least 39 \(overlap 22 \+ one grid step\)$"),
    ([124, 39, 22], r"^clips\[2\]: num_frames must be at least 39 "),
    ([124.0, 124], r"^clips\[0\]: num_frames must be an integer"),
    ([True, 124], r"^clips\[0\]: num_frames must be an integer"),
    ([3915, 3915, 3915], r"^total frames 11701 exceed the cap 11544 \(reduce clip count or lengths\)$"),
    ([5768, 5802], r"^total frames 11548 exceed the cap 11544 "),
])
def test_refusals(frames, match):
    with pytest.raises(ValueError, match=match):
        compute_h3_chain_layout(frames)


def test_the_cap_is_inclusive_of_the_largest_grid_total():
    # 11,531 = 17 * 678 + 5 is the longest 17n+5 timeline under the cap.
    lay = compute_h3_chain_layout([5768, 5785])
    assert lay.total_frames == 11531
    assert compute_h3_chain_layout([481] * 24).total_frames == 481 * 24 - 22 * 23


# --------------------------------------------------------------------------- #
# video-to-video: an input video's 22-frame tail as clip 0's fixed head (台帳 §1-121)
# --------------------------------------------------------------------------- #

V2V_KEYS = ["source_context_px", "n_ctx_v", "n_ctx_a", "trim_px", "v2v_context_junction_px", "new_frames_px"]


def test_v2v_adds_no_frame_and_only_the_v2v_block():
    for frames in ([56, 39], [124, 124, 124], [141, 107], [39, 39, 39]):
        plain = compute_h3_chain_layout(frames)
        v2v = compute_h3_chain_layout(frames, source_video=True)
        d = v2v.to_dict()
        block = d.pop("v2v")
        assert d == plain.to_dict()                        # the same geometry, the same key order
        assert list(block) == V2V_KEYS
        assert block == {"source_context_px": 22, "n_ctx_v": 7, "n_ctx_a": 37, "trim_px": 22,
                         "v2v_context_junction_px": 21, "new_frames_px": plain.total_frames - 22}
        assert v2v.new_frames_px == plain.total_frames - 22


def test_plain_layout_has_no_v2v_block():
    lay = compute_h3_chain_layout([124, 39])
    assert "v2v" not in lay.to_dict() and list(lay.to_dict())[-1] == "audio_drift_flags"
    assert (lay.source_context_px, lay.n_ctx_v, lay.n_ctx_a, lay.trim_px, lay.v2v_context_junction_px) == (
        None, 0, 0, 0, None)
    assert lay.new_frames_px == lay.total_frames == 141


def test_v2v_single_clip():
    lay = compute_h3_chain_layout([39], source_video=True)
    assert (lay.total_frames, lay.total_latent_frames, lay.total_audio_latents) == (39, 12, 65)
    assert lay.seam_start_frames == [] and lay.segment_seam_junctions == [] and lay.all_junctions == []
    assert lay.audio_drift_flags == [False]
    assert lay.to_dict()["v2v"]["new_frames_px"] == 17
    with pytest.raises(ValueError, match=r"^chain needs 2\.\.24 clips \(got 1\)$"):
        compute_h3_chain_layout([39])                      # without a source: still refused


@pytest.mark.parametrize("frames, match", [
    ([22, 39], r"^clips\[0\]: num_frames must be at least 39 \(source context 22 \+ one grid step\)$"),
    ([22], r"^clips\[0\]: num_frames must be at least 39 \(source context 22 "),
    ([], r"^chain needs 1\.\.24 clips \(got 0\)$"),
    ([124] * 25, r"^chain needs 1\.\.24 clips \(got 25\)$"),
    ([39, 22], r"^clips\[1\]: num_frames must be at least 39 \(overlap 22 \+ one grid step\)$"),
    ([40, 39], r"^clips\[0\]: num_frames must be 17n\+5$"),
])
def test_v2v_refusals(frames, match):
    with pytest.raises(ValueError, match=match):
        compute_h3_chain_layout(frames, source_video=True)


def test_v2v_clip_0_at_39_is_accepted():
    assert compute_h3_chain_layout([39, 39], source_video=True).total_frames == 56
    assert compute_h3_chain_layout([22, 39]).total_frames == 39   # without a source 22 stays fine


# --------------------------------------------------------------------------- #
# retake: the one clip is an input video's window, its head and tail fixed (台帳 §1-123)
# --------------------------------------------------------------------------- #

RETAKE_KEYS = ["window_px", "head_px", "tail_px", "n_head_v", "n_tail_v", "n_head_a", "n_tail_a", "free_middle_px"]


def test_retake_min_frames_constant():
    # 17 * 4 + 5: T' = 22 latents, 22 - 7 - 7 = 8 left free.
    assert M.H3_RETAKE_MIN_FRAMES == 73 == 17 * 4 + 5
    assert _latent_frames(M.H3_RETAKE_MIN_FRAMES) - 2 * M.H3_CHAIN_OVERLAP_LATENTS == 8
    assert _latent_frames(M.H3_RETAKE_MIN_FRAMES - 17) - 2 * M.H3_CHAIN_OVERLAP_LATENTS == 3


#: (F, T', A, ts_a = ceil((F-22)*5/3), te_a nominal = min(A, round(F*4000/3)//800), the "retake" block)
RETAKE_EXPECTED = [
    # T' = 22, A = round(121.67) = 122, ts_a = ceil(51*5/3) = 85,
    # round(97333.3) = 97333 // 800 = 121 (latent 121 holds the right zero padding)
    (73, 22, 122, 85, 121, {"window_px": 73, "head_px": 22, "tail_px": 22, "n_head_v": 7, "n_tail_v": 7,
                            "n_head_a": 36, "n_tail_a": 36, "free_middle_px": [22, 51]}),
    # T' = 37, A = round(206.67) = 207, ts_a = ceil(102*5/3) = 170, 165333 // 800 = 206
    (124, 37, 207, 170, 206, {"window_px": 124, "head_px": 22, "tail_px": 22, "n_head_v": 7, "n_tail_v": 7,
                              "n_head_a": 36, "n_tail_a": 36, "free_middle_px": [22, 102]}),
    # T' = 142, A = round(801.67) = 802, ts_a = ceil(459*5/3) = 765, 641333 // 800 = 801
    (481, 142, 802, 765, 801, {"window_px": 481, "head_px": 22, "tail_px": 22, "n_head_v": 7, "n_tail_v": 7,
                               "n_head_a": 36, "n_tail_a": 36, "free_middle_px": [22, 459]}),
]


@pytest.mark.parametrize("f, t_lat, a_lat, ts_a, te_a, block", RETAKE_EXPECTED)
def test_retake_expected_rows(f, t_lat, a_lat, ts_a, te_a, block):
    lay = compute_h3_chain_layout([f], retake=True)
    assert lay.clip_latent_frames == [t_lat] and lay.clip_audio_latents == [a_lat]
    assert lay.total_frames == f and lay.to_dict()["total_px"] == f
    assert (lay.retake_audio_tail_start, lay.retake_audio_tail_start + lay.retake_n_tail_a) == (ts_a, te_a)
    d = lay.to_dict()
    assert list(d["retake"]) == RETAKE_KEYS
    assert d["retake"] == block
    # the free video middle, in latents: T' - 14
    assert t_lat - d["retake"]["n_head_v"] - d["retake"]["n_tail_v"] == t_lat - 14


def test_retake_audio_tail_end_with_and_without_a_padded_latent():
    # F = 73: 73 * 5/3 = 121.67, A = 122 but the samples cover only 121 whole
    # latents, so one latent (121) holds the right zero padding and stays free.
    # te_a (the tail's exclusive end) is ts_a + n_tail_a.
    lay = compute_h3_chain_layout([73], retake=True)
    assert (lay.clip_audio_latents[0], lay.retake_audio_tail_start + lay.retake_n_tail_a) == (122, 121)
    # F = 90 (17*5+5, a multiple of 3): 90 * 4000/3 = 120,000 = 800 * 150 = 800 * A exactly,
    # no padded latent, the tail runs to A.
    assert 90 * 4000 % 3 == 0 and 90 * 4000 // 3 % 800 == 0
    lay = compute_h3_chain_layout([90], retake=True)
    assert (lay.clip_audio_latents[0], lay.retake_audio_tail_start + lay.retake_n_tail_a) == (150, 150)
    assert lay.retake_audio_tail_start == 114 and lay.to_dict()["retake"]["n_tail_a"] == 36
    # Over the whole grid: te_a is A or A - 1, and A - 1 exactly when 5F/3 has a fraction of 2/3 (F = 1 mod 3).
    for f in GRID:
        if f < M.H3_RETAKE_MIN_FRAMES:
            continue
        lay = compute_h3_chain_layout([f], retake=True)
        a = lay.clip_audio_latents[0]
        te = min(a, round(Fraction(f * 4000, 3)) // 800)
        assert te == a - (1 if f % 3 == 1 else 0)
        assert lay.retake_audio_tail_start == -(-(f - 22) * 5 // 3)
        assert lay.retake_n_tail_a == max(0, te - lay.retake_audio_tail_start)


def test_retake_is_a_one_clip_chain_plus_the_retake_block():
    for f in (73, 124, 481):
        d = compute_h3_chain_layout([f], retake=True).to_dict()
        d.pop("retake")
        # The one-clip geometry is what a single-clip V2V layout carries without its block.
        ref = compute_h3_chain_layout([f], source_video=True).to_dict()
        ref.pop("v2v")
        assert d == ref and list(d) == list(ref)
        assert d["segment_seam_junctions"] == [] and d["all_junctions"] == []


def test_plain_and_v2v_layouts_have_no_retake_block():
    for frames, sv in (([73, 56], False), ([124, 124, 124], False), ([124], True), ([56, 39], True)):
        lay = compute_h3_chain_layout(frames, source_video=sv)
        assert "retake" not in lay.to_dict()
        assert lay.retake_window_px is None
        assert (lay.retake_audio_tail_start, lay.retake_n_tail_a) == (0, 0)


@pytest.mark.parametrize("frames, match", [
    ([56], r"^clips\[0\]: retake window \(56\) must be at least 73 frames \(head 22 \+ tail 22 \+ a free middle\)$"),
    ([22], r"^clips\[0\]: retake window \(22\) must be at least 73 frames "),
    ([74], r"^clips\[0\]: num_frames must be 17n\+5$"),
    ([124, 124], r"^retake needs exactly 1 clip \(got 2\)$"),
    ([], r"^chain needs 1\.\.24 clips \(got 0\)$"),
])
def test_retake_refusals(frames, match):
    with pytest.raises(ValueError, match=match):
        compute_h3_chain_layout(frames, retake=True)


def test_retake_with_source_video_is_refused():
    with pytest.raises(ValueError, match=r"^retake cannot be combined with an input video \(source_video\)$"):
        compute_h3_chain_layout([124], source_video=True, retake=True)
