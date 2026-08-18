"""Stage-2 window per-job opt-in (§3-57 sweep outcome) + the §1-14 comfortable
attention-token budget.

Owner decision 2026-08-09: the default stays the frozen 22/18 window; the
19/12 window ("high_resolution" — a shorter window with a WIDER 7-frame
overlap) is unlocked as a manual opt-in for high resolutions. The old
experiment-only env-var knobs (``LTX_STAGE2_V_TILE`` / ``LTX_STAGE2_KT_V``) are
gone; ``chain_math.STAGE2_WINDOW_PRESETS`` is now the single source of truth.

Every expected value below is either (a) arithmetic reproduced independently in
the test, or (b) a number READ OFF the real sweep runs in
``outputs/stage2_window_sweep/`` — cited per-assertion.
"""

import pytest

from pydantic import ValidationError

import chain_math
from api.models import GenerateChainRequest


# ── preset table ─────────────────────────────────────────────────────────────
def test_preset_table_contents_and_derived_overlap():
    """The three presets and their DERIVED overlap (kt_v = v_tile - v_adv)."""
    assert chain_math.STAGE2_WINDOW_PRESETS == {
        "standard": (22, 18),
        "high_resolution": (19, 12),
        "full_length": (61, 61),
    }
    assert chain_math.STAGE2_WINDOW_DEFAULT == "standard"
    kt = {name: v_tile - v_adv
          for name, (v_tile, v_adv) in chain_math.STAGE2_WINDOW_PRESETS.items()}
    # standard keeps the S2-spike 4-frame overlap; high_resolution is the
    # follow-up run's wider 7-frame overlap ("のり代7"); full_length has NO
    # overlap because it has no seam to overlap across (§1-19).
    assert kt == {"standard": 4, "high_resolution": 7, "full_length": 0}


def test_every_multi_tile_preset_advance_is_a_multiple_of_three():
    """The 24fps video/audio advance rounding is only exact when the stage-2
    advance is a multiple of 3 — a guard for any FUTURE preset added here.

    The rule is about the ROUNDING AGREEING ACROSS A TILE SEAM, so it applies to
    presets that can actually produce two or more tiles (kt_v > 0). A window
    with kt_v == 0 degenerates to a single tile — it can never have a seam — so
    it is exempt; ``test_full_length_kt_a_is_negative_and_inert`` records what
    its (unused) audio numbers come out as."""
    exempt: set[str] = set()
    for name, (v_tile, v_adv) in chain_math.STAGE2_WINDOW_PRESETS.items():
        if v_tile - v_adv == 0:
            exempt.add(name)
            continue
        assert v_adv % 3 == 0, f"preset {name!r} advance {v_adv} is not a multiple of 3"
    # Pin WHICH presets take the exemption, so it cannot quietly widen: only
    # full_length, and only because its advance equals its tile size.
    assert exempt == {"full_length"}
    v_tile, v_adv = chain_math.STAGE2_WINDOW_PRESETS["full_length"]
    assert v_adv == v_tile


def test_module_constants_are_the_standard_preset():
    assert (chain_math.STAGE2_V_TILE, chain_math.STAGE2_V_ADV) == \
        chain_math.STAGE2_WINDOW_PRESETS["standard"]
    assert chain_math.STAGE2_KT_V == 4


def test_compute_chain_layout_defaults_are_bound_to_the_standard_preset():
    """ORDERING GUARD (chain_math's own comment): ``compute_chain_layout``'s
    ``v_tile``/``v_adv`` default arguments are bound at `def`-EVALUATION time.
    If the preset table / derived constants are ever moved BELOW that `def`,
    the defaults silently stop tracking the table — and every "omitted window"
    path would quietly change geometry. This pins the binding itself."""
    import inspect

    params = inspect.signature(chain_math.compute_chain_layout).parameters
    assert (params["v_tile"].default, params["v_adv"].default) == \
        chain_math.STAGE2_WINDOW_PRESETS["standard"]


# ── resolve_stage2_window ────────────────────────────────────────────────────
@pytest.mark.parametrize("name", [None, ""])
def test_resolve_stage2_window_absent_is_the_default(name):
    assert chain_math.resolve_stage2_window(name) == (22, 18)


def test_resolve_stage2_window_known_names():
    assert chain_math.resolve_stage2_window("standard") == (22, 18)
    assert chain_math.resolve_stage2_window("high_resolution") == (19, 12)


def test_resolve_stage2_window_unknown_raises():
    """A typo must fail loudly: the window changes the OUTPUT, so a silent
    fallback to the default would produce a job whose recorded window is a lie."""
    with pytest.raises(ValueError, match="unknown stage2_window"):
        chain_math.resolve_stage2_window("hi_res")


# ── §1-14 comfortable attention-token budget ────────────────────────────────
# EXPECTED-VALUE TABLE. The same table is mirrored verbatim in the WebUI
# (webui/src/shell/tokenBudget.test.ts) — if one side is edited without the
# other, the two stop agreeing about which resolutions warn.
#   tokens = (width // 32) * (height // 32) * v_tile
# The first four rows are the §3-57 spill arm's own measured configurations
# (outputs/stage2_window_sweep/SWEEP_RESULTS.md §2b), which is where the
# 40,000 budget comes from.
TOKEN_TABLE = [
    # (width, height, v_tile, expected_tokens, within_budget)
    (1920, 1088, 22, 44_880, False),   # SWEEP §2b: 12984MB peak, 349.8s
    (1920, 1088, 19, 38_760, True),    # SWEEP §2b: 11707MB peak, 276.4s
    (2560, 1472, 13, 47_840, False),   # SWEEP §2b: 13601MB peak, 343.5s
    (2560, 1472, 10, 36_800, True),    # SWEEP §2b: 11292MB peak, 259.8s
    (1216, 1664, 22, 43_472, False),   # portrait high-res: over on standard...
    (1216, 1664, 19, 37_544, True),    # ...and inside the budget on 19.
    (1280, 768, 22, 21_120, True),     # the sweep's quality arm — comfortable
    # The 16:9-ish comfortable point the WebUI draws as the standard window's
    # resolution guide (owner decision 2026-08-12). Confirmed on real hardware:
    # 3 runs at 1792x1024 peaked at 11,846MB / 11,847MB actually allocated with
    # no super-linear cost against 1280x768 (Docs/VERIFICATION_LOG.md §59). The
    # neighbours one 64-step up are all over budget (1856x1024 = 40,832), which
    # is what makes this a GUIDE and not just some resolution that happens to
    # fit; the WebUI's `shell/tokenBudget.test.ts` pins that other side.
    (1792, 1024, 22, 39_424, True),
]


@pytest.mark.parametrize("width,height,v_tile,expected,within", TOKEN_TABLE)
def test_chain_window_tokens_expected_values(width, height, v_tile, expected, within):
    tokens = chain_math.chain_window_tokens(width, height, v_tile)
    assert tokens == expected
    assert (tokens <= chain_math.CHAIN_COMFORT_TOKEN_BUDGET) is within


def test_comfort_budget_value():
    assert chain_math.CHAIN_COMFORT_TOKEN_BUDGET == 40_000


def test_config_default_budget_is_the_chain_math_constant():
    """``config.LimitsConfig`` PUBLISHES this budget so a client can draw its
    resolution guides from a served number instead of hard-coding one
    (2026-08-12). The served default must stay the same object of truth as the
    geometry constant — if someone edits one of the two, this goes red."""
    from config import LimitsConfig

    assert LimitsConfig().chain_comfort_token_budget == \
        chain_math.CHAIN_COMFORT_TOKEN_BUDGET == 40_000


def test_config_default_single_comfort_token_budget():
    """``config.LimitsConfig.single_comfort_token_budget`` is a LITERAL (no
    chain_math constant to mirror — a single `/generate` refines its whole
    clip in one pass, a different geometry from a chain's stage-2 window).
    Pinned here so an accidental edit doesn't silently drift the value the
    WebUI's Create screen calibrates its smart comfort marker against
    (Docs/COMFORT_LIMIT_TABLE.md)."""
    from config import LimitsConfig

    assert LimitsConfig().single_comfort_token_budget == 44_880


def test_single_and_chain_comfort_budgets_are_separate_keys():
    """Regression: single_comfort_token_budget (Create, one-shot) and
    chain_comfort_token_budget (Chained, one stage-2 window) are DIFFERENT
    config keys with DIFFERENT calibrated values covering different
    workloads. A client that reads the wrong key for the wrong screen would
    still type-check and still run — this is the guard that would go red on
    that mistake instead."""
    from config import LimitsConfig

    limits = LimitsConfig()
    assert limits.single_comfort_token_budget == 44_880
    assert limits.chain_comfort_token_budget == 40_000
    assert limits.single_comfort_token_budget != limits.chain_comfort_token_budget


def test_high_resolution_window_is_the_documented_escape_hatch():
    """The whole point of the opt-in: a resolution that overshoots the budget on
    the standard window fits on the high_resolution one."""
    std_tile, _ = chain_math.STAGE2_WINDOW_PRESETS["standard"]
    hi_tile, _ = chain_math.STAGE2_WINDOW_PRESETS["high_resolution"]
    for width, height in [(1920, 1088), (1216, 1664)]:
        assert chain_math.chain_window_tokens(width, height, std_tile) > \
            chain_math.CHAIN_COMFORT_TOKEN_BUDGET
        assert chain_math.chain_window_tokens(width, height, hi_tile) <= \
            chain_math.CHAIN_COMFORT_TOKEN_BUDGET


# ── V2V context ceiling per window ──────────────────────────────────────────
def test_stage2_max_context_px_per_window():
    """``px_from_v_latent(v_tile - 1)`` — the largest 8n+1 context whose frozen
    head leaves stage-2 tile 0 at least one FRESH latent frame.

    161 for the standard window is ABOVE config.limits.v2v_context_frames_max
    (145), so it never binds there — the standard path is unchanged. 137 for
    high_resolution is BELOW 145, so it does bind."""
    assert chain_math.stage2_max_context_px(22) == 161
    assert chain_math.stage2_max_context_px(19) == 137
    # ...and the ceiling really is "one latent frame short of the window":
    assert chain_math.v_latent_frames(137) == 18   # < 19
    assert chain_math.v_latent_frames(145) == 19   # == 19 -> tile 0 100% frozen


# ── layout geometry, pinned against the REAL sweep runs ─────────────────────
def test_high_resolution_layout_matches_the_measured_sweep_run():
    """Pinned against outputs/stage2_window_sweep/runs/w19kt7/result.json —
    the actual GPU run of 3 clips x 121f @24fps, kv=3 with v_tile=19/v_adv=12."""
    layout = chain_math.compute_chain_layout(
        [121, 121, 121], 24.0, kv=3, v_tile=19, v_adv=12
    )
    assert layout.v_tile == 19
    assert layout.v_adv == 12
    assert layout.kt_v == 7
    assert layout.v_tiles == [(0, 19), (12, 19), (24, 18)]
    assert layout.tile_seam_junctions == [144, 240]
    # The clip seams are window-independent (same in every sweep run).
    assert layout.segment_seam_junctions == [120, 224]
    assert layout.total_px == 329
    assert layout.f_total == 42


def test_standard_layout_matches_the_measured_sweep_run():
    """Same clip configuration on the DEFAULT window — outputs/stage2_window_
    sweep/runs/w22/result.json. This is the "nothing changed" control."""
    layout = chain_math.compute_chain_layout([121, 121, 121], 24.0, kv=3)
    assert (layout.v_tile, layout.v_adv, layout.kt_v) == (22, 18, 4)
    assert layout.v_tiles == [(0, 22), (18, 22), (36, 6)]
    assert layout.tile_seam_junctions == [168, 312]


def test_omitted_window_is_identical_to_explicit_standard():
    """The core regression contract: not passing a window must produce exactly
    the layout the code produced before the knob existed."""
    for clip_frames, fps, kv in [
        ([121, 121, 121], 24.0, 3),
        ([49, 49], 24.0, 2),
        ([481] * 4, 30.0, 3),
        ([241, 121, 361], 25.0, 1),
    ]:
        implicit = chain_math.compute_chain_layout(clip_frames, fps, kv=kv)
        explicit = chain_math.compute_chain_layout(
            clip_frames, fps, kv=kv,
            v_tile=chain_math.STAGE2_WINDOW_PRESETS["standard"][0],
            v_adv=chain_math.STAGE2_WINDOW_PRESETS["standard"][1],
        )
        assert implicit.to_dict() == explicit.to_dict()


# ── non-24fps behaviour ─────────────────────────────────────────────────────
# The "advance is a multiple of 3" rule that keeps the audio rounding EXACT is a
# 24fps property. At other frame rates some (fps, clip-count, clip-length, kv)
# combinations make compute_chain_layout raise "audio reassembly N != a_total M".
# That PREDATES this feature and is the same guard for both windows: it fires in
# the API validator, i.e. as a 422 BEFORE any GPU work, never as a corrupt run.
#
# Per the work order, NO new validation is added for this. What the tests below
# do is FIX the measured behaviour so a future change is visible:
#   * at 24fps (and most other rates) the two windows raise on exactly the same
#     set of configurations;
#   * at 23.976 and 30fps they do NOT — in BOTH directions (each window rejects
#     a handful the other accepts). Those are enumerated explicitly.
#   * a SINGLE clip diverges too, at 23.976 with a long clip. That used to be
#     invisible here because 321 was missing from the clip-length list below, and
#     it used to MATTER because the A2V length preflight
#     (chain_math.audio_latents_required -> services/pipeline_manager.py) resolved
#     a_total by building a whole layout on the DEFAULT window: a config the
#     request's own window accepts could still raise inside preflight and surface
#     as a 500. Since §1-16 (long A2V) that preflight computes a_total directly
#     and never touches the stage-2 window, so a divergence is now just an
#     ordinary 422 from the validator, on the window the request asked for.
#     ``test_audio_latents_required_survives_every_window_divergence`` pins that.
_UI_FPS = [12.0, 15.0, 23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0, 59.94, 60.0]
_UI_CLIP_FRAMES = [9, 49, 121, 241, 321, 361, 481]
_UI_CLIP_COUNTS = [1, 2, 3, 4, 8, 12, 24]
_UI_KV = [1, 3, 8]

# The COMPLETE set of (clip_frames, n_clips, kv) where the two windows disagree,
# per frame rate. Measured 2026-08-09, re-measured 2026-08-10 with 321 added to
# the clip-length list. Every entry is a pre-existing audio-rounding fragility
# surfaced as a 422, not a new failure mode.
KNOWN_WINDOW_DIVERGENCES: dict[float, set[tuple[int, int, int]]] = {
    23.976: {
        (49, 8, 3), (121, 2, 3), (121, 8, 3), (241, 3, 8),
        (321, 1, 1), (321, 1, 3), (321, 1, 8), (321, 2, 8), (321, 3, 8),
    },
    30.0: {
        (121, 24, 8), (241, 24, 8), (321, 2, 8), (321, 8, 8),
        (361, 24, 8), (481, 24, 8),
    },
}

# Same thing restricted to ONE clip: (clip_frames, kv) per frame rate. Kept
# separate because the single-clip case is the one A2V used to be limited to,
# and it is still the shape most likely to reach the length preflight.
KNOWN_SINGLE_CLIP_DIVERGENCES: dict[float, set[tuple[int, int]]] = {
    23.976: {(321, 1), (321, 3), (321, 8)},
}


def _raises(clip_frames, fps, kv, v_tile, v_adv) -> str | None:
    try:
        chain_math.compute_chain_layout(
            clip_frames, fps, kv=kv, v_tile=v_tile, v_adv=v_adv
        )
    except ValueError as exc:
        return str(exc)
    return None


@pytest.mark.parametrize("fps", _UI_FPS)
def test_window_divergence_set_is_exactly_the_measured_one(fps):
    """Exhaustive over the (clip length x clip count x kv) space the UI allows."""
    std_tile, std_adv = chain_math.STAGE2_WINDOW_PRESETS["standard"]
    hi_tile, hi_adv = chain_math.STAGE2_WINDOW_PRESETS["high_resolution"]
    diverged: set[tuple[int, int, int]] = set()
    for nf in _UI_CLIP_FRAMES:
        for n in _UI_CLIP_COUNTS:
            for kv in _UI_KV:
                clip_frames = [nf] * n
                std = _raises(clip_frames, fps, kv, std_tile, std_adv)
                hi = _raises(clip_frames, fps, kv, hi_tile, hi_adv)
                if (std is None) != (hi is None):
                    diverged.add((nf, n, kv))
    assert diverged == KNOWN_WINDOW_DIVERGENCES.get(fps, set()), (
        f"stage-2 window divergence set changed at fps={fps}. Every entry is a "
        "config one window accepts and the other 422s (pre-GPU). If this is an "
        "intended geometry change, update KNOWN_WINDOW_DIVERGENCES."
    )


def test_at_24fps_the_two_windows_never_diverge():
    """The supported/documented frame rate has no divergence at all — stated
    separately so it cannot be lost in the table above."""
    assert KNOWN_WINDOW_DIVERGENCES.get(24.0, set()) == set()


@pytest.mark.parametrize("fps", _UI_FPS)
def test_single_clip_window_divergence_set_is_exactly_the_measured_one(fps):
    """Single-clip divergences, enumerated. At 24fps (and everywhere except
    23.976 x 321 frames) there are none. The 23.976/321 entries are the "long
    single clip" hole: the DEFAULT window raises "audio reassembly 334 != a_total
    335" while high_resolution accepts. It is a plain pre-GPU 422 from the
    validator on whichever window the request chose — see the next test for why
    it can no longer become a 500 in the A2V length preflight."""
    std_tile, std_adv = chain_math.STAGE2_WINDOW_PRESETS["standard"]
    hi_tile, hi_adv = chain_math.STAGE2_WINDOW_PRESETS["high_resolution"]
    diverged: set[tuple[int, int]] = set()
    for nf in _UI_CLIP_FRAMES:
        for kv in _UI_KV:
            std = _raises([nf], fps, kv, std_tile, std_adv)
            hi = _raises([nf], fps, kv, hi_tile, hi_adv)
            if (std is None) != (hi is None):
                diverged.add((nf, kv))
    assert diverged == KNOWN_SINGLE_CLIP_DIVERGENCES.get(fps, set()), (
        f"single-clip stage-2 window divergence set changed at fps={fps}. "
        "If this is an intended geometry change, update "
        "KNOWN_SINGLE_CLIP_DIVERGENCES."
    )


@pytest.mark.parametrize("fps", _UI_FPS)
def test_audio_latents_required_survives_every_window_divergence(fps):
    """The A2V length preflight is window-INDEPENDENT (§1-16).

    ``chain_math.audio_latents_required`` computes a_total straight from the clip
    lengths, fps and K_v, so it must return a number for every configuration in
    the UI space — including the ones one stage-2 window rejects. When it still
    resolved a whole ChainLayout on the DEFAULT window, the divergent configs
    above raised inside ``services/pipeline_manager.preflight_source_audio`` and
    came back as a 500 instead of a 422 or a job."""
    for nf in _UI_CLIP_FRAMES:
        for n in _UI_CLIP_COUNTS:
            for kv in _UI_KV:
                if kv >= chain_math.v_latent_frames(nf):
                    continue  # genuinely impossible geometry; 422 by design
                required = chain_math.audio_latents_required([nf] * n, fps, kv=kv)
                assert required > 0


def test_the_nonstandard_fps_failures_actually_exist():
    """Sanity: if NOTHING ever raised, the assertions above would be vacuous.
    29.97fps x 4 clips x 241f raises on BOTH windows (a shared, pre-existing
    fragility)."""
    assert _raises([241] * 4, 29.97, 3, 22, 18) is not None
    assert _raises([241] * 4, 29.97, 3, 19, 12) is not None


# ── API surface ─────────────────────────────────────────────────────────────
_BASE_CHAIN = {
    "prompt": "a cat",
    "width": 512,
    "height": 320,
    "frame_rate": 24.0,
    "clips": [{"num_frames": 121}, {"num_frames": 121}],
}


def test_stage2_window_defaults_to_standard():
    req = GenerateChainRequest(**_BASE_CHAIN)
    assert req.stage2_window == "standard"


def test_stage2_window_accepts_high_resolution():
    req = GenerateChainRequest(**{**_BASE_CHAIN, "stage2_window": "high_resolution"})
    assert req.stage2_window == "high_resolution"


def test_stage2_window_rejects_unknown_value():
    with pytest.raises(ValidationError):
        GenerateChainRequest(**{**_BASE_CHAIN, "stage2_window": "6sec"})


def _capturing_real_backend(captured: list[dict]):
    """A ``_RealBackend`` whose ``_send`` records the worker payload instead of
    writing to a subprocess. Same construction as
    ``tests/test_ltx_runner_payload.py::_capturing_backend`` (kept local rather
    than imported so that module's own fixtures stay private to it)."""
    import threading
    import types
    from pathlib import Path

    from services.ltx_runner import _RealBackend

    be = _RealBackend.__new__(_RealBackend)
    be._proc = types.SimpleNamespace(poll=lambda: None)      # type: ignore[attr-defined]
    be._lock = threading.Lock()

    def _send(msg: dict) -> None:
        captured.append(msg)
        out = Path(msg["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00" * 16)

    be._send = _send                                          # type: ignore[attr-defined]
    be._read_chain_events = (                                 # type: ignore[attr-defined]
        lambda cb: {"event": "done", "seed_used": 7, "peak_vram_mb": 100, "chain": {}}
    )
    return be


def test_omitted_stage2_window_is_absent_from_the_worker_payload(tmp_path):
    """Regression contract: a default chain's worker payload must be
    byte-identical to before this knob existed."""
    captured: list[dict] = []
    be = _capturing_real_backend(captured)
    be.generate_chain(GenerateChainRequest(**_BASE_CHAIN), tmp_path / "out")
    assert "stage2_window" not in captured[0]


def test_high_resolution_reaches_the_worker_payload(tmp_path):
    captured: list[dict] = []
    be = _capturing_real_backend(captured)
    be.generate_chain(
        GenerateChainRequest(**{**_BASE_CHAIN, "stage2_window": "high_resolution"}),
        tmp_path / "out",
    )
    assert captured[0]["stage2_window"] == "high_resolution"


# ── V2V x high_resolution: the 422 ──────────────────────────────────────────
_V2V_CHAIN = {
    "prompt": "a cat",
    "width": 512,
    "height": 320,
    "frame_rate": 24.0,
    "clips": [{"num_frames": 361}],
    "source_video": {"video_id": "vid_1", "context_frames": 145},
}


def test_v2v_context_145_still_allowed_on_the_standard_window():
    """The public cap (config.limits.v2v_context_frames_max = 145) is unchanged:
    145 keeps working exactly as before on the default window."""
    req = GenerateChainRequest(**_V2V_CHAIN)
    assert req.source_video is not None
    assert req.source_video.context_frames == 145


def test_v2v_context_145_rejected_on_the_high_resolution_window():
    """145 -> n_ctx_v = 19 == v_tile, i.e. stage-2 tile 0 would be 100% frozen
    with nothing left to generate — an untested degenerate that
    compute_chain_layout's own ``n_ctx_v > v_tile`` guard waves through."""
    with pytest.raises(ValidationError) as excinfo:
        GenerateChainRequest(**{**_V2V_CHAIN, "stage2_window": "high_resolution"})
    message = str(excinfo.value)
    assert "137" in message                 # the derived ceiling, not a literal
    assert "context_frames" in message
    assert "standard" in message            # tells the user the way out


def test_v2v_context_137_allowed_on_the_high_resolution_window():
    req = GenerateChainRequest(
        **{**_V2V_CHAIN,
           "source_video": {"video_id": "vid_1", "context_frames": 137},
           "stage2_window": "high_resolution"}
    )
    assert req.stage2_window == "high_resolution"


def test_v2v_ceiling_message_is_derived_not_hardcoded():
    """The 422's number comes from ``stage2_max_context_px``, so a future preset
    gets a truthful message for free."""
    assert chain_math.stage2_max_context_px(
        chain_math.STAGE2_WINDOW_PRESETS["high_resolution"][0]
    ) == 137


# ── worker-side resolution ──────────────────────────────────────────────────
def test_worker_resolve_stage2_window():
    """engine.worker._resolve_stage2_window mirrors chain_math's resolver but
    raises RuntimeError (this module's protocol-mismatch convention)."""
    pytest.importorskip("torch")
    from engine.worker import _resolve_stage2_window

    assert _resolve_stage2_window({}) == (22, 18)
    assert _resolve_stage2_window({"stage2_window": "standard"}) == (22, 18)
    assert _resolve_stage2_window({"stage2_window": "high_resolution"}) == (19, 12)
    with pytest.raises(RuntimeError, match="unknown stage2_window"):
        _resolve_stage2_window({"stage2_window": "nope"})


# ── §1-15 reference preprocess frame cap ────────────────────────────────────
def test_preprocess_frame_cap_single_generate_uses_num_frames():
    pytest.importorskip("torch")
    from engine.worker import _preprocess_frame_cap

    assert _preprocess_frame_cap({"num_frames": 121}) == 121


def test_preprocess_frame_cap_absent_is_none():
    """No num_frames and no clips -> None (decode everything, pre-existing
    behaviour for a bare reference_video with no generation-length context)."""
    pytest.importorskip("torch")
    from engine.worker import _preprocess_frame_cap

    assert _preprocess_frame_cap({}) is None
    assert _preprocess_frame_cap({"clips": []}) is None


def test_preprocess_frame_cap_single_clip_chain_matches_the_old_clip0_value():
    """Regression: a 1-clip chain (the only shape this cap covered before
    §1-15, and the only shape a depth reference can still reach post-422) must
    resolve to exactly the same number as the old ``clips[0]["num_frames"]``
    read — the formula's ``- (n-1)*kv`` term vanishes for n=1."""
    pytest.importorskip("torch")
    from engine.worker import _preprocess_frame_cap

    for num_frames in (9, 121, 241, 481):
        msg = {"clips": [{"num_frames": num_frames}], "overlap_frames": 3}
        assert _preprocess_frame_cap(msg) == num_frames


def test_preprocess_frame_cap_multi_clip_chain_is_the_chain_total_px():
    """A multi-clip chain caps to chain_math's own total_px for the same
    (clip_frames, kv) — pinned against ``compute_chain_layout`` directly so a
    future change to either side shows up as a diff, not a silent drift."""
    pytest.importorskip("torch")
    import chain_math
    from engine.worker import _preprocess_frame_cap

    cases = [
        ([121, 121, 121], 3),
        ([241, 121, 361], 1),
        ([9] * 24, 1),
    ]
    for clip_frames, kv in cases:
        msg = {
            "clips": [{"num_frames": f} for f in clip_frames],
            "overlap_frames": kv,
        }
        expected = chain_math.compute_chain_layout(clip_frames, 24.0, kv=kv).total_px
        assert _preprocess_frame_cap(msg) == expected


def test_preprocess_frame_cap_missing_overlap_frames_defaults_to_dev3():
    """``overlap_frames`` is always present on a real chain worker payload
    (``int(msg["overlap_frames"])`` a few lines below the cap's only call
    site), but the cap helper itself is defensive and falls back to
    ``chain_math.DEFAULT_OVERLAP_FRAMES`` (K_v=3) rather than raising."""
    pytest.importorskip("torch")
    import chain_math
    from engine.worker import _preprocess_frame_cap

    msg = {"clips": [{"num_frames": 121}, {"num_frames": 121}]}
    assert chain_math.DEFAULT_OVERLAP_FRAMES == 3
    expected = chain_math.compute_chain_layout([121, 121], 24.0, kv=3).total_px
    assert _preprocess_frame_cap(msg) == expected


# ── mock backend + metadata ─────────────────────────────────────────────────
def _run_chain_job(client, **overrides) -> dict:
    """Submit a chain through the REST API (mock backend) and return the written
    metadata.json. Mirrors tests/test_chain.py's own harness."""
    import json

    payload = {
        "prompt": "a serene mountain lake at dawn",
        "width": 384, "height": 256, "frame_rate": 24.0,
        "num_inference_steps": 8, "guidance_scale": 1.0, "seed": 123,
        "pipeline": "distilled", "overlap_frames": 3, "overlap_strength": 0.5,
        "clips": [{"num_frames": 121}, {"num_frames": 121}, {"num_frames": 121}],
        **overrides,
    }
    response = client.post("/api/v1/generate/chain", json=payload)
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
    metadata_path = client.app_context.config.output_dir / job_id / "metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def test_mock_backed_chain_honours_the_window_end_to_end(client):
    """The mock backend resolves the SAME preset table the real one does —
    otherwise every mock-backed gate would pass with standard-window geometry
    no matter what the request asked for (a silent false green)."""
    metadata = _run_chain_job(client, stage2_window="high_resolution")
    chain_meta = metadata["chain"]
    assert chain_meta["stage2_window"] == "high_resolution"
    assert chain_meta["v_tile"] == 19
    assert chain_meta["kt_v"] == 7
    assert chain_meta["video_tiles"] == [[0, 19], [12, 19], [24, 18]]
    assert chain_meta["tile_seam_junctions"] == [144, 240]


def test_mock_backed_chain_default_window_is_unchanged(client):
    metadata = _run_chain_job(client)
    chain_meta = metadata["chain"]
    assert chain_meta["stage2_window"] == "standard"
    assert chain_meta["v_tile"] == 22
    assert chain_meta["kt_v"] == 4
    assert chain_meta["tile_seam_junctions"] == [168, 312]


# ── Gradio prechecks share the same resolution ──────────────────────────────
def test_gradio_chain_precheck_accepts_and_resolves_the_window():
    from gradio_ui.validation import check_chain_total

    # Default (no window argument) — unchanged behaviour.
    assert check_chain_total([121, 121], 24.0, 3) is None
    # Explicit, both presets — the precheck must not itself reject either.
    assert check_chain_total([121, 121], 24.0, 3, stage2_window="standard") is None
    assert check_chain_total([121, 121], 24.0, 3, stage2_window="high_resolution") is None


def test_gradio_duration_label_accepts_the_window_after_lang():
    """``lang`` is passed POSITIONALLY by ui.py's Gradio ``inputs=[...]`` wiring,
    so the new argument must sit AFTER it — this pins the order."""
    from gradio_ui.presets import compute_chain_duration_label

    positional = compute_chain_duration_label([True, True], [121, 121], 24.0, 3, "ja")
    keyworded = compute_chain_duration_label(
        [True, True], [121, 121], 24.0, 3, "ja", stage2_window="standard"
    )
    assert positional == keyworded


# ── FastAPI end-to-end 422 ──────────────────────────────────────────────────
def test_api_rejects_v2v_high_resolution_over_ceiling(client):
    """The ValueError above must actually surface as a 422 at the endpoint, not
    a 500 — the request never reaches the job layer."""
    response = client.post("/api/v1/generate/chain", json={
        **_V2V_CHAIN, "stage2_window": "high_resolution",
    })
    assert response.status_code == 422, response.text


# ── full_length (§1-19) ─────────────────────────────────────────────────────
# The a2v window: 61/61 -> kt_v 0, i.e. NO のり代 and therefore no tile seam.
# 61 latent frames == 481 pixel frames == ChainClip.num_frames's own ceiling, so
# on a ONE-clip chain the tiler always degenerates to a single tile spanning the
# whole timeline — stage-2 becomes exactly what plain POST /generate does.
_A2V_FULL = {
    "prompt": "a woman speaking",
    "width": 512,
    "height": 320,
    "frame_rate": 24.0,
    "clips": [{"num_frames": 481}],
    "source_audio": {"audio_id": "aud_1"},
    "stage2_window": "full_length",
}


def test_resolve_stage2_window_full_length():
    assert chain_math.resolve_stage2_window("full_length") == (61, 61)
    assert chain_math.STAGE2_WINDOW_FULL_LENGTH == "full_length"
    assert chain_math.STAGE2_WINDOW_PRESETS[chain_math.STAGE2_WINDOW_FULL_LENGTH] == (61, 61)
    # It is NOT the default — an omitted window is still the tiled 22/18 one.
    assert chain_math.STAGE2_WINDOW_DEFAULT != chain_math.STAGE2_WINDOW_FULL_LENGTH


def test_full_length_layout_is_one_tile_over_the_whole_timeline():
    """481 pixel frames (the per-clip maximum) @24fps -> 61 latent frames, which
    is exactly the window: ONE tile, no tile seam anywhere."""
    layout = chain_math.compute_chain_layout([481], 24.0, kv=3, v_tile=61, v_adv=61)
    assert (layout.v_tile, layout.v_adv, layout.kt_v) == (61, 61, 0)
    assert layout.f_total == 61
    assert layout.total_px == 481
    assert layout.n_tiles == 1
    assert layout.v_tiles == [(0, 61)]
    assert layout.tile_seam_junctions == []
    # A single clip has no segment seam either, so the whole junction list — the
    # thing the mock/metadata publish — is empty: a seamless timeline.
    assert layout.segment_seam_junctions == []
    assert layout.all_junctions == []
    # ONE audio tile spanning the full audio latent length.
    assert layout.a_tiles == [(0, layout.a_total)]


@pytest.mark.parametrize("num_frames", [121, 49])
def test_full_length_short_clips_degenerate_to_one_tile(num_frames):
    """Below the ceiling the window simply clips to the timeline: still one
    tile, whose length IS the clip's latent length (not a padded 61)."""
    layout = chain_math.compute_chain_layout(
        [num_frames], 24.0, kv=3, v_tile=61, v_adv=61
    )
    expected_latent = chain_math.v_latent_frames(num_frames)
    assert layout.n_tiles == 1
    assert layout.v_tiles == [(0, expected_latent)]
    assert layout.tile_seam_junctions == []
    assert layout.f_total == expected_latent


def test_full_length_kt_a_is_negative_and_inert():
    """RECORD, not a requirement: with v_adv == v_tile the audio advance (508 @
    24fps) OVERSHOOTS one tile's audio length (501), so the derived kt_a is
    NEGATIVE (-7). It is fps-dependent (23.976 -> -7, 30 -> -6, 12 -> -15).

    Harmless because it is never read: chain_math's own audio-reassembly checks
    live under ``n_tiles > 1``, and engine/pipeline/chain_pipeline.py reads
    ``kt_a`` only inside its ``i >= 1`` tile branch. The one place ``audio_adv``
    is used for tile 0 is ``i * audio_adv``, which is 0."""
    for fps, expected_kt_a in [(24.0, -7), (23.976, -7), (30.0, -6), (12.0, -15)]:
        layout = chain_math.compute_chain_layout(
            [481], fps, kv=3, v_tile=61, v_adv=61
        )
        assert layout.n_tiles == 1
        assert layout.kt_a == expected_kt_a, (fps, layout.kt_a)


def test_full_length_rejects_a_timeline_that_needs_two_tiles():
    """The wall that keeps the window honest wherever the preset NAME comes
    from: with kt_v == 0 a second tile would butt onto the first with nothing
    shared to freeze and blend, so chain_math refuses outright."""
    with pytest.raises(ValueError, match="zero-overlap stage-2 window"):
        chain_math.compute_chain_layout(
            [481, 481], 24.0, kv=3, v_tile=61, v_adv=61
        )


# ── full_length x the API contract ──────────────────────────────────────────
def test_api_accepts_full_length_single_clip_with_audio():
    req = GenerateChainRequest(**_A2V_FULL)
    assert req.stage2_window == "full_length"
    assert len(req.clips) == 1


def test_api_rejects_full_length_on_two_clips():
    """The clip-count guard fires BEFORE chain_math's own wall, so the user gets
    the actionable message rather than the geometry one."""
    with pytest.raises(ValidationError) as excinfo:
        GenerateChainRequest(**{
            **_A2V_FULL,
            "clips": [{"num_frames": 481}, {"num_frames": 481}],
        })
    message = str(excinfo.value)
    assert "full_length" in message
    assert "exactly 1 clip" in message


def test_api_rejects_full_length_without_source_audio():
    """A 1-clip chain WITHOUT audio would tile fine; the window is unlocked for
    the a2v flow only (§1-19), so this is a scope 422, not a geometry one. The
    clip here also carries a reference_video_id + lora so it is a legitimately
    valid 1-clip chain in every OTHER respect."""
    payload = {k: v for k, v in _A2V_FULL.items() if k != "source_audio"}
    with pytest.raises(ValidationError) as excinfo:
        GenerateChainRequest(**{
            **payload,
            "reference_video_id": "vid_1",
            "loras": [{"name": "canny", "strength": 1.0}],
        })
    message = str(excinfo.value)
    assert "full_length" in message
    assert "source_audio" in message


def test_api_accepts_full_length_with_a_reference_video_ic_lora():
    """A2V x IC-LoRA is a supported combination, so the new guard must NOT
    reject a reference video riding along with the audio."""
    req = GenerateChainRequest(**{
        **_A2V_FULL,
        "reference_video_id": "vid_1",
        "loras": [{"name": "canny", "strength": 1.0}],
    })
    assert req.stage2_window == "full_length"
    assert req.reference_video_id == "vid_1"


def test_full_length_reaches_the_worker_payload(tmp_path):
    """The window is non-default, so the additive worker key must ride along."""
    captured: list[dict] = []
    be = _capturing_real_backend(captured)
    be.generate_chain(GenerateChainRequest(**_A2V_FULL), tmp_path / "out")
    assert captured[0]["stage2_window"] == "full_length"


def test_worker_resolves_full_length_without_any_engine_change():
    """engine/ is UNTOUCHED by §1-19: the worker resolves the new name purely
    through chain_math's preset table, so (61, 61) arrives for free."""
    pytest.importorskip("torch")
    from engine.worker import _resolve_stage2_window

    assert _resolve_stage2_window({"stage2_window": "full_length"}) == (61, 61)


def test_full_length_accepts_a_config_the_standard_window_rejects():
    """DELIBERATE BEHAVIOUR WIDENING, recorded here on purpose.

    23.976fps x 321 frames on ONE clip is a known audio-rounding casualty of the
    standard window (``KNOWN_SINGLE_CLIP_DIVERGENCES`` above): it 422s with
    "audio reassembly ... != a_total". full_length ACCEPTS it, because that
    check lives under ``n_tiles > 1`` and full_length never has a second tile.
    Intended: the seam-consistency check has nothing to check when there is no
    seam."""
    assert (321, 3) in KNOWN_SINGLE_CLIP_DIVERGENCES[23.976]
    assert _raises([321], 23.976, 3, 22, 18) is not None      # standard: 422
    assert _raises([321], 23.976, 3, 61, 61) is None          # full_length: OK
    layout = chain_math.compute_chain_layout([321], 23.976, kv=3, v_tile=61, v_adv=61)
    assert layout.n_tiles == 1
