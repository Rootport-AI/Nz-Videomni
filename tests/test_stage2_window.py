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
    """The two presets and their DERIVED overlap (kt_v = v_tile - v_adv)."""
    assert chain_math.STAGE2_WINDOW_PRESETS == {
        "standard": (22, 18),
        "high_resolution": (19, 12),
    }
    assert chain_math.STAGE2_WINDOW_DEFAULT == "standard"
    kt = {name: v_tile - v_adv
          for name, (v_tile, v_adv) in chain_math.STAGE2_WINDOW_PRESETS.items()}
    # standard keeps the S2-spike 4-frame overlap; high_resolution is the
    # follow-up run's wider 7-frame overlap ("のり代7").
    assert kt == {"standard": 4, "high_resolution": 7}


def test_every_preset_advance_is_a_multiple_of_three():
    """The 24fps video/audio advance rounding is only exact when the stage-2
    advance is a multiple of 3 — a guard for any FUTURE preset added here."""
    for name, (_v_tile, v_adv) in chain_math.STAGE2_WINDOW_PRESETS.items():
        assert v_adv % 3 == 0, f"preset {name!r} advance {v_adv} is not a multiple of 3"


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
]


@pytest.mark.parametrize("width,height,v_tile,expected,within", TOKEN_TABLE)
def test_chain_window_tokens_expected_values(width, height, v_tile, expected, within):
    tokens = chain_math.chain_window_tokens(width, height, v_tile)
    assert tokens == expected
    assert (tokens <= chain_math.CHAIN_COMFORT_TOKEN_BUDGET) is within


def test_comfort_budget_value():
    assert chain_math.CHAIN_COMFORT_TOKEN_BUDGET == 40_000


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
#   * for a SINGLE clip — the only shape A2V can take (api/models.py: "source_audio
#     requires exactly 1 clip in v1") — the two windows never diverge at any rate.
#     This is the load-bearing one: the A2V length preflight
#     (chain_math.audio_latents_required -> services/pipeline_manager.py) resolves
#     a_total on the DEFAULT window, so a divergence there would mean a chain that
#     passes preflight and then 422s only because of the opt-in window.
_UI_FPS = [12.0, 15.0, 23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0, 59.94, 60.0]
_UI_CLIP_FRAMES = [9, 49, 121, 241, 361, 481]
_UI_CLIP_COUNTS = [1, 2, 3, 4, 8, 12, 24]
_UI_KV = [1, 3, 8]

# The COMPLETE set of (clip_frames, n_clips, kv) where the two windows disagree,
# per frame rate. Measured 2026-08-09 over the space above. Every entry is a
# pre-existing audio-rounding fragility surfaced as a 422, not a new failure mode.
KNOWN_WINDOW_DIVERGENCES: dict[float, set[tuple[int, int, int]]] = {
    23.976: {(49, 8, 3), (121, 2, 3), (121, 8, 3), (241, 3, 8)},
    30.0: {(121, 24, 8), (241, 24, 8), (361, 24, 8), (481, 24, 8)},
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
def test_single_clip_never_diverges_between_windows(fps):
    """A2V is single-clip only, and its length preflight resolves a_total on the
    DEFAULT window. If a single-clip config ever raised on one window and not the
    other, an A2V job could pass preflight and then 422 purely because of the
    stage-2 window choice."""
    std_tile, std_adv = chain_math.STAGE2_WINDOW_PRESETS["standard"]
    hi_tile, hi_adv = chain_math.STAGE2_WINDOW_PRESETS["high_resolution"]
    for nf in _UI_CLIP_FRAMES:
        for kv in _UI_KV:
            std = _raises([nf], fps, kv, std_tile, std_adv)
            hi = _raises([nf], fps, kv, hi_tile, hi_adv)
            assert (std is None) == (hi is None), (
                f"single-clip divergence at fps={fps} nf={nf} kv={kv}: "
                f"standard={std!r} high_resolution={hi!r}"
            )


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
