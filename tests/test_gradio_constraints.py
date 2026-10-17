"""The Generate tab reads the loaded base model's GET /models ``constraints``
(MiniMax H3 stage 2, PLAN §3.3 / §3.6): sampling sliders, the shape precheck,
the width / height / frames labels and the spill-free warning. A server that
sends no ``constraints`` reads as the LTX rules, so LTX is unchanged.
MockTransport only -- no server, no GPU.
"""

from __future__ import annotations

import json

import gradio as gr
import httpx

from gradio_ui.api_client import ApiClient
from gradio_ui.constraints import (
    LTX_CONSTRAINTS,
    active_scope,
    comfort_table_missing,
    field_labels,
    frame_rule,
    normalize_constraints,
    sampling_payload,
    single_scope_updates,
    single_shape_error,
)
from gradio_ui.handlers import make_generate_handler
from gradio_ui.i18n import LABELS
from gradio_ui.presets import KF_MAX_SLOTS, compute_spill_warning
from gradio_ui.ui import build_ui

#: The H3 block exactly as PLAN §3.3 fixes it.
H3_CONSTRAINTS = {
    "num_frames": {"step": 17, "offset": 5, "min": 22, "max": 481},
    "dimension_multiple": 32,
    "frame_rate": {"min": 24, "max": 24},
    "sampling": {
        "num_inference_steps": {"min": 1, "max": 50, "step": 1, "default": 20},
        "shift_video": {"min": 1, "max": 20, "step": 0.1, "default": 12},
        "shift_audio": {"min": 1, "max": 20, "step": 0.1, "default": 3},
    },
}

LTX_SERVED = {
    "num_frames": {"step": 8, "offset": 1, "min": 9, "max": 481},
    "dimension_multiple": 64,
    "frame_rate": {"min": 1, "max": 60},
    "sampling": None,
}


def _models(active: str, *, with_constraints: bool = True) -> dict:
    entries = [
        {"id": "LTX23", "engine_family": "ltx", "display_name": "LTX 2.3",
         "constraints": LTX_SERVED},
        {"id": "MiniMaxH3", "engine_family": "h3", "display_name": "MiniMax H3",
         "constraints": H3_CONSTRAINTS},
    ]
    base_models = []
    for e in entries:
        e = dict(e, active=(e["id"] == active), unsupported_features=[])
        if not with_constraints:
            e.pop("constraints")
        base_models.append(e)
    return {"active_base_model": active, "base_models": base_models}


def _kf_args():
    return [(False, None, 0, 0.8)] * KF_MAX_SLOTS


def _capture_api(captured: dict) -> ApiClient:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["calls"] = captured.get("calls", 0) + 1
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"job_id": "job-1"})
    return ApiClient("http://test", api_key=None,
                     client=httpx.Client(transport=httpx.MockTransport(handler)))


def _first(gen):
    first = next(gen)
    gen.close()
    return first


# ------------------------------------------------------------- reading ---


def test_no_constraints_reads_as_ltx():
    assert normalize_constraints(None) == LTX_CONSTRAINTS
    scope = active_scope(_models("LTX23", with_constraints=False))
    assert scope == {"engine_family": "ltx", "constraints": LTX_CONSTRAINTS}
    # Before the first GET /models: unknown family, LTX rules.
    assert active_scope(None)["engine_family"] == ""
    assert active_scope({})["constraints"] == LTX_CONSTRAINTS


def test_active_scope_reads_the_loaded_base_model():
    scope = active_scope(_models("MiniMaxH3"))
    assert scope["engine_family"] == "h3"
    assert scope["constraints"] == H3_CONSTRAINTS
    assert active_scope(_models("LTX23"))["constraints"] == LTX_SERVED


def test_incomplete_sampling_is_no_sliders():
    raw = dict(H3_CONSTRAINTS, sampling={"num_inference_steps":
                                         H3_CONSTRAINTS["sampling"]["num_inference_steps"]})
    assert normalize_constraints(raw)["sampling"] is None


def test_ltx_labels_are_the_static_strings():
    for lang in ("en", "ja"):
        labels = field_labels(LTX_CONSTRAINTS, lang)
        assert labels == {"width": LABELS[lang]["lbl_width"],
                          "height": LABELS[lang]["lbl_height"],
                          "frames": LABELS[lang]["lbl_frames"]}


def test_h3_labels_follow_the_grid():
    assert frame_rule(H3_CONSTRAINTS) == "17n+5"
    labels = field_labels(H3_CONSTRAINTS, "ja")
    assert labels["frames"] == "フレーム数 (17n+5)"
    assert labels["width"] == "幅 (32の倍数)"
    assert field_labels(H3_CONSTRAINTS, "en")["frames"] == "Frames (17n+5)"


# ------------------------------------------------------------ precheck ---


def test_shape_precheck_ltx_unchanged():
    assert single_shape_error(512, 320, 49, LTX_CONSTRAINTS) is None
    assert single_shape_error(500, 320, 49, LTX_CONSTRAINTS) is not None
    assert single_shape_error(512, 320, 50, LTX_CONSTRAINTS) is not None
    assert single_shape_error(512, 320, 489, LTX_CONSTRAINTS) is not None
    # 32 is not enough for LTX.
    assert single_shape_error(544, 320, 49, LTX_CONSTRAINTS) is not None


def test_shape_precheck_h3():
    assert single_shape_error(640, 384, 124, H3_CONSTRAINTS) is None
    assert single_shape_error(544, 352, 22, H3_CONSTRAINTS) is None
    # 49 (the LTX default) is not 17n+5; 5 is on the grid but below the min.
    assert single_shape_error(640, 384, 49, H3_CONSTRAINTS) is not None
    assert single_shape_error(640, 384, 5, H3_CONSTRAINTS) is not None
    assert single_shape_error(650, 384, 124, H3_CONSTRAINTS) is not None
    msg = single_shape_error(640, 384, 49, H3_CONSTRAINTS, "en")
    assert "17n+5" in msg and "22" in msg and "481" in msg


# ------------------------------------------------------------- payload ---


def test_sampling_payload():
    # No sliders: nothing is sent (the server writes the family default back).
    assert sampling_payload(LTX_CONSTRAINTS, 30, 5.0, 5.0) == {}
    assert sampling_payload(H3_CONSTRAINTS, 25.0, 10.5, 2.0) == {
        "num_inference_steps": 25, "shift_video": 10.5, "shift_audio": 2.0}
    # Unreadable slider values fall back to the served defaults.
    assert sampling_payload(H3_CONSTRAINTS, None, None, None) == {
        "num_inference_steps": 20, "shift_video": 12.0, "shift_audio": 3.0}


def test_generate_ltx_payload_carries_no_sampling_keys():
    captured: dict = {}
    generate = make_generate_handler(_capture_api(captured))
    _first(generate("p", "", _kf_args(), 512, 320, False, 0, 0, 49, 24.0, -1,
                    num_inference_steps=30, shift_video=5.0, shift_audio=5.0))
    body = captured["body"]
    assert "num_inference_steps" not in body
    assert body["guidance_scale"] == 1.0
    assert "shift_video" not in body and "shift_audio" not in body


def test_generate_with_a_stale_ltx_scope_sends_no_step_count():
    # Code review 2b, M-2: another client switched the server to H3 and this
    # page still holds the LTX scope. 73 frames at 512x320 passes both grids
    # (8n+1 and 17n+5); no fixed 8 may ride along, so the server applies the
    # loaded family's own default instead of silently running 8 steps.
    captured: dict = {}
    generate = make_generate_handler(_capture_api(captured))
    _first(generate("p", "", _kf_args(), 512, 320, False, 0, 0, 73, 24.0, -1))
    assert "num_inference_steps" not in captured["body"]


def test_generate_h3_payload_carries_the_three_sliders():
    captured: dict = {}
    generate = make_generate_handler(_capture_api(captured))
    _first(generate("p", "", _kf_args(), 640, 384, False, 0, 0, 124, 24.0, 42,
                    constraints=H3_CONSTRAINTS,
                    num_inference_steps=20, shift_video=12.0, shift_audio=3.0))
    body = captured["body"]
    assert body["num_frames"] == 124
    assert body["num_inference_steps"] == 20
    assert body["shift_video"] == 12.0
    assert body["shift_audio"] == 3.0


def test_generate_h3_precheck_rejects_ltx_grid_with_zero_calls():
    captured: dict = {}
    generate = make_generate_handler(_capture_api(captured))
    out = list(generate("p", "", _kf_args(), 640, 384, False, 0, 0, 49, 24.0, -1,
                        constraints=H3_CONSTRAINTS))
    assert captured.get("calls", 0) == 0
    assert len(out) == 1 and out[0][1] == "" and out[0][2] is None
    assert "17n+5" in out[0][0]


# ---------------------------------------------------------- UI updates ---


def test_scope_updates_h3_turns_the_sliders_live():
    steps, cfg, sv, sa, cap, w, h, nf = single_scope_updates(H3_CONSTRAINTS, "en")
    assert steps["interactive"] is True and steps["value"] == 20
    assert (steps["minimum"], steps["maximum"], steps["step"]) == (1, 50, 1)
    assert cfg["visible"] is False and cap["visible"] is False
    assert sv["visible"] is True and sv["value"] == 12 and sv["step"] == 0.1
    assert sa["visible"] is True and sa["value"] == 3
    assert nf["label"] == "Frames (17n+5)" and nf["step"] == 17
    assert w["step"] == 32 and h["step"] == 32


def test_scope_updates_ltx_is_the_fixed_look():
    steps, cfg, sv, sa, cap, w, h, nf = single_scope_updates(LTX_CONSTRAINTS, "en")
    assert steps["interactive"] is False and steps["value"] == 8
    assert cfg["visible"] is True and cap["visible"] is True
    assert sv["visible"] is False and sa["visible"] is False
    assert nf["label"] == LABELS["en"]["lbl_frames"] and nf["step"] == 8
    assert w["label"] == LABELS["en"]["lbl_width"] and w["step"] == 64


def test_spill_warning_hidden_for_a_family_without_a_comfort_table():
    cfg = {"limits": {"spill_free_frames": {"1280x768": 257},
                      "comfort_budgets": {"ltx": {"rows": [{}]}}}}
    assert comfort_table_missing(cfg, "h3") is True
    assert comfort_table_missing(cfg, "ltx") is False
    # Unknown family / no comfort_budgets served: the LTX behaviour stays.
    assert comfort_table_missing(cfg, "") is False
    assert comfort_table_missing({"limits": {}}, "h3") is False
    assert compute_spill_warning(1280, 768, 321, cfg, "en", "ltx")["visible"] is True
    assert compute_spill_warning(1280, 768, 321, cfg, "en", "h3")["visible"] is False


# --------------------------------------------------------------- wiring ---


def _demo_with_models(models_json):
    demo = build_ui("http://127.0.0.1:8000", api_key=None)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/status"):
            return httpx.Response(200, json={})
        return httpx.Response(200, json=models_json)

    demo.api._client = httpx.Client(transport=httpx.MockTransport(handler))
    return demo


def test_refresh_fills_the_generate_scope_state():
    demo = _demo_with_models(_models("MiniMaxH3"))
    updates = demo.refresh_model_dropdowns("en", warn=False)
    scope = updates[6]   # base + 4 categories + engine State, then this one
    assert scope == {"engine_family": "h3", "constraints": H3_CONSTRAINTS}


def test_switch_language_keeps_the_h3_frames_label():
    demo = build_ui("http://127.0.0.1:8000", api_key=None)
    registry = demo.label_registry
    updates = demo.switch_language(
        "ja", {}, {"engine_family": "h3", "constraints": H3_CONSTRAINTS})
    frames_idx = next(i for i, (c, k, a) in enumerate(registry)
                      if k == "lbl_frames" and a == "label"
                      and getattr(c, "elem_id", None) == "gen_num_frames")
    assert updates[frames_idx]["label"] == "フレーム数 (17n+5)"
    # Without a scope (and for the Chain tab's own frame fields) the static
    # LTX label stays.
    updates = demo.switch_language("ja", {})
    assert updates[frames_idx]["label"] == LABELS["ja"]["lbl_frames"]


def test_scope_change_is_wired_to_the_sliders_and_reapplies_min_from_labels():
    demo = build_ui("http://127.0.0.1:8000", api_key=None)
    dep = next(d for d in demo.fns.values()
               if getattr(d.fn, "__name__", "") == "on_gen_scope_change")
    labels = [getattr(c, "label", None) for c in dep.outputs]
    for key in ("lbl_steps", "lbl_cfg", "lbl_shift_video", "lbl_shift_audio"):
        assert LABELS["en"][key] in labels
    assert any(isinstance(c, gr.Number) and c.elem_id == "gen_num_frames"
               for c in dep.outputs)
    js = [getattr(d, "js", None) or "" for d in demo.fns.values()]
    assert any("gen_num_frames" in j and "n\\+" in j for j in js)


def test_dispatch_hands_scope_and_sliders_to_generate():
    demo = build_ui("http://127.0.0.1:8000", api_key=None)
    dep = next(d for d in demo.fns.values()
               if getattr(d.fn, "__name__", "") == "dispatch")
    ins = list(dep.inputs)
    labels = [getattr(c, "label", None) for c in ins]
    i_steps = labels.index(LABELS["en"]["lbl_steps"])
    assert isinstance(ins[i_steps - 1], gr.State)          # gen_scope_state
    assert labels[i_steps + 1] == LABELS["en"]["lbl_shift_video"]
    assert labels[i_steps + 2] == LABELS["en"]["lbl_shift_audio"]


# ------------------------------------------- Settings spill-free table ---


def test_settings_spill_table_follows_the_comfort_table():
    from gradio_ui.ui import build_spill_rows

    cfg = {"limits": {"spill_free_frames": {"1280x768": 257},
                      "comfort_budgets": {"ltx": {"rows": [{}]}}}}
    assert build_spill_rows(cfg, "ltx") == [["1280x768", 257]]
    assert build_spill_rows(cfg, "h3") == []
    # Unknown family / no family argument: the LTX rows stay.
    assert build_spill_rows(cfg) == [["1280x768", 257]]
    assert build_spill_rows(cfg, "") == [["1280x768", 257]]


_SPILL_CFG = {"limits": {"spill_free_frames": {"1280x768": 257},
                         "comfort_budgets": {"ltx": {"rows": [{}]}}}}
_H3_SCOPE = {"engine_family": "h3", "constraints": H3_CONSTRAINTS}
_LTX_SCOPE = {"engine_family": "ltx", "constraints": LTX_SERVED}


def _config_demo():
    demo = build_ui("http://127.0.0.1:8000", api_key=None)
    demo.api._client = httpx.Client(transport=httpx.MockTransport(
        lambda _req: httpx.Response(200, json=_SPILL_CFG)))
    return demo


def test_config_fetch_builds_the_settings_table_for_the_loaded_family():
    # The spill table is the 6th output of on_page_load, the 5th of
    # on_config_retry and the 3rd of on_refresh_config.
    demo = _config_demo()
    assert demo.on_page_load({}, "en", _H3_SCOPE)[5]["value"] == []
    assert demo.on_page_load({}, "en", _LTX_SCOPE)[5]["value"] == [["1280x768", 257]]
    # No scope (before the first GET /models): the LTX rows stay.
    assert demo.on_page_load({}, "en")[5]["value"] == [["1280x768", 257]]
    assert demo.on_config_retry({}, 1, "en", _H3_SCOPE)[4]["value"] == []
    assert demo.on_config_retry({}, 1, "en", _LTX_SCOPE)[4]["value"] == [["1280x768", 257]]
    assert demo.on_refresh_config({}, "en", _H3_SCOPE)[2]["value"] == []
    assert demo.on_refresh_config({}, "en", _LTX_SCOPE)[2]["value"] == [["1280x768", 257]]


def test_base_model_change_rebuilds_the_settings_table():
    demo = build_ui("http://127.0.0.1:8000", api_key=None)
    dep = next(d for d in demo.fns.values()
               if getattr(d.fn, "__name__", "") == "_spill_table_for_scope")
    assert dep.fn(_SPILL_CFG, _H3_SCOPE)["value"] == []
    assert dep.fn(_SPILL_CFG, _LTX_SCOPE)["value"] == [["1280x768", 257]]
