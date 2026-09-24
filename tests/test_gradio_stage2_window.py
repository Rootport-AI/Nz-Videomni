"""Clip Chain tab: the stage-2 window dropdown (§3-165).

Covers gradio_ui/comfort.py (window list, budget resolution, recommended size,
labels — pinned against the SAME numbers the WebUI's
``shell/tokenBudget.test.ts`` uses), the chain payload's ``stage2_window`` key,
and the ui.py wiring whose argument layout the new input changed
(``chain_dispatch`` / ``on_chain_estimate`` / ``_make_chain_step`` /
``refresh_model_dropdowns``). MockTransport only — no server, no GPU.
"""

from __future__ import annotations

import json

import gradio as gr
import httpx
import pytest

import chain_math
from gradio_ui import build_ui
from gradio_ui.api_client import ApiClient
from gradio_ui.comfort import (
    STAGE2_WINDOW_CHOICES,
    build_stage2_window_choices,
    chain_comfort_size_16x9,
    effective_acceleration_fields,
    engine_info_from_models,
    resolve_chain_budget,
    stage2_window_choices_for,
    stage2_window_option_label,
    status_availability,
)
from gradio_ui.handlers import make_chain_handler
from gradio_ui.i18n import LABELS
from gradio_ui.presets import CHAIN_MAX_CLIPS, apply_chain_preset, chain_preset_warning

_BASE = "http://127.0.0.1:8000"

_ORDER = ["standard", "high_resolution", "w25", "w28", "w31", "w34", "w37",
          "w40", "w43", "w46", "w49", "w52", "w55", "w58", "w61"]

# Recommended 16:9 size per window at the two served budgets (grid 64).
# Source: the WebUI's chainComfortSize16x9 (shell/tokenBudget.ts) re-computed
# in node for all 30 points (2026-09-25) -- NOT this module's own output. The
# w25/w46/w61 rows (and standard/high_resolution at 40,000) are also the ones
# tokenBudget.test.ts pins.
_SIZES = {
    40000: {
        "standard": (1792, 1024), "high_resolution": (1920, 1088),
        "w25": (1664, 960), "w28": (1600, 896), "w31": (1472, 896),
        "w34": (1408, 832), "w37": (1344, 768), "w40": (1344, 704),
        "w43": (1280, 704), "w46": (1216, 704), "w49": (1216, 640),
        "w52": (1152, 640), "w55": (1088, 640), "w58": (1088, 640),
        "w61": (1088, 576),
    },
    44880: {
        "standard": (1920, 1088), "high_resolution": (2048, 1152),
        "w25": (1792, 1024), "w28": (1664, 960), "w31": (1600, 896),
        "w34": (1536, 832), "w37": (1472, 832), "w40": (1408, 768),
        "w43": (1344, 768), "w46": (1280, 768), "w49": (1280, 704),
        "w52": (1216, 704), "w55": (1216, 640), "w58": (1152, 640),
        "w61": (1152, 640),
    },
}

_LTX_ALL_ON = {
    "attention_backend": "sage", "block_swap_prefetch": True,
    "keep_resident": True, "fused_gguf_dequant_kernel": True,
    "vae_mode": "prune_vaed",
}

# The shape config.py serves (only the fields the resolver reads). The ltx row
# carries a DISTINCT chain budget so a match can be told from the fallback.
_LIMITS = {
    "chain_comfort_token_budget": 40000,
    "comfort_budgets": {
        "ltx": {"rows": [{"requires": dict(_LTX_ALL_ON), "single_budget": 44880,
                          "chain_budget": 41000}]},
        "ltx25": {"rows": [{"requires": {}, "single_budget": 44880,
                            "chain_budget": 44880}]},
    },
}


def _fields(attention="sage", prefetch=True, keep=True, fused=True,
            vae="prune_vaed", kre=False, sage=None, prefetch_avail=None):
    return effective_acceleration_fields(attention, prefetch, keep, fused, vae, kre,
                                         sage_available=sage,
                                         prefetch_available=prefetch_avail)


# --------------------------------------------------------------------------- #
# Window list
# --------------------------------------------------------------------------- #
def test_choices_are_the_chain_math_table_minus_full_length_in_order():
    assert list(STAGE2_WINDOW_CHOICES) == _ORDER
    assert "full_length" not in STAGE2_WINDOW_CHOICES
    assert list(STAGE2_WINDOW_CHOICES) == [
        n for n in chain_math.STAGE2_WINDOW_PRESETS if n != "full_length"]
    values = [v for _label, v in build_stage2_window_choices("en")]
    assert values == _ORDER


def test_dropdown_is_built_with_15_choices_and_standard_default():
    demo = build_ui(_BASE, api_key=None)
    dds = [c for c in demo.blocks.values()
           if isinstance(c, gr.Dropdown) and c.label == LABELS["en"]["lbl_stage2_window"]]
    assert len(dds) == 1
    dd = dds[0]
    assert dd.value == "standard"
    assert [v for _l, v in dd.choices] == _ORDER
    # Built before any /models: no engine name, the 40,000 fallback.
    assert dd.choices[0][0] == "22f (1792×1024)"
    assert dd.info == LABELS["en"]["info_stage2_window"]


# --------------------------------------------------------------------------- #
# Recommended size + label (must equal the WebUI)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("budget", [40000, 44880])
@pytest.mark.parametrize("window", _ORDER)
def test_recommended_size_matches_the_webui(budget, window):
    assert chain_comfort_size_16x9(window, budget) == _SIZES[budget][window]


@pytest.mark.parametrize("budget", [40000, 44880])
@pytest.mark.parametrize("window", _ORDER)
def test_label_matches_the_webui_template(budget, window):
    w, h = _SIZES[budget][window]
    frames = chain_math.STAGE2_WINDOW_PRESETS[window][0]
    assert stage2_window_option_label(
        window, LABELS["en"]["stage2_window_option"], "LTX 2.5", budget,
    ) == f"{frames}f (LTX 2.5 {w}×{h})"
    assert stage2_window_option_label(
        window, LABELS["ja"]["stage2_window_option"], "LTX 2.3", budget,
    ) == f"{frames}f（LTX 2.3 {w}×{h}）"


def test_label_representative_values():
    en = LABELS["en"]["stage2_window_option"]
    ja = LABELS["ja"]["stage2_window_option"]
    assert stage2_window_option_label("w46", ja, "LTX 2.5", 44880) == "46f（LTX 2.5 1280×768）"
    assert stage2_window_option_label("w46", en, "LTX 2.3", 40000) == "46f (LTX 2.3 1216×704)"
    assert stage2_window_option_label("w61", en, "LTX 2.3", 40000) == "61f (LTX 2.3 1088×576)"
    assert stage2_window_option_label("w61", en, "LTX 2.5", 44880) == "61f (LTX 2.5 1152×640)"
    assert stage2_window_option_label("standard", en, "LTX 2.3", 40000) == "22f (LTX 2.3 1792×1024)"
    assert stage2_window_option_label("standard", en, "LTX 2.5", 44880) == "22f (LTX 2.5 1920×1088)"


def test_label_without_engine_drops_the_engine_placeholder():
    assert stage2_window_option_label(
        "w46", LABELS["en"]["stage2_window_option"], "", 40000) == "46f (1216×704)"
    assert stage2_window_option_label(
        "w46", LABELS["ja"]["stage2_window_option"], "", 40000) == "46f（1216×704）"


def test_label_is_frames_only_when_no_size_fits():
    # A budget too small for even one 64px step -> width or height 0.
    assert stage2_window_option_label(
        "w61", LABELS["en"]["stage2_window_option"], "LTX 2.5", 100) == "61f"


# --------------------------------------------------------------------------- #
# Budget resolution (comfortTable.resolveComfortRow + the ?? fallback)
# --------------------------------------------------------------------------- #
def test_budget_first_matching_row():
    assert resolve_chain_budget(_LIMITS, "ltx", _fields()) == 41000
    assert resolve_chain_budget(_LIMITS, "ltx25", _fields(attention="sdpa")) == 44880


def test_budget_no_matching_row_falls_back_to_scalar():
    assert resolve_chain_budget(_LIMITS, "ltx", _fields(vae="default")) == 40000
    limits = dict(_LIMITS, chain_comfort_token_budget=38000)
    assert resolve_chain_budget(limits, "ltx", _fields(vae="default")) == 38000


def test_budget_unknown_engine_or_no_table_falls_back_to_scalar():
    limits = dict(_LIMITS, chain_comfort_token_budget=38000)
    assert resolve_chain_budget(limits, "", _fields()) == 38000
    assert resolve_chain_budget(limits, "other_engine", _fields()) == 38000
    assert resolve_chain_budget({"chain_comfort_token_budget": 38000}, "ltx25",
                                _fields()) == 38000


def test_budget_unusable_numbers_fall_back_to_40000():
    assert resolve_chain_budget(None, "ltx", _fields()) == 40000
    assert resolve_chain_budget({"chain_comfort_token_budget": 0}, "ltx", _fields()) == 40000
    limits = {"comfort_budgets": {"ltx25": {"rows": [{"requires": {}, "chain_budget": -1}]}}}
    assert resolve_chain_budget(limits, "ltx25", _fields()) == 40000


def test_budget_sage_availability_is_folded_in():
    # sage chosen but the server says it is not installed -> runs as sdpa ->
    # the all-on row no longer matches.
    assert resolve_chain_budget(_LIMITS, "ltx", _fields(sage=False)) == 40000
    # unknown (no /status yet) counts as available.
    assert resolve_chain_budget(_LIMITS, "ltx", _fields(sage=None)) == 41000
    assert resolve_chain_budget(_LIMITS, "ltx", _fields(sage=True)) == 41000


def test_keep_resident_is_folded_off_without_prefetch():
    assert _fields(prefetch=False)["keep_resident"] is False
    assert _fields(prefetch_avail=False)["keep_resident"] is False
    assert _fields(prefetch_avail=None)["keep_resident"] is True
    assert resolve_chain_budget(_LIMITS, "ltx", _fields(prefetch_avail=False)) == 40000


def test_engine_and_status_readers():
    models = {"active_base_model": "LTX25", "base_models": [
        {"id": "LTX23", "engine_family": "ltx", "display_name": "LTX 2.3"},
        {"id": "LTX25", "engine_family": "ltx25", "display_name": "LTX 2.5"},
    ]}
    assert engine_info_from_models(models) == ("ltx25", "LTX 2.5")
    assert engine_info_from_models({}) == ("", "")
    assert engine_info_from_models(None) == ("", "")
    assert status_availability({"acceleration": {
        "sage_available": False, "block_swap_prefetch_available": True}}) == (False, True)
    assert status_availability({}) == (None, None)


def test_choices_for_follow_engine_and_acceleration():
    config = {"limits": _LIMITS}
    ltx25 = {"engine_family": "ltx25", "engine_label": "LTX 2.5"}
    ltx = {"engine_family": "ltx", "engine_label": "LTX 2.3"}
    labels = dict((v, lab) for lab, v in stage2_window_choices_for(
        "ja", config, ltx25, "sdpa", True, False, True, "default", False))
    assert labels["w46"] == "46f（LTX 2.5 1280×768）"
    # LTX 2.3 with everything on -> the (test) row's 41,000; with the default
    # VAE -> the 40,000 scalar.
    config_real = {"limits": dict(_LIMITS, comfort_budgets={
        "ltx": {"rows": [{"requires": dict(_LTX_ALL_ON), "chain_budget": 40000}]}})}
    labels = dict((v, lab) for lab, v in stage2_window_choices_for(
        "en", config_real, ltx, "sage", True, True, True, "prune_vaed", False))
    assert labels["w46"] == "46f (LTX 2.3 1216×704)"
    assert labels["w61"] == "61f (LTX 2.3 1088×576)"


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #
def _chain_args(clips=2):
    args = ["Base prompt", "", 1280, 768, False, 0, 0, 24.0, -1, 3, 0.5]
    for i in range(24):
        enabled = i < clips
        if i == 0:
            args.extend([enabled, "", 121, None, 0.8])
        else:
            args.extend([enabled, "", 121])
    args.extend([None, None, None, None, "none", None, 73, False,
                 False, 11.0, 2.5, 0.25, "nag", 1.5, None])
    return args


def _capture_chain(**kwargs):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(409, json={"error": "busy"})

    api = ApiClient("http://test", api_key=None,
                    client=httpx.Client(transport=httpx.MockTransport(handler)))
    list(make_chain_handler(api)(*_chain_args(), **kwargs))
    return captured


def test_payload_omits_the_default_window():
    assert "stage2_window" not in _capture_chain()
    assert "stage2_window" not in _capture_chain(stage2_window="standard")
    assert "stage2_window" not in _capture_chain(stage2_window=None)


def test_payload_carries_a_non_default_window_last():
    captured = _capture_chain(stage2_window="w46")
    assert captured["stage2_window"] == "w46"
    assert list(captured)[-1] == "stage2_window"
    assert _capture_chain(stage2_window="high_resolution")["stage2_window"] == "high_resolution"


def test_precheck_receives_the_selected_window(monkeypatch):
    # The handler forwards the window to check_chain_total as a KEYWORD after
    # lang (validation.py's argument-order note), so the precheck runs the
    # geometry the request will carry.
    import gradio_ui.handlers as handlers
    seen = {}

    def fake_check(clip_frames, fps, kv, lang, **kwargs):
        seen.update(kwargs)
        return "stop here"

    monkeypatch.setattr(handlers, "check_chain_total", fake_check)
    api = ApiClient("http://test", api_key=None, client=httpx.Client(
        transport=httpx.MockTransport(lambda _r: httpx.Response(500))))
    list(make_chain_handler(api)(*_chain_args(), stage2_window="w46"))
    assert seen["stage2_window"] == "w46"


# --------------------------------------------------------------------------- #
# Presets (stage2_window as a keyword after lang)
# --------------------------------------------------------------------------- #
def test_apply_chain_preset_and_warning_accept_the_window():
    cfg = {"generation_presets": {"huge": {"width": 999, "height": 999, "num_frames": 1000}}}
    full = apply_chain_preset("huge", cfg, enabled_flags=[True] * 24, lang="en",
                              stage2_window="w46")
    assert len(full) == 6 + CHAIN_MAX_CLIPS + 1
    warn = chain_preset_warning("huge", cfg, enabled_flags=[True] * 24, lang="en",
                                stage2_window="w46")
    assert warn == full[-1]
    assert warn["visible"] is True


# --------------------------------------------------------------------------- #
# ui.py wiring
# --------------------------------------------------------------------------- #
def _dropdown(demo):
    return next(c for c in demo.blocks.values()
                if isinstance(c, gr.Dropdown)
                and c.label == LABELS["en"]["lbl_stage2_window"])


def test_chain_stage2_window_is_the_last_chain_input():
    demo = build_ui(_BASE, api_key=None)
    dd = _dropdown(demo)
    deps = [d for d in demo.fns.values()
            if getattr(d.fn, "__name__", "") == "chain_dispatch"]
    assert len(deps) == 1
    ins = list(deps[0].inputs)
    assert ins[-1] is dd
    assert ins[-2].label == LABELS["en"]["output_lbl_embed_mp4_metadata"]


def test_chain_dispatch_forwards_the_trailing_values_as_keywords():
    demo = build_ui(_BASE, api_key=None)
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(409, json={"error": "busy"})

    demo.api._client = httpx.Client(transport=httpx.MockTransport(handler))
    # _chain_args minus the positional src_audio (ui.py never wires it), then
    # the 8 keyword-forwarded trailing values in wiring order.
    args = _chain_args()[:-1] + ["sage", False, True, False, "prune_vaed", True,
                                 False, "w61"]
    list(demo.chain_dispatch(*args))
    assert captured["attention_backend"] == "sage"
    assert captured["block_swap_prefetch"] is False
    assert captured["keep_resident"] is True
    assert captured["fused_gguf_dequant_kernel"] is False
    assert captured["vae_mode"] == "prune_vaed"
    assert captured["keep_resident_embeddings"] is True
    assert captured["embed_mp4_metadata"] is False
    assert captured["stage2_window"] == "w61"


def test_chain_estimate_and_step_take_the_window_before_the_slots():
    demo = build_ui(_BASE, api_key=None)
    dd = _dropdown(demo)
    enabled = [True, True] + [False] * (CHAIN_MAX_CLIPS - 2)
    frames = [121] * CHAIN_MAX_CLIPS
    for window in ("standard", "w61"):
        out = demo.on_chain_estimate(2, "en", 24, 3, window, *enabled, *frames)
        assert out["value"]
    step = demo.make_chain_step(+1)(2, "en", 24, 3, "w46", *enabled, *frames)
    assert step[0] == 3
    # Every estimate / step wiring feeds the window at position 4 (right
    # before the 24 enabled flags + 24 frame counts).
    wired = [dep for dep in demo.fns.values()
             if getattr(dep.fn, "__name__", "") in ("on_chain_estimate", "handler")
             and len(dep.inputs) == 5 + 2 * CHAIN_MAX_CLIPS]
    # 48 slot fields + fps + overlap + window + preset .then + language switch
    # for the estimate, and the two ± buttons.
    assert len(wired) >= 2 * CHAIN_MAX_CLIPS + 2
    assert {getattr(d.fn, "__name__", "") for d in wired} == {"on_chain_estimate", "handler"}
    for dep in wired:
        assert dep.inputs[4] is dd


def test_window_preset_warning_handler_runs():
    # Regression: the handler must call the presets FUNCTION, not the
    # same-named gr.Markdown inside build_ui ("'Markdown' object is not
    # callable").
    demo = build_ui(_BASE, api_key=None)
    out = demo.on_chain_window_preset_warning(
        "none", {}, *[True] * CHAIN_MAX_CLIPS, 24.0, 3, "en", "w46")
    assert isinstance(out, dict) and out.get("__type__") == "update"
    assert "visible" in out


def test_window_change_refreshes_estimate_and_preset_warning():
    demo = build_ui(_BASE, api_key=None)
    dd = _dropdown(demo)
    names = {getattr(d.fn, "__name__", "") for d in demo.fns.values()
             if any(t[0] == dd._id and t[1] == "change" for t in d.targets)}
    assert {"on_chain_estimate", "on_chain_window_preset_warning"} <= names


def test_labels_are_rebuilt_on_engine_config_accel_and_language_changes():
    demo = build_ui(_BASE, api_key=None)
    dd = _dropdown(demo)
    relabel = [d for d in demo.fns.values()
               if getattr(d.fn, "__name__", "") == "relabel_stage2_window"]
    # ONE gr.on listener over config_state + chain_engine_state + the 6
    # acceleration controls + lang_dd.
    assert len(relabel) == 1
    assert list(relabel[0].outputs) == [dd]
    assert len(relabel[0].targets) == 9
    upd = demo.relabel_stage2_window(
        "ja", {"limits": _LIMITS}, {"engine_family": "ltx25", "engine_label": "LTX 2.5"},
        "sdpa", True, False, True, "default", False)
    assert dict((v, lab) for lab, v in upd["choices"])["w46"] == "46f（LTX 2.5 1280×768）"


def test_refresh_model_dropdowns_fills_the_engine_state():
    demo = build_ui(_BASE, api_key=None)
    models = {"active_base_model": "LTX25", "base_models": [
        {"id": "LTX23", "engine_family": "ltx", "display_name": "LTX 2.3", "active": False},
        {"id": "LTX25", "engine_family": "ltx25", "display_name": "LTX 2.5", "active": True},
    ]}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/status"):
            return httpx.Response(200, json={"acceleration": {
                "sage_available": False, "block_swap_prefetch_available": True}})
        return httpx.Response(200, json=models)

    demo.api._client = httpx.Client(transport=httpx.MockTransport(handler))
    updates = demo.refresh_model_dropdowns("en", warn=False)
    state = updates[5]   # base + 4 categories, then the engine State
    assert state == {"engine_family": "ltx25", "engine_label": "LTX 2.5",
                     "sage_available": False, "prefetch_available": True}
