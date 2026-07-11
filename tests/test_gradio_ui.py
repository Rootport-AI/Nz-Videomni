"""Smoke test for the Gradio UI mount (spec ch.12).

conftest.py forces LTX_DISABLE_GRADIO=1 for every other test so the suite stays
fast/headless -- which means no test ever exercised mount_gradio()/gradio_ui.py.
This file is the one exception: it explicitly re-enables the UI for a single
app build and checks that /ui actually serves the Gradio app, closing that
blind spot.

Uses the same mock backend as the rest of the suite (no GPU, no models).
"""

from __future__ import annotations

import argparse

import pytest
import yaml

import main


def _make_args(config_path: str) -> argparse.Namespace:
    return argparse.Namespace(
        listen=False, port=None, api_key=None, allow_all_cors=False, config=config_path,
        te_offload=None,
        dit_cpu_load=None,
    )


@pytest.fixture()
def gradio_client(tmp_path, monkeypatch):
    """Same as conftest's `client` fixture, but with the Gradio UI ENABLED."""
    monkeypatch.delenv("LTX_DISABLE_GRADIO", raising=False)

    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {"backend": "mock"},
        "output": {"dir": (tmp_path / "outputs").as_posix()},
        "upload": {"dir": (tmp_path / "uploads").as_posix()},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    app = main.build_app(_make_args(cfg_path.as_posix()))

    # TestClient as a context manager runs FastAPI's startup/shutdown events,
    # which is what tears down Gradio's queue/background threads cleanly
    # (without it, a mounted Gradio app can leave threads dangling between
    # tests).
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


def test_ui_mounts_and_serves_html(gradio_client):
    # /ui (no trailing slash) 307-redirects to /ui/ -- follow_redirects=True
    # (the TestClient default) exercises both hops in one call.
    r = gradio_client.get("/ui")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "LTX-AviUtl2-Bridge" in r.text


def test_ui_config_endpoint(gradio_client):
    r = gradio_client.get("/ui/config")
    assert r.status_code == 200
    body = r.json()
    # Sanity check: it's a real Gradio Blocks config with a component tree,
    # not an empty/broken app.
    assert "components" in body
    assert len(body["components"]) > 0


def test_root_redirects_to_ui_when_mounted(gradio_client):
    r = gradio_client.get("/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/ui"


def test_root_is_json_landing_when_ui_disabled(client):
    # `client` (conftest.py) has LTX_DISABLE_GRADIO=1 -- / must not 404 here.
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["ui"] is None
    assert body["docs"] == "/docs"


# --------------------------------------------------------------------------- #
# WP-UI structure: tab order, removed top buttons, chain-mode 2-choice radio,
# and the presence of the new components (duration line, chain preset dropdown,
# Generate-tab A2V audio, CSS injection). Driven off build_ui() directly (no
# server / mount) so they stay fast and focused on the Blocks graph shape.
# --------------------------------------------------------------------------- #
import gradio as gr  # noqa: E402

from gradio_ui import build_ui  # noqa: E402
from gradio_ui.i18n import LABELS  # noqa: E402
from gradio_ui.presets import format_duration_label  # noqa: E402
from gradio_ui.styles import CUSTOM_CSS  # noqa: E402

_BASE = "http://127.0.0.1:8000"


def _demo():
    return build_ui(_BASE, api_key=None)


def test_tab_order_is_generate_chain_style_jobs_settings():
    demo = _demo()
    tabs = [c.label for c in demo.blocks.values() if isinstance(c, gr.Tab)]
    en = LABELS["en"]
    assert tabs == [en["tab_gen"], en["tab_concat"], en["tab_style_lora"],
                    en["tab_jobs"], en["tab_settings"]]


def test_top_bar_load_unload_buttons_removed():
    demo = _demo()
    en = LABELS["en"]
    button_values = {c.value for c in demo.blocks.values() if isinstance(c, gr.Button)}
    # The Refresh button stays; the top-bar Load/Unload buttons are gone.
    assert en["btn_refresh"] in button_values
    assert en["btn_load_model"] not in button_values
    assert en["btn_unload_model"] not in button_values


def test_chain_mode_radio_is_none_v2v_only():
    demo = _demo()
    radios = [c for c in demo.blocks.values() if isinstance(c, gr.Radio)]
    chain_modes = [r for r in radios if [v for _l, v in r.choices] == ["none", "v2v"]]
    assert chain_modes, "chain-mode none/v2v radio not found"
    # And no radio anywhere still offers the removed a2v choice.
    assert not any("a2v" in [v for _l, v in r.choices] for r in radios)


def test_chain_preset_dropdown_present_with_valid_default():
    demo = _demo()
    dropdowns = [c for c in demo.blocks.values() if isinstance(c, gr.Dropdown)]
    preset_dds = [d for d in dropdowns
                  if d.label == LABELS["en"]["lbl_chain_preset"]]
    assert preset_dds, "chain preset dropdown not found"
    dd = preset_dds[0]
    # Initial value must be one of the choices (no not-in-choices warning).
    assert dd.value in [v for _l, v in dd.choices]


def test_generate_tab_a2v_audio_file_present():
    demo = _demo()
    # The Generate-tab A2V accordion holds an audio gr.File.
    accordions = [c for c in demo.blocks.values() if isinstance(c, gr.Accordion)]
    assert any(a.label == LABELS["en"]["gen_a2v_accordion"] for a in accordions)
    files = [c for c in demo.blocks.values() if isinstance(c, gr.File)]
    assert any(getattr(f, "file_types", None) == ["audio"] for f in files)


def test_duration_panel_fuses_frames_duration_fps_into_one_group():
    # Frames (8n+1) / Duration / Frame rate must live inside a SINGLE
    # gr.Group (elem_classes=["duration-panel"]) so Gradio's BaseForm
    # collapses their individual block borders/backgrounds into one panel,
    # per the owner's mockup (three columns, one dark card).
    demo = _demo()
    groups = [c for c in demo.blocks.values()
              if isinstance(c, gr.Group) and "duration-panel" in (c.elem_classes or [])]
    assert len(groups) == 1, "expected exactly one .duration-panel Group"
    group = groups[0]

    rows = [c for c in group.children if isinstance(c, gr.Row)]
    assert len(rows) == 1, "duration-panel Group must wrap a single Row"
    columns = [c for c in rows[0].children if isinstance(c, gr.Column)]
    assert len(columns) == 3, "duration-panel Row must have 3 columns"

    def _leaf_components(block):
        # Gradio auto-wraps adjacent form components (gr.Number) in an
        # implicit Form block; unwrap one level so leaves are the actual
        # Number/Markdown components.
        leaves = []
        for child in block.children:
            if type(child).__name__ == "Form":
                leaves.extend(child.children)
            else:
                leaves.append(child)
        return leaves

    left_leaves = _leaf_components(columns[0])
    mid_leaves = _leaf_components(columns[1])
    right_leaves = _leaf_components(columns[2])

    en = LABELS["en"]
    assert any(isinstance(c, gr.Number) and c.label == en["lbl_frames"]
               for c in left_leaves)
    assert any(isinstance(c, gr.Number) and c.label == en["lbl_fps"]
               for c in right_leaves)

    # Middle column: a bold "Duration" heading (i18n-frozen, English literal
    # by design) above the value-only Markdown readout -- no input widget.
    assert not any(hasattr(c, "precision") for c in mid_leaves), \
        "middle (Duration) column must not contain an input widget"
    headings = [c for c in mid_leaves
                if isinstance(c, gr.Markdown) and "duration-heading" in (c.elem_classes or [])]
    values = [c for c in mid_leaves
              if isinstance(c, gr.Markdown) and "duration-line" in (c.elem_classes or [])]
    assert headings and "Duration" in headings[0].value
    assert values and values[0].value == format_duration_label(49, 24.0)
    # Column scale: outer (frames/fps) columns wider than the middle one.
    assert columns[0].scale > columns[1].scale
    assert columns[2].scale > columns[1].scale


def test_dimension_frame_numbers_have_no_server_minimum_but_carry_elem_id():
    # Bug fix: width/height (min 64) and frames (min 9) Numbers must NOT set a
    # server-side ``minimum`` — an in-progress sub-minimum keystroke would raise
    # inside Number preprocess on the live duration/spill .change listeners
    # ("Value 8 is less than minimum value 64"). Each still carries an elem_id so
    # the demo.load(js=...) hook can re-apply ``min`` client-side as the HTML
    # step-snap base. step must be preserved (64 / 8).
    demo = _demo()
    by_id = {c.elem_id: c for c in demo.blocks.values()
             if isinstance(c, gr.Number) and getattr(c, "elem_id", None)}
    expected_step = {"gen_width": 64, "gen_height": 64, "chain_width": 64,
                     "chain_height": 64, "gen_num_frames": 8}
    expected_step.update({f"chain_c{i}_frames": 8 for i in range(1, 9)})
    for elem_id, step in expected_step.items():
        assert elem_id in by_id, f"Number missing elem_id={elem_id}"
        c = by_id[elem_id]
        assert c.minimum is None, f"{elem_id} must not set a server-side minimum"
        assert c.step == step, f"{elem_id} step must stay {step}"


def test_load_js_reapplies_min_attribute_client_side():
    # A fn=None demo.load event carries the js that sets the HTML ``min`` on each
    # dimension/frame input (64 for width/height, 9 for the 8n+1 frame fields),
    # keyed off their elem_ids.
    demo = _demo()
    js_blobs = [getattr(fn, "js", None) for fn in demo.fns.values()]
    min_js = [j for j in js_blobs if j and "setAttribute" in j and "min" in j]
    assert min_js, "expected a demo.load js hook that sets the min attribute"
    blob = min_js[0]
    for elem_id in ("gen_width", "gen_height", "chain_width", "chain_height",
                    "gen_num_frames", "chain_c1_frames", "chain_c8_frames"):
        assert elem_id in blob, f"js hook must reference {elem_id}"
    assert "'64'" in blob and "'9'" in blob


def test_duration_panel_css_unifies_background_with_blocks():
    # Owner feedback: the duration panel's background must match the other input
    # panels. The BaseForm wrapper defaults to --border-color-primary; the
    # override repaints .duration-panel with --block-background-fill.
    assert "--block-background-fill" in CUSTOM_CSS
    # And the heading is matched to the Number label metrics (var(--text-sm)).
    assert "--text-sm" in CUSTOM_CSS


def test_custom_css_is_injected_in_tree():
    demo = _demo()
    htmls = [c for c in demo.blocks.values() if isinstance(c, gr.HTML)]
    assert any(c.value and ".negative-greyed" in c.value for c in htmls)
    # sanity: the constant itself carries the expected hooks.
    for hook in (".base-url-box", ".duration-panel", ".duration-col",
                 ".duration-heading", ".duration-line", ".spill-warning"):
        assert hook in CUSTOM_CSS


# --------------------------------------------------------------------------- #
# Feature 1: A2V audio .change wiring (Generate tab).
# --------------------------------------------------------------------------- #
def test_gen_a2v_audio_change_wired_to_num_frames():
    from gradio_ui.handlers import a2v_audio_change_handler

    demo = _demo()
    audio_files = [c for c in demo.blocks.values()
                   if isinstance(c, gr.File) and getattr(c, "file_types", None) == ["audio"]]
    assert audio_files, "Generate-tab A2V audio gr.File not found"
    gen_a2v_audio = audio_files[0]

    num_frames = next(c for c in demo.blocks.values()
                       if isinstance(c, gr.Number) and getattr(c, "elem_id", None) == "gen_num_frames")

    dep = next(v for v in demo.fns.values()
               if v.fn is a2v_audio_change_handler
               and any(t[0] == gen_a2v_audio._id for t in v.targets))
    assert any(o._id == num_frames._id for o in dep.outputs)
    assert any(i._id == gen_a2v_audio._id for i in dep.inputs)


def test_a2v_msg_frames_adjusted_key_translated_both_languages():
    en = LABELS["en"]["a2v_msg_frames_adjusted"]
    ja = LABELS["ja"]["a2v_msg_frames_adjusted"]
    assert en != ja
    assert "{frames}" in en and "{dur" in en
    assert "{frames}" in ja and "{dur" in ja


# --------------------------------------------------------------------------- #
# Feature 3: generate_btn / chain_generate_btn grey-out-while-generating
# click chain -- disable+relabel -> (existing) generate fn -> restore, wired
# with .then() so the restore stage runs even if the generate fn raises.
# --------------------------------------------------------------------------- #
def _click_chain(demo, btn):
    """Resolve the 3-stage click().then().then() chain hanging off ``btn``'s
    click event: (start_dep, middle_dep, end_dep)."""
    start = next(v for v in demo.fns.values()
                 if v.fn is not None and v.fn.__name__ == "on_generate_btn_start"
                 and any(t[0] == btn._id for t in v.targets))
    middle = next(v for v in demo.fns.values() if v.trigger_after == start._id)
    end = next(v for v in demo.fns.values() if v.trigger_after == middle._id)
    return start, middle, end


def test_generate_btn_click_chain_has_disable_generate_restore_stages():
    demo = _demo()
    registry = demo.label_registry
    generate_btn = next(c for c, k, a in registry if k == "btn_generate" and a == "value")

    start, middle, end = _click_chain(demo, generate_btn)
    assert middle.fn is not None and middle.fn.__name__ == "generate"
    assert end.fn is not None and end.fn.__name__ == "_restore"
    # Stage 1 (disable) and stage 3 (restore) both target the SAME button.
    assert any(o._id == generate_btn._id for o in start.outputs)
    assert any(o._id == generate_btn._id for o in end.outputs)
    # Stage 2 (the actual generation) keeps its original outputs untouched.
    assert len(middle.outputs) == 3


def test_chain_generate_btn_click_chain_has_disable_generate_restore_stages():
    demo = _demo()
    registry = demo.label_registry
    chain_btn = next(c for c, k, a in registry if k == "btn_concat" and a == "value")

    start, middle, end = _click_chain(demo, chain_btn)
    assert middle.fn is not None and middle.fn.__name__ == "generate_chain"
    assert end.fn is not None and end.fn.__name__ == "_restore"
    assert any(o._id == chain_btn._id for o in start.outputs)
    assert any(o._id == chain_btn._id for o in end.outputs)
    assert len(middle.outputs) == 3


def test_generate_btn_restore_uses_this_buttons_own_label_key():
    # make_generate_btn_restore is parametrized per-button (btn_generate vs
    # btn_concat) so each button restores to ITS OWN normal label, not the
    # other tab's.
    demo = _demo()
    registry = demo.label_registry
    generate_btn = next(c for c, k, a in registry if k == "btn_generate" and a == "value")
    chain_btn = next(c for c, k, a in registry if k == "btn_concat" and a == "value")

    _gs, _gm, g_end = _click_chain(demo, generate_btn)
    _cs, _cm, c_end = _click_chain(demo, chain_btn)
    assert g_end.fn("en")["value"] == LABELS["en"]["btn_generate"]
    assert c_end.fn("en")["value"] == LABELS["en"]["btn_concat"]
    assert g_end.fn("ja")["value"] == LABELS["ja"]["btn_generate"]
    assert c_end.fn("ja")["value"] == LABELS["ja"]["btn_concat"]


def test_generate_btn_start_stage_disables_and_relabels():
    demo = _demo()
    registry = demo.label_registry
    generate_btn = next(c for c, k, a in registry if k == "btn_generate" and a == "value")
    start, _mid, _end = _click_chain(demo, generate_btn)
    update = start.fn("en")
    assert update["interactive"] is False
    assert update["value"] == LABELS["en"]["btn_generating"]


def test_btn_generating_i18n_key_present_both_languages_and_differs_from_generate():
    for lang in ("en", "ja"):
        assert "btn_generating" in LABELS[lang]
        assert LABELS[lang]["btn_generating"] != LABELS[lang]["btn_generate"]


def test_language_switch_does_not_crash_with_button_mid_generation_relabel():
    # Owner ruling: switching language while a generation is in flight simply
    # snaps the button text back to the CURRENT language's normal label (its
    # `interactive` state is untouched by switch_language, since the registry
    # only ever sets the "value" kwarg for these buttons) -- verify this
    # combination does not raise / produce an inconsistent registry update.
    demo = _demo()
    registry = demo.label_registry
    generate_btn = next(c for c, k, a in registry if k == "btn_generate" and a == "value")

    updates = demo.switch_language("ja", {})
    idx = [c for c, _k, _a in registry].index(generate_btn)
    btn_update = updates[idx]
    assert btn_update["value"] == LABELS["ja"]["btn_generate"]
    assert "interactive" not in btn_update


def test_adapter_change_toggles_reference_video_interactivity():
    # Change A: the reference-video input tracks the adapter selection. "None"
    # (ADAPTER_NONE) / an empty selection greys it out AND clears any uploaded
    # file; a real (control) adapter re-enables it without touching its value.
    from gradio_ui.adapters import ADAPTER_NONE

    demo = _demo()
    fn = demo.on_adapter_change

    off = fn(ADAPTER_NONE)
    assert off["interactive"] is False
    assert off["value"] is None  # stale upload cleared on de-select

    off_empty = fn("")
    assert off_empty["interactive"] is False
    assert off_empty["value"] is None

    on = fn("union-control")  # any real (non-None) adapter name
    assert on["interactive"] is True
    assert "value" not in on  # must NOT clobber an uploaded reference video
