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
import json

import httpx
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
        # §3-97 P5: the runtime-state file, in tmp like every other
        # writable location -- never the repository's own state.json.
        "state_file": (tmp_path / "state.json").as_posix(),
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
from gradio_ui.presets import KF_MAX_SLOTS, format_duration_label  # noqa: E402
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


def test_chain_chunked_upsample_checkbox_defaults_on():
    # Owner decision 2026-07-14 (real-GPU 4-clip gate passed): the chunked-
    # upsample checkbox starts ON. UI default only — the API model default
    # stays False for flag-omitting clients.
    demo = _demo()
    boxes = [c for c in demo.blocks.values()
             if isinstance(c, gr.Checkbox)
             and c.label == LABELS["en"]["chk_chunked_upsample"]]
    assert len(boxes) == 1, "chunked-upsample checkbox not found"
    assert boxes[0].value is True


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
    # precision=0 on the fps Number is a real defence, not cosmetics (台帳
    # §3-71 / §3-72): the live duration/spill readouts and the A2V length
    # precheck read this field RAW, so integer-ising it here is what keeps a
    # pasted 29.97 out of chain_math on the UI path. Pin the attribute.
    fps_numbers = [c for c in right_leaves
                   if isinstance(c, gr.Number) and c.label == en["lbl_fps"]]
    assert fps_numbers
    assert fps_numbers[0].precision == 0

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


def test_every_fps_number_is_integer_only_but_keeps_no_server_bounds():
    # 台帳 §3-71 / §3-72: a non-integer fps breaks the chain's stage-2 audio
    # re-assembly check (an ordinary clip list 422s) and makes the mp4 writer
    # truncate the rate, so long clips drift out of audio sync. handlers.py
    # snaps the value it PUTS IN THE PAYLOAD, but several consumers read the
    # field raw — the live duration/spill readouts, the A2V length precheck and
    # the Chain tab's live layout estimate — so precision=0 is what integer-ises
    # the UI path, and the default value must be an int for the same reason.
    #
    # minimum/maximum deliberately stay unset here, exactly like the dimension
    # fields above: a live .change listener would raise inside Number preprocess
    # on an in-progress keystroke. The [1, 60] range is enforced by the handler
    # precheck and, finally, by the server's own Field(ge=1.0, le=60.0).
    demo = _demo()
    fps_numbers = [c for c in demo.blocks.values()
                   if isinstance(c, gr.Number) and c.label == LABELS["en"]["lbl_fps"]]
    assert len(fps_numbers) == 2, "expected the Generate tab's and the Chain tab's fps fields"
    for c in fps_numbers:
        assert c.precision == 0
        assert c.value == 24 and isinstance(c.value, int)
        assert c.minimum is None and c.maximum is None


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
    # WP3: the middle stage is now the batch-aware ``dispatch`` closure, which
    # delegates verbatim to ``generate`` when batch mode is off.
    assert middle.fn is not None and middle.fn.__name__ == "dispatch"
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
    # Acceleration: the middle stage is now the ``chain_dispatch`` closure,
    # which forwards the appended attention selector to ``generate_chain`` as a
    # keyword (same shape as the Generate tab's ``dispatch`` above).
    assert middle.fn is not None and middle.fn.__name__ == "chain_dispatch"
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
    # WP3: the Generate button's restore is batch-aware — (enable, lang). With
    # batch OFF it restores to the plain "Generate" label; the chain button's
    # restore keeps its single-arg (lang) signature.
    assert g_end.fn(False, "en")["value"] == LABELS["en"]["btn_generate"]
    assert c_end.fn("en")["value"] == LABELS["en"]["btn_concat"]
    assert g_end.fn(False, "ja")["value"] == LABELS["ja"]["btn_generate"]
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


# --------------------------------------------------------------------------- #
# 2nd-round FB — modification A: the batch table uses FIXED PIXEL column widths
# (so the table overflows -> horizontal scrollbar) rather than percentages
# (which always fit the viewport and squeeze the Prompt column).
# --------------------------------------------------------------------------- #
def _batch_table(demo):
    from gradio_ui.ui import _BATCH_COL_WIDTHS
    return next(c for c in demo.blocks.values()
                if isinstance(c, gr.Dataframe)
                and getattr(c, "column_widths", None) == _BATCH_COL_WIDTHS)


def _preorder(block):
    order = [block]
    for ch in getattr(block, "children", []) or []:
        order.extend(_preorder(ch))
    return order


def test_batch_table_column_widths_are_fixed_pixels():
    from gradio_ui.ui import _BATCH_COL_WIDTHS

    demo = _demo()
    batch_df = _batch_table(demo)
    assert len(batch_df.column_widths) == 7
    # Every width is a fixed pixel value (not a "%"): with pixels the frontend
    # sizes the table to the SUM of the columns, overflowing the container so
    # its overflow-x:auto shows a horizontal scrollbar.
    assert all(isinstance(w, str) and w.endswith("px")
               for w in batch_df.column_widths)
    px = [int(w[:-2]) for w in batch_df.column_widths]
    # Prompt column (index 4) is the widest and always ≥480px (click-into room),
    # and the total width comfortably exceeds a typical accordion panel so the
    # scrollbar actually appears.
    assert px[4] == max(px) and px[4] >= 480
    assert sum(px) > 1000
    # wrap stays on: long prompts wrap inside the 480px cell (no truncation).
    assert batch_df.wrap is True
    assert batch_df.column_widths == _BATCH_COL_WIDTHS


# --------------------------------------------------------------------------- #
# 2nd-round FB — modification B: the max-duration + summary readouts sit ABOVE
# the batch table (between the "Set audios" button and the table).
# --------------------------------------------------------------------------- #
def test_batch_maxdur_summary_readouts_are_above_the_table():
    demo = _demo()
    acc = next(c for c in demo.blocks.values()
               if isinstance(c, gr.Accordion)
               and c.label == LABELS["en"]["batch_accordion"])
    order = _preorder(acc)
    pos = {id(b): i for i, b in enumerate(order)}

    batch_df = _batch_table(demo)
    maxdur_md = next(b for b in order
                     if isinstance(b, gr.Markdown) and b.value
                     and "max duration" in b.value)
    set_btn = next(b for b in order
                   if isinstance(b, gr.Button)
                   and getattr(b, "value", None) == LABELS["en"]["batch_set_audios"])
    # Order within the accordion: Set audios -> max-duration/summary -> table.
    assert pos[id(set_btn)] < pos[id(maxdur_md)] < pos[id(batch_df)]


# --------------------------------------------------------------------------- #
# 2nd-round FB — modification C: the Regenerate handler refuses a Skip row
# (leaves stat + table untouched) but still re-queues a non-Skip row.
# --------------------------------------------------------------------------- #
def test_batch_regen_refuses_skip_row(tmp_path):
    from gradio_ui.manifest import BatchRow, STAT_DONE, STAT_SKIP, STAT_WAITING

    demo = _demo()
    fn = demo.on_batch_regen

    # A Skip row is refused: stat stays Skip and the table update is a no-op.
    # The reason code here is deliberately the LEGACY "over-481f" (the current
    # writer emits "over-cap"): this keeps an execution path over a manifest
    # written before Docs/PENDING_TASKS_CLOSED.md's old §4-29 (closed
    # 2026-09-01), which the refusal message must still localize.
    skip_rows = [BatchRow(queue=1, wav="a.wav", stat=STAT_SKIP,
                          skip_reason="over-481f")]
    upd, out_rows = fn(0, skip_rows, str(tmp_path), "en")
    assert out_rows[0].stat == STAT_SKIP        # NOT promoted to Waiting
    assert "value" not in upd                    # table left as-is (no re-render)

    # A non-Skip (Done) row IS re-queued to Waiting (control path).
    done_rows = [BatchRow(queue=1, wav="a.wav", stat=STAT_DONE)]
    upd2, out2 = fn(0, done_rows, str(tmp_path), "en")
    assert out2[0].stat == STAT_WAITING
    assert "value" in upd2                        # table re-rendered


def test_batch_skip_key_maps_current_and_legacy_reason_codes():
    """Docs/PENDING_TASKS_CLOSED.md's old §4-29 (closed 2026-09-01): the
    writer emits "over-cap", but manifests written before the change carry
    "over-481f". Both must render as the SAME localized label, so a legacy
    CSV never shows a raw reason code."""
    from gradio_ui.ui import _BATCH_SKIP_KEY, batch_row_info_text
    from gradio_ui.manifest import BatchRow, STAT_SKIP

    assert _BATCH_SKIP_KEY["over-cap"] == _BATCH_SKIP_KEY["over-481f"]
    for lang in ("en", "ja"):
        label = LABELS[lang][_BATCH_SKIP_KEY["over-cap"]]
        for code in ("over-cap", "over-481f"):
            row = BatchRow(queue=1, wav="a.wav", stat=STAT_SKIP, skip_reason=code)
            assert batch_row_info_text(row, lang) == label


def test_batch_regen_skip_i18n_key_present_both_languages():
    for lang in ("en", "ja"):
        assert "batch_msg_regen_skip" in LABELS[lang]
        assert "{reason}" in LABELS[lang]["batch_msg_regen_skip"]
    assert LABELS["en"]["batch_msg_regen_skip"] != LABELS["ja"]["batch_msg_regen_skip"]


# --------------------------------------------------------------------------- #
# 2nd-round FB — modification D: the foolproof prompt reason codes localize.
# --------------------------------------------------------------------------- #
def test_batch_prompt_foolproof_reason_codes_localize():
    from gradio_ui.ui import _batch_start_reason

    for lang in ("en", "ja"):
        assert "batch_msg_prompt_empty_add" in LABELS[lang]
        assert "batch_msg_prompt_rows_empty" in LABELS[lang]
        # "add" reason maps straight through.
        assert _batch_start_reason("prompt-empty-add", lang) == \
            LABELS[lang]["batch_msg_prompt_empty_add"]
        # "replace" reason carries the empty-row count into the text.
        out = _batch_start_reason("prompt-rows-empty:3", lang)
        assert out == LABELS[lang]["batch_msg_prompt_rows_empty"].format(n="3")
        assert "3" in out


# --------------------------------------------------------------------------- #
# NAG (Normalized Attention Guidance / non-CFG Negative): the shared
# accordion above gr.Tabs() replaces the two old per-tab greyed-out Negative
# textboxes with ONE shared textbox + toggle + method radio + 3 sliders.
# --------------------------------------------------------------------------- #
def test_exactly_one_negative_textbox_and_it_is_disabled_by_default():
    demo = _demo()
    negatives = [c for c in demo.blocks.values()
                 if isinstance(c, gr.Textbox)
                 and c.label == LABELS["en"]["lbl_negative"]]
    assert len(negatives) == 1, "the two old per-tab Negative textboxes must be gone"
    assert negatives[0].interactive is False
    assert negatives[0].value == "blurry, low quality, distorted"


def test_nag_accordion_exists_closed_by_default():
    demo = _demo()
    acc = next(c for c in demo.blocks.values()
               if isinstance(c, gr.Accordion)
               and c.label == LABELS["en"]["nag_accordion"])
    assert acc.open is False


def test_nag_sliders_have_expected_defaults_and_ranges():
    demo = _demo()
    sliders = {s.label: s for s in demo.blocks.values() if isinstance(s, gr.Slider)}
    scale = sliders[LABELS["en"]["nag_lbl_scale"]]
    tau = sliders[LABELS["en"]["nag_lbl_tau"]]
    alpha = sliders[LABELS["en"]["nag_lbl_alpha"]]
    assert (scale.minimum, scale.maximum, scale.value) == (1.0, 20.0, 11.0)
    assert (tau.minimum, tau.maximum, tau.value) == (1.0, 10.0, 2.5)
    assert (alpha.minimum, alpha.maximum, alpha.value) == (0.0, 1.0, 0.25)


def test_nag_enable_toggle_flips_negative_textbox_interactivity():
    demo = _demo()
    fn = demo.on_nag_enable_toggle
    assert fn(True)["interactive"] is True
    assert fn(False)["interactive"] is False


def test_nag_method_change_toggles_group_visibility():
    demo = _demo()
    fn = demo.on_nag_method_change
    nag_upd, vsf_upd = fn("vsf")
    assert nag_upd["visible"] is False
    assert vsf_upd["visible"] is True
    nag_upd2, vsf_upd2 = fn("nag")
    assert nag_upd2["visible"] is True
    assert vsf_upd2["visible"] is False


def test_nag_method_choices_translate_on_language_switch():
    demo = _demo()
    registry = demo.label_registry
    nag_method = next(c for c, k, a in registry if k == "nag_lbl_method" and a == "label")

    updates = demo.switch_language("ja", {})
    idx = [c for c, _k, _a in registry].index(nag_method)
    upd = updates[idx]
    assert upd["choices"] == [
        (LABELS["ja"]["nag_method_nag"], "nag"),
        (LABELS["ja"]["nag_method_vsf"], "vsf"),
    ]


def test_vsf_group_hidden_and_nag_group_shown_by_default():
    demo = _demo()
    groups = [c for c in demo.blocks.values() if isinstance(c, gr.Group)]
    nag_groups = [g for g in groups if "nag-group" in (g.elem_classes or [])]
    vsf_groups = [g for g in groups if "vsf-group" in (g.elem_classes or [])]
    assert len(nag_groups) == 1 and len(vsf_groups) == 1
    assert nag_groups[0].visible is True
    assert vsf_groups[0].visible is False


def test_vsf_scale_slider_has_expected_defaults_and_range():
    demo = _demo()
    sliders = {s.label: s for s in demo.blocks.values() if isinstance(s, gr.Slider)}
    scale = sliders[LABELS["en"]["vsf_lbl_scale"]]
    assert (scale.minimum, scale.maximum, scale.value) == (0, 10, 1.5)


def test_vsf_scale_info_registered_and_translates_on_language_switch():
    # vsf_scale carries an `info=` string (Wave 4 review item 7): registered as
    # a SECOND registry entry (same component, attr="info") next to the
    # existing label entry, so switch_language must update both without the
    # components colliding in the outputs list.
    demo = _demo()
    registry = demo.label_registry

    scale_label_entries = [
        (c, k, a) for c, k, a in registry if k == "vsf_lbl_scale" and a == "label"
    ]
    scale_info_entries = [
        (c, k, a) for c, k, a in registry if k == "vsf_lbl_scale_info" and a == "info"
    ]
    assert len(scale_info_entries) == 1
    # Same underlying component as the label registration, not a stray copy.
    assert scale_info_entries[0][0] is scale_label_entries[0][0]

    updates = demo.switch_language("ja", {})
    assert len(updates) == len(registry) + 2  # matches the pinned-arity contract

    scale_info_idx = next(
        i for i, (c, k, a) in enumerate(registry)
        if k == "vsf_lbl_scale_info" and a == "info"
    )
    assert updates[scale_info_idx]["info"] == LABELS["ja"]["vsf_lbl_scale_info"]
    # The label update (same component, earlier registry position) still
    # fires too -- the two registrations don't clobber each other.
    scale_label_idx = next(
        i for i, (c, k, a) in enumerate(registry)
        if k == "vsf_lbl_scale" and a == "label"
    )
    assert updates[scale_label_idx]["label"] == LABELS["ja"]["vsf_lbl_scale"]


# --------------------------------------------------------------------------- #
# Acceleration section (Settings tab). Everything here is real except the VAE
# radio, a DISABLED placeholder that is not wired into any handler input, so it
# can never reach a payload.
# --------------------------------------------------------------------------- #
def test_acceleration_attention_radio_values_and_default():
    demo = _demo()
    en = LABELS["en"]
    radios = [c for c in demo.blocks.values()
              if isinstance(c, gr.Radio) and c.label == en["accel_lbl_attention"]]
    assert len(radios) == 1, "attention radio not found"
    radio = radios[0]
    # VALUES are the API literals; the visible choice strings are fixed,
    # untranslated text (so switch_language needs no extra branch).
    assert [v for _l, v in radio.choices] == ["sdpa", "sage"]
    assert [label for label, _v in radio.choices] == ["sdpa", "sage attention"]
    assert radio.value == "sdpa"
    assert radio.interactive is not False
    assert radio.info == en["accel_info_attention"]


def test_acceleration_no_mock_controls_remain():
    # PrunaVAED (Docs/PENDING_TASKS_CLOSED.md §3-66, filed as §3-50 at the
    # time; real as of 2026-08-05): the VAE radio was the last remaining mock
    # in this section (the old fused_gguf_dequant_gemm checkbox was removed
    # outright on 2026-08-04). It is now a real, wired control -- Acceleration
    # has zero mocks left. The display label was also corrected from
    # "PruneVAED" to "PrunaVAED" (the API value "prune_vaed" is unchanged --
    # an external contract).
    en = LABELS["en"]
    assert "accel_lbl_fused_gguf" not in en, "old mock label must be gone"

    demo = _demo()
    vae = [c for c in demo.blocks.values()
           if isinstance(c, gr.Radio) and c.label == en["accel_lbl_vae"]]
    assert len(vae) == 1, "VAE radio not found"
    assert vae[0].interactive is not False
    assert [v for _l, v in vae[0].choices] == ["default", "prune_vaed"]
    assert [label for label, _v in vae[0].choices] == ["Default", "PrunaVAED"]
    assert vae[0].info == en["accel_info_vae"]


def test_acceleration_labels_switch_language():
    demo = _demo()
    registry = demo.label_registry
    updates = demo.switch_language("ja", {})
    for key, attr in (("accel_section_title", "value"),
                      ("accel_lbl_attention", "label"),
                      ("accel_info_attention", "info"),
                      ("accel_lbl_vae", "label"),
                      ("accel_info_vae", "info")):
        idx = next(i for i, (_c, k, a) in enumerate(registry)
                   if k == key and a == attr)
        assert updates[idx][attr] == LABELS["ja"][key]


def _wiring_inputs(dep):
    """``dep.inputs`` with the Generate flow's TRAILING keyframe components
    (4 x KF_MAX_SLOTS of them) removed.

    Gradio can only hand ``inputs`` to a function as positionals, so the one
    place a variable-length group can live is the very end -- ui.py's
    ``dispatch(..., *kf_flat)``. That makes the Generate flow's raw negative
    indices point at keyframe components instead of the Acceleration block, so
    the canaries below strip that trailing run first and keep the same negative
    index meaning the same thing on the Generate and Chain deps alike.
    """
    ins = list(dep.inputs)
    if getattr(dep.fn, "__name__", "") == "dispatch":
        ins = ins[:-4 * KF_MAX_SLOTS]
    return ins


def test_acceleration_attention_radio_is_wired_into_generate_and_chain():
    # The selector must be an INPUT of both generate flows -- a section that
    # renders but is not wired is exactly the "displayed only" trap.
    demo = _demo()
    en = LABELS["en"]
    radio = next(c for c in demo.blocks.values()
                 if isinstance(c, gr.Radio) and c.label == en["accel_lbl_attention"])
    deps_with_radio = [d for d in demo.fns.values()
                       if radio in getattr(d, "inputs", [])]
    assert len(deps_with_radio) >= 2, "attention radio not wired into 2 flows"
    # And it is the FIFTH-TO-LAST input of each: the APPENDED wiring discipline
    # put it last when it was the only Acceleration control, then the
    # block-swap prefetch checkbox went after it, the keep-resident checkbox
    # after that, the fused-dequant checkbox after that, and the VAE radio
    # (PrunaVAED, Docs/PENDING_TASKS_CLOSED.md §3-66, filed as §3-50 at the
    # time) after that. This index is the canary for a wiring list and a
    # handler signature drifting apart.
    for dep in deps_with_radio:
        assert _wiring_inputs(dep)[-5] is radio


# --------------------------------------------------------------------------- #
# Block-swap prefetch checkbox (Settings tab). Same reg/i18n/wiring pattern as
# the attention selector above; S4 (2026-08-01) flipped the shared
# gradio_ui.handlers.BLOCK_SWAP_PREFETCH_DEFAULT constant to True once the
# real-device gate (bit-exact output + VRAM headroom, G1-G7) passed.
# --------------------------------------------------------------------------- #
def test_block_swap_prefetch_checkbox_default_and_label():
    from gradio_ui.handlers import BLOCK_SWAP_PREFETCH_DEFAULT

    demo = _demo()
    en = LABELS["en"]
    boxes = [c for c in demo.blocks.values()
             if isinstance(c, gr.Checkbox) and c.label == en["accel_lbl_prefetch"]]
    assert len(boxes) == 1, "block-swap prefetch checkbox not found"
    box = boxes[0]
    assert box.value is BLOCK_SWAP_PREFETCH_DEFAULT
    assert box.value is True, "on by default post-S4 (real-device gate passed)"
    assert box.interactive is not False
    assert box.info == en["accel_info_prefetch"]


def test_block_swap_prefetch_labels_switch_language():
    demo = _demo()
    registry = demo.label_registry
    updates = demo.switch_language("ja", {})
    for key, attr in (("accel_lbl_prefetch", "label"),
                      ("accel_info_prefetch", "info")):
        idx = next(i for i, (_c, k, a) in enumerate(registry)
                   if k == key and a == attr)
        assert updates[idx][attr] == LABELS["ja"][key]


# --------------------------------------------------------------------------- #
# keep-resident checkbox (Settings tab, §48). Same reg/i18n/wiring pattern as
# the two controls above, but default OFF -- it parks ~20GB in main memory, so
# the owner's rule is "never on unless asked, and say 64GB+ in the note".
# --------------------------------------------------------------------------- #
def test_keep_resident_checkbox_default_and_label():
    from gradio_ui.handlers import KEEP_RESIDENT_DEFAULT

    demo = _demo()
    en = LABELS["en"]
    boxes = [c for c in demo.blocks.values()
             if isinstance(c, gr.Checkbox)
             and c.label == en["accel_lbl_keep_resident"]]
    assert len(boxes) == 1, "keep-resident checkbox not found"
    box = boxes[0]
    assert box.value is KEEP_RESIDENT_DEFAULT
    assert box.value is False, "off by default (owner decision: ~20GB resident)"
    # NOT gated/disabled: whether the machine has the RAM is not something the
    # server can answer, so the note informs and the user decides.
    assert box.interactive is not False
    assert box.info == en["accel_info_keep_resident"]


def test_keep_resident_checkbox_is_wired_last_into_generate_and_chain():
    demo = _demo()
    en = LABELS["en"]
    box = next(c for c in demo.blocks.values()
               if isinstance(c, gr.Checkbox)
               and c.label == en["accel_lbl_keep_resident"])
    deps = [d for d in demo.fns.values() if box in getattr(d, "inputs", [])]
    assert len(deps) >= 2, "keep-resident checkbox not wired into 2 flows"
    # THIRD-TO-LAST since §1-11 appended the fused-dequant checkbox after it,
    # and PrunaVAED (Docs/PENDING_TASKS_CLOSED.md §3-66, filed as §3-50 at the
    # time) appended the VAE radio after that.
    for dep in deps:
        assert _wiring_inputs(dep)[-3] is box


def test_keep_resident_labels_switch_language():
    demo = _demo()
    registry = demo.label_registry
    updates = demo.switch_language("ja", {})
    for key, attr in (("accel_lbl_keep_resident", "label"),
                      ("accel_info_keep_resident", "info")):
        idx = next(i for i, (_c, k, a) in enumerate(registry)
                   if k == key and a == attr)
        assert updates[idx][attr] == LABELS["ja"][key]


def test_block_swap_prefetch_checkbox_is_wired_into_generate_and_chain():
    # Same "displayed only" trap check as the attention radio: the checkbox
    # must actually be an INPUT of both generate flows.
    demo = _demo()
    en = LABELS["en"]
    box = next(c for c in demo.blocks.values()
               if isinstance(c, gr.Checkbox) and c.label == en["accel_lbl_prefetch"])
    deps_with_box = [d for d in demo.fns.values()
                     if box in getattr(d, "inputs", [])]
    assert len(deps_with_box) >= 2, "prefetch checkbox not wired into 2 flows"
    # And it is the FOURTH-TO-LAST input of each: APPENDED after
    # attention_backend, then the keep-resident checkbox (§48), the
    # fused-dequant checkbox (§1-11) and the VAE radio (PrunaVAED,
    # Docs/PENDING_TASKS_CLOSED.md §3-66, filed as §3-50 at the time) were
    # appended after IT.
    for dep in deps_with_box:
        assert _wiring_inputs(dep)[-4] is box


# --------------------------------------------------------------------------- #
# Fused GGUF dequantization kernel checkbox (Settings tab, §1-11). It INHERITED
# the screen position of the removed fused_gguf_dequant_gemm mock, but unlike
# that placeholder it is a real, wired control with an interactive checkbox.
# Default ON since 2026-08-04 (§51: real-device gates G1-G8 passed and the owner
# approved the flip, which moved api/models.py + this mirror + MCP together).
# --------------------------------------------------------------------------- #
def test_fused_dequant_checkbox_default_and_label():
    from gradio_ui.handlers import FUSED_GGUF_DEQUANT_KERNEL_DEFAULT

    demo = _demo()
    en = LABELS["en"]
    boxes = [c for c in demo.blocks.values()
             if isinstance(c, gr.Checkbox)
             and c.label == en["accel_lbl_fused_dequant"]]
    assert len(boxes) == 1, "fused-dequant checkbox not found"
    box = boxes[0]
    assert box.value is FUSED_GGUF_DEQUANT_KERNEL_DEFAULT
    assert box.value is True, "on by default since the gates passed (§51)"
    # NOT gated client-side: the server degrades to the eager implementation on
    # its own, so a client-side lockout would only be a second, drifting source
    # of truth (same reasoning as the attention selector).
    assert box.interactive is not False
    assert box.info == en["accel_info_fused_dequant"]


def test_fused_dequant_labels_switch_language():
    demo = _demo()
    registry = demo.label_registry
    updates = demo.switch_language("ja", {})
    for key, attr in (("accel_lbl_fused_dequant", "label"),
                      ("accel_info_fused_dequant", "info")):
        idx = next(i for i, (_c, k, a) in enumerate(registry)
                   if k == key and a == attr)
        assert updates[idx][attr] == LABELS["ja"][key]


def test_fused_dequant_checkbox_is_wired_into_generate_and_chain():
    demo = _demo()
    en = LABELS["en"]
    box = next(c for c in demo.blocks.values()
               if isinstance(c, gr.Checkbox)
               and c.label == en["accel_lbl_fused_dequant"])
    deps = [d for d in demo.fns.values() if box in getattr(d, "inputs", [])]
    assert len(deps) >= 2, "fused-dequant checkbox not wired into 2 flows"
    # SECOND-TO-LAST since PrunaVAED (Docs/PENDING_TASKS_CLOSED.md §3-66,
    # filed as §3-50 at the time) appended the VAE radio after it.
    for dep in deps:
        assert _wiring_inputs(dep)[-2] is box


def test_vae_radio_is_wired_into_generate_and_chain():
    # Same "displayed only" trap check as the other Acceleration controls: the
    # radio must actually be an INPUT of both generate flows, and it is now
    # the LAST Acceleration control (appended after fused-dequant).
    demo = _demo()
    en = LABELS["en"]
    radio = next(c for c in demo.blocks.values()
                 if isinstance(c, gr.Radio) and c.label == en["accel_lbl_vae"])
    deps = [d for d in demo.fns.values() if radio in getattr(d, "inputs", [])]
    assert len(deps) >= 2, "VAE radio not wired into 2 flows"
    for dep in deps:
        assert _wiring_inputs(dep)[-1] is radio


def test_generate_and_chain_trailing_inputs_order_is_locked():
    """The last SIX inputs of both generate flows, in exact order.

    ui.py's ``chain_dispatch`` peels the trailing Acceleration values off with
    NEGATIVE indices (``args[:-5]`` + ``args[-5]``..``args[-1]``), so appending
    one more input without shifting every index silently mis-wires the chain
    handler: the values still arrive, just under the wrong parameter names, and
    nothing raises. ``vsf_scale`` is included as the boundary element -- it is
    the last POSITIONAL argument the handler receives, i.e. exactly where the
    ``args[:-5]`` slice must stop.

    The keyframe components are the ONE exception to "everything new is
    appended at the end": Gradio passes ``inputs`` positionally, so the
    variable-length keyframe run (4 x KF_MAX_SLOTS components, collected by
    ``dispatch(*kf_flat)``) has to sit after every scalar input on the Generate
    flow. ``_wiring_inputs`` strips that trailing run so the order asserted
    here is the order both flows really share -- and nothing may be appended to
    the Generate click's ``inputs`` after the keyframes.
    """
    demo = _demo()
    en = LABELS["en"]

    def _one(cls, label_key):
        found = [c for c in demo.blocks.values()
                 if isinstance(c, cls) and c.label == en[label_key]]
        assert len(found) == 1, f"expected exactly one {label_key}"
        return found[0]

    expected = [
        _one(gr.Slider, "vsf_lbl_scale"),
        _one(gr.Radio, "accel_lbl_attention"),
        _one(gr.Checkbox, "accel_lbl_prefetch"),
        _one(gr.Checkbox, "accel_lbl_keep_resident"),
        _one(gr.Checkbox, "accel_lbl_fused_dequant"),
        _one(gr.Radio, "accel_lbl_vae"),
    ]
    deps = [d for d in demo.fns.values()
            if expected[-1] in getattr(d, "inputs", [])]
    assert len(deps) == 2, "expected exactly the generate + chain flows"
    for dep in deps:
        assert _wiring_inputs(dep)[-6:] == expected


# --------------------------------------------------------------------------- #
# Keyframe grid size + trailing wiring shape. The slot ceiling has ONE source
# of truth (config.MAX_CONDITIONING_IMAGES, re-exported as presets.KF_MAX_SLOTS),
# so these two canaries fail the moment the rendered grid or the click wiring
# stops tracking it.
# --------------------------------------------------------------------------- #
def test_generate_tab_renders_kf_max_slots_keyframe_images():
    demo = _demo()
    en = LABELS["en"]
    images = [c for c in demo.blocks.values()
              if isinstance(c, gr.Image) and c.label == en["lbl_kf_image"]]
    assert len(images) == KF_MAX_SLOTS


def test_generate_click_trailing_inputs_are_the_keyframe_slot_quads():
    # dispatch() rebundles its trailing *kf_flat into 4-tuples, so the wiring's
    # last 4 x KF_MAX_SLOTS components must be exactly the repeating
    # (Use checkbox, Image, frame Number, strength Slider) pattern -- in that
    # order, or the tuples would be scrambled.
    demo = _demo()
    dep = next(d for d in demo.fns.values()
               if getattr(d.fn, "__name__", "") == "dispatch")
    tail = list(dep.inputs)[-4 * KF_MAX_SLOTS:]
    assert len(tail) == 4 * KF_MAX_SLOTS
    for i in range(0, len(tail), 4):
        enabled, image, frame, strength = tail[i:i + 4]
        assert isinstance(enabled, gr.Checkbox)
        assert isinstance(image, gr.Image)
        assert isinstance(frame, gr.Number)
        assert isinstance(strength, gr.Slider)


# --------------------------------------------------------------------------- #
# vae_mode (PrunaVAED, Docs/PENDING_TASKS_CLOSED.md §3-66, filed as §3-50 at
# the time) reaching the request payload: single-generate (T2V and A2V
# branches), chain, and batch (which shares the A2V branch's
# build_a2v_chain_payload builder). Same offline ``httpx.MockTransport``
# pattern tests/test_gradio_handlers.py and tests/test_gradio_batch_runner.py
# use to assert on the exact JSON body without a live server.
# --------------------------------------------------------------------------- #
def _make_client(handler, *, api_key: str | None = "secret"):
    from gradio_ui.api_client import ApiClient

    transport = httpx.MockTransport(handler)
    hc = httpx.Client(transport=transport)
    return ApiClient("http://test", api_key=api_key, client=hc)


def _kf_args():
    """generate()'s single ``kf_slots`` argument with every slot empty: a
    KF_MAX_SLOTS-long list of (enabled, image, frame_idx, strength) tuples --
    matches tests/test_gradio_handlers.py's helper of the same purpose."""
    return [(False, None, 0, 0.8)] * KF_MAX_SLOTS


def _run_until_job_started(gen):
    first = next(gen)
    gen.close()
    return first


def test_vae_mode_reaches_single_generate_payload_when_pruned():
    from gradio_ui.handlers import make_generate_handler

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-prunavaed"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "A calm river", "", _kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
        vae_mode="prune_vaed",
    )
    _run_until_job_started(gen)
    assert captured["vae_mode"] == "prune_vaed"
    # Appended last, after the (absent here) Acceleration/NAG blocks.
    assert list(captured.keys())[-1] == "vae_mode"


def test_vae_mode_default_omitted_from_single_generate_payload():
    from gradio_ui.handlers import make_generate_handler

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-default-vae"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "A calm river", "", _kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
    )
    _run_until_job_started(gen)
    assert "vae_mode" not in captured


def _chain_positional_args(clips, config=None):
    """Flatten a small list of {"enabled", "prompt", "frames"} clip dicts into
    generate_chain's 24-slot positional signature (slot 1 additionally carries
    image + strength). Mirrors tests/test_gradio_handlers.py's ``_chain_args``,
    trimmed to just what these tests need."""
    filled = list(clips) + [None] * (24 - len(clips))
    args = ["Base prompt", "", 1280, 768, False, 0, 0, 24.0, -1, 3, 0.5]
    for i, spec in enumerate(filled[:24]):
        spec = spec or {}
        enabled = spec.get("enabled", False)
        p = spec.get("prompt", "")
        frames = spec.get("frames", 121)
        if i == 0:
            args.extend([enabled, p, frames, spec.get("image"), spec.get("strength", 0.8)])
        else:
            args.extend([enabled, p, frames])
    args.append(config)
    return args


def _run_chain_until_started(gen):
    outs = []
    for out in gen:
        outs.append(out)
        if out[1]:
            gen.close()
            break
    return outs


def test_vae_mode_reaches_chain_payload_when_pruned():
    from gradio_ui.handlers import make_chain_handler

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-prunavaed"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    gen = chain(*_chain_positional_args([
        {"enabled": True, "frames": 121},
        {"enabled": True, "frames": 121},
    ]), vae_mode="prune_vaed")
    _run_chain_until_started(gen)
    assert captured["vae_mode"] == "prune_vaed"
    assert list(captured.keys())[-1] == "vae_mode"


def test_vae_mode_default_omitted_from_chain_payload():
    from gradio_ui.handlers import make_chain_handler

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-default-vae"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    gen = chain(*_chain_positional_args([
        {"enabled": True, "frames": 121},
        {"enabled": True, "frames": 121},
    ]))
    _run_chain_until_started(gen)
    assert "vae_mode" not in captured


def test_vae_mode_reaches_batch_payload_when_pruned(tmp_path):
    # The batch runner's A2V rows go through the SAME build_a2v_chain_payload
    # the single-generate A2V branch uses (gradio_ui/batch.py:441-467), with
    # vae_mode snapshotted from BatchSnapshot -- so this exercises the exact
    # wiring dispatch()'s BatchSnapshot(...) construction in ui.py relies on.
    import wave

    from gradio_ui.batch import BatchRunner, BatchSnapshot
    from gradio_ui.manifest import STAT_WAITING, BatchRow

    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    wav_path = wav_dir / "a.wav"
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(100)
        w.writeframes(b"\x00\x00" * 100)
    out_dir = tmp_path / "out"

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/upload/audio"):
            return httpx.Response(200, json={"audio_id": "aud-vae-1"})
        if path.endswith("/generate/chain"):
            captured.update(json.loads(request.content))
            return httpx.Response(202, json={"job_id": "job-batch-vae"})
        if "/jobs/" in path:
            return httpx.Response(200, json={"status": "completed"})
        return httpx.Response(404, json={"error": "unexpected"})

    api = _make_client(handler)
    snap = BatchSnapshot(
        wav_dir=str(wav_dir), out_dir=str(out_dir),
        prompt_common="base", negative="", prompt_mode="add",
        width=512, height=320, crop_output=None, frame_rate=24.0, seed=7,
        loras=[], shared_images=[], use_adapter=False, ref_video_path=None,
        control_adherence=1.0, reference_strength=1.0,
        poll_interval=0.0, poll_timeout_s=30.0,
        vae_mode="prune_vaed",
    )
    rows = [BatchRow(queue=1, wav="a.wav", image="", stat=STAT_WAITING, frames=49)]

    runner = BatchRunner()
    started, _ = runner.start(snap, rows, api, sync=True)
    assert started is True
    assert captured["vae_mode"] == "prune_vaed"


def test_vae_mode_default_omitted_from_batch_payload(tmp_path):
    import wave

    from gradio_ui.batch import BatchRunner, BatchSnapshot
    from gradio_ui.manifest import STAT_WAITING, BatchRow

    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    wav_path = wav_dir / "a.wav"
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(100)
        w.writeframes(b"\x00\x00" * 100)
    out_dir = tmp_path / "out"

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/upload/audio"):
            return httpx.Response(200, json={"audio_id": "aud-vae-2"})
        if path.endswith("/generate/chain"):
            captured.update(json.loads(request.content))
            return httpx.Response(202, json={"job_id": "job-batch-default-vae"})
        if "/jobs/" in path:
            return httpx.Response(200, json={"status": "completed"})
        return httpx.Response(404, json={"error": "unexpected"})

    api = _make_client(handler)
    snap = BatchSnapshot(
        wav_dir=str(wav_dir), out_dir=str(out_dir),
        prompt_common="base", negative="", prompt_mode="add",
        width=512, height=320, crop_output=None, frame_rate=24.0, seed=7,
        loras=[], shared_images=[], use_adapter=False, ref_video_path=None,
        control_adherence=1.0, reference_strength=1.0,
        poll_interval=0.0, poll_timeout_s=30.0,
    )
    rows = [BatchRow(queue=1, wav="a.wav", image="", stat=STAT_WAITING, frames=49)]

    runner = BatchRunner()
    started, _ = runner.start(snap, rows, api, sync=True)
    assert started is True
    assert "vae_mode" not in captured
