"""UI assembly: ``build_ui()`` assembles the top common bar + gr.Tabs
(Generate / Clip Chain / Jobs / Settings). Placeholder tabs are filled in by
later slices (S3-S6).
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from .adapters import ADAPTER_NONE, build_adapter_choices
from .adapters import (
    MODEL_CATEGORIES,
    MODEL_DEFAULT,
    active_base_model,
    build_base_model_choices,
    build_model_choices,
    build_style_gallery,
    model_active_value,
    style_lora_names,
)
from .api_client import ApiClient
from .formatting import (
    build_jobs_rows,
    format_job_error,
    format_status,
    jobs_table_headers,
)
from .handlers import (
    BLOCK_SWAP_PREFETCH_DEFAULT,
    FUSED_GGUF_DEQUANT_KERNEL_DEFAULT,
    KEEP_RESIDENT_DEFAULT,
    a2v_audio_change_handler,
    delete_finished_jobs,
    fetch_config_safe,
    make_chain_handler,
    make_generate_handler,
    make_join_handler,
    on_config_retry_tick,
)
from .handlers import fetch_models_safe, load_selected_models
from .handlers import (
    _LORA_TOKEN_RE,
    _combine_generate_loras,
    _resolve_fps,
    _resolve_poll,
    parse_prompt_loras,
    suggest_frames_for_audio,
)
from . import manifest as batch_manifest
from .batch import BatchSnapshot, STATE_IDLE, get_runner
from .manifest import (
    IMAGE_SHARED,
    MAX_FRAMES,
    STAT_DONE,
    STAT_FAILED,
    STAT_GENERATING,
    STAT_SKIP,
    STAT_WAITING,
)
from .i18n import L
from .presets import (
    CHAIN_MAX_CLIPS,
    CHAIN_MIN_OPEN,
    KF_MAX_SLOTS,
    KF_MIN_OPEN,
    PRESETS,
    apply_chain_preset,
    apply_preset,
    build_preset_choices,
    compute_chain_duration_label,
    compute_spill_warning,
    format_duration_label,
    pick_default_preset,
    slot_step_state,
)
from .styles import CUSTOM_CSS


def build_spill_rows(config: dict | None) -> list[list]:
    """Rows for the Settings spill-free table: [resolution, max comfortable
    frames] built from /config limits.spill_free_frames."""
    spill = ((config or {}).get("limits") or {}).get("spill_free_frames") or {}
    return [[res, frames] for res, frames in spill.items()]


# --------------------------------------------------------------------------- #
# Batch A2V (WP3) — pure display/formatting helpers. Kept module-level (not in
# batch.py, which is frozen) so the UI wiring below stays terse; none of them
# touch Gradio state, so they are trivially reusable + testable.
# --------------------------------------------------------------------------- #
# Image extensions offered in the per-row image dropdown (mirrors the Generate
# tab's gr.Image filepath input's practical set).
_BATCH_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")

# Initial per-column widths for the 7-column batch table (gr.Dataframe
# ``column_widths``, supported in Gradio 6.19). FIXED PIXEL widths (not
# percentages): with pixels the frontend sizes the header table to the SUM of
# the column widths (Gradio 6.19 Index-*.js: ``d.style.width = `${G}px``` where
# G = Σ column px), so once that sum (here 1270px) exceeds the accordion's width
# the ``.table-wrap`` container (``overflow-x:auto`` in the compiled Dataframe
# CSS) shows a HORIZONTAL SCROLLBAR instead of squeezing every column. A
# percentage list, by contrast, is always resolved as a fraction of the viewport
# width (``v/100*viewport``) so it can never overflow — which is exactly why the
# old percentages shrank the Prompt column rather than scrolling. ``wrap=True``
# is kept, so the 480px Prompt column is always clickable and only text longer
# than 480px wraps inside its own cell.
# Columns: # / Audio / dur. / Image / Prompt / Status / Output.
_BATCH_COL_WIDTHS = ["50px", "220px", "70px", "140px", "480px", "90px", "220px"]

# Row-status -> (emoji, i18n key). The emoji is theme-independent (no CSS
# colouring, which is fragile across Gradio themes); the label localizes.
_BATCH_STAT_DISPLAY = {
    STAT_WAITING: ("⚪", "batch_stat_waiting"),      # white circle
    STAT_GENERATING: ("⏳", "batch_stat_generating"),  # hourglass
    STAT_DONE: ("✅", "batch_stat_done"),            # check mark
    STAT_FAILED: ("❌", "batch_stat_failed"),        # cross mark
    STAT_SKIP: ("⛔", "batch_stat_skip"),            # no-entry
}

# scan_wav_folder's skip_reason strings -> localized i18n key. "over-481f" is
# the LEGACY code this GUI wrote before the cap became DURATION-linked
# (Docs/PENDING_TASKS_CLOSED.md's old §4-29, closed 2026-09-01); manifests
# written back then still carry it, so it maps to the same label as the
# current "over-cap" instead of falling through as a raw string.
_BATCH_SKIP_KEY = {
    "over-cap": "batch_skip_overcap",
    "over-481f": "batch_skip_overcap",
    "wav-only-alpha": "batch_skip_wavonly",
}


def _batch_stat_cell(stat: str, lang: str) -> str:
    emoji, key = _BATCH_STAT_DISPLAY.get(stat, ("", None))
    label = L(key, lang) if key else stat
    return f"{emoji} {label}".strip()


def batch_rows_to_table(rows, lang: str) -> list[list]:
    """Render a list[BatchRow] into the 7-column Dataframe value
    (queue# / Audio / Duration / Image / Prompt / Status / Output)."""
    table: list[list] = []
    for r in rows:
        dur = f"{r.duration_s:.2f}s" if r.duration_s else ""
        table.append([
            r.queue, r.wav, dur, r.image or IMAGE_SHARED, r.prompt,
            _batch_stat_cell(r.stat, lang), r.output,
        ])
    return table


def batch_table_headers(lang: str) -> list[str]:
    return [L("batch_col_queue", lang), L("batch_col_wav", lang),
            L("batch_col_dur", lang), L("batch_col_image", lang),
            L("batch_col_prompt", lang), L("batch_col_stat", lang),
            L("batch_col_output", lang)]


def batch_summary_text(rows, lang: str) -> str:
    counts = {STAT_DONE: 0, STAT_FAILED: 0, STAT_SKIP: 0, STAT_WAITING: 0,
              STAT_GENERATING: 0}
    for r in rows:
        if r.stat in counts:
            counts[r.stat] += 1
    return L("batch_msg_summary", lang).format(
        done=counts[STAT_DONE], failed=counts[STAT_FAILED],
        skip=counts[STAT_SKIP], waiting=counts[STAT_WAITING])


def batch_maxdur_text(width, height, fps, config, lang: str) -> str:
    """The "max duration: 481f (Ns)" line, plus the comfortable (spill-free)
    frame ceiling for the CURRENT resolution when /config advertises one."""
    fps_v = _resolve_fps(fps)
    txt = L("batch_maxdur", lang).format(frames=MAX_FRAMES, secs=MAX_FRAMES / fps_v)
    try:
        spill = ((config or {}).get("limits") or {}).get("spill_free_frames") or {}
        key = f"{int(width)}x{int(height)}"
        thr = spill.get(key)
        if thr:
            txt = f"{txt} — {key}: {thr}f"
    except (TypeError, ValueError):
        pass
    return txt


def batch_image_choices(img_dir) -> list[str]:
    """Dropdown choices for the per-row image: the "Shared" sentinel first,
    then every image file in ``img_dir`` (by name). Empty / missing folder ->
    just the sentinel."""
    choices = [IMAGE_SHARED]
    if img_dir:
        try:
            p = Path(str(img_dir))
            if p.is_dir():
                for f in sorted(p.iterdir()):
                    if f.is_file() and f.suffix.lower() in _BATCH_IMAGE_EXTS:
                        choices.append(f.name)
        except OSError:
            pass
    return choices


def batch_row_info_text(row, lang: str) -> str:
    """The lower edit-panel status line for the selected row: localized
    skip-reason + any error message."""
    parts: list[str] = []
    if row.skip_reason:
        key = _BATCH_SKIP_KEY.get(row.skip_reason)
        parts.append(L(key, lang) if key else row.skip_reason)
    if row.error:
        parts.append(row.error)
    return " / ".join(parts)


def _batch_start_reason(msg: str, lang: str) -> str:
    """Map BatchRunner.start()'s internal reason codes onto the localized
    batch_msg_* text where one exists; otherwise pass it through."""
    if msg.startswith("shared keyframe"):
        return L("batch_msg_no_common_image", lang)
    if msg.startswith("batch already running"):
        return L("batch_msg_already_running", lang)
    if msg.startswith("no rows to process"):
        return L("batch_msg_no_rows", lang)
    if msg.startswith("wav folder not found"):
        return L("batch_msg_wav_dir_missing", lang)
    # Foolproof preflight (empty common prompt).
    if msg == "prompt-empty-add":
        return L("batch_msg_prompt_empty_add", lang)
    if msg.startswith("prompt-rows-empty:"):
        n = msg.split(":", 1)[1]
        return L("batch_msg_prompt_rows_empty", lang).format(n=n)
    return msg


def build_ui(base_url: str, api_key: str | None = None) -> gr.Blocks:
    api = ApiClient(base_url, api_key=api_key)
    generate = make_generate_handler(api)
    chain_generate = make_chain_handler(api)
    chain_join = make_join_handler(api)

    # Component registry: (component, label_key, attr). S6 iterates this to
    # implement live language switching. attr is the gr.update field to set.
    registry: list[tuple[object, str, str]] = []

    def reg(component, key: str, attr: str = "label"):
        registry.append((component, key, attr))
        return component

    # --- top-bar handlers ---
    def refresh_status() -> str:
        try:
            return format_status(api.get_status())
        except Exception as exc:
            return L("status_error").format(err=exc)

    def unload_model() -> str:
        try:
            api.unload_pipeline()
        except Exception as exc:
            return L("status_error").format(err=exc)
        return refresh_status()

    def on_page_load(current_cfg, lang):
        status = refresh_status()
        cfg, err = fetch_config_safe(api, lang)
        if err is not None:
            # Single fetch failed (races server startup, transient error, ...).
            # Surface it instead of silently clobbering with {} forever, keep
            # whatever config_state already held, and arm the retry timer so
            # a single missed fetch never strands the session.
            gr.Warning(err)
            return (status, current_cfg, gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(active=True))
        preset_update = gr.update(choices=build_preset_choices(cfg),
                                  value=pick_default_preset(cfg))
        # Rebuild the adapter choices from /config model.ic_loras; keep the
        # current value (ADAPTER_NONE "None", which is always the first choice).
        adapter_update = gr.update(choices=build_adapter_choices(cfg))
        # Settings: populate the raw-config viewer + spill-free table.
        # NOTE: chain_preset choices are NOT returned here -- they are refreshed
        # by a dedicated config_state.change listener (see events) so this
        # closure keeps its exact return arity (frozen by test_gradio_handlers).
        return (status, cfg, preset_update, adapter_update,
                gr.update(value=cfg), gr.update(value=build_spill_rows(cfg)),
                gr.update(active=False))

    def on_config_retry(current_cfg, attempt, lang):
        # One tick of the Settings-tab auto-retry gr.Timer, armed by
        # on_page_load (or a previous tick) after a failed /config fetch.
        # Keeps retrying until success or CONFIG_RETRY_MAX_ATTEMPTS, then
        # disables itself either way (success: silently; exhausted: with a
        # final gr.Warning pointing at the manual Refresh button).
        cfg, err, next_attempt, keep_retrying = on_config_retry_tick(api, attempt, lang)
        if err is not None:
            if not keep_retrying:
                gr.Warning(err)
            return (current_cfg, gr.update(), gr.update(), gr.update(), gr.update(),
                    next_attempt, gr.update(active=keep_retrying))
        return (cfg,
                gr.update(choices=build_preset_choices(cfg), value=pick_default_preset(cfg)),
                gr.update(choices=build_adapter_choices(cfg)),
                gr.update(value=cfg),
                gr.update(value=build_spill_rows(cfg)),
                next_attempt, gr.update(active=False))

    def on_refresh_config(current_cfg, lang):
        # Top-bar Refresh also refreshes the Settings config viewer + spill table
        # and rebuilds preset/adapter choices. On fetch failure everything is
        # left as-is (only the status line, refreshed separately, changes) and
        # a gr.Warning surfaces the failure instead of clobbering good state.
        cfg, err = fetch_config_safe(api, lang)
        if err is not None:
            gr.Warning(err)
            return current_cfg, gr.update(), gr.update(), gr.update(), gr.update()
        return (cfg,
                gr.update(value=cfg),
                gr.update(value=build_spill_rows(cfg)),
                gr.update(choices=build_preset_choices(cfg)),
                gr.update(choices=build_adapter_choices(cfg)))

    def on_qmode_change(value: str):
        # two_stage_hq is not yet consumed by the backend (ltx_runner ignores
        # pipeline/guidance_scale). Revert to distilled and warn.
        if value == "two_stage_hq":
            gr.Warning(L("warn_hq_unsupported"))
            return gr.update(value="distilled")
        return gr.update()

    def on_crop_toggle(enabled: bool):
        return gr.update(visible=bool(enabled))

    # Feature 3: grey out + relabel generate_btn / chain_generate_btn while a
    # generation is in flight, via a .click().then(generate).then(restore)
    # chain — the restore step runs unconditionally (gradio's default .then()
    # semantics: "regardless of success or failure" of the preceding step),
    # so a failed/raising generation still re-enables the button. The button's
    # normal label lives in the language-switch registry (reg(..., "value"));
    # the restore step looks it up by ``label_key`` so it matches whichever
    # label the OTHER button uses (btn_generate vs btn_concat).
    def on_generate_btn_start(lang):
        return gr.update(interactive=False, value=L("btn_generating", lang))

    def make_generate_btn_restore(label_key: str):
        def _restore(lang):
            return gr.update(interactive=True, value=L(label_key, lang))
        return _restore

    with gr.Blocks(title="LTX-AviUtl2-Bridge") as demo:
        # CUSTOM_CSS wiring. NOTE: gr.Blocks(css=...) is deprecated in Gradio 6
        # AND, because the app is mounted via gr.mount_gradio_app() (main.py)
        # rather than launched, mount_gradio_app unconditionally overwrites
        # blocks.css (routes.py: ``blocks.css = css or ""``) -- so a constructor
        # css= would be silently dropped. Injecting the stylesheet as an in-tree
        # gr.HTML <style> block is the mount-safe equivalent: it lives in the
        # component config and is applied on the page regardless of the mount
        # path, with no deprecation warning.
        gr.HTML(f"<style>{CUSTOM_CSS}</style>", padding=False)
        # /config is fetched on page load and stashed for later slices (presets,
        # limits, ic_loras, etc.).
        config_state = gr.State({})
        # Current UI language (S6). Fed as a runtime input to the generate/chain
        # handlers so flow messages localize, and updated by the Language switch.
        lang_state = gr.State("en")
        # Settings /config auto-retry (bug fix): counts failed attempts since
        # the timer was armed; the timer itself starts inactive and is only
        # switched on by on_page_load / on_config_retry after a fetch failure.
        config_retry_state = gr.State(0)
        config_retry_timer = gr.Timer(3.0, active=False)

        gr.Markdown("# LTX-AviUtl2-Bridge")
        reg(gr.Markdown(L("app_subtitle")), "app_subtitle", "value")

        # ---- top common bar (always visible) ----
        with gr.Row():
            status_box = gr.Textbox(
                label="", value="", interactive=False, show_label=False,
                container=False, scale=4, elem_classes=["status-line"],
            )
            refresh_btn = reg(gr.Button(L("btn_refresh"), scale=0), "btn_refresh", "value")

        # ---- unified draft prompt (always visible, above the tabs) ----
        # Single source of truth for BOTH the Generate and Clip Chain flows:
        # wired DIRECTLY into generate_btn.click and chain_generate_btn.click
        # (no hidden mirror to copy from, so no copy-order race / stale value).
        # Style-LoRA thumbnails append their <lora:...> token here too, so a
        # selection is visible from whichever tab the user is on.
        prompt = reg(gr.Textbox(label=L("lbl_prompt"), lines=3,
                                placeholder=L("ph_prompt")), "lbl_prompt")

        # ---- shared Negative Prompt / NAG accordion (always visible, above the
        # tabs) ----
        # NAG (Normalized Attention Guidance) is the ONLY way a negative prompt
        # has any effect: the distilled model is frozen at CFG=1, so a plain
        # negative_prompt is otherwise a no-op. One negative prompt is shared by
        # Generate / Clip Chain / Batch A2V (same rationale as the shared prompt
        # box above), so this accordion sits outside gr.Tabs() rather than being
        # duplicated per tab. The textbox variable name stays ``negative`` (the
        # pre-NAG Generate-tab component's name) so the existing generate_btn
        # inputs list / dispatch signature below need no renumbering.
        with gr.Accordion(L("nag_accordion"), open=False) as nag_accordion:
            reg(nag_accordion, "nag_accordion", "label")
            reg(gr.Markdown(L("nag_note"), elem_classes=["note"]), "nag_note", "value")
            negative = reg(gr.Textbox(label=L("lbl_negative"),
                                      value="blurry, low quality, distorted",
                                      interactive=False,
                                      info=L("info_negative"),
                                      elem_classes=["negative-greyed"]),
                           "lbl_negative")
            nag_enabled = reg(gr.Checkbox(value=False, label=L("nag_enable")), "nag_enable")
            # Two non-CFG-negative methods: NAG (attention run twice, blended)
            # and VSF (Value Sign Flip -- positive/negative contexts concatenated
            # into ONE attention pass, negative V sign-flipped). Each method's
            # parameter group is shown/hidden by on_nag_method_change below.
            nag_method = reg(gr.Radio(
                choices=[(L("nag_method_nag"), "nag"), (L("nag_method_vsf"), "vsf")],
                value="nag", label=L("nag_lbl_method"),
            ), "nag_lbl_method")
            with gr.Group(elem_classes=["nag-group"]) as nag_group:
                with gr.Row():
                    nag_scale = reg(gr.Slider(1.0, 20.0, value=11.0, step=0.5,
                                              label=L("nag_lbl_scale")), "nag_lbl_scale")
                    nag_tau = reg(gr.Slider(1.0, 10.0, value=2.5, step=0.05,
                                            label=L("nag_lbl_tau")), "nag_lbl_tau")
                    nag_alpha = reg(gr.Slider(0.0, 1.0, value=0.25, step=0.01,
                                              label=L("nag_lbl_alpha")), "nag_lbl_alpha")
            with gr.Group(visible=False, elem_classes=["vsf-group"]) as vsf_group:
                vsf_scale = reg(gr.Slider(
                    0, 10, value=1.5, step=0.1, label=L("vsf_lbl_scale"),
                    info=L("vsf_lbl_scale_info"),
                ), "vsf_lbl_scale")
                reg(vsf_scale, "vsf_lbl_scale_info", "info")

        with gr.Tabs():
            # ============================ Generate ============================
            with gr.Tab(L("tab_gen")) as tab_gen:
                reg(tab_gen, "tab_gen", "label")
                with gr.Row():
                    # LEFT: inputs
                    with gr.Column(scale=3):
                        # Prompt lives in the shared draft box above the tabs.
                        # Negative prompt / NAG now live in the shared accordion
                        # above gr.Tabs() (see ``negative`` there).

                        # quality mode (two_stage_hq is non-selectable in S1)
                        qmode = reg(gr.Radio(
                            choices=[(L("qmode_fast"), "distilled"),
                                     (L("qmode_hq"), "two_stage_hq")],
                            value="distilled", label=L("lbl_qmode"),
                        ), "lbl_qmode")

                        preset = reg(gr.Dropdown(list(PRESETS.keys()), value="minimal",
                                                 label=L("lbl_preset"), info=L("hint_preset")),
                                     "lbl_preset")
                        with gr.Row():
                            # NOTE: no server-side ``minimum=`` here. A live
                            # .change listener (duration/spill) preprocesses the
                            # value on every keystroke, and Gradio's Number
                            # preprocess raises "Value N is less than minimum" for
                            # any in-progress sub-minimum digit (e.g. "8" while
                            # typing "80"), surfacing as a queue/join error. The
                            # min is re-applied CLIENT-SIDE only, as the HTML input
                            # ``min`` attribute (arrow-key step-snap base), by the
                            # demo.load(js=...) hook below — keyed off elem_id.
                            width = reg(gr.Number(value=512, label=L("lbl_width"), precision=0,
                                                  step=64, elem_id="gen_width"),
                                        "lbl_width")
                            height = reg(gr.Number(value=320, label=L("lbl_height"), precision=0,
                                                   step=64, elem_id="gen_height"),
                                         "lbl_height")

                        crop_enabled = reg(gr.Checkbox(value=False, label=L("chk_crop")), "chk_crop")
                        with gr.Row(visible=False) as crop_row:
                            crop_w = reg(gr.Number(value=0, label=L("lbl_crop_w"), precision=0),
                                         "lbl_crop_w")
                            crop_h = reg(gr.Number(value=0, label=L("lbl_crop_h"), precision=0),
                                         "lbl_crop_h")

                        # Frames / Duration / Frame-rate merged into ONE panel:
                        # gr.Group fuses its children's block chrome (border /
                        # shadow / radius all collapse to 0 on Gradio's
                        # BaseForm wrapper) into a single card -- the exact
                        # same technique already used for the Clip Chain slot
                        # panels below, just horizontal here via Row+Column
                        # instead of the slots' vertical stacking. Duration
                        # itself carries no input: a bold "Duration" heading
                        # (deliberately un-i18n'd -- i18n.py is frozen) sits
                        # above the accent-coloured "N.NNs" readout, centred
                        # in a narrower middle column so it reads as a
                        # standout summary rather than a third input field.
                        with gr.Group(elem_classes=["duration-panel"]):
                            with gr.Row():
                                with gr.Column(scale=3, min_width=0):
                                    # minimum omitted server-side (see width note);
                                    # client min=9 applied via demo.load(js=...).
                                    num_frames = reg(gr.Number(
                                        value=49, label=L("lbl_frames"), precision=0,
                                        step=8, elem_id="gen_num_frames"), "lbl_frames")
                                with gr.Column(scale=2, min_width=0,
                                               elem_classes=["duration-col"]):
                                    gr.Markdown("**Duration**",
                                               elem_classes=["duration-heading"])
                                    # Live "N.NNs" readout derived from frames / fps.
                                    duration_md = gr.Markdown(
                                        format_duration_label(49, 24.0),
                                        elem_classes=["duration-line"])
                                with gr.Column(scale=3, min_width=0):
                                    # precision=0 is a real defence, not cosmetics
                                    # (§3-71 / §3-72 — see handlers._snap_frame_rate):
                                    # it makes Gradio hand every downstream consumer a
                                    # whole number, including the live duration/spill
                                    # readouts and the A2V length precheck, which read
                                    # the raw field rather than the snapped payload.
                                    # minimum/maximum stay omitted server-side for the
                                    # same reason as the width field above.
                                    frame_rate = reg(gr.Number(
                                        value=24, label=L("lbl_fps"), precision=0), "lbl_fps")

                        spill_warning = gr.Markdown("", visible=False,
                                                    elem_classes=["spill-warning"])

                        with gr.Row():
                            steps = reg(gr.Slider(1, 50, value=8, step=1, label=L("lbl_steps"),
                                                  interactive=False), "lbl_steps")
                            cfg = reg(gr.Slider(1.0, 12.0, value=1.0, step=0.1, label=L("lbl_cfg"),
                                                interactive=False), "lbl_cfg")
                        cap_lock = reg(gr.Markdown(L("cap_lock")), "cap_lock", "value")

                        seed = reg(gr.Number(value=-1, label=L("lbl_seed"), precision=0), "lbl_seed")

                        # accordion: KF_MAX_SLOTS fixed keyframe slots (I2V
                        # multi-keyframe conditioning; the ceiling is the server's
                        # MAX_CONDITIONING_IMAGES, re-exported by presets.py). Slot 1
                        # suggests "start frame (0)"; the remaining slots are plain
                        # "frame position" slots, all defaulting to 0 (the server snaps
                        # any non-zero value to the 8n+1 grid).
                        kf_slots: list[tuple[object, object, object, object]] = []
                        kf_extra_rows: list[object] = []
                        # Collapsible keyframe grid: slot 1 always visible (the
                        # start frame / batch-A2V required reference), the rest
                        # start hidden and the ± buttons grow/shrink the count.
                        # State is server-side (Gradio has no live-visibility read).
                        kf_open_count = gr.State(KF_MIN_OPEN)
                        with gr.Accordion(L("lbl_kf_accordion"), open=False) as kf_accordion:
                            reg(kf_accordion, "lbl_kf_accordion", "label")
                            for _slot_i in range(1, KF_MAX_SLOTS + 1):
                                frame_key = "lbl_kf_frame_pos0" if _slot_i == 1 else "lbl_kf_frame_pos"
                                with gr.Row(visible=(_slot_i == 1)) as kf_row:
                                    kf_enabled = reg(gr.Checkbox(value=False, label=L("lbl_kf_use")),
                                                     "lbl_kf_use")
                                    kf_image = reg(gr.Image(label=L("lbl_kf_image"), type="filepath"),
                                                   "lbl_kf_image")
                                    kf_frame = reg(gr.Number(value=0, label=L(frame_key), precision=0,
                                                             minimum=0), frame_key)
                                    kf_strength = reg(gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                                               label=L("lbl_kf_strength")),
                                                     "lbl_kf_strength")
                                kf_slots.append((kf_enabled, kf_image, kf_frame, kf_strength))
                                if _slot_i > 1:
                                    kf_extra_rows.append(kf_row)
                            # ± buttons (symbols only, i18n non-registered) grow /
                            # shrink the visible keyframe rows (min KF_MIN_OPEN, max
                            # KF_MAX_SLOTS). "−" also unchecks a hidden row's Use box
                            # and greys out at the floor (the startup state, hence
                            # interactive=False here); "＋" greys out at the
                            # KF_MAX_SLOTS ceiling. Owner-requested order: ＋ left,
                            # − right, plus a language-independent "n/KF_MAX_SLOTS"
                            # counter (digits only -> not i18n-registered) on the
                            # same row.
                            with gr.Row():
                                kf_plus_btn = gr.Button("＋", scale=0)
                                kf_minus_btn = gr.Button("−", scale=0,
                                                         interactive=False)
                                kf_counter_md = gr.Markdown(
                                    f"{KF_MIN_OPEN}/{KF_MAX_SLOTS}")
                            reg(gr.Markdown(L("cap_kf_grid")), "cap_kf_grid", "value")

                        # accordion: reference-video control (IC-LoRA). The
                        # adapter Dropdown's choices are rebuilt on page load
                        # from /config model.ic_loras (static list is the
                        # offline fallback). gr.File(type="filepath") gives a
                        # reliable local path for arbitrary containers
                        # (.mp4/.mov/.webm/.mkv), unlike gr.Video which may
                        # re-encode/preview.
                        with gr.Accordion(L("lbl_iclora_accordion"), open=False) as iclora_accordion:
                            reg(iclora_accordion, "lbl_iclora_accordion", "label")
                            adapter = reg(gr.Dropdown(
                                choices=build_adapter_choices(None),
                                value=ADAPTER_NONE, label=L("lbl_adapter"),
                            ), "lbl_adapter")
                            adapter_strength = reg(gr.Slider(
                                0.05, 2.0, value=1.0, step=0.05,
                                label=L("lbl_adapter_strength"),
                            ), "lbl_adapter_strength")
                            control_adherence = reg(gr.Slider(
                                0.0, 1.0, value=1.0, step=0.05,
                                label=L("lbl_control_adherence"),
                                info=L("info_control_adherence"),
                            ), "lbl_control_adherence")
                            reference_strength_slider = reg(gr.Slider(
                                0.0, 1.0, value=1.0, step=0.05,
                                label=L("lbl_reference_strength"),
                                info=L("info_reference_strength"),
                            ), "lbl_reference_strength")
                            ref_video = reg(gr.File(
                                label=L("lbl_ref_video"), type="filepath",
                                file_count="single", file_types=["video"],
                                # The default adapter is ADAPTER_NONE ("None"), so
                                # the reference-video input starts greyed out; it is
                                # re-enabled by on_adapter_change when a real
                                # (control) adapter is selected.
                                interactive=False,
                            ), "lbl_ref_video")
                            reg(gr.Markdown(L("note_ref128"),
                                            elem_classes=["note"]), "note_ref128", "value")
                            # Static hints for the newer control adapters (depth-control,
                            # deblur). No adapter-conditional show/hide mechanism exists
                            # (only ref_video's enabled state tracks the selection via
                            # on_adapter_change), so these stay always-visible notes like
                            # note_ref128 above rather than adding a new UI mechanism.
                            reg(gr.Markdown(L("note_iclora_aspect"),
                                            elem_classes=["note"]), "note_iclora_aspect", "value")
                            reg(gr.Markdown(L("note_iclora_depth"),
                                            elem_classes=["note"]), "note_iclora_depth", "value")
                            reg(gr.Markdown(L("note_iclora_deblur"),
                                            elem_classes=["note"]), "note_iclora_deblur", "value")

                        # accordion: Audio-to-Video (案A). Attaching an audio
                        # file routes generate() down the A2V path (src_audio,
                        # the handler's last positional input). Style LoRAs
                        # (<lora:> tokens), keyframe images, AND the
                        # reference-video CONTROL adapter above (dropdown +
                        # ref_video) can all combine with audio -- this handler
                        # always sends a single-clip chain, and a chain now
                        # accepts a reference_video_id on any clip count
                        # (1..24 -- owner decision 2026-08-11), so a 1-clip A2V
                        # chain is unaffected either way. All three are wired
                        # into the chain payload below (``loras`` +
                        # ``reference_video_id`` / S3 strength keys).
                        with gr.Accordion(L("gen_a2v_accordion"), open=False) as gen_a2v_accordion:
                            reg(gen_a2v_accordion, "gen_a2v_accordion", "label")
                            reg(gr.Markdown(L("gen_a2v_note"), elem_classes=["note"]),
                                "gen_a2v_note", "value")
                            gen_a2v_audio = reg(gr.File(
                                label=L("a2v_lbl_audio"), type="filepath",
                                file_count="single", file_types=["audio"],
                            ), "a2v_lbl_audio")

                        # accordion: Batch A2V (WP3). A wav folder -> CSV
                        # manifest -> a queue table the in-process BatchRunner
                        # (batch.py) drives one clip at a time for unattended
                        # overnight runs. The heavy lifting (scan/merge/CSV,
                        # payload, execution) lives in manifest.py / batch.py;
                        # this accordion is only the wiring + a Timer that reads
                        # the runner's state back on a tick.
                        # Runtime state (not visible components):
                        #   batch_rows_state — the canonical list[BatchRow]
                        #   batch_sel_state  — the selected row index (edit panel)
                        #   batch_timer      — polls the runner while it runs
                        batch_rows_state = gr.State([])
                        batch_sel_state = gr.State(None)
                        batch_timer = gr.Timer(2.0, active=False)
                        with gr.Accordion(L("batch_accordion"), open=False) as batch_accordion:
                            reg(batch_accordion, "batch_accordion", "label")
                            # Enable panel — the batch toggle, the source folder
                            # paths, and the output-location controls, fused into
                            # one card (gr.Group) so they read as the "set up the
                            # batch" section.
                            with gr.Group():
                                batch_enable = reg(gr.Checkbox(
                                    value=False, label=L("batch_enable")), "batch_enable")
                                batch_wav_dir = reg(gr.Textbox(
                                    label=L("batch_wav_dir")), "batch_wav_dir")
                                batch_img_dir = reg(gr.Textbox(
                                    label=L("batch_img_dir")), "batch_img_dir")
                                batch_out_mode = reg(gr.Radio(
                                    choices=[(L("batch_out_auto"), "auto"),
                                             (L("batch_out_custom"), "custom")],
                                    value="auto", label=L("batch_out_mode"),
                                ), "batch_out_mode")
                                batch_out_dir = reg(gr.Textbox(
                                    label=L("batch_out_dir"), visible=False),
                                    "batch_out_dir")
                            # Prompt mode sits just above "Set audios".
                            batch_add_replace = reg(gr.Radio(
                                choices=[(L("batch_mode_add"), "add"),
                                         (L("batch_mode_replace"), "replace")],
                                value="add", label=L("batch_add_replace"),
                            ), "batch_add_replace")
                            batch_set_btn = reg(gr.Button(L("batch_set_audios")),
                                                "batch_set_audios", "value")
                            # Info readouts sit between "Set audios" and the
                            # table: the max-duration line and the live batch
                            # summary, side by side (owner feedback — they read
                            # as the batch's headline numbers above the queue).
                            with gr.Row():
                                batch_maxdur_md = gr.Markdown(
                                    batch_maxdur_text(512, 320, 24.0, None, "en"),
                                    elem_classes=["note"])
                                batch_summary_md = gr.Markdown("")
                            batch_table = gr.Dataframe(
                                headers=batch_table_headers("en"),
                                datatype="str", column_count=(7, "fixed"),
                                interactive=True,
                                static_columns=[0, 1, 2, 3, 5, 6],
                                wrap=True, max_height=360, value=[],
                                column_widths=_BATCH_COL_WIDTHS,
                            )
                            # lower edit panel — acts on the selected row.
                            with gr.Group():
                                batch_img_dd = reg(gr.Dropdown(
                                    choices=[IMAGE_SHARED], value=IMAGE_SHARED,
                                    label=L("batch_row_image")), "batch_row_image")
                                batch_row_info_md = gr.Markdown(
                                    "", elem_classes=["note"])
                            batch_copy_common_btn = reg(gr.Button(
                                L("batch_copy_common")), "batch_copy_common", "value")
                            batch_regen_btn = reg(gr.Button(
                                L("batch_regen_row")), "batch_regen_row", "value")
                            batch_stop_btn = reg(gr.Button(
                                L("batch_stop"), variant="stop"), "batch_stop", "value")

                    # RIGHT: action panel (Generate first) -> progress -> job id -> video
                    with gr.Column(scale=2):
                        generate_btn = reg(gr.Button(L("btn_generate"), variant="primary"),
                                           "btn_generate", "value")
                        progress_box = reg(gr.Textbox(label=L("lbl_progress"), interactive=False),
                                           "lbl_progress")
                        job_box = reg(gr.Textbox(label=L("lbl_jobid"), interactive=False), "lbl_jobid")
                        video_out = reg(gr.Video(label=L("lbl_result")), "lbl_result")

            # =========================== Clip Chain ==========================
            # One continuous masked AV-latent timeline (POST /generate/chain):
            # shared params + join (overlap) params on the left, 8 FIXED clip
            # slots (only slot 1 carries a start image = conditioning on clip 0),
            # and the action panel / output trio on the right.
            with gr.Tab(L("tab_concat")) as tab_concat:
                reg(tab_concat, "tab_concat", "label")
                chain_clip_slots: list[tuple] = []
                with gr.Row():
                    # LEFT: shared + join params + clip list
                    with gr.Column(scale=3):
                        # ---- generation mode (none / V2V / A2V) ----
                        # ONE radio makes the V2V x A2V exclusivity structural
                        # (mirrors the API's source_video x source_audio 422);
                        # each mode's panel is shown only while selected.
                        chain_mode = reg(gr.Radio(
                            choices=[(L("v2v_mode_none"), "none"),
                                     (L("v2v_mode_v2v"), "v2v")],
                            value="none", label=L("v2v_mode_label"),
                        ), "v2v_mode_label")
                        reg(gr.Markdown(L("v2v_cap_mode"), elem_classes=["note"]),
                            "v2v_cap_mode", "value")

                        # ---- V2V panel (visible in v2v mode only) ----
                        # gr.File(type="filepath") for the source video — same
                        # rationale as the IC-LoRA reference video (reliable
                        # local path, no gr.Video preview re-encode).
                        with gr.Group(visible=False) as v2v_group:
                            v2v_video = reg(gr.File(
                                label=L("v2v_lbl_video"), type="filepath",
                                file_count="single", file_types=["video"],
                            ), "v2v_lbl_video")
                            # 8n+1 grid: start 25, step 8 -> 25, 33, ... 145
                            # (the static max mirrors the server default
                            # limits.v2v_context_frames_max; the precheck uses
                            # the live /config value).
                            v2v_context = reg(gr.Slider(
                                25, 145, value=73, step=8,
                                label=L("v2v_lbl_context"),
                            ), "v2v_lbl_context")
                            # F4: usage guide, same rank/placement as a2v_guide.
                            reg(gr.Markdown(L("v2v_guide"), elem_classes=["note"]),
                                "v2v_guide", "value")
                            reg(gr.Markdown(L("v2v_cap_panel"), elem_classes=["note"]),
                                "v2v_cap_panel", "value")
                            v2v_join_chk = reg(gr.Checkbox(value=True,
                                                           label=L("v2v_chk_join")),
                                               "v2v_chk_join")
                            # F5: crossfade length for the joined version
                            # (JoinRequest.handle_crossfade_ms; server default 300).
                            v2v_crossfade = reg(gr.Dropdown(
                                choices=[("150 ms", 150), ("300 ms", 300),
                                         ("500 ms", 500)],
                                value=300, label=L("v2v_lbl_crossfade"),
                            ), "v2v_lbl_crossfade")
                            reg(gr.Markdown(L("v2v_cap_join"), elem_classes=["note"]),
                                "v2v_cap_join", "value")

                        # A2V for the Clip Chain flow now lives on the Generate
                        # tab's Audio-to-Video accordion (案A). The chain A2V
                        # handler/i18n keys are kept; only this UI entry point
                        # was removed.

                        # Shared prompt lives in the draft box above the tabs
                        # (it is the common base for every clip). Negative
                        # prompt / NAG now live in the shared accordion above
                        # gr.Tabs() (see ``negative`` there) -- this tab reuses
                        # that SAME component; there is no chain_negative.
                        chain_qmode = reg(gr.Radio(
                            choices=[(L("qmode_fast"), "distilled"),
                                     (L("qmode_hq"), "two_stage_hq")],
                            value="distilled", label=L("lbl_qmode"),
                        ), "lbl_qmode")
                        # Chain preset: fills resolution/crop + a recommended
                        # per-clip length into all 24 slots (apply_chain_preset).
                        chain_preset = reg(gr.Dropdown(
                            list(PRESETS.keys()), value="minimal",
                            label=L("lbl_chain_preset"), info=L("info_chain_preset"),
                        ), "lbl_chain_preset")
                        chain_preset_warning = gr.Markdown(
                            "", visible=False, elem_classes=["spill-warning"])
                        with gr.Row():
                            # minimum omitted server-side (see Generate width note);
                            # client min=64 applied via demo.load(js=...).
                            chain_width = reg(gr.Number(value=1280, label=L("lbl_width"),
                                                        precision=0, step=64,
                                                        elem_id="chain_width"),
                                              "lbl_width")
                            chain_height = reg(gr.Number(value=768, label=L("lbl_height"),
                                                         precision=0, step=64,
                                                         elem_id="chain_height"),
                                               "lbl_height")
                        chain_crop_enabled = reg(gr.Checkbox(value=False, label=L("chk_crop")),
                                                 "chk_crop")
                        with gr.Row(visible=False) as chain_crop_row:
                            chain_crop_w = reg(gr.Number(value=0, label=L("lbl_crop_w"),
                                                         precision=0), "lbl_crop_w")
                            chain_crop_h = reg(gr.Number(value=0, label=L("lbl_crop_h"),
                                                         precision=0), "lbl_crop_h")
                        with gr.Row():
                            # precision=0: same real defence as the Generate tab's
                            # fps field (§3-71 / §3-72 — see the note there and
                            # handlers._snap_frame_rate). Here it also guards the
                            # live chain estimate (presets.compute_chain_layout),
                            # which reads this field raw. minimum/maximum stay
                            # omitted server-side (see the Generate width note).
                            chain_fps = reg(gr.Number(value=24, label=L("lbl_fps"),
                                                      precision=0), "lbl_fps")
                            chain_steps = reg(gr.Slider(1, 50, value=8, step=1, label=L("lbl_steps"),
                                                        interactive=False), "lbl_steps")
                            chain_cfg = reg(gr.Slider(1.0, 12.0, value=1.0, step=0.1,
                                                      label=L("lbl_cfg"), interactive=False),
                                            "lbl_cfg")
                        reg(gr.Markdown(L("cap_lock")), "cap_lock", "value")
                        chain_seed = reg(gr.Number(value=-1, label=L("lbl_seed"), precision=0),
                                         "lbl_seed")

                        # join (overlap) params — the cross-fade-like blend.
                        with gr.Row():
                            chain_overlap = reg(gr.Slider(1, 8, value=3, step=1,
                                                          label=L("lbl_overlap")), "lbl_overlap")
                            chain_overlap_strength = reg(gr.Slider(
                                0.0, 1.0, value=0.5, step=0.05, label=L("lbl_overlap_strength"),
                            ), "lbl_overlap_strength")
                        reg(gr.Markdown(L("cap_crossfade")), "cap_crossfade", "value")

                        # Default ON (owner decision 2026-07-14, after the real-GPU
                        # 4-clip gate: complete, seam-free, 9311MB peak). UI-side
                        # default only — the API model keeps False so flag-omitting
                        # clients (WebView2 etc.) are unchanged; Gradio always sends
                        # the checkbox's actual value explicitly.
                        chain_chunked_upsample = reg(
                            gr.Checkbox(value=True, label=L("chk_chunked_upsample")),
                            "chk_chunked_upsample",
                        )

                        # clip list: 24 fixed slots (slots 1-2 shown by default;
                        # the ± buttons grow/shrink the visible count). The open
                        # count lives in a gr.State because Gradio does not expose
                        # a component's live visibility to the server.
                        chain_open_count = gr.State(CHAIN_MIN_OPEN)
                        reg(gr.Markdown(f"### {L('h_clips')}"), "h_clips", "value")
                        # Estimated total duration for the enabled clips. NOT
                        # label-registered (dynamic Markdown, like batch_maxdur_md);
                        # a dedicated lang_dd.change listener re-formats it.
                        chain_duration_md = gr.Markdown(
                            compute_chain_duration_label(
                                [True, True] + [False] * (CHAIN_MAX_CLIPS - 2),
                                [121] * CHAIN_MAX_CLIPS, 24.0, 3, "en"))
                        # slot 1 — the only slot with a start image (clip 0).
                        with gr.Group():
                            c1_enabled = reg(gr.Checkbox(value=True, label=L("clip1")), "clip1")
                            c1_prompt = reg(gr.Textbox(label=L("lbl_clip_prompt"),
                                                       placeholder=L("ph_clip_prompt")),
                                            "lbl_clip_prompt")
                            with gr.Row():
                                c1_frames = reg(gr.Number(value=121, label=L("lbl_frames"),
                                                          precision=0, step=8,
                                                          elem_id="chain_c1_frames"),
                                                "lbl_frames")
                                c1_image = reg(gr.Image(label=L("lbl_clip_start_image"),
                                                        type="filepath"), "lbl_clip_start_image")
                                c1_strength = reg(gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                                            label=L("lbl_kf_strength")),
                                                  "lbl_kf_strength")
                            reg(gr.Markdown(L("cap_first_clip")), "cap_first_clip", "value")
                        chain_clip_slots.append((c1_enabled, c1_prompt, c1_frames,
                                                 c1_image, c1_strength))
                        # slots 2-24 — no start image (later timeline segments).
                        # Each is wrapped in a gr.Group so the ± buttons can
                        # hide/show it non-destructively (gr.update(visible=...)
                        # keeps the field values). Only slot 2's group is visible
                        # at startup (chain_open_count == 2).
                        chain_extra_groups: list[object] = []
                        for _slot_i in range(2, CHAIN_MAX_CLIPS + 1):
                            clip_key = f"clip{_slot_i}"
                            with gr.Group(visible=(_slot_i == 2)) as chain_slot_group:
                                cN_enabled = reg(gr.Checkbox(value=(_slot_i == 2),
                                                             label=L(clip_key)), clip_key)
                                cN_prompt = reg(gr.Textbox(label=L("lbl_clip_prompt"),
                                                           placeholder=L("ph_clip_prompt")),
                                                "lbl_clip_prompt")
                                cN_frames = reg(gr.Number(value=121, label=L("lbl_frames"),
                                                          precision=0, step=8,
                                                          elem_id=f"chain_c{_slot_i}_frames"),
                                                "lbl_frames")
                            chain_clip_slots.append((cN_enabled, cN_prompt, cN_frames))
                            chain_extra_groups.append(chain_slot_group)
                        # ± buttons (symbols only, i18n non-registered): grow /
                        # shrink the visible clip count (min 2, max 24). "＋" also
                        # checks the newly-revealed slot's Use box on (owner
                        # request) and greys out at the 24-slot ceiling; "−"
                        # unchecks the newly-hidden slots' Use box so they are
                        # never submitted (the handler collects every ENABLED
                        # slot regardless of visibility) and greys out at the
                        # 2-slot floor (the startup state, hence
                        # interactive=False here). Owner-requested order: ＋ left,
                        # − right, plus an "n/24" counter (digits only -> not
                        # i18n-registered) — all mirroring the keyframe grid.
                        with gr.Row():
                            chain_plus_btn = gr.Button("＋", scale=0)
                            chain_minus_btn = gr.Button("−", scale=0,
                                                        interactive=False)
                            chain_counter_md = gr.Markdown(
                                f"{CHAIN_MIN_OPEN}/{CHAIN_MAX_CLIPS}")
                        reg(gr.Markdown(L("cap_clip_count")), "cap_clip_count", "value")

                    # RIGHT: action panel (Generate chain first) -> outputs
                    with gr.Column(scale=2):
                        chain_generate_btn = reg(gr.Button(L("btn_concat"), variant="primary"),
                                                 "btn_concat", "value")
                        chain_progress = reg(gr.Textbox(label=L("lbl_progress"), interactive=False),
                                             "lbl_progress")
                        chain_job = reg(gr.Textbox(label=L("lbl_jobid"), interactive=False),
                                        "lbl_jobid")
                        chain_video = reg(gr.Video(label=L("lbl_result")), "lbl_result")

                        # V2V-only: server-side join of the completed
                        # continuation back onto the source video
                        # (POST /jobs/{id}/join). Shown in v2v mode only.
                        with gr.Group(visible=False) as v2v_join_panel:
                            v2v_join_btn = reg(gr.Button(L("v2v_btn_join")),
                                               "v2v_btn_join", "value")
                            v2v_join_msg = gr.Textbox(label="", show_label=False,
                                                      interactive=False, container=False)
                            chain_joined_video = reg(gr.Video(label=L("v2v_lbl_joined")),
                                                     "v2v_lbl_joined")

            # =========================== Style LoRA ==========================
            # Style/character LoRA browser (S2). GET /loras -> gallery of the
            # kind=="style" adapters only (thumbnails served by the API);
            # selecting one appends a <lora:name:1.0> token to the Generate-tab
            # prompt. Control LoRAs (canny/pose/upscaler) are NOT shown here —
            # they stay in the Generate tab's reference-video adapter field.
            with gr.Tab(L("tab_style_lora")) as tab_style_lora:
                reg(tab_style_lora, "tab_style_lora", "label")
                # Style-LoRA names parallel to the gallery order, so a gallery
                # select index resolves 1:1 to a name (mirrors jobs_ids_state).
                style_names_state = gr.State([])
                reg(gr.Markdown(L("style_note"), elem_classes=["note"]),
                    "style_note", "value")
                style_reload_btn = reg(gr.Button(L("style_reload_btn")),
                                       "style_reload_btn", "value")
                style_gallery = gr.Gallery(
                    label=L("style_gallery_label"), columns=4, height="auto",
                    show_label=True, interactive=False, value=[],
                )
                reg(style_gallery, "style_gallery_label", "label")

            # ============================== Jobs =============================
            # GET /jobs list -> Dataframe; row select -> GET /jobs/{id} detail +
            # (completed) video + (failed) localized error; [Cancel / Delete] ->
            # DELETE /jobs/{id} (cancel if active, delete if terminal).
            with gr.Tab(L("tab_jobs")) as tab_jobs:
                reg(tab_jobs, "tab_jobs", "label")
                # Row order mirrors jobs_table; used to resolve a selected row
                # index -> job id (the Dataframe select event gives an index).
                jobs_ids_state = gr.State([])
                selected_job_state = gr.State(None)

                jobs_refresh_btn = reg(gr.Button(L("btn_jobs_refresh")),
                                       "btn_jobs_refresh", "value")
                jobs_table = gr.Dataframe(
                    headers=jobs_table_headers("en"),
                    datatype="str", column_count=(6, "fixed"),
                    interactive=False, wrap=True, value=[],
                )
                job_detail_json = reg(gr.JSON(label=L("lbl_job_detail")),
                                      "lbl_job_detail", "label")
                job_error_box = reg(gr.Textbox(label=L("lbl_job_error"),
                                               interactive=False, visible=False,
                                               lines=3),
                                    "lbl_job_error")
                job_video = reg(gr.Video(label=L("lbl_done_video")),
                                "lbl_done_video")
                with gr.Row():
                    job_action_btn = reg(gr.Button(L("btn_job_action"),
                                                   variant="stop"),
                                         "btn_job_action", "value")
                    job_action_msg = gr.Textbox(label="", show_label=False,
                                                interactive=False, container=False)

            # ============================ Settings ===========================
            with gr.Tab(L("tab_settings")) as tab_settings:
                reg(tab_settings, "tab_settings", "label")

                # ---- Interface (language + theme) ----
                reg(gr.Markdown(f"### {L('h_ui')}"), "h_ui", "value")
                with gr.Row():
                    lang_dd = reg(gr.Dropdown(
                        choices=[("English", "en"), ("日本語", "ja")],
                        value="en", label=L("lbl_lang"),
                    ), "lbl_lang")
                    theme_dd = reg(gr.Dropdown(
                        choices=[(L("opt_dark"), "dark"), (L("opt_light"), "light")],
                        value="dark", label=L("lbl_theme"),
                    ), "lbl_theme")

                # ---- Connection (base_url + api-key badge; both build-time) ----
                reg(gr.Markdown(f"### {L('h_conn')}"), "h_conn", "value")
                base_url_box = reg(gr.Textbox(value=base_url, label=L("lbl_base_url"),
                                              interactive=False, buttons=["copy"],
                                              elem_classes=["base-url-box"]), "lbl_base_url")
                api_badge = reg(
                    gr.Markdown(L("badge_set") if api_key else L("badge_unset")),
                    "badge_set" if api_key else "badge_unset", "value",
                )

                # ---- Behavior (poll cadence -> generate/chain handlers) ----
                reg(gr.Markdown(f"### {L('h_behavior')}"), "h_behavior", "value")
                with gr.Row():
                    poll_interval = reg(gr.Number(value=1.0, label=L("lbl_poll"),
                                                  minimum=0.1), "lbl_poll")
                    poll_timeout = reg(gr.Number(value=120, label=L("lbl_timeout"),
                                                 precision=0, minimum=1), "lbl_timeout")

                # ---- Acceleration (per-job speed options; no restart) ----
                # NOTE: the VAE selector is unrelated to the server's vae_tiling
                # (a VRAM-saving tile split).
                reg(gr.Markdown(f"### {L('accel_section_title')}"),
                    "accel_section_title", "value")
                reg(gr.Markdown(L("accel_note"), elem_classes=["note"]),
                    "accel_note", "value")
                # 実装のあるつまみ（GGUF逆量子化の Triton 1カーネル化）。位置は
                # 撤去した旧モック accel_fused_gguf（受理のみで効果の無かった
                # fused_gguf_dequant_gemm）をそのまま継承している。
                # 利用可否はクライアント側でゲートしない（attention と同じ理由
                # ＝サーバが Triton 不在・例外・自己検証不一致のいずれでも黙って
                # 従来実装へ降格するので、二つ目の真理の源を作らない）。
                accel_fused_dequant = reg(gr.Checkbox(
                    value=FUSED_GGUF_DEQUANT_KERNEL_DEFAULT,
                    label=L("accel_lbl_fused_dequant"),
                    info=L("accel_info_fused_dequant"),
                ), "accel_lbl_fused_dequant")
                reg(accel_fused_dequant, "accel_info_fused_dequant", "info")
                # The radio's VALUES are the API literals ("sdpa"/"sage"); the
                # displayed choice strings are deliberately fixed, untranslated
                # text so switch_language needs no extra branch (only the label
                # and the info line are registered for translation).
                # Availability is NOT gated here: this UI ships with the server,
                # so it runs on the machine that has (or has not) SageAttention
                # installed, and the API degrades a sage request to sdpa on its
                # own — a client-side lockout would only add a second, drifting
                # source of truth.
                attention_backend = reg(gr.Radio(
                    choices=[("sdpa", "sdpa"), ("sage attention", "sage")],
                    value="sdpa", label=L("accel_lbl_attention"),
                    info=L("accel_info_attention"),
                ), "accel_lbl_attention")
                reg(attention_backend, "accel_info_attention", "info")
                # 利用可否は API 側が黙って no-op にするのでクライアント側で
                # ゲートしない（attention と同じ理由。すぐ上の attention_backend
                # の注釈を参照）。
                accel_prefetch = reg(gr.Checkbox(
                    value=BLOCK_SWAP_PREFETCH_DEFAULT, label=L("accel_lbl_prefetch"),
                    info=L("accel_info_prefetch"),
                ), "accel_lbl_prefetch")
                reg(accel_prefetch, "accel_info_prefetch", "info")
                # ここもクライアント側でゲートしない
                # （利用可否ではなく「積んでいるメモリ量しだい」で、サーバから
                # は判定できない。既定offのまま説明文で「64GB以上推奨」と伝え、
                # 判断はユーザーに委ねる方針＝オーナー確定）。
                accel_keep_resident = reg(gr.Checkbox(
                    value=KEEP_RESIDENT_DEFAULT, label=L("accel_lbl_keep_resident"),
                    info=L("accel_info_keep_resident"),
                ), "accel_lbl_keep_resident")
                reg(accel_keep_resident, "accel_info_keep_resident", "info")
                # Display name: "PruneVAED" -> "PrunaVAED" (correct product name
                # per PRUNAVAED_WORKORDER.md §6.1). The API literal value
                # "prune_vaed" is an external contract and is unchanged.
                # PrunaVAED (Docs/PENDING_TASKS_CLOSED.md §3-66, filed as
                # §3-50 at the time): pruned video-VAE decoder, real as of
                # 2026-08-05. Wired the same way as attention_backend/
                # accel_prefetch/accel_keep_resident/accel_fused_dequant above
                # -- see dispatch()/chain_dispatch() below for how the selected
                # value reaches vae_mode in the request payload (single, chain
                # AND batch, via handlers.py / batch.py's BatchSnapshot).
                accel_vae = reg(gr.Radio(
                    choices=[("Default", "default"), ("PrunaVAED", "prune_vaed")],
                    value="default",
                    label=L("accel_lbl_vae"),
                    info=L("accel_info_vae"),
                ), "accel_lbl_vae")
                reg(accel_vae, "accel_info_vae", "info")

                # ---- Server config viewer (raw /config + spill-free table) ----
                reg(gr.Markdown(f"### {L('h_server')}"), "h_server", "value")
                with gr.Accordion(L("sum_config"), open=False) as config_accordion:
                    reg(config_accordion, "sum_config", "label")
                    server_config_json = gr.JSON(value={})
                reg(gr.Markdown(L("lbl_maxframes")), "lbl_maxframes", "value")
                spill_table = gr.Dataframe(
                    headers=[L("col_res"), L("col_maxframes")],
                    datatype="str", column_count=(2, "fixed"),
                    interactive=False, value=[],
                )
                reg(gr.Markdown(L("cap_over"), elem_classes=["note"]), "cap_over", "value")

                # ---- Models (model management: per-category dropdowns) ----
                # Independent section (parent ruling): it shares NO closure or
                # outputs with on_page_load / on_refresh_config / the top-bar
                # load buttons — only additive event listeners below.
                reg(gr.Markdown(f"### {L('model_section_title')}"),
                    "model_section_title", "value")
                # Base model (multi-engine axis; Docs/PENDING_TASKS_CLOSED.md's
                # old §1-25, closed 2026-09-01). Choices come from GET
                # /models' base_models[] — label = display_name, value = id —
                # and the value doubles as the "which base model is active"
                # readout (refresh_model_dropdowns re-selects active_base_model).
                # The labels are language-independent, so switch_language only
                # has to re-stamp this component's own label.
                model_base_dd = reg(gr.Dropdown(
                    choices=[], value=None, label=L("model_base_label")),
                    "model_base_label")
                with gr.Row():
                    model_dd_transformer = reg(gr.Dropdown(
                        choices=[(MODEL_DEFAULT, MODEL_DEFAULT)], value=MODEL_DEFAULT,
                        label=L("model_cat_transformer")), "model_cat_transformer")
                    model_dd_text_encoder = reg(gr.Dropdown(
                        choices=[(MODEL_DEFAULT, MODEL_DEFAULT)], value=MODEL_DEFAULT,
                        label=L("model_cat_text_encoder")), "model_cat_text_encoder")
                with gr.Row():
                    model_dd_video_vae = reg(gr.Dropdown(
                        choices=[(MODEL_DEFAULT, MODEL_DEFAULT)], value=MODEL_DEFAULT,
                        label=L("model_cat_video_vae")), "model_cat_video_vae")
                    model_dd_audio = reg(gr.Dropdown(
                        choices=[(MODEL_DEFAULT, MODEL_DEFAULT)], value=MODEL_DEFAULT,
                        label=L("model_cat_audio")), "model_cat_audio")
                with gr.Row():
                    model_refresh_btn = reg(gr.Button(L("model_btn_refresh")),
                                            "model_btn_refresh", "value")
                    model_load_btn = reg(gr.Button(L("model_btn_load"),
                                                   variant="primary"),
                                         "model_btn_load", "value")
                model_status_box = gr.Textbox(label="", show_label=False,
                                              interactive=False, container=False)
                reg(gr.Markdown(L("model_hint"), elem_classes=["note"]),
                    "model_hint", "value")

                # ---- Danger zone (gated by a confirmation checkbox) ----
                reg(gr.Markdown(f"### {L('h_danger')}"), "h_danger", "value")
                danger_chk = reg(gr.Checkbox(value=False, label=L("chk_danger")),
                                 "chk_danger")
                with gr.Row():
                    unload_confirm_btn = reg(gr.Button(L("btn_unload_confirm"),
                                                       interactive=False),
                                             "btn_unload_confirm", "value")
                    purge_btn = reg(gr.Button(L("btn_purge"), variant="stop",
                                              interactive=False),
                                    "btn_purge", "value")
                purge_msg = gr.Textbox(label="", show_label=False,
                                       interactive=False, container=False)

        # ---- events ----
        refresh_btn.click(refresh_status, outputs=status_box)
        # Top-bar Refresh also refreshes the Settings config viewer + spill table.
        refresh_btn.click(
            on_refresh_config, inputs=[config_state, lang_state],
            outputs=[config_state, server_config_json, spill_table, preset, adapter],
        )

        # Settings /config auto-retry: ticks every 3s while active, calling
        # on_config_retry, which disables the timer on success (or once
        # CONFIG_RETRY_MAX_ATTEMPTS is exhausted).
        config_retry_timer.tick(
            on_config_retry, inputs=[config_state, config_retry_state, lang_state],
            outputs=[config_state, preset, adapter, server_config_json, spill_table,
                     config_retry_state, config_retry_timer],
        )

        # chain_preset choices mirror the Generate preset, but are refreshed via
        # this dedicated config_state.change listener rather than folded into the
        # shared on_page_load / on_config_retry / on_refresh_config returns --
        # those closures' return arity is asserted by test_gradio_handlers and
        # must stay fixed. config_state is updated by every one of those paths,
        # so this single listener covers page-load, refresh and retry uniformly.
        config_state.change(
            lambda cfg: gr.update(choices=build_preset_choices(cfg),
                                  value=pick_default_preset(cfg)),
            inputs=config_state, outputs=chain_preset,
        )

        preset.change(
            apply_preset, inputs=[preset, config_state],
            outputs=[width, height, num_frames, crop_enabled, crop_w, crop_h,
                     crop_row, spill_warning],
        ).then(
            format_duration_label, inputs=[num_frames, frame_rate],
            outputs=duration_md,
        )

        # Live duration readout: recompute on any frames/fps edit.
        for _dctrl in (num_frames, frame_rate):
            _dctrl.change(format_duration_label, inputs=[num_frames, frame_rate],
                          outputs=duration_md)
        qmode.change(on_qmode_change, inputs=qmode, outputs=qmode)
        crop_enabled.change(on_crop_toggle, inputs=crop_enabled, outputs=crop_row)

        # Spill-free warning: recompute on any manual width/height/num_frames
        # edit (preset application already includes it in its own outputs above).
        for _ctrl in (width, height, num_frames):
            _ctrl.change(
                compute_spill_warning,
                inputs=[width, height, num_frames, config_state],
                outputs=spill_warning,
            )

        # Theme: pure-frontend toggle (no backend round-trip). Matches the mount
        # site's dark-default js.
        theme_dd.change(
            None, inputs=theme_dd, outputs=None,
            js="(v) => { document.body.classList.toggle('light', v === 'light'); "
               "document.body.classList.toggle('dark', v === 'dark'); }",
        )

        kf_inputs: list[object] = []
        for kf_enabled, kf_image, kf_frame, kf_strength in kf_slots:
            kf_inputs.extend([kf_enabled, kf_image, kf_frame, kf_strength])

        # A control adapter needs a reference video; "None" (ADAPTER_NONE) and any
        # empty/unset selection do not. Grey out (and CLEAR any uploaded file on)
        # the reference-video input unless a real adapter is chosen, so the field's
        # enabled state matches what the handler actually consumes (handlers.py
        # use_adapter check). Clearing the value on de-select avoids a stale video
        # lingering behind a greyed-out control.
        def on_adapter_change(adapter_value):
            if not adapter_value or adapter_value == ADAPTER_NONE:
                return gr.update(interactive=False, value=None)
            return gr.update(interactive=True)

        # NAG (non-CFG Negative): the shared negative-prompt textbox is
        # interactive ONLY while the checkbox is on (CSS's .negative-greyed
        # selectors key off the disabled/readonly state, so no extra CSS
        # follow-up is needed here).
        def on_nag_enable_toggle(enabled):
            return gr.update(interactive=bool(enabled))

        # Method choice ("nag" / "vsf") toggles which parameter group is shown:
        # each method's group is mutually exclusive (only one non-CFG negative
        # method runs per job).
        def on_nag_method_change(method):
            is_vsf = method == "vsf"
            return gr.update(visible=not is_vsf), gr.update(visible=is_vsf)

        # ---- Generate button dispatch (single vs. batch A2V) ----
        # The middle stage of the Generate button's click chain. When batch A2V
        # is OFF it delegates verbatim to the frozen ``generate`` handler (same
        # values, same 3 outputs). When ON it instead snapshots the whole
        # Generate tab into a BatchSnapshot and hands it to the in-process
        # BatchRunner, then returns immediately (the run proceeds on a daemon
        # thread; the batch_timer below reflects its progress). The 6 batch
        # inputs are APPENDED after the pre-existing scalar positionals so every
        # one of those maps to the exact same generate() argument as before.
        # *kf_flat: the keyframe quads, wired as the TAIL of inputs (see generate_btn.click below).
        def dispatch(prompt_v, negative_v,
                     width_v, height_v, crop_en_v, crop_w_v, crop_h_v,
                     num_frames_v, frame_rate_v, seed_v,
                     adapter_v, adapter_strength_v, control_adherence_v,
                     reference_strength_v, ref_video_v, config_v,
                     lang_v, poll_interval_v, poll_timeout_v, gen_a2v_audio_v,
                     batch_enable_v, batch_rows_v, batch_wav_dir_v,
                     batch_out_mode_v, batch_out_dir_v, batch_add_replace_v,
                     batch_img_dir_v,
                     nag_enabled_v, nag_scale_v, nag_tau_v, nag_alpha_v,
                     nag_method_v, vsf_scale_v, attention_backend_v, accel_prefetch_v,
                     accel_keep_resident_v, accel_fused_dequant_v, accel_vae_v,
                     *kf_flat):
            kf_slot_values = [tuple(kf_flat[i:i + 4])
                              for i in range(0, len(kf_flat), 4)]
            if not batch_enable_v:
                # Single-generation path: byte-identical delegation (these
                # positionals ARE the generate() signature, with the whole
                # keyframe grid handed over as the single ``kf_slots`` list in
                # third position); nag_*/neg_method/vsf_* are passed as keywords
                # since they sit at the very end of generate()'s signature.
                yield from generate(
                    prompt_v, negative_v, kf_slot_values,
                    width_v, height_v, crop_en_v, crop_w_v, crop_h_v,
                    num_frames_v, frame_rate_v, seed_v,
                    adapter_v, adapter_strength_v, control_adherence_v,
                    reference_strength_v, ref_video_v, config_v,
                    lang_v, poll_interval_v, poll_timeout_v, gen_a2v_audio_v,
                    nag_enabled=nag_enabled_v, nag_scale=nag_scale_v,
                    nag_tau=nag_tau_v, nag_alpha=nag_alpha_v,
                    neg_method=nag_method_v, vsf_scale=vsf_scale_v,
                    attention_backend=attention_backend_v,
                    block_swap_prefetch=accel_prefetch_v,
                    keep_resident=accel_keep_resident_v,
                    fused_gguf_dequant_kernel=accel_fused_dequant_v,
                    vae_mode=accel_vae_v)
                return

            rows = batch_rows_v or []
            if not rows:
                yield L("batch_msg_no_wav", lang_v), "", None
                return

            # NAG precheck (owner requirement): non-CFG Negative enabled but the
            # shared negative prompt is empty -> reject with zero API calls,
            # same yield-shape as the "no wav rows" check above (toast handled
            # by the caller reading this message, not gr.Warning here).
            if nag_enabled_v and not (negative_v or "").strip():
                yield L("nag_msg_negative_required", lang_v), "", None
                return

            # --- common prompt: strip <lora:...> tokens exactly as the frozen
            # generate path does, so the runner's compose_prompt gets the clean
            # base text and the extracted tokens flow into ``loras``. ---
            use_adapter_v = bool(adapter_v) and adapter_v != ADAPTER_NONE
            send_prompt = prompt_v
            prompt_loras: list[dict] = []
            if _LORA_TOKEN_RE.search(prompt_v or ""):
                try:
                    lora_list = api.list_loras()
                except Exception as exc:
                    yield L("lora_msg_list_failed", lang_v).format(err=exc), "", None
                    return
                known = [e.get("name") for e in (lora_list or [])
                         if isinstance(e, dict) and e.get("name")]
                send_prompt, prompt_loras, lora_err = parse_prompt_loras(
                    prompt_v, known, lang_v)
                if lora_err is not None:
                    yield lora_err, "", None
                    return
            combined_loras = _combine_generate_loras(
                use_adapter_v, adapter_v, adapter_strength_v, prompt_loras)

            # crop_output — identical arithmetic to the frozen generate path.
            crop_output = None
            try:
                if crop_en_v and int(crop_w_v) > 0 and int(crop_h_v) > 0:
                    crop_output = {"width": int(crop_w_v), "height": int(crop_h_v)}
            except (TypeError, ValueError):
                crop_output = None

            # shared keyframe images (common i2v slots) -> (path, frame, strength),
            # collected the same way the frozen path builds ``to_upload``.
            shared_images: list[tuple] = []
            for en, img, fr, st in kf_slot_values:
                if en and img:
                    try:
                        shared_images.append((img, int(fr or 0), float(st)))
                    except (TypeError, ValueError):
                        continue

            try:
                width_i, height_i = int(width_v), int(height_v)
                seed_i = int(seed_v)
                fps_f = float(frame_rate_v) if frame_rate_v else 24.0
            except (TypeError, ValueError):
                yield L("msg_bad_dimension", lang_v), "", None
                return

            interval, timeout_s = _resolve_poll(poll_interval_v, poll_timeout_v)
            out_dir = str(batch_manifest.resolve_output_dir(
                batch_wav_dir_v, batch_out_mode_v, batch_out_dir_v))

            snapshot = BatchSnapshot(
                wav_dir=batch_wav_dir_v,
                out_dir=out_dir,
                img_dir=batch_img_dir_v or "",
                prompt_common=send_prompt,
                negative=negative_v or "",
                prompt_mode=batch_add_replace_v or "add",
                width=width_i,
                height=height_i,
                crop_output=crop_output,
                frame_rate=fps_f,
                seed=seed_i,
                loras=combined_loras,
                shared_images=shared_images,
                use_adapter=use_adapter_v,
                ref_video_path=ref_video_v,
                control_adherence=control_adherence_v,
                reference_strength=reference_strength_v,
                poll_interval=interval,
                poll_timeout_s=timeout_s,
                nag_enabled=bool(nag_enabled_v),
                nag_scale=float(nag_scale_v),
                nag_tau=float(nag_tau_v),
                nag_alpha=float(nag_alpha_v),
                neg_method=nag_method_v,
                vsf_scale=float(vsf_scale_v),
                # Acceleration: the batch runner builds its payloads from THIS
                # snapshot, not from handler args — without this line an
                # overnight batch would silently stay on sdpa.
                attention_backend=attention_backend_v or "sdpa",
                block_swap_prefetch=bool(accel_prefetch_v),
                keep_resident=bool(accel_keep_resident_v),
                fused_gguf_dequant_kernel=bool(accel_fused_dequant_v),
                vae_mode=accel_vae_v or "default",
                # Skip ceiling for the start-time re-judgment — the SAME value
                # the Set audios scan used, so a Start never re-judges against
                # a different cap than the table the user is looking at.
                num_frames=int(num_frames_v or MAX_FRAMES),
            )

            ok, msg = get_runner().start(snapshot, rows, api)
            if ok:
                n = len([r for r in rows if r.stat in
                         (STAT_WAITING, STAT_FAILED, STAT_GENERATING)])
                yield L("batch_msg_started", lang_v).format(n=n), "", None
            else:
                yield _batch_start_reason(msg, lang_v), "", None

        # Batch-aware restore for stage 3: while batch mode is on AND the runner
        # actually took off (running/stopping), the button stays disabled and
        # shows "Batching a2v..." so the run is visibly in progress; the
        # batch_timer tick below restores it to "Start a2v batch" once the runner
        # returns to idle. Otherwise (batch off, or a start that failed and left
        # the runner idle) it re-enables to the appropriate normal label.
        def _restore(enable, lang):
            if enable and get_runner().state != STATE_IDLE:
                return gr.update(interactive=False, value=L("batch_running", lang))
            key = "batch_start" if enable else "btn_generate"
            return gr.update(interactive=True, value=L(key, lang))

        def arm_batch_timer():
            # Only tick the batch table while a run is actually in flight.
            return gr.update(active=(get_runner().state != STATE_IDLE))

        generate_btn.click(
            on_generate_btn_start, inputs=lang_state, outputs=generate_btn,
        ).then(
            dispatch,
            # nag_enabled/nag_scale/nag_tau/nag_alpha, then nag_method/
            # vsf_scale, then the Acceleration attention selector, the
            # block-swap prefetch checkbox, the keep-resident checkbox, the
            # fused-dequant checkbox AND the VAE radio (PrunaVAED,
            # Docs/PENDING_TASKS_CLOSED.md §3-66, filed as §3-50 at the time),
            # are APPENDED after every pre-existing scalar positional (matching
            # dispatch()'s signature order, which appends them after
            # batch_img_dir_v).
            inputs=[prompt, negative, width, height,
                    crop_enabled, crop_w, crop_h, num_frames, frame_rate, seed,
                    adapter, adapter_strength, control_adherence,
                    reference_strength_slider, ref_video, config_state,
                    lang_state, poll_interval, poll_timeout, gen_a2v_audio,
                    batch_enable, batch_rows_state, batch_wav_dir,
                    batch_out_mode, batch_out_dir, batch_add_replace,
                    batch_img_dir,
                    nag_enabled, nag_scale, nag_tau, nag_alpha,
                    nag_method, vsf_scale, attention_backend, accel_prefetch,
                    accel_keep_resident, accel_fused_dequant, accel_vae,
                    # Nothing may be appended after this: dispatch()'s *kf_flat
                    # swallows everything from here to the end of the list.
                    *kf_inputs],
            outputs=[progress_box, job_box, video_out],
        ).then(
            _restore, inputs=[batch_enable, lang_state], outputs=generate_btn,
        ).then(
            arm_batch_timer, outputs=batch_timer,
        )

        # Enable/disable (and clear) the reference-video input to track the adapter
        # selection (control adapter -> enabled; None/unset -> greyed out + cleared).
        adapter.change(on_adapter_change, inputs=adapter, outputs=ref_video)

        # NAG: checkbox toggles the shared negative textbox's editability;
        # the method radio ("nag"/"vsf") shows/hides that method's own
        # parameter group.
        nag_enabled.change(on_nag_enable_toggle, inputs=nag_enabled, outputs=negative)
        nag_method.change(on_nag_method_change, inputs=nag_method,
                          outputs=[nag_group, vsf_group])

        # Feature 1: attaching a .wav to the A2V audio field auto-adjusts
        # Frames to fit its measured duration (non-wav / unreadable / cleared
        # attachments are a no-op). num_frames' OWN .change listeners (Duration
        # readout, spill-free warning — wired below) fire on this programmatic
        # update too (gradio's .change semantics do not distinguish a
        # user edit from a value set by another event's output), so no extra
        # wiring is needed here to keep those in sync.
        gen_a2v_audio.change(
            a2v_audio_change_handler,
            inputs=[gen_a2v_audio, frame_rate, lang_state],
            outputs=[num_frames],
        )

        # ---- Batch A2V events (WP3) ----
        # Enabling batch mode repurposes the Generate button ("Start a2v batch")
        # and freezes the single-shot audio + Frames inputs (per-wav Frames are
        # auto-computed on scan). switch_language re-stamps generate_btn to its
        # plain label, so a dedicated lang listener below re-applies these while
        # enabled (the frozen switch_language signature/arity is untouched).
        def on_batch_enable_toggle(enable, lang):
            if enable:
                return (gr.update(value=L("batch_start", lang)),
                        gr.update(interactive=False),
                        gr.update(interactive=False,
                                  info=L("batch_frames_auto", lang)))
            return (gr.update(value=L("btn_generate", lang)),
                    gr.update(interactive=True),
                    gr.update(interactive=True, info=""))

        batch_enable.change(
            on_batch_enable_toggle, inputs=[batch_enable, lang_state],
            outputs=[generate_btn, gen_a2v_audio, num_frames],
        )

        # Custom output folder textbox visible only in "custom" mode.
        batch_out_mode.change(
            lambda m: gr.update(visible=(m == "custom")),
            inputs=batch_out_mode, outputs=batch_out_dir,
        )

        # Live "max duration" readout tracks resolution / fps edits (independent
        # of the Duration / spill listeners so it never interferes with them).
        def on_batch_maxdur(width_v, height_v, fps_v, config_v, lang):
            return gr.update(value=batch_maxdur_text(width_v, height_v, fps_v,
                                                     config_v, lang))

        for _mctrl in (width, height, frame_rate):
            _mctrl.change(
                on_batch_maxdur,
                inputs=[width, height, frame_rate, config_state, lang_state],
                outputs=batch_maxdur_md,
            )

        # "Set audios": scan the wav folder, merge over any existing manifest,
        # persist the CSV, and (re)build the table + image dropdown + summary.
        # ``num_frames_v`` is the Generate tab's frame count: it doubles as the
        # batch's Skip ceiling (effective cap = min(num_frames, 481), the same
        # rule the WebView2 frontend uses), so a row over it is excluded here
        # and comes back on the next Set audios once the value is raised.
        def on_batch_set_audios(wav_dir, img_dir, width_v, height_v, fps_v,
                                num_frames_v, config_v, lang):
            # Locked while a run is in flight: re-scanning would rewrite the CSV
            # the runner owns and desync the displayed rows from its state. Leave
            # the table, rows-state, dropdown, maxdur and summary all untouched
            # (bare gr.update() -> the State keeps its current value, see
            # postprocess_data's stateful branch: an update with no "value").
            if get_runner().state != STATE_IDLE:
                gr.Warning(L("batch_msg_running_locked", lang))
                return (gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update())
            if not wav_dir or not os.path.isdir(str(wav_dir)):
                gr.Warning(L("batch_msg_wav_dir_invalid", lang))
                return (gr.update(value=[]), [],
                        gr.update(choices=[IMAGE_SHARED], value=IMAGE_SHARED),
                        gr.update(), gr.update(value=""))
            fps = _resolve_fps(fps_v)
            scanned = batch_manifest.scan_wav_folder(
                wav_dir, fps, frames_for=suggest_frames_for_audio,
                max_frames=num_frames_v)
            if not scanned:
                gr.Warning(L("batch_msg_no_wav", lang))
            existing = batch_manifest.read_manifest(wav_dir)
            rows, warnings = batch_manifest.merge_rows(existing, scanned)
            res = batch_manifest.write_manifest_atomic(wav_dir, rows)
            if res.locked:
                gr.Warning(L("batch_msg_csv_locked", lang))
            for w in warnings:
                gr.Warning(w)
            # spill (comfortable-limit) advisory — count of over-threshold rows.
            spill_free = ((config_v or {}).get("limits") or {}).get(
                "spill_free_frames") or {}
            spill = batch_manifest.compute_spill_warnings(
                rows, width_v, height_v, spill_free)
            if spill:
                key = f"{int(width_v)}x{int(height_v)}"
                gr.Warning(L("batch_warn_spill", lang).format(
                    n=len(spill), res=key, frames=spill_free.get(key)))
            return (gr.update(value=batch_rows_to_table(rows, lang)), rows,
                    gr.update(choices=batch_image_choices(img_dir),
                              value=IMAGE_SHARED),
                    gr.update(value=batch_maxdur_text(width_v, height_v, fps_v,
                                                      config_v, lang)),
                    gr.update(value=batch_summary_text(rows, lang)))

        batch_set_btn.click(
            on_batch_set_audios,
            inputs=[batch_wav_dir, batch_img_dir, width, height, frame_rate,
                    num_frames, config_state, lang_state],
            outputs=[batch_table, batch_rows_state, batch_img_dd,
                     batch_maxdur_md, batch_summary_md],
        )

        # Prompt-column (idx 4) inline edits. Any other column is reverted (the
        # table is re-rendered from the canonical rows); edits are refused while
        # a run is in flight.
        def on_batch_prompt_edit(rows, wav_dir, lang, evt: gr.EditData):
            rows = rows or []
            if get_runner().state != STATE_IDLE:
                gr.Warning(L("batch_msg_running_locked", lang))
                return gr.update(value=batch_rows_to_table(rows, lang)), rows
            index = evt.index
            row_i = index[0] if isinstance(index, (list, tuple)) else index
            col_i = index[1] if isinstance(index, (list, tuple)) and len(index) > 1 else None
            if col_i != 4 or row_i is None or row_i >= len(rows):
                return gr.update(value=batch_rows_to_table(rows, lang)), rows
            rows[row_i].prompt = evt.value or ""
            res = batch_manifest.write_manifest_atomic(wav_dir, rows)
            if res.locked:
                gr.Warning(L("batch_msg_csv_locked", lang))
            return gr.update(value=batch_rows_to_table(rows, lang)), rows

        batch_table.edit(
            on_batch_prompt_edit,
            inputs=[batch_rows_state, batch_wav_dir, lang_state],
            outputs=[batch_table, batch_rows_state],
        )

        # Row select -> load the edit panel (image dropdown + info line).
        def on_batch_row_select(rows, lang, evt: gr.SelectData):
            rows = rows or []
            idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if idx is None or idx >= len(rows):
                return None, gr.update(), gr.update(value="")
            row = rows[idx]
            return (idx, gr.update(value=row.image or IMAGE_SHARED),
                    gr.update(value=batch_row_info_text(row, lang)))

        batch_table.select(
            on_batch_row_select, inputs=[batch_rows_state, lang_state],
            outputs=[batch_sel_state, batch_img_dd, batch_row_info_md],
        )

        # Per-row image assignment (edit panel dropdown).
        def on_batch_img_change(sel_idx, image_name, rows, wav_dir, lang):
            rows = rows or []
            if sel_idx is None or sel_idx >= len(rows):
                return gr.update(), rows
            new_image = image_name or IMAGE_SHARED
            # No-op when the value did not actually change (this .change also
            # fires when row-select programmatically loads the dropdown), so a
            # mere selection neither rewrites the CSV nor warns during a run.
            if rows[sel_idx].image == new_image:
                return gr.update(), rows
            if get_runner().state != STATE_IDLE:
                gr.Warning(L("batch_msg_running_locked", lang))
                return gr.update(), rows
            rows[sel_idx].image = new_image
            res = batch_manifest.write_manifest_atomic(wav_dir, rows)
            if res.locked:
                gr.Warning(L("batch_msg_csv_locked", lang))
            return gr.update(value=batch_rows_to_table(rows, lang)), rows

        batch_img_dd.change(
            on_batch_img_change,
            inputs=[batch_sel_state, batch_img_dd, batch_rows_state,
                    batch_wav_dir, lang_state],
            outputs=[batch_table, batch_rows_state],
        )

        # Copy the common prompt into the selected row.
        def on_batch_copy_common(sel_idx, common_prompt, rows, wav_dir, lang):
            rows = rows or []
            if get_runner().state != STATE_IDLE:
                gr.Warning(L("batch_msg_running_locked", lang))
                return gr.update(), rows
            if sel_idx is None or sel_idx >= len(rows):
                return gr.update(), rows
            rows[sel_idx].prompt = common_prompt or ""
            res = batch_manifest.write_manifest_atomic(wav_dir, rows)
            if res.locked:
                gr.Warning(L("batch_msg_csv_locked", lang))
            return gr.update(value=batch_rows_to_table(rows, lang)), rows

        batch_copy_common_btn.click(
            on_batch_copy_common,
            inputs=[batch_sel_state, prompt, batch_rows_state, batch_wav_dir,
                    lang_state],
            outputs=[batch_table, batch_rows_state],
        )

        # Re-queue the selected row (stat -> Waiting; output/error untouched).
        # A Skip row is still refused here: this button only flips a stat, and
        # a Skip is a judgment about the row's LENGTH ("over-cap" against the
        # Generate tab's frame count, or a non-wav file), which re-queueing
        # alone cannot change. The way back is to raise DURATION (or replace
        # the file) and press "Set audios" again — the rescan clears the Skip
        # (manifest.merge_rows Rule 1) — so the warning says exactly that.
        def on_batch_regen(sel_idx, rows, wav_dir, lang):
            rows = rows or []
            if get_runner().state != STATE_IDLE:
                gr.Warning(L("batch_msg_running_locked", lang))
                return gr.update(), rows
            if sel_idx is None or sel_idx >= len(rows):
                return gr.update(), rows
            if rows[sel_idx].stat == STAT_SKIP:
                raw = rows[sel_idx].skip_reason
                reason = L(_BATCH_SKIP_KEY[raw], lang) if raw in _BATCH_SKIP_KEY \
                    else (raw or "")
                gr.Warning(L("batch_msg_regen_skip", lang).format(reason=reason))
                return gr.update(), rows
            rows[sel_idx].stat = STAT_WAITING
            res = batch_manifest.write_manifest_atomic(wav_dir, rows)
            if res.locked:
                gr.Warning(L("batch_msg_csv_locked", lang))
            return gr.update(value=batch_rows_to_table(rows, lang)), rows

        batch_regen_btn.click(
            on_batch_regen,
            inputs=[batch_sel_state, batch_rows_state, batch_wav_dir, lang_state],
            outputs=[batch_table, batch_rows_state],
        )

        # Stop button: ask the runner to halt (in-flight row rewinds to Waiting).
        def on_batch_stop(lang):
            get_runner().request_stop()
            gr.Info(L("batch_msg_stopped", lang))

        batch_stop_btn.click(on_batch_stop, inputs=lang_state, outputs=None)

        # Timer tick: mirror the runner's live rows + summary into the table and
        # self-disable once the runner is idle (final state stays displayed).
        # It also drives the Generate button's label back: while the run is in
        # flight the button stays "Batching a2v..."/disabled, and on the tick
        # where the runner has returned to idle it is restored to
        # "Start a2v batch"/enabled (enable + lang come from the inputs). With
        # batch mode off the button is left untouched (bare gr.update()).
        def on_batch_tick(enable, lang):
            runner = get_runner()
            rows = runner.snapshot_rows()
            active = runner.state != STATE_IDLE
            if not enable:
                btn = gr.update()
            elif active:
                btn = gr.update(interactive=False, value=L("batch_running", lang))
            else:
                btn = gr.update(interactive=True, value=L("batch_start", lang))
            return (gr.update(value=batch_rows_to_table(rows, lang)),
                    gr.update(value=batch_summary_text(rows, lang)),
                    rows, gr.update(active=active), btn)

        batch_timer.tick(
            on_batch_tick, inputs=[batch_enable, lang_state],
            outputs=[batch_table, batch_summary_md, batch_rows_state, batch_timer,
                     generate_btn],
        )

        # Language switch for the batch table headers + rendered rows/summary.
        # Kept as a SEPARATE lang listener (not folded into switch_language)
        # because test_gradio_handlers pins switch_language's output arity to
        # len(registry)+2; this adds no extra registry/extra outputs.
        def on_batch_lang_switch(lang, rows):
            rows = rows or []
            return (gr.update(headers=batch_table_headers(lang),
                              value=batch_rows_to_table(rows, lang)),
                    gr.update(value=batch_summary_text(rows, lang)))

        # Re-apply the batch-enabled overrides after switch_language resets the
        # Generate button to its plain label (only while batch mode is on).
        def on_batch_reapply_enable(enable, lang):
            if not enable:
                return gr.update(), gr.update(), gr.update()
            return (gr.update(value=L("batch_start", lang)),
                    gr.update(interactive=False),
                    gr.update(interactive=False, info=L("batch_frames_auto", lang)))

        # ---- Clip Chain events ----
        # Reuse the SAME quality-mode revert + crop-toggle handlers as Generate.
        chain_qmode.change(on_qmode_change, inputs=chain_qmode, outputs=chain_qmode)
        chain_crop_enabled.change(on_crop_toggle, inputs=chain_crop_enabled,
                                  outputs=chain_crop_row)

        # Chain preset: the 24 slot Checkbox values feed enabled_flags (count of
        # enabled slots -> total-timeline warning); all 24 frame Numbers receive
        # the recommended per-clip length. apply_chain_preset takes enabled_flags
        # as ONE list, so a thin wrapper gathers the checkbox values.
        chain_enabled_boxes = [_slot[0] for _slot in chain_clip_slots]
        chain_frame_nums = [_slot[2] for _slot in chain_clip_slots]

        def on_chain_preset_change(name, config, *rest):
            # rest = 24 enabled flags + fps + overlap + lang.
            enabled_flags = list(rest[:CHAIN_MAX_CLIPS])
            fps, overlap, lang = rest[CHAIN_MAX_CLIPS:CHAIN_MAX_CLIPS + 3]
            return apply_chain_preset(
                name, config, enabled_flags=enabled_flags,
                fps=fps, overlap_frames=overlap, lang=lang,
            )

        # ---- Chain clip-count estimate (live duration readout) ----
        # ONE common handler recomputes chain_duration_md from the enabled slots.
        # It masks the enabled flags by chain_open_count so a hidden-but-checked
        # slot (which the ± handler keeps off anyway) can never inflate the
        # estimate. Wired to every trigger below AND re-run on lang switch so the
        # readout never keeps a stale-language string.
        chain_est_inputs = [chain_open_count, lang_state, chain_fps, chain_overlap,
                            *chain_enabled_boxes, *chain_frame_nums]

        def on_chain_estimate(open_count, lang, fps, overlap, *enabled_and_frames):
            enabled = list(enabled_and_frames[:CHAIN_MAX_CLIPS])
            frames = list(enabled_and_frames[CHAIN_MAX_CLIPS:2 * CHAIN_MAX_CLIPS])
            try:
                oc = int(open_count)
            except (TypeError, ValueError):
                oc = CHAIN_MAX_CLIPS
            masked = [bool(e) and (i < oc) for i, e in enumerate(enabled)]
            return gr.update(value=compute_chain_duration_label(
                masked, frames, fps, overlap, lang))

        chain_preset.change(
            on_chain_preset_change,
            inputs=[chain_preset, config_state, *chain_enabled_boxes,
                    chain_fps, chain_overlap, lang_state],
            outputs=[chain_width, chain_height, chain_crop_enabled,
                     chain_crop_w, chain_crop_h, chain_crop_row,
                     *chain_frame_nums, chain_preset_warning],
        ).then(
            # Preset rewrites every frame Number; refresh the estimate explicitly
            # (do NOT rely on the frame .change cascade firing in order).
            on_chain_estimate, inputs=chain_est_inputs, outputs=chain_duration_md,
        )

        # Every enabled/frames/fps/overlap edit refreshes the estimate live.
        for _ctrl in (*chain_enabled_boxes, *chain_frame_nums,
                      chain_fps, chain_overlap):
            _ctrl.change(on_chain_estimate, inputs=chain_est_inputs,
                         outputs=chain_duration_md)

        # ---- Clip Chain ± buttons (grow/shrink the visible clip count) ----
        # chain_extra_* cover slots 2..24 (slot 1 is always visible); the ±
        # handler returns, atomically: the new open count, the 23 group
        # visibility updates, the 23 Use-box updates (forced ON for slots newly
        # revealed by "＋" — owner request —, forced OFF when hidden, no-op for
        # slots that merely stay visible so a hand-unchecked box is respected),
        # and the refreshed estimate.
        chain_extra_use = chain_enabled_boxes[1:]

        def _make_chain_step(delta):
            def handler(count, lang, fps, overlap, *enabled_and_frames):
                enabled = list(enabled_and_frames[:CHAIN_MAX_CLIPS])
                frames = list(enabled_and_frames[CHAIN_MAX_CLIPS:2 * CHAIN_MAX_CLIPS])
                new_count, states, minus_on, plus_on, counter = slot_step_state(
                    count, delta, CHAIN_MIN_OPEN, CHAIN_MAX_CLIPS, enable_new=True)
                group_updates = [gr.update(visible=vis) for vis, _use in states]
                use_updates = [gr.update() if use is None else gr.update(value=use)
                               for _vis, use in states]
                # Estimate reflects the POST-step state: slot 1 always in; slot i
                # (2..24) counts with its post-update Use value (a hidden slot's
                # use is False, a newly-revealed one's is True).
                masked = [bool(enabled[0])]
                for idx, (vis, use) in enumerate(states):
                    eff = enabled[idx + 1] if use is None else use
                    masked.append(bool(eff) and vis)
                est = gr.update(value=compute_chain_duration_label(
                    masked, frames, fps, overlap, lang))
                return (new_count, *group_updates, *use_updates,
                        gr.update(interactive=minus_on),
                        gr.update(interactive=plus_on),
                        gr.update(value=counter), est)
            return handler

        _chain_step_inputs = [chain_open_count, lang_state, chain_fps, chain_overlap,
                              *chain_enabled_boxes, *chain_frame_nums]
        _chain_step_outputs = [chain_open_count, *chain_extra_groups,
                               *chain_extra_use, chain_minus_btn, chain_plus_btn,
                               chain_counter_md, chain_duration_md]
        chain_plus_btn.click(_make_chain_step(+1), inputs=_chain_step_inputs,
                             outputs=_chain_step_outputs)
        chain_minus_btn.click(_make_chain_step(-1), inputs=_chain_step_inputs,
                              outputs=_chain_step_outputs)

        # ---- Keyframe ± buttons (grow/shrink the visible keyframe rows) ----
        # slot_step_state (pure, presets.py, shared with the Clip Chain) carries
        # the whole semantic transition: row visibility + Use-off for hidden rows
        # (NO auto-enable — owner asked for that on the Clip Chain only), the
        # −/＋ buttons' grey-out-at-floor/-ceiling flags, and the "n/KF_MAX_SLOTS" counter
        # text. Everything is returned in ONE handler so the update is atomic.
        kf_extra_use = [_slot[0] for _slot in kf_slots[1:]]

        def _make_kf_step(delta):
            def handler(count):
                new_count, states, minus_on, plus_on, counter = slot_step_state(
                    count, delta, KF_MIN_OPEN, KF_MAX_SLOTS)
                row_updates = [gr.update(visible=vis) for vis, _use in states]
                use_updates = [gr.update() if use is None else gr.update(value=use)
                               for _vis, use in states]
                return (new_count, *row_updates, *use_updates,
                        gr.update(interactive=minus_on),
                        gr.update(interactive=plus_on), gr.update(value=counter))
            return handler

        _kf_step_outputs = [kf_open_count, *kf_extra_rows, *kf_extra_use,
                            kf_minus_btn, kf_plus_btn, kf_counter_md]
        kf_plus_btn.click(_make_kf_step(+1), inputs=kf_open_count,
                          outputs=_kf_step_outputs)
        kf_minus_btn.click(_make_kf_step(-1), inputs=kf_open_count,
                           outputs=_kf_step_outputs)

        chain_clip_inputs: list[object] = []
        for _slot in chain_clip_slots:
            chain_clip_inputs.extend(_slot)

        # Acceleration: the attention selector, the block-swap prefetch
        # checkbox, the keep-resident checkbox, the fused-dequant checkbox AND
        # the VAE radio (PrunaVAED, Docs/PENDING_TASKS_CLOSED.md §3-66, filed
        # as §3-50 at the time) are APPENDED at the very end of the chain
        # inputs list below, in that order (attention_backend,
        # accel_prefetch, accel_keep_resident, accel_fused_dequant,
        # accel_vae). generate_chain keeps
        # ``src_audio`` as its last POSITIONAL parameter (never wired from this
        # tab, and relied on positionally by tests/test_gradio_v2v_a2v.py's
        # _chain_args), so the five trailing values cannot be delivered
        # positionally -- this thin wrapper peels them off and forwards them as
        # KEYWORDS, the same discipline the Generate tab's dispatch() uses.
        # NOTE: every negative index below is tied to the LENGTH of that
        # trailing block. Appending one more Acceleration input means shifting
        # ALL of them (and the ``args[:-N]`` slice) by one -- a silent
        # mis-wiring otherwise. tests/test_gradio_ui.py locks the order.
        def chain_dispatch(*args):
            yield from chain_generate(*args[:-5],
                                      attention_backend=args[-5],
                                      block_swap_prefetch=args[-4],
                                      keep_resident=args[-3],
                                      fused_gguf_dequant_kernel=args[-2],
                                      vae_mode=args[-1])

        chain_generate_btn.click(
            on_generate_btn_start, inputs=lang_state, outputs=chain_generate_btn,
        ).then(
            chain_dispatch,
            # Positional-order contract with make_chain_handler.generate_chain
            # (handlers.py): this list stops at ``chunked_upsample`` -- the
            # function's remaining trailing params (nag_enabled, nag_scale,
            # nag_tau, nag_alpha, neg_method, vsf_scale, src_audio)
            # are appended right after it in inputs=[...] below, matching the
            # signature's declared order exactly (chunked_upsample -> nag x4
            # -> neg_method/vsf_scale -> src_audio). ``src_audio``
            # itself is never wired from this tab (A2V lives on Generate), so
            # it is intentionally left off the end and keeps its None default.
            # The Acceleration attention selector, the block-swap prefetch
            # checkbox, the keep-resident checkbox, the fused-dequant checkbox
            # AND the VAE radio (PrunaVAED, Docs/PENDING_TASKS_CLOSED.md
            # §3-66, filed as §3-50 at the time) are APPENDED last (in that
            # order) and reach the handler as keywords via chain_dispatch
            # above.
            inputs=[prompt, negative, chain_width, chain_height,
                    chain_crop_enabled, chain_crop_w, chain_crop_h, chain_fps, chain_seed,
                    chain_overlap, chain_overlap_strength,
                    *chain_clip_inputs, config_state,
                    lang_state, poll_interval, poll_timeout,
                    chain_mode, v2v_video, v2v_context, chain_chunked_upsample,
                    nag_enabled, nag_scale, nag_tau, nag_alpha,
                    nag_method, vsf_scale, attention_backend, accel_prefetch,
                    accel_keep_resident, accel_fused_dequant, accel_vae],
            outputs=[chain_progress, chain_job, chain_video],
        ).then(
            make_generate_btn_restore("btn_concat"),
            inputs=lang_state, outputs=chain_generate_btn,
        )

        # ---- V2V mode switching ----
        # The V2V panel + join panel are visible in v2v mode only; clip 1's start
        # image is unavailable in V2V (the frozen source tail occupies clip 0's
        # head — the server rejects the combination with 422, and the handler
        # prechecks it too). A2V now lives on the Generate tab, so this radio is
        # a none/v2v toggle only.
        def on_chain_mode_change(mode):
            is_v2v = mode == "v2v"
            return (gr.update(visible=is_v2v), gr.update(visible=is_v2v),
                    gr.update(interactive=not is_v2v))

        chain_mode.change(on_chain_mode_change, inputs=chain_mode,
                          outputs=[v2v_group, v2v_join_panel, c1_image])

        # ---- V2V join ("Create joined version") ----
        # Hangs off the finished chain job id shown in chain_job; the checkbox
        # is the two-way "create the smoothed joined version / don't" switch
        # (the GUI only ever requests the server's default smoothed join).
        v2v_join_btn.click(
            chain_join,
            inputs=[chain_job, v2v_join_chk, v2v_crossfade, lang_state],
            outputs=[v2v_join_msg, chain_joined_video],
        )

        # ---- Jobs tab events ----
        def on_jobs_refresh(lang):
            try:
                jobs = api.list_jobs()
            except Exception as exc:
                gr.Warning(L("msg_jobs_refresh_failed", lang).format(err=exc))
                return gr.update(), []
            ids = [j.get("job_id", "") for j in jobs if isinstance(j, dict)]
            return gr.update(value=build_jobs_rows(jobs, lang)), ids

        def on_job_select(job_ids, lang, evt: gr.SelectData):
            # Dataframe.select gives evt.index == (row, col); resolve the row to a
            # job id via the parallel jobs_ids_state captured on the last refresh.
            idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if not job_ids or idx is None or idx >= len(job_ids):
                return (gr.update(), None, gr.update(value=None),
                        gr.update(value="", visible=False))
            job_id = job_ids[idx]
            try:
                job = api.get_job(job_id)
            except Exception as exc:
                gr.Warning(L("msg_job_select_failed", lang).format(err=exc))
                return (gr.update(), job_id, gr.update(value=None),
                        gr.update(value="", visible=False))
            status = job.get("status")
            video = None
            if status == "completed":
                try:
                    video = api.fetch_video(job_id)
                except Exception:
                    video = None
            if status == "failed":
                err_update = gr.update(value=format_job_error(job.get("error"), lang),
                                       visible=True)
            else:
                err_update = gr.update(value="", visible=False)
            return gr.update(value=job), job_id, gr.update(value=video), err_update

        def on_job_action(job_id, lang):
            if not job_id:
                return L("msg_no_job_selected", lang)
            try:
                resp = api.delete_job(job_id)
            except Exception as exc:
                return L("msg_job_action_failed", lang).format(err=exc)
            if resp.get("cancel_requested"):
                return L("msg_job_cancel_requested", lang).format(job_id=job_id)
            if resp.get("deleted"):
                return L("msg_job_deleted", lang).format(job_id=job_id)
            return str(resp)

        jobs_refresh_btn.click(on_jobs_refresh, inputs=lang_state,
                               outputs=[jobs_table, jobs_ids_state])
        jobs_table.select(on_job_select, inputs=[jobs_ids_state, lang_state],
                          outputs=[job_detail_json, selected_job_state,
                                   job_video, job_error_box])
        job_action_btn.click(on_job_action, inputs=[selected_job_state, lang_state],
                             outputs=job_action_msg)

        # ---- Settings: danger zone ----
        def on_danger_toggle(enabled):
            upd = gr.update(interactive=bool(enabled))
            return upd, upd

        danger_chk.change(on_danger_toggle, inputs=danger_chk,
                          outputs=[unload_confirm_btn, purge_btn])
        unload_confirm_btn.click(unload_model, outputs=status_box)
        purge_btn.click(lambda lang: delete_finished_jobs(api, lang),
                        inputs=lang_state, outputs=purge_msg)

        # ---- Settings: language switch (the big one) ----
        # Iterate the label registry and return one gr.update per registered
        # component. Components with language-dependent CHOICES (quality radios,
        # theme dropdown, adapter dropdown) fold a ``choices`` rebuild into the
        # SAME update (values unchanged) so they never appear twice in outputs.
        # Two components carry language-dependent Dataframe HEADERS (jobs / spill)
        # and are not label-registered, so they are appended as explicit extras.
        lang_switch_extras = [jobs_table, spill_table]
        lang_switch_outputs = [c for c, _k, _a in registry] + lang_switch_extras

        def switch_language(lang, config):
            qmode_choices = [(L("qmode_fast", lang), "distilled"),
                             (L("qmode_hq", lang), "two_stage_hq")]
            theme_choices = [(L("opt_dark", lang), "dark"),
                             (L("opt_light", lang), "light")]
            mode_choices = [(L("v2v_mode_none", lang), "none"),
                            (L("v2v_mode_v2v", lang), "v2v")]
            batch_mode_choices = [(L("batch_mode_add", lang), "add"),
                                  (L("batch_mode_replace", lang), "replace")]
            batch_out_choices = [(L("batch_out_auto", lang), "auto"),
                                 (L("batch_out_custom", lang), "custom")]
            nag_method_choices = [(L("nag_method_nag", lang), "nag"),
                                  (L("nag_method_vsf", lang), "vsf")]
            adapter_choices = build_adapter_choices(config, lang)
            updates = []
            for component, key, attr in registry:
                kwargs = {attr: L(key, lang)}
                if component is qmode or component is chain_qmode:
                    kwargs["choices"] = qmode_choices
                elif component is theme_dd:
                    kwargs["choices"] = theme_choices
                elif component is chain_mode:
                    kwargs["choices"] = mode_choices
                elif component is adapter:
                    kwargs["choices"] = adapter_choices
                elif component is batch_add_replace:
                    kwargs["choices"] = batch_mode_choices
                elif component is batch_out_mode:
                    kwargs["choices"] = batch_out_choices
                elif component is nag_method:
                    kwargs["choices"] = nag_method_choices
                updates.append(gr.update(**kwargs))
            updates.append(gr.update(headers=jobs_table_headers(lang)))
            updates.append(gr.update(headers=[L("col_res", lang), L("col_maxframes", lang)]))
            return updates

        lang_dd.change(
            switch_language, inputs=[lang_dd, config_state],
            outputs=lang_switch_outputs,
        ).then(
            # Re-apply the batch overrides AFTER switch_language resets the
            # Generate button to its plain label (no-op unless batch is on).
            on_batch_reapply_enable, inputs=[batch_enable, lang_dd],
            outputs=[generate_btn, gen_a2v_audio, num_frames],
        )
        lang_dd.change(lambda v: v, inputs=lang_dd, outputs=lang_state)
        # Batch table headers + rendered rows/summary localize on switch too.
        lang_dd.change(on_batch_lang_switch, inputs=[lang_dd, batch_rows_state],
                       outputs=[batch_table, batch_summary_md])
        # chain_duration_md is dynamic (not label-registered), so re-format it on
        # language switch; otherwise the old-language estimate string lingers.
        lang_dd.change(on_chain_estimate, inputs=chain_est_inputs,
                       outputs=chain_duration_md)

        # show_progress="hidden": startup background config fetches must not
        # spawn per-component status trackers -- with several simultaneous
        # demo.load events, Gradio 6.19's client can leave a tracker pending
        # forever, and its pointer-events:auto overlay steals clicks from
        # components in collapsed accordions / inactive tabs.
        demo.load(on_page_load, inputs=[config_state, lang_state],
                  outputs=[status_box, config_state, preset, adapter,
                           server_config_json, spill_table, config_retry_timer],
                  show_progress="hidden")

        # ---- Client-side HTML ``min`` attributes (bug fix) ----
        # The width/height/frames Numbers carry NO server-side ``minimum`` (an
        # in-progress sub-minimum keystroke -- e.g. "8" while typing "80" -- would
        # otherwise raise "Value 8 is less than minimum value 64" inside Gradio's
        # Number preprocess on every live duration/spill .change listener, which
        # surfaced as the reported queue/join errors). The ``min`` attribute is
        # still wanted on the rendered <input> purely as the browser's arrow-key /
        # spinner step-snap BASE (width/height -> 64n; frames -> 8n+1, i.e. base
        # 9), so re-apply it CLIENT-SIDE on page load, keyed off each Number's
        # elem_id. This is a fn=None load event whose ONLY effect is the js (the
        # same pattern as theme_dd.change's js= toggle) -- no server round-trip,
        # no outputs, so it cannot re-introduce the preprocess bound check.
        _frame_min_ids = "['gen_num_frames', " + ", ".join(
            f"'chain_c{_i}_frames'" for _i in range(1, CHAIN_MAX_CLIPS + 1)) + "]"
        _min_attr_js = """() => {
            const setMin = (id, v) => {
                const el = document.getElementById(id);
                if (!el) return;
                const inp = el.querySelector('input');
                if (inp) inp.setAttribute('min', v);
            };
            ['gen_width', 'gen_height', 'chain_width', 'chain_height']
                .forEach((id) => setMin(id, '64'));
            __FRAME_IDS__
                .forEach((id) => setMin(id, '9'));
        }""".replace("__FRAME_IDS__", _frame_min_ids)
        demo.load(None, js=_min_attr_js)

        # ---- Settings: model management (INDEPENDENT listeners) ----
        # Parent ruling: the shared on_page_load / on_refresh_config closures
        # above stay untouched — the Models section registers its own page-load
        # hook (gr.Blocks allows several) and its own Refresh button.
        model_dds = [model_dd_transformer, model_dd_text_encoder,
                     model_dd_video_vae, model_dd_audio]
        # The base dropdown is the FIRST output of refresh_model_dropdowns; the
        # four category dropdowns follow in MODEL_CATEGORIES order.
        model_all_dds = [model_base_dd, *model_dds]

        def _base_model_update(models_json):
            """gr.update for the base-model dropdown: its choices plus the
            active base pre-selected. A server that sends no ``base_models``
            leaves the dropdown exactly as it is (bare update)."""
            choices = build_base_model_choices(models_json)
            if not choices:
                return gr.update()
            ids = [value for _label, value in choices]
            active = active_base_model(models_json)
            return gr.update(choices=choices,
                             value=active if active in ids else ids[0])

        def refresh_model_dropdowns(lang, warn: bool = True):
            models_json, err = fetch_models_safe(api, lang)
            if err is not None:
                # Silent on page load (a dead server already warns via the
                # /config path); the explicit Refresh button does warn.
                if warn:
                    gr.Warning(err)
                return tuple(gr.update() for _ in model_all_dds)
            # The legacy top-level ``categories`` block always describes the
            # ACTIVE base model, so the four category dropdowns keep reading it
            # verbatim (no base_model argument) — a refresh always shows what
            # the pipeline is actually on.
            return (_base_model_update(models_json),) + tuple(
                gr.update(choices=build_model_choices(models_json, cat, lang),
                          value=model_active_value(models_json, cat))
                for cat in MODEL_CATEGORIES
            )

        model_refresh_btn.click(refresh_model_dropdowns, inputs=lang_state,
                                outputs=model_all_dds)
        # show_progress="hidden": same rationale as on_page_load's show_progress.
        demo.load(lambda lang: refresh_model_dropdowns(lang, warn=False),
                  inputs=lang_state, outputs=model_all_dds, show_progress="hidden")

        def on_base_model_change(base_id, lang):
            """Re-fill the four category dropdowns from the SELECTED base
            model's own listing (``base_models[].categories``), so the user
            picks parts of the base model they are about to load rather than
            of the one still loaded. Nothing is loaded here — the Load button
            sends the selection.

            Wired to ``.input`` (user edits only), never ``.change``: a
            programmatic ``gr.update(value=...)`` from refresh / page load /
            the post-Load re-pull also fires ``.change``, which would mean a
            second /models fetch on every page load and a warning toast while
            the server is down."""
            models_json, err = fetch_models_safe(api, lang)
            if err is not None:
                # User-initiated (like Refresh), so a failure is worth a toast.
                gr.Warning(err)
                return tuple(gr.update() for _ in MODEL_CATEGORIES)
            return tuple(
                gr.update(
                    choices=build_model_choices(models_json, cat, lang,
                                                base_model=base_id),
                    value=model_active_value(models_json, cat,
                                             base_model=base_id))
                for cat in MODEL_CATEGORIES
            )

        model_base_dd.input(on_base_model_change,
                            inputs=[model_base_dd, lang_state],
                            outputs=model_dds)

        def on_model_load_start(lang):
            # Disable the button + show the "takes minutes" notice while the
            # blocking POST runs (model load is synchronous, not a polled job).
            return gr.update(interactive=False), L("model_loading", lang)

        def on_model_load(tr, te, vv, au, base_id, lang):
            return load_selected_models(api, tr, te, vv, au, lang,
                                        base_model=base_id)

        model_load_btn.click(
            on_model_load_start, inputs=lang_state,
            outputs=[model_load_btn, model_status_box],
        ).then(
            on_model_load, inputs=[*model_dds, model_base_dd, lang_state],
            outputs=model_status_box,
        ).then(
            lambda: gr.update(interactive=True), outputs=model_load_btn,
        ).then(
            # Re-pull /models so the dropdowns reflect the new active marks.
            lambda lang: refresh_model_dropdowns(lang, warn=False),
            inputs=lang_state, outputs=model_all_dds,
        )

        # ---- Style LoRA tab events (INDEPENDENT listeners) ----
        # Same shape as the Models section: its own page-load hook + Refresh
        # button, no shared closure/outputs with the other tabs.
        def load_style_gallery(lang, current_names, warn: bool = True):
            try:
                loras = api.list_loras()
            except Exception as exc:
                if warn:
                    gr.Warning(L("style_list_failed", lang).format(err=exc))
                return gr.update(), current_names
            return (gr.update(value=build_style_gallery(loras, base_url)),
                    style_lora_names(loras))

        def on_style_reload(lang, current_names):
            # POST /loras/reload (explicit rescan) -> re-list -> rebuild gallery +
            # count toast. On failure the gallery/names are left as-is.
            try:
                counts = api.reload_loras()
            except Exception as exc:
                gr.Warning(L("style_reload_failed", lang).format(err=exc))
                return gr.update(), current_names
            try:
                loras = api.list_loras()
            except Exception as exc:
                gr.Warning(L("style_list_failed", lang).format(err=exc))
                return gr.update(), current_names
            gr.Info(L("style_reload_done", lang).format(
                total=counts.get("total", 0), styles=counts.get("styles", 0),
                controls=counts.get("controls", 0)))
            return (gr.update(value=build_style_gallery(loras, base_url)),
                    style_lora_names(loras))

        def on_style_select(prompt_val, names, lang, evt: gr.SelectData):
            # Gallery.select gives evt.index (the selected tile index); resolve
            # it to a name via style_names_state and APPEND a <lora:name:1.0:1.0>
            # token to the Generate-tab prompt (existing value preserved). The
            # 3-arg form surfaces the audio-strength slot up front (audio=1.0
            # numerically matches "follow video", so generation is unchanged).
            idx = evt.index
            if isinstance(idx, (list, tuple)):
                idx = idx[0] if idx else None
            if idx is None or not names or idx >= len(names):
                return gr.update()
            name = names[idx]
            token = f"<lora:{name}:1.0:1.0>"
            base = prompt_val or ""
            new_prompt = f"{base.rstrip()} {token}" if base.strip() else token
            gr.Info(L("style_added", lang).format(name=name))
            return gr.update(value=new_prompt)

        style_reload_btn.click(on_style_reload,
                               inputs=[lang_state, style_names_state],
                               outputs=[style_gallery, style_names_state])
        style_gallery.select(on_style_select,
                             inputs=[prompt, style_names_state, lang_state],
                             outputs=prompt)
        # show_progress="hidden": same rationale as on_page_load's show_progress.
        demo.load(lambda lang, names: load_style_gallery(lang, names, warn=False),
                  inputs=[lang_state, style_names_state],
                  outputs=[style_gallery, style_names_state],
                  show_progress="hidden")

    # Expose the registry + language-switch fn for the S6 handler (and tests).
    demo.label_registry = registry  # type: ignore[attr-defined]
    demo.switch_language = switch_language  # type: ignore[attr-defined]
    demo.lang_switch_outputs = lang_switch_outputs  # type: ignore[attr-defined]
    # Expose the ApiClient + config-fetch closures for tests (mirrors the
    # switch_language exposure above): lets a test swap in a mock transport
    # and drive on_page_load / on_config_retry directly, matching how the
    # Blocks graph actually wires them, without a live server.
    demo.api = api  # type: ignore[attr-defined]
    demo.on_page_load = on_page_load  # type: ignore[attr-defined]
    demo.on_config_retry = on_config_retry  # type: ignore[attr-defined]
    # Style LoRA closures (S2): let a test drive the gallery load / reload /
    # select paths against a mock transport without a live server.
    demo.load_style_gallery = load_style_gallery  # type: ignore[attr-defined]
    demo.on_style_reload = on_style_reload  # type: ignore[attr-defined]
    demo.on_style_select = on_style_select  # type: ignore[attr-defined]
    # Adapter -> reference-video enable/disable closure: lets a test drive the
    # greying logic directly (mirrors the on_page_load / on_style_select exposure).
    demo.on_adapter_change = on_adapter_change  # type: ignore[attr-defined]
    # Batch Regenerate closure (2nd-round FB, modification C): lets a test drive
    # the Skip-row refusal directly without a live event round-trip.
    demo.on_batch_regen = on_batch_regen  # type: ignore[attr-defined]
    # NAG (non-CFG Negative) toggle / method-fallback closures: lets a test
    # drive them directly (mirrors the on_adapter_change exposure above).
    demo.on_nag_enable_toggle = on_nag_enable_toggle  # type: ignore[attr-defined]
    demo.on_nag_method_change = on_nag_method_change  # type: ignore[attr-defined]
    # Models-section closures (base-model dropdown; Docs/PENDING_TASKS_CLOSED.md's
    # old §1-25, closed 2026-09-01): the refresh that fills base + 4 category
    # dropdowns, and the base-model .input handler.
    demo.refresh_model_dropdowns = refresh_model_dropdowns  # type: ignore[attr-defined]
    demo.on_base_model_change = on_base_model_change  # type: ignore[attr-defined]
    # "Set audios" closure (Docs/PENDING_TASKS_CLOSED.md's old §4-29, closed
    # 2026-09-01): lets a test drive the scan/merge/write path with an
    # explicit frame cap without a live event round-trip.
    demo.on_batch_set_audios = on_batch_set_audios  # type: ignore[attr-defined]
    return demo
