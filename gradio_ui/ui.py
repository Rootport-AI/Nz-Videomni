"""UI assembly: ``build_ui()`` assembles the top common bar + gr.Tabs
(Generate / Clip Chain / Jobs / Settings). Placeholder tabs are filled in by
later slices (S3-S6).
"""

from __future__ import annotations

import gradio as gr

from .adapters import ADAPTER_NONE, build_adapter_choices
from .adapters import (
    MODEL_CATEGORIES,
    MODEL_DEFAULT,
    build_model_choices,
    model_active_value,
)
from .api_client import ApiClient
from .formatting import (
    build_jobs_rows,
    format_job_error,
    format_status,
    jobs_table_headers,
)
from .handlers import (
    delete_finished_jobs,
    fetch_config_safe,
    make_chain_handler,
    make_generate_handler,
    make_join_handler,
    on_config_retry_tick,
)
from .handlers import fetch_models_safe, load_selected_models
from .i18n import L
from .presets import PRESETS, apply_preset, build_preset_choices, compute_spill_warning, pick_default_preset


def build_spill_rows(config: dict | None) -> list[list]:
    """Rows for the Settings spill-free table: [resolution, max comfortable
    frames] built from /config limits.spill_free_frames."""
    spill = ((config or {}).get("limits") or {}).get("spill_free_frames") or {}
    return [[res, frames] for res, frames in spill.items()]


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

    def load_model() -> str:
        try:
            api.load_pipeline()
        except Exception as exc:
            return L("status_error").format(err=exc)
        return refresh_status()

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

    with gr.Blocks(title="LTX-AviUtl2-Bridge") as demo:
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
            load_btn = reg(gr.Button(L("btn_load_model"), scale=0), "btn_load_model", "value")
            unload_btn = reg(gr.Button(L("btn_unload_model"), scale=0), "btn_unload_model", "value")

        with gr.Tabs():
            # ============================ Generate ============================
            with gr.Tab(L("tab_gen")) as tab_gen:
                reg(tab_gen, "tab_gen", "label")
                with gr.Row():
                    # LEFT: inputs
                    with gr.Column(scale=3):
                        prompt = reg(gr.Textbox(label=L("lbl_prompt"), lines=3,
                                                placeholder=L("ph_prompt")), "lbl_prompt")
                        negative = reg(gr.Textbox(label=L("lbl_negative"),
                                                  value="blurry, low quality, distorted"),
                                       "lbl_negative")

                        # quality mode (two_stage_hq is non-selectable in S1)
                        qmode = reg(gr.Radio(
                            choices=[(L("qmode_fast"), "distilled"),
                                     (L("qmode_hq"), "two_stage_hq")],
                            value="distilled", label=L("lbl_qmode"),
                        ), "lbl_qmode")

                        preset = reg(gr.Dropdown(list(PRESETS.keys()), value="phase1_default",
                                                 label=L("lbl_preset"), info=L("hint_preset")),
                                     "lbl_preset")
                        with gr.Row():
                            width = reg(gr.Number(value=512, label=L("lbl_width"), precision=0),
                                        "lbl_width")
                            height = reg(gr.Number(value=320, label=L("lbl_height"), precision=0),
                                         "lbl_height")

                        crop_enabled = reg(gr.Checkbox(value=False, label=L("chk_crop")), "chk_crop")
                        with gr.Row(visible=False) as crop_row:
                            crop_w = reg(gr.Number(value=0, label=L("lbl_crop_w"), precision=0),
                                         "lbl_crop_w")
                            crop_h = reg(gr.Number(value=0, label=L("lbl_crop_h"), precision=0),
                                         "lbl_crop_h")

                        with gr.Row():
                            num_frames = reg(gr.Number(value=49, label=L("lbl_frames"), precision=0),
                                             "lbl_frames")
                            frame_rate = reg(gr.Number(value=24.0, label=L("lbl_fps")), "lbl_fps")

                        spill_warning = gr.Markdown("", visible=False,
                                                    elem_classes=["spill-warning"])

                        with gr.Row():
                            steps = reg(gr.Slider(1, 50, value=8, step=1, label=L("lbl_steps"),
                                                  interactive=False), "lbl_steps")
                            cfg = reg(gr.Slider(1.0, 12.0, value=1.0, step=0.1, label=L("lbl_cfg"),
                                                interactive=False), "lbl_cfg")
                        cap_lock = reg(gr.Markdown(L("cap_lock")), "cap_lock", "value")

                        seed = reg(gr.Number(value=-1, label=L("lbl_seed"), precision=0), "lbl_seed")

                        # accordion: up to 5 fixed keyframe slots (I2V multi-keyframe
                        # conditioning). Slot 1 suggests "start frame (0)"; slots 2-5
                        # are plain "frame position" slots, all defaulting to 0 (the
                        # server snaps any non-zero value to the 8n+1 grid).
                        kf_slots: list[tuple[object, object, object, object]] = []
                        with gr.Accordion(L("lbl_kf_accordion"), open=False) as kf_accordion:
                            reg(kf_accordion, "lbl_kf_accordion", "label")
                            for _slot_i in range(1, 6):
                                frame_key = "lbl_kf_frame_pos0" if _slot_i == 1 else "lbl_kf_frame_pos"
                                with gr.Row():
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
                            ref_video = reg(gr.File(
                                label=L("lbl_ref_video"), type="filepath",
                                file_count="single", file_types=["video"],
                            ), "lbl_ref_video")
                            reg(gr.Markdown(L("note_ref128"),
                                            elem_classes=["note"]), "note_ref128", "value")

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
                                     (L("v2v_mode_v2v"), "v2v"),
                                     (L("a2v_mode_a2v"), "a2v")],
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
                            reg(gr.Markdown(L("v2v_cap_join"), elem_classes=["note"]),
                                "v2v_cap_join", "value")

                        # ---- A2V panel (visible in a2v mode only) ----
                        with gr.Group(visible=False) as a2v_group:
                            a2v_audio = reg(gr.File(
                                label=L("a2v_lbl_audio"), type="filepath",
                                file_count="single", file_types=["audio"],
                            ), "a2v_lbl_audio")
                            reg(gr.Markdown(L("a2v_guide"), elem_classes=["note"]),
                                "a2v_guide", "value")
                            reg(gr.Markdown(L("a2v_cap_panel"), elem_classes=["note"]),
                                "a2v_cap_panel", "value")

                        chain_prompt = reg(gr.Textbox(label=L("lbl_prompt_shared"), lines=3,
                                                      placeholder=L("ph_prompt2")),
                                           "lbl_prompt_shared")
                        chain_negative = reg(gr.Textbox(label=L("lbl_negative"),
                                                        value="blurry, low quality, distorted"),
                                             "lbl_negative")
                        chain_qmode = reg(gr.Radio(
                            choices=[(L("qmode_fast"), "distilled"),
                                     (L("qmode_hq"), "two_stage_hq")],
                            value="distilled", label=L("lbl_qmode"),
                        ), "lbl_qmode")
                        with gr.Row():
                            chain_width = reg(gr.Number(value=1280, label=L("lbl_width"),
                                                        precision=0), "lbl_width")
                            chain_height = reg(gr.Number(value=768, label=L("lbl_height"),
                                                         precision=0), "lbl_height")
                        chain_crop_enabled = reg(gr.Checkbox(value=False, label=L("chk_crop")),
                                                 "chk_crop")
                        with gr.Row(visible=False) as chain_crop_row:
                            chain_crop_w = reg(gr.Number(value=0, label=L("lbl_crop_w"),
                                                         precision=0), "lbl_crop_w")
                            chain_crop_h = reg(gr.Number(value=0, label=L("lbl_crop_h"),
                                                         precision=0), "lbl_crop_h")
                        with gr.Row():
                            chain_fps = reg(gr.Number(value=24.0, label=L("lbl_fps")), "lbl_fps")
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

                        # clip list: 8 fixed slots (slots 1-2 enabled by default).
                        reg(gr.Markdown(f"### {L('h_clips')}"), "h_clips", "value")
                        # slot 1 — the only slot with a start image (clip 0).
                        with gr.Group():
                            c1_enabled = reg(gr.Checkbox(value=True, label=L("clip1")), "clip1")
                            c1_prompt = reg(gr.Textbox(label=L("lbl_clip_prompt"),
                                                       placeholder=L("ph_clip_prompt")),
                                            "lbl_clip_prompt")
                            with gr.Row():
                                c1_frames = reg(gr.Number(value=121, label=L("lbl_frames"),
                                                          precision=0), "lbl_frames")
                                c1_image = reg(gr.Image(label=L("lbl_clip_start_image"),
                                                        type="filepath"), "lbl_clip_start_image")
                                c1_strength = reg(gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                                            label=L("lbl_kf_strength")),
                                                  "lbl_kf_strength")
                            reg(gr.Markdown(L("cap_first_clip")), "cap_first_clip", "value")
                        chain_clip_slots.append((c1_enabled, c1_prompt, c1_frames,
                                                 c1_image, c1_strength))
                        # slots 2-8 — no start image (later timeline segments).
                        for _slot_i in range(2, 9):
                            clip_key = f"clip{_slot_i}"
                            with gr.Group():
                                cN_enabled = reg(gr.Checkbox(value=(_slot_i == 2),
                                                             label=L(clip_key)), clip_key)
                                cN_prompt = reg(gr.Textbox(label=L("lbl_clip_prompt"),
                                                           placeholder=L("ph_clip_prompt")),
                                                "lbl_clip_prompt")
                                cN_frames = reg(gr.Number(value=121, label=L("lbl_frames"),
                                                          precision=0), "lbl_frames")
                            chain_clip_slots.append((cN_enabled, cN_prompt, cN_frames))
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
                                              interactive=False), "lbl_base_url")
                api_badge = reg(
                    gr.Markdown(L("badge_set") if api_key else L("badge_unset")),
                    "badge_set" if api_key else "badge_unset", "value",
                )

                # ---- Behavior (poll cadence -> generate/chain handlers) ----
                reg(gr.Markdown(f"### {L('h_behavior')}"), "h_behavior", "value")
                with gr.Row():
                    poll_interval = reg(gr.Number(value=1.0, label=L("lbl_poll"),
                                                  minimum=0.1), "lbl_poll")
                    poll_timeout = reg(gr.Number(value=60, label=L("lbl_timeout"),
                                                 precision=0, minimum=1), "lbl_timeout")

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
        load_btn.click(load_model, outputs=status_box)
        unload_btn.click(unload_model, outputs=status_box)

        preset.change(
            apply_preset, inputs=[preset, config_state],
            outputs=[width, height, num_frames, crop_enabled, crop_w, crop_h,
                     crop_row, spill_warning],
        )
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

        generate_btn.click(
            generate,
            inputs=[prompt, negative, *kf_inputs, width, height,
                    crop_enabled, crop_w, crop_h, num_frames, frame_rate, seed,
                    adapter, adapter_strength, ref_video, config_state,
                    lang_state, poll_interval, poll_timeout],
            outputs=[progress_box, job_box, video_out],
        )

        # ---- Clip Chain events ----
        # Reuse the SAME quality-mode revert + crop-toggle handlers as Generate.
        chain_qmode.change(on_qmode_change, inputs=chain_qmode, outputs=chain_qmode)
        chain_crop_enabled.change(on_crop_toggle, inputs=chain_crop_enabled,
                                  outputs=chain_crop_row)

        chain_clip_inputs: list[object] = []
        for _slot in chain_clip_slots:
            chain_clip_inputs.extend(_slot)

        chain_generate_btn.click(
            chain_generate,
            inputs=[chain_prompt, chain_negative, chain_width, chain_height,
                    chain_crop_enabled, chain_crop_w, chain_crop_h, chain_fps, chain_seed,
                    chain_overlap, chain_overlap_strength,
                    *chain_clip_inputs, config_state,
                    lang_state, poll_interval, poll_timeout,
                    chain_mode, v2v_video, v2v_context, a2v_audio],
            outputs=[chain_progress, chain_job, chain_video],
        )

        # ---- V2V/A2V mode switching ----
        # Only the selected mode's panel is visible; the join panel is V2V-only;
        # clip 1's start image is unavailable in V2V (the frozen source tail
        # occupies clip 0's head — the server rejects the combination with 422,
        # and the handler prechecks it too).
        def on_chain_mode_change(mode):
            is_v2v = mode == "v2v"
            is_a2v = mode == "a2v"
            return (gr.update(visible=is_v2v), gr.update(visible=is_a2v),
                    gr.update(visible=is_v2v), gr.update(interactive=not is_v2v))

        chain_mode.change(on_chain_mode_change, inputs=chain_mode,
                          outputs=[v2v_group, a2v_group, v2v_join_panel, c1_image])

        # ---- V2V join ("Create joined version") ----
        # Hangs off the finished chain job id shown in chain_job; the checkbox
        # is the two-way "create the smoothed joined version / don't" switch
        # (the GUI only ever requests the server's default smoothed join).
        v2v_join_btn.click(
            chain_join,
            inputs=[chain_job, v2v_join_chk, lang_state],
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
                            (L("v2v_mode_v2v", lang), "v2v"),
                            (L("a2v_mode_a2v", lang), "a2v")]
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
                updates.append(gr.update(**kwargs))
            updates.append(gr.update(headers=jobs_table_headers(lang)))
            updates.append(gr.update(headers=[L("col_res", lang), L("col_maxframes", lang)]))
            return updates

        lang_dd.change(switch_language, inputs=[lang_dd, config_state],
                       outputs=lang_switch_outputs)
        lang_dd.change(lambda v: v, inputs=lang_dd, outputs=lang_state)

        demo.load(on_page_load, inputs=[config_state, lang_state],
                  outputs=[status_box, config_state, preset, adapter,
                           server_config_json, spill_table, config_retry_timer])

        # ---- Settings: model management (INDEPENDENT listeners) ----
        # Parent ruling: the shared on_page_load / on_refresh_config closures
        # above stay untouched — the Models section registers its own page-load
        # hook (gr.Blocks allows several) and its own Refresh button.
        model_dds = [model_dd_transformer, model_dd_text_encoder,
                     model_dd_video_vae, model_dd_audio]

        def refresh_model_dropdowns(lang, warn: bool = True):
            models_json, err = fetch_models_safe(api, lang)
            if err is not None:
                # Silent on page load (a dead server already warns via the
                # /config path); the explicit Refresh button does warn.
                if warn:
                    gr.Warning(err)
                return tuple(gr.update() for _ in MODEL_CATEGORIES)
            return tuple(
                gr.update(choices=build_model_choices(models_json, cat, lang),
                          value=model_active_value(models_json, cat))
                for cat in MODEL_CATEGORIES
            )

        model_refresh_btn.click(refresh_model_dropdowns, inputs=lang_state,
                                outputs=model_dds)
        demo.load(lambda lang: refresh_model_dropdowns(lang, warn=False),
                  inputs=lang_state, outputs=model_dds)

        def on_model_load_start(lang):
            # Disable the button + show the "takes minutes" notice while the
            # blocking POST runs (model load is synchronous, not a polled job).
            return gr.update(interactive=False), L("model_loading", lang)

        def on_model_load(tr, te, vv, au, lang):
            return load_selected_models(api, tr, te, vv, au, lang)

        model_load_btn.click(
            on_model_load_start, inputs=lang_state,
            outputs=[model_load_btn, model_status_box],
        ).then(
            on_model_load, inputs=[*model_dds, lang_state],
            outputs=model_status_box,
        ).then(
            lambda: gr.update(interactive=True), outputs=model_load_btn,
        ).then(
            # Re-pull /models so the dropdowns reflect the new active marks.
            lambda lang: refresh_model_dropdowns(lang, warn=False),
            inputs=lang_state, outputs=model_dds,
        )

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
    return demo
