"""UI assembly: ``build_ui()`` assembles the top common bar + gr.Tabs
(Generate / Clip Chain / Jobs / Settings). Placeholder tabs are filled in by
later slices (S3-S6).
"""

from __future__ import annotations

import gradio as gr

from .adapters import ADAPTER_NONE, build_adapter_choices
from .api_client import ApiClient
from .formatting import format_status
from .handlers import make_chain_handler, make_generate_handler
from .i18n import L
from .presets import PRESETS, apply_preset, build_preset_choices, compute_spill_warning, pick_default_preset


def build_ui(base_url: str, api_key: str | None = None) -> gr.Blocks:
    api = ApiClient(base_url, api_key=api_key)
    generate = make_generate_handler(api)
    chain_generate = make_chain_handler(api)

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

    def on_page_load():
        status = refresh_status()
        try:
            cfg = api.get_config()
        except Exception:
            cfg = {}
        preset_update = gr.update(choices=build_preset_choices(cfg),
                                  value=pick_default_preset(cfg))
        # Rebuild the adapter choices from /config model.ic_loras; keep the
        # current value (ADAPTER_NONE "None", which is always the first choice).
        adapter_update = gr.update(choices=build_adapter_choices(cfg))
        return status, cfg, preset_update, adapter_update

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

            # ============================== Jobs =============================
            with gr.Tab(L("tab_jobs")) as tab_jobs:
                reg(tab_jobs, "tab_jobs", "label")
                reg(gr.Markdown(L("msg_coming")), "msg_coming", "value")

            # ============================ Settings ===========================
            with gr.Tab(L("tab_settings")) as tab_settings:
                reg(tab_settings, "tab_settings", "label")
                reg(gr.Markdown(f"### {L('h_ui')}"), "h_ui", "value")
                with gr.Row():
                    # NOTE: language handler is wired in S6; theme is a pure-JS
                    # frontend toggle (below).
                    lang_dd = reg(gr.Dropdown(
                        choices=[("English", "en"), ("日本語", "ja")],
                        value="en", label=L("lbl_lang"),
                    ), "lbl_lang")
                    theme_dd = reg(gr.Dropdown(
                        choices=[(L("opt_dark"), "dark"), (L("opt_light"), "light")],
                        value="dark", label=L("lbl_theme"),
                    ), "lbl_theme")

        # ---- events ----
        refresh_btn.click(refresh_status, outputs=status_box)
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
                    adapter, adapter_strength, ref_video, config_state],
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
                    *chain_clip_inputs, config_state],
            outputs=[chain_progress, chain_job, chain_video],
        )

        demo.load(on_page_load, outputs=[status_box, config_state, preset, adapter])

    # Expose the registry for the S6 language-switch handler (and tests).
    demo.label_registry = registry  # type: ignore[attr-defined]
    return demo
