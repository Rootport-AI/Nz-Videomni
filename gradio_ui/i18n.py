"""i18n label table. English is the default; Japanese entries are ported from the
previous UI + the approved mockup. Labels are plain text (Gradio labels do not
render HTML), so hints that were <span> in the mockup are folded into the text.
"""

from __future__ import annotations

_DEFAULT_LANG = "en"

LABELS: dict[str, dict[str, str]] = {
    "en": {
        # --- top common bar ---
        "btn_refresh": "Refresh",
        "btn_load_model": "Load model",
        "btn_unload_model": "Unload model",
        # --- status line words ---
        "st_pipeline": "Pipeline",
        "st_loaded": "loaded",
        "st_not_loaded": "not loaded",
        "st_free": "free",
        "st_lowvram": "Low-VRAM",
        "st_on": "ON",
        "st_off": "OFF",
        "st_queue": "Queue",
        "st_running": "running",
        "st_waiting": "waiting",
        "st_done": "done",
        "status_error": "status error: {err}",
        # --- tabs ---
        "tab_gen": "Generate",
        "tab_concat": "Clip Chain",
        "tab_jobs": "Jobs",
        "tab_settings": "Settings",
        # --- header ---
        "app_subtitle": "Verification UI — thin client over the frozen REST API (/api/v1/*).",
        "msg_coming": "Coming in a later slice.",
        # --- generate: left column ---
        "lbl_prompt": "Prompt (single generation & clip-chain shared base)",
        "ph_prompt": "A bustling downtown at dusk; crowds weave through the alleys as neon signs flicker on — like a scene from a movie trailer",
        "lbl_negative": "Negative prompt",
        "info_negative": "Disabled: the distilled model runs at CFG=1, so negative prompts have no effect.",
        "lbl_qmode": "Quality mode",
        "qmode_fast": "Fast (distilled) — 8 steps / CFG 1.0",
        "qmode_hq": "High quality (two_stage_hq) — backend support pending",
        "warn_hq_unsupported": "High-quality mode is not yet supported by the backend.",
        "lbl_preset": "Preset",
        "hint_preset": "Fetched automatically from the server /config",
        "lbl_width": "Width (multiple of 64)",
        "lbl_height": "Height (multiple of 64)",
        "chk_crop": "Crop output",
        "lbl_crop_w": "Crop width (min 32)",
        "lbl_crop_h": "Crop height (min 32)",
        "lbl_frames": "Frames (8n+1)",
        "lbl_fps": "Frame rate",
        "lbl_steps": "Steps",
        "lbl_cfg": "CFG scale",
        "cap_lock": "Fixed at 8 / 1.0 (distilled)",
        "lbl_seed": "Seed (-1 = random)",
        "warn_spill_limit": ("Exceeds the comfortable limit for {res} ({limit} frames): "
                              "generation still works but is much slower."),
        # --- generate: keyframe accordion (S3) ---
        "lbl_kf_accordion": "Keyframe images (I2V conditioning, up to 5)",
        "lbl_kf_use": "Use",
        "lbl_kf_image": "Keyframe image",
        "lbl_kf_frame_pos0": "Frame position (0 = start frame)",
        "lbl_kf_frame_pos": "Frame position",
        "lbl_kf_strength": "Strength",
        "cap_kf_grid": "Frame positions are snapped server-side to the 8n+1 grid.",
        "msg_kf_missing_image": "Slot {n}: enabled but no image selected.",
        "msg_kf_negative_frame": "Slot {n}: frame position must be 0 or greater.",
        "msg_uploading_keyframe": "Uploading keyframe {i}/{n}…",
        # --- generate: reference-video control (IC-LoRA) accordion (S4) ---
        "lbl_iclora_accordion": "Reference-video control (IC-LoRA)",
        "lbl_adapter": "Control adapter",
        "adapter_none": "None",
        "lbl_adapter_strength": "Adapter strength",
        "lbl_control_adherence": "Control adherence",
        "info_control_adherence": ("1.0 = follow the control signal (edges/skeleton) strictly; "
                                   "lower = interpret it more freely (recommended 0.5-0.7)."),
        "lbl_reference_strength": "Reference strength",
        "info_reference_strength": ("Usually keep at 1.0. Below 1.0 the reference video may pop "
                                    "or bleed through into the output (official warning)."),
        "lbl_ref_video": "Reference video (mp4/mov/webm/mkv, max 200 MB)",
        "note_ref128": ("When using a reference video, the output width and height must be "
                        "multiples of 128 (e.g. 1280×768); generation will not start otherwise."),
        # --- generate: right column ---
        "btn_generate": "Generate",
        "lbl_progress": "Progress",
        "lbl_jobid": "Job ID",
        "lbl_result": "Result",
        # --- generate: flow messages ---
        "msg_prompt_required": "Please enter a prompt.",
        "msg_upload_done": "Image uploaded: {image_id}",
        "msg_upload_failed": "Upload failed: {err}",
        "msg_job_busy": "Another job is already running (409).",
        "msg_generate_error": "generate error {code}: {text}",
        "msg_generate_failed": "generate failed: {err}",
        "msg_job_started": "Job started ({mode}): {job_id}",
        "msg_generating": "Generating… {pct:.0%} (step {step}/{total})",
        # F3: step-less variant (never print "step None/None") + phase labels
        # appended when the backend reports one.
        "msg_generating_pct": "Generating… {pct:.0%}",
        "stage_encoding": "Encoding",
        "stage_denoise_s1": "Denoising (stage 1)",
        "stage_denoise_s2": "Denoising (stage 2)",
        "stage_denoise": "Denoising",
        "stage_upsample": "Upsampling",
        "stage_decode": "Decoding",
        # Chain clip progress: appended while a chain job reports which clip
        # (stage-1 segment) it is working on, and to the completion line.
        "msg_clip_progress": "clip {clip}/{total}",
        "msg_all_clips_done": "all {n} clips processed",
        "msg_poll_failed": "polling failed: {err}",
        "msg_completing": "Completed. Fetching video…",
        "msg_completed": "Completed: {job_id}",
        "msg_failed": "{status}: {error}",
        "msg_timeout": "Timed out.",
        # --- generate: reference-video flow messages (S4) ---
        "msg_ref_video_required": "Please select a reference video for the control adapter.",
        "msg_ref_bad_extension": "Reference video type not allowed. Allowed: {exts}",
        "msg_ref_too_large": "Reference video exceeds the {limit} MB limit.",
        "msg_ref_resolution": ("Reference-video jobs require width and height divisible by 128 "
                               "(e.g. 1280×768). Adjust the size and retry."),
        "msg_uploading_ref": "Uploading reference video…",
        # --- shared API error-envelope hints (S4), one line each, actionable ---
        "apierr_JOB_BUSY": "Another job is already running. Wait for it to finish, then retry.",
        "apierr_UPLOAD_INVALID_TYPE": "Unsupported file type. Use an allowed image/video format.",
        "apierr_UPLOAD_TOO_LARGE": "The file is too large. Reduce the file size and retry.",
        "apierr_IMAGE_NOT_FOUND": "The uploaded image was not found. Re-upload the keyframe image.",
        "apierr_REFERENCE_VIDEO_NOT_FOUND": ("The reference video was not found. Re-upload the "
                                             "reference video."),
        "apierr_LORA_NOT_FOUND": "The selected control adapter is not registered on the server.",
        "apierr_LORA_PREPROCESS_CONFLICT": ("The selected adapters need conflicting preprocessing. "
                                            "Use one control adapter at a time."),
        "apierr_REFERENCE_RESOLUTION_INVALID": ("Reference-video jobs require width and height "
                                                "divisible by 128 (e.g. 1280×768)."),
        "apierr_JOB_NOT_FOUND": "Job not found. It may have already been deleted.",
        "apierr_VIDEO_NOT_READY": "The video is not ready yet. Wait until the job completes.",
        "apierr_PIPELINE_LOAD_FAILED": ("Failed to load the model pipeline. Check server VRAM / "
                                        "logs and retry."),
        "apierr_GPU_OOM": "Out of GPU memory. Reduce the resolution or frame count and retry.",
        "apierr_GENERATION_FAILED": "Generation failed on the server. Check the server logs.",
        "apierr_UNAUTHORIZED": "Authentication failed. Check the API key.",
        "apierr_VALIDATION_ERROR": "The request was rejected by validation. See the details below.",
        # --- clip chain tab (S5) ---
        "chain_lora_ignored": "Clip Chain does not support LoRA yet — the <lora:...> tag(s) were ignored.",
        "lbl_overlap": "Transition frames (overlap between clips, 1-8)",
        "lbl_overlap_strength": "Transition strength",
        "cap_crossfade": "Clips are joined with a cross-fade-like blend using these settings.",
        "h_clips": "Clip list",
        "lbl_clip_prompt": "Clip prompt",
        "ph_clip_prompt": "Leave blank to use the shared prompt",
        "lbl_clip_start_image": "Start image (first clip only)",
        "cap_first_clip": "A start image can be set only on the first clip.",
        "cap_clip_count": "Enable 2 to 8 clips.",
        "clip1": "Clip 1", "clip2": "Clip 2", "clip3": "Clip 3", "clip4": "Clip 4",
        "clip5": "Clip 5", "clip6": "Clip 6", "clip7": "Clip 7", "clip8": "Clip 8",
        "btn_concat": "Generate chain",
        # --- clip chain: flow / precheck messages (S5) ---
        "msg_bad_dimension": "Width and height must be multiples of 64.",
        "msg_size_limit": "Width/height exceed the server limit ({maxw}×{maxh}).",
        "msg_crop_range": "Crop size must be at least 32 and not exceed the generation size.",
        "msg_fps_range": "Frame rate must be between 1 and 60.",
        "msg_chain_clip_count": "Enable between 2 and 8 clips.",
        "msg_chain_bad_frames": "Clip {n}: frames must be 8n+1 and between 9 and 481.",
        "msg_chain_overlap_too_large": ("Transition frames ({kv}) must be smaller than the shortest "
                                        "enabled clip allows (max {maxkv})."),
        "msg_chain_total_frames": ("The chain timeline ({total} frames) exceeds the {cap}-frame cap. "
                                   "Reduce clip count or clip lengths."),
        "msg_chain_geometry": "The chain geometry is invalid: {err}",
        "msg_chain_started": "Chain job started ({n} clips): {job_id}",
        # --- clip chain: generation mode (none / V2V / A2V) ---
        "v2v_mode_label": "Generation mode",
        "v2v_mode_none": "None (normal clip chain)",
        "v2v_mode_v2v": "V2V continuation — generate a continuation of an uploaded video",
        "a2v_mode_a2v": "A2V audio-driven — match the video (lip movement) to uploaded audio",
        "v2v_cap_mode": ("V2V and A2V cannot be combined — pick one mode. "
                         "\"None\" is the ordinary 2-8 clip chain."),
        # --- clip chain: V2V panel ---
        "v2v_lbl_video": "Source video (mp4/mov/webm/mkv, max 200 MB)",
        "v2v_lbl_context": "Context frames (source tail to continue from, 8n+1)",
        # F4: usage guide (same rank as a2v_guide).
        "v2v_guide": ("**Getting good results with V2V**\n\n"
                      "This feature reads the tail end of the video you upload — the stretch set "
                      "by the context frame count — and generates what comes next, both picture "
                      "and sound. Write your prompt as a continuation of the *same* scene that "
                      "the original video is already showing. If you describe a different scene "
                      "instead, the content will lurch abruptly the moment the reference stretch "
                      "ends. Don't re-instruct any dialogue that has already been spoken in the "
                      "original video, or the character will say it a second time. If you want "
                      "the music to keep going, say so explicitly in the prompt — state that the "
                      "music continues. Finally, a larger context frame count makes the "
                      "generation more stable, so raise it as far as the length of your original "
                      "video allows."),
        "v2v_cap_panel": ("With V2V a single clip is enough (the frozen source tail acts as the "
                          "previous segment). The source video must have at least this many frames, "
                          "and clip 1 must be longer than the context so a new part remains. "
                          "Clip 1's start image cannot be used (the source occupies the head). "
                          "The delivered video is the NEW part only, so its length is roughly "
                          "(total frames - context frames) / 24 seconds "
                          "(e.g. 225 total frames with 73 context frames is about 6.3s)."),
        "v2v_chk_join": "Also create a version joined to the source video (crossfade the audio seam)",
        # F5: crossfade length selector for the joined version.
        "v2v_lbl_crossfade": "Audio crossfade length at the join (ms)",
        "v2v_cap_join": ("In addition to the video of the newly generated portion alone, this also "
                         "exports a combined version joined to your original clip, with the audio "
                         "seam smoothed by a crossfade. Turn it off and the audio may sound like it "
                         "cuts out at the join."),
        "v2v_btn_join": "Create joined version",
        "v2v_lbl_joined": "Joined result (source + continuation)",
        # --- clip chain: V2V flow messages ---
        "v2v_msg_video_required": "Please select a source video for V2V continuation.",
        "v2v_msg_bad_extension": "Source video type not allowed. Allowed: {exts}",
        "v2v_msg_too_large": "Source video exceeds the {limit} MB limit.",
        "v2v_msg_bad_context": "Context frames must be 8n+1 between {mincf} and {maxcf}.",
        "v2v_msg_context_ge_clip": ("Context frames ({cf}) must be smaller than clip 1's frames "
                                    "({clip}) so a new part remains to generate."),
        "v2v_msg_image_conflict": ("V2V cannot use a start image on clip 1 (the source tail "
                                   "occupies the head). Remove the image and retry."),
        "v2v_msg_clip_count": "With V2V enable 1 to 8 clips.",
        "v2v_msg_uploading": "Uploading source video…",
        "v2v_msg_join_disabled": ("Joined-version creation is turned off. Enable the checkbox "
                                  "to create one."),
        "v2v_msg_no_job": "No completed chain job yet. Generate a V2V chain first.",
        "v2v_msg_joining": "Creating the joined version (server-side)…",
        "v2v_msg_join_done": "Joined version created ({mode}): {job_id}",
        "v2v_msg_join_failed": "Failed to create the joined version: {err}",
        # --- clip chain: A2V panel ---
        "a2v_lbl_audio": "Source audio (wav/mp3/m4a/aac/flac/ogg, max 50 MB)",
        "a2v_guide": ("**Getting good results with A2V**\n\n"
                      "This feature generates video with mouth movements matched to the audio you "
                      "upload. It works best with a close-up of a single speaker in a composition "
                      "with limited movement — in that setting the lip movements line up almost "
                      "exactly with the audio. In busier scenes, or when the subject moves around "
                      "within a wide shot, the match tends to weaken. If the result isn't "
                      "convincing, try simplifying the composition first. Setting a start image on "
                      "clip 1 is an effective way to pin the close-up composition.\n\n"
                      "Prompt example (close-up, single speaker):\n\n"
                      "`Cinematic trailer shot, extreme close-up of a weathered detective speaking "
                      "directly to camera in a dim office, warm lamplight raking across his face, "
                      "shallow depth of field, subtle head movement, lips articulating each word "
                      "clearly, tense and intimate mood, film grain, 35mm.`\n\n"
                      "Clear speech works best, and clips somewhat longer than 5-6 seconds tend "
                      "to be more stable."),
        "a2v_cap_panel": ("A2V uses exactly ONE clip (the audio drives that whole clip). The audio "
                          "must be at least as long as the video — shorter audio is rejected. Your "
                          "uploaded audio is kept as-is in the output."),
        # --- clip chain: A2V flow messages ---
        "a2v_msg_audio_required": "Please select a source audio file for A2V.",
        "a2v_msg_bad_extension": "Audio type not allowed. Allowed: {exts}",
        "a2v_msg_too_large": "Audio exceeds the {limit} MB limit.",
        "a2v_msg_clip_count": "A2V uses exactly 1 clip — enable clip 1 only.",
        "a2v_msg_uploading": "Uploading audio…",
        # --- V2V/A2V + join API error-envelope hints ---
        "apierr_SOURCE_VIDEO_NOT_FOUND": ("The source video was not found on the server. "
                                          "Re-upload the source video."),
        "apierr_SOURCE_VIDEO_TOO_SHORT": ("The source video has fewer frames than the requested "
                                          "context. Reduce the context frames or use a longer video."),
        "apierr_SOURCE_AUDIO_NOT_FOUND": ("The source audio was not found on the server. "
                                          "Re-upload the audio file."),
        "apierr_SOURCE_AUDIO_TOO_SHORT": ("The audio is shorter than the video timeline. Use longer "
                                          "audio or fewer frames."),
        "apierr_JOB_NOT_JOINABLE": ("This job is not a V2V continuation, so there is nothing to "
                                    "join it to."),
        "apierr_JOIN_FAILED": "Joining failed on the server. Check the server logs.",
        "apierr_JOINED_NOT_READY": "The joined version has not been created yet. Create it first.",
        # --- settings: interface section ---
        "h_ui": "Interface",
        "lbl_lang": "Language",
        "lbl_theme": "Theme",
        "opt_dark": "Dark",
        "opt_light": "Light",
        # --- jobs tab (S6) ---
        "btn_jobs_refresh": "Refresh list",
        "col_job_id": "Job ID",
        "col_status": "Status",
        "col_progress": "Progress",
        "col_created": "Created",
        "col_completed": "Completed",
        "col_error": "Error",
        "lbl_job_detail": "Selected job details",
        "lbl_done_video": "Completed video",
        "lbl_job_error": "Error",
        "btn_job_action": "Cancel / Delete",
        "msg_jobs_refresh_failed": "Failed to fetch the job list: {err}",
        "msg_job_select_failed": "Failed to load job details: {err}",
        "msg_no_job_selected": "Select a job in the list first.",
        "msg_job_cancel_requested": "Cancel requested: {job_id}",
        "msg_job_deleted": "Deleted: {job_id}",
        "msg_job_action_failed": "Operation failed: {err}",
        # --- settings tab (S6) ---
        "h_conn": "Connection",
        "lbl_base_url": "base_url",
        "lbl_apikey": "API key",
        "badge_set": "Configured",
        "badge_unset": "Not set",
        "h_behavior": "Behavior",
        "lbl_poll": "Polling interval (s)",
        "lbl_timeout": "Timeout (min)",
        "h_server": "Server config (read-only)",
        "sum_config": "Raw /config JSON",
        "warn_config_load_failed": "Failed to load server settings: {err}. Retrying automatically…",
        "warn_config_retry_exhausted": ("Still unable to load server settings after {n} attempts: "
                                        "{err}. Use Refresh to try again."),
        "lbl_maxframes": "Comfortable frame-count limits by resolution",
        "col_res": "Resolution",
        "col_maxframes": "Max frames",
        "cap_over": "Exceeding these still works, but generation becomes much slower.",
        "h_danger": "Danger zone",
        "chk_danger": "Allow these operations",
        "btn_unload_confirm": "Unload model (with confirmation)",
        "btn_purge": "Delete all finished jobs",
        "msg_purge_done": "Deleted {n} finished job(s).",
        "msg_purge_failed": "Failed to delete jobs: {err}",
        # --- settings tab: model management (model_ prefix) ---
        "model_section_title": "Models",
        "model_cat_transformer": "Video model (transformer)",
        "model_cat_text_encoder": "Text encoder (Gemma)",
        "model_cat_video_vae": "Video VAE",
        "model_cat_audio": "Audio model (audio VAE + vocoder)",
        "model_btn_refresh": "Refresh model list",
        "model_btn_load": "Load selected models",
        "model_missing": "file missing",
        "model_loading": ("Loading models… switching rebuilds the engine and can take "
                          "several minutes."),
        "model_load_ok": "Models loaded: {models}",
        "model_load_failed": "Model load failed:\n{err}",
        "model_fetch_failed": "Failed to fetch the model list: {err}",
        "model_hint": ("Selections apply when you press Load. 'default' is the stock "
                       "combination. Switching restarts the engine worker (a few "
                       "minutes). If a load fails, select 'default' everywhere and "
                       "Load again."),
        "apierr_MODEL_NOT_FOUND": ("Unknown model name. Refresh the model list and pick "
                                   "again."),
        "apierr_MODEL_FILE_MISSING": ("The model file is missing on disk. Re-download it "
                                      "or pick another model."),
        "apierr_MODEL_INCOMPATIBLE": ("The selected file is not a valid model for that "
                                      "slot. Pick another model."),
        # --- style / character LoRA tab (S2) ---
        "tab_style_lora": "Style LoRA",
        "style_gallery_label": "Style / character LoRAs",
        "style_reload_btn": "Reload LoRA list",
        "style_note": ("Click a LoRA below to append a <lora:name:1.0> token to the Generate "
                       "tab's prompt. Adjust the weight by editing the number in the prompt "
                       "(0–2.0; 1.0 = the strength the LoRA was trained for). Control LoRAs "
                       "(canny / pose / upscaler) are not shown here — use them as before from "
                       "the reference-video adapter field on the Generate tab."),
        "style_added": "Added to the prompt: {name}",
        "style_reload_done": ("LoRAs reloaded: {total} total ({styles} style, "
                              "{controls} control)."),
        "style_reload_failed": "Failed to reload LoRAs: {err}",
        "style_list_failed": "Failed to load the LoRA list: {err}",
        # --- prompt-embedded <lora:...> messages (S2) ---
        "lora_msg_unknown": ("Unknown LoRA name(s) in the prompt: {names}. Remove or fix the "
                             "<lora:...> token(s) and retry."),
        "lora_warn_weight_clamp": ("LoRA <{name}> weight {given} is out of range (0–2.0); "
                                   "clamped to {clamped}."),
        "lora_msg_list_failed": ("Failed to look up the LoRA list for the <lora:...> tokens: "
                                 "{err}"),
    },
    "ja": {
        # --- top common bar ---
        "btn_refresh": "状態更新",
        "btn_load_model": "モデル読込",
        "btn_unload_model": "モデル解放",
        # --- status line words ---
        "st_pipeline": "パイプライン",
        "st_loaded": "読込済",
        "st_not_loaded": "未読込",
        "st_free": "空き",
        "st_lowvram": "省VRAM",
        "st_on": "ON",
        "st_off": "OFF",
        "st_queue": "キュー",
        "st_running": "実行中",
        "st_waiting": "待機",
        "st_done": "完了",
        "status_error": "状態取得エラー: {err}",
        # --- tabs ---
        "tab_gen": "生成",
        "tab_concat": "クリップ連結",
        "tab_jobs": "ジョブ",
        "tab_settings": "設定",
        # --- header ---
        "app_subtitle": "検証用UI — 凍結REST API (/api/v1/*) の薄いクライアント。",
        "msg_coming": "後のスライスで実装予定。",
        # --- generate: left column ---
        "lbl_prompt": "プロンプト(単発生成・クリップ連結の共通ベース)",
        "ph_prompt": "夕暮れの賑やかな下町、行き交う人々、ネオンが灯りはじめる路地。映画のワンシーンのように——",
        "lbl_negative": "ネガティブプロンプト",
        "info_negative": "無効: 蒸留モデルはCFG=1で動作するため、ネガティブプロンプトは効きません。",
        "lbl_qmode": "品質モード",
        "qmode_fast": "高速 (distilled) — 8ステップ / CFG 1.0",
        "qmode_hq": "高品質 (two_stage_hq) — バックエンド未対応",
        "warn_hq_unsupported": "高品質モードはまだバックエンドが対応していません。",
        "lbl_preset": "プリセット",
        "hint_preset": "サーバの /config から自動取得",
        "lbl_width": "幅 (64の倍数)",
        "lbl_height": "高さ (64の倍数)",
        "chk_crop": "出力をクロップ",
        "lbl_crop_w": "クロップ幅 (32以上)",
        "lbl_crop_h": "クロップ高さ (32以上)",
        "lbl_frames": "フレーム数 (8n+1)",
        "lbl_fps": "フレームレート",
        "lbl_steps": "ステップ数",
        "lbl_cfg": "CFGスケール",
        "cap_lock": "8 / 1.0 に固定 (distilled)",
        "lbl_seed": "シード (-1 = ランダム)",
        "warn_spill_limit": "解像度 {res} の快適上限 ({limit} フレーム) を超えています: 生成は可能ですが大幅に低速化します。",
        # --- generate: keyframe accordion (S3) ---
        "lbl_kf_accordion": "キーフレーム画像 (I2V条件付け・最大5枚)",
        "lbl_kf_use": "使用",
        "lbl_kf_image": "キーフレーム画像",
        "lbl_kf_frame_pos0": "フレーム位置 (0 = 開始フレーム)",
        "lbl_kf_frame_pos": "フレーム位置",
        "lbl_kf_strength": "適用強度",
        "cap_kf_grid": "フレーム位置はサーバ側で8n+1の格子に合わせられます。",
        "msg_kf_missing_image": "スロット{n}: 有効ですが画像が選択されていません。",
        "msg_kf_negative_frame": "スロット{n}: フレーム位置は0以上にしてください。",
        "msg_uploading_keyframe": "キーフレームをアップロード中… {i}/{n}",
        # --- generate: reference-video control (IC-LoRA) accordion (S4) ---
        "lbl_iclora_accordion": "参照動画による制御 (IC-LoRA)",
        "lbl_adapter": "制御アダプタ",
        "adapter_none": "なし",
        "lbl_adapter_strength": "アダプタ強度",
        "lbl_control_adherence": "制御追従度",
        "info_control_adherence": ("1.0=制御信号（輪郭線・骨格）に厳密に従う。"
                                   "下げるほど自由に解釈します（推奨 0.5〜0.7）。"),
        "lbl_reference_strength": "参照強度",
        "info_reference_strength": ("通常は 1.0 のままにします。1.0 未満では参照映像が出力に"
                                    "滲み込む（bleed-through）ことがあります（公式の注意）。"),
        "lbl_ref_video": "参照動画 (mp4/mov/webm/mkv・最大200MB)",
        "note_ref128": ("参照動画を使う場合、出力の幅と高さは128の倍数にしてください"
                        "(例: 1280×768)。満たさない場合は生成を開始しません。"),
        # --- generate: right column ---
        "btn_generate": "生成",
        "lbl_progress": "進捗",
        "lbl_jobid": "ジョブID",
        "lbl_result": "結果",
        # --- generate: flow messages ---
        "msg_prompt_required": "プロンプトを入力してください。",
        "msg_upload_done": "画像アップロード完了: {image_id}",
        "msg_upload_failed": "アップロード失敗: {err}",
        "msg_job_busy": "別のジョブが実行中です (409)。",
        "msg_generate_error": "generate エラー {code}: {text}",
        "msg_generate_failed": "generate 失敗: {err}",
        "msg_job_started": "ジョブ開始 ({mode}): {job_id}",
        "msg_generating": "生成中… {pct:.0%} (step {step}/{total})",
        # F3: step情報なしの表示("step None/None"を出さない)+ 工程名ラベル。
        "msg_generating_pct": "生成中… {pct:.0%}",
        "stage_encoding": "エンコード中",
        "stage_denoise_s1": "デノイズ中 (stage 1)",
        "stage_denoise_s2": "デノイズ中 (stage 2)",
        "stage_denoise": "デノイズ中",
        "stage_upsample": "アップサンプル中",
        "stage_decode": "デコード中",
        # クリップ連結の進捗: 何個目のクリップを処理中かを進捗行と完了行に添える。
        "msg_clip_progress": "クリップ {clip}/{total}",
        "msg_all_clips_done": "全{n}クリップ処理済み",
        "msg_poll_failed": "ポーリング失敗: {err}",
        "msg_completing": "完了。動画を取得中…",
        "msg_completed": "完了: {job_id}",
        "msg_failed": "{status}: {error}",
        "msg_timeout": "タイムアウト。",
        # --- generate: reference-video flow messages (S4) ---
        "msg_ref_video_required": "制御アダプタ用の参照動画を選択してください。",
        "msg_ref_bad_extension": "参照動画の形式が許可されていません。許可形式: {exts}",
        "msg_ref_too_large": "参照動画が上限 {limit} MB を超えています。",
        "msg_ref_resolution": ("参照動画を使う場合、幅と高さは128の倍数にしてください"
                               "(例: 1280×768)。サイズを調整して再試行してください。"),
        "msg_uploading_ref": "参照動画をアップロード中…",
        # --- shared API error-envelope hints (S4), one line each, actionable ---
        "apierr_JOB_BUSY": "別のジョブが実行中です。終了を待ってから再試行してください。",
        "apierr_UPLOAD_INVALID_TYPE": "対応していないファイル形式です。許可された画像/動画形式を使ってください。",
        "apierr_UPLOAD_TOO_LARGE": "ファイルが大きすぎます。サイズを小さくして再試行してください。",
        "apierr_IMAGE_NOT_FOUND": "アップロードした画像が見つかりません。キーフレーム画像を再アップロードしてください。",
        "apierr_REFERENCE_VIDEO_NOT_FOUND": "参照動画が見つかりません。参照動画を再アップロードしてください。",
        "apierr_LORA_NOT_FOUND": "選択した制御アダプタはサーバに登録されていません。",
        "apierr_LORA_PREPROCESS_CONFLICT": "選択したアダプタの前処理が競合しています。制御アダプタは一度に1つにしてください。",
        "apierr_REFERENCE_RESOLUTION_INVALID": "参照動画のジョブは幅と高さを128の倍数にする必要があります(例: 1280×768)。",
        "apierr_JOB_NOT_FOUND": "ジョブが見つかりません。すでに削除された可能性があります。",
        "apierr_VIDEO_NOT_READY": "動画はまだ準備できていません。ジョブの完了を待ってください。",
        "apierr_PIPELINE_LOAD_FAILED": "モデルパイプラインの読み込みに失敗しました。サーバのVRAM/ログを確認して再試行してください。",
        "apierr_GPU_OOM": "GPUメモリが不足しています。解像度やフレーム数を減らして再試行してください。",
        "apierr_GENERATION_FAILED": "サーバ側で生成に失敗しました。サーバのログを確認してください。",
        "apierr_UNAUTHORIZED": "認証に失敗しました。APIキーを確認してください。",
        "apierr_VALIDATION_ERROR": "リクエストが検証で拒否されました。詳細は以下を参照してください。",
        # --- clip chain tab (S5) ---
        "chain_lora_ignored": "クリップ連結ではLoRAは未対応のため、<lora:...>タグを無視しました。",
        "lbl_overlap": "つなぎ目のフレーム数 (クリップ間のオーバーラップ・1〜8)",
        "lbl_overlap_strength": "つなぎ目の強さ",
        "cap_crossfade": "クリップ間はこの設定でクロスフェード的に接続されます。",
        "h_clips": "クリップ一覧",
        "lbl_clip_prompt": "クリッププロンプト",
        "ph_clip_prompt": "空欄なら共通プロンプトを使用",
        "lbl_clip_start_image": "開始画像 (先頭クリップのみ)",
        "cap_first_clip": "開始画像を指定できるのは先頭クリップのみです。",
        "cap_clip_count": "有効にするクリップは2〜8個。",
        "clip1": "クリップ1", "clip2": "クリップ2", "clip3": "クリップ3", "clip4": "クリップ4",
        "clip5": "クリップ5", "clip6": "クリップ6", "clip7": "クリップ7", "clip8": "クリップ8",
        "btn_concat": "連結生成",
        # --- clip chain: flow / precheck messages (S5) ---
        "msg_bad_dimension": "幅と高さは64の倍数にしてください。",
        "msg_size_limit": "幅/高さがサーバの上限 ({maxw}×{maxh}) を超えています。",
        "msg_crop_range": "クロップサイズは32以上かつ生成サイズ以下にしてください。",
        "msg_fps_range": "フレームレートは1〜60の範囲にしてください。",
        "msg_chain_clip_count": "有効にするクリップは2〜8個にしてください。",
        "msg_chain_bad_frames": "クリップ{n}: フレーム数は8n+1かつ9〜481にしてください。",
        "msg_chain_overlap_too_large": "つなぎ目フレーム数 ({kv}) は最短クリップが許す値 (最大 {maxkv}) より小さくしてください。",
        "msg_chain_total_frames": "連結タイムライン ({total} フレーム) が上限 {cap} フレームを超えています。クリップ数か長さを減らしてください。",
        "msg_chain_geometry": "連結ジオメトリが不正です: {err}",
        "msg_chain_started": "連結ジョブ開始 ({n} クリップ): {job_id}",
        # --- clip chain: generation mode (none / V2V / A2V) ---
        "v2v_mode_label": "生成モード",
        "v2v_mode_none": "なし（通常のクリップ連結）",
        "v2v_mode_v2v": "V2V継続 — アップロード動画の続きを生成",
        "a2v_mode_a2v": "A2V音声駆動 — アップロード音声に口の動きを合わせる",
        "v2v_cap_mode": "V2VとA2Vは同時に使えません。どちらか一方を選んでください。「なし」は従来どおりの2〜8クリップ連結です。",
        # --- clip chain: V2V panel ---
        "v2v_lbl_video": "元動画 (mp4/mov/webm/mkv・最大200MB)",
        "v2v_lbl_context": "参照フレーム数 (元動画の末尾から続きの手がかりにする長さ・8n+1)",
        # F4: 使いこなしガイド (a2v_guideと同格)。
        "v2v_guide": ("**V2Vを使いこなすには**\n\n"
                      "この機能は、アップロードした元動画の末尾（参照フレーム数で指定した長さの区間）を"
                      "モデルに読み取らせ、その続きの映像と音声を生成します。"
                      "プロンプトには、元動画がすでに映している場面と同じシーンの「続き」を書いてください。"
                      "別の新しいシーンを書いてしまうと、参照区間が終わった瞬間に内容が急に飛んでしまう原因になります。"
                      "元動画の中ですでに話されたセリフは、プロンプトで改めて指示しないでください"
                      "（同じセリフをもう一度言い直してしまいます）。"
                      "音楽を続けたい場合は、音楽が続いていることをプロンプトにはっきり書いてください。"
                      "また、参照フレーム数は大きいほど生成が安定するので、"
                      "元動画の長さが許す範囲でできるだけ大きくするのがおすすめです。"),
        "v2v_cap_panel": ("V2Vではクリップ1個から生成できます（凍結された元動画の末尾が直前のセグメントの役割を果たします）。"
                          "元動画にはこのフレーム数以上の長さが必要で、クリップ1のフレーム数は参照フレーム数より大きくしてください"
                          "（続きとして生成する余地を残すため）。クリップ1の開始画像は使えません（先頭は元動画が占有します）。"
                          "出力される動画は新しく生成した部分のみで、長さはおおむね"
                          "（総フレーム数−参照フレーム数）÷24秒になります"
                          "（例: 総225フレーム・参照73フレームなら約6.3秒）。"),
        "v2v_chk_join": "元動画と結合した完成版も作る（音声の継ぎ目をクロスフェード）",
        # F5: 結合版のクロスフェード長セレクタ。
        "v2v_lbl_crossfade": "結合部の音声クロスフェード長 (ms)",
        "v2v_cap_join": ("新しく生成した部分だけの動画に加えて、元動画とつないだ完成版も書き出します。"
                         "つなぎ目の音の段差はクロスフェードで滑らかにします。"
                         "オフにすると、つなぎ目で音が途切れて聞こえることがあります。"),
        "v2v_btn_join": "結合版を作成",
        "v2v_lbl_joined": "結合版 (元動画+続き)",
        # --- clip chain: V2V flow messages ---
        "v2v_msg_video_required": "V2V継続に使う元動画を選択してください。",
        "v2v_msg_bad_extension": "元動画の形式が許可されていません。許可形式: {exts}",
        "v2v_msg_too_large": "元動画が上限 {limit} MB を超えています。",
        "v2v_msg_bad_context": "参照フレーム数は8n+1かつ{mincf}〜{maxcf}にしてください。",
        "v2v_msg_context_ge_clip": "参照フレーム数 ({cf}) はクリップ1のフレーム数 ({clip}) より小さくしてください（続きを生成する余地を残すため）。",
        "v2v_msg_image_conflict": "V2Vではクリップ1の開始画像は使えません（先頭は元動画が占有します）。画像を外して再試行してください。",
        "v2v_msg_clip_count": "V2Vでは有効にするクリップは1〜8個にしてください。",
        "v2v_msg_uploading": "元動画をアップロード中…",
        "v2v_msg_join_disabled": "結合版の作成がオフになっています。チェックを入れると作成できます。",
        "v2v_msg_no_job": "完了した連結ジョブがまだありません。先にV2V連結を生成してください。",
        "v2v_msg_joining": "結合版を作成中（サーバー側処理）…",
        "v2v_msg_join_done": "結合版を作成しました ({mode}): {job_id}",
        "v2v_msg_join_failed": "結合版の作成に失敗しました: {err}",
        # --- clip chain: A2V panel ---
        "a2v_lbl_audio": "元音声 (wav/mp3/m4a/aac/flac/ogg・最大50MB)",
        "a2v_guide": ("**A2Vを使いこなすには**\n\n"
                      "この機能は、アップロードした音声に口の動きを合わせて動画を生成します。"
                      "もっとも効果を発揮するのは、顔のクローズアップ・単一話者・動きが控えめな構図です。"
                      "この条件なら、口の動きが音声とほぼぴったり一致します。"
                      "反対に、大勢が行き交う賑やかなシーンや、人物が広い画角の中を動き回る構図では、口の一致は弱くなりがちです。"
                      "うまくいかないときは、まず構図をシンプルに寄せてみてください。"
                      "クリップ1に開始画像を指定してクローズアップ構図を固定するのも効果的です。\n\n"
                      "プロンプト例（クローズアップ・単一話者）:\n\n"
                      "`Cinematic trailer shot, extreme close-up of a weathered detective speaking "
                      "directly to camera in a dim office, warm lamplight raking across his face, "
                      "shallow depth of field, subtle head movement, lips articulating each word "
                      "clearly, tense and intimate mood, film grain, 35mm.`\n\n"
                      "音声は明瞭な発話を、動画の尺は5〜6秒よりやや長めにすると安定しやすくなります。"),
        "a2v_cap_panel": ("A2Vではクリップをちょうど1個使います（1本の音声がそのクリップ全体を駆動します）。"
                          "音声は動画の長さ以上必要で、短い音声は拒否されます。"
                          "出力にはアップロードした音声がそのまま入ります。"),
        # --- clip chain: A2V flow messages ---
        "a2v_msg_audio_required": "A2Vに使う元音声を選択してください。",
        "a2v_msg_bad_extension": "音声の形式が許可されていません。許可形式: {exts}",
        "a2v_msg_too_large": "音声が上限 {limit} MB を超えています。",
        "a2v_msg_clip_count": "A2Vではクリップ1のみを有効にしてください（ちょうど1個）。",
        "a2v_msg_uploading": "音声をアップロード中…",
        # --- V2V/A2V + join API error-envelope hints ---
        "apierr_SOURCE_VIDEO_NOT_FOUND": "元動画がサーバー上に見つかりません。元動画を再アップロードしてください。",
        "apierr_SOURCE_VIDEO_TOO_SHORT": "元動画のフレーム数が参照フレーム数に足りません。参照フレーム数を減らすか、長い動画を使ってください。",
        "apierr_SOURCE_AUDIO_NOT_FOUND": "元音声がサーバー上に見つかりません。音声を再アップロードしてください。",
        "apierr_SOURCE_AUDIO_TOO_SHORT": "音声が動画の長さに足りません。長い音声を使うか、フレーム数を減らしてください。",
        "apierr_JOB_NOT_JOINABLE": "このジョブはV2V継続ではないため、結合する相手がありません。",
        "apierr_JOIN_FAILED": "サーバー側で結合に失敗しました。サーバーのログを確認してください。",
        "apierr_JOINED_NOT_READY": "結合版はまだ作成されていません。先に「結合版を作成」を実行してください。",
        # --- settings: interface section ---
        "h_ui": "表示",
        "lbl_lang": "言語 (Language)",
        "lbl_theme": "テーマ",
        "opt_dark": "ダーク",
        "opt_light": "ライト",
        # --- jobs tab (S6) ---
        "btn_jobs_refresh": "一覧を更新",
        "col_job_id": "ジョブID",
        "col_status": "状態",
        "col_progress": "進捗",
        "col_created": "作成",
        "col_completed": "完了",
        "col_error": "エラー",
        "lbl_job_detail": "選択したジョブの詳細",
        "lbl_done_video": "完了した動画",
        "lbl_job_error": "エラー",
        "btn_job_action": "キャンセル / 削除",
        "msg_jobs_refresh_failed": "ジョブ一覧の取得に失敗しました: {err}",
        "msg_job_select_failed": "ジョブ詳細の取得に失敗しました: {err}",
        "msg_no_job_selected": "先に一覧からジョブを選択してください。",
        "msg_job_cancel_requested": "キャンセルを要求しました: {job_id}",
        "msg_job_deleted": "削除しました: {job_id}",
        "msg_job_action_failed": "操作に失敗しました: {err}",
        # --- settings tab (S6) ---
        "h_conn": "接続情報",
        "lbl_base_url": "base_url",
        "lbl_apikey": "APIキー",
        "badge_set": "設定済み",
        "badge_unset": "未設定",
        "h_behavior": "動作設定",
        "lbl_poll": "ポーリング間隔(秒)",
        "lbl_timeout": "タイムアウト(分)",
        "h_server": "サーバ設定(読み取り専用)",
        "sum_config": "生の /config JSON",
        "warn_config_load_failed": "サーバ設定の取得に失敗しました: {err}。自動的に再試行します…",
        "warn_config_retry_exhausted": "{n}回試行しましたがサーバ設定を取得できませんでした: {err}。更新ボタンで再試行してください。",
        "lbl_maxframes": "解像度別の快適フレーム数上限",
        "col_res": "解像度",
        "col_maxframes": "最大フレーム数",
        "cap_over": "これらを超えても生成は可能ですが、大幅に低速化します。",
        "h_danger": "危険な操作",
        "chk_danger": "これらの操作を許可する",
        "btn_unload_confirm": "モデル解放(確認付き)",
        "btn_purge": "終了済みジョブを全削除",
        "msg_purge_done": "終了済みジョブを {n} 件削除しました。",
        "msg_purge_failed": "ジョブの削除に失敗しました: {err}",
        # --- settings tab: model management (model_ prefix) ---
        "model_section_title": "モデル",
        "model_cat_transformer": "動画モデル (transformer)",
        "model_cat_text_encoder": "テキストエンコーダ (Gemma)",
        "model_cat_video_vae": "動画VAE",
        "model_cat_audio": "音声モデル (音声VAE+ボコーダ)",
        "model_btn_refresh": "モデル一覧を更新",
        "model_btn_load": "選択したモデルを読込",
        "model_missing": "ファイルなし",
        "model_loading": "モデルを読込中… 切替はエンジンの再構築を伴うため数分かかることがあります。",
        "model_load_ok": "モデルを読み込みました: {models}",
        "model_load_failed": "モデルの読込に失敗しました:\n{err}",
        "model_fetch_failed": "モデル一覧の取得に失敗しました: {err}",
        "model_hint": ("選択は「読込」ボタンで反映されます。default は標準構成です。"
                       "切替はエンジンの再起動を伴い数分かかります。読込に失敗した場合は、"
                       "すべて default を選び直して再度読込してください。"),
        "apierr_MODEL_NOT_FOUND": "不明なモデル名です。モデル一覧を更新して選び直してください。",
        "apierr_MODEL_FILE_MISSING": "モデルファイルがディスク上に見つかりません。再ダウンロードするか別のモデルを選んでください。",
        "apierr_MODEL_INCOMPATIBLE": "選択したファイルはこの用途のモデルとして不正です。別のモデルを選んでください。",
        # --- style / character LoRA tab (S2) ---
        "tab_style_lora": "画風LoRA",
        "style_gallery_label": "画風・キャラクターLoRA",
        "style_reload_btn": "LoRA一覧を再読込",
        "style_note": ("下のLoRAをクリックすると、Generateタブのプロンプト末尾に <lora:名前:1.0> が"
                       "追加されます。重みはプロンプト内の数値を書き換えて調整します"
                       "（0〜2.0・1.0=そのLoRAが学習時に想定した強さ）。"
                       "canny／pose／アップスケーラなどの制御LoRAはここには表示されません。"
                       "従来どおりGenerateタブの参照動画アダプタ欄から使ってください。"),
        "style_added": "プロンプトに追加しました: {name}",
        "style_reload_done": "LoRAを再読込しました: 合計{total}件（画風{styles}件・制御{controls}件）。",
        "style_reload_failed": "LoRAの再読込に失敗しました: {err}",
        "style_list_failed": "LoRA一覧の取得に失敗しました: {err}",
        # --- prompt-embedded <lora:...> messages (S2) ---
        "lora_msg_unknown": ("プロンプト内に未知のLoRA名があります: {names}。"
                             "<lora:...> の記述を修正するか削除して再試行してください。"),
        "lora_warn_weight_clamp": ("LoRA <{name}> の重み {given} が範囲外（0〜2.0）です。"
                                   "{clamped} に丸めました。"),
        "lora_msg_list_failed": "<lora:...> を解決するためのLoRA一覧取得に失敗しました: {err}",
    },
}


def L(key: str, lang: str = _DEFAULT_LANG) -> str:
    """Look up a user-visible string. Falls back to English, then to the key."""
    table = LABELS.get(lang, LABELS["en"])
    return table.get(key) or LABELS["en"].get(key, key)
