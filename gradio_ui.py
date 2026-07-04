"""Gradio verification UI (spec ch.12) — S1 scaffold for the 4-tab layout.

The UI is a *thin client* over the frozen REST API (/api/v1/*). It never calls
LTX directly; it exercises the very same endpoints future frontends (AviUtl2,
DaVinci Resolve) will use:

    [optional] POST /api/v1/upload/image  -> image_id
    POST /api/v1/generate                 -> job_id
    poll GET /api/v1/jobs/{job_id}         -> progress
    GET /api/v1/jobs/{job_id}/video        -> mp4

Structure (this file, S1):
  * ``ApiClient``          — the only place /api/v1/* paths + auth headers live.
                             Holds an injectable ``httpx.Client`` so tests can
                             feed it an ``httpx.MockTransport``.
  * ``LABELS`` / ``L()``   — i18n-ready label table (English default, Japanese
                             ported). Every user-visible string is looked up via
                             ``L(key)``. A component registry records each
                             ``(component, label_key, attr)`` so a later slice
                             (S6) can implement live language switching by
                             iterating it.
  * ``format_status()``    — pure formatter for the top status line.
  * ``make_generate_handler()`` — the yield-based generate flow, factored out so
                             it is unit-testable with a mock transport.
  * ``build_ui()``         — assembles the top common bar + gr.Tabs (Generate /
                             Clip Chain / Jobs / Settings). Placeholder tabs are
                             filled in by later slices (S3–S6).

Dark theme + language switching are wired at the mount site (main.py passes a
``js=`` dark-default) and, for the Theme dropdown, a pure-frontend js handler.
No HTTP is performed at build time (build_ui runs before uvicorn listens); the
initial /status + /config fetch happens in ``demo.load``.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import gradio as gr
import httpx

# --------------------------------------------------------------------------- #
# Presets (S1 fallback only — S2 replaces this with /config generation_presets).
# width/height are multiples of 64 (two-stage distilled).
# --------------------------------------------------------------------------- #
PRESETS: dict[str, dict] = {
    "smoke_test": {"width": 384, "height": 256, "num_frames": 17, "crop_w": 0, "crop_h": 0},
    "phase1_default": {"width": 512, "height": 320, "num_frames": 49, "crop_w": 0, "crop_h": 0},
    "phase1_target": {"width": 960, "height": 576, "num_frames": 121, "crop_w": 960, "crop_h": 540},
}


# --------------------------------------------------------------------------- #
# i18n label table. English is the default; Japanese entries are ported from the
# previous UI + the approved mockup. Labels are plain text (Gradio labels do not
# render HTML), so hints that were <span> in the mockup are folded into the text.
# --------------------------------------------------------------------------- #
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
        "lbl_prompt": "Prompt",
        "ph_prompt": "A bustling downtown at dusk; crowds weave through the alleys as neon signs flicker on — like a scene from a movie trailer",
        "lbl_negative": "Negative prompt",
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
        "lbl_prompt_shared": "Prompt (shared)",
        "ph_prompt2": ("A stormy harbor town; a small boat pushes through the swelling waves while "
                       "a distant bell tolls — like a scene from a movie trailer"),
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
        # --- settings: interface section ---
        "h_ui": "Interface",
        "lbl_lang": "Language",
        "lbl_theme": "Theme",
        "opt_dark": "Dark",
        "opt_light": "Light",
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
        "lbl_prompt": "プロンプト",
        "ph_prompt": "夕暮れの賑やかな下町、行き交う人々、ネオンが灯りはじめる路地。映画のワンシーンのように——",
        "lbl_negative": "ネガティブプロンプト",
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
        "lbl_prompt_shared": "プロンプト(共通)",
        "ph_prompt2": "嵐の港町、荒れる波間を進む小舟、遠くで鳴る鐘の音——映画のワンシーンのように",
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
        # --- settings: interface section ---
        "h_ui": "表示",
        "lbl_lang": "言語 (Language)",
        "lbl_theme": "テーマ",
        "opt_dark": "ダーク",
        "opt_light": "ライト",
    },
}


def L(key: str, lang: str = _DEFAULT_LANG) -> str:
    """Look up a user-visible string. Falls back to English, then to the key."""
    table = LABELS.get(lang, LABELS["en"])
    return table.get(key) or LABELS["en"].get(key, key)


# --------------------------------------------------------------------------- #
# REST client — the ONLY place /api/v1/* paths + auth headers live.
# --------------------------------------------------------------------------- #
class ApiClient:
    """Thin wrapper over the frozen REST API.

    ``client`` is injectable so tests can pass
    ``httpx.Client(transport=httpx.MockTransport(handler))``. When omitted a
    real client is created lazily (never at import/build time).
    """

    def __init__(self, base_url: str, api_key: str | None = None,
                 client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=60)
        return self._client

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    # --- read ---
    def get_status(self) -> dict:
        r = self.client.get(self._url("/api/v1/status"), headers=self.headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_config(self) -> dict:
        r = self.client.get(self._url("/api/v1/config"), headers=self.headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_job(self, job_id: str) -> dict:
        r = self.client.get(self._url(f"/api/v1/jobs/{job_id}"), headers=self.headers, timeout=10)
        r.raise_for_status()
        return r.json()

    # --- lifecycle ---
    def load_pipeline(self) -> dict:
        # Model load can be slow; give it a generous timeout.
        r = self.client.post(self._url("/api/v1/pipeline/load"), headers=self.headers, timeout=600)
        r.raise_for_status()
        return r.json()

    def unload_pipeline(self) -> dict:
        r = self.client.post(self._url("/api/v1/pipeline/unload"), headers=self.headers, timeout=60)
        r.raise_for_status()
        return r.json()

    # --- generate flow ---
    def upload_image(self, path: str) -> str:
        with open(path, "rb") as fh:
            files = {"file": (Path(path).name, fh.read())}
        r = self.client.post(self._url("/api/v1/upload/image"), files=files,
                             headers=self.headers, timeout=60)
        r.raise_for_status()
        return r.json()["image_id"]

    def upload_video(self, path: str) -> str:
        # Reference-video upload (POST /upload/video) for the IC-LoRA control
        # adapters. Videos can be large (up to 200MB), so use a longer timeout
        # than image uploads. Returns the server's ``video_id`` (passed to
        # /generate as ``reference_video_id``).
        with open(path, "rb") as fh:
            files = {"file": (Path(path).name, fh.read())}
        r = self.client.post(self._url("/api/v1/upload/video"), files=files,
                             headers=self.headers, timeout=300)
        r.raise_for_status()
        return r.json()["video_id"]

    def generate(self, payload: dict) -> httpx.Response:
        # Return the raw response so the caller can branch on 409 / >=400 while
        # keeping the client thin.
        return self.client.post(self._url("/api/v1/generate"), json=payload,
                                headers=self.headers, timeout=60)

    def generate_chain(self, payload: dict) -> httpx.Response:
        # Clip-chain start (POST /generate/chain). Same thin style as
        # ``generate``: return the raw response so the caller branches on
        # 409 / >=400. The endpoint responds 202 with the job envelope.
        return self.client.post(self._url("/api/v1/generate/chain"), json=payload,
                                headers=self.headers, timeout=60)

    def fetch_video(self, job_id: str) -> str:
        """Download the finished mp4 to a temp file and return its path."""
        r = self.client.get(self._url(f"/api/v1/jobs/{job_id}/video"),
                            headers=self.headers, timeout=60)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(prefix=f"{job_id}_", suffix=".mp4", delete=False)
        tmp.write(r.content)
        tmp.close()
        return tmp.name


# --------------------------------------------------------------------------- #
# Preset handling (S2). The hardcoded PRESETS dict above is now only a fallback
# for when the server /config fetch (demo.load) failed or returned no presets;
# the normal path reads config["generation_presets"] (GET /config, which
# model_dumps config.py's GenerationPreset -- width/height/num_frames/
# crop_output, the last being {"width", "height"} or None).
# --------------------------------------------------------------------------- #
def build_preset_choices(config: dict | None) -> list[tuple[str, str]]:
    """Build Dropdown ``choices`` from the fetched /config generation_presets.

    Falls back to the hardcoded ``PRESETS`` keys (label == value, matching S1
    behaviour) when the server config is empty/unavailable.
    """
    presets = (config or {}).get("generation_presets") or {}
    if not presets:
        return [(name, name) for name in PRESETS]

    choices: list[tuple[str, str]] = []
    for key, p in presets.items():
        w, h, nf = p.get("width"), p.get("height"), p.get("num_frames")
        crop = p.get("crop_output")
        if crop:
            label = f"{key} ({w}×{h} → {crop.get('width')}×{crop.get('height')}, {nf}f)"
        else:
            label = f"{key} ({w}×{h}, {nf}f)"
        choices.append((label, key))
    return choices


def pick_default_preset(config: dict | None) -> str:
    """Pick the Dropdown's initial value: ``standard_720p`` if present, else the
    first server preset, else the S1 fallback default."""
    presets = (config or {}).get("generation_presets") or {}
    if "standard_720p" in presets:
        return "standard_720p"
    if presets:
        return next(iter(presets))
    return "phase1_default"


def compute_spill_warning(width, height, num_frames, config: dict | None,
                           lang: str = _DEFAULT_LANG):
    """Look up limits.spill_free_frames[f"{width}x{height}"] from the fetched
    /config. Returns a gr.update for a Markdown warning: visible+worded when
    num_frames exceeds the comfortable (spill-free) threshold for that
    resolution, hidden when the resolution is unknown or within budget."""
    try:
        w, h, nf = int(width), int(height), int(num_frames)
    except (TypeError, ValueError):
        return gr.update(value="", visible=False)

    spill = ((config or {}).get("limits") or {}).get("spill_free_frames") or {}
    key = f"{w}x{h}"
    threshold = spill.get(key)
    if threshold is not None and nf > threshold:
        text = L("warn_spill_limit", lang).format(res=key, limit=threshold)
        return gr.update(value=text, visible=True)
    return gr.update(value="", visible=False)


def apply_preset(name: str, config: dict | None, lang: str = _DEFAULT_LANG):
    """Resolve a preset name to the Generate-tab field values.

    Reads the server preset (config["generation_presets"][name]) when present;
    falls back to the hardcoded PRESETS dict only when the server config is
    unavailable or does not contain ``name``. Returns a tuple matching the
    ``preset.change`` outputs: (width, height, num_frames, crop_enabled,
    crop_w, crop_h, crop_row_update, spill_warning_update).
    """
    presets = (config or {}).get("generation_presets") or {}
    if name in presets:
        p = presets[name]
        width_v, height_v, frames_v = p["width"], p["height"], p["num_frames"]
        crop = p.get("crop_output")
        crop_w_v = crop["width"] if crop else 0
        crop_h_v = crop["height"] if crop else 0
    else:
        p = PRESETS.get(name) or PRESETS["phase1_default"]
        width_v, height_v, frames_v = p["width"], p["height"], p["num_frames"]
        crop_w_v, crop_h_v = p.get("crop_w", 0), p.get("crop_h", 0)

    crop_enabled_v = bool(crop_w_v) and bool(crop_h_v)
    crop_row_update = gr.update(visible=crop_enabled_v)
    spill_update = compute_spill_warning(width_v, height_v, frames_v, config, lang)
    return (width_v, height_v, frames_v, crop_enabled_v, crop_w_v, crop_h_v,
            crop_row_update, spill_update)


# --------------------------------------------------------------------------- #
# Pure formatter for the top status line (unit-testable with a fake /status).
# Field names verified against api/status.py + services/{gpu_info,low_vram,
# job_store}.py: gpu.{name,vram_free_mb,vram_total_mb}, vram_optimization.
# {low_vram_mode,low_vram_profile}, queue.{running,pending,completed}.
# --------------------------------------------------------------------------- #
def format_status(s: dict, lang: str = _DEFAULT_LANG) -> str:
    gpu = s.get("gpu") or {}
    v = s.get("vram_optimization") or {}
    q = s.get("queue") or {}

    loaded = s.get("pipeline_loaded")
    ptype = s.get("pipeline_type")
    pipe = L("st_loaded", lang) if loaded else L("st_not_loaded", lang)
    if loaded and ptype:
        pipe = f"{pipe} ({ptype})"

    low = L("st_on", lang) if v.get("low_vram_mode") else L("st_off", lang)

    parts = [
        f"server={s.get('server')} v{s.get('version')}",
        f"{L('st_pipeline', lang)}: {pipe}",
        (f"GPU: {gpu.get('name')} "
         f"(VRAM {L('st_free', lang)} {gpu.get('vram_free_mb')} / {gpu.get('vram_total_mb')} MB)"),
        f"{L('st_lowvram', lang)}: {low} (profile={v.get('low_vram_profile')})",
    ]
    if q:
        parts.append(
            f"{L('st_queue', lang)}: "
            f"{L('st_running', lang)} {q.get('running', 0)} / "
            f"{L('st_waiting', lang)} {q.get('pending', 0)} / "
            f"{L('st_done', lang)} {q.get('completed', 0)}"
        )
    return " | ".join(parts)


# --------------------------------------------------------------------------- #
# IC-LoRA reference-video control (S4). The Generate tab exposes an adapter
# Dropdown whose choices are rebuilt on page load from /config model.ic_loras;
# the static list below is the offline fallback (server /config unavailable).
# ``ADAPTER_NONE`` is the sentinel value meaning "no adapter" (payload omits
# loras + reference_video_id entirely). The three known adapter keys get a
# friendly label; any unknown registered key is shown as-is.
# --------------------------------------------------------------------------- #
ADAPTER_NONE = "__none__"

ADAPTER_FRIENDLY: dict[str, str] = {
    "pixel-spatial-upscaler-x2": "Upscale ×2 (pixel-spatial-upscaler-x2)",
    "canny-control": "Canny edge control (canny-control)",
    "pose-control": "Pose control (pose-control)",
}

# Fallbacks used when /config is unavailable (mirrors config.yaml upload.*).
_FALLBACK_VIDEO_EXTS = [".mp4", ".mov", ".webm", ".mkv"]
_FALLBACK_MAX_VIDEO_MB = 200


def build_adapter_choices(config: dict | None, lang: str = _DEFAULT_LANG) -> list[tuple[str, str]]:
    """Build the adapter Dropdown ``choices`` (list of (label, value)).

    "None" is always first (value :data:`ADAPTER_NONE`). The remaining entries
    come from the fetched /config ``model.ic_loras`` keys (value == key); each
    key gets a friendly label when known, else is shown verbatim. Falls back to
    the three static known adapters when the server config has no ic_loras.
    """
    none_choice = (L("adapter_none", lang), ADAPTER_NONE)
    ic_loras = ((config or {}).get("model") or {}).get("ic_loras") or {}
    if not ic_loras:
        return [none_choice] + [
            (ADAPTER_FRIENDLY[k], k)
            for k in ("pixel-spatial-upscaler-x2", "canny-control", "pose-control")
        ]
    return [none_choice] + [(ADAPTER_FRIENDLY.get(key, key), key) for key in ic_loras]


def format_api_error(body: object, lang: str = _DEFAULT_LANG) -> str:
    """Render a REST error envelope into a localized, actionable message.

    ``body`` is the parsed JSON dict (``{"error": {"code", "message", "detail"}}``)
    or, when the response was not JSON, the raw text. Maps ``error.code`` to a
    one-line hint (all 15 real codes). For ``VALIDATION_ERROR`` the ``detail`` is
    a list of ``{loc, msg, type}`` rendered as ``loc: msg`` lines; for other
    codes ``detail`` is a string appended when present. An unknown code or an
    unparseable body falls back to the raw text.
    """
    if not isinstance(body, dict):
        return str(body)
    error = body.get("error")
    if not isinstance(error, dict):
        return json.dumps(body, ensure_ascii=False)
    code = error.get("code")
    hint_key = f"apierr_{code}" if code else None
    if not hint_key or hint_key not in LABELS["en"]:
        # Unknown / unmapped code -> raw text (still useful for debugging).
        return json.dumps(body, ensure_ascii=False)

    lines = [L(hint_key, lang)]
    detail = error.get("detail")
    if code == "VALIDATION_ERROR" and isinstance(detail, list):
        for item in detail:
            loc = ".".join(str(x) for x in (item.get("loc") or []))
            msg = item.get("msg", "")
            lines.append(f"{loc}: {msg}" if loc else str(msg))
    elif isinstance(detail, str) and detail:
        lines.append(detail)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Shared 1s poll loop (factored out of the generate flow so /generate and
# /generate/chain reuse the SAME progress/complete/fail handling). Yields
# (progress_text, job_id, video_path) tuples; behaviour is byte-identical to the
# original inline loop in make_generate_handler.
# --------------------------------------------------------------------------- #
def _poll_job_until_done(api: ApiClient, job_id: str, lang: str = _DEFAULT_LANG):
    for _ in range(3600):
        time.sleep(1.0)
        try:
            job = api.get_job(job_id)
        except Exception as exc:
            yield L("msg_poll_failed", lang).format(err=exc), job_id, None
            continue

        status = job["status"]
        progress = job.get("progress", 0.0)
        if status == "running":
            step = job.get("current_step")
            total = job.get("total_steps")
            yield L("msg_generating", lang).format(pct=progress, step=step, total=total), job_id, None
        elif status == "completed":
            yield L("msg_completing", lang), job_id, None
            try:
                video = api.fetch_video(job_id)
            except Exception:
                video = None
            yield L("msg_completed", lang).format(job_id=job_id), job_id, video
            return
        elif status in ("failed", "cancelled"):
            yield L("msg_failed", lang).format(status=status, error=job.get("error")), job_id, None
            return

    yield L("msg_timeout", lang), job_id, None


# --------------------------------------------------------------------------- #
# Clip-chain total-timeline precheck (S5). Mirrors the SAME arithmetic the API
# validator uses (api/models.py GenerateChainRequest.validate_chain_constraints):
# it delegates to the shared pure-Python ``chain_math.compute_chain_layout`` and
# compares against MAX_CHAIN_TOTAL_PIXEL_FRAMES = 8 * 481 (= 3848), so the GUI and
# the server agree byte-for-byte. Returns a localized error string on violation,
# else None. Note: because a single clip is capped at 481 frames and a chain at 8
# clips, the 3848 cap is unreachable through the 8-slot UI (max 8×481 = 3841 total
# pixel frames after overlap); the check is defence-in-depth that mirrors the
# server and also surfaces chain_math's degenerate-geometry ValueError (e.g. clips
# too short for a continuous audio cross-fade) before any API call.
# --------------------------------------------------------------------------- #
MAX_CHAIN_TOTAL_PIXEL_FRAMES = 8 * 481  # 3848; mirrors api/models.py


def check_chain_total(clip_frames, fps, overlap_frames, lang: str = _DEFAULT_LANG):
    import chain_math
    try:
        layout = chain_math.compute_chain_layout(
            [int(f) for f in clip_frames], float(fps), kv=int(overlap_frames),
        )
    except ValueError as exc:
        return L("msg_chain_geometry", lang).format(err=exc)
    if layout.total_px > MAX_CHAIN_TOTAL_PIXEL_FRAMES:
        return L("msg_chain_total_frames", lang).format(
            total=layout.total_px, cap=MAX_CHAIN_TOTAL_PIXEL_FRAMES,
        )
    return None


# --------------------------------------------------------------------------- #
# Generate flow, factored out for unit testing (mock transport). Yields
# (progress_text, job_id, video_path) tuples, matching the previous behaviour.
# --------------------------------------------------------------------------- #
def make_generate_handler(api: ApiClient, lang: str = _DEFAULT_LANG):
    def generate(prompt, negative_prompt,
                 kf1_enabled, kf1_image, kf1_frame_idx, kf1_strength,
                 kf2_enabled, kf2_image, kf2_frame_idx, kf2_strength,
                 kf3_enabled, kf3_image, kf3_frame_idx, kf3_strength,
                 kf4_enabled, kf4_image, kf4_frame_idx, kf4_strength,
                 kf5_enabled, kf5_image, kf5_frame_idx, kf5_strength,
                 width, height, crop_enabled, crop_w, crop_h, num_frames, frame_rate, seed,
                 adapter=ADAPTER_NONE, adapter_strength=1.0, ref_video_path=None, config=None):
        if not prompt or not prompt.strip():
            yield L("msg_prompt_required", lang), "", None
            return

        # 1) keyframe slots (up to 5, I2V multi-keyframe conditioning). Each
        # FIXED slot is (enabled, image_path, frame_idx, strength); disabled or
        # empty slots are skipped. Pre-validate ALL enabled slots before any
        # upload starts, so a bad slot never leaves earlier slots uploaded.
        slots = [
            (kf1_enabled, kf1_image, kf1_frame_idx, kf1_strength),
            (kf2_enabled, kf2_image, kf2_frame_idx, kf2_strength),
            (kf3_enabled, kf3_image, kf3_frame_idx, kf3_strength),
            (kf4_enabled, kf4_image, kf4_frame_idx, kf4_strength),
            (kf5_enabled, kf5_image, kf5_frame_idx, kf5_strength),
        ]
        to_upload: list[tuple[str, int, float]] = []
        for slot_n, (enabled, image_path, frame_idx, strength) in enumerate(slots, start=1):
            if not enabled:
                continue
            if frame_idx is None or int(frame_idx) < 0:
                yield L("msg_kf_negative_frame", lang).format(n=slot_n), "", None
                return
            if not image_path:
                yield L("msg_kf_missing_image", lang).format(n=slot_n), "", None
                return
            to_upload.append((image_path, int(frame_idx), float(strength)))

        # 1b) IC-LoRA reference-video control (S4). Validate the adapter's
        # reference video BEFORE any upload happens, so a violation costs zero
        # API calls. Prechecks in order: (a) video present, (b) extension
        # allowed, (c) size within limit, (d) width/height divisible by 128
        # (the ÷128 rule the server also enforces with a 422). Keyframes (I2V)
        # and an adapter can combine — the API allows both.
        use_adapter = bool(adapter) and adapter != ADAPTER_NONE
        if use_adapter:
            upload_cfg = (config or {}).get("upload") or {}
            allowed_exts = upload_cfg.get("allowed_video_extensions") or _FALLBACK_VIDEO_EXTS
            max_mb = upload_cfg.get("max_video_size_mb")
            if max_mb is None:
                max_mb = _FALLBACK_MAX_VIDEO_MB

            if not ref_video_path:
                yield L("msg_ref_video_required", lang), "", None
                return
            ext = Path(ref_video_path).suffix.lower()
            if ext not in [e.lower() for e in allowed_exts]:
                yield L("msg_ref_bad_extension", lang).format(exts=", ".join(allowed_exts)), "", None
                return
            try:
                size_mb = os.path.getsize(ref_video_path) / (1024 * 1024)
            except OSError:
                size_mb = 0.0
            if size_mb > max_mb:
                yield L("msg_ref_too_large", lang).format(limit=max_mb), "", None
                return
            if int(width) % 128 != 0 or int(height) % 128 != 0:
                yield L("msg_ref_resolution", lang), "", None
                return

        conditioning: list[dict] = []
        total = len(to_upload)
        for i, (image_path, frame_idx, strength) in enumerate(to_upload, start=1):
            yield L("msg_uploading_keyframe", lang).format(i=i, n=total), "", None
            try:
                image_id = api.upload_image(image_path)
            except Exception as exc:
                yield L("msg_upload_failed", lang).format(err=exc), "", None
                return
            conditioning.append({"image_id": image_id, "frame_idx": frame_idx, "strength": strength})

        # Upload the reference video last (after keyframes), then attach the
        # lora + reference_video_id to the payload below.
        reference_video_id: str | None = None
        if use_adapter:
            yield L("msg_uploading_ref", lang), "", None
            try:
                reference_video_id = api.upload_video(ref_video_path)
            except Exception as exc:
                yield L("msg_upload_failed", lang).format(err=exc), "", None
                return

        # 2) start generation. Quality is locked to distilled in S1 (steps/cfg
        # fixed); the payload keeps the frozen contract.
        crop_output = None
        if crop_enabled and int(crop_w) > 0 and int(crop_h) > 0:
            crop_output = {"width": int(crop_w), "height": int(crop_h)}
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "width": int(width),
            "height": int(height),
            "crop_output": crop_output,
            "num_frames": int(num_frames),
            "frame_rate": float(frame_rate),
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "seed": int(seed),
            "pipeline": "distilled",
            "conditioning_images": conditioning,
        }
        # IC-LoRA (S4): only add loras + reference_video_id when an adapter is
        # selected. Omitting both keys keeps the request byte-identical to the
        # no-adapter path (matches GenerateRequest defaults).
        if use_adapter:
            payload["loras"] = [{"name": adapter, "strength": float(adapter_strength)}]
            payload["reference_video_id"] = reference_video_id
        try:
            resp = api.generate(payload)
        except Exception as exc:
            yield L("msg_generate_failed", lang).format(err=exc), "", None
            return
        if resp.status_code == 409:
            yield L("msg_job_busy", lang), "", None
            return
        if resp.status_code >= 400:
            try:
                err_body: object = resp.json()
            except Exception:
                err_body = resp.text
            yield format_api_error(err_body, lang), "", None
            return
        job_id = resp.json()["job_id"]

        mode = "i2v" if conditioning else "t2v"
        yield L("msg_job_started", lang).format(mode=mode, job_id=job_id), job_id, None

        # 3) poll (1s) — shared with the clip-chain flow.
        yield from _poll_job_until_done(api, job_id, lang)

    return generate


# --------------------------------------------------------------------------- #
# Clip-chain flow (S5), factored out for unit testing (mock transport). Mirrors
# make_generate_handler: prechecks (zero API calls on violation) -> optional
# clip-0 start-image upload -> POST /generate/chain -> shared poll loop. The 8
# FIXED clip slots are flattened into positional args; only slot 1 carries a
# start image + strength (structural guarantee that conditioning lives on clip 0
# only). ``clips`` are emitted in slot order, enabled slots only.
# --------------------------------------------------------------------------- #
def make_chain_handler(api: ApiClient, lang: str = _DEFAULT_LANG):
    def generate_chain(prompt, negative_prompt, width, height,
                       crop_enabled, crop_w, crop_h, frame_rate, seed,
                       overlap_frames, overlap_strength,
                       c1_enabled, c1_prompt, c1_frames, c1_image, c1_strength,
                       c2_enabled, c2_prompt, c2_frames,
                       c3_enabled, c3_prompt, c3_frames,
                       c4_enabled, c4_prompt, c4_frames,
                       c5_enabled, c5_prompt, c5_frames,
                       c6_enabled, c6_prompt, c6_frames,
                       c7_enabled, c7_prompt, c7_frames,
                       c8_enabled, c8_prompt, c8_frames,
                       config=None):
        # --- prechecks (localized; NO API call on any violation) ---
        if not prompt or not prompt.strip():
            yield L("msg_prompt_required", lang), "", None
            return

        try:
            w, h = int(width), int(height)
        except (TypeError, ValueError):
            yield L("msg_bad_dimension", lang), "", None
            return
        if w % 64 != 0 or h % 64 != 0:
            yield L("msg_bad_dimension", lang), "", None
            return
        limits = (config or {}).get("limits") or {}
        max_w = limits.get("max_width", 1920)
        max_h = limits.get("max_height", 1088)
        if w > max_w or h > max_h:
            yield L("msg_size_limit", lang).format(maxw=max_w, maxh=max_h), "", None
            return

        if crop_enabled:
            try:
                cw, ch = int(crop_w), int(crop_h)
            except (TypeError, ValueError):
                cw = ch = 0
            if cw < 32 or ch < 32 or cw > w or ch > h:
                yield L("msg_crop_range", lang), "", None
                return

        try:
            fps = float(frame_rate)
        except (TypeError, ValueError):
            yield L("msg_fps_range", lang), "", None
            return
        if fps < 1.0 or fps > 60.0:
            yield L("msg_fps_range", lang), "", None
            return

        # Collect enabled clips in slot order. Slot 1 additionally carries the
        # start image + strength (image conditions clip 0 only).
        raw_slots = [
            (c1_enabled, c1_prompt, c1_frames, c1_image, c1_strength),
            (c2_enabled, c2_prompt, c2_frames, None, None),
            (c3_enabled, c3_prompt, c3_frames, None, None),
            (c4_enabled, c4_prompt, c4_frames, None, None),
            (c5_enabled, c5_prompt, c5_frames, None, None),
            (c6_enabled, c6_prompt, c6_frames, None, None),
            (c7_enabled, c7_prompt, c7_frames, None, None),
            (c8_enabled, c8_prompt, c8_frames, None, None),
        ]
        enabled = [(p, nf, img, strg) for en, p, nf, img, strg in raw_slots if en]

        if not (2 <= len(enabled) <= 8):
            yield L("msg_chain_clip_count", lang), "", None
            return

        # Per-clip num_frames: 8n+1 within [9, 481].
        clip_frames: list[int] = []
        for n, (_p, nf, _img, _strg) in enumerate(enabled, start=1):
            try:
                nf_i = int(nf)
            except (TypeError, ValueError):
                yield L("msg_chain_bad_frames", lang).format(n=n), "", None
                return
            if nf_i < 9 or nf_i > 481 or (nf_i - 1) % 8 != 0:
                yield L("msg_chain_bad_frames", lang).format(n=n), "", None
                return
            clip_frames.append(nf_i)

        # overlap_frames (K_v LATENT) must be < every clip's stage-1 latent-frame
        # count = (num_frames - 1)//8 + 1 (mirrors api/models.py).
        try:
            kv = int(overlap_frames)
        except (TypeError, ValueError):
            kv = 3
        max_kv = min((nf - 1) // 8 + 1 for nf in clip_frames) - 1
        if kv < 1 or kv > max_kv:
            yield L("msg_chain_overlap_too_large", lang).format(kv=kv, maxkv=max_kv), "", None
            return

        # Total-timeline geometry: same arithmetic as the API validator.
        err = check_chain_total(clip_frames, fps, kv, lang)
        if err is not None:
            yield err, "", None
            return

        # --- clip-0 start image (only slot 1 can carry one) -> upload ---
        conditioning: list[dict] = []
        clip0_image = enabled[0][2]
        clip0_strength = enabled[0][3]
        if clip0_image:
            yield L("msg_uploading_keyframe", lang).format(i=1, n=1), "", None
            try:
                image_id = api.upload_image(clip0_image)
            except Exception as exc:
                yield L("msg_upload_failed", lang).format(err=exc), "", None
                return
            conditioning.append({
                "image_id": image_id, "frame_idx": 0,
                "strength": float(clip0_strength if clip0_strength is not None else 0.8),
            })

        # --- build payload (clips in slot order; per-clip prompt omitted when
        # blank; conditioning attached to clip 0 only) ---
        crop_output = None
        if crop_enabled and int(crop_w) > 0 and int(crop_h) > 0:
            crop_output = {"width": int(crop_w), "height": int(crop_h)}

        clips_payload: list[dict] = []
        for idx, (p, nf, _img, _strg) in enumerate(enabled):
            entry: dict = {"num_frames": int(nf)}
            if p and str(p).strip():
                entry["prompt"] = p
            if idx == 0 and conditioning:
                entry["conditioning_images"] = conditioning
            clips_payload.append(entry)

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "width": w,
            "height": h,
            "crop_output": crop_output,
            "frame_rate": fps,
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "seed": int(seed),
            "pipeline": "distilled",
            "overlap_frames": kv,
            "overlap_strength": float(overlap_strength),
            "clips": clips_payload,
        }

        try:
            resp = api.generate_chain(payload)
        except Exception as exc:
            yield L("msg_generate_failed", lang).format(err=exc), "", None
            return
        if resp.status_code == 409:
            yield L("msg_job_busy", lang), "", None
            return
        if resp.status_code >= 400:
            try:
                err_body: object = resp.json()
            except Exception:
                err_body = resp.text
            yield format_api_error(err_body, lang), "", None
            return
        job_id = resp.json()["job_id"]

        yield L("msg_chain_started", lang).format(n=len(clips_payload), job_id=job_id), job_id, None

        # poll (1s) — shared with the generate flow.
        yield from _poll_job_until_done(api, job_id, lang)

    return generate_chain


# --------------------------------------------------------------------------- #
# UI assembly.
# --------------------------------------------------------------------------- #
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
