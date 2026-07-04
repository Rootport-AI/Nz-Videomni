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
        "lbl_image": "Input image (optional, minimal I2V — frame 0)",
        "lbl_strength": "Image strength",
        "lbl_qmode": "Quality mode",
        "qmode_fast": "Fast (distilled) — 8 steps / CFG 1.0",
        "qmode_hq": "High quality (two_stage_hq) — backend support pending",
        "warn_hq_unsupported": "High-quality mode is not yet supported by the backend.",
        "lbl_preset": "Preset",
        "hint_preset": "Fetched automatically from the server /config (S2)",
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
        "lbl_image": "入力画像 (任意・最小I2V — frame 0)",
        "lbl_strength": "画像適用強度",
        "lbl_qmode": "品質モード",
        "qmode_fast": "高速 (distilled) — 8ステップ / CFG 1.0",
        "qmode_hq": "高品質 (two_stage_hq) — バックエンド未対応",
        "warn_hq_unsupported": "高品質モードはまだバックエンドが対応していません。",
        "lbl_preset": "プリセット",
        "hint_preset": "サーバの /config から自動取得 (S2)",
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

    def generate(self, payload: dict) -> httpx.Response:
        # Return the raw response so the caller can branch on 409 / >=400 while
        # keeping the client thin.
        return self.client.post(self._url("/api/v1/generate"), json=payload,
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
# Generate flow, factored out for unit testing (mock transport). Yields
# (progress_text, job_id, video_path) tuples, matching the previous behaviour.
# --------------------------------------------------------------------------- #
def make_generate_handler(api: ApiClient, lang: str = _DEFAULT_LANG):
    def generate(prompt, negative_prompt, image_path, strength, width, height,
                 crop_enabled, crop_w, crop_h, num_frames, frame_rate, seed):
        if not prompt or not prompt.strip():
            yield L("msg_prompt_required", lang), "", None
            return

        # 1) optional single-image upload (minimal I2V, frame 0).
        conditioning: list[dict] = []
        if image_path:
            try:
                image_id = api.upload_image(image_path)
            except Exception as exc:
                yield L("msg_upload_failed", lang).format(err=exc), "", None
                return
            conditioning = [{"image_id": image_id, "frame_idx": 0, "strength": float(strength)}]
            yield L("msg_upload_done", lang).format(image_id=image_id), "", None

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
        try:
            resp = api.generate(payload)
        except Exception as exc:
            yield L("msg_generate_failed", lang).format(err=exc), "", None
            return
        if resp.status_code == 409:
            yield L("msg_job_busy", lang), "", None
            return
        if resp.status_code >= 400:
            yield L("msg_generate_error", lang).format(code=resp.status_code, text=resp.text), "", None
            return
        job_id = resp.json()["job_id"]

        mode = "i2v" if conditioning else "t2v"
        yield L("msg_job_started", lang).format(mode=mode, job_id=job_id), job_id, None

        # 3) poll (1s).
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

    return generate


# --------------------------------------------------------------------------- #
# UI assembly.
# --------------------------------------------------------------------------- #
def build_ui(base_url: str, api_key: str | None = None) -> gr.Blocks:
    api = ApiClient(base_url, api_key=api_key)
    generate = make_generate_handler(api)

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
        return status, cfg

    def apply_preset(name: str):
        p = PRESETS[name]
        return p["width"], p["height"], p["num_frames"], p["crop_w"], p["crop_h"]

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

                        image = reg(gr.Image(label=L("lbl_image"), type="filepath"), "lbl_image")
                        strength = reg(gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                                 label=L("lbl_strength")), "lbl_strength")

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

                        with gr.Row():
                            steps = reg(gr.Slider(1, 50, value=8, step=1, label=L("lbl_steps"),
                                                  interactive=False), "lbl_steps")
                            cfg = reg(gr.Slider(1.0, 12.0, value=1.0, step=0.1, label=L("lbl_cfg"),
                                                interactive=False), "lbl_cfg")
                        cap_lock = reg(gr.Markdown(L("cap_lock")), "cap_lock", "value")

                        seed = reg(gr.Number(value=-1, label=L("lbl_seed"), precision=0), "lbl_seed")

                    # RIGHT: action panel (Generate first) -> progress -> job id -> video
                    with gr.Column(scale=2):
                        generate_btn = reg(gr.Button(L("btn_generate"), variant="primary"),
                                           "btn_generate", "value")
                        progress_box = reg(gr.Textbox(label=L("lbl_progress"), interactive=False),
                                           "lbl_progress")
                        job_box = reg(gr.Textbox(label=L("lbl_jobid"), interactive=False), "lbl_jobid")
                        video_out = reg(gr.Video(label=L("lbl_result")), "lbl_result")

            # =========================== Clip Chain ==========================
            with gr.Tab(L("tab_concat")) as tab_concat:
                reg(tab_concat, "tab_concat", "label")
                reg(gr.Markdown(L("msg_coming")), "msg_coming", "value")

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

        preset.change(apply_preset, inputs=preset,
                      outputs=[width, height, num_frames, crop_w, crop_h])
        qmode.change(on_qmode_change, inputs=qmode, outputs=qmode)
        crop_enabled.change(on_crop_toggle, inputs=crop_enabled, outputs=crop_row)

        # Theme: pure-frontend toggle (no backend round-trip). Matches the mount
        # site's dark-default js.
        theme_dd.change(
            None, inputs=theme_dd, outputs=None,
            js="(v) => { document.body.classList.toggle('light', v === 'light'); "
               "document.body.classList.toggle('dark', v === 'dark'); }",
        )

        generate_btn.click(
            generate,
            inputs=[prompt, negative, image, strength, width, height,
                    crop_enabled, crop_w, crop_h, num_frames, frame_rate, seed],
            outputs=[progress_box, job_box, video_out],
        )

        demo.load(on_page_load, outputs=[status_box, config_state])

    # Expose the registry for the S6 language-switch handler (and tests).
    demo.label_registry = registry  # type: ignore[attr-defined]
    return demo
