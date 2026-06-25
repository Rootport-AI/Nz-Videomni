"""Gradio test UI (spec ch.12).

The UI never calls LTX directly. It exercises the very same REST API that future
frontends (AviUtl2, DaVinci Resolve) will use:

    [optional] POST /api/v1/upload/image  -> image_id
    POST /api/v1/generate                 -> job_id
    poll GET /api/v1/jobs/{job_id}         -> progress
    GET /api/v1/jobs/{job_id}/video        -> mp4
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import gradio as gr
import httpx

# Presets (spec 11 generation_presets)
PRESETS: dict[str, dict] = {
    "smoke_test": {"width": 384, "height": 224, "num_frames": 17, "crop_w": 0, "crop_h": 0},
    "phase1_default": {"width": 512, "height": 288, "num_frames": 49, "crop_w": 0, "crop_h": 0},
    "phase1_target": {"width": 960, "height": 544, "num_frames": 121, "crop_w": 960, "crop_h": 540},
}


def build_ui(base_url: str, api_key: str | None = None) -> gr.Blocks:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def apply_preset(name: str):
        p = PRESETS[name]
        return p["width"], p["height"], p["num_frames"], p["crop_w"], p["crop_h"]

    def refresh_status() -> str:
        try:
            r = httpx.get(f"{base_url}/api/v1/status", headers=headers, timeout=10)
            r.raise_for_status()
            s = r.json()
            gpu = s.get("gpu", {})
            v = s.get("vram_optimization", {})
            return (
                f"server={s.get('server')} v{s.get('version')} | "
                f"pipeline_loaded={s.get('pipeline_loaded')} | "
                f"gpu={gpu.get('name')} ({gpu.get('vram_free_mb')}MB free) | "
                f"low_vram_mode={v.get('low_vram_mode')} profile={v.get('low_vram_profile')}"
            )
        except Exception as exc:
            return f"status error: {exc}"

    def generate(prompt, negative_prompt, image_path, strength, width, height,
                 crop_w, crop_h, num_frames, frame_rate, seed):
        if not prompt or not prompt.strip():
            yield "プロンプトを入力してください", "", None
            return

        # 1) optional image upload (minimal I2V)
        conditioning = []
        if image_path:
            try:
                with open(image_path, "rb") as fh:
                    files = {"file": (Path(image_path).name, fh.read())}
                up = httpx.post(f"{base_url}/api/v1/upload/image", files=files, headers=headers, timeout=60)
                up.raise_for_status()
                image_id = up.json()["image_id"]
                conditioning = [{"image_id": image_id, "frame_idx": 0, "strength": float(strength)}]
                yield f"画像アップロード完了: {image_id}", "", None
            except Exception as exc:
                yield f"アップロード失敗: {exc}", "", None
                return

        # 2) start generation
        crop_output = None
        if int(crop_w) > 0 and int(crop_h) > 0:
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
            gen = httpx.post(f"{base_url}/api/v1/generate", json=payload, headers=headers, timeout=60)
            if gen.status_code == 409:
                yield "別のジョブが実行中です (409)", "", None
                return
            if gen.status_code >= 400:
                yield f"generate エラー {gen.status_code}: {gen.text}", "", None
                return
            job_id = gen.json()["job_id"]
        except Exception as exc:
            yield f"generate 失敗: {exc}", "", None
            return

        mode = "i2v" if conditioning else "t2v"
        yield f"ジョブ開始 ({mode}): {job_id}", job_id, None

        # 3) poll
        for _ in range(3600):
            time.sleep(1.0)
            try:
                jr = httpx.get(f"{base_url}/api/v1/jobs/{job_id}", headers=headers, timeout=10)
                jr.raise_for_status()
                job = jr.json()
            except Exception as exc:
                yield f"ポーリング失敗: {exc}", job_id, None
                continue

            status = job["status"]
            progress = job.get("progress", 0.0)
            if status == "running":
                step = job.get("current_step")
                total = job.get("total_steps")
                yield f"生成中… {progress:.0%} (step {step}/{total})", job_id, None
            elif status == "completed":
                yield "完了。動画を取得中…", job_id, None
                video = _fetch_video(base_url, job_id, headers)
                yield f"完了: {job_id}", job_id, video
                return
            elif status in ("failed", "cancelled"):
                yield f"{status}: {job.get('error')}", job_id, None
                return

        yield "タイムアウト", job_id, None

    with gr.Blocks(title="LTX-AviUtl2-Bridge") as demo:
        gr.Markdown("# LTX-AviUtl2-Bridge — Phase 1 テストUI")
        gr.Markdown("画像なし → T2V / 画像1枚 → 最小I2V（frame_idx=0固定）。Distilled: 8 steps / CFG=1.0。")

        status_box = gr.Textbox(label="server status", interactive=False, value="")
        status_btn = gr.Button("ステータス更新 (/api/v1/status)")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="prompt", lines=3, placeholder="A calm river flowing through a forest, cinematic")
                negative = gr.Textbox(label="negative_prompt", value="blurry, low quality, distorted")
                image = gr.Image(label="入力画像 (任意・最小I2V)", type="filepath")
                strength = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="image strength")
                preset = gr.Dropdown(list(PRESETS.keys()), value="phase1_default", label="preset")
                with gr.Row():
                    width = gr.Number(value=512, label="width (×32)", precision=0)
                    height = gr.Number(value=288, label="height (×32)", precision=0)
                with gr.Row():
                    crop_w = gr.Number(value=0, label="crop width (0=none)", precision=0)
                    crop_h = gr.Number(value=0, label="crop height (0=none)", precision=0)
                with gr.Row():
                    num_frames = gr.Number(value=49, label="num_frames (8n+1)", precision=0)
                    frame_rate = gr.Number(value=24.0, label="fps")
                seed = gr.Number(value=-1, label="seed (-1=random)", precision=0)
                generate_btn = gr.Button("生成", variant="primary")
            with gr.Column():
                job_box = gr.Textbox(label="job_id", interactive=False)
                progress_box = gr.Textbox(label="進捗", interactive=False)
                video_out = gr.Video(label="結果")

        status_btn.click(refresh_status, outputs=status_box)
        preset.change(apply_preset, inputs=preset, outputs=[width, height, num_frames, crop_w, crop_h])
        generate_btn.click(
            generate,
            inputs=[prompt, negative, image, strength, width, height, crop_w, crop_h, num_frames, frame_rate, seed],
            outputs=[progress_box, job_box, video_out],
        )
        demo.load(refresh_status, outputs=status_box)

    return demo


def _fetch_video(base_url: str, job_id: str, headers: dict) -> str | None:
    try:
        r = httpx.get(f"{base_url}/api/v1/jobs/{job_id}/video", headers=headers, timeout=60)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(prefix=f"{job_id}_", suffix=".mp4", delete=False)
        tmp.write(r.content)
        tmp.close()
        return tmp.name
    except Exception:
        return None
