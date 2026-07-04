"""Generate + Clip-Chain flows, factored out for unit testing (mock transport).
Both yield ``(progress_text, job_id, video_path)`` tuples and share the SAME 1s
poll loop (``_poll_job_until_done``), so /generate and /generate/chain get
byte-identical progress/complete/fail handling.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from .adapters import ADAPTER_NONE, _FALLBACK_MAX_VIDEO_MB, _FALLBACK_VIDEO_EXTS
from .api_client import ApiClient
from .formatting import format_api_error
from .i18n import L, _DEFAULT_LANG
from .validation import check_chain_total


# --------------------------------------------------------------------------- #
# Shared 1s poll loop (factored out of the generate flow so /generate and
# /generate/chain reuse the SAME progress/complete/fail handling). Yields
# (progress_text, job_id, video_path) tuples; behaviour is byte-identical to the
# original inline loop in make_generate_handler.
# --------------------------------------------------------------------------- #
def _poll_job_until_done(api: ApiClient, job_id: str, lang: str = _DEFAULT_LANG,
                         interval: float = 1.0, timeout_s: float = 3600.0):
    # Poll every ``interval`` seconds up to ``timeout_s`` (Settings-tab tunable;
    # defaults preserve the original 1s / 1h behaviour).
    try:
        iterations = max(1, int(float(timeout_s) / float(interval)))
    except (TypeError, ValueError, ZeroDivisionError):
        interval, iterations = 1.0, 3600
    for _ in range(iterations):
        time.sleep(interval)
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
# Generate flow, factored out for unit testing (mock transport). Yields
# (progress_text, job_id, video_path) tuples, matching the previous behaviour.
# --------------------------------------------------------------------------- #
def make_generate_handler(api: ApiClient, lang: str = _DEFAULT_LANG):
    default_lang = lang

    def generate(prompt, negative_prompt,
                 kf1_enabled, kf1_image, kf1_frame_idx, kf1_strength,
                 kf2_enabled, kf2_image, kf2_frame_idx, kf2_strength,
                 kf3_enabled, kf3_image, kf3_frame_idx, kf3_strength,
                 kf4_enabled, kf4_image, kf4_frame_idx, kf4_strength,
                 kf5_enabled, kf5_image, kf5_frame_idx, kf5_strength,
                 width, height, crop_enabled, crop_w, crop_h, num_frames, frame_rate, seed,
                 adapter=ADAPTER_NONE, adapter_strength=1.0, ref_video_path=None, config=None,
                 ui_lang=None, poll_interval=None, poll_timeout_min=None):
        # Runtime language + polling cadence come from Settings-tab gr.State
        # inputs (S6). They are optional so the pre-S6 call signature (and every
        # existing test) keeps working with the build-time default language and
        # the 1s / 60min poll defaults.
        lang = ui_lang or default_lang
        interval, timeout_s = _resolve_poll(poll_interval, poll_timeout_min)
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

        # 3) poll — shared with the clip-chain flow (cadence from Settings).
        yield from _poll_job_until_done(api, job_id, lang, interval, timeout_s)

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
    default_lang = lang

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
                       config=None, ui_lang=None, poll_interval=None,
                       poll_timeout_min=None):
        # Runtime language + poll cadence from Settings (S6); optional so the
        # pre-S6 signature and existing tests are unchanged.
        lang = ui_lang or default_lang
        interval, timeout_s = _resolve_poll(poll_interval, poll_timeout_min)
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

        # poll — shared with the generate flow (cadence from Settings).
        yield from _poll_job_until_done(api, job_id, lang, interval, timeout_s)

    return generate_chain


# --------------------------------------------------------------------------- #
# Settings-tab helpers (S6).
# --------------------------------------------------------------------------- #
def _resolve_poll(poll_interval, poll_timeout_min):
    """Resolve the (interval_s, timeout_s) pair from the Settings gr.Number
    inputs, falling back to 1s / 60min when unset or invalid."""
    try:
        interval = float(poll_interval) if poll_interval else 1.0
        if interval <= 0:
            interval = 1.0
    except (TypeError, ValueError):
        interval = 1.0
    try:
        timeout_min = float(poll_timeout_min) if poll_timeout_min else 60.0
        if timeout_min <= 0:
            timeout_min = 60.0
    except (TypeError, ValueError):
        timeout_min = 60.0
    return interval, timeout_min * 60.0


# Terminal states whose jobs "Delete all finished jobs" removes (the DELETE
# endpoint drops a terminal job + its output dir; active jobs are left alone).
_FINISHED_STATES = ("completed", "failed", "cancelled")


def delete_finished_jobs(api: ApiClient, lang: str = _DEFAULT_LANG) -> str:
    """Client-side "delete all finished jobs": list /jobs and DELETE every job in
    a terminal state (NO new endpoint). Returns a localized count message."""
    try:
        jobs = api.list_jobs()
    except Exception as exc:
        return L("msg_purge_failed", lang).format(err=exc)
    deleted = 0
    for job in jobs or []:
        if not isinstance(job, dict):
            continue
        if job.get("status") in _FINISHED_STATES:
            job_id = job.get("job_id")
            if not job_id:
                continue
            try:
                api.delete_job(job_id)
                deleted += 1
            except Exception:
                # Best-effort: a job may have been removed between list + delete.
                continue
    return L("msg_purge_done", lang).format(n=deleted)
