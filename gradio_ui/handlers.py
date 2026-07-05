"""Generate + Clip-Chain flows, factored out for unit testing (mock transport).
Both yield ``(progress_text, job_id, video_path)`` tuples and share the SAME 1s
poll loop (``_poll_job_until_done``), so /generate and /generate/chain get
byte-identical progress/complete/fail handling.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx

from .adapters import ADAPTER_NONE, _FALLBACK_MAX_VIDEO_MB, _FALLBACK_VIDEO_EXTS
from .adapters import MODEL_CATEGORIES, MODEL_DEFAULT
from .api_client import ApiClient
from .formatting import format_api_error
from .i18n import L, _DEFAULT_LANG
from .validation import check_chain_total, check_v2v_context

# Clip-chain generation modes (the mode Radio's stable values). V2V and A2V are
# mutually exclusive by construction — ONE radio, one value (mirrors the API's
# source_video x source_audio 422 without ever being able to trigger it).
MODE_NONE = "none"
MODE_V2V = "v2v"
MODE_A2V = "a2v"

# A2V source-audio precheck fallbacks used when /config is unavailable (mirrors
# config.py UploadConfig.allowed_audio_extensions / max_audio_size_mb). Kept
# here — not in adapters.py (owned by another work stream) — since only the
# chain flow consumes them.
_FALLBACK_AUDIO_EXTS = [".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"]
_FALLBACK_MAX_AUDIO_MB = 50


# F3: JobResponse.stage -> localized phase-label key. Unknown / absent stages
# show no label (older backends and the mock milestones never send one).
_STAGE_LABEL_KEYS = {
    "encode": "stage_encoding",
    "stage1": "stage_denoise_s1",
    "stage1_denoise": "stage_denoise_s1",
    "stage2_denoise": "stage_denoise_s2",
    "denoise": "stage_denoise",
    "tile": "stage_upsample",
    "decode": "stage_decode",
}


def _format_running_progress(job: dict, lang: str) -> str:
    """Progress line for a running job (F3, display-only).

    * step/total known -> the classic "Generating… 42% (step 3/8)";
    * unknown -> percent only (never the literal "step None/None");
    * a recognized ``stage`` appends its localized phase label.
    """
    progress = job.get("progress", 0.0)
    step = job.get("current_step")
    total = job.get("total_steps")
    if step is not None and total is not None:
        text = L("msg_generating", lang).format(pct=progress, step=step, total=total)
    else:
        text = L("msg_generating_pct", lang).format(pct=progress)
    stage_key = _STAGE_LABEL_KEYS.get(job.get("stage") or "")
    if stage_key:
        text = f"{text} — {L(stage_key, lang)}"
    return text


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
        if status == "running":
            yield _format_running_progress(job, lang), job_id, None
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
                       poll_timeout_min=None,
                       mode=MODE_NONE, src_video=None, context_frames=73,
                       src_audio=None):
        # Runtime language + poll cadence from Settings (S6); optional so the
        # pre-S6 signature and existing tests are unchanged.
        # V2V/A2V (ADDITIVE): ``mode`` + the mode's source input are appended
        # after the S6 params so every pre-existing positional call keeps its
        # meaning; the defaults reproduce the pre-V2V payload byte-for-byte.
        lang = ui_lang or default_lang
        interval, timeout_s = _resolve_poll(poll_interval, poll_timeout_min)
        mode = mode if mode in (MODE_V2V, MODE_A2V) else MODE_NONE
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

        # Clip-count floor mirrors the API validator: plain chain needs 2-8,
        # V2V allows 1-8 (the frozen source tail IS the prior segment), A2V is
        # exactly 1 (one frozen audio latent spans one clip).
        if mode == MODE_V2V:
            if not (1 <= len(enabled) <= 8):
                yield L("v2v_msg_clip_count", lang), "", None
                return
        elif mode == MODE_A2V:
            if len(enabled) != 1:
                yield L("a2v_msg_clip_count", lang), "", None
                return
        elif not (2 <= len(enabled) <= 8):
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

        # --- V2V prechecks (zero API calls on violation; mirrors the API's
        # SourceVideoSpec + cross-validators) ---
        upload_cfg = (config or {}).get("upload") or {}
        if mode == MODE_V2V:
            err = check_v2v_context(context_frames, clip_frames[0], lang, config)
            if err is not None:
                yield err, "", None
                return
            if not src_video:
                yield L("v2v_msg_video_required", lang), "", None
                return
            allowed_exts = upload_cfg.get("allowed_video_extensions") or _FALLBACK_VIDEO_EXTS
            ext = Path(str(src_video)).suffix.lower()
            if ext not in [e.lower() for e in allowed_exts]:
                yield L("v2v_msg_bad_extension", lang).format(exts=", ".join(allowed_exts)), "", None
                return
            max_mb = upload_cfg.get("max_video_size_mb")
            if max_mb is None:
                max_mb = _FALLBACK_MAX_VIDEO_MB
            try:
                size_mb = os.path.getsize(str(src_video)) / (1024 * 1024)
            except OSError:
                size_mb = 0.0
            if size_mb > max_mb:
                yield L("v2v_msg_too_large", lang).format(limit=max_mb), "", None
                return
            # The frozen source tail occupies clip 0's head — a start image on
            # clip 1 would conflict (server 422); reject before any upload.
            if enabled[0][2]:
                yield L("v2v_msg_image_conflict", lang), "", None
                return

        # --- A2V prechecks (zero API calls on violation) ---
        if mode == MODE_A2V:
            if not src_audio:
                yield L("a2v_msg_audio_required", lang), "", None
                return
            allowed_audio = upload_cfg.get("allowed_audio_extensions") or _FALLBACK_AUDIO_EXTS
            ext = Path(str(src_audio)).suffix.lower()
            if ext not in [e.lower() for e in allowed_audio]:
                yield L("a2v_msg_bad_extension", lang).format(exts=", ".join(allowed_audio)), "", None
                return
            max_audio_mb = upload_cfg.get("max_audio_size_mb")
            if max_audio_mb is None:
                max_audio_mb = _FALLBACK_MAX_AUDIO_MB
            try:
                size_mb = os.path.getsize(str(src_audio)) / (1024 * 1024)
            except OSError:
                size_mb = 0.0
            if size_mb > max_audio_mb:
                yield L("a2v_msg_too_large", lang).format(limit=max_audio_mb), "", None
                return

        # Total-timeline geometry: same arithmetic as the API validator. For V2V
        # the frozen-head geometry (incl. the stage-2 tile-fit invariant) is
        # forwarded so the precheck matches the server's compute_chain_layout.
        err = check_chain_total(
            clip_frames, fps, kv, lang,
            source_context_px=int(context_frames) if mode == MODE_V2V else None,
        )
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

        # --- V2V/A2V source upload (after the keyframe image, mirroring the
        # generate flow's "reference video last" ordering) ---
        source_video_id: str | None = None
        source_audio_id: str | None = None
        if mode == MODE_V2V:
            yield L("v2v_msg_uploading", lang), "", None
            try:
                source_video_id = api.upload_video(str(src_video))
            except Exception as exc:
                yield L("msg_upload_failed", lang).format(err=exc), "", None
                return
        elif mode == MODE_A2V:
            yield L("a2v_msg_uploading", lang), "", None
            try:
                source_audio_id = api.upload_audio(str(src_audio))
            except Exception as exc:
                yield L("msg_upload_failed", lang).format(err=exc), "", None
                return

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
        # ADDITIVE source keys: only present in their mode, so a mode="none"
        # request stays byte-identical to the pre-V2V payload (frozen-API
        # discipline mirrored client-side).
        if mode == MODE_V2V:
            payload["source_video"] = {
                "video_id": source_video_id,
                "context_frames": int(context_frames),
            }
        elif mode == MODE_A2V:
            payload["source_audio"] = {"audio_id": source_audio_id}

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
# V2V join flow ("Create joined version" button). Separate from the chain flow:
# joining is a POST-completion, GPU-free server-side ffmpeg step
# (POST /jobs/{id}/join), so it hangs off the finished job id rather than the
# generation generator. Yields (message, joined_video_path) pairs.
#
# Per the approved UI decision the checkbox is a two-way switch — "create the
# smoothed joined version" (default ON) vs "don't" — so the GUI only ever calls
# the server's default smoothed join (audio_smoothing=true); the API's
# hard-concat variant is never sent from here.
# --------------------------------------------------------------------------- #
def make_join_handler(api: ApiClient, lang: str = _DEFAULT_LANG):
    default_lang = lang

    def join(job_id, smoothing_enabled, crossfade_ms=None, ui_lang=None):
        lang = ui_lang or default_lang
        job_id = str(job_id).strip() if job_id else ""
        if not job_id:
            yield L("v2v_msg_no_job", lang), None
            return
        if not smoothing_enabled:
            yield L("v2v_msg_join_disabled", lang), None
            return

        # F5: the crossfade-length Dropdown (150/300/500 ms) rides along as
        # JoinRequest.handle_crossfade_ms; None / junk falls back to the server
        # default (300 ms) by simply omitting the field.
        payload: dict = {}
        try:
            if crossfade_ms is not None:
                payload["handle_crossfade_ms"] = int(crossfade_ms)
        except (TypeError, ValueError):
            payload = {}

        yield L("v2v_msg_joining", lang), None
        try:
            resp = api.join_job(job_id, payload or None)
        except Exception as exc:
            yield L("v2v_msg_join_failed", lang).format(err=exc), None
            return
        if resp.status_code >= 400:
            try:
                err_body: object = resp.json()
            except Exception:
                err_body = resp.text
            yield format_api_error(err_body, lang), None
            return
        info = resp.json()

        try:
            video = api.fetch_joined(job_id)
        except Exception as exc:
            yield L("v2v_msg_join_failed", lang).format(err=exc), None
            return
        yield L("v2v_msg_join_done", lang).format(
            mode=info.get("join_mode", ""), job_id=job_id), video

    return join


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


# --------------------------------------------------------------------------- #
# Settings tab: /config fetch + automatic retry (bug fix). A page load used to
# perform exactly ONE /config fetch and silently swallow any failure into
# ``{}`` -- if that single fetch raced server startup or hit a transient
# error, the Settings-tab spill-free table (and preset/adapter choices) stayed
# empty FOREVER with no error and no retry, indistinguishable from "still
# loading". These two module-level helpers (mirroring ``delete_finished_jobs``
# above: take the ApiClient explicitly, no Gradio runtime needed) are shared by
# ui.py's page-load handler, the manual Refresh button, and a one-shot
# gr.Timer armed after a failed page load, so all three share IDENTICAL
# failure semantics.
# --------------------------------------------------------------------------- #

# Automatic retry attempts (via the Settings-tab gr.Timer) before giving up and
# pointing the user at the manual Refresh button instead of retrying forever
# in the background against a persistently-dead server.
CONFIG_RETRY_MAX_ATTEMPTS = 5


def fetch_config_safe(api: ApiClient, lang: str = _DEFAULT_LANG) -> tuple[dict | None, str | None]:
    """Fetch /config; returns ``(cfg, None)`` on success or ``(None, warning)``
    on failure. Never raises and never guesses at a fallback config -- the
    caller decides what to do with a ``None`` cfg (the safe default is: keep
    whatever was already displayed, do NOT clobber it with ``{}``)."""
    try:
        return api.get_config(), None
    except Exception as exc:
        return None, L("warn_config_load_failed", lang).format(err=exc)


def on_config_retry_tick(api: ApiClient, attempt: int,
                          lang: str = _DEFAULT_LANG) -> tuple[dict | None, str | None, int, bool]:
    """One tick of the Settings-tab auto-retry timer (armed after a failed
    page-load /config fetch). Returns
    ``(cfg_or_none, warning_or_none, next_attempt, keep_retrying)``:

    * success               -> ``(cfg, None, attempt, False)`` -- stop the timer.
    * failure, budget left  -> ``(None, transient_msg, attempt + 1, True)`` --
      keep ticking (no gr.Warning here; the caller only surfaces the initial
      failure and the final exhaustion, not every silent retry).
    * failure, exhausted    -> ``(None, final_msg, attempt + 1, False)`` --
      stop and tell the user to use the manual Refresh button instead.
    """
    cfg, err = fetch_config_safe(api, lang)
    if err is None:
        return cfg, None, attempt, False
    next_attempt = attempt + 1
    if next_attempt >= CONFIG_RETRY_MAX_ATTEMPTS:
        final_msg = L("warn_config_retry_exhausted", lang).format(
            err=err, n=CONFIG_RETRY_MAX_ATTEMPTS)
        return None, final_msg, next_attempt, False
    return None, err, next_attempt, True


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


# --------------------------------------------------------------------------- #
# Model management (Settings tab "Models" section, S3). Standalone handlers —
# the shared generate/chain flows and _poll_job_until_done above are untouched
# (model load is a synchronous POST, not a polled job).
# --------------------------------------------------------------------------- #

def fetch_models_safe(api: ApiClient, lang: str = _DEFAULT_LANG) -> tuple[dict | None, str | None]:
    """Fetch GET /models; ``(models_json, None)`` on success or
    ``(None, warning)`` on failure. Mirrors :func:`fetch_config_safe`: never
    raises and never invents a fallback — the caller keeps the dropdowns
    as-is when the fetch fails."""
    try:
        return api.get_models(), None
    except Exception as exc:
        return None, L("model_fetch_failed", lang).format(err=exc)


def load_selected_models(api: ApiClient, transformer: str | None, text_encoder: str | None,
                         video_vae: str | None, audio: str | None,
                         lang: str = _DEFAULT_LANG) -> str:
    """POST /pipeline/load with the four dropdown selections.

    Returns a localized status line for the Models-section status box. An
    empty/None dropdown value falls back to ``"default"`` (the server-side
    always-safe entry). REST errors are rendered through
    :func:`gradio_ui.formatting.format_api_error` so the model-management
    codes (MODEL_NOT_FOUND / MODEL_FILE_MISSING / MODEL_INCOMPATIBLE /
    JOB_BUSY / PIPELINE_LOAD_FAILED) each get their actionable hint."""
    models = {
        "transformer": transformer or MODEL_DEFAULT,
        "text_encoder": text_encoder or MODEL_DEFAULT,
        "video_vae": video_vae or MODEL_DEFAULT,
        "audio": audio or MODEL_DEFAULT,
    }
    try:
        resp = api.load_pipeline_models(models)
    except httpx.HTTPStatusError as exc:
        try:
            body: object = exc.response.json()
        except Exception:
            body = exc.response.text
        return L("model_load_failed", lang).format(err=format_api_error(body, lang))
    except Exception as exc:
        return L("model_load_failed", lang).format(err=exc)
    active = resp.get("models") or models
    summary = ", ".join(
        f"{category}={active.get(category, MODEL_DEFAULT)}" for category in MODEL_CATEGORIES
    )
    return L("model_load_ok", lang).format(models=summary)
