"""Generate + Clip-Chain flows, factored out for unit testing (mock transport).
Both yield ``(progress_text, job_id, video_path)`` tuples and share the SAME 1s
poll loop (``_poll_job_until_done``), so /generate and /generate/chain get
byte-identical progress/complete/fail handling.
"""

from __future__ import annotations

import math
import os
import re
import time
import wave
from pathlib import Path

import gradio as gr
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


def _wav_duration_seconds(path) -> float | None:
    """Duration in seconds of a ``.wav`` file via the stdlib :mod:`wave` module,
    or ``None`` when ``path`` is not a readable ``.wav`` (a non-wav container, an
    unreadable/short header, or a zero frame rate).

    Used by the A2V length precheck to reject a too-short upload with the exact
    seconds needed BEFORE any API call. A ``None`` result means "cannot measure
    here" and the caller defers to the server's ffprobe preflight — so mp3/m4a/…
    (which the stdlib cannot decode) and malformed wavs still reach the server
    unchanged, matching the pre-precheck behaviour."""
    if not path or Path(str(path)).suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
    except Exception:
        return None
    if not rate or rate <= 0:
        return None
    return frames / float(rate)


def _resolve_fps(fps) -> float:
    """Coerce a (possibly falsy/non-numeric) frame-rate input to a float,
    falling back to 24.0 — the same fallback the A2V length precheck in
    :func:`make_generate_handler` uses for ``frame_rate``."""
    try:
        fps_v = float(fps) if fps else 24.0
    except (TypeError, ValueError):
        fps_v = 24.0
    return fps_v or 24.0


def suggest_frames_for_audio(dur: float, fps) -> int:
    """Suggest a ``Frames`` value (8n+1) that fits ``dur`` seconds of audio at
    ``fps`` (auto-adjust the Generate-tab A2V audio attach, feature 1).

    Starts from the largest 8n+1 frame count the duration covers
    (``((floor(dur*fps) - 1) // 8) * 8 + 1``), then verifies it against the
    SAME arithmetic as the server preflight (mirrors the A2V length precheck
    in :func:`make_generate_handler`: :func:`chain_math.audio_latents_required`
    with ``kv=3`` — mirroring the A2V chain payload's fixed
    ``overlap_frames`` — vs ``round(dur * chain_math.AUDIO_LATENTS_PER_SEC)``),
    shrinking by 8 while the attached audio would VAE-encode to fewer latent
    frames than that frame count requires. The result is finally clamped to
    ``[9, 481]`` (the server's ``num_frames`` range); both the formula and the
    shrink step stay on the 8n+1 grid so the clamp bounds (9 and 481) are the
    only values that can land off-grid, and both of those are themselves
    8n+1."""
    import chain_math

    fps_v = _resolve_fps(fps)
    nf = ((math.floor(dur * fps_v) - 1) // 8) * 8 + 1
    nf = max(nf, 9)
    while nf > 9:
        try:
            required = chain_math.audio_latents_required([nf], fps_v, kv=3)
        except ValueError:
            # nf too small for the fixed kv=3 overlap to even be computed
            # (mirrors the same edge case the server-side arithmetic would
            # hit) — stop shrinking; the final clamp below still applies.
            break
        available = round(dur * chain_math.AUDIO_LATENTS_PER_SEC)
        if available >= required:
            break
        nf -= 8
    return max(9, min(481, nf))


def a2v_audio_change_handler(path, frame_rate, lang: str = _DEFAULT_LANG):
    """``.change`` handler for the Generate-tab A2V audio ``gr.File``
    (feature 1: wav-only auto Frames adjustment).

    A ``.wav`` attach ALWAYS overwrites ``num_frames`` (regardless of its
    current value) with :func:`suggest_frames_for_audio`'s result and surfaces
    a ``gr.Info`` toast naming the new value. Anything :func:`_wav_duration_seconds`
    cannot measure — a non-wav container, an unreadable/malformed wav, or a
    cleared attachment (``path is None``) — is a no-op: ``gr.update()`` with no
    value change, leaving Frames exactly as the user left it."""
    dur = _wav_duration_seconds(path)
    if dur is None:
        return gr.update()
    nf = suggest_frames_for_audio(dur, frame_rate)
    gr.Info(L("a2v_msg_frames_adjusted", lang).format(frames=nf, dur=dur))
    return gr.update(value=nf)


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
    * a recognized ``stage`` appends its localized phase label;
    * chain clip position (``clip``/``clip_count``, additive JobResponse
      fields) appends "clip n/N" so a chain job shows which clip it is on —
      the n -> n+1 change IS the "clip n done, clip n+1 started" signal.
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
    clip = job.get("clip")
    clip_count = job.get("clip_count")
    if clip is not None and clip_count:
        text = f"{text} — {L('msg_clip_progress', lang).format(clip=clip, total=clip_count)}"
    return text


# A queued job that has waited this many seconds without starting switches its
# progress message from a plain "waiting" tick to a "stuck — cancel from Jobs"
# hint. The poll never auto-aborts; this only changes the wording.
QUEUED_WARN_SECONDS = 30


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
    queued_elapsed = 0.0
    for _ in range(iterations):
        time.sleep(interval)
        try:
            job = api.get_job(job_id)
        except Exception as exc:
            yield L("msg_poll_failed", lang).format(err=exc), job_id, None
            continue

        status = job["status"]
        if status == "queued":
            # The single-job guard means a job can sit queued behind another job
            # (or wait for the worker to pick it up). Surface the wait instead of
            # a silent frozen progress box: below the threshold, a plain "waiting"
            # tick; past it, a "stuck" hint pointing at the Jobs-tab cancel. The
            # poll itself keeps going — we never auto-abort a queued job.
            queued_elapsed += interval
            secs = int(queued_elapsed)
            key = "msg_queued_stuck" if queued_elapsed >= QUEUED_WARN_SECONDS else "msg_queued"
            yield L(key, lang).format(secs=secs), job_id, None
        elif status == "running":
            yield _format_running_progress(job, lang), job_id, None
        elif status == "completed":
            yield L("msg_completing", lang), job_id, None
            try:
                video = api.fetch_video(job_id)
            except Exception:
                video = None
            done_text = L("msg_completed", lang).format(job_id=job_id)
            # Chain jobs report their clip total (additive JobResponse field);
            # say so on completion — "all N clips processed" answers the
            # "did every clip actually get generated?" doubt at a glance.
            clip_count = job.get("clip_count")
            if clip_count:
                done_text = f"{done_text} — {L('msg_all_clips_done', lang).format(n=clip_count)}"
            yield done_text, job_id, video
            return
        elif status in ("failed", "cancelled"):
            yield L("msg_failed", lang).format(status=status, error=job.get("error")), job_id, None
            return

    yield L("msg_timeout", lang), job_id, None


# --------------------------------------------------------------------------- #
# Prompt-embedded style/character LoRA tokens: ``<lora:name:weight>`` (S2). The
# weight is optional (default 1.0) and the ``lora`` keyword is case-insensitive
# (``<LORA:...>`` allowed). Nothing is invented client-side — the weight range
# mirrors the server's ``0 < strength <= 2.0`` rule; out-of-range weights are
# clamped into [MIN, MAX] (MIN reuses the Generate-tab adapter-strength slider's
# own 0.05 floor) and a toast warns. Name resolution is against the GET /loras
# name set, case-insensitive with exact match preferred; an unknown name aborts
# the send (mirrors the existing precheck flow).
# --------------------------------------------------------------------------- #
_LORA_TOKEN_RE = re.compile(r"<lora:([^:>]+)(?::([0-9]*\.?[0-9]+))?>", re.IGNORECASE)
LORA_WEIGHT_MIN = 0.05
LORA_WEIGHT_MAX = 2.0
LORA_WEIGHT_DEFAULT = 1.0


def _merge_loras(loras: list[dict]) -> list[dict]:
    """Dedup a lora list by name: the LAST occurrence's strength wins, the FIRST
    occurrence's position is kept (plain dict insertion-order semantics). A
    single-element / empty list round-trips unchanged, so the payload stays
    byte-identical on the adapter-only and no-lora paths."""
    merged: dict[str, dict] = {}
    for item in loras:
        merged[item["name"]] = item
    return list(merged.values())


def parse_prompt_loras(prompt, known_names, lang: str = _DEFAULT_LANG):
    """Extract ``<lora:name:weight>`` tokens from ``prompt``.

    Returns ``(cleaned_prompt, loras, error)``:

    * ``cleaned_prompt`` — the prompt with every matched token removed and the
      whitespace runs left behind collapsed to single spaces (applied ONLY when
      a token was actually removed, so a token-free prompt is returned byte-for-
      byte unchanged — the caller only invokes this when a token is present);
    * ``loras`` — ``[{"name", "strength"}]`` in first-seen order, deduped
      last-wins by resolved name;
    * ``error`` — a localized message (unknown token name) meaning "abort the
      send with zero generate/upload calls", else ``None``. Weight-range clamps
      are non-fatal: they fire a ``gr.Warning`` toast and continue.

    ``known_names`` is the GET /loras name set; resolution is case-insensitive
    with an exact match preferred.
    """
    exact = set(known_names)
    lower_map: dict[str, str] = {}
    for n in known_names:
        lower_map.setdefault(n.lower(), n)

    def _resolve(raw: str):
        if raw in exact:
            return raw
        return lower_map.get(raw.lower())

    collected: list[dict] = []
    unknown: list[str] = []
    for m in _LORA_TOKEN_RE.finditer(prompt or ""):
        raw_name = m.group(1).strip()
        resolved = _resolve(raw_name)
        if resolved is None:
            unknown.append(raw_name)
            continue
        if m.group(2) is None:
            weight = LORA_WEIGHT_DEFAULT
        else:
            weight = float(m.group(2))
            if weight < LORA_WEIGHT_MIN or weight > LORA_WEIGHT_MAX:
                clamped = min(max(weight, LORA_WEIGHT_MIN), LORA_WEIGHT_MAX)
                gr.Warning(L("lora_warn_weight_clamp", lang).format(
                    name=resolved, given=weight, clamped=clamped))
                weight = clamped
        collected.append({"name": resolved, "strength": weight})

    # Unknown name(s) -> abort (mirrors the "reject before any API call" flow).
    if unknown:
        return prompt, [], L("lora_msg_unknown", lang).format(names=", ".join(unknown))
    # No token matched at all -> leave the prompt byte-identical.
    if not collected:
        return prompt, [], None

    cleaned = _LORA_TOKEN_RE.sub("", prompt)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, _merge_loras(collected), None


def _combine_generate_loras(use_adapter, adapter, adapter_strength, prompt_loras):
    """Merge the Generate tab's two LoRA sources into the single ``loras`` payload
    list: the reference-video CONTROL adapter from the dropdown (S4) first, then
    the prompt-embedded ``<lora:...>`` style/character adapters (S2) in prompt
    order, deduped last-wins by name (:func:`_merge_loras`). Returns ``[]`` when
    neither is present, so the caller adds the ``loras`` key ONLY when non-empty
    and the no-lora path stays byte-identical.

    Shared by ``POST /generate`` and the A2V ``POST /generate/chain`` path so both
    build ``loras`` identically. A2V passes the same ``use_adapter``/``adapter``/
    ``adapter_strength`` as the single-shot path (a 1-clip chain also accepts a
    control adapter + ``reference_video_id``), so both the control adapter and
    the prompt style/character tokens are merged the same way on either path."""
    combined: list[dict] = []
    if use_adapter:
        combined.append({"name": adapter, "strength": float(adapter_strength)})
    combined.extend(prompt_loras)
    return _merge_loras(combined)


def build_a2v_chain_payload(
    *,
    audio_id,
    num_frames,
    prompt,
    negative_prompt,
    width,
    height,
    crop_output,
    frame_rate,
    seed,
    conditioning_images=None,
    loras=None,
    use_adapter=False,
    reference_video_id=None,
    control_adherence=1.0,
    reference_strength=1.0,
):
    """Assemble the A2V ``POST /generate/chain`` body (案A): a single ChainClip
    carrying ``num_frames`` + any keyframe ``conditioning_images``, the frozen
    distilled quality contract, ``overlap_frames=3``/``overlap_strength=0.5``, and
    ``source_audio.audio_id``. A pure function (primitives + ID strings in, dict
    out) with NO Gradio / gr.* / ApiClient dependency, so the batch runner can
    build the byte-identical payload off the UI thread.

    Optional keys reproduce the Generate-tab A2V branch exactly: ``conditioning_images``
    only when non-empty, ``loras`` only when the combined list is non-empty, and
    ``reference_video_id`` (+ the S3 ``conditioning_attention_strength`` /
    ``reference_video_strength`` keys, each only below 1.0) only when an adapter is
    used -- so a token-free, adapter-free request stays byte-identical to before."""
    clip_entry: dict = {"num_frames": int(num_frames)}
    if conditioning_images:
        clip_entry["conditioning_images"] = conditioning_images
    chain_payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or "",
        "width": int(width),
        "height": int(height),
        "crop_output": crop_output,
        "frame_rate": float(frame_rate),
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "seed": int(seed),
        "pipeline": "distilled",
        "overlap_frames": 3,
        "overlap_strength": 0.5,
        "clips": [clip_entry],
        "source_audio": {"audio_id": audio_id},
    }
    if loras:
        chain_payload["loras"] = loras
    if use_adapter:
        chain_payload["reference_video_id"] = reference_video_id
        if float(control_adherence) < 1.0:
            chain_payload["conditioning_attention_strength"] = float(control_adherence)
        if float(reference_strength) < 1.0:
            chain_payload["reference_video_strength"] = float(reference_strength)
    return chain_payload


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
                 adapter=ADAPTER_NONE, adapter_strength=1.0,
                 control_adherence=1.0, reference_strength=1.0,
                 ref_video_path=None, config=None,
                 ui_lang=None, poll_interval=None, poll_timeout_min=None,
                 src_audio=None):
        # Runtime language + polling cadence come from Settings-tab gr.State
        # inputs (S6). They are optional so the pre-S6 call signature (and every
        # existing test) keeps working with the build-time default language and
        # the 1s / 60min poll defaults.
        # ``src_audio`` (A2V, case A) is the LAST positional arg -- a gr.File
        # value (path str or None). Wiring the audio input component into
        # ``inputs=[...]`` is owned by another work stream.
        lang = ui_lang or default_lang
        interval, timeout_s = _resolve_poll(poll_interval, poll_timeout_min)
        if not prompt or not prompt.strip():
            yield L("msg_prompt_required", lang), "", None
            return

        # 0) width/height (÷64) + num_frames (8n+1, [9, 481]) precheck, moved
        # over from the chain handler's identical rule (:468-470 / :533) so a
        # violating Generate-tab request never reaches the API either. Zero
        # API calls on violation -- same _precheck_reject discipline as chain.
        try:
            w_i, h_i = int(width), int(height)
        except (TypeError, ValueError):
            yield _precheck_reject(L("msg_bad_dimension", lang)), "", None
            return
        if w_i % 64 != 0 or h_i % 64 != 0:
            yield _precheck_reject(L("msg_bad_dimension", lang)), "", None
            return
        try:
            nf_i = int(num_frames)
        except (TypeError, ValueError):
            yield _precheck_reject(L("msg_chain_bad_frames", lang).format(n=1)), "", None
            return
        if nf_i < 9 or nf_i > 481 or (nf_i - 1) % 8 != 0:
            yield _precheck_reject(L("msg_chain_bad_frames", lang).format(n=1)), "", None
            return

        # 0b) A2V (案A): a src_audio upload routes this SAME handler to
        # POST /generate/chain as a single-clip chain carrying a frozen
        # source_audio latent, instead of POST /generate. A2V+LoRA is allowed:
        # style/character IC-LoRAs (prompt <lora:...> tokens) AND a
        # reference-video CONTROL adapter (the dropdown above =
        # canny/pose/upscaler) are both wired into the chain payload below
        # (``loras`` + ``reference_video_id`` / S3 strength keys) and applied
        # to the clip. The server allows a control adapter on a chain only
        # when it carries exactly one clip -- true here since this handler
        # always sends a single ChainClip.
        use_audio = bool(src_audio)
        if use_audio:
            # Length precheck (wav only): the server rejects audio that
            # VAE-encodes to fewer audio-latent frames than the assembled
            # timeline (422 SOURCE_AUDIO_TOO_SHORT — the single most likely A2V
            # failure). For a .wav we measure the duration locally (stdlib wave)
            # and reject with the exact seconds needed BEFORE any upload/API
            # call, using the SAME arithmetic as the server preflight
            # (services/pipeline_manager.preflight_source_audio): required =
            # chain_math.audio_latents_required([num_frames], fps, kv=3) latent
            # frames, available = round(duration * AUDIO_LATENTS_PER_SEC). kv=3
            # mirrors the A2V chain payload's fixed overlap_frames below. Non-wav
            # (mp3/m4a/…) and unreadable wavs skip this and defer to the server.
            audio_dur = _wav_duration_seconds(src_audio)
            if audio_dur is not None:
                import chain_math

                fps_v = float(frame_rate) if frame_rate else 24.0
                required = chain_math.audio_latents_required([nf_i], fps_v, kv=3)
                available = round(audio_dur * chain_math.AUDIO_LATENTS_PER_SEC)
                if available < required:
                    need_s = required / chain_math.AUDIO_LATENTS_PER_SEC
                    yield _precheck_reject(L("a2v_msg_too_short", lang).format(
                        frames=nf_i, fps=fps_v, video=nf_i / fps_v,
                        need=need_s, have=float(audio_dur))), "", None
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

        # 1c) prompt-embedded <lora:name:weight> tokens (S2). Resolved + stripped
        # here — BEFORE any upload — so an unknown token name aborts the send
        # with zero generate/upload calls (matches the precheck discipline). The
        # GET /loras name lookup runs only when a token is actually present, so a
        # token-free prompt performs no extra call and stays byte-identical.
        send_prompt = prompt
        prompt_loras: list[dict] = []
        if _LORA_TOKEN_RE.search(prompt or ""):
            try:
                lora_list = api.list_loras()
            except Exception as exc:
                yield _precheck_reject(L("lora_msg_list_failed", lang).format(err=exc)), "", None
                return
            known = [e.get("name") for e in (lora_list or [])
                     if isinstance(e, dict) and e.get("name")]
            send_prompt, prompt_loras, lora_err = parse_prompt_loras(prompt, known, lang)
            if lora_err is not None:
                yield _precheck_reject(lora_err), "", None
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

        # --- A2V (案A): upload the audio, then POST /generate/chain with a
        # single ChainClip carrying this same request's num_frames + any
        # collected keyframe conditioning, and source_audio.audio_id. Mirrors
        # make_chain_handler's A2V branch (:640-646, :699-700) but with only
        # ONE clip (this handler has no per-clip slots). ---
        if use_audio:
            yield L("a2v_msg_uploading", lang), "", None
            try:
                audio_id = api.upload_audio(str(src_audio))
            except Exception as exc:
                yield L("msg_upload_failed", lang).format(err=exc), "", None
                return
            # A2V+LoRA (ADDITIVE): wire the prompt <lora:...> style/character
            # adapters AND the reference-video CONTROL adapter (S4 dropdown)
            # into the chain payload's ``loras`` -- same builder as the single
            # /generate path. The key is added ONLY when non-empty, so a
            # token-free, adapter-free A2V request stays byte-identical to
            # before. Payload assembly (including the S3 optional-below-1.0 send
            # discipline for the reference strengths) lives in the pure
            # build_a2v_chain_payload so the batch runner emits the same body.
            a2v_loras = _combine_generate_loras(
                use_adapter, adapter, adapter_strength, prompt_loras)
            chain_payload = build_a2v_chain_payload(
                audio_id=audio_id,
                num_frames=num_frames,
                prompt=send_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                crop_output=crop_output,
                frame_rate=frame_rate,
                seed=seed,
                conditioning_images=conditioning,
                loras=a2v_loras,
                use_adapter=use_adapter,
                reference_video_id=reference_video_id,
                control_adherence=control_adherence,
                reference_strength=reference_strength,
            )
            try:
                resp = api.generate_chain(chain_payload)
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
            yield L("msg_job_started", lang).format(mode="a2v", job_id=job_id), job_id, None
            yield from _poll_job_until_done(api, job_id, lang, interval, timeout_s)
            return

        payload = {
            "prompt": send_prompt,
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
        # loras: combine the adapter-dropdown control LoRA (IC-LoRA, S4) with the
        # prompt <lora:...> style/character LoRAs (S2). Adapter first, then prompt
        # order; deduped last-wins by name. The "loras" key is added ONLY when the
        # merged list is non-empty, and reference_video_id (+ the S3 strength
        # keys) only when an adapter is selected — so the request stays
        # byte-identical to the pre-S2/S4 payload on the no-adapter/no-token path
        # (no keys) AND on the adapter-only path (single-entry list, same order).
        merged_loras = _combine_generate_loras(
            use_adapter, adapter, adapter_strength, prompt_loras)
        if merged_loras:
            payload["loras"] = merged_loras
        if use_adapter:
            payload["reference_video_id"] = reference_video_id
            # S3: control-adherence + reference-strength are optional server-side
            # (default 1.0). Send each key ONLY when the slider is below 1.0 so
            # the default op stays byte-identical to the pre-S3 request.
            if float(control_adherence) < 1.0:
                payload["conditioning_attention_strength"] = float(control_adherence)
            if float(reference_strength) < 1.0:
                payload["reference_video_strength"] = float(reference_strength)
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


def _precheck_reject(message: str) -> str:
    """Surface a chain-precheck rejection as a Gradio toast (``gr.Warning``) in
    ADDITION to the ``chain_progress`` textbox line. The textbox alone proved
    too easy to miss -- a real user read the silent early-return as "the button
    does nothing" and kept clicking. Returns ``message`` unchanged so call
    sites stay one-liners: ``yield _precheck_reject(...), "", None``.

    ``gr.Warning`` is gradio's non-raising notification API: inside a queued
    event it renders the yellow toast modal; outside one (unit tests) it
    degrades to ``warnings.warn`` (verified on gradio 6.19.0,
    ``gradio.helpers.log_message``).
    """
    gr.Warning(message)
    return message


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
        # --- prechecks (localized; NO API call on any violation). Every
        # rejection ALSO fires a gr.Warning toast (_precheck_reject) so the
        # early return is visible even when the progress textbox goes
        # unnoticed. ---
        if not prompt or not prompt.strip():
            yield _precheck_reject(L("msg_prompt_required", lang)), "", None
            return

        try:
            w, h = int(width), int(height)
        except (TypeError, ValueError):
            yield _precheck_reject(L("msg_bad_dimension", lang)), "", None
            return
        if w % 64 != 0 or h % 64 != 0:
            yield _precheck_reject(L("msg_bad_dimension", lang)), "", None
            return
        limits = (config or {}).get("limits") or {}
        max_w = limits.get("max_width", 1920)
        max_h = limits.get("max_height", 1088)
        if w > max_w or h > max_h:
            yield _precheck_reject(L("msg_size_limit", lang).format(maxw=max_w, maxh=max_h)), "", None
            return

        if crop_enabled:
            try:
                cw, ch = int(crop_w), int(crop_h)
            except (TypeError, ValueError):
                cw = ch = 0
            if cw < 32 or ch < 32 or cw > w or ch > h:
                yield _precheck_reject(L("msg_crop_range", lang)), "", None
                return

        try:
            fps = float(frame_rate)
        except (TypeError, ValueError):
            yield _precheck_reject(L("msg_fps_range", lang)), "", None
            return
        if fps < 1.0 or fps > 60.0:
            yield _precheck_reject(L("msg_fps_range", lang)), "", None
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
                yield _precheck_reject(L("v2v_msg_clip_count", lang)), "", None
                return
        elif mode == MODE_A2V:
            if len(enabled) != 1:
                yield _precheck_reject(L("a2v_msg_clip_count", lang)), "", None
                return
        elif not (2 <= len(enabled) <= 8):
            yield _precheck_reject(L("msg_chain_clip_count", lang)), "", None
            return

        # Per-clip num_frames: 8n+1 within [9, 481].
        clip_frames: list[int] = []
        for n, (_p, nf, _img, _strg) in enumerate(enabled, start=1):
            try:
                nf_i = int(nf)
            except (TypeError, ValueError):
                yield _precheck_reject(L("msg_chain_bad_frames", lang).format(n=n)), "", None
                return
            if nf_i < 9 or nf_i > 481 or (nf_i - 1) % 8 != 0:
                yield _precheck_reject(L("msg_chain_bad_frames", lang).format(n=n)), "", None
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
            yield _precheck_reject(L("msg_chain_overlap_too_large", lang).format(kv=kv, maxkv=max_kv)), "", None
            return

        # --- V2V prechecks (zero API calls on violation; mirrors the API's
        # SourceVideoSpec + cross-validators) ---
        upload_cfg = (config or {}).get("upload") or {}
        if mode == MODE_V2V:
            err = check_v2v_context(context_frames, clip_frames[0], lang, config)
            if err is not None:
                yield _precheck_reject(err), "", None
                return
            if not src_video:
                yield _precheck_reject(L("v2v_msg_video_required", lang)), "", None
                return
            allowed_exts = upload_cfg.get("allowed_video_extensions") or _FALLBACK_VIDEO_EXTS
            ext = Path(str(src_video)).suffix.lower()
            if ext not in [e.lower() for e in allowed_exts]:
                yield _precheck_reject(L("v2v_msg_bad_extension", lang).format(exts=", ".join(allowed_exts))), "", None
                return
            max_mb = upload_cfg.get("max_video_size_mb")
            if max_mb is None:
                max_mb = _FALLBACK_MAX_VIDEO_MB
            try:
                size_mb = os.path.getsize(str(src_video)) / (1024 * 1024)
            except OSError:
                size_mb = 0.0
            if size_mb > max_mb:
                yield _precheck_reject(L("v2v_msg_too_large", lang).format(limit=max_mb)), "", None
                return
            # The frozen source tail occupies clip 0's head — a start image on
            # clip 1 would conflict (server 422); reject before any upload.
            if enabled[0][2]:
                yield _precheck_reject(L("v2v_msg_image_conflict", lang)), "", None
                return

        # --- A2V prechecks (zero API calls on violation) ---
        if mode == MODE_A2V:
            if not src_audio:
                yield _precheck_reject(L("a2v_msg_audio_required", lang)), "", None
                return
            allowed_audio = upload_cfg.get("allowed_audio_extensions") or _FALLBACK_AUDIO_EXTS
            ext = Path(str(src_audio)).suffix.lower()
            if ext not in [e.lower() for e in allowed_audio]:
                yield _precheck_reject(L("a2v_msg_bad_extension", lang).format(exts=", ".join(allowed_audio))), "", None
                return
            max_audio_mb = upload_cfg.get("max_audio_size_mb")
            if max_audio_mb is None:
                max_audio_mb = _FALLBACK_MAX_AUDIO_MB
            try:
                size_mb = os.path.getsize(str(src_audio)) / (1024 * 1024)
            except OSError:
                size_mb = 0.0
            if size_mb > max_audio_mb:
                yield _precheck_reject(L("a2v_msg_too_large", lang).format(limit=max_audio_mb)), "", None
                return

        # Total-timeline geometry: same arithmetic as the API validator. For V2V
        # the frozen-head geometry (incl. the stage-2 tile-fit invariant) is
        # forwarded so the precheck matches the server's compute_chain_layout.
        err = check_chain_total(
            clip_frames, fps, kv, lang,
            source_context_px=int(context_frames) if mode == MODE_V2V else None,
        )
        if err is not None:
            yield _precheck_reject(err), "", None
            return

        # --- prompt-embedded <lora:...> tokens from the SHARED prompt (chain
        # LoRA, ADDITIVE). Clip chaining now wires style/character IC-LoRAs into
        # GenerateChainRequest.loras; the strengths apply UNIFORMLY across the
        # whole chain (every clip, every stage — no per-clip strengths in v1).
        # Resolved + stripped here BEFORE any upload, so an unknown token name
        # aborts with zero upload/generate calls (mirrors the Generate tab's 1c
        # block). Only the SHARED prompt is scanned — per-clip prompts are left as
        # authored (out of scope). GET /loras runs ONLY when a token is present,
        # so a token-free prompt makes no extra call and the payload is
        # byte-identical. A reference-video CONTROL token (canny/pose/upscaler) is
        # left to the server's 422 LORA_CONTROL_UNSUPPORTED_IN_CHAIN (rendered via
        # format_api_error) — the chain has no reference video to drive it. ---
        send_prompt = prompt
        chain_loras: list[dict] = []
        if _LORA_TOKEN_RE.search(prompt or ""):
            try:
                lora_list = api.list_loras()
            except Exception as exc:
                yield _precheck_reject(L("lora_msg_list_failed", lang).format(err=exc)), "", None
                return
            known = [e.get("name") for e in (lora_list or [])
                     if isinstance(e, dict) and e.get("name")]
            send_prompt, chain_loras, lora_err = parse_prompt_loras(prompt, known, lang)
            if lora_err is not None:
                yield _precheck_reject(lora_err), "", None
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
            "prompt": send_prompt,
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
        # ADDITIVE chain LoRA: the SHARED prompt's <lora:...> style/character
        # adapters, applied uniformly across the chain. Key added ONLY when
        # non-empty, so a token-free chain stays byte-identical to before.
        if chain_loras:
            payload["loras"] = chain_loras
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
