"""MiniMax H3 chain: several clips joined by carrying latents over (台帳 §1-106).

A clip may carry its own Ref2VA references (台帳 §1-109): they go through the
single job's reference stage for that clip only; a clip without any runs as before.

Video-to-video and a start image (台帳 §1-121): an input video's last 22
frames (cut by the app, ``H3ChainSpec.source``) are encoded and become clip
0's fixed head exactly as a previous clip's tail would (裁定 84-5 keeps the
head's source replaceable); the decoded timeline then loses those 22 frames
and their audio, and the untrimmed audio is kept as the JOIN's handle wav.
Clip 0 may instead carry key frames (``H3ChainClipSpec.images``), handled as
in a single I2V job. The two are exclusive.

Retake (台帳 §1-123): one clip that IS a window of an input video
(``H3ChainSpec.retake``, cut by the app). The whole window is encoded; its
first and last 22 frames are fixed rows (video and audio), the middle is
redrawn, its audio held toward the original by ``audio_strength`` (mask value
``1 - s``), and the whole window is decoded (strength 1.0 muxes the original
waveform). Exclusive with ``source``, key frames and references.

The mechanism is Docs/MINIMAX_H3_DESIGN.md §3.3 (Retake: §3.4); the geometry is
``h3_chain_math`` at the repository root, shared with the app. Each clip's
fixed head is self-checked against the previous tail with ``torch.equal``.

``keep_resident`` means what it means for one job: kept after a successful
job that asks for it, released otherwise (and after any failure).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from engine_h3 import pipeline_h3 as P
from h3_chain_math import (
    H3_AUDIO_SAMPLES_PER_FRAME,
    H3_CHAIN_OVERLAP_FRAMES,
    H3ChainLayout,
    compute_h3_chain_layout,
)

#: The video fixed rows' injection coefficient (裁定 84-2: the latents themselves).
CHAIN_INJECT_CLEAN = 1.0


@dataclass(frozen=True)
class H3ChainClipSpec:
    """One clip: its whole prompt, its frame count, its own Ref2VA references
    (台帳 §1-109; empty = none) and its key frames (clip 0 only, 台帳 §1-121;
    empty = none)."""

    prompt: str
    num_frames: int
    references: tuple[P.H3ReferenceInput, ...] = ()
    images: tuple[P.H3ImageInput, ...] = ()


@dataclass(frozen=True)
class H3ChainSource:
    """The input video of a video-to-video chain (台帳 §1-121): the app's cut of
    its last ``context_frames`` frames at 24 fps (with its audio, if any)."""

    path: str
    context_frames: int


@dataclass(frozen=True)
class H3ChainRetake:
    """The window of a retake (台帳 §1-123): the app's cut of the input video's
    window at 24 fps (with its audio, if any), exactly the one clip's frames.
    ``audio_strength`` (0..1): how much the free middle's audio keeps the
    original (mask value ``1 - s``); 1.0 also muxes the original waveform."""

    path: str
    audio_strength: float


@dataclass
class H3ChainSpec:
    """``base``: the settings every clip shares; its ``prompt`` / ``num_frames``
    / ``seed`` are replaced per clip and its ``output_path`` is the chain's mp4.
    ``base`` carries no images, source audio or references (a clip's key frames
    and references are its own, ``H3ChainClipSpec``). ``source``: the input
    video whose tail is clip 0's fixed head (None = the chain starts from text,
    or from clip 0's key frames). ``retake``: the window whose head and tail
    are fixed and whose middle is redrawn (None = no retake; one clip only)."""

    base: P.H3GenerateRequest
    clips: tuple[H3ChainClipSpec, ...]
    seed: int
    source: H3ChainSource | None = None
    retake: H3ChainRetake | None = None


def _sha_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def latents_sha256(t: torch.Tensor) -> str:
    """The latents' fingerprint: CPU float32 contiguous bytes (c_chain.py's ``_sha_tensor``)."""
    return hashlib.sha256(t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()).hexdigest()


def carry_fix(prev_video: torch.Tensor, prev_audio: torch.Tensor | None, t_lat: int, a_lat: int, k: int, ka: int):
    """``(video_fix, audio_fix)`` for a clip of ``t_lat`` video / ``a_lat`` audio latents.

    The head is ``prev_video[:, :, -k:]`` / ``prev_audio[:, :, -ka:]`` at mask
    value 0, the rest generated (value 1; its clean part is zero and unused).
    Where the tail comes from is the caller's: the previous clip's video
    latents and the concatenated audio, or the input video's encoded tail
    (裁定 84-5 keeps the source replaceable). ``prev_audio`` None (an input
    video without audio) gives ``audio_fix`` None: the audio is generated
    freely. CPU float32.
    """
    shape_v = tuple(prev_video.shape[:2]) + (int(t_lat),) + tuple(prev_video.shape[3:])
    clean_v = torch.zeros(shape_v, dtype=torch.float32)
    if k > 0:
        clean_v[:, :, :k] = prev_video[:, :, -k:].to(torch.float32)
    per_v = [0.0] * k + [1.0] * (int(t_lat) - k)
    if prev_audio is None:
        return (clean_v, per_v), None
    shape_a = tuple(prev_audio.shape[:2]) + (int(a_lat),)
    clean_a = torch.zeros(shape_a, dtype=torch.float32)
    if ka > 0:
        clean_a[:, :, :ka] = prev_audio[:, :, -ka:].to(torch.float32)
    per_a = [0.0] * ka + [1.0] * (int(a_lat) - ka)
    return (clean_v, per_v), (clean_a, per_a)


def retake_fix(clean_v: torch.Tensor, clean_a: torch.Tensor | None, layout: H3ChainLayout, s: float):
    """``(video_fix, audio_fix)`` for a retake window (台帳 §1-123).

    ``clean_v``: the window's encoded video ``[1, 24, T', lh, lw]``; its first
    and last ``retake_n_head_v`` / ``retake_n_tail_v`` latents are fixed (value
    0), the middle generated (1). ``clean_a``: the window's encoded audio
    ``[2, 32, te_a]`` (``te_a`` = what the encoder returned, at most the
    clip's ``A``), or None (no audio: ``audio_fix`` None, generated freely).
    The audio's clean is ``[2, 32, A]``, zero past ``te_a``; its values are 0
    on the head ``[0, h_a)``, ``1 - s`` on the middle ``[h_a, ts_a)``, 0 on the
    tail ``[ts_a, te_a)`` and 1 on ``[te_a, A)`` (the latents the encoder did
    not return: always free). Each range is cut at ``te_a``, so a short audio
    freezes only what was read. CPU float32.
    """
    t_lat = int(layout.clip_latent_frames[0])
    a_lat = int(layout.clip_audio_latents[0])
    head_v, tail_v = int(layout.retake_n_head_v), int(layout.retake_n_tail_v)
    per_v = [0.0] * head_v + [1.0] * (t_lat - head_v - tail_v) + [0.0] * tail_v
    video_fix = (clean_v.to(torch.float32).contiguous(), per_v)
    if clean_a is None:
        return video_fix, None
    te_a = int(clean_a.shape[2])
    full_a = torch.zeros(tuple(clean_a.shape[:2]) + (a_lat,), dtype=torch.float32)
    full_a[:, :, :te_a] = clean_a.to(torch.float32)
    h_a = min(int(layout.retake_n_head_a), te_a)
    ts_a = min(int(layout.retake_audio_tail_start), te_a)
    middle = 1.0 - float(s)
    per_a = [0.0] * h_a + [middle] * (ts_a - h_a) + [0.0] * (te_a - ts_a) + [1.0] * (a_lat - te_a)
    return video_fix, (full_a, per_a)


def clip_requests(spec: H3ChainSpec) -> list[P.H3GenerateRequest]:
    """Clip i's request: ``base`` with its prompt, frames, references, key frames,
    ``seed + i`` and ``keep_resident=True``."""
    return [
        dataclasses.replace(
            spec.base, prompt=clip.prompt, num_frames=int(clip.num_frames), seed=int(spec.seed) + i,
            keep_resident=True, references=tuple(clip.references), images=tuple(clip.images),
        )
        for i, clip in enumerate(spec.clips)
    ]


def _check_spec(spec: H3ChainSpec) -> None:
    """The chain's own shape, refused in Japanese before anything is read or loaded."""
    base = spec.base
    if base.images or base.source_audio is not None or base.references:
        raise ValueError(
            "連結の共通設定（base）にはキーフレーム・音声・参照は付けられません"
            "（冒頭画像はクリップ 0 に、参照はクリップごとに付けます）"
        )
    for i, clip in enumerate(spec.clips):
        if i > 0 and clip.images:
            raise ValueError(f"連結のキーフレーム（冒頭画像）はクリップ 0 にだけ付けられます（clips[{i}]）")
    retake = spec.retake
    if retake is not None:
        if spec.source is not None:
            raise ValueError("Retake と入力動画の続き（V2V）は同時には使えません")
        if len(spec.clips) != 1:
            raise ValueError(f"Retake のクリップは 1 本です（{len(spec.clips)} 本）")
        if spec.clips[0].images:
            raise ValueError("Retake とクリップ 0 のキーフレームは同時には使えません")
        if spec.clips[0].references:
            raise ValueError("Retake と参照（Ref2VA）は同時には使えません")
        s = retake.audio_strength
        if isinstance(s, bool) or not isinstance(s, (int, float)) or not (math.isfinite(s) and 0.0 <= s <= 1.0):
            raise ValueError(f"Retake の audio_strength は 0〜1 の数値で渡してください（{s!r}）")
    source = spec.source
    if source is None:
        return
    if spec.clips and spec.clips[0].images:
        raise ValueError("入力動画の続き（V2V）とクリップ 0 のキーフレームは同時には使えません")
    if int(source.context_frames) != H3_CHAIN_OVERLAP_FRAMES:
        raise ValueError(
            f"入力動画から引き継ぐフレーム数は {H3_CHAIN_OVERLAP_FRAMES} 固定です（{source.context_frames}）"
        )


def _load_key_frames(req: P.H3GenerateRequest):
    """A clip's key frames as a single I2V job reads them (``H3Pipeline.generate``):
    ``(canvases, frame indices, source sizes)``, in ascending frame order."""
    images = sorted(req.images or (), key=lambda img: int(img.frame_idx))
    cond_frame_idx = [int(img.frame_idx) for img in images]
    canvases: list = []
    source_size: list[list[int]] = []
    if images:
        from engine_h3.image_cond import load_canvas_image  # noqa: PLC0415

        for img in images:
            canvas, size = load_canvas_image(img.path, int(req.width), int(req.height))
            canvases.append(canvas)
            source_size.append([int(size[0]), int(size[1])])
    return canvases, cond_frame_idx, source_size


def _read_retake_window(retake: H3ChainRetake, base: P.H3GenerateRequest, layout: H3ChainLayout, log):
    """The retake window, read on the CPU before any model: ``(frames, wave, info)``.

    ``frames``: exactly the window's F frames on the canvas. ``wave``: its
    audio, ``[2, N]`` at 32 kHz with ``N`` at most ``round(F * 4000/3)``, or
    None without an audio track. Below strength 1.0 whatever is read is used
    (``keep_samples=1``; a track without a single sample is generated freely,
    as for video-to-video); at 1.0 the original waveform is muxed, so the whole
    window's audio is required (a shorter one fails). A file without audio and
    a strength above 0 is refused (the app's preflight should have).
    """
    from engine_h3.source_tail import has_audio_stream, read_source_tail  # noqa: PLC0415

    window = int(layout.retake_window_px)
    s = float(retake.audio_strength)
    frames, info = read_source_tail(retake.path, int(base.width), int(base.height), window)
    if not has_audio_stream(retake.path):
        if s > 0.0:
            raise RuntimeError(
                f"Retake: audio_strength={s} には窓の音声が必要ですが、"
                f"入力動画に音声トラックがありません: {retake.path}"
            )
        return frames, None, info
    from engine_h3.source_audio import load_source_waveform  # noqa: PLC0415

    samples = int(round(window * H3_AUDIO_SAMPLES_PER_FRAME))
    if s == 1.0:
        wave, _wave_info = load_source_waveform(retake.path, keep_samples=samples, read_samples=samples)
        return frames, wave, info
    try:
        wave, _wave_info = load_source_waveform(retake.path, keep_samples=1, read_samples=samples)
    except ValueError as exc:
        log(f"chain: retake window audio not usable, generating audio freely ({exc})")
        wave = None
    return frames, wave, info


def _check_retake_fixed(video, audio, video_fix, audio_fix, layout: H3ChainLayout, te_a: int) -> None:
    """Self-check: the window's fixed head and tail come back as its clean latents
    bit for bit (denoise_mask.finish puts the m = 0 rows back); a mismatch fails
    the job (no record key: passing is not stopping). ``te_a``: the encoded
    audio latents (the audio ranges are cut there, as in :func:`retake_fix`)."""
    clean_v = video_fix[0]
    head_v, tail_v = int(layout.retake_n_head_v), int(layout.retake_n_tail_v)
    ok = bool(torch.equal(video[:, :, :head_v], clean_v[:, :, :head_v])) and bool(
        torch.equal(video[:, :, -tail_v:], clean_v[:, :, -tail_v:])
    )
    if ok and audio_fix is not None:
        clean_a = audio_fix[0]
        h_a = min(int(layout.retake_n_head_a), te_a)
        ts_a = min(int(layout.retake_audio_tail_start), te_a)
        ok = bool(torch.equal(audio[:, :, :h_a], clean_a[:, :, :h_a]))
        if ok and te_a > ts_a:
            ok = bool(torch.equal(audio[:, :, ts_a:te_a], clean_a[:, :, ts_a:te_a]))
    if not ok:
        raise RuntimeError("chain: the retake window's fixed head / tail is not the input video's")


def run_chain_h3(
    pipe: P.H3Pipeline,
    spec: H3ChainSpec,
    progress: Callable[..., None],
    log: Callable[[str], None],
) -> dict:
    """Run the chain on a loaded ``pipe`` and write ``spec.base.output_path``.

    ``progress(stage, index, total, *, outer_index=None, outer_total=None)``:
    the clip's stages carry ``outer_index`` = the clip (0-based) and
    ``outer_total`` = the clip count; the decode's progress goes through the
    pipeline's original ``_progress``, with no outer position. Returns the
    single job's ``done`` fields (``num_frames`` = the timeline's) plus ``chain``.

    A clip with references runs the single job's reference steps (``generate``
    without key frames): read before its text encoder, encoded after it in a
    ``reference_encode`` phase by a video encoder built for this clip only, and
    handed to its denoise next to the fixed head.

    With ``spec.source`` (video-to-video) the input video's tail is read on the
    CPU before anything else, encoded in clip 0's ``source_encode`` phase (the
    ``encode`` stage's 4th step) and becomes clip 0's fixed head; the decode
    trims it off and writes ``<stem>_audio_handle.wav``, and ``chain`` gains
    ``v2v``. Clip 0's key frames (``images``) run the single I2V job's steps:
    read before its text encoder, the first / last frame to the vision tower
    (none with references; 裁定 59), encoded in an ``image_encode`` phase.

    With ``spec.retake`` the window is read on the CPU before anything else,
    encoded in clip 0's ``source_encode`` phase, fixed at its head and tail
    (:func:`retake_fix`, self-checked bit for bit after the denoise) and
    decoded whole (``trim_head=0``); ``chain`` gains ``retake``.

    With ``spec.base.negative_prompt`` (VSF, 台帳 §1-127) clip 0's text encoder
    also encodes the negative prompt (one more ``encode`` step, before the key
    frames' or the input video's), every clip's denoise uses those hidden
    states, and each clip record gains ``vsf`` (the single job's ``h3.vsf``).
    """
    if pipe._dit_cfg is None:
        raise RuntimeError("generate before load")
    base = spec.base
    keep = bool(base.keep_resident)
    started = time.perf_counter()
    cuda = pipe.device.type == "cuda"
    ok = False
    lora_report: list[dict] = []
    clip_phases: list[P._Phases] = []
    clip_records: list[dict] = []
    decode_phases = P._Phases(cuda=cuda)
    prefetch_used = "off"
    size_bytes = 0
    layout: H3ChainLayout | None = None
    source = spec.source
    v2v_runtime: dict = {}
    retake = spec.retake
    retake_runtime: dict = {}
    try:
        _check_spec(spec)
        layout = compute_h3_chain_layout(
            [c.num_frames for c in spec.clips], source_video=source is not None, retake=retake is not None,
        )
        requests = clip_requests(spec)
        for req in requests:
            P.validate_request(req)
        # VSF (台帳 §1-127): every clip inherits the base's negative prompt; it is
        # encoded once, on clip 0, and its hidden states are reused by every clip.
        has_neg = bool(base.negative_prompt.strip())
        negative_hidden = None
        n = len(requests)
        k = layout.overlap_latents
        video_cat = audio_cat = prev_video = None
        src_frames = src_wave = src_info = None
        if source is not None:
            # The input video's tail, read on the CPU before any model (a bad file
            # costs no text-encoder read); its audio only when it has some.
            from engine_h3.source_tail import has_audio_stream, read_source_tail  # noqa: PLC0415

            src_frames, src_info = read_source_tail(
                source.path, int(base.width), int(base.height), int(layout.source_context_px),
            )
            if has_audio_stream(source.path):
                from engine_h3.source_audio import load_source_waveform  # noqa: PLC0415

                try:
                    src_wave, _wave_info = load_source_waveform(
                        source.path, keep_samples=1, read_samples=int(layout.n_ctx_a) * P.AUDIO_HOP,
                    )
                except ValueError as exc:
                    # An audio track that ends before the context window has no samples
                    # to freeze: generate the audio freely, as for a file without audio.
                    src_wave = None
                    log(f"chain: input video audio not usable, generating audio freely ({exc})")
            v2v_runtime["source_had_audio"] = src_wave is not None
        retake_fixed = None
        if retake is not None:
            # The retake window, read on the CPU before any model, as the V2V tail.
            src_frames, src_wave, src_info = _read_retake_window(retake, base, layout, log)
            retake_runtime["source_had_audio"] = src_wave is not None
            retake_runtime["muxed_original_waveform"] = float(retake.audio_strength) == 1.0 and src_wave is not None
        for i, req in enumerate(requests):
            has_neg_i = has_neg and i == 0   # the negative's text encoder forward: clip 0 only
            from_source = i == 0 and source is not None
            from_retake = i == 0 and retake is not None
            ka = layout.audio_overlap_latents[i - 1] if i else 0   # clip 0 with a source: set by its encode
            P._free_cuda()
            phases = P._Phases(cuda=cuda)
            clip_phases.append(phases)
            video_fix = audio_fix = None
            if i:
                video_fix, audio_fix = carry_fix(
                    prev_video, audio_cat, layout.clip_latent_frames[i], layout.clip_audio_latents[i], k, ka,
                )
            log(f"chain: clip {i + 1}/{n} seed={req.seed} frames={req.num_frames} "
                + (f"fixed head k={k} ka={ka}" if i else
                   "fixed head from the input video" if from_source else
                   f"retake: fixed head and tail from the input video's window "
                   f"(audio_strength={float(retake.audio_strength)})" if from_retake else "no fixed head"))
            original_progress = pipe._progress
            pipe._progress = functools.partial(progress, outer_index=i, outer_total=n)
            try:
                with torch.no_grad():
                    # Key frames (clip 0 only, 台帳 §1-121): read on the CPU before the
                    # text encoder, as in a single I2V job. Only the first and the last
                    # frame go to the vision tower (裁定 59); none with references.
                    canvases, cond_frame_idx, source_size = _load_key_frames(req)
                    last = int(req.num_frames) - 1
                    vision_images = [c for m, c in zip(cond_frame_idx, canvases) if m in (0, last)]
                    # Ref2VA (台帳 §1-109): the clip's references are read on the CPU
                    # before its text encoder, as in a single job; none = as before.
                    media = pipe._read_references(req)
                    if media:
                        vision_images = []
                    # The key frames' VAE step (4th of the encode stage, 5th after clip 0's negative) also counts the
                    # input video's encode (V2V or a retake window); they never meet in one clip.
                    has_cond = bool(canvases) or from_source or from_retake
                    encode_total = P.encode_steps(
                        has_cond=has_cond, has_audio=False, has_refs=bool(media), has_neg=has_neg_i,
                    )
                    # Only clip 0 of a VSF chain passes the key: every other call is the same as before.
                    neg_kw = {"negative_prompt": req.negative_prompt} if has_neg_i else {}
                    with phases.phase("encode") as entry:
                        encoded = pipe._encode(
                            req.prompt, phases, images=vision_images, has_cond=has_cond,
                            refs=pipe._te_references(media), **neg_kw,
                        )
                        if has_neg_i:
                            hidden, tags, negative_hidden = encoded
                        else:
                            hidden, tags = encoded
                        del encoded
                        entry["text_tokens"] = int(hidden.shape[0])
                        if has_neg_i:
                            entry["negative_tokens"] = int(negative_hidden.shape[0])
                    cond_clean = None
                    if canvases:
                        with phases.phase("image_encode") as entry:
                            cond_clean = pipe._encode_image(canvases, entry, 4 + int(has_neg_i), encode_total)
                            entry["vision_tokens"] = int(phases.data["encode"].get("te_read", {}).get("vision_tokens", 0))
                            entry["canvas"] = [int(req.width), int(req.height)]
                            entry["source_size"] = source_size
                    del canvases, vision_images
                    if from_source:
                        with phases.phase("source_encode") as entry:
                            entry["source_size"] = src_info["source_size"]
                            src_video, src_audio = pipe._encode_source_tail(
                                src_frames, src_wave, entry, 4 + int(has_neg_i), encode_total,
                            )
                            ka = int(entry["freeze_ka"])
                        src_frames = src_wave = None
                        v2v_runtime["freeze_ka"] = ka
                        # The input video's tail is clip 0's fixed head, as a previous clip's.
                        prev_video, audio_cat = src_video, src_audio
                        video_fix, audio_fix = carry_fix(
                            prev_video, audio_cat, layout.clip_latent_frames[0], layout.clip_audio_latents[0], k, ka,
                        )
                        del src_video, src_audio
                    if from_retake:
                        with phases.phase("source_encode") as entry:
                            entry["source_size"] = src_info["source_size"]
                            src_video, src_audio = pipe._encode_source_tail(
                                src_frames, src_wave, entry, 4 + int(has_neg_i), encode_total,
                            )
                        src_frames = None
                        if not retake_runtime["muxed_original_waveform"]:
                            src_wave = None   # kept only to be muxed at strength 1.0
                        te_a = 0 if src_audio is None else int(src_audio.shape[2])
                        video_fix, audio_fix = retake_fix(src_video, src_audio, layout, float(retake.audio_strength))
                        retake_fixed = (video_fix, audio_fix, te_a)
                        retake_runtime["te_a"] = te_a
                        retake_runtime["audio_frozen"] = audio_fix is not None and any(v == 0.0 for v in audio_fix[1])
                        del src_video, src_audio
                    ref_rows = None
                    ref_counts = None
                    if media:
                        with phases.phase("reference_encode") as entry:
                            # This clip's own video encoder (only for a picture / video
                            # reference), dropped once its references are encoded.
                            enc = pipe._build_video_encoder() if any(m["kind"] != "audio" for m in media) else None
                            te_refs = phases.data["encode"].get("te_read", {}).get("references", [])
                            ref_rows = pipe._encode_references(media, entry, encode_total, encode_total, enc, te_refs)
                            enc = None
                            P._free_cuda()
                        media = None   # encoded: drop the read media before the denoise, as a single job
                        counts = {kind: sum(1 for s, _r in ref_rows if s.kind == kind) for kind in P.REFERENCE_KINDS}
                        ref_counts = {"pictures": counts["picture"], "videos": counts["video"], "audios": counts["audio"]}
                    # The clip's own VSF report (the single job's ``h3.vsf``), only for a VSF chain.
                    clip_vsf: dict | None = {} if has_neg else None
                    with phases.phase("denoise") as entry:
                        video, audio, prefetch_used = pipe._denoise(
                            req, hidden, tags, phases, entry,
                            cond_clean=cond_clean, cond_frame_idx=cond_frame_idx, keep=True,
                            lora_report=lora_report if i == 0 else None,
                            video_fix=video_fix, audio_fix=audio_fix, refs=ref_rows,
                            video_inject_clean=CHAIN_INJECT_CLEAN,
                            negative_hidden=negative_hidden if has_neg else None,
                            vsf_scale=float(req.vsf_scale), vsf_report=clip_vsf,
                        )
                    video, audio = video.cpu(), audio.cpu()
            finally:
                pipe._progress = original_progress
            del hidden, tags, video_fix, audio_fix, media, ref_rows, cond_clean
            P._free_cuda()
            if int(video.shape[2]) != layout.clip_latent_frames[i] or int(audio.shape[2]) != layout.clip_audio_latents[i]:
                raise RuntimeError(
                    f"chain: clip {i + 1} latents {list(video.shape)} / {list(audio.shape)}, expected "
                    f"{layout.clip_latent_frames[i]} video / {layout.clip_audio_latents[i]} audio latents"
                )
            head_equal = None
            if i or from_source:
                # Self-check: the fixed head must come back as the previous tail (or the
                # input video's) bit for bit (denoise_mask.finish puts the m = 0 rows
                # back to their clean latents).
                head_equal = bool(torch.equal(video[:, :, :k], prev_video[:, :, -k:])) and (
                    ka == 0 or bool(torch.equal(audio[:, :, :ka], audio_cat[:, :, -ka:]))
                )
                if not head_equal:
                    tail = "the input video's tail" if from_source else "the previous clip's tail"
                    raise RuntimeError(f"chain: clip {i + 1}'s fixed head is not {tail}")
            if from_retake:
                _check_retake_fixed(video, audio, retake_fixed[0], retake_fixed[1], layout, retake_fixed[2])
                retake_fixed = None
            seconds = {"encode": phases.data.get("encode", {}).get("seconds")}
            for name in ("image_encode", "source_encode"):
                if name in phases.data:   # only for a clip with key frames / the input video
                    seconds[name] = phases.data[name].get("seconds")
            if ref_counts is not None:
                seconds["reference_encode"] = phases.data.get("reference_encode", {}).get("seconds")
            seconds["denoise"] = phases.data.get("denoise", {}).get("seconds")
            record = {
                "prompt_sha256": _sha_text(req.prompt),
                "seconds": seconds,
                "latents_sha256": {"video": latents_sha256(video), "audio": latents_sha256(audio)},
                "head_equals_prev_tail": head_equal,
            }
            if ref_counts is not None:
                record["references"] = ref_counts   # only for a clip with references
            if has_neg:
                record["vsf"] = clip_vsf   # only for a VSF chain: the single job's h3.vsf
            clip_records.append(record)
            if i == 0:
                video_cat, audio_cat = video, audio   # with a source: its head is already in it
            else:
                video_cat = torch.cat([video_cat, video[:, :, k:]], dim=2)
                audio_cat = torch.cat([audio_cat, audio[:, :, ka:]], dim=2)
            prev_video = video
            del video, audio
        if int(video_cat.shape[2]) != layout.total_latent_frames or int(audio_cat.shape[2]) != layout.total_audio_latents:
            raise RuntimeError(
                f"chain: concatenated latents {list(video_cat.shape)} / {list(audio_cat.shape)}, expected "
                f"{layout.total_latent_frames} video / {layout.total_audio_latents} audio latents"
            )
        del prev_video, negative_hidden   # the negative's hidden states, reused by every clip
        if not keep:
            # As a single job without the key: its denoise drops the DiT before
            # the decode; every clip here kept it (keep=True), so let it go now.
            pipe.release_resident()
        P._free_cuda()
        out_req = dataclasses.replace(base, num_frames=layout.total_frames, output_path=base.output_path)
        with torch.no_grad():
            with decode_phases.phase("decode") as entry:
                # The decoders compute on the latents' device: hand them over on the pipeline's.
                if retake is not None:
                    # A retake: the whole window, no trim, no audio handle or fade; at
                    # strength 1.0 the original waveform goes into the mp4.
                    size_bytes = pipe._decode(
                        out_req, video_cat.to(pipe.device), audio_cat.to(pipe.device), entry, keep=keep,
                        source_wave=src_wave if retake_runtime["muxed_original_waveform"] else None, trim_head=0,
                    )
                    src_wave = None
                elif source is None:
                    size_bytes = pipe._decode(
                        out_req, video_cat.to(pipe.device), audio_cat.to(pipe.device), entry, keep=keep,
                    )
                else:
                    # Video-to-video: the input video's context frames are cut off the
                    # mp4, and the untrimmed audio is kept beside it for the JOIN.
                    out = Path(str(base.output_path))
                    size_bytes = pipe._decode(
                        out_req, video_cat.to(pipe.device), audio_cat.to(pipe.device), entry, keep=keep,
                        trim_head=int(layout.trim_px), handle_path=str(out.with_name(out.stem + "_audio_handle.wav")),
                    )
        del video_cat, audio_cat
        ok = True
    finally:
        if not (ok and keep):
            pipe.release_resident()
        P._free_cuda()
    mem = P._mem_gib()
    all_phases = clip_phases + [decode_phases]
    h3 = {
        "phases": {"clips": [p.data for p in clip_phases], "decode": decode_phases.data.get("decode", {})},
        "quant_backends": P._backend_summary(pipe.quant_backends),
    }
    if base.loras:
        h3["loras"] = lora_report   # clip 0's: every clip attaches the same files
    chain = layout.to_dict()
    v2v_layout = chain.pop("v2v", None)
    retake_layout = chain.pop("retake", None)
    chain.update({
        "seeds": [int(spec.seed) + i for i in range(len(spec.clips))],
        "clips": clip_records,
        "h3": {"overlap_frames": layout.overlap_frames, "inject_clean": CHAIN_INJECT_CLEAN, "decode": "once"},
    })
    if source is not None:
        # Only with an input video, last: a plain chain's block is unchanged. The
        # layout's six keys plus the run's, the LTX chain's sixteen.
        dec = decode_phases.data.get("decode", {})
        freeze_ka = int(v2v_runtime["freeze_ka"])
        chain["v2v"] = {
            **v2v_layout,
            "context_frames": int(source.context_frames),
            "freeze_ka": freeze_ka,
            "trimmed_px": int(dec["trimmed_frames"]),
            "trimmed_audio_samples": int(dec["trimmed_audio_samples"]),
            "audio_fade_in_samples": int(dec["audio_fade_in_samples"]),
            "source_had_audio": bool(v2v_runtime["source_had_audio"]),
            "audio_head_frozen": freeze_ka > 0,
            "decoded_frames_px": int(dec["decoded_frames"]),
            "audio_handle_filename": dec["audio_handle"],
            "handle_context_seconds": round(int(layout.trim_px) / int(base.frame_rate), 6),
        }
    if retake is not None:
        # Only for a retake, last: the layout's eight keys (n_tail_a the run's,
        # from the encoded audio) plus the run's, the LTX retake's names with
        # audio_strength in place of regenerate_audio.
        dec = decode_phases.data.get("decode", {})
        te_a = int(retake_runtime["te_a"])
        chain["retake"] = {
            **retake_layout,
            "n_tail_a": max(0, te_a - int(layout.retake_audio_tail_start)),
            "audio_strength": float(retake.audio_strength),
            "source_had_audio": bool(retake_runtime["source_had_audio"]),
            "audio_frozen": bool(retake_runtime["audio_frozen"]),
            "muxed_original_waveform": bool(retake_runtime["muxed_original_waveform"]),
            "decoded_frames_px": int(dec["frames"]),
        }
    return {
        "seed_used": int(spec.seed),
        "sampler": pipe.sampler,
        "num_inference_steps": int(base.num_inference_steps),
        "shift_video": float(base.shift_video),
        "shift_audio": float(base.shift_audio),
        "peak_vram_mb": max(p.peak_alloc_mib for p in all_phases) if cuda else None,
        "peak_vram_reserved_mb": max(p.peak_reserved_mib for p in all_phases) if cuda else None,
        "rss_peak_gib": mem.get("rss_peak"),
        "commit_gib": mem.get("commit"),
        "commit_peak_gib": mem.get("commit_peak"),
        "seconds": round(time.perf_counter() - started, 2),
        "num_frames": int(layout.total_frames),
        "frame_rate": int(base.frame_rate),
        "width": int(base.width),
        "height": int(base.height),
        "size_bytes": int(size_bytes),
        "attention_used": pipe._attention_for(base),
        "block_swap_prefetch_used": prefetch_used,
        "keep_resident_used": "on" if keep else "off",
        "vram16_mode_used": "on" if base.vram16_mode else "off",
        "images_used": sum(len(c.images) for c in spec.clips),
        "h3": h3,
        "chain": chain,
    }
