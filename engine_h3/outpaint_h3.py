"""MiniMax H3 Outpainting: the source read, the fixed tokens and the self-check (台帳 §1-125).

設計正本 ``Docs/MINIMAX_H3_DESIGN.md`` §3.1（行ごとの連続値マスク）・§3.4（Retake の
音声の Strength）、計画 §2.3・§6.1。The geometry and the paste-back are the
torch-free ``engine_h3.outpaint_geometry``; this module holds the torch side,
which the single ``H3Pipeline.generate`` calls:

* :func:`read_outpaint_source`: the app's cut of the source video (exactly F
  frames at 24 fps) at its own size, and its audio by the Retake window's rule.
* :func:`outpaint_fix`: the source's latents (encoded alone at the 32-aligned
  box) placed on a zero canvas latent, with one mask value per token
  ``[T', gh, gw]`` (fixed 0 / generated 1); the audio as Retake's
  (``1 - audio_strength`` on what was read, 0 at strength 1.0).
* :func:`check_outpaint_fixed`: after the denoise the fixed tokens' latent
  cells (2 x 2 each) are the clean latents bit for bit, or the job fails.
"""

from __future__ import annotations

import torch

from engine_h3.constants import AUDIO_SAMPLE_RATE, FRAME_RATE

#: The video fixed rows' injection coefficient (the latents themselves, as the
#: chain and Retake: 裁定 84-2). The 0.999 control of the gate replaces this
#: module attribute; ``generate()`` reads it at call time.
OUTPAINT_INJECT_CLEAN = 1.0


def read_outpaint_source(path: str, num_frames: int, frame_rate: int = FRAME_RATE, audio_strength: float = 0.0,
                         *, log=None):
    """The cut source video, read on the CPU before any model: ``(frames_u8, wave, info)``.

    ``frames_u8``: uint8 ``[F, H_s, W_s, 3]`` at the file's own size
    (``source_tail.read_source_frames``). ``wave``: ``[2, N]`` at 32 kHz or
    None, by the Retake window's rule (``chain_h3._read_retake_window``):
    strength 0 reads no audio; below 1.0 whatever is read is used
    (``keep_samples=1``; a track without a single sample is generated freely,
    with one log line); at 1.0 the whole window's audio,
    ``round(F * 32000 / fps)`` samples, is required (the original waveform is
    muxed; a shorter one fails). A file without an audio track and a strength
    above 0 is refused (the app's 422 ``OUTPAINT_VIDEO_NO_AUDIO`` comes first).
    ``info``: ``source_size`` [W_s, H_s], ``frames_read``, ``source_had_audio``
    (the file has a track), ``required_samples`` and ``read_samples`` (0
    without a waveform).
    """
    from engine_h3.source_tail import has_audio_stream, read_source_frames  # noqa: PLC0415

    log = log or (lambda m: None)
    frames_n = int(num_frames)
    s = float(audio_strength)
    frames = read_source_frames(path, frames_n)
    samples = int(round(frames_n * AUDIO_SAMPLE_RATE / int(frame_rate)))
    had_audio = has_audio_stream(path)
    info = {
        "source_size": [int(frames.shape[2]), int(frames.shape[1])],
        "frames_read": int(frames.shape[0]),
        "source_had_audio": bool(had_audio),
        "required_samples": samples,
        "read_samples": 0,
    }
    if not had_audio:
        if s > 0.0:
            raise RuntimeError(
                f"Outpainting: audio_strength={s} には元動画の音声が必要ですが、"
                f"元動画に音声トラックがありません: {path}"
            )
        return frames, None, info
    if s == 0.0:
        return frames, None, info   # free audio: nothing is read or encoded
    from engine_h3.source_audio import load_source_waveform  # noqa: PLC0415

    if s == 1.0:
        wave, _wave_info = load_source_waveform(path, keep_samples=samples, read_samples=samples)
    else:
        try:
            wave, _wave_info = load_source_waveform(path, keep_samples=1, read_samples=samples)
        except ValueError as exc:
            log(f"outpaint: source audio not usable, generating audio freely ({exc})")
            return frames, None, info
    info["read_samples"] = int(wave.shape[-1])
    return frames, wave, info


def outpaint_fix(clean_bbox: torch.Tensor, clean_a: torch.Tensor | None, geom, s: float, t_lat: int, a_lat: int):
    """``(video_fix, audio_fix)`` for ``H3Pipeline._denoise``.

    ``clean_bbox``: the 32-aligned box's encode ``[1, C, T', aligned_h/16,
    aligned_w/16]`` (CPU float32, ``_encode_source_tail``). It is copied onto
    a zero canvas latent ``[1, C, T', canvas_h/16, canvas_w/16]`` at
    ``geom.latent_offset``; the generated tokens' clean is never used (value 1),
    so 0 is fine there. The values: ``outpaint_geometry.token_values``,
    ``[T', gh, gw]`` float32 (fixed 0 / generated 1).
    ``clean_a``: the source audio's encode ``[2, 32, te_a]`` or None (no audio
    fix: generated freely). As ``chain_h3.retake_fix``: the clean is ``[2, 32,
    A]``, zero past ``te_a``; the values are ``1 - s`` on ``[0, te_a)`` (0 at
    ``s`` = 1.0) and 1 on ``[te_a, A)``. CPU float32.
    """
    from engine_h3.outpaint_geometry import LATENT_PX, token_values  # noqa: PLC0415

    t_lat, a_lat = int(t_lat), int(a_lat)
    ah, aw = geom.aligned_h // LATENT_PX, geom.aligned_w // LATENT_PX
    lh, lw = geom.canvas_h // LATENT_PX, geom.canvas_w // LATENT_PX
    if clean_bbox.dim() != 5 or clean_bbox.shape[0] != 1 or tuple(clean_bbox.shape[2:]) != (t_lat, ah, aw):
        raise RuntimeError(
            f"outpaint: source latent {list(clean_bbox.shape)}, expected [1, C, {t_lat}, {ah}, {aw}]"
        )
    oy, ox = geom.latent_offset
    clean_v = torch.zeros((1, int(clean_bbox.shape[1]), t_lat, lh, lw), dtype=torch.float32)
    clean_v[:, :, :, oy:oy + ah, ox:ox + aw] = clean_bbox.to(torch.float32)
    values = torch.from_numpy(token_values(geom, t_lat))
    video_fix = (clean_v, values)
    if clean_a is None:
        return video_fix, None
    te_a = int(clean_a.shape[2])
    if te_a > a_lat:
        raise RuntimeError(f"outpaint: source audio latents {te_a}, more than the job's {a_lat}")
    full_a = torch.zeros(tuple(clean_a.shape[:2]) + (a_lat,), dtype=torch.float32)
    full_a[:, :, :te_a] = clean_a.to(torch.float32)
    per_a = torch.ones(a_lat, dtype=torch.float32)
    per_a[:te_a] = 0.0 if float(s) == 1.0 else 1.0 - float(s)
    return video_fix, (full_a, per_a)


def fixed_cells(values: torch.Tensor) -> torch.Tensor:
    """Token values ``[T', gh, gw]`` -> bool ``[T', 2*gh, 2*gw]``: the fixed (value 0) tokens' latent cells."""
    fixed = values == 0
    return fixed.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)


def check_outpaint_fixed(video: torch.Tensor, video_fix, audio: torch.Tensor, audio_fix, s: float, te_a: int) -> None:
    """Self-check: the fixed tokens' latent cells come back as the source's clean
    latents bit for bit (``denoise_mask.finish`` puts the m = 0 rows back); the
    audio ``[0, te_a)`` too at strength 1.0 (below it those rows are not fixed).
    The clean side is moved to the result's device (the denoise returns its
    latents on the pipeline's device, the clean is on the CPU). A mismatch
    fails the job (as ``chain_h3._check_retake_fixed``)."""
    clean_v, values = video_fix
    cells = fixed_cells(values).to(video.device)
    clean_v = clean_v.to(device=video.device, dtype=video.dtype)
    ok = bool(torch.equal(video[:, :, cells], clean_v[:, :, cells]))
    if ok and audio_fix is not None and float(s) == 1.0 and int(te_a) > 0:
        clean_a = audio_fix[0].to(device=audio.device, dtype=audio.dtype)
        ok = bool(torch.equal(audio[:, :, :te_a], clean_a[:, :, :te_a]))
    if not ok:
        raise RuntimeError("Outpainting: 固定したトークン（元の映像の領域）が元の潜在と一致しません（自己検査）")
