"""Write the H3 result as an mp4: H.264 (libx264, crf 19, yuv420p, BT.709) + AAC stereo.

Why its own writer
------------------
The LTX engines write mp4s through ``ltx_pipelines.utils.media_io`` /
``ltx_core.color``; neither package exists in the H3 venv, so this is a small
PyAV writer that follows the same recipe (read, not imported):

* Video: the RGB -> Y'CbCr conversion is done HERE with the BT.709 matrix and
  limited ("MPEG") range, and the planes are handed to PyAV as ``yuv420p``.
  Only tagging the stream as BT.709 is not enough: if PyAV/swscale did the
  conversion it would use its BT.601 default and the tags would lie (plan
  review m-4). The stream is still tagged (colorspace, primaries, transfer,
  range = BT.709 / limited) so players decode it with the same matrix.
* Audio: float [-1, 1] -> s16 interleaved ``AudioFrame`` -> ``AudioResampler``
  (fltp, stereo, 32000) -> explicit pts -> encoder flush, the order used by
  ``ltx_core/color/audio_mux.py``. The waveform is first cut to the video's
  length: the audio VAE emits whole 800-sample latents, up to 12.5 ms more than
  F / 24 seconds (R2 (3)-15).

A failed write removes the partial file.
"""

from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path

import torch

from engine_h3 import constants as C

CRF = 19
PRESET = "veryfast"   # the LTX engines' value

# BT.709 RGB -> Y'CbCr (rows Y, Cb, Cr), the same matrix as ltx_core/color/yuv.py.
_BT709 = torch.tensor(
    [
        [0.2126, 0.7152, 0.0722],
        [-0.1146, -0.3854, 0.5],
        [0.5, -0.4542, -0.0458],
    ],
    dtype=torch.float32,
)
# FFmpeg enum values (AVCOL_SPC_BT709 / AVCOL_PRI_BT709 / AVCOL_TRC_BT709 / AVCOL_RANGE_MPEG).
_AVCOL_BT709 = 1
_AVCOL_RANGE_MPEG = 1

_CONVERT_BATCH = 16  # frames converted per step (bounds the float32 scratch)


def rgb_to_i420_bt709(frames: torch.Tensor) -> torch.Tensor:
    """uint8 RGB [N, H, W, 3] -> uint8 I420 planes [N, H * 3 // 2, W] (BT.709, limited range).

    Chroma is the 2x2 average of the full-resolution Cb/Cr (centre siting, as
    the LTX converter does). Code values are rounded, not truncated.
    """
    n, h, w, _ = frames.shape
    rgb = frames.to(torch.float32).div_(255.0)
    yuv = rgb @ _BT709.T                                  # [N, H, W, 3]
    y = yuv[..., 0].mul(219.0).add_(16.0)                 # [N, H, W]
    uv = yuv[..., 1:].permute(0, 3, 1, 2)                 # [N, 2, H, W]
    uv = torch.nn.functional.avg_pool2d(uv, kernel_size=2, stride=2).mul_(224.0).add_(128.0)
    # U then V, each (H/2 * W/2) samples, packed contiguously as H/2 rows of W.
    uv = uv.reshape(n, h // 2, w)
    planes = torch.cat([y, uv], dim=1)
    return planes.round_().clamp_(0, 255).to(torch.uint8)


def _tag_bt709(codec_context) -> None:
    codec_context.colorspace = _AVCOL_BT709
    codec_context.color_range = _AVCOL_RANGE_MPEG
    # Present on VideoCodecContext in current PyAV; guarded so an older build
    # still writes the matrix/range tags (which are what decoders use).
    for attr in ("color_primaries", "color_trc"):
        if hasattr(codec_context, attr):
            setattr(codec_context, attr, _AVCOL_BT709)


def write_mp4(path: str, frames: torch.Tensor, *, fps: int, audio: torch.Tensor | None,
              sample_rate: int = C.AUDIO_SAMPLE_RATE) -> int:
    """Write ``frames`` (uint8 [F, H, W, 3], CPU) and optional ``audio`` (float32 [2, N], CPU, -1..1).

    Returns the size of the written file in bytes.
    """
    import av
    from av.audio.resampler import AudioResampler

    if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"write_mp4: frames は uint8 [F, H, W, 3] が必要です（{frames.dtype} {tuple(frames.shape)}）")
    if frames.device.type != "cpu":
        raise ValueError("write_mp4: frames は CPU のテンソルで渡してください")
    num_frames, height, width, _ = frames.shape
    if num_frames < 1:
        raise ValueError("write_mp4: フレームが 0 枚です")
    if height % 2 or width % 2:
        raise ValueError(f"write_mp4: yuv420p は高さと幅が偶数である必要があります（{height}x{width}）")
    if fps <= 0:
        raise ValueError(f"write_mp4: fps が不正です（{fps}）")
    if audio is not None:
        if audio.ndim != 2 or audio.shape[0] != 2 or audio.shape[1] == 0:
            raise ValueError(f"write_mp4: audio は [2, N]（N > 0）が必要です（{tuple(audio.shape)}）")
        if audio.device.type != "cpu":
            raise ValueError("write_mp4: audio は CPU のテンソルで渡してください")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out), mode="w")
    ok = False
    try:
        stream = container.add_stream("libx264", rate=int(fps), options={"crf": str(CRF), "preset": PRESET})
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        _tag_bt709(stream.codec_context)

        audio_stream = None
        if audio is not None:
            audio_stream = container.add_stream("aac", rate=sample_rate)
            audio_stream.codec_context.sample_rate = sample_rate
            audio_stream.codec_context.layout = "stereo"
            audio_stream.codec_context.time_base = Fraction(1, sample_rate)

        for start in range(0, num_frames, _CONVERT_BATCH):
            planes = rgb_to_i420_bt709(frames[start:start + _CONVERT_BATCH]).numpy()
            for k in range(planes.shape[0]):
                vf = av.VideoFrame.from_ndarray(planes[k], format="yuv420p")
                vf.pts = start + k
                container.mux(stream.encode(vf))
        container.mux(stream.encode())

        if audio_stream is not None:
            _write_audio(container, audio_stream, audio, num_frames=num_frames, fps=fps,
                         sample_rate=sample_rate, resampler_cls=AudioResampler)
        ok = True
    finally:
        # close() flushes the muxer and can raise itself. It must neither skip
        # the clean-up of a partial file nor replace an exception already in
        # flight; a close failure after a clean write is still an error.
        try:
            container.close()
        except Exception:
            if ok:
                ok = False
                out.unlink(missing_ok=True)
                raise
        finally:
            if not ok:
                out.unlink(missing_ok=True)
    return os.path.getsize(out)


def audio_keep_samples(available: int, *, num_frames: int, fps: int, sample_rate: int) -> int:
    """Samples kept: the video's duration ``round(F * sample_rate / fps)``, never more than given (no padding)."""
    return min(int(available), round(num_frames * sample_rate / fps))


def _write_audio(container, audio_stream, audio: torch.Tensor, *, num_frames: int, fps: int,
                 sample_rate: int, resampler_cls) -> None:
    import av

    keep = audio_keep_samples(audio.shape[1], num_frames=num_frames, fps=fps, sample_rate=sample_rate)
    samples = audio[:, :keep].to(torch.float32).clamp(-1.0, 1.0)
    pcm = (samples * 32767.0).round_().to(torch.int16)          # [2, N]
    interleaved = pcm.t().contiguous().reshape(1, -1).numpy()    # packed s16: L R L R ...
    frame_in = av.AudioFrame.from_ndarray(interleaved, format="s16", layout="stereo")
    frame_in.sample_rate = sample_rate
    frame_in.pts = 0

    resampler = resampler_cls(format="fltp", layout="stereo", rate=sample_rate)
    next_pts = 0
    for rframe in list(resampler.resample(frame_in)) + list(resampler.resample(None)):
        rframe.pts = next_pts
        rframe.sample_rate = sample_rate
        next_pts += rframe.samples
        container.mux(audio_stream.encode(rframe))
    container.mux(audio_stream.encode())
