"""Read the A2V source audio as a 32 kHz stereo float32 waveform (台帳 §1-103, plan §2.3).

The H3 venv has no torchaudio, so the file is decoded and resampled with PyAV
(the same library :mod:`engine_h3.media_out` muxes with):

* ``AudioResampler(format="fltp", layout=None, rate=32000)``: planar float32
  at 32 kHz, channel layout unchanged (the channels are chosen afterwards, as
  DiffSynth does, instead of letting ffmpeg down-mix a 5.1 file);
* after the read loop the resampler is ALWAYS flushed with ``resample(None)``:
  swresample holds the last few milliseconds back until then. When the input
  is already 32 kHz fltp the resampler is a passthrough; its flush returns
  ``[]`` in the 18.1.0 build and ``[None]`` by the shortcut in
  ``av/audio/resampler.py`` 55-79, so ``None`` elements are dropped either way;
* a stream whose sample format / layout / rate changes part-way makes PyAV
  raise ``ValueError("Frame does not match AudioResampler setup.")``; it is
  re-raised with a Japanese reason (it reaches the worker's error event).

Channels: mono is duplicated to two channels, three or more keep the first two
(DiffSynth ``convert_to_stereo`` + ``waveform[:2]``). No peak normalisation.

Length: reading stops once ``read_samples`` (at 32 kHz) are in hand. The
caller's ``keep_samples`` is the video's duration; a waveform shorter by more
than 32 samples (1 ms, the app's pre-check tolerance) is refused, a shorter
one by at most that is padded with zeros up to ``keep_samples``.
"""

from __future__ import annotations

import numpy as np
import torch

from engine_h3 import constants as C

# The shortfall (samples at 32 kHz) that is padded instead of refused: 1 ms,
# the same tolerance as the app's pre-check (duration + 0.001 < F / fps).
SHORTFALL_TOLERANCE = 32

_FORMAT_CHANGED = "音声の途中でサンプル形式・チャンネル数・サンプリングレートのいずれかが変わっています。一つの形式に揃えた音声ファイルを使ってください"


def _resample(resampler, frame) -> list[np.ndarray]:
    """One ``resample`` call -> planar float32 arrays [C, n]; ``None`` elements (passthrough flush) dropped."""
    try:
        out = resampler.resample(frame)
    except ValueError as exc:
        import av

        if isinstance(exc, av.error.FFmpegError):   # InvalidDataError is also a ValueError
            raise
        raise ValueError(f"{_FORMAT_CHANGED}（{exc}）") from exc
    return [f.to_ndarray() for f in out if f is not None]


def collect_resampled(frames, resampler, *, read_samples: int, skip_samples: int = 0) -> list[np.ndarray]:
    """Push ``frames`` through ``resampler`` until ``read_samples`` are in hand, then flush.

    Returns the planar float32 chunks [C, n] in order (their total can exceed
    ``read_samples``; the caller cuts). The flush always runs, also after an
    early stop, so the samples swresample holds back are not lost.

    ``skip_samples``: the first that many resampled (32 kHz) samples are
    dropped AFTER the conversion (read and thrown away, not seeked: the chunks
    of the skipped stretch are not kept), and ``read_samples`` counts from
    there. 0 (the default) keeps everything, as before.
    """
    chunks: list[np.ndarray] = []
    total = 0
    to_skip = skip_samples

    def keep(arr: np.ndarray) -> None:
        nonlocal total, to_skip
        if to_skip:
            if arr.shape[1] <= to_skip:
                to_skip -= arr.shape[1]
                return
            arr = arr[:, to_skip:]
            to_skip = 0
        chunks.append(arr)
        total += arr.shape[1]

    for frame in frames:
        for arr in _resample(resampler, frame):
            keep(arr)
        if total >= read_samples:
            break
    for arr in _resample(resampler, None):
        keep(arr)
    return chunks


def load_source_waveform(
    path: str, *, keep_samples: int, read_samples: int, start_samples: int = 0,
) -> tuple[torch.Tensor, dict]:
    """Decode ``path`` -> (waveform float32 [2, N] at 32 kHz, info).

    ``N`` is ``min(available, read_samples)``, or ``keep_samples`` when the
    file falls short of it by at most :data:`SHORTFALL_TOLERANCE` samples
    (zero-padded). ``info`` reports ``source_sample_rate``, ``source_channels``,
    ``decoded_samples`` (after the cap, before padding), ``padded_samples``,
    ``channels_used`` ("mono->stereo" / "stereo" / "first2"), ``resampler``
    and ``start_samples``.

    ``start_samples`` (a chain clip's own audio, 台帳 §1-120): the waveform
    starts at that 32 kHz sample; the file is decoded from its head and the
    samples before it are thrown away after the conversion (no seek), so the
    result equals the whole decode's ``[:, start:start + N]``. Every count
    above (``available``, the shortfall) is then counted from the start. 0
    (the default) is the whole file from its head, as before.
    """
    import av
    from av.audio.resampler import AudioResampler

    if keep_samples < 1 or read_samples < keep_samples:
        raise ValueError(
            f"音声の読み込み: keep_samples={keep_samples} / read_samples={read_samples} が不正です"
            "（1 ≦ keep_samples ≦ read_samples が必要です）"
        )
    if isinstance(start_samples, bool) or not isinstance(start_samples, int) or start_samples < 0:
        raise ValueError(f"音声の読み込み: start_samples={start_samples!r} が不正です（0 以上の整数が必要です）")
    try:
        container = av.open(str(path))
    except (av.error.FFmpegError, OSError) as exc:
        raise ValueError(f"音声ファイルを開けませんでした（{path}）: {exc}") from exc
    with container:
        if not container.streams.audio:
            raise ValueError(f"ファイルに音声の流れ（トラック）がありません（{path}）")
        stream = container.streams.audio[0]
        source_rate = int(stream.codec_context.sample_rate or 0)
        source_channels = int(stream.codec_context.layout.nb_channels)
        resampler = AudioResampler(format="fltp", layout=None, rate=C.AUDIO_SAMPLE_RATE)
        try:
            chunks = collect_resampled(
                container.decode(stream), resampler, read_samples=read_samples, skip_samples=start_samples,
            )
        except av.error.FFmpegError as exc:
            raise ValueError(f"音声の復号に失敗しました（{path}）: {exc}") from exc

    if start_samples and not any(c.shape[1] for c in chunks):
        raise ValueError(
            f"開始位置（{start_samples} サンプル＝{start_samples / C.AUDIO_SAMPLE_RATE:.3f} 秒）"
            f"が音声の長さを超えています（{path}）"
        )
    if not chunks:
        raise ValueError(f"音声を 1 サンプルも読めませんでした（{path}）")
    pcm = np.concatenate(chunks, axis=1)[:, :read_samples].astype(np.float32, copy=False)
    channels = pcm.shape[0]
    if channels == 1:
        pcm = np.concatenate([pcm, pcm], axis=0)
        channels_used = "mono->stereo"
    elif channels == 2:
        channels_used = "stereo"
    else:
        pcm = pcm[:2]
        channels_used = "first2"
    wave = torch.from_numpy(np.ascontiguousarray(pcm))

    decoded = wave.shape[1]
    if decoded < keep_samples - SHORTFALL_TOLERANCE:
        sr = C.AUDIO_SAMPLE_RATE
        raise ValueError(
            f"音声が映像より短いです: 音声 {decoded / sr:.3f} 秒（{decoded} サンプル）に対し、"
            f"映像の長さ {keep_samples / sr:.3f} 秒（{keep_samples} サンプル）が必要です。"
            "フレーム数を減らすか、長い音声を使ってください"
        )
    padded = max(0, keep_samples - decoded)
    if padded:
        wave = torch.nn.functional.pad(wave, (0, padded))
    info = {
        "source_sample_rate": source_rate,
        "source_channels": source_channels,
        "decoded_samples": int(decoded),
        "padded_samples": int(padded),
        "channels_used": channels_used,
        "resampler": "pyav",
        "start_samples": int(start_samples),
    }
    return wave, info
