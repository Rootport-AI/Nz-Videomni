"""engine_h3 audio encoder and A2V source-audio reader, checked on the CPU (台帳 §1-103 弾 1a; gate G1 ⑧).

Runs in ``.venv-engine-h3`` (torch, av) through the engine-venv runner::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_audio_encoder.py

What is pinned:
  * the prefix table: every tensor name in the real audio VAE file (header
    only) falls in one part; the encoder key map keeps exactly 134 tensors
    (encoder 110 + pre_block 22 + mean_proj 2) and drops logs_proj 2, the
    decoder and the 2 statistic tensors; the decoder key map is unchanged;
  * the encoder module's state_dict matches the file's names and shapes, and
    a real load leaves nothing on meta (``zero_k_bias`` is a persistent buffer);
  * ``encode`` returns float32 [2, 32, ceil(L / 800)] (zero right-pad);
  * ``load_source_waveform``: the resampler flush (44.1 kHz -> 32 kHz loses
    the tail without it), the 32 kHz fltp passthrough (flush returns [None]),
    the 1 ms shortfall pad vs the refusal, the ``read_samples`` cap, mono /
    first-two channel handling and the format-change refusal.

The real-weight tests skip when ``models/MiniMaxH3/VAE`` has no audio VAE.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
av = pytest.importorskip("av")

from av.audio.resampler import AudioResampler  # noqa: E402

from engine_h3 import audio_vae as A  # noqa: E402
from engine_h3 import source_audio as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AUDIO_VAE = ROOT / "models" / "MiniMaxH3" / "VAE" / "minimax_h3_audio_vae_fp32.safetensors"
needs_weights = pytest.mark.skipif(not AUDIO_VAE.is_file(), reason=f"{AUDIO_VAE} がありません")


# -- key maps / weights --------------------------------------------------------


def _header():
    import sft_quant_format

    return sft_quant_format.read_header(str(AUDIO_VAE))


@needs_weights
def test_key_maps_classify_every_tensor_of_the_file():
    names = list(_header().tensors)
    parts = {n: A.file_part(n) for n in names}
    assert [n for n, p in parts.items() if p is None] == []          # nothing unclassified

    loaded = [n for n in names if A.audio_encoder_key_map(n) is not None]
    skipped = [n for n in names if A.audio_encoder_key_map(n) is None]
    assert len(loaded) == A.ENCODER_TENSOR_COUNT == 134
    assert sum(n.startswith("encoder.") for n in loaded) == 110
    assert sum(n.startswith("pre_block.") for n in loaded) == 22
    assert sum(n.startswith("mean_proj.") for n in loaded) == 2
    assert all(A.audio_encoder_key_map(n) == n for n in loaded)       # file names as they are
    assert sorted(n for n in skipped if n.startswith("logs_proj.")) == ["logs_proj.bias", "logs_proj.weight"]
    assert sorted(n for n in skipped if parts[n] == A.PART_STATS) == ["latents_mean", "latents_std"]
    decoder = [n for n in names if parts[n] == A.PART_DECODER]
    assert len(skipped) == 2 + 2 + len(decoder)

    # The decoder's map is what it was: decoder tensors only, under their own names.
    assert [n for n in names if A.audio_vae_key_map(n) is not None] == decoder
    assert all(A.audio_vae_key_map(n) == n for n in decoder)


def test_decoder_key_map_passes_unknown_names_through():
    # Unchanged behaviour: a name outside the table reaches load_streaming (logged there as unexpected).
    assert A.audio_vae_key_map("something.new") == "something.new"
    assert A.audio_encoder_key_map("something.new") is None
    assert A.audio_vae_key_map("mean_proj.weight") is None
    assert A.audio_encoder_key_map("decoder.conv_pre.weight") is None


@needs_weights
def test_encoder_module_matches_the_file_names_and_shapes():
    header = _header()
    with torch.device("meta"):
        enc = A.H3AudioEncoder()
    state = {k: tuple(v.shape) for k, v in enc.state_dict().items()}
    file_shapes = {n: tuple(header.tensors[n].shape) for n in header.tensors if A.audio_encoder_key_map(n)}
    assert state == file_shapes
    assert "pre_block.attn.zero_k_bias" in state                       # persistent buffer
    assert "latents_mean" not in state and "latents_std" not in state  # non-persistent, from metadata


@needs_weights
def test_real_encoder_loads_134_tensors_and_encodes_a_tenth_of_a_second():
    from engine_h3.sft_load import meta_tensor_names

    enc = A.H3AudioEncoder.from_file(str(AUDIO_VAE), device="cpu")
    assert enc.load_report.loaded == 134
    assert len(enc.load_report.skipped) == len(_header().tensors) - 134
    assert meta_tensor_names(enc) == []
    assert all(p.dtype == torch.float32 for p in enc.parameters())
    z = enc.encode(torch.zeros(2, 3200))
    assert z.shape == (2, 32, 4) and z.dtype == torch.float32
    assert torch.isfinite(z).all()


# -- encode shapes (tiny random encoder) ----------------------------------------


@pytest.fixture(scope="module")
def tiny_encoder():
    torch.manual_seed(0)
    enc = A.H3AudioEncoder(encoder_dim=4, num_heads=2).eval()   # hop 800 kept; width 128
    with torch.no_grad():
        for name, p in enc.named_parameters():
            if name.endswith("alpha"):
                p.fill_(1.0)
            else:
                p.normal_(0.0, 0.05)
        enc.pre_block.attn.zero_k_bias.zero_()
    return enc


@pytest.mark.parametrize("length", [3200, 3201, 800, 1])
def test_encode_shape_dtype(tiny_encoder, length):
    wave_in = 0.1 * torch.randn(2, length)
    z = tiny_encoder.encode(wave_in)
    assert z.shape == (2, 32, math.ceil(length / 800))
    assert z.dtype == torch.float32
    assert torch.isfinite(z).all()


def test_encode_channels_are_a_batch(tiny_encoder):
    mono = 0.1 * torch.randn(1, 4000)
    z = tiny_encoder.encode(mono.repeat(2, 1))
    assert torch.allclose(z[0], z[1], atol=1e-6)   # batched kernels may differ in the last bit


def test_encode_normalises_with_the_statistics(tiny_encoder):
    w = 0.1 * torch.randn(2, 1600)
    base = tiny_encoder.encode(w)
    tiny_encoder.latents_mean.fill_(0.5)
    tiny_encoder.latents_std.fill_(2.0)
    try:
        assert torch.allclose(tiny_encoder.encode(w), (base - 0.5) / 2.0, atol=1e-6)
    finally:
        tiny_encoder.latents_mean.fill_(0.0)
        tiny_encoder.latents_std.fill_(1.0)


def test_encode_refuses_a_wrong_shape(tiny_encoder):
    with pytest.raises(ValueError):
        tiny_encoder.encode(torch.zeros(3200))
    with pytest.raises(ValueError):
        tiny_encoder.encode(torch.zeros(2, 0))


# -- load_source_waveform --------------------------------------------------------


def _write_wav(path: Path, data: np.ndarray, rate: int) -> Path:
    """``data`` float [C, N] in -1..1 -> 16-bit PCM wav."""
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(data.shape[0])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.T.tobytes())
    return path


def _sine(n: int, rate: int, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    return (amp * np.sin(2 * np.pi * freq * np.arange(n) / rate)).astype(np.float32)


def test_mono_44k1_becomes_32k_stereo_and_the_flush_keeps_the_tail(tmp_path):
    n_src = 44100 * 2 + 123
    path = _write_wav(tmp_path / "mono44.wav", _sine(n_src, 44100)[None], 44100)
    expected = round(n_src * 32000 / 44100)

    wave_out, info = S.load_source_waveform(str(path), keep_samples=32000, read_samples=10**9)
    assert wave_out.shape[0] == 2 and wave_out.dtype == torch.float32
    assert abs(wave_out.shape[1] - expected) <= 1
    assert torch.equal(wave_out[0], wave_out[1])
    assert info["channels_used"] == "mono->stereo"
    assert info["source_sample_rate"] == 44100 and info["source_channels"] == 1
    assert info["decoded_samples"] == wave_out.shape[1] and info["padded_samples"] == 0
    assert info["resampler"] == "pyav"

    # Without the flush the same decode ends short: the flush is what keeps the tail.
    with av.open(str(path)) as c:
        r = AudioResampler(format="fltp", layout=None, rate=32000)
        unflushed = sum(f.samples for fr in c.decode(audio=0) for f in r.resample(fr))
    assert unflushed < expected - 1


def test_passthrough_flush_returns_none_and_is_dropped():
    # 32 kHz fltp stereo frames: the resampler is a passthrough and resample(None) gives [None].
    frames = []
    for k in range(3):
        arr = np.full((2, 1000), 0.1 * (k + 1), dtype=np.float32)
        f = av.AudioFrame.from_ndarray(arr, format="fltp", layout="stereo")
        f.sample_rate = 32000
        frames.append(f)
    r = AudioResampler(format="fltp", layout=None, rate=32000)
    chunks = S.collect_resampled(iter(frames), r, read_samples=10**9)
    assert r.is_passthrough
    # The flush of a passthrough: [] in the PyAV 18.1.0 build (the guard against graph-less
    # flushes comes first), [None] by the shortcut in resampler.py 55-79. Both must be harmless.
    assert r.resample(None) in ([], [None])
    assert all(c is not None for c in chunks)
    assert sum(c.shape[1] for c in chunks) == 3000
    assert float(chunks[2][0, 0]) == pytest.approx(0.3)


def test_passthrough_file_32k_fltp_decodes(tmp_path):
    # AAC decodes to fltp, so a 32 kHz stereo m4a takes the passthrough path end to end.
    n = 32000 * 2
    path = tmp_path / "s32.m4a"
    with av.open(str(path), "w") as c:
        st = c.add_stream("aac", rate=32000)
        st.codec_context.layout = "stereo"
        data = np.stack([_sine(n, 32000), _sine(n, 32000, 660.0)])
        for start in range(0, n, 1024):
            f = av.AudioFrame.from_ndarray(np.ascontiguousarray(data[:, start:start + 1024]),
                                           format="fltp", layout="stereo")
            f.sample_rate = 32000
            f.pts = start
            c.mux(st.encode(f))
        c.mux(st.encode())
    with av.open(str(path)) as c:
        first = next(c.decode(audio=0))
        assert first.format.name == "fltp" and first.sample_rate == 32000   # really the passthrough case

    wave_out, info = S.load_source_waveform(str(path), keep_samples=n - 2048, read_samples=10**9)
    assert info["channels_used"] == "stereo" and info["source_sample_rate"] == 32000
    assert abs(wave_out.shape[1] - n) <= 2048            # AAC frames round the length
    assert torch.isfinite(wave_out).all()


def test_video_container_audio_stream_is_read(tmp_path):
    # An uploaded mp4 (video + 32 kHz stereo AAC) is read through its first audio stream.
    n = 32000
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as c:
        vs = c.add_stream("libx264", rate=24)
        vs.width, vs.height, vs.pix_fmt = 64, 64, "yuv420p"
        ast = c.add_stream("aac", rate=32000)
        ast.codec_context.layout = "stereo"
        for k in range(24):
            vf = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), dtype=np.uint8), format="rgb24")
            vf.pts = k
            c.mux(vs.encode(vf))
        c.mux(vs.encode())
        data = np.stack([_sine(n, 32000), _sine(n, 32000, 660.0)])
        for start in range(0, n, 1024):
            f = av.AudioFrame.from_ndarray(np.ascontiguousarray(data[:, start:start + 1024]),
                                           format="fltp", layout="stereo")
            f.sample_rate = 32000
            f.pts = start
            c.mux(ast.encode(f))
        c.mux(ast.encode())

    wave_out, info = S.load_source_waveform(str(path), keep_samples=n - 2048, read_samples=10**9)
    assert wave_out.shape[0] == 2 and info["channels_used"] == "stereo"
    assert abs(wave_out.shape[1] - n) <= 2048            # AAC frames round the length
    assert torch.isfinite(wave_out).all()


def test_short_by_at_most_1ms_is_padded_more_is_refused(tmp_path):
    keep = 32000
    ok = _write_wav(tmp_path / "ok.wav", _sine(keep - 20, 32000)[None], 32000)
    wave_out, info = S.load_source_waveform(str(ok), keep_samples=keep, read_samples=keep + 32000)
    assert wave_out.shape == (2, keep)
    assert info["decoded_samples"] == keep - 20 and info["padded_samples"] == 20
    assert torch.count_nonzero(wave_out[:, keep - 20:]) == 0

    edge = _write_wav(tmp_path / "edge.wav", _sine(keep - 32, 32000)[None], 32000)
    assert S.load_source_waveform(str(edge), keep_samples=keep, read_samples=keep)[0].shape == (2, keep)

    short = _write_wav(tmp_path / "short.wav", _sine(keep - 33, 32000)[None], 32000)
    with pytest.raises(ValueError, match="音声が映像より短いです"):
        S.load_source_waveform(str(short), keep_samples=keep, read_samples=keep)


def test_read_samples_caps_and_stops_reading(tmp_path):
    path = _write_wav(tmp_path / "long.wav", _sine(32000 * 3, 32000)[None], 32000)
    wave_out, info = S.load_source_waveform(str(path), keep_samples=16000, read_samples=32000)
    assert wave_out.shape == (2, 32000) and info["decoded_samples"] == 32000

    pulled = 0

    def counting(frames):
        nonlocal pulled
        for f in frames:
            pulled += 1
            yield f

    with av.open(str(path)) as c:
        total = sum(1 for _ in c.decode(audio=0))
    with av.open(str(path)) as c:
        r = AudioResampler(format="fltp", layout=None, rate=32000)
        S.collect_resampled(counting(c.decode(audio=0)), r, read_samples=32000)
    assert 0 < pulled < total


def test_three_channels_keep_the_first_two(tmp_path):
    n = 8000
    data = np.stack([np.full(n, 0.25), np.full(n, -0.5), np.full(n, 0.75)]).astype(np.float32)
    path = _write_wav(tmp_path / "three.wav", data, 32000)
    wave_out, info = S.load_source_waveform(str(path), keep_samples=n, read_samples=n)
    assert info["channels_used"] == "first2" and info["source_channels"] == 3
    assert float(wave_out[0].mean()) == pytest.approx(0.25, abs=1e-3)
    assert float(wave_out[1].mean()) == pytest.approx(-0.5, abs=1e-3)


def test_format_change_is_refused_in_japanese():
    def frame(rate):
        f = av.AudioFrame.from_ndarray(np.zeros((1, 2000), dtype=np.int16), format="s16", layout="mono")
        f.sample_rate = rate
        return f

    r = AudioResampler(format="fltp", layout=None, rate=32000)
    with pytest.raises(ValueError, match="音声の途中で"):
        S.collect_resampled(iter([frame(44100), frame(48000)]), r, read_samples=10**9)


def test_missing_file_and_bad_arguments_are_refused(tmp_path):
    with pytest.raises(ValueError, match="開けませんでした"):
        S.load_source_waveform(str(tmp_path / "none.wav"), keep_samples=10, read_samples=10)
    path = _write_wav(tmp_path / "a.wav", _sine(100, 32000)[None], 32000)
    with pytest.raises(ValueError):
        S.load_source_waveform(str(path), keep_samples=100, read_samples=50)
