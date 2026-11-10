"""engine_h3.source_tail / image_cond.cover_crop / media_out.cut_head / write_wav_pcm16
and the real ``_encode_source_tail`` / ``_decode(trim_head=...)`` (台帳 §1-121).

Runs in ``.venv-engine-h3`` through the engine-venv runner::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_source_tail.py

CPU only, no weights: the input video tail of a video-to-video chain (read,
canvas, frame count, audio presence), the key frames' cover + centre crop as a
shared function, the head trim and the audio handle wav; the pipeline's two
new steps with stand-in encoders / decoders (the mp4 is really written).
"""

from __future__ import annotations

import wave

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("av")

from PIL import Image  # noqa: E402

from engine_h3 import media_out as MO  # noqa: E402
from engine_h3 import pipeline_h3 as P  # noqa: E402
from engine_h3.image_cond import cover_crop, load_canvas_image  # noqa: E402
from engine_h3.source_tail import has_audio_stream, read_source_frames, read_source_tail  # noqa: E402


def _write_clip(path, frames=22, size=(80, 48), audio=True):
    """A synthetic mp4: frame i is grey 10 * i (+ a colour bar), 24 fps, optional audio."""
    w, h = size
    video = torch.zeros((frames, h, w, 3), dtype=torch.uint8)
    for i in range(frames):
        video[i] = 10 * i
        video[i, :, : w // 4, 0] = 250
    wav = None
    if audio:
        n = round(frames * 32000 / 24)
        t = torch.arange(n, dtype=torch.float32) / 32000
        wav = torch.stack([torch.sin(2 * torch.pi * 440 * t), torch.sin(2 * torch.pi * 220 * t)]) * 0.2
    MO.write_mp4(str(path), video, fps=24, audio=wav)
    return path


# --------------------------------------------------------------------------- #
# cover_crop (a pure refactor of load_canvas_image)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("src", [(100, 60), (61, 37), (64, 64), (33, 200), (96, 64), (1, 1)])
@pytest.mark.parametrize("canvas", [(64, 32), (96, 64), (32, 96)])
def test_cover_crop_is_load_canvas_image(tmp_path, src, canvas):
    rng = np.random.default_rng(sum(src) + sum(canvas))
    arr = rng.integers(0, 256, size=(src[1], src[0], 3), dtype=np.uint8)
    path = tmp_path / "k.png"
    Image.fromarray(arr).save(path)
    want, size = load_canvas_image(str(path), *canvas)
    with Image.open(path) as opened:
        got = cover_crop(opened.convert("RGB"), *canvas)
    assert size == src and got.size == canvas
    assert np.array_equal(np.asarray(got), np.asarray(want))


# --------------------------------------------------------------------------- #
# read_source_tail / has_audio_stream
# --------------------------------------------------------------------------- #

def test_read_source_tail_exact_count_on_the_canvas(tmp_path):
    path = _write_clip(tmp_path / "tail.mp4")
    frames, info = read_source_tail(str(path), 64, 32, 22)
    assert frames.dtype == np.uint8 and frames.shape == (22, 32, 64, 3)
    assert info == {"source_size": [80, 48], "frames_read": 22}
    # In order: the grey level climbs 10 per frame (right of the colour bar).
    greys = [float(frames[i, :, 40:, 1].mean()) for i in range(22)]
    assert all(abs(g - 10 * i) < 6 for i, g in enumerate(greys)), greys
    # Each frame is the cover-crop of the decoded frame.
    import av

    with av.open(str(path)) as c:
        first = next(c.decode(c.streams.video[0])).to_image()
    assert np.array_equal(frames[0], np.asarray(cover_crop(first, 64, 32)))


@pytest.mark.parametrize("want", [21, 23])
def test_read_source_tail_refuses_a_wrong_count(tmp_path, want):
    path = _write_clip(tmp_path / "tail.mp4")
    with pytest.raises(ValueError, match=f"ちょうど {want} フレーム"):
        read_source_tail(str(path), 64, 32, want)


def test_read_source_tail_refuses_a_missing_or_bad_file(tmp_path):
    with pytest.raises(ValueError, match="見つかりません"):
        read_source_tail(str(tmp_path / "none.mp4"), 64, 32, 22)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="読めません"):
        read_source_tail(str(bad), 64, 32, 22)


def test_has_audio_stream(tmp_path):
    assert has_audio_stream(str(_write_clip(tmp_path / "a.mp4", audio=True))) is True
    assert has_audio_stream(str(_write_clip(tmp_path / "v.mp4", audio=False))) is False
    with pytest.raises(ValueError, match="開けません"):
        has_audio_stream(str(tmp_path / "none.mp4"))


# --------------------------------------------------------------------------- #
# read_source_frames (Outpainting, 台帳 §1-125: the source at its own size)
# --------------------------------------------------------------------------- #

def test_read_source_frames_keeps_the_source_size(tmp_path):
    path = _write_clip(tmp_path / "src.mp4")
    frames = read_source_frames(str(path), 22)
    assert frames.dtype == np.uint8 and frames.shape == (22, 48, 80, 3)   # no resize, no crop
    greys = [float(frames[i, :, 40:, 1].mean()) for i in range(22)]
    assert all(abs(g - 10 * i) < 6 for i, g in enumerate(greys)), greys
    # Each frame is the decoded frame's pixels as they are (the paste-back relies on it).
    import av

    with av.open(str(path)) as c:
        decoded = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
    assert all(np.array_equal(frames[i], decoded[i]) for i in range(22))


@pytest.mark.parametrize("want", [21, 23])
def test_read_source_frames_refuses_a_wrong_count(tmp_path, want):
    path = _write_clip(tmp_path / "src.mp4")
    with pytest.raises(ValueError, match=f"元動画の切り出しのフレーム数が合いません.*ちょうど {want} フレーム"):
        read_source_frames(str(path), want)


def test_read_source_frames_refuses_a_missing_or_bad_file(tmp_path):
    with pytest.raises(ValueError, match="元動画の切り出しが見つかりません"):
        read_source_frames(str(tmp_path / "none.mp4"), 22)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="元動画の切り出しを読めません"):
        read_source_frames(str(bad), 22)


def test_read_source_frames_refuses_mixed_frame_sizes(tmp_path, monkeypatch):
    """A file whose frames change size (an encoder will not write one): PyAV is stood in."""
    import av

    class _Frame:
        def __init__(self, w, h):
            self.width, self.height = w, h

        def to_ndarray(self, format):  # noqa: A002 -- PyAV's keyword
            assert format == "rgb24"
            return np.zeros((self.height, self.width, 3), np.uint8)

    class _Container:
        streams = type("S", (), {"video": ["v0"]})()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def decode(self, stream):
            return iter([_Frame(80, 48), _Frame(80, 48), _Frame(48, 80)])

    path = tmp_path / "mixed.mp4"
    path.write_bytes(b"x")
    monkeypatch.setattr(av, "open", lambda *a, **k: _Container())
    with pytest.raises(ValueError, match="寸法がそろっていません（3 フレーム目が 48×80、1 フレーム目は 80×48）"):
        read_source_frames(str(path), 3)
    # The same three frames of one size are read.
    monkeypatch.setattr(_Container, "decode", lambda self, stream: iter([_Frame(80, 48)] * 3))
    assert read_source_frames(str(path), 3).shape == (3, 48, 80, 3)


# --------------------------------------------------------------------------- #
# cut_head / write_wav_pcm16
# --------------------------------------------------------------------------- #

def test_cut_head_arithmetic():
    total = 73                                            # 56 + 39 - 22
    frames = torch.arange(total, dtype=torch.uint8).view(total, 1, 1, 1).expand(total, 2, 2, 3).contiguous()
    wav = torch.randn(2, 800 * P.audio_latents(total), generator=torch.Generator().manual_seed(1))
    before = wav.clone()
    new_frames, full, new_wav, info = MO.cut_head(frames, wav, trim_frames=22, num_frames=total, fps=24, sr=32000)
    assert info == {"trimmed_px": 22, "trimmed_audio_samples": 29333, "audio_fade_in_samples": 960}
    assert new_frames.shape[0] == 51 and int(new_frames[0, 0, 0, 0]) == 22
    assert full.shape == (2, round(total * 32000 / 24)) == (2, 97333)      # the handle: the timeline's length
    assert torch.equal(full, before[:, :97333])
    assert new_wav.shape == (2, 97333 - 29333)
    ramp = torch.linspace(0.0, 1.0, 960)
    assert torch.equal(new_wav[:, :960], before[:, 29333:29333 + 960] * ramp)
    assert torch.equal(new_wav[:, 960:], before[:, 29333 + 960:97333])
    assert torch.equal(wav, before)                       # pure: the input is untouched


def test_cut_head_short_audio_has_no_fade():
    frames = torch.zeros((23, 2, 2, 3), dtype=torch.uint8)
    wav = torch.ones(2, 29334)
    _f, _full, new_wav, info = MO.cut_head(frames, wav, trim_frames=22, num_frames=23, fps=24, sr=32000)
    assert new_wav.shape == (2, 1) and info["audio_fade_in_samples"] == 0 and float(new_wav[0, 0]) == 1.0


def test_write_wav_pcm16_round_trip(tmp_path):
    wav = torch.tensor([[0.0, 0.5, -0.5, 1.5, -2.0], [0.25, -0.25, 1.0, -1.0, 0.0]])
    path = tmp_path / "sub" / "h_audio_handle.wav"
    MO.write_wav_pcm16(str(path), wav, 32000)
    with wave.open(str(path), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (2, 2, 32000, 5)
        data = np.frombuffer(w.readframes(5), dtype="<i2").reshape(5, 2).T
    want = (wav.clamp(-1, 1) * 32767).round().to(torch.int16).numpy()
    assert np.array_equal(data, want)
    with pytest.raises(ValueError, match=r"\[2, N\]"):
        MO.write_wav_pcm16(str(tmp_path / "x.wav"), torch.zeros(1, 4), 32000)


# --------------------------------------------------------------------------- #
# the pipeline's two steps: _encode_source_tail and _decode(trim_head=...)
# --------------------------------------------------------------------------- #

def _pipe(progress=None):
    return P.H3Pipeline(P.H3Paths("t", "te", "vv", "av", "tok"), P.H3LoadOptions(), device="cpu",
                        progress=progress or (lambda *a, **k: None), log=lambda m: None)


@pytest.mark.parametrize("samples, freeze", [(None, 0), (500, 0), (29333, 36), (29600, 37), (40000, 37)])
def test_encode_source_tail_video_and_floored_audio(monkeypatch, samples, freeze):
    """Video: one 22-frame encode to [1, 24, 7, lh, lw]; audio: floor(N / 800) latents
    up to 37 through the A2V encoder (the zero-padded last hop is not frozen)."""
    from test_h3_engine import _a2v_fakes

    wave_t, calls = _a2v_fakes(monkeypatch, samples=samples or 800)
    progress = []
    pipe = _pipe(lambda *a, **k: progress.append(a))
    built = []

    class FakeVideoEncoder:
        def encode_clips(self, frames, *, device, tiled=True):
            built.append((tuple(frames.shape), str(device), tiled))
            return torch.full((1, 24, 7, frames.shape[1] // 16, frames.shape[2] // 16), 0.5)

    monkeypatch.setattr(pipe, "_build_video_encoder", lambda: FakeVideoEncoder())
    frames = np.zeros((22, 32, 64, 3), np.uint8)
    entry: dict = {}
    video, audio = pipe._encode_source_tail(frames, None if samples is None else wave_t, entry, 4, 5)
    assert built == [((22, 32, 64, 3), "cpu", True)]
    assert video.shape == (1, 24, 7, 2, 4) and video.dtype == torch.float32 and video.device.type == "cpu"
    if freeze:
        assert audio.shape == (2, 32, freeze) and audio.dtype == torch.float32 and audio.device.type == "cpu"
        assert calls["encode"][0].shape == (2, samples)        # the whole tail is encoded
    else:
        assert audio is None and calls["build"] == 0
    assert progress == [("encode", 4, 5)]                       # one step either way
    assert entry == {
        "latent_shape": [1, 24, 7, 2, 4], "frames_read": 22, "canvas": [64, 32],
        "source_had_audio": samples is not None, "freeze_ka": freeze,
        "encoded_samples": samples if freeze else 0,
    }


def test_encode_source_tail_refuses_a_wrong_latent_shape(monkeypatch):
    pipe = _pipe()

    class Bad:
        def encode_clips(self, frames, *, device, tiled=True):
            return torch.zeros((1, 24, 6, 2, 4))

    monkeypatch.setattr(pipe, "_build_video_encoder", lambda: Bad())
    with pytest.raises(RuntimeError, match="source encode"):
        pipe._encode_source_tail(np.zeros((22, 32, 64, 3), np.uint8), None, {}, 4, 4)


def _fake_decoders(monkeypatch, frames):
    from engine_h3 import audio_vae as AV
    from engine_h3 import video_vae as VV

    class FakeDecoder:
        def to(self, device):
            return self

        def decode(self, z, tiled=True):
            out = torch.zeros((frames, 32, 64, 3), dtype=torch.uint8)
            for i in range(frames):
                out[i] = 3 * i
            return out

    class FakeAudio:
        def to(self, device):
            return self

        def decode(self, a):
            return torch.full((2, 800 * a.shape[-1]), 0.25)

    monkeypatch.setattr(VV.H3VideoDecoder, "from_file", classmethod(lambda cls, *a, **k: FakeDecoder()))
    monkeypatch.setattr(AV.H3AudioVAE, "from_file", classmethod(lambda cls, *a, **k: FakeAudio()))


def test_decode_trim_head_writes_the_trimmed_mp4_and_the_handle(monkeypatch, tmp_path):
    import av

    total = 73
    _fake_decoders(monkeypatch, total)
    pipe = _pipe()
    out = tmp_path / "chain.mp4"
    handle = tmp_path / "chain_audio_handle.wav"
    req = P.H3GenerateRequest(prompt="p", seed=1, width=64, height=32, num_frames=total, output_path=str(out))
    entry: dict = {}
    a_lat = P.audio_latents(total)
    size = pipe._decode(req, torch.zeros(1, 24, 22, 2, 4), torch.zeros(2, 32, a_lat), entry,
                        trim_head=22, handle_path=str(handle))
    assert size == out.stat().st_size > 0
    with av.open(str(out)) as c:
        decoded = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
    assert len(decoded) == total - 22 and abs(float(decoded[0].mean()) - 66) < 4    # frame 22 = grey 66
    with wave.open(str(handle), "rb") as w:
        assert w.getnframes() == round(total * 32000 / 24) == 97333
    assert entry["decoded_frames"] == total and entry["trimmed_frames"] == 22
    assert entry["trimmed_audio_samples"] == 29333 and entry["audio_fade_in_samples"] == 960
    assert entry["audio_handle"] == "chain_audio_handle.wav"
    assert entry["frames"] == total - 22
    assert entry["audio_samples"] == 97333 - 29333 == round(51 * 32000 / 24)
    assert entry["audio_samples_decoded"] == 800 * a_lat


def test_decode_handle_failure_does_not_fail_the_job(monkeypatch, tmp_path):
    _fake_decoders(monkeypatch, 61)
    logs = []
    pipe = P.H3Pipeline(P.H3Paths("t", "te", "vv", "av", "tok"), P.H3LoadOptions(), device="cpu",
                        progress=lambda *a, **k: None, log=logs.append)
    monkeypatch.setattr(MO, "write_wav_pcm16", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    out = tmp_path / "c.mp4"
    req = P.H3GenerateRequest(prompt="p", seed=1, width=64, height=32, num_frames=61, output_path=str(out))
    entry: dict = {}
    pipe._decode(req, torch.zeros(1, 24, 17, 2, 4), torch.zeros(2, 32, P.audio_latents(61)), entry,
                 trim_head=22, handle_path=str(tmp_path / "c_audio_handle.wav"))
    assert out.is_file() and entry["audio_handle"] is None and entry["frames"] == 39
    assert any("audio handle not written" in m for m in logs)


def test_decode_without_trim_is_as_before(monkeypatch, tmp_path):
    _fake_decoders(monkeypatch, 22)
    pipe = _pipe()
    req = P.H3GenerateRequest(prompt="p", seed=1, width=64, height=32, num_frames=22,
                              output_path=str(tmp_path / "s.mp4"))
    entry: dict = {}
    pipe._decode(req, torch.zeros(1, 24, 7, 2, 4), torch.zeros(2, 32, 37), entry)
    assert set(entry) == {"vae_source", "audio_samples", "audio_samples_decoded", "frames"}
    assert entry["frames"] == 22 and not (tmp_path / "s_audio_handle.wav").exists()
