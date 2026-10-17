"""engine_h3 video VAE / audio VAE / mp4 writer, checked on the CPU (台帳 §1-93 第 1 弾; owner C2).

Runs in ``.venv-engine-h3`` (torch, av) through the engine-venv runner::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_media.py

What is pinned (code review M5 / m2 / m12):
  * the 17n+5 <-> 5n+2 decode arithmetic: F = 22 / 124 / 345 come back with the
    same frame count from 1 / 7 / 20 decoder windows; T' = 1 -> 1 frame;
    T' = 2 (F = 5) is refused;
  * ``split_tiles`` covers the frame exactly (384, 640, 1344 px) with overlaps
    >= 64 px on the 16 px latent grid;
  * the BT.709 limited-range conversion on known colours (red, white, black);
  * the audio is cut to ``round(F * 32000 / 24)`` samples, and ``write_mp4``
    leaves no partial file when the write or the close fails;
  * the float32 islands: norms stay float32 with their F16 values intact
    (no bf16 round trip), convolutions / linears become bf16;
  * the audio decoder's output length (A latents -> A * 800 samples).

A tiny decoder (``ch=32``, one transformer layer, 2 heads) keeps every test
at a few seconds; the weights are random, only shapes and arithmetic are
checked. The real-weight round trip is a separate manual check.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
av = pytest.importorskip("av")

from engine_h3 import audio_vae as A  # noqa: E402
from engine_h3 import constants as C  # noqa: E402
from engine_h3 import media_out as M  # noqa: E402
from engine_h3 import video_vae as V  # noqa: E402


@pytest.fixture(scope="module")
def tiny_vae():
    torch.manual_seed(0)
    vae = V.H3VideoVAE(ch=32, decoder_layers=1, decoder_heads=2, decoder_dim_head=64).eval()
    with torch.no_grad():
        for p in vae.parameters():
            p.normal_(0.0, 0.02)
    return vae


# -- decode arithmetic --------------------------------------------------------


@pytest.mark.parametrize("num_frames, windows", [(22, 1), (124, 7), (345, 20)])
def test_decode_frame_count_and_windows(tiny_vae, monkeypatch, num_frames, windows):
    t_lat = C.latent_frames(num_frames)
    calls = []
    real = tiny_vae._decode_clip

    def counting(z, tiled):
        calls.append(z.shape[2])
        return real(z, tiled)

    monkeypatch.setattr(tiny_vae, "_decode_clip", counting)
    with torch.no_grad():
        out = tiny_vae.decode(torch.randn(1, 24, t_lat, 1, 1))
    assert out.shape == (1, 3, num_frames, 16, 16)
    assert len(calls) == windows
    assert all(n == 7 for n in calls)          # 5 + 2 latents per window
    assert out.dtype == torch.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_decode_single_latent_gives_one_frame(tiny_vae, monkeypatch):
    seen = []
    real = tiny_vae._decode_clip

    def spy(z, tiled):
        seen.append(z.shape[2])
        return real(z, tiled)

    monkeypatch.setattr(tiny_vae, "_decode_clip", spy)
    with torch.no_grad():
        out = tiny_vae.decode(torch.randn(1, 24, 1, 2, 2))
    assert out.shape == (1, 3, 1, 32, 32)
    assert seen == [tiny_vae.tokens_chunk_size]   # decoded as one chunk of copies


def test_decode_two_latents_refused(tiny_vae):
    with torch.no_grad(), pytest.raises(ValueError):
        tiny_vae.decode(torch.randn(1, 24, 2, 1, 1))


def test_encode_shapes(tiny_vae):
    with torch.no_grad():
        z = tiny_vae.encode(torch.rand(1, 3, 22, 32, 32))
        z1 = tiny_vae.encode(torch.rand(1, 3, 1, 32, 32))
    assert z.shape == (1, 24, C.latent_frames(22), 2, 2)
    assert z1.shape == (1, 24, 1, 2, 2)


# -- spatial tiles -------------------------------------------------------------


@pytest.mark.parametrize("length", [384, 640, 1344])
def test_split_tiles_cover_exactly(tiny_vae, length):
    starts, lens, overlaps = tiny_vae.split_tiles(length)
    assert starts[0] == 0
    assert starts[-1] + lens[-1] == length
    assert all(n == tiny_vae.tile_size for n in lens)
    assert len(overlaps) == len(starts) - 1
    for i, ov in enumerate(overlaps):
        assert ov >= tiny_vae.tile_overlap_min
        assert ov % tiny_vae.vae_ratio == 0
        assert starts[i + 1] == starts[i] + lens[i] - ov
    assert all(s % tiny_vae.vae_ratio == 0 for s in starts)


def test_split_tiles_known_layouts(tiny_vae):
    assert tiny_vae.split_tiles(384) == ([0, 128], [256, 256], [128])
    assert tiny_vae.split_tiles(640) == ([0, 192, 384], [256, 256, 256], [64, 64])
    assert tiny_vae.split_tiles(256) == ([0], [256], [])


# -- float32 islands (m2) -----------------------------------------------------


def test_islands_float32_without_bf16_round_trip():
    vae = V.H3VideoVAE(ch=32, decoder_layers=1, decoder_heads=2, decoder_dim_head=64)
    with torch.no_grad():
        for p in vae.parameters():
            p.normal_(1.0, 0.3)
    vae.half()   # what the Comfy-Org file holds
    vae.latents_mean = torch.zeros(24)
    vae.latents_std = torch.ones(24)
    blk = vae.decoder.transformer_blocks[0]
    norm_f16 = blk.norm1.weight.detach().clone()
    scale_f16 = blk.scale1.weight.detach().clone()
    vae.apply_precision(torch.bfloat16)
    assert blk.norm1.weight.dtype == torch.float32
    assert blk.scale1.weight.dtype == torch.float32
    assert vae.decoder.norm_out.weight.dtype == torch.float32
    assert vae.decoder.x_embedder.weight.dtype == torch.float32
    assert vae.post_quant_conv.weight.dtype == torch.float32
    # F16 -> F32 is exact; a bf16 detour would have changed these values.
    assert torch.equal(blk.norm1.weight, norm_f16.float())
    assert torch.equal(blk.scale1.weight, scale_f16.float())
    assert not torch.equal(norm_f16.to(torch.bfloat16).float(), norm_f16.float())  # the detour is visible
    assert vae.encoder.conv_in.weight.dtype == torch.bfloat16
    assert blk.attn.to_qkv.weight.dtype == torch.bfloat16
    assert vae.decoder.proj_out.weight.dtype == torch.bfloat16
    assert vae.latents_mean.dtype == torch.float32


# -- audio ---------------------------------------------------------------------


def test_audio_decode_length():
    torch.manual_seed(0)
    aud = A.H3AudioVAE().eval()
    with torch.no_grad():
        for name, p in aud.named_parameters():
            if name.endswith((".alpha", ".beta")):
                p.zero_()
            else:
                p.normal_(0.0, 0.01)
        for name, b in aud.named_buffers():
            if name.endswith("filter"):
                b.fill_(1.0 / b.shape[-1])
        wav = aud.decode(torch.randn(2, 32, 3))
    assert wav.shape == (2, 3 * C.AUDIO_HOP)
    assert wav.dtype == torch.float32


@pytest.mark.parametrize("num_frames", [22, 124, 345])
def test_audio_keep_samples(num_frames):
    expected = round(num_frames * 32000 / 24)
    longer = C.audio_latents(num_frames) * C.AUDIO_HOP
    assert longer >= expected
    assert M.audio_keep_samples(longer, num_frames=num_frames, fps=24, sample_rate=32000) == expected
    assert M.audio_keep_samples(100, num_frames=num_frames, fps=24, sample_rate=32000) == 100  # never pads


# -- BT.709 conversion ---------------------------------------------------------


@pytest.mark.parametrize("rgb, yuv", [
    ((255, 0, 0), (63, 102, 240)),
    ((255, 255, 255), (235, 128, 128)),
    ((0, 0, 0), (16, 128, 128)),
])
def test_rgb_to_i420_bt709_known_colours(rgb, yuv):
    h, w = 4, 6
    frames = torch.tensor(rgb, dtype=torch.uint8).view(1, 1, 1, 3).expand(1, h, w, 3).contiguous()
    planes = M.rgb_to_i420_bt709(frames)
    assert planes.shape == (1, h * 3 // 2, w)
    y = planes[0, :h]
    uv = planes[0, h:].reshape(-1)
    n = (h // 2) * (w // 2)
    assert torch.all(y == yuv[0])
    assert torch.all(uv[:n] == yuv[1])
    assert torch.all(uv[n:] == yuv[2])


# -- write_mp4 -----------------------------------------------------------------


def _frames(n=5, h=32, w=48):
    return (torch.rand(n, h, w, 3) * 255).to(torch.uint8)


def test_write_mp4_trims_audio(tmp_path):
    num_frames = 22
    keep = round(num_frames * 32000 / 24)
    audio = 0.3 * torch.sin(torch.arange(keep + 400) / 10.0).repeat(2, 1)   # 400 samples too long
    out = tmp_path / "a.mp4"
    size = M.write_mp4(str(out), _frames(num_frames), fps=24, audio=audio)
    assert size == out.stat().st_size > 0
    with av.open(str(out)) as c:
        vs = c.streams.video[0]
        ast = c.streams.audio[0]
        assert vs.codec_context.pix_fmt == "yuv420p"
        assert ast.codec_context.sample_rate == 32000
        assert ast.codec_context.channels == 2
        decoded = sum(f.samples for f in c.decode(audio=0))
    # AAC adds at most one 1024-sample frame of padding; the 400 extra are gone.
    assert keep <= decoded < keep + 1024 + 1


def test_write_mp4_failure_removes_partial_file(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("audio failed")

    monkeypatch.setattr(M, "_write_audio", boom)
    out = tmp_path / "b.mp4"
    with pytest.raises(RuntimeError, match="audio failed"):
        M.write_mp4(str(out), _frames(), fps=24, audio=torch.zeros(2, 32000))
    assert not out.exists()


def test_write_mp4_close_failure_keeps_original_error_and_cleans_up(tmp_path, monkeypatch):
    real_open = av.open

    class Wrapped:
        def __init__(self, c):
            self._c = c

        def __getattr__(self, name):
            return getattr(self._c, name)

        def close(self):
            self._c.close()
            raise OSError("close failed")

    monkeypatch.setattr(av, "open", lambda *a, **k: Wrapped(real_open(*a, **k)))

    def boom(*a, **k):
        raise RuntimeError("audio failed")

    monkeypatch.setattr(M, "_write_audio", boom)
    out = tmp_path / "c.mp4"
    with pytest.raises(RuntimeError, match="audio failed"):
        M.write_mp4(str(out), _frames(), fps=24, audio=torch.zeros(2, 32000))
    assert not out.exists()

    # A close failure after a clean write is an error too, and removes the file.
    monkeypatch.setattr(M, "_write_audio", lambda *a, **k: None)
    out2 = tmp_path / "d.mp4"
    with pytest.raises(OSError, match="close failed"):
        M.write_mp4(str(out2), _frames(), fps=24, audio=None)
    assert not out2.exists()


def test_write_mp4_rejects_bad_frames(tmp_path):
    with pytest.raises(ValueError):
        M.write_mp4(str(tmp_path / "e.mp4"), torch.zeros(2, 31, 32, 3, dtype=torch.uint8), fps=24, audio=None)
    with pytest.raises(ValueError):
        M.write_mp4(str(tmp_path / "f.mp4"), torch.zeros(2, 32, 32, 3), fps=24, audio=None)
