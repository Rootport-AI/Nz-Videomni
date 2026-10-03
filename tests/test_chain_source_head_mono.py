"""LTX 2.3 chain V2V head: a mono source is duplicated to stereo (§1-53).

The audio VAE encoder's conv_in is stereo-only, so ``_encode_source_heads``
must hand it two channels, as ``_encode_end_source`` and the retake / A2V paths
do. Run on CPU with fake decoders and encoders; ``cleanup_memory`` is patched
out (the real one initialises CUDA).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ltx_pipelines")

import ltx_core.model.audio_vae as avae  # noqa: E402
import ltx_pipelines.utils.helpers as helpers  # noqa: E402
import ltx_pipelines.utils.media_io as mio  # noqa: E402

import engine.pipeline.chain_pipeline as cp  # noqa: E402

N_CTX_V, N_CTX_A, N_SAMPLES = 2, 4, 1600


def _encode_with(monkeypatch, waveform: torch.Tensor) -> torch.Tensor:
    seen: list[torch.Tensor] = []
    monkeypatch.setattr(helpers, "cleanup_memory", lambda: None)
    monkeypatch.setattr(cp, "load_video_conditioning_cpu", lambda **_k: torch.zeros(1, 3, 9, 8, 8))
    monkeypatch.setattr(
        mio, "decode_audio_from_file",
        lambda _p, _d: SimpleNamespace(waveform=waveform, sampling_rate=16000),
    )

    def _fake_encode(audio, _enc, _x):
        seen.append(audio.waveform)
        return torch.zeros(1, 8, N_CTX_A, 16)

    monkeypatch.setattr(avae, "encode_audio", _fake_encode)
    encoder = SimpleNamespace(tiled_encode=lambda _x, _cfg: torch.zeros(1, 128, N_CTX_V, 1, 1))
    out = cp._encode_source_heads(
        source=SimpleNamespace(path="s.mp4", context_frames=9),
        layout=SimpleNamespace(n_ctx_v=N_CTX_V, n_ctx_a=N_CTX_A),
        width=16, height=16, video_encoder=encoder, tiling_cfg=None,
        ledger=SimpleNamespace(audio_encoder=lambda: object()),
        device=torch.device("cpu"),
    )
    assert out[4] is True and out[3] == N_CTX_A
    assert len(seen) == 1
    return seen[0]


def test_mono_source_is_duplicated_to_stereo(monkeypatch):
    mono = torch.linspace(-1.0, 1.0, N_SAMPLES).reshape(1, 1, N_SAMPLES)
    wf = _encode_with(monkeypatch, mono)
    assert tuple(wf.shape) == (1, 2, N_SAMPLES)
    assert torch.equal(wf[:, 0], wf[:, 1])
    assert torch.equal(wf[:, 0], mono[:, 0].to(wf.dtype))


def test_stereo_source_passes_through(monkeypatch):
    stereo = torch.stack(
        [torch.linspace(-1.0, 1.0, N_SAMPLES), torch.linspace(1.0, -1.0, N_SAMPLES)]
    ).unsqueeze(0)
    wf = _encode_with(monkeypatch, stereo)
    assert tuple(wf.shape) == (1, 2, N_SAMPLES)
    assert torch.equal(wf, stereo.to(wf.dtype))
