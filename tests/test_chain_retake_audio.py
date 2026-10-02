"""LTX 2.3 chain retake: the window's audio flags (§1-46).

``_encode_retake_window`` is run on CPU with fake encoders: the window video,
the decoded audio and the audio-VAE encode are all stand-ins, and
``cleanup_memory`` is patched out (the real one initialises CUDA).

``had_audio`` answers "did the window have an audio track"; ``rt_a`` is the
audio latents to freeze, or None when there is nothing to freeze.
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

F_TOTAL, A_TOTAL = 3, 10


@pytest.fixture
def run(monkeypatch):
    monkeypatch.setattr(helpers, "cleanup_memory", lambda: None)
    monkeypatch.setattr(cp, "load_video_conditioning_cpu", lambda **_k: torch.zeros(1, 3, 17, 8, 8))

    def _run(*, encoded_a: int | None, regenerate_audio: bool):
        if encoded_a is None:
            monkeypatch.setattr(mio, "decode_audio_from_file", lambda _p, _d: None)
        else:
            monkeypatch.setattr(
                mio, "decode_audio_from_file",
                lambda _p, _d: SimpleNamespace(waveform=torch.zeros(1, 2, 16000), sampling_rate=16000),
            )
            monkeypatch.setattr(
                avae, "encode_audio",
                lambda _audio, _enc, _x: torch.ones(1, 8, encoded_a, 16),
            )
        encoder = SimpleNamespace(tiled_encode=lambda _x, _cfg: torch.zeros(1, 128, F_TOTAL, 1, 1))
        return cp._encode_retake_window(
            retake=SimpleNamespace(path="w.mp4", regenerate_audio=regenerate_audio),
            layout=SimpleNamespace(retake_window_px=17, f_total=F_TOTAL, a_total=A_TOTAL),
            width=16, height=16, video_encoder=encoder, tiling_cfg=None,
            ledger=SimpleNamespace(audio_encoder=lambda: object()),
            device=torch.device("cpu"),
        )

    return _run


def test_short_encode_without_regenerate_keeps_had_audio(run):
    _v_half, _v_full, rt_a, had_audio, orig_wf, sr = run(encoded_a=A_TOTAL - 2, regenerate_audio=False)
    assert had_audio is True
    assert rt_a is None
    assert orig_wf is not None and tuple(orig_wf.shape) == (2, 16000)
    assert sr == 16000


def test_short_encode_with_regenerate_raises(run):
    with pytest.raises(ValueError, match="a_total"):
        run(encoded_a=A_TOTAL - 2, regenerate_audio=True)


def test_no_audio_track(run):
    _v_half, _v_full, rt_a, had_audio, orig_wf, _sr = run(encoded_a=None, regenerate_audio=False)
    assert had_audio is False
    assert rt_a is None
    assert orig_wf is None


def test_full_encode_is_cut_to_window(run):
    _v_half, _v_full, rt_a, had_audio, _orig_wf, _sr = run(encoded_a=A_TOTAL + 3, regenerate_audio=True)
    assert had_audio is True
    assert rt_a is not None and rt_a.shape[2] == A_TOTAL
