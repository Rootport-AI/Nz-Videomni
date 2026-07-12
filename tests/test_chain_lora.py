"""Chain LoRA wiring — /generate/chain style-LoRA support (A2V+LoRA unblock).

Additive to the frozen chain contract: ``GenerateChainRequest.loras`` (same
``LoraSpec`` type as ``GenerateRequest.loras``) applies style/character adapters
uniformly across the whole chain. Control (reference-video) adapters are out of
chain scope and rejected up front. A chain omitting ``loras`` is byte-identical
to before.

Coverage:
  (a) schema — ``loras`` defaults empty, accepts specs, path-like name rejected;
  (b) endpoint — a control adapter on a chain is 422; a style adapter completes;
  (c) plumbing — the mock e2e records ``loras`` in the request dump, and the real
      backend puts a ``loras`` block on the worker payload (absent when empty);
  (d) stale-clear — ``chain_pipeline.run_chain`` calls ``pipe._set_ic_job`` (with
      THIS chain's loras, or [] to clear) BEFORE the transformer is built, so a
      prior single ``generate()``'s LoRA cannot leak into the chain denoise.

The mock backend (conftest ``client``) has no weights, so the forward-time patch
itself is not exercised here — (d) pins the app/engine wiring GPU-free; the real
weight effect is an owner-witnessed GPU check per project rule.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import types
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main
from api.models import GenerateChainRequest
from services.ltx_runner import _RealBackend

STYLE_LORA = "style-adapter"
CANNY_LORA = "canny-control"


def _make_args(config_path: str) -> argparse.Namespace:
    return argparse.Namespace(
        listen=False, port=None, api_key=None, allow_all_cors=False,
        config=config_path, te_offload=None, dit_cpu_load=None,
    )


@pytest.fixture()
def chain_lora_client(tmp_path):
    """A mock-backend client registering one STYLE adapter (metadata-less dummy
    -> kind "style") and one CONTROL adapter (dict entry w/ preprocess canny ->
    kind "control"). The registry checks the file exists; the mock never reads
    it."""
    style_file = tmp_path / "style.safetensors"
    style_file.write_bytes(b"\x00" * 8)
    control_file = tmp_path / "union-control.safetensors"
    control_file.write_bytes(b"\x00" * 8)

    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {
            "backend": "mock",
            "ic_loras": {
                STYLE_LORA: style_file.as_posix(),  # legacy string -> style
                CANNY_LORA: {"path": control_file.as_posix(), "preprocess": "canny"},
            },
        },
        "output": {"dir": (tmp_path / "outputs").as_posix()},
        "upload": {"dir": (tmp_path / "uploads").as_posix()},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    app = main.build_app(_make_args(cfg_path.as_posix()))
    with TestClient(app) as c:
        c.app_context = app.state.context  # type: ignore[attr-defined]
        yield c


BASE = {
    "prompt": "a bustling town square at dusk, cinematic trailer",
    "width": 384,
    "height": 256,
    "frame_rate": 24.0,
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "seed": 123,
    "pipeline": "distilled",
    "overlap_frames": 2,
    "overlap_strength": 0.5,
}


def _run_chain(client, clips, **overrides):
    payload = {**BASE, "clips": clips, **overrides}
    return client.post("/api/v1/generate/chain", json=payload)


# ------------------------------------------------------------------ (a) schema


def _chain_kwargs(**over):
    base = dict(
        prompt="p", width=384, height=256,
        clips=[{"num_frames": 25}, {"num_frames": 25}],
    )
    base.update(over)
    return base


def test_chain_loras_defaults_empty():
    req = GenerateChainRequest(**_chain_kwargs())
    assert req.loras == []


def test_chain_loras_accepts_specs():
    req = GenerateChainRequest(
        **_chain_kwargs(loras=[{"name": "style-x", "strength": 0.8}])
    )
    assert len(req.loras) == 1
    assert req.loras[0].name == "style-x"
    assert req.loras[0].strength == 0.8


def test_chain_lora_path_like_name_rejected():
    with pytest.raises(ValidationError):
        GenerateChainRequest(
            **_chain_kwargs(loras=[{"name": "../../etc/passwd", "strength": 1.0}])
        )


def test_chain_loras_in_request_dump_roundtrips():
    req = GenerateChainRequest(
        **_chain_kwargs(loras=[{"name": "style-x", "strength": 0.8}])
    )
    dumped = req.model_dump()
    assert dumped["loras"] == [{"name": "style-x", "strength": 0.8}]


# --------------------------------------------------------------- (b) endpoint


def test_chain_control_lora_rejected_422(chain_lora_client):
    """A CONTROL adapter on a MULTI-CLIP chain (clips>=2) is still rejected up
    front: reference-video IC-LoRA conditioning is only wired into stage-1's
    clip-0, so a chain of 2+ clips carrying a control adapter is unsupported
    regardless of whether ``reference_video_id`` is present."""
    r = _run_chain(
        chain_lora_client,
        [{"num_frames": 25}, {"num_frames": 25}],
        loras=[{"name": CANNY_LORA, "strength": 1.0}],
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "LORA_CONTROL_UNSUPPORTED_IN_CHAIN"


def test_chain_unknown_lora_404(chain_lora_client):
    r = _run_chain(
        chain_lora_client,
        [{"num_frames": 25}, {"num_frames": 25}],
        loras=[{"name": "no-such-adapter", "strength": 1.0}],
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "LORA_NOT_FOUND"


def test_chain_style_lora_completes_and_records(chain_lora_client):
    """A STYLE adapter on a chain completes end to end (mock) and the additive
    ``loras`` field round-trips into the metadata request dump."""
    r = _run_chain(
        chain_lora_client,
        [{"num_frames": 25}, {"num_frames": 25}],
        loras=[{"name": STYLE_LORA, "strength": 0.9}],
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    job = chain_lora_client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    ctx = chain_lora_client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    assert meta["request"]["loras"][0]["name"] == STYLE_LORA
    assert meta["request"]["loras"][0]["strength"] == 0.9


def test_chain_style_lora_with_source_audio_completes(chain_lora_client, tmp_path):
    """A2V + LoRA together (the whole point of this change): a single-clip chain
    with source_audio AND a style LoRA is accepted (no exclusivity guard)."""
    import struct
    import wave

    wav = tmp_path / "voice.wav"
    n = int(round(3.0 * 16000))
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<%dh" % n, *([0] * n)))
    up = chain_lora_client.post(
        "/api/v1/upload/audio",
        files={"file": ("voice.wav", wav.read_bytes(), "audio/wav")},
    )
    assert up.status_code == 200, up.text
    aid = up.json()["audio_id"]

    r = _run_chain(
        chain_lora_client,
        [{"num_frames": 49}],
        source_audio={"audio_id": aid},
        loras=[{"name": STYLE_LORA, "strength": 1.0}],
    )
    assert r.status_code == 202, r.text
    job = chain_lora_client.get(f"/api/v1/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "completed", job


def test_chain_without_loras_regression(chain_lora_client):
    """A chain omitting ``loras`` defaults to an empty list — byte-shape identical
    to before (no ic_lora block; loras == [] in the request dump)."""
    r = _run_chain(chain_lora_client, [{"num_frames": 25}, {"num_frames": 25}])
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = chain_lora_client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    ctx = chain_lora_client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    assert meta["request"]["loras"] == []


# ---------------------------------------------------- (c) real-backend payload


def _capturing_backend(captured: list[dict]) -> _RealBackend:
    """A _RealBackend built via __new__ (no subprocess); _send captures the
    outgoing payload and drops a stub output file, _read_chain_events returns a
    terminal done. Mirrors tests/test_ltx_runner_payload.py."""
    be = _RealBackend.__new__(_RealBackend)
    be._proc = types.SimpleNamespace(poll=lambda: None)  # type: ignore[attr-defined]
    be._lock = threading.Lock()

    def _send(msg: dict) -> None:
        captured.append(msg)
        out = Path(msg["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00" * 16)

    be._send = _send  # type: ignore[attr-defined]
    be._read_chain_events = (  # type: ignore[attr-defined]
        lambda cb: {"event": "done", "seed_used": 7, "peak_vram_mb": 100, "chain": {}}
    )
    return be


def _chain_request(**over) -> GenerateChainRequest:
    return GenerateChainRequest(**_chain_kwargs(**over))


def test_chain_payload_carries_loras_when_set(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _chain_request(loras=[{"name": STYLE_LORA, "strength": 0.7}])
    be.generate_chain(
        req,
        tmp_path / "out",
        lora_paths=[(tmp_path / "style.safetensors", 0.7, "none")],
    )
    assert captured[0]["loras"] == [
        {"path": str(tmp_path / "style.safetensors"), "strength": 0.7}
    ]


def test_chain_payload_omits_loras_when_empty(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _chain_request()  # no loras
    be.generate_chain(req, tmp_path / "out")
    assert "loras" not in captured[0]  # byte-identical to the pre-LoRA payload


# ------------------------------------------------- (d) run_chain stale-clear


_LTX_STUB_MODULES = {
    "ltx_core": {},
    "ltx_core.components": {},
    "ltx_core.model": {},
    "ltx_core.text_encoders": {},
    "ltx_pipelines": {},
    "ltx_pipelines.utils": {},
    "ltx_core.components.diffusion_steps": {"EulerDiffusionStep": object},
    "ltx_core.components.noisers": {"GaussianNoiser": object},
    "ltx_core.model.audio_vae": {
        "decode_audio": lambda *a, **k: None,
        "encode_audio": lambda *a, **k: None,
    },
    "ltx_core.model.upsampler": {"upsample_video": lambda *a, **k: None},
    "ltx_core.model.video_vae": {"decode_video": lambda *a, **k: None},
    "ltx_core.text_encoders.gemma": {
        "encode_text": lambda te, prompts: [(object(), object())]
    },
    "ltx_core.types": {
        "AudioLatentShape": object,
        "VideoLatentShape": object,
        "VideoPixelShape": object,
        "Audio": object,
    },
    "ltx_pipelines.utils.constants": {
        "DISTILLED_SIGMA_VALUES": [1.0],
        "STAGE_2_DISTILLED_SIGMA_VALUES": [1.0],
    },
    "ltx_pipelines.utils.helpers": {"cleanup_memory": lambda *a, **k: None},
}


class _StopBeforeBuild(Exception):
    """Raised by the fake video_encoder() to halt run_chain right after the
    IC-LoRA state is set but before any heavy build."""


def _run_chain_capturing_set_ic_job(monkeypatch, ic_loras):
    """Drive engine.pipeline.chain_pipeline.run_chain far enough to observe the
    _set_ic_job call ordering, GPU-free, with ltx_core stubbed out. Returns the
    ordered call log."""
    torch = pytest.importorskip("torch")
    for name, attrs in _LTX_STUB_MODULES.items():
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)

    import engine.pipeline.chain_pipeline as cp

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(cp, "default_tiling_config", lambda **k: object())

    calls: list = []

    class FakeLedger:
        def text_encoder(self):
            calls.append("text_encoder")
            return object()

        def video_encoder(self):
            calls.append("video_encoder")
            raise _StopBeforeBuild()

        def transformer(self):
            calls.append("transformer")
            raise _StopBeforeBuild()

    class FakeDP:
        device = "cpu"
        model_ledger = FakeLedger()
        pipeline_components = object()

    class FakePipe:
        pipeline = FakeDP()
        _vae_spatial_tile_size = 0
        _vae_temporal_tile_size = 0

        def _set_ic_job(self, loras, ref, attn):
            calls.append(("set_ic_job", list(loras), ref, attn))

    clips = [
        cp.ChainClipSpec(prompt="a", num_frames=25, images=[]),
        cp.ChainClipSpec(prompt="b", num_frames=25, images=[]),
    ]
    with pytest.raises(_StopBeforeBuild):
        cp.run_chain(
            FakePipe(), clips=clips, width=384, height=256, frame_rate=24.0,
            num_steps=8, seed=1, overlap_frames=2, overlap_strength=0.5,
            output_path="x.mp4", ic_loras=ic_loras,
        )
    return calls


def test_run_chain_sets_ic_loras_before_transformer(monkeypatch):
    """The chain sets THIS chain's LoRAs via _set_ic_job BEFORE the transformer /
    video_encoder is built (so the forward-time patch reads them)."""
    calls = _run_chain_capturing_set_ic_job(
        monkeypatch, ic_loras=[("/p/a.safetensors", 0.7)]
    )
    set_calls = [c for c in calls if isinstance(c, tuple) and c[0] == "set_ic_job"]
    assert set_calls == [("set_ic_job", [("/p/a.safetensors", 0.7)], None, 1.0)]
    # ordering: _set_ic_job strictly precedes the first heavy ledger build.
    assert calls.index(set_calls[0]) < calls.index("video_encoder")


def test_run_chain_clears_stale_ic_loras_when_none(monkeypatch):
    """No loras -> _set_ic_job is STILL called with [] (explicit clear), so a
    prior single generate()'s LoRA cannot leak into the chain denoise."""
    calls = _run_chain_capturing_set_ic_job(monkeypatch, ic_loras=None)
    set_calls = [c for c in calls if isinstance(c, tuple) and c[0] == "set_ic_job"]
    assert set_calls == [("set_ic_job", [], None, 1.0)]
    assert calls.index(set_calls[0]) < calls.index("video_encoder")
