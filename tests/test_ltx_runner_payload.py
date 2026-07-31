"""S1: LTXRunner (_RealBackend) worker-payload construction — torch-free.

The mock backend bypasses ltx_runner entirely, so the IC-LoRA control-
adjustability payload keys (``reference_video.strength`` /
``reference_video.attention_strength``) are covered directly here: a
``_RealBackend`` is built via ``__new__`` (no worker subprocess), its ``_send``
is stubbed to capture the outgoing payload dict (and drop a stub output file so
the post-send existence check passes), and ``_read_worker_events`` is stubbed to
return the terminal ``done`` event. Mirrors the ``_RealBackend.__new__`` harness
in test_step_progress.py.
"""

from __future__ import annotations

import threading
import types
from pathlib import Path

from api.models import GenerateChainRequest, GenerateRequest
from services.ltx_runner import _RealBackend

REGISTERED_LORA = "pixel-spatial-upscaler-x2"


def _capturing_backend(captured: list[dict]) -> _RealBackend:
    be = _RealBackend.__new__(_RealBackend)
    # ``loaded`` is a read-only property over ``_proc.poll()``; a live-looking
    # fake proc makes it True so generate() skips load() (no subprocess spawn).
    be._proc = types.SimpleNamespace(poll=lambda: None)  # type: ignore[attr-defined]
    be._lock = threading.Lock()

    def _send(msg: dict) -> None:
        captured.append(msg)
        # generate() checks the output exists after the send; write a stub.
        out = Path(msg["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00" * 16)

    be._send = _send  # type: ignore[attr-defined]
    be._read_worker_events = (  # type: ignore[attr-defined]
        lambda cb, chain, prefix: {"event": "done", "seed_used": 7, "peak_vram_mb": 100}
    )
    return be


def _lora_request(**over) -> GenerateRequest:
    base = dict(
        prompt="a busy town street",
        width=512,
        height=256,
        num_frames=17,
        frame_rate=24.0,
        num_inference_steps=8,
        guidance_scale=1.0,
        seed=99,
        pipeline="distilled",
        loras=[{"name": REGISTERED_LORA, "strength": 1.0}],
        reference_video_id="vid",
    )
    base.update(over)
    return GenerateRequest(**base)


def test_payload_carries_strength_and_attention_when_set(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _lora_request(
        conditioning_attention_strength=0.6,
        reference_video_strength=0.8,
    )
    be.generate(
        req,
        tmp_path / "out",
        lora_paths=[(tmp_path / "adapter.safetensors", 1.0, "none")],
        reference_video_path=tmp_path / "ref.mp4",
    )
    ref = captured[0]["reference_video"]
    assert ref["strength"] == 0.8
    assert ref["attention_strength"] == 0.6


def test_payload_defaults_when_fields_omitted(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _lora_request()  # no control-adjustability fields
    be.generate(
        req,
        tmp_path / "out",
        lora_paths=[(tmp_path / "adapter.safetensors", 1.0, "none")],
        reference_video_path=tmp_path / "ref.mp4",
    )
    ref = captured[0]["reference_video"]
    assert ref["strength"] == 1.0  # byte-identical default
    assert "attention_strength" not in ref  # key entirely absent


def _chain_request(**over) -> GenerateChainRequest:
    base = dict(
        prompt="a serene mountain lake at dawn",
        width=384,
        height=256,
        frame_rate=24.0,
        num_inference_steps=8,
        guidance_scale=1.0,
        seed=123,
        pipeline="distilled",
        overlap_frames=2,
        overlap_strength=0.5,
        clips=[{"num_frames": 25}, {"num_frames": 25}],
    )
    base.update(over)
    return GenerateChainRequest(**base)


def test_chain_payload_carries_chunked_upsample_true(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _chain_request(chunked_upsample=True)
    be.generate_chain(req, tmp_path / "out")
    # Always-present key (like overlap_frames/overlap_strength), correct bool.
    assert captured[0]["chunked_upsample"] is True


def test_chain_payload_chunked_upsample_false_by_default(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _chain_request()  # flag omitted -> default False
    be.generate_chain(req, tmp_path / "out")
    assert captured[0]["chunked_upsample"] is False


# ─────────────────────────────────────────────────────────────────────────────
# NAG (Normalized Attention Guidance) — additive worker-payload block.
# ─────────────────────────────────────────────────────────────────────────────


def _nag_request(**over) -> GenerateRequest:
    base = dict(
        prompt="a busy town street",
        width=512,
        height=256,
        num_frames=17,
        frame_rate=24.0,
        num_inference_steps=8,
        guidance_scale=1.0,
        seed=99,
        pipeline="distilled",
    )
    base.update(over)
    return GenerateRequest(**base)


def test_generate_payload_carries_nag_when_enabled(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _nag_request(
        nag_enabled=True, negative_prompt="blurry, low quality",
        nag_scale=11.0, nag_tau=2.5, nag_alpha=0.25,
    )
    be.generate(req, tmp_path / "out")
    assert captured[0]["nag"] == {
        "negative_prompt": "blurry, low quality",
        "scale": 11.0,
        "tau": 2.5,
        "alpha": 0.25,
        "method": "nag",
        "vsf_scale": 1.5,
    }


def test_generate_payload_omits_nag_by_default(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _nag_request()  # nag_enabled omitted -> False
    be.generate(req, tmp_path / "out")
    assert "nag" not in captured[0]


def test_chain_payload_carries_nag_when_enabled(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _chain_request(nag_enabled=True, negative_prompt="blurry, low quality")
    be.generate_chain(req, tmp_path / "out")
    assert captured[0]["nag"] == {
        "negative_prompt": "blurry, low quality",
        "scale": 11.0,
        "tau": 2.5,
        "alpha": 0.25,
        "method": "nag",
        "vsf_scale": 1.5,
    }


def test_chain_payload_omits_nag_by_default(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _chain_request()  # nag_enabled omitted -> False
    be.generate_chain(req, tmp_path / "out")
    assert "nag" not in captured[0]


# ─────────────────────────────────────────────────────────────────────────────
# Acceleration — conditional ``attention_backend`` worker-payload key.
#
# Contract: the key appears ONLY for a non-default backend, and the two MOCK
# request fields (fused_gguf_dequant_gemm / vae_mode) never reach the wire at
# all — an inert payload key would be the "displayed but not applied" trap.
# ─────────────────────────────────────────────────────────────────────────────


def test_generate_payload_carries_attention_backend_when_sage(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _nag_request(attention_backend="sage")
    be.generate(req, tmp_path / "out")
    assert captured[0]["attention_backend"] == "sage"


def test_generate_payload_omits_attention_backend_by_default(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _nag_request()  # attention_backend omitted -> "sdpa"
    be.generate(req, tmp_path / "out")
    assert "attention_backend" not in captured[0]


def test_generate_payload_never_carries_mock_acceleration_fields(tmp_path):
    # Both mock fields set to NON-default values, plus sage so the acceleration
    # branch definitely runs: the mock fields must still be absent everywhere in
    # the payload (they are accepted by the API but never consumed by the engine).
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _nag_request(
        attention_backend="sage",
        fused_gguf_dequant_gemm=True,
        vae_mode="prune_vaed",
    )
    be.generate(req, tmp_path / "out")
    assert "fused_gguf_dequant_gemm" not in captured[0]
    assert "vae_mode" not in captured[0]
    assert captured[0]["attention_backend"] == "sage"  # the real one did go


def test_chain_payload_carries_attention_backend_when_sage(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _chain_request(attention_backend="sage")
    be.generate_chain(req, tmp_path / "out")
    assert captured[0]["attention_backend"] == "sage"


def test_chain_payload_omits_attention_backend_by_default(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _chain_request()  # attention_backend omitted -> "sdpa"
    be.generate_chain(req, tmp_path / "out")
    assert "attention_backend" not in captured[0]


def test_chain_payload_never_carries_mock_acceleration_fields(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _chain_request(
        attention_backend="sage",
        fused_gguf_dequant_gemm=True,
        vae_mode="prune_vaed",
    )
    be.generate_chain(req, tmp_path / "out")
    assert "fused_gguf_dequant_gemm" not in captured[0]
    assert "vae_mode" not in captured[0]
    assert captured[0]["attention_backend"] == "sage"


def test_default_payload_key_set_is_unchanged_by_acceleration(tmp_path):
    # Byte-identical contract, stated positively: a fully-default request's key
    # set must be exactly the pre-acceleration one (no acceleration key at all).
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be.generate(_nag_request(), tmp_path / "single")
    assert set(captured[0]) == {
        "op", "prompt", "seed", "height", "width", "num_frames", "frame_rate",
        "num_steps", "images", "loras", "reference_video", "output_path",
    }

    captured_chain: list[dict] = []
    be2 = _capturing_backend(captured_chain)
    be2.generate_chain(_chain_request(), tmp_path / "chain")
    assert set(captured_chain[0]) == {
        "op", "width", "height", "frame_rate", "num_steps", "seed",
        "overlap_frames", "overlap_strength", "chunked_upsample",
        "output_path", "clips",
    }


def test_attention_used_is_relayed_from_the_done_event(tmp_path):
    # The engine's ACTUAL attention backend rides the terminal done event on the
    # same route as seed_used, so a sage->sdpa degrade is visible downstream
    # (pipeline_manager writes it to metadata.json).
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be._read_worker_events = (  # type: ignore[attr-defined]
        lambda cb, chain, prefix: {
            "event": "done", "seed_used": 7, "peak_vram_mb": 100,
            "attention_used": "sage->sdpa",
        }
    )
    outcome = be.generate(_nag_request(attention_backend="sage"), tmp_path / "out")
    assert outcome.attention_used == "sage->sdpa"

    # Chain path relays it too, and a worker that never reports it -> None.
    be2 = _capturing_backend([])
    be2._read_worker_events = (  # type: ignore[attr-defined]
        lambda cb, chain, prefix: {"event": "done", "seed_used": 7}
    )
    outcome2 = be2.generate_chain(_chain_request(), tmp_path / "out2")
    assert outcome2.attention_used is None
