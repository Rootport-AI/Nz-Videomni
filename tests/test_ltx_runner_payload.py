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
from services.lora_registry import ResolvedLora
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


# ─────────────────────────────────────────────────────────────────────────────
# audio_strength (Style LoRA音声強度制御, WP3) — the worker-payload lora entry
# only carries "audio_strength" when the resolved lora has one. Legacy plain
# 3-tuples (no such attribute at all) prove the G-BC no-key contract; a
# ResolvedLora with audio_strength=0.0 proves 0.0 (falsy but not None) survives.
# ─────────────────────────────────────────────────────────────────────────────


def test_generate_payload_omits_audio_strength_for_legacy_3tuple(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _lora_request()
    be.generate(
        req,
        tmp_path / "out",
        lora_paths=[(tmp_path / "adapter.safetensors", 1.0, "none")],
    )
    assert "audio_strength" not in captured[0]["loras"][0]


def test_generate_payload_carries_audio_strength_zero(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _lora_request()
    be.generate(
        req,
        tmp_path / "out",
        lora_paths=[
            ResolvedLora(tmp_path / "adapter.safetensors", 1.0, "none", 0.0)
        ],
    )
    assert captured[0]["loras"][0]["audio_strength"] == 0.0


def test_chain_payload_omits_audio_strength_for_legacy_3tuple(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _chain_request()
    be.generate_chain(
        req,
        tmp_path / "out",
        lora_paths=[(tmp_path / "adapter.safetensors", 1.0, "none")],
    )
    assert "audio_strength" not in captured[0]["loras"][0]


def test_chain_payload_carries_audio_strength_zero(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _chain_request()
    be.generate_chain(
        req,
        tmp_path / "out",
        lora_paths=[
            ResolvedLora(tmp_path / "adapter.safetensors", 1.0, "none", 0.0)
        ],
    )
    assert captured[0]["loras"][0]["audio_strength"] == 0.0


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
# Contract: the key appears ONLY for a non-default backend, and the MOCK
# request field (vae_mode) never reaches the wire at all — an inert payload key
# would be the "displayed but not applied" trap.
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
    # The mock field set to a NON-default value, plus sage so the acceleration
    # branch definitely runs: the mock field must still be absent everywhere in
    # the payload (it is accepted by the API but never consumed by the engine).
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _nag_request(
        attention_backend="sage",
        vae_mode="prune_vaed",
    )
    be.generate(req, tmp_path / "out")
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
        vae_mode="prune_vaed",
    )
    be.generate_chain(req, tmp_path / "out")
    assert "vae_mode" not in captured[0]
    assert captured[0]["attention_backend"] == "sage"


def test_default_payload_key_set_is_unchanged_by_acceleration(tmp_path):
    # Byte-identical contract, stated positively: a fully-default request's key
    # set must be exactly the pre-acceleration one, MODULO the two fields whose
    # own pydantic default was later flipped to True once their real-device
    # gate passed -- block_swap_prefetch (S4, 2026-08-01) and
    # fused_gguf_dequant_kernel (§51, 2026-08-04). A fully-default request
    # therefore carries both (value True). attention_backend/vae_mode still
    # default to their pre-acceleration values and stay absent.
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be.generate(_nag_request(), tmp_path / "single")
    assert set(captured[0]) == {
        "op", "prompt", "seed", "height", "width", "num_frames", "frame_rate",
        "num_steps", "images", "loras", "reference_video", "output_path",
        "block_swap_prefetch", "fused_gguf_dequant_kernel",
    }
    assert captured[0]["block_swap_prefetch"] is True
    assert captured[0]["fused_gguf_dequant_kernel"] is True

    captured_chain: list[dict] = []
    be2 = _capturing_backend(captured_chain)
    be2.generate_chain(_chain_request(), tmp_path / "chain")
    assert set(captured_chain[0]) == {
        "op", "width", "height", "frame_rate", "num_steps", "seed",
        "overlap_frames", "overlap_strength", "chunked_upsample",
        "output_path", "clips", "block_swap_prefetch",
        "fused_gguf_dequant_kernel",
    }
    assert captured_chain[0]["block_swap_prefetch"] is True
    assert captured_chain[0]["fused_gguf_dequant_kernel"] is True


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


# ─────────────────────────────────────────────────────────────────────────────
# Acceleration — conditional ``block_swap_prefetch`` worker-payload key
# (S2 API wiring; backend §44). The key appears ONLY when the RESOLVED
# ``GenerateRequest``/``GenerateChainRequest`` field is truthy -- this is
# already "the true intended value", not a comparison against a default, so
# it stays correct across S4's default flip (2026-08-01, api/models.py):
# omitting the field on a request now resolves it to True via pydantic, which
# this payload builder forwards unchanged; explicitly requesting False still
# omits the key, and the worker's own missing-key fallback is a HARDCODED
# False (services/ltx_runner_worker's ``_resolve_block_swap_prefetch``,
# independent of the API's default) so that omission still means "off" on the
# worker side. Unlike attention_backend this is a pure transfer-mechanism
# switch (no "changes generated bytes" caveat).
# ─────────────────────────────────────────────────────────────────────────────


def test_generate_payload_carries_block_swap_prefetch_when_true(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _nag_request(block_swap_prefetch=True)
    be.generate(req, tmp_path / "out")
    assert captured[0]["block_swap_prefetch"] is True


def test_generate_payload_carries_block_swap_prefetch_by_default(tmp_path):
    # S4: GenerateRequest's own pydantic default flipped to True, so a request
    # that never touches the field forwards True to the worker.
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _nag_request()  # block_swap_prefetch omitted -> pydantic default True
    be.generate(req, tmp_path / "out")
    assert captured[0]["block_swap_prefetch"] is True


def test_generate_payload_omits_block_swap_prefetch_when_explicitly_off(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _nag_request(block_swap_prefetch=False)
    be.generate(req, tmp_path / "out")
    assert "block_swap_prefetch" not in captured[0]


def test_generate_payload_block_swap_prefetch_key_is_last(tmp_path):
    # Key-order contract (WORKORDER §11.3): block_swap_prefetch is appended
    # after attention_backend in the payload dict.
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _nag_request(attention_backend="sage", block_swap_prefetch=True)
    be.generate(req, tmp_path / "out")
    keys = list(captured[0].keys())
    assert keys.index("attention_backend") < keys.index("block_swap_prefetch")


def test_chain_payload_carries_block_swap_prefetch_when_true(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _chain_request(block_swap_prefetch=True)
    be.generate_chain(req, tmp_path / "out")
    assert captured[0]["block_swap_prefetch"] is True


def test_chain_payload_carries_block_swap_prefetch_by_default(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _chain_request()  # block_swap_prefetch omitted -> pydantic default True
    be.generate_chain(req, tmp_path / "out")
    assert captured[0]["block_swap_prefetch"] is True


def test_chain_payload_omits_block_swap_prefetch_when_explicitly_off(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _chain_request(block_swap_prefetch=False)
    be.generate_chain(req, tmp_path / "out")
    assert "block_swap_prefetch" not in captured[0]


def test_chain_payload_block_swap_prefetch_key_is_last(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _chain_request(attention_backend="sage", block_swap_prefetch=True)
    be.generate_chain(req, tmp_path / "out")
    keys = list(captured[0].keys())
    assert keys.index("attention_backend") < keys.index("block_swap_prefetch")


def test_block_swap_prefetch_used_and_peak_reserved_relayed_from_done_event(tmp_path):
    # Mirrors test_attention_used_is_relayed_from_the_done_event: both new
    # GenerationOutcome fields ride the terminal done event on the same route
    # as seed_used/attention_used, additively (peak_vram_mb untouched).
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be._read_worker_events = (  # type: ignore[attr-defined]
        lambda cb, chain, prefix: {
            "event": "done", "seed_used": 7, "peak_vram_mb": 100,
            "attention_used": "sdpa",
            "block_swap_prefetch_used": "on->off",
            "peak_vram_reserved_mb": 8192,
        }
    )
    outcome = be.generate(
        _nag_request(block_swap_prefetch=True), tmp_path / "out"
    )
    assert outcome.block_swap_prefetch_used == "on->off"
    assert outcome.peak_vram_reserved_mb == 8192
    assert outcome.peak_vram_mb == 100  # unaffected, additive-only

    # Chain path relays it too, and a worker that never reports it -> None.
    be2 = _capturing_backend([])
    be2._read_worker_events = (  # type: ignore[attr-defined]
        lambda cb, chain, prefix: {"event": "done", "seed_used": 7}
    )
    outcome2 = be2.generate_chain(_chain_request(), tmp_path / "out2")
    assert outcome2.block_swap_prefetch_used is None
    assert outcome2.peak_vram_reserved_mb is None


# ─────────────────────────────────────────────────────────────────────────────
# keep_resident (cross-job CPU-skeleton cache, §48) — conditional worker-payload
# key. Structurally identical to block_swap_prefetch above BUT the default is
# OFF, so "send only when the resolved field is truthy" means the key rides only
# on an explicit opt-in and a fully-default request is byte-identical to
# pre-keep_resident. The worker's missing-key fallback is likewise False, and
# there it doubles as the explicit "free the ~20GB cache" trigger.
# ─────────────────────────────────────────────────────────────────────────────


def test_generate_payload_carries_keep_resident_when_true(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be.generate(_nag_request(keep_resident=True), tmp_path / "out")
    assert captured[0]["keep_resident"] is True


def test_generate_payload_omits_keep_resident_by_default(tmp_path):
    # Opposite direction from block_swap_prefetch: the pydantic default is
    # False, so an untouched request must NOT carry the key at all.
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be.generate(_nag_request(), tmp_path / "out")
    assert "keep_resident" not in captured[0]

    # ...and an explicit False is the same wire shape as omitting it (which is
    # what makes "absent == off == release the cache" a single rule).
    captured_off: list[dict] = []
    be_off = _capturing_backend(captured_off)
    be_off.generate(_nag_request(keep_resident=False), tmp_path / "out_off")
    assert "keep_resident" not in captured_off[0]


def test_generate_payload_keep_resident_key_is_last(tmp_path):
    # Key-order contract: appended after block_swap_prefetch, which is itself
    # after attention_backend.
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _nag_request(
        attention_backend="sage", block_swap_prefetch=True, keep_resident=True
    )
    be.generate(req, tmp_path / "out")
    keys = list(captured[0].keys())
    assert keys.index("attention_backend") < keys.index("block_swap_prefetch")
    assert keys.index("block_swap_prefetch") < keys.index("keep_resident")


def test_chain_payload_carries_keep_resident_when_true(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be.generate_chain(_chain_request(keep_resident=True), tmp_path / "out")
    assert captured[0]["keep_resident"] is True
    keys = list(captured[0].keys())
    assert keys.index("block_swap_prefetch") < keys.index("keep_resident")


def test_chain_payload_omits_keep_resident_by_default(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be.generate_chain(_chain_request(), tmp_path / "out")
    assert "keep_resident" not in captured[0]


def test_default_payload_key_set_is_unchanged_by_keep_resident(tmp_path):
    # The regression contract, stated positively and INDEPENDENTLY of the
    # feature's own tests: adding keep_resident must not have grown the
    # default-request key set by even one key.
    #
    # Two entries here are NOT "adding a feature grew the key set" but "a
    # server default was deliberately flipped to on after its real-device gate
    # passed", which is the only sanctioned way this set may grow:
    #   * block_swap_prefetch -- flipped 2026-08-01 (§44 S4)
    #   * fused_gguf_dequant_kernel -- flipped 2026-08-04 (§51, gates G1-G8)
    # Every other feature must still leave a default request byte-identical.
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be.generate(_nag_request(), tmp_path / "single")
    assert set(captured[0]) == {
        "op", "prompt", "seed", "height", "width", "num_frames", "frame_rate",
        "num_steps", "images", "loras", "reference_video", "output_path",
        "block_swap_prefetch", "fused_gguf_dequant_kernel",
    }

    captured_chain: list[dict] = []
    be2 = _capturing_backend(captured_chain)
    be2.generate_chain(_chain_request(), tmp_path / "chain")
    assert set(captured_chain[0]) == {
        "op", "width", "height", "frame_rate", "num_steps", "seed",
        "overlap_frames", "overlap_strength", "chunked_upsample",
        "output_path", "clips", "block_swap_prefetch",
        "fused_gguf_dequant_kernel",
    }


def test_keep_resident_used_is_relayed_from_the_done_event(tmp_path):
    # metadata.json's keep_resident_used is the ONLY machine-readable way to
    # tell "asked for it" from "it actually ran" (the real-device gate judges
    # on this field), so the relay gets its own guard on both paths.
    be = _capturing_backend([])
    be._read_worker_events = (  # type: ignore[attr-defined]
        lambda cb, chain, prefix: {
            "event": "done", "seed_used": 7, "peak_vram_mb": 100,
            "keep_resident_used": "on->off",
        }
    )
    outcome = be.generate(_nag_request(keep_resident=True), tmp_path / "out")
    assert outcome.keep_resident_used == "on->off"

    be2 = _capturing_backend([])
    be2._read_worker_events = (  # type: ignore[attr-defined]
        lambda cb, chain, prefix: {
            "event": "done", "seed_used": 7, "keep_resident_used": "on",
        }
    )
    outcome2 = be2.generate_chain(
        _chain_request(keep_resident=True), tmp_path / "out2"
    )
    assert outcome2.keep_resident_used == "on"

    # A worker predating the field -> None (never a fabricated "off").
    be3 = _capturing_backend([])
    be3._read_worker_events = (  # type: ignore[attr-defined]
        lambda cb, chain, prefix: {"event": "done", "seed_used": 7}
    )
    outcome3 = be3.generate(_nag_request(), tmp_path / "out3")
    assert outcome3.keep_resident_used is None


# ─────────────────────────────────────────────────────────────────────────────
# fused_gguf_dequant_kernel (fused Triton GGUF dequantization, §1-11) —
# conditional worker-payload key. Since the 2026-08-04 default flip (§51: gates
# G1-G8 passed, owner approved) the shape is block_swap_prefetch's, NOT
# keep_resident's: the server default is ON, so the key rides on a DEFAULT
# request too and only disappears when the caller explicitly turns it off. The
# worker's missing-key fallback is still False (absent == off).
# ─────────────────────────────────────────────────────────────────────────────


def test_generate_payload_carries_fused_dequant_when_true(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be.generate(_nag_request(fused_gguf_dequant_kernel=True), tmp_path / "out")
    assert captured[0]["fused_gguf_dequant_kernel"] is True
    # Key-order contract: appended after keep_resident, which is itself after
    # block_swap_prefetch.
    captured_all: list[dict] = []
    be_all = _capturing_backend(captured_all)
    be_all.generate(
        _nag_request(
            attention_backend="sage",
            block_swap_prefetch=True,
            keep_resident=True,
            fused_gguf_dequant_kernel=True,
        ),
        tmp_path / "out_all",
    )
    keys = list(captured_all[0].keys())
    assert keys.index("keep_resident") < keys.index("fused_gguf_dequant_kernel")


def test_generate_payload_carries_fused_dequant_by_default(tmp_path):
    # The 2026-08-04 flip: GenerateRequest's own pydantic default is True, so a
    # request that never touches the field forwards True to the worker.
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be.generate(_nag_request(), tmp_path / "out")
    assert captured[0]["fused_gguf_dequant_kernel"] is True


def test_generate_payload_omits_fused_dequant_when_explicitly_off(tmp_path):
    captured_off: list[dict] = []
    be_off = _capturing_backend(captured_off)
    be_off.generate(
        _nag_request(fused_gguf_dequant_kernel=False), tmp_path / "out_off"
    )
    assert "fused_gguf_dequant_kernel" not in captured_off[0]


def test_chain_payload_carries_fused_dequant_when_true(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be.generate_chain(
        _chain_request(keep_resident=True, fused_gguf_dequant_kernel=True),
        tmp_path / "out",
    )
    assert captured[0]["fused_gguf_dequant_kernel"] is True
    keys = list(captured[0].keys())
    assert keys.index("keep_resident") < keys.index("fused_gguf_dequant_kernel")


def test_chain_payload_carries_fused_dequant_by_default(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be.generate_chain(_chain_request(), tmp_path / "out")
    assert captured[0]["fused_gguf_dequant_kernel"] is True


def test_chain_payload_omits_fused_dequant_when_explicitly_off(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be.generate_chain(
        _chain_request(fused_gguf_dequant_kernel=False), tmp_path / "out"
    )
    assert "fused_gguf_dequant_kernel" not in captured[0]


def test_fused_dequant_used_is_relayed_from_the_done_event(tmp_path):
    # Same relay discipline as keep_resident_used: metadata.json's
    # fused_gguf_dequant_kernel_used is the judging criterion for the G1-G8
    # real-device gates, so the relay gets its own guard on both paths.
    be = _capturing_backend([])
    be._read_worker_events = (  # type: ignore[attr-defined]
        lambda cb, chain, prefix: {
            "event": "done", "seed_used": 7, "peak_vram_mb": 100,
            "fused_gguf_dequant_kernel_used": "on->off",
        }
    )
    outcome = be.generate(
        _nag_request(fused_gguf_dequant_kernel=True), tmp_path / "out"
    )
    assert outcome.fused_gguf_dequant_kernel_used == "on->off"

    be2 = _capturing_backend([])
    be2._read_worker_events = (  # type: ignore[attr-defined]
        lambda cb, chain, prefix: {
            "event": "done", "seed_used": 7,
            "fused_gguf_dequant_kernel_used": "on",
        }
    )
    outcome2 = be2.generate_chain(
        _chain_request(fused_gguf_dequant_kernel=True), tmp_path / "out2"
    )
    assert outcome2.fused_gguf_dequant_kernel_used == "on"

    # A worker predating the field -> None (never a fabricated "off").
    be3 = _capturing_backend([])
    be3._read_worker_events = (  # type: ignore[attr-defined]
        lambda cb, chain, prefix: {"event": "done", "seed_used": 7}
    )
    outcome3 = be3.generate(_nag_request(), tmp_path / "out3")
    assert outcome3.fused_gguf_dequant_kernel_used is None
