"""Retake — the worker payload and the mock/engine metadata contract.

Two things are pinned here that nothing else pins:

1. A chain WITHOUT retake must keep its exact payload key set. That set is
   already asserted twice in ``tests/test_ltx_runner_payload.py``; this file
   re-pins it INDEPENDENTLY (own literal, own fixture) so a future edit that
   "fixes" those two by adding the new key has to defeat a third, unrelated
   assertion as well.
2. The mock's ``chain.retake`` metadata must carry the SAME key set as the real
   engine's — minus ``freeze_proof``, which only exists where real latents do.
   A mock that invented a 0.0 freeze proof would forge a pass of the single
   claim this whole feature rests on.
"""

from __future__ import annotations

import ast
import threading
import types
from pathlib import Path

import pytest

from api.models import GenerateChainRequest
from services.ltx_runner import _RealBackend


# The payload key set for a chain that asked for nothing optional. Written out
# here as a literal on purpose — deriving it from the code under test would make
# this pin vacuous.
DEFAULT_CHAIN_PAYLOAD_KEYS = {
    "op", "width", "height", "frame_rate", "num_steps", "seed",
    "overlap_frames", "overlap_strength", "chunked_upsample",
    "output_path", "clips", "block_swap_prefetch",
    "fused_gguf_dequant_kernel",
}


def _backend(captured: list[dict]) -> _RealBackend:
    be = _RealBackend.__new__(_RealBackend)
    be._proc = types.SimpleNamespace(poll=lambda: None)  # type: ignore[attr-defined]
    be._lock = threading.Lock()

    def _send(msg: dict) -> None:
        captured.append(msg)
        out = Path(msg["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00" * 16)

    be._send = _send  # type: ignore[attr-defined]
    be._read_worker_events = (  # type: ignore[attr-defined]
        lambda cb, chain, prefix: {"event": "done", "seed_used": 7, "peak_vram_mb": 100}
    )
    return be


def _req(**over) -> GenerateChainRequest:
    base = dict(
        prompt="a quiet harbour at first light",
        width=384, height=256, frame_rate=24.0,
        num_inference_steps=8, guidance_scale=1.0, seed=123,
        pipeline="distilled", overlap_frames=2, overlap_strength=0.5,
        clips=[{"num_frames": 25}, {"num_frames": 25}],
    )
    base.update(over)
    return GenerateChainRequest(**base)


def _retake_req(**over) -> GenerateChainRequest:
    base = dict(
        clips=[{"num_frames": 169}],
        retake={"video_id": "vid-1", "window_start_sec": 2.0},
    )
    base.update(over)
    return _req(**base)


# ── (1) the no-retake payload is untouched ──────────────────────────────────
def test_default_chain_payload_key_set_is_unchanged(tmp_path):
    captured: list[dict] = []
    _backend(captured).generate_chain(_req(), tmp_path / "out")
    assert set(captured[0]) == DEFAULT_CHAIN_PAYLOAD_KEYS
    assert "retake" not in captured[0]


def test_a_retake_request_without_a_cut_window_still_sends_no_retake_key(tmp_path):
    # Both halves of the guard matter: the window path is what the engine reads,
    # so a request block with no cut file must not produce a half-formed key.
    captured: list[dict] = []
    _backend(captured).generate_chain(_retake_req(), tmp_path / "out")
    assert set(captured[0]) == DEFAULT_CHAIN_PAYLOAD_KEYS


def test_a_cut_window_without_a_retake_request_sends_no_retake_key(tmp_path):
    captured: list[dict] = []
    _backend(captured).generate_chain(
        _req(), tmp_path / "out", retake_window_path=tmp_path / "_retake_window.mp4"
    )
    assert set(captured[0]) == DEFAULT_CHAIN_PAYLOAD_KEYS


# ── the retake payload shape ────────────────────────────────────────────────
def test_retake_payload_shape(tmp_path):
    captured: list[dict] = []
    win = tmp_path / "_retake_window.mp4"
    _backend(captured).generate_chain(
        _retake_req(), tmp_path / "out", retake_window_path=win
    )
    assert set(captured[0]) == DEFAULT_CHAIN_PAYLOAD_KEYS | {"retake"}
    assert captured[0]["retake"] == {
        "path": str(win),
        "head_px": 25,
        "tail_px": 24,
        "regenerate_audio": True,
    }
    # The window LENGTH is not duplicated into the payload — clips[0] carries it,
    # and a second copy could disagree.
    assert "window_px" not in captured[0]["retake"]
    assert captured[0]["clips"][0]["num_frames"] == 169


def test_retake_payload_carries_non_default_glue_and_audio_flag(tmp_path):
    captured: list[dict] = []
    _backend(captured).generate_chain(
        _retake_req(
            retake={
                "video_id": "vid-1", "window_start_sec": 0.0,
                "head_px": 41, "tail_px": 40, "regenerate_audio": False,
            }
        ),
        tmp_path / "out",
        retake_window_path=tmp_path / "w.mp4",
    )
    rt = captured[0]["retake"]
    assert (rt["head_px"], rt["tail_px"], rt["regenerate_audio"]) == (41, 40, False)


# ── (2) mock vs engine metadata key sets ────────────────────────────────────
def _engine_retake_meta_keys() -> set[str]:
    """Keys the engine puts under ``chain.retake``, read out of the source.

    Parsed rather than executed: importing ``chain_pipeline`` needs torch and the
    LTX wheel, which the app venv deliberately does not have. The geometry half
    comes from ``ChainLayout.to_dict()``; this recovers the runtime half from the
    ``retake_meta.update({...})`` literal and the freeze-proof assignment.
    """
    src = Path("engine/pipeline/chain_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    keys: set[str] = set()
    for node in ast.walk(tree):
        # retake_meta.update({...})
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "retake_meta"
                and node.args and isinstance(node.args[0], ast.Dict)):
            keys |= {k.value for k in node.args[0].keys if isinstance(k, ast.Constant)}
        # retake_meta = {"freeze_proof": checks}
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(getattr(t, "id", None) == "retake_meta" for t in node.targets)):
            keys |= {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    assert keys, "could not recover the engine's retake metadata keys"
    return keys


def test_engine_runtime_retake_keys_are_the_expected_contract():
    assert _engine_retake_meta_keys() == {
        "regenerate_audio", "source_had_audio", "audio_frozen",
        "muxed_original_waveform", "decoded_frames_px", "freeze_proof",
    }


def test_mock_matches_the_engine_contract_except_freeze_proof(tmp_path, monkeypatch):
    import chain_math
    from services import ltx_runner

    layout = chain_math.compute_chain_layout([169], 24.0, retake_glue_px=(25, 24))
    geometry_keys = set(layout.to_dict()["retake"])

    monkeypatch.setattr(ltx_runner.video_io, "has_audio_stream", lambda p: True)
    be = ltx_runner._MockBackend.__new__(ltx_runner._MockBackend)
    be.config = types.SimpleNamespace(output=types.SimpleNamespace(keep_raw_frames=False))
    be._loaded = True
    be.load = lambda: None
    be._render_chain_frames = lambda **kw: []
    monkeypatch.setattr(ltx_runner.video_io, "encode_frames_to_mp4", lambda *a, **k: None)
    monkeypatch.setattr(ltx_runner.gpu_info, "reset_peak_vram", lambda: None)
    monkeypatch.setattr(ltx_runner.gpu_info, "peak_vram_mb", lambda: 0)

    outcome = be.generate_chain(
        _retake_req(), tmp_path / "out",
        retake_window_path=tmp_path / "_retake_window.mp4",
    )
    mock_keys = set(outcome.chain_metadata["retake"])
    engine_keys = geometry_keys | _engine_retake_meta_keys()

    assert "freeze_proof" not in mock_keys, (
        "the mock has no latents; a fabricated freeze_proof would forge the one "
        "claim this feature rests on"
    )
    assert mock_keys == engine_keys - {"freeze_proof"}
    # And the geometry half really did come through, not just the runtime half.
    assert outcome.chain_metadata["retake"]["n_head_v"] == 4
    assert outcome.chain_metadata["retake"]["n_tail_a"] == 24


def test_mock_without_retake_emits_no_retake_block(tmp_path, monkeypatch):
    from services import ltx_runner

    be = ltx_runner._MockBackend.__new__(ltx_runner._MockBackend)
    be.config = types.SimpleNamespace(output=types.SimpleNamespace(keep_raw_frames=False))
    be._loaded = True
    be.load = lambda: None
    be._render_chain_frames = lambda **kw: []
    monkeypatch.setattr(ltx_runner.video_io, "encode_frames_to_mp4", lambda *a, **k: None)
    monkeypatch.setattr(ltx_runner.gpu_info, "reset_peak_vram", lambda: None)
    monkeypatch.setattr(ltx_runner.gpu_info, "peak_vram_mb", lambda: 0)

    outcome = be.generate_chain(_req(), tmp_path / "out")
    assert "retake" not in outcome.chain_metadata
