"""Worker dispatch for an ``alpha_gen`` job on LTX 2.5 (台帳 §1-83).

The AlphaGen twin of ``tests/test_worker_ltx25_inpaint_dispatch.py``. The
PRESENCE of the ``alpha_gen`` block routes ``engine25.worker._do_generate`` to
``engine25.alphagen25.run_alpha_gen`` (one stage at the full canvas size);
without it the op is the plain ``Ltx25Pipeline.generate`` -- which is also what
the light mode is. Pinned here: the routing, the arguments, the mutual
exclusion with ``outpaint``/``inpaint``, and the ``done`` event, which is the
plain generation's shape (no additive key).

NO PIPELINE AND NO GPU. ``_PIPE`` is a recording double and the drivers are
replaced by fakes.

Run with ``.venv-engine-ltx25`` and ``--noconftest``::

    .venv-engine-ltx25/Scripts/python.exe \\
        Docs/Outputs-archive/start-end-bridge-2026-09-07/implA_engine_runner/run_ltx25_pytest.py \\
        tests/test_worker_ltx25_alphagen_dispatch.py
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("ltx_core")

from engine25 import alphagen25, inpaint25, outpaint25, worker  # noqa: E402
from engine25.pipeline25 import GenerationResult  # noqa: E402

CANVAS_W, CANVAS_H = 1472, 832
FRAMES = 9


def _result() -> GenerationResult:
    return GenerationResult(
        output_path="out.mp4",
        seed=7,
        width=CANVAS_W,
        height=CANVAS_H,
        num_frames=FRAMES,
        frame_rate=24.0,
        encode_fps=24,
        num_images=0,
        size_bytes=1024,
        seconds=12.5,
        phases={"21_stage1_denoise": {"seconds": 1.0}},
        peak_allocated_gib=9.0,
        peak_reserved_gib=10.0,
        rss_peak_gib=12.0,
        video_chunks=2,
        tiling="TileSizeConfig(...)",
    )


class _RecordingPipe:
    """Everything ``_do_generate`` reads off ``_PIPE``, and nothing else."""

    sampler = "ancestral"
    video_vae_kind = "conv"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def set_acceleration_job(self, **_kw) -> None:
        pass

    def set_nag_job(self, _nag) -> None:
        pass

    def reset_nag_job(self) -> None:
        pass

    def reset_acceleration_job(self) -> None:
        pass

    def block_swap_prefetch_used(self) -> str:
        return "on"

    def fused_gguf_dequant_kernel_used(self) -> str:
        return "on"

    def keep_resident_used(self) -> str:
        return "off"

    def keep_resident_embeddings_used(self) -> str:
        return "off"

    def generate(self, **kwargs):
        self.calls.append(("plain", kwargs))
        return _result()


@pytest.fixture()
def harness(monkeypatch, tmp_path):
    pipe = _RecordingPipe()
    events: list[dict] = []
    logs: list[str] = []

    def _fake(name):
        def run(_pipeline, **kwargs):
            pipe.calls.append((name, kwargs))
            return _result()

        return run

    monkeypatch.setattr(worker, "_PIPE", pipe)
    monkeypatch.setattr(worker, "_log", logs.append)
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    monkeypatch.setattr(alphagen25, "run_alpha_gen", _fake("alpha"))
    monkeypatch.setattr(inpaint25, "run_inpaint", _fake("inpaint"))
    monkeypatch.setattr(outpaint25, "run_outpaint", _fake("outpaint"))

    # ``_resolve_ic_reference`` checks the reference really is a file.
    (tmp_path / "alpha_reference.mp4").write_bytes(b"\x00" * 32)
    return pipe, events, logs, tmp_path


def _msg(tmp_path, **extra) -> dict:
    msg = {
        "prompt": " ",
        "seed": 7,
        "width": CANVAS_W,
        "height": CANVAS_H,
        "num_frames": FRAMES,
        "frame_rate": 24.0,
        "output_path": str(tmp_path / "output.mp4"),
        "loras": [{"path": str(tmp_path / "alpha-gen.safetensors"), "strength": 1.0}],
        "reference_video": {"path": str(tmp_path / "alpha_reference.mp4"), "strength": 1.0},
    }
    msg.update(extra)
    return msg


ALPHA = {"mode": "one_stage"}


# ── 1. routing ──────────────────────────────────────────────────────────────
def test_an_alpha_gen_message_calls_run_alpha_gen(harness):
    pipe, _events, _logs, tmp_path = harness
    worker._do_generate(_msg(tmp_path, alpha_gen=ALPHA))

    assert [name for name, _ in pipe.calls] == ["alpha"]
    kwargs = pipe.calls[0][1]
    assert kwargs["prompt"] == " " and kwargs["seed"] == 7
    assert (kwargs["width"], kwargs["height"]) == (CANVAS_W, CANVAS_H)
    assert kwargs["num_frames"] == FRAMES and kwargs["frame_rate"] == 24.0
    assert kwargs["output_path"] == str(tmp_path / "output.mp4")
    assert kwargs["ic_reference"] == (str(tmp_path / "alpha_reference.mp4"), 1.0)
    assert [entry[0] for entry in kwargs["ic_loras"]] == [str(tmp_path / "alpha-gen.safetensors")]
    assert kwargs["ic_attention_strength"] == 1.0
    assert kwargs["progress"] is worker._emit_progress
    assert "ignored" in kwargs


def test_a_message_without_the_block_is_the_plain_generation(harness):
    """The light mode sends no block: it is the plain two-stage generation."""
    pipe, _events, _logs, tmp_path = harness
    worker._do_generate(_msg(tmp_path))
    assert [name for name, _ in pipe.calls] == ["plain"]


def test_the_parse_log_line_names_the_alpha_mode(harness):
    _pipe, _events, logs, tmp_path = harness
    worker._do_generate(_msg(tmp_path, alpha_gen=ALPHA))
    line = next(entry for entry in logs if entry.startswith("generate "))
    assert "alpha=one_stage " in line


def test_the_parse_log_line_says_none_for_a_plain_job(harness):
    _pipe, _events, logs, tmp_path = harness
    worker._do_generate(_msg(tmp_path))
    line = next(entry for entry in logs if entry.startswith("generate "))
    assert "alpha=none " in line


# ── 2. mutual exclusion ─────────────────────────────────────────────────────
@pytest.mark.parametrize("other", ["inpaint", "outpaint"])
def test_alpha_gen_with_another_block_asserts(harness, other):
    pipe, _events, _logs, tmp_path = harness
    with pytest.raises(AssertionError, match="mutually exclusive"):
        worker._do_generate(_msg(tmp_path, alpha_gen=ALPHA, **{other: {"canvas_width": 1}}))
    assert pipe.calls == []


# ── 3. the done event ───────────────────────────────────────────────────────
def test_the_done_event_is_the_plain_shape(harness):
    _pipe, events, _logs, tmp_path = harness
    worker._do_generate(_msg(tmp_path, alpha_gen=ALPHA))
    done = [e for e in events if e["event"] == "done"]
    assert len(done) == 1
    done = done[0]
    for key in ("alpha", "alpha_gen", "inpaint", "outpaint"):
        assert key not in done
    assert done["ltx25"] == {
        "encode_fps": 24,
        "video_chunks": 2,
        "tiling": "TileSizeConfig(...)",
        "size_bytes": 1024,
        "phases": {"21_stage1_denoise": {"seconds": 1.0}},
    }
    assert done["seed_used"] == 7 and done["num_frames"] == FRAMES
