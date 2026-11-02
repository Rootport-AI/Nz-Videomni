"""engine_h3.worker: the ``generate_chain`` op (台帳 §1-106, Docs/MINIMAX_H3_DESIGN.md 裁定 84).

Runs in ``.venv-engine-h3`` through the engine-venv runner::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_worker_chain.py

CPU only, no model: the parsing, the shared settings, the ignored keys, the
progress line's chain fields and the ``done`` payload's keys (a pipeline whose
three stages are stubbed). The single job's goldens stay in tests/test_h3_engine.py.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from engine_h3 import worker  # noqa: E402
from engine_h3.chain_h3 import H3ChainClipSpec  # noqa: E402
from engine_h3.pipeline_h3 import H3LoraInput  # noqa: E402

CLIPS = [{"prompt": "a", "num_frames": 124}, {"prompt": "a b", "num_frames": 39}]
BASE = {"op": "generate_chain", "prompt": "a", "seed": 5, "width": 640, "height": 384, "frame_rate": 24,
        "output_path": "o.mp4", "clips": CLIPS}

#: The chain block's keys, in order (h3_chain_math.H3ChainLayout.to_dict(), then the run's).
GOLDEN_CHAIN_KEYS = [
    "total_px", "segment_seam_junctions", "tile_seam_junctions", "all_junctions", "n_tiles",
    "overlap_frames", "overlap_strength", "clip_frames", "clip_latent_frames", "clip_audio_latents",
    "overlap_latents", "seam_start_frames", "seam_new_frames", "audio_overlap_latents",
    "total_latent_frames", "total_audio_latents", "audio_latents_expected", "audio_latent_excess",
    "audio_place_error_latents", "audio_head_shift_latents", "audio_drift_flags",
    "seeds", "clips", "h3",
]
GOLDEN_CHAIN_CLIP_KEYS = ["prompt_sha256", "seconds", "latents_sha256", "head_equals_prev_tail"]


def test_chain_fields():
    assert worker._CHAIN_FIELDS == (
        worker._GENERATE_FIELDS - {"num_frames", "images", "source_audio", "references"}
    ) | {"clips", "source"}
    assert worker._CHAIN_CLIP_FIELDS == frozenset({"prompt", "num_frames", "images", "references"})
    assert worker._SOURCE_FIELDS == frozenset({"path", "context_frames"})
    assert "prompt" in worker._CHAIN_FIELDS and "loras" in worker._CHAIN_FIELDS


def test_chain_clips_images_are_each_clips_own():
    from engine_h3.pipeline_h3 import H3ImageInput

    clips, ignored = worker._chain_clips({"clips": [
        {"prompt": "x", "num_frames": 56,
         "images": [{"path": "a.png", "frame_idx": 0}, {"path": "b.png", "frame_idx": 55, "strength": 1.0}]},
        {"prompt": "y", "num_frames": 39, "images": None},
    ]})
    assert clips[0].images == (H3ImageInput("a.png", 0), H3ImageInput("b.png", 55))
    assert clips[1].images == ()
    assert ignored == ["images[].strength"]


@pytest.mark.parametrize("bad", [
    {"prompt": "x", "num_frames": 56, "images": {"path": "a.png"}},
    {"prompt": "x", "num_frames": 56, "images": [{"frame_idx": 0}]},
])
def test_chain_clips_malformed_images_are_refused(bad):
    with pytest.raises(ValueError, match="images"):
        worker._chain_clips({"clips": [bad]})


def test_chain_source():
    from engine_h3.chain_h3 import H3ChainSource

    assert worker._chain_source({}) == (None, [])
    assert worker._chain_source({"source": None}) == (None, [])
    source, ignored = worker._chain_source({"source": {"path": "t.mp4", "context_frames": 22, "video_id": "v"}})
    assert source == H3ChainSource("t.mp4", 22) and ignored == ["video_id"]


@pytest.mark.parametrize("bad", [
    "t.mp4", ["t.mp4"], {"context_frames": 22}, {"path": "", "context_frames": 22},
    {"path": 3, "context_frames": 22}, {"path": "t.mp4"}, {"path": "t.mp4", "context_frames": "22"},
    {"path": "t.mp4", "context_frames": 22.0}, {"path": "t.mp4", "context_frames": True},
])
def test_chain_source_malformed_is_refused(bad):
    with pytest.raises(ValueError, match="source"):
        worker._chain_source({"source": bad})


def test_chain_clips_valid_and_inner_keys_ignored():
    clips, ignored = worker._chain_clips({"clips": [{"prompt": "x", "num_frames": 124, "seed": 3, "image": "i"},
                                                    {"prompt": "y", "num_frames": 39}]})
    assert clips == [H3ChainClipSpec("x", 124), H3ChainClipSpec("y", 39)]
    assert ignored == ["image", "seed"]


def test_chain_clips_references_are_each_clips_own():
    from engine_h3.pipeline_h3 import H3ReferenceInput

    clips, ignored = worker._chain_clips({"clips": [
        {"prompt": "x", "num_frames": 124,
         "references": [{"kind": "picture", "path": "p.png", "slot": 1, "id": "u1"},
                        {"kind": "audio", "path": "a.wav", "slot": 2, "max_sec": 3}]},
        {"prompt": "y", "num_frames": 39, "references": None},
        {"prompt": "z", "num_frames": 39, "references": []},
    ]})
    assert clips[0].references == (H3ReferenceInput("picture", "p.png", 1), H3ReferenceInput("audio", "a.wav", 2, 3.0))
    assert clips[1].references == () and clips[2].references == ()
    assert ignored == ["references[].id"]


@pytest.mark.parametrize("bad", [
    {"prompt": "x", "num_frames": 124, "references": {"kind": "picture"}},
    {"prompt": "x", "num_frames": 124, "references": [{"kind": "picture", "slot": 1}]},
    {"prompt": "x", "num_frames": 124, "references": [{"kind": "picture", "path": "p.png", "slot": "1"}]},
    {"prompt": "x", "num_frames": 124, "references": [{"kind": "video", "path": "v.mp4", "slot": 1, "max_sec": True}]},
    {"prompt": "x", "num_frames": 124, "references": [{"kind": "video", "path": "v.mp4", "slot": 1, "max_sec": "3"}]},
    {"prompt": "x", "num_frames": 124,
     "references": [{"kind": "audio", "path": "a.wav", "slot": 1, "max_sec": float("nan")}]},
])
def test_chain_clips_malformed_references_are_refused(bad):
    with pytest.raises(ValueError, match="references"):
        worker._chain_clips({"clips": [bad]})


@pytest.mark.parametrize("bad", [
    None, "clips", {"prompt": "x"},
    ["x"],
    [{"num_frames": 124}],
    [{"prompt": 3, "num_frames": 124}],
    [{"prompt": "x"}],
    [{"prompt": "x", "num_frames": "124"}],
    [{"prompt": "x", "num_frames": 124.0}],
    [{"prompt": "x", "num_frames": True}],
])
def test_chain_clips_malformed_is_refused(bad):
    with pytest.raises(ValueError, match="clips"):
        worker._chain_clips({"clips": bad} if bad is not None else {})


def test_emit_progress_chain_fields_only_when_given(monkeypatch):
    lines = []
    monkeypatch.setattr(worker, "_emit", lambda event, **f: lines.append({"event": event, **f}))
    worker._emit_progress("encode", 1, 3)
    worker._emit_progress("stage1_denoise", 2, 20, outer_index=1, outer_total=3)
    assert lines[0] == {"event": "progress", "stage": "encode", "index": 1, "total": 3}
    assert lines[1] == {"event": "progress", "stage": "stage1_denoise", "index": 2, "total": 20,
                        "outer_index": 1, "outer_total": 3}


def _capture(monkeypatch, result=None):
    seen, logs, events = [], [], []

    def run(pipe, spec, progress, log):
        seen.append((spec, progress))
        return dict(result or {"attention_used": "x", "h3": {}})

    monkeypatch.setattr(worker, "_PIPE", object())
    monkeypatch.setattr(worker, "_ATTENTION", ("sdpa", False))
    monkeypatch.setattr(worker, "_log", logs.append)
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    monkeypatch.setattr("engine_h3.chain_h3.run_chain_h3", run)
    return seen, logs, events


def test_do_generate_chain_builds_the_spec(monkeypatch):
    seen, logs, events = _capture(monkeypatch)
    msg = {**BASE, "num_inference_steps": 8, "shift_video": 10, "shift_audio": 2.5, "keep_resident": True,
           "block_swap_prefetch": False, "vram16_mode": True, "loras": [{"path": "t.safetensors", "strength": 0.8}]}
    worker._do_generate_chain(msg)
    spec, progress = seen[0]
    base = spec.base
    assert spec.seed == 5 and spec.clips == (H3ChainClipSpec("a", 124), H3ChainClipSpec("a b", 39))
    assert (base.width, base.height, base.frame_rate, base.output_path) == (640, 384, 24.0, "o.mp4")
    assert (base.num_inference_steps, base.shift_video, base.shift_audio) == (8, 10.0, 2.5)
    assert (base.keep_resident, base.block_swap_prefetch, base.vram16_mode) == (True, False, True)
    assert base.loras == (H3LoraInput("t.safetensors", 0.8),)
    assert base.images == () and base.source_audio is None and base.references == ()
    assert base.attention_backend == "sdpa"
    assert progress is worker._emit_progress
    assert events == [{"event": "done", "attention_used": "sdpa", "h3": {}}]
    assert not any("ignored keys" in line for line in logs)


def test_do_generate_chain_defaults_and_ignored_keys(monkeypatch):
    seen, logs, _events = _capture(monkeypatch)
    msg = {**BASE, "num_frames": 124, "images": [{"path": "a.png"}], "source_audio": {"path": "a.wav"},
           "references": [], "overlap_frames": 3, "clips": [{**CLIPS[0], "x": 1}, CLIPS[1]]}
    worker._do_generate_chain(msg)
    base = seen[0][0].base
    assert (base.num_inference_steps, base.shift_video, base.shift_audio) == (20, 12.0, 3.0)
    assert (base.keep_resident, base.vram16_mode, base.loras) == (False, False, ())
    assert base.images == () and base.source_audio is None and base.references == ()
    line = next(line for line in logs if line.startswith("generate_chain:"))
    assert "ignored keys: images, num_frames, overlap_frames, references, source_audio, clips[].x" in line


def test_do_generate_chain_passes_clip_references(monkeypatch):
    seen, logs, _events = _capture(monkeypatch)
    ref = {"kind": "picture", "path": "p.png", "slot": 1, "foo": 0}
    msg = {**BASE, "clips": [{**CLIPS[0], "references": [ref]}, CLIPS[1]], "references": [ref]}
    worker._do_generate_chain(msg)
    spec = seen[0][0]
    assert [len(c.references) for c in spec.clips] == [1, 0] and spec.clips[0].references[0].path == "p.png"
    assert spec.base.references == ()                    # the top-level references stay ignored
    line = next(line for line in logs if line.startswith("generate_chain:"))
    assert "ignored keys: references, clips[].references[].foo" in line


def test_do_generate_chain_passes_source_and_clip_images(monkeypatch):
    from engine_h3.chain_h3 import H3ChainSource

    seen, logs, _events = _capture(monkeypatch)
    msg = {**BASE, "clips": [{**CLIPS[0], "images": [{"path": "k.png", "frame_idx": 0, "strength": 1}]}, CLIPS[1]],
           "source": {"path": "in/tail.mp4", "context_frames": 22, "x": 1}}
    worker._do_generate_chain(msg)
    spec = seen[0][0]
    assert spec.source == H3ChainSource("in/tail.mp4", 22)
    assert [len(c.images) for c in spec.clips] == [1, 0] and spec.base.images == ()
    line = next(line for line in logs if line.startswith("generate_chain:"))
    assert "images=[1, 0] " in line and "source=('tail.mp4', 22) " in line
    assert "ignored keys: clips[].images[].strength, source.x" in line


def test_do_generate_chain_plain_has_no_source(monkeypatch):
    seen, logs, _events = _capture(monkeypatch)
    worker._do_generate_chain(BASE)
    assert seen[0][0].source is None
    line = next(line for line in logs if line.startswith("generate_chain:"))
    assert "source=" not in line and "images=" not in line


def test_do_generate_chain_degraded_attention(monkeypatch):
    seen, _logs, events = _capture(monkeypatch)
    monkeypatch.setattr("engine.transformer.sage_attention_service.probe_sage", lambda: False)
    worker._do_generate_chain({**BASE, "attention_backend": "sage"})
    assert seen[0][0].base.attention_backend == "sdpa"
    assert events[0]["attention_used"] == "sage->sdpa"


def test_do_generate_chain_before_load(monkeypatch):
    monkeypatch.setattr(worker, "_PIPE", None)
    with pytest.raises(RuntimeError, match="before load"):
        worker._do_generate_chain(BASE)


def _golden_chain_run(monkeypatch, msg, *, source_audio=True):
    """``_do_generate_chain(msg)`` on a pipeline whose stages are stubs (the input
    video's reading and encode too); returns the ``done`` event."""
    import numpy as np

    from engine_h3 import pipeline_h3
    from engine_h3 import source_audio as SA
    from engine_h3 import source_tail as ST

    pipe = pipeline_h3.H3Pipeline(
        pipeline_h3.H3Paths("t", "te", "vv", "av", "tok"), pipeline_h3.H3LoadOptions(),
        device="cpu", progress=lambda *a, **k: None, log=lambda m: None,
    )
    pipe._dit_cfg = object()  # "loaded"

    def denoise(req, h, t, phases, entry, **kw):
        tl, al = pipeline_h3.latent_frames(req.num_frames), pipeline_h3.audio_latents(req.num_frames)
        video, audio = torch.zeros(1, 24, tl, 2, 2), torch.zeros(2, 32, al)
        return video, audio, "on"

    def decode(req, v, a, entry, **kw):
        if kw.get("trim_head"):
            entry.update(decoded_frames=req.num_frames, trimmed_frames=kw["trim_head"], trimmed_audio_samples=29333,
                         audio_fade_in_samples=960, audio_handle="o_audio_handle.wav")
        return 1234

    def encode_source_tail(frames, wave, entry, index, total):
        freeze = 0 if wave is None else min(37, int(wave.shape[-1]) // 800)
        entry["freeze_ka"] = freeze
        return torch.zeros(1, 24, 7, 2, 2), (torch.zeros(2, 32, freeze) if freeze else None)

    monkeypatch.setattr(pipe, "_encode", lambda prompt, phases, **kw: (torch.zeros(3, 8), torch.ones(3, dtype=torch.long)))
    monkeypatch.setattr(pipe, "_denoise", denoise)
    monkeypatch.setattr(pipe, "_decode", decode)
    monkeypatch.setattr(pipe, "_encode_source_tail", encode_source_tail)
    monkeypatch.setattr(ST, "read_source_tail", lambda path, w, h, n: (np.zeros((n, h, w, 3), np.uint8),
                                                                        {"source_size": [w, h], "frames_read": n}))
    monkeypatch.setattr(ST, "has_audio_stream", lambda path: source_audio)
    monkeypatch.setattr(SA, "load_source_waveform", lambda path, **kw: (torch.zeros(2, 29333), {}))
    events = []
    monkeypatch.setattr(worker, "_PIPE", pipe)
    monkeypatch.setattr(worker, "_ATTENTION", ("sdpa", False))
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    monkeypatch.setattr(worker, "_log", lambda m: None)
    worker._do_generate_chain(msg)
    return events[-1]


#: A video-to-video chain's v2v block (its keys, in order; the LTX chain's sixteen).
GOLDEN_V2V_KEYS = [
    "source_context_px", "n_ctx_v", "n_ctx_a", "trim_px", "v2v_context_junction_px", "new_frames_px",
    "context_frames", "freeze_ka", "trimmed_px", "trimmed_audio_samples", "audio_fade_in_samples",
    "source_had_audio", "audio_head_frozen", "decoded_frames_px", "audio_handle_filename", "handle_context_seconds",
]


def test_golden_chain_done_keys(monkeypatch):
    """The ``done`` of a chain: the single job's keys (tests/test_h3_engine.py's golden) plus ``chain``."""
    from test_h3_engine import GOLDEN_DONE_KEYS

    done = _golden_chain_run(monkeypatch, BASE)
    assert done["event"] == "done"
    assert list(done)[1:] == GOLDEN_DONE_KEYS + ["chain"]
    assert list(done["chain"]) == GOLDEN_CHAIN_KEYS
    assert [list(c) for c in done["chain"]["clips"]] == [GOLDEN_CHAIN_CLIP_KEYS] * 2
    assert done["num_frames"] == 124 + 39 - 22 and done["chain"]["total_px"] == 141
    assert done["chain"]["seeds"] == [5, 6]
    assert list(done["h3"]) == ["phases", "quant_backends"]
    assert set(done["h3"]["phases"]) == {"clips", "decode"}
    assert done["images_used"] == 0


@pytest.mark.parametrize("audio", [True, False])
def test_golden_v2v_chain_done_keys(monkeypatch, audio):
    """A video-to-video chain: the same keys, ``chain`` with ``v2v`` last (台帳 §1-121)."""
    from test_h3_engine import GOLDEN_DONE_KEYS

    clips = [{"prompt": "a", "num_frames": 56}, {"prompt": "b", "num_frames": 39}]
    done = _golden_chain_run(monkeypatch, {**BASE, "clips": clips,
                                           "source": {"path": "t.mp4", "context_frames": 22}}, source_audio=audio)
    assert list(done)[1:] == GOLDEN_DONE_KEYS + ["chain"]
    assert list(done["chain"]) == GOLDEN_CHAIN_KEYS + ["v2v"]
    assert [list(c) for c in done["chain"]["clips"]] == [GOLDEN_CHAIN_CLIP_KEYS] * 2
    assert done["num_frames"] == 73 and done["chain"]["total_px"] == 73
    v2v = done["chain"]["v2v"]
    assert list(v2v) == GOLDEN_V2V_KEYS
    assert (v2v["new_frames_px"], v2v["freeze_ka"], v2v["source_had_audio"]) == (51, 36 if audio else 0, audio)
    import json

    assert json.loads(json.dumps(done["chain"])) == done["chain"]


def test_golden_chain_with_images_counts_them(monkeypatch, tmp_path):
    from PIL import Image

    path = tmp_path / "k.png"
    Image.new("RGB", (64, 64)).save(path)
    monkeypatch.setattr("engine_h3.pipeline_h3.H3Pipeline._encode_image",
                        lambda self, images, entry, index, total, enc=None: torch.zeros(len(images) * 4, 96))
    clips = [{**CLIPS[0], "images": [{"path": str(path), "frame_idx": 0}]}, CLIPS[1]]
    done = _golden_chain_run(monkeypatch, {**BASE, "clips": clips})
    assert done["images_used"] == 1 and list(done["chain"]) == GOLDEN_CHAIN_KEYS


def test_main_dispatches_generate_chain_and_survives_a_failure(monkeypatch):
    import io
    import json

    calls, events, released = [], [], []

    def fail(msg):
        calls.append(msg["op"])
        raise ValueError("bad chain")

    monkeypatch.setattr(worker, "_do_load", lambda msg: None)
    monkeypatch.setattr(worker, "_do_generate_chain", fail)
    monkeypatch.setattr(worker, "_release_after_failed_job", lambda: released.append(1))
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    monkeypatch.setattr(worker, "_log", lambda m: None)
    monkeypatch.setattr(worker, "_shutdown", lambda: (_ for _ in ()).throw(SystemExit(0)))
    lines = [{"op": "load"}, {**BASE}, {"op": "shutdown"}]
    monkeypatch.setattr("sys.stdin", io.StringIO("".join(json.dumps(m) + "\n" for m in lines)))
    with pytest.raises(SystemExit):
        worker.main()
    assert calls == ["generate_chain"] and released == [1]
    assert events[0]["event"] == "error" and "bad chain" in events[0]["detail"]
