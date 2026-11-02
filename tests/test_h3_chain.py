"""engine_h3.chain_h3: the MiniMax H3 chain's orchestration (台帳 §1-106, Docs/MINIMAX_H3_DESIGN.md 裁定 84).

Runs in ``.venv-engine-h3`` through the engine-venv runner::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_chain.py

CPU only, no weights. (1) a pipeline whose ``_encode`` / ``_denoise`` /
``_decode`` are stubs: the call order, the per-clip requests, the fixed heads,
the progress and the release (the measurement script's self-check 9,
Docs/Outputs-archive/h3-chain-2026-10-29/c_chain.py, ported); (2) the per-clip
Ref2VA references (台帳 §1-109) through the REAL ``_encode`` /
``_read_references`` / ``_encode_references`` with stand-in encoders
(tests/test_h3_engine.py's ``_ref_job``, as a chain); (3) the REAL ``_denoise``
on the tiny DiT of tests/test_h3_engine.py: the fixed rows end as the clean
latents bit for bit and are injected as the latents themselves, also next to
reference rows; (4) video-to-video and clip 0's key frames (台帳 §1-121) on the
stubbed pipeline: the input video's latents as clip 0's fixed head, the trimmed
decode, the ``v2v`` block, the I2V steps of clip 0 and the chain's refusals.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("comfy_kitchen")

from engine_h3 import chain_h3 as CH  # noqa: E402
from engine_h3 import pipeline_h3 as P  # noqa: E402
from h3_chain_math import compute_h3_chain_layout  # noqa: E402

CH_LAT, CH_AUD, LH, LW = 24, 32, 2, 3


def _base(**kw):
    fields = dict(prompt="common", seed=7, width=96, height=64, num_frames=124, output_path="out/chain.mp4",
                  num_inference_steps=4, keep_resident=False)
    fields.update(kw)
    return P.H3GenerateRequest(**fields)


def _spec(frames=(124, 124, 124), seed=42, **kw):
    clips = tuple(CH.H3ChainClipSpec(prompt=f"clip {i}", num_frames=f) for i, f in enumerate(frames))
    return CH.H3ChainSpec(base=_base(**kw), clips=clips, seed=seed)


class _Recorder:
    """The stubs' shared log."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.progress: list[tuple] = []
        self.released = 0


def _stub_pipe(monkeypatch, rec: _Recorder, *, device="cpu", fail_at=None, break_head=False):
    pipe = P.H3Pipeline(P.H3Paths("t", "te", "vv", "av", "tok"), P.H3LoadOptions(), device=device,
                        progress=lambda *a, **k: rec.progress.append(("pipe-original", a, k)), log=lambda m: None)
    pipe._dit_cfg = object()
    original_progress = pipe._progress

    def encode(prompt, phases, images=(), *, has_cond, has_audio=False, refs=None):
        assert "encode" in phases.data           # called inside the "encode" phase
        rec.calls.append(("encode", prompt, tuple(images), has_cond))
        for i in range(4):
            pipe._progress("encode", i, 3)
        return torch.zeros(3, 8), torch.ones(3, dtype=torch.long)

    def denoise(req, hidden, tags, phases, entry, **kw):
        index = sum(1 for c in rec.calls if c[0] == "denoise")
        rec.calls.append(("denoise", req, kw))
        if fail_at == index:
            raise RuntimeError("denoise failed")
        if kw.get("lora_report") is not None:
            kw["lora_report"].append({"name": "x.safetensors"})
        pipe._progress("stage1_denoise", 1, 1)
        t, a = P.latent_frames(req.num_frames), P.audio_latents(req.num_frames)
        g = torch.Generator().manual_seed(int(req.seed))
        video = torch.randn(1, CH_LAT, t, LH, LW, generator=g)
        audio = torch.randn(2, CH_AUD, a, generator=g)
        # What denoise_mask.finish does: the m == 0 rows come back as their clean latents.
        for fix, out in ((kw.get("video_fix"), video), (kw.get("audio_fix"), audio)):
            if fix is not None:
                n = sum(1 for v in fix[1] if v == 0.0)
                out[:, :, :n] = fix[0][:, :, :n]
        if break_head and kw.get("video_fix") is not None:
            video[:, :, 0] += 1.0
        return video, audio, "on"

    def decode(req, video, audio, entry, keep=False, source_wave=None, **kw):
        rec.calls.append(("decode", req, video, audio, keep, kw))
        if kw.get("trim_head"):
            # What the real _decode records for a trimmed decode (tests/test_h3_source_tail.py).
            entry.update(decoded_frames=req.num_frames, trimmed_frames=kw["trim_head"], trimmed_audio_samples=29333,
                         audio_fade_in_samples=960, audio_handle="chain_audio_handle.wav")
        pipe._progress("decode", 3, 3)
        return 1234

    def release():
        rec.released += 1
        rec.calls.append(("release",))
        pipe._resident = None

    monkeypatch.setattr(pipe, "_encode", encode)
    monkeypatch.setattr(pipe, "_denoise", denoise)
    monkeypatch.setattr(pipe, "_decode", decode)
    monkeypatch.setattr(pipe, "release_resident", release)
    return pipe, original_progress


def _progress_into(rec):
    return lambda stage, index, total, **kw: rec.progress.append((stage, index, total, kw))


def test_clip_requests_replace_prompt_frames_seed_and_keep():
    spec = _spec(frames=(124, 39), seed=100, keep_resident=False)
    reqs = CH.clip_requests(spec)
    assert [(r.prompt, r.num_frames, r.seed, r.keep_resident) for r in reqs] == [
        ("clip 0", 124, 100, True), ("clip 1", 39, 101, True)]
    assert all(r.width == 96 and r.output_path == "out/chain.mp4" and r.num_inference_steps == 4 for r in reqs)
    assert all(r.references == () for r in reqs)


def test_clip_requests_carry_each_clips_own_references():
    ref = P.H3ReferenceInput("picture", "p.png", 1)
    spec = CH.H3ChainSpec(base=_base(), seed=1, clips=(
        CH.H3ChainClipSpec("a", 124, references=(ref,)), CH.H3ChainClipSpec("b", 39)))
    reqs = CH.clip_requests(spec)
    assert [r.references for r in reqs] == [(ref,), ()]
    assert spec.base.references == ()


def test_carry_fix_head_is_the_tail_and_the_rest_generated():
    pv = torch.randn(1, CH_LAT, 37, LH, LW)
    pa = torch.randn(2, CH_AUD, 377)
    (cv, per_v), (ca, per_a) = CH.carry_fix(pv, pa, 12, 65, 7, 37)
    assert tuple(cv.shape) == (1, CH_LAT, 12, LH, LW) and tuple(ca.shape) == (2, CH_AUD, 65)
    assert cv.dtype == torch.float32 and ca.dtype == torch.float32
    assert torch.equal(cv[:, :, :7], pv[:, :, -7:]) and bool((cv[:, :, 7:] == 0).all())
    assert torch.equal(ca[:, :, :37], pa[:, :, -37:]) and bool((ca[:, :, 37:] == 0).all())
    assert per_v == [0.0] * 7 + [1.0] * 5 and per_a == [0.0] * 37 + [1.0] * 28


def test_latents_sha256_is_the_scripts_formula():
    import hashlib

    t = torch.randn(1, 2, 3)
    assert CH.latents_sha256(t) == hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("keep", [False, True])
def test_run_chain_order_requests_heads_and_decode(monkeypatch, keep):
    rec = _Recorder()
    pipe, original = _stub_pipe(monkeypatch, rec, device="meta")
    spec = _spec(frames=(124, 124, 124), seed=42, keep_resident=keep)
    done = CH.run_chain_h3(pipe, spec, _progress_into(rec), lambda m: None)
    layout = compute_h3_chain_layout([124, 124, 124])

    kinds = [c[0] for c in rec.calls]
    # Without the key the kept DiT goes before the decode (as a single job's denoise drops
    # it), and the finally releases the rest after it; with the key nothing is released.
    tail = ["release", "decode", "release"] if not keep else ["decode"]
    assert kinds == ["encode", "denoise"] * 3 + tail
    enc = [c for c in rec.calls if c[0] == "encode"]
    assert [c[1] for c in enc] == ["clip 0", "clip 1", "clip 2"] and all(c[2] == () and c[3] is False for c in enc)
    den = [c for c in rec.calls if c[0] == "denoise"]
    assert [c[1].seed for c in den] == [42, 43, 44]
    assert all(c[2]["keep"] is True and c[2]["video_inject_clean"] == 1.0 for c in den)
    assert den[0][2]["video_fix"] is None and den[0][2]["audio_fix"] is None
    # LoRA report: clip 0's list only.
    assert isinstance(den[0][2]["lora_report"], list) and den[1][2]["lora_report"] is None

    # Clip i >= 1: the head is the previous clip's video tail and the concatenated audio's tail.
    outs = [(c[1].seed, P.latent_frames(c[1].num_frames), P.audio_latents(c[1].num_frames)) for c in den]
    regen = []
    for seed, t, a in outs:
        g = torch.Generator().manual_seed(seed)
        regen.append((torch.randn(1, CH_LAT, t, LH, LW, generator=g), torch.randn(2, CH_AUD, a, generator=g)))
    v0, a0 = regen[0]
    (cv1, per_v1), (ca1, per_a1) = den[1][2]["video_fix"], den[1][2]["audio_fix"]
    assert torch.equal(cv1[:, :, :7], v0[:, :, -7:]) and bool((cv1[:, :, 7:] == 0).all())
    assert torch.equal(ca1[:, :, :37], a0[:, :, -37:])
    assert per_v1 == [0.0] * 7 + [1.0] * 30 and per_a1 == [0.0] * 37 + [1.0] * 170

    # Decode: once, the timeline's frame count, the latents on the pipeline's device.
    dec = next(c for c in rec.calls if c[0] == "decode")
    assert dec[1].num_frames == layout.total_frames == 328 and dec[1].output_path == "out/chain.mp4"
    assert dec[2].device == pipe.device and dec[3].device == pipe.device
    assert tuple(dec[2].shape) == (1, CH_LAT, 97, LH, LW) and tuple(dec[3].shape) == (2, CH_AUD, 547)
    assert dec[4] is keep and dec[5] == {}             # a plain chain: no trim, no handle

    # Progress: the clips' lines carry their position, the decode's none; the original is back.
    assert pipe._progress is original
    clip_lines = [p for p in rec.progress if p[0] in ("encode", "stage1_denoise")]
    assert {p[3]["outer_index"] for p in clip_lines} == {0, 1, 2} and all(p[3]["outer_total"] == 3 for p in clip_lines)
    assert [p for p in rec.progress if p[0] == "pipe-original"] == [("pipe-original", ("decode", 3, 3), {})]

    # Release: as for one job.
    assert rec.released == (0 if keep else 2)
    assert done["keep_resident_used"] == ("on" if keep else "off")

    # The done payload.
    assert done["num_frames"] == 328 and done["seed_used"] == 42 and done["size_bytes"] == 1234
    chain = done["chain"]
    assert {k: chain[k] for k in layout.to_dict()} == layout.to_dict()
    assert chain["seeds"] == [42, 43, 44]
    assert chain["h3"] == {"overlap_frames": 22, "inject_clean": 1.0, "decode": "once"}
    assert [c["head_equals_prev_tail"] for c in chain["clips"]] == [None, True, True]
    assert chain["clips"][0]["latents_sha256"] == {"video": CH.latents_sha256(v0), "audio": CH.latents_sha256(a0)}
    assert all(set(c["seconds"]) == {"encode", "denoise"} for c in chain["clips"])
    import hashlib

    assert chain["clips"][1]["prompt_sha256"] == hashlib.sha256(b"clip 1").hexdigest()
    assert len(done["h3"]["phases"]["clips"]) == 3 and "seconds" in done["h3"]["phases"]["decode"]
    assert "loras" not in done["h3"]       # a job without LoRA: as the single job's payload


def test_done_keys_are_the_single_jobs_plus_chain(monkeypatch):
    rec = _Recorder()
    pipe, _original = _stub_pipe(monkeypatch, rec)
    single = pipe.generate(_base())
    chain = CH.run_chain_h3(pipe, _spec(frames=(124, 39)), _progress_into(rec), lambda m: None)
    assert set(chain) == set(single) | {"chain"}
    assert chain["images_used"] == 0 and chain["vram16_mode_used"] == "off"


def test_lora_report_is_clip_0s(monkeypatch):
    rec = _Recorder()
    pipe, _original = _stub_pipe(monkeypatch, rec)
    spec = _spec(frames=(124, 39, 39), loras=(P.H3LoraInput(path="x.safetensors"),))
    monkeypatch.setattr(P, "validate_request", lambda req: None)   # no LoRA file on disk here
    done = CH.run_chain_h3(pipe, spec, _progress_into(rec), lambda m: None)
    assert done["h3"]["loras"] == [{"name": "x.safetensors"}]
    den = [c for c in rec.calls if c[0] == "denoise"]
    assert all(c[1].loras == spec.base.loras for c in den)     # every clip gets the LoRA


def test_failure_releases_and_restores_progress(monkeypatch):
    rec = _Recorder()
    pipe, original = _stub_pipe(monkeypatch, rec, fail_at=1)
    with pytest.raises(RuntimeError, match="denoise failed"):
        CH.run_chain_h3(pipe, _spec(keep_resident=True), _progress_into(rec), lambda m: None)
    assert pipe._progress is original
    assert rec.released == 1                       # a failed chain releases even with keep_resident
    assert not any(c[0] == "decode" for c in rec.calls)


def test_head_that_is_not_the_tail_is_refused(monkeypatch):
    rec = _Recorder()
    pipe, _original = _stub_pipe(monkeypatch, rec, break_head=True)
    with pytest.raises(RuntimeError, match="fixed head"):
        CH.run_chain_h3(pipe, _spec(keep_resident=True), _progress_into(rec), lambda m: None)
    assert rec.released == 1


@pytest.mark.parametrize("frames, kw, match", [
    ((124, 22), {}, r"clips\[1\]: num_frames must be at least 39"),
    ((124,), {}, r"2\.\.24 clips"),
    ((124, 125), {}, "17n\\+5"),
    ((124, 124), {"images": (P.H3ImageInput(path="a.png"),)}, "共通設定"),
    ((124, 124), {"references": (P.H3ReferenceInput("picture", "p.png", 1),)}, "参照はクリップごと"),
    ((124, 124), {"width": 100}, "32 の倍数"),
])
def test_refused_before_any_stage(monkeypatch, frames, kw, match):
    rec = _Recorder()
    pipe, _original = _stub_pipe(monkeypatch, rec)
    with pytest.raises(ValueError, match=match):
        CH.run_chain_h3(pipe, _spec(frames=frames, keep_resident=True, **kw), _progress_into(rec), lambda m: None)
    assert rec.calls == [("release",)] and rec.released == 1   # no stage ran


# --------------------------------------------------------------------------- #
# per-clip references through the REAL reference steps (台帳 §1-109)
# --------------------------------------------------------------------------- #

def _ref_chain_job(monkeypatch, tmp_path, clip_refs, frames=(39, 39, 39), samples=2000, seen=None, decoded=None):
    """``run_chain_h3`` through the REAL ``_encode`` / ``_read_references`` /
    ``_te_references`` / ``_encode_references`` with a stand-in text encoder,
    video encoder, audio encoder and loader (``samples`` long) and video reader
    (as tests/test_h3_engine.py's ``_ref_job``); the denoise (keeping the fixed
    head) and the decode are stubs. 64 x 64. ``clip_refs``: per clip, ``(kind,
    slot, max_sec)`` entries. ``seen`` / ``decoded``: lists to record the denoise
    kwargs / the decode calls into (also readable after a failure). Returns
    ``(done, te_calls, encode progress, denoise kwargs per clip, builds)``."""
    import numpy as np
    from PIL import Image
    from test_h3_engine import _a2v_fakes

    from engine_h3 import reference as RM
    from engine_h3 import text_encoder as TE
    from engine_h3 import video_vae as VV

    _a2v_fakes(monkeypatch, samples=samples)
    te_calls, progress = [], []
    seen = [] if seen is None else seen
    decoded = [] if decoded is None else decoded
    builds = {"video": 0}

    class FakeTE:
        def __init__(self, *a, **k):
            self.read_report = {}

        def load(self):
            return {}

        def encode(self, prompt, images=(), *, refs=None):
            te_calls.append({"prompt": prompt, "images": len(images),
                             "refs": None if refs is None else [(r.kind, r.slot) for r in refs]})
            self.read_report = {"vision_tokens": 0}
            if refs is not None:
                self.read_report["references"] = [
                    {"kind": r.kind, "slot": r.slot, "vision_tokens": 5 * r.slot} for r in refs if r.kind == "picture"
                ]
            return torch.zeros(3, 8), torch.ones(3, dtype=torch.long)

        def release(self):
            pass

    class FakeVideoEncoder:
        def encode(self, x, tiled=True):
            return torch.zeros((1, 24, 1, x.shape[-2] // 16, x.shape[-1] // 16))

        def encode_clips(self, frames, *, device, tiled=True):
            vt = (int(frames.shape[0]) - 5) // 17 * 5 + 2
            return torch.zeros((1, 24, vt, frames.shape[1] // 16, frames.shape[2] // 16))

    def build_video(cls, *a, **k):
        builds["video"] += 1
        return FakeVideoEncoder()

    def read_video(path, *, num_frames, max_sec=None):
        info = {"source_size": [64, 32], "size": [64, 32], "source_fps": 24.0, "source_frames": 24, "used_frames": 22}
        return np.zeros((22, 32, 64, 3), np.uint8), info

    monkeypatch.setattr(TE, "H3TextEncoder", FakeTE)
    monkeypatch.setattr(VV.H3VideoEncoder, "from_file", classmethod(build_video))
    monkeypatch.setattr(RM, "read_reference_video", read_video)
    pipe = P.H3Pipeline(P.H3Paths("t", "te", "vv", "av", "tok"), P.H3LoadOptions(), device="cpu",
                        progress=lambda *a, **k: None, log=lambda m: None)
    pipe._dit_cfg = object()

    def denoise(req, h, t, phases, entry, **kw):
        seen.append(kw)
        video = torch.randn(1, CH_LAT, P.latent_frames(req.num_frames), LH, LW,
                            generator=torch.Generator().manual_seed(int(req.seed)))
        audio = torch.randn(2, CH_AUD, P.audio_latents(req.num_frames))
        for fix, out in ((kw.get("video_fix"), video), (kw.get("audio_fix"), audio)):
            if fix is not None:
                n = sum(1 for v in fix[1] if v == 0.0)
                out[:, :, :n] = fix[0][:, :, :n]
        return video, audio, "on"

    monkeypatch.setattr(pipe, "_denoise", denoise)
    monkeypatch.setattr(pipe, "_decode", lambda req, v, a, entry, **kw: decoded.append(req) or 1)
    monkeypatch.setattr(pipe, "release_resident", lambda: None)
    clips = []
    for i, (n, refs) in enumerate(zip(frames, clip_refs)):
        inputs = []
        for kind, slot, max_sec in refs:
            path = tmp_path / (f"c{i}-{kind}{slot}.png" if kind == "picture" else f"c{i}-{kind}{slot}.bin")
            if kind == "picture":
                Image.new("RGB", (128, 64), (90,) * 3).save(path)
            else:
                path.write_bytes(b"x")
            inputs.append(P.H3ReferenceInput(kind, str(path), slot, max_sec))
        clips.append(CH.H3ChainClipSpec(f"clip {i}", n, references=tuple(inputs)))
    base = P.H3GenerateRequest(prompt="p", seed=1, width=64, height=64, num_frames=frames[0], output_path="o.mp4")
    done = CH.run_chain_h3(pipe, CH.H3ChainSpec(base=base, clips=tuple(clips), seed=1),
                           lambda stage, i, total, **kw: progress.append((stage, i, total, kw)), lambda m: None)
    return done, te_calls, [p for p in progress if p[0] == "encode"], seen, builds


def test_chain_clip_references_go_through_the_reference_steps(monkeypatch, tmp_path):
    """Clip 0: a video and a picture reference; clip 1: none (as before); clip 2: an
    audio reference only (no video encoder). Each clip's references reach its own
    text encoder and denoise, next to the fixed head, with the chain's coefficient."""
    clip_refs = [[("video", 1, None), ("picture", 1, None)], [], [("audio", 1, None)]]
    done, te_calls, progress, kw, builds = _ref_chain_job(monkeypatch, tmp_path, clip_refs)
    assert te_calls == [
        {"prompt": "clip 0", "images": 0, "refs": [("picture", 1), ("video", 1)]},
        {"prompt": "clip 1", "images": 0, "refs": None},          # the TE is called as before
        {"prompt": "clip 2", "images": 0, "refs": [("audio", 1)]},
    ]
    assert builds["video"] == 1                                   # clip 0's own; clip 2 has no visual reference
    assert [None if k["refs"] is None else [s.kind for s, _r in k["refs"]] for k in kw] == [
        ["picture", "video"], None, ["audio"]]
    assert all(k["video_inject_clean"] == CH.CHAIN_INJECT_CLEAN == 1.0 for k in kw)
    assert kw[0]["video_fix"] is None and kw[1]["video_fix"] is not None and kw[2]["video_fix"] is not None
    for k in kw:
        for shape, rows in k["refs"] or ():
            assert rows.device.type == "cpu" and rows.dtype == torch.float32 and rows.shape[0] == shape.rows
    # encode: 3 (TE) + 1 (references) = 4 steps for a clip with references, 3 without;
    # the reference step is the last one and carries the clip's position.
    for i, want in enumerate((4, 3, 4)):
        lines = [p for p in progress if p[3]["outer_index"] == i]
        assert {total for _s, _i, total, _k in lines} == {want} and lines[-1][1:3] == (want, want)
    phases = done["h3"]["phases"]["clips"]
    assert [list(p) for p in phases] == [
        ["encode", "reference_encode", "denoise"], ["encode", "denoise"], ["encode", "reference_encode", "denoise"]]
    assert [(r["kind"], r["slot"]) for r in phases[0]["reference_encode"]["references"]] == [("picture", 1), ("video", 1)]
    assert phases[0]["reference_encode"]["references"][0]["vision_tokens"] == 5
    assert [(r["kind"], r["slot"]) for r in phases[2]["reference_encode"]["references"]] == [("audio", 1)]
    clips = done["chain"]["clips"]
    assert [c.get("references") for c in clips] == [
        {"pictures": 1, "videos": 1, "audios": 0}, None, {"pictures": 0, "videos": 0, "audios": 1}]
    assert [list(c["seconds"]) for c in clips] == [
        ["encode", "reference_encode", "denoise"], ["encode", "denoise"], ["encode", "reference_encode", "denoise"]]
    assert list(clips[1]) == ["prompt_sha256", "seconds", "latents_sha256", "head_equals_prev_tail"]
    assert list(clips[0]) == list(clips[1]) + ["references"]
    assert [c["head_equals_prev_tail"] for c in clips] == [None, True, True]
    assert "references" not in done["h3"]                         # the counts are per clip only


def test_chain_without_references_is_as_before(monkeypatch, tmp_path):
    done, te_calls, progress, kw, builds = _ref_chain_job(monkeypatch, tmp_path, [[], []], frames=(124, 39))
    assert [c["refs"] for c in te_calls] == [None, None] and builds["video"] == 0
    assert all(k["refs"] is None for k in kw)
    assert {total for _s, _i, total, _k in progress} == {3}
    assert all(list(p) == ["encode", "denoise"] for p in done["h3"]["phases"]["clips"])
    assert all("references" not in c and set(c["seconds"]) == {"encode", "denoise"} for c in done["chain"]["clips"])


def test_chain_clip_reference_failure_is_that_clips(monkeypatch, tmp_path):
    """A reference that cannot be used (an audio shorter than one latent) is refused
    at its own clip, not read ahead: clip 0 ran, nothing is decoded."""
    seen, decoded = [], []
    with pytest.raises(ValueError, match="参照音声が短すぎます"):
        _ref_chain_job(monkeypatch, tmp_path, [[], [("audio", 1, None)]], frames=(124, 39), samples=640,
                       seen=seen, decoded=decoded)
    assert len(seen) == 1 and seen[0]["refs"] is None     # only clip 0's denoise ran
    assert decoded == []


# --------------------------------------------------------------------------- #
# the REAL denoise on the tiny DiT (tests/test_h3_engine.py)
# --------------------------------------------------------------------------- #

def test_real_denoise_fixed_rows_are_clean_and_injected_as_is(monkeypatch, tmp_path):
    from test_h3_engine import LAT, _clean_latents, _tiny_denoiser

    from engine_h3.dit import patchify_video

    dit, run = _tiny_denoiser(monkeypatch, tmp_path)
    _clean_a, clean_v = _clean_latents()
    seen_video = []
    real_embed = dit.embed_rows

    def embed(text_refined, audio_rows, video_rows, cond_rows=None):
        seen_video.append(video_rows.clone())
        return real_embed(text_refined, audio_rows, video_rows, cond_rows=cond_rows)

    monkeypatch.setattr(dit, "embed_rows", embed)
    per = [0.0, 0.0, 0.0] + [1.0] * 4
    video, _audio, entry = run(video_fix=(clean_v, per), video_inject_clean=1.0)
    # The initial video noise (first draw of seed 11), as the module's own expression at c = 1.0.
    noise = torch.randn((1, LAT, 7, 4, 4), generator=torch.Generator().manual_seed(11), dtype=torch.float32)
    inject = patchify_video(1.0 * clean_v + 0.0 * noise)
    assert len(seen_video) == 3
    for rows in seen_video:
        assert torch.equal(rows[:12], inject[:12])       # frames 0..2 x grid 4: the latents themselves
    assert torch.equal(video[:, :, :3], clean_v[:, :, :3])
    assert not torch.equal(video[:, :, 3], clean_v[:, :, 3])
    assert entry["mask"] == {"video": {"rows": 28, "min": 0.0, "max": 1.0, "inject_clean": 1.0}}


def test_real_denoise_inject_clean_one_keeps_the_time_cap(monkeypatch, tmp_path):
    """The coefficient changes the injected value only: the fixed rows still carry t = 0.999."""
    from test_h3_engine import _clean_latents, _tiny_denoiser

    _dit, run = _tiny_denoiser(monkeypatch, tmp_path)
    _clean_a, clean_v = _clean_latents()
    seen_t = []
    real_time_rows = P.time_rows

    def time_rows(layout, t_v, t_a, device=None, **kw):
        out = real_time_rows(layout, t_v, t_a, device=device, **kw)
        seen_t.append((layout, out[0][out[1]]))
        return out

    monkeypatch.setattr(P, "time_rows", time_rows)
    run(video_fix=(clean_v, [0.0, 0.0, 0.0] + [1.0] * 4), video_inject_clean=1.0)
    assert len(seen_t) == 3
    for layout, t in seen_t:
        assert t[layout.video][:12].tolist() == [float(torch.tensor(0.999))] * 12   # frames 0..2 x grid 4


def test_real_denoise_in_a_chain_carries_the_tail(monkeypatch, tmp_path):
    """Two clips (22 + 39 frames) through run_chain_h3 with the real denoise: clip 1's
    head is clip 0's whole latent bit for bit, and the decode gets the timeline."""
    from test_h3_engine import TEXT, _fake_int8, _tiny_pipe_and_dit

    _fake_int8(monkeypatch)
    pipe, dit = _tiny_pipe_and_dit(tmp_path)
    pipe._dit_cfg = dit.cfg
    pipe._resident = {"dit": dit, "cfg": dit.cfg}
    text = torch.randn(5, TEXT, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    decoded = []

    def encode(prompt, phases, images=(), *, has_cond, has_audio=False, refs=None):
        return text, torch.ones(5, dtype=torch.long)

    def decode(req, video, audio, entry, keep=False, source_wave=None):
        decoded.append((req.num_frames, video.clone(), audio.clone(), keep))
        return 1

    monkeypatch.setattr(pipe, "_encode", encode)
    monkeypatch.setattr(pipe, "_decode", decode)
    spec = CH.H3ChainSpec(
        base=P.H3GenerateRequest(prompt="p", seed=0, width=64, height=64, num_frames=22, output_path="o.mp4",
                                 num_inference_steps=2, keep_resident=True),
        clips=(CH.H3ChainClipSpec("a", 22), CH.H3ChainClipSpec("b", 39)), seed=11,
    )
    done = CH.run_chain_h3(pipe, spec, lambda *a, **k: None, lambda m: None)
    assert [c["head_equals_prev_tail"] for c in done["chain"]["clips"]] == [None, True]
    frames, video, audio, keep = decoded[0]
    assert frames == 39 and keep is True                  # 22 + 39 - 22
    assert video.shape[2] == 12 and audio.shape[2] == 65  # T'(39) = 12, all of clip 0 is clip 1's head
    assert done["h3"]["phases"]["clips"][1]["denoise"]["mask"]["video"]["inject_clean"] == 1.0
    assert pipe._resident is not None and "dit" in pipe._resident


def test_real_denoise_in_a_chain_with_references(monkeypatch, tmp_path):
    """Two clips with references (one of each kind, as ``_encode_references`` hands
    them over) through run_chain_h3 with the real denoise: the reference rows are in
    each clip's sequence, clip 1's head still comes back as clip 0's tail bit for
    bit with the chain's coefficient, and the references reach the latents."""
    from test_h3_engine import TEXT, _fake_int8, _tiny_pipe_and_dit, _tiny_refs

    _fake_int8(monkeypatch)
    pipe, dit = _tiny_pipe_and_dit(tmp_path)
    pipe._dit_cfg = dit.cfg
    pipe._resident = {"dit": dit, "cfg": dit.cfg}
    text = torch.randn(5, TEXT, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    calls = []

    def encode(prompt, phases, images=(), *, has_cond, has_audio=False, refs=None):
        return text, torch.ones(5, dtype=torch.long)

    def encode_references(media, entry, index, total, enc, te_refs=()):
        calls.append((len(media), index, total, enc is not None))
        entry["references"] = []
        return _tiny_refs()

    monkeypatch.setattr(pipe, "_encode", encode)
    monkeypatch.setattr(pipe, "_decode", lambda req, video, audio, entry, keep=False, source_wave=None: 1)
    monkeypatch.setattr(pipe, "_read_references", lambda req: [{"kind": r.kind} for r in req.references])
    monkeypatch.setattr(pipe, "_te_references", lambda media: [])
    monkeypatch.setattr(pipe, "_build_video_encoder", lambda: object())
    monkeypatch.setattr(pipe, "_encode_references", encode_references)
    refs = []
    for kind in ("picture", "video", "audio"):
        path = tmp_path / f"r.{kind}"
        path.write_bytes(b"x")
        refs.append(P.H3ReferenceInput(kind, str(path), 1))

    def run(with_refs):
        clip_refs = tuple(refs) if with_refs else ()
        spec = CH.H3ChainSpec(
            base=P.H3GenerateRequest(prompt="p", seed=0, width=64, height=64, num_frames=22, output_path="o.mp4",
                                     num_inference_steps=2, keep_resident=True),
            clips=(CH.H3ChainClipSpec("a", 22, references=clip_refs), CH.H3ChainClipSpec("b", 39, references=clip_refs)),
            seed=11,
        )
        return CH.run_chain_h3(pipe, spec, lambda *a, **k: None, lambda m: None)

    done = run(True)
    assert calls == [(3, 4, 4, True)] * 2
    clips = done["chain"]["clips"]
    assert [c["head_equals_prev_tail"] for c in clips] == [None, True]
    assert all(c["references"] == {"pictures": 1, "videos": 1, "audios": 1} for c in clips)
    for p in done["h3"]["phases"]["clips"]:
        seq = p["denoise"]["sequence"]
        assert (seq["ref_video"], seq["ref_audio"]) == (4, 6)
    assert done["h3"]["phases"]["clips"][1]["denoise"]["mask"]["video"]["inject_clean"] == 1.0
    plain = run(False)
    assert plain["chain"]["clips"][0]["latents_sha256"] != clips[0]["latents_sha256"]
    assert "ref_video" not in plain["h3"]["phases"]["clips"][0]["denoise"]["sequence"]


# --------------------------------------------------------------------------- #
# video-to-video and clip 0's key frames (台帳 §1-121)
# --------------------------------------------------------------------------- #

SRC_SEED = 999


def _source_stubs(monkeypatch, pipe, rec, *, samples=29333):
    """The input video's reading (``samples`` of audio; None = a file without audio)
    and ``_encode_source_tail`` as stubs. The stub's latents are seeded with
    SRC_SEED; its ``freeze_ka`` is the real rule (tests/test_h3_source_tail.py)."""
    import numpy as np

    from engine_h3 import source_audio as SA
    from engine_h3 import source_tail as ST

    def read_tail(path, width, height, frames):
        rec.calls.append(("read_source", path, width, height, frames))
        return np.zeros((frames, height, width, 3), np.uint8), {"source_size": [1920, 1080], "frames_read": frames}

    def load_wave(path, *, keep_samples, read_samples):
        rec.calls.append(("read_source_audio", path, keep_samples, read_samples))
        return torch.zeros(2, samples), {}

    def encode_source_tail(frames, wave, entry, index, total):
        rec.calls.append(("source_encode", tuple(frames.shape), None if wave is None else int(wave.shape[-1]),
                          index, total))
        g = torch.Generator().manual_seed(SRC_SEED)
        video = torch.randn(1, CH_LAT, 7, LH, LW, generator=g)
        freeze = 0 if wave is None else min(37, int(wave.shape[-1]) // 800)
        audio = torch.randn(2, CH_AUD, freeze, generator=g) if freeze else None
        rec.source_latents = (video, audio)
        entry.update(frames_read=int(frames.shape[0]), freeze_ka=freeze, source_had_audio=wave is not None)
        pipe._progress("encode", index, total)
        return video, audio

    monkeypatch.setattr(ST, "read_source_tail", read_tail)
    monkeypatch.setattr(ST, "has_audio_stream", lambda path: samples is not None)
    monkeypatch.setattr(SA, "load_source_waveform", load_wave)
    monkeypatch.setattr(pipe, "_encode_source_tail", encode_source_tail)


def _v2v_spec(frames=(56, 39), context=22, **kw):
    spec = _spec(frames=frames, **kw)
    spec.source = CH.H3ChainSource(path="in/tail.mp4", context_frames=context)
    return spec


#: The chain's v2v block: the layout's six keys, then the run's (the LTX chain's sixteen).
GOLDEN_V2V_KEYS = [
    "source_context_px", "n_ctx_v", "n_ctx_a", "trim_px", "v2v_context_junction_px", "new_frames_px",
    "context_frames", "freeze_ka", "trimmed_px", "trimmed_audio_samples", "audio_fade_in_samples",
    "source_had_audio", "audio_head_frozen", "decoded_frames_px", "audio_handle_filename", "handle_context_seconds",
]


def test_carry_fix_without_audio_gives_no_audio_fix():
    pv = torch.randn(1, CH_LAT, 7, LH, LW)
    (cv, per_v), audio_fix = CH.carry_fix(pv, None, 17, 93, 7, 0)
    assert audio_fix is None
    assert torch.equal(cv[:, :, :7], pv) and per_v == [0.0] * 7 + [1.0] * 10


def test_v2v_source_is_clip_0s_fixed_head(monkeypatch):
    rec = _Recorder()
    pipe, original = _stub_pipe(monkeypatch, rec)
    _source_stubs(monkeypatch, pipe, rec)
    done = CH.run_chain_h3(pipe, _v2v_spec(keep_resident=True), _progress_into(rec), lambda m: None)
    layout = compute_h3_chain_layout([56, 39], source_video=True)

    kinds = [c[0] for c in rec.calls]
    # The tail (and its audio) is read before any model; its encode follows clip 0's text encoder.
    assert kinds == ["read_source", "read_source_audio", "encode", "source_encode", "denoise",
                     "encode", "denoise", "decode"]
    assert rec.calls[0] == ("read_source", "in/tail.mp4", 96, 64, 22)
    assert rec.calls[1] == ("read_source_audio", "in/tail.mp4", 1, 37 * 800)
    enc = [c for c in rec.calls if c[0] == "encode"]
    assert [c[3] for c in enc] == [True, False]              # clip 0 counts the VAE step
    assert rec.calls[3] == ("source_encode", (22, 64, 96, 3), 29333, 4, 4)

    # Clip 0's fixed head is the input video's latents: all 7 video, 36 audio (floor).
    src_v, src_a = rec.source_latents
    den = [c for c in rec.calls if c[0] == "denoise"]
    (cv0, per_v0), (ca0, per_a0) = den[0][2]["video_fix"], den[0][2]["audio_fix"]
    assert torch.equal(cv0[:, :, :7], src_v) and bool((cv0[:, :, 7:] == 0).all())
    assert torch.equal(ca0[:, :, :36], src_a) and bool((ca0[:, :, 36:] == 0).all())
    assert per_v0 == [0.0] * 7 + [1.0] * 10 and per_a0 == [0.0] * 36 + [1.0] * (93 - 36)
    assert den[0][2]["video_inject_clean"] == 1.0
    # Clip 1: as before, from clip 0's output.
    assert len([v for v in den[1][2]["video_fix"][1] if v == 0.0]) == 7
    assert [c["head_equals_prev_tail"] for c in done["chain"]["clips"]] == [True, True]

    # The decode: the whole timeline, trimmed by 22, with the handle beside the mp4.
    dec = next(c for c in rec.calls if c[0] == "decode")
    assert dec[1].num_frames == layout.total_frames == 73
    assert dec[5] == {"trim_head": 22, "handle_path": str(__import__("pathlib").Path("out/chain_audio_handle.wav"))}

    # Progress: clip 0's encode counts 4 steps, the source encode being the 4th.
    clip0 = [p for p in rec.progress if p[0] == "encode" and p[3]["outer_index"] == 0]
    clip1 = [p for p in rec.progress if p[0] == "encode" and p[3]["outer_index"] == 1]
    assert clip0[-1][1:3] == (4, 4) and clip1[-1][1:3] == (3, 3)
    assert pipe._progress is original

    # done: num_frames is the timeline's; chain = the plain keys, then v2v.
    assert done["num_frames"] == 73 and done["images_used"] == 0
    chain = done["chain"]
    plain = layout.to_dict()
    v2v_layout = plain.pop("v2v")
    assert list(chain) == list(plain) + ["seeds", "clips", "h3", "v2v"]
    assert {k: chain[k] for k in plain} == plain
    assert list(chain["v2v"]) == GOLDEN_V2V_KEYS
    assert chain["v2v"] == {
        **v2v_layout, "context_frames": 22, "freeze_ka": 36, "trimmed_px": 22, "trimmed_audio_samples": 29333,
        "audio_fade_in_samples": 960, "source_had_audio": True, "audio_head_frozen": True,
        "decoded_frames_px": 73, "audio_handle_filename": "chain_audio_handle.wav",
        "handle_context_seconds": round(22 / 24, 6),
    }
    assert v2v_layout == {"source_context_px": 22, "n_ctx_v": 7, "n_ctx_a": 37, "trim_px": 22,
                          "v2v_context_junction_px": 21, "new_frames_px": 51}
    assert list(chain["clips"][0]["seconds"]) == ["encode", "source_encode", "denoise"]
    assert list(chain["clips"][1]["seconds"]) == ["encode", "denoise"]
    assert list(done["h3"]["phases"]["clips"][0]) == ["encode", "source_encode", "denoise"]


def test_v2v_single_clip(monkeypatch):
    rec = _Recorder()
    pipe, _original = _stub_pipe(monkeypatch, rec)
    _source_stubs(monkeypatch, pipe, rec)
    done = CH.run_chain_h3(pipe, _v2v_spec(frames=(39,)), _progress_into(rec), lambda m: None)
    assert done["num_frames"] == 39 and done["chain"]["v2v"]["new_frames_px"] == 17
    assert [c["head_equals_prev_tail"] for c in done["chain"]["clips"]] == [True]


@pytest.mark.parametrize("samples, freeze", [(None, 0), (500, 0), (20000, 25)])
def test_v2v_without_or_with_short_audio(monkeypatch, samples, freeze):
    """No audio stream: nothing read, no audio head (the audio is generated freely).
    Audio shorter than the context: only the whole hops are frozen (``freeze_ka`` < 37)."""
    rec = _Recorder()
    pipe, _original = _stub_pipe(monkeypatch, rec)
    _source_stubs(monkeypatch, pipe, rec, samples=samples)
    done = CH.run_chain_h3(pipe, _v2v_spec(), _progress_into(rec), lambda m: None)
    assert ("read_source_audio" in [c[0] for c in rec.calls]) == (samples is not None)
    den = [c for c in rec.calls if c[0] == "denoise"]
    if freeze:
        assert den[0][2]["audio_fix"][1] == [0.0] * freeze + [1.0] * (93 - freeze)
        assert torch.equal(den[0][2]["audio_fix"][0][:, :, :freeze], rec.source_latents[1])
    else:
        assert den[0][2]["audio_fix"] is None
    assert den[0][2]["video_fix"] is not None and den[1][2]["audio_fix"] is not None
    v2v = done["chain"]["v2v"]
    assert (v2v["freeze_ka"], v2v["audio_head_frozen"], v2v["source_had_audio"]) == (
        freeze, freeze > 0, samples is not None)
    assert [c["head_equals_prev_tail"] for c in done["chain"]["clips"]] == [True, True]


def test_v2v_audio_track_without_decodable_samples_is_generated_freely(monkeypatch):
    """An audio stream that yields no sample (it ends before the context window) is not
    a failure: the audio is generated freely, as for a file without audio."""
    from engine_h3 import source_audio as SA

    rec = _Recorder()
    pipe, _original = _stub_pipe(monkeypatch, rec)
    _source_stubs(monkeypatch, pipe, rec)

    def no_samples(path, *, keep_samples, read_samples):
        raise ValueError("1 サンプルも読めませんでした")

    monkeypatch.setattr(SA, "load_source_waveform", no_samples)
    logs: list[str] = []
    done = CH.run_chain_h3(pipe, _v2v_spec(), _progress_into(rec), logs.append)
    den = [c for c in rec.calls if c[0] == "denoise"]
    assert den[0][2]["audio_fix"] is None and den[1][2]["audio_fix"] is not None
    v2v = done["chain"]["v2v"]
    assert v2v["source_had_audio"] is False and v2v["freeze_ka"] == 0 and v2v["audio_head_frozen"] is False
    assert any("generating audio freely" in m for m in logs)


def test_v2v_head_that_is_not_the_source_is_refused(monkeypatch):
    rec = _Recorder()
    pipe, _original = _stub_pipe(monkeypatch, rec, break_head=True)
    _source_stubs(monkeypatch, pipe, rec)
    with pytest.raises(RuntimeError, match="input video's tail"):
        CH.run_chain_h3(pipe, _v2v_spec(), _progress_into(rec), lambda m: None)


def _key_frame_files(tmp_path, frames):
    from PIL import Image

    out = []
    for n in frames:
        path = tmp_path / f"k{n}.png"
        Image.new("RGB", (128, 64), (n % 256,) * 3).save(path)
        out.append(P.H3ImageInput(path=str(path), frame_idx=n))
    return tuple(out)


def _image_stub(monkeypatch, pipe, rec):
    def encode_image(images, entry, index, total, enc=None):
        rec.calls.append(("image_encode", [im.size for im in images], index, total))
        pipe._progress("encode", index, total)
        return torch.ones(len(images) * 4, 96)

    monkeypatch.setattr(pipe, "_encode_image", encode_image)


def test_i2v_clip_0_runs_the_single_jobs_key_frame_steps(monkeypatch, tmp_path):
    rec = _Recorder()
    pipe, _original = _stub_pipe(monkeypatch, rec)
    _image_stub(monkeypatch, pipe, rec)
    images = _key_frame_files(tmp_path, (55, 0, 20))     # any order: sorted by frame
    spec = _spec(frames=(56, 39))
    spec.clips = (dataclasses_replace(spec.clips[0], images=images), spec.clips[1])
    done = CH.run_chain_h3(pipe, spec, _progress_into(rec), lambda m: None)
    kinds = [c[0] for c in rec.calls]
    assert kinds == ["encode", "image_encode", "denoise", "encode", "denoise", "release", "decode", "release"]
    enc = [c for c in rec.calls if c[0] == "encode"]
    assert len(enc[0][2]) == 2 and enc[0][3] is True      # the first and the last frame to the vision tower
    assert enc[1][2] == () and enc[1][3] is False
    assert rec.calls[1] == ("image_encode", [(96, 64)] * 3, 4, 4)
    den = [c for c in rec.calls if c[0] == "denoise"]
    assert den[0][2]["cond_frame_idx"] == [0, 20, 55] and den[0][2]["cond_clean"].shape == (12, 96)
    assert den[0][2]["video_fix"] is None
    assert den[1][2]["cond_clean"] is None and den[1][2]["cond_frame_idx"] == []
    assert done["images_used"] == 3
    clips = done["chain"]["clips"]
    assert list(clips[0]["seconds"]) == ["encode", "image_encode", "denoise"]
    assert [c["head_equals_prev_tail"] for c in clips] == [None, True]
    image_entry = done["h3"]["phases"]["clips"][0]["image_encode"]
    assert image_entry["canvas"] == [96, 64] and image_entry["source_size"] == [[128, 64]] * 3
    assert "v2v" not in done["chain"]
    dec = next(c for c in rec.calls if c[0] == "decode")
    assert dec[5] == {}


def test_i2v_clip_0_with_references_sends_no_vision_images(monkeypatch, tmp_path):
    rec = _Recorder()
    pipe, _original = _stub_pipe(monkeypatch, rec)
    _image_stub(monkeypatch, pipe, rec)
    monkeypatch.setattr(pipe, "_read_references", lambda req: [{"kind": "audio"}] if req.references else [])
    monkeypatch.setattr(pipe, "_te_references", lambda media: [])

    def encode_references(media, entry, index, total, enc, te_refs=()):
        rec.calls.append(("reference_encode", index, total))
        return []

    monkeypatch.setattr(pipe, "_encode_references", encode_references)
    ref_path = tmp_path / "a.wav"
    ref_path.write_bytes(b"x")
    spec = _spec(frames=(56, 39))
    spec.clips = (dataclasses_replace(spec.clips[0], images=_key_frame_files(tmp_path, (0, 55)),
                                      references=(P.H3ReferenceInput("audio", str(ref_path), 1),)), spec.clips[1])
    CH.run_chain_h3(pipe, spec, _progress_into(rec), lambda m: None)
    enc = [c for c in rec.calls if c[0] == "encode"]
    assert enc[0][2] == () and enc[0][3] is True
    assert [c for c in rec.calls if c[0] in ("image_encode", "reference_encode")] == [
        ("image_encode", [(96, 64)] * 2, 4, 5), ("reference_encode", 5, 5)]


def dataclasses_replace(obj, **kw):
    import dataclasses

    return dataclasses.replace(obj, **kw)


def _with_images(spec, i, tmp_path, frames=(0,)):
    clips = list(spec.clips)
    clips[i] = dataclasses_replace(clips[i], images=_key_frame_files(tmp_path, frames))
    spec.clips = tuple(clips)
    return spec


@pytest.mark.parametrize("make, match", [
    (lambda tp: _with_images(_spec(frames=(56, 39)), 1, tp), r"クリップ 0 にだけ.*clips\[1\]"),
    (lambda tp: _with_images(_v2v_spec(), 0, tp), "同時には使えません"),
    (lambda tp: _v2v_spec(context=21), "22 固定です（21）"),
    (lambda tp: _v2v_spec(context=73), "22 固定です（73）"),
    (lambda tp: _v2v_spec(frames=(22, 39)), r"clips\[0\]: num_frames must be at least 39 \(source context 22"),
    (lambda tp: _v2v_spec(frames=()), r"chain needs 1\.\.24 clips \(got 0\)"),
    (lambda tp: _with_images(_spec(frames=(56, 39)), 0, tp, frames=(56,)), "0〜55"),
])
def test_v2v_and_i2v_refused_before_any_stage(monkeypatch, tmp_path, make, match):
    rec = _Recorder()
    pipe, _original = _stub_pipe(monkeypatch, rec)
    _source_stubs(monkeypatch, pipe, rec)
    with pytest.raises(ValueError, match=match):
        CH.run_chain_h3(pipe, make(tmp_path), _progress_into(rec), lambda m: None)
    assert rec.calls == [("release",)]                   # nothing read, no stage ran
