"""The text encoder's conditioning kept between jobs (keep_text_conditioning, 台帳 §1-298).

Runs in ``.venv-engine-h3`` through the engine-venv runner::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_te_cache.py

CPU only, no weights: ``generate`` / ``run_chain_h3`` through the REAL
``_encode`` / ``_read_references`` / ``_encode_references`` with a stand-in text
encoder (counting its builds, loads and forwards), video encoder, audio encoder
and readers (tests/test_h3_engine.py's ``_ref_pipe``, tests/test_h3_chain.py's
``_ref_chain_job``); the denoise and the decode are stubs, except in the test of
the REAL ``_denoise`` on the tiny DiT. Plus the worker's key.
"""

from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("comfy_kitchen")

from engine_h3 import pipeline_h3 as P  # noqa: E402
from engine_h3 import ref_latent_cache as RC  # noqa: E402


# --------------------------------------------------------------------------- #
# a single job
# --------------------------------------------------------------------------- #

def _te_pipe(monkeypatch, video_frames=None):
    """``_ref_pipe``'s pipeline with a counting text encoder: ``calls["built"]`` /
    ``calls["loads"]`` / ``calls["encodes"]`` (``(prompt, images, refs or None)``).
    Forward n returns ``[3 + images, 8]`` bf16 filled with n (each forward told
    apart). The denoise records what it was handed (``handed``). Returns
    ``(pipe, calls, progress, handed, builds)``."""
    from test_h3_engine import _ref_pipe

    from engine_h3 import text_encoder as TE

    pipe, _te, progress, _kw, builds = _ref_pipe(monkeypatch, video_frames=video_frames)
    calls = {"built": 0, "loads": 0, "encodes": []}

    class FakeTE:
        def __init__(self, *a, **k):
            calls["built"] += 1
            self.read_report = {}

        def load(self):
            calls["loads"] += 1
            return {"loaded": True}

        def encode(self, prompt, images=(), *, refs=None):
            calls["encodes"].append((prompt, len(images), None if refs is None else [(r.kind, r.slot) for r in refs]))
            n = len(calls["encodes"])
            self.read_report = {
                "read_gib": 1.5, "read_seconds": 0.5, "layers_read": 2, "encode_seconds": 0.7,
                "tokens": 3 + len(images), "vision_tokens": 7 * len(images),
                "vision_tokens_per_image": [7] * len(images),
            }
            if refs is not None:
                self.read_report["references"] = [
                    {"kind": r.kind, "slot": r.slot, "vision_tokens": 5 * r.slot} for r in refs if r.kind == "picture"
                ]
            hidden = torch.full((3 + len(images), 8), float(n), dtype=torch.bfloat16)
            return hidden, torch.full((3 + len(images),), n, dtype=torch.long)

        def release(self):
            pass

    monkeypatch.setattr(TE, "H3TextEncoder", FakeTE)
    handed = []

    def denoise(req, h, t, phases, entry, **kw):
        handed.append({"hidden": h, "tags": t, "negative": kw.get("negative_hidden"), "refs": kw.get("refs")})
        return torch.zeros(1), torch.zeros(1), "on"

    monkeypatch.setattr(pipe, "_denoise", denoise)
    return pipe, calls, progress, handed, builds


def _frame_images(tmp_path, frames, name="k"):
    """``frames`` -> ``H3ImageInput`` on 64 x 64 files ``<name><frame>.png`` (written once)."""
    from PIL import Image

    out = []
    for n in frames:
        path = tmp_path / f"{name}{n}.png"
        if not path.exists():
            Image.new("RGB", (64, 64), (40,) * 3).save(path)
        out.append(P.H3ImageInput(str(path), n))
    return tuple(out)


def _job(pipe, *, prompt="p", neg="", refs=(), images=(), num_frames=22, width=64, keep=True,
         keep_refs=False, keep_resident=False):
    req = P.H3GenerateRequest(prompt=prompt, seed=1, width=width, height=64, num_frames=num_frames,
                              output_path="o.mp4", images=tuple(images), references=tuple(refs),
                              negative_prompt=neg, keep_text_conditioning=keep, keep_reference_latents=keep_refs,
                              keep_resident=keep_resident)
    return pipe.generate(req)


def _enc(done):
    return done["h3"]["phases"]["encode"]


def _cache(done):
    return _enc(done).get("te_cache")


def _encode_progress(progress, since=0):
    return [p for p in progress[since:] if p[0] == "encode"]


def _refs(tmp_path, entries=(("picture", 1, None), ("video", 1, None), ("audio", 1, None))):
    from test_h3_engine import _ref_inputs

    return _ref_inputs(tmp_path, list(entries))


def test_second_job_hits_both_without_the_text_encoder(monkeypatch, tmp_path):
    """The same prompt, negative and references: no text encoder at all (not built,
    not loaded), the very tensors of the first job reach the denoise, the progress
    lines are the first job's, and every record is the first job's."""
    pipe, calls, progress, handed, _builds = _te_pipe(monkeypatch)
    refs = _refs(tmp_path)
    first = _job(pipe, neg="cars", refs=refs)
    p1 = _encode_progress(progress)
    assert _cache(first) == {"positive": "miss", "negative": "miss"}
    assert (calls["built"], calls["loads"]) == (1, 1)
    assert calls["encodes"] == [("p", 0, [("picture", 1), ("video", 1), ("audio", 1)]), ("cars", 0, None)]
    assert len(pipe._te_conditioning) == 2
    n = len(progress)
    second = _job(pipe, neg="cars", refs=refs)
    assert _cache(second) == {"positive": "hit", "negative": "hit"}
    assert (calls["built"], calls["loads"], len(calls["encodes"])) == (1, 1, 2)
    assert _encode_progress(progress, n) == p1
    a, b = handed
    assert b["hidden"] is a["hidden"] and b["tags"] is a["tags"] and b["negative"] is a["negative"]
    e1, e2 = _enc(first), _enc(second)
    assert e2["conditioning_sha256"] == e1["conditioning_sha256"]
    assert list(e2["conditioning_sha256"]) == ["positive", "tags", "negative"]
    # A hit side's read report: the held numbers only (nothing was read, no "te").
    counts = {"tokens": 3, "vision_tokens": 0, "vision_tokens_per_image": [], "references": [
        {"kind": "picture", "slot": 1, "vision_tokens": 5}]}
    assert e2["te_read"] == counts and e1["te_read"] == {**e1["te_read"], **counts}
    assert e2["te_read_negative"] == {"tokens": 3, "vision_tokens": 0, "vision_tokens_per_image": []}
    assert "te" in e1 and "te" not in e2 and "read_gib" in e1["te_read"]
    assert (e2["text_tokens"], e2["negative_tokens"]) == (e1["text_tokens"], e1["negative_tokens"])
    r1 = first["h3"]["phases"]["reference_encode"]["references"]
    r2 = second["h3"]["phases"]["reference_encode"]["references"]
    assert r2[0]["vision_tokens"] == r1[0]["vision_tokens"] == 5   # read from te_read, the same on a hit


def test_only_the_changed_side_goes_through(monkeypatch, tmp_path):
    pipe, calls, _progress, handed, _builds = _te_pipe(monkeypatch)
    _job(pipe, neg="cars")
    done = _job(pipe, prompt="q", neg="cars")                  # the prompt only
    assert _cache(done) == {"positive": "miss", "negative": "hit"}
    assert calls["encodes"][2:] == [("q", 0, None)] and calls["built"] == 2
    assert handed[1]["negative"] is handed[0]["negative"]
    done = _job(pipe, prompt="q", neg="trees")                 # the negative only
    assert _cache(done) == {"positive": "hit", "negative": "miss"}
    assert calls["encodes"][3:] == [("trees", 0, None)] and calls["built"] == 3
    assert handed[2]["hidden"] is handed[1]["hidden"]
    # The progress of a job whose positive is held and whose negative is not: as before.
    assert "te" in _enc(done) and set(_enc(done)["te_read"]) == {"tokens", "vision_tokens", "vision_tokens_per_image"}


def test_a_changed_reference_goes_through_an_audio_alone_does_not(monkeypatch, tmp_path):
    pipe, calls, _progress, _handed, _builds = _te_pipe(monkeypatch)
    refs = _refs(tmp_path)
    _job(pipe, refs=refs)
    picture = tmp_path / "other.png"
    from PIL import Image

    Image.new("RGB", (128, 64), (10,) * 3).save(picture)
    other = [P.H3ReferenceInput("picture", str(picture), 1)] + refs[1:]
    assert _cache(_job(pipe, refs=other)) == {"positive": "miss"}
    assert len(calls["encodes"]) == 2
    # The audio file replaced (another size and time): the text encoder reads its card only.
    with open(refs[2].path, "ab") as f:
        f.write(b"more")
    assert _cache(_job(pipe, refs=other)) == {"positive": "hit"}
    assert len(calls["encodes"]) == 2
    # Another slot is another card: through.
    moved = other[:2] + [P.H3ReferenceInput("audio", refs[2].path, 2)]
    assert _cache(_job(pipe, refs=moved)) == {"positive": "miss"}


def test_resolution_counts_only_with_key_frames(monkeypatch, tmp_path):
    pipe, calls, _progress, _handed, _builds = _te_pipe(monkeypatch)
    _job(pipe)
    assert _cache(_job(pipe, width=96)) == {"positive": "hit"}           # a prompt-only job
    images = _frame_images(tmp_path, (0,))
    assert _cache(_job(pipe, images=images)) == {"positive": "miss"}
    assert _cache(_job(pipe, images=images)) == {"positive": "hit"}
    assert _cache(_job(pipe, images=images, width=96)) == {"positive": "miss"}   # the vision tower's canvas
    assert [e[1] for e in calls["encodes"]] == [0, 1, 1]


def test_a_last_frame_that_becomes_a_middle_one_keys_as_the_prompt_alone(monkeypatch, tmp_path):
    """Frame 21 is the last of 22 (to the vision tower) and a middle one of 39 (condition
    rows only): the 39-frame job's key is the prompt-only job's."""
    pipe, calls, _progress, _handed, _builds = _te_pipe(monkeypatch)
    images = _frame_images(tmp_path, (21,))
    assert _cache(_job(pipe, images=images, num_frames=22)) == {"positive": "miss"}
    assert _cache(_job(pipe, num_frames=39)) == {"positive": "miss"}
    assert _cache(_job(pipe, images=images, num_frames=39)) == {"positive": "hit"}
    assert [e[1] for e in calls["encodes"]] == [1, 0]
    assert P.te_key_frames(P.H3GenerateRequest(prompt="p", seed=1, width=64, height=64, num_frames=39,
                                               output_path="o.mp4", images=images)) == []


def test_key_frames_are_not_read_by_the_text_encoder_with_references(monkeypatch, tmp_path):
    """With a reference the key frames go to the DiT only: neither to the text encoder
    nor into the key (a job with or without them keys the same)."""
    pipe, calls, _progress, _handed, _builds = _te_pipe(monkeypatch)
    refs = _refs(tmp_path, [("audio", 1, None)])
    _job(pipe, refs=refs)
    assert _cache(_job(pipe, refs=refs, images=_frame_images(tmp_path, (0, 21)))) == {"positive": "hit"}
    assert [e[1] for e in calls["encodes"]] == [0]


def test_an_identity_that_cannot_be_taken_keeps_nothing_for_the_positive(monkeypatch, tmp_path):
    pipe, calls, _progress, _handed, _builds = _te_pipe(monkeypatch)
    refs = _refs(tmp_path, [("picture", 1, None)])
    real = RC.media_identity

    def failing(kind, path):
        if kind == "picture":
            raise OSError("no stat")
        return real(kind, path)

    monkeypatch.setattr(RC, "media_identity", failing)
    first = _job(pipe, neg="cars", refs=refs)
    second = _job(pipe, neg="cars", refs=refs)
    assert _cache(first) == {"negative": "miss"}                       # the positive was never looked up
    assert _cache(second) == {"negative": "hit"}                       # the negative is the text alone
    assert [e[0] for e in calls["encodes"]] == ["p", "cars", "p"]
    assert len(pipe._te_conditioning) == 1
    assert "conditioning_sha256" in _enc(second)
    # A key frame that cannot be stat'ed, the same way.
    monkeypatch.setattr(RC, "media_identity", lambda kind, path: failing("picture", path))
    images = _frame_images(tmp_path, (0,))
    done = [_job(pipe, images=images), _job(pipe, images=images)]
    assert [e[1] for e in calls["encodes"][3:]] == [1, 1]
    assert [_cache(d) for d in done] == [{}, {}]                        # nothing looked up: an empty record
    assert all(list(_enc(d)["conditioning_sha256"]) == ["positive", "tags"] for d in done)


def test_start_and_end_images_swapped_go_through(monkeypatch, tmp_path):
    """A at the first frame and B at the last, then B first and A last: the key keeps the
    order the text encoder reads them in, so the second job goes through."""
    pipe, calls, _progress, _handed, _builds = _te_pipe(monkeypatch)
    a = _frame_images(tmp_path, (0,), name="a")[0].path
    b = _frame_images(tmp_path, (0,), name="b")[0].path
    first = _job(pipe, images=(P.H3ImageInput(a, 0), P.H3ImageInput(b, 21)))
    second = _job(pipe, images=(P.H3ImageInput(b, 0), P.H3ImageInput(a, 21)))
    assert [_cache(first), _cache(second)] == [{"positive": "miss"}, {"positive": "miss"}]
    assert [e[1] for e in calls["encodes"]] == [2, 2]


def test_the_same_text_as_positive_and_negative_is_never_mixed_up(monkeypatch, tmp_path):
    """"x" held as a negative only: a job whose prompt is "x" misses (its own forward),
    and from then on both are held side by side."""
    pipe, calls, _progress, handed, _builds = _te_pipe(monkeypatch)
    _job(pipe, prompt="y", neg="x")
    done = _job(pipe, prompt="x", neg="x")
    assert _cache(done) == {"positive": "miss", "negative": "hit"}
    assert [e[0] for e in calls["encodes"]] == ["y", "x", "x"]
    assert handed[1]["negative"] is handed[0]["negative"] and handed[1]["hidden"] is not handed[0]["negative"]
    assert not torch.equal(handed[1]["hidden"], handed[1]["negative"])   # forward 3 / forward 2
    assert len(pipe._te_conditioning) == 2
    done = _job(pipe, prompt="x", neg="x")
    assert _cache(done) == {"positive": "hit", "negative": "hit"}
    assert handed[2]["hidden"] is handed[1]["hidden"] and handed[2]["negative"] is handed[1]["negative"]


# --------------------------------------------------------------------------- #
# the dropping rules
# --------------------------------------------------------------------------- #

def test_off_job_empties_it_and_records_as_before(monkeypatch, tmp_path):
    """Off: everything held goes at its start; the encode, its calls and its record are
    as before (no te_cache, no conditioning_sha256, ``_encode`` without the key)."""
    pipe, calls, _progress, _handed, _builds = _te_pipe(monkeypatch)
    passed = []
    real = pipe._encode
    monkeypatch.setattr(pipe, "_encode", lambda *a, **k: passed.append(set(k)) or real(*a, **k))
    refs = _refs(tmp_path)
    _job(pipe, neg="cars", refs=refs)
    assert len(pipe._te_conditioning) == 2 and "te_keys" in passed[-1]
    off = _job(pipe, neg="cars", refs=refs, keep=False)
    assert len(pipe._te_conditioning) == 0
    assert passed[-1] == {"images", "has_cond", "has_audio", "refs", "negative_prompt"}
    assert calls["encodes"][2:] == [("p", 0, [("picture", 1), ("video", 1), ("audio", 1)]), ("cars", 0, None)]
    assert calls["built"] == 2 and calls["loads"] == 2
    enc = _enc(off)
    assert "te_cache" not in enc and "conditioning_sha256" not in enc
    assert enc["te"] == {"loaded": True} and "read_gib" in enc["te_read"] and "read_gib" in enc["te_read_negative"]
    # A job without anything to key on, off: the call as before (no refs keyword, text only).
    _job(pipe, keep=False)
    assert calls["encodes"][-1] == ("p", 0, None) and passed[-1] == {"images", "has_cond", "has_audio", "refs"}


def test_a_job_without_references_keeps_its_conditioning(monkeypatch, tmp_path):
    pipe, calls, _progress, _handed, _builds = _te_pipe(monkeypatch)
    _job(pipe, refs=_refs(tmp_path))
    _job(pipe)
    assert len(pipe._te_conditioning) == 1                     # the prompt-only one; the other was not used
    assert _cache(_job(pipe)) == {"positive": "hit"} and len(calls["encodes"]) == 2


def test_a_failed_job_drops_what_it_put(monkeypatch, tmp_path):
    """A failed job drops only what it put: the negative it hit stays, its new positive goes."""
    pipe, calls, _progress, _handed, _builds = _te_pipe(monkeypatch)
    _job(pipe, neg="cars")
    real = pipe._denoise

    def failing(*a, **k):
        raise RuntimeError("denoise failed")

    monkeypatch.setattr(pipe, "_denoise", failing)
    with pytest.raises(RuntimeError, match="denoise failed"):
        _job(pipe, prompt="q", neg="cars")
    assert len(pipe._te_conditioning) == 1                     # "p" left at the start, "q" dropped at the end
    monkeypatch.setattr(pipe, "_denoise", real)
    assert _cache(_job(pipe, prompt="q", neg="cars")) == {"positive": "miss", "negative": "hit"}
    assert [e[0] for e in calls["encodes"]] == ["p", "cars", "q", "q"]


def test_a_refused_job_touches_nothing(monkeypatch, tmp_path):
    pipe, calls, _progress, _handed, _builds = _te_pipe(monkeypatch)
    _job(pipe, neg="cars")
    with pytest.raises(ValueError):
        _job(pipe, neg="cars", num_frames=23)
    with pytest.raises(ValueError):
        _job(pipe, neg="cars", keep=False, num_frames=23)       # an off job refused too: before its begin
    assert len(pipe._te_conditioning) == 2
    assert _cache(_job(pipe, neg="cars")) == {"positive": "hit", "negative": "hit"}
    assert calls["built"] == 1


def test_identities_are_taken_before_the_reads(monkeypatch, tmp_path):
    """A key frame's file is stat'ed before its canvas is read, a reference's before its read
    (begin, then the key, then the read); a file replaced while it is read is held under
    its old identity, so the next job goes through again."""
    from engine_h3 import image_cond as IC
    from engine_h3 import reference as RM

    pipe, calls, _progress, _handed, builds = _te_pipe(monkeypatch)
    order = builds["reads"]
    real_identity = RC.media_identity
    monkeypatch.setattr(RC, "media_identity", lambda kind, path: order.append(("stat", str(path)))
                        or real_identity(kind, path))
    real_canvas = IC.load_canvas_image

    def canvas(path, w, h):
        order.append(("load", str(path)))
        out = real_canvas(path, w, h)
        with open(path, "ab") as f:                             # replaced while it is read
            f.write(b"more")
        return out

    monkeypatch.setattr(IC, "load_canvas_image", canvas)
    images = _frame_images(tmp_path, (0,))
    _job(pipe, images=images)
    path = images[0].path
    assert order == [("stat", path), ("stat", path), ("load", path)]
    monkeypatch.setattr(IC, "load_canvas_image", real_canvas)
    assert _cache(_job(pipe, images=images)) == {"positive": "miss"}
    del order[:]
    real_read = RM.read_reference_video

    def replacing_read(p, **kw):
        out = real_read(p, **kw)
        with open(p, "ab") as f:
            f.write(b"more")
        return out

    monkeypatch.setattr(RM, "read_reference_video", replacing_read)
    refs = _refs(tmp_path, [("video", 1, None)])
    _job(pipe, refs=refs)
    vpath = refs[0].path
    assert order == [("stat", vpath), ("stat", vpath), ("read", vpath)]
    monkeypatch.setattr(RM, "read_reference_video", real_read)
    assert _cache(_job(pipe, refs=refs)) == {"positive": "miss"}


# --------------------------------------------------------------------------- #
# apart from the reference latents and from keep_resident
# --------------------------------------------------------------------------- #

def test_both_caches_on_do_not_touch_each_other(monkeypatch, tmp_path):
    pipe, calls, _progress, _handed, builds = _te_pipe(monkeypatch)
    refs = _refs(tmp_path)

    def latent(done):
        return [r.get("latent_cache") for r in done["h3"]["phases"]["reference_encode"]["references"]]

    first = _job(pipe, refs=refs, keep_refs=True)
    assert latent(first) == ["miss"] * 3 and _cache(first) == {"positive": "miss"}
    second = _job(pipe, prompt="q", refs=refs, keep_refs=True)
    assert latent(second) == ["hit"] * 3 and _cache(second) == {"positive": "miss"}
    third = _job(pipe, prompt="q", refs=refs, keep_refs=True)
    assert latent(third) == ["hit"] * 3 and _cache(third) == {"positive": "hit"}
    assert (len(pipe._ref_latents), len(pipe._te_conditioning)) == (3, 1)
    assert builds["video_encodes"] == 2 and len(calls["encodes"]) == 2


def test_conditioning_alone_leaves_the_reference_latents_alone(monkeypatch, tmp_path):
    """Only keep_text_conditioning: the reference latents stay empty and the reference
    records have no latent_cache (1-297's rule is untouched)."""
    pipe, calls, _progress, _handed, builds = _te_pipe(monkeypatch)
    refs = _refs(tmp_path)
    for _ in range(2):
        done = _job(pipe, refs=refs)
        recs = done["h3"]["phases"]["reference_encode"]["references"]
        assert all("latent_cache" not in r for r in recs)
        assert len(pipe._ref_latents) == 0
    assert builds["video_encodes"] == 4 and len(builds["audio_encodes"]) == 2   # encoded every time
    assert len(calls["encodes"]) == 1


def test_reference_latents_alone_run_the_text_encoder_every_time(monkeypatch, tmp_path):
    pipe, calls, _progress, _handed, _builds = _te_pipe(monkeypatch)
    refs = _refs(tmp_path)
    for _ in range(2):
        done = _job(pipe, refs=refs, keep=False, keep_refs=True)
        assert "te_cache" not in _enc(done) and "conditioning_sha256" not in _enc(done)
    assert len(calls["encodes"]) == 2 and len(pipe._te_conditioning) == 0 and len(pipe._ref_latents) == 3


def test_kept_without_keep_resident(monkeypatch, tmp_path):
    pipe, calls, _progress, _handed, _builds = _te_pipe(monkeypatch)
    _job(pipe, neg="cars", keep_resident=False)
    assert pipe._resident is None and len(pipe._te_conditioning) == 2
    pipe.release_resident()
    assert len(pipe._te_conditioning) == 2
    assert _cache(_job(pipe, neg="cars")) == {"positive": "hit", "negative": "hit"}
    assert calls["built"] == 1


# --------------------------------------------------------------------------- #
# the values
# --------------------------------------------------------------------------- #

def _raw_sha(t):
    """The bytes the real tensor holds, read another way (a numpy copy of an integer view)."""
    as_int = {2: torch.int16, 8: torch.int64}[t.element_size()]
    return hashlib.sha256(t.contiguous().view(as_int).numpy().tobytes()).hexdigest()


def test_conditioning_sha256_is_of_the_values_handed_over(monkeypatch, tmp_path):
    pipe, _calls, _progress, handed, _builds = _te_pipe(monkeypatch)
    done = _job(pipe, neg="cars")
    sha = _enc(done)["conditioning_sha256"]
    h = handed[0]
    assert sha == {"positive": _raw_sha(h["hidden"]), "tags": _raw_sha(h["tags"]), "negative": _raw_sha(h["negative"])}
    assert h["hidden"].dtype == torch.bfloat16 and h["tags"].dtype == torch.long
    done = _job(pipe, prompt="q")
    assert list(_enc(done)["conditioning_sha256"]) == ["positive", "tags"]   # no negative in this job
    assert list(_cache(done)) == ["positive"]


def test_tensor_sha256_reads_the_bytes_without_misreading_a_view():
    t = torch.arange(24, dtype=torch.float32).reshape(4, 6).to(torch.bfloat16)
    assert P.tensor_sha256(t) == _raw_sha(t)
    view = t.t()                                                 # not contiguous: the values in its order
    assert not view.is_contiguous() and P.tensor_sha256(view) == P.tensor_sha256(view.contiguous())
    assert P.tensor_sha256(view) != P.tensor_sha256(t)
    tags = torch.tensor([1, 0, 2], dtype=torch.long)
    assert P.tensor_sha256(tags) == hashlib.sha256(tags.numpy().tobytes()).hexdigest()


def test_image_vision_tokens_are_the_same_on_a_hit(monkeypatch, tmp_path):
    pipe, calls, _progress, _handed, _builds = _te_pipe(monkeypatch)
    images = _frame_images(tmp_path, (0, 21))
    first, second = _job(pipe, images=images), _job(pipe, images=images)
    assert [_cache(first), _cache(second)] == [{"positive": "miss"}, {"positive": "hit"}]
    v1 = first["h3"]["phases"]["image_encode"]["vision_tokens"]
    v2 = second["h3"]["phases"]["image_encode"]["vision_tokens"]
    assert v1 == v2 == 14 and len(calls["encodes"]) == 1
    assert _enc(second)["te_read"]["vision_tokens_per_image"] == [7, 7]


def test_held_conditioning_unchanged_by_the_real_denoise(monkeypatch, tmp_path):
    """The held tensors go to the REAL ``_denoise`` (tiny DiT, VSF on) as they are:
    nothing writes into them, and a second run on them gives the same latents."""
    from test_h3_engine import TEXT, _fake_int8, _tiny_pipe_and_dit

    _fake_int8(monkeypatch)
    pipe, dit = _tiny_pipe_and_dit(tmp_path)
    pipe._dit_cfg = dit.cfg
    pipe._resident = {"dit": dit, "cfg": dit.cfg}
    cache = pipe._te_conditioning
    pos_key, neg_key = (("positive", "p", (), (), None), ()), P.te_negative_key("cars")
    cache.begin({pos_key[0], neg_key[0]})
    g = torch.Generator().manual_seed(3)
    cache.put(pos_key, (torch.randn(5, TEXT, generator=g).to(torch.bfloat16), torch.ones(5, dtype=torch.long), {}))
    cache.put(neg_key, (torch.randn(3, TEXT, generator=g).to(torch.bfloat16), {}))
    snapshot = [cache.get(pos_key)[0].clone(), cache.get(pos_key)[1].clone(), cache.get(neg_key)[0].clone()]
    req = P.H3GenerateRequest(prompt="p", seed=11, width=64, height=64, num_frames=22, output_path="o.mp4",
                              num_inference_steps=2)

    def run():
        hidden, tags, _c = cache.get(pos_key)
        neg, _c = cache.get(neg_key)
        with torch.no_grad():
            video, _audio, _used = pipe._denoise(req, hidden, tags, P._Phases(cuda=False), {}, keep=True,
                                                 negative_hidden=neg, vsf_scale=3.0, vsf_report={})
        return video

    v1, v2 = run(), run()
    assert torch.equal(v1, v2)
    held = [cache.get(pos_key)[0], cache.get(pos_key)[1], cache.get(neg_key)[0]]
    assert all(torch.equal(a, b) for a, b in zip(held, snapshot))


# --------------------------------------------------------------------------- #
# a chain
# --------------------------------------------------------------------------- #

def _chain(monkeypatch, tmp_path, clip_refs, **kw):
    from test_h3_chain import _ref_chain_job

    kw.setdefault("shared_files", True)
    kw.setdefault("keep_text_conditioning", True)
    return _ref_chain_job(monkeypatch, tmp_path, clip_refs, **kw)


def _clip_caches(done):
    return [p["encode"].get("te_cache") for p in done["h3"]["phases"]["clips"]]


def test_chain_later_clips_and_the_next_chain_hit(monkeypatch, tmp_path):
    refs = [("picture", 1, None), ("video", 1, None), ("audio", 1, None)]
    done, te_calls, _p, kw, builds = _chain(monkeypatch, tmp_path, [refs] * 3, prompts=["same"] * 3,
                                            negative_prompt="cars")
    assert _clip_caches(done) == [{"positive": "miss", "negative": "miss"}, {"positive": "hit"}, {"positive": "hit"}]
    assert [c["prompt"] for c in te_calls] == ["same", "cars"]
    pipe = builds["pipe"]
    assert len(pipe._te_conditioning) == 2 and len(pipe._ref_latents) == 0
    done, te_calls, _p, kw2, _b = _chain(monkeypatch, tmp_path, [refs] * 3, prompts=["same"] * 3,
                                         negative_prompt="cars", pipe=pipe)
    assert _clip_caches(done) == [{"positive": "hit", "negative": "hit"}, {"positive": "hit"}, {"positive": "hit"}]
    assert te_calls == []
    assert kw2[0]["negative_hidden"] is kw[0]["negative_hidden"]
    shas = [p["encode"]["conditioning_sha256"] for p in done["h3"]["phases"]["clips"]]
    assert list(shas[0]) == ["positive", "tags", "negative"] and list(shas[1]) == ["positive", "tags"]
    assert shas[1]["positive"] == shas[0]["positive"]


def test_chain_one_changed_prompt_goes_through_alone(monkeypatch, tmp_path):
    done, te_calls, _p, _kw, builds = _chain(monkeypatch, tmp_path, [[]] * 3)
    assert _clip_caches(done) == [{"positive": "miss"}] * 3
    pipe = builds["pipe"]
    done, te_calls, _p, _kw, _b = _chain(monkeypatch, tmp_path, [[]] * 3, prompts=["clip 0", "changed", "clip 2"],
                                         pipe=pipe)
    assert _clip_caches(done) == [{"positive": "hit"}, {"positive": "miss"}, {"positive": "hit"}]
    assert [c["prompt"] for c in te_calls] == ["changed"]
    assert len(pipe._te_conditioning) == 3                       # "clip 1" was not used: dropped at the end


def test_chain_negative_is_looked_up_on_clip_0_only(monkeypatch, tmp_path):
    looked: list = []
    _done, _te, _p, _kw, builds = _chain(monkeypatch, tmp_path, [[]] * 3, negative_prompt="cars")
    pipe = builds["pipe"]
    real_get = pipe._te_conditioning.get
    monkeypatch.setattr(pipe._te_conditioning, "get", lambda key: looked.append(key[0][0]) or real_get(key))
    done, te_calls, _p, _kw, _b = _chain(monkeypatch, tmp_path, [[]] * 3, negative_prompt="cars", pipe=pipe)
    assert looked == ["positive", "negative", "positive", "positive"]
    assert te_calls == []
    assert [("negative" in c) for c in _clip_caches(done)] == [True, False, False]


def test_chain_clips_of_two_lengths_keep_both_keys(monkeypatch, tmp_path):
    """A 100-frame reference video reads 39 frames on a 39-frame clip and 56 on a 56-frame
    one: two positive keys of one head, both held, both hit by the next chain."""
    refs = [("video", 1, None)]
    done, te_calls, _p, _kw, builds = _chain(monkeypatch, tmp_path, [refs] * 3, frames=(39, 56, 39),
                                             prompts=["same"] * 3, video_frames=100)
    assert _clip_caches(done) == [{"positive": "miss"}, {"positive": "miss"}, {"positive": "hit"}]
    pipe = builds["pipe"]
    assert len(pipe._te_conditioning) == 2 and len(te_calls) == 2
    done, te_calls, _p, _kw, _b = _chain(monkeypatch, tmp_path, [refs] * 3, frames=(56, 39, 56),
                                         prompts=["same"] * 3, video_frames=100, pipe=pipe)
    assert _clip_caches(done) == [{"positive": "hit"}] * 3 and te_calls == []


def test_chain_a_new_start_image_goes_through_on_clip_0_only(monkeypatch, tmp_path):
    a, b = _frame_images(tmp_path, (0,), name="a")[0], _frame_images(tmp_path, (0,), name="b")[0]
    done, te_calls, _p, _kw, builds = _chain(monkeypatch, tmp_path, [[]] * 3, frames=(39, 39, 39),
                                             clip0_images=[(0, a.path)])
    assert _clip_caches(done) == [{"positive": "miss"}] * 3 and te_calls[0]["images"] == 1
    pipe = builds["pipe"]
    done, te_calls, _p, _kw, _b = _chain(monkeypatch, tmp_path, [[]] * 3, frames=(39, 39, 39),
                                         clip0_images=[(0, a.path)], pipe=pipe)
    assert _clip_caches(done) == [{"positive": "hit"}] * 3 and te_calls == []
    done, te_calls, _p, _kw, _b = _chain(monkeypatch, tmp_path, [[]] * 3, frames=(39, 39, 39),
                                         clip0_images=[(0, b.path)], pipe=pipe)
    assert _clip_caches(done) == [{"positive": "miss"}, {"positive": "hit"}, {"positive": "hit"}]
    assert [(c["prompt"], c["images"]) for c in te_calls] == [("clip 0", 1)]


def test_chain_begin_and_the_clips_make_the_same_heads(monkeypatch, tmp_path):
    """What ``begin`` keeps is exactly the heads the clips look up (a mismatch would drop
    the job's own entries): every key looked up has its head in the begin set."""
    pic = tmp_path / "k0.png"
    _frame_images(tmp_path, (0,))
    refs = [("picture", 1, None), ("video", 1, None), ("audio", 1, None)]
    kw = dict(negative_prompt="cars", clip0_images=[(0, str(pic))], prompts=["a", "b", "b"])
    first, _te, _p, _kw, builds = _chain(monkeypatch, tmp_path, [[], refs, refs], **kw)
    assert _clip_caches(first) == [{"positive": "miss", "negative": "miss"}, {"positive": "miss"},
                                   {"positive": "hit"}]
    pipe = builds["pipe"]
    begun, looked = [], []
    real_begin, real_get = pipe._te_conditioning.begin, pipe._te_conditioning.get
    monkeypatch.setattr(pipe._te_conditioning, "begin", lambda ids: begun.append(set(ids)) or real_begin(ids))
    monkeypatch.setattr(pipe._te_conditioning, "get", lambda key: looked.append(key) or real_get(key))
    done, te_calls, _p, _kw, _b = _chain(monkeypatch, tmp_path, [[], refs, refs], pipe=pipe, **kw)
    assert {k[0] for k in looked} == begun[0] and len(begun[0]) == 3   # clip 0's, clips 1 and 2's, the negative
    assert _clip_caches(done) == [{"positive": "hit", "negative": "hit"}, {"positive": "hit"}, {"positive": "hit"}]
    assert te_calls == []


def test_chain_off_keeps_nothing(monkeypatch, tmp_path):
    _done, _te, _p, _kw, builds = _chain(monkeypatch, tmp_path, [[]] * 2, frames=(39, 39))
    pipe = builds["pipe"]
    assert len(pipe._te_conditioning) == 2
    done, te_calls, _p, _kw, _b = _chain(monkeypatch, tmp_path, [[]] * 2, frames=(39, 39), pipe=pipe,
                                         keep_text_conditioning=False)
    assert len(pipe._te_conditioning) == 0 and len(te_calls) == 2
    for p in done["h3"]["phases"]["clips"]:
        assert "te_cache" not in p["encode"] and "conditioning_sha256" not in p["encode"]


def test_chain_off_passes_no_key_to_encode(monkeypatch, tmp_path):
    """Off: every clip's ``_encode`` gets the keywords it got before (no ``te_keys``); on, each gets it."""
    pic = tmp_path / "k0.png"
    _frame_images(tmp_path, (0,))
    refs = [("picture", 1, None), ("audio", 1, None)]
    kw = dict(frames=(39, 39), negative_prompt="cars", clip0_images=[(0, str(pic))])
    _done, _te, _p, _kw, builds = _chain(monkeypatch, tmp_path, [[], refs], **kw)
    pipe = builds["pipe"]
    passed = []
    real = pipe._encode
    monkeypatch.setattr(pipe, "_encode", lambda *a, **k: passed.append(set(k)) or real(*a, **k))
    _chain(monkeypatch, tmp_path, [[], refs], pipe=pipe, keep_text_conditioning=False, **kw)
    assert passed == [{"images", "has_cond", "refs", "negative_prompt"}, {"images", "has_cond", "refs"}]
    del passed[:]
    _chain(monkeypatch, tmp_path, [[], refs], pipe=pipe, **kw)
    assert passed == [{"images", "has_cond", "refs", "negative_prompt", "te_keys"},
                      {"images", "has_cond", "refs", "te_keys"}]


def test_chain_failed_clip_drops_what_the_chain_put(monkeypatch, tmp_path):
    _done, _te, _p, _kw, builds = _chain(monkeypatch, tmp_path, [[]] * 2, frames=(39, 39), prompts=["a", "b"])
    pipe = builds["pipe"]
    with pytest.raises(RuntimeError, match="denoise failed"):
        _chain(monkeypatch, tmp_path, [[]] * 2, frames=(39, 39), prompts=["a", "c"], pipe=pipe, fail_clip=1)
    assert len(pipe._te_conditioning) == 1                       # "a" (hit) stays, "c" (put) goes, "b" left at begin
    done, te_calls, _p, _kw, _b = _chain(monkeypatch, tmp_path, [[]] * 2, frames=(39, 39), prompts=["a", "c"],
                                         pipe=pipe)
    assert _clip_caches(done) == [{"positive": "hit"}, {"positive": "miss"}]
    assert [c["prompt"] for c in te_calls] == ["c"]


# --------------------------------------------------------------------------- #
# the worker's key
# --------------------------------------------------------------------------- #

BASE = {"prompt": "p", "seed": 1, "width": 640, "height": 384, "num_frames": 124, "output_path": "o.mp4"}


def test_worker_keep_text_conditioning_key(monkeypatch):
    from engine_h3 import worker

    assert "keep_text_conditioning" in worker._GENERATE_FIELDS
    assert "keep_text_conditioning" in worker._CHAIN_FIELDS
    assert worker._generate_request(dict(BASE)).keep_text_conditioning is False
    assert worker._generate_request({**BASE, "keep_text_conditioning": None}).keep_text_conditioning is False
    assert worker._generate_request({**BASE, "keep_text_conditioning": True}).keep_text_conditioning is True
    seen, logs = {}, []

    class StubPipe:
        def generate(self, req):
            seen["req"] = req
            return {}

    monkeypatch.setattr(worker, "_PIPE", StubPipe())
    monkeypatch.setattr(worker, "_ATTENTION", ("sdpa", False))
    monkeypatch.setattr(worker, "_log", logs.append)
    monkeypatch.setattr(worker, "_emit", lambda event, **f: None)
    worker._do_generate({**BASE, "op": "generate", "keep_text_conditioning": True})
    assert seen["req"].keep_text_conditioning is True and seen["req"].keep_reference_latents is False
    assert "keep_text_conditioning=True" in logs[-1] and "ignored keys" not in logs[-1]
    worker._do_generate({**BASE, "op": "generate"})
    assert seen["req"].keep_text_conditioning is False and "keep_text_conditioning=False" in logs[-1]


def test_worker_keep_text_conditioning_reaches_every_clip_of_a_chain(monkeypatch):
    import engine_h3.chain_h3 as CH
    from engine_h3 import worker

    seen, logs = {}, []

    def run(pipe, spec, progress, log):
        seen["spec"] = spec
        return {}

    monkeypatch.setattr(worker, "_PIPE", object())
    monkeypatch.setattr(worker, "_ATTENTION", ("sdpa", False))
    monkeypatch.setattr(worker, "_log", logs.append)
    monkeypatch.setattr(worker, "_emit", lambda event, **f: None)
    monkeypatch.setattr(CH, "run_chain_h3", run)
    worker._do_generate_chain({
        "op": "generate_chain", "seed": 1, "width": 640, "height": 384, "output_path": "o.mp4",
        "keep_text_conditioning": True,
        "clips": [{"prompt": "a", "num_frames": 124}, {"prompt": "b", "num_frames": 39}],
    })
    spec = seen["spec"]
    assert spec.base.keep_text_conditioning is True
    assert all(r.keep_text_conditioning is True for r in CH.clip_requests(spec))
    assert "keep_text_conditioning=True" in logs[-1] and "ignored keys" not in logs[-1]
