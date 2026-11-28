"""The text encoder's combined read (te_combined_read, 台帳 1-303 方向 B), without a GPU.

Runs in ``.venv-engine-h3`` through the engine-venv runner::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_te_combined.py

What is pinned (CPU: the wiring and the clean-up; the numbers 9,216 and 17
themselves are the GPU check's, only their edges are tested here):
  * the share decision (a pure function of the numbers): the edges 9216 / 9217
    and 16 / 17, a text row block cut short by the block boundary, P = 0, no
    vision tokens, one shareable item left alone, a reference part one row apart;
  * ``blocked_causal_attention`` with an offset: the query rows are the last
    n_q of the n_kv rows, the blocks are the n_kv rows' (cut at the offset);
    offset 0 is the call sequence of before;
  * ``H3TextEncoder.encode_many`` on a tiny Qwen3-VL (a character tokenizer, the
    REAL ``presentation``, synthetic pictures and video pairs): each item's
    output is ``encode``'s for that item, bit for bit, shared or not; each layer
    is read once whatever the number of items; a mismatch in a shared group's
    reference part stops it; a read or input failure leaves no weight behind;
  * a single job's ``_encode`` with the switch on (``generate``): a positive and
    a negative through one ``encode_many``; one side held, no negative, or the
    switch off: as before.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import sft_quant_format  # noqa: E402
from engine_h3 import text_encoder as TE  # noqa: E402

# --------------------------------------------------------------------------- #
# the share decision (numbers only)
# --------------------------------------------------------------------------- #


def test_text_attention_blocks_are_the_attentions_own():
    assert TE.text_attention_blocks(1100) == [(0, 512), (512, 1024), (1024, 1100)]
    assert TE.text_attention_blocks(1087) == [(0, 512), (512, 1087)]          # a tail of 63 folded
    assert TE.text_attention_blocks(1088) == [(0, 512), (512, 1024), (1024, 1088)]
    assert TE.text_attention_blocks(5, 2, 0) == [(0, 2), (2, 4), (4, 5)]


def test_shared_text_heights_cut_the_full_lengths_blocks_at_p():
    # one block, cut at P
    assert TE.shared_text_heights(300, 280) == [20]
    # the text crosses a boundary: rows 1000..1024 and 1024..1100
    assert TE.shared_text_heights(1100, 1000) == [24, 76]
    # P % 512 >= 496 and the text crosses the next boundary: a piece of 16 rows
    assert TE.shared_text_heights(1100, 1008) == [16, 76]
    # the same text whose tail is folded: one block
    assert TE.shared_text_heights(1080, 1000) == [80]


@pytest.mark.parametrize("n, p, facts", [
    (9216, 9000, []),
    (9217, 9000, ["tokens 9217 > 9216"]),
    (300, 283, []),                          # 17 text rows
    (300, 284, ["rows 16 < 17"]),            # 16 text rows
    (1100, 1007, []),                        # pieces of 17 and 76 rows
    (1100, 1008, ["rows 16 < 17"]),          # 17+ text tokens, a piece of 16 rows
    (9300, 9290, ["tokens 9300 > 9216", "rows 10 < 17"]),
])
def test_share_facts_edges(n, p, facts):
    assert TE.share_facts(n, p) == facts


def test_share_groups_per_item():
    a, b = ("ids-a", "tags", "key-a"), ("ids-b", "tags", "key-b")
    rows = [
        (400, 300, a),       # 0: shared with 1
        (420, 300, a),       # 1
        (305, 300, a),       # 2: same references, 5 text rows
        (400, 300, b),       # 3: references of its own
        (40, 0, None),       # 4: text only
        (60, 20, None),      # 5: audio only (no vision tokens)
        (9300, 9000, ("x", "t", "k")),   # 6, 7: same references, both too long
        (9400, 9000, ("x", "t", "k")),
        (400, 300, ("y", "t", "k")),     # 8: shareable, but its partner is not
        (303, 300, ("y", "t", "k")),     # 9
    ]
    groups, reasons = TE.share_groups(rows)
    assert groups == [[0, 1]]
    assert reasons == {
        2: "rows 5 < 17",
        3: "no_same_references",
        4: "no_vision_tokens",
        5: "no_vision_tokens",
        6: "tokens 9300 > 9216",
        7: "tokens 9400 > 9216",
        8: "others_with_same_references_not_shared",
        9: "rows 3 < 17",
    }


def test_share_groups_two_groups_and_a_reference_part_one_row_apart():
    a, b = ("ids-a", "t", "k"), ("ids-b", "t", "k")
    groups, reasons = TE.share_groups([(400, 300, a), (400, 300, b), (410, 300, a), (410, 300, b)])
    assert groups == [[0, 2], [1, 3]] and reasons == {}
    # one row of the reference part apart: another prefix, another group
    groups, reasons = TE.share_groups([(400, 300, ("ids", "t", "k")), (401, 301, ("ids+1", "t", "k"))])
    assert groups == [] and reasons == {0: "no_same_references", 1: "no_same_references"}


# --------------------------------------------------------------------------- #
# the attention with an offset
# --------------------------------------------------------------------------- #


def _recording(monkeypatch):
    import torch.nn.functional as F

    original = F.scaled_dot_product_attention
    calls = []

    def rec(query, *args, **kwargs):
        calls.append((query.storage_offset() // query.stride(2), query.shape[2]))
        return original(query, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", rec)
    return calls


@pytest.mark.parametrize("n, offset, block, min_tail, want_calls", [
    (1100, 0, 512, 64, [(0, 512), (512, 512), (1024, 76)]),
    (1100, 700, 512, 64, [(700, 324), (1024, 76)]),        # cut at the offset, tallest first
    (1100, 1000, 512, 64, [(1024, 76), (1000, 24)]),       # the text crosses a boundary
    (1087, 600, 512, 64, [(600, 487)]),                     # the folded block, cut
    (1100, 1024, 512, 64, [(1024, 76)]),                    # exactly on a boundary
    (13, 5, 4, 2, [(8, 5), (5, 3)]),                        # tiny: blocks (4,8),(8,13) cut to (5,8),(8,13)
])
def test_offset_attention_rows_are_the_full_lengths_rows(monkeypatch, n, offset, block, min_tail, want_calls):
    """The last n - offset query rows against all n key / value rows equal the same rows of
    the full computation (close on the CPU; bit equality is the GPU check's), with the
    full length's blocks cut at the offset. Each call is recorded as (absolute start row,
    height): the query is a slice of the full one."""
    torch.manual_seed(1)
    q = torch.randn(1, 4, n, 16)
    k = torch.randn(1, 2, n, 16)
    v = torch.randn(1, 2, n, 16)
    full = TE.blocked_causal_attention(q, k, v, scale=0.25, block=block, min_tail=min_tail)
    calls = _recording(monkeypatch)
    got = TE.blocked_causal_attention(q[:, :, offset:], k, v, scale=0.25, block=block, min_tail=min_tail)
    monkeypatch.undo()
    assert calls == want_calls
    assert got.shape == (1, 4, n - offset, 16)
    assert torch.allclose(got, full[:, :, offset:], rtol=1e-5, atol=1e-6)


def test_offset_attention_refuses_more_query_rows_than_keys():
    q = torch.randn(1, 4, 6, 16)
    k = torch.randn(1, 2, 5, 16)
    with pytest.raises(RuntimeError, match="query rows"):
        TE.blocked_causal_attention(q, k, k, scale=0.25)


# --------------------------------------------------------------------------- #
# a tiny Qwen3-VL behind H3TextEncoder, with the real presentation
# --------------------------------------------------------------------------- #

START, END, IMAGE, VIDEO = 93, 94, 95, 96


class CharTokenizer:
    """One id per character (1..90): deterministic, and a prompt's count is its length."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 90 + 1 for c in str(text)]}


def _tiny_config():
    from transformers import AttentionInterface, Qwen3VLConfig

    AttentionInterface.register(TE.H3_TEXT_ATTENTION, TE.h3_text_attention)
    cfg = Qwen3VLConfig(
        image_token_id=IMAGE, video_token_id=VIDEO, vision_start_token_id=START, vision_end_token_id=END,
        tie_word_embeddings=False,
        text_config=dict(
            hidden_size=64, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, vocab_size=97, max_position_embeddings=1024,
            rope_scaling={"mrope_interleaved": True, "mrope_section": [2, 3, 3], "rope_type": "default"},
            rope_theta=10000, rms_norm_eps=1e-6,
        ),
        vision_config=dict(
            depth=3, hidden_size=32, intermediate_size=48, num_heads=2, out_hidden_size=64,
            patch_size=16, spatial_merge_size=2, temporal_patch_size=2, in_channels=3,
            num_position_embeddings=16, deepstack_visual_indexes=[0, 1], hidden_act="gelu_pytorch_tanh",
        ),
    )
    for c in (cfg, cfg.text_config, cfg.vision_config):
        c._attn_implementation = "sdpa"
    cfg.text_config._attn_implementation = TE.H3_TEXT_ATTENTION
    return cfg


def _fake_vision_inputs(self, images):
    """Per image: a 4 x 4 patch grid (4 vision tokens), pixels drawn from the image's bytes."""
    rows, grids = [], []
    for img in images:
        seed = int(hashlib.sha256(img.tobytes()).hexdigest()[:8], 16)
        rows.append(torch.randn(16, 3 * 2 * 16 * 16, generator=torch.Generator().manual_seed(seed)))
        grids.append([1, 4, 4])
    return torch.cat(rows), torch.tensor(grids, dtype=torch.long), [4] * len(images)


@pytest.fixture()
def tiny(tmp_path, monkeypatch):
    """A loaded ``H3TextEncoder`` (CPU) over a tiny Qwen3-VL read from a Comfy-Org-named file."""
    import h3_format
    import transformers
    from safetensors.torch import save_file
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    cfg = _tiny_config()
    torch.manual_seed(0)
    ref = Qwen3VLModel(cfg).to(torch.bfloat16).eval()
    with torch.no_grad():
        for prm in ref.parameters():
            prm.normal_(0.0, 0.05)
    names = {}
    for k, v in ref.state_dict().items():
        if k.endswith("inv_freq"):
            continue
        names[("model." + k[len("language_model."):]) if k.startswith("language_model.") else k] = v.contiguous()
    path = tmp_path / "te.safetensors"
    save_file(names, str(path))
    del ref

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: CharTokenizer()))
    monkeypatch.setattr(TE, "ensure_extra_special_tokens", lambda tok: {"ids": {}, "added_here": False})
    monkeypatch.setattr(h3_format, "inspect_h3_te", lambda p: (
        sft_quant_format.read_header(p), SimpleNamespace(num_layers=3, quant={})))
    monkeypatch.setattr(TE, "build_config", _tiny_config)
    monkeypatch.setattr(TE, "check_rope", lambda c: {})
    for name, value in (("IMAGE_TOKEN_ID", IMAGE), ("VIDEO_TOKEN_ID", VIDEO),
                        ("VISION_START_ID", START), ("VISION_END_ID", END)):
        monkeypatch.setattr(TE, name, value)
    monkeypatch.setattr(TE.H3TextEncoder, "vision_inputs", _fake_vision_inputs)
    te = TE.H3TextEncoder(str(path), str(tmp_path), device="cpu")
    te.load()
    yield te
    te.release()


def _picture(seed, size=64):
    import numpy as np
    from PIL import Image

    arr = np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _pictures(*seeds):
    return [TE.RefMedia("picture", i + 1, image=_picture(s)) for i, s in enumerate(seeds)]


def _video(seed, pairs=2):
    import numpy as np

    rng = np.random.default_rng(seed)
    out = tuple((rng.integers(0, 256, (2, 64, 64, 3), dtype=np.uint8), f"<{j}.0 seconds>") for j in range(pairs))
    return [TE.RefMedia("video", 1, pairs=out)]


LONG = "The people of the pictures walk slowly across a stone bridge."   # 61 characters = 61 tokens
LONG2 = "A small dog runs along the shore while the waves roll in."


def _one(te, prompt, images=(), refs=None):
    with torch.no_grad():
        if refs:
            h, t = te.encode(prompt, images=list(images), refs=refs)
        else:
            h, t = te.encode(prompt, images=list(images))
    counts = {k: te.read_report[k] for k in ("tokens", "vision_tokens", "vision_tokens_per_image", "references")
              if k in te.read_report}
    return h.to("cpu"), t.to("cpu"), counts


def _many(te, items):
    with torch.no_grad():
        return te.encode_many(items)


def _check_equal(te, items, outs, counts):
    for (prompt, images, refs), (h, t), c in zip(items, outs, counts):
        wh, wt, wc = _one(te, prompt, images, refs)
        assert h.device.type == "cpu" and t.device.type == "cpu"
        assert h.dtype == torch.bfloat16 and t.dtype == torch.long
        assert torch.equal(h, wh) and torch.equal(t, wt)
        assert c == wc


@pytest.mark.parametrize("case", ["text", "mixed", "same", "different", "two_groups", "video", "with_negative"])
def test_encode_many_is_each_items_encode(tiny, case):
    pics = _pictures(1, 2)
    other = _pictures(3, 4)
    items = {
        "text": [(LONG, [], None), (LONG2, [], None), ("short", [], None)],
        "mixed": [(LONG, [_picture(5)], None), (LONG2, [], None), (LONG, [], pics)],
        "same": [(LONG, [], pics), (LONG2, [], pics), (LONG + " Calm.", [], pics)],
        "different": [(LONG, [], pics), (LONG2, [], other)],
        "two_groups": [(LONG, [], pics), (LONG, [], other), (LONG2, [], pics), (LONG2, [], other)],
        "video": [(LONG, [], _video(7)), (LONG2, [], _video(7)), (LONG, [], _video(8))],
        "with_negative": [(LONG, [], pics), (LONG2, [], pics), ("blurry, low quality", [], None)],
    }[case]
    outs, counts, report = _many(tiny, items)
    want_groups = {"text": [], "mixed": [], "same": [[0, 1, 2]], "different": [], "two_groups": [[0, 2], [1, 3]],
                   "video": [[0, 1]], "with_negative": [[0, 1]]}[case]
    assert [g["items"] for g in report["shared"]] == want_groups
    shared = {k for g in want_groups for k in g}
    assert sorted([e["item"] for e in report["not_shared"]] + sorted(shared)) == list(range(len(items)))
    assert report["items"] == len(items) and report["layers_read"] == 3
    assert set(report) == {"items", "read_gib", "read_seconds", "layers_read", "encode_seconds", "shared",
                           "not_shared"}
    _check_equal(tiny, items, outs, counts)


def test_shared_and_not_shared_in_one_read(tiny):
    """A group of two, a same-reference item too short to share, an item of its own, a text item."""
    pics = _pictures(1, 2)
    items = [(LONG, [], pics), ("short one", [], pics), (LONG2, [], pics), (LONG, [], _pictures(9)),
             (LONG2, [], None)]
    outs, counts, report = _many(tiny, items)
    assert report["shared"] == [{"reference_rows": len(_one(tiny, LONG, (), pics)[0]) - len(LONG), "items": [0, 2]}]
    assert report["not_shared"] == [{"item": 1, "reason": "rows 9 < 17"},
                                    {"item": 3, "reason": "no_same_references"},
                                    {"item": 4, "reason": "no_vision_tokens"}]
    _check_equal(tiny, items, outs, counts)


def test_each_layer_is_read_once_whatever_the_items(tiny, monkeypatch):
    real = TE.load_mapped_to
    reads = []

    def counting(module, path, header, mapping, device, *, reader=None):
        reads.append(id(module))
        return real(module, path, header, mapping, device, reader=reader)

    monkeypatch.setattr(TE, "load_mapped_to", counting)
    lm = tiny._model.language_model
    for n in (2, 5):
        del reads[:]
        _many(tiny, [(LONG if i % 2 else LONG2, [], _pictures(1)) for i in range(n)])
        layer_reads = [r for r in reads if r in {id(x) for x in lm.layers}]
        assert layer_reads == [id(x) for x in lm.layers]                 # each once, in order
        assert reads.count(id(lm.embed_tokens)) == 1 and reads.count(id(tiny._model.visual)) == 1
    del reads[:]
    _many(tiny, [(LONG, [], None), (LONG2, [], None)])
    assert id(tiny._model.visual) not in reads                           # no vision tokens: the tower is not read


def _all_meta(te):
    from engine_h3.sft_load import meta_tensor_names

    left = meta_tensor_names(te._model)
    params = [n for n, p in te._model.named_parameters() if not p.is_meta]
    return left, params


def test_a_read_failing_mid_layers_leaves_nothing_and_the_encoder_works_again(tiny, monkeypatch):
    real = TE.load_mapped_to
    lm = tiny._model.language_model
    count = {"layers": 0}

    def failing(module, path, header, mapping, device, *, reader=None):
        if module is lm.layers[1]:
            raise OSError("read failed in layer 1")
        if any(module is x for x in lm.layers):
            count["layers"] += 1
        return real(module, path, header, mapping, device, reader=reader)

    monkeypatch.setattr(TE, "load_mapped_to", failing)
    items = [(LONG, [], _pictures(1)), (LONG2, [], _pictures(1)), (LONG, [], None)]
    with pytest.raises(OSError, match="layer 1"):
        _many(tiny, items)
    assert count["layers"] == 1
    _left, params = _all_meta(tiny)
    assert params == []
    vrot = tiny._model.visual.rotary_pos_emb
    assert not vrot.inv_freq.is_meta                      # the computed buffer is back on the CPU
    monkeypatch.setattr(TE, "load_mapped_to", real)
    outs, counts, _report = _many(tiny, items)
    _check_equal(tiny, items, outs, counts)


def test_an_input_failure_reads_nothing(tiny, monkeypatch):
    reads = []
    monkeypatch.setattr(TE, "load_mapped_to", lambda *a, **k: reads.append(a[0]) or 0)
    with pytest.raises(ValueError, match="no frame pairs"):
        _many(tiny, [(LONG, [], None), (LONG, [], [TE.RefMedia("video", 1, pairs=())])])
    with pytest.raises(ValueError, match="プロンプトが空"):
        _many(tiny, [(LONG, [], None), ("", [], None)])
    with pytest.raises(ValueError, match="cannot both"):
        _many(tiny, [(LONG, [_picture(1)], _pictures(2)), (LONG, [], None)])
    with pytest.raises(ValueError, match="two or more"):
        _many(tiny, [(LONG, [], None)])
    assert reads == []


def test_a_shared_group_whose_reference_part_differs_at_run_time_stops(tiny, monkeypatch):
    """The run-time check of a group is not a fallback: a mismatch raises, and the weights go."""
    real = TE.check_shared_prefix
    seen = []

    def spoiled(states, p):
        seen.append(p)
        states[1]["cos"] = states[1]["cos"].clone()
        states[1]["cos"][0, 0, 0] += 1.0
        return real(states, p)

    monkeypatch.setattr(TE, "check_shared_prefix", spoiled)
    with pytest.raises(RuntimeError, match="cos"):
        _many(tiny, [(LONG, [], _pictures(1)), (LONG2, [], _pictures(1))])
    assert seen and _all_meta(tiny)[1] == []


def test_check_shared_prefix_names_what_differs():
    def state(n=10, vpm_text=False):
        h = torch.zeros(1, n, 4)
        vpm = torch.zeros(1, n, dtype=torch.bool)
        vpm[0, :3] = True
        if vpm_text:
            vpm[0, 8] = True
        return {"h": h, "cos": torch.zeros(1, n, 2), "sin": torch.zeros(1, n, 2), "tpos": None, "vpm": vpm}

    TE.check_shared_prefix([state(), state(12)], 6)
    bad = state()
    bad["h"][0, 2, 0] = 1.0
    with pytest.raises(RuntimeError, match="embeddings"):
        TE.check_shared_prefix([state(), bad], 6)
    bad = state()
    bad["sin"][0, 5, 1] = 1.0
    with pytest.raises(RuntimeError, match="sin"):
        TE.check_shared_prefix([state(), bad], 6)
    bad = state()
    bad["tpos"] = torch.zeros(1, 10, dtype=torch.long)
    with pytest.raises(RuntimeError, match="text position"):
        TE.check_shared_prefix([state(), bad], 6)
    with pytest.raises(RuntimeError, match="image rows"):
        TE.check_shared_prefix([state(), state(vpm_text=True)], 6)
    # a difference after the reference part is the text's own: fine
    other = state()
    other["h"][0, 7, 0] = 1.0
    TE.check_shared_prefix([state(), other], 6)


# --------------------------------------------------------------------------- #
# a single job's _encode with the switch on
# --------------------------------------------------------------------------- #


def _combined_pipe(monkeypatch):
    """tests/test_h3_te_cache.py's ``_te_pipe`` with ``encode_many`` added to its text encoder:
    each call is recorded in ``calls["many"]`` as the list of ``(prompt, images, refs)``."""
    from test_h3_te_cache import _te_pipe

    from engine_h3 import pipeline_h3 as P

    pipe, calls, progress, handed, builds = _te_pipe(monkeypatch)
    calls["many"] = []
    base = TE.H3TextEncoder

    class CombinedTE(base):
        def encode_many(self, items):
            calls["many"].append([(p, len(i), None if not r else [(x.kind, x.slot) for x in r]) for p, i, r in items])
            outs, counts = [], []
            for prompt, images, refs in items:
                h, t = self.encode(prompt, images, refs=refs or None)
                outs.append((h, t))
                counts.append(P._te_counts(self.read_report))
            report = {"items": len(items), "read_gib": 2.0, "read_seconds": 1.0, "layers_read": 50,
                      "encode_seconds": 3.0, "shared": [],
                      "not_shared": [{"item": k, "reason": "no_vision_tokens"} for k in range(len(items))]}
            return outs, counts, report

    monkeypatch.setattr(TE, "H3TextEncoder", CombinedTE)
    return pipe, calls, progress, handed, builds


def _job(pipe, *, prompt="p", neg="", combined=True, keep=False, refs=()):
    from engine_h3 import pipeline_h3 as P

    req = P.H3GenerateRequest(prompt=prompt, seed=1, width=64, height=64, num_frames=22, output_path="o.mp4",
                              negative_prompt=neg, te_combined_read=combined, keep_text_conditioning=keep,
                              references=tuple(refs))
    return pipe.generate(req)


def _enc(done):
    return done["h3"]["phases"]["encode"]


def _encode_lines(progress, since=0):
    return [p for p in progress[since:] if p[0] == "encode"]


def test_request_field_defaults_off():
    from engine_h3 import pipeline_h3 as P

    req = P.H3GenerateRequest(prompt="p", seed=1, width=64, height=64, num_frames=22, output_path="o.mp4")
    assert req.te_combined_read is False


def test_single_job_positive_and_negative_in_one_read(monkeypatch):
    pipe, calls, progress, handed, _b = _combined_pipe(monkeypatch)
    done = _job(pipe, neg="cars", combined=False)
    off_lines, off_enc = _encode_lines(progress), _enc(done)
    n = len(progress)
    done = _job(pipe, neg="cars")
    assert calls["many"] == [[("p", 0, None), ("cars", 0, None)]]
    assert calls["built"] == 2 and calls["loads"] == 2
    enc = _enc(done)
    assert _encode_lines(progress, n) == off_lines                    # the same progress lines
    counts = {"tokens": 3, "vision_tokens": 0, "vision_tokens_per_image": []}
    assert enc["te_read"] == counts and enc["te_read_negative"] == counts
    assert enc["te_combined"] == {"items": 2, "read_gib": 2.0, "read_seconds": 1.0, "layers_read": 50,
                                  "encode_seconds": 3.0, "shared": [],
                                  "not_shared": [{"clip": 0, "reason": "no_vision_tokens"}]}
    assert enc["te"] == off_enc["te"]
    assert "te_cache" not in enc and "conditioning_sha256" not in enc
    assert "te_combined" not in off_enc and "read_gib" in off_enc["te_read"]
    assert set(enc) - {"te_combined"} == set(off_enc)
    h = handed[-1]
    assert h["hidden"].device.type == "cpu" and h["negative"].device.type == "cpu"
    assert (enc["text_tokens"], enc["negative_tokens"]) == (3, 3)


def test_single_job_without_negative_or_switch_off_is_as_before(monkeypatch):
    pipe, calls, _progress, _handed, _b = _combined_pipe(monkeypatch)
    passed = []
    real = pipe._encode
    monkeypatch.setattr(pipe, "_encode", lambda *a, **k: passed.append(set(k)) or real(*a, **k))
    on = _job(pipe)                                  # one prompt: nothing to combine
    off = _job(pipe, combined=False, neg="cars")
    assert calls["many"] == []
    assert passed == [{"images", "has_cond", "has_audio", "refs"},
                      {"images", "has_cond", "has_audio", "refs", "negative_prompt"}]
    assert "te_combined" not in _enc(on) and "read_gib" in _enc(on)["te_read"]
    assert "te_combined" not in _enc(off)


def test_single_job_with_one_side_held_goes_through_encode(monkeypatch):
    pipe, calls, progress, _handed, _b = _combined_pipe(monkeypatch)
    _job(pipe, neg="cars", keep=True)                     # both put
    assert len(calls["many"]) == 1
    done = _job(pipe, prompt="q", neg="cars", keep=True)  # the negative held: one item
    assert len(calls["many"]) == 1 and calls["encodes"][-1] == ("q", 0, None)
    enc = _enc(done)
    assert enc["te_cache"] == {"positive": "miss", "negative": "hit"}
    assert "te_combined" not in enc and "read_gib" in enc["te_read"]
    n = len(calls["encodes"])
    done = _job(pipe, prompt="q", neg="cars", keep=True)  # both held: no text encoder
    assert len(calls["encodes"]) == n and _enc(done)["te_cache"] == {"positive": "hit", "negative": "hit"}
    assert "te" not in _enc(done)


def test_single_job_combined_with_the_cache_puts_both(monkeypatch):
    pipe, calls, _progress, handed, _b = _combined_pipe(monkeypatch)
    first = _job(pipe, neg="cars", keep=True)
    assert _enc(first)["te_cache"] == {"positive": "miss", "negative": "miss"}
    assert list(_enc(first)["conditioning_sha256"]) == ["positive", "tags", "negative"]
    assert len(pipe._te_conditioning) == 2
    second = _job(pipe, neg="cars", keep=True, combined=False)       # the held values are the same form
    assert _enc(second)["te_cache"] == {"positive": "hit", "negative": "hit"}
    assert handed[1]["hidden"] is handed[0]["hidden"] and handed[1]["negative"] is handed[0]["negative"]
    assert _enc(second)["conditioning_sha256"] == _enc(first)["conditioning_sha256"]
