"""The text encoder's unbuffered read (``engine_h3.unbuffered_read``, 台帳 1-303 方向 A), without a GPU.

Runs in ``.venv-engine-h3`` through the engine-venv runner (see ``tests/test_h3_engine.py``)::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_unbuffered_read.py

What is pinned:
  * the 4,096-byte window arithmetic at its edges;
  * the reader gives what ``read_tensors`` gives, byte for byte, dtype and shape
    (a header length that is not a multiple of 4,096, BF16 from an odd offset,
    1-byte dtypes, 0 elements, 0-dim, a file that does not end on a boundary,
    and one tensor that is at once larger than the staging buffer, unaligned and
    the file's last), with a small staging buffer (several windows) and the
    product's one; a held tensor survives the next read; no tensor points into
    the staging buffer;
  * the reader's file handle refuses a 1-byte read: it really is a
    ``FILE_FLAG_NO_BUFFERING`` handle;
  * ``load_mapped_to`` with and without a reader: the same weights, the same
    byte count, the same refusals;
  * the decoder-layer hooks with a reader give the fully loaded model's hidden state;
  * ``H3TextEncoder`` (a tiny Qwen3-VL behind it): ``load`` records
    ``unbuffered_read: "on"`` and the reader does the reads (embedding table,
    vision tower, layers); when the reader cannot be made (not Windows, the
    open, the trial read, the buffer) the buffered read gives the same hidden
    state and ``"on->off"`` with the reason is recorded; a read that fails in the
    middle of ``encode`` raises, every handle is closed and ``release`` (twice)
    lets the buffer go.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("comfy_kitchen")

import sft_quant_format  # noqa: E402
from engine.sft_quant.sft_reader import read_tensors  # noqa: E402
from engine_h3 import unbuffered_read as UR  # noqa: E402

ALIGN = UR.ALIGN
needs_windows = pytest.mark.skipif(sys.platform != "win32", reason="FILE_FLAG_NO_BUFFERING is Windows only")

_ST = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16", torch.int8: "I8", torch.uint8: "U8",
       torch.float8_e4m3fn: "F8_E4M3", torch.float8_e5m2: "F8_E5M2"}


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1).view(torch.uint8)


def _write_sft(path: Path, tensors: dict, *, data_base_mod: int | None = None) -> None:
    """``tests/test_h3_engine.py``'s writer, plus header padding (spaces) so the data
    section starts at ``data_base_mod`` modulo 4,096."""
    doc, blobs, off = {}, [], 0
    for name, ten in tensors.items():
        data = ten.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() if ten.numel() else b""
        doc[name] = {"dtype": _ST[ten.dtype], "shape": list(ten.shape), "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    head = json.dumps(doc).encode()
    if data_base_mod is not None:
        head += b" " * ((data_base_mod - 8 - len(head)) % ALIGN)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for b in blobs:
            f.write(b)


def _awkward_tensors() -> dict:
    g = torch.Generator().manual_seed(3)
    return {
        "u8_first": torch.randint(0, 256, (8,), generator=g, dtype=torch.uint8),
        "bf16_odd": torch.randn(5, 3, generator=g).to(torch.bfloat16),           # starts at an odd offset
        "i8_scalar": torch.tensor(-7, dtype=torch.int8),                           # 0-dim, 1 byte
        "f32_empty": torch.empty(0, 4, dtype=torch.float32),                      # 0 elements
        "f16_scalar": torch.tensor(1.5, dtype=torch.float16),                     # 0-dim
        "f8": torch.randn(33, generator=g).to(torch.float8_e4m3fn),
        "f32_cross": torch.randn(3000, generator=g),                              # crosses window boundaries
        "f8_e5m2": torch.randn(9, generator=g).to(torch.float8_e5m2),
        "bf16_tail": torch.randn(20011, generator=g).to(torch.bfloat16),          # > small buffer, unaligned, last
    }


@pytest.fixture()
def awkward(tmp_path):
    path = tmp_path / "awkward.safetensors"
    tensors = _awkward_tensors()
    _write_sft(path, tensors, data_base_mod=1001)
    header = sft_quant_format.read_header(str(path))
    # The shape the test is about: none of these is on a 4,096 boundary.
    assert header.data_base % ALIGN == 1001
    assert (header.data_base + header.tensors["bf16_odd"].data_offsets[0]) % 2 == 1
    tail = header.tensors["bf16_tail"]
    assert tail.data_offsets[1] == max(i.data_offsets[1] for i in header.tensors.values())
    assert (header.data_base + tail.data_offsets[0]) % ALIGN != 0
    assert header.file_size % ALIGN != 0
    return path, header, tensors


# --------------------------------------------------------------------------- #
# the window arithmetic (any platform)
# --------------------------------------------------------------------------- #

def test_aligned_window_edges():
    w = UR.aligned_window
    assert w(0, 0) == (0, 0, 0)
    assert w(0, 1) == (0, ALIGN, 0)
    assert w(0, ALIGN) == (0, ALIGN, 0)
    assert w(0, ALIGN + 1) == (0, 2 * ALIGN, 0)
    assert w(ALIGN - 1, 1) == (0, ALIGN, ALIGN - 1)
    assert w(ALIGN - 1, 2) == (0, 2 * ALIGN, ALIGN - 1)
    assert w(ALIGN, ALIGN) == (ALIGN, 2 * ALIGN, 0)
    assert w(5000, 0) == (ALIGN, 2 * ALIGN, 5000 - ALIGN)
    assert w(5000, 10) == (ALIGN, 2 * ALIGN, 5000 - ALIGN)
    assert w(3 * ALIGN + 7, 2 * ALIGN) == (3 * ALIGN, 6 * ALIGN, 7)
    with pytest.raises(ValueError):
        w(-1, 4)
    with pytest.raises(ValueError):
        w(0, -1)


def test_not_windows_is_unavailable(monkeypatch, tmp_path):
    """The module imports anywhere; making a reader off Windows says so (and allocates nothing)."""
    path = tmp_path / "x.safetensors"
    _write_sft(path, {"a": torch.zeros(4)})
    monkeypatch.setattr(UR, "_IS_WINDOWS", False)
    monkeypatch.setattr(UR, "_allocate_staging", lambda *a: pytest.fail("allocated off Windows"))
    with pytest.raises(UR.UnbufferedReadUnavailable, match="needs Windows"):
        UR.UnbufferedReader(str(path), "cpu")


# --------------------------------------------------------------------------- #
# the reader against read_tensors
# --------------------------------------------------------------------------- #

@needs_windows
@pytest.mark.parametrize("staging", [3 * ALIGN, 5 * ALIGN + 123, UR.STAGING_BYTES])
def test_reader_matches_read_tensors(awkward, staging):
    path, header, tensors = awkward
    keys = list(tensors)[::-1]   # asked in reverse: both answer in file order
    want = list(read_tensors(str(path), keys, header))
    reader = UR.UnbufferedReader(str(path), "cpu", staging_bytes=staging)
    try:
        staging_ptr = reader._staging.untyped_storage().data_ptr()
        assert reader._window.data_ptr() % ALIGN == 0
        got = list(reader.read_tensors(str(path), keys, header, torch.device("cpu")))
    finally:
        reader.close()
    # the same order (file order; the empty tensor shares its offset with the next one)
    assert [k for k, _ in got] == [k for k, _ in want]
    assert sorted(k for k, _ in got) == sorted(tensors)
    for (key, g), (_, w) in zip(got, want):
        assert g.dtype == w.dtype and g.shape == w.shape, key
        assert g.device.type == "cpu"
        assert torch.equal(_bytes(g), _bytes(w)), key
        assert torch.equal(_bytes(g), _bytes(tensors[key])), key
        if g.numel():
            assert g.untyped_storage().data_ptr() != staging_ptr, key   # nothing points into the buffer
    assert reader.closed


@needs_windows
def test_held_tensor_survives_the_next_read(awkward):
    path, header, tensors = awkward
    reader = UR.UnbufferedReader(str(path), "cpu", staging_bytes=3 * ALIGN)
    try:
        it = reader.read_tensors(str(path), ["bf16_odd", "f32_cross", "bf16_tail"], header, "cpu")
        _, first = next(it)
        kept = first.clone()
        _, second = next(it)
        _, third = next(it)
        it.close()
        assert torch.equal(_bytes(first), _bytes(kept))
        assert torch.equal(_bytes(first), _bytes(tensors["bf16_odd"]))
        assert torch.equal(_bytes(second), _bytes(tensors["f32_cross"]))
        assert torch.equal(_bytes(third), _bytes(tensors["bf16_tail"]))
        # and again, with the earlier results still held
        again = dict(reader.read_tensors(str(path), list(tensors), header, "cpu"))
        assert torch.equal(_bytes(first), _bytes(kept))
        assert torch.equal(_bytes(again["bf16_odd"]), _bytes(kept))
    finally:
        reader.close()


@needs_windows
def test_reader_handle_is_unbuffered(awkward):
    """The proof that the FILE_FLAG_NO_BUFFERING path is the one taken: the reader's
    handle accepts an aligned 4,096-byte read and refuses a 1-byte one
    (ERROR_INVALID_PARAMETER), which a buffered handle would serve."""
    path, header, _ = awkward
    reader = UR.UnbufferedReader(str(path), "cpu")
    try:
        handle = reader._open_handle()
        try:
            addr = reader._window.data_ptr()
            assert reader._read_file(handle, addr, ALIGN) == ALIGN
            with pytest.raises(OSError, match=r"ReadFile of 1 bytes of awkward\.safetensors failed: error 87"):
                reader._read_file(handle, addr, 1)
        finally:
            reader._close_handle(handle)
        with open(path, "rb") as f:      # the buffered read of the same byte is fine
            assert len(f.read(1)) == 1
    finally:
        reader.close()


@needs_windows
def test_reader_opens_and_closes_once_per_call(awkward, monkeypatch):
    path, header, tensors = awkward
    reader = UR.UnbufferedReader(str(path), "cpu", staging_bytes=3 * ALIGN)
    seen = {"open": 0, "close": 0, "fill": 0}
    real_open, real_close, real_fill = reader._open_handle, reader._close_handle, reader._fill

    def counted(name, fn):
        def wrap(*a):
            seen[name] += 1
            return fn(*a)
        return wrap

    monkeypatch.setattr(reader, "_open_handle", counted("open", real_open))
    monkeypatch.setattr(reader, "_close_handle", counted("close", real_close))
    monkeypatch.setattr(reader, "_fill", counted("fill", real_fill))
    list(reader.read_tensors(str(path), list(tensors), header, "cpu"))
    assert seen["open"] == seen["close"] == 1
    assert seen["fill"] > len([t for t in tensors.values() if t.numel()])   # the big ones took several windows
    reader.close()
    reader.close()   # twice is fine
    with pytest.raises(RuntimeError, match="closed"):
        list(reader.read_tensors(str(path), ["u8_first"], header, "cpu"))


@needs_windows
def test_reader_refuses_another_file_and_a_short_file(awkward, tmp_path):
    path, header, tensors = awkward
    reader = UR.UnbufferedReader(str(path), "cpu", staging_bytes=3 * ALIGN)
    try:
        with pytest.raises(ValueError, match="this reader reads awkward.safetensors"):
            list(reader.read_tensors(str(tmp_path / "other.safetensors"), ["u8_first"], header, "cpu"))
        # The file got shorter after the reader was made: a short read before the end it knew.
        with open(path, "r+b") as f:
            f.truncate(header.file_size - 5000)
        with pytest.raises(OSError, match="before the end of the file|short read") as info:
            list(reader.read_tensors(str(path), ["bf16_tail"], header, "cpu"))
        assert str(tmp_path) not in str(info.value)
    finally:
        reader.close()


@needs_windows
def test_reader_unavailable_reasons_are_facts_without_folders(tmp_path, monkeypatch):
    """A directory cannot be opened as a file: the real CreateFileW error, ASCII, file name only."""
    folder = tmp_path / "not_a_file.safetensors"
    folder.mkdir()
    with pytest.raises(UR.UnbufferedReadUnavailable) as info:
        UR.UnbufferedReader(str(folder), "cpu", staging_bytes=3 * ALIGN)
    reason = str(info.value)
    assert reason.startswith("unbuffered read: CreateFileW (FILE_FLAG_NO_BUFFERING) of not_a_file.safetensors failed: error ")
    assert reason.isascii() and "\n" not in reason
    assert str(tmp_path) not in reason
    # a missing file: the size check, no path either
    with pytest.raises(UR.UnbufferedReadUnavailable, match="getting the size of missing.safetensors failed") as info:
        UR.UnbufferedReader(str(tmp_path / "missing.safetensors"), "cpu", staging_bytes=3 * ALIGN)
    assert str(tmp_path) not in str(info.value)
    # the buffer
    path = tmp_path / "x.safetensors"
    _write_sft(path, {"a": torch.zeros(4)})

    def no_buffer(nbytes, pinned):
        raise RuntimeError("CUDA error: out of memory\nmore lines")

    monkeypatch.setattr(UR, "_allocate_staging", no_buffer)
    with pytest.raises(UR.UnbufferedReadUnavailable) as info:
        UR.UnbufferedReader(str(path), "cpu")
    assert str(info.value) == (
        f"unbuffered read: allocating the {UR.STAGING_BYTES}-byte staging buffer (not pinned) failed: "
        "RuntimeError: CUDA error: out of memory"
    )


def test_staging_is_the_exact_size_and_pinned_only_for_cuda(monkeypatch, tmp_path):
    asked = []

    def record(nbytes, pinned):
        asked.append((nbytes, pinned))
        raise RuntimeError("stop here")

    monkeypatch.setattr(UR, "_IS_WINDOWS", True)
    monkeypatch.setattr(UR, "_allocate_staging", record)
    for dev in ("cpu", "cuda"):
        with pytest.raises(UR.UnbufferedReadUnavailable):
            UR.UnbufferedReader(str(tmp_path / "x"), dev)
    assert asked == [(2**27, False), (2**27, True)]


# --------------------------------------------------------------------------- #
# load_mapped_to with and without a reader
# --------------------------------------------------------------------------- #

@needs_windows
def test_load_mapped_to_with_and_without_reader(tmp_path):
    from engine_h3.sft_load import drop_to_meta, load_mapped_to

    torch.manual_seed(0)
    src = torch.nn.Linear(40, 30).to(torch.bfloat16)
    path = tmp_path / "lin.safetensors"
    _write_sft(path, {"x.weight": src.weight.data, "x.bias": src.bias.data}, data_base_mod=77)
    header = sft_quant_format.read_header(str(path))
    plan = {"x.weight": "weight", "x.bias": "bias"}
    cpu = torch.device("cpu")

    def fresh():
        with torch.device("meta"):
            m = torch.nn.Linear(40, 30)
        m.requires_grad_(False)
        return m

    reader = UR.UnbufferedReader(str(path), cpu, staging_bytes=3 * ALIGN)
    try:
        a, b = fresh(), fresh()
        na = load_mapped_to(a, str(path), header, plan, cpu)
        nb = load_mapped_to(b, str(path), header, plan, cpu, reader=reader)
        assert na == nb == (40 * 30 + 30) * 2
        for (ka, ta), (kb, tb) in zip(a.state_dict().items(), b.state_dict().items()):
            assert ka == kb and ta.dtype == tb.dtype == torch.bfloat16 and torch.equal(ta, tb)
        assert not b.weight.requires_grad
        for use in (None, reader):
            m = fresh()
            with pytest.raises(RuntimeError, match="still on meta"):
                load_mapped_to(m, str(path), header, {"x.weight": "weight"}, cpu, reader=use)
            drop_to_meta(m)
            with pytest.raises(RuntimeError, match="no place"):
                load_mapped_to(m, str(path), header, {"x.weight": "weight", "x.bias": "nope"}, cpu, reader=use)
    finally:
        reader.close()


# --------------------------------------------------------------------------- #
# the text encoder
# --------------------------------------------------------------------------- #

@needs_windows
def test_layer_read_hooks_with_reader_match_the_loaded_model(tmp_path):
    """``test_te_direct_read_hooks_match_the_loaded_model`` with the reader handed in."""
    pytest.importorskip("transformers")
    from safetensors.torch import save_file
    from transformers import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLTextRotaryEmbedding

    from engine_h3.sft_load import load_mapped_to, meta_tensor_names, move_module_tree
    from engine_h3.text_encoder import install_layer_read_hooks, te_read_plan

    cfg = Qwen3VLTextConfig(
        hidden_size=64, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, vocab_size=97, max_position_embeddings=128,
        rope_scaling={"mrope_interleaved": True, "mrope_section": [2, 3, 3], "rope_type": "default"},
        rope_theta=10000, rms_norm_eps=1e-6,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    ref = Qwen3VLTextModel(cfg).to(torch.bfloat16).eval()
    ref.norm = torch.nn.Identity()
    with torch.no_grad():
        for prm in ref.parameters():
            prm.normal_(0.0, 0.05)
    path = tmp_path / "te.safetensors"
    save_file({f"model.{k}": v.contiguous() for k, v in ref.state_dict().items()}, str(path))
    ids = torch.tensor([[5, 17, 3, 90, 41, 8]])
    with torch.no_grad():
        want = ref(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state

    header = sft_quant_format.read_header(str(path))
    with torch.device("meta"):
        model = Qwen3VLTextModel(cfg)
    model.norm = torch.nn.Identity()
    model.to(torch.bfloat16)
    model.requires_grad_(False)
    model.eval()
    inv, scaling = Qwen3VLTextRotaryEmbedding.compute_default_rope_parameters(cfg, device="cpu")
    model.rotary_emb.inv_freq = inv.float()
    model.rotary_emb.original_inv_freq = inv.float().clone()
    model.rotary_emb.attention_scaling = scaling
    embed_plan, layer_plans, _visual = te_read_plan(header.tensors, cfg.num_hidden_layers)
    dev = torch.device("cpu")
    stats = {"bytes": 0, "read_seconds": 0.0, "layers": 0}
    reader = UR.UnbufferedReader(str(path), dev, staging_bytes=3 * ALIGN)   # several windows per tensor
    try:
        load_mapped_to(model.embed_tokens, str(path), header, embed_plan, dev, reader=reader)
        move_module_tree(model.rotary_emb, dev, skip=set())
        handles = install_layer_read_hooks(model.layers, layer_plans, str(path), header, dev, stats, reader=reader)
        with torch.no_grad():
            got = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
        for h in handles:
            h.remove()
    finally:
        reader.close()
    assert torch.equal(got, want)
    assert stats["layers"] == 3 and stats["bytes"] > 0
    left = meta_tensor_names(model)
    assert left and all(n.startswith("layers.") for n in left)


def _tiny_vl_config():
    from transformers import Qwen3VLConfig

    cfg = Qwen3VLConfig(
        image_token_id=95, video_token_id=96, vision_start_token_id=93, vision_end_token_id=94,
        tie_word_embeddings=False,
        text_config=dict(
            hidden_size=64, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, vocab_size=97, max_position_embeddings=256,
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
    return cfg


IDS_T2V = [5, 17, 3, 41, 8]
IDS_I2V = [5, 17, 93, 95, 95, 95, 95, 94, 3, 41, 8]


@pytest.fixture()
def tiny_te(tmp_path, monkeypatch):
    """``H3TextEncoder`` over a tiny Qwen3-VL: what ``load`` needs of the real files is
    faked (tokenizer, header inspection, config, rope check), the weights come
    from a synthetic Comfy-Org-named file. Returns (factory, wanted hiddens)."""
    pytest.importorskip("transformers")
    import h3_format
    import transformers
    from safetensors.torch import save_file
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLTextRotaryEmbedding

    import engine_h3.text_encoder as TE

    cfg = _tiny_vl_config()
    torch.manual_seed(0)
    ref = Qwen3VLModel(cfg).to(torch.bfloat16).eval()
    ref.language_model.norm = torch.nn.Identity()
    ref.visual.rotary_pos_emb.inv_freq = TE.vision_inv_freq(ref.visual.rotary_pos_emb.dim)
    r_inv, _ = Qwen3VLTextRotaryEmbedding.compute_default_rope_parameters(cfg.text_config, device="cpu")
    ref.language_model.rotary_emb.inv_freq = r_inv.float()
    ref.language_model.rotary_emb.original_inv_freq = r_inv.float().clone()
    with torch.no_grad():
        for prm in ref.parameters():
            prm.normal_(0.0, 0.05)
    names = {}
    for k, v in ref.state_dict().items():
        names[("model." + k[len("language_model."):]) if k.startswith("language_model.") else k] = v.contiguous()
    path = tmp_path / "te.safetensors"
    save_file(names, str(path))

    grid = torch.tensor([[1, 4, 4]])
    pixels = torch.randn(16, 3 * 2 * 16 * 16)
    want = {}
    with torch.no_grad():
        ids = torch.tensor([IDS_T2V])
        want["t2v"] = ref(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state[0]
        ids = torch.tensor([IDS_I2V])
        want["i2v"] = ref(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
            mm_token_type_ids=torch.tensor([[1 if i == 95 else 0 for i in IDS_I2V]]),
            pixel_values=pixels.to(torch.bfloat16), image_grid_thw=grid,
        ).last_hidden_state[0]

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: object()))
    monkeypatch.setattr(TE, "ensure_extra_special_tokens", lambda tok: {"ids": {}, "added_here": False})
    monkeypatch.setattr(h3_format, "inspect_h3_te", lambda p: (
        sft_quant_format.read_header(p), SimpleNamespace(num_layers=3, quant={})))
    monkeypatch.setattr(TE, "build_config", _tiny_vl_config)
    monkeypatch.setattr(TE, "check_rope", lambda c: {})
    monkeypatch.setattr(TE, "IMAGE_TOKEN_ID", 95)

    def presentation(tok, prompt, image_tokens=(), *, ref_items=None):
        ids = IDS_I2V if image_tokens else IDS_T2V
        return list(ids), [0 if i in (93, 94, 95) else 1 for i in ids]

    monkeypatch.setattr(TE, "presentation", presentation)
    monkeypatch.setattr(TE.H3TextEncoder, "vision_inputs", lambda self, images: (pixels, grid, [4]))

    def make():
        return TE.H3TextEncoder(str(path), str(tmp_path), device="cpu")

    return make, want, tmp_path


def _counting(monkeypatch):
    seen = {"open": 0, "close": 0}
    real_open, real_close = UR.UnbufferedReader._open_handle, UR.UnbufferedReader._close_handle

    def op(self):
        seen["open"] += 1
        return real_open(self)

    def cl(self, handle):
        seen["close"] += 1
        return real_close(self, handle)

    monkeypatch.setattr(UR.UnbufferedReader, "_open_handle", op)
    monkeypatch.setattr(UR.UnbufferedReader, "_close_handle", cl)
    return seen


def _run(te, images):
    info = te.load()
    try:
        hidden, tags = te.encode("a prompt", images=images)
    finally:
        reader = te._reader
        te.release()
    return info, hidden, tags, reader


@needs_windows
@pytest.mark.parametrize("kind", ["t2v", "i2v"])
def test_text_encoder_reads_unbuffered_and_records_on(tiny_te, monkeypatch, kind):
    make, want, _ = tiny_te
    seen = _counting(monkeypatch)
    te = make()
    info, hidden, tags, reader = _run(te, [object()] if kind == "i2v" else [])
    assert info["unbuffered_read"] == "on" and "unbuffered_read_reason" not in info
    assert torch.equal(hidden, want[kind])
    # trial read + embedding table (+ vision tower) + 3 layers, each opened and closed once
    assert seen["open"] == seen["close"] == 1 + 1 + (1 if kind == "i2v" else 0) + 3
    assert reader is not None and reader.closed and te._reader is None
    assert te.read_report["layers_read"] == 3


def _cannot_open(self):
    raise OSError("unbuffered read: CreateFileW (FILE_FLAG_NO_BUFFERING) of te.safetensors failed: error 1 (Incorrect function.)")


def _trial_read_fails(self, handle, address, nbytes):
    raise OSError(f"unbuffered read: ReadFile of {nbytes} bytes of te.safetensors failed: error 87 (The parameter is incorrect.)")


def _no_buffer(nbytes, pinned):
    raise RuntimeError("out of memory")


def _trial_read_not_oserror(self, handle, address, nbytes):
    import ctypes

    raise ctypes.ArgumentError("argument 2: TypeError: wrong type\nsecond line")


@pytest.mark.parametrize("case", ["not_windows", "open", "trial_read", "trial_read_not_oserror", "buffer"])
def test_text_encoder_falls_back_with_the_same_result(tiny_te, monkeypatch, case):
    make, want, tmp_path = tiny_te
    if case == "not_windows":
        monkeypatch.setattr(UR, "_IS_WINDOWS", False)
        expect = "unbuffered read needs Windows"
    elif sys.platform != "win32":
        pytest.skip("the other cases fail a step that only runs on Windows")
    elif case == "open":
        monkeypatch.setattr(UR.UnbufferedReader, "_open_handle", _cannot_open)
        expect = "unbuffered read: CreateFileW (FILE_FLAG_NO_BUFFERING) of te.safetensors failed: error 1"
    elif case == "trial_read":
        monkeypatch.setattr(UR.UnbufferedReader, "_read_file", _trial_read_fails)
        expect = "unbuffered read: ReadFile of 4096 bytes of te.safetensors failed: error 87"
    elif case == "trial_read_not_oserror":
        # not an OSError: still the buffered read, the reason names the step, the type and the first line
        monkeypatch.setattr(UR.UnbufferedReader, "_read_file", _trial_read_not_oserror)
        expect = ("unbuffered read: the trial read of the first 4096 bytes of te.safetensors failed: "
                  "ArgumentError: argument 2: TypeError: wrong type")
    else:
        monkeypatch.setattr(UR, "_allocate_staging", _no_buffer)
        expect = f"unbuffered read: allocating the {UR.STAGING_BYTES}-byte staging buffer (not pinned) failed: RuntimeError: out of memory"
    for kind in ("t2v", "i2v"):
        te = make()
        info, hidden, tags, reader = _run(te, [object()] if kind == "i2v" else [])
        assert info["unbuffered_read"] == "on->off"
        assert info["unbuffered_read_reason"].startswith(expect)
        assert info["unbuffered_read_reason"].isascii() and str(tmp_path) not in info["unbuffered_read_reason"]
        assert reader is None
        assert torch.equal(hidden, want[kind])


@needs_windows
def test_text_encoder_read_failing_mid_stage_raises_and_lets_go(tiny_te, monkeypatch):
    """A read that fails after the trial read (here: in the second decoder layer) is not
    a fallback: ``encode`` raises the OSError, every handle opened was closed, the
    layers are back on meta, and ``release`` (twice) lets the buffer go."""
    from engine_h3.sft_load import meta_tensor_names

    make, _, _ = tiny_te
    seen = _counting(monkeypatch)
    te = make()
    info = te.load()
    assert info["unbuffered_read"] == "on"
    reader = te._reader
    assert seen["open"] == 1                       # the trial read
    real_fill = UR.UnbufferedReader._fill
    fills = {"n": 0}

    def fill(self, handle, start, length):
        # opens: 1 trial read, 2 embedding table, 3 layer 0, 4 layer 1
        if seen["open"] == 4:
            raise OSError("unbuffered read: ReadFile of 4096 bytes of te.safetensors failed: error 23")
        fills["n"] += 1
        return real_fill(self, handle, start, length)

    monkeypatch.setattr(UR.UnbufferedReader, "_fill", fill)
    with pytest.raises(OSError, match="error 23"):
        te.encode("a prompt")
    assert fills["n"] >= 2                         # the embedding table and layer 0 were read
    assert seen["open"] == seen["close"] == 4
    assert all(n.startswith(("language_model.", "visual.")) for n in meta_tensor_names(te._model))
    assert not [n for n, p in te._model.language_model.named_parameters() if not p.is_meta]
    assert not reader.closed            # the stage's owner lets it go
    te.release()
    assert reader.closed and te._reader is None
    te.release()                        # twice is fine
