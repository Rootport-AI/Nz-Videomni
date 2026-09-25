"""§3-167 B-1/B-2: sft_fp8_format — header reader and fp8 acceptance table.

Synthetic, tiny safetensors files only (no real 29 GB checkpoint): a header is
assembled from a {key: (dtype, shape)} spec, the body is zero bytes except for
``comfy_quant`` payloads, which carry real JSON.
"""

from __future__ import annotations

import json
import struct

import pytest

from sft_fp8_format import (
    DTYPE_ITEMSIZE,
    Fp8FormatError,
    Layout,
    MAX_HEADER_LEN,
    detect_prefix,
    inspect,
    parse_metadata,
    read_header,
    read_tensor_bytes,
)

P = "model.diffusion_model."
N_BLOCKS = 48
LINEAR = "attn1.to_q"
CONFIG = json.dumps({"transformer": {"num_layers": N_BLOCKS}, "vae": {}})
FMT_OK = json.dumps({"format": "float8_e4m3fn"}).encode()


def _write(path, spec: dict, metadata: dict | None = None, payloads: dict | None = None):
    """spec: key -> (dtype, shape). payloads: key -> bytes (must match size)."""
    payloads = payloads or {}
    header: dict = {}
    body = bytearray()
    for key, (dtype, shape) in spec.items():
        n = 1
        for d in shape:
            n *= d
        size = n * DTYPE_ITEMSIZE[dtype]
        data = payloads.get(key, b"\0" * size)
        assert len(data) == size, key
        header[key] = {"dtype": dtype, "shape": list(shape), "data_offsets": [len(body), len(body) + size]}
        body += data
    if metadata is not None:
        header["__metadata__"] = metadata
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(body))
    return path


def _quant_u8(fmt_bytes: bytes):
    return ("U8", (len(fmt_bytes),))


def _model(flavor: str = "scaled", *, blocks: int = N_BLOCKS, prefix: str = P, fp8_blocks=range(2, 46)):
    """Return (spec, metadata, payloads) of a minimal LTX-2.3-shaped transformer."""
    spec: dict = {}
    payloads: dict = {}
    for b in range(blocks):
        layer = f"{prefix}transformer_blocks.{b}.{LINEAR}"
        spec[f"{prefix}transformer_blocks.{b}.scale_shift_table"] = ("F32", (6, 4))
        spec[f"{layer}.bias"] = ("BF16", (4,))
        if b in fp8_blocks:
            spec[f"{layer}.weight"] = ("F8_E4M3", (4, 4))
            if flavor == "scaled":
                spec[f"{layer}.weight_scale"] = ("F32", ())
                spec[f"{layer}.comfy_quant"] = _quant_u8(FMT_OK)
                payloads[f"{layer}.comfy_quant"] = FMT_OK
        else:
            spec[f"{layer}.weight"] = ("BF16", (4, 4))
    spec[f"{prefix}video_embeddings_connector.proj.weight"] = ("BF16", (4, 4))
    spec[f"{prefix}audio_embeddings_connector.proj.bias"] = ("F32", (4,))
    spec[f"{prefix}patchify_proj.weight"] = ("BF16", (4, 4))
    # a monolithic checkpoint's non-transformer part (official fp8 style)
    spec["vae.decoder.conv.weight"] = ("BF16", (2, 2))
    metadata = {"config": CONFIG, "model_version": "2.3.0"}
    return spec, metadata, payloads


def _layer(b: int) -> str:
    return f"{P}transformer_blocks.{b}.{LINEAR}"


# --------------------------------------------------------------------------- #
# acceptance (owner ruling 2026-09-25: an fp8 file a typical ComfyUI workflow
# runs must run here when dropped in; judged layer by layer)
# --------------------------------------------------------------------------- #


def test_accepts_sulphur_style_scale_and_marker(tmp_path):
    spec, meta, pay = _model("scaled")
    layout = inspect(_write(tmp_path / "s.safetensors", spec, meta, pay))
    assert isinstance(layout, Layout)
    assert layout.flavor == "scaled"
    assert layout.n_blocks == N_BLOCKS
    assert layout.model_version == "2.3.0"
    assert layout.config == json.loads(CONFIG)
    assert layout.prefix == P
    # names line up with what LTXV_MODEL_COMFY_RENAMING_MAP leaves after the prefix
    assert layout.scaled_layers == frozenset(f"transformer_blocks.{b}.{LINEAR}" for b in range(2, 46))
    assert layout.connector_keys == (
        f"{P}audio_embeddings_connector.proj.bias",
        f"{P}video_embeddings_connector.proj.weight",
    )


def test_accepts_lightricks_official_style_scale_input_scale_metadata_marker(tmp_path):
    """weight_scale + input_scale per layer, no comfy_quant tensors, the marker
    lives in __metadata__._quantization_metadata instead."""
    fp8_blocks = range(4, 47)  # official distilled keeps 0,1,2,3,47 in bf16
    spec, meta, pay = _model("plain", fp8_blocks=fp8_blocks)
    for b in fp8_blocks:
        spec[f"{_layer(b)}.weight_scale"] = ("F32", ())
        spec[f"{_layer(b)}.input_scale"] = ("F32", ())
    meta["_quantization_metadata"] = json.dumps(
        {"format_version": "1.0", "layers": {_layer(b): {"format": "float8_e4m3fn"} for b in fp8_blocks}}
    )
    layout = inspect(_write(tmp_path / "o.safetensors", spec, meta, pay))
    assert layout.flavor == "scaled"
    assert layout.scaled_layers == frozenset(f"transformer_blocks.{b}.{LINEAR}" for b in fp8_blocks)


def test_accepts_unscaled_cast(tmp_path):
    spec, meta, pay = _model("plain", fp8_blocks=range(N_BLOCKS))
    del meta["model_version"]
    layout = inspect(_write(tmp_path / "p.safetensors", spec, meta, pay))
    assert layout.flavor == "plain"
    assert layout.scaled_layers == frozenset()
    assert layout.model_version is None


def test_accepts_scaled_and_unscaled_layers_mixed(tmp_path):
    spec, meta, pay = _model("scaled")
    for suffix in (".weight_scale", ".comfy_quant"):
        del spec[_layer(5) + suffix]
    pay.pop(_layer(5) + ".comfy_quant")
    layout = inspect(_write(tmp_path / "x.safetensors", spec, meta, pay))
    assert layout.flavor == "scaled"
    assert f"transformer_blocks.5.{LINEAR}" not in layout.scaled_layers
    assert f"transformer_blocks.6.{LINEAR}" in layout.scaled_layers


def test_accepts_scaled_layer_without_marker(tmp_path):
    spec, meta, pay = _model("scaled")
    del spec[_layer(5) + ".comfy_quant"]
    pay.pop(_layer(5) + ".comfy_quant")
    layout = inspect(_write(tmp_path / "x.safetensors", spec, meta, pay))
    assert f"transformer_blocks.5.{LINEAR}" in layout.scaled_layers


def test_accepts_e5m2(tmp_path):
    spec, meta, pay = _model("scaled")
    fmt = json.dumps({"format": "float8_e5m2"}).encode()
    spec[_layer(5) + ".weight"] = ("F8_E5M2", (4, 4))
    spec[_layer(5) + ".comfy_quant"] = _quant_u8(fmt)
    pay[_layer(5) + ".comfy_quant"] = fmt
    assert f"transformer_blocks.5.{LINEAR}" in inspect(_write(tmp_path / "e.safetensors", spec, meta, pay)).scaled_layers


def test_accepts_transformer_only_file(tmp_path):
    spec, meta, pay = _model("scaled")
    del spec["vae.decoder.conv.weight"]
    assert inspect(_write(tmp_path / "t.safetensors", spec, meta, pay)).flavor == "scaled"


def test_accepts_fp8_bias_and_custom_block_count(tmp_path):
    spec, meta, pay = _model("plain", blocks=3, fp8_blocks=range(3))
    spec[f"{_layer(0)}.bias"] = ("F8_E4M3", (4,))
    layout = inspect(_write(tmp_path / "b.safetensors", spec, meta, pay), expected_blocks=3)
    assert layout.n_blocks == 3 and layout.flavor == "plain"


def test_tensors_outside_prefix_are_ignored(tmp_path):
    """A monolithic checkpoint carries vae.* / audio_vae.* / vocoder.* /
    text_embedding_projection.* next to the transformer; only fp8 is policed there."""
    spec, meta, pay = _model("scaled")
    spec["vocoder.resblocks.0.num_batches_tracked"] = ("I64", ())
    spec["audio_vae.encoder.stats"] = ("F16", (2,))
    assert inspect(_write(tmp_path / "m.safetensors", spec, meta, pay)).flavor == "scaled"


def test_unreadable_quantization_metadata_is_ignored(tmp_path):
    spec, meta, pay = _model("scaled")
    meta["_quantization_metadata"] = "not json"
    assert inspect(_write(tmp_path / "q.safetensors", spec, meta, pay)).flavor == "scaled"


# --- B-2: rules widened for the real LTX 2.5 community fp8 files ------------ #

_BODY_LAYERS = frozenset(f"transformer_blocks.{b}.{LINEAR}" for b in range(2, 46))


def test_accepts_scale_of_shape_1(tmp_path):
    """ChrisColeTech fp8_scaled stores every weight_scale as F32 shape [1]."""
    spec, meta, pay = _model("scaled")
    for b in range(2, 46):
        spec[f"{_layer(b)}.weight_scale"] = ("F32", (1,))
    layout = inspect(_write(tmp_path / "s1.safetensors", spec, meta, pay))
    assert layout.scaled_layers == _BODY_LAYERS


def test_accepts_fp8_connector_without_scale(tmp_path):
    spec, meta, pay = _model("scaled")
    spec[f"{P}video_embeddings_connector.proj.weight"] = ("F8_E4M3", (4, 4))
    layout = inspect(_write(tmp_path / "c.safetensors", spec, meta, pay))
    assert f"{P}video_embeddings_connector.proj.weight" in layout.connector_keys
    assert layout.scaled_layers == _BODY_LAYERS


def test_accepts_fp8_connector_with_scale_but_keeps_it_out_of_scaled_layers(tmp_path):
    """A scaled connector obeys the body's rules but is the text encoder's layer:
    it must not reach scaled_layers (the transformer's module op would then
    look it up on LTXModel and fail)."""
    spec, meta, pay = _model("scaled")
    conn = f"{P}video_embeddings_connector.proj"
    spec[f"{conn}.weight"] = ("F8_E4M3", (4, 4))
    spec[f"{conn}.weight_scale"] = ("F32", (1,))
    layout = inspect(_write(tmp_path / "cs.safetensors", spec, meta, pay))
    assert f"{conn}.weight_scale" in layout.connector_keys
    assert layout.scaled_layers == _BODY_LAYERS


def test_accepts_fp8_connector_with_comfy_quant_marker(tmp_path):
    """A connector fp8 layer carrying its comfy_quant marker (U8) is accepted:
    the marker is optional and its U8 dtype was already cleared in step 3."""
    spec, meta, pay = _model("scaled")
    conn = f"{P}video_embeddings_connector.proj"
    spec[f"{conn}.weight"] = ("F8_E4M3", (4, 4))
    spec[f"{conn}.comfy_quant"] = _quant_u8(FMT_OK)
    pay[f"{conn}.comfy_quant"] = FMT_OK
    layout = inspect(_write(tmp_path / "cq.safetensors", spec, meta, pay))
    assert f"{conn}.comfy_quant" in layout.connector_keys
    assert layout.scaled_layers == _BODY_LAYERS


def test_refuses_non_scalar_scale_on_connector(tmp_path):
    spec, meta, pay = _model("scaled")
    conn = f"{P}video_embeddings_connector.proj"
    spec[f"{conn}.weight"] = ("F8_E4M3", (4, 4))
    spec[f"{conn}.weight_scale"] = ("F32", (4,))
    with pytest.raises(Fp8FormatError, match="per-row"):
        inspect(_write(tmp_path / "cr.safetensors", spec, meta, pay))


def test_accepts_bare_names(tmp_path):
    """No ``model.diffusion_model.`` prefix at all (ChrisColeTech). Everything
    is "inside" then; text_embedding_projection and the connectors (fp8 and
    scaled here) are checked but kept out of scaled_layers."""
    spec, meta, pay = _model("scaled", prefix="")
    spec["video_embeddings_connector.proj.weight"] = ("F8_E4M3", (4, 4))
    spec["video_embeddings_connector.proj.weight_scale"] = ("F32", (1,))
    spec["text_embedding_projection.video_aggregate_embed.weight"] = ("F8_E4M3", (4, 4))
    spec["text_embedding_projection.video_aggregate_embed.weight_scale"] = ("F32", ())
    del spec["vae.decoder.conv.weight"]
    layout = inspect(_write(tmp_path / "bare.safetensors", spec, meta, pay))
    assert layout.prefix == ""
    assert layout.n_blocks == N_BLOCKS
    assert layout.scaled_layers == _BODY_LAYERS
    assert layout.connector_keys == (
        "audio_embeddings_connector.proj.bias",
        "video_embeddings_connector.proj.weight",
        "video_embeddings_connector.proj.weight_scale",
    )


def test_bare_names_police_every_tensor(tmp_path):
    """With prefix "" nothing is outside: a non-float tensor anywhere is refused."""
    spec, meta, pay = _model("scaled", prefix="")
    spec["audio_vae.encoder.stats"] = ("F16", (2,))
    with pytest.raises(Fp8FormatError, match="F16"):
        inspect(_write(tmp_path / "bare.safetensors", spec, meta, pay))


@pytest.mark.parametrize(
    "keys, expected",
    [
        ([f"{P}transformer_blocks.0.x", "vae.a"], P),
        (["transformer_blocks.0.x", "video_embeddings_connector.w"], ""),
        # the prefixed spelling wins when (oddly) both exist
        ([f"{P}transformer_blocks.0.x", "transformer_blocks.0.x"], P),
    ],
    ids=["prefixed", "bare", "both"],
)
def test_detect_prefix(tmp_path, keys, expected):
    path = _write(tmp_path / "d.safetensors", {k: ("BF16", (1,)) for k in keys})
    assert detect_prefix(read_header(path)) == expected


def test_detect_prefix_refuses_unknown_prefix(tmp_path):
    path = _write(tmp_path / "d.safetensors", {"diffusion_model.transformer_blocks.0.x": ("BF16", (1,))})
    with pytest.raises(Fp8FormatError, match="接頭辞"):
        detect_prefix(read_header(path))


def test_parse_metadata_matches_ltx_core_1_2_shape(tmp_path):
    """Each value JSON-parsed when valid JSON, else kept as the raw string
    (ltx_core 1.2 SafetensorsModelStateDictLoader.metadata)."""
    meta = {
        "config": CONFIG,
        "gemma_source_checkpoint": json.dumps({"ltx_version": "2.5.0"}),
        "model_version": "2.5.0",  # not JSON -> raw string
        "license": "some license text",
        "n": "3",  # valid JSON -> 3
    }
    path = _write(tmp_path / "m.safetensors", {"a": ("BF16", (1,))}, meta)
    assert parse_metadata(read_header(path)) == {
        "config": json.loads(CONFIG),
        "gemma_source_checkpoint": {"ltx_version": "2.5.0"},
        "model_version": "2.5.0",
        "license": "some license text",
        "n": 3,
    }
    assert parse_metadata(read_header(_write(tmp_path / "e.safetensors", {"a": ("BF16", (1,))}))) == {}


# --------------------------------------------------------------------------- #
# refusal table
# --------------------------------------------------------------------------- #


def _marker(fmt: str):
    def mutate(spec, meta, pay):
        data = json.dumps({"format": fmt}).encode()
        spec[f"{_layer(5)}.comfy_quant"] = _quant_u8(data)
        pay[f"{_layer(5)}.comfy_quant"] = data

    mutate.__name__ = f"marker_{fmt}"
    return mutate


def _metadata_int8(spec, meta, pay):
    meta["_quantization_metadata"] = json.dumps(
        {"format_version": "1.0", "layers": {_layer(5): {"format": "int8_tensorwise"}},
         "full_precision_matrix_mult": True}
    )


def _int8_weight(spec, meta, pay):
    spec[f"{_layer(5)}.weight"] = ("I8", (4, 4))


def _f8_e8m0(spec, meta, pay):
    spec[f"{_layer(5)}.weight"] = ("F8_E8M0", (4, 4))


def _per_row_scale(spec, meta, pay):
    spec[f"{_layer(5)}.weight_scale"] = ("F32", (4,))


def _per_block_scale(spec, meta, pay):
    spec[f"{_layer(5)}.weight_scale"] = ("F32", (2, 2))


def _bf16_scale(spec, meta, pay):
    spec[f"{_layer(5)}.weight_scale"] = ("BF16", ())


def _orphan_scale(spec, meta, pay):
    spec[f"{_layer(0)}.weight_scale"] = ("F32", ())  # block 0 stays bf16


def _scaled_fp8_marker(spec, meta, pay):
    spec[f"{P}scaled_fp8"] = ("F8_E4M3", (0,))


def _scale_weight(spec, meta, pay):
    spec[f"{_layer(5)}.scale_weight"] = ("F32", ())


def _no_config(spec, meta, pay):
    del meta["config"]


def _config_not_json(spec, meta, pay):
    meta["config"] = "{nope"


def _no_transformer_in_config(spec, meta, pay):
    meta["config"] = json.dumps({"vae": {}})


def _drop_block_47(spec, meta, pay):
    for key in [k for k in spec if k.startswith(f"{P}transformer_blocks.47.")]:
        del spec[key]


def _no_fp8(spec, meta, pay):
    for key, (dtype, shape) in list(spec.items()):
        if dtype == "F8_E4M3":
            spec[key] = ("BF16", shape)
        if key.endswith((".weight_scale", ".comfy_quant")):
            del spec[key]
            pay.pop(key, None)


def _no_connector(spec, meta, pay):
    for key in [k for k in spec if "_embeddings_connector." in k]:
        del spec[key]


def _f16_tensor(spec, meta, pay):
    spec[f"{P}patchify_proj.weight"] = ("F16", (4, 4))


def _fp8_outside_prefix(spec, meta, pay):
    spec["vae.decoder.conv.weight"] = ("F8_E4M3", (2, 2))


def _fp8_3d(spec, meta, pay):
    spec[f"{P}patchify_proj.weight"] = ("F8_E4M3", (2, 2, 4))


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (_marker("int8_tensorwise"), "int8_tensorwise"),
        (_marker("asym_w4a8_int8"), "asym_w4a8_int8"),
        (_marker("nvfp4"), "nvfp4"),
        (_marker("mxfp8"), "mxfp8"),
        (_metadata_int8, "_quantization_metadata"),
        (_int8_weight, "I8"),
        (_f8_e8m0, "F8_E8M0"),
        (_per_row_scale, "per-row"),
        (_per_block_scale, "per-block"),
        (_bf16_scale, "F32 のスカラー倍率"),
        (_orphan_scale, "孤立した倍率"),
        (_scaled_fp8_marker, "旧形式"),
        (_scale_weight, "旧形式"),
        (_no_config, "config がありません"),
        (_config_not_json, "JSON として読めません"),
        (_no_transformer_in_config, "transformer がありません"),
        (_drop_block_47, "47 個"),
        (_no_fp8, "1 本もありません"),
        (_no_connector, "connector）がありません"),
        (_f16_tensor, "F16"),
        (_fp8_outside_prefix, "の外にあります"),
        (_fp8_3d, "2 次元 .weight"),
    ],
    ids=lambda v: getattr(v, "__name__", None),
)
def test_refusals(tmp_path, mutate, needle):
    spec, meta, pay = _model("scaled")
    mutate(spec, meta, pay)
    path = _write(tmp_path / "x.safetensors", spec, meta, pay)
    with pytest.raises(Fp8FormatError) as ei:
        inspect(path)
    message = str(ei.value)
    assert needle in message
    assert "\n" not in message  # one line


def test_refuses_wrong_prefix(tmp_path):
    spec, meta, pay = _model("scaled", prefix="diffusion_model.")
    path = _write(tmp_path / "x.safetensors", spec, meta, pay)
    with pytest.raises(Fp8FormatError, match="接頭辞"):
        inspect(path)


def test_refuses_data_offsets_out_of_range(tmp_path):
    spec, meta, pay = _model("scaled")
    path = _write(tmp_path / "x.safetensors", spec, meta, pay)
    raw = path.read_bytes()
    path.write_bytes(raw[:-4])  # body shorter than the header claims
    with pytest.raises(Fp8FormatError, match="data_offsets"):
        inspect(path)


# --------------------------------------------------------------------------- #
# read_header boundaries / read_tensor_bytes
# --------------------------------------------------------------------------- #


def test_read_header_fields(tmp_path):
    path = _write(tmp_path / "h.safetensors", {"a": ("BF16", (2, 3)), "b": ("F32", ())}, {"k": "v"})
    h = read_header(path)
    assert h.metadata == {"k": "v"}
    assert h.data_base == 8 + h.header_len
    assert h.file_size == path.stat().st_size
    assert h.tensors["a"].dtype == "BF16" and h.tensors["a"].shape == (2, 3)
    assert h.tensors["a"].data_offsets == (0, 12)
    assert h.tensors["b"].shape == () and h.tensors["b"].data_offsets == (12, 16)


def test_read_header_without_metadata(tmp_path):
    assert read_header(_write(tmp_path / "h.safetensors", {"a": ("U8", (1,))})).metadata == {}


@pytest.mark.parametrize(
    "raw",
    [
        b"",  # shorter than the length prefix
        struct.pack("<Q", 0) + b"{}",  # header length 0
        struct.pack("<Q", 100) + b"{}",  # longer than the file
        struct.pack("<Q", MAX_HEADER_LEN + 1) + b"{}",  # over the 100 MB cap
        struct.pack("<Q", 2) + b"!!",  # not JSON
        struct.pack("<Q", 2) + b"[]",  # not an object
    ],
    ids=["short", "zero", "past_eof", "over_cap", "bad_json", "not_object"],
)
def test_read_header_boundaries(tmp_path, raw):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(raw)
    with pytest.raises(Fp8FormatError):
        read_header(path)


def test_read_header_rejects_length_mismatch(tmp_path):
    blob = json.dumps({"a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 3]}}).encode()
    path = tmp_path / "m.safetensors"
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * 3)
    with pytest.raises(Fp8FormatError, match="一致しません"):
        read_header(path)


def test_read_header_rejects_unknown_dtype(tmp_path):
    blob = json.dumps({"a": {"dtype": "F4", "shape": [2], "data_offsets": [0, 1]}}).encode()
    path = tmp_path / "u.safetensors"
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0")
    with pytest.raises(Fp8FormatError, match="未知"):
        read_header(path)


def test_read_tensor_bytes(tmp_path):
    path = _write(
        tmp_path / "t.safetensors",
        {"a": ("U8", (3,)), "q": _quant_u8(FMT_OK)},
        payloads={"a": b"xyz", "q": FMT_OK},
    )
    h = read_header(path)
    assert read_tensor_bytes(path, h, "a") == b"xyz"
    assert read_tensor_bytes(path, h, "q") == FMT_OK


class _CountingOpen:
    """Wraps builtins.open for one path and totals the bytes read from it."""

    def __init__(self, target):
        import builtins

        self.real_open = builtins.open
        self.target = str(target)
        self.total = 0

    def __call__(self, file, mode="r", *args, **kwargs):
        fh = self.real_open(file, mode, *args, **kwargs)
        if str(file) != self.target:
            return fh
        counter = self
        real_read = fh.read

        class _Proxy:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                fh.close()

            def read(self, n=-1):
                data = real_read(n)
                counter.total += len(data)
                return data

            def __getattr__(self, name):
                return getattr(fh, name)

        return _Proxy()


def _big_body_model(tmp_path):
    """A scaled model whose fp8 weights are large, so reading the body shows."""
    spec, meta, pay = _model("scaled")
    for b in range(2, 46):
        spec[f"{_layer(b)}.weight"] = ("F8_E4M3", (64, 64))
    return _write(tmp_path / "big.safetensors", spec, meta, pay)


def test_read_header_reads_only_the_header(tmp_path, monkeypatch):
    import io

    path = _big_body_model(tmp_path)
    counter = _CountingOpen(path)
    monkeypatch.setattr(io, "open", counter)  # Path.open goes through io.open
    header = read_header(path)
    assert 0 < counter.total <= 8 + header.header_len
    assert header.file_size > 8 + header.header_len + 44 * 64 * 64  # the body is big


def test_inspect_reads_only_header_and_comfy_quant(tmp_path, monkeypatch):
    import io

    path = _big_body_model(tmp_path)
    header = read_header(path)
    quant_keys = [k for k in header.tensors if k.endswith(".comfy_quant")]
    quant_bytes = sum(header.tensors[k].data_offsets[1] - header.tensors[k].data_offsets[0] for k in quant_keys)
    counter = _CountingOpen(path)
    monkeypatch.setattr(io, "open", counter)
    inspect(path)
    assert quant_keys
    assert counter.total <= 8 + header.header_len + quant_bytes
    assert quant_bytes <= 64 * len(quant_keys)  # a few dozen bytes per marker


def test_module_is_torch_free():
    import sft_fp8_format

    source = open(sft_fp8_format.__file__, encoding="utf-8").read()
    for heavy in ("import torch", "import numpy", "import safetensors", "mmap"):
        assert heavy not in source
