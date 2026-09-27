"""§3-167 B-1/B-2 + §3-168 C-1b: sft_quant_format — header reader, the scheme
table and the quantized-safetensors acceptance rules (VERIFICATION_LOG §121.3).

Synthetic, tiny safetensors files only (no real 29 GB checkpoint): a header is
assembled from a {key: (dtype, shape)} spec, the body is zero bytes except for
``comfy_quant`` payloads, which carry real JSON.
"""

from __future__ import annotations

import json
import struct

from collections import Counter

import pytest

from sft_quant_format import (
    DTYPE_ITEMSIZE,
    MAX_HEADER_LEN,
    SCHEME_TABLE,
    SCHEMES,
    SKIPPED_SUFFIXES,
    Layout,
    QuantFormatError,
    aux_shape,
    aux_specs,
    detect_prefix,
    inspect,
    layer_schemes,
    parse_metadata,
    read_header,
    read_tensor_bytes,
    weight_shape,
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


def _scaled_names(layout: Layout) -> frozenset[str]:
    """The fp8_scaled layers (what the retired ``Layout.scaled_layers`` held)."""
    return frozenset(name for name, scheme in layout.layers.items() if scheme == "fp8_scaled")


def _flavor(layout: Layout) -> str:
    """What the retired ``Layout.flavor`` said."""
    return "scaled" if _scaled_names(layout) else "plain"


# --------------------------------------------------------------------------- #
# acceptance (owner ruling 2026-09-25: an fp8 file a typical ComfyUI workflow
# runs must run here when dropped in; judged layer by layer)
# --------------------------------------------------------------------------- #


def test_accepts_sulphur_style_scale_and_marker(tmp_path):
    spec, meta, pay = _model("scaled")
    layout = inspect(_write(tmp_path / "s.safetensors", spec, meta, pay))
    assert isinstance(layout, Layout)
    assert _flavor(layout) == "scaled"
    assert layout.n_blocks == N_BLOCKS
    assert layout.model_version == "2.3.0"
    assert layout.config == json.loads(CONFIG)
    assert layout.prefix == P
    # names line up with what LTXV_MODEL_COMFY_RENAMING_MAP leaves after the prefix
    assert layout.layers == {f"transformer_blocks.{b}.{LINEAR}": "fp8_scaled" for b in range(2, 46)}
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
    assert _flavor(layout) == "scaled"
    assert _scaled_names(layout) == frozenset(f"transformer_blocks.{b}.{LINEAR}" for b in fp8_blocks)


def test_accepts_unscaled_cast(tmp_path):
    spec, meta, pay = _model("plain", fp8_blocks=range(N_BLOCKS))
    del meta["model_version"]
    layout = inspect(_write(tmp_path / "p.safetensors", spec, meta, pay))
    assert _flavor(layout) == "plain"
    assert layout.layers == {f"transformer_blocks.{b}.{LINEAR}": "fp8" for b in range(N_BLOCKS)}
    assert layout.model_version is None


def test_accepts_scaled_and_unscaled_layers_mixed(tmp_path):
    spec, meta, pay = _model("scaled")
    for suffix in (".weight_scale", ".comfy_quant"):
        del spec[_layer(5) + suffix]
    pay.pop(_layer(5) + ".comfy_quant")
    layout = inspect(_write(tmp_path / "x.safetensors", spec, meta, pay))
    assert _flavor(layout) == "scaled"
    assert layout.layers[f"transformer_blocks.5.{LINEAR}"] == "fp8"
    assert f"transformer_blocks.6.{LINEAR}" in _scaled_names(layout)


def test_accepts_scaled_layer_without_marker(tmp_path):
    spec, meta, pay = _model("scaled")
    del spec[_layer(5) + ".comfy_quant"]
    pay.pop(_layer(5) + ".comfy_quant")
    layout = inspect(_write(tmp_path / "x.safetensors", spec, meta, pay))
    assert f"transformer_blocks.5.{LINEAR}" in _scaled_names(layout)


def test_accepts_e5m2(tmp_path):
    spec, meta, pay = _model("scaled")
    fmt = json.dumps({"format": "float8_e5m2"}).encode()
    spec[_layer(5) + ".weight"] = ("F8_E5M2", (4, 4))
    spec[_layer(5) + ".comfy_quant"] = _quant_u8(fmt)
    pay[_layer(5) + ".comfy_quant"] = fmt
    assert f"transformer_blocks.5.{LINEAR}" in _scaled_names(inspect(_write(tmp_path / "e.safetensors", spec, meta, pay)))


def test_accepts_transformer_only_file(tmp_path):
    spec, meta, pay = _model("scaled")
    del spec["vae.decoder.conv.weight"]
    assert _flavor(inspect(_write(tmp_path / "t.safetensors", spec, meta, pay))) == "scaled"


def test_accepts_fp8_bias_and_custom_block_count(tmp_path):
    spec, meta, pay = _model("plain", blocks=3, fp8_blocks=range(3))
    spec[f"{_layer(0)}.bias"] = ("F8_E4M3", (4,))
    layout = inspect(_write(tmp_path / "b.safetensors", spec, meta, pay), expected_blocks=3)
    assert layout.n_blocks == 3 and _flavor(layout) == "plain"


def test_tensors_outside_prefix_are_ignored(tmp_path):
    """A monolithic checkpoint carries vae.* / audio_vae.* / vocoder.* /
    text_embedding_projection.* next to the transformer; only fp8 is policed there."""
    spec, meta, pay = _model("scaled")
    spec["vocoder.resblocks.0.num_batches_tracked"] = ("I64", ())
    spec["audio_vae.encoder.stats"] = ("F16", (2,))
    assert _flavor(inspect(_write(tmp_path / "m.safetensors", spec, meta, pay))) == "scaled"


def test_unreadable_quantization_metadata_is_ignored(tmp_path):
    spec, meta, pay = _model("scaled")
    meta["_quantization_metadata"] = "not json"
    assert _flavor(inspect(_write(tmp_path / "q.safetensors", spec, meta, pay))) == "scaled"


# --- B-2: rules widened for the real LTX 2.5 community fp8 files ------------ #

_BODY_LAYERS = frozenset(f"transformer_blocks.{b}.{LINEAR}" for b in range(2, 46))


def test_accepts_scale_of_shape_1(tmp_path):
    """ChrisColeTech fp8_scaled stores every weight_scale as F32 shape [1]."""
    spec, meta, pay = _model("scaled")
    for b in range(2, 46):
        spec[f"{_layer(b)}.weight_scale"] = ("F32", (1,))
    layout = inspect(_write(tmp_path / "s1.safetensors", spec, meta, pay))
    assert _scaled_names(layout) == _BODY_LAYERS


def test_accepts_fp8_connector_without_scale(tmp_path):
    spec, meta, pay = _model("scaled")
    spec[f"{P}video_embeddings_connector.proj.weight"] = ("F8_E4M3", (4, 4))
    layout = inspect(_write(tmp_path / "c.safetensors", spec, meta, pay))
    assert f"{P}video_embeddings_connector.proj.weight" in layout.connector_keys
    assert _scaled_names(layout) == _BODY_LAYERS


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
    assert _scaled_names(layout) == _BODY_LAYERS


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
    assert _scaled_names(layout) == _BODY_LAYERS


def test_refuses_non_scalar_scale_on_connector(tmp_path):
    spec, meta, pay = _model("scaled")
    conn = f"{P}video_embeddings_connector.proj"
    spec[f"{conn}.weight"] = ("F8_E4M3", (4, 4))
    spec[f"{conn}.weight_scale"] = ("F32", (4,))
    with pytest.raises(QuantFormatError, match="方式 fp8_scaled で受理するのは"):
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
    assert _scaled_names(layout) == _BODY_LAYERS
    assert layout.connector_keys == (
        "audio_embeddings_connector.proj.bias",
        "video_embeddings_connector.proj.weight",
        "video_embeddings_connector.proj.weight_scale",
    )


def test_bare_names_police_every_tensor(tmp_path):
    """With prefix "" nothing is outside: an unaccepted dtype anywhere is refused."""
    spec, meta, pay = _model("scaled", prefix="")
    spec["audio_vae.encoder.stats"] = ("I64", (2,))
    with pytest.raises(QuantFormatError, match="I64"):
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
    with pytest.raises(QuantFormatError, match="接頭辞"):
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


def _fp8_outside_prefix(spec, meta, pay):
    spec["vae.decoder.conv.weight"] = ("F8_E4M3", (2, 2))


def _fp8_3d(spec, meta, pay):
    spec[f"{P}patchify_proj.weight"] = ("F8_E4M3", (2, 2, 4))


@pytest.mark.parametrize(
    "mutate, needle",
    [
        # an int8 marker on an fp8 weight / an I8 weight under an fp8 marker
        (_marker("int8_tensorwise"), "format='int8_tensorwise' と重みの dtype F8_E4M3 が合いません"),
        (_metadata_int8, "format='int8_tensorwise' と重みの dtype F8_E4M3 が合いません"),
        (_int8_weight, "format='float8_e4m3fn' と重みの dtype I8 が合いません"),
        (_marker("asym_w4a8_int8"), "format='asym_w4a8_int8' と重みの dtype F8_E4M3 が合いません"),
        (_marker("nvfp4"), "format='nvfp4' は未対応です"),
        (_marker("mxfp8"), "mxfp8"),
        (_marker("convrot_w4a4"), "convrot_w4a4"),
        (_f8_e8m0, "F8_E8M0"),
        (_per_row_scale, "F32[4] です（方式 fp8_scaled で受理するのは F32 の []／[1] のみ）"),
        (_per_block_scale, "F32[2, 2]"),
        (_bf16_scale, "BF16[]"),
        (_orphan_scale, "孤立した補助テンソル"),
        (_scaled_fp8_marker, "旧形式"),
        (_scale_weight, "旧形式"),
        (_no_config, "config がありません"),
        (_config_not_json, "JSON として読めません"),
        (_no_transformer_in_config, "transformer がありません"),
        (_drop_block_47, "47 個"),
        (_no_fp8, "1 本もありません"),
        (_no_connector, "connector）がありません"),
        (_fp8_outside_prefix, "の外にあります"),
        (_fp8_3d, "2 次元に限ります"),
    ],
    ids=lambda v: getattr(v, "__name__", None),
)
def test_refusals(tmp_path, mutate, needle):
    spec, meta, pay = _model("scaled")
    mutate(spec, meta, pay)
    path = _write(tmp_path / "x.safetensors", spec, meta, pay)
    with pytest.raises(QuantFormatError) as ei:
        inspect(path)
    message = str(ei.value)
    assert needle in message
    assert message.startswith("量子化 safetensors の検査に不合格: ")
    assert "\n" not in message  # one line


def test_refuses_wrong_prefix(tmp_path):
    spec, meta, pay = _model("scaled", prefix="diffusion_model.")
    path = _write(tmp_path / "x.safetensors", spec, meta, pay)
    with pytest.raises(QuantFormatError, match="接頭辞"):
        inspect(path)


def test_refuses_data_offsets_out_of_range(tmp_path):
    spec, meta, pay = _model("scaled")
    path = _write(tmp_path / "x.safetensors", spec, meta, pay)
    raw = path.read_bytes()
    path.write_bytes(raw[:-4])  # body shorter than the header claims
    with pytest.raises(QuantFormatError, match="data_offsets"):
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
    with pytest.raises(QuantFormatError):
        read_header(path)


def test_read_header_rejects_length_mismatch(tmp_path):
    blob = json.dumps({"a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 3]}}).encode()
    path = tmp_path / "m.safetensors"
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * 3)
    with pytest.raises(QuantFormatError, match="一致しません"):
        read_header(path)


def test_read_header_rejects_unknown_dtype(tmp_path):
    blob = json.dumps({"a": {"dtype": "F4", "shape": [2], "data_offsets": [0, 1]}}).encode()
    path = tmp_path / "u.safetensors"
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0")
    with pytest.raises(QuantFormatError, match="未知"):
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
    import sft_quant_format

    source = open(sft_quant_format.__file__, encoding="utf-8").read()
    for heavy in ("import torch", "import numpy", "import safetensors", "mmap"):
        assert heavy not in source



# --------------------------------------------------------------------------- #
# §3-168 C-1b: the scheme table and ComfyUI int8_tensorwise
# --------------------------------------------------------------------------- #

IN = 256  # ConvRot needs the input dimension to be a multiple of 256
_INT8_BLOCKS = range(2, 46)  # blocks 0, 1, 46, 47 stay bf16 (Kijai / silveroxides style)


def _name(b: int) -> str:
    return f"transformer_blocks.{b}.{LINEAR}"


def _conf_bytes(conf: dict) -> bytes:
    return json.dumps(conf).encode()


def _int8_model(
    *,
    scale=(4, 1),
    conf: dict | None = None,
    where: str = "tensor",
    marker_dtype: str = "U8",
    prefix: str = P,
    blocks=_INT8_BLOCKS,
    in_dim: int = IN,
):
    """A minimal transformer whose ``blocks`` Linears are ComfyUI int8_tensorwise.

    where: "tensor" (<layer>.comfy_quant), "meta" (__metadata__._quantization_metadata
    only) or "both" (the same marker in both places).
    """
    conf = {"format": "int8_tensorwise"} if conf is None else conf
    spec, meta, pay = _model("plain", prefix=prefix, fp8_blocks=())
    meta_layers = {}
    for b in blocks:
        layer = f"{prefix}transformer_blocks.{b}.{LINEAR}"
        spec[f"{layer}.weight"] = ("I8", (4, in_dim))
        spec[f"{layer}.weight_scale"] = ("F32", scale)
        if where in ("tensor", "both"):
            data = _conf_bytes(conf)
            spec[f"{layer}.comfy_quant"] = (marker_dtype, (len(data),))
            pay[f"{layer}.comfy_quant"] = data
        if where in ("meta", "both"):
            meta_layers[layer] = conf
    if meta_layers:
        meta["_quantization_metadata"] = json.dumps({"format_version": "1.0", "layers": meta_layers})
    return spec, meta, pay


def _inspect(tmp_path, spec, meta, pay, name="i.safetensors", **kw):
    return inspect(_write(tmp_path / name, spec, meta, pay), **kw)


def _all(scheme: str, blocks=_INT8_BLOCKS) -> dict[str, str]:
    return {_name(b): scheme for b in blocks}


# --- the table and the helpers derived from it ------------------------------ #


def test_scheme_table_rows():
    assert SCHEMES == ("fp8", "fp8_scaled", "int8", "int8_convrot", "w4a8")
    assert tuple(SCHEME_TABLE) == SCHEMES
    assert SKIPPED_SUFFIXES == (".input_scale", ".comfy_quant")


def test_aux_names_is_the_union_of_the_table():
    import sft_quant_format

    assert sft_quant_format.AUX_NAMES == frozenset(
        {"weight_scale", "weight_s_rel", "weight_s_channel", "weight_codebook"}
    )
    assert "AUX_NAMES" in sft_quant_format.__all__


def test_aux_specs():
    assert aux_specs("fp8") == {}
    assert aux_specs("fp8_scaled") == {"weight_scale": ("F32", "()")}
    assert aux_specs("int8") == {"weight_scale": ("F32", "(o,1)")}
    assert aux_specs("int8_convrot") == {"weight_scale": ("F32", "(o,1)")}
    assert aux_specs("w4a8") == {
        "weight_s_rel": ("F8_E4M3", "(o,i/16)"),
        "weight_s_channel": ("F32", "(o,)"),
        "weight_codebook": ("F32", "(16,)"),
    }


@pytest.mark.parametrize(
    "rule, expected",
    [("()", ()), ("(1,)", (1,)), ("(o,1)", (32, 1)), ("(o,)", (32,)), ("(16,)", (16,)), ("(o,i/16)", (32, 16))],
)
def test_aux_shape(rule, expected):
    assert aux_shape(rule, 32, 256) == expected


def test_aux_shape_refuses_unknown_rule():
    with pytest.raises(ValueError):
        aux_shape("(i,)", 32, 256)


@pytest.mark.parametrize("scheme", ["fp8", "fp8_scaled", "int8", "int8_convrot"])
def test_weight_shape(scheme):
    assert weight_shape(scheme, 4096, 16384) == (4096, 16384)


def test_w4a8_weight_shape_and_aux_shapes_agree():
    """``i`` is the LOGICAL input width: the stored I8 has i/2 columns, s_rel i/16."""
    assert weight_shape("w4a8", 4096, 16384) == (4096, 8192)
    assert aux_shape("(o,i/16)", 4096, 16384) == (4096, 1024)
    import sft_quant_format

    assert sft_quant_format.W4A8_GROUP_SIZE == 16 and "W4A8_GROUP_SIZE" in sft_quant_format.__all__
    row = SCHEME_TABLE["w4a8"]
    assert (row.weight, row.packing, row.in_multiple, row.quant_bias) == ("I8", 2, 256, False)
    assert row.aux["weight_s_rel"].raw_dtypes == frozenset({"F8_E4M3", "U8"})


def test_placement_is_derived_from_the_table():
    from sft_quant_format import _PLACEMENT

    assert _PLACEMENT == {
        "I8": frozenset({"weight", "comfy_quant"}),
        "U8": frozenset({"comfy_quant", "weight_s_rel"}),
        "F8_E4M3": frozenset({"weight", "bias", "weight_s_rel"}),
        "F8_E5M2": frozenset({"weight", "bias"}),
    }


# --- acceptance --------------------------------------------------------------- #


@pytest.mark.parametrize("scale", [(), (1,)], ids=["scalar", "shape_1"])
def test_accepts_int8_scalar_scale(tmp_path, scale):
    """silveroxides int8mixedtensorwise: one F32 scale per layer."""
    layout = _inspect(tmp_path, *_int8_model(scale=scale))
    assert layout.layers == _all("int8")


def test_accepts_int8_row_scale(tmp_path):
    """A per-row [o,1] scale (Kijai style) without the convrot flag."""
    assert _inspect(tmp_path, *_int8_model(scale=(4, 1))).layers == _all("int8")


def test_accepts_int8_convrot(tmp_path):
    spec, meta, pay = _int8_model(conf={"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256})
    assert _inspect(tmp_path, spec, meta, pay).layers == _all("int8_convrot")


def test_accepts_int8_convrot_in_nested_params(tmp_path):
    """The ``params`` nesting ComfyUI also writes is flattened before reading:
    missing it would silently drop the rotation (adversarial review, critical 1)."""
    conf = {"format": "int8_tensorwise", "params": {"convrot": True, "convrot_groupsize": 256}}
    assert _inspect(tmp_path, *_int8_model(conf=conf)).layers == _all("int8_convrot")


def test_top_level_marker_keys_win_over_params(tmp_path):
    """``{**params, **conf}``: a top-level key overrides the nested one."""
    conf = {"format": "int8_tensorwise", "convrot": False, "params": {"convrot": True}}
    assert _inspect(tmp_path, *_int8_model(conf=conf)).layers == _all("int8")


@pytest.mark.parametrize("where", ["tensor", "meta", "both"])
def test_accepts_int8_marker_in_any_place(tmp_path, where):
    conf = {"format": "int8_tensorwise", "convrot": True}
    assert _inspect(tmp_path, *_int8_model(conf=conf, where=where)).layers == _all("int8_convrot")


def test_metadata_marker_wins_per_layer(tmp_path):
    """Per layer, a ``_quantization_metadata`` entry replaces that layer's
    comfy_quant (ComfyUI convert_old_quants); other layers keep their tensor."""
    spec, meta, pay = _int8_model(conf={"format": "int8_tensorwise"}, where="tensor")
    meta["_quantization_metadata"] = json.dumps(
        {"layers": {_layer(5): {"format": "int8_tensorwise", "convrot": True}}}
    )
    layout = _inspect(tmp_path, spec, meta, pay)
    assert layout.layers[_name(5)] == "int8_convrot"
    assert layout.layers[_name(6)] == "int8"


def test_metadata_marker_wins_even_over_a_broken_tensor_marker(tmp_path):
    """The overridden comfy_quant is not read at all (ComfyUI overwrites it)."""
    spec, meta, pay = _int8_model(where="both")
    pay[_layer(5) + ".comfy_quant"] = b"{" * len(pay[_layer(5) + ".comfy_quant"])
    assert _inspect(tmp_path, spec, meta, pay).layers == _all("int8")


def test_accepts_i8_comfy_quant(tmp_path):
    assert _inspect(tmp_path, *_int8_model(marker_dtype="I8")).layers == _all("int8")


def test_marker_unknown_keys_and_full_precision_matrix_mult_are_ignored(tmp_path):
    conf = {"format": "int8_tensorwise", "full_precision_matrix_mult": True, "some_future_key": [1, 2]}
    assert _inspect(tmp_path, *_int8_model(conf=conf)).layers == _all("int8")


def test_int8_input_scale_is_ignored(tmp_path):
    spec, meta, pay = _int8_model()
    spec[_layer(5) + ".input_scale"] = ("F32", ())
    assert _inspect(tmp_path, spec, meta, pay).layers == _all("int8")


def test_accepts_int8_bare_names_with_bare_metadata_keys(tmp_path):
    spec, meta, pay = _int8_model(prefix="", where="meta")
    del spec["vae.decoder.conv.weight"]
    spec["video_embeddings_connector.proj.weight"] = ("I8", (4, IN))
    spec["video_embeddings_connector.proj.weight_scale"] = ("F32", (4, 1))
    qm = json.loads(meta["_quantization_metadata"])
    qm["layers"]["video_embeddings_connector.proj"] = {"format": "int8_tensorwise"}
    meta["_quantization_metadata"] = json.dumps(qm)
    layout = _inspect(tmp_path, spec, meta, pay)
    assert layout.prefix == ""
    assert layout.layers == _all("int8")  # the connector stays out of layers
    assert "video_embeddings_connector.proj.weight_scale" in layout.connector_keys


def test_accepts_gate_row_scale(tmp_path):
    """The attention gate Linear has 32 outputs: weight [32, i], scale [32, 1]."""
    spec, meta, pay = _int8_model()
    gate = f"{P}transformer_blocks.5.attn1.to_gate_logits"
    data = _conf_bytes({"format": "int8_tensorwise"})
    spec[f"{gate}.weight"] = ("I8", (32, IN))
    spec[f"{gate}.weight_scale"] = ("F32", (32, 1))
    spec[f"{gate}.comfy_quant"] = _quant_u8(data)
    pay[f"{gate}.comfy_quant"] = data
    assert _inspect(tmp_path, spec, meta, pay).layers["transformer_blocks.5.attn1.to_gate_logits"] == "int8"


def test_accepts_f16_unquantized_layers(tmp_path):
    """F16 non-quantized tensors are accepted (patientxtr) — weights and biases."""
    spec, meta, pay = _int8_model()
    spec[f"{P}patchify_proj.weight"] = ("F16", (4, 4))
    spec[f"{_layer(0)}.weight"] = ("F16", (4, 4))
    spec[f"{_layer(0)}.bias"] = ("F16", (4,))
    assert _inspect(tmp_path, spec, meta, pay).layers == _all("int8")


def test_float_layer_with_a_marker_stays_unquantized(tmp_path):
    spec, meta, pay = _int8_model()
    data = _conf_bytes({"format": "int8_tensorwise"})
    spec[_layer(0) + ".comfy_quant"] = _quant_u8(data)
    pay[_layer(0) + ".comfy_quant"] = data
    assert _name(0) not in _inspect(tmp_path, spec, meta, pay).layers


def test_edge_blocks_stay_bf16(tmp_path):
    layout = _inspect(tmp_path, *_int8_model())
    for b in (0, 1, 46, 47):
        assert _name(b) not in layout.layers


def test_accepts_fp8_and_int8_layers_mixed(tmp_path):
    spec, meta, pay = _int8_model(blocks=range(2, 24))
    for b in range(24, 46):
        spec[f"{_layer(b)}.weight"] = ("F8_E4M3", (4, 4))
        if b % 2:
            spec[f"{_layer(b)}.weight_scale"] = ("F32", ())
    layout = _inspect(tmp_path, spec, meta, pay)
    assert Counter(layout.layers.values()) == {"int8": 22, "fp8_scaled": 11, "fp8": 11}


def test_layer_schemes_is_prefixed_and_includes_connectors(tmp_path):
    spec, meta, pay = _int8_model()
    conn = f"{P}video_embeddings_connector.proj"
    spec[f"{conn}.weight"] = ("F8_E4M3", (4, 4))
    path = _write(tmp_path / "ls.safetensors", spec, meta, pay)
    schemes = layer_schemes(path, read_header(path), P)
    assert schemes[conn] == "fp8"
    assert schemes[_layer(5)] == "int8"
    assert len(schemes) == len(_INT8_BLOCKS) + 1


# --- refusals ----------------------------------------------------------------- #


def _set_conf(conf):
    def mutate(spec, meta, pay):
        data = _conf_bytes(conf)
        spec[_layer(5) + ".comfy_quant"] = _quant_u8(data)
        pay[_layer(5) + ".comfy_quant"] = data

    mutate.__name__ = "conf_" + "_".join(f"{k}={v}" for k, v in conf.items())
    return mutate


def _i8_no_marker(spec, meta, pay):
    del spec[_layer(5) + ".comfy_quant"]
    pay.pop(_layer(5) + ".comfy_quant")


def _i8_no_scale(spec, meta, pay):
    del spec[_layer(5) + ".weight_scale"]


def _extra_aux(name):
    def mutate(spec, meta, pay):
        spec[f"{_layer(5)}.{name}"] = ("F32", (4,))

    mutate.__name__ = f"extra_{name}"
    return mutate


def _scale_shape(shape, dtype="F32"):
    def mutate(spec, meta, pay):
        spec[_layer(5) + ".weight_scale"] = (dtype, shape)

    mutate.__name__ = f"scale_{dtype}_{'x'.join(map(str, shape)) or 'scalar'}"
    return mutate


def _convrot_on_128_inputs(spec, meta, pay):
    spec[_layer(5) + ".weight"] = ("I8", (4, 128))
    _set_conf({"format": "int8_tensorwise", "convrot": True})(spec, meta, pay)


def _i8_bias(spec, meta, pay):
    spec[_layer(5) + ".bias"] = ("I8", (4,))


def _quanto_data(spec, meta, pay):
    spec[_layer(5) + ".weight._data"] = ("I8", (4, IN))


def _i8_outside_prefix(spec, meta, pay):
    spec["vae.decoder.conv.weight"] = ("I8", (2, 2))


def _i8_1d_weight(spec, meta, pay):
    spec[_layer(5) + ".weight"] = ("I8", (4 * IN,))


def _no_quantized_weight(spec, meta, pay):
    for key in [k for k in spec if k.endswith((".weight_scale", ".comfy_quant"))]:
        del spec[key]
        pay.pop(key, None)
    for key, (dtype, shape) in list(spec.items()):
        if dtype == "I8":
            spec[key] = ("BF16", shape)


def _metadata_entry_not_object(spec, meta, pay):
    meta["_quantization_metadata"] = json.dumps({"layers": {_layer(5): "int8_tensorwise"}})


def _legacy_scaled_fp8(spec, meta, pay):
    spec[f"{P}scaled_fp8"] = ("F8_E4M3", (0,))


def _e8m0_scale(spec, meta, pay):
    spec[_layer(5) + ".weight_scale"] = ("F8_E8M0", (4, 1))


_QUANTO_KEY = f"'{P}transformer_blocks.5.{LINEAR}.weight._data'"


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (_set_conf({"format": "nvfp4"}), "format='nvfp4' は未対応です"),
        (_set_conf({"format": "mxfp8"}), "format='mxfp8' は未対応です"),
        (_set_conf({"format": "convrot_w4a4"}), "format='convrot_w4a4' は未対応です"),
        (_set_conf({"format": "asym_w4a8_int8", "group_size": 16}),
         "（方式 w4a8）に補助テンソル ['weight_codebook', 'weight_s_channel', 'weight_s_rel'] がありません"),
        (_set_conf({"convrot": True}), "format がありません"),  # old INT8-Fast
        (_i8_no_marker, "量子化の印がありません"),
        (_set_conf({"format": "float8_e4m3fn"}), "format='float8_e4m3fn' と重みの dtype I8 が合いません"),
        (_i8_no_scale, "補助テンソル ['weight_scale'] がありません"),
        (_extra_aux("weight_correction"), "未対応の補助テンソル ['weight_correction']"),
        (_extra_aux("pre_quant_scale"), "未対応の補助テンソル ['pre_quant_scale']"),
        (_extra_aux("weight_scale_2"), "未対応の補助テンソル ['weight_scale_2']"),
        (_scale_shape((4, 2)), "F32[4, 2] です（方式 int8 で受理するのは F32 の []／[1]／[4, 1] のみ）"),
        (_scale_shape((4,)), "F32[4] です"),
        (_scale_shape((4, 1), "BF16"), "BF16[4, 1] です"),
        (_convrot_on_128_inputs, "入力次元 128 が 256 の倍数ではありません"),
        (_set_conf({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 128}),
         "convrot_groupsize=128 は未対応です"),
        (_set_conf({"format": "int8_tensorwise", "convrot": "yes"}), "convrot が真偽値ではありません"),
        (_set_conf({"format": "int8_tensorwise", "params": [1]}), "params が JSON オブジェクトではありません"),
        (_metadata_entry_not_object, "量子化の印が JSON オブジェクトではありません"),
        (_i8_bias, "I8 を置けるのは .comfy_quant・.weight だけです"),
        (_quanto_data, f"I8 を置けるのは .comfy_quant・.weight だけです（{_QUANTO_KEY}）"),
        (_i8_outside_prefix, f"量子化テンソル 'vae.decoder.conv.weight' が '{P}' の外にあります"),
        (_i8_1d_weight, "2 次元に限ります"),
        (_no_quantized_weight, "量子化された重みが 1 本もありません"),
        (_legacy_scaled_fp8, "旧形式"),
        (_e8m0_scale, "F8_E8M0 は未対応です"),
    ],
    ids=lambda v: getattr(v, "__name__", None) if callable(v) else "",
)
def test_int8_refusals(tmp_path, mutate, needle):
    spec, meta, pay = _int8_model()
    mutate(spec, meta, pay)
    with pytest.raises(QuantFormatError) as ei:
        _inspect(tmp_path, spec, meta, pay)
    message = str(ei.value)
    assert needle in message
    assert message.startswith("量子化 safetensors の検査に不合格: ")
    assert len(message.splitlines()) == 1


def test_int8_orphan_scale_on_float_layer_is_refused(tmp_path):
    spec, meta, pay = _int8_model()
    spec[_layer(0) + ".weight_scale"] = ("F32", (4, 1))
    with pytest.raises(QuantFormatError, match="孤立した補助テンソル"):
        _inspect(tmp_path, spec, meta, pay)


# --- reads the header and the markers only ------------------------------------ #


def _big_int8_model(tmp_path, where):
    spec, meta, pay = _int8_model(where=where, conf={"format": "int8_tensorwise", "convrot": True})
    for b in _INT8_BLOCKS:
        spec[f"{_layer(b)}.weight"] = ("I8", (64, IN))
        spec[f"{_layer(b)}.weight_scale"] = ("F32", (64, 1))
    return _write(tmp_path / f"big_{where}.safetensors", spec, meta, pay)


@pytest.mark.parametrize("where", ["tensor", "meta", "both"])
def test_inspect_int8_reads_only_header_and_needed_markers(tmp_path, monkeypatch, where):
    """Metadata markers make the comfy_quant tensors unnecessary: nothing past
    the header is read then. Otherwise only the markers' few bytes are."""
    import io

    path = _big_int8_model(tmp_path, where)
    header = read_header(path)
    quant_bytes = sum(
        info.data_offsets[1] - info.data_offsets[0]
        for key, info in header.tensors.items()
        if key.endswith(".comfy_quant")
    )
    counter = _CountingOpen(path)
    monkeypatch.setattr(io, "open", counter)
    assert set(inspect(path).layers.values()) == {"int8_convrot"}
    assert counter.total <= 8 + header.header_len + (quant_bytes if where == "tensor" else 0)
    assert header.file_size > 8 + header.header_len + 44 * 64 * IN  # the body is big


# --------------------------------------------------------------------------- #
# §3-168 C-3: ComfyUI asym_w4a8_int8 (4-bit codes + codebook, always ConvRot)
# --------------------------------------------------------------------------- #

W4A8_CONF = {"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 256}


def _put_w4a8(spec, layer: str, *, o: int = 4, i: int = IN, s_rel: str = "F8_E4M3"):
    """One w4a8 Linear with ``o`` outputs and LOGICAL input width ``i``."""
    spec[f"{layer}.weight"] = ("I8", (o, i // 2))
    spec[f"{layer}.weight_s_rel"] = (s_rel, (o, i // 16))
    spec[f"{layer}.weight_s_channel"] = ("F32", (o,))
    spec[f"{layer}.weight_codebook"] = ("F32", (16,))


def _put_marker(spec, pay, layer: str, conf: dict):
    data = _conf_bytes(conf)
    spec[f"{layer}.comfy_quant"] = _quant_u8(data)
    pay[f"{layer}.comfy_quant"] = data


def _w4a8_model(*, conf: dict | None = None, where: str = "tensor", s_rel: str = "F8_E4M3",
                prefix: str = P, blocks=_INT8_BLOCKS):
    """A minimal transformer whose ``blocks`` Linears are w4a8 (``where`` as in _int8_model)."""
    conf = W4A8_CONF if conf is None else conf
    spec, meta, pay = _model("plain", prefix=prefix, fp8_blocks=())
    meta_layers = {}
    for b in blocks:
        layer = f"{prefix}transformer_blocks.{b}.{LINEAR}"
        _put_w4a8(spec, layer, s_rel=s_rel)
        if where in ("tensor", "both"):
            _put_marker(spec, pay, layer, conf)
        if where in ("meta", "both"):
            meta_layers[layer] = conf
    if meta_layers:
        meta["_quantization_metadata"] = json.dumps({"format_version": "1.0", "layers": meta_layers})
    return spec, meta, pay


# --- acceptance --------------------------------------------------------------- #


@pytest.mark.parametrize("s_rel", ["F8_E4M3", "U8"])
@pytest.mark.parametrize("where", ["tensor", "meta", "both"])
def test_accepts_w4a8(tmp_path, where, s_rel):
    layout = _inspect(tmp_path, *_w4a8_model(where=where, s_rel=s_rel))
    assert layout.layers == _all("w4a8")


@pytest.mark.parametrize(
    "conf",
    [
        {"format": "asym_w4a8_int8"},  # both group sizes default (16 / 256, ComfyUI ops.py)
        {"format": "asym_w4a8_int8", "convrot": True, "convrot_groupsize": 256, "group_size": 16},
        {"format": "asym_w4a8_int8", "convrot": False},  # not read: w4a8 is always rotated
        {"format": "asym_w4a8_int8", "params": {"group_size": 16, "convrot_groupsize": 256}},
    ],
    ids=["defaults", "full", "convrot_false", "nested_params"],
)
def test_accepts_w4a8_marker_variants(tmp_path, conf):
    assert _inspect(tmp_path, *_w4a8_model(conf=conf)).layers == _all("w4a8")


def test_accepts_redgraft_style_mix(tmp_path):
    """REDGraft 3250230: int8 ConvRot + w4a8 + bf16 edge blocks, a w4a8
    connector, every marker a comfy_quant tensor."""
    spec, meta, pay = _int8_model(blocks=range(2, 24), conf={"format": "int8_tensorwise", "convrot": True})
    conn = f"{P}video_embeddings_connector.proj"
    for layer in [_layer(b) for b in range(24, 46)] + [conn]:
        _put_w4a8(spec, layer)
        _put_marker(spec, pay, layer, W4A8_CONF)
    path = _write(tmp_path / "redgraft.safetensors", spec, meta, pay)
    layout = inspect(path)
    assert Counter(layout.layers.values()) == {"int8_convrot": 22, "w4a8": 22}
    assert f"{conn}.weight_codebook" in layout.connector_keys
    assert layer_schemes(path, read_header(path), P)[conn] == "w4a8"


def test_accepts_joaozaokk_style(tmp_path):
    """JoaoZaokk 2.3 distilled-1.1 w4a8: markers in the metadata only, s_rel
    stored as U8, no ``convrot`` key."""
    layout = _inspect(tmp_path, *_w4a8_model(conf=W4A8_CONF, where="meta", s_rel="U8"))
    assert layout.layers == _all("w4a8")


def test_accepts_tsolful_style_bare_names(tmp_path):
    """tsolful 2.5: bare tensor names and bare metadata layer names."""
    conf = {"convrot": True, "convrot_groupsize": 256, "format": "asym_w4a8_int8", "group_size": 16}
    spec, meta, pay = _w4a8_model(conf=conf, where="meta", prefix="")
    del spec["vae.decoder.conv.weight"]
    layout = _inspect(tmp_path, spec, meta, pay)
    assert layout.prefix == ""
    assert layout.layers == _all("w4a8")


def test_accepts_w4a8_gate_and_wide_layers(tmp_path):
    """o and i are read per layer: a 32-output gate and a 4x-wide ff input."""
    spec, meta, pay = _w4a8_model()
    for layer, o, i in [(f"{P}transformer_blocks.5.attn1.to_gate_logits", 32, IN),
                        (f"{P}transformer_blocks.5.ff.net.2", 4, 4 * IN)]:
        _put_w4a8(spec, layer, o=o, i=i)
        _put_marker(spec, pay, layer, W4A8_CONF)
    layers = _inspect(tmp_path, spec, meta, pay).layers
    assert layers["transformer_blocks.5.attn1.to_gate_logits"] == "w4a8"
    assert layers["transformer_blocks.5.ff.net.2"] == "w4a8"


# --- refusals ----------------------------------------------------------------- #


def _w4a8_conf(conf):
    def mutate(spec, meta, pay):
        _put_marker(spec, pay, _layer(5), conf)

    mutate.__name__ = "conf_" + "_".join(f"{k}={v}" for k, v in conf.items())
    return mutate


def _w4a8_drop(name):
    def mutate(spec, meta, pay):
        del spec[f"{_layer(5)}.{name}"]

    mutate.__name__ = f"no_{name}"
    return mutate


def _w4a8_aux(name, dtype, shape):
    def mutate(spec, meta, pay):
        spec[f"{_layer(5)}.{name}"] = (dtype, shape)

    mutate.__name__ = f"{name}_{dtype}_{'x'.join(map(str, shape)) or 'scalar'}"
    return mutate


def _w4a8_on_128_inputs(spec, meta, pay):
    _put_w4a8(spec, _layer(5), i=128)


def _w4a8_marker_on_f8_weight(spec, meta, pay):
    spec[_layer(5) + ".weight"] = ("F8_E4M3", (4, IN // 2))


def _w4a8_on_bf16_weight(spec, meta, pay):
    spec[_layer(5) + ".weight"] = ("BF16", (4, IN // 2))


def _f8_s_rel_on_f8_layer(spec, meta, pay):
    """An fp8 layer carrying an s_rel: placement allows it, the aux set does not."""
    spec[_layer(0) + ".weight"] = ("F8_E4M3", (4, 4))
    spec[_layer(0) + ".weight_s_rel"] = ("F8_E4M3", (4, 1))


_S_REL = f"'{P}transformer_blocks.5.{LINEAR}.weight_s_rel'"


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (_w4a8_drop("weight_codebook"), "（方式 w4a8）に補助テンソル ['weight_codebook'] がありません"),
        (_w4a8_drop("weight_s_rel"), "（方式 w4a8）に補助テンソル ['weight_s_rel'] がありません"),
        (_w4a8_drop("weight_s_channel"), "（方式 w4a8）に補助テンソル ['weight_s_channel'] がありません"),
        (_w4a8_aux("weight_correction", "F32", (4,)), "（方式 w4a8）に未対応の補助テンソル ['weight_correction']"),
        (_w4a8_aux("weight_scale", "F32", ()), "（方式 w4a8）に未対応の補助テンソル ['weight_scale']"),
        (_w4a8_conf({**W4A8_CONF, "group_size": 32}), "group_size=32 は未対応です（受理: 16 のみ"),
        (_w4a8_conf({"format": "asym_w4a8_int8", "params": {"group_size": 64}}), "group_size=64 は未対応です"),
        (_w4a8_conf({**W4A8_CONF, "convrot_groupsize": 128}), "convrot_groupsize=128 は未対応です"),
        (_w4a8_aux("weight_s_rel", "F8_E4M3", (4, 8)),
         f"補助テンソル {_S_REL} が F8_E4M3[4, 8] です（方式 w4a8 で受理するのは F8_E4M3・U8 の [4, 16] のみ）"),
        (_w4a8_aux("weight_s_rel", "F32", (4, 16)), "F32[4, 16] です"),
        (_w4a8_aux("weight_s_rel", "BF16", (4, 16)), "BF16[4, 16] です"),
        (_w4a8_aux("weight_s_channel", "F32", (4, 1)), "F32[4, 1] です（方式 w4a8 で受理するのは F32 の [4] のみ）"),
        (_w4a8_aux("weight_s_channel", "F32", ()), "F32[] です"),
        (_w4a8_aux("weight_codebook", "F32", (8,)), "F32[8] です（方式 w4a8 で受理するのは F32 の [16] のみ）"),
        (_w4a8_aux("weight_codebook", "F8_E4M3", (16,)),
         "F8_E4M3 を置けるのは .bias・.weight・.weight_s_rel だけです"),
        (_w4a8_aux("weight_s_channel", "U8", (4,)), "U8 を置けるのは .comfy_quant・.weight_s_rel だけです"),
        (_w4a8_on_128_inputs, "（方式 w4a8）の入力次元 128 が 256 の倍数ではありません"),
        (_w4a8_marker_on_f8_weight, "format='asym_w4a8_int8' と重みの dtype F8_E4M3 が合いません"),
        (_w4a8_on_bf16_weight, "孤立した補助テンソル"),
        (_f8_s_rel_on_f8_layer, "（方式 fp8）に未対応の補助テンソル ['weight_s_rel']"),
    ],
    ids=lambda v: getattr(v, "__name__", None) if callable(v) else "",
)
def test_w4a8_refusals(tmp_path, mutate, needle):
    spec, meta, pay = _w4a8_model()
    mutate(spec, meta, pay)
    with pytest.raises(QuantFormatError) as ei:
        _inspect(tmp_path, spec, meta, pay)
    message = str(ei.value)
    assert needle in message
    assert message.startswith("量子化 safetensors の検査に不合格: ")
    assert len(message.splitlines()) == 1


def test_int8_marker_on_w4a8_layer_is_refused(tmp_path):
    """The reverse of the int8 refusal table's w4a8 row: the aux sets differ."""
    spec, meta, pay = _w4a8_model(conf={"format": "int8_tensorwise", "convrot": True})
    with pytest.raises(QuantFormatError, match=r"（方式 int8_convrot）に補助テンソル \['weight_scale'\] がありません"):
        _inspect(tmp_path, spec, meta, pay)
