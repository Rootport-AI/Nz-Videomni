"""h3_format: the torch-free MiniMax H3 header inspection (台帳 §1-93 第 1 弾).

Runs in the app venv (no torch). Every DiT case builds a REAL small
safetensors file (tiny widths, zero data) so the path through
``sft_quant_format.read_header`` and the ``comfy_quant`` seek-and-read is the
shipped one. The text-encoder cases need the real Qwen3-VL widths (25 GB of
zeros), so there the header is built in memory and only the ``comfy_quant``
bytes live in a small file.

    .venv/Scripts/python.exe -m pytest tests/test_h3_format.py
"""

from __future__ import annotations

import json
import struct

import pytest

import h3_format
from h3_format import H3FormatError, QuantSpec
from sft_quant_format import DTYPE_ITEMSIZE, Header, TensorInfo

# Tiny but self-consistent H3 widths.
HID, HEADS, HD, FFN, TEXT, LAT, AUD = 64, 2, 16, 48, 40, 24, 32
INNER = HEADS * HD
INT8_MARK = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}


def _dit_tensors(*, layers=3, refiner=2, grid=1025, rank=8, quant=True, pruned=True,
                 hid=HID, heads=HEADS, hd=HD, ffn=FFN, fmt_of=None):
    """name -> (dtype, shape) of a minimal H3 DiT (no prefix).

    ``fmt_of(layer)`` -> None (bf16) or a format of :func:`_quant_entries`, for
    every Linear of the blocks and the refiner; by default int8 in the blocks
    (``quant``) and bf16 in the refiner. A ``comfy_quant`` entry may carry its
    own marker as a third element (else ``_write``'s ``marker``).
    """
    HID, HD, FFN = hid, hd, ffn  # noqa: N806 -- the module widths, overridable
    INNER = heads * hd  # noqa: N806
    t: dict[str, tuple] = {
        "video_patch_proj.weight": ("F32", (HID, LAT * 4)),
        "video_patch_proj.bias": ("F32", (HID,)),
        "audio_patch_proj.weight": ("F32", (HID, AUD)),
        "audio_patch_proj.bias": ("F32", (HID,)),
        "condition_proj.weight": ("BF16", (HID, TEXT)),
        "condition_proj.bias": ("BF16", (HID,)),
        "token_refiner.final_norm.weight": ("BF16", (HID,)),
        "final_layer.norm.weight": ("BF16", (HID,)),
        "final_layer.adaln_proj.linear.weight": ("F16", (2 * HID, rank)),
        "final_layer.adaln_proj.linear.bias": ("F16", (2 * HID,)),
        "final_layer.video_out.weight": ("F32", (LAT * 4, HID)),
        "final_layer.video_out.bias": ("F32", (LAT * 4,)),
        "final_layer.audio_out.weight": ("F32", (AUD, HID)),
        "final_layer.audio_out.bias": ("F32", (AUD,)),
        "rope.inv_freq": ("F32", (16,)),
    }
    if pruned:
        t["adaln_t_table"] = ("F32", (grid, rank))
    else:
        t["time_embedder.proj_in.weight"] = ("F32", (HID, 256))
    lin = {
        "attn.qkv_proj": (3 * INNER, HID),
        "attn.out_proj": (HID, INNER),
        "mlp.fc1": (2 * FFN, HID),
        "mlp.fc2": (HID, FFN),
    }

    def block(head: str, adaln: bool, q: bool):
        for name, shape in lin.items():
            fmt = fmt_of(f"{head}{name}") if fmt_of is not None else ("int8" if q else None)
            if fmt is None:
                t[f"{head}{name}.weight"] = ("BF16", shape)
            else:
                t.update(_quant_entries(f"{head}{name}", shape, fmt))
        t[f"{head}attn.q_norm.weight"] = ("BF16", (HD,))
        t[f"{head}attn.k_norm.weight"] = ("BF16", (HD,))
        t[f"{head}norm1.weight"] = ("BF16", (HID,))
        t[f"{head}norm2.weight"] = ("BF16", (HID,))
        if adaln:
            t[f"{head}adaln_proj.linear.weight"] = ("F16", (18 * HID, rank))
            t[f"{head}adaln_proj.linear.bias"] = ("F16", (18 * HID,))

    for i in range(layers):
        block(f"blocks.{i}.", True, quant)
    for i in range(refiner):
        block(f"token_refiner.blocks.{i}.", False, False)
    return t


FP8_MARK = {"format": "float8_e4m3fn"}


def _w4a8_mark(g=16):
    return {"format": "asym_w4a8_int8", "group_size": g, "convrot": True, "convrot_groupsize": 256}


#: Widths at which every Linear's K is a multiple of 256 (the w4a8 condition).
WIDE = {"hid": 256, "heads": 2, "hd": 128, "ffn": 256}


def _quant_entries(layer, shape, fmt):
    """The tensors of one quantized Linear ``[N, K]``. ``fmt``: "int8" / "fp8" / "fp8_full" /
    "fp8_noinput" / "w4a8" (G 16, codebook) / a dict ``{"fmt": "w4a8", "g": 8, "codebook": False}``."""
    n, k = shape
    opts = fmt if isinstance(fmt, dict) else {"fmt": fmt}
    kind = opts["fmt"]
    if kind == "int8":
        return {f"{layer}.weight": ("I8", (n, k)), f"{layer}.weight_scale": ("F32", (n, 1)),
                f"{layer}.comfy_quant": ("U8", (0,))}  # _write's ``marker``
    if kind.startswith("fp8"):
        out = {f"{layer}.weight": ("F8_E4M3", (n, k)), f"{layer}.weight_scale": ("F32", ())}
        if kind != "fp8_noinput":
            out[f"{layer}.input_scale"] = ("F32", ())
        mark = {**FP8_MARK, "full_precision_matrix_mult": True} if kind == "fp8_full" else FP8_MARK
        out[f"{layer}.comfy_quant"] = ("U8", (0,), mark)
        return out
    assert kind == "w4a8", kind
    g = opts.get("g", 16)
    out = {f"{layer}.weight": ("I8", (n, k // 2)), f"{layer}.weight_s_rel": ("F8_E4M3", (n, k // g)),
           f"{layer}.weight_s_channel": ("F32", (n,)), f"{layer}.comfy_quant": ("U8", (0,), _w4a8_mark(g))}
    if opts.get("codebook", True):
        out[f"{layer}.weight_codebook"] = ("F32", (16,))
    return out


def _write(path, tensors, *, prefix="", marker=INT8_MARK, metadata=None):
    """Write a real safetensors file: zero data, ``comfy_quant`` = JSON bytes of the
    entry's own marker (a third element) or of ``marker``."""
    doc: dict = {}
    blobs: list[bytes] = []
    off = 0
    for name, entry in tensors.items():
        dtype, shape = entry[0], entry[1]
        if name.endswith(".comfy_quant"):
            own = entry[2] if len(entry) > 2 else marker
            mark = own if isinstance(own, bytes) else json.dumps(own).encode()
            data, shape = mark, (len(mark),)
        else:
            n = 1
            for d in shape:
                n *= d
            data = bytes(n * DTYPE_ITEMSIZE[dtype])
        doc[prefix + name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    if metadata:
        doc["__metadata__"] = metadata
    head = json.dumps(doc).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for b in blobs:
            f.write(b)
    return str(path)


# --------------------------------------------------------------------------- #
# detection and prefixes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("prefix", h3_format.H3_PREFIXES)
def test_each_prefix_is_detected_and_stripped(tmp_path, prefix):
    path = _write(tmp_path / "dit.safetensors", _dit_tensors(), prefix=prefix)
    header = h3_format.read_header(path)
    assert h3_format.is_h3_dit_header(header)
    assert h3_format.detect_h3_prefix(header.tensors) == prefix
    _, cfg = h3_format.inspect_h3_dit(path)
    assert cfg.prefix == prefix
    assert "blocks.0.attn.qkv_proj" in cfg.quant  # quant keys are prefix-stripped
    assert cfg.unknown == ()


def test_config_is_derived_from_shapes(tmp_path):
    path = _write(tmp_path / "dit.safetensors", _dit_tensors(layers=3, refiner=2))
    _, cfg = h3_format.inspect_h3_dit(path)
    assert (cfg.hidden_size, cfg.num_layers, cfg.token_refiner_layers) == (HID, 3, 2)
    assert (cfg.num_heads, cfg.head_dim, cfg.ffn_hidden) == (HEADS, HD, FFN)
    assert (cfg.text_dim, cfg.latents_dim, cfg.audio_latents_dim) == (TEXT, LAT, AUD)
    assert cfg.rope_inv_freq_len == 16
    assert cfg.pruned is True
    # 4 quantized Linears per block, none in the refiner (the Comfy-Org layout)
    assert len(cfg.quant) == 4 * 3
    assert set(cfg.quant.values()) == {QuantSpec("int8_tensorwise", True, 256)}


@pytest.mark.parametrize("grid,rank", [(1025, 8), (4097, 8), (1025, 16), (4097, 16)])
def test_pruned_table_grid_and_rank(tmp_path, grid, rank):
    path = _write(tmp_path / "dit.safetensors", _dit_tensors(grid=grid, rank=rank))
    _, cfg = h3_format.inspect_h3_dit(path)
    assert (cfg.adaln_grid, cfg.adaln_rank) == (grid, rank)


def test_ltx_header_is_not_h3(tmp_path):
    t = {
        "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": ("BF16", (8, 8)),
        "model.diffusion_model.patchify_proj.weight": ("BF16", (8, 8)),
    }
    path = _write(tmp_path / "ltx.safetensors", t, metadata={"config": "{}"})
    assert not h3_format.is_h3_dit_header(h3_format.read_header(path))
    with pytest.raises(H3FormatError, match="LTX"):
        h3_format.inspect_h3_dit(path)


def test_video_patch_alone_is_not_h3(tmp_path):
    t = _dit_tensors()
    del t["audio_patch_proj.weight"]
    path = _write(tmp_path / "dit.safetensors", t)
    assert not h3_format.is_h3_dit_header(h3_format.read_header(path))
    with pytest.raises(H3FormatError, match="audio_patch_proj"):
        h3_format.inspect_h3_dit(path)


def test_detect_prefix_without_video_patch_raises():
    with pytest.raises(H3FormatError):
        h3_format.detect_h3_prefix(["blocks.0.norm1.weight"])


# --------------------------------------------------------------------------- #
# comfy_quant
# --------------------------------------------------------------------------- #

def test_comfy_quant_unknown_keys_are_ignored():
    raw = json.dumps({
        "format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 64,
        "per_row": True, "orig_dtype": "bfloat16", "full_precision_matrix_mult": False,
    }).encode()
    assert h3_format.parse_comfy_quant(raw) == QuantSpec("int8_tensorwise", True, 64)


def test_comfy_quant_params_nesting_and_defaults():
    raw = json.dumps({"format": "int8_tensorwise", "params": {"convrot": True, "foo": 1}}).encode()
    assert h3_format.parse_comfy_quant(raw) == QuantSpec("int8_tensorwise", True, 256)
    plain = json.dumps({"format": "int8_tensorwise"}).encode()
    assert h3_format.parse_comfy_quant(plain) == QuantSpec("int8_tensorwise", False, 256)


@pytest.mark.parametrize("fmt", ["int4", "nvfp4", None])
@pytest.mark.parametrize("accepted", [h3_format.TE_FORMATS, h3_format.DIT_FORMATS])
def test_comfy_quant_other_formats_are_refused(fmt, accepted):
    with pytest.raises(H3FormatError, match="format"):
        h3_format.parse_comfy_quant(json.dumps({"format": fmt}).encode(), accepted=accepted)


@pytest.mark.parametrize("fmt", ["float8_e4m3fn", "asym_w4a8_int8"])
def test_comfy_quant_fp8_and_w4a8_only_where_accepted(fmt):
    raw = json.dumps({"format": fmt}).encode()
    with pytest.raises(H3FormatError, match="format"):
        h3_format.parse_comfy_quant(raw)  # the default is the text encoder's: int8 only
    spec = h3_format.parse_comfy_quant(raw, accepted=h3_format.DIT_FORMATS)
    assert spec.format == fmt


def test_comfy_quant_unknown_keys_ignored_in_a_file(tmp_path):
    marker = {**INT8_MARK, "convrot_groupsize": 128, "per_row": True, "orig_dtype": "bfloat16"}
    path = _write(tmp_path / "dit.safetensors", _dit_tensors(), marker=marker)
    _, cfg = h3_format.inspect_h3_dit(path)
    assert set(cfg.quant.values()) == {QuantSpec("int8_tensorwise", True, 128)}


def test_fp8_marker_on_int8_weights_is_refused(tmp_path):
    """The marker decides the format; the tensors must then be that format's."""
    path = _write(tmp_path / "dit.safetensors", _dit_tensors(), marker={"format": "float8_e4m3fn"})
    with pytest.raises(H3FormatError, match="F8_E4M3"):
        h3_format.inspect_h3_dit(path)


def test_unsupported_marker_in_a_file_is_refused(tmp_path):
    path = _write(tmp_path / "dit.safetensors", _dit_tensors(), marker={"format": "nvfp4"})
    with pytest.raises(H3FormatError, match="nvfp4"):
        h3_format.inspect_h3_dit(path)


def test_int8_without_scale_is_refused(tmp_path):
    t = _dit_tensors()
    del t["blocks.1.mlp.fc2.weight_scale"]
    path = _write(tmp_path / "dit.safetensors", t)
    with pytest.raises(H3FormatError, match="weight_scale"):
        h3_format.inspect_h3_dit(path)


def test_int8_without_marker_is_refused(tmp_path):
    t = _dit_tensors()
    del t["blocks.2.attn.qkv_proj.comfy_quant"]
    path = _write(tmp_path / "dit.safetensors", t)
    with pytest.raises(H3FormatError, match="comfy_quant"):
        h3_format.inspect_h3_dit(path)


def test_unquantized_dit_has_no_quant_entries(tmp_path):
    path = _write(tmp_path / "dit.safetensors", _dit_tensors(quant=False))
    _, cfg = h3_format.inspect_h3_dit(path)
    assert cfg.quant == {}


# --------------------------------------------------------------------------- #
# refusals and unknown names
# --------------------------------------------------------------------------- #

def test_non_pruned_dit_is_refused_with_a_reason(tmp_path):
    path = _write(tmp_path / "dit.safetensors", _dit_tensors(pruned=False))
    with pytest.raises(H3FormatError, match="枝刈り"):
        h3_format.inspect_h3_dit(path)


def test_missing_required_tensor_is_refused(tmp_path):
    t = _dit_tensors()
    del t["final_layer.audio_out.weight"]
    path = _write(tmp_path / "dit.safetensors", t)
    with pytest.raises(H3FormatError, match="final_layer.audio_out.weight"):
        h3_format.inspect_h3_dit(path)


def test_missing_block_member_is_refused(tmp_path):
    t = _dit_tensors()
    del t["blocks.1.norm2.weight"]
    path = _write(tmp_path / "dit.safetensors", t)
    with pytest.raises(H3FormatError, match="norm2"):
        h3_format.inspect_h3_dit(path)


def test_shape_contradiction_is_refused(tmp_path):
    t = _dit_tensors()
    t["blocks.2.mlp.fc2.weight"] = ("I8", (HID, FFN + 1))
    path = _write(tmp_path / "dit.safetensors", t)
    with pytest.raises(H3FormatError, match="矛盾"):
        h3_format.inspect_h3_dit(path)


def test_unknown_tensors_are_reported_not_refused(tmp_path):
    t = _dit_tensors()
    t["blocks.0.attn.to_gate_compress.weight"] = ("BF16", (4, 4))  # VSA extra
    t["some_future_head.weight"] = ("F32", (2,))
    path = _write(tmp_path / "dit.safetensors", t)
    _, cfg = h3_format.inspect_h3_dit(path)
    assert cfg.unknown == ("blocks.0.attn.to_gate_compress.weight", "some_future_head.weight")


def test_corrupt_header_is_an_h3_format_error(tmp_path):
    p = tmp_path / "bad.safetensors"
    p.write_bytes(b"\x00" * 4)
    with pytest.raises(H3FormatError):
        h3_format.inspect_h3_dit(str(p))


# --------------------------------------------------------------------------- #
# fp8 / w4a8 and the metadata markers (台帳 §1-283)
# --------------------------------------------------------------------------- #

def _all(fmt):
    """fmt_of: every block Linear in ``fmt``, the refiner bf16."""
    return lambda layer: fmt if layer.startswith("blocks.") else None


def _wide(tmp_path, fmt_of, *, name="dit.safetensors", metadata=None, drop=(), extra=None, **kw):
    t = _dit_tensors(layers=2, refiner=1, fmt_of=fmt_of, **WIDE, **kw)
    for n in drop:
        del t[n]
    t.update(extra or {})
    return _write(tmp_path / name, t, metadata=metadata)


def _layers_meta(layers: dict) -> dict:
    return {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": layers})}


@pytest.mark.parametrize("fmt,full,inp", [("fp8", False, True), ("fp8_noinput", False, False), ("fp8_full", True, True)])
def test_fp8_layers_are_accepted(tmp_path, fmt, full, inp):
    _, cfg = h3_format.inspect_h3_dit(_wide(tmp_path, _all(fmt)))
    assert len(cfg.quant) == 8 and cfg.unknown == ()
    assert set(cfg.quant.values()) == {
        QuantSpec("float8_e4m3fn", False, 256, group_size=None, full_precision=full, has_input_scale=inp)
    }


@pytest.mark.parametrize("g", [16, 8, 4])
@pytest.mark.parametrize("codebook", [True, False])
def test_w4a8_layers_are_accepted_with_packed_shapes(tmp_path, g, codebook):
    _, cfg = h3_format.inspect_h3_dit(_wide(tmp_path, _all({"fmt": "w4a8", "g": g, "codebook": codebook})))
    # the shape table read the [N, K/2] weights at [N, K]: the widths come out right
    assert (cfg.hidden_size, cfg.num_heads, cfg.head_dim, cfg.ffn_hidden) == (256, 2, 128, 256)
    assert cfg.unknown == ()  # weight_s_rel / weight_s_channel / weight_codebook are known
    assert set(cfg.quant.values()) == {
        QuantSpec("asym_w4a8_int8", True, 256, group_size=g, has_codebook=codebook)
    }


@pytest.mark.parametrize("extra", [{}, {"convrot": False}, {"convrot": "yes"}])
def test_w4a8_convrot_key_is_not_read(tmp_path, extra):
    """ComfyUI never reads ``convrot`` for asym_w4a8_int8: always rotated, whatever the key says."""
    t = _dit_tensors(layers=2, refiner=1, fmt_of=_all("w4a8"), **WIDE)
    for n in [n for n in t if n.endswith(".comfy_quant")]:
        t[n] = ("U8", (0,), {"format": "asym_w4a8_int8", "group_size": 16, **extra})
    _, cfg = h3_format.inspect_h3_dit(_write(tmp_path / "dit.safetensors", t))
    assert {q.convrot for q in cfg.quant.values()} == {True}


def test_mixed_formats_in_one_file(tmp_path):
    fmts = {"attn.qkv_proj": "int8", "attn.out_proj": "fp8", "mlp.fc1": {"fmt": "w4a8", "g": 8}, "mlp.fc2": "fp8_full"}

    def fmt_of(layer):
        if layer.startswith("token_refiner."):
            return "w4a8" if layer.endswith("fc1") else None
        return fmts[layer.split(".", 2)[2]]

    _, cfg = h3_format.inspect_h3_dit(_wide(tmp_path, fmt_of))
    assert cfg.unknown == ()
    assert cfg.quant["blocks.1.attn.qkv_proj"] == QuantSpec("int8_tensorwise", True, 256)
    assert cfg.quant["blocks.1.attn.out_proj"].has_input_scale is True
    assert cfg.quant["blocks.1.mlp.fc1"].group_size == 8
    assert cfg.quant["blocks.1.mlp.fc2"].full_precision is True
    assert cfg.quant["token_refiner.blocks.0.mlp.fc1"].group_size == 16
    assert len(cfg.quant) == 9


def test_metadata_markers_alone(tmp_path):
    """Kijai's layout: no comfy_quant tensor, every marker in the header metadata."""
    fmt_of = _all({"fmt": "w4a8", "g": 16})
    t = _dit_tensors(layers=2, refiner=1, fmt_of=fmt_of, **WIDE)
    layers = {n[: -len(".comfy_quant")]: _w4a8_mark(16) for n in t if n.endswith(".comfy_quant")}
    for n in [n for n in t if n.endswith(".comfy_quant")]:
        del t[n]
    _, cfg = h3_format.inspect_h3_dit(_write(tmp_path / "dit.safetensors", t, metadata=_layers_meta(layers)))
    assert len(cfg.quant) == 8 and {q.group_size for q in cfg.quant.values()} == {16}
    assert cfg.unknown == ()


def test_metadata_keys_are_the_tensor_names_as_is(tmp_path):
    """A prefixed file lists its layers WITH the prefix; a key without it names no tensor and is ignored."""
    prefix = "model.diffusion_model."
    t = _dit_tensors(layers=2, refiner=1, fmt_of=_all("int8"), **WIDE)
    layers = {n[: -len(".comfy_quant")]: INT8_MARK for n in t if n.endswith(".comfy_quant")}
    for n in [n for n in t if n.endswith(".comfy_quant")]:
        del t[n]
    good = _write(tmp_path / "good.safetensors", t, prefix=prefix,
                  metadata=_layers_meta({prefix + k: v for k, v in layers.items()}))
    _, cfg = h3_format.inspect_h3_dit(good)
    assert len(cfg.quant) == 8 and "blocks.0.attn.qkv_proj" in cfg.quant
    bad = _write(tmp_path / "bad.safetensors", t, prefix=prefix, metadata=_layers_meta(layers))
    with pytest.raises(H3FormatError, match="量子化の印がありません"):
        h3_format.inspect_h3_dit(bad)


def test_old_metadata_marker_on_int8(tmp_path):
    """The h3ErosMax beta5 layout: int8 marked only in __metadata__._quantization_metadata (裁定 101 ⑤)."""
    t = _dit_tensors()
    layers = {n[: -len(".comfy_quant")]: INT8_MARK for n in t if n.endswith(".comfy_quant")}
    for n in [n for n in t if n.endswith(".comfy_quant")]:
        del t[n]
    _, cfg = h3_format.inspect_h3_dit(_write(tmp_path / "dit.safetensors", t, metadata=_layers_meta(layers)))
    assert set(cfg.quant.values()) == {QuantSpec("int8_tensorwise", True, 256)} and len(cfg.quant) == 12


def test_metadata_marker_wins_over_comfy_quant(tmp_path):
    """Both carriers: the metadata is taken (G 8 there, G 16 in the comfy_quant bytes; tensors laid out for 8)."""
    t = _dit_tensors(layers=2, refiner=1, fmt_of=_all({"fmt": "w4a8", "g": 8}), **WIDE)
    layers = {}
    for n in [n for n in t if n.endswith(".comfy_quant")]:
        t[n] = ("U8", (0,), _w4a8_mark(16))
        layers[n[: -len(".comfy_quant")]] = _w4a8_mark(8)
    _, cfg = h3_format.inspect_h3_dit(_write(tmp_path / "dit.safetensors", t, metadata=_layers_meta(layers)))
    assert {q.group_size for q in cfg.quant.values()} == {8}


def test_metadata_marker_wins_even_when_comfy_quant_is_unreadable(tmp_path):
    t = _dit_tensors(layers=2, refiner=1, fmt_of=_all("fp8"), **WIDE)
    layers = {}
    for n in [n for n in t if n.endswith(".comfy_quant")]:
        t[n] = ("U8", (0,), b"not json")
        layers[n[: -len(".comfy_quant")]] = FP8_MARK
    _, cfg = h3_format.inspect_h3_dit(_write(tmp_path / "dit.safetensors", t, metadata=_layers_meta(layers)))
    assert {q.format for q in cfg.quant.values()} == {"float8_e4m3fn"}


def test_full_precision_is_read_from_the_top_level_only():
    nested = json.dumps({"format": "float8_e4m3fn", "params": {"full_precision_matrix_mult": True}}).encode()
    top = json.dumps({"format": "float8_e4m3fn", "full_precision_matrix_mult": True}).encode()
    assert h3_format.parse_comfy_quant(nested, accepted=h3_format.DIT_FORMATS).full_precision is False
    assert h3_format.parse_comfy_quant(top, accepted=h3_format.DIT_FORMATS).full_precision is True


def test_w4a8_correction_is_refused(tmp_path):
    path = _wide(tmp_path, _all("w4a8"), extra={"blocks.1.mlp.fc2.weight_correction": ("F32", (16, 256))})
    with pytest.raises(H3FormatError, match="weight_correction"):
        h3_format.inspect_h3_dit(path)


@pytest.mark.parametrize("marker,match", [
    ({"format": "asym_w4a8_int8", "group_size": 16, "convrot": True, "convrot_groupsize": 128}, "convrot_groupsize=128"),
    ({"format": "asym_w4a8_int8", "group_size": 16, "full_precision_matrix_mult": True}, "full_precision_matrix_mult"),
])
def test_w4a8_marker_refusals(tmp_path, marker, match):
    t = _dit_tensors(layers=2, refiner=1, fmt_of=_all("w4a8"), **WIDE)
    t["blocks.0.mlp.fc1.comfy_quant"] = ("U8", (0,), marker)
    with pytest.raises(H3FormatError, match=match):
        h3_format.inspect_h3_dit(_write(tmp_path / "dit.safetensors", t))


@pytest.mark.parametrize("g", [2, 12, 48, 512])
def test_w4a8_group_size_condition_of_the_layout(tmp_path, g):
    """AsymW4A8Int8Layout.Params: G >= 4, K % G == 0, (16 % G == 0 or G % 16 == 0).
    The s_rel shape follows G so only the condition can refuse (512 > K = 256)."""
    t = _dit_tensors(layers=2, refiner=1, fmt_of=_all("w4a8"), **WIDE)
    t["blocks.0.attn.out_proj.comfy_quant"] = ("U8", (0,), _w4a8_mark(g))
    t["blocks.0.attn.out_proj.weight_s_rel"] = ("F8_E4M3", (256, max(1, 256 // g)))
    with pytest.raises(H3FormatError, match="group_size"):
        h3_format.inspect_h3_dit(_write(tmp_path / "dit.safetensors", t))


def test_w4a8_input_width_must_be_a_multiple_of_256(tmp_path):
    """The small widths (K = 64 / 32 / 48) fail K % 256 of the layout."""
    path = _write(tmp_path / "dit.safetensors", _dit_tensors(fmt_of=_all("w4a8")))
    with pytest.raises(H3FormatError, match="K="):
        h3_format.inspect_h3_dit(path)


@pytest.mark.parametrize("name,entry,match", [
    ("blocks.0.attn.qkv_proj.weight_s_rel", ("F32", (768, 16)), "weight_s_rel"),
    ("blocks.0.attn.qkv_proj.weight_s_channel", ("F32", (768, 1)), "weight_s_channel"),
    ("blocks.0.attn.qkv_proj.weight_codebook", ("F32", (15,)), "weight_codebook"),
])
def test_w4a8_aux_tensors_are_checked_exactly(tmp_path, name, entry, match):
    path = _wide(tmp_path, _all("w4a8"), extra={name: entry})
    with pytest.raises(H3FormatError, match=match):
        h3_format.inspect_h3_dit(path)


def test_w4a8_missing_s_channel_is_refused(tmp_path):
    path = _wide(tmp_path, _all("w4a8"), drop=["blocks.1.attn.out_proj.weight_s_channel"])
    with pytest.raises(H3FormatError, match="weight_s_channel"):
        h3_format.inspect_h3_dit(path)


@pytest.mark.parametrize("name,entry,match", [
    ("blocks.0.mlp.fc2.weight_scale", ("F32", (256, 1)), "weight_scale"),
    ("blocks.0.mlp.fc2.input_scale", ("F16", ()), "input_scale"),
])
def test_fp8_scales_are_checked_exactly(tmp_path, name, entry, match):
    path = _wide(tmp_path, _all("fp8"), extra={name: entry})
    with pytest.raises(H3FormatError, match=match):
        h3_format.inspect_h3_dit(path)


def test_fp8_without_weight_scale_and_with_convrot_are_refused(tmp_path):
    with pytest.raises(H3FormatError, match="weight_scale"):
        h3_format.inspect_h3_dit(_wide(tmp_path, _all("fp8"), drop=["blocks.0.mlp.fc1.weight_scale"]))
    t = _dit_tensors(layers=2, refiner=1, fmt_of=_all("fp8"), **WIDE)
    t["blocks.0.mlp.fc1.comfy_quant"] = ("U8", (0,), {**FP8_MARK, "convrot": True})
    with pytest.raises(H3FormatError, match="convrot"):
        h3_format.inspect_h3_dit(_write(tmp_path / "rot.safetensors", t))


@pytest.mark.parametrize("dtype", ["F8_E4M3", "F8_E5M2", "I8"])
def test_quantized_weight_without_a_marker_is_refused(tmp_path, dtype):
    t = _dit_tensors(layers=2, refiner=1, fmt_of=_all("fp8"), **WIDE)
    del t["blocks.1.mlp.fc1.comfy_quant"]
    t["blocks.1.mlp.fc1.weight"] = (dtype, (512, 256))
    with pytest.raises(H3FormatError, match="量子化の印がありません"):
        h3_format.inspect_h3_dit(_write(tmp_path / "dit.safetensors", t))


def test_stray_weight_scale_on_a_float_layer_is_refused(tmp_path):
    t = _dit_tensors()
    t["token_refiner.blocks.0.mlp.fc1.weight_scale"] = ("F32", (2 * FFN, 1))
    with pytest.raises(H3FormatError, match="token_refiner.blocks.0.mlp.fc1.weight_scale"):
        h3_format.inspect_h3_dit(_write(tmp_path / "dit.safetensors", t))


def test_marker_on_a_float_weight_is_refused(tmp_path):
    t = _dit_tensors()
    t["token_refiner.blocks.0.mlp.fc1.comfy_quant"] = ("U8", (0,))
    with pytest.raises(H3FormatError, match="I8 ではありません"):
        h3_format.inspect_h3_dit(_write(tmp_path / "dit.safetensors", t))


def test_shape_contradiction_of_a_packed_weight_is_refused(tmp_path):
    """The table checks the LOGICAL width: a w4a8 fc2 stored as [N, K] (not K/2) reads as 2K."""
    t = _dit_tensors(layers=2, refiner=1, fmt_of=_all({"fmt": "w4a8", "codebook": False}), **WIDE)
    t["blocks.1.mlp.fc2.weight"] = ("I8", (256, 512))
    t["blocks.1.mlp.fc2.weight_s_rel"] = ("F8_E4M3", (256, 1024 // 16))
    with pytest.raises(H3FormatError, match="矛盾"):
        h3_format.inspect_h3_dit(_write(tmp_path / "dit.safetensors", t))


# --------------------------------------------------------------------------- #
# text encoder (in-memory header; only the comfy_quant bytes are on disk)
# --------------------------------------------------------------------------- #

def _te_header(tmp_path, *, layers=50, visual=True, marker=INT8_MARK):
    mark = json.dumps(marker).encode()
    blob = tmp_path / "te_marks.bin"
    blob.write_bytes(mark)
    shapes = {
        "self_attn.q_proj": (8192, 5120), "self_attn.k_proj": (1024, 5120),
        "self_attn.v_proj": (1024, 5120), "self_attn.o_proj": (5120, 8192),
        "mlp.gate_proj": (25600, 5120), "mlp.up_proj": (25600, 5120), "mlp.down_proj": (5120, 25600),
    }
    tensors: dict[str, TensorInfo] = {"model.embed_tokens.weight": TensorInfo("BF16", (151936, 5120), (0, 0))}
    for i in range(layers):
        h = f"model.layers.{i}."
        for name, shp in shapes.items():
            tensors[f"{h}{name}.weight"] = TensorInfo("I8", shp, (0, 0))
            tensors[f"{h}{name}.weight_scale"] = TensorInfo("F32", (shp[0], 1), (0, 0))
            tensors[f"{h}{name}.comfy_quant"] = TensorInfo("U8", (len(mark),), (0, len(mark)))
        for leaf in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
            tensors[h + leaf] = TensorInfo("BF16", (128,), (0, 0))
        for leaf in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            tensors[h + leaf] = TensorInfo("BF16", (5120,), (0, 0))
    if visual:
        tensors["visual.patch_embed.proj.weight"] = TensorInfo("BF16", (1152, 3, 2, 16, 16), (0, 0))
    meta = {"minimax_h3_te": json.dumps({"num_hidden_layers": 50, "output": "unnormalized_hidden_after_layer_50"})}
    header = Header(tensors=tensors, metadata=meta, header_len=0, data_base=0, file_size=len(mark))
    return str(blob), header


def test_te_inspection(tmp_path, monkeypatch):
    path, header = _te_header(tmp_path)
    monkeypatch.setattr(h3_format, "read_header", lambda p: header)
    _, cfg = h3_format.inspect_h3_te(path)
    assert cfg.num_layers == 50
    assert len(cfg.quant) == 350
    assert cfg.quant["model.layers.0.self_attn.q_proj"] == QuantSpec("int8_tensorwise", True, 256)
    assert cfg.has_visual is True
    assert cfg.metadata["minimax_h3_te"]["num_hidden_layers"] == 50


def test_te_with_too_few_layers_is_refused(tmp_path, monkeypatch):
    path, header = _te_header(tmp_path, layers=49, visual=False)
    monkeypatch.setattr(h3_format, "read_header", lambda p: header)
    with pytest.raises(H3FormatError, match="50"):
        h3_format.inspect_h3_te(path)


@pytest.mark.parametrize("marker", [FP8_MARK, _w4a8_mark(16)])
def test_te_refuses_fp8_and_w4a8_markers(tmp_path, monkeypatch, marker):
    """The text encoder stays int8 (裁定 101 ①): its inspection accepts int8 markers only."""
    path, header = _te_header(tmp_path, marker=marker)
    monkeypatch.setattr(h3_format, "read_header", lambda p: header)
    with pytest.raises(H3FormatError, match=marker["format"]):
        h3_format.inspect_h3_te(path)


def test_dit_file_is_not_a_te(tmp_path):
    path = _write(tmp_path / "dit.safetensors", _dit_tensors())
    with pytest.raises(H3FormatError, match="embed_tokens"):
        h3_format.inspect_h3_te(path)


def test_check_model_paths(tmp_path):
    f = tmp_path / "a.safetensors"
    f.write_bytes(b"x" * 3)
    lines = h3_format.check_model_paths(files={"transformer_path": str(f)}, dirs={"tokenizer_dir": str(tmp_path)})
    assert len(lines) == 2 and "transformer_path OK (3 bytes)" in lines[0]
    with pytest.raises(ValueError, match="transformer_path"):
        h3_format.check_model_paths(files={"transformer_path": ""}, dirs={})
    with pytest.raises(FileNotFoundError, match="video_vae_path"):
        h3_format.check_model_paths(files={"video_vae_path": str(tmp_path / "missing")}, dirs={})
    with pytest.raises(FileNotFoundError, match="tokenizer_dir"):
        h3_format.check_model_paths(files={}, dirs={"tokenizer_dir": str(f)})
