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


def _dit_tensors(*, layers=3, refiner=2, grid=1025, rank=8, quant=True, pruned=True):
    """name -> (dtype, shape) of a minimal H3 DiT (no prefix)."""
    t: dict[str, tuple[str, tuple[int, ...]]] = {
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
            if q:
                t[f"{head}{name}.weight"] = ("I8", shape)
                t[f"{head}{name}.weight_scale"] = ("F32", (shape[0], 1))
                t[f"{head}{name}.comfy_quant"] = ("U8", (0,))  # filled by _write
            else:
                t[f"{head}{name}.weight"] = ("BF16", shape)
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


def _write(path, tensors, *, prefix="", marker=INT8_MARK, metadata=None):
    """Write a real safetensors file: zero data, ``comfy_quant`` = JSON bytes of ``marker``."""
    mark = marker if isinstance(marker, bytes) else json.dumps(marker).encode()
    doc: dict = {}
    blobs: list[bytes] = []
    off = 0
    for name, (dtype, shape) in tensors.items():
        if name.endswith(".comfy_quant"):
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


@pytest.mark.parametrize("fmt", ["float8_e4m3fn", "int4", None])
def test_comfy_quant_other_formats_are_refused(fmt):
    with pytest.raises(H3FormatError, match="format"):
        h3_format.parse_comfy_quant(json.dumps({"format": fmt}).encode())


def test_comfy_quant_unknown_keys_ignored_in_a_file(tmp_path):
    marker = {**INT8_MARK, "convrot_groupsize": 128, "per_row": True, "orig_dtype": "bfloat16"}
    path = _write(tmp_path / "dit.safetensors", _dit_tensors(), marker=marker)
    _, cfg = h3_format.inspect_h3_dit(path)
    assert set(cfg.quant.values()) == {QuantSpec("int8_tensorwise", True, 128)}


def test_fp8_marker_in_a_file_is_refused(tmp_path):
    path = _write(tmp_path / "dit.safetensors", _dit_tensors(), marker={"format": "float8_e4m3fn"})
    with pytest.raises(H3FormatError, match="float8_e4m3fn"):
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
