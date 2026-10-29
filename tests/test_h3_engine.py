"""engine_h3 pieces that can be checked without a GPU (台帳 §1-93 第 1 弾).

Runs in ``.venv-engine-h3`` (torch, comfy-kitchen, transformers 5.14.1). That
venv has no pytest and the app ``conftest.py`` imports FastAPI, so it is run
through the engine-venv runner (app site-packages APPENDED, ``--noconftest``)::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_engine.py

What is pinned (plan section 3.4, R2 (3)):
  * the pruned AdaLN interpolation never reads past the table (t = 1.0);
  * the 17n+5 frame rule (124 -> 37 / 207, 345 -> 102 / 575);
  * the two sigma schedules for N = 20 against the R2 G table, N steps = N
    evaluations, and the Euler step's direction (x0 = x + sigma v);
  * the timestep numbering M (1 at step 0, 2 afterwards) and the modality rows;
  * the RoPE coordinates on a small grid, computed by hand;
  * the text encoder's config after transformers' rope translation, the extra
    special token ids (only when a tokenizer folder is present), and the
    Comfy-Org -> model name mapping against the real meta-built models;
  * the text encoder's direct read (第 2 弾 2a, §1-98): the read plan, and a
    tiny Qwen3-VL text model read layer by layer from a synthetic file through
    the forward hooks gives the same hidden state as the fully loaded model,
    with every weight back on meta afterwards;
  * a tiny H3 DiT streamed from a synthetic file: no meta tensor left;
  * ``H3Int8Linear`` against the dequantized bf16 ``F.linear`` (GPU only,
    opt-in with ``H3_GPU_TESTS=1``);
  * VRAM 16GB mode (``engine_h3.lowvram``, 台帳 §1-101): the equal row chunks, the
    swap window and the allocator cap arithmetic, and the block / final layer /
    whole forward / whole pipeline giving the off path's numbers bit for bit
    (the int8 matmul faked in float64);
  * no ``inference_mode`` anywhere under ``engine_h3``;
  * the worker's unknown-op handling.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("comfy_kitchen")

import h3_format  # noqa: E402
import numpy as np  # noqa: E402
from engine_h3 import constants as C  # noqa: E402
from engine_h3.scheduler import euler_step, sigma_schedule  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Tiny but self-consistent H3 widths. head_dim stays 128: RoPE rotates its first 96 channels.
HID, HEADS, HD, FFN, TEXT, LAT, AUD = 64, 1, 128, 48, 40, 24, 32
INNER = HEADS * HD


def _tiny_cfg(*, layers=2, refiner=2, grid=1025, rank=8, quant=None, heads=HEADS):
    return h3_format.H3DiTConfig(
        prefix="", hidden_size=HID, num_layers=layers, token_refiner_layers=refiner,
        num_heads=heads, head_dim=HD, ffn_hidden=FFN, text_dim=TEXT, latents_dim=LAT,
        audio_latents_dim=AUD, pruned=True, adaln_grid=grid, adaln_rank=rank,
        rope_inv_freq_len=16, quant=quant or {}, unknown=(),
    )


# --------------------------------------------------------------------------- #
# AdaLN table interpolation
# --------------------------------------------------------------------------- #

def test_adaln_interpolation_ends_do_not_read_past_the_table():
    from engine_h3.dit import H3DiT

    g, r = 1025, 8
    dit = H3DiT(_tiny_cfg(grid=g, rank=r))
    table = torch.arange(g * r, dtype=torch.float32).view(g, r)
    dit.adaln_t_table = table
    t = torch.tensor([0.0, 1.0, 0.999, 1.5, -0.5, 0.5], dtype=torch.float32)
    e = dit.adaln_embed(t)
    assert e.shape == (6, r)
    assert torch.equal(e[0], table[0])
    assert torch.equal(e[1], table[g - 1])        # t = 1.0: last row, no table[G]
    assert torch.equal(e[3], table[g - 1])        # clamped above
    assert torch.equal(e[4], table[0])            # clamped below
    pos = 0.999 * (g - 1)
    i0 = math.floor(pos)
    w = pos - i0
    assert torch.allclose(e[2], table[i0] + w * (table[i0 + 1] - table[i0]), rtol=0, atol=1e-2)
    assert torch.allclose(e[5], table[512], rtol=0, atol=1e-4)  # 0.5 * 1024 = 512 exactly


# --------------------------------------------------------------------------- #
# frame rule, schedules
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("frames,lat,aud", [(124, 37, 207), (345, 102, 575), (192, 57, 320), (243, 72, 405)])
def test_frame_rule(frames, lat, aud):
    assert C.latent_frames(frames) == lat
    assert C.audio_latents(frames) == aud


@pytest.mark.parametrize("frames", [123, 125, 361, 0])
def test_frame_rule_refuses_off_grid(frames):
    with pytest.raises(ValueError):
        C.latent_frames(frames)


#: R2 G (DiffSynth / SGLang counting: 21 points, 20 evaluations).
SIGMA_VIDEO_20 = [1.0000, 0.9956, 0.9908, 0.9855, 0.9796, 0.9730, 0.9655, 0.9571, 0.9474, 0.9362,
                  0.9231, 0.9076, 0.8889, 0.8660, 0.8372, 0.8000, 0.7500, 0.6792, 0.5714, 0.3871, 0.0]
SIGMA_AUDIO_20 = [1.0000, 0.9828, 0.9643, 0.9444, 0.9231, 0.9000, 0.8750, 0.8478, 0.8182, 0.7857,
                  0.7500, 0.7105, 0.6667, 0.6176, 0.5625, 0.5000, 0.4286, 0.3462, 0.2500, 0.1364, 0.0]


@pytest.mark.parametrize("shift,expected", [(12.0, SIGMA_VIDEO_20), (3.0, SIGMA_AUDIO_20)])
def test_sigma_schedule_n20(shift, expected):
    s = sigma_schedule(20, shift)
    assert s.dtype == torch.float32
    assert s.shape == (21,)                       # 20 evaluated sigmas + the terminal 0
    assert torch.allclose(s, torch.tensor(expected), rtol=0, atol=6e-5)
    assert float(s[-1]) == 0.0 and bool((s[:-1] > 0).all())


def test_video_and_audio_schedules_differ():
    assert not torch.allclose(sigma_schedule(20, C.DEFAULT_SHIFT_VIDEO), sigma_schedule(20, C.DEFAULT_SHIFT_AUDIO))


def test_euler_step_goes_towards_the_data():
    x = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
    v = torch.tensor([0.5, 4.0])
    out = euler_step(x, v, 0.75, 0.25)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), torch.tensor([1.25, 0.0]))
    # The last step lands on x0 = x + sigma v exactly.
    assert torch.allclose(euler_step(torch.zeros(1), torch.ones(1), 0.3871, 0.0), torch.tensor([0.3871]))


def test_schedule_rejects_bad_arguments():
    with pytest.raises(ValueError):
        sigma_schedule(0, 12.0)
    with pytest.raises(ValueError):
        sigma_schedule(20, 0.0)


# --------------------------------------------------------------------------- #
# timestep numbering and modality rows
# --------------------------------------------------------------------------- #

def test_time_numbering_m_is_one_then_two():
    from engine_h3.dit import H3SeqLayout
    from engine_h3.pipeline_h3 import time_rows

    layout = H3SeqLayout.make(text_rows=5, cond_rows=0, audio_rows=4, video_rows=6)
    sv, sa = sigma_schedule(20, 12.0), sigma_schedule(20, 3.0)
    ms = []
    for i in range(20):
        t_v, t_a = 1.0 - float(sv[i]), 1.0 - float(sa[i])
        t_unique, idx = time_rows(layout, t_v, t_a)
        ms.append(int(t_unique.numel()))
        assert idx.dtype == torch.long and idx.shape == (layout.total,)
        # text and video rows share the video t, audio rows the audio t
        assert torch.allclose(t_unique[idx[layout.text]], torch.tensor(t_v))
        assert torch.allclose(t_unique[idx[layout.video]], torch.tensor(t_v))
        assert torch.allclose(t_unique[idx[layout.audio]], torch.tensor(t_a))
    assert ms[0] == 1 and ms[1:] == [2] * 19


def test_modality_rows():
    from engine_h3.dit import H3SeqLayout
    from engine_h3.pipeline_h3 import modality_rows

    layout = H3SeqLayout.make(text_rows=3, cond_rows=0, audio_rows=4, video_rows=2)
    m = modality_rows(layout, torch.full((3,), C.MODALITY_TEXT, dtype=torch.long))
    assert m.tolist() == [1, 1, 1, 2, 2, 2, 2, 0, 0]


def test_patchify_and_audio_rows_round_trip():
    from engine_h3.dit import audio_to_rows, patchify_video, rows_to_audio, unpatchify_video

    v = torch.randn(1, 24, 7, 4, 6)
    rows = patchify_video(v)
    assert rows.shape == (7 * 2 * 3, 96)
    assert torch.equal(unpatchify_video(rows, 7, 2, 3), v)
    a = torch.randn(2, 32, 5)
    ar = audio_to_rows(a)
    assert ar.shape == (10, 32) and torch.equal(ar[:5], a[0].T)
    assert torch.equal(rows_to_audio(ar, 5), a)


# --------------------------------------------------------------------------- #
# RoPE coordinates, by hand
# --------------------------------------------------------------------------- #

def test_rope_coordinates_small_grid():
    from engine_h3.rope import build_coords, rope_tables

    L, T, gh, gw, A = 2, 6, 1, 2, 2
    g = build_coords(text_len=L, latent_frames=T, grid_h=gh, grid_w=gw, audio_latents=A)
    assert g.dtype == torch.float64 and g.shape == (L + 2 * A + T * gh * gw, 3)

    # latent grid 2 x 4: sqrt(area) = sqrt(8); h axis has 1 point, w axis 2.
    sq = math.sqrt(8.0)
    h_ratio, w_ratio = 2 / sq, 4 / sq
    h0 = (1 - h_ratio) / 2 * 32
    w_left = (1 - w_ratio) / 2
    w0, w1 = w_left * 32, (w_left + w_ratio / 2) * 32
    rows = [(0, 0, 0), (1, 0, 0)]                                  # text
    rows += [(L + 0, 0, w0), (L + 1, 0, w0), (L + 0, 0, w1), (L + 1, 0, w1)]  # audio ch0, ch1
    spans = [5 / 3, 20 / 3, 20 / 3, 20 / 3, 20 / 3]
    t = L
    for k in range(T):                                              # video, t -> h -> w
        rows += [(t, h0, w0), (t, h0, w1)]
        t += spans[k % 5] if k < T - 1 else 0
    expected = torch.tensor(rows, dtype=torch.float64)
    assert torch.allclose(g, expected, rtol=0, atol=1e-9)
    assert abs(w1 - 16.0) < 1e-12  # the right end of this grid is exactly 16

    inv = 1.0 / (10000 ** (torch.arange(16, dtype=torch.float32) / 16))
    cos, sin = rope_tables(g, inv)
    assert cos.shape == (g.shape[0], 96) and cos.dtype == torch.float32
    # t axis of the last video row, first frequency
    assert torch.allclose(cos[-1, 0], torch.cos(torch.tensor(float(rows[-1][0]), dtype=torch.float32) * inv[0]))
    assert torch.allclose(cos[:, :48], cos[:, 48:])  # the 48-block is repeated for rotate_half


def test_keyframe_t_last_frame_is_the_oracle_expression_bit_for_bit():
    """台帳 §1-97: the last frame's time is DiffSynth ``_build_packed_fl2va``'s
    ``(float(L) + temporal_span) - 5/3`` (the oracle, written inline here:
    pipeline_coords.py:38-45 and :86-99), for every 17n+5 count and L = 0..8192.

    Also counts how often the exact-arithmetic twin ``L + (5/3)(F-1)`` gives the
    same float64: it does NOT always (VERIFICATION_LOG §170), which is why
    ``keyframe_t`` keeps the F-1 branch."""
    from engine_h3.rope import keyframe_t

    total = matched = 0
    for n in range(1, 29):
        num_frames, t_lat = 17 * n + 5, 5 * n + 2
        sp = np.ones(t_lat, dtype=np.float64) * (5.0 / 3.0)
        for k, per in enumerate((1, 4, 4, 4, 4)):
            sp[k::5] *= per
        span = float(sp.sum())
        for L in range(8193):
            oracle = (float(L) + span) - 5.0 / 3.0
            assert keyframe_t(num_frames - 1, t_lat, L) == oracle, (num_frames, L)
            total += 1
            matched += float(L) + (5.0 / 3.0) * (num_frames - 1) == oracle
    print(f"L + (5/3)(F-1) matches the oracle in {matched} of {total} (F, L) pairs")
    assert total == 28 * 8193
    assert matched < total  # so the F-1 branch of keyframe_t is needed


def test_keyframe_t_first_middle_and_errors():
    from engine_h3.rope import keyframe_t

    t_lat, F, L = 7, 22, 300
    assert keyframe_t(0, t_lat, L) == float(L)
    for n in (1, F // 2, F - 2):
        assert keyframe_t(n, t_lat, L) == float(L) + (5.0 / 3.0) * n
    for bad in (-1, F, F + 5):
        with pytest.raises(ValueError):
            keyframe_t(bad, t_lat, L)
    with pytest.raises(ValueError):
        keyframe_t(0, 6, L)  # not 5n+2


@pytest.mark.parametrize("gh, gw", [(1, 2), (2, 3), (12, 20)])
@pytest.mark.parametrize("frames", [[0], [21], [0, 21], [1], [11], [20], [0, 11, 21]])
def test_build_coords_keyframe_blocks(gh, gw, frames):
    """One block of grid rows per key frame, in the given order, each with its own
    t and the video grid's (h, w); the text / audio / video rows do not move."""
    from engine_h3.rope import build_coords, keyframe_t

    L, T, A = 10, 7, 4   # T' = 7: F = 22
    grid = gh * gw
    g = build_coords(text_len=L, latent_frames=T, grid_h=gh, grid_w=gw, audio_latents=A, cond_frame_idx=frames)
    plain = build_coords(text_len=L, latent_frames=T, grid_h=gh, grid_w=gw, audio_latents=A)
    K = len(frames)
    assert g.shape == (L + K * grid + 2 * A + T * grid, 3) and g.dtype == torch.float64
    assert torch.equal(g[:L], plain[:L])
    assert torch.equal(g[L + K * grid:], plain[L:])                 # audio and video unchanged
    video_first = plain[L + 2 * A:L + 2 * A + grid]
    for k, n in enumerate(frames):
        block = g[L + k * grid:L + (k + 1) * grid]
        assert bool((block[:, 0] == keyframe_t(n, T, L)).all())
        assert torch.equal(block[:, 1:], video_first[:, 1:])
    if frames == [0]:
        assert bool((g[L:L + grid, 0] == float(L)).all())          # the first-frame anchor as before
    if 0 < frames[0] < 21:
        assert float(g[L, 0]) == float(L) + (5.0 / 3.0) * frames[0]


def test_build_coords_refuses_bad_keyframes():
    from engine_h3.rope import build_coords

    kw = dict(text_len=10, latent_frames=7, grid_h=1, grid_w=2, audio_latents=4)
    for bad in ([22], [-1], [0, 0], [5, 21, 5]):
        with pytest.raises(ValueError):
            build_coords(**kw, cond_frame_idx=bad)


# --------------------------------------------------------------------------- #
# text encoder config, tokens, names
# --------------------------------------------------------------------------- #

def test_te_config_after_rope_translation():
    pytest.importorskip("transformers")
    from engine_h3.text_encoder import build_text_config, check_rope

    cfg = build_text_config()
    got = check_rope(cfg)
    assert got["mrope_section"] == [24, 20, 20]
    assert got["rope_theta"] == 5000000.0
    assert got["head_dim"] == 128
    assert got["num_attention_heads"] == 64 and got["num_key_value_heads"] == 8
    assert got["num_hidden_layers"] == 50
    assert cfg.hidden_size == 5120 and cfg.intermediate_size == 25600
    assert abs(cfg.rms_norm_eps - 1e-6) < 1e-12
    # transformers' Qwen3-VL text rotary always interleaves; the flag is kept
    # for the record when the translation carries it.
    assert got["mrope_interleaved"] in (True, None)


def _tokenizer_dir() -> Path | None:
    for cand in (
        ROOT / "models" / "MiniMaxH3" / "TextEncoder" / "qwen3vl_tokenizer",
        ROOT / "models" / ".dl" / "MiniMaxH3-stage" / "official" / "FL2VA" / "text_encoder",
    ):
        if (cand / "tokenizer.json").is_file() and (cand / "tokenizer_config.json").is_file():
            return cand
    return None


def test_extra_special_token_ids():
    pytest.importorskip("transformers")
    d = _tokenizer_dir()
    if d is None:
        pytest.skip("no Qwen3-VL tokenizer folder present")
    from transformers import AutoTokenizer

    from engine_h3.text_encoder import EXTRA_SPECIAL_TOKENS, ensure_extra_special_tokens

    tok = AutoTokenizer.from_pretrained(str(d))
    info = ensure_extra_special_tokens(tok)
    assert [info["ids"][t] for t in EXTRA_SPECIAL_TOKENS] == list(range(151669, 151676))
    ids = tok("<d>[English] Hello.</d><|cutoff|>", add_special_tokens=False)["input_ids"]
    assert ids[0] == 151669 and 151670 in ids and ids[-1] == 151671
    # no chat template, no specials: a plain sentence starts with its own first token
    plain = tok("A cat.", add_special_tokens=False)["input_ids"]
    assert all(i < 151643 for i in plain)


def test_te_name_mapping():
    from engine_h3.text_encoder import te_key_map, te_quant_names

    # Stage 2 (2a, I2V): the encoder is Qwen3VLModel, so the text side lives under
    # ``language_model.`` and the vision tower (``visual.*``) is mapped as is.
    assert te_key_map("model.layers.0.self_attn.q_proj.weight") == "language_model.layers.0.self_attn.q_proj.weight"
    assert (te_key_map("model.layers.49.mlp.down_proj.weight_scale")
            == "language_model.layers.49.mlp.down_proj.weight_scale")
    assert te_key_map("model.layers.50.mlp.down_proj.weight") is None
    assert te_key_map("model.embed_tokens.weight") == "language_model.embed_tokens.weight"
    assert te_key_map("visual.blocks.0.attn.qkv.weight") == "visual.blocks.0.attn.qkv.weight"
    for dropped in ("lm_head.weight", "model.norm.weight"):
        assert te_key_map(dropped) is None
    q = te_quant_names({"model.layers.3.self_attn.o_proj": "spec", "model.layers.55.mlp.up_proj": "x"})
    assert q == {"language_model.layers.3.self_attn.o_proj": "spec"}


def test_te_names_match_the_meta_built_model():
    """Every Comfy-Org TE name we read lands on a key of the real (meta) Qwen3VLModel
    (text side cut to 50 layers); with the real TE file present, its 351 vision
    tensors match the model's vision tower exactly."""
    pytest.importorskip("transformers")
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    from engine_h3.quant_linear import H3Int8Linear, swap_in_int8_linears
    from engine_h3.text_encoder import build_config, te_key_map, te_quant_names

    spec = h3_format.QuantSpec("int8_tensorwise", True, 256)
    linears = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    quant = {f"model.layers.{i}.{n}": spec for i in range(50) for n in linears}
    with torch.device("meta"):
        model = Qwen3VLModel(build_config())
        model.language_model.norm = torch.nn.Identity()
        swapped = swap_in_int8_linears(model, te_quant_names(quant), bias_dtype_of=lambda n: torch.bfloat16)
    assert swapped == 350
    assert isinstance(model.language_model.layers[0].self_attn.q_proj, H3Int8Linear)
    all_keys = set(model.state_dict())
    keys = {k for k in all_keys if k.startswith("language_model.")}
    assert all_keys - keys == {k for k in all_keys if k.startswith("visual.")}
    names = ["model.embed_tokens.weight"]
    for i in range(50):
        h = f"model.layers.{i}."
        names += [f"{h}{n}.{leaf}" for n in linears for leaf in ("weight", "weight_scale")]
        names += [f"{h}{leaf}" for leaf in ("self_attn.q_norm.weight", "self_attn.k_norm.weight",
                                            "input_layernorm.weight", "post_attention_layernorm.weight")]
    mapped = {te_key_map(n) for n in names}
    assert mapped == keys  # exactly: nothing unread, nothing unknown
    assert "language_model.norm.weight" not in keys
    te_file = ROOT / "models" / "MiniMaxH3" / "TextEncoder" / "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    if te_file.is_file():
        import sft_quant_format

        header = sft_quant_format.read_header(str(te_file))
        file_visual = {te_key_map(n) for n in header.tensors if n.startswith("visual.")}
        assert file_visual == {k for k in all_keys if k.startswith("visual.")}
        assert {header.tensors[n].dtype for n in header.tensors if n.startswith("visual.")} == {"BF16"}


def test_te_read_plan_splits_embedding_and_layers():
    from engine_h3.text_encoder import te_read_plan

    names = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.weight_scale",
        "model.layers.0.self_attn.q_proj.comfy_quant",
        "model.layers.1.mlp.down_proj.weight", "model.layers.50.mlp.down_proj.weight",
        "visual.blocks.0.attn.qkv.weight", "lm_head.weight", "model.norm.weight",
    ]
    embed, layers, visual = te_read_plan(names, num_layers=2)
    assert embed == {"model.embed_tokens.weight": "weight"}
    assert visual == {"visual.blocks.0.attn.qkv.weight": "blocks.0.attn.qkv.weight"}
    assert layers[0] == {
        "model.layers.0.self_attn.q_proj.weight": "self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight_scale": "self_attn.q_proj.weight_scale",
    }
    assert layers[1] == {"model.layers.1.mlp.down_proj.weight": "mlp.down_proj.weight"}


def test_load_mapped_to_and_drop_to_meta(tmp_path):
    import sft_quant_format
    from safetensors.torch import save_file

    from engine_h3.sft_load import drop_to_meta, load_mapped_to, meta_tensor_names

    torch.manual_seed(0)
    src = torch.nn.Linear(4, 3).to(torch.bfloat16)
    path = tmp_path / "lin.safetensors"
    save_file({"x.weight": src.weight.data.contiguous(), "x.bias": src.bias.data.contiguous()}, str(path))
    header = sft_quant_format.read_header(str(path))
    with torch.device("meta"):
        dst = torch.nn.Linear(4, 3)
    dst.requires_grad_(False)
    n = load_mapped_to(dst, str(path), header, {"x.weight": "weight", "x.bias": "bias"}, torch.device("cpu"))
    assert n == (12 + 3) * 2
    assert dst.weight.dtype == torch.bfloat16 and torch.equal(dst.weight, src.weight)
    assert not dst.weight.requires_grad
    drop_to_meta(dst)
    assert sorted(meta_tensor_names(dst)) == ["bias", "weight"]
    assert dst.weight.dtype == torch.bfloat16 and tuple(dst.weight.shape) == (3, 4)
    # a tensor the file does not give is refused by name
    with pytest.raises(RuntimeError, match="still on meta"):
        load_mapped_to(dst, str(path), header, {"x.weight": "weight"}, torch.device("cpu"))
    drop_to_meta(dst)
    with pytest.raises(RuntimeError, match="no place"):
        load_mapped_to(dst, str(path), header, {"x.weight": "weight", "x.bias": "nope"}, torch.device("cpu"))


def test_te_direct_read_hooks_match_the_loaded_model(tmp_path):
    """A tiny Qwen3-VL text model: the unmodified HF forward with the per-layer read
    hooks gives exactly the hidden state of the fully loaded model (the CPU
    version of gate 2a-G2), and leaves every layer on meta."""
    pytest.importorskip("transformers")
    import sft_quant_format
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
    load_mapped_to(model.embed_tokens, str(path), header, embed_plan, dev)
    move_module_tree(model.rotary_emb, dev, skip=set())
    handles = install_layer_read_hooks(model.layers, layer_plans, str(path), header, dev, stats)
    with torch.no_grad():
        got = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
    for h in handles:
        h.remove()
    assert torch.equal(got, want)
    assert stats["layers"] == 3 and stats["bytes"] > 0
    left = meta_tensor_names(model)
    assert left and all(n.startswith("layers.") for n in left)   # every layer dropped, embedding still held
    assert not [n for n, _ in model.layers.named_parameters() if not _.is_meta]


def test_presentation_t2v_and_fl2va():
    pytest.importorskip("transformers")
    d = _tokenizer_dir()
    if d is None:
        pytest.skip("no Qwen3-VL tokenizer folder present")
    from transformers import AutoTokenizer

    from engine_h3.text_encoder import mm_token_type_ids, presentation

    tok = AutoTokenizer.from_pretrained(str(d))
    prompt = "A cat walks. <d>[English] Hello.</d>"
    ids, tags = presentation(tok, prompt)
    assert ids == tok(prompt, add_special_tokens=False)["input_ids"] and tags == [1] * len(ids)
    ids2, tags2 = presentation(tok, prompt, image_tokens=(240,))
    label = tok("<Picture 1>: ", add_special_tokens=False)["input_ids"]
    assert label == [21604, 3826, 220, 16, 26818, 220]
    n = len(label)
    assert ids2[:n] == label and ids2[n] == 151652 and ids2[n + 1:n + 241] == [151655] * 240
    assert ids2[n + 241] == 151653 and ids2[n + 242:] == ids
    assert tags2 == [1] * n + [0] * 242 + [1] * len(ids)
    mm = mm_token_type_ids(ids2)
    assert sum(mm) == 240 and mm[n] == 0 and mm[n + 241] == 0

    # Two pictures (first frame, last frame): label 1 + block 1 + label 2 + block 2 + prompt.
    ids3, tags3 = presentation(tok, prompt, image_tokens=(240, 100))
    label2 = tok("<Picture 2>: ", add_special_tokens=False)["input_ids"]
    m = len(label2)
    block1 = [151652] + [151655] * 240 + [151653]
    block2 = [151652] + [151655] * 100 + [151653]
    assert ids3 == label + block1 + label2 + block2 + ids
    assert tags3 == [1] * n + [0] * 242 + [1] * m + [0] * 102 + [1] * len(ids)
    assert sum(mm_token_type_ids(ids3)) == 340
    with pytest.raises(ValueError):
        presentation(tok, prompt, image_tokens=(240, 0))


def test_te_direct_read_with_vision_matches_the_loaded_model(tmp_path):
    """I2V version of the direct-read check: a tiny Qwen3VLModel with a vision tower
    and DeepStack, read from a synthetic Comfy-Org-named file (text layers through
    the hooks, the vision tower read whole before the forward), equals the fully
    loaded model bit for bit."""
    pytest.importorskip("transformers")
    import sft_quant_format
    from safetensors.torch import save_file
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLTextRotaryEmbedding

    from engine_h3.sft_load import drop_to_meta, load_mapped_to, meta_tensor_names, move_module_tree
    from engine_h3.text_encoder import install_layer_read_hooks, te_read_plan

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
    torch.manual_seed(0)
    ref = Qwen3VLModel(cfg).to(torch.bfloat16).eval()
    ref.language_model.norm = torch.nn.Identity()
    # .to(bf16) also rounded the rotary buffers; the encoder keeps them float32.
    ref.visual.rotary_pos_emb.inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))
    r_inv, _ = Qwen3VLTextRotaryEmbedding.compute_default_rope_parameters(cfg.text_config, device="cpu")
    ref.language_model.rotary_emb.inv_freq = r_inv.float()
    ref.language_model.rotary_emb.original_inv_freq = r_inv.float().clone()
    with torch.no_grad():
        for prm in ref.parameters():
            prm.normal_(0.0, 0.05)
    names = {}
    for k, v in ref.state_dict().items():
        if k.startswith("language_model."):
            names["model." + k[len("language_model."):]] = v.contiguous()
        else:
            names[k] = v.contiguous()
    path = tmp_path / "te.safetensors"
    save_file(names, str(path))
    grid = torch.tensor([[1, 4, 4]])                       # 4 vision tokens after the 2x2 merge
    pixels = torch.randn(16, 3 * 2 * 16 * 16)
    ids = [5, 17, 93, 95, 95, 95, 95, 94, 3, 41, 8]
    input_ids = torch.tensor([ids])
    kw = dict(attention_mask=torch.ones_like(input_ids), use_cache=False,
              mm_token_type_ids=torch.tensor([[1 if i == 95 else 0 for i in ids]]),  # tiny image token 95
              pixel_values=pixels.to(torch.bfloat16), image_grid_thw=grid)
    with torch.no_grad():
        want = ref(input_ids=input_ids, **kw).last_hidden_state

    header = sft_quant_format.read_header(str(path))
    with torch.device("meta"):
        model = Qwen3VLModel(cfg)
    model.language_model.norm = torch.nn.Identity()
    model.to(torch.bfloat16)
    model.requires_grad_(False)
    model.eval()
    rot = model.language_model.rotary_emb
    inv, scaling = Qwen3VLTextRotaryEmbedding.compute_default_rope_parameters(cfg.text_config, device="cpu")
    rot.inv_freq, rot.original_inv_freq, rot.attention_scaling = inv.float(), inv.float().clone(), scaling
    vrot = model.visual.rotary_pos_emb
    vrot.inv_freq = 1.0 / (10000.0 ** (torch.arange(0, vrot.dim, 2, dtype=torch.float32) / vrot.dim))
    embed_plan, layer_plans, visual_plan = te_read_plan(header.tensors, 3)
    dev = torch.device("cpu")
    stats = {"bytes": 0, "read_seconds": 0.0, "layers": 0}
    load_mapped_to(model.language_model.embed_tokens, str(path), header, embed_plan, dev)
    load_mapped_to(model.visual, str(path), header, visual_plan, dev)
    move_module_tree(model.language_model.rotary_emb, dev, skip=set())
    handles = install_layer_read_hooks(model.language_model.layers, layer_plans, str(path), header, dev, stats)
    with torch.no_grad():
        got = model(input_ids=input_ids, **kw).last_hidden_state
    for h in handles:
        h.remove()
    assert torch.equal(got, want)
    assert stats["layers"] == 3
    drop_to_meta(model.visual)
    assert not [n for n, t in model.visual.named_parameters() if not t.is_meta]


# --------------------------------------------------------------------------- #
# the text side's attention (G) and the early drop of the embedding table and
# the vision tower (J), 台帳 §1-116
# --------------------------------------------------------------------------- #


def test_blocked_causal_attention_matches_causal_sdpa(monkeypatch):
    """Query blocks against the full key / value with absolute-position mask slices equal
    plain causal GQA attention (close, not bit for bit: the CPU fp32 math adds in a
    block-dependent order; bit equality is the GPU gate's job). A tail shorter than
    ``min_tail`` is folded into the block before it (block heights recorded per call)."""
    import torch.nn.functional as F

    from engine_h3.text_encoder import blocked_causal_attention

    original = F.scaled_dot_product_attention
    heights = []

    def recording_sdpa(query, *args, **kwargs):
        heights.append(query.shape[2])
        return original(query, *args, **kwargs)

    torch.manual_seed(0)
    cases = (
        (1100, {}, [512, 512, 76]),                    # ragged, 76 >= 64: not folded
        (1087, {}, [512, 575]),                        # tail 63 folded
        (1088, {}, [512, 512, 64]),                    # tail 64 not folded
        (5, {"block": 2, "min_tail": 0}, [2, 2, 1]),   # the mask alignment on a tiny shape
    )
    for n, kw, want_heights in cases:
        q = torch.randn(1, 4, n, 16)
        k = torch.randn(1, 2, n, 16)
        v = torch.randn(1, 2, n, 16)
        want = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True, scale=0.25)
        heights.clear()
        monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", recording_sdpa)
        got = blocked_causal_attention(q, k, v, scale=0.25, **kw)
        monkeypatch.undo()
        assert heights == want_heights
        assert got.shape == want.shape
        assert torch.allclose(got, want, rtol=1e-5, atol=1e-6)


def test_h3_text_attention_contract():
    """A mask reaching the function stops it; otherwise (B, N, H, D) contiguous and None."""
    from types import SimpleNamespace

    from engine_h3.text_encoder import h3_text_attention

    module = SimpleNamespace(scaling=0.25)
    q = torch.randn(1, 4, 6, 16)
    k = torch.randn(1, 2, 6, 16)
    v = torch.randn(1, 2, 6, 16)
    with pytest.raises(RuntimeError, match="attention_mask"):
        h3_text_attention(module, q, k, v, torch.ones(1, 1, 6, 6, dtype=torch.bool))
    out, weights = h3_text_attention(module, q, k, v, None)
    assert tuple(out.shape) == (1, 6, 4, 16) and out.is_contiguous()
    assert weights is None


def test_build_config_registers_the_text_attention():
    """build_config registers the function and names it on the text side only; the
    real (meta) Qwen3VLModel builds with it (an unregistered name would raise)."""
    pytest.importorskip("transformers")
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    from engine_h3.text_encoder import H3_TEXT_ATTENTION, build_config

    config = build_config()
    assert H3_TEXT_ATTENTION in ALL_ATTENTION_FUNCTIONS
    assert config.text_config._attn_implementation == H3_TEXT_ATTENTION
    assert config.vision_config._attn_implementation == "sdpa"
    with torch.device("meta"):
        model = Qwen3VLModel(config)
    assert model.language_model.layers[0].self_attn.config._attn_implementation == H3_TEXT_ATTENTION
    assert model.visual.config._attn_implementation == "sdpa"


def test_te_drop_after_forward_keeps_the_hidden_state(tmp_path):
    """J: with the drop hooks on the embedding table and the vision tower, both are on
    meta before layer 0 runs, and the hidden state equals the fully loaded model's
    bit for bit (both on the registered text attention; G vs sdpa is the test above)."""
    pytest.importorskip("transformers")
    import sft_quant_format
    from safetensors.torch import save_file
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLTextRotaryEmbedding

    from engine_h3.sft_load import load_mapped_to, move_module_tree
    from engine_h3.text_encoder import (
        H3_TEXT_ATTENTION, build_config, install_drop_after_forward_hooks, install_layer_read_hooks, te_read_plan,
    )

    build_config()   # registers H3_TEXT_ATTENTION (the product's registration)
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
    # Before either model is built (they share cfg): the reference must run the same attention.
    cfg.text_config._attn_implementation = H3_TEXT_ATTENTION
    torch.manual_seed(0)
    ref = Qwen3VLModel(cfg).to(torch.bfloat16).eval()
    ref.language_model.norm = torch.nn.Identity()
    ref.visual.rotary_pos_emb.inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))
    r_inv, _ = Qwen3VLTextRotaryEmbedding.compute_default_rope_parameters(cfg.text_config, device="cpu")
    ref.language_model.rotary_emb.inv_freq = r_inv.float()
    ref.language_model.rotary_emb.original_inv_freq = r_inv.float().clone()
    with torch.no_grad():
        for prm in ref.parameters():
            prm.normal_(0.0, 0.05)
    names = {}
    for k, v in ref.state_dict().items():
        if k.startswith("language_model."):
            names["model." + k[len("language_model."):]] = v.contiguous()
        else:
            names[k] = v.contiguous()
    path = tmp_path / "te.safetensors"
    save_file(names, str(path))
    grid = torch.tensor([[1, 4, 4]])
    pixels = torch.randn(16, 3 * 2 * 16 * 16)
    ids = [5, 17, 93, 95, 95, 95, 95, 94, 3, 41, 8]
    input_ids = torch.tensor([ids])
    kw = dict(attention_mask=torch.ones_like(input_ids), use_cache=False,
              mm_token_type_ids=torch.tensor([[1 if i == 95 else 0 for i in ids]]),
              pixel_values=pixels.to(torch.bfloat16), image_grid_thw=grid)
    with torch.no_grad():
        want = ref(input_ids=input_ids, **kw).last_hidden_state

    header = sft_quant_format.read_header(str(path))
    with torch.device("meta"):
        model = Qwen3VLModel(cfg)
    model.language_model.norm = torch.nn.Identity()
    model.to(torch.bfloat16)
    model.requires_grad_(False)
    model.eval()
    assert model.language_model.layers[0].self_attn.config._attn_implementation == H3_TEXT_ATTENTION
    rot = model.language_model.rotary_emb
    inv, scaling = Qwen3VLTextRotaryEmbedding.compute_default_rope_parameters(cfg.text_config, device="cpu")
    rot.inv_freq, rot.original_inv_freq, rot.attention_scaling = inv.float(), inv.float().clone(), scaling
    vrot = model.visual.rotary_pos_emb
    vrot.inv_freq = 1.0 / (10000.0 ** (torch.arange(0, vrot.dim, 2, dtype=torch.float32) / vrot.dim))
    embed_plan, layer_plans, visual_plan = te_read_plan(header.tensors, 3)
    dev = torch.device("cpu")
    stats = {"bytes": 0, "read_seconds": 0.0, "layers": 0}
    lm = model.language_model
    load_mapped_to(lm.embed_tokens, str(path), header, embed_plan, dev)
    load_mapped_to(model.visual, str(path), header, visual_plan, dev)
    move_module_tree(lm.rotary_emb, dev, skip=set())
    seen = {}

    def before_layer0(module, args):
        tensors = [*lm.embed_tokens.parameters(), *lm.embed_tokens.buffers(),
                   *model.visual.parameters(), *model.visual.buffers()]
        seen["all_meta"] = all(t.is_meta for t in tensors)

    handles = [lm.layers[0].register_forward_pre_hook(before_layer0)]
    handles += install_drop_after_forward_hooks([lm.embed_tokens, model.visual])
    handles += install_layer_read_hooks(lm.layers, layer_plans, str(path), header, dev, stats)
    with torch.no_grad():
        got = model(input_ids=input_ids, **kw).last_hidden_state
    for h in handles:
        h.remove()
    assert seen == {"all_meta": True}
    assert stats["layers"] == 3
    assert torch.equal(got, want)


# --------------------------------------------------------------------------- #
# a tiny DiT streamed from a synthetic file
# --------------------------------------------------------------------------- #

_ST_DTYPES = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16", torch.int8: "I8", torch.uint8: "U8"}


def _tiny_dit_tensors(layers: int, refiner: int, grid: int, rank: int, heads: int = HEADS) -> dict:
    g = torch.Generator().manual_seed(0)
    inner = heads * HD

    def rnd(*shape, dtype=torch.bfloat16):
        if dtype == torch.int8:
            return torch.randint(-127, 128, shape, generator=g, dtype=torch.int8)
        return torch.randn(shape, generator=g).to(dtype)

    mark = torch.frombuffer(bytearray(json.dumps(
        {"format": "int8_tensorwise", "convrot": False}).encode()), dtype=torch.uint8)
    t = {
        "video_patch_proj.weight": rnd(HID, LAT * 4, dtype=torch.float32),
        "video_patch_proj.bias": rnd(HID, dtype=torch.float32),
        "audio_patch_proj.weight": rnd(HID, AUD, dtype=torch.float32),
        "audio_patch_proj.bias": rnd(HID, dtype=torch.float32),
        "condition_proj.weight": rnd(HID, TEXT),
        "condition_proj.bias": rnd(HID),
        "token_refiner.final_norm.weight": rnd(HID),
        "final_layer.norm.weight": rnd(HID),
        "final_layer.adaln_proj.linear.weight": rnd(2 * HID, rank, dtype=torch.float16),
        "final_layer.adaln_proj.linear.bias": rnd(2 * HID, dtype=torch.float16),
        "final_layer.video_out.weight": rnd(LAT * 4, HID, dtype=torch.float32),
        "final_layer.video_out.bias": rnd(LAT * 4, dtype=torch.float32),
        "final_layer.audio_out.weight": rnd(AUD, HID, dtype=torch.float32),
        "final_layer.audio_out.bias": rnd(AUD, dtype=torch.float32),
        "rope.inv_freq": 1.0 / (10000 ** (torch.arange(16, dtype=torch.float32) / 16)),
        "adaln_t_table": rnd(grid, rank, dtype=torch.float32),
    }
    lin = {"attn.qkv_proj": (3 * inner, HID), "attn.out_proj": (HID, inner), "mlp.fc1": (2 * FFN, HID), "mlp.fc2": (HID, FFN)}
    for i in range(layers):
        h = f"blocks.{i}."
        for n, shp in lin.items():
            t[f"{h}{n}.weight"] = rnd(*shp, dtype=torch.int8)
            t[f"{h}{n}.weight_scale"] = rnd(shp[0], 1, dtype=torch.float32).abs() / 100
            t[f"{h}{n}.comfy_quant"] = mark.clone()
        for leaf, n in (("attn.q_norm.weight", HD), ("attn.k_norm.weight", HD), ("norm1.weight", HID), ("norm2.weight", HID)):
            t[h + leaf] = rnd(n)
        t[f"{h}adaln_proj.linear.weight"] = rnd(18 * HID, rank, dtype=torch.float16)
        t[f"{h}adaln_proj.linear.bias"] = rnd(18 * HID, dtype=torch.float16)
    for i in range(refiner):
        h = f"token_refiner.blocks.{i}."
        for n, shp in lin.items():
            t[f"{h}{n}.weight"] = rnd(*shp)
        for leaf, n in (("attn.q_norm.weight", HD), ("attn.k_norm.weight", HD), ("norm1.weight", HID), ("norm2.weight", HID)):
            t[h + leaf] = rnd(n)
    return t


def _write_sft(path: Path, tensors: dict, prefix: str = "") -> None:
    doc, blobs, off = {}, [], 0
    for name, ten in tensors.items():
        data = ten.contiguous().view(torch.uint8).numpy().tobytes() if ten.numel() else b""
        doc[prefix + name] = {"dtype": _ST_DTYPES[ten.dtype], "shape": list(ten.shape), "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    head = json.dumps(doc).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for b in blobs:
            f.write(b)


def test_tiny_dit_streams_in_with_no_meta_left(tmp_path):
    from engine_h3.dit import H3DiT
    from engine_h3.pipeline_h3 import dit_key_map_for
    from engine_h3.quant_linear import H3Int8Linear, swap_in_int8_linears
    from engine_h3.sft_load import load_streaming, meta_tensor_names

    tensors = _tiny_dit_tensors(layers=3, refiner=2, grid=1025, rank=8)
    tensors["some_future_head.weight"] = torch.zeros(2)  # unknown: must be dropped, not fatal
    path = tmp_path / "tiny.safetensors"
    _write_sft(path, tensors, prefix="model.diffusion_model.")
    header, cfg = h3_format.inspect_h3_dit(str(path))
    assert cfg.prefix == "model.diffusion_model." and cfg.unknown == ("some_future_head.weight",)
    with torch.device("meta"):
        dit = H3DiT(cfg)
        swapped = swap_in_int8_linears(dit, cfg.quant, bias_dtype_of=lambda n: torch.bfloat16)
    assert swapped == 12
    report = load_streaming(dit, str(path), header, key_map=dit_key_map_for(cfg))
    assert meta_tensor_names(dit) == []
    qkv = dit.blocks[0].attn.qkv_proj
    assert isinstance(qkv, H3Int8Linear) and qkv.weight.dtype == torch.int8
    assert qkv.weight_scale.dtype == torch.float32 and tuple(qkv.weight_scale.shape) == (3 * INNER, 1)
    assert qkv.spec == h3_format.QuantSpec("int8_tensorwise", False, 256)
    assert torch.equal(qkv.weight, tensors["blocks.0.attn.qkv_proj.weight"])
    assert torch.equal(dit.adaln_t_table, tensors["adaln_t_table"])
    assert torch.equal(dit.rope_inv_freq, tensors["rope.inv_freq"])
    # the file's dtype is kept per tensor (the F16 AdaLN projections stay F16)
    assert dit.blocks[1].adaln_proj.linear.weight.dtype == torch.float16
    sd_keys = set(dit.state_dict())
    file_keys = {n for n in tensors if not n.endswith(".comfy_quant") and n != "some_future_head.weight"}
    assert sd_keys == file_keys
    assert "model.diffusion_model.some_future_head.weight" in report.skipped


def test_load_streaming_refuses_a_missing_tensor(tmp_path):
    from engine_h3.dit import H3DiT
    from engine_h3.pipeline_h3 import dit_key_map_for
    from engine_h3.quant_linear import swap_in_int8_linears
    from engine_h3.sft_load import load_streaming

    tensors = _tiny_dit_tensors(layers=2, refiner=2, grid=1025, rank=8)
    path = tmp_path / "tiny.safetensors"
    _write_sft(path, tensors)
    header, cfg = h3_format.inspect_h3_dit(str(path))
    with torch.device("meta"):
        dit = H3DiT(cfg)
        swap_in_int8_linears(dit, cfg.quant, bias_dtype_of=lambda n: torch.bfloat16)

    def drop_one(name):
        return None if name == "final_layer.audio_out.bias" else dit_key_map_for(cfg)(name)

    with pytest.raises(RuntimeError, match="final_layer.audio_out.bias"):
        load_streaming(dit, str(path), header, key_map=drop_one)


# --------------------------------------------------------------------------- #
# H3Int8Linear vs the dequantized bf16 linear (GPU, opt-in)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not (torch.cuda.is_available() and os.environ.get("H3_GPU_TESTS") == "1"),
    reason="GPU test: set H3_GPU_TESTS=1 (and have the supervisor's GPU permission)",
)
@pytest.mark.parametrize("convrot", [True, False])
def test_int8_linear_matches_dequantized_bf16(convrot):
    import torch.nn.functional as F

    from engine.sft_quant.dequant import dequantize
    from engine_h3.quant_linear import H3Int8Linear, require_cuda_int8

    require_cuda_int8()
    dev = torch.device("cuda")
    n, k, m = 512, 1024, 64
    g = torch.Generator().manual_seed(1)
    w = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8)
    s = (torch.rand(n, 1, generator=g) * 0.01 + 1e-3).float()
    b = torch.randn(n, generator=g).to(torch.bfloat16)
    spec = h3_format.QuantSpec("int8_tensorwise", convrot, 256)
    lin = H3Int8Linear(k, n, bias=True, spec=spec)
    lin.weight = torch.nn.Parameter(w.to(dev), requires_grad=False)
    lin.weight_scale = s.to(dev)
    lin.bias = torch.nn.Parameter(b.to(dev), requires_grad=False)
    x = torch.randn(m, k, generator=g).to(dev, torch.bfloat16)
    with torch.no_grad():
        got = lin(x).float()
        wd = dequantize("int8_convrot" if convrot else "int8", w.to(dev), {"weight_scale": s.to(dev)}, torch.bfloat16)
        ref = F.linear(x, wd, b.to(dev)).float()
    rel = (got - ref).norm() / ref.norm()
    assert float(rel) < 3e-2, f"relative error {float(rel):.4f}"  # W8A8 vs W8A16


# --------------------------------------------------------------------------- #
# source rules and the worker protocol
# --------------------------------------------------------------------------- #

def test_no_inference_mode_under_engine_h3():
    offenders = []
    for p in sorted((ROOT / "engine_h3").glob("**/*.py")):
        for no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"inference_mode", line):
                offenders.append(f"{p.name}:{no}: {line.strip()}")
    assert offenders == []


def test_worker_module_scope_is_torch_free():
    src = (ROOT / "engine_h3" / "worker.py").read_text(encoding="utf-8")
    top = [ln for ln in src.splitlines() if re.match(r"^(?:import|from)\s+torch\b", ln)]
    assert top == []


def _drive(monkeypatch, lines):
    from engine_h3 import worker

    events: list[dict] = []

    def _shutdown():
        raise SystemExit(0)

    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    monkeypatch.setattr(worker, "_log", lambda msg: None)
    monkeypatch.setattr(worker, "_do_load", lambda msg: None)
    monkeypatch.setattr(worker, "_shutdown", _shutdown)
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO("".join(json.dumps(x) + "\n" for x in lines)))
    with pytest.raises(SystemExit) as exc:
        worker.main()
    return events, exc.value.code


def test_worker_unknown_op_after_load_is_answered(monkeypatch):
    events, code = _drive(monkeypatch, [{"op": "load"}, {"op": "frobnicate"}, {"op": "generate_chain"}])
    assert code == 0
    assert [e["event"] for e in events] == ["error", "error"]
    assert "frobnicate" in events[0]["detail"] and "generate_chain" in events[1]["detail"]


def test_worker_unknown_op_before_load_is_silent(monkeypatch):
    events, code = _drive(monkeypatch, [{"op": "frobnicate"}, {"op": "load"}])
    assert code == 0
    assert events == []


def test_worker_generate_defaults(monkeypatch):
    from engine_h3 import worker

    req = worker._generate_request({
        "prompt": "p", "seed": 1, "width": 640, "height": 384, "num_frames": 124, "output_path": "o.mp4",
    })
    assert (req.num_inference_steps, req.shift_video, req.shift_audio, req.frame_rate) == (20, 12.0, 3.0, 24)


def test_request_validation():
    from engine_h3.pipeline_h3 import H3GenerateRequest, validate_request

    ok = dict(prompt="p", seed=1, width=640, height=384, num_frames=124, output_path="o.mp4")
    validate_request(H3GenerateRequest(**ok))
    validate_request(H3GenerateRequest(**{**ok, "num_frames": 22}))  # n = 1: the shortest the VAE decodes
    for bad in (dict(width=650), dict(height=0), dict(num_frames=125), dict(frame_rate=25),
                dict(num_inference_steps=0), dict(shift_video=0.0), dict(prompt="  "),
                dict(num_frames=5)):  # 17n+5 with n = 0: refused before the TE / DiT run
        with pytest.raises(ValueError):
            validate_request(H3GenerateRequest(**{**ok, **bad}))


def test_load_options_treat_null_as_default(monkeypatch):
    from engine_h3 import worker

    msg = {"blocks_on_gpu": None, "attention_backend": None,
           "block_swap_prefetch": None, "deterministic": None}
    assert worker._value_or(msg, "blocks_on_gpu", 8, int) == 8
    assert worker._value_or(msg, "attention_backend", "sdpa", str) == "sdpa"
    assert worker._value_or({"blocks_on_gpu": 4}, "blocks_on_gpu", 8, int) == 4
    monkeypatch.setattr("engine.transformer.sage_attention_service.probe_sage", lambda: True)
    assert worker._resolve_attention(msg) == ("sdpa", False)
    assert worker._resolve_attention({"attention_backend": "sage"}) == ("sage", False)
    with pytest.raises(RuntimeError):
        worker._resolve_attention({"attention_backend": "flash"})


def test_attention_backends_have_one_source():
    from engine_h3 import dit, pipeline_h3, worker

    assert C.ATTENTION_BACKENDS == ("sdpa", "sage")
    assert pipeline_h3.ATTENTION_BACKENDS is C.ATTENTION_BACKENDS
    assert worker.ATTENTION_BACKENDS is C.ATTENTION_BACKENDS
    backends = getattr(dit, "ATTENTION_BACKENDS", getattr(dit, "_ATTENTION_BACKENDS", None))
    assert tuple(backends) == C.ATTENTION_BACKENDS


# --------------------------------------------------------------------------- #
# golden key sets of the protocol (the second-stage adapter builds against these)
# --------------------------------------------------------------------------- #

#: Stage 2 (2a, 台帳 §1-98 / plan §3.2): ``te_layers_on_gpu`` is gone -- the text
#: encoder is read layer by layer straight from the file onto the GPU, so a
#: CPU-resident layer window no longer exists to size.
GOLDEN_LOAD_KEYS = [
    "op", "transformer_path", "text_encoder_path", "video_vae_path", "audio_vae_path", "tokenizer_dir",
    "blocks_on_gpu", "attention_backend", "block_swap_prefetch", "deterministic",
]
#: Stage 2 (2a, INTERFACE2 §5.1 / plan §3.2): the I2V image, keep_resident and the
#: per-job attention / prefetch were added at the end.
GOLDEN_GENERATE_KEYS = [
    "op", "prompt", "seed", "width", "height", "num_frames", "frame_rate",
    "num_inference_steps", "shift_video", "shift_audio", "output_path",
    "images", "keep_resident", "attention_backend", "block_swap_prefetch",
    "vram16_mode",   # VRAM 16GB mode (台帳 §1-101)
    "loras",         # LoRA (台帳 §1-104), at the end; [] = none, always sent
]
#: Keys the worker accepts but a message carries only when used (appended after
#: ``GOLDEN_GENERATE_KEYS`` in this order): A2V's ``source_audio`` (台帳 §1-103).
GOLDEN_GENERATE_OPTIONAL_KEYS = ["source_audio"]
#: Ref2VA's ``references`` (台帳 §1-107), sent only when used, after the keys above.
GOLDEN_GENERATE_REFERENCE_KEYS = ["references"]
GOLDEN_DONE_KEYS = [
    "seed_used", "sampler", "num_inference_steps", "shift_video", "shift_audio",
    "peak_vram_mb", "peak_vram_reserved_mb", "rss_peak_gib", "commit_gib", "commit_peak_gib", "seconds",
    "num_frames", "frame_rate", "width", "height", "size_bytes", "attention_used",
    "block_swap_prefetch_used", "keep_resident_used", "vram16_mode_used",   # VRAM 16GB mode (台帳 §1-101)
    "images_used", "h3",   # 2 keys added in stage 2 (2a)
]
GOLDEN_H3_KEYS = ["phases", "quant_backends"]


def _selftest_args():
    from engine_h3 import worker

    return worker._selftest_parser().parse_args([
        "--transformer", "t", "--text-encoder", "te", "--video-vae", "vv", "--audio-vae", "av",
        "--tokenizer-dir", "tok", "--output", "o.mp4",
    ])


def test_golden_load_message_keys():
    from engine_h3 import pipeline_h3, worker

    msg = worker._selftest_load_message(_selftest_args())
    assert list(msg) == GOLDEN_LOAD_KEYS
    assert not hasattr(pipeline_h3.H3LoadOptions(), "te_layers_on_gpu")
    # every path field the worker checks is in the message
    assert set(worker._LOAD_FILE_FIELDS) | set(worker._LOAD_DIR_FIELDS) <= set(msg)


def test_golden_generate_message_keys():
    from engine_h3 import worker

    msg = worker._selftest_generate_message(_selftest_args(), "o.mp4")
    assert list(msg) == GOLDEN_GENERATE_KEYS            # a job without the optional keys: as before
    assert worker._GENERATE_FIELDS == frozenset(
        GOLDEN_GENERATE_KEYS + GOLDEN_GENERATE_OPTIONAL_KEYS + GOLDEN_GENERATE_REFERENCE_KEYS)
    args = _selftest_args()
    args.source_audio, args.audio_strength = "a.wav", 0.5
    msg = worker._selftest_generate_message(args, "o.mp4")
    assert list(msg) == GOLDEN_GENERATE_KEYS + GOLDEN_GENERATE_OPTIONAL_KEYS
    assert msg["source_audio"] == {"path": "a.wav", "strength": 0.5}


def test_golden_done_keys(monkeypatch, tmp_path):
    """The ``done`` payload's top-level keys, from a generate whose three stages are stubbed."""
    from engine_h3 import pipeline_h3, worker

    pipe = pipeline_h3.H3Pipeline(
        pipeline_h3.H3Paths("t", "te", "vv", "av", "tok"), pipeline_h3.H3LoadOptions(),
        device="cpu", progress=lambda *a: None, log=lambda m: None,
    )
    pipe._dit_cfg = object()  # "loaded"
    monkeypatch.setattr(pipe, "_encode", lambda prompt, phases, **kw: (torch.zeros(3, 8), torch.ones(3, dtype=torch.long)))
    monkeypatch.setattr(pipe, "_denoise", lambda req, h, t, phases, entry, **kw: (torch.zeros(1), torch.zeros(1), "on"))
    monkeypatch.setattr(pipe, "_decode", lambda req, v, a, entry, **kw: 1234)
    req = pipeline_h3.H3GenerateRequest(prompt="p", seed=1, width=640, height=384, num_frames=124, output_path="o.mp4")
    done = pipe.generate(req)
    assert list(done) == GOLDEN_DONE_KEYS
    assert list(done["h3"]) == GOLDEN_H3_KEYS
    assert set(done["h3"]["phases"]) == {"encode", "denoise", "decode"}
    assert (done["keep_resident_used"], done["images_used"]) == ("off", 0)
    assert done["vram16_mode_used"] == "off"   # the worker's default

    events = []
    monkeypatch.setattr(worker, "_PIPE", pipe)
    monkeypatch.setattr(worker, "_ATTENTION", ("sage", True))
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    monkeypatch.setattr(worker, "_log", lambda m: None)
    monkeypatch.setattr("engine.transformer.sage_attention_service.probe_sage", lambda: False)
    msg = worker._selftest_generate_message(_selftest_args(), "o.mp4")
    msg["attention_backend"] = "sage"   # per job; no SageAttention here -> degraded
    worker._do_generate(msg)
    assert events[0]["event"] == "done"
    assert list(events[0])[1:] == GOLDEN_DONE_KEYS
    assert events[0]["attention_used"] == "sage->sdpa"


def test_generate_frees_cuda_cache_after_a_failure(monkeypatch):
    from engine_h3 import pipeline_h3

    calls = []
    monkeypatch.setattr(pipeline_h3, "_free_cuda", lambda: calls.append(1))
    pipe = pipeline_h3.H3Pipeline(
        pipeline_h3.H3Paths("t", "te", "vv", "av", "tok"), pipeline_h3.H3LoadOptions(),
        device="cpu", progress=lambda *a: None, log=lambda m: None,
    )
    pipe._dit_cfg = object()

    def boom(prompt, phases, **kw):
        raise RuntimeError("encode failed")

    monkeypatch.setattr(pipe, "_encode", boom)
    req = pipeline_h3.H3GenerateRequest(prompt="p", seed=1, width=640, height=384, num_frames=124, output_path="o.mp4")
    with pytest.raises(RuntimeError, match="encode failed"):
        pipe.generate(req)
    assert calls  # the finally ran


def test_block_swap_service_is_resident_across_jobs(tmp_path):
    """One BlockSwapService per pipeline (engine25's shape): the pinned pool and the
    transfer stream are not rebuilt per job. CPU device: prefetch falls back."""
    from engine_h3.dit import H3DiT
    from engine_h3.pipeline_h3 import H3LoadOptions, H3Paths, H3Pipeline, dit_key_map_for
    from engine_h3.quant_linear import swap_in_int8_linears
    from engine_h3.sft_load import load_streaming

    path = tmp_path / "tiny.safetensors"
    _write_sft(path, _tiny_dit_tensors(layers=3, refiner=2, grid=1025, rank=8))
    header, cfg = h3_format.inspect_h3_dit(str(path))
    pipe = H3Pipeline(H3Paths(str(path), "te", "vv", "av", "tok"), H3LoadOptions(blocks_on_gpu=1),
                      device="cpu", progress=lambda *a: None, log=lambda m: None)
    services = []
    for _ in range(2):
        with torch.device("meta"):
            dit = H3DiT(cfg)
            swap_in_int8_linears(dit, cfg.quant, bias_dtype_of=lambda n: torch.bfloat16)
        load_streaming(dit, str(path), header, key_map=dit_key_map_for(cfg))
        service, used = pipe._place_dit(dit)
        assert used == "on->off"  # asked for, no CUDA
        services.append(service)
        service.teardown_prefetch()
        service.release_installed()
    assert services[0] is services[1] is pipe._swap_service


# --------------------------------------------------------------------------- #
# stage 2 (2a): I2V plumbing, keep_resident, per-job settings (INTERFACE2 §2-§8)
# --------------------------------------------------------------------------- #

def test_condition_rows_time_is_max_of_video_t_and_0999():
    from engine_h3.dit import H3SeqLayout
    from engine_h3.pipeline_h3 import time_rows

    layout = H3SeqLayout.make(text_rows=5, cond_rows=4, audio_rows=4, video_rows=6)
    t_unique, idx = time_rows(layout, 0.0, 0.0)            # step 0: both sigmas 1
    assert torch.allclose(t_unique[idx[layout.cond]], torch.tensor(0.999))
    assert int(t_unique.numel()) == 2
    t_unique, idx = time_rows(layout, 0.9995, 0.5)
    assert torch.allclose(t_unique[idx[layout.cond]], torch.tensor(0.9995))
    assert torch.allclose(t_unique[idx[layout.text]], torch.tensor(0.9995))


def test_request_validation_images():
    from engine_h3.pipeline_h3 import H3GenerateRequest, H3ImageInput, validate_request

    ok = dict(prompt="p", seed=1, width=640, height=384, num_frames=124, output_path="o.mp4")
    for good in ((H3ImageInput("a.png", 0),), (H3ImageInput("a.png", 0), H3ImageInput("b.png", 123)),
                 (H3ImageInput("a.png", 60),), (H3ImageInput("b.png", 123), H3ImageInput("a.png", 0)),
                 tuple(H3ImageInput(f"{k}.png", k * 12) for k in range(10))):
        validate_request(H3GenerateRequest(**ok, images=good))
    for bad, words in (((H3ImageInput("a.png", 0), H3ImageInput("b.png", 0)), "重複"),
                       ((H3ImageInput("a.png", 124),), "0〜123"),
                       ((H3ImageInput("a.png", -1),), "0〜123"),
                       ((H3ImageInput("", 60),), "パスが空")):
        with pytest.raises(ValueError, match=words):
            validate_request(H3GenerateRequest(**ok, images=bad))
    with pytest.raises(ValueError):
        validate_request(H3GenerateRequest(**ok, attention_backend="flash"))


def test_worker_images_and_per_job_keys():
    from engine_h3 import worker

    base = {"prompt": "p", "seed": 1, "width": 640, "height": 384, "num_frames": 124, "output_path": "o.mp4"}
    req = worker._generate_request(base)
    assert (req.images, req.keep_resident, req.attention_backend, req.block_swap_prefetch) == ((), False, None, None)
    req = worker._generate_request({**base, "images": [{"path": "a.png", "frame_idx": 0, "strength": 0.8}],
                                    "keep_resident": True, "attention_backend": "sage", "block_swap_prefetch": False})
    assert [(i.path, i.frame_idx) for i in req.images] == [("a.png", 0)]
    assert (req.keep_resident, req.attention_backend, req.block_swap_prefetch) == (True, "sage", False)
    assert worker._images({"images": [{"path": "a.png", "strength": 1}]})[1] == ["strength"]
    for bad in ({"images": "a.png"}, {"images": [{"frame_idx": 0}]}):
        with pytest.raises(ValueError):
            worker._generate_request({**base, **bad})
    req = worker._generate_request({**base, "images": [{"path": "b.png", "frame_idx": 123},
                                                       {"path": "a.png", "frame_idx": 0}]})
    assert [(i.path, i.frame_idx) for i in req.images] == [("b.png", 123), ("a.png", 0)]


def test_worker_selftest_keyframe_option(capsys):
    from engine_h3 import worker

    def args(*extra):
        return worker._selftest_parser().parse_args([
            "--transformer", "t", "--text-encoder", "te", "--video-vae", "vv", "--audio-vae", "av",
            "--tokenizer-dir", "tok", "--output", "o.mp4", *extra,
        ])

    assert worker._selftest_generate_message(args(), "o.mp4")["images"] == []
    # --image is frame 0, as before (the fingerprint commands do not change).
    assert worker._selftest_generate_message(args("--image", "a.png"), "o.mp4")["images"] == [
        {"path": "a.png", "frame_idx": 0}
    ]
    msg = worker._selftest_generate_message(
        args("--image", "a.png", "--keyframe", "123=b.png", "--keyframe", "60=c:/d.png"), "o.mp4")
    assert msg["images"] == [
        {"path": "a.png", "frame_idx": 0}, {"path": "b.png", "frame_idx": 123}, {"path": "c:/d.png", "frame_idx": 60},
    ]
    with pytest.raises(ValueError, match="重複"):
        worker._selftest_images(args("--image", "a.png", "--keyframe", "0=b.png"))
    with pytest.raises(ValueError, match="重複"):
        worker._selftest_images(args("--keyframe", "5=a.png", "--keyframe", "5=b.png"))
    for bad in ("a.png", "x=a.png", "5=", "1.5=a.png"):
        with pytest.raises(ValueError, match="FRAME=PATH"):
            worker._selftest_images(args("--keyframe", bad))
    # The CLI entry reports it on stderr with a non-zero code (no traceback).
    code = worker._selftest_generate([
        "--transformer", "t", "--text-encoder", "te", "--video-vae", "vv", "--audio-vae", "av",
        "--tokenizer-dir", "tok", "--output", "o.mp4", "--keyframe", "a.png",
    ])
    assert code != 0
    assert "FRAME=PATH" in capsys.readouterr().err


def test_condition_noise_is_one_draw_shared_by_every_block():
    """The condition rows' noise: K = 1 is bit for bit the first-frame-only
    engine's formula (the third draw of the job's generator); with K = 3 every
    block gets the same noise."""
    from engine_h3.constants import KEYFRAME_NOISE_AUG
    from engine_h3.pipeline_h3 import noisy_condition

    grid = 6

    def generator_after_video_and_audio():
        gen = torch.Generator(device="cpu").manual_seed(42)
        torch.randn((1, 4, 7, 2, 2), generator=gen, dtype=torch.float32)
        torch.randn((2, 3, 5), generator=gen, dtype=torch.float32)
        return gen

    clean1 = torch.randn(grid, 96, generator=torch.Generator().manual_seed(1))
    gen = generator_after_video_and_audio()
    old_noise = torch.randn(tuple(clean1.shape), generator=gen, dtype=torch.float32)
    old = KEYFRAME_NOISE_AUG * clean1.to(torch.float32) + (1.0 - KEYFRAME_NOISE_AUG) * old_noise
    assert torch.equal(noisy_condition(clean1, grid, generator_after_video_and_audio()), old)

    clean3 = torch.randn(3 * grid, 96, generator=torch.Generator().manual_seed(2))
    out = noisy_condition(clean3, grid, generator_after_video_and_audio())
    for k in range(3):
        rows = slice(k * grid, (k + 1) * grid)
        assert torch.equal(out[rows], noisy_condition(clean3[rows], grid, generator_after_video_and_audio()))
    zeros = noisy_condition(torch.zeros(3 * grid, 96), grid, generator_after_video_and_audio())
    assert torch.equal(zeros[:grid], zeros[grid:2 * grid]) and torch.equal(zeros[:grid], zeros[2 * grid:])
    with pytest.raises(RuntimeError):
        noisy_condition(torch.zeros(grid + 1, 96), grid, generator_after_video_and_audio())


def _keyframe_job(monkeypatch, tmp_path, frames):
    """Run ``generate`` with key frames at ``frames`` (64 x 64, 124 frames) through the
    REAL ``_encode`` / ``_encode_image`` with a stand-in text encoder and VAE
    encoder; returns (what the TE was given, the encode progress, the denoise's
    keyword arguments, done). Each image is a flat grey of level 10 + frame, so a
    condition block tells which image it came from."""
    from PIL import Image

    from engine_h3 import pipeline_h3 as P
    from engine_h3 import text_encoder as TE
    from engine_h3 import video_vae as VV

    images = []
    for n in frames:
        path = tmp_path / f"k{n}.png"
        Image.new("RGB", (80, 64), (10 + n,) * 3).save(path)
        images.append(P.H3ImageInput(str(path), n))
    te_images, progress, denoise_kw = [], [], {}

    class FakeTE:
        def __init__(self, *a, **k):
            self.read_report = {}

        def load(self):
            return {}

        def encode(self, prompt, images=()):
            te_images.append([img.getpixel((0, 0))[0] for img in images])
            self.read_report = {"vision_tokens": 7 * len(images)}
            return torch.zeros(3, 8), torch.ones(3, dtype=torch.long)

        def release(self):
            pass

    class FakeEncoder:
        def encode(self, x, tiled=True):
            return torch.full((1, 24, 1, 4, 4), float(round(float(x.mean()) * 255)))

    monkeypatch.setattr(TE, "H3TextEncoder", FakeTE)
    monkeypatch.setattr(VV.H3VideoEncoder, "from_file", classmethod(lambda cls, *a, **k: FakeEncoder()))
    pipe = P.H3Pipeline(P.H3Paths("t", "te", "vv", "av", "tok"), P.H3LoadOptions(), device="cpu",
                        progress=lambda stage, i, total: progress.append((stage, i, total)), log=lambda m: None)
    pipe._dit_cfg = object()

    def denoise(req, h, t, phases, entry, **kw):
        denoise_kw.update(kw)
        return torch.zeros(1), torch.zeros(1), "on"

    monkeypatch.setattr(pipe, "_denoise", denoise)
    monkeypatch.setattr(pipe, "_decode", lambda req, v, a, entry, **kw: 1)
    req = P.H3GenerateRequest(prompt="p", seed=1, width=64, height=64, num_frames=124, output_path="o.mp4",
                              images=tuple(images))
    done = pipe.generate(req)
    return te_images, [p for p in progress if p[0] == "encode"], denoise_kw, done


def test_keyframes_sorted_and_only_the_ends_go_to_the_vision_tower(monkeypatch, tmp_path):
    te_images, progress, kw, done = _keyframe_job(monkeypatch, tmp_path, [123, 60, 0])
    assert te_images == [[10, 133]]                     # frame 0 then frame 123; not the middle one
    assert kw["cond_frame_idx"] == [0, 60, 123]
    rows = kw["cond_clean"]
    assert tuple(rows.shape) == (3 * 4, 96)             # 64 x 64: grid 2 x 2 per key frame
    assert [float(rows[k * 4, 0]) for k in range(3)] == [10.0, 70.0, 133.0]
    assert {total for _s, _i, total in progress} == {4} and progress[-1] == ("encode", 4, 4)
    assert done["images_used"] == 3
    image_encode = done["h3"]["phases"]["image_encode"]
    assert image_encode["source_size"] == [[80, 64]] * 3 and image_encode["vision_tokens"] == 14


def test_last_only_and_middle_only_key_frames(monkeypatch, tmp_path):
    te_images, progress, kw, _done = _keyframe_job(monkeypatch, tmp_path, [123])
    assert te_images == [[133]] and kw["cond_frame_idx"] == [123]   # the TE labels it <Picture 1>
    te_images, progress, kw, _done = _keyframe_job(monkeypatch, tmp_path, [60])
    assert te_images == [[]] and kw["cond_frame_idx"] == [60]       # no vision tower, as T2V
    # The stage counts 4 because the job has a key frame (never 3/3 then 4/4).
    assert {total for _s, _i, total in progress} == {4} and progress[-1] == ("encode", 4, 4)


def test_t2v_encode_counts_three_steps(monkeypatch, tmp_path):
    te_images, progress, kw, done = _keyframe_job(monkeypatch, tmp_path, [])
    assert te_images == [[]] and kw["cond_clean"] is None and kw["cond_frame_idx"] == []
    assert progress == [("encode", 0, 3), ("encode", 1, 3), ("encode", 2, 3), ("encode", 3, 3)]
    assert "image_encode" not in done["h3"]["phases"]


def test_canvas_image_cover_and_centre_crop(tmp_path):
    from PIL import Image

    from engine_h3.image_cond import canvas_to_tensor, load_canvas_image

    # 200 x 100 source onto a 64 x 64 canvas: scale = max(64/100, 64/200) = 0.64 ->
    # 128 x 64, then the centre 64 columns. A white band marks the source centre.
    src = Image.new("RGB", (200, 100), (0, 0, 0))
    for x in range(90, 110):
        for y in range(100):
            src.putpixel((x, y), (255, 255, 255))
    path = tmp_path / "src.png"
    src.save(path)
    img, size = load_canvas_image(str(path), 64, 64)
    assert img.size == (64, 64) and size == (200, 100)
    x = canvas_to_tensor(img)
    assert x.shape == (1, 3, 1, 64, 64) and x.dtype == torch.float32
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0
    centre = x[0, 0, 0, :, 28:36].mean()
    edge = x[0, 0, 0, :, :8].mean()
    assert centre > 0.8 and edge < 0.05       # cut from the centre, not stretched
    same, _ = load_canvas_image(str(path), 200, 100)   # already the canvas: untouched
    assert same.tobytes() == src.tobytes()
    with pytest.raises(ValueError):
        load_canvas_image(str(tmp_path / "missing.png"), 64, 64)
    (tmp_path / "bad.png").write_bytes(b"not an image")
    with pytest.raises(ValueError):
        load_canvas_image(str(tmp_path / "bad.png"), 64, 64)


def _tiny_pipe_and_dit(tmp_path, blocks_on_gpu=1):
    from engine_h3.dit import H3DiT
    from engine_h3.pipeline_h3 import H3LoadOptions, H3Paths, H3Pipeline, dit_key_map_for
    from engine_h3.quant_linear import swap_in_int8_linears
    from engine_h3.sft_load import load_streaming

    path = tmp_path / "tiny.safetensors"
    _write_sft(path, _tiny_dit_tensors(layers=3, refiner=2, grid=1025, rank=8))
    header, cfg = h3_format.inspect_h3_dit(str(path))
    pipe = H3Pipeline(H3Paths(str(path), "te", "vv", "av", "tok"), H3LoadOptions(blocks_on_gpu=blocks_on_gpu),
                      device="cpu", progress=lambda *a: None, log=lambda m: None)
    with torch.device("meta"):
        dit = H3DiT(cfg)
        swap_in_int8_linears(dit, cfg.quant, bias_dtype_of=lambda n: torch.bfloat16)
    load_streaming(dit, str(path), header, key_map=dit_key_map_for(cfg))
    return pipe, dit


def test_block_swap_wrappers_are_stripped_and_never_doubled(tmp_path):
    """Plan §3.2 / review M-1: a kept DiT goes back unwrapped, and installing again
    wraps each block exactly once. On the CPU every install falls back to the
    synchronous path (``on->off``), so this pins the synchronous double-wrap
    only; the prefetch path and the "nothing left on the GPU" check are shown
    on the GPU by gate 2a-G4 (32 tensors parked after a prefetch job, 46 after
    a synchronous one) -- review m-6."""
    from engine.transformer.block_swap_service import _BLOCK_SWAP_ATTR

    pipe, dit = _tiny_pipe_and_dit(tmp_path)
    for prefetch in (True, False, True, False):
        service, _used = pipe._place_dit(dit, prefetch=prefetch)
        for block in dit.blocks:
            original = getattr(block, _BLOCK_SWAP_ATTR)
            assert getattr(original, "__func__", None) is type(block).forward   # the class forward, not a wrapper
        service.teardown_prefetch()
        service.release_installed()
        pipe._park_dit(dit)
        for block in dit.blocks:
            assert not hasattr(block, _BLOCK_SWAP_ATTR)
            assert getattr(block.forward, "__func__", None) is type(block).forward
        assert all(t.device.type == "cpu" for t in dit.state_dict().values())


def _stubbed_pipe(monkeypatch, fail=False):
    from engine_h3 import pipeline_h3

    pipe = pipeline_h3.H3Pipeline(
        pipeline_h3.H3Paths("t", "te", "vv", "av", "tok"), pipeline_h3.H3LoadOptions(),
        device="cpu", progress=lambda *a: None, log=lambda m: None,
    )
    pipe._dit_cfg = object()

    def encode(prompt, phases, **kw):
        if fail:
            raise RuntimeError("encode failed")
        return torch.zeros(3, 8), torch.ones(3, dtype=torch.long)

    monkeypatch.setattr(pipe, "_encode", encode)
    monkeypatch.setattr(pipe, "_denoise", lambda req, h, t, phases, entry, **kw: (torch.zeros(1), torch.zeros(1), "on"))
    monkeypatch.setattr(pipe, "_decode", lambda req, v, a, entry, **kw: 1)
    return pipe


@pytest.mark.parametrize("keep, fail, kept_after", [(True, False, True), (False, False, False), (True, True, False)])
def test_resident_kept_only_while_the_key_is_true(monkeypatch, keep, fail, kept_after):
    """INTERFACE2 §8.1: kept after a keep_resident job; released by a job without
    the key; released at once by a failed job."""
    from engine_h3.pipeline_h3 import H3GenerateRequest

    pipe = _stubbed_pipe(monkeypatch, fail=fail)
    pipe._resident = {"decoder": torch.nn.Linear(1, 1), "audio_vae": torch.nn.Linear(1, 1)}
    req = H3GenerateRequest(prompt="p", seed=1, width=640, height=384, num_frames=124, output_path="o.mp4",
                            keep_resident=keep)
    if fail:
        with pytest.raises(RuntimeError):
            pipe.generate(req)
    else:
        done = pipe.generate(req)
        assert done["keep_resident_used"] == ("on" if keep else "off")
    assert (pipe._resident is not None) == kept_after


def test_attention_and_prefetch_per_job_fall_back_to_load_values():
    from engine_h3.pipeline_h3 import H3GenerateRequest, H3LoadOptions, H3Paths, H3Pipeline

    pipe = H3Pipeline(H3Paths("t", "te", "vv", "av", "tok"),
                      H3LoadOptions(attention_backend="sage", block_swap_prefetch=True),
                      device="cpu", progress=lambda *a: None, log=lambda m: None)
    base = dict(prompt="p", seed=1, width=640, height=384, num_frames=124, output_path="o.mp4")
    assert pipe._attention_for(H3GenerateRequest(**base)) == "sage"
    assert pipe._attention_for(H3GenerateRequest(**base, attention_backend="sdpa")) == "sdpa"
    assert pipe._prefetch_for(H3GenerateRequest(**base)) is True
    assert pipe._prefetch_for(H3GenerateRequest(**base, block_swap_prefetch=False)) is False


# --------------------------------------------------------------------------- #
# after the code review (REVIEW_code2a): M-1, m-4, m-7
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", ["missing_image", "duplicate_frames", "bad_frame_rate"])
def test_failed_check_or_image_releases_what_is_kept(monkeypatch, tmp_path, bad):
    """Review M-1: a job refused by the request check or the image load is a failed
    job too, and releases the kept models (INTERFACE2 §8.1) -- before the TE runs."""
    from engine_h3.pipeline_h3 import H3GenerateRequest, H3ImageInput

    pipe = _stubbed_pipe(monkeypatch)
    calls = []
    monkeypatch.setattr(pipe, "_encode", lambda *a, **k: calls.append("encode"))
    pipe._resident = {"decoder": torch.nn.Linear(1, 1), "audio_vae": torch.nn.Linear(1, 1)}
    base = dict(prompt="p", seed=1, width=640, height=384, num_frames=124, output_path="o.mp4", keep_resident=True)
    req = {
        "missing_image": H3GenerateRequest(**base, images=(H3ImageInput(str(tmp_path / "nope.png"), 0),)),
        "duplicate_frames": H3GenerateRequest(**base, images=(H3ImageInput("a.png", 0), H3ImageInput("b.png", 0))),
        "bad_frame_rate": H3GenerateRequest(**base, frame_rate=30),
    }[bad]
    with pytest.raises(ValueError):
        pipe.generate(req)
    assert pipe._resident is None and calls == []


def test_worker_releases_after_a_job_refused_before_generate(monkeypatch):
    """Review M-1, worker side: a malformed message fails before ``generate``; the
    GENERATE_FAILED branch still releases the kept models."""
    import io

    from engine_h3 import worker

    class FakePipe:
        sampler = "euler"
        released = 0

        def release_resident(self):
            FakePipe.released += 1

        def generate(self, req):  # never reached
            raise AssertionError("generate must not run")

        def close(self):
            pass

    events = []
    monkeypatch.setattr(worker, "_PIPE", FakePipe())
    monkeypatch.setattr(worker, "_do_load", lambda msg: None)
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append(event))
    monkeypatch.setattr(worker, "_log", lambda m: None)
    lines = [
        json.dumps({"op": "load"}),
        json.dumps({"op": "generate", "prompt": "p", "seed": 1, "width": 640, "height": 384, "num_frames": 124,
                    "output_path": "o.mp4", "images": "not-a-list"}),
    ]
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO("\n".join(lines) + "\n"))
    with pytest.raises(SystemExit):
        worker.main()
    assert events == ["error"] and FakePipe.released == 1


def test_vision_inv_freq_matches_transformers():
    """Review m-4: the recomputed vision rotary buffer is HF's own initial value."""
    pytest.importorskip("transformers")
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionRotaryEmbedding

    from engine_h3.text_encoder import vision_inv_freq

    for dim in (8, 36):
        assert torch.equal(vision_inv_freq(dim), Qwen3VLVisionRotaryEmbedding(dim).inv_freq)


@pytest.mark.parametrize("vram16", [False, True])
def test_keep_resident_state_transitions_with_a_tiny_dit(monkeypatch, tmp_path, vram16):
    """Review m-7 / plan §4-5: keep -> reuse -> release, through the REAL ``_denoise``
    and ``_decode`` (a tiny DiT on the CPU; the int8 matmul replaced by its
    dequantized float64 form, the VAEs and the mp4 writer by stand-ins).

    J1 keep: the DiT is built from the file, parked on the CPU unwrapped, the
       decoder / audio VAE kept.  J2 keep: nothing is read again (same objects),
       same latents bit for bit.  J3 no key: reused, then released.
    ``vram16`` (VRAM 16GB mode, 台帳 §1-101): the token refiner and
    ``condition_proj`` go to the CPU in a keep job and to meta in the job
    without the key; J4, in the OTHER mode and from the file again, gives the
    same latents bit for bit (on == off).
    """
    from engine.transformer.block_swap_service import _BLOCK_SWAP_ATTR
    from engine_h3 import audio_vae as AV
    from engine_h3 import media_out as MO
    from engine_h3 import pipeline_h3 as P
    from engine_h3 import sft_load as SL
    from engine_h3 import video_vae as VV

    _fake_int8(monkeypatch)
    path = tmp_path / "tiny.safetensors"
    _write_sft(path, _tiny_dit_tensors(layers=3, refiner=2, grid=1025, rank=8))
    _header, cfg = h3_format.inspect_h3_dit(str(path))
    pipe = P.H3Pipeline(P.H3Paths(str(path), "te", "vv", "av", "tok"), P.H3LoadOptions(blocks_on_gpu=1),
                        device="cpu", progress=lambda *a: None, log=lambda m: None)
    pipe._dit_cfg = cfg
    builds, decoders, avaes, latents, moved = [], [], [], [], []
    real_build = pipe._build_dit
    monkeypatch.setattr(pipe, "_build_dit", lambda entry: builds.append(1) or real_build(entry))
    real_move, real_drop = SL.move_module_tree, SL.drop_to_meta
    monkeypatch.setattr(SL, "move_module_tree",
                        lambda root, device, skip: moved.append(("cpu", type(root).__name__)) or real_move(root, device, skip))
    monkeypatch.setattr(SL, "drop_to_meta", lambda module: moved.append(("meta", type(module).__name__)) or real_drop(module))
    monkeypatch.setattr(pipe, "_encode", lambda prompt, phases, **kw: (
        torch.randn(5, TEXT, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16),
        torch.ones(5, dtype=torch.long)))

    class FakeDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        def decode(self, z, tiled=True):
            latents.append(z.clone())
            return torch.zeros((22, 64, 64, 3), dtype=torch.uint8)

    class FakeAudio(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        def decode(self, a):
            return torch.zeros((2, 800 * a.shape[-1]))

    monkeypatch.setattr(VV.H3VideoDecoder, "from_file",
                        classmethod(lambda cls, *a, **k: decoders.append(1) or FakeDecoder()))
    monkeypatch.setattr(AV.H3AudioVAE, "from_file", classmethod(lambda cls, *a, **k: avaes.append(1) or FakeAudio()))
    monkeypatch.setattr(MO, "write_mp4", lambda *a, **k: 1)

    def job(keep, mode=vram16):
        moved.clear()
        req = P.H3GenerateRequest(prompt="p", seed=7, width=64, height=64, num_frames=22, output_path="o.mp4",
                                  num_inference_steps=2, keep_resident=keep, vram16_mode=mode)
        done = pipe.generate(req)
        assert done["vram16_mode_used"] == ("on" if mode else "off")
        assert ("vram16" in done["h3"]["phases"]["denoise"]) == mode
        return done

    text_modules = {"H3TokenRefiner", "Linear"}   # dit.token_refiner, dit.condition_proj

    def evicted(kind):
        return {name for k, name in moved if k == kind} & text_modules

    d1 = job(True)
    assert evicted("cpu") == (text_modules if vram16 else set()) and evicted("meta") == set()
    kept = dict(pipe._resident)
    assert d1["keep_resident_used"] == "on" and pipe.resident_parts == ["audio_vae", "decoder", "dit"]
    assert d1["h3"]["phases"]["denoise"]["dit_source"] == "file"
    assert all(not hasattr(b, _BLOCK_SWAP_ATTR) for b in kept["dit"].blocks)     # parked unwrapped
    assert all(t.device.type == "cpu" for t in kept["dit"].state_dict().values())
    assert pipe._swap_service.blocks_on_gpu == 1                                  # load value 1 <= 4: kept
    d2 = job(True)
    assert evicted("cpu") == (text_modules if vram16 else set()) and evicted("meta") == set()
    assert (len(builds), len(decoders), len(avaes)) == (1, 1, 1)                # nothing read again
    assert pipe._resident["dit"] is kept["dit"] and pipe._resident["decoder"] is kept["decoder"]
    assert d2["h3"]["phases"]["denoise"]["dit_source"] == "resident"
    assert d2["h3"]["phases"]["decode"]["vae_source"] == "resident"
    assert torch.equal(latents[0], latents[1])                                  # reuse changes nothing
    d3 = job(False)
    assert evicted("meta") == (text_modules if vram16 else set()) and evicted("cpu") == set()
    assert d3["keep_resident_used"] == "off" and d3["h3"]["phases"]["denoise"]["dit_source"] == "resident"
    assert pipe._resident is None and torch.equal(latents[2], latents[0])
    assert len(builds) == 1
    job(False, mode=not vram16)
    assert len(builds) == 2 and torch.equal(latents[3], latents[0])             # on == off, bit for bit


# --------------------------------------------------------------------------- #
# VRAM 16GB mode (台帳 §1-101, engine_h3.lowvram): same numbers as the off path
# --------------------------------------------------------------------------- #

def _int8_linear_as_float64(x2d, weight, weight_scale, bias, *, out_dtype, convrot, convrot_groupsize):
    """``comfy_kitchen.int8_linear`` on the CPU: the dequantized matmul in float64,
    rounded to ``out_dtype`` once, so a row or column split cannot change the bf16
    result through the summation order. ``convrot`` must be off (the tiny files are
    written with ``convrot=False``)."""
    assert not convrot
    w = weight.to(torch.float64) * weight_scale.to(torch.float64)
    b = None if bias is None else bias.to(torch.float64)
    return torch.nn.functional.linear(x2d.to(torch.float64), w, b).to(out_dtype)


def _fake_int8(monkeypatch):
    """Fake the kernel, not the module: the REAL ``H3Int8Linear.forward`` and
    ``forward_rows`` (slicing, casts, reshape) run on the CPU."""
    import types

    from engine_h3 import quant_linear

    fake = types.SimpleNamespace(int8_linear=_int8_linear_as_float64)
    monkeypatch.setattr(quant_linear, "_ck", lambda: fake)


def _fake_shared_quant(monkeypatch):
    """VRAM 16GB mode's shared qkv quantization (台帳 §1-105 ①) on the CPU.

    ``shares_input`` says yes (the tiny files are ``convrot=False``, K = 64); the
    fake ``comfy_kitchen.backends.cuda`` quantizer hands back the input in float64
    (scale 1, unused) and the fake GEMM is ``_int8_linear_as_float64`` on it, so a
    slice is the number ``forward_rows`` gives. The REAL ``quantize_input`` and
    ``forward_rows_q`` (cast, slicing, bias, ``lora.apply(x=None)``) run. Neither
    fake keeps its arguments (the ``h`` weakref check relies on it)."""
    import types

    from engine_h3 import quant_linear

    def quantize(x2d, group_size):
        return x2d.to(torch.float64), torch.ones(x2d.shape[0], 1, dtype=torch.float64)

    def gemm(xq, xs, weight, weight_scale, bias, out_dtype):
        return _int8_linear_as_float64(xq, weight, weight_scale, bias, out_dtype=out_dtype,
                                       convrot=False, convrot_groupsize=256)

    monkeypatch.setattr(quant_linear, "_SHARED_CB", types.SimpleNamespace(quantize_int8_rowwise_convrot64=quantize))
    monkeypatch.setattr(quant_linear.H3Int8Linear, "shares_input", lambda self, x: True)
    monkeypatch.setattr(quant_linear, "_int8_gemm_prequantized", gemm)


def _count_qkv_calls(monkeypatch):
    """Count ``forward_rows`` / ``quantize_input`` / ``forward_rows_q`` calls. Returns
    ``(calls, h_gone)``: ``h_gone`` gets, at each ``forward_rows_q`` call, whether the
    tensor last given to ``quantize_input`` has been freed (only a weakref is kept)."""
    import weakref

    from engine_h3.quant_linear import H3Int8Linear

    calls = {"forward_rows": 0, "quantize_input": 0, "forward_rows_q": 0}
    refs, h_gone = [], []
    real_rows, real_quant, real_rows_q = (
        H3Int8Linear.forward_rows, H3Int8Linear.quantize_input, H3Int8Linear.forward_rows_q)

    def rows(self, *args, **kwargs):
        calls["forward_rows"] += 1
        return real_rows(self, *args, **kwargs)

    def quant(self, x):
        calls["quantize_input"] += 1
        refs.append(weakref.ref(x))
        return real_quant(self, x)

    def rows_q(self, *args, **kwargs):
        calls["forward_rows_q"] += 1
        h_gone.append(refs[-1]() is None)
        return real_rows_q(self, *args, **kwargs)

    monkeypatch.setattr(H3Int8Linear, "forward_rows", rows)
    monkeypatch.setattr(H3Int8Linear, "quantize_input", quant)
    monkeypatch.setattr(H3Int8Linear, "forward_rows_q", rows_q)
    return calls, h_gone


def test_shared_quant_symbols_of_the_pinned_comfy_kitchen(monkeypatch):
    """台帳 §1-105 ①: the real comfy-kitchen 0.2.37 passes the probe's checks (version,
    symbols, argument lists, the extension's docstrings); no CUDA call is made."""
    import importlib.metadata
    import types

    import comfy_kitchen.backends.cuda as cb

    from engine_h3 import quant_linear

    assert quant_linear._shared_quant_missing(cb) is None
    partial = types.SimpleNamespace(**{n: getattr(cb, n) for n in quant_linear._SHARED_CB_NAMES
                                       if n != "get_cublas_workspace"})
    assert quant_linear._shared_quant_missing(partial) == "comfy_kitchen.backends.cuda.get_cublas_workspace is missing"
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.2.38")
    assert quant_linear._shared_quant_missing(cb) == "comfy-kitchen 0.2.38 != 0.2.37"


def test_shared_quant_probe_never_raises_and_says_why(monkeypatch):
    """A failed check (or a check that raises) leaves the sharing off with the reason;
    the probe starts from "off" each time, so a previous pass does not linger."""
    from engine_h3 import quant_linear

    monkeypatch.setattr(quant_linear, "_SHARED_CB", object())   # as if an earlier probe passed
    monkeypatch.setattr(quant_linear, "_SHARED_REASON", None)
    why = "comfy_kitchen.backends.cuda._C.cutlass_int8_dequant is missing"
    monkeypatch.setattr(quant_linear, "_shared_quant_missing", lambda cb: why)
    quant_linear._probe_shared_quant()
    assert quant_linear.shared_quant_status() == (False, why) and quant_linear._SHARED_CB is None

    def boom(cb):
        raise RuntimeError("boom")

    monkeypatch.setattr(quant_linear, "_shared_quant_missing", boom)
    quant_linear._probe_shared_quant()
    shared, reason = quant_linear.shared_quant_status()
    assert not shared and "RuntimeError" in reason and quant_linear._SHARED_CB is None


def test_shares_input_conditions(monkeypatch):
    """Each condition of ``H3Int8Linear.shares_input`` broken alone gives False."""
    import types

    from engine_h3 import quant_linear
    from engine_h3.quant_linear import H3Int8Linear

    fits, cutlass = [True], [True]
    fake = types.SimpleNamespace(
        _CONVROT_FUSED_MAX_K=512,
        _convrot_fused_shared_memory_fits=lambda x, k, group_size: fits[0],
        _cuda_device_supports_cutlass_int8_dequant=lambda t: cutlass[0],
    )

    def lin(k=256, convrot=True, group=256):
        return H3Int8Linear(k, 8, bias=False, spec=h3_format.QuantSpec("int8_tensorwise", convrot, group))

    def x(m=3, k=256):
        return torch.zeros(m, k, dtype=torch.bfloat16)

    monkeypatch.setattr(quant_linear, "_SHARED_CB", None)
    assert not lin().shares_input(x())                         # 1: the probe did not pass
    monkeypatch.setattr(quant_linear, "_SHARED_CB", fake)
    assert lin().shares_input(x()) and lin(512).shares_input(x(k=512))
    assert not lin(convrot=False).shares_input(x())            # 2: ConvRot, groups of 256
    assert not lin(group=64).shares_input(x())
    for k in (128, 384, 768):                                  # 3: K < 256, K % 256, K > _CONVROT_FUSED_MAX_K
        assert not lin(k).shares_input(x(k=k)), k
    fits[0] = False                                            # 4: shared memory
    assert not lin().shares_input(x())
    fits[0] = True
    assert not lin().shares_input(x(m=2))                      # 5: m > 2
    cutlass[0] = False                                         # 6: CUTLASS int8 (not Turing)
    assert not lin().shares_input(x())
    cutlass[0] = True
    for name in ("weight", "weight_scale"):                    # 7: weight and scale on x's device
        mod = lin()
        setattr(mod, name, torch.nn.Parameter(torch.empty_like(getattr(mod, name), device="meta"), requires_grad=False))
        assert not mod.shares_input(x()), name


def _tiny_vram16_dit(tmp_path, heads=4, layers=2):
    from engine_h3.dit import H3DiT
    from engine_h3.pipeline_h3 import dit_key_map_for
    from engine_h3.quant_linear import swap_in_int8_linears
    from engine_h3.sft_load import load_streaming

    path = tmp_path / f"tiny_h{heads}.safetensors"
    _write_sft(path, _tiny_dit_tensors(layers=layers, refiner=1, grid=1025, rank=8, heads=heads))
    header, cfg = h3_format.inspect_h3_dit(str(path))
    assert cfg.num_heads == heads
    with torch.device("meta"):
        dit = H3DiT(cfg)
        swap_in_int8_linears(dit, cfg.quant, bias_dtype_of=lambda n: torch.bfloat16)
    load_streaming(dit, str(path), header, key_map=dit_key_map_for(cfg))
    dit.normalize_dtypes()
    return dit


def _tiny_forward_inputs(dit, text=5, t_lat=7, gh=2, gw=2, a_lat=4):
    """tokens [S, HID] bf16 and the forward's keyword arguments; S = 5 + 8 + 28 = 41."""
    from engine_h3.dit import H3SeqLayout
    from engine_h3.pipeline_h3 import modality_rows, time_rows
    from engine_h3.rope import build_coords, rope_tables

    layout = H3SeqLayout.make(text_rows=text, cond_rows=0, audio_rows=2 * a_lat, video_rows=t_lat * gh * gw)
    coords = build_coords(text_len=text, latent_frames=t_lat, grid_h=gh, grid_w=gw, audio_latents=a_lat)
    rope_cos, rope_sin = rope_tables(coords, dit.rope_inv_freq)
    t_unique, time_index = time_rows(layout, 0.3, 0.6)
    modality = modality_rows(layout, torch.ones(text, dtype=torch.long))
    tokens = torch.randn(layout.total, HID, generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
    return tokens, dict(layout=layout, t_unique=t_unique, time_index=time_index, modality=modality,
                        rope_cos=rope_cos, rope_sin=rope_sin, attention_backend="sdpa")


def test_vram16_row_chunks_are_equal():
    from engine_h3.lowvram import row_chunks

    assert row_chunks(4096, 4096) == [(0, 4096)]                 # not split
    assert row_chunks(4097, 4096) == [(0, 2049), (2049, 4097)]   # 2 equal chunks, no 1-row sliver
    for s, chunk in ((41, 16), (9326, 4096), (39764, 4096), (100352, 4096)):
        chunks = row_chunks(s, chunk)
        sizes = [r1 - r0 for r0, r1 in chunks]
        assert chunks[0][0] == 0 and chunks[-1][1] == s
        assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))
        assert len(chunks) == math.ceil(s / chunk) and max(sizes) - min(sizes) <= 1
    assert [r1 - r0 for r0, r1 in row_chunks(9326, 4096)] == [3109, 3109, 3108]   # selftest default prompt (S=9,326); gate A-G3b used the G5 prompt (S=9,335), also 3 chunks


def test_vram16_blocks_on_gpu_for():
    from engine_h3.lowvram import blocks_on_gpu_for

    # Gate A-G5: no time difference between 4 and 8, so the window is 4 whatever S is.
    assert C.VRAM16_BLOCKS_ON_GPU == 4 and not hasattr(C, "VRAM16_BLOCKS_S_THRESHOLD")
    assert blocks_on_gpu_for(8, num_blocks=50) == 4
    assert blocks_on_gpu_for(2, num_blocks=50) == 2       # never more than asked
    for n in (0, -1, 50, 60):                             # the whole DiT on the GPU: kept as is
        assert blocks_on_gpu_for(n, num_blocks=50) == n
    assert blocks_on_gpu_for(3, num_blocks=3) == 3


def test_vram16_derive_memory_cap():
    from engine_h3.lowvram import derive_memory_cap

    gib = 2**30
    assert derive_memory_cap(10 * gib, 16 * gib, 1 * gib, 1 * gib) == 10 / 16
    assert derive_memory_cap(0, 16 * gib, 0, 1 * gib) == 0.0           # clamped below
    assert derive_memory_cap(16 * gib, 16 * gib, 4 * gib, 0) == 1.0    # clamped above


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("head_group, row_chunk", [(3, 16), (1, 7), (4, 4096), (2, 41)])
def test_vram16_block_forward_matches_off(monkeypatch, tmp_path, head_group, row_chunk, shared):
    """4 heads in groups of ``head_group`` (3 = an uneven last group; 4 = one group,
    the attention of ``H3Attention.forward`` with q, k, v projected apart), rows in
    chunks of ``row_chunk`` (S = 41): the same bf16 block output, bit for bit.
    ``shared``: the qkv input quantized once for the block (台帳 §1-105 ①, faked):
    no ``forward_rows``, one ``quantize_input``, 3 x groups ``forward_rows_q``, and
    ``h`` already freed when the first slice runs."""
    from engine_h3.lowvram import Vram16Plan

    _fake_int8(monkeypatch)
    if shared:
        _fake_shared_quant(monkeypatch)
        calls, h_gone = _count_qkv_calls(monkeypatch)
    dit = _tiny_vram16_dit(tmp_path)
    tokens, kw = _tiny_forward_inputs(dit)
    t_emb = dit.adaln_embed(kw["t_unique"])
    adaln_index = kw["time_index"] * 3 + kw["modality"]
    block = dit.blocks[0]
    out_proj = block.attn.out_proj
    seen = []   # out_proj's input: the attention output itself, before any rounding downstream

    def record(x):
        seen.append(x.clone())
        return type(out_proj).forward(out_proj, x)

    out_proj.forward = record
    with torch.no_grad():
        want = block(tokens, t_emb, adaln_index, kw["rope_cos"], kw["rope_sin"])
        block.vram16 = Vram16Plan(head_group=head_group, row_chunk=row_chunk)
        got = block(tokens, t_emb, adaln_index, kw["rope_cos"], kw["rope_sin"])
    del out_proj.forward
    assert len(seen) == 2 and torch.equal(seen[1], seen[0])
    assert torch.isfinite(want.float()).all() and got.dtype == want.dtype
    assert torch.equal(got, want)
    if shared:
        groups = math.ceil(4 / head_group)
        assert calls == {"forward_rows": 0, "quantize_input": 1, "forward_rows_q": 3 * groups}
        assert h_gone == [True] * (3 * groups)


def test_vram16_final_rows_matches_off(tmp_path):
    from engine_h3.lowvram import Vram16Plan

    dit = _tiny_vram16_dit(tmp_path)
    tokens, kw = _tiny_forward_inputs(dit)
    t_emb = dit.adaln_embed(kw["t_unique"])
    with torch.no_grad():
        want = dit.final_rows(tokens, t_emb, kw["time_index"], kw["layout"])
        got = dit.final_rows(tokens, t_emb, kw["time_index"], kw["layout"], vram16=Vram16Plan(row_chunk=5))
    for g, w in zip(got, want):
        assert g.dtype == torch.float32 and torch.equal(g, w)


def test_vram16_dit_forward_matches_off_and_frees_the_embedding(monkeypatch, tmp_path):
    """On == off bit for bit; the box is emptied and the embedding is gone before
    block 1 runs; an off forward afterwards clears every block's plan."""
    import weakref

    from engine_h3.lowvram import Vram16Plan

    _fake_int8(monkeypatch)
    dit = _tiny_vram16_dit(tmp_path, layers=3)
    tokens, kw = _tiny_forward_inputs(dit)
    seen = []
    block1 = dit.blocks[1]

    def spy(*args):
        seen.append(ref() is None)
        return type(block1).forward(block1, *args)

    with torch.no_grad():
        want = dit(tokens, **kw)
        box = [tokens.clone()]
        ref = weakref.ref(box[0])
        block1.forward = spy
        got = dit(None, tokens_box=box, vram16=Vram16Plan(head_group=3, row_chunk=16), **kw)
        del block1.forward
        assert box == [] and seen == [True] and ref() is None
        assert all(b.vram16 is not None for b in dit.blocks)
        again = dit(tokens, **kw)
    assert all(b.vram16 is None for b in dit.blocks)
    for g, w, a in zip(got, want, again):
        assert torch.isfinite(w).all() and torch.equal(g, w) and torch.equal(a, w)


def test_vram16_off_forward_never_calls_lowvram(monkeypatch, tmp_path):
    from engine_h3 import lowvram

    def boom(*a, **k):
        raise AssertionError("the off path must not reach engine_h3.lowvram")

    monkeypatch.setattr(lowvram, "block_forward", boom)
    monkeypatch.setattr(lowvram, "final_rows_chunked", boom)
    _fake_int8(monkeypatch)
    dit = _tiny_vram16_dit(tmp_path)
    tokens, kw = _tiny_forward_inputs(dit)
    with torch.no_grad():
        dit(tokens, **kw)


def test_vram16_falls_back_when_not_shared(monkeypatch, tmp_path):
    """No shared quantization (the probe did not pass; set explicitly, since the GPU
    tests leave the module state behind): every slice through ``forward_rows``."""
    from engine_h3 import quant_linear
    from engine_h3.lowvram import Vram16Plan

    _fake_int8(monkeypatch)
    monkeypatch.setattr(quant_linear, "_SHARED_CB", None)
    calls, _h_gone = _count_qkv_calls(monkeypatch)
    dit = _tiny_vram16_dit(tmp_path)
    tokens, kw = _tiny_forward_inputs(dit)
    plan = Vram16Plan(head_group=3, row_chunk=16)
    with torch.no_grad():
        want = dit(tokens, **kw)
        got = dit(None, tokens_box=[tokens.clone()], vram16=plan, **kw)
    for g, w in zip(got, want):
        assert torch.equal(g, w)
    groups = math.ceil(4 / plan.head_group)
    assert calls == {"forward_rows": 3 * groups * len(dit.blocks), "quantize_input": 0, "forward_rows_q": 0}


@pytest.mark.parametrize("vram16", [False, True])
def test_vram16_cuda_oom_is_explained(monkeypatch, tmp_path, vram16):
    """裁定 47: under VRAM 16GB mode a CUDA OOM becomes a RuntimeError that says what
    to change (and still contains "CUDA out of memory"); off, the OOM goes out as is.
    The mode's record is written in the finally either way."""
    from engine_h3 import pipeline_h3 as P

    pipe, dit = _tiny_pipe_and_dit(tmp_path)
    pipe._dit_cfg = dit.cfg
    pipe._resident = {"dit": dit, "cfg": dit.cfg}

    def oom(*a, **k):
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB")

    monkeypatch.setattr(pipe, "_place_dit", oom)
    req = P.H3GenerateRequest(prompt="p", seed=1, width=64, height=64, num_frames=22, output_path="o.mp4",
                              vram16_mode=vram16)
    entry = {}
    phases = P._Phases(cuda=False)
    with pytest.raises(RuntimeError) as exc:
        pipe._denoise(req, torch.zeros(5, TEXT, dtype=torch.bfloat16), torch.ones(5, dtype=torch.long), phases, entry)
    if vram16:
        assert not isinstance(exc.value, torch.OutOfMemoryError)
        assert isinstance(exc.value.__cause__, torch.OutOfMemoryError)
        assert str(exc.value) == P.VRAM16_OOM_MESSAGE and "CUDA out of memory" in str(exc.value)
        assert "解像度かフレーム数" in str(exc.value) and "画角" not in str(exc.value)
        assert entry["vram16"]["blocks_on_gpu"] == 1 and entry["vram16"]["cap_fraction"] is None
    else:
        assert isinstance(exc.value, torch.OutOfMemoryError)
        assert "vram16" not in entry
    assert pipe._resident is not None and "dit" not in pipe._resident    # a failed job lets the DiT go


def test_denoise_runs_with_several_key_frame_blocks(monkeypatch, tmp_path):
    """The REAL ``_denoise`` on a tiny DiT with two key frames (first and last):
    K x grid condition rows, recorded in the sequence; a block count that does
    not match the frames is refused before the DiT runs."""
    from engine_h3 import pipeline_h3 as P

    _fake_int8(monkeypatch)
    pipe, dit = _tiny_pipe_and_dit(tmp_path)
    pipe._dit_cfg = dit.cfg
    pipe._resident = {"dit": dit, "cfg": dit.cfg}
    req = P.H3GenerateRequest(prompt="p", seed=1, width=64, height=64, num_frames=22, output_path="o.mp4",
                              num_inference_steps=1)
    text = torch.zeros(5, TEXT, dtype=torch.bfloat16)
    tags = torch.ones(5, dtype=torch.long)
    cond = torch.randn(2 * 4, 96, generator=torch.Generator().manual_seed(4))
    entry = {}
    with torch.no_grad():
        video, audio, _used = pipe._denoise(req, text, tags, P._Phases(cuda=False), entry,
                                            cond_clean=cond, cond_frame_idx=[0, 21], keep=True)
    assert entry["sequence"]["cond"] == 8 and torch.isfinite(video).all() and torch.isfinite(audio).all()
    with pytest.raises(RuntimeError, match="condition rows"):
        pipe._denoise(req, text, tags, P._Phases(cuda=False), {}, cond_clean=cond, cond_frame_idx=[0], keep=True)


def test_worker_vram16_mode_key():
    from engine_h3 import worker

    base = {"prompt": "p", "seed": 1, "width": 640, "height": 384, "num_frames": 124, "output_path": "o.mp4"}
    assert worker._generate_request(base).vram16_mode is False
    assert worker._generate_request({**base, "vram16_mode": None}).vram16_mode is False
    assert worker._generate_request({**base, "vram16_mode": True}).vram16_mode is True
    args = worker._selftest_parser().parse_args([
        "--transformer", "t", "--text-encoder", "te", "--video-vae", "vv", "--audio-vae", "av",
        "--tokenizer-dir", "tok", "--output", "o.mp4", "--vram16",
    ])
    assert worker._selftest_generate_message(args, "o.mp4")["vram16_mode"] is True
    assert worker._selftest_generate_message(_selftest_args(), "o.mp4")["vram16_mode"] is False


def test_vram16_window_of_one_job_does_not_stay(tmp_path):
    """Plan #8: the window is written to the one resident service before every
    install, so a job's smaller window does not carry over to the next job."""
    pipe, dit = _tiny_pipe_and_dit(tmp_path, blocks_on_gpu=2)
    for asked, expected in ((1, 1), (None, 2)):
        service, _used = pipe._place_dit(dit, blocks_on_gpu=asked)
        assert service is pipe._swap_service and service.blocks_on_gpu == expected
        service.teardown_prefetch()
        service.release_installed()
        pipe._park_dit(dit)


def test_vram16_finish_lifts_the_cap_and_never_raises(monkeypatch):
    from engine_h3 import pipeline_h3 as P

    logs, fractions = [], []
    pipe = P.H3Pipeline(P.H3Paths("t", "te", "vv", "av", "tok"), P.H3LoadOptions(),
                        device="cpu", progress=lambda *a: None, log=logs.append)
    stats = {"num_alloc_retries": 5, "num_ooms": 1, "inactive_split_bytes.all.peak": 3 * 2**20}

    def indexed(dev):
        # torch refuses torch.device("cuda") without an index here (found at gate A-G5)
        assert isinstance(dev, int) or (isinstance(dev, torch.device) and dev.index is not None), dev
        return dev

    def set_fraction(f, dev=None):
        indexed(dev)
        fractions.append(f)

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", set_fraction)
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda dev=None: indexed(dev) is not None and dict(stats))
    record = {}
    pipe._vram16_finish(torch.device("cuda", 0), True, {"num_alloc_retries": 2, "num_ooms": 1}, record)
    assert fractions == [1.0] and logs == []
    assert record == {"num_alloc_retries": 3, "num_ooms": 0, "inactive_split_peak_mib": 3}

    def broken(dev=None):
        raise RuntimeError("no stats")

    monkeypatch.setattr(torch.cuda, "memory_stats", broken)
    record = {}
    pipe._vram16_finish(torch.device("cuda", 0), True, {"num_alloc_retries": 0, "num_ooms": 0}, record)   # no raise
    assert fractions == [1.0, 1.0] and len(logs) == 1 and "no stats" in logs[0]


# GPU (opt-in, H3_GPU_TESTS=1; gate A-G2 / plan section 3.5): the int8 kernel's splits, and sage's k.mean.

_GPU = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.environ.get("H3_GPU_TESTS") == "1"),
    reason="GPU test: set H3_GPU_TESTS=1 (and have the supervisor's GPU permission)",
)


def _gpu_int8_linear(n, k, *, convrot, bias, seed=11):
    from engine_h3.quant_linear import H3Int8Linear, require_cuda_int8

    require_cuda_int8()
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(seed)
    lin = H3Int8Linear(k, n, bias=bias, spec=h3_format.QuantSpec("int8_tensorwise", convrot, 256))
    lin.weight = torch.nn.Parameter(torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8).to(dev),
                                    requires_grad=False)
    lin.weight_scale = (torch.rand(n, 1, generator=g) * 0.01 + 1e-3).float().to(dev)
    if bias:
        lin.bias = torch.nn.Parameter(torch.randn(n, generator=g).to(dev, torch.bfloat16), requires_grad=False)
    return lin, g


@_GPU
@pytest.mark.parametrize("convrot", [True, False])
@pytest.mark.parametrize("bias", [False, True])
def test_gpu_forward_rows_join_equals_forward(convrot, bias):
    """The qkv shape (3 x 7168 = 21504 x 5376), cut in 1024-row head groups (8 heads)."""
    lin, g = _gpu_int8_linear(3 * 7168, 5376, convrot=convrot, bias=bias)
    x = torch.randn(4096, 5376, generator=g).to("cuda", torch.bfloat16)
    with torch.no_grad():
        want = lin(x)
        got = torch.cat([lin.forward_rows(x, n0, n0 + 1024) for n0 in range(0, 3 * 7168, 1024)], dim=1)
    assert torch.equal(got, want)


@_GPU
@pytest.mark.parametrize("m", [4096, 39764])
@pytest.mark.parametrize("cutlass", [True, False])
@pytest.mark.parametrize("bias", [False, True])
def test_gpu_shared_quant_join_equals_forward(monkeypatch, bias, cutlass, m):
    """台帳 §1-105 ①: the qkv input quantized once, 21 head-group slices of 1,024 rows
    through ``forward_rows_q``: each slice == ``forward_rows``, the join == ``forward``.
    ``cutlass=False`` sends both paths to cuBLAS + dequant (``_DISABLE_CUTLASS_INT8``
    is read at every call by ``int8_linear`` and by the copy). m = 39,764 is the
    I2V 896 x 1152 x 124 sequence (about 4 GB of outputs)."""
    import comfy_kitchen.backends.cuda as cb

    from engine_h3.quant_linear import shared_quant_status

    lin, g = _gpu_int8_linear(3 * 7168, 5376, convrot=True, bias=bias)
    assert shared_quant_status() == (True, None)
    if not cutlass:
        monkeypatch.setattr(cb, "_DISABLE_CUTLASS_INT8", True)
    x = torch.randn(m, 5376, generator=g).to("cuda", torch.bfloat16)
    assert lin.shares_input(x)
    with torch.no_grad():
        want = lin(x)
        xq, xs = lin.quantize_input(x)
        parts = []
        for n0 in range(0, 3 * 7168, 1024):
            part = lin.forward_rows_q(xq, xs, n0, n0 + 1024)
            assert torch.equal(part, lin.forward_rows(x, n0, n0 + 1024)), n0
            parts.append(part)
        got = torch.cat(parts, dim=1)
    assert torch.equal(got, want)


@_GPU
@pytest.mark.parametrize("n, k", [(1024, 5376), (5376, 7168), (28672, 5376), (5376, 14336)])
def test_gpu_forward_of_row_chunks_equals_forward(n, k):
    """forward(x[r]) stacked == forward(x) for m = 8192 rows cut as 2 x 4096 and as
    129 + the rest (the qkv group width, out_proj, fc1, fc2; ConvRot on, as the checkpoint)."""
    lin, g = _gpu_int8_linear(n, k, convrot=True, bias=False)
    x = torch.randn(8192, k, generator=g).to("cuda", torch.bfloat16)
    with torch.no_grad():
        want = lin(x)
        for cuts in ([(0, 4096), (4096, 8192)], [(0, 129), (129, 8192)]):
            got = torch.cat([lin(x[r0:r1]) for r0, r1 in cuts], dim=0)
            assert torch.equal(got, want), cuts


@_GPU
def test_gpu_k_mean_of_a_head_group(record_property):
    """sage subtracts ``k.mean`` over the sequence; with 8 heads instead of 56 the
    reduction might be scheduled differently. Recorded, not asserted (ruling 46)."""
    g = torch.Generator().manual_seed(13)
    k = torch.randn(1, 56, 4096, 128, generator=g).to("cuda", torch.bfloat16)
    with torch.no_grad():
        full = k.mean(dim=2, keepdim=True)
        diffs = []
        for h0 in range(0, 56, 8):
            part = k[:, h0:h0 + 8].contiguous().mean(dim=2, keepdim=True)
            diffs.append(float((part.float() - full[:, h0:h0 + 8].float()).abs().max()))
    worst = max(diffs)
    record_property("k_mean_max_abs_diff", worst)
    print(f"k.mean head groups of 8 vs 56: max abs diff {worst} (equal={worst == 0.0})")


def test_vram16_begin_passes_an_indexed_device(monkeypatch, tmp_path):
    """Gate A-G5 found it: the pipeline's ``torch.device("cuda")`` has no index, which
    ``set_per_process_memory_fraction`` refuses. Every cuda call gets an integer index."""
    from engine_h3 import pipeline_h3 as P

    _pipe, dit = _tiny_pipe_and_dit(tmp_path)
    seen = []

    def check(name, ret):
        def fn(*args):
            dev = args[-1]
            assert isinstance(dev, int) and not isinstance(dev, bool), (name, dev)
            seen.append(name)
            return ret
        return fn

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", check("mem_get_info", (10 * 2**30, 16 * 2**30)))
    monkeypatch.setattr(torch.cuda, "memory_reserved", check("memory_reserved", 2**30))
    monkeypatch.setattr(torch.cuda, "memory_stats", check("memory_stats", {}))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", check("set_fraction", None))
    plan, record, counters0, cap_set, n = P.H3Pipeline._vram16_begin(torch.device("cuda"), 8, dit)
    assert cap_set and counters0 == {"num_alloc_retries": 0, "num_ooms": 0} and n == 8   # 3-block DiT: 8 >= 3 = no swap, kept
    assert record["cap_fraction"] == round(10 / 16, 4)
    assert seen == ["mem_get_info", "memory_reserved", "memory_stats", "set_fraction"]


# --------------------------------------------------------------------------- #
# LoRA (台帳 §1-104, engine_h3.lora): a float32 low-rank path on the output
# --------------------------------------------------------------------------- #

_LORA_TARGET = re.compile(r"(token_refiner\.)?blocks\.\d+\.(attn\.qkv_proj|attn\.out_proj|mlp\.fc1|mlp\.fc2)")


def _lora_targets(dit):
    return [(n, m) for n, m in dit.named_modules() if _LORA_TARGET.fullmatch(n)]


def _tiny_lora_tensors(dit, *, rank=4, dtype=torch.float32, alpha=None, naming="ab", refiner=True, seed=1, scale=0.3):
    """A LoRA file's tensors for every target of ``dit`` (A [r, in], B [out, r])."""
    g = torch.Generator().manual_seed(seed)
    t = {}
    for name, mod in _lora_targets(dit):
        if name.startswith("token_refiner") and not refiner:
            continue
        a = (torch.randn(rank, mod.in_features, generator=g) * scale).to(dtype)
        b = (torch.randn(mod.out_features, rank, generator=g) * scale).to(dtype)
        if naming == "ab":
            stem, sa, sb = f"diffusion_model.{name}", "lora_A", "lora_B"
        elif naming == "kohya":
            stem, sa, sb = name, "lora_down", "lora_up"
        else:
            stem, sa, sb = "lora_unet_" + name.replace(".", "_"), "lora_down", "lora_up"
        t[f"{stem}.{sa}.weight"] = a
        t[f"{stem}.{sb}.weight"] = b
        if alpha is not None:
            t[f"{stem}.alpha"] = torch.tensor(float(alpha))
    return t


def _write_lora(tmp_path, name, tensors) -> Path:
    """``_write_sft`` plus 0-dim tensors (the alpha scalars of real files have shape [])."""
    path = tmp_path / name
    doc, blobs, off = {}, [], 0
    for key, ten in tensors.items():
        data = ten.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        doc[key] = {"dtype": _ST_DTYPES[ten.dtype], "shape": list(ten.shape), "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    head = json.dumps(doc).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for blob in blobs:
            f.write(blob)
    return path


def _module_state(dit) -> dict:
    """Everything attach may touch, per module (names only: valid on meta too)."""
    return {
        n: (list(m._buffers), sorted(m._non_persistent_buffers_set), list(m._forward_hooks),
            sorted(k for k in m.__dict__ if k.startswith("_lora")))
        for n, m in dit.named_modules()
    }


def test_lora_col_chunk_is_one_head_group():
    from engine_h3.lowvram import Vram16Plan

    assert C.LORA_COL_CHUNK == C.VRAM16_HEAD_GROUP * 128 == Vram16Plan().head_group * HD   # head_dim 128
    assert 7168 % C.LORA_COL_CHUNK == 0   # H3's q / k / v width: a head group never straddles a chunk


def test_lora_attach_and_detach_restore_every_module(monkeypatch, tmp_path):
    from engine_h3 import lora as L
    from engine_h3.sft_load import drop_to_meta

    _fake_int8(monkeypatch)
    dit = _tiny_vram16_dit(tmp_path)
    tokens, kw = _tiny_forward_inputs(dit)
    before, sd_keys = _module_state(dit), list(dit.state_dict())
    assert not any(k.startswith("_lora") for m in dit.modules() for k in m.__dict__)
    with torch.no_grad():
        want = dit(tokens, **kw)
    lf = L.load_lora_file(_write_lora(tmp_path, "l.safetensors", _tiny_lora_tensors(dit, alpha=4)))
    targets = len(_lora_targets(dit))
    (rep,) = L.attach(dit, [(lf, 1.0)])
    assert rep == {"name": "l.safetensors", "strength": 1.0, "layout": "ab", "dtype": "F32",
                   "pairs_applied": targets, "skipped": []}
    qkv = dit.blocks[0].attn.qkv_proj
    assert {"_lora_A_0", "_lora_B_0"} <= set(qkv._buffers) & qkv._non_persistent_buffers_set
    assert len(qkv._forward_hooks) == 1 and qkv._lora_A_0.device.type == "cpu"
    assert list(dit.state_dict()) == sd_keys            # non-persistent: the state dict does not change
    with torch.no_grad():
        lit = dit(tokens, **kw)
    assert not torch.equal(lit[0], want[0])
    assert L.detach(dit) == targets and L.detach(dit) == 0     # idempotent
    assert _module_state(dit) == before
    with torch.no_grad():
        again = dit(tokens, **kw)
    assert all(torch.equal(a, w) for a, w in zip(again, want))
    # on meta too (a released DiT)
    L.attach(dit, [(lf, 1.0)])
    drop_to_meta(dit)
    assert dit.blocks[0].attn.qkv_proj._lora_A_0.device.type == "meta"
    assert L.detach(dit) == targets and _module_state(dit) == before


def test_lora_fp32_proof_and_the_chunked_reference():
    """Part 1: an input a bf16 LoRA provably loses (bf16 rounds 1 + 2^-12 to 1, so
    the sum is 0), computed in f32. Part 2: rows > VRAM16_ROW_CHUNK, columns >
    LORA_COL_CHUNK, two LoRAs: equal to an explicit f32 reference built with the
    same chunks, rounded to bf16 once."""
    from engine_h3 import lora as L
    from engine_h3.lowvram import row_chunks

    k = 256
    root = torch.nn.Module()
    root.lin = torch.nn.Linear(k, 2, bias=False).to(torch.bfloat16)
    torch.nn.init.zeros_(root.lin.weight)
    a = torch.empty(1, k)
    a[0, 0::2], a[0, 1::2] = 1 + 2**-12, -1.0
    b = torch.tensor([[1.0], [3.0]])
    assert a.bfloat16().float().sum() == 0
    L.attach(root, [(L.H3LoraFile("t", "ab", "F32", {"lin": (a, b, 1.0)}), 1.0)])
    x = torch.ones(3, k, dtype=torch.bfloat16)
    with torch.no_grad():
        got = root.lin(x)
    ref = torch.mm(torch.mm(x.float(), a.t()), b.t()).bfloat16()
    bf16_way = (x @ a.bfloat16().t()) @ b.bfloat16().t()
    assert torch.equal(got, ref) and torch.equal(got[0], torch.tensor([2**-5, 3 * 2**-5]).bfloat16())
    assert not torch.equal(got, bf16_way)

    g = torch.Generator().manual_seed(9)
    m, k, n = C.VRAM16_ROW_CHUNK + 5, 64, C.LORA_COL_CHUNK + 300
    root = torch.nn.Module()
    root.lin = torch.nn.Linear(k, n, bias=True).to(torch.bfloat16)
    a0, b0 = torch.randn(3, k, generator=g), torch.randn(n, 3, generator=g)
    a1, b1 = torch.randn(5, k, generator=g).bfloat16(), torch.randn(n, 5, generator=g).bfloat16()
    L.attach(root, [(L.H3LoraFile("f0", "ab", "F32", {"lin": (a0, b0, 0.5)}), 0.8),
                    (L.H3LoraFile("f1", "ab", "BF16", {"lin": (a1, b1, 1.0)}), 1.3)])
    x = torch.randn(m, k, generator=g).bfloat16()
    with torch.no_grad():
        got = root.lin(x)
        ref = torch.nn.functional.linear(x, root.lin.weight, root.lin.bias)
        for r0, r1 in row_chunks(m, C.VRAM16_ROW_CHUNK):
            xs0 = torch.mm(x[r0:r1].float(), a0.t()) * (0.8 * 0.5)
            xs1 = torch.mm(x[r0:r1].float(), a1.float().t()) * 1.3
            for c0 in range(0, n, C.LORA_COL_CHUNK):
                c1 = min(c0 + C.LORA_COL_CHUNK, n)
                d = torch.mm(xs0, b0[c0:c1].t())
                d.addmm_(xs1, b1[c0:c1].float().t())
                ref[r0:r1, c0:c1] = (ref[r0:r1, c0:c1].float() + d).bfloat16()
    assert len(row_chunks(m, C.VRAM16_ROW_CHUNK)) == 2 and torch.equal(got, ref)


def test_lora_every_matmul_operand_is_f32(monkeypatch, tmp_path):
    """``torch.mm`` / ``Tensor.addmm_`` spied during an off and a VRAM 16GB mode
    forward with a BF16 and an F16 LoRA (the int8 kernel is faked with F.linear, the
    DiT has no torch.mm of its own): every operand of every LoRA matmul is f32."""
    # Covers torch.mm / Tensor.addmm_ only; the torch.equal FP32 proof and the vram16 == off test cover the rest.
    from engine_h3 import lora as L
    from engine_h3.lowvram import Vram16Plan

    _fake_int8(monkeypatch)
    dit = _tiny_vram16_dit(tmp_path)
    tokens, kw = _tiny_forward_inputs(dit)
    files = [(L.load_lora_file(_write_lora(tmp_path, f"{n}.safetensors", _tiny_lora_tensors(dit, dtype=dt, seed=s))), 1.0)
             for n, dt, s in (("bf", torch.bfloat16, 1), ("hf", torch.float16, 2))]
    L.attach(dit, files)
    seen = []
    real_mm, real_addmm = torch.mm, torch.Tensor.addmm_

    def mm(a, b, *args, **kwargs):
        seen.append(("mm", a.dtype, b.dtype))
        return real_mm(a, b, *args, **kwargs)

    def addmm_(self, a, b, *args, **kwargs):
        seen.append(("addmm_", self.dtype, a.dtype, b.dtype))
        return real_addmm(self, a, b, *args, **kwargs)

    monkeypatch.setattr(torch, "mm", mm)
    monkeypatch.setattr(torch.Tensor, "addmm_", addmm_)
    with torch.no_grad():
        dit(tokens, **kw)
        dit(tokens.clone(), vram16=Vram16Plan(), **kw)
    assert any(s[0] == "mm" for s in seen) and any(s[0] == "addmm_" for s in seen)
    assert all(dt == torch.float32 for s in seen for dt in s[1:]), sorted(set(seen), key=str)


def test_lora_alpha_scale_and_no_alpha(monkeypatch, tmp_path):
    """alpha / rank (here 8 / 4 = 2) and the strength multiply the LoRA; without alpha the
    scale is 1.0. Doubling is exact, so alpha 8 == B pre-multiplied by 2, bit for bit."""
    from engine_h3 import lora as L

    _fake_int8(monkeypatch)
    dit = _tiny_vram16_dit(tmp_path)
    tokens, kw = _tiny_forward_inputs(dit)
    with_alpha = _tiny_lora_tensors(dit, alpha=8.0)
    no_alpha = {k: (v * 2 if ".lora_B." in k else v) for k, v in _tiny_lora_tensors(dit).items()}
    fa = L.load_lora_file(_write_lora(tmp_path, "a.safetensors", with_alpha))
    fn = L.load_lora_file(_write_lora(tmp_path, "n.safetensors", no_alpha))
    assert {s for _a, _b, s in fa.modules.values()} == {2.0} and {s for _a, _b, s in fn.modules.values()} == {1.0}
    outs = []
    for lf, strength, scale in ((fa, 1.0, 2.0), (fn, 1.0, 1.0), (fa, 0.5, 1.0)):
        L.attach(dit, [(lf, strength)])
        assert dit.blocks[0].mlp.fc1._lora_specs[0].scale == scale
        with torch.no_grad():
            outs.append(dit(tokens, **kw))
        L.detach(dit)
    assert all(torch.equal(a, b) for a, b in zip(outs[0], outs[1]))
    assert not torch.equal(outs[2][0], outs[0][0])


def test_lora_kohya_and_ab_names_give_the_same_output(monkeypatch, tmp_path):
    from engine_h3 import lora as L

    _fake_int8(monkeypatch)
    dit = _tiny_vram16_dit(tmp_path)
    tokens, kw = _tiny_forward_inputs(dit)
    outs, layouts = [], []
    for naming in ("ab", "kohya", "kohya_underscore"):
        lf = L.load_lora_file(_write_lora(tmp_path, f"{naming}.safetensors",
                                          _tiny_lora_tensors(dit, alpha=4.0, naming=naming)))
        (rep,) = L.attach(dit, [(lf, 1.0)])
        layouts.append(rep["layout"])
        assert rep["pairs_applied"] == len(_lora_targets(dit))
        with torch.no_grad():
            outs.append(dit(tokens, **kw))
        L.detach(dit)
    assert layouts == ["ab", "kohya", "kohya_underscore"]
    assert all(torch.equal(a, b) for o in outs[1:] for a, b in zip(o, outs[0]))


def test_lora_misfit_pairs_are_skipped_and_recorded(tmp_path):
    from engine_h3 import lora as L

    dit = _tiny_vram16_dit(tmp_path)
    t = _tiny_lora_tensors(dit)
    t["diffusion_model.blocks.0.mlp.fc1.lora_B.weight"] = torch.zeros(2 * FFN, 3)        # rank 3 vs A's 4
    t["diffusion_model.blocks.9.mlp.fc1.lora_A.weight"] = torch.zeros(4, HID)            # no block 9
    t["diffusion_model.blocks.9.mlp.fc1.lora_B.weight"] = torch.zeros(2 * FFN, 4)
    t["diffusion_model.blocks.1.mlp.fc2.lora_A.weight"] = torch.zeros(4, FFN, dtype=torch.uint8)   # dtype
    t["diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight"] = torch.zeros(4, 8)   # outside the targets
    lf = L.load_lora_file(_write_lora(tmp_path, "odd.safetensors", t))
    (rep,) = L.attach(dit, [(lf, 1.0)])
    reasons = {s["module"]: s["reason"] for s in rep["skipped"]}
    assert set(reasons) == {"(unmapped keys)", "blocks.1.mlp.fc2", "blocks.0.mlp.fc1", "blocks.9.mlp.fc1"}
    assert "U8" in reasons["blocks.1.mlp.fc2"] and "shape" in reasons["blocks.0.mlp.fc1"]
    assert "not in this model" in reasons["blocks.9.mlp.fc1"]
    assert rep["pairs_applied"] == len(_lora_targets(dit)) - 2
    assert not hasattr(dit.blocks[0].mlp.fc1, "_lora_specs") and not hasattr(dit.blocks[1].mlp.fc2, "_lora_specs")
    assert rep["name"] == "odd.safetensors"   # the file name only, never the path


def test_lora_several_files_accumulate():
    """Two LoRAs on one float32 Linear: base + s0 (x A0^T) B0^T + s1 (x A1^T) B1^T."""
    from engine_h3 import lora as L

    g = torch.Generator().manual_seed(3)
    root = torch.nn.Module()
    root.lin = torch.nn.Linear(16, 12)
    a0, b0, a1, b1 = (torch.randn(*s, generator=g) for s in ((2, 16), (12, 2), (3, 16), (12, 3)))
    x = torch.randn(7, 16, generator=g)
    with torch.no_grad():
        base = root.lin(x)
        L.attach(root, [(L.H3LoraFile("0", "ab", "F32", {"lin": (a0, b0, 1.0)}), 0.5),
                        (L.H3LoraFile("1", "ab", "F32", {"lin": (a1, b1, 0.25)}), 2.0)])
        got = root.lin(x)
    want = base + 0.5 * (x @ a0.t()) @ b0.t() + 0.5 * (x @ a1.t()) @ b1.t()
    assert [s.k for s in root.lin._lora_specs] == [0, 1]
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5) and not torch.allclose(got, base)


def test_lora_reaches_the_token_refiner_through_the_hook(monkeypatch, tmp_path):
    from engine_h3 import lora as L

    dit = _tiny_vram16_dit(tmp_path)
    text = torch.randn(5, TEXT, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
    with torch.no_grad():
        want = dit.refine_text(text)
    L.attach(dit, [(L.load_lora_file(_write_lora(tmp_path, "l.safetensors", _tiny_lora_tensors(dit))), 1.0)])
    hit = []
    real_apply = L.apply
    monkeypatch.setattr(L, "apply", lambda mod, x, out, *a, **k: hit.append(mod) or real_apply(mod, x, out, *a, **k))
    with torch.no_grad():
        got = dit.refine_text(text)
    refiner = {id(m) for n, m in _lora_targets(dit) if n.startswith("token_refiner")}
    assert refiner and {id(m) for m in hit} == refiner
    assert type(dit.token_refiner.blocks[0].attn.qkv_proj) is torch.nn.Linear   # not replaced
    assert not torch.equal(got, want)


@pytest.mark.parametrize("heads, head_group, row_chunk", [(16, None, None), (4, 2, 16), (4, 1, 7)])
def test_lora_vram16_matches_off(monkeypatch, tmp_path, heads, head_group, row_chunk):
    """LoRA on: the whole DiT forward in VRAM 16GB mode == off, bit for bit (the
    int8 kernel faked in float64, the LoRA's own f32 matmuls real). (16, None, None)
    = the shipped constants (2 head groups of 8 = 1,024 columns each); the other
    rows shrink the head group, the column chunk (= head group x 128, as shipped)
    and the row chunk together. f32(h) @ A^T runs once per block (``xa_for``) and
    every ``forward_rows`` gets it."""
    from engine_h3 import lora as L
    from engine_h3.lowvram import Vram16Plan

    _fake_int8(monkeypatch)
    if head_group is None:
        plan = Vram16Plan()
    else:
        monkeypatch.setattr(C, "LORA_COL_CHUNK", head_group * HD)
        monkeypatch.setattr(C, "VRAM16_ROW_CHUNK", row_chunk)
        plan = Vram16Plan(head_group=head_group, row_chunk=row_chunk)
    dit = _tiny_vram16_dit(tmp_path, heads=heads)
    tokens, kw = _tiny_forward_inputs(dit)
    with torch.no_grad():
        plain = dit(tokens, **kw)
    files = [(L.load_lora_file(_write_lora(tmp_path, f"{s}.safetensors", _tiny_lora_tensors(dit, seed=s, rank=r))), st)
             for s, r, st in ((1, 4, 1.0), (2, 6, 0.7))]
    L.attach(dit, files)
    xa_calls, rows_xa = [], []
    real_xa, real_apply = L.xa_for, L.apply
    monkeypatch.setattr(L, "xa_for", lambda mod, x: xa_calls.append(mod) or real_xa(mod, x))

    def apply(mod, x, out, n0=0, n1=None, xa=None):
        if n1 is not None:
            rows_xa.append(xa is not None)
        return real_apply(mod, x, out, n0, n1, xa=xa)

    monkeypatch.setattr(L, "apply", apply)
    with torch.no_grad():
        want = dit(tokens, **kw)
        got = dit(None, tokens_box=[tokens.clone()], vram16=plan, **kw)
    assert not torch.equal(want[0], plain[0])
    for g, w in zip(got, want):
        assert torch.isfinite(w).all() and torch.equal(g, w)
    assert xa_calls == [b.attn.qkv_proj for b in dit.blocks]          # once per block
    groups = math.ceil(heads / plan.head_group)
    assert len(rows_xa) == 3 * groups * len(dit.blocks) and all(rows_xa)


@pytest.mark.parametrize("heads, head_group, row_chunk", [(16, None, None), (4, 2, 16), (4, 1, 7)])
def test_lora_vram16_shared_matches_off(monkeypatch, tmp_path, heads, head_group, row_chunk):
    """``test_lora_vram16_matches_off`` with the qkv input quantized once per block
    (台帳 §1-105 ①, faked): still == off bit for bit; every slice's ``apply`` gets
    ``x=None`` and the block's ``xa`` (rows taken from ``xa``)."""
    from engine_h3 import lora as L
    from engine_h3.lowvram import Vram16Plan

    _fake_int8(monkeypatch)
    _fake_shared_quant(monkeypatch)
    calls, h_gone = _count_qkv_calls(monkeypatch)
    if head_group is None:
        plan = Vram16Plan()
    else:
        monkeypatch.setattr(C, "LORA_COL_CHUNK", head_group * HD)
        monkeypatch.setattr(C, "VRAM16_ROW_CHUNK", row_chunk)
        plan = Vram16Plan(head_group=head_group, row_chunk=row_chunk)
    dit = _tiny_vram16_dit(tmp_path, heads=heads)
    tokens, kw = _tiny_forward_inputs(dit)
    files = [(L.load_lora_file(_write_lora(tmp_path, f"{s}.safetensors", _tiny_lora_tensors(dit, seed=s, rank=r))), st)
             for s, r, st in ((1, 4, 1.0), (2, 6, 0.7))]
    L.attach(dit, files)
    rows_args = []
    real_apply = L.apply

    def apply(mod, x, out, n0=0, n1=None, xa=None):
        if n1 is not None:
            rows_args.append((x is None, xa is not None))
        return real_apply(mod, x, out, n0, n1, xa=xa)

    monkeypatch.setattr(L, "apply", apply)
    with torch.no_grad():
        want = dit(tokens, **kw)
        got = dit(None, tokens_box=[tokens.clone()], vram16=plan, **kw)
    for g, w in zip(got, want):
        assert torch.isfinite(w).all() and torch.equal(g, w)
    groups = math.ceil(heads / plan.head_group)
    n = 3 * groups * len(dit.blocks)
    assert rows_args == [(True, True)] * n
    assert h_gone == [True] * n
    assert calls == {"forward_rows": 0, "quantize_input": len(dit.blocks), "forward_rows_q": n}


def test_lora_buffers_are_in_the_prefetch_slots(tmp_path):
    from engine.transformer.block_swap_prefetch import _enumerate_slots
    from engine_h3 import lora as L

    dit = _tiny_vram16_dit(tmp_path)
    L.attach(dit, [(L.load_lora_file(_write_lora(tmp_path, "l.safetensors", _tiny_lora_tensors(dit))), 1.0)])
    qkv = dit.blocks[0].attn.qkv_proj
    slots = [(mod, name, is_param) for mod, name, is_param, _t in _enumerate_slots(dit.blocks[0])]
    assert (qkv, "_lora_A_0", False) in slots and (qkv, "_lora_B_0", False) in slots
    L.detach(dit)
    assert not any(name.startswith("_lora") for _m, name, *_ in _enumerate_slots(dit.blocks[0]))


def test_lora_buffers_ride_the_synchronous_block_swap(monkeypatch, tmp_path):
    """The synchronous swap path (``_patch_block``: ``blk.to(device)`` for the window)
    hands the LoRA buffers to ``Tensor.to`` with their block (device "cpu" here, so
    the call is spied rather than seen as a move); the swapped forward gives the
    unswapped LoRA forward's numbers; after detach no LoRA buffer is left."""
    from engine.transformer.block_swap_service import BlockSwapService
    from engine_h3 import lora as L

    _fake_int8(monkeypatch)
    dit = _tiny_vram16_dit(tmp_path, layers=3)
    tokens, kw = _tiny_forward_inputs(dit)
    L.attach(dit, [(L.load_lora_file(_write_lora(tmp_path, "l.safetensors", _tiny_lora_tensors(dit))), 1.0)])
    with torch.no_grad():
        want = dit(tokens, **kw)
    service = BlockSwapService(blocks_on_gpu=1, device=torch.device("cpu"))
    service.install(dit)
    assert service.last_prefetch_used == "off"
    moved = []
    real_to = torch.Tensor.to

    def spy_to(self, *a, **k):
        moved.append(id(self))
        return real_to(self, *a, **k)

    t_emb = dit.adaln_embed(kw["t_unique"])
    adaln_index = kw["time_index"] * 3 + kw["modality"]
    monkeypatch.setattr(torch.Tensor, "to", spy_to)
    with torch.no_grad():
        dit.blocks[0](tokens, t_emb, adaln_index, kw["rope_cos"], kw["rope_sin"])
    monkeypatch.setattr(torch.Tensor, "to", real_to)
    assert id(dit.blocks[0].attn.qkv_proj._lora_A_0) in moved and id(dit.blocks[0].mlp.fc2._lora_B_0) in moved
    assert id(dit.blocks[1].attn.qkv_proj._lora_A_0) not in moved          # outside the window of 1
    with torch.no_grad():
        got = dit(tokens, **kw)
    assert all(torch.equal(g, w) for g, w in zip(got, want))
    service.release_installed()
    L.detach(dit)
    for block in dit.blocks:
        for mod in block.modules():
            assert not any(n.startswith("_lora") for n in mod._buffers)
            assert not any(n.startswith("_lora") for n in mod._non_persistent_buffers_set)


def test_lora_buffer_dtype_is_watched(tmp_path):
    from engine_h3 import lora as L

    dit = _tiny_vram16_dit(tmp_path)
    L.attach(dit, [(L.load_lora_file(_write_lora(tmp_path, "l.safetensors", _tiny_lora_tensors(dit))), 1.0)])
    fc2 = dit.token_refiner.blocks[0].mlp.fc2
    fc2._buffers["_lora_A_0"] = fc2._lora_A_0.half()
    with torch.no_grad(), pytest.raises(RuntimeError, match="LoRA buffer _lora_A_0"):
        fc2(torch.zeros(3, FFN, dtype=torch.bfloat16))


def _tiny_job_pipe(monkeypatch, tmp_path):
    """The REAL generate / _denoise / _decode on a tiny DiT (template:
    ``test_keep_resident_state_transitions_with_a_tiny_dit``); returns
    ``(pipe, latents, job)``."""
    from engine_h3 import audio_vae as AV
    from engine_h3 import media_out as MO
    from engine_h3 import pipeline_h3 as P
    from engine_h3 import video_vae as VV

    _fake_int8(monkeypatch)
    path = tmp_path / "tiny.safetensors"
    _write_sft(path, _tiny_dit_tensors(layers=3, refiner=2, grid=1025, rank=8))
    _header, cfg = h3_format.inspect_h3_dit(str(path))
    pipe = P.H3Pipeline(P.H3Paths(str(path), "te", "vv", "av", "tok"), P.H3LoadOptions(blocks_on_gpu=1),
                        device="cpu", progress=lambda *a: None, log=lambda m: None)
    pipe._dit_cfg = cfg
    latents = []
    monkeypatch.setattr(pipe, "_encode", lambda prompt, phases, **kw: (
        torch.randn(5, TEXT, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16),
        torch.ones(5, dtype=torch.long)))

    class FakeDecoder(torch.nn.Module):
        def decode(self, z, tiled=True):
            latents.append(z.clone())
            return torch.zeros((22, 64, 64, 3), dtype=torch.uint8)

    class FakeAudio(torch.nn.Module):
        def decode(self, a):
            return torch.zeros((2, 800 * a.shape[-1]))

    monkeypatch.setattr(VV.H3VideoDecoder, "from_file", classmethod(lambda cls, *a, **k: FakeDecoder()))
    monkeypatch.setattr(AV.H3AudioVAE, "from_file", classmethod(lambda cls, *a, **k: FakeAudio()))
    monkeypatch.setattr(MO, "write_mp4", lambda *a, **k: 1)

    def job(keep, loras=(), vram16=False):
        req = P.H3GenerateRequest(prompt="p", seed=7, width=64, height=64, num_frames=22, output_path="o.mp4",
                                  num_inference_steps=2, keep_resident=keep, vram16_mode=vram16,
                                  loras=tuple(P.H3LoraInput(str(p), s) for p, s in loras))
        return pipe.generate(req)

    return pipe, latents, job


def _tiny_job_lora(tmp_path, name="l.safetensors", seed=1):
    """A LoRA file for the 3-block, 2-refiner tiny DiT of ``_tiny_job_pipe`` (shapes from a meta build)."""
    from engine_h3.dit import H3DiT
    from engine_h3.quant_linear import swap_in_int8_linears

    _header, cfg = h3_format.inspect_h3_dit(str(tmp_path / "tiny.safetensors"))
    with torch.device("meta"):
        shapes = H3DiT(cfg)
        swap_in_int8_linears(shapes, cfg.quant, bias_dtype_of=lambda n: torch.bfloat16)
    return _write_lora(tmp_path, name, _tiny_lora_tensors(shapes, seed=seed, alpha=4.0))


def test_lora_detach_runs_after_the_prefetch_teardown(monkeypatch, tmp_path):
    """Review Major 5: ``_release`` writes ``mod._buffers[name]`` back by name, so
    detach comes after ``teardown_prefetch`` (and ``release_installed``), and before
    ``_park_dit``."""
    from engine.transformer.block_swap_service import BlockSwapService
    from engine_h3 import lora as L

    pipe, _latents, job = _tiny_job_pipe(monkeypatch, tmp_path)
    lora = _tiny_job_lora(tmp_path)
    order = []
    for name in ("teardown_prefetch", "release_installed"):
        real = getattr(BlockSwapService, name)
        monkeypatch.setattr(BlockSwapService, name, lambda self, _n=name, _r=real: order.append(_n) or _r(self))
    real_detach, real_park = L.detach, pipe._park_dit
    monkeypatch.setattr(L, "detach", lambda dit: order.append("detach") or real_detach(dit))
    monkeypatch.setattr(pipe, "_park_dit", lambda dit: order.append("park") or real_park(dit))
    done = job(True, loras=[(lora, 1.0)])
    assert order[-4:] == ["teardown_prefetch", "release_installed", "detach", "park"], order
    (rep,) = done["h3"]["loras"]
    assert rep["name"] == "l.safetensors" and rep["pairs_applied"] == 3 * 4 + 2 * 4 and rep["skipped"] == []
    assert str(tmp_path) not in json.dumps(done["h3"]["loras"])


@pytest.mark.parametrize("vram16", [False, True])
def test_lora_job_then_plain_job_equals_the_baseline(monkeypatch, tmp_path, vram16):
    """J0 no LoRA (from the file, released) -> J1 LoRA, kept -> J2 no LoRA on the kept
    DiT: J2 == J0 bit for bit, J1 differs, and the kept DiT carries no LoRA
    buffer, non-persistent entry, hook or attribute. ``h3.loras`` only in J1."""
    pipe, latents, job = _tiny_job_pipe(monkeypatch, tmp_path)
    lora = _tiny_job_lora(tmp_path)
    d0 = job(False, vram16=vram16)
    d1 = job(True, loras=[(lora, 1.0)], vram16=vram16)
    kept = pipe._resident["dit"]
    for mod in kept.modules():
        assert not any(n.startswith("_lora") for n in list(mod._buffers) + list(mod._non_persistent_buffers_set))
        assert not mod._forward_hooks and not any(k.startswith("_lora") for k in mod.__dict__)
    d2 = job(False, vram16=vram16)
    assert d2["h3"]["phases"]["denoise"]["dit_source"] == "resident"
    assert torch.equal(latents[2], latents[0]) and not torch.equal(latents[1], latents[0])
    assert list(d0["h3"]) == list(d2["h3"]) == GOLDEN_H3_KEYS
    assert list(d1["h3"]) == GOLDEN_H3_KEYS + ["loras"]


def test_lora_failed_job_still_detaches(monkeypatch, tmp_path):
    from engine_h3 import lora as L
    from engine_h3.dit import H3DiT

    pipe, _latents, job = _tiny_job_pipe(monkeypatch, tmp_path)
    lora = _tiny_job_lora(tmp_path)
    attached = []
    real_attach = L.attach
    monkeypatch.setattr(L, "attach", lambda dit, files: attached.append(dit) or real_attach(dit, files))

    def boom(self, *a, **k):
        raise RuntimeError("embed failed")

    monkeypatch.setattr(H3DiT, "embed_rows", boom)
    with pytest.raises(RuntimeError, match="embed failed"):
        job(True, loras=[(lora, 1.0)])
    (dit,) = attached
    assert not any(k.startswith("_lora") for m in dit.modules() for k in list(m._buffers) + list(m.__dict__))
    assert not any(m._forward_hooks for m in dit.modules())
    assert pipe._resident is None


def test_request_validation_loras(monkeypatch, tmp_path):
    """Refused in Japanese inside generate's try, before the text encoder runs."""
    from engine_h3.pipeline_h3 import H3GenerateRequest, H3LoraInput, validate_request

    good = _write_lora(tmp_path, "good.safetensors", {"diffusion_model.blocks.0.mlp.fc1.lora_A.weight": torch.zeros(2, 4)})
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"\xff" * 64)
    base = dict(prompt="p", seed=1, width=640, height=384, num_frames=124, output_path="o.mp4")
    validate_request(H3GenerateRequest(**base, loras=(H3LoraInput(str(good)), H3LoraInput(str(good), 2.0))))
    for lora, msg in (
        (H3LoraInput(" "), "パスが空"),
        (H3LoraInput(str(good), 0.0), "強度"),
        (H3LoraInput(str(good), -1.0), "強度"),
        (H3LoraInput(str(good), float("nan")), "強度"),
        (H3LoraInput(str(good), float("inf")), "強度"),
        (H3LoraInput(str(good), "abc"), "数値"),
        (H3LoraInput(str(tmp_path / "nope.safetensors")), "ありません"),
        (H3LoraInput(str(broken)), "読めません"),
    ):
        with pytest.raises(ValueError, match=msg):
            validate_request(H3GenerateRequest(**base, loras=(lora,)))
    pipe = _stubbed_pipe(monkeypatch)
    calls = []
    monkeypatch.setattr(pipe, "_encode", lambda *a, **k: calls.append("encode"))
    with pytest.raises(ValueError):
        pipe.generate(H3GenerateRequest(**base, loras=(H3LoraInput(str(broken)),)))
    assert calls == []


def test_inspect_lora_file_reads_the_header_only(tmp_path):
    from engine_h3 import lora as L

    path = _write_lora(tmp_path, "k.safetensors", {
        "lora_unet_blocks_0_mlp_fc1.lora_down.weight": torch.zeros(2, 4),
        "lora_unet_blocks_0_mlp_fc1.lora_up.weight": torch.zeros(6, 2),
        "lora_unet_blocks_0_adaln_proj_linear.lora_down.weight": torch.zeros(2, 4),
    })
    km = L.inspect_lora_file(path)
    assert km.layout == "kohya_underscore" and set(km.pairs) == {"blocks.0.mlp.fc1"} and not km.supported


def test_worker_loras_key(monkeypatch):
    from engine_h3 import worker

    loras, ignored = worker._loras({"loras": [{"path": "a.safetensors", "strength": 0.5, "audio_strength": 0.3},
                                              {"path": "b.safetensors", "strength": None}]})
    assert [(x.path, x.strength) for x in loras] == [("a.safetensors", 0.5), ("b.safetensors", 1.0)]
    assert ignored == ["audio_strength"]
    assert worker._loras({}) == ((), []) and worker._loras({"loras": None}) == ((), [])
    for bad in ("a.safetensors", [{"strength": 1.0}], ["a"]):
        with pytest.raises(ValueError):
            worker._loras({"loras": bad})
    base = {"prompt": "p", "seed": 1, "width": 640, "height": 384, "num_frames": 124, "output_path": "o.mp4"}
    assert worker._generate_request(base).loras == ()
    assert worker._generate_request({**base, "loras": [{"path": "x"}]}).loras[0].path == "x"

    logs, seen = [], []

    class FakePipe:
        def generate(self, req):
            seen.append(req)
            return {"attention_used": "sdpa"}

    monkeypatch.setattr(worker, "_PIPE", FakePipe())
    monkeypatch.setattr(worker, "_emit", lambda event, **f: None)
    monkeypatch.setattr(worker, "_log", logs.append)
    worker._do_generate({**base, "op": "generate", "loras": [{"path": "d/a.safetensors", "audio_strength": 1.0}]})
    assert seen[0].loras[0].path == "d/a.safetensors"
    assert "loras[].audio_strength" in logs[0] and "a.safetensors" in logs[0]


def test_worker_selftest_lora_options():
    from engine_h3 import worker

    args = worker._selftest_parser().parse_args([
        "--transformer", "t", "--text-encoder", "te", "--video-vae", "vv", "--audio-vae", "av",
        "--tokenizer-dir", "tok", "--output", "o.mp4", "--lora", "a.safetensors", "--lora", "b.safetensors",
        "--lora-strength", "0.7",
    ])
    assert worker._selftest_generate_message(args, "o.mp4")["loras"] == [
        {"path": "a.safetensors", "strength": 0.7}, {"path": "b.safetensors", "strength": 0.7}]
    assert worker._selftest_generate_message(_selftest_args(), "o.mp4")["loras"] == []


def test_golden_done_h3_keys_with_lora(monkeypatch, tmp_path):
    """``h3.loras`` is added only for a job with LoRA; the top-level keys do not change."""
    from engine_h3 import pipeline_h3

    good = _write_lora(tmp_path, "good.safetensors", {"diffusion_model.blocks.0.mlp.fc1.lora_A.weight": torch.zeros(2, 4)})
    pipe = _stubbed_pipe(monkeypatch)

    def denoise(req, h, t, phases, entry, lora_report=None, **kw):
        lora_report.append({"name": "good.safetensors"})
        return torch.zeros(1), torch.zeros(1), "on"

    monkeypatch.setattr(pipe, "_denoise", denoise)
    base = dict(prompt="p", seed=1, width=640, height=384, num_frames=124, output_path="o.mp4")
    done = pipe.generate(pipeline_h3.H3GenerateRequest(**base, loras=(pipeline_h3.H3LoraInput(str(good)),)))
    assert list(done) == GOLDEN_DONE_KEYS and list(done["h3"]) == GOLDEN_H3_KEYS + ["loras"]
    assert done["h3"]["loras"] == [{"name": "good.safetensors"}]
    done = pipe.generate(pipeline_h3.H3GenerateRequest(**base))
    assert list(done["h3"]) == GOLDEN_H3_KEYS


_LORA_DIR = ROOT / "models" / "MiniMaxH3" / "LoRA"
_DIT_FILE = ROOT / "models" / "MiniMaxH3" / "Weights" / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"


@_GPU
@pytest.mark.parametrize("name, pairs", [
    ("minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors", 208),
    ("3dsrx_minimax_h3.safetensors", 200),
    ("camera_motion_h3_lora_v1_3000_pruned.safetensors", 208),
])
def test_gpu_trial_lora_files_attach_to_the_real_dit(name, pairs):
    """The three trial files against the real DiT's shapes (built on meta; the LoRA
    tensors are read, so this is opt-in with the GPU tests)."""
    from engine_h3 import lora as L
    from engine_h3.dit import H3DiT
    from engine_h3.quant_linear import swap_in_int8_linears

    path = _LORA_DIR / name
    if not (path.is_file() and _DIT_FILE.is_file()):
        pytest.skip("trial LoRA or DiT file not present")
    _header, cfg = h3_format.inspect_h3_dit(str(_DIT_FILE))
    with torch.device("meta"):
        dit = H3DiT(cfg)
        swap_in_int8_linears(dit, cfg.quant, bias_dtype_of=lambda n: torch.bfloat16)
    (rep,) = L.attach(dit, [(L.load_lora_file(path), 1.0)])
    assert (rep["pairs_applied"], rep["skipped"]) == (pairs, [])
    assert L.detach(dit) == pairs


def test_fp32_matmul_check(monkeypatch):
    """The torch defaults pass; TF32 by any of the three switches is refused."""
    import types

    from engine_h3 import pipeline_h3 as P

    P.require_fp32_matmul()   # this venv's defaults
    ok = {"allow_tf32": False, "fp32_precision": "none"}
    for bad in ({"allow_tf32": True}, {"fp32_precision": "tf32"}, {"precision": "high"}):
        state = {**ok, **bad}
        monkeypatch.setattr(torch.backends.cuda, "matmul", types.SimpleNamespace(
            allow_tf32=state["allow_tf32"], fp32_precision=state["fp32_precision"]))
        monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda s=state: s.get("precision", "highest"))
        with pytest.raises(RuntimeError, match="TF32"):
            P.require_fp32_matmul()
    for fp32 in ("ieee", "none"):
        monkeypatch.setattr(torch.backends.cuda, "matmul", types.SimpleNamespace(allow_tf32=False, fp32_precision=fp32))
        monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "highest")
        P.require_fp32_matmul()
    monkeypatch.setattr(torch.backends.cuda, "matmul", types.SimpleNamespace(allow_tf32=False))   # older torch: no attribute
    P.require_fp32_matmul()


# --------------------------------------------------------------------------- #
# A2V: per-row masks in the step loop and the source audio (台帳 §1-103, gate G1)
# --------------------------------------------------------------------------- #

def _time_rows_before_a2v(layout, t_video, t_audio, device=None):
    """A frozen copy of ``time_rows``'s body as it was before the A2V masks (G1 ⑦)."""
    t = torch.full((layout.total,), float(t_video), dtype=torch.float32, device=device)
    t[layout.audio] = float(t_audio)
    t[layout.cond] = max(float(t_video), C.KEYFRAME_NOISE_AUG)
    t_unique, time_index = torch.unique(t, sorted=True, return_inverse=True)
    return t_unique, time_index.to(torch.long)


def test_time_rows_without_masks_is_the_function_of_before():
    """G1 ⑦: called without the new keywords, the same numbers and dtypes as before."""
    from engine_h3.dit import H3SeqLayout
    from engine_h3.pipeline_h3 import time_rows

    sv, sa = sigma_schedule(20, 12.0), sigma_schedule(20, 3.0)
    for cond in (0, 4, 8):
        layout = H3SeqLayout.make(text_rows=5, cond_rows=cond, audio_rows=6, video_rows=12)
        pairs = [(1.0 - float(sv[i]), 1.0 - float(sa[i])) for i in range(20)] + [(0.0, 0.0), (0.9995, 0.5)]
        for t_v, t_a in pairs:
            got, want = time_rows(layout, t_v, t_a), _time_rows_before_a2v(layout, t_v, t_a)
            got_none = time_rows(layout, t_v, t_a, audio_t=None, video_t=None)
            for g in (got, got_none):
                assert torch.equal(g[0], want[0]) and torch.equal(g[1], want[1])
                assert g[0].dtype == want[0].dtype and g[1].dtype == want[1].dtype


def test_request_validation_source_audio(tmp_path):
    from engine_h3.pipeline_h3 import H3AudioInput, H3GenerateRequest, validate_request

    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")   # only its existence is checked here; the decode is source_audio's
    ok = dict(prompt="p", seed=1, width=640, height=384, num_frames=124, output_path="o.mp4")
    assert H3GenerateRequest(**ok).source_audio is None
    assert H3AudioInput("x").strength == 0.5
    for s in (0.0, 0.5, 1.0, 1):
        validate_request(H3GenerateRequest(**ok, source_audio=H3AudioInput(str(wav), s)))
    for audio, match in ((H3AudioInput("  "), "パスが空"), (H3AudioInput(str(tmp_path / "none.wav")), "見つかりません"),
                         (H3AudioInput(str(wav), -0.01), "0〜1"), (H3AudioInput(str(wav), 1.01), "0〜1"),
                         (H3AudioInput(str(wav), float("nan")), "0〜1"), (H3AudioInput(str(wav), float("inf")), "0〜1"),
                         (H3AudioInput(str(wav), "x"), "数値")):
        with pytest.raises(ValueError, match=match):
            validate_request(H3GenerateRequest(**ok, source_audio=audio))


def _tiny_denoiser(monkeypatch, tmp_path):
    """The REAL ``_denoise`` on the tiny DiT (template: ``test_denoise_runs_with_several_key_frame_blocks``);
    returns ``(dit, run)``; ``run(**kw) -> (video, audio, entry)`` for a 64 x 64, 22-frame job
    (T' = 7, grid 2 x 2, A = 37)."""
    from engine_h3 import pipeline_h3 as P

    _fake_int8(monkeypatch)
    pipe, dit = _tiny_pipe_and_dit(tmp_path)
    pipe._dit_cfg = dit.cfg
    pipe._resident = {"dit": dit, "cfg": dit.cfg}
    text = torch.randn(5, TEXT, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    tags = torch.ones(5, dtype=torch.long)

    def run(steps=3, vram16=False, **kw):
        req = P.H3GenerateRequest(prompt="p", seed=11, width=64, height=64, num_frames=22, output_path="o.mp4",
                                  num_inference_steps=steps, vram16_mode=vram16)
        entry = {}
        with torch.no_grad():
            video, audio, _used = pipe._denoise(req, text, tags, P._Phases(cuda=False), entry, keep=True, **kw)
        return video, audio, entry

    return dit, run


def _clean_latents(seed=21):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, AUD, 37, generator=g), torch.randn(1, LAT, 7, 4, 4, generator=g)


def test_denoise_all_ones_mask_is_the_unmasked_run_bit_for_bit(monkeypatch, tmp_path):
    """G1 ①: an all-ones mask is never built, and the whole denoise is the plain one."""
    from engine_h3 import denoise_mask as DM

    _dit, run = _tiny_denoiser(monkeypatch, tmp_path)
    clean_a, clean_v = _clean_latents()
    assert DM.audio_mask(clean_a, torch.ones(37)) is None
    v0, a0, e0 = run()
    v1, a1, e1 = run(audio_fix=(clean_a, torch.ones(37)), video_fix=(clean_v, torch.ones(7)))
    assert torch.isfinite(v0).all() and torch.isfinite(a0).all()
    assert torch.equal(v1, v0) and torch.equal(a1, a0)
    assert "mask" not in e0 and "mask" not in e1 and e1["m_per_step"] == e0["m_per_step"]


def test_denoise_kept_rows_feed_inject_and_end_clean(monkeypatch, tmp_path):
    """G1 ③④: audio m = 0 (and video m = 0 on frame 0): at every step the DiT is given
    the injected rows and they carry t = 1.0 (audio) / 0.999 (video); the result is
    clean bit for bit on those rows."""
    from engine_h3 import denoise_mask as DM
    from engine_h3 import pipeline_h3 as P
    from engine_h3.dit import audio_to_rows, patchify_video

    dit, run = _tiny_denoiser(monkeypatch, tmp_path)
    clean_a, clean_v = _clean_latents()
    seen_audio, seen_video, seen_t = [], [], []
    real_embed = dit.embed_rows
    real_time_rows = P.time_rows

    def embed(text_refined, audio_rows, video_rows, cond_rows=None):
        seen_audio.append(audio_rows.clone())
        seen_video.append(video_rows.clone())
        return real_embed(text_refined, audio_rows, video_rows, cond_rows=cond_rows)

    def time_rows(layout, t_v, t_a, device=None, **kw):
        out = real_time_rows(layout, t_v, t_a, device=device, **kw)
        seen_t.append((layout, out[0][out[1]]))
        return out

    monkeypatch.setattr(dit, "embed_rows", embed)
    monkeypatch.setattr(P, "time_rows", time_rows)
    per_frame = torch.tensor([0.0] + [1.0] * 6)
    video, audio, entry = run(audio_fix=(clean_a, torch.zeros(37)), video_fix=(clean_v, per_frame))
    assert len(seen_audio) == 3
    # The video inject reuses the job's initial video noise (the first draw of seed 11).
    noise = torch.randn((1, LAT, 7, 4, 4), generator=torch.Generator().manual_seed(11), dtype=torch.float32)
    inject_v = patchify_video(0.999 * clean_v + 0.001 * noise)
    for rows_a, rows_v, (layout, t) in zip(seen_audio, seen_video, seen_t):
        assert torch.equal(rows_a, audio_to_rows(clean_a))
        assert torch.allclose(rows_v[:4], inject_v[:4], atol=1e-6)            # frame 0's 4 grid rows
        assert bool((t[layout.audio] == 1.0).all())
        assert t[layout.video][:4].tolist() == [float(torch.tensor(0.999))] * 4
    assert torch.equal(audio, clean_a)
    assert torch.equal(video[:, :, 0], clean_v[:, :, 0]) and not torch.equal(video[:, :, 1], clean_v[:, :, 1])
    assert entry["mask"] == {"audio": {"rows": 74, "min": 0.0, "max": 0.0},
                             "video": {"rows": 28, "min": 0.0, "max": 1.0}}


def test_denoise_with_more_than_three_times_passes_adaln_and_vram16(monkeypatch, tmp_path):
    """G1 ④: a mask of 37 different values makes M > 3 at every step; the AdaLN table
    and VRAM 16GB mode's chunked path take it, and on == off bit for bit."""
    _dit, run = _tiny_denoiser(monkeypatch, tmp_path)
    clean_a, _clean_v = _clean_latents()
    values = torch.linspace(0.0, 0.9, 37)
    v_off, a_off, e_off = run(audio_fix=(clean_a, values))
    v_on, a_on, e_on = run(audio_fix=(clean_a, values), vram16=True)
    assert min(e_off["m_per_step"]) > 3 and e_on["m_per_step"] == e_off["m_per_step"]
    assert torch.isfinite(v_off).all() and torch.isfinite(a_off).all()
    assert torch.equal(v_on, v_off) and torch.equal(a_on, a_off)
    assert torch.equal(a_off[:, :, 0], clean_a[:, :, 0])                     # the m = 0 latent ends clean


def _a2v_fakes(monkeypatch, samples=40000, *, encoder_tensors=134):
    """Stand-ins for ``engine_h3.source_audio.load_source_waveform`` (PyAV decode) and
    ``engine_h3.audio_vae.H3AudioEncoder`` (the encoder's weights), so these tests need
    neither a real audio file nor the audio VAE file; the real modules are tested in
    tests/test_h3_audio_encoder.py. Returns ``(wave, calls)``."""
    import sys
    import types

    from engine_h3 import audio_vae as AV

    wave = torch.randn(2, samples, generator=torch.Generator().manual_seed(8)) * 0.1
    calls = {"load": [], "build": 0, "encode": []}

    def load_source_waveform(path, *, keep_samples, read_samples):
        calls["load"].append((path, keep_samples, read_samples))
        return wave, {"channels_used": [0, 1], "resampler": "fake-swr"}

    class FakeEncoder:
        load_report = types.SimpleNamespace(loaded=encoder_tensors)

        @classmethod
        def from_file(cls, path, *, device):
            calls["build"] += 1
            return cls()

        def encode(self, w):
            calls["encode"].append(w.clone())
            n = math.ceil(w.shape[-1] / 800)
            return torch.randn(2, 32, n, generator=torch.Generator().manual_seed(9))

    module = types.ModuleType("engine_h3.source_audio")
    module.load_source_waveform = load_source_waveform
    monkeypatch.setitem(sys.modules, "engine_h3.source_audio", module)
    monkeypatch.setattr(AV, "H3AudioEncoder", FakeEncoder, raising=False)
    return wave, calls


def _a2v_stage_job(monkeypatch, tmp_path, strength, frames=()):
    """``generate`` through the REAL ``_encode`` / ``_encode_image`` / ``_encode_audio``
    (stand-in TE, VAE encoders and source loader); the denoise and the decode are
    stubs that record their keyword arguments. 64 x 64, 22 frames (A = 37)."""
    from PIL import Image

    from engine_h3 import pipeline_h3 as P
    from engine_h3 import text_encoder as TE
    from engine_h3 import video_vae as VV

    wave, calls = _a2v_fakes(monkeypatch)
    audio_file = tmp_path / "a.wav"
    audio_file.write_bytes(b"RIFF")
    images = []
    for n in frames:
        path = tmp_path / f"k{n}.png"
        Image.new("RGB", (64, 64), (40,) * 3).save(path)
        images.append(P.H3ImageInput(str(path), n))
    progress, seen = [], {}

    class FakeTE:
        def __init__(self, *a, **k):
            self.read_report = {}

        def load(self):
            return {}

        def encode(self, prompt, images=()):
            return torch.zeros(3, 8), torch.ones(3, dtype=torch.long)

        def release(self):
            pass

    class FakeVideoEncoder:
        def encode(self, x, tiled=True):
            return torch.zeros((1, 24, 1, 4, 4))

    monkeypatch.setattr(TE, "H3TextEncoder", FakeTE)
    monkeypatch.setattr(VV.H3VideoEncoder, "from_file", classmethod(lambda cls, *a, **k: FakeVideoEncoder()))
    pipe = P.H3Pipeline(P.H3Paths("t", "te", "vv", "av", "tok"), P.H3LoadOptions(), device="cpu",
                        progress=lambda stage, i, total: progress.append((stage, i, total)), log=lambda m: None)
    pipe._dit_cfg = object()

    def denoise(req, h, t, phases, entry, **kw):
        seen["denoise"] = kw
        return torch.zeros(1), torch.zeros(1), "on"

    def decode(req, v, a, entry, **kw):
        seen["decode"] = kw
        entry["audio_samples"] = 29333
        return 1

    monkeypatch.setattr(pipe, "_denoise", denoise)
    monkeypatch.setattr(pipe, "_decode", decode)
    req = P.H3GenerateRequest(prompt="p", seed=1, width=64, height=64, num_frames=22, output_path="o.mp4",
                              images=tuple(images), source_audio=P.H3AudioInput(str(audio_file), strength))
    done = pipe.generate(req)
    return done, [p for p in progress if p[0] == "encode"], seen, wave, calls


@pytest.mark.parametrize("strength", [1.0, 0.5])
def test_a2v_audio_encode_phase_and_its_progress(monkeypatch, tmp_path, strength):
    done, progress, seen, wave, calls = _a2v_stage_job(monkeypatch, tmp_path, strength)
    required, read = round(22 * 32000 / 24), 37 * 800 + 32000
    assert calls["load"] == [(str(tmp_path / "a.wav"), required, read)] and calls["build"] == 1
    assert torch.equal(calls["encode"][0], wave[:, :40000])                    # 40000 < 37*800 + 32000
    assert progress == [("encode", i, 4) for i in range(5)]                    # TE 0..3, then the audio 4/4
    clean, values = seen["denoise"]["audio_fix"]
    assert tuple(clean.shape) == (2, 32, 37) and clean.dtype == torch.float32
    want = torch.randn(2, 32, 50, generator=torch.Generator().manual_seed(9))[:, :, :37]
    assert torch.equal(clean, want)                                            # the first 37 latents
    assert values.tolist() == [1.0 - strength] * 37
    assert (seen["decode"]["source_wave"] is wave) == (strength == 1.0)
    phases = done["h3"]["phases"]
    assert phases["audio_encode"]["encoded_samples"] == 40000 and phases["audio_encode"]["encoder_tensors"] == 134
    assert list(done["h3"]) == GOLDEN_H3_KEYS + ["a2v"]
    assert done["h3"]["a2v"] == {
        "mask_value": 1.0 - strength, "audio_latents": 37, "required_samples": required, "encoded_samples": 40000,
        "channels_used": [0, 1], "resampler": "fake-swr", "encoder_tensors": 134,
        "audio_output": "original" if strength == 1.0 else "decoded", "muxed_audio_samples": 29333,
    }


def test_a2v_with_a_key_frame_counts_five_and_strength_zero_encodes_nothing(monkeypatch, tmp_path):
    done, progress, seen, _wave, calls = _a2v_stage_job(monkeypatch, tmp_path, 1.0, frames=(0,))
    assert progress == [("encode", i, 5) for i in range(6)]                    # TE, key frame 4/5, audio 5/5
    assert list(done["h3"]["phases"])[:3] == ["encode", "image_encode", "audio_encode"]
    done, progress, seen, _wave, calls = _a2v_stage_job(monkeypatch, tmp_path, 0.0)
    assert calls["build"] == 0 and calls["encode"] == [] and len(calls["load"]) == 1
    assert progress == [("encode", i, 3) for i in range(4)]
    assert "audio_encode" not in done["h3"]["phases"]
    assert seen["denoise"]["audio_fix"] is None and seen["decode"]["source_wave"] is None
    a2v = done["h3"]["a2v"]
    assert (a2v["mask_value"], a2v["encoded_samples"], a2v["encoder_tensors"], a2v["audio_output"]) == (
        1.0, 0, None, "decoded")


def test_a2v_tiny_job_audio_output_switch(monkeypatch, tmp_path):
    """The REAL generate / _denoise / _decode on the tiny DiT: at 1.0 the mp4 gets the
    source waveform itself and no audio VAE is built; at 0.5 the generated latents are
    decoded; at 0.0 nothing is encoded and the latents are the plain job's, bit for bit.
    A job without ``source_audio`` has no ``a2v`` / ``mask`` / ``audio_encode``."""
    from engine_h3 import audio_vae as AV
    from engine_h3 import media_out as MO
    from engine_h3 import pipeline_h3 as P

    pipe, latents, _job = _tiny_job_pipe(monkeypatch, tmp_path)
    wave, calls = _a2v_fakes(monkeypatch)
    audio_file = tmp_path / "a.wav"
    audio_file.write_bytes(b"RIFF")
    written, decoded, builds = [], [], []
    monkeypatch.setattr(MO, "write_mp4", lambda path, frames, **k: written.append(k["audio"]) or 1)
    real_from_file = AV.H3AudioVAE.from_file

    def audio_vae_from_file(*a, **k):
        builds.append(1)
        avae = real_from_file(*a, **k)
        real_decode = avae.decode
        avae.decode = lambda z: decoded.append(z.clone()) or real_decode(z)
        return avae

    monkeypatch.setattr(AV.H3AudioVAE, "from_file", audio_vae_from_file)
    real_denoise = pipe._denoise
    audio_out = []
    monkeypatch.setattr(pipe, "_denoise", lambda *a, **k: (lambda r: audio_out.append(r[1]) or r)(real_denoise(*a, **k)))

    def job(strength=None):
        source = None if strength is None else P.H3AudioInput(str(audio_file), strength)
        req = P.H3GenerateRequest(prompt="p", seed=7, width=64, height=64, num_frames=22, output_path="o.mp4",
                                  num_inference_steps=2, source_audio=source)
        return pipe.generate(req)

    plain = job()
    assert "a2v" not in plain["h3"] and "audio_encode" not in plain["h3"]["phases"]
    assert "mask" not in plain["h3"]["phases"]["denoise"]
    assert "audio_output" not in plain["h3"]["phases"]["decode"] and len(builds) == 1   # unchanged record

    full = job(1.0)
    assert written[-1] is wave and len(builds) == 1                     # no audio VAE built
    decode = full["h3"]["phases"]["decode"]
    assert (decode["audio_output"], decode["audio_vae_source"], decode["audio_samples"]) == ("original", "skipped", 29333)
    clean = torch.randn(2, 32, 50, generator=torch.Generator().manual_seed(9))[:, :, :37]
    assert torch.equal(audio_out[-1], clean)                            # the kept latents end clean
    assert full["h3"]["a2v"]["audio_output"] == "original" and full["h3"]["a2v"]["muxed_audio_samples"] == 29333
    assert full["h3"]["phases"]["denoise"]["mask"]["audio"] == {"rows": 74, "min": 0.0, "max": 0.0}

    half = job(0.5)
    assert len(builds) == 2 and torch.equal(decoded[-1], audio_out[-1])
    assert written[-1] is not wave and half["h3"]["a2v"]["audio_output"] == "decoded"
    assert half["h3"]["phases"]["decode"]["audio_output"] == "decoded"
    assert "audio_vae_source" not in half["h3"]["phases"]["decode"]
    assert half["h3"]["a2v"]["muxed_audio_samples"] == min(37 * 800, 29333)

    builds_before = calls["build"]
    zero = job(0.0)
    assert calls["build"] == builds_before and "audio_encode" not in zero["h3"]["phases"]
    assert "mask" not in zero["h3"]["phases"]["denoise"]
    assert torch.equal(latents[-1], latents[0]) and torch.equal(audio_out[-1], audio_out[0])
    assert zero["h3"]["a2v"]["audio_output"] == "decoded"


# --------------------------------------------------------------------------- #
# Ref2VA: references in the denoise and in generate (台帳 §1-107, plan §3.1)
# --------------------------------------------------------------------------- #

def _modality_rows_before_ref(layout, text_tags):
    """A frozen copy of ``modality_rows``'s body as it was before the references."""
    m = torch.full((layout.total,), C.MODALITY_VIDEO, dtype=torch.long, device=text_tags.device)
    m[layout.text] = text_tags.to(torch.long)
    m[layout.audio] = C.MODALITY_AUDIO
    return m


def _tiny_refs(seed=31):
    """One picture (latent 4 x 2 -> 2 rows), one video (vt 2, latent 2 x 2 -> 2 rows) and
    one audio (rt 3 -> 6 rows) reference, as ``_encode_references`` hands them over."""
    from engine_h3.rope import RefShape

    g = torch.Generator().manual_seed(seed)
    return [
        (RefShape.picture(4, 2), torch.randn(2, 96, generator=g)),
        (RefShape.video(2, 2, 2), torch.randn(2, 96, generator=g)),
        (RefShape.audio(3), torch.randn(6, AUD, generator=g)),
    ]


def test_time_and_modality_rows_without_references_are_the_functions_of_before():
    """Reference ranges empty: the per-row times, their numbering and the modality
    rows are the ones of before, for every step of a 20-step job."""
    from engine_h3.dit import H3SeqLayout
    from engine_h3.pipeline_h3 import modality_rows, time_rows

    sv, sa = sigma_schedule(20, 12.0), sigma_schedule(20, 3.0)
    tags = torch.tensor([1, 1, 0, 0, 1])
    for cond in (0, 4):
        layout = H3SeqLayout.make(text_rows=5, cond_rows=cond, audio_rows=6, video_rows=12)
        explicit = H3SeqLayout.make(text_rows=5, cond_rows=cond, audio_rows=6, video_rows=12,
                                    ref_video_rows=0, ref_audio_rows=0)
        assert explicit == layout
        assert torch.equal(modality_rows(layout, tags), _modality_rows_before_ref(layout, tags))
        for i in range(20):
            t_v, t_a = 1.0 - float(sv[i]), 1.0 - float(sa[i])
            got, want = time_rows(layout, t_v, t_a), _time_rows_before_a2v(layout, t_v, t_a)
            assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def test_time_and_modality_rows_of_the_reference_ranges():
    from engine_h3.dit import H3SeqLayout
    from engine_h3.pipeline_h3 import modality_rows, time_rows

    layout = H3SeqLayout.make(text_rows=3, cond_rows=2, audio_rows=4, video_rows=6,
                              ref_video_rows=5, ref_audio_rows=4)
    for t_v, t_a in ((0.0, 0.0), (0.5, 0.25), (0.9995, 0.7)):
        t_unique, index = time_rows(layout, t_v, t_a)
        t = t_unique[index]
        assert bool((t[layout.ref_video] == torch.tensor(max(t_v, 0.999))).all())
        assert bool((t[layout.ref_audio] == 1.0).all())
        assert bool((t[layout.cond] == torch.tensor(max(t_v, 0.999))).all())
        assert bool((t[layout.audio] == torch.tensor(t_a)).all())
    m = modality_rows(layout, torch.ones(3, dtype=torch.long))
    assert m[layout.ref_video].tolist() == [C.MODALITY_VIDEO] * 5
    assert m[layout.ref_audio].tolist() == [C.MODALITY_AUDIO] * 4


def test_ordered_references_kind_then_slot():
    from engine_h3.pipeline_h3 import H3ReferenceInput as R
    from engine_h3.pipeline_h3 import ordered_references

    given = [R("audio", "a2", 2), R("video", "v1", 1), R("picture", "p3", 3), R("audio", "a1", 1),
             R("picture", "p1", 1), R("video", "v2", 2)]
    assert [(r.kind, r.slot) for r in ordered_references(given)] == [
        ("picture", 1), ("picture", 3), ("video", 1), ("video", 2), ("audio", 1), ("audio", 2)]
    assert ordered_references(()) == [] and ordered_references(None) == []


def test_denoise_without_references_is_the_same_job(monkeypatch, tmp_path):
    """refs=None / [] / absent: the same latents, the same M per step, the same
    sequence record (no ref keys) and the same random draws. The three runs all
    go through the new ``_denoise``; equality with the code before the change is
    carried by G-R0 (the fingerprint) and the per-part tests against frozen copies."""
    from engine_h3 import pipeline_h3 as P

    _dit, run = _tiny_denoiser(monkeypatch, tmp_path)
    draws = []
    real_randn = torch.randn

    def counting_randn(*a, **k):
        if k.get("generator") is not None:
            draws.append(tuple(a[0]))
        return real_randn(*a, **k)

    monkeypatch.setattr(P.torch, "randn", counting_randn)
    runs = []
    for kw in ({}, {"refs": None}, {"refs": []}):
        draws.clear()
        runs.append((*run(**kw), list(draws)))
    (v0, a0, e0, n0) = runs[0]
    assert len(n0) == 2                                           # video, then audio: nothing more
    for v, a, e, n in runs[1:]:
        assert n == n0 and torch.equal(v, v0) and torch.equal(a, a0)
        assert e["sequence"] == e0["sequence"] and e["m_per_step"] == e0["m_per_step"]
    assert "ref_video" not in e0["sequence"] and "ref_audio" not in e0["sequence"]


def test_denoise_with_a_picture_a_video_and_an_audio_reference(monkeypatch, tmp_path):
    """The REAL ``_denoise`` on the tiny DiT with one reference of each kind: the
    reference rows are mixed (picture / video, after the video and audio draws)
    or handed over clean (audio), given unchanged at every step with t = 0.999 /
    1.0, and recorded in the sequence; VRAM 16GB mode gives the same numbers."""
    from engine_h3 import pipeline_h3 as P

    dit, run = _tiny_denoiser(monkeypatch, tmp_path)
    refs = _tiny_refs()
    seen, seen_t = [], []
    real_embed = dit.embed_rows
    real_time_rows = P.time_rows

    def embed(text_refined, audio_rows, video_rows, cond_rows=None, **kw):
        seen.append({k: (None if v is None else v.clone()) for k, v in kw.items()})
        return real_embed(text_refined, audio_rows, video_rows, cond_rows=cond_rows, **kw)

    def time_rows(layout, t_v, t_a, device=None, **kw):
        out = real_time_rows(layout, t_v, t_a, device=device, **kw)
        seen_t.append((layout, out[0][out[1]]))
        return out

    monkeypatch.setattr(dit, "embed_rows", embed)
    monkeypatch.setattr(P, "time_rows", time_rows)
    video, audio, entry = run(refs=refs)
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()
    seq = entry["sequence"]
    assert (seq["ref_video"], seq["ref_audio"]) == (4, 6)
    assert seq["total"] == seq["text"] + 4 + 6 + seq["audio"] + seq["video"]
    # One draw per picture / video reference after the job's video and audio draws
    # (no key frame here), in row order.
    gen = torch.Generator().manual_seed(11)
    torch.randn((1, LAT, 7, 4, 4), generator=gen, dtype=torch.float32)
    torch.randn((2, AUD, 37), generator=gen, dtype=torch.float32)
    want = torch.cat([P.noisy_condition(refs[0][1], 2, gen), P.noisy_condition(refs[1][1], 2, gen)])
    assert len(seen) == 3
    for kw in seen:
        assert torch.equal(kw["ref_video_rows"], want)
        assert torch.equal(kw["ref_audio_rows"], refs[2][1])
    for layout, t in seen_t:
        assert (layout.ref_video.stop - layout.ref_video.start, layout.ref_audio.stop - layout.ref_audio.start) == (4, 6)
        assert t[layout.ref_video].tolist() == [float(torch.tensor(0.999))] * 4
        assert t[layout.ref_audio].tolist() == [1.0] * 6
    v_on, a_on, _e = run(refs=refs, vram16=True)
    assert torch.equal(v_on, video) and torch.equal(a_on, audio)
    v_plain, _a, _e = run()
    assert not torch.equal(v_plain, video)                        # the references reach the result
    # An audio reference listed first changes nothing: the picture / video rows and
    # the audio rows are taken by kind, and the coordinates follow the same order.
    v_order, a_order, _e = run(refs=[refs[2], refs[0], refs[1]])
    assert torch.equal(v_order, video) and torch.equal(a_order, audio)
    with pytest.raises(RuntimeError, match="do not match"):
        run(refs=[(refs[0][0], refs[0][1][:1])])


def test_denoise_with_a2v_and_an_audio_reference(monkeypatch, tmp_path):
    """A2V (``audio_fix``) and an audio reference in one job: ``ref_audio`` and the
    target ``audio`` are separate ranges of the sequence, the reference rows carry
    t = 1.0 and the A2V rows the mask's time (m = 0 -> 1.0, m = 1 -> the step's t),
    and the record holds both the reference count and the mask."""
    from engine_h3 import pipeline_h3 as P

    dit, run = _tiny_denoiser(monkeypatch, tmp_path)
    clean_a, _clean_v = _clean_latents()
    ref = _tiny_refs()[2]
    seen_ref, seen_t = [], []
    real_embed = dit.embed_rows
    real_time_rows = P.time_rows

    def embed(text_refined, audio_rows, video_rows, cond_rows=None, **kw):
        seen_ref.append(kw.get("ref_audio_rows"))
        return real_embed(text_refined, audio_rows, video_rows, cond_rows=cond_rows, **kw)

    def time_rows(layout, t_v, t_a, device=None, **kw):
        out = real_time_rows(layout, t_v, t_a, device=device, **kw)
        seen_t.append((layout, t_a, out[0][out[1]]))
        return out

    monkeypatch.setattr(dit, "embed_rows", embed)
    monkeypatch.setattr(P, "time_rows", time_rows)
    mask = torch.tensor([0.0] * 10 + [1.0] * 27)
    video, audio, entry = run(audio_fix=(clean_a, mask), refs=[ref])
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()
    seq = entry["sequence"]
    assert (seq["ref_video"], seq["ref_audio"], seq["audio"]) == (0, 6, 74)
    assert seq["total"] == seq["text"] + 6 + seq["audio"] + seq["video"]
    assert entry["mask"]["audio"] == {"rows": 74, "min": 0.0, "max": 1.0}
    assert len(seen_ref) == 3 and all(torch.equal(r, ref[1]) for r in seen_ref)
    for layout, t_a, t in seen_t:
        assert layout.ref_audio.stop <= layout.audio.start                    # no overlap
        assert (layout.ref_audio.stop - layout.ref_audio.start) == 6
        assert t[layout.ref_audio].tolist() == [1.0] * 6
        kept = (mask == 0).repeat(2)                                         # channel 0's latents, then channel 1's
        t_audio = t[layout.audio]
        assert bool((t_audio[kept] == 1.0).all())
        assert bool((t_audio[~kept] == torch.tensor(t_a)).all())
    assert torch.equal(audio[:, :, :10], clean_a[:, :, :10])               # the kept latents end clean


def _ref_job(monkeypatch, tmp_path, references, frames=()):
    """``generate`` through the REAL ``_encode`` / ``_encode_image`` / ``_read_references`` /
    ``_encode_references`` with a stand-in text encoder, video encoder, audio encoder and
    loader (``_a2v_fakes``, 2,000 samples) and video reader; the denoise and the decode
    are stubs. 64 x 64, 22 frames. ``references``: ``(kind, slot, max_sec)``.
    Returns ``(done, te_calls, encode progress, denoise kwargs, builds)``."""
    import numpy as np
    from PIL import Image

    from engine_h3 import pipeline_h3 as P
    from engine_h3 import reference as RM
    from engine_h3 import text_encoder as TE
    from engine_h3 import video_vae as VV

    _wave, calls = _a2v_fakes(monkeypatch, samples=2000)
    te_calls, progress, seen = [], [], {}
    builds = {"video": 0, "audio_loads": calls["load"], "audio_encodes": calls["encode"]}

    class FakeTE:
        def __init__(self, *a, **k):
            self.read_report = {}

        def load(self):
            return {}

        def encode(self, prompt, images=(), *, refs=None):
            te_calls.append({"images": len(images), "refs": None if refs is None else [(r.kind, r.slot) for r in refs]})
            self.read_report = {"vision_tokens": 7 * len(images)}
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
        n = 22 if max_sec is None else 5
        info = {"source_size": [64, 32], "size": [64, 32], "source_fps": 24.0, "source_frames": n + 2, "used_frames": n}
        return np.zeros((n, 32, 64, 3), np.uint8), info

    monkeypatch.setattr(TE, "H3TextEncoder", FakeTE)
    monkeypatch.setattr(VV.H3VideoEncoder, "from_file", classmethod(build_video))
    monkeypatch.setattr(RM, "read_reference_video", read_video)
    pipe = P.H3Pipeline(P.H3Paths("t", "te", "vv", "av", "tok"), P.H3LoadOptions(), device="cpu",
                        progress=lambda stage, i, total: progress.append((stage, i, total)), log=lambda m: None)
    pipe._dit_cfg = object()

    def denoise(req, h, t, phases, entry, **kw):
        seen.update(kw)
        return torch.zeros(1), torch.zeros(1), "on"

    monkeypatch.setattr(pipe, "_denoise", denoise)
    monkeypatch.setattr(pipe, "_decode", lambda req, v, a, entry, **kw: 1)
    images = []
    for n in frames:
        path = tmp_path / f"k{n}.png"
        Image.new("RGB", (64, 64), (40,) * 3).save(path)
        images.append(P.H3ImageInput(str(path), n))
    refs = []
    for kind, slot, max_sec in references:
        path = tmp_path / (f"{kind}{slot}.png" if kind == "picture" else f"{kind}{slot}.bin")
        if kind == "picture":
            Image.new("RGB", (128, 64), (90,) * 3).save(path)
        else:
            path.write_bytes(b"x")
        refs.append(P.H3ReferenceInput(kind, str(path), slot, max_sec))
    req = P.H3GenerateRequest(prompt="p", seed=1, width=64, height=64, num_frames=22, output_path="o.mp4",
                              images=tuple(images), references=tuple(refs))
    done = pipe.generate(req)
    return done, te_calls, [p for p in progress if p[0] == "encode"], seen, builds


def test_references_sorted_key_frames_kept_off_the_vision_tower(monkeypatch, tmp_path):
    refs = [("audio", 2, None), ("video", 1, None), ("picture", 3, None), ("picture", 1, None), ("audio", 1, 0.5)]
    done, te_calls, progress, kw, builds = _ref_job(monkeypatch, tmp_path, refs, frames=(0, 21))
    order = [("picture", 1), ("picture", 3), ("video", 1), ("audio", 1), ("audio", 2)]
    assert te_calls == [{"images": 0, "refs": order}]         # the key frames do not reach the vision tower
    assert [s.kind for s, _r in kw["refs"]] == ["picture", "picture", "video", "audio", "audio"]
    assert kw["cond_frame_idx"] == [0, 21] and tuple(kw["cond_clean"].shape) == (2 * 4, 96)
    assert builds["video"] == 1                              # one video encoder for key frames and references
    # encode: 3 (TE) + key frames + references = 5 steps, image_encode at 4, reference_encode at 5.
    assert {total for _s, _i, total in progress} == {5} and progress[-2:] == [("encode", 4, 5), ("encode", 5, 5)]
    phases = done["h3"]["phases"]
    assert list(phases) == ["encode", "image_encode", "reference_encode", "denoise", "decode"]
    assert phases["image_encode"]["vision_tokens"] == 0       # only the key frames' share, none here
    assert "seconds" in phases["reference_encode"]
    recs = phases["reference_encode"]["references"]
    assert [(r["kind"], r["slot"]) for r in recs] == order
    pic = recs[0]
    # 128 x 64 for a 64 x 64 job (match): scale sqrt(1/2), aspect kept -> 96 x 32 (multiples of 32).
    assert set(pic) == {"kind", "slot", "source_size", "size", "vision_tokens", "rows", "seconds"}
    assert (pic["source_size"], pic["size"], pic["vision_tokens"]) == ([128, 64], [96, 32], 5)
    assert pic["rows"] == 1 * 3
    vid = recs[2]
    assert set(vid) == {"kind", "slot", "source_frames", "used_frames", "latent_t", "pairs", "rows", "seconds"}
    assert (vid["source_frames"], vid["used_frames"], vid["latent_t"], vid["pairs"]) == (24, 22, 7, 1)
    assert vid["rows"] == 7 * 1 * 2
    aud = recs[3]
    assert set(aud) == {"kind", "slot", "samples", "latent_t", "rows", "seconds"}
    assert (aud["samples"], aud["latent_t"], aud["rows"]) == (2000, 3, 6)
    # The audio references: keep_samples=1, min(max_sec, 15 s) x 32000 read; one encoder.
    assert [(k, r) for _p, k, r in builds["audio_loads"]] == [(1, 16000), (1, 15 * 32000)]
    assert done["h3"]["references"] == {"pictures": 2, "videos": 1, "audios": 2}
    assert list(done) == GOLDEN_DONE_KEYS and list(done["h3"]) == GOLDEN_H3_KEYS + ["references"]
    for shape, rows in kw["refs"]:
        assert rows.device.type == "cpu" and rows.dtype == torch.float32 and rows.shape[0] == shape.rows


def test_audio_only_references_build_no_video_encoder(monkeypatch, tmp_path):
    done, te_calls, progress, kw, builds = _ref_job(monkeypatch, tmp_path, [("audio", 1, None)])
    assert builds["video"] == 0
    assert te_calls == [{"images": 0, "refs": [("audio", 1)]}]
    assert progress[-1] == ("encode", 4, 4)
    assert list(done["h3"]["phases"]) == ["encode", "reference_encode", "denoise", "decode"]
    assert done["h3"]["references"] == {"pictures": 0, "videos": 0, "audios": 1}


def test_reference_audio_shorter_than_one_latent_is_refused(monkeypatch, tmp_path):
    import sys
    import types

    from engine_h3 import pipeline_h3 as P

    module = types.ModuleType("engine_h3.source_audio")
    module.load_source_waveform = lambda path, *, keep_samples, read_samples: (torch.zeros(2, 640), {})
    monkeypatch.setitem(sys.modules, "engine_h3.source_audio", module)
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    pipe = P.H3Pipeline(P.H3Paths("t", "te", "vv", "av", "tok"), P.H3LoadOptions(), device="cpu",
                        progress=lambda *a: None, log=lambda m: None)
    req = P.H3GenerateRequest(prompt="p", seed=1, width=64, height=64, num_frames=22, output_path="o.mp4",
                              references=(P.H3ReferenceInput("audio", str(f), 1),))
    with pytest.raises(ValueError, match="参照音声が短すぎます"):
        pipe._read_references(req)


def test_job_without_references_records_nothing_new(monkeypatch, tmp_path):
    done, te_calls, progress, kw, builds = _ref_job(monkeypatch, tmp_path, [], frames=(0,))
    assert te_calls == [{"images": 1, "refs": None}]           # the TE is called as before (no refs keyword)
    assert kw["refs"] is None and builds["video"] == 1
    assert "reference_encode" not in done["h3"]["phases"] and "references" not in done["h3"]
    assert list(done["h3"]) == GOLDEN_H3_KEYS
    assert {total for _s, _i, total in progress} == {4}


def test_request_validation_references(tmp_path):
    from engine_h3.pipeline_h3 import H3GenerateRequest, H3ReferenceInput, validate_request

    f = tmp_path / "r.png"
    f.write_bytes(b"x")
    ok = dict(prompt="p", seed=1, width=640, height=384, num_frames=124, output_path="o.mp4")
    assert H3GenerateRequest(**ok).references == ()
    validate_request(H3GenerateRequest(**ok, references=(
        H3ReferenceInput("picture", str(f), 1), H3ReferenceInput("video", str(f), 1, 3.0),
        H3ReferenceInput("audio", str(f), 1, 2), H3ReferenceInput("picture", str(f), 3),
    )))
    bad = [
        ((H3ReferenceInput("image", str(f), 1),), "種類"),
        ((H3ReferenceInput("picture", str(f), 0),), "1 以上"),
        ((H3ReferenceInput("picture", str(f), True),), "整数"),
        ((H3ReferenceInput("picture", str(f), 1.0),), "整数"),
        ((H3ReferenceInput("video", str(f), 2), H3ReferenceInput("video", str(f), 2)), "重複"),
        ((H3ReferenceInput("audio", "  ", 1),), "パスが空"),
        ((H3ReferenceInput("audio", str(tmp_path / "none.wav"), 1),), "見つかりません"),
        ((H3ReferenceInput("video", str(f), 1, 0.0),), "正の有限"),
        ((H3ReferenceInput("video", str(f), 1, -1.0),), "正の有限"),
        ((H3ReferenceInput("video", str(f), 1, float("inf")),), "正の有限"),
        ((H3ReferenceInput("video", str(f), 1, float("nan")),), "正の有限"),
        ((H3ReferenceInput("video", str(f), 1, "3"),), "正の有限"),
    ]
    for refs, match in bad:
        with pytest.raises(ValueError, match=match):
            validate_request(H3GenerateRequest(**ok, references=refs))
