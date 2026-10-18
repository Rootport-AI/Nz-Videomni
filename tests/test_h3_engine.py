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
from engine_h3 import constants as C  # noqa: E402
from engine_h3.scheduler import euler_step, sigma_schedule  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Tiny but self-consistent H3 widths. head_dim stays 128: RoPE rotates its first 96 channels.
HID, HEADS, HD, FFN, TEXT, LAT, AUD = 64, 1, 128, 48, 40, 24, 32
INNER = HEADS * HD


def _tiny_cfg(*, layers=2, refiner=2, grid=1025, rank=8, quant=None):
    return h3_format.H3DiTConfig(
        prefix="", hidden_size=HID, num_layers=layers, token_refiner_layers=refiner,
        num_heads=HEADS, head_dim=HD, ffn_hidden=FFN, text_dim=TEXT, latents_dim=LAT,
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
    ids2, tags2 = presentation(tok, prompt, image_tokens=240)
    label = tok("<Picture 1>: ", add_special_tokens=False)["input_ids"]
    assert label == [21604, 3826, 220, 16, 26818, 220]
    n = len(label)
    assert ids2[:n] == label and ids2[n] == 151652 and ids2[n + 1:n + 241] == [151655] * 240
    assert ids2[n + 241] == 151653 and ids2[n + 242:] == ids
    assert tags2 == [1] * n + [0] * 242 + [1] * len(ids)
    mm = mm_token_type_ids(ids2)
    assert sum(mm) == 240 and mm[n] == 0 and mm[n + 241] == 0


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
# a tiny DiT streamed from a synthetic file
# --------------------------------------------------------------------------- #

_ST_DTYPES = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16", torch.int8: "I8", torch.uint8: "U8"}


def _tiny_dit_tensors(layers: int, refiner: int, grid: int, rank: int) -> dict:
    g = torch.Generator().manual_seed(0)

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
    lin = {"attn.qkv_proj": (3 * INNER, HID), "attn.out_proj": (HID, INNER), "mlp.fc1": (2 * FFN, HID), "mlp.fc2": (HID, FFN)}
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
]
GOLDEN_DONE_KEYS = [
    "seed_used", "sampler", "num_inference_steps", "shift_video", "shift_audio",
    "peak_vram_mb", "peak_vram_reserved_mb", "rss_peak_gib", "commit_gib", "commit_peak_gib", "seconds",
    "num_frames", "frame_rate", "width", "height", "size_bytes", "attention_used",
    "block_swap_prefetch_used", "keep_resident_used", "images_used", "h3",   # 2 keys added in stage 2 (2a)
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
    assert list(msg) == GOLDEN_GENERATE_KEYS
    assert worker._GENERATE_FIELDS == frozenset(GOLDEN_GENERATE_KEYS)


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
    validate_request(H3GenerateRequest(**ok, images=(H3ImageInput("a.png", 0),)))
    for bad in ((H3ImageInput("a.png", 0), H3ImageInput("b.png", 0)), (H3ImageInput("a.png", 9),),
                (H3ImageInput("", 0),)):
        with pytest.raises(ValueError):
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

@pytest.mark.parametrize("bad", ["missing_image", "two_images", "bad_frame_rate"])
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
        "two_images": H3GenerateRequest(**base, images=(H3ImageInput("a.png", 0), H3ImageInput("b.png", 0))),
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


def test_keep_resident_state_transitions_with_a_tiny_dit(monkeypatch, tmp_path):
    """Review m-7 / plan §4-5: keep -> reuse -> release, through the REAL ``_denoise``
    and ``_decode`` (a tiny DiT on the CPU; the int8 matmul replaced by its
    dequantized float form, the VAEs and the mp4 writer by stand-ins).

    J1 keep: the DiT is built from the file, parked on the CPU unwrapped, the
       decoder / audio VAE kept.  J2 keep: nothing is read again (same objects),
       same latents bit for bit.  J3 no key: reused, then released.
    """
    import torch.nn.functional as Fn

    from engine.transformer.block_swap_service import _BLOCK_SWAP_ATTR
    from engine_h3 import audio_vae as AV
    from engine_h3 import media_out as MO
    from engine_h3 import pipeline_h3 as P
    from engine_h3 import video_vae as VV
    from engine_h3.quant_linear import H3Int8Linear

    def int8_as_float(self, x):
        w = self.weight.to(torch.float32) * self.weight_scale.to(torch.float32)
        b = None if self.bias is None else self.bias.to(torch.float32)
        return Fn.linear(x.to(torch.float32), w, b).to(x.dtype)

    monkeypatch.setattr(H3Int8Linear, "forward", int8_as_float)
    path = tmp_path / "tiny.safetensors"
    _write_sft(path, _tiny_dit_tensors(layers=3, refiner=2, grid=1025, rank=8))
    _header, cfg = h3_format.inspect_h3_dit(str(path))
    pipe = P.H3Pipeline(P.H3Paths(str(path), "te", "vv", "av", "tok"), P.H3LoadOptions(blocks_on_gpu=1),
                        device="cpu", progress=lambda *a: None, log=lambda m: None)
    pipe._dit_cfg = cfg
    builds, decoders, avaes, latents = [], [], [], []
    real_build = pipe._build_dit
    monkeypatch.setattr(pipe, "_build_dit", lambda entry: builds.append(1) or real_build(entry))
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

    def job(keep):
        req = P.H3GenerateRequest(prompt="p", seed=7, width=64, height=64, num_frames=22, output_path="o.mp4",
                                  num_inference_steps=2, keep_resident=keep)
        return pipe.generate(req)

    d1 = job(True)
    kept = dict(pipe._resident)
    assert d1["keep_resident_used"] == "on" and pipe.resident_parts == ["audio_vae", "decoder", "dit"]
    assert d1["h3"]["phases"]["denoise"]["dit_source"] == "file"
    assert all(not hasattr(b, _BLOCK_SWAP_ATTR) for b in kept["dit"].blocks)     # parked unwrapped
    assert all(t.device.type == "cpu" for t in kept["dit"].state_dict().values())
    d2 = job(True)
    assert (len(builds), len(decoders), len(avaes)) == (1, 1, 1)                # nothing read again
    assert pipe._resident["dit"] is kept["dit"] and pipe._resident["decoder"] is kept["decoder"]
    assert d2["h3"]["phases"]["denoise"]["dit_source"] == "resident"
    assert d2["h3"]["phases"]["decode"]["vae_source"] == "resident"
    assert torch.equal(latents[0], latents[1])                                  # reuse changes nothing
    d3 = job(False)
    assert d3["keep_resident_used"] == "off" and d3["h3"]["phases"]["denoise"]["dit_source"] == "resident"
    assert pipe._resident is None and torch.equal(latents[2], latents[0])
    assert len(builds) == 1
