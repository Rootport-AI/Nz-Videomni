"""Gate G4 (DiT part): engine_h3's DiT parts against the DiffSynth oracle (GPU).

Run (only after the supervisor's GPU go-ahead, with the weights in place):
    set PYTHONPATH=<repo root>
    .venv-engine-h3\\Scripts\\python.exe outputs\\h3-gate1\\g4_compare.py [--report P.json] [--blocks 0-49]

Oracle = verbatim copies of DiffSynth-Studio's minimax_h3_dit.py and
minimax_h3_dit_comfy.py (outputs/h3-gate1/oracle/diffsynth_h3/, with SDPA and a
pass-through checkpoint stand-in for its two core imports) whose Linears are
plain nn.Linear holding the weight DEQUANTIZED to bf16 by
engine/sft_quant/dequant.dequantize (no quantized tensor type). Same file, same
inputs, same t and indices, same RoPE.

Parts
  A  coords   rope.build_coords vs DiffSynth _build_packed_fl2va (float64,
              exact), T2V and first-frame I2V, the G5 shape and a small one;
              the modality tags; rope.inv_freq of the file vs DiffSynth's
              formula; rope_tables vs cos/sin of MiniMaxH3Rope.
  B  adaln    H3DiT.adaln_embed vs MiniMaxH3ComfyPrunedTimeEmbedder (float32),
              table ends included (t = 0, 0.999, 1.0).
  C  text     refine_text vs condition_proj + MiniMaxH3TokenRefiner;
              embed_rows vs plain float32 projections; final_rows vs
              MiniMaxH3FinalLayer (pruned AdaLN, f32 heads).
  D  e2e      a 2-block H3DiT (int8, STRICT load of every non-block key and
              blocks 0-1 = the state-dict names) vs a 2-block
              MiniMaxH3DiTComfyPruned run the way DiffSynth runs it (bf16
              everywhere): checks the forward's composition (row order,
              AdaLN row = time_index*3 + tag, final row selection).
  E  blocks   every block i, one at a time (only one block's tensors on the
              GPU): mine int8 vs oracle (quant_rel), and mine with the SAME
              dequantized bf16 weights vs oracle (wiring_rel). Both are
              measured on the residual delta (out - x); x would otherwise
              dominate the norm and hide everything.

Tolerances: wiring (same weights, same precision) should be bf16-noise,
~1e-3; W8A8 vs W8A16 adds ~1e-2 per Linear (activation int8 after Hadamard).
A wrong qkv/fc1 split, a SiLU too many or a swapped AdaLN row is O(1).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORACLE = Path(__file__).resolve().parent / "oracle"
for p in (ROOT, ORACLE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from engine.sft_quant.dequant import dequantize, normalize_aux  # noqa: E402
from engine.sft_quant.sft_reader import read_tensors  # noqa: E402
from engine_h3 import rope as h3rope  # noqa: E402
from engine_h3.dit import H3DiT, H3DiTBlock, H3SeqLayout  # noqa: E402
from engine_h3.quant_linear import require_cuda_int8, swap_in_int8_linears  # noqa: E402
from h3_format import inspect_h3_dit  # noqa: E402

from diffsynth_h3.models.minimax_h3_dit import (  # noqa: E402
    MiniMaxH3DiTBlock,
    MiniMaxH3FinalLayer,
    MiniMaxH3Rope,
    MiniMaxH3TokenRefiner,
)
from diffsynth_h3.models.minimax_h3_dit_comfy import (  # noqa: E402
    MiniMaxH3ComfyPrunedTimeEmbedder,
    MiniMaxH3DiTComfyPruned,
    _comfy_attention_forward,
    _to_comfy_pruned_adaln_proj,
)
from diffsynth_h3.pipeline_coords import MiniMaxH3Unit_PackedSequenceBuilder  # noqa: E402

WEIGHTS = ROOT / "models" / "MiniMaxH3" / "Weights" / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
WIRING_TOL = 1e-2
QUANT_TOL = 5e-2
E2E_TOL = 5e-2
PART_TOL = 1e-2
SEED = 2026

# Small test sequence: text 64 | audio 2 x 24 | video 4 x 10 x 10  -> S = 512.
L, T_LAT, GH, GW, A = 64, 4, 10, 10, 24
SIGMA_V, SIGMA_A = 0.9, 0.75


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double()
    b = b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def _maxabs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max()) if a.numel() else 0.0


class _Reader:
    def __init__(self, path: Path) -> None:
        self.path = str(path)
        self.header, self.cfg = inspect_h3_dit(self.path)
        self.prefix = self.cfg.prefix

    def read(self, sub: str, *, strip: str = "") -> dict[str, torch.Tensor]:
        """Tensors whose (prefix-less) name starts with ``sub``; names lose ``strip``."""
        full = self.prefix + sub
        keys = [k for k in self.header.tensors if k.startswith(full) and not k.endswith(".comfy_quant")]
        out = {}
        for k, t in read_tensors(self.path, keys, self.header):
            name = k[len(self.prefix):]
            out[name[len(strip):] if strip and name.startswith(strip) else name] = t
        return out

    def read_nonblock(self) -> dict[str, torch.Tensor]:
        keys = [
            k
            for k in self.header.tensors
            if not k[len(self.prefix):].startswith("blocks.") and not k.endswith(".comfy_quant")
        ]
        return {k[len(self.prefix):]: t for k, t in read_tensors(self.path, keys, self.header)}


def _layout() -> H3SeqLayout:
    return H3SeqLayout.make(text_rows=L, cond_rows=0, audio_rows=2 * A, video_rows=T_LAT * GH * GW)


def _step_indices(layout: H3SeqLayout, dev) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """t_unique [M], time_index [S], modality [S] for one T2V step (text and video at t_v)."""
    t_v, t_a = 1.0 - SIGMA_V, 1.0 - SIGMA_A
    per_row = torch.empty(layout.total, dtype=torch.float32)
    per_row[layout.text] = t_v
    per_row[layout.video] = t_v
    per_row[layout.audio] = t_a
    t_unique, time_index = torch.unique(per_row, sorted=True, return_inverse=True)
    modality = torch.empty(layout.total, dtype=torch.long)
    modality[layout.text] = 1
    modality[layout.audio] = 2
    modality[layout.video] = 0
    return t_unique.to(dev), time_index.to(dev), modality.to(dev)


class _Pipe:  # what _build_packed_fl2va reads: pipe.dit.vdn_enabled (absent -> False)
    dit = object()


def _oracle_coords(text_len, latent_t, grid_h, grid_w, audio_t, keyframes):
    b = MiniMaxH3Unit_PackedSequenceBuilder()
    packed = b._build_packed_fl2va(_Pipe(), text_len, latent_t, 2 * grid_h, 2 * grid_w, audio_t, keyframes)
    used = int(packed["cu_seqlens"][1])
    return packed["img_position_ids"][0][:used], packed["token_tags"][:used]


def part_a(reader: _Reader, dev, rep: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cases = {
        "small_t2v": dict(text_len=L, latent_frames=T_LAT, grid_h=GH, grid_w=GW, audio_latents=A, cond_frames=0),
        "small_i2v": dict(text_len=L, latent_frames=T_LAT, grid_h=GH, grid_w=GW, audio_latents=A, cond_frames=1),
        # G5: 640x384, 124 frames -> T'=37, grid 12x20, A=207; L is an example length.
        "g5_t2v": dict(text_len=301, latent_frames=37, grid_h=12, grid_w=20, audio_latents=207, cond_frames=0),
        "portrait_i2v": dict(text_len=17, latent_frames=7, grid_h=24, grid_w=14, audio_latents=40, cond_frames=1),
    }
    out = {}
    for name, c in cases.items():
        mine = h3rope.build_coords(**c)
        orc, tags = _oracle_coords(
            c["text_len"], c["latent_frames"], c["grid_h"], c["grid_w"], c["audio_latents"],
            [0] if c["cond_frames"] else [],
        )
        lay = H3SeqLayout.make(
            text_rows=c["text_len"],
            cond_rows=c["cond_frames"] * c["grid_h"] * c["grid_w"],
            audio_rows=2 * c["audio_latents"],
            video_rows=c["latent_frames"] * c["grid_h"] * c["grid_w"],
        )
        my_tags = torch.empty(lay.total, dtype=torch.long)
        my_tags[lay.text] = 1
        my_tags[lay.cond] = 0
        my_tags[lay.audio] = 2
        my_tags[lay.video] = 0
        out[name] = {
            "rows": int(mine.shape[0]),
            "shape_equal": list(mine.shape) == list(orc.shape),
            "exact": bool(mine.shape == orc.shape and torch.equal(mine, orc)),
            "max_abs": _maxabs(mine, orc) if mine.shape == orc.shape else None,
            "tags_equal": bool(my_tags.shape == tags.shape and torch.equal(my_tags, tags)),
        }
    rep["A_coords"] = out

    inv_file = reader.read("rope.inv_freq")["rope.inv_freq"].to(torch.float32)
    inv_ds = MiniMaxH3Rope(inv_file.shape[0])._build_inv_freq()
    coords = h3rope.build_coords(text_len=L, latent_frames=T_LAT, grid_h=GH, grid_w=GW, audio_latents=A)
    cos, sin = h3rope.rope_tables(coords, inv_file.to(dev))
    freqs = MiniMaxH3Rope(inv_file.shape[0]).to(dev)(coords[None].to(dev))
    rep["A_rope"] = {
        "inv_freq_file_vs_formula_max_abs": _maxabs(inv_file, inv_ds),
        "cos_max_abs": _maxabs(cos, torch.cos(freqs)),
        "sin_max_abs": _maxabs(sin, torch.sin(freqs)),
        "table_shape": list(cos.shape),
    }
    rep["A_pass"] = bool(
        all(v["exact"] and v["tags_equal"] for v in out.values())
        and rep["A_rope"]["cos_max_abs"] < 1e-5
        and rep["A_rope"]["sin_max_abs"] < 1e-5
    )
    return cos, sin, freqs


def _nonblock_dit(reader: _Reader, nonblock: dict, dev) -> H3DiT:
    with torch.device("meta"):
        dit = H3DiT(reader.cfg)
    missing, unexpected = dit.load_state_dict(nonblock, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"H3DiT: unexpected non-block keys {unexpected[:10]}")
    bad = [k for k in missing if not k.startswith("blocks.")]
    if bad:
        raise RuntimeError(f"H3DiT: non-block keys missing from the file {bad[:10]}")
    for name, child in dit.named_children():
        if name != "blocks":
            child.to(dev)
    dit.adaln_t_table = dit.adaln_t_table.to(dev)
    return dit


def part_b(dit: H3DiT, dev, rep: dict) -> None:
    t = torch.tensor([0.0, 1e-4, 0.1, 0.25, 0.5, 0.75, 0.999, 0.99999, 1.0], dtype=torch.float32, device=dev)
    mine = dit.adaln_embed(t)
    orc = MiniMaxH3ComfyPrunedTimeEmbedder(lambda: dit.adaln_t_table)(t, dtype=torch.float32)
    rep["B_adaln"] = {"t": t.tolist(), "max_abs": _maxabs(mine, orc), "exact": bool(torch.equal(mine, orc))}
    rep["B_pass"] = rep["B_adaln"]["max_abs"] < 1e-6


def _patch_comfy_attn(blocks) -> None:
    for b in blocks:
        b.attn.forward = types.MethodType(_comfy_attention_forward, b.attn)


def part_c(reader: _Reader, dit: H3DiT, nonblock: dict, layout, t_unique, time_index, dev, rep: dict) -> None:
    cfg = reader.cfg
    g = torch.Generator().manual_seed(SEED)
    text = torch.randn(L, cfg.text_dim, generator=g).to(torch.bfloat16).to(dev)

    # refine_text
    mine = dit.refine_text(text)
    with torch.device("meta"):
        cond = nn.Linear(cfg.text_dim, cfg.hidden_size, bias=True)
        refiner = MiniMaxH3TokenRefiner(
            cfg.token_refiner_layers, cfg.hidden_size, cfg.num_heads, cfg.head_dim, cfg.ffn_hidden, 1e-5, 1e-5, 1e-5
        )
    _patch_comfy_attn(refiner.blocks)
    cond.load_state_dict(
        {"weight": nonblock["condition_proj.weight"].to(dev, torch.bfloat16),
         "bias": nonblock["condition_proj.bias"].to(dev, torch.bfloat16)}, assign=True,
    )
    refiner.load_state_dict(
        {k[len("token_refiner."):]: v.to(dev, torch.bfloat16) for k, v in nonblock.items() if k.startswith("token_refiner.")},
        strict=True, assign=True,
    )
    cu = torch.tensor([0, L], dtype=torch.int32, device=dev)
    with torch.no_grad():
        orc = refiner(cond(text), cu_seqlens=cu, max_seqlen=L)
    rep["C_refine_text"] = {"rel": _rel(mine, orc), "max_abs": _maxabs(mine, orc), "dtype": str(mine.dtype)}
    del refiner, cond

    # embed_rows
    audio_rows = torch.randn(2 * A, cfg.audio_latents_dim, generator=g).to(dev)
    video_rows = torch.randn(T_LAT * GH * GW, cfg.latents_dim * 4, generator=g).to(dev)
    tok = dit.embed_rows(mine, audio_rows, video_rows)
    ref_tok = torch.cat([
        mine.to(torch.bfloat16),
        F.linear(audio_rows, nonblock["audio_patch_proj.weight"].to(dev).float(), nonblock["audio_patch_proj.bias"].to(dev).float()).to(torch.bfloat16),
        F.linear(video_rows, nonblock["video_patch_proj.weight"].to(dev).float(), nonblock["video_patch_proj.bias"].to(dev).float()).to(torch.bfloat16),
    ])
    rep["C_embed_rows"] = {"exact": bool(torch.equal(tok, ref_tok)), "shape": list(tok.shape), "dtype": str(tok.dtype)}

    # final_rows
    x = torch.randn(layout.total, cfg.hidden_size, generator=g).to(torch.bfloat16).to(dev)
    t_emb = dit.adaln_embed(t_unique)
    mv, ma = dit.final_rows(x, t_emb, time_index, layout)
    with torch.device("meta"):
        fl = MiniMaxH3FinalLayer(
            cfg.hidden_size, cfg.adaln_rank, 2 * cfg.hidden_size, cfg.latents_dim, cfg.audio_latents_dim, (1, 2, 2), 1e-5
        )
        fl.adaln_proj = _to_comfy_pruned_adaln_proj(fl.adaln_proj)
    sd = {}
    for k, v in nonblock.items():
        if not k.startswith("final_layer."):
            continue
        k2 = k[len("final_layer."):]
        dt = torch.bfloat16 if k2.startswith("norm.") else torch.float32
        sd[k2] = v.to(dev, dt)
    fl.load_state_dict(sd, strict=True, assign=True)
    for head in (fl.video_out, fl.audio_out):  # f32 heads fed the bf16 modulated rows, as diffusers does
        head.register_forward_pre_hook(lambda m, a: (a[0].to(m.weight.dtype),))
    with torch.no_grad():
        ov, oa = fl(x, t_emb=t_emb, inverse_indices=time_index)
    ov, oa = ov[layout.video], oa[layout.audio]
    rep["C_final_rows"] = {
        "video_rel": _rel(mv, ov), "audio_rel": _rel(ma, oa),
        "video_shape": list(mv.shape), "audio_shape": list(ma.shape), "dtype": str(mv.dtype),
    }
    del fl
    rep["C_pass"] = bool(
        rep["C_refine_text"]["rel"] < PART_TOL
        and rep["C_embed_rows"]["exact"]
        and rep["C_final_rows"]["video_rel"] < PART_TOL
        and rep["C_final_rows"]["audio_rel"] < PART_TOL
    )


def _quant_of(cfg, block: int) -> dict:
    p = f"blocks.{block}."
    return {k[len(p):]: v for k, v in cfg.quant.items() if k.startswith(p)}


def _dequant_block_sd(raw: dict, quant: dict, dev, *, adaln_dtype=torch.float32) -> dict:
    """raw block tensors (relative names) -> bf16 dequantized Linears, bf16 norms, adaln in ``adaln_dtype``."""
    out = {}
    for layer, spec in quant.items():
        if spec.format != "int8_tensorwise":
            raise RuntimeError(f"{layer}: format {spec.format}")
        if spec.convrot and spec.convrot_groupsize != 256:
            raise RuntimeError(f"{layer}: dequant.py supports ConvRot groupsize 256 only ({spec.convrot_groupsize})")
        scheme = "int8_convrot" if spec.convrot else "int8"
        w = raw[layer + ".weight"].to(dev)
        o, i = w.shape
        s = normalize_aux(scheme, "weight_scale", raw[layer + ".weight_scale"], o, i).to(dev)
        out[layer + ".weight"] = dequantize(scheme, w, {"weight_scale": s}, torch.bfloat16)
        if layer + ".bias" in raw:
            out[layer + ".bias"] = raw[layer + ".bias"].to(dev, torch.bfloat16)
    for k, v in raw.items():
        if k in out or k.endswith(".weight_scale") or any(k == q + ".weight" for q in quant):
            continue
        out[k] = v.to(dev, adaln_dtype if "adaln_proj." in k else torch.bfloat16)
    return out


def part_d(reader: _Reader, nonblock: dict, layout, t_unique, time_index, modality, cos, sin, freqs, dev, rep: dict) -> None:
    cfg = reader.cfg
    n = 2
    raw = {f"blocks.{b}.{k}": v for b in range(n) for k, v in reader.read(f"blocks.{b}.", strip=f"blocks.{b}.").items()}

    # mine: STRICT load = the state-dict names of H3DiT are the file's names
    def keep(name: str) -> bool:
        return not name.startswith("blocks.") or int(name.split(".")[1]) < n

    cfg2 = dataclasses.replace(cfg, num_layers=n, quant={k: v for k, v in cfg.quant.items() if keep(k)})
    with torch.device("meta"):
        mine = H3DiT(cfg2)
    swapped = swap_in_int8_linears(mine, cfg2.quant, bias_dtype_of=lambda name: torch.bfloat16)
    unknown = set(cfg.unknown)
    sd = {k: v for k, v in nonblock.items() if k not in unknown}
    sd.update(raw)
    mine.load_state_dict(sd, strict=True, assign=True)
    mine.to(dev)

    # oracle, run as DiffSynth runs it: bf16 everywhere
    with torch.device("meta"):
        orc = MiniMaxH3DiTComfyPruned(
            num_layers=n, token_refiner_num_layers=cfg.token_refiner_layers, hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_heads, attention_head_dim=cfg.head_dim, ffn_hidden_size=cfg.ffn_hidden,
            latents_dim=cfg.latents_dim, audio_latents_dim=cfg.audio_latents_dim, text_dim=cfg.text_dim,
            adaln_out_features=6 * cfg.hidden_size * 3, final_adaln_out_features=2 * cfg.hidden_size,
            rope_inv_freq_len=cfg.rope_inv_freq_len, adaln_curve_grid=cfg.adaln_grid, time_embed_dim=cfg.adaln_rank,
        )
    osd = {}
    for k, v in nonblock.items():
        if k in unknown:
            continue
        keep_f32 = k in ("adaln_t_table", "rope.inv_freq")
        osd[k] = v.to(dev, torch.float32 if keep_f32 else torch.bfloat16)
    for b in range(n):
        q = _quant_of(cfg, b)
        rb = {k[len(f"blocks.{b}."):]: v for k, v in raw.items() if k.startswith(f"blocks.{b}.")}
        for k, v in _dequant_block_sd(rb, q, dev, adaln_dtype=torch.bfloat16).items():
            osd[f"blocks.{b}.{k}"] = v
    missing, unexpected = orc.load_state_dict(osd, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"oracle load: missing {missing[:5]} unexpected {unexpected[:5]}")

    g = torch.Generator().manual_seed(SEED + 7)
    text = torch.randn(L, cfg.text_dim, generator=g).to(torch.bfloat16).to(dev)
    audio_rows = torch.randn(2 * A, cfg.audio_latents_dim, generator=g).to(dev)
    video_rows = torch.randn(T_LAT * GH * GW, cfg.latents_dim * 4, generator=g).to(dev)
    with torch.no_grad():
        tokens = mine.embed_rows(mine.refine_text(text), audio_rows, video_rows)
        mv, ma = mine(
            tokens, layout=layout, t_unique=t_unique, time_index=time_index, modality=modality,
            rope_cos=cos, rope_sin=sin,
        )
        s = layout.total
        vpos = torch.arange(layout.video.start, layout.video.stop, device=dev)
        apos = torch.arange(layout.audio.start, layout.audio.stop, device=dev)
        tpos = torch.arange(layout.text.start, layout.text.stop, device=dev)
        x_full = torch.zeros(1, s, video_rows.shape[1], device=dev, dtype=torch.bfloat16)
        x_full[0, vpos] = video_rows.to(torch.bfloat16)
        a_full = torch.zeros(1, s, audio_rows.shape[1], device=dev, dtype=torch.bfloat16)
        a_full[0, apos] = audio_rows.to(torch.bfloat16)
        coords = h3rope.build_coords(text_len=L, latent_frames=T_LAT, grid_h=GH, grid_w=GW, audio_latents=A)
        ov, oa = orc(
            x_full, a_full, coords[None].to(dev), t_unique, time_index,
            torch.ones(vpos.shape[0], device=dev, dtype=torch.bfloat16), modality, text,
            {"position_ids": vpos}, {"position_ids": apos}, {"position_ids": tpos}, {"position_ids": vpos},
            {"cu_seqlens_q": torch.tensor([0, s], dtype=torch.int32), "max_seqlen_q": s},
            {"cu_seqlens_q": torch.tensor([0, L], dtype=torch.int32), "max_seqlen_q": L},
        )
    rep["D_e2e_2blocks"] = {
        "int8_layers_swapped": swapped,
        "strict_load": True,
        "video_rel": _rel(mv, ov), "audio_rel": _rel(ma, oa),
        "video_shape": list(mv.shape), "audio_shape": list(ma.shape),
        "note": "oracle runs bf16 everywhere (DiffSynth's own precision, bf16 AdaLN); mine keeps the f32 islands",
    }
    rep["D_pass"] = bool(rep["D_e2e_2blocks"]["video_rel"] < E2E_TOL and rep["D_e2e_2blocks"]["audio_rel"] < E2E_TOL)
    del mine, orc, osd, sd, raw


def part_e(reader: _Reader, blocks: list[int], dit: H3DiT, layout, t_unique, time_index, modality, cos, sin, freqs, dev, rep: dict) -> None:
    cfg = reader.cfg
    t_emb = dit.adaln_embed(t_unique)
    adaln_index = time_index * 3 + modality
    cu = torch.tensor([0, layout.total], dtype=torch.int32, device=dev)
    rows = []
    for b in blocks:
        t1 = time.perf_counter()
        raw = reader.read(f"blocks.{b}.", strip=f"blocks.{b}.")
        quant = _quant_of(cfg, b)
        with torch.device("meta"):
            mine = H3DiTBlock(cfg.hidden_size, cfg.num_heads, cfg.head_dim, cfg.ffn_hidden, cfg.adaln_rank)
            mine_deq = H3DiTBlock(cfg.hidden_size, cfg.num_heads, cfg.head_dim, cfg.ffn_hidden, cfg.adaln_rank)
            orc = MiniMaxH3DiTBlock(
                cfg.hidden_size, cfg.num_heads, cfg.head_dim, cfg.ffn_hidden, cfg.adaln_rank,
                6 * cfg.hidden_size * 3, 1e-5, 1e-5,
            )
            orc.adaln_proj = _to_comfy_pruned_adaln_proj(orc.adaln_proj)
        _patch_comfy_attn([orc])
        swap_in_int8_linears(mine, quant, bias_dtype_of=lambda name: torch.bfloat16)
        mine.load_state_dict(raw, strict=True, assign=True)
        mine.to(dev)
        deq = _dequant_block_sd(raw, quant, dev)
        mine_deq.load_state_dict(deq, strict=True, assign=True)
        orc.load_state_dict(deq, strict=True, assign=True)

        g = torch.Generator().manual_seed(SEED + 100 + b)
        x = torch.randn(layout.total, cfg.hidden_size, generator=g).to(torch.bfloat16).to(dev)
        with torch.no_grad():
            y_mine = mine(x, t_emb, adaln_index, cos, sin)
            y_deq = mine_deq(x, t_emb, adaln_index, cos, sin)
            y_orc = orc(x, t_emb=t_emb, combined_indices=adaln_index, rope_freqs=freqs, cu_seqlens=cu, max_seqlen=layout.total)
        xf = x.double()
        row = {
            "block": b,
            "wiring_rel": _rel(y_deq.double() - xf, y_orc.double() - xf),
            "quant_rel": _rel(y_mine.double() - xf, y_orc.double() - xf),
            "out_rel": _rel(y_mine, y_orc),
            "finite": bool(torch.isfinite(y_mine).all()),
            "seconds": time.perf_counter() - t1,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del mine, mine_deq, orc, deq, raw, x, y_mine, y_deq, y_orc
        torch.cuda.empty_cache()
    rep["E_blocks"] = rows
    import statistics

    rep["E_worst"] = {
        "wiring_rel_max": max(r["wiring_rel"] for r in rows),
        "wiring_rel_median": statistics.median(r["wiring_rel"] for r in rows),
        "quant_rel_max": max(r["quant_rel"] for r in rows),
        "quant_rel_median": statistics.median(r["quant_rel"] for r in rows),
        "out_rel_max": max(r["out_rel"] for r in rows),
        "worst_quant_block": max(rows, key=lambda r: r["quant_rel"])["block"],
    }
    rep["E_pass"] = bool(
        all(r["finite"] and r["wiring_rel"] < WIRING_TOL and r["quant_rel"] < QUANT_TOL for r in rows)
    )


def _parse_blocks(spec: str, n: int) -> list[int]:
    if spec == "all":
        return list(range(n))
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(Path(__file__).with_name("g4_result.json")))
    ap.add_argument("--blocks", default="all", help='"all" or "a-b"')
    ap.add_argument("--skip-e2e", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda", 0)
    rep: dict = {
        "gate": "G4-dit",
        "weights": str(WEIGHTS),
        "tolerances": {"wiring": WIRING_TOL, "quant": QUANT_TOL, "e2e": E2E_TOL, "part": PART_TOL},
        "sequence": {"text": L, "latent_frames": T_LAT, "grid": [GH, GW], "audio_latents": A, "sigma_v": SIGMA_V, "sigma_a": SIGMA_A},
    }
    t0 = time.perf_counter()
    try:
        rep["backends"] = require_cuda_int8()
        reader = _Reader(WEIGHTS)
        rep["config"] = {f.name: getattr(reader.cfg, f.name) for f in dataclasses.fields(reader.cfg) if f.name not in ("quant", "unknown")}
        rep["config"]["quant_layers"] = len(reader.cfg.quant)
        rep["config"]["unknown"] = list(reader.cfg.unknown)
        layout = _layout()
        t_unique, time_index, modality = _step_indices(layout, dev)
        with torch.no_grad():
            cos, sin, freqs = part_a(reader, dev, rep)
            unknown = set(reader.cfg.unknown)  # h3_format's verdict: warned about and dropped
            nonblock = {k: v for k, v in reader.read_nonblock().items() if k not in unknown}
            dit = _nonblock_dit(reader, nonblock, dev)
            part_b(dit, dev, rep)
            part_c(reader, dit, nonblock, layout, t_unique, time_index, dev, rep)
            # keep only what part E needs on the GPU
            dit.token_refiner.to("meta")
            torch.cuda.empty_cache()
            if not args.skip_e2e:
                part_d(reader, nonblock, layout, t_unique, time_index, modality, cos, sin, freqs, dev, rep)
                torch.cuda.empty_cache()
            part_e(reader, _parse_blocks(args.blocks, reader.cfg.num_layers), dit, layout, t_unique, time_index, modality, cos, sin, freqs, dev, rep)
        rep["peak_vram_mb"] = torch.cuda.max_memory_allocated(dev) / 2**20
        rep["pass"] = bool(
            rep["A_pass"] and rep["B_pass"] and rep["C_pass"] and rep.get("D_pass", True) and rep["E_pass"]
        )
    except Exception as exc:  # noqa: BLE001 — the report must say why
        import traceback

        rep["pass"] = False
        rep["error"] = f"{type(exc).__name__}: {exc}"
        rep["traceback"] = traceback.format_exc()
    rep["seconds"] = time.perf_counter() - t0
    Path(args.report).write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    summary = {k: rep[k] for k in rep if k.endswith("_pass") or k in ("pass", "error", "E_worst", "seconds")}
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
