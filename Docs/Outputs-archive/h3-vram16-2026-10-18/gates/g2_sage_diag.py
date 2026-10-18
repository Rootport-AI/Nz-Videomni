"""Gate A-G2 (VRAM 16GB mode): every DiT block, on vs off, at the product shape (GPU).

The g4_compare.py part E way (Docs/Outputs-archive/h3-gate1/): one block at a
time is built on meta, its int8 tensors read from the real file, moved to the
GPU, and run on the SAME synthetic input twice -- ``block.vram16 = None`` (the
off path) and ``block.vram16 = Vram16Plan()`` (the product defaults of
engine_h3/constants.py) -- with sage and with sdpa. No oracle.

Product shape: I2V 896x1152x124f -> S = 39,764 = text 1,046 | cond 1,008 |
audio 414 | video 37,296 (one step's t: sigma_v 0.9, sigma_a 0.75).

    set PYTHONUTF8=1
    .venv-engine-h3\\Scripts\\python.exe outputs\\h3-vram16-gates\\g2_block_equal.py [--blocks 0-49] [--report P]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # Nz-Videomni (outputs/h3-vram16-gates/..)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from engine.sft_quant.sft_reader import read_tensors  # noqa: E402
from engine_h3 import rope as h3rope  # noqa: E402
from engine_h3.dit import H3DiTBlock, H3SeqLayout  # noqa: E402
from engine_h3.lowvram import Vram16Plan  # noqa: E402
from engine_h3.pipeline_h3 import modality_rows, time_rows  # noqa: E402
from engine_h3.quant_linear import require_cuda_int8, swap_in_int8_linears  # noqa: E402
from h3_format import inspect_h3_dit  # noqa: E402

WEIGHTS = ROOT / "models" / "MiniMaxH3" / "Weights" / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TEXT, COND, T_LAT, GH, GW, A_LAT = 1046, 1008, 37, 36, 28, 207
SIGMA_V, SIGMA_A = 0.9, 0.75
SEED = 2026


def read(path: str, header, prefix: str, names: list[str]) -> dict:
    out = {}
    for k, t in read_tensors(path, [prefix + n for n in names], header):
        out[k[len(prefix):]] = t
    return out


def diff(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> dict:
    """a = on, b = off; float64."""
    a64, b64, x64 = a.double(), b.double(), x.double()
    d = a64 - b64
    return {
        "max_abs": float(d.abs().max()),
        "rel_out": float(d.norm() / b64.norm().clamp_min(1e-300)),
        "rel_delta": float(d.norm() / (b64 - x64).norm().clamp_min(1e-300)),
        "elements_differ": int((a != b).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="0-49")
    ap.add_argument("--report", default=str(HERE / "g2_block_equal.json"))
    ap.add_argument("--repeat", action="store_true",
                    help="also off vs off and on vs on (is the backend itself run-to-run identical?)")
    args = ap.parse_args()
    a, _, b = args.blocks.partition("-")
    blocks = list(range(int(a), int(b or a) + 1))

    dev = torch.device("cuda", 0)
    rep: dict = {"gate": "A-G2 blocks on/off", "weights": str(WEIGHTS), "plan": None, "rows": []}
    t0 = time.perf_counter()
    require_cuda_int8()
    path = str(WEIGHTS)
    header, cfg = inspect_h3_dit(path)
    p = cfg.prefix
    plan = Vram16Plan()
    rep["plan"] = {"head_group": plan.head_group, "row_chunk": plan.row_chunk}
    top = read(path, header, p, ["adaln_t_table", "rope.inv_freq"])
    table = top["adaln_t_table"].to(dev, torch.float32)
    inv = top["rope.inv_freq"].to(torch.float32)

    layout = H3SeqLayout.make(text_rows=TEXT, cond_rows=COND, audio_rows=2 * A_LAT, video_rows=T_LAT * GH * GW)
    rep["sequence"] = {"text": TEXT, "cond": COND, "audio": 2 * A_LAT, "video": T_LAT * GH * GW, "total": layout.total}
    coords = h3rope.build_coords(text_len=TEXT, latent_frames=T_LAT, grid_h=GH, grid_w=GW,
                                 audio_latents=A_LAT, cond_frames=1)
    cos, sin = h3rope.rope_tables(coords, inv)
    cos, sin = cos.to(dev), sin.to(dev)
    t_unique, time_index = time_rows(layout, 1.0 - SIGMA_V, 1.0 - SIGMA_A, device=dev)
    modality = modality_rows(layout, torch.ones(TEXT, dtype=torch.long, device=dev))
    # H3DiT.adaln_embed, on the table alone
    g_ = table.shape[0]
    pos = t_unique.float().clamp(0.0, 1.0) * (g_ - 1)
    i0 = pos.floor().long().clamp(max=g_ - 2)
    t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
    adaln_index = time_index * 3 + modality
    rep["m"] = int(t_unique.numel())

    from engine_h3 import dit as D
    real_attn = D._attention
    for blk in blocks:
        bp = f"blocks.{blk}."
        names = [k[len(p) + len(bp):] for k in header.tensors
                 if k.startswith(p + bp) and not k.endswith(".comfy_quant")]
        raw = read(path, header, p + bp, names)
        quant = {k[len(bp):]: v for k, v in cfg.quant.items() if k.startswith(bp)}
        with torch.device("meta"):
            block = H3DiTBlock(cfg.hidden_size, cfg.num_heads, cfg.head_dim, cfg.ffn_hidden, cfg.adaln_rank)
        swap_in_int8_linears(block, quant, bias_dtype_of=lambda name: torch.bfloat16)
        block.load_state_dict(raw, strict=True, assign=True)
        block.to(dev)
        block.adaln_proj.linear.to(torch.float32)
        block.requires_grad_(False)
        g = torch.Generator().manual_seed(SEED + blk)
        x = torch.randn(layout.total, cfg.hidden_size, generator=g).to(torch.bfloat16).to(dev)
        calls = []

        def spy(q, k, v, backend):
            out = real_attn(q, k, v, backend)
            calls.append((q.clone(), k.clone(), v.clone(), out.clone()))
            return out

        D._attention = spy
        block.attn.attention_backend = "sage"
        with torch.no_grad():
            block.vram16 = None
            block(x, t_emb, adaln_index, cos, sin)
            block.vram16 = plan
            block(x, t_emb, adaln_index, cos, sin)
            block.vram16 = None
        D._attention = real_attn
        fq, fk, fv, fo = calls[0]
        row = {"block": blk, "groups": []}
        gsz = plan.head_group
        with torch.no_grad():
            for j, (q, k, v, o) in enumerate(calls[1:]):
                h0, h1 = j * gsz, j * gsz + q.shape[1]
                e = {"heads": [h0, h1], "q_equal": bool(torch.equal(q, fq[:, h0:h1])),
                     "k_equal": bool(torch.equal(k, fk[:, h0:h1])), "v_equal": bool(torch.equal(v, fv[:, h0:h1])),
                     "out_equal": bool(torch.equal(o, fo[:, h0:h1]))}
                # the same group's inputs through sage again, but inside a 56-head call (cut from the full)
                if not e["out_equal"]:
                    e["out_max_abs"] = float((o.double() - fo[:, h0:h1].double()).abs().max())
                    e["out_rel"] = float((o.double() - fo[:, h0:h1].double()).norm() / fo[:, h0:h1].double().norm())
                    alone = real_attn(fq[:, h0:h1].contiguous(), fk[:, h0:h1].contiguous(), fv[:, h0:h1].contiguous(), "sage")
                    e["full_inputs_sliced_then_grouped_equals_on"] = bool(torch.equal(alone, o))
                row["groups"].append(e)
        print(json.dumps(row), flush=True)
        rep["rows"].append(row)
        del block, x, calls
        torch.cuda.empty_cache()
    Path(args.report).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return 0


def _unused():
    pass


if __name__ == "__main__":
    sys.exit(main())
