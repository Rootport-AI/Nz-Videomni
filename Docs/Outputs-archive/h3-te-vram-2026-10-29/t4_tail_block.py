r"""§1-116 T4: is the N=6,148 mismatch (4 rows) the very small last query block? (hypothesis H-tail)

Run from the repo root (one process per N):

    set PYTHONPATH=%CD%
    set PYTHONUTF8=1
    .venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-te-vram-2026-10-29\t4_tail_block.py --n 6148 --report <json>
    ... --n 70 --block 16 --cpu      (dry run, no GPU)

Steps per N: (1) ref = stock F.sdpa(is_causal=True, enable_gqa=True); (2) the product's blocked_causal_attention
vs ref, with the differing rows; (3) tail sweep: the last T rows as one block (math, absolute-position mask) vs ref;
(4) position check: 4 rows at 1024..1028; (5) fold check: the last block+4 rows (516) as one block.
Inputs: bf16 q (1,64,N,128), k/v (1,8,N,128) from torch.randn(generator seed 0); scale = 128**-0.5.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

HEADS, KV_HEADS, HEAD_DIM = 64, 8, 128
TAILS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 51, 64, 96, 128, 256, 512]


def block_attn(q, k, v, s, e, scale):
    """Rows s..e as ONE block, as the product does (math kernel, mask from absolute positions)."""
    kv_idx = torch.arange(k.shape[2], device=q.device)
    mask = torch.arange(s, e, device=q.device)[:, None] >= kv_idx[None, :]
    with sdpa_kernel([SDPBackend.MATH]):
        return F.scaled_dot_product_attention(q[:, :, s:e], k, v, attn_mask=mask, is_causal=False,
                                              enable_gqa=True, scale=scale)


def parse_tails(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, z = part.split("-", 1)
            out.extend(range(int(a), int(z) + 1))
        else:
            out.append(int(part))
    return out


def cmp(out, ref):
    return {"equal": bool(torch.equal(out, ref)),
            "max_abs_diff": float((out.float() - ref.float()).abs().max()),
            "mismatch_elements": int((out != ref).sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--block", type=int, default=None, help="product block (default: the product's 512)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--report")
    ap.add_argument("--tails", default=None, help="comma-separated heights, ranges allowed (e.g. 17-23,64-575)")
    ap.add_argument("--skip-product", action="store_true", help="skip the product blocked_causal_attention step")
    args = ap.parse_args()
    n = args.n
    tails = parse_tails(args.tails) if args.tails else TAILS
    if not args.skip_product:
        from engine_h3.text_encoder import blocked_causal_attention

    dev = torch.device("cpu" if args.cpu else "cuda")
    if not args.cpu:
        from engine_h3.pipeline_h3 import require_fp32_matmul
        require_fp32_matmul()
    scale = HEAD_DIM ** -0.5
    block = args.block or 512
    rec: dict = {"n": n, "block": block, "device": str(dev), "torch": torch.__version__, "scale": scale,
                 "tail_rows_of_last_block": n % block or block}
    if not args.cpu:
        rec["device_name"] = torch.cuda.get_device_name(0)
        rec["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
        rec["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    ok = True
    try:
        g = torch.Generator().manual_seed(0)
        q = torch.randn(1, HEADS, n, HEAD_DIM, generator=g, dtype=torch.bfloat16).to(dev)
        k = torch.randn(1, KV_HEADS, n, HEAD_DIM, generator=g, dtype=torch.bfloat16).to(dev)
        v = torch.randn(1, KV_HEADS, n, HEAD_DIM, generator=g, dtype=torch.bfloat16).to(dev)
        if not args.cpu:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True, scale=scale)
        if not args.cpu:
            torch.cuda.synchronize()
            rec["ref_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            rec["ref_peak_over_inputs_mb"] = round((torch.cuda.max_memory_allocated() - base) / 2**20, 1)
            rec["ref_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        rec["ref_seconds"] = round(time.perf_counter() - t0, 2)
        print(f"ref done {rec['ref_seconds']} s", flush=True)

        if not args.skip_product:
            prod = blocked_causal_attention(q, k, v, scale=scale, **({"block": args.block} if args.block else {}))
            rec["product"] = cmp(prod, ref)
            rows = (prod != ref).any(dim=3).any(dim=1).any(dim=0).nonzero().flatten().tolist()
            rec["product"]["differing_rows"] = rows
            rec["product"]["differing_row_count"] = len(rows)
            print("product", rec["product"], flush=True)
            del prod

        sweep = {}
        for t in tails:
            if t > n:
                continue
            sweep[str(t)] = cmp(block_attn(q, k, v, n - t, n, scale), ref[:, :, n - t:])
            print("tail", t, sweep[str(t)], flush=True)
        rec["tail_sweep"] = sweep

        s0 = min(1024, max(0, n - 8))
        rec["position_check"] = {"rows": [s0, s0 + 4], **cmp(block_attn(q, k, v, s0, s0 + 4, scale), ref[:, :, s0:s0 + 4])}
        t = min(block + 4, n)
        rec["fold_check"] = {"rows": [n - t, n], **cmp(block_attn(q, k, v, n - t, n, scale), ref[:, :, n - t:])}
        print("position", rec["position_check"], "fold", rec["fold_check"], flush=True)
    except Exception as exc:  # noqa: BLE001
        ok = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
    rec["ok"] = ok
    report = Path(args.report or HERE / f"t4_tail_block_n{n}.json")
    text = json.dumps(rec, indent=2, ensure_ascii=False, default=str)
    report.write_text(text, encoding="utf-8")
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
