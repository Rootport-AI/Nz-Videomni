"""M0: ConvRot int8 activation quantization vs int8_linear, single-kernel microbench (台帳 §1-105 ①).

Product code is not touched; this only calls comfy-kitchen 0.2.37 directly.
Writes m0_result.json next to this file.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch
import comfy_kitchen
from comfy_kitchen.backends import cuda as ckc

HERE = Path(__file__).resolve().parent
K = 5376
N = 1024
GS = 256
WARMUP = 10
ITERS = 100


def time_fn(fn) -> dict:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ms = []
    for _ in range(ITERS):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ms.append(a.elapsed_time(b))
    return {"median_ms": round(statistics.median(ms), 4), "min_ms": round(min(ms), 4),
            "max_ms": round(max(ms), 4), "mean_ms": round(statistics.fmean(ms), 4), "iters": ITERS,
            "warmup": WARMUP}


def bench(S: int, seed: int) -> dict:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((S, K), generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    w = torch.randint(-127, 128, (N, K), generator=g, device="cuda", dtype=torch.int8)
    ws = (torch.rand((N, 1), generator=g, device="cuda", dtype=torch.float32) * 1e-3 + 1e-4)
    torch.cuda.synchronize()

    quant = time_fn(lambda: ckc.quantize_int8_rowwise_convrot64(x, GS))
    lin = time_fn(lambda: comfy_kitchen.int8_linear(x, w, ws, None, out_dtype=torch.bfloat16,
                                                    convrot=True, convrot_groupsize=GS))

    # determinism: two calls on the same input
    q1, s1 = ckc.quantize_int8_rowwise_convrot64(x, GS)
    q2, s2 = ckc.quantize_int8_rowwise_convrot64(x, GS)
    torch.cuda.synchronize()
    det = {"q_equal": bool(torch.equal(q1, q2)), "scale_equal": bool(torch.equal(s1, s2)),
           "q_dtype": str(q1.dtype), "q_shape": list(q1.shape), "scale_dtype": str(s1.dtype),
           "scale_shape": list(s1.shape)}
    # two int8_linear calls (same kernel path as the product forward_rows)
    o1 = comfy_kitchen.int8_linear(x, w, ws, None, out_dtype=torch.bfloat16, convrot=True, convrot_groupsize=GS)
    o2 = comfy_kitchen.int8_linear(x, w, ws, None, out_dtype=torch.bfloat16, convrot=True, convrot_groupsize=GS)
    torch.cuda.synchronize()
    det["int8_linear_out_equal"] = bool(torch.equal(o1, o2))
    del q1, q2, s1, s2, o1, o2

    in_bytes = S * K * 2
    out_bytes = S * K + S * 4
    return {
        "S": S, "K": K, "N": N, "seed": seed,
        "quantize_int8_rowwise_convrot64": quant,
        "int8_linear_N1024": lin,
        "ratio_quant_over_linear_median": round(quant["median_ms"] / lin["median_ms"], 4),
        "quant_effective_GBps_median": round((in_bytes + out_bytes) / (quant["median_ms"] * 1e-3) / 1e9, 1),
        "quant_bytes_moved_MB": round((in_bytes + out_bytes) / 1e6, 1),
        "quant_x1000_seconds_median": round(quant["median_ms"] * 1000 / 1e3, 3),
        "determinism": det,
    }


def main() -> int:
    assert torch.cuda.is_available()
    out = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": torch.__version__,
           "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
           "comfy_kitchen": getattr(comfy_kitchen, "__version__", None),
           "fused_shared_memory_fits": None, "runs": []}
    probe = torch.empty((8, K), device="cuda", dtype=torch.bfloat16)
    try:
        out["fused_shared_memory_fits"] = bool(ckc._convrot_fused_shared_memory_fits(probe, K, GS))
    except Exception as exc:  # noqa: BLE001
        out["fused_shared_memory_fits"] = f"error: {exc!r}"
    del probe
    for S in (39764, 89388):
        r = bench(S, seed=1234)
        out["runs"].append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        torch.cuda.empty_cache()
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (HERE / "m0_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
