"""Gate G2: comfy-kitchen int8_linear smoke test through H3Int8Linear (GPU).

Run (only after the supervisor's GPU go-ahead):
    set PYTHONPATH=<repo root>
    .venv-engine-h3\\Scripts\\python.exe outputs\\h3-gate1\\g2_int8_smoke.py [--report P.json]

What it checks
  1. require_cuda_int8() passes (CUDA backend has int8_linear; triton and
     eager are disabled afterwards, so a call off the CUDA path RAISES).
  2. H3Int8Linear(5376 -> 28672, ConvRot 256) built on meta, filled with
     load_state_dict(assign=True), moved to the GPU, runs on x [9600, 5376] bf16
     (the first-stage sequence size).
  3. The result matches the same weight dequantized by
     engine/sft_quant/dequant.dequantize("int8_convrot") and applied with a
     float32 F.linear, over every row INCLUDING THE TAIL (a kernel that
     mishandles the last M-tile shows up only there).

Tolerance: the reference keeps the activation in float32 (W8A16-like) while
the kernel quantizes it per row to int8 after the Hadamard rotation (W8A8).
For Gaussian rows of K=5376 that alone is ~0.9e-2 relative (max ~3.9 sigma,
step max/127, noise step/sqrt(12)); a wiring / layout error is O(1). The
pass line is REL_TOL below; the measured value is recorded either way.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from engine.sft_quant.dequant import dequantize, normalize_aux  # noqa: E402
from engine_h3.quant_linear import H3Int8Linear, require_cuda_int8  # noqa: E402
from h3_format import QuantSpec  # noqa: E402

N, K, M = 28672, 5376, 9600
M_ODD = 9537  # not a multiple of 128 / 64: exercises a partial last tile
TAIL_ROWS = 64
REL_TOL = 2e-2
SEED = 1234


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(Path(__file__).with_name("g2_result.json")))
    args = ap.parse_args()

    dev = torch.device("cuda", 0)
    report: dict = {"gate": "G2", "shape": {"N": N, "K": K, "M": M}, "rel_tol": REL_TOL}
    t0 = time.perf_counter()
    try:
        backends = require_cuda_int8()
        report["backends"] = backends
        report["device"] = torch.cuda.get_device_name(dev)

        g = torch.Generator().manual_seed(SEED)
        w = torch.randint(-127, 128, (N, K), generator=g, dtype=torch.int8)
        # Row scales of a realistic magnitude: |W| ~ 0.02 * 127 / 64.
        scale = (torch.rand(N, 1, generator=g, dtype=torch.float32) + 0.5) * (0.02 / 64)
        x = torch.randn(M, K, generator=g, dtype=torch.float32).to(torch.bfloat16)

        spec = QuantSpec(format="int8_tensorwise", convrot=True, convrot_groupsize=256)
        with torch.device("meta"):
            lin = H3Int8Linear(K, N, bias=False, spec=spec)
        lin.load_state_dict({"weight": w, "weight_scale": scale}, assign=True)
        lin.to(dev)
        xd = x.to(dev)

        with torch.no_grad():
            torch.cuda.synchronize(dev)
            t1 = time.perf_counter()
            out = lin(xd)
            torch.cuda.synchronize(dev)
            report["kernel_seconds_first_call"] = time.perf_counter() - t1
            t1 = time.perf_counter()
            out2 = lin(xd)
            torch.cuda.synchronize(dev)
            report["kernel_seconds_second_call"] = time.perf_counter() - t1
            report["repeat_bit_identical"] = bool(torch.equal(out, out2))
            del out2
            report["out_dtype"] = str(out.dtype)
            report["out_shape"] = list(out.shape)
            report["out_finite"] = bool(torch.isfinite(out).all())

            # Reference: dequantize -> float32 F.linear, in row chunks.
            aux = {"weight_scale": normalize_aux("int8_convrot", "weight_scale", scale, N, K).to(dev)}
            wd = dequantize("int8_convrot", w.to(dev), aux, torch.float32)
            num = torch.zeros((), dtype=torch.float64, device=dev)
            den = torch.zeros((), dtype=torch.float64, device=dev)
            row_rel = torch.empty(M, dtype=torch.float32, device=dev)
            chunk = 1024
            for s in range(0, M, chunk):
                ref = F.linear(xd[s:s + chunk].float(), wd)
                diff = out[s:s + chunk].float() - ref
                num += diff.double().pow(2).sum()
                den += ref.double().pow(2).sum()
                row_rel[s:s + chunk] = diff.norm(dim=1) / ref.norm(dim=1).clamp_min(1e-30)
                del ref, diff
            ref_tail = F.linear(xd[-TAIL_ROWS:].float(), wd)
            report["tail_rel"] = _rel(out[-TAIL_ROWS:], ref_tail)
            # Same check with M off the 128-row tile grid (a partial last tile).
            out_odd = lin(xd[:M_ODD])
            report["odd_m"] = M_ODD
            report["odd_m_tail_rel"] = _rel(out_odd[-TAIL_ROWS:], F.linear(xd[M_ODD - TAIL_ROWS:M_ODD].float(), wd))
            report["odd_m_rows_match_full_run"] = bool(torch.equal(out_odd, out[:M_ODD]))
            del out_odd
            report["rel_global"] = float((num / den).sqrt())
            report["row_rel_max"] = float(row_rel.max())
            report["row_rel_mean"] = float(row_rel.mean())
            report["row_rel_max_tail"] = float(row_rel[-TAIL_ROWS:].max())
            report["peak_vram_mb"] = torch.cuda.max_memory_allocated(dev) / 2**20

        ok = (
            report["out_finite"]
            and report["rel_global"] < REL_TOL
            and report["tail_rel"] < REL_TOL
            and report["odd_m_tail_rel"] < REL_TOL
            and report["row_rel_max"] < 2 * REL_TOL
        )
        report["pass"] = bool(ok)
    except Exception as exc:  # noqa: BLE001 — the report must say why
        import traceback

        report["pass"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    report["seconds"] = time.perf_counter() - t0
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k not in ("traceback", "backends")}, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
