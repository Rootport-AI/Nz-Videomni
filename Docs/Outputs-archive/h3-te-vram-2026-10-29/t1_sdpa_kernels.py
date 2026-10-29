r"""§1-116 T1: one attention call of the H3 text encoder's shape, per SDPA kernel (no weights).

Run from the repo root (one condition = one process; run_t1.ps1 does the whole list):

    set PYTHONPATH=%CD%
    set PYTHONUTF8=1
    .venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-te-vram-2026-10-29\t1_sdpa_kernels.py --n 3130 --kernel today --save outputs\h3-te-vram-2026-10-29\t1_today_n3130.pt
    ... --kernel math --n 3130 --baseline outputs\h3-te-vram-2026-10-29\t1_today_n3130.pt
    ... --selfcheck            (CPU only: block-mask logic, repeat_interleave == HF repeat_kv, SDPAParams signature)

Shapes: q (1, 64, N, 128), k / v (1, 8, N, 128), bf16, ``torch.manual_seed(0)``,
drawn on the CPU then moved (every kernel sees the same input).

Kernels (plan §4):
  today  F.sdpa(q, k, v, is_causal=True, enable_gqa=True), default backend choice (= Hugging Face's call)
  math   the same call inside sdpa_kernel([MATH])
  probe  can_use_efficient_attention / can_use_cudnn_attention with SDPAParams(q,k,v,None,0.0,True,True), debug=True
  f0     sdpa_kernel([CUDNN_ATTENTION]), is_causal=True, enable_gqa=True
  f1     k / v repeat_interleave(8, dim=1) (= HF repeat_kv), sdpa_kernel([EFFICIENT_ATTENTION]), is_causal=True
  g256 / g512   math forced, query blocks of Bq rows against the FULL k / v with a bool mask slice
         arange(s, e)[:, None] >= arange(N)[None, :] (True = attend), enable_gqa=True, is_causal=False

The JSON report goes to --report (default: t1_<kernel>_n<N>.json next to this script).
Exit code 0 = the kernel ran (or the probe answered); 1 = it raised (the reason is in "error").
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import statistics
import sys
import time
import traceback
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

HEADS, KV_HEADS, HEAD_DIM = 64, 8, 128
GROUPS = HEADS // KV_HEADS
KERNELS = ("today", "math", "probe", "f0", "f1", "g256", "g512")


# -- the attention variants (also imported by t2_te_ab.py) ---------------------


def g_blocked_attention(q, k, v, bq: int, *, scale=None):
    """Causal GQA attention, query rows in blocks of ``bq`` against the FULL k / v.

    The mask slice keeps the rows' ABSOLUTE positions (``is_causal=True`` on a
    block would align the triangle top-left, which is wrong for any block but
    the first). The caller forces the backend (math). The last block may be
    shorter than ``bq``.
    """
    n_q, n_kv = q.shape[2], k.shape[2]
    kv_idx = torch.arange(n_kv, device=q.device)
    outs = []
    for s in range(0, n_q, bq):
        e = min(s + bq, n_q)
        mask = torch.arange(s, e, device=q.device)[:, None] >= kv_idx[None, :]   # (Bq, N) -> (1, 64, Bq, N)
        outs.append(F.scaled_dot_product_attention(
            q[:, :, s:e], k, v, attn_mask=mask, is_causal=False, enable_gqa=True, scale=scale,
        ))
    return torch.cat(outs, dim=2)


def repeat_kv_heads(x, groups: int):
    """k / v from (B, kv_heads, N, D) to (B, kv_heads * groups, N, D): head h reads kv head h // groups."""
    return x.repeat_interleave(groups, dim=1)


def run_kernel(kernel: str, q, k, v, *, scale=None):
    """One forward of ``kernel`` (not probe). Returns (B, H, N, D)."""
    if kernel == "today":
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True, scale=scale)
    if kernel == "math":
        with sdpa_kernel([SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True, scale=scale)
    if kernel == "f0":
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True, scale=scale)
    if kernel == "f1":
        groups = q.shape[1] // k.shape[1]
        kr, vr = repeat_kv_heads(k, groups), repeat_kv_heads(v, groups)
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            return F.scaled_dot_product_attention(q, kr, vr, is_causal=True, scale=scale)
    if kernel in ("g256", "g512"):
        with sdpa_kernel([SDPBackend.MATH]):
            return g_blocked_attention(q, k, v, int(kernel[1:]), scale=scale)
    raise ValueError(kernel)


# -- comparison ------------------------------------------------------------------


def compare(out_cpu: torch.Tensor, base: torch.Tensor) -> dict:
    rec = {"shape_match": list(out_cpu.shape) == list(base.shape), "dtype_match": out_cpu.dtype == base.dtype}
    if not rec["shape_match"]:
        return rec
    d = (out_cpu.float() - base.float()).abs()
    rec.update({
        "torch_equal": bool(torch.equal(out_cpu, base)),
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "mismatch_elements": int((out_cpu != base).sum()),
    })
    return rec


# -- environment record ------------------------------------------------------------


def sdp_flags() -> dict:
    cb = torch.backends.cuda
    rec = {}
    for name in ("is_flash_attention_available", "flash_sdp_enabled", "mem_efficient_sdp_enabled",
                 "cudnn_sdp_enabled", "math_sdp_enabled", "fp16_bf16_reduction_math_sdp_allowed"):
        fn = getattr(cb, name, None)
        try:
            rec[name] = bool(fn()) if fn is not None else None
        except Exception as exc:  # noqa: BLE001
            rec[name] = f"error: {exc}"
    try:
        order = torch._C._get_sdp_priority_order()
        rec["sdp_priority_order"] = [int(i) for i in order]
        rec["sdp_priority_order_names"] = [SDPBackend(int(i)).name for i in order]
    except Exception as exc:  # noqa: BLE001
        rec["sdp_priority_order"] = f"unavailable: {exc}"
    rec["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    rec["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    rec["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    rec["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    return rec


def env_record() -> dict:
    rec = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "python": platform.python_version(),
           "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}
    if torch.cuda.is_available():
        rec["device_name"] = torch.cuda.get_device_name(0)
        rec["device_total_mb"] = int(torch.cuda.get_device_properties(0).total_memory // 2**20)
    return rec


def sdpa_params_signature() -> str:
    from torch.backends.cuda import SDPAParams

    doc = (SDPAParams.__init__.__doc__ or "").strip().splitlines()
    return doc[0] if doc else repr(inspect.signature(SDPAParams))


def make_inputs(n: int, device) -> tuple:
    torch.manual_seed(0)
    q = torch.randn(1, HEADS, n, HEAD_DIM, dtype=torch.bfloat16)
    k = torch.randn(1, KV_HEADS, n, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(1, KV_HEADS, n, HEAD_DIM, dtype=torch.bfloat16)
    return q.to(device), k.to(device), v.to(device)


# -- probe -----------------------------------------------------------------------


def probe(q, k, v) -> dict:
    from torch.backends.cuda import SDPAParams, can_use_cudnn_attention, can_use_efficient_attention

    params = SDPAParams(q, k, v, None, 0.0, True, True)   # (q, k, v, attn_mask, dropout, is_causal, enable_gqa)
    rec = {"sdpa_params_signature": sdpa_params_signature(),
           "params": {"is_causal": bool(params.is_causal), "enable_gqa": bool(params.enable_gqa),
                      "dropout": float(params.dropout), "attn_mask_is_none": params.attn_mask is None}}
    for name, fn in (("efficient", can_use_efficient_attention), ("cudnn", can_use_cudnn_attention)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                ok = bool(fn(params, True))
                err = None
            except Exception as exc:  # noqa: BLE001
                ok, err = None, f"{type(exc).__name__}: {exc}"
        rec[name] = {"can_use": ok, "error": err, "warnings": [str(w.message) for w in caught]}
    return rec


# -- main --------------------------------------------------------------------------


def main_gpu(args) -> int:
    from engine_h3.pipeline_h3 import require_fp32_matmul

    require_fp32_matmul()   # the pipeline's check before encode (TF32 would change the math kernel's sums)
    dev = torch.device("cuda")
    rec: dict = {"kernel": args.kernel, "n": args.n, "shape": {"q": [1, HEADS, args.n, HEAD_DIM],
                 "kv": [1, KV_HEADS, args.n, HEAD_DIM], "dtype": "bfloat16"},
                 "env": env_record(), "sdp": sdp_flags(), "warmup": args.warmup, "repeat": args.repeat}
    ok = True
    q, k, v = make_inputs(args.n, dev)
    torch.cuda.synchronize()
    rec["inputs_alloc_mb"] = round(torch.cuda.memory_allocated() / 2**20, 1)
    torch.cuda.reset_peak_memory_stats()
    try:
        if args.kernel == "probe":
            rec["probe"] = probe(q, k, v)
        else:
            out = None
            for _ in range(args.warmup):
                out = run_kernel(args.kernel, q, k, v)
                torch.cuda.synchronize()
                del out
                out = None
            times = []
            for _ in range(max(1, args.repeat)):
                if out is not None:
                    del out
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = run_kernel(args.kernel, q, k, v)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
            rec["ms"] = round(statistics.median(times), 3)
            rec["ms_all"] = [round(t, 3) for t in times]
            rec["peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            rec["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
            rec["peak_alloc_over_inputs_mb"] = round(rec["peak_alloc_mb"] - rec["inputs_alloc_mb"], 1)
            out_cpu = out.detach().to("cpu")
            del out
            rec["output"] = {"shape": list(out_cpu.shape), "dtype": str(out_cpu.dtype),
                             "finite": bool(torch.isfinite(out_cpu.float()).all())}
            if args.save:
                Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                torch.save(out_cpu, args.save)
                rec["saved"] = args.save
            if args.baseline:
                base = torch.load(args.baseline)
                rec["baseline"] = args.baseline
                rec["vs_baseline"] = compare(out_cpu, base)
    except Exception as exc:  # noqa: BLE001 -- the reason goes into the JSON
        ok = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
        rec["peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        rec["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    rec["ok"] = ok
    report = Path(args.report or HERE / f"t1_{args.kernel}_n{args.n}.json")
    text = json.dumps(rec, indent=2, ensure_ascii=False, default=str)
    report.write_text(text, encoding="utf-8")
    print(text)
    return 0 if ok else 1


# -- self-check (CPU) ----------------------------------------------------------------


def _line(results: list, ok: bool, what: str) -> None:
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {what}")


def selfcheck() -> int:
    from transformers.integrations.sdpa_attention import repeat_kv

    results: list[bool] = []
    torch.manual_seed(0)
    for n in (300, 1100):
        for dtype, tol in ((torch.float32, 2e-6), (torch.bfloat16, 2e-2)):
            q = torch.randn(1, HEADS, n, HEAD_DIM, dtype=dtype)
            k = torch.randn(1, KV_HEADS, n, HEAD_DIM, dtype=dtype)
            v = torch.randn(1, KV_HEADS, n, HEAD_DIM, dtype=dtype)
            with sdpa_kernel([SDPBackend.MATH]):
                full = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
                rep = F.scaled_dot_product_attention(q, repeat_kv_heads(k, GROUPS), repeat_kv_heads(v, GROUPS),
                                                     is_causal=True)
            for bq in (128, 512):
                with sdpa_kernel([SDPBackend.MATH]):
                    blk = g_blocked_attention(q, k, v, bq)
                d = float((blk.float() - full.float()).abs().max())
                _line(results, blk.shape == full.shape and d <= tol,
                      f"g-block vs is_causal math  N={n} Bq={bq} {str(dtype)[6:]}: max|d|={d:.3g} "
                      f"(tol {tol:g}) torch.equal={bool(torch.equal(blk, full))}")
            d = float((rep.float() - full.float()).abs().max())
            _line(results, d <= tol, f"repeat_interleave(8) no-gqa vs enable_gqa math  N={n} {str(dtype)[6:]}: "
                                     f"max|d|={d:.3g} torch.equal={bool(torch.equal(rep, full))}")
            # A top-left-aligned is_causal block would be wrong: make sure the test can see that.
            with sdpa_kernel([SDPBackend.MATH]):
                wrong = F.scaled_dot_product_attention(q[:, :, 128:256], k, v, is_causal=True, enable_gqa=True)
            dw = float((wrong.float() - full[:, :, 128:256].float()).abs().max())
            _line(results, dw > 10 * tol, f"is_causal=True on a later block differs (as expected)  N={n} "
                                          f"{str(dtype)[6:]}: max|d|={dw:.3g}")
    k = torch.randn(1, KV_HEADS, 37, HEAD_DIM)
    _line(results, torch.equal(repeat_kv_heads(k, GROUPS), repeat_kv(k, GROUPS)),
          "k.repeat_interleave(8, dim=1) == transformers repeat_kv(k, 8)")
    sig = sdpa_params_signature()
    print(f"      SDPAParams: {sig}")
    from torch.backends.cuda import SDPAParams, can_use_efficient_attention

    q = torch.zeros(1, HEADS, 4, HEAD_DIM, dtype=torch.bfloat16)
    kk = torch.zeros(1, KV_HEADS, 4, HEAD_DIM, dtype=torch.bfloat16)
    p = SDPAParams(q, kk, kk, None, 0.0, True, True)
    _line(results, sig.count("arg") == 7 and p.is_causal is True and p.enable_gqa is True and p.attn_mask is None,
          "SDPAParams(q,k,v,None,0.0,True,True): 7 positional args, arg5=is_causal, arg6=enable_gqa")
    _line(results, list(inspect.signature(can_use_efficient_attention).parameters) == ["params", "debug"],
          f"can_use_efficient_attention{inspect.signature(can_use_efficient_attention)}")
    print("      sdp flags (CPU process):", json.dumps(sdp_flags()))
    ok = all(results)
    print(f"SELFCHECK {'PASS' if ok else 'FAIL'} ({sum(results)}/{len(results)})")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int)
    ap.add_argument("--kernel", choices=KERNELS)
    ap.add_argument("--report")
    ap.add_argument("--save", help="write the output (CPU bf16 .pt)")
    ap.add_argument("--baseline", help="a --save .pt to compare with (torch.equal, max|diff|)")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=3, help="timed runs (median); 1 is fine for math at large N")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()
    if args.n is None or args.kernel is None:
        ap.error("--n and --kernel are required (or --selfcheck)")
    return main_gpu(args)


if __name__ == "__main__":
    raise SystemExit(main())
