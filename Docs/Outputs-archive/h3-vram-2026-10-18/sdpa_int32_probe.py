"""T6: SDPA（torch の scaled_dot_product_attention）の mem-efficient 経路の int32 の掛け算の溢れの合成試験。

仮説 H5（計画 PLAN_measure.md §1）: mem-efficient は v の行の刻み（21,504）× キー位置を int32 で掛けるため、
系列長 約 99,968 を超えると誤った値か CUDA エラーになる。v を連続にすれば起きない。

並びは製品と同じにする（R2 §5-3・engine_h3/dit.py）:
  q・k = 連続の [S,56,128] bf16 を transpose(0,1).unsqueeze(0) した view（行の刻み 7,168）
  v    = [S,3,56,128] bf16 から unbind(1) した 3 本目を同じく転置した view（行の刻み 21,504）
  ヘッドは [:, :H]（既定 8 本）に切り出した view で試す（行の刻みは保たれる。時間と VRAM を減らすため）。
正解＝v を .contiguous() にした版。比べる組:
  (a) 既定の選択      : sdpa(q, k, v_view) と sdpa(q, k, v_view.contiguous())
  (b) EFFICIENT 強制  : 同じ 2 本を torch.nn.attention.sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION) の中で
結果は probe_result.json（この台本のフォルダ）に S ごとに書き足す。例外で CUDA の文脈が壊れることがあるので、
S ごとに別プロセスで回す:  python sdpa_int32_probe.py --seq 98304 / --seq 100352

必要な VRAM の目安（S=100,352・56 ヘッドの土台を作るため）: qkv 約 4.3GB＋q・k 各 約 1.4GB＋出力 ≈ 7.5GB。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_case(torch, F, q, k, v, forced: bool) -> dict:
    from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: PLC0415

    out: dict = {"forced_efficient": forced}
    try:
        t0 = time.perf_counter()
        if forced:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                o_view = F.scaled_dot_product_attention(q, k, v)
                o_ref = F.scaled_dot_product_attention(q, k, v.contiguous())
        else:
            o_view = F.scaled_dot_product_attention(q, k, v)
            o_ref = F.scaled_dot_product_attention(q, k, v.contiguous())
        torch.cuda.synchronize()
        out["seconds"] = round(time.perf_counter() - t0, 3)
        diff = (o_view.float() - o_ref.float()).abs()
        out["max_abs_diff"] = float(diff.max().item())
        out["mean_abs_diff"] = float(diff.mean().item())
        out["identical"] = bool(out["max_abs_diff"] == 0.0)
        out["view_finite"] = bool(torch.isfinite(o_view).all().item())
        out["ref_finite"] = bool(torch.isfinite(o_ref).all().item())
        out["ref_abs_mean"] = float(o_ref.float().abs().mean().item())
        # 行（クエリ位置）ごとの差: どこから壊れるか
        row = diff.amax(dim=(0, 1, 3))
        bad = (row > 0).nonzero().flatten()
        out["first_bad_query_row"] = int(bad[0].item()) if bad.numel() else None
        out["bad_query_rows"] = int(bad.numel())
        del o_view, o_ref, diff, row, bad
    except Exception as exc:  # noqa: BLE001 -- 例外も結果
        out["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000],
                        "traceback": traceback.format_exc()[-3000:]}
    return out


def run_isolated(torch, F, q, k, v, mode: str) -> dict:
    """既定の選択で 1 本だけ呼ぶ（view＝sdpa(q,k,v_view)／contiguous＝sdpa(q,k,v_view.contiguous())）。"""
    out: dict = {"mode": mode}
    try:
        vv = v.contiguous() if mode == "contiguous" else v
        t0 = time.perf_counter()
        o = F.scaled_dot_product_attention(q, k, vv)
        torch.cuda.synchronize()
        out["seconds"] = round(time.perf_counter() - t0, 3)
        out["finite"] = bool(torch.isfinite(o).all().item())
        out["abs_mean"] = float(o.float().abs().mean().item())
        out["error"] = None
    except Exception as exc:  # noqa: BLE001 -- 例外も結果
        out["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000],
                        "traceback": traceback.format_exc()[-3000:]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="T6: SDPA mem-efficient の int32 刻み溢れの合成試験（S ごとに 1 プロセス）")
    ap.add_argument("--seq", type=int, required=True, help="系列長 S（計画では 98304 と 100352）")
    ap.add_argument("--heads", type=int, default=8, help="切り出すヘッド数（56 本のうち先頭から）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "probe_result.json"))
    ap.add_argument("--mode", choices=("both", "view", "contiguous"), default="both",
                    help="both＝view 版と連続版を比べる（既定）／view・contiguous＝その 1 本だけを既定の選択で呼ぶ"
                         "（結果は probe_result.json の S の下に isolated_<mode> として書き足す）")
    a = ap.parse_args(argv)

    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    S, NH, D = a.seq, 56, 128
    res: dict = {"seq": S, "heads": a.heads, "torch": torch.__version__, "cuda": torch.version.cuda,
                 "device": torch.cuda.get_device_name(0), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "pid": os.getpid()}
    res["sdpa_paths"] = {n: getattr(torch.backends.cuda, n)() for n in
                         ("is_flash_attention_available", "mem_efficient_sdp_enabled", "flash_sdp_enabled",
                          "math_sdp_enabled") if hasattr(torch.backends.cuda, n)}
    try:
        g = torch.Generator(device="cuda").manual_seed(a.seed)
        q_full = torch.randn((S, NH, D), dtype=torch.bfloat16, device="cuda", generator=g)
        k_full = torch.randn((S, NH, D), dtype=torch.bfloat16, device="cuda", generator=g)
        qkv = torch.randn((S, 3, NH, D), dtype=torch.bfloat16, device="cuda", generator=g)
        v_rows = qkv.unbind(1)[2]                                   # [S,56,128] 刻み (21504,128,1)
        q = q_full.transpose(0, 1).unsqueeze(0)[:, : a.heads]       # [1,H,S,128] 刻み (.,128,7168,1)
        k = k_full.transpose(0, 1).unsqueeze(0)[:, : a.heads]
        v = v_rows.transpose(0, 1).unsqueeze(0)[:, : a.heads]       # [1,H,S,128] 刻み (.,128,21504,1)
        res["strides"] = {"q": list(q.stride()), "k": list(k.stride()), "v_view": list(v.stride()),
                          "v_contiguous": list(v.contiguous().stride())}
        res["int32_products"] = {"v_row_stride_x_S": v.stride(2) * S, "q_row_stride_x_S": q.stride(2) * S,
                                 "int32_max": 2 ** 31 - 1}
        if a.mode == "both":
            res["cases"] = {"default": run_case(torch, F, q, k, v, forced=False)}
            res["cases"]["efficient_forced"] = run_case(torch, F, q, k, v, forced=True)
        else:
            res["isolated"] = run_isolated(torch, F, q, k, v, a.mode)
    except Exception as exc:  # noqa: BLE001
        res["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000],
                        "traceback": traceback.format_exc()[-3000:]}
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

    path = Path(a.out)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {}
    if a.mode == "both":
        data[str(S)] = res
    else:
        data.setdefault(str(S), {})[f"isolated_{a.mode}"] = res
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
