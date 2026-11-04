# -*- coding: utf-8 -*-
"""ノイズ除去の割り当ての山を計画 §5.2 の当てはめ（11-03 MECHANISM §3.2(b) の 8 点の式）と比べ、
1 ステップ秒を時間モデル t＝28.67·s＋57.86·s²（s＝S÷10 万）と比べる。points.json を読むだけ。
  python -I -X utf8 analysis/fit_and_time.py > analysis/fit_and_time_output.txt"""
import io, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(io.open(os.path.join(HERE, "points.json"), encoding="utf-8"))
MiB = 1048576
# 4 ステップの完走点（20 ステップの s1080_345_20 は 4 ステップの s1080_345 と同じ形・同じ山なので、当てはめ直しの重複を避けて除く）
big = [p for p in P if p["verdict"] == "comfortable" and p["steps"] == 4]
print("== ノイズ除去の割り当ての山（MiB）: 実測 vs 2,230.4＋51,227B·S＋20,397B·text＋35B·R ==")
print("%-14s %8s %8s %8s %7s %8s %8s %8s %8s" % ("label", "S", "text", "R", "peak", "fit", "resid", "reserved", "res-peak"))
res = []
for p in big:
    r = p["den_peak"] - p["fit_alloc_mib"]; res.append(r)
    print("%-14s %8d %8d %8d %7d %8.1f %+8.1f %8d %8d" % (p["label"], p["S_meas"], p["text"], p["R"], p["den_peak"], p["fit_alloc_mib"], r, p["den_res"], p["den_res"] - p["den_peak"]))
print("残差 min %+.1f max %+.1f" % (min(res), max(res)))
# 今夜の完走 8 点で当てはめ直す（参考。結論には使わない）
A = np.array([[1, p["S_meas"], p["text"], p["R"]] for p in big], float)
y = np.array([p["den_peak"] * MiB for p in big], float)
c, *_ = np.linalg.lstsq(A, y, rcond=None)
pred = A @ c
print("今夜の 8 点での当てはめ直し（参考）: %.1fMiB＋%.0fB·S＋%.0fB·text＋%.0fB·R、残差 max %.1fMiB" % (c[0] / MiB, c[1], c[2], c[3], np.max(np.abs(pred - y)) / MiB))
print()
print("== OOM の 2 点の当てはめの予測（その形で完走したら届くはずの割り当ての山）==")
for p in P:
    if p["verdict"] == "oom":
        print("%-10s fit %.1fMiB  cap(前後の点) 13,999  余裕 %+.1fMiB" % (p["label"], p["fit_alloc_mib"], 13999 - p["fit_alloc_mib"]))
for p in big:
    print("%-14s fit %.1f  cap %d  余裕(cap−fit) %+.1f  cap−予約山 %+d  cap−割当山 %+d" % (p["label"], p["fit_alloc_mib"], p["cap"], p["cap"] - p["fit_alloc_mib"], p["cap"] - p["den_res"], p["cap"] - p["den_peak"]))
print()
print("== 1 ステップ秒: 実測（meta）vs t＝28.67·s＋57.86·s² ==")
for p in P:
    if p["S_meas"] and p["S_meas"] > 30000 and (p["sps_meta"] or p["sps_log"]):
        m = p["sps_meta"] or p["sps_log"]
        src = "meta" if p["sps_meta"] else "log(ステップ 0)"
        print("%-14s S %7d  実測 %7.2f(%s)  モデル %7.2f  差 %+6.1f%%" % (p["label"], p["S_meas"], m, src, p["t_model"], 100 * (m - p["t_model"]) / p["t_model"]))
