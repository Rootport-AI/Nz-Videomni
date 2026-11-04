# -*- coding: utf-8 -*-
"""計画 §8 の結論の形の判定。快適／OOM の点を制約として並べ、軸ごとに 1 本の線で分かれるかを見る。
重さの線は  L = video + cond + audio + w_ref·(ref_video+ref_audio) + w_text·text  （映像・条件行・音声の行を 1）。
points.json を読むだけ。  python -I -X utf8 analysis/line_forms.py > analysis/line_forms_output.txt"""
import io, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
P = [p for p in json.load(io.open(os.path.join(HERE, "points.json"), encoding="utf-8"))
     if p["verdict"] in ("comfortable", "oom")]
C = [p for p in P if p["verdict"] == "comfortable"]; O = [p for p in P if p["verdict"] == "oom"]
def base(p): return p["video"] + p["cond"] + p["audio"]
axes = {
    "S": lambda p: p["S_meas"],
    "eff＝S＋0.40·text": lambda p: p["S_meas"] + 0.4 * p["text"],
    "vc（映像＋条件行）": lambda p: p["vc"],
    "vcr（映像＋条件行＋参照）": lambda p: p["vcr"],
    "S−text": lambda p: p["S_meas"] - p["text"],
}
def report(name, f, pts_c, pts_o):
    cm = max(pts_c, key=f); om = min(pts_o, key=f)
    gap = f(om) - f(cm)
    print("  %-26s 快適の最大 %-14s %10.0f ／ OOM の最小 %-10s %10.0f  → %s（差 %+.0f）" % (
        name, cm["label"], f(cm), om["label"], f(om), "分かれる" if gap > 0 else "分かれない", gap))
print("== 参照なしの点だけ（FL2VA K=1 の 1080・1440・720、Ref2VA K=0 の r1080_362）==")
nc = [p for p in C if p["R"] == 0]; no = [p for p in O if p["R"] == 0]
for k, f in axes.items(): report(k, f, nc, no)
print("== 全点（参照つき v720_345_15s・v720_379_15s・i1080_243_p9 を含む）==")
for k, f in axes.items(): report(k, f, C, O)
print()
print("== 解像度ごとの帯（参照なし・FL2VA K=1）==")
for res in ("1920×1088", "2560×1472"):
    W, H = map(int, res.split("×"))
    cc = [p for p in nc if (p["W"], p["H"]) == (W, H) and p["K"] == 1]; oo = [p for p in no if (p["W"], p["H"]) == (W, H)]
    c = max(cc, key=lambda p: p["S_meas"]); u = min(oo, key=lambda p: p["S_meas"])
    print("  %s: c=%s（S %d・eff %.0f）U=%s（S %d・eff %.0f）" % (res, c["F"], c["S_meas"], c["eff"], u["F"], u["S_meas"], u["eff"]))
print()
print("== 重さの線: w_text を固定したときの w_ref の取りうる範囲（快適 ≦ θ ＜ OOM。θ は自由）==")
def feasible(wr, wt):
    L = lambda p: base(p) + wr * p["R"] + wt * p["text"]
    return max(L(p) for p in C) < min(L(p) for p in O)
for wt in (0.0, 0.5, 0.75, 0.752, 0.76, 1.0, 1.2, 1.4, 2.0, 3.0, 4.5, 4.6):
    ok = [wr / 1000 for wr in range(0, 3001) if feasible(wr / 1000, wt)]
    print("  w_text=%.3f: w_ref ∈ %s" % (wt, ("[%.3f, %.3f]" % (min(ok), max(ok))) if ok else "なし"))
# 解析的な境界（拘束になる組）
print()
print("== 拘束になる組（解析）==")
pairs = []
for c in C:
    for o in O:
        # base(c)+wr R_c+wt T_c < base(o)+wr R_o + wt T_o
        a = c["R"] - o["R"]; b = c["text"] - o["text"]; k = base(o) - base(c)
        pairs.append((c["label"], o["label"], a, b, k))
for c, o, a, b, k in pairs:
    if a != 0 or b != 0:
        print("  %-14s < %-10s :  %+d·w_ref %+d·w_text < %d" % (c, o, a, b, k))
