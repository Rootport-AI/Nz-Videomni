# -*- coding: utf-8 -*-
"""【当てはめ・結果を見てから】ノイズ除去の予約の山＝割り当ての山＋差（予約済みの空き）。差を映像の行 Nv·U で割った値 φ を点ごとに出し、
「割り当て（今夜の 8 点で当てはめ直した式）＋φ·Nv·U ≦ cap」で全点が分かれるかを見る。points.json を読むだけ。
  python -I -X utf8 analysis/reserve_gap.py > analysis/reserve_gap_output.txt"""
import io, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(io.open(os.path.join(HERE, "points.json"), encoding="utf-8"))
MiB = 1048576; U = 10752
def alloc_refit(p):  # fit_and_time.py の当てはめ直し（今夜の完走 8 種）
    return 2252.1 + (51102 * p["S_meas"] + 10287 * p["text"] + 17 * p["R"]) / MiB
print("%-14s %7s %7s %6s %7s %7s %6s %6s %8s" % ("label", "Nv", "S", "retr", "予約山", "割当山", "差", "φ", "cap−予約"))
phis = []
for p in P:
    if p["verdict"] == "comfortable":
        gap = p["den_res"] - p["den_peak"]; phi = gap * MiB / (p["video"] * U)
        print("%-14s %7d %7d %6d %7d %7d %6d %6.3f %8d" % (p["label"], p["video"], p["S_meas"], p["retries"], p["den_res"], p["den_peak"], gap, phi, p["cap"] - p["den_res"]))
        if p["retries"] >= 8 and p["label"] != "i1080_243_p9":
            phis.append(phi)
lo, hi = min(phis), max(phis)
print("cap に近い完走（retries 8 以上。i1080_243_p9 を除く）の φ: %.3f〜%.3f" % (lo, hi))
print()
print("== 予約の山の予測＝割り当て（当てはめ直し）＋φ·Nv·U と cap 13,999MiB ==")
for p in P:
    if p["verdict"] in ("comfortable", "oom") and p["S_meas"] > 100000:
        a = alloc_refit(p)
        r_lo = a + lo * p["video"] * U / MiB; r_hi = a + hi * p["video"] * U / MiB
        cap = p["cap"] or 13999
        print("%-14s %-11s 割当 %8.1f  予約の予測 %8.1f〜%8.1f  cap %d  → %s" % (p["label"], p["verdict"], a, r_lo, r_hi, cap,
              "上限内" if r_hi <= cap else ("上限超" if r_lo > cap else "またぐ")))
print()
vrow = 51102 + lo * U; vrow2 = 51102 + hi * U
print("この形での 1 行あたりの重さ（映像の行を 1）: 参照・条件行・音声 %.3f〜%.3f、text %.3f〜%.3f" % (
    51102 / vrow2, 51102 / vrow, (51102 + 10287) / vrow2, (51102 + 10287) / vrow))
