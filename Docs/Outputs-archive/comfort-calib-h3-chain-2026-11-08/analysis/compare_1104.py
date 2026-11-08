# -*- coding: utf-8 -*-
"""連結のクリップと 11-04 の Single の同じ形の点を、ノイズ除去の段の山・予約・retries で並べる（読むだけ）。
  python -I analysis/compare_1104.py
今夜の値は analysis/points.json（collect.py の出力）、11-04 の値は
Docs/Outputs-archive/comfort-calib-h3-2026-11-04/analysis/points.json（11-04 の collect.py の出力の複写）。
「マスクの見積もり」は計画 cozy-gliding-pudding §1・§3.3 の「映像 1 行あたり約 768 バイト」（float32 の 2 本）に
固定行を持つクリップの映像＋条件の行数を掛けた値（MiB）。音声のマスクは小さいので含めない。"""
import io, json, os, sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(HERE))
tonight = {p["label"]: p for p in json.load(io.open(os.path.join(HERE, "analysis", "points.json"), encoding="utf-8"))}
old = {p["label"]: p for p in json.load(io.open(os.path.join(
    REPO, "Docs", "Outputs-archive", "comfort-calib-h3-2026-11-04", "analysis", "points.json"), encoding="utf-8"))}

# (今夜の点, クリップ, 11-04 の点, 固定行の有無)
PAIRS = [("A1", 1, "r1080_362", True), ("A1_r2", 1, "r1080_362", True),
         ("F", 0, "r1080_362", True), ("F", 1, "r1080_362", True),
         ("cb1080", 1, "b1080r", True), ("cb1080v", 0, "b1080r", True),
         ("E", 1, "v720_345_15s", True),
         ("A1", 0, "s1080_345", False), ("A1_r2", 0, "s1080_345", False),
         ("B1", 0, "s1440_192", False), ("C1", 0, "s720_481", False),
         ("cb1080", 0, "b1080", False), ("cb1440", 0, "b1440", False), ("cb720", 0, "b720", False)]

print("| 今夜の点・クリップ | 固定行 | 11-04 の点 | S（今夜／11-04） | den 山 MiB（今夜／11-04／差） | den 予約 MiB（今夜／11-04／差） "
      "| 予約−山 MiB（今夜／11-04） | 上限までの余裕 MiB（今夜／11-04） | retries（今夜／11-04） | inactive_split MiB（今夜／11-04） | マスクの見積もり MiB |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for L, i, O, fixed in PAIRS:
    c = [x for x in tonight[L]["clips"] if x["clip"] == i][0]
    o = old[O]
    mask = None
    if fixed:
        info = c.get("mask") or {}
        rows = (info.get("video") or {}).get("rows")
        mask = rows * 768 / 1048576 if rows else None
    print("| %s・クリップ %d | %s | %s | %s／%s | %s／%s／%+d | %s／%s／%+d | %d／%d | %d／%d | %s／%s | %s／%s | %s |" % (
        L, i, "あり" if fixed else "なし", O, "{:,}".format(c["S_meas"]), "{:,}".format(o["S_meas"]),
        "{:,}".format(c["den_peak"]), "{:,}".format(o["den_peak"]), c["den_peak"] - o["den_peak"],
        "{:,}".format(c["den_res"]), "{:,}".format(o["den_res"]), c["den_res"] - o["den_res"],
        c["den_res"] - c["den_peak"], o["den_res"] - o["den_peak"],
        c["cap"] - c["den_res"], o["cap"] - o["den_res"],
        c["retries"], o["retries"], "{:,}".format(c["inactive_split"]), "{:,}".format(o["inactive_split"]),
        "—" if mask is None else "%.1f" % mask))
