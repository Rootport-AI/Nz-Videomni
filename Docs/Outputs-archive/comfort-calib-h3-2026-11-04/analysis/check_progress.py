# -*- coding: utf-8 -*-
"""progress.md（計測担当の記録）の表の各セルを run.json（analysis/points.json）と突き合わせる。progress.md は読むだけ。
  python -I -X utf8 analysis/check_progress.py > analysis/check_progress_output.txt"""
import io, json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P = {p["label"]: p for p in json.load(io.open(os.path.join(HERE, "points.json"), encoding="utf-8"))}
md = io.open(os.path.join(ROOT, "progress.md"), encoding="utf-8").read().splitlines()

def nums(s):
    return [float(x.replace(",", "")) for x in re.findall(r"[+\-−]?\d[\d,]*(?:\.\d+)?", s.replace("−", "-"))]

checks = []  # (label, item, progress, run, ok)
def chk(lab, item, pv, rv, tol=0.051):
    if pv is None and rv is not None:
        checks.append((lab, item, "記録なし（—）", rv, None)); return
    ok = (pv is None and rv is None) or (pv is not None and rv is not None and abs(pv - rv) <= tol)
    checks.append((lab, item, pv, rv, ok))

hdr = None
for line in md:
    if not line.startswith("|"):
        if line.startswith("#"):
            hdr = None   # 表の途中の箇条（"  - …の事実"）では見出しを保つ
        continue
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if cells and cells[0] == "ラベル":
        hdr = cells; continue
    if hdr is None or set(cells[0]) <= set("-: "):
        continue
    row = dict(zip(hdr, cells))
    m = re.search(r"(warm_fl_rec\d+|warm_fl2|warm_fl|warm2|warm|b1080r|b720r|b1080|b720|b1440|[a-z]\d+_\d+(?:_15s|_p9|_3s)?(?:_r2|_20)?)", cells[0])
    if not m or m.group(1) not in P:
        print("表の行をラベルに対応づけられない:", cells[0][:40]); continue
    L = m.group(1); p = P[L]
    for col, val in row.items():
        v = val.replace("**", "")
        if col.startswith("S（"):
            n = nums(v)
            if len(n) >= 2: chk(L, "S predict", n[0], p["S_pred"], 0); chk(L, "S 実測", n[1], p["S_meas"], 0)
        elif col in ("cap_mib", "cap"):
            n = nums(v); chk(L, "cap", n[0] if n else None, p["cap"], 0)
        elif col == "vc":
            n = nums(v); chk(L, "vc", n[0] if n else None, p["vc"], 0)
        elif col == "vcr":
            n = nums(v); chk(L, "vcr", n[0] if n else None, p["vcr"], 0)
        elif col == "text":
            n = nums(v); chk(L, "text", n[0] if n else None, p["text"], 0)
        elif col == "1 ステップ秒":
            n = nums(v)
            if n: chk(L, "1 ステップ秒(meta)", n[0], p["sps_meta"] if p["sps_meta"] is not None else p["sps_log"], 0.006 if len(str(n[0]).split(".")[-1]) >= 2 else 0.051)
            else: chk(L, "1 ステップ秒(meta)", None, p["sps_meta"])
        elif col == "所要秒":
            n = nums(v); chk(L, "所要秒(wall)", n[0] if n else None, None if p["wall"] is None else round(p["wall"]), 1.0)
        elif col == "retries":
            n = nums(v); chk(L, "retries", n[0] if n else None, p["retries"], 0)
        elif col == "inactive_split":
            n = nums(v); chk(L, "inactive_split", n[0] if n else None, p["inactive_split"], 0)
        elif col in ("used_ok", "used_check_h3"):
            pv = {"真": 1.0, "偽": 0.0}.get(v); rv = None if p["used_ok"] is None else float(p["used_ok"]); chk(L, "used_ok", pv, rv, 0)
        elif col == "images_used":
            n = nums(v); chk(L, "images_used", n[0] if n else None, p["images_used"], 0)
        elif col in ("denoise 予約／山",):
            n = nums(v); chk(L, "denoise 予約", n[0], p["den_res"], 0); chk(L, "denoise 山", n[1], p["den_peak"], 0)
        elif col in ("decode 予約／山", "復号の予約／山"):
            n = nums(v)
            if n: chk(L, "decode 予約", n[0], p["dec_res"], 0); chk(L, "decode 山", n[1], p["dec_peak"], 0)
        elif col == "窓 Δ（denoise／decode）":
            n = nums(v)
            if n: chk(L, "Δdenoise", n[0], p["d_den"]); chk(L, "Δdecode", n[1], p["d_dec"])
        elif col == "判定":
            mm = re.search(r"denoise 予約 ([\d,]+)MB・山 ([\d,]+)MB|denoise の予約 ([\d,]+)MB・割り当ての山 ([\d,]+)MB", v)
            if mm:
                a, b = [x for x in mm.groups() if x][:2]
                chk(L, "denoise 予約(判定欄)", float(a.replace(",", "")), p["den_res"], 0); chk(L, "denoise 山(判定欄)", float(b.replace(",", "")), p["den_peak"], 0)
            mm = re.search(r"窓 denoise Δ([+\-−]?[\d.]+)", v)
            if mm: chk(L, "Δdenoise(判定欄)", float(mm.group(1).replace("−", "-")), p["d_den"])
            mm = re.search(r"decode Δ([+\-−]?[\d.]+)", v)
            if mm: chk(L, "Δdecode(判定欄)", float(mm.group(1).replace("−", "-")), p["d_dec"])
            mm = re.search(r"eff ([\d,]+)", v)
            if mm: chk(L, "eff(判定欄)", float(mm.group(1).replace(",", "")), round(p["eff"]), 0.6)
            verdict = "oom" if "OOM" in v.split("（")[0] else ("comfortable" if "comfortable" in v else ("baseline" if "baseline" in v else ("smoke" if "smoke" in v or "煙" in v else None)))
            checks.append((L, "判定", verdict, p["verdict"], verdict == p["verdict"]))
            if L == "b1440":
                mm = re.search(r"復号の段の予約 ([\d,]+)MB", v); chk(L, "decode 予約(判定欄)", float(mm.group(1).replace(",", "")), p["dec_res"], 0)

bad = [c for c in checks if c[4] is False]
blank = [c for c in checks if c[4] is None]
print("突き合わせた項目 %d・差のある項目 %d・progress 側が「—」で比べなかった項目 %d" % (len(checks) - len(blank), len(bad), len(blank)))
for c in blank:
    print("  空欄:", c[:4])
for c in bad:
    print("  差:", c)
labs = sorted({c[0] for c in checks})
print("突き合わせた点 %d: %s" % (len(labs), ", ".join(labs)))
print("progress.md の表に無い点:", sorted(set(P) - set(labs)))
