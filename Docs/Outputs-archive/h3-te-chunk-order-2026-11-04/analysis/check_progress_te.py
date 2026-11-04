# -*- coding: utf-8 -*-
"""再較正 (B) の progress.md（計測担当の一次記録）の表の各セルを run.json（analysis/points_te.json）と突き合わせる。
progress.md・runs/ は読むだけ。先に collect_te.py を流すこと。
  python -I -X utf8 analysis/check_progress_te.py > analysis/check_progress_te_output.txt"""
import io, json, os, re
from datetime import datetime
HERE = os.path.dirname(os.path.abspath(__file__))
OUTS = os.path.dirname(os.path.dirname(HERE))
PROG = os.path.join(OUTS, "comfort-calib-h3-2026-11-04-te", "progress.md")
P = {p["label"]: p for p in json.load(io.open(os.path.join(HERE, "points_te.json"), encoding="utf-8"))}
md = io.open(PROG, encoding="utf-8").read().splitlines()

def num(s):
    return float(s.replace(",", "").replace("−", "-").replace("+", ""))

checks = []
def chk(lab, item, pv, rv, tol=0.0):
    ok = (pv is None and rv is None) or (pv is not None and rv is not None and abs(pv - rv) <= tol + 1e-9)
    checks.append((lab, item, pv, rv, ok))

def find(pat, s):
    m = re.search(pat, s)
    return None if not m else [num(x) for x in m.groups()]

hdr = None
for line in md:
    if not line.startswith("|"):
        if line.startswith("#"): hdr = None
        continue
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if cells[0] == "ラベル":
        hdr = cells; continue
    if hdr is None or set(cells[0]) <= set("-: "):
        continue
    row = dict(zip(hdr, cells))
    m = re.search(r"(warm_fl_rec\d+|warm_fl|warm|b1080r|b720r|b1080|b1440|[a-z]\d+_\d+(?:_15s|_p9)?)", cells[0])
    L = m.group(1) if m else None
    if L not in P:
        print("対応づけられない行:", cells[0]); continue
    t, o = P[L]["te"], P[L]["old"]
    whole = " ".join(cells).replace("**", "")
    # 判定
    j = row["判定"].replace("**", "")
    jw = "oom" if j.startswith("OOM") else j.split("（")[0].strip()
    checks.append((L, "判定", jw, t["verdict"], jw == t["verdict"]))
    if "11-04 と同じ" in j and o:
        checks.append((L, "判定（11-04）", t["verdict"], o["verdict"], t["verdict"] == o["verdict"]))
    if t["oom"]:
        oo = t["oom"]
        n = find(r"Tried to allocate ([\d.]+) [GM]iB", j); chk(L, "OOM 要求", n[0], num(oo["tried"].split()[0]))
        n = find(r"([\d.]+) GiB free", j); chk(L, "OOM 空き", n[0], num(oo["free"].split()[0]))
        n = find(r"([\d.]+) GiB allowed", j); chk(L, "OOM 許可", n[0], num(oo["allowed"].split()[0]))
        n = find(r"allocated ([\d.]+)", j); chk(L, "OOM 割り当て", n[0], num(oo["allocated"].split()[0]))
        n = find(r"unallocated ([\d.]+)", j); chk(L, "OOM 予約済みの空き", n[0], num(oo["reserved_unalloc"].split()[0]))
        n = find(r"（11-04 は ([\d.]+)）", j)
        if n: chk(L, "OOM 空き（11-04）", n[0], num(o["oom"]["free"].split()[0]))
        if "数値も 11-04 と同一" in j:
            checks.append((L, "OOM 文言（11-04 と同一）", str(t["oom"]), str(o["oom"]), t["oom"] == o["oom"]))
    # 開始〜終了（progress は計測担当の時計。run.json の wall と秒の差を記録）
    tt = re.findall(r"(\d\d):(\d\d):(\d\d)", row["開始〜終了"])
    for (h, mi, se), key in zip(tt, ("start", "end")):
        w = datetime.strptime(t[key][:19], "%Y-%m-%d %H:%M:%S")
        d = (w.hour * 3600 + w.minute * 60 + w.second) - (int(h) * 3600 + int(mi) * 60 + int(se))
        checks.append((L, "時刻 " + key + "（run.json−progress 秒）", "%s:%s:%s" % (h, mi, se), t[key][11:19], abs(d) <= 5, d))
    n = find(r"([\d,]+)／([\d,]+)", row["S（predict／実測）"]); chk(L, "S predict", n[0], t["S_pred"]); chk(L, "S 実測", n[1], t["S_meas"])
    chk(L, "text", num(row["text"]), t["text"])
    n = find(r"^([\d,]+)", row["cap"].replace("**", "")); chk(L, "cap", n[0] if n else None, t["cap"])
    n = find(r"11-04 も ([\d,]+)", row["帯"])
    if n: chk(L, "cap（11-04）", n[0], o["cap"])
    for col in ("retries", "inactive_split"):
        v = row[col]; chk(L, col, None if v == "—" else num(v), t[col])
    v = row["used_ok"]; chk(L, "used_ok", {"真": 1.0, "偽": 0.0}.get(v), None if t["used_ok"] is None else float(t["used_ok"]))
    # 窓の差（列名は表ごとに違う）
    wc = [k for k in row if k.startswith("窓")][0]; w = row[wc].replace("**", "")
    for name, key, nkey in (("denoise", "d_den", "n_den"), ("decode", "d_dec", None), ("encode", "d_enc", "n_enc")):
        n = find(name + r" Δ([+\-−]?[\d.]+)", w)
        if n: chk(L, "Δ " + name, n[0], t[key])
        n2 = find(name + r" Δ[+\-−]?[\d.]+（n=([\d,]+)", w)
        if n2 and nkey: chk(L, "標本 " + name, n2[0], t[nkey])
    n = find(r"基準 denoise 中央値 ([\d,.]+)MB", w)
    if n: chk(L, "基準 denoise 中央値", n[0], t["base_den"])
    n = find(r"11-04 は \+([\d,.]+)", w)
    if n: chk(L, "Δ encode（11-04）", n[0], o["d_enc"])
    # 備考
    b = row["備考"].replace("**", "")
    n = find(r"denoise 予約 ([\d,]+)・山 ([\d,]+)", b)
    if n: chk(L, "denoise 予約", n[0], t["den_res"]); chk(L, "denoise 山", n[1], t["den_peak"])
    if n and "11-04 と同値" in b and o: chk(L, "denoise 予約（11-04）", n[0], o["den_res"]); chk(L, "denoise 山（11-04）", n[1], o["den_peak"])
    n = find(r"decode 予約 ([\d,]+)・山 ([\d,]+)", b)
    if n: chk(L, "decode 予約", n[0], t["dec_res"]); chk(L, "decode 山", n[1], t["dec_peak"])
    n = find(r"(?<!log )1 ステップ ([\d.]+) 秒", b)
    if n: chk(L, "1 ステップ秒 meta", n[0], t["sps_meta"], 0.051)
    n = find(r"log 1 ステップ ([\d.]+) 秒", b)
    if n: chk(L, "1 ステップ秒 log", n[0], t["sps_log"], 0.051)
    m2 = re.search(r"TE 段 予約 ([\d,]+)(?:（11-04 は ([\d,]+)）)?・山 ([\d,]+)(?:（([\d,]+)）)?・([\d.]+) 秒(?:（([\d.]+)）)?", b)
    if m2:
        g = [None if x is None else num(x) for x in m2.groups()]
        chk(L, "TE 予約", g[0], t["enc_res"]); chk(L, "TE 山", g[2], t["enc_peak"]); chk(L, "TE 秒", g[4], t["enc_s"], 0.051)
        if g[1] is not None: chk(L, "TE 予約（11-04）", g[1], o["enc_res"])
        if g[3] is not None: chk(L, "TE 山（11-04）", g[3], o["enc_peak"])
        if g[5] is not None: chk(L, "TE 秒（11-04）", g[5], o["enc_s"], 0.051)
    n = find(r"encode ([\d.]+) 秒・山 ([\d,]+)", b)
    if n: chk(L, "TE 秒", n[0], t["enc_s"], 0.006); chk(L, "TE 山", n[1], t["enc_peak"])
    n = find(r"reference_encode ([\d.]+) 秒", b)
    if n: chk(L, "参照の符号化 秒", n[0], t["ref_s"], 0.051)

bad = [c for c in checks if not c[4]]
for c in checks:
    print(("OK  " if c[4] else "NG  ") + " | ".join(str(x) for x in c))
print("\n項目数 %d・不一致 %d" % (len(checks), len(bad)))
for c in bad:
    print("不一致:", c)
