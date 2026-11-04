# -*- coding: utf-8 -*-
"""台帳 §1-275 の再較正 (B) の run.json（comfort-calib-h3-2026-11-04-te/runs）と、11-04 の同じラベル
（comfort-calib-h3-2026-11-04/runs）を並べる（読むだけ。runs/ は変えない）。
  python -I -X utf8 analysis/collect_te.py > analysis/collect_te_output.txt
出力: analysis/points_te.json（両日の値）・標準出力に比較表。"""
import io, json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
OUTS = os.path.dirname(os.path.dirname(HERE))          # .../outputs
TE = os.path.join(OUTS, "comfort-calib-h3-2026-11-04-te", "runs")
OLD = os.path.join(OUTS, "comfort-calib-h3-2026-11-04", "runs")
ORDER = ["warm_fl", "b1080", "s1080_345", "s1080_362", "warm_fl_rec1105002124", "b1440", "s1440_192", "s1440_209",
         "warm", "b720r", "b1080r", "v720_379_15s", "i1080_243_p9"]

def g(d, path):
    for k in path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d

def oom(r):
    e = r.get("error") or ""
    m = re.search(r"Tried to allocate ([\d.]+ [GM]iB).*?of which ([\d.]+ [GM]iB) is free\. ([\d.]+ [GM]iB) allowed; Of the allocated memory ([\d.]+ [GM]iB) is allocated by PyTorch, and ([\d.]+ [GM]iB) is reserved", e, re.S)
    return dict(zip(["tried", "free", "allowed", "allocated", "reserved_unalloc"], m.groups())) if m else None

def row(path, L):
    p = os.path.join(path, L, "run.json")
    if not os.path.exists(p):
        return None
    r = json.load(io.open(p, encoding="utf-8"))
    ax = g(r, "h3.axes") or {}; seq = g(r, "h3.sequence")
    return dict(
        label=L, verdict=r.get("verdict_v3dprime"), status=r.get("status"),
        oom_stage=g(r, "v3dprime.oom_location.stage"), oom=oom(r),
        S_pred=g(r, "predict.S_estimate"), S_meas=seq["total"] if seq else ax.get("S"),
        text=seq["text"] if seq else ax.get("text"), text_diff=ax.get("text_minus_predicted"),
        cap=g(r, "h3.vram16.cap_mib"), retries=g(r, "h3.vram16.num_alloc_retries"),
        inactive_split=g(r, "h3.vram16.inactive_split_peak_mib"),
        den_res=g(r, "h3.phase_peak_vram_reserved_mb.denoise"), den_peak=g(r, "h3.phase_peak_vram_mb.denoise"),
        dec_res=g(r, "h3.phase_peak_vram_reserved_mb.decode"), dec_peak=g(r, "h3.phase_peak_vram_mb.decode"),
        enc_res=g(r, "h3.phase_peak_vram_reserved_mb.encode"), enc_peak=g(r, "h3.phase_peak_vram_mb.encode"),
        enc_s=g(r, "h3.phase_seconds.encode"), ref_s=g(r, "h3.phase_seconds.reference_encode"),
        d_den=g(r, "v3dprime.windows.denoise.delta_mb"), d_dec=g(r, "v3dprime.windows.decode.delta_mb"),
        d_enc=g(r, "v3dprime.windows.encode.delta_mb"),
        n_den=g(r, "v3dprime.windows.denoise.wddm_samples"), n_enc=g(r, "v3dprime.windows.encode.wddm_samples"),
        base_den=g(r, "v3dprime.windows.denoise.shared_median_mb") if g(r, "v3dprime.is_baseline") else g(r, "v3dprime.windows.denoise.baseline_shared_median_mb"),
        enc_ded_med=g(r, "vram.encode_window.dedicated_median_mb"), enc_sh_med=g(r, "vram.encode_window.shared_median_mb"),
        sps_meta=g(r, "h3.sec_per_step_meta"), sps_log=g(r, "h3.sec_per_step_log"),
        wall=g(r, "wall.seconds"), start=g(r, "wall.start_iso"), end=g(r, "wall.end_iso"),
        used_ok=g(r, "used_matches_request.ok"), baseline=g(r, "v3dprime.baseline_label"),
        prefetch_degraded=g(r, "v3dprime.prefetch_degraded"),
    )

pts = []
for L in ORDER:
    pts.append(dict(label=L, te=row(TE, L), old=row(OLD, L)))
json.dump(pts, io.open(os.path.join(HERE, "points_te.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

def f(v, nd=0):
    if v is None: return "—"
    if isinstance(v, bool): return str(v)
    if isinstance(v, (int, float)): return ("{:,.%df}" % nd).format(v)
    return str(v)

def table(title, cols):
    print("\n## " + title)
    print("| ラベル | " + " | ".join(c[0] for c in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for p in pts:
        t, o = p["te"] or {}, p["old"] or {}
        print("| " + p["label"] + " | " + " | ".join("%s（%s）" % (fn(t), fn(o)) for _, fn in cols) + " |")

table("判定・S・cap・text（te（11-04））", [
    ("判定", lambda x: f(x.get("verdict"))), ("S 実測", lambda x: f(x.get("S_meas"))),
    ("text", lambda x: f(x.get("text"))), ("text 差", lambda x: f(x.get("text_diff"))),
    ("cap", lambda x: f(x.get("cap"))), ("used_ok", lambda x: f(x.get("used_ok")))])
table("ノイズ除去・復号（te（11-04））", [
    ("denoise 予約／山", lambda x: "%s／%s" % (f(x.get("den_res")), f(x.get("den_peak")))),
    ("decode 予約／山", lambda x: "%s／%s" % (f(x.get("dec_res")), f(x.get("dec_peak")))),
    ("retries", lambda x: f(x.get("retries"))), ("inactive_split", lambda x: f(x.get("inactive_split")))])
table("窓の差 Δ（te（11-04））", [
    ("Δ denoise", lambda x: f(x.get("d_den"), 1)), ("Δ decode", lambda x: f(x.get("d_dec"), 1)),
    ("Δ encode", lambda x: f(x.get("d_enc"), 1)), ("基準 denoise 中央値", lambda x: f(x.get("base_den"), 1))])
table("TE 段（te（11-04））", [
    ("encode 予約／山", lambda x: "%s／%s" % (f(x.get("enc_res")), f(x.get("enc_peak")))),
    ("encode 秒", lambda x: f(x.get("enc_s"), 2)), ("1 ステップ秒 meta", lambda x: f(x.get("sps_meta"), 2)),
    ("所要秒", lambda x: f(x.get("wall"), 0))])
print("\n## OOM の文言")
for p in pts:
    for k in ("te", "old"):
        x = p[k]
        if x and x.get("oom"):
            print(p["label"], k, x["oom"])
