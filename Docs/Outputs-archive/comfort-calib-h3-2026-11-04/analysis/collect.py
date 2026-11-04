# -*- coding: utf-8 -*-
"""今夜（2026-11-04）の全点の run.json から集計表を作る（読むだけ。runs/ は変えない）。
  python -I analysis/collect.py  →  analysis/points.json と analysis/points_table.md
"""
import io, json, os, re, glob
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "analysis")
MiB = 1024 * 1024

def g(d, path):
    for k in path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d

ORDER = ["warm", "b1080r", "b720r", "v720_345_15s", "i1080_243_p9",
         "warm_fl", "b1080", "s1080_277", "b720", "s720_481", "b1440",
         "s1080_362", "warm_fl_rec1104124216", "s1080_328", "s1080_345", "s1080_345_r2",
         "s1440_192", "s1440_209", "warm_fl_rec1104144354", "s1440_192_r2",
         "warm2", "v720_379_15s", "r1080_362"]
# run.json のある点だけ（走行中の点は飛ばす）
labels = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(HERE, "runs", "*", "run.json")))
extra = [l for l in labels if l not in ORDER]
ORDER = [l for l in ORDER if l in labels] + extra

def oom_info(r):
    e = r.get("error") or ""
    m = re.search(r"Tried to allocate ([\d.]+ [GM]iB).*?of which ([\d.]+ [GM]iB) is free\. ([\d.]+ [GM]iB) allowed; Of the allocated memory ([\d.]+ [GM]iB) is allocated by PyTorch, and ([\d.]+ [GM]iB) is reserved by PyTorch but unallocated", e, re.S)
    lines = [(fn.replace(chr(92), "/").split("/")[-1], ln, fu) for fn, ln, fu in re.findall(r'File "([^"]+)", line (\d+), in (\w+)', e)]
    tail = re.findall(r'\n\s+(o = torch\.empty\(\(s, inner\)[^\n]*|h2 = [^\n]*|out = torch\.empty[^\n]*)', e)
    return (dict(zip(["tried", "free", "allowed", "allocated", "reserved_unalloc"], m.groups())) if m else None,
            ["%s:%s %s" % x for x in lines], tail)

rows = []
for L in ORDER:
    r = json.load(io.open(os.path.join(HERE, "runs", L, "run.json"), encoding="utf-8"))
    geo = r["geometry"]; ax = g(r, "h3.axes") or {}; pr = r.get("predict") or {}
    seq = g(r, "h3.sequence")
    S_meas = seq["total"] if seq else ax.get("S")
    text = seq["text"] if seq else ax.get("text")
    refs = (r.get("request_sent") or {}).get("references") or []
    R = (ax.get("ref_video") or 0) + (ax.get("ref_audio") or 0)
    oi, frames, tail = oom_info(r)
    loc = g(r, "v3dprime.oom_location")
    row = dict(
        label=L, W=geo["width"], H=geo["height"], F=geo["num_frames"], K=g(r, "tokens.keyframes"),
        refs=[x.get("kind") for x in refs], transformer=g(r, "used.transformer_used") or g(r, "used_matches_request.expected.transformer_used"),
        status=r.get("status"), verdict=r.get("verdict_v3dprime"),
        oom_stage=(loc or {}).get("stage"), oom_frames=frames, oom_tail=tail, oom=oi,
        S_pred=pr.get("S_estimate"), S_meas=S_meas, S_source=ax.get("S_source"),
        vc=ax.get("vc"), vcr=ax.get("vcr"), text=text, text_pred=pr.get("text_estimate"),
        video=ax.get("video"), cond=ax.get("cond"), audio=ax.get("audio"), R=R,
        eff=None if S_meas is None else round(S_meas + 0.40 * text, 1),
        sps_meta=g(r, "h3.sec_per_step_meta"), sps_log=g(r, "h3.sec_per_step_log"),
        wall=g(r, "wall.seconds"), cap=g(r, "h3.vram16.cap_mib"),
        retries=g(r, "h3.vram16.num_alloc_retries"), num_ooms=g(r, "h3.vram16.num_ooms"),
        inactive_split=g(r, "h3.vram16.inactive_split_peak_mib"),
        den_res=g(r, "h3.phase_peak_vram_reserved_mb.denoise"), den_peak=g(r, "h3.phase_peak_vram_mb.denoise"),
        dec_res=g(r, "h3.phase_peak_vram_reserved_mb.decode"), dec_peak=g(r, "h3.phase_peak_vram_mb.decode"),
        enc_res=g(r, "h3.phase_peak_vram_reserved_mb.encode"), enc_peak=g(r, "h3.phase_peak_vram_mb.encode"),
        ref_res=g(r, "h3.phase_peak_vram_reserved_mb.reference_encode"), ref_peak=g(r, "h3.phase_peak_vram_mb.reference_encode"),
        d_den=g(r, "v3dprime.windows.denoise.delta_mb"), d_dec=g(r, "v3dprime.windows.decode.delta_mb"),
        d_enc=g(r, "v3dprime.windows.encode.delta_mb"),
        n_den=g(r, "v3dprime.windows.denoise.wddm_samples"),
        baseline=g(r, "v3dprime.baseline_label"), prefetch_degraded=g(r, "v3dprime.prefetch_degraded"),
        used_ok=g(r, "used_matches_request.ok"), images_used=g(r, "used.images_used"),
        prefetch_used=g(r, "used.block_swap_prefetch_used"), steps=geo["num_inference_steps"],
        attn=g(r, "used.attention_used") or geo.get("attention_backend"),
        denoise_last_done=g(r, "h3_spans.denoise.last_done"), denoise_log_s=g(r, "h3_spans.denoise.log_elapsed_s"),
        diff_pred=None if S_meas is None or pr.get("S_estimate") is None else S_meas - pr["S_estimate"],
        text_diff=ax.get("text_minus_predicted"),
    )
    # 当てはめ（計画 §5.2・11-03 MECHANISM §3.2(b) の 8 点の式）: 2,230.4MiB＋51,227B×S＋20,397B×text＋35B×R
    if S_meas:
        row["fit_alloc_mib"] = round(2230.4 + (51227 * S_meas + 20397 * text + 35 * R) / MiB, 1)
        s = S_meas / 1e5
        row["t_model"] = round(28.67 * s + 57.86 * s * s, 2)
    rows.append(row)

json.dump(rows, io.open(os.path.join(OUT, "points.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

def f(v, nd=0):
    if v is None: return "—"
    if isinstance(v, bool): return str(v)
    if isinstance(v, (int, float)):
        return ("{:,.%df}" % nd).format(v)
    return str(v)

hdr = ["ラベル", "形", "判定", "OOM の段", "S predict／実測", "vc", "vcr", "text", "eff", "1 ステップ秒（meta／log）", "所要秒", "cap", "retries", "inactive_split 山", "denoise 予約／山", "decode 予約／山", "Δ denoise／decode", "used_ok"]
out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
for x in rows:
    shape = "%d×%d×%d" % (x["W"], x["H"], x["F"]) + ("・K=%s" % x["K"]) + (("＋" + "・".join("%s×%d" % ({"picture": "画像", "video": "動画", "audio": "音声"}.get(k, k), x["refs"].count(k)) for k in sorted(set(x["refs"])))) if x["refs"] else "")
    out.append("| " + " | ".join([
        x["label"], shape, f(x["verdict"]), f(x["oom_stage"]),
        "%s／%s" % (f(x["S_pred"]), f(x["S_meas"])), f(x["vc"]), f(x["vcr"]), f(x["text"]), f(x["eff"], 0),
        "%s／%s" % (f(x["sps_meta"], 2), f(x["sps_log"], 2)), f(x["wall"], 0), f(x["cap"]), f(x["retries"]), f(x["inactive_split"]),
        "%s／%s" % (f(x["den_res"]), f(x["den_peak"])), "%s／%s" % (f(x["dec_res"]), f(x["dec_peak"])),
        "%s／%s" % (f(x["d_den"], 1), f(x["d_dec"], 1)), f(x["used_ok"])]) + " |")
io.open(os.path.join(OUT, "points_table.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
for x in rows:
    if x["oom"]:
        print(x["label"], x["oom"], x["oom_frames"][-3:], x["oom_tail"], "last_done", x["denoise_last_done"], x["denoise_log_s"])
