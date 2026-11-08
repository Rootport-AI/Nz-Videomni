# -*- coding: utf-8 -*-
"""2026-11-08 の連結較正の全点を runs/<label>/run.json から抜き出す（読むだけ。runs/ は変えない）。
  python -I analysis/collect.py  ->  analysis/points.json・analysis/points_table.md（標準出力にも同じ表）
連結の点はクリップごとに 1 行。Single の点（warm・b1080・D-K3・D-K3_r2・自動復帰の warm）は 1 行。"""
import io, json, os, sys

sys.stdout.reconfigure(encoding="utf-8")  # -I は PYTHONIOENCODING を読まない

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "analysis")


def g(d, path):
    for k in path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


# 走らせた順（manifest.jsonl）
ORDER = []
for line in io.open(os.path.join(HERE, "manifest.jsonl"), encoding="utf-8"):
    if line.strip():
        L = json.loads(line)["label"]
        if L not in ORDER:
            ORDER.append(L)


def n(x):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return "{:,}".format(int(round(x)))
    return str(x)


def raw(x):
    return "—" if x is None else str(x)


def win(v, name):
    w = (v.get("windows") or {}).get(name) or {}
    return w.get("delta_mb"), w.get("wddm_samples"), w.get("plateau")


rows = []
for L in ORDER:
    r = json.load(io.open(os.path.join(HERE, "runs", L, "run.json"), encoding="utf-8"))
    v = r.get("v3dprime") or {}
    geo = r.get("geometry") or {}
    b = dict(label=L, kind=r.get("kind") or "single", status=r.get("status"),
             verdict=r.get("verdict_v3dprime"), reasons=v.get("reasons") or [],
             causes=v.get("causes") or [], oom_clip=g(v, "oom_clip.clip"),
             baseline=v.get("baseline_label"), wall_s=g(r, "wall.seconds"),
             wall_start=g(r, "wall.start_iso"), wall_end=g(r, "wall.end_iso"),
             used_ok=g(r, "used_matches_request.ok"),
             width=geo.get("width"), height=geo.get("height"),
             commit_peak_gib=g(r, "h3.commit_peak_gib"))
    if b["kind"] == "chain":
        dec = g(r, "h3.decode") or {}
        dd, dn, dp = win(v, "decode")
        b.update(decode_s=dec.get("seconds"), decode_frames=dec.get("frames"),
                 decode_peak_mb=dec.get("peak_vram_mb"), decode_res_mb=dec.get("peak_vram_reserved_mb"),
                 decode_delta_mb=dd, decode_n=dn, decode_plateau=dp,
                 source_video=bool(geo.get("source_video")))
        refs_by_clip = geo.get("clip_references") or []
        kf = (geo.get("keyframes") or {}).get("frame_idx") or []
        clips = []
        for c in g(r, "h3.clips") or []:
            i = c["clip"]
            seq = c.get("sequence") or c.get("sequence_estimate") or {}
            pr = c.get("predict") or {}
            vr = c.get("vram16") or {}
            d, wn, pl = win(v, "clip%d.denoise" % i)
            refs = refs_by_clip[i] if i < len(refs_by_clip) and refs_by_clip[i] else []
            clips.append(dict(
                clip=i, F=c.get("num_frames"),
                K=len(kf) if (i == 0 and not b["source_video"]) else 0,
                refs=[str(x.get("kind")) + str(x.get("slot", "")) for x in refs],
                S_pred=pr.get("total"), S_meas=seq.get("total"), S_src=g(c, "axes.S_source"),
                text=seq.get("text"), text_minus_pred=g(c, "axes.text_minus_predicted"),
                cap=vr.get("cap_mib"), den_peak=g(c, "phase_peak_vram_mb.denoise"),
                den_res=g(c, "phase_peak_vram_reserved_mb.denoise"),
                retries=vr.get("num_alloc_retries"), num_ooms=vr.get("num_ooms"),
                inactive_split=vr.get("inactive_split_peak_mib"),
                s_step_meta=c.get("sec_per_step_meta"), s_step_log=c.get("sec_per_step_log"),
                denoise_s=g(c, "phase_seconds.denoise"), dit_source=c.get("dit_source"),
                prefetch=c.get("block_swap_prefetch_used"), mask=c.get("mask"),
                den_delta=d, den_n=wn, den_plateau=pl))
        b["clips"] = sorted(clips, key=lambda c: c["clip"])
    else:
        seq = g(r, "h3.sequence") or {}
        vr = g(r, "h3.vram16") or {}
        d, wn, pl = win(v, "denoise")
        dd, dn, dp = win(v, "decode")
        b.update(F=geo.get("num_frames"), K=g(r, "tokens.keyframes"),
                 S_pred=g(r, "predict.S_estimate"), S_meas=seq.get("total"), text=seq.get("text"),
                 cap=vr.get("cap_mib"), den_peak=g(r, "h3.phase_peak_vram_mb.denoise"),
                 den_res=g(r, "h3.phase_peak_vram_reserved_mb.denoise"),
                 retries=vr.get("num_alloc_retries"), inactive_split=vr.get("inactive_split_peak_mib"),
                 s_step_meta=g(r, "h3.sec_per_step_meta"), s_step_log=g(r, "h3.sec_per_step_log"),
                 den_delta=d, den_n=wn, decode_delta_mb=dd, decode_n=dn,
                 decode_frames=geo.get("num_frames"),
                 decode_res_mb=g(r, "h3.phase_peak_vram_reserved_mb.decode"))
    rows.append(b)

io.open(os.path.join(OUT, "points.json"), "w", encoding="utf-8").write(
    json.dumps(rows, ensure_ascii=False, indent=1))

COLS = ["ラベル", "形", "クリップ", "F", "K", "参照", "S 予測／実測", "text", "cap MiB",
        "den 山／予約 MiB", "retries", "inactive_split MiB", "s/step（meta／log）",
        "den Δ MB（標本）", "DiT の出どころ", "先読み", "decode フレーム", "decode Δ MB（標本）",
        "decode 予約 MiB", "判定", "理由", "原因", "所要秒"]
lines = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
for b in rows:
    if b["kind"] == "chain":
        shape = "%d×%d [%s]%s" % (b["width"], b["height"], ", ".join(
            "%s%s" % (c["F"], " K%d" % c["K"] if c["K"] else "") for c in b["clips"]),
            " V2V" if b["source_video"] else "")
        for j, c in enumerate(b["clips"]):
            f = j == 0
            src = "" if c["S_src"] == "metadata" else "（%s）" % c["S_src"]
            cells = [b["label"] if f else "", shape if f else "", str(c["clip"]), raw(c["F"]), raw(c["K"]),
                     ",".join(c["refs"]) or "—", "%s／%s%s" % (n(c["S_pred"]), n(c["S_meas"]), src),
                     n(c["text"]), n(c["cap"]), "%s／%s" % (n(c["den_peak"]), n(c["den_res"])),
                     n(c["retries"]), n(c["inactive_split"]),
                     "%s／%s" % (raw(c["s_step_meta"]), raw(c["s_step_log"])),
                     "%s（%s）" % (raw(c["den_delta"]), n(c["den_n"])), raw(c["dit_source"]), raw(c["prefetch"]),
                     n(b["decode_frames"]) if f else "",
                     "%s（%s）" % (raw(b["decode_delta_mb"]), n(b["decode_n"])) if f else "",
                     n(b["decode_res_mb"]) if f else "",
                     raw(b["verdict"]) if f else "", (",".join(b["reasons"]) or "—") if f else "",
                     (",".join(b["causes"]) or "—") if f else "", n(b["wall_s"]) if f else ""]
            lines.append("| " + " | ".join(cells) + " |")
    else:
        cells = [b["label"], "%d×%d×%s・K=%s（Single）" % (b["width"], b["height"], b["F"], b["K"]), "—",
                 raw(b["F"]), raw(b["K"]), "—", "%s／%s" % (n(b["S_pred"]), n(b["S_meas"])), n(b["text"]),
                 n(b["cap"]), "%s／%s" % (n(b["den_peak"]), n(b["den_res"])), n(b["retries"]),
                 n(b["inactive_split"]), "%s／%s" % (raw(b["s_step_meta"]), raw(b["s_step_log"])),
                 "%s（%s）" % (raw(b["den_delta"]), n(b["den_n"])), "—", "—", n(b["decode_frames"]),
                 "%s（%s）" % (raw(b["decode_delta_mb"]), n(b["decode_n"])), n(b["decode_res_mb"]),
                 raw(b["verdict"]), ",".join(b["reasons"]) or "—", ",".join(b["causes"]) or "—", n(b["wall_s"])]
        lines.append("| " + " | ".join(cells) + " |")
txt = "\n".join(lines) + "\n"
io.open(os.path.join(OUT, "points_table.md"), "w", encoding="utf-8").write(txt)
print(txt)

# 照合: 予測と実測の S・text の差、cap の帯、照合の真偽、先読み、mask
print("## 照合")
for b in rows:
    if b["kind"] != "chain":
        print("%s: S 差 %s・cap %s・used_ok %s" % (
            b["label"], None if b["S_meas"] is None else b["S_meas"] - b["S_pred"], b["cap"], b["used_ok"]))
        continue
    for c in b["clips"]:
        print("%s clip%d: S 差 %s・text 差 %s・cap %s・prefetch %s・mask %s・num_ooms %s・used_ok %s" % (
            b["label"], c["clip"], None if c["S_meas"] is None else c["S_meas"] - c["S_pred"],
            c["text_minus_pred"], c["cap"], c["prefetch"], c["mask"], c["num_ooms"], b["used_ok"]))
caps = sorted({c["cap"] for b in rows if b["kind"] == "chain" for c in b["clips"] if c["cap"] is not None}
              | {b["cap"] for b in rows if b["kind"] != "chain" and b["cap"] is not None})
print("cap の値の集合:", caps)
tot = sum(b["wall_s"] or 0 for b in rows)
print("所要の合計 %.1f 秒＝%.1f 分（%d 点。自動復帰の warm を含む）" % (tot, tot / 60, len(rows)))
print("最初の開始 %s・最後の終了 %s" % (rows[0]["wall_start"], rows[-1]["wall_end"]))
