# -*- coding: utf-8 -*-
"""台帳 §1-285（2026-11-13）H3 の形式別の快適上限 -- 結果の突き合わせ台本。

計画 graceful-fluttering-reddy §5（判定 v3''）・§7（点の表）・§10（結論の形）・§12（検証）。
runs/<label>/{run.json, metadata.json, point.json, request.json, job.json, wddm.csv} だけを読み、
次を出す（GPU・サーバーには触れない）:

  analysis/results_table.md   点ごとの表・形式ごとの結論・予測との照合・整合の検査
  analysis/results.json       同じ内容の機械可読

判定は run.json の verdict を写さず、ここで v3'' を独立に組み直して run.json と突き合わせる:
  OOM  = job の error が "GPU_OOM:" で始まる
  退避 = denoise／decode の窓の共有メモリの中央値（wddm.csv から再計算）が同じ形式の基準点より
         +300MB を超える（calib.py analyse_h3 と同じ ">"）
  降格 = 完走して block_swap_prefetch_used が "on" でない

  set PYTHONUTF8=1 & ..\\..\\.venv\\Scripts\\python.exe check_results_quant.py
"""
import glob
import io
import json
import os
import re
import statistics
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "analysis")
MiB = 2 ** 20

# ---------------------------------------------------------------- 固定値（計画から）
FORMAT_OF = {  # 計画の transformer（point.json）→ 形式
    "default": "int8",
    "minimax_h3_fl2va_pruned_w4a8_mixed": "w4a8",
    "minimax_h3_fl2va_pruned_fp8_scaled": "fp8",
    "10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized": "eros",
}
FORMATS = ("int8", "w4a8", "fp8", "eros")
EXPECT_LOAD = {  # §5「形式の記録」
    "int8": ({"int8_tensorwise": 200}, None),
    "fp8": ({"float8_e4m3fn": 200}, "scaled"),
    "w4a8": ({"asym_w4a8_int8": 200}, None),
    "eros": ({"asym_w4a8_int8": 180, "int8_tensorwise": 20}, None),
}
LOAD_FILE = {"int8": "load_int8.txt", "w4a8": "load_w4a8.txt", "fp8": "load_fp8.txt", "eros": "load_eros.txt"}
PREDICT_AB = {  # §4.2: 形式の山 = int8 の山 + a×S + b（a バイト／行・b MiB）
    "int8": (0, 0.0), "fp8": (-4, -2.0), "w4a8": (5372, -920.0), "eros": (5372, 0.0),
}
# int8 の現行値（快適上限表 §17・記録簿 §196）
INT8_CU = {"1080": (345, 362), "1440": (192, 209)}
INT8_LINE = 219516
# int8 の完走点の予約の山（11-04 runs/<label>/run.json の h3.phase_peak_vram_reserved_mb.denoise・
# cap_mib。FL2VA default・sage・4 ステップ・K=1。記録簿 §196 の数値。計画 §4.2 の「余裕」の元）
INT8_REF = {
    (1920, 1088, 277): {"label": "s1080_277", "S": 172322, "cap": 13999, "res_denoise": 13376, "retries": 4},
    (1920, 1088, 328): {"label": "s1080_328", "S": 203092, "cap": 13999, "res_denoise": 12964, "retries": 9},
    (1920, 1088, 345): {"label": "s1080_345", "S": 213348, "cap": 13999, "res_denoise": 13486, "retries": 9},
    (2560, 1472, 192): {"label": "s1440_192", "S": 217798, "cap": 13999, "res_denoise": 13764, "retries": 9},
    (1280, 768, 481): {"label": "s720_481", "S": 139882, "cap": 13999, "res_denoise": 11352, "retries": 4},
}
CAP_BAND = (13999, 14025)
CAP_RESERVE_MB = 50      # §5: 帯から 50MiB 以上外れた点は留保
MARGIN_RESERVE_MB = 50   # §5: 余裕 50MiB 未満の境目の点は留保
PLATEAU_MB = 300.0
JUDGED = ("denoise", "decode")
OOM_PREFIX = "GPU_OOM:"
DEFAULT_SHIFT = (12.0, 3.0)
# 監督の裁定（2026-11-13）: 事故の runs は表に残し、注記する
ACCIDENT = {"b1080_w4a8": "共有メモリの標本なし（二重起動の事故。progress.md 参照）・数値は 1 本目のジョブの値"}
RES_OF_H = {1088: "1080", 1472: "1440", 768: "720", 384: "384"}

LABEL_RE = re.compile(r"^(?:s(?P<sres>\d+)_(?P<F>\d+)_(?P<sfmt>int8|w4a8|fp8|eros)(?P<r2>_r2)?"
                      r"|b(?P<bres>\d+)_(?P<bfmt>int8|w4a8|fp8|eros)(?P<br2>_r2)?"
                      r"|warm__(?P<wtr>.+?)(?P<rec>_rec\d+|_x)?)$")  # _rec<時刻> は OOM 後の復帰の暖機（drive2.sh recover）、_x は弾 X（計画外の 2 点・オーナー了承）の煙


# ---------------------------------------------------------------- 読み出し
def jload(path):
    if not os.path.exists(path):
        return None
    return json.load(io.open(path, encoding="utf-8"))


def get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


CACHE = {}


def files(label):
    if label not in CACHE:
        d = os.path.join(HERE, "runs", label)
        CACHE[label] = {n: jload(os.path.join(d, n + ".json")) for n in ("run", "metadata", "point", "request", "job")}
    return CACHE[label]


def labels():
    out = []
    for f in sorted(glob.glob(os.path.join(HERE, "runs", "*", "run.json"))):
        L = os.path.basename(os.path.dirname(f))
        if LABEL_RE.match(L):
            out.append(L)
    return out


MANIFEST = None


def manifest_status(label):
    """manifest.jsonl のその点の行の status の一覧（submit_failed の有無を注記に使う）。"""
    global MANIFEST
    if MANIFEST is None:
        MANIFEST = []
        p = os.path.join(HERE, "manifest.jsonl")
        if os.path.exists(p):
            for line in io.open(p, encoding="utf-8"):
                try:
                    MANIFEST.append(json.loads(line))
                except ValueError:
                    pass
    return [(m.get("status"), m.get("http_status")) for m in MANIFEST if m.get("label") == label]


def ts(text):
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f").timestamp()


def wddm(label):
    rows = []
    p = os.path.join(HERE, "runs", label, "wddm.csv")
    if not os.path.exists(p):
        return rows
    for line in io.open(p, encoding="ascii", errors="replace"):
        parts = line.strip().split(",")
        if len(parts) < 4 or parts[0] == "timestamp":
            continue
        try:
            rows.append((ts(parts[0]), float(parts[2])))
        except ValueError:
            pass
    return rows


def window_median(label, name):
    """共有メモリの中央値を wddm.csv から組み直す（窓の端は run.json の v3dprime.windows）。"""
    w = get(files(label)["run"], "v3dprime.windows." + name) or {}
    if not w.get("start_iso") or not w.get("end_iso"):
        return None, 0
    t0, t1 = ts(w["start_iso"]), ts(w["end_iso"])
    v = [s for t, s in wddm(label) if t0 <= t <= t1]
    return (round(statistics.median(v), 1) if v else None), len(v)


# ---------------------------------------------------------------- OOM の文
UNIT = r"([\d.]+) (GiB|MiB)"
OOMRE = re.compile(r"Tried to allocate %s\. GPU \d+ has a total capacity of %s of which %s is free\. %s allowed; "
                   r"Of the allocated memory %s is allocated by PyTorch, and %s is reserved" % ((UNIT,) * 6))


def gib(v, u):
    return round(float(v) / 1024.0, 4) if u == "MiB" else float(v)


def oom_numbers(error):
    m = OOMRE.search(error or "")
    if not m:
        return None
    g = m.groups()
    vals = [gib(g[i], g[i + 1]) for i in range(0, 12, 2)]
    return {"要求_GiB": vals[0], "総量_GiB": vals[1], "free_GiB": vals[2], "許可_GiB": vals[3],
            "割り当て_GiB": vals[4], "予約済みの空き_GiB": vals[5],
            "表記": {k: "%s %s" % (g[i], g[i + 1]) for k, i in (("要求", 0), ("free", 4), ("許可", 6), ("割り当て", 8), ("予約済みの空き", 10))},
            "原文": m.group(0)}


SAGE_SRC = None
LOWVRAM = None


def sage_lines():
    global SAGE_SRC
    if SAGE_SRC is None:
        SAGE_SRC = set()
        for f in glob.glob(os.path.join(REPO, ".venv-engine-h3", "Lib", "site-packages", "sageattention", "**", "*.py"),
                           recursive=True):
            for l in io.open(f, encoding="utf-8", errors="replace"):
                if l.strip():
                    SAGE_SRC.add(l.strip())
    return SAGE_SRC


def lowvram_mlp_lines():
    """engine_h3/lowvram.py の block_forward のうち FFN の段（n2 = block.norm2 から関数の終わりまで）。"""
    global LOWVRAM
    if LOWVRAM is None:
        src = io.open(os.path.join(REPO, "engine_h3", "lowvram.py"), encoding="utf-8").read().splitlines()
        start = next(i for i, l in enumerate(src) if l.strip() == "n2 = block.norm2(x2)")
        end = next(i for i in range(start + 1, len(src)) if src[i].startswith("def "))
        LOWVRAM = {"mlp_lines": set(range(start + 1, end + 1)),
                   "mlp_text": {src[i].strip() for i in range(start, end) if src[i].strip()},
                   "src": src}
    return LOWVRAM


def oom_place(error):
    """traceback の行から場所を分類: o の確保 / sage の中 / MLP（FFN の段）/ その他。"""
    head = (error or "").split("torch.OutOfMemoryError")[0]
    lines = [l for l in head.splitlines() if l.strip() and not set(l.strip()) <= set("^~")]
    code = lines[-1].strip() if lines else ""
    frames = re.findall(r'(?:File "([^"\n]*)"|([^\s"]*)"), line (\d+), in (\w+)', head)
    frames = [((a or b), int(n), fn) for a, b, n, fn in frames]
    last = frames[-1] if frames else None
    lv = lowvram_mlp_lines()
    if code.startswith("o = torch.empty"):
        ok = "o = torch.empty((s, inner), dtype=torch.bfloat16, device=x.device)" in [l.strip() for l in lv["src"]]
        return "o の確保", code, last, ok
    # traceback は途中が切られ、最後の行も頭が欠けることがある（11-04 s1440_209）→ 部分一致で探す
    if any("sageattention" in f for f, _, _ in frames) or (len(code) >= 20 and any(code in l for l in sage_lines())):
        return "sage の中", code, last, True
    if any(f.endswith("lowvram.py") and n in lv["mlp_lines"] for f, n, _ in frames) or code in lv["mlp_text"]:
        return "MLP", code, last, True
    if last and "comfy_kitchen" in last[0] and last[2].startswith("quantize_int8_rowwise"):
        return "線形層の入力の量子化", code, last, True
    return "その他", code, last, None


def linear_of_request(req_gib, S):
    """「線形層の入力の量子化」の要求を S×K（int8・2MiB 単位に切り上げ）と照らし、層を名指す。
    K は engine_h3/dit.py: qkv_proj の入力 5,376（hidden）、out_proj の入力 7,168（heads×head_dim）。"""
    for name, k in (("out_proj", 7168), ("qkv", 5376)):
        b = -(-S * k // (2 * MiB)) * 2 * MiB
        if abs(b / 2 ** 30 - req_gib) < 0.005:
            return "%s（要求＝S×%s の int8）" % (name, format(k, ","))
    return None


# ---------------------------------------------------------------- 1 点
def fmt_of_label(L):
    m = LABEL_RE.match(L)
    if m.group("sfmt"):
        return m.group("sfmt")
    if m.group("bfmt"):
        return m.group("bfmt")
    return FORMAT_OF.get(m.group("wtr"))


def role_of(L):
    m = LABEL_RE.match(L)
    return "点" if m.group("sfmt") else ("基準点" if m.group("bfmt") else "煙")


def load_ok_of(L, fmt):
    r = files(L)["run"]
    load = get(r, "worker_conditions.load") or {}
    if "load_ok_dit" in load and load["load_ok_dit"] is not None:
        return load["load_ok_dit"], "run.json"
    p = os.path.join(HERE, LOAD_FILE.get(fmt, ""))
    if fmt in LOAD_FILE and os.path.exists(p):
        line = [l for l in io.open(p, encoding="utf-8") if "LOAD_OK {" in l]
        if line:
            dit = (json.loads(line[-1][line[-1].index("{"):]).get("dit") or {})
            return ({k: dit.get(k) for k in ("quant_layers", "quant_layers_by_format", "fp8_mode", "quant_specs")},
                    "%s（run.json に欄なし＝差分 1 より前の点）" % LOAD_FILE[fmt])
    return None, "なし"


def shift_of(r, meta):
    sv, sa = get(meta, "h3.shift_video"), get(meta, "h3.shift_audio")
    if sv is not None and sa is not None:
        return sv, sa, "metadata"
    g = get(r, "worker_conditions.generate_line") or ""
    m = re.search(r"shift=([\d.]+)/([\d.]+)", g)
    if m:
        return float(m.group(1)), float(m.group(2)), "ワーカーの generate 行（metadata なし）"
    return DEFAULT_SHIFT[0], DEFAULT_SHIFT[1], "既定 12／3（記録なし）"


def point_row(L):
    F = files(L)
    r, meta, pt, rq, job = F["run"], F["metadata"] or {}, F["point"] or {}, F["request"] or {}, F["job"] or {}
    g = r["geometry"]
    fmt = fmt_of_label(L)
    tr_plan = pt.get("transformer")
    dit_meta = get(meta, "h3.phases.denoise") or {}
    vram16 = dit_meta.get("vram16") or {}
    completed = r.get("status") == "completed"
    err = r.get("error") or job.get("error") or ""
    is_base = bool(pt.get("is_baseline"))
    smoke = bool(pt.get("smoke")) or role_of(L) == "煙"
    # --- S
    S_pred = get(r, "tokens.predict.S_estimate")
    if completed and dit_meta.get("sequence"):
        S_meas, S_src = dit_meta["sequence"]["total"], "metadata"
    else:
        S_meas, S_src = get(r, "h3.axes.S"), get(r, "h3.axes.S_source")
    # --- 窓（再計算）
    wins = {}
    for n in JUDGED:
        med, n_s = window_median(L, n)
        wins[n] = {"median_mb": med, "samples": n_s,
                   "run_json_median_mb": get(r, "v3dprime.windows.%s.shared_median_mb" % n)}
    base = get(r, "v3dprime.baseline_label")
    # --- 判定 v3''
    oom = err.startswith(OOM_PREFIX)
    reasons = []
    if is_base or smoke:
        verdict = "煙" if smoke else "基準点"
    elif not completed:
        verdict = "OOM" if oom else (r.get("status") or "不明")
    else:
        base_empty = []
        for n in JUDGED:
            if base is None:
                continue
            bmed, bn = window_median(base, n)
            if bn == 0:
                base_empty.append(n)
            m = wins[n]["median_mb"]
            wins[n]["baseline_median_mb"] = bmed
            wins[n]["baseline_samples"] = bn
            if m is not None and bmed is not None:
                wins[n]["delta_mb"] = round(m - bmed, 1)
                if m > bmed + PLATEAU_MB:
                    reasons.append("退避:%s" % n)
        if (meta.get("block_swap_prefetch_used") or get(r, "used.block_swap_prefetch_used")) != "on":
            reasons.append("降格")
        decidable = (base is not None and not base_empty
                     and all(wins[n]["median_mb"] is not None and wins[n]["samples"] >= 10 for n in JUDGED))
        verdict = "・".join(reasons) if reasons else ("快適" if decidable else "判定不能")
    # --- OOM の中身
    oomd = None
    if oom:
        place, code, last, src_ok = oom_place(err)
        nums = oom_numbers(err)
        S_here = get(r, "h3.axes.S")
        if place == "線形層の入力の量子化" and nums and S_here:
            which = linear_of_request(nums["要求_GiB"], S_here)
            place = "%s・%s" % (place, which or "層は要求の大きさから決まらない")
        done = get(r, "h3_spans.denoise.last_done")
        loc = get(r, "v3dprime.oom_location") or {}
        oomd = {"場所": place, "行": code, "最後のフレーム": ("%s:%d %s" % last) if last else None,
                "行の実在": src_ok, "段": loc.get("stage"), "線に数える": loc.get("counts_for_line"),
                "落ちたステップ": (done or 0) + 1 if loc.get("stage") == "denoise" else None,
                "job_current_step": job.get("current_step"), "total_steps": job.get("total_steps"),
                "数": oom_numbers(err)}
    res_dn = dit_meta.get("peak_vram_reserved_mb")
    res_dc = get(meta, "h3.phases.decode.peak_vram_reserved_mb")
    al_dn = dit_meta.get("peak_vram_mb")
    al_dc = get(meta, "h3.phases.decode.peak_vram_mb")
    cap = vram16.get("cap_mib")
    sv, sa, ssrc = shift_of(r, meta)
    lo, lo_src = load_ok_of(L, fmt)
    return {
        "label": L, "役割": role_of(L), "形式": fmt, "形式（計画の transformer から）": FORMAT_OF.get(tr_plan),
        "計画の transformer": tr_plan,
        "重みの名前（metadata）": get(meta, "models.selection.transformer.name"),
        "重みのファイル（metadata）": get(meta, "models.selection.transformer.file"),
        "transformer_path": get(r, "worker_conditions.load.transformer_path"),
        "load_ok_dit": lo, "load_ok_dit の出所": lo_src,
        "W": g["width"], "H": g["height"], "F": g["num_frames"], "res": RES_OF_H.get(g["height"], str(g["height"])),
        "K": len(get(g, "keyframes.frame_idx") or []), "steps": g.get("num_inference_steps"),
        "attention": g.get("attention_backend"),
        "S_predict": S_pred, "S_実測": S_meas, "S_実測の出所": S_src,
        "S_差": (S_meas - S_pred) if (S_meas is not None and S_pred is not None) else None,
        "status": r.get("status"), "判定": verdict, "判定（run.json）": r.get("verdict_v3dprime"),
        "基準点": base, "窓": wins,
        "cap_MiB": cap, "予約の山_denoise": res_dn, "予約の山_decode": res_dc,
        "割り当ての山_denoise": al_dn, "割り当ての山_decode": al_dc,
        # §4.2 の余裕は denoise の予約の山で取る（cap はノイズ除去の段に掛かる。11-04 の 1440 は decode が cap を超える）
        "余裕_MiB": (cap - res_dn) if (cap is not None and res_dn is not None) else None,
        "retries": vram16.get("num_alloc_retries"), "inactive_split_MiB": vram16.get("inactive_split_peak_mib"),
        "1ステップ秒_meta": get(r, "h3.sec_per_step_meta"), "1ステップ秒_log": get(r, "h3.sec_per_step_log"),
        "OOM": oomd, "shift": [sv, sa], "shift の出所": ssrc,
        "used_check_h3": get(r, "used_matches_request.ok"),
        "used_check_h3_mismatches": get(r, "used_matches_request.mismatches"),
        "prefetch": meta.get("block_swap_prefetch_used") or get(r, "used.block_swap_prefetch_used"),
        "run_json_cap": get(r, "h3.vram16.cap_mib"), "run_json_res_denoise": get(r, "h3.phase_peak_vram_reserved_mb.denoise"),
        "run_json_S": get(r, "h3.axes.S"),
        "request_matches_point": (rq.get("body") or {}).get("num_frames") == g["num_frames"]
        and (rq.get("body") or {}).get("width") == g["width"] and (rq.get("body") or {}).get("height") == g["height"],
        "所要秒": get(r, "wall.seconds"), "開始": get(r, "wall.start_iso") or "",
    }


# ---------------------------------------------------------------- 快適の回数・結論
def shape_key(p):
    return (p["形式"], p["W"], p["H"], p["F"], p["K"], p["steps"], p["attention"])


def add_counts(rows):
    for p in rows:
        if p["役割"] != "点":
            p["快適の回数"] = None
            continue
        same = [q for q in rows if q["役割"] == "点" and shape_key(q) == shape_key(p)]
        p["快適の回数"] = sum(1 for q in same if q["判定"] == "快適")
        p["同じ形の点"] = [q["label"] for q in same]


def conclude(rows):
    out = {}
    for fmt in FORMATS:
        per = {}
        for res, (W, H) in (("1080", (1920, 1088)), ("1440", (2560, 1472))):
            pts = [p for p in rows if p["役割"] == "点" and p["形式"] == fmt and p["W"] == W and p["H"] == H
                   and p["K"] == 1 and p["steps"] == 4 and p["attention"] == "sage"]
            if not pts:
                continue
            byF = {}
            for p in pts:
                byF.setdefault(p["F"], []).append(p)
            ok = sorted(F for F, ps in byF.items() if any(q["判定"] == "快適" for q in ps))
            ng = {}
            excluded = []
            excluded_u = []
            for F, ps in byF.items():
                for q in ps:
                    if q["判定"] == "OOM":
                        if q["OOM"] and q["OOM"]["線に数える"] is False:
                            excluded.append(q["label"]); continue
                        ng.setdefault(F, []).append(("OOM", q["label"]))
                    elif q["判定"] in ("判定不能",):
                        excluded_u.append(q["label"])
                    elif q["判定"] not in ("快適",):
                        ng.setdefault(F, []).append((q["判定"], q["label"]))
            c = max(ok) if ok else None
            U = min(ng) if ng else None
            notes = []
            if c is not None and U is not None and c > U:
                notes.append("逆転: 快適 %d が非快適 %d より上" % (c, U))
            if c is not None:
                n_ok = sum(1 for q in byF[c] if q["判定"] == "快適")
                if n_ok < 2:
                    notes.append("c=%d は快適 1 回（_r2 は弾 Z。§5 の根拠を添える）" % c)
            U_fixed = True
            if U is not None:
                kinds = [k for k, _ in ng[U]]
                if "OOM" not in kinds and len(kinds) < 2:
                    U_fixed = False
                    notes.append("U=%d は %s 1 回（退避・降格は 2 回要る＝未確定）" % (U, kinds[0]))
            if excluded_u:
                notes.append("判定不能で c・U に数えない点: %s" % "・".join(excluded_u))
            # 基準点: 形式×解像度で「標本のある最新のもの」（wall の開始が最も遅い基準点）。
            # 各点の退避は run.json の baseline_label で測るので、食い違えば注記する。
            bases = [b for b in rows if b["役割"] == "基準点" and b["形式"] == fmt and (b["W"], b["H"]) == (W, H)]
            with_s = [b for b in bases if all(b["窓"][k]["samples"] > 0 for k in JUDGED)]
            chosen = max(with_s, key=lambda b: b["開始"])["label"] if with_s else None
            for q in pts:
                if q["基準点"] != chosen:
                    notes.append("%s の baseline_label %s が標本のある最新の基準点 %s と違う" % (q["label"], q["基準点"], chosen))
            if excluded:
                notes.append("線から外した OOM（ノイズ除去より前の段）: %s" % "・".join(excluded))
            Sof = lambda F: byF[F][0]["S_実測"]
            band = [Sof(c) if c is not None else None, Sof(U) if U is not None else None]
            ic, iu = INT8_CU[res]
            if c is None or U is None or not U_fixed:
                direction = "未確定"
            elif c == ic and U == iu:
                direction = "同"
            elif c < ic:
                direction = "下"
            elif c >= iu:
                direction = "上"
            else:
                direction = "未確定"
            per[res] = {"基準点（標本のある最新）": chosen, "c": c, "U": U, "帯_S": band, "向き": direction, "点": sorted(p["label"] for p in pts),
                        "int8 の現行": {"c": ic, "U": iu}, "注記": notes}
        if not per:
            continue
        overlap = None
        if all(r in per and None not in per[r]["帯_S"] for r in ("1080", "1440")):
            lo = max(per[r]["帯_S"][0] for r in ("1080", "1440"))
            hi = min(per[r]["帯_S"][1] for r in ("1080", "1440"))
            overlap = {"重なる": lo < hi, "共通の帯_S": [lo, hi] if lo < hi else None,
                       "int8 の線 219,516 が共通の帯の中": (lo < INT8_LINE <= hi) if lo < hi else False}
        if fmt == "int8":
            out[fmt] = {"解像度": per, "帯の重なり": overlap,
                        "結論の形": "対照（弾 I・§7.2）: 1080 の U=362 が %s%s" % (
                            "OOM（環境は 11-04 と同じかは §5 の 4 つの数で見る）" if per.get("1080", {}).get("U") == 362 else "OOM でない → 止めて監督へ",
                            "" if "1440" not in per else "。追加の点（弾 X）: 1440×%s が %s（Kijai との比較用。線の判定には使わない）" % (
                                "・".join(str(F) for F in sorted({q["F"] for q in rows if q["役割"] == "点" and q["形式"] == "int8" and q["H"] == 1472})),
                                "・".join(sorted({q["判定"] for q in rows if q["役割"] == "点" and q["形式"] == "int8" and q["H"] == 1472}))))}
            continue
        if fmt == "eros":
            p345 = [p for p in rows if p["役割"] == "点" and p["形式"] == "eros" and (p["W"], p["H"], p["F"]) == (1920, 1088, 345)]
            e_pts = [p for p in rows if p["役割"] == "点" and p["形式"] == "eros"]
            if not p345:
                form = "未測"
            elif all(p["判定"] == "OOM" for p in e_pts) and len({p["F"] for p in e_pts}) > 1:
                low = min(e_pts, key=lambda p: p["S_実測"])
                form = "%s とも OOM → Eros の線は S %s より下。量は未定（オーナー裁定で打ち止め）" % (
                    "・".join(str(F) for F in sorted({p["F"] for p in e_pts}, reverse=True)), format(low["S_実測"], ","))
            elif any(p["判定"] == "OOM" for p in p345) and not any(p["判定"] == "快適" for p in e_pts):
                form = "OOM → Eros の線は 213,348 より下。量は未定（1 点の裁定）"
            elif any(p["判定"] == "OOM" for p in p345):
                form = "345 は OOM・%s は快適 → 線はその間（量は帯で書く）" % "・".join(
                    str(p["F"]) for p in e_pts if p["判定"] == "快適")
            elif any(p["判定"] == "快適" for p in p345):
                form = "快適 → §4.1 の「山を決めるのは w4a8 のブロック」の読みが誤り"
            else:
                form = "非快適（OOM でない: %s）→ §5 の 2 択に当たらない。監督へ" % "・".join(p["判定"] for p in p345)
        else:
            dirs = {r: per[r]["向き"] for r in per}
            vals = set(dirs.values())
            if len(per) < 2 or "未確定" in vals:
                form = "未確定（%s）" % "・".join("%s=%s" % kv for kv in sorted(dirs.items()))
            elif len(vals) == 1:
                form = {"同": "(同)", "下": "(下)", "上": "(上)"}[vals.pop()]
            else:
                form = "(混) 向きが違う（%s）" % "・".join("%s=%s" % kv for kv in sorted(dirs.items()))
            if overlap and overlap["重なる"] is False and not form.startswith("(混)"):
                form += "／(混) 1080 と 1440 の帯が S で重ならない"
        out[fmt] = {"解像度": per, "帯の重なり": overlap, "結論の形": form}
    return out


# ---------------------------------------------------------------- 予測（§4.2）
# Kijai の実測から当て直した a（analysis/mechanism_w4a8.md §5。b は形式の値のまま）。照合の参考に並べる
ALT_READINGS = {"割り当ての読み a=+5,116": 5116, "予約の読み a=+7,228（弾 X で否定）": 7228}
MATCH_MB = 50  # 山の差がこれ未満なら「一致」（§5 の留保の幅 50MiB と同じ目盛り）


def match_of(pred_ok, verdict, meas, pred_res):
    """予測と実測の照合の言葉。判定（快適か OOM か）と、完走した点は山の差で見る。"""
    if verdict == "OOM":
        return "一致（OOM）" if not pred_ok else "外れ（予測は快適・実測は OOM＝山が高い側）"
    if verdict != "快適":
        return "対象外（%s）" % verdict
    if not pred_ok:
        return "外れ（予測は OOM・実測は快適＝山が低い側）"
    d = meas - pred_res
    if abs(d) < MATCH_MB:
        return "一致（差 %+.1f MiB）" % d
    return "判定は一致・山は予測より%s側（差 %+.1f MiB）" % ("高い" if d > 0 else "低い", d)
def predictions(rows):
    out = []
    # int8 の同じ形の完走点: 今回の runs（弾 X の s1440_175_int8 など）を優先し、無ければ 11-04 の定数
    cur = {}
    for q in rows:
        if q["役割"] == "点" and q["形式"] == "int8" and q["判定"] == "快適" and q["予約の山_denoise"] is not None:
            cur[(q["W"], q["H"], q["F"])] = {"label": q["label"] + "（今回）", "S": q["S_実測"], "cap": q["cap_MiB"],
                                            "res_denoise": q["予約の山_denoise"], "retries": q["retries"]}
    for p in rows:
        if p["役割"] != "点" or p["形式"] == "int8":
            continue
        ref = cur.get((p["W"], p["H"], p["F"])) or INT8_REF.get((p["W"], p["H"], p["F"]))
        if not ref or p["K"] != 1 or p["steps"] != 4:
            continue
        a, b = PREDICT_AB[p["形式"]]
        add = a * ref["S"] / MiB + b
        pred_res = ref["res_denoise"] + add
        pred_margin = ref["cap"] - pred_res
        meas = p["予約の山_denoise"]
        out.append({"label": p["label"], "形式": p["形式"], "int8 の点": ref["label"], "S": ref["S"],
                    "int8 の予約の山": ref["res_denoise"], "int8 の余裕": ref["cap"] - ref["res_denoise"],
                    "a×S＋b_MiB": round(add, 1), "予測の予約の山": round(pred_res, 1), "予測の余裕": round(pred_margin, 1),
                    "予測": "快適" if pred_margin >= 0 else "OOM",
                    "実測の判定": p["判定"], "実測の予約の山": meas,
                    "差_MiB（実測−予測）": round(meas - pred_res, 1) if meas is not None else None,
                    "実測の cap": p["cap_MiB"], "実測の余裕": p["余裕_MiB"], "実測の retries": p["retries"],
                    "照合": match_of(pred_margin >= 0, p["判定"], meas, pred_res),
                    "別の読みの余裕": {k: round(ref["cap"] - ref["res_denoise"] - (aa * ref["S"] / MiB + b), 1)
                                     for k, aa in ALT_READINGS.items()} if p["形式"] in ("w4a8", "eros") else None})
    return out


# ---------------------------------------------------------------- 整合の検査
def checks(rows):
    viol, notes = [], []
    by = {p["label"]: p for p in rows}
    for p in rows:
        L = p["label"]
        if p["形式"] != p["形式（計画の transformer から）"]:
            viol.append((L, "ラベルの形式 %s と計画の transformer %s（%s）が違う" % (p["形式"], p["計画の transformer"], p["形式（計画の transformer から）"])))
        if p["status"] == "completed":
            if p["used_check_h3"] is not True:
                viol.append((L, "used_check_h3 が真でない: %s %s" % (p["used_check_h3"], p["used_check_h3_mismatches"])))
            if p["重みの名前（metadata）"] != p["計画の transformer"]:
                viol.append((L, "metadata の重みの名前 %s が計画 %s と違う" % (p["重みの名前（metadata）"], p["計画の transformer"])))
            if p["run_json_cap"] != p["cap_MiB"] or p["run_json_res_denoise"] != p["予約の山_denoise"]:
                viol.append((L, "run.json と metadata.json の cap／予約の山が違う"))
            cap = p["cap_MiB"]
            if cap is None:
                viol.append((L, "cap が記録されていない"))
            elif not (CAP_BAND[0] <= cap <= CAP_BAND[1]):
                away = min(abs(cap - CAP_BAND[0]), abs(cap - CAP_BAND[1]))
                (notes if away < CAP_RESERVE_MB else viol).append(
                    (L, "cap %d が帯 13,999〜14,025 の外（%d MiB%s）" % (cap, away, "・留保" if away >= CAP_RESERVE_MB else "")))
            if p["役割"] == "点" and p["余裕_MiB"] is not None and p["余裕_MiB"] < MARGIN_RESERVE_MB:
                notes.append((L, "余裕 %d MiB（50 未満の境目・留保）" % p["余裕_MiB"]))
        else:
            notes.append((L, "used_check_h3 は対象外（%s・metadata なし）" % p["status"]))
        if p["S_差"] != 0:
            viol.append((L, "S の予測 %s と実測 %s（%s）の差 %s" % (p["S_predict"], p["S_実測"], p["S_実測の出所"], p["S_差"])))
        if p["run_json_S"] != p["S_実測"]:
            viol.append((L, "run.json の h3.axes.S %s と実測 %s が違う" % (p["run_json_S"], p["S_実測"])))
        if not p["request_matches_point"]:
            viol.append((L, "request.json の幅・高さ・F が点と違う"))
        # 形式の記録
        exp_by, exp_mode = EXPECT_LOAD[p["形式"]]
        lo = p["load_ok_dit"]
        if lo is None:
            viol.append((L, "load_ok_dit が無い（%s）" % p["load_ok_dit の出所"]))
        else:
            if lo.get("quant_layers_by_format") != exp_by:
                viol.append((L, "quant_layers_by_format %s が期待 %s と違う" % (lo.get("quant_layers_by_format"), exp_by)))
            if p["形式"] == "fp8" and lo.get("fp8_mode") != exp_mode:
                viol.append((L, "fp8_mode %s が期待 scaled と違う" % lo.get("fp8_mode")))
            if p["load_ok_dit の出所"] != "run.json":
                notes.append((L, "load_ok_dit は %s から" % p["load_ok_dit の出所"]))
        # 基準点の形式
        if p["役割"] == "点":
            b = p["基準点"]
            if b is None:
                viol.append((L, "基準点が無い"))
            elif b not in by:
                viol.append((L, "基準点 %s の runs が無い" % b))
            elif by[b]["計画の transformer"] != p["計画の transformer"]:
                viol.append((L, "基準点 %s の transformer %s が点の %s と違う" % (b, by[b]["計画の transformer"], p["計画の transformer"])))
            elif (by[b]["W"], by[b]["H"]) != (p["W"], p["H"]):
                notes.append((L, "基準点 %s の解像度が点と違う" % b))
        # 判定の突き合わせ（独立に組み直した v3'' と run.json）
        if L in ACCIDENT:
            notes.append((L, ACCIDENT[L]))
        elif p["役割"] in ("基準点", "煙") and any(p["窓"][k]["samples"] == 0 for k in JUDGED):
            notes.append((L, "共有メモリの標本なし（%s）" % "・".join("%s n=%d" % (k, p["窓"][k]["samples"]) for k in JUDGED)))
        bad = [m for m in manifest_status(L) if m[0] == "submit_failed"]
        if bad:
            notes.append((L, "manifest.jsonl に submit_failed の行 %d 件（HTTP %s）" % (len(bad), "・".join(str(h) for _, h in bad))))
        if p["役割"] == "点" and p["判定"] == "判定不能":
            empty = [k for k, w in p["窓"].items() if w.get("baseline_samples") == 0]
            notes.append((L, "退避は判定不能（基準点 %s の %s の標本 n=0）" % (p["基準点"], "・".join(empty)) if empty
                          else "退避は判定不能（窓の標本不足）"))
        mine = {"快適": "comfortable", "OOM": "oom", "基準点": "baseline", "煙": "smoke", "判定不能": "undecidable"}.get(p["判定"], "uncomfortable" if ("退避" in p["判定"] or "降格" in p["判定"]) else p["判定"])
        if mine != p["判定（run.json）"]:
            viol.append((L, "判定 %s（ここ）と run.json の %s が違う" % (p["判定"], p["判定（run.json）"])))
        for n, w in p["窓"].items():
            if w["median_mb"] != w["run_json_median_mb"]:
                viol.append((L, "%s の窓の中央値 %s（wddm.csv から）と run.json %s が違う" % (n, w["median_mb"], w["run_json_median_mb"])))
        if p["OOM"]:
            o = p["OOM"]
            if o["数"] is None:
                viol.append((L, "OOM の文の数が読めない"))
            if o["場所"] == "その他":
                notes.append((L, "OOM の場所を分類できない: %s" % o["行"]))
            if o["行の実在"] is False:
                viol.append((L, "OOM の行 %s が現行のソースに無い" % o["行"]))
            if o["段"] == "denoise" and o["job_current_step"] is not None and o["落ちたステップ"] != o["job_current_step"] + 1:
                viol.append((L, "落ちたステップ %s と job.current_step %s が合わない" % (o["落ちたステップ"], o["job_current_step"])))
        if p["shift の出所"] != "metadata":
            notes.append((L, "shift %s／%s は %s" % (p["shift"][0], p["shift"][1], p["shift の出所"])))
        elif tuple(p["shift"]) != DEFAULT_SHIFT:
            notes.append((L, "shift %s／%s（既定 12／3 と違う）" % tuple(p["shift"])))
    return viol, notes


# ---------------------------------------------------------------- 表
def n(v, fmt="{:,}"):
    if v is None:
        return "—"
    if isinstance(v, float) and fmt == "{:,}":
        return "{:,}".format(int(v)) if v == int(v) else ("{:,.3f}".format(v).rstrip("0"))
    return fmt.format(v)


def md(rows, concl, preds, viol, notes):
    o = []
    o.append("# H3 形式別の快適上限 — 突き合わせの表（check_results_quant.py が生成）\n")
    o.append("runs/ の記録だけから組み直した。判定は計画 §5 の v3''（ここで独立に組み直し、run.json と突き合わせた）。\n")
    o.append("## 1. 点ごとの行\n")
    o.append("| ラベル | 形式 | 重み（名前／ファイル） | load_ok_dit | W×H×F | K | S（predict／実測・差） | status | 判定 | cap | 予約の山（denoise／decode） | 割り当ての山（denoise／decode） | 余裕 | retries | 1 ステップ秒（meta／log） | OOM の場所 | OOM の 4 つの数（要求／割り当て／予約済みの空き／許可）・free | 落ちたステップ | shift | 快適の回数 |")
    o.append("|" + "---|" * 20)
    for p in rows:
        lo = p["load_ok_dit"] or {}
        lo_s = "%s%s%s" % (json.dumps(lo.get("quant_layers_by_format"), ensure_ascii=False),
                           " fp8_mode=%s" % lo.get("fp8_mode") if lo.get("fp8_mode") else "",
                           "" if p["load_ok_dit の出所"] == "run.json" else "（%s）" % p["load_ok_dit の出所"].split("（")[0])
        name = p["重みの名前（metadata）"] or ("計画 %s" % p["計画の transformer"])
        fname = os.path.basename((p["transformer_path"] or "").replace("\\", "/")) or "—"
        oo = p["OOM"]
        place = oo4 = step = "—"
        if oo:
            place = "%s（%s・`%s`）" % (oo["場所"], oo["段"], oo["行"])
            d = oo["数"]
            if d:
                t = d["表記"]
                oo4 = "%s／%s／%s／%s・free %s" % (t["要求"], t["割り当て"], t["予約済みの空き"], t["許可"], t["free"])
            step = "%s／%s" % (oo["落ちたステップ"], oo["total_steps"])
        sh = "%g／%g%s" % (p["shift"][0], p["shift"][1], "" if p["shift の出所"] == "metadata" else "（%s）" % p["shift の出所"])
        o.append("| %s | %s | %s／`%s` | %s | %d×%d×%d | %d | %s／%s・%s | %s | **%s** | %s | %s／%s | %s／%s | %s | %s | %s／%s | %s | %s | %s | %s | %s |" % (
            p["label"], p["形式"], name, fname, lo_s, p["W"], p["H"], p["F"], p["K"],
            n(p["S_predict"]), n(p["S_実測"]), n(p["S_差"]), p["status"], p["判定"], n(p["cap_MiB"]),
            n(p["予約の山_denoise"]), n(p["予約の山_decode"]), n(p["割り当ての山_denoise"]), n(p["割り当ての山_decode"]),
            n(p["余裕_MiB"]), n(p["retries"]), n(p["1ステップ秒_meta"]), n(p["1ステップ秒_log"]),
            place, oo4, step, sh, n(p["快適の回数"])))
    o.append("")
    o.append("- 余裕 ＝ cap − ノイズ除去の予約の山（計画 §4.2 の int8 の余裕 513／235 と同じ取り方）。retries が 0 でない点では OOM までの距離そのものではない（§5）。")
    o.append("- 退避の窓（共有メモリの中央値 MB・wddm.csv から再計算。Δ は同じ形式の基準点との差）:")
    for p in rows:
        if p["役割"] == "点" and p["status"] == "completed":
            o.append("  - %s（基準 %s）: %s" % (p["label"], p["基準点"], "・".join(
                "%s %s（Δ %s・n=%d）" % (k, n(w["median_mb"]), n(w.get("delta_mb")), w["samples"]) for k, w in p["窓"].items())))
    o.append("\n## 2. 形式ごとの結論（計画 §10。Eros は §5 の 2 択）\n")
    o.append("int8 の現行: 1080 c=345／U=362（S 213,348／223,604）、1440 c=192／U=209（S 217,798／236,254）、線 219,516。\n")
    o.append("| 形式 | 解像度 | c | U | 帯（S: c の S〜U の S） | int8 との比較 | 注記 |")
    o.append("|---|---|---|---|---|---|---|")
    for fmt, c in concl.items():
        for res, v in c["解像度"].items():
            o.append("| %s | %s | %s | %s | %s〜%s | %s（int8 c=%d／U=%d） | %s |" % (
                fmt, res, n(v["c"]), n(v["U"]), n(v["帯_S"][0]), n(v["帯_S"][1]), v["向き"],
                v["int8 の現行"]["c"], v["int8 の現行"]["U"], "；".join(v["注記"]) or "—"))
    o.append("")
    for fmt, c in concl.items():
        ov = c["帯の重なり"]
        ovs = "—（両解像度の c・U がそろっていない）" if ov is None else (
            "重なる（共通 %s〜%s・219,516 は%s）" % (n(ov["共通の帯_S"][0]), n(ov["共通の帯_S"][1]), "中" if ov["int8 の線 219,516 が共通の帯の中"] else "外")
            if ov["重なる"] else "重ならない")
        o.append("- **%s**: 結論の形 %s。1080 と 1440 の帯の重なり: %s" % (fmt, c["結論の形"], ovs))
    o.append("\n## 3. 予測との照合（計画 §4.2: 形式の山 ＝ int8 の山 ＋ a×S ＋ b）\n")
    o.append("int8 の山は同じ形の int8 の完走点（今回の runs にあればそれ、無ければ 11-04・記録簿 §196）。a・b: fp8 −4 B/行・−2MiB、w4a8 +5,372・−920、Eros +5,372・0。\n")
    o.append("| 点 | 形式 | int8 の点 | S | int8 の山／余裕 | a×S＋b | 予測の山／余裕 | 予測 | 実測の判定 | 実測の山 | 差（実測−予測） | 実測の cap／余裕／retries | 照合 | 別の読みの余裕 |")
    o.append("|" + "---|" * 14)
    for q in preds:
        o.append("| %s | %s | %s | %s | %s／%s | %s | %s／%s | %s | %s | %s | %s | %s／%s／%s | %s | %s |" % (
            q["label"], q["形式"], q["int8 の点"], n(q["S"]), n(q["int8 の予約の山"]), n(q["int8 の余裕"]),
            n(q["a×S＋b_MiB"]), n(q["予測の予約の山"]), n(q["予測の余裕"]), q["予測"], q["実測の判定"],
            n(q["実測の予約の山"]), n(q["差_MiB（実測−予測）"]), n(q["実測の cap"]), n(q["実測の余裕"]), n(q["実測の retries"]), q["照合"],
            "・".join("%s: %s" % (k, n(v)) for k, v in (q["別の読みの余裕"] or {}).items()) or "—"))
    if not preds:
        o.append("| （int8 の同じ形の完走点がある点はまだ無い） |" + " |" * 13)
    o.append("\n## 4. 整合の検査\n")
    o.append("- 違反: %d 件" % len(viol))
    for L, m in viol:
        o.append("  - %s: %s" % (L, m))
    o.append("- 注記: %d 件" % len(notes))
    for L, m in notes:
        o.append("  - %s: %s" % (L, m))
    return "\n".join(o) + "\n"


def main():
    rows = [point_row(L) for L in labels()]
    order = {f: i for i, f in enumerate(FORMATS)}
    rows.sort(key=lambda p: (order.get(p["形式"], 9), {"煙": 0, "基準点": 1, "点": 2}[p["役割"]], p["H"], p["F"], p["label"]))
    add_counts(rows)
    concl = conclude(rows)
    preds = predictions(rows)
    viol, notes = checks(rows)
    os.makedirs(OUT, exist_ok=True)
    io.open(os.path.join(OUT, "results_table.md"), "w", encoding="utf-8").write(md(rows, concl, preds, viol, notes))
    json.dump({"points": rows, "conclusions": concl, "predictions": preds,
               "checks": {"violations": [{"label": L, "msg": m} for L, m in viol],
                          "notes": [{"label": L, "msg": m} for L, m in notes]}},
              io.open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("points:", len(rows), " violations:", len(viol), " notes:", len(notes))
    for L, m in viol:
        print("  VIOLATION", L, m)
    print("wrote analysis/results_table.md, analysis/results.json")
    return 1 if viol else 0


if __name__ == "__main__":
    sys.exit(main())
