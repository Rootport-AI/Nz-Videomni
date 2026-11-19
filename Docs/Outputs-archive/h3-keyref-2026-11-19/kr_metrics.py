r"""§1-115 の集計（計画 glistening-splashing-hoare.md §5、段階 A・A2・B・C）。GPU を使わない（ffmpeg・numpy・PIL、
キーフレームの画布だけ製品の関数を CPU で呼ぶ）。

Per run (the mp4 and its kr_single report):
  * the key frames come from the report (condition.keyframes: frame and path; any number). Each is put on
    the video's canvas by the product's own engine_h3.image_cond.load_canvas_image (EXIF, RGB, cover +
    centre crop with LANCZOS). A run without a key frame (or a --run without a report) uses --key at frame 0.
  * psnr_first: frame 0 vs the frame-0 key frame (or --key). A run whose key frames miss frame 0 (stage B)
    has no psnr_first / psnr_vs_key (None, "-"): read its per-key-frame numbers.
  * frames 0..24: PSNR of each frame vs that key, and of each adjacent pair (t-1, t).
  * per key frame n: the PSNR at n, and for n-12..n+12 the PSNR vs that key frame and the adjacent PSNR.
  * freeze / jump marks: the adjacent PSNR over the median adjacent PSNR of the same run's pairs with both
    frames >= 25 (FREEZE_RATIO / JUMP_RATIO below); on t = 1..24 (``marks``) and near each key frame
    (``keyframe_marks``, "jump@36(kf34)").
  * from the report: the frame count check, the sequence rows, the seconds per stage, the VRAM peaks, the L
    self-check.
PSNR as Docs/Outputs-archive/h3-gate2/run_2b_g2.py:179-197: the mp4 decoded by ffmpeg to rgb24, MSE over
every pixel and channel, peak 255 (identical frames: PSNR_CAP instead of infinity).
--runs / --run need --out-prefix (a partial list never takes a stage's name); an existing output of the same
name is moved aside with a time stamp first. Writes <out-dir>\<prefix>.json, <prefix>.md and the contact sheet (stage A: kr_metrics.json / .md and
kr_contact.png, as before; others <prefix>_contact.png): the chosen frames side by side, one run per row, the
run's name at the left; the first row holds the key frames' canvases in their frames' columns.

    .venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-keyref-2026-11-19\kr_metrics.py --stage A2
    ... --runs a1_010_conflict,a2_010_consistent --out-prefix kr_metrics_x --contact-frames 0,17,123
    ... --run NAME=MP4 [--run ...]   (other mp4s, e.g. a try on older outputs; no report needed)
    ... --first-psnr --mp4 <mp4> --report <json>   (one run's frame 0 vs its frame-0 key frame only;
                                                     prints FIRST_PSNR=<dB>, exit 0; 1 on an error)
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
OUT_ROOT = ROOT / "outputs/h3-keyref-2026-11-19"
KEY = ROOT / "uploads/key-ref-development/key-001.jpg"
#: The runs of each stage in their order, the contact sheet's frames and the output name.
STAGES = {
    "A": {"runs": ("r01_010_harness", "r02_010_product", "r03_ctrlF", "r04_ctrlR", "r05_110", "r06_001",
                   "r07_011", "r08_000", "r09_100", "r10_101", "r11_111"),
          "frames": (0, 1, 4, 8, 17, 40, 80, 123), "prefix": "kr_metrics"},
    "A2": {"runs": ("a1_010_conflict", "a2_010_consistent", "a3_ctrlF_conflict", "a4_ctrlR_conflict",
                    "a5_110_conflict", "a6_001_conflict", "a7_011_conflict"),
           "frames": (0, 1, 4, 8, 17, 40, 80, 123), "prefix": "kr_metrics_A2"},
    # A3 also lists A2's a1 / a2 / a5 first, to read side by side (their reports and mp4s are only read).
    "A3": {"runs": ("a1_010_conflict", "a2_010_consistent", "a5_110_conflict", "a8_110_decl", "a9_keyasref_decl",
                    "a10_keyasref_nodecl"),
           "frames": (0, 1, 4, 8, 17, 40, 80, 123), "prefix": "kr_metrics_A3"},
    "B": {"runs": ("b1_010_f17_harness", "b1_010_f17_product", "b2_001_f17", "b3_011_f17"),
          "frames": (0, 8, 15, 16, 17, 18, 19, 24, 40, 80, 123), "prefix": "kr_metrics_B"},
    "C": {"runs": ("c1_010_audio_harness", "c1_010_audio_product", "c2_001_audio", "c3_011_audio",
                   "c4_ctrl_noaudio"),
          "frames": (0, 17, 34, 42, 51, 68, 85, 102, 119, 123), "prefix": "kr_metrics_C"},
}
EXPECTED_FRAMES = 124          # plan §4, when there is no report to read the frame count from
HEAD = 25                      # frames 0..24 (plan §5)
TAIL_FROM = 25                 # the reference: adjacent pairs with both frames >= 25
WINDOW = 12                    # frames on each side of a key frame
PSNR_CAP = 99.0
#: A pair this much above the run's own later median looks still (1.3 x ~30 dB = ~39 dB: about 8x less MSE).
FREEZE_RATIO = 1.3
#: A pair this much below it looks like a cut (0.7 x ~30 dB = ~21 dB: about 8x more MSE).
JUMP_RATIO = 0.7
THUMB_W = 320
LABEL_W = 170


def find_tool(name: str) -> str:
    local = ROOT / "tools" / "ffmpeg" / "bin" / f"{name}.exe"
    if local.exists():
        return str(local)
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name} not found (tools\\ffmpeg\\bin or PATH)")
    return found


def probe(mp4: Path) -> tuple[int, int]:
    proc = subprocess.run([find_tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=width,height", "-of", "json", str(mp4)],
                          capture_output=True, encoding="utf-8", errors="replace", check=True)
    s = json.loads(proc.stdout)["streams"][0]
    return int(s["width"]), int(s["height"])


def frames_rgb(mp4: Path, width: int, height: int, limit: int | None = None):
    """The mp4's frames as uint8 [H, W, 3], one at a time (ffmpeg rawvideo rgb24); ``limit`` = the first n."""
    size = width * height * 3
    cmd = [find_tool("ffmpeg"), "-v", "error", "-i", str(mp4)]
    if limit is not None:
        cmd += ["-frames:v", str(int(limit))]
    proc = subprocess.Popen(cmd + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", errors="replace")
        proc.stderr.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed on {mp4}: {err.strip()}")


def key_canvas(path: Path, width: int, height: int) -> np.ndarray:
    """The product's own canvas (engine_h3/image_cond.load_canvas_image), on the CPU."""
    from engine_h3.image_cond import load_canvas_image

    canvas, _size = load_canvas_image(str(path), int(width), int(height))
    return np.asarray(canvas.convert("RGB"), dtype=np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return PSNR_CAP if mse <= 0 else min(PSNR_CAP, 10 * math.log10(255.0 ** 2 / mse))


def _r(x, n=2):
    return None if x is None else round(float(x), n)


def _abs(p) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def run_keyframes(rep: dict, fallback: Path) -> tuple[list, bool]:
    """``([{"frame", "path"}], from_report)``: the report's key frames, or --key at frame 0."""
    kfs = [{"frame": int(k["frame"]), "path": str(_abs(k["path"]))} for k in rep.get("keyframes") or []]
    if kfs:
        return sorted(kfs, key=lambda k: k["frame"]), True
    return [{"frame": 0, "path": str(fallback)}], False


def _mark(adj_t: float, med: float | None) -> tuple[float | None, str]:
    ratio = None if not med else adj_t / med
    mark = "" if ratio is None else ("freeze" if ratio >= FREEZE_RATIO else "jump" if ratio <= JUMP_RATIO else "")
    return ratio, mark


def measure(mp4: Path, keyframes: list, contact_frames) -> dict:
    """Every number of one run. ``keyframes``: ``[{"frame", "path"}]``. The head numbers (psnr_first,
    psnr_vs_key) only when a key frame is at frame 0; otherwise None / [] (read the per-key-frame numbers)."""
    w, h = probe(mp4)
    canv = {k["frame"]: key_canvas(Path(k["path"]), w, h) for k in keyframes}
    head_key = canv.get(0)
    vs_key, adj, thumbs = [], [], {}
    near: dict = {n: {} for n in canv}            # n -> {t: PSNR(frame t, key n)}
    prev = None
    count = 0
    for t, fr in enumerate(frames_rgb(mp4, w, h)):
        if t < HEAD and head_key is not None:
            vs_key.append(psnr(fr, head_key))
        for n, c in canv.items():
            if abs(t - n) <= WINDOW:
                near[n][t] = psnr(fr, c)
        if prev is not None:
            adj.append(psnr(prev, fr))          # adj[t-1] = PSNR(frame t-1, frame t)
        if t in contact_frames:
            thumbs[t] = Image.fromarray(fr.copy())
        prev = fr
        count += 1
    tail = [adj[t - 1] for t in range(TAIL_FROM + 1, count)]
    med = statistics.median(tail) if tail else None
    head = []
    for t in range(1, min(HEAD, count)):
        ratio, mark = _mark(adj[t - 1], med)
        head.append({"t": t, "psnr_adj": _r(adj[t - 1]), "ratio": _r(ratio, 3), "mark": mark})
    per_kf, kf_marks = [], []
    for k in keyframes:
        n = k["frame"]
        rows = []
        for t in range(max(0, n - WINDOW), min(count, n + WINDOW + 1)):
            ratio, mark = (None, "") if t == 0 else _mark(adj[t - 1], med)
            rows.append({"t": t, "psnr_vs_key": _r(near[n].get(t)),
                         "psnr_adj": None if t == 0 else _r(adj[t - 1]), "ratio": _r(ratio, 3), "mark": mark})
            if mark:
                kf_marks.append(f"{mark}@{t}(kf{n})")
        per_kf.append({"frame": n, "path": k["path"], "psnr": _r(near[n].get(n)), "window": rows})
    return {
        "size": [w, h], "frames": count, "psnr_first": _r(vs_key[0]) if vs_key else None,
        "psnr_vs_key": [_r(x) for x in vs_key], "adjacent_head": head,
        "adjacent_tail_median": _r(med), "marks": [f"{x['mark']}@{x['t']}" for x in head if x["mark"]],
        "keyframes": per_kf, "keyframe_marks": kf_marks, "thumbs": thumbs,
        # Every frame (a cut anywhere): adjacent_all[t-1] = PSNR(t-1, t); the same jump rule as the marks.
        "adjacent_all": [_r(x) for x in adj],
        "jumps_all": [t for t in range(1, count) if _mark(adj[t - 1], med)[1] == "jump"],
    }


def first_psnr(mp4: Path, report: Path) -> float:
    """Frame 0 vs the report's frame-0 key frame on the product's canvas (the A2 reproduction gate)."""
    j = json.loads(report.read_text(encoding="utf-8-sig"))
    kfs = [k for k in (j.get("condition") or {}).get("keyframes") or [] if int(k["frame"]) == 0]
    if not kfs:
        raise ValueError(f"no key frame at frame 0 in {report}")
    w, h = probe(mp4)
    key = key_canvas(_abs(kfs[0]["path"]), w, h)
    for fr in frames_rgb(mp4, w, h, limit=1):
        return psnr(fr, key)
    raise RuntimeError(f"no frame decoded from {mp4}")


def from_report(path: Path | None) -> dict:
    """The kr_single report's numbers (None when there is no report)."""
    if path is None or not path.is_file():
        return {"report": None}
    j = json.loads(path.read_text(encoding="utf-8-sig"))
    run = (j.get("runs") or [{}])[0]
    s = run.get("summary") or {}
    res = run.get("result") or {}
    cond = j.get("condition") or {}
    return {"report": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "ok": j.get("ok"), "mode": run.get("mode"), "paths": run.get("paths"),
            "num_frames": cond.get("num_frames"), "keyframes": cond.get("keyframes"),
            # The gate's second run (the product's): it reuses what the first run reserved.
            "gate_second": bool(j.get("gate_of")) and run.get("mode") == "product",
            "sequence": s.get("sequence"), "seconds": s.get("seconds"), "peak_vram_mb": s.get("peak_vram_mb"),
            "peak_vram_mb_job": s.get("peak_vram_mb_job"), "text_tokens": s.get("text_tokens"),
            "mask": s.get("mask"), "L_check": res.get("L_check") if isinstance(res, dict) else None,
            "mp4": run.get("mp4")}


def contact_sheet(rows: list, frames, out: Path) -> None:
    """One row per run, ``frames`` across, the name at the left; the first row holds the key frames'
    canvases in the columns of their frames (the first given path for a frame)."""
    sizes = [r["m"]["size"] for r in rows if r.get("m")]
    if not sizes:
        return
    w0, h0 = sizes[0]
    th = round(THUMB_W * h0 / w0)
    keys: dict = {}
    for r in rows:
        for k in r.get("kf") or []:
            keys.setdefault(k["frame"], k["path"])
    font = ImageFont.load_default()
    sheet = Image.new("RGB", (LABEL_W + THUMB_W * len(frames), th * (len(rows) + 1) + 20), "white")
    draw = ImageDraw.Draw(sheet)
    for j, t in enumerate(frames):
        draw.text((LABEL_W + j * THUMB_W + 4, 4), f"frame {t}", fill="black", font=font)
    draw.text((6, 20 + th // 2), "key frames", fill="black", font=font)
    for j, t in enumerate(frames):
        if t in keys:
            key = Image.fromarray(key_canvas(Path(keys[t]), w0, h0)).resize((THUMB_W, th), Image.LANCZOS)
            sheet.paste(key, (LABEL_W + j * THUMB_W, 20))
    for i, r in enumerate(rows, start=1):
        y = 20 + i * th
        draw.text((6, y + th // 2), r["name"], fill="black", font=font)
        for j, t in enumerate(frames):
            im = (r.get("m") or {}).get("thumbs", {}).get(t)
            if im is not None:
                sheet.paste(im.resize((THUMB_W, th), Image.LANCZOS), (LABEL_W + j * THUMB_W, y))
    sheet.save(out)


def markdown(rows: list) -> str:
    lines = [
        "| ラン | フレーム数 | 先頭 PSNR | key との PSNR (1・4・8・17・24) | 隣の PSNR の最小 (1〜24) |"
        " 後ろの中央値 (25〜) | 印 | キーフレームの PSNR | キーフレーム付近の印 | 全フレームの jump |"
        " 行数 text／cond／ref_video／ref_audio／video | 秒 encode／denoise／decode | VRAM の山 MB | L の自己検査 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        m, rep = r.get("m") or {}, r.get("rep") or {}
        if r.get("error"):
            lines.append(f"| {r['name']} | {r['error']} |||||||||||||")
            continue
        frames = f"{m.get('frames')}" + ("" if r.get("frames_ok") else f"（NG: {r.get('frames_expected')} のはず）")
        note = "※" if rep.get("gate_second") else ""
        vk = m.get("psnr_vs_key") or []
        pick = "・".join(str(vk[t]) if t < len(vk) else "-" for t in (1, 4, 8, 17, 24)) if vk else "-"
        head = [x["psnr_adj"] for x in m.get("adjacent_head") or [] if x["psnr_adj"] is not None]
        kfp = "・".join(f"{k['frame']}: {k['psnr']}" for k in m.get("keyframes") or [])
        if not r.get("kf_from_report"):
            kfp += "（--key）"
        seq = rep.get("sequence") or {}
        sec = rep.get("seconds") or {}
        lc = rep.get("L_check")
        lcs = "-" if not lc else (f"{'ok' if lc.get('ok') else 'NG'} (0 の行 {lc.get('mask_zero_rows')}、"
                                  f"一致 {[x.get('equal') for x in lc.get('latents') or []]})")
        lines.append(
            f"| {r['name']} | {frames} | {m.get('psnr_first') if m.get('psnr_first') is not None else '-'} | {pick} | {min(head) if head else '-'} | "
            f"{m.get('adjacent_tail_median')} | {' '.join(m.get('marks') or []) or '-'} | {kfp} | "
            f"{' '.join(m.get('keyframe_marks') or []) or '-'} | "
            f"{' '.join(f'jump@{t}' for t in m.get('jumps_all') or []) or '-'} | "
            f"{seq.get('text', '-')}／{seq.get('cond', '-')}／{seq.get('ref_video', '-')}／"
            f"{seq.get('ref_audio', '-')}／{seq.get('video', '-')} | "
            f"{sec.get('encode', '-')}／{sec.get('denoise', '-')}／{sec.get('decode', '-')}{note} | "
            f"{rep.get('peak_vram_mb_job', '-')}{note} | {lcs} |")
    lines.append("")
    if any((r.get("rep") or {}).get("gate_second") for r in rows):
        lines.append("※ 門の 2 本目（1 本目が確保した領域を使い回すので、ほかのランと比べられない）。")
        lines.append("")
    lines.append(f"印: 隣り合うフレームの PSNR ÷ 同じランのフレーム 25 以降の中央値が {FREEZE_RATIO} 以上で freeze、"
                 f"{JUMP_RATIO} 以下で jump（「印」は 1〜24、「キーフレーム付近の印」は各キーフレームの前後 "
                 f"{WINDOW} フレーム。kfN はキーフレームのフレーム番号）。PSNR は mp4 を ffmpeg で rgb24 にした"
                 f"全画素・全チャンネル、ピーク 255（同一フレームは {PSNR_CAP}）。キーフレームは製品と同じ前処理の画布。")
    return "\n".join(lines) + "\n"


def move_aside(path: Path, stamp: str) -> None:
    """An existing output -> <stem>.<yyyyMMdd-HHmmss><suffix> beside it (run_common.ps1's Move-Aside)."""
    if path.exists():
        path.rename(path.with_name(f"{path.stem}.{stamp}{path.suffix}"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=tuple(STAGES), default="A")
    ap.add_argument("--runs", default=None, help="comma-separated run names (replaces the stage's list)")
    ap.add_argument("--out-prefix", default=None, help="output file name (default: the stage's)")
    ap.add_argument("--contact-frames", default=None, help="comma-separated frames (default: the stage's)")
    ap.add_argument("--reports-dir", default=str(HERE))
    ap.add_argument("--mp4-dir", default=str(OUT_ROOT))
    ap.add_argument("--key", default=str(KEY))
    ap.add_argument("--out-dir", default=str(OUT_ROOT))
    ap.add_argument("--run", action="append", default=[], help="NAME=MP4 (repeatable; replaces the run list)")
    ap.add_argument("--first-psnr", action="store_true", help="only frame 0's PSNR of --mp4 (needs --report)")
    ap.add_argument("--mp4", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    if args.first_psnr:
        try:
            if not args.mp4 or not args.report:
                raise ValueError("--first-psnr needs --mp4 and --report")
            value = first_psnr(_abs(args.mp4), _abs(args.report))
        except Exception as exc:  # noqa: BLE001 -- the caller reads the exit code
            print(f"[kr_metrics] first-psnr failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return 1
        print(f"FIRST_PSNR={value:.2f}", flush=True)
        return 0

    if (args.runs or args.run) and not args.out_prefix:
        # A partial list never takes a stage's own file name (its table would be replaced by part of it).
        print("[kr_metrics] --runs / --run need --out-prefix (the stage's own name is for its full list)",
              file=sys.stderr, flush=True)
        return 2
    stage = STAGES[args.stage]
    prefix = args.out_prefix or stage["prefix"]
    frames = (tuple(int(x) for x in args.contact_frames.split(",") if x.strip()) if args.contact_frames
              else stage["frames"])
    runs = tuple(x.strip() for x in args.runs.split(",") if x.strip()) if args.runs else stage["runs"]
    key_path = Path(args.key)
    if args.run:
        items = []
        for raw in args.run:
            name, sep, mp4 = raw.partition("=")
            if not sep or not name or not mp4:
                raise SystemExit(f"--run must be NAME=MP4 ({raw!r})")
            items.append((name, Path(mp4), None))
    else:
        items = [(n, Path(args.mp4_dir) / f"{n}.mp4", Path(args.reports_dir) / f"{n}.json") for n in runs]
    rows = []
    for name, mp4, rep in items:
        row = {"name": name, "mp4": str(mp4), "rep": from_report(rep)}
        row["kf"], row["kf_from_report"] = run_keyframes(row["rep"], key_path)
        if not mp4.is_file():
            row["error"] = "mp4 が無い"
        else:
            try:
                row["m"] = measure(mp4, row["kf"], frames)
                expected = row["rep"].get("num_frames") or EXPECTED_FRAMES
                row["frames_expected"] = int(expected)
                row["frames_ok"] = row["m"]["frames"] == int(expected)
            except Exception as exc:  # noqa: BLE001 -- one bad run never stops the others
                row["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[kr_metrics] {name}: " + (row.get("error") or f"first {row['m']['psnr_first']} dB, "
                                          f"key frames {[(k['frame'], k['psnr']) for k in row['m']['keyframes']]}, "
                                          f"marks {row['m']['marks'] + row['m']['keyframe_marks']}"), flush=True)
        rows.append(row)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    contact = out / ("kr_contact.png" if prefix == "kr_metrics" else f"{prefix}_contact.png")
    data = {"stage": args.stage, "runs_selected": list(runs), "key": str(key_path), "contact_frames": list(frames),
            "constants": {"head": HEAD, "tail_from": TAIL_FROM, "window": WINDOW, "freeze_ratio": FREEZE_RATIO,
                          "jump_ratio": JUMP_RATIO, "psnr_cap": PSNR_CAP},
            "runs": [{k: v for k, v in r.items() if k != "m"}
                     | {"metrics": {k: v for k, v in (r.get("m") or {}).items() if k != "thumbs"}} for r in rows]}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for p in (out / f"{prefix}.json", out / f"{prefix}.md", contact):
        move_aside(p, stamp)
    (out / f"{prefix}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / f"{prefix}.md").write_text(markdown(rows), encoding="utf-8")
    contact_sheet(rows, frames, contact)
    print(f"[kr_metrics] wrote {out / (prefix + '.json')}, {prefix}.md, {contact.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
