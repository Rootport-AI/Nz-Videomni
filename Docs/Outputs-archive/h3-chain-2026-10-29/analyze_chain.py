r"""§1-106 Chained (latent carry-over) analysis: CPU only (PyAV + numpy; no matplotlib in the venv -> CSV).

Run from the repo root after the conditions (run_c.ps1 does it last):

    set PYTHONPATH=%CD%
    set PYTHONUTF8=1
    .venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-chain-2026-10-29\analyze_chain.py

Input: the condition reports ``<archive>\c*.json`` written by c_chain.py (mp4 paths, geometry, the
in-memory ``seam_metrics``, latent paths). Output (plan §6.3), all in the archive:

  analysis_summary.csv       one row per condition x decode variant x seam: S_i, J_i, the jump ratio at
                             J_i (and S_i) from the mp4 and from memory, clip 0's largest interior ratio,
                             C7's ratio at the same seam, per-region luma / chroma and their change from
                             clip 0 (burn-in, H7), the audio click peak at the seam (H6)
  analysis_curve_<name>_<variant>.csv   d(t) from the mp4 next to d(t) from memory (seam_metrics)
  analysis_audio_<name>_<variant>.csv   RMS of 50 ms windows over J/24 s +- 0.5 s, per seam
  analysis_h2_latents.csv    C1 vs C3 (inject noise vs clean): max / mean |diff| of the new part of
                             each clip's latents (video [k:], audio [ka:])

Definitions (shared with c_chain.py's seam_metrics; the self-check there compares the two):
  Y      BT.709 luma of the 0..255 RGB values, 0.2126 R + 0.7152 G + 0.0722 B (in the mp4: the Y plane,
         limited range brought back to 0..255)
  chroma |(Cb, Cr)| with BT.709 Cb / Cr on the 0..255 RGB values (in the mp4: the 4:2:0 U / V planes)
  d(t)   mean over pixels of |Y_t - Y_(t-1)|, t = 1 .. F-1 (stored as a list whose index 0 is t = 1)
  ratio  d(J) / median of d over [J-12, J+12] without J-1, J, J+1 (plan §6.3 (3))
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # leave no __pycache__

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

FPS = 24
#: BT.709 rows Y, Cb, Cr (the same matrix as engine_h3/media_out.py's _BT709).
Y_COEF = (0.2126, 0.7152, 0.0722)
CB_COEF = (-0.1146, -0.3854, 0.5)
CR_COEF = (0.5, -0.4542, -0.0458)
RATIO_HALF = 12
RATIO_EXCLUDE = 1
AUDIO_HALF_SEC = 0.5
AUDIO_WIN_SEC = 0.05
CLICK_HALF_SEC = 0.01
BATCH = 8


# -- per-frame statistics ------------------------------------------------------------------


class FrameStats:
    """Streaming per-frame statistics: y_mean, y_sq_mean, chroma_mean per frame and d(t) between frames."""

    def __init__(self) -> None:
        self.y_mean: list[float] = []
        self.y_sq_mean: list[float] = []
        self.chroma_mean: list[float] = []
        self.d: list[float] = []
        self._prev: np.ndarray | None = None

    def add(self, y: np.ndarray, chroma_mean: float) -> None:
        y = y.astype(np.float64, copy=False)
        self.y_mean.append(float(y.mean()))
        self.y_sq_mean.append(float((y * y).mean()))
        self.chroma_mean.append(float(chroma_mean))
        if self._prev is not None:
            self.d.append(float(np.abs(y - self._prev).mean()))
        self._prev = y

    def as_dict(self) -> dict:
        return {"frames": len(self.y_mean), "y_mean": self.y_mean, "y_sq_mean": self.y_sq_mean,
                "chroma_mean": self.chroma_mean, "d": self.d}


def stats_from_rgb(frames: np.ndarray) -> dict:
    """uint8 RGB [F, H, W, 3] -> per-frame statistics (the reference for c_chain's torch seam_metrics)."""
    st = FrameStats()
    for start in range(0, int(frames.shape[0]), BATCH):
        rgb = frames[start:start + BATCH].astype(np.float64)
        y = rgb @ np.array(Y_COEF)
        cb = rgb @ np.array(CB_COEF)
        cr = rgb @ np.array(CR_COEF)
        chroma = np.sqrt(cb * cb + cr * cr)
        for j in range(rgb.shape[0]):
            st.add(y[j], float(chroma[j].mean()))
    return st.as_dict()


def read_mp4_video_stats(path: Path) -> dict:
    """The mp4's video stream -> per-frame statistics from the yuv420p planes (BT.709 limited range)."""
    import av

    st = FrameStats()
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        for frame in c.decode(stream):
            if frame.format.name != "yuv420p":
                frame = frame.reformat(format="yuv420p")
            planes = frame.to_ndarray()                       # [H * 3 / 2, W] uint8
            h, w = frame.height, frame.width
            y = (planes[:h].astype(np.float64) - 16.0) * (255.0 / 219.0)
            uv = planes[h:].reshape(2, h // 2, w // 2).astype(np.float64)
            uv = (uv - 128.0) * (255.0 / 224.0)
            chroma = np.sqrt(uv[0] * uv[0] + uv[1] * uv[1])
            st.add(y, float(chroma.mean()))
    return st.as_dict()


def read_mp4_audio(path: Path) -> tuple[np.ndarray, int]:
    """The mp4's audio stream -> float32 [channels, N] and the sample rate (empty when no audio)."""
    import av

    chunks: list[np.ndarray] = []
    rate = 0
    with av.open(str(path)) as c:
        if not c.streams.audio:
            return np.zeros((2, 0), dtype=np.float32), 0
        stream = c.streams.audio[0]
        rate = int(stream.codec_context.sample_rate)
        for frame in c.decode(stream):
            arr = frame.to_ndarray()
            if frame.format.is_planar:
                chunks.append(arr.astype(np.float32))
            else:
                chunks.append(arr.reshape(-1, len(frame.layout.channels)).T.astype(np.float32))
    if not chunks:
        return np.zeros((2, 0), dtype=np.float32), rate
    return np.concatenate(chunks, axis=1), rate


def mp4_frame_count(path: Path) -> int:
    import av

    with av.open(str(path)) as c:
        return sum(1 for _ in c.decode(c.streams.video[0]))


# -- seams ------------------------------------------------------------------------------------


def d_at(d: list, t: int) -> float | None:
    """d(t) for t = 1 .. F-1 (None outside)."""
    return float(d[t - 1]) if 1 <= t <= len(d) else None


def jump_ratio(d: list, t: int, half: int = RATIO_HALF, exclude: int = RATIO_EXCLUDE) -> float | None:
    """d(t) / median of d over [t-half, t+half] without |u - t| <= exclude (None when undefined)."""
    here = d_at(d, t)
    if here is None:
        return None
    ref = [d_at(d, u) for u in range(t - half, t + half + 1) if abs(u - t) > exclude]
    ref = [v for v in ref if v is not None]
    if not ref:
        return None
    med = float(np.median(ref))
    return None if med <= 0 else here / med


def interior_max_ratio(d: list, lo: int, hi: int) -> dict:
    """The largest jump ratio for t in [lo, hi] whose whole window lies in [1, hi + half] (clip 0's interior)."""
    best, at = None, None
    for t in range(max(lo, 1 + RATIO_HALF), hi + 1):
        r = jump_ratio(d, t)
        if r is not None and (best is None or r > best):
            best, at = r, t
    return {"max": best, "at": at}


def region_stats(stats: dict, a: int, b: int) -> dict:
    """Pooled luma mean / std and mean chroma of frames [a, b)."""
    ym = np.array(stats["y_mean"][a:b])
    ysq = np.array(stats["y_sq_mean"][a:b])
    ch = np.array(stats["chroma_mean"][a:b])
    if ym.size == 0:
        return {"frames": [a, b], "y_mean": None, "y_std": None, "chroma_mean": None}
    mean = float(ym.mean())
    var = max(0.0, float(ysq.mean()) - mean * mean)
    return {"frames": [a, b], "y_mean": mean, "y_std": math.sqrt(var), "chroma_mean": float(ch.mean())}


def summarize(stats: dict, geom: dict, *, cut: bool = False) -> dict:
    """Seam numbers for one decoded clip chain (plan §6.3 (1)(3)(4)).

    ``geom``: c_chain.geometry's dict. The output's frame f is global frame f (every variant has the
    same F_total frames, the per-clip variant included). ``cut``: whether the frames came out of the
    per-clip decode (only J is a seam there); recorded only.
    """
    d = stats["d"]
    n = len(stats["y_mean"])
    seams = []
    for i, (s, j) in enumerate(zip(geom["S"], geom["J"]), start=1):
        seams.append({"clip": i, "S": s, "J": j, "d_J": d_at(d, j), "ratio_J": jump_ratio(d, j),
                      "d_S": d_at(d, s), "ratio_S": jump_ratio(d, s) if s != j else None})
    bounds = [0] + list(geom["J"]) + [n]
    regions = [region_stats(stats, bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    base = regions[0]
    for r in regions:
        for key in ("y_mean", "y_std", "chroma_mean"):
            r[f"delta_{key}"] = (None if r[key] is None or base[key] is None else r[key] - base[key])
    first_new = geom["J"][0] if geom["J"] else n
    return {"frames": n, "cut": bool(cut), "seams": seams, "regions": regions,
            "clip0_interior": interior_max_ratio(d, 1, first_new - 1 - RATIO_HALF)}


def audio_seam(wave: np.ndarray, rate: int, t_sec: float) -> dict:
    """RMS of 50 ms windows over [t - 0.5, t + 0.5] s (channel mean) and the adjacent-sample jump (click)."""
    if wave.size == 0 or rate <= 0:
        return {"rms": [], "click_peak": None, "click_p99": None, "click_ratio": None}
    mono = wave.mean(axis=0)
    win = int(round(AUDIO_WIN_SEC * rate))
    lo = int(round((t_sec - AUDIO_HALF_SEC) * rate))
    n_win = int(round(2 * AUDIO_HALF_SEC / AUDIO_WIN_SEC))
    rms = []
    for w in range(n_win):
        a, b = lo + w * win, lo + (w + 1) * win
        seg = mono[max(a, 0):max(min(b, mono.size), 0)]
        rms.append(float(np.sqrt(np.mean(seg * seg))) if seg.size else None)
    diff = np.abs(np.diff(wave, axis=1)).max(axis=0)      # [N-1], the larger of the two channels
    c = int(round(t_sec * rate))
    near = diff[max(c - int(CLICK_HALF_SEC * rate), 0):max(min(c + int(CLICK_HALF_SEC * rate), diff.size), 0)]
    wide = diff[max(lo, 0):max(min(lo + n_win * win, diff.size), 0)]
    peak = float(near.max()) if near.size else None
    p99 = float(np.percentile(wide, 99)) if wide.size else None
    return {"rms": rms, "click_peak": peak, "click_p99": p99,
            "click_ratio": None if not peak or not p99 else peak / p99}


# -- H2: latents -------------------------------------------------------------------------------


def latent_new_part_diff(rep_a: dict, rep_b: dict) -> list[dict]:
    """Per clip: max / mean |a - b| of the new part (video [k:], audio [ka:]) of the saved latents."""
    import torch

    geom = rep_a["geometry"]
    out = []
    for i, (ca, cb) in enumerate(zip(rep_a.get("clips", []), rep_b.get("clips", []))):
        pa, pb = ca.get("latents_path"), cb.get("latents_path")
        row = {"clip": i, "a": pa, "b": pb}
        if not pa or not pb or not _resolve(pa).is_file() or not _resolve(pb).is_file():
            row["missing"] = True
            out.append(row)
            continue
        la, lb = torch.load(_resolve(pa)), torch.load(_resolve(pb))
        k = 0 if i == 0 else int(geom["k"])
        ka = 0 if i == 0 else int(geom["ka"][i - 1])
        for key, cut in (("video", k), ("audio", ka)):
            x, y = la[key][:, :, cut:].float(), lb[key][:, :, cut:].float()
            diff = (x - y).abs()
            row[f"{key}_max_abs"] = float(diff.max())
            row[f"{key}_mean_abs"] = float(diff.mean())
            row[f"{key}_ref_abs_mean"] = float(x.abs().mean())
            row[f"{key}_equal"] = bool(torch.equal(x, y))
        out.append(row)
    return out


# -- driver ----------------------------------------------------------------------------------------


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([_fmt(v) for v in r])


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def analyze(reports: list[Path], out_dir: Path, c1: str, c3: str, c7: str, report: Path | None = None) -> int:
    written: list[str] = []
    loaded = {}
    for path in reports:
        try:
            rep = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:  # noqa: BLE001
            print(f"skip {path.name}: {exc}")
            continue
        if not isinstance(rep, dict) or "geometry" not in rep or not rep.get("name"):
            print(f"skip {path.name}: not a c_chain report")
            continue
        loaded[rep["name"]] = rep
    print(f"reports: {sorted(loaded)}")
    summary_rows: list[list] = []
    per_seam: dict = {}
    for name, rep in sorted(loaded.items()):
        geom = rep["geometry"]
        for variant, rec in sorted((rep.get("decode") or {}).items()):
            if variant == "fingerprint" or not isinstance(rec, dict) or not rec.get("path"):
                continue
            mp4 = _resolve(rec["path"])
            if not mp4.is_file():
                print(f"{name}/{variant}: missing {mp4}")
                continue
            print(f"{name}/{variant}: reading {mp4}")
            stats = read_mp4_video_stats(mp4)
            summ = summarize(stats, geom, cut=(variant == "perclip"))
            mem = (rec.get("seam_metrics") or {})
            mem_d = (mem.get("stats") or {}).get("d") or []
            written.append(f"analysis_curve_{name}_{variant}.csv")
            _write_csv(out_dir / f"analysis_curve_{name}_{variant}.csv", ["t", "d_mp4", "d_memory"],
                       [[t, d_at(stats["d"], t), d_at(mem_d, t)] for t in range(1, stats["frames"])])
            wave, rate = read_mp4_audio(mp4)
            audio_rows = []
            mem_seams = {s["clip"]: s for s in (mem.get("summary") or {}).get("seams", [])}
            for seam in summ["seams"]:
                a = audio_seam(wave, rate, seam["J"] / FPS)
                seam["audio"] = a
                for w, v in enumerate(a["rms"]):
                    audio_rows.append([seam["clip"], seam["J"], round(-AUDIO_HALF_SEC + w * AUDIO_WIN_SEC, 3), v])
                reg = summ["regions"][seam["clip"]]
                ms = mem_seams.get(seam["clip"], {})
                per_seam[(name, variant, seam["clip"])] = seam["ratio_J"]
                summary_rows.append([
                    name, variant, rep.get("condition", {}).get("strength"), rep.get("condition", {}).get("inject"),
                    geom["O"], seam["clip"], seam["S"], seam["J"], seam["d_J"], seam["ratio_J"], ms.get("ratio_J"),
                    seam["ratio_S"], summ["clip0_interior"]["max"], summ["clip0_interior"]["at"], None,
                    reg["y_mean"], reg["y_std"], reg["chroma_mean"],
                    reg["delta_y_mean"], reg["delta_y_std"], reg["delta_chroma_mean"],
                    a["click_peak"], a["click_p99"], a["click_ratio"],
                    stats["frames"], wave.shape[1] if wave.size else 0,
                ])
            written.append(f"analysis_audio_{name}_{variant}.csv")
            _write_csv(out_dir / f"analysis_audio_{name}_{variant}.csv", ["clip", "J", "window_start_sec", "rms"],
                       audio_rows)
    # C7's ratio at the same seam (its only variant is perclip).
    for row in summary_rows:
        row[14] = per_seam.get((c7, "perclip", row[5]))
    header = ["name", "variant", "strength", "inject", "O", "seam_clip", "S", "J", "d_J_mp4", "ratio_J_mp4",
              "ratio_J_memory", "ratio_S_mp4", "clip0_interior_max_ratio", "clip0_interior_max_at",
              "c7_ratio_J", "region_y_mean", "region_y_std", "region_chroma_mean", "delta_y_mean",
              "delta_y_std", "delta_chroma_mean", "click_peak", "click_p99", "click_ratio",
              "frames_mp4", "audio_samples_mp4"]
    _write_csv(out_dir / "analysis_summary.csv", header, summary_rows)
    written.append("analysis_summary.csv")
    print(f"wrote {out_dir / 'analysis_summary.csv'} ({len(summary_rows)} rows)")
    if c1 in loaded and c3 in loaded:
        rows = latent_new_part_diff(loaded[c1], loaded[c3])
        keys = ["clip", "video_max_abs", "video_mean_abs", "video_ref_abs_mean", "video_equal",
                "audio_max_abs", "audio_mean_abs", "audio_ref_abs_mean", "audio_equal", "missing"]
        _write_csv(out_dir / "analysis_h2_latents.csv", keys, [[r.get(k) for k in keys] for r in rows])
        print(f"wrote {out_dir / 'analysis_h2_latents.csv'}")
        written.append("analysis_h2_latents.csv")
    else:
        print(f"H2: {c1} / {c3} not both present; skipped")
    if report is not None:
        report.write_text(json.dumps({"name": "analysis", "ok": bool(summary_rows), "reports": sorted(loaded),
                                      "summary_rows": len(summary_rows), "files": written}, indent=2,
                                     ensure_ascii=False), encoding="utf-8")
    return 0 if summary_rows else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", nargs="*", help="report JSONs (default: <archive>\\c*.json)")
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--c1", default="c1_base")
    ap.add_argument("--c3", default="c3_inject_clean")
    ap.add_argument("--c7", default="c7_control")
    ap.add_argument("--report", help="a small JSON of what was written (run_common.ps1 passes one)")
    args = ap.parse_args()
    reports = [Path(p) for p in args.reports] if args.reports else sorted(HERE.glob("c*.json"))
    return analyze(reports, Path(args.out_dir), args.c1, args.c3, args.c7,
                   Path(args.report) if args.report else None)


if __name__ == "__main__":
    sys.exit(main())
