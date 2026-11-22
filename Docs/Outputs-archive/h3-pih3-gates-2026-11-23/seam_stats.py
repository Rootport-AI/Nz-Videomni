r"""Brightness (YAVG) and saturation (SATAVG) of ffmpeg signalstats, averaged over the 4 frames before and the
4 frames after each chain seam (the spike's 223.9 method: frames s-4..s-1 vs s..s+3).

    .venv/Scripts/python.exe -X utf8 -B outputs/h3-pih3-gates-2026-11-23/seam_stats.py [ITEM ...]

Writes seam_stats.json (CPU only).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FFMPEG = REPO / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
SEAMS = {"LONG": [124, 226, 328, 430], "ACH": [124], "E3b4": [124], "E3b2": [124],
         "C3b_4step (spike)": ([124], REPO / "outputs" / "h3-pih3-spike-2026-11-22" / "runs" / "C3b_4step" / "output.mp4")}


def stats(mp4: Path) -> list[tuple[float, float]]:
    r = subprocess.run([str(FFMPEG), "-v", "error", "-i", str(mp4), "-map", "0:v:0", "-vf",
                        "signalstats,metadata=print:file=-", "-f", "null", "-"], capture_output=True, text=True)
    y = [float(m) for m in re.findall(r"lavfi\.signalstats\.YAVG=([\d.]+)", r.stdout)]
    s = [float(m) for m in re.findall(r"lavfi\.signalstats\.SATAVG=([\d.]+)", r.stdout)]
    return list(zip(y, s))


def main() -> int:
    want = set(sys.argv[1:])
    out = read = {}
    try:
        read = json.loads((HERE / "seam_stats.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        read = {}
    out = dict(read)
    for name, spec in SEAMS.items():
        if want and name not in want:
            continue
        seams, mp4 = (spec if isinstance(spec, tuple) else (spec, HERE / "runs" / name / "output.mp4"))
        if not mp4.is_file():
            continue
        st = stats(mp4)
        rows = []
        for s in seams:
            before, after = st[s - 4:s], st[s:s + 4]
            yb, ya = sum(v[0] for v in before) / 4, sum(v[0] for v in after) / 4
            sb, sa = sum(v[1] for v in before) / 4, sum(v[1] for v in after) / 4
            rows.append({"seam_frame": s, "yavg_before": round(yb, 2), "yavg_after": round(ya, 2),
                         "yavg_diff": round(ya - yb, 2), "satavg_before": round(sb, 2), "satavg_after": round(sa, 2),
                         "satavg_diff": round(sa - sb, 2)})
        out[name] = {"frames": len(st), "seams": rows}
        print(name, [(r["seam_frame"], r["yavg_diff"], r["satavg_diff"]) for r in rows])
    (HERE / "seam_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
