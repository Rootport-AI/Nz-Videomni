r"""Contact sheets for the owner's eyes: 5 frames (first, 1/4, 1/2, 3/4, last) side by side, each 640 wide.

    .venv/Scripts/python.exe -X utf8 -B outputs/h3-pih3-gates-2026-11-23/make_sheets.py

Writes sheets/<name>.png for every finished mp4 of the listed items (CPU only: ffmpeg + Pillow).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FFMPEG = REPO / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE = REPO / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"
TARGETS = {"P1": "output.mp4", "P2": "output.mp4", "P3": "output.mp4", "P4": "output.mp4",
           "L1": "output.mp4", "L2": "output.mp4", "L3": "output.mp4", "I2V1": "output.mp4", "I2V2": "output.mp4",
           "REF": "output.mp4", "NEG": "output.mp4", "V16OFF": "output.mp4", "Q1": "output.mp4",
           "A2V_pi": "A2V/A2V_pi.mp4", "A2V_turbo": "A2V/A2V_turbo.mp4",
           "X1": "output.mp4", "X2": "output.mp4", "X3": "output.mp4", "REFc": "output.mp4",
           "REFt": "output.mp4", "L3b": "output.mp4", "L4": "output.mp4", "L5": "output.mp4",
           "C0": "output.mp4", "M3": "output.mp4", "E3b4": "output.mp4", "E3b2": "output.mp4", "ER1": "output.mp4",
           "EI1": "output.mp4", "EO1": "output.mp4", "ACH": "output.mp4", "AV2V": "output.mp4", "ART": "output.mp4",
           "AIN": "output.mp4", "AOUT": "output.mp4", "AAUD": "output.mp4", "R05_pi": "R05/R05_pi.mp4",
           "R05_turbo": "R05/R05_turbo.mp4", "LONG": "output.mp4",
           "G-R1_2026-10-28": str(REPO / "outputs" / "h3-ref2va-2026-10-28" / "G-R1.mp4"),
           "G3_runB_s0.5": str(REPO / "outputs" / "h3_a2v_g3" / "runB_s0.5" / "output.mp4")}
ONLY = set(sys.argv[1:])


def frames(mp4: Path) -> int:
    r = subprocess.run([str(FFPROBE), "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                        "stream=nb_read_frames", "-of", "json", str(mp4)], capture_output=True, text=True)
    return int(json.loads(r.stdout)["streams"][0]["nb_read_frames"])


def sheet(mp4: Path, out: Path) -> list[int]:
    from PIL import Image  # noqa: PLC0415

    n = frames(mp4)
    idx = sorted({0, (n - 1) // 4, (n - 1) // 2, 3 * (n - 1) // 4, n - 1})
    with tempfile.TemporaryDirectory(dir=str(HERE)) as tmp:
        sel = "+".join(f"eq(n\\,{i})" for i in idx)
        subprocess.run([str(FFMPEG), "-v", "error", "-y", "-i", str(mp4), "-vf", f"select='{sel}',scale=640:-2",
                        "-fps_mode", "passthrough", str(Path(tmp) / "f_%d.png")], check=True)
        ims = [Image.open(Path(tmp) / f"f_{k + 1}.png").convert("RGB") for k in range(len(idx))]
        w, h = ims[0].size
        s = Image.new("RGB", (w * len(ims), h))
        for k, im in enumerate(ims):
            s.paste(im, (k * w, 0))
        out.parent.mkdir(parents=True, exist_ok=True)
        s.save(out)
        for im in ims:
            im.close()
    return idx


#: Chain seams (the first frame of each new clip in the output) whose 2 frames before / after are shown.
SEAMS = {"LONG": [124, 226, 328, 430], "ACH": [124], "E3b4": [124], "E3b2": [124], "AAUD": [124, 243]}


def boundary_sheet(name: str, bounds_in: list[int] | None = None, tag: str = "boundaries") -> list[int] | None:
    """Frames b-2, b-1, b, b+1 around the first two DWD chunk boundaries (from the job's metadata), each 640 wide."""
    from PIL import Image  # noqa: PLC0415

    if bounds_in is None:
        meta = json.loads((HERE / "runs" / name / f"{name}.metadata.json").read_text(encoding="utf-8"))
        chunks = (((meta.get("h3") or {}).get("phases") or {}).get("decode") or {}).get("dwd", {}).get("chunks") or []
        bounds, acc = [], 0
        for c in chunks[:-1]:
            acc += int(c["frames"])
            bounds.append(acc)
        bounds = bounds[:2]
    else:
        bounds = list(bounds_in)
    if not bounds:
        return None
    idx = [i for b in bounds for i in (b - 2, b - 1, b, b + 1)]
    mp4 = HERE / "runs" / name / "output.mp4"
    with tempfile.TemporaryDirectory(dir=str(HERE)) as tmp:
        sel = "+".join(f"eq(n\\,{i})" for i in sorted(set(idx)))
        subprocess.run([str(FFMPEG), "-v", "error", "-y", "-i", str(mp4), "-vf", f"select='{sel}',scale=640:-2",
                        "-fps_mode", "passthrough", str(Path(tmp) / "f_%d.png")], check=True)
        order = sorted(set(idx))
        ims = {n: Image.open(Path(tmp) / f"f_{k + 1}.png").convert("RGB") for k, n in enumerate(order)}
        w, h = next(iter(ims.values())).size
        s = Image.new("RGB", (w * 4, h * len(bounds)))
        for r in range(len(bounds)):
            for c in range(4):
                s.paste(ims[idx[r * 4 + c]], (c * w, r * h))
        s.save(HERE / "sheets" / f"{name}_{tag}.png")
        for im in ims.values():
            im.close()
    return idx


def main() -> int:
    made = {}
    for name in ("L4", "L5", "LONG"):
        if (not ONLY or name in ONLY) and (HERE / "runs" / name / "output.mp4").is_file():
            print(f"{name}: DWD chunk boundary frames {boundary_sheet(name)} -> sheets/{name}_boundaries.png")
    for name, seams in SEAMS.items():
        if (not ONLY or name in ONLY) and (HERE / "runs" / name / "output.mp4").is_file():
            print(f"{name}: seam frames {boundary_sheet(name, seams, 'seams')} -> sheets/{name}_seams.png")
    for name, sub in TARGETS.items():
        if ONLY and name not in ONLY:
            continue
        if Path(sub).is_absolute():
            mp4 = Path(sub)
        else:
            mp4 = HERE / "runs" / name / sub if "/" not in sub else HERE / "runs" / sub
        if not mp4.is_file():
            print(f"{name}: no mp4, skipped")
            continue
        made[name] = sheet(mp4, HERE / "sheets" / f"{name}.png")
        print(f"{name}: frames {made[name]} -> sheets/{name}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
