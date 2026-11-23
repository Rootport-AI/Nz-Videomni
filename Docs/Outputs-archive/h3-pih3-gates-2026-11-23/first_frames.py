r"""First 8 frames side by side (each 400 wide) and the differences of consecutive frames 0-1 .. 6-7.

    .venv/Scripts/python.exe -X utf8 -B outputs/h3-pih3-gates-2026-11-23/first_frames.py

Writes sheets/<name>_first8.png and first_frames.json (CPU only: ffmpeg + numpy + Pillow).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FFMPEG = REPO / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE = REPO / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"
TARGETS = {"A2V_pi": "A2V/A2V_pi.mp4", "A2V_turbo": "A2V/A2V_turbo.mp4", "X1": "X1/output.mp4",
           "X2": "X2/output.mp4", "X3": "X3/output.mp4", "I2V1": "I2V1/output.mp4",
           "G3_runB_s0.5": str(REPO / "outputs" / "h3_a2v_g3" / "runB_s0.5" / "output.mp4")}
N = 8


def first_frames(mp4: Path):
    import numpy as np  # noqa: PLC0415

    r = subprocess.run([str(FFPROBE), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "json", str(mp4)], capture_output=True, text=True)
    s = json.loads(r.stdout)["streams"][0]
    w, h = int(s["width"]), int(s["height"])
    raw = subprocess.run([str(FFMPEG), "-v", "error", "-i", str(mp4), "-frames:v", str(N), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def main() -> int:
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    out = {}
    for name, sub in TARGETS.items():
        mp4 = Path(sub) if Path(sub).is_absolute() else HERE / "runs" / sub
        if not mp4.is_file():
            print(f"{name}: no mp4")
            continue
        f = first_frames(mp4)
        diffs = []
        for i in range(len(f) - 1):
            d = np.abs(f[i].astype(np.int16) - f[i + 1].astype(np.int16)).astype(np.float64)
            mse = float((d ** 2).mean())
            diffs.append({"pair": f"{i}-{i + 1}", "mean_abs_diff": round(float(d.mean()), 2),
                          "psnr_db": None if mse == 0 else round(float(10 * np.log10(255 ** 2 / mse)), 2)})
        out[name] = {"mp4": str(mp4.relative_to(REPO)).replace("\\", "/"), "frames": int(len(f)), "diffs": diffs}
        ims = [Image.fromarray(x).resize((400, round(400 * x.shape[0] / x.shape[1])), Image.LANCZOS) for x in f]
        sheet = Image.new("RGB", (400 * len(ims), ims[0].size[1]))
        for k, im in enumerate(ims):
            sheet.paste(im, (400 * k, 0))
        (HERE / "sheets").mkdir(exist_ok=True)
        sheet.save(HERE / "sheets" / f"{name}_first8.png")
        print(name, " ".join(f"{d['pair']}:{d['mean_abs_diff']}/{d['psnr_db']}dB" for d in diffs))
    (HERE / "first_frames.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
