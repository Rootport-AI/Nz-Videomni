r"""CPU only: PSNR between frames of a generated mp4 and the key-frame images (gate 1-G4).

    .venv-engine-h3\Scripts\python.exe -I outputs\h3-keyframes-2026-10-19\psnr_keyframes.py ^
        OUT.mp4 --ref 0=outputs\h3-keyframes-2026-10-19\kf_first.png ^
                --ref last=outputs\h3-keyframes-2026-10-19\kf_last.png [--json P]

FRAME is a pixel frame index of the mp4, or ``last`` (the last decoded frame). The reference
is put on the canvas the way the worker does (engine_h3.image_cond.load_canvas_image:
cover + centre crop; a 640x384 image on a 640x384 canvas is unchanged). Frames are
decoded with PyAV as rgb24 and the PSNR is 10*log10(255^2 / MSE) over float64 --
the same as ``psnr_first_frame`` in Docs/Outputs-archive/h3-gate2/g4_i2v_resident.py
(2a-G4: 30.3 dB) and ``psnr`` in run_2b_g2.py (2b-G2: 29.78 dB, ffmpeg rgb24 there).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import av
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from engine_h3.image_cond import load_canvas_image  # noqa: E402  (PIL only; no torch use)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mp4")
    ap.add_argument("--ref", action="append", required=True, metavar="FRAME=PATH")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    refs = []
    for spec in args.ref:
        frame, _, path = spec.partition("=")
        refs.append((-1 if frame == "last" else int(frame), path))
    frames = []
    with av.open(args.mp4) as c:
        for f in c.decode(video=0):
            frames.append(f.to_ndarray(format="rgb24"))
    h, w = frames[0].shape[:2]
    rec = {"mp4": args.mp4, "sha256": hashlib.sha256(Path(args.mp4).read_bytes()).hexdigest(),
           "frames": len(frames), "size": [w, h], "psnr": {}}
    for idx, path in refs:
        canvas, src = load_canvas_image(path, w, h)
        ref = np.asarray(canvas, dtype=np.float64)
        got = frames[idx].astype(np.float64)
        mse = float(((got - ref) ** 2).mean())
        rec["psnr"][f"{idx}={Path(path).name}"] = {
            "frame": idx if idx >= 0 else len(frames) + idx, "source_size": list(src),
            "psnr_db": round(10 * math.log10(255.0 ** 2 / mse), 2) if mse > 0 else None,
            "mse": round(mse, 3),
        }
    text = json.dumps(rec, indent=2, ensure_ascii=False)
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
