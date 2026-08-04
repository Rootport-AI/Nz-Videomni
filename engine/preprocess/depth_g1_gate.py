"""G1 gate harness for the depth control signal (needs the ENGINE venv + a GPU).

    .venv-engine\\Scripts\\python.exe -m engine.preprocess.depth_g1_gate \\
        --video uploads/videos/<id>/source.mp4 --frames 121 --dump outputs/g1_depth

Runs the REAL production path once — ``get_processor("depth")`` +
``driver.preprocess_video`` — while tapping the pre-encode frames, so all three
G1 measurements describe exactly what a job would produce:

  (a) near = white. Reported as statistics + side-by-side PNG dumps for the
      visual call. It can also be made OBJECTIVE by naming two boxes in the
      source, one on something near the camera and one on something far:
      ``--near-box x,y,w,h --far-box x,y,w,h`` then PASS/FAILs on
      mean(near) > mean(far) + margin.
  (b) inter-frame flicker. Raw consecutive-frame deltas confound flicker with
      real motion, so the headline number is measured ONLY over pixels the
      SOURCE barely moved in (source delta < --static-delta): on those pixels
      the depth should not move either.
  (c) mp4 round-trip degradation. The pre-encode control frames are compared
      against the decoded mp4 (mean/max absolute error, PSNR) plus a banding
      proxy: how many distinct gray levels survive the encode. The threshold is
      calibrated on the driver's existing mp4v encoder, which canny/pose already
      ship through — this catches a gradation COLLAPSE, not codec loss per se.

Exit code 0 only when every enabled check passes. (a) without the two boxes is
reported but not gated — that call is the owner's, on the dumped frames.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from engine.preprocess import get_processor, preprocess_video


class _Tap:
    """Wraps the real depth processor to keep the in/out frames for measuring."""

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.sources: list[np.ndarray] = []
        self.controls: list[np.ndarray] = []

    def process_video(self, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
        self.sources = [f.copy() for f in frames_bgr]
        out = self._inner.process_video(frames_bgr)  # type: ignore[attr-defined]
        self.controls = [f.copy() for f in out]
        return out

    def release(self) -> None:
        release = getattr(self._inner, "release", None)
        if callable(release):
            release()


def _box(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 4:
        raise SystemExit(f"--*-box wants x,y,w,h — got {text!r}")
    return (parts[0], parts[1], parts[2], parts[3])


def _mean_in(frame: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x, y, w, h = box
    return float(frame[y : y + h, x : x + w].mean())


def _decode(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="raw reference video")
    ap.add_argument("--frames", type=int, default=121, help="frame_cap (generation length)")
    ap.add_argument("--dump", default="", help="directory for src/depth PNG pairs")
    ap.add_argument("--near-box", default="", help="x,y,w,h over something NEAR")
    ap.add_argument("--far-box", default="", help="x,y,w,h over something FAR")
    ap.add_argument("--near-far-margin", type=float, default=20.0, help="gray levels")
    ap.add_argument("--static-delta", type=float, default=2.0, help="source delta that counts as static")
    ap.add_argument("--flicker-threshold", type=float, default=2.0, help="gray levels")
    # Calibrated against the driver's existing mp4v encoder — the SAME lossy
    # codec canny/pose already ship through — on a 1280x720 clip: mean|err|=4.11,
    # PSNR 35.7 dB, 199 of 234 gray levels surviving. The gate is there to catch
    # a COLLAPSE of the gradation, not to re-litigate the codec choice.
    ap.add_argument("--roundtrip-threshold", type=float, default=6.0, help="mean abs gray levels")
    args = ap.parse_args()

    src = Path(args.video)
    if not src.exists():
        raise SystemExit(f"no such video: {src}")
    out_dir = Path(args.dump) if args.dump else Path("outputs") / "g1_depth"
    out_dir.mkdir(parents=True, exist_ok=True)
    control_mp4 = out_dir / "control_depth.mp4"

    tap = _Tap(get_processor("depth"))
    t0 = time.perf_counter()
    n = preprocess_video(src, control_mp4, tap, frame_cap=args.frames)
    elapsed = time.perf_counter() - t0
    print(f"[run] {src} -> {control_mp4}  frames={n}  cap={args.frames}  {elapsed:.2f}s "
          f"({n / max(elapsed, 1e-9):.2f} fps)")

    controls = [c[:, :, 0].astype(np.int16) for c in tap.controls]
    sources = [cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.int16) for s in tap.sources]
    height, width = controls[0].shape
    print(f"[run] resolution={width}x{height}  (preserved end-to-end)")

    failures: list[str] = []

    # ---- (a) near = white ---------------------------------------------------
    lo = min(int(c.min()) for c in controls)
    hi = max(int(c.max()) for c in controls)
    print(f"\n[a] gray range over the clip: {lo}..{hi} "
          f"(clip-wide min-max normalisation should give 0..255)")
    for i in (0, n // 2, n - 1):
        print(f"[a] frame {i:4d}: min={int(controls[i].min()):3d} "
              f"max={int(controls[i].max()):3d} mean={float(controls[i].mean()):6.2f}")
    near, far = _box(args.near_box), _box(args.far_box)
    if near and far:
        near_means = [_mean_in(c, near) for c in controls]
        far_means = [_mean_in(c, far) for c in controls]
        gap = float(np.mean(near_means) - np.mean(far_means))
        ok = gap > args.near_far_margin
        print(f"[a] near={np.mean(near_means):.2f} far={np.mean(far_means):.2f} "
              f"gap={gap:+.2f} (need > {args.near_far_margin}) -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append("a: near is not brighter than far")
    else:
        print("[a] no --near-box/--far-box given: orientation is a VISUAL call on "
              f"the dumps in {out_dir} (not gated)")

    # ---- (b) flicker --------------------------------------------------------
    raw_deltas, static_deltas, static_px = [], [], 0
    for i in range(1, n):
        d_ctrl = np.abs(controls[i] - controls[i - 1])
        d_src = np.abs(sources[i] - sources[i - 1])
        raw_deltas.append(float(d_ctrl.mean()))
        mask = d_src < args.static_delta
        if mask.any():
            static_deltas.append(float(d_ctrl[mask].mean()))
            static_px += int(mask.sum())
    static_mean = float(np.mean(static_deltas)) if static_deltas else 0.0
    static_p95 = float(np.percentile(static_deltas, 95)) if static_deltas else 0.0
    static_max = float(np.max(static_deltas)) if static_deltas else 0.0
    coverage = static_px / max((n - 1) * height * width, 1)
    print(f"\n[b] consecutive-frame |delta|, all pixels: mean={np.mean(raw_deltas):.3f} "
          f"max={np.max(raw_deltas):.3f} gray levels (includes real motion)")
    print(f"[b] STATIC pixels only (source delta < {args.static_delta}, "
          f"{coverage * 100:.1f}% of pixels): mean={static_mean:.3f} "
          f"p95={static_p95:.3f} max={static_max:.3f}")
    ok = static_mean <= args.flicker_threshold
    print(f"[b] need mean <= {args.flicker_threshold} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("b: depth flickers on static pixels")

    # ---- (c) mp4 round trip -------------------------------------------------
    decoded = _decode(control_mp4)
    if len(decoded) != n:
        failures.append(f"c: mp4 has {len(decoded)} frames, expected {n}")
        print(f"\n[c] FAIL: mp4 frame count {len(decoded)} != {n}")
    else:
        errs = []
        levels_before, levels_after = [], []
        for i in range(n):
            after = decoded[i][:, :, 0].astype(np.int16)
            errs.append(np.abs(controls[i] - after))
            levels_before.append(len(np.unique(controls[i])))
            levels_after.append(len(np.unique(after)))
        stacked = np.concatenate([e.ravel() for e in errs])
        mean_err = float(stacked.mean())
        max_err = int(stacked.max())
        mse = float((stacked.astype(np.float64) ** 2).mean())
        psnr = 10.0 * np.log10(255.0 * 255.0 / mse) if mse > 0 else float("inf")
        print(f"\n[c] mp4 round trip: mean|err|={mean_err:.3f} max|err|={max_err} "
              f"PSNR={psnr:.2f} dB")
        print(f"[c] distinct gray levels per frame: before={np.mean(levels_before):.1f} "
              f"after={np.mean(levels_after):.1f} (banding proxy)")
        ok = mean_err <= args.roundtrip_threshold
        print(f"[c] need mean|err| <= {args.roundtrip_threshold} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append("c: mp4 encode degrades the depth gradation")

    # ---- dumps for the visual call -----------------------------------------
    idxs = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    for i in idxs:
        cv2.imwrite(str(out_dir / f"src_{i:04d}.png"), tap.sources[i])
        cv2.imwrite(str(out_dir / f"depth_{i:04d}.png"), tap.controls[i])
    print(f"\n[dump] {len(idxs)} src/depth pairs -> {out_dir}")

    if failures:
        print("\nG1 FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nG1 checks passed (visual items excluded)")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
