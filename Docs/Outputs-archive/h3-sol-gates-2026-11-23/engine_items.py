r"""Sol-H3-VAE GPU checks, the items that call the engine directly (engine venv; product code read only).

Started by ``h3_sol_gates.py`` (which runs the GPU gate first); one item per call:

    S1   the experiment's 5 latents (outputs/h3-sol-spike-2026-11-23/runs/gen_a..gen_e) decoded by the
         product's ``engine_h3.sol_vae.SolDecoder`` and by the experiment's ``sol_decoder.SolDecoder``
         (configuration B); the uint8 frames must be equal bit for bit. The product's frames are also
         written as a video-only mp4 (``write_mp4(audio=None)``, as the experiment wrote B.mp4) so the
         orchestrator can compare its stream MD5 with the experiment's ``compare_<x>/B.mp4``.
    S6   a broken Sol file (the first 64 MiB of the real one, in this folder; ``models/`` untouched) through
         ``H3Pipeline.generate``: the job must fail at ``sol_start`` (before the text encoder) with the
         sentence of the facts found.
    S9   the Sol weights' read time: ``SolDecoder.from_file`` 3 times (the OS file cache warm after the
         first), and one read of the whole file with ``FILE_FLAG_NO_BUFFERING`` (bypasses the OS file
         cache: the disk's own time, the part a cold read adds). Nothing of the OS or the product is changed.
    S10  the size limit at 2560 x 1472 (latent 92 x 160), through the product's ``H3Pipeline._decode``: T' = 112
         (1,648,640 latent tokens, inside ``MAX_LATENT_TOKENS``) must decode with Sol (VRAM peaks recorded,
         the WDDM shared memory sampled by the orchestrator), T' = 117 (1,722,240, above) must fall back to
         the stock video decoder with ``sol_fallback.reason == "too_large"``. The latents are latent a tiled
         (time and space); the audio is a silent source waveform (the audio VAE is not built).

    PYTHONPATH=<repo> .venv-engine-h3/Scripts/python.exe -B outputs/h3-sol-gates-2026-11-23/engine_items.py \
        --item S1 --out outputs/h3-sol-gates-2026-11-23/runs/S1 [--dry-run]

``--dry-run`` hides the GPU and checks only the files, the shapes and the imports (no weight is read on a
device). The record is ``<out>/engine.json``; no path with the user's name is written (``rel``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

if "--dry-run" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPIKE = REPO / "outputs" / "h3-sol-spike-2026-11-23"
SOL_INT8 = REPO / "models" / "MiniMaxH3" / "VAE" / "Sol" / "minimax_h3_sol_vae_int8_convrot.safetensors"
BROKEN = HERE / "_material" / "broken_sol_first64MiB.safetensors"
BROKEN_BYTES = 64 * 2**20
PATHS = {
    "transformer": REPO / "models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "text_encoder": REPO / "models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "video_vae": REPO / "models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": REPO / "models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors",
    "tokenizer_dir": REPO / "models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer",
}
ITEMS_S1 = "abcde"
#: S10: 2560 x 1472 -> a 92 x 160 latent grid; 112 = 5n + 2 inside the limit, 117 the next one above.
S10_GRID = (92, 160)
S10_T = {"inside": 112, "above": 117}

sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True


def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(REPO).as_posix()
    except (ValueError, OSError):
        return Path(p).name


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")


def load_latent(item: str):
    import torch  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    return load_file(str(SPIKE / "runs" / f"gen_{item}" / "latent.safetensors"))["video_latent"].to(torch.float32)


def latent_shape(item: str) -> list[int]:
    p = SPIKE / "runs" / f"gen_{item}" / "latent.safetensors"
    with p.open("rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        head = json.loads(fh.read(n))
    return list(head["video_latent"]["shape"])


def peaks() -> dict:
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        return {}
    return {"allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
            "reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20)}


def free_cuda() -> None:
    import gc  # noqa: PLC0415

    import torch  # noqa: PLC0415

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def pin_like_the_worker() -> dict:
    """What the product's load does before a job: comfy-kitchen pinned to its CUDA backend, cuDNN deterministic."""
    import torch  # noqa: PLC0415

    import h3_format  # noqa: PLC0415
    from engine_h3.quant_linear import require_cuda_quant  # noqa: PLC0415

    back = require_cuda_quant((h3_format.INT8,))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {k: ({kk: vv for kk, vv in v.items() if kk in ("available", "disabled")} if isinstance(v, dict) else str(v))
            for k, v in back.items()}


# --------------------------------------------------------------------------- #
# S1
# --------------------------------------------------------------------------- #

def s1(out: Path, dry: bool) -> dict:
    rec: dict = {"latents": {it: latent_shape(it) for it in ITEMS_S1},
                 "experiment_B_mp4": {it: (SPIKE / "runs" / f"compare_{it}" / "B.mp4").is_file() for it in ITEMS_S1}}
    sys.path.insert(0, str(SPIKE))
    import sol_decoder as SD  # noqa: PLC0415  (the experiment's module, read only)

    from engine_h3 import sol_vae as SV  # noqa: PLC0415

    rec["experiment_config_B"] = SD.CONFIGS["B"].label()
    rec["file_problems"] = SV.file_problems(SOL_INT8)
    if dry:
        rec["ok"] = not rec["file_problems"] and all(rec["experiment_B_mp4"].values())
        return rec
    import torch  # noqa: PLC0415

    from engine_h3.media_out import write_mp4  # noqa: PLC0415

    dev = torch.device("cuda")
    rec["backends"] = pin_like_the_worker()
    rec["items"] = {}
    for it in ITEMS_S1:
        r: dict = {}
        lat = load_latent(it)
        free_cuda()
        dec = SV.SolDecoder.from_file(SOL_INT8, device=dev)
        t0 = time.perf_counter()
        prod = dec.decode(lat)
        torch.cuda.synchronize()
        r["product"] = {"seconds": round(time.perf_counter() - t0, 2), "peak": peaks(), "load": dec.load_report,
                        "stats": dec.last_stats, "frames": int(prod.shape[0])}
        del dec
        free_cuda()
        ref_dec = SD.SolDecoder.from_file(SOL_INT8, config=SD.CONFIGS["B"], device=dev)
        t0 = time.perf_counter()
        ref = ref_dec.decode(lat)
        torch.cuda.synchronize()
        r["experiment"] = {"seconds": round(time.perf_counter() - t0, 2), "peak": peaks()}
        del ref_dec
        free_cuda()
        r["equal"] = bool(prod.shape == ref.shape and torch.equal(prod, ref))
        if not r["equal"] and prod.shape == ref.shape:
            d = (prod.to(torch.int16) - ref.to(torch.int16)).abs()
            r["differ"] = {"values": int((d > 0).sum()), "max_abs": int(d.max())}
        mp4 = out / f"product_{it}.mp4"
        write_mp4(str(mp4), prod, fps=24, audio=None)
        r["product_mp4"] = rel(mp4)
        rec["items"][it] = r
        print(f"S1 {it}: equal={r['equal']} product {r['product']['seconds']} s / experiment "
              f"{r['experiment']['seconds']} s", flush=True)
        del prod, ref, lat
    rec["ok"] = all(v["equal"] for v in rec["items"].values())
    return rec


# --------------------------------------------------------------------------- #
# S6
# --------------------------------------------------------------------------- #

def make_broken() -> dict:
    BROKEN.parent.mkdir(parents=True, exist_ok=True)
    if not BROKEN.is_file() or BROKEN.stat().st_size != BROKEN_BYTES:
        with SOL_INT8.open("rb") as src, BROKEN.open("wb") as dst:
            dst.write(src.read(BROKEN_BYTES))
    return {"file": rel(BROKEN), "bytes": BROKEN.stat().st_size, "of": SOL_INT8.stat().st_size}


def pipeline(device: str):
    from engine_h3 import pipeline_h3 as P  # noqa: PLC0415

    paths = P.H3Paths(*(str(PATHS[k]) for k in ("transformer", "text_encoder", "video_vae", "audio_vae",
                                                 "tokenizer_dir")))
    return P, P.H3Pipeline(paths, P.H3LoadOptions(deterministic=True), device=device,
                           progress=lambda *a, **k: None, log=lambda m: print(f"[pipe] {m}", flush=True))


def s6(out: Path, dry: bool) -> dict:
    from engine_h3 import sol_vae as SV  # noqa: PLC0415

    rec: dict = {"broken": make_broken()}
    rec["file_problems"] = SV.file_problems(BROKEN)
    rec["paths_present"] = {k: p.exists() for k, p in PATHS.items()}
    if dry:
        rec["ok"] = len(rec["file_problems"]) >= 1 and all(rec["paths_present"].values())
        return rec
    P, pipe = pipeline("cuda")
    calls: list[str] = []
    real_encode = pipe._encode

    def encode(*a, **k):
        calls.append("encode")
        return real_encode(*a, **k)

    pipe._encode = encode
    rec["load"] = {k: v for k, v in (pipe.load() or {}).items() if k in ("device",)}
    req = P.H3GenerateRequest(prompt="A calm river at dawn.", seed=1, width=640, height=384, num_frames=22,
                              output_path=str(out / "never.mp4"), num_inference_steps=4,
                              sol_vae=P.H3SolVaeInput(str(BROKEN)))
    t0 = time.perf_counter()
    try:
        pipe.generate(req)
        rec["raised"] = None
    except Exception as exc:  # noqa: BLE001
        rec["raised"] = {"type": type(exc).__name__, "message": str(exc).replace(str(REPO), "<repo>")}
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    rec["text_encoder_ran"] = bool(calls)
    rec["resident_after"] = pipe.resident_parts
    pipe.close()
    msg = (rec["raised"] or {}).get("message", "")
    rec["ok"] = (rec["raised"] is not None and not calls and not (out / "never.mp4").exists()
                 and msg.startswith(f"Sol-H3-VAE のファイル（{BROKEN.name}）の照合で、合わない所が 1 か所ありました: "))
    return rec


# --------------------------------------------------------------------------- #
# S9
# --------------------------------------------------------------------------- #

def unbuffered_read_seconds(path: Path) -> dict:
    """Read the whole file once with FILE_FLAG_NO_BUFFERING (the OS file cache is bypassed for this read)."""
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                             wintypes.LPVOID]
    k32.VirtualAlloc.restype = wintypes.LPVOID
    k32.VirtualAlloc.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
    k32.VirtualFree.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD]
    GENERIC_READ, SHARE_READ, OPEN_EXISTING = 0x80000000, 0x1, 3
    NO_BUFFERING, SEQUENTIAL = 0x20000000, 0x08000000
    chunk = 8 * 2**20                                     # a multiple of any sector size
    h = k32.CreateFileW(str(path), GENERIC_READ, SHARE_READ, None, OPEN_EXISTING, NO_BUFFERING | SEQUENTIAL, None)
    if h in (None, wintypes.HANDLE(-1).value):
        return {"error": f"CreateFileW failed ({ctypes.get_last_error()})"}
    buf = k32.VirtualAlloc(None, chunk, 0x3000, 0x04)    # page-aligned
    got, total = wintypes.DWORD(0), 0
    t0 = time.perf_counter()
    try:
        while True:
            if not k32.ReadFile(h, buf, chunk, ctypes.byref(got), None):
                return {"error": f"ReadFile failed ({ctypes.get_last_error()})"}
            if got.value == 0:
                break
            total += got.value
    finally:
        k32.CloseHandle(h)
        k32.VirtualFree(buf, 0, 0x8000)
    sec = time.perf_counter() - t0
    return {"bytes": total, "seconds": round(sec, 2), "mib_per_s": round(total / 2**20 / sec, 1)}


def s9(out: Path, dry: bool) -> dict:
    rec: dict = {"file": rel(SOL_INT8), "bytes": SOL_INT8.stat().st_size}
    if dry:
        rec["ok"] = SOL_INT8.is_file() and os.name == "nt"
        return rec
    import torch  # noqa: PLC0415

    from engine_h3 import sol_vae as SV  # noqa: PLC0415

    pin_like_the_worker()
    rec["no_buffering_read"] = unbuffered_read_seconds(SOL_INT8)
    rec["from_file"] = []
    for _ in range(3):
        free_cuda()
        t0 = time.perf_counter()
        dec = SV.SolDecoder.from_file(SOL_INT8, device=torch.device("cuda"))
        torch.cuda.synchronize()
        rec["from_file"].append({"seconds_wall": round(time.perf_counter() - t0, 2), "load_report": dec.load_report})
        del dec
    free_cuda()
    rec["ok"] = "error" not in rec["no_buffering_read"]
    return rec


# --------------------------------------------------------------------------- #
# S10
# --------------------------------------------------------------------------- #

def s10_latent(T: int):
    """Latent a ([1, 24, 37, 48, 80]) tiled to [1, 24, T, 92, 160]."""
    import torch  # noqa: PLC0415

    a = load_latent("a")
    h, w = S10_GRID
    reps = (1, 1, -(-T // a.shape[2]), -(-h // a.shape[3]), -(-w // a.shape[4]))
    return a.repeat(*reps)[:, :, :T, :h, :w].contiguous().to(torch.float32)


def s10(out: Path, dry: bool) -> dict:
    from engine_h3 import dwd  # noqa: PLC0415
    from engine_h3 import sol_vae as SV  # noqa: PLC0415

    h, w = S10_GRID
    rec: dict = {"limit": SV.MAX_LATENT_TOKENS, "cases": {}}
    for name, T in S10_T.items():
        rec["cases"][name] = {"latent": [1, 24, T, h, w], "tokens": T * h * w, "frames": dwd.frames_for_latents(T),
                              "size_fallback": SV.size_fallback((1, 24, T, h, w))}
    rec["shape_ok"] = (rec["cases"]["inside"]["size_fallback"] is None
                       and rec["cases"]["above"]["size_fallback"] == "too_large"
                       and tuple(s10_latent(5).shape) == (1, 24, 5, h, w))
    if dry:
        rec["ok"] = rec["shape_ok"] and SOL_INT8.is_file()
        return rec
    import torch  # noqa: PLC0415

    P, pipe = pipeline("cuda")
    pin_like_the_worker()
    marks_file = out / "marks.json"
    marks: dict = {}
    for name, T in S10_T.items():
        lat = s10_latent(T).to("cuda")
        frames = dwd.frames_for_latents(T)
        req = P.H3GenerateRequest(prompt="s10", seed=0, width=w * 16, height=h * 16, num_frames=frames,
                                  output_path=str(out / f"{name}.mp4"))
        wave = torch.zeros(2, round(frames * 32000 / 24), dtype=torch.float32)
        entry: dict = {}
        free_cuda()
        marks[name] = {"t0": time.time()}
        write_json(marks_file, marks)
        t0 = time.perf_counter()
        try:
            pipe._decode(req, lat, None, entry, source_wave=wave, sol={"path": str(SOL_INT8), "fallback": None})
            err = None
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        torch.cuda.synchronize()
        marks[name]["t1"] = time.time()
        write_json(marks_file, marks)
        c = rec["cases"][name]
        c.update(seconds=round(time.perf_counter() - t0, 2), peak=peaks(), error=err,
                 vae_source=entry.get("vae_source"), sol=entry.get("sol"), sol_fallback=entry.get("sol_fallback"),
                 frames_written=entry.get("frames"))
        print(f"S10 {name}: T'={T} vae_source={c['vae_source']} {c['seconds']} s peak {c['peak']} err={err}", flush=True)
        del lat, wave
        free_cuda()
        (out / f"{name}.mp4").unlink(missing_ok=True)        # 2560 x 1472 x ~380 frames: not kept
    pipe.close()
    rec["marks"] = marks
    ins, abv = rec["cases"]["inside"], rec["cases"]["above"]
    rec["ok"] = (ins["error"] is None and ins["vae_source"] == "sol" and abv["error"] is None
                 and abv["vae_source"] in ("file", "resident") and (abv["sol_fallback"] or {}).get("reason") == "too_large"
                 and ins["frames_written"] == ins["frames"] and abv["frames_written"] == abv["frames"])
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", required=True, choices=["S1", "S6", "S9", "S10"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec: dict = {"item": a.item, "dry_run": a.dry_run, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        import torch  # noqa: PLC0415

        rec["torch"] = torch.__version__
        rec["cuda_available"] = torch.cuda.is_available()
        rec.update({"S1": s1, "S6": s6, "S9": s9, "S10": s10}[a.item](out, a.dry_run))
    except BaseException as exc:  # noqa: BLE001
        rec.update(ok=False, error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc().replace(str(REPO), "<repo>"))
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json(out / "engine.json", rec)
    print(f"{a.item}: ok={rec.get('ok')}" + (f" error={rec.get('error')}" if rec.get("error") else ""), flush=True)
    return 0 if rec.get("ok") else 3


if __name__ == "__main__":
    sys.exit(main())
