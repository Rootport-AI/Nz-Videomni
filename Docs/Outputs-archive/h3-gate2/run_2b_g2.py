r"""Gate 2b-G2 (MiniMax H3 stage 2, plan §6): real generation THROUGH THE API. GPU.

Drives the running backend the established way (VERIFICATION_LOG §39.5 / §39.6):
``.venv\Scripts\python.exe -m mcp_server`` as a stdio child, called from the
``mcp`` package's ClientSession (initialize -> tools/call). ONE MCP session for
the whole run. Nothing is imported from the repository; the script only reads
``outputs/<job_id>/metadata.json`` / ``output.mp4`` and ``logs/*.log``.

Run (the backend must already be listening on 127.0.0.1:18620, state unloaded
or ready, no job running):

    S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\.venv\Scripts\python.exe <this file>

Steps
  0  backend_status (reachable, no job running); state.json copied aside
  1  list_models: MiniMaxH3 is listed + installed, ``constraints`` == plan §3.3
  2  load_pipeline(base_model="MiniMaxH3"): state ``ready``, base_model
     MiniMaxH3, server.log shows the REAL engine_h3 worker (not the mock)
     -- the app-side precheck (inspect_h3_dit / inspect_h3_te) runs on the
     real files inside this call; a refusal comes back as MODEL_INCOMPATIBLE
  3  T2V #1  keep_resident=true  (embed_mp4_metadata=false so the sha256 can be
     compared with the stage-1 / 2a fingerprint 0da6111c...)
  4  T2V #2  keep_resident=true  (expected: DiT + VAE from ``resident``, faster)
  5  upload_image(i2v_input_square.png, 384x384) -> I2V on the 640x384 canvas,
     conditioning_images=[{image_id, frame_idx: 0}], NO keep_resident key
     (uses what is kept, then releases it); embed_mp4_metadata left at default
     (true) and read back with get_mp4_info
  6  per-job table from metadata.json -> report JSON
  7  sha256 of both T2V mp4s vs 0da6111c (recorded; a mismatch is a finding,
     not an abort)
  8  PSNR of the I2V first frame vs the cover+centre-cropped input (the same
     geometry as engine_h3/image_cond.load_canvas_image, re-implemented here
     with PIL so this script needs no torch)
  9  unload_pipeline (stops the H3 worker; skip with --keep-loaded)

All conditions: 640x384, 124 f (17*7+5), 24 fps, 20 steps, shift 12 / 3,
seed 42, sdpa, prefetch on (the API defaults), the G4/G5 prompt.

Exit code 0 = every gate item true. The full record is
``<out>/g2b_api.json``; a one-line-per-job table is printed at the end.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent
REPO_DEFAULT = Path(r"S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni")

T2V_SHA = "0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9"
PROMPT = (
    "A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in "
    "puddles. She looks at the camera and says <d>[English] It finally stopped raining.</d>"
)
W, H, F, FPS, SEED = 640, 384, 124, 24.0, 42
STEPS, SHIFT_V, SHIFT_A = 20, 12.0, 3.0
BASE = "MiniMaxH3"
REAL_BACKEND = "minimax-h3"
REAL_LOAD_LINE = "Loading pipeline (REAL engine_h3 worker)"

#: plan §3.3, verbatim.
EXPECTED_CONSTRAINTS = {
    "num_frames": {"step": 17, "offset": 5, "min": 22, "max": 481},
    "dimension_multiple": 32,
    "frame_rate": {"min": 24, "max": 24},
    "sampling": {
        "num_inference_steps": {"min": 1, "max": 50, "step": 1, "default": 20},
        "shift_video": {"min": 1, "max": 20, "step": 0.1, "default": 12},
        "shift_audio": {"min": 1, "max": 20, "step": 0.1, "default": 3},
    },
}

POLL_SEC = 10.0
TERMINAL = {"completed", "failed", "cancelled"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class ToolFailed(RuntimeError):
    pass


def payload(res) -> dict:
    """structuredContent (unwrapping FastMCP's ``{"result": ...}``) or the JSON text."""
    if getattr(res, "isError", False):
        texts = [getattr(c, "text", None) or str(c) for c in (res.content or [])]
        raise ToolFailed(" | ".join(texts))
    sc = getattr(res, "structuredContent", None)
    if isinstance(sc, dict):
        if set(sc) == {"result"} and isinstance(sc["result"], (dict, list)):
            return sc["result"]
        return sc
    for c in getattr(res, "content", []) or []:
        text = getattr(c, "text", None)
        if text:
            try:
                return json.loads(text)
            except ValueError:
                return {"_text": text}
    return {}


def same_number_tree(a, b) -> bool:
    """Structural equality with 12 == 12.0."""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same_number_tree(a[k], b[k]) for k in a)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return float(a) == float(b)
    return a == b


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log_offset(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def read_from(path: Path, offset: int) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(offset)
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def find_ffmpeg(repo: Path) -> str:
    local = repo / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
    if local.exists():
        return str(local)
    found = shutil.which("ffmpeg")
    if not found:
        raise SystemExit("ffmpeg not found (tools\\ffmpeg\\bin or PATH)")
    return found


def canvas_like_worker(src: Path, width: int, height: int):
    """engine_h3/image_cond.load_canvas_image, copied: EXIF transpose, RGB, cover
    (ceil) with LANCZOS, centre crop. Returns (PIL image, (src_w, src_h))."""
    from PIL import Image, ImageOps

    with Image.open(src) as opened:
        img = ImageOps.exif_transpose(opened).convert("RGB")
    src_w, src_h = img.size
    scale = max(height / src_h, width / src_w)
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)
    if (new_w, new_h) != (src_w, src_h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
    top = (new_h - height) // 2
    left = (new_w - width) // 2
    return img.crop((left, top, left + width, top + height)), (src_w, src_h)


def first_frame_rgb(ffmpeg: str, mp4: Path, width: int, height: int):
    import numpy as np

    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(mp4), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
    )
    arr = np.frombuffer(proc.stdout, dtype=np.uint8)
    if arr.size != width * height * 3:
        raise RuntimeError(f"first frame size {arr.size} != {width}x{height}x3")
    return arr.reshape(height, width, 3)


def psnr(a, b) -> dict:
    import numpy as np

    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return {"psnr_db": round(10 * math.log10(255.0 ** 2 / mse), 2) if mse > 0 else None, "mse": round(mse, 3)}


def job_row(kind: str, md: dict, client_seconds: float, mp4: Path) -> dict:
    req = md.get("request") or {}
    h3 = md.get("h3") or {}
    phases = h3.get("phases") or {}
    vram = md.get("vram_optimization") or {}
    return {
        "kind": kind,
        "job_id": md.get("job_id"),
        "backend": md.get("backend"),
        "generation_mode": md.get("generation_mode"),
        "seed_used": md.get("seed_used"),
        "request_steps": req.get("num_inference_steps"),
        "request_shift_video": req.get("shift_video"),
        "request_shift_audio": req.get("shift_audio"),
        "request_keep_resident": req.get("keep_resident"),
        "request_images": len(req.get("conditioning_images") or []),
        "done_steps": h3.get("num_inference_steps"),
        "done_shift_video": h3.get("shift_video"),
        "done_shift_audio": h3.get("shift_audio"),
        "done_sampler": h3.get("sampler"),
        "images_used": h3.get("images_used"),
        "keep_resident_used": md.get("keep_resident_used"),
        "attention_used": md.get("attention_used"),
        "block_swap_prefetch_used": md.get("block_swap_prefetch_used"),
        "dit_source": (phases.get("denoise") or {}).get("dit_source"),
        "vae_source": (phases.get("decode") or {}).get("vae_source"),
        "phase_seconds": {k: (v or {}).get("seconds") for k, v in phases.items() if isinstance(v, dict)},
        "image_encode": {
            k: (phases.get("image_encode") or {}).get(k)
            for k in ("canvas", "source_size", "vision_tokens", "seconds")
        } if "image_encode" in phases else None,
        "worker_seconds": h3.get("seconds"),
        "generation_time_seconds": md.get("generation_time_seconds"),
        "client_wall_seconds": round(client_seconds, 1),
        "peak_vram_mb": vram.get("peak_vram_mb"),
        "peak_vram_reserved_mb": md.get("peak_vram_reserved_mb"),
        "rss_peak_gib": h3.get("rss_peak_gib"),
        "commit_gib": h3.get("commit_gib"),
        "commit_peak_gib": h3.get("commit_peak_gib"),
        "quant_backends": h3.get("quant_backends"),
        "output_resolution": (md.get("output") or {}).get("resolution"),
        "output_duration_seconds": (md.get("output") or {}).get("duration_seconds"),
        "sha256": sha256(mp4) if mp4.exists() else None,
    }


async def run(args) -> int:
    repo: Path = args.repo
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "g2b_api.json"
    report: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "conditions": {
        "width": W, "height": H, "num_frames": F, "frame_rate": FPS, "seed": SEED,
        "num_inference_steps": STEPS, "shift_video": SHIFT_V, "shift_audio": SHIFT_A,
        "prompt": PROMPT, "image": str(args.image)}}

    def save() -> None:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    server_log = repo / "logs" / "server.log"
    worker_log = repo / "logs" / "h3_worker.log"
    server_off, worker_off = log_offset(server_log), log_offset(worker_log)
    ffmpeg = find_ffmpeg(repo)

    # state.json carries the selected base model across restarts; the load
    # below switches it to MiniMaxH3. Kept aside so the README's clean-up can
    # put the owner's selection back.
    state_json = repo / "state.json"
    if state_json.exists():
        shutil.copy2(state_json, out / "state.json.before")

    params = StdioServerParameters(
        command=str(repo / ".venv" / "Scripts" / "python.exe"),
        args=["-m", "mcp_server"],
        cwd=str(repo),
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    jobs: dict[str, dict] = {}
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()

            async def call(tool: str, a: dict | None = None):
                return payload(await s.call_tool(tool, a or {}))

            async def run_job(name: str, submit_args: dict) -> dict:
                log(f"{name}: submit_generate")
                t0 = time.monotonic()
                sub = await call("submit_generate", submit_args)
                job_id = sub["job_id"]
                log(f"{name}: job_id={job_id}")
                last = None
                while True:
                    await asyncio.sleep(POLL_SEC)
                    js = await call("job_status", {"job_id": job_id})
                    line = f"{js.get('status')} progress={js.get('progress')} stage={js.get('stage')}"
                    if line != last:
                        log(f"  {int(time.monotonic() - t0)}s {line}")
                        last = line
                    if js.get("status") in TERMINAL:
                        break
                    if time.monotonic() - t0 > args.job_timeout:
                        raise SystemExit(f"{name}: TIMEOUT after {args.job_timeout}s (job {job_id} left running)")
                wall = time.monotonic() - t0
                rec = {"job_id": job_id, "submit_args": submit_args, "final": js, "client_wall_seconds": wall}
                if js.get("status") != "completed":
                    rec["error"] = js.get("error")
                    jobs[name] = rec
                    report["jobs"] = jobs
                    save()
                    raise SystemExit(f"{name}: job ended {js.get('status')}: {js.get('error')}")
                job_dir = repo / "outputs" / job_id
                md = json.loads((job_dir / "metadata.json").read_text(encoding="utf-8"))
                shutil.copy2(job_dir / "output.mp4", out / f"{name}_output.mp4")
                shutil.copy2(job_dir / "metadata.json", out / f"{name}_metadata.json")
                rec["row"] = job_row(name, md, wall, job_dir / "output.mp4")
                jobs[name] = rec
                report["jobs"] = jobs
                save()
                row = rec["row"]
                log(f"{name}: done {row['generation_time_seconds']}s backend={row['backend']} "
                    f"dit={row['dit_source']} vae={row['vae_source']} keep={row['keep_resident_used']}")
                if row["backend"] != REAL_BACKEND:
                    raise SystemExit(f"{name}: backend is {row['backend']!r}, not {REAL_BACKEND!r} "
                                     "(the app fell back to the mock: see server.log for the missing paths)")
                return rec

            # 0 ------------------------------------------------------------
            st = await call("backend_status")
            report["backend_status_before"] = st
            if not st.get("reachable"):
                raise SystemExit(f"backend not reachable: {st.get('error')}")
            q = (st.get("status") or {}).get("queue") or {}
            if q.get("running") or q.get("queued"):
                raise SystemExit(f"a job is already queued/running: {q}")
            log(f"backend: state={st['status'].get('state')} base_model={st['status'].get('base_model')}")

            # 1 ------------------------------------------------------------
            models = await call("list_models")
            h3 = next((b for b in models.get("base_models", []) if b.get("id") == BASE), None)
            report["models_h3"] = h3
            save()
            if h3 is None:
                raise SystemExit("MiniMaxH3 is not in GET /models base_models[]")

            # 2 ------------------------------------------------------------
            log("load_pipeline(base_model=MiniMaxH3)")
            t0 = time.monotonic()
            load = await call("load_pipeline", {"base_model": BASE, "wait_sec": 45})
            report["load_result"] = load
            while True:
                st = await call("backend_status")
                state = (st.get("status") or {}).get("state")
                if state in ("ready", "error") and not st.get("pipeline_load_in_flight"):
                    break
                if time.monotonic() - t0 > args.load_timeout:
                    raise SystemExit(f"load: TIMEOUT after {args.load_timeout}s (state={state})")
                await asyncio.sleep(5)
            report["load_seconds"] = round(time.monotonic() - t0, 1)
            report["backend_status_after_load"] = st
            server_text = read_from(server_log, server_off)
            report["load_real_line_seen"] = REAL_LOAD_LINE in server_text
            save()
            log(f"load: state={state} base_model={st['status'].get('base_model')} "
                f"{report['load_seconds']}s real_line={report['load_real_line_seen']}")
            if state != "ready" or st["status"].get("base_model") != BASE:
                raise SystemExit(f"load did not reach ready on {BASE}: {st.get('status')}")

            common = {
                "prompt": PROMPT, "width": W, "height": H, "num_frames": F, "frame_rate": FPS,
                "seed": SEED, "num_inference_steps": STEPS, "shift_video": SHIFT_V, "shift_audio": SHIFT_A,
            }
            # 3, 4 ---------------------------------------------------------
            await run_job("t2v1", {**common, "keep_resident": True, "embed_mp4_metadata": False})
            await run_job("t2v2", {**common, "keep_resident": True, "embed_mp4_metadata": False})

            # 5 ------------------------------------------------------------
            up = await call("upload_image", {"file_path": str(args.image)})
            report["upload"] = up
            log(f"upload: image_id={up.get('image_id')} {up.get('width')}x{up.get('height')}")
            await run_job("i2v", {**common, "conditioning_images": [{"image_id": up["image_id"], "frame_idx": 0}]})
            try:
                info = await call("get_mp4_info", {"path": str(repo / "outputs" / jobs["i2v"]["job_id"] / "output.mp4")})
                comment = info.get("comment")
                report["i2v_mp4_comment_present"] = bool(comment)
                report["i2v_mp4_comment_has_h3"] = bool(comment) and '"h3"' in comment
            except ToolFailed as exc:
                report["i2v_mp4_info_error"] = str(exc)

            # 9 ------------------------------------------------------------
            if not args.keep_loaded:
                report["unload"] = await call("unload_pipeline")
                log(f"unload: {report['unload']}")
            report["backend_status_end"] = await call("backend_status")

    # 7, 8 ---------------------------------------------------------------
    t1, t2, i2v = (jobs[k]["row"] for k in ("t2v1", "t2v2", "i2v"))
    # stored_path is relative to the repository (upload_store.stored_relpath).
    stored = Path(up.get("stored_path") or "")
    if up.get("stored_path") and not stored.is_absolute():
        stored = repo / stored
    ref_src = stored if up.get("stored_path") and stored.is_file() else Path(args.image)
    canvas, src_size = canvas_like_worker(ref_src, W, H)
    canvas.save(out / "i2v_canvas_reference.png")
    import numpy as np
    from PIL import Image

    frame0 = first_frame_rgb(ffmpeg, out / "i2v_output.mp4", W, H)
    Image.fromarray(frame0).save(out / "i2v_first_frame.png")
    report["i2v_first_frame_vs_canvas"] = {**psnr(frame0, np.asarray(canvas)), "reference_from": str(ref_src),
                                           "source_size": list(src_size)}

    def echoes(row: dict, prefix: str) -> bool:
        return (row[f"{prefix}_steps"] == STEPS and row[f"{prefix}_shift_video"] is not None
                and float(row[f"{prefix}_shift_video"]) == SHIFT_V and row[f"{prefix}_shift_audio"] is not None
                and float(row[f"{prefix}_shift_audio"]) == SHIFT_A)

    rows = (t1, t2, i2v)
    h3 = report["models_h3"]
    report["verdict"] = {
        "models_h3_installed": bool(h3.get("installed")),
        "constraints_match_plan": same_number_tree(h3.get("constraints"), EXPECTED_CONSTRAINTS),
        "load_ready_on_h3": True,  # reaching here required it
        "load_real_worker": report["load_real_line_seen"],
        "backend_real_all": all(r["backend"] == REAL_BACKEND for r in rows),
        "h3_block_in_metadata_all": all(r["done_steps"] is not None for r in rows),
        "steps_shift_in_request_all": all(echoes(r, "request") for r in rows),
        "steps_shift_echoed_in_done_all": all(echoes(r, "done") for r in rows),
        "keep_resident_used_on_off": [t1["keep_resident_used"], t2["keep_resident_used"], i2v["keep_resident_used"]]
        == ["on", "on", "off"],
        "t2v1_read_from_file": t1["dit_source"] == "file" and t1["vae_source"] == "file",
        "t2v2_reads_skipped": t2["dit_source"] == "resident" and t2["vae_source"] == "resident",
        "t2v2_faster": (t2["generation_time_seconds"] or 0) < (t1["generation_time_seconds"] or 0),
        "i2v_mode_and_image_used": i2v["generation_mode"] == "i2v" and i2v["images_used"] == 1,
        "i2v_crop_path": bool(i2v["image_encode"]) and list(i2v["image_encode"].get("source_size") or []) == [384, 384]
        and list(i2v["image_encode"].get("canvas") or []) in ([W, H], [H, W]),
    }
    report["recorded"] = {
        "t2v1_sha256_matches_0da6111c": t1["sha256"] == T2V_SHA,
        "t2v2_sha256_matches_0da6111c": t2["sha256"] == T2V_SHA,
        "t2v1_equals_t2v2": t1["sha256"] == t2["sha256"],
        "t2v1_sha256": t1["sha256"],
        "t2v2_sha256": t2["sha256"],
        "note_sha": "the app sends deterministic=true on load; the 2a-G4 selftest ran with deterministic=false. "
                    "A mismatch is recorded, then judged by the supervisor.",
        "i2v_psnr_db": report["i2v_first_frame_vs_canvas"]["psnr_db"],
        "i2v_mp4_comment_present": report.get("i2v_mp4_comment_present"),
        "i2v_mp4_comment_has_h3": report.get("i2v_mp4_comment_has_h3"),
        "seconds_t2v1_t2v2_i2v": [r["generation_time_seconds"] for r in rows],
        "dit_vae_source": [[r["dit_source"], r["vae_source"]] for r in rows],
        "peak_vram_mb": [r["peak_vram_mb"] for r in rows],
        "commit_peak_gib": [r["commit_peak_gib"] for r in rows],
        "commit_gib_after_job": [r["commit_gib"] for r in rows],
    }
    (out / "server.log.part").write_text(read_from(server_log, server_off), encoding="utf-8")
    (out / "h3_worker.log.part").write_text(read_from(worker_log, worker_off), encoding="utf-8")
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()

    print()
    hdr = ("kind", "backend", "steps", "shift_v", "shift_a", "keep", "imgs", "dit", "vae", "gen_s", "vramMB", "commitPk")
    print(" | ".join(hdr))
    for r in rows:
        print(" | ".join(str(x) for x in (r["kind"], r["backend"], r["done_steps"], r["done_shift_video"],
                                          r["done_shift_audio"], r["keep_resident_used"], r["images_used"],
                                          r["dit_source"], r["vae_source"], r["generation_time_seconds"],
                                          r["peak_vram_mb"], r["commit_peak_gib"])))
    print(json.dumps({"verdict": report["verdict"], "recorded": report["recorded"]}, indent=2, ensure_ascii=False))
    return 0 if all(report["verdict"].values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    ap.add_argument("--out", type=Path, default=None, help="default: <repo>/outputs/h3-gate2/g2b_api")
    ap.add_argument("--image", type=Path, default=HERE / "i2v_input_square.png")
    ap.add_argument("--job-timeout", type=float, default=1800.0)
    ap.add_argument("--load-timeout", type=float, default=600.0)
    ap.add_argument("--keep-loaded", action="store_true", help="skip the final unload_pipeline")
    args = ap.parse_args()
    if args.out is None:
        args.out = args.repo / "outputs" / "h3-gate2" / "g2b_api"
    if not args.image.is_file():
        raise SystemExit(f"image not found: {args.image}")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
