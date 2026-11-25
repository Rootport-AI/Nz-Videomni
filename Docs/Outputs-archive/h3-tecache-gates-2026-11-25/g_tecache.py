r"""MiniMax H3 keep_text_conditioning (台帳 §1-298): GPU gates G1 (Single), G2 (Chained, shared reference video, both
caches), G3 (Chained-i2v, no reference). Entry point is MCP (stdio, ``python -m mcp_server``).

    .venv\Scripts\python.exe -X utf8 -B outputs\h3-tecache-gates-2026-11-25\g_tecache.py selfcheck   # CPU, no server
    ... g_tecache.py g1 | g2 | g3      # load (selected weights as they are), upload once, run the jobs one by one
    ... g_tecache.py report            # tables (effect, main memory) from runs\*.json
    ... g_tecache.py cleanup           # MCP: unload_pipeline

The server (127.0.0.1:18620) is started and stopped outside this script. Records: runs\<gate>.json (rerunning a gate
resumes: finished jobs are skipped, a submitted but unfinished job is only waited for, never submitted again).
A gate stops at the first result that differs from the expectation (nothing is investigated or retried).
Paths in the records are relative to the repository.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = HERE / "runs"
PY = REPO / ".venv" / "Scripts" / "python.exe"
FFMPEG = REPO / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE = REPO / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"

TURBO = "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16"
REF2VA = "minimax_h3_ref2va_pruned_int8_convrot"

MAT = {
    "picture": "Docs/Outputs-archive/h3-gate2/i2v_input_square.png",
    "video_short": "outputs/comfort-calib-h3-ref-2026-11-03/materials/H3-Ref-1080p-3s.mp4",
    "audio": "outputs/h3-ref2va-2026-10-28/ref_audio_raincoat_speech.wav",
    "video_15s": "outputs/comfort-calib-h3-ref-2026-11-03/materials/H3-Ref-1080p-15s.mp4",
    "image_x": "images/aux2_001.jpg",
    "image_y": "images/aux2_002.jpg",
}

COMMON = dict(width=1280, height=768, frame_rate=24.0, seed=42, num_inference_steps=4, shift_video=12.0,
              shift_audio=3.0, attention_backend="sdpa", keep_resident=True, embed_mp4_metadata=False,
              loras=[{"name": TURBO, "strength": 0.8}])

# ---- prompts (G1) -------------------------------------------------------------------------------------------
P_A = ("<Picture 1> is the person in this scene. The scene moves like <Video 1>. "
       "The person speaks with the voice from <Audio 1>.")
P_B = ("<Picture 1> is the person in this scene. The scene moves like <Video 1>. "
       "The person speaks softly with the voice from <Audio 1>.")
N_A = "blurry, low quality, distorted faces"
N_B = "flicker, jitter, watermark, text overlay"
# ---- prompts (G2) -------------------------------------------------------------------------------------------
G2_COMMON = "A calm cinematic scene in the look of <Video 1>. Soft light, gentle camera movement."
G2_CLIP_P = ["She walks slowly along the wooden pier.",
             "She stops at the end of the pier and looks at the sea.",
             "A gull glides low over the water beside her.",
             "She turns and smiles as the sun rises."]
G2_P3_CHANGED = "A gull circles high above the harbor behind her."
G2_FRAMES = [73, 124, 73, 73]
# ---- prompts (G3) -------------------------------------------------------------------------------------------
G3_COMMON = "A quiet harbor at dawn. Soft light, gentle camera movement."
G3_CLIP_P = ["Small boats rock gently beside the pier.",
             "Mist drifts slowly over the calm water.",
             "The first sunlight reaches the far shore."]


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(p: str | Path) -> str:
    try:
        return os.path.relpath(str(p), REPO).replace("\\", "/")
    except ValueError:
        return Path(p).name


def ffprobe(path: Path) -> dict:
    r = subprocess.run([str(FFPROBE), "-v", "error", "-count_frames", "-show_entries",
                        "stream=codec_type,width,height,r_frame_rate,nb_read_frames,sample_rate,duration_ts,duration",
                        "-of", "json", str(path)], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path.name}: {r.stderr[:500]}")
    return json.loads(r.stdout)


def stream_md5(path: Path) -> dict:
    out = {}
    for name, sel in (("video", "0:v"), ("audio", "0:a")):
        r = subprocess.run([str(FFMPEG), "-v", "error", "-i", str(path), "-map", sel, "-f", "md5", "-"],
                           capture_output=True, text=True, encoding="utf-8")
        out[name] = r.stdout.strip() if r.returncode == 0 else f"error: {r.stderr[:200]}"
    return out


def find_key_paths(obj, key: str, prefix: str = "") -> list[str]:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            if k == key:
                out.append(p)
            out.extend(find_key_paths(v, key, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(find_key_paths(v, key, f"{prefix}[{i}]"))
    return out


def job_artifacts(job_id: str) -> dict:
    d = REPO / "outputs" / job_id
    meta = read_json(d / "metadata.json")
    mp4 = d / "output.mp4"
    return {"metadata": meta, "mp4_sha256": sha256_file(mp4), "mp4_size": mp4.stat().st_size,
            "mp4_probe": ffprobe(mp4), "mp4_rel": rel(mp4)}


# ---------------------------------------------------------------------------------------------
# request bodies (arguments of submit_generate / submit_chain)
# ---------------------------------------------------------------------------------------------

def g1_args(keep_text: bool, prompt: str, neg: str, ids: dict) -> dict:
    a = dict(COMMON)
    a.update(prompt=prompt, negative_prompt=neg, nag_enabled=True, neg_method="vsf", num_frames=73,
             references=[{"kind": "picture", "id": ids["picture"], "slot": 1},
                         {"kind": "video", "id": ids["video_short"], "slot": 1},
                         {"kind": "audio", "id": ids["audio"], "slot": 1, "max_sec": 12.0}])
    if keep_text:
        a["keep_text_conditioning"] = True
    return a


def g2_args(keep: bool, p3: str, ids: dict) -> dict:
    prompts = list(G2_CLIP_P)
    prompts[2] = p3
    a = dict(COMMON)
    a.update(prompt=G2_COMMON,
             clips=[{"num_frames": n, "prompt": p, "references": [{"kind": "video", "id": ids["video_15s"], "slot": 1}]}
                    for n, p in zip(G2_FRAMES, prompts)])
    if keep:
        a["keep_text_conditioning"] = True
        a["keep_reference_latents"] = True
    return a


def g3_args(keep_text: bool, image_key: str, ids: dict) -> dict:
    a = dict(COMMON)
    clips = [{"num_frames": 73, "prompt": p} for p in G3_CLIP_P]
    clips[0]["conditioning_images"] = [{"image_id": ids[image_key], "frame_idx": 0}]
    a.update(prompt=G3_COMMON, clips=clips)
    if keep_text:
        a["keep_text_conditioning"] = True
    return a


# ---------------------------------------------------------------------------------------------
# job plans.  expect: te = single: {"positive","negative"} | None (OFF);  chain: list per clip of
# {"positive"} (+ "negative") | None.   lat = chain G2: per clip list of reference latent_cache | None (no key)
# eq = (label, what) checks against an earlier label; what in mp4 | latents | csha | csha_pos | csha_neg
# ---------------------------------------------------------------------------------------------

def plan_g1(ids: dict) -> list[dict]:
    mm = {"positive": "miss", "negative": "miss"}
    return [
        dict(label="1", cond="ON, prompt A, negative Na", args=g1_args(True, P_A, N_A, ids), te=mm, eq=[]),
        dict(label="2", cond="OFF, same input as 1", args=g1_args(False, P_A, N_A, ids), te=None, eq=[("1", "mp4")]),
        dict(label="3", cond="ON, same input as 1", args=g1_args(True, P_A, N_A, ids), te=mm,
             eq=[("1", "mp4"), ("1", "csha")]),
        dict(label="4", cond="ON, same input as 1", args=g1_args(True, P_A, N_A, ids),
             te={"positive": "hit", "negative": "hit"}, eq=[("1", "mp4"), ("1", "csha")]),
        dict(label="5", cond="ON, prompt A, negative Nb", args=g1_args(True, P_A, N_B, ids),
             te={"positive": "hit", "negative": "miss"}, eq=[("1", "csha_pos")]),
        dict(label="6", cond="ON, prompt B, negative Nb", args=g1_args(True, P_B, N_B, ids),
             te={"positive": "miss", "negative": "hit"}, eq=[("5", "csha_neg")]),
        dict(label="7", cond="OFF, same input as 5", args=g1_args(False, P_A, N_B, ids), te=None, eq=[("5", "mp4")]),
        dict(label="8", cond="OFF, same input as 6", args=g1_args(False, P_B, N_B, ids), te=None, eq=[("6", "mp4")]),
    ]


def plan_g2(ids: dict) -> list[dict]:
    allm = [{"positive": "miss"}] * 4
    allh = [{"positive": "hit"}] * 4
    return [
        dict(label="1", cond="both OFF", args=g2_args(False, G2_CLIP_P[2], ids), te=None, lat=None, eq=[]),
        dict(label="2", cond="both ON", args=g2_args(True, G2_CLIP_P[2], ids), te=allm,
             lat=[["miss"], ["miss"], ["hit"], ["hit"]], eq=[("1", "mp4"), ("1", "latents")]),
        dict(label="3", cond="both ON, same input as 2", args=g2_args(True, G2_CLIP_P[2], ids), te=allh,
             lat=[["hit"]] * 4, eq=[("1", "mp4"), ("1", "latents"), ("2", "csha")]),
        dict(label="4", cond="both ON, clip 3 prompt changed", args=g2_args(True, G2_P3_CHANGED, ids),
             te=[{"positive": "hit"}, {"positive": "hit"}, {"positive": "miss"}, {"positive": "hit"}],
             lat=[["hit"]] * 4, eq=[]),
        dict(label="5", cond="both OFF, same input as 4", args=g2_args(False, G2_P3_CHANGED, ids), te=None, lat=None,
             eq=[("4", "mp4"), ("4", "latents")]),
    ]


def plan_g3(ids: dict) -> list[dict]:
    allm = [{"positive": "miss"}] * 3
    allh = [{"positive": "hit"}] * 3
    return [
        dict(label="1", cond="ON, image X", args=g3_args(True, "image_x", ids), te=allm, eq=[]),
        dict(label="2", cond="OFF, same input as 1", args=g3_args(False, "image_x", ids), te=None,
             eq=[("1", "mp4"), ("1", "latents")]),
        dict(label="3", cond="ON, same input as 1", args=g3_args(True, "image_x", ids), te=allm,
             eq=[("1", "mp4"), ("1", "latents"), ("1", "csha")]),
        dict(label="4", cond="ON, same input as 1", args=g3_args(True, "image_x", ids), te=allh,
             eq=[("1", "mp4"), ("1", "latents"), ("1", "csha")]),
        dict(label="5", cond="ON, image Y (rest same)", args=g3_args(True, "image_y", ids),
             te=[{"positive": "miss"}, {"positive": "hit"}, {"positive": "hit"}], eq=[]),
        dict(label="6", cond="OFF, same input as 5", args=g3_args(False, "image_y", ids), te=None,
             eq=[("5", "mp4"), ("5", "latents")]),
    ]


GATES = {
    "g1": dict(plan=plan_g1, tool="submit_generate", chain=False,
               uploads=[("picture", "upload_image", "image_id"), ("video_short", "upload_video", "video_id"),
                        ("audio", "upload_audio", "audio_id")]),
    "g2": dict(plan=plan_g2, tool="submit_chain", chain=True,
               uploads=[("video_15s", "upload_video", "video_id")]),
    "g3": dict(plan=plan_g3, tool="submit_chain", chain=True,
               uploads=[("image_x", "upload_image", "image_id"), ("image_y", "upload_image", "image_id")]),
}


# ---------------------------------------------------------------------------------------------
# summaries of one job record
# ---------------------------------------------------------------------------------------------

def _enc_summary(enc: dict) -> dict:
    enc = enc or {}
    return {
        "te_cache": enc.get("te_cache"), "conditioning_sha256": enc.get("conditioning_sha256"),
        "seconds": enc.get("seconds"), "text_tokens": enc.get("text_tokens"),
        "negative_tokens": enc.get("negative_tokens"),
        "rss_process_peak_gib": enc.get("rss_process_peak_gib"), "commit_process_peak_gib": enc.get("commit_process_peak_gib"),
        "rss_end_gib": enc.get("rss_end_gib"), "commit_end_gib": enc.get("commit_end_gib"),
        "te_loaded": "te" in enc,
        "keys": sorted(enc.keys()),
    }


def summarize(art: dict, chain: bool) -> dict:
    meta = art["metadata"]
    h3 = meta.get("h3") or {}
    ph = h3.get("phases") or {}
    out = {"mp4_sha256": art["mp4_sha256"], "mp4_size": art["mp4_size"], "mp4_rel": art["mp4_rel"],
           "backend": meta.get("backend"), "attention_used": meta.get("attention_used"),
           "generation_time_seconds": meta.get("generation_time_seconds"),
           "rss_peak_gib": h3.get("rss_peak_gib"), "commit_peak_gib": h3.get("commit_peak_gib"),
           "latent_cache_paths": find_key_paths(meta, "latent_cache"),
           "te_cache_paths": find_key_paths(meta, "te_cache"),
           "conditioning_sha256_paths": find_key_paths(meta, "conditioning_sha256"),
           "loras": h3.get("loras"), "steps_shift": [h3.get("num_inference_steps"), h3.get("shift_video"), h3.get("shift_audio")],
           "request_keep_text_conditioning": (meta.get("request") or {}).get("keep_text_conditioning"),
           "request_keep_reference_latents": (meta.get("request") or {}).get("keep_reference_latents"),
           "request_embed_mp4_metadata": (meta.get("request") or {}).get("embed_mp4_metadata")}
    if not chain:
        re_ = ph.get("reference_encode") or {}
        out["encode"] = _enc_summary(ph.get("encode"))
        out["reference_encode_seconds"] = re_.get("seconds")
        out["references"] = re_.get("references")
        out["latents_sha256"] = None
    else:
        clips, pclips = h3.get("clips") or [], ph.get("clips") or []
        per = []
        for i in range(max(len(clips), len(pclips))):
            c = clips[i] if i < len(clips) and isinstance(clips[i], dict) else {}
            p = pclips[i] if i < len(pclips) and isinstance(pclips[i], dict) else {}
            re_ = p.get("reference_encode") or {}
            per.append({"encode": _enc_summary(p.get("encode")), "latents_sha256": c.get("latents_sha256"),
                        "clip_seconds": c.get("seconds"), "reference_encode_seconds": re_.get("seconds"),
                        "references": re_.get("references"), "seed": c.get("seed")})
        out["clips"] = per
        out["seeds"] = h3.get("seeds")
    return out


# ---------------------------------------------------------------------------------------------
# expectation / comparison checks
# ---------------------------------------------------------------------------------------------

def _csha(s: dict, chain: bool):
    return [c["encode"]["conditioning_sha256"] for c in s["clips"]] if chain else s["encode"]["conditioning_sha256"]


def check_job(job: dict, s: dict, chain: bool, done: dict) -> list[str]:
    problems = []
    if s["backend"] != "minimax-h3":
        problems.append(f"backend {s['backend']}")
    if s["request_embed_mp4_metadata"] is not False:
        problems.append(f"embed_mp4_metadata {s['request_embed_mp4_metadata']}")
    want_keep = bool(job["args"].get("keep_text_conditioning"))
    if bool(s["request_keep_text_conditioning"]) != want_keep:
        problems.append(f"request.keep_text_conditioning {s['request_keep_text_conditioning']} != {want_keep}")
    encs = [c["encode"] for c in s["clips"]] if chain else [s["encode"]]
    exp = job["te"]
    if exp is None:   # OFF: no key anywhere
        if s["te_cache_paths"] or s["conditioning_sha256_paths"]:
            problems.append(f"OFF job has te_cache/conditioning_sha256: {s['te_cache_paths']} {s['conditioning_sha256_paths']}")
    else:
        exps = exp if chain else [exp]
        if len(encs) != len(exps):
            problems.append(f"{len(encs)} encode records, expected {len(exps)}")
        for i, (e, x) in enumerate(zip(encs, exps)):
            if e["te_cache"] != x:
                problems.append(f"te_cache{'' if not chain else f' clip{i + 1}'}: got {e['te_cache']}, expected {x}")
            if not e["conditioning_sha256"]:
                problems.append(f"conditioning_sha256 missing{'' if not chain else f' clip{i + 1}'}")
    # reference latent cache
    if chain and "lat" in job:
        lat = job["lat"]
        if lat is None:
            if s["latent_cache_paths"]:
                problems.append(f"latent_cache keys present in a both-OFF job: {s['latent_cache_paths']}")
        else:
            for i, c in enumerate(s["clips"]):
                got = [r.get("latent_cache") for r in (c["references"] or [])]
                if got != lat[i]:
                    problems.append(f"latent_cache clip{i + 1}: got {got}, expected {lat[i]}")
    elif s["latent_cache_paths"]:
        problems.append(f"unexpected latent_cache keys: {s['latent_cache_paths']}")
    # comparisons with earlier jobs
    for (ref, what) in job["eq"]:
        o = done.get(ref)
        if o is None:
            problems.append(f"comparison target {ref} missing")
            continue
        os_ = o["summary"]
        if what == "mp4":
            if s["mp4_sha256"] != os_["mp4_sha256"]:
                problems.append(f"mp4 differs from job {ref}: {s['mp4_sha256'][:8]} vs {os_['mp4_sha256'][:8]}")
        elif what == "latents":
            a = [c["latents_sha256"] for c in s["clips"]]
            b = [c["latents_sha256"] for c in os_["clips"]]
            if a != b:
                problems.append(f"latents_sha256 differ from job {ref}: {a} vs {b}")
        elif what == "csha":
            if _csha(s, chain) != _csha(os_, chain):
                problems.append(f"conditioning_sha256 differs from job {ref}")
        elif what == "csha_pos":
            if s["encode"]["conditioning_sha256"]["positive"] != os_["encode"]["conditioning_sha256"]["positive"]:
                problems.append(f"positive hash differs from job {ref}")
        elif what == "csha_neg":
            if s["encode"]["conditioning_sha256"]["negative"] != os_["encode"]["conditioning_sha256"]["negative"]:
                problems.append(f"negative hash differs from job {ref}")
    return problems


# ---------------------------------------------------------------------------------------------
# selfcheck
# ---------------------------------------------------------------------------------------------

def selfcheck() -> int:
    import functools
    import anyio
    import httpx
    sys.path.insert(0, str(REPO))
    from api.models import GenerateChainRequest, GenerateRequest
    from mcp_server.client import BackendClient, set_client
    from mcp_server.params import ChainClipArg, ConditioningImageArg, LoraArg, ReferenceArg
    from mcp_server.settings import Settings
    from mcp_server.tools import generate

    problems: list[str] = []
    mats = {}
    for k, v in MAT.items():
        p = REPO / v
        info = {"path": v, "exists": p.is_file()}
        if p.is_file():
            info["size_bytes"] = p.stat().st_size
            info["sha256"] = sha256_file(p)
            info["streams"] = ffprobe(p).get("streams")
        else:
            problems.append(f"missing {k}")
        mats[k] = info
    if not problems:
        v15 = next(s for s in mats["video_15s"]["streams"] if s["codec_type"] == "video")
        fr = int(v15["nb_read_frames"])
        if not (fr > 124 and v15["r_frame_rate"] == "24/1"):
            problems.append(f"video_15s frames {fr} rate {v15['r_frame_rate']}")
        if mats["image_x"]["sha256"] == mats["image_y"]["sha256"]:
            problems.append("image X and Y are the same file")
        vs = next(s for s in mats["video_short"]["streams"] if s["codec_type"] == "video")
        if int(vs["nb_read_frames"]) != 73:
            problems.append(f"video_short frames {vs['nb_read_frames']}")
    ids = {k: f"id-{k}" for k in MAT}
    cap: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        cap.append((request.url.path, json.loads(request.content)))
        return httpx.Response(202, json={"job_id": "j1", "status": "queued", "created_at": "t", "num_clips": 3})

    set_client(BackendClient(Settings(base_url="http://testserver", api_key=None, output_dir=Path(".")),
                             transport=httpx.MockTransport(handler)))

    def conv(args: dict) -> dict:
        a = dict(args)
        a["loras"] = [LoraArg(**x) for x in a["loras"]]
        if "references" in a:
            a["references"] = [ReferenceArg(**x) for x in a["references"]]
        if "clips" in a:
            cl = []
            for c in a["clips"]:
                c = dict(c)
                if "references" in c:
                    c["references"] = [ReferenceArg(**x) for x in c["references"]]
                if "conditioning_images" in c:
                    c["conditioning_images"] = [ConditioningImageArg(**x) for x in c["conditioning_images"]]
                cl.append(ChainClipArg(**c))
            a["clips"] = cl
        return a

    bodies = {}
    for g, spec in GATES.items():
        for job in spec["plan"](ids):
            n0 = len(cap)
            fn = getattr(generate, spec["tool"])
            anyio.run(functools.partial(fn, **conv(job["args"])))
            path, body = cap[n0]
            model = GenerateChainRequest if spec["chain"] else GenerateRequest
            try:
                req = model.model_validate(body)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{g}-{job['label']}: {type(exc).__name__}: {str(exc)[:300]}")
                continue
            want = bool(job["args"].get("keep_text_conditioning"))
            if bool(body.get("keep_text_conditioning")) != want or req.keep_text_conditioning != want:
                problems.append(f"{g}-{job['label']}: keep_text_conditioning body={body.get('keep_text_conditioning')}")
            for k, w in (("num_inference_steps", 4), ("shift_video", 12), ("shift_audio", 3), ("seed", 42),
                         ("width", 1280), ("height", 768)):
                if body.get(k) != w:
                    problems.append(f"{g}-{job['label']}: {k}={body.get(k)}")
            if "keep_alive" in body:
                pass
            bodies[f"{g}-{job['label']}"] = {"path": path, "body": body}
    if not (REPO / "models").is_dir() or not sorted((REPO / "models").rglob(f"{TURBO}.*")):
        problems.append("turbo lora file not found under models")
    rec = {"at": now(), "materials": mats, "bodies": bodies, "problems": problems, "ok": not problems}
    write_json(RUNS / "selfcheck.json", rec)
    print(json.dumps({"problems": problems, "ok": rec["ok"], "n_bodies": len(bodies)}, ensure_ascii=False, indent=1))
    return 0 if rec["ok"] else 2


# ---------------------------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------------------------

class Mcp:
    def __init__(self, s, calls):
        self.s, self.calls = s, calls

    async def call(self, name, args=None):
        t0 = time.monotonic()
        r = await self.s.call_tool(name, args or {}, read_timeout_seconds=timedelta(seconds=360))
        text = "\n".join(getattr(c, "text", "") for c in (r.content or []) if getattr(c, "text", None))
        e = {"tool": name, "args": args or {}, "is_error": bool(r.isError),
             "seconds": round(time.monotonic() - t0, 2), "at": now()}
        if r.isError:
            e["error_text"] = text
            self.calls.append(e)
            raise RuntimeError(f"{name}: {text}")
        data = r.structuredContent if isinstance(r.structuredContent, dict) else (json.loads(text) if text else {})
        if name not in ("list_models", "wait_for_job", "list_loras"):
            e["result"] = data
        self.calls.append(e)
        return data


async def with_mcp(fn, tag):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    params = StdioServerParameters(command=str(PY), args=["-m", "mcp_server"], cwd=str(REPO),
                                   env={**os.environ, "PYTHONUTF8": "1"}, encoding="utf-8")
    (HERE / "logs").mkdir(exist_ok=True)
    calls: list = []
    with open(HERE / "logs" / "mcp.stderr.log", "a", encoding="utf-8") as err:
        async with stdio_client(params, errlog=err) as (rd, wr):
            async with ClientSession(rd, wr, read_timeout_seconds=timedelta(seconds=360)) as s:
                await s.initialize()
                try:
                    return await fn(Mcp(s, calls))
                finally:
                    write_json(RUNS / f"mcp_calls_{tag}_{int(time.time())}.json", calls)


async def wait_done(m: Mcp, job_id: str, unreach: dict, limit: float = 14400) -> dict:
    t0 = time.monotonic()
    consecutive = 0
    while True:
        try:
            st = await m.call("wait_for_job", {"job_id": job_id, "timeout_sec": 45, "poll_interval_sec": 5})
            consecutive = 0
        except RuntimeError as exc:
            if "BACKEND_UNREACHABLE" not in str(exc) or consecutive >= 30:
                raise
            consecutive += 1
            unreach["count"] += 1
            unreach["at"].append(now())
            print(f"  BACKEND_UNREACHABLE ({consecutive}) {now()}", flush=True)
            await asyncio.sleep(10)
            continue
        if not st.get("timed_out"):
            return st
        print(f"  {job_id[:8]} {st.get('status')} stage={st.get('stage')} clip={st.get('clip')} "
              f"step={st.get('current_step')}/{st.get('total_steps')} t={time.monotonic() - t0:.0f}s", flush=True)
        if time.monotonic() - t0 > limit:
            st["wait_limit_exceeded"] = True
            return st


async def wait_loaded(m: Mcp, limit: float = 1500) -> dict:
    t0 = time.monotonic()
    while True:
        st = await m.call("backend_status")
        s = st.get("status") or {}
        print(f"  load: state={s.get('state')} loaded={s.get('pipeline_loaded')} t={time.monotonic() - t0:.0f}s", flush=True)
        if s.get("pipeline_loaded") and s.get("state") in ("ready", "idle"):
            lm = await m.call("list_models")
            tr = (lm.get("categories") or {}).get("transformer") or {}
            return {"status": s, "active_base_model": lm.get("active_base_model"), "transformer_active": tr.get("active")}
        if s.get("state") == "error" or time.monotonic() - t0 > limit:
            raise RuntimeError(f"pipeline did not load: {s}")
        await asyncio.sleep(10)


def phase_gate(g: str) -> int:
    spec = GATES[g]
    out = RUNS / f"{g}.json"
    rec: dict = read_json(out) if out.is_file() else {"gate": g, "started": now(), "jobs": {}, "unreachable": {"count": 0, "at": []}}
    rec.setdefault("resumes", []).append(now())
    rec.pop("stopped", None)
    unreach = rec["unreachable"]
    chain = spec["chain"]

    async def body(m: Mcp):
        if "loaded" not in rec:
            rec["backend_status"] = await m.call("backend_status")
            lm = await m.call("list_models")
            tr = (lm.get("categories") or {}).get("transformer") or {}
            rec["models_before"] = {"active_base_model": lm.get("active_base_model"), "transformer_active": tr.get("active")}
            if lm.get("active_base_model") != "MiniMaxH3":
                raise RuntimeError(f"active base model is {lm.get('active_base_model')}: stop")
            ll = await m.call("list_loras")
            if TURBO not in json.dumps(ll, ensure_ascii=False):
                raise RuntimeError("turbo lora not in list_loras")
            rec["load_call"] = await m.call("load_pipeline", {"wait_sec": 45})   # selected weights as they are
            rec["loaded"] = await wait_loaded(m)
            write_json(out, rec)
        if "upload_ids" not in rec:
            ids, ups = {}, {}
            for name, tool, key in spec["uploads"]:
                r = await m.call(tool, {"file_path": str(REPO / MAT[name])})
                ids[name] = r[key]
                ups[name] = {k: v for k, v in r.items() if k != "stored_path"}
                ups[name]["source_rel"] = MAT[name]
                ups[name]["stored_path_rel"] = rel(r["stored_path"]) if r.get("stored_path") else None
            rec["upload_ids"], rec["uploads"] = ids, ups
            write_json(out, rec)
        ids = rec["upload_ids"]
        plan = spec["plan"](ids)
        for job in plan:
            label = job["label"]
            ent = rec["jobs"].setdefault(label, {})
            if ent.get("summary") and ent.get("checked_ok"):
                continue
            ent.update(label=label, cond=job["cond"], args=job["args"])
            t0 = time.monotonic()
            if not ent.get("job_id"):
                sub = await m.call(spec["tool"], job["args"])
                ent["job_id"] = sub["job_id"]
                ent["submitted_at"] = now()
                write_json(out, rec)
                print(f"{g}-{label}: job {ent['job_id']} ({job['cond']})", flush=True)
            jid = ent["job_id"]
            fin = await wait_done(m, jid, unreach)
            ent["final"] = {k: fin.get(k) for k in ("status", "error", "timed_out", "wait_limit_exceeded")}
            ent["wall_seconds"] = round(time.monotonic() - t0, 1)
            write_json(out, rec)
            if fin.get("status") != "completed":
                rec["stopped"] = {"label": label, "problems": [f"job ended {fin.get('status')}: {fin.get('error')}"]}
                write_json(out, rec)
                return
            art = job_artifacts(jid)
            write_json(RUNS / f"{g}_{label}.metadata.json", art["metadata"])
            ent["summary"] = summarize(art, chain)
            ent["mp4_probe"] = art["mp4_probe"]
            done = {k: v for k, v in rec["jobs"].items() if v.get("summary") and v.get("checked_ok")}
            problems = check_job(job, ent["summary"], chain, done)
            ent["problems"] = problems
            ent["checked_ok"] = not problems
            sm = ent["summary"]
            if chain:
                te = [c["encode"]["te_cache"] for c in sm["clips"]]
                lc = [[r.get("latent_cache") for r in (c["references"] or [])] for c in sm["clips"]]
            else:
                te, lc = sm["encode"]["te_cache"], [r.get("latent_cache") for r in (sm["references"] or [])]
            print(f"{g}-{label}: mp4 {sm['mp4_sha256'][:8]} te={te} lat={lc} gen={sm['generation_time_seconds']} "
                  f"problems={problems}", flush=True)
            write_json(out, rec)
            if problems:
                # fact observation allowed by the brief: stream hashes of the two mp4 when an mp4 differs
                for p in problems:
                    if p.startswith("mp4 differs from job "):
                        other = p.split("job ")[1].split(":")[0]
                        a, b = REPO / sm["mp4_rel"], REPO / rec["jobs"][other]["summary"]["mp4_rel"]
                        ent["stream_md5_observation"] = {"this": stream_md5(a), other: stream_md5(b)}
                rec["stopped"] = {"label": label, "problems": problems}
                write_json(out, rec)
                return

    try:
        asyncio.run(with_mcp(body, g))
        rec["ok_run"] = True
    except BaseException as exc:  # noqa: BLE001
        rec["ok_run"] = False
        subs = getattr(exc, "exceptions", None)
        rec["error"] = f"{type(exc).__name__}: {exc}" + (f" | {[repr(e) for e in subs]}" if subs else "")
        print("run error:", rec["error"], flush=True)
    rec["finished"] = now()
    all_ok = (not rec.get("stopped") and rec.get("ok_run") and
              all((rec["jobs"].get(j["label"]) or {}).get("checked_ok") for j in GATES[g]["plan"](rec.get("upload_ids") or {})))
    rec["gate_pass"] = bool(all_ok)
    write_json(out, rec)
    write_json(RUNS / f"verdict_{g}.json", {
        "gate": g, "pass": bool(all_ok), "stopped": rec.get("stopped"), "error": rec.get("error"),
        "jobs": {k: {"job_id": v.get("job_id"), "cond": v.get("cond"), "checked_ok": v.get("checked_ok"),
                     "problems": v.get("problems")} for k, v in rec["jobs"].items()},
        "unreachable": rec["unreachable"]})
    print("GATE", g, "PASS" if all_ok else "NOT PASSED", flush=True)
    return 0 if all_ok else 1


def phase_cleanup() -> int:
    rec: dict = {"started": now()}

    async def body(m: Mcp):
        rec["unload"] = await m.call("unload_pipeline")
        rec["status_after"] = await m.call("backend_status")

    try:
        asyncio.run(with_mcp(body, "cleanup"))
        rec["ok"] = ((rec.get("status_after") or {}).get("status") or {}).get("state") == "unloaded"
    except BaseException as exc:  # noqa: BLE001
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
    write_json(RUNS / "cleanup.json", rec)
    print("cleanup", rec.get("ok"), rec.get("error"))
    return 0 if rec["ok"] else 1


# ---------------------------------------------------------------------------------------------
# report (effect table, main-memory table, cache sizes)
# ---------------------------------------------------------------------------------------------

POS_BYTES, NEG_BYTES = 10248, 10240


def cache_bytes(s: dict, chain: bool, on: bool) -> dict:
    if not on:
        return {"entries": 0, "bytes": 0}
    encs = [c["encode"] for c in s["clips"]] if chain else [s["encode"]]
    n, b = 0, 0
    for e in encs:
        if e["text_tokens"] is not None:
            n += 1
            b += int(e["text_tokens"]) * POS_BYTES
        if e["negative_tokens"] is not None:
            n += 1
            b += int(e["negative_tokens"]) * NEG_BYTES
    return {"entries": n, "bytes": b}


def phase_report() -> int:
    rep: dict = {"gates": {}}
    for g in ("g1", "g2", "g3"):
        p = RUNS / f"{g}.json"
        if not p.is_file():
            continue
        r = read_json(p)
        chain = GATES[g]["chain"]
        rows = []
        for label, j in sorted(r["jobs"].items()):
            s = j.get("summary")
            if not s:
                continue
            on = bool(j["args"].get("keep_text_conditioning"))
            if chain:
                encs = [{"enc_seconds": c["encode"]["seconds"], "te_cache": c["encode"]["te_cache"],
                         "tokens": c["encode"]["text_tokens"], "neg_tokens": c["encode"]["negative_tokens"],
                         "latent_cache": [x.get("latent_cache") for x in (c["references"] or [])],
                         "latents_sha8": {k: (v or "")[:8] for k, v in (c["latents_sha256"] or {}).items()},
                         "csha8": {k: (v or "")[:8] for k, v in (c["encode"]["conditioning_sha256"] or {}).items()},
                         "rss_peak": c["encode"]["rss_process_peak_gib"], "commit_peak": c["encode"]["commit_process_peak_gib"]}
                        for c in s["clips"]]
            else:
                e = s["encode"]
                encs = [{"enc_seconds": e["seconds"], "te_cache": e["te_cache"], "tokens": e["text_tokens"],
                         "neg_tokens": e["negative_tokens"],
                         "latent_cache": [x.get("latent_cache") for x in (s["references"] or [])],
                         "csha8": {k: (v or "")[:8] for k, v in (e["conditioning_sha256"] or {}).items()},
                         "rss_peak": e["rss_process_peak_gib"], "commit_peak": e["commit_process_peak_gib"]}]
            rows.append({"label": label, "cond": j["cond"], "job_id": j["job_id"], "on": on,
                         "mp4_sha8": s["mp4_sha256"][:8], "generation_time_seconds": s["generation_time_seconds"],
                         "wall_seconds": j.get("wall_seconds"), "rss_peak_gib": s["rss_peak_gib"],
                         "commit_peak_gib": s["commit_peak_gib"], "clips": encs,
                         "cache_held": cache_bytes(s, chain, on), "problems": j.get("problems")})
        rep["gates"][g] = {"rows": rows, "unreachable": r["unreachable"], "stopped": r.get("stopped"),
                           "upload_ids": r.get("upload_ids"), "loaded": (r.get("loaded") or {}).get("transformer_active")}
    write_json(RUNS / "report.json", rep)
    for g, d in rep["gates"].items():
        print("==", g, "unreachable", d["unreachable"]["count"], "stopped", d["stopped"], "weights", d["loaded"])
        for r in d["rows"]:
            print(f"  [{r['label']}] {r['cond']} | job {r['job_id']} | mp4 {r['mp4_sha8']} | gen {r['generation_time_seconds']}"
                  f" | rss {r['rss_peak_gib']} commit {r['commit_peak_gib']} | cache {r['cache_held']}")
            for i, c in enumerate(r["clips"]):
                print(f"      c{i + 1}: te={c['te_cache']} lat={c['latent_cache']} enc={c['enc_seconds']}s tok={c['tokens']}/"
                      f"{c['neg_tokens']} csha={c['csha8']} lsha={c.get('latents_sha8')} peak={c['rss_peak']}/{c['commit_peak']}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    c = a[0]
    if c == "selfcheck":
        sys.exit(selfcheck())
    if c in GATES:
        sys.exit(phase_gate(c))
    if c == "report":
        sys.exit(phase_report())
    if c == "cleanup":
        sys.exit(phase_cleanup())
    raise SystemExit("unknown")
