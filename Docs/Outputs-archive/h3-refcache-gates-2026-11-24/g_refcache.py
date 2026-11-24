r"""MiniMax H3 keep_reference_latents (台帳 §1-297, VERIFICATION_LOG §233.9): the GPU gates G1 (Single, MCP) and
G2 (Chained, HTTP). Reused latents must give the same bits as encoding again.

    .venv\Scripts\python.exe -X utf8 -B outputs\h3-refcache-gates-2026-11-24\g_refcache.py --selfcheck   # CPU, no server
    ... g_refcache.py g1        # MCP: status, list_models/list_loras, load Ref2VA, uploads (once), G1 (1) (2) (3)
    ... g_refcache.py g2        # HTTP POST /generate/chain: G2 on, then off (only when g1 passed)
    ... g_refcache.py verdict   # verdict.json from the run records
    ... g_refcache.py cleanup   # MCP: load_pipeline(transformer=default), then unload

The server (127.0.0.1:18620) is started and stopped by the operator, not here. Records: runs\<name>.json.
The MCP server runs over stdio as in VERIFICATION_LOG §39.5 / §39.6 (``.venv\Scripts\python.exe -m mcp_server``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import timedelta
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = HERE / "runs"
BASE = "http://127.0.0.1:18620/api/v1"
PY = REPO / ".venv" / "Scripts" / "python.exe"
FFPROBE = REPO / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"

REF2VA = "minimax_h3_ref2va_pruned_int8_convrot"
TURBO = "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16"

MAT = {
    "picture": "Docs/Outputs-archive/h3-gate2/i2v_input_square.png",
    "video_short": "outputs/comfort-calib-h3-ref-2026-11-03/materials/H3-Ref-1080p-3s.mp4",
    "audio": "outputs/h3-ref2va-2026-10-28/ref_audio_raincoat_speech.wav",
    "video_long": "Docs/Outputs-archive/h3-vsf-2026-11-09/g0_off/g0_off.mp4",
}

PROMPT_G1 = ("<Picture 1> is the person in this scene. The scene moves like <Video 1>. "
             "The person speaks with the voice from <Audio 1>.")
PROMPT_G2 = ("A calm cinematic scene in the look of <Video 1>, following the motion of <Video 2>. "
             "Soft light, gentle camera movement.")

COMMON = dict(frame_rate=24.0, seed=42, num_inference_steps=4, shift_video=12.0, shift_audio=3.0,
              attention_backend="sdpa", keep_resident=True, embed_mp4_metadata=False,
              loras=[{"name": TURBO, "strength": 0.8}])

G1_RUNS = [  # (label, width, height, keep_reference_latents, audio max_sec)
    ("g1_1", 960, 576, True, 8.0),
    ("g1_2", 1280, 768, True, 12.0),
    ("g1_3", 1280, 768, False, 12.0),
]
G2_CLIPS = [124, 90, 141, 107]
FPS = 24
SR = 32000
REF_MAX_SEC = 15.0


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


def ffprobe(path: Path) -> dict:
    r = subprocess.run([str(FFPROBE), "-v", "error", "-count_frames", "-show_entries",
                        "stream=codec_type,width,height,r_frame_rate,nb_read_frames,sample_rate,duration_ts,duration",
                        "-of", "json", str(path)], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path.name}: {r.stderr[:500]}")
    return json.loads(r.stdout)


def http(method: str, path: str, body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            t = r.read().decode()
            return r.status, (json.loads(t) if t else None)
    except urllib.error.HTTPError as e:
        t = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(t)
        except Exception:  # noqa: BLE001
            return e.code, {"raw": t[:2000]}


def find_key_paths(obj, key: str, prefix: str = "") -> list[str]:
    """Every path under ``obj`` where a dict has ``key`` (to report where the records are)."""
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
            "mp4_probe": ffprobe(mp4)}


# ---------------------------------------------------------------------------------------------
# expectations (selfcheck)
# ---------------------------------------------------------------------------------------------

def probe_materials() -> dict:
    out = {}
    for name, rel in MAT.items():
        p = REPO / rel
        info = {"path": rel, "exists": p.is_file()}
        if p.is_file():
            info["size_bytes"] = p.stat().st_size
            info["sha256"] = sha256_file(p)
            streams = ffprobe(p).get("streams") or []
            info["streams"] = streams
        out[name] = info
    return out


def expectations(mats: dict) -> dict:
    def vstream(name):
        return next(s for s in mats[name]["streams"] if s.get("codec_type") == "video")

    pic = vstream("picture")
    pic_size = [int(pic["width"]), int(pic["height"])]
    short_frames = int(vstream("video_short")["nb_read_frames"])
    long_frames = int(vstream("video_long")["nb_read_frames"])
    a = next(s for s in mats["audio"]["streams"] if s.get("codec_type") == "audio")
    audio_samples = int(a["duration_ts"]) if int(a["sample_rate"]) == SR else None
    cap = int(REF_MAX_SEC * FPS)

    def used(src, clip_frames=None):
        n = min(src, cap)
        return n if clip_frames is None else min(n, clip_frames)

    def hm(extents):
        seen, out = set(), []
        for e in extents:
            out.append("hit" if e in seen else "miss")
            seen.add(e)
        return out

    short_used = [used(short_frames, c) for c in G2_CLIPS]
    long_used = [used(long_frames, c) for c in G2_CLIPS]
    return {
        "picture_size": pic_size,   # smaller than both outputs: no resize
        "video_short_frames": short_frames, "video_long_frames": long_frames,
        "audio_sample_rate": int(a["sample_rate"]), "audio_samples": audio_samples,
        "audio_seconds": audio_samples / SR if audio_samples else None,
        "g1": {"g1_1": {"picture": "miss", "video": "miss", "audio": "miss"},
               "g1_2": {"picture": "hit", "video": "hit", "audio": "hit"},
               "g1_3": None,
               "video_used_frames": used(short_frames)},
        "g2": {"short_used": short_used, "short_cache": hm(short_used),
               "long_used": long_used, "long_cache": hm(long_used)},
    }


def selfcheck() -> int:
    mats = probe_materials()
    rec = {"phase": "selfcheck", "at": now(), "materials": mats}
    problems = [f"missing {k}" for k, v in mats.items() if not v["exists"]]
    if not problems:
        exp = expectations(mats)
        rec["expected"] = exp
        want = {"picture_size": [384, 384], "video_short_frames": 73, "video_long_frames": 124, "audio_samples": 165888}
        for k, v in want.items():
            if exp[k] != v:
                problems.append(f"{k}: got {exp[k]}, brief says {v}")
        if exp["g2"]["short_cache"] != ["miss", "hit", "hit", "hit"]:
            problems.append(f"g2 short cache {exp['g2']['short_cache']}")
        if exp["g2"]["long_cache"] != ["miss", "miss", "hit", "miss"]:
            problems.append(f"g2 long cache {exp['g2']['long_cache']}")
        rec["g1_bodies_example"] = g1_body("g1_2", 1280, 768, True, 12.0,
                                           {"picture": "<img>", "video_short": "<vid>", "audio": "<aud>"})
        rec["g2_body_example"] = g2_body(True, {"video_short": "<vid>", "video_long": "<vid2>"})
    rec["problems"] = problems
    rec["ok"] = not problems
    write_json(RUNS / "selfcheck.json", rec)
    print(json.dumps({k: rec.get(k) for k in ("expected", "problems", "ok")}, ensure_ascii=False, indent=1))
    return 0 if rec["ok"] else 2


# ---------------------------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------------------------

def g1_body(label, w, h, keep, audio_sec, ids) -> dict:
    body = dict(prompt=PROMPT_G1, width=w, height=h, num_frames=73, **COMMON)
    body["references"] = [
        {"kind": "picture", "id": ids["picture"], "slot": 1},
        {"kind": "video", "id": ids["video_short"], "slot": 1},
        {"kind": "audio", "id": ids["audio"], "slot": 1, "max_sec": audio_sec},
    ]
    if keep:
        body["keep_reference_latents"] = True
    return body


def g2_body(keep: bool, ids) -> dict:
    refs = [{"kind": "video", "id": ids["video_short"], "slot": 1},
            {"kind": "video", "id": ids["video_long"], "slot": 2}]
    body = dict(prompt=PROMPT_G2, width=1280, height=768, **COMMON,
                clips=[{"num_frames": n, "references": [dict(r) for r in refs]} for n in G2_CLIPS],
                keep_reference_latents=keep)
    return body


# ---------------------------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------------------------

class Mcp:
    def __init__(self, session, calls):
        self.s, self.calls = session, calls

    async def call(self, name: str, args: dict | None = None) -> dict:
        t0 = time.monotonic()
        r = await self.s.call_tool(name, args or {}, read_timeout_seconds=timedelta(seconds=360))
        text = "\n".join(getattr(c, "text", "") for c in (r.content or []) if getattr(c, "text", None))
        entry = {"tool": name, "args": args or {}, "is_error": bool(r.isError), "seconds": round(time.monotonic() - t0, 2)}
        if r.isError:
            entry["error_text"] = text
            self.calls.append(entry)
            raise RuntimeError(f"{name}: {text}")
        data = r.structuredContent if isinstance(r.structuredContent, dict) else (json.loads(text) if text else {})
        if name not in ("list_models", "wait_for_job", "list_loras"):
            entry["result"] = data
        self.calls.append(entry)
        return data


async def with_mcp(fn):
    from mcp import ClientSession, StdioServerParameters  # noqa: PLC0415
    from mcp.client.stdio import stdio_client  # noqa: PLC0415

    params = StdioServerParameters(command=str(PY), args=["-m", "mcp_server"], cwd=str(REPO),
                                   env={**os.environ, "PYTHONUTF8": "1"}, encoding="utf-8")
    (HERE / "logs").mkdir(exist_ok=True)
    calls: list = []
    with open(HERE / "logs" / "mcp.stderr.log", "a", encoding="utf-8") as err:
        async with stdio_client(params, errlog=err) as (rd, wr):
            async with ClientSession(rd, wr, read_timeout_seconds=timedelta(seconds=360)) as s:
                await s.initialize()
                m = Mcp(s, calls)
                try:
                    return await fn(m)
                finally:
                    write_json(RUNS / f"mcp_calls_{int(time.time())}.json", calls)


async def wait_done(m: Mcp, job_id: str, limit: float = 7200) -> dict:
    t0 = time.monotonic()
    unreachable = 0
    while True:
        try:
            st = await m.call("wait_for_job", {"job_id": job_id, "timeout_sec": 45, "poll_interval_sec": 5})
            unreachable = 0
        except RuntimeError as exc:
            # The first run of 2026-11-24 saw one transient BACKEND_UNREACHABLE while the job kept running:
            # wait again (the job is not touched), give up after 30 in a row.
            if "BACKEND_UNREACHABLE" not in str(exc) or unreachable >= 30:
                raise
            unreachable += 1
            print(f"  {job_id[:8]} wait_for_job: BACKEND_UNREACHABLE ({unreachable}), waiting again", flush=True)
            await asyncio.sleep(10)
            continue
        if not st.get("timed_out"):
            return st
        print(f"  {job_id[:8]} {st.get('status')} stage={st.get('stage')} step={st.get('current_step')}"
              f"/{st.get('total_steps')} t={time.monotonic() - t0:.0f}s", flush=True)
        if time.monotonic() - t0 > limit:
            return st


async def wait_loaded(m: Mcp, transformer: str, limit: float = 1200) -> dict:
    t0 = time.monotonic()
    while True:
        st = await m.call("backend_status")
        s = st.get("status") or {}
        print(f"  load: state={s.get('state')} loaded={s.get('pipeline_loaded')} t={time.monotonic() - t0:.0f}s", flush=True)
        if s.get("pipeline_loaded") and s.get("state") in ("ready", "idle"):
            lm = await m.call("list_models")
            tr = (lm.get("categories") or {}).get("transformer") or {}
            return {"status": s, "transformer_active": tr.get("active"),
                    "h3": next((b for b in lm.get("base_models") or [] if b.get("id") == "MiniMaxH3"), None)}
        if s.get("state") == "error" or time.monotonic() - t0 > limit:
            raise RuntimeError(f"pipeline did not load: {s}")
        await asyncio.sleep(10)


def summarize_single(art: dict) -> dict:
    meta = art["metadata"]
    h3 = meta.get("h3") or {}
    ph = h3.get("phases") or {}
    re_ = ph.get("reference_encode") or {}
    return {
        "mp4_sha256": art["mp4_sha256"], "mp4_size": art["mp4_size"],
        "backend": meta.get("backend"), "attention_used": meta.get("attention_used"),
        "generation_time_seconds": meta.get("generation_time_seconds"),
        "reference_encode_seconds": re_.get("seconds"),
        "references": re_.get("references"),
        "sequence": (ph.get("denoise") or {}).get("sequence"),
        "latent_cache_paths": find_key_paths(meta, "latent_cache"),
        "request_embed_mp4_metadata": (meta.get("request") or {}).get("embed_mp4_metadata"),
        "request_keep_reference_latents": (meta.get("request") or {}).get("keep_reference_latents"),
        "request_references": (meta.get("request") or {}).get("references"),
        "loras": h3.get("loras"),
        "steps_shift": [h3.get("num_inference_steps"), h3.get("shift_video"), h3.get("shift_audio")],
    }


def phase_g1(resume: dict | None = None) -> int:
    """``resume``: {label: job_id} of jobs already submitted through submit_generate by the first attempt (the
    MCP client fell over while waiting; the job kept running). Those are waited for and read, not submitted
    again; the load and the uploads of the first attempt are reused (runs/g1.first_attempt.json)."""
    out = RUNS / "g1.json"
    rec: dict = {"phase": "g1", "started": now()}
    if resume:
        first = read_json(out)
        write_json(RUNS / "g1.first_attempt.json", first)
        rec.update({k: first.get(k) for k in ("backend_status", "models_before", "loras_listed", "load_call", "loaded",
                                              "upload_ids", "uploads")})
        rec["resumed_jobs"] = resume
        rec["first_attempt_error"] = first.get("error")

    async def body(m: Mcp):
        if resume:
            st = await m.call("backend_status")
            rec["backend_status_at_resume"] = st
            await run_all(m, rec["upload_ids"])
            return
        await prepare(m)
        await run_all(m, rec["upload_ids"])

    async def prepare(m: Mcp):
        st = await m.call("backend_status")
        rec["backend_status"] = st
        lm = await m.call("list_models")
        tr = (lm.get("categories") or {}).get("transformer") or {}
        rec["models_before"] = {"active_base_model": lm.get("active_base_model"), "transformer_active": tr.get("active"),
                                "transformer_names": [e.get("name") for e in tr.get("entries") or []]}
        ll = await m.call("list_loras")
        rec["loras_listed"] = ll
        names = json.dumps(ll, ensure_ascii=False)
        if REF2VA not in rec["models_before"]["transformer_names"]:
            raise RuntimeError(f"{REF2VA} is not listed: {rec['models_before']['transformer_names']}")
        if TURBO not in names:
            raise RuntimeError(f"{TURBO} is not in list_loras")
        rec["load_call"] = await m.call("load_pipeline", {"models": {"transformer": REF2VA}, "wait_sec": 45})
        rec["loaded"] = await wait_loaded(m, REF2VA)
        if rec["loaded"]["transformer_active"] != REF2VA:
            raise RuntimeError(f"active transformer is {rec['loaded']['transformer_active']}")
        ids = {}
        ups = {}
        for name, tool, key in (("picture", "upload_image", "image_id"), ("video_short", "upload_video", "video_id"),
                                ("audio", "upload_audio", "audio_id"), ("video_long", "upload_video", "video_id")):
            r = await m.call(tool, {"file_path": str(REPO / MAT[name])})
            ids[name] = r[key]
            ups[name] = {k: v for k, v in r.items() if k != "stored_path"}
            ups[name]["stored_path_rel"] = os.path.relpath(r.get("stored_path", ""), REPO) if r.get("stored_path") else None
        rec["upload_ids"] = ids
        rec["uploads"] = ups
        write_json(RUNS / "uploads.json", {"ids": ids, "uploads": ups})

    async def run_all(m: Mcp, ids: dict):
        rec["runs"] = {}
        for label, w, h, keep, asec in G1_RUNS:
            args = g1_body(label, w, h, keep, asec, ids)
            r: dict = {"args": args, "submitted_at": now()}
            rec["runs"][label] = r
            t0 = time.monotonic()
            if resume and label in resume:
                jid = resume[label]
                r["resumed"] = True
                r["submitted_at"] = "first attempt (see g1.first_attempt.json)"
            else:
                sub = await m.call("submit_generate", args)
                jid = sub["job_id"]
            r["job_id"] = jid
            print(f"{label}: job {jid}", flush=True)
            write_json(out, rec)
            fin = await wait_done(m, jid)
            r["final"] = {k: fin.get(k) for k in ("status", "error", "timed_out")}
            r["wall_seconds"] = round(time.monotonic() - t0, 1)
            if fin.get("status") != "completed":
                write_json(out, rec)
                raise RuntimeError(f"{label} ended {fin.get('status')}: {fin.get('error')}")
            art = job_artifacts(jid)
            write_json(RUNS / f"{label}.metadata.json", art["metadata"])
            r["summary"] = summarize_single(art)
            r["mp4_probe"] = art["mp4_probe"]
            print(f"{label}: {r['summary']['mp4_sha256']} refs={[(x.get('kind'), x.get('latent_cache')) for x in r['summary']['references'] or []]}"
                  f" gen={r['summary']['generation_time_seconds']}", flush=True)
            write_json(out, rec)

    try:
        asyncio.run(with_mcp(body))
        rec["ok_run"] = True
    except Exception as exc:  # noqa: BLE001
        rec["ok_run"] = False
        subs = getattr(exc, "exceptions", None)
        rec["error"] = f"{type(exc).__name__}: {exc}" + (f" | {[repr(e) for e in subs]}" if subs else "")
        print("g1 error:", rec["error"], flush=True)
    rec["finished"] = now()
    write_json(out, rec)
    return 0 if rec["ok_run"] else 1


# ---------------------------------------------------------------------------------------------
# G2 (HTTP)
# ---------------------------------------------------------------------------------------------

def summarize_chain(art: dict) -> dict:
    meta = art["metadata"]
    h3 = meta.get("h3") or {}
    ph = h3.get("phases") or {}
    clips = h3.get("clips") or []
    pclips = ph.get("clips") or []
    per = []
    for i in range(max(len(clips), len(pclips))):
        c = clips[i] if i < len(clips) and isinstance(clips[i], dict) else {}
        p = pclips[i] if i < len(pclips) and isinstance(pclips[i], dict) else {}
        refs = (p.get("reference_encode") or {}).get("references")
        per.append({
            "latents_sha256": c.get("latents_sha256"),
            "seconds": c.get("seconds"),
            "reference_encode_seconds_phase": (p.get("reference_encode") or {}).get("seconds"),
            "references": refs,
            "head_equals_prev_tail": c.get("head_equals_prev_tail"),
            "seed": c.get("seed"),
        })
    req = meta.get("request") or {}
    return {
        "mp4_sha256": art["mp4_sha256"], "mp4_size": art["mp4_size"], "backend": meta.get("backend"),
        "attention_used": meta.get("attention_used"), "generation_time_seconds": meta.get("generation_time_seconds"),
        "seeds": h3.get("seeds"), "clips": per,
        "latent_cache_paths": find_key_paths(meta, "latent_cache"),
        "request_embed_mp4_metadata": req.get("embed_mp4_metadata"),
        "request_keep_reference_latents": req.get("keep_reference_latents"),
        "request_clip_references": [c.get("references") for c in req.get("clips") or []],
        "loras": h3.get("loras"),
        "steps_shift": [h3.get("num_inference_steps"), h3.get("shift_video"), h3.get("shift_audio")],
    }


def phase_g2() -> int:
    g1v = RUNS / "verdict_g1.json"
    if not g1v.is_file() or read_json(g1v).get("pass") is not True:
        print("g2 refused: G1 has not passed", flush=True)
        return 1
    ids = read_json(RUNS / "uploads.json")["ids"]
    rec: dict = {"phase": "g2", "started": now(), "runs": {}}
    out = RUNS / "g2.json"
    try:
        for label, keep in (("g2_on", True), ("g2_off", False)):
            body = g2_body(keep, ids)
            r: dict = {"body": body, "submitted_at": now()}
            rec["runs"][label] = r
            st, resp = http("POST", "/generate/chain", body, timeout=300)
            r["submit"] = {"http": st, "resp": resp}
            if st >= 400 or not isinstance(resp, dict) or "job_id" not in resp:
                raise RuntimeError(f"{label}: POST -> {st} {resp}")
            jid = resp["job_id"]
            r["job_id"] = jid
            print(f"{label}: job {jid}", flush=True)
            write_json(out, rec)
            t0 = time.monotonic()
            last = None
            while True:
                code, s = http("GET", f"/jobs/{jid}")
                if code == 200 and isinstance(s, dict):
                    key = (s.get("status"), s.get("stage"), s.get("clip"), s.get("current_step"))
                    if key != last:
                        print(f"  [{time.monotonic() - t0:7.0f}s] {key}", flush=True)
                        last = key
                    if s.get("status") in ("completed", "failed", "cancelled"):
                        break
                if time.monotonic() - t0 > 14400:
                    raise RuntimeError(f"{label}: no end in 4 h")
                time.sleep(10)
            r["final"] = {k: s.get(k) for k in ("status", "error")}
            r["wall_seconds"] = round(time.monotonic() - t0, 1)
            if s.get("status") != "completed":
                write_json(out, rec)
                raise RuntimeError(f"{label} ended {s.get('status')}: {s.get('error')}")
            art = job_artifacts(jid)
            write_json(RUNS / f"{label}.metadata.json", art["metadata"])
            r["summary"] = summarize_chain(art)
            r["mp4_probe"] = art["mp4_probe"]
            print(f"{label}: {r['summary']['mp4_sha256']} gen={r['summary']['generation_time_seconds']}", flush=True)
            write_json(out, rec)
        rec["ok_run"] = True
    except Exception as exc:  # noqa: BLE001
        rec["ok_run"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
        print("g2 error:", rec["error"], flush=True)
    rec["finished"] = now()
    write_json(out, rec)
    return 0 if rec["ok_run"] else 1


# ---------------------------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------------------------

def _chk(checks, name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def verdict_g1() -> dict:
    exp = read_json(RUNS / "selfcheck.json")["expected"]
    g1 = read_json(RUNS / "g1.json")
    ids = g1.get("upload_ids") or {}
    runs = g1.get("runs") or {}
    checks: list = []
    s = {k: (runs.get(k) or {}).get("summary") for k in ("g1_1", "g1_2", "g1_3")}
    _chk(checks, "3 jobs completed", all(((runs.get(k) or {}).get("final") or {}).get("status") == "completed"
                                         for k in s), {k: (runs.get(k) or {}).get("final") for k in s})
    if all(s.values()):
        cache = {k: {r["kind"]: r.get("latent_cache") for r in (s[k]["references"] or [])} for k in s}
        _chk(checks, "(2) and (3) mp4 sha256 equal", s["g1_2"]["mp4_sha256"] == s["g1_3"]["mp4_sha256"],
             [s["g1_2"]["mp4_sha256"], s["g1_3"]["mp4_sha256"]])
        _chk(checks, "(1) picture/video/audio miss", cache["g1_1"] == exp["g1"]["g1_1"], cache["g1_1"])
        _chk(checks, "(2) picture/video/audio hit", cache["g1_2"] == exp["g1"]["g1_2"], cache["g1_2"])
        _chk(checks, "(3) no latent_cache key anywhere in metadata", not s["g1_3"]["latent_cache_paths"],
             s["g1_3"]["latent_cache_paths"])
        _chk(checks, "(2) and (3) denoise.sequence equal", s["g1_2"]["sequence"] == s["g1_3"]["sequence"],
             [s["g1_2"]["sequence"], s["g1_3"]["sequence"]])

        def ext(k):
            d = {r["kind"]: r for r in s[k]["references"] or []}
            return {"picture_size": (d.get("picture") or {}).get("size"),
                    "video_used_frames": (d.get("video") or {}).get("used_frames"),
                    "audio_samples": (d.get("audio") or {}).get("samples")}
        e1, e2 = ext("g1_1"), ext("g1_2")
        want = {"picture_size": exp["picture_size"], "video_used_frames": exp["g1"]["video_used_frames"],
                "audio_samples": exp["audio_samples"]}
        _chk(checks, "extents of (1) and (2) equal the expectation", e1 == want and e2 == want,
             {"g1_1": e1, "g1_2": e2, "expected": want})
        _chk(checks, "embed_mp4_metadata false in every request",
             all(s[k]["request_embed_mp4_metadata"] is False for k in s), [s[k]["request_embed_mp4_metadata"] for k in s])
        ref_ids = [[(r.get("kind"), r.get("id")) for r in (s[k]["request_references"] or [])] for k in s]
        want_ids = [("picture", ids.get("picture")), ("video", ids.get("video_short")), ("audio", ids.get("audio"))]
        _chk(checks, "same reference ids in every job", all(x == want_ids for x in ref_ids), ref_ids)
        _chk(checks, "backend is minimax-h3, attention sdpa",
             all(s[k]["backend"] == "minimax-h3" and s[k]["attention_used"] == "sdpa" for k in s),
             [(s[k]["backend"], s[k]["attention_used"]) for k in s])
    failed = [c["name"] for c in checks if not c["ok"]]
    v = {"gate": "G1", "pass": not failed and len(checks) > 1, "failed": failed, "checks": checks,
         "records": {k: {"job_id": (runs.get(k) or {}).get("job_id"),
                         "generation_time_seconds": (s[k] or {}).get("generation_time_seconds"),
                         "reference_encode_seconds": (s[k] or {}).get("reference_encode_seconds"),
                         "per_reference_seconds": [(r.get("kind"), r.get("seconds")) for r in ((s[k] or {}).get("references") or [])]}
                     for k in s}}
    write_json(RUNS / "verdict_g1.json", v)
    return v


def verdict_g2() -> dict:
    exp = read_json(RUNS / "selfcheck.json")["expected"]["g2"]
    ids = read_json(RUNS / "uploads.json")["ids"]
    g2 = read_json(RUNS / "g2.json")
    runs = g2.get("runs") or {}
    checks: list = []
    s = {k: (runs.get(k) or {}).get("summary") for k in ("g2_on", "g2_off")}
    _chk(checks, "2 jobs completed", all(((runs.get(k) or {}).get("final") or {}).get("status") == "completed"
                                         for k in s), {k: (runs.get(k) or {}).get("final") for k in s})
    records: dict = {}
    if all(s.values()):
        on, off = s["g2_on"], s["g2_off"]
        _chk(checks, "mp4 sha256 equal", on["mp4_sha256"] == off["mp4_sha256"], [on["mp4_sha256"], off["mp4_sha256"]])
        first_diff = None
        lat = []
        for i in range(max(len(on["clips"]), len(off["clips"]))):
            a = (on["clips"][i] if i < len(on["clips"]) else {}).get("latents_sha256") or {}
            b = (off["clips"][i] if i < len(off["clips"]) else {}).get("latents_sha256") or {}
            eq = {"video": bool(a.get("video")) and a.get("video") == b.get("video"),
                  "audio": bool(a.get("audio")) and a.get("audio") == b.get("audio")}
            lat.append(eq)
            if first_diff is None and not (eq["video"] and eq["audio"]):
                first_diff = i + 1
        _chk(checks, "4 clips latents_sha256 equal (video, audio)", len(lat) == 4 and first_diff is None,
             {"per_clip": lat, "first_differing_clip_1based": first_diff})
        cache = {"short": [], "long": []}
        used = {"short": [], "long": []}
        for c in on["clips"]:
            d = {r.get("slot"): r for r in c["references"] or [] if r.get("kind") == "video"}
            cache["short"].append((d.get(1) or {}).get("latent_cache"))
            cache["long"].append((d.get(2) or {}).get("latent_cache"))
            used["short"].append((d.get(1) or {}).get("used_frames"))
            used["long"].append((d.get(2) or {}).get("used_frames"))
        _chk(checks, "on: short video miss/hit/hit/hit", cache["short"] == exp["short_cache"],
             {"got": cache["short"], "used_frames": used["short"], "expected": exp["short_cache"], "expected_used": exp["short_used"]})
        _chk(checks, "on: long video miss/miss/hit/miss", cache["long"] == exp["long_cache"],
             {"got": cache["long"], "used_frames": used["long"], "expected": exp["long_cache"], "expected_used": exp["long_used"]})
        _chk(checks, "off: no latent_cache key", not off["latent_cache_paths"], off["latent_cache_paths"])
        heads = {k: [c.get("head_equals_prev_tail") for c in s[k]["clips"][1:]] for k in s}
        _chk(checks, "head_equals_prev_tail true (clips 2-4, both jobs)",
             all(h is True for k in heads for h in heads[k]) and all(len(heads[k]) == 3 for k in heads), heads)
        _chk(checks, "embed_mp4_metadata false", all(s[k]["request_embed_mp4_metadata"] is False for k in s),
             [s[k]["request_embed_mp4_metadata"] for k in s])
        want = [("video", ids["video_short"], 1), ("video", ids["video_long"], 2)]
        got = {k: [[(r.get("kind"), r.get("id"), r.get("slot")) for r in (cr or [])] for cr in s[k]["request_clip_references"]]
               for k in s}
        _chk(checks, "same reference ids in every clip of both jobs",
             all(len(got[k]) == 4 and all(x == want for x in got[k]) for k in got), got)
        _chk(checks, "backend minimax-h3, attention sdpa",
             all(s[k]["backend"] == "minimax-h3" and s[k]["attention_used"] == "sdpa" for k in s),
             [(s[k]["backend"], s[k]["attention_used"]) for k in s])
        records = {k: {"job_id": (runs.get(k) or {}).get("job_id"),
                       "generation_time_seconds": s[k]["generation_time_seconds"],
                       "clip_seconds": [c.get("seconds") for c in s[k]["clips"]],
                       "clip_reference_encode_phase_seconds": [c.get("reference_encode_seconds_phase") for c in s[k]["clips"]],
                       "seeds": s[k]["seeds"]} for k in s}
    failed = [c["name"] for c in checks if not c["ok"]]
    v = {"gate": "G2", "pass": not failed and len(checks) > 1, "failed": failed, "checks": checks, "records": records}
    write_json(RUNS / "verdict_g2.json", v)
    return v


def phase_verdict(which: str) -> int:
    v = verdict_g1() if which == "g1" else verdict_g2()
    print(json.dumps({"gate": v["gate"], "pass": v["pass"], "failed": v["failed"]}, ensure_ascii=False))
    for c in v["checks"]:
        print(("OK  " if c["ok"] else "NG  ") + c["name"], "|", json.dumps(c["detail"], ensure_ascii=False)[:600])
    return 0 if v["pass"] else 2


# ---------------------------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------------------------

def phase_cleanup() -> int:
    rec: dict = {"phase": "cleanup", "started": now()}

    async def body(m: Mcp):
        rec["status_before"] = await m.call("backend_status")
        rec["load_default"] = await m.call("load_pipeline", {"models": {"transformer": "default"}, "wait_sec": 45})
        rec["loaded_default"] = await wait_loaded(m, "default")
        rec["unload"] = await m.call("unload_pipeline")
        rec["status_after"] = await m.call("backend_status")

    try:
        asyncio.run(with_mcp(body))
        rec["ok"] = ((rec.get("status_after") or {}).get("status") or {}).get("state") == "unloaded"
    except Exception as exc:  # noqa: BLE001
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["finished"] = now()
    write_json(RUNS / "cleanup.json", rec)
    print("cleanup:", rec.get("ok"), rec.get("error"), (rec.get("loaded_default") or {}).get("transformer_active"))
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    if a[:1] == ["--selfcheck"]:
        sys.exit(selfcheck())
    cmd = a[0]
    if cmd == "g1":
        res = dict(x.split("=", 1) for x in a[1:] if "=" in x)   # g1 g1_1=<job_id>
        sys.exit(phase_g1(res or None))
    if cmd == "g2":
        sys.exit(phase_g2())
    if cmd == "verdict":
        sys.exit(phase_verdict(a[1]))
    if cmd == "cleanup":
        sys.exit(phase_cleanup())
    raise SystemExit(f"unknown command {cmd}")
