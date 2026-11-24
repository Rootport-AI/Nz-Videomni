r"""submit_chain の num_inference_steps / shift_video / shift_audio が MiniMax H3 の連結に届くかの確認 (dev a3cbdcc6).

    .venv\Scripts\python.exe -X utf8 -B outputs\mcp-chain-steps-gate-2026-11-25\g_chainsteps.py selfcheck   # CPU, no server
    ... run        # MCP: load (selected weights as they are), submit_chain once, wait, read metadata
    ... resume <job_id>   # only waits for an already submitted job (never submits again)
    ... verdict
    ... cleanup    # MCP: unload_pipeline
The server (127.0.0.1:18620) is started/stopped outside this script.
"""
from __future__ import annotations

import asyncio
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
FFPROBE = REPO / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"
TURBO = "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16"
PROMPT = "A woman in a red raincoat walks along a quiet harbor at dawn, soft light, gentle camera movement."
CLIP_PROMPTS = ["She walks slowly along the wooden pier.", "She stops at the end of the pier and looks at the sea."]
ARGS = dict(
    prompt=PROMPT, width=1280, height=768, frame_rate=24.0, seed=42,
    loras=[{"name": TURBO, "strength": 0.8}],
    num_inference_steps=4, shift_video=12, shift_audio=3,
    attention_backend="sdpa", keep_resident=True,
)
CLIPS = [{"num_frames": 73, "prompt": CLIP_PROMPTS[0]}, {"num_frames": 73, "prompt": CLIP_PROMPTS[1]}]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def write_json(p: Path, o):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def selfcheck() -> int:
    import functools
    import anyio
    import httpx
    sys.path.insert(0, str(REPO))
    from mcp_server.client import BackendClient, set_client
    from mcp_server.params import ChainClipArg, LoraArg
    from mcp_server.settings import Settings
    from mcp_server.tools import generate
    from api.models import GenerateChainRequest
    cap: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        cap["body"] = json.loads(request.content)
        return httpx.Response(202, json={"job_id": "j1", "status": "queued", "created_at": "t", "num_clips": 2})

    set_client(BackendClient(Settings(base_url="http://testserver", api_key=None, output_dir=Path(".")),
                             transport=httpx.MockTransport(handler)))
    kw = {k: v for k, v in ARGS.items() if k not in ("prompt", "loras")}
    kw["loras"] = [LoraArg(**ln) for ln in ARGS["loras"]]
    anyio.run(functools.partial(generate.submit_chain, ARGS["prompt"], [ChainClipArg(**c) for c in CLIPS], **kw))
    body = cap["body"]
    req = GenerateChainRequest.model_validate(body)
    problems = []
    for k, want in (("num_inference_steps", 4), ("shift_video", 12), ("shift_audio", 3)):
        if body.get(k) != want or getattr(req, k) != want:
            problems.append(f"{k}: body={body.get(k)} model={getattr(req, k)}")
    if "num_inference_steps" not in req.model_fields_set:
        problems.append("num_inference_steps not in model_fields_set")
    found = sorted((REPO / "models").rglob(f"{TURBO}.*"))
    if not found:
        problems.append("turbo lora file not found under models")
    rec = {"at": now(), "body": body, "model_fields_set": sorted(req.model_fields_set),
           "turbo_files_rel": [str(p.relative_to(REPO)) for p in found],
           "problems": problems, "ok": not problems}
    write_json(RUNS / "selfcheck.json", rec)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    return 0 if rec["ok"] else 2


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


UNREACH = {"count": 0, "at": []}


async def wait_done(m, job_id, limit=2400):
    t0 = time.monotonic()
    unreachable = 0
    while True:
        try:
            st = await m.call("wait_for_job", {"job_id": job_id, "timeout_sec": 45, "poll_interval_sec": 5})
            unreachable = 0
        except RuntimeError as exc:
            if "BACKEND_UNREACHABLE" not in str(exc) or unreachable >= 30:
                raise
            unreachable += 1
            UNREACH["count"] += 1
            UNREACH["at"].append(now())
            print(f"  BACKEND_UNREACHABLE ({unreachable}) {now()}", flush=True)
            await asyncio.sleep(10)
            continue
        if not st.get("timed_out"):
            return st
        print(f"  {st.get('status')} stage={st.get('stage')} step={st.get('current_step')}/"
              f"{st.get('total_steps')} t={time.monotonic() - t0:.0f}s", flush=True)
        if time.monotonic() - t0 > limit:
            st["wait_limit_exceeded"] = True
            return st


async def wait_loaded(m, limit=1200):
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


def ffprobe(p: Path):
    r = subprocess.run([str(FFPROBE), "-v", "error", "-show_entries",
                        "format=duration:stream=codec_type,duration,nb_frames", "-of", "json", str(p)],
                       capture_output=True, text=True, encoding="utf-8")
    return json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[:500]}


def phase_run(resume_id):
    out = RUNS / "run.json"
    rec: dict = {"started": now(), "resume": resume_id}
    if resume_id:
        rec = read_json(out)
        rec["resumed_at"] = now()

    async def body(m: Mcp):
        if not resume_id:
            rec["backend_status"] = await m.call("backend_status")
            lm = await m.call("list_models")
            tr = (lm.get("categories") or {}).get("transformer") or {}
            rec["models_before"] = {"active_base_model": lm.get("active_base_model"),
                                    "transformer_active": tr.get("active")}
            ll = await m.call("list_loras")
            if TURBO not in json.dumps(ll, ensure_ascii=False):
                raise RuntimeError("turbo lora not in list_loras")
            if lm.get("active_base_model") != "MiniMaxH3":
                raise RuntimeError(f"active base model is {lm.get('active_base_model')}: stop (not switching on my own)")
            # load the selected weights as they are (no models arg => no selection change)
            rec["load_call"] = await m.call("load_pipeline", {"wait_sec": 45})
            rec["loaded"] = await wait_loaded(m)
            args = dict(ARGS)
            args["clips"] = CLIPS
            rec["submit_args"] = args
            sub = await m.call("submit_chain", args)
            jid = sub["job_id"]
            rec["job_id"] = jid
            rec["submitted_at"] = now()
            (HERE / "logs" / "job_id.txt").write_text(jid, encoding="utf-8")
            write_json(out, rec)
            print("job", jid, flush=True)
        else:
            rec["job_id"] = resume_id
        jid = rec["job_id"]
        t0 = time.monotonic()
        fin = await wait_done(m, jid)
        rec["final"] = {k: fin.get(k) for k in ("status", "error", "timed_out", "wait_limit_exceeded", "stage")}
        rec["wait_seconds"] = round(time.monotonic() - t0, 1)
        rec["unreachable"] = dict(UNREACH)
        write_json(out, rec)
        if fin.get("status") == "completed":
            d = REPO / "outputs" / jid
            write_json(RUNS / "metadata.json", read_json(d / "metadata.json"))
            rec["mp4_probe"] = ffprobe(d / "output.mp4")
        write_json(out, rec)

    try:
        asyncio.run(with_mcp(body, "run"))
        rec["ok_run"] = True
    except BaseException as exc:  # noqa: BLE001
        rec["ok_run"] = False
        subs = getattr(exc, "exceptions", None)
        rec["error"] = f"{type(exc).__name__}: {exc}" + (f" | {[repr(e) for e in subs]}" if subs else "")
        rec["unreachable"] = dict(UNREACH)
        print("run error:", rec["error"], flush=True)
    rec["finished"] = now()
    write_json(out, rec)
    return 0 if rec["ok_run"] else 1


def num_eq(a, b):
    return isinstance(a, (int, float)) and not isinstance(a, bool) and float(a) == float(b)


def phase_verdict():
    run = read_json(RUNS / "run.json")
    checks = []

    def chk(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    fin = run.get("final") or {}
    chk("job completed", fin.get("status") == "completed", fin)
    ref = {}
    mp = RUNS / "metadata.json"
    if mp.is_file():
        meta = read_json(mp)
        req, h3 = meta.get("request") or {}, meta.get("h3") or {}
        for k, w in (("num_inference_steps", 4), ("shift_video", 12), ("shift_audio", 3)):
            chk(f"request.{k}=={w}", num_eq(req.get(k), w), req.get(k))
            chk(f"h3.{k}=={w}", num_eq(h3.get(k), w), h3.get(k))
        pc = (h3.get("phases") or {}).get("clips") or []
        den = [((c or {}).get("denoise") or {}) for c in pc]
        chk("2 clips in h3.phases.clips", len(pc) == 2, len(pc))
        for i, d in enumerate(den):
            mps = d.get("m_per_step")
            chk(f"clip{i} denoise.m_per_step length 4", isinstance(mps, list) and len(mps) == 4,
                {"len": len(mps) if isinstance(mps, list) else None, "keys_in_denoise": sorted(d.keys())})
        ref = {"request.loras": req.get("loras"), "h3.loras": h3.get("loras"), "backend": meta.get("backend"),
               "models": meta.get("models"), "attention_used": meta.get("attention_used"),
               "generation_time_seconds": meta.get("generation_time_seconds"),
               "denoise_seconds": [d.get("seconds") for d in den],
               "request.clips_num_frames": [c.get("num_frames") for c in req.get("clips") or []],
               "mp4_probe": run.get("mp4_probe"), "unreachable": run.get("unreachable"),
               "top_level_keys": sorted(meta.keys()), "h3_keys": sorted(h3.keys())}
    failed = [c["name"] for c in checks if not c["ok"]]
    v = {"pass": not failed and len(checks) > 1, "failed": failed, "checks": checks, "reference": ref,
         "loaded_weights": (run.get("loaded") or {}).get("transformer_active")}
    write_json(RUNS / "verdict.json", v)
    print(json.dumps(v, ensure_ascii=False, indent=1)[:6000])
    return 0 if v["pass"] else 2


def phase_cleanup():
    rec = {"started": now()}

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


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    c = a[0]
    if c == "selfcheck":
        sys.exit(selfcheck())
    if c == "run":
        sys.exit(phase_run(None))
    if c == "resume":
        sys.exit(phase_run(a[1]))
    if c == "verdict":
        sys.exit(phase_verdict())
    if c == "cleanup":
        sys.exit(phase_cleanup())
    raise SystemExit("unknown")
