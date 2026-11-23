r"""pi-H3 stage 4: the MCP tools submit_generate / submit_chain with pi_h3, against the OWNER's running backend
(127.0.0.1:18620; never started, stopped or reloaded here). The MCP server is started over stdio as in
VERIFICATION_LOG §39.6 (``.venv/Scripts/python.exe -m mcp_server``) and the tools are called through the
``mcp`` client.

    .venv/Scripts/python.exe -X utf8 -B outputs/h3-pih3-gates-2026-11-23/mcp_check.py pre     # gate, items 1 / 3 / 5
    .venv/Scripts/python.exe -X utf8 -B outputs/h3-pih3-gates-2026-11-23/mcp_check.py single  # item 2 (MCP, then API)
    .venv/Scripts/python.exe -X utf8 -B outputs/h3-pih3-gates-2026-11-23/mcp_check.py chain   # item 4

Each phase first checks: no job running / pending on the owner's server, H3 loaded, no python process other
than the server's own on the GPU beyond what was there (counts only are recorded). Records: runs/MCP<n>/.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import h3_pih3_gates as G  # noqa: E402  -- helpers only (scrub, md5, write_json); nothing is started by importing

BASE = "http://127.0.0.1:18620/api/v1"
PY = REPO / ".venv" / "Scripts" / "python.exe"
CITY = G.CITY
BEACH = G.CHAIN_SAME_PROMPT
#: The stage-1 P1 conditions (sage, VRAM 16GB mode on, prefetch on, not resident).
SINGLE = dict(prompt=CITY, width=1280, height=768, num_frames=73, frame_rate=24.0, seed=1234,
              attention_backend="sage", vram16_mode=True, block_swap_prefetch=True, keep_resident=False)


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


def server_state() -> dict:
    st, s = http("GET", "/status")
    _, m = http("GET", "/models")
    tr = (m.get("categories") or {}).get("transformer") or {}
    act = tr.get("active")
    f = [e for e in tr.get("entries") or [] if e.get("name") == act]
    h3 = next((b for b in m.get("base_models") or [] if b.get("id") == "MiniMaxH3"), {})
    gpu = G.gpu_busy_check()
    return {"status_http": st, "state": s.get("state"), "base_model": s.get("base_model"),
            "pipeline_loaded": s.get("pipeline_loaded"), "queue": s.get("queue"),
            "transformer": os.path.basename(f[0].get("path", "")) if f else None,
            "h3_constraints_pi_h3": (h3.get("constraints") or {}).get("pi_h3"),
            "gpu": {k: gpu.get(k) for k in ("memory_used_mib", "compute_apps_count", "python_like_count")}}


def gate(rec: dict) -> bool:
    s = server_state()
    rec["state_before"] = s
    q = s.get("queue") or {}
    bad = []
    if q.get("running") or q.get("pending"):
        bad.append(f"ジョブが実行中か待ち（running {q.get('running')}・pending {q.get('pending')}）")
    if s.get("base_model") != "MiniMaxH3" or not s.get("pipeline_loaded"):
        bad.append(f"MiniMax H3 が読み込まれていない（{s.get('base_model')}・{s.get('pipeline_loaded')}）")
    if not s.get("h3_constraints_pi_h3"):
        bad.append("constraints.pi_h3 が配られていない")
    g = s["gpu"]
    if (g.get("python_like_count") or 0) > 1 or (g.get("memory_used_mib") or 0) > 6000:
        bad.append(f"GPU をほかが使っている疑い（python {g.get('python_like_count')} 件・{g.get('memory_used_mib')} MiB）")
    rec["gate_blocking"] = bad
    for b in bad:
        print("gate refuses:", b, flush=True)
    return not bad


class Mcp:
    def __init__(self, session, calls):
        self.s, self.calls = session, calls

    async def call(self, name: str, args: dict | None = None, *, expect_error: bool = False) -> dict:
        t0 = time.monotonic()
        r = await self.s.call_tool(name, args or {}, read_timeout_seconds=timedelta(seconds=360))
        text = "\n".join(getattr(c, "text", "") for c in (r.content or []) if getattr(c, "text", None))
        entry = {"tool": name, "args": args or {}, "is_error": bool(r.isError), "seconds": round(time.monotonic() - t0, 2)}
        if r.isError:
            entry["error_text"] = text
            self.calls.append(entry)
            if expect_error:
                return {"_error": text}
            raise RuntimeError(f"{name}: {text}")
        data = r.structuredContent if isinstance(r.structuredContent, dict) else (json.loads(text) if text else {})
        if name not in ("list_models", "wait_for_job"):
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
    with open(HERE / "logs" / "mcp_check.stderr.log", "a", encoding="utf-8") as err:
        async with stdio_client(params, errlog=err) as (rd, wr):
            async with ClientSession(rd, wr, read_timeout_seconds=timedelta(seconds=360)) as s:
                init = await s.initialize()
                tools = (await s.list_tools()).tools
                return await fn(Mcp(s, calls), init, tools), calls


async def wait_done(m: Mcp, job_id: str, limit: float = 3600) -> dict:
    t0 = time.monotonic()
    while True:
        st = await m.call("wait_for_job", {"job_id": job_id, "timeout_sec": 45, "poll_interval_sec": 5})
        if not st.get("timed_out"):
            return st
        print(f"  {job_id[:8]} {st.get('status')} stage={st.get('stage')} step={st.get('current_step')}", flush=True)
        if time.monotonic() - t0 > limit:
            return st


def job_files(job_id: str, video_path: str | None) -> tuple[Path | None, dict]:
    p = Path(video_path) if video_path else None
    meta = {}
    if p and (p.parent / "metadata.json").is_file():
        meta = json.loads((p.parent / "metadata.json").read_text(encoding="utf-8"))
    return p, meta


def meta_picks(meta: dict) -> dict:
    h3 = meta.get("h3") or {}
    return {"pi_h3_used": meta.get("pi_h3_used"), "steps_shift": [h3.get("num_inference_steps"), h3.get("shift_video"),
                                                                  h3.get("shift_audio")],
            "attention_used": meta.get("attention_used"), "vram16_mode_used": meta.get("vram16_mode_used"),
            "generation_time_seconds": meta.get("generation_time_seconds"),
            "transformer": ((meta.get("models") or {}).get("selection") or {}).get("transformer", {}).get("file"),
            "pi_h3": {k: (h3.get("pi_h3") or {}).get(k) for k in ("choice", "steps", "file", "dwd")},
            "request": meta.get("request")}


# ---------------------------------------------------------------------------------------------
# Item 5's tool: the payload submit_generate / submit_chain would POST, captured in-process (no POST)
# ---------------------------------------------------------------------------------------------

def captured_payload(tool: str, **kw) -> dict:
    import mcp_server.tools.generate as T  # noqa: PLC0415
    from mcp_server import client as C  # noqa: PLC0415

    real = C.get_client()

    class Stub:
        def __getattr__(self, n):
            return getattr(real, n)

        async def get_json(self, path, *, params=None):   # read-only GETs go to the real server
            return await real.get_json(path, params=params)

        async def post_json(self, path, *, json=None, **k):   # noqa: A002
            self.sent = {"path": path, "json": json}
            return {"job_id": "captured-not-sent", "status": "queued"}

    stub = Stub()
    C.set_client(stub)
    try:
        asyncio.run(getattr(T, tool)(**kw))
    finally:
        C.set_client(None)
        try:
            asyncio.run(real.aclose())
        except Exception:  # noqa: BLE001
            pass
    return stub.sent


# ---------------------------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------------------------

def phase_pre() -> int:
    out = HERE / "runs" / "MCP_pre"
    out.mkdir(parents=True, exist_ok=True)
    rec: dict = {"phase": "pre", "started_at": G.now_iso(), "git": G.git_info()}
    if not gate(rec):
        G.write_json(out / "result.json", rec)
        return 1

    async def body(m: Mcp, init, tools):
        r: dict = {"server_info": {"name": init.serverInfo.name, "protocol": init.protocolVersion}}
        by = {t.name: t for t in tools}
        r["tool_count"] = len(tools)
        r["pi_h3_param"] = {n: (by[n].inputSchema or {}).get("properties", {}).get("pi_h3") for n in
                            ("submit_generate", "submit_chain")}
        lm = await m.call("list_models")
        bm = {b["id"]: b for b in lm.get("base_models") or []}
        r["item1"] = {"h3_constraints_pi_h3": (bm.get("MiniMaxH3", {}).get("constraints") or {}).get("pi_h3"),
                      "ltx_unsupported_has_pi_h3": {k: "pi_h3" in (bm.get(k, {}).get("unsupported_features") or [])
                                                    for k in ("LTX23", "LTX25")},
                      "active_base_model": lm.get("active_base_model")}
        jobs_before = (await m.call("list_jobs")).get("jobs") or []
        res = await m.call("submit_generate", {**SINGLE, "pi_h3": "4steps", "num_inference_steps": 20},
                           expect_error=True)
        jobs_after = (await m.call("list_jobs")).get("jobs") or []
        r["item3"] = {"error_text": res.get("_error"), "returned": None if "_error" in res else res,
                      "jobs_before": len(jobs_before), "jobs_after": len(jobs_after)}
        return r

    r, calls = asyncio.run(with_mcp(body))
    rec.update(r)
    rec["item5"] = {
        "off_payload": captured_payload("submit_generate", **SINGLE)["json"],
        "4steps_payload": captured_payload("submit_generate", **SINGLE, pi_h3="4steps")["json"],
    }
    rec["item5"]["off_has_pi_h3_key"] = "pi_h3" in rec["item5"]["off_payload"]
    rec["item5"]["4steps_has_steps_or_shift"] = [k for k in ("num_inference_steps", "shift_video", "shift_audio")
                                                 if k in rec["item5"]["4steps_payload"]]
    rec["calls"] = calls
    rec["finished_at"] = G.now_iso()
    G.write_json(out / "result.json", rec)
    print(json.dumps({k: rec[k] for k in ("item1", "item3")}, ensure_ascii=False, indent=1))
    print("item5 off has pi_h3:", rec["item5"]["off_has_pi_h3_key"], "4steps steps/shift keys:",
          rec["item5"]["4steps_has_steps_or_shift"])
    return 0


def phase_single() -> int:
    out = HERE / "runs" / "MCP2"
    out.mkdir(parents=True, exist_ok=True)
    rec: dict = {"phase": "single", "started_at": G.now_iso(), "git": G.git_info()}
    if not gate(rec):
        G.write_json(out / "result.json", rec)
        return 1
    args = {**SINGLE, "pi_h3": "4steps"}
    rec["mcp_args"] = args
    rec["mcp_payload_captured"] = captured_payload("submit_generate", **args)["json"]

    async def body(m: Mcp, init, tools):
        r: dict = {}
        t0 = time.monotonic()
        sub = await m.call("submit_generate", args)
        jid = sub["job_id"]
        r["mcp_job_id"] = jid
        fin = await wait_done(m, jid)
        r["mcp_final"] = {k: fin.get(k) for k in ("status", "error")}
        r["mcp_seconds_wall"] = round(time.monotonic() - t0, 1)
        vp = await m.call("get_job_video_path", {"job_id": jid})
        r["mcp_video_path"] = vp.get("path")
        return r

    if len(sys.argv) > 2:
        # resume: the MCP job already ran (submitted through the tool); only its files are read here
        async def body2(m: Mcp, init, tools):
            jid = sys.argv[2]
            fin = await m.call("job_status", {"job_id": jid})
            vp = await m.call("get_job_video_path", {"job_id": jid})
            return {"mcp_job_id": jid, "mcp_final": {k: fin.get(k) for k in ("status", "error")},
                    "mcp_video_path": vp.get("path"), "resumed": True}
        r, calls = asyncio.run(with_mcp(body2))
    else:
        r, calls = asyncio.run(with_mcp(body))
    rec.update(r)
    rec["calls"] = calls
    mp4, meta = job_files(rec["mcp_job_id"], rec.get("mcp_video_path"))
    if mp4 and mp4.is_file():
        shutil.copy2(mp4, out / "mcp.mp4")
        G.write_text(out / "mcp.metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
    rec["mcp_meta"] = meta_picks(meta)
    G.write_json(out / "result.json", rec)
    # API direct, after: the very body the MCP tool sent (captured in-process, equal to what it POSTed)
    body_api = rec["mcp_payload_captured"]
    st, resp = http("POST", "/generate", body_api, timeout=300)
    rec["api_submit"] = {"http": st, "resp": resp}
    if st == 202:
        jid = resp["job_id"]
        rec["api_job_id"] = jid
        t0 = time.monotonic()
        while True:
            _, s = http("GET", f"/jobs/{jid}")
            if s.get("status") in ("completed", "failed", "cancelled") or time.monotonic() - t0 > 3600:
                break
            time.sleep(5)
        rec["api_final"] = {k: s.get(k) for k in ("status", "error")}
        rec["api_seconds_wall"] = round(time.monotonic() - t0, 1)
        vp = Path(rec["mcp_video_path"]).parent.parent / jid / "output.mp4" if rec.get("mcp_video_path") else None
        mp4b, metab = job_files(jid, str(vp) if vp else None)
        if mp4b and mp4b.is_file():
            shutil.copy2(mp4b, out / "api.mp4")
            G.write_text(out / "api.metadata.json", json.dumps(metab, ensure_ascii=False, indent=2))
        rec["api_meta"] = meta_picks(metab)
        if (out / "mcp.mp4").is_file() and (out / "api.mp4").is_file():
            rec["compare"] = G.compare(out / "mcp.mp4", {"label": "API direct (same body)", "mp4": out / "api.mp4"})
            rec["sha256"] = {"mcp": G.sha256(out / "mcp.mp4"), "api": G.sha256(out / "api.mp4")}
        # the request bodies the server recorded, key by key
        a, b = (rec["mcp_meta"].get("request") or {}), (rec["api_meta"].get("request") or {})
        rec["request_diff"] = {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
    rec["finished_at"] = G.now_iso()
    G.write_json(out / "result.json", rec)
    print(json.dumps({k: rec.get(k) for k in ("mcp_job_id", "mcp_final", "api_job_id", "api_final", "request_diff")},
                     ensure_ascii=False))
    print("mcp meta:", {k: v for k, v in rec["mcp_meta"].items() if k != "request"})
    print("compare:", {k: (rec.get("compare") or {}).get(k) for k in ("video_equal", "audio_equal")}, rec.get("sha256"))
    return 0


def phase_chain() -> int:
    out = HERE / "runs" / "MCP4"
    out.mkdir(parents=True, exist_ok=True)
    rec: dict = {"phase": "chain", "started_at": G.now_iso(), "git": G.git_info()}
    if not gate(rec):
        G.write_json(out / "result.json", rec)
        return 1
    args = {"prompt": BEACH, "clips": [{"num_frames": 124}, {"num_frames": 124}], "width": 640, "height": 384,
            "seed": 1234, "attention_backend": "sage", "block_swap_prefetch": True, "keep_resident": False,
            "pi_h3": "4steps"}
    rec["mcp_args"] = args

    async def body(m: Mcp, init, tools):
        r: dict = {}
        t0 = time.monotonic()
        sub = await m.call("submit_chain", args)
        jid = sub["job_id"]
        r["mcp_job_id"] = jid
        fin = await wait_done(m, jid)
        r["mcp_final"] = {k: fin.get(k) for k in ("status", "error")}
        r["mcp_seconds_wall"] = round(time.monotonic() - t0, 1)
        vp = await m.call("get_job_video_path", {"job_id": jid})
        r["mcp_video_path"] = vp.get("path")
        return r

    r, calls = asyncio.run(with_mcp(body))
    rec.update(r)
    rec["calls"] = calls
    mp4, meta = job_files(rec["mcp_job_id"], rec.get("mcp_video_path"))
    if mp4 and mp4.is_file():
        shutil.copy2(mp4, out / "mcp.mp4")
        G.write_text(out / "mcp.metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
        rec["probe"] = G.ffprobe(out / "mcp.mp4")
    rec["mcp_meta"] = meta_picks(meta)
    rec["finished_at"] = G.now_iso()
    G.write_json(out / "result.json", rec)
    print(json.dumps({k: rec.get(k) for k in ("mcp_job_id", "mcp_final", "mcp_seconds_wall")}, ensure_ascii=False))
    print("meta:", {k: v for k, v in rec["mcp_meta"].items() if k != "request"},
          (rec.get("probe") or {}).get("video", {}).get("nb_read_frames"))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit({"pre": phase_pre, "single": phase_single, "chain": phase_chain}[sys.argv[1]]())
