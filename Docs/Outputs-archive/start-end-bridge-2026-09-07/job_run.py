"""Submit ONE job and poll job_status every 10s inside a single MCP session.

Usage: python job_run.py <spec.json> <result.json>

spec.json:
  {"submit_tool": "submit_generate"|"submit_chain",
   "submit_args": {...},
   "timeout_sec": 1200,
   "save": {"dest_dir": "...", "filename": "..."}   # optional
  }
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from mcp_call import result_payload, server_params

POLL_SEC = 10.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def main() -> None:
    spec_path, out_path = sys.argv[1], sys.argv[2]
    with open(spec_path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)

    tool = spec["submit_tool"]
    args = spec["submit_args"]
    timeout = float(spec.get("timeout_sec", 1200))
    save = spec.get("save")

    out: dict = {"submit_tool": tool, "submit_args": args}

    async with stdio_client(server_params()) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()

            # Guard: make sure nothing is running (single-job queue).
            st = result_payload(await s.call_tool("backend_status", {}))
            out["backend_status_before"] = st
            q = (st.get("status") or {}).get("queue") or {}
            log(f"queue before submit: {q}")
            if q.get("running"):
                out["error"] = f"ABORT: a job is already running: {q}"
                log(out["error"])
                _write(out_path, out)
                sys.exit(3)

            log(f"submitting {tool} ...")
            sub = result_payload(await s.call_tool(tool, args))
            out["submit_result"] = sub
            job_id = sub.get("job_id")
            if not job_id:
                out["error"] = "ABORT: no job_id in submit result"
                log(f"{out['error']}: {json.dumps(sub, ensure_ascii=False)[:2000]}")
                _write(out_path, out)
                sys.exit(4)
            out["job_id"] = job_id
            log(f"job_id={job_id}")

            t0 = time.monotonic()
            status = None
            last = None
            while True:
                await asyncio.sleep(POLL_SEC)
                js = result_payload(await s.call_tool("job_status", {"job_id": job_id}))
                status = js.get("status")
                prog = js.get("progress")
                stage = js.get("stage") or js.get("message")
                line = f"{status} progress={prog} stage={stage}"
                if line != last:
                    log(f"  {int(time.monotonic()-t0)}s {line}")
                    last = line
                if status in ("completed", "failed", "cancelled"):
                    out["final_job_status"] = js
                    break
                if time.monotonic() - t0 > timeout:
                    out["final_job_status"] = js
                    out["error"] = f"TIMEOUT after {timeout}s (status={status})"
                    log(out["error"])
                    break

            out["elapsed_sec"] = round(time.monotonic() - t0, 1)
            out["status"] = status
            log(f"finished status={status} elapsed={out['elapsed_sec']}s")

            if status == "completed":
                vp = result_payload(await s.call_tool("get_job_video_path", {"job_id": job_id}))
                out["video_path"] = vp
                log(f"video_path: {json.dumps(vp, ensure_ascii=False)}")
                if save:
                    sv = result_payload(
                        await s.call_tool(
                            "save_job_video",
                            {"job_id": job_id, "dest_dir": save["dest_dir"],
                             "filename": save["filename"], "no_clobber": True},
                        )
                    )
                    out["save_result"] = sv
                    log(f"saved: {json.dumps(sv, ensure_ascii=False)}")

    _write(out_path, out)
    if out.get("error") or status != "completed":
        sys.exit(5)


def _write(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
