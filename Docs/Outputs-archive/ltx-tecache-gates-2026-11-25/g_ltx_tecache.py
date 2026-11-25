r"""LTX 2.3 / LTX 2.5 keep_text_conditioning (ledger 1-299): GPU gates G1, G2 (LTX 2.3) and G5, G6 (LTX 2.5).
Entry point is MCP (stdio, ``python -m mcp_server``). Modelled on the 1-298 script (g_tecache.py).

    .venv\Scripts\python.exe -X utf8 -B outputs\ltx-tecache-gates-2026-11-25\g_ltx_tecache.py selfcheck
    ... g_ltx_tecache.py switch LTX23|LTX25|MiniMaxH3   # load_pipeline(base_model=...) and wait until loaded
    ... g_ltx_tecache.py g1 | g2 | g5 | g6              # run the jobs one by one (resumable)
    ... g_ltx_tecache.py unload                         # unload_pipeline

The server (127.0.0.1:18620) is started and stopped outside this script. Records: runs\<gate>.json.
A gate stops at the first result that differs from the expectation (nothing is investigated or retried).
Paths in the records are relative to the repository.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.dont_write_bytecode = True
SUF = os.environ.get("GATE_SUFFIX", "")  # "2" for the second run (G3/G4/G7): record names get this suffix
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
RUNS = HERE / "runs"
PY = REPO / ".venv" / "Scripts" / "python.exe"
FFPROBE = REPO / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"

# bytes per held string (ledger 1-299): LTX 2.3 12,582,912 per string; LTX 2.5 12,582,912 + 8,192 per entry
B23 = 12_582_912
B25 = 12_582_912 + 8_192

P_A = "A woman in a red raincoat walks along a wooden pier at dawn. Soft light, gentle camera movement."
P_B = "A woman in a red raincoat walks slowly along a wooden pier at dawn. Soft light, gentle camera movement."
N_A = "blurry, low quality, distorted faces"
N_B = "flicker, jitter, watermark, text overlay"
S = "A quiet harbor at dawn. Soft light, gentle camera movement."
CP = ["Small boats rock gently beside the pier.", "Mist drifts slowly over the calm water.",
      "The first sunlight reaches the far shore."]
P2X = "A gull glides low over the calm water."

P_O = "A quiet harbor with wooden piers and soft morning mist stretches beyond the frame. Gentle camera movement."
P_I = "A bright red umbrella stands on the wooden pier at dawn. Soft light, gentle camera movement."
SRC_REL = "outputs/43761f33-1030-48ec-babf-0fc0798c3aef/output.mp4"
MASK_REL = "outputs/inpaint_spike/material_full/mask.mp4"
LORA = {"name": "in-outpainting", "strength": 1.0}

COMMON = dict(width=1280, height=768, frame_rate=24.0, seed=42, keep_resident=False,
              keep_resident_embeddings=False, embed_mp4_metadata=False)


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


def rel(p) -> str:
    try:
        return os.path.relpath(str(p), REPO).replace("\\", "/")
    except ValueError:
        return Path(p).name


def ffprobe(path: Path) -> dict:
    r = subprocess.run([str(FFPROBE), "-v", "error", "-count_frames", "-show_entries",
                        "stream=codec_type,width,height,r_frame_rate,nb_read_frames,duration",
                        "-of", "json", str(path)], capture_output=True, text=True, encoding="utf-8")
    return json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[:300]}


def find_paths(obj, pred, prefix=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            if pred(k):
                out.append((p, v if not isinstance(v, (dict, list)) else "<{}>".format(type(v).__name__)))
            out.extend(find_paths(v, pred, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(find_paths(v, pred, f"{prefix}[{i}]"))
    return out


def has_key(obj, key) -> bool:
    return any(find_paths(obj, lambda k: k == key))


# ---------------------------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------------------------

def single_args(keep: bool, prompt: str, neg: str | None, method: str = "nag", resident: bool = False,
                resident_emb: bool = False) -> dict:
    a = dict(COMMON)
    a.update(prompt=prompt, num_frames=121, nag_enabled=neg is not None)
    if neg is not None:
        a.update(negative_prompt=neg, neg_method=method)
    if keep:
        a["keep_text_conditioning"] = True
    if resident:
        a["keep_resident"] = True
    if resident_emb:
        a["keep_resident_embeddings"] = True
    return a


def chain_args(keep: bool, prompts: list[str]) -> dict:
    a = dict(COMMON)
    a.update(prompt=S, nag_enabled=False, clips=[{"num_frames": 121, "prompt": p} for p in prompts])
    if keep:
        a["keep_text_conditioning"] = True
    return a


# ---------------------------------------------------------------------------------------------
# plans.  te for LTX23 single: {"positive","negative"?}; chain: {"clips":[..],"stage2":..}; LTX25: "hit"|"miss"; None = OFF
# eq: (label, what) with what in mp4 | csha
# ---------------------------------------------------------------------------------------------

def plan_g1():
    mm = {"positive": "miss", "negative": "miss"}
    hh = {"positive": "hit", "negative": "hit"}
    J = []

    def add(label, cond, args, te, eq):
        J.append(dict(label=label, cond=cond, args=args, te=te, eq=eq))
    add("1", "ON, A, Na, nag", single_args(True, P_A, N_A), mm, [])
    add("2", "OFF, A, Na, nag", single_args(False, P_A, N_A), None, [("1", "mp4")])
    add("3", "ON, A, Na, nag", single_args(True, P_A, N_A), mm, [("1", "mp4"), ("1", "csha")])
    add("4", "ON, A, Na, nag", single_args(True, P_A, N_A), hh, [("1", "mp4")])
    add("5", "ON, A, Nb, nag", single_args(True, P_A, N_B), {"positive": "hit", "negative": "miss"}, [])
    add("6", "ON, B, Nb, nag", single_args(True, P_B, N_B), {"positive": "miss", "negative": "hit"}, [])
    add("7", "ON, B, Nb, vsf", single_args(True, P_B, N_B, "vsf"), hh, [])
    add("8", "ON, B, Nb, vsf, keep_resident", single_args(True, P_B, N_B, "vsf", resident=True), hh, [("7", "mp4")])
    add("9", "OFF, A, Nb, nag", single_args(False, P_A, N_B), None, [("5", "mp4")])
    add("10", "OFF, B, Nb, nag", single_args(False, P_B, N_B), None, [("6", "mp4")])
    add("11", "OFF, B, Nb, vsf", single_args(False, P_B, N_B, "vsf"), None, [("7", "mp4"), ("8", "mp4")])
    return J


def plan_g2():
    def te(c, s):
        return {"clips": c, "stage2": s}
    p2x = [CP[0], P2X, CP[2]]
    return [
        dict(label="1", cond="OFF", args=chain_args(False, CP), te=None, eq=[]),
        dict(label="2", cond="ON", args=chain_args(True, CP), te=te(["miss"] * 3, "miss"), eq=[("1", "mp4")]),
        dict(label="3", cond="ON (same)", args=chain_args(True, CP), te=te(["hit"] * 3, "hit"),
             eq=[("1", "mp4"), ("2", "mp4")]),
        dict(label="4", cond="ON, clip 2 -> p2'", args=chain_args(True, p2x), te=te(["hit", "miss", "hit"], "hit"), eq=[]),
        dict(label="5", cond="OFF (same input as 4)", args=chain_args(False, p2x), te=None, eq=[("4", "mp4")]),
    ]


def plan_g5():
    J = []

    def add(label, cond, args, te, eq):
        J.append(dict(label=label, cond=cond, args=args, te=te, eq=eq))
    add("1", "ON, A, no negative", single_args(True, P_A, None), "miss", [])
    add("2", "OFF, A", single_args(False, P_A, None), None, [("1", "mp4")])
    add("3", "ON, A", single_args(True, P_A, None), "miss", [("1", "mp4"), ("1", "csha")])
    add("4", "ON, A", single_args(True, P_A, None), "hit", [("1", "mp4")])
    add("5", "ON, A, Na, nag", single_args(True, P_A, N_A), "miss", [])
    add("6", "ON, A, Na, nag", single_args(True, P_A, N_A), "hit", [("5", "mp4")])
    add("7", "ON, A, Na, nag, keep_resident+embeddings", single_args(True, P_A, N_A, resident=True, resident_emb=True),
        "hit", [("5", "mp4")])
    add("8", "ON, A, Na, vsf", single_args(True, P_A, N_A, "vsf"), "hit", [])
    add("9", "OFF, A, Na, nag", single_args(False, P_A, N_A), None, [("5", "mp4"), ("6", "mp4"), ("7", "mp4")])
    add("10", "OFF, A, Na, vsf", single_args(False, P_A, N_A, "vsf"), None, [("8", "mp4")])
    return J


def plan_g6():
    p2x = [CP[0], P2X, CP[2]]
    return [
        dict(label="1", cond="OFF", args=chain_args(False, CP), te=None, eq=[]),
        dict(label="2", cond="ON", args=chain_args(True, CP), te="miss", eq=[("1", "mp4")]),
        dict(label="3", cond="ON (same)", args=chain_args(True, CP), te="hit", eq=[("1", "mp4"), ("2", "mp4")]),
        dict(label="4", cond="ON, clip 2 -> p2'", args=chain_args(True, p2x), te="miss", eq=[]),
        dict(label="5", cond="OFF (same input as 4)", args=chain_args(False, p2x), te=None, eq=[("4", "mp4")]),
    ]


def ids() -> dict:
    p = RUNS / "uploads2.json"
    return read_json(p) if p.is_file() else {"source_video_id": "SRC", "mask_video_id": "MASK"}


def outpaint_args(keep: bool) -> dict:
    a = single_args(keep, P_O, None)
    a.update(width=1536, height=896, reference_video_id=ids()["source_video_id"], outpaint_pad_left=128,
             outpaint_pad_right=128, outpaint_pad_top=64, outpaint_pad_bottom=64, outpaint_blend_dilation_stage1=5)
    return a


def inpaint_body(keep: bool) -> dict:
    """POST /generate body (direct API; MCP does not support LTX Inpainting)."""
    return dict(prompt=P_I, width=1280, height=768, num_frames=121, frame_rate=24.0, seed=42, nag_enabled=False,
                keep_resident=False, keep_resident_embeddings=False, embed_mp4_metadata=False,
                keep_text_conditioning=keep, reference_video_id=ids()["source_video_id"], loras=[dict(LORA)],
                inpaint=dict(mask_video_id=ids()["mask_video_id"], window_start_sec=0.0,
                             blend_dilation_stage1=5, blend_dilation_stage2=2))


def plan_oi(argf, pos_hit):
    return [
        dict(label="1", cond="ON", args=argf(True), te=pos_hit[0], eq=[]),
        dict(label="2", cond="ON (same)", args=argf(True), te=pos_hit[1], eq=[("1", "mp4"), ("1", "csha")], hitphases=True),
        dict(label="3", cond="OFF", args=argf(False), te=None, eq=[("1", "mp4"), ("2", "mp4")]),
    ]


def plan_g3():
    return [dict(j, size=[1536, 896]) for j in plan_oi(outpaint_args, [{"positive": "miss"}, {"positive": "hit"}])]


def plan_g4():
    return plan_oi(inpaint_body, [{"positive": "miss"}, {"positive": "hit"}])


def plan_g7():
    return [dict(j, size=[1536, 896]) for j in plan_oi(outpaint_args, ["miss", "hit"])]


GATES = {
    "g1": dict(plan=plan_g1, tool="submit_generate", chain=False, base="LTX23"),
    "g2": dict(plan=plan_g2, tool="submit_chain", chain=True, base="LTX23"),
    "g5": dict(plan=plan_g5, tool="submit_generate", chain=False, base="LTX25"),
    "g6": dict(plan=plan_g6, tool="submit_chain", chain=True, base="LTX25"),
    "g3": dict(plan=plan_g3, tool="submit_generate", chain=False, base="LTX23"),
    "g4": dict(plan=plan_g4, tool="api_generate", chain=False, base="LTX23"),
    "g7": dict(plan=plan_g7, tool="submit_generate", chain=False, base="LTX25"),
}

MEM_RE = re.compile(r"rss|commit|ram|working_set|pagefile|host_mem|memory", re.I)


def summarize(meta: dict, mp4: Path, base: str, chain: bool) -> dict:
    ltx23 = meta.get("ltx23")
    ltx25 = meta.get("ltx25") or {}
    out = {
        "mp4_sha256": sha256_file(mp4), "mp4_size": mp4.stat().st_size, "mp4_rel": rel(mp4),
        "backend": meta.get("backend"), "attention_used": meta.get("attention_used"),
        "generation_time_seconds": meta.get("generation_time_seconds"),
        "models": meta.get("models"),
        "request_embed_mp4_metadata": (meta.get("request") or {}).get("embed_mp4_metadata"),
        "request_keep_text_conditioning": (meta.get("request") or {}).get("keep_text_conditioning"),
        "request_keep_resident": (meta.get("request") or {}).get("keep_resident"),
        "request_keep_resident_embeddings": (meta.get("request") or {}).get("keep_resident_embeddings"),
        "request_seed": (meta.get("request") or {}).get("seed"),
        "request_outpaint": (meta.get("request") or {}).get("outpaint"),
        "request_inpaint": (meta.get("request") or {}).get("inpaint"),
        "request_loras": (meta.get("request") or {}).get("loras"),
        "request_num_frames": (meta.get("request") or {}).get("num_frames"),
        "request_prompt": (meta.get("request") or {}).get("prompt"),
        "request_size": [(meta.get("request") or {}).get("width"), (meta.get("request") or {}).get("height")],
        "ltx23_top": ltx23, "has_ltx23_key": "ltx23" in meta,
        "te_cache_paths": [p for p, _ in find_paths(meta, lambda k: k == "te_cache")],
        "csha_paths": [p for p, _ in find_paths(meta, lambda k: k == "conditioning_sha256")],
        "keep_resident_fields": find_paths(meta, lambda k: k.startswith("keep_resident") or k.startswith("block_swap")),
        "memory_fields": [x for x in find_paths(meta, lambda k: bool(MEM_RE.search(k)) and "vram" not in k.lower())],
        "ltx25_te_cache": ltx25.get("te_cache"), "ltx25_csha": ltx25.get("conditioning_sha256"),
        "ltx25_phases": ltx25.get("phases"),
    }
    if base == "LTX25":
        ph = ltx25.get("phases") or {}
        out["phase_seconds"] = {k: ph.get(k) for k in ("10_prompt_encode", "10a_te_build", "10b_ep_build")}
    # held amount at job end
    if base == "LTX23" and ltx23:
        tc = ltx23["te_cache"]
        strings = ((len(tc["clips"]) + 1) if chain else 1) + (1 if "negative" in tc else 0)
        out["held"] = {"strings": strings, "bytes": strings * B23}
    elif base == "LTX25" and out["ltx25_csha"] is not None:
        n = len(out["ltx25_csha"])
        out["held"] = {"bundles": 1, "entries": n, "bytes": n * B25}
    else:
        out["held"] = {"strings": 0, "bytes": 0}
    return out


def check_job(job, s, base, chain, done):
    pr = []
    if s["request_embed_mp4_metadata"] is not False:
        pr.append(f"embed_mp4_metadata {s['request_embed_mp4_metadata']}")
    want = bool(job["args"].get("keep_text_conditioning"))
    if bool(s["request_keep_text_conditioning"]) != want:
        pr.append(f"request.keep_text_conditioning {s['request_keep_text_conditioning']} != {want}")
    if s["request_seed"] != 42:
        pr.append(f"seed {s['request_seed']}")
    if s["request_size"] != job.get("size", [1280, 768]):
        pr.append(f"size {s['request_size']}")
    exp = job["te"]
    if base == "LTX23":
        if exp is None:
            if s["has_ltx23_key"]:
                pr.append("OFF job has ltx23 key")
            if s["te_cache_paths"] or s["csha_paths"]:
                pr.append(f"OFF job has te_cache/csha: {s['te_cache_paths']} {s['csha_paths']}")
        else:
            l23 = s["ltx23_top"]
            if not l23:
                pr.append("ltx23 missing in ON job")
            else:
                got = l23.get("te_cache")
                if not isinstance(got, dict) or any(got.get(k) != v for k, v in exp.items()):
                    pr.append(f"te_cache got {got}, expected {exp}")
                if not l23.get("conditioning_sha256"):
                    pr.append("conditioning_sha256 missing")
    else:
        if exp is None:
            if s["te_cache_paths"] or s["csha_paths"]:
                pr.append(f"OFF job has te_cache/csha: {s['te_cache_paths']} {s['csha_paths']}")
        else:
            if s["ltx25_te_cache"] != exp:
                pr.append(f"te_cache got {s['ltx25_te_cache']}, expected {exp}")
            if not s["ltx25_csha"]:
                pr.append("conditioning_sha256 missing")
    if job.get("hitphases") and base == "LTX25":
        ph = s["ltx25_phases"] or {}
        if "10a_te_build" in ph or "10b_ep_build" in ph:
            pr.append("hit job has 10a_te_build/10b_ep_build phases")
        if ((ph.get("10_prompt_encode") or {}).get("seconds", 99)) > 1.0:
            pr.append(f"hit job 10_prompt_encode {(ph.get('10_prompt_encode') or {}).get('seconds')}")
    # job 8 (G1) / 7 (G5): resident actually effective is only recorded, not gated
    for ref, what in job["eq"]:
        o = done.get(ref)
        if o is None:
            pr.append(f"comparison target {ref} missing")
            continue
        os_ = o["summary"]
        if what == "mp4":
            if s["mp4_sha256"] != os_["mp4_sha256"]:
                pr.append(f"mp4 differs from job {ref}: {s['mp4_sha256'][:8]} vs {os_['mp4_sha256'][:8]}")
        elif what == "csha":
            a = (s["ltx23_top"] or {}).get("conditioning_sha256") if base == "LTX23" else s["ltx25_csha"]
            b = (os_["ltx23_top"] or {}).get("conditioning_sha256") if base == "LTX23" else os_["ltx25_csha"]
            if a != b:
                pr.append(f"conditioning_sha256 differs from job {ref}")
    return pr


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
        e = {"tool": name, "args": args or {}, "is_error": bool(r.isError), "seconds": round(time.monotonic() - t0, 2),
             "at": now()}
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


async def api_generate(body):
    from mcp_server.client import BackendClient
    from mcp_server.settings import load_settings
    c = BackendClient(load_settings())
    try:
        return await c.post_json("/generate", body, timeout=120)
    finally:
        await c.aclose()


async def wait_done(m, job_id, unreach, limit=14400):
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


async def wait_loaded(m, want_base, limit=1800):
    t0 = time.monotonic()
    while True:
        st = await m.call("backend_status")
        s = st.get("status") or {}
        print(f"  load: state={s.get('state')} loaded={s.get('pipeline_loaded')} t={time.monotonic() - t0:.0f}s", flush=True)
        if s.get("pipeline_loaded") and s.get("state") in ("ready", "idle"):
            lm = await m.call("list_models")
            tr = (lm.get("categories") or {}).get("transformer") or {}
            return {"status": s, "active_base_model": lm.get("active_base_model"), "transformer_active": tr.get("active"),
                    "categories_active": {k: (v or {}).get("active") for k, v in (lm.get("categories") or {}).items()}}
        if s.get("state") == "error" or time.monotonic() - t0 > limit:
            raise RuntimeError(f"pipeline did not load: {s}")
        await asyncio.sleep(10)


def phase_switch(base: str) -> int:
    rec = {"started": now(), "want": base}

    async def body(m):
        rec["models_before"] = await m.call("list_models")
        rec["status_before"] = await m.call("backend_status")
        rec["load_call"] = await m.call("load_pipeline", {"base_model": base, "wait_sec": 45})
        rec["loaded"] = await wait_loaded(m, base)

    try:
        asyncio.run(with_mcp(body, f"switch_{base}"))
        rec["ok"] = (rec.get("loaded") or {}).get("active_base_model") == base
    except BaseException as exc:  # noqa: BLE001
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["models_before"] = {"active_base_model": rec.get("models_before", {}).get("active_base_model")}
    write_json(RUNS / f"switch{SUF}_{base}.json", rec)
    print("switch", base, rec["ok"], rec.get("error"), (rec.get("loaded") or {}).get("transformer_active"))
    return 0 if rec["ok"] else 1


def phase_restore(base: str, transformer: str) -> int:
    """Put the transformer selection back through the server's own entry (load_pipeline models=...)."""
    rec = {"started": now(), "base": base, "transformer": transformer}

    async def body(m):
        lm = await m.call("list_models")
        rec["before"] = {"active_base_model": lm.get("active_base_model"),
                         "transformer_active": ((lm.get("categories") or {}).get("transformer") or {}).get("active"),
                         "transformer_entries": [e.get("name") for e in ((lm.get("categories") or {}).get("transformer") or {}).get("entries", [])]}
        rec["load_call"] = await m.call("load_pipeline", {"models": {"transformer": transformer}, "base_model": base,
                                                             "wait_sec": 45})
        rec["loaded"] = await wait_loaded(m, base)

    try:
        asyncio.run(with_mcp(body, "restore"))
        rec["ok"] = (rec.get("loaded") or {}).get("transformer_active") == transformer
    except BaseException as exc:  # noqa: BLE001
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
    write_json(RUNS / f"restore{SUF}.json", rec)
    print("restore", rec["ok"], rec.get("error"), rec.get("before"), (rec.get("loaded") or {}).get("transformer_active"))
    return 0 if rec["ok"] else 1


def phase_gate(g: str) -> int:
    spec = GATES[g]
    out = RUNS / f"{g}.json"
    rec = read_json(out) if out.is_file() else {"gate": g, "started": now(), "jobs": {},
                                                   "unreachable": {"count": 0, "at": []}}
    rec.setdefault("resumes", []).append(now())
    rec.pop("stopped", None)
    unreach = rec["unreachable"]
    chain, base = spec["chain"], spec["base"]

    async def body(m):
        if "pre" not in rec:
            lm = await m.call("list_models")
            tr = (lm.get("categories") or {}).get("transformer") or {}
            rec["pre"] = {"active_base_model": lm.get("active_base_model"), "transformer_active": tr.get("active"),
                          "categories_active": {k: (v or {}).get("active") for k, v in (lm.get("categories") or {}).items()},
                          "status": (await m.call("backend_status")).get("status")}
            if lm.get("active_base_model") != base:
                raise RuntimeError(f"active base model is {lm.get('active_base_model')}, want {base}")
            if not (rec["pre"]["status"] or {}).get("pipeline_loaded"):
                raise RuntimeError("pipeline not loaded")
            write_json(out, rec)
        for job in spec["plan"]():
            label = job["label"]
            ent = rec["jobs"].setdefault(label, {})
            if ent.get("summary") and ent.get("checked_ok"):
                continue
            ent.update(label=label, cond=job["cond"], args=job["args"])
            t0 = time.monotonic()
            if not ent.get("job_id"):
                sub = await (api_generate(job["args"]) if spec["tool"] == "api_generate"
                             else m.call(spec["tool"], job["args"]))
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
            d = REPO / "outputs" / jid
            meta = read_json(d / "metadata.json")
            write_json(RUNS / f"{g}_{label}.metadata.json", meta)
            mp4 = d / "output.mp4"
            ent["summary"] = summarize(meta, mp4, base, chain)
            ent["mp4_probe"] = ffprobe(mp4)
            done = {k: v for k, v in rec["jobs"].items() if v.get("summary") and v.get("checked_ok")}
            problems = check_job(job, ent["summary"], base, chain, done)
            ent["problems"] = problems
            ent["checked_ok"] = not problems
            sm = ent["summary"]
            te = (sm["ltx23_top"] or {}).get("te_cache") if base == "LTX23" else sm["ltx25_te_cache"]
            print(f"{g}-{label}: mp4 {sm['mp4_sha256'][:8]} te={te} gen={sm['generation_time_seconds']} "
                  f"phases={sm.get('phase_seconds')} problems={problems}", flush=True)
            write_json(out, rec)
            if problems:
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
              all((rec["jobs"].get(j["label"]) or {}).get("checked_ok") for j in spec["plan"]()))
    rec["gate_pass"] = bool(all_ok)
    write_json(out, rec)
    write_json(RUNS / f"verdict_{g}.json", {
        "gate": g, "pass": bool(all_ok), "stopped": rec.get("stopped"), "error": rec.get("error"),
        "jobs": {k: {"job_id": v.get("job_id"), "cond": v.get("cond"), "checked_ok": v.get("checked_ok"),
                     "problems": v.get("problems")} for k, v in rec["jobs"].items()},
        "unreachable": rec["unreachable"]})
    print("GATE", g, "PASS" if all_ok else "NOT PASSED", flush=True)
    return 0 if all_ok else 1


def phase_upload() -> int:
    """Upload the stand-in source video and the mask through the MCP upload_video tool; record ids and hashes."""
    rec = {"started": now()}
    src, mask = REPO / SRC_REL, REPO / MASK_REL
    for k, f in (("source", src), ("mask", mask)):
        rec[f"{k}_rel"] = rel(f)
        rec[f"{k}_sha256"] = sha256_file(f)
        rec[f"{k}_probe"] = ffprobe(f)

    async def body(m):
        for k, f in (("source", src), ("mask", mask)):
            r = await m.call("upload_video", {"file_path": str(f)})
            rec[f"{k}_video_id"] = r.get("video_id")
            rec[f"{k}_upload"] = {x: r.get(x) for x in ("original_filename", "content_type", "size_bytes", "trimmed")}

    try:
        asyncio.run(with_mcp(body, "upload"))
        rec["ok"] = bool(rec.get("source_video_id") and rec.get("mask_video_id"))
    except BaseException as exc:  # noqa: BLE001
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
    write_json(RUNS / "uploads2.json", rec)
    print("upload", rec["ok"], rec.get("error"), rec.get("source_video_id"), rec.get("mask_video_id"))
    return 0 if rec["ok"] else 1


def phase_unload() -> int:
    rec = {"started": now()}

    async def body(m):
        rec["unload"] = await m.call("unload_pipeline")
        rec["status_after"] = await m.call("backend_status")
        rec["models_after"] = {"active_base_model": (await m.call("list_models")).get("active_base_model")}

    try:
        asyncio.run(with_mcp(body, "unload"))
        rec["ok"] = ((rec.get("status_after") or {}).get("status") or {}).get("state") == "unloaded"
    except BaseException as exc:  # noqa: BLE001
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
    write_json(RUNS / f"unload{SUF}.json", rec)
    print("unload", rec.get("ok"), rec.get("error"))
    return 0 if rec["ok"] else 1


def selfcheck() -> int:
    import functools
    import anyio
    import httpx
    sys.path.insert(0, str(REPO))
    from api.models import GenerateChainRequest, GenerateRequest
    from mcp_server.client import BackendClient, set_client
    from mcp_server.params import ChainClipArg
    from mcp_server.settings import Settings
    from mcp_server.tools import generate
    problems, cap = [], []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={})
        cap.append((request.url.path, json.loads(request.content)))
        return httpx.Response(202, json={"job_id": "j1", "status": "queued", "created_at": "t", "num_clips": 3})

    set_client(BackendClient(Settings(base_url="http://testserver", api_key=None, output_dir=Path(".")),
                             transport=httpx.MockTransport(handler)))
    bodies = {}
    for g, spec in GATES.items():
        for job in spec["plan"]():
            a = dict(job["args"])
            if "clips" in a:
                a["clips"] = [ChainClipArg(**c) for c in a["clips"]]
            n0 = len(cap)
            if spec["tool"] == "api_generate":
                path, body = "/generate (direct)", a
            else:
                anyio.run(functools.partial(getattr(generate, spec["tool"]), **a))
                path, body = cap[n0]
            model = GenerateChainRequest if spec["chain"] else GenerateRequest
            try:
                req = model.model_validate(body)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{g}-{job['label']}: {type(exc).__name__}: {str(exc)[:300]}")
                continue
            want = bool(job["args"].get("keep_text_conditioning"))
            if bool(body.get("keep_text_conditioning")) != want:
                problems.append(f"{g}-{job['label']}: keep body={body.get('keep_text_conditioning')}")
            sz = job.get("size", [1280, 768])
            for k, w in (("seed", 42), ("width", sz[0]), ("height", sz[1]), ("embed_mp4_metadata", False)):
                if body.get(k) != w:
                    problems.append(f"{g}-{job['label']}: {k}={body.get(k)}")
            bodies[f"{g}-{job['label']}"] = {"path": path, "body": body}
    write_json(RUNS / f"selfcheck{SUF}.json", {"at": now(), "bodies": bodies, "problems": problems, "ok": not problems})
    print(json.dumps({"problems": problems, "n": len(bodies)}, ensure_ascii=False, indent=1))
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    c = a[0]
    if c == "selfcheck":
        sys.exit(selfcheck())
    if c == "switch":
        sys.exit(phase_switch(a[1]))
    if c == "restore":
        sys.exit(phase_restore(a[1], a[2]))
    if c in GATES:
        sys.exit(phase_gate(c))
    if c == "upload":
        sys.exit(phase_upload())
    if c == "unload":
        sys.exit(phase_unload())
    raise SystemExit("unknown")
