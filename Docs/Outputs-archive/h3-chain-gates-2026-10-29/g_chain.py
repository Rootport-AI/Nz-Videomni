"""Section 1-106 MiniMax H3 Chained: the GPU gates G-C0 / G-C1 / G-C2 through the product's HTTP API.

An HTTP client for the running Nz-Videomni backend (a real backend: MiniMax H3 is loaded by
this script through ``POST /pipeline/load``). It does not import the product. One gate = one
process (``run_gates.ps1``)::

    python g_chain.py G-C0 --compare <c_chain report json> --report G-C0.json
    python g_chain.py G-C1 --report G-C1.json
    python g_chain.py G-C2 --report G-C2.json
    python g_chain.py cleanup --report cleanup.json      # unload the pipeline (the end of the run)
    python g_chain.py --selfcheck                         # CPU only, no HTTP

What a gate does: (a) ``GET /status`` (recorded), (b) ``POST /pipeline/load`` with
``base_model`` MiniMaxH3 and the FL2VA transformer by name (``timeout=None``: a load can take
minutes), (c) ``POST /generate/chain``, (d) ``GET /jobs/{id}`` every 5 s (progress / stage / clip
trace, monotonicity check), (e) after the job: ``outputs/<job_id>/metadata.json``, the mp4's
size / sha256 / frame count, ``metadata["h3"]`` (clips / seeds / vram16_mode_used / loras /
phases), and the verdict.

Exit codes: 0 = every criterion passed, 1 = the run failed (exception, job failed, timeout, the
backend is not real), 2 = the run finished but a criterion failed (G-C0: the latents' sha256
differ from the measurement script's).

The sha256 of a prompt is the product's: ``sha256(clip_prompt.encode("utf-8"))`` where the clip
prompt is ``" ".join(non-empty(shared.strip(), own.strip()))`` (engine_h3/chain_h3.py
``_sha_text``; api/models.py ``compose_clip_prompt``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

BASE_MODEL = "MiniMaxH3"
FL2VA = "default"  # the registry name of MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors (manifest default_file; the transformer c_chain.py measured with)
TURBO = "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16"  # models/MiniMaxH3/LoRA (the registered name is the file stem)
REAL_BACKEND_H3 = "minimax-h3"
DEFAULT_PROMPTS = REPO / "Docs" / "Outputs-archive" / "h3-chain-2026-10-29" / "prompts_c.json"
CLIP_FRAMES = 124
OVERLAP_FRAMES = 22
TERMINAL = ("completed", "failed", "cancelled")

#: gate -> its fixed conditions (plan sec. 7). ``own_prompts``: True = clips 1.. carry the appended part.
GATES: dict[str, dict] = {
    "G-C0": dict(width=640, height=384, clips=3, attention="sdpa", steps=20, shift_video=12.0, shift_audio=3.0,
                 vram16_mode=None, embed_mp4_metadata=False, loras=[], frames=328, job_timeout=1450),
    "G-C1": dict(width=1344, height=768, clips=3, attention="sage", steps=8, shift_video=6.0, shift_audio=3.0,
                 vram16_mode=True, embed_mp4_metadata=None, loras=[{"name": TURBO, "strength": 0.8}], frames=328,
                 job_timeout=2350),
    "G-C2": dict(width=1344, height=768, clips=13, attention="sage", steps=20, shift_video=12.0, shift_audio=3.0,
                 vram16_mode=None, embed_mp4_metadata=None, loras=[], frames=1348, job_timeout=10900),
}


# ---------------------------------------------------------------------------------------------
# Pure helpers (the self-check exercises these).
# ---------------------------------------------------------------------------------------------

def sha_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def compose_clip_prompt(shared: str, own: str | None) -> str:
    """api/models.py ``compose_clip_prompt``: both ends trimmed, non-empty parts joined by one space."""
    parts = (shared.strip(), (own or "").strip())
    return " ".join(part for part in parts if part)


def split_prompts(prompts: list[str]) -> tuple[str, list[str]]:
    """prompts_c.json (the full 3 clip prompts) -> (shared prompt, appended part per clip).

    Every later prompt must be the first one plus an appended sentence; the appended part is what
    ``clips[i].prompt`` carries, so the product composes the same strings back.
    """
    if not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
        raise ValueError("the prompts file must be a non-empty JSON list of non-empty strings")
    shared = prompts[0]
    owns = [""]
    for i, text in enumerate(prompts[1:], start=1):
        if not text.startswith(shared):
            raise ValueError(f"prompts[{i}] does not start with prompts[0] (the shared prompt)")
        own = text[len(shared):]
        owns.append(own.strip())
    for i, text in enumerate(prompts):
        if compose_clip_prompt(shared, owns[i]) != text:
            raise ValueError(f"prompts[{i}] cannot be recomposed from the shared part and its appended part")
    return shared, owns


def expected_clip_prompts(prompts: list[str], n_clips: int) -> tuple[str, list[str], list[str]]:
    """(shared, own per clip, the composed prompt per clip) for ``n_clips`` clips.

    Clip 0 is the shared prompt alone. Clips 1.. cycle through the appended parts of the prompts file
    (3 clips = prompts_c.json as is; 13 clips alternate the two appended sentences).
    """
    shared, owns = split_prompts(prompts)
    appended = [o for o in owns[1:] if o] or [""]
    own_per_clip = [""] + [appended[(i - 1) % len(appended)] for i in range(1, n_clips)]
    return shared, own_per_clip, [compose_clip_prompt(shared, o) for o in own_per_clip]


def build_request(gate: str, prompts: list[str], seed: int = 42) -> dict:
    """The ``POST /generate/chain`` body of a gate (only the fields the gate pins; the rest are the API defaults)."""
    cond = GATES[gate]
    shared, owns, _ = expected_clip_prompts(prompts, cond["clips"])
    clips = []
    for own in owns:
        clip: dict = {"num_frames": CLIP_FRAMES}
        if own:
            clip["prompt"] = own
        clips.append(clip)
    body: dict = {
        "prompt": shared,
        "width": cond["width"],
        "height": cond["height"],
        "seed": seed,
        "attention_backend": cond["attention"],
        "num_inference_steps": cond["steps"],
        "shift_video": cond["shift_video"],
        "shift_audio": cond["shift_audio"],
        "clips": clips,
    }
    if cond["vram16_mode"] is not None:
        body["vram16_mode"] = cond["vram16_mode"]
    if cond["embed_mp4_metadata"] is not None:
        body["embed_mp4_metadata"] = cond["embed_mp4_metadata"]
    if cond["loras"]:
        body["loras"] = [dict(spec) for spec in cond["loras"]]
    return body


def expected_frames(n_clips: int, clip_frames: int = CLIP_FRAMES, overlap: int = OVERLAP_FRAMES) -> int:
    return n_clips * clip_frames - (n_clips - 1) * overlap


def check_monotonic(samples: list[dict]) -> tuple[bool, list[str]]:
    """The job trace: ``progress`` never goes down and stays in [0, 1]; ``clip`` never goes down;
    ``clip_count`` does not change once set. ``samples`` are dicts with progress / clip / clip_count."""
    problems: list[str] = []
    last_progress = None
    last_clip = None
    count = None
    for i, s in enumerate(samples):
        p = s.get("progress")
        if p is not None:
            if not (-1e-9 <= p <= 1.0 + 1e-9):
                problems.append(f"sample {i}: progress {p} out of [0, 1]")
            if last_progress is not None and p < last_progress - 1e-9:
                problems.append(f"sample {i}: progress went down {last_progress} -> {p}")
            last_progress = p if last_progress is None else max(last_progress, p)
        c = s.get("clip")
        if c is not None:
            if last_clip is not None and c < last_clip:
                problems.append(f"sample {i}: clip went down {last_clip} -> {c}")
            last_clip = c if last_clip is None else max(last_clip, c)
        cc = s.get("clip_count")
        if cc is not None:
            if count is not None and cc != count:
                problems.append(f"sample {i}: clip_count changed {count} -> {cc}")
            count = cc if count is None else count
    return (not problems), problems


def compare_latents(product_clips: list[dict], reference_clips: list[dict]) -> list[str]:
    """G-C0's first criterion: the product's ``clips[i].latents_sha256`` ({"video", "audio"}) against the
    script's ``clips[i].latents`` ({"v_sha256", "a_sha256"}). Returns the mismatches (empty = all equal)."""
    out: list[str] = []
    if len(product_clips) != len(reference_clips):
        out.append(f"clip count: product {len(product_clips)}, script {len(reference_clips)}")
    for i, (p, r) in enumerate(zip(product_clips, reference_clips)):
        ps = (p or {}).get("latents_sha256") or {}
        rs = (r or {}).get("latents") or {}
        for kind, pk, rk in (("video", "video", "v_sha256"), ("audio", "audio", "a_sha256")):
            if not ps.get(pk) or not rs.get(rk):
                out.append(f"clips[{i}] {kind}: sha256 missing (product {ps.get(pk)!r}, script {rs.get(rk)!r})")
            elif ps[pk] != rs[rk]:
                out.append(f"clips[{i}] {kind}: product {ps[pk][:12]}.. != script {rs[rk][:12]}..")
    return out


def check_prompts(metadata: dict, expected: list[str], reference_clips: list[dict] | None) -> list[str]:
    """The composed prompts reached the engine unchanged: the request's shared + appended parts recompose to the
    expected strings, and the product's ``clips[i].prompt_sha256`` (and the script's ``prompt_sha``) equal
    sha256(expected[i])."""
    out: list[str] = []
    req = metadata.get("request") or {}
    req_clips = req.get("clips") or []
    shared = req.get("prompt", "")
    if len(req_clips) != len(expected):
        out.append(f"request has {len(req_clips)} clips, expected {len(expected)}")
    for i, text in enumerate(expected[: len(req_clips)]):
        composed = compose_clip_prompt(shared, req_clips[i].get("prompt"))
        if composed != text:
            out.append(f"request clips[{i}] composes to a different string than expected")
    h3_clips = (metadata.get("h3") or {}).get("clips") or []
    for i, text in enumerate(expected):
        if i < len(h3_clips) and (h3_clips[i] or {}).get("prompt_sha256") != sha_text(text):
            out.append(f"h3.clips[{i}].prompt_sha256 != sha256(expected)")
        if reference_clips and i < len(reference_clips):
            ref = reference_clips[i].get("prompt_sha")
            if ref is not None and ref != sha_text(text):
                out.append(f"script clips[{i}].prompt_sha != sha256(expected)")
    return out


def psnr_db(a, b) -> float:
    import numpy as np
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return math.inf if mse == 0.0 else 10.0 * math.log10(255.0 * 255.0 / mse)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def count_frames(path: Path) -> dict:
    """The mp4's decodable video frame count (PyAV; ffprobe when PyAV is missing)."""
    try:
        import av
    except ImportError:
        av = None
    if av is not None:
        n = 0
        with av.open(str(path)) as c:
            stream = c.streams.video[0]
            for _ in c.decode(stream):
                n += 1
            audio = c.streams.audio[0] if c.streams.audio else None
            rate = float(stream.average_rate) if stream.average_rate else None
        return {"frames": n, "frame_rate": rate, "has_audio": audio is not None, "reader": "av"}
    ffprobe = shutil.which("ffprobe") or str(REPO / "tools" / "ffmpeg" / "bin" / "ffprobe.exe")
    r = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return {"frames": int(r.stdout.strip()), "frame_rate": None, "has_audio": None, "reader": "ffprobe"}


def mp4_psnr(a: Path, b: Path) -> dict:
    """Per-frame PSNR (RGB) between two mp4s, frame by frame (streaming). Needs PyAV."""
    import av
    vals: list[float] = []
    with av.open(str(a)) as ca, av.open(str(b)) as cb:
        fa = ca.decode(ca.streams.video[0])
        fb = cb.decode(cb.streams.video[0])
        n_a = n_b = 0
        for x, y in zip(fa, fb):
            vals.append(psnr_db(x.to_ndarray(format="rgb24"), y.to_ndarray(format="rgb24")))
        for _ in fa:
            n_a += 1
        for _ in fb:
            n_b += 1
    finite = [v for v in vals if math.isfinite(v)]
    return {
        "frames_compared": len(vals),
        "extra_frames_a": n_a, "extra_frames_b": n_b,
        "identical_frames": len(vals) - len(finite),
        "min_psnr": min(finite) if finite else None,
        "mean_psnr": (sum(finite) / len(finite)) if finite else None,
        "all_identical": bool(vals) and not finite and not n_a and not n_b,
    }


# ---------------------------------------------------------------------------------------------
# Main-memory sampler (Windows; ctypes only: the venvs have no psutil).
# ---------------------------------------------------------------------------------------------

class ProcSampler(threading.Thread):
    """Every ``interval`` s: the summed working set and private bytes (the commit) of the process TREE under
    ``root_pid`` (the backend and its H3 worker). Keeps the maxima and a thin trace."""

    def __init__(self, root_pid: int, interval: float = 10.0) -> None:
        super().__init__(daemon=True)
        self.root_pid = root_pid
        self.interval = interval
        self.stop_event = threading.Event()
        self.max_working_set_gib = 0.0
        self.max_commit_gib = 0.0
        self.trace: list[list[float]] = []
        self.error: str | None = None
        self._t0 = time.time()

    def tree(self) -> list[int]:
        import ctypes
        from ctypes import wintypes

        class ENTRY(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                        ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                        ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                        ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD),
                        ("szExeFile", wintypes.WCHAR * 260)]

        k32 = ctypes.windll.kernel32
        k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        snap = k32.CreateToolhelp32Snapshot(0x2, 0)
        children: dict[int, list[int]] = {}
        entry = ENTRY()
        entry.dwSize = ctypes.sizeof(ENTRY)
        ok = k32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            children.setdefault(entry.th32ParentProcessID, []).append(entry.th32ProcessID)
            ok = k32.Process32NextW(snap, ctypes.byref(entry))
        k32.CloseHandle(snap)
        out, todo = [], [self.root_pid]
        while todo:
            pid = todo.pop()
            out.append(pid)
            todo.extend(children.get(pid, []))
        return out

    @staticmethod
    def memory(pid: int) -> tuple[int, int] | None:
        import ctypes
        from ctypes import wintypes

        class COUNTERS(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]

        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = wintypes.HANDLE
        handle = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            c = COUNTERS()
            c.cb = ctypes.sizeof(COUNTERS)
            if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(c), c.cb):
                return None
            return int(c.WorkingSetSize), int(c.PrivateUsage)
        finally:
            k32.CloseHandle(handle)

    def sample_once(self) -> tuple[float, float]:
        ws = commit = 0
        for pid in self.tree():
            m = self.memory(pid)
            if m:
                ws += m[0]
                commit += m[1]
        ws_g, c_g = ws / 2**30, commit / 2**30
        self.max_working_set_gib = max(self.max_working_set_gib, ws_g)
        self.max_commit_gib = max(self.max_commit_gib, c_g)
        return ws_g, c_g

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                ws_g, c_g = self.sample_once()
                self.trace.append([round(time.time() - self._t0, 1), round(ws_g, 2), round(c_g, 2)])
                self.stop_event.wait(self.interval)
        except Exception as exc:   # a sampler must never fail a gate
            self.error = f"{type(exc).__name__}: {exc}"

    def finish(self) -> dict:
        self.stop_event.set()
        self.join(timeout=5)
        return {
            "root_pid": self.root_pid, "interval_s": self.interval, "samples": len(self.trace),
            "max_working_set_gib": round(self.max_working_set_gib, 2),
            "max_commit_gib": round(self.max_commit_gib, 2),
            "trace": self.trace[:: max(1, len(self.trace) // 400)],   # at most ~400 points
            "error": self.error,
        }


# ---------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------

class Api:
    def __init__(self, base_url: str, api_key: str | None) -> None:
        import httpx
        self.httpx = httpx
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=30.0)

    def get(self, path: str, **kw):
        r = self.client.get(path, **kw)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, body: dict | None = None, timeout=30.0):
        r = self.client.post(path, json=body, timeout=timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"POST {path} -> HTTP {r.status_code}: {r.text[:2000]}")
        return r.json()


def poll_job(api: Api, job_id: str, poll: float, timeout: float, trace_every: float = 60.0) -> dict:
    """GET /jobs/{id} every ``poll`` seconds until a terminal status. Keeps a sample whenever (progress, stage,
    clip, clip_count, current_step) changes, plus one at least every ``trace_every`` s. The monotonicity check
    runs on EVERY poll (``all_polls``), not only on the kept samples."""
    started = time.time()
    samples: list[dict] = []
    stage_log: list[dict] = []
    last_key = None
    last_kept = 0.0
    last_stage = object()
    all_polls = 0
    monotonic_problems: list[str] = []
    last_p = None
    last_c = None
    count = None
    errors = 0
    job: dict = {}
    while True:
        elapsed = time.time() - started
        try:
            job = api.get(f"/jobs/{job_id}")
            errors = 0
        except Exception as exc:   # a transient failure must not end a 3 hour gate
            errors += 1
            if errors >= 12:
                raise RuntimeError(f"GET /jobs/{job_id} failed {errors} times in a row: {exc}")
            time.sleep(poll)
            continue
        all_polls += 1
        s = {"t": round(elapsed, 1), "status": job.get("status"), "progress": job.get("progress"),
             "stage": job.get("stage"), "clip": job.get("clip"), "clip_count": job.get("clip_count"),
             "current_step": job.get("current_step"), "total_steps": job.get("total_steps")}
        ok, probs = check_monotonic([{"progress": last_p, "clip": last_c, "clip_count": count}, s])
        if not ok and len(monotonic_problems) < 50:
            monotonic_problems.extend(f"t={s['t']}: {p}" for p in probs)
        if s["progress"] is not None:
            last_p = s["progress"] if last_p is None else max(last_p, s["progress"])
        if s["clip"] is not None:
            last_c = s["clip"] if last_c is None else max(last_c, s["clip"])
        if s["clip_count"] is not None and count is None:
            count = s["clip_count"]
        key = (s["progress"], s["stage"], s["clip"], s["clip_count"], s["current_step"], s["status"])
        if key != last_key or elapsed - last_kept >= trace_every:
            samples.append(s)
            last_key, last_kept = key, elapsed
        if s["stage"] != last_stage:
            stage_log.append({"t": s["t"], "stage": s["stage"], "progress": s["progress"], "clip": s["clip"]})
            last_stage = s["stage"]
        print(f"[{s['t']:8.1f}s] {s['status']} progress={s['progress']} stage={s['stage']} "
              f"clip={s['clip']}/{s['clip_count']} step={s['current_step']}/{s['total_steps']}", flush=True)
        if s["status"] in TERMINAL:
            break
        if elapsed >= timeout:
            return {"timed_out": True, "job": job, "samples": samples, "stage_log": stage_log, "polls": all_polls,
                    "monotonic_ok": not monotonic_problems, "monotonic_problems": monotonic_problems,
                    "max_progress": last_p, "seconds": round(elapsed, 1)}
        time.sleep(poll)
    return {"timed_out": False, "job": job, "samples": samples, "stage_log": stage_log, "polls": all_polls,
            "monotonic_ok": not monotonic_problems, "monotonic_problems": monotonic_problems,
            "max_progress": last_p, "seconds": round(time.time() - started, 1)}


# ---------------------------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------------------------

def _check(checks: list[dict], name: str, ok: bool | None, detail="", kind: str = "criterion") -> None:
    checks.append({"name": name, "kind": kind, "ok": ok, "detail": detail})


def vram_peak(meta_h3: dict) -> dict:
    phases = meta_h3.get("phases") or {}
    clip_entries = []
    for cp in phases.get("clips") or []:
        clip_entries.extend(v for v in cp.values() if isinstance(v, dict))
    decode = phases.get("decode") or {}
    alloc = [e.get("peak_vram_mb") for e in clip_entries + [decode] if e.get("peak_vram_mb") is not None]
    reserved = [e.get("peak_vram_reserved_mb") for e in clip_entries + [decode] if e.get("peak_vram_reserved_mb") is not None]
    return {"max_allocated_mb": max(alloc) if alloc else None, "max_reserved_mb": max(reserved) if reserved else None,
            "decode_allocated_mb": decode.get("peak_vram_mb"), "decode_reserved_mb": decode.get("peak_vram_reserved_mb")}


def evaluate(gate: str, ctx: dict) -> dict:
    """``ctx``: job (final GET /jobs/{id}), poll (poll_job's dict), metadata (metadata.json), mp4 (path facts),
    expected_prompts, reference (c_chain report or None), mem (sampler record or None). -> verdict dict."""
    cond = GATES[gate]
    n_clips = cond["clips"]
    checks: list[dict] = []
    job, poll, meta = ctx.get("job") or {}, ctx.get("poll") or {}, ctx.get("metadata") or {}
    h3 = meta.get("h3") or {}
    clips = h3.get("clips") or []
    mp4 = ctx.get("mp4") or {}

    _check(checks, "job completed", job.get("status") == "completed", f"status={job.get('status')} error={job.get('error')!r}")
    _check(checks, "backend is the real H3 engine", meta.get("backend") == REAL_BACKEND_H3, f"metadata.backend={meta.get('backend')!r}")
    _check(checks, "progress / clip trace is monotonic", poll.get("monotonic_ok"),
           "; ".join(poll.get("monotonic_problems") or []) or f"{poll.get('polls')} polls, max progress {poll.get('max_progress')}")
    final_ok = job.get("progress") == 1.0 or (job.get("progress") is not None and job["progress"] >= 1.0 - 1e-9)
    _check(checks, "final progress is 1.0", final_ok, f"progress={job.get('progress')}")
    _check(checks, "clip_count is the clip count", (job.get("clip_count") in (None, n_clips)) and
           (poll.get("max_progress") is not None), f"job.clip={job.get('clip')} clip_count={job.get('clip_count')}", kind="record")
    _check(checks, f"{n_clips} clip records in metadata.h3.clips", len(clips) == n_clips, f"{len(clips)} records")
    heads = [c.get("head_equals_prev_tail") for c in clips[1:]]
    _check(checks, "head_equals_prev_tail is true for every clip after the first",
           bool(heads) and all(h is True for h in heads), f"{heads}")
    frames = mp4.get("frames")
    _check(checks, f"mp4 plays and has {cond['frames']} frames", frames == cond["frames"],
           f"frames={frames} (expected {cond['frames']} = {n_clips}x{CLIP_FRAMES} - {n_clips - 1}x{OVERLAP_FRAMES}) reader={mp4.get('reader')}")
    seeds = h3.get("seeds")
    _check(checks, "seeds are seed + i", seeds == [42 + i for i in range(n_clips)], f"{seeds}")
    _check(checks, "attention used", meta.get("attention_used") == cond["attention"], f"{meta.get('attention_used')!r}")
    _check(checks, "VRAM 16GB mode used", meta.get("vram16_mode_used") == "on", f"{meta.get('vram16_mode_used')!r}")
    _check(checks, "steps / shifts as requested",
           (h3.get("num_inference_steps"), h3.get("shift_video"), h3.get("shift_audio")) ==
           (cond["steps"], cond["shift_video"], cond["shift_audio"]),
           f"{(h3.get('num_inference_steps'), h3.get('shift_video'), h3.get('shift_audio'))}")
    problems = check_prompts(meta, ctx.get("expected_prompts") or [], (ctx.get("reference") or {}).get("clips"))
    _check(checks, "composed prompts reach the engine unchanged", not problems, "; ".join(problems) or "equal")

    loras = h3.get("loras")
    if cond["loras"]:
        names = [(x or {}).get("name", "") for x in (loras or [])]
        applied = [(x or {}).get("pairs_applied") for x in (loras or [])]
        _check(checks, "h3.loras reports the Turbo LoRA", bool(loras) and len(loras) == len(cond["loras"]) and
               all(TURBO in n for n in names) and all(isinstance(a, int) and a > 0 for a in applied), f"{loras}")
    else:
        _check(checks, "no LoRA reported", not loras, f"{loras}")

    ref = ctx.get("reference")
    if gate == "G-C0":
        if ref is None:
            _check(checks, "FIRST CRITERION: clips[i].latents_sha256 equals the script's", False,
                   "no --compare report: the script's sha256 values are missing")
        else:
            mism = compare_latents(clips, ref.get("clips") or [])
            _check(checks, "FIRST CRITERION: clips[i].latents_sha256 equals the script's (video and audio, every clip)",
                   not mism, "; ".join(mism) or f"{len(clips)} clips x (video, audio) equal")
            ref_seeds = [c.get("seed") for c in ref.get("clips") or []]
            _check(checks, "seeds equal the script's", seeds == ref_seeds, f"product {seeds}, script {ref_seeds}")
            once = (ref.get("decode") or {}).get("once") or {}
            equal = bool(once.get("sha256")) and mp4.get("sha256") == once.get("sha256")
            detail = {"product_sha256": mp4.get("sha256"), "script_once_sha256": once.get("sha256"), "equal": equal,
                      "psnr": ctx.get("psnr")}
            _check(checks, "SECOND (record only): mp4 sha256 equals the script's once.mp4", None if not equal else True,
                   json.dumps(detail, ensure_ascii=False), kind="record")
        _check(checks, "embed_mp4_metadata was false", (meta.get("request") or {}).get("embed_mp4_metadata") is False,
               f"{(meta.get('request') or {}).get('embed_mp4_metadata')!r}")

    # records: always filled in, never a verdict
    seconds_clips = [(c.get("seconds") or {}) for c in clips]
    phases = h3.get("phases") or {}
    decode = phases.get("decode") or {}
    rec = {
        "job_seconds": meta.get("generation_time_seconds"),
        "per_clip_seconds": seconds_clips,
        "decode_seconds": decode.get("seconds"),
        "vram": vram_peak(h3),
        "worker_rss_peak_gib": h3.get("rss_peak_gib"),
        "worker_commit_peak_gib": h3.get("commit_peak_gib"),
        "decode_rss_process_peak_gib": decode.get("rss_process_peak_gib"),
        "decode_commit_process_peak_gib": decode.get("commit_process_peak_gib"),
        "sampler": {k: v for k, v in (ctx.get("mem") or {}).items() if k != "trace"} or None,
    }
    if gate == "G-C2":
        _check(checks, "decode seconds and VRAM recorded", decode.get("seconds") is not None and
               decode.get("peak_vram_mb") is not None, f"decode={ {k: decode.get(k) for k in ('seconds','peak_vram_mb','peak_vram_reserved_mb')} }")
        _check(checks, "main memory recorded (worker's own or the sampler's)",
               h3.get("commit_peak_gib") is not None or bool((ctx.get("mem") or {}).get("max_commit_gib")),
               f"worker commit_peak_gib={h3.get('commit_peak_gib')} sampler max_commit_gib={(ctx.get('mem') or {}).get('max_commit_gib')}")

    failed = [c for c in checks if c["kind"] == "criterion" and c["ok"] is not True]
    return {"gate": gate, "pass": not failed, "failed": [c["name"] for c in failed], "checks": checks, "records": rec}


# ---------------------------------------------------------------------------------------------
# Running a gate
# ---------------------------------------------------------------------------------------------

def repo_path(p: str | None) -> Path | None:
    if not p:
        return None
    path = Path(p)
    return path if path.is_absolute() else REPO / path


def run_gate(args) -> int:
    gate = args.gate
    cond = GATES[gate]
    report_path = Path(args.report) if args.report else HERE / f"{gate}.json"
    rec: dict = {"gate": gate, "conditions": {k: v for k, v in cond.items() if k != "job_timeout"},
                 "base_url": args.base_url, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "ok": False}
    exit_code = 1
    sampler = None
    api = None
    t0 = time.time()
    try:
        prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8-sig"))
        shared, owns, expected = expected_clip_prompts(prompts, cond["clips"])
        rec["prompts_file"] = str(args.prompts)
        rec["expected_prompts"] = expected
        rec["expected_prompt_sha256"] = [sha_text(t) for t in expected]
        body = build_request(gate, prompts, seed=args.seed)
        rec["request_body"] = body
        reference = None
        if args.compare:
            reference = json.loads(repo_path(args.compare).read_text(encoding="utf-8-sig"))
            rec["compare"] = {"path": str(args.compare), "name": reference.get("name"),
                              "condition": reference.get("condition")}
        if gate == "G-C0" and reference is not None:
            c = reference.get("condition") or {}
            diffs = [f"{k}: script {c.get(k)!r}" for k, v in (("width", 640), ("height", 384), ("seed", 42), ("steps", 20),
                     ("attention", "sdpa"), ("clips", 3), ("clip_frames", 124)) if k in c and c.get(k) != v]
            if reference.get("ok") is not True:
                diffs.append("the script's report is not ok")
            if diffs:
                raise RuntimeError("the --compare report was made under other conditions: " + "; ".join(diffs))

        api = Api(args.base_url, args.api_key)
        rec["status_before"] = api.get("/status")
        rec["models_before"] = _try(lambda: api.get("/models"))
        if args.backend_pid and sys.platform == "win32":
            sampler = ProcSampler(args.backend_pid, args.sample_interval)
            sampler.start()

        # (b) load. No timeout: a swap from another base model rebuilds the worker (minutes).
        load_body = {"base_model": BASE_MODEL, "models": {"transformer": args.transformer}}
        t_load = time.time()
        rec["load_response"] = api.post("/pipeline/load", load_body, timeout=None)
        rec["load_seconds"] = round(time.time() - t_load, 2)
        status = api.get("/status")
        rec["status_after_load"] = status
        if status.get("state") not in ("ready", "running") or status.get("base_model") != BASE_MODEL:
            raise RuntimeError(f"after the load: state={status.get('state')!r} base_model={status.get('base_model')!r}")
        if not (status.get("acceleration") or {}).get("sage_available"):
            raise RuntimeError("acceleration.sage_available is false: the backend looks like the mock (or sage is missing); "
                               "this gate needs a real backend")
        rec["models_after_load"] = _try(lambda: api.get("/models"))

        # (c) submit
        submitted = api.post("/generate/chain", body)
        job_id = submitted["job_id"]
        rec["job_id"] = job_id
        rec["submit_response"] = submitted
        print(f"job {job_id} submitted ({gate})", flush=True)

        # (d) poll
        timeout = args.job_timeout if args.job_timeout else cond["job_timeout"]
        poll = poll_job(api, job_id, args.poll, timeout)
        rec["poll"] = {k: v for k, v in poll.items() if k != "job"}
        rec["job"] = poll["job"]
        if sampler is not None:
            rec["memory_sampler"] = sampler.finish()
            sampler = None
        if poll["timed_out"]:
            raise RuntimeError(f"job {job_id} did not finish within {timeout} s (it keeps running in the backend; "
                               "stop it by stopping the backend)")
        if poll["job"].get("status") != "completed":
            raise RuntimeError(f"job {job_id} ended as {poll['job'].get('status')}: {poll['job'].get('error')}")

        # (e) artifacts
        out_dir = REPO / "outputs" / job_id
        meta_path = out_dir / "metadata.json"
        mp4_path = out_dir / "output.mp4"
        metadata = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        mp4 = {"path": str(mp4_path), "metadata_path": str(meta_path), "size_bytes": mp4_path.stat().st_size,
               "sha256": sha256_file(mp4_path)}
        mp4.update(count_frames(mp4_path))
        rec["metadata_path"] = str(meta_path)
        rec["mp4"] = mp4
        rec["metadata_h3"] = metadata.get("h3")
        rec["metadata_summary"] = {k: metadata.get(k) for k in (
            "backend", "attention_used", "vram16_mode_used", "block_swap_prefetch_used", "keep_resident_used",
            "peak_vram_reserved_mb", "generation_time_seconds", "seed_used", "models", "chain", "output")}
        rec["metadata_request"] = metadata.get("request")
        psnr = None
        if gate == "G-C0" and reference is not None:
            once = (reference.get("decode") or {}).get("once") or {}
            if once.get("path") and mp4["sha256"] != once.get("sha256"):
                try:
                    psnr = mp4_psnr(mp4_path, repo_path(once["path"]))
                except Exception as exc:   # PSNR is a record; it never fails the gate
                    psnr = {"error": f"{type(exc).__name__}: {exc}"}
            rec["mp4_psnr_vs_script_once"] = psnr
        verdict = evaluate(gate, {"job": poll["job"], "poll": poll, "metadata": metadata, "mp4": mp4,
                                  "expected_prompts": expected, "reference": reference, "psnr": psnr,
                                  "mem": rec.get("memory_sampler")})
        rec["verdict"] = verdict
        rec["ok"] = verdict["pass"]
        exit_code = 0 if verdict["pass"] else 2
        print(f"{gate}: {'PASS' if verdict['pass'] else 'FAIL'}  failed={verdict['failed']}", flush=True)
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
        print("ERROR:", rec["error"], flush=True)
        exit_code = 1
    finally:
        if sampler is not None:
            rec["memory_sampler"] = sampler.finish()
        if args.cleanup_after and api is not None:
            rec["cleanup"] = _try(lambda: api.post("/pipeline/unload", None, timeout=300.0))
        rec["seconds_total"] = round(time.time() - t0, 1)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return exit_code


def _try(fn):
    try:
        return fn()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_cleanup(args) -> int:
    """The end of the run (the plan's G-R5 pattern): give the GPU back. The backend's remembered base model is put
    back by the runner (it restores state.json and stops the backend it started)."""
    report_path = Path(args.report) if args.report else HERE / "cleanup.json"
    rec: dict = {"gate": "cleanup", "ok": False}
    try:
        api = Api(args.base_url, args.api_key)
        rec["status_before"] = api.get("/status")
        rec["unload"] = api.post("/pipeline/unload", None, timeout=300.0)
        rec["status_after"] = api.get("/status")
        rec["ok"] = rec["status_after"].get("state") == "unloaded"
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    report_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("cleanup:", "ok" if rec["ok"] else rec.get("error", "state is not unloaded"))
    return 0 if rec["ok"] else 1


# ---------------------------------------------------------------------------------------------
# Self-check (CPU only, no HTTP)
# ---------------------------------------------------------------------------------------------

def selfcheck(report: str | None = None) -> int:
    results: list[bool] = []

    def line(ok: bool, what: str) -> None:
        results.append(bool(ok))
        print(("ok   " if ok else "FAIL ") + what, flush=True)

    prompts = json.loads(DEFAULT_PROMPTS.read_text(encoding="utf-8-sig"))
    line(len(prompts) == 3, f"prompts_c.json has 3 prompts ({DEFAULT_PROMPTS.name})")

    # 1. the composition and the sha256 formula
    shared, owns, composed = expected_clip_prompts(prompts, 3)
    line(shared == prompts[0] and owns[0] == "" and all(o for o in owns[1:]), "split: shared = prompts[0]; clips 1, 2 carry an appended part")
    line(composed == prompts, "compose(shared, own) == prompts_c.json for the 3 clips (exact strings)")
    line(compose_clip_prompt("  a ", " b  ") == "a b" and compose_clip_prompt("a", None) == "a" and
         compose_clip_prompt("a", "  ") == "a" and compose_clip_prompt("  ", "b") == "b", "compose trims both ends, one space, skips empty parts")
    line(sha_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha_text is sha256 of the UTF-8 bytes")
    line(sha_text(prompts[0]) == "d83e7d299e6e41b5ee60551bbf4d43f56be4011a904dfe506662933f72b908e3",
         "sha256(prompts_c[0]) == the measurement script's clip-0 prompt_sha (c0_gate.json)")
    try:
        import importlib
        sys.path.insert(0, str(REPO))
        models = importlib.import_module("api.models")
        same = all(models.compose_clip_prompt(shared, o) == c for o, c in zip(owns, composed))
        line(same, "compose_clip_prompt equals the product's api.models.compose_clip_prompt")
    except Exception as exc:
        print(f"skip (product not importable here: {type(exc).__name__}): compare with api.models.compose_clip_prompt")

    # 2. the request bodies
    for gate in GATES:
        body = build_request(gate, prompts)
        cond = GATES[gate]
        n = cond["clips"]
        ok = (body["prompt"] == shared and len(body["clips"]) == n and all(c["num_frames"] == 124 for c in body["clips"])
              and "prompt" not in body["clips"][0] and body["width"] == cond["width"] and body["height"] == cond["height"]
              and body["seed"] == 42 and body["attention_backend"] == cond["attention"]
              and body["num_inference_steps"] == cond["steps"])
        line(ok, f"{gate} body: {cond['width']}x{cond['height']} {n}x124 {cond['attention']} steps {cond['steps']}")
        json.dumps(body)
    b0, b1, b2 = (build_request(g, prompts) for g in ("G-C0", "G-C1", "G-C2"))
    line(b0["clips"][1]["prompt"] == owns[1] and b0["clips"][2]["prompt"] == owns[2] and
         b0["embed_mp4_metadata"] is False and "vram16_mode" not in b0 and "loras" not in b0 and "crop_output" not in b0 and
         (b0["shift_video"], b0["shift_audio"]) == (12.0, 3.0), "G-C0 body: appended parts, embed false, vram16 default, no LoRA, no crop")
    line(b1["vram16_mode"] is True and b1["loras"] == [{"name": TURBO, "strength": 0.8}] and b1["num_inference_steps"] == 8 and
         "embed_mp4_metadata" not in b1, "G-C1 body: VRAM 16GB mode, the Turbo LoRA by registered name, 8 steps")
    line("loras" not in b2 and b2["num_inference_steps"] == 20 and len(b2["clips"]) == 13 and
         [c.get("prompt") for c in b2["clips"][1:5]] == [owns[1], owns[2], owns[1], owns[2]],
         "G-C2 body: 13 clips, appended parts alternate, no LoRA, 20 steps")
    _, _, c13 = expected_clip_prompts(prompts, 13)
    line(c13[0] == prompts[0] and c13[1] == prompts[1] and c13[2] == prompts[2] and c13[3] == prompts[1], "13-clip composed prompts cycle")
    bad = False
    try:
        split_prompts(["a b", "x"])
    except ValueError:
        bad = True
    line(bad, "split_prompts refuses a prompt that does not start with the shared one")

    # 3. frames
    line((expected_frames(3), expected_frames(13)) == (328, 1348) and GATES["G-C0"]["frames"] == 328 and GATES["G-C2"]["frames"] == 1348,
         "expected frames: 3 clips = 328, 13 clips = 1348")

    # 4. monotonicity
    good = [{"progress": 0.0, "clip": None, "clip_count": None}, {"progress": 0.1, "clip": 1, "clip_count": 3},
            {"progress": 0.1, "clip": 1, "clip_count": 3}, {"progress": 0.5, "clip": 2, "clip_count": 3},
            {"progress": 0.9, "clip": 3, "clip_count": 3}, {"progress": 1.0, "clip": 3, "clip_count": 3}]
    line(check_monotonic(good)[0], "monotonic: a rising trace passes")
    line(not check_monotonic(good[:3] + [{"progress": 0.05, "clip": 1, "clip_count": 3}])[0], "monotonic: progress going down is caught")
    line(not check_monotonic(good[:4] + [{"progress": 0.6, "clip": 1, "clip_count": 3}])[0], "monotonic: clip going down is caught")
    line(not check_monotonic(good[:2] + [{"progress": 0.2, "clip": 1, "clip_count": 4}])[0], "monotonic: clip_count changing is caught")
    line(not check_monotonic([{"progress": 1.2}])[0], "monotonic: progress above 1 is caught")

    # 5. latents sha256 comparison (product vs script)
    ref = [{"latents": {"v_sha256": f"v{i}", "a_sha256": f"a{i}"}, "seed": 42 + i} for i in range(3)]
    prod = [{"latents_sha256": {"video": f"v{i}", "audio": f"a{i}"}} for i in range(3)]
    line(compare_latents(prod, ref) == [], "compare_latents: equal -> no mismatch")
    prod_bad = json.loads(json.dumps(prod))
    prod_bad[1]["latents_sha256"]["audio"] = "zz"
    line(len(compare_latents(prod_bad, ref)) == 1 and "clips[1] audio" in compare_latents(prod_bad, ref)[0], "compare_latents: one audio sha differs -> 1 mismatch")
    line(len(compare_latents(prod[:2], ref)) >= 1, "compare_latents: a missing clip is caught")
    line(len(compare_latents([{"latents_sha256": None}] + prod[1:], ref)) == 2, "compare_latents: missing sha values are caught")

    # 6. a whole verdict on small data
    n = 3
    meta = {"backend": REAL_BACKEND_H3, "attention_used": "sdpa", "vram16_mode_used": "on", "generation_time_seconds": 600.0,
            "request": {"prompt": shared, "embed_mp4_metadata": False,
                        "clips": [{"num_frames": 124}] + [{"num_frames": 124, "prompt": o} for o in owns[1:]]},
            "h3": {"seeds": [42, 43, 44], "num_inference_steps": 20, "shift_video": 12.0, "shift_audio": 3.0,
                   "rss_peak_gib": 23.7, "commit_peak_gib": 32.4,
                   "clips": [{"prompt_sha256": sha_text(t), "seconds": {"encode": 9.0, "denoise": 100.0},
                              "latents_sha256": p["latents_sha256"], "head_equals_prev_tail": None if i == 0 else True}
                             for i, (t, p) in enumerate(zip(composed, prod))],
                   "phases": {"clips": [{"encode": {"seconds": 9, "peak_vram_mb": 1400, "peak_vram_reserved_mb": 1500},
                                         "denoise": {"seconds": 100, "peak_vram_mb": 3800, "peak_vram_reserved_mb": 3900}}] * 3,
                              "decode": {"seconds": 40.0, "peak_vram_mb": 5600, "peak_vram_reserved_mb": 5800}}}}
    ctx = {"job": {"status": "completed", "progress": 1.0, "clip": 3, "clip_count": 3},
           "poll": {"monotonic_ok": True, "polls": 100, "max_progress": 1.0},
           "metadata": meta, "mp4": {"frames": 328, "sha256": "x", "reader": "av"},
           "expected_prompts": composed, "reference": {"clips": ref, "decode": {"once": {"sha256": "x"}}}, "psnr": None, "mem": None}
    v = evaluate("G-C0", ctx)
    line(v["pass"] and v["records"]["vram"]["decode_allocated_mb"] == 5600 and v["records"]["vram"]["max_allocated_mb"] == 5600,
         "evaluate G-C0: a consistent run passes; VRAM peaks are read from phases")
    ctx_bad = json.loads(json.dumps(ctx))
    ctx_bad["metadata"]["h3"]["clips"][2]["latents_sha256"]["video"] = "other"
    v = evaluate("G-C0", ctx_bad)
    line(not v["pass"] and any("FIRST CRITERION" in f for f in v["failed"]), "evaluate G-C0: one differing latent sha fails the first criterion")
    ctx_bad = json.loads(json.dumps(ctx))
    ctx_bad["mp4"]["sha256"] = "different"
    v = evaluate("G-C0", ctx_bad)
    line(v["pass"], "evaluate G-C0: a differing mp4 sha256 is a record, not a failure")
    ctx_bad = json.loads(json.dumps(ctx))
    ctx_bad["mp4"]["frames"] = 327
    line(not evaluate("G-C0", ctx_bad)["pass"], "evaluate G-C0: 327 frames fails")
    ctx_bad = json.loads(json.dumps(ctx))
    ctx_bad["metadata"]["h3"]["clips"][1]["head_equals_prev_tail"] = False
    line(not evaluate("G-C0", ctx_bad)["pass"], "evaluate G-C0: head_equals_prev_tail false fails")
    ctx_bad = json.loads(json.dumps(ctx))
    ctx_bad["metadata"]["request"]["clips"][1]["prompt"] = "different"
    line(not evaluate("G-C0", ctx_bad)["pass"], "evaluate G-C0: a request whose composed prompt differs fails")
    ctx_bad = json.loads(json.dumps(ctx))
    ctx_bad["reference"] = None
    line(not evaluate("G-C0", ctx_bad)["pass"], "evaluate G-C0: no --compare report fails (the criterion cannot be judged)")
    ctx_bad = json.loads(json.dumps(ctx))
    ctx_bad["job"]["status"] = "failed"
    line(not evaluate("G-C0", ctx_bad)["pass"], "evaluate G-C0: a failed job fails")

    # G-C1: LoRA report
    prompts_clip = expected_clip_prompts(prompts, 3)[2]
    meta1 = json.loads(json.dumps(meta))
    meta1.update(attention_used="sage")
    meta1["h3"].update(num_inference_steps=8, shift_video=6.0, shift_audio=3.0,
                       loras=[{"name": TURBO + ".safetensors", "strength": 0.8, "pairs_applied": 208, "skipped": []}])
    ctx1 = {"job": ctx["job"], "poll": ctx["poll"], "metadata": meta1, "mp4": {"frames": 328, "sha256": "x"},
            "expected_prompts": prompts_clip, "reference": None, "psnr": None, "mem": None}
    line(evaluate("G-C1", ctx1)["pass"], "evaluate G-C1: Turbo reported, 328 frames, heads equal -> pass")
    meta1b = json.loads(json.dumps(meta1))
    meta1b["h3"]["loras"] = []
    ctx1b = dict(ctx1, metadata=meta1b)
    line(not evaluate("G-C1", ctx1b)["pass"], "evaluate G-C1: no LoRA report fails")

    # G-C2: 13 clips
    _, _, c13 = expected_clip_prompts(prompts, 13)
    meta2 = json.loads(json.dumps(meta))
    meta2["attention_used"] = "sage"
    meta2["h3"]["seeds"] = [42 + i for i in range(13)]
    meta2["h3"]["clips"] = [{"prompt_sha256": sha_text(t), "seconds": {"denoise": 400.0}, "latents_sha256": {"video": "v", "audio": "a"},
                             "head_equals_prev_tail": None if i == 0 else True} for i, t in enumerate(c13)]
    _, own13, _ = expected_clip_prompts(prompts, 13)
    meta2["request"]["clips"] = [{"num_frames": 124, **({"prompt": o} if o else {})} for o in own13]
    ctx2 = {"job": dict(ctx["job"], clip=13, clip_count=13), "poll": ctx["poll"], "metadata": meta2,
            "mp4": {"frames": 1348, "sha256": "x"}, "expected_prompts": c13, "reference": None, "psnr": None,
            "mem": {"max_working_set_gib": 28.0, "max_commit_gib": 37.0}}
    v2 = evaluate("G-C2", ctx2)
    line(v2["pass"] and v2["records"]["decode_seconds"] == 40.0 and v2["records"]["sampler"]["max_commit_gib"] == 37.0,
         "evaluate G-C2: 13 clips, 1348 frames pass; decode seconds and sampler recorded")
    ctx2b = json.loads(json.dumps(ctx2))
    ctx2b["mp4"]["frames"] = 1347
    line(not evaluate("G-C2", ctx2b)["pass"], "evaluate G-C2: 1347 frames fails")

    # 7. PSNR helper and the memory sampler
    try:
        import numpy as np
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = a.copy()
        b[0, 0, 0] = 255
        line(psnr_db(a, a) == math.inf and abs(psnr_db(a, b) - 10 * math.log10(255.0 ** 2 / (255.0 ** 2 / 48))) < 1e-9, "psnr_db: identical = inf; one saturated sample of 48 = 10log10(48) dB")
    except ImportError:
        print("skip (numpy missing): psnr_db")
    if sys.platform == "win32":
        s = ProcSampler(os.getpid(), 0.2)
        try:
            ws, commit = s.sample_once()
            line(ws > 0.0 and commit > 0.0 and os.getpid() in s.tree(), f"ProcSampler: this process tree has a working set {ws:.2f} GiB and a commit {commit:.2f} GiB")
        except Exception as exc:
            line(False, f"ProcSampler failed: {type(exc).__name__}: {exc}")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "x.bin"
        f.write_bytes(b"abc")
        line(sha256_file(f) == sha_text("abc"), "sha256_file")

    ok = all(results)
    summary = f"SELFCHECK {'PASS' if ok else 'FAIL'} ({sum(results)}/{len(results)})"
    if report:
        Path(report).write_text(json.dumps({"selfcheck": "PASS" if ok else "FAIL", "passed": sum(results), "total": len(results)}, indent=2), encoding="utf-8")
    print(summary)
    if ok:
        print("SELFCHECK PASS")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gate", nargs="?", choices=[*GATES, "cleanup"])
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--base-url", default="http://127.0.0.1:18629/api/v1")
    ap.add_argument("--api-key")
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS), help="prompts_c.json (3 prompts: the shared one, then it plus an appended sentence)")
    ap.add_argument("--compare", help="G-C0: the measurement script's report (c_chain.py --name g_c0_harness ...)")
    ap.add_argument("--transformer", default=FL2VA, help="registered name of the transformer to load")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--job-timeout", type=float, default=None, help="seconds to wait for the job (default: the gate's, a little under the runner's)")
    ap.add_argument("--backend-pid", type=int, help="root of the backend process tree (main-memory sampler)")
    ap.add_argument("--sample-interval", type=float, default=10.0)
    ap.add_argument("--cleanup-after", action="store_true", help="unload the pipeline at the end of this gate")
    ap.add_argument("--report")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck(args.report)
    if not args.gate:
        ap.error("a gate name (G-C0 / G-C1 / G-C2 / cleanup) or --selfcheck is required")
    if args.gate == "cleanup":
        return run_cleanup(args)
    return run_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
