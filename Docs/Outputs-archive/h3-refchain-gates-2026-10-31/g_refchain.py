"""Section 1-109 MiniMax H3 reference chain: the GPU gates G-1 / G-1b / G-2 through the product's HTTP API.

A copy-and-extension of ``h3-chain-gates-2026-10-29/g_chain.py`` (the helper blocks between the
"verbatim" marks are copied as they are). An HTTP client for the running Nz-Videomni backend (a real
backend: MiniMax H3 is loaded by this script through ``POST /pipeline/load``). It does not import the
product. One gate = one process (``run_gates.ps1``)::

    python g_refchain.py G-1  --report g_1.json     # FL2VA regression: single T2V fingerprint + chain without references
    python g_refchain.py G-1b --report g_1b.json    # Ref2VA chain with one picture per clip: bit-exact against the measurement
    python g_refchain.py G-2  --report g_2.json     # Ref2VA chain, 1344x768, sage, 1 picture then 2 pictures
    python g_refchain.py cleanup --report cleanup.json    # unload the pipeline (the end of the run)
    python g_refchain.py --selfcheck                       # CPU only, no HTTP

What a gate does: (a) ``GET /status`` + ``GET /models`` (recorded), (b) ``POST /pipeline/load`` with
``base_model`` MiniMaxH3 and the transformer by registered name (FL2VA = ``default``; Ref2VA = the file
stem ``minimax_h3_ref2va_pruned_int8_convrot``; ``timeout=None``: a load can take minutes), then the
``GET /models`` ``transformer_role`` of the loaded file is checked, (c) for the Ref2VA gates:
``POST /upload/image`` (multipart ``file``; the answer's ``image_id`` is the ``id`` of a reference),
(d) ``POST /generate`` (G-1 only) and ``POST /generate/chain`` with ``clips[i].references`` (plan 3.1),
(e) ``GET /jobs/{id}`` every 5 s (progress / stage / clip trace, monotonicity), (f) after the job:
``outputs/<job_id>/metadata.json``, the mp4's size / sha256 / frame count, and the verdict.

Exit codes: 0 = every criterion passed, 1 = the run failed (exception, job failed, timeout, the backend
is not real), 2 = the run finished but a criterion failed (a sha256 mismatch, a reference record...).

Prompts: the product composes ``" ".join(non-empty(shared.strip(), own.strip()))`` per clip
(api/models.py ``compose_clip_prompt``). A prompts file holds the full prompt of every clip; this script
splits it into the longest common leading part that ends at a blank (= the shared ``prompt``) and the
remainder per clip (= ``clips[i].prompt``) and checks that the product's composition gives the file's
strings back. When clip 0 is a prefix of the other clips (prompts_c.json, prompts_c2.json) the shared
part is clip 0's whole prompt; prompts_c3.json adds a sentence INSIDE the later prompts, so its shared
part is only the leading ``<Picture 1> is the elderly gentleman.`` and clip 0 carries its own rest.
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
ARCHIVE = REPO / "Docs" / "Outputs-archive"

BASE_MODEL = "MiniMaxH3"
FL2VA = "default"  # registry name of MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors (the manifest default_file)
REF2VA = "minimax_h3_ref2va_pruned_int8_convrot"  # the registered name of a non-default weight file is its stem
REAL_BACKEND_H3 = "minimax-h3"
CLIP_FRAMES = 124
OVERLAP_FRAMES = 22
TERMINAL = ("completed", "failed", "cancelled")

# the fixed expectations (VERIFICATION_LOG 184.6 / 167.4 / 180.4)
FINGERPRINT_SHA = "0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9"   # single T2V, sdpa, 640x384, 124 f, 20 steps, seed 42, the G5 prompt
G_C0_MP4_SHA = "4a98be0884ae7129e34f58578a3fe7ca3117ca5a764545cbac404e12a70cbec5"       # G-C0: the chain without references
FINGERPRINT_PROMPT = (
    "A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in "
    "puddles. She looks at the camera and says <d>[English] It finally stopped raining.</d>"
)
#: DiT rows one picture adds (``denoise.sequence.ref_video``), by canvas
PICTURE_ROWS = {(640, 384): 247, (1344, 768): 1014}

PROMPTS_G1 = ARCHIVE / "h3-chain-2026-10-29" / "prompts_c.json"          # G-C0's prompts (the yellow raincoat)
PRIOR_G_C0 = ARCHIVE / "h3-chain-gates-2026-10-29" / "g_c0.json"        # G-C0's product record (latents, mp4)
REFCHAIN = ARCHIVE / "h3-refchain-2026-10-31"
PROMPTS_C2 = REFCHAIN / "prompts_c2.json"
PROMPTS_C3 = REFCHAIN / "prompts_c3.json"
REF_GATE = REFCHAIN / "g_ref2va_gate.json"                               # G-1b's comparison target
IMAGES = {
    "old_man": ARCHIVE / "h3-ref2va-2026-10-28" / "17fde0c1-image.png",
    "girl": ARCHIVE / "h3-ref2va-2026-10-28" / "d80320dd-image.png",
}

#: gate -> its fixed conditions (plan 6 M6 / 7). ``refs``: per clip the list of (image key, slot) or None.
GATES: dict[str, dict] = {
    "G-1": dict(transformer=FL2VA, role="fl2va", width=640, height=384, clips=3, attention="sdpa", steps=20,
                shift_video=12.0, shift_audio=3.0, vram16_mode=True, embed_mp4_metadata=False, frames=328,
                prompts=PROMPTS_G1, refs=None, compare=None, job_timeout=1450, single_timeout=600),
    "G-1b": dict(transformer=REF2VA, role="ref2va", width=640, height=384, clips=3, attention="sdpa", steps=20,
                 shift_video=12.0, shift_audio=3.0, vram16_mode=True, embed_mp4_metadata=False, frames=328,
                 prompts=PROMPTS_C2, refs=[[("old_man", 1)], [("old_man", 1)], [("old_man", 1)]], compare=REF_GATE,
                 job_timeout=1450),
    "G-2": dict(transformer=REF2VA, role="ref2va", width=1344, height=768, clips=3, attention="sage", steps=20,
                shift_video=12.0, shift_audio=3.0, vram16_mode=True, embed_mp4_metadata=None, frames=328,
                prompts=PROMPTS_C3,
                refs=[[("old_man", 1)], [("old_man", 1), ("girl", 2)], [("old_man", 1), ("girl", 2)]], compare=None,
                job_timeout=3500),
}


# ---------------------------------------------------------------------------------------------
# Pure helpers. Copied verbatim from g_chain.py (h3-chain-gates-2026-10-29): sha_text, compose_clip_prompt
# ---------------------------------------------------------------------------------------------

def sha_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def compose_clip_prompt(shared: str, own: str | None) -> str:
    """api/models.py ``compose_clip_prompt``: both ends trimmed, non-empty parts joined by one space."""
    parts = (shared.strip(), (own or "").strip())
    return " ".join(part for part in parts if part)


# ---- verbatim from g_chain.py: check_monotonic, compare_latents, check_prompts ----

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


# ---- verbatim: sha256_file, count_frames ----

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


# ---- verbatim: _check, vram_peak ----

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


# ---- verbatim: repo_path, _try, run_cleanup ----

def repo_path(p: str | None) -> Path | None:
    if not p:
        return None
    path = Path(p)
    return path if path.is_absolute() else REPO / path


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
# New in this copy: prompts, request bodies, metadata readers, verdicts
# ---------------------------------------------------------------------------------------------

def split_common_prefix(texts: list[str]) -> tuple[str, list[str]]:
    """The full prompt of every clip -> (shared prompt, appended part per clip).

    The shared part is the longest common leading part of ALL the prompts that ends at a blank (or at the end
    of one of them), stripped. The appended part of a clip is what follows it, stripped. The product composes
    ``shared + " " + own`` per clip, so the result must give every text back (checked here, raises otherwise).
    """
    if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
        raise ValueError("the prompts file must be a non-empty JSON list of non-empty strings")
    k = len(os.path.commonprefix(texts))
    while k > 0 and not all(k == len(t) or t[k].isspace() for t in texts):
        k -= 1
    shared = texts[0][:k].strip()
    if not shared:
        raise ValueError("the prompts have no common leading part (the shared prompt would be empty)")
    owns = [t[k:].strip() for t in texts]
    for i, t in enumerate(texts):
        if compose_clip_prompt(shared, owns[i]) != t:
            raise ValueError(f"prompts[{i}] cannot be recomposed from the shared part and its appended part")
    return shared, owns


def expected_clip_prompts(texts: list[str]) -> tuple[str, list[str], list[str]]:
    shared, owns = split_common_prefix(texts)
    return shared, owns, [compose_clip_prompt(shared, o) for o in owns]


def clip_refs(gate: str) -> list[list[tuple[str, int]]]:
    cond = GATES[gate]
    return cond["refs"] if cond["refs"] is not None else [[] for _ in range(cond["clips"])]


def reference_inputs(gate: str, image_ids: dict[str, str]) -> list[list[dict]]:
    """Per clip: the ``clips[i].references`` list the gate sends (plan 3.1: ``{kind, id, slot}``)."""
    return [[{"kind": "picture", "id": image_ids[key], "slot": slot} for key, slot in refs] for refs in clip_refs(gate)]


def expected_ref_rows(gate: str) -> list[int]:
    """DiT rows the pictures of each clip add (``denoise.sequence.ref_video``)."""
    cond = GATES[gate]
    rows = PICTURE_ROWS[(cond["width"], cond["height"])]
    return [len(refs) * rows for refs in clip_refs(gate)]


def build_chain_request(gate: str, texts: list[str], image_ids: dict[str, str], seed: int = 42) -> dict:
    """The ``POST /generate/chain`` body of a gate (only the fields the gate pins; the rest are the API defaults)."""
    cond = GATES[gate]
    shared, owns, _ = expected_clip_prompts(texts)
    refs = reference_inputs(gate, image_ids)
    clips = []
    for own, r in zip(owns, refs):
        clip: dict = {"num_frames": CLIP_FRAMES}
        if own:
            clip["prompt"] = own
        if r:
            clip["references"] = r
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
        "vram16_mode": cond["vram16_mode"],
        "clips": clips,
    }
    if cond["embed_mp4_metadata"] is not None:
        body["embed_mp4_metadata"] = cond["embed_mp4_metadata"]
    return body


def build_single_request(seed: int = 42) -> dict:
    """G-1 (a): the fingerprint's single T2V (``POST /generate``): 640x384, 124 frames, sdpa, 20 steps, the G5 prompt."""
    cond = GATES["G-1"]
    return {
        "prompt": FINGERPRINT_PROMPT, "width": cond["width"], "height": cond["height"], "num_frames": CLIP_FRAMES,
        "frame_rate": 24.0, "seed": seed, "attention_backend": cond["attention"], "num_inference_steps": cond["steps"],
        "shift_video": cond["shift_video"], "shift_audio": cond["shift_audio"], "vram16_mode": cond["vram16_mode"],
        "embed_mp4_metadata": False,
    }


def expected_frames(n_clips: int, clip_frames: int = CLIP_FRAMES, overlap: int = OVERLAP_FRAMES) -> int:
    return n_clips * clip_frames - (n_clips - 1) * overlap


def validate_reference(gate: str, ref: dict, expected: list[str]) -> list[str]:
    """G-1b: the comparison record (g_ref2va_gate.json) was made under the gate's conditions, with the same prompts
    and the same picture. Returns the differences (empty = comparable)."""
    cond = GATES[gate]
    c = ref.get("condition") or {}
    wants = (("width", cond["width"]), ("height", cond["height"]), ("seed", 42), ("steps", cond["steps"]),
             ("attention", cond["attention"]), ("clips", cond["clips"]), ("clip_frames", CLIP_FRAMES),
             ("shift_video", cond["shift_video"]), ("shift_audio", cond["shift_audio"]), ("vram16", True),
             ("inject", "clean"), ("video_inject_clean", 1.0), ("strength", 1.0))
    diffs = [f"{k}: record {c.get(k)!r}, gate {v!r}" for k, v in wants if c.get(k) != v]
    if ref.get("ok") is not True:
        diffs.append("the record is not ok")
    if not str(c.get("prompts_file", "")).replace("\\", "/").endswith("prompts_c2.json"):
        diffs.append(f"prompts_file: {c.get('prompts_file')!r}")
    clips = ref.get("clips") or []
    if len(clips) != cond["clips"]:
        diffs.append(f"the record has {len(clips)} clips")
    for i, text in enumerate(expected[: len(clips)]):
        if clips[i].get("prompt") != text:
            diffs.append(f"clips[{i}].prompt differs from the gate's composed prompt")
        if clips[i].get("prompt_sha") != sha_text(text):
            diffs.append(f"clips[{i}].prompt_sha != sha256(composed)")
        latents = clips[i].get("latents") or {}
        if not latents.get("v_sha256") or not latents.get("a_sha256"):
            diffs.append(f"clips[{i}].latents has no v_sha256 / a_sha256")
        if [(r.get("kind"), r.get("slot")) for r in clips[i].get("references") or []] != [("picture", 1)]:
            diffs.append(f"clips[{i}].references is not one picture in slot 1")
    want_img = ((clips[0].get("references") or [{}])[0].get("file_sha256") if clips else None)
    if want_img and IMAGES["old_man"].is_file() and sha256_file(IMAGES["old_man"]) != want_img:
        diffs.append("the picture file differs from the one the record was made with (file_sha256)")
    return diffs


def prior_clips_as_reference(prior: dict | None) -> list[dict] | None:
    """G-1: G-C0's product record (g_c0.json) in the shape compare_latents expects (record only)."""
    if not prior:
        return None
    out = []
    for c in (prior.get("metadata_h3") or {}).get("clips") or []:
        s = (c or {}).get("latents_sha256") or {}
        out.append({"latents": {"v_sha256": s.get("video"), "a_sha256": s.get("audio")}})
    return out or None


def clip_lists(meta: dict) -> list[tuple[str, list]]:
    """Where the product keeps its per-clip chain records: ``h3.clips`` (as in G-C0's metadata.json) and, if the
    release also shows them there, ``chain.clips`` (the plan names them ``done.chain.clips``)."""
    out = []
    h3 = (meta.get("h3") or {}).get("clips")
    if isinstance(h3, list):
        out.append(("h3.clips", h3))
    chain = (meta.get("chain") or {}).get("clips")
    if isinstance(chain, list):
        out.append(("chain.clips", chain))
    return out


def clip_field(meta: dict, i: int, key: str):
    """(where, value) of the first per-clip record that has ``key`` for clip ``i``; (None, None) when none has it."""
    for where, lst in clip_lists(meta):
        if i < len(lst) and isinstance(lst[i], dict) and lst[i].get(key) is not None:
            return where, lst[i][key]
    return None, None


def transformer_role(models: dict | None) -> str | None:
    for b in (models or {}).get("base_models") or []:
        if b.get("id") == BASE_MODEL:
            return b.get("transformer_role")
    return None


def _inputs_key(items) -> list[tuple]:
    return [(x.get("kind"), x.get("slot"), x.get("id")) for x in items or []]


def single_checks(checks: list[dict], jr: dict) -> None:
    job, poll, meta, mp4 = jr.get("job") or {}, jr.get("poll") or {}, jr.get("metadata") or {}, jr.get("mp4") or {}
    p = "single: "
    _check(checks, p + "job completed", job.get("status") == "completed", f"status={job.get('status')} error={job.get('error')!r}")
    _check(checks, p + "backend is the real H3 engine", meta.get("backend") == REAL_BACKEND_H3, f"metadata.backend={meta.get('backend')!r}")
    _check(checks, p + "progress is monotonic", poll.get("monotonic_ok"),
           "; ".join(poll.get("monotonic_problems") or []) or f"{poll.get('polls')} polls")
    _check(checks, p + "attention / VRAM 16GB mode as requested",
           meta.get("attention_used") == "sdpa" and meta.get("vram16_mode_used") == "on",
           f"attention_used={meta.get('attention_used')!r} vram16_mode_used={meta.get('vram16_mode_used')!r}")
    _check(checks, p + f"mp4 plays and has {CLIP_FRAMES} frames", mp4.get("frames") == CLIP_FRAMES, f"frames={mp4.get('frames')}")
    _check(checks, p + "embed_mp4_metadata was false", (meta.get("request") or {}).get("embed_mp4_metadata") is False,
           f"{(meta.get('request') or {}).get('embed_mp4_metadata')!r}")
    _check(checks, p + "no references record in a single job without references", "references" not in meta,
           f"keys={sorted(k for k in meta if 'ref' in k)}")
    _check(checks, p + "FIRST CRITERION: mp4 sha256 equals the fingerprint 0da6111c...", mp4.get("sha256") == FINGERPRINT_SHA,
           f"product {mp4.get('sha256')}, fingerprint {FINGERPRINT_SHA}")


def chain_checks(checks: list[dict], gate: str, jr: dict, ctx: dict) -> dict:
    """All the criteria of the chain job; returns the records."""
    cond = GATES[gate]
    n = cond["clips"]
    job, poll, meta, mp4 = jr.get("job") or {}, jr.get("poll") or {}, jr.get("metadata") or {}, jr.get("mp4") or {}
    h3 = meta.get("h3") or {}
    phases = h3.get("phases") or {}
    p = "chain: " if gate == "G-1" else ""
    _check(checks, p + "job completed", job.get("status") == "completed", f"status={job.get('status')} error={job.get('error')!r}")
    _check(checks, p + "backend is the real H3 engine", meta.get("backend") == REAL_BACKEND_H3, f"metadata.backend={meta.get('backend')!r}")
    _check(checks, p + "progress / clip trace is monotonic", poll.get("monotonic_ok"),
           "; ".join(poll.get("monotonic_problems") or []) or f"{poll.get('polls')} polls, max progress {poll.get('max_progress')}")
    final_ok = job.get("progress") is not None and job["progress"] >= 1.0 - 1e-9
    _check(checks, p + "final progress is 1.0", final_ok, f"progress={job.get('progress')}")
    _check(checks, p + f"{n} clip records in metadata", all(len(lst) == n for _, lst in clip_lists(meta)) and bool(clip_lists(meta)),
           f"{[(w, len(lst)) for w, lst in clip_lists(meta)]}")
    heads = [clip_field(meta, i, "head_equals_prev_tail")[1] for i in range(1, n)]
    _check(checks, p + "head_equals_prev_tail is true for every clip after the first", all(h is True for h in heads), f"{heads}")
    frames = mp4.get("frames")
    _check(checks, p + f"mp4 plays and has {cond['frames']} frames", frames == cond["frames"],
           f"frames={frames} (expected {n}x{CLIP_FRAMES} - {n - 1}x{OVERLAP_FRAMES}) reader={mp4.get('reader')}")
    seeds = h3.get("seeds")
    _check(checks, p + "seeds are seed + i", seeds == [42 + i for i in range(n)], f"{seeds}")
    _check(checks, p + "attention used", meta.get("attention_used") == cond["attention"], f"{meta.get('attention_used')!r}")
    _check(checks, p + "VRAM 16GB mode used", meta.get("vram16_mode_used") == "on", f"{meta.get('vram16_mode_used')!r}")
    got = (h3.get("num_inference_steps"), h3.get("shift_video"), h3.get("shift_audio"))
    _check(checks, p + "steps / shifts as requested", got == (cond["steps"], cond["shift_video"], cond["shift_audio"]), f"{got}")
    problems = check_prompts(meta, ctx.get("expected_prompts") or [], (ctx.get("reference") or {}).get("clips"))
    _check(checks, p + "composed prompts reach the engine unchanged", not problems, "; ".join(problems) or "equal")
    _check(checks, p + "no LoRA reported", not h3.get("loras"), f"{h3.get('loras')}")
    if cond["embed_mp4_metadata"] is False:
        _check(checks, p + "embed_mp4_metadata was false", (meta.get("request") or {}).get("embed_mp4_metadata") is False,
               f"{(meta.get('request') or {}).get('embed_mp4_metadata')!r}")

    product_latents = [{"latents_sha256": clip_field(meta, i, "latents_sha256")[1]} for i in range(n)]
    records: dict = {"clip_record_source": [w for w, _ in clip_lists(meta)]}

    # ---- the references ----
    ids = ctx.get("image_ids") or {}
    if cond["refs"] is None:
        seqs = [(((phases.get("clips") or [{}] * n)[i] or {}).get("denoise") or {}).get("sequence") or {} for i in range(n)]
        stray = ("references" in meta, any(isinstance(c, dict) and c.get("references") for _, lst in clip_lists(meta) for c in lst),
                 any("ref_video" in s for s in seqs))
        _check(checks, p + "no references record in a chain without references (metadata.references, clip records, sequence.ref_video)",
               not any(stray), f"metadata.references={stray[0]} clip-record references={stray[1]} sequence.ref_video={stray[2]}")
    else:
        sent = reference_inputs(gate, ids)
        n_pic = [len(r) for r in sent]
        req_clips = (meta.get("request") or {}).get("clips") or []
        got_req = [_inputs_key((req_clips[i] if i < len(req_clips) else {}).get("references")) for i in range(n)]
        _check(checks, "request.clips[i].references is what was sent (kind, slot, id)",
               got_req == [_inputs_key(r) for r in sent], f"{got_req}")
        mref = meta.get("references")
        mclips = (mref or {}).get("clips") if isinstance(mref, dict) else None
        ok_shape = isinstance(mclips, list) and len(mclips) == n
        detail = []
        if ok_shape:
            for i in range(n):
                e = mclips[i]
                good = (isinstance(e, dict) and (e.get("pictures"), e.get("videos"), e.get("audios")) == (n_pic[i], 0, 0)
                        and _inputs_key(e.get("inputs")) == _inputs_key(sent[i]))
                detail.append(good)
            ok_shape = all(detail)
        _check(checks, "metadata.references.clips: pictures per clip " + str(n_pic) + " and the inputs (kind, slot, id) in order",
               ok_shape, json.dumps(mref, ensure_ascii=False)[:1500] if mref is not None else "metadata has no references")
        records["metadata_references"] = mref
        counts, where = [], []
        for i in range(n):
            w, v = clip_field(meta, i, "references")
            where.append(w)
            counts.append(v)
        ok_counts = all(isinstance(v, dict) and (v.get("pictures"), v.get("videos"), v.get("audios")) == (n_pic[i], 0, 0)
                        for i, v in enumerate(counts))
        _check(checks, "chain clip records carry references {pictures, videos, audios} per clip " + str(n_pic), ok_counts,
               f"{counts} (found in {where})")
        want_rows = expected_ref_rows(gate)
        seq_rows, enc_rows = [], []
        for i in range(n):
            pc = (phases.get("clips") or [])
            clip_phase = pc[i] if i < len(pc) and isinstance(pc[i], dict) else {}
            seq_rows.append(((clip_phase.get("denoise") or {}).get("sequence") or {}).get("ref_video"))
            refs_enc = (clip_phase.get("reference_encode") or {}).get("references")
            enc_rows.append(sum(int(r.get("rows", 0)) for r in refs_enc) if isinstance(refs_enc, list) else None)
        _check(checks, f"phases.clips[i].denoise.sequence.ref_video equals the rows of the pictures {want_rows} "
                       f"({PICTURE_ROWS[(cond['width'], cond['height'])]} rows per picture)",
               seq_rows == want_rows, f"sequence.ref_video={seq_rows}")
        records["ref_video_rows"] = seq_rows
        records["reference_encode_rows"] = enc_rows
        records["reference_encode_seconds"] = [clip_field(meta, i, "seconds")[1] for i in range(n)]
        records["reference_encode_phase_present"] = [bool(((phases.get("clips") or [{}] * n)[i] or {}).get("reference_encode"))
                                                     for i in range(n)]

    # ---- the bit-exact / fingerprint criteria ----
    if gate == "G-1":
        _check(checks, p + "FIRST CRITERION: mp4 sha256 equals G-C0's (4a98be08...)", mp4.get("sha256") == G_C0_MP4_SHA,
               f"product {mp4.get('sha256')}, G-C0 {G_C0_MP4_SHA}")
        prior = prior_clips_as_reference(ctx.get("prior"))
        if prior is None:
            _check(checks, p + "RECORD: latents equal G-C0's product record (g_c0.json)", None, "g_c0.json not readable", kind="record")
        else:
            mism = compare_latents(product_latents, prior)
            _check(checks, p + "RECORD: latents equal G-C0's product record (g_c0.json)", not mism, "; ".join(mism) or "equal", kind="record")
    if gate == "G-1b":
        ref = ctx.get("reference")
        if ref is None:
            _check(checks, "FIRST CRITERION: latents_sha256 equals the measurement's", False, "no comparison record")
        else:
            mism = compare_latents(product_latents, ref.get("clips") or [])
            _check(checks, "FIRST CRITERION: clips latents_sha256 equal g_ref2va_gate.json (video and audio, every clip)",
                   not mism, "; ".join(mism) or f"{n} clips x (video, audio) equal")
            ref_seeds = [c.get("seed") for c in ref.get("clips") or []]
            _check(checks, "seeds equal the measurement's", seeds == ref_seeds, f"product {seeds}, measurement {ref_seeds}")
        _check(checks, "SECOND (record only): mp4 sha256 of this run", None, f"{mp4.get('sha256')}", kind="record")
    return records


def evaluate(gate: str, ctx: dict) -> dict:
    """``ctx``: jobs ({label: {job, poll, metadata, mp4}}), expected_prompts, image_ids, reference, prior,
    models_after_load, mem, uploads. -> verdict dict."""
    cond = GATES[gate]
    checks: list[dict] = []
    jobs = ctx.get("jobs") or {}
    role = transformer_role(ctx.get("models_after_load"))
    _check(checks, f"the loaded transformer reports transformer_role {cond['role']!r}", role == cond["role"], f"transformer_role={role!r}")
    if gate == "G-1":
        single_checks(checks, jobs.get("single") or {})
    for name, up in (ctx.get("uploads") or {}).items():
        _check(checks, f"RECORD: the stored upload of {name} is byte-identical to the source file", up.get("stored_equal"),
               json.dumps({k: up.get(k) for k in ("image_id", "stored_path", "source_sha256", "stored_sha256")}), kind="record")
    chain_jr = jobs.get("chain") or {}
    records = chain_checks(checks, gate, chain_jr, ctx)
    meta = chain_jr.get("metadata") or {}
    h3 = meta.get("h3") or {}
    decode = (h3.get("phases") or {}).get("decode") or {}
    records.update({
        "job_seconds": meta.get("generation_time_seconds"),
        "per_clip_seconds": [clip_field(meta, i, "seconds")[1] for i in range(GATES[gate]["clips"])],
        "decode_seconds": decode.get("seconds"),
        "vram": vram_peak(h3),
        "worker_rss_peak_gib": h3.get("rss_peak_gib"),
        "worker_commit_peak_gib": h3.get("commit_peak_gib"),
        "sampler": {k: v for k, v in (ctx.get("mem") or {}).items() if k != "trace"} or None,
    })
    failed = [c for c in checks if c["kind"] == "criterion" and c["ok"] is not True]
    return {"gate": gate, "pass": not failed, "failed": [c["name"] for c in failed], "checks": checks, "records": records}


# ---------------------------------------------------------------------------------------------
# HTTP additions and running a gate
# ---------------------------------------------------------------------------------------------

class ApiX(Api):
    def upload_image(self, path: Path) -> dict:
        """POST /upload/image (multipart ``file``). The answer: image_id, original_filename, stored_path, width, height."""
        with open(path, "rb") as f:
            r = self.client.post("/upload/image", files={"file": (path.name, f, "image/png")}, timeout=120.0)
        if r.status_code >= 400:
            raise RuntimeError(f"POST /upload/image -> HTTP {r.status_code}: {r.text[:1000]}")
        return r.json()


def run_job(api: ApiX, label: str, path: str, body: dict, timeout: float, poll_every: float, rec: dict) -> dict:
    """Submit, poll, and read the artifacts of one job. Raises on a timeout / a job that did not complete."""
    submitted = api.post(path, body)
    job_id = submitted["job_id"]
    print(f"job {job_id} submitted ({label}: POST {path})", flush=True)
    poll = poll_job(api, job_id, poll_every, timeout)
    jr: dict = {"job_id": job_id, "submit_response": submitted, "poll": {k: v for k, v in poll.items() if k != "job"},
                "job": poll["job"]}
    rec.setdefault("jobs", {})[label] = jr
    if poll["timed_out"]:
        raise RuntimeError(f"job {job_id} ({label}) did not finish within {timeout} s (it keeps running in the backend; "
                           "stop it by stopping the backend)")
    if poll["job"].get("status") != "completed":
        raise RuntimeError(f"job {job_id} ({label}) ended as {poll['job'].get('status')}: {poll['job'].get('error')}")
    out_dir = REPO / "outputs" / job_id
    meta_path, mp4_path = out_dir / "metadata.json", out_dir / "output.mp4"
    metadata = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    mp4 = {"path": str(mp4_path), "metadata_path": str(meta_path), "size_bytes": mp4_path.stat().st_size,
           "sha256": sha256_file(mp4_path)}
    mp4.update(count_frames(mp4_path))
    jr.update({"metadata_path": str(meta_path), "mp4": mp4, "metadata": metadata})
    return jr


def run_gate(args) -> int:
    gate = args.gate
    cond = GATES[gate]
    report_path = Path(args.report) if args.report else HERE / f"{gate}.json"
    transformer = args.transformer or cond["transformer"]
    rec: dict = {"gate": gate, "conditions": {k: (str(v) if isinstance(v, Path) else v) for k, v in cond.items()},
                 "transformer_loaded": transformer, "base_url": args.base_url, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "ok": False}
    exit_code = 1
    sampler = None
    api = None
    t0 = time.time()
    try:
        texts = json.loads(Path(cond["prompts"]).read_text(encoding="utf-8-sig"))
        if len(texts) != cond["clips"]:
            raise RuntimeError(f"{cond['prompts']} has {len(texts)} prompts, the gate has {cond['clips']} clips")
        shared, owns, expected = expected_clip_prompts(texts)
        rec.update(prompts_file=str(cond["prompts"]), shared_prompt=shared, appended_prompts=owns, expected_prompts=expected,
                   expected_prompt_sha256=[sha_text(t) for t in expected])
        reference = None
        if cond["compare"]:
            reference = json.loads(Path(cond["compare"]).read_text(encoding="utf-8-sig"))
            diffs = validate_reference(gate, reference, expected)
            rec["compare"] = {"path": str(cond["compare"]), "name": reference.get("name"), "condition": reference.get("condition")}
            if diffs:
                raise RuntimeError("the comparison record cannot be used for this gate: " + "; ".join(diffs))
        prior = None
        if gate == "G-1" and PRIOR_G_C0.is_file():
            prior = json.loads(PRIOR_G_C0.read_text(encoding="utf-8-sig"))
            rec["prior_record"] = {"path": str(PRIOR_G_C0), "mp4_sha256": (prior.get("mp4") or {}).get("sha256")}

        api = ApiX(args.base_url, args.api_key)
        rec["status_before"] = api.get("/status")
        rec["models_before"] = _try(lambda: api.get("/models"))
        if args.backend_pid and sys.platform == "win32":
            sampler = ProcSampler(args.backend_pid, args.sample_interval)
            sampler.start()

        # (b) load. No timeout: a swap from another base model rebuilds the worker (minutes); a weight swap re-reads the file.
        t_load = time.time()
        rec["load_response"] = api.post("/pipeline/load", {"base_model": BASE_MODEL, "models": {"transformer": transformer}}, timeout=None)
        rec["load_seconds"] = round(time.time() - t_load, 2)
        status = api.get("/status")
        rec["status_after_load"] = status
        if status.get("state") not in ("ready", "running") or status.get("base_model") != BASE_MODEL:
            raise RuntimeError(f"after the load: state={status.get('state')!r} base_model={status.get('base_model')!r}")
        if not (status.get("acceleration") or {}).get("sage_available"):
            raise RuntimeError("acceleration.sage_available is false: the backend looks like the mock (or sage is missing); "
                               "this gate needs a real backend")
        rec["models_after_load"] = api.get("/models")
        rec["transformer_role"] = transformer_role(rec["models_after_load"])

        # (c) uploads (the Ref2VA gates)
        image_ids: dict[str, str] = {}
        uploads: dict[str, dict] = {}
        for key in sorted({k for refs in clip_refs(gate) for k, _ in refs}):
            src = IMAGES[key]
            up = api.upload_image(src)
            image_id = up.get("image_id") or up.get("id")
            if not image_id:
                raise RuntimeError(f"POST /upload/image answered without image_id: {up}")
            image_ids[key] = image_id
            info = {"source": str(src), "source_sha256": sha256_file(src), "response": up, "image_id": image_id,
                    "stored_path": up.get("stored_path")}
            stored = repo_path(up.get("stored_path"))
            info["stored_sha256"] = sha256_file(stored) if stored is not None and stored.is_file() else None
            info["stored_equal"] = info["stored_sha256"] == info["source_sha256"]
            uploads[key] = info
            print(f"uploaded {key}: image_id={image_id} {up.get('width')}x{up.get('height')} stored_equal={info['stored_equal']}", flush=True)
        rec["uploads"] = uploads

        # (d)(e)(f) the jobs
        poll_every = args.poll
        if gate == "G-1":
            single_body = build_single_request(args.seed)
            rec["single_request_body"] = single_body
            run_job(api, "single", "/generate", single_body, args.job_timeout or cond["single_timeout"], poll_every, rec)
        chain_body = build_chain_request(gate, texts, image_ids, seed=args.seed)
        rec["request_body"] = chain_body
        chain = run_job(api, "chain", "/generate/chain", chain_body, args.job_timeout or cond["job_timeout"], poll_every, rec)
        rec["job_id"] = chain["job_id"]
        if sampler is not None:
            rec["memory_sampler"] = sampler.finish()
            sampler = None

        jobs_ctx = {}
        for label, jr in rec["jobs"].items():
            m = jr.pop("metadata")   # the full metadata.json is on disk; the report keeps its parts
            jr["metadata_h3"] = m.get("h3")
            jr["metadata_references"] = m.get("references")
            jr["metadata_request"] = m.get("request")
            jr["metadata_summary"] = {k: m.get(k) for k in (
                "backend", "attention_used", "vram16_mode_used", "block_swap_prefetch_used", "peak_vram_reserved_mb",
                "generation_time_seconds", "seed_used", "models", "chain", "output")}
            jobs_ctx[label] = {"job": jr["job"], "poll": jr["poll"], "metadata": m, "mp4": jr["mp4"]}
        verdict = evaluate(gate, {"jobs": jobs_ctx, "expected_prompts": expected, "image_ids": image_ids,
                                  "reference": reference, "prior": prior, "models_after_load": rec["models_after_load"],
                                  "mem": rec.get("memory_sampler"), "uploads": uploads})
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
        for jr in (rec.get("jobs") or {}).values():
            jr.pop("metadata", None)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return exit_code


# ---------------------------------------------------------------------------------------------
# Self-check (CPU only, no HTTP)
# ---------------------------------------------------------------------------------------------

FAKE_IDS = {"old_man": "id-old-man", "girl": "id-girl"}


def synthetic_chain(gate: str, texts: list[str], reference: dict | None = None) -> dict:
    """A metadata.json-shaped dict that a correct product would write for the gate (self-check data)."""
    cond = GATES[gate]
    n = cond["clips"]
    shared, owns, composed = expected_clip_prompts(texts)
    sent = reference_inputs(gate, FAKE_IDS)
    rows = PICTURE_ROWS[(cond["width"], cond["height"])]
    h3_clips, phase_clips, ref_clips, req_clips = [], [], [], []
    for i in range(n):
        k = len(sent[i])
        lat = {"video": f"v{i}", "audio": f"a{i}"}
        if reference is not None:
            lat = {"video": reference["clips"][i]["latents"]["v_sha256"], "audio": reference["clips"][i]["latents"]["a_sha256"]}
        c = {"prompt_sha256": sha_text(composed[i]), "seconds": {"encode": 11.0, "denoise": 100.0}, "latents_sha256": lat,
             "head_equals_prev_tail": None if i == 0 else True}
        seq = {"text": 300, "cond": 0, "audio": 414, "video": 8880, "total": 9600}
        phase = {"encode": {"seconds": 11}, "denoise": {"sequence": seq}}
        if k:
            c["references"] = {"pictures": k, "videos": 0, "audios": 0}
            c["seconds"]["reference_encode"] = 1.0
            seq.update(ref_video=k * rows, ref_audio=0)
            phase["reference_encode"] = {"seconds": 1.0, "references": [{"kind": "picture", "rows": rows}] * k}
            ref_clips.append({"pictures": k, "videos": 0, "audios": 0,
                              "inputs": [{**x, "max_sec": None} for x in sent[i]]})
        else:
            ref_clips.append(None)
        h3_clips.append(c)
        phase_clips.append(phase)
        rc: dict = {"num_frames": CLIP_FRAMES}
        if owns[i]:
            rc["prompt"] = owns[i]
        if k:
            rc["references"] = sent[i]
        req_clips.append(rc)
    meta = {"backend": REAL_BACKEND_H3, "attention_used": cond["attention"], "vram16_mode_used": "on",
            "generation_time_seconds": 600.0,
            "request": {"prompt": shared, "embed_mp4_metadata": False if cond["embed_mp4_metadata"] is False else True, "clips": req_clips},
            "h3": {"seeds": [42 + i for i in range(n)], "num_inference_steps": cond["steps"], "shift_video": cond["shift_video"],
                   "shift_audio": cond["shift_audio"], "loras": [], "rss_peak_gib": 23.7, "commit_peak_gib": 32.4,
                   "clips": h3_clips, "phases": {"clips": phase_clips,
                                                 "decode": {"seconds": 25.0, "peak_vram_mb": 5600, "peak_vram_reserved_mb": 5800}}}}
    if any(sent):
        meta["references"] = {"clips": ref_clips}
    return meta


def selfcheck(report: str | None = None) -> int:
    results: list[bool] = []

    def line(ok: bool, what: str) -> None:
        results.append(bool(ok))
        print(("ok   " if ok else "FAIL ") + what, flush=True)

    def clone(x):
        return json.loads(json.dumps(x))

    p_c0 = json.loads(PROMPTS_G1.read_text(encoding="utf-8-sig"))
    p_c2 = json.loads(PROMPTS_C2.read_text(encoding="utf-8-sig"))
    p_c3 = json.loads(PROMPTS_C3.read_text(encoding="utf-8-sig"))
    ref = json.loads(REF_GATE.read_text(encoding="utf-8-sig"))

    # 1. the composition
    line(compose_clip_prompt("  a ", " b  ") == "a b" and compose_clip_prompt("a", None) == "a" and compose_clip_prompt("  ", "b") == "b",
         "compose trims both ends, one space, skips empty parts")
    for name, texts in (("prompts_c.json (G-1)", p_c0), ("prompts_c2.json (G-1b)", p_c2), ("prompts_c3.json (G-2)", p_c3)):
        shared, owns, composed = expected_clip_prompts(texts)
        line(len(texts) == 3 and composed == texts, f"{name}: shared + own recompose to the 3 prompts (exact strings)")
    s0, o0, _ = expected_clip_prompts(p_c0)
    s2, o2, _ = expected_clip_prompts(p_c2)
    s3, o3, _ = expected_clip_prompts(p_c3)
    line(s0 == p_c0[0] and o0[0] == "" and all(o0[1:]), "G-1: shared = clip 0's prompt, clips 1, 2 carry an appended part")
    line(s2 == p_c2[0] and o2[0] == "" and all(o2[1:]) and o2[1].startswith("He turns"), "G-1b: shared = clip 0's prompt (the difference per clip is appended)")
    line(s3 == "<Picture 1> is the elderly gentleman." and all(o3) and o3[1].startswith("<Picture 2> is a young girl."),
         "G-2: the later prompts add a sentence inside, so shared is only the leading sentence and every clip carries the rest")
    line(sha_text(p_c0[0]) == "d83e7d299e6e41b5ee60551bbf4d43f56be4011a904dfe506662933f72b908e3" and p_c0[0] == FINGERPRINT_PROMPT,
         "G-1's clip-0 prompt is the fingerprint's G5 prompt (and the measurement script's clip-0 prompt_sha)")
    bad = False
    try:
        split_common_prefix(["a b", "x"])
    except ValueError:
        bad = True
    line(bad, "split_common_prefix refuses prompts with no common leading part")
    try:
        import importlib
        sys.path.insert(0, str(REPO))
        models = importlib.import_module("api.models")
        line(all(models.compose_clip_prompt(s, o) == t for (s, os_, ts) in ((s0, o0, p_c0), (s2, o2, p_c2), (s3, o3, p_c3))
                 for o, t in zip(os_, ts)), "compose equals the product's api.models.compose_clip_prompt for all three files")
    except Exception as exc:
        print(f"skip (product not importable here: {type(exc).__name__}): compare with api.models.compose_clip_prompt")

    # 2. the comparison record (G-1b)
    _, _, c2_composed = expected_clip_prompts(p_c2)
    line(validate_reference("G-1b", ref, c2_composed) == [], "g_ref2va_gate.json is comparable: condition, prompts, prompt_sha, one picture in slot 1, file_sha256")
    line(all(len(c["latents"]["v_sha256"]) == 64 and len(c["latents"]["a_sha256"]) == 64 for c in ref["clips"]) and len(ref["clips"]) == 3,
         "g_ref2va_gate.json: 3 clips with v_sha256 / a_sha256 (64 hex)")
    wrong = clone(ref)
    wrong["condition"]["width"] = 1344
    line(any("width" in d for d in validate_reference("G-1b", wrong, c2_composed)), "validate_reference catches another condition")
    line(any("prompt" in d for d in validate_reference("G-1b", ref, p_c3)), "validate_reference catches other prompts")
    line(all(p.is_file() for p in IMAGES.values()), "both reference pictures exist")
    line(sha256_file(IMAGES["old_man"]) == ref["clips"][0]["references"][0]["file_sha256"], "the old man's picture is byte-identical to the measurement's (file_sha256)")
    prior = json.loads(PRIOR_G_C0.read_text(encoding="utf-8-sig"))
    line(prior["mp4"]["sha256"] == G_C0_MP4_SHA and prior["request_body"]["prompt"] == s0 and
         [c["prompt_sha256"] for c in prior["metadata_h3"]["clips"]] == [sha_text(t) for t in p_c0],
         "g_c0.json (G-C0's product record): mp4 sha256 4a98be08..., same shared prompt and clip prompts as G-1")

    # 3. the request bodies
    b1 = build_chain_request("G-1", p_c0, {})
    line(all("references" not in c for c in b1["clips"]) and b1["embed_mp4_metadata"] is False and b1["vram16_mode"] is True and
         (b1["width"], b1["height"], b1["seed"], b1["attention_backend"], b1["num_inference_steps"], b1["shift_video"], b1["shift_audio"]) ==
         (640, 384, 42, "sdpa", 20, 12.0, 3.0) and b1["prompt"] == s0 and [c.get("prompt", "") for c in b1["clips"]] == o0 and
         all(c["num_frames"] == 124 for c in b1["clips"]), "G-1 chain body: G-C0's conditions, no references")
    bs = build_single_request()
    line(bs["prompt"] == FINGERPRINT_PROMPT and (bs["width"], bs["height"], bs["num_frames"], bs["frame_rate"], bs["seed"]) == (640, 384, 124, 24.0, 42) and
         bs["attention_backend"] == "sdpa" and bs["num_inference_steps"] == 20 and bs["embed_mp4_metadata"] is False and "references" not in bs,
         "G-1 single body: the fingerprint's conditions (640x384, 124 f, sdpa, 20 steps, seed 42, embed false)")
    bb = build_chain_request("G-1b", p_c2, FAKE_IDS)
    line([c["references"] for c in bb["clips"]] == [[{"kind": "picture", "id": "id-old-man", "slot": 1}]] * 3 and
         bb["prompt"] == s2 and bb["embed_mp4_metadata"] is False and (bb["width"], bb["height"], bb["attention_backend"]) == (640, 384, "sdpa"),
         "G-1b body: every clip has references=[{kind: picture, id, slot: 1}]; prompts from prompts_c2.json")
    b2 = build_chain_request("G-2", p_c3, FAKE_IDS)
    line([[(r["kind"], r["id"], r["slot"]) for r in c["references"]] for c in b2["clips"]] ==
         [[("picture", "id-old-man", 1)], [("picture", "id-old-man", 1), ("picture", "id-girl", 2)],
          [("picture", "id-old-man", 1), ("picture", "id-girl", 2)]] and (b2["width"], b2["height"], b2["attention_backend"]) == (1344, 768, "sage") and
         "embed_mp4_metadata" not in b2 and b2["prompt"] == s3 and [c["prompt"] for c in b2["clips"]] == o3,
         "G-2 body: clip 0 one picture, clips 1 and 2 two pictures (slots 1, 2); 1344x768 sage; prompts from prompts_c3.json")
    for gate_, texts in (("G-1", p_c0), ("G-1b", p_c2), ("G-2", p_c3)):
        json.dumps(build_chain_request(gate_, texts, FAKE_IDS))
    line(expected_ref_rows("G-1b") == [247] * 3 and expected_ref_rows("G-2") == [1014, 2028, 2028] and expected_ref_rows("G-1") == [0, 0, 0],
         "expected reference rows: 247 per picture at 640x384, 1014 at 1344x768")
    line(expected_frames(3) == 328 == GATES["G-1"]["frames"] == GATES["G-2"]["frames"], "expected frames: 3 clips = 328")
    line(ref["clips"][0]["sequence"]["ref_video"] == 247, "the measurement's ref_video for one picture at 640x384 is 247")

    # 4. the verdicts on small data
    def ctx_for(gate, texts, meta, mp4_sha="x", frames=328, reference=None, role=None):
        cond = GATES[gate]
        _, _, composed = expected_clip_prompts(texts)
        ctx = {"jobs": {"chain": {"job": {"status": "completed", "progress": 1.0}, "poll": {"monotonic_ok": True, "polls": 9, "max_progress": 1.0},
                                  "metadata": meta, "mp4": {"frames": frames, "sha256": mp4_sha, "reader": "av"}}},
               "expected_prompts": composed, "image_ids": FAKE_IDS, "reference": reference, "prior": None, "mem": None,
               "models_after_load": {"base_models": [{"id": BASE_MODEL, "transformer_role": role or cond["role"]}]}, "uploads": {}}
        return ctx

    m1b = synthetic_chain("G-1b", p_c2, ref)
    c1b = ctx_for("G-1b", p_c2, m1b, reference=ref)
    v = evaluate("G-1b", c1b)
    line(v["pass"], "evaluate G-1b: a consistent run passes" + ("" if v["pass"] else f" (failed: {v['failed']})"))
    bad = clone(c1b)
    bad["jobs"]["chain"]["metadata"]["h3"]["clips"][2]["latents_sha256"]["audio"] = "other"
    line(not evaluate("G-1b", bad)["pass"], "evaluate G-1b: one differing audio sha256 fails the first criterion")
    bad = clone(c1b)
    bad["reference"] = None
    line(not evaluate("G-1b", bad)["pass"], "evaluate G-1b: no comparison record fails")
    bad = clone(c1b)
    bad["jobs"]["chain"]["metadata"]["h3"]["phases"]["clips"][1]["denoise"]["sequence"]["ref_video"] = 1014
    line(not evaluate("G-1b", bad)["pass"], "evaluate G-1b: ref_video not equal to the picture rows fails")
    bad = clone(c1b)
    del bad["jobs"]["chain"]["metadata"]["references"]
    line(not evaluate("G-1b", bad)["pass"], "evaluate G-1b: no metadata.references fails")
    bad = clone(c1b)
    bad["jobs"]["chain"]["metadata"]["references"]["clips"][1]["inputs"][0]["slot"] = 2
    line(not evaluate("G-1b", bad)["pass"], "evaluate G-1b: a wrong input in metadata.references fails")
    bad = clone(c1b)
    del bad["jobs"]["chain"]["metadata"]["h3"]["clips"][0]["references"]
    line(not evaluate("G-1b", bad)["pass"], "evaluate G-1b: a clip record without references fails")
    bad = clone(c1b)
    bad["jobs"]["chain"]["metadata"]["request"]["clips"][2]["references"] = []
    line(not evaluate("G-1b", bad)["pass"], "evaluate G-1b: a request that lost a clip's references fails")
    bad = clone(c1b)
    bad["models_after_load"]["base_models"][0]["transformer_role"] = "fl2va"
    line(not evaluate("G-1b", bad)["pass"], "evaluate G-1b: the FL2VA role loaded instead of Ref2VA fails")
    bad = clone(c1b)
    bad["jobs"]["chain"]["metadata"]["h3"]["clips"][1]["head_equals_prev_tail"] = False
    line(not evaluate("G-1b", bad)["pass"], "evaluate G-1b: head_equals_prev_tail false fails")
    bad = clone(c1b)
    bad["jobs"]["chain"]["mp4"]["frames"] = 327
    line(not evaluate("G-1b", bad)["pass"], "evaluate G-1b: 327 frames fails")
    # the per-clip records may sit under chain.clips instead of h3.clips
    alt = clone(c1b)
    am = alt["jobs"]["chain"]["metadata"]
    am["chain"] = {"clips": am["h3"].pop("clips")}
    line(evaluate("G-1b", alt)["pass"], "evaluate G-1b: per-clip records found under chain.clips as well")

    m2 = synthetic_chain("G-2", p_c3)
    c2 = ctx_for("G-2", p_c3, m2)
    v = evaluate("G-2", c2)
    line(v["pass"] and v["records"]["ref_video_rows"] == [1014, 2028, 2028], "evaluate G-2: pictures 1, 2, 2 and ref_video 1014, 2028, 2028 pass" +
         ("" if v["pass"] else f" (failed: {v['failed']})"))
    bad = clone(c2)
    bad["jobs"]["chain"]["metadata"]["references"]["clips"][0]["pictures"] = 2
    line(not evaluate("G-2", bad)["pass"], "evaluate G-2: clip 0 with 2 pictures in the record fails")
    bad = clone(c2)
    bad["jobs"]["chain"]["metadata"]["h3"]["phases"]["clips"][2]["denoise"]["sequence"]["ref_video"] = 1014
    line(not evaluate("G-2", bad)["pass"], "evaluate G-2: ref_video 1014 for a clip with two pictures fails")

    m1 = synthetic_chain("G-1", p_c0)
    m1["request"]["embed_mp4_metadata"] = False
    c1 = ctx_for("G-1", p_c0, m1, mp4_sha=G_C0_MP4_SHA)
    meta_s = {"backend": REAL_BACKEND_H3, "attention_used": "sdpa", "vram16_mode_used": "on", "request": {"embed_mp4_metadata": False}}
    c1["jobs"]["single"] = {"job": {"status": "completed", "progress": 1.0}, "poll": {"monotonic_ok": True, "polls": 5},
                            "metadata": meta_s, "mp4": {"frames": 124, "sha256": FINGERPRINT_SHA}}
    v = evaluate("G-1", c1)
    line(v["pass"], "evaluate G-1: fingerprint 0da6111c... and G-C0's mp4 sha256 4a98be08... pass" + ("" if v["pass"] else f" (failed: {v['failed']})"))
    bad = clone(c1)
    bad["jobs"]["single"]["mp4"]["sha256"] = "other"
    line(not evaluate("G-1", bad)["pass"], "evaluate G-1: a different single fingerprint fails")
    bad = clone(c1)
    bad["jobs"]["chain"]["mp4"]["sha256"] = "other"
    line(not evaluate("G-1", bad)["pass"], "evaluate G-1: a different chain mp4 sha256 fails")
    bad = clone(c1)
    bad["jobs"]["chain"]["metadata"]["references"] = {"clips": [None, None, None]}
    line(not evaluate("G-1", bad)["pass"], "evaluate G-1: a references record in a chain without references fails")
    bad = clone(c1)
    bad["models_after_load"]["base_models"][0]["transformer_role"] = "ref2va"
    line(not evaluate("G-1", bad)["pass"], "evaluate G-1: the Ref2VA role loaded instead of FL2VA fails")

    # 5. monotonicity, latents comparison, sha, helpers
    good = [{"progress": 0.0, "clip": None, "clip_count": None}, {"progress": 0.1, "clip": 1, "clip_count": 3},
            {"progress": 0.5, "clip": 2, "clip_count": 3}, {"progress": 1.0, "clip": 3, "clip_count": 3}]
    line(check_monotonic(good)[0] and not check_monotonic(good[:3] + [{"progress": 0.2, "clip": 2, "clip_count": 3}])[0],
         "monotonic: a rising trace passes, progress going down is caught")
    rr = [{"latents": {"v_sha256": f"v{i}", "a_sha256": f"a{i}"}} for i in range(3)]
    pp = [{"latents_sha256": {"video": f"v{i}", "audio": f"a{i}"}} for i in range(3)]
    line(compare_latents(pp, rr) == [] and len(compare_latents(pp[:2], rr)) >= 1, "compare_latents: equal -> none; a missing clip is caught")
    line(sha_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha_text is sha256 of the UTF-8 bytes")
    line(transformer_role({"base_models": [{"id": "LTX23", "transformer_role": None}, {"id": BASE_MODEL, "transformer_role": "ref2va"}]}) == "ref2va",
         "transformer_role reads the MiniMaxH3 entry of GET /models")
    line(REF2VA == "minimax_h3_ref2va_pruned_int8_convrot" and (REPO / "models" / "MiniMaxH3" / "Weights" / (REF2VA + ".safetensors")).is_file()
         or not (REPO / "models").is_dir(), "the Ref2VA registered name is the stem of an existing weight file")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "x.bin"
        f.write_bytes(b"abc")
        line(sha256_file(f) == sha_text("abc"), "sha256_file")
    if sys.platform == "win32":
        s = ProcSampler(os.getpid(), 0.2)
        try:
            ws, commit = s.sample_once()
            line(ws > 0.0 and commit > 0.0 and os.getpid() in s.tree(), f"ProcSampler: this process tree has a working set {ws:.2f} GiB and a commit {commit:.2f} GiB")
        except Exception as exc:
            line(False, f"ProcSampler failed: {type(exc).__name__}: {exc}")

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
    ap.add_argument("--transformer", help="registered name of the transformer to load (default: the gate's)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--job-timeout", type=float, default=None, help="seconds to wait for each job (default: the gate's, a little under the runner's)")
    ap.add_argument("--backend-pid", type=int, help="root of the backend process tree (main-memory sampler)")
    ap.add_argument("--sample-interval", type=float, default=10.0)
    ap.add_argument("--cleanup-after", action="store_true", help="unload the pipeline at the end of this gate")
    ap.add_argument("--report")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck(args.report)
    if not args.gate:
        ap.error("a gate name (G-1 / G-1b / G-2 / cleanup) or --selfcheck is required")
    if args.gate == "cleanup":
        return run_cleanup(args)
    return run_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
