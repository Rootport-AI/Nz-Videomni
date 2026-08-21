"""Persistent LTX-2.5 generation worker (§3-98 Phase 1 skeleton).

Runs inside .venv-engine-ltx25 -- the ONLY environment that has torch 2.9.1+cu128
together with official LTX-2 v1.2.0 (ltx_core / ltx_pipelines @ d151147) and
transformers 5.x. It is a SIBLING of engine/worker.py, never a replacement: the
2.3 worker keeps its own venv (.venv-engine, transformers 4.57) and neither
process can import the other's stack.

The app process (FastAPI, its own torch-less .venv) spawns ONE of these per
loaded model and talks to it over the same tiny JSON-lines protocol the 2.3
worker uses, so the parent-side plumbing is shared. Only small control JSON
crosses the pipe; the mp4 is written by the engine directly to a shared-disk
path.

WHAT THIS FILE IS *NOT*, YET
    Phase 1 delivers the skeleton only: bootstrap, logging, the framed protocol
    and the main loop. ``load`` is a STUB -- it checks that the paths it was
    handed exist and answers ``ready``; it builds nothing. ``generate`` /
    ``generate_chain`` do not exist here at all (the LTX25Runner rejects them
    upstream until Phase 2d wires the real pipeline). The pieces the 2.3 worker
    installs at import time -- the pre-denoise ``empty_cache`` monkeypatch, the
    ``progress_shim`` tqdm swap, the SageAttention probe, the CUDA pre-warm --
    are DELIBERATELY absent: each of them is tuned to the 2.3 wheel's internals,
    and re-applying them blind to a different pipeline is exactly the kind of
    borrowed-assumption bug this separate engine exists to avoid. They come back
    (or do not) one at a time, each on its own evidence.

Bootstrap (mirrors engine/worker.py, minus the 2.3-specific pre-warm):
  * TORCH_COMPILE_DISABLE=1 set before importing torch.
  * chdir(ROOT) + sys.path.insert(0, ROOT) so the first-party ``engine25.*``
    package (and the venv-installed ``ltx_core`` / ``ltx_pipelines``) import,
    whether launched as ``python -m engine25.worker`` or as a bare script path.
  * torch is NOT imported at module scope. The skeleton must be able to answer
    ``load``/``shutdown`` on a machine with no CUDA context yet, and Phase 2
    will add the imports next to the code that actually needs them.

Protocol (one JSON object per line; parent -> worker):
  {"op": "load", transformer_path, text_encoder_path, video_vae_path,
   audio_vae_path, spatial_upsampler_path}
      Phase 1: each value that is present and non-empty is checked for
      existence; a missing file is a fatal load error (exit 1), exactly as a
      failed build would be. Nothing is read or parsed.
  {"op": "shutdown"}

Replies are framed with the SAME unique prefix as the 2.3 worker so the shared
parent-side reader needs no branch, and so library/tqdm stdout noise stays
ignorable. Every protocol line: @@LTX@@<compact-json>, flushed. All other
logging goes to STDERR.
  @@LTX@@{"event":"ready","sampler":null,"sage_available":false}
  @@LTX@@{"event":"error","detail":...}

``ready.sampler`` is the sampler this process actually built. It is null here
because the skeleton builds nothing; from Phase 2d it carries the resolved
sampler name, which is how the ``use_ancestral_sampler`` assertion becomes
observable to the app instead of living only in a log line (the official code
silently downgrades that flag to False on GGUF paths and only WARNs).

``ready.sage_available`` is always false for this engine and is expected to stay
false: LTX 2.5 v1 is SDPA-only by scope decision, and sageattention is not even
installed in .venv-engine-ltx25. The field is present anyway because the app
publishes it as ``acceleration.sage_available`` on GET /status for whichever
engine is loaded -- an absent key would read as "unknown", which is wrong.
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# import cwd = project root: make the first-party `engine25.*` package (and the
# venv-installed `ltx_core`/`ltx_pipelines`) importable. Launched as
# `python -m engine25.worker` with PYTHONPATH=<root>, but we also insert the
# root on sys.path + chdir(ROOT) here so a bare `python engine25/worker.py`
# still works.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Unique frame prefix for every protocol line the parent parses. Intentionally
# IDENTICAL to engine/worker.py's: the parent's line reader is shared, and the
# two workers never run in the same process, so a second prefix would buy
# nothing and cost a branch on the app side.
PREFIX = "@@LTX@@"


def _configure_worker_logging() -> None:
    """Route module-level ``logging`` calls to STDERR.

    Same shape as the 2.3 worker's: ONE StreamHandler on the root logger aimed
    at STDERR (which the parent redirects into the worker log file), tagged so
    re-entry cannot double it. STDOUT is never touched -- that channel belongs
    exclusively to the ``@@LTX@@`` protocol. The ``[ltx25_worker]`` prefix is
    what tells the two engines' log lines apart when both have run.
    """
    root = logging.getLogger()
    if not any(getattr(h, "_ltx25_worker_handler", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[ltx25_worker] %(name)s: %(message)s"))
        handler._ltx25_worker_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(logging.INFO)


_configure_worker_logging()


def _log(msg: str) -> None:
    """Human/diagnostic logging -> STDERR only (never the protocol channel)."""
    print(f"[ltx25_worker] {msg}", file=sys.stderr, flush=True)


def _emit(event: str, **fields: object) -> None:
    """Write one framed protocol line to STDOUT and flush."""
    payload = {"event": event, **fields}
    print(PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)


def _detail(exc: BaseException) -> str:
    """repr + head/tail of the traceback for an error reply.

    Head AND tail: tail-only truncation drops the OUTER frames, which is where
    "which of our call sites raised" lives.
    """
    tb = "".join(traceback.format_exc())
    if len(tb) > 2800:
        tb = f"{tb[:1200]}\n...[traceback truncated]...\n{tb[-1600:]}"
    return f"{exc!r}\n{tb}"


# The load payload's path fields, in the order they are reported. Every one is
# OPTIONAL at the protocol level (an absent or empty value is simply not
# checked) so the adapter can grow the payload without this skeleton rejecting
# older or shorter messages; what the app actually guarantees is enforced on the
# app side by the manifest's REQUIRED_ASSETS.
_LOAD_PATH_FIELDS = (
    "transformer_path",
    "text_encoder_path",
    "video_vae_path",
    "audio_vae_path",
    "spatial_upsampler_path",
)

_LOADED = False


def _do_load(msg: dict) -> None:
    """Phase 1 STUB: verify the handed-down paths exist, then answer ``ready``.

    Building the real pipeline is Phase 2d. Until then this exists so the whole
    parent-side path -- spawn, load, ready, shutdown -- can be exercised end to
    end against the real .venv-engine-ltx25 interpreter, which is what proves the
    venv and the launch plumbing before any model code is written.

    A missing file raises, and the caller turns that into ``error`` + exit 1:
    the same failure mode a real build would produce, so the app's load-failure
    handling is exercised too.
    """
    global _LOADED
    if _LOADED:
        _emit("ready", sampler=None, sage_available=False)
        return

    checked = 0
    for field in _LOAD_PATH_FIELDS:
        raw = msg.get(field)
        if not raw:
            _log(f"load: {field} absent -- not checked")
            continue
        path = Path(str(raw))
        if not path.exists():
            raise FileNotFoundError(f"{field} does not exist: {path}")
        _log(f"load: {field} OK ({path.stat().st_size} bytes) {path}")
        checked += 1

    _log(f"LOAD_STUB_OK ({checked} path(s) verified; no model was built)")
    _LOADED = True
    # sampler=None: nothing was built, so there is no sampler to report yet.
    # sage_available=False: permanent for this engine (see the module docstring).
    _emit("ready", sampler=None, sage_available=False)


def _shutdown() -> None:
    """Best-effort free, then exit 0.

    torch is not imported at module scope in the skeleton, so the CUDA teardown
    the 2.3 worker performs is done only if torch actually came in along the way
    (it will, from Phase 2). Looking the module up in ``sys.modules`` rather than
    importing it keeps ``shutdown`` free of a multi-second import on a process
    that never touched the GPU.
    """
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
    sys.exit(0)


def main() -> None:
    # First, block for the load op (a fatal error here -> exit 1 so the parent
    # sees the worker die before 'ready').
    loaded = False
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            _log(f"ignoring non-JSON stdin line: {line[:120]!r}")
            continue

        op = msg.get("op")

        if not loaded:
            if op != "load":
                # Tolerate shutdown/EOF before load; ignore anything else.
                if op == "shutdown":
                    _shutdown()
                _log(f"ignoring op={op!r} before load")
                continue
            try:
                _do_load(msg)
                loaded = True
            except BaseException as exc:  # noqa: BLE001
                _log("LOAD_FAILED")
                _emit("error", detail=_detail(exc))
                sys.exit(1)
            continue

        # Serving loop (post-load). Phase 1 has no generate ops: an unsupported
        # op is answered rather than ignored, because the parent blocks waiting
        # for a reply and silence would read as a hang.
        if op == "shutdown":
            _shutdown()
        if op in ("generate", "generate_chain"):
            _log(f"op={op!r} is not implemented in the Phase 1 skeleton")
            _emit("error", detail=f"engine25 worker: op {op!r} is not implemented yet")
            continue
        _log(f"ignoring unknown op={op!r}")

    # EOF on stdin -> graceful shutdown.
    _shutdown()


if __name__ == "__main__":
    main()
