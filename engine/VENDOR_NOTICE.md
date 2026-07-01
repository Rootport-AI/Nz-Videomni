# Vendor notice — LTX-Desktop-LOW-VRAM (backend)

This directory is a **vendored third-party fork**, modified in place for this project.

- **Upstream**: the *Kandyman-iac* low-VRAM fork of LTX-Desktop (GGUF transformer + block-swap
  low-VRAM engine), wrapping the official LTX-2 `ltx_core` / `ltx_pipelines` pinned at
  rev `00dc53d` (ModelLedger API). Chosen because the official 1.1.6 safetensors loader
  crashes on this Windows machine; the fork runs LTX-2.3 within 16 GB VRAM.
- **What is version-controlled here**: only `backend/` **source** (`*.py` + small configs).
  The fork's `.venv/`, model weights (`*.safetensors`/`*.gguf`), caches, `uv.lock`, and the
  unused web frontend (`public/`, `images/`, …) are intentionally git-ignored. The fork's
  own git history was disabled (renamed to `.git_fork_disabled`) so these files are tracked
  as plain source by the outer project repo rather than as a submodule.
- **Our modifications** (Approach-W resident worker `_ltx_worker.py`, GGUF Gemma
  `gemma_gguf_quant_service.py`, block-swap / pipeline wiring, Path-B component-files,
  the per-job block-swap retention leak fix, etc.) are documented with rationale and
  verification in `Docs/VERIFICATION_LOG.md` and `Docs/NEXT_SESSION_HANDOFF.md`.
- **To refresh from upstream**: re-clone the fork at the recorded rev and re-apply the
  patches recorded in `Docs/VERIFICATION_LOG.md` (the fork's original history lives in
  `.git_fork_disabled` if needed for diffing; it can be deleted to reclaim ~42 MB).
