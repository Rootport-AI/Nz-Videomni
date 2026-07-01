# Provenance notice — engine/ (first-party low-VRAM LTX-2.3 engine)

This `engine/` package is **first-party** project code (git-tracked, project root
`./engine`). It was derived from a vendored third-party fork, then adopted and
reorganized in place (the original fork tree was deleted in Stage 2b of the
`refactor/engine-firstparty-cleanup` work).

- **Origin**: the *Kandyman-iac* low-VRAM fork of **LTX-Desktop** (GGUF transformer
  + block-swap low-VRAM engine), wrapping the official LTX-2 `ltx_core` /
  `ltx_pipelines` pinned at rev **`00dc53d`** (ModelLedger API). Chosen because the
  official 1.1.6 safetensors loader crashes on this Windows machine; this engine
  runs LTX-2.3 within 16 GB VRAM. The fork's `backend/` source was the starting
  point; it has since been refactored into this first-party package (files moved
  under `engine/{worker,pipeline,gguf,gemma,transformer,...}`, imports rewritten,
  dead branches pruned — algorithms unchanged).

- **Runtime dependencies**: the engine worker runs on the dedicated
  **`./.venv-engine`** interpreter (separate from the app's torch-free `./.venv`).
  It carries torch 2.9.1+cu128 plus the LTX inference stack. Three of those
  distributions are **git direct-url installs (NOT PyPI)**:
    - `ltx-core`      — git `github.com/Lightricks/LTX-2` `packages/ltx-core` @ `00dc53d…`
    - `ltx-pipelines` — git `github.com/Lightricks/LTX-2` `packages/ltx-pipelines` @ `00dc53d…`
    - `diffusers`     — git `github.com/huggingface/diffusers` @ `01de02e8…`
  These are resolved from the venv's site-packages (normal, non-editable installs),
  not from any vendored source tree.

- **Reproducibility**: the fork tree that held `uv.lock` was deleted, so the venv is
  captured two ways in this directory:
    - `engine/engine-venv-pyproject.toml` — the dependency spec + `[tool.uv.sources]`
      (torch cu128 index, the three git revs above) to re-resolve/rebuild the venv.
    - `engine/venv-engine.freeze.txt` — an exact `name==version` snapshot of the
      current known-good `.venv-engine` install (generated via importlib.metadata;
      the git direct-url revs are recorded in its header comment).

- **License**: the upstream LTX-2 (`ltx_core` / `ltx_pipelines`) and LTX-Desktop
  license terms continue to apply to the derived engine code and to the LTX
  packages installed in `.venv-engine`. Retain upstream attribution when
  redistributing.

- **Our modifications & verification**: the Approach-W resident worker, GGUF Gemma
  per-layer quant/offload, block-swap wiring, Path-B component files, the DiT
  CPU-build load path, and the per-job block-swap retention leak fix are documented
  with rationale and 16 GB verification in `Docs/VERIFICATION_LOG.md`.
