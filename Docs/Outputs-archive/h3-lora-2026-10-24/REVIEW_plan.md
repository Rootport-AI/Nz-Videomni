# Adversarial review of the §1-104 plan (Opus, 2026-10-24)

Reviewer brief: fact-check the design, work split, gates and docs list of `PLAN_h3_lora.md` against the code; flag over-engineering; find gaps. Owner decisions were fixed. Result: Critical 0 / Major 7 / Minor 14. Supervisor triage: all 7 Major adopted; 13 Minor adopted; Minor 17 (notice wording) decided by the owner (ruling 73: keep the original wording). None rejected.

## Verified as stated in the plan
- Hook placement is sound: `H3Attention.forward` calls `self.qkv_proj(x)` and only then `.view(...)` (`engine_h3/dit.py:240`); token-refiner blocks go through `self.attn(...)` / `self.mlp(...)` (`dit.py:357-359`); the pipeline is `torch.no_grad()` only (`pipeline_h3.py:65,542`); the CUDA `int8_linear` allocates `out` with `torch.empty`; `F.linear` returns a fresh tensor.
- `move_module_tree` / `drop_to_meta` walk `module._buffers`, non-persistent included (`engine_h3/sft_load.py:144-160,195-209`); hooks survive `.to()` and meta. `_enumerate_slots` yields every `_buffers` entry (`engine/transformer/block_swap_prefetch.py:284-301`); `_plan` handles any plain dtype (512-byte alignment, `view(uint8)`, contiguifies itself); `_build_views` rebuilds with `raw.view(dtype).view(shape)`. The synchronous swap uses `blk.to(device)` / `prev.to("cpu")`.
- `release_installed()` touches no tensor (`block_swap_service.py:169-176`). `teardown_prefetch` → `_release` writes `mod._buffers[name] = cpu_t` by name (`:655-680`) — so detach must follow teardown.
- `forward_rows` callers: `lowvram.py:139-141` and `tests/test_h3_engine.py:1881` only.
- Nothing between `api/context.py:83` and `PipelineManager` (`:94`) uses `lora_registry`; `ModelRegistry.descriptor(base_model=None)` exists (`services/model_registry.py:356`); `config._abs` / `config.models_dir` as assumed; `_require_rel_path` rejects absolute and `..`.
- `.gitignore:36-39` keeps `models/MiniMaxH3/LoRA/put_LoRA_safetensors_here.txt` tracked (`git check-ignore` matched the `!models/**/put_*_here.txt` rule).
- `parse_prompt_loras` has 3 call sites (`handlers.py:891,1467`, `batch.py:353`).
- Cost figures add up (Turbo ≈7.5e13 FLOP/step; 60,928 output columns per row → ≈2.4 TB/step at 20 B/element; ring 6/10 arenas → +235/+390 MB).
- `api/generate.py:128-140` will not 422 an ordinary H3 style LoRA (`kind` is "control" only with `reference_downscale_factor` in the header).

## Major
1. **Two sources for "the active family".** `/generate` validates against `pipeline_manager.active_engine_family` (runner descriptor; `api/generate.py:79,91`, `pipeline_manager.py:251-261`), while the plan's provider read `model_registry.descriptor()` (changes only in `_remember`, `:623`). During an LTX→H3 load a request could be validated against H3 while LoRA names resolve in the LTX folder. → Point the provider at the runner's descriptor (adopted).
2. **A bad LoRA file is found only after text encoding.** `generate()` runs `validate_request` and images before `_encode` (`pipeline_h3.py:520-546`); a failure inside `_denoise` also drops the resident DiT (`:989-993`). → Check existence + header + key map in `validate_request`; read tensors at attach (adopted). Precision check moved to `load`.
3. **Gates do not prove "FP32 path, no silent bf16".** A 1-ulp tolerance test passes a bf16 implementation. → Cancellation-style input where bf16 provably differs, `torch.equal` vs explicit f32 reference; operand-dtype spy on `mm`/`addmm_`; no `except` in `apply` (adopted).
4. **Block-swap coverage incomplete.** Gates ran only with prefetch on; sync path (`block_swap_service.py:239-290`) and full residency (`_place_dit`, `n<=0 or n>=len`) untested with LoRA buffers. → CPU test of the sync path; one GPU run with prefetch off; document that full residency puts all LoRA buffers on GPU (adopted).
5. **Detach-after-teardown order required but unwritten.** `_release` re-installs `_buffers[name]` for blocks in flight (`block_swap_prefetch.py:676-680`). → Comment at the detach line, DESIGN §6.6, CPU ordering test (adopted).
6. **Cheaper formulation with one bf16 rounding.** `delta = mm(xa·s, Bᵀ)` into an f32 chunk, then `out_chunk.add_(delta)` (in-place add of f32 into bf16 computes in f32, rounds once; `torch.can_cast(f32, bf16)` is true): 12 B/element instead of 20 (≈1.5 TB vs 2.4 TB per step). Note `comfy_kitchen.int8_linear` already has `residual`/`residual_scale` epilogue arguments (`backends/cuda/__init__.py:2035-2090`; dtype handling unchecked) → ledger follow-up (adopted).
7. **Adopt 1024-column chunks for the qkv B-step up front.** Same split as `lowvram`'s q/k/v slices; cost ≈ +0.1–0.3 s/step of kernel launches; removes the measure-then-decide branch and keeps ruling 46's bit-for-bit standard. → Adopted as a uniform rule for all four modules (`LORA_COL_CHUNK = 1024`).

## Minor
8. `test_honoured_fields_are_read_by_generate` greps for `request.<field>` in H3 `generate` (`tests/test_engine_h3_adapter.py:102-105`) → read `request.loras` in the adapter; update `GOLDEN_GENERATE_KEYS_H3` (`:491-507`). (adopted)
9. Re-governing the two strengths by `reference_video_id` is correct; say in the spec that a stray value is silently inert. (adopted)
10. `metadata.json` `ic_lora` echoes `audio_strength` and `preprocess_for` reads config only (`pipeline_manager.py:121-134`) → normalise `audio_strength` to None in H3's `validate_for_family`; make `preprocess_for` scoped. (adopted)
11. Gradio `drop_audio` plumbing redundant once the server normalises. (adopted: not built)
12. Machinery larger than needed: `LoraFormatRule.strict` (→ "no verdict = strict legacy"), `done.h3.lora_fp32_matmul` constant (dropped), trim `done` fields, `--lora PATH[:STRENGTH]` parsing (→ `--lora PATH` + `--lora-strength`). (adopted)
13. Detach via `delattr` so `_non_persistent_buffers_set` is cleaned. (adopted)
14. `PinnedStagingPool.ensure` only grows (`block_swap_prefetch.py:260-282`): after a Turbo job the resident service keeps ≈2×39 MB pinned. Document. (adopted)
15. Drop the `H3LoraFile` dict after attach to avoid duplicate host copies of token-refiner tensors. (adopted)
16. `useLoras` has 4 call sites (`AppShell.tsx:317`, `InventoryScreen.tsx:42`, `useOutpaintForm.ts:404`, `useBatchI2vLongForm.ts:436`); refresh all, and change the EXISTING AppShell call. (adopted)
17. Notice wording on LTX: LTX still 422s unsupported formats, so "may have no effect" is inaccurate there. → Owner ruling 73: keep the original wording for both families.
18. `done.h3.loras[].path` would embed the absolute path into the mp4 comment via `recipe_text` (`services/video_io.py:1754-1756`; `pipeline_manager.py:1993-2001`) → record the file name only. (adopted)
19. Gate 4 should record `commit_peak_gib`; add J1 Turbo → J2 Style resident. (adopted)
20. Also check `torch.backends.cuda.matmul.fp32_precision` (torch 2.13 new API; `torch/backends/cuda/__init__.py:159,185`). (adopted)
21. `POST /loras/reload` returns counts only; `GET` carries `lora_dir`. State in API_REFERENCE §3.7. (adopted)

## Overall verdict (reviewer)
The engine half holds up against the code: the hook and the explicit `lora_xa` path, the non-persistent buffers, attaching before `install()`, detaching after teardown. LoRA-off bit-identity holds because no hook or buffer exists and `forward_rows` only adds one attribute lookup. Main risks were the active-family source, late failure, and gate coverage; the 12-byte formulation and the 1024-column split make the plan cheaper and simpler at once. With the Major findings handled, the plan is ready to implement.
