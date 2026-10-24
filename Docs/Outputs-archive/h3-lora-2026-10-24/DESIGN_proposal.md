# Design proposal for §1-104 (Plan agent, Opus, 2026-10-24)

Input: the supervisor's design direction (hook/subclass, FP32 add, descriptor `lora_dir`, scoped registry, FEATURE_UI row `lora_audio_strength`) plus reports R1–R3 and E1–E3. Output: the INTERFACE block, file-by-file changes, numerics, tests, gates, open questions. The approved plan (`PLAN_h3_lora.md`) supersedes this document where they differ (the adversarial review changed several points; see `REVIEW_plan.md`).

## Five points where the designer departed from the supervisor's direction (all accepted)
1. **Hook mechanism.** Use `register_forward_hook` instead of editing `H3Int8Linear.forward` and swapping in an `H3LoraLinear` subclass for the token refiner. Every non-row call goes through `Module.__call__` (`dit.py:240,248,260-261`, `lowvram.py:152,165`, token refiner); only `forward_rows` (`lowvram.py:139-141`) bypasses it and gets an explicit argument. LoRA-off: no hook registered, `dit.py` untouched → bit-identity by construction.
2. **Row chunking in both paths** with `row_chunks(M, VRAM16_ROW_CHUNK)`: an unchunked off-path qkv f32 accumulator at S≈40k is ≈3.4 GB; and off/vram16 only present cuBLAS identical shapes if chunks line up.
3. **Format strictness** lives in the registry scope, not at the call sites (`api/generate.py:128-129`, `pipeline_manager.py:690-695`, chain paths unchanged).
4. **Construction order**: `LoraRegistry` is built at `api/context.py:83` before `model_registry`/`pipeline_manager`; the scope provider must be read lazily and construction moved below `set_active_base_model`. Scope should follow the runner's descriptor (same source as `active_engine_family`).
5. **Golden `h3` keys** (`tests/test_h3_engine.py:933,978`) → `done.h3.loras` only when the job had LoRAs.

Verified facts: `h3_format.py` is torch-free and already imported by the app (`services/model_registry.py:46`); `.gitignore:36-39` covers the new guide file; `read_tensors` gives contiguous per-tensor buffers (`engine/sft_quant/sft_reader.py:47-82`); prefetch `_plan` uses the module's own CPU tensor as master (`block_swap_prefetch.py:466-475`).

## INTERFACE (as proposed; the approved plan §3.1 is the final form)
- `h3_lora_format.py`: `H3_LORA_TARGETS`, `LoraPairKeys(a,b,alpha)`, `H3LoraKeyMap(layout, pairs, unmapped, supported, reason)`, `map_lora_keys(names, shapes=None)`; dotted and underscore regexes; `supported := pairs non-empty AND no DoRA/LoHa/LoKr AND unmapped empty AND rank consistent`.
- Descriptor key `lora_dir` (models_dir-relative) → `BaseModelDescriptor.lora_dir: str | None`.
- `LoraFormatRule(verdict, strict)` + `LoraScope(family, lora_dir, lora_dir_display, include_config, rule)`; `LoraRegistry(config, scope_provider)`; `LoraEntryInfo += supported, unsupported_reason`. (Review Minor 12 reduced the rule to a plain callable.)
- `services/engines/__init__.py`: `lora_format_rule(family)`, `lora_scope_for(descriptor, config)`.
- `GET /loras`: rows + `supported`, `unsupported_reason`; top-level `lora_dir`.
- Feature name `lora_audio_strength` (H3 `UNSUPPORTED_FEATURES`, not in `REJECT_TABLE`).
- H3 generate payload `"loras": [{"path","strength"}]` always present.
- `done.h3.loras[]` report (only with LoRAs). (Review Minor 18: file name, not path.)
- `engine_h3/lora.py`: `H3LoraInput`, `load_lora_file`, `attach`, `detach`, `xa_for`, `apply`; buffers `_lora_A_{k}`/`_lora_B_{k}` (persistent=False), `_lora_specs`, `_lora_hook`; `H3Int8Linear.forward_rows(..., lora_xa=None)`.
- Webui: UiTarget `promptBar.loraAudioStrength`, `loraAudioStrengthHiddenFor`, props `audioStrengthHidden` / `loraAudioHidden` / `loraUnsupportedNotice`, `useLoras({refreshKey})`, `LoraEntry.supported/unsupported_reason`, `lora_dir`, `omitLoraAudioStrength`.

## Numerics (as proposed)
- LoRA-off bit-identical: no attach → no buffers/specs/hooks; `forward_rows` adds one `getattr`; `xa_for` returns None; prefetch sees today's tensors; J1 LoRA → J2 none: detach before `_park_dit`.
- Off vs vram16 with LoRA: xa identical (same row chunks); fc1/fc2/out_proj B-steps identical; qkv B-step differed in N (`[m,r]@[r,21504]` vs `[m,r]@[r,1024]`) → proposed "measure, then owner judges" with a column-sliced fallback. (Review Major 7: adopt the column split up front, uniformly.)
- Cost (S≈40k, 50 blocks): Turbo ≈7.5e13 FLOP/step ≈1.3 s; rank-16 ≈0.1 s; f32 accumulator round trip ≈2.4 TB/step ≈2.4 s at 20 B/element (Review Major 6: 12 B/element ≈1.5 TB). Without the xa cache in vram16: +1.3 (rank 16) to +4 s (Turbo) per step. Memory: arena growth = file-dtype bytes per block (Turbo bf16 ≈39 MB); ring +235 MB (vram16) / +390 MB (window 8); f32 chunk ≤470 MB (fc1, before column chunking); qkv xa `[S,384]` f32 ≈61 MB; full residency +1.9 GB.

## Tests, gates, open questions
Transferred into the approved plan §5 and §7 (the five open questions were answered by the supervisor: unsupported = any unmappable key; IC-LoRA config entries hidden on H3; identity by construction; cost accepted with ledger follow-ups; Library tag kept as 3-argument).
