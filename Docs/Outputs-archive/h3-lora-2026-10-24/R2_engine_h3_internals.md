# R2: MiniMax H3 engine internals (facts for runtime LoRA)

Repo: `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni`. Read-only investigation, 2026-10-24.
Only facts, with `path:line`. No design. "Header" = read from the safetensors JSON header with a
read-only script (`python -I`); "hash" = sha256 of tensor bytes, compared between two files.

**Correction to the brief's assumed shapes (load-bearing):** H3 has `inner = heads*head_dim =
56*128 = 7168`, which is larger than hidden 5376. So qkv is **5376→21504** (not 16128), out_proj is
**7168→5376** (not 5376→5376), fc1 is 5376→28672 (= 2×ffn, ffn 14336), and fc2 is **14336→5376**
(not 28672→5376). Sources: `engine_h3/dit.py:221-232`, `h3_format.py:148-152,360-365`, and the
header of the official file (§1 table).

---

## 1. `engine_h3/quant_linear.py`: `H3Int8Linear`

- Class `engine_h3/quant_linear.py:59`. Constructor `:70-99`: `(in_features, out_features, *, bias: bool,
  spec: QuantSpec, bias_dtype=torch.bfloat16)`.
  - `self.weight`: `nn.Parameter` int8 `[N, K]`, `requires_grad=False` (`:84-87`).
  - `self.weight_scale`: buffer f32 `[N, 1]`, `persistent=True` (`:88-92`). Per-output-row scale.
  - `self.bias`: optional Parameter of `bias_dtype` (`:93-98`). On the official file the 4 int8 layers
    have **no bias** (header: no `.bias` keys; `nn.Linear(..., bias=False)` at `dit.py:229,232,256,257`).
  - `self.spec`: plain attribute (`QuantSpec`: format / convrot / convrot_groupsize), NOT a tensor, so it
    is not in the state dict or in the block-swap arena (`:65-67,99`).
  - **No rotation tensor is stored.** ConvRot is applied to the activation inside the kernel:
    stored weight is `W @ H^T` per 256-column group, `int8_linear(convrot=True)` rotates x by H
    (`:25-29`). The official file's spec, read from `comfy_quant`:
    `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}` (same in the
    PulpCut turbo file).
- `forward(x)` `:101-121`: flattens to 2-D; **input cast to bf16** if not bf16 (`:104-107`);
  `.contiguous()`; bias cast to bf16 (`:109-111`); calls
  `comfy_kitchen.int8_linear(x2d, weight, weight_scale, bias, out_dtype=torch.bfloat16, convrot=..., convrot_groupsize=...)`
  (`:112-120`); **output bf16**, reshaped to `[*lead, out_features]`.
- `forward_rows(x, n0, n1)` `:123-154`: same as forward but with `self.weight[n0:n1]`,
  `self.weight_scale[n0:n1]`, `bias[n0:n1]` sliced **on every call, never stored** (block swap
  rebinds `.data`; a kept slice would point at the old placement, `:126-133`). Output `[*lead, n1-n0]` bf16.
- Backend: `require_cuda_int8()` `:164-201` checks comfy-kitchen's CUDA backend has `int8_linear`,
  then `disable_backend("triton")` and `disable_backend("eager")` (`:199-200`). So only the CUDA
  path exists; a call it refuses raises `NoCapableBackendError` (fail loud, `:16-23`). Called once in
  `H3Pipeline.load` (`engine_h3/pipeline_h3.py:465-467`). comfy-kitchen pinned at 0.2.37, torch 2.13.0+cu130
  (`engine_h3/venv-engine-h3.freeze.txt:50,79`).
- Swap-in: `swap_in_int8_linears(root, quant, *, bias_dtype_of)` `:204-243` replaces each `nn.Linear`
  named in `cfg.quant` (dotted, prefix-stripped, e.g. `"blocks.0.attn.qkv_proj"`) by an `H3Int8Linear`
  built on meta; unknown name / non-Linear raises.
- Which layers are int8 is **decided by the file** (every `.weight` with dtype I8,
  `h3_format.py:236-291`). Official file and PulpCut turbo file (header): **200 I8 layers =
  50 blocks × {attn.qkv_proj, attn.out_proj, mlp.fc1, mlp.fc2}**. Recorded jobs agree:
  `"int8_linears": 200` (e.g. `outputs/193b5585-.../metadata.json:167`).

| Module (official file) | dtype in file | Compute dtype | Class |
|---|---|---|---|
| `blocks.{i}.attn.qkv_proj` `[21504, 5376]` | I8 + F32 scale `[21504,1]` | int8 kernel, bf16 out | `H3Int8Linear` |
| `blocks.{i}.attn.out_proj` `[5376, 7168]` | I8 + F32 scale | same | `H3Int8Linear` |
| `blocks.{i}.mlp.fc1` `[28672, 5376]` | I8 + F32 scale | same | `H3Int8Linear` |
| `blocks.{i}.mlp.fc2` `[5376, 14336]` | I8 + F32 scale | same | `H3Int8Linear` |
| `blocks.{i}.norm1/norm2` `[5376]`, `attn.q_norm/k_norm` `[128]` | BF16 | bf16 `nn.RMSNorm` | — |
| `blocks.{i}.adaln_proj.linear` `[96768, 8]` + bias | **F16** (→F32 by `normalize_dtypes`) | f32 | `nn.Linear` (f32 island) |
| `token_refiner.blocks.{0,1}.attn.qkv_proj/out_proj`, `mlp.fc1/fc2` (same shapes as blocks) | **BF16** | bf16 | plain `nn.Linear` |
| `condition_proj` (5120→5376) | — | bf16 (`_linear_bf16`) | `nn.Linear` |
| `video_patch_proj`, `audio_patch_proj`, `final_layer.video_out/audio_out`, `final_layer.adaln_proj.linear` | — | f32 (`_linear_f32`) | `nn.Linear` (f32 islands) |

Islands: `engine_h3/dit.py:44-50`, `normalize_dtypes` `:437-473` (f32 set `:457-459`; **`H3Int8Linear`
modules are skipped entirely** `:462-463`; every other floating Parameter/buffer → bf16 `:464`).

## 2. `engine_h3/dit.py`: block structure

Config (`h3_format.py:145-161`, values from the official file): hidden 5376, num_layers **50**,
token_refiner_layers **2**, heads 56, head_dim 128, ffn_hidden 14336, text_dim 5120, latents_dim 24,
audio_latents_dim 32, adaln grid 1025 / rank 8, rope_inv_freq 16. Prefix of the official file: none
(`blocks.0...` at the top level).

`H3DiT.__init__` `dit.py:407-435`: `video_patch_proj`, `audio_patch_proj`, `condition_proj`,
`token_refiner` (`H3TokenRefiner`), `blocks` (`nn.ModuleList` of 50 `H3DiTBlock`), `final_layer`,
`rope` (`inv_freq` buffer), buffer `adaln_t_table`. `requires_grad_(False)` `:435`.

One `H3DiTBlock` (`dit.py:311-344`), attribute names = file tensor names:
- `norm1`, `norm2`: `nn.RMSNorm(5376)` bf16 (`:319-320`)
- `attn` = `H3Attention` (`:219-248`): `qkv_proj` Linear(5376 → 3×7168=21504) **fused, stacked
  `[q; k; v]`** (not per-head interleave, `:32-34`); `q_norm`, `k_norm` RMSNorm(128); `out_proj`
  Linear(7168 → 5376); `attention_backend` plain attribute (`:234`).
- `mlp` = `H3MLP` (`:251-261`): `fc1` Linear(5376 → 28672) rows `[gate; value]`; `fc2`
  Linear(14336 → 5376); SwiGLU `fc2(silu(gate)*value)` (`:259-261`).
- `adaln_proj` = `H3AdaLNProj(rank 8, 5376, expand 6, modality 3)` (`:264-285`): `linear` Linear(8 →
  96768) f32 with bias; returns 6 chunks of `[M*3, 5376]` f32 (shift/scale/gate × msa/mlp).
- Class attr `vram16 = None` (`:315`).

Off-path forward `dit.py:325-344` (args are tensors only, positional — for block swap, `:52-61`):
`adaln_proj(t_emb)` → `_modulate(norm1(x))` → `attn` → `_gate` → `_modulate(norm2(x))` → `mlp` → `_gate`.
Inside `H3Attention.forward` (`:236-248`): `qkv = qkv_proj(x).view(s, 3, heads, head_dim)`;
`q, k, v = qkv.unbind(dim=1)` (`:240-241`) — this is the Q/K/V split in the off path (all 56 heads at
once); q_norm/k_norm, RoPE (`apply_rope`, first 96 dims), `_attention` (`:202-211`, SDPA or sage,
`[S,H,D]`), then `out_proj(out.reshape(s, 7168))`.

Token refiner (`dit.py:347-371`): `H3TokenRefiner.blocks` = 2 × `H3TokenRefinerBlock` with the
**same module names** `norm1`, `norm2`, `attn.{qkv_proj,q_norm,k_norm,out_proj}`, `mlp.{fc1,fc2}`, **no
`adaln_proj`**, no RoPE (`attn(..., None, None)`, `:358`); plus `token_refiner.final_norm`. Runs once per
job via `refine_text` (`:484-491`: `_linear_bf16(condition_proj)` → token_refiner). Its Linears are plain
bf16 `nn.Linear` (`F.linear` in bf16 via `nn.Linear.forward`).

Precision (what runs where):
- f32: `_modulate`/`_gate` arithmetic (index_select of f32 tables, result cast back to x dtype,
  `:288-308`); `H3AdaLNProj.forward` (`F.linear` all f32, `:279-285`); `adaln_embed` lerp
  (`:520-532`); `_linear_f32` patch projections and output heads (`:393-396`, used `:511-513,554`);
  RoPE angles (`rope.py`, design §3).
- bf16: block stack activations (`x = tokens.to(torch.bfloat16)` `:601`), all norms, int8 linears'
  in/out, attention, token refiner, condition_proj.
- `H3DiT.forward` `:558-609`: sets `block.attn.attention_backend` and `block.vram16` on every block every
  call (`:593-595`); loops `for block in self.blocks: x = block(x, t_emb, adaln_index, rope_cos, rope_sin)`
  (`:604-605`); `final_rows` (`:534-556`).

## 3. `engine_h3/lowvram.py`: VRAM 16GB mode

Branch: `H3DiTBlock.forward` → `lowvram.block_forward` when `self.vram16 is not None` (`dit.py:334-337`);
`final_rows` → `final_rows_chunked` (`dit.py:544-547`).
- `Vram16Plan(head_group=8, row_chunk=4096)` `lowvram.py:43-49`; constants
  `engine_h3/constants.py:52,57,62,64` (`VRAM16_HEAD_GROUP=8`, `VRAM16_ROW_CHUNK=4096`,
  `VRAM16_BLOCKS_ON_GPU=4`, `VRAM16_CAP_MARGIN_MIB=1024`).
- `row_chunks(s, chunk)` `:52-71`: `[0,s)` into `ceil(s/chunk)` **equal** ranges (sizes differ by ≤1);
  `s ≤ chunk` → one range. Purpose: avoid int8 kernel small-m paths (`:55-57`).
- `block_forward` `:107-170` — how each projection is called:
  - AdaLN modulate per row chunk into a preallocated `h` (`:126-129`).
  - **qkv**: per head group `h0..h1` (56/8 = **7 groups**, g=8 heads, 1024 output rows each):
    three calls `attn.qkv_proj.forward_rows(h, j*inner + h0*d, j*inner + h1*d)` for j=0,1,2 (q, k, v)
    (`:136-141`), each over **all S rows** of `h` (rows of the sequence not chunked here) → 21 int8
    calls per block (design §6.5 "範囲と既知の限界"). Fused `[S, 21504]` never exists (`:16-19`).
  - group output copied into `o[:, h0*d:h1*d]` of one `[S, 7168]` bf16 buffer (`:135,149`).
  - **out_proj**: called **once**, on the full `o [S, 7168]` (`:152`) — not chunked.
  - residual `_gate` per row chunk (`:156-157`).
  - **fc1 / fc2**: via `block.mlp(h2[r0:r1])` per row chunk (`:165`), i.e. `H3MLP.forward` →
    `fc1.forward` and `fc2.forward` on 4096-row chunks (sequence-row slicing; weights not sliced).
  - final residual per row chunk (`:169-170`).
- `final_rows_chunked` `:173-192`: norm+modulate per row chunk into an f32 buffer, then one
  `_linear_f32` per head (not split).
- Allocator cap: `derive_memory_cap(free, total, reserved, margin)` `:88-97` =
  `(reserved + free − margin)/total`, clamped 0..1. Applied in `H3Pipeline._vram16_begin`
  (`pipeline_h3.py:794-822`) with `torch.cuda.set_per_process_memory_fraction(fraction, index)` before
  `_place_dit`; lifted to 1.0 in `_vram16_finish` (`:824-837`) from `_denoise`'s finally.
  Recorded cap for I2V 896×1152×124: 0.857 = 14,033 MiB (VERIFICATION_LOG §169.3).
- Swap window in this mode: `blocks_on_gpu_for` `:74-85` → `min(n_load, 4)` unless whole-DiT residency.
- Token refiner + condition_proj evicted after `refine_text` (`pipeline_h3.py:775-791,901-903`): to CPU
  if `keep_resident`, else `drop_to_meta`.
- "Same rows" fact for an adapter: in `forward_rows`, output columns `[n0, n1)` correspond to weight
  rows `[n0, n1)` (`quant_linear.py:145-148`); for fc1/fc2 and the AdaLN parts the split is over
  sequence rows (inputs `h2[r0:r1]`), so a per-row-independent add-on computes the same result per chunk.
  Off/on bit-identity is a stated contract (裁定 46, `lowvram.py:9-13`); sage differs in 2/50 blocks
  (裁定 53, VERIFICATION_LOG §169.3).

## 4. Block swap / prefetch

Files: `engine/transformer/block_swap_service.py` (shared, not modified by H3) and
`engine/transformer/block_swap_prefetch.py`.
- Placement: `H3Pipeline._place_dit` `pipeline_h3.py:733-772`. Strips old wrappers (`:756`); if
  `blocks_on_gpu ≤ 0 or ≥ 50` → `move_module_tree(dit, device, skip=set())` (whole DiT, every
  param/buffer, `:759-762`); else moves all **non-block** tensors (`skip = ids of every module under
  every block`, `:763-764`) and `BlockSwapService(... hold_arenas=True)` (one per process, `:766-767`),
  sets `blocks_on_gpu`, `prefetch_requested`, `install(dit)` (`:769-771`). Default load
  `blocks_on_gpu=8`, prefetch on (`pipeline_h3.py:133-135`).
- Block discovery: `_get_blocks` tries attr `blocks` (`block_swap_service.py:199-223`) → `dit.blocks`.
  token_refiner is **not** in the swap list (its `.blocks` is a different module).
- `install` `block_swap_service.py:99-167`: `block.to("cpu")` for all (`:130-131`), builds the
  prefetch engine if asked (`:135`), patches each block's `forward` (`:143-147`). Marker attr
  `_block_swap_original_forward` (`:34`).
- **Sync path** `_patch_block` `:225-288`: window `[idx, idx+blocks_on_gpu)`; `blk.to(device)` if any
  parameter is on CPU (`:259-268`; checks `parameters()` only, but `Module.to` moves buffers too);
  evicts `idx-1` with `prev.to("cpu")` (`:272-277`); moves args to device (`:286-287`).
- **Prefetch path** (`block_swap_prefetch.py`): CPU copy taken at install is the authoritative
  "master" (正本 on CPU); no D2H on eviction (`:8-11`). `prepare()` `:356-419`: `_plan` per block
  packs **every tensor in `block.modules()` — `_parameters` AND `_buffers` (persistent or not)** — into
  one uint8 arena layout (`_enumerate_slots` `:284-301`, `_plan` `:421-479`, 512-byte alignment `:142`).
  Pinned staging: `PinnedStagingPool` with **2 slots** of the largest block's size (`:147,236-281`,
  `torch.empty(..., pin_memory=True)` `:175`). Arena ring: `min(total, blocks_on_gpu + 2)` device
  arenas sized to the largest block (`:405-409`). Issue-ahead: up to 2 issues per step (`:152`),
  window-ahead (`on_block_forward` `:517-568`). Views installed by `mod._parameters[name].data = dev_t`
  / `mod._buffers[name] = dev_t` (`:648-653`); release points back at CPU master (`:655-691`).
  Prefetch setup failure → silent fallback to sync for that job (`block_swap_service.py:296-330`).
- Per-block state transferred: all of the block's tensors (int8 weights, f32 scales, bf16 norms,
  f32/f16 adaln_proj). Header size of block 0 = 387,359,520 B = **369.41 MiB**; arena slot measured
  **371.1 MiB** (VERIFICATION_LOG §168.6 item 4; `pipeline_h3.py:431-433`).
- Hook points for extra per-block tensors (facts, not design): any tensor registered as a
  Parameter or buffer on a module under `dit.blocks[i]` is (a) moved by `block.to()` in the sync path,
  (b) packed into that block's arena + CPU master by `_plan` in the prefetch path, (c) skipped by the
  non-block `move_module_tree` in `_place_dit`, (d) moved to CPU by `_park_dit` (`pipeline_h3.py:839-851`),
  (e) increases `max_bytes` → every ring arena and both pinned slots grow (sized to the largest block).
  Plain Python attributes holding tensors are NOT moved by either path. `normalize_dtypes` would
  cast floating buffers to bf16 on non-int8 modules but skips `H3Int8Linear` (`dit.py:462-464`); it runs
  once in `_build_dit` (`pipeline_h3.py:698`).
  Prior art in the repo: IC/Style-LoRA A/B as `register_buffer(..., persistent=False)` on the target
  Linear plus a spec list attribute, "ride block-swap automatically"
  (`engine/gguf/ic_lora_common.py:275-353`, detach `:384-408`; prefetch comment about Style-LoRA A/B
  buffers `block_swap_prefetch.py:665-667`; selfcheck with `_ic_lora_A_0` buffers
  `engine/transformer/block_swap_prefetch_selfcheck.py:187-192`). That LTX path merges a dequantized
  delta into the weight (`engine/sft_quant/quant_service.py:273-292`) — different from the requested
  activation-side add.
- Residency (`keep_resident`): resident DiT is reused without `_build_dit` (`pipeline_h3.py:881-887`);
  at job end `_park_dit` moves every tensor to CPU and strips wrappers (`:988-990`); failed job or
  no key → released (`:991-993`, `generate` finally `:578-585`). So module-attached extra tensors on a
  resident DiT would persist into the next job unless removed (fact about lifetime only).

## 5. `engine_h3/sft_load.py` and name mapping

- `load_streaming(model, path, header, *, key_map, skip_suffixes=(".comfy_quant",))`
  `sft_load.py:58-118`: per file key → `key_map(key)` (None = skip), duplicate-target check (`:79-80`),
  read each tensor with `engine.sft_quant.sft_reader.read_tensors` (no mmap), one
  `load_state_dict(state, strict=False, assign=True)` (`:90`); unexpected keys logged and dropped
  (`:92-97`); **any tensor still on meta → RuntimeError listing names** (`:98-105`).
- `key_map` for the DiT: `dit_key_map_for(cfg)` `pipeline_h3.py:251-264` = strip `cfg.prefix`, drop
  names in `cfg.unknown`, else the stripped name **as is**. No name map: module attribute names equal
  Comfy-Org tensor names (`dit.py:71-75,214-215`).
- 揺れ absorbed in `h3_format.py`: prefixes `("", "model.diffusion_model.", "model.model.", "net.")`
  (`h3_format.py:45`, `detect_h3_prefix` `:127-136`); config from tensor shapes (`:293-400`); known
  names regex `_DIT_KNOWN` `:164-178`; unknown → warning + dropped (`pipeline_h3.py:459-463`).
- Canonical module paths (state-dict keys minus `.weight`):
  - block i (0..49): `blocks.{i}.attn.qkv_proj`, `blocks.{i}.attn.out_proj`, `blocks.{i}.attn.q_norm`,
    `blocks.{i}.attn.k_norm`, `blocks.{i}.mlp.fc1`, `blocks.{i}.mlp.fc2`, `blocks.{i}.norm1`,
    `blocks.{i}.norm2`, `blocks.{i}.adaln_proj.linear`.
  - token refiner j (0..1): `token_refiner.blocks.{j}.attn.qkv_proj`, `...attn.out_proj`,
    `...attn.q_norm`, `...attn.k_norm`, `...mlp.fc1`, `...mlp.fc2`, `...norm1`, `...norm2`;
    `token_refiner.final_norm`.
  - others: `condition_proj`, `video_patch_proj`, `audio_patch_proj`, `final_layer.norm`,
    `final_layer.adaln_proj.linear`, `final_layer.video_out`, `final_layer.audio_out`, `rope.inv_freq`,
    `adaln_t_table`.
  - `cfg.quant` keys use the same dotted form (`h3_format.py:160`).
- Turbo evidence (hash comparison, official vs `PulpCut_minimax_h3_fl2va_pruned_turbo_int8_convrot.safetensors`,
  whose `__metadata__` says `"lora": ".../minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors", "lora_strength": "1.0"`,
  converted by `h3ddle convert-turbo-package.py`): **DIFFERENT** = `token_refiner.blocks.0.attn.qkv_proj.weight`,
  `token_refiner.blocks.1.mlp.fc2.weight`, `blocks.{0,25}.*.weight_scale` (re-quantized int8 layers);
  **SAME** = `condition_proj`, `blocks.0.adaln_proj.linear`, `blocks.0.norm1`, `blocks.0.attn.q_norm`,
  `video_patch_proj`, `final_layer.video_out`, `final_layer.adaln_proj.linear`, `adaln_t_table`.
  So the merged Turbo LoRA touched block linears **and the bf16 token_refiner linears** (sampled
  tensors only; the LoRA file itself is not in the repo).

## 6. Precision settings (TF32)

- Repo-wide grep (excluding Docs/outputs/venvs/vendor) for `allow_tf32`, `set_float32_matmul_precision`,
  `allow_bf16_reduced_precision_reduction`, `allow_fp16_reduced_precision_reduction`, `fp32_precision`,
  `TF32`: **0 hits in code**. The only torch.backends writes in engine_h3 are cuDNN flags when
  `deterministic`: `engine_h3/pipeline_h3.py:474-475` (`cudnn.deterministic=True`, `cudnn.benchmark=False`).
  (engine25 sets `cudnn.benchmark=False` at `engine25/pipeline25.py:278`.)
- `.venv-engine-h3` deps: no hits in `comfy_kitchen/`, `sageattention/`; in `transformers/` only the
  `enable_tf32` helper (`transformers/utils/import_utils.py:653-675`, uses `torch.backends.fp32_precision`
  on torch ≥ 2.9), called only from `transformers/training_args.py:1614-1627` (not used by inference).
- So the effective precision is torch 2.13's default (not overridden anywhere). Existing f32 islands
  (`_linear_f32`, `H3AdaLNProj`) run under that same default. Prior record: VERIFICATION_LOG line 15728
  (§ on int8 ConvRot dequant) states "TF32 は使っていない（`allow_tf32` 等 Grep 0 件）" and
  `Docs/Outputs-archive/int8-sft-2026-09-26/README_runner.md:573` recorded `allow_tf32 = False` at runtime
  (for the LTX venv, not the H3 venv).

## 7. Pipeline & worker

- Worker protocol `engine_h3/worker.py:26-72`. `load` fields: `transformer_path`, `text_encoder_path`,
  `video_vae_path`, `audio_vae_path`, `tokenizer_dir`, `[blocks_on_gpu]`, `[attention_backend]`,
  `[block_swap_prefetch]`, `[deterministic]` (`_do_load` `:211-262`; fields `:157-158,244-252`).
  `generate` fields whitelist `_GENERATE_FIELDS` `:97-102`: `op, prompt, seed, width, height,
  num_frames, frame_rate, num_inference_steps, shift_video, shift_audio, output_path, images,
  keep_resident, attention_backend, block_swap_prefetch, vram16_mode`; image entry keys `path, frame_idx`
  (`:104`). Unknown keys logged and ignored (`:327-339`).
- Steps / shifts: `_generate_request` `:288-319` → `_value_or(msg, key, default, cast)` (`:177-185`, absent
  or null → default) with `DEFAULT_STEPS=20`, `DEFAULT_SHIFT_VIDEO=12.0`, `DEFAULT_SHIFT_AUDIO=3.0`
  (`constants.py:25-27`). Into `H3GenerateRequest` (`pipeline_h3.py:147-172`); validated in
  `validate_request` (`:175-206`); used in `_denoise`: `sigma_schedule(steps, shift_video/audio)`
  (`pipeline_h3.py:928-929`, loop `:937-967`).
- `load` reads no weights (header checks + `require_cuda_int8`, `pipeline_h3.py:444-510`).
  `generate` (`:512-599`, `torch.no_grad()` at `:542`) runs encode → image_encode → denoise → decode.
- DiT build: `_build_dit` `pipeline_h3.py:677-707`: re-inspect header (`:685`), `with torch.device("meta"):
  dit = H3DiT(cfg); swap_in_int8_linears(...)` (`:692-694`), `load_streaming(..., assign=True)` (`:695`),
  `normalize_dtypes()` (`:698`), `requires_grad_(False)`, `eval()`. Called from `_denoise` only when no
  resident DiT (`:881-887`), then `_vram16_begin` (`:897`), `_place_dit` (`:898`), `refine_text` (`:901`),
  step loop calling `dit(...)` (`:949-960`).
- Natural location for a "load LoRA" step (location only): in `_denoise` between `_build_dit`/resident
  reuse (`pipeline_h3.py:881-887`) and `_place_dit` (`:898`) — i.e. after the meta+assign load and
  `normalize_dtypes`, before tensors are distributed CPU/GPU and before `install` snapshots CPU masters.
  The app side currently refuses `loras` for H3 (`services/engines/h3/adapter.py:201`, REJECT_TABLE),
  and `lora_paths` is accepted only for the shared signature (`adapter.py:424-439`). The load payload
  builder is `adapter.py:384-411`; golden key tests `tests/test_h3_engine.py:945-962`.
- Errors (fail loud): load failure → `_emit("error", detail=_detail(exc))` + `sys.exit(1)`
  (`worker.py:404-410`); generate failure → `_release_after_failed_job()` + `error` event, process stays
  alive (`:413-422`); `_detail` = repr + truncated traceback (`:140-145`). Inside the pipeline: Japanese
  `ValueError`/`RuntimeError` (e.g. `:175-206`, NaN check `:970-971`), vram16 OOM wrapped into
  `VRAM16_OOM_MESSAGE` (`:389`, `:974-977`); `sft_load` raises on meta leftovers; `swap_in_int8_linears`
  raises on unknown layers.

## 8. Tests and gates

- Engine tests: `tests/test_h3_engine.py` (1940 lines), `tests/test_h3_format.py`,
  `tests/test_h3_media.py`, `tests/test_engine_h3_adapter.py`.
- Runner: `scripts/run_h3_pytest.py:1-24` — run with `.venv-engine-h3\Scripts\python.exe
  scripts\run_h3_pytest.py tests\test_h3_engine.py ...`; appends app venv site-packages for pytest,
  `--noconftest`. GPU tests opt-in with `H3_GPU_TESTS=1` (`test_h3_engine.py:778-783,1851-1853`).
- Relevant existing tests: CPU fake of the int8 kernel that runs the real `forward`/`forward_rows`
  (`_fake_int8` `:1560-1568`, kernel as float64), tiny DiT builder via the real meta + swap + load
  path (`_tiny_vram16_dit` `:1571-1586`), vram16 equivalence `test_vram16_block_forward_matches_off`
  (`:1642`), `..._final_rows_matches_off` (`:1672`), `..._dit_forward_matches_off_and_frees_the_embedding`
  (`:1685`), `test_vram16_off_forward_never_calls_lowvram` (`:1717`); GPU `test_gpu_forward_rows_join_equals_forward`
  (`:1875`), `test_gpu_forward_of_row_chunks_equals_forward` (`:1887`), `test_int8_linear_matches_dequantized_bf16`
  (`:783`); block swap: `test_block_swap_service_is_resident_across_jobs` (`:1018`),
  `test_block_swap_wrappers_are_stripped_and_never_doubled` (`:1299`), `test_keep_resident_state_transitions_with_a_tiny_dit`
  (`:1448`); `test_tiny_dit_streams_in_with_no_meta_left` (`:720`); `test_no_inference_mode_under_engine_h3` (`:814`).
- Fingerprint gate: `python -m engine_h3.worker --selftest-generate` (`worker.py:536-601`) calls
  `_do_load`/`_do_generate` with synthesized messages and records `sha256` of the output mp4 per round
  (`:577-578`). Defaults: 640×384, 124 frames, seed 42, 20 steps, shifts 12/3, sdpa, blocks_on_gpu 8,
  prefetch on, deterministic off, vram16 off (`worker.py:439-475`). The reference `0da6111c…` requires the
  G5 prompt; exact command `Docs/Outputs-archive/h3-gate3/g2_selftest_g5prompt.cmd` (official DiT file,
  yellow-raincoat prompt). Design §8 "セルフテスト" (`Docs/MINIMAX_H3_DESIGN.md:365`); results: off and on
  (`--vram16`) both `0da6111c…` (VERIFICATION_LOG §169.3 A-G3/A-G3b).

## 9. Memory numbers and LoRA size arithmetic

Recorded (cite only):
- Per block on GPU: arena slot 371.1 MiB (VERIFICATION_LOG §168.6 item 4); header bytes of block 0
  369.41 MiB. Whole DiT read: 19.53 GiB (`"loaded_gib": 19.53`, job metadata). Token refiner bf16
  weights 1,470 MiB (§168.6 item 4).
- 16GB mode, I2V 896×1152×124 (S=39,764, sage): denoise alloc/reserved 4,411 / 7,262 MiB, cap 0.857 =
  14,033 MiB, ~22 s/step; ring 6 arenas ≈ 2,226 MiB (VERIFICATION_LOG §169.3; design §6.5 "山の内訳").
  Off: 12,807 / 16,828 MiB (design §6.5 conclusion, §169.3). Window 8→4 saves 1,488 MiB (§169.3).

LoRA FP32 A/B per block, rank r = 128, 4 bytes/element; params = r × (in + out):

| Layer | in → out (actual) | r(in+out) | bytes (FP32) |
|---|---|---|---|
| qkv_proj | 5376 → 21504 | 128 × 26,880 = 3,440,640 | 13,762,560 |
| out_proj | 7168 → 5376 | 128 × 12,544 = 1,605,632 | 6,422,528 |
| fc1 | 5376 → 28672 | 128 × 34,048 = 4,358,144 | 17,432,576 |
| fc2 | 14336 → 5376 | 128 × 19,712 = 2,523,136 | 10,092,544 |
| **per block** | | **11,927,552** | **47,710,208 B = 45.50 MiB** |

- 50 blocks: 2,385,510,400 B = 2,275 MiB ≈ 2.22 GiB (FP32). BF16 storage would be half (22.75 MiB/block).
- Relative to block weight 369.41 MiB: +12.3%. If stored as block tensors: arena slot ~371.1 → ~416.6 MiB;
  ring in 16GB mode (4+2=6 slots) +273 MiB; default window 8 (10 slots) +455 MiB; pinned staging 2 slots
  +91 MiB host; CPU masters +2.22 GiB host RAM. (Arithmetic only.)
- Token refiner (if targeted, 2 blocks, same shapes): +91 MiB FP32.
- With the brief's assumed dims (qkv 5376→16128, out 5376→5376, fc2 28672→5376) the total would be
  12,845,056 params = 49.0 MiB/block — those dims do not match this model.

## Open questions / not found

1. No H3 LoRA file (Style or Turbo) exists in the repo; key naming of the Turbo LoRA
   (`minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors`, named only in PulpCut's metadata) and its
   rank/targets are unknown. Hash evidence suggests it also targets bf16 `token_refiner` linears; not
   checked whether it touches `adaln_proj`, all 50 blocks, or `condition_proj` beyond the sampled tensors.
2. Effective TF32 state of torch 2.13 in `.venv-engine-h3` at runtime was not measured (no code sets it;
   default assumed but not observed on this venv). `torch.backends.fp32_precision` API exists on ≥2.9.
3. Whether comfy-kitchen `int8_linear` accepts a non-contiguous or f32 bias / non-bf16 out_dtype in
   ways relevant to an add-on was not inspected (kernel source not read).
4. Sync-path `_patch_block` checks `parameters()` only to decide `blk.to(device)` (`block_swap_service.py:259-262`);
   blocks always have int8 Parameters so buffers move with them in practice — not tested for a
   buffer-only addition.
5. `normalize_dtypes` skips `H3Int8Linear` modules entirely; any f32 tensor registered on other block
   modules (norms, adaln_proj, Attention/MLP containers) before it runs would be cast to bf16 (or kept
   f32 only on the island set). Order of any future attach vs `normalize_dtypes` (`pipeline_h3.py:698`) matters.
6. `VERIFICATION_LOG` §168/§169 contain no LoRA measurements.
