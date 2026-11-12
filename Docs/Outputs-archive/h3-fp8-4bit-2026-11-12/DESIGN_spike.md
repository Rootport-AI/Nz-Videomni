# Spike design: H3 fp8 / w4a8 prototype layers (2026-11-12)

Implementer-facing design for `h3_quant_spike.py`. The approved plan (Japanese, authoritative on scope and pass criteria) is `C:\Users\<user>\.claude\plans\imperative-crafting-lightning.md`. Where this file and the plan differ, the plan wins. Line numbers refer to the project at dev `35ec3da` and comfy_kitchen 0.2.37 installed in `.venv-engine-h3`.

## 0. Ground rules
- Product code is never edited. Everything lives in `outputs/h3-fp8-w4a8-spike-2026-11-12/`. Patching is by module-attribute assignment inside a context manager that restores in `finally`.
- Fidelity principle: call the SAME comfy_kitchen public functions with the SAME arguments ComfyUI uses. Never re-implement their arithmetic in torch. Our only deliberate differences: fail loud instead of silent fallback; LoRA unsupported in the spike.
- Run from the repo root: `export PYTHONPATH="$PWD" PYTHONUTF8=1; ./.venv-engine-h3/Scripts/python.exe outputs/h3-fp8-w4a8-spike-2026-11-12/h3_quant_spike.py <args>`. `--dry-run` and `--s4` set `CUDA_VISIBLE_DEVICES=-1` before importing torch.

## 1. Command line
`--dry-run`, `--s1`, `--s2`, `--s3 {int8,fp8,fp8_dequant,w4a8_kijai,w4a8_eros}`, `--s3-all` (each condition in its own subprocess, same interpreter, so peaks and patches never mix), `--s4`, `--post` (sheet/hf folder). Copy `Log`, `PATHS`, `build_request`, `run_generation` patterns from `Docs/Outputs-archive/h3-inpaint-2026-11-11/h3_inpaint_spike.py` (`:58-130`, `:400-411`, `:536-626`). Default settings: 1280×768, 73 frames, seed 1234, `ATTENTION="sage"`, `VRAM16=True`, `BLOCKS_ON_GPU=8`, `PREFETCH=True`, `keep_resident=False`, one fixed T2V prompt (write it as a constant; English, neutral content, with audible sound, e.g. a street scene with footsteps and traffic).

Files (all in `models/MiniMaxH3/Weights/`): `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (baseline), `minimax_h3_fl2va_pruned_fp8_scaled.safetensors`, `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` (Kijai; marker ONLY in header metadata), `10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors` (20 int8 per_row layers = blocks 0,1,47,48,49; 180 w4a8 with G 16/8/4 and F32 codebook; markers in both carriers; `adaln_t_table` BF16). Text encoder / VAEs / tokenizer: same as the inpaint spike `PATHS`.

## 2. Spec type
`QuantSpec` (`h3_format.py:54-58`) is frozen with 3 fields; int8 layers keep it. Add:
```python
@dataclass(frozen=True)
class SpikeSpec:
    format: str                 # "float8_e4m3fn" | "asym_w4a8_int8"
    convrot: bool               # fp8: False; w4a8: True   (read by pipeline_h3.py:812-813 for EVERY quant entry)
    convrot_groupsize: int      # fp8: 0; w4a8: 256
    group_size: int | None = None
    full_precision: bool = False
    has_input_scale: bool = False
    has_codebook: bool = False
    marker_source: str = "tensor"   # "tensor" | "metadata" | "both"
```
Global `SPIKE` object: `fp8_force_dequant: bool`, `dequant_cache: bool` (read at call time); `COUNTS: dict[str,int]`.

## 3. `inspect_h3_dit_spike(path)`
Copy the body of `h3_format.inspect_h3_dit` (`:295-402`) and reuse its private helpers; do NOT patch `h3_format._shape` / `_DIT_KNOWN` globally (`inspect_h3_te` uses them).
1. Markers: `meta = json.loads(header.metadata.get("_quantization_metadata","{}")).get("layers",{})` (keys may be unprefixed or prefixed; strip the detected prefix). Read every `<layer>.comfy_quant` tensor with one file handle (seek/read like `:263-285`). If both carriers exist for a layer, compare `format/group_size/convrot/convrot_groupsize`; refuse on mismatch. `full_precision_matrix_mult` is read from the tensor marker only. Options may sit under `params` (top level wins), as `parse_comfy_quant` does.
2. Classify layers by marker format, NOT by weight dtype (w4a8 weights are I8 too). Per format, check every tensor exactly:
   - fp8: `weight` F8_E4M3 `[N,K]`; `weight_scale` F32 `[]`; optional `input_scale` F32 `[]`. A `scaled` layer without `input_scale` → refuse with a reason (not present in these files).
   - w4a8: `weight` I8 `[N,K/2]`; `weight_s_rel` F8_E4M3 `[N,K/G]`; `weight_s_channel` F32 `[N]`; optional `weight_codebook` F32 `[16]`; refuse any `*_correction`; require `convrot` True and `convrot_groupsize` 256; require `K % G == 0` and (`16 % G == 0` or `G % 16 == 0`) (mirrors `AsymW4A8Int8Layout.Params`, `tensor/w4a8_int8.py:154-168`).
3. Shape table (`:359-379`): run on a logical view where each w4a8 weight's shape is `(N, 2*(K/2))`, via `dataclasses.replace(TensorInfo, shape=...)`.
4. int8 layers: call the original `h3_format._quant_layers` on the tensor table minus every tensor of fp8/w4a8 layers (keeps the stray check and all int8 checks unchanged).
5. `unknown`: extend known names per format — int8 `{weight_scale, comfy_quant}`, fp8 `{weight_scale, input_scale, comfy_quant}`, w4a8 `{weight_s_rel, weight_s_channel, weight_codebook, comfy_quant}`. Eros's `adaln_basis`/`adaln_mean` stay unknown (dropped with WARNING; record).
6. Return `(header, H3DiTConfig(... quant={int8: QuantSpec, others: SpikeSpec}, unknown=...))`.

## 4. Prototype layers (both subclass `H3Int8Linear`; `__init__` calls `nn.Module.__init__(self)` directly and never `super().__init__`)
Common: set `self.in_features`, `self.out_features` (int), `self.spec` (the SpikeSpec), `register_parameter("bias", None)` (refuse if the replaced Linear had a bias), override `forward`, `forward_rows(x, n0, n1, lora_xa=None)`, `forward_rows_q(xq, xs, n0, n1, lora_xa=None)`, `shares_input(x)`, `quantize_input(x)`, `extra_repr`. Raise `NotImplementedError` if `lora_xa is not None` or `getattr(self, "_lora_specs", None)`. Inputs: reshape to 2-D, `.to(bfloat16)`, `.contiguous()`; output reshaped back to `[*lead, n]`. Count every call in `COUNTS` (keys below).

### `H3Fp8Linear`
- `weight = nn.Parameter(torch.empty(N, K, dtype=torch.float8_e4m3fn), requires_grad=False)`; `register_buffer("weight_scale", torch.empty((), torch.float32))`; `input_scale`: buffer `torch.empty((), float32)` or `register_buffer("input_scale", None)` when absent (None is skipped by state_dict, the meta check `sft_load.py:51-55`, and the arena `block_swap_prefetch.py:296-300`).
- `_mode()`: `"dequant"` if `spec.full_precision` or `SPIKE.fp8_force_dequant` or `torch.cuda.get_device_capability(weight.device) < (8, 9)`; else `"scaled"` (raise if `input_scale is None`).
- scaled: `xq = ck.quantize_per_tensor_fp8(x2d, self.input_scale)` (signature `quantize_per_tensor_fp8(x, scale, output_type=float8_e4m3fn)`, `comfy_kitchen/__init__.py:387`; CUDA constraints need float32 scale on CUDA, `cuda/__init__.py:3915-3940`) → `scaled_mm_v2(xq.contiguous(), self.weight.t(), scale_a=self.input_scale, scale_b=self.weight_scale, bias=None, out_dtype=torch.bfloat16)` from `comfy_kitchen.scaled_mm_v2` (passes `use_fast_accum=False`). COUNTS: `fp8_scaled_calls`.
- dequant: `F.linear(x2d, self._w_bf16())`, where `_w_bf16()` = `ck.dequantize_per_tensor_fp8(self.weight, self.weight_scale, torch.bfloat16)` (`__init__.py:406`), with the per-block cache (below). COUNTS: `fp8_deq_miss`, `fp8_deq_hit`, `fp8_deq_uncached`.
- `forward_rows`: assert `n0 % 16 == 0 and (n1-n0) % 16 == 0`; scaled → `self.weight[n0:n1].t()` (strides (1,K) = column-major; `F.scaled_mm` needs K % 16 and both mat2 dims % 16: 5376/7168/14336 and 1024 pass); dequant → `ck.dequantize_per_tensor_fp8(self.weight[n0:n1], self.weight_scale, bf16)` uncached. COUNTS: `fp8_rows`.
- `shares_input(x)`: `_mode()=="scaled" and x.is_cuda and x.device == weight.device == input_scale.device`. `quantize_input(x)` → `(ck.quantize_per_tensor_fp8(x2d, input_scale), self.input_scale)` (second value goes into lowvram's `xs` slot, `lowvram.py:150`). COUNTS: `fp8_quantize_input`. `forward_rows_q(xq, xs, n0, n1)` → `scaled_mm_v2(xq, w[n0:n1].t(), scale_a=xs, scale_b=weight_scale, ...)`. COUNTS: `fp8_rows_q`.
- Dequant cache: module global `_DEQ_CACHE: dict | None = None`. Patch `engine_h3.lowvram.block_forward` (resolved at call time by `dit.py:372-374`):
  ```python
  def _bf(block, *a, **k):
      global _DEQ_CACHE; prev, _DEQ_CACHE = _DEQ_CACHE, ({} if SPIKE.dequant_cache else None)
      try: return _orig_bf(block, *a, **k)
      finally: _DEQ_CACHE = prev
  ```
  `_w_bf16()`: if `_DEQ_CACHE is not None`: key `id(self)`, value `(weight.data_ptr(), w_bf16)`; on hit assert the data_ptr is unchanged (arena rebinds `.data`). The normal (non-VRAM16) path calls fc2 once per block and needs no cache. VRAM cost: fc2 147 MiB; with `fp8_force_dequant` fc1 too (441 MiB total).

### `H3W4A8Linear`
- `weight = nn.Parameter(torch.empty(N, K//2, dtype=torch.int8), requires_grad=False)`; buffers `weight_s_rel` `empty(N, K//G, float8_e4m3fn)` (file is F8 already; no `.view` needed), `weight_s_channel` `empty(N, float32)`, `weight_codebook` `empty(16, float32)` or `None`.
- `forward`: `ck.w4a8_int8_linear(x2d, self.weight, self.weight_s_rel, self.weight_s_channel, self.weight_codebook, None, None, G, 256, torch.bfloat16)` (signature `w4a8_int8_linear(x, qdata, s_rel, s_channel, codebook=None, correction=None, bias=None, group_size=16, convrot_groupsize=256, out_dtype=bf16)`, `tensor/w4a8_int8.py:101-127`). The kernel quantizes x itself. COUNTS: `w4a8_calls`.
- `forward_rows`: slice `weight[n0:n1]`, `weight_s_rel[n0:n1]`, `weight_s_channel[n0:n1]`; assert `n0 % 16 == 0`. COUNTS: `w4a8_rows`.
- `shares_input` → False; `quantize_input`/`forward_rows_q` raise.
- SwiGLU stays as the engine does it (`dit.py:296-298`, bf16 `F.silu(gate) * value` before fc2) — identical to comfy_kitchen 0.2.41's fused version bit for bit.

## 5. `swap_spike(root, quant, *, bias_dtype_of)`
Split `quant` into `QuantSpec` entries (pass to the ORIGINAL `swap_in_int8_linears`) and `SpikeSpec` entries (same parent lookup and checks as `quant_linear.py:512-535`: `rpartition(".")`, `root.get_submodule(parent)`, must be `nn.Linear`, build the new class under `torch.device("meta")`, `setattr`). Return the total count.

## 6. Patches (context manager `spike_patches()`, entered BEFORE `pipe.load()` and kept until `generate` returns; `_build_dit` runs per job from `generate`, `pipeline_h3.py:1796`)
- `h3_format.inspect_h3_dit` → `inspect_h3_dit_spike` (called at `pipeline_h3.py:775` and `:1546`).
- `engine_h3.quant_linear.swap_in_int8_linears` → `swap_spike` (function-local import at `:1541`). Text encoder unaffected (`text_encoder.py:76` binds at import).
- `engine_h3.lowvram.block_forward` → cache wrapper.
- `comfy_kitchen.backends.cuda.eager_w4a8_int8_linear` (module global imported at `cuda/__init__.py:211`, called at `:2531`) → counting wrapper, COUNTS `w4a8_eager`. `comfy_kitchen.backends.cuda._dequant_int4_grouped_to_int8` (defined `:2292`; the 2-pass path, NOT eager) → counting wrapper, COUNTS `w4a8_twopass` (information only).
- Wrap `engine_h3.dit.H3DiT.normalize_dtypes` to record a dtype audit after the original runs: for every H3Fp8Linear/H3W4A8Linear/H3Int8Linear list (name, dtype) of weight and scales; assert fp8/int8 weights and F32 scales unchanged; record `adaln_t_table` dtype (must be F32).
- Do NOT patch `load_streaming` (VAEs use it too). `.comfy_quant` is skipped by default.

## 7. Dry run (CPU; `CUDA_VISIBLE_DEVICES=-1`)
For all 4 files: (a) the ORIGINAL `inspect_h3_dit` result or its `H3FormatError` message; (b) the spike inspector summary (per-format counts, G histogram, codebook/full_precision/input_scale counts, marker_source, unknown list); (c) build `H3DiT(cfg)` on meta, apply `swap_spike`, then compare `(shape, dtype)` of every header key mapped by `pipeline_h3.dit_key_map_for(cfg)` against `dit.state_dict(keep_vars=True)` — report missing keys in both directions (must be zero except `.comfy_quant` and the known-unknown `adaln_basis/adaln_mean`); (d) `normalize_dtypes()` on meta + dtype audit; (e) S4: `sft_quant_format.layer_schemes(path, header, prefix)` on the 3 new files — accepted layers per scheme and refusal reasons (catch the exception text per file). Write `dry_run.json`.

## 8. S1 (GPU, bit-identity)
Setup like `load()`: `require_cuda_int8()`, `require_fp32_matmul()`, `pin_cublas_workspace(dev)` (`pipeline_h3.py:455/476`) — in that order. Record for `quantize_per_tensor_fp8`, `dequantize_per_tensor_fp8`, `w4a8_int8_linear`, `int8_linear`: `registry.get_implementation(name, kwargs=<real kwargs>).__module__`; also `has_scaled_mm_v2()`, `comfy_kitchen.backends.cuda._W4A8_CHUNKED`, env `COMFY_KITCHEN_W4A8_CHUNKED` (must be unset), ck/torch versions, device name/capability.
Weights: `engine.sft_quant.sft_reader.read_tensors(path, keys, header)` (seek/readinto; no mmap). Layers: fp8 and Kijai → blocks 0/25/49 × {qkv_proj, out_proj, fc1, fc2}; Eros → first layer of each (G, type) pair. Build modules on CPU, `load_state_dict(assign=True)`, `.cuda()`.
Inputs: M ∈ {4096, 3977, 39764}; seed `20261112 + 1000*block + layer_idx` on a CUDA generator; `randn` f32 × σ → bf16; σ = 1.0; for fp8 scaled layers also σ = 448·input_scale/4 and σ = 448·input_scale (saturating).
References (comfy_kitchen 0.2.37, `comfy_kitchen.tensor`): fp8 scaled → `xq = QuantizedTensor.from_float(x2d, "TensorCoreFP8Layout", scale=input_scale)`; `wq = QuantizedTensor(w, "TensorCoreFP8Layout", TensorCoreFP8Layout.Params(scale=weight_scale, orig_dtype=torch.bfloat16, orig_shape=tuple(w.shape)))`; `ref = F.linear(xq, wq)` (`tensor/fp8.py:134-165`). fp8 dequant → **`F.linear(x2d, wq.dequantize())`** — this is ComfyUI's actual path for `full_precision_matrix_mult` layers and for Ampere (`comfy/ops.py`: `if self._full_precision_mm and isinstance(weight, QuantizedTensor): weight = weight.dequantize()` then `self._forward(input, weight, bias)`; `QuantizedTensor.dequantize` → `tensor/base.py:308` → `TensorCoreFP8Layout.dequantize` → `ck.dequantize_per_tensor_fp8`, `tensor/fp8.py:70-71`). CORRECTION 2026-11-12: the first version of this design used `F.linear(x2d, wq)` with the QuantizedTensor operand; that goes through comfy_kitchen's `t`→`mm` handlers and multiplies a dequantized, transposed contiguous K×N matrix, which cuBLAS may schedule differently (fc2, K=14,336, differed at M=4096/39,764 in S1 attempt 1). It is NOT ComfyUI's path; keep it only as an information row. w4a8 → `AsymW4A8Int8Layout.Params(scale=s_rel, s_channel=..., codebook=..., orig_dtype=bf16, orig_shape=(N,K), group_size=G, convrot_groupsize=256)`; `F.linear(x2d, QuantizedTensor(qdata, "AsymW4A8Int8Layout", params))`. Attach a counting handler to `logging.getLogger("comfy_kitchen.tensor.fp8")` (and the w4a8 module logger): any WARNING = fail (silent fallback in the reference).
Comparisons (pass = `torch.equal` unless noted): ① full vs reference (fp8 scaled, fp8 dequant, w4a8); ② w4a8 21×1,024-row `forward_rows` slices concatenated vs full — expected equal; if not, RECORD the diff (count, max) and continue (chunk_cols depends on n, `cuda/__init__.py:880`); fc1 sliced in 1,024-row blocks likewise; ③ fp8 slices vs full: information — count of differing elements, max and p99.9 difference in bf16 ulps (`i = t.view(int16).int(); o = torch.where(i < 0, -(i & 0x7FFF), i)`); ④ fp8 `forward_rows_q(*quantize_input(x), n0, n1)` vs `forward_rows` — equal; ⑤ fc2 cached vs uncached — equal; ⑥ sanity vs f32 `x @ W_deq.T` (`W_deq` = `ck.dequantize_w4a8_int8_weight(...)` or fp8 dequantize) — relative L2 < 0.2, applied ONLY to non-saturating σ (saturating σ relies on ① alone). Write `s1_result.json`.

## 9. S2 (GPU, per-layer timing)
Block 10 real weights per file. `torch.cuda.Event`, 3 warm-ups, median of 10, `synchronize`. Shapes: (a) normal M=39,764: qkv [21504,5376], out_proj [5376,5376], fc1 [28672,5376], fc2 [5376,14336]; (b) VRAM16: qkv slices N=1,024 at M=39,764 (×21 total), out_proj M=39,764, fc1/fc2 at M=3,977. Variants: `ck.int8_linear` (official int8, convrot=True), fp8 scaled, fp8 dequant+linear (dequant timed separately), w4a8 G=16/8/4 (Eros layers). Transient memory: `reset_peak_memory_stats()`, peak − (inputs + outputs + weights). Estimate per-step linear time in VRAM16: `50·[21·t_qkv_slice + t_out + C·(t_fc1 + t_fc2)] + 50·t_deq(fc2)`, C = ceil(S/4096). Flag if fp8 scaled > 1.5× int8. Write `s2_result.json`.

## 10. S3 (GPU, end-to-end), each condition in its own subprocess
| Condition | File | Steps | Shift v/a | Switches |
|---|---|---|---|---|
| int8 | official int8 | 20 | 12/3 | patches installed but no spike layers (delegation check) |
| fp8 | official fp8 | 20 | 12/3 | `fp8_force_dequant=False`, cache on |
| w4a8_kijai | Kijai | 20 | 12/3 | — |
| w4a8_eros | Eros TURBO-hybrid | 8 | 10/3 | — |
| fp8_dequant | official fp8 | 20 | 12/3 | `fp8_force_dequant=True`, cache on (fc1+fc2); run LAST, only if the four above passed |
Shared: 1280×768, 73 frames, sage, `vram16_mode=True`, prefetch on, `keep_resident=False`, seed 1234, same prompt, `loras=()`. Check before any GPU run that no `run.ps1`/uvicorn server is using the GPU (`nvidia-smi` process list).
`result.json` keys: `condition, file, conditions{...}, spike{fp8_force_dequant, dequant_cache}, inspect{...}, modules{H3Int8Linear, H3Fp8Linear, H3W4A8Linear, dtype_audit}, backends{list_backends, impl_modules, has_scaled_mm_v2, w4a8_chunked}, load_seconds, engine{seconds, peak_vram_mb, peak_vram_reserved_mb, h3.phases}, sec_per_step, counts, self_check{expected, actual, passed}, outputs, status, error, traceback`.
Self-checks: T = steps; S = sequence length from the `generate` record (`h3.sequence` or phases); C = ceil(S/4096) must be an integer consistent with observed fc2 calls ÷ (50·T) (expect ≈6 at this size). B = quantized blocks of that format (50; Eros w4a8 45). fp8: `fp8_quantize_input = 50T`, `fp8_rows_q = 1050T`, `fp8_deq_miss = 50T`, `fp8_deq_hit = 50T(C−1)`, `fp8_scaled_calls = 50T(22+C)`... (count exact: scaled calls = rows_q + fc1 per chunk + out_proj → derive and assert). fp8_dequant: `fp8_scaled_calls = 0`; qkv slice dequant = 1050T; fc1 and fc2 misses 50T each. w4a8: `w4a8_calls + w4a8_rows = B·T(22+2C)`, `w4a8_eager = 0`. Every run: dtype audit passed; `adaln_t_table` F32.
Outputs: mp4 per condition under the spike folder; `summary.md` table (condition, load s, total s, s/step, peak VRAM alloc/reserved, self-check, notes); `--post` makes a contact sheet per clip (ffmpeg, like the inpaint spike) and copies clips + README into `hf-round16/`.

## 11. Known risks (from the plan §6) and what to do
- `F.scaled_mm` with 0-dim scales / column-major slice fails → record the error text and stop (do not work around with copies silently; a `.contiguous()` on the INPUT is fine and expected).
- VRAM shortfall in `fp8_dequant` with the 441 MiB cache → rerun with fc1 cache off (`SPIKE.dequant_cache` per-layer switch), record.
- Eros int8 per_row layers and BF16 `adaln_t_table` → expected to pass through the original int8 path and `normalize_dtypes`; if not, record and stop.
- Any ambiguity → stop and report to the supervisor with the exact error text and numbers.
