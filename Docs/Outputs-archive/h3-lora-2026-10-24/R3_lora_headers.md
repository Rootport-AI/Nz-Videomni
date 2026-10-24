# R3 MiniMax H3 LoRA headers (header-only; scripts insp.py / chk.py in this folder)
Raw key lists: R3_keys_turbo.txt, R3_keys_style.txt, R3_keys_motion.txt (key, dtype, shape). Raw output: out_*.txt.

Common: H3 hidden 5376; attn q/k/v each 7168 (qkv fused out 21504 = 3x7168; out_proj in 7168); mlp fc1 out 28672 (=2x14336, SwiGLU), fc2 in 14336. 50 main blocks.

## Turbo: minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors (1,956,193,000 B)
- Metadata: training_scale 0.0625; qkv_fusion "block diagonal B; concat A; alpha multiplied by 3"; swi_glu_mapping "Diffusers [value;gate] -> ComfyUI [gate;value]"; training_rank 128; target_format "ComfyUI generic LoRA"; source_format "Diffusers PEFT LoRA"; format pt; base_model "Comfy-Org/MiniMax-H3 minimax_h3_fl2va_bf16.safetensors"; training_alpha 8.0.
- 624 keys; BF16 x416 (A/B), F32 x208 (alpha scalars); tensor bytes 1,956,119,360.
- Scheme: ComfyUI-style `diffusion_model.<path>.lora_A.weight / .lora_B.weight / .alpha`.
- Examples: diffusion_model.blocks.0.attn.out_proj.alpha ; ...blocks.0.attn.qkv_proj.alpha ; ...blocks.0.mlp.fc1.alpha ; ...blocks.0.mlp.fc2.alpha ; (A/B keys follow the same pattern, see keys file).
- Modules (blocks 0-49 all, no gaps; token_refiner.blocks 0-1 too), prefix `diffusion_model.`:

  | module | A | B | rank | alpha |
  |---|---|---|---|---|
  | blocks.{i}.attn.qkv_proj | [384,5376] | [21504,384] | 384 | 24.0 |
  | blocks.{i}.attn.out_proj | [128,7168] | [5376,128] | 128 | 8.0 |
  | blocks.{i}.mlp.fc1 | [128,5376] | [28672,128] | 128 | 8.0 |
  | blocks.{i}.mlp.fc2 | [128,14336] | [5376,128] | 128 | 8.0 |
  | token_refiner.blocks.{i}.* (i=0,1) | same four shapes | | 384 / 128 | 24 / 8 |
- qkv is ONE fused module. No single_blocks, AdaLN/mod/norm, patch_proj, final layer or audio-specific keys.
- Non-A/B keys: only the 208 `.alpha` scalars (F32). Alpha: 156 x 8.0, 52 x 24.0 (the 52 = qkv of 50 blocks + 2 token_refiner). Effective scale alpha/rank = 0.0625 everywhere (matches training_scale).
- Ranks differ: qkv 384 (=3x128), others 128.
- Block-diagonal check (blocks.0 qkv_proj lora_B, 21504x384 BF16, 16.5 MB; also token_refiner.blocks.0): B split into 3 row segments (Q/K/V, 7168 rows each) x 3 column slices (128 each): diagonal blocks fully nonzero (917504 of 917504 entries), all 6 off-diagonal blocks exactly 0. A's three 128-row slices differ from each other. CONFIRMED: rank 384 block-diagonal (training rank 128, alpha 8 -> 24 after x3). Only those two tensors were checked.

## Style: 3dsrx_minimax_h3.safetensors (298,260,424 B)
- Metadata (kohya-style trainer, module networks.lora_minimax_h3): ss_network_dim 16, ss_network_alpha 16.0, ss_minimax_h3_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2", ss_base_model_version minimax_h3, ss_sd_model_name minimax_h3_fl2va_pruned_int8_convrot.safetensors, ss_minimax_h3_base_family fl2va, ss_minimax_h3_task t2va, ss_minimax_h3_one_frame True, ss_minimax_h3_video_only True, ss_minimax_h3_convrot_int8 True, ss_mixed_precision bf16, ss_full_bf16 False, ss_fp8_base False, optimizer AdamW8bit, lr 1e-4, ss_steps 1250 (epoch 23/28), shift_video 12.0 / shift_audio 3.0, ss_mode_scale 1.29, seed 42, modelspec.architecture "MiniMax-H3/lora", modelspec.title 3dsrx_minimax_h3_rank16, resolution 1280x720, date 2026-09-26T20:40:11, dataset 54 images 512x512 "3d_animation", plus more ss_minimax_h3_* guidance/loss fields and sshs hashes (full list in out_style.txt).
- 600 keys; all F32; tensor bytes 298,189,600.
- Scheme: kohya: `lora_unet_blocks_{i}_attn_qkv_proj.lora_down.weight / .lora_up.weight / .alpha` (module path dots flattened to underscores).
- Examples: lora_unet_blocks_0_attn_out_proj.alpha ; ..._attn_out_proj.lora_down.weight [16,7168] ; ..._lora_up.weight [5376,16] ; lora_unet_blocks_0_attn_qkv_proj.{alpha, lora_down.weight [16,5376], lora_up.weight [21504,16]} ; lora_unet_blocks_0_mlp_fc1.{alpha, down [16,5376], up [28672,16]} ; lora_unet_blocks_0_mlp_fc2.alpha.
- Modules (blocks 0-49 all, no gaps), rank 16 everywhere, alpha 16.0 on all 200 (scale 1.0): attn_qkv_proj down [16,5376] up [21504,16] (fused, plain rank 16); attn_out_proj [16,7168]/[5376,16]; mlp_fc1 [16,5376]/[28672,16]; mlp_fc2 [16,14336]/[5376,16].
- No token_refiner, single_blocks, AdaLN, norm, patch, final, audio keys. Only non-pair keys: the 200 alpha scalars. Ranks uniform.

## Motion: camera_motion_h3_lora_v1_3000_pruned.safetensors (155,109,680 B)
- Metadata: adaln_keys_removed "100"; note "All AdaLN LoRA adapters removed: source targets pruned model AdaLN input dim 8."; source_file "yunjing2_camera_motion_lora_v1_000003000"; converted_by "convert_h3_lora.py". No ss_* keys.
- 416 keys; all F16; tensor bytes 155,058,176.
- Scheme: ComfyUI-style `diffusion_model.<path>.lora_A.weight / .lora_B.weight`, NO alpha keys (scale 1.0 by convention, rank 16).
- Examples: diffusion_model.blocks.0.attn.out_proj.lora_A.weight [16,7168] ; ...lora_B.weight [5376,16] ; ...attn.qkv_proj.lora_A.weight [16,5376] ; ...lora_B.weight [21504,16] ; ...mlp.fc1.lora_A.weight [16,5376] ; ...fc1.lora_B.weight [28672,16] ; ...mlp.fc2.lora_A.weight [16,14336] ; ...fc2.lora_B.weight [5376,16].
- Modules: blocks 0-49 (no gaps) x {attn.qkv_proj, attn.out_proj, mlp.fc1, mlp.fc2} plus token_refiner.blocks 0-1 x the same four; shapes identical to Style's, in A/B naming. All rank 16. qkv fused, plain rank 16 by shape.
- The 100 AdaLN adapter keys (50 blocks x A/B) were removed by the converter; no AdaLN/mod keys remain. Only A/B pairs; no other keys.

## Comparison
| | Turbo | Style | Motion |
|---|---|---|---|
| format family | ComfyUI (from diffusers PEFT) | kohya | ComfyUI |
| key prefix | diffusion_model. | lora_unet_ (underscored) | diffusion_model. |
| suffix | lora_A/lora_B.weight + .alpha | lora_down/lora_up.weight + .alpha | lora_A/lora_B.weight, no alpha |
| dtype | BF16 (+F32 alpha) | F32 | F16 |
| rank | 384 qkv (3x128 block-diag), 128 others | 16 | 16 |
| alpha | 24 qkv / 8 others (scale 0.0625) | 16 (scale 1) | none |
| modules | qkv_proj,out_proj,fc1,fc2 | same four | same four |
| blocks | 0-49 + token_refiner 0-1 | 0-49 only | 0-49 + token_refiner 0-1 |
| size | 1.96 GB | 298 MB | 155 MB |
| oddities | block-diag qkv, SwiGLU order converted to ComfyUI [gate;value], 4-step turbo | trained on int8-convrot pruned base, one-frame | AdaLN adapters stripped; pruned base |

Declared dtypes (bf16/fp32/fp16) all match; Turbo also carries F32 alphas.

## Open questions
- Style/Motion qkv B: not checked for internal zero structure (assumed plain dense rank 16 over fused qkv).
- Motion has no alpha: loader must default alpha=rank (scale 1.0); confirm intended.
- Style trained on an int8-convrot pruned base and Motion on a "pruned model" (AdaLN input dim 8): do dims and fc1 gate/value ordering match our target base? Style/Motion fc1 order unverified against Turbo's explicit "[gate;value]" ComfyUI order.
- token_refiner LoRA is present in Turbo/Motion, absent in Style: loader must tolerate missing modules.
- Turbo qkv block-diagonality verified only on blocks.0 and token_refiner.blocks.0; other blocks assumed the same.
