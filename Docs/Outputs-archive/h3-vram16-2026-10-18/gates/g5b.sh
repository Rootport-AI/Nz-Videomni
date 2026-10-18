#!/usr/bin/env bash
cd "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/outputs/h3-vram16-gates"
COMMON=(--width 896 --height 1152 --num-frames 124 --attention-backend sage --blocks-on-gpu 8 --steps 2
  --image "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/uploads/71c540dc-970b-470e-aa0e-856dce76dfdc/input.png"
  --prompt-file "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/outputs/h3-vram16-gates/f0_prompt.txt"
  --keep-resident --deterministic --vram16)



./arm.sh G5a_h8_r4096_b4 "${COMMON[@]}" --vram16-head-group 8 --vram16-row-chunk 4096 --vram16-blocks 4
./arm.sh G5b_h8_r8192_b4 "${COMMON[@]}" --vram16-head-group 8 --vram16-row-chunk 8192 --vram16-blocks 4
./arm.sh G5b_h8_r2048_b4 "${COMMON[@]}" --vram16-head-group 8 --vram16-row-chunk 2048 --vram16-blocks 4
