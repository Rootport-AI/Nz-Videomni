#!/usr/bin/env bash
cd "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/outputs/h3-vram16-gates"
COMMON=(--width 896 --height 1152 --num-frames 124 --attention-backend sage --blocks-on-gpu 8 --steps 2
  --image "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/uploads/71c540dc-970b-470e-aa0e-856dce76dfdc/input.png"
  --prompt-file "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/outputs/h3-vram16-gates/f0_prompt.txt"
  --keep-resident --deterministic --vram16)



./arm.sh G5c_I2V896_h8_r4096_b8 "${COMMON[@]}" --vram16-head-group 8 --vram16-row-chunk 4096 --vram16-blocks 8
SMALL=(--width 640 --height 384 --num-frames 124 --attention-backend sage --steps 2
  --prompt-file "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/outputs/h3-vram16-gates/g5_prompt.txt"
  --keep-resident --deterministic --vram16)
./arm.sh G5c_T2V640_h8_r4096_b4 "${SMALL[@]}" --blocks-on-gpu 4
./arm.sh G5c_T2V640_h8_r4096_b8 "${SMALL[@]}" --blocks-on-gpu 8
