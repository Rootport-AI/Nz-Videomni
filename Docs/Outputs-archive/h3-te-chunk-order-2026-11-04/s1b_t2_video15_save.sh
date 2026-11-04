#!/usr/bin/env bash
# S1-b (plan §5.3): pre-fix baseline of the whole TE with the product attention, --refs video15 (N=16,314), --cap
# (--cap = the VRAM 16GB mode allocator cap during encode, same condition as §182.7 G-I3b).  Body must be pre-fix (S0).
source "/s/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/outputs/h3-te-equal-chunks-2026-11-04/common.sh"
run_logged s1b_t2_video15_save "$PY" "$T2" --mode product --refs video15 --cap   --save "$(wp pre_video15.pt)" --report "$(wp pre_video15.json)"
