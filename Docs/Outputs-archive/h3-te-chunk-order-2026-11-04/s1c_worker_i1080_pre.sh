#!/usr/bin/env bash
# S1-c (plan §5.3): worker direct call, the TE shape of i1080_243_p9 (N=18,462) with 56 frames, sdpa, --vram16, one round.
# Same arguments for S1-c (pre-fix) and S3-d (post-fix): i1080_cmd in common.sh.  AviUtl2 and the server closed.
# Watch logs/s1c_worker_i1080_pre.console.log in the first minutes: 'encode: 18462 text tokens' must appear; otherwise stop the run.
source "/s/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/outputs/h3-te-equal-chunks-2026-11-04/common.sh"
i1080_cmd pre
run_logged s1c_worker_i1080_pre "${CMD[@]}"; rc=$?
i1080_after pre s1c_worker_i1080_pre | tee -a "$LOGS/s1c_worker_i1080_pre.meta.log"
exit $rc
