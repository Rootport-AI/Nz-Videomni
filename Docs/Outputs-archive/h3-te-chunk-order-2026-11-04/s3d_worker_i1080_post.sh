#!/usr/bin/env bash
# S3-d (plan §5.3): worker direct call, the TE shape of i1080_243_p9 (N=18,462) with 56 frames, sdpa, --vram16, one round.
# Same arguments for S1-c (pre-fix) and S3-d (post-fix): i1080_cmd in common.sh.  AviUtl2 and the server closed.
# Watch logs/s3d_worker_i1080_post.console.log in the first minutes: 'encode: 18462 text tokens' must appear; otherwise stop the run.
source "/s/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/outputs/h3-te-equal-chunks-2026-11-04/common.sh"
i1080_cmd post
run_logged s3d_worker_i1080_post "${CMD[@]}"; rc=$?
i1080_after post s3d_worker_i1080_post | tee -a "$LOGS/s3d_worker_i1080_post.meta.log"
exit $rc
