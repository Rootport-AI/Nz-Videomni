#!/usr/bin/env bash
# S3-c (plan §5.3): post-fix TE vs the S1 pre-fix .pt, pictures9 then video15 (one process each), --cap.
# Pass (§5.4): vs_baseline.bit_identical true, tokens_match_expected true, allocator.num_ooms 0, attention_calls 0.
source "/s/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/outputs/h3-te-equal-chunks-2026-11-04/common.sh"
rc=0
for refs in pictures9 video15; do
  run_logged s3c_t2_${refs}_compare "$PY" "$T2" --mode product --refs $refs --cap \
    --baseline "$(wp pre_$refs.pt)" --report "$(wp post_$refs.json)" || rc=1
done
exit $rc
