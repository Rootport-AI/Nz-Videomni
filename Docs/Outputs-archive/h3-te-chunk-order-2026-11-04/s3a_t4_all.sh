#!/usr/bin/env bash
# S3-a gate 1 (plan §5.3, §5.4): the product's blocked_causal_attention vs stock SDPA (all rows) per N, one process each.
# --tails 256,512 (comma-separated; ranges like 64-575 are also accepted by t4).  --skip-product is off by default,
# so the product comparison runs.  Pass: product.equal true and product.differing_row_count 0 for all 13.
# tail_rows_of_last_block, fold_check and position_check are old-rule fields: NOT used for the verdict.
# Summary afterwards: summarize_t4.py.
source "/s/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/outputs/h3-te-equal-chunks-2026-11-04/common.sh"
rc=0
for n in 513 544 575 576 998 1087 1088 1100 2078 3123 3718 4100 6148; do
  run_logged s3a_t4_n$n "$PY" "$T4" --n $n --tails 256,512 --report "$(wp t4_n$n.json)" || rc=1
done
"$PY" "$(wp summarize_t4.py)" | tee "$W/t4_summary.md"
exit $rc
