# スナップショットの集計: `snapshot_step1.pickle`

- 履歴の件数: 26722（種類別 {'segment_alloc': 53, 'alloc': 8922, 'free_requested': 8868, 'free_completed': 8868, 'segment_free': 11}）
- 履歴の頭の状態（逆にたどって復元）: セグメント 0 個・予約 0.0 MiB・割り当て 0.0 MiB

## (1) 割り当て合計が最大になった瞬間

- 位置: 履歴の 16759 番目
- 割り当て 12893.1 MiB／予約 16968.0 MiB／未使用の予約 4074.9 MiB（セグメント 42 個・最大の連続した隙間 2210.0 MiB・small プールの未使用 1.3 MiB）

未使用の連続した隙間の大きさの等級別（large プール）:

| 等級 | 個数 | 合計 MiB |
|---|---:|---:|
| <64MiB | 31 | 77.0 |
| 64-256MiB | 3 | 403.0 |
| 256-1024MiB | 1 | 277.6 |
| 1024-2048MiB | 1 | 1106.0 |
| >2048MiB | 1 | 2210.0 |

新しいセグメントの確保（segment_alloc）: 53 件。うち「直前の空きの合計が要求以上だった」もの 18 件・10482.0 MiB（規則: large プールで同じストリームの空きの合計 ≥ 要求（次の alloc の大きさ）なのに新しいセグメントを確保した）

| 履歴の位置 | セグメント MiB | 要求 MiB | 直前の空きの合計 MiB | 直前の最大の隙間 MiB | 呼び出し元 |
|---:|---:|---:|---:|---:|---|
| 307 | 20.0 | 1.6 | 3.6 | 1.0 | `activations.py:103:forward <- modeling_qwen3_vl.py:525:forward <- modeling_qwen3_vl.py:567:forward <- modeling_layers.py:110:__call__` |
| 16339 | 20.0 | 7.4 | 17.3 | 1.5 | `rope.py:165:rope_tables <- pipeline_h3.py:723:_denoise <- pipeline_h3.py:477:generate <- worker.py:332:_do_generate` |
| 16341 | 16.0 | 14.8 | 29.9 | 12.6 | `rope.py:166:rope_tables <- pipeline_h3.py:723:_denoise <- pipeline_h3.py:477:generate <- worker.py:332:_do_generate` |
| 16343 | 16.0 | 14.8 | 31.1 | 12.6 | `rope.py:167:rope_tables <- pipeline_h3.py:723:_denoise <- pipeline_h3.py:477:generate <- worker.py:332:_do_generate` |
| 16345 | 16.0 | 14.8 | 32.3 | 12.6 | `rope.py:167:rope_tables <- pipeline_h3.py:723:_denoise <- pipeline_h3.py:477:generate <- worker.py:332:_do_generate` |
| 16389 | 20.0 | 8.5 | 33.8 | 7.9 | `dit.py:377:_linear_f32 <- dit.py:493:embed_rows <- pipeline_h3.py:756:_denoise <- pipeline_h3.py:477:generate` |
| 16442 | 416.0 | 414.3 | 884.1 | 410.0 | `dit.py:320:forward <- block_swap_service.py:359:swapped_forward_prefetch <- dit.py:567:forward <- pipeline_h3.py:757:_denoise` |
| 16447 | 830.0 | 828.6 | 885.8 | 410.0 | `dit.py:284:_modulate <- dit.py:320:forward <- block_swap_service.py:359:swapped_forward_prefetch <- dit.py:567:forward` |
| 16449 | 830.0 | 828.6 | 887.2 | 410.0 | `dit.py:287:_modulate <- dit.py:320:forward <- block_swap_service.py:359:swapped_forward_prefetch <- dit.py:567:forward` |
| 16460 | 1658.0 | 1657.3 | 2338.6 | 830.0 | `__init__.py:2190:int8_linear <- __init__.py:1054:int8_linear <- quant_linear.py:112:forward <- dit.py:228:forward` |
| 16467 | 554.0 | 552.4 | 1994.1 | 416.0 | `dit.py:230:forward <- dit.py:321:forward <- block_swap_service.py:359:swapped_forward_prefetch <- dit.py:567:forward` |
| 16475 | 554.0 | 552.4 | 1995.7 | 416.0 | `dit.py:231:forward <- dit.py:321:forward <- block_swap_service.py:359:swapped_forward_prefetch <- dit.py:567:forward` |
| 16497 | 554.0 | 552.4 | 2120.6 | 416.0 | `rope.py:184:apply_rope <- dit.py:233:forward <- dit.py:321:forward <- block_swap_service.py:359:swapped_forward_prefetch` |
| 16533 | 554.0 | 552.4 | 1998.8 | 416.0 | `dit.py:196:_attention <- dit.py:235:forward <- dit.py:321:forward <- block_swap_service.py:359:swapped_forward_prefetch` |
| 16544 | 554.0 | 552.4 | 1448.0 | 416.0 | `quant_per_thread.py:159:per_thread_int8 <- core.py:747:sageattn_qk_int8_pv_fp8_cuda <- core.py:168:sageattn <- dit.py:196:_attention` |
| 16551 | 554.0 | 553.0 | 1447.4 | 416.0 | `quant.py:275:per_channel_fp8 <- core.py:763:sageattn_qk_int8_pv_fp8_cuda <- core.py:168:sageattn <- dit.py:196:_attention` |
| 16621 | 2210.0 | 2209.7 | 6906.3 | 1658.0 | `__init__.py:2190:int8_linear <- __init__.py:1054:int8_linear <- quant_linear.py:112:forward <- dit.py:248:forward` |
| 16628 | 1106.0 | 1104.9 | 6008.9 | 830.0 | `dit.py:249:forward <- dit.py:324:forward <- block_swap_service.py:359:swapped_forward_prefetch <- dit.py:567:forward` |

セグメントごと（その瞬間。大きい順に先頭 40 個）:

| 番地 | プール | ストリーム | 予約 MiB | 使用中 MiB | 未使用 MiB | 最大の隙間 MiB | ブロック数 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0x1667200000 | large | 0 | 2210.0 | 0.0 | 2210.0 | 2210.0 | 0 |
| 0x152ea00000 | large | 0 | 1658.0 | 1657.3 | 0.7 | 0.7 | 1 |
| 0x16f1400000 | large | 0 | 1106.0 | 0.0 | 1106.0 | 1106.0 | 0 |
| 0x14c6e00000 | large | 0 | 830.0 | 552.4 | 277.6 | 277.6 | 1 |
| 0x14fac00000 | large | 0 | 830.0 | 829.5 | 0.5 | 0.5 | 2 |
| 0x1460000000 | large | 0 | 820.0 | 690.5 | 129.5 | 129.5 | 2 |
| 0x1597600000 | large | 0 | 554.0 | 552.4 | 1.6 | 1.6 | 1 |
| 0x15ba000000 | large | 0 | 554.0 | 414.3 | 139.7 | 139.7 | 1 |
| 0x15dca00000 | large | 0 | 554.0 | 552.4 | 1.6 | 1.6 | 1 |
| 0x15ff400000 | large | 0 | 554.0 | 552.4 | 1.6 | 1.6 | 1 |
| 0x1621e00000 | large | 0 | 554.0 | 552.4 | 1.6 | 1.6 | 1 |
| 0x1644800000 | large | 0 | 554.0 | 552.4 | 1.6 | 1.6 | 1 |
| 0x14ace00000 | large | 0 | 416.0 | 414.3 | 1.7 | 1.7 | 1 |
| 0x1493400000 | large | 0 | 410.0 | 276.2 | 133.8 | 133.8 | 1 |
| 0x1350400000 | large | 0 | 372.0 | 371.1 | 0.9 | 0.9 | 1 |
| 0x138b800000 | large | 0 | 372.0 | 371.1 | 0.9 | 0.9 | 1 |
| 0x13a2c00000 | large | 0 | 372.0 | 371.1 | 0.9 | 0.9 | 1 |
| 0x13ba000000 | large | 0 | 372.0 | 371.1 | 0.9 | 0.9 | 1 |
| 0x13d1400000 | large | 0 | 372.0 | 371.1 | 0.9 | 0.9 | 1 |
| 0x13e8800000 | large | 0 | 372.0 | 371.1 | 0.9 | 0.9 | 1 |
| 0x13ffc00000 | large | 0 | 372.0 | 371.1 | 0.9 | 0.9 | 1 |
| 0x1417000000 | large | 0 | 372.0 | 371.1 | 0.9 | 0.9 | 1 |
| 0x142e400000 | large | 0 | 372.0 | 371.1 | 0.9 | 0.9 | 1 |
| 0x1445800000 | large | 0 | 372.0 | 371.1 | 0.9 | 0.9 | 1 |
| 0x130b200000 | large | 0 | 294.0 | 294.0 | 0.0 | 0.0 | 1 |
| 0x1334a00000 | large | 0 | 294.0 | 294.0 | 0.0 | 0.0 | 1 |
| 0x1326c00000 | large | 0 | 222.0 | 220.5 | 1.5 | 1.5 | 1 |
| 0x1372600000 | large | 0 | 222.0 | 220.5 | 1.5 | 1.5 | 1 |
| 0x131d800000 | large | 0 | 148.0 | 147.0 | 1.0 | 1.0 | 1 |
| 0x1347000000 | large | 0 | 148.0 | 147.0 | 1.0 | 1.0 | 1 |
| 0x1380400000 | large | 0 | 74.0 | 73.5 | 0.5 | 0.5 | 1 |
| 0x1384e00000 | large | 0 | 74.0 | 73.5 | 0.5 | 0.5 | 1 |
| 0x1304e00000 | large | 0 | 54.0 | 52.5 | 1.5 | 1.5 | 1 |
| 0x1303a00000 | large | 0 | 20.0 | 14.2 | 5.8 | 5.8 | 4 |
| 0x1308a00000 | large | 0 | 20.0 | 0.0 | 20.0 | 20.0 | 0 |
| 0x145ec00000 | large | 0 | 20.0 | 0.0 | 20.0 | 20.0 | 0 |
| 0x1389800000 | large | 0 | 16.0 | 14.6 | 1.4 | 1.4 | 1 |
| 0x145cc00000 | large | 0 | 16.0 | 14.8 | 1.2 | 1.2 | 1 |
| 0x145dc00000 | large | 0 | 16.0 | 14.8 | 1.2 | 1.2 | 1 |
| 0x130a800000 | small | 0 | 2.0 | 2.0 | 0.0 | 0.0 | 8 |

## (2) その瞬間の最大の確保 20 件

| MiB | ストリーム | 呼び出し元（内側から） |
|---:|---:|---|
| 1657.3 | 0 | `__init__.py:2190:int8_linear <- __init__.py:1054:int8_linear <- quant_linear.py:112:forward <- dit.py:228:forward` |
| 553.0 | 0 | `quant.py:275:per_channel_fp8 <- core.py:763:sageattn_qk_int8_pv_fp8_cuda <- core.py:168:sageattn <- dit.py:196:_attention` |
| 552.4 | 0 | `rope.py:184:apply_rope <- dit.py:233:forward <- dit.py:321:forward <- block_swap_service.py:359:swapped_forward_prefetch` |
| 552.4 | 0 | `rope.py:184:apply_rope <- dit.py:234:forward <- dit.py:321:forward <- block_swap_service.py:359:swapped_forward_prefetch` |
| 552.4 | 0 | `dit.py:196:_attention <- dit.py:235:forward <- dit.py:321:forward <- block_swap_service.py:359:swapped_forward_prefetch` |
| 552.4 | 0 | `dit.py:196:_attention <- dit.py:235:forward <- dit.py:321:forward <- block_swap_service.py:359:swapped_forward_prefetch` |
| 552.4 | 0 | `dit.py:196:_attention <- dit.py:235:forward <- dit.py:321:forward <- block_swap_service.py:359:swapped_forward_prefetch` |
| 552.4 | 0 | `core.py:749:sageattn_qk_int8_pv_fp8_cuda <- core.py:168:sageattn <- dit.py:196:_attention <- dit.py:235:forward` |
| 414.3 | 0 | `dit.py:495:embed_rows <- pipeline_h3.py:756:_denoise <- pipeline_h3.py:477:generate <- worker.py:332:_do_generate` |
| 414.3 | 0 | `dit.py:296:_gate <- dit.py:325:forward <- block_swap_service.py:359:swapped_forward_prefetch <- dit.py:567:forward` |
| 414.3 | 0 | `dit.py:288:_modulate <- dit.py:320:forward <- block_swap_service.py:359:swapped_forward_prefetch <- dit.py:567:forward` |
| 371.1 | 0 | `block_swap_prefetch.py:405:prepare <- block_swap_service.py:321:_build_prefetch_engine <- block_swap_service.py:134:install <- pipeline_h3.py:661:_place_dit` |
| 371.1 | 0 | `block_swap_prefetch.py:405:prepare <- block_swap_service.py:321:_build_prefetch_engine <- block_swap_service.py:134:install <- pipeline_h3.py:661:_place_dit` |
| 371.1 | 0 | `block_swap_prefetch.py:405:prepare <- block_swap_service.py:321:_build_prefetch_engine <- block_swap_service.py:134:install <- pipeline_h3.py:661:_place_dit` |
| 371.1 | 0 | `block_swap_prefetch.py:405:prepare <- block_swap_service.py:321:_build_prefetch_engine <- block_swap_service.py:134:install <- pipeline_h3.py:661:_place_dit` |
| 371.1 | 0 | `block_swap_prefetch.py:405:prepare <- block_swap_service.py:321:_build_prefetch_engine <- block_swap_service.py:134:install <- pipeline_h3.py:661:_place_dit` |
| 371.1 | 0 | `block_swap_prefetch.py:405:prepare <- block_swap_service.py:321:_build_prefetch_engine <- block_swap_service.py:134:install <- pipeline_h3.py:661:_place_dit` |
| 371.1 | 0 | `block_swap_prefetch.py:405:prepare <- block_swap_service.py:321:_build_prefetch_engine <- block_swap_service.py:134:install <- pipeline_h3.py:661:_place_dit` |
| 371.1 | 0 | `block_swap_prefetch.py:405:prepare <- block_swap_service.py:321:_build_prefetch_engine <- block_swap_service.py:134:install <- pipeline_h3.py:661:_place_dit` |
| 371.1 | 0 | `block_swap_prefetch.py:405:prepare <- block_swap_service.py:321:_build_prefetch_engine <- block_swap_service.py:134:install <- pipeline_h3.py:661:_place_dit` |

## (3) 書き出し時点のセグメントの一覧

セグメント 42 個・予約 16968.0 MiB・割り当て 5305.2 MiB

| 番地 | 種別 | ストリーム | 予約 MiB | 使用中 MiB | active MiB | 最大の空きブロック MiB | ブロック数 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0x1667200000 | large | 0 | 2210.0 | 0.0 | 0.0 | 2210.0 | 1 |
| 0x152ea00000 | large | 0 | 1658.0 | 0.0 | 0.0 | 1658.0 | 1 |
| 0x16f1400000 | large | 0 | 1106.0 | 0.0 | 0.0 | 1106.0 | 1 |
| 0x14c6e00000 | large | 0 | 830.0 | 0.0 | 0.0 | 830.0 | 1 |
| 0x14fac00000 | large | 0 | 830.0 | 0.0 | 0.0 | 830.0 | 1 |
| 0x1460000000 | large | 0 | 820.0 | 0.0 | 0.0 | 820.0 | 1 |
| 0x1597600000 | large | 0 | 554.0 | 0.0 | 0.0 | 554.0 | 1 |
| 0x15ba000000 | large | 0 | 554.0 | 0.0 | 0.0 | 554.0 | 1 |
| 0x15dca00000 | large | 0 | 554.0 | 0.0 | 0.0 | 554.0 | 1 |
| 0x15ff400000 | large | 0 | 554.0 | 0.0 | 0.0 | 554.0 | 1 |
| 0x1621e00000 | large | 0 | 554.0 | 0.0 | 0.0 | 554.0 | 1 |
| 0x1644800000 | large | 0 | 554.0 | 0.0 | 0.0 | 554.0 | 1 |
| 0x14ace00000 | large | 0 | 416.0 | 0.0 | 0.0 | 416.0 | 1 |
| 0x1493400000 | large | 0 | 410.0 | 14.6 | 14.6 | 380.7 | 3 |
| 0x1350400000 | large | 0 | 372.0 | 372.0 | 372.0 | 0.0 | 1 |
| 0x138b800000 | large | 0 | 372.0 | 372.0 | 372.0 | 0.0 | 1 |
| 0x13a2c00000 | large | 0 | 372.0 | 372.0 | 372.0 | 0.0 | 1 |
| 0x13ba000000 | large | 0 | 372.0 | 372.0 | 372.0 | 0.0 | 1 |
| 0x13d1400000 | large | 0 | 372.0 | 372.0 | 372.0 | 0.0 | 1 |
| 0x13e8800000 | large | 0 | 372.0 | 372.0 | 372.0 | 0.0 | 1 |
| 0x13ffc00000 | large | 0 | 372.0 | 372.0 | 372.0 | 0.0 | 1 |
| 0x1417000000 | large | 0 | 372.0 | 372.0 | 372.0 | 0.0 | 1 |
| 0x142e400000 | large | 0 | 372.0 | 372.0 | 372.0 | 0.0 | 1 |
| 0x1445800000 | large | 0 | 372.0 | 372.0 | 372.0 | 0.0 | 1 |
| 0x130b200000 | large | 0 | 294.0 | 294.0 | 294.0 | 0.0 | 1 |
| 0x1334a00000 | large | 0 | 294.0 | 294.0 | 294.0 | 0.0 | 1 |
| 0x1326c00000 | large | 0 | 222.0 | 220.5 | 220.5 | 1.5 | 2 |
| 0x1372600000 | large | 0 | 222.0 | 220.5 | 220.5 | 1.5 | 2 |
| 0x131d800000 | large | 0 | 148.0 | 148.0 | 148.0 | 0.0 | 1 |
| 0x1347000000 | large | 0 | 148.0 | 148.0 | 148.0 | 0.0 | 1 |
| 0x1380400000 | large | 0 | 74.0 | 74.0 | 74.0 | 0.0 | 1 |
| 0x1384e00000 | large | 0 | 74.0 | 74.0 | 74.0 | 0.0 | 1 |
| 0x1304e00000 | large | 0 | 54.0 | 52.5 | 52.5 | 1.5 | 2 |
| 0x1303a00000 | large | 0 | 20.0 | 12.1 | 12.1 | 7.9 | 4 |
| 0x1308a00000 | large | 0 | 20.0 | 0.0 | 0.0 | 20.0 | 1 |
| 0x145ec00000 | large | 0 | 20.0 | 0.0 | 0.0 | 20.0 | 1 |
| 0x1389800000 | large | 0 | 16.0 | 0.0 | 0.0 | 16.0 | 1 |
| 0x145cc00000 | large | 0 | 16.0 | 14.8 | 14.8 | 1.2 | 2 |
| 0x145dc00000 | large | 0 | 16.0 | 14.8 | 14.8 | 1.2 | 2 |
| 0x130a800000 | small | 0 | 2.0 | 1.6 | 1.6 | 0.4 | 6 |
| 0x1367800000 | small | 0 | 2.0 | 0.2 | 0.2 | 1.8 | 2 |
| 0x1367e00000 | small | 0 | 2.0 | 1.7 | 1.7 | 0.2 | 27 |
