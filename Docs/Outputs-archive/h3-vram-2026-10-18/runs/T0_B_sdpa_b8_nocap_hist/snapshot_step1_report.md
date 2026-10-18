# スナップショットの集計: `snapshot_step1.pickle`

- 履歴の件数: 24315（種類別 {'segment_alloc': 46, 'alloc': 8122, 'free_requested': 8068, 'free_completed': 8068, 'segment_free': 11}）
- 履歴の頭の状態（逆にたどって復元）: セグメント 0 個・予約 0.0 MiB・割り当て 0.0 MiB

## (1) 割り当て合計が最大になった瞬間

- 位置: 履歴の 16728 番目
- 割り当て 6661.2 MiB／予約 7588.0 MiB／未使用の予約 926.8 MiB（セグメント 35 個・最大の連続した隙間 192.0 MiB・small プールの未使用 2.7 MiB）

未使用の連続した隙間の大きさの等級別（large プール）:

| 等級 | 個数 | 合計 MiB |
|---|---:|---:|
| <64MiB | 23 | 102.2 |
| 64-256MiB | 7 | 821.8 |
| 256-1024MiB | 0 | 0.0 |
| 1024-2048MiB | 0 | 0.0 |
| >2048MiB | 0 | 0.0 |

新しいセグメントの確保（segment_alloc）: 46 件。うち「直前の空きの合計が要求以上だった」もの 11 件・2056.0 MiB（規則: large プールで同じストリームの空きの合計 ≥ 要求（次の alloc の大きさ）なのに新しいセグメントを確保した）

| 履歴の位置 | セグメント MiB | 要求 MiB | 直前の空きの合計 MiB | 直前の最大の隙間 MiB | 呼び出し元 |
|---:|---:|---:|---:|---:|---|
| 307 | 20.0 | 1.6 | 3.6 | 1.0 | `activations.py:103:forward <- modeling_qwen3_vl.py:525:forward <- modeling_qwen3_vl.py:567:forward <- modeling_layers.py:110:__call__` |
| 16341 | 20.0 | 3.4 | 17.8 | 1.5 | `rope.py:167:rope_tables <- pipeline_h3.py:723:_denoise <- pipeline_h3.py:477:generate <- worker.py:332:_do_generate` |
| 16438 | 96.0 | 95.6 | 215.0 | 92.0 | `dit.py:320:forward <- block_swap_service.py:359:swapped_forward_prefetch <- dit.py:567:forward <- pipeline_h3.py:757:_denoise` |
| 16443 | 192.0 | 191.3 | 215.3 | 92.0 | `dit.py:284:_modulate <- dit.py:320:forward <- block_swap_service.py:359:swapped_forward_prefetch <- dit.py:567:forward` |
| 16445 | 192.0 | 191.3 | 216.1 | 92.0 | `dit.py:287:_modulate <- dit.py:320:forward <- block_swap_service.py:359:swapped_forward_prefetch <- dit.py:567:forward` |
| 16456 | 384.0 | 382.5 | 551.5 | 192.0 | `__init__.py:2190:int8_linear <- __init__.py:1054:int8_linear <- quant_linear.py:112:forward <- dit.py:228:forward` |
| 16463 | 128.0 | 127.5 | 473.3 | 96.4 | `dit.py:230:forward <- dit.py:321:forward <- block_swap_service.py:359:swapped_forward_prefetch <- dit.py:567:forward` |
| 16471 | 128.0 | 127.5 | 473.8 | 96.4 | `dit.py:231:forward <- dit.py:321:forward <- block_swap_service.py:359:swapped_forward_prefetch <- dit.py:567:forward` |
| 16493 | 128.0 | 127.5 | 502.8 | 96.4 | `rope.py:184:apply_rope <- dit.py:233:forward <- dit.py:321:forward <- block_swap_service.py:359:swapped_forward_prefetch` |
| 16566 | 512.0 | 510.0 | 1223.9 | 384.0 | `__init__.py:2190:int8_linear <- __init__.py:1054:int8_linear <- quant_linear.py:112:forward <- dit.py:248:forward` |
| 16573 | 256.0 | 255.0 | 1018.7 | 192.0 | `dit.py:249:forward <- dit.py:324:forward <- block_swap_service.py:359:swapped_forward_prefetch <- dit.py:567:forward` |

セグメントごと（その瞬間。大きい順に先頭 40 個）:

| 番地 | プール | ストリーム | 予約 MiB | 使用中 MiB | 未使用 MiB | 最大の隙間 MiB | ブロック数 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0x14bc000000 | large | 0 | 512.0 | 510.0 | 2.0 | 2.0 | 1 |
| 0x148c000000 | large | 0 | 384.0 | 255.0 | 129.0 | 129.0 | 1 |
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
| 0x14dc000000 | large | 0 | 256.0 | 255.0 | 1.0 | 1.0 | 1 |
| 0x1326c00000 | large | 0 | 222.0 | 220.5 | 1.5 | 1.5 | 1 |
| 0x1372600000 | large | 0 | 222.0 | 220.5 | 1.5 | 1.5 | 1 |
| 0x1474000000 | large | 0 | 192.0 | 0.0 | 192.0 | 192.0 | 0 |
| 0x1480000000 | large | 0 | 192.0 | 95.6 | 96.4 | 96.4 | 1 |
| 0x145cc00000 | large | 0 | 184.0 | 95.6 | 88.4 | 88.4 | 1 |
| 0x131d800000 | large | 0 | 148.0 | 147.0 | 1.0 | 1.0 | 1 |
| 0x1347000000 | large | 0 | 148.0 | 147.0 | 1.0 | 1.0 | 1 |
| 0x14a4000000 | large | 0 | 128.0 | 95.6 | 32.4 | 32.4 | 1 |
| 0x14ac000000 | large | 0 | 128.0 | 95.6 | 32.4 | 32.4 | 1 |
| 0x14b4000000 | large | 0 | 128.0 | 0.0 | 128.0 | 128.0 | 0 |
| 0x146e000000 | large | 0 | 96.0 | 0.0 | 96.0 | 96.0 | 0 |
| 0x1468400000 | large | 0 | 92.0 | 0.0 | 92.0 | 92.0 | 0 |
| 0x1380400000 | large | 0 | 74.0 | 73.5 | 0.5 | 0.5 | 1 |
| 0x1384e00000 | large | 0 | 74.0 | 73.5 | 0.5 | 0.5 | 1 |
| 0x1304e00000 | large | 0 | 54.0 | 52.5 | 1.5 | 1.5 | 1 |
| 0x1303a00000 | large | 0 | 20.0 | 15.3 | 4.7 | 4.7 | 4 |
| 0x1308a00000 | large | 0 | 20.0 | 6.8 | 13.2 | 13.2 | 2 |
| 0x130a800000 | small | 0 | 2.0 | 1.6 | 0.4 | 0.3 | 7 |
| 0x1367800000 | small | 0 | 2.0 | 0.0 | 2.0 | 2.0 | 0 |
| 0x1367e00000 | small | 0 | 2.0 | 1.7 | 0.3 | 0.3 | 25 |

## (2) その瞬間の最大の確保 20 件

| MiB | ストリーム | 呼び出し元（内側から） |
|---:|---:|---|
| 510.0 | 0 | `__init__.py:2190:int8_linear <- __init__.py:1054:int8_linear <- quant_linear.py:112:forward <- dit.py:248:forward` |
| 371.1 | 0 | `block_swap_prefetch.py:405:prepare <- block_swap_service.py:321:_build_prefetch_engine <- block_swap_service.py:134:install <- pipeline_h3.py:661:_place_dit` |
| 371.1 | 0 | `block_swap_prefetch.py:405:prepare <- block_swap_service.py:321:_build_prefetch_engine <- block_swap_service.py:134:install <- pipeline_h3.py:661:_place_dit` |
| 371.1 | 0 | `block_swap_prefetch.py:405:prepare <- block_swap_service.py:321:_build_prefetch_engine <- block_swap_service.py:134:install <- pipeline_h3.py:661:_place_dit` |
| 371.1 | 0 | `block_swap_prefetch.py:405:prepare <- block_swap_service.py:321:_build_prefetch_engine <- block_swap_service.py:134:install <- pipeline_h3.py:661:_place_dit` |
| 371.1 | 0 | `block_swap_prefetch.py:405:prepare <- block_swap_service.py:321:_build_prefetch_engine <- block_swap_service.py:134:install <- pipeline_h3.py:661:_place_dit` |
| 371.1 | 0 | `block_swap_prefetch.py:405:prepare <- block_swap_service.py:321:_build_prefetch_engine <- block_swap_service.py:134:install <- pipeline_h3.py:661:_place_dit` |
| 371.1 | 0 | `block_swap_prefetch.py:405:prepare <- block_swap_service.py:321:_build_prefetch_engine <- block_swap_service.py:134:install <- pipeline_h3.py:661:_place_dit` |
| 371.1 | 0 | `block_swap_prefetch.py:405:prepare <- block_swap_service.py:321:_build_prefetch_engine <- block_swap_service.py:134:install <- pipeline_h3.py:661:_place_dit` |
| 371.1 | 0 | `block_swap_prefetch.py:405:prepare <- block_swap_service.py:321:_build_prefetch_engine <- block_swap_service.py:134:install <- pipeline_h3.py:661:_place_dit` |
| 371.1 | 0 | `block_swap_prefetch.py:405:prepare <- block_swap_service.py:321:_build_prefetch_engine <- block_swap_service.py:134:install <- pipeline_h3.py:661:_place_dit` |
| 294.0 | 0 | `sft_load.py:141:move_module_tree <- pipeline_h3.py:655:_place_dit <- pipeline_h3.py:711:_denoise <- pipeline_h3.py:477:generate` |
| 294.0 | 0 | `sft_load.py:141:move_module_tree <- pipeline_h3.py:655:_place_dit <- pipeline_h3.py:711:_denoise <- pipeline_h3.py:477:generate` |
| 255.0 | 0 | `dit.py:249:forward <- dit.py:324:forward <- block_swap_service.py:359:swapped_forward_prefetch <- dit.py:567:forward` |
| 255.0 | 0 | `dit.py:249:forward <- dit.py:324:forward <- block_swap_service.py:359:swapped_forward_prefetch <- dit.py:567:forward` |
| 220.5 | 0 | `sft_load.py:141:move_module_tree <- pipeline_h3.py:655:_place_dit <- pipeline_h3.py:711:_denoise <- pipeline_h3.py:477:generate` |
| 220.5 | 0 | `sft_load.py:141:move_module_tree <- pipeline_h3.py:655:_place_dit <- pipeline_h3.py:711:_denoise <- pipeline_h3.py:477:generate` |
| 147.0 | 0 | `sft_load.py:141:move_module_tree <- pipeline_h3.py:655:_place_dit <- pipeline_h3.py:711:_denoise <- pipeline_h3.py:477:generate` |
| 147.0 | 0 | `sft_load.py:141:move_module_tree <- pipeline_h3.py:655:_place_dit <- pipeline_h3.py:711:_denoise <- pipeline_h3.py:477:generate` |
| 95.6 | 0 | `dit.py:495:embed_rows <- pipeline_h3.py:756:_denoise <- pipeline_h3.py:477:generate <- worker.py:332:_do_generate` |

## (3) 書き出し時点のセグメントの一覧

セグメント 35 個・予約 7588.0 MiB・割り当て 5270.5 MiB

| 番地 | 種別 | ストリーム | 予約 MiB | 使用中 MiB | active MiB | 最大の空きブロック MiB | ブロック数 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0x14bc000000 | large | 0 | 512.0 | 0.0 | 0.0 | 512.0 | 1 |
| 0x148c000000 | large | 0 | 384.0 | 0.0 | 0.0 | 384.0 | 1 |
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
| 0x14dc000000 | large | 0 | 256.0 | 0.0 | 0.0 | 256.0 | 1 |
| 0x1326c00000 | large | 0 | 222.0 | 220.5 | 220.5 | 1.5 | 2 |
| 0x1372600000 | large | 0 | 222.0 | 220.5 | 220.5 | 1.5 | 2 |
| 0x1474000000 | large | 0 | 192.0 | 0.0 | 0.0 | 192.0 | 1 |
| 0x1480000000 | large | 0 | 192.0 | 0.0 | 0.0 | 192.0 | 1 |
| 0x145cc00000 | large | 0 | 184.0 | 0.0 | 0.0 | 184.0 | 1 |
| 0x131d800000 | large | 0 | 148.0 | 148.0 | 148.0 | 0.0 | 1 |
| 0x1347000000 | large | 0 | 148.0 | 148.0 | 148.0 | 0.0 | 1 |
| 0x14a4000000 | large | 0 | 128.0 | 0.0 | 0.0 | 128.0 | 1 |
| 0x14ac000000 | large | 0 | 128.0 | 0.0 | 0.0 | 128.0 | 1 |
| 0x14b4000000 | large | 0 | 128.0 | 0.0 | 0.0 | 128.0 | 1 |
| 0x146e000000 | large | 0 | 96.0 | 0.0 | 0.0 | 96.0 | 1 |
| 0x1468400000 | large | 0 | 92.0 | 0.0 | 0.0 | 92.0 | 1 |
| 0x1380400000 | large | 0 | 74.0 | 74.0 | 74.0 | 0.0 | 1 |
| 0x1384e00000 | large | 0 | 74.0 | 74.0 | 74.0 | 0.0 | 1 |
| 0x1304e00000 | large | 0 | 54.0 | 52.5 | 52.5 | 1.5 | 2 |
| 0x1303a00000 | large | 0 | 20.0 | 12.1 | 12.1 | 7.9 | 4 |
| 0x1308a00000 | large | 0 | 20.0 | 10.1 | 10.1 | 6.5 | 5 |
| 0x130a800000 | small | 0 | 2.0 | 1.2 | 1.2 | 0.7 | 8 |
| 0x1367800000 | small | 0 | 2.0 | 0.0 | 0.0 | 2.0 | 1 |
| 0x1367e00000 | small | 0 | 2.0 | 1.7 | 1.7 | 0.2 | 27 |
