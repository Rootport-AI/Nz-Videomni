> **このファイルは実機の `outputs/comfort-calib-h3-ref-2026-11-03/mem_history/analysis/dryrun_selftest.md`（git追跡外）のスナップショットである（2026-11-03複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# snapshot_step1.pickle の集計

## 要約

```json
{
 "snapshot": "runs/dryrun/snapshot_step1.pickle",
 "segments": 2,
 "reserved_bytes": 88080384,
 "allocated_bytes": 70255104,
 "reserved_free_bytes": 17825280,
 "reserved_gib": 0.082,
 "allocated_gib": 0.0654,
 "reserved_free_gib": 0.0166,
 "free_in_hypothesis_C_origins_gib": 0.0,
 "free_in_hypothesis_C_origins_share": 0.0,
 "free_in_block_9lines_gib": 0.0,
 "free_in_block_9lines_share": 0.0,
 "free_in_block_all_lines_gib": 0.0,
 "free_in_block_all_lines_share": 0.0,
 "trace_len": 4,
 "trace_actions": {
  "segment_alloc": 2,
  "alloc": 2
 },
 "allocator_settings": {
  "PYTORCH_CUDA_ALLOC_CONF": "",
  "max_split_size": -1,
  "garbage_collection_threshold": 0.0,
  "expandable_segments": false,
  "pinned_num_register_threads": 1,
  "release_lock_on_cudamalloc": false,
  "pinned_use_cuda_host_register": false,
  "graph_capture_record_stream_reuse": false,
  "max_round_threshold": -1,
  "max_cached_size": -1,
  "roundup_power2_divisions": {
   "1": 0,
   "2": 0,
   "4": 0,
   "8": 0,
   "16": 0,
   "32": 0,
   "64": 0,
   "128": 0,
   "256": 0,
   "512": 0,
   "1024": 0,
   "2048": 0,
   "4096": 0,
   "8192": 0,
   "16384": 0,
   "32768": 0
  }
 },
 "segment_keys_seen": [
  "active_size",
  "address",
  "allocated_size",
  "blocks",
  "device",
  "frames",
  "is_expandable",
  "requested_size",
  "segment_pool_id",
  "segment_type",
  "stream",
  "total_size"
 ],
 "block_keys_seen": [
  "address",
  "frames",
  "requested_size",
  "size",
  "state"
 ],
 "trace_keys_seen": [
  "action",
  "addr",
  "compile_context",
  "frames",
  "pool_id",
  "size",
  "stream",
  "time_us",
  "user_metadata"
 ],
 "segments_with_nonempty_frames": 1
}
```

## 起点の分類ごとの合計（予約済みの空きの多い順）

| 起点の分類 | セグメント数 | 大きさ GiB | 使用中 GiB | 空き GiB | 最大の連続した空き GiB |
|---|---|---|---|---|---|
| その他（engine_h3 の外） | 2 | 0.082 | 0.0654 | 0.0166 | 0.0166 |

## 起点のフレームごとの合計

| 起点のフレーム | セグメント数 | 大きさ GiB | 使用中 GiB | 空き GiB | 最大の連続した空き GiB |
|---|---|---|---|---|---|
| outputs/comfort-calib-h3-ref-2026-11-03/mem_history/measure_h3_vram.py:435 dry_run | 1 | 0.0195 | 0.0029 | 0.0166 | 0.0166 |
| outputs/comfort-calib-h3-ref-2026-11-03/mem_history/measure_h3_vram.py:433 dry_run | 1 | 0.0625 | 0.0625 | 0.0 | 0.0 |

## セグメントの表（アドレス順）

| アドレス | 大きさ MiB | 使用中 MiB | active MiB | 空き MiB | 最大の連続した空き MiB | ブロック数 | 種類 | 起点のフレーム | 起点のいちばん内側 | 起点の出どころ | segment_alloc の履歴番号 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0x1303000000 | 20.0 | 3.0 | 3.0 | 17.0 | 17.0 | 2 | large | outputs/comfort-calib-h3-ref-2026-11-03/mem_history/measure_h3_vram.py:435 dry_run | outputs/comfort-calib-h3-ref-2026-11-03/mem_history/measure_h3_vram.py:435 dry_run | trace.segment_alloc | 2 |
| 0x130b200000 | 64.0 | 64.0 | 64.0 | 0.0 | 0.0 | 1 | large | outputs/comfort-calib-h3-ref-2026-11-03/mem_history/measure_h3_vram.py:433 dry_run | outputs/comfort-calib-h3-ref-2026-11-03/mem_history/measure_h3_vram.py:433 dry_run | trace.segment_alloc | 0 |

## 大きいセグメント（256MiB 以上）の中身（アドレス順のブロック）

## 時系列（セグメントの出入りと 512MiB 以上の確保・解放。Nv f32 は Nv×5376×4 バイトの確保）

履歴の件数 4・OOM の位置 None・Nv×5376×4＝21504 バイト

| 履歴の番号 | 動作 | アドレス | 大きさ MiB | Nv f32 | 起点 | 履歴の頭からの予約の増減 MiB | device_free |
|---|---|---|---|---|---|---|---|
| 0 | segment_alloc | 0x130b200000 | 64.0 | False | outputs/comfort-calib-h3-ref-2026-11-03/mem_history/measure_h3_vram.py:433 dry_run | 64.0 | None |
| 2 | segment_alloc | 0x1303000000 | 20.0 | False | outputs/comfort-calib-h3-ref-2026-11-03/mem_history/measure_h3_vram.py:435 dry_run | 84.0 | None |
