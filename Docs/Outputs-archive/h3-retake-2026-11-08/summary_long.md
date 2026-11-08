# MiniMax H3 の Retake の GPU の門（記録）

- 開始: 2026-11-08T14:38:10+09:00／終了: 2026-11-08T14:55:07+09:00
- 素材: `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\h3-retake-gates-2026-11-08\_material\H3-Ref-1080p-30s.mp4`
- 窓: 開始 2.0 秒（48 フレーム目）・481 フレーム・640×384・種 1234・注意の計算 sdpa
- transformer: `default`（transformer_role=fl2va）
- 判定は付けない（PSNR は記録のみ）。数値の正本は記録簿 §200。

| 条件 | strength | 状態 | job_id | 秒 | 出力フレーム | muxed_original_waveform | decoded_frames_px | n_tail_a | 音声の有無（元） | 頭 PSNR 平均／最小 | 尾 PSNR 平均／最小 | 参照 | エラー |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long481-s0 | 0.0 | completed | `7c24d979-c20d-4f10-bca1-cf4aba40dbd5` | 1005.10 | 481 | False | 481 | 35 | True | 27.35／27.09 | 27.51／27.06 | job_retake_window | – |

欄の意味は台本の README.md「JSON の欄」。
