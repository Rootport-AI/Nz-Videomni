# MiniMax H3 の Retake の GPU の門（記録）

- 開始: 2026-11-08T13:33:21+09:00／終了: 2026-11-08T14:21:22+09:00
- 素材: `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\uploads\h3-ref-calibration\H3-Ref-1080p-15s.mp4`
- 窓: 開始 4.0 秒（96 フレーム目）・124 フレーム・1280×736・種 1234・注意の計算 sdpa
- transformer: `default`（transformer_role=fl2va）
- 判定は付けない（PSNR は記録のみ）。数値の正本は記録簿 §200。

| 条件 | strength | 状態 | job_id | 秒 | 出力フレーム | muxed_original_waveform | decoded_frames_px | n_tail_a | 音声の有無（元） | 頭 PSNR 平均／最小 | 尾 PSNR 平均／最小 | 参照 | エラー |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | 0.0 | completed | `503cb4d0-78f6-4677-a31f-47912515356c` | 955.40 | 124 | False | 124 | 36 | True | 32.79／32.37 | 33.14／32.53 | job_retake_window | – |
| s0.5 | 0.5 | completed | `be5a3fea-9382-4c72-8191-0c9f4621548d` | 955.40 | 124 | False | 124 | 36 | True | 32.79／32.37 | 33.14／32.53 | job_retake_window | – |
| s1.0 | 1.0 | completed | `e517254f-529a-4161-8ac9-b9e90f824ccc` | 955.00 | 124 | True | 124 | 36 | True | 32.80／32.37 | 33.15／32.53 | job_retake_window | – |

欄の意味は台本の README.md「JSON の欄」。
