# MiniMax H3 の Retake の GPU の門（記録）

- 開始: 2026-11-08T14:22:09+09:00／終了: 2026-11-08T14:38:09+09:00
- 素材: `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\uploads\videos\5e0cd652-ad6d-4a68-948a-5faea7480b01\input.mp4`
- 窓: 開始 1.0 秒（24 フレーム目）・124 フレーム・1280×736・種 1234・注意の計算 sdpa
- transformer: `default`（transformer_role=fl2va）
- 判定は付けない（PSNR は記録のみ）。数値の正本は記録簿 §200。

| 条件 | strength | 状態 | job_id | 秒 | 出力フレーム | muxed_original_waveform | decoded_frames_px | n_tail_a | 音声の有無（元） | 頭 PSNR 平均／最小 | 尾 PSNR 平均／最小 | 参照 | エラー |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| silent-s0 | 0.0 | completed | `9d150674-4639-42c2-802b-fbb36cee4beb` | 955.00 | 124 | False | 124 | 0 | False | 36.85／36.07 | 36.61／36.08 | job_retake_window | – |

欄の意味は台本の README.md「JSON の欄」。
