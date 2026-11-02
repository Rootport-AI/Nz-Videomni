# MiniMax H3 の Chained の V2V／i2v：実機 GPU の門 ①〜⑤（2026-11-02）

- 対象コード: commit `ced1578`（実機バックエンド・`http://127.0.0.1:18620`）。① の再実行と ⑤ は、`services/video_io.py` で ffmpeg の stderr を UTF-8 で読むよう直してから再起動したバックエンドで行った
- モデル: MiniMaxH3・`minimax_h3_fl2va_pruned_int8_convrot.safetensors`（FL2VA int8）
- 共通設定: 640×384・24 fps・20 ステップ（既定）・vram16_mode 既定（on）・`keep_resident: true`・`attention_backend: sage`・`block_swap_prefetch: true`・seed 1234
- 素材: `outputs/0b57ea7d-b494-43ea-a504-2a5637380581/output.mp4`（640×384・56 フレーム・2.333 秒・AAC 音声あり）
- プロンプト（全ジョブ共通）: 「コスプレイヤーの日本人女性が、こちらに向かって歩き続ける。彼女は片手を伸ばしたまま、こちらに微笑みかける。」
- 画像の保存先: `outputs/h3-v2v-gates-2026-11-02/`

## 結果の一覧

| 門 | 内容 | ジョブ ID | 判定 | 壁時計時間 | generation_time_seconds |
|---|---|---|---|---|---|
| ① | V2V・音声あり（56＋39） | `31c8a99a-a5d9-43ff-b65f-8d62d673f4e0` | 合格 | 125.2 秒（DiT の読み込み込み） | 123.73 秒 |
| ② | V2V・音声なし（56＋39） | `58b4aef8-7e7b-4060-b098-75804b73ef94` | 合格 | 110.2 秒 | 108.27 秒 |
| ③ | i2v（冒頭画像 2 枚・56＋39） | `63138ec0-9b8a-47ce-9792-f55030b4af32` | 合格 | 110.1 秒 | 109.21 秒 |
| ④ | V2V・1 クリップ（56） | `b12fb0e0-196e-4693-9911-b3a110da3d0e` | 合格 | 65.2 秒 | 60.92 秒 |
| ①（再実行） | 直した後に ① を同じ条件で再実行 | `0627e76e-3ea4-4877-a71a-cadc69ed78ac` | 合格 | 115.2 秒 | 113.75 秒 |
| ⑤ | ①（再実行）のジョブで結合（150／300／500 ms） | `0627e76e-3ea4-4877-a71a-cadc69ed78ac` | 合格（3 回とも handle_crossfade。直す前の 1 回目は HTTP 500、下の参考を参照） | — | — |

nvidia-smi の使用量: 開始前 416 MiB → ① の後 2162 MiB（DiT を常駐）→ ④ の後 2174 MiB → 終了時 2177 MiB。

## ① V2V・音声あり

| 項目 | 期待値 | 実測 |
|---|---|---|
| `h3.clips[0].head_equals_prev_tail` | true | true |
| `h3.clips[1].head_equals_prev_tail` | true | true |
| `v2v.freeze_ka` | 36 か 37 | 37 |
| `v2v.source_had_audio` | true | true |
| `v2v.audio_head_frozen` | true | true |
| `v2v.trimmed_px` | 22 | 22 |
| `v2v.trimmed_audio_samples` | 29333 | 29333 |
| `v2v.audio_fade_in_samples` | 960 | 960 |
| `v2v.decoded_frames_px`／`chain.total_frames` | 73 | 73／73 |
| `v2v.new_frames_px` | 51 | 51 |
| `v2v.audio_handle_filename` | — | `output_audio_handle.wav` |
| `v2v.handle_context_seconds` | 0.916667 | 0.916667 |
| `output.duration_seconds` | 2.125 | 2.125 |
| ffprobe の output.mp4 | 51 フレーム | 映像 51 フレーム・2.125 秒／音声 2.125 秒（32 kHz） |
| handle wav の長さ − mp4 の長さ | 約 0.9167 秒 | 3.041656 − 2.125 = 0.916656 秒 |
| `h3.clips[0].seconds.source_encode` | ある | 5.12 秒 |

- VRAM: `peak_vram_mb` 5061（デコード）・`peak_vram_reserved_mb` 5328。入力動画のエンコード（`h3.phases.clips[0].source_encode`）は peak 4159／reserved 4992 MB・22 フレームを読み込み・`encoded_samples` 29600。デノイズは 3775／3872 MB・vram16 のメモリ不足（OOM）は 0 回。
- 時間の内訳: clip 0 はテキストエンコード 17.43 秒・入力動画のエンコード 5.12 秒・デノイズ 42.7 秒（DiT をファイルから読み込み、うち 13.33 秒）。clip 1 はテキストエンコード 19.99 秒・デノイズ 26.31 秒（DiT は常駐分を使用）。デコード 8.65 秒。
- 目視用の画像: `gate1_source_last.png`（入力動画の最後のフレーム）と `gate1_output_first.png`（出力の最初のフレーム）。両者の PSNR は 27.06 dB で、動きが続いている。

## ② V2V・音声なし（入力動画から `-an` で音声を除去）

| 項目 | 期待値 | 実測 |
|---|---|---|
| `v2v.freeze_ka` | 0 | 0 |
| `v2v.source_had_audio` | false | false |
| `v2v.audio_head_frozen` | false | false |
| mp4 の音声 | あり（生成した音声） | あり・2.125 秒・32 kHz |
| handle wav | 書き出される | `output_audio_handle.wav`・3.041656 秒 |
| head_equals_prev_tail（clip 0／clip 1） | — | true／true |
| new_frames_px／trimmed_px／total | — | 51／22／73 |

- VRAM: 5064／5378 MB。入力動画のエンコードは 4.83 秒・`encoded_samples` 0。

## ③ i2v（画像 0 枚目を frame_idx 0、画像 55 枚目を frame_idx 28 に）

| 項目 | 期待値 | 実測 |
|---|---|---|
| 投入 | 202 で受け付け→完了 | 202 → completed |
| `h3.images_used` | 2 | 2 |
| `h3.clips[0].seconds.image_encode` | ある | 0.78 秒（VRAM peak 1030 MB） |
| `chain.total_frames`／出力のフレーム数 | 73／73（切り詰めなし） | 73／ffprobe で 73 フレーム・3.0417 秒 |
| 出力の 0 フレーム目と入力画像の比較 | ほぼ一致 | PSNR 34.93 dB |
| 出力の 28 フレーム目と入力画像の比較 | — | PSNR 30.24 dB |
| metadata の v2v ブロック／handle wav | なし | どちらもない（出力は `metadata.json`・`output.mp4` だけ） |

- 条件付けの行数: clip 0 の `denoise.sequence.cond` は 480（画像 2 枚 × 240）、clip 1 は 0。`head_equals_prev_tail` は clip 0 が null、clip 1 が true。
- 目視用の画像: `gate3_input_frame0.png` と `gate3_output_first.png`、`gate3_input_frame28_lastsrc.png` と `gate3_output_f28.png`。

## ④ V2V・1 クリップ（56）

| 項目 | 期待値 | 実測 |
|---|---|---|
| 状態 | completed | completed |
| `v2v.new_frames_px` | 34 | 34 |
| v2v ブロック | ある | ある（freeze_ka 37・source_had_audio true・audio_head_frozen true・decoded 56・trimmed 22） |
| ffprobe の mp4 | — | 映像 34 フレーム・1.416667 秒／**音声 1.409 秒** |
| handle wav の長さ − mp4 の長さ | 約 0.9167 秒 | 2.325 − 1.416667 = **0.908333 秒** |

- `decode.audio_samples_decoded` は 74400（56 フレーム分なら 74667 サンプルのはず。267 サンプル＝約 8.3 ms 足りない）、`decode.audio_samples` は 45067（34 フレーム分なら 45333）。2 クリップの ①〜③ では `audio_samples_decoded` が 97600 で、73 フレーム分の 97333 より多いため、この不足は出ていない。

## ① の再実行と ⑤ JOIN（stderr を UTF-8 で読むよう直した後・バックエンドを再起動）

- `services/video_io.py` で ffmpeg の stderr を UTF-8 で読むよう直してバックエンドを再起動した。ジョブはメモリにしか残らないため、① の条件を変えずに作り直し、そのジョブで結合した。
- ① の再実行: ジョブ `0627e76e-3ea4-4877-a71a-cadc69ed78ac`・合格・壁時計時間 115.2 秒（DiT の読み込み込み）・generation_time_seconds 113.75 秒。
  - head_equals_prev_tail は true／true。freeze_ka 37、new_frames_px 51、trimmed_px 22、trimmed_audio_samples 29333、decoded 73。
  - 出力は 51 フレーム・2.125 秒。handle wav は 3.041656 秒で、mp4 との差は 0.916656 秒。
  - VRAM は peak 5061／reserved 5328 MB。clip 0 と clip 1 の潜在の sha256 は 1 回目（`31c8a99a…`）と完全に一致した。
  - nvidia-smi の使用量: 開始前 426 MiB → ① の後 2165 MiB → ⑤ の後 2180 MiB。
- ⑤ の結合: `{"audio_smoothing": true, "handle_crossfade_ms": N, "source_tail_seconds": 5.0}` を 150 → 300 → 500 の順に、前の応答を待ってから送った。3 回とも HTTP 200。

| handle_crossfade_ms | join_mode | handle_crossfade_ms_applied | handle_context_seconds | trimmed_source_seconds | source_normalized | loudness_matched（source／continuation LUFS） | fade_ms_applied | 結合ファイル（ffprobe） |
|---|---|---|---|---|---|---|---|---|
| 150 | handle_crossfade | 150 | 0.916656 | 0.0 | false | true（−15.15／−13.45） | 0 | 映像 h264 107 フレーム・4.458333 秒／音声 aac 48 kHz・4.461 秒 |
| 300 | handle_crossfade | 300 | 0.916656 | 0.0 | false | true（同上） | 0 | 同上 |
| 500 | handle_crossfade | 500 | 0.916656 | 0.0 | false | true（同上） | 0 | 同上 |

- 3 回とも handle の経路を通った（`join_mode` は handle_crossfade、`fade_ms_applied` は 0）。2 本の音声をフェードでつなぐ方式に落ちた回はない。
- 107 フレームは、入力動画 56 フレームと新しく作った 51 フレームの合計。
- 3 本のファイルは中身がすべて異なる（md5 が別）。保存先は `outputs/h3-v2v-gates-2026-11-02/gate5_joined_150ms.mp4`・`gate5_joined_300ms.mp4`・`gate5_joined_500ms.mp4`。聴いて確かめるのはオーナー。

### 参考：直す前の 1 回目の失敗

- ① の 1 回目のジョブ `31c8a99a…` で結合したところ、3 回とも HTTP 500 になった。
- 原因は `_loudnorm_measure` が `subprocess.run(..., text=True)` を使い、stderr を cp932 で読もうとしていたこと。ffmpeg は入力動画の mp4 に埋め込まれた日本語のプロンプトを stderr に出すため、そこで cp932 の解読に失敗した。その結果 stderr が None になり、`AttributeError: 'NoneType' object has no attribute 'rfind'` で落ちた。
- この関数は ced1578 で入ったものではなく、それ以前からある。stderr を UTF-8 で読むよう直して解消した（上の再実行）。
