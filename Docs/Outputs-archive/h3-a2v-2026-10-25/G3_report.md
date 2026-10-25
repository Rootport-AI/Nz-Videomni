# §1-103 H3 の A2V — GPU の門 G3-0・G3 の報告（2026-10-25）

## 結論

- **G3-0 は合格。** main（`7f1f27d`）と dev（`6285a90`）で同じ I2V（音声なし）を流した。mp4 の md5 が一致し、復号した映像（rgb24）と音声（s16le）もバイト単位で一致した（最大の絶対差は 0）。
- **G3 は 2 本とも完走し、合格。** 896×1152 のまま通った（OOM・再試行 0。小さい画角へは落としていない）。
  - Run A（strength 1.0）: `audio_output` は "original"。mp4 の音声は 120,000 サンプル。入力との相関は r = 0.99993、遅延は 0。
  - Run B（strength 0.5）: `audio_output` は "decoded"。`m_per_step` は全ステップで 3。
- **GPU を使ったのは了承された 4 本だけ。** 同時に動かしたのは常に 1 本。製品のコードと試験には触っておらず、コミットもしていない。最後の状態はブランチ dev・HEAD `6285a90`・`git status` は空。バックエンドは始めと同じく停止している。

## 事前の確認

- `nvidia-smi`: 使用中のメモリは 1,091MiB／16,376MiB。並んでいたのはデスクトップのアプリ（C+G）だけ。`--query-compute-apps` に Python は無かった。権限不足で名前が出なかった 3 つは dwm・TabTip・Taskmgr。
  - GPU を使っていない Python が 2 つ動いていた（PID 54300 はシステムの Python310、PID 55160 は `S:\OriginalApps\hf-cli\hf-env`）。私が起動したものではないので触っていない。
- git: ブランチ dev・HEAD `6285a90`・作業ツリーはきれい。main は `7f1f27d`（main..dev はこの 1 コミットだけ）。
- 重み: `models/MiniMaxH3/` に Weights・TextEncoder・VAE（音声・映像）・LoRA がそろっていた。`state.json` の `active_base_model` は MiniMaxH3、選択はすべて "default"。
- バックエンド: 始めは停止していた（18620 番ポートに応答なし）。

## 手順（§176.6 門 3 の作法に合わせた）

- 起動は run.ps1 と同じ環境。`tools/ffmpeg/bin` と `tools/uv` を PATH の先頭に足し、`.venv\Scripts\python.exe main.py` を既定のポート 18620 で起動した。
- 続いて `POST /api/v1/pipeline/load {"base_model":"MiniMaxH3"}` で読み込んだ。
- 素材は `POST /upload/image` と `POST /upload/audio` で上げた。ジョブは `POST /generate` で出し、`GET /jobs/{id}` で終わるまで見た。
- 終わったら、自分で起動したプロセスの木を `taskkill /T /F` で止めた。
- 台本は `outputs/h3_a2v_g3/drive.py`、解析は `outputs/h3_a2v_g3/analyze.py`。
- 各回の数値の出どころは `metadata.json`。
  - VRAM の山は `vram_optimization.peak_vram_mb` と `peak_vram_reserved_mb`、段ごとの値は `h3.phases.*`。トップレベルに `peak_vram_mb` というキーは無い。
  - GPU のメモリは 3 秒おきに標本を取った。専用メモリは `nvidia-smi` の memory.used、共有メモリは `typeperf "\GPU Adapter Memory(*)\Shared Usage"` の最大値。
- 注意: G3-0 の dev の回は、台本の出力パスの扱いの誤り（相対パスのまま開いた）で、ジョブの完了後にコピーで落ちた。
  - ジョブそのものは完走している（job `1de91766-…`）。mp4 と metadata は手で `g3-0_dev/` に写した。
  - この回だけ `row.json` と GPU の標本が無い。実時間は job.json の時刻から約 47 秒（`generation_time_seconds` 47.03）。
  - 誤りは直してから、残りの 3 本を回した。

## G3-0 — main と dev の回帰のビット一致

条件は 2 本とも同じ。

- I2V。キーフレームは `00043-2168133585.jpg` を frame 0 に、強度は指定しない。
- 384×480・39 フレーム・24 fps・4 ステップ・seed 12345・shift 12.0／3.0（既定）。
- sdpa（既定）・VRAM 16GB mode on（既定）・常駐 off・LoRA なし・`embed_mp4_metadata: false`。
- プロンプトは "a woman speaking to the camera, soft light"。

| 項目 | dev（6285a90） | main（7f1f27d） |
|---|---|---|
| job_id | 1de91766-b156-4e64-9cef-636f3b5b456d | c96d21b2-e05d-4355-81fd-df449496f6e7 |
| mp4 の md5 | 7b723738f8d61062e47b83a230462ace | 7b723738f8d61062e47b83a230462ace |
| 映像（rgb24） | 39 フレーム・21,565,440 バイト | 同じ |
| 音声（s16le） | 104,448 サンプル（2 チャンネル合計）・208,896 バイト | 同じ |
| バイト一致・最大の絶対差 | 映像 true・0／音声 true・0 | — |
| `m_per_step` | [2, 3, 3, 3] | [2, 3, 3, 3] |
| `peak_vram_mb`／予約 | 4,980／5,282 MB | 4,980／5,282 MB |
| ノイズ除去の段の山（割り当て／予約） | 3,804／3,826 MB | 3,804／3,826 MB |
| `generation_time_seconds` | 47.03 | 45.09 |
| ノイズ除去の段の秒 | 17.76 | 17.8 |
| 実時間 | 約 47 秒 | 48.1 秒 |
| GPU のメモリの山（nvidia-smi）／共有メモリの山 | 標本なし | 6,143 MiB／1,207.8 MiB |

**判定: 合格。** mp4 の md5 が一致し、復号した映像と音声もバイト単位で一致した。マスクを使わない経路は、変更の前後で同じ結果になる。結果は `outputs/h3_a2v_g3/g30_cmp.json`。

## G3 — オーナーの素材で A2V（dev）

条件は 2 本とも同じ。

- 素材は API で上げた。image_id は `dd7f9f97-ee94-4cff-88fe-4a663b4d0675`、audio_id は `81b7942a-d480-49c9-abb8-001e478789a7`。
- 896×1152・90 フレーム・24 fps（3.75 秒）・20 ステップ・seed 12345・shift 12.0／3.0。
- sdpa（既定）・VRAM 16GB mode on・常駐 off（既定）・LoRA なし。
- キーフレームは `conditioning_images: [{image_id, frame_idx: 0}]`（強度は指定しない）。
- プロンプトは "An anime-style girl speaks cheerfully to the camera, natural lip movement, gentle head motion, soft indoor light"。

| 項目 | Run A | Run B |
|---|---|---|
| ブランチ | dev | dev |
| 画角・フレーム・ステップ・seed | 896×1152・90・20・12345 | 同じ |
| strength | 1.0 | 0.5 |
| job_id | e8cb73f4-26d4-4cc6-a27e-17d943d94365 | 651142e9-14ca-4fd4-b50b-5087ffca6b5f |
| 実時間 | 679.3 秒 | 670.8 秒 |
| `generation_time_seconds` | 677.38 | 668.32 |
| 段の秒（encode／image_encode／audio_encode／denoise／decode） | 17.54／1.39／0.36／628.7／29.09 | 15.72／1.25／0.29／621.3／29.55 |
| DiT の読み込み（denoise に含む） | 11.73 秒 | — |
| `peak_vram_mb`／予約 | 6,380／7,604 MB（decode の段） | 6,380／7,604 MB |
| ノイズ除去の段の山（割り当て／予約） | 3,939／6,118 MB | 3,939／6,118 MB |
| `num_ooms`・`num_alloc_retries` | 0・0 | 0・0 |
| `inactive_split_peak_mib` | 568 | 568 |
| GPU のメモリの山（nvidia-smi） | 8,802 MiB | 8,812 MiB |
| 共有メモリ（最小〜最大） | 680.3〜1,217.1 MiB | 1,175.7〜1,202.5 MiB |
| `m_per_step` | 20 ステップすべて 3 | 20 ステップすべて 3 |
| `denoise.mask` | audio: rows 300・min 0.0・max 0.0 | audio: rows 300・min 0.5・max 0.5 |
| `audio_output` | original | decoded |
| 音声のサンプル数（mp4 の duration_ts） | 120,000 | 120,000 |
| 入力の 32 kHz 版との r／遅延 | 0.9999290981895034／0 | 0.9790604614351652／0（参考値） |
| 出力 | `outputs/h3_a2v_g3/runA_s1.0/output.mp4` | `outputs/h3_a2v_g3/runB_s0.5/output.mp4` |

**共有メモリについて:** WDDM の共有メモリは、生成前の待機中に 680 MiB 前後、生成中も約 1.2GB の水準で止まった。G3-0 の小さい画角でも 1,207.8 MiB で、§177 の A-9 に記録された「約 1.2GB」と同じ水準。大きな溢れは出ていない。

**系列の長さ（Run A）:** text 1,038・cond 1,008・audio 300・video 27,216・合計 29,562。潜在は 27 フレーム・音声の潜在は 150・格子は [36, 28]。

### `metadata.a2v`（全キー）

Run A の値（Run B と違うキーは矢印で示す）:

- `mask_value`: 0.0 → Run B は 0.5
- `audio_latents`: 150
- `required_samples`: 120000
- `encoded_samples`: 124356
- `channels_used`: "mono->stereo"
- `resampler`: "pyav"
- `encoder_tensors`: 134
- `audio_output`: "original" → Run B は "decoded"
- `muxed_audio_samples`: 120000
- `source_audio_id`: "81b7942a-d480-49c9-abb8-001e478789a7"
- `strength`: 1.0 → Run B は 0.5
- `source`: {duration_sec 3.8861, sample_rate 44100, channels 1}

その他:

- `metadata.h3` に `a2v` のキーは無く、トップレベルの `metadata.a2v` に出た（計画 §2.6 のとおり）。
- `h3.phases.decode`:
  - Run A: `audio_vae_source: "skipped"`・`audio_samples: 120000`・`audio_samples_source: 124356`
  - Run B: `vae_source: "file"`・`audio_samples: 120000`・`audio_samples_decoded: 120000`
- `h3.phases.audio_encode`: `encoder_tensors` 134・`encoded_samples` 124356・`latent_shape` [2, 32, 156]・山 657／760 MB（割り当て／予約）。
- 計画 §2.3 の例では、手元のサンプル数を 124,352 と見積もっていた。実測は 124,356（PyAV の変換による 4 サンプルの差）で、潜在は 156 個、うち先頭の 150 個を使う点は計画どおり。

### 音声の検査（CPU・app の venv の numpy・同梱の ffmpeg）

- 出力の音声は AAC・32 kHz・2 チャンネル。mp4 のストリームの長さは `duration_ts` 120000（3.750000 秒）で、120000 = round(90·32000/24) と一致した。
- ffmpeg で長さを指定せずに PCM へ復号すると、1 チャンネルあたり 120,832 サンプルになる。
  - 理由: 末尾にある AAC の最後のフレームの埋め草（119 フレーム × 1024 − 先頭の 1024 = 120,832）まで出てくるため。
  - `-t 3.75` で切ると 480,000 バイト（120,000 サンプル × 2 チャンネル）になる。
  - 先頭の遅延（priming）は ffmpeg が除くので、遅延は 0 になった。
- 相関の取り方: 入力の wav を ffmpeg で 32 kHz・2 チャンネル（124,356 サンプル）にし、±2,400 サンプルの範囲で遅延を探して Pearson の相関の最大値を取った（左右の平均と、左チャンネル単独）。
  - Run A: 平均 r = 0.9999290981895034（遅延 0）／左 0.9999228688339342（遅延 0）。**合格（基準は r ≥ 0.99）。**
  - Run B（参考値）: 平均 r = 0.9790604614351652（遅延 0）／左 0.9789201697977215（遅延 0）。
- ffprobe: 4 本とも h264＋aac の 2 ストリームが読めた。
  - G3-0: 39 フレーム・1.625 秒。
  - G3: 90 フレーム・3.75 秒。
- 画質の判定はしていない（オーナーの G4）。

### ログ

- 各フォルダに、その回のぶんだけを切り出した `h3_worker.log.excerpt` と `server.log.excerpt` を置いた。
- warn・error・traceback・unknown・ignore・oom の語で調べたところ、該当は 4 本とも 0 件だった。

## 気づいたこと（判断はしていない）

- G3 のノイズ除去は 20 ステップで約 621〜629 秒、1 ステップあたり約 31 秒（DiT の読み込み約 12 秒を含む）。
  - §177.6 の O-1（896×1152×124 フレーム・**sage**）の約 20.5 秒／ステップより遅い。
  - 今回は API の既定の `attention_backend: "sdpa"` のまま回した（課題の「既定のまま」に従った）。この差は注意の計算の方式の違いによると見られる（推測。確かめていない）。

## 成果物

- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\h3_a2v_g3\`（git の管理外）
  - `g3-0_dev\`・`g3-0_main\`: output.mp4・metadata.json・job.json・request.json・ログの抜粋（row.json は main だけ）
  - `runA_s1.0\`・`runB_s0.5\`: 上と同じもの＋row.json＋`audio_check.json`
  - `g30_cmp.json`（G3-0 の比較）・`drive.py`・`analyze.py`
- 元のジョブのフォルダ: `outputs\1de91766-…`・`c96d21b2-…`・`e8cb73f4-…`・`651142e9-…`
- 素材を上げたもの: `uploads\` に画像 3 件（11942544-…・581a38f8-…・dd7f9f97-…）、`uploads\audios\81b7942a-…`。Uploads は一時置き場なので、そのまま残している。
- 起動ログ: `scratchpad\h3a2v\g3\backend_*.log`・`g30_*.out`・`g3.out`
