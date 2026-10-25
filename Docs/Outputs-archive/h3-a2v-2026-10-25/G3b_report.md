# §1-103 H3 の A2V — GPU の門 G3b（既定の強度の比べ合わせ）の報告（2026-10-25）

## 結論

- **4 本ともそろった。** 3 本を新しく回し、3 本とも完走した（OOM・再試行 0、896×1152 のまま）。残りの 1 本（kf_s0.50）は G3 の Run B を使い回した。
- **4 本とも `audio_output` は "decoded"。** 入力の音声との相関は r = 0.9788〜0.9797、遅延はすべて 0。
- **キーフレームなしの 2 本は約 1 割速い。** キーフレームありは実時間 670.8〜685.3 秒、なしは 607.5〜616.2 秒。キーフレームなしでは、条件画像の流れ（cond）が無くなり、`m_per_step` が 3 から 2 に減った。
- **画質は判定していない。** 画質はオーナーが見て判断する。口の周りの動きの指標は、`analyze.py` に似た機能が無いので取っていない（指示どおり、新しい解析の道具は作っていない）。
- **GPU で回したのは了承された 3 本だけ。** 同時に動かしたのは常に 1 本。製品のコードと試験には触っておらず、コミットもしていない。最後の状態は、バックエンドが停止（18620 番ポートは応答なし）・ブランチ dev・HEAD `6285a90`・`git status` は空。
- **オーナーの `hf.exe download` には触っていない。** 止めたのは、自分で起動したバックエンドのプロセスの木だけ。

## 事前の確認

- git: ブランチ dev・HEAD `6285a90`・作業ツリーはきれい。
- `nvidia-smi`: 使用中は 977 MiB／16,376 MiB。計算のプロセスに Python は無かった。
- バックエンド: 始めは停止していた。

## kf_s0.50 で Run B を使い回した理由

`outputs/h3_a2v_g3/runB_s0.5/metadata.json` に保存されたリクエストを確かめ、指定の条件とすべて一致した。

- seed 12345・896×1152・90 フレーム・24 fps・20 ステップ。
- shift 12.0／3.0（既定）・`attention_backend` は "sdpa"・`vram16_mode` は true・LoRA なし。
- プロンプトは同じ文。
- `conditioning_images` は frame 0（強度は null）、`source_audio.strength` は 0.5。
- ブランチと HEAD も同じ（dev・`6285a90`）。

そのため回し直していない。フォルダはそのまま `outputs/h3_a2v_g3b/kf_s0.50/` に写し、出どころを書いた `REUSED_FROM_G3_runB.txt` を添えた（job `651142e9-14ca-4fd4-b50b-5087ffca6b5f`）。

## 手順

G3 と同じ作法で回した。

1. `main.py` をポート 18620 で起動し、`POST /pipeline/load MiniMaxH3` で読み込んだ。
2. 素材を上げ直した。image_id は `14da35c1-14e1-4ebc-99ea-5c3e93bb3e97`、audio_id は `9869ff5b-b3e2-456a-a806-e0212921cd3e`。
3. 3 本を順に `POST /generate` で出し、終わるまで見て、出力を写した。
4. 最後にバックエンドを止めた。

- 台本は `outputs/h3_a2v_g3b/drive_g3b.py`。G3 の `drive.py` の関数をそのまま読み込んで使った。
- 音声の検査は G3 の `analyze.py audio` をそのまま使った。
- キーフレームなしの 2 本は、`conditioning_images` を省いた。`generation_mode` は "t2v"、`images_used` は 0 になった。

## 比べ合わせの表

| run | strength | キーフレーム | 実時間 | `peak_vram_mb`／予約 | `m_per_step` | r／遅延 | 出力の場所 |
|---|---|---|---|---|---|---|---|
| kf_s0.50（Run B を使い回し） | 0.5 | あり | 670.8 秒 | 6,380／7,604 MB | 20 ステップすべて 3 | 0.9790604614351652／0 | `outputs/h3_a2v_g3b/kf_s0.50/`（元は `outputs/h3_a2v_g3/runB_s0.5/`） |
| kf_s0.75 | 0.75 | あり | 685.3 秒 | 6,380／7,604 MB | 20 ステップすべて 3 | 0.9796949437140606／0 | `outputs/h3_a2v_g3b/kf_s0.75/` |
| nokf_s0.50 | 0.5 | なし | 607.5 秒 | 6,380／7,608 MB | 20 ステップすべて 2 | 0.978847654156697／0 | `outputs/h3_a2v_g3b/nokf_s0.50/` |
| nokf_s0.75 | 0.75 | なし | 616.2 秒 | 6,380／7,608 MB | 20 ステップすべて 2 | 0.9797313288793691／0 | `outputs/h3_a2v_g3b/nokf_s0.75/` |

- r は左右の平均の値。
- `peak_vram_mb` は 4 本とも decode の段の値。

## 各回の細かい数値

| 項目 | kf_s0.50 | kf_s0.75 | nokf_s0.50 | nokf_s0.75 |
|---|---|---|---|---|
| job_id | 651142e9-14ca-4fd4-b50b-5087ffca6b5f | 91256d8f-f1e7-48b6-942b-b9fefea4ae1a | 9937adf2-5b83-4ed2-a218-b42ae20a06fd | 2eca7aca-f6ff-4744-86d4-3bc9f2942b97 |
| `generation_mode` | i2v | i2v | t2v | t2v |
| `generation_time_seconds` | 668.32 | 682.75 | 605.03 | 614.79 |
| encode の段の秒 | 15.72 | 29.43 | 20.17 | 19.75 |
| image_encode の段の秒 | 1.25 | 1.2 | （段なし） | （段なし） |
| audio_encode の段の秒 | 0.29 | 0.38 | 0.64 | 0.61 |
| denoise の段の秒 | 621.3 | 620.82 | 554.76 | 564.18 |
| decode の段の秒 | 29.55 | 29.47 | 29.26 | 30.04 |
| DiT の読み込み（denoise に含む） | 11.82 秒 | 11.84 秒 | 11.39 秒 | 11.62 秒 |
| ノイズ除去の段の山（割り当て／予約） | 3,939／6,118 MB | 3,939／6,118 MB | 3,838／4,726 MB | 3,838／4,726 MB |
| `num_ooms`・`num_alloc_retries` | 0・0 | 0・0 | 0・0 | 0・0 |
| `inactive_split_peak_mib` | 568 | 568 | 676 | 676 |
| 系列の長さ（text／cond／audio／video／合計） | 1038／1008／300／27216／29562 | 同じ | 22／0／300／27216／27538 | 同じ |
| `denoise.mask` の audio | rows 300・0.5 | rows 300・0.25 | rows 300・0.5 | rows 300・0.25 |
| GPU のメモリの山（nvidia-smi） | 8,812 MiB | 8,841 MiB | 8,848 MiB | 8,827 MiB |
| 共有メモリ（最小〜最大） | 1,175.7〜1,202.5 MiB | 846.0〜1,234.6 MiB | 1,171.3〜1,200.8 MiB | 1,169.0〜1,208.8 MiB |
| 左チャンネル単独の r／遅延 | 0.9789201697977215／0 | 0.9796029416232643／0 | 0.9787168924176372／0 | 0.9796997157093611／0 |

### `metadata.a2v`

4 本で共通の値:

- `audio_latents` 150・`required_samples` 120000・`encoded_samples` 124356
- `channels_used` "mono->stereo"・`resampler` "pyav"・`encoder_tensors` 134
- `audio_output` "decoded"・`muxed_audio_samples` 120000
- `source`: {duration_sec 3.8861, sample_rate 44100, channels 1}

回ごとに違う値:

- `mask_value` と `strength`:
  - strength 0.5 の 2 本は `mask_value` 0.5。
  - strength 0.75 の 2 本は `mask_value` 0.25（= 1 − strength）。
- `source_audio_id`:
  - kf_s0.50 は `81b7942a-…`（G3 のときに上げたもの）。
  - 残りの 3 本は `9869ff5b-…`。

`h3.phases.decode` は 4 本とも次のとおり。

- `vae_source` "file"・`audio_samples` 120000・`audio_samples_decoded` 120000
- `frames` 90・`audio_output` "decoded"

### ffprobe と音声の検査

- 4 本とも同じ結果。
  - 映像は 90 フレーム・3.750000 秒。
  - 音声は 32 kHz・2 チャンネル・3.750000 秒。
  - ファイルの長さは 3.750000 秒。
- 長さを指定せずに復号すると、1 チャンネルあたり 120,832 サンプルになる。AAC の最後のフレームの埋め草によるもので、G3 の報告と同じ。
- 相関の取り方は G3 と同じ。
  - 入力の wav を 32 kHz・2 チャンネルにした。
  - ±2,400 サンプルの範囲で遅延を探し、Pearson の相関の最大値を取った。
- 結果はそれぞれの `audio_check.json`。

### ログ

- 新しく回した 3 本は、各フォルダに `h3_worker.log.excerpt` と `server.log.excerpt` を置いた。
- warn・error・traceback・unknown・ignore・oom の語で調べたところ、該当は 0 件。

## 気づいたこと（判断はしていない）

- キーフレームなしでは denoise が約 57〜66 秒短かった（554.76／564.18 秒と 620.82／621.3 秒の比較）。
  - 系列から条件画像の 1,008 トークンが消え、`m_per_step` が 3 から 2 になった。
  - 速くなったのはこのためと見られる（推測）。
- kf_s0.75 の encode の段は 29.43 秒で、ほかの 15.72〜20.17 秒より長かった。中身は確かめていない。
- r は strength 0.75 の 2 本（0.9797 前後）が、0.5 の 2 本（0.9788〜0.9791）よりわずかに高い。どちらも decoded の経路で、差は小さい。

## 成果物

- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\h3_a2v_g3b\`（git の管理外）
  - `kf_s0.50\`: G3 の Run B を写したもの＋`REUSED_FROM_G3_runB.txt`
  - `kf_s0.75\`・`nokf_s0.50\`・`nokf_s0.75\`: output.mp4・metadata.json・job.json・request.json・row.json・audio_check.json・ログの抜粋
  - `drive_g3b.py`（台本）・`g3b.out`（台本の出力）・`backend_g3b.log`（起動ログ）
- 元のジョブのフォルダ: `outputs\91256d8f-…`・`outputs\9937adf2-…`・`outputs\2eca7aca-…`
- 素材を上げたもの: `uploads\14da35c1-…`（画像）・`uploads\audios\9869ff5b-…`（音声）。Uploads は一時置き場なので、そのまま残している。
