# 1-G2〜1-G4 の準備の報告（台帳 §1-97 第 1 弾）

## 結論

- 1-G2・1-G3・1-G4 の台本、試験画像、手順書をそろえました。GPU は使っていません。CPU だけの予行も含めて、CUDA は一度も初期化していません（`cuda_initialized: false`）。
- コミットはしていません。管理下のファイルも変えていません。書いたのは `outputs\h3-keyframes-2026-10-19\`（git の管理外。`.gitignore:26 outputs/*`）と、この報告だけです。
- 1-G2 の台本の予行（`--dry-run`。CPU だけで重みは読まない）は終了コード 0 でした。正解データの import は全部通りました。CPU で計算できる部分は、2 通りとも全部一致しました。
  - A: トークン列と印
  - B: 画像の入力
  - E: 条件行の並び・印・時刻の番号・RoPE 座標の float64

## 作ったもの（すべて `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\h3-keyframes-2026-10-19\`）

| ファイル | 中身 |
|---|---|
| `g2_keyframe_parts.py`／`.cmd` | 1-G2 の台本。g3 を複数枚の新しい引数で書き直したもの。2 通り（`first_last`＝[0,123]／[0,−1]、`last_only`＝[123]／[−1]）で A・B・C・E を照合し、D は画像ごとに記録だけ。結果は `g2_keyframe_parts.json`、標準出力の写しは `.log`。`--dry-run` を付けると CPU だけで動く |
| `g2_keyframe_parts.dryrun.json`／`.dryrun.log`／`.dryrun.stdout` | 予行の結果 |
| `g3_t2v.cmd`・`g3_i2v_first.cmd`・`g3_i2v_first_vram16.cmd` | 1-G3 のセルフテスト（合否 2 本と記録 1 本） |
| `g4_first_last.cmd`・`g4_last_only.cmd`・`g4_kf10.cmd` | 1-G4 のセルフテスト（vram16 on・先読み on・sdpa） |
| `psnr_keyframes.py` | PSNR を計算する（CPU）。`--ref N=PATH`、末尾は `--ref last=PATH` と書く |
| `extract_keyframes.py`／`.json` | 試験画像を取り出す台本（CPU・PyAV） |
| `kf_first.png`・`kf_last.png`・`kf10_000〜123.png` | 試験画像 |
| `worker_help.txt` | エンジン venv で出した `python -m engine_h3.worker --help`（モデルは読まない） |
| `RUNBOOK.md` | 監督向けの手順書（日本語）。コマンド・作業場所・venv・見込み時間・読む鍵・要確認 |

- `.cmd` 6 本の引数は、CPU の差し替え版（`_selftest_parser` と `_selftest_images` だけを呼ぶもの）で全部解釈できることを確かめました。画像・位置・vram16・先読みの値が意図どおりです。
- 差し替え版は scratchpad の `kf197\parsecheck\` に置きました。

## 予行の出力（要点）

- ファイルの有無: TE・tokenizer・VAE・正解データ・画像 2 枚がすべて存在します。
- 正解データの import:
  - `minimax_h3_text_encoder` から `MiniMaxH3TextEncoder`・`image_token_counts`・`presentation_fl2va`
  - `minimax_h3_video_vae` から `MiniMaxH3VideoVAE`
  - `pipeline_coords` から `MiniMaxH3Unit_PackedSequenceBuilder`
  - 読んだのはどれも `outputs\h3-gate2\oracle\diffsynth_h3\` の下のファイルです。
- `outputs\h3-gate2\oracle\diffsynth_h3\pipeline_coords.py` は、保管庫の `Docs\Outputs-archive\h3-gate1\oracle\diffsynth_h3\pipeline_coords.py` と改行コードを除いて同じでした。
- `first_last`:
  - トークン 537 個（視覚 240＋240）、grid [[1,24,40],[1,24,40]]、pixel_values [1920,1536]
  - 使う行 10,311、条件行 480
  - 時刻の座標は ours も DiffSynth も 537.0 と 742.0
  - A・B 入力・E がすべて一致
- `last_only`:
  - トークン 289 個、条件行 240
  - 時刻の座標は ours も DiffSynth も `493.99999999999994`
  - 単純な式 `L + (5/3)(F−1)` だと 494.0 になる組です。1a で F−1 だけ特別に扱う分岐を残したことが、ここで効いています。
  - すべて一致
- B のうち画像読み取り部の出力と DeepStack、C、D は重みが要るので、本番（GPU）でしか確かめられません。

## 試験画像

- 元の動画 `outputs\h3-gate1\g5_t2v_after_review.mp4` はありました。sha256 は `0da6111c…` で、T2V の基準そのものです。フレームは 124 枚。
- `kf_first.png`: 640×384・275,423 バイト。sha256 `d96dcb89…` で、`outputs\h3-gate2\i2v_input.png` とバイト単位で同じでした。
- `kf_last.png`: 640×384・254,034 バイト。sha256 `8dac049b…`。
- 10 枚の位置: 0・13・27・41・55・68・82・96・110・123。元の動画の同じ位置のフレームを使います。

## 指紋の条件（見つけたものと出典）

- **T2V `0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9`**
  - 条件: G5 のプロンプト（黄色いレインコートの文。g3 の `PROMPT` と同じ）・seed 42・640×384・124 フレーム・20 ステップ・shift 12／3・sdpa・先読み on・blocks 8・vram16 off・deterministic なし
  - 出典: VERIFICATION_LOG §165 の訂正行・§166.3、§167.4（3-G2 の取り直し）。実行に使った台本は `Docs\Outputs-archive\h3-gate3\g2_selftest_g5prompt.cmd`。
  - 既定のプロンプトでは `9161e030…` になります（§167.4 の教訓）。
- **先頭 I2V `77e29687bf84f58a2af1b0215e6d1ebebc307f1df3346390504c784b5cddd15a`**
  - 2a-G4 の J1 の条件です。上の条件に `images=[{i2v_input.png, 0}]`・`block_swap_prefetch=false` を加え、`keep_resident` の鍵を外したもの。
  - 出典: `Docs\Outputs-archive\h3-gate2\g4_i2v_resident.py`（JOBS と selftest の引数）、`outputs\h3-gate2\g4_i2v_resident.json` の J1、VERIFICATION_LOG §166.3 の表。
  - セルフテストでは `keep_resident: false` を送ります。ワーカーは鍵が無いときの既定を False とするので（`_or("keep_resident", False, bool)`）、同じ意味です。
  - vram16 は、当時は鍵自体が無く、今は false を送ります。これも同じ意味です。
- PSNR の台本は 2a-G4 と同じ計算です。J1 の mp4 で先頭フレーム 30.3dB を CPU で再現しました。§166 の 29.78dB は 2b-G2 の値で、計算式は同じです（ffmpeg で取り出したもの）。
- 比べる相手として、J1（先頭だけ）の末尾フレームと `kf_last` の PSNR を測りました。25.35dB です。手順書 §3.4 に書きました。

## API 経由（1-G3 の記録だけ）

- 台本は `Docs\Outputs-archive\h3-gate2\run_2b_g2.py` を使い、アプリの `.venv` で動かします。**`--out …\h3-keyframes-2026-10-19\g3_api` を必ず付けます。** 付けないと第 2 弾の `outputs\h3-gate2\g2b_api` を上書きします。
- 今回の改修で、判定項目のうち 2 つが偽になることが分かっています（終了コードは 1 になる見込み）。
  - `constraints_match_plan`: 制約に `keyframes` が加わったため。
  - `i2v_crop_path`: 1a で `source_size` が `[[384,384]]` というリストに変わったため。
  - 手順書には「この 2 つの偽は無視して、残りの 12 項目を見る」と書きました。
- API の既定は vram16 on（裁定 44）・deterministic true です。I2V の sha256 は `jobs.i2v.row.sha256` に入ります。

## 解決できなかったこと（要確認）

1. 所要時間が分からないもの: vram16 on と先読み off の I2V、条件行 2 枚と 10 枚の生成。実測がありません。手順書の見込みは推定です。
2. 10 枚のときの VRAM の山。
3. 1-G2 の B（画像読み取り部の出力と DeepStack）・C・D は GPU でしか確かめられません。
4. 監督が決めること: API 経由の I2V に使う画像。既定の正方形 384×384 なら 2b-G2 の 29.78dB と比べられます。640×384 の `i2v_input.png` を使う場合は、`i2v_crop_path` がもう一段偽になります。
