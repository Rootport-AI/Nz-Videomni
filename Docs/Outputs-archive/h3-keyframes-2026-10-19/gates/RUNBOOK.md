# 台帳 §1-97 第 1 弾 GPU の門（1-G2・1-G3・1-G4）の実行手順

作成 2026-10-19（準備担当。GPU は一度も使っていない）。計画は `C:\Users\phenotype nezu\.claude\plans\scalable-floating-thimble.md` の §2.3・§2.4・§5。

## 結論（先に）

- 実行するものは全部で 9 本（1-G2 が 1 本、1-G3 が セルフテスト 3 本と API 経由 1 回〔ジョブ 3 本〕、1-G4 が 3 本）。**どれも GPU を使うので、必ず 1 本ずつ順に実行する。** 合計の見込みは 約 45〜60 分。
- 合否の門は 3 つ。①1-G2 の終了コードが 0、②`g3_t2v` の指紋が `0da6111c…`、③`g3_i2v_first` の指紋が `77e29687…`。残りは記録だけ（基準がない）。
- 台本と試験画像は全部このフォルダにある。作業場所はどれも**リポジトリ直下** `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni`。
- 「要確認」は GPU を使わないと確かめられない点。§5 にまとめた。

## 0. 共通の前提

- 作業場所: `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni`（PowerShell で `Set-Location` してから実行）。
- 仮想環境（venv）:
  - セルフテストと 1-G2: エンジン用の `.venv-engine-h3`。各 `.cmd` の中で `PYTHONPATH` と `PYTHONUTF8=1` を設定済み。
  - API 経由: アプリ用の `.venv`。
- モデルのパス: 4 本と tokenizer は第 2 弾の台本（`Docs/Outputs-archive/h3-gate2/g4_i2v_resident.py`・`Docs/Outputs-archive/h3-gate3/g2_selftest_g5prompt.cmd`）と同じで、リポジトリ直下からの相対パス。
  - `models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors`
  - `models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors`
  - `models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors`
  - `models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors`
  - `models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer`
  - 5 つとも存在することは確認済み。
- プロンプトは全部、第 1 弾 G5 の「黄色いレインコート」の文。
  - `A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in puddles. She looks at the camera and says <d>[English] It finally stopped raining.</d>`
  - セルフテストの既定のプロンプトでは指紋が `9161e030…` になるので、必ず `--prompt` で渡す（VERIFICATION_LOG §167.4 の教訓）。`.cmd` には全部入れてある。
- 共通の条件: seed 42・640×384・124 フレーム・20 ステップ・shift 12／3・ブロック 8 枚を GPU に置く・deterministic なし。どれもセルフテストの既定値で、`python -m engine_h3.worker --help` で確かめた（`worker_help.txt`）。
- 始める前に確かめること（GPU を使わない）:
  ```powershell
  Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Select-Object ProcessId,CommandLine   # H3 のワーカーやバックエンドが動いていないこと
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv
  ```
  セルフテストと 1-G2 は、バックエンドが動いていると VRAM を取り合う。**API 経由の手順（§2.4）以外では、バックエンドを止めておく。**

## 1. 1-G2 正解データとの照合（合否）

| 項目 | 内容 |
|---|---|
| 台本 | `outputs\h3-keyframes-2026-10-19\g2_keyframe_parts.py`（第 2 弾の `g3_i2v_parts.py` を、複数枚を受ける新しい引数に合わせて書き直したもの） |
| コマンド | `cmd /c outputs\h3-keyframes-2026-10-19\g2_keyframe_parts.cmd` |
| 見込み | 約 3〜5 分。第 2 弾の 2a-G3 は 1 通りで 71 秒。今回は 2 通りで、DiffSynth 側のテキストエンコーダを 4 回動かす |
| 合否 | 終了コード 0。2 通り（[先頭＋末尾]・[末尾だけ]）とも A・B・C の配線・E が一致 |
| 読むもの | `g2_keyframe_parts.json` の `verdict.pass_A_B_C_E_both_cases`、`cases.<名前>.verdict`、`D_vae_encode.first／last`（D は記録だけ。2a-G3 では最大差 1.52e-2）。標準出力の写しは `g2_keyframe_parts.log` |

- 照合の相手（正解データ）は `outputs\h3-gate2\oracle\diffsynth_h3\`。テキストエンコーダと VAE の写しはここにしかない。`pipeline_coords.py` は保管庫の `Docs\Outputs-archive\h3-gate1\oracle\` と同じ中身（改行コードを除いて差 0 を確認）。
- 2 通りの組み立て:
  - `first_last`: 位置 [0, 123]・画像 [kf_first, kf_last]・DiffSynth 側は `keyframe_indices=[0, -1]`
  - `last_only`: 位置 [123]・画像 [kf_last]・DiffSynth 側は `[-1]`
- CPU だけの予行（`--dry-run`）は実施済みで、合格。結果は `g2_keyframe_parts.dryrun.json` にある。重みは読んでいない。
  - 次の 3 点が 2 通りとも全部一致した。A のトークン列と印、B の画像の入力（pixel_values と grid）、E（条件行の並び・AdaLN の印・20 ステップの時刻の番号・RoPE 座標の float64 の完全一致）。
  - 末尾だけの時刻の座標は、ours も DiffSynth も `493.99999999999994` だった。単純な式 `L + (5/3)(F−1)` で計算すると 494.0 になる組で、1a 報告の「F−1 だけ特別に扱う分岐を残した」判断を裏づける。
  - GPU の本番で新たに確かめるのは、B の画像読み取り部（vision tower）の出力と DeepStack、C のテキストエンコーダの出力、D の VAE。

## 2. 1-G3 指紋の回帰

### 2.1 T2V（合否）

| 項目 | 内容 |
|---|---|
| コマンド | `cmd /c outputs\h3-keyframes-2026-10-19\g3_t2v.cmd` |
| 条件 | 3-G2 の取り直し（`Docs/Outputs-archive/h3-gate3/g2_selftest_g5prompt.cmd`）と同じ。G5 のプロンプト・sdpa・先読み on・vram16 off。フラグは明示した |
| 見込み | 約 2.5〜3 分（3-G2 の取り直しで生成 143.6 秒） |
| 合否 | `g3_t2v.json` の `rounds[0].sha256` が `0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9` |
| 参考 | 根拠は VERIFICATION_LOG §165 の訂正行・§166.3・§167.4。テキストのトークン数は 41 になるはず（`rounds[0].done.h3.phases.encode.text_tokens`） |

### 2.2 先頭だけの I2V（合否）

| 項目 | 内容 |
|---|---|
| コマンド | `cmd /c outputs\h3-keyframes-2026-10-19\g3_i2v_first.cmd` |
| 条件 | 2a-G4 の J1（`Docs/Outputs-archive/h3-gate2/g4_i2v_resident.py`）と同じ。G5 のプロンプト・`--image outputs/h3-gate2/i2v_input.png`・`--no-prefetch`・sdpa・`--no-vram16` |
| 見込み | 約 4〜4.5 分（J1 は先読み off で約 250 秒） |
| 合否 | `g3_i2v_first.json` の `rounds[0].sha256` が `77e29687bf84f58a2af1b0215e6d1ebebc307f1df3346390504c784b5cddd15a`（`outputs/h3-gate2/g4_i2v_resident.json` の J1 の値） |
| 参考 | 先頭フレームの PSNR は 30.3dB のはず。§3.4 の台本で `--ref 0=outputs/h3-gate2/i2v_input.png` を渡す。今回の台本で J1 の mp4 から 30.3dB が再現することを CPU で確かめた |

- 入力画像 `outputs/h3-gate2/i2v_input.png`（640×384・275,423 バイト）があることは確認済み。中身は `kf_first.png` とバイト単位で同じ（sha256 `d96dcb89…`）。
- J1 との違いは 1 点だけ: J1 は生成の伝言から `keep_resident` の鍵を外していた。セルフテストは `keep_resident: false` を送る。
  - ワーカーは鍵が無いときの既定を False として読む（`engine_h3/worker.py` の `_or("keep_resident", False, bool)`）ので、意味は同じ。
  - J1 の頃は `vram16_mode` の鍵自体がまだ無かった。今は `false` を送り、ワーカーの既定も False なので、これも同じ意味。

### 2.3 先頭だけの I2V・vram16 on（記録だけ）

| 項目 | 内容 |
|---|---|
| コマンド | `cmd /c outputs\h3-keyframes-2026-10-19\g3_i2v_first_vram16.cmd` |
| 条件 | §2.2 から `--vram16` だけを変えた（先読み off も同じ。変える条件を 1 つにするため） |
| 見込み | 約 4〜5 分（要確認。vram16 on と先読み off を組み合わせた実測はまだ無い） |
| 読むもの | `g3_i2v_first_vram16.json` の `rounds[0].sha256`（新しい基準として §170 に残す）、`77e29687…` と一致するか、`rounds[0].done.peak_vram_mb`／`peak_vram_reserved_mb`、`vram16_mode_used` |

- T2V では on と off の指紋が全桁一致した（§169 A-G3b）。I2V で一致するかどうかは未知なので、合否にはしない。

### 2.4 API 経由の T2V と先頭だけの I2V（記録だけ）

第 2 弾 2b-G2 の台本 `Docs\Outputs-archive\h3-gate2\run_2b_g2.py` をそのまま使う。手順の全体は同じフォルダの `README_2b_g2.md`。アプリの `.venv` で動き、MCP サーバーを stdio の子プロセスとして起動して叩く（§39.5／§39.6 の経路）。リポジトリのコードは import しない。

1. バックエンドを起動する（**本計画の改修を入れたコードで起動し直す**。古いプロセスが残っていれば、オーナーに断ってから止める）。
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File "S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\run.ps1"
   ```
   起動の確認: `(Invoke-RestMethod http://127.0.0.1:18620/api/v1/status) | Select-Object server,state,base_model,pipeline_loaded`
2. 台本を実行する。**`--out` を必ず付ける。** 付けないと第 2 弾の結果 `outputs\h3-gate2\g2b_api\` を上書きする。
   ```powershell
   & "S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\.venv\Scripts\python.exe" "S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\Docs\Outputs-archive\h3-gate2\run_2b_g2.py" --out "S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\h3-keyframes-2026-10-19\g3_api"
   ```
   - 画像は既定の `i2v_input_square.png`（384×384。台本と同じフォルダ）。2b-G2 と同じ切り抜きの経路を通り、先頭フレームの PSNR を 2b-G2 の 29.78dB と比べられる。
   - 640×384 のまま渡したいときは `--image "S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\h3-gate2\i2v_input.png"` を足す。ただし台本の判定項目 `i2v_crop_path` は 384×384 を前提にしているので、偽になる。どちらにするかは監督が選ぶ。
3. 後片付け: `README_2b_g2.md` §3 のとおり。①ワーカーが残っていないか確かめる。②バックエンドを止める。③`state.json` を `outputs\h3-keyframes-2026-10-19\g3_api\state.json.before` から戻す（止めた後に行う）。

| 項目 | 内容 |
|---|---|
| ジョブ | T2V（常駐 on）→ T2V（常駐 on）→ I2V（常駐の鍵なし。保持中のものを使い、終わりに解放） |
| 条件 | API の既定。先読み on・**vram16 on**（裁定 44 で H3 の既定は on になった。2b-G2 の頃は vram16 自体が無かった）・読み込みは `deterministic: true`・sdpa |
| 見込み | 約 12〜20 分（2b-G2 の実績: T2V 157／130 秒・I2V 139 秒＋読み込み） |
| 読むもの | `g3_api\g2b_api.json` の `recorded`。`t2v1_sha256`・`t2v2_sha256`・`t2v1_sha256_matches_0da6111c`・`i2v_psnr_db`・`peak_vram_mb`。I2V の sha256 は `jobs.i2v.row.sha256`（`job_row` の中身で確認済み） |

- **台本の終了コードは 1 になる見込み。** 次の 2 項目は、今回の改修で偽になることが分かっている。今回は記録が目的なので、この 2 つの偽は無視してよい。そのほかの 12 項目が真かどうかを見る。
  - `constraints_match_plan`: 第 2 弾の制約の JSON と完全に一致することを求めている。今回の改修で `GET /models` の `constraints` に `"keyframes": {"step": 1, "offset": 0}` が加わった。
  - `i2v_crop_path`: `image_encode.source_size` が `[384, 384]` であることを求めている。1a で `source_size` は画像ごとのリスト `[[384, 384]]` に変わった（1a 報告）。切り抜きの経路を通ったかどうかは、`jobs.i2v.row.image_encode.source_size` が `[[384, 384]]` で `canvas` が `[640, 384]` であることを目で確かめる。
- T2V の指紋は `0da6111c…` になる見込み（§169 で vram16 on も off も一致・2b-G2 で deterministic true でも一致）。ただし API 経由で vram16 on の T2V を測るのは初めてなので、記録だけ。
- I2V の指紋は、API 経由では初めて基準として残す。2b-G2 では画像が違い（正方形の切り抜き）、mp4 にレシピを埋め込んでいた（`embed_mp4_metadata` が既定）ので、sha256 は比べる相手がない。

## 3. 1-G4 末尾と中間のキーフレーム（記録だけ・合否の線なし）

条件は製品の既定。vram16 on・先読み on・sdpa。セルフテストでは vram16 の既定が off なので、`--vram16` を明示した。試験画像は `outputs/h3-gate1/g5_t2v_after_review.mp4`（sha256 `0da6111c…`。T2V の基準の動画そのもの）から CPU で取り出した（`extract_keyframes.py`・結果 `extract_keyframes.json`）。

| 画像 | 大きさ | 中身 |
|---|---|---|
| `kf_first.png` | 640×384・275,423 バイト | 先頭フレーム（`i2v_input.png` と同一） |
| `kf_last.png` | 640×384・254,034 バイト | 末尾フレーム（124 枚目＝frame 123） |
| `kf10_000.png`〜`kf10_123.png` | 640×384 | frame 0・13・27・41・55・68・82・96・110・123 |

### 3.1 先頭＋末尾

- コマンド: `cmd /c outputs\h3-keyframes-2026-10-19\g4_first_last.cmd`（`--keyframe 0=kf_first.png --keyframe 123=kf_last.png --vram16`）
- 見込み: 約 3〜4 分（要確認。vram16 on の T2V はノイズ除去が 115.5 秒。条件行が 2 枚で約 480 行増える）
- 読むもの:
  - `g4_first_last.json` の `rounds[0].sha256`
  - `rounds[0].done.images_used`（2 のはず）・`peak_vram_mb`・`peak_vram_reserved_mb`・`vram16_mode_used`
  - 進捗の分母。`rounds[0].progress` の encode が 4 段のはず
  - テキストのトークン数 `rounds[0].done.h3.phases.encode.text_tokens`（537 のはず。1-G2 の予行の値）
- PSNR（§3.4）: `--ref 0=… --ref last=…`

### 3.2 末尾だけ

- コマンド: `cmd /c outputs\h3-keyframes-2026-10-19\g4_last_only.cmd`（`--keyframe 123=kf_last.png --vram16`）
- 見込み: 約 3〜4 分（要確認）
- 読むもの: §3.1 と同じ。`images_used` は 1 のはず。`text_tokens` は 289 のはず。
- PSNR（§3.4）: `--ref last=…`。比べるために `--ref 0=kf_first.png` も付ける。

### 3.3 10 枚（VRAM の山の記録）

- コマンド: `cmd /c outputs\h3-keyframes-2026-10-19\g4_kf10.cmd`
- 位置: 0・13・27・41・55・68・82・96・110・123。0 と 123 を含む 10 か所を、約 13〜14 フレームおきに選んだ。
- 画像: 元動画の同じ位置のフレーム。
- 画像読み取り部に入るのは 0 と 123 の 2 枚だけ。`text_tokens` は 537 のはず。
- 条件行は 10×240＝2,400 行増える（全体で約 12,000 行）。
- 見込み: 約 3.5〜5 分（要確認）
- 読むもの:
  - `g4_kf10.json` の `rounds[0].done.peak_vram_mb`／`peak_vram_reserved_mb`。§3.1・§3.2 と並べて台帳に 1 行書く（計画 §7.5）。
  - `rounds[0].done.h3.phases.denoise` の vram16 の記録。
  - sha256。
- PSNR（任意）: 各位置の画像を `--ref N=kf10_NNN.png` で 10 個渡せる。

### 3.4 PSNR の計算（CPU だけ・数秒）

- 台本: `outputs\h3-keyframes-2026-10-19\psnr_keyframes.py`
- 計算のしかた: 2a-G4 の `psnr_first_frame` と同じ。PyAV で rgb24 に復号し、`10·log10(255²/MSE)` を float64 で計算する。参照画像は、ワーカーと同じ `load_canvas_image` でキャンバスに置く。
- 2b-G2 の 29.78dB は、`run_2b_g2.py` の `psnr` が ffmpeg で rgb24 を取り出して計算したもの。式は同じ。
- 末尾は `last` と書く（`-1` と書くと引数の解釈器がオプションと取り違える）。

```powershell
.venv-engine-h3\Scripts\python.exe -I outputs\h3-keyframes-2026-10-19\psnr_keyframes.py outputs\h3-keyframes-2026-10-19\g4_first_last.mp4 --ref 0=outputs\h3-keyframes-2026-10-19\kf_first.png --ref last=outputs\h3-keyframes-2026-10-19\kf_last.png --json outputs\h3-keyframes-2026-10-19\g4_first_last.psnr.json
.venv-engine-h3\Scripts\python.exe -I outputs\h3-keyframes-2026-10-19\psnr_keyframes.py outputs\h3-keyframes-2026-10-19\g4_last_only.mp4 --ref 0=outputs\h3-keyframes-2026-10-19\kf_first.png --ref last=outputs\h3-keyframes-2026-10-19\kf_last.png --json outputs\h3-keyframes-2026-10-19\g4_last_only.psnr.json
```

- 読むもの: `psnr."last=kf_last.png".psnr_db`。
- 比べる相手（CPU で測った既存の値）:
  - 先頭だけの I2V（J1 の mp4 `outputs/h3-gate2/g4_J1_i2v.mp4`）では、先頭フレーム対 `kf_first` が 30.3dB、**末尾フレーム対 `kf_last` が 25.35dB**だった。
  - 同じ seed とプロンプトなので、末尾を指定しなくても末尾は元動画に近い。末尾を条件に入れた効果は、この 25.35dB からどれだけ上がるかで見る。合否の線は置かない。

## 4. 実行の順序と所要時間の目安

| 順 | 門 | コマンド | 見込み |
|---|---|---|---|
| 1 | 1-G2 | `g2_keyframe_parts.cmd` | 3〜5 分 |
| 2 | 1-G3 合否 | `g3_t2v.cmd` | 2.5〜3 分 |
| 3 | 1-G3 合否 | `g3_i2v_first.cmd` | 4〜4.5 分 |
| 4 | 1-G3 記録 | `g3_i2v_first_vram16.cmd` | 4〜5 分 |
| 5 | 1-G4 | `g4_first_last.cmd` → PSNR | 3〜4 分 |
| 6 | 1-G4 | `g4_last_only.cmd` → PSNR | 3〜4 分 |
| 7 | 1-G4 | `g4_kf10.cmd` | 3.5〜5 分 |
| 8 | 1-G3 記録（API） | バックエンドを起動 → `run_2b_g2.py --out …\g3_api` → 後片付け | 12〜20 分 |

- 主メモリ: 常駐を使わないセルフテストでは、コミット量の山が約 34.5GiB（3-G2）。API 経由の常駐中は約 43GiB（2b-G2）。
- どれかが失敗したときに見る場所:
  - セルフテストでは、コンソールの出力と `logs\h3_worker.log`（ワーカーの `chdir` 先はリポジトリ直下）。
  - API 経由では `README_2b_g2.md` の「失敗したときに見る場所」。

## 5. 要確認（GPU なしでは確かめられなかった点）

1. §2.3・§3.1〜3.3 の所要時間（vram16 on と先読み off の組み合わせ、条件行が 2 枚と 10 枚の場合の実測が無い）。
2. §3.3 の VRAM の山。vram16 on の分割は全体の行数だけを見る（計画 §7.5）。10 枚で上限の予約に収まるかどうかは、実測しないと分からない。
3. 1-G2 の B（画像読み取り部の出力と DeepStack）・C（テキストエンコーダの出力）・D（VAE）。予行では重みを読んでいない。
