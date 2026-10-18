# MiniMax H3 第 2 弾の門（2a と 2b-G2）の台本と結果

作成: 2026-10-17／担当 C1。`outputs/h3-gate2/` は git の管理外なので、門をやり直せるように台本と結果の JSON をここに写した。記録の正本は `Docs/VERIFICATION_LOG.md`（第 2 弾の節）。境界の仕様は `../h3-stage2-reports-2026-10-17/INTERFACE2.md`。

## 置いてあるもの

| 門 | 台本 | 結果 |
|---|---|---|
| 2a-G2（TE の直接読みのビット一致と T2V の回帰） | `g2_te_direct.py`（`--mode baseline` は改修前のコードで基準を取る。`--mode compare` で照合）・`g2_encode_phase_timing.py`・`g2_encode_phase_in_worker.py` | `te_baseline_t2v.json`・`g2_te_direct.json`・`g2_t2v_regression.json`・`g2_encode_phase_*.json`。`g3_te_t2v_after_qwen3vlmodel.json` は TE を `Qwen3VLModel` に替えた後の同じ照合 |
| 2a-G3（I2V の部品を DiffSynth と照合） | `g3_i2v_parts.py` | `g3_i2v_parts.json` |
| 2a-G4（I2V 1 本・T2V の回帰・常駐） | `g4_i2v_resident.py`・`g4_allocated_probe.py` | `g4_i2v_resident.json`・`g4_allocated_probe.json` |
| 2a-G4 の取り直し（レビュー後。常駐あり・先読みありで T2V 2 ラウンド） | `python -m engine_h3.worker --selftest-generate … --prefetch --keep-resident --rounds 2` | `g4_retake_after_review.json` |
| 2a-G5（映像の復号の VRAM） | `g5_decode_vram.py` | `g5_decode_vram.json` |
| 2b-G2（API 経由の実生成。H3 の読み込み・T2V 2 本〔常駐 on〕・I2V 1 本〔384×384 の画像を 640×384 へ切り抜き〕。2026-10-18） | `run_2b_g2.py`（手順は `README_2b_g2.md`。試験画像 `i2v_input_square.png`）。アプリの venv で動き、MCP サーバーを stdio で叩く（VERIFICATION_LOG §39.5／§39.6 の経路）。実行は監督が PowerShell から、オーナーが `run.bat` で起動したバックエンドに対して行った | `g2b_api.json`（14 項目すべて合格。mp4・ログの抜粋・先頭フレームは `outputs/h3-gate2/g2b_api/` にあり、ここには写していない） |

- `oracle/diffsynth_h3/` は正解側を import するための置き物（`core/` と `pipeline_coords.py` は第 1 弾の `../h3-gate1/oracle/` と同じ）。DiffSynth-Studio（Apache-2.0）のコミット `4e34969` から `diffsynth/models/minimax_h3_text_encoder.py` と `diffsynth/models/minimax_h3_video_vae.py` を `oracle/diffsynth_h3/models/` に置く（ここには複写していない。取得の手順は `../h3-gate1/README.md` と同じで、`git show 4e34969:<パス>`）。
- 実行: このフォルダの中身を `outputs/h3-gate2/` に写し、リポジトリ直下で `PYTHONPATH` をリポジトリ直下にして `.venv-engine-h3\Scripts\python.exe outputs\h3-gate2\<台本>`。どれも GPU を使うので同時に 1 本だけ。G2 の基準（`te_baseline_t2v.pt`）は 2a 前半の改修前のコードでしか取れない。G3・G4 の試験画像 `i2v_input.png` は `outputs/h3-gate1/g5_t2v_after_review.mp4` の先頭フレーム（640×384）。

## 注記

- **第 1 弾の保管 `../h3-gate1/` の台本は、2a で消した旧クラス `H3VideoVAE`（今は `H3VideoEncoder`／`H3VideoDecoder` に分かれている）と旧 TE の名前（`Qwen3VLTextModel`・`layers.N`。今は `Qwen3VLModel`・`language_model.layers.N`）を参照するので、今のコードではそのままでは動かない。旧台本は直していない（当時の記録として残す）。**

- **2a-G4 の判定の鍵の意味（レビュー m-13 を受けた監督の裁定、2026-10-18）**
  - 門の文言は「合間の allocated が 0」から「常駐による上乗せが 0（常駐しないジョブと同じ値。cuBLAS の作業領域は除く）」に改めた。
  - 台本 `g4_i2v_resident.py` の判定の鍵も `resident_adds_no_allocation_between_jobs`（全ジョブの後の allocated が、何も保持しない J0 の後と同じ値）に直した。
  - **既に取った `g4_i2v_resident.json` は再実行していない。** その中の `verdict.allocated_zero_between_jobs: false` は旧い文言での判定である。新しい意味では、6 ジョブとも 8,650,752 B で同じ値なので成立する。
  - 残っている値は torch の cuBLAS の作業領域で、`g4_allocated_probe.json` で確かめた。
