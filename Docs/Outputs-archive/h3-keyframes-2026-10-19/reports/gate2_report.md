# 2-G1（中間のキーフレーム）の実行報告（台帳 §1-97 第 2 弾）

## 結論

- 2 本とも最後まで動きました（終了コード 0・OOM 0 回）。合否の線はありません。判断はオーナーの目視です。
- テキストエンコーダに入った画像は frame 0 と 123 だけでした。トークン数は見込みどおりです（中間だけ＝41、先頭＋中間＋末尾＝537）。
- 中間を条件に入れると、frame 62 の PSNR が上がりました。
  - 中間だけ: 31.56dB。比べる相手の末尾だけ（g4_last_only）は 24.38dB です。
  - 先頭＋中間＋末尾: 32.59dB。比べる相手の先頭＋末尾（g4_first_last）は 28.31dB です。
- VRAM の山は変わりませんでした（全体 5,065MB は VAE の復号）。条件行が 720 行に増えて、ノイズ除去が 132.2 秒になりました（先頭＋末尾は 129.1 秒）。
- 管理下のファイルは変えていません。コミットもしていません。GPU は空いています（Python のプロセス 0 個・使用 700MiB・使用率 0%）。

## 試験画像

- `kf_mid62.png`: `g5_t2v_after_review.mp4` の frame 62（124÷2）。640×384・282,840 バイト・sha256 `4f64ffa81599c7944f577e70146705fd4297a90ff1c3bd50d808666a8441809e`。
- 取り出し方は `extract_keyframes.py` と同じ（PyAV の rgb24 → PIL で保存）です。既存の画像と `extract_keyframes.json` を書き換えないように、同じ手順の小さな台本を scratchpad に置きました（`kf197\gate2\extract_mid.py`）。記録は `extract_kf_mid62.json` です。

## 結果

| 名前 | sha256 | 実時間／生成／ノイズ除去 秒 | 全体の山 MB | ノイズ除去の山 MB | images_used | text_tokens | 条件行／全長 |
|---|---|---|---|---|---|---|---|
| g5_mid_only | `135e1fb0b928c5ab51e2f85897e61f6bec6a9959f5941b3e823a13fd1f91cd6e` | 142.8／139.3／116.8 | 5,065／5,382 | 3,779／3,790 | 1 | 41 | 240／9,575 |
| g5_first_mid_last | `323e8fcc13cc6bb3a42c74ed33518837d0967330a633c7ab98945ec0f7369d55` | 167.3／163.7／132.2 | 5,065／5,378 | 3,858／3,896 | 3 | 537 | 720／10,551 |

- トークン数の出どころ: `rounds[0].done.h3.phases.encode.text_tokens`。`te_read.vision_tokens_per_image` は、中間だけで `[]`、3 枚で `[240, 240]` でした。
- `image_encode.source_size` は画像の数だけ並びました（1 枚／3 枚）。

## PSNR（dB）

| 動画 | frame 0 | frame 62 | frame 123 |
|---|---|---|---|
| g5_mid_only | 26.37 | **31.56** | 27.65 |
| g5_first_mid_last | 30.59 | **32.59** | 34.12 |
| g4_first_last（比較） | 30.54 | 28.31 | 34.09 |
| g4_last_only（比較） | 21.61 | 24.38 | 33.96 |
| g3_i2v_first（参考） | 30.30 | 25.91 | 25.35 |
| g4_kf10（参考・62 は条件に入っていない） | 30.55 | 32.73 | 34.32 |
| g3_t2v（元の動画と同一） | — | MSE 0 | — |

## 気づいたこと

1. g4_kf10 の frame 62 は 32.73dB でした。62 そのものは条件に入っていませんが、隣の 55 と 68 が入っています。3 枚で 62 を直接指定したとき（32.59dB）と同じくらいです。
2. 中間だけのとき、両端の PSNR（26.37／27.65）は、片方の端だけを指定したときの反対側の端（21.61／25.35）より高くなりました。
3. 中間だけの実行は、テキストエンコーダの段で T2V と同じ扱いになりました（画像トークン 0・エンコード 10.4 秒）。それでも進捗の encode は 4 段でした（画像の VAE エンコードの段があるため）。

## 実行したコマンド（作業場所 `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni`）

```
.venv-engine-h3\Scripts\python.exe -I <scratchpad>\kf197\gate2\extract_mid.py
<scratchpad>\kf197\gates\run_one.ps1 -Name g5_mid_only          # 中身は cmd /c outputs\h3-keyframes-2026-10-19\g5_mid_only.cmd
<scratchpad>\kf197\gates\run_one.ps1 -Name g5_first_mid_last
.venv-engine-h3\Scripts\python.exe -I outputs\h3-keyframes-2026-10-19\psnr_keyframes.py <mp4> --ref 0=kf_first.png --ref 62=kf_mid62.png --ref last=kf_last.png --json <name>.psnr.json
#   比較用: g4_first_last・g4_last_only・g3_t2v・g3_i2v_first・g4_kf10 を --ref 62=kf_mid62.png で → baseline_mid62_<name>.psnr.json
```

## 成果物（`S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\h3-keyframes-2026-10-19\`）

- 試験画像: `kf_mid62.png`・`extract_kf_mid62.json`
- 台本: `g5_mid_only.cmd`・`g5_first_mid_last.cmd`
- 実行ごと: `g5_mid_only.mp4`／`.json`／`.stdout`／`.exit.txt`／`.gpu_before.txt`、`g5_first_mid_last` も同じ組
- PSNR: `g5_mid_only.psnr.json`・`g5_first_mid_last.psnr.json`・`baseline_mid62_g4_first_last.psnr.json`・`baseline_mid62_g4_last_only.psnr.json`・`baseline_mid62_g3_t2v.psnr.json`・`baseline_mid62_g3_i2v_first.psnr.json`・`baseline_mid62_g4_kf10.psnr.json`
- 結果のまとめ: `GATES_RESULT.md` の末尾に節を追加、`gates_result.json` に `stage2` の鍵を追加（既存の鍵は変えていません）
- scratchpad: `kf197\gate2\extract_mid.py`・`kf197\gate2\addjson.py`
