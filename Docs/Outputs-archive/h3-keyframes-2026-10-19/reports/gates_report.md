# 1-G2・1-G3・1-G4 の実行報告（台帳 §1-97 第 1 弾）

## 結論

- **合否のある門は全部合格です。**
  - 1-G2: 2 通りとも A・B・C・E が一致しました。
  - 1-G3: T2V の指紋 `0da6111c…` と、先頭だけの I2V の指紋 `77e29687…` が一致しました。
- GPU 付き pytest: 176 件合格・飛ばし 0・失敗 0（20.9 秒）。
- 記録だけの項目も全部実施しました。API 経由の手順も実施しました。
  - 手順に迷うところがなく、ポートを使っているプロセスも無かったためです。
  - バックエンドは私が起動し、最後に私が止めました。
- 管理下のファイルは変えていません。コミットもしていません。
- 表と数値の全体は `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\h3-keyframes-2026-10-19\GATES_RESULT.md` にあります。機械が読む形は同じフォルダの `gates_result.json` です。

## 実行の一覧

| 順 | 名前 | 開始 | 実時間 | 見込み | 実行前の空き VRAM |
|---|---|---|---|---|---|
| 0 | GPU 付き pytest | 13:35 | 22 秒 | — | 15,372MiB |
| 1 | 1-G2 `g2_keyframe_parts` | 13:36 | 70 秒 | 3〜5 分 | 15,374MiB |
| 2 | `g3_t2v` | 13:38 | 132 秒 | 2.5〜3 分 | 15,374MiB |
| 3 | `g3_i2v_first` | 13:40 | 254 秒 | 4〜4.5 分 | 15,365MiB |
| 4 | `g3_i2v_first_vram16` | 13:45 | 259 秒 | 4〜5 分 | 15,365MiB |
| 5 | `g4_first_last` | 13:49 | 162 秒 | 3〜4 分 | 15,374MiB |
| 6 | `g4_last_only` | 13:52 | 156 秒 | 3〜4 分 | 15,365MiB |
| 7 | `g4_kf10` | 13:55 | 201 秒 | 3.5〜5 分 | 15,353MiB |
| 8 | API 経由 `run_2b_g2.py` | 13:59 | 446 秒 | 12〜20 分 | 15,349MiB |

- どの実行の前も、Python のプロセスは 0 個でした。GPU の使用は 688〜713MiB で、画面表示のぶんだけです。
- 1-G2 と API 経由は、見込みよりずっと短く終わりました。
  - 1-G2 は 70 秒で、2 通りでも第 2 弾の 2a-G3（1 通り）と同じくらいでした。
  - API 経由は 7.4 分でした。

## 結果の表

| 名前 | sha256 | 生成の秒数（ノイズ除去・1 秒あたりのステップ数） | 全体の山（使用／予約 MB） | ノイズ除去の山 | images_used | text_tokens |
|---|---|---|---|---|---|---|
| g3_t2v | `0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9` **一致** | 128.1（107.4・0.186） | 6,677／7,590 | 同じ | 0 | 41 |
| g3_i2v_first | `77e29687bf84f58a2af1b0215e6d1ebebc307f1df3346390504c784b5cddd15a` **一致** | 249.9（218.6・0.091） | 6,390／7,668 | 同じ | 1 | 289 |
| g3_i2v_first_vram16 | `77e29687…`（vram16 off と全桁一致。新しい基準） | 256.0（223.2・0.090） | 5,065／5,382 | 2,657／3,586 | 1 | 289 |
| g4_first_last | `37e66bba9ddcc87eebeea33eb7331617a75a099f608530b7cf329a4f3cdb6d52` | 158.8（129.1・0.155） | 5,065／5,378 | 3,858／3,894 | 2 | 537 |
| g4_last_only | `84272c2af13fce8bc3d1c3268eebc48a9163e3ae9531a738313d0745951525ab` | 152.0（120.7・0.166） | 5,065／5,392 | 3,819／3,858 | 1 | 289 |
| g4_kf10 | `1026bd06fab9c7b5aabc458cad60ad2523c7df9352827aeb2b9ad7f0af3512cd` | 197.0（163.9・0.122） | 5,065／5,378 | 3,859／3,896 | 10 | 537 |
| API T2V 1 本目／2 本目 | 2 本とも `0da6111c…` | 143.0／133.0（2 本目は常駐で読み込みを省略） | 5,065／5,382・5,066／5,402 | — | 0 | — |
| API I2V | `105b3fb532da7493eadd6065a8255f6034f3b3e54f4f047d11055ab105b60468`（新しい基準。mp4 にレシピを埋め込み・正方形の画像） | 144.1 | 5,067／5,410 | — | 1 | — |

- 画像のトークン数:
  - 1 枚のときは 240。
  - 2 枚と 10 枚のときは [240, 240]。10 枚でも、画像読み取り部に入るのは 0 と 123 の 2 枚だけです。
- 条件行と系列の全長:
  - 1 枚: 240 行・全長 9,823
  - 2 枚: 480 行・全長 10,311
  - 10 枚: 2,400 行・全長 12,231
- 進捗の encode は、画像ありで 4 段、T2V で 3 段でした。
- vram16 on の 4 本は、どれも OOM 0 回・確保のやり直し 0 回でした。上限は 14,023MiB、GPU に置くブロックは 4 枚です。
- 主メモリのコミット量の山:
  - セルフテスト: 28.5〜34.5GiB
  - API 経由（常駐中）: 40.3〜40.8GiB

### PSNR（dB）

| 動画 | 先頭対 kf_first | 末尾対 kf_last |
|---|---|---|
| J1 の基準 `outputs\h3-gate2\g4_J1_i2v.mp4`（同じ台本で測り直した） | 30.30 | 25.35 |
| g3_i2v_first・g3_i2v_first_vram16 | 30.30 | 25.35 |
| g4_first_last | 30.54 | **34.09** |
| g4_last_only | 21.61 | **33.96** |
| g4_kf10（各位置対 kf10_NNN） | 30.55 | 34.32。中間は 13: 31.46・27: 31.96・41: 32.71・55: 33.35・68: 32.85・82: 33.30・96: 34.00・110: 34.59 |
| API I2V（台本が計算。先頭対 切り抜き後の入力） | 29.78（2b-G2 と同じ値） | — |

## GPU 付き pytest

- 176 件合格・0 件飛ばし・0 件失敗です。
- `-rs` を付けても、飛ばしの行は出ませんでした。GPU のテストは全部実行されたことになります。
- 以前の飛ばし 11 件の正確な内訳は、手元では照合していません。

## 1-G2 の項目ごとの判定（`verdict.pass_A_B_C_E_both_cases: true`）

| 項目 | first_last | last_only |
|---|---|---|
| A トークン列と印 | 合格（537） | 合格（289） |
| B pixel_values／grid | 合格（[1920,1536]・[[1,24,40]×2]） | 合格（[960,1536]） |
| B 画像読み取り部の出力 | 合格（[480,5120]） | 合格（[240,5120]） |
| B DeepStack | 合格 | 合格 |
| C 配線の差 | 0.0（ビット一致） | 0.0（ビット一致） |
| E 並び・AdaLN の印・時刻の番号（20 ステップ）・RoPE（float64） | 合格（差 0。時刻 537.0／742.0） | 合格（差 0。時刻 493.99999999999994） |
| D VAE（記録だけ） | ビット一致ではない。最大差 1.52e-2・相対 1.45e-3 | ビット一致ではない。最大差 1.63e-2・相対 1.48e-3 |

- この実行全体の VRAM の山は、予約 4,752MB でした。

## 予想と違ったこと・気づいたこと

1. **VRAM 16GB mode の先頭 I2V の指紋が、off と全桁一致しました**（`77e29687…`）。手順書では「一致するかは未知」としていた点です。
2. **vram16 on のとき、生成全体の VRAM の山を決めるのは VAE の復号です。** 値は 5,065MB で、全条件で同じでした。ノイズ除去の山は、2 枚と 10 枚の差が 1MB しかありません（3,858→3,859）。10 枚にしても VRAM の山は増えず、増えたのは時間でした（ノイズ除去 129.1→163.9 秒・+27%）。
3. **1-G2 の C の量子化の差（記録だけの値）が、2 枚のときだけ大きく出ました。** int8 と bf16 の相対差です。

   | 範囲 | first_last | last_only |
   |---|---|---|
   | 全行 | 0.203 | 0.015 |
   | 画像の行 | 0.277 | 0.010 |
   | テキストの行 | 0.022 | 0.021 |

   - 合否の項目（配線の差 0）とは別の量です。
   - ただし、2 枚目の画像の行で int8 の誤差が大きくなる何かがありそうです。監督の判断材料として挙げます。
   - 画像の出力（B）は、正解データと一致しています。
4. T2V の出力と kf10 の画像の MSE は、全位置で 0 でした。試験画像を同じ動画から取り出したことの裏づけです。
5. 標準出力の写しにある `NativeCommandError`／`RemoteException` の行は、PowerShell が標準エラーを包んだものです。終了コードはすべて 0 で、エラーではありません。ログには traceback も OOM もありませんでした。
6. API 経由の台本の終了コードは、予想どおり 1 でした。
   - 偽になったのは `constraints_match_plan` と `i2v_crop_path` の 2 つだけで、残りの 12 項目は真です。
   - 切り抜きの経路は、目で確かめました。`image_encode.source_size` が `[[384,384]]`、`canvas` が `[640,384]` です。
   - API の 3 本とも `vram16_mode: true`／`vram16_mode_used: on` でした（metadata.json）。
7. バックエンドの裏の作業は「終了コード 255 で失敗」と表示されました。私が止めたためで、問題ではありません。

## 実行したコマンド（作業場所 `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni`）

```powershell
# 0
$env:H3_GPU_TESTS='1'; .venv-engine-h3\Scripts\python.exe scripts\run_h3_pytest.py tests\test_h3_engine.py tests\test_h3_media.py tests\test_h3_format.py -rs *> outputs\h3-keyframes-2026-10-19\pytest_gpu.log
# 1〜7（各回の前に nvidia-smi と python プロセス数を <name>.gpu_before.txt に記録。scratchpad の kf197\gates\run_one.ps1 が実行）
cmd /c outputs\h3-keyframes-2026-10-19\<name>.cmd *> outputs\h3-keyframes-2026-10-19\<name>.stdout
#   <name> = g2_keyframe_parts, g3_t2v, g3_i2v_first, g3_i2v_first_vram16, g4_first_last, g4_last_only, g4_kf10
# PSNR（CPU）
.venv-engine-h3\Scripts\python.exe -I outputs\h3-keyframes-2026-10-19\psnr_keyframes.py <mp4> --ref 0=...kf_first.png --ref last=...kf_last.png --json <name>.psnr.json
#   g4_first_last, g4_last_only, g3_i2v_first, g3_i2v_first_vram16, outputs\h3-gate2\g4_J1_i2v.mp4（→baseline_J1.psnr.json）
#   g4_kf10 と g3_t2v は --ref N=kf10_NNN.png を 10 個（123 は last）
# 8 API 経由
powershell -NoProfile -ExecutionPolicy Bypass -File "S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\run.ps1"   # 裏で起動。PID 59764（run.ps1 52292 の子）
& .venv\Scripts\python.exe Docs\Outputs-archive\h3-gate2\run_2b_g2.py --out outputs\h3-keyframes-2026-10-19\g3_api
Stop-Process -Id 59764     # ポート 18620 の持ち主が自分の起動したものであることを確かめてから止めた
Copy-Item outputs\h3-keyframes-2026-10-19\g3_api\state.json.before state.json   # 中身は元から同じ（ハッシュ一致）
```

## 機械の最終状態

- Python のプロセスは 0 個です。ポート 18620 は待ち受けていません。GPU の使用は 713MiB・使用率 0% です。
- `state.json` のハッシュ `60AF82F0…` は、実行前と同じです。

## 成果物

`S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\h3-keyframes-2026-10-19\` に置きました。

- 結果: `GATES_RESULT.md`・`gates_result.json`
- pytest: `pytest_gpu.log`
- 1-G2: `g2_keyframe_parts.json`・`.log`・`.stdout`・`.exit.txt`・`.gpu_before.txt`
- セルフテスト 6 本: それぞれの `.mp4`・`.json`・`.stdout`・`.exit.txt`・`.gpu_before.txt`。名前は `g3_t2v`・`g3_i2v_first`・`g3_i2v_first_vram16`・`g4_first_last`・`g4_last_only`・`g4_kf10` です。
- PSNR: `g4_first_last.psnr.json`・`g4_last_only.psnr.json`・`g4_kf10.psnr.json`・`g3_i2v_first.psnr.json`・`g3_i2v_first_vram16.psnr.json`・`baseline_J1.psnr.json`・`g3_t2v_vs_kf10.psnr.json`
- API 経由: `g3_api.stdout`・`g3_api.exit.txt`・`g3_api.gpu_before.txt`、`g3_api\` フォルダ
  - `g2b_api.json`
  - `t2v1_output.mp4`・`t2v2_output.mp4`・`i2v_output.mp4` と、それぞれの `*_metadata.json`
  - `i2v_canvas_reference.png`・`i2v_first_frame.png`
  - `server.log.part`・`h3_worker.log.part`・`state.json.before`

scratchpad（`C:\Users\PHENOT~1\AppData\Local\Temp\claude\s--OriginalApps-12-Nz-LTX23-AviUtl2\6080e139-5d66-42d3-a48f-feb0e61dd152\scratchpad\kf197\gates\`）:

- `run_one.ps1`・`summ.py`
- `backend_console.log`

フォルダの外に、製品が自動で書いたもの（git の管理外）:

- `logs\h3_worker.log`・`logs\server.log`（追記）
- API のジョブのフォルダ 3 つ:
  - `outputs\94c132d9-5554-4bd9-bfef-6859244ad34e`
  - `outputs\391e6680-2502-4712-b995-98a19d10e31f`
  - `outputs\1d8eec07-f65d-4512-b0b6-7a43dba3c809`
- アップロードした画像 `uploads\807ba554-71a6-4f21-af87-b927fb8df90d\`

これらを消すかどうかは監督の判断に任せます。写しは `g3_api\` にあります。
