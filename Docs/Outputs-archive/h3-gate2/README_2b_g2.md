# 2b-G2（API 経由の実生成）の手順書

計画 `Docs/Outputs-archive/h3-stage2-reports-2026-10-17/PLAN_stage2.md` §6 の 2b-G2 を、オーナーの了承の後に監督が実行するための手順。**GPU を使う。了承の前には何もしない。**

## 結論（先に）

- 準備時点（2026-10-18 00:40 頃）の観測: **バックエンドは動いていない**（ポート 18620 に待ち受け無し・`python.exe` のプロセス無し。GPU の使用は 1,703MiB＝画面表示ぶん）。止める作業は不要。
- 手順は「バックエンドを起動 → 台本を 1 本実行 → バックエンドを止める → `state.json` を戻す」の 4 つ。

> 実施の記録（2026-10-18）: 起動はオーナーの `run.bat`。終了後もサーバーは止めず、`state.json` は戻していない（`Docs/VERIFICATION_LOG.md` §166.4）。

- 所要時間の見込みは**合計 12〜20 分**（内訳は下の「所要時間」）。

## 0. 事前の確認（読むだけ）

```powershell
Get-NetTCPConnection -LocalPort 18620 -ErrorAction SilentlyContinue   # 何も出なければ未起動
Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Select-Object ProcessId,CommandLine
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

- 既にバックエンドが動いていたら: ジョブが走っていない（`GET /status` の `queue.running` が 0）ならそのまま使ってよい。ただし**起動後に `engine_h3/` やアダプタのコードが変わっていれば、古いコードで動いている**ので、オーナーに断って止めてから起動し直す（止め方は §3）。
- AviUtl2 の操作パネルが同じバックエンドに繋がっていると、ジョブの取り合い（409 JOB_BUSY）が起きうる。実行中は操作パネルから生成しない。
- 主メモリ: 常駐中（T2V 2 本と I2V の間）はワーカーのコミット量が約 36GiB、瞬間の山は約 43GiB（2a-G4 の実測）。他の重いアプリは閉じておく。
- 設定はそのまま使える: `config.yaml` は `model.backend: "auto"`・`model.engine_python_h3` が `.venv-engine-h3\Scripts\python.exe` を指す・ポート 18620。C3 の報告で `H3Runner._real_available()` は実ファイルで `True`（mock に落ちない）。

## 1. バックエンドの起動

`run.bat` は最後に `pause` するので、裏で動かすときは `run.ps1` を直接呼ぶ（ffmpeg の PATH 追加なども run.ps1 がやる）。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\run.ps1"
```

（Claude Code からは Bash／PowerShell ツールの `run_in_background` で上の 1 行を動かす。）

起動の確認（数秒〜十数秒。起動時にはモデルを読み込まない＝GPU はまだ使わない）:

```powershell
(Invoke-RestMethod http://127.0.0.1:18620/api/v1/status) | Select-Object server,state,base_model,pipeline_loaded
```

`server: running`・`state: unloaded`・`base_model: LTX23` が出れば準備完了（`base_model` はこの時点では LTX23 のままで正しい。台本が H3 へ切り替える）。

## 2. 台本の実行（1 コマンド）

```powershell
& "S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\.venv\Scripts\python.exe" "C:\Users\PHENOT~1\AppData\Local\Temp\claude\s--OriginalApps-12-Nz-LTX23-AviUtl2\93b90569-6ea3-4591-8f76-72824574064d\scratchpad\h3\g2b\run_2b_g2.py"
```

台本がやること（MCP サーバーを stdio の子プロセスとして起動し、`mcp` パッケージのクライアントで叩く。VERIFICATION_LOG §39.5／§39.6 と同じ経路）:

| 段 | ツール | 確かめること |
|---|---|---|
| 0 | `backend_status` | 繋がる・ジョブが走っていない。`state.json` を結果フォルダへ退避 |
| 1 | `list_models` | `MiniMaxH3` があり `installed`。`constraints` が計画 §3.3 の JSON と一致 |
| 2 | `load_pipeline(base_model="MiniMaxH3")` | 事前検査（`inspect_h3_dit`／`inspect_h3_te`）が実ファイルで通り、`state: ready`。`server.log` に `Loading pipeline (REAL engine_h3 worker)` が出る（mock に落ちていない） |
| 3 | `submit_generate` T2V 1 本目 | `keep_resident: true`・`embed_mp4_metadata: false`（指紋比較のため）。DiT と VAE はファイルから |
| 4 | `submit_generate` T2V 2 本目 | 同じ条件。**DiT と VAE が `resident`**（読み込みが飛ぶ）・1 本目より速い |
| 5 | `upload_image` → `submit_generate` I2V | 384×384 の画像を 640×384 のキャンバスへ（覆うように拡縮して中央で切り抜く経路）。`keep_resident` は送らない（保持中のものを使い、終わりに解放）。`embed_mp4_metadata` は既定のまま→`get_mp4_info` で読み戻す |
| 6 | （ローカル） | 各ジョブの `metadata.json` から表を作る（`h3`・ステップ数と shift の要求と `done` のこだま・`keep_resident_used`・`images_used`・秒数・VRAM・コミット量） |
| 7 | （ローカル） | T2V 2 本の mp4 の sha256 を基準 `0da6111c…` と比べる（**違っても止めずに記録**） |
| 8 | （ローカル） | I2V の先頭フレームと、切り抜き後の入力画像の PSNR |
| 9 | `unload_pipeline` | H3 のワーカーを止める（`--keep-loaded` で省略可） |

共通の条件: 640×384・124 フレーム・24fps・ステップ 20・shift 12／3・seed 42・sdpa・先読み on（API の既定）・プロンプトは 2a-G4／第 1 弾 G5 と同じもの。

合否（台本の終了コード 0＝全部 true）: `verdict` の 14 項目（H3 が導入済み・制約が一致・読み込みが ready・実ワーカー・3 本とも `backend: minimax-h3`・`metadata.json` に `h3`・要求に 20／12／3・`done` に 20／12／3・`keep_resident_used` が on／on／off・1 本目はファイルから・2 本目は読み込み省略・2 本目が速い・I2V が 1 枚使い切り抜き経路を通った）。`recorded` は判断材料（指紋・PSNR・mp4 の埋め込み・秒数・VRAM・コミット量）。

### 置き場

`S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\h3-gate2\g2b_api\`

- `g2b_api.json`（全記録。`verdict`・`recorded`・各ジョブの表と最終状態）
- `t2v1_output.mp4`・`t2v2_output.mp4`・`i2v_output.mp4` と各 `*_metadata.json`（ジョブのフォルダ `outputs\<job_id>\` からの写し）
- `i2v_canvas_reference.png`（切り抜き後の入力）・`i2v_first_frame.png`（I2V の先頭フレーム）
- `server.log.part`・`h3_worker.log.part`（台本の実行中に増えたログの部分だけ）
- `state.json.before`（実行前の `state.json` の写し）

## 3. 後片付け

1. 台本が最後に `unload_pipeline` を呼んでワーカーを止めている。残っていないか確認:
   ```powershell
   Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object CommandLine -match 'engine_h3' | Select-Object ProcessId,CommandLine
   ```
2. バックエンドを止める（ポートの持ち主を止める。裏で動かした run.ps1 の場合は、そのバックグラウンドの作業を止めてもよい）:
   ```powershell
   $p = (Get-NetTCPConnection -LocalPort 18620 -State Listen).OwningProcess; Stop-Process -Id $p -Confirm:$false
   ```
   ワーカーが残っていた場合（手順 1 で出た場合）は、そのプロセスも止める。**ワーカーを残したまま親だけ止めると、主メモリを掴んだままになる。**
3. **`state.json` を戻す**: 読み込みで選択中のベースモデルが `MiniMaxH3` に変わり、`state.json` に保存されている（再起動しても H3 のまま）。オーナーの選択（準備時点は `LTX23`）へ戻すため、バックエンドを止めた後で写しを戻す:
   ```powershell
   Copy-Item "S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\h3-gate2\g2b_api\state.json.before" "S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\state.json"
   ```
   （止める前に戻すと、動いているバックエンドが上書きしうる。）
4. 生成されたジョブのフォルダ `outputs\<job_id>\`（3 つ）は残る。写しは `g2b_api\` にあるので、消すかどうかは監督の判断。ジョブの記録はメモリ上だけなので、バックエンドを止めれば一覧からは消える。アップロードした画像は `uploads\` に残る（一時置き場）。

## 所要時間の見込み

| 段 | 見込み | 根拠 |
|---|---|---|
| 起動 | 10〜30 秒 | 起動時はモデルを読まない |
| 読み込み（段 2） | 10〜60 秒 | ワーカーの起動と import だけ。重みは各ジョブの中で読む（2a-G4 で load 直後の作業セット 0.59GiB） |
| T2V 1 本目 | 約 160 秒（ディスクのキャッシュが冷えていれば 3〜4 分） | 2a-G4 の J0 が 158 秒（TE 24GiB と DiT 19.5GiB をファイルから読む） |
| T2V 2 本目 | 約 135〜145 秒 | 2a-G4 の J4 が 135 秒（常駐で読み込み省略） |
| I2V | 約 250 秒 | 2a-G4 の J3 が 248 秒（条件行のぶん denoise が長い） |
| 合計 | **約 12 分（最大 20 分程度）** | 上の和＋ポーリングの間隔（10 秒） |

台本のジョブごとの打ち切りは 30 分（`--job-timeout`）、読み込みの打ち切りは 10 分（`--load-timeout`）。

## 失敗したときに見る場所

- 台本の出力（標準出力の進捗行）と `g2b_api.json`（途中までの記録も毎回書き出す）。
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\logs\server.log`（アプリ側。読み込みの拒否、`_real_available` が False のときの欠けたパス、ジョブの開始と終了）。
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\logs\h3_worker.log`（H3 のワーカー。生成中の例外・メモリ・各段の秒数）。
- 結果フォルダの `server.log.part`／`h3_worker.log.part`（最後まで進んだ場合のみ書き出す）。
- よくある失敗:
  - `backend not reachable` → 起動していない／ポート違い。
  - `backend is 'mock-h3'` → アプリが mock に落ちた。`server.log` に欠けたファイルのパスが出る。台本はそこで止まる。
  - `MODEL_INCOMPATIBLE` → 事前検査が実ファイルを拒否した（理由は日本語で detail に入る）。
  - `JOB_BUSY` → 操作パネルなど別の経路からジョブが入っている。
  - 指紋が `0da6111c…` と違う → 止めずに記録される。考えられる差: アプリは load で `deterministic: true` を送るが、2a-G4 の台本は `false` で走っていた（cuDNN のアルゴリズム選択が変わりうる）。T2V 1 本目と 2 本目が互いに一致するかも `recorded.t2v1_equals_t2v2` に出る。

## 補足

- 試験画像 `i2v_input_square.png`（同じフォルダ）は `outputs\h3-gate2\i2v_input.png`（640×384）の中央を 384×384 で切り抜いたもの。依頼の例の 480×480 は元画像の高さ（384）を超えるので取れず、正方形の最大の 384×384 にした。キャンバス 640×384 に対しては、640×640 へ拡大してから上下を切る経路を通る。
- 台本はリポジトリのコードを import しない（切り抜きの幾何は `engine_h3/image_cond.load_canvas_image` を写したもの。アプリの venv に torch と av が無いため）。先頭フレームの取り出しは `tools\ffmpeg\bin\ffmpeg.exe`（無ければ PATH の ffmpeg）。
- PSNR は 2a-G4 では 30.3dB（640×384 の入力をそのまま）。今回は拡大を挟むので値は変わりうる。合否には入れず記録だけ。
