> **このファイルは実機の `outputs/comfort-calib-2026-09-25/progress.md`（git追跡外）のスナップショットである（2026-09-25複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# progress — comfort-calib-2026-09-25（§3-165 広い窓の確認較正）

- 00:30 サーバー起動（run.ps1 を保護なし PowerShell から切り離し起動。wrapper PID 8800・uvicorn PID 572・ポート 18620）。`server start --pid 8800` で登録。
- 00:31 422 プローブで API が 16 名（standard/high_resolution/full_length/w25..w61）を受け付けることを確認（GPU 未使用）。
- 00:32 `idle-calib --minutes 3`（LTX23・Sulphur-2 Q6_K）開始。
- 00:36 idle-calib 完了（exit 0・shared_plateau_limit_mb 367.7）。Tier 1 開始: plan_T1_ltx23_inner.json（6 ラン・--job-timeout 2400）。
- 00:52 a_b1216 完了（tiles 5・stage2 131 s・共有中央値 1155 MB＝基準）。a_w46 完了（tiles 3・stage2 256 s・共有中央値 2101 MB・Δ+945 MB → **stage2 窓で退避**・restore Δ−0.1）。仮説: Q6_K では潜在 46 のタイルが解像度に依らず退避する（§113 の孤立点も潜在 46）。a_w46_r2・a_w61 の結果で追加診断点を判断。
- 00:54 計画外の追加診断（仮説「Q6_K では潜在 46 のタイルが解像度に依らず退避する」の切り分け）用に plan_T1x_ltx23_diag.json（a_w40・a_w43・a_b1088・a_w46lo）と plan_T1q_ltx23_q4.json（a4_b1216・a4_w46＝Sulphur-2 Q4_K_M 対照）を生成・preflight ok。実行するかは a_w46_r2・a_w61 の結果を見て判断（preflight.json は最後の実行分で上書きされる点に注意）。
- 01:03 a_w46_r2 完了（tiles 3・stage2 254 s・Δ+915 MB → 退避・2 回一致で再現）。restore Δ−1.9。→ Tier 1 終了後に plan_T1x_ltx23_diag.json（w40・w43・1088×576 の基準点と w46）を実行し、Tier 3a → Q4_K_M 対照（plan_T1q）→ 2.5 default 切替 → Tier 2・3b → 2 段階復元の順とする。aux2 ビルド／配置は計測終了後。
- 01:13 a_b1152 完了（tiles 5・stage2 95 s＝基準）。a_w61 完了（tiles 2・stage2 92 s・Δ+16 MB → **快適**・restore Δ−0.1）。線の 99% の w61 が快適で 96% の w46 が退避 → 潜在 46 固有の仮説を支持。
- 01:19 Tier 1 完了（6 ラン・exit 0）。a_w61_r2 快適（Δ−0.2）。内側 4 点の結果: w46@1216×704 退避×2／w61@1152×576 快適×2。冷却 150 秒後に T1x（plan_T1x_ltx23_diag.json・4 ラン）開始。
- 01:46 T1x 完了（4 ラン・exit 0）: a_w40@1216×704（33,440・84%）快適 Δ−6.8／a_w43（35,948・90%）快適 Δ−14.7（専有ピーク 15,541 MB＝ぎりぎり）／a_b1088 基準／a_w46lo@1088×576（28,152・70%）**快適** Δ−15.8（専有ピーク 12,795）。→ 仮説「潜在 46 固有」は**棄却**。解釈: 1216×704 では w43→w46 の +2,500 トークンで専有 VRAM の天井（約 15.5 GB）を越え約 1 GB が共有へ退避。1152×576 の w61（39,528）が快適なのは全長バッファ（解像度×総尺 705f）が軽いため。→ 線は解像度・総尺・GGUF サイズ（Q6_K）に依存する余裕の問題。T1q（Q4_K_M 対照）が要。冷却 150 秒後に T3a 開始。
- 02:03 T3a 完了（exit 0）: a_b1280 基準（w22・専有ピーク 11,120）／a_o46@1280×768（44,160・110%）Stage-2 Δ+201 MB・共有ピーク 2,494・PCIe 受信中央値 343 MB/s・専有ピーク 15,615 → 規則 v3 では快適（+300 未満）。1216×704 の w46（96%）が退避×2で 1280×768 の w46（110%）が通る＝上限付近は非単調。→ T1q（Q4_K_M 対照）開始: load → idle-calib 3 分 → 2 ラン。
- 02:19 T1q 完了（exit 0）: Q4_K_M へ切替（load 成功・idle-calib 3 分）。a4_b1216 基準（専有ピーク 9,805・共有底 627 MB＝Q6_K の 1,150 より約 500 低い）／**a4_w46@1216×704（38,456・96%）快適 Δ+8.3 MB・専有ピーク 13,839・stage2 127 s**（Q6_K は 15,484 で退避・256 s）。→ w46 の退避は Q6_K 固有（重み分約 1.7 GB で余裕が削られる）。Q4_K_M 系で較正された配信値の線はこの点で成立。→ LTX25 default へ切替 → idle-calib → T2（6 ラン）→ 冷却 150 秒 → T3b（1 ラン）を一続きで開始。
- 03:13 T2 完了（6 ラン）: b_w46@1280×768（44,160・98%）快適×2（Δ−49.5／−32.9・専有ピーク 15.3 GB）／b_w61@1152×640（43,920・98%）快適×2（Δ−33.6／−33.2）。T3b: b_o52@1280×768（49,920・111%）**退避** Δ+1,551 MB・stage2 298 s（基準 147 s）→ 線は効く。全 21 ラン完了。→ 2 段階復元開始。
- 03:16 2 段階復元 成功（LTX25→REDGraft Q6_K 7.1 s／LTX23→Sulphur-2 Q6_K 6.8 s・state.json は original_state.json と完全一致・active LTX23）。calib.py report・make_results.py（PYTHONUTF8=1 が必要。cp932 では失敗）出力済み。GPU が空いたので Retake w61（481f）の REST 実機テスト（素材＝a_w61 の出力 1152×576・705f）を開始。文書 D 第 2 回（較正結果の記入・README・RESULTS・Outputs-archive）を起動。
- 03:21 Retake w61 REST 実機テスト（計画 §7 目視 5 の機械部分）: 素材＝a_w61 出力（1152×576・705f）・開始 1.0 s・窓 481f・stage2_window=w61 → **completed**（330.2 s・peak_vram_mb 12,169・metadata の chain.stage2_window=w61）。出力 outputs/0a15b0f4-20e1-41f4-90c5-4b830f90dc5a/output.mp4（目視用に残置）。一括アップサンプル（chunked_upsample 省略）のまま完走。
- 03:22 機械ゲート（最終）: pytest 全件 2,750・失敗 1（既知の環境要因 test_mcp_registration・サーバー稼働中）・スキップ 49／typecheck 緑／vitest 149 ファイル 3,052 件全合格（機械が空いた状態。負荷下では別々の 1 件が揺れた）／lint 警告 31（変更前と同数）。aux2 ビルド開始（build.ps1 -Config Release -RunTests）。
- 03:23 aux2 ビルド成功（build.ps1 -Config Release -RunTests・doctest 384 件合格・埋め込み Web UI dist-single 710.95 kB）→ deploy.ps1 で AviUtl2 Plugin フォルダ・配布コピー・Language ファイルへ配置。3 箇所の SHA-256 一致（先頭 00BB69CD70D31B7F・1,459,712 バイト）。AviUtl2 は未起動（ロックなし）。
