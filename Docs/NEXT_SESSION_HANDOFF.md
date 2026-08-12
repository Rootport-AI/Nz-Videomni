# 次セッション引き継ぎ書 — LTX 2.3 を 16GB で動かす

---

## ▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-08-12 Single a2vの全長stage-2化＝§1-19＝**実装・機械検証・デプロイ完了。残るはオーナー実機ゲート（G-B1〜G-B7）とクローズ処理のみ**）（**本ブロックが日付として最新**）

> **本ブロックが「日付として最新」の座を継ぐ。** 以下の▶節（2026-08-04ブロック以降）はすべて歴史記録として残す。2026-08-04以降にも複数のテーマ（VSFの残課題整理・骨格常駐トグル・PrunaVAED・stage-2窓プリセット`high_resolution`〔§53〕・Outpainting〔§54〕・Retake〔§55〕・長尺A2V〔§56〕・長尺IC-LoRA〔§57〕等）が完結しているが、本ブロックはそれらを遡って書き足すものではない。**各テーマの最新状態は必ず[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md)の該当節（節末尾の『状態』表記が正）を参照すること。**
>
> **今回完了した内容**: SingleタブのA2V（音声から動画を生成する機能）は内部的に1クリップのチェーンとして実行されるため、これまでstage-2（アップスケール工程）が潜在22フレームの固定窓によるタイル処理になっていた。その結果、隠れたつなぎ目・チェーンと同じ約1.86メガピクセルの解像度上限・警告や切替手段の不在という3点の乖離が生じていた（設計根拠の正本は[`CHAIN_STAGE2_RESEARCH_NOTES.md`](CHAIN_STAGE2_RESEARCH_NOTES.md) §7の棲み分け原則）。これを解消するため、stage-2窓プリセットへ**`"full_length"`（潜在61フレーム＝481フレーム相当、前進61、のり代0、タイル数1）**を追加し、SingleとBatchのA2Vが常にこの全長窓で動くようにした。エンジン（GPU側コード）は差分ゼロで、`chain_math.py`のプリセット追加・`api/models.py`のバリデーション拡張・`gradio_ui/handlers.py`のペイロード追随・フロントエンドの型とビルダー改修（合計数十〜数百行）で実現した。
>
> **実装・機械検証・デプロイはすべて完了している。** バックエンドpytest 1185 passed/20 skipped、フロントエンドvitest 2261 passed/10 skipped・typecheckクリーン。`build.ps1`→`deploy.ps1`でビルド・配置済みで、実機とバックエンドリポジトリ配布コピーのSHA-256は3値一致（`4C9B0EE915AFA7F5B82AEFBF8897A5D74EE05263178A834FF79A01E9A5227B2C`。2026-08-12のセンタリングのデプロイで更新済み。現行値はフロントエンド[`DEVLOG.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/DEVLOG.md) §75.2）。**コミットは未実施。**
>
> **残っているのはオーナーによる実機ゲート（G-B1〜G-B7）だけ**である。詳細は台帳フロントエンド[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §2-1、実装・機械検証・kt_a負値やトークン予算の技術記録の正本は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §58、設計根拠の研究ノートは[`CHAIN_STAGE2_RESEARCH_NOTES.md`](CHAIN_STAGE2_RESEARCH_NOTES.md) 8節。
>
> 次セッションへの引き継ぎ事項:
>
> 1. **実機ゲートG-B1〜G-B6は必須、G-B7は任意。** 960×576/481f・1280×768/257f・1280×768/481f（対照）・1920×1088/153fの完走とVRAM同等性確認（G-B1〜G-B4）、512×320/481fでのつなぎ目ゼロの目視確認（G-B5）、改修前生成物との比較によるつなぎ目消失の判定（G-B6）、任意で12fps×481fの音声破綻観測（G-B7）。合格基準の詳細は台帳[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §2-1を参照。
> 2. **実測が`config.yaml`の`spill_free_frames`テーブル（182〜187行）とずれた場合のみ**、同テーブルを実測値へ更新する追加作業が発生する。A2Vは音声VAE分がVRAMに乗るため、快適上限が下がる可能性が最有力の追加作業として想定されている。
> 3. **オーナーの実機ゲートが全項目合格したら**、台帳§1-19を[`PENDING_TASKS_CLOSED.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md)へクローズ移設し、§2-1のチェックリストを見出しごと削除する（長尺IC-LoRA〔§3-78〕・長尺A2V〔§3-74〕と同じクローズ作法）。
> 4. **コミットが未実施。** 次セッションでオーナーの承認を得てコミット・プッシュを行うこと。
> 5. **同日、別セッションでOutpaintingのセンタリング（描き足す量を対称に保つチェックボックス）を実装した。** フロントエンドのみの改修でバックエンドは無改修。実装・機械検証・**デプロイ・コミット済み**であり、残るのはオーナー目視ゲートG-C1〜G-C5のみ（本件G-B1〜G-B7とは別）。詳細はフロントエンド[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §2-2、[`DEVLOG.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/DEVLOG.md) §75を参照。**本ブロック記載のSHA-256（`4C9B0EE9...`）は§74時点の値であり、センタリングのデプロイ後（新値`E13EC924...`）は一致しない。**

---

## ▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-08-04＝本ブロックは日付として最新ではない。2026-08-12ブロック参照）

> **最新は §51 GGUF逆量子化の1カーネル化（`fused_gguf_dequant_kernel`）。** Q4_K・Q5_K・Q6_Kの逆量子化をTritonカーネル3本へ融合したもので、**実装完了・機械検証全PASS・実機ゲートG1〜G8全項目合格・既定on**（オーナー承認済み）。生成結果はビット単位で不変。実測は**25.28秒短縮（約17.5%）**。正本は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §51、台帳はフロントエンド[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §1-11。
>
> **§49 IC-LoRA Depth（深度制御）・Deblur（ぼけ除去）の追加はテーマ完結。** G3・G4ともオーナー実機で合格し、コミット・プッシュ・デプロイまで完了している（`e273052`〜`9701fbe`）。正本は [`ICLORA_DEPTH_DEBLUR_WORKORDER.md`](ICLORA_DEPTH_DEBLUR_WORKORDER.md)。

---

## ▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-08-03 時点のテーマ一覧）

> **2026-08-03 時点で走っている／終わったテーマは次の4つ。** いずれも正本は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) の該当節である。
>
> - **§44 先読み block swap**（テーマ完結・既定 ON）／**§45 `audio_strength`**（Style LoRA の音割れ対策・実機A/B合格でテーマ完結）／**§48 モデル骨格の常駐（`keep_resident`）**（本命の高速化はオーナー実機確認済み・残るのは細目のオーナー目視ゲート）。
> - **§49 IC-LoRA Depth（深度制御）・Deblur（ぼけ除去）の追加。実装完了・G0／G1／G2 全PASS・コミット＆プッシュ済み（backend `fd6d43f` / frontend `d375028`）。G3（オーナー実機 real）・G4（既存 canny/pose/upscaler の回帰）も2026-08-04に合格し、テーマ完結。正本は [`ICLORA_DEPTH_DEBLUR_WORKORDER.md`](ICLORA_DEPTH_DEBLUR_WORKORDER.md)。**
> - フロントエンド[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §1-10 だった骨格常駐トグルのprefetch連動グレーアウトは実装・デプロイ・コミット＆プッシュ済み（frontend `b39eaa0` / backend `e583c03`）で、2026-08-04にオーナー目視合格し[`PENDING_TASKS_CLOSED.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md) §3-62へ移設・クローズ済み。記録はフロントエンド[`DEVLOG.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/DEVLOG.md) §60。
>
> pytest の現在のベースラインは **910 passed / 9 skipped**（§49.5）。IC-LoRA は現在5エントリ（`pixel-spatial-upscaler-x2` / `canny-control` / `pose-control` / `depth-control` / `deblur`。うち union-control の1ファイルを3つの論理名で共用）。

---

## ▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-08-01 Acceleration（生成の高速化）機能＝SageAttention 2.2.0 のジョブ単位切替＋モック2項目＝**実装・機械検証・実機ゲート（G0〜G9）すべて完了・オーナー実機目視も合格**）（**生成機能の正本**）

> **（当時の記録）本ブロックは2026-08-01時点の記録である。日付として最新なのは冒頭の2026-08-04ブロック。**
>
> **生成機能についてはこのブロックが正本。以降の▶節（直下の2026-07-29 VSFブロック・2026-07-28 NAGブロック・2026-07-26配布・導入ブロックを除く）はすべて歴史記録。** 配布・導入まわりは下の「2026-07-26 α版インストール導線の整備」ブロックが引き続き正（そちらは本ブロックと独立に併走している）。旧「生成機能の正本」だった「2026-07-29 VSF」ブロックは本ブロックに置き換わった（NAG・VSFはいずれも非CFGネガティブプロンプトの方式として現役のまま）。
>
> 実装・機械検証・実機ゲートの正本は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §43。APIフィールドは [`../LTX23_Backend_Specification.md`](../LTX23_Backend_Specification.md) §6.2・§6.5b（v0.5.6）、利用者向け説明は [`../README.md`](../README.md)「生成の高速化（Acceleration）」節。台帳は [`../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md) §3-58（クローズ済み）。

### 本日完了した内容の要約

操作パネル（AviUtl2連携UI）とGradio UIの設定画面に「Acceleration（生成の高速化）」という区画を新設し、生成そのものを速くする切替を3項目ぶん置いた。**実装があるのは attention（注意機構）の実装選択だけ**で、`sdpa`（PyTorch標準・既定）と `sage`（SageAttention 2.2.0＝量子化を使って注意機構の計算を速くする外部カーネル）から選ぶ。残る2項目——fused GGUF dequant + GEMM（GGUFの逆量子化と行列積の融合）と PruneVAED（枝刈り版のVAEデコーダ）——は将来の実装枠として置いた**モック（受理はするが効果が無い）**で、UI上は常にグレーアウトし、workerへのペイロードにも `GET /status` にも載せていない。

切替の単位はジョブで、サーバーの再起動もパイプラインの再読み込みも要らない（リクエストの `attention_backend` フィールド、既定 `"sdpa"`）。MCPにも同じ1項目だけを公開した（モック2件は非公開）。エンジン側は新規の `engine/transformer/sage_attention_service.py` が48ブロック×6種＝**288モジュール**の注意機構を差し替え、マスク付きの呼び出し（IC-LoRAで `conditioning_attention_strength<1.0` を使う経路）はSDPAへ自動で戻す。カーネルが例外を出した場合はそのジョブ内で `sdpa` に固定する。**実際に使われた方式は `outputs/{job_id}/metadata.json` の `attention_used`（`"sdpa"` / `"sage"` / `"sage->sdpa"`）に記録される**——過去に `fp8_transformer` が「表示はあるが挙動を変えない」状態で残った反省から入れた仕組みで、実機ゲートの合否もログではなくこの値で判定した。

**既定を `sdpa` のまま据え置いたのは意図的な判断**である。`sage` は数値精度が異なるため、同じシードを指定しても生成結果の細部が変わる（構図は同じで、質感やノイズの出方が変わる）。アップデートで利用者の生成結果を黙って変えないこと、`sdpa` を常に正しい参照実装として残すこと、切替が1クリックで済み選択も保存されることの3点による。

依存としては `sageattention` 2.2.0（woct0rdho 版のWindows用ビルド済み wheel を直リンクで固定）と `triton-windows` 3.5.1.post24 をエンジン用仮想環境の標準同梱に戻した。**2026-07-28（§40）に削除した依存の再導入だが、当時の判断と矛盾しない**——削除したのは旧世代1.0.6の死重依存で、今回は実際の消費者があるため（詳細は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §43.3）。`-ResolveLatest` を使った環境では動作保証外である旨をコメントに明記してある。

機械検証はバックエンドの pytest **817 passed / 6 skipped**（従来776件＋新規41本）、フロントエンドの型検査0エラー・vitest **1585 passed / 10 skipped**（+40本）、ネイティブの doctest **267ケース全PASS**、エンジン用仮想環境の `sage_selfcheck` **3/3 PASS**。実機ゲートはG0〜G9の全項目が合格し、**2026-08-01のオーナー実機確認（i2v＋NAG、1344×1728・153フレームで 460.63秒 → 366.85秒＝1.26倍）で目視ゲートも合格**した。720pの対比較では平均1.167倍・VRAM差0.08%以内。

> **速度を測り直すときの注意（重要）**: workerプロセスの**初回ジョブだけ約8%速い**という再現性のある挙動があるため、単純に「1本目に `sdpa`、2本目に `sage`」を流して比べると1.09倍程度にしか見えず**偽のFAIL**になる。`sdpa` と `sage` を交互に流す対比較を行うこと。あわせて `sage` は初回に triton のJITコンパイルが走るため、計測は2ジョブ目以降で行う。詳細は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §43.6。

### 次セッションの残課題（2026-08-01起票）

1. **コミットはオーナー指示待ち。** 本ブロック作成時点で未コミット。
2. **モック2項目の実装**（fused GGUF dequant + GEMM／PruneVAED）は将来課題として起票済み。fused GGUF dequant + GEMM は[`../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md) **§3-61（no-go再確認のうえクローズ済み）**で、後継の「逆量子化の1カーネル化」が[`../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) **§1-11＝実装完了**。PruneVAED は同 §3-50。
3. **`sage` を既定にするかどうかの再検討**は、フィールドでの安定実績が溜まってからの判断事項として起票済み（同 §4-22）。現状は再現性を優先して `sdpa` 既定。
4. **無関係だが紛らわしい既知事象**: `GET /status` の `gpu` ブロックは常に `available: false` を返す。アプリ用仮想環境にCUDA版torchを入れない2プロセス構成に由来する既存の挙動で、今回の改修とは無関係（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §43.8）。

---

## ▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-07-29 VSF（Value Sign Flip）非CFGネガティブプロンプト第2方式＝**VSFテーマ全クローズ（フロントエンド目視ゲート含む）・残課題なし**）（歴史記録・非CFGネガティブプロンプトについては引き続き正）

> **本ブロックは2026-08-01のAccelerationブロックに「生成機能の正本」の座を譲ったが、非CFGネガティブプロンプト（NAG／VSF）についての記述は引き続き正である。** 以降の▶節（本ブロック直下の2026-07-28 NAGブロック・2026-07-26配布・導入ブロックを除く）はすべて歴史記録。配布・導入まわりは下の「2026-07-26 α版インストール導線の整備」ブロックが引き続き正（そちらは本ブロックと独立に併走している）。旧「生成機能の正本」だった「2026-07-28 NAG」ブロックは本ブロックに置き換わった（NAG自体は非CFGネガの第1方式として現役のまま。VSFは第2方式の追加）。
>
> **なお、本ブロック本文に出てくる「pytest 763 passed/6 skipped」は2026-07-29時点の数字である。** 現在のベースラインは **910 passed / 9 skipped**（2026-08-03・IC-LoRA Depth／Deblur まで反映。[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §49.5）。
>
> **2026-07-29追記: フロントエンド（React／AviUtl2連携UI）の目視ゲートもオーナーが実機で全件合格と判定し、VSFテーマは実装・機械検証・実機ゲート・デバッグスイッチ縮退・フロントエンド追随・目視ゲートのすべてが完了、残課題なしで全クローズした。詳細は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §41.11、台帳のクローズ記録は[`../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md) §3-55。将来の再訪条件（コミュニティのVSF scaleベストプラクティス報告が出たとき、既定値1.5を見直すか検討）も同節に記載。**
>
> 正式なワークオーダーは本セッション実行時点でオーナーのプランファイル（`reactive-weaving-umbrella.md`、リポジトリ外）が正本で、実装・機械検証・実機ゲートチェックリストの正本は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §41。台帳は [`../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md) §3-55（クローズ済み）。

### 本日完了した内容の要約

NAG（§38）に続く2つ目の非CFGネガティブプロンプト手法として、VSF（Value Sign Flip, arXiv:2508.10931）を実装した。VSFは正負のコンテキストを連結して1回のattentionで処理し、負側のV（value）だけを−scale倍する方式で、NAGより排除力が強い一方、正プロンプト忠実度はNAGが上という性格違いのため、方式を選べるUI（`neg_method: "nag"|"vsf"`）にした。エンジン新規モジュール `engine/transformer/vsf_service.py`（連結1回attention・負側V×(−scale)・エンコード時実トークンスライス・AdaLN 3モード〔raw/modulated/v_scale〕をデバッグ用に切替可能なスタッシュ窓contextmanager・負側softmax質量mのINFOログ）を中心に、`nag_service.py`拡張・両パイプライン配線・`worker.py`分岐・API 3フィールド（`neg_method`/`vsf_scale`/`vsf_adaln`）・MCP `submit_generate`/`submit_chain`末尾3引数・Gradio UI（方式ラジオ・スライダー出し分け・デバッグアコーディオン）まで一気通貫で配線した。敵対的レビュー2ラウンド（計画時）＋コードレビュー1回（実装後）を実施し、指摘は全て修正済み。エンジンvenvのselfcheckが`vsf_selfcheck` 5/5・`nag_selfcheck`回帰6/6でPASS、アプリvenvのpytestが763 passed/6 skipped（既存テストの改修は計画どおり4本のみ）。**実機ゲートはMCP経由で親エージェント自身が実施し、G0回帰（単発T2V/I2V・chain OFF/chunked双方）・VSF疎通＋mログ実測・AdaLN 3モード・経路網羅（chain/A2V/NAG回帰）・NAG対VSF効き比較セット収集まで全て合格した（詳細は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §41.5）。目視評価用の成果物一式はHugging Faceの非公開datasetへアップロード済み: https://huggingface.co/datasets/Rootport/Nz-LTX23-vsf-eval-20260729 。**残るのはオーナーによる目視評価のみ**（効き具合・scale適正値・AdaLNモードの判断）。

途中、`.gitignore`の`tools/`パターンが`mcp_server/tools/`（MCPツール実装一式・別セッション由来）を不可視化していた致命的な欠陥を発見し`/tools/`へ修正した。**次回コミット時は`git add mcp_server/tools/`を忘れないこと。**

### 次セッションの残課題（2026-07-29起票・同日中に大半解消）

1. ~~**HFの動画をオーナーが目視して効き具合・scale適正値・AdaLNモードを判断すること。**~~ → 同日中に第1・第2ラウンドで完結。既定=raw・実用域scale1.5〜5（詳細は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §41.9）。
2. ~~**既定値確定後、デバッグスイッチ（AdaLNラジオ等）を縮退するかどうかは別途承認が必要。**~~ → オーナー決定により縮退第3弾として`vsf_adaln`を全レイヤーから撤去済み。VSF APIは`neg_method`＋`vsf_scale`（0〜10）の2フィールドに確定した（詳細は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §41.10）。実験の歴史はコミット`f2124e1`に保存。
3. **βノブ（`vsf_offset`）は未実装のまま。** mの実測（video約0.8〜1.2%・audio約0.5%、raw時）を踏まえてもαだけでは不足すると判明した場合に限り、第2弾として起票する。
4. **コミットはオーナー指示待ち。** `mcp_server/tools/`を含む`git add`が必要（上記gitignore修正参照）。
5. ~~**フロントエンド（React）追随は本Waveの範囲外・未着手のまま残っている。**~~ → 2026-07-29中に追随実装＋オーナー実機目視ゲートまで完結し、VSFテーマは全クローズした（詳細は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §41.11、台帳は[`../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md) §3-55）。残タスクはない。

---

## ▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-07-28 NAG（非CFGネガティブプロンプト）機能＝Wave 0〜3実装完了・機械検証（selfcheck 6/6・pytest 658 passed/6 skipped）全PASS・**実機ゲート全項目合格（2026-07-29 オーナー実機確認で完了）**）

> **生成機能についてはこのブロックが正本。以降の▶節（本ブロック直下の2026-07-26配布・導入ブロックを除く）はすべて歴史記録。** 配布・導入まわりは直下の「2026-07-26 α版インストール導線の整備」ブロックが引き続き正（そちらは本ブロックと独立に併走している）。旧「生成機能の正本」だった「2026-07-14 Clip Chain拡張」ブロックは本ブロックに置き換わった。
>
> 正式なワークオーダーは本セッション実行時点でリポジトリ外（オーナーのプランファイル）にあり、実装・機械検証・実機ゲートチェックリストの正本は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §38。API フィールドは [`../LTX23_Backend_Specification.md`](../LTX23_Backend_Specification.md) §6.2（v0.5.3）、利用者向け説明は [`../README.md`](../README.md)「非CFGネガティブプロンプト（NAG）」節。

### 本日完了した内容の要約

蒸留版 LTX 2.3 は CFG（`guidance_scale=1.0`）凍結によりネガティブプロンプトが no-op だった問題を、NAG（Normalized Attention Guidance）という非CFG手法で解消した。Wave 0（エンジンコア `engine/transformer/nag_service.py` + `nag_selfcheck.py`）→ Wave 1（`fast_video_pipeline.py`/`chain_pipeline.py`/`worker.py` の配線）→ Wave 2（`api/models.py`/`services/ltx_runner.py` のAPI・ペイロード）→ Wave 3（GUIの共有 Negative Prompt アコーディオン）を段階的に実装し、単発Generate・Clip Chain・バッチA2Vの全経路に配線済み。エンジンvenvのselfcheckが6/6 PASS、アプリvenvのpytestが658 passed/6 skipped（既存テストの改修はpositional `_chain_args`ヘルパとi18nキー一覧の2件のみ＝NAGが加算的拡張であることの裏付け）。**Wave 4（本ブロック）でドキュメント化まで完了し、実機ゲート（G0〜G7・V-UI）も2026-07-29のオーナー実機確認で全項目合格した（詳細は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §38.4）。**

### 次セッションの残課題（2026-07-28起票）

1. ~~**実機ゲートG0〜G7・V-UIが未実施。** [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §38.4 のゲート表を上から順に、オーナーの実機（RTX 4070 Ti SUPER 16GB）で実施すること。特にG0（NAG OFFでの回帰＝バイト一致）を最優先で通し、既存の生成経路を壊していないことを先に確定させる。~~ **（2026-07-29解消）** オーナーが実機確認を行い、G0〜G7・V-UIの全項目が合格した（詳細は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §38.4）。
2. **STG（Spatio-Temporal Guidance。既存の別ガイダンス機構）× NAG併用は未検証。** 両方を同時に有効化した場合の挙動・干渉の有無を確認していない。
3. **`compile_transformer` はNAGと非互換（現状は死コード）。** `torch.compile`はNAGのforwardパッチ（`block.attn2.forward`等の差し替え）と衝突する設計のため、`fast_video_pipeline.py`にコメントを追記して回避したが、`compile_transformer`自体は呼び出し元ゼロの死コードのまま（本番未使用）。将来これを復活させる場合はNAGとの共存方式を再設計する必要がある。
4. **ラジオ「Other」の実体実装は将来課題。** 現状は「NAG / Other」ラジオでOtherを選ぶと即座にNAGへ戻すフォールバックのみで、Other（NAG以外の非CFG手法）自体の実装は無い。
5. **クリップ毎の個別ネガティブは範囲外。** 現状のnegative_promptはチェーン全体で1本のみ（`GenerateChainRequest`レベル）。クリップごとに異なるネガティブプロンプトを持たせる機能は実装していない。
6. **`_run_inference`のモジュールグローバル差し替えが5本に達した。** `fast_video_pipeline.py`の`_run_inference`は、NAGの`encode_text`パッチを含めてモジュールレベルのグローバル関数/属性を5本差し替えるようになった（Wave 1で`try:`の開始位置を前倒しし全5本をfinally復元の傘に入れる形で安全性は担保済みだが）。次にこの方式で6本目以降を足す必要が出た場合は、モンキーパッチの本数がスケールしない設計であることを踏まえ、方式そのもの（例: コンテキストマネージャ化、専用のパッチスタック管理）を再検討すべき。

### MCPサーバー実装完了の追記（2026-07-28・NAGとは別系統・並走）

上記NAGとは別に、同日のセッションでClaude Code等のMCPクライアントからバックエンドを操作するための `mcp_server/` パッケージ（Web操作パネル相当の22ツール）を実装した。Wave 0〜7を完了し、pytestは658→675→699→**736**（730 passed / 6 skipped、退行ゼロ）、`echo "" | python -m mcp_server` のstdoutが0バイトであることも確認済み。`.mcp.json`は`scripts/setup.ps1`が絶対パス入りで自動生成する方式にし、`.gitignore`に追加済み。**ただし実機（Claude Codeからの実際の操作）での検証はまだ一つも行っていない。** 承認ゲート→22ツール表示→T2V/I2V/A2Vバッチ/join/purge dry_run/JOB_BUSY挙動/api_key再起動の実機チェックリストは[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §39に転記済み（全項目⬜未実施）。設計判断の記録は新設の[`MCP_SERVER_DESIGN.md`](MCP_SERVER_DESIGN.md)、利用者向け説明は[`../README.md`](../README.md)「AIエージェント連携（MCPサーバー）」節（§8）を参照。次セッションでオーナーの実機確認を行うこと。**（2026-08-04追記＝本段落は当時の記録。実機検証はその後 [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §39.6 で実施され、Claude Code UIの承認導線を除く全項目が合格した。残るのは承認ダイアログと`/mcp`画面の目視1件だけで、これはオーナーがClaude Codeを起動しないと確認できない。）**

---

## ▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-07-26 α版インストール導線の整備＝実装完了・**コミット＆プッシュ済み**・**オーナーのサブマシン実機検証は 2026-07-27 に全項目合格**）

> **配布・導入まわりについては本ブロックが引き続き正本。** 生成機能そのものの正本は**上の「2026-07-28 NAG」ブロック**（旧・正本だった「2026-07-14 Clip Chain拡張」ブロックはそちらに置き換わった）。二系統が並走している（配布・導入は本ブロック、生成機能は上の2026-07-28ブロックが正）。それ以降の▶節はすべて歴史記録。

### ✅ 最初に読むこと — 作業ツリーの状態（2026-07-27 更新。旧「未コミット警告」は役目を終えた）

**下記の実装はすべてコミット＆プッシュ済みである。** `Nz-LTX23-backend` と
`Nz-LTX23-frontend-AviUtl2` は、いずれも 2026-07-27 時点で `origin/main` と同期済みであり、
作業ツリーはクリーン（`nothing to commit, working tree clean`）だった。以後のコミットについては
個々のハッシュを本書に転記せず、`git log` を参照すること（転記するとすぐ古くなるため）。

コミットは **`git add -A` による一括コミット**で行われ、懸念されていた壊れた中間コミットは回避された。
バックエンド側のコミットには `config.yaml => config.yaml.example` のリネーム・`.gitignore` への追加・
`setup.bat` / `run.bat` / `scripts/setup.ps1` の新設・`NzLTX23-1.0.0-rc1.au2pkg.zip` の同梱が
**すべて同一コミットに入っている**。

> **（歴史記録）2026-07-26 時点では全変更が未コミットで、しかも `git index` には `config.yaml` と
> `wheels/.gitkeep` の削除だけがステージ済みという中途半端な状態だった。** そのまま素の `git commit` を
> 打っていれば、「`config.yaml` は消えたが `config.yaml.example` も `.gitignore` の追加も入っていない」
> コミット——**[`PENDING_TASKS_CLOSED.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md) §3-37 が
> 「絶対に避けろ」と書いている壊れた中間コミットそのもの**——ができるところだった。
> 「ひな型の追加・`.gitignore` への追加・`git rm --cached` は**同一コミットで**行う（分けると中間の
> コミットを引いた人が詰む）」という原則は、今後同種の作業をするときも必ず守ること。なお本プロジェクトの
> 慣行として、**コミットとプッシュはオーナーが手動で行う**。

### 完了している内容

- **モデルの再ホストが完了**（HuggingFace `Rootport` アカウント・**4リポジトリ**）。`Nz-LTX23-weights`（LTX本体5点＋**IC-LoRA 3点＋VDA深度前処理器2点**＝10ファイル・25,219,475,913 B。2026-08-03 に Deblur と VDA を追加）／`Nz-Gemma3-12B`（11ファイル・7,339,810,357 B）／`Nz-DWPose`（**新設**・前処理器2点・352,756,773 B）／`Nz-Sulphur2`（自家変換GGUF 1点・インストーラの取得対象外）。全リポジトリが Public 非 Gated で、**HFアカウントもトークンも不要**。
- **`install_ltx.ps1` は 3 リポジトリから、5 回のダウンロード呼び出しで合計 23 ファイル・32,912,043,043 バイト（約30.7GiB）を取得する。** 従来インストーラの管理外だった `models/ltx-2.3-ic-lora/` と `models/preprocessors/` も**自動取得の対象になった**（手動配置は不要）。step 6 の検証表は 10 → **16 項目**。
- **エンドユーザー入口 `setup.bat` / `run.bat` を新設**（前提ツールは git のみ。`uv` と `ffmpeg`/`ffprobe` は `tools/` へ取り込む）。`config.yaml` は追跡外化し `config.yaml.example` から複製する。フロントエンドの `.au2pkg.zip`（378,689 B）をバックエンドリポジトリ直下に同梱。
- **従来は Gated リポジトリにあった spatial upsampler と Gemma tokenizer 一式も、この再ホストで非 Gated になった**（ブラウザでのライセンス承諾とアクセストークンの発行が不要になった）。`install_ltx.ps1` からトークン関連の引数（`-HfToken`）と、ログイン補助スクリプト `scripts/hf_login.ps1` は削除済み。
- **本番トランスフォーマー GGUF の配置も整理した。** 旧取得元は `models/ltx-2.3-gguf/LTX-2.3-distilled-1.1/` というサブフォルダ構造だったため `install_ltx.ps1` がダウンロード後に1階層上へ移動していたが、新リポジトリは最初から `models/ltx-2.3-gguf/` **直下**の構造で持っているので、この移動処理そのものを削除した。
  **旧 `install_ltx.ps1` でインストールした環境がサブフォルダ配置のまま残っている場合は、手動で移動すること**（再実行しても自動では直らない）。

  ```powershell
  Move-Item "models\ltx-2.3-gguf\LTX-2.3-distilled-1.1\*.gguf" "models\ltx-2.3-gguf\"
  Remove-Item "models\ltx-2.3-gguf\LTX-2.3-distilled-1.1" -Force
  ```

  サブフォルダ配置でも再帰スキャンでモデル自体は認識されるが、以後の公式手順・ドキュメントの既定パスは直下を前提にする。`config.yaml` に `model.gguf_transformer_path` を明示指定している場合は直下のパス（既定値 `./models/ltx-2.3-gguf/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf`）へ書き換えること。指定していない場合はコード側の既定値が既に直下パスを指すため編集不要。
- **正本**: フロントエンド側 [`Docs/PENDING_TASKS_CLOSED.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md) §3-36（再ホストと取得元差し替え）・§3-37（`setup.bat`/`run.bat`・`.au2pkg.zip` 同梱・**コミット分割の注意**）、再ホスト作業の手順書＝[`Nz-HF-Rehost/README.md`](../../Nz-HF-Rehost/README.md)、本書の基盤アーカイブ「install スクリプト」項、`LTX23_Backend_Specification.md` §2.5／§5.1b、`README.md` §1。

### 実機検証の結果（2026-07-27・全項目合格）

オーナーがサブマシンで実機検証を行い、**全項目に合格した**。検証環境は **RAM 32GB／RTX 3080 mobile（VRAM 16GB・Ampere 世代）／AviUtl2 を導入していない新規環境**で、AviUtl2 は **2026-07-25 更新の公開最新版**（開発機の v2.0.54 より新しい）を新規に導入した。合格したのは次の 9 項目である。

1. `setup.bat` でのインストール。
2. `run.bat` でのサーバー起動。
3. ブラウザで WebUI を開く。
4. `smoke_test` サイズの動画生成。
5. **IC-LoRA DWPose（pose-control）768p・257 フレーム** — コンソールに `loras=pose-control(strength=1)` と表示され、エラーなく**制御された動画の生成に成功**。
6. **IC-LoRA canny 768p・257 フレーム** — 同様に成功。
7. `NzLTX23.aux2` を AviUtl2 のプレビュー画面から D&D でインストール。
8. 再起動後、Nz-LTX23 の操作パネル表示を確認。
9. タイムラインからの動画生成と、生成済み動画を右クリックからタイムラインへ配置。

この結果から確定したこと:

- **RAM 32GB で、最小構成の生成どころか 768p／257 フレームの IC-LoRA 制御生成まで動く**（「32GB の実測データが存在しない」という従来の制約は解消した。ただし**この 1 台での実測合格**であり、あらゆる 32GB 環境での動作保証ではない）。
- **Ampere（`sm_86`）での実動を確認した**（従来は理論上の互換のみ）。
- **`config.yaml` の自動複製経路が初めて実際に通った。** 公開リポジトリに `config.yaml` は無いため新規環境では `.example` からの複製が必ず走り、real バックエンドで生成できたことがそのまま「正しく複製された」証拠になる。
- **2026-07-25 更新版の AviUtl2 との互換を実証した**（開発機は v2.0.54 固定だが、より新しい版で全機能が動作した）。

なお **`Ctrl+C` で停止したときの日本語表示は依然として未検証**である（`README.md` §1「サーバーの止め方」は `Ctrl+C` を意図的に案内していない）。

### 残っていること

1. **README の文面をオーナーが手書きで仕上げる**（冒頭に置く「スピードガイド」。備忘の正本＝[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §1-4）。
2. **リポジトリを public にする。**
3. コミット＆プッシュは完了済み（上記「作業ツリーの状態」）。実機検証も完了済み（上記）。**コード側の設計・実装の未了はない。**

---

## ▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-07-14 Clip Chain拡張（±ボタン式の折りたたみUI・クリップ連結上限8→24）＝実装完了・**GPU実機ゲート✅合格（オーナー確認済み）**・pytest 598 passed / 1 skipped）（**生成機能の正本**）

> **生成機能については本ブロックが正本。** これ以降の▶節はすべて歴史記録。食い違ったら本ブロックが正（配布・導入まわりは上の 2026-07-26 ブロックが正）。正式なワークオーダー＝[`CHAIN_UI_EXPANSION_WORKORDER.md`](CHAIN_UI_EXPANSION_WORKORDER.md)（実装完了の要約・オーナー決定事項・2026-07-14の実機ゲート結果〔二点測定の実測値・スピル発生箇所の特定・外挿と判定〕を冒頭の歴史記録ブロックに記載済み）。

### 本日完了した内容の要約

前回の引き継ぎ課題①「Clip Chain拡張」（[`CHAIN_UI_EXPANSION_WORKORDER.md`](CHAIN_UI_EXPANSION_WORKORDER.md)）を実装完了し、GPU実機ゲートまで通した。

- **Clip Chainタブ・Generateタブのクリップ入力欄／キーフレーム入力欄を±ボタン式の折りたたみUIにした**（既定はクリップ2枠・キーフレーム1枚のみ表示、＋で開き−で閉じる非破壊方式）。
- **クリップ連結の上限を8→24（総尺 24×481＝11544ピクセルフレーム）へ引き上げた**（API・UI両側）。「Clip list」見出しの直下に連結後の推計秒数を常時表示するようにした。ジョブ監視タイムアウトの既定値を60分→120分へ延長。
- pytest **598 passed / 1 skipped**（退行ゼロ）。

#### GPU実機ゲート結果（合格・オーナー確認済み・2026-07-14）

着手前の必須調査だったVAEデコードのメモリ挙動は、タイル分割の逐次処理＋ディスクへのストリーミング書き出しで、VRAM使用量が総尺に依存しないことを確定済み。加えて残タスクだったVRAM実測をオーナーが実施し、API自動実行での768p二点測定も行った。要点は以下（詳細な実測値・表は [`CHAIN_UI_EXPANSION_WORKORDER.md`](CHAIN_UI_EXPANSION_WORKORDER.md)「実機ゲート結果（2026-07-14）」節）。

- **GUI実機確認はオーナー確認済みでクローズ**（±ボタン・カウンター・グレーアウト・推計秒数すべて想定どおり）。
- **機能スパイク**: 512x320・49f×24クリップ・overlap3が1160.3秒で完走、peak_vram=9525MB、出力785フレーム。**24クリップ連結の機能そのものは成立**。
- **768p二点測定でスピルの主因を特定**: stage-1完了直後の「連結タイムライン全体を一括で高解像度化するアップサンプル処理」（`engine/pipeline/chain_pipeline.py:728-733` の `upsample_video`）が、総尺に比例して増える最大のVRAM項であり、これがスピル（GPU専用メモリ→共有GPUメモリへの溢れ）・OOMの主因。stage-2の仕上げデノイズは時間タイル毎に処理されスピルなし＝タイル化は有効。
- **判定**: 768pの24クリップはハードOOM確実（合計需要ベースで約98GB外挿・機体総上限約48GB）。**安全性は「総尺 × 解像度」の積で決まる**。320pの24クリップは9.5GBで安全。
- **オーナー決定（2026-07-14）**: 上限24クリップの実装は**今回このままリリースする**（320p帯で実証済み。768p長尺は現状スピルにより低速・リスクありという既知の制約として明記）。根本対策は次セッションでアップサンプルの時間チャンク化に取り組む（下記「次セッションの課題」参照）。

### 次セッションの課題（2026-07-14起票）

Clip Chain拡張の完結を受けて、次セッションの課題は以下の2件。**優先順位はオーナー判断とする。**

① ~~アップサンプルの時間チャンク化~~ → **実装完了・GPU実機ゲート合格（2026-07-14、コミット ff8a0e2/10be98d/cab1ca5/8b65226、修正 eb3c82d、GUI既定オン 3ba4a5d）。正本＝[`CHUNKED_UPSAMPLE_WORKORDER.md`](CHUNKED_UPSAMPLE_WORKORDER.md)。APIの既定値は互換維持のためFalseのまま（GUIのチェックボックスのみ既定オン）。**

② **WebView2フロントエンドのバッチA2Vパリティ**（AviUtl2拡張・別リポジトリの課題）: 正本＝[`BATCH_A2V_CSV_SPEC.md`](BATCH_A2V_CSV_SPEC.md)。

> **2026-07-15追記**: フロントエンドのバックエンド追随はより広い範囲を[`FRONTEND_CATCHUP_WORKORDER.md`](FRONTEND_CATCHUP_WORKORDER.md)（2026-07-15新設）で正本化した。上記②（バッチA2Vパリティ）は同書のグループ2に統合済み。課題①（チャンク化アップサンプル）は上記のとおり実装完了・実機ゲート合格済み。

なお、長尺化まわりの将来研究課題（単発生成へのチャンク化移植・クリップ毎のキャラクター特徴注入など、いずれも「すぐには改修しない」オーナー確定事項）は、**2026-07-27の整理で[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §3-42・§3-43へ移設した**。あわせて、実運用で得た経験則（連結点の歪みの正体・キャラクター設計のドリフト・「1クリップ最長×連結4〜5個」という実用最適解）と調査の一次情報URLは[`PHASE3_CLIP_CONCAT_STATUS.md`](PHASE3_CLIP_CONCAT_STATUS.md)「実運用で得た経験則（2026-07-14〜15・オーナーの長尺使い込み観察）」節へ吸収し、両者をまとめていた旧ノート`LONGFORM_RESEARCH_TOPICS.md`（2026-07-15新設）は削除した。

---

## ▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-07-13 バッチA2Vのフールプルーフ判定（開始前チェック）の画像判定を強化＝実装完了・pytest 589 tests / 588 passed / 1 skipped・**コミット＆プッシュ完了（2分割: 18fb08e=feat バッチA2V機能一式、2ccf8fb=fix コンソールログ静音化、origin/main）・バッチA2Vの課題は全て完結**）（歴史記録）

## ▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-07-12 バッチA2V機能（就寝中に音声素材100〜200個ぶんを一括生成するGUI機能）＝実装完了・客観ゲート全PASS（pytest 585 passed / 1 skipped）・**GPU実機目視ゲート✅合格（オーナー確認済み）・実機フィードバック修正5点＋第2次フィードバック対応6点も実装済み**）（歴史記録）

> **（歴史記録）本ブロックは上位の「2026-07-14 Clip Chain拡張」ブロックに置き換わった。** 本ブロック末尾の「次セッションの課題」①「Clip Chain拡張」は**実施済み**（最新ブロック参照）。食い違ったら最新ブロックが正。詳細＝`VERIFICATION_LOG.md` §36（バッチA2V機能・§36.6〜§36.8にGPU実機ゲート結果と修正ラウンド、§36.9に第2次フィードバック対応を追記済み）。正式なワークオーダー＝[`BATCH_A2V_WORKORDER.md`](BATCH_A2V_WORKORDER.md)（§4の未決5論点の決着内容・GPU実機ゲート結果・オーナー決定2件・修正5点・第2次フィードバック対応6点を含む・実装後は歴史記録として本文を残しつつ冒頭に決着ブロックを追記済み）。CSVマニフェストの共通仕様（WebView2フロントエンド＝AviUtl2拡張・Reactベースのフロントエンドとの相互運用の正本）＝新設[`BATCH_A2V_CSV_SPEC.md`](BATCH_A2V_CSV_SPEC.md)。

### 本日完了した内容の要約

前回セッションの引き継ぎ課題（下の「2026-07-11 reference動画付きIC-LoRAのchain対応」ブロック末尾の「次セッションの課題＝バッチA2V機能の設計＆実装」・正本は[`BATCH_A2V_WORKORDER.md`](BATCH_A2V_WORKORDER.md)）を実装完了。Generateタブの A2V（音声から動画を生成する機能。アップロードした音声の長さに合わせて映像を生成する）アコーディオンの下に「Batch A2V」アコーディオンを新設し、ゆっくり実況・VOICEROID実況（音声合成ソフトによるキャラクター実況動画）の制作者が、音声素材100〜200個ぶんの i2v（画像から動画を生成する機能）口パクアニメーションを、就寝前に一括投入して朝までに仕上げられるようにした。

Enableチェック（既定オフ）をオンにすると Generate ボタンが「Start a2v batch」に変わり、単発の音声欄・Frames欄はグレーアウトする（Framesは行ごとに wav の長さから自動算出されるため）。操作は「音声フォルダのパス入力→（任意で）画像フォルダのパス入力→Set audiosで表を生成→表の上で行ごとのプロンプト・画像を編集→Start a2v batchで開始」という流れ。生成設定（解像度・fps・シード・LoRA・共通プロンプト・共通キーフレーム画像・参照動画）はバッチ開始時点の Generate タブの値をスナップショットとして全行に適用する。

#### バッチのループはサーバー内バックグラウンドスレッドで回る（就寝中運用の核心）

バッチのループは Gradio の画面イベント内では実行せず、サーバープロセス内のデーモンスレッド（`BatchRunner`）に持たせた。**ブラウザを閉じてもタブがスリープしても生成は止まらない。** 画面側は2秒間隔の `gr.Timer` が表を再描画するだけの読み取り専用クライアントで、翌朝ブラウザを開き直せば現状がそのまま表示される。行状態の正典はCSVマニフェストであり、サーバーAPI自体は無改修（既存の `POST /generate/chain` を1件ずつ逐次投入、同時1ジョブ制のまま）。

#### ワークオーダー§4の未決5論点の決着（実装時にオーナー承認済み・詳細＝`BATCH_A2V_WORKORDER.md`冒頭ブロック）

1. **Replace切替時の挙動**: オーナー原案（切替時に共通プロンプトを全行へ転記＋Undo必須）ではなく、対案の**非破壊方式**を採用。行のテキストは常に「個別プロンプトのみ」を保持し、送信時にAdd（共通の末尾に行のテキストを追記）／Replace（行のテキストが共通を置換。行が空なら共通にフォールバック）というグローバル切替に応じて合成する。個別プロンプトが空の行を下敷き編集したいニーズには、「共通をこの行にコピー」ボタンで対応した。
2. **Prompt列の編集方式**: セル直接編集と下部パネル経由の折衷案を採用。Prompt列のみ表のセル直接編集（`.edit`イベントで差分検知）を許し、image列（選択式が必要）は行選択→下部編集パネルの`gr.Dropdown`経由にした。
3. **Skipの閾値**: 「快適上限は警告のみで表示し、Skip自体は481フレーム基準にする」の有力案どおりに確定。481フレーム（APIのハード上限）超過のみをSkipとし、解像度別の快適上限（VRAMあふれの無い目安）超過は警告表示のみでSkipしない。
4. **バッチAPIの実装層**: 第一候補（GUI側ループが既存の`POST /generate/chain`を1件ずつ叩く・API側無改修）を採用。ただし前述のとおり、ループ自体はGradioの画面イベントではなくサーバープロセス内のバックグラウンドスレッドに持たせる設計にした。
5. **CSVマニフェストの形式**: 10列・UTF-8 BOM付き（Excelでダブルクリックしても文字化けしない）・CRLF改行・音声フォルダ直下に`batch_a2v_manifest.csv`として配置、と確定。WebView2フロントエンドとの共通仕様として、詳細を新設の[`BATCH_A2V_CSV_SPEC.md`](BATCH_A2V_CSV_SPEC.md)に正本化した。

#### 実装ファイル

- `gradio_ui/manifest.py`（新規・CSVマニフェストの純ロジック。Gradio/スレッド/HTTPに非依存で他層から安全に呼べる）
- `gradio_ui/batch.py`（新規・`BatchRunner`。サーバープロセス内デーモンスレッドで動く実行本体）
- `gradio_ui/handlers.py`（既存の単発A2V送信ロジックから`build_a2v_chain_payload`を関数抽出し、バッチ側と共有。凍結A2V送信とバイト等価であることを維持）
- `gradio_ui/ui.py`（Batch A2Vアコーディオンの配線・2秒間隔`gr.Timer`によるポーリング表示）
- `gradio_ui/i18n.py`（EN/JA文言48キー追加）

#### 品質記録

- **pytest 574 passed / 1 skipped**（前回§35基準の526+1=527 → +48・退行ゼロ）。新規テストファイル3本（`tests/test_gradio_batch_manifest.py`・`tests/test_gradio_batch_runner.py`・`tests/test_gradio_batch_fullstack.py`）＋既存`tests/test_gradio_handlers.py`の更新。
- **独立レビュー実施済み**: must-fix（必須修正）1件（画像フォルダの解決処理の不具合）を修正済み。should-fix（推奨修正）3件のうち2件（一時ファイルの掃除漏れ・Set audios実行中のロック不足）を修正済み。残り1件（サーバー側`outputs`/`uploads`フォルダの自動削除）は、自動削除が`metadata.json`のseed記録も一緒に消してしまうトレードオフがあるため、オーナー判断待ちの検討事項として記録のみとした（下記「次セッションの課題」参照）。
- フルスタックテスト（実サーバー・mockバックエンド経由）で、失敗継続・再開・出力ファイルの付番・Add/Replaceモードの実送信を検証済み。
- **GPU実機目視ゲート✅合格（オーナー確認済み・2026-07-12）**。詳細は次項。

#### GPU実機目視ゲート結果（合格・オーナー確認済み・2026-07-12）

実ログで確認された動作: 3行バッチ（うち1行はSkip）の2行連続生成→就寝想定の完走、ガチャ再生成（同じセリフの撮り直し）で`_2`〜`_4`の自動付番（既存mp4は無傷）、停止ボタンで「実行中1件は完走→次行を投入せず停止」という設計どおりの挙動（ログに`stop requested, halting before row 2`）、再開時にDone/Skip行を正しくスキップ、日本語ファイル名の音声で正常動作、解像度変更（1280x768→896x1152）がスナップショットへ正しく反映されること。**オーナーの評価は「事前に期待していた要件はいずれも満たしている」。** 詳細＝`VERIFICATION_LOG.md` §36.6。

#### オーナー決定2件（判断待ちだった2件が決着・2026-07-12）

1. サーバー側`outputs`/`uploads`フォルダの掃除は**ユーザーの手動削除で運用**と決定。UI側に削除機能は追加しない。理由＝ローカル専用ツールでありユーザーは日常的にエクスプローラーで`\outputs`を直接操作するため、直接操作のほうが便利という判断。
2. 行の個別プロンプトでの`<lora:名前:強度>`トークン指定は**将来の研究課題へ格下げ**。ユーザーから要望が出た時点で検討する。現状どおりLoRAは共通プロンプト側／バッチ開始時のGenerateタブ設定（スナップショット）で全行共通のまま。

#### 実機フィードバックに基づく修正5点（実装済み・pytest 574 passed / 1 skipped＝退行ゼロ・2026-07-12）

1. バッチ実行中はGenerateボタンが「Batching a2v...」表示に変わり押下不可になり、完了で「Start a2v batch」に復元される（誤操作による二重投入の防止。i18nキー`batch_running`を新設）。
2. Batch A2Vアコーディオン内の要素を実際の操作順に並べ替え: Enableパネル（`gr.Group`にEnable／音声フォルダ／画像フォルダ／出力先〔Auto/Custom〕を集約）→ Prompt mode（Add/Replace）はSet audiosボタンの直上 → Set audios → 表 → 表の直下にmax durationとサマリ（Done/Failed/Skip/Waiting）を横並びで集約 → 行編集パネル（画像ドロップダウン・行情報）→ 共通プロンプトコピー → 行再生成 → 停止ボタン。
3. 表の列名を短縮: queue#→「#」、Duration→「dur.」（表示崩れ対策）。
4. `gr.Dataframe`の`column_widths`でPrompt列の初期幅を41%確保（Set audios直後からセルをクリックして編集しやすくした）。
5. コンソールログの静音化: uvicornのアクセスログから`/ui/gradio_api/`を含む行（Gradioの内部イベント通信。2秒間隔の表更新タイマー等が出す大量ログ）を除外するフィルタを`main.py`に追加（既存の`JobPollingAccessFilter`と同方式）。GradioがStarletteの古い定数を使うことで出る`StarletteDeprecationWarning`もメッセージ・カテゴリ限定で抑制。`/api/v1/*`のアクセスログと生成進捗ログは従来どおり表示され、表更新タイマーの間隔（2秒）自体は変更なし。

#### 第2次フィードバック対応6点（実装済み・pytest 585 passed / 1 skipped＝退行ゼロ・2026-07-12）

上記の修正5点をオーナーが実機で使い込んだ結果、さらに6点のフィードバックが寄せられ、同一セッション内で実装・検証まで完了した。詳細＝`VERIFICATION_LOG.md` §36.9。

1. **表の横スクロール復活**: 列幅を固定ピクセル（#=50px／音声=220px／dur.=70px／画像=140px／Prompt=480px／状態=90px／出力=220px、合計1270px）に変更。Gradioの割合指定の列幅はビューポートに常に収まる仕様のため、以前はPrompt列が潰れて折り返しが多発していた。合計幅がパネル幅を超えると横スクロールバーが出るようになった（セル内折り返し自体は禁止していない）。
2. **表示位置の調整**: max durationとサマリ（Done/Failed/Skip/Waiting）の行を「Set audios」ボタンと表の間（表の直上）に移動。
3. **Start押下時のSkip再判定**: バッチ開始時に、処理対象の全行についてフレーム数とSkip判定を開始時点のフレームレートで再計算する。手動でWaitingに戻した上限超過行も開始時に確実にSkipへ再マークされ、「Set audios」後にfps・解像度を変更してもフレーム数は開始時設定で計算し直される。**注意**: 行のframes値を手で編集していても、開始時にこの再計算で上書きされる。判定・再計算がすべて通ってからCSVへ書き込みバッチを開始するため、開始不可のときは表もCSVも変化しない。
4. **Regenerateガード**: Skip行を選んで「この行を再生成」を押してもWaitingに戻らず、理由付きの警告が出る。
5. **フールプルーフ（開始前チェック・オーナー承認済みの判定表）**: 検査対象は処理対象行（Waiting/Failed/クラッシュ遺残）のみ。プロンプトは共通プロンプトが空でAddモードなら開始不可、Replaceモードなら処理対象行のプロンプトが1行でも空なら開始不可（空の行数を表示）。画像は共通キーフレーム画像が未設定で"Shared"のままの行が1つでも残っていれば開始不可。入力忘れの単純ミスで一部の行だけ走ってFailedが量産される事態を防ぐ狙い。全行に個別入力を済ませた使い方（Replace＋全行個別プロンプト、全行個別画像）は従来どおり可能。
6. **テスト**: 既存574件＋新規11件＝**585 passed / 1 skipped**。フルスタックテストの1本（音声が短すぎて422になるシナリオ）は、フレーム数常時再計算によりそのエラー自体が到達不能になったため、別の失敗ベクタ（存在しない個別画像）に差し替えて再開機能の検証を維持。

#### 画像判定の強化（実装済み・pytest 589 tests / 588 passed / 1 skipped＝退行ゼロ・2026-07-13）

上記5.の画像判定を、「共通キーフレーム画像が1枚でもあればOK」から「**1枚目**（Keyframe imagesのスロット1・先頭フレーム0。Useチェックがオン かつ 画像ありが条件）が入っているか」に厳密化した（オーナー承認済み仕様）。スロット2以降だけ設定されている状態は1枚目扱いにならず、"Shared"行が残っていれば開始不可のまま。ui.py側のdispatchはもともとUseチェックがオンかつ画像ありのスロットしか収集しないため、Useチェックの担保自体は変更なし。個別画像ファイルの実在チェックは追加していない（オーナー決定）。詳細＝`VERIFICATION_LOG.md` §36.9、`BATCH_A2V_WORKORDER.md`「画像判定の強化」節。

#### 既知の制約（詳細＝`BATCH_A2V_WORKORDER.md`冒頭ブロック・`BATCH_A2V_CSV_SPEC.md`）

1. 行の個別プロンプトに`<lora:名前:強度>`トークンは書けない（解析されず文字列として残ってしまう）。**研究課題へ格下げ済み（上記オーナー決定2）。**
2. シード欄が固定値だと全行が同一シードになる（同じセリフを撮り直す「ガチャ」運用では-1=ランダム推奨）。実際に使われたシードはサーバーのジョブ記録（`outputs/{job_id}/metadata.json`）に残る。
3. バッチ実行中は表の編集・Set audiosの再実行がロックされる。
4. サーバー側の`outputs`/`uploads`フォルダは自動掃除されない（200件回すとその分成長する）。**手動削除運用と決定済み（上記オーナー決定1）。**
5. 生成中の1件を即時中断する手段は従来通り無い（Phase 1からの既知制約）。

### 次セッションの課題（0〜3は完了済み・歴史記録として残置）

0. ~~**筆頭課題＝GPU実機目視ゲート（オーナー立ち会い）**~~ → **合格・完了（2026-07-12）**。詳細は上記「GPU実機目視ゲート結果」節・VERIFICATION_LOG §36.6。
1. ~~**オーナー判断待ち2件**~~ → **決定済み（2026-07-12）**。詳細は上記「オーナー決定2件」節。
2. ~~**筆頭課題＝コミット／プッシュ（オーナー判断）**~~ → **完了済み（2026-07-13、18fb08e／2ccf8fbとしてpush、origin/main）**。18fb08e＝feat（バッチA2V機能一式）、2ccf8fb＝fix（コンソールログ静音化）。**本セッション以降を通じて、バッチA2V機能に関する課題は全て完結した。**
3. **WebView2フロントエンド（AviUtl2拡張・別リポジトリ）側のCSVマニフェスト対応**は、本リポジトリのスコープ外の別リポ課題（未着手のまま）。仕様は[`BATCH_A2V_CSV_SPEC.md`](BATCH_A2V_CSV_SPEC.md)を正本として実装すること。

### 次セッションの課題（2026-07-13時点・未着手）

バッチA2V機能の完結を受けて、次セッションの課題は以下の2件。**優先順位はオーナー判断とする。**

① ~~**Clip Chain拡張**（Clip Chainタブ・Generateタブのクリップ入力欄／キーフレーム入力欄を折りたたみ式UIにし、クリップ連結の上限を24へ引き上げる）~~ → **実装完了・GPU実機ゲート✅合格（2026-07-14）**。詳細は最上部の「2026-07-14 Clip Chain拡張」ブロック。正本＝[`CHAIN_UI_EXPANSION_WORKORDER.md`](CHAIN_UI_EXPANSION_WORKORDER.md)。実機ゲートで判明した768p長尺のスピル対策は次セッション課題①「アップサンプルの時間チャンク化」（[`CHUNKED_UPSAMPLE_WORKORDER.md`](CHUNKED_UPSAMPLE_WORKORDER.md)）へ引き継いだ。

② **WebView2フロントエンドのバッチA2Vパリティ**（AviUtl2拡張・別リポジトリの課題。上記3.と同一の課題）: 正本＝[`BATCH_A2V_CSV_SPEC.md`](BATCH_A2V_CSV_SPEC.md)。

---

## ▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-07-11 **reference動画付きIC-LoRAのchain対応（α版・clips=1限定）＝実装完了**＝客観ゲート全PASS（pytest 526 passed / 0 failed / 1 skipped）・独立レビュー must-fix ゼロ・mock E2E PASS・**GPU実機目視ゲート✅完了・ユーザー受容（2026-07-12）**・**2コミットに分割してpush済み（7c50c32=機能追加、2d4b086=GUIスピナー修正）**／2026-07-12追記: **GUIスピナー退行（`1e18dea`起因）を`show_progress="hidden"`で修正し、オーナー実機確認も完了＝VERIFICATION_LOG §35クローズ**）（歴史記録）

> **（歴史記録）本ブロックは上位の「2026-07-12 バッチA2V機能」ブロックに置き換わった。** 本ブロック末尾の「次セッションの課題＝バッチA2V機能の設計＆実装」は**実施済み**（最新ブロック参照）。食い違ったら最新ブロックが正。詳細＝`VERIFICATION_LOG.md` §34（reference動画付きIC-LoRAのchain対応）・§35（GUIスピナー退行の修正）。旧「次セッションの課題」節（下記に残る2026-07-11 A2V＋LoRA併用の解禁ブロックの末尾）に記載していた「reference付きIC-LoRAのチェーン対応」課題は**本ブロックのセッションで実装完了**（α版・clips=1限定）。

### 本日完了した内容の要約

前回セッションの引き継ぎ課題（下の「2026-07-11 A2V＋LoRA併用の解禁」ブロック「次セッションの課題」節・要件確定はオーナー決定済み）を実装完了。**`clips` がちょうど1つのチェーン（A2V を含む）に限り**、reference動画付き control 系 IC-LoRA（参照動画から輪郭線 canny・骨格 pose 等を読み取って条件付けするアダプタ）をチェーンで使えるようにした。`clips` が2つ以上のチェーンは従来どおり 422（`LORA_CONTROL_UNSUPPORTED_IN_CHAIN`）で拒否を維持。

#### 実装（加算的変更・詳細＝VERIFICATION_LOG §34.2）

- **`api/models.py`**: `GenerateChainRequest` に `reference_video_id`／`conditioning_attention_strength`／`reference_video_strength`（いずれも optional・単発 `GenerateRequest` と同型）を追加。reference 指定時は clips=1 を許容するようクリップ数下限判定を調整し、reference＋loras空／reference＋clips≠1／strength系＋loras空／reference＋source_video排他の4種を新たに422で拒否。
- **`api/generate_chain.py`**: reference 不明ID→404／÷128違反→422 `REFERENCE_RESOLUTION_INVALID`／control＋clips≥2→従来どおり422 `LORA_CONTROL_UNSUPPORTED_IN_CHAIN`／control＋clips=1＋reference無し→422 `LORA_REQUIRES_REFERENCE`（単発と同一）／preprocess種別重複→422 `LORA_PREPROCESS_CONFLICT`。
- **`services/pipeline_manager.py`／`services/ltx_runner.py`**: `reference_video_id`→パス解決→実バックエンドは reference 有り時のみ worker ペイロードに `reference_video` ブロックを追加（省略時は byte 同一）。
- **`engine/worker.py`**: 単発 `_do_generate` にあった reference 解析ロジックを `_resolve_ic_reference()` として関数抽出（単発側の挙動は逐語不変）し、`_do_generate_chain` からも同じヘルパーを呼ぶよう変更。
- **`engine/pipeline/chain_pipeline.py run_chain`**: `ic_reference`／`ic_attention_strength` を追加し `_set_ic_job` へ渡す（無指定時は従来どおり `(None, 1.0)` の stale クリア）。**stage1のクリップ0（非V2V分岐）のみ** reference latent を注入＝stage2には注入しない（単発生成と同じ「全体へ一様適用・stage1のみ」の意味論）。
- **GUI（`gradio_ui/handlers.py`）**: A2V の送信前拒否（「adapter が None 以外なら拒否」）を撤去。adapter 選択時は単発経路と同じロジックで `reference_video_id`（＋1.0未満のときのみ strength 系2キー）を chain payload へ配線。adapter 選択＋参照動画未指定は既存の `msg_ref_video_required` で従来どおり事前拒否。デッドキー `a2v_control_lora_unsupported` を EN/JA から削除、`LORA_CONTROL_UNSUPPORTED_IN_CHAIN` の文言を「2クリップ以上のチェーンでは使えない」旨に正確化。

#### 受理／拒否マトリクス（新設・α版）

| 条件 | 結果 |
|---|---|
| `clips=1` ＋ control アダプタ ＋ `reference_video_id` | 受理（202→completed） |
| `clips>=2` ＋ control アダプタ（`reference_video_id` 有無を問わず） | 422 `LORA_CONTROL_UNSUPPORTED_IN_CHAIN` |
| `clips=1` ＋ control アダプタ ＋ `reference_video_id` 無し | 422 `LORA_REQUIRES_REFERENCE`（単発と同一） |
| `reference_video_id` ＋ `loras` 空 | 422（スキーマ検証） |
| `reference_video_id` ＋ `clips` が1でない | 422「requires exactly 1 clip in v1」（スキーマ検証） |
| `conditioning_attention_strength`／`reference_video_strength` 単独＋`loras` 空 | 422（スキーマ検証） |
| `reference_video_id` ＋ `source_video`（V2V） | 422（排他・スキーマ検証） |
| `reference_video_id` が未知ID | 404 |
| 幅・高さが128の倍数でない＋`reference_video_id` あり | 422 `REFERENCE_RESOLUTION_INVALID` |
| 2種以上の preprocess 種別が混在 | 422 `LORA_PREPROCESS_CONFLICT` |

**precedence 注記**: pydantic のスキーマ検証がエンドポイントより先に走るため、`clips>=2` ＋ `reference_video_id` の複合誤設定は、上表の個別コードではなく汎用の `VALIDATION_ERROR`（422）になる。

#### 既知事項（詳細＝VERIFICATION_LOG §34.6）

1. 上記 precedence 注記のとおり。
2. `reference_video_id` ＋ スタイル系 LoRA のみ（control 系無し）はスキーマ上は受理されるが、worker 側の `_set_ic_job` で `RuntimeError`（ジョブ失敗）になる。単発 `/generate` の既存挙動の忠実な写像であり、GUI からは到達不能（直接 API のみ）。早期422化は将来の改善余地。
3. `GET /jobs/{id}` 応答の `request` ブロックは chain 固有フィールド（`source_audio`／`loras`／`reference_video_id`）を載せない（`to_clip_request` 経由の `GenerateRequest` 形のため。前セッションの `loras` 追加時からの既存の表現制約であり退行ではない）。正式な記録は `metadata.json`。

#### テスト・検証（詳細＝VERIFICATION_LOG §34.3〜34.5）

- **pytest 526 passed / 0 failed / 1 skipped**（§33 基準 511+1 → +15・退行ゼロ。既存 skip 1件のみ）。新規 `tests/test_chain_reference.py`（14件）＋既存 `tests/test_chain_lora.py`／`tests/test_gradio_handlers.py` の更新。
- **独立レビュー（実装非関与）**: must-fix ゼロ。
- **mock E2E**（実サーバー起動・port 18901・scratchpad 上の隔離 config）: `clips=1`＋control＋reference＋A2V→202→completed（metadata.json に記録確認）／`clips=2`＋control→422／`clips=1`＋control＋reference無し→422／従来A2V（loras無し）→completed（回帰なし）＝いずれも想定どおり。
- **GPU実機目視ゲート✅完了・ユーザー受容（2026-07-12）**。観点＝①単発生成との一致（同一seed比較）②A2V音声との共存③省略時の従来動作維持。実績＝GUIのGenerateタブから音声wav＋参照動画＋control系アダプタで生成成功（1280×768・201frames・8steps・約300秒/本・ピークVRAM 9241〜9537MB。pose-control〔strength=1〕×3本＋canny-control〔strength=1〕×1本、いずれもcompleted。ジョブ開始ログに`loras=pose-control(strength=1)`等の配線物証あり）。詳細＝VERIFICATION_LOG §34.7。

#### GUIスピナー退行の修正（実機ゲート準備中に発覚・対応済み・オーナー実機確認済み・**§35クローズ**・詳細＝VERIFICATION_LOG §35）

上記 GPU 実機目視ゲートに着手する前提として実機ブラウザで GUI を触ったところ、本節の本題とは別系統の退行が見つかった。**Control adapter の永久スピナー・参照動画欄の操作不能（3ブラウザで再現）**という症状で、原因は `1e18dea`（WebGUI 大規模改修）単独＝複数の `demo.load` 同時発火時に Gradio 6.19 のクライアント側 status-tracker が pending のまま残り、そのオーバーレイがクリックを奪う既知動作だった。`gradio_ui/ui.py` の `demo.load` 3箇所（`on_page_load`／モデル管理ドロップダウン初期化／Style ギャラリー初期化）へ `show_progress="hidden"` を追加して修正済み。検証は **pytest（526 passed / 0 failed / 1 skipped）・CDP 実ブラウザ再検証・オーナー実機確認**のすべてが完了済み（詳細＝VERIFICATION_LOG §35.4）。オーナー実機確認（2026-07-12）では `run.ps1` 起動→`/ui` で Control adapter が即表示・操作可能なことに加え、**「以前よりGUIが開くまでの時間が短縮され軽快になった」という体感改善**も確認された。

既知の無害事象＝GPU実機目視ゲート中、2本目のジョブ完了直後にサーバーログへ asyncio の `ConnectionResetError [WinError 10054]` が1回出力されたが、Windows の asyncio proactor がブラウザ側の強制切断を後処理する際の既知の無害なノイズであり機能影響なし・対応不要（オーナー判断で無視と決定。詳細＝VERIFICATION_LOG §34.6の4）。

### 次セッションの課題（0・1は完了済み・歴史記録として残置）

0. ~~**スピナー修正のオーナー実機確認**~~ → **完了済み（2026-07-12）**。詳細は上記「GUIスピナー退行の修正」節・VERIFICATION_LOG §35.4。
1. ~~**GPU実機目視ゲート（オーナー立ち会い）**~~ → **完了済み・ユーザー受容（2026-07-12）**。詳細は上記・VERIFICATION_LOG §34.7。
2. ~~**筆頭課題＝コミット／プッシュ（オーナー判断）**~~ → **完了済み（2026-07-12、7c50c32／2d4b086としてpush）**。7c50c32＝機能追加（reference動画付きIC-LoRAのchain対応・α版）、2d4b086＝GUIスピナー修正（`show_progress="hidden"`）。
3. ~~**次セッションの課題＝バッチA2V機能の設計＆実装（正本＝[`BATCH_A2V_WORKORDER.md`](BATCH_A2V_WORKORDER.md)）**~~ → **実装完了（2026-07-12）**。§4の未決5論点はすべて決着し、客観ゲート全PASS（pytest 574 passed / 1 skipped）。詳細は本ドキュメント冒頭の最新ブロック・`VERIFICATION_LOG.md` §36。**GPU実機目視ゲートのみ未実施で次セッションへ持ち越し。**

---

## ▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-07-11 **A2V＋LoRA併用の解禁＝GPU実機目視ゲート含め完了（ユーザー受容）＋ジョブ起動ログ拡充＋GUIバグ2件修正（queued詰まり解消・ref_video無効化）**＝客観ゲート PASS（pytest 511 passed / 1 skipped＝512件）・独立レビュー2巡 must-fix残存なし）（歴史記録）

> **（歴史記録）本ブロックは上位の「2026-07-11 reference動画付きIC-LoRAのchain対応（α版）」ブロックに置き換わった。** 本文末尾「次セッションの課題」2点（reference付きIC-LoRAのチェーン対応・α版着手）は**実施済み**（最新ブロック参照）。食い違ったら最新ブロックが正。詳細＝`VERIFICATION_LOG.md` §32（A2V＋LoRA併用の解禁・目視ゲート✅クローズ）・§33（ログ拡充＋GUIバグ2件修正＋JobStore CAS化）。

### 本日完了した内容の要約

前回セッションの引き継ぎ課題（下の「2026-07-11 Gradio WebGUI 大規模改修」ブロック §「次セッション: A2V＋LoRA併用の解禁」）を実装完了。A2V（音声から動画を生成する機能。アップロードした音声の長さに合わせて映像を生成する）と、LoRA（少量のデータで画風・キャラクターなどを追加学習させた軽量アダプタ。生成モデル本体を変えずに絵柄だけ変えられる）を同時に使えるようにした。**さらに、後述のGPU実機目視ゲートとその調査を機に、ジョブ起動ログの拡充とGUIバグ2件の修正も本セッション内で完了した。**

#### 設計論点の決着（ユーザー決定・確定事項）

前回ブロックで「未決」としていた2点は、本セッションでユーザーが決定した:

1. **LoRA強度はチェーン全体で共通**。クリップ（連結される動画の1区間）ごとに強度を変える機能は v1 では見送り（`GenerateChainRequest.loras` はリクエスト全体で1本のリスト）。
2. **reference動画付きIC-LoRA（参照動画から輪郭線canny・骨格poseなどを読み取って条件付けする「control系」アダプタ）はチェーン非対応**。チェーン（`POST /generate/chain`）は `reference_video_id` を持たない構造のため、control系アダプタが指定された場合は API 側でジョブ予約前に 422 エラー（新設エラーコード `LORA_CONTROL_UNSUPPORTED_IN_CHAIN`）として拒否する。
3. GUI は Generate タブ（単発生成・A2V の窓口）と Clip Chain タブ（クリップ連結）の**両方**で解禁する。

#### バックエンド実装（引き継ぎ資料の「加算的変更5箇所」を全実装）

既存の凍結 API 契約への加算的変更（省略時は従来と byte 同一）として、以下5箇所を実装した:

1. **`api/models.py`**: `GenerateChainRequest` に `loras: list[LoraSpec]` を追加（`GenerateRequest.loras` と同じ型・同じバリデーション）。
2. **`api/generate_chain.py`**: ジョブ予約前に、指定された LoRA 名をすべて解決（未知/欠落なら 404）し、control系アダプタが含まれていれば 422（`LORA_CONTROL_UNSUPPORTED_IN_CHAIN`）で拒否。
3. **`services/pipeline_manager.py` → `services/ltx_runner.py` → `engine/worker.py`**: 解決済み LoRA（パス・強度）を worker へのペイロードに配線。
4. **`engine/pipeline/chain_pipeline.py run_chain`（`engine/pipeline/fast_video_pipeline.py` 経由）**: transformer（生成の中核モデル）をビルドする**前に**必ず `pipe._set_ic_job(...)` を呼び、LoRA を明示的にセットまたはクリアするようにした。LoRA 指定が無いチェーンでも空リストで明示的にクリアする。これは同時に、**stale（前のジョブの LoRA 適用状態が次のジョブに残留すること）** の防止修正でもある。従来は直前の単発 `generate()` で使った LoRA がチェーン側の常駐パイプラインに残ったまま denoise（ノイズ除去＝生成の中心処理）に持ち越される恐れがあったが、今回の修正でチェーン側も毎回明示的に状態をセット/クリアするようになった。

`loras` を省略した場合は worker へのペイロードにキー自体を含めない（従来と byte 同一）。

#### GUI 実装

- `gradio_ui/handlers.py`: A2V と LoRA の相互排他プリチェックを撤去。**reference動画付き control アダプタ＋A2V の組み合わせのみ**、新しい文言（`a2v_control_lora_unsupported`）で事前拒否する。単発生成経路の LoRA 合成ロジックを `_combine_generate_loras` として共通化し、A2V 経路でも同じロジックを使うようにした。
- Clip Chain タブ: これまでプロンプト内の `<lora:...>` トークンを除去して警告（`chain_lora_ignored`）を出していたのを撤去し、トークンを `loras` として API に送出するように変更。
- `gradio_ui/i18n.py`: EN/JA 両方を更新（`gen_a2v_conflict_lora`・`chain_lora_ignored` を削除、`apierr_LORA_CONTROL_UNSUPPORTED_IN_CHAIN` を新設）。

#### テスト・独立レビュー

- **pytest 500 passed / 1 skipped**（前回ブロックの基準 485+1 → 本セッションで +15・退行ゼロ）。新規テストファイル `tests/test_chain_lora.py`（13件）＝スキーマ検証・422/404・A2V＋LoRA併用の完走・worker ペイロード疎通・`_set_ic_job` のセット/クリア順序（GPU 不要のユニットテスト）をカバー。既存 `tests/test_gradio_handlers.py` も GUI 側の新仕様に合わせて更新・新規追加。
- **独立レビュー実施済み**: must-fix（必須修正）／should-fix（推奨修正）はゼロ。nit（軽微な指摘）1件のみ＝「control系アダプタの名前をプロンプトに `<lora:...>` として手打ちした場合、音声または画像を1回アップロード消費したあとサーバー側の 422 で拒否される（拒否メッセージ自体は正しくローカライズされる。Style ギャラリーのサムネイルをクリックする通常の操作では発生しない）」。既知の軽微事項として記録し、対応は見送り。

コード状態（A2V＋LoRA併用の解禁の実装のみの時点）: 全テスト **500 passed / 1 skipped** PASS。

---

### GPU実機目視ゲート（ユーザー立ち会い）＝✅完了・ユーザー受容（2026-07-11）

客観ゲート（pytest・独立レビュー）はすべて PASS していたが、実際に GPU 上で LoRA の効果を目視確認する「目視ゲート」は当初未実施だった。ユーザー立ち会いのもと以下4点を確認する過程で「LoRA が効いていないのでは」という報告が発生し、2段階の調査で原因を解明した。

- **調査(a)＝ログ面の誤解**: コンソールには LoRA 適用ログがそもそも出力されない仕様だった。物証として `logs/ltx_worker.log` の「N Linear(s) attached」行と `outputs/<job_id>/metadata.json` を確認したところ、**全ジョブで LoRA は当初から正常に適用されていた**ことが判明した。
- **調査(b)＝本質的な原因＝トリガーワード（LoRA を発動させるための合言葉となる特殊な単語）の欠落**: 使用していた Pixar_Toon LoRA は「P1x4r pixar style character」等のトリガー語をプロンプトに含めないと画風が変化しない設計であり、過去の成功例は全てトリガー語入りだった。報告のあった生成ではこのトリガー語が抜けていたため、LoRA 自体は適用されていても画風変化が目視で確認できなかった。
- 併せて alpha／rank 正規化（Pixar_Toon は alpha16／dim32＝×0.5 のため実効強度が指定値の半分になる仕様）と seed の影響も、体感的な効き方のばらつき要因として確認した。
- **トリガー語を入れた実機テストでユーザーが成功を確認**し、下記4点を含めて A2V＋LoRA 併用を受容。**目視ゲート✅クローズ**。

確認済みの4点:

1. A2V＋スタイル LoRA で、チェーン全体（全クリップ・全ステージ）に LoRA が効いていること。
2. 「単発 generate（LoRA有り）→chain（LoRA無し）」の順で実行し、stale な LoRA が chain 側に残留していないこと。
3. Clip Chain タブで LoRA が実際に適用されること。
4. `loras` 省略時のチェーン生成が従来と同じ出力になること（回帰なし）。

---

### ジョブ起動ログの拡充（オーナー要望・実装済み）

上記の目視ゲート調査を機に、コンソール（`ltx.pipeline` ロガー）のジョブ開始ログを拡充した。

- 実際に使われる seed（`-1` 指定＝ランダム決定の場合でも解決後の実値を表示。**表示値＝使用値の一致を保証**）。
- ベースとなる transformer GGUF のファイル名。
- 適用 LoRA 名（要求 strength／実効 strength の両方を併記。LoRA 無しの場合は `loras=none` と明示）。
- プロンプト先頭200字（改行はエスケープして1行に収める）。

今後同種の問い合わせがあっても、ログのみで seed・LoRA 適用状況を即座に確認できるようになった。

---

### GUIバグ2件の修正（計画→実装→独立レビュー→追修正→再検証まで完了）

1. **adapter=None＋参照動画のサイレント無視**: LoRA アダプタ未選択（None）のまま参照動画をアップロードしても警告なく無視される不具合。`gradio_ui/ui.py` で参照動画（ref_video）欄を初期状態で非活性化し、adapter 選択の `change` イベントで interactive を連動トグルするように修正。None へ切り替えた際はアップロード済みの動画も自動クリアする。
2. **queued 詰まりで無言ハング＆サーバー再起動が必須**: ジョブが queued 状態のまま進まなくなっても、GUI は何も表示せずユーザーはサーバー再起動以外に手段が無かった。以下3点で対応した。
   - ポーリング表示に queued 状態を追加し、30秒を超えて解消しない場合は「Jobs タブからキャンセルしてください」と誘導する文言を表示（i18n 新設キー `msg_queued`／`msg_queued_stuck`、EN／JA）。
   - `DELETE /jobs/{id}` が queued 状態のジョブに対しても即座に `cancelled` へ遷移させ、詰まりを解放できるようにした（従来は running 以降のみ対応）。
   - `JobStore` に `_lock` 配下の CAS（compare-and-swap）ヘルパー（`start_job`／`cancel_if_queued`）を新設し、`run_job`／`run_chain_job` の queued→running 昇格とキャンセル操作を相互排他化（独立レビュー指摘 S1＝TOCTOU を解消）。実バックエンド使用時はジョブ実行を専用 daemon スレッドへ切り出し（mock バックエンドは従来通り BackgroundTasks のまま）、`Thread.start()` 失敗時はジョブを failed へ遷移させガードを解放（独立レビュー指摘 S2）。AnyIO のスレッドリミッタを200へ拡大。
- **独立レビュー2巡**: 1巡目で should-fix 2件（S1・S2）を指摘→追修正→2巡目（再検証）で must-fix／should-fix とも残存なし。クローズ。

---

### 本セッション全体のコード状態

全テスト **511 passed / 1 skipped**（A2V＋LoRA併用の解禁時点の500+1 → 本日の追加分で+11・退行ゼロ）。**git commit は依然未実施（作業ツリーの変更のまま・ユーザー判断待ち）。**

---

### 次セッションの課題（本ブロック内の記録・いずれも実施済み／歴史記録）

1. **筆頭課題＝コミット／プッシュ（オーナー判断）**。本セッションの変更一式（A2V＋LoRA併用の解禁・ログ拡充・GUIバグ2件修正・JobStore CAS化）はすべて未コミットのまま。→ **未コミットのまま次セッションへ持ち越され、reference付きIC-LoRAのchain対応（α版）とあわせて本ドキュメント冒頭の最新ブロックで扱っている（依然コミット判断待ち）。**
2. **残課題＝reference付き IC-LoRA（参照動画による条件付けアダプタ＝control系）のチェーン対応は未実装**。§32.1 の決定どおり、チェーン（`POST /generate/chain`）は `reference_video_id` を持たない構造のため v1 のスコープ外とした。→ **実装完了（2026-07-11・本ドキュメント冒頭の最新ブロック・VERIFICATION_LOG §34 参照）。GPU実機目視ゲートのみ未実施で次セッションへ持ち越し。**
   - **→ 要件確定（オーナー決定・2026-07-11）＝α版として着手する。スコープは以下のとおり**:
     - **α版スコープ: `clips` がちょうど1つのチェーンに限り、reference動画付き IC-LoRA（control系＝参照動画から輪郭線canny・骨格pose等を読み取って条件付けするアダプタ）を許可する。`clips` が2つ以上のチェーンは従来どおり 422（`LORA_CONTROL_UNSUPPORTED_IN_CHAIN`）で拒否を維持する。**
     - **根拠**: 主眼のユースケースは A2V（音声から動画を生成する機能。アップロードした音声の長さに合わせて映像を生成する）との併用であり、A2V は内部的に「1クリップのチェーン」として実行されるため、clips=1 限定で目的を満たせる。
     - **実装方針のヒント**: 単発 `/generate` 側の reference 動画処理（`engine/worker.py` の前処理〜reference latent の配線。機構の詳細は [`IC_LORA_PHASE_B_STATUS.md`](IC_LORA_PHASE_B_STATUS.md)／[`IC_LORA_PHASE_C_STATUS.md`](IC_LORA_PHASE_C_STATUS.md) を参照）を clips=1 のチェーンへ流用する。意味論は単発と同じ「動画全体に一様適用」。
     - この形なら、stage 1（セグメント毎のノイズ除去）と stage 2（全体の高解像度化・仕上げ）の途中で LoRA を付け外しする複雑な実装を回避でき、実機テストも「単発生成との一致確認＋A2V 音声との共存確認」に収まる。

**研究課題へ格下げ（オーナー決定・2026-07-11）**: 次の2件は次セッション課題から除外し、将来の研究課題へ格下げした。

- 「control系アダプタ名をプロンプトに `<lora:...>` として手打ちした場合、1回分のアップロードが無駄になる」既知の軽微事項（nit・上記「テスト・独立レビュー」節参照）。ユーザーからイシューが上がってから対応する方針（着手条件＝ユーザー報告起点）。
- **クリップ毎の参照動画入力**（Clip 2 には Clip 2 専用の参照動画を与える、という方式）。理由＝ユースケースが不明瞭で、API・GUI・実機テストの工数が大きい。連結2本目以降に「同じ参照動画から抽出した動き」を反映させる用途も想定できない。上記α版（clips=1 限定・参照動画は1本）で主眼ユースケースは満たせる。

---

## ▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-07-11 **Gradio WebGUI 大規模改修**＝実装完了・客観ゲート PASS（pytest 485 passed / 1 skipped＝486件）・**次セッション＝A2V＋LoRA併用の解禁（オーナー決定済み）**）（歴史記録）

> **（歴史記録）本ブロックは上位の「2026-07-11 A2V＋LoRA併用の解禁」ブロックに置き換わった。** 本文中の「次セッション: A2V＋LoRA併用の解禁」（下記）は**実施済み**（最新ブロック参照）。食い違ったら最新ブロックが正。

### 本日完了した内容の要約

Gradio WebGUI の使い勝手改修一式。主戦場は `gradio_ui/` だが、**GUI だけの改修ではない**: プリセット再編と動画 GGUF の配置変更に伴い `config.py`・`config.yaml`・`services/model_registry.py`・`scripts/install_ltx.ps1`・テストにも同期変更が入っている（凍結 API 契約そのものは不変）。詳細は `LTX23_Backend_Specification.md` §11.4/§12.2 と `README.md`（Gradio UI 節・モデル配置節）に反映済み。

1. **Frames / Duration / Frame rate の統合パネル**（`gr.Group` 内・横3カラム）。中央カラムは入力欄を持たず、num_frames と frame_rate から算出した Duration（"N.NNs"）をアクセントカラーで常時表示。
2. **`gr.Number` のサーバー側 `minimum` 撤去 ＋ ページロード時 JS で `min` 属性付与**（width/height/num_frames 対象）。手入力途中（例: "80" と打つ途中の "8"）で最小値未満エラーが飛ぶ Gradio の不具合を根治。手入力は自由・スピナー矢印のみ 64刻み/8n+1刻みでスナップ。サーバー側の 8n+1・÷64 検証は従来どおり有効。
3. **A2V の音声長事前チェック**: `.wav` は送信前にクライアント側（stdlib `wave`）で長さを測定し、設定（フレーム数・fps）に対して不足していれば必要秒数を明示して送信を拒否（API 呼び出しゼロ）。`.wav` 以外はクライアント側で測定できないためスキップし、サーバーの `422 SOURCE_AUDIO_TOO_SHORT` をヒント付きで表示する。
4. **wav 添付時の Frames 自動調整**: 音声に収まる最大の 8n+1 値を自動計算して num_frames へ入力し、トースト通知。式は `((floor(秒×fps)-1)//8)*8+1` を起点に `chain_math.audio_latents_required` で latent 検算しながら8刻みで縮小、最終的に `[9, 481]` へクランプ。`.wav` 以外は対象外。
5. **生成中ボタンのグレーアウト**: 両方の生成ボタン（Generate / Generate chain）がクリック→無効化（"Generating..." / "生成中…"）→生成→必ず復元、の3段イベント連鎖（`.click().then().then()`。失敗時も `.then()` は実行されるため必ず復元される）。
6. **A2V のキーフレーム画像は5枚すべて配線済み**（従来から）であることをドキュメントに明記。`frame_idx>0` はサーバー側で 8n+1 グリッドへスナップ＋動画尺内クランプ（`api/models.py`）。
7. **生成プリセットの改名・追加**: `phase1_default`→`minimal`、`phase1_target`→`small` に改名し、`FHD_1080p`（1920×1088→1080クロップ・153f）と `WQHD_1440p`（2560×1472→1440クロップ・81f）を追加＝計6種。**`config.yaml` の `generation_presets` が単一の真実源**（GUI/フロントエンドは `GET /config` で動的取得）。WebView2 フロントエンドのフォールバック定数（`webui/src/modes/create/defaultConfig.ts`）も同期済み。
8. **動画 GGUF の公式配置を `models\ltx-2.3-gguf` 直下に変更**: 旧配置はサブフォルダ `LTX-2.3-distilled-1.1\` 内。`config.py`（`gguf_transformer_path` の既定を直下パスへ変更）・`services/model_registry.py`（transformer の `parent_levels` 2→1）・`scripts/install_ltx.ps1`（DL 後に1階層上へ自動移動）・テストを同期し、実ファイルも移動済み。サブフォルダ配置も再帰スキャンで引き続き動作する（README「追加の transformer GGUF / LoRA を配置する」節に移行手順あり）。

コード状態: 全テスト **485 passed / 1 skipped**（計486件）PASS。**本セッションの改修一式（バックエンド約22ファイル、フロントエンド3ファイル）はセッション末尾に main へコミット＆push 済み。**

---

### 次セッション: A2V＋LoRA併用の解禁（チェーンAPIへの `loras` 追加・オーナー決定済み）

> **（実施済み・歴史記録）本節の計画は2026-07-11に実装完了した。** 下記の「未決の設計論点」2点もユーザーが決定済み（①強度はチェーン全体で共通 ②reference付きIC-LoRAはchain非対応・422で拒否）。詳細・実装内容は本ドキュメント冒頭の最新ブロックを参照。

#### 目的と背景

現在、A2V（`POST /generate/chain` + `source_audio`）は IC-LoRA / スタイルLoRA と併用できない（GUI 側で事前拒否・`gen_a2v_conflict_lora`）。これは **LTX 2.3 モデル自体の制約ではなく、単なる配線欠落**。根拠:

- forward 時に LoRA を適用する機構（`engine/gguf/quant_service.py` の `patched_transformer` / `ggml_linear_forward`）は、chain の denoise 呼び出しにもそのまま効く。LoRA 適用は transformer の `forward()` にフックされており、単発 `generate()` かチェーンの `generate_chain()` かを区別しない。
- chain（`engine/pipeline/chain_pipeline.py run_chain` / `fast_video_pipeline.py generate_chain`）は **transformer を1回だけビルドして Stage1/Stage2 の全クリップ・全タイルで使い回す**構造。したがって chain に LoRA を配線すれば、**チェーン全体（＝実質すべてのクリップ）に適用される**。単発 `generate()` の一部の上流実装（LoRA を Stage1 のみに適用するもの）とは効き方が異なる点に注意（本チェーン実装は Stage1/Stage2 両方に効く＝より強く効く方向）。

#### 加算的変更が必要な5箇所

既存の凍結 API 契約への**加算的変更**（optional フィールド追加・省略時 byte 同一）。オーナー承認済み。

1. **`api/models.py`**: `GenerateChainRequest` に `loras: list[LoraSpec] = Field(default_factory=list)` を追加（既存 `LoraSpec` をそのまま流用。`GenerateRequest.loras` と同じ型・同じバリデーション）。
2. **`services/pipeline_manager.py`**: `run_chain_job` で `lora_registry.resolve(...)` を呼び名前→(path, strength, preprocess) を解決する。単発 `run_generation` の該当箇所（:235-237、`lora_paths = [self.lora_registry.resolve(spec.name, spec.strength) for spec in job.request.loras]`）がそのままコピー元になる。
3. **`services/ltx_runner.py`**: `generate_chain` のペイロード（worker へ渡す dict）に `loras` を追加（`generate` の `loras_payload` 組み立て・:1151/:1181 と同型）。
4. **`engine/worker.py`**: `_do_generate_chain`（:371〜）で `loras` を解析し、`_PIPE.generate_chain(..., ic_loras=...)` を呼ぶ。単発 `_do_generate`（:263〜、`ic_loras = [...]`・:285・:346）が実装の型。
5. **`engine/pipeline/fast_video_pipeline.py generate_chain`（:939〜）／`engine/pipeline/chain_pipeline.py run_chain`（:402〜）**: 両方に `ic_loras` 引数を追加し、**`ledger.transformer()` を呼び出す前に必ず `_set_ic_job(...)`（:267〜）で明示的にセット/クリアする**こと。単発 `generate()` は毎回 `_set_ic_job(eff_loras, ...)`（:894/:903）を呼んでから transformer を触っているが、chain は現状これを一切呼ばない。**直前の単発 `generate()` 由来の LoRA（`self._ic_loras`）がそのまま chain の denoise に持ち越される stale 問題を防ぐため、chain 側にも同じ明示セット（LoRA 無指定なら空リストで明示クリア）を必ず入れること。** これが今回の配線で最も事故りやすい箇所。

#### GUI 側の変更

- `gradio_ui/handlers.py` の A2V×LoRA 相互排他プリチェック（`gen_a2v_conflict_lora`、`make_generate_handler` 内 :364-367）を解除。
- Clip Chain タブの `<lora:...>` トークン除去＋警告（`chain_lora_ignored`、`make_chain_handler` 内 :862-865）を解除。
- `gradio_ui/i18n.py` の該当文言（EN/JA）を更新。

#### 未決の設計論点（着手前にユーザーと合意すること）

1. **LoRA 強度はチェーン全体で共通か、クリップ毎に変えられるか。** v1 は「チェーン全体共通」を推奨（実装がシンプル・`GenerateChainRequest.loras` はリクエストレベルの1リストで足りる）。クリップ毎にしたい場合は `ChainClip` 側にも `loras` を持たせる設計が必要になり、スコープが広がる。
2. **reference 付き IC-LoRA（参照動画による条件付け・`reference_video_id`）は chain に未実装。** 今回のスコープに含めるか、v1 はスタイル/キャラクター系 LoRA（reference 不要）のみとして reference 付きは別途スコープ外にするか、要合意。

#### テスト・検証の指針

- スキーマ疎通（`GenerateChainRequest.loras` の受理・バリデーション・省略時 byte 同一）は mock backend で pytest 化できる。
- 実際に LoRA が chain の生成結果に効いているかの GPU 実生成検証（stale LoRA が残っていないかの確認含む）は、**オーナー立ち会いで実施**すること（このプロジェクトの運用ルール＝目視検証はユーザーが行う）。

---

## ▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-07-06 **GUI プロンプト欄の一本化 ＋ ネガティブ欄グレーアウト**＝実装完了・客観ゲート PASS・**目視／実機ゲート✅クローズ・main マージ＆push 済**）（歴史記録）

> **（歴史記録）本ブロックは上位の「2026-07-11 Gradio WebGUI 大規模改修」ブロックに置き換わった。** 食い違ったら最新ブロックが正。

| 項目 | 状態 |
|---|---|
| リポジトリ | **main マージ＆push 済（2026-07-06・ユーザー承認）**・実装 `fd35877`＋docs。回帰 PASS（pytest **422 passed / 1 skipped**）。**画風/キャラ LoRA 対応（§29/§30）も目視ゲート✅クローズ・マージ済**。作業ツリー clean |
| **実装サマリ** | GUI のメインプロンプト欄を一本化＝Generate の `prompt` と Clip Chain の共通 `chain_prompt` を、タブ外・タブ群の上に置く**単一「下書き」欄**へ統合。方式＝下書き欄を**唯一の本物**にして両生成ハンドラへ**直結**（隠し欄＋コピーはしない＝残留・順序事故を構造的に回避）。**変更4ファイル**＝①`gradio_ui/ui.py`（上部共通バー直後・`gr.Tabs()` 直前に3行 Textbox〔下書き欄〕を新設・旧 `prompt`／`chain_prompt` 削除・両生成ボタンと Style gallery 選択を下書き欄へ直結）②`gradio_ui/handlers.py`（`make_chain_handler` で共通プロンプトの `<lora:...>` を除去＋警告・トークン無しは payload byte 同一・各クリップ個別プロンプトは除去せず）③ネガティブ欄 `negative`／`chain_negative` を `interactive=False`＋info（蒸留 CFG＝1 で無効・将来の非蒸留対応モックとして残置・ペイロード不変）④`gradio_ui/i18n.py`（文言更新・info／キー新設／旧キー削除 EN/JA）。**GUI のみ（API/engine/services 不可触）**。**入口＝[`PROMPT_UNIFICATION_WORKORDER.md`](PROMPT_UNIFICATION_WORKORDER.md)・詳細＝VERIFICATION_LOG §31** |
| **使い方** | プロンプトは上部の**単一欄**に書く（Generate／Clip Chain 共通）。Clip Chain では各クリップ枠が空ならこの共通プロンプトを流用する。LoRA サムネイルをクリックすると上部欄に `<lora:名前:1.0>` が追記される。連結タブでは LoRA タグは無視（除去＋警告）＝連結は LoRA 未対応 |
| **目視ゲート** | **✅クローズ（2026-07-06）**: ①プロンプト一本化の実機 4 点（下書き欄1つ・全タブ表示／Style LoRA 選択→上部欄タグ／連結で LoRA タグ除去警告／ネガティブ欄グレーアウト）②画風/キャラ LoRA のスパイク動画 6 本（`dspike_out`）を、いずれもユーザーが確認し「問題ない・完璧」と受容 |
| 残＝ユーザー宿題（次をブロックしない） | 前回持ち越しの GUI 実機確認 3 点（事前チェック拒否の黄トースト／チェーン進捗「クリップ n/N」／クリップ別プロンプトでクリップ2の実在目視）＝**この承認とは別・ユーザーが後日実施（継続）** |
| **次セッション** | **AviUtl2 拡張フロントエンド（Phase 2＝最終目的・spec §13.3／§14）**。ユーザーは Claude Design で UI/UX 叩き台を作成予定＝ブリーフ `Docs/AVIUTL2_DESIGN_BRIEF.md`（自己完結・push 済）。着手の入口＝AviUtl2 SDK の「拡張が取れる UI 形態」の確定スパイク（要調査＝ブリーフ §7） |
| 研究課題（格下げ済み・AviUtl2 統合より後） | ①**連結タブで LoRA を効かせるバックエンド改修**（`GenerateChainRequest.loras` 加算＋worker 配線・全体 or クリップ毎）＝今回のワークオーダー §5 ②negative/CFG の worker 配線＆API 露出（非蒸留 dev 向け）③Sulphur-2 distilled の GGUF 化。詳細＝下の（旧最新）ブロックのバックログ行 |
| pytest 基準 | **422 passed / 1 skipped**（§31 時点） |

---

## ▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-07-06 画風/キャラクター LoRA 対応 S1+S2 実装完了・回帰全 PASS・**main マージ＆push 済（ユーザー承認）・最終目視ゲートは未完了＝持ち越し**）（歴史記録）

> **（歴史記録）本ブロックは上位の「GUI プロンプト欄の一本化」ブロックに置き換わった。** 未完了の目視ゲート宿題は最新ブロックの「引き継ぐ宿題」に集約済み。食い違ったら最新ブロックが正。**詳細＝VERIFICATION_LOG §29／§30。**

| 項目 | 状態 |
|---|---|
| リポジトリ | **main マージ＆push 済（2026-07-06・ユーザー承認・目視未完了を承知の上での承認）**。中身＝`bcdde05`（S1 バックエンド）→`aed220b`／`591d4be`（S2 GUI）＋docs（`34e5b41` ほか）。回帰は全 PASS |
| 実装サマリ | 画風/キャラクター LoRA 対応の要件3点を S1（バックエンド）＋S2（GUI）で実装。**S0 スパイク＝GO（§29）→S1+S2＝実装完了（§30）**。①`models/loras/` にディレクトリスキャン＋登録制のマージ（衝突は config 勝ち）②kind 判定（メタ／preprocess で control か style か）③alpha/rank を strength に自動畳み込み（weight 1.0＝学習想定・不可触機構の外側）④all-or-nothing 撤廃＋control かつ参照なしは 422 `LORA_REQUIRES_REFERENCE`⑤新設 `GET /loras`／`POST /loras/reload`／`GET /loras/{name}/thumbnail`⑥GUI＝プロンプト内 `<lora:名前:weight>` パース（トークン無し時は payload byte 同一）＋「Style LoRA」サムネイルタブ（クリックでコマンド自動入力・Reload ボタン）。詳細＝**VERIFICATION_LOG §29（S0）＋§30（S1/S2）** |
| **使い方** | `models/loras/` に `.safetensors` を置く → GUI の「Style LoRA」タブで Reload → サムネイルをクリックするか、プロンプトに手書きで `<lora:名前:weight>` を書く。weight 1.0＝学習が想定したとおりの効き（alpha を自動で畳み込む）。制御 LoRA（canny／pose／upscaler）は従来どおりアダプタ欄から使う |
| **残＝ユーザー宿題** | **最終目視ゲート＝未完了（ユーザーが外出先から「目視未完了のままマージしてよい」と承認・2026-07-06）**。宿題＝①S0 スパイクの動画 6 本（セッション scratchpad の `dspike_out`・720p 4本＋スモーク）の目視 ②GUI 実機操作（Style LoRA タブ／Reload／クリック挿入）の確認。§30.5 OPEN のまま持ち越し。問題が見つかったら通常の不具合改修として扱う |
| 残る宿題（次をブロックしない） | 前回持ち越しの GUI 実機確認 3 点（事前チェック拒否の黄色トースト／チェーン進捗「クリップ n/N」／クリップ別プロンプトでクリップ2の実在目視）＝**残（ユーザーが後日実施すると表明・2026-07-06）** |
| バックログの正本 | 「2026-07-05 audio-to-video 完結」ブロックの**バックログ（三次トリアージ済み）行**＝近い将来（GUI V2V/A2V 露出※・モデル管理※・ログ改修※・**IC-LoRA strength＝§28 で実施済み**）／将来改修／研究課題／不要。※印3件は §26 で実施済み。IC-LoRA の重ね掛け（多重制御）＝**研究課題**（メモ＝`IC_LORA_PHASE_C_STATUS.md` 末尾・公式は単一制御が流儀・機構は複数可）。**注記**: negative／CFG／pipeline は worker 未配線＝GUI 露出禁止（継続） |
| **研究課題へ格下げ（ユーザー決定 2026-07-06）** | 次の2件は将来の研究課題（AviUtl2 統合より後・着手は独立セッションで）: **①negative／CFG の worker 配線＆API 露出**＝非蒸留（dev 系）モデル向け。リサーチ結論（2026-07-06）＝非蒸留ユーザーの本流は「公式 distilled LoRA（`ltx-2.3-22b-distilled-lora-384.safetensors`・rank384）を重ねて 8step/CFG1.0 化」であり CFG 配線の需要は限定的。CFG 有効時は cond+uncond をバッチ連結＝ピーク VRAM 増・dev 実践値 25〜35step で 720p 5秒≈20〜30分・ComfyUI 公式テンプレも cfg=1 出荷。着手するなら「ステップ可変＋CFG＋negative」の3点1スライス（dev GGUF は QuantStack に Q4_K_M 17.8GB が実在＝本機で検証可能）。**②Sulphur-2 distilled の GGUF 化と実行テスト**＝CivitAI のファインチューン。**Distilled 版が公式に存在**（`sulphur2Base_distilled.safetensors`・蒸留 LoRA マージ済み）＝自前蒸留は不要。経路＝HF の bf16 版（CivitAI の fp8mixed は変換ソースに不適）→ city96 `convert.py`＋`llama-quantize`（CPU/RAM 作業・目安 RAM~44GB・一時ディスク~90GB・学習不要）→ 既存モデル管理（`GET /models`＋`POST /pipeline/load`）で切替＝バックエンド無改修。要検証3点＝キー互換（Dスパイクの型）／LTX-2.3 の 5D テンソル後処理の要否／llama-quantize のパッチ済みビルド |
| pytest 基準 | **418 passed / 1 skipped**（§30 時点・§28 の 365+1 → S1 +26・S2 +27） |

---

## ▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-07-06 IC-LoRA strength 可変化 完結＝目視受容・**main マージ＆push 済**）（歴史記録）

> **（歴史記録）本ブロックは上位の「2026-07-06 画風/キャラクター LoRA 対応」ブロックに置き換わった。** 食い違ったら最新ブロックが正。**詳細＝VERIFICATION_LOG §28。**

| 項目 | 状態 |
|---|---|
| リポジトリ | **main マージ＆push 済（merge `e94559e`・2026-07-06・ユーザー承認）**。中身＝`2a2bd06`（S3 GUI）→`3e5f367`（S1 API）→`e2039f6`（S2 engine）→`75b4d46`/`b8b33e6`（docs）。マージ後 pytest 緑（365/1）。feature ブランチは削除済み |
| 実装サマリ | 「制御にどれだけ従わせるか」を可変化。**②`conditioning_attention_strength`（本命ノブ・上流未配線だったものを DistilledPipeline 経路へ新配線）**＋**①`reference_video_strength`（送信側の固定解除）** を optional 2フィールドで加算（**省略時＝byte 同一**）。3層＝API（`api/models.py`）／engine（`engine/worker.py`＋`fast_video_pipeline.py`・<1.0 のみ `ConditioningItemAttentionStrengthWrapper` で包む・IC-LoRA 重みパッチ機構は不可触）／GUI（Generate タブにスライダー2本・既定 1.0・値<1.0 のみキー送出）。目視＝**✅ユーザー受容（2026-07-06・「参照動画の動きを反映した生成」を確認）**。詳細＝**VERIFICATION_LOG §28** |
| 次セッション（実施済み・歴史記録） | 画風/キャラクター LoRA 対応＝**実施済み**（最新ブロック参照・§29／§30） |
| pytest 基準 | **365 passed / 1 skipped**（§28 時点・旧 354→+11＝S1 6本・S2 2本・S3 3本。マージ後再確認済み） |

---

## ▶▶▶▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-07-06 push 完了・次セッション＝IC-LoRA 制御の効き具合（strength）可変化（ユーザー決定））（歴史記録）

> **（歴史記録）本ブロックは上位の「2026-07-06 IC-LoRA strength 可変化」ブロックに置き換わった。** ここで「次セッション」に挙げた IC-LoRA strength 可変化は**実施済み**（客観ゲート全 PASS＝VERIFICATION_LOG §28・目視／マージはユーザー承認待ち）。これ以降の▶節はすべて歴史記録。食い違ったら最新ブロックが正。

| 項目 | 状態 |
|---|---|
| リポジトリ | **main＝origin 同期済（`3778113`・2026-07-06 push・ユーザー承認）**。作業ツリー clean・worktree は本体のみ（旧 wtA/wtB/wtC は撤去済みと `git worktree list` で確認済） |
| **次セッション** | **IC-LoRA 制御の効き具合（strength）可変化（ユーザー決定 2026-07-06）**＝現在コードに固定値で埋め込まれている「制御（輪郭線 canny／骨格 pose）にどれだけ厳密に従わせるか」を調整可能にする。3層＝①エンジン配線 ②API に optional フィールド加算（**省略時＝現行既定・byte 同一**の定型ゲート）③GUI スライダー。入口＝[`IC_LORA_PHASE_C_STATUS.md`](IC_LORA_PHASE_C_STATUS.md) スコープ外節・進め方の型＝リサーチ→設計→ユーザー合意→（要すればスパイク）→スライス実装→回帰ゲート |
| 残る宿題（次セッションをブロックしない） | ユーザーの GUI 実機確認 3 点（①事前チェック拒否の黄色トースト ②チェーン進捗「クリップ n/N」③クリップ別プロンプトでクリップ2の実在目視）＝**ユーザーが後日実施すると表明（2026-07-06）** |
| バックログの正本 | 「2026-07-05 audio-to-video 完結」ブロックの**バックログ（三次トリアージ済み）行**＝近い将来（GUI V2V/A2V 露出※・モデル管理※・ログ改修※・**IC-LoRA strength←今回昇格**）／将来改修／研究課題／不要。※印3件は「GUI V2V/A2V 露出＋モデル管理＋ログ改修」セッションで**実施済み**（VERIFICATION_LOG §26） |
| pytest 基準 | **354 passed / 1 skipped**（§27 時点） |

---

## ▶▶▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-07-05 GUI 経由クリップ連結＋結合の症状解消 完了・目視ゲートもクローズ・branch `fix/v2v-join-silent-loudnorm`）（歴史記録）

> **（歴史記録）本ブロックは上位の「2026-07-06 push 完了」ブロックに置き換わった。** 本文の OPEN 2 件は解消済み＝push/main マージ実施済（`3778113`）・GUI 実機確認 3 点はユーザーが後日実施。これ以降の▶節はすべて歴史記録。食い違ったら最新ブロックが正。詳細＝**VERIFICATION_LOG §27**。

本セッション＝§26.4 の持ち越し（GUI 経由クリップ連結＋結合の症状）の解消。

### 完了サマリ

- **H1 修正（本命・確定）**: 元動画の音声がデジタル無音のとき、`join_v2v` の2パス loudnorm が実測 `-inf` を第2パスに無ガードで渡し ffmpeg が拒否→503。対処＝実測値が非有限/範囲外（目標側 I=[-70,-5]・実測側 [-99,0]）なら**音量整合をスキップ**（`loudness_matched=false`・クロスフェードは実施・警告ログ＋内部 `loudness_skip_reason`）。修正前に実素材で再現→修正→同手順成功の順で検証。凍結 API（`JoinResponse`）無変更。
- **テスト**: anullsrc（無音ストリーム）フィクスチャ＋3本追加（無音 source／無音 continuation／handle 経路×無音 source）。
- **H2**: 結合版は手動ボタン＋503 失敗の複合＝「作られない」に見えた→コード修正で解消。
- **H3**: クリップ2は生成済み・出力は新規部分のみ 152f≈6.3 秒＝**仕様**（参照フレーム 73 がクリップ1の頭を置換）。GUI の V2V 説明文（日英）に「出力の長さ≈（総フレーム数−参照フレーム数）÷24 秒」を追記。
- **実機 e2e（PASS）**: 新ジョブ `e4a34bf5-941f-4ed5-a9cc-ed6d82783964`＝無音 16fps 素材→V2V チェーン生成（2×121f・312.66s・peak_vram_mb **9889** 維持）→join **HTTP 200（503 解消）**・joined.mp4＝21.54 秒 517f・音声あり・`loudness_skip_reason` はレスポンス非露出（凍結 API 無変更の実証）・503/ERROR/Traceback なし。
- **ユーザー目視ゲート（2026-07-05 夜）＝クローズ**: ①output.mp4（`03c0a691`・新規部分 6.3 秒）②joined.mp4（`e4a34bf5`）とも「問題なし」→ **H3＝仕様で最終確定**。
- **GUI 再検証で判明した追加事象と対処（詳細＝§27.9）**: V2V で「Generate chain」が無反応に見えた原因＝**共有プロンプト空欄**の事前チェック早期 return（テキスト欄 1 行のみで見落とし・コード退行なしを git 履歴突き合わせで確認）。プロンプトを入れた再操作ではチェーン job `9b4a184d`（2×121f・314.5s・peak 9889）＝**両クリップ生成→join→GUI 内の結合版プレビューまで全工程成功**。対処＝チェーン生成の事前チェック全 22 箇所の拒否を**トースト警告（gr.Warning）＋テキスト欄の二重表示**に（`5557dcb`）。
- **チェーン進捗のクリップ位置表示（ユーザー要望・詳細＝§27.10）**: 「今いくつめのクリップまで進んだか」が GUI で分からなかった→ `JobResponse` に optional `clip`/`clip_count` を加算（凍結 API の加算的変更・F3 の `stage` と同型）し、GUI 進捗テキストを「生成中… 20% (step 3/8) — デノイズ中 (stage 1) — **クリップ 1/2**」形式に。完了時は「全 N クリップ処理済み」。クリップごとの VAE デコードプレビューは**不採用（ユーザー判断＝デコードの無駄）**。実機 e2e PASS: 2 クリップチェーンで clip が None→1/2→2/2 と遷移・stage-2/デコード/完了まで 2/2 保持・従来コンソール行不変・**peak_vram_mb 8440（チェーン回帰基準値と一致）**。
- pytest 基準の新値＝**354 passed / 1 skipped**（343+1 → +11）。既存有音経路・単発生成経路テスト緑＝非退行。
- コミット列（branch `fix/v2v-join-silent-loudnorm`・計9）: `c7ad906`（fix: 無音 loudnorm）→`6ba83ae`（test）→`1422543`（i18n 尺説明）→`2322ac8`（docs §27）→`5557dcb`（GUI トースト）→`395a45b`（docs §27.9）→`8bfd559`（clip 進捗: サーバー側）→`338519f`（clip 進捗: GUI 側）→docs 追記コミット。

### OPEN（次のアクション）＝すべて解消済（最新ブロック参照）

1. **ユーザーの GUI 実機確認（3 点）**＝ユーザーが後日実施すると表明（2026-07-06）。①共有プロンプト空欄→黄色い警告ポップアップ ②進捗テキストに「クリップ n/N」 ③クリップ1/2 に別々の個別プロンプト→出力後半で内容切替＝クリップ2の実在確認。
2. **push／main マージ**＝実施済（`3778113`・2026-07-06 ユーザー承認）。

---

## ▶▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-07-05 GUI V2V/A2V 露出＋モデル管理＋ログ改修 完了・mainマージ済）（歴史記録）

> **（歴史記録）本ブロックは上位ブロックに置き換わった。** 本文の「★次セッション＝GUI 経由クリップ連結＋結合の症状解消」は**実施済み**（結果＝§27 ブロック）。食い違ったら最新ブロックが正。**詳細＝VERIFICATION_LOG §26。**

Fable5 親＋Opus 子の並行オーケストレーション（フェーズ1＝3子・フェーズ2＝1子・各自 worktree・GPU 所有権は親が逐次貸与）で 3 題目を実装し G3 目視→フィードバック対応→再ゲートまで通した。

- **実装3題目**（正本＝VERIFICATION_LOG §26）: ①コンソールログ改修（子C・httpx ポーリング行 68%→0・byte-match 3系統一致・peak_vram **8440** 再現）②モデル管理ドロップダウン（子B・正本=[`MODEL_MANAGEMENT_DESIGN.md`](MODEL_MANAGEMENT_DESIGN.md)・swap→同 seed→SHA 一致×3）③GUI V2V/A2V 露出＋結合 API（子A・正本=[`mockups/JOIN_API_PROPOSAL.md`](mockups/JOIN_API_PROPOSAL.md)）＋G3 フィードバック F1-F5。
- **新設 API**: `GET /models`・`POST /pipeline/load`（optional モデル指定・省略時 byte 同一）・`POST /jobs/{id}/join`・`GET /jobs/{id}/joined`（`services/join_manager.py`）。F1-F5 要点＝F1 uvicorn ポーリング GET 抑制／F2 wheel の tqdm を `engine/progress_shim.py` へ実行時差し替え（wheel 無改変・数値非干渉）でステップ進捗／F3 GUI 進捗整形（`JobResponse` に optional `stage`）／F4 V2V 継続の作法ガイド／F5 クロスフェード既定 150→**300ms**＋GUI ドロップダウン 150/300/500。pytest **343 passed / 1 skipped**。
- **ユーザーゲート（クローズ）**: A2V（GUI 経由・リップシンク含む）/i2v/joined.mp4/ログ/GUI 進捗＝すべて完璧。**A2V 音量 +3dB＝違和感なし→研究課題へ格下げ（ユーザー決定 2026-07-05）**。**モデル管理の実代替モデル DL＝しない方針（ユーザー決定）**＝切替配線は別名二重登録で実証済み・実重み投入は将来ユーザー任意。

---

## ▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-07-05 audio-to-video 完結・mainマージ済）（歴史記録）

> **（歴史記録）本ブロックは上位ブロックに置き換わった。** 食い違ったら最新ブロックが正。**詳細＝VERIFICATION_LOG §25（判定詳細＝§25.5）。**

- **audio-to-video ✅完結（全客観ゲート PASS＋G3 試聴＝条件付き受容 2026-07-05・main マージ/push 済）**。`POST /upload/audio` 新設＋`POST /generate/chain` に optional `source_audio{audio_id}`（加算的・省略時 byte 同一）。機構＝音声 latent を全長ハード凍結＋動画のみ denoise・出力は元波形 mux（vocoder 不使用）。v1 スコープ＝1クリップ・A2V×V2V 排他・トリミング非露出・短い音声 422・`conditioning_images` 併用可。**正本: 設計=[`A2V_DESIGN.md`](A2V_DESIGN.md)・G1〜G3=VERIFICATION_LOG §25**。pytest **212 passed / 1 skipped**。
- **リップシンク深掘り ✅クローズ（研究課題へ格下げ・ユーザー 2026-07-05）**＝マトリクス試聴でシーン要因（動き・画角）で確定＝モデル性質（クローズアップ×単一話者＝高精度・複雑シーン＝大ズレ）。リサーチ=[`A2V_LIPSYNC_COMMUNITY_RESEARCH.md`](A2V_LIPSYNC_COMMUNITY_RESEARCH.md)・判定=VERIFICATION_LOG §25.5.3。運用指針＝A2V は「クローズアップ・単一話者・動き控えめ」で使う機能として案内。
- **GUI 目視ゲート（旧宿題）✅クローズ（ユーザー実施 2026-07-05）**: 表示崩れなし・ENG/JPN・Dark/Light 切替 OK・動画生成＆GUI 内再生 OK。
- 用語: **「元音声」「元動画」**＝ユーザーがアップロードする入力素材（旧表記「源音声／源動画」は同義）。

以下2行＝**最新ブロックから参照される正本（文言保持）**:

| **バックログ（ユーザー処置 2026-07-05・三次トリアージ済み）** | **近い将来にやる**: ①GUI への V2V/A2V 露出（音声スムージング UI 要件含む） ②モデル管理（**ドロップダウン選択のみ・ディレクトリ再編なし・選択対象=本体/テキストエンコーダ/VAE/音声系**。正本=[`MODEL_MANAGEMENT_FUTURE_WORKORDER.md`](MODEL_MANAGEMENT_FUTURE_WORKORDER.md) 冒頭★注記2つ）＋**コンソールログ改修**（調査=[`CONSOLE_LOG_FORGE_NEO_RESEARCH.md`](CONSOLE_LOG_FORGE_NEO_RESEARCH.md)） ③**IC-LoRA 制御の効き具合（strength）可変化**。**将来の改善事項（一旦クローズ）**: 高品質モード実配線（D節）・IC-LoRA 他制御タイプ（depth/Motion-Track 等）・A2V×V2V 併用・**GUI 細目**（言語/テーマ永続化・動的行・デフォルト negative 欄＝使い勝手改修のため格下げ・ユーザー決定 2026-07-05）。**研究課題（改善事項より低優先）**: リップシンク強化（VERIFICATION_LOG §25.5 保留）・V2V 音声継ぎ目の浅い凹み（§24.7）・IC-LoRA 前処理キャッシュ・A2V 複数クリップ音声窓割り。**不要と確定**: A2V 音声トリミング指定（AviUtl2 と機能重複） |
| **次セッション計画（ユーザー提案 2026-07-05）（実施済み・歴史記録＝最新ブロック参照）** | **①GUI V2V/A2V 露出と②モデル管理+ログ改修を並行実施**: Fable 5=親（監督・統合・ゲート裁定）、Opus 子エージェント2人が各題目を担当、**子には孫/曾孫の起動権限を付与**。ガードレール（監督案）: (a) 子は各自 worktree ブランチで作業し親が順にマージ（両題目とも `gradio_ui/` に触れるため衝突は親が解消） (b) **GPU ジョブの所有権は常に1つ**＝親が貸与/回収（並行 GPU 実行禁止・従来規律の維持） (c) GUI V2V/A2V は実装前に画面モックでユーザー仕様合意（前回 GUI セッションの型） |

- ※印3件（GUI 露出・モデル管理・ログ改修）は「2026-07-05 GUI V2V/A2V 露出＋モデル管理＋ログ改修」ブロックで**実施済み**（VERIFICATION_LOG §26）。IC-LoRA strength は次セッションへ昇格（最新ブロック）。

---

## ▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-07-05 video-to-video 継続 完結・mainマージ済）（歴史記録）

> **（歴史記録）本ブロックは上位ブロックに置き換わった。** 食い違ったら最新ブロックが正。**詳細＝VERIFICATION_LOG §24（音声接続の知見=§24.7）。**

**video-to-video 継続 ✅完結（客観ゲート全 PASS＋G3 試聴 PASS）**（merge `18296b2`・`origin/main` 同期・マージ後 pytest 緑）: `POST /generate/chain` に optional `source_video{video_id, context_frames=73}`（加算的・省略時 byte 同一）＝元動画の続きを映像+音声で生成・出力は新規部分のみ・fps 自動リサンプル。音声結合＝二段構え（既定フェードペア＋オプトインのハンドル真クロスフェード `join_v2v(handle_audio=…)`・エンジンが `<stem>_audio_handle.wav` サイドカー出力）。同梱の重要修正＝①クリップ長不揃いチェーンの潜伏クラッシュ（`e8557cb`）②WDDM ページ降格カスケード是正（`b2c20ee`・共有溢れ 12.5GB→ゼロ・2.09× 高速化・720p/257f spill-free 化）③GUI Settings spill 表の無限読み込み（`b95a38c`）。回帰＝T2V/I2V/同一シード V2V byte 一致・pytest 191 passed/1 skipped。**正本: 設計=[`V2V_CONTINUATION_DESIGN.md`](V2V_CONTINUATION_DESIGN.md)・VERIFICATION_LOG §24・音声知見=[`V2V_AUDIO_JOIN_RESEARCH.md`](V2V_AUDIO_JOIN_RESEARCH.md)**。

---

## ▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-07-04 検証用 Gradio GUI 4タブ再構築）（歴史記録）

> **（歴史記録）本ブロックは上位ブロックに置き換わった。** 食い違ったら最新ブロックが正。**詳細＝VERIFICATION_LOG §23。**

旧 `gradio_ui.py`（Phase 1 初期・186行）を Phase 3／IC-LoRA B/C を検証できる **4タブ構成**（`gradio_ui/` パッケージ 9モジュール＝Generate/Clip Chain/Jobs/Settings＋上部ステータスバー）へ作り替え。事前チェック（÷128・拡張子・サイズ・overlap／総フレーム＝サーバーと同じ `chain_math` 流用）とエラー整形（`format_api_error` 15コード）を UI 側に実装＝**凍結 API の薄いクライアント**に徹する（AviUtl2 統合の予行）。進め方＝HTML モックアップでユーザー仕様合意→実装（既定英語・ダーク既定・Generate は右カラム）→Opus サブエージェントで完全性監査（9件反映）。客観ゲート PASS（pytest 147 passed/1 skipped）。同時に**本家 LTX Desktop との機能比較リサーチ**でユーザーが優先順位決定（V2V=当時の次セッション／A2V=近い将来／静止画・プロンプト強化=不要／高品質モード=遠い将来）＝[`FEATURE_RESEARCH_2026-07-04.md`](FEATURE_RESEARCH_2026-07-04.md)。**目視ゲートはユーザーが後日実施**（客観PASS≠目視の規律）。正本＝VERIFICATION_LOG §23。

---

## ▶▶▶▶▶▶▶▶ 最新ステータス（2026-07-03〜04 IC-LoRA Phase A/B/C＋Phase 3 スライス1/2）（歴史記録）

> **（歴史記録）本ブロックは上位ブロックに置き換わった。** 食い違ったら最新ブロックが正。**詳細＝VERIFICATION_LOG §17/§19/§20/§21/§22。**

- **IC-LoRA Phase C ✅完了・全ゲート PASS（G5=ユーザー受容 2026-07-04・merge `8bdf90a`）**: 制御系アダプタ=LTX-2.3-22b **Union-Control** 一本（654MB→`models/ltx-2.3-ic-lora/union-control/`）＋engine 内前処理段 canny/DWPose（TorchScript 版・追加 pip 不要）新設＝「動き維持で内容置換」成立。**÷128制約**（参照付きジョブは出力 w/h が 128 の倍数・422 事前検出）に注意。正本=[`IC_LORA_PHASE_C_STATUS.md`](IC_LORA_PHASE_C_STATUS.md)（**スコープ外節＝次セッションの strength 可変化の入口**）・VERIFICATION_LOG §22。
- **IC-LoRA Phase A ✅**（bf16 融合スパイク=§20）／**Phase B ✅**（forward 時 GPU LoRA 適用＝per-layer-quant 本番経路・VRAM 増ゼロ・ジョブ毎切替可＋API 露出＝`loras`/`reference_video_id`/`POST /upload/video`・基準 SHA=`735a6de9…272`=§21）。正本=[`IC_LORA_PHASE_B_STATUS.md`](IC_LORA_PHASE_B_STATUS.md)。
- **Phase 3 スライス1 多キーフレーム誘導 ✅**（merge `7f31935`・目視受容 2026-07-03＝「途中キーフレームは磁石・間の遷移はプロンプト支配」を仕様として受容）=§17／**スライス2 クリップ連結 ✅**（masked AV-latent 連結で再実装・merge `2cc4cac`・目視/試聴 PASS＝720p 級2セグで継ぎ目不可視まで実証）=§19（旧 latent-extend 方式は境界 hard cut・音声断絶 FAIL=§18）。正本=[`PHASE3_CLIP_CONCAT_STATUS.md`](PHASE3_CLIP_CONCAT_STATUS.md)・設計=[`PHASE3_CLIP_CONCAT_DESIGN.md`](PHASE3_CLIP_CONCAT_DESIGN.md)。ユーザー向け解説=[`FEATURE_GUIDE_KEYFRAMES_AND_ICLORA.md`](FEATURE_GUIDE_KEYFRAMES_AND_ICLORA.md)。

---

## ▶ 基盤アーカイブ（2026-06-28〜07-02・16GB fit 達成／de-fork／QAT 回収 ほか）（歴史記録）

> **（歴史記録）以下は 16GB バックエンドの土台を作った完結済みマイルストーンの要約。** 実測・経緯の正本は VERIFICATION_LOG（§番号）。食い違ったら冒頭の最新ブロックが正。

### 基盤マイルストーン（すべて完了・mainマージ済）

- **720p 達成＋連続生成 commit 枯渇解決**（2026-06-30）: 1280×768 二段を本番 API で完走（~167–171s・RTX 4070 Ti SUPER 16GB）→任意で 1280×720 crop。連続マルチジョブも commit 枯渇せず PASS。レシピ＝comp=1（component-files/Path B）＋`LTX_KEEP_RESIDENT=0`＋bs=8＋vae 512/64。正本=VERIFICATION_LOG §10・[`SCALEUP_16GB_RESEARCH.md`](SCALEUP_16GB_RESEARCH.md)。解像度×尺の能力＝[`RESOLUTION_DURATION_CAPABILITY.md`](RESOLUTION_DURATION_CAPABILITY.md)（尺 cap 257→481f(20s) 緩和・解像度別 spill-free frames=720p:257/1080p:153/1440p:81 を `/api/v1/config` の `limits.spill_free_frames` に露出）=§10.6。
  - ※`LTX_KEEP_RESIDENT` は当時の手順。2026-08-02 に環境変数の経路は撤去され、現在は API の `keep_resident` フィールド（`POST /generate`・`POST /generate/chain`。既定 `false`＝keep=0 と同じ状態）で指定する（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §48）。
- **I2V マルチジョブ＋音声 連続 実機 PASS**（2026-07-02）=§10.7。**音声 Phase 1**（native joint audio・AAC/48kHz/stereo・crop 後も保持）=§9.8。
- **load/encode 16GB fit**: `--te-offload`（Gemma encode ピーク低減・既定 ON）=§11／`--dit-cpu-load`（transformer load スパイク除去・既定 ON・transformer ブロックを直接 CPU 構築）=§12＝512×320 で whole-job ceiling 16,944→~9.2GB。残るのは den2（denoise stage2）の解像度×尺スケーリング軸のみ（バグでない・`RESOLUTION_DURATION_CAPABILITY.md §8` が正本）。
- **残課題A（worker 再利用 crash）解決**=§9.7: 真因＝`BlockSwapService._installed_transformers` の per-job リーク（append し続け解放しない）。修正＝install() で append 前に `clear()`（keep-latest）＋`_do_generate()` 完了直後に `gc.collect(); torch.cuda.empty_cache()`。6ジョブ実機 PASS・出力 byte 一致。真因確定の経緯（当初 mmap 破損説→棄却→Windows commit（仮想メモリ）枯渇で確定）=§8。
- **de-fork（engine first-party 化）**=§13: 旧同梱フォーク backend→`engine/` パッケージ（`worker`/`pipeline`/`gguf`/`gemma`/`transformer`）へ採用・アルゴリズム不変・43GB モノリス物理削除（`checkpoint_path` は reference-only 化）・全段 SHA256 一致。**QAT 回収**=§14: Gemma を text-only（`Gemma3ForCausalLM`・vision 無し）化し 22.7GB の QAT dir を物理削除・gemma_root は ~40MB tokenizer-only dir（`models/gemma-3-12b-it-tokenizer/`）に差替え・models/ 50.9→28.15GB・byte 完全一致。
- **keep=1 常駐モード＝調査 CLOSE**（16GB で原理的 non-viable・利得は既存経路で捕捉済み・既定 keep=0 不変）=§15。※この「keep=1 は non-viable」という結論は §47 のバグ修正と §48 の製品化で覆った。現在の正本は §48。**dead-code 整理**（text-only/de-fork の残 no-op 除去・byte-match ゲート）=§16。**Phase 5(A) 配線**（Approach W 常駐サブプロセスワーカー・`@@LTX@@` JSON-lines プロトコル・app `.venv` は torch 非 import）=§6・**Phase 5(B)**（denoise 直前 `empty_cache` で shared 溢れ 3969→742MB）=§7。
- **install スクリプト**: `scripts/install_ltx.ps1` は冪等クリーンインストーラ（2026-07-02 全面書き換え済＝現行の正・両venv・現行~30GBのみDL）。**モデル取得は 2026-07-26 に 3 リポジトリ化し、2026-08-03 現在は 5 回のダウンロード呼び出し・23 ファイル・32,912,043,043 B（約30.7GiB）**＝`Rootport/Nz-LTX23-weights`（LTX本体5点＋IC-LoRA 3点＋VDA深度前処理器2点）／`Rootport/Nz-Gemma3-12B`（11点）／`Rootport/Nz-DWPose`（前処理器2点・新設）。いずれも Public 非 Gated でトークン不要、内部構造が `models/` と 1 対 1 のため後処理なしで展開される。**従来インストーラ管理外だった `models/ltx-2.3-ic-lora/` と `models/preprocessors/` も自動取得の対象になった**（手動配置は不要）ため、step 6 の PASS/MISSING 検証表は 10 → **16 項目**（IC-LoRA 2＋DWPose 2＋Deblur 1＋VDA 1 を追加）。この 6 点は `_real_available()` の判定対象ではない＝欠けても mock に落ちず、`config.yaml` と `gradio_ui/adapters.py` が 5 アダプタを無条件に見せるため、選んだ瞬間 404 という分かりにくい壊れ方をする。だから検証表であえて名指しする。**冪等ガードは `-Check` による「ディレクトリ単位の独立判定」**＝`@{ Dir=…; Min=… }` を展開先ディレクトリの数だけ並べ、**全部がそれぞれ自分の `Min` を満たしたときだけ SKIP**（1つでも下回れば再DL）。**合計を単一しきい値と比べる旧 `-CheckDir`＋`MinBytes` は 2026-07-26 に廃止**＝大ファイル1つが丸ごと欠けた兄弟 dir を覆い隠すため（Gemma で実発現：7.3GB GGUF だけで合計を超え、トークナイザ dir 全欠落でも SKIP → 検証表 `gemma_root` MISSING → exit 1 → 再実行しても直らない復旧不能デッドロック）。各 `Min` は「**そのディレクトリ**の合計 −**そのディレクトリ内で step 6 の検証表が個別にゲートしている最小ファイル**」より上・想定合計より少し下に置く（＝検証表が MISSING にできる欠落は必ずガードも割る、という対応を作る）。検証表が個別に見ないファイル（`gemma_root` は dir 全体を 20MB で 1 行）は捕捉不要（狙うと 35 バイト幅の窓になり破綻）。`models/ltx-2.3-gguf/` は自家変換 GGUF が同居しサイズ判定が緩むが、これは検証表が受け持つ。`hf download --include` は **1 つの `--include` に全パターンを並べる**こと（`nargs="*"` のため flag を繰り返すと最後の組しか効かない。ただしこれは `huggingface_hub 0.36.2` 固有で、1.20.1 では逆に単一 flag 複数パターンが無視される＝venv の同パッケージを上げたら書き換えが要る）。**GPU アーキの自動判定（`nvidia-smi` 読み取り・`-GpuArch` 引数）は 2026-07-26 に廃止**＝全アーキで SDPA 固定のため分岐する理由が無く、アーキ別のプリビルド wheel 自動導入も行わない（`scripts/build_xformers.ps1` は手動ツールとして残す）。**エンジン venv の再同期を 2026-07-26 に追加**＝従来は `.venv-engine` が存在するだけでブロックごとスキップし、freeze が変わっても二度と適用されなかった（＝「`git pull` したら `setup.bat` 再実行」が機能しない）。現行は freeze 本文＋スクリプト内の3つの git rev（`diffusers`/`ltx-core`/`ltx-pipelines`）を連結した SHA-256 を `.venv-engine/.nz-engine-state`（追跡外・完了マーカー兼用）と突き合わせ、**一致するときだけ SKIP**する。マーカー不在は「再適用」に倒す（中断遺残と、この仕組み以前に作られた venv を救うため）。2段構えの適用処理（git pin 先行インストール → cu128 index ＋ `--index-strategy unsafe-best-match` で freeze 適用）は `Invoke-EngineFreezeApply` に切り出し、新規作成パスと再適用パスの両方から呼ぶ。`-RunSmoke` では `uv sync --extra dev` を明示（素の `uv sync` が optional の dev extra を刈り取り `.venv` から pytest が消えるため）。**エンドユーザー入口として `setup.bat` / `run.bat` を 2026-07-26 に新設**＝`setup.bat`→`scripts/setup.ps1`（前提ツール `uv`/`ffmpeg` を `tools/` へ取り込み・`config.yaml` を `.example` から複製・`logs/setup_<日時>.log` へ記録・事前チェックは警告のみ）→`install_ltx.ps1` を `&` で呼ぶ。`run.bat`→**リポジトリ直下の** `run.ps1`（移動禁止＝`$PSScriptRoot` 依存）。`.bat` は純 ASCII・CRLF・末尾 `pause` で日本語は `.ps1` 側（`.gitattributes` に `*.bat text eol=crlf`）。`config.yaml` は追跡外化し `config.yaml.example` を配布。互換メモ＝torch は必ず cu128 index から／attention＝**既定は SDPA・ジョブ単位で SageAttention を選べる**（xformers/flash-attn は引き続きインストールも import もしない。**2026-07-31 更新**＝以前ここに書いていた「SDPA 固定」「`sageattention==1.0.6` は死重依存」という記述は、次の2段階の変更で古くなった。①2026-07-28 の依存整理（`VERIFICATION_LOG.md` §40.1）で旧世代の `sageattention` 1.0.6 と `triton-windows` 3.6.0 を削除した。②2026-07-31 の Acceleration〔生成の高速化〕機能〔同 §43〕で、**実際の消費者があるものとして** `sageattention` **2.2.0**〔woct0rdho 版の Windows 用ビルド済み wheel を直リンクで固定〕と `triton-windows==3.5.1.post24` を標準同梱へ戻した。`sageattention` は `engine/venv-engine.freeze.txt` の直リンク pin 経由でのみ入り、`engine/engine-venv-pyproject.toml` の `dependencies` にも `[tool.uv.sources]` にも載せていない＝**`-ResolveLatest` を使った環境では動作保証外**。エンドユーザーに Visual Studio は要求しない〔triton-windows が TinyCC/ptxas を同梱〕。既定のジョブは従来どおり SDPA で、`sage` を選んだジョブだけが差し替わる）／Blackwell は R570+ ドライバ（全世代 R570+ 推奨）。

### 凍結してある契約・構成（不変・壊さない）

- **凍結 API 契約**: ÷64 解像度（`api/models.py`）・8n+1 フレーム・T2V/最小I2V（frame_idx0・conditioning≤1）・distilled 8step/CFG1.0・`GET /status` の `vram_optimization`（`services/low_vram.py` `_STATUS_KEYS`）・`metadata.json` スキーマ・limits/generation_presets。**加算的変更のみ許可**（optional フィールド・省略時 byte 同一が定型ゲート）。
- **2プロセス・2venv**: app=`./.venv`（torch 無し・FastAPI/Gradio/mock backend）／engine=`./.venv-engine`（torch 2.9.1+cu128＋`ltx_core`/`ltx_pipelines`@`00dc53d`＋`gguf`）。同一インタプリタで共存させない（双方が `services` トップレベルパッケージを持つため）。
- **本番 env**: `LTX_KEEP_RESIDENT=0` 既定（keep=1 だと 720p の Gemma 移動で native crash・§10.2）／`use_component_files: true`（Path B）／`te_offload`・`dit_cpu_load` 既定 ON。設定は `config.yaml` が正。起動＝`python -m engine.worker`（別プロセス・別venv・`cwd=root`）。
  - ※`LTX_KEEP_RESIDENT` は当時の手順。2026-08-02 に環境変数の経路は撤去され、現在は API の `keep_resident` フィールド（`POST /generate`・`POST /generate/chain`。既定 `false`＝keep=0 と同じ状態）で指定する（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §48）。
- **本番モデル（実行に要る ~28GB）**: GGUF transformer Q4_K_M ~17GB＋GGUF Gemma Q4_K_M ~7.3GB（`ggml-org/gemma-3-12b-it-GGUF`）＋component VAE/audio/projection ~3.9GB＋spatial upsampler ~0.95GB＋tokenizer-only gemma_root ~40MB。43GB モノリス・QAT dir は削除済。**インストーラはこれに IC-LoRA 2点（1.22GiB）＋DWPose 前処理器 2点（0.33GiB）＋Deblur 1点（0.91GiB）＋VDA 深度前処理器 2点（0.12GiB）を加えた ~33GB（30.7GiB）を取得する**（後者4種は `_real_available()` の対象外＝生成の中核ではないが、欠けると IC-LoRA 選択時に 404 になる）。
- **回帰基準 SHA**: T2V(seed=12345, "a calm ocean wave…") 512×320/49f=`23844b4e…6bb7bf`／最小I2V=`a511eda4…c217`／peak_vram_mb 8440。IC-LoRA 付き出力の基準 SHA=`735a6de9…272`（旧 `outputs/ic_lora_phaseA/spike.mp4` `8e10aa59…` は旧コード出力＝照合に使わない）。

### アーキテクチャ（二層・壊さない）

- **凍結層**: REST API 形（`api/models.py`）・ジョブ管理（in-memory 単一・実行中 busy 409）・出力 `outputs/{job_id}/output.mp4`+`metadata.json`（metadata は `pipeline_manager._finalize()` が書く）・`LTXRunner.generate(...)→GenerationOutcome` 契約・mock backend（torch 無し app `.venv` のテスト用・温存）。
- **エンジン層**: `engine/` パッケージ。公式 `DistilledPipeline`(ltx_core@00dc53d) をラップし GGUF transformer+block-swap・GGUF Q4 Gemma（per-layer dequant で GPU 推論）・VAE/attention tiling を配線。常駐 worker 経由で `services/ltx_runner.py` の `_RealBackend` から呼ぶ。

### 将来項目・未着手（現行スコープ外・記録のみ）

**2026-07-27の整理で、本節に列挙していた将来項目の管理は[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md)へ一本化した。** 現在の所在は次のとおり（同書は番号だけでなくファイル名を添えて参照すること）。

- **Phase 3 残パリティのうち、非蒸留(dev)モデル向けの生成つまみ一式**（negative／CFG／ステップ数／STG／sigma schedule／denoise loop／seed lock／延長尺~30s／空間アップスケーラのユーザー操作露出）＝同書**§4-1**。**negative／CFG／`pipeline` は worker 未配線＝GUI 露出禁止（継続）**という現行制約も同項へ転記済み。
- **高品質モード（`two_stage_hq`・pipeline/guidance_scale 消費）**＝同書**§4-28**（2026-08-04に§3-2から降格。着手の入口＝`services/ltx_runner.py` の payload 未配線箇所・非蒸留×量子化 dev 重み・計算コスト~7〜12倍まで転記済み。背景資料は[`FEATURE_RESEARCH_2026-07-04.md`](FEATURE_RESEARCH_2026-07-04.md) D節）。
- **Gap Fill**＝同書**§3-10**／**Retake・Inpaint**＝同書**§4-12**／**attention tiling の本番投入**＝同書**§4-2**／**VLM(vision) 再導入**（enhance_i2v・フレームを見た Gap Fill 提案。QAT text-only 化を巻き戻すため計画外）＝同書**§4-5**。
- **本節にのみ残る（台帳に未収録の）項目**: 生成キュー、text-only プロンプト強化（spec §13.4 由来）。いずれも**台帳へは起票しない**——前者は下記「やらない」のユーザー決定（「1ジョブ＋busy 409」が正しい設計）およびタイムライン側の「順番待ちは作らない（バグ温床）」決定（[`TIMELINE_ALPHA_REQUIREMENTS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/TIMELINE_ALPHA_REQUIREMENTS.md)）と衝突し、後者は[`FEATURE_RESEARCH_2026-07-04.md`](FEATURE_RESEARCH_2026-07-04.md) E節でユーザー決定「不要」と重複するため。
- **Phase 2 ＝ AviUtl2 拡張機能統合**（＝本プロジェクトの最終ゴール・「早く統合して使いながら育てる」early-integration 方針）＝当時はユーザーのプラグイン開発環境整備待ちでブロック中と記録していた（現在は別リポジトリ `Nz-LTX23-frontend-AviUtl2` で実装済み）。REST API は AviUtl2 専用にしない（DaVinci Resolve 等も想定）。spec §0.1/1.3。
- **やらない（削除済みスコープ・ユーザー決定・spec §13.5）**: ①1080p アップスケール「機能」（＝外部ツール推奨。内部二段 upsampler は生成の仕組みゆえ残す）②多人数インフラ（本格ジョブキュー/認証/インターネット公開/永続 DB＝単一ユーザー想定で不要・現状の「1ジョブ＋busy 409」が正しい設計）。

### 運用ルール（最新の正・memory にも記録）

- **目視検証**: 720p級（1280×768）以上＋映画トレイラー風プロンプト＋「賑やかな町＋セリフ」題材（512×320級は顔溶けで判断不能）。客観PASS（byte-match・pytest・メトリクス）とユーザー目視ゲートを混同しない。実験前に仮説→Web/コードで裏取り（手当たり次第禁止）。
- **サブエージェント**: Opus 以下（**Fable5 禁止**）・非破壊・**能動ポーリング監視**（ウォッチャー待ち停止禁止）・異常時は続行せず報告。GPU 計測は dedicated+shared 両監視（一次ソース＝`ltx_worker.log` の peak_vram_mb・道具=`_gpu_mem_sampler.ps1`）。
- **人間向け説明**: 内部略語禁止だが「普通のまともな日本語」で（過剰な言い換えも逆効果）。判断を仰ぐ前に機能説明を届ける。解説文書は Opus サブエージェントに執筆させ監督はレビュー。
- **作業原則**: コード着手前に実装計画を提示して合意・テストが落ちたら勝手に直さず原因分析して報告・先行事例を複製し独自発明しない・push/マージはユーザー承認・環境隔離厳守（システム Python 不可触・すべてプロジェクトローカル venv）。

### 旧・詳細設計/リファレンスのポインタ

- LTX 2.3 一般リファレンス=[`LTX23_REFERENCE.md`](LTX23_REFERENCE.md)（解像度契約 ÷32/÷64・2段・VAE 32×圧縮・VRAM 重み支配・720p=1280×768→crop）。設計比較=[`DESIGN_COMPARISON_and_direction.md`](DESIGN_COMPARISON_and_direction.md)（fp4_mixed 不採用・GGUF Q4 採用が結論）。公式 offload+fp8 戦略の破棄理由=[`note.md`](note.md)。設計正本=`../LTX23_Backend_Specification.md`（v0.5）。
