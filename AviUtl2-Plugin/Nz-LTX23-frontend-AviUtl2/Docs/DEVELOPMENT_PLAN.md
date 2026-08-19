# Nz-LTX23 AviUtl2 フロントエンド開発計画

最終更新: 2026-07-20(キーフレーム欄の大規模改修〔2026-07-18、DEVLOG §18/§20〕の実施およびバッチA2V周りの第4〜6弾ミニバッチ〔2026-07-18〜19、DEVLOG §26/§27/§28〕の完了を反映。2026-07-17時点の記録は大規模UIリデザイン〔2026-07-16〜17〕と実機フィードバック起点のデバッグ・UI改修〔2026-07-17、4波構成〕、IC-LoRA UI再設計〔2026-07-17、第5波〕、および`deploy.ps1`のwebuiコピー廃止=埋め込み専用運用への一本化〔2026-07-17、第6波〕の完了。2026-07-15時点の記録はフロントエンド追随グループ1〜3完了、およびパリティ実装 第1〜3陣・第4陣(前半N13/N4・後半N5/N8)完了=第4陣全完了) / 出典: 調査エージェント報告(2026-07-07)

**状態: 1.0.0-rc1 実装完了(2026-07-08)。リリースパッケージ: `dist/NzLTX23-1.0.0-rc1.au2pkg.zip`**。全マイルストーン(M1-0〜M7c)完了。開発経緯・実機検証結果・重要な発見は[DEVLOG.md](DEVLOG.md)、ブリッジ契約の最終形は[BRIDGE_CONTRACT.md](BRIDGE_CONTRACT.md)を参照。rc1後の追加セッションで右クリック生成の配線(ルーティング→プリフィル)を実装済み。本ページ内「1.0.0-rc1後の追加作業」節と[TIMELINE_ALPHA_REQUIREMENTS.md](TIMELINE_ALPHA_REQUIREMENTS.md)第7節を参照。

承認日: 2026-07-07
対象環境: AviUtl2 v2.0.54ポータブル(`D:\For_Videos\AviUtl2\aviutl2_v2.0.54\`。rc1開発当時はbeta52。Cドライブ容量都合でポータブル固定運用) / VS Community 2026 / WebView2

関連ドキュメント: [API_REFERENCE.md](API_REFERENCE.md) ／ [SDK_REFERENCE.md](SDK_REFERENCE.md) ／ [WEB_RESEARCH.md](WEB_RESEARCH.md) ／ [BRIDGE_CONTRACT.md](BRIDGE_CONTRACT.md) ／ [DEVLOG.md](DEVLOG.md)

---

## フロントエンド追随(グループ1〜3)完了(2026-07-15)

バックエンド側の正本ワークオーダー`Nz-LTX23-backend/Docs/FRONTEND_CATCHUP_WORKORDER.md`(2026-07-15起票)のグループ1〜3を全完了した。バックエンド(`Nz-LTX23-backend`)は引き続き一切変更していない(凍結方針を維持)。

- **グループ1(APIコントラクト追随)**: Clip Chainのクリップ上限を8→24に拡張、`chunked_upsample`(アップサンプル処理を小分けにしてVRAM使用量を抑えるオプション)トグルを追加(既定オンで常に明示送信)、チェーンLoRAのストリップ廃止(「chainにloras不可」という誤った前提の撤回)、chain参照動画(IC-LoRA)対応(α版、Clipsサブモード限定・排他制約込み)、尺の算術(オーバーラップ融合・context置換による減算を`chainUtils.ts`の純関数群へ転記)を反映した。
- **グループ2(機能パリティ)**: バッチA2V「就寝中一括生成」をCreate画面内の折りたたみ節として実装(CSVマニフェスト相互運用・原子的書込み・Done行保護マージ——**2026-07-18撤去・ステートレス化。CSVマニフェストによる中断再開機能自体が廃止され、フォルダスキャン→メモリ内行リスト→実行のみの設計に変更された。DEVLOG §26参照**)、モデル管理UI(`GET /models`/`POST /pipeline/load`)、Chainプリセットドロップダウン、wav長→Frames自動調整＋事前チェックを追加した。ブリッジ契約v6(`ui.pickFolder`/`fs.listFiles`/`fs.readTextFile`/`fs.writeTextFileAtomic`/`fs.probeAudioDuration`等——**`fs.readTextFile`/`fs.writeTextFileAtomic`は2026-07-18撤去・ステートレス化。§4.17/§4.18・DEVLOG §26参照**)をnative側(C++)に実装完了し、v6の「実装(C++側)を正とする」原則を回復した。
- **グループ3(設計判断項目＋品質負債)**: 単発A2V(音声から動画を生成する機能)をChainからCreate画面へ移設(Chainのサブモードは Clips/V2V の2択に整理)、Width/Heightスナップをgradio式(手入力は素通し・ステッパー/スライダー操作時のみ64または128の倍数に丸め)へ統一、README・webui README・CMakeLists.txtのバージョン表記更新、`scripts/deploy.ps1`/`scripts/run-aviutl.ps1`の既定パスをAviUtl2 v2.0.54ポータブル構成へ恒久修正、`Docs/REAL_BACKEND_CHECKLIST.md`を新設した。

**残課題**: 実バックエンド(実GPU)での通し確認は、オーナーが後日まとめて実施する最終ゲートとして残置している(準備〔手順書・パス修正〕は完了済み。`Docs/REAL_BACKEND_CHECKLIST.md`参照)。

詳細は[DEVLOG.md](DEVLOG.md) §8(グループ1)・§9(グループ2)・§10(グループ3)、正本はバックエンド側`Nz-LTX23-backend/Docs/FRONTEND_CATCHUP_WORKORDER.md`を参照。

---

## パリティ実装 第1〜3陣 完了(2026-07-15)

グループ1〜3の完了後、2026-07-15の棚卸しでGradio GUI(検証用の簡易画面)↔フロントエンドの機能差分としてN1〜N13を新規発見し、`Docs/WEBVIEW2_PARITY_BACKLOG.md`へ「実装難易度＋依存順序」で並べた推奨実装順(第1〜第4陣)を整理した。このうち**第1陣〜第3陣の計7項目を実装完了**した(バックエンド〔`Nz-LTX23-backend`〕は凍結方針を維持し、7項目すべてバックエンド変更ゼロ)。

- **第1陣(クイックウィン)**: N6(V2V結合の音声クロスフェード長〔`handle_crossfade_ms`〕を150/300/500msから選ぶUI)、N11(設定画面にspill-free〔解像度ごとの快適フレーム上限〕一覧表)、N10(設定画面に生の`/config`〔設定情報〕JSONビューア)。N11・N10はSettingsPanelで`useConfig()`を共有。
- **第2陣(Create画面のIC-LoRA周り、N3→N2の依存順)**: N3(Create画面にIC-LoRA強度スライダー2種〔`conditioning_attention_strength`/`reference_video_strength`〕。`ReferenceStrengthField`をChainから共有化し、`loras`非空ゲートで従来の潜在バグ〔`reference_video_id`無条件送信による422〕を是正)、N2(コントロールLoRA選択ドロップダウン。`/config`の`model.ic_loras`から選択肢を生成)。N2の参照動画必須ゲート実装により`Docs/PENDING_TASKS.md`§2(4段階再編後の現在は`Docs/PENDING_TASKS_CLOSED.md`§3-1)も同時クローズ。
- **第3陣(共有UIコンポーネント新設／デッドコード配線)**: N1(Create/Chainに Crop出力欄。LTXはVAE空間圧縮率が32倍のため解像度を32で割り切れる必要があり、当初は32刻みスナップ＋生成サイズ以下clampを実装。※2026-07-17に32刻みスナップは撤廃され、「32以上・生成サイズ以下の任意整数」の自由入力へ変更されている。詳細はDEVLOG §14.3参照)、N7(バッチA2VのShared共有キーフレーム画像を本番配線。Gradio原典`batch.py`の`_validate()`とパリティ)。

あわせて、**N9(ポーリング間隔・タイムアウトのユーザー設定欄)をオーナー決定で「将来の研究課題」へ降格**した(フロントのポーリングが堅牢で機能的問題がないため)。これにより能動的な実装対象はN1〜N8・N10・N11・N13の11件となり、うち第1〜3陣の7項目が完了、**残るは第4陣(N4・N5・N8・N13)の4件**である。

**2026-07-15追記(第4陣・前半完了)**: このうちN13(APIキーバッジ)とN4(危険ゾーン=パイプラインunload・完了ジョブ一括purge)は、同日中に第4陣・前半として実装完了・push済みである(コミット`46fe95b`feat／`4f10d57`docs)。**残る能動的な実装対象は第4陣・後半のN5(バッチ表の行内編集)とN8(テーマ切替dark/light)の2件のみ**となった。詳細は[WEBVIEW2_PARITY_BACKLOG.md](WEBVIEW2_PARITY_BACKLOG.md)、[DEVLOG.md](DEVLOG.md) §12.8を参照。

**2026-07-15追記(第4陣・後半完了=第4陣全完了)**: 残っていたN5(バッチ表の行内編集)とN8(テーマ切替dark/light)も同日中に実装が完了した。N5はバッチ画面の各行へプロンプト直接編集・共通プロンプト流し込み・行別画像割当(`fs.listFiles`で画像フォルダを列挙)の3操作を追加、N8は設定画面へdark/lightトグルを追加し、`ThemeProvider`/`useTheme`(`LanguageContext`と同型)＋`index.css`のCSS変数化(dark既定＋`:root[data-theme="light"]`上書き)＋`localStorage`(キー`nzltx23.theme`)単独永続化で実現した。いずれもバックエンド・ブリッジ契約の変更はゼロ。敵対的レビュー指摘M1(画像フォルダ切替時の旧一覧残留)・M2(打鍵ごとのCSV書き込み過剰)・L1(スキャン中の編集ゲート漏れ)を是正済みで、機能破壊級バグはなし。**これで第4陣(N4・N5・N8・N13)は全完了となり、2026-07-15棚卸しの能動的な実装対象11件(N1〜N8・N10・N11・N13)がすべて実装済みになった**(N9は将来の研究課題へ降格、N12はクローズのため対象外)。品質: `npm run typecheck`0エラー、vitest 581件緑(10件skip)、`npm run build:single`成功。コードは本追記時点で未コミット(オーナーが後日コミット予定)。詳細は[WEBVIEW2_PARITY_BACKLOG.md](WEBVIEW2_PARITY_BACKLOG.md)、[DEVLOG.md](DEVLOG.md) §12.9を参照。

**品質**: `npm run typecheck`(tsc -b)0エラー、vitest 553件緑、`npm run build:single`成功。コードはオーナーがコミット・push済み(コミット`e647d01`、`origin/main`と同期)。実機ゲート(モック目視・実GPU E2E)はオーナー後日。

**(2026-07-15時点の)次の着手【史料】(更新: 2026-07-15、第4陣・後半完了後=第4陣全完了)**: 以下は2026-07-15時点の記録であり、直後の注記(2026-07-17)のとおり現行ではない。最新の「次の着手」は本節末尾(2026-07-17更新分)を参照。`WEBVIEW2_PARITY_BACKLOG.md`が能動的な実装対象としていたN1〜N13のうち11件(N1〜N8・N10・N11・N13)は本セッションまでにすべて実装が完了し、**第4陣(N4・N5・N8・N13)を含め、着手すべき能動的な実装項目はもう残っていない**(N9は将来の研究課題、N12はクローズ)。次に着手すべきはコード実装ではなく、オーナーによる後日ゲートのみ: (1) 実バックエンド(実GPU)での通し確認(`Docs/REAL_BACKEND_CHECKLIST.md`)、(2) WebView2実機での一連の新規UI(N1〜N13分すべて。特に本セッション分のバッチ表行内編集のCSV往復とlight/dark目視)の目視確認。入口は`Docs/PENDING_TASKS.md`の「近日中の改修項目」§4(オーナー目視確認の残項目)と§1(実バックエンド通し確認)。詳細は[DEVLOG.md](DEVLOG.md) §12.9を参照。

**注記(2026-07-17)**: 本欄は2026-07-15時点の状態を示す。その後2026-07-16〜17に大規模UIリデザインと実機フィードバック起点のデバッグ・UI改修を実施したため、最新の状態・次の着手は次節「大規模UIリデザイン+実機フィードバック起点のデバッグ・UI改修 完了」を参照。

---

## 大規模UIリデザイン(2026-07-16〜17)+実機フィードバック起点のデバッグ・UI改修(2026-07-17)完了

パリティ実装(第1〜3陣・第4陣)の完了後、オーナーの実機使用感を踏まえてUI全体のレイアウト・操作モデルを作り直す大規模UIリデザインを2日間で実施した(Create/Chain/Batch画面の再構成、予約機構・ジョブレーン・`GenerationPanel`の廃止とジョブ台帳`JobLedger`への統合、ネイティブブリッジのordered_json化など)。続けて同日中に、そのリデザインを実機で使ったオーナーのフィードバックに基づくデバッグ＋UI改修を4波構成で実施した(Crop出力の是正、プリセット再適用ボタンの新設〔※2026-07-18/19廃止、自動適用＋A→B→A再適用へ回帰。DEVLOG §22〕、DURATION入力方式の刷新、ファイル選択ダイアログのタイムアウト撤廃、Chainソース入力欄の一本化、ドラッグ&ドロップ対応〔ブリッジ契約v7〕)。あわせて、前段(§フロントエンド追随・パリティ実装)から持ち越していたCMakeの`OBJECT_DEPENDS`未修正・`useChainForm.test.ts`のflakyテストの2件も本セッションで解消した。バックエンド(`Nz-LTX23-backend`)は、アクセスログ抑制のみの例外(リデザイン内、生成ロジック不変)を除き引き続き一切変更していない(凍結方針を維持)。詳細は[DEVLOG.md](DEVLOG.md) §13(大規模UIリデザイン)・§14(実機フィードバック起点のデバッグ＋UI改修)を参照。

**品質**: `npm run typecheck`(tsc -b)エラー0、vitest 684件緑・10件skip、native doctest(`NzLTX23_tests.exe`)215ケース(2026-07-18の第4弾バッチ〔`Docs/DEVLOG.md` §26〕でCSV系ブリッジメソッド削除・`reuseIfPresent`追加を経て217ケースへ更新済み)、`npm run build:single`成功。実機(AviUtl2 v2.0.54ポータブル構成)へ、リデザイン完了時および4波それぞれの完了時にデプロイして挙動を確認済み。**全変更は本節を記録した時点で未コミット**(オーナー承認後にコミットする方針は従来どおり)。

**追記(2026-07-17・第5波完了)**: 同日中に追加セッションとして、IC-LoRA(参照動画で構図・動きを制御する制御用LoRA)の入口をプロンプトタグ機構から分離するUI再設計(第5波)を実施し完了した。Style LoRAはプロンプトタグ＋チップのまま維持し、IC-LoRAはCreate画面の参照動画アコーディオン内の状態持ちドロップダウン(「なし」付き・選択保持)＋重みスライダーへ入口を分離、手打ちタグの自動移行、Chainでの制御タグ検出時の警告＋Generate無効化、Libraryの Controlタブ削除などを実装した。あわせて「制御LoRAの重ねがけ(SDXLのControlNet的な併用)」を調査し、実装構造上不可能である(同一Union-Control LoRAファイルの別名登録・単一参照スロットのため)ことを確認、正道である合成前処理モードの追加を将来の研究課題として起票した。品質ゲートは`npm run typecheck`エラー0・vitest 729件緑(10件skip、§14終了時点から+45)・`npm run build:single`成功で、実機デプロイ済み・本節記録時点では未コミット。詳細は[DEVLOG.md](DEVLOG.md) §15、[PENDING_TASKS.md](PENDING_TASKS.md)§2・§4を参照。

**次の着手(更新: 2026-07-17、第5波完了後)**: 着手すべき能動的な実装項目はコード面ではもう残っていない。次に着手すべきは以下の4点で、いずれもオーナー判断・実機ゲート・別セッション予定のものであり、コード実装ではない。

1. **オーナー目視ゲート(第1〜4波のUI改修分＋第5波のIC-LoRA UI再設計分を含む全項目)**: `Docs/PENDING_TASKS.md`§2「オーナー目視確認の残項目」。ドラッグ&ドロップの実配線部分は特にWebView2依存のため実機でのみ完全検証できる(REALDEVICE-VERIFY)。
2. **BatchTableデバッグ・タイムライン右クリックデバッグ**: ~~オーナー指示で後回しのまま未着手~~。**（2026-07-19解消）** BatchTableは2026-07-19に、タイムライン右クリックは右クリック再設計（[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md)第8版）の第1段階＝増分I1〜I14・G1/G2/G3実機ゲートでいずれもクローズ済み。`Docs/PENDING_TASKS_CLOSED.md`§3-17・§3-22。
3. **実バックエンド(実GPU)での通し確認**: `Docs/REAL_BACKEND_CHECKLIST.md`。従来どおりオーナーが後日まとめて実施する最終ゲート。
4. **キーフレーム欄の大規模改修**: **実施済み(2026-07-18、開始フレーム専用枠廃止含む。DEVLOG §18/§20参照)**(第5波までのドラッグ&ドロップ対応・IC-LoRA UI再設計はいずれも意図的にキーフレーム欄を対象外としていたが、その直後のセッションで着手・完了した)。

**追記(2026-07-17・第6波完了)**: `Docs/PENDING_TASKS.md`§1-3が「無害な既知の余剰動作」として残していた「`deploy.ps1`が`Plugin\NzLTX23\webui\`フォルダもコピーする」を、オーナー判断により埋め込み専用運用へ一本化して解消した(`scripts/build.ps1`の`-NoEmbedWebui`新設・既定反転、`scripts/deploy.ps1`のコピー廃止＋埋め込みガード＋残骸削除)。実機検証6手順すべて合格(負のテストで実機aux2のSHA256不変を確認、native doctest 215件緑)。上記4点の次の着手には変更なし。詳細は[DEVLOG.md](DEVLOG.md) §16、[PENDING_TASKS_CLOSED.md](PENDING_TASKS_CLOSED.md)§3-7(当時の§1-3。後日の台帳スリム化で§1-3は削除・番号整理され、その後§1-3が指す項目は何度も入れ替わった。2026-07-27に最後の§1-3〔自家製GGUFの配布〕が`PENDING_TASKS_CLOSED.md`§3-47へ移設されたため、**現在の`PENDING_TASKS.md`に§1-3は存在しない**〔欠番〕)を参照。

---

## Context

動画生成AI「LTX 2.3」のローカルバックエンド(`Nz-LTX23-backend`、FastAPI、β版完成・**凍結**)を操作するAviUtl2プラグインをゼロから開発する。バックエンドは変更禁止。モックは見た目の参考のみ(元ソースなし・厳密再現不要)。親エージェント(Fable 5)は監督に徹し、実装・テスト・調査はサブエージェント(**Opus以下、Fable5禁止**)に委任する。

### 調査で確定した事実(3体のExploreエージェント＋親の裏取り済み)

**バックエンドAPI**(一次ソース: `backend/gradio_ui/api_client.py`, `backend/LTX23_Backend_Specification.md`。詳細は [API_REFERENCE.md](API_REFERENCE.md) 参照)

- REST(JSON)のみ、`http://127.0.0.1:18620/api/v1`。ジョブは非同期: `POST /generate`(または`/generate/chain`) → 202+job_id → `GET /jobs/{id}` を1秒ポーリング → `GET /jobs/{id}/video` でmp4取得。
- アップロード: `POST /upload/image|video|audio`(multipart, field=`file`)→ id を generate に渡す。
- 能力取得: `GET /status`(疎通/GPU/混雑)、`GET /config`(プリセット・制約・spill_free_frames警告閾値)、`GET /loras`(＋`/loras/{name}/thumbnail`は`<img>`直参照可)。
- 凍結制約: width/height 64の倍数、num_frames 8n+1(9〜481)、同時1ジョブ(409 JOB_BUSY)、キャンセル非即時。**UIに出してはいけない機能**: negative_prompt / guidance_scale / pipeline切替 / steps(設計ブリーフ§5が明記。CFG・ネガティブは無効グレーアウトで場所のみ確保)。
- CORS許可はlocalhostオリジンのみ(凍結)。mockバックエンドでGPU不要のE2Eテスト可(API/スキーマ同一)。
- エラーは `{"error":{code,message,...}}` エンベロープ。api_key設定時のみ書き込み系にBearer。

**AviUtl2 SDK**(`frontend/aviutl2_sdk/`、親がヘッダで裏取り済み。詳細は [SDK_REFERENCE.md](SDK_REFERENCE.md) 参照)

- 種別は**汎用プラグイン `.aux2`**(`plugin2.h`)。雛形は `WindowClient.cpp`。
- 独自ウィンドウ: 自前HWND(WS_POPUP)→`host->register_window_client(name,hwnd)`(plugin2.h:791、WS_CHILD化されドッキング)。
- タイムライン取込: `call_edit_section_param`(plugin2.h:592)のコールバック内(メインスレッド・編集ロック下)で `create_object_from_media_file(file,layer,frame,length)`(plugin2.h:289)。
- 編集情報: `get_edit_info()`→EDIT_INFO(width/height/rate/scale/frame/layer)。I2V用フレーム画像: `rendering_scene_video`(plugin2.h:680、別スレッドcb・PIXEL_RGBAバッファは即コピー必須)。
- 多言語: `CONFIG_HANDLE::translate`＋`Language/*.aul2`(既定英語・日英対応)。設定保存APIはSDKに無い→`app_data_path`配下に自前JSON。HTTP APIも無い→WinHTTP自前実装。
- 文字コード: API=UTF-16(LPCWSTR)、alias/PROJECT_FILE param=UTF-8、SDKヘッダのコメント=Shift-JIS。

**環境(確認済み)**

- VS Community 2026(MSVC 14.51, `C:\Program Files\Microsoft Visual Studio\18\Community`)、git有り、cmake単体はPATH無し(VS同梱を使用)。
- AviUtl ExEdit2 **beta52**(当時。現行はv2.0.54ポータブル): `D:\For_Videos\AviUtl2-beta52\aviutl2.exe`。プラグイン配置先 `C:\ProgramData\aviutl2\Plugin`。WebView2ランタイム有り。
- **手元SDKは2026/7/5版(beta53a世代)で本体beta52より新しい**。関数テーブルは追記式なので、beta52時点で存在するAPIのみ使用すれば安全(中核APIは2025/10〜12追加で問題なし。2026/6/14追加の`register_event_listener`等は使用前に実機確認)。
- frontendは既に`git init`済み(mainブランチ、初期コミット`f2dad6e "sdk"`、.gitignore整備済み)。

### ユーザー決定事項

- モック元ソースなし→UIはゼロから実装(見た目のみ参考)。
- Web UIスタック: **React + TypeScript + Vite**。
- **beta52＋手元SDKに固定**して開発(最新版追従は後回し)。

## アーキテクチャ(要点)

- **`.aux2`一本**: 自前HWNDにWebView2をホスト、React UIを表示。
- **通信方式(ハイブリッド)**: JSON制御系は `postMessage`→ネイティブ(WinHTTP)→バックエンドの**プロキシ**(CORS回避・Bearer秘匿・multipart対応)。サムネ/動画プレビューは`<img>/<video>`の**src直参照**(CORS対象外)。
- **スレッドモデル**: UIスレッド=WebView2＋`call_edit_section_param`のみ(編集ロック下ではDL/HTTP禁止、`create_object_from_media_file`呼び出しのみ)。HTTPワーカースレッドで通信、完了をUIスレッドへPostMessageでマーシャリング。ポーリングはJS側setInterval(1s)→bridge経由。
- **I2Vフロー**: `get_edit_info().frame`→`rendering_scene_video`→cb内で即バッファコピー→ワーカーでWIC PNG化→`/upload/image`→image_id。
- **配布**: 開発時は`SetVirtualHostNameToFolderMapping`で`webui/dist`をマウント。リリース時はvite-plugin-singlefileで単一HTML化→DLLリソース埋め込み→単一`.aux2`配布。
- **依存はすべて`third_party/`にベンダリング**(WebView2 SDK(nupkg展開・静的loader)、WIL、nlohmann/json、doctest)。HTTP=WinHTTP、PNG=WIC(OS標準)。vcpkg不使用。
- **ビルド**: CMake(VS同梱cmake＋`CMakePresets.json`)。デバッグ用にVSジェネレータで.sln生成可(`aviutl2.exe`をデバッグ対象にアタッチ)。

### ディレクトリレイアウト(frontend配下に新設)

```
native/           # C++層: plugin_main / webview_host / bridge / http_client /
                  #        aviutl_edit / frame_capture / settings / log
  third_party/    # ベンダリング依存
  tests/          # doctest
webui/            # React+TS+Vite: bridge(本番webview/devモック切替) / api / modes(Create,Chain,Library) /
                  #                components(JobRail,PromptBar,StatusHeader) / i18n(en,ja) / state
                  #                ※JobRailは計画当時の記述(現在は廃止、DEVLOG §13参照)
Language/         # *.aul2(ネイティブUI文言)
scripts/          # build.ps1 / deploy.ps1 / run-aviutl.ps1
Docs/             # 全ドキュメント
```

## マイルストーン(各Mは動作確認できる状態で区切る)

- [x] **M1-0 調査ドキュメント整備**(最初のタスク・コード前): `Docs/`に4点作成 — `API_REFERENCE.md`(バックエンド全エンドポイント・凍結制約・UIに出さない機能)、`SDK_REFERENCE.md`(.aux2仕様・スレッド規約・文字コード規約・beta52とのAPI年代整合表)、`WEB_RESEARCH.md`(WebView2組込・CORS切り分け・デバッグ手順・先行事例)、`DEVELOPMENT_PLAN.md`(本計画)。3体の調査報告が原稿。(コミット`d17296a`)
- [x] **M0 最小.aux2**: 空ウィンドウ＋ボタン＋ログの`.aux2`をCMakeでビルド→deploy→beta52でドッキング表示確認。doctest基盤も稼働。`RequiredVersion`はbeta52で通る値に校正。(コミット`af06d6e`)
- [x] **M1 WebView2＋ブリッジ疎通**: 最小Reactアプリを仮想ホストで表示、ping/pong往復＋`get_edit_info`のサイズをUIに表示(「AviUtl2のサイズ取得」ボタンの土台)。(コミット`7100613`)
- [x] **M2 T2V E2E**: WinHTTPプロキシ、`/status`・`/config`をヘッダ表示、プロンプト＋サイズ(64刻み)＋尺(8n+1)＋シードで生成→ポーリング→mp4 DL→タイムライン挿入。**mockバックエンドでE2E確認**。(コミット`5ca0949`)
- [x] **M3 ジョブレーン**: 右側常設ジョブレーン、進捗％/stage、完了トースト、キャンセル(非即時表示)、409は「予約」で吸収、サーバー状態ヘッダ(待機/混雑/未起動)＋再接続。(コミット`766bb35`)
  - **追補（2026-07-17・予約機構とジョブレーンは廃止）**: 「409を予約で吸収して自動再送信する」機構は取り除きました。同時1ジョブの制約はそのままに、実行中は生成ボタンをビジー状態で無効化して弾く方式へ変更しています。あわせて画面右に常設していたジョブレーンも廃止し、ジョブ一覧は生成ボタンと同じパネル内の台帳（全ジョブの一覧表）に統合しました（生成ボタンのあるパネルを画面の最右上に置き、その右には何も表示しないというオーナー要望への追随）。完了トースト・キャンセル非即時表示・進捗表示はこの台帳側で従来どおり機能します。
  - **※例外(2026-07-17)**: この同じセッションで、`backend/main.py`のアクセスログフィルタ2種(`StatusAccessFilter`/`JobsListAccessFilter`)のみオーナー承認済みの意図的な変更を行っています(上記「Verification」節の凍結方針に対する唯一の例外。詳細は[DEVLOG.md](DEVLOG.md) §13.5)。revert禁止。
- [x] **M4 I2V**: フレームキャプチャ→PNG→upload、キーフレーム枠UI(最大5枚・位置・強さ)、手持ち画像アップロードも対応。(コミット`f8d0591`, `4610d21`, `48f7620`)
- [x] **M5 LoRA/素材棚**: `/loras`一覧＋サムネ直参照、`<lora:name:strength>`チップ(±0.1、0〜2.0)、reload、生成履歴。(コミット`46767f9`)
- [x] **M6 Chain/V2V/A2V**: 最大8クリップ・個別プロンプト・Seam blend・Chain時LoRA無効の明示、V2V(upload→続き生成→join→joined)、A2V。(コミット`e71375b`)
- [x] **M7 仕上げ**: 日英切替(`translate`＋aul2＋Web i18n、既定英語)、接続先設定の永続化(`app_data_path`配下JSON)、本体テーマ調和(`get_color_code`等)、CFG/ネガティブの無効プレースホルダ、レスポンシブ最終調整、単一.aux2リリースビルド。(M7a: コミット`be20e54`、M7c: コミット`bc7beeb`、M7b: コミット`0b2d159`)

各マイルストーン完了時: mockバックエンド＋AviUtl2 beta52での手動確認手順を実施し、結果をDocsの開発ログに記録してコミット。→ 実績は[DEVLOG.md](DEVLOG.md) §2-3参照(native doctest 81ケース、webui vitest 299件、mockバックエンド＋beta52での手動確認)。残課題(日本語aul2フォールバック目視・実機UI全体目視・real backend実機確認・backend venv恒久修正)は[DEVLOG.md](DEVLOG.md) §5参照。

## 1.0.0-rc1後の追加作業: 右クリック生成の配線(2026-07-08)

**注記(後日追記・2026-07-19〜20)**: 本節は2026-07-08の初回配線セッションの記録である。右クリック導線はその後の**タイムライン右クリック再設計**（[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md)第8版・[`DEVLOG.md`](DEVLOG.md) §33〜§40）で全面的に再設計・実装され、メニュー構成（旧「全10アクション」→素材種別ごとの新メニュー）・仮オブジェクト運用・生成パラメータ決定が刷新された。**現行仕様の唯一の正は同SPEC**。下記「未着手・既知課題」5点も、その後のセッションで大半が消化されている（`insertProvisional`実配線・A4再生成のjobIDタグ付与など。[`PENDING_TASKS.md`](PENDING_TASKS.md) §3参照）。以下は史料として原文のまま残す。

1.0.0-rc1リリース後の追加セッションとして、[TIMELINE_ALPHA_REQUIREMENTS.md](TIMELINE_ALPHA_REQUIREMENTS.md)第7節が次セッションの設計スコープとして送っていた「右クリックメニュー→WebView2パネル→生成キュー」の配線を実装した。**完了した内容**:

- `timeline.menuInvoked`イベントを`useMenuRouter`が受け取り、`shell/AppShell.tsx`の`onRoute`が消費してモード切替＋intent＋初期解像度をプリフィルする配線(以前は`useMenuRouter`の消費者が存在しなかった)。全10アクション配線済み(パネル誘導方式)。
- 選択ペイロード(`timeline.getSelection`/`timeline.menuInvoked`)への`mediaWidth`/`mediaHeight`追加(native `get_media_info`配線)と、それを使う解像度決定の一元化(`webui/src/timeline/deriveGenerationParams.ts`)。
- IC-LoRA(`referenceVideo`)ルーティングの是正(誤ってChainへ送っていたのをCreateへ付け替え、Createフォームへの`reference_video_id`送信配線を新設)。

詳細は[DEVLOG.md](DEVLOG.md) §7、契約面の詳細は[BRIDGE_CONTRACT.md](BRIDGE_CONTRACT.md) §4.13/§4.14、バックエンド仕様面は[API_REFERENCE.md](API_REFERENCE.md) §12.2を参照。

**次セッションへの引き継ぎ(未着手・既知課題)**:

1. **既知課題(要対応)**: `ReplaceObjectEditProc`(`native/src/bridge.cpp`)が完了クリップの生成時に`set_object_name`を呼んでいない。このため完了後のクリップにjobIDタグが付与されず、「完了クリップを選んで元のjobIDを逆引き再生成する」導線(A4再生成)が成立しない。
2. 選択オブジェクトの「素材そのもの」の自動投入(切り出し→レンダリング→アップロードして素材IDをプリフィルへ載せる)。今回はファイルピッカーでユーザーが素材を供給する前提で配線した。
3. 生成キュー投入→仮オブジェクト予約(`insertProvisional`)→完了差し戻し(`resolveProvisional`)の実配線。`lengthFrames`の供給と挿入位置決定を含む。
4. β版のレターボックス(入力の自動リサイズ)。`deriveGenerationParams`のseamに差し込む設計(第7-5節参照)。
5. Libraryからの履歴再submit導線(regenerateの受け皿)。

## サブエージェント運用ルール(親=監督専任)

- モデル選択: 設計・難所実装=Opus、通常実装・テスト=Sonnet、機械的作業=Haiku。**Fable5禁止**。
- **ファイル所有権**: Track A(native基盤: plugin_main/webview_host/bridge/log/settings)、B(http_client＋webui/src/api)、C(aviutl_edit/frame_capture)、D(webui UI全般)、E(tests/scripts)。ブリッジRPC型定義は単一契約ファイルに固定し先行確定、1ファイル1所有者、並行エージェントには担当外ファイルの変更を禁止と明示。
- **GPU/プロセス競合**: 開発テストはmockバックエンド使用(GPU不要)。realバックエンドやAviUtl2実機起動を伴う検証は同時1エージェントに限定。バックエンドのコード変更は全エージェントに禁止と明示。
- PC本体のPython環境等には触れない(バックエンドは自前venv完結、フロントはnpm/CMakeのみ)。
- 実験は仮説→コード/WEBで裏取り→スコープ確定→実施の順を徹底。

## リスクと対策

| リスク | 対策 |
|---|---|
| SDK(7/5版)＞本体beta52の版ずれ | beta52以前に追加されたAPIのみ使用(中核APIは2025/10〜12で安全)。`RequiredVersion`をM0で実機校正。SDK呼び出しは`aviutl_edit`に隔離 |
| WebView2初期化失敗 | 失敗時もプラグインは生存しネイティブ最小UIでエラー表示。UserDataFolderは`app_data_path`配下 |
| 文字コード混在 | 変換ユーティリティを1箇所に集約、規約をSDK_REFERENCEに明文化 |
| 編集ロック下の長時間処理 | コールバック内は`create_object_from_media_file`のみ。`wait_rendering_task`をロック下で呼ばない |
| クラッシュ解析 | .sln生成→VSで`D:\For_Videos\AviUtl2-beta52\aviutl2.exe`(当時。現行はv2.0.54ポータブル)にアタッチ。WebView2はDevToolsで解析 |

## Verification

- **M0〜**: `scripts/deploy.ps1`でPluginフォルダへ配置→AviUtl2起動(当時beta52・現行はv2.0.54ポータブル。`scripts/deploy.ps1`/`scripts/run-aviutl.ps1`の既定パスは2026-07-15にv2.0.54へ更新済み)→ウィンドウ表示・ログ確認。
- **M2〜**: mockバックエンド起動(`backend/run.ps1`、config `backend: mock`)→ T2V生成→タイムラインにmp4が挿入されるE2Eを手動確認。
- **自動テスト**: native=doctest(文字コード変換・8n+1/64倍数バリデーション・JSON整形)、webui=vitest＋bridgeモック(状態遷移・予約ロジック。※「予約ロジック」は計画当時の記述で現在は廃止、DEVLOG §13参照)、HTTPプロキシ=mockバックエンド相手のintegrationテスト。
- backend配下は一切変更しないこと(git statusで無変更を確認)。
  - **※例外(2026-07-17)**: `backend/main.py`のアクセスログフィルタ2種(`StatusAccessFilter`/`JobsListAccessFilter`)のみ、オーナー承認済みの意図的な変更。生成ロジック(`api`/`engine`)には一切手を入れていない、ログ出力抑制のみの例外。詳細はフロント側[DEVLOG.md](DEVLOG.md) §13.5参照。**revert禁止**。

## 実装開始直後の手順

1. M1-0: Docsサブエージェント(Sonnet)に3調査報告を渡し4ドキュメント作成→親がレビュー→コミット。
2. M0: Track Aサブエージェント(Opus)がCMake骨格＋最小.aux2→ビルド→deploy→ユーザーにAviUtl2での表示確認を依頼(実機確認はユーザーの目が必要な唯一のポイント)。
3. 以降、マイルストーン単位でTrack並行→結合→検証→コミットを反復。
