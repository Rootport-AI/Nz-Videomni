# Nz-Videomni 開発ログ

最終更新: 2026-09-05（**§111 機能名→UIの宣言表`shell/featureScope.ts`を新設し、画角拡張の予算を自前の表からサーバー配信へ移した**〔フロントエンドが自前で抱えていた「モデルごとの知識」を2つとも手放した回。①サーバーが名指しする機能名がどのUI部品を閉じ、閉じたあとにどの保存値をサーバー既定へ書き戻すかを、1枚の宣言表`FEATURE_UI`だけで決める形にした——**サーバーの機能名を知ってよいファイルはこれで1つになり**、書き戻しの`useEffect`2本も表の`resets`定義から導かれる汎用の1本へまとまった。②画角拡張の快適トークン予算は`GET /config`の`limits.comfort_budgets[系統].outpaint_budget`から読む形になり、自前の系統別定数表と据え置き値40,000は**概念ごと廃止**した——**「線が無ければ警告を出さない」（`null`）が唯一の規則**である（オーナー最終裁定）。**画面の見た目は何も変えておらず、線の値も1つも動いていない。** 新しい基準は vitest **138ファイル・2,734件全緑・0 skipped**〔実バックエンドを叩く`api/backend.integration.test.ts`を除外した実行が基準。**従来の「137ファイル・2,707件＋10 skip」は除外していない実行の数**である〕・`npm run lint`エラー0で**警告31本**〔`AppShell`の`exhaustive-deps`常設警告1本が消えて32本から減った〕・`npm run typecheck`エラー0で、配布aux2は**1,269,248バイト・SHA-256 `58BD0323…`**がビルド出力・実機・リポジトリ配布コピーの3値で一致。**§111.5 オーナーの画面目視G-V1〜G-V4は同日中に全項目合格し、台帳§2-8はクローズした**〔目視結果の詳細はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §99.13、台帳の記録は[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-135〕。設計判断とゲート実数の正本は同[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §99、予算の値の正本は同[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md) §9のまま。詳細は本書§111〕を追加。以下は同日それ以前の記録——**§110 Edit画面〔画角拡張・撮り直し〕に、Settingsの加速設定が載るようにした**〔塞いでいたのは`shell/AppShell.tsx`が`EditScreen`へ`acceleration`を渡していない配線1本で、両サブタブが同じ理由で同時に抜けていた。**Edit側にエンジン別のガードは足していない**——エンジンが断る設定の表はバックエンドのアダプタが正本であり、送出層で二重に担保しない規律に従った。再発防止のガードは`App.editRoute.test.tsx`の実描画の通し1本である。**この改修で画角拡張の快適上限の前提が変わったので、同日バックエンド側で加速を全onにした構成の再較正を行い、線を測り直した**——予算の数値の正本はバックエンド[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md) §9、較正と改修の記録は同[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §98、**§110.6 オーナーの画面目視V1〜V6は同日中に全項目合格し、判断J1・J3の裁定も出て台帳§2-7はクローズした**〔加速がSettingsどおり載ることと既定へ戻ることを両サブタブで確認、エンジンを切り替えても422は1件も無し、LTX 2.5 側では**再較正の前は警告が出ていた尺で警告が消える**ことを目視。J1は「両系統とも固定線のままでよい——**ただし系統ごとの個別裁定であって一般の設計思想にはしない**」。目視結果と裁定の正本は同[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §98.11、台帳は同[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-143〕。自動ゲートは`npm run typecheck`エラー0・vitest **137ファイル・2,707件全緑**〔実装で+6件〕・`npm run lint`エラー0で警告32件は増減なし、配布aux2は**1,267,712バイト・SHA-256 `46F1AC00…`**がビルド出力・実機・リポジトリ配布コピーの3値で一致。あわせて§108へ追補§108.8〔§108.6が残していた「加速設定を載せる改修が入ったら要再較正」という前提の消化〕を足した。詳細は本書§110〕を追加。以下は2026-09-04時点までの記録——**§108 画角拡張〔Outpainting〕の快適上限を、エンジン系統ごとの実測値へ置き換えた**〔§66以来ずっと連結生成の予算からの流用だった線を、画角拡張自身の較正で引き直した。`COMFORT_TOKEN_BUDGET = 40,000`は消さず、**系統が分からないとき**（`GET /models`が届く前など）の据え置き値へ役割を変えた。系統ごとの予算はフロントエンドが持つ凍結定数`OUTPAINT_COMFORT_TOKEN_BUDGETS`で、`resolveOutpaintComfortBudget()`がただ1つの入口である。あわせて表示と判定で食い違っていたトークン数の式を1つ（切り捨て形）へ揃えた。**警告専用・非ブロッキングであることは不変で、変わったのは線の位置だけ**である。**§108.7 オーナーの画面目視ゲートV1〜V3〔下の§107.11のV-1〜V-3とは別のゲートである〕は同日中に合格した**——V1（LTX 2.3）とV2（LTX 2.5）で**同じ幾何でも系統が違えば警告の出方が変わる**ことを実機で確認し、V3（系統不明時のフォールバック）は実機では到達できないため未確認のままの合格扱いとした（オーナー裁定）。自動ゲートは`npm run typecheck`エラー0・`npm test` **136ファイル・2,700件全緑**・`npm run lint`エラー0で、配布aux2は**1,267,712バイト・SHA-256 `fa179e34…`**がビルド出力・実機・リポジトリ配布コピーの3値で一致。予算の数値の正本はバックエンド[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md) §9、較正と目視結果は同[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §95、台帳は同[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-134。詳細は本書§108〕を追加。以下は同日それ以前の記録——**§107.11追補の実機ゲートV-1〜V-3が全項目合格し、§3-140追補は決着した**〔361フレーム・24fps＝15.042秒の素材で検証。V-1＝挿入動画が①二色リボン＋右クリック「音声を分離」あり②最終フレーム121（別素材でも確認）③エイリアス保存の3点合格、V-2＝✨予約の仮オブジェクトの最終フレームが要求どおり361（修正前は+1だった）、V-3＝✨→Generate→🎞置換の通しが正常。監督が実機の`data\Alias\361_insert.object`と`361_d_and_d.object`を突き合わせ、ヘッダ`frame=0,360`（包含＝361フレーム）・`音声付き=1`・`再生位置=0.000,15.042`が完全一致（差はファイルパスのみ）することを確認した——挿入がD&Dと意味論同一になったことの直接証明である。記録は本書§107.11末尾とバックエンド[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-140の追補〕を追加。以下は同日それ以前の記録——**§107.11 §3-140の追補——エイリアス`frame=`ヘッダの端点は「包含」だったので、書いていた値を1減らして1フレーム過長を直した**〔同日いったん書いた「`OBJECT_LAYER_FRAME.end`は排他と確定」「🎞挿入側のほうが素材に忠実」「D&D側は本体の丸めによる切り捨て」の**3点はいずれも誤りで撤回した**。一次証拠は実機が書き出したエイリアス4本と`plugin.log`の5行で、§107.11と[`SDK_REFERENCE.md`](SDK_REFERENCE.md) §16 (h)へ転記してある。修正はネイティブに閉じており〔`NormalizeAliasObjectFrameHeader`が`frame=0,length−1`を書く／同族のオフバイワン3件〕、**webUI・モックブリッジ・契約は無改修**。§44.4の「`frameEnd`はinclusive」は復権。自動ゲートはnative doctest 292ケース・アサーション1,504件で0失敗、webuiは無改修のまま全緑〔136ファイル・2,696件〕〕を追加。以下は同日それ以前の記録——§107 🎞ボタンで挿入したオブジェクトが青一色になっていた症状の修正〔メディア挿入を`create_object_from_media_file`の単発呼び出しから**エイリアス方式**（`create_object_from_alias`）へ差し替えた。原因は同APIが直列化キー`音声付き`を立てないことで、実害は右クリックに「音声を分離」が出ないこと。従来APIは三重の安全網として温存。副作用として、詰めたリボンからのV2Vが全尺送信からトリム送信へ変わる（ドラッグ＆ドロップ由来のオブジェクトと揃う方向）。**オーナーの実機ゲートR1〜R8は2026-09-04に全項目合格し、§3-140はクローズした**（記録はバックエンド[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-140。R4のエイリアス突き合わせで`音声付き=1`・`再生位置`完全一致を確認した。なお`OBJECT_LAYER_FRAME.end`の端点は**包含**である——このゲート中にいったん「排他」と結論したが、同日§107.11で撤回・訂正済み〔[`SDK_REFERENCE.md`](SDK_REFERENCE.md) §16 (h)〕）〕を追加。あわせて文書の誤りを2件訂正した——エイリアス書式を「XML形式」と書いていた2箇所（[`SDK_REFERENCE.md`](SDK_REFERENCE.md) §5.1・[`WEB_RESEARCH.md`](WEB_RESEARCH.md) §5。正しくはINI風）と、`音声付き`を「存在しない」と書き過ぎていた2026-09-01の訂正注記（[`TIMELINE_ALPHA_REQUIREMENTS.md`](TIMELINE_ALPHA_REQUIREMENTS.md) 第0節B。UI項目としては無いがエイリアスの直列化キーとしては現役）。自動ゲートはnative doctest 289ケース・アサーション1,497件で0失敗、webuiは無改修のまま全緑（136ファイル・2,696件）。以下は2026-09-03時点までの記録——§106 埋め込み処理器の常駐トグルの追加〔Settingsの高速化6行目`keep_resident_embeddings`。**LTX 2.5のときだけ見える**——名指ししている部品が2.5にしか無いため、非対応を宣言するのはLTX 2.3の側であり、§100.3で入れた「非対応のベースモデルでは行ごと消す」作法の2例目かつ逆向きの1例目である。バックエンド台帳[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §3-114、実測は同[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §92、オーナー実機目視は未了〕を追加。自動ゲートは全緑（136ファイル・2,696件）。以下は2026-09-02時点までの記録——§104 バッチA2Vの走行を画面の作り直しに耐えるところへ移した件〔ランナーをモジュールレベルのシングルトンへ移し、i2v-longと寿命規則を揃えた。台帳の記録はバックエンド[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-47-02、オーナー実機目視ゲートG-P1は同日中に合格し完結（追記は§104末尾、バックエンド§2-1は節ごと削除済み）。あわせて§54へ「§104で解消」の追記を1行足した〕を追加。自動ゲートは全緑（135ファイル・2,653件）。以下は同日それ以前の記録——§103 撮り直し（Retake）でスタイルLoRAを使えるようにした件〔塞いでいたのは`buildRequest`の1関数だけで、バックエンドは1行も変えていない。台帳の記録はバックエンド[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-62-02、後継の制御系IC-LoRA対応は同[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §3-141〕と、§102 Editタブの右クリック自動読み込みが二重にアップロードしていた不具合の修正〔隠れているOutpaintingパネルの自動読み込みが一緒に走っていた。同§3-63-02〕を追加。どちらも自動ゲートは全緑で、`.aux2`は再ビルドして実機と配布コピーの2箇所へ配置済み。§103の実機ゲートはバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §88.5のG7〜G9として合格している。以下は2026-09-01時点までの記録——§101 素材のフレームレートを右クリックプリフィルへ流し込めるようにした件〔ブリッジ契約v11。Media Foundationで素材ファイルの宣言だけを読む新規モジュールを足し、整数へのスナップはwebui側で行う。あわせてMCPとモックに閉じていた3件も片付けた。バックエンド台帳§3-13・§3-115・§3-116・§3-122〕を追加。自動ゲートは全緑で、§3-13のオーナー実機ゲートG1〜G8のみ未了。§100 Settingsの並び順の入れ替えと非対応ベースモデルでのPrunaVAED非表示〔同日オーナー実機目視合格・追記済み〕、§99 快適上限マーカーのサーバー配信テーブル化を追加。以下は2026-08-31時点までの記録——§98 モデル読み込み中の即時バッジとGenerate凍結の一本化——`POST /pipeline/load`の往復自体をバッジの材料にし〔§87(2)で却下した(c)案をJobsContext一本化により覆した〕、全タブの生成ボタンを`serverBusy`1本のルールへ統一。台帳は`PENDING_TASKS.md` §1-24、オーナーの実機ゲート待ち）を追加。§97 未導入ベースモデルの案内を導入バッチの名指しへ戻した＝`baseModelInstaller()`の復活〔バックエンドで`install-LTX25.bat`が実在するようになったため〕を追加。§91・§92・§93へ、それぞれの目視ゲートの結果を追記した——§91はSageAttentionのG8が2026-08-25に合格〔正本はバックエンドVERIFICATION_LOG §77.10〕、§92は撮り直し・素材（末尾）のG8が2026-08-30に合格〔同§78.14。M8とR-1の監督裁定も同日にオーナーが追認して決着した——同§78.15。§92には追記2として格上げを1つ足してある〕、§93は画角拡張のG8′が2026-08-29に合格〔同§79.11、既定値2点の裁定は同§79.12〕。**いずれも追記専用の規律どおり本文は不変**である。§96 LTX 2.5での非CFGネガティブプロンプト〔NAG／VSF〕の開通＝モックの宣言1語の削除と、そこで見つかったchain側の潜在バグの修正を追加。§93 LTX 2.5での画角拡張〔Outpainting〕の開通、§94 「脱緑ブレンド」と§66「採らなかった指摘(1)」との和解、§95 画角拡張の目視ゲートG8′合格を追加。§74 Single a2vの全長stage-2化、§75 バッチa2vスキップ上限の調査結果〔改修不要でクローズ〕とOutpaintingのセンタリングを追加。以下は2026-08-10時点までの記録——§55 Acceleration〔生成の高速化〕フロントエンド追随、§56 V2Vリボン範囲トリムの実機調査確定・仕上げ実装・IC-LoRA参照動画への拡張・実機ゲート全項目合格とテーマクローズ〔§56.6〕・保存領域の方針確定〔§56.7〕を追加。あわせて§49へ「配布方法は2026-07-31に`.aux2`単体同梱へ置き換え済み」の注記を付した。§68・§69（Retake本体実装と目視フィードバック改修バッチ）を追加。§70（長尺A2V＝クリップ連結の全体へ音声1本を添付する機能の実装）を追加。§71（Retake改修第2弾＝❌／🔁ボタン・表示順・文言）を追加。**§68・§69・§71は同日中にオーナー目視が全項目合格し、状態行を「2026-08-10 オーナー目視合格」へ更新した**（開閉区間は閉区間で確定。§70＝長尺A2Vのみ目視待ち）。**さらに同日、最後に残っていた最小窓73フレームの仕様判断が「73フレーム維持（注意文で伝える方式）」で確定し、Retakeのテーマは完結した**（台帳の記録は[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-73）。§57以降には未コミットの節がある——コミット状況は各節末尾の『状態』行を正とする） / 出典: `git log`(コミット本文)・各マイルストーンの実装ソース・実機/自動テスト結果(§13の大規模UIリデザインは`80e99bd`/`9bd5172`、§14の実機フィードバック起点のデバッグ＋UI改修は`86e8689`/`c1a5a17`、§15のIC-LoRA UI再設計〔第5波〕は`0dacdf4`/`60b9108`、§16の埋め込み専用運用への一本化〔第6波〕は`a884677`/`198db67`、§18のキーフレームタイムラインUI本実装は`3ab643c`(feat)/本コミット(docs)として、いずれもコミット・`origin/main`へのpushまで完了済み。本節冒頭より後の各節に残る「未コミット」等の記述は、その節を記録した時点のスナップショットであり、その後のセッションでコミット・pushされている(§27・§28は`6ff17c0`／`9324f05`、§52(NAG追随)は`260e73a`としていずれも`origin/main`へpush済みで確認済み。§57以降には未コミットの節がある——コミット状況は各節末尾の『状態』行を正とする)

関連ドキュメント: [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) ／ [API_REFERENCE.md](API_REFERENCE.md) ／ [SDK_REFERENCE.md](SDK_REFERENCE.md) ／ [BRIDGE_CONTRACT.md](BRIDGE_CONTRACT.md)

---

## 1. 概要

2026-07-07〜08の2日間で、[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)のマイルストーンM1-0からM7cまでを完了し、**1.0.0-rc1**としてリリースパッケージ`dist/NzVideomni-1.0.0-rc1.au2pkg.zip`を作成した。バックエンド(`Nz-Videomni`)は一切変更していない(凍結方針を維持)。

## 2. マイルストーン表

| M | 内容 | 主なコミット | 日時(ローカル) |
|---|---|---|---|
| M1-0 | 調査ドキュメント整備(`API_REFERENCE.md`/`SDK_REFERENCE.md`/`WEB_RESEARCH.md`/`DEVELOPMENT_PLAN.md`の初版) | `d17296a` | 2026-07-07 02:08 |
| M0 | 最小`.aux2`(空ウィンドウ+ログ)、CMake/Ninja基盤、doctest基盤 | `af06d6e` | 2026-07-07 02:14 |
| M1 | WebView2ホスト+JSON-RPCブリッジ疎通(ping/getEditInfo)、React webui基盤 | `7100613` | 2026-07-07 02:41 |
| M2 | T2V E2E(WinHTTPプロキシ、Create画面、生成→ポーリング→タイムライン挿入) | `5ca0949` | 2026-07-07 03:08 |
| M3 | ジョブレーン(常設進捗表示、予約/待機UX) | `766bb35` | 2026-07-07 03:40 |
| M4 | I2Vフレームキャプチャ(`rendering_scene_video`→WIC PNG)、multipartアップロード(native)+ ファイルピッカー/サムネイル(native)+ I2Vキーフレームパネル(webui) | `f8d0591`, `4610d21`, `48f7620` | 2026-07-07 03:29〜03:57 |
| M5 | LoRAブラウザ、`<lora:name:strength>`チップ同期、生成履歴 | `46767f9` | 2026-07-07 04:14 |
| M7a | 接続設定の永続化(`settings.json`)、動的バックエンドURL、日英言語ファイル(native側) | `be20e54` | 2026-07-07 04:21 |
| M6 | Chainモード(クリップエディタ、V2V継続+join、A2V) | `e71375b` | 2026-07-07 08:09 |
| M7c | 単一ファイルWeb UI埋め込み、`.au2pkg.zip`パッケージング、`backend.request`の`timeoutMs` | `bc7beeb` | 2026-07-07 08:25 |
| M7b | Web UI側i18n(en/ja)、設定パネル、無効プレースホルダ、単一ファイルビルド | `0b2d159` | 2026-07-07 08:43 |

コード面の全13コミットは2026-07-07中に完了した。その後の実機総合確認(AviUtl2 beta52での手動確認)・`.au2pkg.zip`パッケージング検証・本ドキュメント群の最終整備を2026-07-08にかけて実施し、**1.0.0-rc1として実装完了(2026-07-08)**とした。

## 3. 検証結果

### 3.1 自動テスト

| スイート | 件数 | 内容 |
|---|---|---|
| native doctest(`NzVideomni_tests.exe`) | **81ケース** | `test_bridge_core.cpp`(RPCパース/ディスパッチ、8n+1・64倍数等のバリデーションはバックエンド仕様のミラーではなく契約のparamsパース)、`test_settings.cpp`(`SettingsStore`の自己修復・URL正規化)、`test_strconv.cpp`(UTF-16/UTF-8変換)、`test_wic_png.cpp`(PNG/JPEGエンコード・RGBA→BGRA変換・サムネイルサイズ計算)、`test_http_client.cpp`(WinHTTPラッパー)。CTest登録済み。 |
| webui vitest | **299件** | ブリッジ(`mockBridge`/`requestDispatcher`)の状態遷移・タイムアウト・イベント購読、API層(生成リクエストのバリデーション・ポーリング)、各モード画面(Create/Chain/Library)のコンポーネントテスト、i18n。 |

### 3.2 手動実機確認(AviUtl2 beta52)

各マイルストーン完了時に`scripts/deploy.ps1`でPluginフォルダへ配置し、AviUtl2 beta52を起動してドッキング表示・機能疎通を確認する運用を徹底した(mockバックエンド使用、GPU不要)。M2以降はT2V生成→ポーリング→タイムライン挿入のE2Eをmockバックエンドで確認。M6のChain/V2V/A2VはE2E経路の疎通を確認したが、joinや長時間ポーリングを含む網羅的シナリオは一部未実施(§5残課題参照)。

## 4. 開発中の重要な発見

開発を通じて、事前調査(M1-0時点のドキュメント)だけでは分からなかった以下の挙動を発見した。今後の保守・追加開発のために記録する。

1. **AviUtl2 beta52は初回ロード時に信頼承認ダイアログが必要。** 新規追加されたネイティブプラグインは、起動時に「スクリプト・プラグインの追加」ダイアログで「このプラグイン・スクリプトを信頼して使用する」を明示的にクリックするまで`RegisterPlugin`が一切実行されない(ログも出ない)。承認結果は`C:\ProgramData\aviutl2\module.ini`の`[NzVideomni\NzVideomni.aux2] trust=1`として永続化される。自動デプロイ後の初回起動では必ずこの手動承認が要る(出典: `README.md`)。

2. **AviUtl2はディスクにログを書かない。** ホストの`LOG_HANDLE`はアプリ内ログビューにのみ出力し、ファイルへの永続化を一切行わない。自動検証(スクリプトやサブエージェント)が動作結果を確認できる唯一のチャネルとして、`%LOCALAPPDATA%\NzVideomni\logs\plugin.log`への自前ファイルログ実装(`native/src/log.cpp`、UTF-8・タイムスタンプ付き・ホストログにもミラー)が必須だった。

3. **VS2026(v18)にC++ CMakeツールが未導入。** インストール済みVisual Studio 18(2026 preview)は「C++ CMakeツール」コンポーネントを同梱せず、かつ既存のCMakeジェネレータはまだ「Visual Studio 18」ジェネレータに対応していなかった。対策として`scripts/build.ps1`が初回利用時にポータブルCMake+Ninjaを`%LOCALAPPDATA%\NzVideomni\buildtools`へ自動取得する(per-userキャッシュ、PATH非変更、グローバルインストールしない)。実働構成は「Ninja + MSVC(`vcvars64.bat`経由)」で、VSジェネレータ(`.sln`)はデバッガアタッチ用のスタブとして残置。

4. **バックエンドのvenvはフォルダ移動でpyvenv.cfgのhomeが旧パスのまま壊れている。** `Nz-Videomni/.venv`(および`.venv-engine`)がリポジトリ移動後、`pyvenv.cfg`の`home`が移動前の絶対パスを指したままで、venvのPython実行が正しく解決されない状態を確認した。フロントエンド開発中は**`PYTHONPATH`設定+基盤(システム)Python直接起動**で回避して開発・mock検証を継続した。バックエンドは凍結対象のため恒久修正(`.venv`再構築等)はスコープ外とし、**ユーザー承認待ち**としている。

5. **`PIXEL_RGBA`はR,G,B,A順、WICのPNGエンコーダはネイティブでBGRAを要求する。** `rendering_scene_video`のコールバックが渡す`PIXEL_RGBA`バッファ(`{r,g,b,a}`各1バイト)はDXGIの`R8G8B8A8_UNORM`と同じバイト順で、コピー自体はチャネル入れ替え不要。しかしWICの32bit PNGエンコーダのネイティブピクセルフォーマットは`GUID_WICPixelFormat32bppBGRA`であるため、`wic_png.cpp`はRGBAソースを`GUID_WICPixelFormat32bppRGBA`として認識させた上でWICの`IWICFormatConverter`にBGRAへ変換させてからエンコードしている。単純に生バイト列をそのままPNGへ書き込むと色が反転する。

6. **`.aul2`言語ファイルのセクション名はプラグインの出力ファイル名(`NzVideomni.aux2`)。** `Language/English.NzVideomni.aul2`等の`[NzVideomni.aux2]`セクションが`CONFIG_HANDLE::translate`の対象になる。プラグイン名(表示名"Nz-Videomni")ではなく、ビルド成果物のファイル名と一致させる必要がある点は事前調査で確定していなかった。

7. **mockバックエンドはchainジョブの`clip`/`clip_count`を常に`null`で返す。** `services/pipeline_manager.py`のchain用`on_progress`コールバックは`clip is not None`のときのみ`job.clip`/`job.clip_count`を更新するが、`services/ltx_runner.py`の`_MockBackend.generate_chain`は`progress_callback(None, None, ...)`しか呼ばず`clip`/`clip_count`引数を一切渡さない(実バックエンドの`engine/pipeline/chain_pipeline.py`側はstage-1で`clip=`/`clip_count=`を渡す設計)。結果として**mockバックエンドで確認する限り、chainジョブの`GET /jobs/{id}`は常に`clip: null, clip_count: null`を返す**。[API_REFERENCE.md](API_REFERENCE.md) §6のJobResponse例(`"clip": 2, "clip_count": 4`)は実バックエンドの挙動を示したものであり、mockでの開発時に「進捗表示にclip/clip_countが出ない」ことに驚かないよう明記しておく(§付録にも追記した)。

8. **`overlap_frames`はclip0のstage-1 latentフレーム数未満でなければ実バックエンドで拒否される(未文書化制約)。** `chain_math.py::compute_chain_layout`は`kv >= L`(`kv` = `overlap_frames`、`L`は各クリップの`v_latent_frames(num_frames)` ≈ `(num_frames-1)//8+1`)の場合に例外を送出する(`"overlap_frames (K_v=...) must be < every clip's stage-1 latent frames [...]"`)。[API_REFERENCE.md](API_REFERENCE.md) §5.2は`overlap_frames`の範囲を「1-8」としか記載しておらず、この「各クリップの`num_frames`に依存する上限」は事前調査時点では判明していなかった。短い`num_frames`のクリップ(例: 9〜16フレーム、stage-1 latentは2〜3)を含むchainでは`overlap_frames`の既定値3でも拒否され得るため、フロントの入力バリデーションに反映した(§付録にも追記)。

9. **設定パネルの`settings.set`は`baseUrl`省略時に読み取り専用no-opとして動作する。** これは事前設計どおりだが、実装完了時に改めてテストで固定した(`test_bridge_core.cpp`)。

## 5. 残課題

1. **日本語`.aul2`フォールバックラベルの目視確認未実施。** `Language/Japanese.NzVideomni.aul2`の文言(WebView2初期化失敗時のフォールバック表示等)は実装・自動テスト上は英語版とキー一致を確認済みだが、AviUtl2を日本語ロケールで起動した際の実際の表示崩れ・文言の自然さは未確認。
2. **実機UI全体の目視確認が未実施。** 各マイルストーンの疎通確認(§3.2)は行ったが、全画面(Create/Chain/Library/設定パネル)を通しで目視レビューする最終確認は完了していない。
3. **backend実機(`real`)での動作確認が未実施。** 開発・自動検証はすべて`config.model.backend=mock`で行っており、GPU接続の実バックエンド(`real`)に対する疎通・生成E2Eはまだ実施していない。特に§4-7/§4-8で確認したmockとの挙動差(clip/clip_count、overlap_frames制約)は実バックエンドで再検証が望ましい。
4. **バックエンドvenvのpyvenv.cfg破損の恒久修正はユーザー承認待ち。**(§4-4参照)

## 6. 参照

- `git log --oneline`(本ドキュメントのコミットハッシュの一次ソース)
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\README.md`
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\native\tests\`
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\webui\src\`(`*.test.ts(x)`)
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\services\ltx_runner.py`(`_MockBackend.generate_chain`)
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\services\pipeline_manager.py`(chain用`on_progress`)
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\chain_math.py`(`compute_chain_layout`の`overlap_frames`検証)

## 7. 右クリック生成の配線(ルーティング→プリフィル)

1.0.0-rc1リリース後の追加セッションとして、[TIMELINE_ALPHA_REQUIREMENTS.md](TIMELINE_ALPHA_REQUIREMENTS.md)第7節で「次セッションの設計スコープ」として送っていた、右クリックメニューからの生成呼び出しの配線を実装した。バックエンド(`Nz-Videomni`)は引き続き一切変更していない(凍結方針を維持)。

### 7.1 配線内容

右クリックメニュー(`timeline.menuInvoked`)からの意図(intent)を`useMenuRouter`が受け取り、`shell/AppShell.tsx`が消費する形で、対象パネル(Create/Chain/Library)へモード切替＋intent＋初期解像度をプリフィルする配線を実装した。全10アクションを配線済み(パネル誘導方式。7-1のオーナー方針どおり、右クリックから直接生成キューへは飛ばさず、必ずパネルでの確認・調整を挟む)。Reactの実装は、「keyによる再マウント＋lazyな`useState`初期化」で一度だけプリフィルを適用する方式を採用した(`useEffect`での後追い適用は不使用)。統合テスト`App.prefill.test.tsx`を追加した。

### 7.2 生成パラメータ(解像度)の決定方式

新規の純粋関数`deriveGenerationParams`(`webui/src/timeline/deriveGenerationParams.ts`)に、右クリック生成時の解像度決定ロジックを一元化した(7-3-B/7-5で提示していた「一箇所のseam(接ぎ目)に集約する」方針の実装)。

- 選択素材の実解像度(後述の`get_media_info`由来の`mediaWidth`/`mediaHeight`)を最優先し、無ければプロジェクト既定へフォールバックする。
- ÷64(IC-LoRA使用時は÷128)へ丸め、上限は`floorToMultiple`で(一般は64／IC-LoRAは128の)それぞれの倍数へ切り下げる(`max_height`=1088→1024のクランプ破綻を回避)。
- 調整理由を`notes`に日本語で提示する。
- fps/num_framesは従来どおりフォーム既定＋ユーザー調整のまま(β版のレターボックスはこのseamに差し込む予定)。

### 7.3 native `get_media_info`配線

`bridge.cpp`の`GetSelectionEditProc`で、選択オブジェクトのメディアファイルパスから`EDIT_SECTION::get_media_info`を呼び、選択ペイロードにper-objectの`mediaWidth`/`mediaHeight`(整数、取得不能時は0)を追加した。webui側の`getSelection`型・mockも対応済み。`get_media_info`の実際の返り値は実機でのみ確認可能なため、コード中に`// REALDEVICE-VERIFY:`コメントを残置し、[TIMELINE_ALPHA_REQUIREMENTS.md](TIMELINE_ALPHA_REQUIREMENTS.md)第6節の実機確認リストにも項目を追加した。

### 7.4 IC-LoRA(referenceVideo)ルーティングの是正

[TIMELINE_ALPHA_REQUIREMENTS.md](TIMELINE_ALPHA_REQUIREMENTS.md)第7-3-Aで指摘していたIC-LoRAのルーティング不整合(chainへ送っていたが、IC-LoRAは`/generate`専用で`/generate/chain`には存在しない)を是正した。

- `menuRouting`の行き先をchain→createへ付け替えた。
- Createフォームに参照動画のアップロードUI(chainの`useSourceUpload`を再利用)と`reference_video_id`の送信配線を実装した(`GenerateRequest`型に`reference_video_id`を追加、`toGenerateRequest`に合流)。
- IC-LoRA有効時はwidth/heightを128の倍数に丸める。

### 7.5 検証結果

| スイート | 件数 | 備考 |
|---|---|---|
| webui tsc(strict) | エラー0 | |
| webui vitest | **394件** | `App.prefill.test.tsx`等を追加 |
| native doctest(`NzVideomni_tests.exe`) | **161ケース／911アサーション** | 新規警告0 |

`NzVideomni.aux2`の生成(ビルド・リンク)も成功を確認した。実機(beta52)での実`get_media_info`・実生成の確認は次段(§7.7)。

### 7.6 既知課題(重要)

**完了クリップへのjobIDタグ再付与が未実装。** nativeの`ReplaceObjectEditProc`(仮オブジェクトを完了クリップへ置換する処理)で`set_object_name`が呼び出されていない。これが無いと、完了後のクリップを選択して「元のjobIDを逆引きして再生成する」という導線(A4再生成)が成立しない。次段で必ず対応が必要な既知課題として記録する。

### 7.7 次段送り(実機必須。seamのみ用意)

- 選択オブジェクトの「素材そのもの」の自動投入(切り出し→レンダリング→アップロードして素材IDをプリフィルに載せる)。今回は既存のファイルピッカーでユーザーが素材を供給する前提で配線した。
- 生成キュー投入→仮オブジェクト予約→完了差し戻し(`insertProvisional`/`resolveProvisional`。`lengthFrames`の供給と挿入位置決定を含む)。
- §7.6の`set_object_name`未呼び出しの解消。
- 入力の自動リサイズ／レターボックス(β版。`deriveGenerationParams`のseamに差し込む予定)。
- Libraryからの履歴再submit導線(regenerateの受け皿)。
- 実機(beta52)での実`get_media_info`・実生成の確認([TIMELINE_ALPHA_REQUIREMENTS.md](TIMELINE_ALPHA_REQUIREMENTS.md)第6節参照)。
- `reference_video_id`単独指定時の422対策(control LoRA同伴の強制)。送信配線自体は完了したが、UIはcontrol LoRA(`loras`)同伴を強制しておらず、参照動画のみで送信すると`reference_video_id`は単独指定不可(§5.1)の制約により422になり得る([API_REFERENCE.md](API_REFERENCE.md) §12.2参照)。

## 8. フロントエンド追随 グループ1(APIコントラクト追随、2026-07-15)

バックエンド側の正本ワークオーダー`Nz-Videomni/Docs/FRONTEND_CATCHUP_WORKORDER.md`(2026-07-15起票)のグループ1(APIコントラクト追随。既存クライアントを壊さない加算的〔additive〕な変更への追従)を完了した。バックエンド(`Nz-Videomni`)は引き続き一切変更していない(凍結方針を維持)。

### 8.1 項目1: Clip Chainのクリップ上限 8→24

Clip Chain(複数のクリップ〔動画の一区間〕を連結して1本の動画にする機能)で扱えるクリップ数の上限を、バックエンドの拡張(2026-07-14、コミット061cccc)に合わせて8から24へ引き上げた。`webui/src/modes/chain/chainUtils.ts`の`MAX_CLIPS`を24へ、合計フレーム上限`MAX_CHAIN_TOTAL_FRAMES`を3848(8×481)から11544(24×481)へ更新し、24枚のクリップ枠を横スクロールできるようCSSを調整した。

### 8.2 項目2: `chunked_upsample`トグル

`chunked_upsample`(アップサンプル処理を小分けにしてGPUメモリの使用量を抑えるオプション)の送信トグルをChain画面に追加した。APIの既定値は互換維持のため意図的にFalseのままなので、フロントエンドは既定オンのUIとしたうえで、リクエストには値を省略せず常に明示送信する実装にした。省略すると旧経路(一括アップサンプル)に戻ってしまい、768pの長尺チェーンでVRAM(GPU専用メモリ)が溢れる不具合を回避できないためである。

### 8.3 項目3: チェーンLoRAのストリップ廃止

`webui/src/api/types.ts`に残っていた「`GenerateChainRequest`に`loras`フィールドは無いため、送信前にプロンプト中の`<lora:...>`タグをストリップ(除去)する」という誤った前提のコメントと実装を撤回した。バックエンドは2026-07-03からチェーンの`loras`を受理しており、2026-07-11のA2V(音声から動画を生成する機能)＋LoRA併用解禁で全面解禁済みだったため、タグはパースして`loras`配列としてそのままchain APIへ送信するよう修正した。あわせて、この誤った前提に基づく「LoRAは無視されます」というトースト通知と関連注記を撤去した。

### 8.4 項目4: chain参照動画対応(α版)

`reference_video_id`(輪郭線・骨格などの手がかりを読み取る元になる参照動画のID)と、条件付けの強さを指定する`conditioning_attention_strength`／`reference_video_strength`の2つのスライダー(0.0〜1.0、未指定時はnull)をChain画面に追加した。バックエンドの制約(`clips=1`かつ`source_video`〔V2Vの元動画〕と排他)に合わせ、Clipsサブモード限定のオプション節として実装し、参照動画を添付している間はクリップ数を1にピン留めする。また「参照動画ありならLoRA必須」というバックエンド側の制約を`isValid`のバリデーションに反映し、送信前にブロックすることで422エラーを予防した。

### 8.5 項目5: ジョブポーリング120分化は対応不要と判定

バックエンド側ワークオーダーの項目5は、Gradio GUI(検証用の簡易画面)のジョブポーリング(生成ジョブの進捗を定期的に問い合わせる処理)タイムアウトを60分から120分へ延長したというものだが、調査の結果**フロントエンド側には対応不要**と判定した。理由は、フロントエンドのジョブ監視(`useGeneration.ts`)が1秒間隔の無限ポーリングであり、そもそも打ち切りのデッドラインを一切持たない構造だからである。Gradio側の「120分」は、ジョブ投入から完了までを一本の同期ブロッキング呼び出しとして扱う設計に紐づくタイムアウト概念であり、フロントエンドの「ジョブを投入して即座に制御を返し、別途ポーリングで状態を追う」非同期構造には、そもそも移植すべき対応概念が存在しない。ドキュメント面では、[API_REFERENCE.md](API_REFERENCE.md)に残っていた3600秒(60分)表記を7200秒(120分)へ更新した。

### 8.6 項目6: 尺の算術の反映

クリップの`num_frames`(各クリップの生成時フレーム数)の単純合計は出力動画の尺そのものにはならない(オーバーラップ融合による減算、V2Vのcontext置換による減算があるため)という、バックエンド正本`Nz-Videomni/Docs/PHASE3_CLIP_CONCAT_STATUS.md`末尾の算術を、`chain_math.compute_chain_layout`の式から純関数(`vLatentFrames`/`pxFromVLatent`/`computeOutputFrames`)として`chainUtils.ts`へ転記した(JSDocコメントに正本パスを明記)。Chain画面に「予想出力尺」のプレビュー表示と、減算理由(オーバーラップ融合・context置換)の注記を追加した。進捗表示(`JobResponse.clip`/`clip_count`/`stage`)は既に実装済みだったため変更していない。

### 8.7 付帯修正・検証結果

`webui/src/api/types.ts`と[API_REFERENCE.md](API_REFERENCE.md)を、上記全項目の新契約(clips上限24・chainの`loras`/`chunked_upsample`/`reference_video_id`対応・ポーリング120分)へ同期した。`webui/src/bridge/mockBridge.ts`に残っていた旧上限8のバリデーションも24へ修正した。

| スイート | 件数 | 備考 |
|---|---|---|
| webui tsc(strict) | エラー0 | |
| webui vitest | **413件(40ファイル)** | 改修前398件から退行なし・新規テスト15件追加 |

実バックエンド(GPU接続の`real`)での通し確認は、`FRONTEND_CATCHUP_WORKORDER.md`のグループ3・項目13が最終ゲートとして扱う方針のため、本セッションでは未実施のまま残置している(§5残課題3も参照)。

## 9. フロントエンド追随 グループ2(機能パリティ、2026-07-15)

バックエンド側の正本ワークオーダー`Nz-Videomni/Docs/FRONTEND_CATCHUP_WORKORDER.md`のグループ2(Gradio GUIにあってフロントエンドに無い機能への追従)を完了した。バックエンド(`Nz-Videomni`)は引き続き一切変更していない(凍結方針を維持)。本グループでは新たにブリッジ契約v6のnative実装が必要になったため、native側(`native/src/`配下のC++)にも今回初めて手を入れている。

### 9.1 項目7: バッチA2V「就寝中一括生成」

正本`Nz-Videomni/Docs/BATCH_A2V_CSV_SPEC.md`(版1.0)準拠のCSVマニフェスト(10列・UTF-8 BOM付き・CRLF改行)相互運用によるバッチA2V(音声フォルダを丸ごと指定し、就寝中などまとまった時間に複数のA2V〔音声から動画を生成する機能〕ジョブを順に流す機能)を、Create画面内の折りたたみ節(アコーディオン)として実装した。

- **stat列のみを再開判定の正典として扱う**。CSVの他の列(出力パス等)から状態を推測せず、`stat`列(各行の処理状態)だけを見て再開/スキップを判定する。
- **書込みの原子性**: 一時ファイルへ全内容を書き込んでから原子的リネームで置き換える方式(`fs.writeTextFileAtomic`)を使用し、`WRITE_LOCKED`(他プロセスによるロックが5回のリトライ後も解消しない場合)を検知したらautosave(自動保存)側へ退避して書き込み内容を失わないようにした。
- **Done行の保護マージ**: 既に完了(Done)している行は、再読込・再書込みの際も上書きしない形でマージする。
- **出力フォルダ規約**: `{音声フォルダ名}_a2v_out`。
- **ファイル名対応規約**: `音声名.wav` → 同名の`音声名.mp4`(AviUtl2タイムラインへ取り込む際の対応付けのため)。既存ファイルとの衝突時は`noClobber`付番(`_2`, `_3`...)で回避する。
- API追加は不要で、既存の`POST /generate/chain`をループ呼び出しする形で実現した。

### 9.2 ブリッジ契約v6のnative実装完了

[BRIDGE_CONTRACT.md](BRIDGE_CONTRACT.md)に記録していた「v6時点でnative未実装、`types.ts`が暫定的な一次ソース」という状態を解消し、`ui.pickFolder` / `fs.listFiles` / `fs.readTextFile` / `fs.writeTextFileAtomic` / `fs.probeAudioDuration`の5メソッドと、`backend.downloadVideo`への`destDir`/`fileName`/`noClobber`params拡張、新エラーコード`WRITE_LOCKED`を、すべてnative側(C++)に実装した。

- 新規モジュール`wav_probe.h/.cpp`(wavヘッダを読んで再生時間を計測)、`fs_util.h/.cpp`(フォルダ列挙・テキスト読み書き・原子的置換・noClobber付番)を追加し、`bridge_core.cpp`/`bridge.cpp`から配線した。
- 両モジュールとも専用のdoctestファイル(`native/tests/test_wav_probe.cpp`、`native/tests/test_fs_util.cpp`)で単体検証済み。
- これにより、[BRIDGE_CONTRACT.md](BRIDGE_CONTRACT.md)冒頭の「実装(C++側)を正とする」という原則がv6についても回復した(v6作成時点では例外的に`types.ts`を正としていたが、本セッションでnative実装が追いついたため)。

### 9.3 モデル管理UI

`SettingsPanel`内に、`GET /models`(モデル一覧取得)と`POST /pipeline/load`(モデル切替)を使うモデル管理UIを追加した。

- 4カテゴリ(想定: チェックポイント/LoRA/VAE等、バックエンドの分類に追従)のドロップダウン＋Refreshボタン＋Loadボタンの構成。
- `loadPipeline`呼び出しは、モデル読み込みが数分単位になり得ることを踏まえ`timeoutMs`を600000(10分)に明示指定した。
- バックエンドが読み込み中の別ジョブでビジー状態(HTTP 409、`JOB_BUSY`)を返した場合は、専用の分かりやすいメッセージを表示するようにした。

### 9.4 Chainプリセットドロップダウン

Chain画面に、`config.generation_presets`(バックエンドの`/config`から動的取得する生成プリセット一覧)を駆動源とするプリセットドロップダウンを追加した。選択すると、`spill_free_frames`(VRAM〔GPU専用メモリ〕が溢れない範囲でのフレーム数、優先的に採用する推奨値)を優先した推奨クリップ長を、全クリップスロットへ一律で自動反映する。Create画面の既存プリセット(ボタン形式)とは別実装だが、`config.generation_presets`という同じデータソースを使っているため、バックエンド側のプリセット改名・追加にも動的に追従する。

### 9.5 wav長→Frames自動調整・事前チェック

`chain_math`由来の算術(§8.6で転記した`chainUtils.ts`の純関数群)に、新たに`suggestFramesForAudio`等のwav長からFrames候補を導く関数を追加し、A2Vのソース音声アップロード時に自動でFramesへ反映するようにした。長さの実測には、契約v6で新設した`fs.probeAudioDuration`(wavヘッダを読んで再生時間を計測するブリッジメソッド)を使用しており、Gradio側のクライアント計測(stdlibの`wave`)と同等の役割をnative側(WICではなくwavヘッダパース)で担う。音声に対して現在の設定が短すぎる場合は、送信前に必要秒数を明示して事前にブロックする。

### 9.6 検証結果

| スイート | 件数 | 備考 |
|---|---|---|
| webui vitest | **534件(48ファイル)** | 改修前413件から退行なし |
| `npm run typecheck`(`tsc -b`) | エラー0 | §9.7-2参照。`npx tsc --noEmit -p .`は偽合格になるため使用しないこと |
| native doctest(`NzVideomni_tests.exe`) | **201ケース**(mockモードの実バックエンド接続時、integrationテストを含めて**207件**全PASS) | `test_wav_probe.cpp`・`test_fs_util.cpp`を新規追加、警告ゼロ |

埋め込みWeb UI(`aux2`)への単一ファイル組み込みと、`.au2pkg.zip`パッケージングも生成済みを確認した。

### 9.7 実機ゲート(2026-07-15)

1. **native統合テスト**: 実バックエンド(mockモード)を起動した状態でnative側のintegrationテストを実行し、207件全PASSを確認した。
2. **バッチA2VのE2E**: 実ファイルシステム上でBOM(バイト順マーク)・CRLF改行が実バイトレベルで仕様どおりになっていることを確認し、バックエンド正本の`manifest.py`(Python実装)が生成/読解するCSVとの相互運用(同一マニフェストを両者が読み書きできること)を突き合わせた。再開時にDone行がスキップされること、`noClobber`付番が正しく機能することも確認した。
3. **AviUtl2実機**: **v2.0.54**(ポータブル構成、`D:\For_Videos\AviUtl2\aviutl2_v2.0.54\data\`配下)でクリーンロードし、Create画面のバッチ節・SettingsPanelのモデル管理・Chainプリセット・wav自動調整を含む全経路の疎通を確認した。

### 9.8 α版の意図的省略

以下は今回スコープ外として意図的に見送った。α版の制約としてUI上にも明記済み。

- **バッチの共有キーフレーム**: CSVのimage列が`Shared`(共有)を指す場合でも、バッチA2Vはconditioning(画像による条件付け)なしのA2Vとしてのみ扱う。
- **バッチでの参照動画**: IC-LoRA用の参照動画指定は、バッチ経路では未対応。
- **開始時の事前検証(re-judgement)**: バッチ開始前に全行を一括で再検証する仕組みは持たない。
- **ウィンドウを開いている間だけ進行する制約**: バッチはCreate画面(WebView2)を開いている間だけ処理が進む。裏で永続的に動き続けるバックグラウンドサービスではないことをUI上に明記した。

### 9.9 開発中の重要な発見

1. **GradioのバッチランナーはchunkedUpsampleを送っていない。** バックエンド側のGradioバッチ実行経路(`services`配下のバッチ処理)は、チェーンリクエストに`chunked_upsample`を明示送信していない既存のギャップがある。フロントエンドのバッチA2Vは常に明示送信する実装で先行しており、この点ではフロントエンドの方がバックエンドの既存実装より進んだ状態になっている。バックエンド側の残課題として`FRONTEND_CATCHUP_WORKORDER.md`にも記録した。
2. **webuiの`npx tsc --noEmit -p .`は偽合格になる。** プロジェクト参照(project references)構成のためこのコマンドは実際にはほとんど何も型検査せずに成功してしまう。正しい型検査ゲートは`npm run typecheck`(内部で`tsc -b`を実行する)である。今後の検証手順はすべて後者を使うこと。
3. **`scripts/deploy.ps1`・`scripts/run-aviutl.ps1`の既定パスが旧環境のまま。** 両スクリプトの既定パスは`beta52`/`C:\ProgramData`を指しており、現行の実機構成(AviUtl2 v2.0.54、`D:\For_Videos\AviUtl2\aviutl2_v2.0.54\data\`)と一致していない。本セッションでは手動でパスを読み替えて実機ゲートを実施したが、スクリプト自体の恒久修正は残課題である。

### 9.10 オーナー目視確認の残項目

自動テスト・実機ゲート(§9.7)では機能疎通を確認したが、以下はオーナー本人による目視確認が未実施のまま残っている。

- Create画面のバッチA2V折りたたみ節のUI(レイアウト・文言の自然さ)
- `ui.pickFolder`のネイティブフォルダ選択ダイアログの見た目・挙動
- SettingsPanelのモデル管理パネル(4カテゴリドロップダウン・Refresh・Load)
- Chainプリセットドロップダウンの見た目・推奨クリップ長反映結果
- wav長→Frames自動調整のUI(自動反映時の表示・事前チェックのエラーメッセージ)
- Excelなど他アプリによるCSVロック時のautosave退避動作(`WRITE_LOCKED`発生時の実際のユーザー体験)

## 10. グループ3(設計判断項目＋品質負債) — 2026-07-15

`FRONTEND_CATCHUP_WORKORDER.md`の項目11〜14。着手前にオーナーへ設計判断を仰ぎ、確定した方針で実装した。

### 10.1 項目12: Width/Heightスナップをgradio式へ統一

数値を64(または参照動画IC-LoRA時は128)の倍数へ丸める「スナップ」の方式を、Gradio側に合わせた。

- 従来: 入力のたびに常時スナップ(1文字打つそばから丸められる)。
- 現行: 手入力は自由に素通し、ステッパー矢印・上下キー・スライダー操作のときのみスナップ。Gradioが`<input type=number step min>`のブラウザ標準挙動に委ねているのと同じ体験。
- 実装: `paramUtils.ts`に純関数`isStepEvent`(onChangeの`nativeEvent.inputType`が空ならステッパー由来と判定。WebView2はChromiumなのでこの挙動に依拠できる)と`isDimensionOnGrid`(送信前バリデーション)を追加。`useGenerationForm`/`useChainForm`の`setWidth`/`setHeight`に`snap`引数を足し、手入力時は丸めず、送信は`isValid`のグリッド判定でゲートする。Createは64/128の切替、Chainは常に64。

### 10.2 項目11: 単発A2V(音声から動画を生成する機能)をCreate画面へ移設

オーナー確定=(1)Gradio式の任意音声添付欄、(2)distilled固定パラメータ、(3)ChainからA2Vを削除。

- Create画面に「音声を添付」欄を常設。音声を添付すると自動的にA2Vモードになり、`buildA2vChainPayload`(バッチA2Vと共有・Gradioハンドラのbyte移植・distilled固定=推論8ステップ/guidance1.0/overlap 3/0.5)を使って`POST /generate/chain`(単一クリップ)へ`submitChain`で送信する。`A2vChainPayload`は`GenerateChainRequest`の構造的スーパーセットのため、API型定義(`api/types.ts`)の改修もbatchの改変も不要だった。
- wav長を`fs.probeAudioDuration`で実測してnumFramesを自動調整。音声が短すぎる場合は送信前に必要秒数つきで拒否。
- 三者排他: 音声＋I2Vキーフレームは併用可、音声とIC-LoRA参照動画は相互排他(A2Vは/generate/chainのdistilled・64グリッド、IC-LoRAは/generateの128グリッドで曖昧な第4モードを作らない)。
- ChainからA2Vサブモードを削除し、サブモードはClips(クリップ連結)とV2V(動画の続き)の2択に。音声の尺計算の純関数(`suggestFramesForAudio`等)は`chainUtils`に残し、Createとバッチが共有する。
- i18n文言は`chain.sourceAudio`/`chain.subModes.a2v`/陳腐化した`chain.comingSoon`を削除し、`create.sourceAudio`を新設(英日両方)。

### 10.3 項目14: README等の更新負債

ルート`README.md`・`webui/README.md`のM1マイルストーン表記を廃し、現行1.0.0-rc1の機能一覧へ再編。`CMakeLists.txt`の`project`バージョンを0.1.0→1.0.0(CMakeは数値のみで`-rc1`接尾辞不可)へ更新し、完全版バージョン文字列の一次ソースは`native/src/bridge_core.h`の`kPluginVersion`と`scripts/package.ps1 -Version`である旨をコメントで明記して二重管理を解消した。

### 10.4 項目13: 実バックエンド(実GPU)での通し確認 — 準備まで

実GPU生成での目視確認はオーナーが後日まとめて実施する前提のため、本セッションでは準備(ブロッカー除去＋手順書)までを完了とした。

- §9.9-3で残課題としていた`scripts/deploy.ps1`(既定PluginDir)・`scripts/run-aviutl.ps1`(既定ExePath)の旧環境パスを、AviUtl2 v2.0.54ポータブル構成(`D:\For_Videos\AviUtl2\aviutl2_v2.0.54\`)へ恒久修正した(オーナー承認済み。AviUtl2はCドライブ容量都合でポータブル固定運用)。これで`-PluginDir`/`-ExePath`の毎回指定が不要になった。
- `Docs/REAL_BACKEND_CHECKLIST.md`を新規作成。起動手順(バックエンド既存手順の引用)、mock-real挙動差2点の優先再検証(§4-7/§4-8)、実GPU通しマトリクス(T2V/I2V/IC-LoRA/Chain Clips/Chain V2V/単発A2V/バッチA2V/モデルロード/chunked_upsample)、記録欄を含む。

### 10.5 品質ゲート

`npm run typecheck`(=`tsc -b`)エラー0、vitest 48ファイル548件パス(10 skip)、`npm run build`成功。実装はサブエージェントに1ファイル1所有者で分割し、各Wave末に親が統合ゲートを回した。§9.9-3のスクリプトパス残課題は本グループで解消済み。実GPU通しはオーナー後日実行の最終ゲートとして残置。

## 11. ドキュメント整備とパリティ再分析(2026-07-15、グループ3実装後)

グループ1〜3の実装(§8〜§10)が完了した後、次セッションでの着手準備として、ドキュメント整備とGradio GUI(検証用の簡易画面)↔フロントエンドのパリティ再棚卸しを行った。**本セクションで扱う作業はドキュメント整備のみであり、コード実装は本セッションでは行っていない(次セッションで着手予定)。**バックエンド(`Nz-Videomni`)は引き続き一切変更していない(凍結方針を維持。ただし追跡専用の設計文書1本のみ§11.1末尾の通り追随)。

### 11.1 ドキュメント監査と修正

批判的レビュー3体でDocs全体を現行コードと突合し、更新漏れ・矛盾・不明瞭点を修正した(フロント コミット`18bff11`)。主な修正内容は以下の通り。

- [WEBVIEW2_PARITY_BACKLOG.md](WEBVIEW2_PARITY_BACKLOG.md): 旧・項目1(A2VのCreate画面移設)・項目6(Width/Heightスナップ方式統一)がグループ3で解決済みであるにもかかわらず「判断待ち」のままだった消し込み漏れを是正。
- [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md): グループ1〜3の完了節を追加し、対象実機環境をbeta52からv2.0.54ポータブル構成へ更新。
- [BRIDGE_CONTRACT.md](BRIDGE_CONTRACT.md): §0の「native実装はv4.1まで」という記載がv5・v6まで到達済みの実態と食い違っていた自己矛盾を是正。
- [API_REFERENCE.md](API_REFERENCE.md): §3.3の`POST /pipeline/load`応答に`models`エコー欄の記載漏れがあったのを追記。
- `CMakeLists.txt`: コメントに残っていた「milestone M7c」という陳腐化した表記を削除。
- `README.md`: IC-LoRA(In-Context LoRA、輪郭線や骨格などの手がかりを読み取って動画の動きを制御する追加学習アダプタ)の参照動画添付が、Create画面のUI上ではcontrol LoRA同伴を強制しておらず単独指定時に422エラーになり得ることを明示。
- [SDK_REFERENCE.md](SDK_REFERENCE.md): 右クリックメニュー系のAviUtl2 SDK API(`register_object_menu`等)を追記。
- [TIMELINE_ALPHA_REQUIREMENTS.md](TIMELINE_ALPHA_REQUIREMENTS.md)・[WEB_RESEARCH.md](WEB_RESEARCH.md): 実機対象がbeta52からv2.0.54へ移行した旨の注記を追加。

あわせて、バックエンド側の追跡専用デザイン文書`AVIUTL2_DESIGN_BRIEF.md`(バックエンド版。単発A2VをChainのサブモードと位置づけていた旧デザイン)の§3-5・§6も、グループ3・項目11で確定したCreate画面移設に追随させた(backend コミット`dcd72ee`。ドキュメントのみで`api`/`engine`は不変)。

### 11.2 未着手タスク台帳の新設

散在していた未着手タスクを一元化するため、`Docs/PENDING_TASKS.md`を新設した。「近日中の改修項目」(実バックエンド(real backend、実GPU接続の本物の生成エンジン)通しゲート／IC-LoRA参照動画のCreate画面UI非強制422／右クリックメニュー系SDK APIのv2.0.54実機再検証／オーナー目視確認の残項目／`webui/package.json`のversionが`0.0.0`のまま／バッチA2Vの共有キーフレーム画像、§6として仕様確定済みのパリティ取りこぼしゆえ格上げ。※当時の§6。4段階再編後の現在は§3-2で完了・クローズ済み)と、「将来の研究課題」(バッチA2Vのα版で意図的に省略した機能・`two_stage_hq`〔非量子化モデル用の高品質パイプラインモード〕・backend側の研究文書リンク集)の二段階構成である。**次セッションでの未着手タスクの入口はこの台帳**である旨を本書に明記している。

### 11.3 パリティバックログの再構成・再優先度化

`Docs/WEBVIEW2_PARITY_BACKLOG.md`を、「実装済み(旧・項目1〜7、および§8-a〜8-g=グループ1〜3で決着)」と「未実装の改修項目 N1〜N13(2026-07-15の棚卸しで新規発見)」の2部構成に再編した。N1〜N13の主な内容はCrop出力(`crop_output`)、IC-LoRAコントロールLoRAの選択ドロップダウン、Create画面のIC-LoRA強度スライダー(`control_adherence`/`reference_strength`)、設定の「危険ゾーン」(パイプラインunload・完了ジョブ一括purge)、バッチ表の行内編集、V2V結合のクロスフェード長(`crossfade_ms`)、バッチA2VのShared共有キーフレーム画像などである。なお、このうちバッチA2VのShared共有キーフレーム画像(N7)だけは完全な新規発見ではなく、§9.8で「意図的なα省略」として記録済みだった項目を、仕様確定済みのパリティ取りこぼしとして再分類・格上げしたものである(`Docs/PENDING_TASKS.md`§6と相互参照。※当時の§6。現在は§3-2)。

オーナー方針により、優先度の軸を従来の「機能重要性」から**「実装難易度＋依存順序」**へ組み替えた。各項目に「難易度(小/中/大)」と「依存関係」の2行を持たせ、末尾に4段階の推奨実装順(第1陣クイックウィン→第2陣・N3→N2の横展開→第3陣・共有UIコンポーネント新設→第4陣・独立中規模項目)と依存マップを付与した(旧版は5段階想定だったが、後述のN12クローズにより旧・第5陣が消滅し実質4段階になっている)。**次セッションのパリティ実装の入口はこの「未実装の改修項目」セクション**である。

- N2・N4・N10・N13の「要確認」としていた前提を、コード調査によりいずれも解決した。**4件ともバックエンド変更ゼロ**で着手できることを確認済みである。
  - N2: IC-LoRAのデータ源はA案(`/config`の`model.ic_loras`キーを列挙)に決定した(Gradio側の選定方式と一致)。
  - N4: `POST /api/v1/pipeline/unload`がバックエンドに実在することを確認した(`ApiClient`へ1メソッド追加のみで着手可)。
  - N13: `/config`の`server.api_key`から設定有無を導出可能であることを確認した(api-keyバッジは実質第1陣級のクイックウィンへ前倒し可能)。
  - N10: `/config`は8セクション全てを返すことを確認した(生JSONビューアは既存データの整形表示のみで足りる)。
- N12(`two_stage_hq`。非量子化モデル向けの高品質パイプラインモード)は、オーナー決定により**将来の研究課題へ降格・クローズ**した。非量子化モデルを動かせるハイスペックな開発マシンをオーナーが未所有のため保留し、将来調達した際にバックエンド側`engine/`から再開発する方針。詳細は`Docs/PENDING_TASKS.md`と当該バックログのN12項目を参照。この結果、**能動的な実装対象はN1〜N11・N13の12件**である。（※この「12件」はN12クローズ時点の史料。その後2026-07-15にN9〔ポーリング間隔・タイムアウトのユーザー設定欄〕も将来の研究課題へ降格したため、**能動対象は11件＝N1〜N8・N10・N11・N13に更新済み**。経緯は§12.4を参照。）

### 11.4 次セッションへの申し送り

本セッションでは実装に着手していない(ドキュメント整備のみ)。次セッションで着手する方針とし、入口を以下の2点に明示した。

1. `Docs/PENDING_TASKS.md` — 全未着手タスクの「近日中の改修項目」「将来の研究課題」二段階台帳。
2. `Docs/WEBVIEW2_PARITY_BACKLOG.md`の「未実装の改修項目」節 — N1〜N13の難易度・依存関係・推奨実装順。

実バックエンド(real backend)での通し確認は、`Docs/REAL_BACKEND_CHECKLIST.md`の手順書に従いオーナーが後日実施する最終ゲートとして、従来どおり残置している。

## 12. パリティ実装 第1〜3陣(N6/N11/N10/N3/N2/N1/N7、2026-07-15)

§11でドキュメント整備のみ済ませて止まっていた「未実装の改修項目 N1〜N13」(2026-07-15の棚卸しで新規発見したGradio GUI↔フロントエンドの機能差分)について、前任が定めた推奨実装順の**第1陣〜第3陣の計7項目**を本セッションで実装した。着手順は`Docs/WEBVIEW2_PARITY_BACKLOG.md`「推奨実装順」節に従い、第1陣(N6・N11・N10)→第2陣(N3→N2)→第3陣(N1・N7)の順に進めた。

進め方は監督役方式を採った。親(Opus 4.8)は監督に専念し、陣ごとに専任のサブエージェントを直列に立て、各陣の着手前に敵対的レビュー(前提を疑う批判的レビュー)と、一次情報(バックエンドの実コード・SDK)およびフロントエンド実コードでの裏取りを行ってから実装に入った。バックエンド(`Nz-Videomni`)は凍結方針を維持し、本セッションでも一切変更していない(7項目すべてバックエンド変更ゼロで実現)。第1〜3陣のコードはオーナーがコミット・push済み(コミット`e647d01`「1st-3rd jin」、`origin/main`と同期)。

### 12.1 第1陣: N6・N11・N10(クイックウィン)

- **N6 V2V結合のクロスフェード長**: 動画どうしをつなぐV2V結合で、境目の音声クロスフェード長を150/300/500msから選ぶドロップダウンを`JoinControls`に追加した。送信フィールドは`handle_crossfade_ms`(バックエンドの`JoinRequest`のフィールド名。従来フロントは値を渡さずサーバー既定の300ms固定だった)。型と`apiClient.joinJob(jobId, body)`のプラミングは既存で揃っていたため、UIとbody組み立ての追加のみ。
- **N11 spill-free一覧表**: 設定画面に、解像度ごとの快適フレーム上限(spill-free frames、VRAMがあふれずに生成できるフレーム数の目安)の一覧表を追加した。データ`config.limits.spill_free_frames`は既存の`AppConfig`にあり、表として描画するのみ。
- **N10 生の`/config` JSONビューア**: 設定画面に、バックエンドの`/config`(設定情報)応答の生JSONをそのまま整形表示するビューアを追加した。
- N11・N10はいずれもSettingsPanel(設定パネル)のサブセクションとして実装し、両者で`useConfig()`(設定取得フック)を1回だけ呼んで共有する形にまとめた。

### 12.2 第2陣: N3→N2(Create画面のIC-LoRA周り)

IC-LoRA(In-Context LoRA、輪郭線や骨格などの手がかりを読み取って動画の動きを制御する追加学習アダプタ)関連の2項目を、依存順どおりN3を先に、その土台の上にN2を乗せる形で実装した。

- **N3 Create画面のIC-LoRA強度スライダー**: Create画面(生成画面)に、`conditioning_attention_strength`(制御信号にどれだけ厳密に従うか)と`reference_video_strength`(参照動画の効き具合)の2スライダーを追加した。既にChain画面(§8.4のα版対応分)に実装済みだった`ReferenceStrengthField`をChain固有から共有コンポーネントへ抽出し、Createへ移植した。あわせて**「参照動画を使うならcontrol LoRA(`loras`)が非空でないと送信できない」ゲートを新設**し、従来`reference_video_id`だけを無条件送信していた潜在バグ(`loras`が空だとバックエンドが422〔検証エラー〕で拒否し得た)を是正した。
- **N2 コントロールLoRA選択ドロップダウン**: Create画面に、コントロールLoRA(IC-LoRA)を選ぶドロップダウンを追加した(Chainには今回は載せず、Createのみ)。選択肢は`/config`の`model.ic_loras`のキーから生成し(バックエンドのGradio側と同じA案。バックエンド変更不要)、選ぶと`appendLoraTag`でプロンプトへ`<lora:名:1.0>`タグを注入する。型は`AppConfig`に`model?`をoptionalで追加し、参照は`config.model?.ic_loras`と防御的アクセスにして、mock(モックバックエンド)へ`ic_loras`のfixture(テスト用データ)を追加した。あわせて**「IC-LoRA選択時は参照動画が未設定だと送信できない」ゲートも実装**した(LTXのIC-LoRAは参照動画が必須で、無いと422になるため)。この結果、`Docs/PENDING_TASKS.md`§2(Create画面のIC-LoRA参照動画のUI非強制422。※当時の§2。4段階再編後の現在は§3-1)も同時にクローズした。
  - 計画外に必要になった変更として、ドロップダウンの選択結果をプロンプト文字列へ書き戻すため、`CreateScreen.tsx`と`AppShell.tsx`に`onPromptChange`の配線を追加した点を記録しておく。

### 12.3 第3陣: N1・N7(共有UIコンポーネント新設／デッドコード配線)

- **N1 Crop出力欄**: Create画面とChain画面に、最終出力を指定サイズへ切り出すCrop出力(`crop_output`)の入力欄を追加した。共有コンポーネント`CropOutputField`を`CommonGenerationFields.tsx`に新設し、両画面へ同じ部品を配線している。**LTXはVAE(動画を圧縮・復元する部品)の空間圧縮率が32倍のため、出力解像度が32で割り切れる必要がある**。これに合わせて入力値は32刻みにスナップし、さらに生成サイズを超えないようclamp(上限で頭打ち)する検証を入れた。バッチA2V画面のCrop対応は今回のスコープからは見送っている。
- **N7 バッチA2VのShared共有キーフレーム画像**: バッチA2V(音声フォルダを一括指定して複数のA2Vジョブを順に流す機能)のCSVで`image`列が「Shared(共有)」の行に、共有キーフレーム画像群を条件付けとして流し込む配線を実装した。バックエンド相当のロジックを写した純関数`resolveConditioningImages`は既に存在していたが本番未配線のデッドコードだったため、これを`batchRunner.ts`へ本番配線した。**Gradio原典`batch.py`の`_validate()`とパリティを取り**、(a)Shared行/空行はアップロードしないというガード順序を原典に合わせ、(b)共有キーフレーム必須ゲート(`frame_idx=0`のスロット1が1枚必須)は実行対象行(UNFINISHED_STATS〔未完了状態の集合〕)に限定して`canStart`(実行開始可否)と共有する形にした。

### 12.4 N9降格(第1〜3陣とは別のオーナー決定)

第1〜3陣の実装とあわせて、N9(ポーリング間隔・タイムアウトのユーザー設定欄)を、オーナー決定により能動的な実装対象から外し「将来の研究課題」へ降格した。現物のコードでフロントのジョブ進捗ポーリングが堅牢(常設ポーリングは`GET /jobs`を2秒間隔、投げたジョブ専用は`GET /jobs/{id}`を1秒間隔で回し、無期限だが進捗が止まる不具合なし。本番`console.*`は0件)であり、機能的問題のない見た目のパリティにとどまるためである。復活条件は「ポーリングまわりで不具合を発見した場合、またはユーザーからの要望があった場合」。詳細は`Docs/PENDING_TASKS.md`「将来の研究課題」節および当該バックログのN9項目を参照。

### 12.5 品質ゲート

`npm run typecheck`(=`tsc -b`)エラー0、`npm test`(vitest)553件緑、`npm run build:single`(WebView2向けの単一HTMLビルド)成功、lintの新規警告なし。実機ゲート(モックバックエンドでの目視確認と実GPUでのE2E通し確認)はオーナーが後日実施する。

### 12.6 次セッションへの申し送り(第4陣・前半 = N13 ＋ N4 の着手ブリーフ)

次に着手するのは**第4陣・前半、すなわちN13(APIキーバッジ)とN4(危険ゾーン)**である。第4陣は独立・中規模の4件(N4・N5・N8・N13)で、後半はN5→N8の順(N8のテーマ切替は新規インフラ導入を伴うため最後)。着手の入口は`Docs/WEBVIEW2_PARITY_BACKLOG.md`の「推奨実装順」節。

**N13(APIキーバッジ)の確定事実**:

- 現状`AppConfig`(`api/types.ts`)には`server`セクションが無い。`server?: { api_key?: string | null }`をoptionalで追加する(バックエンドの`config.py`の`ServerConfig.api_key`が`/config`に載っているため、**バックエンド変更は不要**)。
- SettingsPanelの既存`configState`(`useConfig()`。N11/N10で既に共有済み)から`config.server?.api_key`を読み、**「設定済みか否か」の真偽値だけをバッジ表示する。APIキーの値そのものは絶対に画面へ出さないこと**(現状`/config`はapi-keyの値そのものをワイヤに載せている留意点があるため、なおさら表示しない)。設置場所はSettingsPanelのconfig ready(設定取得完了)ブロック。

**N4(危険ゾーン)の確定事実**:

- パイプラインunload(生成モデルをVRAMから降ろす操作)は`POST /api/v1/pipeline/unload`を叩く。このメソッドは`api/client.ts`に**未実装のため新規追加が必要**(バックエンドの`api/pipeline.py`に実在。ジョブ実行中は409〔競合〕を返す)。
- 完了ジョブの一括purge(まとめて削除)は、新エンドポイントを足さず、既存の`listJobs`＋`deleteJob`を完了ジョブへループ適用して実現する。
- 設置場所はSettingsPanelの独立サブセクション。

**直列必須の理由と運用**: N4とN13はいずれも同じ`SettingsPanel.tsx`と`i18n/strings.ts`(en/jaのペア。`strings.test.ts`がキーの完全一致を厳格に検査する)を編集するため、**並行させず直列で実装する**こと。監督はOpus 4.8親、陣ごとに専任サブエージェントを立てる方式を継続する。品質ゲートは`npm run typecheck`(tsc -b)＋`npm test`(vitest)、必要に応じて`npm run build:single`。

**git状態の申し送り**: 第1〜3陣のコードはコミット`e647d01`でコミット・push済み(`origin/main`と同期)。その上に、後述§12.7のN10ビューア伏字化修正(`redactConfig.ts`/`redactConfig.test.ts`/`SettingsPanel.tsx`)の**コードの未コミット差分**が積まれている。したがって**現在の未コミットは「ドキュメント数点(N9降格の2文書＋本§12の追記等)＋N10伏字化のコード修正一式」**である。次担当は着手前に、これらのコミット状況(オーナーが既にコミット済みか否か)をオーナーに確認すること。コミットはオーナーが行う。

### 12.7 敵対的レビュー由来の是正: N10ビューアの`server.api_key`伏字化(2026-07-15)

第1〜3陣の実装後に行った敵対的ドキュメントレビューで、N10(生の`/config` JSONビューア)が設定情報をそのまま表示するため`server.api_key`(APIキーの値)まで画面に平文表示してしまう、という漏れ(重大度HIGH)が判明した。これはN13(APIキーバッジ)で定めた「鍵の値そのものは画面に出さない」設計意図とも矛盾するため、オーナー承認のうえコードで是正した。

- 新規`webui/src/shell/redactConfig.ts`に純関数`redactConfigForDisplay(config)`を新設。表示直前にconfigをディープコピー(深い複製)し、`server.api_key`が非null・非空文字列なら固定トークン`"***"`(鍵長も漏らさない)へ置換する。元オブジェクトは非破壊。
- `webui/src/shell/SettingsPanel.tsx`のN10ビューアを、`JSON.stringify(redactConfigForDisplay(configState.config), null, 2)`で描画するよう変更した。
- 新規テスト`webui/src/shell/redactConfig.test.ts`(4ケース)を追加。
- この是正により、N10ビューアとN13バッジの表示方針が「鍵の値は画面に出さない」で一貫した(次担当がN13着手時に「N10がまだ平文で出しているのでは」と混乱しないよう、§12.6のN13ブリーフとあわせて参照)。

### 12.8 第4陣・前半 N13/N4 実装(2026-07-15)

§12.6の申し送りに従い、第4陣・前半のN13(APIキーバッジ)→N4(危険ゾーン)を、同じ`SettingsPanel.tsx`と`i18n/strings.ts`を編集するため直列で実装した。進め方は前任と同じ監督役方式(親=Opus 4.8が監督に専念し、陣ごとに専任サブエージェントを立てる)を継続し、着手前に敵対的レビューと一次情報の裏取りを行った。バックエンド(`Nz-Videomni`)は本セッションでも一切変更していない(N13・N4ともバックエンド変更ゼロで実現)。

**N13 APIキーバッジ**:

- SettingsPanelに「APIキー設定済み／未設定／不明」のバッジを追加した。`/config`の`server.api_key`の**有無(真偽)だけ**を判定して表示し、値そのものは一切画面に出さない。
- 判定ロジックは新規純関数`shell/apiKeyStatus.ts`に切り出した。値が非null・非空文字列なら`set`(設定済み)、明示的にnull・空文字なら`unset`(未設定)、フォールバック構成(usingFallback。サーバー未接続などで実際の設定を確認できていない状態)を使っている場合は`unknown`(不明)を返す3値の設計にしている。この判定規約は、N10是正(§12.7)で新設した`redactConfig.ts`の判定条件と一致させた。
- バッジの見た目には中立色の新規スタイル`.badge-neutral`を`App.css`に追加した(既存の成功・失敗系バッジ色と混同しないため)。
- 型面では`api/types.ts`の`AppConfig`に`server?: { api_key?: string | null }`をoptionalで追加した。

**N4 危険ゾーン(パイプライン解放・完了ジョブ一括削除)**:

- SettingsPanelの最下部に「危険ゾーン」節を新設した。
- (1) パイプライン解放(unload。生成モデルをVRAMから降ろす操作): `api/client.ts`に`unloadPipeline()`を新規追加し、`POST /api/v1/pipeline/unload`を叩く。あわせて`bridge/mockBridge.ts`に`handlePipelineUnload`を追加してモックバックエンドでも疎通できるようにした。ジョブ実行中は409(`JOB_BUSY`)が返るため、その場合は「解放できません」とインライン表示する。
- (2) 完了ジョブ一括削除(purge): 新規エンドポイントは追加せず、`listJobs`から終了3状態(completed/failed/cancelled。実行中・待機中は対象外)のジョブを抽出し、既存の`deleteJob`を1件ずつ逐次呼び出す形で実装した。Gradioの`delete_finished_jobs`と同仕様で、1件の削除失敗が他件を止めないよう個別にtry/catchで例外を握りつぶし、成功件数を数える。結果表示は「対象なし／全件成功／部分・全失敗(purgePartial)」の3分岐とし、SettingsPanelの既存の慣習に合わせトーストではなくインライン表示にした。
- 新規`shell/useDangerZone.ts`(ロジック)と`shell/DangerZonePanel.tsx`(表示)を追加した。

**敵対的レビュー(Opus)の結果と是正**:

- レビューでは「APIキーの値が画面上のどこにも露出していないか」「purgeが実行中(running)・待機中(queued)のジョブを誤って削除しないか」の2点を重点確認し、いずれも問題なしと確認した。
- 指摘M1: 「終了ジョブが1件以上あるのに全件の削除が失敗した場合に、結果表示が"対象なし"になってしまい実際には失敗が起きたことが伝わらない」という表示上の不備を指摘された。これを是正するため、結果種別に`purgePartial`(部分・全失敗)を新設し、「対象なし」は本当に終了ジョブが0件だった場合に限定するよう分岐を修正した。

**品質ゲート**: `npm run typecheck`(=`tsc -b`)エラー0、`npm test`(vitest)569件緑・10件skip、`npm run build:single`成功。実機ゲート(WebView2での目視確認、実GPUでのE2E通し確認)はオーナーが後日実施する。

**次セッションへの申し送り(第4陣・後半 = N5 → N8)**: 残る能動的な実装対象は第4陣・後半のN5(バッチ表の行内編集)とN8(テーマ切替dark/light)の2件のみ。`Docs/WEBVIEW2_PARITY_BACKLOG.md`「推奨実装順」節の記載どおり、N8は新規インフラ(CSS変数＋ルート属性＋永続化)導入を伴うため、N5→N8の順で着手するのが妥当。N5・N8はいずれも独立項目で、他N項目の土台にはならない。
- 品質: `npm run typecheck`0エラー、vitest 557件緑(4件増)、`npm run build:single`成功。**未コミット**(上記§12.6のgit状態の申し送りのとおり、`e647d01`の上に積んだコード差分。コミットはオーナーが後で行う)。

### 12.9 第4陣・後半 N5/N8 実装(2026-07-15)

§12.8の申し送りに従い、第4陣・後半のN5(バッチ表の行内編集)→N8(テーマ切替dark/light)を、記載順どおり直列で実装した。**これで第4陣(N4・N5・N8・N13)は全完了であり、2026-07-15の棚卸しで発見したN1〜N13のうち能動的な実装対象だった11件(N1〜N8・N10・N11・N13)がすべて実装済みになった**(残るはN9=将来の研究課題・N12=クローズのみで、いずれも能動的な実装スケジュールには乗らない)。進め方は前任と同じ監督役方式(親=Opus 4.8が監督に専念し、項目ごとに専任サブエージェントを立てる)を継続し、着手前に一次情報(バックエンドの実コード)の裏取りを行った。バックエンド(`Nz-Videomni`)は本セッションでも一切変更していない(N5・N8ともバックエンド変更ゼロで実現)。

**N5 バッチ表の行内編集**:

- バッチ画面の各行(`webui/src/modes/batch/BatchTable.tsx`)に3つのインライン編集操作を追加した。
  - **①プロンプト直接編集**: 行の`prompt`セルを`<input>`にした。`onChange`で新規`setRowPromptLocal`(`useBatchForm.ts`)がin-memoryの`rows`state を即座に更新するのみでCSVへは書き込まず、`onBlur`で新規`commitRows`が既存の`flushManifest`(アトミック書込み＋`WRITE_LOCKED`時autosaveフォールバックを持つ既存関数、変更なし)を呼んでCSVへ確定する2段構えにし、打鍵のたびにCSVへ書き込むI/O過剰を避けている。
  - **②共通プロンプト流し込み**: 各行に追加したボタンで、Create画面の共通プロンプト(`prompt`引数として`BatchSection.tsx`から渡される)をその行の`prompt`へコピーする。新規`copyCommonPromptToRow`はボタン押下という離散操作であるため、①とは異なりコピーと同時に`flushManifest`を呼び即時反映する。
  - **③行別画像割当**: 行の`image`セルを`<select>`にした。選択肢は`scan()`実行時に`fs.listFiles`(既存の契約v6ブリッジメソッド、変更なし)で画像フォルダを列挙し名前順にソートしたものに、先頭固定で`Shared`を付与する(新規`imageOptions`)。選択値が現在値と同じ場合は`updateRowImage`が`setRows`/`flushManifest`を呼ばずno-opにし、空文字選択は`Shared`へ正規化する。
- 行の`prompt`列はCSV仕様(`BATCH_A2V_CSV_SPEC.md`)§8に従い、共通プロンプトと異なり`parseLoraPrompt`を通さず生文字列のまま保存する(`<lora:...>`タグの解釈対象外)。
- 編集は生成実行中(`runnerState !== "idle"`)に加え、スキャン中(`isScanning`)も無効化する(`BatchSection.tsx`の`disabled`ゲート)。
- 新規i18n文字列(`promptInputLabel`／`copyCommonPromptButton`／`imageSelectLabel`)を`webui/src/i18n/strings.ts`のen/ja双方に追加した。
- バックエンド・ブリッジ契約の変更はゼロ(`fs.listFiles`は既存の契約v6メソッドをそのまま再利用)。

**N8 テーマ切替dark/light**:

- 設定画面(`SettingsPanel.tsx`)に、既存の言語トグルの隣にdark/lightトグルを追加した。
- 新規`webui/src/shell/ThemeContext.tsx`に`ThemeProvider`／`useTheme`を、`i18n/LanguageContext.tsx`と構造的に同型(コンテキスト＋プロバイダ＋フック＋`localStorage`読み書きヘルパー)で実装した。`AppShell.tsx`の最外周(`LanguageProvider`よりさらに外側)に配置している。
- 永続化は`localStorage`単独(新規`THEME_STORAGE_KEY = "nzvideomni.theme"`、既定`"dark"`)とし、`useSettings.ts`(ネイティブブリッジ`settings.get`/`settings.set`経由の永続化基盤)は使わない設計判断とした。言語選択の既存実装が同じく`localStorage`単独で完結している前例に倣ったもので、native側(C++)・ブリッジ契約・`mockBridge`はいずれも変更していない。
- `index.css`を、`:root`をdark既定の色変数群、`:root[data-theme="light"]`をその上書きという構成に再編した。新規`--warning`／`--warning-bg`／`--warning-border`のCSS変数を追加し、`CreateScreen.css`(`.warning-banner-mild`／`.badge-busy`)と`jobs.css`(`.badge-running`)に直値(`#d9a441`等)で散在していたamber系の警告色をこの変数へ置き換えた。`App.css`の`.primary-button:hover`のホバー色も、テーマ固有のhex直値から`filter: brightness(1.15)`(現在の`--accent`から導出)へ変更している。
- **動画プレビューの黒背景(`CreateScreen.css`の`.result-video { background: #000; }`、`jobs.css`のサムネイル背景)とボタンの白文字(`App.css`／`SettingsPanel.css`の`color: #fff`)は両テーマ共通で意図的に据え置いた**(テーマ変数化していない)。
- `main.tsx`で、`createRoot(...).render(...)`の前に`applyThemeToDocument(readStoredTheme())`を同期呼び出しし、`<html data-theme="...">`を初回描画前に確定させることで、永続化済みの設定が`light`の場合に一瞬darkが表示されるちらつき(FOUC)を回避している。
- lightパレットの色調はオーナー承認済み。

**敵対的レビュー(Opus)の結果と是正**:

レビューでは機能破壊級のバグは見つからなかったが、以下3点の指摘を受けて是正した(コード中のコメントにも同じ呼称で記録している)。

- **指摘M1(画像フォルダ切替時に旧一覧が残る)**: 画像フォルダを`pickImgDir`で切り替える、または`clearImgDir`でクリアすると、直前フォルダでスキャンした`imageFileNames`が残ったままになり、既に存在しないフォルダのファイル名を`<select>`が選択可能なまま提示し続けてしまう欠陥があった。両関数で`imageFileNames`を即座に空へリセットし、次の`scan()`まで選択肢を`Shared`のみへ収縮させるよう是正した。
- **指摘M2(プロンプト打鍵ごとのCSV書き込み過剰)**: 初期実装ではプロンプト入力の`onChange`のたびに`flushManifest`を呼んでおり、CSVへのI/Oが打鍵回数分発生する設計になっていた。前述の①のとおり、`onChange`は state更新のみ・`onBlur`で確定書込みという2段構えへ是正した。
- **指摘L1(スキャン中の編集ゲート)**: `disabled`ゲートが生成実行中(`runnerState !== "idle"`)のみを見ており、スキャン中(`isScanning`)の非同期マージ(`setRows`)が編集中の値を上書きしてしまう競合の余地があった。`BatchSection.tsx`の`disabled`条件へ`|| form.isScanning`を追加して是正した。

**品質ゲート**: `npm run typecheck`(=`tsc -b`)エラー0、`npm test`(vitest)581件緑・10件skip、`npm run build:single`成功。

**残るゲート(オーナー後日)**: WebView2実機での行内編集・CSV往復・light/dark目視、実GPUでの通し確認(`Docs/REAL_BACKEND_CHECKLIST.md`)は、従来どおりオーナーが後日まとめて実施する最終ゲートとして残置している。

**git状態の申し送り**: 本セッションのコード差分(`webui/src/shell/ThemeContext.tsx`/`ThemeContext.test.tsx`の新規2ファイル、`BatchTable.tsx`/`useBatchForm.ts`/`useBatchForm.test.ts`/`BatchSection.tsx`/`BatchSection.css`/`i18n/strings.ts`/`index.css`/`main.tsx`/`AppShell.tsx`/`App.css`/`jobs.css`/`CreateScreen.css`/`SettingsPanel.tsx`/`SettingsPanel.test.tsx`の既存ファイル改修)は、本節を記録した時点でまだコミットされていない。コミットはオーナーが行う。

**結論**: 2026-07-15の棚卸しで発見したN1〜N13のうち、能動的な実装対象だった11件(N1〜N8・N10・N11・N13)はすべて実装が完了した。残るのはN9(将来の研究課題へ降格)とN12(クローズ)のみ。次にフロントエンド側で着手すべき能動的な実装項目は無く、残るのは`Docs/PENDING_TASKS.md`および`Docs/REAL_BACKEND_CHECKLIST.md`に記載のオーナー後日ゲート(実機目視・実GPU通し)のみである。

## 13. 大規模UIリデザイン(2026-07-16〜17、2日間)

パリティ実装(§8〜§12)がひととおり片付いた後、オーナーの実機使用感を踏まえて、UI全体のレイアウトと情報設計を作り直す大規模リデザインを2日間で実施した。機能の追加ではなく、既存機能の**置き場所・見せ方・操作モデルの作り直し**が主眼である。実機(AviUtl2 v2.0.54ポータブル構成)へデプロイして挙動を確認済みだが、**本節を記録した時点ではまだコミットされていない**(コミットは後続フェーズでオーナーが行う)。

このセッションは、従来「バックエンドは一切変更しない(凍結)」としてきた方針に対して**ログ出力のみの例外**を1点含む(後述§13.5)。`api`/`engine`など生成ロジックには一切手を入れていない。

### 13.1 第1弾(2026-07-16): Create / Chain / Batch の再構成

**Create画面**:

- **プリセットをタグボタンの列からドロップダウンへ**変更した(`PresetDropdown`)。CreateとChainで同じ部品を共用する(`config.generation_presets`駆動という従来のデータ源は不変)。
- **IC-LoRA(参照動画で動きを制御する機能)とA2V(音声から動画を生成する機能)の入力欄を、キーフレーム欄の下の折りたたみ(アコーディオン、初期は閉じた状態)へ移し、常設化した**。両アコーディオンは常に描画される(Gradio準拠)。
- 音声と参照動画の**排他を双方向化**した(どちらか一方を使っている間はもう一方を無効化)。
- キーフレーム欄に「n/5」カウンター(最大5枚のうち何枚使用中か)を追加。
- モードバッジ(`mode-badge`)にIC-LoRA使用中の表示を追加。
- 参照動画の選択をキャンセルした際に解像度が勝手に128グリッドへ丸められてしまうバグを修正。

**Chain画面**:

- **Clips / Continue video のサブタブを廃止**し、ソース動画(V2Vの元動画)が添付されているかどうかで**モードを自動判別**するようにした(添付あり=V2V継続、添付なし=ゼロから連結)。判別結果はモードバッジで表示する(`chain.modeBadge.v2v` / `chain.modeBadge.scratch`)。ソース動画スロットにクリアボタンを新設した。
- clip0(先頭クリップ)のキーフレームは、「ゼロから連結」時のみ表示される「Start frame(開始フレーム)」1枚欄(`StartFramePanel`)に改名・縮小した。
- **Chain側にあった参照動画(IC-LoRA)のUIは削除**し、Createへ一本化した(Chainの`ChainScreen.tsx`/`useChainForm.ts`から`reference_video`関連の参照は無くなっている)。
- 縦横比〜プリセットの数値ブロックを画面最上部へ移動した。

**Batch A2V画面**:

- 見出しから「(overnight)」表記を削除した。
- **W / H / FPS / SEED の独自入力欄を廃止し、Create本体の値へ一本化**した(`BatchGenerationValues`を注入)。あわせてガードを3点追加した(64グリッド検査／スキャン時のFPSスナップショット照合／IC-LoRA使用中の警告)。
- フォルダ3欄を「ラベル＋手打ち入力＋📁ボタン」形式へ変更した(onBlurで確定＋存在チェック)。

### 13.2 第1弾の共通変更

- Generateボタンを右カラムへ移した(`GenerateButtonBar`。※§13.3で右上パネルへさらに作り直す)。
- 780pxのメディアクエリを**コンテナクエリ**(実効幅680px)へ置き換えた。
- ダーク／ライトテーマ双方に対応した共通のアコーディオンCSSを追加した。

### 13.3 第2弾(2026-07-17): 予約機構の廃止とジョブ台帳への統合

実機使用感を踏まえ、ジョブ管理の操作モデルそのものを作り直した。

- **ジョブレーン(画面右端に常設していた縦長のサイドバー、`JobRail`)を廃止**した。
- **`GenerationPanel`(自分が投げた1件の状態だけを表示するパネル)を廃止**した。
- **予約機構(`useGenerationQueue`)を廃止**した。バックエンドは同時1ジョブ・キューを持たず・ビジー時は409で即拒否する仕様のため、「予約して現ジョブ完了後に自動送信」という方式をやめ、**実行中はGenerateボタンを「処理中…(busyButton)」で無効化して弾く**方式へ変更した(同時1ジョブという制約も、409応答の仕様も変わっていない)。
- **右上パネルの新構成**: 全幅のGenerateボタン＋所要時間の見積りヒント → (送信エラー行) → **`JobLedger`(全ジョブの台帳)**。台帳は`JobCard`ベースで、進捗表示／完了プレビュー(completedへ遷移したときだけ自動展開・以降は開閉トグル)／タイムライン挿入／V2V結合／キャンセル／削除を担う。完了トーストをクリックすると該当ジョブへスクロールする(Libraryタブ表示中はCreateへ切り替える)。
- 送信ロジックを送信専任の`useGenerationSubmit`へ絞った(`modes/create/useGeneration.ts`のエクスポートは`useGenerationSubmit`)。フォームは送信中のみdisableし、生成の実行中も編集は可能にした。
- レイアウト: `.app-body`を1カラム化。`.create-layout`は両側にキャップを置いて`space-between`とし、Generateボタンのあるパネルが画面の最も右上に来て、その右には何も表示しない形にした。台帳の高さは`clamp(240px, 60vh, 720px)`(狭幅では25vh)。

### 13.4 第2弾のその他

- **プリセットドロップダウンのソート**: 向き(横長→正方形→縦長)→面積の昇順→名前、の順に並べる。向きが混在するときだけ`optgroup`の見出しを出す。
- **ネイティブブリッジのordered_json化**: JSONのキー順が壊れる問題を根本から直すため、`native/src`の`bridge_core.h`/`bridge_core.cpp`/`bridge.cpp`/`plugin.cpp`を順序保持JSON(`nlohmann::ordered_json`)へ移行した。
- **i18n**: 予約系・`GenerationPanel`専用で使われなくなった死にキー10個を削除し、`busyButton`等を追加した(en/jaのキー完全一致は`strings.test.ts`が引き続き検査)。
- **既存設計文書への追補注記(2026-07-17実装フェーズで挿入済み)**: [API_REFERENCE.md](API_REFERENCE.md) §7、[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) M3、AVIUTL2_DESIGN_BRIEF.md(フロントMock版、[../Mock/AVIUTL2_DESIGN_BRIEF.md](../Mock/AVIUTL2_DESIGN_BRIEF.md)) §11に、予約機構・ジョブレーンの廃止とジョブ台帳への統合を示す追補注記を挿入した。

### 13.5 バックエンドの凍結例外(ログ出力のみ)

`Nz-Videomni/main.py`に、アクセスログを抑制するログフィルタ2種を追加した(`StatusAccessFilter`=`/api/v1/status`ポーリングのGET 200を抑制、`JobsListAccessFilter`=`/jobs`一覧のGET 200を抑制)。生成ロジック(`api`/`engine`)には一切手を入れておらず、変更はログ出力の抑制のみである。従来の「バックエンド完全凍結」からの例外だが、機能・契約への影響は無い。

### 13.6 検証と状態

- 実機(AviUtl2 v2.0.54ポータブル構成)へデプロイして挙動を確認済み。
- **未コミット**。コミットは後続フェーズでオーナーが行う。
- 自動テスト・型検査の再実行(`npm run typecheck`=`tsc -b` / `npm test`)の最終ゲートは後続フェーズ扱い。i18nの死にキー削除・`busyButton`追加に伴う`strings.test.ts`のキー一致は本リデザインの前提。

### 13.7 次セッションへの申し送り(残課題)

- **バッチ表(`BatchTable`)のデバッグ**: オーナー指示で後回し。
- **タイムライン右クリック挙動のデバッグ**: 同上。
- **同一プリセットの再適用ができない**: プリセットが永続選択式になった副作用。対処案(適用ボタンの併設等)はオーナー判断待ち。→ §14.3で解消(`PENDING_TASKS.md`§3-4)。
- **CMakeの`OBJECT_DEPENDS`1行修正**: webuiだけを変更した場合にaux2が再ビルドされない罠の恒久修正。承認待ち。現状のワークアラウンドは`build\ninja-release\CMakeFiles\NzVideomni.dir\webui_embedded.rc.res`を削除してから`build.ps1`を実行すること。→ §14.2で解消(`PENDING_TASKS.md`§3-5)。
- **アクセスログが毎秒2連発だった件**: フロントを2画面同時に接続していた疑い(要実機確認)。→ 将来の研究課題`PENDING_TASKS.md`§4-6へ降格。
- **`useChainForm.test.ts`のflakyなタイミングテスト1件**: 単独実行では通る。→ §14.2で解消(`PENDING_TASKS.md`§3-6)。
- **`deploy.ps1`が`Plugin\NzVideomni\webui\`フォルダもコピーする**: 埋め込み優先のため無害。→ §16で解消(`PENDING_TASKS.md`§3-7)。
- **`two_stage_hq`(非量子化の高品質2段生成)は引き続き保留**(既存方針。§11.3のN12・[PENDING_TASKS.md](PENDING_TASKS.md) §4-2参照)。

## 14. 実機フィードバック起点のデバッグ＋UI改修(2026-07-17、4波構成)

§13の大規模UIリデザインを実機(AviUtl2 v2.0.54ポータブル構成)へデプロイしてオーナーが実際に触った結果のフィードバックと、§13.7で「後回し」「判断待ち」としていた既存課題を合わせて解消する追加セッションを、同日中に4波構成で実施した。バックエンド(`Nz-Videomni`)は本セッションでも一切変更していない(凍結方針を維持)。

### 14.1 経緯とスコープ

オーナーの実機フィードバック6点(Crop出力の32刻み仕様不一致、プリセット再適用不可、DURATIONの入力方式、ファイル選択ダイアログのタイムアウト破棄、Chainのソース入力欄の分離による使いにくさ、ドラッグ＆ドロップ未対応)に、§13.7から持ち越していた既存課題のうち3点(CMakeの`OBJECT_DEPENDS`未修正、`useChainForm.test.ts`のflakyテスト、同一プリセットの再適用不可——最後の1点はオーナーフィードバックと重複)を合わせてスコープとした。進め方は各波とも「Sonnet実装→品質ゲート→(第3・4波はOpus敵対的レビュー→修正)→実機デプロイ」を反復する監督役方式を踏襲した。品質ゲートは`npm run typecheck`(`tsc -b`)／`npm test`(vitest)／native doctest／`npm run build:single`の4点。**本節を記録した時点では未コミットだったが、オーナー指示(外出中のためオンラインバックアップを優先)により、目視ゲート実施前の2026-07-17中に`86e8689`(feat)／`c1a5a17`(docs)としてコミット・`origin/main`へpush済み**。

### 14.2 第1波(独立小修正)

他の波と依存関係の無い小粒の修正をまとめて先行実施した。

- **`CMakeLists.txt`への`OBJECT_DEPENDS`1行追加**: §12.6・§13.7で「承認待ち」としていた、webuiだけを変更してもaux2が再ビルドされないという罠を恒久修正した。従来のワークアラウンド(`build\ninja-release\CMakeFiles\NzVideomni.dir\webui_embedded.rc.res`の手動削除)は今後不要になったことを実地検証で確認した。
- **`useChainForm.test.ts`のflakyテスト安定化**: §13.7で残課題としていたタイミング依存テストを、`mockBridge`に手動解放ゲート(`holdUploads`/`releaseUploads`)を追加することで実時間依存を排し決定論的にした。5連続実行で安定を確認した。
- **ファイル選択ダイアログのタイムアウト撤廃**: `ui.pickFile`/`ui.pickFolder`をJS側`RequestDispatcher`の10秒ローカルタイムアウト対象外(`NO_LOCAL_TIMEOUT_METHODS`)にした。従来はモーダルダイアログで10秒以上迷うと、WebUI側が先に諦めて選択結果をサイレントに破棄するバグがあった。[BRIDGE_CONTRACT.md](BRIDGE_CONTRACT.md) §4.9/§6は実装工程で既に更新済み(本ドキュメント整備セッションでは触れていない)。
- **台帳(`PENDING_TASKS.md`)の更新**: `webui/package.json`のversion`0.0.0`放置と、バックエンドのアクセスログ毎秒2連発疑いの2点を「将来の研究課題」へ降格する編集は、本セッション内の別エージェントが第1波と並行して実施済み(重複編集を避けるため本節では扱わない)。

### 14.3 第2波(Create/共有フォーム是正)

- **Crop出力の是正**: 32刻みスナップを撤廃し、APIの実仕様(「32以上・生成サイズ以下の任意整数」)に整合させた。これにより例えば1920×1080のような値がそのまま入力できるようになった(従来はN1実装時の32刻みスナップにより不整数のみ通っていた)。プリセット適用時にはconfig由来の`crop_output`(例: `FHD_1080p`→1920×1080)を反映し、`crop_output`を持たないプリセット(`smoke_test`等)を適用した場合はクロップをオフへ戻す。手動解像度変更時のクロップON初期値は現在のW/Hをそのままコピーする。IC-LoRA有効時は丸め後の生成サイズを上限にクランプする。
- **プリセット「適用」ボタンの新設**: §13.7で「オーナー判断待ち」としていた「プリセットがドロップダウンの永続選択式になった副作用で、同一プリセットを選び直しても再適用が走らない」問題を解決した。`PresetDropdown`(Create/Chain共通コンポーネント)へ明示的な適用ボタンを追加し、選択に連動した自動適用から「選択→ボタン押下で適用」方式へ変更した。これにより同一プリセットの再適用が可能になった。
- **DURATIONのスライダー＋数値入力併記**: `SizeFields`(width/height)のパターンを踏襲した新規`DurationField`をCreateの尺・Chainの各クリップ尺に適用した。手打ちは自由に通し、ステッパー操作時のみ8n+1へスナップする(§12.1のWidth/Heightスナップ方式と同じ設計)。送信時ゲートとして新規`isNumFramesOnGrid`を追加し(Chainは全クリップを検査)、半端な値のまま送信できないようにした。

### 14.4 第3波(Chainソース入力欄の一本化)

オーナーから「ソース動画とSTART FRAMEが別々の欄で分かりにくい」というフィードバックを受け、Chain画面のソース入力欄を単一の`SourceInputPanel`(`webui/src/modes/chain/SourceInputPanel.tsx`)に統合した。旧`StartFramePanel`は削除した。

- **拡張子ルーティング**: 新規`sourceRouting.ts`(`webui/src/modes/chain/sourceRouting.ts`)が、単一の入力窓口に投入されたファイルの拡張子を見て経路を振り分ける——画像はstart frame経路(scratchモード)、動画はsource video経路(V2Vモードへ自動遷移)。画像↔動画は単一スロットで相互排他(片方を設定するともう片方は自動クリア)。
- **`ui.pickFile`への新kind`"imageOrVideo"`追加**: ネイティブ側のファイルフィルタも対応済み。[BRIDGE_CONTRACT.md](BRIDGE_CONTRACT.md) §4.9に既に追補されている。
- **寸法不変の設計**: 動画入力時は画像入力時と同寸のサムネ枠に暫定プレースホルダー(正式な画像はオーナーが後日支給予定)を表示し、STRENGTHスライダーの枠にCONTEXT FRAMESスライダーを差し替え表示する。入力種別・状態遷移が変わってもパネルの寸法そのものは変化しない設計とし、DOM構造の対称化＋`min-height`で担保した(構造対称性を検証するテストを追加)。
- **アップロード世代カウンタ**: `useSourceUpload`にアップロード世代カウンタを導入し、クリア操作後に古い(世代の古い)アップロード完了イベントが復活して表示を上書きする競合を根絶した。`useSourceUpload.uploadPath`/`useKeyframes.addFromPath`の注入口を抽出し、ドラッグ&ドロップ(第4波)からも再利用できるようにした。
- **`.clip-list`の`max-height: 420px`撤廃**: 「Add clipするとスクロールバーが出る」問題を解消した。クリップ一覧はパネル全体が伸びる形になり、あふれた分はページ側のスクロールに委ねる。
- **Opus敵対的レビュー**: MAJOR 1件(動画入力時のヒント文がスライダー枠を圧迫し寸法対称性が崩れる)とMINOR 4件を検出し、全件修正済み。

### 14.5 第4波(ドラッグ＆ドロップ、ブリッジ契約v7)

対象は3箇所——Create画面の参照動画(IC-LoRA)欄・CreateのA2V音声欄・Chainの統合ソース入力欄(キーフレーム欄はドラッグ&ドロップの対象外。別セッションで予定されている大規模改修で扱う)。

- **仕組み**: WebView2の`postMessageWithAdditionalObjects`でドロップされたファイルを送出し、ネイティブの`WebMessageReceived`ハンドラが`ICoreWebView2File::get_Path()`で実パスを解決、`params.__droppedPaths`へ注入する。新規RPC`ui.resolveDroppedFiles`(contract v7)がこれを返し、既存のアップロード経路(第3波で抽出した注入口)へ合流させる。複数ファイルドロップ時は先頭1件のみを採用する。ドキュメント面ではすでに[BRIDGE_CONTRACT.md](BRIDGE_CONTRACT.md) §4.21・§0が実装工程でv7として更新済み。
- 誤ドロップによるWebUI破棄を防ぐため、document全体でdragover/dropの既定動作を抑止した。
- **セキュリティ**: `__droppedPaths`は予約キーとして扱われ、ネイティブ側が常にページ側の値を上書きする(ページ側が偽装した値は生き残れないことをOpusレビューで証明済み)。
- 音声のドラッグ&ドロップは、既存のwav長自動調整(§9.5)との整合を保つため`lastAudioFilePathRef`を先にセットしてから既存経路に合流させている。
- **BLOCKER発見と修正(教訓)**: Opus敵対的レビューで、注入発火条件の判定順序に設計欠陥が見つかった。当初の実装は「`params`に`"__droppedPaths"`という部分文字列が含まれるか」を**AdditionalObjectsの有無より先に**チェックしていたため、正規のドロップ経路(`params: {}`の空オブジェクトで送られてくる)が常にfast pathへ素通しし、注入処理が一度も実行されず**実機でD&Dが全く動かない**状態だった。`mockBridge`がこの注入機構自体を経由せず`.name`から直接結果を合成していたため、webuiのテストもnative側のdoctestもこの穴を検出できなかった——**モックが本物の契約を迂回すると、多層のテストが同じ穴を共有してしまう**という教訓として明記しておく。修正は、AdditionalObjectsの有無を最優先で確認する3条件ゲートへ判定順序を再構成し、注入ロジック自体を純関数`InjectDroppedPathsIntoRequest`(`native/src/bridge_core.h/.cpp`)へ抽出してdoctestで再発防止し、`mockBridge`もネイティブの注入契約に忠実な実装へ修正した。修正後の確認レビューでPASSし、偽装対策(セキュリティ規律)が緩んでいないことも再証明された。
- **残る実機必須確認**: `ExtractDroppedFilePaths`(WebView2の`WebMessageReceived`ハンドラでの実際のファイルパス解決)は、`native/src/webview_host.cpp`に`REALDEVICE-VERIFY`コメントを残置しており、実機でのみ完全検証できる。

### 14.6 品質ゲートとデプロイ

| スイート | 件数 | 備考 |
|---|---|---|
| webui vitest | **684件パス／10件skip** | 第3波の構造対称性テスト、第4波の`sourceRouting`/D&D関連テストなどを新規追加 |
| native doctest(`NzVideomni_tests.exe`) | **215ケース** | `InjectDroppedPathsIntoRequest`・`ParseResolveDroppedFiles`等の新規doctestを含む |
| `npm run typecheck`(`tsc -b`) | エラー0 | |
| `npm run build:single` | 成功 | |

各波の完了ごとに実機(`D:\For_Videos\AviUtl2\aviutl2_v2.0.54`)へ`scripts/deploy.ps1`でデプロイして挙動を確認した。第4波分は本ドキュメント整備と並行してデプロイを実施している(デプロイ済み・最新ハッシュはコミット時に確定)。

### 14.7 オーナー目視ゲート項目と次セッションへの申し送り

自動テスト・実機ゲート(サブエージェントによる疎通確認)では機能面を確認済みだが、レイアウト・操作感・文言の自然さの最終判断はオーナー本人の目視確認が必要。詳細な項目一覧は[PENDING_TASKS.md](PENDING_TASKS.md) §2に転記した。要点は以下。

1. **Crop**: 1080pプリセット適用→クロップON→1920×1080が入っている／手動で1080と打っても丸まらない／`crop_output`無しプリセット(`smoke_test`等)適用でクロップがオフに戻る。
2. **プリセット**: 同一プリセットを「適用」ボタンで再適用できる。
3. **DURATION**: 数値手打ちが自由に通る／ステッパー・スライダー操作では8n+1にスナップ／半端な値のままだとGenerateが無効。
4. **ファイル選択**: ダイアログを1分以上放置してから選択→正しく反映される。
5. **Chain統合ソース入力**: 1つのボタンから画像/動画の両方を選べる／画像=STRENGTH・動画=CONTEXT FRAMESが同じ枠に出てパネル寸法が変わらない／動画時に仮プレースホルダーが出る／クリアで両モードとも初期化／Add clipしてもスクロールバーが出ずパネルが伸びる。
6. **D&D**: 参照動画欄に動画・A2V欄に音声・Chainソース欄に画像/動画をドロップできる／音声ドロップでwav長自動調整が発火する／対象外ファイルはエラーメッセージ／ドロップゾーン外に落としてもWebUIが壊れない(**REALDEVICE-VERIFY**: D&Dは実機でのみ完全検証可能)。

次セッションへの申し送りは以下の通り(詳細は[PENDING_TASKS.md](PENDING_TASKS.md)参照)。

- 上記オーナー目視ゲート6点はいずれも未実施のまま残る。
- §13.7から持ち越していた**バッチ表(`BatchTable`)のデバッグ**と**タイムライン右クリック挙動のデバッグ**(オーナー指示で後回し)は、本セッションのスコープ外のため引き続き未着手のまま残る。
- 実バックエンド(実GPU)での通し確認(`Docs/REAL_BACKEND_CHECKLIST.md`)は従来どおりオーナー後日実施の最終ゲートとして残置している。
- キーフレーム欄の大規模改修は、オーナー予告のとおり別セッションで扱う(本セッションのD&D対応はキーフレームを意図的に対象外とした)。

## 15. IC-LoRA UI再設計(2026-07-17、第5波)

§14の実機フィードバック起点のデバッグ＋UI改修に続けて、同日中に追加セッションとして、IC-LoRA(In-Context LoRA、参照動画で構図・動きを制御する制御用LoRA)の入口をプロンプトタグ機構から分離するUI再設計を実施した。バックエンド(`Nz-Videomni`)は本セッションでも一切変更していない(凍結方針を維持)。

### 15.1 経緯(機能重複の指摘と実体)

オーナーの実機操作で2点の指摘があった。(1)メインプロンプトへのコマンド入力と＋/−付きタグ(チップ)の機能が重複しているように見える。(2)IC-LoRAのドロップダウンが、選択した直後にplaceholder表示へ戻ってしまい、有効化されていることに気づきにくい。

調査の結果、LoRAはプロンプト内の`<lora:名前:強さ>`タグが唯一の真実(single source of truth)であり、チップ・ドロップダウン・Libraryカードはいずれも同じ機構への入口が分散していただけだと判明した。旧IC-LoRAドロップダウンは`value=""`固定のワンショット実装で、選択のたびにプロンプトへタグを注入するだけの一方向の書き込み口であり、選択状態を保持していなかった。加えて、異なる制御LoRAを選び直すと古いタグが残ったままタグが積み上がる問題もあった。これはバックエンドが複数の制御LoRA同時使用を`lora_preprocess_conflict`で422拒否する仕様のため、そのままエラーの温床になっていた。

### 15.2 重ねがけ調査の結論

オーナーから「SDXLのControlNetのように(制御LoRAを)重ねがけできるのでは」という問いがあり、調査のうえ以下のとおり回答・記録した。

- 異なる前処理種別の制御LoRA併用(例: canny-control＋pose-control)は、バックエンドが`api/generate.py`・`ltx_runner.py`の二重チェックで明示的に400/422拒否する。
- そもそもcanny-controlとpose-controlは、同一のUnion-Control LoRAファイル(Canny/Depth/Pose統合型)を前処理違いで2登録しているだけの別名である。
- upstream(LTX-2公式wheel)のパイプライン自体は複数参照動画のリストを受け付ける形状を持つが、本バックエンドは実装簡素化のため単一`reference_video_id`・配列先頭LoRA依存の設計に絞ってある。
- したがって「同じ参照動画からポーズと輪郭の両方で制御したい」という需要への正道は、LoRAの重ねがけではなく、1本の制御動画にcanny＋pose等を合成する前処理モードの追加である。これはバックエンド凍結解除後の将来課題として`PENDING_TASKS.md`§4に起票した(§15.3参照)。

### 15.3 実装内容(Gradio方式への回帰。webui/のみ)

native・バックエンドは無変更。

- **入口の分離**: Style LoRA(画風LoRA)はプロンプトタグ＋チップのまま現状維持。IC-LoRAはCreate画面の参照動画アコーディオン内の状態持ちUIへ分離した——「なし」を含む選択保持ドロップダウン＋LoRA重みスライダー(0.05〜2.0、既定1.0)。単一選択。
- **状態の所有者**: 状態(`controlLora`)・LoRA一覧取得(`useLoras`)・自動移行effectは**AppShellが所有**する(プロンプトと同じ寿命とすることで、タブを往復しても選択が消えない)。新規フック`useControlLoraNames`を追加し、参照の安定化をメモ化で担保している。
- **データ源**: 選択肢と判定は`GET /loras`の`kind="control"`を正とし、未ロード・失敗時は`config.model.ic_loras`キーへフォールバックする。フォールバック時は`lora_dir`スキャン由来の制御LoRAを網羅できないため、その場合のみ従来どおりサーバー422に落ちる残余がある(許容)。
- **手打ちタグの自動移行**: プロンプトに手打ちされた制御LoRAタグは自動でパネル選択へ移行する(強度を引き継ぎ・複数ある場合は最後のタグが勝ち・タグはプロンプトから除去・トースト通知)。Create画面のみの挙動。関数型セッターで入力の巻き戻しレースを回避し、IME合成中は移行を保留する。
- **送信時マージ規約**: GradioのGenerateタブと同一の規約に統一した——制御LoRAをloras配列の先頭に置き、プロンプト由来のstyle系LoRAをその後続に置き、名前が重複する場合は後勝ちとする。これはエンジンが配列先頭から`reference_downscale_factor`を読む制約に整合させるため。
- **参照強度の既定値変更**: 参照強度2種(conditioning attention strength / reference video strength)の有効化時初期値を0.5→1.0へ変更した(Gradio・公式ガイダンス「1.0推奨、1.0未満は参照がにじむ可能性」に準拠)。
- **Chain画面**: プロンプトに制御LoRAタグが含まれていると警告バナーを表示しGenerateを無効化する(従来は黙って送信されサーバーが422で拒否していた)。Chainでは自動移行は行わない(チップの×で手動除去する)。
- **Library画面**: Controlタブを削除しStyle専用画面へ整理した(旧Controlタブはクリック不能なプレースホルダで、「M6で追加されます」という陳腐化した文言の遺物だった)。`LoraCard`のkindバッジも削除した。
- **プロセス**: 計画段階でOpus敵対的レビューを実施し、BLOCKER1(チップの配線先の誤認)・MAJOR3(自動移行のレース・タブ往復での選択消失・Setの不安定ループ等)を検出→計画を修正。実装後にもOpus敵対的レビューを実施し、MAJOR1(メモ化依存の非対称ミス)ほかMINOR数件を検出→修正。教訓として「フック戻り値オブジェクトをuseMemoの依存配列に入れない(内側の安定値を使う)」ことを記録しておく。

### 15.4 レビューと品質ゲート

| スイート | 件数 | 備考 |
|---|---|---|
| `npm run typecheck`(`tsc -b`) | エラー0 | |
| webui vitest | **729件パス／10件skip** | §14終了時点の684件から+45 |
| `npm run build:single` | 成功 | |

実機(AviUtl2 v2.0.54ポータブル構成)へデプロイ済み。

**既知の理論上の制約**(コードコメントにも記載済み):

- effectのスナップショット〜commit窓のごく短い間に別の制御タグが打ち込まれた場合の消失(実質到達不能な理論上の穴)。
- マウント時に制御タグが既に存在する場合のStrictModeによるトースト二重表示(現状到達経路なし)。

### 15.5 オーナー目視ゲート項目

自動テスト・敵対的レビューでは機能疎通・設計不備を確認済みだが、以下はオーナー本人による目視確認が未実施のまま残っている(項目は[PENDING_TASKS.md](PENDING_TASKS.md) §2にも転記)。

1. ドロップダウンの選択が保持され、「なし」で解除できる。
2. 選択してもプロンプトにタグが出ず、パネル内に重みスライダーが表示される。
3. 制御タグを手打ちすると、自動でパネル選択に移りトーストが出る。
4. 参照動画なしで制御LoRAを選択すると、警告とともにGenerateが無効化される。
5. Chainで制御タグを検出すると、警告とともにGenerateが無効化される。
6. LibraryにControlタブが無い。
7. 参照強度の有効化時初期値が1.0になっている。

## 16. `deploy.ps1`のwebuiコピー廃止=埋め込み専用運用への一本化(2026-07-17、第6波)

`Docs/PENDING_TASKS.md`§1-3(当時の番号。後日の台帳スリム化で削除・番号整理され、現在の§1-3は別項目)が「無害な既知の余剰動作」として残していた「`deploy.ps1`が`Plugin\NzVideomni\webui\`フォルダもコピーする」を解消した。このコピーは非埋め込みビルド(`build.ps1`をオプション無しで実行した場合の既定)のUI供給源として現役だったが、非埋め込みモードの存在理由(webui変更をネイティブ再ビルドなしで反映)は§13.7・§3-5のCMake`OBJECT_DEPENDS`修正で消滅しており、直近の全開発は埋め込みビルドのみ使用していた。オーナー判断: α版公開へのカウントダウン中であり、非埋め込みデプロイは開発期のジャンクとして**埋め込み専用運用に一本化**する。方針は「黙って壊れる(フォールバック表示)」を「明示的に止まる(エラー)」に変えること。ネイティブのフォールバック経路(`webview_host.cpp`のフォルダマッピング)自体は安全網として変更していない。バックエンド(`Nz-Videomni`)は本セッションでも一切変更していない(凍結方針を維持)。承認済み実装計画: `twinkling-whistling-salamander.md`。

### 16.1 実装内容

- **`scripts/build.ps1`**: 新スイッチ`-NoEmbedWebui`を追加し、埋め込みが既定になるよう分岐を反転(`$embed = -not $NoEmbedWebui`)。既存の`-EmbedWebui`は後方互換のため残し、既定と同じ意味の冗長指定として受理する。`-EmbedWebui -NoEmbedWebui`の同時指定は`throw`でエラーにする。docstringを新しい既定に合わせて書き換えた。
- **`scripts/deploy.ps1`**: `webui\dist`をコピーするブロックを削除し、代わりにデプロイ対象の`.aux2`(ビルド成果物)へ埋め込み有無の判定ガードを追加した。判定は当初計画どおり`scripts/package.ps1`(L53-72)と同じ「`.aux2`のバイト列からUTF-16LEの`NZVIDEOMNI_WEBUI_INDEX`を総当たり検索する」ロジックを流用する予定だったが、**実機検証(手順3の負のテスト)でこのロジック単体では埋め込み有無を判定できないことが判明した**(§16.2参照)。是正として、実際に埋め込まれたHTML本文の先頭バイト列(`<!doctype html`、Viteのsingle-file出力が必ずこの書式で始まる)を追加の判定条件とし、両方が一致した場合のみ「埋め込みあり」と判定するよう変更した。埋め込みなしと判定した場合は`Write-Error`+`exit 1`で停止し、実機側を一切変更しない(コピー・削除いずれの操作もこの判定より後に置いている)。埋め込みありの場合は通常デプロイに加え、実機に残っている旧`Plugin\NzVideomni\webui\`フォルダを削除する。ヘッダコメントを埋め込み専用運用の説明へ書き換えた。
- **ドキュメント**: README.md(Build節・Deploy節・Release packaging節・「webui-onlyリビルドの罠」節を現状=OBJECT_DEPENDS修正済みへ更新)、`Docs/PENDING_TASKS.md`(当時の§1-3の当該項目をクローズし§3-7へ移動。§1-3という番号は後日の台帳スリム化で削除・番号整理済み)、`Docs/DEVELOPMENT_PLAN.md`(「次の着手」に完了追記)を更新した。

### 16.2 実機検証で発見した計画からの逸脱(判定ロジックの是正)

承認済み計画は「package.ps1 L53-72と同じバイト列検索」をそのまま`deploy.ps1`の**ハードゲート**(`Write-Error`+`exit 1`)へ転用する設計だったが、検証の手順3(負のテスト: `-NoEmbedWebui`ビルド→deploy.ps1がエラーで停止するはず)を実施したところ、**このロジック単体では常に「埋め込みあり」と誤判定し、非埋め込みビルドのデプロイを素通りさせてしまうことが実機確認で判明した**。

原因: `native/src/webview_host.cpp`の`FindResourceW(module, L"NZVIDEOMNI_WEBUI_INDEX", RT_RCDATA)`呼び出しに使われている文字列リテラル`L"NZVIDEOMNI_WEBUI_INDEX"`は、`NZVIDEOMNI_EMBED_WEBUI`のON/OFFに関わらずコードとして無条件にコンパイルされるため、そのUTF-16LEバイト列は埋め込みの有無を問わずすべてのビルド成果物に存在する。つまり「リソース名文字列の検索」は埋め込みの実体(RCDATAリソースそのもの)ではなく、常に真になるコード側の定数を見ているだけであり、判定として機能しない。これは`package.ps1`の既存の同一ロジック(そちらは`Write-Warning`のみのソフトチェック)にも同様に存在する潜在的な欠陥だが、警告のみで実害が無かったため今まで表面化していなかった。今回`deploy.ps1`でハードゲートに転用したことで、この欠陥が「負のテストが素通りする」という形で顕在化した。

是正: `NZVIDEOMNI_WEBUI_INDEX`検索に加えて、埋め込まれたHTML本文そのものの先頭バイト列(ASCII`<!doctype html`。`native/res/webui.rc.in`が`webui\dist-single\index.html`の生バイトをそのままRCDATAへ埋め込むため、埋め込み時のみ実際に出現する)を追加の判定条件とし、両方成立した場合のみ「埋め込みあり」とする(`Test-ByteNeedle`ヘルパーで2つの探索を実施)。この是正後、負のテストは実機で正しくエラー停止することを確認した(§16.3手順3参照)。

**（2026-07-17追記・オーナー承認済み移植）**: `package.ps1`側に残っていた同一の潜在的欠陥は、オーナー承認のうえ本セッション内で移植・解消した。`package.ps1`(L57-89)の判定ロジックを`deploy.ps1`と同一の二重ニードル判定(`Test-ByteNeedle`ヘルパー＋UTF-16LEリソース名＋ASCII`<!doctype html`)へ置き換え、相互参照コメント(「deploy.ps1と同一。変更時は両方更新」)を両ファイルに対で記載した。あわせて、埋め込み専用運用に一本化した以上「非埋め込みaux2をau2pkg（正式配布物）に梱包する」のは確実に壊れたリリースであるため、従来の`Write-Warning`(ソフトチェック)を`package.ps1`の既存流儀である`throw`に格上げした(エラーメッセージは`deploy.ps1`のものに準拠し「埋め込み無し・build.ps1（既定で埋め込み）でビルドし直せ」の趣旨)。詳細は§16.5参照。

この逸脱は実装計画の記述(「package.ps1 L53-72と同じバイト列検索で埋め込み有無を判定」)から外れているが、計画の意図(埋め込みビルドと非埋め込みビルドを確実に判別してゲートする)を実現するために必須の是正であり、検証手順で実機確認したうえでの判断である。

### 16.3 実機検証(手順1〜6、順序どおり実施)

対象実機: `D:\For_Videos\AviUtl2\aviutl2_v2.0.54\`(AviUtl2は検証開始時点・終了時点とも未起動を確認。デプロイのブロックは発生しなかった)。

1. **`scripts\build.ps1`(オプション無し)**: `npm run build:single`が自動実行され、CMakeに`NZVIDEOMNI_EMBED_WEBUI=ON`が渡り、ビルド成功。成果物`build\ninja-release\NzVideomni.aux2`に`NZVIDEOMNI_WEBUI_INDEX`マーカーが存在することを確認(**合格**)。SHA256: `021573F8FD3378A8F51EA7C5C7DBFA9F035A4DE643427CF4913462FF0C4368E4`。
2. **`scripts\deploy.ps1`**: 正常デプロイ。実機の`Plugin\NzVideomni\webui\`フォルダ(旧非埋め込みデプロイの残骸として実在していた)が削除されることを確認(**合格**)。デプロイ後の実機aux2ハッシュは手順1のビルド成果物と一致。
3. **負のテスト**: `scripts\build.ps1 -NoEmbedWebui`でビルド成功(npm不要・C++のみの完全リビルド)。デプロイ前に実機aux2のSHA256(`91474B2B557403D12222AA7CF63F3795C1804C63706E9EE82F9B188ECB1DEB08`、手順2完了時点の値)を控えたうえで`deploy.ps1`を実行したところ、**当初の判定ロジック(NZVIDEOMNI_WEBUI_INDEX検索のみ)では誤って正常デプロイが成立してしまい、実機aux2が非埋め込みビルドで上書きされる事故が発生した**(§16.2の発見の経緯そのもの)。判定ロジックを是正(`<!doctype html`本文検索を追加)した後に同じ非埋め込みビルドで再実行したところ、`Write-Error`+`exit 1`で停止し、実機aux2のSHA256が直前の値(事故で上書きされた`91BA854D2C399EFF01684CC175C1BC8B6E13434B65DFEC2F92BA025CC1E181C5`)から不変であることを確認した(**是正後は合格**)。事故で一時的に非埋め込みビルドがデプロイされていた期間があったが、手順5で埋め込みビルドに復元済み。
4. **`-EmbedWebui`(旧表記)**: `-EmbedWebui -NoEmbedWebui`の同時指定が`throw`で即エラーになることを確認(**合格**)。`-EmbedWebui -SkipWebuiBuild`単独では従来どおりビルドが通ることを確認(**合格**)。
5. **実機の復元**: 手順1・2をもう一度実行し(オプション無しの`build.ps1`→`deploy.ps1`)、実機を埋め込みビルド配置状態に戻した。最終デプロイ済みaux2のSHA256: `6B3E774E3EC68C82FC96202CA1B980B9222239ED9C46D3E507897F59A3A1FB30`。`Plugin\NzVideomni\webui\`フォルダは存在しないことを確認済み。
6. **リグレッション**: `NzVideomni_tests.exe --test-suite-exclude=integration`を実行し、215 test cases / 1066 assertions すべて緑(6 skipped)であることを確認(基準215件と一致、**合格**)。webuiのtypecheck/vitestはソース不変のため実施していない(計画どおり)。

### 16.4 変更ファイル一覧

`scripts/build.ps1`・`scripts/deploy.ps1`・`README.md`・`Docs/PENDING_TASKS.md`・`Docs/DEVLOG.md`(本節)・`Docs/DEVELOPMENT_PLAN.md`の6点。webui/・native/・CMakeLists.txt・バックエンドは無変更(git操作もコミット待ちのため未実施)。

### 16.5 `package.ps1`への移植(オーナー承認後の追加作業)

オーナー承認を受け、§16.2で`deploy.ps1`にのみ適用していた二重ニードル判定(`Test-ByteNeedle`ヘルパー＋UTF-16LEリソース名`NZVIDEOMNI_WEBUI_INDEX`＋ASCII`<!doctype html`本文検索)を`scripts/package.ps1`(L57-89)へ移植した。判定失敗時の扱いは、埋め込み専用運用に一本化した以上「非埋め込みaux2をau2pkg(正式配布物)に梱包する」のは確実に壊れたリリースであるため、従来の`Write-Warning`(ソフトチェック)を`package.ps1`の既存流儀である`throw`へ格上げした(エラーメッセージは`deploy.ps1`のものに準拠)。両ファイルの判定ブロックに「もう一方と同一。変更時は両方更新」の相互参照コメントを対で記載した。ヘッダの`.DESCRIPTION`も「警告する」から「エラーで停止する」へ書き換えた。あわせてREADME.mdのRelease packaging節の記述(旧: 「package.ps1は警告する」)を整合させた。

**変更ファイル(本追加作業分)**: `scripts/package.ps1`・`README.md`・`Docs/DEVLOG.md`(本節)の3点。`scripts/deploy.ps1`は今回のオーナー指示の変更許可対象に含まれていないため無変更。ただし`deploy.ps1`のコメント(§16.2で追加)は`package.ps1`(L53-72)への行番号参照と「ソフトチェック」という記述を含んでおり、`package.ps1`側の今回の変更(L57-89へシフト・ハードエラー化)後は**やや古い記述として残っている**。実害はない(コメントのみ)が、次に`deploy.ps1`を触る機会があれば、当該コメント(「NOTE: scripts\package.ps1 (L53-72) runs the same ... but ONLY as a soft Write-Warning」の部分)も現状に合わせて更新することを推奨する。

**検証(オーナー指示の(a)〜(c)を実施)**:

(a) 現在の埋め込み成果物(`build\ninja-release\NzVideomni.aux2`、本節時点のハッシュ`6B3E774E3EC68C82FC96202CA1B980B9222239ED9C46D3E507897F59A3A1FB30`)に対し`package.ps1 -Config Release`を実行したところ、「Embedded Web UI resource detected in NzVideomni.aux2.」と表示され正常完走、`.au2pkg.zip`(Language 2件・package.ini・package.txt・`Plugin\NzVideomni\NzVideomni.aux2`)が生成されることを確認した(**合格**)。

**出力パスに関する注意(オーナーへの報告事項)**: `package.ps1`の既定出力は`<repo>\dist\NzVideomni-<Version>.au2pkg.zip`で、`-Version`の既定値は`kPluginVersion`と同じ`1.0.0-rc1`である。リポジトリの`dist\`には既存の`NzVideomni-1.0.0-rc1.au2pkg.zip`(1.0.0-rc1正式リリース物、2026-07-17 1:32時点のタイムスタンプ)が既に存在しており、`package.ps1`は同名ファイルがあれば`Remove-Item -Force`してから`Compress-Archive`で上書きする仕様である(既存の仕様で、本タスクでは変更していない)。つまり**バージョンを明示的に変えずに`package.ps1`を実行すると、既存のrc1正式リリースzipは無条件に上書きされる**。このため本検証(a)(b)は`-OutDir`をスクラッチディレクトリへ向けて実施し、実際の`dist\NzVideomni-1.0.0-rc1.au2pkg.zip`には一切触れていない(タイムスタンプ・内容とも検証前後で不変)。新バージョンをリリースパッケージとして正式に作る際は`-Version`を明示的に指定することを推奨する(この注意点自体は今回の変更範囲外の既存仕様のため、`package.ps1`のコード自体は変更していない)。

(b) `scripts\build.ps1 -NoEmbedWebui`で非埋め込みビルドを作成後、同じ`package.ps1 -Config Release`(スクラッチ出力)を実行したところ、`throw`により`S:\...\NzVideomni.aux2 does not contain the embedded Web UI resource. Packaging a non-embedded build is not supported (embedded-only operation): ...`で停止(終了コード1)し、zipは生成されなかった(スクラッチディレクトリの内容は(a)で作った1件のみで増えていないことを確認)。実機の`dist\`側は最初から触っていないため無変更(**合格**)。

(c) `scripts\build.ps1`(オプション無し)を再実行し、`build\ninja-release\NzVideomni.aux2`を埋め込み状態へ戻した(再ビルド後のハッシュ`4FD244F6537E9E5BBAE24F7616FB7EFD52E3272E9BE73D7F9858AB478E6630A7`。同一ソースの再リンクのため§16.3の値とはハッシュが異なるが、これはPEリンカのタイムスタンプ埋め込み等によるビルドの非決定性であり、埋め込み有無とは無関係)。オーナー指示どおりデプロイは実施していない。実機(`D:\For_Videos\AviUtl2\aviutl2_v2.0.54\...`)にデプロイ済みのaux2ハッシュ(`6B3E774E...`)は本作業中一貫して不変であることを確認した(**合格**)。

**追補**: `-RunTests`が`ctest --output-on-failure`をフィルタ無しで実行するため、mock未起動環境で`"integration"`スイート込みの完全実行となりビルド全体が`throw`で失敗する問題を根治した。`ctest`呼び出しをやめ、テストexe(`$buildDir\NzVideomni_tests.exe`)を`--test-suite-exclude=integration`付きで直接起動する方式に変更し、mock未起動でも`-RunTests`が安全に完走するようにした。`"integration"`込みの完全実行が必要な場合はmockバックエンドを起動したうえで同exeを引数無しで手動実行する(docstringに記載)。オーナー指示の検証(mock未起動状態で`build.ps1 -Config Release -RunTests -SkipWebuiBuild`を実行)を実施し、ビルド成功・テスト215件緑(integration除外)・成果物が埋め込み状態(既定動作どおり)であることを確認した(**合格**)。実機デプロイは指示どおり実施していない(成果物内容は不変のため不要)。


## 17. キーフレームタイムラインUIの設計ディスカッション(2026-07-17)

同日の作業の最後に、Create画面のキーフレーム入力欄(最大5枚の画像を任意のフレーム位置に条件付けするI2V欄)の作り直しについてオーナーとブレインストーミングを行った。実装とコード変更は行っていない(本節は設計話し合いの記録のみ)。

課題は「キーフレーム間の間隔が動きの速さを決めるのに、現状の数値入力では間隔を直感的に把握できない」点。オーナーが作成した叩き台モック(`Mock/モック_キーフレーム/スライド1.JPG`・`スライド2.JPG`)を出発点に、タイムライン上のピンとその距離で間隔を可視化する方向で合意した。合意仕様6点・設計時の宿題3点・先行事例調査(Pika/Dreamina/W3C複数つまみスライダー等)の詳細は、本ディスカッションの正本として[`KEYFRAME_TIMELINE_DESIGN_NOTES.md`](KEYFRAME_TIMELINE_DESIGN_NOTES.md)にまとめた。着手は別セッションで、Create画面へ最初から本実装し、実機でオーナーが操作感を確認して仕上げる方針とする(当初は捨てる前提のHTMLプロトタイプを先に作る二段構えとしていたが、確認の流れが速く回っており試作を挟むのが冗長なため、オーナー判断で廃止)。台帳の位置づけは[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-3。

## 18. キーフレームタイムラインUIの本実装(2026-07-18)

§17の設計ディスカッションを受け、Create画面のキーフレーム欄を「開始フレーム専用枠＋タイムラインバー＋サムネイルピン＋下部カード」へ全面改修する本実装セッションを実施した。正本[`KEYFRAME_TIMELINE_DESIGN_NOTES.md`](KEYFRAME_TIMELINE_DESIGN_NOTES.md)は、本セッションの実装検討の過程でオーナーと2026-07-18に追加合意した4点(玉突き退避の廃止→操作中ピンのみが動く着地解決ルールへの切替、ピン追い越し禁止の撤廃、「キーフレームを追加」ボタン=空カード追加としての明確化、位置0専用枠のタイムラインバーからの空間分離)を反映して改訂済み(詳細は同文書§10参照)。バックエンド(`Nz-Videomni`)は本セッションでも一切変更していない(凍結方針を維持)。

### 18.1 進め方

プランモードで実装計画を立てたうえで、着手前に敵対的レビューを実施した。計画段階のレビューでCritical3件を検出し、実装前に潰した。

1. ファイル追加(空カード追加)経路が位置0と衝突する未定義動作。
2. はみ出し判定の閾値が二重定義になっていた(新設の`maxPlaceable`=グリッドの`floor`から8を引いた値、旧`isFrameIdxOutOfRange`の`numFrames-1`という別の基準が併存)。
3. 尺ガードを迂回できる経路(DURATIONのrangeキーボード操作、blur確定前のクリック)が、Generateボタンの最終防壁と整合していなかった。

これらを計画段階で解決したうえで、3波構成で実装した。

- **第1波**: `keyframeGrid`(純粋ロジック、34テスト)、i18n文字列、PointerEventテスト基盤(`setupTests.ts`)、`DurationField`の`onCommit`、`useDurationShrinkGuard`。
- **第2波**: `KeyframeTimeline`(ARIA複数つまみ・Pointer Eventsによるドラッグ)、`KeyframeCard`、`useKeyframes`の後方互換拡張(`initialFrameIdx`を経由する3経路・`addEmpty`・`replaceFile`・`pickReplaceFile`)、`KeyframeShrinkModal`。
- **第3波**: `KeyframesPanel`の全面書き換え、`GenerationForm`/`CreateScreen`/`BatchSection`の配線、Generateボタンでの送信時最終防壁。

### 18.2 実装後の敵対的コードレビュー

3波の実装完了後、Opusによる敵対的コードレビューを実施した。結果はCritical 0・Major 2・Minor 8で、全件を修正した。

- pointercancel(ポインタ操作の強制中断イベント)が未処理でドラッグ状態が固着する不具合。
- ラベル(区間のフレーム数・秒数)を常時表示していたのを、近接クラスタ(ピンが密集した箇所)のみhoverで縮退表示する方式へ変更。
- 開始フレーム専用枠へのドラッグ中にゴースト(ドラッグ中の仮表示)が出ていなかった欠落。
- キーボード操作で位置0↔1をまたぐ際のフォーカス復元漏れ。
- 値を変更していないのにblurするとDURATION縮小モーダルが誤発火する不具合。
- 実装過程で残った不要コメントの削除。

### 18.3 品質ゲート

| スイート | 件数 | 備考 |
|---|---|---|
| `npm run typecheck`(`tsc -b`) | エラー0 | |
| webui vitest | **69ファイル・835件パス／10件skip** | 改修前729件から+106。10 skipは従来どおり |
| native doctest(`NzVideomni_tests.exe`) | **215ケース** | 全緑(native側は無変更のため件数は§14.6から不変) |
| `scripts/build.ps1` | 成功 | CMakeの`OBJECT_DEPENDS`修正(§14.2/§3-5)が機能し、webui変更のみで`.res`が自動再ビルドされることを確認 |

`scripts/deploy.ps1`で実機(`D:\For_Videos\AviUtl2\aviutl2_v2.0.54\`、v2.0.54ポータブル構成)へ配置し、以下を確認した。

- 実機側`aux2`に埋め込まれたWeb UIのSHA256一致、および§16.2で追加した二重ニードル判定(`NZVIDEOMNI_WEBUI_INDEX`＋`<!doctype html`本文検索)で埋め込みありと判定されることを確認。
- mockバックエンドAPIに対し、`generate`→`completed`→動画DLの一連の疎通を確認(バックエンドは一切無変更。`config.yaml`は`model.backend: "auto"`のままで、`--config`で一時的にmock指定して検証した)。

### 18.4 Chain/Batch非退行の確認

`useKeyframes`への拡張(`initialFrameIdx`・`addEmpty`・`replaceFile`・`pickReplaceFile`)はすべてoptional引数の追加にとどめ、`DurationField`の`onCommit`もoptionalとした。これにより`useChainForm`・`useGenerationForm`のテストは全緑のまま(既存呼び出し側の変更不要)。`BatchSection`には新たに`frameRate`の配線を追加した(タイムライン表示の秒数換算に必要なため)。

### 18.5 教訓

1. **並行サブエージェントへのtypecheck実行を禁止し、`tsbuildinfo`競合を回避した**。各波の実装はサブエージェントへ分割したが、`npm run typecheck`(`tsc -b`)は親が波ごとに一括で実行する運用にした。vitestはesbuildで型情報を剥がして実行するため、厳格な`tsc -b`で初めて検出される型エラーが本セッション中に2回あり、この運用がなければサブエージェント間で`tsbuildinfo`が競合して誤検知・見落としを招いていた可能性が高い。
2. **計画段階の敵対的レビューが実装前にCritical3件を検出した**(§18.1)。実装後のレビュー(§18.2)だけに頼らず、計画段階でも一次情報の裏取りと批判的レビューを行う価値が今回もあらためて確認された。

### 18.6 オーナー判断待ちとして記録する事項

1. **画像未設定の空カードが送信をブロックする件**: 空カード(画像未設定・そもそも送信対象に含まれない)が範囲外の位置にある場合でも、Generate最終防壁は生成を止める仕様にした(安全側の実装判断)。「送信されないはずのカードが生成を止めるのは不可解」という見方もあり得るため、オーナーの目視確認時に妥当性を判断してもらう。
2. **バッチA2Vの共有キーフレームのnumFrames基準がバッチ1行目固定である既存課題**: 共有キーフレームの範囲判定はバッチの1行目(`row[0]`)のnumFramesを基準にしており、行ごとにwav長由来のフレーム数が異なる場合、短い行では範囲外→サーバー側の丸めで衝突が起き得る。これは本セッションが持ち込んだ問題ではなく既存の設計上の課題だが、今回のキーフレーム任意位置配置(旧来は固定位置がほとんどだった)により顕在化しやすくなったため、あらためて記録しておく。
3. **バックエンド`config.yaml`の`model.backend`が`"auto"`のままである件**: 実モデル一式が揃った本機では、素朴な起動で`real`バックエンドが選ばれる状態になっている。mock固定運用の要否はオーナー判断待ち(本セッションの検証は`--config`での一時mock指定で回避した)。

### 18.7 次セッションへの申し送り

- 上記§18.6の3点はオーナー判断待ちのまま残る。
- オーナー本人による実機目視確認は未実施。個別の確認項目は[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-2に転記した。

## 19. オーナー実機確認・中間報告(2026-07-18)

オーナーがAviUtl2実機(v2.0.54ポータブル構成)でフロントエンドの目視確認を行い、中間報告として以下5点を合格と回答した。t2v、PRESETSドロップダウン(適用ボタンでの再適用を含む)、CROP OUTPUT、i2vの機能面(§14・§15由来の項目)、IC-LoRA(Create画面側の5項目: ドロップダウンの選択保持・タグ非表示＋重みスライダー・手打ちタグの自動移行・参照動画なしゲート・参照強度の既定値)。

後日オーナーへ確認したところ、この目視確認は`real`バックエンド(実GPU接続の本物の生成エンジン)で行われたことが確定した。オーナーが明示的に確認したのはt2vで、i2v・IC-LoRA(Create画面)は同一セッション内の確認のためrealバックエンドでの機能確認済みと推定される。`REAL_BACKEND_CHECKLIST.md`の実GPU通しマトリクスのうちt2v以外(Chain Clips/Chain V2V/単発A2V/バッチA2V/モデルロード/chunked_upsample等)は引き続き未実施のまま残っている(詳細は[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-1)。またIC-LoRAの合格はCreate画面のみで、Chain/Library画面は未確認であることも確定した。

一方、i2vのKEYFRAMESカードUI(§18本実装)は見た目のデザインが不合格となり、改修を別途行う方針が決まった(機能疎通そのものは合格)。Chain/LibraryのIC-LoRA項目(制御タグ検出・Controlタブ削除)は当該画面が未確認であることが確定したため、合格記録には含めていない。

台帳側は[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-2から合格5点を除去し、§3-8として合格記録を追記した。§2-1(実バックエンド通し確認)も、t2vがrealバックエンドで合格し残りマトリクスが未実施である旨を反映して更新した。KEYFRAMESのデザイン改修後はあらためて目視確認が必要。

## 20. KEYFRAMESカードの再改修・開始フレーム専用枠の廃止・バッチA2V Shared仕様変更・既存実装2件の敵対的レビュー修正(2026-07-18)

§19でKEYFRAMESカードUI(§18本実装)が不合格となったことを受け、同日中に承認済み計画に基づきサブエージェント群で以下4件を実装した。全て未コミット。バックエンド(`Nz-Videomni`)は本セッションでも一切変更していない(凍結方針を維持)。

### 20.1 KEYFRAMESカードの再改修

カード外の「Capture current frame」「Choose image file…」ボタンを廃止した。「Add keyframe (n/5)」はアコーディオン見出しと同格の全幅ボタンへ独立させた。カード内は右上に絵文字ボタンを縦に3つ積む構成(上から❌Remove・📁Choose・📸Capture、title/aria-labelは従来の文言を踏襲)に変更した。📸ボタンは「タイムライン全体で現在フレームを取り込む」のではなく**そのカード自身へ現在フレームを取り込む**挙動とし、`useKeyframes`に新関数`captureIntoCard`を追加した(押した瞬間に即uploading状態へ切り替えて連打をガード)。FRAME数値入力とSTRENGTHスライダーはサムネイル下へ移動し、従来「入力欄がボタンの下に隠れる」flexレイアウトの不具合を構造ごと解消した。画像のドラッグ&ドロップの受け皿はドロップ欄からカード全体へ拡大した(エラー文言表示込み)。位置0に対応するカードには「開始フレーム(固定)」バッジを表示する。

### 20.2 開始フレーム専用枠の廃止

LTX 2.3ではframe 0のキーフレームが非必須であることを確定させた(バックエンドの検証コードにframe0強制は無く、upstreamも任意位置のconditioningに対応していることを確認)。この確認を根拠に、§18で採用した位置0の「開始フレーム」専用枠(タイムラインバーから空間的に分離した受け皿)を廃止し、全ピンをバー上の自由な位置(0を含む)へ統一した。位置0のピンは四角形・アクセント色とし、位置1のピンと近接しやすい左端付近でも埋もれないようz-indexを上げて描画する。バー左端には0への吸着域を設け、式は`max(位置1のピクセル位置の半分, 14px)`とした(位置1はマウスでのドラッグではほぼ選べなくなるが、矢印キーまたはカードのFRAME欄で確実に指定できるため許容する、というオーナー承認済みのUXトレードオフ)。`keyframeGrid`の`nearestGridPosition`は0を特別扱いせず通常のスナップ候補の1つとして扱うよう変更した。専用枠向けに§18.2で追加していたゴースト表示機構と`focusRestoreId`によるフォーカス復元機構は、専用枠の廃止に伴い削除した。`focusRestoreId`については、削除前に「位置0↔1を矢印キーで往復してもフォーカスを失わない」ことを確認するテストを通し、安全に削除できることを確かめてから外した。エンジン内部の挙動差(位置0=latent置換で完全固定、位置1以降=誘導)は、専用枠に代えてカードのバッジで表現する。

正本[`KEYFRAME_TIMELINE_DESIGN_NOTES.md`](KEYFRAME_TIMELINE_DESIGN_NOTES.md)は本節の内容を反映して2回目の改訂を行った(旧仕様の記述には廃止の注記を付記済み)。

### 20.3 バッチA2VのShared仕様変更(オーナー確定仕様)

Batch節の専用キーフレーム欄(`KeyframesPanel`)を撤去し、Create画面のKEYFRAMES状態に一本化した(Gradio原典もBatch専用のキーフレーム欄を持たないため、この一本化がGradio原典に忠実)。Shared行の挙動は「Create画面KEYFRAMESの1枚目(frameIdxが最小のready画像)1枚だけを、スライダー位置に関係なくframe_idx=0へ強制して送る。strengthは維持する。2枚目以降は無視する」に変更した。開始ゲートは、従来の「frame_idx===0のキーフレームが必須」から「ready画像が1枚以上あること」へ緩和した。`useBatchForm`はkeyframesを引数として注入される契約に変更した(Create画面のKEYFRAMES状態をBatchセクションから参照する形)。

この仕様変更により、[`PENDING_TASKS.md`](PENDING_TASKS.md)旧§1-3で記録していた既知課題「バッチA2Vの共有キーフレームのnumFrames基準がバッチ1行目固定である」は構造的に解消した。frame 0はどの行のnumFramesであっても常に置ける範囲内(グリッドの最小値)であるため、行ごとのwav長差に起因する範囲外エラーが原理的に発生しなくなったことによる。台帳側はこの項目を§3-9(解消済み)へ移動した。

### 20.4 既存実装2件への敵対的レビューと修正

直近で実装済みだった以下2件について、敵対的レビューを実施し、検出した問題を修正した。

1. **Insert尺の明示計算**(`native/src/bridge.cpp`の`InsertMediaEditProc`、`get_media_info`の`total_time`×プロジェクトfpsで挿入オブジェクトの尺を算出する処理): レビューで、`total_time`が異常に巨大な値や`inf`だった場合に`double`→`int`変換が未定義動作(UB)になり得ることを指摘された。是正として、`timeline_math.h`/`.cpp`の`ProjectFramesForSeconds`に非有限値(NaN/inf)ガードと上限クランプ(`2000000000`、int安全上限)を追加した。native doctestに7アサーションを追加し、native側の総ケース数は215件から217件(全緑)になった。
2. **`preload="auto"`**(`webui/src/jobs/JobCard.tsx`・`JoinControls.tsx`。タブ復帰後にプレビュー動画のサムネイルが描画されない不具合の対策。Chromiumの`preload="metadata"`には初期フレームが描画されない既知のバグがあり、`preload="auto"`が正当な回避策と判定していた): レビューで、`JoinControls`の`onError`ハンドラが、先読み中(`preload="auto"`によるフェッチ)の一過性の失敗でjoin済みUIをidle状態へ巻き戻してしまうリスクを指摘された。是正として、joinされたURLに対する初回の`error`イベントでは`video.load()`による1回の再読込を試み、それでも2回目の`error`が発生した場合にのみidleへ巻き戻す緩和ロジックへ変更し、回帰テストを追加した。

### 20.5 進め方と敵対的レビューの2段構え

プランモードで実装計画を立てたうえで、計画段階の敵対的レビューと実装後のコードレビューの2段構えで進めた。計画レビューはFableの使用をオーナーが本セッション限りで特別に許可した(通常はサブエージェントによるFable使用は禁止方針)。コードレビューではMajor 2件(§20.4に記載した`ProjectFramesForSeconds`のUBリスクと`JoinControls`のidle巻き戻しリスク)を検出し、いずれも修正済みである。

### 20.6 品質ゲート

各タスクの実装単位でvitest緑・`npm run typecheck`(`tsc -b`)エラー0を確認し、W3(§20.3のバッチA2V配線)完了時点でwebui全体のtypecheckがexit 0であることを確認した。native doctestは217ケース全緑(§20.4参照)。最終の全体ゲート(typecheck全体の再実行・vitest全体の再実行・`scripts/build.ps1`)は本節時点では未実施で、後続で実施する。

### 20.7 残る手順

1. 最終の全体品質ゲート(typecheck/vitest全体/`build.ps1`)を実施する。
2. オーナーの承認を得る。
3. 承認後にコミットする。
4. 実機デプロイは、オーナーのAviUtl2終了の合図を待って実施する。

台帳側は[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-3(バッチA2VのnumFrames基準課題)を§3-9(解消済み)へ移動し、§2-2へ本節の目視確認待ち項目を追加した。

## 21. 「mockで検証していた」記述の是正とオーナー申告リストの反映(2026-07-18)

[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-1、[`REAL_BACKEND_CHECKLIST.md`](REAL_BACKEND_CHECKLIST.md)が「これまでの開発・検証の大半はmockで行われていた」という前提で書かれていたことについて、オーナーから是正の指摘があった。この記述はエージェント側の記録だけを根拠にした誇張だった。

### 21.1 何が誇張だったか

- **エージェントによる自動検証・デプロイ疎通確認は`config.model.backend=mock`経由**というのは正しい。GPUを占有しない安全策として意図的に選んだ運用であり、API・スキーマ・ジョブ管理の疎通確認が目的。
- しかし**オーナー自身は当初から`Nz-Videomni`を`.\run.ps1`で起動して検証してきた**。`model.backend="auto"`(既定値)により、必要なモデル一式が揃っている本機では起動時に自動的に`real`バックエンドが選ばれるため、オーナーの検証は最初から実GPU接続の本物の生成エンジンによるものだった。「開発・検証の大半はmockだった」という記述は、この区別を欠いたまま「エージェントはmockしか使わない」という自分の作業実態を投影して書かれたものであり、事実誤認だった。

### 21.2 オーナー申告(2026-07-18、realエンジンで検証済みの機能)

- **Createタブ(すべてreal合格)**: プリセット選択／CROP OUTPUT／t2v動画生成／i2v(冒頭キーフレームのみ)／i2v(複数キーフレーム)／スライダーによるキーフレームの位置調整UI／IC-LoRAでの動画生成／a2v動画生成／a2v+冒頭キーフレーム動画生成／**バッチa2vの機能テスト合格(2026-07-18、本日の新Shared仕様デプロイ〔§20〕後にテスト)**。
- **Chainタブ(すべてreal合格)**: プリセット選択／CROP OUTPUT／t2vクリップ連結動画生成／i2v(冒頭キーフレーム)クリップ連結動画生成／v2vで動画の続きを生成。Chunked Upsampleは常にEnabledで使用しているとの申告があり、chunked_upsample経路も実GPUで反復検証済みと解釈できる。

### 21.3 台帳・チェックリストへの反映

- [`PENDING_TASKS.md`](PENDING_TASKS.md) §2-1を、「realでの検証が未実施」という位置づけから「`REAL_BACKEND_CHECKLIST.md`のマトリクスのうち未確認の残項目を潰す」という位置づけへ書き換え、上記申告内容と残項目(モデル管理のロード操作、各項目内の一部個別技術チェック)を整理した。
- [`REAL_BACKEND_CHECKLIST.md`](REAL_BACKEND_CHECKLIST.md)の§1.1(目的)を是正し、§3(優先再検証2点)・§4.1〜4.9(実GPU通しマトリクス)・§6(記録欄・項目別結果表)に申告内容を反映した。マトリクス9項目中8項目(T2V/I2V/IC-LoRA/Chain Clips/Chain V2V/単発A2V/バッチA2V/chunked_upsample)を合格とし、モデル管理(4.8)は未実施のまま明示した(※この後同日中に4.8も合格へ更新。§21.4参照)。
- 優先再検証事項2点(`clip`/`clip_count`進捗表示、`overlap_frames`の`kv >= L`制約)は、オーナーがChainをrealで反復使用してきた事実を踏まえつつも、勝手に合格にはしなかった。進捗表示は「意識して観察した申告が無い」ため要注意観察のまま残し、`overlap_frames`制約は通常長クリップでの動作は合格とみなす一方、短いクリップ構成での明示テストが未実施である点のみ残した。
- §2-2(オーナー目視確認の残項目)は本節では変更していない。見た目・操作感の合否は別途オーナーが判断するため。

### 21.4 追加決定: モデル管理の合格追加と細部項目の再分類(2026-07-18)

上記の是正後、オーナーから2点の追加決定があった。

1. **モデル管理(チェックリスト4.8)を合格へ**。オーナーが実機でモデル管理のロード操作を検証し、サーバーログで裏付けた: `POST /api/v1/pipeline/load 200` → `Model-management overrides: {'transformer': 'models\ltx-2.3-gguf\Sulphur-2-base-distil-Q4_K_M.gguf'}` → `LTX worker ready` という切替完了、その後投入したt2v生成ジョブが`base=Sulphur-2-base-distil-Q4_K_M.gguf`で実行され264.7秒・peak_vram=10637MBで完走・動画DLも成功(2026-07-18 15:04-15:11のログ)。これにより`REAL_BACKEND_CHECKLIST.md`のマトリクス9項目(4.1〜4.9)は**9/9すべてで少なくとも基本パスが合格**した。ただし409`JOB_BUSY`排他挙動と危険ゾーンのunload→再ロードのサイクルは今回のログ範囲外のため、引き続きオーナー確認待ちとして残している。
2. **細部項目群の再分類**: 前回「オーナー確認待ち」として残していた個別技術チェックのうち、オーナーが「後日確認する」と明言した以下7点を、ブロッカー的な扱いから「実装済みでテスト待ち」という扱いへ整理し直した。
   - IC-LoRAの強度スライダー(`conditioning_attention_strength`等)の効き
   - Chain V2Vの`timeline.cutoutRange`(切り出し範囲)併用フロー
   - wav長→Frames自動調整の細部
   - バッチA2Vの再開(アプリ再起動後の続き実行)・上書き防止(`noClobber`付番)
   - 冒頭キーフレームなしi2v生成の実品質確認
   - 短いクリップ構成での`overlap_frames`(`kv >= L`エッジ)の明示テスト
   - Chainの`clip`/`clip_count`進捗表示の観察

   [`PENDING_TASKS.md`](PENDING_TASKS.md) §2-1にこの7点を「オーナーが後日確認すると明言(2026-07-18決定)」した項目として明記し、[`REAL_BACKEND_CHECKLIST.md`](REAL_BACKEND_CHECKLIST.md)の該当行(§3.1・§3.2・§4.2・§4.3・§4.5・§4.6・§4.7)には「オーナー後日確認予定(2026-07-18決定)」の注記を統一して付けた。上記7点に含まれない残りの細部(128グリッドUI、A2Vの5スロット全体併用・スタイルLoRA併用・短すぎる音声のブロック、バッチの開始前チェック、Chainの予想出力尺一致、モデル管理の409排他・unload→再ロード)は、オーナーからの後日確認の明言が無いため、従来どおり単純な「オーナー確認待ち」のまま据え置いた。

## 22. UI改修5点バッチ(2026-07-18)

同日中に、オーナー指示に基づくUI改修5点を実装した。全タスクでvitest緑・`npm run typecheck`エラー0を確認済み、未コミット。バックエンド(`Nz-Videomni`)は`config.yaml`の`limits`変更(1件のみ)を除き変更していない(凍結方針を維持)。

### 22.1 実装5点の要約

1. **生成サイズ上限4096化**: バックエンド`config.yaml`の`limits.max_width`/`max_height`を1920/1088(1080p実用上限)から4096(モデル絶対上限、64の倍数、`api/models.py`のge/le検証に一致)へ変更した。反映はバックエンド再起動後。フロント側はW/Hスライダーを新クラス`.size-field-stack`でCreate/Chain両タブとも上下並びへ変更した(既存の`.field-row`は不可侵のまま維持)。
2. **プリセットApplyボタンの完全廃止**: `PresetDropdown`(Create/Chain共通)から`applyButton`関連の文言・ボタンを削除した。別プリセットへの切り替えは従来どおり選択と同時に自動適用される。同一プリセットの再適用は「A→B→A」と経由すれば完全に再現できることを事前に裏取りしたうえで、Applyボタンによる明示適用方式(§13.7・§3-4で導入した仕様)を廃止した。`strings.ts`の`applyButton`4箇所を削除した(`shrinkModal.applyButton`は尺縮小モーダルの別ボタンであり対象外・温存)。
3. **Chainの「Remove clip」ボタンの❌絵文字化**: KEYFRAMESカード(§20.1)で採用した`.icon-action-button`共通クラスへ寄せ、導線を統一した。あわせて絵文字ボタンのCSS実体を`.kfc-action-button`から`index.css`の`.icon-action-button`へ格上げし、Keyframe/Clip/Jobカードの3者が同一定義を共用する構成にした。
4. **ジョブパネルの高さ動的化**: `.job-ledger-body`の`max-height`を固定寄りの`clamp(240px, 60vh, 720px)`から、ビューポート実高さに追随する`max(240px, calc(100dvh - var(--job-ledger-offset, 360px)))`へ変更した。狭幅時の25vhコンテナクエリは変更していない。stickyヘッダー化は不採用。縦900px未満の小ウィンドウでは旧仕様より表示領域が狭くなるトレードオフがあるが、内部スクロールで機能自体は維持される。
5. **JobCardのボタンをヘッダー右肩の絵文字群へ集約**: `completed`は🎞️(Insert。実行中は⏳、完了後は✅に切り替わる)＋❌(Delete)、`queued`/`running`は🚫(Cancel)、`failed`/`cancelled`は❌(Delete)を、いずれもヘッダー右肩へ配置した。下部の従来アクション行は撤去した(Insertエラー表示・`JoinControls`は変更していない)。アクセシブル名は既存の文言をそのまま`aria-label`として維持し、`App`/`App.chain`/`LibraryScreen`の既存テストとの互換性を保った。

### 22.2 Apply廃止の判断根拠(A→B→A裏取り)

Applyボタンは§13.7で「同一プリセットの再適用ができない」課題への対策として§3-4(2026-07-17)で導入された経緯があるが、オーナー方針転換(2026-07-18)により撤去することになった。撤去にあたり、「別のプリセットを一度経由してから元のプリセットへ戻る(A→B→A)」操作で同一プリセットの再適用が完全に再現できることをコードレベルで裏取りした。`applyPreset`(プリセット適用処理の実体)は、管轄下のフィールド(W/H/FPS/DURATION等)を選択のたびに無条件で上書きする実装になっており、「直前と同じ値だったから適用をスキップする」ような差分判定は存在しない。したがって、A→Bで一度別プリセットの値に上書きされたあと、B→Aで再びAの値に上書きされる際、Aへの2度目の適用はAへの初回適用と処理として完全に同一であり、「同一プリセットを選び直しても何も起きない」という当初の不具合はA→B→A経由では発生しない。この裏取りにより、Applyボタン(明示適用UI)という専用の解決策を廃止しても、既存のドロップダウン選択の自動適用だけで実用上の代替手段(A→B→A迂回)が成立することを確認したうえでの廃止と判断した。

### 22.3 1440p超の無警告OOM整理

生成サイズ上限を4096まで開放したことに伴い、大きなサイズでのVRAM溢れ(OOM)リスクの扱いを整理した。既存のspill_free_frames警告(チャンク化アップサンプル導入時に追加した、VRAM溢れの兆候を検出する警告UI)は1440p程度までの解像度を検知対象として設計されており、1440pを超える解像度の生成は警告の対象外である。つまり、1920pや4096pなど1440pを超えるサイズでOOMが起きても、事前・事中の警告は一切出ない「無警告OOM領域」になる。これは実装漏れではなく、オーナー方針「大きすぎるサイズはユーザーが自分で実験して確かめる(プリセットの提供までがこちらの仕事という整理)」に基づく意図的な判断であり、警告UIの追加は行わない。なお、バックエンド未接続時のフォールバックconfig(`/config`取得失敗時にフロントが使う既定値)の上限は1920のまま据え置いた。未接続時はそもそも生成自体が行えないため、この据え置きによる実害は無い。

### 22.4 敵対的レビュー所見と反映

実装着手前の計画段階で敵対的レビューを実施し、Major 3件を検出して計画へ反映した。

1. **JobCardの消費者の網羅漏れ**: JobCardのボタン配置変更(§22.1-5)は`JobCard`単体だけでなく、これを利用するChainタブのジョブ台帳・Library画面の履歴カード表示にも及ぶことを、計画段階で明示的に洗い出した。
2. **1440p超無警告OOMの明示不足**: 上限4096化(§22.1-1)の計画時点では「大きいサイズはOOMし得る」という記述に留まっていたが、レビューで「spill_free_frames警告の検知範囲外である」という具体的な事実を明示すべきと指摘され、§22.3のとおり整理した。
3. **埋め込み鮮度確認手順の欠落**: webui変更後の埋め込み(aux2)リビルドが正しく反映されているかを確認する手順が計画に無かったため、デプロイ前の確認手順として追加した。

### 22.5 品質ゲート(この時点の各タスク結果)

各タスクの実装単位でvitest緑・`npm run typecheck`(`tsc -b`)エラー0を確認済み。最終の全体品質ゲート(typecheck全体の再実行・vitest全体の再実行・`scripts/build.ps1`)は本節時点では未実施で、この後実施する。

### 22.6 残る手順

1. 最終の全体品質ゲート(typecheck/vitest全体/`build.ps1`)を実施する。
2. オーナーの承認を得る。
3. 承認後にコミットする。
4. 実機デプロイと、`config.yaml`変更(§22.1-1)を反映させるためのバックエンド再起動を実施する。

台帳側は[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-3(4096化)を§2「実装済み・ユーザーのテスト待ち」へ移動し、§2-2へ本節分の目視確認待ち項目を追加、§3-4にApplyボタン廃止の整合注記を追加した。

**補足(ログ中のエラーノイズについて)**: オーナーのログにあった`ERROR asyncio ... ConnectionResetError [WinError 10054]`は、動画ストリーミング(206 Partial Content)中にクライアント側(video要素)が接続を打ち切った際のWindows Proactorイベントループの既知の無害なノイズであり、機能上の問題ではない(エラー後も206応答は継続しており、生成・DLの成否には影響していない)。

## 23. オーナー目視一斉レビューの結果(2026-07-18、同日2回目)

2026-07-18、オーナーが実機で`Docs/PENDING_TASKS.md` §2-2に列挙していたオーナー目視確認の残項目を一斉レビューした。結果は合格多数・前提消滅によるクローズ4件・実機で新たに見つかったバグ4件・要望4件・従来「意味不明」だった項目4件の説明書き直し、という内訳になった。台帳側の反映はすべて完了している。

- **合格多数**: §2-2に残っていた項目のうち大半(バッチA2V折りたたみ節・`ui.pickFolder`・SettingsPanelのモデル管理・Chainプリセットドロップダウン・テーマ切替N8・Create/Chain/Batch画面の再構成・右上パネル新構成・DURATION・ファイルダイアログ・`SourceInputPanel`・ドラッグ&ドロップ一式・LibraryのControlタブ削除・キーフレームタイムラインUI一式(ピン操作・間隔ラベル・空カード・尺縮小モーダル・グレー帯・カードD&D・絵文字ボタン・Add keyframe・専用枠廃止確認・左端吸着域・位置0バッジ・Batch欄撤去・Shared冒頭化・Insert実尺)・UI改修5点バッチ一式(W/H上下並び・4096上限・Apply廃止・Chain❌統一・狭幅折返し・KEYFRAMES回帰なし)・バッチ行内編集(基本部分)・wav長→Frames自動調整の再計算挙動)が合格した。`Docs/PENDING_TASKS.md` §3-10「オーナー目視レビュー合格(2026-07-18第2回)」にまとめて記録した。
- **前提消滅によるクローズ(4件)**: Chainの制御タグ検出警告(IC-LoRA UI再設計〔§15〕で制御LoRAタグの手打ち自体が不可能になったため)、範囲外ピン時のGenerate無効化(キーフレームタイムラインUI〔§18・§20〕が範囲外にピンを打てない構造になったため)、旧「開始フレーム専用枠＋ゴースト表示」項目(§20で専用枠自体を廃止したため)、「Create/Chain/Library/設定パネルを通しで目視レビューする最終確認」(オーナー指示により「意味が広すぎて無意味」としてクローズ)の4件。`Docs/PENDING_TASKS.md` §3-11に記録した。
- **実機で新たに見つかったバグ(4件)**: ①A2V音声欄でRemove audioを押してもUnsupported file typeエラーメッセージが消えない、②ジョブパネルの高さが依然短く縦長動画1つでもスクロールが発生する、③タブ復帰後のプレビューサムネイル消失が`preload="auto"`では未解消(タブ状態保持の問題を含む)、④ジョブカードの🎞️→✅→🎞️サイクル後、再挿入ができなくなる。`Docs/PENDING_TASKS.md` §1-3〜§1-6として新規起票した。
- **新規要望(4件)**: ⑤ジョブカードへのシード値表示＋🎲(Random seed)／♻(Reuse seed)ボタンの追加、⑥バッチ表の行内編集ボタンの絵文字化(📝/🔁)、⑦ChainのSOURCEカード(`SourceInputPanel`)の絵文字ボタン化・縦積み配置・D&D範囲拡大、⑧デフォルト生成サイズ・フレーム数のstandardプリセット相当(1280×768・257f)への変更。`Docs/PENDING_TASKS.md` §1-7〜§1-10として新規起票した。
- **意味不明だった項目の説明書き直し(4件)**: 従来「Excelなど他アプリによるCSVロック時のautosave退避動作」「日本語ロケールでの`.aul2`フォールバック文言」「SettingsPanelのAPIキーバッジ(N13)」「SettingsPanelの危険ゾーン(N4)」は、何を・どう確認し、どうならOKかが本文から読み取れない状態だった。`BRIDGE_CONTRACT.md`・本ログ§5・§12.8の一次情報を調査し、各項目の対象機能・確認手順・OK/NG基準を明記する形に書き直したうえで`Docs/PENDING_TASKS.md` §2-2に残した(いずれもオーナー確認は未了)。あわせて、「V2VカードのJoinControlsテキストボタンとの併存」項目は、`JoinControls`(完了V2Vジョブのカードに出る「ソース動画と結合」用テキストボタン群であり、ChainのSOURCEカード`SourceInputPanel`とは別物)の説明を明記したうえで同節に残した。
- **`Docs/REAL_BACKEND_CHECKLIST.md`側の反映**: 「Chainの`clip`/`clip_count`進捗表示」(§3.1・4.4)と「バッチA2Vの再開(アプリ再起動後の続き実行)」(§4.7)は、同日中のオーナー追加確認によりいずれも合格へ更新した。対応する`Docs/PENDING_TASKS.md` §2-1の「後日確認7項目」リストからもこの2つを除去し、6項目へ整理した。

本節の反映により、`Docs/PENDING_TASKS.md`は§1に新規8項目(§1-3〜§1-10)を追加、§2-2を5項目(説明書き直し4件＋JoinControls1件)まで縮小、§3に§3-10・§3-11を追加、§2-1の後日確認リストを7点から6点へ整理した。

## 24. 次期改修8件バッチ(2026-07-18、同日3回目)

§23のオーナー目視一斉レビューで新規起票した`Docs/PENDING_TASKS.md`§1-3〜§1-10の8チケット(実機で見つかったバグ4件＋新規要望4件)を、同日中に実装した。バックエンド(`Nz-Videomni`)は§24.1-8の`config.yaml`(`generation_defaults`)1件のみを変更し、他は一切変更していない(凍結方針を維持)。**本節時点では全項目未コミット**。

### 24.1 【本丸】タブ状態保持(§1-5)

タブ切替(Create/Chain/Library)のたびにプリセット選択・フォーム値・プレビュー展開・IC-LoRA選択が失われる問題と、タブ復帰後にプレビューサムネイルが消える問題(§20で`preload="auto"`にしても未解消だった件)を、根本原因である「非アクティブタブのReactツリーがアンマウントされていた」構造から解消した。

- **方式変更**: 全タブ(Create/Chain/Library)を常時マウントし、非アクティブなタブは`hidden`属性で画面から隠す方式へ変更した。これによりアンマウント→再マウントによるstate初期化が発生しなくなり、プレビュー動画要素も破棄されないためサムネイル消失も構造ごと解消した。
- **keyの再設計**: 従来、各タブのkeyは右クリックルートからのプリフィル(`pendingIntent`)由来で決まっており、タブ切替のたびに実質的な再マウントが起きる設計だった。これをモードごとの独立カウンタ`remountTokens`(Create/Chain/Library各々)へ差し替え、**右クリックルートでプリフィルが飛んできたときだけ該当モードのトークンを+1する**方式にした。通常のタブクリックによる切替ではカウンタが変化しないため、常時マウント方式と組み合わせても編集中のstateが保持される。
- **事前に潰した退行リスク(敵対的レビュー由来)**: 計画段階のレビューで、素朴な「keyをプリフィルのたびに変える」設計のままだと「プリフィル発生→別タブへ切替→元のタブへ戻る」操作で編集中stateが消える退行が起きることが判明したため、上記のカウンタ方式(右クリックルート時のみ変化・通常のタブ往復では不変)へ設計を修正してから実装した。
- **検収テスト**: タブ往復後の状態保持(プリセット・フォーム値・プレビュー展開・IC-LoRA選択)と、プリフィル→別タブ→戻るでも編集中stateが保持されることを確認する回帰テストを追加した。

### 24.2 ジョブパネル高さ(§1-4)

§22.1-4で`.job-ledger-body`のみを動的化した対策では、縦長動画1つでもスクロールが発生するほどジョブパネルが短いままだった原因を、`.create-layout`側の`align-items: start`(左右カラムの高さを独立させる指定)が左フォームの実際の高さへジョブパネルを追従させない構造にあると特定した。

- `.create-layout`の`align-items: start`を撤去し、`generation-column`→`job-ledger`→`body`の3階層をflexで一続きに連結(flexリレー)することで、ジョブパネルが左フォームと同じ丈まで伸びる構造へ変更した。
- ビューポート上限(`calc(100dvh - ...)`)は、従来`.job-ledger-body`に付いていたものを`generation-column`側へ移設した。これにより、多ジョブでカードが積み上がってもページ全体が際限なく伸びる(ページ暴走)ことを防ぎつつ、左フォーム側は`.job-ledger-body`の内部スクロールに閉じ込められず適切に連動する。狭幅時の25vh(コンテナクエリ)は変更していない。
- **事前に潰した退行リスク(敵対的レビュー由来)**: 計画段階のレビューで、単純に`align-items: start`とビューポート上限を両方撤去すると、多ジョブ時にページ全体が無制限に伸びる退行(ページ暴走)が生じることが判明したため、上限を撤去するのではなく`generation-column`側へ移設する設計に修正してから実装した。

### 24.3 Insert✅サイクル(§1-6)

**真因**: ジョブカードの🎞️(Insert)→✅(挿入済み)→🎞️(押し直し)のサイクルで、✅から🎞️へ戻した後に再度🎞️を押しても挿入されなかった原因は、✅クリック時に**状態をリセットするだけでなく再挿入も行っていた**ことによる。1回目のInsertで確定した挿入位置(カーソル位置)へ、✅クリック時にもう一度同じ位置への挿入を試みるため、AviUtl2タイムライン側でオブジェクトの重なり(overlap)扱いとなり失敗していた。

- **修正方針**: ✅クリックの挙動を「状態のリセットのみ(再挿入はしない)」に変更した。これにより🎞️→✅→🎞️の3段階サイクルが、①挿入して✅へ切替 → ②✅クリックは表示を🎞️へ戻すだけ → ③再度🎞️クリックで(新しいカーソル位置への)再挿入が可能、という意図どおりの循環になった。
- **エラー時の表示改善**: 従来、Insertがエラーになった場合の表示がidle(未挿入)状態と見分けがつかなかった。エラー時は⚠️アイコン＋`title`属性にエラー文言を表示するよう変更し、失敗したことが視覚的に分かるようにした。

### 24.4 A2V音声欄のD&Dエラー残留(§1-3)

A2V音声欄へ非対応ファイルをドラッグ&ドロップした際に表示される「Unsupported file type...」エラーメッセージが、その後「Remove audio」を押しても消えずに残る不具合を解消した。

- 共有フック`useFileDrop`・`DropZone`コンポーネントに新規`resetErrorSignal`を追加し、「Remove audio」(Clear)操作からこのシグナルを発火してドロップエラー表示を確実にリセットするよう配線した。

### 24.5 シード値表示＋🎲／♻(§1-7)

ジョブカードのヘッダーにシード値表示と、SEED欄横のクイック操作ボタン2種を追加した。

- **シード値表示**: ジョブカードヘッダーに`#シード値`を表示する。シードが未解決(バックエンドが乱数を選んだ)の場合は「ランダム」と表示する。値は既存の`result.seed_used`(APIに既存のフィールド、バックエンド変更不要)からそのまま取得する。
- **🎲(Random seed)**: SEED欄の横に配置し、押すとSEED欄へ`-1`(ランダム化の合図)をセットする。
- **♻(Reuse seed)**: 押すとジョブパネル最上段のジョブカードのシード値をSEED欄へコピーする。最上段カードにシード値が無い場合はdisabledにする。
- 新規`webui/src/jobs/seedUtils.ts`にシード関連の純粋関数(表示用フォーマット・最新シード抽出等)を集約した。

### 24.6 バッチ表ボタン絵文字化(§1-8)

バッチ表(N5、§12.9)の行内編集にあったテキストボタンのうち、「Use shared prompt」→📝、「Reset to waiting」→🔁へ絵文字化した。アクセシブル名(文言)は`title`/`aria-label`にそのまま維持し、見た目のみ既存の`.icon-action-button`共通クラス(§20・§22.1-3で導入)へ寄せた。

### 24.7 SOURCEカード改修(§1-9)

ChainタブのSOURCEカード(`SourceInputPanel`)を改修した。

- 「Choose image or video...」→📁、「Clear」→🔁へ絵文字化し、両ボタンをカード右肩へ縦積み配置(上から🔁・📁の順)にした。
- ドラッグ&ドロップの受け入れ範囲を、従来のドロップ欄限定からカード全体へ拡大した。ドラッグ中はカード全体をハイライトし、非対応ファイルドロップ時はカード内にエラー行を表示する。
- KEYFRAMESカード(§20.1)の絵文字ボタン配置・カード全体D&Dと導線を揃える狙いで実装した。

### 24.8 デフォルト生成サイズ・フレーム数のstandard化(§1-10)

デフォルトの生成サイズ・フレーム数(従来512×320・49f)を、standardプリセット(1280×768・257f)相当へ完全一致させた。

- バックエンド`config.yaml`の`generation_defaults`を、`width=1280`・`height=768`・`num_frames=257`・`crop_output`(1280×720)へ変更した(`standard_720p`プリセット定義との突合済み)。
- フロント側の`cropOutput`初期化ロジックをハードコードから`/config`のプリセット参照へ変更し、バックエンド側の値と自動的に整合する構造にした。
- **反映にはバックエンド再起動が必要**(§22.1-1の4096化と同様、`config.yaml`変更は起動時読み込みのため)。

### 24.9 品質ゲート

| スイート | 件数 | 備考 |
|---|---|---|
| `npm run typecheck`(`tsc -b`) | エラー0 | |
| webui vitest | **880件パス／10件skip** | skip 10件は既知の`backend.integration`(mock未起動環境では実行しない既存スイート) |
| native doctest(`NzVideomni_tests.exe`) | **217ケース** | 全緑(native側は§24.4〜24.8ではソース不変のため件数は§20.4から不変) |
| `scripts/build.ps1`(embedded) | 成功 | |
| 埋め込み(aux2)鮮度確認 | OK | webui変更後の埋め込みリビルドが正しく反映されていることを確認(§22.4-3で追加した確認手順を踏襲) |

### 24.10 計画段階の敵対的レビュー(事前に反映したMajor2件)

実装着手前の計画レビューでMajor 2件を検出し、いずれも計画へ反映してから実装した(詳細は§24.1・§24.2参照)。

1. **プリフィルkey反転退行**: タブ状態保持(§24.1)の素朴な実装案では、「プリフィル発生→別タブへ切替→元のタブへ戻る」で編集中stateが消える退行が起きる設計だった。→ keyを「プリフィル由来」から「モードごとのremountTokensカウンタ(右クリックルート時のみ+1)」へ再設計して回避した。
2. **ビューポート上限撤去の退行**: ジョブパネル高さ(§24.2)の素朴な実装案では、`align-items: start`とビューポート上限を両方単純に撤去すると、多ジョブ時にページ全体が無制限に伸びる退行(ページ暴走)が起きる設計だった。→ 上限を`generation-column`側へ移設する形に修正して回避した。

### 24.11 残る手順

1. オーナーの承認を得る。
2. 承認後にコミットする。
3. 実機デプロイを実施する。
4. バックエンド再起動(§24.8のデフォルト値反映のため必須)を実施する。

台帳側は[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-3〜§1-10(本節で実装した8チケット)を§2-9としてまとめて「実装済み・ユーザーのテスト待ち」へ移動し、§2-2へ本節分の目視確認待ち項目を追加した。

## 25. オーナー目視確認(2026-07-18、第3弾バッチ)

第2弾UI改修バッチ(§24)についてオーナーが実機で目視確認した結果、合格5件・不合格2件(新症状)・新規相談1件という内訳になった。

- **合格(5件、`Docs/PENDING_TASKS.md` §2-2該当分から§3-12へ移動)**: タブ状態保持(Createでプリセット選択・数値変更・プレビュー展開→Chain→Library→Create往復ですべて保持。プリフィル経由の確認のみ§2-2に残置)、ジョブカードのシード表示＋SEED欄🎲♻、音声欄エラーがRemove audioで消える、バッチ表の📝🔁、SOURCEカードの右肩🔁📁＋カード全体D&D。
- **不合格(2件、新症状。`Docs/PENDING_TASKS.md` §1-3・§1-4として再起票)**:
  - **ジョブパネル同丈化**: §24.2の対策後も依然として左パネルと同じ丈にならない。前回対策で`generation-column`側へ移設したビューポート上限が原因の疑いがあり、深掘り調査が必要。
  - **Insert再挿入**: §24.3の修正で✅→🎞️へのリセットまでは合格したが、その後の🎞️再挿入操作が「Could not open destination file」エラーで失敗する新症状が見つかった。あわせてオーナーから、⚠️(エラー)状態のクリックも✅と同様に🎞️へ戻すだけ(再挿入はしない)へ変更する仕様追加の確定指示があった。
- **新規相談(1件、`Docs/PENDING_TASKS.md` §1-5として起票)**: バッチA2VのCSV状態管理(中断再開機能)の廃止検討。オーナー提案は、状態保存を持たずフォルダの中身から都度バッチを組み立てるだけのステートレス設計への変更。理由は操作ミスの温床になりやすいこと・再生成手順の分かりにくさ・手動運用(出力先確認＋完了済み音声ファイルの移動)で十分という判断。方針議論中。
- **台帳側の整理**: 冗長化していた`Docs/PENDING_TASKS.md`を全面整理した。旧§2-3〜§2-9(パリティ実装総括・UIリデザイン一式・4波デバッグ・IC-LoRA UI再設計・生成サイズ上限4096化バッチ・次期改修8件バッチ)は実装済み・合否決着済みとして除去し、§3-12(本日の合格5件)・§3-13(旧バッチ群のクローズ整理注記)へ集約した。§2に残したのは、プリフィル経由のタブ状態保持・backend再起動後のデフォルト値確認・従来「意味不明」だった4項目(CSVロック退避・aul2フォールバック・APIキーバッジ・危険ゾーン、うちCSVロック退避はバッチA2V廃止検討の結論待ちである旨を注記)・JoinControls関連・キーフレームタイムラインの判断待ち2点(§2-3)のみ。§1には上記不合格2件(§1-3・§1-4)と新規相談(§1-5)を追加した。

## 26. 第4弾バッチ(2026-07-18、同日4回目): ジョブパネル同丈・Insert再挿入・バッチA2Vステートレス化・ブリッジ道連れ削除

§25で不合格となった2件(ジョブパネル同丈化・Insert再挿入)の再デバッグ、相談中だったバッチA2VのCSV状態管理(中断再開機能)廃止の実装、およびそれに伴うネイティブブリッジの道連れ削除を同日中に実施した。全タスク緑。本節時点では全項目未コミット。バックエンド(`Nz-Videomni`)は一切変更していない(凍結方針を維持)。

### 26.1 ジョブパネル同丈化(三度目の正直・削除中心)

§24.2の対策(`align-items: start`撤去＋ビューポート上限の`generation-column`側への移設)後もなお左パネルと同じ丈にならなかった(§25で新症状として再起票)。

- **真因**: §24.2の対策で`.generation-column`へ移設したビューポート上限(`calc(100dvh - ...)`)が、grid stretchによる同丈化そのものを頭打ちにしていたこと。上限があるかぎり、左フォームがどれだけ長くなってもジョブパネル側はその上限で頭打ちになり、左と同じ丈まで伸びきれない。
- **対策**: `.generation-column`のビューポート上限と`--job-ledger-offset`カスタムプロパティを削除した。あわせて`.job-ledger-body`を`flex: 1 1 0`(`flex-basis: 0`)へ変更し、ジョブ内容の実高さをカラム自体の高さ計算から除外することで、行の高さが常に左フォーム駆動になるようにした。
- **狭幅時の復帰**: 狭幅の`@container`条件では`flex: 1 1 auto`へ戻した。`basis: 0`のままだと1カラムレイアウト時に台帳が高さ0まで潰れてしまうため。
- **デッドCSS削除**: 併せて未使用だった`.app`/`.app-main`のCSSも削除した。
- **検証**: 実機同等のChromiumで、通常幅・長フォーム・大量ジョブ・狭幅ドッキングの各条件を実測し、左右のカラムが同丈になることを確認した。

### 26.2 Insert再挿入(「Could not open destination file」の解消)

§24.3で修正した✅→🎞️のリセット挙動は合格したが、その後の🎞️再挿入操作が新症状として失敗していた(§25)。

- **真因**: 1回目のInsertでダウンロードした動画ファイルを、AviUtl2がタイムライン上のオブジェクトとして開いたまま保持するため、同じ`jobId`固定のダウンロード先へ2回目のDLを試みるとファイル共有違反になっていた。
- **修正方針**:
  - ✅/⚠️のクリックはいずれも状態のリセットのみとし、再挿入は行わない(⚠️についてはオーナーから§25で確定指示があった仕様追加)。
  - 再挿入(🎞️の押し直し)は、webui側で記憶しておいたファイルパスが指す先を使ってダウンロードをスキップする。
  - ネイティブの`backend.downloadVideo`に`reuseIfPresent`パラメータを新設した。**plain(結合前の単一)動画限定のオプトイン**で、宛先ファイルが既に存在しサイズが0より大きければ再ダウンロードをスキップする。joined(V2V結合済み)動画はクロスフェード長の変更で内容そのものが変わり得るため対象外とし、常に再取得する。

### 26.3 バッチA2VのCSV状態管理廃止(ステートレス化)

§25で起票した相談(§1-5)の結論として、バッチA2Vの中断再開機能(`batch_a2v_manifest.csv`による行状態の読み書き)自体を廃止し、状態を一切持たないステートレス設計へ変更した。

- **廃止した範囲**: `batch_a2v_manifest.csv`の読み書き・autosave(退避)・`WRITE_LOCKED`処理を全廃した。フォルダスキャン→メモリ内の行リスト構築→実行、という経路のみが残る。
- **維持した範囲**: 行内編集(プロンプト直接編集・🔁「Reset to waiting」・📝「Use shared prompt」)・画像割当ドロップダウン・開始前ゲート・Shared行の1枚目frame0強制条件付けは、いずれもメモリ内の状態として維持した。
- **新しい挙動**: ウィンドウ(WebView2/Create画面)を閉じると行状態は消える。バッチ開始・再開はすべて手動で、「再実行」は常に全行の再生成になる。既存の出力ファイルは`noClobber`の連番別名(`_2`, `_3`...)で保護されるため上書きはされない。notice文言はこの新しい実態(非保存・手動再開・再実行は全行再生成)に合わせて更新した。
- **Clearボタン**: 調査の結果CSVとは無関係な機能と判明したため、そのまま存置した。
- **削減規模**: `manifestCsv.ts`とそのテスト(約517行)を含め、正味650行超のコードを削減した。

### 26.4 ネイティブブリッジの道連れ削除

§26.3のCSV廃止に伴い、CSV読み書き専用だったネイティブブリッジのメソッドを道連れで削除した。

- `fs.readTextFile` / `fs.writeTextFileAtomic` / `WRITE_LOCKED`エラーコードを、契約(`webui/src/bridge/types.ts`)・mockブリッジ・nativeハンドラ・doctestのすべてから削除した。
- `BRIDGE_CONTRACT.md`は撤去注記方式(該当節を「撤去済み」と明記し、契約バージョンはv6のまま)で更新済み。`downloadVideo`の`reuseIfPresent`は§4.20として別途文書化済み。**`BRIDGE_CONTRACT.md`自体は別タスクが既に更新済みのため、本タスクでは触っていない。**

### 26.5 計画段階の敵対的レビュー(Major4件を反映)

実装着手前の計画段階で敵対的レビューを実施し、Major 4件を検出して計画へ反映した。

1. **joined(結合済み)動画の再取得保全漏れ**: `reuseIfPresent`の素朴な適用案では、V2V結合済み(joined)動画も再ダウンロードをスキップし得る設計だった。クロスフェード長を変えて再joinした場合に古い内容のまま挿入されてしまう退行になるため、joined動画は対象外とし常に再取得する設計に修正した(§26.2)。
2. **`flushManifest`呼び出し箇所の網羅漏れ**: CSV廃止にあたり、`flushManifest`の呼び出し元が計画時点で洗い出せていなかった。実装前に呼び出し箇所5箇所をすべて洗い出し、道連れで削除すべき経路を確定させてから着手した。
3. **Clearボタンの扱い誤り**: 当初はCSV廃止に伴いClearボタンも削除する想定だったが、レビューでCSVと無関係な機能であることが判明し、存置する判断に修正した。
4. **テストimportの漏れ**: `manifestCsv.ts`削除に伴うテストファイルのimport文の漏れが計画時点で見つかり、削除前に洗い出して反映した。

### 26.6 品質ゲート

| スイート | 件数 | 備考 |
|---|---|---|
| `npm run typecheck`(`tsc -b`) | エラー0 | |
| webui vitest | **858件緑** | ブリッジ削除(§26.4)後の全体実行 |
| native doctest(`NzVideomni_tests.exe`) | **217ケース(新基準)** | 前回基準(§24.9)215ケースから、`reuseIfPresent`関連のParseテスト+3・fs系(`fs.readTextFile`/`fs.writeTextFileAtomic`関連)削除分−3を反映した実測値 |
| `scripts/build.ps1`(embedded) | 成功 | |

### 26.7 残る手順

1. オーナーの承認を得る。
2. 承認後にコミットする。
3. 実機デプロイを実施する。
4. **バックエンドの変更が無いため、再起動は不要。**

台帳側は[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-3(ジョブパネル同丈化)・§1-4(Insert再挿入)・§1-5(バッチA2VのCSV状態管理廃止相談)を「2. 実装済み・ユーザーのテスト待ち」§2-4へ1項目に集約して移動した。§2-2の「CSVロックした状態での自動保存退避動作」の目視項目は機能自体の廃止で前提が消滅したため削除し、代わりにジョブパネル同丈(4条件)・Insert再挿入・V2V再join再挿入・バッチA2Vの新仕様(CSV非生成・状態非保存)の4点を新規目視項目として追加した。

## 27. 第5弾ミニバッチ(2026-07-19): バッチ開始時再判定・fpsMismatchガード引き算・Clearボタン削除

§26のバッチA2Vステートレス化(§26.3)後に見つかった不具合の根本対応、既存ガードの整理、およびオーナーの明示指示による撤回対応の3件を実施した。全テスト緑、本節時点では未コミット。バックエンド(`Nz-Videomni`)は本セッションでも一切変更していない(凍結方針を維持)。

### 27.1 バッチ開始時再判定の移植

Skip行を🔁(Reset to waiting)で戻してからStartすると、`num_frames>=9`という意味不明なFailedになるバグが見つかった。これはステートレス化(§26.3)でCSVによる状態保存を廃止した際、fps変更後の再判定を欠いたまま実行に入ってしまう経路が生じていたことが根本原因である。

Gradio原典の`_plan_rejudgement`/`_apply_plan`相当の純粋関数`rejudgeRows`を新規`manifestMerge.ts`へ追加し、`useBatchForm.start()`の冒頭で全Waiting/Failed行を現在のfpsで再判定してから実行する構造にした。再判定は、481フレーム超になる行を強制的にSkipへ戻し、有効な行はframesを再計算する。

原典との意図的な差分3点をJSDocに明記した。

1. `duration<=0`の行は強制的にSkipにする。
2. 有効化(Skip解除)時に`skipReason`をクリアする。
3. `Generating`(実行中)行は再判定の対象から除外する。

判定ゲートには`rawFramesFor`(無クランプ版)を使用している。framesForの誤用による判定死(有効な行が無条件にFailedへ落ちる不具合)を防ぐための差別化テストを追加した。

### 27.2 fpsMismatchガードの引き算

「fps変更後は再スキャンまでStartを無効化する」という既存ガードは、開始時再判定(§27.1)が存在しなかった時代の代替品だったため、`canStart`から除外した(Start無効化という役目は再判定の実装により終了した)。表示は無効化理由ではなく、「FPSがスキャン時と異なります。フレーム数はバッチ開始時に現在のFPSで自動的に再計算されます」という情報ヒントへ変更した。

これにより、「fpsを変えたのち🔁でSkip行を戻し、新しいfpsでのフレーム数のまま実行する」というルートが実際に機能するようになった。従来はこのルート自体が旧fpsMismatchガードに塞がれて到達不能になっており、この点は敵対的レビューが発見した。

### 27.3 IMAGE FOLDERのClearボタン削除

§26.3(バッチA2Vステートレス化)の敵対的レビューでは「Clearボタンは調査の結果CSVと無関係と判明したため存置した」と判断していたが、本弾ではオーナーの明示指示によりこの判断を撤回し、IMAGE FOLDERのClearボタンを削除した。手打ちで入力欄を空にしてフォーカスアウトする経路(`setImgDir("")`)がClearボタンと完全に同じ後始末を行うことを確認したうえでの削除であり、`clearImgDir`本体とen/ja両方の関連文言も削除した。

### 27.4 品質ゲート

manifestMergeの新規テスト12件、batch/i18n関連テスト83件がいずれも緑、`npm run typecheck`(`tsc -b`)エラー0を確認済み。最終の全体品質ゲート(typecheck全体の再実行・vitest全体の再実行・`scripts/build.ps1`)は本節時点では未実施で、この後実施する。

### 27.5 残る手順

1. 最終の全体品質ゲート(typecheck/vitest全体/`build.ps1`)を実施する。
2. オーナーの承認を得る。
3. 承認後にコミットする。
4. 実機デプロイを実施する(バックエンドは無変更のため再起動は不要)。

台帳側は[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-4の「Clearボタンは調査の結果CSVと無関係と判明したため存置した」という記述を撤回注記へ改稿し、§4-1の「開始時の事前検証(re-judgement)は持たない」という記述を実装済みへ更新し、本節の3項目を「実装済み・ユーザーのテスト待ち」へ新規項目として追加、§2-2へ本節分の目視確認待ち項目3点を追加した。

## 28. 第6弾ミニバッチ(2026-07-19): バッチ上限のDURATION連動・Skipツールチップ・notice文言差し替え

§27(第5弾ミニバッチ)に続けて同日中に実施した3件。全テスト緑、本節時点では未コミット。バックエンド(`Nz-Videomni`)は本セッションでも一切変更していない(凍結方針を維持)。

### 28.1 バッチ上限のDURATION連動(新機能)

従来、バッチA2VのSkip判定における上限フレーム数は481固定(Gradio原典`batch.py`も同じ固定値)だった。これをCreateタブのDURATIONスライダー値(`generationValues.numFrames`)に連動させ、バッチが「今設定しているDURATIONより長い音声」をSkipする形へ変更した。

- `manifestMerge.ts`の`scanToRows`/`rejudgeRows`(§27.1で新設)に、両者共通のオプション引数`maxFrames`を追加した。実効上限は`Math.min(opts.maxFrames ?? 481, 481)`で計算する——**省略時は従来どおり481が上限**、**481を超える値(例: 9999)を渡しても内部で481へクランプ**され、サーバーの実ハード上限(481フレーム)を超えることはない。
- skipReasonのキーを`"over-481f"`から`"over-cap"`へリネームした(可変上限になったため、固定値481を示唆する旧名を廃止)。
- `useBatchForm.ts`の`scan()`・`start()`双方の経路へ`maxFrames: generationValues.numFrames`を配線した。`generationValues`は`CreateScreen.tsx`から渡される(後述)。
- **敵対的レビュー指摘への対処**: `CreateScreen.tsx`が`BatchSection`へ渡す`generationValues`オブジェクトの`useMemo`依存配列に、新規追加した`form.values.numFrames`が漏れていた場合、DURATIONを変更してもメモ化された値が古いまま据え置かれ、バッチ側のSkip上限が黙って追随しなくなる(機能が無言で死ぬ)罠があった。依存配列へ`form.values.numFrames`を追加して封じた。`useBatchForm.ts`側の`scan`用`useCallback`の依存配列にも同様に`generationValues.numFrames`が必要であることをコード中コメントで明記し、レビューで確認済み。
- **既定挙動の変化(重要)**: Createタブの既定DURATIONは257フレーム(24fpsで約10.7秒)のため、**バッチのSkip上限も既定で約10.7秒に下がる**。従来は481フレーム(約20秒)がハード上限だったため、10.7秒〜20秒の音声は従来どおり通っていたが、本弾以降は**DURATIONを上げない限り既定でSkipになる**。20秒までの音声を通したい場合は、Createタブ側でDURATIONを481(または相応の値)まで上げる必要がある。
- `fpsMismatch`ヒントの文言を意味拡張した。従来「FPSがスキャン時と異なります」だったのを「FPSまたはDURATIONがスキャン時と異なります」へ変更し(en/ja両方)、スキャン後にDURATIONが変わった場合もこのヒントが出るようにした(`scannedMaxFrames`を新規追跡し、スキャン時に固定したfpsと同様にDURATION上限もスナップショットして現在値と比較する)。

### 28.2 Skip理由のツールチップ

`BatchTable`のSkip行❌バッジにhoverすると理由が表示されるようにした。

- 新規純関数`skipReasonTitle(stat, skipReason, skipReasons)`(`BatchTable.tsx`)を追加し、`stat === "Skip"`かつ`skipReason`が既知キー(`strings.batch.skipReasons`)の場合のみ`title`属性を返す。**未知のreason(将来のビルド差分等で見たことのない値が来た場合)は`title`省略で安全側に倒す**(空白・文字化けしたツールチップを出さない)。
- 表示文言は2種(en/ja)。
  - `wav-only-alpha`: 「.wavファイルでないか、長さを読み取れなかった」旨。
  - `over-cap`: 「音声が現在のDURATION上限を超えている」旨に加え、**「DURATIONを上げる(またはFPSを下げる)→🔁で待機に戻す→開始」という復帰手順を文言内に明記**した。これは、**Skip行が§27.1の開始時再判定(`rejudgeRows`)の対象から除外されている**ため、DURATIONを上げただけでは自動復帰せず、必ずユーザーが🔁で明示的に待機へ戻す操作が要ることを踏まえた案内である。
- 新規テストファイル`webui/src/modes/batch/BatchTable.test.tsx`を追加し、`over-cap`行のバッジに`title`属性として該当文言が付与されることを検証した。

### 28.3 バッチ説明文の差し替え

`batch.notice`(バッチA2V節の説明文、Batch画面上部に常設表示)をオーナー決定の文面へ差し替えた。

- **新文面(en/ja)**: 「1ファイルにつき1本の動画をまとめて生成します。(出力フォルダは音声フォルダの隣に自動作成されます)」という趣旨。
- **出力フォルダの自動作成は事実**: native側のコード実読で、`bridge.cpp`の`EnsureDirectoryTree`(出力先ディレクトリツリーを再帰的に作成する関数、`backend.downloadVideo`の宛先ディレクトリ確定時に呼ばれる)と、`http_client`側の親ディレクトリ再帰作成の2箇所で二重に担保されていることを確認済み。
- **旧文面から脱落した情報(オーナー採択によりこの位置からは撤去)**:
  1. **stateless警告**: 「バッチの状態はどこにも保存されません。ウィンドウを閉じると行の一覧と進行状況は失われます」という趣旨の警告文。
  2. **再開が手動である旨**: 「再開したいときはご自分でもう一度『開始』を押してください」という案内。
  3. **再スキャン時の全行再生成**: 「再スキャンして実行し直すと、すべての行が最初から生成し直されます。既存の出力ファイルは上書きされず、新しい実行結果は連番付きの別名で保存されます」という`noClobber`連番保存の説明。
  - これら3点はいずれも§26.3(バッチA2Vステートレス化)で確定した実装仕様そのものを否定するものではなく、単に本notice文面の掲載位置から外れただけである(実装・挙動自体は変更なし)。表示位置の再検討・別途の掲載場所の要否はオーナー判断待ちとして記録しておく。

### 28.4 品質ゲート

`npm run typecheck`(`tsc -b`)エラー0、webui vitest **71ファイル884件パス・10件skip(894件)**(§26.3時点の858件から+26)。すべて本セッション実施分。

### 28.5 残る手順

1. オーナーの承認を得る。
2. 承認後にコミットする。
3. 実機デプロイを実施する(バックエンドは無変更のため再起動は不要)。

台帳側は[`PENDING_TASKS.md`](PENDING_TASKS.md) 「2. 実装済み・ユーザーのテスト待ち」へ本節の3項目を新規追加し、§2-2へ本節分の目視確認待ち項目5点を追加した。

## 29. 第6弾ミニバッチのオーナー目視合格(2026-07-19)

§28(第6弾ミニバッチ)の実装分について、同日中にオーナーが実機で目視確認し、以下5点をすべて合格と回答した。

- **DURATION連動**: CreateタブのDURATIONを下げてバッチをスキャンするとDURATION超の音声がSkipになり、DURATIONを上げて🔁で待機へ戻したうえでStartすると実行される(§28.1)。
- **FPS・DURATION変更時の情報ヒント**: スキャン後にFPSまたはDURATIONを変えても、Startが無効化されるのではなく情報ヒントが表示される(§28.1)。
- **Skipバッジのツールチップ**: ❌バッジのhoverで理由(`wav-only-alpha`/`over-cap`)が表示され、`over-cap`では復帰手順(DURATIONを上げる→🔁で待機に戻す→開始)まで含まれる(§28.2)。
- **notice新文面(en/ja)**: 「1ファイルにつき1本の動画をまとめて生成/出力フォルダは音声フォルダの隣に自動作成」の文面へ差し替わっている(§28.3)。
- **出力フォルダの自動作成**: 出力フォルダが存在しない状態から1件実行し、フォルダが自動生成されることを実機で実証(§28.3)。

あわせて、既定DURATION257(24fpsで約10.7秒)での挙動変化——従来通っていた10.7〜20秒の音声が既定でSkipになる——も想定どおりであることを確認した(§28.1)。

台帳側は[`PENDING_TASKS.md`](PENDING_TASKS.md)で本節の目視5点と§28の実装記録を「3. 実装済み」(§3-15・§3-16)へ集約し、§2をチェックリスト形式へ再編した。コミット・デプロイの実施可否はオーナー判断(§28.5の残る手順は据え置き)。

## 30. オーナー追加確認合格(2026-07-19): バッチ改修一式・実GPU細部・BatchTableクローズ

§29(第6弾の目視合格)に続けて、オーナーから「実際には確認済み」の項目についてまとめて申告があり、以下を合格としてクローズした。台帳の§2チェックリストに残っていたが文脈が伝わっていなかった/以前のセッションで既に確認されていた項目の棚卸しである。

- **BatchTableデバッグ(旧`PENDING_TASKS.md` §1-2の弾丸)**: UIリデザイン(§13)で後回しにしていたバッチ表のデバッグは、一連のバッチ改修(📝🔁の絵文字化・CSV廃止によるステートレス化〔§26.3〕・開始時再判定〔§27.1〕・Skipツールチップ〔§28.2〕・DURATION連動〔§28.1〕)でまとめて完了とオーナーが判断。クローズした。
- **第4・5弾バッチの目視項目(今回セッションで合格)**: ジョブパネル同丈(§26.1)/V2V結合の再join・再挿入(§26.2)/JoinControls(V2V結合UI)の見た目に違和感なし/backend再起動後の初期値がstandard一致(§24)/Batchの4項目(ステートレス新仕様〔§26.3〕・バッチ開始時再判定でのSkip強制復帰〔§27.1〕・fps変更後の再スキャン無しStart＋再判定反映〔§27.2〕・IMAGE FOLDERのClearボタン削除〔§27.3〕)。
- **実GPU細部(以前のセッションで確認済み・2026-07-19申告)**: IC-LoRA強度スライダーの効き/wav長→Frames自動調整の細部/冒頭キーフレームなしi2v生成の実品質/バッチA2VのnoClobber付番。
- **実GPU細部(今回セッションで合格)**: バッチA2Vの開始前チェック/Chainの予想出力尺プレビュー一致。

台帳側は[`PENDING_TASKS.md`](PENDING_TASKS.md)へ新規まとめ項目§3-17を追加して上記を集約し、§1-2からBatchTable弾丸を除去、§2-1(実GPU細部)・§2-2(目視)から合格分を除去した。[`REAL_BACKEND_CHECKLIST.md`](REAL_BACKEND_CHECKLIST.md) §4.2/§4.3/§4.4/§4.6/§4.7の該当行も合格へ整合させた。この結果、§2に残るテスト待ちは「プリフィル経由のタブ状態保持」「リロード後の再挿入」「Chain V2VのcutoutRange併用」「短いクリップのoverlap_frames制約」等のごく少数のみとなった(§2-3のオーナー判断待ち2点は別扱い)。コミット・デプロイの実施可否はオーナー判断。

## 31. §2-1「その他オーナー確認待ち(後日明言なし)」6項目のクローズ(2026-07-19最終判定)

§30に続けて同日中に、オーナーから`PENDING_TASKS.md` §2-1「その他オーナー確認待ち(後日明言なし)」6項目について最終判定があった。ドキュメントのみの反映で、コード変更は無し。

- **IC-LoRA 128グリッドUI**: 合格。
- **A2Vの5スロット(KEYFRAMES最大5枚)全体併用**: 合格。
- **スタイルLoRA併用**: (a) A2V＋スタイルLoRAタグは合格。(b) A2V＋参照動画制御LoRAの併用は確認項目から削除した。2026-07-15確定の「音声⇔参照動画の相互排他」設計により、この組み合わせを送信するUI上の入口がそもそも存在せず、本ツールのスコープ外と判明したため。チェックリストがGradio側の能力(音声と参照動画を同時に送れる構成)を誤って転記していたことが原因である。
- **短すぎる音声の事前ブロック**: 合格。Generate無効化とDURATION自動調整の両方の挙動を確認した。
- **モデル管理の409排他・危険ゾーンunload→再ロード**: 合格。生成中のLoad試行が「Cannot switch models while a generation job is running.」で拒否されること、生成完了後のモデル切替→t2v/a2v成功を確認した。あわせて、オーナーから「危険ゾーンが見当たらない」との申告があったため、`DangerZonePanel`の表示有無の切り分けは`PENDING_TASKS.md` §2-2の危険ゾーン(N4)項目へ一言注記して引き継いだ(本項目は操作自体の合否のみを扱う)。
- **chunked_upsampleの768pストレステスト**: 合格・決定的証跡あり。`Nz-Videomni/outputs/a2472b84-c8a6-4f04-b84f-6b505f8f5cce/metadata.json`を実読し、768p(`width`1280×`height`768、`crop_output`1280×720)・24クリップ×257フレーム＝総`5777`フレーム(出力`240.708`秒、約4分)・`chunked_upsample: true`・`vram_optimization.peak_vram_mb: 9780`(16GB内・スピルなし。机上推定10.4GBと整合)で完走したことを確認した(2026-07-14生成ジョブ、GPU: RTX 4070 Ti SUPER)。

台帳側は[`PENDING_TASKS.md`](PENDING_TASKS.md)へ新規まとめ項目§3-18を追加して上記を集約し、§2-1から「その他オーナー確認待ち(後日明言なし)」の行を除去した。これにより§2-1に残るのは「Chain V2VのcutoutRange併用」「短いクリップのoverlap_frames制約」の2項目のみとなった。§2-2の危険ゾーン(N4)項目には表示有無の確認込みである旨を追記した。[`REAL_BACKEND_CHECKLIST.md`](REAL_BACKEND_CHECKLIST.md) §4.3(128グリッド)・§4.6(5スロット/スタイルLoRA/短すぎる音声ブロック)・§4.8(409排他/unload再ロード)・§4.9(chunked_upsample 768pストレス)・第6節結果表・総合判定も合格へ整合させた。

## 32. 目視合格2件・APIキーバッジのスコープ外クローズ・mock運用方針の確定(2026-07-19最終判定)

§31に続けて同日中に、オーナーから残る目視確認・判断待ち項目についてさらに3件の最終判定があった。ドキュメントのみの反映で、コード変更は無し。

- **リロード後の再挿入(reuseIfPresent保険経路)**: 合格。WebUIリロード(またはAviUtl2再起動)でwebui側の記憶が消えた状態でも、完了ジョブの🎞️でnative側の`reuseIfPresent`がダウンロード済みファイルを再利用して挿入が成功することを確認した。通常経路(§26.2・§29の目視分)に続き保険経路も合格したことで、Insert再挿入まわりの目視確認はすべて完了した。
- **危険ゾーン(N4)**: 合格。①パイプライン解放(unload)→VRAM解放後の生成・モデル切替再開、ジョブ実行中の409(`JOB_BUSY`)インライン表示、②完了ジョブ一括削除(purge)→終了ジョブ削除と「対象なし／全件成功／部分・全失敗」の分岐表示、いずれもレイアウト・警告文言含めて確認した。§31で引き継いだ「危険ゾーンが見当たらない」というオーナー申告は記憶違いと判明し、実際には生成中のunloadでbusy表示が出る想定どおりの挙動だった。表示バグの疑いは解消した。
- **APIキーバッジ(N13)**: α版ではスコープ外としてクローズ。サーバーは`127.0.0.1`(ループバック)バインドで同一PC以外からアクセス不能なローカル前提ツールであり、APIキー検証の実益がないというオーナー判断による。機能自体は実装済み・無害のため撤去はしない。LAN公開(バインド変更)する時に再訪する。
- **mock固定運用の判断確定**: `Nz-Videomni/config.yaml`の`model.backend`は`"auto"`のまま維持する(人間ユーザーのデフォルト＝`real`。摩擦ゼロが理由)。あわせて、AIエージェント向けの運用規律を確定した。**AIエージェントがバックエンドを起動する場合(バックエンド改修セッション等)は、`config.yaml`を書き換えず`--config <一時yaml>`方式(バックエンド自動テストと同じ流儀: 一時フォルダに`model.backend: mock`を書いたyamlを生成して渡す)でmock起動する。realでの起動・生成はオーナーの明示指示がある場合のみ行う。** `--backend` CLIフラグの追加は不採用(オーナー判断: 既存手段〔`--config`方式〕で十分であり過剰装備)。バックエンドは凍結方針のため、この不採用判断についても新規チケットは起票しない。

台帳側は[`PENDING_TASKS.md`](PENDING_TASKS.md)へ新規まとめ項目§3-19を追加して上記を集約した。§2-2からリロード後の再挿入・APIキーバッジ・危険ゾーンの3項目を除去し、残る目視確認は「プリフィル経由のタブ状態保持」「`.aul2`日本語フォールバック文言」の2項目のみとなった。§2-3からmock固定運用の判断待ちを除去し、残るのは「空カードのGenerate最終防壁の是非」1項目のみとなった。[`REAL_BACKEND_CHECKLIST.md`](REAL_BACKEND_CHECKLIST.md) §4.7(reuseIfPresent保険経路)・§4.8(危険ゾーン表示)も合格へ整合させた。

## 33. タイムライン右クリックの現状調査と再設計ディスカッション開始(2026-07-19)

§7で配線した右クリック生成機能について、v2.0.54実機での検証消化判定とコード調査を行った。あわせて、UI/UXの観点からの再設計ディスカッションが始まった。ドキュメントのみの反映で、コード変更は無し。

### 33.1 実機テスト結果(オーナーによるv2.0.54実機操作)

- タイムラインのオブジェクト右クリックで「Nz-Videomni」サブメニュー7項目が表示され、「Send to generation panel」「Video from this image」のクリックでイベントがwebuiに到達し、パネルのプリフィル(タブ切替・解像度書き換え・スピル警告点灯・configの再取得)が動作することを確認した。これにより`register_object_menu`(または`_param`版)の登録・発火はv2.0.54実機で検証済みとなった。
- レイヤー右クリック(何もない場所)でも3項目の表示を確認した(`register_layer_menu`の登録は検証済み。項目クリックの発火自体は同一機構のため低リスクだが、明示テストはしていない)。
- pngやwavのオブジェクトでも同じ7項目が表示されることを確認した。これはSDKの制約(`register_object_menu`にオブジェクト種別フィルタ引数が存在しない。`plugin2.h:810-813`)による仕様どおりの挙動と判明した。
- `get_media_info`の返り値の決定的検証は未了。今回のテストでは選択画像の解像度がパネルに反映されたように見えるが、`deriveGenerationParams`にはプロジェクト解像度フォールバックがあるため、「プロジェクト解像度と異なるサイズのpngを右クリックして幅・高さ欄がpng実寸に一致するか」という1分程度の確認で決着する(台帳`PENDING_TASKS.md` §1-1に残置)。
- `register_event_listener`はネイティブコードのどこからも呼ばれていないことがコード調査で確定した(grep結果ゼロ)。よって実機検証の対象から除外する。
- `register_project_load_handler`の発火、`ReplaceObjectEditProc`のアンドゥ挙動(delete+createが1ステップにまとまるか)、`UpdateObjectTextEditProc`のカーソル依存書込、`CutoutRangeWorker`/`ExtractAudioWorker`の実機検証は未了のまま。ただし33.3のオーナー決定により、これらが必要になる「仮オブジェクトの完了時自動置換」がβ版以降へ先送りされたため、検証時期もその実装時へ先送りする。

### 33.2 コード調査で判明した既知の食い違い

- 右クリックの全10アクションは現状「タブ切替＋幅・高さ欄のプリフィルのみ」の骨組み実装。素材ファイルパスの自動投入はどのアクションも未実装(`filePath`は取得後に未使用のまま)。Chain行き4アクションとLibrary行きRegenerateは、画面に「次段で対応」のノートを出すのみ。
- 「Video from audio(Aud2Vid)」のルーティング先がChain画面のままになっている(`webui/src/timeline/menuRouting.ts`の`audioToVideo`エントリ、`targetMode: "chain"`)。単発A2Vは2026-07-15にCreate画面へ移設済み(§10.2、`PENDING_TASKS.md` §3-3)のため、移設前の取り残しと判明した。再設計で解消予定。
- 「Video from this image」実行時にKEYFRAMESカードが追加されないのは未実装のため(不具合ではない)。解像度プリフィルによりスピル警告(Generation slows down 2-4x…)が点灯することがあり、これが今回の操作で唯一の目立つフィードバックになっている。
- config再取得は、§7.1で採用した「keyによる再マウント＋lazyな`useState`初期化」方式(パネルをkey付きで再マウントして一度だけプリフィルを適用する)の副作用にすぎない。パネルコンポーネントが初期化時に持つ通常のconfig取得処理が再マウントのたびに再実行されているだけで、右クリック専用の特別な再取得ロジックがあるわけではない。

### 33.3 論点A: 仮オブジェクトの完了時自動置換のβ送り(オーナー決定)

タイムライン右クリック再設計ディスカッションの論点Aとして、仮オブジェクトの完了時自動置換(`webui/src/timeline/provisionalFlow.ts`の`insertProvisional`→`resolveProvisional`配線、§7.7で次段送りにしていた項目)についてオーナーが判断した。

- **決定**: 完了時自動置換は「将来の研究課題」へ優先度を下げる。α版は実装済みのシンプルな🎞ボタンによる手動挿入(実機で動作確認済み)に留め、自動置換はβ版以降の課題とする。
- これに伴い、33.1で列挙した`register_project_load_handler`・アンドゥ挙動・`CutoutRangeWorker`/`ExtractAudioWorker`の実機検証も、実装時期に合わせてβ送りとなった。

台帳側は`PENDING_TASKS.md`「4. 将来の研究課題」に新規項目(仮オブジェクト完了時自動置換のβ送り)を追加し、上記の先送り検証項目をまとめて畳み込んだ。

### 33.4 タイムライン右クリックUI/UXの再設計方針(議論継続中)

オーナーから、タイムライン右クリック機能をUI/UXの観点から再設計する方針が提示された。以下3方針が軸になっている。

1. 操作はすべて操作パネル(Create/Chain/Library)を経由する。
2. 生成の状態管理はパネルに集約する。
3. 仮オブジェクトは編集時のヒントに徹する。

メニュー項目の絞り込み、素材の自動読み込み(第一段階は動画/画像/音声/テキストのファイルパス直載せ、範囲切り出しは第二段階)などの具体設計はディスカッション継続中で、まだ確定していない。次段の実装着手前に、あらためて確定方針をDEVLOGへ記録する。

### 33.5 台帳への反映

`PENDING_TASKS.md`を以下のとおり更新した。

- §1-1: 実機検証消化分を§3-20として新規記録し、§1-1自体のスコープを`get_media_info`の決定的確認1点へ絞り込んだ。
- §1-2: 再設計方針の議論開始と3方針の一言追記、②仮オブジェクト未配線の注記を「4. 将来の研究課題」の新項目へ整合。
- §4: 仮オブジェクト完了時自動置換のβ送り項目を新規追加。

### 33.6 論点C: 音声送り機能はmix経路に限定、solo分離は将来課題へ(オーナー決定)

タイムライン右クリック再設計ディスカッションの論点C(「動画の音声をa2v(音声から動画生成)に送る」機能の実現方式)について、オーナーが判断した。

- **決定**: 音声抽出は、ネイティブ側の音声抽出RPC(`timeline.extractAudio`)の実機検証済みであるmix経路(タイムライン全体で鳴っている音をそのまま録る)だけで実現すると割り切る。不要な音を除きたい場合は、ユーザーが手作業で対象外レイヤーを無効化すればよい(AviUtl2ではレイヤー無効化の操作が簡単なため)。
- これに伴い、solo分離(対象レイヤー以外を自動無効化して単独の音声を抽出する機能。実機未検証・コミュニティ参考実装もないリスクパス)は「将来の研究課題」へ優先度を下げる。
- あわせて、素材自動読み込みの第一段階の対応種別を「動画/画像/音声/テキスト」とする方針を確認した(§33.4からの継続。テキストはプロンプト欄への流し込みを想定)。

### 33.7 調査で判明した事実

論点Cの検討にあたり、以下の事実がコード調査で判明した。

- バックエンドの音声アップロードAPIは拡張子リスト(`.wav`/`.mp3`/`.m4a`/`.aac`/`.flac`/`.ogg`)で判定しており、mp4等の動画コンテナは400エラーで拒否される。エンジン内部(PyAV/ffprobe)はコンテナ形式非依存で動画ファイルの音声トラックも読める汎用実装になっているが、バックエンド凍結方針のため入口(API)は変更しない。したがって、右クリックからのa2v送りはネイティブ側の音声抽出(wav化)で迂回する。
- ネイティブの音声抽出(`ExtractAudioWorker`)、webui側のオーケストレーター(`extractAudioAndUpload`。テスト済みだが未配線)、Create画面のA2V受け皿(wav長からFramesを自動調整する処理を含む)は、いずれも既存部品であることを確認した。新規実装が必要なのは配線部分(ルーティング修正・範囲変換・状態橋渡し・A2Vの初期有効化・エラー表示)のみで、抽出・アップロード・受け取りのロジック自体は作り直さなくてよい。
- `webui/src/timeline/menuRouting.ts`の`audioToVideo`が依然Chain画面行きになっている件(§33.2で既知の食い違いとして記録済み)について、Chain画面側のコードを調査したところ、音声関連のコードは一切残っておらず完全な配線切れであることを確認した。単発A2Vの2026-07-15のCreate画面移設(§10.2)で置き去りになったまま。
- テキストオブジェクトの本文取得は、SDKの`get_object_item_value`(effect名・item名とも`"テキスト"`)で実現できる見込みが高い。書き込み側の対となる`set_object_item_value`は同じキー名で本番コード(仮オブジェクトの失敗テキスト書き換え)にすでに稼働実績がある。webui側にもプロンプト流し込みの拡張ポイントが`AppShell.tsx`にコメントで用意済み。
- AviUtl2 SDK最新版(2026/7/18版、v2.1.1)を公式サイトから取得し全ファイルを突合した結果、メニュー登録API(`register_object_menu`等)への種別フィルタ引数の追加は無かった(§33.1で確認した制約は健在)。新規追加は`register_event_listener`の`CHANGE_FOCUS_OBJECT`イベント(選択変更検知)と、`set_focus_object(nullptr)`による選択解除の2点のみ。あわせて、手元のSDKミラーが2026/7/5版で2週間古いことも判明した(差分は`plugin2.h`の上記2箇所のみで、既存コードへの影響は無い。ミラー更新はどこかのタイミングで実施する価値がある)。

### 33.8 議論継続中の項目(まだ決定ではない)

- メニュー項目の命名規則案「種別: やること」(例: `"Video: extend this video (send to v2v)"`)をオーナーが提示した。絵文字の使用可否は調査中。
- メニュー内容一覧のブレインストーミング、競合ツール(Adobe・DaVinci Resolve・LTX desktop等)のUX調査を進行中。

### 33.9 台帳への反映(追補)

`PENDING_TASKS.md`を以下のとおり更新した。

- §4: solo分離(音声抽出のレイヤー単独化)の優先度格下げ項目を新規追加(§4-9)。
- §1-1: SDKミラーが2週間古い件を軽い参考情報として追記(対応の緊急度は低い)。

### 33.10 絵文字テストの実施と実機結果・採用決定

§33.8で「調査中」としていたメニュー項目の絵文字使用可否について、実機テストを行い決着した。

- コミット7e9e9adで、`native/src/plugin.cpp`の`kObjectMenuItems`(7項目)・`kLayerMenuItems`(3項目)のname_key先頭に、試験的に絵文字10個(🎬🖼♻🔁📤🎵✨🧩📸等)を追加した。plugin.cppのASCII限定規約を守るため、絵文字はユニバーサル文字名エスケープ(`\U....`)で埋め込み、生の非ASCIIバイトは混入させていない(検査で0バイト確認済み)。BMP内記号(♻ U+267B・✨ U+2728)とBMP外絵文字(🎬🖼🎵等)を意図的に混在させ、描画差の有無も合わせて観察した。アクションIDと配線は変更していない。
- v2.0.54へデプロイし、オーナーが実機で目視確認した。結果、全10個の絵文字が欠けなく表示された。表示はモノクロだった(Windows標準のメニュー描画がカラー絵文字フォントに対応していないため)。BMP内記号とBMP外絵文字とで表示差は見られなかった。
- **オーナー判定: 採用決定。** モノクロ表示でも、メニュー項目の視認性向上には十分と判断された。
- 技術ノート: この表示は、Windows標準のフォントフォールバック(Segoe UI Emojiのモノクロ輪郭グリフ)によるものであり、AviUtl2が動くWindows 10/11環境であれば、ほぼ全ユーザーで同一の見え方になる見込み。今後メニューに絵文字を追加する際は、2020年以降に新しく追加された絵文字は避ける方針とする(古いWindows環境でグリフが未収録で欠落するリスクを避けるため)。

### 33.11 ブレインストーミングの決定事項(2026-07-19)

タイムライン右クリック再設計のブレインストーミングで、以下が決定した。

- **右クリックメニューに載せる機能の判定原理**を確立した。「その機能の入力が、タイムライン上に存在するか」を基準とする。
  - 入力が素材(画像・動画・音声・テキストがタイムライン上にある)の場合は、右クリック適性が高い。ファイルピッカーで探し直す手間をまるごと消せるため。
  - 入力が位置・範囲(カーソル位置、隙間、選択範囲)の場合は、適性が最も高い。タイムラインでしか表現できない文脈であり、操作パネル側では代替できない(「ここにAI生成を挿入」「隙間を埋める」がこれにあたる)。
  - 入力が設定・フォルダ・履歴(バッチ処理対象のフォルダ、モデル選択、プリセット)の場合は、適性がない。タイムラインに文脈が存在せず、操作パネルや設定画面の領分である。
  - この原理により、バッチa2v(音声フォルダの一括処理)を右クリックに載せないこと、モデル管理・プリセットを右クリックに載せないことが説明できる。詳細は`TIMELINE_FEATURE_CANDIDATES.md`にも追記した。
- メニュー項目の命名規則を「絵文字＋『種別: やること』」形式で確定する方向とした(例: 「🎬 Video: この動画の続きを生成」)。§33.8で提示していた命名規則案に、33.10で採用が決まった絵文字を組み合わせる形。
- 「Image: 最後のクリップのキーフレームに追加」案は廃案とした。現状の操作パネルはClip Chain(複数クリップの長尺生成)の途中クリップにキーフレームを適用できず、バックエンドもその前提の設計になっていないため。
- チェーン投入系(テキストや画像を構築中のチェーンの特定クリップへ送る導線)は第2段階へ先送りする。「どのクリップに入れるかの指定方法」ごと設計が必要なため。
- 第1段階の画像系メニューに「🖼 Image: この画像から長尺動画を生成 (i2v Clip Chain)」を追加する方向とした(i2vからClip Chainの冒頭キーフレームへ送る経路。オーナー提案)。
- 隙間埋め(Fill the gap)は、LTX Desktopの類似機能(Fill With Video)の調査結果をドキュメントに残したうえで、第2段階または将来の研究課題へ送る方針とした。オーナーが挙げた設計論点は次の4点。
  1. 隙間が広すぎるときの生成尺(DURATION)の扱い。
  2. 画像→画像の隙間は、既存のキーフレーム機能の使い回しで比較的容易に埋められる。
  3. ただし、LTX 2.3はキーフレームを置ける位置が8n+1グリッドに固定されるため、任意長の隙間を正確には埋められない。
  4. 最難関は動画→動画の隙間で、現行バックエンドにこれを埋める機能がない。「動画→画像」「画像→動画」のケースも別途考慮が必要。
  - 調査自体は別エージェントが実施中であり、ここには方針と論点のみを記録する。

### 33.12 Chain V2Vのセマンティクス調査(裏取り中・保留)

Chain V2V(ソース動画を指定してのチェーン生成)が「動画の変換」なのか「動画の続きを生成」なのか、認識に食い違いがあることが分かり、バックエンドコードの裏取り調査を開始した。結果が出るまで、右クリックメニューの「Video: 続きを生成」項目(A1相当)の実装方式は保留とする。

**追記(2026-07-19、同日中に決着)**: 裏取り調査が完了し、「動画の続きを生成」説で確定した。根拠は以下のとおり。

- `source_video`の末尾`context_frames`(接続に使う末尾フレーム数)だけをVAE(動画を圧縮表現〈latent〉に変換するモデル部品)でエンコードし、それをチェーンの1番目のクリップの先頭latentとして凍結する処理になっている。これはクリップとクリップの間をつなぐ既存のoverlap(重なり区間)の仕組みと同一の機構であり、V2V専用の別処理ではない。
- 出力されるmp4にソース動画の区間そのものは含まれない。生成されるのは「続きの部分」のみ。
- 元動画と生成した続きを1本のファイルに合体させるのは、join(V2V結合)という別の後続機能の役割になっている。
- 上記はバックエンドの`backend/engine/pipeline/chain_pipeline.py`等のコードで確認した。

これにより、右クリックメニュー案「🎬 Video: この動画の続きを生成 (v2v)」は看板どおりChain V2V送りで実装してよいことが確定し、実装方式の保留は解消した。

### 33.13 Fill the gap(タイムライン隙間埋め)先行調査の完了

§33.11で第2段階または将来の研究課題へ送ることにしたFill the gap(タイムラインの隙間をAI生成動画で埋める機能)について、競合ツールLTX Desktopのソースコード読解とWEB調査による先行調査が完了した。

- LTX Desktopの「Fill With Video」は想定より浅い実装だった。境界フレーム(隙間の両隣のクリップの端フレーム)はGemini(視覚言語モデル)へ渡して橋渡しプロンプトを自動生成させる材料にしか使われておらず、実際の動画生成への画像条件付けには使われていない。実際の生成は原則プロンプト駆動のT2V(テキストから動画生成)で、UIにあるStart/End frameトグルもソース上は生成呼び出しに配線されていない未使用UIだった。
- 尺の扱いは「隙間以上になる最小の離散候補値へ切り上げ→8n+1グリッドへスナップ→生成後は先頭を隙間長ぴったりで使いヘッドトリム」という方式で、隙間が最大候補値を超える場合は生成ボタンを無効化する割り切りだった。
- 本プラグインは、片側キーフレームi2v(画像から動画生成)を使えば現行バックエンド無改修でLTX Desktop実装を上回る接続品質を狙える見込み。動画→動画の両側接続はLTX Desktopにも前例が無い未解決課題として残る。
- 調査結果の詳細は[`GAP_FILL_RESEARCH.md`](GAP_FILL_RESEARCH.md)に記録した。台帳側は`PENDING_TASKS.md`「4. 将来の研究課題」に§4-10として新規追加した。

## 34. 右クリック再設計 Phase A実装とG1実機ゲート(2026-07-19)

§33で確定した[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md)(第3版)に基づき、ワークオーダー(増分I1〜I14。本節で実装したのはこのうちPhase A分の増分I1〜I4で、I5以降は§35〜§40を参照)を承認・実装した(Phase A)。あわせてオーナーがAviUtl2実機(v2.0.54ポータブル構成)でG1実機ゲートを行い、2件の判明事実と3件の決定が生まれた。

### 34.1 Phase A実装完了(増分I1〜I4。ワークオーダー全体はI1〜I14で、I5以降は§35〜§40参照)

以下の増分を実装した。全増分でtypecheck/vitest/doctestが緑であることを確認しながら進めた(最終状態: vitest 896件pass、doctest 236件pass)。

- **I1 `getSelection`拡張**: `timeline.getSelection`の応答に、テキストオブジェクトの本文(`textContent`)・実尺秒数(`mediaDurationSec`)・テキスト種別判定を追加した(§4・§4-4・§4-5で設計した内容の実装)。
- **I2 メニュー8+2項目の差し替え**: `native/src/plugin.cpp`の`kObjectMenuItems`(8項目)・`kLayerMenuItems`(2項目)を、絵文字つき・日本語訳ありの新項目一式へ差し替えた。旧`regenerate`/`replace`/`sendToChain`/`fillGap`を廃止し、`audioToVideo`のルーティング先をChain画面からCreate画面へ修正した(§7-1・§33.2・§33.7で既知としていた食い違いの解消)。
- **I3 仮オブジェクト系RPC**: `insertProvisional`を段階移行し、新RPC `timeline.updateProvisionalReservation`(§5-3)を実装した。配置系統(A)素材の直後／(B)素材と同開始で`layer_max+1`／(C)右クリック位置(§5-9)と、位置衝突時の`layer_max+1`フォールバック(§5-4)を実装した。
- **I4 ✨の最小配線**: `webui/src/timeline/provisionalReservation.ts`モジュールを新設し、予約席の追跡(§5-6の予約席1つルール)と、G1検証用の仮設トースト通知を実装した。

実装はオーナーがコミット・プッシュし、デプロイ済み。

### 34.2 G1実機ゲート結果(オーナー実機確認)

- **合格した項目**:
  - アンドゥ1件化: ✨挿入のCtrl+Zが1回でまとまる、予約移動のCtrl+Zも1回でまとまることを確認した(RIGHTCLICK_REDESIGN_SPEC §8 #3)。
  - `get_media_info`の実寸確認: プロジェクト解像度と異なるサイズのpngを右クリックし、パネルの幅・高さ欄がpng実寸に一致することを確認した(§8 #4)。これにより`PENDING_TASKS.md` §1-1の残置チェックが消化された。
  - テキスト本文の取得: テキストオブジェクトの本文取得を確認した(§8 #5)。
  - 予約席の一意性・移動・Generate後の位置固定: §5-6の予約席1つルールどおりに、予約席が常に1つに保たれ、移動し、Generate後は位置が固定されることを確認した。
  - レイヤー番号の前面/背面の向き: レイヤー番号が大きいほうが前面であることを確認した(§8 #6)。

### 34.3 判明1(D1): `EDIT_INFO`カーソルは赤カーソル(プレビュー表示フレーム)を返す

策定時の見込み(RIGHTCLICK_REDESIGN_SPEC §5-1・§5-9(C)・§8 #1)は「`EDIT_INFO.frame`は、右クリックで移動する灰色カーソルの位置と一致する見込み」だったが、実機確認の結果、実際に返るのは**赤カーソル(プレビュー表示フレーム)**であり、見込みは外れた。

- **対応方針**: `get_mouse_layer_frame`(最後のマウス移動メッセージ由来の位置を返すAPI)で上書きするテスト配線を実験中。Win32の一般論として、ポップアップメニュー表示中は親ウィンドウにマウス移動メッセージが届かず、右クリックした地点で位置が凍結される見込みがあるため、成功見込みは高いと判断している。
- **退路(承認済み)**: 上記が実機で不成立と判明した場合は、メニュー名を「Insert AI generation at current frame」(現在フレームにAI生成を挿入)へ変更し、赤カーソル(現在フレーム)位置への挿入として仕様を確定する。
- 台帳側の反映は§34.6にまとめる。仕様書側の反映は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md)第4版(§5-1追記・§5-9(C)注記・§8 #1注記)で完了済み。

### 34.4 判明2(D2): 空き不足時に仮オブジェクトが黙って短縮される件

空き不足の場所で✨すると、`layer_max+1`フォールバック(§5-4)が発動せず、代わりに仮オブジェクトが黙って短縮されて成功してしまう不具合が実機で見つかった。

- **原因**: `create_object_from_alias`の長さ引数に**リテラル0**を渡していたため、SDK仕様「長さに0を指定した場合は、長さや追加位置が自動調整される」が発動し、意図しない短縮・自動配置が起きていた。
- **修正方針(承認済み・実装中)**: 挿入前に`find_object`で空き長をスキャンし、不足していれば直接`layer_max+1`へ挿入する。あわせて、長さ引数は常に明示値を渡す(リテラル0を渡さない)。

### 34.5 実機の制約として判明した事実: 既存オブジェクト上ではレイヤーメニューが出ない

既存オブジェクトの真上で右クリックすると、必ずオブジェクト選択(オブジェクトメニュー)になり、レイヤーメニュー自体が表示されないことが実機で判明した。したがって、「カーソル位置が埋まっている」という衝突は、実質的に「(レイヤーメニューが出る空き位置において)後方の空きが不足している」という形でのみ起こる。§5-4のフォールバック設計自体への影響はない。

### 34.6 オーナー決定(承認済み・実装中)

- **D3: #9(✨)・#10(📸)のリマウント廃止**: フォーム初期化(リマウント)をやめ、パネル状態を保持する。予約の長さは、config既定値ではなく、その時点のパネルDURATION実値を使う。理由は、ユーザーが設定した解像度等の値が✨のたびに初期値へ戻ってしまう手間を排除するため。素材系#1〜#7のリマウントはこれまでどおり維持する。仕様書側は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §3-6(#9/#10のリマウント欄)・§5-1(長さの記述)を第4版として更新済み。
- **表示規約の新設**: ユーザー向け通知に出るレイヤー・フレーム番号は、すべて+1(1始まり。AviUtl2本体のUI表記と一致)で表示する。SDK内部は0始まりのまま。仕様書§6-3として新設済み。
- **α版で不採用と決めたもの**: 低レベルマウスフック等、SDK外のハックによる右クリック位置取得。リスク過大と判断し、採用しない。

### 34.7 Phase Aで想定どおり未実装と確認された項目(誤解防止のための記録)

G1実機ゲートで、以下は「実装漏れ」ではなく、Phase Aの計画どおりの中間状態であることを確認した。

- 🎞の自動置き換えなし(α方針どおり。§33.3のβ送り決定を継続)。
- 📝/i2v/v2v/IC-LoRAの素材自動読み込みは未配線(Phase Cで予定)。
- ✨以外の項目への仮オブジェクト配線は未配線(I7で予定)。

### 34.8 台帳への反映

`PENDING_TASKS.md`を以下のとおり更新した。

- §1-1: `get_media_info`の1分チェックがG1で消化されたため、§1-1をクローズし「3. 実装済み」へ§3-21として新規移動した。
- §1-2: 右クリック機能の実装がPhase Aまで進行中であることを1行追記した。

---

## 35. Phase B実装とG2実機ゲート(2026-07-19深夜〜07-20)

§34で完了したPhase A(I1〜I4)に続き、Phase B(I5〜I7)を実装した。あわせてオーナーがAviUtl2実機でG2実機ゲートを行い、合格判定・重要な仕様知見2件・不具合1件(6-b)が生まれた。G2はクローズ。

### 35.1 Phase B実装完了(ワークオーダーI5〜I7)

- **I5 共通ノート領域(NoteArea)**: [`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) 第6節で設計した共通ノート領域を本設した。受領ノート・ミスマッチ案内・フォールバック通知等を1枠へ集約する仕組みを実装し、G1で使っていた仮設トースト通知(§34.1のI4)を撤去した。
- **I6 種別判定/ミスマッチ/複数選択/長さガード**: §4で設計した種別判定・ミスマッチ案内・複数選択ガード・長さガードを実装した。仮オブジェクトへメタデータを付与する処理もあわせて実装した。長さガードのしきい値は次のとおり確定した。
  - v2v(#1、この動画の続きを生成)のしきい値: `context_frames`既定値73フレーム(約3.04秒)。
  - IC-LoRA(#2、この動画を参照にIC-LoRA生成)のしきい値: 保守的な仮置きとして約1.04秒。バックエンドの正確な最小尺の裏取りはTODOとして残す。
  - `mediaDurationSec`が0(尺不明)のケースは長さガードでは弾かずに通し、最終的にサーバー側の422エラーを判定として使う方針にした。
- **I7 routing刷新**: `handleRoute`を6段の固定順処理へ再設計し、予約席ビジーガード(§5-6)を実装した。あわせて`scanProvisionals`による起動時の予約再構築を実装した(策定時に「β版で検討」としていた掃除機構〔RIGHTCLICK_REDESIGN_SPEC §5-7〕の一部を前倒しで実装したもの)。

### 35.2 G1後クリーンアップ

Phase B着手前に、G1仮設実装の残骸を撤去した。

- `webui/src/timeline/provisionalFlow.ts`(§7.7・§33.1・§33.3以来「配線未接続」と記録していたテストコード専用モジュール、`PENDING_TASKS.md` §4-8で自動置換のβ送りとともに保留していたもの)を撤去した。
- `insertProvisional`のレガシー3フィールド(I3で段階移行していたもの、§34.1参照)を撤去した。

### 35.3 テスト基準

vitest 967件pass、doctest 236件pass(§34.1時点の896件pass/236件passから増加)。

### 35.4 G2実機ゲート結果(オーナー確認、G2はクローズ)

- **合格した項目**:
  - 配置系統A(画像の直後へ予約が移動、Ctrl+Z1回でまとまる)。
  - 配置系統B(IC-LoRA・動画音声a2v・音声a2vとも素材と同開始・最前面)。
  - ビジーガード(生成中の右クリックは案内のみで何も起きない、完了後は予約可能)。
  - ミスマッチ案内(タブも動かない)。
  - #5(キーフレーム追加)はCreateタブ切替のみ、#8(テキスト追記)は無反応(いずれもI11で予定している挙動どおりで、想定内)。

### 35.5 観察1(仕様として明文化): 予約移動でのID再発行

予約の「移動」は実装上「削除＋新規作成」であり、仮オブジェクトのID(`pending-*`)は移動のたびに新しく発行される。オーナーが実機で確認し、この挙動を現状維持のまま正式仕様として確定した。[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §5-3に正式挙動として明記した(第5版)。

### 35.6 観察2(重要な仕様知見): 「作業中のオブジェクト」

AviUtl2では、複数オブジェクトを選択していても「作業中のオブジェクト」(右パネルに詳細が表示され、タイムライン上で破線リボンになる、SDKの`get_focus_object`に相当するもの)が常に1つ存在し、右クリックメニューの対象はこの作業中オブジェクトになることが実機で判明した。ネイティブ実装はフォーカスがあればそれ1件だけを`timeline.getSelection`で返す構造のため、[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §4-2で懸念していた複数選択ミスマッチは実質発生しない。同節は「フォーカス不在時の防御網」という位置づけへ改めた(第5版)。

### 35.7 不具合6-b: 仮オブジェクトを残した.aup2の開き直しで予約席の再構築が効かない

仮オブジェクトを残したまま`.aup2`プロジェクトを保存し、後で開き直すと、§35.1のI7で実装した`scanProvisionals`による再構築が効かないことがG2で見つかった。予約が残存していると、次の生成起点の右クリックが移動ではなく新規挿入になってしまい、また生成中の予約が残っている場合はビジーガードが働かない。

- **原因確定**: webui側に`timeline.projectLoaded`イベントの購読が存在せず、`scanProvisionals`による再構築はマウント時に1回しか走らない。この1回はプロジェクト読込より前に空振りしてしまうため、開き直し後の状態を拾えない。
- **修正方針(オーナー承認済み)**: 次の2点をPhase Cで実施する。
  1. `timeline.projectLoaded`イベントを購読し、発火のたびに`scanProvisionals`による再構築(reconcile)を再実行する。
  2. 右クリック時にもオンデマンドで再構築を行う保険を追加する(①が何らかの理由で発火しなかった場合の保険)。
- **優先度と実施タイミング(オーナー決定)**: 優先度は低いためPhase C内で実施する。理由は、この不具合が他機能と干渉しないこと、また実害が「生成中にAviUtl2を閉じた場合の救済」に限られること、そしてエクスプローラーから完成ファイルを手動挿入するフォールバックが常に存在することの3点。

### 35.8 SDK裏取り

`register_project_load_handler`は、SDK上「プロジェクトロード直後・初期化時にも呼ばれる」と明記されている。更新履歴には発火漏れのバグ修正が3件(起動時初期化・引数起動・編集レジューム)記録されており、これらの経緯からも広範な発火が意図された設計であることがうかがえる。ただし、**手動でのファイル→開くによるオープン時の発火は実機未検証**であり、Phase Cで確認する([`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) 第8節に確認項目として追加した)。

また、予約検出の名前フォールバックが使う`object_name`は、AviUtl2本体の正式なUI機能(ユーザーが変更・消去可能)であり、`.aup2`保存時に永続化されることはほぼ確実と考えられるが、一次情報による確証はまだない。予約検出はテキスト本文の`[#id]`マーカーを第一手段、`object_name`による名前フォールバックを二重化する設計にしてあるため、この不確実性による実害は小さい([`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) 第8節に確認項目として追加した)。

### 35.9 台帳への反映

`PENDING_TASKS.md` §1-2を「Phase B完了・G2クローズ・Phase C進行中」へ更新した。6-b修正がPhase C枠であることは同節内の記述で足りるため、新規チケットは起票しない。

---

## 36. Phase C/D夜間実装バッチ(2026-07-20未明)

オーナー就寝中の自走分として、Phase C（右クリック再設計ワークオーダーI8〜I13）とPhase D相当の関連修正を一括で実装した。すべて実装完了・自動テスト全緑・**未コミット・未デプロイ・オーナー目視未実施**。§35.7で合意した6-b修正もこのバッチに含めて実施した。

### 36.1 不具合6-b修正（Phase C枠）

§35.7で確定した方針どおり、2点を実装した。

- `timeline.projectLoaded`イベントの購読を新設した。イベントを受信するたびに予約席（仮オブジェクトの追跡状態）を再構築する。
- 右クリック時のオンデマンド再構築を追加した。`handleRoute`（右クリックのルーティング処理）のビジーガード直前で、毎回タイムライン実体を走査して予約席の状態を作り直す。

イベント購読だけに頼るとプロジェクト読込タイミングの取りこぼしリスクが残るため、右クリック時の走査を保険として二重化した設計にした。イベント発火自体が実機未検証（§35.8参照）でも、保険側の走査だけで正しく動く構成になっている。あわせて、モックの`scanProvisionals`（予約席再構築処理）をライブのタイムライン状態を反映する形へ改良した。

### 36.2 I8 Chain系（#1続きを生成・#6長尺動画）

- **#1「続きを生成」**: 右クリックした動画のパスを、Chain画面のソース欄へ自動で読み込むようにした。
- **#6「長尺動画」**: 右クリックした画像を、Clip Chain（複数クリップを連結するChain編集の一種）の冒頭キーフレームへ自動設定するようにした。
- 上記2件について、「右クリックから読み込みました」という受領ノート（NoteArea領域に出る短い案内文）を新設した。また、素材が見つからない場合の欠損ノートも新設した。
- **Chain編集中の破棄確認ダイアログ**: `ConfirmDialog`を新設し、`useChainForm`（Chain画面のフォーム状態管理フック）に編集中フラグ`isDirty`を実装した。共有参照（ref）経由で外部へ公開し、破棄が起きる操作の前に確認を挟めるようにした。
- 旧来表示していた「素材はまだ自動で読み込まれません」というノートは、上記の自動読み込み実装により不要になったため表示を廃止した。

### 36.3 I9 Create系（#2 IC-LoRA・#4 i2v・#10 📸現在フレーム）

- **#2 IC-LoRA**: 右クリックした動画を参照動画欄へ自動読み込みするようにした。既存の128刻み丸め処理（解像度をIC-LoRAの制約に合わせて128の倍数へ丸める既存動作）はそのまま踏襲する。
- **#4 i2v**: 右クリックした画像をKEYFRAMES（キーフレームタイムライン）の1枚目へ自動追加し、あわせてI2Vバッジ（i2vモードであることを示す表示）を点灯させるようにした。
- **#10 📸現在フレーム取り込み**: §34.6のD3決定（#9/#10はリマウントしない）に対応するため、リマウントなしでも配線できるライブコマンドチャネル`createLiveCommands.ts`を新設した。既存のKEYFRAMESカードにある📸ボタンと同一のセマンティクス（現在フレームを取り込む処理）で取り込みを行う。取り込み成功時・失敗時それぞれのノートも新設した。

### 36.4 I10 音声系（#7 音声a2v・#3 動画音声a2v）

- **#7 音声a2v**: 右クリックした音声ファイルのパスをCreate画面へ直載せし、A2V（音声から動画を生成するモード）を有効化するようにした。wavファイルの場合は既存のFrames自動調整（wav長から必要フレーム数を逆算する処理）が働く。
- **#3 動画音声a2v**: 選択オブジェクトの実尺範囲（`frameEnd`を含む区間として`frameCount = end - start + 1`で算出）を使い、`timeline.extractAudio`（タイムラインから音声を抽出するネイティブRPC）を直接呼び出し、得られたwavパスを`attachSourceAudioByPath`へ渡す方式にした。既存の`extractAudioAndUpload`は二重アップロードになるため今回は使わない。
  - 抽出中・mix（タイムライン全体でミックスされた音を録る方式）の説明・無音検出・タイムアウト・汎用失敗、それぞれの文言分岐を実装した。ネイティブ側は`EXTRACT_FAILED`という単一コードしか返さないため、エラーメッセージの部分一致で原因を分類している。

### 36.5 I11 #5キーフレーム追記・#8プロンプト追記

- **#8 プロンプト追記**: ガード順序を「ビジーガード（生成中は追記もしない）→空本文ガード→改行1つ区切りで末尾追記→受領ノート（追記内容の先頭20字を表示）→placement Bへ予約」とした。タブ切替は行わない。
- **#5 キーフレーム追記**: `publishKeyframeCount`（キーフレーム枚数を公開する共有参照）を使い、追記直前の枚数を判定する。1枚目の追加なら「生成起点」として扱い、ビジーガード＋placement Aへの予約＋追記を行う。2枚目以降の追加では予約には触れず、追記のみを行う。上限（5枚）到達時のノートも新設した。

### 36.6 I12 仮オブジェクトの4段階状態遷移

仮オブジェクトの表示テキストを、生成の進行に応じて4段階で書き換える仕組みを実装した。

- ネイティブ側（`native/src/bridge.cpp`想定）を最小拡張し、`insertProvisional`・`updateProvisionalReservation`へオプショナル引数`textPrefix`（表示テキストの接頭辞）を追加した。`BuildProvisionalTextAlias`（表示テキストを組み立てる内部処理）の接頭辞部分を可変化した。`[#id]`タグと`object_name`（AviUtl2のオブジェクト名機能）の二重タグ付けは従来どおり維持している。
- 4段階の遷移は次のとおり。
  1. 挿入直後: 「⏳生成予約：（この位置に生成されます）」
  2. Generate受理時: 「⏳生成中：〔プロンプト先頭〕…」
  3. 生成失敗時: 「❌失敗：〔理由〕」
  4. 生成完了時: 「✅完了：🎞ボタンで挿入できます（自動では置き換わりません）」
  - 段階3・4への遷移は`JobsContext`（ジョブの進捗状態を管理するコンテキスト）が`updateProvisionalText`経由で行う。書き換え時も`[#id]`を再付与し、再発見性（予約席をID経由で再度見つけられる性質）を維持している。
- あわせて、これまでCreate画面側にしか配線していなかった`bindToJob`（ジョブと仮オブジェクトを紐付ける処理）を、ChainScreenのGenerateにも配線した。従来Chain経由の生成では4段階遷移が効かない穴があったため、これを解消した。

### 36.7 I13 ✅完了仮オブジェクトの自動削除

- 新設RPC `timeline.deleteProvisionalByJob`を実装した。jobIdが一致する仮オブジェクトのうち最初の1件のみを削除し、冪等（同じjobIdで複数回呼んでも安全）で、削除結果（`deleted`）を返却する。
- `downloadAndInsert.ts`の`insertMedia`（🎞ボタンによる挿入処理）成功直後に、上記RPCをベストエフォートで呼び出すようにした。JobCard（ジョブ台帳のカード）・Create結果パネルの両方の挿入導線に効く。
- ❌（失敗）マーカーは対象外とし、自動削除しない。

### 36.8 テスト基準の推移

- vitest: 971件pass→1023件pass（10件skip）
- doctest: 236件pass→244件pass（6件skip）

上記いずれの増分でも`npm run typecheck`エラー0・埋め込みフルビルド成功を確認している。

### 36.9 残作業

オーナー起床後に想定している流れは次のとおり。

1. デプロイ。
2. Phase C/D一括分の実機目視確認。
3. I14（仕上げ）の実施: 言語ファイルの最終整合、`menuSourceNote`等の残骸整理、LibraryScreenの`regenerateNote`デッドブランチの整理、関連文書の整合確認。

### 36.10 台帳への反映

`PENDING_TASKS.md` §1-2の状態行を「Phase C/D実装完了・オーナー実機確認待ち（デプロイ含む）」へ更新した。個々の増分（6-b・I8〜I13）の内容は本節（§36）を出典として参照する形にとどめ、§1-2内での逐条記載はしない。

## 37. Phase C/D実機目視結果と残修正(2026-07-20朝)

§36で実装したPhase C/D夜間実装バッチ（6-b修正、ワークオーダーI8〜I13）を、オーナーがAviUtl2実機で目視確認した。大半は合格したが、条件付きで要修正の2件と、設計変更が必要な不合格1件が見つかった。あわせて、修正不要と判断された軽微バグ1件の記録、および将来の研究課題1件の追加がある。

### 37.1 合格項目

以下はすべて合格した。

- **素材自動読み込み4種**: v2vソース・IC-LoRA（In-Context LoRA）参照・i2vキーフレーム・音声a2v（音声から動画を生成するモード）のいずれも、右クリックからの自動読み込みと受領ノート（NoteArea領域に出る短い案内文）の表示が合格。
- **動画音声a2vの主経路**: mix抽出（タイムライン全体でミックスされた音を録る方式）・wavの自動読み込み・Frames自動調整（wav長から必要フレーム数を逆算する処理）の一連が合格。
- **📝プロンプト追記**: 改行区切りでの末尾追記、受領ノート、字幕位置への予約B（配置系統B）が合格。
- **🖼キーフレーム追記の枚数分岐**: 1枚目の追加時は「生成起点」として画像直後に予約A（配置系統A）が入り、2枚目以降の追加では予約が動かない、という枚数による分岐挙動が合格。
- **📸現在フレーム取り込み**: 取り込み後も設定が保持されることを含め合格。
- **Chain破棄確認ダイアログ**: キャンセル操作をしても状態が壊れないこと、未編集（`isDirty`が立っていない）状態ではダイアログ自体が出ないことを含め合格。
- **リロード後の予約移動（9a）**: WebUIリロード後も予約席（仮オブジェクトの追跡状態）の移動が正しく働くことが合格。
- **生成中リロードのビジーガード（9b）**: 生成中にリロードしてもビジーガードが正しく機能することが合格。

### 37.2 条件付き・要修正1: 無音タイムラインでの動画音声a2v誤文言

無音のタイムラインで動画音声a2vを実行し失敗させた場合、期待される「無音失敗」の文言ではなく、近縁の誘導文言（`To use this video's sound, choose...`）が誤って表示される。§36.4で実装した「ネイティブ側が単一コード`EXTRACT_FAILED`しか返さないためエラーメッセージの部分一致で原因を分類する」仕組みの分類ロジックに不備がある可能性が高い。原因調査中。

### 37.3 条件付き・要修正2: 仮オブジェクトの4段階テキストの絵文字豆腐化と文言簡潔化

§36.6で実装した仮オブジェクトの4段階状態遷移（⏳生成予約→⏳生成中→❌失敗／✅完了）は、遷移そのものは正しく動作した。一方で次の2点が要修正として指摘された。

- **(a) 絵文字の豆腐化**: ⏳✅❌の各絵文字が、タイムラインのテキストオブジェクトでは豆腐（文字化けの四角表示）になる。AviUtl2のテキストオブジェクトが使う既定フォントに絵文字グリフが含まれていないことが原因と見られる。
- **(b) 文言改善（オーナー指示）**: 表示文言を英語に統一する。「予約」という開発者向けの内輪の言い回しをやめ、誰にでも通じる自然な英語表現へ変更する。✅完了時の操作説明（`insert it with the 🎞 button...`）は過剰な情報であり削除する（テキストオブジェクトは長文になるとタイムライン上で重要な情報が画面外に出てしまうため）。

対応方針として、絵文字を全廃しASCII文字のみの英語文言へ変更する方向で検討中。

### 37.4 不合格・設計変更: 🎞挿入位置の仮オブジェクト連動化

🎞（生成済み動画の挿入ボタン）による挿入位置が不合格となった。

- **現状の問題**: 挿入位置が「選択中レイヤーの現在フレーム位置」に固定されており、仮オブジェクトの位置とは無関係になっている。また、挿入先として空きレイヤーの選択がユーザーに強制される。
- **オーナー指定の新フロー**: 🎞を押したとき、対応する仮オブジェクトが存在すれば、その仮オブジェクトの位置（レイヤー・フレームとも）へ仮オブジェクトを削除したうえで生成済み動画を挿入する（レイヤー選択は不問）。対応する仮オブジェクトが存在しない場合は、現在フレームの最前面レイヤーへ挿入する。
- **未確定事項**: タイムライン上の余白が生成済み動画の尺に対して不足している場合の挙動（尺を短縮して収める、等）は要調査。

実装方式を調査中。

### 37.5 軽微バグの記録（修正不要・記録のみ）

生成中に`.aup2`（AviUtl2プロジェクトファイル）を保存→AviUtl2を終了→（バックエンド側で生成完了後に）AviUtl2を再起動してプロジェクトを開く、という手順を踏むと、ジョブ自体は完了済みであるにもかかわらず、仮オブジェクトが「生成中」の表示のまま残ってしまい、新しい予約を挿入できなくなる。生成済み動画を🎞で挿入すれば（＝古い仮オブジェクトが消えれば）解消する。

- **原因の見込み**: 再起動後に走るジョブポーリングは、その時点でのジョブ状態のスナップショットしか取得できず、「非終端状態→終端状態への遷移」というイベントそのものを観測できない。そのため、遷移をトリガーに動く終端処理（仮オブジェクトのテキスト更新・予約席の解放）が発火しない。
- **オーナー判断**: 有害度が低い（🎞挿入という通常操作で自然に解消する）ため、**修正は行わず記録のみ**とする。

### 37.6 将来の研究課題への追加: 操作パネルの状態復帰

AviUtl2再起動後、操作パネル（Create画面のメインプロンプト欄等）は終了前の状態を復帰しない。「生成中にAviUtl2を閉じる」という事故自体がレアケースであることと、エクスプローラーからの手動挿入などのフォールバックが常に存在することから、オーナー判断により優先度低・あったら便利級の将来課題として`PENDING_TASKS.md` §4へ追加する。

### 37.7 台帳への反映

`PENDING_TASKS.md` §1-2の状態行を「目視ほぼ合格・残修正3件（誤メッセージ・仮オブジェクト文言/豆腐・🎞置き換え挿入）対応中」へ更新し、§37.5の軽微バグ記録への参照を添えた。§4へ新規項目§4-11「操作パネルの状態復帰」を追加した。

## 38. 目視残修正A/B/Cの決着(2026-07-20)

§37で残った3件（無音時誤メッセージ・仮オブジェクト文言の絵文字豆腐化・🎞挿入位置の設計変更）が決着した。Cは調査の結果コード修正不要と判明してクローズ、Aは文言の全面刷新を実装、Bは敵対的レビューを経て新設RPCとして実装した。いずれも実装完了・全自動テスト緑で、デプロイとG3実機ゲートは未実施。

### 38.1 C: 無音タイムラインでの動画音声a2v誤文言の原因調査(コード修正なしでクローズ)

§37.2で報告された、無音タイムラインでの動画音声a2v失敗時に近縁誘導文言（`To use this video's sound, choose...`）が誤って表示される件を調査した。コード上、実際の無音失敗（音声抽出失敗）から近縁誘導文言（#7誤選択時の案内。`RIGHTCLICK_REDESIGN_SPEC.md` §4-3）が出る経路は存在しないことを確認した。この結果は、オーナーの観察が「動画オブジェクトを選択したまま#7（🎵この音声からa2v）を誤クリックし、近縁誘導ガードが仕様どおり正しく発動した」という操作と一致する。オーナーが再検証したところ、正しい無音失敗の文言（`RIGHTCLICK_REDESIGN_SPEC.md` §3-4 #3の「選択範囲の音声が無音のため抽出できませんでした」）が表示されることを確認し、人為的な操作ミスと確定した。コード修正なしでクローズする。

### 38.2 A: 仮オブジェクト文言の絵文字全廃とASCII英語化(実装完了)

§37.3(a)(b)で指摘された絵文字豆腐化・文言簡潔化に対応し、仮オブジェクトの4段階文言（`RIGHTCLICK_REDESIGN_SPEC.md` §5-5）を全面刷新した。

- 絵文字（⏳✅❌）を全廃し、ASCII文字のみの英語文言へ統一した。**この文言はAviUtl2側のUI言語設定に関わらず常に英語で表示する**（タイムライン上という特殊な表示領域での豆腐化リスクを避けるための判断）。
- 4段階の最終文言:
  1. `AI video will be placed here… [#id]`（native側の1フィールドにある16文字切り詰め制約への対策として、先頭断片をprefix側、残りをテキスト本体側へ分割配置する）
  2. `Generating: 〔プロンプト先頭〕… [#id]`
  3. `Failed: 〔理由〕 [#id]`（キャンセル時は`Failed: canceled [#id]`）
  4. `Done [#id]`
- 旧文言にあった操作説明（「🎞ボタンで挿入できます」等）は、タイムライン上で長文になると重要情報が画面外に出る問題があるため削除した。
- 各段階の文言へ`[#id]`マーカーを付与し、§5-7の予約検出（テキスト本文中のIDマーカー照合）と表示文言の形式を統一した。

### 38.3 B: 🎞挿入位置の仮オブジェクト連動化(新設RPC・敵対的レビュー反映)

§37.4で不合格となった🎞挿入位置（選択レイヤー・現在フレーム固定で仮オブジェクトの位置と無関係）を、オーナー指定の新フロー（マーカー位置への削除→置換）に沿って設計・実装した。新設RPC `timeline.insertMediaForJob {jobId, filePath}`。

- 仮オブジェクトが存在する場合: 全レイヤー検索でその位置を特定し、1編集セクションで原子的に「削除→同位置に実尺の明示長でcreate」する（選択中レイヤーは無視）。空き事前チェックは意図的に行わない（理由: (1) 削除前の仮オブジェクト自身の誤検知の穴、(2) セクション内削除可視性という未検証依存を避けるため）。
- createが真に失敗（null）した場合のみ`layer_max+1`へ退避する（`usedFallback`）。それも失敗すればエラーを返す（アンドゥ1回でのマーカー復元が退路）。
- 仮オブジェクトが存在しない場合: 従来の`insertMedia`と完全同一（現在選択レイヤー・現在フレーム）。オーナー決定・案1により、パネル起点の通常生成（仮オブジェクトなしが主経路）の使い勝手を変えない。
- V2V結合（joined）は対象外とし、従来の`insertMedia`のまま除外した（結合動画がマーカー位置に誤って置換される事故を防ぐため）。
- 挿入成功時は予約席stateを同期的に解放し、リロード後の誤ビジー窓を防ぐ。
- ロールバック手段としてwebui側に機能フラグ`REPLACE_INSERT_ENABLED`を新設し、G3実機ゲート不合格時は旧経路（`insertMedia`＋`deleteProvisionalByJob`）へ1行で復帰できるようにした。撤去はG3合格後。
- 応答: `{mode: "replaced" | "inserted", layer, frame, usedFallback}`

**敵対的レビュー**: 実装前に2件のBlockerが指摘され、当初案から修正した。

1. 当初案の「衝突→create nullを返す→長さ0まで短縮して再create」というラダー方式は、G1実機ゲートの既知事実（`create_object_from_alias`は衝突時もnullを返さず自動的に短縮してcreateする）と矛盾するため廃止した。
2. 「仮オブジェクトが無ければ最前面レイヤーへ挿入する」案は、パネル起点の通常生成（主経路）の挙動を変える回帰にあたるため、オーナー決定の案1（従来`insertMedia`と完全同一）へ変更した。

`RIGHTCLICK_REDESIGN_SPEC.md` §5-11で述べる「裏取りのない経路を自動で走らせない」というα原則と、本改修（衝突時のSDK実挙動が実機未検証のまま実装している点）との矛盾は、「G3実機ゲート合格を条件とするα昇格」として整理した。βの`resolveProvisional`（完了時自動置換、§4-8）とは異なり、あくまでユーザーの🎞ボタン明示操作に紐づく処理である点も切り分けて明記した。

### 38.4 実装結果とテスト基準

A・Bとも実装完了。テスト基準はvitest 1027件pass（`backend.integration`除外時）・doctest 251件pass/6件skipで全緑を確認した。デプロイと、G3実機ゲート（下記6項目）はオーナー確認待ち。

- ①マーカー位置への置換（選択レイヤー・カーソル無視）
- ②アンドゥ1回（delete+create）
- ③余白不足時のSDK実挙動観察（短縮かnullか。null時の`usedFallback`発火）
- ④joined非影響
- ⑤通常生成の挙動不変
- ⑥新文言4段階の表示（豆腐なし）

### 38.5 台帳への反映

`RIGHTCLICK_REDESIGN_SPEC.md`を第6版へ改訂した（改訂履歴・§5-5・§5-10・§5-11・§7-2・第8節）。`PENDING_TASKS.md` §1-2の状態行を「残修正A/B実装済み・G3実機ゲート待ち（デプロイ含む）」へ更新した。

## 39. G3クローズとJoin機能のスコープ外化(2026-07-20)

§38で実装完了したG3実機ゲート（🎞挿入の位置連動、6項目）をオーナーが実機で確認した。6項目中5項目は合格、残る1項目（④joined非影響）の確認過程でJoinボタンの表示バグが発覚し、調査のうえスコープ外化してクローズした。

### 39.1 5項目の合格結果

以下の5項目はいずれも実機で合格した。

- **①マーカー位置への置換**: 選択中レイヤーやカーソル位置を無視し、仮オブジェクトのマーカー位置（レイヤー・フレーム）へ正しく置換されることを確認した。
- **②アンドゥ1回**: 削除→createの一連の付け替えが、Ctrl+Z1回でまとまって戻ることを確認した。
- **③余白不足時のSDK実挙動の観察**: マーカー位置に生成済み動画の実尺が収まらない場合の挙動を実機で観察した結果、**`create_object_from_media_file`は`create_object_from_alias`とは異なり、衝突時に自動短縮せずnullを返す**というSDK挙動の非対称が確定した。G1実機ゲートで確認済みの「`create_object_from_alias`は衝突時も短縮してcreateする」（§38.3敵対的レビュー参照）とは別のAPIであり、同じ「衝突時挙動」でもAPIによって振る舞いが異なるという新知見である。null返却時に`usedFallback`が正しく発火し、`layer_max+1`への退避が実機で正常に動作することもあわせて確認した。
- **⑤通常生成の挙動不変**: 仮オブジェクトを伴わないパネル起点の通常生成で、🎞挿入の挙動（現在選択レイヤー・現在フレームへの挿入）が従来から変化していないことを確認した。
- **⑥新文言4段階の表示**: §5-5で改訂した4段階の英語文言が、タイムライン上で絵文字の豆腐化なく正しく表示されることを確認した。

### 39.2 ④joined非影響の確認中に発覚したJoinボタン表示バグ

残る④joined非影響（V2V結合動画の挿入が本改修の対象外として従来どおりの経路のままか）を確認しようとしたところ、そもそもJoinボタンが実機で一度も表示されないことが判明した。

調査の結果、原因はAPI契約の断絶と確定した。フロントの表示判定は`job.request.source_video`の有無を見ているが、実バックエンドの`GET /jobs`は`to_clip_request`のホワイトリストで組んだ`GenerateRequest`を返すため、v2v情報（`chain_request`という内部フィールドにあり公開APIの外）を含まない応答になる。判定は常にfalseとなりボタンは恒久的に非表示だった。自動テストが緑だった理由は、モックブリッジがリクエストボディを丸ごとエコーバックし実サーバーの絞り込みを再現していなかったためであり、「テスト緑・実機不可視」を生むモックと現実の乖離の一例として記録する。

あわせて、オーナーが当初提示していたjoined動画挿入時の理想挙動（(1)仮オブジェクトの自動消去、(2)元動画のリボンを消さない、(3)元動画と同じ開始位置・最前面レイヤーへの挿入）も、現状の実装（素の`insertMedia`。マーカーに一切触れないため(2)のみ結果的に満たす）とは乖離していることが判明した。修復には`JobResponse`への`is_v2v`フィールド追加などバックエンド側の変更を要し、着手にはバックエンド凍結の解除判断が必要になる。理想挙動(1)(3)の実現にはjobIdとタイムライン位置を紐づける仕組みの新設も要る。

### 39.3 オーナー決定とG3クローズ

調査結果を受け、オーナーはJoin機能の修復（表示バグ・理想挙動・長尺対策のすべて）をα版のスコープ外とし、将来改修へ格下げする決定を下した。理由は、α版ではv2vの続き動画を独立した生成物として扱えば実用上は足りること、元動画との連結はユーザーがAviUtl2上で手動配置しても代替できることの2点。

この決定により、G3実機ゲートの④joined非影響は「スコープ外化によりクローズ」として扱い、G3実機ゲート自体は5項目合格・1項目スコープ外化で完了とした。調査記録と将来改修時の申し送り事項は[`JOIN_FEATURE_RESEARCH.md`](JOIN_FEATURE_RESEARCH.md)へ新規にまとめた。

### 39.4 台帳への反映

`JOIN_FEATURE_RESEARCH.md`を新設した。`PENDING_TASKS.md`は§1-2の状態行を「G3クローズ（joined項目はスコープ外化）・I14仕上げのみ残」へ更新し、§4-12「V2V結合（Join）機能の修復と理想挙動の実装」を新規に追加した。`RIGHTCLICK_REDESIGN_SPEC.md`は第8節のG3ブロックへ結果注記を追加し、第9節のスコープ外一覧へJoin機能修復の項目を追加した（第7版）。

## 40. I14仕上げとワークオーダー完了(2026-07-20)

§39でG3実機ゲートがクローズ（5項目合格・1項目スコープ外化）したことを受け、右クリック再設計ワークオーダーの最終増分I14「仕上げ」を実施した。機能追加はなく、G3合格を条件としていた撤去・整理・整合のみである。これをもって、ワークオーダー第1段階（増分I1〜I14）の全工程が完了した。

### 40.1 ロールバックフラグの撤去（🎞挿入の一本化）

G3合格を条件に温存していたロールバック手段を撤去した。`webui/src/jobs/downloadAndInsert.ts`の機能フラグ`REPLACE_INSERT_ENABLED`と、`insertForJob`のelse側に保持していた旧経路（`insertMedia`＋`deleteProvisionalByJob`の2呼び出し）を削除し、`timeline.insertMediaForJob`（マーカー位置への削除→置換を1RPCで行う新設RPC。§38.3）経路へ一本化した。V2V結合（joined）が無条件に素の`insertMedia`を使う分岐（§38.3のMed 6対応）は従来どおり維持している。`downloadAndInsert.test.ts`の旧経路専用テスト2件（`insertForJob (legacy rollback branch)` describe内）も削除した。なお`resolveProvisional`・`deleteProvisionalByJob`・`insertMedia`の各RPCは現役または将来用のため撤去していない（削除したのはwebui側の旧経路の配線とフラグのみ）。

### 40.2 ネイティブ既定プレフィックスのASCII化

`native/src/alias_util.cpp`の`kGeneratingPrefix`を「⏳生成中：」（絵文字＋日本語）から「`Generating: `」（ASCII）へ変更した。webuiは常に明示のローカライズ済みプレフィックス（§5-5の4段階文言）を渡すため、この既定値はcaller未指定の非常経路でのみ到達するフォールバックだが、万一その経路を通っても絵文字の豆腐化を起こさないための整合である。`native/tests/test_alias_util.cpp`の該当doctest（`honors an explicit textPrefix`）の期待値も同ASCII文字列へ更新した。TEST_CASE数は不変のため、doctest件数（251pass/6skip）は変わらない。

### 40.3 残骸整理（旧アクション時代の未参照キー・デッドブランチ）

I8で表示を廃止していた`strings.ts`の`chain.menuSourceNote`（英日2件）、およびI2で`regenerate`アクションを廃止して到達不能になっていた`LibraryScreen.tsx`の`initialIntent`ガード付きノート表示と、それが引く`strings.ts`の`library.regenerateNote`（英日2件）を、いずれもgrepで参照ゼロを確認のうえ削除した。あわせて、`regenerate`アクションの受け皿だった`LibraryScreen`の`initialIntent` propと、`AppShell.tsx`側の`libraryIntent`受け渡し（メニュールーティング表に`targetMode: "library"`が1件も残っておらず常に`undefined`だった）も削除した。`GenerationPrefill`の型・`CreateScreen`/`ChainScreen`の`initialIntent`は現役のため据え置いた。Library自体は通常のタブとして存続し、`remountTokens.library`もその`key`として現役のため残した。

### 40.4 言語ファイルの最終整合

`Language/English.NzVideomni.aul2`・`Language/Japanese.NzVideomni.aul2`を精査した。両ファイルとも、プラグイン（`native/src/plugin.cpp`の`kObjectMenuItems`8項目＋`kLayerMenuItems`2項目）が`Translate()`で引く現行メニュー10項目のキーが過不足なく揃っており、絵文字テスト時代や廃止4項目時代の古いエントリは残っていなかった。UI/エラー文言（起動・WebView2系）のキーも現役分のみで、削除対象はなかった（変更なし）。

### 40.5 文書の最終整合

`RIGHTCLICK_REDESIGN_SPEC.md`を第8版へ改訂した。第8節の実機チェックリストを最終点検し、未消化のまま残る2項目（#9 `register_project_load_handler`の手動オープン発火・#10 `object_name`の`.aup2`永続化）に「将来検証」の状態注記を追記した。いずれも予約検出がテキスト本文の`[#id]`マーカーを第一手段とする二重化設計のため、未確認でも実害が小さく第1段階のスコープでは決定的確認を必須としない旨を明記している。`PENDING_TASKS.md`は§1-2（タイムライン右クリック系のデバッグ）をクローズし、「3. 実装済み」§3-22として集約した。

### 40.6 ワークオーダー完了宣言と最終テスト基準

以上により、右クリック再設計ワークオーダー第1段階（増分I1〜I14）は全工程完了とする。実機ゲートはG1（Phase A・§34）・G2（Phase B・§35）・G3（残修正B/A・§39）をすべて消化した（G3の④joinedはスコープ外化でクローズ）。最終テスト基準は以下のとおり全緑を確認した。

- **typecheck**: `npm run typecheck`（`tsc -b`）エラー0。
- **vitest**: 1025件pass（`backend.integration`除外時）。旧基準1027件から、§40.1で削除した旧経路専用テスト2件分の減（内訳: `insertForJob (legacy rollback branch)` describe内の2 `it`）。
- **doctest**: 251件pass/6件skip（§40.2のプレフィックス期待値更新はTEST_CASE内の文字列変更のみで件数は不変）。

### 40.7 台帳への反映

`PENDING_TASKS.md` §1-2をクローズし§3-22へ移動した。`RIGHTCLICK_REDESIGN_SPEC.md`を第8版へ改訂した（改訂履歴・第8節#9/#10）。

### 40.8 実機チェックリスト#1の消化確定（2026-07-20追記）

第8節#1（右クリック位置と挿入位置の一致）は、2026-07-20の実機再確認で`get_mouse_layer_frame`方式の正式採用が確定した。右クリック位置に挿入されること、メニュー内で選択前にマウスを大きく動かしても右クリック位置に入る（メニュー内のマウス移動に汚染されない）ことを確認し、消化済み・合格とした。

## 41. タイムライン右クリック再設計 第2段階（オーナー実機フィードバック起点、2026-07-20）

§40でワークオーダー第1段階（増分I1〜I14）が完了した後、オーナーが実機で第1段階を触った結果のフィードバック9件と、新機能の要望を受けた。これらを踏まえて方針を確定し、計画（敵対的レビューでBlocker0件）を経て、増分実装（バックエンド`config.yaml`側1件・ネイティブ側数点・webui側複数点）を行った。以下、実装した内容を機能単位で記録する。[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md)を第9版へ改訂し、本節の内容を反映済み。

### 41.1 バックエンド`config.yaml`: minimal/smallプリセットの481フレーム化（凍結例外・値のみ）

`Nz-Videomni/config.yaml`はバックエンド凍結方針の例外として、値のみの変更を行った。`generation_presets.minimal`（512×320）と`generation_presets.small`（960×576）の`num_frames`を、それぞれ49・121から**481**（20秒@24fps上限）へ引き上げた。あわせて`limits.spill_free_frames`に`"512x320": 481`・`"960x576": 481`の2エントリを追加し、これら低解像度プリセットについても快適上限テーブルへ載せた。コード（ロジック）の変更は無く、YAMLの数値のみの変更である。

### 41.1a G4実機ゲートによるsmallプリセットの保守側retreat（2026-07-20、追記）

§41.1で481フレーム化した2プリセットのうち、`small`（960×576）はG4実機ゲートでの実生成において**わずかにVRAMスピルが発生**した。`minimal`（512×320）は溢れずそのまま481フレームを維持できたため、変更対象は`small`のみである。オーナー判断により、`small`の`num_frames`を保守側の**457**フレームへ変更した（退路として承認済み）。457の根拠は、`Nz-Videomni/Docs/RESOLUTION_DURATION_CAPABILITY.md`のトークン式から見積もった960×576の溢れ境界（462〜576フレーム帯）を下回る、最大の8n+1グリッド値（`8×57+1=457`）である。

これに伴い、`Nz-Videomni/config.yaml`の`generation_presets.small.num_frames`と`limits.spill_free_frames["960x576"]`を457へ変更し、`webui/src/modes/create/defaultConfig.ts`の`FALLBACK_APP_CONFIG`・`webui/src/bridge/mockBridge.ts`の`MOCK_CONFIG_BODY`を同値へ追随させた。`webui/src/modes/create/spillUtils.test.ts`の期待値も457へ更新している。`minimal`（512×320）の481フレームおよび他プリセットの値には一切触れていない。

### 41.1b smallプリセットの481フレーム復帰（2026-07-20、続報）

§41.1aで457へ後退させた`small`（960×576）の観測結果について、オーナーがログを確認したところ、G4で見えていた「わずかな共有メモリ使用」は速度低下を伴わない無害なベースラインであると判明した。これは`Nz-Videomni/Docs/RESOLUTION_DURATION_CAPABILITY.md` §8.2が定義する「オーガニック（非溢れ）ベースライン ≈1GB」、および`VERIFICATION_LOG.md`の「ambient」記載に相当するもので、§8.2の判定基準（フレーム数を481／457／425と変えても溢れ量が変わらない＝フレーム数に依存しない固定ベースラインであれば実害の無い溢れ）にも合致する。つまり457への引き下げは不要だったことになる。

これを受けて、`small`の`num_frames`と`limits.spill_free_frames["960x576"]`を**481へ復帰**した。変更箇所は§41.1aで457化した4ファイル（`Nz-Videomni/config.yaml`・`webui/src/modes/create/defaultConfig.ts`・`webui/src/bridge/mockBridge.ts`・`webui/src/modes/create/spillUtils.test.ts`）と、`webui/src/App.prefill.test.tsx`の関連2ケース（DURATIONライブSETの期待値）。`PENDING_TASKS.md` §2-2⑥は「合格」として消化し、`REAL_BACKEND_CHECKLIST.md`・`RIGHTCLICK_REDESIGN_SPEC.md`の457記載も481へ整合させた。バックエンド`config.yaml`の変更を実環境に反映するには、稼働中のバックエンドプロセスの再起動が必要である。

### 41.2 webuiフォールバックのバックエンド追随（`defaultConfig.ts`／`mockBridge.ts`）

`webui/src/modes/create/defaultConfig.ts`の`FALLBACK_APP_CONFIG`と`webui/src/bridge/mockBridge.ts`の`MOCK_CONFIG_BODY`は、`Nz-Videomni/config.yaml`の現行値から取り残されていた（`generation_defaults`が旧値の512×320/49フレームのまま、`limits.max_width`/`max_height`も旧値1920×1088のままだった）。両ファイルを、バックエンド`config.yaml`が既に持っていた現行値（`generation_defaults`: 1280×768（`crop_output`1280×720）/257フレーム/24fps、`limits.max_width`/`max_height`: 4096×4096）へ同期し、あわせて§41.1の`minimal`/`small`の481フレーム化と`spill_free_frames`の2エントリ追加も反映した。`webui/src/api/client.test.ts`・`webui/src/modes/create/spillUtils.test.ts`の期待値も新しい値へ更新している。

### 41.3 レイヤーメニューの4項目化と📷への絵文字統一（`native/src/plugin.cpp`）

レイヤー右クリックメニュー（`kLayerMenuItems`）を、従来の✨（`textToVideoHere`）・📸（`imageFromCurrentFrame`）の2項目から、次の2項目を新設して4項目へ拡張した。

- **`addCurrentFrameAsKeyframe`**（📷 Add the current frame as a keyframe / 📷 現在フレームをキーフレームに追加）: #5「この画像をキーフレームに追加」の「取り込み版」。現在のタイムラインフレームを取り込み、Create画面のフォーム状態を保ったままキーフレーム末尾へ追加する。
- **`currentFrameToClipChain`**（📷 Generate a long video from the current frame (i2v Clip Chain) / 📷 現在フレームから長尺動画を生成 (i2v Clip Chain)）: #6「この画像から長尺動画を生成」の「取り込み版」。現在のタイムラインフレームを取り込み、Chain画面のClip Chain冒頭キーフレームへ自動設定する。

あわせて、`imageFromCurrentFrame`のアイコンを📸（camera-with-flash、U+1F4F8）から📷（camera、U+1F4F7）へ変更し、新設2項目も同じ📷で統一した（3項目とも「現在フレームを扱う」という共通の意味を絵文字で揃える狙い）。`Language/English.NzVideomni.aul2`・`Language/Japanese.NzVideomni.aul2`の両言語ファイルへ、新2項目の訳文と📷への差し替えを反映した。

### 41.4 「現在フレーム」系3項目の配置基準の精緻化（`use_mouse_frame`、`native/src/plugin.cpp`）

第1段階では、右クリックの挿入位置（系統(C)）はレイヤー・フレームとも`get_mouse_layer_frame`（右クリック位置）で統一されていた。今回、「現在フレーム」を扱うカメラ系3項目（`imageFromCurrentFrame`・`addCurrentFrameAsKeyframe`・`currentFrameToClipChain`）について、キャプチャする内容自体が「いま表示しているプレビューフレーム」（`EDIT_INFO.frame`、赤カーソル）であることに合わせ、仮オブジェクトの**フレーム位置も赤カーソルに揃える**よう変更した。レイヤー位置は従来どおり右クリック位置（マウスのレイヤー）のままである。

実装は`SnapshotSelection(edit, snap, use_mouse_frame)`という真偽値引数の新設で行った。`EmitMenuInvoked`がアクション名で判定し、上記3項目のみ`use_mouse_frame=false`（フレームをカーソルのまま上書きしない）を渡す。✨（`textToVideoHere`）には「現在フレーム」という概念が無いため、従来どおり`use_mouse_frame=true`（フレーム・レイヤーとも右クリック位置）のままである。

### 41.5 仮オブジェクトの書式変更（`native/src/alias_util.cpp`）

仮オブジェクト（テキストオブジェクト）の文字サイズを`100`から**`34`**へ縮小した。サイズ100は仮オブジェクトの短い尺（既定でも数秒〜数十秒程度）に対して大きすぎ、タイムライン上での視認性を損なっていたための調整である。あわせて、文字揃え項目（「文字揃え」＝「中央揃え[中]」）をエイリアスへ追加し、仮オブジェクトの文字が中央に揃うようにした。`native/tests/test_alias_util.cpp`にサイズ34への期待値更新と、文字揃え項目が正しく埋め込まれ`ParseAliasItemValue`で読み出せることを確認する新規TEST_CASEを追加した（doctest件数+1）。

**未検証事項**: この書式変更は、doctestでエイリアス文字列への埋め込み・読み出しは確認済みだが、**AviUtl2本体がこの文字揃え項目を実際に受理し中央揃えとして描画するかは実機で未検証**である。受理されなかった場合の退路として、文字揃えの行だけをエイリアスから削除しサイズ34のみを残す縮退構成も用意している。実機確認はG4（[`PENDING_TASKS.md`](PENDING_TASKS.md) §2）で行う。

### 41.6 DURATION（`num_frames`）自動決定エンジンの新設（`webui/src/timeline/deriveDuration.ts`）

第1段階では、右クリックプリフィルの幅・高さ（解像度）は`deriveGenerationParams.ts`が決定する一方、DURATIONは「別途の未決定事項」として素通しにされていた。今回、新設モジュール`deriveDuration.ts`でこの空白を埋めた。3方式（`DurationPolicy`）を用意している。

- **`comfortCeiling`（快適上限）**: 素材由来の尺の制約が無い項目（t2v・i2v等）向け。解像度から`spill_free_frames`の快適上限を引き、常にその値へ合わせる。
- **`materialClampedToCeiling`（素材長に上限クランプ）**: 参照動画・続き生成のソース動画を持つ項目向け。素材の実尺を8n+1グリッドへ切り下げた値を基本とし、快適上限を超える場合はさらに切り下げる（素材より長い提案はしない）。ただしフォーム最小DURATIONを下回る場合は最小値へ引き上げる。
- **`untouched`（触らない）**: DURATIONがプリフィル契約に含まれない項目。フォームの既存値をそのまま残す。

主な適用先: #4 i2vと#1/#6/#12（Chainの各Clip 0）は`comfortCeiling`、#2 IC-LoRAは`materialClampedToCeiling`、#3/#7のA2Vは初期シードとしては`untouched`（wav長からの自動調整に委ねる）。#9/#10はリマウントしないため初期シード経路ではなく、`AppShell`が右クリックのたびに`computeTargetNumFrames`をライブなフォーム幅・高さから直接計算し、新設のライブコマンド`setDuration`でCreateフォームのDURATIONだけをその場で書き換える。#5/#8/#11はDURATIONに触れない。

あわせて、A2Vのwav長自動調整（`useGenerationForm.ts`の`suggestFramesForAudio`提案値）にも、現在の解像度の快適上限によるキャップを追加した（ピック／ドラッグ＆ドロップ／右クリック#3・#7を含む全A2V添付経路に適用される、オーナー承認済みの意図的な挙動変更）。長い音声を添付しても、生成が快適上限を超えてスピルする状態には自動では入らなくなる。

### 41.7 右クリックプリフィルの解像度・FPS方針設定（Settings、`PrefillPolicyContext.tsx`）

Settingsパネルに「右クリック時のサイズとFPS」という3択の設定を新設した（`localStorage`キー`nzvideomni.prefillResolutionPolicy`、永続化はテーマ切替と同じくクライアント側のみ）。

1. **既定値**: 選択素材の実寸を無視し、常にバックエンドの`config.generation_defaults`を使う。
2. **プロジェクトに合わせる**: AviUtl2プロジェクト自体の解像度・fpsを`timeline.getEditInfo`で1回だけ取得し、プリフィル直後に上書きする。
3. **素材に合わせる（既定）**: 選択素材の実寸とプロジェクトのフレームレートから求めたfpsを使う（従来からの挙動）。

この設定は右クリックプリフィルの初期値の決め方だけに影響し、通常のパネル編集・プリセット適用・「Get size from AviUtl2」ボタンの挙動は変えない。DURATION（§41.6）とも連動しており、②「プロジェクトに合わせる」を選んだ場合はDURATIONも`getEditInfo`の解像度から改めて計算し直す。

### 41.8 アコーディオン自動展開（`GenerationForm.tsx`）

右クリックプリフィルが#2（参照動画）・#3（動画音声a2v）・#7（音声a2v）のいずれかで、素材がIC-LoRAまたはA2Vのアコーディオン（`<details>`）の中へ読み込まれるとき、そのアコーディオンを開いた状態で表示するようにした。従来は常に折りたたみ状態のままで、素材が読み込まれても「どこへ行ったか」が一見して分からなかった。実装は`<details>`要素への`ref`を使った`useLayoutEffect`によるマウント時1回限りの`.open = true`書き込みで、以後は完全に非制御（ユーザーの手動開閉を妨げない）のままにしている。`useEffect`ではなく`useLayoutEffect`を使っているのは、`CreateScreen`側の素材自動読み込みエフェクト（「Extracting audio…」ノート表示等）と同じコミットフェーズでバッチングされ、テスト（`App.prefill.test.tsx`のvideoAudioToVideoケース）が過渡的なノート表示を観測できなくなる不具合が実際に起きたためで、レイアウトエフェクトはコミット内のどのpassiveエフェクトよりも先に走ることでこれを避けている。

### 41.9 #10のframe 0衝突修正（置換方式へ、`CreateScreen.tsx`）

第1段階の実装は、📷（`imageFromCurrentFrame`、#10）を実行するたびに`addFromCapture(0)`で**常に新規カードを追加**していたため、KEYFRAMESに既にframe 0のカードがある状態で#10を実行すると、frame 0の位置に2枚のカードが重なる見落としがあった。今回、frame 0のカードが既にある場合はその画像をキャプチャで**置き換える**（`captureIntoCard`。カード枚数は変わらず、frameIdx・strengthは維持）よう修正した。frame 0のカードが無い場合は従来どおり新規カードを追加する（上限到達時は追加せず失敗として通知）。

### 41.10 新設アクションのwebui配線（#11/#12）

`webui/src/timeline/menuRouting.ts`に`addCurrentFrameAsKeyframe`・`currentFrameToClipChain`のルーティングエントリを追加した。

- **#11 `addCurrentFrameAsKeyframe`**: Create行き・リマウントなし。`AppShell.handleRoute`に専用の早期リターン分岐を追加し、キーフレーム末尾グリッドへキャプチャを追加する新設ライブコマンド`addCaptureAsKeyframe`をCreate画面へ送る。追記実行時点のキーフレーム枚数が0（＝1枚目になる場合）のときだけ、右クリック位置（配置系統C、フレームは赤カーソル・レイヤーは右クリック位置。§41.4）へ仮オブジェクトを予約する。2枚目以降になる場合は仮オブジェクトに一切触れない（#5と同じ分岐則）。
- **#12 `currentFrameToClipChain`**: Chain行き・リマウントあり（#1/#6と同じくChain編集破棄の確認ダイアログつき）。`ChainScreen.tsx`の自動読み込みエフェクトに専用ルーチンを追加し、`timeline.captureFrame`でキャプチャしたPNGをクリップ1冒頭キーフレームへ読み込む。仮オブジェクトは配置系統Cで右クリック位置（フレームは赤カーソル・レイヤーは右クリック位置）へ予約する。

### 41.11 既知の設計メモ（据え置き事項）

リマウント系のプリフィル予約（仮オブジェクトの長さ）は、従来どおりconfig既定値ベースのまま据え置いた。フォームの初期DURATION（§41.6のDURATION決定エンジンによる値、例: 481）と、仮オブジェクトの予約長（リマウント直後はまだconfig既定値を使う）は一致しない場合がある。実際にGenerateを押した時点で、フォームの実DURATION値へ貼り直される（§5-3の`updateProvisionalReservation`用途(a)）ため実害は無いが、リマウント直後の一瞬だけ仮オブジェクトの長さとフォームのDURATION表示がずれて見える可能性がある。今回のスコープでは対応せず、既知の設計メモとして残す。

### 41.12 最終テスト基準

- **typecheck**: `npm run typecheck`（`tsc -b`）エラー0。
- **vitest**: **1082件pass／10件skip**（77ファイル、`backend.integration`除外時）。
- **doctest**: **252件pass／6件skip**（`--test-suite-exclude=integration`。§41.5で追加した1 TEST_CASE分、旧251から+1）。
- **埋め込みフルビルド**: 成功。

デプロイ・実機ゲート（G4、[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) 第8節「G4実機ゲート」）は未実施でオーナー待ち。チェックリストは[`PENDING_TASKS.md`](PENDING_TASKS.md) §2へ追加した。

### 41.13 メニュー日本語化症状の原因判明とREADME注記追加（G4追加確認、2026-07-20）

G4実機ゲートの目視で、「レイヤー右クリックメニューが日本語で表示されない」という症状が報告された。調査の結果、コード側に問題はなく、`native/src/plugin.cpp`の英語キー12件（オブジェクト右クリック8項目＋レイヤー右クリック4項目）はすべて`Language/Japanese.NzVideomni.aul2`の訳文キーと一致していることを確認した。

原因は、メニューの翻訳が**プラグイン登録時（AviUtl2起動時）に1回だけ**行われ、どの言語ファイル（`Language\Japanese.NzVideomni.aul2`等）を引くかを**AviUtl2本体の「設定→言語の設定」**が決める、という仕組みそのものにあった。操作パネル（WebView2）側のLANGUAGE設定は、パネル内の文字列だけを切り替えるものであり、ネイティブメニューの翻訳とは無関係（意図した仕様であり、不具合ではない）。オーナーはこれまで言語設定を**Default**（AviUtl2本体内蔵の日本語文言を使う設定で、プラグインの言語ファイルは引かれない）のまま使用していたため、メニューが常に英語キーのフォールバック表示になっていた。設定→言語の設定で**Japaneseを明示選択**し、AviUtl2を**再起動**したところ、メニューが正しく日本語化されることを実機で確認した。翻訳の質には改善の余地があるが、α版としては合格とオーナーが判定した（訳文の磨き込みは将来課題として起票。[`PENDING_TASKS.md`](PENDING_TASKS.md) §4-15）。

この経緯を踏まえ、`README.md`の「Run AviUtl2」節に、メニューを日本語表示するための設定手順（設定→言語の設定→Japaneseを選択→AviUtl2再起動）の注記を追加した。あわせて[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §5-14にこの翻訳の仕組みを仕様として明記し、[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-2のG4チェックリスト③（メニュー並び・📷表示・訳）を合格として消化した。

### 41.14 第2段階の完結（G4全9項目合格・コミット＆プッシュ・次セッション予定）

G4実機ゲートの残り項目もオーナーが実機で確認し、**全9項目に合格**した（2026-07-20）。①仮テキストの文字揃え受理、②📷3項目の配置基準、③メニュー4項目の並び・📷表示・訳（§41.13）、④アコーディオン自動展開、⑤DURATIONマトリクス代表ケース、⑥minimal・small各481f実生成のVRAM（§41.1b、smallは481で確定）、⑦#10のframe 0置換、⑧新規📷2項目の予約アンドゥ1件化、⑨Settings3択の切替動作、のすべてである。これにより右クリック再設計の第2段階は完結した（[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-23へクローズ集約、G4チェックリストは§2-2から除去）。

オーナーがコミット＆プッシュを手動で実施した。`git log`上のコミットは`a3bbb5f`「R Click」（第2段階の本体: レイヤーメニュー4項目化・DURATION自動決定エンジン・Settings3択・アコーディオン自動展開・#10置換ほか）と`8126601`「R click」（smallプリセットの481復帰と、それに伴う`DEVLOG`／`PENDING_TASKS`／`REAL_BACKEND_CHECKLIST`／`RIGHTCLICK_REDESIGN_SPEC`／`README`とテスト2ファイルの整合）の2件。

**次セッションの予定（オーナー宣言）**: Reference video（IC-LoRA）および Audio to video（A2V）のアコーディオン内の**カードのリデザイン**を行う。詳細仕様は未定でオーナーが次セッションで指示予定。現状の実装箇所は`webui/src/modes/create/GenerationForm.tsx`のIC-LoRA節（`ReferenceVideoSection`）・A2V節（`SourceAudioSection`）配下のカードUI。起票は[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-1。

## 42. Reference video（IC-LoRA）／Audio to video（A2V）カードのリデザインと尺表示（`fs.probeMediaInfo`新設、contract v9、2026-07-20）

§41.14で予告したとおり、Create画面のアコーディオン内カード2種——参照動画（IC-LoRA。参照した動画の動きや輪郭をなぞって生成する仕組み）欄の`ReferenceVideoSection`と、音声（A2V。音声から動画を生成する機能）欄の`SourceAudioSection`——のカードUIを、他のカード類（Chainの統合ソース入力欄`SourceInputPanel`・KEYFRAMESカード）とデザインを統一する形で作り直した。あわせて、添付ファイルの尺を「12.3s」の形で表示するためのブリッジ契約v9（`fs.probeMediaInfo`）を新設した。[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-1として起票していた項目である。

### 42.1 オーナー指示の詳細仕様

§1-1の起票時点では「詳細仕様はオーナーが次セッションで指示」だったところ、本セッション冒頭でオーナーから具体仕様が示された。要点は次のとおり。

- 他のカード類（Chain SOURCEカード・KEYFRAMESカード）とデザインを統一する。
- 🔁でクリア（添付を外す）、📁でファイル選択、カード全体をドラッグ&ドロップの受け入れエリアにする。
- A2Vカードには♫マークの仮サムネイル（音声にはプレビュー画像が無いため）を表示する。
- 両カードにファイル名と尺（「12.3s」表記）を表示する。IC-LoRA側の尺表示も**必須スコープ**（オーナー明示指示）。
- KEYFRAMESカードほど情報を満載にはしないが、現状カードが表示していた情報（排他ノート・各種警告・control LoRA選択・強度スライダー等）は一切欠落させない。

### 42.2 計画プロセスと敵対的レビューの致命的指摘

プランモードで計画を立案し、敵対的レビューエージェントによる検証を行った。**致命的指摘が1件**あり、計画に反映した。

- **致命的指摘（probe失敗時のreject問題）**: 尺取得の`fs.probeMediaInfo`を、当初計画では「取得に失敗したらrejectする」設計にしていた。しかしこれは、プロジェクト未オープン・未対応形式・`get_media_info`失敗といった**正常運用中に日常的に起こりうるケース**でまでrejectを飛ばすことになり、（a）webui側のprobeエフェクトに`.catch()`が無いとunhandled rejectionになる、（b）そもそも尺表示はベストエフォートのUIヒントに過ぎず失敗を例外扱いする必然性が無い、という二重の問題があった。**是正**: `fs.probeAudioDuration`（[`BRIDGE_CONTRACT.md`](BRIDGE_CONTRACT.md) §4.19相当）と同じ「取得不能はすべて`{durationSec:0, width:0, height:0}`の成功応答へ穏当に劣化させ、rejectは`filePath`不正の`BAD_REQUEST`のみ」という設計へ変更した。あわせてwebui側のprobeエフェクトには`.catch()`を必須とする規律を確定した（唯一残る`BAD_REQUEST`のreject経路がunhandled rejectionにならないための保険）。**教訓**: ベストエフォートのUIヒント用途のブリッジメソッドは、正常運用中に頻発する取得失敗を例外（reject）ではなく「不明＝0」の正常応答で表現し、UI側は穏当な非表示へ劣化させるのが正しい。`fs.probeAudioDuration`の設計に最初から倣うべきだった。

このほか複数の要修正が指摘され、計画へ織り込んだ。オーナー判断としては3点を確定した。①ドラッグ&ドロップの案内文（`dropHint`/`dropActive`）は削除してカードのデザインを統一する、②`DropZone`一式は本番利用がGenerationFormのみになりデッドコード化するため削除する、③A2Vの尺表示はwavのみ（非wavはファイル名＋♫サムネイルのみで尺は出さない）。

### 42.3 ネイティブ実装（`fs.probeMediaInfo`新設、Bridge Contract v9）

新規ブリッジメソッド`fs.probeMediaInfo`を追加した。契約の詳細は[`BRIDGE_CONTRACT.md`](BRIDGE_CONTRACT.md) §4.22・§8（v9行）を参照。

- リクエストは`{ filePath: string }`（非空必須）、成功レスポンスは`{ durationSec: number, width: number, height: number }`（3つとも非nullable、不明時は`0`。`durationSec`は秒、静止画は`0`）。
- 実装は、AviUtl2 SDKの`EDIT_SECTION::get_media_info`を`call_edit_section_param`経由でUIスレッド同期に呼ぶ（既存の`InsertMediaEditProc`/`GetSelectionEditProc`と同じ流儀）。`HandleMessage`の非同期インターセプトには入れず、`HandleRequestJson`の同期分岐（`ui.resolveDroppedFiles`の後）に置いた。UTF-8のfilePathからワイド文字列への変換は`ProbeMediaInfoEditProc`内ローカルの`std::wstring`で行い、コールバックのライフタイム内でのみ有効にすることでライフタイム安全を担保している。
- **設計上の要点（§42.2の是正を反映）**: 失敗してもrejectしない。編集ハンドル無し（プロジェクト未オープン）・`get_media_info`失敗・未対応形式（webm/mkv等）・`MediaInfoProvider`未配線はすべて`{0,0,0}`の成功応答となり、UI側は尺非表示へ落ちる穏当な劣化になる。エラー（`BAD_REQUEST`）は`filePath`が欠落/空/非文字列のときのみ。
- 変更ファイル: `native/src/bridge_core.h`（`ProbeMediaInfoRequest`／`MediaInfoSnapshot`／`MediaInfoProvider`／`RequestContext::probe_media_info`／各宣言）、`native/src/bridge_core.cpp`（`ParseProbeMediaInfo`／`MakeMediaInfoResult`／ディスパッチ）、`native/src/bridge.cpp`（`ProbeMediaInfoEditProc`とprovider配線）、`native/tests/test_bridge_core.cpp`（6ケース追加）。
- テスト: doctest **252→258 pass/6 skip**、退行なし。

### 42.4 webui実装

- **`bridge/types.ts`・`mockBridge.ts`**: `fs.probeMediaInfo`の型（contract v9のJSDoc）とモック（`probeMediaInfoDurationSec`等のオプション、未設定時は`0`）を追加した。
- **`useSourceUpload.ts`（Chain共用フック）**: `SourceUploadState`に`filePath: string | null`を追加公開した（全`setState`経路）。追加専用の拡張でChain側の既存挙動は不変。`useSourceUpload.test.ts`の`toEqual`3箇所に`filePath: null`を追記した。
- **`useGenerationForm.ts`**: 既存の内部stateだった`audioDurationSec`を返り値へ公開し、`referenceVideoDurationSec`を新設した（参照動画がready時に`fs.probeMediaInfo`で取得、`durationSec > 0`のときのみ表示、`.catch()`付き、`probedReferenceIdRef`＋`cancelled`/`settled`パターンで古い結果の混線を防止）。**将来の掃除課題（二重機構）**: 音声側の既存スヌープ機構（`audioProbingBridge`/`lastAudioFilePathRef`）は今回無改修で残置した。`useSourceUpload`の`filePath`公開と役割が重複する二重機構になっており、将来どちらかへ寄せる整理が必要（§42.6）。
- **`paramUtils.ts`**: 「12.3s」表記の共通フォーマッタ`formatSecondsLabel(seconds)`を新設した。
- **`shell/thumbnailPlaceholders.ts`（新規）**: `VIDEO_PLACEHOLDER_DATA_URL`（`SourceInputPanel`から移設）と`AUDIO_PLACEHOLDER_DATA_URL`（♫グリフの、同配色64×64 SVG data URI）を集約した。`SourceInputPanel.tsx`はimport置換のみ。
- **`GenerationForm.tsx`**: `ReferenceVideoSection`／`SourceAudioSection`を、Chain SOURCEカード（`SourceInputPanel`）方式へ全面置換した。カード全体をドラッグ&ドロップ受け入れエリア化（`useFileDrop`直結・`source-section-dragover`ハイライト）、右上に🔁（ファイルあり時のみ活性）と📁を縦積みの`icon-action-button`で配置、`keyframe-thumb`サムネイル（uploading→スピナー／ready→SVGプレースホルダ／idle→「—」）、ファイル名＋尺表示。現状カードが表示していた情報（`icLoraNote`・各種警告・排他ノート・control LoRAドロップダウン・強度スライダー・参照強度スライダー2本・`tooShortWarning`等）は全数維持した。A2Vの`dropResetToken`は`resetErrorSignal`直結にしてClear時のエラー消去挙動も維持。アコーディオンの自動展開機構（§41.8の`useLayoutEffect`）は不変更。
- **i18n `strings.ts`**: 新設2件——`create.referenceVideo.videoPlaceholderAlt`（en「Reference video」／ja「参照動画」）・`create.sourceAudio.audioPlaceholderAlt`（en「Audio file」／ja「音声ファイル」）。削除3件——`dnd.dropHint`・`dnd.dropActive`（オーナー判断①でD&D案内文を廃止）と、死にキー`create.sourceAudio.choosingButton`（いずれもen/ja両方）。
- **削除（引き算）**: `shell/DropZone.tsx`・`DropZone.test.tsx`・`DropZone.css`（本番利用がGenerationFormのみになりデッドコード化。オーナー判断②）。共有フックの`useFileDrop`は存続。
- **新規テスト**: `GenerationForm.referenceVideo.test.tsx`（9件）・`GenerationForm.sourceAudio.test.tsx`（8件）。

### 42.5 最終テスト・ビルド結果

- **typecheck**: `npm run typecheck`（`tsc -b`）エラー0。
- **vitest**: **1082→1091件pass／10件skip（78ファイル）**。新規17件追加、`DropZone.test.tsx`削除で8件減。既存の回帰確認対象4テスト（`GenerationForm.accordion`・`AppShell.controlLora`・`App.prefill`・`SourceInputPanel`）は無改修で合格。
  - **注記（flaky）**: 初回の全体実行で`App.prefill.test.tsx`の音声抽出ノート系1件が、並列負荷由来のタイミングで一度だけ落ちた（flaky）。単独実行では49/49合格、全体再実行でも合格しており、ロジックの退行ではない。
- **native doctest**: 258件pass／6件skip。
- **埋め込みフルビルド**: `scripts/build.ps1 -RunTests`（埋め込み既定）は全工程成功。`NZVIDEOMNI_WEBUI_INDEX`の埋め込み成立をタイムスタンプ連鎖と`webui_embedded.rc`で確認。成果物は`build\ninja-release\NzVideomni.aux2`。
- 未コミット（コミット可否はオーナー判断）。`deploy.ps1`は未実行。

チェックリストは[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-2へ追加した（後述§42.7でオーナー実機目視合格・クローズ済み）。尺表示の実機挙動（`fs.probeMediaInfo`の`get_media_info`実呼び出し）はAviUtl2上でのみ確認できるため、実機確認項目として残していた。

### 42.6 残る将来の掃除課題

- **音声filePathスヌープの二重機構（§42.4）**: `useGenerationForm.ts`の既存スヌープ機構（`audioProbingBridge`/`lastAudioFilePathRef`）と、今回`useSourceUpload`へ追加した`filePath`公開が、役割の重複する二重機構になっている。今回は動作を壊さないため既存機構を残置したが、将来どちらかへ一本化する整理が望ましい。α版スコープでは実害が無いため据え置き。

### 42.7 デプロイとオーナー実機目視合格（2026-07-20）

`deploy.ps1`を実行して実機へ配置した（実機配置完了）。オーナーが実機（AviUtl2上）で目視確認し、①カードの見た目・操作感（両カードのボタン🔁📁・サムネイル・ファイル名・尺表示・カード全体へのドラッグ＆ドロップ、音声⇔参照動画の相互排他時のカード無効化＝ドラッグ＆ドロップの排他含む）、②尺表示の実機挙動（`fs.probeMediaInfo`による「12.3s」表記、大容量動画添付時のDURATIONクランプ動作含む）の**2項目とも合格**とした。[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-2へ起票していた2項目は§3-24へ集約・除去した。

### 42.8 IC-LoRA×A2Vの相互排他の経緯調査（結論、2026-07-20〜21）

§42のカードリデザインで音声（A2V）と参照動画（IC-LoRA）のカードを並べて扱ううちに、オーナーから「IC-LoRAでポーズを制御しているキャラクターにA2Vでリップシンクさせたい（ポーズ制御とリップシンクの同時適用）」というユースケースが提示された。ところがフロントエンドは現在この2つを相互排他にしている。これがバックエンド由来の制約なのか、フロント専用の設計判断なのかを裏取りした結果、**フロント専用の設計判断であってバックエンド制約由来ではない**という結論に至った。以下がその調査結論である。

- **バックエンドは併用対応済み（確定）**: バックエンド側コミット`3f29776`（2026-07-11、A2V＋LoRA併用解禁）→`7c50c32`（2026-07-12、A2V＋参照動画つきIC-LoRAのチェーン対応、`clips=1`限定）で、併用が正規に解禁されている。同日GPU実機ゲートにも合格しオーナー受容済み（バックエンド[`../../../Docs/VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §34.7、pose-control×3＋canny-control×1、1280×768／201フレーム、ピークVRAM 9.5GB）。
- **スキーマ上も相互排他ではない（確定）**: バックエンド`api/models.py`の`GenerateChainRequest`バリデータで排他なのは「`source_audio`×`source_video`」「`reference_video_id`×`source_video`」の2組であり、**`source_audio`と`reference_video_id`は相互排他ではない**。両者に共通して課されるのは「`clips=1`限定」という同一条件だけで、音声＋参照動画の同時送信は最初から通る。
- **Gradio原典は実際に併用送信している（確定）**: Gradio原典（バックエンド`gradio_ui/handlers.py`の`generate`）は、`src_audio`添付時に`use_adapter`（参照動画IC-LoRA）を**同じ1クリップchainペイロードへ合流**させている（`build_a2v_chain_payload`に`reference_video_id`／`control_adherence`／`reference_strength`を配線）。フロントの併用送信もこの1クリップchain方式に倣うのが正道である。
- **フロントの排他はフロント専用判断だった（確定）**: フロントの相互排他は`45fc836`（2026-07-15、単発A2VのCreate移設）で「曖昧な第4モードを作らない」というフロント専用の設計判断として付随的に入り、`80e99bd`（2026-07-17）で双方向化された（本DEVLOG §10.2の三者排他メモがこの経緯を反映している）。バックエンドの制約を写したものではない。
- **誤解の元になっていた古いコメント（記録）**: バックエンド`gradio_ui/ui.py` L560-567付近に、併用解禁**前**（`3f29776`時点）の「reference-video CONTROL adapter cannot combine」の趣旨の古いコメントが放置されている。これは`7c50c32`で制約が撤廃済みの現状を反映していない残骸であり、判断の根拠にしてはならない。

**尺の非対称性（併用時の設計根拠）**: A2V音声とIC-LoRA参照動画では、素材尺が生成フレーム数に対して不足／超過したときのバックエンド挙動が非対称である。A2V音声は不足＝422でハード拒否（フロントの`audioLengthPrecheck`でも事前ブロック）・超過＝切り詰め。IC-LoRA参照動画は不足＝エラーなしのサイレントな制御劣化（品質のみ低下、長さのハード制約が`api/models.py`に存在しない）・超過＝先頭`numFrames`分に切り詰め。この非対称性ゆえ、併用時のDURATIONはA2V側の計算値を優先し、参照動画が生成尺より短いケースはハード拒否ではなくソフト警告で知らせるのが実挙動と整合する。

**§3-18判定の訂正（重要）**: 2026-07-19に[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-18でクローズした「A2V＋参照動画制御LoRAの併用」判定——「UIに送信の入口が無く本ツールのスコープ外」「チェックリストがGradio側の能力を誤って転記していた」——は、上記調査により**事実誤認だったと判明した**。Gradioは実際に併用可能であり、「Gradioの能力を誤って転記」したのではなく、当時のフロント側の相互排他（フロント専用判断）を根拠にスコープ外と結論づけたのが誤りだった。§3-18には2026-07-21付の訂正注記を追記し、既存記述は履歴として残置した。

**結論と次アクション**: フロントの排他を解除してGradio原典と同じ1クリップchain送信経路へ合流させる改修を、[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-1「IC-LoRA×A2V併用解禁」の新チケットとして起票した（オーナー承認済み、次セッション着手予定）。要件・実装箇所マップ・検証条件の正本は[`A2V_ICLORA_COMBO_WORKORDER.md`](A2V_ICLORA_COMBO_WORKORDER.md)にまとめてある。**本セッションはドキュメント整備のみで、実装は次セッションで行う。**

## 43. 仮オブジェクトの尺を自動計算DURATIONへ揃える改修（`prefillSeed.ts`新設、2026-07-21）

右クリックで置かれる仮オブジェクト（生成待ちを表すタイムライン上のプレースホルダー）の長さ（リボン尺）が、パネルに表示される自動計算DURATIONと食い違う——「仮オブジェクトの長さが変わらない」というオーナーの実機報告を起点に、原因調査から実装・デプロイまでを行った。

### 43.1 経緯（初回調査の誤りと再調査）

- **オーナー実機報告**: 素材を右クリックして仮オブジェクトを置いても、その長さが素材やDURATION設定に応じて変わらず一定に見える。
- **初回調査の誤り**: 最初の調査は「直近の改修（`a3bbb5f`右クリック第2段階など）は仮オブジェクトの予約経路に非干渉」で止まった。これに対しオーナーから「既存機構がデッドコード化している可能性を調べていない」という正当な批判があった。
- **git考古学による再調査（確定）**: 予約経路の歴史をコミット単位で遡ったところ、**「素材系の予約長を素材尺に揃える機構は歴史上一度も存在しなかった」**ことが確定した。仮オブジェクト挿入RPC（`timeline.insertProvisional`）を導入した`727a8d5`の時点から、素材系（リマウント方式）の予約長は`config`既定のDURATIONで固定であり、素材尺へ揃える機構はデッドコードとしてすら存在していなかった。つまり「壊れた／死んだ機構」ではなく「元から無かった」。
- **乖離が顕在化した理由**: `a3bbb5f`（右クリック第2段階）で`deriveDuration.ts`が新設され、**フォーム側のDURATIONだけ**が素材の実尺を反映するようになった（§41.6）。予約側は従来どおり`config`既定のままだったため、「フォームは素材尺に追従するのに仮オブジェクトのリボンは固定」という乖離が表面化した。この乖離は`AppShell`内コメントに「既知の受容ギャップ（accepted divergence）」として当時から明文化されていた（予約長は`config`既定にフォールバックする、と）。
- **調査の教訓**: 「直近改修が予約経路のコードを触っていない＝無関係」と早合点したのが初回調査の誤りだった。実際には直近改修（`deriveDuration`新設）がフォーム側の挙動を変えたことで、**従来から`config`既定固定だった予約側との差が初めて見えるようになった**という間接的な因果だった。「触っていない＝影響なし」ではなく、片側の変化が既存の非対称を顕在化させる経路を疑うべきだった。

### 43.2 オーナー確定要件

1. **配置時**: 仮オブジェクトを置く瞬間に、自動計算DURATION（`deriveDuration`の結果）でリボンを揃える。
2. **Generate時**: ユーザーがDURATIONを手で変えていた場合も含め、Generate実行時点の実DURATIONへ仮オブジェクトの長さを調整する。

### 43.3 実装（プランモード＋敵対的レビュー通過）

プランモードで計画し、敵対的レビュー（致命的・要修正いずれもゼロで通過）を経て実装した。

- **単一ソースの純関数`resolvePrefillSeed`（新規`webui/src/timeline/prefillSeed.ts`）**: 右クリックプリフィルの幅・高さ・fps・DURATION（`num_frames`）を1か所で決める純関数を新設した。従来は`CreateScreen`／`ChainScreen`がそれぞれ`deriveGenerationParams`＋`computeTargetNumFrames`のuseMemo群でフォームのシードを計算し、一方`AppShell`の予約経路は`config`既定へフォールバックしていた（両者が同じ右クリックから別々の値を導けてしまう構造）。`resolvePrefillSeed`をフォームシードと予約長の**両方の単一ソース**にすることで、両者が乖離できない構造にした。intentごとのDURATION方式表`DURATION_POLICY_BY_INTENT`（全intentを網羅）と、intent別の素材尺決定`materialDurationForIntent`をこの純関数に集約している。既存の凍結シーム（`deriveGenerationParams`＝幅・高さ、`computeTargetNumFrames`＝`num_frames`）は再実装せず合成するのみで、`deriveDuration.ts`自体は本モジュールをimportしないプリミティブに保ち循環依存を避けた。
- **予約経路（`AppShell`）**: 素材系項目（#1/#2/#3/#4/#6/#7/#12）の予約を、`timeline.insertProvisional`へ**最初から**`resolvePrefillSeed`由来の`numFrames`／`genFps`を渡して行う。置いた後に長さを直す二度手間（RPC2回・ちらつき）ではなく、RPC1回で正しい長さに置く。[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §5-2の「DURATIONのリアルタイム追従はしない（配置後にパネルのDURATIONを変えても仮オブジェクトは追従しない＝アンドゥ履歴汚染の回避）」原則には**非抵触**——追従させるのではなく、配置の初期値を正しくするだけだからである。
- **intent別のduration入力（`materialDurationForIntent`）**: #2 参照動画・#7 音声a2vは**素材フルファイルの`mediaDurationSec`**（`get_media_info`の`total_time`、アップロード実体と一致）、#3 動画音声a2vは**タイムラインのspan長**（`frameEnd - frameStart + 1`フレームをプロジェクト`rate`/`scale`で秒へ換算）、その他（`comfortCeiling`系）は素材尺を使わず解像度の快適上限。音声の`mediaDurationSec`が`0`（不明＝probe失敗）のときは、でっち上げの0秒クリップとせず`undefined`を返して`config`既定DURATIONへ穏当にフォールバックする。
- **Generate時の再スタンプ**: 要件②は既存の`bindToJob`が`timeline.updateProvisionalReservation`で仮オブジェクトを本jobIDと確定長へ貼り替える経路（§5-3 用途a）で充足済み。ここはコードの新規挙動ではなく、テストで固定して「G4で未検証だった穴」を埋めた。
- **`CreateScreen`のproject上書きも統一**: Settings「プロジェクトに合わせる」時の`getEditInfo`による非同期上書き（§5-13）が使うduration入力も、同じ`materialDurationForIntent`（#3のspanセマンティクス）を経由するよう`resolvePrefillSeed`経由に統一し、シード時と上書き時で計算がずれないようにした。
- **ネイティブ・`mockBridge`は無変更**: `insertProvisional`は元から`numFrames`/`genFps`パラメータを受け取れる形（contract v5）だったため、渡す値を変えただけでネイティブ側の変更は不要。

### 43.4 既知の設計挙動（仕様として記録）

- **projectポリシー時の一時乖離**: Settings「プロジェクトに合わせる」では、`getEditInfo`のasync上書きがフォームのDURATIONを一拍遅れて書き換えるため、配置直後の仮オブジェクトのリボンと最終的なフォームDURATIONが短時間だけ食い違う。この乖離はGenerate時の再スタンプで収束するため許容とする（バグではない）。
- **A2Vフォームの1回ジャンプ**: A2V（#3/#7）は、配置時の初期シードが「素材の推定値」で、その後wavを実測した`suggestFramesForAudio`の値へ1回だけDURATIONがジャンプする。
- **#7はフルファイル長**: #7 音声a2vはトリム済み音声を扱う場合でも、予約の初期尺はフルファイル長（アップロード実体＝`mediaDurationSec`と一致）を使う。#3がタイムラインspan長を使うのと対照的だが、いずれも各intentの素材の意味づけに沿っている。

### 43.5 テスト・ビルド・デプロイ

- **typecheck**: `npm run typecheck`（`tsc -b`）エラー0。
- **vitest**: **1117件pass／10件skip（79ファイル）**。+26件（既存テストの改変はゼロ、追加のみ）。
- **native doctest**: 258件pass維持（ネイティブ無変更）。
- **埋め込みビルド**: `scripts/build.ps1`の埋め込みビルド成功。
- **デプロイ**: `deploy.ps1`実行済み（2026-07-21、実機配置完了）。オーナーが手動でコミット＆プッシュを実施済み。

### 43.6 台帳への反映

実機確認項目を[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-2へ起票した（配置時のリボン長・DURATION手動変更後のGenerate追従・Settings3択の挙動差・音声`mediaDurationSec`の実機可否）。[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §5-1／§5-12の「素材系の予約長は`config`既定」という旧記述は、本改修後の姿（`resolvePrefillSeed`由来の導出長で置く）へ整合させた（§5-2の追従禁止原則は維持）。

## 44. IC-LoRA×A2V併用解禁とV2V結合（Join）機能の復活（増分I1〜I6、2026-07-21）

2つのテーマを増分I1〜I6として実装した。テーマ1（I1/I2）は「参照動画（IC-LoRA）と音声（A2V）の同時添付」の解禁（フロントのみ）、テーマ2（I3〜I6）はJoin機能の復活（末尾トリム方式・表示バグ修復を含む・バックエンドは凍結の限定解除）。いずれもプランモードで計画し敵対的レビューを通してから着手した。正本は[`A2V_ICLORA_COMBO_WORKORDER.md`](A2V_ICLORA_COMBO_WORKORDER.md)（テーマ1）と[`JOIN_FEATURE_RESEARCH.md`](JOIN_FEATURE_RESEARCH.md)第4部（テーマ2）。

### 44.1 テーマ1 IC-LoRA×A2V併用解禁（I1/I2）

- **三方排他の解禁**: `useGenerationForm.ts`の相互排他を、`isA2v = audioReady`（音声が付けばA2V。参照動画の有無とは独立）へ変更した。従来は「音声⇔参照動画」を双方向に無効化していた（2026-07-15のフロント専用判断）が、バックエンドは併用対応済み（`GenerateChainRequest`スキーマ上`source_audio`と`reference_video_id`は排他ではなく、両者に課されるのは`clips=1`という同一条件のみ）であることが§42.8の調査で確定していた。
- **mergedLoras切替**: `buildA2vRequest`が`mergedLoras`（制御LoRAの選択＋プロンプトのSTYLEタグを`combineLoras`で統合・名前でdedup）を使い、N3ゲート（`reference_video_id !== null && loras.length > 0`）を満たすときだけ参照フィールド（`reference_video_id`・強度2種）を1クリップchainペイロードへ同梱する。ペイロード層[`buildA2vChainPayload.ts`](../webui/src/modes/batch/buildA2vChainPayload.ts)は無改修——既存の`use_adapter`分岐がそのまま効くため。Gradio原典の送信経路（`src_audio`添付時に`use_adapter`を同じ1クリップchainへ合流）と同じ方式。
- **UI**: 両picker（音声・参照動画）のdisabled排他を撤去。併用ノートは既存2キー（`blockedByAudioNote`/`exclusiveWithReferenceNote`）を**削除せず文言変更で存続**させ、両方入力時に両カードへ「併用可」の趣旨で表示する。ソフト警告`icLoraSpillWarning`（生成尺>参照尺・warning-banner-mild・isValid非連動・numFrames自動調整後の値で評価）を新設。モードバッジは三項式（排他前提の1枚選択）から**独立判定2枚**へ変更し、併用時は「IC-LoRA」「A2V」を併記する。

### 44.2 テーマ2 V2V結合（Join）復活（I3〜I6）

- **表示バグの根本修復（I4）**: Joinボタンが実装以来一度も実機で表示されなかった真因は、フロントが`job.request.source_video`の有無でv2vを判定していたのに対し、実サーバーの`GET /jobs`は`to_clip_request`のホワイトリストで応答を組むため`source_video`が構造的に含まれないことだった（§39で特定）。判定を`job.is_v2v`ベースへ切り替え、あわせて**mockBridgeのエコーバックを是正**した（リクエストボディ丸ごと返しをやめ、実サーバーの`to_clip_request`相当のホワイトリスト化——チェーンジョブでは`loras`空・`reference_video_id` null・`source_*`/`clips`/`overlap_*`を落とす）。これで「テスト緑・実機不可視」の乖離を構造的に潰した。実サーバー形状での表示検証統合テスト`joinVisibility.test.tsx`を新設。
- **バックエンド（凍結の限定解除・生成エンジン非接触）**: `JoinRequest.source_tail_seconds`（tail-keep秒・既定5.0・0=全長連結）、`JoinResponse.trimmed_source_seconds`/`source_fps`、`JobResponse.is_v2v`（`chain_request.source_video`由来）/`joined`（`joined.mp4`ファイル存在ベース）を追加。`join()`はprobe後・normalize前に`cut_tail_mp4`でtail-keepトリム（ソース実測fpsを渡す＝リサンプル無し）し、出力はtmp書き→`os.replace`のatomic renameで差し替える。
- **Join UI（I5）**: Join/Unjoinトグル（初期値はマウント時のみ`job.joined`参照・ポーリング追従は`true→false`の片方向のみ・Unjoinはサーバー非接触の表示トグル・再Joinは上書き）。トリム長ラジオ120f/240f（生成動画fpsで秒換算し`source_tail_seconds`送信・秒併記）。音声クロスフェード（150/300/500ms 3択維持・ラベルを「音声クロスフェード」へ正直化）。fps注意文`fpsMismatchNote`（source_normalized または プロジェクトfps≠生成fpsのときのみ）。新規純関数[`jobs/fpsConvert.ts`](../webui/src/jobs/fpsConvert.ts)。
- **統合（I6）**: プレビューは併存→**差し替え**（JobCardが単一プレビュー・JoinControlsは制御UI専任）。🎞は表示中クリップを挿入し、joined表示中は位置付き挿入＋`timeline.deleteProvisionalByJob`で仮オブジェクト自動削除。新設[`timeline/sourceLocationMap.ts`](../webui/src/timeline/sourceLocationMap.ts)（jobId→元動画位置の揮発マップ。記録=右クリックextend-video時・rekey=bindToJob時・リロード消失は位置指定なし挿入へ縮退）。挿入位置＝`frameEnd + 1 − round(keep実長秒 × プロジェクトfps)`（frameEndはinclusiveとネイティブ現物で確定）・レイヤーは`layerMax+1`。

### 44.3 主要な設計判断

- **秒ブリッジ**: 元動画・プロジェクト・生成動画のfpsが三者三様になりうるため、フレーム計算は必ず秒を経由して丸める（`fpsConvert.ts`に集約）。トリム長も「フレーム2択→生成fpsで秒換算→`source_tail_seconds`」と秒で持ち回る。
- **atomic rename**: `joined.mp4`の存在＝Join済み判定なので、連結途中でクラッシュしても壊れた中間ファイルが「存在」扱いされないよう、tmp書き→`os.replace`で原子的に差し替える。
- **片方向追従**: `joined`のポーリング追従は`true→false`のみ（ファイル消失で[Join]復帰）。`false→true`は追従しない——ローカルUnjoinした表示を遅延ポーリングが勝手に巻き戻すのを防ぐ。
- **ホワイトリスト化**: mockBridgeを実サーバーの`to_clip_request`挙動へ寄せたことが、表示バグ再発防止の本丸。エコーバックのままだと同種の乖離が別項目でも再発し、テストでは永遠に検出できない。

### 44.4 敵対的レビューで事前に潰した欠陥（教訓）

- **クロスフェードの意味づけ**: 当初要件は「フレーム単位スライダー」だったが、実装前に現物のffmpegコマンドを確認したところ`handle_crossfade_ms`は音声（`acrossfade`）にしか効かず映像は常にハードカットと判明。フレーム単位化は無意味なので撤回し、3択ミリ秒維持＋ラベル「音声クロスフェード」へ正直化した（[`JOIN_FEATURE_RESEARCH.md`](JOIN_FEATURE_RESEARCH.md) §4.2）。「要件を実装する前に現物を確認する」で無駄実装を1つ回避した。
- **`frameEnd`のinclusive/exclusive**: 挿入位置式は`frameEnd`の解釈でオフバイワンを生みやすい。ネイティブ現物で「`frameEnd`はinclusive・実尺は`frameEnd − frameStart + 1`・末尾の次は`frameEnd + 1`」を確定してから式を組んだ。
- **既知の縮退を明示**: 位置マップが無い3ケース（Chain画面手動v2v・リロード後・mount復元joinedで`trimmed`不明）は、でっち上げの位置ではなく位置指定なし挿入へ穏当に縮退させ、仮オブジェクト自動削除も行わない設計にした。

### 44.5 テスト・ゲート

- **webui typecheck**: `npm run typecheck`（`tsc -b`）エラー0。
- **webui vitest**: 82ファイル・1159件pass／10件skip。テーマ1の通しE2E（制御LoRAのみ＝STYLEタグ無し＋音声＋参照動画を`buildA2vRequest`→mockBridgeへ送信→completed→`GET /jobs`のホワイトリスト応答検証）1本を`useGenerationForm.test.ts`へ追加した（本増分I7の仕上げ）。テーマ2は`mockBridge.test.ts`・`joinVisibility.test.tsx`・`JoinControls.test.tsx`・`JobCard.test.tsx`で通し検証済み。
- **backend pytest**: Join関連の追加6件を含め該当分緑。
- **既存の無関係failing test（今回発覚・未修正）**: backend `tests/test_logging.py::test_build_uvicorn_log_config_attaches_filter_without_mutating_default` はHEAD時点（本増分の変更前）から失敗する古いテスト。§13.5時代のアクセスログフィルタ増加（2個→4個）に未追随。凍結範囲外のため未修正、[`PENDING_TASKS.md`](PENDING_TASKS.md) §4-17へ起票（要オーナー判断）。

### 44.6 台帳・ドキュメントへの反映

実機確認チェックリストを[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-4（テーマ1）・§2-5（テーマ2）へ起票し、旧§1-1／§1-2はクローズ（§3-25／§3-26に完了宣言）。[`JOIN_FEATURE_RESEARCH.md`](JOIN_FEATURE_RESEARCH.md)第4部を実装後の確定内容へ改訂（§4.2案A確定・§4.3〜§4.5に実装形・§4.7残論点を全解消）、[`A2V_ICLORA_COMBO_WORKORDER.md`](A2V_ICLORA_COMBO_WORKORDER.md)冒頭に実装完了の歴史記録を追記、[`BRIDGE_CONTRACT.md`](BRIDGE_CONTRACT.md)・[`API_REFERENCE.md`](API_REFERENCE.md)へJoin/getEditInfo関連の追記を行った。オーナー実機ゲート待ち。

## 45. Join実機検証で発覚した2バグの修正（トリムラジオのname衝突・fps注意文の分岐化、2026-07-21）

### 45.1 経緯と真因

§44のJoin機能はオーナー実機検証（[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-28）で機能面は全項目合格したが、2件の表示バグが発覚した。

- **トリムラジオの選択消失**: トリム長ラジオが「選択なし」の状態を取り得て、選択しても2秒（ポーリング周期）で消える。真因は**重複ラジオname**。ラジオの`name`が`join-trim-{job_id}`とジョブ単位でしか一意でない一方、AppShellはCreate/Chain/Libraryの3画面を常時マウント（§24の設計）しており、同一ジョブの`JoinControls`が最大3インスタンス同時にDOMへ存在する。HTMLのラジオグループは`<form>`外ではdocument全体で`name`単位に合流するため、計6個のラジオが1グループになり、非表示インスタンスにチェックを奪われて可視側が未選択表示になる。毎ポーリングの再レンダーが衝突を再発火する。なお送信値はReactのstate（既定120）から読むため、見た目が未選択でもJoin自体は正しい値で成功していた（実害は視覚のみ）。
- **fps注意文の誤情報**: 検証中に「project 24 / video 24 なのに did not match」という矛盾した注意文が表示され、オーナーがJoin失敗と誤認する事態になった（実際はJoin成功。エラー時は[Retry join]表示になる実装のため、[Unjoin]へ遷移した時点で成功が確定する）。真因は2つ。①サーバーの正規化判定`needs_norm`は**解像度差でも**発火するため、fps全一致でも旧文言「The video frame rates did not match」が表示される。②実際にfpsが食い違ったケース（ソース30fps vs 生成24fps）でも、旧文言は当事者のソースfps（`JoinResponse.source_fps`）を出さず、無関係な「(project / video)」の2値だけを表示していた。

### 45.2 修正内容（フロントのみ・承認済みプランどおり）

- **ラジオ**: `JoinControls.tsx`にReactの`useId`を導入し、`name`を`join-trim-{インスタンス一意ID}`へ変更。3画面並存でもグループが合流しない。
- **注意文の3部品化**: 旧`strings.jobs.fpsMismatchNote`を廃止し、発火理由別に分岐——(a)ソースfpsと生成fpsの差>0.001なら`joinSourceFpsNote`（ソースfpsを明示して"did not match"）、(b)差がないのに`source_normalized: true`なら`joinNormalizedNote`（fpsに言及せず「生成動画の形式に合わせて再変換した」）、(c)プロジェクトfpsと生成fpsの差>0.001のときのみ推奨行`joinProjectFpsAdvice`を併記。本文も推奨行も無ければ注意領域自体を出さない。本文はプロジェクトfps不明（`getEditInfo`失敗）でも表示するよう改善（従来は全非表示）。
- **fps表示の丸め**: `fpsConvert.ts`へ`formatFps`（小数第2位まで・末尾ゼロ除去）を新設。**比較は生値（許容差0.001）・表示のみ丸め**で統一し、29.97系プロジェクトで丸め表示後に見かけ同値の推奨行が出る事故を防ぐ。
- **mockBridge**: テスト用に`joinSourceNormalized`／`joinSourceFps`オプションを追加（既定値は現行応答と同一で既存テスト無影響。`joinSourceFps`は明示`null`を尊重する実装）。

### 45.3 テスト・ゲート

`JoinControls.test.tsx`へ6ケース（fps差あり本文・fps差なし正規化のみ・推奨行の出し分け・注意領域なし・projectFps不明でも本文表示・2インスタンスのラジオ独立性=バグ①の回帰テスト）、`fpsConvert.test.ts`へ`formatFps`3ケースを追加。既存「I5 ⑥」テストは新仕様へ書き換え。`npm run typecheck`エラー0・vitest 82ファイル1165件pass／10件skip（+6）。埋め込みフルビルド→`deploy.ps1`で実機配置済み。オーナー実機ゲート（旧§2-6）は2026-07-21に全4項目合格した（結果は[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-29）。

## 46. フロントエンド微調整バッチ（W1〜W9、2026-07-22）

オーナーの使い勝手フィードバックを起点に、右クリックまわりとSettings・生成フォームの細かな改善を1つのバッチ（W1〜W9）としてまとめて実装した。いずれもフロントエンド中心の小改修で、バックエンドはテストファイル1点の修正（W8）を除いて非接触（凍結方針の維持）。着手前にプランモードで計画し、敵対的レビューを通してから実装した。W9は本ドキュメント更新そのものである。テーマごとの正本は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md)（第12版、W1〜W5）と[`PENDING_TASKS.md`](PENDING_TASKS.md)（§2-4の確認チェックリスト・§3-30/§3-31のクローズ記録）に置いた。

### 46.1 W1 Settings「RIGHT-CLICK SIZE & FPS」の2軸分離

- **経緯**: 従来、右クリックプリフィルのサイズとFPSは1つの3択（既定値／プロジェクトに合わせる／素材に合わせる〔既定〕）でまとめて決めていた。しかし「サイズは素材に合わせたいが、FPSはプロジェクトに合わせたい」という軸別の要求に応えられなかった。
- **実装**: サイズ用・FPS用の独立した2つの3択（2軸）へ分離した。選択肢「Defaults／既定値」は「Developmental／開発用」へ改名。新しい既定値は**サイズ＝素材に合わせる・FPS＝プロジェクトに合わせる**。`localStorage`キーは新設2つ（`nzvideomni.prefillSizePolicy`／`nzvideomni.prefillFpsPolicy`）で、旧キー`nzvideomni.prefillResolutionPolicy`は読み捨てる。CreateScreen/ChainScreenの「プロジェクトに合わせる」上書き`useEffect`は軸別に分割し、交差ケース（サイズ＝project × FPS＝material 等）ではDURATION再計算を「projectの軸はプロジェクト値・非projectの軸はフォームのシード値」で行う。詳細は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §5-13。

### 46.2 W2 仮オブジェクト右クリックからの簡易挿入（⬇、#13）

- **実装**: オブジェクトメニューに9項目目「⬇ Insert this generated result now」（日本語「⬇ この生成結果を今すぐ挿入」、アクションID`insertProvisionalResult`）を新設した（[`PENDING_TASKS.md`](PENDING_TASKS.md) 旧§4-18の格上げ）。右クリックされたオブジェクトのオブジェクト名から`NzVideomni#<ジョブID>`を抽出し、完了済みなら操作パネルの🎞ボタンと同じ置換挿入（仮オブジェクト自動削除。§5-10の`timeline.insertMediaForJob`経路）、実行中なら「完了を待て」の警告トースト、失敗/キャンセル済み・非仮オブジェクト・台帳に無い場合もそれぞれトーストで案内する。
- **契約**: 新RPCなし。`timeline.menuInvoked`の`action`に`insertProvisionalResult`が1つ増えただけで、ブリッジ契約バージョンは不変（[`BRIDGE_CONTRACT.md`](BRIDGE_CONTRACT.md) §4.14）。

### 46.3 W3 レイヤー右クリック「最新の生成結果をここに挿入」（⬇、#14）

- **実装**: レイヤーメニューに5項目目「⬇ Insert the latest generation result here」（日本語「⬇ 最新の生成結果をここに挿入」、アクションID`insertLatestResultHere`）を新設した（[`PENDING_TASKS.md`](PENDING_TASKS.md) 旧§4-19の格上げ）。最新の完了ジョブ（`completed_at`降順・`null`は最古扱い）の動画を、右クリック位置（レイヤー・フレームとも）へ素の`timeline.insertMedia`で挿入する。**仮オブジェクトは消さない**（消えるのは操作パネルの🎞と#13だけ、というルールを維持）。完了ジョブ0件は警告トースト。`downloadAndInsert.ts`に位置指定の素挿入オプション`plainInsertAt`を追加した。
- **契約**: 新RPCなし（同上、`action`が1つ増えただけ）。

### 46.4 W4 #2/#7の尺のトリム追従＋#2最短尺ガードのspan化

- **実装**: 素材系右クリック#2（この動画を参照にIC-LoRA）・#7（この音声からa2v）のDURATION自動決定を、#3（この動画の音声でa2v）と同じくタイムライン上のトリム済みspan長（トリムされた後の長さ）基準へ統一した（従来はファイルのフル尺。[`PENDING_TASKS.md`](PENDING_TASKS.md) 旧§4-20の解消）。仮オブジェクトのリボン長も配置時からspan長になる。共通ヘルパ`spanDurationSec`を新設。#2の「参照動画が短すぎる」ガードもspan長基準へ（#1はフル尺のまま）。
- **#7のwavキャップ**: #7は音声ファイルをフル尺でアップロードするため、wav実測からのFrames自動調整がspanシードを上書きしないよう、**プリフィル由来の音声に限り**提案値をspan由来の上限（`a2vSeedMaxFrames`）でキャップする。音声を差し替えるとキャップは解除される。詳細は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §5-12追記。

### 46.5 W5 反対スロット保持

- **実装**: #2/#3/#7のプリフィルはリマウント（フォーム初期化）を維持しつつ、反対側スロットの素材だけ（#2実行時→既存音声、#3/#7実行時→既存参照動画＋強度2種）をアップロード済みIDの再利用で引き継ぐようにした。`publishSourceSlots`チャネル（`provisionalReservation.ts`）と`GenerationPrefill.carryOver`を新設。回帰対策として、参照動画carry時は128グリッド初期化（`seedICLora`）、音声carry時はwav自動調整の発火（`lastAudioFilePathRef`初期化）を実装した。uploading中のスロットはcarryしない。
- **既知の制約**: `isICLoraIntent`がマウント時凍結のため、#7に参照動画をcarryした後で参照動画を外しても128グリッドのまま（一般グリッドへ戻すにはタブ再マウントか❌ボタン）。DURATIONは両方セット時A2V優先、仮オブジェクト位置は後勝ち（いずれも現行設計のまま）。[`PENDING_TASKS.md`](PENDING_TASKS.md) §4-22・[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §5-15に記録。

### 46.6 W6 IC-LoRAカードの❌ボタン／W7 Generate不可理由ノート

- **W6**: Reference video（IC-LoRA）カードの🔁の隣に❌を追加した。押すと参照動画クリア＋制御LoRAドロップダウンをnoneへ＋強度2種リセット＝IC-LoRAを全く使わない状態へ戻す。参照動画・制御LoRA・強度のどれも無いときのみ無効。文言キー`removeIcLoraButton`。
- **W7**: Generateボタンが押せないとき、ボタン直下に「押せるようになるための命令形の指示」を複数行で表示する（warning-banner-mild様式・新規`GenerateReasonsNote`コンポーネント）。`useGenerationForm`／`useChainForm`の妥当性判定を理由コード配列`validityReasons`へ再構成した（`isValid`＝理由ゼロ、と論理同値）。Create側11コード（`promptEmpty`／`dimensionsOffGrid`／`numFramesOffGrid`／`audioTooShort`／`referenceNeedsLoras`／`controlNeedsReference`／`cropInvalid`＋画面側`keyframeUploading`／`audioUploading`／`referenceNotReady`／`overflowKeyframes`）、Chain側9コード（源泉系3ブールは`sourceUploading`／`sourceNotReady`の1理由へ正規化）。同一文言はde-dup。`submitting`／`hasActiveJob`はボタン文言が変わるため対象外。既存の範囲外キーフレーム単独ノートは理由ノートへ畳み込んだ。

### 46.7 W8 backend test_logging修正

- **実装**: `tests/test_logging.py`のみを修正した（バックエンド凍結の限定解除）。期待するアクセスログフィルタを2個→実装どおり4個へ合わせ、増えた2フィルタの`is`検証を追加。22件全pass。実装コードには一切触れていない。[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-1（旧）の解消（→§3-30）。

### 46.8 オーナー決定事項（操作パネル自動表示の見送り・空カード防壁のクローズ）

- **操作パネルの自動表示は見送り**: 生成起点の右クリック後に操作パネルを自動で前面表示できないか検討したが、AviUtl2 SDKの`plugin2.h`全917行を精査した結果、プラグインウィンドウの表示/アクティブ化APIが存在しないことを確認した。Win32の`ShowWindow`直叩きは本体のドッキング/タブ管理と衝突するリスクがあり、実機検証なしに安全性を保証できないため、α版では見送りとした。[`PENDING_TASKS.md`](PENDING_TASKS.md) §4-21へ将来課題として起票。
- **空カード防壁は現状維持でクローズ**: 画像未設定の空カードが範囲外位置にあるときGenerate最終防壁が止める仕様の是非（旧§2-3の判断待ち）について、範囲外ピンのみブロック・空カードはペイロードに載らない・既にボタン直下に警告ノートが出る現状を確認のうえ、オーナーが現状維持と判断した（[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-31）。

### 46.9 敵対的レビューと逸脱3件

- **敵対的レビュー**: 実装前後のレビューで、要修正4件（R1〜R4）・軽微6件（M1〜M6）を発見し、すべて反映した。
- **逸脱3件（計画からの意図的なずれ）**:
  1. **toast種別にwarningを採用**: 当初は情報トースト（info）を想定していた箇所を、W2の「完了を待て」など注意喚起の性質に合わせてwarning種別へ変更した。
  2. **App.prefill.testの追随**: W5の反対スロット保持でプリフィルの初期化挙動が変わったため、既存の`App.prefill.test`を新仕様へ追随修正した。
  3. **表示テストの純コンポーネント化**: W7のGenerate不可理由ノートの表示検証を、フック全体を通す統合テストではなく`GenerateReasonsNote`単体（純コンポーネント）のテストとして書いた（理由コード配列を入力に与える形にして表示ロジックを分離）。

### 46.10 テスト・ゲート

- **webui typecheck**: `npm run typecheck`（`tsc -b`）エラー0。
- **webui vitest**: 85ファイル・1230件pass／10件skip。バッチ着手前の基準1175件から、各Wの実装とテスト追加で1222件へ、最終仕上げで1230件へと推移した。
- **backend**: `tests/test_logging.py` 22件pass（W8）。実装コード非接触。
- **native**: doctest 258件pass／6件skip（従来基準と一致）。埋め込みフルビルド成功（webui変更はCMakeの`OBJECT_DEPENDS`により自動リビルドされることを差分ビルドの3ステップで確認）。
- **実機**: `scripts/deploy.ps1`による実機デプロイを同日中（2026-07-22深夜）に実施済み（新しい`NzVideomni.aux2`と`Language/`2ファイルの配置をタイムスタンプで確認）。オーナー実機ゲートは未了（テスト待ち）。確認チェックリストは[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-4へ起票した。

### 46.11 台帳・ドキュメントへの反映

[`PENDING_TASKS.md`](PENDING_TASKS.md)へ、確認チェックリスト（§2-4）を新設、`test_logging`修正の完了（§3-30）・空カード防壁のクローズ（§3-31）を記録、旧§4-18/§4-19/§4-20へ格上げ・実装完了の注記、操作パネル自動表示の見送り（§4-21）・反対スロット保持の既知の制約（§4-22）を起票した。旧§1-1（`test_logging`）は§1から除去、旧§2-3（空カード防壁の判断待ち）は§2-3から除去した。[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md)を第12版へ改訂（メニュー9/5項目・#2/#7尺基準・反対スロット保持§5-15・Settings2軸化§5-13）、[`BRIDGE_CONTRACT.md`](BRIDGE_CONTRACT.md) §4.14へアクション識別子2件（契約不変）を追記した。

## 47. フロントエンド微調整バッチ 第2波（X1〜X6、2026-07-22）

第1波（§46、W1〜W9）のオーナー実機ゲートが**全項目合格**（[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-32）した後、その合格を確認するためのテスト中に発覚したバグ2件・改善4件を、第2波バッチ（X1〜X6）としてまとめて実装した。第1波と同じく、フロントエンド（webui）のみの小改修で、バックエンド・ネイティブは非接触（凍結方針の維持。第1波のW8のようなテストファイル修正も今回は無い）。着手前にプランモードで計画し、敵対的レビューを通してから実装した。実機デプロイとオーナー実機ゲートはこれから（別担当が並行でビルド・デプロイ中のため、本記録では「デプロイ実施済み」の断定を避ける）。テーマごとの正本は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md)（第13版、X2b/X4/X5）と[`PENDING_TASKS.md`](PENDING_TASKS.md)（§2-4の確認チェックリスト・§3-32/§3-33のクローズ記録・§4-13の調査結論反映）に置いた。

### 47.1 X1 FPS軸「素材に合わせる」の無効化とFPS素材の調査結論

- **経緯**: 第1波のW1でSettingsの右クリックプリフィルをサイズ用・FPS用の独立した2軸へ分離し（§46.1）、各軸に「開発用／プロジェクトに合わせる／素材に合わせる」の3択を持たせた。しかしFPS軸の「素材に合わせる」は、実体としては選択オブジェクト個別のfpsではなくプロジェクト全体のフレームレートを代用しているだけで（[`PENDING_TASKS.md`](PENDING_TASKS.md) §4-13の既知課題）、名前と挙動が食い違っていた。
- **調査結論**: 真の素材fpsを取得できるか調べたところ、**AviUtl2 SDKの`get_media_info`が返す`MEDIA_INFO`構造体には素材のフレームレート（fps）を表すフィールドが無く、fpsを逆算できるフレーム総数のフィールドも無い**ことが確定した。したがって`get_media_info`の戻り値を`timeline.getSelection`へ流すだけでは実現できず、`get_media_info`拡張では実現不可と結論づけた。真の素材fps取得には入力プラグイン経由で素材ファイルを自前解析する中規模のネイティブ実装が必要になる（詳細は[`PENDING_TASKS.md`](PENDING_TASKS.md) §4-13を今回の結論で更新した）。
- **実装（見送りの明示）**: FPS軸の「素材に合わせる」は機能実装を見送り、Settingsのボタンを**グレーアウトして残す**（撤去はしない）。保存済み設定値が「素材」だった場合は、fps軸を「プロジェクトに合わせる」へ**自動フォールバック**する。サイズ軸は3択のまま（素材実寸は`mediaWidth`/`mediaHeight`で取得できるため）。詳細は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §5-13。

### 47.2 X2 予約詰まりの修正（2段構造・台帳認識reconcileへの設計変更）

- **症状**: Chainでバックエンドが検証エラー（HTTP 422）を返した後、右クリックからの新規生成が「A new reservation cannot be made until the current generation finishes.」（前の生成が完了するまで新しい予約はできません）で恒久的にブロックされ、実際には何も生成していないのに予約席が固着する不具合。
- **真因（2段構造）**: 原因は2つ重なっていた。
  1. **(a) 送信の同期失敗時の掃除漏れ**: 右クリック由来の予約席とタイムライン上の仮オブジェクト（プレースホルダ）を、送信が同期的に失敗したときに掃除する処理が無かった。422で送信が弾かれても、予約席とプレースホルダが残ってしまう。
  2. **(b) 終端ジョブの予約席への誤復元**: 右クリックのたびに走る予約席の再構築処理（`scanProvisionals`系）は、タイムライン上に残る仮オブジェクトのタグを見て予約席を復元する。完了済み（終端）ジョブのタグがタイムラインに残っていると、これを`generating`（生成中）として復元してしまう一方、ジョブ進捗ポーリングの解放通知は「非終端→終端の遷移」でしか発火しないため、初回観測時に既に終端になっているジョブは二度と解放されず、席が固着していた。
- **修正内容**:
  - **(a)への対応**: 送信失敗時に、予約席（`reserved`状態のもののみ）とプレースホルダを掃除する処理を追加した（既存の`deleteProvisionalByJob`を再利用。新規RPCは無し）。
  - **(b)への対応**: 予約席の再構築処理を、**ジョブ台帳（`JobLedger`）で終端になっているジョブのタグは復元対象から除外する「台帳認識（reconcile）方式」へ変更**した。タグ自体は消さない（🎞挿入のマーカー特定機構を維持するため）。
- **設計変更の経緯（敵対的レビューでの棄却）**: 当初案は「初回観測時に既に終端だったジョブを、あとから解放する（後付け解放）」というものだった。しかし敵対的レビューで、この後付け解放案は「解放イベントの発火条件を複雑にし、遷移ベースの通知とスナップショットの二重管理を招く」として棄却された。代わりに、再構築（reconcile）の時点で台帳の終端状態を参照し、終端ジョブのタグを最初から復元対象に含めない**台帳認識方式**へ設計変更した。これにより「復元してから解放する」の2段階を「そもそも復元しない」の1段階へ畳み、状態の一貫性を保った。詳細は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §5-7。
- **維持した挙動**: 実際に生成が実行中のあいだの新規予約ブロック（§5-6の予約席1つルール）は従来どおり働く。今回の修正は「終端ジョブ・失敗送信で不当に固着した席」だけを解放するもので、正しいブロックには手を付けていない。

### 47.3 X3 短いクリップのクライアント側ゲート

- **症状**: Chainで短いクリップ（9〜16フレーム相当）に既定のつなぎ目ブレンド（`overlap_frames`）を指定するとバックエンドが422を返して生成できない（[`PENDING_TASKS.md`](PENDING_TASKS.md) 旧§2-1の制約）。従来はサーバーエラーになるまでユーザーが気付けなかった。
- **実装**: クライアント側でGenerateボタンを無効化し、ボタン直下に命令形ノート「現在のつなぎ目ブレンド設定では、各クリップは{n}フレーム以上が必要です」を表示する未然防止を追加した。必要最小フレーム数 **n＝8×overlap_frames+1**（既定`overlap_frames`＝3なら25フレーム）。つなぎ目ブレンドの値を変えるとnも連動して変わる。サーバー側のエラー表示はフォールバックとして残す。第1波W7のGenerate不可理由ノート機構（`validityReasons`）へ理由コードを1つ追加する形で実装した。

### 47.4 X4 右クリックv2vのクリップ1枚化

- **症状**: タイムラインの動画を右クリックして「この動画の続きを生成」（v2v、#1 `extendVideo`）でChainへ渡すと、操作パネルのクリップカードが2枚になっていた。
- **実装**: 右クリック経路（#1）でChainへプリフィルするとき、クリップカードを1枚にした。操作パネルから手動でソース動画を添付した場合は従来どおりクリップ数を維持する（右クリック経路だけの変更で、手動フローには触れない）。詳細は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §3-4 #1。

### 47.5 X5 併用プリフィルでアコーディオンを閉じない

- **症状**: Create画面で参照動画（IC-LoRA）を入力済みの状態で右クリックから音声（A2V）を送ると、リマウント後にA2Vのアコーディオンだけ開いてIC-LoRA側が閉じてしまっていた（逆順も同様）。第1波W5の反対スロット保持（§46.5）で素材自体は残るのに、アコーディオンだけ閉じるちぐはぐさがあった。
- **実装**: アコーディオンの自動展開（`autoOpenReference`/`autoOpenA2v`）が、**carryOver（引き継いだ素材）がある側のアコーディオンも開いた状態で初期化する**よう修正した。反対スロット保持で素材が残るのだから、その素材を収めるアコーディオンも開いて見せるのが自然、という整合。詳細は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §5-15。

### 47.6 X6 Settings 2軸ラベルの表記変更

- **症状**: 第1波で「開発用（Developmental）」ラベルが長く、ボタンが2行に折り返していた。
- **実装**: オーナー確定の2段構成へ変更した。共通見出し「右クリックからの動画生成時:（Right-click menu:）」の下に、「サイズはどれに合わせる？（Match Gen video size to the...）」＋ボタン［開発用／Dev］［プロジェクト／project］［素材／materials］、「fpsはどれに合わせる？（Match Gen video FPS to the...）」＋同じ3ボタンを並べる。ラベルを短縮して1行に収めた。fps軸の「素材／materials」はX1によりグレーアウトする。詳細は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §5-13。

### 47.7 敵対的レビューと反映（致命的1・要修正2）

実装前後の敵対的レビューで、致命的1件・要修正2件を発見し、すべて反映した。

- **致命的1件（X1のテスト破壊）**: FPS軸「素材」のグレーアウト化により、「fps軸で素材を選べる」ことを前提にした既存テストが壊れることが判明した。グレーアウト（実質2択）とフォールバック仕様に合わせてテストを是正した。
- **要修正2件**:
  1. **X2の設計変更**: 当初の「後付け解放」案を棄却し、「台帳認識（reconcile）方式」へ設計変更した（§47.2の経緯）。
  2. **X5の型コアース**: carryOver（引き継いだ素材）まわりの配線に型コアース（暗黙の型変換に依存した危うい箇所）があり、アコーディオン開閉の判定が意図せず揺れうる点を是正した。

### 47.8 テスト・ゲート

- **webui typecheck**: `npm run typecheck`（`tsc -b`）エラー0。
- **webui vitest**: 全緑。第1波完了時の基準1230件から、各X項目の実装とテスト追加で**1261件pass／10件skip**へ推移した。
- **native / backend**: 非接触。native doctest 258件pass／6件skip、backend `test_logging` 22件passはいずれも第1波（§46.10）から不変。
- **実機**: 埋め込みフルビルド成功（webui変更はCMakeの`OBJECT_DEPENDS`で自動リビルド、ネイティブソースは非変更のため再コンパイルなし）・`scripts/deploy.ps1`による実機デプロイを同日中に実施済み（新しい`NzVideomni.aux2`の配置をタイムスタンプで確認。`Language/`は今回変更なしのため据え置き）。オーナー実機ゲートは未了（テスト待ち）。確認チェックリストは[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-4へ起票した。

### 47.9 台帳・ドキュメントへの反映

[`PENDING_TASKS.md`](PENDING_TASKS.md)へ、第2波（X1〜X6）の確認チェックリストを§2-4へ新設（第1波のチェックリストは§3-32へクローズ移設）、短いクリップのoverlap_frames制約の機能面合格とX3のクライアント側ゲート追加を§3-33へ記録（旧§2-1から除去）、§4-13（素材fpsの取得）を今回の調査結論で全面更新した。[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md)を第13版へ改訂（X4の右クリックv2v 1クリップ化＝§3-4 #1、X5の併用プリフィルのアコーディオン挙動＝§5-15、X2bの予約席の台帳認識reconcile＝§5-7）。[`BRIDGE_CONTRACT.md`](BRIDGE_CONTRACT.md)は契約変更なし（X2の掃除は既存`deleteProvisionalByJob`の再利用、reconcileはwebui内部ロジック）であることを§4.14へ一文追記した。

## 48. フロントエンド微調整バッチ 第3波（Y1〜Y3、2026-07-22）

第2波（§47、X1〜X6）のオーナー実機ゲートで、X1（FPS軸グレーアウト＋保存値フォールバック）・X3（短クリップゲート）・X4（右クリックv2vのクリップ1枚化）・X5（併用アコーディオン）が合格し、X2の予約詰まり修正も「実行中の正しいブロック」「未挿入の完了ジョブが残った状態での新規生成」の2経路が合格した（[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-34）。X2の「検証エラー（422）後の予約詰まり解消」は、第2波X3で短すぎるクリップがクライアント側でブロックされるようになった結果、422を発生させる操作手段そのものが無くなり検証不能となった（暫定合格として§4-23へ将来課題化）。X6（2軸ラベル）は2段構成の要件を満たすものの、共通見出しの余白が広すぎるとの指摘があった。

この実機ゲートのテスト中にオーナーが挙げた、新バグ1件（Y1）・改善1件（Y3）・余白微調整1件（Y2）を、第3波バッチ（Y1〜Y3）としてまとめて実装した。第1波・第2波と同じく、フロントエンド（webui）のみの小改修で、バックエンド・ネイティブは非接触（凍結方針の維持）。着手前にプランモードで計画し、敵対的レビューを通してから実装した。実機デプロイとオーナー実機ゲートはこれから（別担当が並行でビルド・デプロイ中のため、本記録では「デプロイ実施済み」の断定を避ける）。テーマごとの正本は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md)（第14版、Y1/Y3）と[`PENDING_TASKS.md`](PENDING_TASKS.md)（§2-4の確認チェックリスト・§3-34/§4-23のクローズ記録・§4-22の解消）に置いた。

### 48.1 Y1 参照動画を外すとGenerate復帰不能バグの根治

- **症状**: Create画面でタイムラインの動画を右クリック→「この動画を参照にIC-LoRA」（#2 `referenceVideo`）で参照動画を入れたあと、その参照動画を❌または🔁で外すと、Generate不可理由ノート（§46.6のW7）に「Attach a reference video.（参照動画を設定してください。）」が残り続け、Generateボタンが押せないまま復帰できなくなる。手動で参照動画を入れた場合は外せば理由が消えるのに、右クリック#2起点のときだけ消えない、という非対称だった。
- **真因**: 右クリック#2で起動したとき、「IC-LoRAモードである」という判定（`isICLoraIntent`）がマウント時に固定（凍結）され、参照動画を外してもモード判定が真のまま残ることが原因。IC-LoRAモードが真のあいだは「参照動画が必須」ゲートが外れず、Generateが復帰しなかった。この`isICLoraIntent`のマウント時凍結は、[`PENDING_TASKS.md`](PENDING_TASKS.md) §4-22（第2波W5で記録した「#7に参照動画を引き継ぐと128グリッドが粘着する」既知の制約）と**完全に同根**である。
- **修正（案A採用）**: IC-LoRAモードの判定を、マウント時凍結から「参照動画または制御LoRA（`loras`）が実在するときに真」という動的判定へ変更した。両方が無くなればモードが解除され、Generate不可理由も128グリッドも自動で戻る（根治）。これにより§4-22の128グリッド粘着も同時に解消した。なお、上記§4-22の「概要」にあった「❌ボタンが復帰手段」という記述は誤記であり（案A前は❌でも`isICLoraIntent`固着で復帰不能だった）、§4-22へ訂正注記を付した。
- **派生回帰の封じ込め**: モード判定を動的化した副作用として、参照動画をアップロード中（まだファイルIDが確定していない一瞬）にIC-LoRAモードが未成立と見なされてGenerateブロックが外れ、参照動画なしのプレーンなt2v（テキストから動画生成）が投げられ得る新たな回帰が敵対的レビューで判明した。参照動画アップロード中を待つゲート（理由コード`referenceUploading`）を新設してこれを封じた（§48.4の要修正2）。

### 48.2 Y2 Settings共通見出しの余白微調整

- **経緯**: 第2波X6（§47.6）で追加した共通見出し「右クリックからの動画生成時:（Right-click menu:）」が、上下等間隔で配置されていたため上のグループ（テーマ設定）に近く見え、どのグループの見出しなのかが分かりにくかった（第2波X6の実機確認で「余白が広すぎる」との指摘）。
- **実装**: 共通見出しの上の余白を広げ、下の余白を詰めて、下のグループ（サイズ／FPS方針）側へ寄せた。CSSの余白（マージン）のみの調整で、他の設定項目のレイアウトには影響しない。

### 48.3 Y3 W2挿入時のジョブカード🎞→✅化

- **経緯**: 右クリックメニュー「この生成結果を今すぐ挿入」（§46.2のW2、`insertProvisionalResult`、#13）で動画を挿入しても、操作パネルのジョブカードの挿入ボタンは🎞のままで、挿入済みであることがカード側に反映されなかった。
- **実装**: 挿入済みジョブIDの共有ストア（`provisionalReservation.ts`のmodule-levelなpublish/subscribe）を新設し、ジョブカードがそれを購読して✅（挿入済み）表示へ切り替える方式にした。W2挿入のみこの共有ストアを経由して✅化する。
- **手動🎞挿入との非対称（オーナー確定）**: ジョブカードの🎞ボタンを手動で押したときの挙動（カード単位のローカルな✅・リロードで消える揮発性）は一切変えていない。W2挿入だけを共有ストア経由で✅にする非対称の設計とした。W3「最新の生成結果をここに挿入」（`insertLatestResultHere`、#14）は、同じ動画を複数箇所へ追加挿入でき✅の意味が弱いため据え置き（🎞のまま）とした（オーナー確定でW2のみ対象）。
- **再挿入時のファイルパス記憶**: 再挿入に使うファイルパスも共有ストアに載せ、W2挿入後にジョブカードから再挿入しても再ダウンロードのエラーを踏まないようにした。

### 48.4 敵対的レビューと反映（要修正3件）

実装前後の敵対的レビューで、要修正3件を発見し、すべて反映した。

- **要修正2（最重要・アップロード中ゲートの新設）**: Y1でIC-LoRAモード判定を動的化した副作用として、参照動画のアップロード中にGenerateブロックが外れ、参照動画なしのプレーンなt2vが投げられ得る回帰が判明した。参照動画アップロード中を待つ理由コード`referenceUploading`を新設し、アップロード確定までGenerateを無効化してこれを封じた（§48.1）。3件のうち最も影響が大きい修正。
- **要修正1（テスト差し替え手段の訂正）**: Y1の検証テストで当初想定していた差し替え手段が動的判定の新仕様と食い違っていたため、新仕様に合わせてテストの組み立て方を訂正した。
- **要修正3（手動挿入中のW2上書き抑制）**: Y3の共有ストア方式で、ジョブカードの手動挿入操作中にW2由来の共有ストア更新が割り込むと、手動挿入側の状態を上書きしてしまう恐れがあった。手動挿入中はW2の上書きを抑制するようにして是正した。

### 48.5 逸脱1件（計画からの意図的なずれ）

- **`AppShell.controlLora.test.tsx`のヘルパのスコープ化**: Y1の動的判定に合わせて既存テストを追随させる際、テスト内で使うヘルパを（ファイル共有ではなく）当該テストのスコープ内へ閉じ込める形へ変更した。計画では既存ヘルパをそのまま流用する想定だったが、動的判定の新仕様に合わせてヘルパの前提を局所化する必要があったための意図的な逸脱。

### 48.6 テスト・ゲート

- **webui typecheck**: `npm run typecheck`（`tsc -b`）エラー0。
- **webui vitest**: 全緑。第2波完了時の基準1261件から、各Y項目の実装とテスト追加で**88ファイル・1274件pass／10件skip**へ推移した。
- **native / backend**: 非接触。native doctest・backend `test_logging`はいずれも第2波（§47.8）から不変。
- **実機**: 埋め込みフルビルドと`scripts/deploy.ps1`による実機デプロイは、別担当が並行で進めており同日中に実施の見込み（本記録では実施済みと断定しない）。オーナー実機ゲートは未了（テスト待ち）。確認チェックリストは[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-4へ起票した。

### 48.7 台帳・ドキュメントへの反映

[`PENDING_TASKS.md`](PENDING_TASKS.md)へ、第3波（Y1〜Y3）の確認チェックリストを§2-4へ新設（第2波のチェックリストのうちX1/X3/X4/X5/X2該当分は§3-34へクローズ移設）、X2の422予約詰まりの検証不能を§4-23へ将来課題化、§4-22（IC-LoRA 128グリッド粘着）をY1で解消済みとしてクローズ（あわせて「❌で復帰できる」という誤記を訂正）、§2-1の実GPU細部イントロから歴史記述を除去、§2-3（オーナー判断待ち・空項目）を除去した。[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md)を第14版へ改訂（Y3のW2挿入で🎞→✅化・W3据え置き＝§3-4 #13／§3-6 #14、Y1のIC-LoRAモード判定の動的化＝§5-15の既知の制約の解消）。Y2はCSSの余白調整のみのため設計仕様書には非該当。

### 48.8 オーナー実機ゲートの結果（2026-07-25・全項目合格）

第3波（Y1〜Y3）の確認チェックリスト（[`PENDING_TASKS.md`](PENDING_TASKS.md) 旧§2-4）を、オーナーがAviUtl2実機（realバックエンド）で確認し、**全項目合格**とした。

- **Y1**: 右クリック#2で入れた参照動画を❌／🔁で外すと「参照動画を設定してください。」の理由が消えGenerateが復帰する／制御LoRAをNoneに戻しても同様に復帰する／両方を外すとキーフレームグリッドが128グリッドから一般グリッド（64）へ戻る——の3点を確認。§4-22（128グリッド粘着）の同時解消も実機で裏付けられた。
- **Y2**: 共通見出し「右クリックからの動画生成時:（Right-click menu:）」が下のサイズ／FPS方針グループ側へ寄り、どのグループの見出しかが一目で分かることを確認。
- **Y3**: W2「この生成結果を今すぐ挿入」でジョブカードの挿入ボタンが🎞→✅になる／手動🎞挿入の挙動（ローカル✅・リロードで揮発）は不変／W3は🎞のまま据え置き——の非対称設計が意図どおりであることを確認。
- **検証不能1点（暫定合格）**: 「参照動画アップロード中はGenerate無効」（§48.1・§48.4の要修正2で新設した`referenceUploading`ゲート）は、実機での参照動画の読み込みが速すぎて「アップロード確定前の一瞬」を実操作で捉えられず、検証できなかった。検証不能・暫定合格としてクローズし、[`PENDING_TASKS.md`](PENDING_TASKS.md) §4-24へ将来課題として記録した（ゲート自体の単体挙動はwebui vitestで確認済み）。
- **台帳への反映**: 旧§2-4のチェックリストを除去し、合格記録を[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-35へ移設した。あわせて§4-22の末尾注記の参照先を§3-35へ更新した。
- **残作業**: 第1〜3波（W／X／Y）の変更はいずれも未コミットで、コミットとプッシュはオーナーが手動で行う（本プロジェクトの慣行）。

## 49. `.au2pkg.zip`のバックエンドリポジトリへの同梱と、その更新手順（2026-07-26）

> **【2026-07-31に置き換え済み・本節は歴史記録】** 配布方法はzip（`.au2pkg.zip`）から**`NzVideomni.aux2`単体の同梱**へ変更された（詳細は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-56の2026-07-31追記）。**§49.2の4手順（ビルド→`package.ps1`→コピー→コミット）を手で踏む必要はもう無い**——`scripts/deploy.ps1`が実機の`Plugin`ディレクトリと、バックエンドリポジトリ内の配布用コピー`Nz-Videomni\AviUtl2-Plugin\NzVideomni.aux2`の**両方を自動で更新する**（`-DistDir`既定値）。あわせて同スクリプトには、調査用ビルドを誤って配置・配布しないための`[PROBE]`バイト検索ガードが入っている（[`DEVLOG.md`](DEVLOG.md) §56.2）。本節はzip同梱時代の判断と実測値の記録として残す。

α版公開に向けたインストール導線の整備（[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-2）の作業項目4として、フロントエンドの配布物`.au2pkg.zip`をバックエンドのリポジトリ直下へ同梱した。同梱しないと、ユーザーが`setup.bat`と`run.bat`でバックエンドを動かせるようになってもプラグイン本体が手元に無く、導線が完成しないため。本節は**その更新手順の正本**である。フロントエンドを改修してデプロイしたら、この手順で同梱物も差し替えること。

### 49.1 なぜ同梱するのか（設計判断）

- AviUtl2のプラグイン導入は、**パッケージファイルをプレビュー画面へドラッグ＆ドロップする**のが公式の方法（本体添付の`aviutl2.txt`に記載）。`scripts\package.ps1`が生成する`.au2pkg.zip`がそのまま対象になる。
- 一方、フロントエンドの`dist/`は`.gitignore`の対象で、ビルド成果物はリポジトリに残らない。**バックエンド側リポジトリへ実体をコミットする**方式を採った（オーナー決定）。ユーザーは`git clone`した時点でプラグインを手にしている状態になる。
- **Git LFSは使わない**（オーナー決定）。約378KBなら通常のgit管理で何の問題も無く、Git LFSを使うとユーザー全員にgit-lfsの導入を強いることになるため。
- バックエンドの`.gitignore`は巨大バイナリ対策で`*.zip`を除外しているため、`!*.au2pkg.zip`の例外行を追加した（除外の直後に置く。gitの否定パターンは順序に依存する）。

### 49.2 更新手順（ビルド → パッケージ → コピー → コミット）

フロントエンドのリポジトリ（`Nz-Videomni-frontend-AviUtl2`）の直下で、次の順に実行する。

1. **ビルド**（Web UIの埋め込みを含む。埋め込みが既定なのでスイッチは不要）

   ```powershell
   .\scripts\build.ps1 -Config Release
   ```

2. **パッケージ**（`dist\NzVideomni-<Version>.au2pkg.zip`を作り直す）

   ```powershell
   .\scripts\package.ps1
   ```

   `-Version`の既定値は`1.0.0-rc1`（プラグインの`kPluginVersion`に対応）。バージョンを上げるときは`-Version`で明示する。`package.ps1`は**Web UIが実際にDLLへ埋め込まれているかを2つの目印で検査し、埋め込まれていなければ`throw`で止まる**（リソース名`NZVIDEOMNI_WEBUI_INDEX`だけでは判定にならない——その文字列は`FindResourceW`の引数として全ビルドに含まれるため。埋め込まれたHTML本体の先頭バイト列`<!doctype html`との**両方**が必要。§16.2／§16.5）。UIの無い壊れた配布物を出さないための硬いゲートなので、迂回しないこと。

3. **バックエンドリポジトリ直下へコピー**

   ```powershell
   Copy-Item .\dist\NzVideomni-1.0.0-rc1.au2pkg.zip ..\Nz-Videomni\ -Force
   ```

4. **バックエンド側でコミット**する。`.gitignore`の`!*.au2pkg.zip`例外により、通常のgitオブジェクトとして追跡される。

### 49.3 同梱物の実測値（差し替え時の照合用）

| 項目 | 値 |
|------|-----|
| `NzVideomni-1.0.0-rc1.au2pkg.zip` | 378,689バイト |
| 　└ `Plugin\NzVideomni\NzVideomni.aux2` | 1,041,920バイト |
| 　└ `Language\English.NzVideomni.aul2` | 2,290バイト |
| 　└ `Language\Japanese.NzVideomni.aul2` | 2,525バイト |
| 　└ `package.ini` | 160バイト |
| 　└ `package.txt` | 337バイト |

差し替えたのに`.aux2`のサイズが変わっていない場合は、**ビルドが走っていない可能性を疑うこと**。従来リポジトリにあった版は2026-07-17時点のビルド（`.aux2`が942,080バイト）で、07-18以降の全改修が未反映のまま陳腐化していた。同じ事故を繰り返さないよう、差し替え後は必ず`package.ps1`が最後に表示する内容一覧でサイズを確認する。

### 49.4 ユーザー側の導入・更新

- **導入**: バックエンドのリポジトリ直下にある`NzVideomni-1.0.0-rc1.au2pkg.zip`を、AviUtl2のプレビュー画面へドラッグ＆ドロップ → AviUtl2を再起動。
- **更新**: 同名パッケージの上書きインストールにAviUtl2が対応しているため、**新しいzipを同じようにドラッグ＆ドロップするだけ**でよい。`git pull`で新しいzipが手元に降りてくる。
- バックエンド側の更新手順（`git pull` → **`setup.bat`を再実行**）とは別物なので、混同しないこと。バックエンドの更新手順は`Nz-Videomni/README.md` §1「更新のしかた」が正本。

## 50. α版インストール導線の整備 — 同日の残り3系統（モデル再ホスト・`install_ltx.ps1`差し替え・`setup.bat`新設、2026-07-26）

前節（§49）は`.au2pkg.zip`の同梱1本だけを扱っているが、**2026-07-26の作業はそれを含む4系統**であり、残る3系統がこの時系列ログに1件も残っていなかった。本節はその穴を埋めるための**索引**である。**内容の正本は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-36・§3-37**であり、ここに詳細は重複させない。DEVLOGを時系列で追う人が「07-26に何をしたのか」を取りこぼさないための入口として読むこと。

### 50.1 この日やったこと（4系統）

| 系統 | 何をしたか | 正本 |
|---|---|---|
| ① モデルの再ホスト | 上流に散っていた本番セットを、オーナーのHuggingFaceアカウント`Rootport`の**4リポジトリ**（`Nz-LTX23-weights`／`Nz-Gemma3-12B`／`Nz-DWPose`〔新設〕／`Nz-Sulphur2`）へ再ホストし、認証なしでの到達まで検証した。**エンドユーザーはHFアカウントもトークンも不要**になった。 | [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-36、作業手順書＝[`Nz-HF-Rehost/README.md`](../../../../Nz-HF-Rehost/README.md) |
| ② `install_ltx.ps1`のダウンロード元差し替え | 5つの上流リポジトリへの呼び出しを**3リポジトリ・3呼び出し**へ集約。合計20ファイル・31,889,519,494バイト（約29.7GiB）。従来インストーラの管理外だったIC-LoRA 2点と姿勢推定の前処理器2点を**自動取得の対象に加え**、step 6の検証テーブルを10→**14項目**にした。HFトークン機構（`-HfToken`・`-Gated`・`scripts/hf_login.ps1`）は全撤去。GGUFの平坦化後処理も削除（新リポジトリが最初から`ltx-2.3-gguf/`直下の構造を持つため）。 | [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-36 |
| ③ `setup.bat`／`run.bat`の新設と`config.yaml`の追跡外化 | ユーザーが自分で用意するのは**gitだけ**になった（`uv`とポータブル版`ffmpeg`は`setup.bat`が`tools/`へ取り込み、プロセスの`PATH`先頭に足す）。`.bat`は純ASCII・CRLF・末尾`pause`で、日本語は全て`.ps1`側。`config.yaml`はgit追跡から外し`config.yaml.example`から複製する方式へ。`.venv-engine`は「存在すればスキップ」を改め、**ピン留め依存のハッシュが変わったときだけ貼り直す**方式にした。 | [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-37 |
| ④ `.au2pkg.zip`の同梱 | §49（本ログの前節）。 | 本ログ §49 |

### 50.2 注意（次にこのログを読む人へ）

- **`Nz-Videomni`の変更は、この時点で全て未コミットである。** しかも`git index`には`config.yaml`と`wheels/.gitkeep`の**削除だけがステージ済み**という中途半端な状態になっている。素の`git commit`を打つと、`PENDING_TASKS_CLOSED.md` §3-37が「避けろ」と書いている壊れた中間コミット（`config.yaml`が消えただけで`config.yaml.example`も`.gitignore`の追加も入っていないコミット）ができる。**`git add -A`で一括コミットすること。** 詳細な警告は[`Nz-Videomni/Docs/NEXT_SESSION_HANDOFF.md`](../../../Docs/NEXT_SESSION_HANDOFF.md)の冒頭ブロックに置いた。
- **`Nz-HF-Rehost`のアップロードスクリプトは再実行しないこと。** `20_upload.ps1`は`-Repo`を省くと既定の`both`で走り、**公開中のモデルカードを無警告で上書き公開する**。同フォルダのREADME冒頭に警告を追記済み。
- **オーナーの実機検証が未了。** サブマシンでの`setup.bat`→`run.bat`→生成、およびAviUtl2へのD&D導入の3項目（[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-4「実機検証の段取り」）。とくに**`config.yaml`が無い状態からの複製経路は一度も実行されたことがない**ので、そこを見ること。

> **2026-07-27追記: 上記2点のうち「未コミット」と「実機検証が未了」はどちらも解消した。** コミット＆プッシュはオーナーが手動で実施済み（バックエンド`f5b02c8`／フロントエンド`db3856c`）、実機検証は全項目合格。次節§51を参照。`Nz-HF-Rehost`のアップロードスクリプトを再実行しない件は引き続き有効。

## 51. α版インストール導線のサブマシン実機検証 — 全項目合格（2026-07-27）

§50（および[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-36・§3-37）で整備したα版のインストール導線を、オーナーがサブマシンで実機検証し、**全項目に合格した**。検証環境は**RAM 32GB／RTX 3080 mobile（VRAM 16GB・Ampere世代）／AviUtl2を導入していない新規環境**で、AviUtl2は**2026-07-25更新の公開最新版**（開発機のv2.0.54より新しい版）を新規に導入している。

合格したのは、`setup.bat`での導入 → `run.bat`での起動 → ブラウザでWebUIを開く → `smoke_test`サイズの生成 → **IC-LoRAのpose-controlとcannyをそれぞれ768p・257フレームで制御生成**（コンソールに`loras=pose-control(strength=1)`が出てエラーなく制御された動画を生成）→ `NzVideomni.aux2`のD&D導入 → 再起動後の操作パネル表示 → タイムラインからの生成と右クリックでのタイムライン配置、の9項目。

これにより、**RAM 32GBでの実測合格**（従来は実測データ皆無。ただし**この1台での実測**という位置づけ）・**Ampere（`sm_86`）での実動確認**（従来は理論上の互換のみ）・**`config.yaml`自動複製経路が初めて実際に通ったこと**・**2026-07-25更新版AviUtl2との互換**の4点が確定した。`Ctrl+C`停止時の日本語表示は今回の検証範囲外で、引き続き未検証である。

**詳細の正本は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-38**。バックエンド`README.md` §7（制限事項）と[`Nz-Videomni/Docs/NEXT_SESSION_HANDOFF.md`](../../../Docs/NEXT_SESSION_HANDOFF.md)冒頭ブロックにも同日反映済み。公開前に残るタスクは、**READMEの文面をオーナーが手書きで仕上げること（[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-4のスピードガイド）と、リポジトリをpublicにすること**の2点のみ。

## 52. NAG（Normalized Attention Guidance）フロントエンド追随（2026-07-28）

バックエンド（`Nz-Videomni`、コミット**2ae497b**）にNAG（CFGを使わずにネガティブプロンプトを効かせる手法）が実装されたのを受け、フロントエンド側の「死んだ」ネガティブ欄（送信されないReservedFields）を、実際に効くUIへ置き換えた。正本は計画書`nag-generate-clip-chain-a2v-indexed-leaf.md`（監督管理）。

### 52.1 決定事項（オーナー確定UI/UX要件）

1. **共有アコーディオン**: PromptBar直下・タブの外に「Negative Prompt」アコーディオンを1つだけ置き、Create/Chain/Batchの3画面が同じ状態を読む。Batchは独自UIを持たず、Createの設定を無言で自動適用する（注記なし）。
2. **本文のみ永続化**: `localStorage`（キー`nzvideomni.nagNegativeText`）にネガティブ本文だけを保存する。scale/tau/alphaはセッションローカルで保存しない。
3. **チェックは毎セッションOFF開始**: アプリを開き直すたびに、チェックボックス自体はOFFから始まる（本文は前回の続きが残る）。チェックOFFでも値自体は保持される（soft disable）。
4. **Otherは常時グレーアウト**: 「NAG／Other」のラジオのうち、Otherは最初から選択不可のグレー表示（将来の拡張に備えた場所取りのみ）。
5. **scale1本＋詳細設定**: nag_scaleスライダーを表に1本だけ出し、nag_tau／nag_alphaは入れ子の「詳細設定」アコーディオンへ格納する。NAG本家の推奨手順（tau/alphaは固定し、scaleだけ動かす）に沿った表出。🔄ボタンでこの3値だけを既定（scale 11.0／tau 2.5／alpha 0.25）へ戻し、ネガティブ本文には触れない。
6. **既定文5語とその出典**: 初期値は`blurry, low quality, distorted, watermark, text`。LTX公式ブログの推奨と一致しており、特に"watermark, text"はLTX-2.3が字幕・透かしを混入させやすいという既知の弱点への定番対処。

### 52.2 敵対的レビューで修正した5件の要点

実装着手前にOpusレビュアーが計画の全主張を実コード照合（20箇所以上のアンカー検証＋Node実測によるJSキー順序確認）した。致命的欠陥は無かったが、以下5件を計画に反映してから実装した。

1. `useBatchForm`の`start()` useCallback依存配列に`nag`を追加し忘れると、古いNAG状態でバッチが走る間欠バグになる——明記して対応。
2. `A2vChainPayload`は`negative_prompt: string`を必須フィールドとして既に持っていたため、payload型へ追加するのは`nag_*`4フィールドのみに限定した（`negative_prompt`の重複宣言はtscエラーになる）。
3. 空文字がそのまま永続化されると既定文5語へ戻す導線が消えるため、`readStoredNagText`に「保存値が空白のみなら既定文へフォールバックする」処理を追加した。
4. 実GPUでの効き確認は目視チェックリスト（`PENDING_TASKS.md` §2-2）ではなく実GPUチェックリスト（§2-1）の領分と判断し、振り分けを修正。§2-1にあった「現在ない」という文の訂正もあわせて行った。
5. 規範文書`Mock/AVIUTL2_DESIGN_BRIEF.md` §5・§11の「ネガティブ欄は入力不可」という記述の更新漏れを洗い出し、Wave 4のドキュメント作業に組み込んだ。

### 52.3 Wave構成

- **Wave 1（契約・純粋層、挙動変化ゼロ）**: `api/types.ts`のコメント改訂と5フィールド追加、`shell/nagSettings.ts`新設（型・定数・永続化ヘルパ・`nagRequestFields`）、`i18n/strings.ts`の`nag`名前空間新設、`bridge/mockBridge.ts`への422ミラー追加。
- **Wave 2（payload配線＋検証、UI不可視）**: `useGenerationForm.ts`・`chainUtils.ts`・`useChainForm.ts`・`buildA2vChainPayload.ts`・`batchRunner.ts`・`useBatchForm.ts`へ`nag`パラメータと検証ロジックを配線。死んでいた`BatchRunnerSettings.negativePrompt`は削除（引き算原則）。
- **Wave 3（UI）**: `shell/useNagSettings.ts`・`shell/NagAccordion.tsx`・`modes/create/FloatSliderField.tsx`を新設し、`AppShell.tsx`へ組み込み。`CommonGenerationFields.tsx`のReservedFieldsからネガティブ`<textarea>`ブロックを削除（CFG表示は存置）。
- **Wave 4（ドキュメント＋実機配布準備）**: 本節を含む各種ドキュメント更新と、ビルド・テスト・デプロイの最終確認。

### 52.4 テスト件数推移

Wave開始前の1274件から、Wave 1〜3完了時点で**1330件**（10件skip）まで増加。型検査（`tsc -b`）は全Wave通じて0エラー。新規テストは`shell/nagSettings.test.ts`・`modes/create/FloatSliderField.test.tsx`・`shell/NagAccordion.test.tsx`・`App.nag.test.tsx`の4ファイルで、既存の`chainUtils.test.ts`／`useChainForm.test.ts`／`buildA2vChainPayload.test.ts`／`batchRunner.test.ts`／`useBatchForm.test.ts`／`CommonGenerationFields.test.tsx`／`mockBridge.test.ts`にも回帰テストを追記している。`useGenerationForm.test.ts`の既存の「不在アサーション」4行は、`nagRequestFields`がnag未指定時に`{}`を返す設計により無改修のまま通り続けるため、コメント更新のみで維持した（ON時の5フィールド検証は別のitとして追加）。

### 52.5 v1スコープ外（今回は入れていない）

ジョブカード／台帳へのNAG表示、右クリックプリフィル連携、Gradio版（バックエンドの検証用UI）への【🔴ON】バッジの逆輸入。いずれも将来必要になれば追加できるよう入り口は塞いでいない。

**詳細の正本は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-38**。バックエンド`README.md` §7（制限事項）と[`Nz-Videomni/Docs/NEXT_SESSION_HANDOFF.md`](../../../Docs/NEXT_SESSION_HANDOFF.md)冒頭ブロックにも同日反映済み。公開前に残るタスクは、**READMEの文面をオーナーが手書きで仕上げること（[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-4のスピードガイド）と、リポジトリをpublicにすること**の2点のみ。

## 53. VSF（Value Sign Flip）フロントエンド追随（2026-07-29）

バックエンド側でVSF（NAGに続く2つ目の非CFGネガティブプロンプト手法。正負のコンテキストを連結し1回のattentionで済ませつつ負側のV（value）だけを符号反転×スケールする方式）が`neg_method`／`vsf_scale`の2フィールドに確定した（backend `VERIFICATION_LOG.md` §41、縮退第3弾まで完結）のを受け、§52で実装したNAGアコーディオンに方式選択を追加した。正本は[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-46。

- **実装要点**: NAGアコーディオン内に「NAG／VSF」方式ラジオを新設（常時操作可・既定NAG）。方式に応じてスライダー群を出し分け、NAGは従来どおりscale＋Advanced(tau/alpha)、VSFはvsf_scale1本（0〜10・既定1.5・step0.1）を表示する。🔄リセットの対象をscale/tau/alphaの3値からscale/tau/alpha/vsfScaleの4値へ拡張し、ボタンの文言も方式非依存の汎用表現へ改めた。VSFラジオのlabelには「実用域は1.5〜5・0を指定しても無効化はされない」という実測知見（backend §41.9〜§41.10）をtitleツールチップとして添えた。
- **リクエスト契約**: `shell/nagSettings.ts`の`nagRequestFields()`を、ON時は方式に関わらず7キー（`negative_prompt`/`nag_enabled`/`nag_scale`/`nag_tau`/`nag_alpha`/`neg_method`/`vsf_scale`）を常に返す設計に拡張した。VSF選択時でもNAG用の3フィールドを送り続けるのは、バックエンドが`neg_method`で参照フィールドを切り替える設計のため無害であり、フィールドを出し分けるより単純だからである。OFF時の`{}`は不変（byte-identical維持）で、既存のJSON.stringifyキー順序テストは無改修のまま通り続けた。method・vsfScaleはNAGのscale/tau/alphaと同じくセッションローカルとし、`localStorage`永続化は引き続きネガティブ本文のみに限定した。
- **検証**: `npm run typecheck`（`tsc -b`）0エラー。vitestは1331件から**1343 passed / 10 skipped**へ増加（削除ゼロ）、VSF経路を実フックで通す統合テストを含む。oxlintに新規警告なし。敵対的コードレビューを1回実施し、重要指摘1件（VSF経路の統合テスト不足）を追加テストで解消、軽微指摘4件（文言・コメント）も反映済み。`build.ps1`＋`deploy.ps1`によるデプロイをAviUtl2非起動状態で実施し成功した。
- **状態**: 実装・機械検証・デプロイまで完了。残るのはオーナーによるAviUtl2実機での目視ゲート（Create/Chain/Batchの3タブ、確認ポイントは[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-46参照）のみで、合格後にコミット・プッシュと台帳クローズを行う。

## 54. V2Vリボン範囲トリム（§1-6）とバッチi2v-long（§1-7）の実装（2026-07-30）

> **追記（2026-09-02）**: 本節（および§54.6・§54.9）が「バッチA2V側は`useRef`保持のままなので走行中のリマウントで孤児化する」と書いた既知の限界は、**§104で解消した**——A2Vのランナーもモジュールレベルのシングルトンへ移し、両バッチの寿命規則が揃った。**本文は追記専用の規律どおり不変**である（当時の記録として読むこと）。なお§54.6が主症状としていた「共有ロックがアプリ再読み込みまで残る」滞留のほうは、翌日の§54.9で既に解消していた。

台帳[`PENDING_TASKS.md`](PENDING_TASKS.md)の§1-6・§1-7を同一セッションで実装した。両テーマはファイル競合がほぼ無い（共通改修は`ChainScreen.tsx`と`i18n/strings.ts`の2ファイルのみで、ハンクも別）ため並行で進めた。正本は各作業指示書——[`V2V_RIBBON_TRIM_WORKORDER.md`](V2V_RIBBON_TRIM_WORKORDER.md)と[`BATCH_I2V_WORKORDER.md`](BATCH_I2V_WORKORDER.md)（後者は本セッションで全面改訂した。§54.3参照）。

### 54.1 §1-6 V2Vリボン範囲トリム（実装完了・実機調査待ち）

> **【2026-08-01時点の状態はここではなく§56を読むこと】** 本小節が書かれた2026-07-30の時点では機能は意図的に不活性だったが、2026-08-01に実機調査が完了して**有効化され、実機ゲートも全項目合格してテーマはクローズした**（§56、[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-59）。以下はその時点のスナップショットである。

タイムライン上の動画オブジェクト（リボン）が元動画ファイルの一部しか占めていないとき、その範囲だけを切り出したmp4をV2Vの冒頭クリップにする機能。採用案は(b-1)＝`/upload/video`への引数加算＋ブリッジ契約v10。

- **backend**: `services/video_io.py::cut_range_mp4`を新設（`cut_tail_mp4`のクローンだが**リサンプルしない**——ソース実測fpsのまま`select='between(n,a,b)'`でフレーム単位に切る。末尾超過はエラーではなくクランプ）。`/upload/video`に`trim_start_sec`/`trim_duration_sec`を`Query(None)`で加算し、`services/video_upload_store.py::save`が「使える窓のときだけ切る／それ以外と失敗はすべて無傷の元アップロードへ縮退」を担う。応答に`trimmed: bool`を加算。検証記録は[`Nz-Videomni/Docs/VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §42。
- **native**: 契約v10のうち**query部分だけ**を実装した。`backend.uploadFile`に`query?: Record<string,string>`を加え、`BuildQueryString`/`AppendQueryToUrl`を`bridge_core.h/.cpp`の共有純関数として切り出して`backend.request`側と共有した（`AppendQueryToUrl`が「空クエリならURLをそのまま返す」1箇所に集約されていることが、無トリム時バイト等価のnative側の構造的保証になる）。あわせて`NZVIDEOMNI_PROBE_VIDEO_ITEMS`という調査ビルド（既定OFF・プリプロセッサ段階で丸ごと消えるため出荷物に一切混入しない・`build.ps1 -ProbeVideoItems`で有効化）と、採取依頼書[`V2V_TRIM_PROBE_GUIDE.md`](V2V_TRIM_PROBE_GUIDE.md)（R1〜R8の8ケース）を作った。
- **webui**: 新規`timeline/sourceTrim.ts`（`decideSourceTrim`＝「疑わしきはトリムしない」保守判定・`playbackStartSec`＝実機調査後に式1本だけ差し替わる関数・`trimQuery`＝無トリム時は`undefined`を返してキーごと落とす）。ゲートA（`menuSelection.ts`の長さガード）とゲートB（`useChainForm`の`sourceVideoTooShortForContext`。1フレームの安全マージン付き`Math.floor(d*fps)-1 < contextFrames`）、および`sourceTrimFailed`（トリムを要求したのに応答`trimmed`が`true`でなければGenerateを止める。手動ピック経路は対象外なのでV2V自体が使えなくなる袋小路は生じない）。
- **現時点では機能が意図的に不活性**: ブリッジがリボンの再生位置を返していないため（`timeline.getSelection`への再生位置系4フィールド追加は実機調査待ち）、`decideSourceTrim`は常に`unknownPlaybackPosition`でスキップし、**アップロードはv10以前とバイト等価**になる。骨格を先に着地させたのは、実機調査の結果が変えるのを「`playbackStartSec`の式1本＋ブリッジのフィールド追加」だけに閉じ込めるためである。無トリム時バイト等価は4層で機械保証した——webuiの厳密等価アサーション／nativeのdoctest（空クエリでURL不変）／backendの「`cut_range_mp4`未呼び出し」assert／既存roundtripテストのバイト一致。

### 54.2 §1-7 バッチi2v-long（実装完了・基本フローは実機合格。プロンプトUXは§54.7で再設計）

Chain画面の下部に折りたたみ節を1つ足し、画像フォルダを指定すると**画像1枚ごとにClip Chain（`POST /generate/chain`）を1本ずつ直列生成**する機能。**バックエンドは無改修**。

- **設計の背骨はテンプレートスナップショット**: Chain画面の`buildRequest()`が返す1本分の完全な`GenerateChainRequest`をテンプレートとして受け取り、行ごとに変わる「clip 0の画像1点」だけを差し替える（`modes/batch-i2v-long/buildI2vLongPayload.ts`）。バッチA2Vのようにペイロードビルダーを自前でもう1本持つ方式を採らなかったのは、chainにはクリップ本数・クリップ別フレーム数・オーバーラップがあり、2本目のビルダーが恒久的なドリフト要因になるからである。この方式なら将来Chainに増えるフィールドは何もせずに透過する。Chain画面へのimportは構造型`ChainSnapshotSource`1本で回避し、その代入可能性はコンパイル時アサーションで固定した（Chainがフィールド名を変えたら型検査で落ちる）。
- **新規ファイル**: `modes/batch-i2v-long/`に9モジュール（`chainSnapshot.ts`／`imageRows.ts`／`buildI2vLongPayload.ts`／`batchI2vLongRunner.ts`／`runtime.ts`／`useBatchI2vLongRunner.ts`／`useBatchI2vLongForm.ts`／`BatchI2vLongSection.tsx`／`BatchI2vLongTable.tsx`）＋各テスト。A2Vへの依存は`BatchSection.css` 1本のみで、`FolderRow`など未exportの小片はローカル複製した。`ChainScreen.tsx`への改修は`<details>`節を1段追加しただけ（`.create-layout`の外に全幅の兄弟として置く）。
- **`shell/runLock.ts`（新規）**: バッチA2V × バッチi2v-longの相互排除ロック。**所有者トークン方式**にしたのは、`AppShell`が全タブを常時マウントするため、素朴な「自分のランナーがidleになったら解放」だともう一方のパネルのマウント直後（そのランナーは当然idle）に他人のロックを解放してしまうからである。バッチA2V側への変更は2箇所だけ——`modes/batch/useBatchForm.ts`のロック組み込み（+61行）と`BatchSection.tsx`のロック中警告1行（+4行）。ランナーの持ち方（`useRef`）には触れていないため、A2V側の孤児化問題は残る（§54.6の新規起票）。
- **ブロック理由は11種＋Chain画面の理由の個別展開**: `imgDirMissing`/`outDirMissing`/`noRows`/`noRunnableRows`/`sourceVideoAttached`/`clipsTooFew`/`promptEmpty`/`promptTooLong`/`unknownLoraTag`/`jobActive`/`lockedByOther`。加えてChain画面自身の`validityReasons`を「chainの設定が不正です」に丸めず1行ずつ展開する（不透明な1行では、夜間バッチが動かない原因を誰も特定できない）。非ブロックの注意は4種（seed固定・clip 0の個別プロンプト＝最も気づきにくい破綻なので専用の強警告・clip 1以降の個別プロンプト・冒頭キーフレームの置き換え）。
- **中止は`DELETE /jobs/{id}`を投げない（バッチA2Vとの意図的な差）**: バックエンドは実行中のchainジョブを途中で止められず（キャンセルは走り切ったあとに印を付けるだけ）、DELETEしてもGPU時間を丸ごと消費した末に完成品を捨てるだけになる。厳密な直列実行なので安く消せる`queued`も存在しない。よって中止＝「以降の行を投入しない」だけで、**現在生成中の1枚は完走して保存される**。Stopの反応に数十分かかるのが仕様どおりである（当初はこれを画面上の注意文で伝えていたが、§54.7のオーナー判断で説明文はUIから削除し、正本の記載に一本化した）。

### 54.3 仕様変更の経緯: 旧§1-7はオーナー意図の誤解だった

本セッション着手前の`BATCH_I2V_WORKORDER.md`初版は、「大量の画像を1枚ずつi2v生成する」という要件を**単発生成（`POST /generate`）を1枚ずつ回す機能**と解釈し、§2で「土台は単発`/generate`一択。chainは使わない」と結論していた。オーナーが求めていたのは「画像1枚ごとに**長尺動画**を作る」機能——1枚からClip Chainを1本まるごと生成し、それを画像の枚数ぶん繰り返すものである。

初版が引いていた調査事実（素の1クリップchainは`chain requires at least 2 clips`で422になる）自体は正しく、誤っていたのは**そこから「だからchainは使えない」と結論した推論**である。バリデータの条件は「ソース3種のいずれも無い**かつ**clipsが2本未満」であり、2本以上のクリップを組む長尺バッチにはそもそも当たらない。この訂正により、初版が未決事項に挙げていた「1-clip chain許可の凍結例外」も不要になった。機能名も「簡易バッチi2v」から「バッチi2v-long」へ改め、設置場所をCreate画面からChain画面へ移した（借用する設定がChain画面のものになるため）。

### 54.4 敵対的レビュー2巡で事前に潰した致命4件

実装着手前に計画へ敵対的レビューを2ラウンド当て、**致命（F級）4件**を実装前に修正した。いずれも「テストは緑のまま実機で壊れる」種類の欠陥である。

1. **§1-6・ゲートAの「span一本化」が機能退行だった**: 正本§3.3が指示していた「`extendVideo`の長さガードを`spanDurationSec`基準へ一本化する（実質1行）」をそのまま実装すると、**無トリム経路で誤ブロックが起きる**。無トリム経路とは、再生位置が不明・再生速度が1.0以外・中間点が複数など保守的にトリムを見送るすべての場合であり、**実機調査が終わるまでの現行動作そのもの**でもある。実際にアップロードされるのは元動画ファイル全体なのだから、短いspanを根拠に弾くのは誤りである。「実際に上がる素材の尺で測る」方式（`measured = decision.trim ? decision.durationSec : item.mediaDurationSec`）へ差し替え、正本§3.3も訂正した。
2. **§1-6・ffmpegがイベントループを塞いでいた**: `api/uploads.py`の`upload_video`は`async def`であり、そこから同期的にffmpegへshell outすると**切り出しのあいだ他の全APIが無応答になる**（進捗ポーリングを含む）。`await run_in_threadpool(store.save, ...)`へ逃がした。
3. **§1-7・ランナーの孤児化とrunLockの永久ロック化**: Chain画面は右クリックのintentルーティングで`remountTokens`により`key`リマウントされる。ランナーをフック内の`useRef`に持たせると、走行中バッチが孤児化して**UIからは「実行していない」ように見えるのに共有ロックだけが握られたまま**になる。`runtime.ts`のモジュールレベルシングルトンへ移し、リマウント後は購読し直すだけで走行中バッチへ再接続できる形にした。
4. **§1-7・中止が「完成品を捨てるだけ」になっていた**: 移植元のバッチA2Vランナー（`modes/batch/batchRunner.ts`）は`stop()`でベストエフォートに`DELETE /jobs/{id}`を送る作りであり、そのままコピーすると同じ挙動を引き継ぐ。ところがバックエンドの実装（`pipeline_manager.py`のキャンセルは、ジョブが走り切った**あと**に`cancelled`と印を付けるだけ）を実読した結果、chainジョブに対するDELETEは**数十分のGPU計算を消費した末に出力を破棄するだけ**だと判明した。厳密な直列実行なので安く消せる`queued`のジョブも存在しない。DELETEを送らない設計へ変え、「現在の1枚は完走保存される／Stopの反応に時間がかかる」ことを画面の注意文と正本へ明記した。

このほか重要度S/M/L級の指摘も反映している（ゲートBの1フレーム安全マージン＝サーバー側の二重丸めをWebUIでは再現できないため境界は保守側へ倒す／`trimmed`が返らないときの`sourceTrimFailed`ブロック／`Query()`に`ge=`等を付けず新422経路を作らない／`REAL_BACKEND_CHECKLIST.md`は既存の§4.5を書き換えず新節§4.12として足す／`buildI2vLongTemplate`の回帰ピンを「差分3キーのみ」ではなく`Object.keys`の集合そのものにする〔将来Chainに新フィールドが生えたとき素通しで422になるのを防ぐ〕など）。

### 54.5 テスト件数の推移

| 対象 | 本セッション前 | 本セッション後 | 差 |
|---|---|---|---|
| webui vitest | 1343 pass / 10 skip（§53時点） | **1519 pass / 10 skip**（105ファイル） | +176 |
| native doctest | 258 pass / 6 skip（v9時点） | **267 pass / 6 skip** | +9 |
| backend pytest | 753 pass / 6 skip（backend §41.10時点） | **776 pass / 6 skip** | +23 |

いずれも既存テストの削除・書き換えはゼロで、完全に加算的な拡張である。doctestは`build/ninja-release/NzVideomni_tests.exe --test-suite-exclude=integration`、pytestはアプリvenv（skip 6件はいずれも既存の`torch`未導入によるエンジン系テストの収集スキップ）。型検査は`npm run typecheck`（**`tsc -b`**。`npx tsc --noEmit -p .`は偽合格になるので使わない）で0エラー。

### 54.6 残作業

- **§1-6**: ①[`V2V_TRIM_PROBE_GUIDE.md`](V2V_TRIM_PROBE_GUIDE.md)のR1〜R8をオーナーが実機採取 → ②`再生位置`の単位確定 → ③`GetSelectionEditProc`へのフィールド加算＋`playbackStartSec`の式確定（[`BRIDGE_CONTRACT.md`](BRIDGE_CONTRACT.md) §4.13への収録も） → ④実機ゲート（[`REAL_BACKEND_CHECKLIST.md`](REAL_BACKEND_CHECKLIST.md) §4.12） → ⑤調査コードの撤去。
- **§1-7**: 実機テスト（[`REAL_BACKEND_CHECKLIST.md`](REAL_BACKEND_CHECKLIST.md) §4.13。台帳は[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-1）。→ **§54.7のとおり、基本フローは同日中に合格した。**
- **新規起票**: バッチA2Vランナーの孤児化とrunLock滞留（[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-47）。孤児化自体は従来からの既存問題だが、今回の共有ロックで「ロックがアプリ再読み込みまで残る」滞留が上乗せされた。i2v-longと同じruntimeシングルトン化で解消できる。

### 54.7 オーナー実機フィードバックの反映: プロンプトを画像ごとへ再設計（2026-07-30）

同日、オーナーがAviUtl2実機でバッチi2v-longを確認し、**基本フロー（フォルダ指定／出力フォルダの自動作成／Scan／バッチ生成／Statusの遷移）は合格**（[`REAL_BACKEND_CHECKLIST.md`](REAL_BACKEND_CHECKLIST.md) §4.13の該当3項目をチェック済みへ）。あわせて出た指摘4件を反映した。

1. **【本丸】バッチ全体のプロンプト欄を撤去し、画像（行）ごとのプロンプトへ再設計**。オーナーの指摘は「**全画像に同じ文を足す欄には意味がない**——それなら上の共通プロンプトを書き換えれば済む。プロンプトは画像ごとに指定したい」。参考実装はバッチA2V（音声ファイルごとに追加プロンプトを持つ）で、その作法へ揃えた: `I2vLongRow`に`prompt`列を足し（スキャン時は空、再スキャンで消える）、テーブルの`プロンプト`列を編集可能な`<input>`＋📝（共通プロンプト流し込み）にし、add/replaceラジオは**バッチ全体で1個のまま**（モードは全体・文面は行ごと、というA2Vとまったく同じ配置）。
   - **テンプレートは触らない設計へ**: `buildI2vLongTemplate`はもう`prompt`を合成せず、Chain画面の共通プロンプトをそのまま持つ。合成は新設の`buildRowPayload(template, { imageId, rowPrompt, promptMode })`が**行ごとに**1回だけ行う（画像の刻印と同じ1パスで、コピーオンライトも従来どおり）。旧`withRowImage`はこれに置き換えた。ランナーには`I2vLongRunnerSettings`（今日は`promptMode`のみ）を渡す形にし、**開始時にモードを凍結**する——A2Vの`BatchRunnerSettings`と同じ作法で、走行中にラジオを触っても投入済みのバッチには届かない。
   - **ガードも行単位へ**: `promptEmpty`は「合成後が空の実行対象行がある」、`promptTooLong`は「合成後が2000字超の実行対象行がある」で判定する（スキャン前は行が無いので共通プロンプトだけで判定）。理由ノートには**該当行の`#`番号**を並べる（10件超は`…(+N)`で省略）。判定に使う`composeBatchPrompt`はランナーが呼ぶものと同一関数なので、「ブロックされた理由」と「実際に送られる本文」が食い違わない。
2. **タイトルを「バッチi2v-long（プロトタイプ）」／"Batch i2v-long (prototype)"へ**（プロトタイプであることを画面上で明示）。
3. **パネル冒頭の長い説明文を削除**（`notice`・`chainSettingsNote`のi18nキーごと撤去）。「ウィンドウを閉じるな」「中止しても現在の1枚は完走保存される」といった注意はUIから消えるが、正本（[`BATCH_I2V_WORKORDER.md`](BATCH_I2V_WORKORDER.md) §4）に記載済みである。**残したのは`chainSummary`の1行だけ**——「クリップN本・Mフレーム・画像1枚あたり約X秒」は文章ではなく**設定の実効値**であり、ドキュメントでは代替できないため。
4. **共通プロンプト欄の下の注記（`promptBar.chainNote`「つなげるモードでも画風LoRAタグは適用されます」）を削除**。これに伴い`PromptBar`は現在のモードに依存しなくなったので`mode`プロップも撤去し、`.prompt-bar-chain-note`のCSS（高さ確保用の空スロット）も削除して、その20pxぶんの間隔は`.prompt-bar`の下パディング側へ寄せた（NAGアコーディオンとの間隔が詰まらないように）。

検証: `npm run typecheck`（`tsc -b`）0エラー、vitest **105ファイル / 1535 pass / 10 skip**（§54.5の1519 passから、行プロンプトのテストを足し撤去分を差し引いて+16）、`npm run lint`は新規の警告なし（exit 0）。

### 54.8 初回行だけSTATUSがWaitingのまま進む表示バグの修正（2026-07-30）

再設計後の実機確認で、オーナーから「生成・行ごとプロンプトの反映はほぼ合格。ただし**1つ目の行だけ**、生成が始まってもSTATUSがWaitingのままで、完了した瞬間に✅Doneへ飛ぶ。2行目以降は正常に⏳Generatingになる」との報告。調査の結果、[`runtime.ts`](../webui/src/modes/batch-i2v-long/runtime.ts)の`run()`の**書き込み順序**が原因だった。

- `BatchI2vLongRunner.start()`はasyncだが、最初のawait（1行目の画像アップロード）までは**同期実行**され、その中で既に1行目を⏳Generatingにしたスナップショットを発行し終えている。ところが`run()`は制御が戻った直後に、開始時に受け取った全行Waitingの配列でスナップショットを**上書き**していた。1行目のGeneratingだけが握り潰され、次の更新は完了時（Done）——報告どおりの症状になる。2行目以降のGeneratingは非同期発火のため上書きされない。
- バッチA2Vに同症状が無いのは、`useBatchForm`が`setRows`を`batchRunner.run()`より**前**に呼ぶ（＝後勝ちがrunner側のGenerating）ため。この「前に書くか後に書くか」の差がそのまま原因だった。
- 修正は`run()`内の2箇所の入れ替え: 開始時入力（imgDir/outDir/rows）の公開を`instance.start()`の**前**へ移動し、`start()`後は`state`のみ更新。再発防止のコメントを添えた。修正前に失敗する再現テスト3件（runtime 2件・フォーム層1件。アップロードを保留して決定的に観測）を先に書き、修正後に全PASSを確認。

検証: vitest **105ファイル / 1538 pass / 10 skip**（+3）、`tsc -b`0エラー、lint緑。修正版はビルド・配置済み。

### 54.9 バッチ相互排他の実バグ2件の修正と、Stopボタン不在報告の調査結果（2026-07-31）

オーナーの実機報告「A2Vとの相互排他がフロントエンドで効いていない（片方の実行中にもう片方のStartが押せてしまう）」「i2v-longにStopボタンが見つからない」を調査した。

**相互排他は実バグ2件**:

1. **バッチA2Vに`hasActiveJob`ゲートが無かった**。共有ランロック（`shell/runLock.ts`）はブラウザ内の揮発状態のため、①WebView再読み込み後（ジョブはサーバー側で継続中）②Create画面リマウントで走行が孤児化した後③バッチ以外のGenerate実行中——のいずれでもA2VのStartは有効のままで、押しても`acquireRunLock`がnullを返して**黙って何も起きない**「死んだボタン」になっていた。修正: A2Vにも`hasActiveJob`ゲート＋案内バナーを追加（i2v-long側には当初からある`jobActive`と同じ構造。`CreateScreen`が`JobsContext`の`hasActiveJob`を渡す）。
2. **A2Vのロック返却がReactのeffect頼み**だったため、走行中にCreate画面がアンマウント（リマウント）されるとロックが永久滞留し、以後i2v-longが開始不能になっていた。修正: `useBatchRunner.run()`に`onSettled`コールバックを追加し、実行Promiseの決着時（アンマウント後でも発火する）にロックを返却する形へ移設。

**Stopボタンはi2v-long側に欠陥なし**。描画位置・条件はA2Vと同一（`.generate-row`内・`runnerState !== "idle"`）で、走行中に表示されることと`key`リマウント後も表示が続くことをテストでピン。配布物のビルド取り残しも否定（07-30 23:44ビルドに実装済みを確認）。報告時点では走行が既に終わっていたか、パネルが再読み込みされていた可能性が高い。なお**A2V側**は走行中リマウントで孤児化しStopが消える既知の限界が残る（§3-47のトリガー待ち。恒久対応はi2v-longと同じランナーのシングルトン化）。

検証: 新規`shell/runLock.crossPanel.test.tsx`（実物コンポーネント2枚を並べた両方向の排他・Stop実効性・リマウント耐性・再読み込み相当の5件）＋`useBatchForm.test.ts`2件（修正前に失敗することを実測）。vitest **106ファイル / 1545 pass / 10 skip**、`tsc -b`0エラー、lint緑。実機＋配布用コピー（`deploy.ps1`の`-DistDir`、2026-07-31新設）の両方へ配置済み。

## 55. Acceleration（生成の高速化）フロントエンド追随 — SageAttentionのジョブ単位切替とモック2項目（2026-07-31〜2026-08-01）

バックエンド側で「生成そのものを速くする」切替が実装された（backend [`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §43）のに追随し、Settings（設定パネル）へ**Acceleration**の区画を新設した。台帳のクローズ記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-58。

区画に並ぶのは3項目だが、**実際に効くのは真ん中のattention（注意機構）の実装選択だけ**である。

| 行 | 選択肢 | 状態 |
|---|---|---|
| Fused GGUF dequant + GEMM | On / Off | **モック**（常時disabled・「未実装です（将来対応予定）」のツールチップ） |
| Attention | `sdpa` / `sage attention` | **実装済み**・既定`sdpa` |
| VAE (video decode) | Default / PruneVAED | **モック**（同上） |

- **実装要点**: 新規`shell/accelerationSettings.ts`（型・既定値・`localStorage`永続化・`accelerationRequestFields()`）と`shell/useAccelerationSettings.ts`の2ファイルを土台にした。**Contextは作っていない**——`shell/nagSettings.ts`が冒頭で明文化している「AppShellが`useState`で持ちpropsで配る、Contextは使わない」という本リポジトリの流儀に従ったもので、結果としてテスト側にProviderを足す必要もなくなった。**能力取得の新しいフックも作っていない**——AppShellが既に`useServerStatus`（10秒ごとのポーリング）で`/status`を保持しているため、その`serverStatus`をSettingsPanelへpropsで渡すだけで足りる。`StatusResponse`には`acceleration?`をoptionalで足した（`StatusResponse`は`/status`の部分型なので、optionalの作法は既存の`AppConfig.model?`に倣った）。
- **リクエスト契約**: `accelerationRequestFields()`が単一の集約点で、**既定（`sdpa`）なら`{}`を返す**。したがって切り替えていないユーザーのリクエストJSONは従来とバイト等価で、キー順序を固定している既存テストは無改修のまま通り続けた。`sage`を選んだときだけ`attention_backend`が1キー増える。**モック2件はこのWebUIから一切送らない**（受理されるだけで何もしないフィールドを送る意味がないため）。型の上では設定オブジェクトに載せてあり、「決して送らない」というルール自体をテストで固定できる形にしてある。この関数はCreate（`useGenerationForm.ts`の`toGenerateRequest`）・Chain（`chainUtils.ts`・`useChainForm.ts`）・Batch（`buildA2vChainPayload.ts`・`batchRunner.ts`・`useBatchForm.ts`・`BatchSection.tsx`）の全経路から呼ばれる。バッチi2v-longは`{...withoutSource, clips}`の逐語パススルーのため**ソース改修ゼロ**で、パススルーを固定するテストだけを足した。
- **sageが使えないときの扱い**: `/status`の`acceleration.sage_available`が明示的に`false`のときだけ`sage`ボタンをdisabledにし、ツールチップで理由を出す。**値が取れない・不明なときは封じない**——実際に使えなくてもサーバーが`sdpa`へ降格して完走するため、封じると理由もなく機能を奪うことになる。
- **文言**: `sage`を選んでいるあいだ、attentionの行の下に「同じシードでも生成結果の細部が変わります（数値精度が異なるため）。速度は約1.2〜1.6倍」という注意文を出す（en/ja両方）。**選択肢のラベル`sdpa`／`sage attention`は技術的な固有名として翻訳しない**方針にし、Gradio UI側とも揃えた（言語切替の分岐を増やさないためでもある）。
- **永続化**: `localStorage`の`nzvideomni.acceleration`キーに、選ばれたattentionの実装名だけを素の文字列で保存する（`ThemeContext.tsx`の`THEME_STORAGE_KEY`と同じ形と、同じ「未知の値なら既定へ戻す」防御的な読み方）。キー名をフィールド名ではなく区画名にしてあるのは、将来2つ目の永続項目が増えたときに既存の利用者のキー名を変えずに済ませるため。モック2件は操作できないので保存対象にしていない。
- **検証**: `npm run typecheck`（`tsc -b`）0エラー。vitestは§54.9の1545件から**1585 passed / 10 skipped**へ増加（削除ゼロ）。新規は`shell/accelerationSettings.test.ts`・`shell/useAccelerationSettings.test.ts`と、SettingsPanel側の4観点（3行の描画・モック2件のdisabled・sage選択時の注意文・`sage_available:false`時のdisabled）、および各payload系テスト（既定時はJSON完全一致・`sage`時はキーが1個だけ増える）。`SettingsPanel.test.tsx`の`beforeEach`には新しい`localStorage`キーの`removeItem`を追加した。
- **状態**: 実装・機械検証・デプロイ（実機＋配布用コピーの2箇所）まで完了し、**2026-08-01にオーナー実機で目視ゲートも合格**（i2v＋NAG、1344×1728・153フレームで460.63秒→366.85秒＝1.26倍を実測。UI3項目の見た目・文言も合格）。派生した将来課題は[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-49・§3-50・§4-22へ起票済み。コミットはオーナー判断待ち。


## 56. V2Vリボン範囲トリム（§1-6）— 実機調査の確定と仕上げ実装（2026-08-01）

§54で骨格だけ着地させ「意図的に不活性」にしてあったV2Vリボン範囲トリムを、実機調査の完了をもって**有効化した**。正本は[`V2V_RIBBON_TRIM_WORKORDER.md`](V2V_RIBBON_TRIM_WORKORDER.md)（確定事実は同§4）、実機ゲートは[`REAL_BACKEND_CHECKLIST.md`](REAL_BACKEND_CHECKLIST.md) §4.12。台帳は起票時は`PENDING_TASKS.md` §1-6だったが、**同日中に全ゲート合格でクローズし[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-59へ移設した**（§56.6）。以後この番号で参照すること。

### 56.1 実機調査で確定したこと

これが本テーマの前提を丸ごと決めた部分である。

- **`再生位置`の生値は4フィールドのCSV** `開始,終了,再生範囲,0`（例: `"2.000,10.700,再生範囲,0"`）。**先頭2つは素材の時間軸上の秒**（小数3桁固定）。無加工のリボンでも`0.000,<素材全長>`が省略されずに入る。頭を削ると第1値が、末尾を削ると第2値が動く。
- **単位はプロジェクトのフレームレートに依存しない**。同じ「頭を2秒削る」操作が30fpsでも24fpsでも`2.000`になることを実測した。したがって換算式は不要で、`sourceTrim.ts`の`playbackStartSec()`は素通しの読み出しになった（起票時に想定していた3通りの分岐は消えた）。
- **`get_object_track_value`は第1値しか返さない**。終了位置をトラックAPIから取る手段が無いため、**生文字列のパース一択**である。
- **`再生速度`は百分率の小数2桁文字列**（`"100.00"`／`"200.00"`）。nativeが100で割って1.0スケールに正規化する。
- **`ループ再生`という項目が実在した**（生値`"0"`/`"1"`）。起票時の設計に無かった要素で、判定理由`loopEnabled`を新設して無トリムに落とす。
- **中間点は`再生位置`の値に一切現れない**。区間数は`get_object_section_num`だけが頼りである。
- **R4型のはみ出しが実在した（設計上の決め手）**: 開始位置をずらしてもリボンの長さが変わらない状態があり、このときリボンのほうが実際に再生される窓より長い（末尾は終端フレームの静止画）。逆に24fpsでは素材フレームの量子化により`終了−開始`がリボンの秒数を最大1フレーム弱**上回る**ケースも実測された。そこで切り出し尺を**`min(リボンの秒数, 終了−開始)`**とした。**`min`の両側がそれぞれ実在のケースに対応している**ので、どちらか一方だけでは破れる。

### 56.2 native側の実装

- **純関数`ParsePlaybackRange`を`bridge_core`に新設**した（doctest可能な層）。カンマ分割の**先頭2フィールドだけ**を`std::from_chars`（ロケール非依存）で読み、第3フィールド以降は完全に無視する。**フィールド数にもモード名にも依存しない**のは、第3フィールドが日本語のモード名で将来変わりうるためである。フィールド不足・数値として読み切れない（`"2.0x"`のような末尾ゴミを黙って切り捨てない）・start<0・end<startはすべて`false`を返し、出力は書き換えない。`再生速度`用の`ParsePlaybackSpeedPercent`も同様の作法で並べた。
- **`SelectionItem`に6項目を追加**した（`playback_start_sec` / `playback_end_sec` / `has_playback_range` / `playback_speed` / `loop_play` / `section_count`）。`media_width`/`media_height`と同じ**「非nullable＋明示のhasフラグ」**の作法である。`0.0`は「先頭から再生」という正当な値でもあるため、値そのものでは「読めなかった」を表現できない。既定値はすべて**保守側**（範囲なし・速度1.0・ループなし・1区間）に寄せてあり、読めなかったオブジェクトが「トリムしてよい」側に倒れることはない。
- **読み出しは`get_object_item_value`優先・エイリアスへフォールバック**という2段構えにした（`ReadVideoFileItem`）。SDKのゲッターは生値を必ず持つが、ホストのビルドによっては存在しない可能性があるためで、エイリアスは`GetSelectionEditProc`が既にコピー済みなので追加コストがない。SDKの文字列は「次の文字列返却呼び出しまでしか有効でない」規約なので、**受け取った直後に`std::string`へコピー**している。
- **調査コードを全撤去**した: `NZVIDEOMNI_PROBE_VIDEO_ITEMS`のCMakeオプション（宣言と`target_compile_definitions`）・`build.ps1`の`-ProbeVideoItems`スイッチ・`plugin.cpp`のプローブA（本体と呼び出し2か所と`<atomic>`のinclude）・`bridge.cpp`のプローブB（本体と呼び出し2か所）。撤去後にソース全体をgrepして`PROBE_VIDEO_ITEMS`と`[PROBE]`が残っていないことを確認した。**`scripts/deploy.ps1`の`[PROBE]`バイト検索ガードだけは残した**——調査ビルドを誤って配置しないための保険であり、今後また同種の調査をするときにも効くので無害である。

### 56.3 WebUI側の実装

- **契約v10のもう半分を`bridge/types.ts`へ収録**した。6項目とも**任意フィールド（`?`）**として宣言してある。これは古いプラグインが返さない場合に**保守側（＝トリムせず全体をアップロード）へ倒れる**ためだけの措置で、現行のnativeは常に6項目とも返す。必須にすると既存のテストfixture 12ファイル・42箇所を機械的に書き換えることになり、得られる保証（native側は非nullableで既に固い）に見合わなかった。
- **`sourceTrim.ts`のローカル型`TrimSelectionItem`を契約型の再エクスポートへ縮退**させた（暫定名`playbackPositionRaw`/`hasPlaybackPosition`は廃止。まだ製品経路で使われていなかったので互換は取っていない）。§54でこのファイルのコメントが予告していた「採取が済んだらこのinterfaceは再エクスポートに崩れる」がそのとおりになった形である。
- **判定`decideSourceTrim`の確定**: 既存条件1〜3（素材尺が既知／リボンの秒数が解決できる／リボンが素材全体より**1プロジェクトフレーム以上**短い）はそのまま維持し、`loopEnabled`を追加、`hasPlaybackRange`で判定するよう変更した。速度の比較は**厳密一致をやめて`1e-6`の許容差**にした——nativeが`"100.00"`を100で割った結果は計算値であり、ビット等価を要求すると丸め屑が「速度が変わっている」に化けて全トリムを封じる恐れがあるためである。切り出し尺は`min(spanSec, playbackEndSec − playbackStartSec, mediaDurationSec − startSec)`。
- **無トリム時のバイト等価は経路が変わったが維持されている**。無加工のリボンは`再生位置`に`0.000,<素材全長>`を**明示的に持つ**ので、「再生位置が読めないから見送る」経路ではなく**条件3（`spanCoversWholeMedia`）**で無トリムに落ちる。この差し替えは危険な箇所なので、`ChainScreen.prefill.test.tsx`のT24（`toHaveBeenCalledWith`でパラメータオブジェクト全体を厳密比較し、`query`キーがどんな形でも現れたら落ちる）に加えて`sourceTrim.test.ts`のR1ケースでも二重に固定した。

### 56.4 検証

- **native**: 通常ビルド成功（プローブ撤去後）＋doctest **267 → 274 pass / 6 skip**（`ParsePlaybackRange`の正常系・実測値3種・フィールド数非依存・異常系9種・`ParsePlaybackSpeedPercent`・v10フィールドの直列化と既定値で7ケース追加）。
- **webui**: `npm run typecheck`（`tsc -b`）0エラー、`npm run test` **1585 → 1597 pass / 10 skip**（108ファイル、削除ゼロ）、`npm run lint`は既存の警告のみでエラーなし。テストは**実測したリボン状態R1〜R4をそのままケース化**してある（R1無加工→無トリム／R2頭2秒削り→`{2.0, 8.7}`／R3末尾削り→`{0, 7.042}`／R4はみ出し→尺が7.367へクランプ／24fps量子化→尺がリボン側8.667になる）。
- **残（この時点）**: 実機ゲート③（短尺でGenerate無効化＋案内）と④（トリム後V2V→Join→🎞挿入で**中身の連続性**）。④は「挿入位置」ではなく「継ぎ目の直前に映っている絵がリボンで選んだ範囲の末尾かどうか」で判定する。→ **いずれも同日中に合格した（§56.6）。**

### 56.5 右クリック#2「IC-LoRA参照動画」への拡張（2026-08-01・オーナー承認済み）

§56.4の時点で残していた同型の不整合（正本§7-3）を、オーナー承認のうえ同日中に潰した。

- **実機症状**: 321フレームの素材をタイムラインへ置き、リボンを31〜120フレームに詰めて右クリック→「IC-LoRA参照動画」を実行すると、**参照として効く絵が元動画の1〜90フレーム**になる。原因はV2Vとまったく同じで、アップロードがフル尺だったこと——パイプラインはファイル先頭から`frame_cap`分を読むので、切っていないファイルを渡せば必ず先頭が使われる。
- **無改修で済んだ範囲**: `sourceTrim.ts`（純関数）・`useSourceUpload.ts`（`uploadPath`の`query`引数と`trimFailed`フラグ）・バックエンド（`reference_video_id`も同じ`/upload/video`＋同じトリムクエリで完結する）・native。§54で「#2も同じ関数で将来対応できる設計にしておく」と書いた予告がそのまま効いた形で、**足りなかったのは呼び出し側1行だけ**だった。
- **配線（`modes/create/CreateScreen.tsx`）**: autoLoad effectの`reference-video`分岐が`decideSourceTrim(item, selection)`→`trimQuery(decision)`をアップロードへ渡すようにした。`trimQuery`は無トリム判定で`undefined`を返すので、**無トリム時の要求パラメータは従来と完全に同一**（`query`キーが存在しない）。新設した`CreateScreen.prefill.test.tsx`のR2で、`toHaveBeenCalledWith`によるパラメータオブジェクト全体の厳密比較として固定してある（ChainのT24と同型）。R1側はオーナーの再現条件をそのままfixtureにして、`trim_start_sec: "1.000"` / `trim_duration_sec: "3.000"`が渡ることを見ている。
- **手動📁ピックは従来どおり無トリム**である。`pick()`にはタイムラインのオブジェクトが無く、測るものが存在しないため。仕様として実機チェックリストにも明記した。
- **ゲートAの統一（`timeline/menuSelection.ts`）**: #2専用だった`spanDurationSec`基準を廃し、#1と同じ`measured = decision.trim ? decision.durationSec : item.mediaDurationSec`へ一本化した。§3.3で「span一本化は誤ブロックになる」と結論した論拠が、そのまま#2にも当てはまるようになったためである——トリムが成立しない#2（再生位置が読めない・速度が1.0でない・中間点が複数など）では**本当にフル尺が上がる**のだから、そこで短いspanを根拠に弾いてはならない。W4当時にspanを測っていたのは「#2はまだフル尺を上げる」という前提での次善策で、その前提自体が消えた。
- **既存テストの挙動変化（意図的）**: `menuSelection.test.ts`の#2ケースは、更新理由をテストコメントに明記したうえで書き換えた。特に「span基準なら弾いていたが、統一後は通る」ケース（長い素材＋短いリボン＋再生位置なし＝フル尺アップロード）は**期待値を反転**させてある。逆に「トリムが成立して窓が短い」ケースは新しくfixtureに再生位置を持たせて弾かれることを確認している。
- **`referenceTrimFailed`ゲート**: 切り出しを頼んだのに応答が`trimmed: true`でなかった場合（古いプラグインがクエリを捨てる／ffmpegが無い）にGenerateを止める理由コードを`useGenerationForm`へ追加した。Chain側`sourceTrimFailed`と同型・同趣旨の文面（en/ja）である。アップロード自体は成功して`ready`＋有効なidを持つので、放っておくと**ユーザーが選んでいない範囲で生成が走ってしまう**——それを黙って通さないための一段である。
- **検証**: `npm run typecheck`（`tsc -b`）0エラー、`npm run test` **1604 pass / 10 skip（109ファイル）**（IC-LoRA拡張分で7本増）、`npm run lint`は既存の警告のみでエラーなし。※skipの10件は`backend.integration.test.ts`が`describe.skipIf`で自動スキップした分（実バックエンドを起動していないとき。この1ファイルは疎通不可を報告する1件だけがpassする）。**この数はファイルを除外せずに`npm run test`をそのまま流した実測値**であり、同ファイルを除外して数えた「108ファイル・1603 pass」という書き方とは1件ずれるので、以後は実測値のほうで書く。
- **残（この時点）**: 実機ゲート⑤（[`REAL_BACKEND_CHECKLIST.md`](REAL_BACKEND_CHECKLIST.md) §4.12）。オーナーの再現手順そのまま——321フレーム素材の31〜120フレームのリボン→IC-LoRA→参照が31〜120相当になること。あわせて切り出し後のファイル長とDURATIONシードの整合も見る。→ **同日中に合格した（§56.6）。**

### 56.6 実機ゲート全項目合格とテーマクローズ（2026-08-01）

同日のうちにオーナー実機で残りのゲートがすべて通り、**本テーマは完結した**。

- **③短尺ゲート合格**: 短すぎるリボンでV2Vを実行しようとすると「This video is too short. A video of about 4 seconds or longer is required.」が表示され、冒頭動画が挿入されないことを確認した。クライアント側で止めているので**サーバーの422は発生していない**（[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-32の「製品UIから422を起こす操作手段が無い」状態は維持されている）。
- **④トリム後Joinの中身連続性合格（本改修の決め手）**: 末尾を削ったリボンのV2Vで、**切られた位置から続きが生成される**ことを確認した。改修前は元動画ファイルそのものの末尾が文脈になっていたので、ここが変わったことが機能の本体である。判定を「挿入位置」ではなく「継ぎ目の直前に映っている絵」で行った点も§6の訂正どおり。
- **⑤IC-LoRA参照動画の範囲反映合格**: リボンをクリップした動画オブジェクトからのIC-LoRAで、正しくクリップ区間を参照した動画が生成されることを確認した（拡張前の症状は「参照が元動画の1〜90フレームになる」）。
- **最終的な機械検証の実測値**: backend pytest **817 pass / 6 skip**（Acceleration〔sage〕の追加分を含む最新値）、native doctest **274 pass / 6 skip**、webui vitest **1604 pass / 10 skip（109ファイル）**（`npm run test`をそのまま流した実測値。skipの10件は`backend.integration.test.ts`が実バックエンド未起動時に自動スキップする分）。
- **台帳のクローズ**: [`PENDING_TASKS.md`](PENDING_TASKS.md) §1-6を[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) **§3-59**へ移設した（`PENDING_TASKS.md`側の番号は欠番のまま残す運用）。実機ゲートの記録は[`REAL_BACKEND_CHECKLIST.md`](REAL_BACKEND_CHECKLIST.md) §4.12（全5項目チェック済み）、バックエンド側は[`Nz-Videomni/Docs/VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §42。
  - **番号衝突の修復（次に読む人への注意）**: 移設時、クローズ台帳の**§3-58は先にAcceleration（§55）が使っていた**のに本テーマも同じ番号で書き始めてしまい、一時的に衝突していた。V2Vリボン範囲トリムを**§3-59へ改番して解消済み**である。したがって**§3-58＝Acceleration／§3-59＝V2Vリボン範囲トリム**が正しい。他文書に「V2Vリボン範囲トリム＝§3-58」と書いてあるものがあれば§3-59の誤りなので直すこと（2026-08-01時点では両リポジトリのDocs/を全文検索して残骸ゼロを確認済み）。
- **配布物の扱い**: `scripts/deploy.ps1`は実機の`Plugin`ディレクトリと、バックエンドリポジトリ内の配布用コピー`Nz-Videomni\AviUtl2-Plugin\NzVideomni.aux2`の**両方**を自動更新する。§56.2で残した`[PROBE]`バイト検索ガードがここに効いており、調査コードが混ざったビルドは配布用コピーへ流れない。

### 56.7 派生: 保存領域（Outputs／Uploads）の方針確定（2026-08-01）

本テーマのトリムがアップロード保管庫（`uploads/`）にファイルを増やす性質のものだったことから、オーナーとのディスカッションで**保存領域の設計原則を確定した**。

- **原則**: 「**Outputsは宝物、Uploadsは一時ファイル置き場**」。`outputs/`はユーザーが失いたくない成果物（完成動画とmetadata.json）で、バックアップ対象はここだけ。`uploads/`は生成のために一時的に預かった入力素材で、キャッシュとして扱う。**新機能で「後から必要になる唯一のコピー」を`uploads/`へ保存してはならない。**
- **正本**: 新設した[`Nz-Videomni/Docs/STORAGE_POLICY.md`](../../../Docs/STORAGE_POLICY.md)（実構造・IDの紐づき・ユーザー向けのディスク整理ルールまで記載）。
- **一本化案はスコープ外**: 「同じジョブのものは同じディレクトリへ」というOutputs/Uploads一本化案は、アップロードがジョブ誕生前に起きること・1アップロードを複数ジョブが使い回すことから見送り、[`PENDING_TASKS.md`](PENDING_TASKS.md) **§4-23**（スコープ外）へ起票した。軽い改善候補として「起動時に古いuploadsを年齢ベースで自動掃除」を同書に残してある。

## 57. Style LoRA音声強度制御（audio_strength）— バックエンドAPI＋MCP＋Gradio＋AviUtl2フロントエンド一気通貫実装（2026-08-02）

LTX 2.3でStyle LoRA（画風・キャラクターLoRA）適用時に音声出力が壊れる（雑音・音割れ）というコミュニティ報告を受け、LoRAごとに音声側の適用強度を映像側と独立に制御できる`audio_strength`をAPIへ追加した。正本は[`Nz-Videomni/Docs/LORA_AUDIO_STRENGTH_WORKORDER.md`](../../../Docs/LORA_AUDIO_STRENGTH_WORKORDER.md)（詳細な分類ルール・実装ファイル一覧はそちら）、API利用者向けの仕様は[`API_REFERENCE.md`](API_REFERENCE.md) §5.3。

- **設計の骨子（オーナー確定・2軸モデル）**: 既存`strength`=映像軸、新設`audio_strength`=音声軸。**省略時(null)は音声側も`strength`に追従し、既存リクエストと完全に同一のバイト・生成結果になる**（後方互換の最優先事項）。`audio_strength`は`strength`と異なり**0を許容**し、0=音声側の重みを一切適用しない（=スキップ）という設計で、VRAM・速度の両面で有利。どちらの軸に属するかはLoRAの重みキーのモジュール名（`video_to_audio_attn`および`audio_`始まりが音声軸、`audio_to_video_attn`を含むそれ以外は映像軸——書き込み先ストリームで判定する特例）で決める。クロス注意の方向別制御（4軸案）はUXを優先してスコープ外化し、[`PENDING_TASKS.md`](PENDING_TASKS.md) §4-24へ起票した。
- **タグ構文**: `<lora:名前:映像の強さ:音声の強さ>`の第3引数として追加。省略時は映像側に追従、クランプ範囲は映像側0.05〜2.0に対し音声側は0〜2.0（0を許容する点が差）。`<lora:名前::0>`（映像側だけ省略）は非対応。画風LoRAカードの既定挿入トークンは`<lora:名前:1.0>`から`<lora:名前:1.0:1.0>`へ変わった。
- **敵対的レビュー**: 実装着手前にPlanエージェント起案の計画へ敵対的レビューを実施し、**BLOCKER 2件**（うち1件は`parseLoraPrompt`のコールバックが3引数化でoffsetズレを起こし、LoRAチップの±/×ボタンが誤位置を編集する不具合）を含む計11件の指摘をすべて計画へ反映してから着手した。
- **フロントエンド側の実装要点**: `webui/src/lora/loraTags.ts`に`LORA_AUDIO_STRENGTH_MIN/MAX`と`clampLoraAudioStrength`・`formatLoraTag`（±ボタンや音声強度セットが第3引数を握り潰さないよう経由を一本化）・`setLoraTagAudioStrength`を新設。`LoraChips.tsx`は音声バッジとミュートトグル（🔊/🔇）を常時表示し、2引数タグ（音声強度省略）では映像強度の値を実効値としてそのまま表示する（`audioDisplay = audioStrength ?? tag.strength`）。`types.ts`の`LoraSpec.audio_strength?`はsnake_caseのまま内部でも統一し、既存のリクエストビルダー（create/chain/batch/batch-i2v-long）は変換層なしに素通しする。
- **検証**: バックエンドpytest・フロントエンドの型検査（`tsc -b`）とvitestが全緑（件数は本リポジトリ[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md)、バックエンド側は[`Nz-Videomni/Docs/VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md)の対応節）。
- **実機ゲート結果（2026-08-02・オーナー実施）**: real環境・実GPUでA/B比較を実施し、音声強度0.0で音割れが明らかに減少することを確認、分類ログ（`muted=672 linears`）も想定どおり出力された。合格。詳細はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §45.2。
- **UIフィードバック対応（2026-08-02再デプロイ）**: LoRAチップの並び順を改善し、同日オーナーが実機で目視確認、合格（詳細は同§45.3）。ミュートトグルは🔇=音声強度0.0、🔊解除=1.0固定復帰（元の追従状態には戻らない）。中間値の指定はタグの手編集のみ。
- **状態**: 実装・機械検証・オーナー実機A/Bゲートまですべて完了。**テーマ完結**。コミットはオーナー判断待ち。

## 58. モデル骨格の常駐（`keep_resident`）フロントエンド追随 — Settings > Acceleration の4つ目のトグル（2026-08-02〜03）

バックエンドが「モデルのCPU側骨格をジョブ間で使い回す」切替を per-job フィールド（ジョブごとのリクエスト項目）として製品化した（backend [`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §48）のに追随し、Settings（設定パネル）の **Acceleration** 区画へ4つ目の行を足した。台帳は[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-9（オーナー目視ゲート待ちのため、まだ[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md)へは移していない）。

- **UI の形**: 直上の Block-swap prefetch 行と**同じ2ボタントグル**（有効／無効）。違いは**可否ゲートが無い**ことで、`sage` の `acceleration.sage_available` や prefetch の `block_swap_prefetch_available` に相当するフラグが `/status` に存在しない——サーバーが「できる／できない」を答えられる性質のものではなく、**このマシンに十分なメインメモリがあるかという利用者側の選択**だからである。したがってどちらのボタンも決して disabled にせず、ツールチップも付けていない。
- **既定は off**、そして **ON のあいだだけ**ヒント文を出す（「メモリ64GB以上を推奨。…メインメモリを約20GB常駐で使用します。生成結果は変わりません（同じシードならビット単位で同一）。」）。prefetch のヒントと同じく**警告ではない**——出力品質の話ではなく、その瞬間に読み手が知るべきなのはメモリの代償だけだからである。既定を off にしたのはサーバー側の既定（`KEEP_RESIDENT_DEFAULT=False`）に揃えたもので、`block_swap_prefetch`（サーバー既定 on）とは向きが逆である点に注意。
- **リクエスト契約**: 既存の単一集約点 `accelerationRequestFields()` に相乗りし、**ON のときだけ** `keep_resident: true` を1キー足す（§55 で確立した「サーバー既定と異なるときだけ明示送信する」規律。§44.7 の事故経路をフロント側でも踏まないための書き方）。既定のままの利用者のリクエストJSONは従来とバイト等価で、キー順を固定している既存テストは無改修で通り続けた。Create・Chain・バッチA2V・バッチi2v-long の4経路すべてがこの関数を通るため、**呼び出し側の改修はゼロ**である。
- **永続化のスキーマ移行**: `localStorage` の `nzvideomni.acceleration` キーは §44（prefetch 追加）で素の文字列からJSONオブジェクトへ移行済みだったので、今回は `keepResident` の1フィールドを足しただけである。`readStoredAcceleration()` は**3つの形**を受ける——①読めない／無い→既定、②2026-08-01 以前の素の文字列（`"sdpa"` / `"sage"`）→そのバックエンド＋他は既定、③現行のJSON。③はさらに**フィールドごとに**既定へフォールバックするので、`keepResident` キーを持たない §48 以前のJSONもそのまま読める。`ThemeContext.tsx` の `readStoredTheme` と同じ防御的な読み方である。
- **触ったファイル**: `webui/src/shell/accelerationSettings.ts`（`keepResident: boolean`・`KEEP_RESIDENT_SERVER_DEFAULT=false`・`accelerationRequestFields()` の1キー加算・`readStoredAcceleration()` のフィールド追加）／`webui/src/shell/useAccelerationSettings.ts`（復元と `useEffect` 側での書き出し、`onKeepResidentChange`）／`webui/src/shell/SettingsPanel.tsx`（トグル行＋ONのときだけのヒント）／`webui/src/i18n/strings.ts`（en/ja の4文言）／`webui/src/api/types.ts`（`GenerateRequest`・`GenerateChainRequest` の `keep_resident?`）。**Context は作っていない**（AppShell が `useState` で持ち props で配る本リポジトリの流儀どおり）。
- **検証**: `npm run typecheck`（`tsc -b`）**0エラー**、`npm run test` **1675 passed**（失敗7件は実バックエンドを起動しているときだけ走る `backend.integration.test.ts` の統合テストで、これは既知の環境依存）、`npm run lint` **0**。バックエンド側は pytest 899 passed（失敗1件はポート18620稼働中のみ落ちる既知の環境依存テスト）＋エンジンvenvの新規27 passed。
- **エージェント実機ゲートは R1〜R7・R9（エージェント担当分）が全項目合格**（2026-08-02深夜〜03未明・清浄環境・全10ジョブ完走・crash 0）。**R8はオーナー目視の別枠**である。HIT時の前処理**5.32秒**（ベースライン68.6〜75.0秒）、ON4本のSHA256一致、OFF復帰時の解放**19.4GB**、chain・MCP経由も完走。実測値の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §48.5。
- **状態**: 実装・機械検証・エージェント実機ゲートまで完了。**残りはオーナー目視ゲート（R8）のみ**——トグルの見た目・既定off・ONのときだけ出るヒント・`localStorage` の永続・4経路でキーが載ること・ja/en の文言・実GPUでのHIT体感の7点で、[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-1／§2-2 に「何を操作して確認するか → どうなれば合格か」の形で起票済み。**実装は backend `c67f860`／frontend `fdc4fc8` でコミット済み（2026-08-03）**で、未コミットは当日の文書追記のみである。
- **追記（2026-08-03）**: 上の7点のうち**7（実GPUでのHIT体感）はオーナーが実機で確認して合格**した（「生成が大幅に高速化すること」を確認）。残るのは見た目・文言・永続化・4経路の細目である。
- **追記（2026-08-03・派生の起票）**: あわせてオーナーから、**先読み block swap が「無効」のときは骨格常駐トグルを自動的に「無効」にしてグレーアウトする**（prefetchを「有効」に戻すと解除）という改修の指示が出た。バックエンドのガードG-C（`block_swap_prefetch=false` × `keep_resident=1` → 警告＋自動off）が正しく働いていても、UI上は「Onにしたのに効かない」と見えるためである。[`PENDING_TASKS.md`](PENDING_TASKS.md) **§1-10** として起票済み（未着手）→ 2026-08-04に実装・合格し、[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-62としてクローズ（本書§60）。
- **追記（2026-08-06・テーマ完結とヒント文の改修）**: **R8の残る細目（4経路・並び順・初回既定off・ヒント文の出方・`localStorage`永続化・ja/enの文言）がすべてオーナー実機／目視で合格**し、本テーマは完結した。台帳は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) **§3-64**（`PENDING_TASKS.md` §1-9は欠番）。合格と同時にオーナーからヒント文のブラッシュアップ指示が出たので、`accelKeepResidentNote`（ja/en）を差し替えた——**実測値「約70秒→約10秒」とビット一致の但し書きを落とし、メモリ常駐量を推奨要件のすぐ横（第1文の括弧内）へ移した**。3文が並列に並んでいた旧文面より、読み手が最初に知るべき「64GB推奨とその内訳」が前に出る。バックエンド同梱Gradioの`accel_info_keep_resident`（`gradio_ui/i18n.py`、ja/en）も同内容へ揃えてある——2つのUIで同じ機能の説明が食い違わないようにするのは§59のヒント文言修正と同じ作法である。

## 59. IC-LoRA Depth（深度制御）・Deblur（ぼけ除去）の追加 — バックエンド主導・フロントエンドは型とヒント文の追随（2026-08-03）

フロントエンド台帳[`PENDING_TASKS.md`](PENDING_TASKS.md) §4-8に残っていた未対応IC-LoRA 4種のうち、**DepthとDeblurの2種を製品化**した。正本はバックエンド[`ICLORA_DEPTH_DEBLUR_WORKORDER.md`](../../../Docs/ICLORA_DEPTH_DEBLUR_WORKORDER.md)、検証記録は[`Nz-Videomni/Docs/VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §49。初期実装はコミット・プッシュ済み（backend `fd6d43f` / frontend `d375028`、2026-08-03）。**G0〜G4の全ゲートがオーナー実機で合格し、ヒント文言の表現修正も完了して、本テーマはテーマ完全完結した（2026-08-04）。**

- **Depthを塞いでいた「前処理器の選定」が確定した**: 公式ComfyUIワークフロー`LTX-2.3_ICLoRA_Union_Control_Distilled.json`をJSONとして直接解読し、公式もVideo-Depth-Anythingの**Small版（vits）**を`input_size=518`・`max_res=960`・`fp32`・グレースケール出力で使っていることを実値で確認した。ライセンスはApache-2.0。ただし同ワークフローで配線が完成しているのは**canny経路のみ**で、depth／poseのノードは「例示」の位置づけである点は留保として記録した。
- **Deblurは前処理不要**: Lightricks公式アダプタの safetensors ヘッダを実測し、`reference_downscale_factor`キーが**値`"1"`で実在**することを確認。既存のメタデータキー判定がそのまま制御系へ自動分類するため、**configスキーマの拡張は不要**と確定した。値が1（参照を縮小せずに条件付けに使う）であることが、エンジン側の縮小係数ガードを緩める必要が生じた理由であり、同時にDeblurのVRAM増（stage-1の総トークン数が、**参照なしを1として、縮小係数2で1.25倍・係数1で2.0倍＝既存の制御系比で約1.6倍**）の理由でもある。
- **APIの新エラー`REFERENCE_REQUIRES_CONTROL_LORA`（422）**: ガード緩和で「参照動画＋画風LoRAだけ」の誤用を弾く網が消えるため、API層に逆方向のチェックを新設した（単発・チェーンの両方）。これは[`PENDING_TASKS.md`](PENDING_TASKS.md) **§4-11の①**（アップロードもジョブ開始も済んだあとで落ちるので、リクエスト時点で422にするほうが親切）を本テーマの副産物として解消したことになる。**G2で実サーバー上の挙動も確認済み**で、バックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §34.6に「早期422化は将来の改善余地」として残っていた既知事項#2もこれでクローズした。
- **フロントエンド側の実装は小さい**: `webui/src/api/types.ts`の`IcLoraEntry.preprocess`へ`"depth"`を追加／`webui/src/i18n/strings.ts`（en/ja）へヒント3種（`controlAdapterAspectHint`・`controlAdapterDepthHint`・`controlAdapterDeblurHint`）／`GenerationForm.tsx`の参照動画セクションへ3行の`field-hint`表示／`mockBridge.ts`のフィクスチャへ`depth-control`（マップ形式）と`deblur`（文字列形式）の2件。**アダプタの選択肢そのものは`/config`経由で自動的に増える**ので、ドロップダウンの改修は不要だった。
- **ヒント文は常時表示にした**: アダプタの選択に連動して出し分ける仕組みは既存に無く（連動しているのは参照動画欄の有効・無効だけ）、そのためだけに新しいUI機構を足すのは避けて、既存の`icLoraNote`と同じ常時表示の注記として並べた。バックエンド同梱のGradio側も同じ判断である。
- **値の自動セットはしない（オーナー決定）**: 深度の推奨値0.6は**ヒント文の表示のみ**。加えてヒント文は、推奨0.6の対象が**制御追従度（`conditioning_attention_strength`）**であり**制御LoRAの強さは1.0のまま**であるという区別を明記している——後者を下げると参照が滲み込む（bleed-through）と公式が警告しているためで、この2つを取り違えやすいという判断による。3つのノブの切り分けはバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §28.1で照合済み。
- **検証**: `npm run typecheck`（`tsc -b`）**0エラー**、vitest **1673緑**。バックエンド側はアプリ用仮想環境のpytest **910 passed / 9 skipped**（着手前900）、エンジン用仮想環境 **28 passed**（着手前15）、前処理の自己診断 **11/11 PASS**、Gradio系 **277緑**。深度前処理のG1ゲート（近＝白184.07対15.49／静止画素フリッカ平均0.561・p95 0.693〔閾値2.0〕／mp4往復PSNR 35.67dB〔閾値6.0〕）も**全項目PASS**。
- **G2（モック通し）＝全項目PASS（2026-08-03）**: 隔離config（`backend: "mock"`・ポート18902・出力先とアップロード先もscratchpadへ隔離）で**実サーバーを起動**し、HTTP経由で6項目を確認した（リポジトリ側・`outputs/`・`uploads/`への書き込みはゼロ）。①`depth-control`＋参照＋制御追従度0.6の単発生成が202→completedで`metadata.json`に`preprocess: depth`／参照ID／`0.6`が正記録 ②`deblur`＋参照の単発生成が完走 ③**1クリップchain（Create画面のA2V相当）が完走**（`num_frames=25`。17は`overlap_frames=3`との既存の境界バリデーションで422になるが、これはバグではない）④2クリップchain＋制御系は`422 LORA_CONTROL_UNSUPPORTED_IN_CHAIN`（既存仕様の維持）⑤参照＋画風系LoRAのみは単発・chainの両方で`422 REFERENCE_REQUIRES_CONTROL_LORA` ⑥`/config`・`/loras`に新2件が出現し、`deblur`は実メタデータ由来で`kind: control`。実測はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §49.7。
- **状態**: **テーマ完全完結（ヒント文言確定済み）**。バックエンドの**G3（オーナー実機real）・G4（既存canny/pose/upscalerの回帰）とも2026-08-04にオーナー実機で合格**した。G3ではdepth-controlの通常経路とA2V経由（2026-08-03）に続き、deblurの完走と目視品質（G-fix3）、**DeblurのVRAM実測**（ピーク13.9GBで16GB推奨環境に収まるためVRAM警告は不要とオーナーが裁定）を確認。G4は改修前コミット`24c67fd`と現行コミットの同一条件3本（canny/pose/upscaler）がSHA-256完全一致。ヒント文言中の不正確な「出力と同じ解像度で処理」という表現も2026-08-04に「縮小せずに条件付けに使う」へ置換済み（`gradio_ui/i18n.py`・webui `strings.ts`の両方、再ビルド・再デプロイ済み）。定義・実測の正本はバックエンド[`ICLORA_DEPTH_DEBLUR_WORKORDER.md`](../../../Docs/ICLORA_DEPTH_DEBLUR_WORKORDER.md) §7と[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §49.11。初期実装はコミット・プッシュ済み（backend `fd6d43f` / frontend `d375028`、2026-08-03）。第3修正・G-fix3・G4・ヒント文言修正に対応する実装差分のコミット・プッシュは未実施。**追記（2026-08-04）**: 実施済み（backend `e273052`〜`9701fbe`／frontend `df85d3f`）。

## 60. 骨格常駐トグルのprefetch連動グレーアウト（実効値方式）（2026-08-03）

[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-10（§58の実機運用から派生し2026-08-03に起票）を実装した。先読み block swap（prefetch）が「無効」のあいだ、モデル骨格の常駐トグルを自動的にOff表示＋操作不能にし、prefetchが「有効」に戻ったら自動的に元の選択へ復帰する。

- **背景**: バックエンドのガードG-C（`block_swap_prefetch=false`×`keep_resident=1`→警告つきで常駐だけ自動off）は正しく働いているが、Settings画面には「有効」のまま映り続けるため、利用者には「Onにしたのに速くならない」としか見えない。サーバー側は触らず、フロントエンドの見せ方だけを変えて、そもそも成立しない組み合わせを選べなくする方針とした。
- **実装時に決めることだった二択の決着**: グレーアウト中に`localStorage`の保存値をどう扱うかは、**表示上だけ「無効」に見せて保存値は保持する**方を採用した（値を`false`へ書き戻す案は不採用）。表示・リクエスト送出の両方を`accelerationSettings.ts`の`keepResidentEffective`／`effectiveAcceleration`（保存On かつ prefetchトグルOn かつ 能力ゲートが明示的falseでない、の3条件で導出する実効値）に一本化した。能力ゲートの畳み込みはAppShellの1箇所に集約し、リクエスト組み立ての既存3地点（Create・Chain・バッチ）は無改修のまま実効値を受け取るだけになる。
- **見た目**: グレーアウト中はOff表示＋`disabled`属性＋ツールチップ。新規文言キー`accelKeepResidentPrefetchOffTooltip`（ja「先読みblock swapが有効なときだけ使えます」）をen/ja両方に追加した。
- **検証**: webui 8ファイル改修、`npm run typecheck`0エラー、`npm run test` **1682 passed**。
- **状態**: **目視合格・テーマ完全クローズ（2026-08-04）**。ビルド・デプロイ済みで、実機（AviUtl2側）とバックエンドリポジトリ配布コピー（`Nz-Videomni\AviUtl2-Plugin\NzVideomni.aux2`）の両方がビルド成果物とSHA-256一致（`38F7FAED...`）を確認済み。コミット・プッシュ済み（frontend `b39eaa0` / backend `e583c03`、2026-08-03）。**オーナー実機目視で、prefetch Offにより常駐トグルがOff表示＋グレーアウトし、prefetch On復帰で自動的に元の選択へ復活することを確認（2026-08-04）。残作業なし。**

## 61. GGUF逆量子化の1カーネル化（`fused_gguf_dequant_kernel`）— バックエンド主導・フロントエンドは5つ目のトグルと旧モックの撤去（2026-08-04）

[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-11 を実装した。**正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §51**（実装・自己検証・実機ゲートG1〜G8・既定値反転の実測値はすべてそちらにある）。本節はフロントエンド側から見た要約である。

- **何をした機能か**: GGUFファイルの中で圧縮された形で持っている重みを、計算に使える形へ展開する処理（逆量子化）を、量子化の形式ごとに1個のGPU処理へまとめた。これまでは1回の展開のたびに18〜33個の細かいGPU処理に分かれていた。**生成結果は1ビットも変わらない**——同じシードならOn/Offでまったく同じ動画になることを、必須要件として設計・検証してある。
- **効果**: 768p・257フレームの交互対比較3組で**約17.5%短縮**（中央値144.5秒→119.2秒）。VRAMの予約量は変わらず、実確保量はむしろ約386MB減った。
- **フロントエンドの改修**: Settings > Acceleration の**5つ目**の切替として追加した（`accelerationSettings.ts`の型・定数・保存・読み出し・リクエスト組み立て／`useAccelerationSettings.ts`／`SettingsPanel.tsx`／`AppShell.tsx`／`api/types.ts`／`i18n/strings.ts`／バッチA2Vの2ファイル）。**この機能には可否ゲートが無い**——サーバーが自力で従来実装へ降格するので、クライアント側で押せなくする第二の真理の源を作らない、という既存の判断（attention選択と同じ）に揃えた。したがって§1-10のようなグレーアウトの連動も無い。
- **旧モック`fused_gguf_dequant_gemm`は完全撤去した**（オーナー裁定2026-08-04）。受理はするが生成に一切影響しなかった項目で、新しいトグルが画面上の同じ位置を引き継いだ。§55で入れた「モック2件」は`vae_mode`の1件だけになり、モックの作法（常時disabled＋ツールチップ）とそのi18n資産はそちらに残る。
- **バッチA2Vの配線に注意点があった**: バッチだけは`useBatchForm.ts`の手書きのOR判定（「どれか1つでも既定と違えば`acceleration`をランナーへ渡す」）と、`buildA2vChainPayload.ts`のローカルなsnake_case型という2箇所を別途足す必要がある。ここを忘れると**バッチA2Vでだけトグルが効かない**という、他のすべての自動テストをすり抜ける不具合になる。単発・チェーン・バッチの3経路すべてで実際にキーがワイヤーに載ることを、専用の回帰テスト（`fusedGgufDequantKernel.paths.test.ts`）で押さえた。
- **既定値の反転で送信の向きが逆になった**: 実機ゲート全項目合格を受けて、2026-08-04にオーナー承認のうえサーバー既定を`false`から`true`へ反転した。フロントエンドは以前から「**サーバー既定と異なるときだけ送る**」規約で書いてあるため、送信ロジックは1行も変えずに済んだが、**キーが載るのはOFFにしたときだけ**へ向きが変わった（`block_swap_prefetch`と同じ向き、`keep_resident`とは逆向き）。テストの向きもこれに合わせて更新してある。**サーバー既定はフロントエンドのバンドルに焼き込まれる値なので、反転にあたって再ビルドと再デプロイが必須**である（実施済み）。
- **検証**: `npm run typecheck`（`tsc -b`）**0エラー**、`npm run test`（vitest）**1700 passed**。バックエンド側はアプリ用仮想環境のpytest **941 passed**（唯一のFAILは実機ゲート用にサーバーを起動したままにしていた環境要因）、エンジン用仮想環境 **9 passed**、Tritonカーネルの自己検証 **C1〜C10全PASS**（実GGUFから機械列挙した31形状すべてビット一致）。
- **状態**: **テーマ完結（2026-08-06）**。ビルド・デプロイ済み（実機とバックエンドリポジトリ配布コピーの2か所）。**2026-08-06にオーナー目視の4項目（既定Onでの表示と日英切替・旧モックの消滅・切替と保存・On時の生成完走と高速化の体感）がすべて合格**し、台帳は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) **§3-65**へ移してクローズした（`PENDING_TASKS.md` §1-11は欠番）。コミットは frontend `d8ba267`／backend `aed4773`。

## 62. PrunaVAED（枝刈り版の映像VAEデコーダ）— Acceleration最後のモック`vae_mode`の実機能化。バックエンド主導・フロントエンドは5つ目の行の有効化（2026-08-05）

[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-12（旧§3-50）を実装した。**正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §52**（設計・重み変換・実機ゲートG1〜G7の実測値はすべてそちらにある）。仕様と設計判断の正本はバックエンド[`PRUNAVAED_WORKORDER.md`](../../../Docs/PRUNAVAED_WORKORDER.md)。本節はフロントエンド側から見た要約である。

- **名称の訂正が入った**: 台帳や既存のUIでは「**PruneVAED**」と書いていたが、これは上流の正式名称の誤記だった。正しくは **PrunaVAED**（開発元 Pruna AI の名を冠したもの。Hugging Face の `PrunaAI/PrunaVAED`）である。表示名・文書はすべて PrunaVAED へ統一した。**ただしAPIのフィールド値 `vae_mode="prune_vaed"` は改名していない**——これは既存の外部コントラクトであり、表示名の訂正に追随させると古いクライアントが壊れるためである。この「表示名だけ直し、ワイヤー上の値は据え置く」という切り分けは`strings.ts`のコメントにも明記してある。
- **何をした機能か**: 映像の復元処理（潜在表現から実際の映像フレームを作る処理＝映像VAEデコード）を、枝刈り（pruning＝寄与の小さいチャンネルを削ること）と蒸留（distillation＝軽くしたモデルに元の出力を教え込むこと）を施した軽量デコーダへ、**ジョブ単位で**差し替えられるようにした。**§1-11（GGUF逆量子化の1カーネル化）とは性質が正反対で、こちらは出力の絵が変わる**——sage attention と同じ「速くなるが結果が変わる」系の項目である。
- **効果**: 768p・257フレームの交互対比較4組で**合計生成時間が平均12.54秒（10.52%）短縮**（119.20秒→106.66秒）、VRAMの予約量が**2.6GB減**（13,911→11,276MB）。デコード区間そのものは1.608倍で、合計短縮の98.1%がこれで説明できる。画質はオーナー目視で「劣化は肉眼ではほとんど分からない」と判定された（PSNRの実測は36.06dB）。
- **既定は恒久OFF**である。§1-11や先読みblock swapのような「ゲート合格後に既定をONへ反転する」ステップは**置かない**（オーナー確定事項）。絵が変わる機能は利用者が意図して選ぶべきもの、という判断による。したがって**送信の向きは「`prune_vaed`のときだけキーを載せる」**で固定であり、`keep_resident`と同じ向き（`block_swap_prefetch`・`fused_gguf_dequant_kernel`とは逆向き）である。
- **フロントエンドの改修**: Settings > Acceleration の**VAEの行（5つ目）**は§55以来ずっと`disabled`のプレースホルダだったので、それを外して`onClick`を配線した（`accelerationSettings.ts`の型・`localStorage`の保存と読み出し・リクエスト組み立て／`useAccelerationSettings.ts`の`setVaeMode`／`SettingsPanel.tsx`／`AppShell.tsx`／`api/types.ts`のMOCK注記／`i18n/strings.ts`）。**注意文（`accelVaeNote`）は他のトグルのヒント文と違って「警告」である**——prefetch・keep_resident・fused kernelの3つはいずれも「結果は変わりません」と書けたが、ここは書けない。文面はオーナー指定の「**出力品質がわずかに低下する可能性があります**」を軸に、サーバー側に枝刈りデコーダが無ければ自動で通常のデコーダが使われ生成は止まらない旨を添えた（`accelSageNote`と同じ性格の文言）。
- **モックの作法とそのi18n資産はここで役目を終えた**。§55で入れた「常時disabled＋『まだ実装されていません』ツールチップ」という作法は、§61で1件（`fused_gguf_dequant_gemm`）が撤去され、今回`vae_mode`が実装されたことで**Accelerationからモックが1件も無くなった**。将来またモック枠を作るときは§55とこの節を読み返すこと。
- **バッチA2Vの配線は§61と同じ2箇所を別途足した**: `useBatchForm.ts`の手書きのOR判定（「どれか1つでも既定と違えば`acceleration`をランナーへ渡す」）と、`buildA2vChainPayload.ts`のローカルなsnake_case型。ここを忘れるとバッチA2Vでだけトグルが効かないという、他のすべての自動テストをすり抜ける不具合になる。単発・チェーン・バッチの3経路すべてで実際にキーがワイヤーに載ることを、専用の回帰テスト`vaeMode.paths.test.ts`（§61の`fusedGgufDequantKernel.paths.test.ts`と同型）で押さえた。
- **検証**: `npm run typecheck`（`tsc -b`）**0エラー**、`npm run test`（vitest）**1728件PASS**（着手前1711件・**+17件・退行ゼロ**）。バックエンド側はアプリ用仮想環境のpytest **946 PASS／4 skip**（唯一のFAILは実機ゲート用にサーバーを起動したままにしていた既知の環境要因）、エンジン用仮想環境**91件PASS**、G2自己検証**6項目全PASS**、変換ツール**89件（75 PASS／14 skip）**。
- **MCPサーバーにも公開済み**（G9・2026-08-05）。`submit_generate`／`submit_chain`の両方に`vae_mode`が列挙値2つで現れ、**ツール本数は22のまま不変**。外出先からMCP経由で操作するときも、AviUtl2の設定パネルと同じ5項目が揃う（バックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §52.10-b）。
- **状態**: **実装完了・実機ゲートG1〜G7＋G9全項目合格（オーナー目視も2026-08-05に合格）**。残るのは**フロントエンドの再ビルド・デプロイ**と、コミット（オーナーの手動作業。バックエンド・フロントエンド・変換ツール`Nz-GGUF-Converter-LTX23`の3リポジトリ）である。**再ビルドしないとVAE行はAviUtl2実機でグレーアウトのままになる**点に注意（既定値は変わらないが、行を操作可能にした変更そのものがバンドルの中にある）。

## 63. Editタブの実タブ化とサブタブ（Retake／Outpainting／Inpainting）の骨組み新設（2026-08-09）

[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-13（Outpainting＝動画キャンバス拡張）の**前提作業**として、2026-08-08のタブ改称のときにモック（押せないタブ）のまま残してあったEditタブを実タブへ昇格させ、その配下にサブタブの枠を作った。設計方針の正本は[`OUTPAINTING_DESIGN_NOTES.md`](OUTPAINTING_DESIGN_NOTES.md)で、同書§5の再調査項目8「Editタブの実タブ化」がここで消し込まれる。**Outpainting本体の機能は一行も実装していない**——この節は「置き場所を先に作った」という記録である。

- **Editが実タブになった**: `AppMode`（`shell/AppShell.tsx`）に`"edit"`を足し、`remountTokens`へ`edit: 0`を足し、既存3画面とまったく同じ常時マウント方式（`<div role="tabpanel" hidden={mode !== "edit"}>`）でパネルを1つ増やした。`shell/ModeTabs.tsx`側は`MockTabId`から`"edit"`を外したので、**残るモックタブはToolboxの1つだけ**になった。タブの並び（Toolbox／Single／Chained／Edit／Inventory）と初期モード（Single）は変えていない。右クリックの転送先を表す`MenuTargetMode`は`AppMode`とは別の独立した型なので、こちらは無改修である（Editは右クリックの行き先にはならない）。
- **ジョブトーストの転送先にEditも加えた**: ジョブ台帳（生成ジョブの一覧表）はCreate／Chainの2画面にしか無く、`handleSelectJob`は「台帳を持たない画面にいるならSingleへ切り替えてから強調する」という作りになっていた。この判定はこれまで`inventory`だけを見ていたため、Editタブを表示中にトーストを押すと**何も起きないように見える**不具合になるところだった。判定を`inventory`または`edit`へ広げてある。既存コメント（MJ-5b）の意図をそのまま延長した形である。
- **サブタブ機構はこれが最初の1つ**である（アプリ内に前例が無かった）。`modes/edit/EditSubTabs.tsx`は`shell/ModeTabs.tsx`の作法をそのまま写している——**`disabled`を判別子にしたunion型**（無効タブ側だけがモック用のidを許すので、`onChange(tab.id)`を呼ぶ有効タブ側では型が自動的に実体のあるモードへ絞られる）と、**無効タブには`onClick`ハンドラそのものを付けない**（`disabled`属性だけに頼らない）という2点である。並びはRetake／Outpainting／Inpaintingで、**Inpaintingだけがモック**（`.mode-tab:disabled`と同じ薄い表示＋`cursor: not-allowed`。Windowsでは標準の🚫カーソルが出る）。Inpaintingにはパネルを作っていない——サブタブ自体が押せないので、置いても到達不能なコードになるためである。
- **サブタブのパネルには`role="tabpanel"`を付けていない**。これは意図的な制約で、アプリ全体のテスト7ファイルが`getByRole("tabpanel")`の**単数取得**で「いま見えている画面」を掴む書き方をしているため、Edit表示中にtabpanelが2つ見えるとその前提が壊れる。ariaの紐付けも既存のメインタブと同水準（`role="tab"`と`aria-selected`まで）に留め、`aria-controls`等は付けていない。この判断はコード側のコメントにも残してある。
- **サブタブの選択状態はEditScreenの内部`useState`で持つ**（AppShellへ持ち上げていない）。Editのサブモードへ外部から飛んでくる経路が無いので、シェルへ上げると読み手のいない第2の真理の源になる。タブを跨いでも状態が残るのは、AppShellが全画面を常時マウントしているおかげで自然に成立する。2つのパネル自体も`hidden`属性で出し分けており（`display`系のCSSを書かずにUAの`[hidden] { display: none }`に任せる、AppShellと同じ作法）、将来パネルがフォーム状態を持ってもサブタブ切替で消えない。
- **文言**: `i18n/strings.ts`のen/ja両方に`edit`名前空間を新設した。**サブタブのラベル（Retake／Outpainting／Inpainting）は日英で同一表記**にしてある——2026-08のタブ改称でメインタブのラベルを日英同一にしたオーナー決定に揃えた。日本語側は見出しに短い解説を添える書式（「Retake（リテイク：撮り直し）」「Outpainting（アウトペインティング：画面の描き足し）」）を採り、機能の一行説明と「準備中」の一文を本文に置いた。
- **変更ファイル**: 新規5点（`modes/edit/EditScreen.tsx`／`EditSubTabs.tsx`／`RetakePanel.tsx`／`OutpaintingPanel.tsx`／`EditScreen.css`）＋新規テスト2点（`modes/edit/EditScreen.test.tsx`／`App.editTab.test.tsx`）、改修5点（`shell/AppShell.tsx`／`shell/ModeTabs.tsx`／`shell/ModeTabs.test.tsx`／`shell/AppShell.css`／`i18n/strings.ts`）。
- **検証**: `npm run typecheck`（`tsc -b`）**0エラー**、`npm run test`（vitest）**1734 PASS／10 skip**（新規10件・退行ゼロ）、`npm run lint`は既存25件の警告のみでEdit関連の新規指摘はゼロ。
- **状態**: 実装・機械検証は完了。ビルド・デプロイ済み。**残りはオーナー目視ゲート**——タブバーでEditが押せること・サブタブ3つの見た目と並び・Inpaintingの薄い表示と🚫カーソル・日英の文言の4点である。

## 64. Edit系右クリック導線の共通スパイン（W0）— Outpainting／Retakeの2項目とEditタブへのルーティング配線（2026-08-09）

§63でEditタブを実タブ化しサブタブの枠まで作ったので、その続きとして**右クリックからEditタブへ入る導線の共通部分**を通した。Outpainting本体（[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-13、設計方針の正本は[`OUTPAINTING_DESIGN_NOTES.md`](OUTPAINTING_DESIGN_NOTES.md)）もRetake本体も、**機能そのものは一行も実装していない**——この節は「配線だけ先に通した」という記録である。パネルの中身は後続のワークストリームが実装する。仕様の正本は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §3-4（#15・#16）・§4（範囲系ガード）・§5-9（配置系統D）。

- **メニュー項目が2つ増えて、オブジェクト右クリックは9項目→11項目になった**。`native/src/plugin.cpp`の`kObjectMenuItems`に`outpaintVideo`（🎬 この動画を描き足す (Outpainting)）と`retakeRange`（🎬 選択範囲を作り直す (Retake)）を追加し、`Language/Japanese.NzVideomni.aul2`と`English.NzVideomni.aul2`に対応する行を足した。**C++の改修はこのメニュー登録だけ**で、ブリッジのRPCは1本も増えていない（`timeline.menuInvoked`の`action`が2つ増えるだけなので、ブリッジ契約のバージョンは不変）。
- **絵文字は2項目とも🎬にした**。作業計画の当初案ではOutpaintingに🖼を当てていたが、既存メニューの絵文字は§3-2の約束どおり**必須種別**を表している（🎬＝動画／🖼＝画像／🎵＝音声／📝＝テキスト）。Outpaintingは「画面を広げる」機能ではあるが右クリックする対象は動画オブジェクトなので、🖼だと「画像を右クリックする項目」と誤読される。**絵文字は機能の絵ではなく、右クリックすべき対象の種別を指す**、というのが既存の一貫した読み方である。
- **配置系統(D)を新設したが、nativeには足していない**。Retakeの仮オブジェクトは「素材の頭」ではなく「**ユーザーが選んだ範囲の頭**」に揃えたいので、位置の意図としては既存の(B)と別物である。しかし着手前に`native/src/bridge_core.cpp`の`ResolveProvisionalPlacement`を現物で確認したところ、(B)（`kSameStartFront`）は`layer = layer_max + 1`／`frame = material_frame_start`で、**(D)と計算式が完全に同一**だった。そこで系統名だけwebui側に足し、`provisionalReservation.ts`の`placementParams`で(D)→(B)へ写像し、呼び出し側（`AppShell`）が素材のフレーム範囲ではなく選択範囲を渡し分ける形にした。`bridge/types.ts`・モックブリッジ・nativeはいずれも無改修である。
  - **この写像が安全な理由**を明記しておく。ワイヤー上の`placement`は`bridge/types.ts`でリテラル型`"A" | "B" | "C"`と定義されており、`placementParams`の戻り値型が`Pick<ParamsOf<"timeline.insertProvisional">, …>`なので、**"D"を1箇所でも写像し忘れると型検査で落ちる**。if文の連鎖には網羅性チェックが効かないという弱点があるが、ここは戻り値型が安全網として機能している（native側の`ParsePlacementFields`は"A"/"B"/"C"以外をエラーで弾くので、写像漏れを実行時まで持ち越すと確実に壊れる。それを型で塞いだ）。
- **`hasRange`というフックがここで初めて起きた**。Retakeは選択範囲そのものが入力なので、範囲が選ばれていなければ弾く。判定に使う`timeline.getSelection`の`hasRange`は契約v5からnative側が`EDIT_SECTION::info->select_range_start/end`から埋めていたが、**webuiの本体コードでは一度も読まれていなかった**（テストとモックブリッジにしか登場しなかった）。ガードの順序は種別チェックが先で、動画ですらないオブジェクトには種別ミスマッチのほうを案内する。
- **「範囲が短すぎる」の案内は器だけ先に確保した**。`MenuGuardNote`のunionに`rangeTooShort`を足し、日英の文言と`formatMenuGuardNote`の分岐まで用意したが、**判定そのものは入れていない**。最小窓のしきい値はfpsに依存するのでRetake本体を実装する側が決めるべきものだからである。このunionは後続の2つのワークストリームが両方とも触る衝突点なので、**形だけ先に確定させておく**という判断をした（呼び出し元がまだ無いことは承知のうえで、書式テストだけは通してある）。
- **EditScreenへのintent受け渡しは、新しい型を一切作らずに済んだ**。Create／Chainが`initialIntent`として受けている`GenerationPrefill`が、そのままEditに必要な情報を全部持っていたからである（`intent`と、素材のファイルパス・レイヤー・フレーム範囲・解像度・実尺を含む`selection`、および選択範囲`hasRange`／`rangeStart`／`rangeEnd`とプロジェクトの`rate`／`scale`）。Edit専用のスナップショット型を新設する案は、**既存の型で足りるので捨てた**。サブタブの自動選択（outpaint→Outpainting／retake→Retake）も、専用モジュールや純関数を作らず`useState`の遅延初期化の中の三項演算子1つに収めた——右クリックのたびに`remountTokens.edit`が上がって画面が作り直されるので、prop追従のeffectは要らない。
- **`remountTokens.edit`は既存コードがそのまま動いた**。§63で「Editは右クリックの行き先にならないが網羅性のために置いてある」と書いた枠が、`setRemountTokens(prev => ({...prev, [target]: prev[target] + 1}))`という既存の1行だけでそのまま機能した。専用の分岐は書いていない（コメントだけ現状に合わせて直した）。
- **敵対的レビューで拾って直したもの**: `AppShell`内の`placeProvisional`ヘルパーがローカルに`placement: "A" | "B" | "C"`という型を持っており、(D)を通すとここで型エラーになる——計画では見落としていた。ほかにテストの固定件数（14→16）、`MenuTargetMode`／`MenuPlacement`の許容値リスト、そして**§63で書いた「Editのサブタブ状態をAppShellへ持ち上げない根拠」のコメントが前提ごと崩れる**点（「Editのサブモードへ外部から飛んでくる経路が無い」が今回で成立しなくなった）も指摘され、根拠文を書き直した。結論（持ち上げない）自体はリマウント方式で維持できている。
- **敵対的レビューの指摘のうち採らなかったもの**: (1)「(D)は要らない、(B)のまま呼び出し側で出し分ければ型もテストも増えない」——差分は確かに小さくなるが、そうすると`AppShell`側に`action === "retakeRange"`というアクション名の特別扱いが生まれ、かつルーティング表の(B)の説明（「素材と頭を揃える」）が嘘になる。**位置の意図が違うものは別の名前で呼ぶ**ほうが後から読める、と判断した。(2)「`rangeTooShort`は呼び出し元ゼロなので後続で足せばよい」——衝突点を先に固定するのがW0の役目なので残した。(3)「パネルに未使用のpropを持たせるのは無駄」——これは採り、`EditPanelProps`という型は定義して後続へ渡す一方、パネルの引数には**まだ生やしていない**（`_props`のような書き方はこのリポジトリに前例がゼロだった）。
- **配線そのものを通しで押さえる新規テストを1本だけ足した**（`webui/src/App.editRoute.test.tsx`）。W0が納品するのは「ネイティブのイベント→ルーティング→ガード→タブ切替→リマウント→サブタブ自動選択→配置系統Dがワイヤーに載る」という**一本の背骨**であり、これは単体テストのどれ1つでも証明できない。特に「選択範囲の開始フレーム(48)が載っていて、素材オブジェクト自身の開始フレーム(10)ではない」ことをアプリ全体の経路で確認している。
  - この新規テストを書くときに1つ落とし穴を踏んだので記録しておく。`handleRoute`は非同期（§6-bの再同期を待つ）で、かつモックブリッジは`delayMs: 0`でも**`setTimeout`（マクロタスク）**でRPCを解決する。そのため`act()`の中でマイクロタスクをいくら流しても経路は完了せず、アサーションが先に走って落ちる。すべて`waitFor`／`findBy*`（実タイマーを跨いでポーリングする）で待つ必要がある。「何も起きないこと」を確かめるテストも、先に**肯定的な信号**（タブ切替、あるいはガードのノート表示）で待ち合わせてから不在を確認している——そうしないと単に経路を追い越しただけで通ってしまう。
- **変更ファイル**: webui 改修10点（本体6点＝`timeline/menuRouting.ts`／`timeline/menuSelection.ts`／`timeline/provisionalReservation.ts`／`shell/AppShell.tsx`／`modes/edit/EditScreen.tsx`／`i18n/strings.ts`、テスト4点＝`timeline/menuRouting.test.ts`／`timeline/menuSelection.test.ts`／`timeline/provisionalReservation.test.ts`／`modes/edit/EditScreen.test.tsx`）、webui 新規1点（`App.editRoute.test.tsx`）、native 1点（`native/src/plugin.cpp`）、言語ファイル2点（`Language/Japanese.NzVideomni.aul2`／`English.NzVideomni.aul2`）、文書2点（本節と[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md)）。**新規は通し配線テスト1本だけ**で、本体側の新規ファイルはゼロである。
- **検証**: `npm run typecheck`（`tsc -b`）**0エラー**、`npm run test`（vitest）**1761 PASS／10 skip**（着手前1734 PASS／10 skip・**+27件・退行ゼロ**）、`npm run lint`は既存25件の警告のみで**新規指摘ゼロ**（変更したファイルには1件も出ていない）、`scripts/build.ps1`によるネイティブビルド**成功**（`NzVideomni.aux2`生成。警告は既存の`bridge_core.h`のC4819のみで新規指摘なし）。`plugin.cpp`がASCIIのみという同ファイルの制約も維持している（絵文字はUCNエスケープ）。言語ファイルはUTF-8（BOMなし）・CRLFという既存の書式を保っている（追加後も27行すべてCRLF）。
- **状態**: 実装・機械検証は完了。**デプロイはしていない**（後続ワークストリームとまとめて判断する）。パネルの中身が未実装なので、この段階でメニューから飛んでも「準備中」のプレースホルダが出るだけである点に注意。

---

## 65. 仕上げの前進幅（stage-2の窓）を選べるようにした件と、チェーンの解像度警告（W1）（2026-08-09）

**やったこと**: バックエンド §3-57 のスイープ結論を製品に落とし込み、あわせて §1-14（チェーンの解像度が快適な範囲を超えたときの警告）を恒久策として実装した。バックエンドとフロントエンドの両方にまたがる。実装の背景と測定値の正本はバックエンドの[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §53、手順書は`Nz-Videomni/outputs/stage2_window_optin/RUNBOOK.md`。

- **前提が調査で覆った件を先に書いておく**。台帳 §3-57 のUI方針は「解像度から窓サイズを**自動選択**する（既定は自動）」だったが、実測の結果**不採用**になった。数値上は窓19（継ぎ目のMAD比0.39）が窓22（1.05）より2倍以上良かったのに、**オーナーの目視所見は逆**だったからである。2つの窓は崩れ方の種類が違い、窓22は「乱視のようなブレ」、窓19は「絵がじわりと別物へ変わっていくモーフィング」だった。MAD比は隣り合うフレームの差を見る指標なので、**後者は原理的に検出できない**——むしろ差が小さいまま進行するので「良い」方向に振れてしまう。**自動で選ばせるには判定指標が信用できなかった**、というのが不採用の理由である。結論は「既定は22据え置き、窓19・のり代7を手動で選べるようにする（注意文つき）」。
- **UIの見せ方は「秒」で、APIは「幾何」で名乗る**。APIのフィールド名を`"6sec"`のような秒数にする案は採らなかった。窓の前進量は潜在フレーム数なので、実時間の長さは`frame_rate`次第だからである（同じ設定でも24fpsなら6.0秒、30fpsなら4.8秒）。そこで`stage2_window: "standard" | "high_resolution"`という幾何の名前をAPIに置き、**秒への変換は画面側が現在のフレームレートから毎回導出する**。画面のラベルは「仕上げの前進幅」で、「窓」「タイル」「latent」はUI文言に一切出していない（テストで文字列の不在を確認している）。
- **この秒の導出で1つ踏んだ**。前進量をピクセルに直すのは`v_adv × 8`であって、`px_from_v_latent`（`(n-1)×8+1`）ではない。あの`+1`は「区間の先頭にある因果キーフレーム」なので**窓の長さ**には付くが、窓の開始位置どうしの差である**前進量**には付かない。最初`px_from_v_latent`で書いてしまい、オーナー指定の「6.0秒／4.0秒」にならず5.7秒／3.7秒になった。**期待値をテストに先に書いておいたおかげでその場で気づけた**（`chain_math`も`audio_adv`を`v_adv × VIDEO_TIME_FACTOR`から導いており、同じ区別をしている）。
- **既定時にキーを送らない側に倒した**。`buildChainRequest`で`stage2_window`は既定（`"standard"`）のとき**キーごと出さない**。同じファイルの`chunked_upsample`は逆に「常に明示」なので一見不統一に見えるが、実際にはこの`buildChainRequest`の任意フィールド（`crop_output`／`source_video`／`loras`／`reference_video_id`ほか）は**すべて「出さない」方式**で、`chunked_upsample`だけが例外である。今回は多数派に合わせた。サーバー既定がそのまま欲しい値なので、黙っているのが正しい。バックエンド側の`test_ltx_runner_payload.py`がワーカーへ送るペイロードの**キー集合そのもの**を固定しているため、ここを間違えると既存テストが落ちる仕組みになっている。
- **V2Vとの組み合わせに穴が1つ見つかった**。窓19で引き継ぎ145フレームを指定すると、凍結した頭が**最初の窓を丸ごと埋め尽くす**（19潜在フレームの窓に19フレームの頭）。生成すべきものが何も残らない、検証されたことのない状態である。しかも既存のガードは「頭が窓より**大きい**とき」しか弾かないので（19は19より大きくない）素通りしてしまう。`stage2_max_context_px(v_tile) = px_from_v_latent(v_tile - 1)`を導入し、**窓19のときだけ137フレーム上限**にした。窓22での同じ計算は161で、公開している`v2v_context_frames_max`（145）より大きいため**既定の挙動は何も変わらない**。公開値145そのものは変更していない。画面側でも同じ式で生成ボタンを止めるようにした（サーバーの422が利用者に届かないようにするという既存方針に合わせた）。
- **C++側の同じ定数には手を出さなかった**。`native/src/timeline_math.h`にも`kStage2VTile = 22`という写しがあるが、プラグインには前進幅を選ぶ手段が無く、プラグイン自身が出すものにとっては22が正しい値である。**追従させると逆に壊れる**ので、コメントだけ「22は既定の窓の値であり、WebUIが選べる別の窓では上限が違う」と書き足した（動作は変えていない）。
- **敵対的レビューで拾って直したもの**: (1)`stage2_max_context_px(22)`を計画では169と書いていたが正しくは**161**（`px_from_v_latent(21)`）。テストの期待値に書く前に直した。(2)Gradio側の関数に引数を足すとき、`lang`が`ui.py`から**位置引数**で渡されているので**`lang`より後ろ**に置く必要がある（前に入れると3か所の配線が全部ずれる）。テストで順序を固定した。(3)C++の写し（上記）の存在。(4)非24fpsの検証を「代表的な組み合わせ」で済ませず**総当たり**にすべきという指摘——これは効いた（次項）。
- **総当たりにしたら、当初の想定が間違っていたことが分かった**。「窓を変えても失敗する組み合わせは同じはず」と考えて計画に書いていたが、フレームレート11種×クリップ長6種×本数7種×のり代3種を全部回したところ、**23.976fpsと30fpsでのみ食い違う**ことが判明した。しかも双方向で、窓22だけが落ちる構成も窓19だけが落ちる構成もある（各4件）。いずれもGPUを動かす前の422なので壊れた動画が出るわけではない。作業指示どおり**新しい検証は足さず**、代わりにこの8件を`KNOWN_WINDOW_DIVERGENCES`として列挙して固定した。あわせて「**クリップ1本の構成ではどのフレームレートでも食い違わない**」ことを独立のテストにした——A2Vは1本限定で、その音声長の事前チェックは既定の窓で計算するので、ここが食い違うと「事前チェックは通ったのに窓の選択だけを理由に422になる」ことになるからである。**計画に書いた「同一のはず」という主張をそのままテストにしていたら、嘘を固定するところだった。**
- **敵対的レビューの指摘のうち採らなかったもの**: (1)「Gradio側への引数追加は、Gradioに窓の選択UIが無い以上どこからも既定以外が渡らないので丸ごと削れる」——指摘自体は事実だが、作業指示に明記された項目であり、また6行程度で「同じ幾何を共有している」ことをコード上に残せるので実装した（判断は親へ回す）。(2)「`shell/tokenBudget.ts`ではなく`modes/chained/chainUtils.ts`に置くべき」——置き場所は並行するワークストリームとの取り決めなので指示どおりにした。ただし**チェーン専用であることをファイル冒頭と関数名で明示**し、`modes/`から何もimportしない自己完結にしてある。(3)「40,000という予算は既存の`spill_free_frames`と二重の快適判定になる」——同じ軸ではない（あちらは「1クリップの**長さ**の上限」、こちらは「仕上げの窓1枚の**広さ**の上限」）ので両方残したが、**なぜ2系統なのか**を両側のコメントに書いた。
- **変更ファイル（フロントエンド）**: webui 新規2点（`shell/tokenBudget.ts`／`shell/tokenBudget.test.ts`）＋新規1点（`modes/chained/stage2Window.test.tsx`）、webui 改修6点（`api/types.ts`／`modes/chained/chainUtils.ts`／`modes/chained/useChainForm.ts`／`modes/chained/ChainedScreen.tsx`／`modes/chained/generateReasonMessages.ts`／`i18n/strings.ts`、テスト1点＝`modes/chained/chainUtils.test.ts`）、native 1点（`native/src/timeline_math.h`・**コメントのみ**）、文書2点（本節と[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-14／§3-57 の状態更新）。`i18n/strings.ts`は`chained.*`配下への追加のみで、並行ワークストリームの変更とは衝突しない。
- **検証**: `npm run typecheck`（`tsc -b`）**0エラー**、`npm run test`（vitest）**1798 PASS／0 FAIL／10 skip**（着手前1761 PASS／10 skip・**+37件・退行ゼロ**）、`npm run lint`は既存25件の警告のみで**新規指摘ゼロ**（変更・新規のファイルには1件も出ていない）。バックエンド側は`pytest` **1010 PASS／14 skip／0 FAIL**（着手前955件collected・全PASS）。
  - 着手前のvitestを2回走らせたところ、1回目だけ`App`系のテストが2件落ちた（`findByText(/generating/i)`のタイムアウト）。2回目とその後の全実行では再現しないので**環境要因のちらつき**と判断した。改修後の実行では0件である。
- **状態**: 実装・機械検証は完了。**実機ゲートは未実施**（GPUを使うため。手順書とスクリプトは`Nz-Videomni/outputs/stage2_window_optin/`に用意した）。**デプロイもコミットもしていない。**
- **【2026-08-09 追記・その後】** 実機ゲートG1〜G3は同日中に**全項目合格**し（バックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §53.9）、オーナーの目視・実機確認も通って**テーマは完結した**（台帳[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-68〔解像度警告〕・§3-69〔クリップ長の選択〕）。ただし**上に書いた画面まわりの2点は、その後の目視ラウンドで変わっている**——ラベルは「仕上げの前進幅」ではなく「**Stage-2（アップスケール工程）のクリップ長**」になり、秒数もフレームレートからの導出をやめて固定の概数（潜在22フレーム＝約7.0秒／潜在19フレーム＝約6.0秒）になった。「潜在フレーム」という言い方も、隠さずに【解説】を添えて出す方針へ変わっている。経緯は§67と`VERIFICATION_LOG.md` §53.11。

---

## 66. Outpainting（動画のキャンバス拡張）の本体実装 — 公式ワークフローの再現（W2）（2026-08-09）

**やったこと**: §63でEditタブに枠だけ作り、§64で右クリックの導線を通してあったOutpaintingを、実際に動く機能として実装した。バックエンドとフロントエンドの両方にまたがる。設計方針の正本は[`OUTPAINTING_DESIGN_NOTES.md`](OUTPAINTING_DESIGN_NOTES.md)（実装で判明した訂正を§5-Aとして追記した）、実機ゲートの手順書は`Nz-Videomni/uploads/_outpaint_verify/RUNBOOK_W2.md`。

- **設計時の想定が1つ、根本から覆った**。設計書§3-4は「公式はマスクを注意強度に掛けている。だから自前のマスク条件付けクラスを書いて条件付け経路に差し込む」としていたが、**公式ワークフローの配線を精読したら、マスク条件付けは使われていなかった**。該当ノードの`attention_mask`入力は未接続で、`attention_strength`は既定の1.0のままである。設計時に「マスク設定」と読んでいた`'disabled'`という値は、実は**cropの設定**だった。したがって公式におけるマスクの役割は「緑を塗る範囲を決める」「ブレンドの重みになる」の2つだけで、**条件付けクラスは1行も要らなかった**。難易度の見積り（S〜M）よりさらに軽く済んだことになる。
- **代わりに本体は「2段の間に割り込む」ことだった**。既存の単発生成は、ライブラリの`DistilledPipeline.__call__`が1段目と2段目を続けて回してしまうので、その間に手を入れる隙間が無い。公式のやり方は「1段目を復号→緑キャンバスとブレンド→ピクセルを2倍→符号化し直す→2段目」なので、**2段を自前で制御する`run_outpaint()`を新設**した。前例はチェーン生成の`run_chain()`で、同じように既存パイプラインの部品と高速化機構（NAG/SageAttention/block swap/GGUF/常駐/PrunaVAED/VAEタイル）をそのまま引き継ぐ書き方にしてある。
- **緑参照が1段目にしか効かないことを、公式のノード実装を取り寄せて確定させた**。`LTXVCropGuides`はComfyUI本体の組み込みノードで、実装は「潜在の末尾からガイド用フレームを物理的に削る」ものだった。当方には等価な機構がすでにあり（参照条件は解像度が半分のときしか作られない／余分なトークンは`clear_conditioning`が末尾から切る）、**専用の引数を足す必要は無かった**。「たぶん大丈夫」で済ませずに実装を取りに行ったのは、ここを間違えると2段目に緑が持ち越されて全部が台無しになるからである。
- **緑は可逆RGBで書かないと保てない、というのをテストが捕まえた**。ffmpegの`pad`フィルタは色をRGBで受け取るが、**その時点のフィルタ内の形式で塗る**。入力が普通のyuv420pだと緑(102,255,0)が一度YUVに変換され、可逆RGBで書き出す段でまた戻され、**(101,253,0)になる**。公式LoRAは「その色ちょうど」を描き替えるよう学習されているので、これは無視できない。`pad`の前に`format=rgb24`を挟んで解決した。最初のテスト実行で落ちて分かったもので、**書いていなければ実機で「なんとなく緑が残る」として現れ、原因にたどり着くのは相当あとになっていた**。
- **キャンバスの寸法は「リクエストの縦横」を正本にした**。設計書§4-2の書きぶりは「パディング量からキャンバス寸法が決まる」だったが、実装では逆にした。元動画の実測値から逆算すると、リクエストの値と実測値という**情報源が2つ**できてしまい、ずれたときに参照動画の読み込み処理が黙って拡大縮小と中央切り抜きを行う。「元の画素はそのまま残る」というOutpaintingの唯一の約束が、誰にも気づかれずに壊れる経路だった。いまは元動画の寸法をffprobeで実測して照合し、合わなければ422で弾く。画面の見た目（パディングのスライダー4本）は設計どおりである。
- **ブレンドが効く幅が、設計時の認識より1桁大きかった**。ブレンドのマスク膨張は「マスクを長辺64画素まで縮めてから半径rで膨らませて元に戻す」処理なので、実効的な幅は`r × キャンバス長辺 ÷ 64`になる。キャンバス長辺1920・公式既定のr=5なら**約150画素**である。しかも膨張は生成側を広げる向きなので、**保持領域の外周150画素は元映像と生成映像の混合**になる。設計書§3-2に「ピクセル完全一致ではない」とは書いてあったが、この幅は想定していなかった。対応として(1)保持領域が縦横とも256画素以上であることをAPIとエンジンの両方で検証し、(2)パネルに「境目から内側へおよそ150画素は混ぜ合わされる」という説明文を出すことにした。**黙って出すと、あとで必ず「外周がぼやける」という報告になる**性質の挙動である。
- **2段目のノイズ量だけは公式に合わせた**。ライブラリ既定は0.909375から始まるが、公式は1段低い0.725から始める。2段目の出発点は**ブレンド済みの絵を符号化し直したもの**なので、ここが高いとせっかく直した継ぎ目を作り直してしまう。「公式準拠」を掲げておきながら、いちばん効く数値だけ黙って自製に倒すのは筋が通らない、という指摘が敵対的レビューから出て採用した。
- **ラプラシアンピラミッドブレンドの移植は、本物のkorniaと照合して確定させた**。公式実装はkorniaに依存しているが当プロジェクトには入っていないので純torchへ書き写した。ここで**恒等性テスト（マスク全1なら元の絵に戻る）は移植の誤りを検出できない**——ピラミッドを作るときと戻すときで同じ関数を使うので、係数を間違えていても誤差が打ち消し合ってしまう。そこで一時的な作業領域にだけ本物のkornia 0.8.3を展開し、公式コードをそのまま写したスクリプトで3ケースの正解データを作って固定した。**誤差0.0（完全一致）**である。
- **メインメモリの使い方を先に潰した**。1920×1152の241フレームを32bit小数で持つと1本6.0GBになり、「生成結果」「緑キャンバス」「出力」の3本で18GBになる。8bitのまま持ってチャンクごとに変換する設計にして約4.5GBに収めた。これも敵対的レビューの指摘（「数GB」という見積りが1桁甘い）から拾ったものである。
- **敵対的レビューで拾って直したもの**: 上記の2段目シグマ・メモリ見積り・キャンバス寸法の二重情報源・ブレンド幅150画素に加えて、(1)元動画が指定尺より短いと2段目の内部検証で落ちる（公式は「短いほうに合わせて切る」で吸収しているが当方の構造では吸収できない）ので、APIで先に422にしたうえでキャンバス生成側でもフレーム数を保証する二重の手当てを入れた、(2)ブレンド用マスクを`expand`しても`F.pad`が実体化するのでメモリは減らない（1枚だけ処理する設計に変更）、(3)`crop_output`との併用が「拡張した部分を真っ先に切り落とす」ので排他にした、(4)音声を`-c:a copy`でキャンバスに載せるとコーデックによってはmp4に入らない（キャンバスは映像のみにして、音声は元のアップロードから読む）、(5)参照が無いときの`None`アンパック、(6)ブレンド入力のdtype指定漏れ、(7)既存の`probe_resolution`／`frame_count`を見落として新設しようとしていた点。
- **敵対的レビューの指摘のうち採らなかったもの**: (1)「ブレンドの相手に緑キャンバスを使うと低周波で緑がにじむので、緑でない画を用意すべき」——**公式が緑を使っており**、代案は拡張帯で両者が一致してブレンドが恒等写像に退化する。レビュー自身も追試して「代案は間違い」と結論を翻したので、公式どおりにした。(2)「`engine/outpaint/`というパッケージは大げさ、`engine/pipeline/`に平置きすべき」——好みの問題であり、承認済み計画のファイル配置を美観のために動かす理由が無い（W3が同じ`engine/pipeline/`で作業している事情もある）。(3)「full解像度のブレンドを完全にストリーミング化すべき」——既存のチェーン生成も出力テンソルを一度作ってから書き出しており、8bit保持で18GB→4.5GBに落ちれば当面の目的は足りる。抽象を増やさない方を採った。
- **モックの絵にも幾何を反映させた**。モックバックエンドは`width`/`height`をそのまま使うので拡張後サイズで出るのだが、それだけだと**普通の生成と見分けがつかない**。元動画があったはずの矩形を緑の枠線で描くようにした（幾何がずれていても気づけないのは困るため）。
- **重みの再ホストは成功した**。公式In-Outpainting IC-LoRA（1.31GB）を`Rootport/Nz-LTX23-weights`の`ltx-2.3-ic-lora-in-outpainting/`へ上げ、`install_ltx.ps1`にも取得と検証の行を足した（既存のdeblurと同じ、独立した検査ディレクトリを持つ形。理由は同スクリプトのコメントに詳しい）。
- **変更ファイル（フロントエンド）**: webui 新規6点（`modes/edit/outpaintGeometry.ts`／`outpaintGeometry.test.ts`／`OutpaintPreview.tsx`／`useOutpaintForm.ts`／`editReasonMessages.ts`／`OutpaintingPanel.test.tsx`）、webui 改修6点（`modes/edit/OutpaintingPanel.tsx`／`EditScreen.tsx`（1行）／`EditScreen.css`／`EditScreen.test.tsx`／`i18n/strings.ts`（`edit.outpainting.*`のみ）／`api/types.ts`）＋`App.editRoute.test.tsx`（Createが隠れていることの判定を、Outpaintingが生成ボタンを持ったことで成り立たなくなったため差し替え）、文書2点（本節と[`OUTPAINTING_DESIGN_NOTES.md`](OUTPAINTING_DESIGN_NOTES.md) §5-A）。
- **変更ファイル（バックエンド）**: 新規5点（`engine/outpaint/__init__.py`／`canvas.py`／`pyramid_blend.py`、`engine/pipeline/outpaint_pipeline.py`、`uploads/_outpaint_verify/RUNBOOK_W2.md`）＋ゲート道具2点（`run_w2_gate.ps1`／`analyze_w2.py`）＋テスト3点（`tests/test_outpaint_canvas.py`／`test_outpaint_pyramid_blend.py`／`test_outpaint_api.py`）＋固定データ1点（`tests/fixtures/outpaint_pyramid_golden.npz`）、改修9点（`api/models.py`／`api/generate.py`／`api/errors.py`／`services/video_io.py`／`services/pipeline_manager.py`／`services/ltx_runner.py`／`engine/worker.py`／`engine/pipeline/fast_video_pipeline.py`／`scripts/install_ltx.ps1`）＋設定2点（`config.yaml`／`config.yaml.example`）。**`engine/pipeline/chain_pipeline.py`と`chain_math.py`は読み取り専用で、1文字も変えていない**（並行するW3の占有領域のため。必要な関数はimportして呼んでいる）。
- **検証**: バックエンド`pytest`（appのvenv）**1040 PASS／15 skip／0 FAIL**、engineのvenv **163 PASS／0 FAIL**（着手前123件）。フロントエンド`npm run typecheck` **0エラー**、`npm run test` **1836 PASS／10 skip／0 FAIL**（着手前1798 PASS・**+38件・退行ゼロ**）、`npm run lint`は既存の警告のみで**変更・新規ファイルに新規指摘ゼロ**。**モック通しゲート18項目すべてPASS**（隔離した設定で実際のuvicornを起動しHTTP経由で確認。リポジトリの`outputs/`・`uploads/`への書き込みはゼロであることも確認済み）。
- **状態**: 実装・機械検証は完了。**実機ゲート（GPU）は未実施**——手順書とスクリプトは`Nz-Videomni/uploads/_outpaint_verify/`に用意した（機械判定4項目＋目視1枠）。**デプロイもコミットもしていない。**
- **【2026-08-09 追記・その後】** 実機ゲートW2は同日中に**機械判定4項目すべてPASS**（`uploads/_outpaint_verify/w2_gate/summary.json`。正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §54.4）、**オーナーの実機生成も合格**した。画面まわりはこのあと目視3ラウンドで仕上げており、その差分は§67にある。台帳では[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-70としてクローズした。

---

## 67. Outpaintingパネルの仕上げ — オーナー目視3ラウンドで確定した見た目・操作・文言（2026-08-09）

§66でOutpaintingが動くようになったあと、オーナーの目視を3ラウンド回して画面を仕上げた。**機能そのもの（生成の中身）は変えていない**——この節は「どう見せるか」だけの記録である。§65（Stage-2のクリップ長）で決めた文言も、このラウンドの中で最終形に変わったので、あわせてここに書く。**最終的な文言は`webui/src/i18n/strings.ts`の現物が正本**であり、本節はそこへ至った理由を残すためのものである。

- **プロンプト欄をアプリ全体で1つに統一した**。着手時のパネルには自前のプロンプト入力欄を置いていたが、**`PromptBar`はタブ共通の部品**（Single／Chainedと同じものがEditでも見えている）だと分かったので、パネル内の欄は撤去した。同じ文章を入れる場所が画面に2つあると、どちらが効くのか説明できない。フォームの側は`prompt`を**外から受け取るだけ**にしてある（`useOutpaintForm`が自分でプロンプトの状態を持たない）。
- **ジョブ一覧をEditタブにも常駐させ、2カラムにした**。Create／Chainは「左＝入力フォーム／右＝生成ボタンとジョブ台帳」という2カラムだが、Editだけ1カラムで、生成しても進み具合が同じ画面で見えなかった。既存の`single-layout`をそのまま使い、右の列に`GenerateButtonBar`とジョブ台帳（`jobs/JobLedger`。3画面が共有する同じ部品・同じポーリング）を置いた。**台帳はサブタブの切り替えの外側**に出してあるので、RetakeサブタブでもOutpaintingサブタブでも見えている——生成の進み具合はサブタブと関係がないためである。逆に生成ボタンのほうはOutpaintingのときだけ出す（Retake側にはまだ生成するものが無い）。
- **「マスクブラー」スライダーを新設した**（このラウンド最大の追加）。§66の時点では、ブレンドの膨張半径はAPIにはあるが画面には出していなかった。しかし**緑が残ったときに利用者が自分で対処する手段が無い**のは不親切なので、露出することにした。設計は次のとおりである。
  - **内部は膨張の段数（0〜15。サーバー側の範囲そのまま）**で持ち、**画面に出る数字はすべて実寸ピクセル**（`段数 × キャンバス長辺 ÷ 64`）に直す。段数はキャンバスに対する相対量なので、**ピクセルを状態として保存すると嘘になる**（キャンバスが変われば同じ段数でも実寸が変わる）。だから状態は段数のまま持ち、表示のたびに導出している。
  - **2段目は公式サンプルと同じ5:2の比で1段目に追随させる**。2つを別々に触れるようにすると、比が崩れた組み合わせを作れてしまう（公式が「最重要パラメータ」と呼んでいる値である）。
  - **既定（1段目5）のままなら、リクエストにキーを載せない**。この2つのフィールドを知らない古いサーバーでも既定のリクエストが通る。
  - **プレビューの破線・凡例・説明文は、すべてこの1つの実寸値を見る**。「境界から内側へ約○○pxは元動画と拡張部分が混ぜ合わされる」という説明は、以前は1920固定の「150px」と書いていたが、**そのキャンバスでの実際の値**を出すようにした。
  - **上げすぎたときは警告を出すが、生成は止めない**（向かい合う2本の帯が出会うと、元動画に混ざっていない芯が残らなくなる）。
  - **「緑が残ったらこの値を上げる」という助言は、説明文の中の括弧書きへ畳み込んだ**（オーナー指摘）。以前は独立した2行目（`blurGreenNote`）にしていたため、同じスライダーの話が2段落に分かれて見えていた。キーごと廃止している。
- **素材カード・初期配置・尺まわりを整えた**。素材は他画面と同じカード表現（サムネイル枠＋差し替え／クリア＋「1920×1080ピクセル／7.0秒」の読み上げ）にし、**初期配置は「縦（上詰め・中央・下詰め）」「横（左詰め・中央・右詰め）」のラジオ**に分けた。尺のスライダーは**上限を元動画の長さから毎描画導出**する（`durationSec × fps`を8n+1へ切り捨て、最大481フレーム）。上限は**保存しない**——保存すると素材を外したときにクランプだけが残る（設計方針書§4-4と同じ教訓）。あわせて**快適上限の目盛り**をスライダーに1点だけ描いた（Create画面と同じ`<datalist>`の作法）。**ただし引くのは元動画の寸法ではなく拡張後のキャンバス**である——実際に生成されるのはそちらなので、元動画で引くと必ず甘い値になる。
- **警告は2種類あり、意味が違うことをコメントに明記した**。尺スライダーの目盛りと警告は「この解像度での**尺**の目安」、パネル末尾の警告は「解像度と尺を掛け合わせた**総処理量**の目安（設計方針書§4-5）」で、片方だけが出る組み合わせがある。文言もSingle画面とは別キーにしてある（Single側の文言を変えない方針のため）。
- **Stage-2のクリップ長（§65）の文言もここで確定した**。「仕上げの前進幅」→「**Stage-2（アップスケール工程）のクリップ長**」、選択肢は「潜在22フレーム（約7.0秒）」「潜在19フレーム（約6.0秒／VRAM溢れ軽減）」。**「窓」「タイル」は出さないが「潜在フレーム」は出す**——隠して遠回りな言い方をするより、【解説】を添えて正しく呼ぶほうが分かる、というオーナー判断である。ヒント文にはStage-1→Stage-2→VAEデコードの3工程と、「長すぎる仮動画を一定の長さで区切って拡大し、あとで繋ぎ直す。その長さの指定である」という説明を入れた。秒数はフレームレートからの導出をやめ、固定の概数にした（示すべきは窓の**長さ**であって前進量ではないため。§65の該当箇所に追記済み）。
- **右クリックメニューの並びを直した**。§64で足した2項目が末尾に付いていたため、動画オブジェクト向けの項目が離れて並んでいた。**動画系の5項目（続き生成・参照生成・音声から生成・描き足す・撮り直す）が連続する**よう並べ替え、日本語の文言も「🎬 Video: この動画に描き足す (Outpainting)」「🎬 Video: 選択範囲を撮り直す (Retake)」へ整えた。絵文字が**機能の絵ではなく右クリックすべき対象の種別**を指すという既存の読み方（§64）は維持している。
- **状態**: オーナーの目視・実機とも**全項目合格**。デプロイ済み。**コミットもオーナーが実施済み**（フロントエンド`6738fe1`／バックエンド`d205f6c`。§65・§66・本節の実装がまとめて入っている）。台帳の該当項目は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-68〜§3-70としてクローズした。

---

## 68. Retake（選択範囲の撮り直し）の本体実装 — 右クリックからパネル・可動窓・予約の打ち直しまで（2026-08-10）

**やったこと**: §63でEditタブに枠だけ作り、§64で右クリックの導線（「🎬 Video: 選択範囲を撮り直す (Retake)」）を通してあったRetakeを、実際に動く機能として実装した。**バックエンド側は2026-08-09に実装が終わっている**（正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §55.8〜§55.10）ので、本節はフロントエンドの記録である。台帳は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-73（記録当時は`PENDING_TASKS.md` §1-17。2026-08-10のテーマ完結でクローズ移設）。

- **窓の判定を、Reactにもbridgeにもi18nにも依存しない純関数へ出した**（`timeline/retakeWindow.ts`）。この判定は「右クリックのガード」「Retakeパネルの表示」「送信直前の再検証」という**3か所から別々に呼ばれる**ので、三者が一字一句同じ答えを出さないと、画面に出ている区間と実際に撮り直される区間がずれる。§56のV2Vリボン範囲トリム（`timeline/sourceTrim.ts`）と同じ規律である。窓の長さの下限73・上限169（上限は§69でプリセット連動へ変更。standard 169／high_resolution 145）と8n+1の格子も**この1ファイルだけ**が持ち、`RangeBand`は定数を直接読まずpropsの既定値として受け取る形にした（将来`AppConfig.limits`でサーバーから上書きするときに直す場所を1か所に閉じるため）。
- **「選んだ範囲」を「素材ファイル自身の秒」へ写すところが、この実装でいちばん危ない箇所だった**。ここを間違えると**ユーザーが指したのとは別の場所を無言で撮り直す**——エラーも出ないので、出来上がった動画を見るまで気づかない種類の故障になる。潰したものを列挙する。
  - **リボンの尻尾が素材を再生しているとは限らない**。AviUtl2では頭を右へ引っぱるとリボンの長さは変わらず、尾が静止フレームになる（§56で確定した所見）。だから交差クランプの相手はリボン全体ではなく**実際に素材が流れている区間**にした。静止フレームの上を選ばれたら、そこに対応する素材秒はもう存在しない。
  - **`decideSourceTrim`の早期returnが、後続3つの検査を追い越していた**。同関数は「リボンが素材の全長を占めている」を再生速度・ループ・中間点より**先に**判定して即returnする。Retakeにとって全長リボンは障害ではない（切り詰める必要がないだけ）ので通したいのだが、素通しにすると「0〜0.5秒をループ再生して2秒のリボンを埋めている全長オブジェクト」がそのまま通り、まさに上の故障が別の口から入る。同じしきい値・同じ順序で3つをやり直す関数（`fullLengthRibbonSkipReason`）を置いて塞いだ。**敵対的レビューで拾った指摘である。**
  - **再生開始が0でない全長リボンは弾く**。長さが全長のままでも再生窓が2秒目から始まっている状態が実在するので、起点0と決めつけられない。
- **選択範囲の開閉区間の規約が未文書だったので、仮説を1行に隔離した**。`timeline.getSelection`が返す`rangeStart`/`rangeEnd`について、**閉区間（`rangeEnd`を含む）か半開区間かを決めている文書がTS側にもnative側にも無い**（nativeの`bridge.cpp`はSDKの`select_range_start`/`select_range_end`を無変換で流すだけ）。同じスナップショット内の`frameStart`/`frameEnd`が閉区間であることは確立済みなので**閉区間と仮定**したが、根拠は慣行だけである。そこで**規約だけを依存ゼロの葉ファイル`timeline/selectionRange.ts`へ出し、直す場所を`rangeEnd + 1`の1行に閉じた**。葉に出したのは循環importの回避も兼ねている（`retakeWindow.ts`→`sourceTrim.ts`→`prefillSeed.ts`という実行時importの環があり、規約を要る`prefillSeed`から`retakeWindow`を呼べない。かといって`+ 1`を両方に書き写すと「1か所で直せる」という要件そのものが壊れる）。`retakeWindow.test.ts`の"closed-interval hypothesis"テストが仮説をpinしており、**実機で確定したら1行と期待値を直すだけ**で済む。確定のための操作は[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-2のチェックリストに入れた。
  - **【2026-09-04追記】この§2-2は節ごと削除済みである**（項目が空になったため運用規則どおり見出しごと削除された。バックエンド[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md)を探しても見つからない）。なお本項の`rangeStart`/`rangeEnd`は下の**状態**行のとおり2026-08-10に閉区間で実測確定しており、**【2026-09-04 追記2】その`frameStart`/`frameEnd`の端点解釈も同日中に決着した——「包含」である**（`rangeStart`/`rangeEnd`と同じ規約になった。経緯と一次証拠は本書**§107.11**）。以下の「未決」という記述は当時のものとして残してある。現在も未決なのは**別物の`frameStart`/`frameEnd`の端点解釈**のほうである——その正本は[`BRIDGE_CONTRACT.md`](BRIDGE_CONTRACT.md) §4.13（`timeline.getSelection`）の`selected[]`の項である。
- **仮オブジェクトの尺に3つ目の家系ができた**。従来のDURATIONは「解像度から決める（`comfortCeiling`）」か「素材の長さから決める（`materialClampedToCeiling`）」の2択だったが、Retakeはどちらでもない——**ユーザーが選んだフレーム範囲の長さ**が尺である。`prefillSeed.ts`に`selectedRangeLength`という3つ目の方針を足し、その入力（範囲のフレーム数）は`materialDurationForIntent`に相乗りさせず**別の関数**にした。片方は素材の実時間、もう片方は選択のフレーム数で、単位も問いも違うためである。数え方は上記の`selectionRange.ts`越しなので、閉区間仮説が覆っても自動で追随する。
- **予約は「打ち直す」方式にした**（本実装の設計上の要点）。右クリックの時点で仮オブジェクトの席は取られるが、その時の尺は**選択範囲そのままの長さ**である。ところが実際に生成される窓は8n+1へ丸められ[73, 169]へクランプされ（同上）、素材の終端に当たれば開始位置ごと手前へずれる——**つまり右クリック時の席と、確定した窓は一致しない**。そこでGenerateを押した瞬間に、確定した**窓の開始位置と窓長**で`reservePlacement`をもう一度呼んで席を打ち直し、そのうえで送信する。受理されたら`bindToJob`でジョブへ渡し、422などの同期的な失敗では`rollback`で席を片付ける。**ここを省いて右クリック時の席のまま流すと、タイムラインに出る仮の帯と、返ってくる動画の長さ・位置が食い違う。**
- **区間バー（`RangeBand`）はv1で4層**。素材全長の下敷き／窓の塗り／窓の**内側**の糊代の縞／ハンドル2つ、である。**糊代の縞は非対話**（`pointer-events: none`＋`aria-hidden`）にした——あれは「前後となめらかにつなぐために元の映像がそのまま残る部分」であって、ユーザーが個別に動かすものではないからである。**8n+1へのスナップはバーの中では一切やらない**。バーがやるのは下限・上限へのクランプだけ（ハンドルが端で止まる触覚はこれが作る）で、格子への吸着は親から制御された値として戻ってくる——Reactのcontrolled inputと同じ形にして、格子の定義が2か所に散るのを防いだ。ポインタ作法は`modes/single/KeyframeTimeline.tsx`踏襲（`setPointerCapture`＋`pointercancel`と`lostpointercapture`の両方を購読）、矢印キーで1フレーム・Shiftで8フレーム。「右クリックした瞬間の範囲を薄く残す5層目」はv1.1送りにした（後述のずれ注意文が先に同じ役目を果たすため）。
- **のりしろの値はサーバーへ送らない**。`head_px`/`tail_px`はAPIにはあるが、リクエストには載せていない。実測で較正された値（頭25／尾24。バックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §55.5）をサーバー側の既定に自動追随させるためで、**画面から動かせないつまみを送ると、送った値が正であるかのように見えてしまう**。フロント側の25/24はバーの縞と説明文1行を描くためだけの写しである。窓長の正は`clips[0].num_frames`1本。
- **パネルにわざと置かなかったもの**: DURATION（尺）欄——長さは区間バーが決めるので、数字を別に置くと正が2つになる。素材の差し替え——右クリックしたオブジェクトに対して測った区間なので、素材だけ入れ替えると意味が変わる。プロンプト欄——§67で全画面共通に一本化した`PromptBar`を読む。のりしろのつまみ——上記のとおり。
- **生成時の再検証は注意文にとどめ、ブロックしない**（着手時のオーナー確定事項どおり）。パネルを開いたあとにタイムラインの選択が変わっていたら「撮り直されるのは、ここに表示されている範囲です」と出すだけで、生成ボタンは止めない。
- **LoRAはv1では対応しない**。`loras`は送らないが、共有プロンプト欄には`<lora:名:強度>`が書かれうるので、**Create/Chainと同じパーサでタグを文字列としても取り除いてから**送る。素通しにするとタグの字面がそのまま指示文の一部として効いてしまう。解禁の検討は[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-62へ起票した。 **→ スタイルLoRAは2026-09-02に解禁した（§103）。台帳の記録はバックエンド[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-62-02**（この行の`PENDING_TASKS.md`リンクは当時から相対パスが実体を指していない。バックエンドの文書は`../../../Docs/`配下である）。
- **プロンプト空欄を許している**のは意図である。同じ内容のまま「もう一度撮る」のは撮り直しとして正当な使い方だからである。
- **音声ハンドルのサイドカーは探していない**。バックエンドが窓まるごとを返すので、重ね置きに必要な重複素材は出力mp4の中にある（2026-08-09の撤回。同§55.8）。
- **変更ファイル（webui）**: 新規7点（`timeline/retakeWindow.ts`／`retakeWindow.test.ts`／`timeline/selectionRange.ts`、`modes/edit/RangeBand.tsx`／`RangeBand.css`／`RangeBand.test.tsx`、`modes/edit/RetakePanel.tsx`）＋`modes/edit/useRetakeForm.ts`／`useRetakeForm.test.tsx`／`EditScreen.retake.test.tsx`、改修は`modes/edit/EditScreen.tsx`／`EditScreen.css`／`editReasonMessages.ts`／`timeline/prefillSeed.ts`（＋テスト）／`timeline/menuSelection.ts`／`shell/AppShell.tsx`／`i18n/strings.ts`（`edit.retake.*`のみ）／`api/types.ts`。
- **検証**: フロントエンド`npm run test` **1985 PASS**、e2e緑、`npm run typecheck`（`tsc -b`。`npx tsc --noEmit`は偽合格になるので使わない）0エラー。バックエンド側の機械検証と実GPUゲートはバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §55.9〜§55.13。
- **状態**: **2026-08-10 オーナー目視合格。** 実装・機械検証は完了し、当時の確認項目6点（うち1件は上記の開閉区間の規約を確定させる操作）はすべて合格した。**開閉区間は閉区間で実測確定した**——上記`selectionRange.ts`の`rangeEnd + 1`は仮説ではなく正式な仕様になった。**最小窓は73フレーム維持（選択肢(b)・注意文で伝える方式）で2026-08-10に確定し、テーマは完結した**（コード変更なし。記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-73）。デプロイ済み（実機＋backend配布コピーの2箇所。deploy.ps1）。**コミットは未実施。**→ 2026-08-11、オーナーがコミット・プッシュ済み。

## 69. Retake目視フィードバック改修バッチ — busyガード・素材カード・読み出し行・幅高さスライダー・Stage-2ドロップダウン移植・説明文2件（2026-08-10）

**やったこと**: §68で実装したRetakeを、オーナーの実機目視フィードバック（台帳[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-2の6項目に対する回答）にもとづき7件改修した。バックエンド側の「Retake×潜在19フレーム併用422拒否」撤去とGPU実測ゲート2本による裏取りはバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §55.15が正本。台帳は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-73（記録当時は`PENDING_TASKS.md` §1-17。2026-08-10のテーマ完結でクローズ移設）。

- **① busyガード**: サーバーがジョブ実行中のとき、Editタブ（Retake）のGenerateボタンと、タイムラインの右クリック「選択範囲を撮り直す」を不可にした。右クリック側は`retakeRange`起動に限定したガードで、他の右クリック生成起点（既存のガード対象）は変更していない。
- **② 素材カード**: 表示を「ファイル名 m秒 nフレーム（パネルFPS換算）＋解像度」へ改めた。m秒は実際に再生されている区間長（頭出しトリムを除いた長さ）で、値が取得できない（0の）部分は省いて表示する。
- **③ 読み出し行**: 主表示を「タイムラインの{start}〜{end}フレーム目を撮り直します」（1始まり表示）へ変更した。窓のフレーム数と秒は括弧書きの副表示へ回し、ドラッグ中もリアルタイムに追随する。
- **④ 幅・高さスライダーの編集可能化**: 従来は素材実寸から64px倍数へ自動導出するだけで、値は非表示だった。手で編集できるようにしたが、初期値は従来どおりの自動導出値のままとし、ユーザーが触るまでは変わらない。エンジン側のリサイズ＋中央クロップはどの幅・高さでも起きるため、「クロップ先をユーザーが選べるようになった」というのが正確な説明である。
- **⑤ Stage-2のクリップ長ドロップダウンをChainedからRetakeへ移植**: 文言はChainedと共通のi18nキーを直読みし、二重管理を避けた。「潜在19フレーム」選択時はRetakeの最大窓が145フレームへ縮む（窓19の1タイル上限）。これに対応して、バックエンドの「Retake×潜在19」一律422拒否を撤去し、`retake_max_window_px(v_tile)`による動的上限検証（standard=169・high_resolution=145）へ差し替えてもらった。
- **⑥ 説明文2件の改訂**: summary文を「動画の選択された区間を再生成する機能です。選択されていない部分はそのまま残ります。（最小73f。短すぎると元動画からほとんど変化しません）」へ、のりしろ文を「のりしろは前25フレーム・後ろ24フレームです。」へ改めた。
- **変更ファイル（webui、未コミット）**: `App.editRoute.test.tsx`／`i18n/strings.ts`／`jobs/useJobsPoll.ts`／`modes/edit/EditScreen.retake.test.tsx`／`modes/edit/EditScreen.test.tsx`／`modes/edit/EditScreen.tsx`／`modes/edit/OutpaintingPanel.test.tsx`／`modes/edit/RangeBand.css`／`modes/edit/RangeBand.tsx`／`modes/edit/RetakePanel.tsx`／`modes/edit/editReasonMessages.ts`／`modes/edit/useRetakeForm.test.tsx`／`modes/edit/useRetakeForm.ts`／`modes/single/CommonGenerationFields.tsx`／`shell/AppShell.tsx`／`timeline/retakeWindow.ts`。バックエンド側は`api/models.py`／`config.py`／`tests/test_retake_api.py`。
- **検証**: フロントエンド`npm run typecheck`0エラー・vitest **2001件PASS**（+16件・退行ゼロ。§68時点1985→2001）・lint新規ゼロ。バックエンドpytest **1115件PASS**（+7件・退行ゼロ）。422検証マトリクス実測7ケースとGPU実測ゲート2本（窓145f×standard／窓145f×high_resolution、出力mp4のSHA256が完全一致）の詳細はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §55.15。
- **状態**: **2026-08-10 オーナー目視合格**（再確認6項目〔busyガード・素材カード・読み出し行・幅高さスライダー・Stage-2ドロップダウン・説明文2件〕すべて）。実装・機械検証は完了。デプロイ済み（実機＋backend配布コピーの2箇所。deploy.ps1）。**コミットは未実施。**→ 2026-08-11、オーナーがコミット・プッシュ済み。

---

## 70. 長尺A2V（クリップ別の参照音声）— チェーン全体へ音声1本を添付する機能の実装（2026-08-10）

**やったこと**: A2V（音声から動画を生成する機能）は、これまでクリップ連結では「クリップ1本のとき」しか使えなかった。この制限を外し、**連結したタイムライン全体に音声を1本添付して、各クリップの担当区間をサーバーが自動で割り当てる**形の長尺A2Vを実装した。台帳は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-74（記録当時は`PENDING_TASKS.md` §1-16。2026-08-10のテーマ完結でクローズ移設）。バックエンド側の実装と検証の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §56で、本節はフロントエンドの記録である（バックエンドの要点だけ、フロントエンドの設計に効いた範囲で最初に触れる）。

- **バックエンドはガード1行の削除では済まなかった**。塞いでいたのは`api/models.py`の「A2Vはクリップ1本のみ」という検証1本だが、エンジン側（`engine/pipeline/chain_pipeline.py`の`_denoise_av_with_carry`）のコメント自身が「A2Vは1本しか来ないので、音声用の凍結マスク0.0が映像に触ることはない」と前提を書いていた。**この前提のまま複数クリップを通すと、映像のつなぎ目までハードに凍結され、つなぎ目ブレンド強度（`overlap_strength`）が無言で効かなくなる**。実装前にこれを検出し、既にある`tail_mask_value`と同型の`audio_mask_value`引数を足して、**映像のつなぎ目は通常のcarryブレンド、音声の窓だけを0.0で凍結**する形に直してもらった（引数を省けば従来経路とビット同一）。あわせて`chain_math.py`の`audio_latents_required`が`compute_chain_layout`を丸ごと呼んでいた作りをやめ、必要な`a_total`だけを直接計算する形にした——stage-2の窓の検算に巻き込まれて事前検査が500になる経路（**現行の単クリップA2Vにも既にあった穴**）が同時に塞がっている。開発用のGradio GUIは1クリップのまま据え置き（文言とコメントのみ更新）で、長尺A2VはAviUtl2フロントエンド専用の機能である。
- **サーバーの幾何計算をTypeScriptへ写す作業が、この実装の芯だった**。フロントエンドは「音声の長さに合わせてクリップ構成を自動で組む」ので、**組んだ結果をサーバーが422で突き返すなら、そのボタンは壊れている**。そこで`chainUtils.ts`にミラー群を足した。
  - **`pyRound()`（Pythonの`round`のミラー）**。Pythonの`round`は**銀行丸め**（ちょうど.5のとき偶数側へ丸める）で、JavaScriptの`Math.round`（.5は常に切り上げ）と食い違う。しかもこれは理屈だけの話ではなく、音声潜在フレーム数の計算がちょうど.5に着地する入力の族が実在する（たとえば50fpsでは8n+1のクリップ長すべてが該当する。`49/50*25 = 24.5`）。**潜在1フレームの差が、受理される要求と422の差になる**ため、バックエンドの`round`を写している関数はすべてこれを通すよう既存分も含めて差し替えた。
  - **`chainLayoutError()`（`compute_chain_layout`が投げる例外のミラー）**。stage-2の音声タイルの再組立検算などを、Python側と同じ順序で判定して最初に落ちる理由を返す。**非標準のフレームレートでは、この検算が「ごく普通のクリップ構成」を弾く**——たとえば`[257, 257]`は24fpsなら通るのに29.97fpsでは通らない。これはA2V固有ではなく素のチェーンにもある既存の制約で、今回は音声を添付しているときだけこのミラーを効かせている（素のチェーンへの適用拡大は[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-71として起票した）。
  - ほかに`audioLatentsAvailable()`（アップロードした音声が何潜在フレームになるか）、`audioSegmentWindows()`（各クリップの担当区間。後述の🎵バッジの値）、`maxChainAudioLatents()`、`minClipFramesForKv()`を追加した。既に複数クリップ対応で書かれていた`audioLatentsRequired`はそのまま再利用している。
- **自動調整のリゾルバは1つの純関数に閉じた**（新規`audioFit.ts`の`planAudioFit()`）。Reactにも文言にも依存しない。**オーナーが確定した規則は「ユーザーが触ったカードには手を出さない」の一点に集約される**。各クリップカードは`intact`（未操作かどうか）という**単一のフラグ**を持ち、プロンプトや尺を実際に変更した時点で**不可逆に**falseになる。falseになったカードは、リゾルバから見て**長さ・位置・存在そのもの**が凍結される。フラグを1本にしたのは、「尺だけ触った」「位置だけ触った」と種類を増やすと、どのカードが動くのかをユーザーが予測できなくなるためである。
  - 未操作のカードはスライダー「追加クリップの長さ」の値へ揃え、**配列順で最後の未操作カード1枚だけが端数を吸収する**（下限は`max(9, 8×重なり幅+1)`。既存の「重なり幅は各クリップの潜在フレーム数より小さいこと」という制約から来る実効下限である）。枚数は2〜24の範囲で探索し、24枚でも足りない場合はエラーにせず**24枚まで埋めて「末尾の何秒かは使われません」と注記する**（生成自体は許可する）。
  - **提案した構成は必ず`chainLayoutError`で検査してから返す**。不合格なら端数のカードを8フレーム刻みで調整し、それでも駄目なら1枚少ない安全側の構成へ落ちる。「未操作のカードが1枚も無い」「音声が短すぎて最小構成にも足りない」場合は**何も提案せず、呼び出し元のカードをそのまま返す**（このときだけ上記の保証は付かない——何も提案していないため）。
  - **安全マージンとして音声潜在1フレーム（0.04秒）を使い残す**。サーバーはffprobeで、フロントエンドは別の経路で音声の長さを測るので、両者はコンテナのヘッダ由来の丸めぶんだけずれる。最後の1フレームまで使い切る構成は、サーバーの計測が1フレーム短く出た瞬間に422になる。0.04秒を捨てるだけでこの失敗の形が丸ごと消える。
- **画面はChainedタブに音声専用カードを1枚足した**（新規`ChainAudioPanel.tsx`）。見出しは「音声（長尺A2V）」で、作りは既存の`SourceInputPanel`（V2Vの元動画カード）と同じ枠組みに乗せた（ドラッグ＆ドロップとファイル選択の両方に対応）。**音声は1本だけで、分割ファイルは一切作らない**——クリップごとの担当区間はサーバーが割り当てるので、フロントエンドが音声を切る理由がない。カードには「↔️再生時間の自動調整」ボタンと「追加クリップの長さ」スライダー（8n+1刻み・9〜481・初期値257）を置いた。スライダーはプリセットを適用したときにそのプリセットの推奨クリップ尺へ**再シード**される（固定の257のままだと、たとえば768p系のプリセットを選んだ直後に↔️を押したときに推奨尺を上書きしてしまう）。音声を添付したときは、この自動調整が**1回だけ**自動で走る。
- **各クリップカードに「🎵 担当時間帯 X.XX〜Y.YY秒」バッジを出した**（`ClipCard.tsx`）。値はサーバーの窓割り当てちょうどのミラー計算である（`audioSegmentWindows`）。これは飾りではなく、**このバッジが無いと「どのクリップにどの音が乗るのか」を確かめる手段がユーザーの側に存在しない**（生成し終えるまで分からない）ためである。
- **生成できるかどうかの判定は、自動調整とは切り離してSingle画面の作法に揃えた**。↔️ボタンを押さないと生成できない、という作りにはしていない。
  - 音声のほうが**長い**ときは、末尾を切り捨てて生成できる（注記だけを出す）。
  - 音声のほうが**短い**ときはブロックし、「音声が短すぎます。動画を短くするか、↔️再生時間の自動調整を押してください。」と出す。
  - V2V（動画の続きを生成する機能）の元動画と同時に入力されているときは「v2vとa2vは併用できません」でブロックする。**どちらか一方を外せば復帰する**（サーバー側で排他になっているため、片方を残す判断はユーザーに委ねる）。
  - そもそも**Chainedで扱うには短すぎる音声**——1本目のクリップの現在の尺で判定して、2本目に最小尺すら確保できない長さ——は「短い音声はSingleで生成してください。Chainedには2クリップ以上が必要です。」でブロックし、単発生成へ誘導する。
  - 24枠×スライダー値でも収まらない音声は、24枠まで埋めて注記を出し、生成は許可する。24×481フレーム相当（チェーンの総尺上限）を超える音声は、そもそも添付の時点で弾く。
  - 上記の`chainLayoutError`は、自動調整の結果検査と生成可否の**両方**に配線した。非標準のフレームレートで、サーバーの幾何検算に落ちる構成をユーザーが手で組んでしまった場合にも、生成ボタンの側で先回りして止まる。
- **右クリックにも導線を2つ足した**（`plugin.cpp`のオブジェクトメニューが11→13項目）。「🎬 Video: この動画の音声でlong a2v」と「🎵 Audio: この音声からlong a2v」で、どちらもChainedタブへ遷移して音声を自動添付する。**タイムラインのリボン（オブジェクトの表示帯）の範囲を尊重して、右クリックした時点で音声を切り出す**（既存の`timeline.extractAudio`をそのまま使う。バックエンドには切り出し機能を足していない）。🎬は**mix録り**（タイムラインで実際に鳴っている音）、🎵は**solo録り**（そのオブジェクトのレイヤーだけを鳴らす）で、既存の単発a2v（#3・#7）と対になる意味論である。**切り出しの返り値に音声の長さが入っているので、それをそのまま自動調整へ渡す**——ファイルを測り直さないぶん、長さの計測誤差が入らない。設計判断の正本は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §3-4の#17・#18。
- **既存機能への波及は2箇所**。Single画面のA2V欄の注記を「長い音声はChainedのlong a2vが使えます。」へ差し替えた（長い音声を持ち込んだユーザーの行き先ができたため）。バッチi2v-long（画像フォルダから長尺動画をまとめて作る機能）は、Chainedフォームの音声を**防御的に剥がして**送る——各画像がそれぞれチェーンの先頭になる作りなので、音声が付いたままだと意味が通らない。
- **変更ファイル（webui、未コミット）**: 新規5点（`modes/chained/audioFit.ts`／`audioFit.test.ts`、`modes/chained/ChainAudioPanel.tsx`／`ChainAudioPanel.test.tsx`、`modes/chained/ChainedScreen.audio.test.tsx`）。改修は`modes/chained/chainUtils.ts`（＋テスト）／`useChainForm.ts`（＋テスト）／`ChainedScreen.tsx`／`ChainedScreen.css`／`ClipCard.tsx`／`generateReasonMessages.ts`、`modes/single/GenerationForm.tsx`、`modes/batch-i2v-long/`の4点（`buildI2vLongPayload.ts`／`chainSnapshot.ts`／`useBatchI2vLongForm.ts`＋各テスト）、`timeline/menuRouting.ts`／`menuSelection.ts`／`prefillSeed.ts`（＋各テスト）、`shell/useFileDrop.ts`、`i18n/strings.ts`、`App.prefill.test.tsx`。native/Languageは`native/src/plugin.cpp`／`Language/Japanese.NzVideomni.aul2`／`Language/English.NzVideomni.aul2`。
- **検証**: フロントエンド`npm run test` **2169 passed／10 skipped**（着手前2001）、`npm run typecheck`（`tsc -b`。`npx tsc --noEmit`は偽合格になるので使わない）0エラー、nativeビルド（MSVC）成功。バックエンドpytest **1131 PASS／15 skip／0 FAIL**（着手前1115）。リゾルバの性質テストは**フレームレート7種（23.976／24／25／29.97／30／50／60）×重なり幅1〜8**で回し、「結果が常に`chainLayoutError`なし」「操作済みカードが変わらない」「同じ操作を二度しても結果が変わらない」の3点を確かめている（24fpsだけで回すと丸めと幾何の穴をすり抜けてしまう）。
- **実装の進め方**: 着手前に敵対的レビューを1回入れ、14件の指摘のうち**致命2件（複数クリップでの映像つなぎ目の硬直／非標準フレームレートでの422の量産）を実測で裏づけてから**計画へ織り込んだ。上の2点はどちらも、実装してから気づいたのでは「生成してみるまで分からない」種類の故障である。
- **既知の限界（意図的）**: (1)非整数フレームレート（23.976fpsなど）では、動画エンコードへ渡すフレームレートが整数へ切り捨てられるため映像が約4.2%長くなり、長尺では音ズレとして目に見える。これは長尺A2V固有ではなく**全生成経路に共通する既存の問題**で、修正がvendorパッケージの改修と再配布を伴うため[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-72として起票した。(2)非標準フレームレートでの422は、音声を添付していない素のチェーンでは依然として起こりうる（同§3-71）。
- **状態**: 実装・機械検証は完了。**mock通しの最終確認と、AIエージェントによる実機ゲート、オーナーの実機確認が残っている**（オーナーの確認項目は当時の[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-2「長尺A2V（§1-16）の確認」6項目。合格後は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-74の本文へ吸収済み）。**コミットは未実施。**→ 下記の追記のとおり、2026-08-10に全項目合格・コミット済み。
- **追記（2026-08-10・オーナー目視ゲート全項目合格でテーマ完結）**: **6項目すべて合格した**。①右クリック🎵からの一連の流れ（Chainedタブが開く・音声の自動添付・クリップ構成の自動調整とトースト）が通り、トリムした音声オブジェクトではリボンに見えている長さぶんだけが読み込まれること、②右クリック🎬（mix録り）も同じ流れで通ること、③手でドラッグ＆ドロップ／ファイル選択しても同じ流れになり「🎵 担当時間帯」表示がクリップ構成の変更に追従すること、④↔️ボタンと「追加クリップの長さ」スライダーの操作感（手で触ったカードは動かず、触っていないカードだけが作り直される）、⑤ブロックと注記の文言4種、⑥生成結果の品質（つなぎ目が固まらない・音声が全編にわたって元の波形どおり）。これでテーマは完結した。台帳は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-74（`PENDING_TASKS.md` §1-16は欠番）。デプロイ済み（実機＋backend配布コピーの2箇所。deploy.ps1）。**コミット・プッシュ済み**（バックエンド`aaa08fb`／フロントエンド`401b3b4`）。
- **追記（2026-08-11・目視ゲート後のオーナー追加注文2件）**: 合格と同時にオーナーから見た目の注文が2件出たので実装した。**(a)余り注記の文言**——音声が余るときの注記を「**動画の尺が足りないため、**音声の末尾 約X秒は使われません。」へ改め、なぜ余るのかがその場で読めるようにした（`i18n/strings.ts`の`chained.sourceAudio`配下2キー。クリップ枠が上限24枚に達したときの注記も同じ文頭へ揃えてあり、2つの注記が食い違わない）。**(b)素材カードの解像度表示**——Chainedタブの素材カード（`SourceInputPanel.tsx`。動画モード・画像モードの両方）に、添付した素材の解像度を「1280×768」形式でファイル名の横へ表示するようにした。幅・高さのスライダーを合わせるときの手がかりになるためで、手本はEditタブ→Retakeサブタブの素材カード（本書§69の②）である。取得は`fs.probeMediaInfo`を添付1回につき1度呼ぶ形で、幅か高さが0（読み取れない）のときは解像度の部分ごと省く。**この2件は同じ2026-08-11のオーナー目視でどちらも合格し、追加改修も完結した**（コミットはフロントエンド`f2c4650`／バックエンド配布コピー`595c5d8`。合格の記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-74へ吸収済みで、[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-2の2項目は除去した）。

---

## 71. Retake改修第2弾 — ❌ボタン（全リセット）・🔁ボタン（設定持ち越し・一度きり）・読み出し行の文言・設定欄の並び順（2026-08-10）

**やったこと**: §69で改修したRetakeパネルに、素材カード右上へ2つのボタンを新設した。台帳は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-73（記録当時は`PENDING_TASKS.md` §1-17。2026-08-10のテーマ完結でクローズ移設）。

- **①❌ボタン（全リセット）**: 素材カード右上に配置。押すと、Retakeサブタブが右クリック前の案内文だけの状態へ戻る。設定もすべて既定値へ戻し、タイムラインに出ている⏳予約リボンも片付ける。**Retakeサブタブの外（メインプロンプト・Outpaintingサブタブ）には手を出さない。**
- **②🔁ボタン（設定を残したまま撮り直す）**: ❌の下、素材が読み込まれているときだけ表示。押すと、動画本体と選択区間だけを捨て、設定欄（撮り直す対象・幅・高さ・fps・シード・Stage-2クリップ長）は数値ごと画面に残ったままになる。次にタイムラインで動画を右クリックすると、対象・fps・シード・Stage-2クリップ長がそのまま復元される。**幅・高さだけは新しい素材の実寸に合わせて導出し直す**（オーナー確定仕様——古い解像度の値を新しい素材に押し付けると壊れた見た目になるため）。**持ち越しは一度きり**——🔁を押した直後の右クリック1回にだけ効き、それ以降（🔁を押さずに普通に右クリックした場合）は従来どおり全項目が初期値に戻る。この一度きりの機構はReactのstateではなくモジュールスコープに置いた（`useRetakeForm.ts`）。パネルがアンマウント・リマウントされても（別タブへ切り替えて戻る、など）持ち越しの意図を保つためである。
- **③読み出し行の文言**: §69で入れた「タイムラインの{start}〜{end}フレーム目を撮り直します」の末尾に「（※フレーム数は「8の倍数+1」です）」を追記した（日英とも）。窓が8n+1へ丸められる仕様そのものは§68から変わっていないが、丸めの理由がその場で読めるようにした。
- **④設定欄の表示順変更**: 撮り直す対象→幅→高さ→フレームレート→シード→Stage-2クリップ長、の順に統一した（注意書きとstale警告は最後尾のまま）。
- **安全設計（❌／🔁と予約リボンの整合）**: ❌・🔁が⏳予約リボンを片付けるのは、**「その席が今のRetake右クリックに由来する席のままか」をpendingIdで突き合わせてから**にした（`provisionalReservation.ts`の小関数2つ）。Retakeを右クリックしたあと、別タブ（Single等）で別の生成を右クリックして席が移動していた場合、Editタブへ戻って❌を押しても、他タブ側の予約リボンには触れない。専用のテストで両ケース（一致／不一致）を確認している。
- **変更ファイル（webui、未コミット）**: `modes/edit/useRetakeForm.ts`（`awaitingSource`／`clearAll`／`resetSource`とモジュールスコープの一度きり持ち越し機構）・`useRetakeForm.test.tsx`（+6本）・`modes/edit/RetakePanel.tsx`（3状態分岐と`RetakeSettings`の切り出し）・`modes/edit/EditScreen.retake.test.tsx`（+5本／更新2本）・`timeline/provisionalReservation.ts`（片付け用の小関数2つ）・`shell/AppShell.tsx`（`retakeRange`分岐の`onReserved`1行）・`i18n/strings.ts`（新規キー3・読み出し行への追記）。
- **検証**: `npm run typecheck`0エラー、vitest **2182 passed**（§70時点2169→2182、+13。退行ゼロ）、lint新規ゼロ。※検証実行中に`backend.integration.test.ts`の7本が失敗したが、これは実バックエンドが別プロセスで起動・ビジー中だったためで、フロントエンドの変更とは無関係である（バックエンド不在時は自動でskipされるファイルであり、今回のフロント差分に起因する失敗ではない）。
- **状態**: **2026-08-10 オーナー目視合格**（❌ボタン・🔁ボタン・持ち越しは一度きり・予約リボンの誤爆なし・文言と並び順の5項目すべて）。実装・機械検証は完了。**最後に残っていた最小窓73フレームの仕様判断も同日に「73フレーム維持（選択肢(b)・注意文で伝える方式）」で確定し、Retakeのテーマは完結した**（記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-73）。デプロイ済み（実機＋backend配布コピーの2箇所。deploy.ps1）。**コミットは未実施。**→ 2026-08-11、オーナーがコミット・プッシュ済み。

---

## 72. 長尺IC-LoRA（クリップ別の参照動画）— 参照動画1本をサーバーが自動で切り分け、各クリップのstage-1へ注入する（2026-08-11）

**やったこと**: IC-LoRA（参照動画で動きを制御する追加学習データ）は、これまでクリップ連結（Chained）では「クリップ1本のとき」しか使えなかった。この制限を外し、**長い参照動画を1本だけ添付すると、各クリップが担当する区間をサーバーが自動で切り出して注入する**形の長尺IC-LoRAを実装した。台帳は[`PENDING_TASKS.md`](PENDING_TASKS.md) §2（実装当時の§1-15。旧§1-15は欠番）で、バックエンド側の実装と機械検証の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §57である。本節はフロントエンドを軸に、設計に効いたバックエンドの要点を添えた記録である。

- **入力はトップレベルの1フィールドのままにした**。クリップごとに参照動画を添付させる形にはしていない。ユーザーが8n+1というフレーム数の刻みや、つなぎ目の重なり幅を計算して自分で動画を切り分けるのは現実的でないためである。サーバーは`chain_math.py`に新設した`video_segment_windows`で、クリップiが担当する参照の窓を**開始＝8×そのクリップのグローバルな潜在開始位置／長さ＝そのクリップのフレーム数**として求める。**隣り合う窓は`8kv−7`ピクセルフレームだけ重なる**（`kv`＝つなぎ目の重なり潜在フレーム数）——これは引き継ぎ（carry）が実際に舐める画素の範囲そのものなので、**つなぎ目の参照映像が前後のクリップで一致し、整合が自動的に成り立つ**。注入先はstage-1（低解像度で全体の動きを作る第1段階）だけで、stage-2（フル解像度で仕上げる第2段階）には入らない。VRAMの天井は悪化しない。
- **切り出しはフレーム番号ベースで、fps（1秒あたりのコマ数）は見ていない**。参照の潜在フレームiと生成の潜在フレームiが同じ座標に載るという性質がそのまま効くためで、逆に言えば**参照動画と生成動画のfpsが違うと、意図とずれた区間が参照される**。そこでパネルには「fpsは、参照動画と生成動画で一致させることを推奨。参照動画の（秒数ではなく）フレーム単位で参照されるため。」という注記（オーナー指定の文言）を常設した。
- **参照が足りないときはエラーにしない**。参照動画が生成の尺より短ければ、足りないぶんのクリップは参照なしで生成される（Single＝単発生成の既存挙動と一貫させた）。数値入りの警告は作らず、「参照動画が生成の尺より短い場合、足りない分は参照なしで生成されます。」という静的な文だけを置いている——fpsが取れない以上、「あと何秒足りない」を正しく言えないためである。
- **エンジン側は遅延の窓ジェネレータにした**（`chain_pipeline.py`）。参照動画のデコードを1本のジェネレータとしてstage-1のループ開始前に開き、**ループの中で各クリップの窓を順に消費する**（窓の開始まで読み捨て→窓の長さぶん溜める→符号化→重なりぶんを残して次へ）。事前に全編を一括で符号化する案は、進捗表示が長時間無音になるため採らなかった。溜めるバッファの上限は最も長いクリップ1本ぶんで、1パスで済み、ピークメモリも最小になる。
- **符号化した参照の潜在はGPUに置いたままにした**。24クリップぶんでも約23MBしかない一方、CPUへ退避すると適用時にデバイスの食い違いで壊れる。「節約になっていない節約」を入れないという判断である。
- **depth（深度）系のアダプタは多クリップでは見送り、明示的な422にした**（`LORA_DEPTH_CHAIN_UNSUPPORTED`）。深度マップを作るVideo-Depth-Anythingの前処理が全編を一度にメモリへ載せる設計で、チェーン全体の尺では入出力を合わせて数十GBになるためである。しかも32フレームの窓を10フレーム重ねて全編を舐める作りかつ全編で1回の正規化を行うので、素朴にチャンク化すると**時間の整合と明るさ（奥行きの尺度）の両方が壊れる**。画面側でも、選んだアダプタの前処理種別が`depth`でクリップが2本以上なら先回りしてブロックする（種別は`GET /loras`へ新たに露出させた`preprocess`で判定する）。解禁の検討は[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-75へ起票した。
- **バッチi2v-long（画像フォルダから長尺動画をまとめて作る機能）では参照を剥がす**（オーナー決定）。各画像がそれぞれチェーンの先頭になる作りなので、共通の参照動画が付いたままでは意味が通らない。元動画・音声を剥がしているのと同じ位置で参照と強度系のフィールドも落とし、画面にはブロック理由を出す（既存2種の前例に揃えた）。
- **参照が有効なあいだは解像度スライダーの刻みそのものを128へ切り替えた**。参照つきの生成は幅・高さが128の倍数である必要があるが、Chained画面の刻みは64でハードコードされていた。**1回スナップするだけの直し方では、その後スライダーを動かした瞬間に64刻みへ戻って恒久的にブロックされる**——Single画面が既に持っていた方式（参照が有効なら刻み自体を128にする）をそのまま写した。ゲート（`referenceDimensionsOffGrid`）は保険として残してある。
- **アップロードの時点で11544フレームへ切り詰める**。チェーンの総フレーム数の上限がこの値なので、それを超える参照は使いようがない。`POST /uploads/video`に`max_frames`を足し、**実測が上限以下なら一切変換しない**（再エンコードしないので画質は劣化しない）。超過したときだけ先頭を切り出す。
- **アップロード失敗表示の偽陽性を直した**（`useSourceUpload.ts`）。「トリムを頼んだのに行われなかった」を判定する式が、クエリ文字列が付いているかどうかで判断していた。`max_frames`は**トリムされないのが正常**なので、そのままでは参照動画を添付するたびに失敗表示が出る。判定をトリム系のキー（`trim_start_sec`）の有無へ改めた（V2V経路の意味は変えていない）。
- **stage-1の負荷警告は琥珀色・非ブロッキングにした**。参照はstage-1のトークン数をおよそ2倍にするため、チェーンでは初めてstage-1が詰まりうる。見積もりの式はバックエンド`chain_math.py`の`chain_stage1_tokens`に置き、`webui/src/shell/tokenBudget.ts`はその**ミラー**として同じ形の式を書いている（既存の規律に合わせ、フロントエンド専有の定数は作らない）。**閾値25000トークンは実装時点では暫定値**だったが、実機ゲートG4（後述）の較正で実測の膝と一致することを確認し据え置きが確定した。担当秒数のバッジは付けない（オーナー決定。fpsに依存する値を秒で見せると、fpsがずれている構成で嘘を表示することになる）。
- **画面はChainedタブに参照専用カードを1枚足した**（新規`ChainReferencePanel.tsx`）。既存の`ChainAudioPanel`（長尺A2Vの音声カード）を雛形にし、制御アダプタの選択・強度スライダー2本・サムネイル・解像度表示を載せている。**制御アダプタの選択状態はChained画面のローカルに持ち、Single画面とは共有しない**（両画面の入力欄を分離するという方針どおり）。
- **「制御アダプタが選ばれているか」の判定は、マージ後のLoRA一覧を見るようにした**。パネルの選択状態だけを見ると、プロンプトへ手打ちした`<lora:…>`タグで制御アダプタを指定したケースを取りこぼし、サーバーの422に落ちる。到達不能になった旧エラーコード`LORA_CONTROL_UNSUPPORTED_IN_CHAIN`と、それに紐づく画面側のバナー・文言は削除した（引き算の原則）。
- **敵対的レビューで潰した主な欠陥**: 上記のうち①depth前処理のメモリ不足②参照潜在をCPUへ退避するとデバイスの食い違いで壊れる③手打ちタグの取りこぼし④128刻みへ切り替えないと恒久ブロックになる⑤`trimFailed`の偽陽性——の5点は、いずれも**設計段階の原案には無かったか、逆向きに書かれていた**ものである。実装前の敵対的レビュー（指摘はすべて実コードで裏取り）で検出し、計画へ織り込んでから着手した。あわせて原案にあった「潜在のCPU退避」「部分的な窓を8n+1へ切り下げる処理」「上限に+8する余裕」「事前一括符号化」は、過剰設計またはバグの源として削っている。
- **変更の規模**: バックエンド26ファイル（+1486／−449行）、フロントエンド20ファイル＋新規3点。バックエンドの主な変更先は`chain_math.py`（`video_segment_windows`／`chain_stage1_tokens`／`CHAIN_STAGE1_COMFORT_TOKEN_BUDGET`）・`api/models.py`・`api/generate_chain.py`（クリップ1本限定のガード解除とdepthの422新設）・`engine/pipeline/common.py`（フレーム逐次のイテレータ化）・`fast_video_pipeline.py`（画素から条件を作る部分の切り出し。段の判別は外側に残した）・`chain_pipeline.py`（遅延の窓ジェネレータと全セグメントへの注入）・`engine/worker.py`（前処理のフレーム上限をチェーン全体の尺へ。canny／dwposeにも上限を効かせた）・`services/video_io.py`／`video_upload_store.py`／`api/uploads.py`（アップロード時のトリム）・`services/pipeline_manager.py`（チェーンのmetadataへ`ic_lora`ブロック）・`services/lora_registry.py`（`GET /loras`へ`preprocess`と`reference_downscale_factor`を露出）。フロントエンドは`modes/chained/useChainForm.ts`・新規`ChainReferencePanel.tsx`・`ChainedScreen.tsx`・`shell/tokenBudget.ts`・`shell/useSourceUpload.ts`・`modes/batch-i2v-long/`・`i18n/strings.ts`（英日）が主な変更先である。
- **検証**: バックエンドのアプリ用仮想環境のpytest **1189件 PASS**（skip 19件はtorchを積んでいない環境での既知のもの）、エンジン用仮想環境（torchあり）**55件 PASS**、フロントエンドvitest **128ファイル2232件 PASS**（skip 10件は既存）、`npm run typecheck`（`tsc -b`。`npx tsc --noEmit`は偽合格になるので使わない）0エラー。加えて**mockの実サーバーへ通した**——`max_frames=11544`付きでアップロード → 2クリップ＋canny＋参照で送信 → 202 → 完了まで到達し、metadataの`ic_lora`ブロックに`reference_segment_windows: [[0, 25], [16, 25]]`が載ることを確認した。窓の式がPython側とTypeScript側で逐語一致していることも目視で突き合わせている。
- **実機ゲートの前提として見つかったアップロード不能バグ（2026-08-11）**: 実機ゲートの初回実行時、参照動画のアップロードが全経路で422になる不具合が見つかった。原因はネイティブHTTPクライアント`native/src/http_client.cpp`の`CrackUrl()`にあり、`WinHttpCrackUrl`は`ExtraInfo`を要求しないモードではクエリ文字列を`UrlPath`に含めて返す仕様なのに、呼び出し側がさらに手でクエリを連結していたため`?max_frames=11544`が二重に付いたURL（`?max_frames=11544?max_frames=11544`）になっていた。単一キーのクエリを使う製品コード経路は長尺IC-LoRAが初めてで、これまで露見していなかった。手打ちの連結処理4行を削除して修正し、`CrackUrl`を`nzvideomni::detail`へ露出したうえで純関数の単体テストを4件追加した（修正前に失敗を再現→修正後に既定スイート278件全緑）。非2xxレスポンス時に実際に送ったURLをログへ残す診断行も1行足した。修正版は09:54にデプロイ済みだが、**オーナーによる実機カードの再確認はまだ済んでいない**。
- **実機ゲートG1〜G8の結果（2026-08-11、G4を除き全PASS）**: 上記バグの修正後、実GPUでG1〜G8のうちG4以外を実施した。**G2（長尺A2Vとの併用）・G3（参照が足りないとき）・G5（単発生成の回帰）・G7（アップロード時のトリム）・G8（depth系×多クリップの拒否）はPASS**——G5は現行コミット2本＋旧コミット1本の同一シード生成がSHA256完全一致し、リファクタが動きを変えていないことを実機で確定した。G3は481f×4クリップ（必要参照1873fに対し実参照921f）でも422にならず、参照が尽きた後半は参照なしで完走した。**G1（本命の追従ゲート）は機械判定分（`reference_segment_windows`が幾何の理論値と完全一致）は合格したが、つなぎ目で被写体が飛ばないかどうかのオーナー目視はまだ済んでいない。** **G6（前処理コストの実測）は数値の実測は完了した（canny約7.8ms/フレーム、dwpose約48.25ms/フレーム）ものの「許容範囲」の判定基準が未確定**である。**G4（stage-1負荷閾値の較正）はオーナーの開始許可を得たうえで同日中に実施し、PASS**した。1152×1536・24fps・canny-controlで参照なし361f/481f・参照あり241f/361f/425f/481fの計6点を計測し、全体所要時間は参照なし→ありの倍率が361fで1.39倍・481fで1.42倍。stage-1本体の時間はトークン数に対して滑らかな冪則で膝が無く、**真の膝は参照動画のVAEエンコード（タイル化されていない処理）にあった**——エンコード単価が241fの0.0205秒/フレームから481fで0.1263秒/フレーム（6.17倍）へ急変し、ジョブ全体のVRAMピークもこの区間のピークと一致する。VRAMピークはトークン数に対して線形（`peak_MB = 1076 + 0.8094 × トークン数`）で、物理16376MBを超えるのは18903トークン（273フレーム相当）。**結論として暫定閾値`CHAIN_STAGE1_COMFORT_TOKEN_BUDGET = 25000`は据え置き**——1152×1536で361フレームちょうどを許可する現行閾値がコスト曲線の肘に正確に載っていることを実測で裏付けた。留意点として、①この無害さはNVIDIAのsysmem fallbackが有効な前提であり無効環境では実質閾値が下がる（[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-77へ起票）、②真の膝である参照VAEエンコードのタイル化不足は改修候補として同§3-76へ起票した。詳細はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §57.6・§57.7。
- **【2026-09-05 追記】上のG4が残した留意点①②は、どちらもその後クローズした。** ①のsysmem fallback無効環境への配慮は2026-09-02にREADMEへの注記の実施で（バックエンド[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-77）、②の参照動画VAEエンコードのタイル化は2026-09-05に実装とオーナー裁定で（同書§3-76）決着している。**参照自身の潜在トークン数がしきい値を超えると符号化が自動でタイル化へ切り替わるようになったので、上の「エンコード単価が6.17倍へ急変する」という膝は、その帯にはもう現れない**（実装・実測の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §101）。**ただし閾値`CHAIN_STAGE1_COMFORT_TOKEN_BUDGET = 25000`は据え置きのままで、フロントエンドは1行も変わっていない**——線の再較正は後継の課題（バックエンド[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §4-39〔旧§3-144〕）が引き継いだ。**生きた台帳側の§3-76・§3-77はどちらも欠番である**ので、番号をたどるときはクローズ側を見ること（この段落の①②が書いている`PENDING_TASKS.md`リンクも、当時から相対パスが実体を指していない。バックエンドの文書は`../../../Docs/`配下である）。**本文は不変である。**
- **状態**: 実装・機械検証・実機ゲート（G1〜G8全項目）は完了。**デプロイは実施済み。残っているのはG1のつなぎ目目視、オーナー目視（パネルの意匠・文言）である**（確認項目は[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-1・§2-2）。**コミットは未実施。**→ 2026-08-11、オーナーがコミット・プッシュ済み。

---

## 73. 長尺IC-LoRAのオーナー目視合格と目視フィードバック改修バッチ、右クリック新経路の追加（2026-08-11）

**やったこと**: §72で実装した長尺IC-LoRA（クリップ別の参照動画）について、残っていたオーナー目視（実機ゲートG1・G6、台帳[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-2の初弾6項目）の結果を受け取り、指摘に沿った改修バッチを実装・デプロイした。あわせてタイムラインの動画オブジェクトから直接この機能へ入れる右クリック経路を新設した。バックエンド側のオーナー目視合格の反映は[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §57.6が正本である。

- **オーナー目視結果（実機ゲート）**: **G1（多クリップ追従）は合格**。オーナー自身のテスト2本——3クリップ+canny（`outputs/2c982529-…`）・4クリップ+DWpose（`outputs/75b2e678-…`）——で、参照なし生成と同程度の連結歪み（つなぎ目のモーフ）はあるものの、参照からの形状・動きの制御は全編を通じて反映され、タイミングのズレや動きの飛びは無かった。**G6（前処理コスト）も合格**。実測値（canny約7.8ms/フレーム、dwpose約48.25ms/フレーム）を確認したうえで実用的な時間内と判断した。これで実機ゲートG1〜G8は全項目が名実ともに完結した。
- **§2-2初弾6項目の結果**: カード見た目＝**合格**（ブラッシュアップは下記のレイアウト再構成で実施）、注記の出し分け＝**概ね合格**（V2V競合時の文言のみ調整）、probe失敗ケース＝**未検証だが素材が無くレアケースにつきオーナー判断でクローズ**、stage-1警告バナー＝**合格**（位置は下記のレイアウト再構成で調整）、ブロック理由＝**合格**（アップロード中ゲートはローカルでは高速すぎて検証不能につきオーナー判断でクローズ、バッチi2v-long文言は下記で調整）。
- **文言3件の改修**: fps注記の助詞を修正した。V2V（元動画）とIC-LoRAの併用不可を伝える文言を「v2v（動画延長）とIC-LoRAは併用できません。」系へ統一し、カード内の注記とボタン下の姉妹文言の両方に揃えた（従来は2箇所で言い回しが微妙に食い違っていた）。バッチi2v-longのブロック理由を「IC-LoRAの参照動画を外してください。このバッチは画像だけから生成します。」へ短縮した。
- **depth系アダプタの文言**: depth系のIC-LoRAアダプタを選んだときは常時ブロックし、「未実装のIC-LoRAアダプタが選択されています。」と表示するようにした。**これは画面側の意図的な判断で、サーバーの実際の仕様（1クリップ+depthは通る）より厳しい**。1クリップ+depthは他のゲートで到達不能な組み合わせのため実害は無い——単クリップと多クリップで文言を出し分ける複雑さより、単純に常時ブロックするほうを選んだ。
- **Chainedタブのレイアウト再構成**: VRAM系アドバイザリーバナー2本（stage-2予算超過・stage-1参照超過）をバッジ行の直下へ移設した。予想出力表示を「クリップを追加」ボタンの直下へ移した。「つなぎ目のブレンドやV2Vの…」という説明文は削除した。参照動画・音声カードをアコーディオン化し、「参照動画（IC-LoRA）」→「音声から動画（A2V）」の順で並べた（Single画面と同じ作法）。右クリックからのプリフィル時はアコーディオンが自動展開される。アコーディオンと「つなぎ目のブレンド」の間には罫線を1本入れた。
- **右クリック新経路の新設**: タイムラインの動画オブジェクトを右クリックすると「🎬 Video: この動画を参照に長尺IC-LoRA生成（Chained）」が選べるようにした（右クリックメニューは13→14項目）。選ぶとChainedタブへ新規プリフィルされ、その動画が参照動画カードへ自動アップロードされる。**リボンの範囲選択トリムがここでも適用され**、11544フレーム相当へクランプされる。トリムに失敗したときは新設の`referenceTrimFailed`ゲートが生成をブロックする——**この失敗ゲートは実装前の敵対的レビューで欠落に気づいて追加したもの**で、トリム窓がチェーンの`max_frames`上限を素通しする仕様への対処でもある（フロントエンド側で上限へクランプする形にした）。
- **敵対的レビューで事前に潰した欠陥**: ①参照trim失敗ゲートの欠落（上記）②trim窓が`max_frames`上限を素通しする仕様（FE側クランプで対処）③DURATION方針表・128グリッドseedへの新経路の登録漏れ④既存テスト3本を壊す予見（レイアウト再構成・アコーディオン化の影響範囲を先に洗い出した）⑤バナー用の新しい「枠」コンポーネントを新設する過剰設計案の回避（既存のバッジ行・カード構造をそのまま転用した）。
- **検証**: vitest **2242件緑**（実サーバー使用中による既存の統合テスト失敗7件を除く）、`npm run typecheck` **0エラー**、ネイティブテスト **278件全緑**。デプロイ済み（15:27、両aux2＋Languageの2ファイル）。
- **状態**: **実装・検証・デプロイ完了。長尺IC-LoRAの実機ゲートG1〜G8はG1・G6のオーナー目視合格をもって全項目完結した**（台帳[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-1は本結果を受けて除去済み）。本バッチ自体の確認項目（新レイアウト・右クリック新経路・depth文言・文言3件）は新たな目視待ちとして当時の台帳`PENDING_TASKS.md` §2-2へ差し替えた（同§2-2は全項目合格後に見出しごと削除済み。合格記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-78）。**コミットは未実施。**→ 2026-08-11、オーナーがコミット・プッシュ済み。

**追記（同日16:32デプロイ、微修正バッチ）**: §2-2の確認項目のうちstage-1バナー文言とin-outpaintingの扱いについて、オーナー指定に沿った微修正をもう一段実装・デプロイした。

- **stage-1参照超過バナーの文言を最終指定へ**: 「長すぎるクリップがあるため、VRAM溢れにより生成速度が低下するリスクがあります。各クリップのフレーム数を短くするか、動画の解像度を下げることを推奨します。」に差し替えた。stage-2バナー（予算超過）と同時に点灯するケースでも、冒頭の言い回しで両者を区別できることを意識した文面にしている。
- **in-outpaintingアダプタをSingle/Chainedの制御アダプタドロップダウンから除外**: in-outpainting（Editタブ・Outpainting専用のIC-LoRAアダプタ）は、Single・Chained画面の「制御アダプタ」ドロップダウンに出す意味が無い（Outpainting以外の用途では使えない）ため、選択肢から外した。**表示用の集合（`selectableControlLoraNames`）と判定用の集合（`resolveControlLoraNames`）を分離**する形で実装し、ドロップダウンの表示だけを絞り込んだ——手打ちの`<lora:…>`タグの事前ブロック・チップの抑制表示・Editタブでのピン留め表示は、判定用集合を参照したままなので無傷である。あわせて**mockフィクスチャにもin-outpaintingを追加**した（mockの「`config.yaml`と1対1で持つ」という既存規定に追随したもの）。この追随の副産物として、**mock環境でEditタブが`loraMissing`表示になっていた既存の不具合も解消**した。
- **depth系との作法の違い**: depth系のアダプタは従来どおり「選択肢には残すが生成時にブロックする」（§73既述のとおり）。in-outpaintingは専用の置き場（Editタブ）があるためドロップダウンから完全に除外する——同じ「使えない組み合わせ」でも、置き場の有無で2つの作法を使い分けている。
- **既知の割り切り**: Gradio UI（バックエンド同梱の簡易UI）のアダプタ選択にはこの除外が効かない。Gradio側は`/config`のキーを直接列挙する実装のため。**α版はWebUI側のみで対応する**方針とした。
- **古い記述2件の修正**: `api/generate_chain.py`のコメントが「control系アダプタは全部`reference_downscale_factor=2`」という古い前提のままだったのを、「2または1（deblurなど）」へ修正した。[`FEATURE_GUIDE_KEYFRAMES_AND_ICLORA.md`](FEATURE_GUIDE_KEYFRAMES_AND_ICLORA.md)のIn-Outpainting「未対応」表記も、実装済みの実態に合わせて更新した。
- **検証**: vitest **2246件緑**・`npm run typecheck` **0エラー**・ネイティブテスト**278件全緑**・バックエンドpytestも全緑。**デプロイ済み（16:32、両aux2一致）。**
- **台帳の反映**: 当時の台帳`PENDING_TASKS.md` §2-2に「stage-1バナー文言＝調整済み・新文言の目視待ち」「in-outpaintingがドロップダウンから消えていることの確認」を追加した（同§2-2は後日全項目合格して見出しごと削除済み。合格記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-78）。

**追記（実機ゲートG9・G10）**: G1〜G8はunion-control系（`reference_downscale_factor=2`）での多クリップ実測だったため、`reference_downscale_factor=1`のdeblur・pixel-spatial-upscaler系（タイル化エンコード経路）が多クリップでどう振る舞うかは未実測のまま残っていた。エージェントが実GPUで追加実施した。

- **G9（deblur×2クリップ×217フレーム、1152×640・シード12345）**: **PASS**。325秒で完走。`ic_lora.reference_segment_windows`は`[[0, 217], [200, 217]]`で理論値一致。peak_vram 8846MB・reserved 15136MB。出力は`outputs/fd1a7a50-…/output.mp4`。**factor=1タイル化エンコード経路の初の多クリップ実測**（G4の線形モデルの適用外領域を埋めるもの）。
- **G10（pixel-spatial-upscaler-x2×同構成）**: **PASS**。273秒で完走。窓一致。peak_vram 8835MB・reserved 14446MB。出力は`outputs/6598a98e-…/output.mp4`。**参照の当て方が本来用途（低解像度参照の2倍拡大）と異なるため、出力の見た目は品質評価には使わない**——疎通確認が目的。
- 両ジョブとも参照動画のタイル割り（n_tiles=3、`[[0, 22], [18, 22], [36, 17]]`）が一致し、**タイル化エンコード経路の窓割りが多クリップ構成でも安定している**ことを確認した。
- 詳細はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §57.8が正本。台帳[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-2に「G9/G10出力の目視（G9は品質参考、G10は疎通確認扱い）」を確認項目として追加した。

**追記（2026-08-11夕、最終目視合格・テーマクローズ）**: 上記の改修バッチ・微修正バッチ・G9/G10を含む§2-2の確認項目すべてについて、オーナーの最終目視で合格の回答を得た。これにより実機ゲートG1〜G10・2巡の目視・微修正バッチのすべてが決着し、長尺IC-LoRA（クリップ別の参照動画）はテーマとして完結した。台帳[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-2は見出しごと削除し、完結の記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-78へ移設した（旧§1-15は欠番）。**コミット・プッシュ済み**（オーナー本人による実施）。

---

## 74. Single a2vの全長stage-2化（stage-2窓の「全長」プリセット追加）— §1-19実装完了（2026-08-12）

**やったこと**: SingleタブのA2V（音声から動画を生成する機能）は内部的に1クリップのチェーンとして実行されるため、これまでstage-2（アップスケール工程）が潜在22フレームの固定窓によるタイル処理になっていた（台帳[`PENDING_TASKS.md`](PENDING_TASKS.md) §1-19、バックエンド[`CHAIN_STAGE2_RESEARCH_NOTES.md`](../../../Docs/CHAIN_STAGE2_RESEARCH_NOTES.md) §7の棲み分け原則が出典）。stage-2窓プリセットへ`"full_length"`（潜在61フレーム＝481フレーム相当、前進61、のり代0、タイル数1）を新設し、SingleとBatchのA2Vが常にこの全長窓で動くようにした。エンジン（GPU側コード）は無改修で、既存のノイズ量系列・デノイズループがそのまま「全編を一度に仕上げるstage-2」として働く。実装・機械検証・デプロイの正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md)の該当節。本節はフロントエンド側の記録である。

- **型設計の判断（共有型は意図的に2値のまま据え置き）**: `webui/src/shell/tokenBudget.ts`が持つ共有型`Stage2Window`は`"standard" | "high_resolution"`の2値のままとし、`full_length`は加えていない。理由は、この型がChained・Retakeという「窓を選ぶ画面」の入力ドロップダウンの型として使われているためで、A2Vの`full_length`はユーザーが選ぶ選択肢ではなくSingle/Batchのリクエストビルダーが常に固定送出する値だからである。型を3値に広げると、Chained・Retakeのドロップダウンにも`full_length`が選択肢として現れかねない実装ミスの経路を新たに開くことになる。ワイヤ型（`api/types.ts`側の`GenerateRequest`等）は3値化したが、UI選択用の共有型とは意図的に分離した。
- **露出防止ガード**: `full_length`がChained・Retakeのフォームに紛れ込まないよう、型レベル（`STAGE2_WINDOW_OPTIONS`をUIドロップダウン用の2件固定配列として維持し、`full_length`を含めるとコンパイルエラーになる構成）とテスト（Retakeフォームに`full_length`を渡そうとすると`@ts-expect-error`が効く形の型ガードテスト）の両輪で再発を防止した。
- **`buildA2vChainPayload.ts`**: Single・BatchのA2Vペイロード生成関数が`stage2_window: "full_length"`を常時・固定で送出するようにした。パラメータ化はしていない（ユーザーが選ぶ余地を作らない設計）。
- **既定パスの非退行確認**: `stage2_window`を送らない従来のA2Vリクエスト（Chainedの標準経路等）は引き続き22窓のまま変わらないことをテストで固定した。
- **挙動の拡大（意図的な仕様変更）**: 23.976fps×321フレームの1クリップA2Vは、従来のstandard窓では音声再構成の不整合により422で拒否されていたが、全長窓（タイル1枚のため整合検証自体が不要になる）では受理されるようになった。この変化はテストに明示的に記録した。
- **検証**: `npm run typecheck`クリーン、vitest **2261 passed / 10 skipped**（既存skip）、lintクリーン。追加テストの要点は、481フレームでタイル数1・つなぎ目リスト空になる幾何の固定、短尺（121f・49f）での退化挙動、多タイル入力を要求した場合のエラー、IC-LoRA参照動画併用時の非破壊確認、mockを使ったE2E（本物の契約経路を通して`metadata.json`を検証）。
- **デプロイ**: `build.ps1 -Config Release` → `deploy.ps1 -Config Release`が成功し、実機（`D:\For_Videos\AviUtl2\aviutl2_v2.0.54\data\Plugin\NzVideomni\NzVideomni.aux2`）とバックエンドリポジトリ配布コピーの2箇所へ配置した。SHA-256は3値一致: `4C9B0EE915AFA7F5B82AEFBF8897A5D74EE05263178A834FF79A01E9A5227B2C`（2026-08-12のセンタリングのデプロイで更新済み、さらに同日の快適上限マーカーのデプロイで更新。現行値は本ファイル§76）。
- **オーナー実機ゲートの結果 — テーマ完結（2026-08-12）**: **G-B1〜G-B4は全合格**（完走・`chain.v_tile=61`／`n_tiles=1`・VRAMが単発生成と同等）。**G-B5・G-B6（つなぎ目の目視比較）はG-B1の合格をもって推定合格**とした——「潜在22窓の時代からつなぎ目はほとんど判別できなかった」というオーナー所見のとおり、目視比較そのものに判定力が無いためである。**任意G-B7（12fps×481フレームでの音声破綻の観測）は「想定していない使用方法」として未確認のままクローズ**した。あわせて`config.yaml`の`spill_free_frames`テーブルの更新は不要と確定した。合格記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-80へ移し、台帳の受け皿だった[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-1は見出しごと削除した。
- **状態**: **完結（2026-08-12）**。実装・機械検証・デプロイ・オーナー実機ゲートすべて完了。**2026-08-12にコミット済み**（両リポジトリ）。

---

## 75. バッチa2vのスキップ上限の調査（改修不要でクローズ）と、Outpaintingのセンタリング（2026-08-12）

**やったこと**: 同日の§74（全長stage-2化）に続くセッションで、2つのテーマを扱った。1つ目のバッチa2v（audio to video、音声から動画を生成する機能）のスキップ上限は、**調査の結果すでに実装済みと判明し、何も実装せずクローズ**した。2つ目のOutpainting（動画のキャンバス拡張）のセンタリングは、上下左右に描き足す量を対称に保つ入力補助として実装・デプロイ・コミットし、**同日のオーナー実機ゲートG-C1〜G-C5に全項目合格してテーマ完結**した（記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-81）。センタリングの仕様の正本は[`OUTPAINTING_DESIGN_NOTES.md`](OUTPAINTING_DESIGN_NOTES.md) §5-Dであり、本節は重複を避けて経緯・レビューの教訓・検証だけを記録する。

### 75.1 バッチa2vのスキップ上限をDURATION連動にする改修 — 調査の結果、不要と判明してクローズ

- **発案**: オーナーから「バッチa2vのスキップ判定が481フレーム固定になっているので、SingleタブのDURATION（生成するフレーム数）の値に連動させたい」という改修案が出た。
- **調査結果——フロントエンドでは2026-07-19に実装済みだった**。コミット`9324f05`（「feat: バッチA2VのSkip上限をCreateタブのDURATIONスライダーに連動」）で、**スキャン時の`scanToRows`と、Start押下時に全行を再判定する`rejudgeRows`の両経路**が`generationValues.numFrames`（SingleタブのDURATION値）を上限として受け取る形になっている（`webui/src/modes/batch/manifestMerge.ts`・`webui/src/modes/batch/useBatchForm.ts`）。実効の上限は`Math.min(渡された上限, 481)`であり、481はサーバー絶対上限（`api/models.py`）へのクランプとしてのみ残っている。スキップ理由のキーも`over-cap`（上限超過）であって、フレーム数を名前に埋め込んだものではない。
- **オーナー決定——改修は不要**。フロントエンドは既に望みどおりの挙動であるため、本セッションでは実装を行わなかった。
- **バックエンド同梱Gradio GUIへの追随は見送り**: バックエンドの`gradio_ui/manifest.py`は`MAX_FRAMES = 481`固定の`over_frame_limit()`で判定しており、DURATION連動になっていない。製品の入口はフロントエンドであるため優先度が低いというオーナー判断で見送った。将来のセッションが両者の挙動差を見て混乱しないよう、台帳[`PENDING_TASKS.md`](PENDING_TASKS.md) §4-29（スコープ外）へ起票してある。
- **記述の誤りを1件訂正した**: バックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §58.9-3が「Batchタブのスキップ機能は解像度を見ない一律481フレーム上限」と書いていたのを、実際の「上限＝SingleタブのDURATION値と481の小さいほう」へ直した。「解像度を見ない」「全長化との数値整合は取れている」という論旨自体は正しいのでそのまま残してある。

### 75.2 Outpaintingのセンタリング（描き足す量を対称に保つチェックボックス）

**仕様・設計判断の正本は[`OUTPAINTING_DESIGN_NOTES.md`](OUTPAINTING_DESIGN_NOTES.md) §5-D**である。以下は実装側の記録に絞る。

- **足したもの**: パッド（描き足す量）のスライダー4本の**上**、見出し「描き足す量」の直下にチェックボックスを1個置いた。入にしている間だけ上下・左右が連動し、片辺を動かすと対辺に同じ値が入る。切→入の瞬間には各軸の合計を2で割った値（**切り捨て**）を両辺へ入れて等分する。既定は切で、切の間の挙動は改修前と1ビットも変わらない。ラベルは日本語「センタリング」／英語 "Keep centered"。**生成リクエストには一切入らない**（送られるのはミラーの結果であるパッド4値だけで、バックエンドは無改修）。
- **[`OUTPAINTING_DESIGN_NOTES.md`](OUTPAINTING_DESIGN_NOTES.md) §5-Cで廃止した「対辺連動」の復活ではない**: あちらが廃止したのは片辺の値が対辺の**上限**を動かす仕組み（常時働き、切れない）であった。今回入れたのは明示的なオプトインで、動くのは上限ではなく値である。上限の定数2本（`PAD_SLIDER_MAX = 220`・`MAX_PAD = 4096`）は不変のままである。
- **実装ファイル**: `webui/src/modes/edit/outpaintGeometry.ts`（純関数`centerPads`＝切→入の等分。冪等かつ入力非破壊）、`webui/src/modes/edit/useOutpaintForm.ts`（`centering`の状態。`setPad`は入のとき1回のupdaterで対辺2値を同時に書くので、非対称な中間状態も`useEffect`による相互監視も存在しない）、`webui/src/modes/edit/OutpaintingPanel.tsx`（チェックボックスの描画）、`webui/src/i18n/strings.ts`（日英の文言）。
- **敵対的レビュー2巡で捕まえた実バグ1件——`field field-inline`**: プラン段階と実装差分の2回に分けてレビューを行い、実装差分のレビューで**チェックボックスの`className`が`field-inline`単独になっており、`display:flex`が付かずレイアウトが不発になる**欠陥を見つけて`field field-inline`へ直した。リポジトリの定型は**2クラスを並べる**書き方であり、`field-inline`だけでは横並びにならない。**JSDOM（自動テストの仮想DOM）は実際のCSSを解決しないため、この種の欠陥はテストでは絶対に捕まらない**——CSSクラスの綴りは差分レビューで人の目が確かめるしかない、という教訓として記録する。
- **テストの増分**: 17本追加した（`outpaintGeometry.test.ts`＝`centerPads`の等分・切り捨て・冪等性・非破壊、`OutpaintingPanel.test.tsx`＝チェックボックスの既定値・位置・ミラー動作）。
- **検証**: `npm run typecheck`**合格**、`npm test`**2278 passed / 10 skipped・失敗0**（§74時点の2261から+17）、`npm run lint`**31 warning・0 error**（ベースライン維持）。
- **既知の帰結（すべて[`OUTPAINTING_DESIGN_NOTES.md`](OUTPAINTING_DESIGN_NOTES.md) §5-Dに記録済み・オーナー確認済みの許容事項）**: ①片軸の合計が1のときは入にすると0/0になり、四辺すべてが0なら既存の`padsZero`が生成を止める ②幅または高さが奇数の元動画は、入のままでは128の格子に絶対に乗らない（入の間の1軸の増分は必ず偶数になるため）③右クリックからEdit系のルートへ入り直すと`remountTokens.edit`で画面ごと作り直され、チェックは既定の切に戻る ④入の間に数値ボックスを空にすると、対辺も0になる。いずれも例外ルールを増やさないという[`OUTPAINTING_DESIGN_NOTES.md`](OUTPAINTING_DESIGN_NOTES.md) §5-Cの判断と同じ線で、専用の案内も特例も設けていない。
- **オーナー実機ゲートは全項目合格・テーマ完結（2026-08-12）**: G-C1〜G-C5の5項目すべてに合格した。**G-C1では、実装差分のレビューで直した`field field-inline`の横並びレイアウト（チェックボックスとラベルが1行に収まること）も実機で確認済み**である——自動テストでは捕まえられない類の欠陥だったので、この確認をもって上記の教訓が実地で裏づけられたことになる。残る4項目（切→入の等分・スライダーと数値ボックス双方からのミラー・入→切で値が残ること・寸法の異なる素材へ差し替えたときのパッド0復帰とチェック維持）も想定どおりの挙動だった。これによりセンタリングのテーマは完結し、台帳の受け皿だった[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-2は見出しごと削除して[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-81へ合格記録を移した。
- **状態**: **完結（2026-08-12）**。実装・敵対的レビュー・機械検証・デプロイ・コミット・オーナー実機ゲートすべて完了。**デプロイ**: `build.ps1 -Config Release` → `deploy.ps1 -Config Release`が成功し、実機（`D:\For_Videos\AviUtl2\aviutl2_v2.0.54\data\Plugin\NzVideomni\NzVideomni.aux2`）とバックエンドリポジトリ配布コピーの2箇所へ配置した。SHA-256は3値一致（ビルド成果物・実機・配布コピー）: `E13EC924E6905678A8B0EDAFF9493985EB55E499FD97703481164DB75174D8FC`（2026-08-12の快適上限マーカーのデプロイで更新。現行値は本ファイル§76）。

---

## 76. Chainedの快適上限マーカー（解像度スライダーに「ここまでが快適」の線を引く）— §1-20実装完了（2026-08-12）

**やったこと**: Chainedタブの幅・高さスライダーには目安が何も無く、いま選んでいる仕上げ工程（stage-2＝アップスケール工程）の長さで**どこまでの解像度なら快適に生成できるのか**が分からなかった。判断材料そのものは既に存在していた——バックエンドの`CHAIN_COMFORT_TOKEN_BUDGET = 40,000`（1回の仕上げ工程が快適に扱える注意トークン数の上限。トークン＝（幅÷32）×（高さ÷32）×潜在フレーム数）で、超過時の警告文としては§1-14で既に使っている。今回はこの線をスライダーの上に**見える形**で描いた。実測（1792×1024の3Run）と実装記録の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §59、研究ノートとしての位置づけは同[`CHAIN_STAGE2_RESEARCH_NOTES.md`](../../../Docs/CHAIN_STAGE2_RESEARCH_NOTES.md) §9であり、本節はフロントエンド側の記録である。

- **白い線と赤い線は別の意味を持つ（この区別が本テーマの肝）**: 予算はトークン数、つまり幅と高さの**積**に効くので、「快適な幅の上限」は単独では決まらない。そこで線を2種類にした。**白い線＝16:9に近い形で釣り合う推奨点**（潜在22フレームなら1792×1024、潜在19フレームなら1920×1088）で、**その軸単独の絶対上限ではない**——高さを下げれば幅は白い線より上でも予算内に収まる。**赤い線＝もう一方の現在値を前提に、この軸が収まる最大値**で、16:9ではなく実際の入力値から逆算する。赤は「相手の軸が自分の白い線を**超えた**とき」にだけ現れる（**超えた、であって達したではない**——推奨点にぴったり合わせている状態は快適そのものなので、そこに赤を出しては意味が反転する）。tooltipの文言もこの区別を守って書き分けた。
- **赤が現在値より上に出る場合も、そのまま出す**: 幅だけが白い線を超えていて総トークンはまだ予算内、というときは、高さ側の赤い線が現在の高さより**上**に出る。例外規則を設けて隠すことはしていない。文言を「もう一方の現在値のままで、この項目が快適な範囲に収まる上限」としてあるので、意味の齟齬は起きない。
- **警告文を2文に割り、両方の仕上げ工程で出すようにした**: 従来は「潜在19フレームに切り替えれば解消する場合＝潜在22フレームのときだけ」表示していた（[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-68由来の旧仕様）。**2026-08-12のオーナー決定でこれを反転**した——潜在19フレームでも予算超過は普通に起こり、そのとき黙っているのは「重くなる」という事実を隠すことになるためである。事実（「VRAM溢れにより生成が遅くなる可能性があります。」）は常に、誘導（「潜在19フレームで軽減できる可能性があります。」）は効き目のある潜在22フレームのときだけ、空白1つで連結して見せる。**Generateはブロックしない**（§1-14の既存原則）。`ChainedScreen.tsx`の旧仕様の根拠コメントも同じ内容へ書き換えてある。
- **閾値のハードコードは禁止した**: 40,000はバックエンドの`GET /config`が`limits.chain_comfort_token_budget`として配信する値を使う（[`API_REFERENCE.md`](API_REFERENCE.md) §3.2）。フロントは必ず`resolveChainComfortBudget`を通して読み、欠落・0・負・NaNのときだけミラー定数40,000へ落ちる。VRAM容量の異なる機体では正しい値も変わるため、`config.yaml`を書き換えてサーバーを再起動するだけで線が動くようにした。
- **色はテーマ追随。リテラルの白は使わない**: `--marker-comfort: var(--text)`を`:root`にだけ置き、ダーク（既定）ではほぼ白、ライトでは黒い線になるようにした。赤は両テーマとも`--danger`である。
- **到達できない位置には線を引かない**: 参照動画を添付すると幅・高さが128の倍数へ制約される（§1-15）ので、そのときは線も128の倍数へ**切り捨てる**（潜在19フレームの高さ1088→1024）。切り捨ては常に安全側にしか動かない。
- **実装ファイル**: `webui/src/shell/tokenBudget.ts`（純関数3本`chainComfortAxisMax`／`chainComfortSize16x9`／`chainWindowBudgetMarkers`と`resolveChainComfortBudget`を新設。`isChainWindowOverBudget`は末尾に予算の省略可能引数を足した）、`webui/src/modes/chained/useChainForm.ts`（予算の解決と`chainWindowMarkers`の公開。マーカーは**状態に持たず全て派生値**なので、仕上げ工程の切替・予算の変更・参照動画の着脱に自動追随する）、`webui/src/modes/single/CommonGenerationFields.tsx`（レンジ入力を`SizeSlider`へ切り出し、絶対配置の縦線を重ねる）、`webui/src/modes/chained/ChainedScreen.tsx`（配線とバナー統合）、`webui/src/index.css`・`webui/src/modes/single/SingleScreen.css`（色と位置の定義）、`webui/src/api/types.ts`・`webui/src/modes/single/defaultConfig.ts`・`webui/src/i18n/strings.ts`。
- **共有部品への無影響**: `budgetMarkers`は**省略可能なプロパティ**（`onGetSize`の前例と同じ方式）で、渡さないCreate・RetakeはDOMも見た目も一切変わらない。線を描かない側では包みの`<span>`が1段増えるだけで、`<label>`とスライダーの結び付き（`getByRole("slider", {name})`で引けること）も維持されていることをテストで固定した。
- **`chain_math.py`に対応物が無いことを明記した**: 新設の純関数3本は「スライダーのどこに線を引くか」というUI固有の問いに答えるもので、バックエンドは同じ問いを持たない。ミラー文化（`tokenBudget.ts`は`chain_math.py`の逐語的な写しであるという既存の規律）に反する差分と誤解されないよう、コメントで**ミラーの欠落ではない**と明示してある。
- **テストの増分**: 37本追加した（実装時点で34本〔`tokenBudget.test.ts`17本＝推奨点の対・隣接64ステップが全て超過であること・自己整合の不変条件（**64グリッド限定**。128グリッドでは成り立たないので固定しない）・逆算の実例・64の倍数性・予算可変・32px未満の防御・`resolveChainComfortBudget`のフォールバック4種・128グリッドでの切り捨て、および**赤い線の境界のピン留め**〔幅1792→赤なし／1856→赤あり、高さ側も対称〕、`stage2Window.test.tsx`11本＝配信予算の差し替え〔画面は`useConfig()`から設定を取るため差し替え口が無く、**フック経由でしか書けない**〕・線の値・仕上げ工程の切替追随・赤い線の出現・バナーの2文と1文・Generateが押せたままであること、`CommonGenerationFields.test.tsx`6本＝渡さない画面に線が出ないこと・範囲外の線を描かないこと・上端ちょうどの線は描くこと・DOM順・`aria-hidden`。**すべて本テーマの追加**（17+11+6=34。従来記載の「うち30本」は誤記だった）〕、2026-08-12の敵対的レビュー後の追補で3本〔`CommonGenerationFields.test.tsx`に1本＝マーカーの`title`属性が`comfortMarkerTitle`/`limitMarkerTitle`の文言を持つことの表明（A-1。`pointer-events: none`を外し当たり判定を12pxへ広げた変更の対。**A-1当時12px、現在は11px**。§76.1）、`bridge/mockBridge.test.ts`に1本（新規）＝`MOCK_CONFIG_BODY.limits`が`FALLBACK_APP_CONFIG.limits`の全鍵を持つことのパリティ守衛（B-1。`chain_comfort_token_budget`欠落の再発防止）、`stage2Window.test.tsx`に1本＝`high_resolution`側でも予算超過時にバナーが1文（誘導なし）のままであることの描画確認（B-2）〕）。
- **モック環境の上限は1920×1088で、潜在19フレームの推奨点と完全に一致する**。線の描画条件を**両端を含む閉区間**（`min <= 値 <= max`）にしてあるのはこのためで、`>= max`で除外する実装にすると推奨点が消えてテストごと落ちる。実機の設定は上限4096なのでスライダーの範囲は変わらず、線だけが動く。同じ理由で、B-2のテストは`high_resolution`かつ予算超過という組み合わせを再現するのに幅を上限（1920）超えの4096まで自由入力させており、そのぶんGenerateは（予算とは無関係な）範囲外エラーで無効化される——検証しているのは「予算超過そのものはvalidityReasonsを増やさない」という不変条件のレンダリング側であって、Generateの有効/無効ではない。
- **検証**: `npm run typecheck`**合格**、`npm test`**2315 passed / 10 skipped・失敗0**（§75.2時点の2278から+37、**すべて本テーマの追加**〔うち3本は2026-08-12の敵対的レビュー後の追補〕）、`npm run lint`**31 warning・0 error**（ベースライン維持）。バックエンドは`pytest tests/test_stage2_window.py`**84 passed / 7 skipped**（Docs 2ファイルのみの追補で再確認・不変）、全件`pytest`**1188 passed / 20 skipped**（実装段階の実測。今回の追補はバックエンドのコード・テストを一切変更していないため未変更のはず）。
- **状態（初回デプロイ時点）**: 実装・機械検証・**デプロイ・コミット済み**。**デプロイ**: `build.ps1 -Config Release` → `deploy.ps1 -Config Release`が成功し、実機（`D:\For_Videos\AviUtl2\aviutl2_v2.0.54\data\Plugin\NzVideomni\NzVideomni.aux2`）とバックエンドリポジトリ配布コピーの2箇所へ配置した。SHA-256は3値一致（ビルド成果物・実機・配布コピー）: `00036A8CB32358377635BA720859788E75E38A4C56BC2A75F333958A879ED00E`（2026-08-12の本テーマのデプロイで更新）。
- **状態（太さ修正版）**: 下記§76.1の太さ修正を入れて同じ手順（`build.ps1 -Config Release` → `deploy.ps1 -Config Release`）で再デプロイし、SHA-256は`A5492D40146C745FD25B8FBFACE3711D78E16B892303D1B700B61BC53AB04352`の3値一致（ビルド成果物・実機・バックエンドリポジトリ配布コピー）になった。上の`00036A8C...`は初回デプロイ時点の歴史値である。**なお2026-08-16のEnd source（§77）のデプロイでこの値も更新され、現在実機に入っているのは`760425DF...`である**（太さ修正はそこに含まれているので、G-D1の再確認はそのまま実施できる）。

### 76.1 オーナー目視ゲートG-D1〜G-D10の結果と、それを受けた修正（2026-08-12）

**結論から言うと、10項目のうち不合格は「線の太さ」1点だけだった。** 位置・色・追随・逆算・警告文・ツールチップはすべて合格である。

- **合格した項目**: G-D2（仕上げ工程の切替に線が追随）・G-D3（赤い線の逆算）・G-D4（警告文の2文と1文の出し分け、およびGenerateが押せたままであること）・G-D6（明暗どちらのテーマでも白い線が見えること）・G-D7（参照動画添付時に128の倍数へ切り下がること）・G-D8（バックエンド停止時にフォールバックの40,000ベースで線が出ること）・G-D10（ツールチップの表示）。**G-D9（`config.yaml`の値を30000／60000へ変えて再起動すると線が動くこと）は未検証のまま合格扱いでクローズ**した。
- **G-D1・G-D1-2は「位置は合格・太さが不合格」**。線がスライダーのつまみとぴったり重なること、赤い線が出ること、その状態で警告文が出ないことはいずれも確認できたが、**線が線ではなく「帯」に見える**という指摘だった。要求は「Singleタブのフレーム数スライダーに出ているネイティブの目盛りと同じ太さにすること」である。
- **原因はCSSショートハンドによる`background-clip`の巻き戻しだった（実バグ）**。`.size-marker`は当たり判定を広げるために「実幅12px＝左右5pxの透明パディング＋中央2pxの可視線」という構造で、`background-clip: content-box`によって中央のcontent-boxにだけ色を塗る設計だった。ところが色を与える`.size-marker-comfort`／`.size-marker-limit`が`background:`**ショートハンド**を使っていたため、CSSの仕様どおり「書かなかったサブプロパティは初期値へ戻る」が働き、`background-clip`が`border-box`へリセットされていた。両クラスは`.size-marker`と同じ詳細度なので、ソース上あとに来るこちらが勝ち、**透明パディングも含めた12px全体が塗られていた**——これが帯の正体である。修正は`background-color:`への変更で、これだけで設計どおりcontent-boxにしか塗られなくなる。**この落とし穴は再発しやすいので、CSSに警告コメントを残した。**
- **あわせて可視線を2px → 1pxへ細くした**。要素の実幅を12px→11px、パディングは左右5pxのまま据え置いたので、content-boxが1pxになる。総幅が奇数になったぶん`translateX(-50%)`の中心が半ピクセルへ落ちてアンチエイリアスがかかるが、**Singleタブのネイティブ`<datalist>`の目盛りも同じく細く柔らかい描画なので、狙いどおり見た目が揃う**。当たり判定（ツールチップのホバー領域）は11px（12pxから1px縮小）で、G-D10の合格を壊す幅ではない。
- **G-D5（生成中に線が減光すること）は仕様ごと削除した**。オーナーの判定は「減光がよく分からないし、不要」である。**Singleタブのフレーム数マーカーにはもともと減光の仕様が無く、Chained側だけに入れていたのが不揃いだった**ので、`.size-slider-wrap input[type="range"]:disabled ~ .size-marker { opacity: 0.4; }`のルールを丸ごと削除して両タブの仕様を揃えた。関連コメントも同時に整理してある。減光を検証しているテストは存在しなかった（マーカーのテストは`data-`属性・`title`・`aria-hidden`しか見ておらず、いずれも見た目に依存しない）ため、テストの削除は発生していない。
- **この修正の検証**: `npm run typecheck`**合格**、`npm test`**2315 passed / 10 skipped・失敗0**（修正前と同数——CSSと注釈だけの変更なので当然の結果であり、テストが見た目に依存していないことの裏づけでもある）、`npm run lint`**31 warning / 0 error**（ベースライン維持）。なお`npm test`のフル実行では`src/App.prefill.test.tsx`の音声抽出待ちがまれにタイムアウトすることがある（同ファイル単独実行では全緑。本テーマとは無関係の並列実行時フレーク）。
- **残件**: **G-D1・G-D1-2の「太さだけ」を見直す再確認1件のみ**（台帳[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-3）。太さ修正版は2026-08-12に再デプロイ済みで、実機に入っているのは§76の「状態（太さ修正版）」のSHA-256（`760425DFF6AA25C4C33042D35EA377165717C747A5FD5CD9A654DA4D2FA7EC22`）である。オーナーはAviUtl2を起動してそのまま確認できる。

---

## 77. End source（素材（末尾））— 添付した画像・動画で終わる動画を作る（2026-08-15〜16）

**その後（2026-08-17・§79）**: 窓内モードへ作り替え、本節の記述は旧方式の記録になった。

**やったこと**: Chainedタブに「素材（末尾）」のカードを新設し、**添付した画像・動画で終わる動画**を生成できるようにした。既存の素材スロットは対になるよう「素材（冒頭）」（"Start source"）へ改名した。両方を添付すれば冒頭と末尾を与えた補間になる。バックエンド側の設計（内部区画・またがり凍結・帯長の自動決定）と機械ゲートの正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §60、幾何の研究ノートは同[`CHAIN_STAGE2_RESEARCH_NOTES.md`](../../../Docs/CHAIN_STAGE2_RESEARCH_NOTES.md) §10であり、本節はフロントエンド側の記録である。台帳の実装記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-82。

- **出力は「クリップ合計＋帯」に伸びる（この一点が第1版からの最大の違い）**: クリップのフレーム数は**新規生成分そのもの**であり、素材になる区間（帯）はその外側にバックエンドが継ぎ足す。第1版はクリップの内側に帯を埋め込む方式で、クリップを自動で押し上げるeffectまで持っていたが、v2ではその押し上げごと削除した。`form.outputFrames`は`computeOutputFrames(...)`に帯を足すだけになり、これがタイムラインの仮オブジェクトの尺にも`bindToJob`にも自動で効く。
- **帯の長さはstateではなく導出値にした**: サーバーがアップロード応答で返す実測値（`frame_count`／`fps`）→生成フレームレートへの保守換算（一致なら恒等、不一致なら切り下げ−1）→8切り下げ→136クランプ、という**一方向の計算**で、どこにも「ユーザーが決めた帯長」という状態を持たない。**実測を上回る帯長は原理的に出せない**。画像は8フレーム固定である。スライダーUI一式と、それに付いていた検証（`snapEndContextFrames`／`isEndContextFramesValid`／`endContextFitsChain`／`endContextRequiredLastClipFrames`と3件のブロック理由）、および`tokenBudget.ts`の`stage2MaxEndContextFrames`は削除した。
- **長さ・のりしろに関するブロック理由は3件に整理した**: `endSourceTooShort`（素材の動画が25フレーム＝約1秒未満）・`endSourceLengthUnknown`（長さが取れなかった。**黙って422を食わせない**ためのフォールバックの最終段）・`endSourceNeedsOverlap`（のりしろ1では使えない。2以上へ上げてもらう）。既存の排他（a2v・IC-LoRA参照動画）とRetakeの排他はそのまま維持している。
- **予想出力は内訳で見せる**: 「予想出力: ≈23.7秒（569フレーム）・うち末尾3.0秒（72フレーム）は素材」。帯が無いときは従来の表示のままで、日英とも新設の文言関数に集約した。
- **`estimateGenerationSeconds`には呼び出しの引数にだけ帯を足した**。`totalFrames`本体に足すと、クリップ合計の妥当性検査（`isTotalFramesValid`）が壊れるためである。副作用として、バッチi2v-longのサマリー表示に出る見積りは帯込みのままになるが、素材（末尾）を添付している間はバッチ自体がブロックされる（`endSourceAttached`）ので実害はない——既知の表示上の癖として記録しておく。
- **仮オブジェクトの「末尾合わせ」（系統E）はC++を触らずに実現した**: 新設した`timeline/tailAlign.ts`が「開始＝素材の開始−(出力長−帯長)」を計算し（ネイティブの四捨五入を逐語ミラー、負は0クランプ、フレームレート不明時は頭揃えへ退避）、`menuRouting.ts`の`MenuPlacement`に`"E"`を足したうえで、`provisionalReservation.ts`が**ネイティブへは`"B"`（頭揃え）としてシフト済みの座標を渡す**。Retake（系統D）で確立した規律と同じで、**ネイティブ側に新しい配置系統は存在しない**。予約時はベストエフォートの見積り（画像8／動画はリボン区間の秒数から同じ式／不明なら長め安全側の136）、Generate時に確定値で打ち直す。
- **素材の位置アンカーは`sourceLocationMap`へ相乗りさせず、`ChainedScreen`のローカルrefに持たせた**: 打ち直しのたびに新しい予約IDが採番されるため共有マップ側ではアンカーが孤児化すること、Join機能との合成事故のリスクがあることの2点を検証で確認したうえでの判断である。`handleGenerate`をasync化し、**席が予約済み・ローカルにアンカーあり・素材（末尾）ありの3条件が揃ったときだけ**確定値で打ち直してから送信する（手動のGenerateではアンカーが無いのでスキップ——誤って新規の仮オブジェクトを挿し込まない）。async化に伴う連打対策として`inFlightRef`のガードを1つ足した。
- **右クリック「これで終わる動画を作る」はクリップ1本で開く**（`forceSingleClip`）。v2vの「続きを生成」と同じ挙動で、予約時の尺の見積りが正確になり、Generate時の打ち直しでの跳ねが最小になる。素材の使用区間はリボンのトリム区間を尊重する（既存の`decideSourceTrim`）。右クリックで拾える素材の最低長は`25/24`秒（約1秒）に変更した。
- **バッチi2v-longでは素材（末尾）を剥がす**: テンプレート合成に`withoutEndSource()`を追加し、添付中はブロック理由`endSourceAttached`でStartを止める。
- **主な実装ファイル**: `webui/src/modes/chained/ChainEndSourcePanel.tsx`（カード）・`useSourceUpload.ts`（`frames`／`fps`の保持）・`useChainForm.ts`（帯長の導出・ブロック理由・出力長）・`ChainedScreen.tsx`（アンカー保存とGenerate打ち直し）、`webui/src/timeline/tailAlign.ts`（新設）・`menuRouting.ts`・`menuSelection.ts`・`provisionalReservation.ts`、`webui/src/shell/AppShell.tsx`（Step 8の`"E"`分岐）、`webui/src/modes/batch-i2v-long/buildI2vLongPayload.ts`、`webui/src/api/types.ts`・`webui/src/bridge/types.ts`（`UploadVideoResponse`の`frame_count`／`fps`）、`webui/src/i18n/strings.ts`。
- **付随して直した既存の問題2件（フロントエンド側）**: ①**仮オブジェクトの予約席が移動すると、V2Vの素材位置メモが孤児になっていた**——`provisionalReservation.ts`の移動処理は新しい予約IDを採番するのに、`sourceLocationMap`のメモを引き継いでいなかったため、Join（V2V結合）の挿入位置が「位置なし」へ退化していた。RPCが成功したあとで`rekeySourceLocation`を呼ぶようにして解消した（拒否された移動ではメモが現行の予約の下に残る）。②同モジュールのコメントが「移動時は同じ予約IDで再記録する」と実装と食い違っていたので、①の修正後の実挙動に合わせて書き直した。記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-83。
- **検証**: `npm run typecheck`**0エラー**、`npm test`**2442 passed / 10 skipped・失敗0**（§76時点の2315から+127）、`npm run lint`**31 warning / 0 error**（ベースライン維持）。バックエンドは`pytest`**1455 passed / 20 skipped・失敗0**（§76時点の1188から+267）。実GPUの機械ゲート8ジョブはバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §60.6。
- **状態**: 実装・機械検証・**デプロイ済み**（2026-08-16）。`build.ps1 -Config Release` → `deploy.ps1 -Config Release`が成功し、実機（`D:\For_Videos\AviUtl2\aviutl2_v2.0.54\data\Plugin\NzVideomni\NzVideomni.aux2`）とバックエンドリポジトリ配布コピーの2箇所へ配置した。SHA-256は3値一致: `760425DFF6AA25C4C33042D35EA377165717C747A5FD5CD9A654DA4D2FA7EC22`。**右クリックの新項目を足したので`native/src/plugin.cpp`（オブジェクトメニュー14→15項目）と言語ファイル2本は変更している**が、**配置の系統Eはネイティブ無改修**である（既存の`"B"`への写像）。ブリッジ契約のバージョンも不変（`action`に`endWithThis`が増えただけ）。残件は**オーナーの目視ゲートG-E1〜G-E7**（[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-4）だけである。

## 78. End sourceのUI非露出化・文言4件の調整・ICCプロファイル起因のバグ修正（2026-08-16）

**その後（2026-08-17・§79）**: 窓内モードへ作り替え、本節の記述は旧方式の記録になった。

**やったこと**: オーナー目視ゲートの結果を受けて、**End source（素材（末尾））をユーザーからの到達経路ごと塞いだ**。裁定の理由（現状ではほぼ確実にクロスフェードの動画になるため実用性が低い／将来の改修に備えて撤去はしない）の正本は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-82で、本節は作業側の記録である。あわせて、目視ゲートG-E5の失敗原因だった既存バグの修正と、オーナー指定の文言4件の調整を同じバッチで行った。

- **塞いだのは2箇所だけである**: `ChainedScreen.tsx`の「素材（末尾）」カード（アコーディオンの`<details>`ブロック）と、`native/src/plugin.cpp`のオブジェクトメニュー項目「これで終わる動画を作る」（15→14項目。**C++の再ビルドを伴う**）。**裏の状態管理・バッチのブロック・予想出力の内訳表示・ブロック理由はいずれも「素材（末尾）が添付されている」ことを条件にしているため、入口が消えれば条件不成立で自然に休眠する**（実装を1件ずつ追って確認済み）。Gradio・MCPには元から露出が無く、変更していない。
- **温存したもの**: `ChainEndSourcePanel.tsx`（未マウントになるだけ）・`useChainForm`のend source系ロジック一式・`timeline/tailAlign.ts`・`menuRouting.ts`／`menuSelection.ts`のendWithThis系エントリ（ネイティブがactionを発しないので死経路になる）・フォールバック関数`OnMenu_EndWithThis`・言語ファイルの訳・i18nのendSourceキー群・バックエンドの実装一式。`endSourceAccordionRef`とその自動展開effectも、refがnullのまま無害なので残してある。
- **テストは「非露出であることの検証」へ書き換えた**: `ChainedScreen.endSource.test.tsx`はChainedScreen全体を描画してカードの実在と右クリックintentでの自動投入を検証していたため、JSXを消せば全件落ちる。**カードのDOM要素が存在しないこと・`end-with-this`のintentを注入しても何も現れないこと・予想出力が通常の1行表示のままであること**を見るテストへ全面的に置き換えた。`ChainEndSourcePanel.test.tsx`（コンポーネントを直接描画する）は非露出化の影響を受けないのでそのまま残る。
- **API消費者向けの注意書きを4箇所に置いた**: MCPサーバーの`INSTRUCTIONS`・`submit_chain`のdocstring・`api/models.py`の`EndSourceSpec`のdocstring・[`API_REFERENCE.md`](API_REFERENCE.md) §5.2。**APIには`end_source`が残っている**ので、UIから消しただけではAIエージェント経由で使われうるためである。
- **ICCプロファイル起因のバグ（目視ゲートG-E5の失敗原因）は一般バグとして直した**: 入力PNGのICCプロファイルが出力mp4のside dataとして残り、`services/video_io.py`の`frame_count()`が使うffprobeのCSV書式（`csv=p=0`）が空フィールドを付けて`9,`のような文字列を返すため、フレーム数の解析に失敗していた。出力形式を`default=nokey=1:noprint_wrappers=1`へ改め、ICCプロファイル入りの素材を使う回帰テストを追加した。**ICCプロファイルは再エンコードを越えて残るので、同じ地雷はアップロード時のフレーム数計測など`frame_count()`を使う他の経路にもあった**——end source固有の問題ではない。詳細はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §60.13。
- **文言4件（`strings.ts`。日英とも）**: 素材（冒頭）の説明を「入力された素材（動画 or 画像）に続く動画を生成します。」とし、素材（末尾）側の3件（説明・音声モードの注記「映像＋音声は未実装です。」・帯長のヒント）も対になる表現へ揃えた。**素材（末尾）側は非露出化により当面ユーザーの目に触れない**が、コード温存の方針に合わせて指定どおり更新している。
- **台帳・文書の整理**: 目視ゲートの結果と裁定を[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-82へ追記し、本バッチを同§3-86として記録した。快適上限マーカーの再確認合格（2026-08-16）で同§3-85をクローズしたことにより、[`PENDING_TASKS.md`](PENDING_TASKS.md)の「2. 実装済み・ユーザーのテスト待ち」は空になったので節ごと削除し、実用化の研究テーマをフロントエンド[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-84として起票した。
- **状態**: 実装・デプロイ済み（2026-08-16。実機＋バックエンドリポジトリ配布コピーの2箇所）。**C++を再ビルドしているため`.aux2`の版数は§77から更新されている**。

## 79. End source（素材（末尾））のUI復活 — 「この素材へ繋がる動画を作る」機能として作り直した（2026-08-17）

**やったこと**: 前日に塞いだばかりの「素材（末尾）」を、**使える機能としてUIへ戻した**。塞いだ理由（生成結果が例外なく素材へクロスフェードしてしまう。§78）は、同じ日にバックエンドを**窓内モード**へ作り替えたことで解消している——クリップ自身の末尾8フレームを素材の冒頭で凍結し、stage-1（低解像度で動きの骨格を作る第1段階）の**1つのデノイズ窓の中に素材を同居させる**方式で、注意機構（生成時にどのフレームを参照するかを決める仕組み）から素材が全区間に見えるため、生成は最初から素材へ向かって進む。**バックエンドの改修と実機実験の記録はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §61が正本**であり（v2＝内部区画方式の歴史記録は同§60）、本節はそれに追随したフロントエンド側の記録である。

- **出力の長さが変わらなくなった（利用者から見て最大の違い）**: 凍結フレームはクリップの内側にあるので、配信される動画の長さは**クリップの設定そのもの**である。§77（v2＝帯をクリップ列の後ろへ継ぎ足す内部区画方式）で入れた「予想出力＝クリップ合計＋帯」の加算と、その内訳表示（「うち末尾3.0秒は素材」）は削除した。仮オブジェクト（生成中であることをタイムライン上に示す仮置きのオブジェクト）の尺も、これに合わせて自動的に正しくなる。
- **錨（末尾の凍結フレーム）は8フレーム固定にした**: 素材の実測長から帯の長さを導出するv2の計算式と、それに付いていた定数・関数（`END_BAND_MAX_FRAMES`ほか）は削除し、`timeline/tailAlign.ts`の`END_SOURCE_CONTEXT_FRAMES = 8`を**常に明示送信する**（サーバー既定の72は使わない）。実機比較で8が最良と分かっているためで、錨を長くすると窓の大部分を素材の再現に費やすことになり、創造性が落ちてクロスフェードが戻ってくる。
- **素材の最低長は9フレームへ下げた**（v2は25フレーム）: 読み取るのは錨の8フレームと、その手前に1枚だけ要る因果VAE（時間方向に過去だけを見る圧縮器）のプライマで、合わせて9フレームだからである。フレームレート換算を保守側へ1フレーム倒している経路では実効10フレーム相当になるが、9フレームという下限で1フレームの余裕は利用者が意図して踏める差ではないため、そのまま許容した（コメントに明記）。
- **クリップは1本だけに制限した**: 窓内モードが成立するのはクリップ1本のときだけで、2本以上ではバックエンドが旧方式（帯の継ぎ足し＝クロスフェード問題あり）へ落ちる。素材（末尾）を添付している間は**「クリップを追加」ボタンを理由つきで無効化**し、先に2本以上にしてから添付した場合は新設のブロック理由`endSourceSingleClipOnly`でGenerateを止める。**勝手にクリップを消すことはしない**（利用者の入力を黙って捨てないという既存の原則）。
- **クリップが長すぎるときは警告を出す（止めはしない）**: stage-2（アップスケール工程）は決まった長さのタイル単位で仕上げるため、クリップが**タイル1枚（標準169フレーム・高解像度145フレーム）**を超えると、凍結フレームとそこへ繋がるべきフレームが別々のタイルに分かれ、境界でにじみ・ちらつきが出る。実験で確認済みの現象なので、超過時に穏やかな警告バナーを出す。**サーバーは何も判定しないのでGenerateはブロックしない**（§1-14以来の「事実は伝える、決定は利用者」の原則）。判定式は既存の`chainUtils.pxFromVLatent`を再利用し、同じ計算の3つ目の写しを作らないようにした。
- **右クリック導線だけはクリップ長を169フレームへ丸める**: 「これで終わる動画を作る」は素材投入まで済んだ状態で画面が開くので、そのままGenerateを押しても警告に引っかからないよう、初期のクリップ長を`min(快適上限, 169)`にした（`timeline/prefillSeed.ts`の`END_SOURCE_SEED_MAX_FRAMES`）。**手で長くしたときは警告で気づいてもらう**という2026-08-17のオーナー裁定に沿った切り分けである。
- **仮オブジェクトの「末尾合わせ」（配置系統E）は引き算だけになった**: 開始位置は「素材の開始 −（出力長 − 8）」で、v2の帯長の見積り（`endSourceBandFrames`）は不要になり削除した。因果VAEのプライマ1枚に由来する±1フレームの誤差は残るが、仮オブジェクトは生成後に本物へ置き換わる座席なので許容し、由来をコメントに残した。**ネイティブ側は無改修**（従来どおり頭揃え`"B"`へシフト済み座標を渡す規律）。
- **右クリックメニューは15項目へ戻した**: `native/src/plugin.cpp`の1行復元で、**C++の再ビルドを伴う**。件数を書いたコメント2箇所も更新した。
- **テストは「非露出であることの検証」から実機能の検証へ戻した**: §78で全面的に書き換えていた`ChainedScreen.endSource.test.tsx`を復元・窓内モード仕様へ修正し、`useChainForm.test.ts`のend source節（約590行。136/72/25フレーム前提だった）も全面改稿した。追加した観点は、動画でも錨が8であること・9フレーム境界・予想出力に帯を足さないこと・クリップ追加ボタンの無効化・`endSourceSingleClipOnly`の点灯と解消・警告閾値が169⇄145で窓に連動すること・送信内容が`context_frames: 8`であること・右クリック導線のシードが169フレームへ丸まること。
- **主な変更ファイル**: `webui/src/timeline/tailAlign.ts`（v2の帯計算を削除し定数2本へ）・`prefillSeed.ts`（シードの丸め）・`menuSelection.ts`、`webui/src/modes/chained/useChainForm.ts`（1本制限・警告値・出力長）・`ChainedScreen.tsx`（カード復活）・`ChainEndSourcePanel.tsx`（警告バナー）・`generateReasonMessages.ts`、`webui/src/shell/AppShell.tsx`（系統E）、`webui/src/i18n/strings.ts`（日英同時）、`native/src/plugin.cpp`。
- **文書の追随**: [`API_REFERENCE.md`](API_REFERENCE.md) §5.2を**2モード制**（クリップ1本＝窓内モード・推奨／2本以上＝旧方式・非推奨）の表へ書き換え、v2前提のまま残っていたコード内の説明コメント（`api/types.ts`・`chainUtils.ts` 2箇所・`tokenBudget.ts`）と[`README.md`](../README.md)も現状へ揃えた。
- **検証**: `npm run typecheck`**合格**、`npm test`**2450 passed / 7 failed**、`npm run lint`**31 warning / 0 error**（ベースライン維持）。**失敗7件はすべて`src/api/backend.integration.test.ts`で、原因は本テーマとは無関係**——実バックエンドが起動しているがジョブ実行中だったため、生成を投げる項目が202ではなく409（JOB_BUSY）を受け取っている。改善案とともに台帳[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-89へ起票した。
- **状態**: 実装・機械検証・**デプロイ済み**（2026-08-17）。`build.ps1 -Config Release` → `deploy.ps1 -Config Release`が成功し、実機（`D:\For_Videos\AviUtl2\aviutl2_v2.0.54\data\Plugin\NzVideomni\NzVideomni.aux2`）とバックエンドリポジトリ配布コピーの2箇所へ配置した。SHA-256は3値一致: `192CB388C89AE579AE050C555CDB3A1ADEDBFE0D7419966D77B4283B771AC682`。**残件はオーナーの目視ゲート8項目**（メニュー15項目・カードの位置・右クリック導線でクリップ169フレーム・仮オブジェクトの末尾が素材の頭に重なること・予想出力がクリップ長のままであること・警告の点灯と消灯・1本制限・実生成で素材へ到達すること）で、[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-5へG-F1〜G-F8として起票した。
- **その後（2026-08-17）**: 目視ゲートG-F1〜G-F8が**全8項目合格**し、本バッチは完結した（台帳の§2-5は運用注記どおり見出しごと削除）。品質警告の文言は同日オーナー指示で「クリップが n フレームを超えると、生成される動画の品質が下がります。」へ簡素化した（別コミット）。
- **その後（2026-08-17・±1補正）**: オーナーの実機目視（透明度重ね確認）で、出力の末尾8フレームは素材の1〜8フレーム目ではなく**2〜9フレーム目**に一致し、素材の1フレーム目そのものは出力の第(出力長−錨)フレーム位置（早期収束による準複製）に来ることが判明。系統Eの厳密なリード値は`出力長−錨−1`が正しく、`AppShell.tsx`と`ChainedScreen.tsx`の`leadPixelFrames`を`numFrames − END_SOURCE_CONTEXT_FRAMES`から`numFrames − (END_SOURCE_CONTEXT_FRAMES + 1)`へ補正した（台帳[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-87・フロントエンド[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-84の該当項目はクローズ）。

## 80. End source第2段階・バッチ1（錨の固定強度スライダー新設＋v1音声モックの撤去）のフロントエンド実装（2026-08-18）

第2段階（逆順Chained）を2バッチに分けたうちの**バッチ1**のフロントエンド側の記録。バッチ1は「錨（素材そのもの）の凍結強度をStage-1だけソフト化できるようにする」小改修で、幾何計算・出力長の式には触れていない。バックエンドの実装・機械検証・実機ゲートの正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §63、API契約は[`API_REFERENCE.md`](API_REFERENCE.md) §5.2である。

- **強度スライダーを新設した**: `ChainEndSourcePanel.tsx`に、`ChainedScreen.tsx`の`overlapStrength`スライダーと同じ作り（`type="range"`・`min=0`・`max=1`・`step=0.05`、`field-hint`に現在値を`toFixed(2)`で表示）の強度スライダーを追加した。錨の**長さ**には従来どおり操作項目が無く（クリップ自身の末尾8フレームで固定）、今回追加したのは錨の**固定強度**——両者は別の性質で、強度は下げても最終フレームは素材どおりになる（Stage-2が常にハード凍結するため）。`useChainForm.ts`へ`endSourceStrength`/`setEndSourceStrength`のstateを`overlapStrength`の隣に追加し、`chainUtils.ts`の`clampEndSourceStrength`で`[0, 1]`にクランプする。
- **`strength`は既定値でも常に明示送信する**: `chainUtils.ts`の`buildChainRequest`が`request.end_source`を組み立てる際、`context_frames`の前例に揃えて`strength`を毎回送る。リクエストJSONを見て何が起きたか分かることを優先した判断で、`DEFAULT_END_SOURCE_STRENGTH`（1.0）はサーバー側の既定と一致させてある。
- **v1音声モックのラジオボタンを削除した**: 「動画のみ」/「動画+音声」の2択ラジオ（`chain-end-source-audio-mock`の`<fieldset disabled>`）を撤去した。フロントエンドは実装当初から音声を素材とは独立に生成しており、このラジオはどちらを選んでも挙動が変わらない死んだ選択肢だった。「音声は素材からは取り込まれず、独立して生成されます。」という注記1行（`audioModeNote`）だけは残した——backend `_encode_end_source`のdocstring（`chain_pipeline.py`）が「音声が自由生成であることはUIが明示している」と述べているため、注記を全部消すと実装の記述と食い違うことになる。
- **`strings.ts`**: `audioModeLabel`/`audioModeVideoOnly`/`audioModeVideoAndAudio`（EN/JA）を削除し、`audioModeNote`の文言を上記の1行へ差し替えた。新設した`strengthLabel`（JA「素材の固定強度」）・`strengthHint`（JA「1.00で素材のとおりに終わります。下げると素材へのなじみ方がゆるくなります（最終フレームは素材のままです）。」）を追加した。
- **テスト**: `ChainEndSourcePanel.test.tsx`に「音声ラジオが1つも存在しないこと（v1モックが完全に消えたことの回帰）」と「スライダーが`setEndSourceStrength`を呼ぶこと」の観点を追加し、既存の音声モック関連アサーションは削除した。`chainUtils.test.ts`・`useChainForm.test.ts`へ`end_source.strength`の送信・既定1.0・`[0, 1]`クランプの検査を追加した。
- **主な変更ファイル**: `webui/src/modes/chained/ChainEndSourcePanel.tsx`（強度スライダー新設・音声モック撤去）・`chainUtils.ts`（`MIN/MAX/DEFAULT_END_SOURCE_STRENGTH`・リクエスト配線）・`useChainForm.ts`（state新設）・`webui/src/i18n/strings.ts`（EN/JA同時）・`webui/src/api/types.ts`（`EndSourceSpec`のdocコメントへ`strength`追記）。
- **検証**: `npm run typecheck`合格、`npm run lint`合格、`npm test`全PASS。バックエンド側の機械検証（backend `pytest`・G1-M3モック通し）と実機ゲートG1-R1〜G1-R3はバックエンド`VERIFICATION_LOG.md` §63参照。
- **状態**: 実装・機械検証は完了。**オーナーの目視ゲート（G1-R2/R3の見え方・G1-R4のUIスライダー確認）は起床後**であり、本節の時点ではバッチ1は完結していない。
- **（2026-08-19追記）オーナー目視・試聴合格で完結。** 詳細はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §63.6、台帳の記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-84。

## 81. End source第2段階・バッチ2（逆順Chained）のフロントエンド実装 — クリップ1枚制限の撤去（2026-08-18）

第2段階（逆順Chained）を2バッチに分けたうちの**バッチ2＝本体**のフロントエンド側の記録。バックエンドが複数クリップへEnd sourceの生成方式（逆順Chained・`"reverse"`モード）を拡張したのを受け、フロントエンド側のクリップ1本制限を撤去し、既存のゲート・警告をモードごとに正しく振り分けた。バックエンドの実装・機械検証・実機ゲートM1〜M7の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §64、API契約は[`API_REFERENCE.md`](API_REFERENCE.md) §5.2である。

- **クリップ1枚制限を撤去した**: `useChainForm.ts`の`endSourceSingleClipOnly`理由コードと、それを積んでいたバリデーション箇所を削除した。`canAddClip`から`!hasEndSource`条件を外し、「クリップを追加」ボタンが素材（末尾）添付時にも押せるようにした。`ChainedScreen.tsx`のヒント文の3項分岐は`maxClipsReached`のみへ単純化し、`generateReasonMessages.ts`・`strings.ts`の`singleClipNote`等の付随コードも削除した。`api/types.ts`の`EndSourceSpec`docコメントを3モード制の説明へ更新した。
- **右クリック導線とクリップ長初期値は変更していない**: `forceSingleClip`（初期状態はクリップ1本、追加はユーザー操作）は残した。`prefillSeed.ts`の`END_SOURCE_SEED_MAX_FRAMES=169`も値はそのまま残し、コメントを「クリップ1本の窓内モードの実験結果に基づく初期値」へ訂正した——169fという数値自体は変えていない。
- **のりしろ既定値の対称遷移effectを新設した**: 「素材（末尾）×2本以上」への遷移の瞬間に1回だけ`overlapFrames`を1へ切り替え、**逆方向の遷移（1本へ戻す・素材を外す）では従来既定値（`DEFAULT_OVERLAP_FRAMES`）へ復帰させる**対称な設計にした（`useChainForm.ts`、`useRef`で直前の状態のみを追跡し、毎レンダーでは押し戻さない＝ユーザーが手で変更した値を上書きしない）。片側だけの遷移だと、2本→1本へ戻した瞬間にアプリ自身が設定した`overlapFrames=1`が窓内モードの`overlapFrames >= 2`必須ゲートに引っかかり、ユーザーが詰む問題があったための設計。
- **品質警告をクリップ1本限定にした**: クリップ長169f/145fの品質警告（`useChainForm.ts`の該当条件）へ`clips.length === 1`を追加し、複数クリップでは表示しないようにした。この警告は窓内モード（クリップ1本）の実験結果に基づくものであり、逆順Chainedには適用されないため。`strings.ts`の`qualityWarning`docコメントへその旨を明記した。
- **鏡ゲート3種を新設した**: サーバー側の新設受理検査・免除がGenerateボタンを押すまでUIに見えないのを防ぐため、`useChainForm.ts`へ3つの鏡（サーバー側検査のフロント版）を追加した。
  1. `endSourceNeedsOverlap`（`overlap_frames >= 2`必須ゲート）を`clips.length === 1`条件で窓内モード限定へ縮小し、逆順Chained（2本以上）では発火しないようにした（サーバー側の免除の鏡）。
  2. 音声のりしろ予算の鏡（`chainUtils.ts`に`sum_ka >= n_join`相当を見る純関数を新設）。既存の`chainLayoutInvalid`は`audioReady`前提のため、`end_source`×`source_audio`がAPI排他である末尾素材チェーンでは発火しない穴があり、これを塞いだ。`overlap_frames=1`はfpsやクリップ長の組み合わせによっては音声予算を割り込む構成が現実にあるため（例: 30fps×257f×2クリップ）、該当時はGenerateをブロックし理由を表示する。
  3. 最終クリップの潜在数検査の鏡（サーバー側の新設受理検査①の鏡）。既存の`clipTooShortForOverlap`理由コードを流用し、表面積を増やさないようにした。
- **出力長計算は無改修**: `chainUtils.ts`の`computeOutputFrames`はそのまま正しく、`outputTailFrames`・内訳表示も無改修で正しい。旧コメント（「複数クリップ許容時は呼び出し側にロジックを足すべき」という予告）は「結局不要だった（逆順モードの帯もクリップ合計の内側で、加算が要るのは旧方式だけでありUIはそもそも到達しない）」へ訂正した。
- **系統E（末尾合わせ）・バッチi2v-longは無改修**: `tailAlign.ts`の純関数群、`ChainedScreen.tsx`のGenerate時打ち直し、`AppShell.tsx`の右クリック時処理はいずれも複数クリップでも同じ式で正しく動くため触っていない（docコメントへその旨を追記）。バッチi2v-longの`buildI2vLongPayload.ts`も`end_source`キーを剥がすだけの実装で無改修。`useBatchI2vLongForm.test.ts`のサンプル理由コード`"endSourceSingleClipOnly"`は削除済みコードのため既存の別コードへ差し替えた。
- **テスト**: `useChainForm.test.ts`（1枚制限ケースの削除・複数クリップ追加可・のりしろ遷移effect・警告が2本以上で出ないことを追加）、`ChainedScreen.endSource.test.tsx`、`chainUtils.test.ts`（`computeOutputFrames`不変の回帰）、`useBatchI2vLongForm.test.ts`（理由コード差し替え）を更新した。
- **検証**: `npm run typecheck`合格、`npm run lint`合格、`npm test`2,480件PASS（既知7件のみ失敗、§3-89の`JOB_BUSY`起因で本テーマとは無関係）。バックエンド側の機械検証・実機ゲートM1〜M7はバックエンド`VERIFICATION_LOG.md` §64参照。
- **状態**: 実装・機械検証は完了。**オーナーの目視ゲート（V1〜V6・R2-7・R2-9〜R2-11）は起床後**であり、本節の時点ではバッチ2は完結していない。フロントエンド[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-2に目視ゲート項目を起票してある。
- **（2026-08-18追記）** 目視・試聴は同日決着。裁定と完結記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-84。
- **（2026-08-19追記）オーナー目視・試聴合格で完結。** R2-7（新旧A/B目視）を含む残る目視項目もすべて合格・確認済みとなり、End source第2段階（逆順Chained）は完結した。詳細はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §64.7、台帳の記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-84。

## 82. End source第3弾（錨への素材音声の凍結）のフロントエンド実装 — 文言のみ・ロジック無改修（2026-08-18）

第2弾までの2バッチ（§80＝錨の固定強度、§81＝逆順Chained）に続く**第3弾**は、バックエンドが「錨の素材に音声トラックがあれば常にその音声も凍結する」よう本体仕様化した改修で、フロントエンド側は**新しいstate・新しいAPIフィールドの送信・新しい分岐を一切持たない**。素材に音声があれば常に取り込まれる仕様になったため、UIに選択肢を増やす必要がそもそも無い。バックエンドの実装・機械検証・機械ゲートA1〜A9の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §65、API契約は[`API_REFERENCE.md`](API_REFERENCE.md) §5.2である。

- **`audioModeNote`の文言を書き換えた**: v1時代の「音声は素材からは取り込まれず、独立して生成されます。」から、「素材に音声があれば、その音声も末尾に取り込まれます（音声の無い素材・静止画では、音声は独立して生成されます）。」へ差し替えた（`strings.ts`、EN/JA同時）。この注記は**無条件に描画される**（`chain-end-source-audio-note`。素材の種類やstrengthの値による出し分けが無い）ため、静止画や音声の無い動画を添付した場合でも正しく読める文言にする必要がある——カッコ内の補足はそのためのもので、条件分岐を増やさずに両方の場合を1文でカバーする設計にした。
- **`strengthHint`へ音声の除外を1句追記した**: `strength`が映像専用のつまみになった（バックエンド側でstrengthが音声のマスク値に一切影響しないことが確定した）ことを受け、JA「1.00で素材のとおりに終わります。下げると素材へのなじみ方がゆるくなります（最終フレームは素材のまま・**音声は強度によらず常に素材のまま**です）。」へ、太字部分の1句を追加した（EN側も対応する1句を追記）。強度スライダーを下げても音声だけは変わらないことを、スライダーの隣で明示する狙い。
- **VAE往復による音質変化・無音素材の扱いはUIに書かない**: 錨区間の音声がVAE＋ボコーダを往復すること、無音の素材は無音のまま凍結される（無音検出をしない設計）ことは、UIの1行注記には含めず、READMEおよびバックエンド[`Videomni_Backend_Specification.md`](../../../Videomni_Backend_Specification.md)・本リポジトリ[`API_REFERENCE.md`](API_REFERENCE.md)側の説明に委ねた。UI注記は1行の単純さを保つのがオーナーの一貫した方針であり、詳細説明を全部載せるとかえって読みにくくなるため。
- **`ChainEndSourcePanel.tsx`のコメントと`ChainEndSourcePanel.test.tsx`のテスト名を改訂した**: `audioModeNote`表示部の直上コメントを「素材に音声トラックがあれば末尾へ凍結して取り込む。音声の無い素材・静止画は従来どおり自由生成にフォールバックするため、注記は両方を1文でカバーする必要がある」という趣旨へ書き換えた。テスト名も同じ趣旨に合わせて改名し、アサーション自体（新文言が表示されること）は既存のまま変更していない。
- **ロジックは1行も変更していない**: `useChainForm.ts`・`chainUtils.ts`のリクエスト組み立て・バリデーション・ゲート判定はすべて無改修。`end_source`に音声用のフィールドが増えていないため、フロントエンドが送信するJSONの形も変わっていない。
- **主な変更ファイル**: `webui/src/i18n/strings.ts`（`audioModeNote`・`strengthHint`、EN/JA同時）・`webui/src/modes/chained/ChainEndSourcePanel.tsx`（コメントのみ）・同`ChainEndSourcePanel.test.tsx`（テスト名のみ）。
- **検証**: `npm run typecheck`合格、`npm run lint`合格、`npm test`全PASS（既知失敗はいずれも本テーマと無関係）。バックエンド側の機械検証・機械ゲートA1〜A9はバックエンド`VERIFICATION_LOG.md` §65参照。
- **状態**: 実装・機械検証は完了。**オーナーの試聴ゲートは未実施**であり、本節の時点で本弾は完結していない。フロントエンド[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-2に試聴ゲート項目を起票してある。
- **（2026-08-18追記）** 目視・試聴は同日決着。裁定と完結記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-84。
- **（2026-08-19追記）オーナー目視・試聴合格で完結。** 詳細はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §65.9、台帳の記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-84。

## 83. 本日のUIバッチ3件 — End source複数クリップ警告の実装・CROP OUTPUT既定off・Retake窓長尺化の起票取り下げ（2026-08-18）

§82（End source第3弾）に続けて同日中に、オーナーが指示した独立の小粒改修3件を1バッチとして処理した。

- **End source複数クリップ時の品質警告を実装した**: §3-84（End source実用化テーマの完結）が起票していたUI警告を、オーナーが本日確定した最終文言「素材（末尾）つきで複数クリップを連結すると、生成結果の品質が低下します。」（EN: "Chaining multiple clips together with an end source attached lowers the quality of the generated result."）で実装した。`webui/src/modes/chained/useChainForm.ts`に派生値`endSourceMultiClipQualityWarning`を新設し、Chainedタブで「素材（末尾）」を添付かつクリップが2本以上のときにtrueを返す。`ChainEndSourcePanel.tsx`側は`warning-banner-mild`のバナーとして表示し、Generateはブロックしない。169フレーム跨ぎ警告（§79の窓内モード警告）とは表示条件が条件排他になるよう配線した。`strings.ts`にEN/JA文言を追加し、テスト6件を新設した。`npm run typecheck`・`npm run lint`・`npm test`はいずれも緑（既知の409起因7件のみ。§3-89で起票済みの既知事象で本項とは無関係）。台帳の記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-94（旧§1-22からクローズ）。
- **CROP OUTPUTの既定をOFFへ変更した**: オーナー指示（実際に使うとoffにする機会が多いため）を受け、バックエンド`config.yaml`（実運用ファイル）と`config.yaml.example`の`generation_defaults.crop_output`をnullへ変更し、フロントエンド`webui/src/config/defaultConfig.ts`のフォールバックもnullへ揃えた。プリセット（standard_720p等）自体が持つcrop値は据え置きで、プリセットを明示的に選択したときは従来どおり適用される。変更にあたってCROP OUTPUTのUI露出箇所を棚卸しした結果、Single/Chainedの共有コンポーネント1箇所のみに存在し、バッチ系2画面（バッチA2V・バッチi2v-long）は親フォームの値を継承する設計であること、Retake・Outpaintingには（outpaintとAPI相互排他のため）そもそも存在しないことを確認した。反映はバックエンドの再起動後。
- **§1-18（Retake窓の長尺化）の起票を取り下げてクローズした**: オーナー裁定（2026-08-18）——End sourceの実装により、本項が狙っていた長尺の撮り直し・作り直しは、AviUtl2の動画編集機能とChainedタブの組み合わせで幾何的に等価に実現できるようになったため、stage-1複数セグメント分割＋stage-2タイル別凍結スケジュールという専用機構を新設する動機が失われた。未着手のまま起票取り下げ。台帳の記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-93（旧§1-18からクローズ）。
- **状態**: 3件とも実装・機械検証・台帳整理が完了。デプロイとオーナー実機確認は別途。

## 84. Singleタブの賢い快適上限マーカー（`single_comfort_token_budget`の新設）— §3-12実装完了（2026-08-18）

**やったこと**: Singleタブ（Create画面）のフレーム数スライダーの快適上限マーカーは、従来`spill_free_frames`（3解像度キーのテーブル）を面積最近傍で丸めるだけで、プリセット解像度の間では粗くしか動かなかった。2026-08-18の実機検証（4段階・21ジョブ）で、**5つの高速化トグル（sage・block_swap_prefetch・keep_resident・fused_gguf_dequant_kernel・vae_mode=prune_vaed）が全てonの構成に限り**、快適上限のトークン線が**44,880**であることが3解像度×縦横両向きで較正されたため（正本バックエンド[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md)）、任意の解像度・縦横比で1フレーム単位（8n+1グリッド）で正しく動くマーカーへ置き換えた。台帳§3-12の後継で、実装・機械検証の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §66であり、本節はフロントエンド側の設計判断の記録である。

- **式であって表ではない**: `webui/src/modes/single/spillUtils.ts`に逆算式`singleComfortFrames`を1本だけ新設した（`cells = floor(幅/32)*floor(高さ/32)` → `N_latent_max = floor(予算/cells)` → `frames = 8*(N_latent_max-1)+1` → `[minNumFrames, maxNumFrames]`へclamp）。`COMFORT_LIMIT_TABLE.md`の60行の表はコードへ焼き込んでいない——表は人が検算するための資料であって、実装の入力ではない。`snapNumFrames`は挟まない（`8*(n-1)+1`は定義上8n+1グリッド上にあり、clampだけで足りる）。
- **置き場所の理由**: 予算・逆算式は`webui/src/modes/single/spillUtils.ts`へ置き、既存の`webui/src/shell/tokenBudget.ts`には一切触れていない。`tokenBudget.ts`は冒頭のコメントで「CHAIN-ONLYのトークン予算幾何——単発`/generate`はここに含めるな、`CHAIN_COMFORT_TOKEN_BUDGET`をアプリ全体の定数へ一般化するな」というスコープ宣言を明示的に持っており、Singleからのimportを禁止している。`VAE_SPATIAL_FACTOR=32`に相当する定数も`spillUtils.ts`側にローカルで意図的に重複させ、コメントでミラーの欠落ではないと明記した。
- **全on判定とsage可否3値の扱い**: 全on判定`isFullAcceleration`は`webui/src/shell/accelerationSettings.ts`へ新設した。sageは`sageAvailable`（`true`/`false`/`null`）を、既存の`sageAvailability`が持つ「明示的な`false`にだけ反応する」規律のまま扱う——`false`（サーバー不可と判明）はフォールバック側、`null`（`/status`未着、または旧サーバー）は効果的にonとみなす。**`/status`未着の数秒間はsage不明→賢いマーカー表示→着信で従来値へ跳ぶちらつきを許容した**（既存のボタン無効化と同じ規律の一貫性を優先。追加の抑制ロジックは入れていない）。引数は`effectiveAcceleration`通過後の実効オブジェクトを要求し、生値を渡すと`keepResident`を誤判定する（prefetch off時の畳み込みを見誤る）ため、JSDocでMUST-be-effectiveと明記した。
- **低解像度でフォールバック値と一致し、文言だけ変わる仕様**: 512×320のような低解像度では、逆算式の出力（clampで481=上限）と既存`spill_free_frames`のフォールバック値がたまたま一致する。この場合、**マーカーの位置は動かず、超過時の警告文言だけが全on/非全onで入れ替わる**——`isComfortMarkerSmart`は値の一致・不一致とは独立に、5条件の構成だけで決まるためである。**「文言は値ではなく構成で決まる」**という仕様であり、値が同じだから文言も同じになるはずという誤解をしないこと。
- **文言の出し分け**: 超過時の警告は`single.comfortWarningSmart`を新設して出し分けた。既存`single.spillWarning`「これを超えると生成速度が2〜4倍遅くなります。」は既定構成（非全on）の実測値であり、全on構成のスマートマーカー時にそのまま出すと事実と食い違うため。命名は`single.size.comfortMarkerTitle`（Chained解像度スライダー用ガイドのツールチップ、別機能）・`edit.comfortWarning`（Outpaintingの別軸予算の警告）との混同を避けるため`Smart`を付した。`GenerationForm.tsx`が`isComfortMarkerSmart`で三項に出し分け、`CommonGenerationFields.tsx`は文字列を受け取るだけで無改修。
- **目盛りはネイティブdatalistのまま**: `CommonGenerationFields.tsx`の`<datalist>`目盛り（:416-420）は無改修で動く。マーカー自体にラベルは付けない既存作法も維持した。フィールド名`spillThresholdFrames`も改名していない（改名の連鎖を避けるため、JSDocで「賢い値とフォールバック値の両方を指す」二面性を説明した）。
- **配線**: `AppShell.tsx`から`sageAvailable`を`SingleScreen`→`useGenerationForm`へprops 1本追加した（Chained・Settingsは無改修）。`useGenerationForm.ts:967-971`のマーカー算出を「全onなら`singleComfortFrames`の賢い値（nullなら以下へフォールバック）・そうでなければ既存`resolveSpillFreeFrames`」へ差し替え、返り値へ`isComfortMarkerSmart: boolean`を追加した。既存`resolveSpillFreeFrames`・`isOverSpillThreshold`のロジックは無改修——`deriveDuration.ts`（右クリック尺）・`OutpaintingPanel.tsx`・A2V wav自動調整（`useGenerationForm.ts:846`）は引き続きこの関数を読み、今回のマーカー改修の対象外とした（据え置きの詳細と将来課題は台帳[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-95）。
- **実装ファイル**: `webui/src/modes/single/spillUtils.ts`（`SINGLE_COMFORT_TOKEN_BUDGET`・`resolveSingleComfortBudget`・`singleComfortFrames`を新設）、`webui/src/shell/accelerationSettings.ts`（`isFullAcceleration`を新設）、`webui/src/modes/single/useGenerationForm.ts`（マーカー算出の差し替え）、`SingleScreen.tsx`・`webui/src/shell/AppShell.tsx`（`sageAvailable`配線）、`GenerationForm.tsx`（文言の出し分け）、`webui/src/api/types.ts`（`AppLimits.single_comfort_token_budget?`をオプショナルで追加）、`webui/src/i18n/strings.ts`（`single.comfortWarningSmart`のEN/JA）、`defaultConfig.ts`・`bridge/mockBridge.ts`（新キーの両追加。B-1パリティ守衛が検出するのは「フォールバックにあってモックに無い」方向のみのため、`defaultConfig.ts`側を先に追加した）。
- **検証**: `npm run typecheck`**0エラー**、`npm test`（関連3ファイル）**104 passed / 104**（`spillUtils.test.ts`・`accelerationSettings.test.ts`・`useGenerationForm.test.ts`）、`npm test`（全件）**2472 passed / 2480**（失敗8件は`JobsContext.test.tsx`の既知409起因フレーク2ファイル分で本テーマとは無関係）、`npm run lint`**31 warning / 0 error**（ベースライン維持）。バックエンドは`pytest`関連2ファイル**117 passed / 7 skipped**、全件**1721 passed / 20 skipped / 1 failed**（失敗1件はMCP `backend_status`の既知の環境依存フレークで本テーマとは無関係）。追加テストの内訳・鍵取り違えの回帰の詳細はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §66.4を参照。
- **状態**: 実装・機械検証は完了。デプロイ・オーナー目視ゲートは別途（台帳[`PENDING_TASKS.md`](PENDING_TASKS.md) §2-4）。
- **（2026-08-19追記）オーナー目視合格で完結。** 台帳§2-4の10項目すべてに合格した（文言修正2件の要望は別途、コードは別エージェントが実装中）。詳細はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §66.6、台帳の記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-12。

## 85. 文言修正2件・プリセットのフレーム数引き上げ・ツール名ドロップダウン（モック）の3件（2026-08-19）

**やったこと**: いずれもオーナー指示による小改修3件。前節§84の目視合格を受けた文言修正2件、快適上限マーカーの較正値（44,880）に合わせたプリセットのフレーム数引き上げ、将来のLTX 2.5対応に備えたヘッダーのツール名ドロップダウン（モック）の新設。

- **文言修正2件**: `single.comfortWarningSmart`（Singleタブ・賢い快適上限マーカーの超過警告）を「VRAM溢れにより生成が遅くなる可能性があります。」へ、Settingsの`spillFreeHint`（`spill_free_frames`表の見出し説明）を「各解像度における、VRAM溢れが起きずに生成が2～4倍遅くならないフレーム数です。（暫定版）」へ、それぞれオーナー確定文言に差し替えた。いずれも`webui/src/i18n/strings.ts`のEN/JA文字列定義のみの変更で、呼び出し側のロジックは無改修。
- **プリセットのフレーム数引き上げ**: オーナー指示により、`COMFORT_LIMIT_TABLE.md`（正本バックエンド[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md)）が定める快適上限線44,880（全高速化on構成）の逆算式で、`standard_720p`（1280×768）を257→**361**、`FHD_1080p`（1920×1088）を153→**169**、`WQHD_1440p`（2560×1472）を81→**89**フレームへ引き上げた。`generation_defaults`も`standard_720p`と一致させるため257→**361**へ揃えた。`small`・`minimal`は式の値が481クランプに収まる側なので変更なし、`smoke_test`は検証用サイズのため据え置き。変更箇所は`config.yaml`（実運用設定）・`config.yaml.example`・`webui/src/modes/single/defaultConfig.ts`・`webui/src/bridge/mockBridge.ts`のモック応答の4点で、値を完全一致させてある。
  - **`limits.spill_free_frames`（粗い5点の実測テーブル）は意図的に据え置いた。** こちらは5つの高速化トグルが全onでない既定構成向けのフォールバック値であり、全on構成限定の44,880線とは別の物理条件から出た別の数値である（両者を混ぜてはならないことは`COMFORT_LIMIT_TABLE.md`第6節に明記済み）。プリセットの引き上げによって両者の値が意図的に食い違う状態になったが、これは仕様であり不整合ではない。
- **ツール名ドロップダウン（モック）**: ヘッダーの静的な「Nz-Videomni」表示を撤去し、「LTX 2.3」（既定）／「LTX 2.5」を選べるドロップダウンへ置き換えた（`webui/src/shell/AppShell.tsx`）。**LTX 2.5を選ぶと、トーストも警告も出さず無言で「LTX 2.3」へ戻る**——LTX 2.5は実体を持たないモックの選択肢で、将来LTX 2.5対応を実装する際の置き場所を先取りしただけである。選択状態は`useState`のみでブラウザ間の永続化はしない（ページ再読み込みで既定へ戻る）。文言は`strings.toolVersion`（`ariaLabel`/`ltx23`/`ltx25`）に新設。
- **検証**: 3件とも文言・表示値・UIコンポーネントの変更に留まり、生成ロジック・API契約・バックエンドの改修は無い。プリセット値の反映は**バックエンド再起動後**（`config.yaml`はプロセス起動時に読み込まれるため）。
- **状態**: 実装完了。オーナー目視確認は次回セッション以降。

## 86. ヘッダーのベースモデルドロップダウンの実配線（マルチエンジン土台§3-97のP7）（2026-08-20）

**やったこと**: 前節§85でモックとして置いたヘッダーのドロップダウンを、バックエンドのベースモデル軸（§3-97のP0〜P6で新設済み）へ実際に配線した。**選んだ瞬間にそのベースモデルの読み込みが始まる**——設計正本はバックエンド[`MULTI_ENGINE_DESIGN.md`](../../../Docs/MULTI_ENGINE_DESIGN.md) §6。本節はフロントエンドのみで、Pythonは1行も触っていない。

- **仕様がモックから逆転した点（本節でいちばん重要）**: §85のモックは「LTX 2.5を選ぶと**無言で**LTX 2.3へ戻る」だった。裏に何も無いのだから説明することも無い、という判断で、当時は正しかった。実配線後はこれが**真逆の間違い**になる——選択が本当に効く世界では、無言で戻る挙動は「操作が届かなかったのか、壊れているのか」を利用者が判別できない。新仕様は**「元へ戻す。ただし理由を必ず伝える」**（§6.2のガード(4)は「表示を切替前へ戻す」ことだけを要求しており、黙って戻すことは要求していない）。この反転に伴い、旧仕様を明示的にpinしていた`AppShell.toolVersion.test.tsx`は全面書き換えになり、台帳§2-1のモック目視条件は破棄した（記録はバックエンド[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-99）。
- **選択肢はサーバー由来にした**: `GET /models`が返す`base_models[]`の`display_name`をそのまま描画する。`strings.toolVersion`が持っていた`ltx23`/`ltx25`のリテラルは削除した——**ベースモデルの追加がサーバー側の記述ファイル1本の追加で済む**という§3-97の目的を、フロントエンドのハードコードで台無しにしないため。表示は導入状態で3通りに出し分ける（完全導入＝そのまま／一部導入＝「（一部未導入）」／未導入＝「（未導入）」）。
- **`present`と`installed`を分けて扱う**: 「1つも無い（`present: false`）」ときだけAPIを叩かずに`install-<id>.bat`の案内を出して打ち切る。**一部導入は短絡させずサーバーへ行かせる**——どのカテゴリが足りないかを一番よく知っているのはサーバーであり、こちら側で言い当てようとすると必ず古くなるため。バッチファイル名は記述子のidから組み立てる（`shell/useBaseModels.ts`の`baseModelInstaller`）ので、将来のベースモデルでも配線変更は要らない。
- **失敗の出し分け**: 409 `JOB_BUSY`（生成中）／409 `PIPELINE_LOADING`（読み込み中の二重切替）／422（非互換等）／その他（404の未知id・通信断）の4通り。**422はサーバーの`detail`をそのまま出す**——LTX 2.5を選んだときに出る「このtransformerはltxv 2.5.0です。LTX 2.5エンジンは次段階(PENDING_TASKS §3-98)で実装予定のため、まだ読み込めません。」のような、サーバーだけが知っている事実を言い換えないため（§2.5「フールプルーフは作らない。エラー品質で解決する」）。これに合わせて`BackendApiError`へ`detail`を追加した（エンベロープの`detail`が文字列のときだけ拾う。`VALIDATION_ERROR`の配列形式は`undefined`扱い）。
- **表示の復帰は`finally`で行う**: 表示値は「進行中の切替先 ?? サーバーの`active_base_model`」で、進行中フラグを`finally`で必ず落とす。どの失敗経路を通っても表示が切替前へ戻ることが構造で保証され、経路ごとに戻し忘れる余地が無い。
- **トーストは効果ではなくフックの戻り値から出す**: `switchBaseModel`が結果（`switched`/`not-installed`/`busy`/`loading`/`rejected`/`failed`）を**返し**、`AppShell`がそれを見て1操作につき1つトーストを出す。フックの状態変化を`useEffect`で監視する方式だと、ある描画の状態が「いま押した切替」のものか古いものかを判別できず、二重発火・取りこぼしの温床になるため採らなかった。
- **ヘッダーバッジに「読み込み中」を追加**: `GET /status`に加わった`state`が`"loading"`のとき、`modes/single/useServerStatus.ts`が新しい`loading-models`を返す。**判定は`queue.running`より前に置いた**——読み込み中は`pipeline_loaded`が`false`のままで数分続くため、これが無いとバッジは「サーバー稼働中」に見え、利用者は生成できると誤解して失敗を踏む（§6.5。当初「offlineに見える」と書かれていた根拠は誤りで、実際にはonlineのまま留まる）。見た目は生成中と同じ`badge-busy`、文言だけ別（「モデル読み込み中…」）。
- **カテゴリの表示順をサーバー駆動へ**: Settingsのモデル選択欄の並び順を、アクティブなベースモデルの`categories`のキー順（＝記述子の宣言順）から取るようにした（`useModels`の`categoryOrder`）。`MODEL_CATEGORIES`定数は**型の器として温存**した——`Record<ModelCategory, string>`という選択状態の型が共用体を必要とするためで、完全撤去は型が広範囲へ波及する（設計書§4.3の「FE MODEL_CATEGORIES撤去」の記述は「型温存・表示サーバー駆動」が正）。あわせて`services/model_registry.py:69-97`という古い行番号参照のコメントを、記述子ベースの説明へ書き直した。
- **モックブリッジの追随**: `GET /models`を3層化（`active_base_model`＋`base_models[]`。従来の`categories`はアクティブなベースモデルのブロックそのものにして互換を保つ）、`POST /pipeline/load`を`base_model`対応（未知id→404、LTX 2.5→422を実物と同一文面の`detail`付きで返す）、`GET /status`へ`state`/`base_model`を追加した。**`models`が空でも`base_model`があればレガシー早期returnへ落ちないようにした**——これは設計書§5.5(a)が警告している罠そのもの（ドロップダウンが送るのはまさに`base_model`だけの呼び出し）で、モック側で再現しておかないとUI開発中に気づけない。新オプション2つ（`ltx25Install`＝ディスク上の導入状態、`supportedBaseModels`＝エンジンが実際に動かせるid集合）は**別軸として分けた**——「ファイルはあるが動かせない」が現実の状態だからで、§3-98が出荷したら後者の既定値を1行変えるだけで済む。
- **実装ファイル**: `webui/src/shell/useBaseModels.ts`（新設）・`useBaseModels.test.ts`（新設）、`webui/src/shell/AppShell.tsx`（配線・トースト）、`webui/src/shell/AppShell.toolVersion.test.tsx`（全面書き換え）、`webui/src/shell/useModels.ts`（`categoryOrder`）・`ModelsPanel.tsx`、`webui/src/modes/single/useServerStatus.ts`・`StatusHeader.tsx`、`webui/src/api/types.ts`（`BaseModelBlock`新設ほかオプショナル加算）・`client.ts`（`loadPipeline`の第2引数・`BackendApiError.detail`）、`webui/src/i18n/strings.ts`（EN/JA）、`webui/src/bridge/mockBridge.ts`、`webui/src/i18n/strings.test.ts`（削除したリテラルのアサーション差し替え）。
- **検証**: `npm run typecheck`**0エラー**、`npm run test -- --run`**2486 passed / 10 skipped（133ファイル全緑）**（前回2482→+4。内訳はこのテーマで新設・書き換えた分）、`npm run lint`**0 error**（警告はベースライン維持）、`npm run build:single`**成功**（631.34 kB）。
- **積み残し**: 「生成ジョブ実行中はドロップダウンを無効化する」ガードは実装済み（`serverStatus.kind === "busy"`）だが、**自動テストは書けていない**——`AppShell`内の`useServerStatus()`はテスト用に注入したブリッジではなくアプリ全体のシングルトンを見る既存の作りのため、テストからこの状態を作れない。実機ゲート側（§3-97のP8）で確認する。

**P8での追記（2026-08-20）**

- **デプロイは完了した**。`build.ps1 -Config Release` → `deploy.ps1` で、実機（`...\aviutl2_v2.0.54\data\Plugin\NzVideomni\`）とリポジトリ配布コピー（`AviUtl2-Plugin\NzVideomni.aux2`）の2か所へ配布した。両者のSHA-256は一致（1,239,552バイト）。
- **本節の文言が実際に`.aux2`へ入っていることをバイト検索で確認した**。`Base model`／`ベースモデル`／`loading-models`／`Loading models`／`モデル読み込み中`／`partly installed`／`一部未導入`／`切り替えられません` を検出、モック時代の`Tool version`／`ツールのバージョン`は**不検出**。埋め込みリソースが古いまま配布される、という事故が起きていないことの確認である。
- **上の積み残し（生成中の無効化）は、P8でも自動では確認できなかった。** これはテストの都合ではなく**確認手段そのものの制約**である——確かめるには実際に生成ジョブを走らせながらAviUtl2のヘッダーを目で見る必要があり、エージェントにはそれができない。したがって**オーナーの目視ゲートへ引き継ぐ**：バックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §68.8 の3番目の項目がこれにあたる。同節には他に「ドロップダウンの2項目表示」「LTX 2.5選択時の422トーストとLTX 2.3への復帰」「読み込み中バッジ」の3点が並んでいる。
- **サーバー側の振る舞いは実機で確認済み**なので、目視で見るのは「UIがそれをどう見せるか」だけである。たとえば422の文面は、§68.4に実機の応答全文が記録されている（本節が「サーバーの`detail`をそのまま出す」と書いたとおりの文字列が、実際にサーバーから返っている）。
- **状態**: 実装・機械検証・デプロイまで完了（2026-08-20）。残るのはオーナーの目視4点のみ（[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §68.8）。

## 87. オーナー目視で出た2件の修正 — カテゴリ表示順の退行と「モデル読み込み中」バッジの取りこぼし（2026-08-20）

**やったこと**: §86（P7）の実配線を実機で見たオーナーから上がった2件を直した。どちらも「実装は入っているのに、利用者の目に届いていない」種類の不具合である。

### (1) Settingsのモデル選択欄の並び順が交換頻度順から崩れていた

- **症状**: 並びが「音声モデル／テキストエンコーダ／動画モデル／動画VAE」——カテゴリ名（`audio`/`text_encoder`/`transformer`/`video_vae`）のアルファベット順そのものになっていた。あるべき並びは**交換頻度順**（動画モデル→テキストエンコーダ→動画VAE→音声モデル）である。
- **調べたこと（順に潰した）**: 記述子`scripts/manifests/10-ltx23.json`・`20-ltx25.json`のキー順＝交換頻度順で**正しい**。`services/base_models.py`の読み込みは宣言順を保つ（`dict`の挿入順）。`api/models_registry.py`の組み立ても記述子順のまま。**実機で動いているバックエンドへ直接`curl`した応答も記述子順**（`transformer`,`text_encoder`,`video_vae`,`audio`）。フロントエンドの`useModels.ts`／`ModelsPanel.tsx`は`Object.keys`をそのまま使っており、JSのオブジェクトは文字列キーの挿入順を保つ。ビルド済みバンドルと配布済み`.aux2`の中身もソースと一致していた。ネイティブ側も`nlohmann::ordered_json`（`bridge_core.h`の`json_t`）で統一されていて、順序を壊す実装は無い。
- **残った疑い**: 両端が正しいのに順序が失われているのだから、疑いは**唯一こちらで検証できない区間**——AviUtl2プラグインのWebView2メッセージ経路（`PostWebMessageAsJson`でホストからページへ渡す所）に残る。ここを確かめるにはAviUtl2を動かして目で見るしかなく、しかも今回それはできない（後述のとおり起動中）。
- **直し方**: 疑いの当否に依存しない形にした。**表示順をJSONの「配列」で運ぶ**——`GET /models`の`base_models[]`に`category_order`（記述子の宣言順そのまま）を加算し、フロントエンドはこの配列を読む。**配列の要素順はどんな転送でも保たれる**ので、オブジェクトのキー順がどこかで並べ替えられても表示順は動かない。キー順は従来どおり同じ並びで送り続ける（既存の利用者のため）が、**表示順の根拠としては使わない**。
- **回帰テスト**: バックエンドは「記述子がわざと変な順（`audio`/`video_vae`/`transformer`/`text_encoder`）で宣言したベースモデルを、応答がそのとおりに返すか」を`tests/test_models_endpoint_compat.py`で、「出荷している記述子2本が交換頻度順を宣言しているか」を`tests/test_base_model_contract.py`で固定した。フロントエンドは`useModels.test.ts`で**キー順をアルファベット順に並べ替えた応答**（＝実機で起きたことの再現）を食わせ、それでも`categoryOrder`が交換頻度順になることを固定した。`category_order`が無い旧バックエンド向けのフォールバック（キー順→定数）も1本置いた。
- **文言も同時に変えた**（オーナー指定）: 「動画モデル (transformer)」→**「動画モデル (checkpoint)」**（CivitAI等の流儀に合わせ、利用者が普段見る呼び名にする）、「テキストエンコーダ (Gemma)」→**「テキストエンコーダ」**（他エンジン対応時にUIの文言を触らなくて済むように、エンジン名を落とす）。EN/JA両方と、バックエンド同梱のGradio UI（`gradio_ui/i18n.py`）も揃えた。

### (2) 「モデル読み込み中…」バッジが短い読み込みで出ない

- **症状**: Settingsで`default`→`Sulphur`へ切り替える（実測約8秒。バックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §68.5に同じ8秒の記録がある）とバッジが変化しない。
- **裏取り**: 配線は正しい。`GET /status`の`state`は読み込みの**全期間**`"loading"`であり（`services/pipeline_manager.py`の`reload`はロック内で`STATE_LOADING`にしてから戻る）、`useServerStatus`の分岐も`queue.running`より前に`state === "loading"`を見ている。プラグインのHTTPワーカーは4本あるので、読み込み中の`POST`が`/status`のポーリングを塞ぐこともない。**残る差は周期だけ**で、`DEFAULT_INTERVAL_MS`が10秒固定だったため、8秒の読み込みがまるごとポーリングの隙間に落ちうる。
- **直し方**: **周期を2.5秒に短縮した（(a)案）**。(b)「`loading`を見ている間だけ速くする」は、**その`loading`を1度も観測できないのが今回の症状**なので原理的に効かない。(c)「`POST /pipeline/load`を出した側が即時refresh＋高頻度化」は効くが、Settingsのパネルとヘッダーのバッジという別々の持ち主の間に新しい配線を1本増やすことになり、**例外を増やさない**という方針に反する。`GET /status`はメモリ上の値＋`torch.cuda.mem_get_info`1回だけの軽い読み取りで、実機のプラグインログでも往復**約2ms**、しかも相手はlocalhostである。ジョブ台帳のポーリングが既に2秒周期で回っている隣で2.5秒にしても桁は変わらない。
- **副作用の確認**: `useServerStatus`の利用者はヘッダーのバッジ（`StatusHeader`）、生成中のドロップダウン無効化（`serverStatus.kind === "busy"`）、`/status`本文から読む機能フラグ（sage・block swap prefetch）の3つで、いずれも**同じ値がより新しくなるだけ**である。判定式も分岐も変えていない。
- **回帰テスト**: タイマーをモックし、「マウント1秒後から9秒間だけ`state: "loading"`を返すサーバー」に対して**既定の周期のまま**バッジが1度以上立つことを固定した。この窓は10秒周期だと確実に取りこぼす位置に置いてあるので、既定値を10秒へ戻すとこのテストが落ちる。

- **実装ファイル**: `webui/src/shell/useModels.ts`・`useModels.test.ts`、`webui/src/api/types.ts`、`webui/src/bridge/mockBridge.ts`、`webui/src/i18n/strings.ts`、`webui/src/modes/single/useServerStatus.ts`・`useServerStatus.test.ts`（バックエンド側は`api/models_registry.py`、`gradio_ui/i18n.py`、`tests/`2本）。
- **検証**: `npm run typecheck` **0エラー** ／ `npm run test -- --run` **2490 passed（132ファイル）**（実機バックエンド前提の`src/api/backend.integration.test.ts`は除外。理由は下記）／ `npm run lint` **0 error**（警告31件はベースライン維持）／ `npm run build:single` **成功**（631.37 kB）。バックエンドは`pytest` **1806 passed / 20 skipped**（前回1804＋新規2）。
- **`.aux2`は再ビルドしてデプロイした**: 作業中はAviUtl2が起動していた（`deploy.ps1`は起動中の配置を拒否する仕様）が、ビルド完了時点で終了していたため`build.ps1 -Config Release` → `deploy.ps1`まで通した。実機（`...\Plugin\NzVideomni\`）とリポジトリ配布コピー（`AviUtl2-Plugin\NzVideomni.aux2`）の2か所へ配布し、ビルド成果物を含む3つのSHA-256が一致することを確認済み。新しい文言（`Video model (checkpoint)`）と`category_order`が埋め込みリソースに入っていることもバイト検索で確認した。
- **積み残し**: (1)の真因（WebView2メッセージ経路が本当にキー順を並べ替えるのか）は**未確定のまま**である。今回の修正は真因に依存せず効くが、原因そのものを確かめるにはAviUtl2上での目視が要る。**他の場所でサーバー応答のキー順に依存している箇所は無い**ことは確認済みで（依存していたのはこの1か所だけ）、実害は残っていない。

## 88. LTX 2.5でChainedタブ・V2V・A2V・Batch A2Vを開放した（バックエンド§3-102のC4）（2026-08-23）

**やったこと**: LTX 2.5エンジンが連結生成・V2V継続・A2Vに対応したのに合わせ、フロントエンドの機能スコープを**「タブ全体を落とす」から「素材ごとに落とす」へ一段細かくした**うえで、対応済みになった機能のグレーアウトを外した。3コミットに分かれているが、話としては1つである。設計正本はバックエンド[`MULTI_ENGINE_DESIGN.md`](../../../Docs/MULTI_ENGINE_DESIGN.md) §5.6・§8.5、機械ゲートの記録は[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §72.5（連結生成）・§73.5（V2V・A2V）。

### (1) Chainedタブの開放と4パネルの個別グレーアウト（`de5442c`）

- **`useBaseModels.ts`**: `unsupportedFeatures`から4つのbooleanを導出する純関数`chainPanelsDisabledFor`を新設した（既存の`batchA2vDisabledFor`と同じ作法）。`v2v`／`a2v`／`end_source`／`reference_video`をそれぞれ1パネルへ対応させる。**`chain`は見ない**——連結そのものができないベースモデルは`disabledModesFor`がタブごと落とすので、ここで重ねて判断すると**同じ裁定を2箇所で持つ**ことになるからである。`MODE_REQUIREMENTS`は無変更。
- **`AppShell.tsx` / `ChainedScreen.tsx`**: 計算済みの4boolean（`v2vUnavailable`／`a2vUnavailable`／`endSourceUnavailable`／`referenceUnavailable`）を`ChainedScreen`へ渡し、`SourceInputPanel`／`ChainAudioPanel`／`ChainEndSourcePanel`／`ChainReferencePanel`の4枚を**既存の`disabled`機構**で無効化して、それぞれの上に理由を1行出す。**新しいUI部品は作っていない**——Batch A2Vが既に使っている`warning-banner`の書式に合わせた。
- **切り替え前に取り付けてあった素材は消さない**。灰色になるだけで、送信そのものはサーバーの422 `FEATURE_UNSUPPORTED`でfail loudに断られる（LoRAタグやNAGと同じ扱い）。
- **`mockBridge.ts`**: LTX25の`unsupported_features`から`chain`を外し、`handleGenerateChain`が**アクティブなベースモデルが宣言している機能に限り**、対応フィールドが非空なら本物と同形の422封筒を返すようにした。LTX 2.3は何も宣言しないので既存のChainedテスト群は1件も影響を受けない。
- **テスト**: 2515 → **2538**（+23）。

### (2) 素材（冒頭）欄は動画の口だけを閉じる（`33d28df`・(1)の追補）

- **なぜ追補が要ったか**: (1)は`SourceInputPanel`を**パネルごと**無効化していたが、この欄は「クリップ1の開始画像（I2V）」と「V2Vの元動画」という**別々の素材を1枚のカードで兼ねている**。開始画像はLTX 2.5でも使える連結生成本体の一部なので、丸ごと落とすと対応済みのI2V連結がWebUIから組めなくなっていた（監督裁定）。そこで**半分だけ閉じる**形へ改めた。
- **`useChainForm.ts`**: `SourceAttachOptions{imagesOnly}`を新設し、`pickSource(options?)`／`attachSourceByPath(..., options?)`が受け取るようにした。`imagesOnly`のときはネイティブのダイアログを`kind:"image"`で開くので、そもそも動画を選べない。別経路（打ち込んだパスなど）で動画が届いた場合は、**どちらのスロットにも触れずに**`sourceError="VIDEO_SOURCE_UNSUPPORTED"`で断る——既に付けてある開始画像が巻き添えで消えないよう、判定は分岐の前に置いた。**判定をパネルではなくフォームに置いた**のは、2つの取り付け経路がどちらも同じ`attachRoutedSource`へ合流するためで、パネル側に書くと同じ振り分けを二重に持つことになる。
- **`SourceInputPanel.tsx`**: 新プロップ`videoUnavailable`。ドロップの受理拡張子から動画を外し（動画のドロップは既存の「非対応」行に落ちる）、選択ボタンは`imagesOnly`で呼び、ラベルも「画像を選択…」へ切り替える。CONTEXT FRAMESスライダー（動画が付いているときだけ現れる＝切り替え前の残留分）は無効化。**画像側の選択・クリア・強度・サムネイルはすべて有効のまま**。
- **`ChainedScreen.tsx` / `i18n/strings.ts`**: `disabled`は元に戻し、`videoUnavailable={v2vUnavailable}`を渡す。理由文は「素材に動画を使うことだけが使えない／開始画像は影響を受けない」趣旨へ限定し、`chooseImageOnlyButton`をja/enへ1行ずつ追加した。
- **テスト**: 2538 → **2544**（+6）。`SourceInputPanel.test.tsx`に「2.5でも開始画像は付く」「ダイアログは`kind:"image"`で開く」「動画は断られ、既存の開始画像は残る」「プロップ無しなら従来どおり動画も選べる」を追加した。

### (3) V2V・A2V・Batch A2Vの開放（`05bdb9a`）

- **製品コードは無変更で済んだ**。`mockBridge`が宣言するLTX 2.5の使えない機能一覧から`"v2v"`と`"a2v"`を外しただけで、グレーアウトはサーバー宣言駆動なので`chainPanelsDisabledFor`／`batchA2vDisabledFor`を1行も触らずにV2V素材パネル・A2V音声パネル・Batch A2V欄が復帰する。素材（末尾）と参照動画は宣言に残るので、これまでどおり個別に灰色のままになる。**(1)で「素材ごとに落とす」形にしておいたことが、ここで効いている。**
- **`MOCK_CHAIN_FEATURE_FIELDS`の`source_video`／`source_audio`の行は残した**——422を決めるのは行の有無ではなく**宣言一覧のメンバシップ**で、行は「どの引数がどの機能名に属するか」を言っているだけだからである。
- **テストは3本が丸ごと反転した**（「無効であること」を確かめていたものを「有効であること」を確かめる形へ書き換えた）: `AppShell.featureScope.test.tsx`（2.5でV2V・A2Vのパネルが有効・素材（末尾）と参照だけが理由文つきで灰色・ソース選択ボタンの説明が「画像または動画」へ復帰・Batch A2V欄が有効）／`mockBridge.test.ts`（`v2v`・`a2v`を422の一覧から外し、2.5で`source_video`／`source_audio`／1クリップ`full_length`のA2Vが202で通ること、素材（末尾）との併用は422のままであることを追加）／`useBaseModels.test.ts`（宣言に残る機能名`end_source`で確認する形へ変更）。LTX 2.3側のテストは無傷（宣言が空のままなので全機能が通る）。

### 検証とデプロイ

- **機械検証**: `npm run typecheck` **0エラー**、`npm run build`／`build:single` **成功**、vitest は(2)の時点で**2,544 passed / 10 skipped（合計2,554件・134ファイル）**（[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §72.5）。(3)はテストの書き換えのみで、総数の増減は記録に残していない。
- **`.aux2`は3回とも再ビルドして2箇所へ配った**（実機のAviUtl2インストール先の`Plugin\NzVideomni\NzVideomni.aux2`と、リポジトリの配布用コピー`AviUtl2-Plugin\NzVideomni.aux2`）。最終形は**1,245,184バイト**で両者一致している（同 §73.5）。**ビルドは必ず`build.ps1` → `deploy.ps1`を通すこと**（ninjaを直接叩くとWeb UIの埋め込みが更新されない）。
- **バックエンド側の対応する記録**: MCPの説明文もこの日に2回直している（`ea6cff8`＝連結生成の解禁、`74baeea`＝V2V・A2Vの解禁と「A2Vはclipsがちょうど1件」という失効記述の訂正）。どちらも文言のみで、ツールの引数・振る舞いは1つも変えていない。
- **状態**: 実装・機械検証・デプロイまで完了（2026-08-23）。**オーナーの実機確認も決着済み**——連結生成は2026-08-23、V2V継続とA2Vは2026-08-24に合格した（[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §72.10・§73.10）。

## 89. LTX 2.5でスタイルLoRAと参照動画（IC-LoRA）を開放した（バックエンド§3-102のC4）（2026-08-24）

**やったこと**: LTX 2.5エンジンがスタイルLoRA（絵柄を寄せる追加学習ファイル）とIC-LoRA（参照動画から輪郭線・骨格・深度などの制御信号を作って生成を導く仕組み。複数クリップにまたがる長尺IC-LoRAを含む）に対応したのに合わせ、フロントエンド側のグレーアウトを外した。コミットは`7fab22d`の1本である。設計正本はバックエンド[`MULTI_ENGINE_DESIGN.md`](../../../Docs/MULTI_ENGINE_DESIGN.md) §5.6・§8.5、機械ゲートの記録は[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §74.6、実機ゲートは同 §74.7。

### (1) 製品コードの変更は「宣言から2語を消す」だけ

- **`mockBridge.ts`**: 宣言するLTX 2.5の使えない機能一覧から`"loras"`と`"reference_video"`を外した。グレーアウトはサーバー宣言駆動なので、`chainPanelsDisabledFor`を**1行も触らずに**Chainedタブの参照動画パネルが復帰する。Create（Single）タブの参照欄とLoRA欄は**もともと機能スコープを見ていない**ので、こちらも無変更である（スタイルLoRAはプロンプト内のタグで指定する方式で、元から閉じていなかった）。
- **宣言に残るのは`end_source`・`retake`・`outpaint`・`nag`・`prune_vaed`・`sage_attention`・`keep_resident`・`two_stage_hq`の8件**で、Chainedタブの中で個別に灰色のまま残るのは**素材（末尾）の1カードだけ**になった。
- **`MOCK_CHAIN_FEATURE_FIELDS`の`loras`／`reference_video_id`の行は残した**——422を決めるのは行の有無ではなく**宣言一覧のメンバシップ**で、行は「どの引数がどの機能名に属するか」を言っているだけだからである（`v2v`／`a2v`のときと同じ扱い）。

### (2) テスト

- **`mockBridge.test.ts`**: 宣言一覧の期待値から2語を外し、**戻ってきたら気づけるよう否定の確認を足した**。422を並べた表からも2行を外し、`source_video`／`source_audio`の前例と同じ形で「2.5で202が返る」受理テストへ移した。**参照付きのリクエストは実サーバーが128の倍数の解像度を要求するので、高さは384で送っている**（512×320では通らない）。「素材（末尾）と参照を拒む」という名前のテストは、実際には素材（末尾）しか送っていなかったので実態に合わせて改名した。
- **`AppShell.featureScope.test.tsx`**: 参照パネルを「灰色を確かめる組」から「開くことを確かめる組」へ移し、グレーアウトのテストは素材（末尾）1枚だけを見る形へ書き換えた。あわせて、**2.5でCreateとChainedの両方の参照動画パネルが操作でき、素材（末尾）だけが理由文つきで灰色のまま**であることを確かめるテストを新設した。
- **`useBaseModels.test.ts`は無変更**（手書き配列の純関数テストで、宣言に残る`end_source`で確認する形は今回も有効）。
- **LTX 2.3側のテストは無傷**（宣言が空のままなので全機能が通る）。
- **件数**: vitest 2,544 → **2,543**（422の表から2行×2件＝4件減り、受理2件と参照パネル1件を新設した）。型検査 **0エラー**、`build`／`build:single` とも成功。

### (3) デプロイ

- **`.aux2`は再ビルドして2箇所へ配った**（実機のAviUtl2インストール先の`Plugin\NzVideomni\NzVideomni.aux2`と、リポジトリの配布用コピー`AviUtl2-Plugin\NzVideomni.aux2`）。**ビルドは必ず`build.ps1` → `deploy.ps1`を通すこと**（ninjaを直接叩くとWeb UIの埋め込みが更新されない）。
- **バックエンド側のMCPサーバーの説明文も同日に直している**（`41b5bb4`）。LTX 2.5で使えない引数の一覧からLoRAと参照動画が消えたことを反映しただけで、ツールの引数・振る舞いは1つも変えていない。

### 利用者に伝わる注意（画面には出していない）

- **スタイルLoRAは、そのLoRAの「トリガー語」をプロンプトに入れないと効きが穏やかである**（[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §74.9(2)）。
- **LTX 2.5ではLoRAがやや弱く効く傾向がある**が、**既定の強度はLTX 2.3と同じ1.0に据え置いてある**。強めたいときはLoRAタグで強度を指定する（同 §74.9(1)）。
- **輪郭線制御（canny）は鮮明な参照素材でないとエッジがほとんど出ない**（ぼかした素材では制御動画が真っ黒になる。同 §74.9(3)）。

- **状態**: 実装・機械検証・デプロイまで完了（2026-08-24）。**オーナーの実機目視はこれから**である（見る5項目は[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §74.10）。

## 90. LTX 2.5で`keep_resident`（モデル骨格の常駐）が開通した — モックの宣言を実サーバーへ合わせた（バックエンド高速化第2弾）（2026-08-25）

**やったこと**: LTX 2.5エンジンが`keep_resident`（モデルのCPU側「骨格」をジョブ間で保持して2本目以降を速くする機能）を受理するようになったのに合わせ、**フロントエンド側のモックが宣言していた「使えない機能」の一覧から`keep_resident`を外した**。フロントエンドのファイルは、バックエンドのコミット`69b58ec`（アプリ層の開通）と`d515ffa`（利用者に見える文言の追随と`.aux2`の再配布）の2本に分かれて入っている。実測の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §76、設計正本は同[`MULTI_ENGINE_DESIGN.md`](../../../Docs/MULTI_ENGINE_DESIGN.md) §5.6・§8.5。

### (1) 製品コードの変更は0行 — 画面の見え方も変わらない

- **`mockBridge.ts`**: 宣言するLTX 2.5の使えない機能一覧から`"keep_resident"`の**1語だけ**を削除した（実サーバー側の`services/engines/ltx25/adapter.py`の`HONOURED_FIELDS`に合わせるため。なぜ抜けたのかの理由もコメントで残した）。
- **画面の挙動は1つも変わらない。** 常駐のトグルは**もともと機能スコープを見ていない**——`keep_resident`は`GET /status`に可否フラグを持たない（「このマシンにメインメモリが足りるか」は利用者側の選択であってサーバーが答えられる性質ではない、という意図的な設計）ので、フロントは常に操作できる形になっている。**したがって今回の削除は「モックが実サーバーと同じことを言うようにした」だけ**であり、グレーアウトが解けたわけではない。

### (2) テスト

- **`mockBridge.test.ts`**: 宣言一覧の期待値（`arrayContaining`）から`"keep_resident"`を外し、**戻ってきたら気づけるよう`expect(features).not.toContain("keep_resident")`を1本足した**。§89と同じ理由で、**期待値から消すだけでは古いモックのままでもテストが緑で通ってしまう**ためである。
- **件数**: vitest **2,553件（2,543 passed ＋ 10 skipped・134ファイル）で0失敗＝増減なし**（宣言一覧の中身が変わっただけで、テストの本数は増えていない）。**件数を引用するときは「passedだけか、skipped込みか」を必ず添えること**（バックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §76.8(6)）。型検査 **0エラー**。

### (3) 利用者に見える文言の追随（`d515ffa`）

- **`i18n/strings.ts`の`accelKeepResidentNote`（日本語・英語とも）**: 常駐が使うメインメモリを**LTX 2.3の値だけ**で書いていた（「約20GB」）。**LTX 2.5が常駐させるのはテキストエンコーダの状態辞書ただ1つで実測7.68GiB**なので、「LTX 2.3では約20GB、LTX 2.5では約8GB」の併記へ改めた。**推奨要件の「メモリ64GB以上」はそのまま残してある**（LTX 2.5単体でも生成中のメインメモリの山が約26GiBあり、常駐をonにすると合計約34.0GiBになるため、2.5でも下がらない）。
- **`api/types.ts`のコメント**: `unsupported_features`の「執筆時点で知られている名前」の一覧から`keep_resident`を取り除き、**この一覧は契約ではなく履歴であること・読むべきなのは実際に届いた配列のほうであること**を添えた。
- **`.aux2`は再ビルドして2箇所へ配った**（実機のAviUtl2インストール先と、リポジトリの配布用コピー）。**ビルドは必ず`build.ps1` → `deploy.ps1`を通すこと。**

- **状態**: 完了（2026-08-25）。**この弾にはオーナーの目視作業が無い**（出力がビット単位で不変であることを機械で確認済みのため）。

---

## 91. LTX 2.5でSageAttention（`attention_backend`）が開通した — モックの宣言とコメントの追随（バックエンド高速化第3弾）（2026-08-25）

**やったこと**: LTX 2.5エンジンがSageAttention（注意機構の計算を量子化して速くする外部カーネル）を受理するようになったのに合わせ、**モックが宣言していた「使えない機能」の一覧から`sage_attention`を外し、型定義のコメントを追随させた**。フロントエンドのファイルはバックエンドのコミット`cb97613`（アプリ層の開通）に含まれる。実測の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §77。

### (1) 製品コードの変更は0行 — ただしグレーアウトは実際に解けた（理由は別のところにある）

- **`mockBridge.ts`**: 宣言一覧から`"sage_attention"`の**1語だけ**を削除した（理由をコメントで残し、**これでLTX 2.5について宣言に残る高速化系の名前は`prune_vaed`ただ1つになった**ことも書き添えた）。
- **設定画面のattentionの選択は、LTX 2.5でも押せるようになった。** **ただしその理由をこの一覧に帰さないこと**——**`SettingsPanel`は`unsupported_features`をそもそも読んでいない。** attentionの可否は`GET /status`の`acceleration.sage_available`**だけ**で決まり、**押せるようになったのはバックエンドがその真理値表からLTX 2.5の例外（恒久的に`false`を返す上書き）を削除したためである**（バックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §77.4・同 §77.9(5)）。**§90の`keep_resident`のように、この一覧が減っても画面が何も変わらない項目もある**——2つを混ぜて覚えないこと。
- **`api/types.ts`のコメント**: `unsupported_features`の履歴の一覧へ**「`sage_attention`も2026-08-25に抜けた（LTX 2.5がLTX 2.3のsageカーネルを動かすようになったため）」**を書き足し、正本の件数を**「仕様書 §6.10(b)、LTX 2.5では7件」から「6件」へ**直した。

### (2) テスト

- **`mockBridge.test.ts`**: 宣言一覧の期待値から`"sage_attention"`を外し、**`expect(features).not.toContain("sage_attention")`を1本足した**（§90・§89とまったく同じ止め方である）。
- **件数**: vitest **2,553件（2,543 passed ＋ 10 skipped）で0失敗＝増減なし**。

### (3) この弾だけの注意

- **`attention_backend: "sage"`は、選ぶと同じシードでも生成結果の細部が変わる。** 構図や被写体は同じままで、質感やノイズの出方が変わる（数値精度の違いによるもので不具合ではない）。**既定は`"sdpa"`のまま**で、利用者が設定画面で明示的に選んだときだけ効く。
- **効き目は動画の大きさに強く左右される。** 実測は1280×768・2クリップ×121フレームの連結生成で**1.1017倍**だが、**512×320級では効かないか、かえって遅くなる**。画面やドキュメントで「onにすれば必ず速くなる」と書かないこと。
- **画面に出る文言は1文字も変えていない**ので、**`.aux2`の作り直しは本弾では行っていない**（直近の再配布は§90の`d515ffa`）。

- **状態**: 実装・機械検証まで完了（2026-08-25）。**この弾だけは「絵が変わる」種類の高速化なので、オーナーの目視ゲート（G8）が残っている＝目視待ち**である（見るものと聴くものはバックエンド[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §2-1、依頼の全文は[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §77.6）。

> **【2026-08-30 追記】この目視ゲートG8は 2026-08-25 に合格し、SageAttentionのテーマは完結した。** 絵の見比べ4項目と音の聴き取り1項目のすべてが合格で、**既定は`"sdpa"`のまま据え置き**である。**参照先として挙げているバックエンド台帳の§2-1は、全項目合格の運用規則どおり見出しごと削除されている**（§2そのものも、最後に残っていた項目が2026-08-30に合格した時点で節ごと消えた）ので、**目視の結果はバックエンドの[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §77.10 が正本である。** **フロントエンドの製品コードは、この合格でも1行も変わっていない。**

---

## 92. LTX 2.5で撮り直し（Retake）と素材（末尾）（End source）が開通した — Editサブタブ単位のグレーアウト新設と、無効モード行き右クリックの遮断（バックエンド§3-102の準必須級2件）（2026-08-26）

**やったこと**: LTX 2.5エンジンが撮り直し（Retake）と素材（末尾）（End source）を受理するようになったのに合わせ、**Editタブを「サブタブ単位」で灰色にできる仕組みを新設し、あわせて今日でも再現する既存バグを1件直した**。フロントエンドのファイルはバックエンドのコミット`dc480f4`（C0＝土台とバグ修正）と`18a501e`（C2＝撮り直し開通に伴うモックの1語）・`a53d086`（C3＝素材（末尾）開通に伴うモックの1語と`.aux2`の再配布）の3本に分かれて入っている。実測の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §78（**訂正と補足は同 §78.13**）、設計正本は同[`MULTI_ENGINE_DESIGN.md`](../../../Docs/MULTI_ENGINE_DESIGN.md) §5.6・§8.5。

**この段は§90・§91と種類が違う。** `keep_resident`（§90）と`sage_attention`（§91）のときに触ったのはモックだけで**製品コードは0行**だったが、**本段は製品コード4ファイルとi18nに手が入っている**。理由は開通の形にある——開通後のEditタブは「撮り直しとOutpaintingの**どちらか一方でも動けば生きたまま**」になるので、**タブ単位のグレーアウトではOutpaintingだけを閉じられない**。

### (1) 製品コードの変更（C0・4ファイル＋i18n）

| ファイル | 変更点 |
|---|---|
| `shell/useBaseModels.ts` | **`editSubTabsDisabledFor`を新設**（`chainPanelsDisabledFor`と同型の純関数）。**サーバーの機能名`outpaint`とサブタブ名`outpainting`の読み替えは、この関数の中だけで行う**——下流のどこもどちらの名前も知らなくてよい形にした。両方`true`になりうるが呼び出し側は見ない（両方だめなベースモデルは`disabledModesFor`の`needsAnyOf: ["retake", "outpaint"]`でEditタブごと落ちるため、この画面がマウントされない）。**それでも正直に答えさせている**のは、関数を特別扱いのない素の表のままに保つためである |
| `modes/edit/EditSubTabs.tsx` | 有効なidにも`disabled`を許すよう**判別可能ユニオンを拡張**した。`disabled: true`はリテラルのままなので、`onChange(tab.id)`側の型の絞り込みは従来どおり効く。**無効タブにはonClickを付けない**作法（`shell/ModeTabs.tsx`準拠）と、**灰色の見た目は1種類に留めて理由はツールチップで伝える**判断も踏襲した |
| `modes/edit/EditScreen.tsx` | `subTabsDisabled`を受け取ってサブタブへ配る |
| `shell/AppShell.tsx` | **`handleRoute`の冒頭で、無効モード行きのルートを弾く**（下記(2)） |
| `i18n/strings.ts` | **36行の追加**。`edit.unavailableOnBaseModel.retake` / `.outpainting`（サブタブ1つにつき1行のツールチップ。`chained.unavailableOnBaseModel`と同じ形——Editタブ自体は生きているので、汎用の「非対応」ではどちらの半分が範囲外なのか分からなくなる）と、`modeUnsupportedByBaseModel`（右クリックを弾いたときの案内ノート）の日英。**Inpaintingのモックには意図的にツールチップを付けていない**——どのベースモデルでも「まだ作っていない」ものなので、パネルが無いこと自体が既に説明になっている |

### (2) 挙動が実際に変わるのは、既存バグの修正1点だけである

- **タイムラインの右クリックメニューはAviUtl2が描いている。** このアプリがタブを灰色にしても、Edit系の項目はメニューに出たままになる。
- これまで`handleRoute`は`disabledModes`を見ずに走り切っていたため、**`reservePlacement`が⏳仮オブジェクトをタイムラインへ書き**、`setMode("edit")`でタブを開き、**そのあとでバウンス用の`useEffect`がSingleへ戻していた**。タブ切替は取り消せても、**書き込まれた仮オブジェクトは残る**——どのジョブにも紐づかず、片付ける主体もいない。**「LTX 2.5で右クリック撮り直し→ゴミ仮オブジェクト残留」は今日でも再現する実バグだった。**
- **C0は`handleRoute`の冒頭（選択ガードよりさらに前、当然あらゆる書き込みより前）で弾く。** 弾き方は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §4の選択ガードと同じ契約——**案内ノートを出して、それ以外は何もしない**（タブ切替なし・リマウントなし・予約なし・プリフィルなし）。**Singleへ誘導はしない**: ユーザーが押したのはタブではなくタイムライン上のオブジェクトなので、拒否のうえに勝手な画面移動を重ねると驚きが2つになる。既存のバウンス`useEffect`は「すでにそのモードに居るとき」を引き続き担当し、こちらは「そもそも到着させない」を担当する、という住み分けである。
- **土台のグレーアウトそのものは、C0の時点では観測できない。** 当時のLTX 2.5は`retake`と`outpaint`の両方を非対応と申告していたので、`disabledModesFor`がEditタブごと灰色にしてしまい、サブタブに到達できなかった。**C2で`retake`が抜けた瞬間からEditタブが生き、Outpaintingのサブタブだけが灰色になる**——ここで初めて仕組みが表に出る。

### (3) モックの追随（C2・C3）とテスト

- **`bridge/mockBridge.ts`**: 宣言するLTX 2.5の使えない機能一覧から、**C2で`"retake"`・C3で`"end_source"`の1語ずつ**を削除した（§90・§91とまったく同じ作法）。**これで宣言に残るのは`two_stage_hq`／`outpaint`／`nag`／`prune_vaed`の4件**になり、**連結生成のモード名は1つも残っていない**。
- **`test/unsupportedFeatures.ts`を新設した**（C2）。期待値をテストごとに手で並べるのをやめ、**1箇所の表から引くようにした**——**期待値から消すだけでは、古いモックのままでもテストが緑で通ってしまう**（§89以来の止め方）ので、`not.toContain`の側も同じ表から作れる形にしてある。
- **件数**: vitest **2,576件（2,566 passed ＋ 10 skipped・134ファイル）で0失敗**。**基準は§90・§91の 2,553件（2,543 passed ＋ 10 skipped）**なので、**passedで+23件・skipは増減なし**である。**件数を引用するときは「passedだけか、skipped込みか」を必ず添えること**（バックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §76.8(6)）。型検査 **0エラー**。

### (4) `.aux2`は再ビルドして2箇所へ配った（C3）

- **画面に出る文言が増えている**ので（上記(1)の`strings.ts` 36行）、**`.aux2`の作り直しが要る段である**。`build.ps1 -Config Release` → `deploy.ps1`を通し、**実機のAviUtl2インストール先（`D:\For_Videos\AviUtl2\aviutl2_v2.0.54\data\Plugin\NzVideomni\NzVideomni.aux2`）と、リポジトリの配布用コピー（`AviUtl2-Plugin\NzVideomni.aux2`）の2箇所**へ配置した。
- **SHA-256は3値一致**（ビルド成果物`build/ninja-release/NzVideomni.aux2`・実機・配布用コピー）: **`cbbf71d284ba910768b4fd839d9ac54d0a793ce31fd57f9172afe245cf587254`**（**1,246,720バイト**）。
- **ビルドは必ず`build.ps1` → `deploy.ps1`を通すこと**（ninjaを直接叩くとWeb UIの埋め込みが更新されず、**画面の文言だけが古いまま配られる**）。

### (5) この段だけの注意

- **`unsupported_features`が4件に減っても、画面が灰色になるのはOutpaintingだけである。** 残る4件のうち**このWebUIが読んでいるのは`outpaint`だけ**——機能名を灰色へ翻訳しているのは`shell/useBaseModels.ts`の3つの表（`MODE_REQUIREMENTS`・`chainPanelsDisabledFor`・`editSubTabsDisabledFor`）と`batchA2vDisabledFor`だけで、そこに現れる語は`chain`／`retake`／`outpaint`／`v2v`／`a2v`／`end_source`／`reference_video`の**7語しかない**。**`two_stage_hq`・`nag`・`prune_vaed`はどの表にも無く、画面のどの部品とも結びついていない**（NAGのトグルや非蒸留の選択は`unsupported_features`を見ずに描かれ、押せば422が返る）。**§90・§91と同じ注意である**——「この一覧が減った」と「画面の何かが変わった」を、**混ぜて覚えないこと**。
- **モックのクリップ数の下限が、実APIとずれている**（本段で見つかった範囲外の不具合）。`mockBridge.ts`はクリップ1本の連結生成を許す例外として`source_video`と`source_audio`しか持っていないが、実APIの`api/models.py`は`retake`・`reference_video_id`・`end_source`も例外にしている。**したがってクリップ1本の素材（末尾）チェーンが、モックでだけ422になる。C3以前からのずれ**で、本段では直していない（テスト側は2クリップへ寄せ、理由をテストに書いた）。起票はバックエンド[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §3-116。

- **状態**: 実装・機械検証・デプロイまで完了（2026-08-26）。**オーナーの目視ゲート（G8）が残っている＝目視・試聴待ち**である（見るものと聴くものはバックエンド[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §2-1、素材は`outputs/ltx25-rtes-gate/g8_material/`）。**M8（キーフレーム印の述語）の裁定は監督裁定であって、オーナー追認待ちである**——理論から導いた述語が2シードの実測で否定され、暫定でキャリー基準を出荷構成に入れてある（バックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §78.7・§78.13(1)）。**押し込み（`git push`）はオーナー承認のうえで行う運用なので、エージェントはコミットまでで止める。**

> **【2026-08-30 追記】この目視ゲートG8は 2026-08-30 に合格し、撮り直しと素材（末尾）のテーマは完結した。** 判定は**AviUtl2の操作パネルからの実機生成**で行われ、**撮り直しは4観点**（プロンプトどおりに再生成される／継ぎ目が分からない／「Picture and sound」で音声も生成される／「Picture only (keep the original sound)」で音声はそのまま映像だけが作り直される）**が全件合格**、**逆順Chainedは3観点**（プロンプトどおり／継ぎ目で数フレームの早期収束／音声の継ぎ目がはっきり分かる）**で、後の2つはオーナー裁定で「想定の範囲内なので合格」**である。**参照先として挙げているバックエンド台帳の§2-1は、全項目合格の運用規則どおり見出しごと削除されている**（§2そのものも空になったので節ごと消えた）ので、**目視の結果はバックエンドの[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §78.14 が正本である。**
>
> **ただし本節が「オーナー追認待ちである」と書いたM8（キーフレーム印の述語）の監督裁定は、いまも追認待ちのままである**——**G8の合格宣言に、この件についての明示の言葉は含まれていなかった**（同 **§78.14(4)**）。**R-1（のりしろの一致度を判定する天井の取り方）も同じ状態である。** **「G8に合格した」と「監督裁定が追認された」を混ぜて読まないこと。**
>
> **フロントエンドの製品コードは、本追記でも1行も変えていない。** 追随したのは[`API_REFERENCE.md`](API_REFERENCE.md)の`retake`行と`end_source`節の2箇所（どちらも「G8は未了」という断りを結果へ差し替えたもの）だけである。**`.aux2`の作り直しも不要である。**
>
> **【2026-08-30 追記2・監督裁定2件（R-1・M8）にオーナーの追認が出た。上の追記を格上げする】** **上の追記は「M8（キーフレーム印の述語）とR-1（のりしろの天井の読み）の監督裁定はいまも追認待ちのままである」と書いたが、その状態は同じ 2026-08-30 のうちに解消した。** **オーナーから「監督裁定を追認するよ」という明示の言葉が出て、R-1＝同一の処理鎖を通った天井（読みB）、M8＝キャリー基準の両方が、監督裁定のとおり確定した。**（裁定の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) **§78.15**、G8 の合格宣言に含まれていなかった経緯は同 §78.14(4)、実測の正本は同 §78.4(1)・§78.7）。**本追記が、上の追記の「追認待ちのままである」という部分を上書きする。** **フロントエンドにとっての結論は3つの記述を通して一度も変わっていない——送信値も画面も1つも変えなくてよい。** 変わったのは根拠だけで、**いまは「オーナーが読みBとキャリー基準を選んだ」と読んでよい**（以前は読んではならなかった）。**したがって将来この2点を見直すときは「まだ誰も決めていない」ではなく「オーナーの裁定を覆す」話として起票すること。**

## 93. LTX 2.5で画角拡張（Outpainting）が開通した — モックの1語削除と、テストの「待ち合わせの合図」の作り替え（バックエンド§3-102の準必須級・最後の1件）（2026-08-29）

**やったこと**: LTX 2.5エンジンが画角拡張（Outpainting）を受理するようになったのに合わせ、**モックの宣言から`"outpaint"`の1語を落とし、それによって前提が崩れるテストを書き換えた**。フロントエンドのファイルはバックエンドのコミット`9997db1`（C0-FE＝下地）と`22203cc`（C2-FE＝開通に伴う追随）の2本に分かれて入っている。実測の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §79、設計正本は同[`MULTI_ENGINE_DESIGN.md`](../../../Docs/MULTI_ENGINE_DESIGN.md) §5.6・§8.5。

**この段は§92と種類が違い、§90・§91の側に戻っている。** §92（撮り直しと素材（末尾））では**製品コード4ファイルとi18nに手が入った**——Editタブをサブタブ単位で灰色にできる仕組みそのものを新設したからである。**本段の製品コードの変更は0行**で、触ったのは**モック1ファイルとテスト4ファイルだけ**である。**§92でその仕組みを先に作り終えていたことが、そのまま本段の配当になっている。**

**`.aux2`の再ビルドも不要だった。** 画面に出る文言（`i18n/strings.ts`）に1文字も触っていないためである（§78.13(3)の申し送り「**フロントエンドの文言に1文字でも触った段は、`.aux2`の版が動く段である**」の裏返しで、**触っていない段では版は動かない**）。

### (1) モックの追随（C2-FE）

- **`bridge/mockBridge.ts`**: 宣言するLTX 2.5の使えない機能一覧（`MOCK_UNSUPPORTED_FEATURES.LTX25`）から**`"outpaint"`の1語**を削除した（§90・§91・§92とまったく同じ作法）。サーバーの`UNSUPPORTED_FEATURES`が3件になったので、**ここが残っていると動くモードのためにEditタブの「画角拡張」サブタブを灰色にし続けることになる。**
- **単発の`/generate`には、このフィクスチャに項目ごとの拒否ループが元々無い**ので、`retake`のときと同じく**1行の削除で済んでいる**（意味の無い422テストは書いていない）。
- **これで宣言に残るのは`two_stage_hq`／`nag`／`prune_vaed`の3件**になり、**モード名は1つも残っていない**。**そしてこの3件は、どれもこのWebUIが読んでいない**——機能名を灰色へ翻訳しているのは`shell/useBaseModels.ts`の3つの表（`MODE_REQUIREMENTS`・`chainPanelsDisabledFor`・`editSubTabsDisabledFor`）と`batchA2vDisabledFor`だけで、そこに現れる語は`chain`／`retake`／`outpaint`／`v2v`／`a2v`／`end_source`／`reference_video`の**7語しかない**。**したがって、いま画面で灰色になるタブ・サブタブ・カードは1つも無い。**

### (2) テストの「待ち合わせの合図」を作り替えた（C0-FE と C2-FE）

**ここが本段でいちばん手間のかかった部分である。** `AppShell.featureScope.test.tsx`は、**「Outpaintingサブタブが灰色になったこと」を、LTX 2.5への切替が完了したことの待ち合わせの合図に流用していた**（5箇所）。開通すると実在のフィクスチャが`outpaint`を灰色にしなくなるので、**この待ちは永久に成立しなくなる。**

- **C0-FE（`9997db1`）で、その5箇所を先に合成注入へ置き換えた。** `withExtraUnsupportedFeatures(bridge, "LTX25", ["retake"])`で**存在しない灰色化を作り、それを合図に使う**形である。**変更前後どちらのモック状態でも緑**なので、開通と切り離して安全に先入れできる——**これが「下地を先に入れる」段の意味**である。
- **C2-FE（`22203cc`）で、残りのテストを直した。** LTX 2.5は`retake`に続いて`outpaint`も失ったため、**実在の一覧ではタブもサブタブも1つも灰色にならない**——つまり**「応答を読んだ上で何も灰色にならない」と「応答をそもそも読んでいない」を区別する手段が無くなる。** そこで既定の作法どおり合成注入で合図を作り、**その合図が出た後に、実在の`outpaint`が灰色でないことを見る**形へ移した（増分の証拠は合図の後で初めて意味を持つ）。
  - `AppShell.featureScope.test.tsx`: タブ据え置きの1本と2.3への往復の1本を合成注入へ。加えて、**Editタブ全体の灰色化を扱う「別のモードへ弾かれる」1本**は実在の`outpaint`で半分を賄っていたので、注入を`["retake", "outpaint"]`の**2語**にした（**Editは両方のサブモードが拒否されて初めて落ちる**ため）。
  - `App.editRoute.test.tsx`: 右クリック経路のゲート3本も同じ理由で2語注入へ。ヘルパ1箇所の変更で足りている。
  - `useBaseModels.test.ts`: 3系統を更新。**一覧が「無いもの」の列挙だけにならないよう肯定の確認を1つ残し**、`editSubTabsDisabledFor`の表には**今日の一覧**の行を足した（**どちらのサブタブも灰色にならない、が今の答えである**）。純粋関数を手渡しの一覧で叩く行は**メカニズムの試験**なので、そのままにしてある。
  - `mockBridge.test.ts`: 公開一覧から`outpaint`を外し、**否定側の確認を1行足した**——`arrayContaining`から消すだけでは、フィクスチャが古いままでも緑になってしまう（§89以来の止め方）。

### (3) 件数

- **vitest 134ファイル・2,567件PASS・10スキップ・0失敗**（＝実行数 2,577）。**基準は§92の 2,576件（2,566 passed ＋ 10 skipped）**なので、**passedで+1件・skipは増減なし**である。**件数を引用するときは「passedだけか、skipped込みか」を必ず添えること**（バックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §76.8(6)）。
- **型検査 0エラー**（`npm run typecheck`＝`tsc -b --noEmit`）。

### (4) この段だけの注意

- **`editSubTabsDisabledFor`を消してはいけない。** いまはどちらのサブタブも灰色にしないので**画面上は何も起きていないように見える**が、**将来また片方だけを断るベースモデルが載ったときにそのまま効く仕組み**である。`useBaseModels.test.ts`には「今日の一覧では両方とも灰色にならない」という行を足してあるので、**うっかり消せばそこが落ちる。**
- **「この一覧が減った」と「画面の何かが変わった」を混ぜて覚えないこと**（§90・§91・§92と同じ注意）。本段では**実際に画面が変わった**（Editタブの「画角拡張」サブタブの灰色が解けた）が、**残る3件`two_stage_hq`・`nag`・`prune_vaed`はどの表にも無く、画面のどの部品とも結びついていない。**
- **`useOutpaintForm.ts`は1行も触っていない。** このフォームは元から`in-outpainting`の制御アダプタを固定で送る作りで、**それがそのままバックエンドの受理条件**だった（`GET /loras`で`kind: control`／`preprocess: none`として見えることが、ボタンを活性にするためのデータ条件でもある）。**前処理つきのアダプタを送るとAPI層が`OUTPAINT_PREPROCESS_CONFLICT`で弾く**——緑のセンチネルにcanny等をかけることになるためである。

- **状態**: 実装・機械検証まで完了（2026-08-29）。**製品コードの変更が0行なので`.aux2`の再ビルドと再配布は行っていない。** **オーナーの目視ゲート（G8）が残っている＝目視・試聴待ち**である（見るものと聴くものはバックエンド[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §2-2、素材は`outputs/ltx25-outpaint-gate/g8_material/`）。**オーナーの裁定をお願いしたい項目が2つある**（ぼかしの既定値5/2対5/6と、stage-2の2ステップ対3ステップ。**どちらも監督の推奨は既定の維持**）。**押し込み（`git push`）はオーナー承認のうえで行う運用なので、エージェントはコミットまでで止める。**

> **【2026-08-30 追記】この目視ゲートG8は 2026-08-29 に合格し（正しくは取り直し版のG8′）、画角拡張のテーマは完結した。** **裁定をお願いしていた2点も同日に決着している**——**ぼかしの既定値は5/2のまま、stage-2は2ステップのまま**で、オーナーの明示の裁定は「どちらも今のままでよい」であった（[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) **§79.12**）。**これは「決めていないから今のまま」ではなく「オーナーが今のままを選んだ」である**ので、将来見直すならその裁定を覆す話として起票すること。**参照先として挙げているバックエンド台帳の§2-2は、全項目合格の運用規則どおり見出しごと削除されている**（§2そのものも節ごと消えた）ので、**目視の結果は同 §79.11 が正本である**（本節が見た素材は脱緑より前のものなので、**§79.10 の取り直しを経た §79.11 のほうを読むこと**）。**フロントエンドの製品コードは、この合格でも1行も変わっていない。**

---

## 94. 画角拡張の緑かぶりを取り除いた（「脱緑ブレンド」）— §66で「採らなかった指摘(1)」との和解（フロントエンドの変更は0行）（2026-08-29）

**フロントエンドのコードは1行も変えていない。** 本節は**記録のためだけの1節**である。**§66（2026-08-09）で「敵対的レビューの指摘のうち採らなかったもの」として書き残した(1)が、2026-08-29に実質的に採用されたので、その相互参照をここに置く。** 実装と実測の正本はバックエンドの[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) **§79.10**である。

### (1) 何が起きたのか

**画角拡張の出力には、「新しく描かれた領域の全体が淡く緑がかる」という現象があった。** 2026-08-29、これを**約25行の修正**で取り除いた——**ブレンドへ渡すキャンバスの余白（pad帯）を、その同じコマの生成画素で先に置き換える**というものである。**LTX 2.3 と LTX 2.5 の両方に同時に入れ、両エンジンで実GPUゲートを取り直して全項目合格した。**

**副作用は無かったどころか、残すはずだった元映像の再現もよくなった**（LTX 2.5 の基準腕で元素材との一致度が 36.59 → **38.11 dB**、LTX 2.3 の W2 ゲートで 33.32 → **37.81 dB**）。**継ぎ目の機械判定の上限（2.0）をはじめ、判定線は1つも動かしていない。**

### (2) §66「採らなかった指摘(1)」との和解

§66にはこう書いてあった。

> **敵対的レビューの指摘のうち採らなかったもの**: (1)「ブレンドの相手に緑キャンバスを使うと低周波で緑がにじむので、緑でない画を用意すべき」——**公式が緑を使っており**、代案は拡張帯で両者が一致してブレンドが恒等写像に退化する。レビュー自身も追試して「代案は間違い」と結論を翻したので、公式どおりにした。

**正直に書く。当時のレビューは、症状の診断については正しかった。**「低周波で緑がにじむ」は、まさに今回実測した現象である——**ラプラシアンピラミッドのマスクは、いちばん粗い段（level 6）では pad帯の奥でも 0.76 にしかならず、1.0 にならない。** つまり**キャンバスの低周波の色は、生成領域の奥まで必ず混ざる。** そして混ざっていたのが公式センチネル色 `#66FF00` だった。

**却下の理由のほうが、いま見ると成り立たない。**「拡張帯で両者が一致してブレンドが恒等写像に退化する」——**これは事実だが、欠陥ではない。** 継ぎ目の処理は **keep矩形の内側の膨張帯**（§66で「境目から内側へおよそ150画素」と書いたあの帯）で行われており、**そこでは元映像と生成画像という別々の絵が両オペランドに載ったままである。** 恒等になるのは**余白の奥だけ**で、**そこには元々調停すべきものが無い。** **退化を「機能が失われる」と読んだのが、当時の取り違えであった。**

### (3) ただし「公式が緑を使っている」のほうは、いまも正しい

**混同しやすいので、はっきり分けて書いておく。緑には出番が2つあり、今回触ったのは片方だけである。**

| どこの緑か | 何のためか | 今回の扱い |
|---|---|---|
| **モデルに渡す条件付け**（緑のキャンバスをIC-LoRAの参照として符号化する） | モデルは `#66FF00` を見て「ここを描け」と読む。**公式LoRAはその色ちょうどを描き替えるよう学習されている**（§66の「緑は可逆RGBで書かないと保てない」の節を参照） | **触っていない。緑のままである。センチネルの学習は1ビットも損なわれていない** |
| **画素空間のブレンドに渡す相手**（1回目・2回目のブレンドのキャンバス側オペランド） | 元映像を継ぎ目でなじませて戻すため | **余白の部分だけを、その同じコマの生成画素へ置き換えた** |

**この2つは、パイプラインの上で何分も離れている。** 条件付けは生成の入り口、ブレンドは復号したあとである。**「緑でない画をモデルに見せる」という話ではまったくない**ので、公式ワークフローとの整合はそのまま保たれている。

### (4) 画面への影響

**無い。** 変えたのはバックエンドのブレンド処理だけで、**API契約・送信する値・画面の部品・文言のいずれも変わっていない。** `webui/src/modes/edit/` 配下は1ファイルも触っていない。**「マスクブラー」スライダー（§67で新設したもの）の意味も変わっていない**——ぼかし幅は依然として継ぎ目のやわらかさを決めるつまみである。

**ただし、そのつまみの「役目」が1つ減った。** §67では、緑が残ったときに利用者が自分で対処する手段としてこのスライダーを露出した経緯があったが、**緑かぶりのほうが根治したので、いまは純粋に画質の好みで選ぶつまみになった。** **既定（5/2）を変える予定は無い**——ぼかしを広げると継ぎ目はやわらぐが、そのぶん元映像の外周が余計にブレンドに食われる（一致度が 38.11 → 33.41 dB）ためで、**監督の推奨は既定の維持**である。この点はオーナーの目視ゲートの裁定材料になっている（バックエンド[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §2-2 の C1）。

### (5) 状態

**フロントエンドは実装・検証・デプロイのいずれも不要**（変更が0行のため）。**バックエンド側は実装コミット `a7dd474` と文書コミット1本で、いずれも押し込みはオーナー承認待ちである。** **オーナーの目視ゲートは、出力が全面的に変わったため取り直し（G8′）になり、素材も全面的に作り直した**——チェックリストはバックエンド[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §2-2、素材は`outputs/ltx25-outpaint-gate/g8_material/`（**前回からの変更点だけを書いた `README_G8prime_DELTA.md` が同フォルダにある**）。**§93の「状態」行にある「オーナーの目視ゲート（G8）が残っている」は、本節の時点ではG8′に置き換わっている。**

## 95. 画角拡張（Outpainting）の目視ゲート G8′ が合格した — **「マスクブラー」の既定 5/2 は据え置きで確定**（フロントエンドの変更は0行）（2026-08-29）

**フロントエンドのコードは1行も変えていない。** 本節も**記録のためだけの1節**である。**§93（LTX 2.5 での画角拡張の開通）と §94（脱緑ブレンド）が「オーナーの目視ゲート待ち」で終わっていたので、その決着をここに書き足す。** 実装と実測の正本はバックエンドの[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §79・§79.10、**目視の結果の正本は同 §79.11**である。

### (1) 結果

**2026-08-29、オーナーの目視ゲート G8′ が合格した。** **「緑かぶりが綺麗に消えて、自然な色味になっていた」**というのがそのときの言葉である。**判定は2通りの確かめ方の両方で行われている**——**①脱緑後に作り直した素材の目視**（`outputs/ltx25-outpaint-gate/g8_material/`。本題は新設の `F_before_after_degreen.png` ＝改修前と改修後の並置）と、**②AviUtl2 の操作パネルからの実機生成**である。**これをもって画角拡張の LTX 2.5 開通は完結した。**

**§93 と §94 の状態行は当時のまま残す**——本書は追記専用の記録簿なので、**目視の結果については本節が正本である。**

### (2) フロントエンドに効く決着が1つある — **「マスクブラー」の既定は 5/2 のまま**

**§94(4) で「役目が1つ減ったが、既定（5/2）を変える予定は無い」と書いたつまみについて、正式な裁定が出た。**

**目視ゲートには、見比べとは別に、オーナーの裁定をお願いしていた項目が2つあった。**——①**ぼかしの既定値**（現行の出荷値 5/2 と、公式ワークフローの値 5/6 のどちらを既定にするか）と、②**stage-2 のステップ数**（2ステップと3ステップのどちらを既定にするか）である。**どちらも監督の推奨は「既定の維持」で、合格の宣言に既定を変えるようにという指示は含まれていなかった。したがって既定は両方とも据え置きである。**

**フロントエンドにとっての意味は「送信値を1つも変えなくてよい」ということである。** `webui/src/modes/edit/` の「マスクブラー」スライダーの既定は **5/2 のまま**で、バックエンドの `api/models.py` の既定値も変わらない。**もし将来オーナーが既定を見直すことになれば、それはアプリ層（既定値とフロントエンドの送信値）の改修になるので、別テーマとして起票する**——本段の契約は「リクエストが運んでくる値をそのまま使う」なので、切り出しはいつでもできる。

### (3) 画面のグレーアウトの状態 — 変化なし

**本節では画面に何も起きていない。** グレーアウトの表（`webui/src/shell/useBaseModels.ts` の3つ）に現れる語は `chain` / `retake` / `outpaint` / `v2v` / `a2v` / `end_source` / `reference_video` の7語のままで、**`outpaint` が `unsupported_features` から抜けたのは §93（2026-08-29 の開通）の時点である。** **いま画面で灰色になるタブ・サブタブ・カードは1つも無く、Edit タブの2つのサブタブは両方とも生きている。** **`editSubTabsDisabledFor`（§92 で新設した仕組み）は今日どちらも灰色にしないが、将来また片方だけを断るベースモデルが載ったときに効くので消してはいけない。**

### (4) 状態

**フロントエンドは実装・検証・デプロイのいずれも不要**（本節も変更が0行のため）。**押し込み（`git push`）は済んでいる**——§94 が「オーナー承認待ち」と書いていた3本（`a7dd474` 実装／`f3c1102` バックエンド文書／`f5fcfa8` 本書 §94）は、**2026-08-29 にオーナーが手で押し込み済みである**（`main` と `origin/main` はともに `f5fcfa8`）。**モノレポなので、バックエンドとフロントエンドの押し込み状態は `git log origin/main..main` の1回で確認できる。**

**バックエンド側の台帳では、目視ゲートのチェックリストを置いていた[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §2-2 が、運用規則（全項目が合格して空になった節は見出しごと削除する）どおり削除された。** **§2 そのものは残っている**——**撮り直し（Retake）と素材（末尾）（End source）の目視ゲート G8〔同 §2-1〕がまだ未了だからである**（そちらはフロントエンド側では §92 が対応する段である）。

**追記（2026-08-29・敵対的文書レビューの確定指摘による訂正）**: **上の (2) の見出しと本文にある「正式な裁定が出た」「据え置きで確定した」という書き方は行き過ぎである。本追記が (2) の当該2文を上書きする。** 正しくは——**C1（ぼかしの既定値）と C2（stage-2 のステップ数）について、オーナーの明示の裁定は存在しない。** 合格の宣言は総括であって、この2点についてどちらを選ぶという言葉は無かった。**したがって「既定を変える指示が無かったため、据え置きのままである」が事実の正確な言い方であり、「オーナーが 5/2 と2ステップを選んだ」と読んではならない。** **結論（送信値を1つも変えなくてよい・既定は 5/2 と2ステップのまま）は変わらないが、その根拠は「裁定」ではなく「指示が無かったこと」である。** **将来この既定を見直すときに「一度オーナーが決めた」という前提で議論を始めないこと。** 経緯はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) **§79.11(3)** が正本である（同節はもともとこの区別を正しく書いてあり、下流の本書・[`MULTI_ENGINE_DESIGN.md`](../../../Docs/MULTI_ENGINE_DESIGN.md)・[`Videomni_Backend_Specification.md`](../../../Videomni_Backend_Specification.md)・[`NEXT_SESSION_HANDOFF.md`](../../../Docs/NEXT_SESSION_HANDOFF.md) の4文書だけが語調を強めていた。他の3つは生きた文書なので直接訂正した）。

**追記2（2026-08-29・C1／C2にオーナーの明示裁定が出た。上の「追記」を格上げする）**: **上の追記は「オーナーの明示の裁定は存在しない／据え置きは指示が無かった結果である」と書いたが、その状態は同じ 2026-08-29 のうちに解消した。** **オーナーから明示の裁定が出て、「どちらも今のままでいい」——C1（ぼかしの既定値）は 5/2 を維持、C2（stage-2 のステップ数）は 2 ステップを維持——と決まった。** **本追記が、上の追記の「明示の裁定ではない」という部分を上書きする**（§95(2) 本文の「正式な裁定が出た」という当初の書き方は、結果として正しい側に着地したことになるが、**間に「まだ裁定は無い」という時期が実在した**ので、経緯としては 追記 → 追記2 の順に読むこと）。**フロントエンドにとっての結論は3つの記述を通して一度も変わっていない——送信値は1つも変えなくてよい。** 変わったのは根拠だけで、**いまは「オーナーが 5/2 と2ステップを選んだ」と読んでよい**（以前は読んではならなかった）。**したがって、将来この既定を見直すときは「まだ誰も決めていない」ではなく「オーナーの裁定を覆す」話として起票すること。** **なお裁定には「経緯が文書で振り返れることを確認したうえでの了承」という条件が付いている**ので、`webui/src/modes/edit/` の「マスクブラー」まわりの説明や既定値を触るときは、**判断材料が追える状態を壊さないこと。** 裁定の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) **§79.12**、判断材料は同 §79.11(3)、測定は同 §79.10 である。

---

## 96. LTX 2.5で非CFGネガティブプロンプト（NAG／VSF）が開通した — モックの宣言を1語外し、そこで見つかったchain側の潜在バグを直した（バックエンド§3-102の実験的1件）（2026-08-30）

**結論から言うと、バックエンドがLTX 2.5でもネガティブプロンプト（「こういう絵にはしないでほしい」を言葉で指定する機能）を通すようになったので、モックの宣言から`nag`を外した。フロントエンド本体の変更は0行である。** ただし**外しただけでは済まなかった**——`nag`を外すと、モックのchain側にあった「フィールド単位で断るループ」に**発火できる行が1つも無くなる**ので、仕組みごと静かに検査されなくなってしまう。そこで**`pipeline`と`vae_mode`の2行を、この表へ初めて追加した。**

### 96.1 何が変わったのか

**バックエンド側**: `unsupported_features`が**3件 → 2件**（残るのは`two_stage_hq`／`prune_vaed`）。7つのフィールド（`nag_enabled`／`negative_prompt`／`nag_scale`／`nag_tau`／`nag_alpha`／`neg_method`／`vsf_scale`）が**3つの分類から同時に「動作」へ**移り、**「従属」の分類は単発・連結とも空になった**。実測の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) **§80**。

**フロントエンド側**: **製品コードは0行**。触ったのは`src/bridge/mockBridge.ts`と`src/bridge/mockBridge.test.ts`だけである。

**画面は1ドットも変わらない。** `nag`は、機能名をグレーアウトへ翻訳している3つの表（`shell/useBaseModels.ts`の`MODE_REQUIREMENTS`・`chainPanelsDisabledFor`・`editSubTabsDisabledFor`）と`batchA2vDisabledFor`の**どこにも現れない語**である。NAGのトグルは`unsupported_features`を見ずに描かれていて、押せば422が返る、という関係だった。**§90・§91・§93と同じ注意である**——**「この一覧が減った」と「画面の何かが変わった」を、混ぜて覚えないこと。** 本段は**前者だけ**が起きた。

### 96.2 `isSet`述語の導入（型を変えた1箇所）

`MOCK_CHAIN_FEATURE_FIELDS`は「どのフィールドがどの機能名に属するか」の対応表で、モックのchainエンドポイントはこの表を回して422を返す。**既存の5行（`source_video`／`source_audio`／`reference_video_id`／`loras`／`nag_enabled`）は、どれも「`null`か・空配列か・`false`か」で「利用者が求めたか」を判定できた**ので、共有の述語`isChainFieldSet`ひとつで足りていた。

**新しく足した2行はそうではない。** `pipeline`の既定値は`"distilled"`、`vae_mode`の既定値は`"default"`で、**どちらも空でない文字列**である。`isChainFieldSet`は**非空文字列を「要求された」と読む**ので、そのまま足すと**あらゆるchainリクエストが422になる**。

そこで**要素の型に`isSet?`（この行だけの判定関数）を足し**、ループを`(isSet ?? isChainFieldSet)(req[field])`にした。**変更は3箇所**——型・行・ループである。

```ts
{ field: "pipeline",  feature: "two_stage_hq", isSet: (v) => v != null && v !== "distilled" },
{ field: "vae_mode",  feature: "prune_vaed",   isSet: (v) => v != null && v !== "default" },
```

**この罠は実際に踏みうるものだったので、テストで固定した**——「既定値は素通し」のテストに`pipeline: "distilled"`と`vae_mode: "default"`を足してある。**述語を付け忘れた実装では、このテストが落ちる。**

### 96.3 ★ここで見つかったchain側の潜在バグ（単発側は無傷である）

**`two_stage_hq`と`prune_vaed`は、モックが「使えない」と宣言していたのに、chainの拒否ループには行が無く、実際には通っていた。** つまり**モックの宣言と、モックの振る舞いが食い違っていた。**

**単発の`/generate`側には、そもそもフィールド単位の拒否ループが無い**（このモックは単発生成では何も断っていない）。**だからあちらには何の間違いも無かったし、何も直していない。** **これはchain側だけの話である**——この非対称は、`retake`が`MOCK_CHAIN_FEATURE_FIELDS`に行を持っていなかったときの§92の注記と同じ構造である。

**なぜ今まで表に出なかったのか**: `two_stage_hq`も`prune_vaed`も、**WebUIから送る経路が無い**（`pipeline`は常に`"distilled"`、`vae_mode`は設定画面から変えられるがLTX 2.3でしか触らない前提だった）。**モックのテストも、宣言と振る舞いの一致までは見ていなかった。** 今回`nag`が抜けて**「発火できる行がゼロになる」という別の問題に気づいたことで、ついでに見つかった。**

### 96.4 テストの変更

- **`CASES`（refuse/pass-throughの2ループを駆動する表）から`nag`行を外し、`prune_vaed`行を置いた。** **空にはしていない**——空にすると**LTX 2.5の422も、LTX 2.3の素通りも、両方のループが黙って引退する**。「行が消えた」ことは証拠にならないので、**代わりの題材を必ず1つ残す。**
- **NAG受入とVSF受入の202を各1本追加した。** 表から行が消えたことは証拠にならず、**実際にそのフィールドを載せたリクエストが202で返ることだけが証拠になる**（§92・§93と同じ作法）。
- **「検証より前に拒否する」「拒否してもジョブを作らない」の題材2本を`vae_mode`へ移した**（相方の要らないフィールドなので、本体が素のchainに1値を足しただけになる）。
- **`unsupported_features`の期待値から`"nag"`を外し、`not.toContain("nag")`を足した。** **定義だけ直してもテストは緑のまま通ってしまう**ので、2箇所を対で固定する——これは§90以来この節が守っている、このファイル自身の規律である。
- **`useBaseModels.test.ts`の`"nag"`には触っていない。** あれは**この一覧のミラーではなく合成データ**で、グレーアウトの純関数が「知らない機能名を渡されても壊れない」ことを見るためのものである。

### 96.5 状態

**vitestは134ファイル・2,569件が0失敗**（10 skip）、**`npm run typecheck`はエラー0**。**コミットは`2bc864f`（C5-FE）で、本節を含む文書コミットとあわせて未押し込みである**（押し込みはオーナー承認のうえで行う運用）。

**バックエンド側にはオーナーの目視ゲートG8が残っている**（素材は`outputs/ltx25-nagvsf-gate/g8_material/`に6本。チェックリストはバックエンド[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §2-2）。**フロントエンドの作業は本段で完了している。**

> **【2026-08-30 追記】この目視ゲートG8は同じ2026-08-30に合格し、NAG／VSFのテーマは完結した。** **参照先として挙げているバックエンド台帳の§2-2は、全項目合格の運用規則どおり見出しごと削除されている**ので、**目視の結果はバックエンドの[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §80.10 が正本である**。**フロントエンドのコードは、この合格でも1行も変わっていない。**

## 97. 未導入のベースモデルの案内を、導入バッチの名指しへ戻した — `baseModelInstaller()`の復活（バックエンド§3-111）（2026-08-30）

**結論から言うと、`install-LTX25.bat`が実在するようになったので、「LTX 2.5（未導入）」を選んだときのトーストを、バッチ名を名指しする文面へ戻した。** 変更したのは`useBaseModels.ts`（関数を1本復活）・`strings.ts`（英日の文面を1つずつ）・`AppShell.tsx`（呼び出し1行）と、そのテスト3本＋モックのコメント1箇所である。

### 97.1 なぜ汎用文だったのか、なぜ戻せるのか

**この文面は、もともとバッチ名を名指ししていた。** それを`cda3006`（2026-08-22・§3-98の実装）で汎用文へ畳んだのは、**当時`install-LTX25.bat`が存在しなかった**からである——**存在しないファイルの実行を促す案内は出せない。**

**2026-08-30にバックエンドがそのバッチを作った**（バックエンド[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §3-111、実測は[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §82）。**だから畳んだ理由が消え、名指しへ戻せる。** 設計の正本は[`MULTI_ENGINE_DESIGN.md`](../../../Docs/MULTI_ENGINE_DESIGN.md) §6.2ガード(2)である。

### 97.2 バッチ名は記述子のidから組み立てる

```ts
export function baseModelInstaller(id: string): string {
  return `install-${id}.bat`;
}
```

**規約は`install-<ID>.bat`で、`ID`は記述子の`id`（＝`models/<ID>/`のフォルダ名）である。** 名前の表を持たずidから導くので、**ベースモデルが増えてもフロントエンドは何も足さなくてよい。**

**`BaseModelSwitchOutcome`の型は広げていない。** `AppShell`が受け取る`outcome`には既に`id`が入っているので、`notInstalled(outcome.displayName, baseModelInstaller(outcome.id))`と書けば足りる——**型を広げるのは、呼び出し側が持っていない情報を運ぶときだけである。**

**LTX 2.3には対応するバッチが無い**（`setup.bat`が同梱するため）。**この案内へ到達するのは、利用者が2.3の重みを自分で消した場合だけ**で、そのとき存在しない`install-LTX23.bat`を名指しすることになる。**これは許容済みの副作用である**——例外を1つ増やして規則を複雑にするより、規約を単純に保つほうがよいという判断による。

### 97.3 文面 — 「開き直してください」を必ず入れる

- 日: `${displayName}はまだ導入されていません。バックエンドのフォルダにある${installer}をダブルクリックして重みファイルを取得したあと、この画面を開き直してから選び直してください。`
- 英: `${displayName} is not installed. Double-click ${installer} in the backend folder to download its weight files, then reopen this screen and pick it again.`

**「開き直してから」は飾りではない。** ベースモデルの一覧を取り直す`refresh()`は、**マウント時と切り替え成功後にしか呼ばれない**。つまり画面を開いたままバッチを走らせても、一覧はいつまでも「未導入」のままである。**この案内文が、その仕様を利用者に埋め合わせている唯一の場所である。**

### 97.4 テスト

- `useBaseModels.test.ts`: `baseModelInstaller("LTX25")`が`"install-LTX25.bat"`になることを1本追加。**既存の`toEqual`には触っていない。**
- `AppShell.toolVersion.test.tsx`: トーストが`install-LTX25.bat`を**含む**ことを追加した。**既存の「`LTX 2.5 is not installed`を含む」はそのまま残してある**——文面を作り替えたときに、名指しの追加と本文の維持を両方固定するためである。テスト名も「setup guidance」から「the installer name」へ改めた。
- `strings.test.ts`: `notInstalled`が2引数になったのに合わせ、英日が異なることの確認を2引数で呼ぶ形へ。
- `mockBridge.ts`: `ltx25Install: "none"`の説明コメントが「setup guideの案内が出る」と書いていたので、`install-LTX25.bat`の案内へ直した。**振る舞いは1バイトも変えていない。**

### 97.5 状態

**vitestは134ファイル・2,570件が0失敗**（10 skip）、**`npm run typecheck`はエラー0**、**`npm run lint`もエラー0**（既存の警告のみ・新規の警告は無い）。**`npx tsc --noEmit`は偽合格になるので使っていない。****webuiはaux2に埋め込まれるので`scripts/build.ps1 -Config Release`から作り直し、`scripts/deploy.ps1`で実機とリポジトリ配布コピーの2か所へ配布済みである。**

## 98. モデル読み込み中の即時バッジとGenerate凍結の一本化（バックエンド§1-24）（2026-08-31）

### 98.1 結論

**ヘッダーの「モデル読み込み中…」バッジを`GET /status`のポーリング頼みから、自分が出した`POST /pipeline/load`の往復そのものへ差し替えた。** これに伴い、全タブのGenerateボタン・ヘッダーのベースモデル選択・Settingsの「読み込み」ボタンとカテゴリ選択を「サーバーが仕事中（ジョブ実行中 or モデル読み込み中）なら不可」という1本のルールへ統一した。ポーリング周期は10秒（2.5秒だった旧値から復帰）。バックエンドの`/generate`・`/generate/chain`はジョブ作成前に同期で409 `PIPELINE_LOADING`を返すようになった（バックエンド`Videomni_Backend_Specification.md` §6.9(f)）。台帳は`PENDING_TASKS.md` §1-24。

### 98.2 背景

**バッジは最大2.5秒遅れて点き、Generateボタンはそもそもモデル読み込み中を見ていなかった。** バッジは`GET /status`のポーリング（2.5秒周期）でしか点かないため、切り替え開始から最大2.5秒バッジが出ない窓があった。しかもSingle／Chained／Edit・バッチA2V／i2v-longのGenerateボタンはどれも`serverStatus`を読んでおらず、`GET /jobs`由来の`hasActiveJob`だけで凍結していたので、バッジが出ていてもGenerateを押せてしまい、押すとバックエンドはジョブを202で受理したあとジョブスレッド内で`PIPELINE_LOADING`により失敗させ、台帳に失敗ジョブが残っていた。

**2026-08-20当時（§87(2)）は、いま採った方式を却下していた。** モデル切り替えは常にフロントエンド自身が出す同期の`POST /pipeline/load`（発行元はヘッダーの`shell/useBaseModels.ts`とSettingsの`shell/useModels.ts`）なので、フロントエンドは開始と終了を0ミリ秒で知っている。当時もこの事実は分かっていたが、§87(2)は3つの選択肢のうち(c)「`POST /pipeline/load`を出した側が即時refresh＋高頻度化する」案を「Settingsのパネルとヘッダーのバッジという別々の持ち主の間に新しい配線を1本増やすことになり、例外を増やさないという方針に反する」として却下し、代わりにポーリング周期を2.5秒へ縮める(a)案を採っていた。

**今回それを覆せたのは、配線先が変わったからである。** 生成ゲートの読み手（Single／Chained／Edit）はすでに`useJobsContext()`を直読みしていたので、旗の置き場を「パネルとヘッダーという別々の持ち主の間」ではなく、両方の発行元がもとから読み書きしている**単一の持ち主`JobsContext`**にした。発行元は自分の`POST /pipeline/load`を`trackPipelineLoad`に渡すだけで、読み手（ヘッダー・Settings・全タブの生成ボタン）は同じ場所の`pipelineLoading`／`serverBusy`を読むだけになる。**§87(2)が懸念していた「新しい配線」自体が要らなくなった**ので、例外は増えていない——却下の理由がそのまま消えた、という関係である。

### 98.3 実装差分

- **`jobs/JobsContext.tsx`**: `pipelineLoading`（自分が出した`POST /pipeline/load`が飛行中かどうかの真偽値）と`serverBusy`（`hasActiveJob || pipelineLoading`）を追加し、`trackPipelineLoad`（`<T>(promise: Promise<T>) => Promise<T>`。旗を立てて`promise.finally`で下ろすだけの素通し）を公開した。`hasActiveJob`はコンテキストの公開値から外した（`jobs/useJobsPoll.ts`側は内部関数へ降格）。
- **`shell/useBaseModels.ts`**: `trackLoad?: TrackPipelineLoad`を追加し、`switchBaseModel`の本体を包む。**`switching`を削除**——読み手は`serverBusy`に一本化された。
- **`shell/useModels.ts` / `shell/ModelsPanel.tsx`**: 同じ`trackLoad?`を`useModels`に追加。`ModelsPanel`は`useJobsContext()`を直読みし、`busy`を`serverBusy`にした。
- **`modes/single/useServerStatus.ts`**: `UseServerStatusDeps.localLoading?: boolean`を追加。真なら`GET /status`の応答を待たずに`{ kind: "loading-models", status: null }`を返す。`DEFAULT_INTERVAL_MS`を10秒へ戻した。立ち下がり（true→false）で`check()`を1回叩く。
- **ゲートの読み替え**: `shell/AppShell.tsx`のドロップダウン無効化・右クリック→撮り直しの遮断、`modes/single/SingleScreen.tsx`・`modes/chained/ChainedScreen.tsx`・`modes/edit/EditScreen.tsx`の各Generateボタンを、すべて`serverBusy`（または`hasActiveJob`からの改名）へ揃えた。**`modes/edit/EditScreen.tsx`の画角拡張（Outpainting）のGenerateには、従来凍結が無かったので新規に追加した**（撮り直しと同じ作法）。
- **バッチ**: `modes/batch/BatchSection.tsx`・`useBatchForm.ts`、`modes/batch-i2v-long/BatchI2vLongSection.tsx`・`useBatchI2vLongForm.ts`のプロップ名を`hasActiveJob`→`serverBusy`へ改名。**理由コード`jobActive`とi18nキー名`batch.jobActive`／`batchI2vLong.blockReasons.jobActive`は据え置き**——文言だけを「ジョブ実行中かモデル読み込み中」の両方を覆う1本に変えた。
- **バックエンド**: `services/pipeline_manager.py`に`reject_if_loading()`を新設（判定と断り文自体は既存の`_reject_while_loading()`のまま、施錠だけを担う）。`api/generate.py`・`api/generate_chain.py`の単一ジョブガード直前でこれを呼ぶ。テストは`tests/test_base_model_axis.py`に新規2本を追加し、最終防衛線のテスト`test_auto_load_during_a_load_fails_the_job_readably`はHTTPを経由せず`job_store.create`+`pipeline_manager.run_job`を直接呼ぶ形へ書き換えた（`POST /generate`が新しい同期409で先に断るようになったため）。
- **実装済み・意図した1点の乖離**: `shell/ModelsPanel.tsx`の「読み込みには数分かかることがあります」通知（`models.loadingNotice`）は`serverBusy`ではなく`pipelineLoading`のときにだけ出す。通常の生成ジョブ中にこの通知を出すと「モデルを読み込んでいる」という偽の説明になるためである。**副作用として、サーバーが仕事中（ジョブ実行中またはモデル読み込み中）の間はSettingsのモデルパネル（読み込みボタン・更新ボタン・カテゴリ選択）が1本のルールで凍結されるようになった**（従来はローカルな読み込み中だけが対象だった）。詳細は`PENDING_TASKS.md` §1-24。

### 98.4 テスト

- **フロントエンド**: `npm run typecheck`0エラー、`npm run lint`0エラー（既存警告29件は不変）、`npm run test` → **Test Files 134 passed (134)** ／ **Tests 2578 passed | 10 skipped (2588)**（8本追加・1本置き換え）。
- **バックエンド**: `.venv\Scripts\python.exe -m pytest -q tests`全緑（`tests/test_base_model_axis.py`に2本追加、1本書き換え）。

### 98.5 状態

**実装・機械検証まで完了。** `scripts/build.ps1 -Config Release` → `scripts/deploy.ps1`（実機と配布コピーの2箇所）を実施。**オーナーの実機ゲート（`PENDING_TASKS.md` §1-24のG1〜G10）は本節の時点では未了である。**

**バックエンド側には実ダウンロードを伴うサブマシンゲートが残っている**（[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §82.4）。**そのうちG7aがこの文面を実機で見る項目**で、「トーストに`install-LTX25.bat`が出る／選択が2.3へ戻る／`POST /pipeline/load`が送られない」の3点を確認する。**フロントエンドの作業は本段で完了している。**

> **【2026-08-31 追記】実機ゲート合格。** 記録はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) **§83**が正本（G7・G8・G9の裁定を含む）。台帳は[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) **§3-129**へ移した。**フロントエンドの製品コードは、この合格でも1行も変わっていない。**

## 99. 快適上限マーカーをサーバー配信テーブルへ作り替えた — エンジン系統の軸を入れ、右クリック尺とa2vの自動尺をマーカーへ揃えた（バックエンド§3-95・§3-12の後続）（2026-08-31）

### 99.1 結論

**賢い快適上限マーカーの「線をどこに引くか」という知識を、フロントエンドのハードコードした条件式から、サーバーが配信する表（`GET /config`の`limits.comfort_budgets`）へ移した。** 従来この線は「LTX 2.3 かつ5つの高速化トグルが全on」という5条件をフロントエンドが数え上げて判定しており、**エンジン系統〔ベースモデルの世代〕の軸をそもそも持っていなかった**。表にしたことで、LTX 2.5 は高速化の設定によらず常に賢い線が引かれるようになり、**将来のモデルや高速化トグルが増えても、フロントエンドの条件式ではなく表の行を1本足すだけで対応できる**。

あわせて、これまで別々の値を使っていた**Single系の右クリック直後の尺**と**a2v（音声から動画を生成する機能）にwavを付けたときの自動尺調整**を、マーカーと同じ値へ揃えた。

**数値（線の値・レガシー表の値・境界）はこの文書には書かない。** 線の正本はバックエンド[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md) §1.1、較正と判定規則v3の記録は同[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §84、契約は[`API_REFERENCE.md`](API_REFERENCE.md) §3.2である。

### 99.2 設計

- **横断モジュール`shell/comfortTable.ts`を新設した。** 線の解決はSingle専用の`modes/single/spillUtils.ts`にもChained専用の`shell/tokenBudget.ts`にも置けない——この2つは互いにimportしない規律があり、線はその両方から使うからである。**新モジュールはどちらにも属さない場所（`shell/`）へ置き、Single側からはimportしない**（`tokenBudget.ts`からのimportだけを、Chainedのマーカーへ予算を渡すための例外として認めた）。
- **`resolveComfortRow()`が1つの入口である。** 実効の高速化設定・エンジン系統・sageの利用可否を受け取り、表を上から照合して「一致した行」か`null`（＝一致行なし）を返す。**呼び手はこの関数だけを通す**——5条件を数え上げる判定関数（`isFullAcceleration`）は削除した。**条件はもうコードではなく表の側にある**、という一点が今回の設計の中心である。
- **`requires`の照合はサーバー語彙で行う。** `shell/accelerationSettings.ts`に`effectiveAccelerationFields()`を新設し、ブラウザ側の設定（keep_residentをblock_swap_prefetchの実効状態で畳み込んだ後の値、sageは利用可否を織り込んだ後の値）を、リクエストのフィールド名と同じ語彙へ写してから照合する。**生の設定値で照合すると、自動offされたkeep_residentを見誤る。**
- **「一致行なし」は正常系である。** そのときは従来どおりレガシー表`spill_free_frames`へ落ちる。LTX 2.3 の既定構成は**意図的に行を持たない**ので、この経路を通るのが正しい（理由は[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md) §1.1）。**「行が無いのは設定漏れだ」と誤解して`requires`が空の行を足すと、退避が起きる領域まで快適と表示してしまう**ので、コード・型定義・バックエンドのテストの3か所に防波堤を置いた。
- **エンジン系統の配線を通した。** `shell/useBaseModels.ts`が選択中のベースモデルの系統を公開し、`AppShell`経由でSingle・Chainedの各画面へ渡す。**起動直後やオフラインで系統が分からないときは互換シムへ倒す**——ここで`null`（線なし）にすると、全on構成の利用者のマーカーが一瞬レガシー表の値へ落ちてちらつくためである。
- **右クリック尺は「天井を外から渡す」形にした。** `timeline/deriveDuration.ts`に加速や系統の知識を持ち込まず、呼び出し側が計算した天井を引数で受け取る。どの右クリック項目が単発生成へ向かうのかは`timeline/menuRouting.ts`の既存の経路表から逆引きする（**新しい対応表を二重に持たない**）。**Chain系の右クリック尺は今回そのままである**（判断は台帳[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §3-132）。
- **a2vのwavクランプもマーカーと同じ値を見る。** 従来はレガシー表を直接読んでいたため、全on構成ではマーカーと食い違っていた。
- **旧経路は縮小して残した。** `spillUtils.ts`はレガシー表の解決（面積が最も近い鍵への丸め込み）だけを担う「フォールバックの正」に絞り、賢い線の計算はすべて新モジュールへ移した。
- **レガシー表の値そのものも更新した**（バックエンド側の変更。フロントエンドのミラーとモックも追随させた）。この結果、**Chained側の推奨クリップ長とChain系の右クリック尺は全エンジンで動く**——賢い線の導入ではなくレガシー表の再測定による、意図した変化である。

### 99.3 後方互換

- **古いサーバー（表を配信しない）＋新しいフロントエンド**: 互換シムが働き、今日と同じ挙動（5トグル全onのときだけ賢い線）になる。
- **新しいサーバー＋古いフロントエンド**: 未知の鍵は無視されるので不変。
- **`config.yaml`に表を書かない場合**: バックエンドのコード既定の表が配信される（書けば上書きできる）。
- **系統が未知（起動直後・オフライン）**: 互換シムへ倒すのでちらつかない。
- **表の値が壊れている（0・負・NaN・係数が1未満）**: その値だけを定数へ正規化する。
- **知らない`requires`の鍵**: 不一致として扱い、安全側（レガシー表）へ落ちる。

### 99.4 テストと状態

ゲート結果（2026-08-31）: `npm run typecheck` エラー0／`npm test` 135ファイル 2619 passed・10 skipped・0 failed／`npm run lint` 警告31件（基準値と同数）。配置は `build.ps1`→`deploy.ps1`、SHA-256 `C89003BF…7615A` が3箇所一致。詳細と一次記録はバックエンド `Docs/VERIFICATION_LOG.md` §84。

## 100. Settingsの並び順の入れ替えと、非対応のベースモデルでPrunaVAEDの行を画面から消した件（バックエンド台帳§2-1・§2-4）（2026-09-01）

### 100.1 結論

**Settingsの画面でModelsのブロックをAccelerationの前へ動かし、あわせて「いま選んでいるベースモデルが受け付けない高速化の項目は、画面から消す」という扱いを1件目として入れた。** 消す対象はPrunaVAED（枝刈り版の映像VAEデコーダ）の行で、判定の材料はサーバーが配信する`unsupported_features`である。**ベースモデルの名前を画面のコードへ直接書かない**——ここが今回いちばん守った線で、将来モデルが増えても画面側は触らずに済む。

台帳の項目番号は、実装前は§1-25・§1-26・§4-29だったが、実装済み・テスト待ちとして§2へ移った（バックエンド[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md)）。

### 100.2 Settingsの並び順 — 過去の配置意図を書き換えた

**`ModelsPanel`がSave/Closeの下に置いてあったのは、過去のタスク指示による意図的な配置だった。** 「モデルの切り替えは重い操作なので、接続設定の主ボタンへ向かう手の動線から遠ざける」という理由で、`DangerZonePanel`（パイプラインの解放と終了ジョブの一括削除）と同じ考え方の隣に並べていた。

**2026-09-01のオーナー指示でこれを覆し、Right-click menu（Fps policy）の直後・Accelerationの見出しの前へ移した。** 新しい理由は「ベースモデルの操作を、高速化の行を全部スクロールし切らずに触れる場所に置きたい」である。**移動そのものはJSXのブロックを1つ動かすだけで足りた**（state・ref・CSSの隣接依存は無い）。ただし**旧理由がコメントとして現地に残っていた**ので、そこを新しい配置理由へ書き換え、同じ理由づけを共有していた`DangerZonePanel`側のコメントにも「Modelsは2026-09-01に上へ移った」と書き足した。`SettingsPanel.css`の「上の接続設定と区切る」旨のコメントも新しい隣接関係へ直している。**書き換えなかった場合に残るのは、いま存在しない配置を説明する注釈だけである。**

### 100.3 PrunaVAEDの非表示と、保存値の正規化

**症状は「LTX 2.3で選んだPrunaVAEDが、LTX 2.5へ切り替えたあとも送られ続けて、以後の生成が全部422になる」だった。** 開いている口は2つ（画面が項目を落とさない／送信の判定にエンジンの軸が無い）あったが、**塞いだのは画面側だけである。**

- **表示**: `unsupported_features`に`prune_vaed`が入っている間は、PrunaVAEDの行と、その下の注意文を**1つの条件でまとめて囲って消す**。この2つは兄弟要素なので、行だけを条件にすると注意文が取り残される。**グレーアウトではなく非表示**にしたのは、この項目が「受理はされるが効かない」たぐいではなく、422で断られるからである。
- **保存値の正規化**: 「非対応のベースモデルがアクティブだ」と観測した時点で`setVaeMode("default")`を呼ぶ`useEffect`を**1本だけ**足した。起動時にブラウザへ残っていた場合と、操作で切り替えた場合の両方が、この1本で塞がる。永続化は既存の書き戻しが追随するので、**新しい通信もポーリングも増えていない。** 置き場は`AppShell`——`useBaseModels`と`useAccelerationSettings`は互いを知らない設計で、両方を持つのはここだけだからである。形は既存の同型の前例（無効になったモードから`single`へ跳ね返すeffect）に揃えた。
- **送信ロジック（`accelerationRequestFields()`）は1行も変えていない。** 表示を消して保存値を正規化すれば送られる値は`default`になる——**同じ結果を2か所で担保しない**、という判断である。

**受け入れた帰結**: LTX 2.3→2.5→2.3と往復すると、PrunaVAEDの選択は消えている（手で入れ直す）。入れ直すまで快適上限マーカーはレガシー表の低い線になるが、**これは実際の設定を正しく映した結果なので直さない**（マーカーの設計は§99）。

**表駆動化はやらなかった。** 「モデルごとにどのUIを見せるかの表を作り、機械的に表示と非表示を決める」という案は出たが、いまは`unsupported_features`を要素ごとに読む軽い方式で足りる。**個別の判定が5〜10個に増えて見通しが悪くなった時点で消費側を束ね直す**という判断で、研究課題としてバックエンド台帳§3-135へ起票してある。

### 100.4 バックエンド側で同時に入った1件

**Gradio同梱WebUIのバッチa2v（音声から動画を生成する機能をまとめて流す機能）のスキップ判定が、フロントエンドの規約へ追随した。** これで実効上限も表（CSV）へ書くスキップ理由コードも両GUI共通になり、§9.1以来の逆方向の差分が1つ解消している。**フロントエンドの製品コードは1行も変わっていない。** 契約の正本はバックエンド[`BATCH_A2V_CSV_SPEC.md`](../../../Docs/BATCH_A2V_CSV_SPEC.md)、追随の記録と確認項目はバックエンド台帳§2-3である。

### 100.5 テストと状態

ゲート結果（2026-09-01）: `npm run typecheck` エラー0／`npm test` 135ファイル **2621 passed・10 skipped・0 failed**（`AppShell.featureScope.test.tsx`へ2本追加——LTX 2.5へ切り替えるとPrunaVAEDの行と注意文が両方消えること、ブラウザに`prune_vaed`が残っていてもLTX 2.5がアクティブになれば保存値が`default`へ戻ること）。

**オーナーの実機目視ゲートは未了である**（項目はバックエンド台帳§2-1・§2-4のチェックリスト）。実機で見るには`build.ps1`→`deploy.ps1`での配置が要る。

> **【2026-09-01 追記】実機目視ゲート合格・配置済み。** オーナーの実機目視ゲートG-O1〜G-O4（本節の2件に加え、バックエンド側で同時に入った§100.4の1件とGradioのベースモデル露出）は**4本とも全項目合格**し、`build.ps1`→`deploy.ps1`での配置（実機＋リポジトリ内の配布コピーの2箇所）も同日に完了した。ゲート結果の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) **§85**で、台帳の項目は[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) **§3-136〜§3-139**へ移した（§100.1が挙げる§1-25・§1-26・§4-29と、その後の§2-1〜§2-4はいずれも欠番である）。

## 101. 素材のフレームレートを右クリックプリフィルへ流し込めるようにした（契約v11）— あわせてMCPとモックに閉じていた3件を片付けた（バックエンド台帳§3-13・§3-115・§3-116・§3-122）（2026-09-01）

### 101.1 結論

**タイムラインで選んだ素材そのもののフレームレートを読めるようにし、Settingsのfps軸「素材に合わせる」を実際に機能させた。** これまでこの選択肢は名前だけで、中身はプロジェクトのフレームレートの代用だった（2026-07-22のX1でグレーアウトして残してあった）。**塞いでいたのはAviUtl2 SDKの`get_media_info`にフレームレートの欄が無いことだったが、Windowsが元から持っているMedia Foundationで素材ファイルの宣言だけを読めば取れる**——これが今回いちばん大きな発見で、当初「入力プラグイン経由の中規模なネイティブ実装が要る」と見積もっていた作業が、新規モジュール1本で片づいた。

同じ日に、**作業領域が重ならない軽い3件**（AIエージェント向けMCPサーバーの撮り直し・画角拡張の引数、フロントエンドのモックのクリップ数の下限）も同時に片付けている。こちらは**実機で確認できる要素が原理的に無い**ので、自動ゲートが全部緑になった時点で決着とした。

**本節が引く台帳の番号は、いずれも起票時のものである**（[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md)側ではすべて欠番になった）。§3-13は同書の「2. 実装済み・ユーザーのテスト待ち」へ**§2-5**として移り、§3-115・§3-116・§3-122は[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md)の同番号へ移っている。

### 101.2 §3-13 素材fpsの取得（契約v11）

- **読み方**: 新規モジュール`native/src/media_fps_probe.{h,cpp}`が、素材ファイルに`IMFSourceReader`を開き、**最初の映像ストリームの「ネイティブ型」**（コンテナ自身の宣言）から`MF_MT_FRAME_RATE`（分子と分母の対）を読む。**デコーダを作らず1フレームも復号しない**ので、コストは動画の長さに比例しない。
- **契約は`mediaFps`の1フィールドだけ**（`timeline.getSelection`の`selected[]`へ追加、contract v11）。**「不明」は`0`**で、`mediaWidth`／`mediaHeight`とまったく同じ非nullableの規約に揃えた。契約の正本は[`BRIDGE_CONTRACT.md`](BRIDGE_CONTRACT.md) §4.13・§8である。
- **nativeは生値を返し、整数へのスナップはwebui側で行う。** 29.97は`29.97…`のまま渡り、`29.97→30`・`23.976→24`という丸めは`timeline/prefillSeed.ts`の`snapMaterialFps`が担う。**`media*`は「素材由来の事実」を運ぶ枠で、方針を混ぜないという既存の規約に従った**——この分担のおかげで、丸め方を変えたくなってもネイティブを触らずに済む。スナップは`Math.round`一本のあと`[1, 60]`へクランプする（120fpsや240fpsの素材が、そのまま送られてサーバーに422で断られる穴を塞ぐため。クランプの形は既存のfps入力欄のsetterと同じである）。

  > **追記（2026-09-02）**: この`snapMaterialFps`は、素材段だけでなくfpsが決まりうる全入口を丸める`modes/single/paramUtils.ts`の`snapFrameRate`へ統合された（台帳§3-71/§3-72対策）。後日談は§105参照。
- **プリフィルのfpsは3段**になった——①fps軸が「素材に合わせる」なら素材のfps、②読めなければプロジェクトのfps、③それも無ければバックエンドの既定値。**「読めない」は異常ではない**（古いプラグイン・動画以外のオブジェクト・Media Foundationが対応しないコンテナ）。
- **mkv／webmで読めないのは正常系である。** AviUtl2はbeta10でMedia Foundationのファイルリーダーを外してL-SMASH Worksが標準構成になったので、**「AviUtl2が読めるファイル」と「Media Foundationがfpsを答えられるファイル」は同じ集合ではない。** 読めなければプロジェクトのfpsへ落ちるだけなので、ログにも出さない。
- **X1の巻き戻し**: fps軸「素材に合わせる」を強制的に「プロジェクトに合わせる」へ倒していた二重の防御（保存値の読み込み時と、設定を変えるときのsetter）と、ボタンのグレーアウト・「将来対応予定」のツールチップ（と、その文言の英日2キー）をすべて撤去した。**X1の挙動を固定していたテストは削除ではなく反転させてある**——同じ場所で反対の期待値を見張るほうが、次に誰かが同じ防御を足したときに気づける。
- **STAスレッド上でMedia Foundationを同期に読むのは、本リポジトリでは初めての経路である。** 既存の実績（`Mp4Writer`の書き込み）はすべてHTTPワーカー＝MTA側で、UIスレッドから開いた前例が無い。**自動テストは全部通っているが、これは「テストが通ったから安全」とは言えない類の変更**なので、実機ゲートのG3（mkv・破損ファイル・巨大ファイルで固まらないこと）・G5（大きい動画で右クリックの体感が変わらないこと）・G8（🎞挿入の書き出し中に右クリックしても両者が喧嘩しないこと）で見てもらう。**もしハングが観測されたら、非同期化を別途起票してフォールバック挙動のまま出す**という進め方を先に決めてある。
- **`MFStartup`はプロセスで1回だけ呼び、対応する`MFShutdown`は意図的に呼ばない。** プラグインの寿命はプロセスの寿命であり、毎クリックの起動・終了の往復は無駄なうえ、`Mp4Writer`が動いている最中に参照カウントが0へ落ちる懸念も同時に消える。**理由はヘッダのコメントが正本**である。

### 101.3 §3-116 モックのクリップ数の下限 — テストどうしの汚染が表に出た件

**フロントエンドのモックバックエンドは、クリップ1本の連結生成を許す例外として`source_video`と`source_audio`の2つしか持っていなかった。** 実APIは`retake`・`reference_video_id`・`end_source`も例外にしているので、**実機では通る構成がモックでだけ422になる**——モックで確かめた手順が実機で使える保証にならない、という形の穴である。判定1行に3つ足して**5例外**へ揃え、モック側とバックエンドのテスト側の両方に相互参照のコメントを置いた（一覧を2つの言語で持つ以上、足し忘れは必ずまた起きるため）。

**ここで、直したこと自体が正しい帰結として既存テスト2本を落とした。** `ChainedScreen.endSource.test.tsx`の2本で、Generateボタンが「Busy…」表示のまま見つからなくなったのである。**原因はテストどうしの状態の汚染だった**——アプリ全体が使うモックはモジュールのシングルトンなので、前のテストが投げたジョブが次のテストの描画時点でも「サーバー上に」残っており、`serverBusy`が全部のGenerateボタンを改名してしまう。**このファイルは今まで偶然その汚染を免れていた**: モックが1クリップの`end_source`を422で断っていたおかげで、ジョブがそもそも作られていなかったからである。修正でモックが202を返すようになり、隠れていた前提が表に出た。テスト側に**ジョブ台帳を空にするドレイン処理**を足して解決している。

**教訓は「テストが通っていることの理由を、当のテストは知らない」である。** 後片付けを書かずに済んでいたのは設計ではなく、別の不具合の副作用だった。**モックの振る舞いを実物へ寄せると、こういう『不具合に支えられていた合格』が落ちる**——落ちたときは、まずテスト側の前提を疑うより先に「何が変わってこれが表に出たのか」をたどるほうが早い。

### 101.4 バックエンド側で同時に入った3件（詳細はあちらが正本）

いずれもAIエージェント向けMCPサーバーの改修で、**フロントエンドの製品コードは1行も変わっていない。** 設計判断の正本はバックエンド[`MCP_SERVER_DESIGN.md`](../../../Docs/MCP_SERVER_DESIGN.md)のD19・D20・D21で、ここには書き写さない。

- **§3-115**: `submit_chain`に撮り直し（Retake）の引数を公開した（D19）。
- **§3-122**: `submit_generate`に画角拡張（Outpainting）の引数を公開した（D20）。
- **`upload_video`の拡張**: 尺とフレームレートを実測して返せるようにし、撮り直しの窓開始秒の下調べがMCPの中だけで完結するようにした（D21）。

### 101.5 テストと状態

ゲート結果（2026-09-01）: `npm run typecheck` エラー0／`npm test` **2,637 passed・10 skipped・0 failed**。AviUtl2プラグイン（C++）側は**doctest 282ケース・アサーション1452件**で0失敗（改修前は278ケース・1416アサーション）。バックエンドの全体スイートは**junit集計で2,146件・failures 0・errors 0・skipped 23**である。**数値の物差しと一次記録の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §86**で、ここには要点だけを引く。

**自動ゲートは全部緑だが、§3-13のオーナー実機ゲートG1〜G8は未了である**（項目はバックエンド台帳[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md)「2. 実装済み・ユーザーのテスト待ち」のチェックリスト）。実機で見るには`build.ps1`→`deploy.ps1`での配置が要る。**MCPとモックに閉じた3件（§3-115・§3-116・§3-122）は実機で確認できる要素が原理的に無いため、オーナー裁定により§2を経ずに直接クローズした**（[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-115・§3-116・§3-122）。

## 102. Editタブの右クリック自動読み込みが二重にアップロードしていた不具合を修正した（バックエンド台帳§3-63）（2026-09-02）

### 102.1 結論

**Editタブで「選択範囲を撮り直す」（Retake）へ右クリックで入ると、見えていないOutpaintingパネルの自動読み込みも一緒に走り、同じ素材ファイルが`uploads/`へもう一度アップロードされていた。** `EditScreen`はOutpaintingとRetakeの両フォームを常時マウントする設計（サブタブの切り替えは表示だけで、状態は両方生きている）だが、素材を読み込む一発処理のうちOutpainting側だけが、自分宛ての右クリックかどうかを見ずに`initialIntent`を素通しにしていた。**台帳の起票時の記述「逆向きも同様」は事実誤りだった**——調査の結果、Retake側（`useRetakeForm.ts`の`snapshotSelection`）は当時から`intent === "retake"`を見る自衛が既にあり、不具合は**片方向（Outpainting側）のみ**だった。

### 102.2 直したこと

`webui/src/modes/edit/useOutpaintForm.ts:213`の一発右クリック自動読み込みを、次の1行へ変更した。

```ts
const selection = initialIntent?.intent === "outpaint" ? initialIntent.selection : undefined;
```

判定基準をサブタブ選択（`EditScreen.tsx`の`initialIntent?.intent === "outpaint"`）に揃え、Retake側`useRetakeForm.ts`の`snapshotSelection`と双子の形にした。**`initialIntent`の同ファイル内の読み出しはこの1箇所のみ**なのでヘルパー関数化はしていない。

### 102.3 テスト

`EditScreen.retake.test.tsx`の既存の通しテストを強化した（新規フィクスチャは起こしていない）。

- 「件数は1と決め打たない」というNOTEを削除し、`backend.uploadFile`が**ちょうど1回**であることを固定するアサートへ置き換えた。
- 同じテストに、Outpaintingパネルの中に素材名`take1.mp4`が**無い**ことのアサートを1行追加した——`document.querySelector(".outpaint-panel")`をDOMから直接取って`within`で絞る形にしている。**両パネルとも`hidden`属性つきで常時レンダリングされる**ため、`getByRole`系のクエリでは`hidden`配下が見えず偽合格する点に注意。
- 「Outpaintingパネルにも同名が出る」という、事実でなくなった旧NOTEの記述は書き換えた。
- 敵対的レビューの採用: 当初案にあった`OutpaintingPanel.test.tsx`への新規否定テストは、上記1行の追加で同じ主張が満たせるため見送った。

### 102.4 受容した挙動変化と休眠経路

**オーナー決定**: 修正後は、撮り直し（Retake）の右クリックで入ってからOutpaintingサブタブへ切り替えると、Outpainting側の素材欄は空になる。これは**正しい挙動として受容済み**である（そもそも入ってきた素材はRetake用で、Outpaintingが無条件で拾うことのほうが誤りだった）。

**休眠経路（敵対的レビューで指摘・記録）**: Retakeサブタブがベースモデルによって灰色化されている構成では、`intent: "retake"`が表示中のOutpaintingパネルへフォールバックする（`EditScreen.tsx:130`）。本修正後はそこにも素材が入らなくなる。**現行はLTX 2.3・LTX 2.5どちらの系統でもRetakeが有効なので、この経路は休眠状態である。**

### 102.5 テストと状態

`npm run typecheck` エラー0。`EditScreen.retake.test.tsx` **17/17 合格**。`build.ps1` → `deploy.ps1`で、実機のPluginフォルダとバックエンドリポジトリの配布コピー`AviUtl2-Plugin/NzVideomni.aux2`の2箇所へ配置済み（2026-09-02 00:00）。**台帳はクローズした**（[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-63-02。本書には旧§3-63〔IC-LoRA Depth/Deblur〕が既にあるため`-02`）。

**教訓（`npm test`のスイート全体実行が抱える罠）**: 本テーマの並行作業で、フロントエンド側を担当したエージェントが、稼働中の実バックエンドに気づかないまま`npm test`をスイート全体で実行し、`src/api/backend.integration.test.ts`が18620番ポートの実バックエンドを検出して実際の生成ジョブを送信してしまう事故があった（LTX 2.3がGPUへロードされ、ジョブが1本`outputs/`に残っている）。**今後の規律**: `GET /status`が到達不能であると確認できた場合を除き、エージェントが`webui`のテストを回すときは`backend.integration.test.ts`を除外すること。本節のテスト（`EditScreen.retake.test.tsx`単体実行）はこの罠を踏まない。詳細はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §87.4。

## 103. 撮り直し（Retake）でスタイルLoRAを使えるようにした — 塞いでいたのは`buildRequest`の1関数だけだった（バックエンド台帳§3-62）（2026-09-02）

### 103.1 結論

**撮り直しのプロンプト欄に`<lora:名:強度>`と書くと、そのLoRAが効くようになった。** §68のv1で「LoRAは対応しない」と決めて以来、`useRetakeForm.ts`はタグを**文字列として剥がすだけ**で`loras`をリクエストへ載せていなかった。ところが**バックエンドは、LTX 2.3・LTX 2.5のどちらも、撮り直しのジョブで`loras`を素通しで受け付ける**——撮り直しは1クリップのチェーンジョブ（`POST /generate/chain`）であり、`loras`はそのスキーマに元から在って検証も通る。**塞いでいたのはフロントエンドの1関数だけで、バックエンドは1行も変えていない。**

### 103.2 直したこと

`webui/src/modes/edit/useRetakeForm.ts`の`buildRequest`で、`parseLoraPrompt`の戻り値を**両方**使うようにした（従来は`strippedPrompt`だけを取り出して`loras`を捨てていた）。

```ts
const { strippedPrompt, loras } = parseLoraPrompt(prompt);
return {
  prompt: strippedPrompt.trim(),
  // タグが 1 つも無ければキーごと省く（Chain の `buildRequest` と同じ作法）。
  ...(loras.length > 0 ? { loras } : {}),
```

**キーを省く作法は意味を持っている**——LoRAを使わない撮り直しのリクエストは、この改修の前後でバイト単位で同じである。

**`combineLoras`は通していない。** あれはCreate／Chainedが持つ制御系（IC-LoRA）の選択パネルとプロンプト内のタグを混ぜるための関数で、**Editタブの撮り直しにそのパネルは無い**。無いものを混ぜる関数を通すと、読む人に「どこかに制御系の選択UIがあるはずだ」と思わせてしまう。

**制御系LoRAの名前を手打ちされても、クライアント側では止めない。** 撮り直しは`reference_video_id`を送れない（両者はAPIで排他）ので、制御系を指定したリクエストはバックエンドの既存422 `LORA_REQUIRES_REFERENCE`がそのまま断る。**ここにだけ新しいゲートを足すと「どのタブで何が弾かれるか」の規則が1つ増える**ので、既存の422に委ねた（UIのルールは例外を増やさず単純に保つ）。

### 103.3 テスト

`useRetakeForm.test.tsx`の「`loras`を送らない」という固定（pin）を**反転**させ、あわせて2件を新しく固定した。

- LoRAタグは指示文（`prompt`）から外し、`loras[]`として送る（Create／Chainedと同じ扱い）。
- タグが1つも無ければ、`loras`キー自体を載せない。
- 制御系LoRAの名前を手打ちしてもクライアント側では止めず、そのまま`loras[]`に載せる（バックエンドの422に委ねる、という上記の設計判断の固定）。

**フロントエンドのテストは2,638件すべて緑**（`npm run typecheck`＝`tsc -b`は0エラー、`npm run lint`も通過）。**実バックエンドが稼働しているあいだにスイート全体を回すときは、`src/api/backend.integration.test.ts`を必ず`--exclude`すること**——実GPUへ本物のジョブを投げてしまう（§102の教訓）。

### 103.4 ゲートと状態

**実機ゲートはバックエンド側の記録が正本である**——[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) **§88.5**のG7〜G9で全合格した（MCP経由でバックエンドの受け口を直接踏む腕。フロントエンドの差分そのものは上記のvitestが担保する）。**撮り直し＋スタイルLoRAで、`metadata.json`の`retake.freeze_proof`の8項目（映像の頭・尾×2ステージ、音声の頭・尾×2ステージ）がすべて0.0・`pass:true`のまま維持された**ことを、LTX 2.3・LTX 2.5・「音声を作り直さない×`audio_strength=0.0`」の3構成で確認している。

**制御系のIC-LoRA（参照動画つきの制御）は本テーマに含めていない。** アプリ層の422と2つのエンジンのassert、計3箇所で明示的に禁止されており、解禁は別テーマである（バックエンド[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) **§3-141**へ起票した）。

- **変更ファイル（webui）**: `modes/edit/useRetakeForm.ts`／`modes/edit/useRetakeForm.test.tsx`の2点のみ。
- **`.aux2`は再ビルドして2箇所へ配った**（`build.ps1` → `deploy.ps1`。実機のPluginフォルダと、バックエンドリポジトリの配布コピー`AviUtl2-Plugin/NzVideomni.aux2`。2026-09-02。**この配布コピーも同日のコミットに含まれる**）。
- **状態**: 実装・テスト・実機ゲートとも完了。台帳の記録はバックエンド[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-62-02。

## 104. バッチA2Vの走行を、画面の作り直しに耐えるところへ移した — ランナーのシングルトン化（バックエンド台帳§3-47）（2026-09-02）

### 104.1 結論

**バッチA2Vを走らせている最中にCreate画面が作り直されても、走行が見えなくならなくなった。** バッチi2v-longが最初から採っている置き場所（モジュールレベルのシングルトン）へ、A2Vのランナー実体を移しただけである。新しく作られた画面は、走っているそのバッチへつなぎ直す——Stopボタンも行の進捗も、走行に使っているフォルダも、そのまま戻る。

**これは「珍しい壊れ方を見つけて弾く」たぐいの改修ではない。** 共有ロック（`shell/runLock.ts`）でつながっている2つのバッチランナーが、画面の作り直しに対して**逆の寿命規則**を持っていた——その非対称のほうを消した引き算である。非対称を「既知の限界」としてピン留めしていた例外テストは、期待値ごと反転した。

### 104.2 起票の半分は、着手時点で既に失効していた

台帳§3-47（2026-07-30起票・§54.6）が主症状として書いていたのは「**共有ロックがアプリ再読み込みまで解放されない**」という滞留だった。しかしこれは**2026-07-31に解消済み**である（§54.9）——ロックの返却を実行Promiseの`onSettled`へ移し、フックがアンマウントされていても返るようにした改修が入っており、テストでも固定されていた。台帳の本文だけが、その後の改修に追随しないまま残っていた。

**着手時点で本当に残っていたのは、走行そのものの孤児化だけだった。** ランナー実体だけがフック内の`useRef`——マウントの寿命に縛られる置き場——に残っており、走行中に右クリックのintentルーティングで`remountTokens`によるリマウントが起きると、**走行は裏で続くのに画面からは消える**。無言の故障にはならない（`serverBusy`ゲートが新しい開始を塞ぐ）が、走っているものを止める手段が無くなるという実害が残っていた。

### 104.3 やったこと

新規`webui/src/modes/batch/runtime.ts`（265行）へ、走行状態一式を移した——ランナー実体・行のスナップショット・走行入力・ロックトークン・購読者リスト。`modes/batch-i2v-long/runtime.ts`（235行）の写経である。

- `useBatchRunner.ts`は`useRef`保持をやめ、`useSyncExternalStore`でそのスナップショットを購読するだけの薄い層になった。
- `useBatchForm.ts`からはロックの取得・解放ブロックが丸ごと消え、代わりに**遅延初期化子**でスナップショットから走行状態を復元するようになった。
- **`batchRunner.ts`（481行のフレームワーク非依存クラス）と`BatchSection.tsx`は1行も変えていない。** ランナーの中身ではなく置き場所だけの問題だったことの裏返しである。

**順序の規律も写した**: 開始時の行スナップショットの公開は`start()`の**前**に行う。`BatchRunner.start()`は最初のawaitまで同期実行され、その中で1行目の`Generating`が既に流れているため、後から開始時の行を書くとその更新を上書きしてしまう（1行目だけWaitingのまま止まって見え、完了した瞬間にDoneへ飛ぶ）。i2v-long側で2026-07-30に実害として報告された症状で、警告コメントごと持ってきている。

### 104.4 i2v-longからの意図的な逸脱（3件）と、実装時の一貫化（1件）

1. **復元は「走行中のときだけ」行う。** スナップショットには走行入力（行・3つのフォルダ）に加えて**スキャン由来の3項目**——行ごとの画像`<select>`の選択肢・スキャン時のfps・スキャン時のDURATION上限——も凍結記録しているが、リマウント後にそれを読むのは走行中のときだけである。走っていないときのリマウントは従来どおり白紙で、A2Vのステートレスバッチ設計（行はメモリのみ・CSVを持たない。オーナー裁定2026-07-18）と整合する。走行中は行編集がUI無効なので、凍結した選択肢と実際の走行が食い違うことも起きない。
2. **「実行対象ゼロのときの同期即時解放」は捨てた。** 1マイクロタスクぶんのロック点滅を防ぐためだけのA2V固有の作り込みで、離散イベントのReact同期フラッシュ内で解放が完了するため描画上は見えない。返却経路が`.then`の1本になった。
3. **`outDirIsAuto`（出力フォルダを自動導出しているかの旗）は据え置いた。** ここはi2v-longの初期化子を写してはいけない箇所である——`runner.outDir === null`をそのまま持ってくると、**一度走行した後は自動導出が恒久的に死に、音声フォルダを変えても出力先が前回のフォルダのままになる**（別バッチの成果物が同じフォルダへ混ざる）。この旗は利用者の意思であって、走行入力から導出できるものではない。**写経の罠として記録に残す。**
4. **（実装時に判明した一貫化）行の更新を受け取る`useEffect`にも、同じ「走行中だけ」の門を付けた。** i2v-longのまま無条件に写すと、走行が**終わった後**のリマウントで前回の行だけが表に復活する——フォルダ欄は空なのに行だけ残る、という上の1に反する半端な状態になる。判定はマウント時に凍結している（走行終了で状態がidleへ落ちた瞬間に門を閉じると、最後の行更新を取りこぼしうるため）。**機構を足したのではなく、決めてあった条件を効果側にも一貫させただけである**（オーナー承認済み）。

### 104.5 テスト

**フロントエンド全体で135ファイル・2,653件すべて緑**（`npm run typecheck`＝`tsc -b`は0エラー、`npm run lint`も通過）。基準2,638件に対し+15件で、内訳は次のとおり。

- **新規`modes/batch/runtime.test.ts` 12件** — i2v-long版（11件）を雛形に、走行中に購読者が入れ替わっても状態・行・走行入力が生き、**あとから購読した側のStopがその走行に効く**ことを直接固定する1件を足した。
- **`useBatchRunner.test.ts`の全面書き換え**（4→5件） — アンマウント→再マウントで走行中バッチへ再接続することを追加。
- **`useBatchForm.test.ts`に2件** — 走行中のリマウントは復元する／走行が終わった後のリマウントは白紙から始まる、という上の設計判断1の両側。
- **`shell/runLock.crossPanel.test.tsx`の反転** — 「A2Vは孤児化する（既知の限界）」というピンを、「走行へ再接続してStopが効き、ロックも保たれる」へ期待値ごと反転した。

**期待値を反転させただけでは証明にならない**ので、`useBatchForm.ts`の復元条件を一時的に無効化して**反転版が実際に落ちること**（リマウント後にフォルダ欄が空になる）と、行更新の経路を潰して**リマウント後の進捗が届かなくなること**を、どちらも実測してから実装を戻している。

**`useBatchForm.test.ts`はモックを差し込まず実ランナーを走らせる作り**なので、モジュールレベル状態がテストを跨いで汚れる。`beforeEach`の`__resetBatchRuntimeForTests()`に加えて、**走行を待ち切らないテストへidle待ちを補った**（ヘルパ経由で実質9件）。一時的な検出用の`afterEach`を仕込んで走り残しゼロを確認したうえで撤去してある。**実バックエンドが稼働しているあいだにスイート全体を回すときは、`src/api/backend.integration.test.ts`を必ず`--exclude`すること**（§102の教訓）。

### 104.6 状態

- **変更ファイル（webui）**: 新規`modes/batch/runtime.ts`・`modes/batch/runtime.test.ts`、改修`modes/batch/useBatchRunner.ts`・`useBatchRunner.test.ts`・`useBatchForm.ts`・`useBatchForm.test.ts`・`shell/runLock.crossPanel.test.tsx`の7点。
- **状態**: 実装・自動ゲートとも完了。**オーナー実機目視待ち**——確認項目はバックエンド[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §2-1、実装の記録は同[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-47-02。

> **追記（2026-09-02）**: 上記のオーナー実機目視ゲートG-P1（走行中リマウントでのStop・進捗生存／Stopでの停止とロック解放）に同日中に合格し、完結した。バックエンド[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §2-1は節ごと削除済み、合格記録は同[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-47-02。**本文は追記専用の規律どおり不変**である（当時の記録として読むこと）。

## 105. 非整数fps（29.97/23.976）対策 — webui・MCP・Gradioの全入口を整数へスナップした（バックエンド台帳§3-71・§3-72）（2026-09-02）

### 105.1 結論

**Chainedの音声タイル再組立検算が丸め誤差で落ちる422（台帳§3-71）と、mp4書き出し時のfps切り捨てによる長尺音ズレ（§3-72）は、いずれも非整数fps（29.97・23.976のようなNTSC系の値）が生成リクエストに乗ることが引き金だった。** バックエンド側の恒久修正（`chain_math.py`の丸め誤差対策・`media_io`のfps切り捨て対策）は凍結領域を広く触ることになり見送り、**代わりにクライアント側でfpsが決まりうる全ての入口を整数へ丸め、両バグの再現経路そのものを塞いだ**（オーナー裁定・案A採用）。

**正本は`webui/src/modes/single/paramUtils.ts`の`snapFrameRate`一箇所である。** 素材のfpsだけを丸めていた旧`timeline/prefillSeed.ts`の`snapMaterialFps`（§3-13・契約v11）はここへ統合して削除した——素材段はこの丸め方針が最初に生まれた場所だったが、429・音ズレの引き金は素材段に限らないため、対象をプロジェクトのrate/scale・4フォームの手入力・lazy initまで広げる必要があった。

### 105.2 webui: 全入口の丸め

**丸め規則は`Math.round`のあと`[1, 60]`へクランプ**（`FRAME_RATE_MIN`=1・`FRAME_RATE_MAX`=60。フォームのmin/maxとも共通の正本）。非有限・0以下・`null`・`undefined`は`undefined`を返し、呼び出し側がフォールバック値を選ぶ。**整数fpsはAPIの§5.1凍結制約ではなくUI方針である**旨を、既存の凍結制約と混同されないようdocに明記した。

対象にした入口は次の6つである。

1. **`timeline/prefillSeed.ts`のfps3段決定**（素材→プロジェクト→config既定）——プロジェクト層の`rate`/`scale`もこの改修で新たに丸め対象へ加わった。`PrefillSeed`に`frameRateSnappedFrom: number | undefined`（スナップで値が実際に変わったときだけ生値。トースト専用）を追加した。**`projectFps`（Retakeの選択範囲実時間換算が使う生の値）はこれとは別に温存し、スナップしない**——ここへスナップを通すと`deriveDuration.ts`の実時間換算が壊れるため、禁止の1文コメントと回帰テストで固定してある。
2. **`SingleScreen.tsx`の`projectGenFps`定義**（`snapFrameRate(editInfo.rate/editInfo.scale) ?? config.generation_defaults.frame_rate`）。
3. **4フォームのsetter**（`useGenerationForm.ts`・`useChainForm.ts`・`useRetakeForm.ts`・`useOutpaintForm.ts`）——`snapFrameRate(raw) ?? FRAME_RATE_MIN`に統一。Retake/Outpaintは従来生の`setState`で0が入ると窓計算が壊れる穴も同時に塞いだ。setterの名前（`setFrameRate`）はRetakeの3箇所の呼び出し（init/clearAll/公開）を壊さないようそのまま維持した。
4. **4フォームのlazy init**（`snapFrameRate(seeded) ?? FRAME_RATE_FALLBACK`。`FRAME_RATE_FALLBACK = 24`は`defaultConfig.ts`の既定値と同値である旨を相互参照コメントで固定）。
5. **`useChainForm.ts`の`baselineCommon.frameRate`**（config既定が非整数だった場合にマウント直後からisDirtyになる保険。現行config=24.0では発火しない）。

### 105.3 webui: トースト（プリフィル時のみ・1枚）

**発火点はSingleScreenとChainedScreenの2箇所だけ**——`seed.frameRateSnappedFrom`が存在するとき、one-shot ref＋空depsの`useEffect`で警告トーストを1枚出す。**手入力とマウント上書きは無音**（前者は欄の表示がその場で丸まるため、後者はプリフィルのシードトーストと二重報告になるため）。文言は`i18n/strings.ts`の新名前空間`prefill.fpsSnappedToast`（en/ja両辞書）で、元値の整形は既存の`formatFps`を流用する。

**§3-13-02実機ゲートG2で合格済みの「素材fpsの無音スナップ」は、本改修でトースト付きに変わった**（オーナー裁定②による意図的な変更であり、退行ではない）。

### 105.4 MCP・Gradio: サーバー層のミラー実装

**MCP**（`mcp_server/tools/generate.py`）に純関数`_snap_frame_rate`を追加し、`submit_generate`・`submit_chain`のpayloadへ適用した。**丸めは`math.floor(value + 0.5)`であって`round()`ではない**（Pythonの`round()`は偶数丸めで、webuiの`Math.round`と食い違うため）。**[1, 60]内の値だけ丸め、範囲外・非有限は素通しする**——サーバー側の`Field(ge=1.0, le=60.0)`の422に判断を委ねる設計で、「黙って24に差し替える」形は取っていない（誤発火したGPUジョブを隠蔽してしまうため）。

**Gradio**（`gradio_ui/handlers.py`）にも同じ規則の`_snap_frame_rate`を追加した。**適用位置はChainだけ特別**——`handlers.py`のfps解析点（`fps = float(frame_rate)`の直後）に置いた。payload直前ではなく解析点に置いたのは、その手前で幾何プリチェック（`validation.py`の`check_chain_total`→`compute_chain_layout`。§3-71の422と同じ数式のローカル版）が生fpsのまま走ってしまうためで、解析点に置けば「[1,60]事前チェック・幾何プリチェック・payload」の3箇所が同じ丸め済みの値を見る。単発・A2Vチェーンはそれぞれの解析点1箇所ずつ。`ui.py`の`gr.Number`には`precision=0`と`value=24`も付与した——これは見た目の整形ではなく、ライブ推定経路（Chainのプリセット推定・A2Vプリチェック）が生fpsのまま残ることに対する実防御である。

**3実装の丸め規則自体は共通だが、範囲外の扱いは意図的に異なる**——webuiはクランプ（`[1,60]`へ強制）、MCP・Gradioは素通し（サーバー422へ委譲）。この非対称は「3面ミラー」と呼ばない設計にしてある——書き切らずに写経すると、後任が誤ったパリティテストを書く恐れがあるため。

### 105.5 テスト・検証結果

**webui**: `npm run typecheck`エラー0。`npm test`は**2,688 passed・10 skipped**（基準2,653件から+35件、内訳は`paramUtils.test.ts`の新規スナップ単体・`prefillSeed.test.ts`の移設・4フォームのsetter/lazy initケース・トースト統合テスト2本・Create/ChainのUI経由テストなど）。失敗ゼロ。

**MCP**: `tests/test_mcp_*.py`が**141 passed → 164 passed**（新規は29.97/23.976/120/0の各ケース×2ツール）。

**Gradio**: `tests/test_gradio_handlers.py`・`tests/test_gradio_ui.py`が**261 passed → 281 passed**（単発・A2V・Chainのpayload丸めケースと、`gr.Number`の`precision == 0`アサート追加）。

**いずれも失敗ゼロ。** 詳しい検証記録・設計要点はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §91が正本。

### 105.6 既知の副作用（29.97プロジェクトに限る・許容済み）

1. **バックエンドのリサンプル経路が常時発動する**——29.97プロジェクトでは素材が29.97のままリクエストは30になるため、`pipeline_manager`のフレームレート変換が毎回走る。素材fpsスナップ（§3-13-02）で既に許容されていたのと同じ性質の副作用であり、新しく生まれたものではない。
2. **Joinの「プロジェクトfpsと動画fpsの不一致」非ブロッキング案内が常時表示される**——29.97プロジェクトでは事実として値が食い違っているので、この表示自体は正しい。

### 105.7 状態

- **変更ファイル（webui）**: `modes/single/paramUtils.ts`（正本追加）・`timeline/prefillSeed.ts`・`modes/single/SingleScreen.tsx`・`modes/single/useGenerationForm.ts`・`modes/chained/useChainForm.ts`・`modes/chained/ChainedScreen.tsx`・`modes/edit/useRetakeForm.ts`・`modes/edit/useOutpaintForm.ts`・`i18n/strings.ts`、および対応する`*.test.ts`/`*.test.tsx`群。
- **変更ファイル（バックエンド）**: `mcp_server/tools/generate.py`・`gradio_ui/handlers.py`・`gradio_ui/ui.py`、および対応するテスト。
- **状態**: 実装・自動ゲートとも完了。**オーナー実機目視待ち**——確認項目はバックエンド[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §2-2（G1〜G11）、台帳本文は同書§3-71・§3-72（実機ゲート合格後に[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md)へ移送）。

> **【2026-09-03 追記】G1〜G11は同日オーナーの実機ゲートで全項目合格し、§3-71・§3-72はバックエンドの[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md)へ移送済みである。** **参照先として挙げているバックエンド台帳の§2-2は、全項目合格の運用規則どおり見出しごと削除されている**ので、**結果の正本はバックエンドの[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §91.6 である。**

## 106. 埋め込み処理器の常駐トグルを追加した — Settingsの高速化6行目・**LTX 2.5のときだけ見える**（バックエンド台帳§3-114）（2026-09-03）

### 106.1 結論

**Settingsの「生成の高速化（Acceleration）」へ6行目`keep_resident_embeddings`（埋め込み処理器〔embeddings processor＝プロンプトを読み取ったあとの内部表現を整える部品〕のジョブ間常駐）を足した。** 手本にしたのは既存のPrunaVAEDと`keep_resident`の2行で、**新しい仕組みは1つも作っていない**。

**この件のいちばん大事な性質は「向きが逆」であることである。** これまで「使えないから画面から消す」判定は必ずLTX 2.5の側の話だったが、**埋め込み処理器はLTX 2.5にしか無い部品なので、非対応を宣言するのはLTX 2.3の側**になる。したがってこの行は**LTX 2.3を選んでいるあいだ見えない**。**§100.3で入れた「非対応のベースモデルでは行ごと消す」作法の2例目であり、逆向きの1例目でもある。**

### 106.2 表示と保存値の正規化 — §100.3と同型・向きだけが逆

- **表示**（`SettingsPanel.tsx`）: `unsupported_features`に`keep_resident_embeddings`が入っているあいだは、**行と、その下の注記を1つの条件でまとめて囲って消す**。この2つは兄弟要素なので、行だけを条件にすると注記が取り残される——§100.3とまったく同じ理由である。**グレーアウトではなく非表示**にしたのも同じ理由で、422で断られる項目だからである。
- **先読みblock swapへの依存は付けなかった。** すぐ上の`keep_resident`の行は先読みblock swapがoffのときボタンが封じられるが、**これは別のスイッチで別の対象を常駐させる**ので、同じ依存を写す理由が無い。両ボタンは先読みblock swapの状態に関わらず常に押せる。
- **保存値の正規化**（`AppShell.tsx`）: 「非対応のベースモデルがアクティブだ」と観測した時点で`setKeepResidentEmbeddings(false)`を呼ぶ`useEffect`を**1本だけ**足した。保存値は残るので、LTX 2.5でonにしたままLTX 2.3へ戻ると、以後の全リクエストにキーが載って毎回422になる——それを塞ぐ1本である。
- **画面のコードにベースモデル名は書いていない。** 判定は必ず`GET /models`の`unsupported_features`から引く。**どちらのエンジンが部品を持っていないかは、サーバーが述べる事実である。**
- **`exhaustive-deps`の警告を増やさないために、effectの入力を先にローカルへ取り出した。** 上のPrunaVAED用effectは`accelerationControls.acceleration.x`の連鎖をそのまま依存配列に書いており、ルールが連鎖の先を見通せずオブジェクト全体を要求するため警告が1件立ったままになっている。**同じ書き方を写すと警告がもう1件増えるので、意味の同じことをルールが検査できる形で書いた**（lintの基準値は動かない）。

### 106.3 送信3経路 — 2つは無変更

**送出点は`accelerationSettings.ts`の`accelerationRequestFields()`ただ1つ**という既存の規約がそのまま効くので、**Create（Single）とChainedは1行も変えていない**。加えたのは`keepResidentEmbeddings`と`KEEP_RESIDENT_EMBEDDINGS_SERVER_DEFAULT`（`false`）で、規約どおり**サーバー既定と違うときだけ**鍵が載る——**この「既定なら鍵ごと出さない」性質が、LTX 2.3で422にならない理由でもある**（既定のままのジョブはキーを持たないので、サーバーの「既定値と違うか」判定に引っかからない）。

**手を入れたのはバッチA2Vだけ**で、こちらはローカルのpayload型（`buildA2vChainPayload.ts`）とOR-list（`useBatchForm.ts`）へ1行ずつ足す必要があった。**この経路が`accelerationRequestFields()`を通らないのは以前からの構造**で、本件で作った歪みではない。

### 106.4 モック — LTX 2.3の非対応リストが、初めて名前を持った

`mockBridge.ts`の`LTX23`の`unsupported_features`は、これまでずっと空配列だった。**本件でここに初めて名前が1つ入る。** あわせてchain側の拒否ループにも1行足してある（フィールド名と機能名の対応表）。**画面のグレーアウトは1つも変わらない**——`keep_resident_embeddings`は`useBaseModels.ts`の灰色化の表（`chain` / `retake` / `outpaint` / `v2v` / `a2v` / `end_source` / `reference_video`の7語）のどこにも現れない語で、Settingsの行は灰色ではなく非表示という別の作法で扱われるからである。**「配列が減っても（増えても）グレーアウトが変わらない項目がある」という区別の3例目**である。

### 106.5 触ったファイル

- **webui（製品コード）**: `api/types.ts`・`bridge/mockBridge.ts`・`i18n/strings.ts`・`shell/accelerationSettings.ts`・`shell/useAccelerationSettings.ts`・`shell/AppShell.tsx`・`shell/SettingsPanel.tsx`・`modes/batch/buildA2vChainPayload.ts`・`modes/batch/useBatchForm.ts`。
- **webui（テスト）**: 新規`shell/keepResidentEmbeddings.paths.test.ts`（送信3経路の横断——onなら3経路とも`keep_resident_embeddings: true`を運び、サーバー既定のままなら3経路とも鍵を持たないこと）、`shell/AppShell.featureScope.test.tsx`（逆向きの表示／正規化2本）、`shell/SettingsPanel.test.tsx`・`shell/accelerationSettings.test.ts`・`shell/useAccelerationSettings.test.ts`・`shell/useBaseModels.test.ts`・`shell/comfortTable.test.ts`・`bridge/mockBridge.test.ts`・`modes/batch/useBatchForm.test.ts`・`modes/single/useGenerationForm.test.ts`・`timeline/prefillSeed.test.ts`。**既存の期待値は新しい事実へ更新しただけで、条件は1つも緩めていない。**

### 106.6 テストと状態

ゲート結果（2026-09-03）: `npm run typecheck` エラー0／`npm test` 136ファイル **2,696件 全緑**／`npm run lint` エラー0・警告32（基準どおり）。

**オーナーの実機目視ゲートは未了である。** 実機で見るには`build.ps1`→`deploy.ps1`での配置が要る。契約の正本はバックエンド[`Videomni_Backend_Specification.md`](../../../Videomni_Backend_Specification.md) §6.2・§6.10、フロントエンド側の契約は[`API_REFERENCE.md`](API_REFERENCE.md) §3.5・§5.1・§5.2、**実測（短縮幅・常駐するメインメモリの増分・SHA-256の一致）の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §92**、台帳は同[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §3-114である。

## 107. 🎞ボタンで挿入したオブジェクトが青一色になっていた症状を直した — メディア挿入をエイリアス方式へ差し替え（バックエンド台帳§3-140）（2026-09-03〜04）

### 107.1 結論

**操作パネルの「🎞」ボタンでタイムラインへ送った動画が、ファイルをドラッグ＆ドロップで置いたときと同じ二色リボン（上が映像・下が音声）になるようにした。** 原因は表示だけの問題ではなく、**`create_object_from_media_file`がエイリアスの直列化キー`音声付き`を立てないこと**だった。直し方は、**このAPIを呼ぶ代わりに、ドラッグ＆ドロップ相当のエイリアス文字列を自分で組み立てて`create_object_from_alias`へ渡す**ことである。

**実害があったことも確定している。** 青一色のオブジェクトは、右クリックメニューに**「音声を分離」の項目自体が出ない**——起票時（2026-09-01）の「見た目だけの問題と見られる」という見立てのほうが誤りだった。回避経路（生成物をダウンロードしてD&Dで置き直す）は存在したが、それは修正の理由を弱めるものではない。

### 107.2 原因を突き止めるまで — E0とE1の2段

- **E0（2026-09-01・オーナー実機）**: 同じ生成mp4を手でドラッグ＆ドロップすると二色になる。したがって**ファイル自体は正常であり、API経路の問題**と確定した。この時点で候補は「SDKからは触れない本体の内部フラグ」と「エイリアスに書かれる何か」の2つに絞られた。
- **E1（2026-09-03・オーナー実機）**: ドロップで置いた物と🎞で挿入した物をそれぞれ「エイリアスをファイルに保存」し、`.object`のテキストを突き合わせた。**差分は`[Object.0]`ブロックの2キーだけ**で、**`音声付き`**（ドロップ＝`1`／🎞挿入＝`0`）と**`再生位置`の第2値**（ドロップ＝ソース尺／🎞挿入＝`0.000`）である。`[Object.1]`（映像再生ブロック）は完全一致し、ファイルパスの違いは無関係だった。
- **これで「表示種別は本体の内部フラグでSDK経由では直せない＝作者への報告事案」という仮説が外れた。** 起票時に置いた入口（「同一なら報告事案、**差分があればその行が答え**」）がそのまま効いた形である。差分の全文はバックエンド[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-140に転記してある（**生きた台帳`PENDING_TASKS.md`側の§3-140は2026-09-04のクローズに伴い欠番である**。実測エイリアスそのものはAviUtl2ポータブル配下でリポジトリ外）。SDK側の確定知見としての集約先は[`SDK_REFERENCE.md`](SDK_REFERENCE.md) §16 (g) である。

**あわせて文書の誤りが2件見つかったので訂正した。** ひとつは[`SDK_REFERENCE.md`](SDK_REFERENCE.md) §5.1と[`WEB_RESEARCH.md`](WEB_RESEARCH.md) §5の「**XML形式**エイリアス」で、実際は**INI風**（`[Object]`／`[Object.N]`のセクション見出し＋`key=value`行）である（SDK同梱サンプル`WindowClient.cpp`:69-82にそのままのリテラルがある）。もうひとつは[`TIMELINE_ALPHA_REQUIREMENTS.md`](TIMELINE_ALPHA_REQUIREMENTS.md) 第0節Bの2026-09-01の訂正注記で、**`音声付き`はUIの設定項目としては存在しないが、エイリアスの直列化キーとしては現役**という両立を書き落としていた（同節に再訂正注記を足した）。

### 107.3 方式の選定 — 「後から直す」ではなく「最初から正しく作る」

**採らなかった案**：`create_object_from_media_file`で作ったあとに`set_object_item_value`で`音声付き`を後書きする。この案を棄却した理由は3点である。

1. **二色リボンの表示種別は生成時に決まっている傍証がある。** 同じファイルでも作り方（D&D／API）で結果が違うのだから、後から項目値だけ書き換えて表示種別まで切り替わる保証が無い。**確かめてから採るなら、確かめる作業のほうがエイリアス方式の実装より大きい。**
2. **`set_object_item_value`系の書込はカーソル位置に作用する**（既存の`UpdateObjectTextEditProc`が「カーソル移動→書込→復元」でわざわざ回避している）。挿入という頻度の高い経路にこの副作用を持ち込みたくない。
3. **「作ってから直す」2段構成は失敗の入口が2つになる。** 片方だけ成功した中間状態が生まれ、同種の症状の再発地点になる。

**採った案**：最初からエイリアスで作る1段構成。参考実装（MITライセンスの`clean262/sam3_bb_gb_generator`）に`create_object_from_alias`で外部処理の結果をタイムラインへ入れる動作実績があることも後押しになった。

### 107.4 設計の要点 — 引き算・二重ピン・三重の安全網

- **エイリアスに書くのは「媒体によって変わる値」だけにした（引き算）。** 新設した純関数`BuildMediaObjectAlias`（`native/src/alias_util.cpp`）が書くのは`再生位置`（`0.000,<ソース尺・小数3桁>,再生範囲,0`）・`ファイル`・`音声付き`の3キーだけで、`[Object.1]`は`effect.name=映像再生`の1行である。E1で採取したD&D由来のエイリアスには`再生速度=100.00`・`トラック=0`・`ループ再生=0`・`YUV=`も並んでいたが、**これらは媒体に依らない静的な既定値なので書かない**——書けば「本体の既定が変わったとき、こちらだけ古い値を押し付ける」種になる。
- **小数3桁の整形を`snprintf("%.3f")`でやっていない。** `%.3f`はCロケールの小数点を尊重するため、ホストプロセスが`setlocale`を呼んでいると`10,042`と出力してカンマ区切りの`再生位置`を壊す。整数演算で組み立てることでこの依存を断ってある。
- **尺は二重にピン留めした。** エイリアス自身が持つフレーム範囲は`length`引数を上書きするため、**エイリアス内へ`frame=0,N`を書き込む（`NormalizeAliasObjectFrameHeader`）のに加えて、`length`引数にも同じNを渡す**。この形は仮オブジェクト挿入（`insertProvisional`）が実機で実績を持つ組み合わせである。
- **尺の計算式は§3-140の前後で1文字も変えていない。** `round(総時間秒 × プロジェクトfps)`のまま（`get_media_info`の`total_time`と`EDIT_INFO`の`rate`/`scale`）。**変えたのは「作り方」だけで「長さの決め方」ではない**、という切り分けを保つためである。
- **従来APIを三重の安全網として残した。**
  1. **非動画・尺0・パス不正は最初から従来API**。静止画・音声のみのファイル（リボンは元から正しい）、素材情報が読めない場合、1フレーム未満に丸まる場合、パスにCR/LFを含む場合（この形式に行のエスケープが無いため）は、**§3-140以前とバイト単位で同じ引数**で`create_object_from_media_file`を呼ぶ。
  2. **`create_object_from_alias`が`null`を返したら、同じ`layer`／`frame`／`length`で従来APIを1回だけ試す。** エイリアス固有の失敗（効果名が解決しない等）が起きても、結果は今日と同じところまでしか落ちない。
  3. **生成後に実尺を検証する。** `get_object_layer_frame`で実際の占有範囲を測り、**要求より短ければdeleteして従来APIで作り直す**。要求値と実測値はどちらの枝でも必ずログに出す。
- **未実測の端点解釈は、片側判定で運用する。** `OBJECT_LAYER_FRAME.end`が包含か排他かは実機で観測していないので、**「短いときだけ失敗とみなす」**片側判定にした。包含として読んだ値が実は排他だった場合は長さを1だけ多く見積もるだけで通過するが、等値判定にしていたら**毎回の挿入が静かに従来APIへ落ちて修正そのものが無効化される**。ここは意図的な非対称である。同様に`create_object_from_alias`へ明示尺を渡したときの衝突挙動（`null`か、黙って短縮か）も未実測で、上の二重ピンと実尺検証はその未実測を埋めるための構えである。**実機ゲートR7のログで確定させる。**

### 107.5 挙動が1つ変わる — 詰めたリボンからのV2Vがトリム送信になる

**`再生位置`を明示するようになったため、🎞で置いたオブジェクトのリボンを手で詰めてからV2Vへ送ると、これまでの全尺送信ではなく「詰めた範囲だけの送信」に変わる。** これは**ドラッグ＆ドロップで置いたオブジェクトの挙動と一致させる方向の変化**であり、直す対象ではなく揃った結果である（同じ操作をD&D由来のオブジェクトで行えば以前からトリム送信だった）。

**トリムしていない場合の挙動は変わらない。** `decideSourceTrim`の`spanCoversWholeMedia`のゲートが従来どおり全尺送信のまま通す。判定側（webui）のコードもブリッジ契約も1文字も変えていない——変わったのは「オブジェクトが正しい`再生位置`を持つようになった」という入力側の事実だけである。契約側の注記は[`BRIDGE_CONTRACT.md`](BRIDGE_CONTRACT.md) §4.5に置いた。

### 107.6 適用した場所と、あえて適用しなかった場所

- **メディア生成を1つのヘルパへ集約した。** 新設の`CreateMediaObject`（`native/src/bridge.cpp`）が、素材情報の採取・尺の算出・エイリアスの組み立て・二重ピン・三重の安全網・ログをまとめて引き受ける。**生きている挿入経路のcreate 3箇所**（`InsertMediaEditProc`の1つと`ReplaceMediaForJobEditProc`の2つ＝マーカー位置へのcreateと`layer_max+1`への退避create）が、いずれもこのヘルパを通る。
- **未使用の`ReplaceObjectEditProc`には適用していない。** 現在どこからも呼ばれていない経路なので、**動かない場所を一緒に書き換えて検証対象を増やすことはしない**——その旨のコメントだけを残した。
- **エラーメッセージを経路中立にした。** `INSERT_FAILED`の説明は`create_object_from_media_file returned null`と特定APIを名指ししていたが、いまは**両経路とも失敗した**ことを意味するので`media object creation failed …`へ改めた（[`BRIDGE_CONTRACT.md`](BRIDGE_CONTRACT.md) §4.7・§5）。

### 107.7 再ゲート条件 — 「無音生成」モードが増えたとき

**今回の実機ゲートで実際に通るのは`音声付き=1`の枝だけである。** 現在の生成物はいずれも音声トラックを持つため、`音声付き=0`の枝（＝映像のみのmp4を挿入する道）は実装としては存在するが実機で踏まれない。

**したがって、将来「音声を出さない生成モード」が加わったときは、その時点で`音声付き=0`の枝の実機確認を1回起こすこと。** 見るのは「音声トラックの無いmp4を🎞で挿入したとき、リボンが青一色（＝音声レーン無し）で正しく、`create_object_from_alias`が`null`を返さない」の2点である。**この再ゲート条件をここに書いておくのは、通っていない枝を「通った」と勘違いしないためである。**

### 107.8 触ったファイル

- **native（製品コード）**: 新規の純関数群を`src/alias_util.cpp`・`src/alias_util.h`へ（`BuildMediaObjectAlias`＝エイリアスの組み立て。`NormalizeAliasObjectFrameHeader`は既存の再利用）、集約ヘルパ`CreateMediaObject`と3箇所の差し替えを`src/bridge.cpp`へ、エラーメッセージの中立化を`src/bridge_core.cpp`・`src/bridge_core.h`へ。
- **native（テスト）**: `tests/test_alias_util.cpp`に`BuildMediaObjectAlias`のケースを追加（`音声付き`の0／1、小数3桁の整形〔ロケール非依存〕、`=`を含むパスの素通し、拒否すべき入力5種〔空パス・CR混入・LF混入・尺0・負の尺〕、`NormalizeAliasObjectFrameHeader`との往復）。
- **webui**: **1行も触っていない。** ブリッジ契約は不変で、変わったのはnative内部の生成プリミティブだけである。
- **文書**: [`SDK_REFERENCE.md`](SDK_REFERENCE.md)（§5.1の書式訂正・§16へ(g)(h)を追加）、[`WEB_RESEARCH.md`](WEB_RESEARCH.md)（§5の書式訂正）、[`BRIDGE_CONTRACT.md`](BRIDGE_CONTRACT.md)（§4.5・§4.7・§4.14.1・§5）、[`TIMELINE_ALPHA_REQUIREMENTS.md`](TIMELINE_ALPHA_REQUIREMENTS.md)（第0節Bの再訂正・第6節の非対称記述への追記）、本節。

### 107.9 テストと状態

ゲート結果（2026-09-03〜04）: **native doctest 289ケース・アサーション1,497件 で0失敗**（`build/ninja-release/NzVideomni_tests.exe --test-suite-exclude=integration`）。**webuiは無改修だが回帰確認として実行し全緑（136ファイル・2,696件）**。

**オーナーの実機ゲートR1〜R8は2026-09-04に全項目合格した。**

### 107.10 実機ゲート結果 R1〜R8（2026-09-04・オーナー実機・全合格）

- **R1（🎞置換枝）・R2（マーカー無し枝）・R3（⬇位置指定）**: いずれも二色リボン（上が映像・下が音声）で挿入され、右クリックに「音声を分離」が表示され、実行すると音が鳴ることを確認した。
- **R4（エイリアス突き合わせ）**: `data\Alias\R4_insert.object`（🎞挿入由来）と`R4_d_and_d.object`（ドラッグ＆ドロップ由来）を突き合わせた。**`音声付き=1`**（修正前の§3-140 E1採取では`0`だった）で一致し、**`再生位置=0.000,5.042,再生範囲,0`も完全一致**した。`BuildMediaObjectAlias`が書かない`再生速度`・`トラック`・`ループ再生`・`YUV`と、`[Object.1]`の映像再生ブロックは両者とも同一で、いずれもホストの既定値で埋まっている——**媒体に依らない値は書かない（引き算）という107.4の設計判断どおり、1行エイリアスからの外挿が実機で成立した。**
- **R5（見た目D&D同一）・R6（詰めたリボンのV2Vがトリム送信）・R8（Ctrl+Z 1回で復帰）**: いずれも合格。R6は107.5で述べた「詰めたリボンからのV2Vが全尺送信からトリム送信へ変わる」変化が実機でもD&D由来のオブジェクトと同じ挙動になることを確認したものである。
- **R7（占有レイヤーで尺短縮せず退避）——【2026-09-04 訂正】**: 本節は当初「退避を確認して合格」と記録したが、**`plugin.log`と照合した結果、退避が起きたことを示す記録は無かった**。ゲート区間の`CreateMediaObject`ログは**全5件とも`INFO`で`WARN`は0件**、しかも**5件すべてが要求どおりの`layer`/`frame`へ一発で成功し、実尺も要求`length`と一致している**（R7に相当する最終行は`2026-09-04 00:38:42`・layer 0・frame 96・length 120 → `start 96, end 216`。要求位置のまま、尺も不変）。つまり**衝突シナリオそのものが成立しておらず、退避（`layer_max+1`へのフォールバック）も安全網2・3も実機では発火していない**。したがって`create_object_from_alias`へ明示尺を渡したときの**衝突時挙動は依然として未実測**であり、安全網は保険として維持する（[`SDK_REFERENCE.md`](SDK_REFERENCE.md) §16 (h) の前段が正本）。**オーナーが観察した内容の詳細は確認中**である。
- **観察事項1件（不具合ではない）——【2026-09-04 撤回】この項の結論は逆だった。本書§107.11を参照のこと。** 以下は当時の記述として残す。「初期尺が🎞挿入＝121フレーム・D&D＝120フレームで1フレーム差がある。**挿入側の121は実際に生成されたフレーム数（LTXの8n+1）と一致し、D&D側の120はAviUtl2本体側の丸めによる切り捨てである**——挿入側のほうが素材の実尺に忠実であり、直す対象ではない。」 **正しくは、忠実だったのはD&D側の120で、挿入側は`frame=0,121`＝包含なので122フレームの過長だった。「本体側の丸めによる切り捨て」という機序説明も誤りである**（同じ素材をレガシー経路で挿入した`from_insert_button.object`もD&Dと同じ`frame=0,240`になる）。**不具合であり、§107.11で修正した。**
- **`OBJECT_LAYER_FRAME.end`の端点解釈がこのゲート中のログから確定した——【2026-09-04 撤回】この結論は誤りだった。正しくは「包含」である。本書§107.11を参照のこと。** 以下は当時の記述として残す。「`plugin.log`の`CreateMediaObject`ログ5件（layer 0〜3・length 120／121）はいずれも`end - start`が要求`length`と完全一致しており、**`end`は排他（次フレームの先頭を指す）と確定した**。」 **その一致は端点規約の証拠ではなく、作られたオブジェクトが要求より1フレーム長かったことの表れだった。**詳細と実測値は[`SDK_REFERENCE.md`](SDK_REFERENCE.md) §16 (h) に記録した（同項も§107.11に合わせて書き換え済み）。107.4で「片側判定」にしていた実尺検証の安全網は、このゲート中は一度も発火していない（`WARN`ログ0件）。

台帳の記録はバックエンド[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-140（クローズ：2026-09-04）。

**状態**: 実装コミット済み（S1＝`78df6e1`〔純関数＋doctest〕／S2＝`36cd55e`〔挿入経路の差し替え〕／S3＝`691d205`〔文書更新〕／S4＝`93e58a4`〔配布コピー反映〕／**S5＝`984f2c2`〔実機ゲートR1〜R8全合格・台帳クローズ〕／`37fcb21`〔完結後の失効掃除〕＋文書整合はレビュー反映コミット（第2回）**）、実機ゲートR1〜R8全合格・完結（**R7の記録は上記の訂正のとおり**）。配布aux2は**1,267,200バイト・SHA-256 `4c202b8d8a45…`** で、ビルド出力・実機配置・リポジトリ配布コピーの3値が一致している（バックエンド[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-140が正本）。（その後の§108のビルドで配布コピーは**1,267,712バイト・SHA-256 `fa179e34…`**へ更新された。本節および§107.11のゲートに合格したのは、当時のバイナリである。）

### 107.11 追補（2026-09-04）— 端点は「包含」だった。1フレーム過長の修正と、前日の判断3点の撤回

§3-140をクローズした直後に、**挿入されるオブジェクトが常に1フレーム長い**というバグが見つかった。オーナーの「仮オブジェクトを実装したとき、実機で`+1`が必要だと確かめた記憶がある」という指摘が発端である。再検証の結果、**§107.10と§16 (h)に書いた端点解釈は誤りで、正しくは「包含」**だった。本節はその訂正と修正の記録である。

**撤回する判断3点**（いずれも2026-09-04に本DEVLOGおよび[`SDK_REFERENCE.md`](SDK_REFERENCE.md) §16 (h)へ書いたもの）:

1. **「`OBJECT_LAYER_FRAME.end`は排他と確定した」——誤り。正しくは包含である。** §107.10の該当項（`plugin.log`5件から排他を導いた項）と、§107.4の「未実測の端点解釈は、片側判定で運用する」の項は、いずれも本節が上書きする。
2. **「🎞挿入側の121のほうが素材の実尺に忠実で、直す対象ではない」——結論が逆。** §107.10の「観察事項1件（不具合ではない）」がこれで、**忠実だったのはD&D側の120**であり、挿入側は**122フレーム**（`frame=0,121`は包含なので122個）だった。**不具合であり、直す対象だった。**
3. **「D&D側の120はAviUtl2本体側の丸めによる切り捨てである」——機序の説明そのものが誤り。** 切り捨ては存在しない（下記の`from_insert_button.object`が証拠）。

**一次証拠**（ファイルは複製せず、ここへ転記する）:

- **実機が書き出したエイリアス4本**（`D:\For_Videos\AviUtl2\aviutl2_v2.0.54\data\Alias\`。オーナーが「エイリアスをファイルに保存」で採取）
  - `R4_d_and_d.object`（121フレーム素材のD&D）→ **`frame=0,120`**。1始まりの画面表示で最終フレームは121（オーナー実機確認）。
  - `R4_insert.object`（同じ素材の🎞挿入・修正前）→ **`frame=0,121`** ＝ **122フレーム**。1フレーム過長。
  - `from_d_and_d.object`（241フレーム素材のD&D）→ **`frame=0,240`**。
  - **`from_insert_button.object`（同素材・§3-140以前のレガシー経路。`create_object_from_media_file`へ`length=241`を渡した）→ `frame=0,240`でD&Dと完全一致。** これが決め手である——**`length`引数は「フレーム数（個数）」で、本体がそれを包含終端`240`へ直して書き出している**。丸めも切り捨ても存在しない。
- **`plugin.log`の`CreateMediaObject` INFOログ5件**（2026-09-04 00:32:14〜00:38:42）。`end - start`が要求`length`と一致していたのは`end`が排他だからではなく、**作られたオブジェクトが1フレーム長かった**からである（包含で読めば`end − start + 1 = length + 1`）。同じログは**エイリアスのヘッダが`length`引数に勝つことの実機証明**でもある: `frame=0,121`を書いたエイリアスへ`length=121`を渡した行が`start 0, end 121`（=122フレーム）で返っており、**採用されたのはヘッダ側**だった。

**バグの実体と修正（方式A1）**: `frame=a,b`は0始まり・両端を含む範囲なのに、`NormalizeAliasObjectFrameHeader`は`frame=0,<length>`を書いていた——**「個数」と「包含終端」の混同**である。修正は**`frame=0,<length−1>`を書く**の1点で、二重ピン（ヘッダ＋`length`引数）の形はそのまま維持した。実機実績のある形を崩さず、書き込む数値だけを1減らす方式である。§107.4の「尺は二重にピン留めした」の項が`frame=0,N`と書いているのは、正しくは`frame=0,N−1`（Nはフレーム数）と読むこと。

**同時に直した同族のオフバイワン**（いずれも`provisional`側。ネイティブに閉じており、webUI・モックブリッジ・契約はいずれも無改修）:

- `provisional.h`の`ScannedObject`のdocstringが「half-open: [frame_start, frame_end)」と書いていたのを包含へ訂正。
- `provisional.cpp`の位置フォールバック（別名マーカーを消された仮オブジェクトを予約位置から拾い直す枝）の比較を`reserved_frame < o.frame_end`から`<=`へ。予約がオブジェクトの最終フレームちょうどに乗っている場合に取りこぼしていた。
- `provisional.cpp`の孤児掃除が報告する尺を`frame_end - frame_start`から`+1`へ。
- **仮オブジェクトも同じピンを共有していたため、従来から1フレーム過長だった**。置換時に実寸へ差し替わるので実害は軽微だが、A1で一括是正される（マーカーの尺と置換後の媒体の尺が一致するようになる）。

**副次的な便益**: `HasRoomForLength`は`[frame, frame + length − 1]`の範囲を検査しているのに、実際に作られるオブジェクトはそれより1フレーム長かった——**検査していない1フレームへのはみ出し**という潜在的な衝突源があったことになる。A1でこれが消える。

**ゲート**: native doctest **292ケース・アサーション1,504件で0失敗**（新規3本＝①121フレームで`frame=0,120`になること〔`R4_d_and_d.object`と同値。出典コメント付き〕②`length=1`で`frame=0,0`になる境界③`FindProvisionalIndex`で`reserved_frame == frame_end`が当たること）。**webuiは無改修のまま全緑（136ファイル・2,696件）・`npm run typecheck`も通過**——包含確定はwebui側の`frameEnd - frameStart + 1`が元から正しかったことの確認でもあり、フロント側は1行も変えていない。

**他文書の訂正**: [`SDK_REFERENCE.md`](SDK_REFERENCE.md) §16 (h)（全面書き換え・一次証拠の転記先）／[`BRIDGE_CONTRACT.md`](BRIDGE_CONTRACT.md) §4.13（未決→**包含で解決**）／[`TIMELINE_ALPHA_REQUIREMENTS.md`](TIMELINE_ALPHA_REQUIREMENTS.md) 第0節の決着注記／[`V2V_RIBBON_TRIM_WORKORDER.md`](V2V_RIBBON_TRIM_WORKORDER.md)の誘導行（凍結文書なので本文は不変・誘導行のみ差し替え）／バックエンド[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-140の追補。**なお§44.4の「`frameEnd`はinclusive」という記録は復権した。**

**教訓**: 前日の誤りは、ログの`end - start == length`という一致を**「端点規約の証拠」としてだけ読み、「オブジェクトの尺そのものの証拠」として読まなかった**ことに尽きる。一致は2通りに説明でき、そのうち採らなかったほうが正しかった。**実機が自分で書き出した現物（エイリアス4本）を先に見ていれば、ログの解釈に頼る必要はなかった**——ヘッダの数値は規約を直接示している。

**状態**: 追補の実装コミット済み（S1＝`9537855`〔`frame=`ヘッダの包含化＋テスト〕）。オーナーの実機ゲート（V-1〜V-3。**前提: プロジェクト24fps・生成fps24**）は**2026-09-04に全項目合格した**（361フレーム・24fps＝15.042秒の素材で実施）。V-1＝挿入動画が①二色リボン＋右クリック「音声を分離」あり②最終フレーム121（別素材でも確認）③エイリアス保存の3点合格。V-2＝✨予約の仮オブジェクトの最終フレームが要求どおり**361**（修正前は+1だった）。V-3＝✨→Generate→🎞置換の通し正常。監督が実機の`data\Alias\361_insert.object`と`361_d_and_d.object`を突き合わせ、ヘッダ`frame=0,360`（包含＝361フレーム）・`音声付き=1`・`再生位置=0.000,15.042`が**完全一致**（差はファイルパスのみ）することを確認し、挿入がD&Dと意味論同一になったことの直接証明を得た。**これで§3-140追補は全ゲート決着した**（記録はバックエンド[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-140の追補）。

## 108. 画角拡張の快適上限を、エンジン系統ごとの実測値へ置き換えた — 40,000 は「系統が分からないとき」の据え置き値になった（バックエンド台帳§3-134→§2-5）（2026-09-04）

### 108.1 結論

**画角拡張（Outpainting）の快適上限の警告に使うトークン予算を、エンジン系統〔ベースモデルの世代〕ごとの実測値へ置き換えた。** この予算は§66の実装以来ずっと**連結生成の予算からの流用**で、画角拡張自身の実測で較正したことが一度も無かった（設計書[`OUTPAINTING_DESIGN_NOTES.md`](OUTPAINTING_DESIGN_NOTES.md) §4-5が「流用」と明記していたとおりである）。今回そのワークロード自身で境界を測り、系統ごとの線を引いた。

**数値（系統ごとの予算・境界・掃引した点）はこの文書には書かない。** 予算の正本はバックエンド[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md) §9、較正そのものの記録は同[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §95である。本節はフロントエンド側の記録である。

**この線が警告専用・非ブロッキングであることは変えていない。** 超えても生成は止まらず、`outpaintGeometry.ts`の`outpaintReasons`にトークン予算の理由コードは無いままで、`editReasonMessages.ts`にも対応する文言は無い。**変わったのは線の位置だけ**である。

### 108.2 40,000 は消していない — 役割が「運用値」から「系統不明時のフォールバック」へ変わった

- **`COMFORT_TOKEN_BUDGET = 40,000`はそのまま残した。** ただし運用値ではなくなり、**エンジン系統が分からないとき**（`GET /models`が届く前・オフライン・表に無い系統）に使う据え置き値になった。**系統が判明していないセッションの挙動を、今日と1ピクセルも変えないため**である。この定数だけは実測値ではないので、実測の表に合わせて動かしてはならない——その旨をコード側のdocコメントに書いた。
- **系統ごとの表は凍結した定数である。** `OUTPAINT_COMFORT_TOKEN_BUDGETS`（`Object.freeze`）が系統キー（`GET /models`の`engine_family`）から予算を引く表で、`resolveOutpaintComfortBudget()`がその**ただ1つの入口**である。表に無い系統・`undefined`・空文字はすべてフォールバックへ落ち、値が数値でない・非有限・0以下のときも同じところへ落ちる。
- **サーバー配信にはしなかった。** §99でSingle／Chainedの線をサーバーの表（`limits.comfort_budgets`）へ移したのとは対照的だが、**その表に画角拡張の列は無い**（[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md) §9.4）。加えて`outpaintGeometry.ts`は**import ゼロ**を規律にしているモジュールなので、`shell/comfortTable.ts`から引く形も採らない。**フロントエンドが自分で持つ定数である**という事実を、そのまま定数の置き場所に写した。
- **`shell/comfortTable.ts`の`SINGLE_COMFORT_TOKEN_BUDGET`とは別軸である。** 同じトークンの式を使うが別のワークロードで（画角拡張は生成画素に加えて素材動画自身のVAE符号化と拡張マスクを抱える）、**統合してはならない**。片方の系統の値がたまたまSingleの線と同じ数になったので、**一方を他方で置き換えないこと**をコメントで名指しして書いた。

### 108.3 トークン数の式を1つに揃えた — 表示と判定の食い違いを先に潰す

- **`comfortTokenEstimate`を`shell/comfortTable.ts`の`comfortFramesForBudget`と同じ切り捨て形（`⌊幅/32⌋ × ⌊高さ/32⌋ × 潜在フレーム数`）へ統一した。** 32×32 の画素ブロック1つが潜在1マスなので、端数の画素はマスを増やさない。**この修正で数値が動くのは、128 の格子から外れた「編集途中」の表示だけ**である——生成できる唯一の形である 128 の倍数のキャンバスでは切り捨てが効かず、値は変わらない。台帳§3-134が着手前に「較正時にどちらの式へ揃えるかを先に決めること」と条件を置いていた項目であり、ここで決着させた。
- **`useOutpaintForm.ts`に手書きされていた同じ式を消した。** 警告文が引用する概算トークン数は`Math.round((canvas.width / 32) * ...)`とインラインで書き直されており、判定側の`isOverComfortBudget`とは切り捨ての有無が違っていた。**同じ式が2箇所にあること自体が、表示と判定を食い違わせる入口**なので、両方が`comfortTokenEstimate`ただ1つを通る形にした。
- **`isOverComfortBudget`は予算を引数で受ける署名に変えた。** 系統ごとに予算が変わった以上、閾値を関数の内側に固定しておく理由が無い。判定は渡された予算だけを見る（境界は`>`なので、ちょうど予算どおりは超過ではない）。

### 108.4 系統の配線 — 既存の道に1本足しただけ

`AppShell`はすでに`useBaseModels().activeEngineFamily`を持っており、§99でCreateとChainedの快適上限マーカーへ渡している。**今回はその同じ値をEditへも渡すだけ**で、新しい状態も新しい取得経路も作っていない（`AppShell` → `EditScreen` → `useOutpaintForm`）。

**プロパティは省略可能にした。** 省略・`undefined`・`""`はいずれも「系統が分からない」として`COMFORT_TOKEN_BUDGET`へ落ちるので、`EditScreen`や`useOutpaintForm`を直接描画する既存のテストは1つも書き換えずに通り、**2026-09-04 以前とまったく同じ挙動**になる。`""`を`undefined`と同じ意味で見るのは、`activeEngineFamily`が「まだ分からない」を空文字で表すためで、`shell/comfortTable.ts`の`resolveComfortRow`と同じ2値の読み方である。

### 108.5 触ったファイル

- **webui（製品コード）**: `modes/edit/outpaintGeometry.ts`（`OUTPAINT_COMFORT_TOKEN_BUDGETS`と`resolveOutpaintComfortBudget`の新設、`comfortTokenEstimate`の切り捨て統一、`isOverComfortBudget`の署名変更）・`modes/edit/useOutpaintForm.ts`（インライン式の解消と`engineFamily`の受け取り）・`modes/edit/EditScreen.tsx`・`shell/AppShell.tsx`（いずれも`engineFamily`の素通し）。
- **webui（テスト）**: `modes/edit/outpaintGeometry.test.ts`。系統ごとの引き当てと、表に無い系統・空文字・`undefined`・`Object.prototype`の名前がすべてフォールバックへ落ちること、**同じ幾何でも系統が違えば警告の出方が変わること**、128の格子から外れた寸法で各軸が切り捨てられること、ちょうど予算どおりが超過でないことを足した。**既存の期待値は新しい事実へ更新しただけで、条件は1つも緩めていない。**
- **文書**: [`OUTPAINTING_DESIGN_NOTES.md`](OUTPAINTING_DESIGN_NOTES.md) §4-5（2026-08-31の追記が「40,000をいまも使っている・較正の受け皿は台帳§3-134」という旧状態のままだったので、現状へ差し替えた）、本節。**数値と較正の正本はいずれもバックエンド側**である。

### 108.6 テストと状態

ゲート結果（2026-09-04）: `npm run typecheck` エラー0／`npm test` 136ファイル **2,700件 全緑**／`npm run lint` エラー0（警告は基準どおりで増減なし）。配置は`build.ps1`→`deploy.ps1`で、**1,267,712バイト・SHA-256 `fa179e34…`** がビルド出力・実機・バックエンドリポジトリ配布コピーの3値で一致している。

**オーナーの画面目視ゲート（V1〜V3）は未了である。** 手順と合格条件、および合格後にどの文書を直すかは、バックエンド台帳[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §2-5が正本である（§3-134からの移送先）。

**再較正の前提を1つ残してある。** Edit画面の画角拡張は生成要求にSettingsの加速設定を載せていないため、この較正は**サーバー既定の加速構成**で測ってある。**加速設定を載せる改修（同[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §1-27）が入ったら、この予算は要再較正である**（前提の正本は[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md) §9.2）。

### 108.7 追補（2026-09-04）— 画面目視V1〜V3が合格し、台帳§2-5はクローズした

**§108.6が「未了」と書いたオーナーの画面目視ゲートは、同じ2026-09-04のうちに合格した**（§108.6の記述は当時のまま残し、本節が上書きする）。**V1（LTX 2.3）は177コマで警告が出て169コマでは出ず、V2（LTX 2.5）は同じ177コマで出ず185コマで出た**——**系統を切り替えると同じ幾何で警告の出方が変わる**という§108.4の配線が、実機で意図どおりに効いている。**V3（系統が分からないとき）は「未確認のままの合格扱い」でクローズした**（オーナー裁定）——バックエンドが起動していない状態では素材をアップロードできず、**余白のスライダーそのものが画面に出ない**ため、実機ではこの枝へ到達できないからである。フォールバックの分岐自体は`modes/edit/outpaintGeometry.test.ts`（`undefined`・空文字・表に無い系統がいずれも`COMFORT_TOKEN_BUDGET`へ落ちること）が担保している。**目視結果の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §95.9、台帳の記録は[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-134**である（予算の数値の正本が[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md) §9であることは変わらない）。合格に伴い**台帳`PENDING_TASKS.md`の§2-5は節ごと削除された**ので、**§108.6が指している同書§2-5は現在は欠番である**——参照するときは上記のクローズ側へたどること。

### 108.8 追補（2026-09-05）— §108.6が残した再較正の前提は§110で消化した

**§108.6が「Edit画面が加速設定を載せる改修が入ったら要再較正」と残していた前提は、§110（2026-09-05）で消化した**——改修が入り、線は加速を全onにした構成で測り直されている（数値の正本はバックエンド[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md) §9のまま）。

## 109. A2Vの音声ファイルの場所を覚える仕組みが二重になっていたのを、1つへ寄せた — 契約v9時代のスヌープ機構の撤去（バックエンド台帳§3-36→§2-6）（2026-09-04）

### 109.1 結論

**Createタブの音声スロットが「音声ファイルがどこにあるか」を覚えるやり方が2つあったので、古いほうを撤去して1つに寄せた。** 残したのは`useSourceUpload`が状態として持っている`filePath`のほうで、撤去したのはファイル選択ダイアログの応答を横から覗き見て控えておく仕組み（`audioProbingBridge`／`lastAudioFilePathRef`）である。

**画面の見え方も、送る要求も、何ひとつ変えていない。** これは純粋な整理であり、**機能の追加も仕様の変更も1つも含まない**。台帳[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md)に「音声のfilePath機構を触るとき」という発火条件つきで置かれていたスタブ（§3-36）の消化である。

### 109.2 なぜ二重になっていたか — 順番の問題であって、設計の誤りではなかった

- **先にできたのは覗き見のほうである。** Chain由来の共有フック`modes/chained/useSourceUpload.ts`は、当初`fileName`しか外へ出しておらず、**wavの長さから尺を自動調整するために要るローカルのパスが、どこからも読めなかった**。そこで`modes/single/useGenerationForm.ts`は、ブリッジを薄く包んで音声のファイル選択だけ戻り値を覗き、その場所をrefへ控える形をとった。
- **あとから共有フック自身が`filePath`を持った。** 参照動画の尺表示（契約v9）を作ったときに、`SourceUploadState`へ`filePath`を足している。**ここで役割が重なった**——同じ事実を2箇所が別々に覚える形である。当時は動くものを壊さないために覗き見のほうを残置し、掃除課題として台帳へ記録した。
- **残っていたのはCreateの音声スロット1箇所だけだった。** Chain・Edit・そしてCreate自身の参照動画は、いずれもすでに`state.filePath`を読む形へ移っている。**Createの音声だけが古い作法のまま取り残されていた**、というのが着手時点の姿である。

### 109.3 寄せても安全だと言える根拠

**`useSourceUpload`は、状態が変わるときに必ず`filePath`も一緒に書く。** ファイル選択ダイアログ経由のpick、パスを直接渡す`uploadPath`（ドラッグ＆ドロップ・右クリック#3・#7）、リマウントをまたぐW5の`initial`シード——**旧refが覚えていた入口3つが、そのまま同じフックの`filePath`でも覆われている**。加えて、**同じファイルの中に手本があった**（参照動画の尺プローブはとっくにこの読み方をしている）ので、新しい作法を発明する必要も無かった。

**W4のspan cap（#7で切り詰めたリボンへ尺を追従させる仕掛け）の比較だけは、読む時点がずれないかを確かめた。** 旧コードは非同期の応答が返った瞬間のrefの現在値を読み、新コードはプローブを始めるときに控えたローカルの値を読む。**この2つが食い違う場面は存在しない**——音声が差し替わるとこのeffectは畳まれ、応答は`cancelled`のガードで止まって比較そのものに到達しないためである。理由が読み取れるようにコード側へコメントで残した。

### 109.4 `attachSourceAudioByPath`は消していない — もう1つ仕事を持っているため

**関数自体は残した。** パスの下ごしらえという仕事は無くなったが、**span capの捕捉（`a2vSeedCapPathRef`）というもう1つの仕事**を持っているからである。右クリック#7で音声を読み込んだとき、切り詰めたリボンの長さに尺が追従するのはこの捕捉のおかげで、**呼び出し側がここを飛ばして`sourceAudio.uploadPath`へ直接つなぐと、その追従だけが静かに壊れる**。撤去によって「なぜこの関数を経由しなければならないのか」の理由が1つ減ったので、**残った理由を関数のdocコメントと両方の呼び出し側のコメントへ書き直した**。

### 109.5 触ったファイル

- **webui（製品コード）**: `modes/single/useGenerationForm.ts`のみ。refと覗き見の包みを削除し、`useSourceUpload("audio", ...)`へ素の`nativeBridge`を戻し、パスを読んでいた2箇所を`sourceAudio.state.filePath`へ付け替え、プローブのeffectの依存へ同じ値を足し、使われなくなった型のimportを整理した。
- **webui（コメントのみ・実行されるコードは0行）**: `modes/single/GenerationForm.tsx`（ドロップ先の説明）・`modes/single/SingleScreen.tsx`（右クリック#7の説明）。どちらも「refを仕込むため」と書いてあった理由を、上の§109.4の理由へ差し替えただけである。
- **webui（テスト）**: `modes/single/useGenerationForm.test.ts`はコメントの現状化のみで、**期待値もアサートも1つも変えていない**。挙動不変という主張は、まさにこの「テストを1行も直さずに通った」という事実が裏づけている。

### 109.6 テストと状態

ゲート結果（2026-09-04）: `npm run typecheck` エラー0／vitest 136ファイル **2,700件 全緑**（実バックエンドを叩く`backend.integration.test.ts`は規律どおり除外。ポート18620が非リッスンであることを事前に確認した）／`npm run lint` エラー0・**警告32件で改修前と同数**（触った3ファイルからの警告は改修前後とも0件）。配置は`build.ps1`→`deploy.ps1`で、**1,267,200バイト・SHA-256 `88ea10c1…`** がビルド出力・実機・バックエンドリポジトリ配布コピーの3値で一致している。

**オーナーの画面目視ゲート（V1〜V3）は未了である。** 手順と合格条件、および合格後にどの文書を直すかは、バックエンド台帳[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §2-6が正本である（§3-36からの移送先）。**挙動不変のリファクタなので、目視は「壊れていないこと」の確認であって、新しい見え方を確かめるものではない。**

### 109.7 追補（2026-09-04）— 画面目視V1〜V3が合格し、台帳§2-6はクローズした

**§109.6が「未了」と書いたオーナーの画面目視ゲートは、同じ2026-09-04のうちに全項目合格した**（§109.6の記述は当時のまま残し、本節が上書きする）。**V1（ドラッグ＆ドロップ）・V2（#3「動画から音声を抽出してA2V」）・V3（#7の音声オブジェクト。さらにその音声のまま生成を1本走らせて正常に完了）のいずれも、音声カードにファイル名と長さが出て尺がその長さへ自動で動き、「今までと同じ」であることが確認された**——**新しい見え方が1つも無いことこそが合格条件**だった本項では、これが挙動不変という主張の実機側の裏づけである（機械側の裏づけは§109.5の「テストを1行も直さずに通った」）。**台帳の記録の正本はバックエンド[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-36-02**である（**同書の`旧§3-36`はモデルの再ホストというまったくの別件なので、番号だけでたどらないこと**）。合格に伴い**台帳`PENDING_TASKS.md`の§2-6は節ごと削除された**ので、**§109.6が指している同書§2-6は現在は欠番である**——参照するときは上記のクローズ側へたどること。あわせて、§2-6が「合格を待たずに直してよい」としていた[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §5-15の失効記述（撤去したスヌープrefを現行の仕組みとして書いていた箇所）も同日中に現状化された。

## 110. Edit画面（画角拡張・撮り直し）に、Settingsの加速設定が載るようにした — 塞いでいたのはシェルの配線1本（バックエンド台帳§1-27→§2-7）（2026-09-05）

### 110.1 結論

**Edit画面の画角拡張（Outpainting）と撮り直し（Retake）が、生成要求にSettingsの「生成高速化」（加速設定）を載せるようになった。** それまでこの画面から出したジョブだけは、利用者がSettingsで何を選んでいても**常にサーバー既定の構成**で走っていた。Create・Chained・バッチはいずれも`shell/accelerationSettings.ts`の`accelerationRequestFields()`——どの鍵をどんな条件で載せるかの契約の正本——を通っており、**Editだけがその契約の外にいた。**

**この改修は、画角拡張の快適上限の前提を変える。** 線はもともと「Editは加速設定を載せない」という当時の現状のまま測ってあったので、同じ日にバックエンド側で**加速を全部onにした構成での再較正**を行い、線を測り直している。**予算・境界の数値はこの文書には書かない**——**数値の正本はバックエンド[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md) §9、較正と改修の記録は同[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §98、オーナーの画面目視の受け皿は同[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §2-7**である。

**画面の見た目は変えていない。** Editに加速のつまみは置かず、Settingsで選んだ値が黙ってジョブへ載るようになっただけである。

### 110.2 なぜEditだけ抜けていたか — `AppShell`の1本の配線

**原因は単純で、`shell/AppShell.tsx`が`EditScreen`へ`acceleration`を渡していなかった**。値そのものは`AppShell`が既に持っており、Create・Chainedへは渡していた。**Editへ伸びる線が1本無かったので、その先の`useOutpaintForm`・`useRetakeForm`は加速設定を受け取りようがなく、`buildRequest`が`accelerationRequestFields()`を呼ぶ書き方にもならなかった**——つまり両サブタブが同じ理由で同時に抜けていた。

**Editが後から生えた画面であることが、そのまま抜けの理由である。** Create・Chained・バッチは加速設定の導入時（§55）に一斉に配線されたが、その後に育ったEditの2サブタブは、生成要求を自前で組み立てながらこの1本だけ受け取っていなかった。**「どのタブが契約を通っているか」を機械で確かめる場所がどこにも無かったこと**が、気づかれないまま残った原因である。今回そのガードを1本置いた（§110.4のT3）。

### 110.3 エンジンが断る設定をEdit側で弾かない理由

**Edit側にエンジン別のガードは足していない。** LTX 2.3 は`keep_resident_embeddings`の真を、LTX 2.5 は`vae_mode`の`prune_vaed`を、それぞれ422で断る。「送る前に画面側で弾くべきでは」という発想は自然だが、**採らなかった。理由は4つある。**

1. **断りの表は、単発・chainの両経路×両エンジンに対で実在する**（バックエンドの`services/engines/ltx/adapter.py`・`services/engines/ltx25/adapter.py`）。**契約の判定はそこが正本である。**
2. **`AppShell`は既に、サーバーが宣言する`unsupported_features`を見て違反する値を画面から隠し、設定を書き戻している**（§100.3で入れた作法）。**利用者が非対応の値を選んだまま生成へ進む経路が、そもそも塞がっている。**
3. **モデルの切替中は`serverBusy`でGenerateが凍る**（§98）。**「切り替えた直後の、古い値のまま飛ぶジョブ」も無い。**
4. **送出層で二重に担保しないことは`shell/accelerationSettings.ts`に明文で書いてある。** 送る側にもう1枚の表を置くと、**同じ契約の正本が2箇所になる**——片方だけが更新されたときに、静かに食い違う。

**足すべきガードは「エンジン整合」ではなく「配線が抜けていないこと」のほうだった**、というのが今回の判断である。

### 110.4 触ったファイル

**改修（コミット`5f4f0e4`・8ファイル）**

- **webui（製品コード）**: `shell/AppShell.tsx`（`EditScreen`へ`acceleration`を渡す1行。Create／Chainedと同じ並びに置いた）・`modes/edit/EditScreen.tsx`（受け取って両フックへ素通し。`exactOptionalPropertyTypes`が有効なので型は`AccelerationSettings | undefined`）・`modes/edit/useOutpaintForm.ts`／`modes/edit/useRetakeForm.ts`（`buildRequest`の要求オブジェクトの**末尾**へ`...accelerationRequestFields(acceleration)`を展開し、依存配列へ加えた）。**末尾に置いたのは、既存テストが`Object.keys`の並びで要求を比較しているためである。**
- **webui（テスト）**: `modes/edit/useRetakeForm.test.tsx`（既定なら鍵が1つも増えないこと＝バイト等価と、`sage`を選んだときに`attention_backend`だけが1鍵増えること）・`modes/edit/OutpaintingPanel.test.tsx`（実際に飛ぶ`/generate`の本文で同じ2件）・**`App.editRoute.test.tsx`（今回の欠落そのものの再発を防ぐ本丸）**。
- **文書**: [`API_REFERENCE.md`](API_REFERENCE.md) の送出点の契約文へ、Editの2サブタブを加える1文。

**T3（`App.editRoute.test.tsx`）が、シェルの配線を守る唯一のガードである。** `AppShell`を実際に描画し、localStorageへ`sage`を仕込んでから右クリック意図の発火→Editタブ→撮り直し→Generateまで通し、**ブリッジが受け取った`POST /generate/chain`の本文に`attention_backend: "sage"`が載ること**を固定した。**フックだけを直接描画するテストでは、今回抜けていた`AppShell`の1行は守れない**——だから実描画の通しにした。

**再較正の反映（コミット`3feb7f3`・2ファイル）**

- `modes/edit/outpaintGeometry.ts`（`OUTPAINT_COMFORT_TOKEN_BUDGETS`の**片方の系統の行だけ**を新しい実測値へ。あわせてdocコメントの日付と「この線は加速を全onにした構成で測った実測値である」という前提を書き直した）・`modes/edit/outpaintGeometry.test.ts`（期待値の差し替えのみ。**条件は1つも緩めていない**）。
- **`COMFORT_TOKEN_BUDGET`（系統が分からないときの据え置き値）と`resolveOutpaintComfortBudget`は触っていない。** 前者は実測値ではないので実測の表に合わせて動かさない、という§108.2の原則のままである。
- **加速設定に連動して線を切り替える仕組みは足していない。** 画角拡張の線はエンジン系統だけで決まる固定線のままで、Create／Chainedが持つ「構成が合わなければレガシー表へ落ちる」仕掛けは無い。**この設計のままでよいかはオーナー判断で、受け皿はバックエンド台帳[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §2-7 の判断J1である。**

### 110.5 テストと状態

ゲート結果（2026-09-05）: `npm run typecheck` **エラー0**（実装後・定数反映後の2回とも）／vitest **137ファイル・2,707件 全緑**（+10 skip。実装で**+6件**。定数反映は期待値の差し替えだけなので件数は増えていない。実バックエンドを叩く`api/backend.integration.test.ts`は規律どおり除外し、**ポート18620が非リッスンであることを事前に確認**している）／`npm run lint` **エラー0**（実装後は警告の出力が改修前とバイト一致、定数反映後は**警告32件で増減なし**）。配置は`build.ps1`→`deploy.ps1`で、**1,267,712バイト・SHA-256 `46F1AC00…D1BC`** がビルド出力・実機・バックエンドリポジトリ配布コピーの3値で一致している。

**オーナーの画面目視ゲート（V1〜V6）と判断（J1・J3）は未了である。** 手順と合格条件、および合格後にどの文書を直すかは、バックエンド台帳[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §2-7が正本である（§1-27からの移送先）。**予算の数値の正本は[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md) §9、較正と改修の記録は[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §98**である。

### 110.6 追補（2026-09-05）— 画面目視V1〜V6が合格し、台帳§2-7はクローズした

**§110.5が「未了」と書いたオーナーの画面目視ゲートは、翌朝2026-09-05のうちに全項目合格した**（§110.5の記述は当時のまま残し、本節が上書きする）。**画角拡張・撮り直しのどちらでも`metadata.json`の`*_used`がSettingsの選択どおりになり、既定へ戻せることも確認された。エンジンを切り替えても422は1件も出ていない**——**送出層にエンジン別のガードを置かないという§110.3の判断の、実機側の裏づけである。** 快適上限の警告位置も両系統で意図どおりで、**LTX 2.5 側では「再較正の前は警告が出ていた尺で警告が消える」ことが目視で確認された**（線が動いたことが画面から直接見える1点）。**判断J1・J3の裁定も同日に出ている**——J1は「両系統とも固定線のままでよい（実害が小さい）。**ただしこれは系統ごとの個別裁定であって、一般の設計思想にはしない**」である。**目視結果と裁定の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §98.11、台帳の記録は[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-143**である（予算の数値の正本が[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md) §9であることは変わらない）。合格に伴い**台帳`PENDING_TASKS.md`の§2-7は節ごと削除された**ので、**§110.4・§110.5が指している同書§2-7は現在は欠番である**——参照するときは上記のクローズ側へたどること。

## 111. 機能名→UIの宣言表を1枚にまとめ、画角拡張の予算を自前の表からサーバー配信へ移した（バックエンド台帳§3-135→§2-8）（2026-09-05）

### 111.1 結論

**フロントエンドが自前で抱えていた「モデルごとの知識」を2つとも手放した回である。**

1. **機能名→UIの宣言表`shell/featureScope.ts`を新設した。** サーバーが`unsupported_features`で名指しする機能名（`chain`・`outpaint`・`prune_vaed`…）が、どのタブ・パネル・サブタブ・Settings行を閉じ、閉じたあとにどの保存値をサーバー既定へ書き戻すかを、**1枚の表`FEATURE_UI`だけで決める**形にした。それまでこの知識は`shell/useBaseModels.ts`の4つの写像関数と`shell/AppShell.tsx`の2箇所の`unsupportedFeatures.includes(...)`に散っており、「機能Xはどこを閉じるのか」に答えるには全文検索するしかなかった。**サーバーの機能名を知ってよいファイルは、これで1つになった**——下流の画面はできあがった真偽値だけを受け取る。
2. **画角拡張（Outpainting）の快適トークン予算を、サーバー配信へ合流させた。** 自前の系統別定数表`OUTPAINT_COMFORT_TOKEN_BUDGETS`を廃止し、`GET /config`の`limits.comfort_budgets[系統].outpaint_budget`を`shell/outpaintBudget.ts`の`resolveOutpaintComfortBudget`1つで読む形にした。**単発・連結の予算がとっくにサーバー配信なのに、画角拡張だけが例外として残っていた**——その例外の解消である。
3. **隠した設定行の書き戻しを1本へ汎用化した。** `AppShell`にあった`useEffect`2本（PrunaVAED用・Embeddings processor常駐用）を削除し、**表の`resets`定義から導かれる汎用の1本**へ置き換えた。「どの行を隠すか」と「隠したとき何を書き戻すか」が別々の場所にあると、表に行を足したときに書き戻しの追加を忘れる——**保存値は`localStorage`に残り続けるので、隠すだけでは足りず、毎ジョブ422になる。**

**画面の見た目は何も変えていない。** 1と3は純粋な同値リファクタで、2も**線の値は1つも動いていない**（配る経路が変わっただけである）。

**数値・設計判断・ゲート実数の正本は、いずれもバックエンド側である**——予算の値は[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md) §9、実装と設計判断とゲートの実数は[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §99、オーナーの画面目視の受け皿は[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §2-8である。**本節へは書き写さない。**

### 111.2 `null`方式 — 「線が無ければ警告を出さない」

**`resolveOutpaintComfortBudget`の戻り値は`number | null`で、`null`＝線なし＝画角拡張の快適超過の警告を一切出さない。** 以前フロントエンドが持っていた据え置き値40,000（`COMFORT_TOKEN_BUDGET`）は、どこかへ移設したのではなく**概念ごと廃止した**——未較正の系統に仮の数字を当てると、画面に出る線は較正済みの線と見分けがつかないためである（オーナー最終裁定2026-09-05）。**webuiは`strict`と`noUncheckedIndexedAccess`が有効なので、`null`の処理漏れは`npm run typecheck`が機械的に落とす**——規則を型検査へ預けられることが、この方式のもう一方の値打ちである。

**解決器を`shell/comfortTable.ts`へ同居させず別ファイルにしたのは、Chained専用の`shell/tokenBudget.ts`が分かれているのと同じ軸の分離である。** この線は`rows`／`requires`を一切見ない固定線であり、単発・連結の予算とは別軸のワークロードを測ったものである（理由の正本は[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md) §9.4）。その2点は`comfortTable.test.ts`へ足した「`resolveComfortRow`は`outpaint_budget`を一切見ない」1本で機械固定した。

**旧テスト2本の期待値は、意図して反転させている**（「系統不明なら40,000の線で警告が出る」→「線なし＝警告なし」）。**実機の挙動を変えない根拠は§108.7のオーナー裁定**——バックエンドが起動していない状態では素材をアップロードできず、余白のスライダーそのものが画面に出ないため、この枝には実機で到達できない。**反転後の意図は、レイヤーを移して固定し直してある**（`shell/outpaintBudget.test.ts`の各分岐と、`modes/edit/OutpaintingPanel.test.tsx`の統合1本）。詳細はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §99.6。

### 111.3 表を読むときの約束

- **機能名のunion型は作らなかった。** 境界は`readonly string[]`のままである——**WebUIのビルドは相手のサーバーより古いことのほうが多く、知らない機能名が来るのは異常ではなく通常だからである。** 閉じるUI部品を持たない機能名（`two_stage_hq`・`nag`・`loras`・`sage_attention`…）は、空の`disables`で載せるのではなく**表に載せない。**
- **表が決めるのは「どこが閉じるか」だけである。** グレーにするのか行ごと隠すのか、添える注記の文言、サブタブが閉じたときの選択の退避先——**「どう見えるか」は描画する画面の側に残した。**
- **入れ物（`CONTAINER_TARGETS`）は単一パスで解く。** Editタブのように「配下が全部閉じたときだけ閉じる」ものは、`whenAll`に別の入れ物を名指ししてはならない——名指しすると答えが配列の並び順に依存する。**入れ子が要るようになったら、そのとき設計を見直す。**

設計判断の恒久記録はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §99.2〜§99.4が正本である。

### 111.4 テストと状態

**新しい基準を1つ確定させた。** vitestは**実バックエンドを叩く`api/backend.integration.test.ts`を除外した実行**が基準であり、**現在の値は138ファイル・2,734件 全緑・0 skipped**である。**従来あちこちが引いていた「137ファイル・2,707件＋10 skip」は除外していない実行の数**で、除外実行での旧基準は136ファイル・2,706件だった——**2つの物差しを混ぜて引き算すると、退行が無いのにファイルが減ったように見える。** `npm run lint`は**エラー0・警告31本**が新しい基準である（`AppShell`に常設だった`exhaustive-deps`警告1本が、書き戻しの1本化で依存配列を完全化したことにより消え、32本から減った）。`npm run typecheck`はエラー0。**工程ごとの内訳と、バックエンドのpytestを含むゲート実数の正本は[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §99.10である。**

配置は`build.ps1 -Config Release`→`deploy.ps1`で、**1,269,248バイト・SHA-256 `58BD0323…02D5`**がビルド出力・実機・バックエンドリポジトリ配布コピーの3値で一致している。

**オーナーの画面目視ゲート（G-V1〜G-V4）は未了である。** 手順と合格条件はバックエンド台帳[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §2-8が正本である（§3-135からの移送先）。**目視の前にバックエンドをC1以降のツリーで起動し直すこと**——`.aux2`だけが新しくサーバーが古いままだと、`outpaint_budget`が配信されず画角拡張の警告が出なくなり、**受容済みの退行を実装の不具合と誤診する**（[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §99.7）。

**なお§100.3が「表駆動化はやらなかった。……研究課題としてバックエンド台帳§3-135へ起票してある」と書いているのは2026-09-01時点の記述で、本節がそれを消化した**（当時の記述は残してある）。

### 111.5 追補（2026-09-05）— 画面目視G-V1〜G-V4が合格し、台帳§2-8はクローズした

**§111.4が「未了」と書いたオーナーの画面目視ゲートは、同日のうちに全項目合格した**（§111.4の記述は当時のまま残し、本節が上書きする）。**目視結果の詳細と設計判断の記録はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §99.13、台帳の記録は[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-135へ移送済み**である（予算の数値の正本が[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md) §9であることは変わらない）。合格に伴い**台帳`PENDING_TASKS.md`の§2-8は節ごと削除された**ので、**§111.4が指している同書§2-8は現在は欠番である**——参照するときは上記のクローズ側へたどること。

## 112. バージョンを1.0.0へ統一した — rc卒業と`.aux2`の再ビルド配布（バックエンド台帳§1-4→CLOSED §3-145。§3-5-02の併合分）（2026-09-06）

### 112.1 結論

**Publicリリースへ切り替えるというオーナー裁定に伴い、`1.0.0-rc1`をやめて`1.0.0`へ揃えた回である。** 直したのはバージョン文字列だけで、**振る舞いは1行も変えていない。**

**バージョンの一次ソースは3つあり、それぞれ別に管理されていた**——`native/src/bridge_core.h`の`kPluginVersion`（ブリッジがフロントエンドへ返す版）、`scripts/package.ps1`の既定`-Version`（配布パッケージの版）、`webui/package.json`の`version`（初期値`0.0.0`のまま放置されていた）。**この3つが同じ値になったことが、この回の本体である。** 3つ目は、バックエンド台帳が長らく別項目（同§3-5）として抱えていたもので、2026-09-03のオーナー裁定で§1-4へ併合され、ここで一緒に片づいた。

### 112.2 触った10ファイル

- **バージョンの一次ソース3件**: `native/src/bridge_core.h`（`kPluginVersion`）・`scripts/package.ps1`（既定`-Version`）・`webui/package.json`。
- **それに追随させた4件**: `webui/package-lock.json`（`npm install --package-lock-only`で追随させた。差分は冒頭2箇所の`version`フィールドだけである）・`native/tests/test_bridge_core.cpp`（`pluginVersion`をassertするCHECK）・`native/src/plugin.cpp`（AviUtl2へ渡すプラグイン情報の文字列）・`CMakeLists.txt`（コメント中の版表記2箇所）。
- **生きた文書2件**: [`BRIDGE_CONTRACT.md`](BRIDGE_CONTRACT.md)（`pluginVersion`の現状を述べている3箇所）・[`README.md`](../README.md)（Current releaseの表記と`-Version`の説明）。
- **成果物1件**: `AviUtl2-Plugin/NzVideomni.aux2`（再ビルドしたバイナリ）。

### 112.3 線引き — 「生きた記述」だけを直し、日付つきの歴史記述は据え置いた

リポジトリ全体を`1.0.0-rc1`でgrepしたうえで、**「現在の値はこれである」と述べている記述だけを直した。** 直していないのは次の2種類である。

- **追記専用の記録簿**（本書・バックエンドの[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md)・同[`HANDOFF_ARCHIVE.md`](../../../Docs/HANDOFF_ARCHIVE.md)）。
- **日付つきの歴史記述**（[`API_REFERENCE.md`](API_REFERENCE.md)冒頭の変更履歴・[`DEVELOPMENT_PLAN.md`](DEVELOPMENT_PLAN.md)の「史料として原文のまま残す」節・[`WEBVIEW2_PARITY_BACKLOG.md`](WEBVIEW2_PARITY_BACKLOG.md)・バックエンド[`FRONTEND_CATCHUP_WORKORDER.md`](../../../Docs/FRONTEND_CATCHUP_WORKORDER.md)の各追記専用ログ）。

**「1.0.0-rc1で実装を完了した」という当時の事実を1.0.0へ書き換えると、記録そのものが嘘になる。** 版数の一斉置換でいちばん壊しやすいのがここなので、**判断の軸は「その文が現在を述べているか、過去を述べているか」の1つだけに置いた。**

### 112.4 ゲートと配置

- `npm run typecheck` **エラー0**。
- vitestは、実バックエンドを叩く`api/backend.integration.test.ts`を除外した基準の実行（§111.4）で**138ファイル・2,734件成功・0 skip**——基準どおりである。**1回目の実行で`App.nag.test.tsx`内の1件が落ちたが、単体でも全体でも再実行すると成功し、落ちる箇所も実行ごとに違ったため、既存のflakyと判断した**（版数の変更に由来するものではない）。
- `npm run lint` **エラー0・警告31本**——これも基準どおりである。
- ネイティブ側のdoctestは**292ケース・1,504アサーション全成功**（`pluginVersion`のアサーションの更新を含む）。
- 配置は`build.ps1 -Config Release`→`deploy.ps1`で、**SHA-256 `EDD1411188F3C9A63232CC74CC28850E62408904BA26CB5B9387FE8677A89B73`**がビルド出力・実機・バックエンドリポジトリ配布コピーの3値で一致している。

台帳の記録はバックエンド[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) **§3-145**（§1-4のクローズ記録であり、§3-5-02の併合分の完了記録でもある）。実装そのものの正本はコミット`20a7b00`である。

## 113. 素材（冒頭）と素材（末尾）の併用を止めていた鏡ゲートを撤去し、遡り生成の規則を1つに束ねた（バックエンド台帳§3-90→§2-9）（2026-09-07）

### 113.1 結論

**フロントエンドがやったことは、引き算1つと集約1つである。**

1. **理由コード`endSourceWithSourceVideoMultiClip`を、参照ごと削除した。** これはサーバーの422を先回りして映していた鏡のゲートで、素材（冒頭）と素材（末尾）を両方付けたままクリップを2本以上にすると「クリップは1本だけです」と言って Generate を止めていた。**サーバーがその組み合わせを受理するようになったので、鏡そのものが不要になった**。落としたのは、型の union・算出・`validityReasons`への積み上げ・`buildRequest`が素材（末尾）を落とす防御条件・英日の文言・`generateReasonMessages`の配線である。**新しい文言・要素・配置規則は1つも足していない。**
2. **「遡って生成するのはどういうときか」という規則を、`isReverseEndSource`という1つの真偽値に束ねた。** 定義は「素材（末尾）があり、クリップが2本以上で、**素材（冒頭）が無い**」で、サーバーがモードを決めている1箇所の規則をそのまま写したものである。**画面側でこの規則を使う場所は3つあり、いずれもこの1つを見る。** のりしろの既定値を1へ寄せる一発遷移、素材（末尾）の複数クリップ品質警告、そして`ChainedScreen`の遡り生成のヒントである。**素材（冒頭）を付けたブリッジモードは、この3つのどれから見ても普通の正順チェーンである。**

**契約とゲート結果の正本はバックエンド側である。** 契約は[`API_REFERENCE.md`](API_REFERENCE.md) §5.2、実装と実測とゲートの実数はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §102、オーナーの目視の受け皿は同[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §2-9である。**本節へは書き写さない。**

### 113.2 一発遷移は、参照の初期値と本体の両方を同じ規則へ差し替えた

のりしろの既定値は、対称な一発遷移で動く。遡り生成の状態へ**入った瞬間**に1へ、**出た瞬間**に通常の既定値へ戻る。この遷移は`useRef`が「1つ前の状態」だけを覚えることで、レンダーのたびに利用者と綱引きしないように作られている。

**ここで直す場所は2つあり、片方だけでは壊れる。** `useRef`の**初期値**（マウント時の基準）と、`useEffect`の中の**現在の判定**の両方が同じ規則を持っていなければならない。片方だけを新しい規則にすると、素材（冒頭）と素材（末尾）を両方持った2本以上の状態でフォームが最初に描かれたときに基準と判定が食い違い、**誰も頼んでいない寄せが初回に1回だけ走る**。計画段階の敵対的レビューがこの見落としを拾い、実装では両方を`isReverseEndSource`へ差し替えてある。

### 113.3 エスカレーション2件（記録として残す）

- **「素材つきの状態でマウントする」テストは書けなかった。** 上の初期値の証明として「素材（冒頭）＋素材（末尾）＋2本の状態でマウントしたフォームは、初回に寄せを走らせない」というテストを予定していたが、**このフックの入口からはその状態を作れない。** 素材は非同期の添付操作を通してしか入らないので、マウント時点では必ず素材なしから始まる。**到達できない状態のテストは書かず**、代わりに「素材（冒頭）を先に付けてから2本目を足しても、のりしろに1が一度も現れない」という**同じ壊れ方を捕まえる**テストを置いた。
- **配布スクリプトは PowerShell ツールから起動できず、Bash 経由で走らせた。** `scripts\deploy.ps1` はこのセッションの PowerShell ツールでは実行を断られたため、Bash から `powershell.exe` を呼ぶ形で実行した。**配布物の中身も配置先も通常どおりで、SHA-256の3値一致で確認している**（次項）。同じ壁に当たったら、この回避で通る。

### 113.4 テストと状態

- vitestは基準の実行（§111.4。実バックエンドを叩く`api/backend.integration.test.ts`を除外）で**138ファイル・2,737件 全緑**（基準の2,734件に対し +3）。**5件を足し、規則が消えて意味を失った2件を落とした**という内訳である。足した5件は、意味を反転させた2件（「両方は送れない」→「**`buildRequest`の結果に`source_video`と`end_source`の両方が載り、`validityReasons`が空**」／「素材（末尾）と2本以上が揃えば必ず警告」→「**素材（冒頭）が付いていれば遡り生成ではない**」）と、新設の3件（素材（冒頭）を付ける・外すでのりしろが3と1を往復すること／素材（冒頭）を先に付けた場合にのりしろ1が一度も現れないこと／`ChainedScreen`で素材（冒頭）を付けると遡り生成のヒントが消えること）である。**モック経由の受理テストは作っていない。** `mockBridge.ts`にはこの組み合わせの拒否がそもそも無く、通っても何の証明にもならないためである。
- `npm run lint`は**エラー0・警告31本**、`npm run typecheck`はエラー0で、どちらも基準どおりである。**件数の基準と、バックエンド側を含むゲートの実数はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §102が正本である。**
- 配置は`build.ps1 -Config Release`→`deploy.ps1`で、**SHA-256 `BEDEEDD4C6DE4F9401991C635F9402DFFD51ABAC96B4BDF4AB7A8B54B5369E78`**がビルド出力・実機・バックエンドリポジトリ配布コピーの3値で一致している。
- **オーナーの画面目視は未了である。** 手順と合格条件はバックエンド台帳[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §2-9が正本で、**目視の前にバックエンドを作業ブランチ`feature/start-end-bridge`のツリーで起動し直すこと。** 古いサーバーのままだと、画面が送れるようになった組み合わせを422で断られ、実装の不具合と読み違えてしまう。

### 113.5 追補（2026-09-07）— オーナー裁定で一括受容され、台帳§2-9はクローズした

**§113.4が「未了」と書いたオーナーの画面目視は、同日のオーナー裁定で一括受容された**（§113.4の記述は当時のまま残し、本節が上書きする）。裁定は「今の生成結果も仕様として受け入れる。そもそも素材（冒頭）と素材（末尾）の併用は実験的な機能なので、現状で受容する」というもので、**画面側の確認項目U1〜U3も生成結果の目視V1〜V4も、個別の報告ではなくこの裁定でまとめて閉じている。** **裁定の記録はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §102.13、台帳の記録は[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-90へ移送済み**である（早期収束の観察と考察は[`CHAIN_STAGE2_RESEARCH_NOTES.md`](../../../Docs/CHAIN_STAGE2_RESEARCH_NOTES.md) §11の【2026-09-07】の項）。合格に伴い**台帳`PENDING_TASKS.md`の§2は節ごと削除された**ので、**§113.1と§113.4が指している同書§2-9は現在は欠番である**。参照するときは上記のクローズ側へたどること。
