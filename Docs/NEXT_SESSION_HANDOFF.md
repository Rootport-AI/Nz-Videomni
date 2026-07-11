# 次セッション引き継ぎ書 — LTX 2.3 を 16GB で動かす

---

## ▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶ 最新ステータス（2026-07-11 **A2V＋LoRA併用の解禁＝GPU実機目視ゲート含め完了（ユーザー受容）＋ジョブ起動ログ拡充＋GUIバグ2件修正（queued詰まり解消・ref_video無効化）**＝客観ゲート PASS（pytest 511 passed / 1 skipped＝512件）・独立レビュー2巡 must-fix残存なし・**次セッション＝コミット／プッシュ（オーナー判断）**）

> **本ブロックが最新の正本。** これ以降の▶節はすべて歴史記録。食い違ったら本ブロックが正。詳細＝`VERIFICATION_LOG.md` §32（A2V＋LoRA併用の解禁・目視ゲート✅クローズ）・§33（ログ拡充＋GUIバグ2件修正＋JobStore CAS化）。

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

### 次セッションの課題

1. **筆頭課題＝コミット／プッシュ（オーナー判断）**。本セッションの変更一式（A2V＋LoRA併用の解禁・ログ拡充・GUIバグ2件修正・JobStore CAS化）はすべて未コミットのまま。
2. **残課題＝reference付き IC-LoRA（参照動画による条件付けアダプタ＝control系）のチェーン対応は未実装**。§32.1 の決定どおり、チェーン（`POST /generate/chain`）は `reference_video_id` を持たない構造のため v1 のスコープ外とした。
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
- **I2V マルチジョブ＋音声 連続 実機 PASS**（2026-07-02）=§10.7。**音声 Phase 1**（native joint audio・AAC/48kHz/stereo・crop 後も保持）=§9.8。
- **load/encode 16GB fit**: `--te-offload`（Gemma encode ピーク低減・既定 ON）=§11／`--dit-cpu-load`（transformer load スパイク除去・既定 ON・transformer ブロックを直接 CPU 構築）=§12＝512×320 で whole-job ceiling 16,944→~9.2GB。残るのは den2（denoise stage2）の解像度×尺スケーリング軸のみ（バグでない・`RESOLUTION_DURATION_CAPABILITY.md §8` が正本）。
- **残課題A（worker 再利用 crash）解決**=§9.7: 真因＝`BlockSwapService._installed_transformers` の per-job リーク（append し続け解放しない）。修正＝install() で append 前に `clear()`（keep-latest）＋`_do_generate()` 完了直後に `gc.collect(); torch.cuda.empty_cache()`。6ジョブ実機 PASS・出力 byte 一致。真因確定の経緯（当初 mmap 破損説→棄却→Windows commit（仮想メモリ）枯渇で確定）=§8。
- **de-fork（engine first-party 化）**=§13: 旧同梱フォーク backend→`engine/` パッケージ（`worker`/`pipeline`/`gguf`/`gemma`/`transformer`）へ採用・アルゴリズム不変・43GB モノリス物理削除（`checkpoint_path` は reference-only 化）・全段 SHA256 一致。**QAT 回収**=§14: Gemma を text-only（`Gemma3ForCausalLM`・vision 無し）化し 22.7GB の QAT dir を物理削除・gemma_root は ~40MB tokenizer-only dir（`models/gemma-3-12b-it-tokenizer/`）に差替え・models/ 50.9→28.15GB・byte 完全一致。
- **keep=1 常駐モード＝調査 CLOSE**（16GB で原理的 non-viable・利得は既存経路で捕捉済み・既定 keep=0 不変）=§15。**dead-code 整理**（text-only/de-fork の残 no-op 除去・byte-match ゲート）=§16。**Phase 5(A) 配線**（Approach W 常駐サブプロセスワーカー・`@@LTX@@` JSON-lines プロトコル・app `.venv` は torch 非 import）=§6・**Phase 5(B)**（denoise 直前 `empty_cache` で shared 溢れ 3969→742MB）=§7。
- **install スクリプト**: `scripts/install_ltx.ps1` は冪等クリーンインストーラ（2026-07-02 全面書き換え済＝現行の正・両venv・現行~28GBのみDL・GpuArch 自動判定）。互換メモ＝torch は必ず cu128 index から／attention＝**SDPA 固定**（xformers/flash-attn/sageattention は足さない）／Blackwell は R570+ ドライバ。

### 凍結してある契約・構成（不変・壊さない）

- **凍結 API 契約**: ÷64 解像度（`api/models.py`）・8n+1 フレーム・T2V/最小I2V（frame_idx0・conditioning≤1）・distilled 8step/CFG1.0・`GET /status` の `vram_optimization`（`services/low_vram.py` `_STATUS_KEYS`）・`metadata.json` スキーマ・limits/generation_presets。**加算的変更のみ許可**（optional フィールド・省略時 byte 同一が定型ゲート）。
- **2プロセス・2venv**: app=`./.venv`（torch 無し・FastAPI/Gradio/mock backend）／engine=`./.venv-engine`（torch 2.9.1+cu128＋`ltx_core`/`ltx_pipelines`@`00dc53d`＋`gguf`）。同一インタプリタで共存させない（双方が `services` トップレベルパッケージを持つため）。
- **本番 env**: `LTX_KEEP_RESIDENT=0` 既定（keep=1 だと 720p の Gemma 移動で native crash・§10.2）／`use_component_files: true`（Path B）／`te_offload`・`dit_cpu_load` 既定 ON。設定は `config.yaml` が正。起動＝`python -m engine.worker`（別プロセス・別venv・`cwd=root`）。
- **本番モデル（実行に要る ~28GB）**: GGUF transformer Q4_K_M ~17GB＋GGUF Gemma Q4_K_M ~7.3GB（`ggml-org/gemma-3-12b-it-GGUF`）＋component VAE/audio/projection ~3.9GB＋spatial upsampler ~0.95GB＋tokenizer-only gemma_root ~40MB。43GB モノリス・QAT dir は削除済。
- **回帰基準 SHA**: T2V(seed=12345, "a calm ocean wave…") 512×320/49f=`23844b4e…6bb7bf`／最小I2V=`a511eda4…c217`／peak_vram_mb 8440。IC-LoRA 付き出力の基準 SHA=`735a6de9…272`（旧 `outputs/ic_lora_phaseA/spike.mp4` `8e10aa59…` は旧コード出力＝照合に使わない）。

### アーキテクチャ（二層・壊さない）

- **凍結層**: REST API 形（`api/models.py`）・ジョブ管理（in-memory 単一・実行中 busy 409）・出力 `outputs/{job_id}/output.mp4`+`metadata.json`（metadata は `pipeline_manager._finalize()` が書く）・`LTXRunner.generate(...)→GenerationOutcome` 契約・mock backend（torch 無し app `.venv` のテスト用・温存）。
- **エンジン層**: `engine/` パッケージ。公式 `DistilledPipeline`(ltx_core@00dc53d) をラップし GGUF transformer+block-swap・GGUF Q4 Gemma（per-layer dequant で GPU 推論）・VAE/attention tiling を配線。常駐 worker 経由で `services/ltx_runner.py` の `_RealBackend` から呼ぶ。

### 将来項目・未着手（現行スコープ外・記録のみ）

- **Phase 2 ＝ AviUtl2 拡張機能統合**（＝本プロジェクトの最終ゴール・「早く統合して使いながら育てる」early-integration 方針）＝ユーザーのプラグイン開発環境整備待ちでブロック中。REST API は AviUtl2 専用にしない（DaVinci Resolve 等も想定）。spec §0.1/1.3。
- **Phase 3 残パリティ**（未着手・spec §13.4）: Gap Fill／Retake（大規模）・生成キュー・延長尺(~30s)・text-only プロンプト強化・negative/CFG/STG/sigma schedule/denoise loop/seed lock 露出・空間アップスケーラのユーザー操作露出・LoRA/attention tiling 再導入（de-fork で削除）。**negative/CFG/pipeline は worker 未配線＝GUI 露出禁止（継続）。**
- **高品質モード（`two_stage_hq`・pipeline/guidance_scale 消費）**＝遠い将来。現状 `services/ltx_runner.py` の payload がこれらをエンジンに渡さず常に distilled 経路のため GUI は当該オプションを表示するが無効化＋「バックエンド対応待ち」注記。着手の入口＝非蒸留×量子化 dev 重み（`unsloth/LTX-2.3-GGUF` の `ltx-2.3-22b-dev-Q4_K_M.gguf` 等）＝[`FEATURE_RESEARCH_2026-07-04.md`](FEATURE_RESEARCH_2026-07-04.md) D節。
- **VLM(vision) 再導入**（enhance_i2v・フレームを見た Gap Fill 提案）＝QAT text-only 化を巻き戻すため計画外・要件化時に別途判断。
- **やらない（削除済みスコープ・ユーザー決定・spec §13.5）**: ①1080p アップスケール「機能」（＝外部ツール推奨。内部二段 upsampler は生成の仕組みゆえ残す）②多人数インフラ（本格ジョブキュー/認証/インターネット公開/永続 DB＝単一ユーザー想定で不要・現状の「1ジョブ＋busy 409」が正しい設計）。

### 運用ルール（最新の正・memory にも記録）

- **目視検証**: 720p級（1280×768）以上＋映画トレイラー風プロンプト＋「賑やかな町＋セリフ」題材（512×320級は顔溶けで判断不能）。客観PASS（byte-match・pytest・メトリクス）とユーザー目視ゲートを混同しない。実験前に仮説→Web/コードで裏取り（手当たり次第禁止）。
- **サブエージェント**: Opus 以下（**Fable5 禁止**）・非破壊・**能動ポーリング監視**（ウォッチャー待ち停止禁止）・異常時は続行せず報告。GPU 計測は dedicated+shared 両監視（一次ソース＝`ltx_worker.log` の peak_vram_mb・道具=`_gpu_mem_sampler.ps1`）。
- **人間向け説明**: 内部略語禁止だが「普通のまともな日本語」で（過剰な言い換えも逆効果）。判断を仰ぐ前に機能説明を届ける。解説文書は Opus サブエージェントに執筆させ監督はレビュー。
- **作業原則**: コード着手前に実装計画を提示して合意・テストが落ちたら勝手に直さず原因分析して報告・先行事例を複製し独自発明しない・push/マージはユーザー承認・環境隔離厳守（システム Python 不可触・すべてプロジェクトローカル venv）。

### 旧・詳細設計/リファレンスのポインタ

- LTX 2.3 一般リファレンス=[`LTX23_REFERENCE.md`](LTX23_REFERENCE.md)（解像度契約 ÷32/÷64・2段・VAE 32×圧縮・VRAM 重み支配・720p=1280×768→crop）。設計比較=[`DESIGN_COMPARISON_and_direction.md`](DESIGN_COMPARISON_and_direction.md)（fp4_mixed 不採用・GGUF Q4 採用が結論）。公式 offload+fp8 戦略の破棄理由=[`note.md`](note.md)。設計正本=`../LTX23_Backend_Specification.md`（v0.5）。
