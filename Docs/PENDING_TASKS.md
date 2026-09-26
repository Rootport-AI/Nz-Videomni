# 未着手タスク台帳

- 作成: 2026-07-15／最終更新: 2026-09-27（§3-168 の状態欄を現況に更新——C-3〔`asym_w4a8_int8`〕までコードが完結し dev `c18d9aa` に積んであること、客観検査 G1・G2 は合格、実機の門 G3〜G8 はサーバー起動待ちであること、C-4〔較正〕は手順 v2 まで確定していることを反映。§1-31 の判断材料に、較正値の材料は[`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md) 第14節〔C-4 の後〕である旨の参照を追記。前回 2026-09-26: §3-168「ComfyUI 標準の int8 safetensors を置くだけで使えるようにする」を起票し、§3-167 の状態欄に int8 の扱いを §3-168 へ移した旨を追記。同日それ以前: §1-31 の判断材料に、操作パネルの快適上限表へ暫定で足した fp8 の 2 列も吸収する旨を追記。さらにそれ以前: §1 を立て直し §1-31「快適上限のマニフェスト一本化」を起票し、§3-167 の判断事項 2 はそこへ移した）
- 位置づけ: **セッション開始時に「次に何をすべきか」を確認するための台帳であり、セッションの入口は本書ただ 1 つである**（引き継ぎ専用の文書＝`NEXT_SESSION_HANDOFF.md`・`NEXT_SESSION_WORKORDER.md`のような役割の重複する文書は、新設しない）。プロジェクト全体（バックエンド `Nz-Videomni` と、フロントエンド `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2`）の課題をここへ一本化している。優先度の高い順に次の4つへ分ける（**運用規則の正本は末尾「本台帳の位置づけ（運用規則）」節**）。
  1. **近日中の改修項目** — 実装・修正の内容が具体的で、まだ着手していないもの。**全項目が片づいて空になったら、本節は見出しごと削除する**（次に着手すべき項目が出た時点で節ごと立て直す）。**現在は §1-31 が立っている。**
  2. **実装済み・ユーザーのテスト待ち** — 実装は済んでいて、オーナー本人の実機・目視・実GPUテストが未了のもの。書式は**チェックリスト形式**である——各項目を「何を操作して確認するか → どうなれば合格か」の1〜2行にし、`- [ ]`の箇条書きを画面・機能ごとの小見出しでまとめる。テストではなく仕様の是非をオーナーが判断する項目は「オーナー判断待ち」の小見出しへ分ける。**全項目が合格して空になったら、本節は見出しごと削除する**（次に確認待ちの項目が出た時点で節ごと立て直す）。**現在は該当項目が無いので削除してある。**
  3. **将来の研究課題** — 調査・検討段階の大きめのテーマ。着手時期は未定。冒頭に、オーナーが指定した階層「将来の改修項目＞将来の研究課題」に従って**改修項目のグループ**を置く。
  4. **スコープ外（さらに先の将来）** — §3よりもさらに優先度が低く、当面は着手しないと判断したもの。前提が変わったときに読み返すための置き場。
- **完了してクローズした項目は本書に残さず、[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md)へ移す。** 本書の「3.」と同書の「3.」は別物なので、**参照するときは番号だけで書かず、必ずファイル名を添えること**。
- **旧番号（旧§3-xx・旧§4-xx等）の読み替えと欠番の対応は、[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md)冒頭の「『旧』ラベルの定義と本書の成り立ち」を参照。**

---

## 1. 近日中の改修項目

実装・修正の内容が具体的で、まだ着手していないもの。全項目が片づいて空になったら、本節は見出しごと削除する。

### 1-31. 快適上限のマニフェスト一本化（起票：2026-09-26）

- **概要**: 快適上限の線（配信値）を `config.py` の既定値から、ベースモデルごとの定義ファイル（`scripts/manifests/*.json`）へ移して正本を 1 箇所にする。**同じ作業の中で、読み込んだ重みファイルの種別（GGUF の量子化・fp8）ごとの行を足し、§3-167 B-3 の較正値を配信値へ反映する**（[`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md) 第13節）。サーバーが選択中の transformer の種別を `GET /models` で名乗り、操作パネルと Gradio は既存の「加速設定で行を照合する」仕組みに種別を 1 項目足すだけにする。UI の隠し方の表（`featureScope.ts`）は触らない。
- **判断材料**: 反映できる実測は LTX 2.5 の fp8（快適側の上端 38,304〜39,424・幾何差は潜在 1 コマ未満で単一の値で表せる）と Q6_K の 2 件（同 第10節＝REDGraft 2.5 は解像度で割れる・第11節＝Sulphur 2.3 は線は動かないが w46 に孤立した溢れ）。LTX 2.3 の fp8 は全 on を測り切れていない（第13.2節）。種別の数は絞る（規則は単純に・例外を増やさない）。`config.yaml` の表ごと上書きを残すか 1 本に絞るかは設計時に決める。パネル側の手書き例外（`comfortDisplayTable.ts` の Q6_K 3 点）はこの表に吸収する。fp8 の 2 列（2.5 の固定の線・2.3 の 1 点。フロントエンド `Docs/DEVLOG.md` §126）も吸収する。**int8 系（`int8_tensorwise`・`asym_w4a8_int8`）の較正値の材料は、§3-168 の C-4 較正が終わったあとの [`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md) 第14節です**（C-4 の手順は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §121.11 が正本）。
- **出典**: [`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md) 第10.4・11.4・12.4・13節、[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §119.5、`config.py` の `_default_comfort_budgets()`・`services/base_models.py`・`services/model_registry.py`（`GET /models`）、`webui/src/shell/comfortTable.ts`・`comfortDisplayTable.ts`、`gradio_ui/comfort.py`、`tests/test_comfort_budgets.py`（キー集合の固定）。

---

## 3. 将来の研究課題

調査・検討段階の大きめのテーマ。着手時期は未定。欠番の対応は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md)冒頭を参照。

### 改修項目（研究課題より優先。オーナー指定の階層）

オーナーが指定した優先順位の階層「**将来の改修項目＞将来の研究課題**」のうち、上位にあたるグループ。

#### 3-53. 生成時間の小粒最適化の積み上げ（起票：2026-08-03）

- **概要**: 生成時間を1〜2秒級の小粒最適化で積み上げて縮めるテーマ。**分解計測は完了している**（内訳＝テキストエンコード／stage1／アップサンプル／stage2／VAEデコード／音声はバックエンド[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §50、**現ベースライン119.20秒は同 §52.8**が正本。§50の146〜147.60秒は`fused_gguf_dequant_kernel`が既定onになる前の値なので混同しないこと）。**分解計測もベースラインもLTX 2.3のものだけで、LTX 2.5側には同種の内訳が無い**——2.5で本項を扱うなら分解計測から始めること。
  - **映像VAEデコードは決着済み**（PrunaVAED＝[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-66。短縮幅の正本はバックエンド[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §52.8）。**ただし既定OFFの選択制**なので、既定のジョブの分母は縮んでいない点に注意。
  - **残る最大の候補は動画エンコード（x264）である。** 「デコードが速くなってもエンコード側が律速になる」という懸念は実測では顕在化しなかった（同§3-66のG6）ため、**エンコード側には手つかずの余地が残っている**という読みになる。
    - **着手の入口（2026-09-03調査）**: エンコードの実体は固定ホイール`ltx_pipelines/utils/media_io.py`の`encode_video`（呼び出し口は`engine/pipeline/common.py`の`encode_video_output`）。PyAV＋libx264で**preset/crf未指定＝x264既定のまま**・デコードと同一スレッドで交互実行＝**並行化なし**。改修するならラッパーの中身を自前実装へ差し替える確立パターン。**NVENC（h264_nvenc）はGPUの専用エンコーダASICで動くためCUDA演算と競合せず、CPUエンコード比で数倍高速になることも珍しくない。**
- **候補一覧の正本**: バックエンド[`ACCELERATION_RESEARCH_NOTES.md`](ACCELERATION_RESEARCH_NOTES.md)。
- **状態**: 将来の改修項目（着手はオーナー判断待ち）。

#### 3-149. Toolbox（物体追尾）のマスク形状の変化を滑らかにする（起票：2026-09-14）

- **概要**: 物体追尾は1フレームごとに枠の形状を決めるため、動画によってはマスクの縁が痙攣するようにちらつく。直前の形状から滑らかに変化させる方法を研究する。
- **出典**: オーナー発案（Inpainting実機ゲート中の観察）。
- **状態**: 将来の研究課題（着手時期未定）。

### 研究課題（上の改修項目より優先度が下）

#### 3-1. バッチA2Vのα版で意図的に省略した機能

バッチA2V（音声フォルダを丸ごと指定し、就寝中などまとまった時間に複数のA2Vジョブを順に流す機能。2026-09-15からi2vモードも同じパネルに乗る。正本は`AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/BATCH_A2V_I2V_MODE.md`）にα版として実装していない機能のうち、現在も残っているのは次の3点。出典: [`DEVLOG.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/DEVLOG.md) §9.8、[`WEBVIEW2_PARITY_BACKLOG.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/WEBVIEW2_PARITY_BACKLOG.md)「既知の意図的α省略」、`webui/src/modes/batch/`（`buildA2vChainPayload.ts`・`useBatchForm.ts`の実装ノート）。

- **バッチでの参照動画**: IC-LoRA用の参照動画系フィールド（Gradio原典`batch.py`の`use_adapter`／`ref_video_path`／`control_adherence`／`reference_strength`）に相当する設定がバッチ経路に無い（`webui/src/modes/batch/batchRunner.ts`の実装ノートに「No reference-video (control IC-LoRA) adapter support」と明記）。
- **開始前の一括妥当性検証（preflight）**: Gradio原典の`_validate`相当にあたる、画像/プロンプトのfoolproof preflightは未実装。
- **行ごとのAdd／Replace切替**: プロンプトと追加プロンプトの合成方式（Add＝連結／Replace＝行で置き換え）は、現状バッチ全体で1つの設定（`promptMode`）であり、行ごとには切り替えられない。

なお本節の項目は、オーナー方針（Gradio同梱UIとのパリティは最終的に全項目を実装対象とする）のもとでは、いずれ実装側へ戻る前提である（[`WEBVIEW2_PARITY_BACKLOG.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/WEBVIEW2_PARITY_BACKLOG.md)）。

### 再訪条件つきでクローズした項目（トリガーが成立したら着手する）

[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md)で「いまは決着だが、ある条件が成立したら見直す」としてクローズした項目を、条件を見落とさないよう本節へ集めたもの。**各項目は「どんなときに読み返すか」を行頭に書く**。詳しい経緯はいずれも同書の該当項を参照する（本書には歴史を書かない方針のため）。

#### 3-30. LAN公開するとき → APIキーバッジの再検証

**サーバーのバインドを`127.0.0.1`（同一PCからのみ到達）から変更してLANへ公開するとき** → APIキーの検証まわりを作り直すか判断する（詳しくは[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-19）。α版ではローカル前提ツールとしてスコープ外クローズしており、バッジの表示機能自体は実装済み・無害のため撤去していない。

#### 3-32. 422の別経路ができたとき → 予約詰まりの再検証

**新しい422経路へ実操作で到達できると分かったとき、または短クリップのクライアント側ゲート（X3）を緩める改修をしたとき** → 422後に仮オブジェクトの予約が詰まらないことを実機で確認する（詳しくは[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-45）。422経路`REFERENCE_REQUIRES_CONTROL_LORA`は単発・chainの両方に存在する（バックエンド[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §49.7）が、フロントエンドは参照動画に制御LoRAの同伴を必須とするゲートを持つため、通常の操作ではこの422に到達しない。**このゲートを迂回して実操作で422を出せると分かったら、予約詰まりの再検証を実施する**。

#### 3-33. アップロードに時間がかかる状況ができたとき → `referenceUploading`ゲートの再検証

**長尺・大容量の参照動画や低速なストレージ・リモートのバックエンドでアップロードに時間がかかる状況が生じたとき、またはアップロード中の進捗表示・スピナーを設ける改修をしたとき** → 「アップロードが確定するまでGenerateを無効化する」ゲートを実機で確認する（詳しくは[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-46）。ローカルでは速すぎて「まだ確定していない一瞬」を捉えられず、検証不能・暫定合格でクローズしている（単体挙動は自動テストで確認済み）。

#### 3-34. 範囲切り出しを配線すると判断したとき → 先に判断材料3点を読む

**タイムラインの範囲選択切り抜き（`cutoutRange`）を製品UIへ配線すると判断したとき** → 着手前に判断材料3点を必ず読む（詳しくは[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-48）。3点とは、①1フレームずつシーン合成でレンダリングするため処理が重い（実装はフレームあたり10秒のタイムアウトを見込む）②焼き込まれるのは対象オブジェクト単独ではなくシーン合成全体の絵③現状は範囲選択が無言で無視され、V2Vには常に元ファイル全体が送られる、である。実装済みコード（`CutoutRangeWorker`・`cutoutRangeAndUpload`）は死にコードとして残置している。**補足**: 判断材料③が指すギャップ（V2Vに常に元ファイル全体が送られる）は、V2Vリボン範囲トリム（元動画ファイルを時間で切る**別方式**。シーン合成レンダリングはしない。[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-59）が有効化済みなので既に塞がっている。`cutoutRange`（合成結果を焼く方式）の配線判断は未着手のままで、本項の再訪トリガーはこれでは成立しない。

#### 3-38. 該当GPUが手に入ったとき → Turing・Hopper・Blackwellの実測

**Turing・Hopper・Blackwellいずれかの世代のGPUを入手したとき** → 実機で生成を通し、動作実績として記録する（詳しくは[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-38）。Ampere（`sm_86`）はサブマシンで実動を確認済みだが、残る3世代はtorch 2.9.1+cu128の同梱カーネル一覧とCUDAのバイナリ互換性からの**理論上の互換**にとどまる。

### 研究課題（つづき）

#### 3-40. チェーン投入系の導線（構築中のチェーンへ素材やテキストを送る）

- **概要**: タイムラインで選んだテキストや画像を、いま組み立てているチェーン（Clip Chain）の**特定のクリップへ**送る導線。「どのクリップに入れるかをどう指定するか」の設計そのものが未着手のため、UIを描く前に方式の検討が要る。
- **統合**: 選択オブジェクトの「素材そのもの」の自動投入（切り出し→レンダリング→アップロードして素材IDをプリフィルへ載せる）も、同じ「素材をチェーンへ流し込む」テーマなので本項へまとめる。現在はファイルピッカーでユーザーが素材を供給する前提で配線している。
- **出典**: [`RIGHTCLICK_REDESIGN_SPEC.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/RIGHTCLICK_REDESIGN_SPEC.md) 第9節、[`DEVELOPMENT_PLAN.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/DEVELOPMENT_PLAN.md)「次セッションへの引き継ぎ」2。
- **状態**: 将来の研究課題（着手時期未定）。

#### 3-41. 入力素材の自動リサイズ（レターボックス）

- **概要**: 生成サイズと縦横比が違う入力素材を、引き伸ばして切り取るのではなく**縦横比を保ったまま余白を足して収める**（レターボックス）方式。`deriveGenerationParams`の継ぎ目に差し込む設計で、β版以降の想定。
- **土台**: 上流のwheelに`resize_and_reflect_pad`（縦横比を保って縮小し、足りない側を鏡像で埋める関数）と`ResizeMode.REFLECT_PAD`が実在するが、**当方のコード（`api/`・`services/`・`engine/`・`gradio_ui/`）からは一度も呼ばれていない**——つまり使われていないフックが既にある状態。
- **出典**: [`DEVELOPMENT_PLAN.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/DEVELOPMENT_PLAN.md)「次セッションへの引き継ぎ」4・第7-5節、`Nz-Videomni/vendor/LTX-2`の`ltx_pipelines/utils/media_io.py`。
- **状態**: 将来の研究課題（着手時期未定）。

#### 3-43. クリップ毎のキャラクター特徴注入（キャラ設計ドリフト対策）

- **概要**: 長いチェーンでクリップが進むほど登場人物の見た目がずれていく（ドリフト）問題への対策。参照シートからキャラクターの同一性を注入する**公式のIngredients IC-LoRA**（`Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients`。ユーザー側の追加学習は不要）が既に存在するため、研究は2段階になる——①既存のIC-LoRA機構でそのまま動くかの検証②現行の「参照はクリップ0のstage-1のみ・1クリップ構成限定」という制約を外すAPI・エンジンの拡張。
- **当面の緩和策（改修不要）**: キャラクターLoRAを学習してチェーン全体へ一様適用すれば、現行の仕組みのままでドリフトを大きく引き戻せる。
- **出典**: [`Nz-Videomni/Docs/PHASE3_CLIP_CONCAT_STATUS.md`](PHASE3_CLIP_CONCAT_STATUS.md)「キャラクター特徴注入研究の一次情報」節（一次情報URL 16本。本テーマの正本は本項）。
- **状態**: 将来の研究課題（着手時期未定）。

#### 3-58. 中間クリップのキーフレーム（クリップ毎`conditioning_images`の解禁）（起票：2026-08-06）

- **概要**: 現在キーフレーム画像はクリップ0だけに許されている（`api/models.py`）。2本目以降にも解禁する。触る場所はstage-1の`i>=1`分岐（現状`conds=[]`）への注入と、stage-2の`clips[0].images`決め打ちの解消の2つ。
- **新規ロジックは1つだけ**: クリップ内のローカルなフレーム位置を、連結後タイムラインのグローバル位置へ変換する処理。**stage-1だけに入れるのは不可**（stage-2はσ0.909から作り直すため、クリップ0とだけ非対称になり2本目以降の構図が緩む）。両段へ対称に注入する。
- **UI方針（オーナー確定）**: クリップカード内に「Keyframes／キーフレーム制御」のon/offボタンを置き、onのときだけスライダー等を表示する（クリップ枠が最大24あるため常時表示は冗長）。
- **状態**: 将来の研究課題（着手時期未定）。
- **出典**: バックエンド[`CHAIN_STAGE2_RESEARCH_NOTES.md`](CHAIN_STAGE2_RESEARCH_NOTES.md) 2・4-(1)・5節。

#### 3-59. Outpainting（動画キャンバス拡張）のチェーン対応（長尺化）（起票：2026-08-09）

- **概要**: 現在のOutpaintingは単発生成と同じ経路なので、扱える尺も単発生成と同じ枠に収まる（仕上げ工程を時間方向に分割しない）。長い動画のキャンバスを広げたいという要望に応えるには、クリップ連結（Chain）と同じ分割の仕組みへ載せる必要がある。
- **何が要るか**: 緑キャンバスの生成とラプラシアンピラミッドのブレンドを、連結の窓ごとに繰り返す形へ拡張すること。ブレンドは窓の外縁とも干渉するため、継ぎ目が二重になる（連結の継ぎ目とブレンドの帯）点の設計が要る。
- **状態**: 将来の研究課題（着手時期未定）。
- **出典**: [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-70、バックエンド[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §54.7（LTX 2.3）・**同§79（LTX 2.5の画角拡張の開通。幾何計算とブレンドの実装、および復号側のVRAMの性質＝本書§4-34）**。

#### 3-75. depth系IC-LoRAを多クリップのチェーンでも使えるようにする（深度前処理のチャンク化）（起票：2026-08-11）

- **概要**: 長尺IC-LoRA（クリップ別の参照動画。実装記録は[`DEVLOG.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/DEVLOG.md) §72）では、canny（輪郭線）・dwpose（姿勢）・deblur（ぼけ除去）の制御アダプタは複数クリップのチェーンでも使えるが、**depth（深度）系だけはクリップ2本以上で422（`LORA_DEPTH_CHAIN_UNSUPPORTED`）になる**。本項はその解禁を検討するもの。クリップ1本のチェーンと単発生成では、現状でもdepth系を使える。
- **何が塞いでいるか**: 深度マップを作るVideo-Depth-Anythingの前処理が**全編を一度にメモリへ載せる設計**（`driver.py`のVideoProcessor分岐が全フレームをリストへ溜める）であること。単発生成やクリップ1本のチェーンなら扱える尺だが、チェーン全体の総フレーム数（上限11544フレーム）では入出力を合わせて数十GBになり、メモリが尽きる。
- **素朴なチャンク化では壊れる論点が2つある**: (1)**時間の整合**——Video-Depth-Anythingは32フレームの窓を10フレーム重ねながら全編を順に舐める設計で、単純に区切ると継ぎ目で深度が飛ぶ。(2)**全編の正規化**——出力は全編を通した最小値・最大値で正規化されるため、区切って別々に正規化すると、明るさ（＝奥行きの尺度）がチャンクごとに食い違う。したがって「重なりを取りながら流し、正規化の統計だけは全編で1回に揃える」といった設計が要る。
- **状態**: 将来の改修項目（着手時期未定・depth系をチェーンで使いたいという要望が出たときが自然な着手条件）。
- **出典**: [`DEVLOG.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/DEVLOG.md) §72、バックエンド[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §57、`Nz-Videomni`の`api/generate_chain.py`（422の新設箇所）・`engine/worker.py`（前処理のフレーム上限）。

#### 3-87. End sourceのUI/UX調整（起票：2026-08-17）

- **概要**: End source（素材（末尾））のUI/UX調整の置き場。**残っているのは次の2件**である。
  1. **末尾の凍結フレームを出力mp4から切り落とす改修**（現行は出力に含める・Retakeと同じ重ね置き規律）。着手する場合は、仮オブジェクト（配置系統E＝末尾合わせ）の位置計算と、「予想出力: ≈n秒（m フレーム）」の表示も同時に控除側へ揃える必要がある（切り落とさない現行仕様では、出力＝クリップ長という今の表示が正しい）。**この条件は複数クリップの逆順Chained・ブリッジモードでも同じ**（末尾の凍結フレームは最終クリップの内側にあるため、着手する場合はその2モードの出力計算も同時に控除側へ揃える必要がある）。
  2. **キーフレーム（`conditioning_images`）×末尾の凍結フレームの重なりを422で拒否する検査**。窓内モードでは凍結フレームがクリップの内側にあるため、キーフレームが静かに上書きされうる。導入時はバックエンドの既存テスト・`api/models.py`のコメント・`VERIFICATION_LOG.md`の「衝突検査は消滅した」記述の3点セットを同時に反転させる必要がある。
- **状態**: 将来の研究課題（着手時期未定）。**以上の内容はLTX 2.3前提。着手時にはLTX 2.5についても検討する。**
- **出典**: [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-84（実用化本体・オーナー裁定によるテーマ完結の記録）・同§3-82、[`DEVLOG.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/DEVLOG.md) §79。

#### 3-92. グローバル音声パス（音声だけ全長一括生成する再設計）（起票：2026-08-18）

- **概要**: 逆順Chainedの複数クリップ構成では、**最終クリップだけ**がEnd sourceと同じ雰囲気の音楽になり、他のクリップは各々異なる音楽・効果音になる（試聴で確認済み。バックエンド[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §65.8）。原因は、逆順の継ぎ目が運ぶ音声のりしろが1潜在（約40ミリ秒）しかなく、楽曲の構造を伝えるには短すぎるためである（映像の錨8フレーム＝運動情報とは時間スケールが違う）。この項目は、音声だけをチェーン全体で一括生成し、クリップ単位の分割を持たないパスへ再設計する構想の置き場である。
- **見込まれる効果**: 音声の一貫性（曲調・テンポ）がクリップ境界をまたいで保たれるようになる可能性がある。ただし映像側のセグメント分割・Stage-1のループ構造には手を入れない前提で、音声だけを独立した生成単位に切り出せるかどうかは未検証。
- **規模感**: **大掛かりな再設計になる見込み**。現行のチェーン機構は映像・音声を同じセグメント単位で扱う設計になっており、音声だけを分離するには`chain_math`の幾何・エンジンのStage-1ループ・凍結スケジュールの複数箇所に影響が及ぶ可能性が高い。着手前に設計調査（音声VAEの時間支持区間・チャンク境界の扱い・既存のA2V経路との関係整理）が要る。
- **状態**: 将来の研究課題（着手時期未定・規模が大きいため設計調査を先に行う）。
- **出典**: バックエンド[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §65.8（試聴結果と考察の正本）、[`CHAIN_STAGE2_RESEARCH_NOTES.md`](CHAIN_STAGE2_RESEARCH_NOTES.md) §11（考察の正本）、[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-84（End source実用化テーマの完結記録）。

#### 3-96. End source付き連結クリップの改善研究（起票：2026-08-19）

- **概要**: End source（素材（末尾））の複数クリップ時の継ぎ目品質を、より使いやすい形で解決する方法を実用の中で探る。旧方式（正順・内部区画）と新方式（逆順Chained）は継ぎ目の弱点がちょうど裏返しで、A/B目視でも構造的な差だと確認できている（理由は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-91と同じ非対称）。
- **判断材料**: 切替UIを作るなら、旧方式は現在APIから到達不能の死蔵状態（`chain_math.py`の`end_source_mode_override`はテスト・切り戻し専用のキーワード専用引数）なので、**APIへ露出する設計が新たに要る**。「本体は正順Chained→最終クリップだけEnd sourceで補間」という実用手順（手作業のままで、通しの実機検証はしていない）も本項の材料である。**なお2026-09-07のブリッジモード開通で、素材（冒頭）を併用する構図はこの手順と実質同じ形が1ジョブで走るようになった**（[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-90）ので、本項に残るのは**素材（末尾）だけを使う場合**の改善である。
- **この実用手順の弱点**: 本体を正順で生成する以上、**素材（末尾）の雰囲気が届くのは最終クリップだけ**で、それより前のクリップは素材と無関係に生成される。素材（末尾）だけを使いたい場面では、全クリップが素材の世界で生成されることを保証する現行の遡り生成のほうが勝る（[`CHAIN_STAGE2_RESEARCH_NOTES.md`](CHAIN_STAGE2_RESEARCH_NOTES.md) §11）。
- **状態**: 将来の研究課題（着手時期未定）。
- **出典**: バックエンド[`CHAIN_STAGE2_RESEARCH_NOTES.md`](CHAIN_STAGE2_RESEARCH_NOTES.md) §11（A/Bの考察と実用手順の正本）、[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §64.2（死蔵引数）・§64.7のR2-7追記（A/Bの記録）、[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-84（本体テーマの完結記録）・同§3-91（非対称の考察の完結記録）。

#### 3-130. LTX 2.3 既定構成にも賢い快適上限線を引けるようにする（境界のモデル化）（起票：2026-08-31）

- **何が開いているか**: 賢い快適上限マーカーの配信テーブルは、**LTX 2.3 の既定構成にだけ意図的に行を持っていない**（この構成ではレガシー表`spill_free_frames`へ落ちる）。素のVAEデコーダを使うため復元（デコード）の段で共有GPUメモリへの退避が起き、快適と非快適の境界がトークン数に対して単調にならないからである。
- **なぜ効くか**: 既定構成の利用者だけが、解像度ごとの粗い表の値しか受け取れない。**非単調性を説明できる式が見つかれば、この構成にも行を1本足せる。**
- **完了条件**: 境界を説明する式が実測で裏づけられ、配信テーブルに`ltx`の既定構成の行を足せること。**「`requires`が空の行」を検証なしに足してはならない**——足せば退避が起きる領域まで快適と表示する。
- **判断材料**: モデル化の候補3つと未解明の点は[`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md) §8が正本（**本書には写さない**）。着手時は、境界のすぐ内側で同一条件でも判定が割れるため**1点あたり最低2回は走らせる較正の作法**が要る（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §84.7）。**LTX 2.3 は半年ほどで既定のベースモデルから外れる見込みなので、優先度は低い。**
- **着手時の前提**: 前ジョブの居残りによる2ジョブ目以降のVRAMのせり上がりは、もう起きない（[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-105）。**ただし広がるのは前処理区間のヘッドルームだけで、本項が扱うデコード段の境界は動かない**ので、既存の較正値をこの件のために読み替える必要はない（正本は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §100）。
- **状態**: 将来の研究課題（着手時期未定）。
- **出典**: バックエンド[`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md) §1.1（行を置かない理由）・§4.8（実測の境界）・§8（モデル化の候補）、[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §84。

#### 3-141. RetakeのIC-LoRA（参照動画つき制御）対応（起票：2026-09-02）

- **概要**: 撮り直し（Retake）でスタイルLoRA（画風・キャラクター系）は使えるようになった（[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-62-02）。**残るのは、参照動画を要する制御系のIC-LoRA（canny／pose／depth等）を撮り直しでも使えるようにすることである。** スタイルLoRA側は「フロントエンドが`loras`を送っていなかっただけ」で済んだが、こちらはそうではない——**現在は3つの層で明示的に禁止されている。**
- **禁止している3箇所（着手時に外す対象。すべて現物を確認済み）**:
  1. **アプリ層（422）**: `api/models.py:1251-1256`（行番号がずれていたら`retake and reference_video_id are mutually exclusive`で引き直すこと）。`GenerateChainRequest`のバリデータが「`retake`と`reference_video_id`は排他（制御アダプタの参照条件付けが、撮り直しの凍結帯と競合する）」として弾く。
  2. **LTX 2.3のエンジン**: `engine/pipeline/chain_pipeline.py:1099-1104`。`assert ic_reference is None or (source is None and retake is None and end_source is None)`。**分岐ではなくassertにしてある**のは、排他が「場合分けして扱うもの」ではなく不変条件だという設計判断からである。
  3. **LTX 2.5のエンジン**: `engine25/chain25.py:1846-1851`。同文のassert。
- **参照動画を窓区間で配る機構は既にある**: `chain_math.py`の`video_segment_windows`（:1681）が、1本の長い参照動画をstage-1セグメントごとの`(開始画素, 長さ)`へ切り分ける。**撮り直しは1クリップなので、これは窓1つ（`[(0, clip_frames[0])]`）へ自然に縮退する**——同関数のdocstringが単発生成と同一になる旨を明記している。**ただし同じdocstringの:1699-1703が「参照はAPI層で`source_video`・retakeと排他なので、`retake_glue_px`を持つレイアウトが参照つきでここへ到達することはない」を契約として書いている。** 排他を外すならこの契約文の見直しが必要で、**書き換えずに通すと、コードと契約文が食い違ったまま残る。**（行番号は`chain_math.py`へ定数を足すたびにずれる。ずれていたら`def video_segment_windows`と`No V2V / retake terms appear here on purpose`で引き直すこと。）
- **参照条件をどこへ挿すか（LTX 2.3）**: 撮り直しの分岐は`conds = []`固定である（`chain_pipeline.py:1494`。「retakeとconditioning_imagesは排他（アプリ層が保証）」というコメント付き）。参照条件の注入はその後段の`conds = conds + pipe._reference_conditioning_from_pixels(...)`（:1741-1746）なので、**改修は「head側の分岐が作った空リストと、後段の注入を繋ぐ」形になる。**
  - **凍結位置のずれは、この経路では既に対策済みである。** 尾側の凍結範囲は**絶対添字**（`chain_math.retake_tail_token_range`）で取っており、`m[:, -k*hw:]`のような負の添字は使っていない（`chain_pipeline.py:455-457`のdocstringが「条件付けトークンを後ろへ足すと負の添字は静かに凍結不足になる」と理由まで書いている）。**つまり条件付けトークンが増えても、凍結する場所は動かない。**
- **LTX 2.5は別実装になる**: `engine25/chain25.py`はセグメントごとの参照条件を`ref_conds`という別のリストで持ち（:2320で全セグメント空リストに初期化）、`conds_v = band_v + (stage1_conds if i == 0 else []) + ref_conds[i]`（:2770）で足し合わせる。**載せる場所は構造としては既にあるが、直前のコメント（:2765-2769）が「撮り直しではこれは`band_v`だけになる。それは偶然ではなく、関数冒頭の2つの排他が保証している」と明記している。** 2.3側とは書き方が違うので、**改修は2エンジンぶんの別作業になる**（共有できるのは`chain_math`の窓計算だけである）。
- **未検証の懸念**: **凍結帯の境界で、制御の効きが不連続になる可能性がある。** 撮り直しの窓は前後ののりしろ（既定で頭25・尾24画素フレーム）が元映像のまま凍結され、その内側だけが作り直される。参照動画による制御は窓全体へ一様に掛かるので、**凍結された帯では制御が効かず、自由部分では効く**という段差が継ぎ目に出ないかは、実機で見るまで分からない。**着手時はここを最初に確かめること。**
- **状態**: 未着手（将来の研究課題）。**急ぎではない**——スタイルLoRAの解禁で、撮り直しにLoRAが使えない状態そのものは解消している。
- **出典**: [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-62-02（スタイルLoRA側の解禁記録と、そこで置いた設計判断）、`api/models.py`・`engine/pipeline/chain_pipeline.py`・`engine25/chain25.py`・`chain_math.py`（上記の各行）、[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §57（長尺IC-LoRAの窓機構）・§55.3（両側凍結の検証）。

#### 3-167. CivitAI 等で配布されている fp8 の safetensors を models ディレクトリに配置するだけで使えるようにする（起票：2026-09-25）

- **状態**: **B-1（LTX 2.3）完結・オーナー目視合格（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §117）。B-2（LTX 2.5）はオーナー目視 7 項目全合格で完結しました（同 §118.10）。B-3（fp8 の快適上限の較正）は計測まで完了しました（同 §119）。結論は [`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md) 第13節です。配信値は変えていません。** 規則の正本は §117.3 と、その改定の §118.3 です。残りはオーナーの判断です。
  1. **第12.4節の最終判断**: Sulphur-2 Q6_K の w46 の VRAM 溢れを配信値へ反映するか。材料は [`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md) 第13.4節です。反映する場合の受け皿は §1-31（重みの種別ごとの行）です。
  2. **配信値への反映**: §1-31「快適上限のマニフェスト一本化」の中で行う（起票済み・2026-09-26）。
  3. **§3-167 全体を CLOSED へ移すか**。
  4. **int8 への拡張は §3-168 へ（2026-09-26）**。
- **概要**: CivitAI などで配布される LTX 2.3／2.5 のファインチューン（多くは fp8 の safetensors）を、変換せずに `models\<系統>\Weights\` へ置くだけで選べるようにする。A1111 SD WebUI の「モデルを置けば使える」体験が目標。対象は fp8 に限る。
- **出典**: [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §117（裁定・設計・判定規則・VRAM の見立て・申し送り）・§118（LTX 2.5）・§119（fp8 の快適上限の較正）、[`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md) 第13節、`sft_fp8_format.py`、`engine/fp8/`、`engine25/`、README「追加の transformer（GGUF／fp8 safetensors）/ LoRA を配置する」。

#### 3-168. ComfyUI 標準の int8 safetensors（int8_tensorwise・asym_w4a8_int8）を models ディレクトリに配置するだけで使えるようにする（起票：2026-09-26）

- **状態**: **コードは C-3（`asym_w4a8_int8`）まで実装が完結し、dev ブランチ `c18d9aa` に積んであります。** 進んだ段は 改称（C-1a）→C-1b（LTX 2.3 の `int8_tensorwise`。エンジン外・エンジン内）→C-1c（敵対的コードレビュー）→C-2（LTX 2.5 の `int8_tensorwise`。実装はコード共通で追加なし）→C-3（`asym_w4a8_int8`。LTX 2.3・2.5 同時。エンジン外・エンジン内）→C-3c（敵対的コードレビュー）です。客観検査のうち **G1（NumPy 突き合わせ）・G2（単体テスト全件）は合格しました**。**実機の門 G3〜G8 は未実施です**（サーバー停止中。監督からの起動は自動モードのため拒否され、オーナーの `run.bat` 起動を待っています）。**続く段は C-4（重みの種別ごとの快適上限の較正。敵対的レビュー反映済みの手順 v2 まで確定・実施は G3〜G8 の後）→オーナー目視→main マージ**です。各段の合否は監督（Claude）が MCP（Model Context Protocol。ツール呼び出しの共通規格）経由の客観検査 G1〜G8 で判断します（内容は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §121.4）。C-4 の手順は同 §121.11 が正本です。
- **概要**: §3-167 で完成させた fp8（重みを 8 ビットの浮動小数点で持つ形式）の safetensors を置くだけで使う仕組みを、同じ考え方のまま int8（重みを 8 ビットの整数で持つ形式）へも広げるテーマです。**到達条件は REDGraft LTX 2.5（CivitAI 3250230）が動くこと**です。調べたところ、この配布物は独自形式ではなく、ComfyUI（画像・動画生成の定番 UI）が標準で読む 2 つの量子化形式——`int8_tensorwise`（層ごとに 1 個の倍率を持つ int8。多くはアダマール回転という前処理＝ConvRot を伴う）と `asym_w4a8_int8`（4 ビットの重みをコードブックで復元する方式）——の混在でした。**復元方式はオーナー裁定により、fp8 と同じく forward（推論の順伝播）のたびに bf16（16 ビット浮動小数点の一種）へ戻す方式とし**、ComfyUI 自身の int8 行列積（活性値側も量子化してから掛ける方式）は模倣しません。**fp8 専用だった経路を「量子化 safetensors 一般」へ一般化し、その上に int8 を載せます**（fp8 と並ぶ別経路を新設するのではなく一本化。ファイル名・クラス名などの改称を伴いますが、ワーカーへのペイロード・API・UI は変わりません）。ComfyUI の受理集合に無い形式（`nvfp4` など）や、印と重みの食い違い、補助テンソルの過不足があるものは断ります。
- **出典**: [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §121（判定規則と設計の正本）、`Nz-GGUF-Converter-LTX23` の `comfy_dequant.py`（ComfyUI 標準 2 形式の NumPy（Python の数値計算ライブラリ）実装。移植元・数値の突き合わせ相手として使い、編集はしません）。

---

## 4. スコープ外（さらに先の将来）

**注記**: 本節の§4-xxは、旧文書に現れる旧§4-xx（将来の研究課題）とは無関係な採番である。読み替えは[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md)冒頭を参照。

§3（将来の研究課題）よりもさらに優先度が低く、当面は着手しないと判断したものの置き場。**捨てずに残すのは、前提が変わったときに読み返すため**である（たとえば非蒸留モデルが動くマシンを入手した、上流のSDKに機能が増えた、など）。各項目には「何が塞いでいるか」を1〜2行で書く。

### 4-1. 非蒸留（dev）モデル向けの生成つまみのGUI露出

- **概要**: CFG（プロンプトへの従い具合）・ネガティブプロンプト・ステップ数・STG・sigmaスケジュール・denoiseループの選択・seedのロック・延長尺（約30秒）・空間アップスケーラのユーザー操作露出——といった、非蒸留（dev）モデルを前提とするつまみ一式。
- **何が塞いでいるか**: 塞ぎ方が2種類ある。
  - **ステップ数**は`services/engines/ltx/adapter.py`（旧パス`services/ltx_runner.py`は再エクスポートのshim）が組み立てるworkerペイロードへ`num_steps`として**配線済み**だが、API層（`api/models.py`のバリデータ）が`pipeline="distilled"`のとき**8ステップ・CFG 1.0を強制**するため値を動かせない。露出するにはこの強制を解く必要がある。
  - **CFG（`guidance_scale`）と`pipeline`**は、そもそもworkerペイロードへ**未配線**のため、GUIへ露出すること自体が禁止（露出すると「操作できるのに効かない」死んだUIになる）。
- **例外的な解禁**: **ネガティブプロンプトのみ、NAG（Normalized Attention Guidance。CFGを使わずにネガティブプロンプトを効かせる手法）とVSF（Value Sign Flip）経由で解禁済みで、これはLTX 2.3・LTX 2.5の両エンジンに及ぶ**（`nag_*`に加えて`neg_method`／`vsf_scale`もCFGを迂回してworkerへ配線されている。`services/engines/ltx/adapter.py`・同`ltx25/adapter.py`）。**未配線のまま残っているのは`guidance_scale`と`pipeline`の2つで、`num_steps`は配線済み・API層の強制だけが塞いでいる**（上記のとおり）。
- **相互参照**: 上記3点の復活条件は**§4-28**（`two_stage_hq`）と同じで、非蒸留モデルを動かせるハイスペックマシンを用意したとき。
- **出典**: [`Nz-Videomni/Docs/A2V_DESIGN.md`](A2V_DESIGN.md) §2.5、[`API_REFERENCE.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/API_REFERENCE.md) §7（NAG）。
- **背景知識（着手時の前提。独自 GGUF を選んだ経緯から）**:
  - **なぜ transformer が GGUF なのか**: 当初は公式 `ltx_pipelines` で bf16 の safetensors をオフロードして動かす計画だった（[`note.md`](note.md)）。実機（VRAM 16GB・Windows）で公式ローダが safetensors の一括読み込み中に native crash し（VRAM 容量の問題ではなく、オフロードでも同じ）、低 VRAM フォーク由来の「GGUF transformer（圧縮のまま VRAM に置き、層ごとに逆量子化）＋ block swap」へ転換した。Q4_K_M はコミュニティの 16GB レシピと一致する実績優先の選択で、Q6 は実績が無いため後回しにした。正本: `engine/VENDOR_NOTICE.md`・[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §1〜§2・§5・[`DESIGN_COMPARISON_and_direction.md`](DESIGN_COMPARISON_and_direction.md)・[`LTX23_REFERENCE.md`](LTX23_REFERENCE.md) §5〜§6・README の GGUF の段落。
  - **クラッシュの真因は後日特定済み**（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §2.9）: ローダが CUDA 初期化後に mmap で開いた safetensors へ `.to(non_blocking=True, copy=False)` を掛けていた実装ミスで、safetensors 形式そのものの問題ではない。ただし bf16 は 16GB に入らず、公式のオフロードはコミット量を悪化させ GGUF とも非互換なため退けた（同 §9.2）。
  - **Windows のコミット（仮想メモリ）制約**（同 §9.1）: 巨大な safetensors をコピーオンライトの mmap で開くと、読む量に関係なくファイルサイズ分のコミットが予約され、巨大なページファイルなしでは落ちる。OS 側にこの予約を止める設定は無い。現行設計はモノリスを開かず、小さな部品ファイルと GGUF（共有読み取り専用の memmap）だけを読むので当たらない。大きな safetensors を直接読む必要が出たら、safetensors 0.8 以降の `backend="pread"`（`.venv-engine-ltx25` の 0.8.0 にはあり、`.venv-engine` の 0.7.0 には無い）か「テンソルを1本ずつ読んで即 GPU へ送る」で収める。
  - **dev モデルの扱い**: dev も同じ 22B 構造なので `Nz-GGUF-Converter-LTX23` で GGUF にでき、読み込み自体は現行の 16GB 機で塞がっていない（前例＝非蒸留の 10Eros を Q4_K_M に変換し読み込みまで確認、Converter の VERIFICATION §6。蒸留 LoRA は `StyleLoRA` から実行時に当てられる。Q6_K への変換も同ツールで可能）。本項の本当のコストは時間で、CFG はステップごとに順伝播が2回になり、ステップ数も 8 ではなく数十になる。2回の順伝播をバッチにすると活性化が倍になり快適上限を割りやすく、逐次にすると時間が伸びる、という設計上の分かれ目がある。「CFG あり・多ステップ」の推論経路は未検証（10Eros は蒸留 LoRA 込みの 8 ステップで確認しただけ）。着手時は、dev チェックポイントのテンソル名・形状が参照 GGUF の型マップと過不足なく一致することの照合から始める。§4-28（`two_stage_hq`＝非量子化の第2段）は bf16 を VRAM に載せる前提なので、本項とは別に塞がったまま。

### 4-2. attention tilingの本番投入

- **概要**: attention（注意機構）の計算を分割してピークVRAMを抑える仕組み。SDPAへのグローバルパッチとして**実装・配線は済んでいるが、既定はOFFのまま**（`config.yaml`の`attention_tile_size`が`null`）で、本番の生成経路では一度も使っていない。
- **何が塞いでいるか**: 現行の16GB運用が二段パイプライン＋block-swapで足りており、ONにする動機と実測がまだ無い。高解像度側へ踏み込むときに、推奨値（256／512／1024／2048）で効果と副作用を測るところから。
- **出典**: [`Nz-Videomni/Docs/SCALEUP_16GB_RESEARCH.md`](SCALEUP_16GB_RESEARCH.md) §2、`Nz-Videomni/config.yaml`。

### 4-3. VRAMの残レバー3件

- **概要**: ①**peak②の低減**——ジョブ全体のVRAM天井はdenoise／transformerロード段（peak②）で決まり、`--te-offload`はここに触れない。②**`layers_on_gpu`が未露出**——Gemmaの層オフロードは`layers_on_gpu=2`のハードコードで、1に下げればencodeピークがさらに下がり3〜4に上げれば速度と交換できるが、`config.yaml`から触れない。③**opt-inの層ストリーミング**——「CPUに正本を置き、GPUには限定サブセットだけ載せて残りを層ごとに流す」方式（ComfyUI-GGUF／HF accelerateと同型）。
- **何が塞いでいるか**: いずれも現行の16GB運用では実害が出ておらず、着手には計測環境（`torch.cuda.max_memory_allocated`＋WDDMのDedicated／Shared＋committed bytes）の用意が前提になる。③は長尺連結でメモリの漸増が実害化したときの再着手先として記録されている。
- **出典**: [`Nz-Videomni/Docs/VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §11.5・§11.6・§15。

### 4-4. VRAM 8GB環境での能力見積り（参考メモ）

- **概要**: VRAM 8GBのGPUで何が動くかのフェルミ推定表（480p約177フレーム／720p約73〜81フレーム／1080p約33フレーム／1440p約9フレーム／4Kは静止画1枚のみ）。実測ではなく16GBの実測枠組みからの外挿で、Windowsのメモリ断片化により**実値はこれより低くなりうる楽観的上限**として扱う前提。
- **何が塞いでいるか**: 8GB対応は現プロジェクトの対象外というオーナー方針。要件化されたときの入口として残す。着手時の有効な調整軸は`block_swap_blocks_on_gpu`（本番既定の8から減らすとベースラインVRAMが下がり上限フレームが増えるが、代償としてPCIeストリーミングが増えて低速化する）。
- **出典**: [`Nz-Videomni/Docs/RESOLUTION_DURATION_CAPABILITY.md`](RESOLUTION_DURATION_CAPABILITY.md) §8.5。

### 4-5. VLM（画像を見る言語モデル）の再導入

- **概要**: 入力画像をモデルに見せてプロンプトを補強する`enhance_i2v`や、フレームを見たうえでの隙間埋め提案。
- **何が塞いでいるか**: テキストエンコーダのGemmaを**text-only化してVRAMを22.7GB回収した**現行構成と正面から衝突する。巻き戻す判断が必要なため計画外で、要件化されたときに別途判断する。
- **出典**: [`Nz-Videomni/Docs/PHASE3_NEXT_WORK_SURVEY.md`](PHASE3_NEXT_WORK_SURVEY.md)。

### 4-6. バックエンド同梱Gradio UIの残4件

- **概要（残る4件）**: ①`GET /jobs/{id}/metadata`エンドポイント（GUIでVRAMピークやバックエンド種別を表示する用途。「新規エンドポイントを足さない」方針で見送り）②`gr.BrowserState`による言語／テーマの永続化（固定secretと実機検証が必要）③`gr.render`によるキーフレーム／クリップ行の動的追加（現状は固定スロット）④Settingsのデフォルトnegative promptの設定欄（NAG経由ならnegative promptは生きるが、**既定negative promptをどこに持たせるかの設計が未着手**のため見送り）。
- **何が塞いでいるか**: いずれもバックエンド同梱GUIの利便性向上であり、製品の入口はフロントエンド側という位置づけのため優先度が低い。
- **①の判断材料**: 既定の設定のとき（`output.save_metadata_json`が真で、要求の`embed_mp4_metadata`がon）、完了したジョブの記録（`metadata.json`と同じJSON。`backend`と、`vram_optimization`ブロックの中の`peak_vram_mb`を含む）は、Gradioの「MP4 Info（mp4 情報）」タブで`output.mp4`を開けば読める（読み出し口は`POST /utils/mp4-info`。書き込みの規則はバックエンド[`Videomni_Backend_Specification.md`](../Videomni_Backend_Specification.md) §6.6、画面は同§12.2）。①に着手する前に、この画面で用が足りるかを確かめること。なお①を見送った理由の「新規エンドポイントを足さない」方針は、`POST /utils/mp4-info`を足したので、それだけでは理由として成り立たなくなっている。
- **既知の差分**: in-outpaintingアダプタをIC-LoRA制御アダプタのドロップダウンから除外する挙動（フロントエンド[`DEVLOG.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/DEVLOG.md) §73）はWebUIのみに効き、Gradioには効かない。`gradio_ui/adapters.py`が`/config`のキーを直接列挙して選択肢を作る実装のため。α版の割り切りとしてWebUI側のみで対応する方針である。
- **出典**: [`Nz-Videomni/Docs/VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §23.5、フロントエンド[`DEVLOG.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/DEVLOG.md) §73。

### 4-7. V2V／チェーンの音声まわりの残件3点

- **概要**: ①stage-2の音声タイル継ぎ目（チェーン由来のbacklogと同族でV2V固有ではない）②fpsリサンプルが**全体変換**になっている（末尾のcontext区間だけの部分変換に最適化できる。単一ユーザーでは実害小。可変フレームレート素材は未ストレステスト）③ハンドル方式のクロスフェード窓に残る**浅い一時的な凹み**（深さ0.42。源とハンドルが同一音楽の別レンダのため位相干渉が部分的に残る）。
- **何が塞いでいるか**: ①②③とも、オーナーの試聴で実用上受容済み。③はノイズフロア整合などの追加手段が候補として挙がっている段階。
- **①②③はLTX 2.3の試聴・実測だけである**: **LTX 2.5では同じ3点を計測していない。** 着手するときは2.5でも測り直すこと。**題材が同じ「連結の継ぎ目の音」なので、[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-117（逆順Chainedの音声の継ぎ目。原因はモデル〔音声VAE・ボコーダ〕の性質と確認済み）の知見を踏まえ、1つの試聴セッションにまとめるのが合理的である。**
- **出典**: [`Nz-Videomni/Docs/VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §24.5・§24.7、[`Nz-Videomni/Docs/V2V_AUDIO_JOIN_RESEARCH.md`](V2V_AUDIO_JOIN_RESEARCH.md) §4。

### 4-8. 未対応のIC-LoRAアダプタ（Motion-Track）＋`conditioning_attention_mask`の露出

**(B) Motion-Track（動きの軌跡追従）＝ 未対応**（枝番(A)は欠番）

- 未対応のIC-LoRA制御アダプタとして残っているのは**Motion-Track だけ**である（Depth・Deblurは[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-63、In-Outpaintingは同§3-70で対応済み）。
- **何が塞いでいるか**: **需要が確認できていない**こと。技術的に塞がれているわけではない——「制御信号を作る前処理器を新設して登録する」経路はDepthで実証済みなので、必要になれば同じ型で足せる。なお旧19b世代のアダプタは、適用してもエラーは出ないが**視覚効果がゼロ**と報告されており流用できない。

**(C) `conditioning_attention_mask`のAPI露出 ＝ 未着手**

- **概要**: 参照条件の効かせ方を画面の場所ごとに絞るマスク。IC-LoRA Phase Cのスコープ外項目として唯一未着手のまま残っている。
- **何が塞いでいるか**: **`conditioning_attention_mask`のAPI露出の設計が未着手であること**（マスクの受け渡し契約そのものは[`INPAINTING_DESIGN.md`](INPAINTING_DESIGN.md) §6で**確定・実装済み**である）。**契約の正本は同§6**（ピクセル精度1本・1chグレースケール動画・uploadsレール・受信側の二値化）で、**本項はその応用先の1つである**——契約を本項で作り直さないこと。受け口の入口設計は§4-25と同じ話である。
- **参考（2026-09-15）**: 公式のComfyUIノード`LTXAddVideoICLoRAGuideAdvanced`（Lightricks/ComfyUI-LTXVideoの`iclora.py`）は、参照の効き方を領域ごとに変える`attention_mask`（F×H×W、0〜1）と`attention_strength`を持つ。本項の露出設計を始めるときは、この公式実装が先例になる。
- **出典**: [`Nz-Videomni/Docs/IC_LORA_PHASE_C_STATUS.md`](IC_LORA_PHASE_C_STATUS.md)「スコープ外」（着手時の入口）、[`Nz-Videomni/Docs/IC_LORA_PHASE_C_RESEARCH.md`](IC_LORA_PHASE_C_RESEARCH.md)。

### 4-9. A2V API層の残5点

- **概要**: v1のスコープ外として明示された5点——①複数クリップのA2V（音声の窓割り）②A2VとV2Vの同時指定（現在は422で排他）③vocoderで生成した音声の返却（現在は原波形をそのまま多重化する一択）④`audio_start_time`／`audio_max_duration`の露出（音声のトリミング）⑤`modality_scale`の露出。
- **何が塞いでいるか**: いずれもv1の設計時点で意図的に落としたもので、需要が確認できていない。凍結APIへの加算的拡張として後から足せる形は保たれている。
- **出典**: [`Nz-Videomni/Docs/A2V_DESIGN.md`](A2V_DESIGN.md) §2.5。

### 4-10. 120クリップ級の長尺チェーン構想

- **概要**: クリップ連結の上限をさらに大きく（120クリップ級＝理論上8時間）伸ばす構想。**チェーンは不可分の単一ジョブである**（途中で失敗すると全部やり直し）というリスクがあるため、固定枠を伸ばす形は採らず、上限は24としている。
- **何が塞いでいるか**: やるとしても「固定枠の拡張」という現行方式ではなく、バッチA2Vと同じ**表＋状態管理のUI＋区間分割生成して繋ぐ再開可能な仕組み**として、別の設計課題で扱うべきものと位置づけられている。
- **出典**: [`Nz-Videomni/Docs/CHAIN_UI_EXPANSION_WORKORDER.md`](CHAIN_UI_EXPANSION_WORKORDER.md)。

### 4-11. 参照動画まわりの早期バリデーション（1件）

- **概要**: Chain画面で制御系LoRAのタグを手打ちするとGenerateは正しくブロックされるが、そこへ至るまでにソース動画のアップロードは走り終えており**無駄なアップロードになる**。
- **何が塞いでいるか**: 「動かない」わけではなく、失敗の伝え方が遅いという性質の問題のため優先度が低い。
- **現行ゲートでの再確認が要る**: 起票時の`hasControlLoraTag`（チェーンで制御系LoRAを一律に禁じるハードブロック）は**撤去済み**で、いまは`controlLoraNeedsReference`（制御系LoRAに参照動画の同伴を求める交差検査）に置き換わっている。**この形でもなお無駄なアップロードが起きるのかを、まず実地で確かめること。**
- **出典**: `webui/src/modes/chained/useChainForm.ts`（`controlLoraNeedsReference`）・`webui/src/lora/controlLoras.ts`、`Nz-Videomni`の`api/generate_chain.py`。

### 4-13. 時間アップスケーラ（temporal upscaler x2）

- **概要**: LTX 2.3には空間アップスケーラ（x2）と並んで**時間方向のアップスケーラ（x2）**が公式に存在するが、本プロジェクトは空間側だけを導入しており、時間側は取得も配線もしていない。フレーム数を後段で2倍に増やす（＝滑らかにする）用途にあたる。
- **何が塞いでいるか**: 需要が未確認で、モデルの追加取得（ダウンロード容量の増加）とVRAM影響の実測が要る。
- **出典**: [`Nz-Videomni/Docs/LTX23_REFERENCE.md`](LTX23_REFERENCE.md)（モデル変種一覧）、`Nz-Videomni`の`services/base_models.py`とベースモデル記述子の`assets`ブロック（空間アップサンプラはここにあり、時間側は無い。旧`config.yaml`の`spatial_upsampler_path`は`config.py`の`DEPRECATED_MODEL_KEYS`へ移行済み）。

### 4-16. Join（V2V結合）の既知の縮退3ケース

- **概要**: joined動画を🎞で挿入するとき、位置指定つき挿入と仮オブジェクトの自動削除が効かず、カーソル位置への素挿入へ穏当に縮退する場合が3つある——①Chain画面での手動v2v②WebUIリロード後（jobId→元動画位置の揮発マップが消える）③マウント時に`job.joined`から復元した連結済み表示（トリム済み秒数が不明で位置計算できない）。
- **何が塞いでいるか**: 実装時に**でっち上げの位置を使うより素挿入のほうが安全**と判断して受容した設計。直すには位置マップの永続化が要る。
- **出典**: [`JOIN_FEATURE_RESEARCH.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/JOIN_FEATURE_RESEARCH.md) 第4部、[`DEVLOG.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/DEVLOG.md) §44.2・§44.4。

### 4-18. 図形オブジェクトのエイリアス実書式

- **概要**: AviUtl2の図形オブジェクトを`create_object_from_alias`で作るための正確な書式が未取得（同梱プリセットが空で、実機ダンプが要る）。effect名が`図形`で項目が`図形の種類`・`色`・`ライン幅`等であることまでは判明している。
- **何が塞いでいるか**: **仮オブジェクトはテキストで作る方針**（テキストの書式は確認済み）なので、図形の書式は現状どこからも必要とされていない。図形を使う機能を作ると決めたときに実機ダンプする。
- **出典**: [`TIMELINE_ALPHA_REQUIREMENTS.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/TIMELINE_ALPHA_REQUIREMENTS.md)（未確定事項・図形／テキスト生成）。

### 4-19. SDK可否表をv2.0.54基準で取り直す

- **概要**: [`SDK_REFERENCE.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/SDK_REFERENCE.md) §10の「本体バージョン整合表」は**beta52基準**で書かれた、API追加日からの状況証拠ベースの可否判定である。実機ランタイムはv2.0.54ポータブルなので、v2.0.54での動作実績は取り直しが要る。
- **何が塞いでいるか**: 右クリック再設計で実際に使うAPIは実機で一次確認済み（同書の「実機確定知見」節）で、実務上の困りごとが出ていない。表全体の取り直しは網羅的な実機作業になる。
- **出典**: [`SDK_REFERENCE.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/SDK_REFERENCE.md) 冒頭注記・§10。

### 4-20. オブジェクトメニューの種別フィルタ

- **概要**: 右クリックのオブジェクトメニューは、png・wavなど生成に使えない種別のオブジェクトでも同じ項目が全部出る。
- **何が塞いでいるか**: **SDKの`register_object_menu`にオブジェクト種別を絞る引数が存在しない**ため（`plugin2.h`で確認済み）、仕様どおりの挙動である。AviUtl2本体側にフィルタ引数が追加されたときに再訪する。
- **出典**: [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-20、`AviUtl_ExEdit2 SDK`の`plugin2.h`。

### 4-21. 音声抽出のsolo分離（対象レイヤー単独化）

- **再訪条件**: ユーザーから「mix経路では不要な音が混ざる」という具体的な不満・要望が出たとき。
- **概要**: タイムライン右クリックの「動画の音声をa2v（音声から動画生成）へ送る」で、対象レイヤー以外を自動無効化して単独音声を抽出するsolo分離モード。
- **何が塞いでいるか**: 実機検証済みのmix経路（タイムライン全体の音をそのまま録る）で実用上十分であり、不要な音はユーザーがレイヤーを手動無効化すれば代替できる。solo分離は実機未検証・コミュニティ参考実装なしでリスクが高く、需要も未確認。
- **出典**: [`DEVLOG.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/DEVLOG.md) §33.6。

### 4-22. SageAttentionを既定にするかどうかの再検討（起票：2026-08-01）

- **再訪条件**: SageAttention（生成の高速化＝Acceleration機能。[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-58）が、フィールド（オーナー以外の実際の利用環境）で十分な期間・十分な件数にわたって問題なく動いた実績が溜まったとき。あるいは、利用者から「毎回わざわざ切り替えるのが面倒」という声が出たとき。
- **概要**: 現在、attention（注意機構）の実装の既定は`sdpa`（PyTorch標準）で、`sage`は利用者が設定画面で選んだときだけ有効になる。これを`sage`既定へ切り替えるかどうかを再検討する。
- **本項の対象はLTX 2.3・LTX 2.5の両方である**: **LTX 2.5でもSageAttentionは開通しているが、既定は2.3とまったく同じ理由（アップデートで利用者の生成結果を黙って変えない）で`sdpa`のままである。したがって既定を反転するかどうかの議論は、2.3と2.5をまとめて本項で扱う**（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §77）。**判断材料が1つ増えた**——LTX 2.5では効き目が形状に強く依存し、512×320級では効かないかむしろ遅くなる実測があるので、「既定onにすれば誰でも速くなる」とは言えない。
- **いま`sdpa`既定にしている理由（オーナー確定）**:
  1. **アップデートで生成結果を黙って変えない**。`sage`は数値精度が異なるため、同じシードを指定しても生成結果の細部が変わる（構図は同じで、質感やノイズの出方が変わる。実測でPSNR 27〜28dB程度の差）。既定を差し替えると、利用者が「昨日と同じ設定なのに絵が違う」という説明のつかない体験をすることになる。
  2. **`sdpa`は常に正しい参照実装**であり、比較の基準として動かさない価値がある。
  3. **切替のコストが小さい**。UIの1クリックで済み、選択は保存される。
- **着手の前提は満たされた**: 先に片づけるべきだった「`vae_mode`にエンジン軸が無い」件は解消済みで（[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-139）、**保存済み設定がエンジンを跨いだときの振る舞いが決まったので、既定を反転しても説明のつかない422を踏ませることはない。** 残るのは上記の再訪条件（フィールドでの実績が溜まること）だけである。
- **切り替えるとしたら何を決めるか**: 既存プロジェクトの再現性をどう扱うか（既定変更の告知方法、あるいは「以前と同じ結果が欲しいなら`sdpa`」の案内の出し方）。`sageattention`未導入環境での降格挙動はすでに実装済みなので、そちらは追加作業にならない。
- **出典**: バックエンド[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §43.1・§43.9、[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-58、[`DEVLOG.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/DEVLOG.md) §55。

### 4-23. OutputsとUploadsの保存領域一本化（「同じジョブのものは同じディレクトリへ」案）（起票：2026-08-01）

- **概要**: バックエンドの保存領域は現在、成果物の`outputs/{job_id}/`（output.mp4＋metadata.json）と、入力素材の`uploads/`（画像/動画/音声の3系統・UUIDディレクトリ）に分かれている。これを「同じジョブで使ったものは同じディレクトリに保存する」形へ一本化すれば、ユーザーが後からファイル整理しやすいのではないか、という案。
- **スコープ外とした理由**: ①アップロードはジョブ誕生前に起きるためジョブディレクトリへ直接置けず、ステージング＋移動の機構が要る。②1つのアップロードを複数ジョブが使い回す実態がある（バッチの画像/参照動画キャッシュ・画面再マウント時のスロット引き継ぎ）ため、ジョブごとのコピーは大きな動画の重複を生む。③参考にしたSD WebUIの思想はむしろ「outputsだけが永続で、入力・中間物は保存しない」であり、一本化はその方向とも一致しない。
- **代わりの現方針**: 「**Outputsは宝物、Uploadsは事実上の一時ファイル置き場**」という区別が設計原則である（正本: [`Nz-Videomni/Docs/STORAGE_POLICY.md`](STORAGE_POLICY.md)）。将来の軽い改善候補として「起動時に古いuploadsを自動掃除（年齢ベースGC）」がある（同書に記載）。
- **再訪条件**: 上記の現方針で実運用上の不都合が出たとき。
- **出典**: [`Nz-Videomni/Docs/STORAGE_POLICY.md`](STORAGE_POLICY.md)。

### 4-24. Style LoRAの音声強度制御を4軸へ拡張する「Style LoRA Advanced mode」（起票：2026-08-02）

- **概要**: 音声強度制御（`audio_strength`）は映像軸・音声軸の**2軸**で実装済みである。LTX 2.3の実構造ではクロス注意の方向別に軸を分けると4軸まで取りうるが、「どちらのストリームに書き込むか」で2軸へ畳み込んで実装した。
- **スコープ外とした理由**: 4軸のほうが厳密だが、2軸のほうがユーザーが理解しやすく使いやすいと判断した（UX優先）。クロス注意の方向別に個別制御したいという需要は未確認。
- **再訪条件**: リップシンクにかかわるStyle LoRA需要を発見したとき。既定の2軸UIは維持したまま「Style LoRA Advanced mode」を新設して4軸指定を追加する形を想定。
- **出典**: バックエンド[`LORA_AUDIO_STRENGTH_WORKORDER.md`](LORA_AUDIO_STRENGTH_WORKORDER.md) §1・§3。

### 4-25. AviUtl2側で作成したマスク動画のバックエンド接続（起票：2026-08-03）

- **概要**: AviUtl2には、他者が作成したプラグインとして、物体認識や追尾マスク（動く被写体を自動で追いかけて切り抜くマスク）を作成できるものが既に存在する。これらを使ってAviUtl2側でマスク動画（白黒の適用範囲指定動画）を作り、それをNz-Videomniのバックエンドへ渡す汎用の受け口を作るというアイディア。
- **本項はInpaintingのマスク契約の応用先である**: **受け渡しの形（ピクセル精度1本・1chグレースケール動画・uploadsレール・受信側の二値化）は[`INPAINTING_DESIGN.md`](INPAINTING_DESIGN.md) §6が正本**で、本項が扱うのは**その入口をAviUtl2側の既存プラグインへ繋ぐこと**である。ほかの応用先は§4-8(C)（`conditioning_attention_mask`の露出）である。
- **入力源は他者のプラグインだけではない（2026-09-11以降）**: 本プロジェクト自身の物体追尾（[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-54-02）が動かした部分フィルタも、契約から見れば同じ「AviUtl2側のオブジェクト」である（[`OBJECT_TRACKING_DESIGN.md`](OBJECT_TRACKING_DESIGN.md) §8・§13）。したがって「他者プラグインの調査」は本項の唯一の入口ではなく、自前のオブジェクトから先に着手することもできる。
- **何が塞いでいるか**: **利用可能な物体認識・追尾マスクプラグインの調査が未着手**で、それらが吐き出すマスク動画の形式が[`INPAINTING_DESIGN.md`](INPAINTING_DESIGN.md) §6の契約に合うかどうかも分かっていない。着手はこの調査から始める。
- **出典**: オーナー発案。

### 4-26. Control系IC-LoRAの複数同時適用

- **概要**: canny-control・pose-control等、複数の前処理種別を同じ参照動画に併用したい需要への対応。単一制御に限定しているのは当方の3箇所のみ——①`reference_video_id`が単数（`api/models.py`）②前処理種が2種以上だと400 `LORA_PREPROCESS_CONFLICT`（`api/generate.py`）③エンジン側の`_ic_reference`が単一タプル（`engine/pipeline/fast_video_pipeline.py`）。
- **判断材料**: 複数信号を1本の動画へ重ね描きする方式は公式・コミュニティとも実例ゼロで、公式チュートリアルは複数モード同時実行をVRAM問題として明示的に非推奨としている。実現するなら公式ComfyUIの`LTXVAddGuideMulti`と同型の「制御動画を複数本、独立に条件付けする」方式が本筋で、上記3箇所の拡張が正面から必要になる（調査は2026-08-03）。
- **何が塞いでいるか**: 公式が実運用で非推奨としており、需要の証拠も薄いため。
- **出典**: [`DEVLOG.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/DEVLOG.md) §15.2、[`Nz-Videomni/Docs/IC_LORA_PHASE_C_STATUS.md`](IC_LORA_PHASE_C_STATUS.md)、`Nz-Videomni`の`api/models.py`・`api/generate.py`・`engine/pipeline/fast_video_pipeline.py`、[LTX公式 IC-LoRA Adapters](https://docs.ltx.io/open-source-model/integration-tools/ic-lo-ra-adapters)、[ltxworkflow.com IC-LoRAガイド](https://ltxworkflow.com/resources/tutorials/ic-lora-ltx-2-3-complete-guide)、[ComfyUI-LTXVideo issue #479](https://github.com/Lightricks/ComfyUI-LTXVideo/issues/479)（いずれも2026-08-03調査）。

### 4-28. `two_stage_hq`（非量子化モデル用の高品質パイプラインモード）（起票：2026-07-15）

- **概要**: 品質モード`two_stage_hq`は、非量子化モデル向けの高品質パイプライン種別として、API列挙型（`pipeline: Literal["distilled","two_stage_hq"]`）とGradio UIのラジオに**モック（枠）だけ**用意されている機能。現行唯一の実働パイプライン`distilled`も内部的には二段（Stage1半解像度→x2アップスケール→Stage2）だが、**別物なので混同しないこと**。なおLTX 2.5では既定以外の`pipeline`は422で断る（[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-102）。
- **何が塞いでいるか**: **非量子化モデルを動かせるハイスペックマシンをオーナーが所有していないこと**（**§4-1**と同根）。計算コストはステップ増（8→30〜50）とCFGの2回forwardで**約7〜12倍**になり、活性値も増えるためVRAM 16GBでの実測が必須。
- **着手するときの入口**: ①未配線箇所は`services/engines/ltx/adapter.py`（旧パス`services/ltx_runner.py`は再エクスポートのshim）が組み立てるworkerペイロード——`op: "generate"`の辞書に`pipeline`と`guidance_scale`が無く、常にdistilled経路になる（ステップ数の扱いは§4-1）。②入口となる重みは非蒸留×量子化のGGUFが実在する（`unsloth/LTX-2.3-GGUF`の`ltx-2.3-22b-dev-Q4_K_M.gguf`、約14.3GB）。
- **出典**: [`WEBVIEW2_PARITY_BACKLOG.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/WEBVIEW2_PARITY_BACKLOG.md) N12、`Nz-Videomni`の`api/models.py`（pipeline列挙型）・`gradio_ui/i18n.py`（「バックエンド未対応」ラベル）・`services/engines/ltx/adapter.py`（generateペイロード）、[`Nz-Videomni/Docs/FEATURE_RESEARCH_2026-07-04.md`](FEATURE_RESEARCH_2026-07-04.md) D節。

### 4-30. stage-1の快適予算（参照動画つき）の配信化（起票：2026-08-12）

- **概要**: 参照動画つきstage-1の快適予算（`chain_math.py`の`CHAIN_STAGE1_COMFORT_TOKEN_BUDGET`＝25,000トークン）だけが、**サーバーから配信されずフロントエンドのミラー定数のまま**である。stage-2側の予算はエンジン系統ごとの配信テーブル`limits.comfort_budgets`（正本は[`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md) §1.1・§6④）で配信済みで、この25,000はそのテーブルの対象外である。
- **何が塞いでいるか**: 何も塞いでいない。必要が生じた時点（VRAM容量の異なる機体で25,000線を調整したくなった時点）で着手すればよい。25,000は実機ゲートで実測の急変点と一致することが確認済みの暫定値である（バックエンド[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §57のG4）。
- **着手時の最初の分岐**: 「`LimitsConfig`へ独立の鍵を1つ足す」か「`comfort_budgets`のテーブルへ行として持たせる」かを選ぶこと。後者を選ぶなら、この予算が**参照動画つきstage-1という別軸**である点を表側でどう表すかの設計が要る。
- **着手時に必ず同時に決めること**: 参照エンコードのタイル化のしきい値`REFERENCE_ENCODE_TILE_TOKEN_BUDGET`（5,000＝25,000÷5）は、**この線に自動追随しない**（コードのコメントにその旨が明記してある）。25,000を動かすなら、しきい値をどうするかも同じ改修の中で決めること——**放置すると、画面の警告が出る位置と実際にタイル化へ切り替わる位置がずれる。** 正本は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §101.2。
- **関連**: [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-133（参照動画つき単発生成の実測。**単発側の線は動かないという結論で決着済み**であり、この25,000とは別軸のままである）・同書§3-76（参照動画VAEエンコードのタイル化。**しきい値をこの25,000から機械的に割り出した固定値で導入しており、線そのものは動かしていない**——再較正は本書§4-39の担当である）。
- **出典**: [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-85（Chained快適上限マーカーの実装記録）、`Nz-Videomni/chain_math.py`、`webui/src/shell/tokenBudget.ts`、[`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md) §6。

### 4-31. 失敗ジョブの`metadata.json`が必要になったとき → 失敗経路でも書き出す（起票：2026-08-03）

**失敗したジョブの`metadata.json`（そのジョブが実際にどんな要求で・どんな高速化設定で走ったかの記録）が必要になったとき** → 失敗経路にも簡易な書き出しを足す。

- **現状**: `metadata.json`の書き出しは成功時のみで、`services/pipeline_manager.py`の失敗経路では書いていない。したがって失敗したジョブの高速化設定・リクエストの実値はディスクに残らず、**サーバーを再起動すると失われる**（稼働中のサーバーのAPIから回収できるうちしか原因調査ができない）。
- **何が塞いでいるか**: 何も塞いでいない。**「成功したジョブの記録だけ残ればよい」という現状の仕様でオーナーが不便を感じていない**ため、優先度を下げて本節へ置いた。必要が生じた時点で着手すればよい。
- **着手時の形**: 失敗時にも、要求内容と失敗理由だけを記録した簡易な`metadata.json`を書き出す。成功時の書式と同じ場所（`outputs/{job_id}/`）へ置くか、失敗専用の名前にするかは着手時に決める。
- **出典**: `Nz-Videomni/services/pipeline_manager.py`（成功経路だけが書き出していることの現物）。

### 4-32. 1.86Mpx超の解像度におけるSingle長尺クリップの実現（起票：2026-07-27）

**1.86Mpx（Chainedのstage-2固定窓設計の解像度上限）を超える解像度で、長尺クリップの需要が実際に生じたとき** → 単発生成（`/generate`）へチャンク化アップサンプルとstage-2のタイル化を移植できるかを検証する。

- **何が塞いでいるか＝前提の変更**: LTX 2.5でChainedの窓22が1088pでも快適上限に収まり、ドリフトもほぼ不可視だと実証されたため、**「高解像度×長尺」という需要の大半はChainedが回収した**。SingleとChainedの役割分担（Single＝解像度自由・尺制約／Chained＝解像度制約・尺自由）を守るというオーナーの戦略判断による。**移植してもSingleはAPI上限481フレームを超えられない**という天井も変わらない。
- **再訪時の入口（技術的な中身の要約）**: 単発生成の快適上限を縛っているのは「一括アップサンプル」と「フル解像度の一括仕上げデノイズ」の2工程で、どちらもChained側には対策（チャンク化アップサンプル／stage-2の22latent固定窓によるタイル化）が実装・実機検証済みで存在する。**この2つをセットで移植する**のが本項の中身で、片方だけでは上限は伸びない。起点となる現行値は配信テーブルとレガシー表（**数値は本書へ写さない**。正本は[`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md)）。なおIC-LoRA参照動画のVAEエンコードは、本テーマの最小スライスとしてタイル化を先行実施済みである（倍率1のアダプタは常時タイル化＝正本は[`ICLORA_DEPTH_DEBLUR_WORKORDER.md`](ICLORA_DEPTH_DEBLUR_WORKORDER.md)、倍率2の制御アダプタは参照の潜在トークン数がしきい値を超えたときに自動でタイル化＝正本は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §101）。
- **移植後の推定（フェルミ推定・実測前の目安）**: 960×576は約481／1280×768は481（いずれもAPI上限に到達）／1920×1088は約300（最も実測が必要）／2560×1472は約100〜150（下振れリスク大）。最小確認パスは、1080pを現行値から241→361→481と昇順に振るところから始めること。**この推定はLTX 2.3のものであり、LTX 2.5へ援用するかは着手時に別途判断する。**
- **出典**: [`Nz-Videomni/Docs/PHASE3_CLIP_CONCAT_STATUS.md`](PHASE3_CLIP_CONCAT_STATUS.md)「実運用で得た経験則」節（本テーマの正本は本項）、[`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md)、[`CHAIN_STAGE2_RESEARCH_NOTES.md`](CHAIN_STAGE2_RESEARCH_NOTES.md) 6〜7節（SingleとChainedの棲み分けの正本）。

### 4-33. 拡散デコーダ版の映像VAE（DiffVAE）を採用するかどうか（起票：2026-08-22）

**24GB以上のGPUへの移行／natten（近傍注意を計算する専用ライブラリ）のWindowsでの実用化／公式のメモリ見積りのTriton経路向け再較正——のいずれかが成立したとき、または省メモリ設定（RoPEのbf16化・注意チャンク数の増加・`feat_s4`のCPU退避）と自前の較正で720p級を狙う投資判断がついたとき** → LTX 2.5の映像VAEを、既定の畳み込みデコーダ版（conv）から拡散デコーダ版（DiffVAE）へ切り替えるかどうかを判断する。

- **概要**: LTX 2.5 の映像VAE（映像の圧縮展開器）には**畳み込みデコーダ版（conv）**と**拡散デコーダ版（DiffVAE）**の2種類があり、現在の既定は前者である（[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-98）。ファイル自体は `models/LTX25/VAE/diffvae/` へ退避してあり、カテゴリの走査は非再帰なので自動認識の対象にならない（置いてあるが選べない状態）。
- **何が塞いでいるか**: **VRAM 16GBの実機で、実用域の解像度・尺が成立しないこと。** 実用域にあたる 1280×768/121f と 1920×1088/121f は、公式推奨器が返す最小級のタイルを強制してもメモリ不足で失敗する（さらに小さいタイルは推奨器が算出を拒否する）。実測8条件のうち復号が通ったのは 768×512/41f と 1280×768/49f の2つだけで、後者は確保ピークが物理VRAM 16GBを超えている。**数値の正本は一次記録 `outputs/diffvae-research-2026-09-01/RESULTS.md` §2**（**このディレクトリは git 管理外である**）。
- **画質は比較していない**: 2026-09-01の調査はランダム潜在での動作可否・所要時間・VRAM消費・決定性の機序に限っており、**画質の比較実験は行っていない**（オーナー裁定により実施しない）。
- **決定性は「壊れる」のではなく、条件つきで保てる**: 実機重みは `default_num_inference_steps=1`・`model_output_type="x0"` で反復せず、拡散ブロックはデコーダ重みの約2%である。雑音は `torch.Generator` で固定できる。環境によって出力が変わる原因は、**`AUTO_TILING` のタイル表が実行時の空きVRAMから決まること**——1ステップ時の雑音はタイルごとに独立に引かれるため、タイル表が乱数の消費順序を決める。**明示 `TileSizeConfig` と固定 `generator` を組み合わせれば環境非依存になる。** 正本は同 RESULTS.md §5・§6。
- **natten の現状**: `ltx_core` 1.2.0 の `natten` extra は environment marker が Linux（x86_64／aarch64）限定で、Windows では依存そのものが要求されない。NATTEN 公式も Windows ビルドを experimental と明記し、公式 wheel は Linux 向けのみである。不在時は Triton の `na3d` 経路へフォールバックして動作するが、**公式のメモリ見積り係数は natten 経路を前提にしている**ため、Triton 経路では実測が見積りを大きく上回る（同 RESULTS.md §4・§6.3）。
- **連結生成の実装が作った前提**: `engine25/chain25.py`は、素材の潜在化と復号で**同じタイル設定を使い回す**ために、その解決を**まだモデルを1つも載せていない時点**へ前倒ししてある。畳み込みデコーダ版では自動タイル化が縦横比だけを見る分岐を通り空きVRAM量を読まないので結果は決定的だが、**DiffVAEは空きVRAMを見る分岐へ入る**ため、採用するならこの前倒しの位置を見直す必要がある（空きが最も楽観的に見える位置で読むことになるため）。詳細は[`CHAIN_STAGE2_RESEARCH_NOTES.md`](CHAIN_STAGE2_RESEARCH_NOTES.md) 12節末尾。
- **出典**: 一次記録 `outputs/diffvae-research-2026-09-01/RESULTS.md`（実測8条件・実装コード読解・公式／コミュニティ情報の正本。git 管理外。複写: [`Docs/Outputs-archive/diffvae-research-2026-09-01/RESULTS.md`](Outputs-archive/diffvae-research-2026-09-01/RESULTS.md)）、[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) §10、[`LTX25_RESEARCH_NOTES.md`](LTX25_RESEARCH_NOTES.md)、[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §69.8・§69.14、オーナー裁定（2026-09-01）。

### 4-34. LTX 2.5の復号が、FHD級×241フレームでVRAMからこぼれる（起票：2026-08-29）

**FHD級×長尺の生成が遅すぎるという苦情・要望が実際に出たとき、または高解像度×長尺を正式にサポートすると決めたとき** → LTX 2.5の映像の復号（潜在表現から画素へ戻す処理）のVRAMの膨張を減らせるかを測り、減らせるなら手を入れる。

- **概要**: LTX 2.5の復号は、**1920×1088×241フレームで予約VRAMがカードの総量を超え、WDDMの共有システムメモリへこぼれる**（実測値の正本は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §79.2。**本書には写さない**）。**落ちはしないが、這うように遅くなる。**
- **何が塞いでいるか**: 何も塞いでいない。**画角拡張が持ち込んだ問題ではなく、出荷済みの復号経路そのものの性質**であり、画角拡張を使わない普通の生成でも同じ形状なら同じことが起きる。**この形状に到達するのは高解像度×長尺を狙った利用者だけで、そのとき起きるのは失敗ではなく減速である**——直す対象は画角拡張ではなく復号側であり、いま急いで着手する理由が無い、というのが優先度を下げて本節へ置いた理由である。
- **着手時の入口**: **符号化の側では、同じ形状の膨張が`set_conv3d_memory_format`（重みの並びを変えるだけの処置）で大きく落ちている**（往復の一致度はほぼ不変）。原因は bf16 の3次元畳み込みが im2col 経路へ落ちることなので、**復号側にも同種の余地があるかどうかをまず測ること。** 測る前に実装へ進まないこと——符号化で効いたから復号でも効く、とは限らない。
- **関連**: [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-134（画角拡張のトークン予算）——**その較正で退避へ反転した点は、拡張なしの対照が平坦だったことから画角拡張に固有と確定しており、本項には帰属しない**（[`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md) §9.1）。ほかに§3-59（画角拡張のチェーン対応）。
- **出典**: [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §79.2、`outputs/ltx25-outpaint-prep/RESULTS.md`（複写: [`Outputs-archive/ltx25-outpaint-prep/RESULTS.md`](Outputs-archive/ltx25-outpaint-prep/RESULTS.md)）の G0-d、オーナー裁定（2026-09-03）。

### 4-35. FFNチャンキング（残り唯一の未実装VRAMレバー）

**VRAM 8GB級の低VRAM対応に着手するとき** → Transformer内部のFFN（全結合層）チャンキングの実装を検討する。

- **何が塞いでいるか**: VRAM 8GB対応自体が現プロジェクトのスコープ外というオーナー方針（本書§4-4と同根）。FFNチャンキングは主に長尺（シーケンス長が支配的な領域）向けで720pの空間スケールには必須ではなく、16GB運用では効果が薄いため、いま着手する理由が無い。
- **概要**: FFNが隠れ次元を4倍に広げるときの巨大な中間テンソルを、シーケンス方向に分割して計算する省メモリ手法。VRAM削減レバーの棚卸しでは、二段パイプライン・attentionタイル・VAEの空間/時間タイル・block-swapがいずれも実装済みなのに対し、FFNチャンキングだけが未実装である。
- **参考値**: コミュニティ実装（LTX-2専用の`ffn_chunks`）は600フレームで8・800フレームで12〜16・900フレーム以上で16〜24を推奨し、最大で約8分の1までピークを削減すると謳う。ただし実績は24GB環境のみで、GGUF・block-swapとの併用可否は記載が無い。
- **実装形は決まっている（オーナー決定）**: **採るのはチャンク化である。** 中間テンソルはタッチ回数こそ少ないが総バイト量が大きいため、メインメモリへ展開する方式（activation offloading相当）は採らない——PCIe帯域が先読みblock swapの経路と衝突する。チャンク化は分割数を過剰にしても損失が数%で済むので、**最適な分割数を探す境界探索も要らない。**
- **出典**: [`Nz-Videomni/Docs/SCALEUP_16GB_RESEARCH.md`](SCALEUP_16GB_RESEARCH.md) §2（レバー棚卸し）・付録A、オーナーとの議論による実装形の決定（2026-09-05）。**同書は本項を旧番号「§3-44」・旧リポジトリの相対パスで参照したままである**——凍結文書のため本文は直していない（冒頭バナーに注記あり）。

### 4-36. 仮オブジェクトの完了時自動置換（`resolveProvisional`配線）

**β版以降の作り込みに入るとき** → 生成の投入時にタイムラインへ置いた仮オブジェクトを、ジョブ完了時に生成結果の動画へ自動で差し替える機能を載せるかどうかを、右クリック再設計の方針（「仮オブジェクトは編集時のヒントに徹する」）との整合も含めて判断する。

- **何が塞いでいるか**: 何も塞いでいない。α版は手動挿入（操作パネルの🎞ボタン、および右クリックの「⬇ この生成結果を今すぐ挿入」）に留めるというオーナー決定によるものである。
- **現行機構**: 仮オブジェクトは`webui/src/timeline/provisionalReservation.ts`とブリッジRPC`updateProvisionalReservation`（予約席の更新）で成り立っているため、着手する場合は**この現行機構の上に自動置換を載せる**。`webui/src/timeline/provisionalFlow.ts`（`runProvisionalFlow`）は配線されないままの試作モジュールとして撤去済みなので、**それを復活させる話ではない**（[`DEVLOG.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/DEVLOG.md) 35.2）。
- **付随して先送りされる実機検証項目**（いずれも`native/src`に`REALDEVICE-VERIFY`注記として現存し、未検証のまま残っている）:
  - `ReplaceObjectEditProc`のアンドゥ挙動（delete+createが1ステップにまとまるか）。
  - `UpdateObjectTextEditProc`のカーソル依存書込の実機挙動。
  - `ExtractAudioWorker`の残検証（音声抽出のsolo分離は本書§4-21を参照）。※`CutoutRangeWorker`（範囲選択切り抜き）は製品UIから呼ぶ配線が無いため実機検証の対象外（[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-48。配線を再検討するときの入口は本書§3-34）。
- **`register_project_load_handler`の発火は検証済みである**（手動での「ファイル→開く」でも発火する。実測の正本は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §97、クローズ記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-35-02、仕様書側の実機チェックリスト#9も確認済み＝フロントエンド[`RIGHTCLICK_REDESIGN_SPEC.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/RIGHTCLICK_REDESIGN_SPEC.md) 第8節）。なお予約検出はテキスト本文の`[#id]`マーカーを第一手段とする二重化設計なので、縮退しても実害は小さい。
- **出典**: [`DEVLOG.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/DEVLOG.md) §7.6、§7.7、§33.1、§33.3、35.2、`native/src/plugin.cpp`・`native/src/bridge.cpp`の`REALDEVICE-VERIFY`注記。

### 4-37. 操作パネルの状態復帰（起票：2026-07-20）

**生成中にAviUtl2を終了してしまって実害が出た（既存のフォールバック手段で困った）事例・要望が出たとき** → AviUtl2再起動後、操作パネル（Create画面のメインプロンプト欄等）が終了前の状態を復帰しない件に手を入れるかどうかを判断する。

- **何が塞いでいるか**: 何も塞いでいない。「生成中にAviUtl2を閉じる」という事故自体がレアケースであることと、生成済み動画をエクスプローラーから手動でタイムラインへ挿入するといったフォールバックが常に存在することから、優先度は低く、あったら便利という水準にとどまる。
- **出典**: [`DEVLOG.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/DEVLOG.md) §37。

### 4-38. メインメモリ不足時の速度低下を賢く避ける機構（起票：2026-08-02）

**通常運用（生成中は他の重量級作業を控える）でもdenoise減速が頻発すると分かったとき** → 対策の設計に進む。ただし着手時はまずハードページフォルト（`\Memory\Pages Input/sec`等）のstage2実行中の直接計測で機序を確定させてから設計へ入ること。

- **概要**: CPU骨格キャッシュ（`keep_resident`。[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-64）を有効にした状態で物理メモリが不足すると、block swapのCPU側マスター（約11.4GB。denoiseの毎ステップ全ブロックを読む熱い経路）がページファイルへ追い出され、denoiseがSSD読み戻しに律速されて遅くなることがある。これを賢く避ける機構（例: マスターのピン留め＝page-locked化で追い出し対象から外し、denoise中は暇なキャッシュ側に追い出しを引き受けさせる。ほかにキャッシュ対象の絞り込み等）の検討。RAM監視・自動降格の機構もオーナー判断で本項の担当範囲である。
- **何が塞いでいるか**: 何も塞いでいない。着手を正当化する観察——通常運用での減速の頻発——がまだ得られていないだけである。
- **判断材料**: 減速を観測した唯一の実測（バックエンド[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §47.4）は裏で重量級の並行作業が走っていた**条件の汚れた計測**で、機序（ページアウト）も推定にとどまる。同じ状態でも速度低下ゼロの条件がある。**したがって着手時は既存の実測を出発点にせず、測り直しから始めること。** その測り直しには、**未確定のまま残っている仮説H5——「前ジョブの居残りが、CPU側マスターのページキャッシュを追い出す」（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §100.6）——の検証も含めること**（本項と同じCPU側マスターを扱う仮説であり、別々に測ると同じ計測を二度やることになる）。
- **LTX 2.5でも状況は同じである**: 2.5にも`keep_resident`は開通しているが、RAM監視・自動降格の機構は無い（「常駐に十分なメモリがあるかは利用者側の選択であって、サーバーが判定すべき環境の能力ではない」という判断。[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §48.1-10。同じ理由で`GET /status`にも能力フラグを載せていない＝同 §48.7）。**32GB級のマシンでLTX 2.5＋常駐onという組み合わせが、本項が扱う減速のいちばん出やすい条件である**（2.5は常駐offでも生成中のメインメモリが大きい。実測は同 §76.2(3)）。

### 4-39. 参照エンコードのタイル化の残件（起票：2026-09-05）

**参照つきの長尺でIC-LoRAの制御の精度が落ちたと体感したとき、またはその報告が上がったとき** → 下の3点に着手する。**新しい分岐が発火するのは参照自身の潜在トークン数がしきい値を超えたときで、1152×1536なら単一クリップ369フレーム以上がこれに当たる**（解像度が低いほど、発火に必要な尺は長くなる。正本は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §101.2）。

- **何が塞いでいるか**: 何も塞いでいない。しきい値以下の帯は出力のバイト同一性を優先して据え置いてあり、制御の精度が落ちたという報告が上がるまでは着手しない、というのが優先度を下げている理由である。
- **やること（3点）**:
  1. **LTX 2.5に残る小さな実差の原因を切り分ける。** タイル化した参照の潜在には、LTX 2.3より大きい誤差が残っている。有力仮説（時間方向の重なりが2.5のVAEの受容野に足りていない）と、切り分けの実験設計（タイル幾何のクロスと重なりの掃引。各エンコードは数秒で済む）は出典にある。**2.5が落ちた場合の分岐（エンジン別のしきい値／重なりの引き上げ／受容）は、着手前にオーナーの裁定を取ること。**
  2. **本番の制御動画コンテンツで測り直す**（出典での呼称は「B-1改」）。canny（輪郭線）・pose（姿勢）系の本番の入力は前処理済みの制御動画だが、手元にある測定値は生の動画を入力にしたものである。
  3. **快適バナーの線を再較正し、あわせて残っているスピルの扱いを決める**（出典での呼称は「ゲートC較正」）。タイル化のしきい値はバナーの線に合わせた固定値で、線そのものは動いていない。しきい値の下（217〜361フレーム相当の帯）には軽いスピル（VRAMが足りないぶんをメインメモリへ退避する挙動）が意図的に残してあるので、これを解消するかどうかも同じ機会に判断する。**動かす対象の線は本書§4-30（stage-1の快適予算の配信化）が扱う25,000と同じ値なので、着手するときは両項をセットで見ること。**
- **着手時の注意**: 測定設計の落とし穴が3点（出力側の二値化・判定定数の置き方・生成物どうしを比べる指標）あり、いずれも出典に書いてある。**外から持ってきた判定定数をそのまま使わないこと。** また、**LTX 2.5側は計測の足場がLTX 2.3に劣る**——参照エンコード区間のピークがログに出ず（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §101.8）、ワーカーログには時刻も付いていない（同§100.9）。2.5を測るなら、**まずログの書式を2.3へ揃えるところから工数を見積もること。**
- **出典**: バックエンド[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) **§101**（本テーマ全体の正本。残件と再訪材料・測定設計の落とし穴の一覧は§101.11）、[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-76（クローズ記録）。

### 4-41. 短い連結に限った一括アップサンプルへの切替（起票：2026-08-23）

**連結生成の1回あたり数秒以下の短縮に確かな需要が出て、かつ「尺によって出力の性質が変わる隠れた切替」を受け入れると判断したとき** → LTX 2.5のChainedで、連結の拡大工程（`chunked_upsample`）を短い連結に限って一括処理へ切り替えるかどうかを判断する。

- **概要**: 連結生成の拡大工程（Stage-1のあとStage-2の前に、連結した潜在表現を空間方向へ2倍に引き伸ばす1回きりの工程。ノイズ除去はしない）は、時間方向に32フレーム＋のり代18のチャンクへ分けて処理している（リクエスト項目`chunked_upsample`。APIの既定はオフ、Gradio検証UIと操作パネルの既定はオン）。**Stage-2を潜在22フレームの窓で処理する仕組みとは別の機構である。** LTX 2.5ではVRAM消費がトークン数に線形なので、総尺が約20万トークン以内（1280×768で約69秒・1920×1088で約32秒）の連結なら、一括で拡大しても溢れない。これを根拠に、短い連結に限って一括拡大へ切り替える案があった。
- **何が塞いでいるか**: ①得られるのは拡大工程の数秒以下の短縮だけである。②無条件に一括へ戻すと連結の総尺に上限が生まれ、「連結生成は長さが柔軟」という棲み分け原則（[`CHAIN_STAGE2_RESEARCH_NOTES.md`](CHAIN_STAGE2_RESEARCH_NOTES.md) §7）と衝突する。③上限を超えたときだけチャンク化へ自動で戻す条件つきにしても、チャンク化経路は3次元畳み込みのメモリ配置（`channels_last_3d`）を変えるため一括と出力が一致せず、尺が上限をまたいだ瞬間に出力の性質が静かに変わる隠れた切替になる。「規則は単純に、例外を増やさない」という設計方針に反する。**オーナー裁定（2026-09-09）: 実装しない。**
- **出典**: [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §71.4・§71.5・§72.2(d)・§72.6、[`LTX25_RESEARCH_NOTES.md`](LTX25_RESEARCH_NOTES.md) 11節、`CHUNKED_UPSAMPLE_WORKORDER.md`、[`CHAIN_STAGE2_RESEARCH_NOTES.md`](CHAIN_STAGE2_RESEARCH_NOTES.md) §7（棲み分け原則）。

---

## 本台帳の位置づけ（運用規則）

節の分け方と各節の意味は**冒頭の「位置づけ」に一本化**してある（本節では繰り返さない）。個々の項目の正本は各出典ドキュメント（`DEVLOG.md`・`API_REFERENCE.md`・`TIMELINE_ALPHA_REQUIREMENTS.md`・`REAL_BACKEND_CHECKLIST.md`・`WEBVIEW2_PARITY_BACKLOG.md`、およびバックエンド`Docs/`の各文書等）にあり、本書はそれを優先度順に一覧するための派生的な台帳である。運用は次のとおり。

- 新たな未着手タスクが判明した場合は、出典を明記のうえ「1. 近日中の改修項目」「3. 将来の研究課題」「4. スコープ外」のいずれかへ追記する。**該当する節が無い場合は、冒頭の「位置づけ」の書式で節を立て直してから追記する。** §3と§4の線引きは、**前提が変われば着手しうるもの＝§3／当面は着手しないと判断済みのもの＝§4**とする。§3の内部はさらに「改修項目（先）＞研究課題（後）」に分かれる（オーナー指定の階層）。
- **§4「スコープ外」の項目には「何が塞いでいるか」を必ず書く。** 前提が変わったかどうかを、あとから読んだ人が判定できるようにするため。**再訪の条件がはっきりしているものは、条件を行頭に置いて書く**（§3-30〜§3-38の書式）。
- **1項目は「見出し＋本文2〜4行＋出典」を上限の目安にする。** 詳細は正本の文書へ置き、台帳は「次に何をすべきか」が読み取れる密度に保つ。**判断材料が他文書に無いときは、先に出典側（研究ノート等）へ書いてから本書を参照に切り替える**——縮約で記録が消えてよいわけではない。
- **本書には歴史的経緯を書かない**（オーナー決定・全節に適用）。**本書に載せてよいのは「これから何をするか」「そのための判断材料」「本台帳の運用規則」の3つだけ**であり、「過去に何があったか」は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md)の責任である。本書はオーナーやAIエージェントが「次に何をすべきか」を確認するために読む場所であり、歴史はノイズになる。旧番号の欠番はそのままでよい。
- **歴史的経緯（クローズ・移動・欠番の来歴）は本書に書かない——それは[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md)の役割であり、両書を読み比べれば分かることを本書へ書くのは単なる重複で、読む人（とAIエージェント）の時間とトークンの浪費になる**（オーナー指示・2026-09-01）。
- **クローズした項目の跡地に説明文は残さない**（記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md)および`DEVLOG.md`等の出典へ移す）。**例外は各節の冒頭に置く「欠番の対応は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md)冒頭を参照」という総括の1行だけ**——番号ごとの個別の欠番説明・移動の説明は置かない（同書冒頭が正本）。
- 項目の実装が完了したら「2. 実装済み・ユーザーのテスト待ち」へ移す（書式は冒頭の位置づけ2番を参照）。**該当する節が無い場合は、冒頭の「位置づけ」の書式で節を立て直してから移す。** オーナーのテスト（実機・目視・実GPU）に合格したら[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md)へ移すか、本書から除去する。正本側の更新にも合わせること。
- **例外（オーナー裁定2026-09-01）: 実機で確認できる要素が原理的に無い項目——MCPサーバーやフロントエンドのモックに閉じた改修のように、GPUもAviUtl2本体も関与しないもの——は、自動ゲートが全部緑になった記録を根拠に、§2を経ず直接[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md)へ移してよい。** 根拠記録（どの物差しで何件通ったか）は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md)へ必ず残し、クローズ記録からそこを指すこと。
- **ブランチは`main`（利用者に配る）と`dev`（開発中。その日の改修を全部載せる）の2本だけで、テーマごとの`feature/<slug>`ブランチは作らない**（オーナー裁定2026-09-15。2026-09-07の運用を置き換え）。理由は、機能ごとに枝を分けると機能どうしの相互作用をmainへ入れる前に見つけられないから——開発中の1本に丸ごと載せていれば、実機ゲート・目視・指紋比較が常に組み合わさった状態に対して走る。各セッションは`dev`へコミット＆プッシュし（コミットはテーマごとに分け、明示パスで足す——1件が落ちたときに`git revert`で切り離して残りをmergeするため）、機械検証とその日のオーナー目視が済んだら`git merge --no-ff dev`でmainへ入れる。mergeの合図は暦ではなく検証である。**ユーザーはmainを`git clone`／`git pull`する**ので、検証の済んでいないものをmainへ置いてはならない。
- **全項目が合格して空になった節は、見出しごと削除する**（オーナー決定）。合格記録は本書に残さず[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md)へ一本化し、他文書からの参照もそちらへ付け替える。**節番号の再採番はしない**——過去の文書・記憶が番号で参照しているため、削除した節の番号は欠番のままにする。
- **本書に断りなく現れる略号の凡例**（初めて読む人向け）:
  - **W1〜W9／X1〜X6／Y1〜Y3**＝フロントエンド微調整バッチの第1〜第3波（[`DEVLOG.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/DEVLOG.md) §46〜§48）。
  - **G0・G1・G4…**＝各テーマの実機ゲート（オーナーが実機で通す確認項目）の番号。番号と内容の対応は**テーマごとに独立**しているので、必ずそのテーマの正本（[`DEVLOG.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/DEVLOG.md)の該当節・各WORKORDER・バックエンド[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md)）で引くこと。
  - **N1〜N13**＝バックエンド同梱のGradio UIに対するパリティ（同等機能）項目の番号（[`WEBVIEW2_PARITY_BACKLOG.md`](../AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/WEBVIEW2_PARITY_BACKLOG.md)）。
- **他文書から本書や`PENDING_TASKS_CLOSED.md`を参照するときは、必ずファイル名を添えて書く。行番号は書かない**（本書は頻繁に増減するため、行番号はすぐ古くなる。過去に節番号の振り替えも起きており、番号だけでは新旧どちらを指すのか判別できないため）。
