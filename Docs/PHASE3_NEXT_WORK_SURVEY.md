# PHASE3_NEXT_WORK_SURVEY — 次作業サーベイ（意思決定用メモ）

- 作成日: 2026-07-03
- 作成方法: リサーチ・コーディネーションセッション（コード読解サブエージェント×2 + Webリサーチ×5系統の統合）
- リポジトリ状態の前提: **main @ 1ab5e0c**（サーベイ執筆時点の状態。現在の main は `c7da787`）（clip-concat＝マスク付きAVラテント連結アーキテクチャ、マージ済・push済）。旧 `_EXTEND` monkeypatch 機構は削除済（byte-match検証済）。
- 表記規約: 各主張に **[code-verified]**（コードを実際に読んだ）／ **[web-sourced: URL]**（一次ソースあり）／ **[inference]**（推論・未検証）を付す。プロジェクト規約どおり **research-first**：コミット前にスパイクが必要な箇所は明示的にフラグする。
- 本ファイルは未コミット（次セッションの計画材料）。

---

## 0. エグゼクティブサマリ

1. **IC-LoRA はユーザー最優先関心事項であり、技術的には「思ったより近い」**。wheel に `ICLoraPipeline` と state-dict レベルの LoRA 融合（`fuse_loras.apply_loras`）が完備しており [code-verified]、我々の GGUF ローダは**ロード時に全量 bf16 へ dequant して wheel の StateDict 型を返す** [code-verified] ため、融合パスの型がそのまま噛み合う。make-or-break だった「GGUF量子化重みにLoRAを当てられるか」は、**我々の設計では非問題**（量子化重みに当てるのではなく、dequant後のbf16に融合する）[code-verified + inference の合成、§2.3]。
2. ただし IC-LoRA は **LTX-Desktop パリティを超えるスコープ**（Desktop 本体は IC-LoRA UI を持たない [web-sourced]）。仕様 §13.4b が Phase 4 に置いた理由がこれ。着手はユーザーのスコープ判断が前提。
3. 目の前に**ゼロコストのゲート**が2つ：キーフレーム視覚検証のユーザー目視レビュー（未マージブランチのマージ判断）と、chain A/B 最終目視。これを先に消化すると計画が確定する。
4. 推奨順序: **①レビューゲート消化 → ②IC-LoRA スパイク（research-first）→ ③Retake/Gap Fill か 小粒プロダクト価値項目（ユーザー選択）**。詳細は §4。

（訂正: 上記3.の「未マージブランチのマージ判断」は執筆時点の状態。本branchは既に merge `7f31935` で main 入り済み。残るのは目視サインオフのみ＝§3.7参照）

---

## 1. 候補一覧（サマリ表）

| # | 項目 | 何か | 事前資産 | 工数 | リスク | 16GB実現性 | ユーザー判断依存 |
|---|------|------|----------|------|--------|-----------|------------------|
| 1 | **IC-LoRA**（深掘り§2） | 参照/制御信号をトークン列に連結する条件付け + LoRA | wheel完備・engine配線は削除済 | スパイクS→本実装M | 中（RAM・トークン増） | 高（§2.5） | **要**（パリティ超えスコープ） |
| 2 | Gap Fill | 2キーフレーム間の中割り生成 | slice-1条件付けが下地 | M | 中 | 高 | 要（優先度） |
| 3 | Retake | 区間差し替え再生成 | wheel `RetakePipeline`+`TemporalRegionMask` | M〜L | 中 | 高[inference] | 要（優先度） |
| 4 | V2V | 映像→映像変換 | wheel/公式は IC-LoRA 制御系で実現 | L | 高 | 要スパイク | 要（Phase4扱い） |
| 5 | 標準LoRA再導入 | de-forkで削除したLoRA機能の復活 | fuse-at-load路線ならIC-LoRAスパイクと同一基盤 | S〜M | 低 | 高 | 小 |
| 6 | 生成キュー | 逐次+キャンセル | 仕様§13.4項目4 | S〜M | 低 | 無関係 | 小 |
| 7 | 長尺 ~30s | chain延長 | chain基盤マージ済（22s検証済） | S〜M | 中（品質逓減） | 実測要 | 小 |
| 8 | プロンプト強化（text-only） | LLMによるprompt enhancement | Gemma text-only常駐済 | S〜M | 低 | 高 | 小 |
| 9 | パラメータ露出 | STG/σスケジュール/negative/seed-lock | wheelに存在 | S | 低 | 無関係 | 小 |
| 10 | 空間アップスケーラ露出 | 既存アップスケーラのUI/API露出 | 実装済・露出のみ | S | 低 | 高 | 小 |
| 11 | attention tiling再導入 | de-forkで削除 | fork履歴に残存 | M | 中 | （長尺の天井に効く可能性） | 小 |
| 12 | チューニング諸課題 | §3参照（seam口パク停止 等3件） | 実装済機能の品質調整 | 各S | 低 | 無関係 | 小 |
| 13 | レビューゲート | キーフレーム目視+chain A/B最終目視 | 成果物生成済 | **0（ユーザー作業）** | — | — | **本体** |

除外確定: VLM/vision再導入は **やらない**（text-only Gemma による QAT 22.7GB 回収と正面衝突。仕様側でも除外済）[code-verified: 仕様§13.4]。

---

## 2. IC-LoRA 深掘り

⚠️ 本§2はPhase A実装前の理解。実装で確定した実態＝本番既定はper_layer_quant=True（量子化のまま）、全量bf16 dequantはLoRA融合用の副経路。正確な経路整理は IC_LORA_PHASE_A_STATUS.md 参照。

### 2.1 メカニズム（正確な理解）

- **IC-LoRA = 「凍結バックボーン + LoRA」×「参照キャンバスのin-context連結」**。参照画像/制御信号（pose・depth・canny・参照シート等）を VAE でラテント化し、**生成対象と同じトークン列に連結**する。相互作用は通常の self-attention で起き、denoise 完了後に参照側ラテントを **crop（切り落とし）** する。チャンネル連結（ControlNet系）では不可能な柔軟性が売り。[web-sourced: AVControl論文 arXiv:2603.24793 / https://matanby.github.io/AVControl/ ／Lightricks開発者のHFコメント「crop the guiding latents after the denoising process」]
- 強度制御は attention-mask への **log空間の加算バイアス**で行う（参照ごとの強度・空間マスク）。[web-sourced: ComfyUI-LTXVideo の GuideAttentionMask 実装（GitHub）]
- 概念的起源は In-Context LoRA（arXiv:2410.23775）[web-sourced]。
- **重要な既視感**: この「guideトークンを列に append → denoise後に strip」は、我々が slice-1（多キーフレーム条件付け）で既に自前再現したパターン（`VideoConditionByKeyframeIndex` 系）と**同じ条件付けファミリー** [code-verified: 我々のslice-1実装 + web-sourced: ComfyUI `LTXVAddGuide.append_keyframe` が `torch.cat([latent, guide], dim=2)` で温度軸連結]。エンジンは既にこのパターンを実運用している。

### 2.2 wheel の既存サポート（すべて [code-verified]）

- `ltx_pipelines/ic_lora.py` に **`ICLoraPipeline`** が存在。二段distilledパイプラインで、**LoRA は stage-1 の ModelLedger にのみ適用**。`VideoConditionByReferenceLatent` + 参照用 attention masking を使用。`reference_downscale_factor` を LoRA safetensors のメタデータから読む。
- `ltx_core/loader/fuse_loras.py` の **`apply_loras()`** が、`SingleGPUModelBuilder.build()` 内・**モデル実体化の前**に、**state-dict レベルで** `B @ (A × strength)` を基底重みへ融合する。bf16 と FP8 二形態に対応。**GGUF int4 には非対応**（が、後述のとおり我々には不要）。
- キー形式: wheel は `lora_A`/`lora_B`（+ ComfyUI の `diffusion_model.` プレフィクスを `LTXV_LORA_COMFY_RENAMING_MAP` で剥がす）を期待。CivitAI/Kohya 形式（`lora_down`/`lora_up`）のリマップは**削除済フォークファイルにのみ**存在（`git show d0d3df5^:vendor/LTX-Desktop-LOW-VRAM/backend/services/lora_service.py`）。

### 2.3 make-or-break 問題：GGUF Q4 + block-swap スタックに LoRA を当てられるか

**結論：当てられる見込みが高い。しかも我々の構成では「量子化重みへのLoRA適用」という難問自体が発生しない。** 根拠の分解:

- [code-verified] 我々の `engine/gguf/loader_service.py` の `GGUFStateDictLoader.load()` は **GGUF 全体をロード時に bf16 へ dequant** し、wheel の StateDict 型を返す。つまり融合対象は常に bf16 テンソルであり、`apply_loras()` の bf16 分岐がそのまま食える。
- [code-verified] 融合は `SingleGPUModelBuilder.build()` 内＝ **BlockSwapService / DitCpuLoadService のインストールより前**に完了する。順序衝突なし。ブロックは融合後も CPU からストリームされるので、**VRAM 天井は変わらない**（変わるのは RAM 側の一時コスト）。
- [inference] コスト面: 一時的に「フル bf16 state dict + 融合デルタ」が RAM に載る。再量子化ステップは存在しないため融合後ブロックは bf16 のまま——ただしこれは**現行ロードパスでも同じ**（我々は元々 bf16 で保持・ストリームしている）ので、増分は LoRA デルタ計算の一時メモリのみ、のはず。**スパイクで実測すべき**（RAM≥32GB前提の維持確認、[[no-large-pagefile-disk-requirement]] 規約に抵触しないこと）。
- 対照となる prior art [web-sourced: city96/ComfyUI-GGUF GitHub]: ComfyUI-GGUF は量子化重みを保ったまま **毎 forward で dequant→calculate_weight→cast** するパッチ・アット・コンピュート方式（実験的だが動作、LTX-2 でも実用報告あり。Lightricks#407 では GGUF ワークフローの LoRA が公式フル精度より良好だったという報告すらある。厳密な品質ベンチは存在しない[negative finding]）。**我々は fuse-at-load が自然**（ステップ毎コストゼロ）で、ComfyUI 方式を真似る必要がない。
- 代替ルート [code-verified: git履歴]: 削除済みフォークの 370 行 `LoraService`（nn.Linear への実行時 forward-hook、「FP8対応・block-swapのデバイス移動を生き延びる」）は履歴から復元可能。ただし**旧 wheel API（`apply_lora_to_model`）前提で wheel 1.0.0 には存在しない**ため、復活にはAPI適合作業が要る。fuse-at-load 路線が転けた場合のフォールバック。

### 2.4 ベースチェックポイント適合性

- 我々のベース: **LTX-2.3-22B-distilled-1.1 の Q4_K_M GGUF**（models/ 実在確認 [code-verified]）。
- 公式 IC-LoRA は **ベースチェックポイント特異的**（2.3-22b 系 vs 2-19b 系）[web-sourced: HFモデルカード群]。**2.3-22b 系ファミリーが我々のベースと一致**。wheel の `ICLoraPipeline` 自体が「distilled モデルを使う」設計 [code-verified] ——好適合。
- コミュニティ注記 [web-sourced]: 「distilled-lora は dev/full ベースにのみ必要」という区別あり。我々は distilled ベースなので追加の distilled-lora は不要のはず [inference]。

### 2.5 16GB / リソース影響

- VRAM の公表数値は**どこにも無い** [negative finding — Web全域]。
- 理詰めの見積り [inference]: LoRA ファイル自体は数十〜数百MB。fuse-at-load なら実行時 VRAM 増分ほぼゼロ（ブロックはCPU常駐→ストリームのため）。attention バイアスマスクは無視できる規模。
- **本当の増分は「参照キャンバスによるトークン列延長」**：参照は出力解像度の 0.5x で連結（Union カード記載 [web-sourced: HF Lightricks/ltx-2.3-union-control 相当カード]）→ attention コストとアクティベーションが伸びる。我々の VRAM 天井は denoise ②が支配（[[vram-spill-timing-16gb]]）なので、**尺・解像度の上限が下がる可能性がある。スパイクで実測必須**。

### 2.6 この製品（AviUtl2 ブリッジ／映像制作用途）で価値の高いアダプタ

公式リリース済（HF Lightricks org、2.3-22b 系）[web-sourced]:

| 優先度 | アダプタ | 用途 |
|--------|----------|------|
| 高 | **Pixel-Spatial-Upscaler x2/x4** | 1080p 到達手段として既に外部アップスケーラを想定中——本命候補 |
| 高 | **Detailer** | 品質底上げ（19b 系のみの可能性あり→適合確認要 [inference]） |
| 高 | **In-Outpainting** | マスク・二段・Laplacian blending。編集ワークフローに直結 |
| 中 | **Pose / Union-Control (canny+depth+pose)** | 制御生成。LTX Studio では「Pose/Depth/Edges」という平易な名前で露出 [web-sourced] → UI もこの語彙に倣う |
| 中 | **Ingredients** | 参照シートによるキャラ/小道具/ロケ一貫性（768x448/121f・strength 1.4・30 steps 学習）——キャラ物制作に効く |
| 中 | **Motion-Track** | モーション追従 |
| 低 | HDR / Deblur / Colorization / Decompression / Day-To-Night / Water-Simulation / Instant-Shave / Cross-Eyed | ノベルティ・特化系 |
| 別枠 | カメラ制御 LoRA（static/dolly/jib） | **非IC の標準 LoRA**。項目5（標準LoRA再導入）だけで載る |

### 2.7 スコープ上の位置づけ（ユーザー判断ポイント）

- **LTX-Desktop は IC-LoRA/制御 UI を出していない**（Lightricks の 2026 ロードマップ扱い）[web-sourced: Desktop README + CrePal レビュー]。Web 版 LTX Studio は Pose/Depth/Edges の V2V を出している [web-sourced]。
- ゆえに仕様 §13.4b は IC-LoRA を Phase 4 に置いた（「Desktop 自身が露出していない」が根拠）[code-verified: 仕様]。**着手＝意図的なパリティ超え**。ユーザーは明示的に関心を示しているため、これを「意図的スコープ決定」として確認するのが筋。

### 2.8 段階的アプローチ案（research-first 規約準拠）

- **Phase A（スパイク・最小・~1セッション）**: 公式 2.3-22b IC-LoRA を1本（候補: Pose か Upscaler）、fuse-at-load で我々の GGUF-dequant パスに載せ、512×320 で参照条件付き生成を1回通す。計測: RAM ピーク／VRAM ピーク／トークン数増分／生成時間。**回帰確認: LoRA 無しストック経路の byte-match**（決定性資産を活用）。engine 側変更は `fast_video_pipeline.py:213-219` の `loras=None` ハードコード解除 + キー形式正規化の最小限。
- **Phase B（条件付け配線）**: `VideoConditionByReferenceLatent` 相当を我々の条件付け層（slice-1 の append-and-strip 基盤）に統合。`reference_downscale_factor` メタデータ読取り。chain（クリップ連結）との併用可否を確認（wheel は stage-1 のみ LoRA 適用——我々の chain 設計とも整合 [code-verified]）。
- **Phase C（API/UI 露出）**: /generate へのアダプタ選択+参照入力。命名は LTX Studio 流の平易語彙（Pose/Depth/Edges/Upscale）。
- **Phase D（拡張）**: 複数 LoRA 同時・強度 UI・In-Outpainting のマスク UI。

### 2.9 未解決の問い（ユーザーへ）

1. パリティ超えスコープに正式着手して良いか（仕様 §13.4b の Phase 4 前倒し）。
2. 最初に載せたいアダプタはどれか（推奨: Pixel-Spatial-Upscaler か Pose——製品価値と検証容易性のバランス）。
3. fuse-at-load の RAM 一時コストの許容ライン（32GB 機で成立が条件、[[no-large-pagefile-disk-requirement]]）。
4. UI 露出の語彙・粒度（Studio 式の平易名 vs アダプタ名そのまま）。

---

## 3. その他候補の詳細

### 3.1 Gap Fill（仕様 item 3・意図的後回し）
2キーフレーム間の中割り生成。slice-1 の多キーフレーム条件付け（実装+客観検証PASS、目視レビュー待ち・未マージ branch `feature/phase3-api-unfreeze-conditioning`）が直接の下地 [code-verified: docs]。キーフレームレビューで観測された **f17 ゴースト／static-hold-then-jump 補間特性**は Gap Fill の品質にそのまま効くため、**レビュー結果を見てから着手判断**が合理的。工数 M・リスク中・16GB 問題なし [inference]。

### 3.2 Retake（仕様 item 3・意図的後回し）
生成済みクリップの区間差し替え。wheel に **`RetakePipeline` + `TemporalRegionMask`** が存在 [code-verified]。配管は wheel 資産に乗れるが、chain 新アーキテクチャ（masked AV-latent concat）との整合設計が必要 [inference]。工数 M〜L・編集ワークフロー価値は高い。

### 3.3 V2V（仕様 §13.4b Phase 4）
公式エコシステムでは V2V 制御は実質 **IC-LoRA（Pose/Depth/Edges）で実現**されている [web-sourced: LTX Studio]。つまり **IC-LoRA をやれば V2V の大半が従属的に手に入る**——独立項目として先行させる理由は薄い。工数 L・単独着手は非推奨。

### 3.4 標準 LoRA 再導入（仕様 item 4）
de-fork で削除。**IC-LoRA スパイク Phase A の fuse-at-load 基盤がそのまま標準 LoRA 基盤**（カメラ制御 LoRA 等が即載る）。IC-LoRA をやるなら実質無料の副産物、やらなくても S〜M で単独成立。

### 3.5 小粒プロダクト項目（仕様 item 4）
- **生成キュー**（逐次+キャンセル）: S〜M・リスク低・製品価値高。
- **長尺 ~30s**: chain 22s 検証済 [supervisor確認]。延長は品質逓減とVRAM/時間の実測が要る。S〜M。
- **プロンプト強化（text-only）**: Gemma text-only 常駐と両立 [code-verified: QAT回収設計]。S〜M。
- **パラメータ露出**（STG/σ/negative/seed-lock）: wheel 資産の露出のみ。S。
- **空間アップスケーラ露出**: 実装済機能の露出。S。IC-LoRA Upscaler と役割整理が要る [inference]。
- **attention tiling 再導入**: fork 履歴から復元可。長尺天井に効く可能性 [inference]。M。

### 3.6 チューニングバックログ（各 S・実装済機能の品質調整）
1. **seam 口パク停止**: 発話 chain の継ぎ目で口が止まる緩和。
2. **tile-seam ドリフト**: stage2 タイル継ぎ目後のわずかな漂い。
3. **ハーネス音声閾値**: 発話開始での false-positive 修正。
いずれも chain マージ済み基盤の上の調整。IC-LoRA スパイクの合間の埋め草に好適。

### 3.7 レビューゲート（工数ゼロ・最優先）
- **キーフレーム目視**: `outputs/phase3_multikey_smoke/visual_bookend|visual_multikey` 生成済み。レビュー→ branch `feature/phase3-api-unfreeze-conditioning` のマージ判断＋Gap Fill 品質見通し。（訂正: 本branchは既に merge `7f31935` で main 入り済み。残るのは目視サインオフのみ）
- **chain A/B 最終目視**: マージ非ブロッキングだが完了宣言に必要。

---

## 4. 推奨順序と根拠

1. **レビューゲート消化**（ユーザー目視2件）——工数ゼロで、キーフレームブランチのマージと Gap Fill の品質見通しが同時に確定する。全計画の前提。
2. **IC-LoRA Phase A スパイク**——ユーザー最優先関心 + 技術的追い風（wheel完備×bf16-dequantパスの適合）+ research-first 規約に沿う最小検証。副産物として標準 LoRA 基盤（3.4）とカメラ制御 LoRA が手に入り、V2V（3.3）の必要性判断材料にもなる。**着手前にユーザーの「パリティ超えOK」確認**。
3. **スパイク結果で分岐**: 成立→ Phase B/C（本実装）。不成立/コスト過大→ フォールバック（フォーク LoraService 復元の API 適合スパイク）か、Retake/Gap Fill へ転進。
4. **並行の埋め草**: チューニング3件（3.6）と小粒項目（キュー・パラメータ露出）はスパイクのGPU待ち時間や合間に消化可能。
5. **Gap Fill / Retake** はキーフレームレビュー結果と IC-LoRA の帰趨を見てから優先度を再判定。
6. **V2V 単独着手はしない**（IC-LoRA に従属）。

**最大の未知数（スパイクで潰す）**: ①fuse-at-load の RAM 一時ピーク実測、②参照キャンバスのトークン延長が 16GB denoise 天井に与える影響、③公式 IC-LoRA キー形式が wheel の `lora_A/B` 期待と一致するか（一致しない場合は正規化シム、フォーク削除ファイルに前例あり）。

---

## 5. 主要ソース一覧

- wheel: `.venv-engine/Lib/site-packages/ltx_pipelines/ic_lora.py`（ICLoraPipeline）、`ltx_core/loader/fuse_loras.py`（apply_loras）[code-verified]
- engine: `engine/gguf/loader_service.py`（GGUFStateDictLoader＝全量bf16 dequant）、`engine/pipeline/fast_video_pipeline.py:213-219`（loras=None ハードコード）、`engine/lora_types.py`（スタブ残存）[code-verified]
- 削除済フォーク LoraService: `git show d0d3df5^:vendor/LTX-Desktop-LOW-VRAM/backend/services/lora_service.py` [code-verified: 履歴]
- IC-LoRA 機構: AVControl arXiv:2603.24793 / https://matanby.github.io/AVControl/ ／In-Context LoRA arXiv:2410.23775 [web-sourced]
- 推論ノード実装: Lightricks/ComfyUI-LTXVideo（LTXICLoRALoaderModelOnly・LTXVAddGuide.append_keyframe・GuideAttentionMask）[web-sourced: GitHub]
- GGUF×LoRA prior art: city96/ComfyUI-GGUF（per-forward dequant+patch、実験的サポート）、Lightricks/LTX-Video#407 実用報告 [web-sourced]
- 公式 IC-LoRA モデル群: HuggingFace Lightricks org（Union-Control / Motion-Track / Pixel-Spatial-Upscaler / In-Outpainting / Ingredients ほか）[web-sourced]
- Desktop UI 状況: LTX-Desktop README + CrePal レビュー（IC-LoRA UI 無し・2026 ロードマップ）[web-sourced]

---

## 6. 監督・ユーザー議論による補足（2026-07-03）

> サーベイ本体（§0〜5）作成後、監督とユーザーの間で以下を議論し方針を確定した。次セッションの着手判断はこちらを優先する。

- **Gap Fill／Retake の位置づけ**: フロントエンド（AviUtl2 タイムライン）の意味論に依存する機能のため、**Phase 2 最小統合後に実用から要件を逆算する**（仕様書 §13.1 の early-integration 原則と一致）。汎用の受け入れ口は engine 層の既存プリミティブ（任意時刻への参照 latent 注入＋時間領域 denoise mask）に留め、API はフィーチャー毎の薄いエンドポイントとする（凍結APIの教訓＝実在する消費者なしに汎用APIを切らない）。→ §3.1/§3.2 の Gap Fill／Retake は次セッションでは着手せず据え置き。
- **IC-LoRA の構成の再確認**: (a) LoRA 重み融合（ロード時・全パイプラインに直交）＋ (b) 参照トークン列連結（既存の append-and-strip 族、slice-1 と同じ条件付けファミリー）の2部品のみで**専用新機構は不要**と確認できた → 「基本パイプライン先行・高度機能（連結/Gap Fill）との併用は後段の合成」という段階導入が技術的に成立する。併用時（chain との組み合わせ）の参照トークン×タイル化の相互作用は要 spike（§2.8 Phase B で確認）。
- **実装上の必須ゲート**（§2.8 Phase A スパイクに追加で課す）: LoRA 無効時の byte-match 完全一致、および LoRA 付け外しによる RAM transformer キャッシュ無効化コストの計測。
- **最初のアダプタ選定方針**: Pose/Depth/Canny 系は信号抽出前処理器（DWPose 等）の依存追加が必要なため後回し。生動画参照系（Pixel-Spatial-Upscaler／In-Outpainting／Deblur）を先行させる。**推奨第一号＝Pixel-Spatial-Upscaler x2/x4**（既知の「1080pは外部アップスケーラ」制約を生成的に置換し得る・2.3-22b世代一致）。
- **Phase 2（AviUtl2統合）の状態**: ユーザーのプラグイン開発環境整備待ちで**ブロック中**。→ **次セッションは IC-LoRA Phase A をメインスコープとする**。
