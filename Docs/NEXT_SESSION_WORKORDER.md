# （歴史記録）旧・次セッション ワークオーダー — Phase 1 残タスク

> **（歴史記録・2026-07-05 注記）本書は 2026-07-02 時点の計画であり「次セッション」の指示書ではない。** 現在の正本＝[`NEXT_SESSION_HANDOFF.md`](NEXT_SESSION_HANDOFF.md) 冒頭ブロック（次セッション＝GUI への V2V/A2V 露出＋モデル管理の並行オーケストレーション）。本書記載のうち shared 溢れ最適化は V2V セッションの WDDM 是正（`b2c20ee`）で解消済み。

- 作成: 2026-07-02
- 対象: 次セッション担当者
- スコープ: **Phase 1 残のうち 4 件** — ①dead-code 整理 ②shared 溢れ最適化(e) ③keep_resident 恒久化 ④開発ゴミ掃除
- 除外: Gradio 手動動作確認（別途）
- 上位文書: [`NEXT_SESSION_HANDOFF.md`](NEXT_SESSION_HANDOFF.md)（現状ステータス＋チェックリスト）／[`../LTX23_Backend_Specification.md`](../LTX23_Backend_Specification.md)（§13 ロードマップ）／[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md)（実測の正本）

> **共通ルール（このプロジェクトの鉄則）**: システム Python を汚さない（全て project-local venv）。手当たり次第の実験禁止＝着手前に仮説→Docs/コードで裏取り→検証スコープを明確化。コード改変タスクは **SHA256 byte-match ゲート**（下記）で退行確認。破壊的操作は監督チェックポイント。

---

## ⚠️ 着手前に読む「誤認防止」サマリ

このセッションの調査で判明した、着手前に押さえるべき前提:

1. **タスク②(e)「shared 溢れ」は“ほぼ解消済み”**。§11 te-offload と §12 dit-cpu-load で load/encode の一時 shared 溢れ（旧 ~2–2.5GB）は実測でほぼ消えている。残る可変軸は **denoise stage2（高トークン時）＝解像度×尺の本質的な容量限界であり、バグではない**。→ **(e) を新規実装タスクとして着手しないこと**。まず「done 相当に再分類 or denoise stage2 軸を別枠化」をユーザーと確認する（§後述）。
2. **タスク③keep_resident 恒久化は“信頼性ブロッカーではない”**。現状 comp=1/keep=0 は連続 T2V/I2V+音声で安定 PASS（§10.3/§10.7）。これは速度最適化（任意）で、しかも vendor wheel 凍結の制約で単純な in-place 化が効かない可能性がある。**軽い気持ちで着手せず、計測前提の設計をしてから**。
3. **タスク①dead-code は安全に進む部分と要確認部分がある**。高確度の 2 箇所は byte-match ゲートで淡々と消せる。署名互換の no-op 引数は upstream caller 確認が要る。
4. **タスク④ゴミ掃除は方法をユーザーと相談してから**（何を残すかの判断が要る・.venv 等の重い再構築物は掃除対象から切り分ける）。

---

## タスク① dead-code 整理

**目的**: text-only Gemma 化・de-fork で no-op / 未使用化したコードを、退行なく除去して保守性を上げる。
**規模**: 小。 **依存**: コード編集（byte-match ゲート必須）。 **前提**: 独立。

### 退行確認ゲート（全サブタスク共通・最重要）
`Docs/VERIFICATION_LOG.md §13.2 / §14.3` の手法。削除コミット後に T2V と 最小I2V を生成し、出力 mp4 の SHA256 が baseline と**バイト一致**することを確認する。
- T2V baseline: `23844b4e…6bb7bf`（§14.3 記載条件）
- 最小I2V baseline: `a511eda4…c217`（§14.3 記載条件）
- ※正確な生成条件（prompt/seed/解像度/frames/steps）と値は **§14.3 を正**とする。まず §14.3 を読んでから着手。

### 削除候補（確度順）

| # | 対象 | 場所 | 現状 | 確度 | 備考 |
|---|------|------|------|:---:|------|
| 1 | `_SkipGemmaLMSDOps`（クラス全体） | `engine/gemma/gguf_quant_service.py:201-231` | 完全 no-op（QAT 消去で skip 対象キーが存在しない） | **高** | §14.4「belt-and-suspenders で温存」 |
| 2 | `_read_target_vocab_from_header` の未使用 `path` 引数 | 定義 `…:234` ＋ 呼び出し元 `…:399` | 純粋な署名互換ダミー（本文で未参照・:248 に明記） | **高** | 呼び出し元は1箇所。関数本体は不変 |
| 3 | Defensive strip ループ | `…:427-435` | no-op（#1 が効くため base_sd に該当キー無し） | 中 | **#1 とセットで同時削除**（単独削除は防御が薄くなるので不可） |
| 4 | `attention_tile_size` / `loras` 署名パラメータ | `engine/pipeline/fast_video_pipeline.py:28,34,51,57,76,82,117,155` | no-op（Service 削除済・:221-227 に明記） | 中 | **upstream caller 確認必須**（worker.py / config / API が明示指定していないか）。config 既定 0/空なら実害なし |
| 5 | Layer offload / text_encoder configurator の旧 namespace 参照（docstring・comment） | `engine/gemma/layer_offload_service.py:31,90,177-181` ほか `text_encoder_configurator.py` | 実装は robust（fallback 探索）だが docstring が旧 multimodal namespace を参照 | 低 | **削除でなく docstring 更新のみ**（教育的価値あり温存推奨） |

**削除しないもの（dead に見えるが機能している）**: RMSNorm `+1` 補正（`…:612-623`・city96 verbatim・**必須**）／`multi_modal_guider_factory_denoising_func`（:465・"multimodal"=video+audio 意で text-only でも使用）。

### 進め方（推奨）
1. §14.3/§14.4 を読む → baseline SHA と温存理由を把握。
2. **#1+#3 を1コミット**（依存関係）→ byte-match ゲート。
3. **#2 を1コミット** → byte-match ゲート。
4. #4 は upstream caller を grep 確認 → 使用箇所ゼロを確認できたら削除・ゲート。使用があれば温存し理由を記録。
5. #5 は docstring 更新のみ（ゲート不要だが軽く mock pytest）。

### 監督/ユーザー確認事項
- #4（署名パラメータ削除）: 外部 API / config.yaml / worker が `attention_tile_size` / `loras` を明示的に使っていないか、消す前に確認。

---

## タスク② shared 溢れ最適化(e) — ✅ done（ユーザー確認 2026-07-02）

> **確定**: (e) は **done**。§11/§12 で load/encode の一時溢れは解消済みで、残る軸は denoise stage2（能力限界・バグでない）。**次セッションでの着手不要**。以下は根拠の記録。

**結論から**: 旧残(e)「load/encode の一時 shared 溢れ（~2–2.5GB）」は **§11/§12 で実質解消済み**。

### 事実（VERIFICATION_LOG より）
- §7.9 で load/encode の shared 溢れを実測（Gemma encode shared **2039MB**、transformer load shared **2457MB**）＝これが旧(e)。
- §11 te-offload: Gemma encode shared **1,616→0MB**、encode peak 15,839→10,540MB（−33%）・出力バイト一致・速度改善。
- §12 dit-cpu-load: transformer-load shared → **419MB**、load peak 15,815→1,444MB（−14.4GB）・出力バイト一致・速度改善。
- §12.5 注記: 全体天井は 512×320 で 16,944→~9.2GB。**残る可変軸は「高トークン時の denoise stage2」のみ**。

### 次セッションの動き
- **(e) を新規実装タスクとして着手しない。**
- 選択肢（ユーザー判断）:
  - (A) **(e) を done 相当に再分類**し、handoff/spec のチェックリストを更新（推奨）。
  - (B) 「高解像度・長尺での denoise stage2 の容量」を**別タスク**として切り出す。ただしこれは**バグでなく能力限界**（`Docs/RESOLUTION_DURATION_CAPABILITY.md` §8.4/§8.6 が正本＝spill-free frames の話）。attention tiling / FFN チャンキング等は spec §13 では Phase 3+ の最適化枠。
- どちらにせよ、まず `RESOLUTION_DURATION_CAPABILITY.md §8` と `VERIFICATION_LOG §7/§11/§12` を読んで現状を確認してから決める。

---

## タスク③ keep=1 常駐モードの新設 ＝ ✅ 調査完了につき CLOSE（2026-07-02）

> ※この「keep=1 は non-viable」という結論は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §47 のバグ修正と §48 の製品化で覆った。現在の正本は §48。

> **確定（2026-07-02・コード読解＋Web リサーチ＋ユーザー決定）**: 当初構想の keep=1（フル GPU 常駐でジョブ間再ビルドを消す）は **16GB では原理的に non-viable**、かつ狙った利得は既存経路（`--dit-cpu-load`/`--te-offload`＋OS RAM/mmap キャッシュ）で概ね捕捉済み。よって **新規実装せず close**。既定 keep=0 は不変。**詳細＝`VERIFICATION_LOG.md §15`（仮説 H1–H4 の検証・一次情報つき）**。将来 Phase3 の長尺連結で漸増が実害化したら、GPU 常駐ではなく §15.4 の「CPU 正本温存＋層ストリーミング」で再着手する。
> ↓以下は当時の調査前メモ（背景として温存・結論は上記 §15 が正）。

> **方針（2026-07-02 ユーザー決定・調査前）**: 既定 keep=0 は**壊さない**。その上で opt-in の keep=1 常駐モードを新設する構想だった。
> **進め方**: ①read-only 実現可能性調査（out-of-place 移動が first-party `engine/` 内か wheel 側かの切り分け＝**実装可否の分水嶺**）→ ②監督/ユーザーで go/no-go → ③go なら branch で実装＋byte-match/VRAM/commit 検証。→ **①②実施結果＝non-viable と判定し close（§15）**。

**目的**: keep=0 の「毎ジョブ再 materialize による gen 時間漸増（720p 4本で ~115→134s, +~20%）」を、opt-in の keep=1 常駐モードで解消し、720p でも crash させない。
**規模**: 中。 **依存**: 独立（信頼性ブロッカーではない）。 **性質**: 速度最適化・任意・Phase3 先行投資。

### 根本原因（VERIFICATION_LOG §10.2・確信度高の read-only 調査）
keep=1 → `ModelLedger._target_device()` が CPU → GGUF Gemma を **CPU ビルド → `model._apply(t.to(cuda))` で out-of-place に全体を GPU コピー**（`engine/gemma/gguf_quant_service.py` L1280 build / L1305-1313 move）。**CPU 側 ~11GB ＋ 新規 GPU 側**の瞬間二重在が薄い 16GB マージンを超過し、720p の Gemma text-encode 中に native crash（traceback 無し、dedicated ~15.86GB で即死）。

### 該当コード
- `engine/pipeline/fast_video_pipeline.py:170-173`（keep_resident_weights で StateDictRegistry 有効化＝CPU キャッシュ）
- `engine/gemma/gguf_quant_service.py:1280`（`build_device = _target_device()` が CPU に）
- `engine/gemma/gguf_quant_service.py:1305-1313`（out-of-place `.to(cuda)` の発生点）
- `engine/worker.py`（`keep_resident_weights` / `LTX_KEEP_RESIDENT` 分岐）
  - ※`LTX_KEEP_RESIDENT` は当時の手順。2026-08-02 に環境変数の経路は撤去され、現在は API の `keep_resident` フィールド（`POST /generate`・`POST /generate/chain`。既定 `false`＝keep=0 と同じ状態）で指定する（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §48）。

### 修正の方向性（すべて**仮説・実装前・要計測**）
1. **in-place 化**: `model._apply(lambda t: t.to(cuda))` を param 単位の in-place（`param.data = param.data.to(dev, copy=False)` 等）へ。瞬間二重在を消す狙い。**リスク大**: `GGMLQuantizedTensor` の subclass attrs（`_ggml_type`/`_float_shape`）が out-of-place `.to()` override で保持されている前提を崩す恐れ。かつ **vendor wheel 凍結の `ltx_core` 側 `_apply` はオーバーライドできない**可能性＝根本構造を変えられない懸念。
2. **移動前 CPU 解放**: 移動直前に `del`＋`gc.collect()`＋`empty_cache()` で CPU 側を先に解放。簡単だが**スコープ内参照で GC されない可能性＝実測必須**。
3. **keep=1×te-offload の相互作用**を精査（L1304-1309 の layer_offload 分岐）。CPU 層を置き去りにして一部だけ GPU 移動する現挙動が正しいか未確認。

### 検証（推定・§11/§12 に倣う）
- 計測: `torch.cuda.max_memory_allocated()` ＋ Windows perf-counter の Dedicated/Shared Usage（`torch` の max_alloc だけでは WDDM shared 溢れを見逃す）＋ committed bytes（commit 枯渇判定）。
- 条件: 1280×768/121f・seed 固定・distilled・keep=1 明示・te/dit 既定 ON。
- 受け入れ: ①native crash 消滅（exit 0）②gen 時間が keep=0 より改善 ③出力が keep=0 と数値等価（byte 一致 or 許容）④連続ジョブで commit 平坦（leak なし・§9.7 参照）。

### 代替（割り切り）
§9.7 でリーク修正済＝ comp=1/keep=0 でも commit は bounded。単一ユーザー逐次運用では gen 漸増は許容範囲との判断もあり。**着手コストとリターンをユーザーと確認してから**。

---

## タスク④ 開発ゴミ掃除 ＝ 一部done・残はユーザー手動キュレーションへ委譲（2026-07-02）

> **確定（2026-07-02・ドライラン提示後のユーザー決定）**:
> - ✅ **`.claude/settings.json` の `Bash(git clean *)` 許可は削除済**（commit `efb406e`）。
> - **`outputs/` は削除しない**＝生成時間・VRAM 溢れの**一次情報**でドキュメント（VERIFICATION_LOG / RESOLUTION_DURATION）の根拠。後日ユーザーが手動整理する。`uploads/`・`logs/` も同性質（`logs/ltx_worker.log`＝peak_vram 一次ソース）ゆえ保全。
> - `__pycache__`(108)/`.pytest_cache` は純粋な再生成物で任意消去可（今回は未実施）。
> - ドライラン実測（参考）: outputs/ の非.py 全ファイル＝304 files / 105.6MB。保全すべき .py ハーネスは 16 本（うち `outputs/qat_reclaim_verify_textonly/run_verify.py` は byte-match ゲート本体）。
> ↓以下は当時の調査メモ（背景・温存）。

**目的**: リポジトリの散らかりを整理。 **前提**: **掃除方法（削除スクリプト/手動/段階）はユーザーと相談してから**（ユーザー指示 2026-06-30）。

### 現状（.gitignore は良好＝追跡漏れ・大型バイナリ流入なし）

**フェーズ A（明白なゴミ・低リスク・ただし実行はユーザー合意後）**
- `outputs/`（~127MB・72 dir。テスト生成物と名前付きスイープ）→ `.gitkeep` のみ残して掃除
- `uploads/`（~68KB・テスト入力 PNG 2枚）
- `logs/`（~172KB・ランタイム再生成）
- `__pycache__` / `.pytest_cache` / `engine/**/__pycache__`（<1MB・自動再生成）

**フェーズ B（要ユーザー判断＝再現性が要るか）**
- `outputs/phase5b_diag/`（1.6MB・145 files・診断スクリプト＋測定ログ）
- `outputs/vram_sweep/`（診断スクリプト・manifest）
- `hf_home/`（~3.7MB・HF DL キャッシュ。オフライン/低速回線なら残す）

**掃除対象から切り分ける（＝「開発ゴミ」ではない・触るのは別判断）**
- `.venv` / `.venv-engine`（~9.7GB）・`.uv_cache`（~12GB）・`.python`（~68MB）= **環境本体**。消すと全再インストール。ルーチンのゴミ掃除には含めない（クリーン再構築を意図する時だけ）。

**残す（削除厳禁）**: `models/`（~29GB・実行必須）・`engine/` `api/` `services/` `scripts/`・`config.*` `pyproject.toml` `requirements.txt`・`Docs/`・`.git/`・`vendor/LTX-2`。

### 別途フラグ（settings レビュー）
- 調査中、`.claude/settings.json` に許可行 `"Bash(git clean *)"` が含まれることを検出。`git clean` の誤発火は未追跡物の一括消去につながるため、**この許可の要否をユーザーに確認**（不要なら削除を推奨）。掃除タスクで `git clean` を使う場合は必ず `-n` ドライラン→目視→実行。

### 検証（掃除後）
`git status`（clean 確認）／`git clean -nd`（ドライランで消し残り/消しすぎ確認）／`du -sh .`（容量確認）。

---

## 推奨着手順（順序は規定でなく提案）
1. **① dead-code #1+#3, #2**（安全・byte-match で淡々と）→ 保守性が上がり以降が読みやすい。
2. **② (e) の再分類判断**（実装不要・ドキュメント整合のみ）をユーザーと確定。
3. **④ ゴミ掃除フェーズ A**（方法合意後）＋ settings の `git clean` 許可レビュー。
4. **③ keep_resident 恒久化**（最も重い・計測前提。やるか割り切るかをユーザーと決めてから）。

各タスクの done は [`NEXT_SESSION_HANDOFF.md`](NEXT_SESSION_HANDOFF.md) の「Phase 1〜3 やることリスト」に反映する。
