# 次セッション ワークオーダー — Phase 3 スライス1「凍結 API の解凍（条件付け露出）」

- 作成: 2026-07-02
- 対象: 次セッション担当者
- 上位: [`../LTX23_Backend_Specification.md` §13.4](../LTX23_Backend_Specification.md)（Phase 3＝LTX-Desktop 生成パリティ）／[`NEXT_SESSION_HANDOFF.md`](NEXT_SESSION_HANDOFF.md)／[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md)
- 由来: 2026-07-02 のクリップ連結 UX 議論＋LTX-Desktop パリティ調査（下記「調査の結論」）でユーザーが確定した方針

> **鉄則（不変）**: システム Python 不可触（全 project-local venv）。手当たり次第の実験禁止＝仮説→裏取り→スコープ確定→実験。監督役はコーディング/テスト/調査を専門サブエージェントへ委譲。破壊/契約変更は監督＋ユーザー チェックポイント。

---

## 0. スコープ（ユーザー確定 2026-07-02）

- **今回やる**: 凍結 API の「解凍」＝**条件付けの露出**（多キーフレーム・first+last ブックエンド・任意 frame_idx・複数条件・per-item strength）。engine 内で完結する見込み（新パイプライン不要）。
- **今回やらない（Phase 3 だが大規模・後続）**: Gap Fill／Retake。
- **Phase 4（対象外）**: IC-LoRA／V2V。
- **将来課題（計画外）**: VLM(vision) 再導入（enhance_i2v・フレームを見た Gap Fill 提案）。text-only プロンプト強化は可。

## 1. 調査の結論（着手前に押さえる・再調査不要）

3系統（我々のコード読解／LTX-2 公式 repo／diffusers）で裏取り済み:

- **下層は既に対応済み**: 凍結 wheel `ltx_core`/`ltx_pipelines`（@`00dc53d`）が `VideoConditionByLatentIndex`（frame_idx=0・初期 latent 置換）・`VideoConditionByKeyframeIndex`（**任意 frame_idx＝終了フレーム含む・`num_pixel_frames` で複数フレーム**）・`VideoConditionByReferenceLatent`（参照動画）を持つ。ヘルパ `combined_image_conditionings` が **frame_idx==0→LatentIndex / >0→KeyframeIndex** に自動振り分け、複数 conditioning は `state_with_conditionings` で**順次適用**。
- **ブロックは我々の凍結 API のみ**（`api/models.py` の検証＋`services/ltx_runner.py` の frame_idx ハードコード）。
- **first+last ブックエンドは 2 条件（frame_idx=0 と frame_idx=末尾）で実現**。「clip2 開始＝clip1 終了」共有境界が frame-identical にできる。ただし公式には「実験的・未 blessed」＝strength 調整必須（**last は first より僅かに弱める**）・端点ドリフトあり。
- **LTX-Desktop の露出 UX**: First / Last / 中間最大3スロット、各 strength＋位置(25/50/75%)。→ 我々の API は **conditioning item のリスト `{image_id, frame_idx, strength}`** で表現するのが素直（位置＝frame_idx）。

## 2. 変更面（file:line は要再確認・ドリフトあり）

| 対象 | 現状 | 変更 |
|---|---|---|
| `api/models.py` `ConditioningImage`（~L28-32: `image_id, frame_idx=0, strength=0.8(0-1), crf`） | 検証 `len(conditioning_images) ≤ 1`（~L84）／`frame_idx != 0` を拒否（~L87-88） | **複数許可**＋**frame_idx≠0 許可**。新規検証: frame_idx は**8の倍数**・`[0, num_frames)`・件数上限（LTX-Desktop 相当で先頭1+末尾1+中間3＝計5 目安）。8n+1 フレーム／÷64 は不変。 |
| `services/ltx_runner.py`（~L665-673） | `images` list を組むが **frame_idx を 0 にハードコード**（~L670）・`crf` を drop | request の frame_idx を**そのまま渡す**（複数対応）。 |
| `engine/worker.py`（~L177-184） | `ImageConditioningInput(path, frame_idx, strength)` | 変更不要の見込み（複数対応済みか確認）。 |
| `engine/pipeline/fast_video_pipeline.py`（~L619） | images を `_LtxImageInput` に包み distilled pipeline へ | 変更不要の見込み（`combined_image_conditionings` が振り分け）。 |
| 凍結契約の周辺 | — | 契約変更につき `GET /status`（`_STATUS_KEYS` は不変のはず）・`/api/v1/config` の limits・`metadata.json` スキーマ・generation_presets・**mock pytest** を整合更新。API バージョン注記。 |

## 3. ★着手前に検証する技術リスク（最優先・read-only→最小実験）

1. **二段パイプラインの条件付け再注入**: LTX-2 は「条件付けフレームを **Stage 2（アップスケール）でも再注入**しないと詳細が失われる」と報告。我々の `DistilledPipeline`（二段）が **Stage 1/Stage 2 双方に条件付けを適用するか**をコード読解で確認 → 不足なら Stage 2 へも適用する最小改修が要る。**これが実装可否の分水嶺。まずここを潰す。**
2. **latent 粒度**: latent 1枚 ≒ pixel 8枚。「最後の1フレーム」条件付けは実際には末尾 ~8 フレームを縛る。frame_idx は8の倍数に制約。
3. **strength チューニング**: first+last は端点ドリフトしやすい。last をやや弱めるのが定石。既定値と許容レンジを決める。

## 4. 検証計画（byte-match は「不変経路の回帰」用・新機能は機能検証）

- **回帰（不変経路）**: 既存の単一先頭フレーム I2V ＋ T2V は **byte-match 維持**が条件（T2V `23844b4e…6bb7bf` / 最小I2V `a511eda4…c217`・seed=12345/distilled/8step/512×320/49f・`outputs/qat_reclaim_verify_textonly/run_verify.py` 系ハーネス）。解凍で既定経路の数値が変わってはならない。
- **新機能（数値 baseline なし＝機能/目視検証）**: 受け入れ基準
  1. `frame_idx=末尾` 条件付けが**実際に末尾フレームを固定**する（目視）。
  2. first+last（2条件）が**両端を尊重した中間補間**を出す（目視・strength 調整込み）。
  3. 多キーフレーム（中間含む）が各指定位置を反映。
  4. **VRAM が 16GB fit** 内（`RESOLUTION_DURATION_CAPABILITY.md` の spill-free 閾値内で計測）。
  5. mock pytest 緑（契約テスト更新後）。
- 計測は `torch.cuda.max_memory_allocated` ＋ WDDM Dedicated/Shared（過去 §11/§12 に倣う）。

## 5. 監督/ユーザー チェックポイント（着手前に確認）

- **API スキーマ形状の確定**: 条件付けを「item のリスト `{image_id, frame_idx, strength}`」で表現してよいか。件数上限・frame_idx 制約（8の倍数）・strength 既定/レンジ。`num_pixel_frames`（複数フレーム条件）や参照動画（`VideoConditionByReferenceLatent`）は**今回は出さず将来**でよいか。
- 契約変更なので、旧クライアント互換（frame_idx 省略時=0・単一画像）を保つこと。

## 6. 進め方（推奨）

1. **§3-1 の二段再注入をコード読解で確認**（サブエージェント・read-only）→ go/no-go。
2. API スキーマ案をユーザーに提示・確定。
3. branch で実装（validator 緩和＋ltx_runner passthrough＋契約テスト更新）。
4. 回帰 byte-match（不変経路）＋新機能の機能/目視検証＋VRAM 計測。
5. handoff / spec §13.4 のチェックリスト更新。

## 7. 一次情報ポインタ

- 能力マトリクス（我々の engine vs 凍結 API vs wheel）＝本セッションのコード調査（要点は §1-§2 に凝縮）。
- LTX-2 条件付けクラス＝`vendor/LTX-2/packages/ltx-core/src/ltx_core/conditioning/`（`keyframe_cond.py`/`latent_cond.py`/`reference_video_cond.py`）・`ltx-pipelines/.../helpers.py`（`combined_image_conditionings`/`state_with_conditionings`）。
- 公式: `github.com/Lightricks/LTX-2`・`Lightricks/ComfyUI-LTXVideo`（`LTXVAddGuide`/`LTXVConditioning`）・`Lightricks/LTX-Desktop`（パリティ目標）。FLF 実務: `ltx23.org/blog/ltx-23-first-last-frame-comfyui`。
- 連結/シーム先行事例（Gap Fill/Retake 実装時に効く）＝`ComfyUI-LTXVideo/looping_sampler.md`（overlap＋AdaIN＋normalizing latents）。
