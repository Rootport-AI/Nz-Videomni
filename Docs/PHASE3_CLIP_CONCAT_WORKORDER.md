# 次セッション ワークオーダー — Phase 3 スライス2「クリップ連結（生成プリミティブ）」

> **（歴史記録・2026-07-05 注記）本スライス2＝実施済み・main マージ済（実施記録＝[`PHASE3_CLIP_CONCAT_STATUS.md`](PHASE3_CLIP_CONCAT_STATUS.md)・VERIFICATION_LOG）。本書は 2026-07-02 の着手時ワークオーダーで「次セッション」の現役指示ではない。現在の正本＝[`NEXT_SESSION_HANDOFF.md`](NEXT_SESSION_HANDOFF.md) 冒頭ブロック。**

- 作成: 2026-07-02（監督＋ユーザー合意）
- 対象: 次セッション担当者
- 上位: [`../LTX23_Backend_Specification.md` §13.4 #2](../LTX23_Backend_Specification.md)（Phase 3 の作業 #2＝クリップ連結）／[`NEXT_SESSION_HANDOFF.md`](NEXT_SESSION_HANDOFF.md)／[`VERIFICATION_LOG.md` §17](VERIFICATION_LOG.md)（スライス1＝API解凍の実施記録）
- 前提: **スライス1（凍結API解凍＝多キーフレーム条件付け）は完了・main入り済**（merge `7f31935`）。本スライスはその上に立つ。

> **鉄則（不変）**: システム Python 不可触（全 project-local venv）。**手当たり次第の実験禁止＝仮説→裏取り（WEB/コード読解）→スコープ確定→実験**。監督役はコーディング/テスト/調査を専門サブエージェントへ委譲（read-only 調査／非破壊テスト）。破壊/契約変更は監督＋ユーザー チェックポイント。長文脈に引きずられず本来の目的を見失わない。

---

## 0. スコープ

- **今回の主眼＝spec §13.4 #2「クリップ連結（生成プリミティブ）」**: ブックエンド I2V（前クリップ終了＝次クリップ開始の共有境界）＋**自己回帰 extend** で 5s クリップを繋ぎ長尺化する、**バックエンド完結**の生成プリミティブ。
- **★本スライスの入口は「実装」ではなく「リサーチ」**（§1）。LTX 2.3 の extend/連結が**潜在（latent）レベルの文脈**をどう扱うかを掘り下げてから、作るものの粒度を決める。
- **今回やらない**:
  - **AviUtl2 側（タイムライン配置・SDK・フロントエンド）**＝spec でも「AviUtl2 が担う」と明記。リポジトリ未整備ゆえ本スライスは**バックエンド＋ハーネス/API のみで完結・検証**する。
  - **Gap Fill／Retake**（spec §13.4 #3・大規模・後続）。
  - **IC-LoRA／V2V**（Phase 4）。
  - **VLM(vision) 再導入**（将来課題・§13.4c）。

## 1. ★入口＝先行事例＆基礎機能リサーチ（最優先・read-only／WEB・実装前の分水嶺）

**目的**: 「末尾フレームから次を作る」を**素朴なピクセル連結**として実装しないこと。ユーザーの見立て＝**LTX 2.3 は末尾数フレームの潜在テンソルに含まれる"文脈"から生成する**性質を持つ。この**文脈（context）の扱い**を正しく理解してからスコープを決める。

**裏取りすべき問い（サブエージェントに調査させる・コード＋WEB）**:

1. **LTX 2.3 の native な extend / 自己回帰の機構**は何か。
   - 前クリップの**潜在（latent）**を次生成の条件にする経路はあるか（例: `VideoConditionByReferenceLatent`＝参照動画、あるいは denoise 初期状態への prior-latent 注入）。**インストール済み wheel（`.venv-engine`）に何が実在するか**を必ず確認（スライス1 の教訓＝vendor ミラーでなく**実際に import/実行される engine＋wheel**を読む）。
   - 「文脈」を担うのは**末尾の何 latent フレーム**か（latent 1枚 ≒ pixel 8枚）。overlap 幅・重み付けの定石。
2. **先行事例の実装**:
   - `Lightricks/ComfyUI-LTXVideo` の `looping_sampler.md`（overlap＋AdaIN＋normalizing latents）＝シーム/ドリフト対策の具体。
   - LTX-Desktop／公式が**長尺・extend**をどう実現しているか（アプリのUX＋下層API）。
   - LTX-2 の pipelines に**専用の extend/continue 経路**があるか（`ltx_pipelines`）。
3. **文脈の受け渡し形状**: 潜在の連続性を保つために、どのテンソル（clean_latent／latent／denoise_mask／RoPE 位置）をどう引き継ぐか。スライス1 で使った `VideoConditionByLatentIndex`/`VideoConditionByKeyframeIndex` と、この「latent 文脈引き継ぎ」の関係。

**アウトプット**: 「LTX 2.3 の正しい extend 機構＝◯◯（latent 文脈の扱い含む）／インストール済み wheel での実現可否／先行事例が付加している連続性処理」を根拠付き（file:line＋URL）で確定し、**§2 のスコープ分岐に判断材料を与える**。

## 2. スコープ分岐（§1 の結論で確定・現時点で厚薄を決め打ちしない）

- **薄い版**: 連結はクライアント編成主体（末尾フレーム/latent 抽出→次クリップ条件）。バックエンドは補助（末尾取り出し・プロンプト伝播）＋ドキュメント。規模小。
- **厚い版**: バックエンドが overlap＋AdaIN＋latent 正規化＋latent 文脈引き継ぎでシーム/ドリフトを吸収する連結プリミティブを提供。LTX-Desktop パリティに近い。規模中〜大。
- 判断は §1 で「wheel/engine が既にどこまで持つか」を見てから（スライス1 は"部品は既に wheel にあった"＝厚い実装を回避できた。同じ問いを立てる）。

## 3. 連続性の3論点（設計時に潰す）

1. **シーム（継ぎ目）**: 境界の色/動き不連続。overlap＋AdaIN＋latent 正規化（looping_sampler）。
2. **ドリフト蓄積**: 多クリップで色/露出/動きが漸増ずれ。アンカー/正規化戦略。
3. **プロンプト伝播**: 「グローバル基底＋クリップ毎 override」（text-only）。API 形状を決める。

## 4. 検証計画

- **回帰（不変経路）**: T2V `23844b4e…6bb7bf`／単一 I2V `a511eda4…c217`（seed=12345/distilled/8step/512×320/49f・`outputs/qat_reclaim_verify_textonly/run_verify.py`）は **byte-match 維持**が条件。連結機能で既定経路の数値が変わってはならない。
- **新機能（機能/目視）**: ①2クリップ連結が**境界で破綻しない**（目視・連続性）②多クリップ（3本以上）で**ドリフトが許容内** ③プロンプト伝播が効く ④**VRAM 16GB fit**（`RESOLUTION_DURATION_CAPABILITY.md` の spill-free 内・keep=0 の gen 時間漸増も実本数で再計測＝連結は本数が多い）⑤mock pytest 緑。
- 計測は `torch.cuda.max_memory_allocated`＋WDDM Dedicated/Shared（§11/§12 に倣う）。

## 5. 監督/ユーザー チェックポイント

- §1 リサーチ後に **スコープ（薄い/厚い）＋API 形状**を提示・確定してから実装。
- 契約に触る場合（新エンドポイント/フィールド）は旧クライアント互換を保つ。
- 破壊/契約変更・engine 改変はユーザー チェックポイント。

## 6. 進め方（推奨）

1. **入口ゲート＝スライス1 のキーフレーム目視チェック**（ユーザー・`PHASE3_KEYFRAME_VISUAL_VERIFICATION.md`）。連結はブックエンド条件付けの上に立つため、目視 OK を前提にしたい（NG でも main 入り済ゆえ strength/グリッド微調整で対応・連結着手はブロックしない）。
2. **§1 リサーチ**（read-only／WEB・サブエージェント）→ LTX 2.3 の latent 文脈 extend 機構＋wheel 実在＋先行事例を確定 → go/no-go＆スコープ材料。
3. スコープ（薄い/厚い）＋API 形状をユーザーに提示・確定。
4. branch で実装 → 回帰 byte-match＋新機能の機能/目視（連続性）＋VRAM／gen 時間漸増。
5. handoff / spec §13.4 チェックリスト更新。

## 7. 一次情報ポインタ

- スライス1 の実装＆教訓＝[`VERIFICATION_LOG.md` §17](VERIFICATION_LOG.md)（特に §17.7 プロセス学習＝**実行される実コードで裏取り**／§17.1 engine 実経路の真実＝インストール済み wheel の `DistilledPipeline`＝`combined_image_conditionings` は無く replacing_latent＋add_guiding_latent、モジュール global 差し替えで hybrid 化）。
- 条件付けクラス（インストール済み wheel を必ず確認）: `ltx_core/conditioning/types/`（`latent_cond.py`/`keyframe_cond.py`/`reference_video_cond.py`）・`ltx_pipelines/utils/helpers.py`（`image_conditionings_by_replacing_latent`/`..._by_adding_guiding_latent`）・`ltx_pipelines/distilled.py`。
- engine 実経路: `engine/pipeline/fast_video_pipeline.py`（`_run_inference` の monkeypatch 群＝sigma/euler/conditioning hybrid）・`engine/worker.py`。
- 先行事例: `Lightricks/ComfyUI-LTXVideo`（`looping_sampler.md`＝overlap＋AdaIN＋normalizing latents／`LTXVAddGuide`）・`Lightricks/LTX-2`（`ltx_pipelines`）・`Lightricks/LTX-Desktop`（長尺 UX）・`ltx23.org`。
- 能力/尺リファレンス: `RESOLUTION_DURATION_CAPABILITY.md`（解像度×尺の spill-free・gen 時間）。
