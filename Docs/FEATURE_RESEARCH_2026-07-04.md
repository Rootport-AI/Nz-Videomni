# 機能リサーチまとめ（2026-07-04）— フロントエンド計画の入口

> **（歴史記録・2026-07-06 注記）本書の「次セッション＝B（video-to-video）」は達成済み。** video-to-video・audio-to-video・GUI 露出はいずれも実装済み・mainマージ済（VERIFICATION_LOG §24/§25/§26）。現在の次セッションは **IC-LoRA strength 可変化**（[`NEXT_SESSION_HANDOFF.md`](NEXT_SESSION_HANDOFF.md) 冒頭が正）。本書は将来判断の背景資料（高品質モード D節等）として温存。

> **本書の位置づけ**: 本家 LTX Desktop との機能比較と、今後の実装候補（video-to-video / audio-to-video / プロンプト強化 / 高品質モード）を調べた結果を、次セッションがすぐ設計に入れる形でまとめた参照資料。会話ログの再録ではなく要点のみ。出典は各節に記載。
>
> **次セッションの本題は B（video-to-video 継続）の設計→実装**。まず B を読めば足りる。A・C〜G は背景と将来判断のための材料。
>
> **ユーザー決定の優先順位（本節が最優先の正）**: ①video-to-video＝次セッションで設計→実装 ②audio-to-video＝近い将来 ③静止画生成・プロンプト強化＝不要 ④高品質モード＝遠い将来。

---

## A. 本家 LTX Desktop との比較（要点）

**LTX Desktop とは**: Lightricks 製のオープンソース（Apache-2.0）デスクトップアプリ。バージョン 1.0.5（2026-04-28）。ノンリニア動画編集（NLE）とローカル生成を統合したアプリで、NVIDIA 16GB 以上・ディスク 160GB 以上を要求する。Web サービスの「LTX Studio」とは別物。公開 API は持たない。

**我々が本家を超えている点（コードで裏取り済み）**:
- マルチキーフレーム 5 点の条件付け（`api/models.py`）。
- IC-LoRA の canny/pose 制御を露出（本家はアプリ側で LoRA 自体を "yet"＝まだ非対応）。
- 潜在（latent）の overlap によるクリップ連結。
- REST API を持つ（本家は API なし）。
- ネイティブな同期音声＝動画と音声を同時生成（`engine/pipeline/fast_video_pipeline.py:878` が video と audio を同時に生成）。
- 必要ディスク 28GB（本家 160GB）。

**我々に足りない点**:
- video-to-video の入力経路（→ 次セッションで解消）。
- audio-to-video（音声から動画を作る運用）。
- Retake（動画の一部領域だけ再生成）。
- 既存動画の後処理アップスケール。
- 静止画生成。
- プロンプト強化。
- 高品質モードの実配線。

**主要出典**: github.com/Lightricks/LTX-Desktop（README／Releases）、ltx.io/ltx-desktop、crepal.ai・ltx-23.org のレビュー記事。

---

## B. video-to-video 継続 — ✅達成済み（節をスリム化）

**やりたいこと**: アップロードした動画を入力にして、その「続き」を生成する。**機構**: 入力動画をVAEで潜在化し、末尾の潜在を固定してそこから続きをdenoiseする（クリップ連結の「overlapの潜在を引き継ぐ」仕組みと同じ型）。

本節は当時「次セッションの本題＝設計の入口」として書いたもので、**設計・実装・検証はすべて完了している**。着手前の見込み作業リスト（VAE encode経路の新設・チェーン機構への接続・API表層と422事前検証）は役目を終えたため削除した。現在の正本は[`V2V_CONTINUATION_DESIGN.md`](V2V_CONTINUATION_DESIGN.md)（設計・幾何・ゲート表）と[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §24（検証実績・§24.7＝音声接続の知見）。引き継ぐ制約（÷64・8n+1・IC-LoRA併用時÷128・16GB天井・凍結APIは加算的拡張のみ）は現在も不変。

---

## C. audio-to-video — ✅達成済み（機構・出典の参照用に残す）

**やりたいこと**: 用意した音声（波形）に合わせて動画を生成する。ASR（音声認識）ではない。

**機構**: 音声波形→mel→音声 VAE→音声潜在（1 潜在 ≈ 1/25 秒・128 次元）を固定し、動画・音声が組になった潜在のうち動画側だけを denoise する（＝同時生成の逆向きの運用）。LTX-2 は動画 14B／音声 5B の双方向クロスアテンション構造を持つため、リップシンク（口の動きと音声の一致）が生成の中で成立する。オープンウェイトで実行可能で、クラウド専用ではない。

**現状**: 実装・検証とも完了済み（設計正本＝[`A2V_DESIGN.md`](A2V_DESIGN.md)／検証＝[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §25）。本節は[`A2V_ENTRY.md`](A2V_ENTRY.md)の読む順（2番目）から「機構・出典」の参照先として指されているため残す。

**出典**: arXiv 2601.03233（LTX-2 論文）、ltx.io/model/model-blog/how-to-generate-video-from-audio、ComfyUI の ia2v ワークフロー。

---

## D. 高品質モード（遠い将来・ユーザー決定で当面は着手しない） → 台帳へ移設済み

**本節の内容は2026-07-27の整理で[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §3-2（`two_stage_hq`）へ移設した。以後の管理は同書で行う。** 同項には、ここでの「品質」の定義（解像度ではなく生成の忠実度。distilled＝8ステップ・CFG無効／dev＝30〜50ステップ・CFG有効）、着手の入口（未配線箇所＝`services/ltx_runner.py`のworkerペイロードに`pipeline`／`guidance_scale`／`negative_prompt`が無い／入口の重み＝`unsloth/LTX-2.3-GGUF`の`ltx-2.3-22b-dev-Q4_K_M.gguf`・HF表示~14.3GB／計算コスト~7〜12倍で16GB実測が必須）まで転記済み。

**本節にのみ残す補足**: 公式fp8にも dev 版がある（`Lightricks/LTX-2.3-fp8`）。コミュニティ言及「dev は動きとプロンプト追従が良い／distilled は画は良いが動きが乏しい」（HF RuneXX/LTX-2.3-Workflows #71）。多様性の定量比較は未確認。

---

## E. 不要と決定（遠い将来）— 静止画生成・プロンプト強化

**ユーザー決定により実装しない**（2026-07-04。§Gの優先順位③）。判断の根拠として残す参考情報は次の1点のみ——LTX公式の旧実装はプロンプト強化にLlama-3.2-3B-Instructをローカルで使っていた（テキストのみ・~120語未満で自動発火）。我々のテキストエンコーダgemma-3-12b-itはinstruction-tunedなので流用自体は可能だが、CPU推論が1〜3 tok/sと遅く、効果のA/B実証も薄い。

---

## F. フロントエンド設計への注意（重要）

- **`negative_prompt` / `guidance_scale` / `pipeline` は worker の payload に未配線＝UI に露出しても効かない**（Gradio GUI の `two_stage_hq` はグレーアウト＋注記済み）。これを踏まえて設計する。
- `÷64`・`8n+1`・`÷128`・`limits.max_*`・`spill_free_frames` は `GET /config` から取得して事前バリデーションする（Gradio GUI が実装例＝`gradio_ui/`）。

---

## G. ユーザー決定の優先順位（2026-07-04・再掲）

1. **video-to-video** ＝ 次セッションで設計→実装。
2. **audio-to-video** ＝ 近い将来。
3. **静止画生成・プロンプト強化** ＝ 不要。
4. **高品質モード** ＝ 遠い将来。
