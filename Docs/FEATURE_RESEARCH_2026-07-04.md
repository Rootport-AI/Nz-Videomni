# 機能リサーチまとめ（2026-07-04）— フロントエンド計画の入口

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

## B. video-to-video 継続（次セッションの本題・設計の入口）

**やりたいこと**: アップロードした動画を入力にして、その「続き」を生成する。

**機構**: 入力動画を VAE で潜在化し、末尾の潜在を固定して、そこから続きを denoise する。既存のクリップ連結が使っている「overlap 部分の潜在を引き継ぐ」仕組みと同じ型。

**我々の現状（重要な区別）**:
- `reference_video_id` は IC-LoRA の条件付け専用であり、続きの生成には使えない（`api/generate.py`）。
- クリップ連結（chain）の clip1..N は「自分で生成した潜在」を overlap で引き継ぐものであって、アップロードした動画から開始することはできない（`api/models.py` では clip0 だけが conditioning を受け取る）。
- つまり「外部から持ち込んだ動画を起点に続きを作る入力口」が存在しないのが差分。

**必要になる作業（見込み・設計時に確定させる）**:
- エンジン側に「動画→潜在」のエンコード経路（VAE encode）を新設する。
- それを既存のチェーン機構（`engine/pipeline/chain_pipeline.py` の overlap-freeze mask ／ initial latent の仕組み）へ接続する。
- API 表層を用意する（不正な入力を弾く 422 事前検証を含む）。
- **設計論点**: 音声トラックも同時にエンコードするか。動画と音声の潜在の整合をどう取るか（`chain_math.py` の `AUDIO_LATENTS_PER_SEC=25.0` が関係）。

**引き継ぐ制約**:
- 解像度は 64 の倍数（÷64）、フレーム数は 8n+1、IC-LoRA 併用時は 128 の倍数（÷128）。
- 16GB VRAM 天井。
- 凍結 API（新パラメータの追加は API 設計として慎重に。既存契約を壊さない形で）。

---

## C. audio-to-video（近い将来）

**やりたいこと**: 用意した音声（波形）に合わせて動画を生成する。ASR（音声認識）ではない。

**機構**: 音声波形→mel→音声 VAE→音声潜在（1 潜在 ≈ 1/25 秒・128 次元）を固定し、動画・音声が組になった潜在のうち動画側だけを denoise する（＝同時生成の逆向きの運用）。LTX-2 は動画 14B／音声 5B の双方向クロスアテンション構造を持つため、リップシンク（口の動きと音声の一致）が生成の中で成立する。オープンウェイトで実行可能で、クラウド専用ではない。

**我々との距離**: `chain_pipeline.py` の `initial_audio_latent` ＋ overlap-freeze mask が同じ型の機構。主に未配線なのは「外部音声をエンコードして入れる入力口」だけ。

**出典**: arXiv 2601.03233（LTX-2 論文）、ltx.io/model/model-blog/how-to-generate-video-from-audio、ComfyUI の ia2v ワークフロー。

---

## D. 高品質モード（遠い将来・ユーザー決定で当面は着手しない）

**ここでの「品質」とは**: 解像度ではなく生成の忠実度のこと。
- distilled（蒸留版）: 8 ステップ・CFG 無効（＝negative prompt が効かない）。現行の既定経路。
- dev/base（非蒸留版）: 30〜50 ステップ・CFG 有効（＝negative prompt が効く）。手・文字・色・動きの忠実度が上がる。

**非蒸留 × 量子化だけの重みは実在する**（＝サイズを抑えつつ dev を使う道がある）:
- `unsloth/LTX-2.3-GGUF` リポジトリ直下の `ltx-2.3-22b-dev-Q4_K_M.gguf`（HF 表示 ~14.3GB）。現行の distilled Q4（実測 16.55GiB）と同クラス・同じ 22B アーキ。
- 公式 fp8 も `Lightricks/LTX-2.3-fp8` に dev 版がある。

**着手する場合の入口**:
- 重みの差し替え ＋ worker の配線（`services/ltx_runner.py:849-862` の payload に `pipeline`／`guidance_scale`／`negative_prompt` が未配線）。
- 計算コストは ~7〜12 倍（ステップ増 × CFG の 2 回 forward）。CFG 時はアクティベーションも増えるので 16GB での実測が必要。

**参考**: コミュニティ言及「dev は動きとプロンプト追従が良い／distilled は画は良いが動きが乏しい」（HF RuneXX/LTX-2.3-Workflows #71）。多様性の定量比較は未確認。

---

## E. 不要と決定（遠い将来）— 静止画生成・プロンプト強化

ユーザー決定により実装しない。参考情報のみ残す:
- LTX 公式の旧実装はプロンプト強化に Llama-3.2-3B-Instruct をローカルで使い、テキストのみで強化していた（プロンプトが ~120 語未満のとき自動発火）。
- 我々のテキストエンコーダ gemma-3-12b-it は instruction-tuned（一次資料で確認済み）なので流用は可能だが、CPU 推論は 1〜3 tok/s と遅い。
- 効果の A/B 実証は薄い。

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
