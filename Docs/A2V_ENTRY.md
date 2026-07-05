# audio-to-video（A2V）セッション入口 — （歴史記録）

> **（歴史記録・2026-07-05 注記）A2V は完結した**（設計→実装→全ゲート→G3 受容→main マージ・push 済）。本書は着手時の入口であり役目を終えた。現在の正本＝[`NEXT_SESSION_HANDOFF.md`](NEXT_SESSION_HANDOFF.md) 冒頭ブロック・設計/検証＝[`A2V_DESIGN.md`](A2V_DESIGN.md)＋VERIFICATION_LOG §25。

- 作成: 2026-07-05（V2Vセッション末・ユーザー決定「次セッション=A2V」を受けた導線整理）
- 位置づけ: ~~**次セッションの出発点**~~（役目済み）。ここから[リサーチ→設計→ユーザー合意→スパイク→実装]に入る。機能リサーチの原典=[`FEATURE_RESEARCH_2026-07-04.md`](FEATURE_RESEARCH_2026-07-04.md) C節・プロジェクト現在地=[`NEXT_SESSION_HANDOFF.md`](NEXT_SESSION_HANDOFF.md) 冒頭ブロック。
- 用語: 「元動画」「元音声」= ユーザーがアップロードする入力素材。

## 0. やりたいこと（ユーザー要望・確定）

**用意した音声（波形）に合わせて動画を生成する**。ASR（音声認識）ではない。機構=音声波形→mel→音声VAE→音声latentを**凍結**し、AV結合latentのうち**動画側だけをdenoise**する（=同時生成の逆向き運用）。LTX-2は動画14B/音声5Bの双方向クロスアテンション構造なので、リップシンクは生成の中で成立する（C節・arXiv 2601.03233）。

## 1. 大前提（V2Vセッションで裏取り済みの事実・再調査不要）

### 1.1 上流の手本が pin に実在する（最重要）

- **`A2VidPipelineTwoStage`** = `.uv_cache\git-v0\checkouts\821c13058d1842d3\00dc53d\packages\ltx-pipelines\src\ltx_pipelines\a2vid_two_stage.py:41`。`audio_path` を受け取り `decode_audio_from_file(audio_path, device, audio_start_time, audio_max_duration)`（:118）で読む**公式のA2V二段パイプライン**。V2Vにおける `retake.py` と同様、機構の一次参照はこれ。まずこのファイルを通読すること。
- 音声の全凍結プリミティブ: `ltx_pipelines/utils/blocks.py` の `ModalitySpec(frozen=True)`（denoise_maskを全ゼロ化・RetakePipelineの音声分岐とA2Vidが使用）。
  - 【訂正 2026-07-05 設計セッション】`blocks.py`／`ModalitySpec` は**実在しない**（該当ファイル・クラスとも無し）。音声全凍結の**実体は `denoise_video_only`（`ltx_pipelines/utils/helpers.py:476-524`）の denoise_mask 全ゼロ方式**（`noise_scale=0.0`・毎ステップ clean latent へ置き戻し）。正＝[`A2V_DESIGN.md`](A2V_DESIGN.md) §1.1。
- wheelは不可触（v2vと同じ規律）。engine側から直接呼ぶ/自前orchestrationする。

### 1.2 我々のコードベースに既にある部品（V2Vで配線済み・file:line）

- **音声デコード→VAEエンコード→latent**: `engine/pipeline/chain_pipeline.py` `_encode_source_heads`（:236-317）= `decode_audio_from_file`→`ltx_core.model.audio_vae.encode_audio`→latent切り出し。`ledger.audio_encoder()` のロードは実測**45.7MB**（VRAM上の心配なし）。
- **音声latentの凍結denoise**: `_denoise_av_with_carry`（同ファイル :122-196）の `freeze_ka`/mask機構（毎ステップre-pin）。A2Vは「freeze_ka=全長・mask=0」の極限ケース＝`ModalitySpec(frozen=True)` と等価になるはず（要設計判断: どちらの経路を使うか）。
- **音声ジオメトリの単一情報源**: `chain_math.py`（`AUDIO_LATENTS_PER_SEC=25.0`・`a_frames_for_px(px, fps)`）。
- **音声サイドカー/結合の資産**: エンジンの `<stem>_audio_handle.wav` 出力・`services/video_io.py` の `join_v2v`（loudnorm+qsin。A2Vでは出力に元音声をそのままmuxするか等の設計判断に関係）。

### 1.3 入力口が無い（=A2Vの本体作業）

- アップロード機構は**動画専用**（`services/video_upload_store.py`・拡張子 .mp4/.mov/.webm/.mkv・コーデック検査なし）。**音声ファイル（wav/mp3等）の入力口が存在しない** — 新設 or 拡張が設計論点。
- `/generate` 系に音声を渡すAPIフィールドも当然無い。凍結API規律=**加算的拡張のみ**（前例: IC-LoRA Phase B・V2V `source_video`。省略時byte同一+回帰テスト、が定型）。

## 2. 設計セッションで確定させる論点（先回りメモ・勝手に決めない）

1. **API形**: `/generate` に optional `source_audio_id`（+新アップロードエンドポイント `POST /upload/audio`?）か、既存 `/upload/video` の拡張子緩和か。チェーン（`/generate/chain`）との関係（A2V×チェーン併用はv1スコープ?）。
2. **尺の整合**: 元音声の長さ→num_frames（8n+1）へのマッピング（切り詰め/パディング/422）。`audio_start_time`/`audio_max_duration`（上流A2Vidが持つ）を露出するか。
3. **凍結方式**: `ModalitySpec(frozen=True)`（全凍結）か `freeze_ka` mask（部分凍結・strength可変）か。上流A2Vidの実装に従うのが原則（検証の規律=先行事例優先）。
4. **出力音声**: 生成mp4の音声トラックは「元音声そのまま」（=凍結latentのvocoderデコードではなく原波形をmux）か、vocoderデコードか。品質と同期の観点で要検討（vocoderデコードは音質劣化・原波形muxはlatentとのズレの可能性）。上流A2Vidがどうしているかをまず確認。
5. **目視/試聴素材**: 720p・映画トレイラー風の規律は維持しつつ、A2Vは**リップシンク検証**が本丸=「人物が喋る音声」（+音楽のみのケース）を元音声に。素材の入手/生成方法も設計時に決める。

## 3. 引き継ぐ制約・規律（不変）

- 凍結API=加算的のみ・÷64・8n+1・16GB天井・SDPA一択・torch 2.9.1+cu128固定・2プロセス/2venv・巨大ディスク要求禁止。
- **蒸留パイプライン=CFG/negative不可**（worker未配線・露出禁止）。
- 回帰ゲート: T2V `23844b4e…6bb7bf`／I2V `a511eda4…c217` byte-match・pytest基準=**191 passed / 1 skipped**・既存チェーン/V2Vの同一シードbyte一致。
- GPU計測: dedicated+shared両監視（一次ソース=`logs/ltx_worker.log` の `GENERATED_OK/CHAIN_OK peak_vram_mb`）。**WDDMページ降格に注意**=一過性の超過で降格したページは戻らず後続フェーズ全体が共有メモリ実行になる（V2Vで実証・大テンソルはCPU組み立て+`tiled_encode`ストリームが定石）。
- サブエージェント: Opus以下（Fable5禁止）・非破壊・早期報告・**能動ポーリング必須**・**GPUジョブの所有者は1つ**（孫請け禁止・死亡判断前にnvidia-smi+tasklistで裏取り）。
- 進め方: リサーチ→仮説→裏取り→スパイク（GO/NO-GO）→スライス実装→ゲート、の型（V2V=[`V2V_CONTINUATION_DESIGN.md`](V2V_CONTINUATION_DESIGN.md) が直近の成功例）。仕様はユーザー合意後に実装。

## 4. ドキュメント地図（A2V関連の読む順）

1. 本書 → 2. [`FEATURE_RESEARCH_2026-07-04.md`](FEATURE_RESEARCH_2026-07-04.md) C節（機構・出典）→ 3. pin の `a2vid_two_stage.py`（機構の一次参照）→ 4. [`V2V_CONTINUATION_DESIGN.md`](V2V_CONTINUATION_DESIGN.md)（直近の設計/ゲート様式・§2.2幾何・§3ゲート表）→ 5. `engine/pipeline/chain_pipeline.py` `_encode_source_heads`（音声エンコードの実配線）→ 6. VERIFICATION_LOG §24（V2Vの検証実績・§24.7=音声接続の知見）→ 7. [`V2V_AUDIO_JOIN_RESEARCH.md`](V2V_AUDIO_JOIN_RESEARCH.md)（音声接合の定石）。

## 5. スコープ外（混同注意）

- GUIへのV2V/A2V露出（別セッション・**音声スムージングON/OFFチェックボックス要件**が記録済み=VERIFICATION_LOG §24.7将来項目①）。
- モデル管理=[`MODEL_MANAGEMENT_FUTURE_WORKORDER.md`](MODEL_MANAGEMENT_FUTURE_WORKORDER.md)。
- 高品質モード（two_stage_hq）・Retake・プロンプト強化=従来どおり将来/不要。
- GUI目視ゲート+実機e2e=旧宿題のまま（A2VセッションをブロックしないとV2V時に確認済みの扱いを踏襲するが、着手前にユーザーへ一言確認するとよい）。
