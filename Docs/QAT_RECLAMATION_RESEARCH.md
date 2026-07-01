# QAT gemma_root 22.7GB 回収 — 事前調査（次セッションの実装用）

作成: 2026-07-01（de-fork リファクタ完了直後、branch は main へマージ済 `a857a3b`）。
本ドキュメントは **調査結果と実装方針の引き継ぎ**。実装は次セッション（コンテクスト刷新後）。
調査は3並列サブエージェント（web検索＋wheel一次ソース読解、一部は実コードでシミュレーション）による。**憶測でなく wheel rev `00dc53d` の実ソースで裏取り済み**。

---

## TL;DR
- `models/gemma-3-12b-it-qat/`（22.7GB）は、**重みバイトが production 経路で読まれていない**（GGUF Gemma が LM 重みを供給、wheel の `_SkipGemmaLMSDOps` が `language_model.model.*` を全 skip）。
- QAT dir が「構築時に必須」なのは、wheel が **①実トークナイザ群（`tokenizer.model` 等）② `preprocessor_config.json` ③ glob と `safe_open` を通す有効な `model*.safetensors`** を要求するため。**重みの中身は不要**。
- 従って **wheel 構築の書き換えは不要**。**最小 gemma_root ディレクトリを用意して `config.model.gemma_root` を向けるだけ**（＋必要なら `model_path` から shard を外す小改造）で **~18〜22GB 回収**できる見込み。既存の `use_component_files`（monolith→小分け）方式と同型で低リスク。
- **決定的実験は1つだけ**: 最小 gemma_root で1ジョブ生成し、現状22.7GB版との**出力バイト一致（SHA256）**を確認（既存の検証様式 [[VERIFICATION_LOG]] と同じ、baseline は下記）。

---

## 1. 根本原因（wheel rev 00dc53d 実ソース）
`gemma_root` の消費点は **`ltx_pipelines/utils/model_ledger.py:158-169` の2箇所のみ**:
```python
module_ops = module_ops_from_gemma_root(self.gemma_root_path)              # tokenizer.model + preprocessor_config.json を glob
model_folder = find_matching_file(gemma_root, "model*.safetensors").parent  # ①最初の一致で親フォルダ特定
weight_paths = [str(p) for p in model_folder.rglob("*.safetensors")]        # ②フォルダ内全 *.safetensors をパス列挙
self.text_encoder_builder = Builder(model_path=(checkpoint, *weight_paths), ...)
```
- **config は読まれない**: Gemma3 config は wheel 内蔵ハードコード `GEMMA3_CONFIG_FOR_LTX`（`ltx_core/text_encoders/gemma/config.py`、`encoder_configurator.py:28-30`）。gemma_root の `config.json` は不使用。
- **tokenizer / preprocessor は実ファイル必須**: `base_encoder.py:202-204` が `find_matching_file(gemma_root, "tokenizer.model")` と `"preprocessor_config.json"` を要求 → `tokenizer.py:18` `AutoTokenizer.from_pretrained(..., local_files_only=True)` ＆ `AutoImageProcessor.from_pretrained(...)`。**GGUFからは供給不可**（下記§4）。
- **重みシャードは「存在」だけ必要・バイトは非read**: `find_matching_file` は内容非検証（`utils.py:55-62`）。後段 `sft_loader.py:29-35` が **各 shard を `safe_open` で開いてキー列挙**するが、`_SkipGemmaLMSDOps.apply_to_key`（`engine/gemma/gguf_quant_service.py:191-221`）が `language_model.model.*` に `None` を返し `get_tensor` を skip（＝24GB materialize は既に回避済）。**ただし `safe_open` が開くので、shard は「有効な safetensors」である必要（ゼロバイト不可）**。
- **順序**: glob は `ModelLedger.__init__`（＝`DistilledPipeline(...)` 構築、`engine/pipeline/fast_video_pipeline.py:142`）で走る。我々の GGUF install（同 `:196` → `gguf_quant_service.py:1056` の `dc_replace(model_path=...)`）は **builder 生成後**。よって **glob は必ず先に走る**＝gemma_root に一致ファイルが物理存在しないと構築時 `FileNotFoundError`（dir 退避で `tokenizer.model` 欠落 → FileNotFoundError の既知事象と一致、rename test で実証）。

## 2. gemma_root の最小必須集合（確定）
**必ず実物が要る（小物・合計 ~40MB）**:
- `tokenizer.model`(4.7MB), `tokenizer.json`(33MB), `tokenizer_config.json`(1.2MB), `special_tokens_map.json`, `added_tokens.json`, `chat_template.json` … AutoTokenizer 用
- `preprocessor_config.json`(570B), `processor_config.json` … AutoImageProcessor 用
- （`config.json` は wheel 内蔵ゆえ不要だが、残しても無害・数KB）

**`model*.safetensors` は「有効ファイルの存在」が必要、中身の扱いが唯一の論点**:
- glob(`find_matching_file`) は最初の1個で親フォルダを取るだけ。`rglob` が全 `*.safetensors` を `model_path` に列挙 → `safe_open` で全部開く。
- **shard #2〜#5（~19.4GB）**: 中身は全て `language_model.model.*`＝全 skip（実コードシミュレーションで read=0 確認）。→ **中身不要**。
- **shard #1（4.98GB）**: `vision_tower.*`(437)＋`multi_modal_projector.*`(2)＝**439テンソルが実際に read される**（skip対象外、`model.model.vision_tower.*` 等にマップ）＋ `embed_tokens.weight` の**ヘッダ（shape）**が `_read_target_vocab_from_header`（`gguf_quant_service.py:224-243`）で読まれる。

## 3. ★唯一の未確定点＝回収量を決める分岐（要 実機A/B 1回）
**vision_tower / multi_modal_projector（shard#1 の survivor）は T2V/最小I2V で実際に使われるか？**
- 状況証拠は「**未使用**」寄り: web証言「LTX-2 は vision tower 重みを使わない」、既存メモリ [[scaleup-16gb-research]]/VERIFICATION_LOG §9「24GB qat は vision_tower/multi_modal_projector 専用で T2V/最小I2V では未使用の公算」。
- ただし R1 は「構築時に**読み込まれ model に載る**」ことを実証（使用/未使用は別問題）。
- **決着法**: 下記いずれかの gemma_root で生成し **SHA256 が baseline と一致するか**を見る。一致すれば survivor 不要＝より小さく回収可。

## 4. 却下した選択肢
- **(b) tokenizer/config を GGUF から供給**: **不可**。GGUF はトークナイザを内蔵するが Gemma では transformers が **Unigram で誤構築（正: BPE）**＆ **gemma3 GGUF 直ロード未サポート**（transformers [#41494](https://github.com/huggingface/transformers/issues/41494), [#37002](https://github.com/huggingface/transformers/issues/37002)）。実 `tokenizer.model` が常に必須。config だけは wheel 内蔵で既に不要化済。
- **(c) GGUF時に text_encoder builder 構築を skip**: ltx_core に専用APIなし。ModelLedger 構築を丸ごとバイパスする必要＝**高侵襲・上流追従が重い**。非推奨。

## 5. 推奨アプローチ（データ準備主体・低リスク）
**gemma_root を「小物(~40MB) ＋ 目的別の縮小 `model*.safetensors`」の軽量ディレクトリに置換**し、`config.yaml` の `gemma_root` をそこへ向ける。侵襲度が段階的に上がる3変種:

| 変種 | gemma_root サイズ | 内容 | 回収 | リスク/検証 |
|---|---|---|---|---|
| **V1 保守** | ~5GB | 小物 + **shard#1 実体を温存** + shard#2〜5 を「有効な空safetensors(ヘッダのみ)」に置換 | ~19.4GB | 最安全。survivor 保持ゆえ出力不変が理論保証。空safetensorsが `safe_open` を通るか1回確認 |
| **V2 中間** | ~1.5〜2GB | 小物 + **survivor(vision/mm_projector)＋embed_tokens を抽出した1ファイル**（既存 component-files 抽出と同型） + 空stub | ~20〜21GB | survivor 抽出の正しさを byte一致で確認。embed_tokens shape(§2)を含める必要 |
| **V3 最小** | ~40MB(+embed?) | 小物 + **全キー `language_model.model.*` の極小 stub**（survivor を捨てる） | ~最大22.7GB | **survivor が本当に未使用の場合のみ**成立（§3 の実験で判定）。embed_tokens header 要件の回避も要検証 |

- **実装レバー**: 既に `use_component_files` 経路で `dc_replace(builder, model_path=new_model_path, ...)`（`gguf_quant_service.py:1138-1143`）を実行済。ただし `model_path` **初期値**は `__init__` の `rglob` で shard を含む＝glob(`find_matching_file`) は避けられない。→ **gemma_root には最低1つの有効 `model*.safetensors` を必ず置く**。加えて shard を model_path から外したいなら install 前の builder.model_path 書換で対応可（コード側完結、ファイル改変不要）。
- **embed_tokens ヘッダ**（§2）: `_read_target_vocab_from_header` が shape を読む。stub に正しい shape の embed_tokens を含めると ~2GB。回避可否（GGUF から vocab を取る／読み飛ばす）は要調査（V3 の前提）。→ **まず V1（確実）で ~19.4GB 回収し、V2/V3 は追加最適化**が堅実。

## 6. 次セッションの実装ステップ（案）
1. baseline 再確認（現状 22.7GB版）: 512×320/49f/8steps/seed=12345/T2V, prompt `a calm ocean wave rolling onto a sandy beach at sunset, cinematic` → SHA256 `23844b4e…6bb7bf`（[[engine-firstparty-refactor]] 参照）。
2. **V1 を作成**: 新 dir（例 `models/gemma-3-12b-it-min/`）に小物をコピー、shard#1 を hardlink/コピー、shard#2〜5 を「有効な空 safetensors（0テンソル or `language_model.model.*` の極小ダミー）」で生成（`safetensors.torch.save_file({}, ...)` 等）。`config.yaml: gemma_root` を向ける。
3. **実機 A/B**: 1ジョブ生成 → SHA256 が baseline 一致を確認（＋ /status・metadata 不変・OOM無し）。一致すれば V1 確定＝~19.4GB 回収。
4. （任意）V2/V3 へ: survivor 抽出 or 全stub化を試し、それぞれ byte一致を確認して回収量を最大化。§3 の「survivor 未使用か」もこの過程で判明。
5. 22.7GB の元 QAT dir は **V系が byte一致で通ってから**物理削除（models/ は git外・復旧は再DL、premise 3 バックアップ前提）。

## 7. 参照
**wheel 一次ソース**（`.venv-engine/Lib/site-packages/`）:
- `ltx_pipelines/utils/model_ledger.py:158-169`（gemma_root 2消費点・shard glob/列挙）
- `ltx_core/text_encoders/gemma/encoders/base_encoder.py:202-227`（`module_ops_from_gemma_root`）
- `ltx_core/text_encoders/gemma/tokenizer.py:18`（AutoTokenizer local_files_only）
- `ltx_core/text_encoders/gemma/encoders/encoder_configurator.py:28-30,100-131`（config内蔵・AV key-ops）
- `ltx_core/text_encoders/gemma/config.py`（`GEMMA3_CONFIG_FOR_LTX`）
- `ltx_core/utils.py:55-62`（`find_matching_file`＝内容非検証）
- `ltx_core/loader/sft_loader.py:29-35`（全shard `safe_open`＋`expected_name is None: continue`）
- `ltx_core/loader/single_gpu_model_builder.py:86-99`（`strict=False`・meta残留）

**our engine/config**:
- `engine/gemma/gguf_quant_service.py`（`_SkipGemmaLMSDOps` L191-221、`_read_target_vocab_from_header` L224-243、`dc_replace(model_path=...)` L1138-1143、survivor=vision/mm_projector は QAT由来と明記 L37-43,171-188）
- `services/ltx_runner.py:160-186,482-486`（gemma_root ゲート＝存在必須・重み非読込の注記）
- `config.py`（gemma_root 注記）/ `config.yaml:29-32`（`gemma_root: ./models/gemma-3-12b-it-qat`）
- 実dir `models/gemma-3-12b-it-qat/`（shard#1=4.98GB に vision/mm_projector 全部・embed header／#2〜5=~19.4GB は language_model のみ・非read／小物 ~40MB）

**外部**:
- transformers [#41494](https://github.com/huggingface/transformers/issues/41494)（Gemma GGUF tokenizer 誤構築）, [#37002](https://github.com/huggingface/transformers/issues/37002)（gemma3 GGUF 未サポート）
- [ComfyUI-LTXVideo gemma_encoder.py](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/gemma_encoder.py)（config同梱・tokenizer別ロード）, [#455](https://github.com/Lightricks/ComfyUI-LTXVideo/issues/455)（実tokenizer必須）
- [QuantStack/LTX-2-GGUF README](https://huggingface.co/QuantStack/LTX-2-GGUF/blob/main/README.md)（GGUF＝tokenizer/VAE/connector別途必須）
- [Lightricks/LTX-2 #106](https://github.com/Lightricks/LTX-2/issues/106)（単一ファイルGemma推奨）, [#303](https://github.com/Lightricks/LTX-2/issues/303)（量子化Gemma公式未実装）
- HF: [google/gemma-3-12b-it-qat-q4_0-unquantized/tree/main](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/tree/main)（非重み37.5MB／重み22.70GiB）, [huggingface_hub download](https://huggingface.co/docs/huggingface_hub/en/guides/download)（`allow_patterns` で小物のみDL）
</content>
