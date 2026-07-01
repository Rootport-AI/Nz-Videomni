# QAT gemma_root 22.7GB 回収 — 事前調査（次セッションの実装用）

作成: 2026-07-01（de-fork リファクタ完了直後、branch は main へマージ済 `a857a3b`）。
本ドキュメントは **調査結果と実装方針の引き継ぎ**。実装は次セッション（コンテクスト刷新後）。
調査は3並列サブエージェント（web検索＋wheel一次ソース読解、一部は実コードでシミュレーション）による。**憶測でなく wheel rev `00dc53d` の実ソースで裏取り済み**。

---

## TL;DR
- `models/gemma-3-12b-it-qat/`（22.7GB）は、**重みバイトが production 経路で読まれていない**（GGUF Gemma が LM 重みを供給、wheel の `_SkipGemmaLMSDOps` が `language_model.model.*` を全 skip）。
- QAT dir が「構築時に必須」なのは、wheel が **①実トークナイザ群（GGUFからは供給不可）② `preprocessor_config.json` ③ 有効な `model*.safetensors`**（glob＋safe_open で触られる）を要求するため。ただし shard から**実際に読まれるのは shard#1 の vision_tower/multi_modal_projector＋embed_tokens ヘッダのみ**、残り(~18.1GiB / ~19.4GB)は非read。
- **本リファクタの目的は「スパゲッティ/設計の不合理の解消・保守性」であって"動けばいい"ではない**（それなら旧コードでも動く）。よって**要求の根を断つ案Bを本筋**とする。
- **本筋＝案B: loader を小パッチし、重みファイル要求そのものを消す**（既存の wheel monkeypatch＝denoise と同流儀）。gemma_root は tokenizer のみ ~40MB、偽ファイルも死蔵ロードも無し、コードが「LM=GGUF/vision=不要/tokenizerだけ要る」と正直に表現される＝保守性向上。**~22.7GB回収**。確認は「vision_tower 未使用か」を生成1回で判定するだけ（未使用の公算大）。
- **保険＝案A: 実際に読まれるテンソルだけ本物の小ファイルに抽出**（component-files 同型・byte一致構造保証・コード変更なし）→ gemma_root ~3GB・~19.7GB回収。案Bの確認で万一 vision_tower が使われていた/patchが脆い場合の fallback。使わない vision_tower を温存する冗長さが残り設計的には案Bに劣る。
- 「空/ダミー safetensors で読んだフリ」は**不採用**（偽ファイル＝本末転倒）。

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
- **shard #2〜#5（~18.1GiB / ~19.4GB）**: 中身は全て `language_model.model.*`＝全 skip（実コードシミュレーションで read=0 確認）。→ **中身不要**。
- **shard #1（~4.6GiB / 4.98GB）**: `vision_tower.*`(437)＋`multi_modal_projector.*`(2)＝**439テンソルが実際に read される**（skip対象外、`model.model.vision_tower.*` 等にマップ）＋ `embed_tokens.weight` の**ヘッダ（shape）**が `_read_target_vocab_from_header`（`gguf_quant_service.py:224-243`）で読まれる。

## 3. ★唯一の未確定点＝回収量を決める分岐（要 実機A/B 1回）
> **※用語の別次元に注意**: §2 の「shard#1 の vision_tower/multi_modal_projector が read される」は、構築時に
> `safe_open`＋`get_tensor` で **model へ物理ロードされる**の意（＝メモリに載る）。本節（§3）の「未使用の公算」は、
> その載ったテンソルが **T2V/最小I2V の出力に寄与するか（機能的使用）**という**別次元**の話。案B の byte一致テストが
> 判定するのは後者（機能的に使われているか＝出力が変わるか）であって、前者（構築時 read の有無）ではない。

**vision_tower / multi_modal_projector（shard#1 の survivor）は T2V/最小I2V で実際に使われるか？**
- 状況証拠は「**未使用**」寄り: web証言「LTX-2 は vision tower 重みを使わない」、既存メモリ [[scaleup-16gb-research]]/VERIFICATION_LOG §9「24GB qat は vision_tower/multi_modal_projector 専用で T2V/最小I2V では未使用の公算」。
- ただし R1 は「構築時に**読み込まれ model に載る**」ことを実証（使用/未使用は別問題）。
- **決着法**: 下記いずれかの gemma_root で生成し **SHA256 が baseline と一致するか**を見る。一致すれば survivor 不要＝より小さく回収可。

## 4. 却下した選択肢
- **(b) tokenizer/config を GGUF から供給**: **不可**。GGUF はトークナイザを内蔵するが Gemma では transformers が **Unigram で誤構築（正: BPE）**＆ **gemma3 GGUF 直ロード未サポート**（transformers [#41494](https://github.com/huggingface/transformers/issues/41494), [#37002](https://github.com/huggingface/transformers/issues/37002)）。実 `tokenizer.model` が常に必須。config だけは wheel 内蔵で既に不要化済。
- **(c) GGUF時に text_encoder builder 構築を skip**: ltx_core に専用APIなし。ModelLedger 構築を丸ごとバイパスする必要＝**高侵襲・上流追従が重い**。非推奨。

## 5. 推奨アプローチ（本筋＝案B・保険＝案A）
本リファクタの目的は設計の合理性・保守性の回復。"動けばいい"なら旧スパゲッティでも動く。よって**要求の根を断つ案Bを本筋**とし、案Aは fallback。（「空/ダミー safetensors で読んだフリ」は偽ファイルゆえ不採用。）

### 案B（本筋）— loader を小パッチし、重みファイル要求そのものを消す
**根本原因**: wheel が GGUF 前提でないため、①構築時に重みファイルの存在を要求し、②使わない vision_tower を読み込む。本プロジェクトは既に wheel を monkeypatch 済（denoise の gc/empty_cache）＝レシピに手を入れるのは前例ある手法。同流儀で **GGUF Gemma 時に Gemma builder の重み glob/ロードをバイパス**する。
- 結果、gemma_root は **tokenizer群 ~40MB だけ**で成立（偽ファイルも重みも無し）。config が「実体の無いモデルdir」を指す不自然さも消え、コードが「LM=GGUF／vision=不要／tokenizerだけ要る」と**正直に表現される＝保守性向上**（＝本リファクタの本旨）。
- 実装で要るもの: (i) 重み glob（`find_matching_file(model*.safetensors)`）を GGUF時に回避、(ii) vocab shape（embed_tokens ヘッダ, §2）を GGUF or 既知定数(vocab=262208)から供給し `_read_target_vocab_from_header` 依存を外す、(iii) vision_tower/mm_projector を読み込まない（meta 残置）。**patch はできるだけ我々の `_install_gemma_gguf` 境界に局所化**し、共有ユーティリティ（`find_matching_file`）の広域 patch は避ける（builder を我々が用意 or gemma builder 構築を局所介入）。
- **確認1回**: 生成 → baseline と byte一致（＋vision欠落でcrashしない）→ **vision_tower 未使用が確定＝~22.7GB 回収**。不一致なら vision 使用→案Aへ。
- **正直な代償**: wheel 内部（rev 00dc53d に pin 済）への monkeypatch ゆえ上流更新時の追随点が増える。ただし既存 denoise patch と同種＆pin 済で、意図の明確な小 patch は「不要な3GB＋中身の無いモデルdir」を残すより保守的に健全。

### 案A（保険）— 実際に読まれるテンソルだけ本物の小ファイルに抽出
案Bの確認で vision_tower が使われていた場合、または wheel patch が想定外に脆い場合の **fallback**。wheel が読む `vision_tower.*`＋`multi_modal_projector.*`＋`embed_tokens.weight` を1つの本物 safetensors に抽出（46GB monolith→VAE/text_projection 抽出と同型の component-files 手法・偽物でない）。gemma_root ~3GB・~19.7GB回収・コード変更なし・byte一致は構造保証。ただし**使わない vision_tower を温存する冗長さが残り、設計的には案Bに劣る**。

## 6. 次セッションの実装ステップ（本筋＝案B）
1. baseline 再確認: 512×320/49f/8steps/seed=12345/T2V, prompt `a calm ocean wave rolling onto a sandy beach at sunset, cinematic` → SHA256 `23844b4e…6bb7bf`（[[engine-firstparty-refactor]]）。
2. **重み要求のバイパス実装**（`engine/gemma/gguf_quant_service.py` / `engine/pipeline/fast_video_pipeline.py` 近辺・我々の境界に局所化）: GGUF Gemma 時に (i) `find_matching_file(gemma_root,"model*.safetensors")` の glob を回避（gemma builder を我々が用意 or 該当 glob を局所介入）、(ii) vocab shape を GGUF/既知定数(262208)から供給して `_read_target_vocab_from_header` の shard 依存を外す、(iii) vision_tower/mm_projector を読み込まない（meta 残置・`_return_model` の meta 許容に乗る）。
3. gemma_root を **tokenizer群＋`preprocessor_config.json` だけの ~40MB dir**（例 `models/gemma-3-12b-it-tokenizer/`）にし `config.yaml: gemma_root` を向ける。
4. **実機検証**: 生成 → SHA256 が baseline 一致（＋/status・metadata 不変・OOM無し・vision欠落でcrashしない）を確認。一致＝vision_tower 未使用が確定。
5. 一致後、元 QAT dir(22.7GB) 物理削除＝**~22.7GB 回収**（models/ は git外・復旧は再DL、premise 3 バックアップ前提）。
6. **不一致だった場合のみ案A へ切替**（vision_tower 等を本物抽出した ~3GB gemma_root、~19.7GB回収）＝設計的には劣るが安全網。

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
- 実dir `models/gemma-3-12b-it-qat/`（shard#1=~4.6GiB(4.98GB) に vision/mm_projector 全部・embed header／#2〜5=~18.1GiB(~19.4GB) は language_model のみ・非read／小物 ~40MB。合計 22.7GiB）

**外部**:
- transformers [#41494](https://github.com/huggingface/transformers/issues/41494)（Gemma GGUF tokenizer 誤構築）, [#37002](https://github.com/huggingface/transformers/issues/37002)（gemma3 GGUF 未サポート）
- [ComfyUI-LTXVideo gemma_encoder.py](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/gemma_encoder.py)（config同梱・tokenizer別ロード）, [#455](https://github.com/Lightricks/ComfyUI-LTXVideo/issues/455)（実tokenizer必須）
- [QuantStack/LTX-2-GGUF README](https://huggingface.co/QuantStack/LTX-2-GGUF/blob/main/README.md)（GGUF＝tokenizer/VAE/connector別途必須）
- [Lightricks/LTX-2 #106](https://github.com/Lightricks/LTX-2/issues/106)（単一ファイルGemma推奨）, [#303](https://github.com/Lightricks/LTX-2/issues/303)（量子化Gemma公式未実装）
- HF: [google/gemma-3-12b-it-qat-q4_0-unquantized/tree/main](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/tree/main)（非重み37.5MB／重み22.70GiB）, [huggingface_hub download](https://huggingface.co/docs/huggingface_hub/en/guides/download)（`allow_patterns` で小物のみDL）
</content>
