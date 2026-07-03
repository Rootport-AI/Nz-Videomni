# IC-LoRA Phase B ワークオーダー — forward時GPU LoRA適用（per-layer-quant経路）＋API露出

- 作成: 2026-07-03（branch `feature/ic-lora-phase-b`・base = main merge `a578c83`）
- 併読: [`IC_LORA_PHASE_A_STATUS.md`](IC_LORA_PHASE_A_STATUS.md)（Phase A成果）／[`VERIFICATION_LOG.md` §20](VERIFICATION_LOG.md)（Phase Aゲート数値）／[`NEXT_SESSION_HANDOFF.md`](NEXT_SESSION_HANDOFF.md)
- ユーザー決定（2026-07-03・本セッション）: 機構=**① per-layer-quant経路＋forward時GPU適用**／周辺スコープ=**API露出のみ**（keep-resident整合・oracle照合・x4/他アダプタは今回外）／loras=**generate毎**／参照動画=**POST /upload/video 新設**／**bf16融合経路は温存**（確定事項）

---

## 0. リサーチ結論（設計根拠・実装前に読む）

### ComfyUIエコシステムの実証パターン（Web調査・ソース確認済み）
- **ComfyUI-GGUF（city96）は「dequant時ウェイトパッチ」方式**: `GGMLLayer.get_weight()` が量子化テンソルをdequant→ `comfy.lora.calculate_weight` でfp32のdelta（`strength·(alpha/rank)·B@A`）を加算→forwardごとに再計算・キャッシュなし。量子化バイトは決して変異させない。
- LoRA delta は **fp32 intermediate** で計算し weight dtype へ1回だけキャスト（我々の Phase A `016f442` fp32 fuse 判断と同型）。
- 複数LoRA＝patchesリストの逐次加算で自然に合成。off＝patchesを空にするだけで厳密・可逆。
- 事前fuse/再量子化・deltaキャッシュはメインラインでは不使用（不可逆な量子化誤差・トグル喪失のため意図的に回避）。
- side-path方式（出力側に低ランク項を加算）は**不採用**: per-layer dequant経路はどのみち毎forwardで全重みを実体化するためVRAM利得ゼロ、かつ下記G2のbyte-match検証レバーを失う。

### 自エンジンのフック点（コード読解済み・file:line）
- **挿入点＝`engine/gguf/quant_service.py:625-643` `ggml_linear_forward`**。`types.MethodType`で各`nn.Linear`インスタンスにバインドされる（`_patch_linear_for_ggml_dequant` :612・`_patch_model_for_ggml_dequant` :648・ModuleOps経由 :658, :687-745）。カスタムLinearサブクラスは存在しない（クラスは素の`nn.Linear`・`m.forward`差し替え＋`m.weight`が`GGMLQuantizedTensor`バッファ）。
- **block-swapとの整合**: `BlockSwapService._patch_block`（`engine/transformer/block_swap_service.py:153`）はブロックを`.to()`移動するだけでモジュール差し替えはしない→ **Linearインスタンスに載せたA/Bバッファとforwardパッチはswapを自動で生き残る**。注意: swapped_forwardは`original_forward`を直接呼ぶ（:216）ためブロックレベルのhookはスキップされる（Linearレベルなら発火）。
- **Phase A流用部品**（`engine/gguf/loader_service.py`）: safetensorsロード＋`LTXV_LORA_COMFY_RENAMING_MAP`（:223-240）・lora_A/lora_Bプレフィックスペアリング＋shapeチェック（:248-278）はそのまま流用。**in-place `weight.add_`（:279）だけが差し替え対象**（bf16融合経路では現状のまま温存）。
- **差し替えるガード**: `engine/pipeline/fast_video_pipeline.py:363-368` の「`per_layer_quant=True`＋`ic_loras`→raise」を新機構のインストールに置換。`per_layer_quant=False`分岐（`GGUFLoaderService`＋`_fuse_ic_loras`）は無傷で温存＝bf16経路の選択可能性維持。
- **参照動画条件付けは機構非依存**: `_reference_conditioning_for_stage`（`fast_video_pipeline.py:490-546`・stage1のみ・hybrid-conditioning monkeypatch :706-721 経由）はfuse機構に触れないためPhase Bでも不変。ただし現状`_ic_reference`は`_ic_loras`非空を要求（:107-112・メタデータ読取のため）。
- **keep-resident汚染問題は構造的に消滅**: dequant結果（`bf16`テンソル）は毎回新規生成・使い捨て（:638-639, `del bf16`）。deltaをここに加算する限り`StateDictRegistry`のキャッシュは不変。
- **注意（api_types.py）**: `engine/api_types.py`の既存`IcLora*`クラス群（:242-366）は**別物のレガシーcanny/depthスキーマで未配線**。流用しない。配線テンプレートは`conditioning_images`（Phase 3 スライス1）。

## 1. スコープ

### Stage 1: エンジン機構（forward時ウェイトパッチ）
1. `ggml_linear_forward`内、dequant直後にLoRA delta加算:
   ```python
   bf16 = dequantize_ggml_tensor(...)          # 既存（毎回新規テンソル）
   if <このLinearにLoRAあり>:
       bf16 += (strength * B.float() @ A.float()).to(bf16.dtype)   # fp32計算→1回キャスト→fresh tensorへin-place加算
   result = torch.nn.functional.linear(x, bf16, self.bias)          # 既存
   ```
   - delta計算はPhase A `_fuse_ic_loras`と**同一の式・同一の演算順序**（`torch.matmul(B.float()*strength, A.float())`→`.to(weight.dtype)`→加算）にする。G2のbyte-match狙いのため勝手に最適化しない。
   - LoRA無しのLinearは既存コードパスと**完全同一**（分岐追加のみ・G1保証）。
2. A/B因子は対象`nn.Linear`にbuffer登録（block-swapの`.to()`に自動相乗り・計~654MB bf16が480層に分散＝ブロックあたり~14MB）。
3. キーマッチング: `_fuse_ic_loras`のロード＋ペアリング部を共有ヘルパへ抽出し、state-dictキー(prefix)→モジュール名の対応でLinearへattach。マッチ0件は大声WARN（既存踏襲）。
4. **非量子化（float weight）Linearに対象キーが載る場合**: 保存済みweightのin-place変異は禁止（キャッシュ汚染）。out-of-place（`w + delta`）で対応。該当キーが実在するかはPhase Aのマッチログで確認（おそらく全480キーは量子化Linear）。
5. generate毎の切替: attach/detachをジョブ単位で行う（keep_resident=0では毎ジョブtransformer再構築なので、build後のattachステップとして自然に実装。`fast_video_pipeline.py:469-478`の`patched_transformer`クロージャ内が候補）。detach忘れ＝次ジョブ汚染がないことをG3で検証。
6. `ic_loras`/`ic_reference`をcreate時引数からgenerate時引数へ拡張（後方互換: create時指定も温存可）。

### Stage 2: API露出
1. `POST /api/v1/upload/video` 新設（既存 upload/image と同型・video_id返却・mp4等の最小バリデーション）。spec §5.2の既定路線。
2. `GenerateRequest`（`api/models.py`）に**optional・default付き**フィールドを追加（凍結契約はADDのみ許容・`conditioning_images`/`crop_output`の前例踏襲）:
   - `loras: list[LoraSpec] = []`（LoraSpec = アダプタ識別子＋strength。識別子はサーバー側`models/ltx-2.3-ic-lora/`配下の登録名で解決・生パスは受けない）
   - `reference_video_id: str | None = None`（upload/videoで得たID）
   - バリデーション: loras指定時はreference_video_id必須（Pixel-Spatial-Upscalerは参照必須）。÷64・8n+1等の既存バリデータは不変。
3. 配線ホップ（テンプレ=conditioning_images）: `api/generate.py:21-31`（変更不要・request丸ごと流れる）→ `services/pipeline_manager.py:125-140`（video_id→パス解決）→ `services/ltx_runner.py:756-849`（payloadに`loras`/`reference_video`追加 ~:806）→ `@@LTX@@` generate op → `engine/worker.py:173-204` `_do_generate`（parse→pipeline.generateへ）。
4. mock backend対応＋pytest追加（loras付きリクエストの受理・バリデーション・mock経路疎通）。
5. `GET /status`の`vram_optimization`（凍結キー）・`metadata.json`スキーマは不変。metadataへのloras記録は追加フィールドとしてのみ可。

## 2. 検証ゲート（客観・全てPASSでStage完了）

| # | 内容 | 期待 |
|---|---|---|
| G1 | 回帰byte-match（LoRA off・本番per-layer経路） | T2V `23844b4e…6bb7bf`／I2V `a511eda4…c217` SHA完全一致・peak_vram 8440不変・pytest全green |
| G2 | **新経路 vs Phase A spike出力**（同seed・1024×640/25f・x2 LoRA＋参照条件付け） | SHA一致が理想（Phase Aゲート2で両経路はLoRA無しbyte一致・delta式も同一のため）。不一致なら差分原因を特定してから判断（勝手に「近いからOK」としない） |
| G3 | 非汚染トグル（LoRAあり→なし→あり を同一ワーカーで連続） | 「なし」ジョブがbase出力とSHA一致（G1と同値）＝attach/detach漏れなし |
| G4 | VRAM/速度 | per-layer経路のpeak_vramデルタ僅少（A/B分散+一時delta）・生成時間デルタをログ（rank64は理論<1%） |
| G5 | API e2e | upload/video→loras付きgenerate→完走（mock pytest＋実機スモーク1本） |

- G2/G3/G4は実GPU実行。Phase Aハーネス`outputs/ic_lora_phaseA/run_spike.py`（untracked・モードparity/base/spike/toggle）を改修流用してよい。
- Phase A基準値: spike実行=1024×640/25f・VRAM溢れなし・トークン+25%／bf16経路はper-layer比で約3倍遅＋VRAM+3.6GB（新経路はこのペナルティが消えるはず＝それ自体が成果指標）。

## 3. やらないこと（スコープ外・Phase C以降）
- keep_resident=1でのLoRAトグル設計（今回はkeep=0前提。ただし機構自体は非汚染なので将来対応の障害はない）
- wheelの`ICLoraPipeline` oracle照合（stage1のみ適用 vs 両ステージ適用の挙動差は文書化済みのまま）
- x4バリアント・In-Outpainting/Deblur等の他アダプタ
- Gradio UI露出（APIのみ。UIはPhase 1残(a)と合流させる）

## 4. 運用注意
- サブエージェントはOpus以下（Fable5禁止）。実験前に仮説→裏取り。異常検知時は続行せず報告。
- システムPython不可触・全て`./.venv`（app）と`./.venv-engine`（engine）内で。
- 実GPU実行はワーカーログ`logs/ltx_worker.log`のpeak_vramを一次ソースとする。
- 目視題材は「賑やかな町＋セリフ」系（波・静的部屋NG）— ただし本Phase BのゲートはSHA/メトリクス主体で、目視はfix-later方針を踏襲。
