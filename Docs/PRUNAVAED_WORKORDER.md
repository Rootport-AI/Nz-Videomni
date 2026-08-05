# PrunaVAED（枝刈り版・映像VAEデコーダ）実装ワークオーダー

> **【ステータス 2026-08-05】実装完了・実機ゲート G1〜G7 全項目合格・G6 の三段判定は①採用。** 実測と検証記録の正本は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) **§52**（本書は仕様・設計判断の正本として残す）。要点だけ挙げると——**G1**（変換ツール）: 変換ツールの pytest 89件〔75 PASS／14 skip〕・実重みの変換で `PrunaVAED-decoder-bf16.safetensors` 690,047,968バイト／102テンソル／**パラメータ 345,006,256**（上流の報告値と1個の違いも無く一致）・自己検証8項目全 PASS（**全102テンソルの MD5 全単射照合**を含む）・2回実行でバイト単位同一。**G2**（実重みの読み込み）: 6項目全 PASS——未初期化パラメータの警告ゼロ・`missing_keys` / `unexpected_keys` ともに空・射影 resnet の `norm3` が2つとも `ChannelLayerNorm3d`。**G3**（数値の健全性）: PSNR 36.06dB・NaN/Inf ゼロ（追加で SSIM も計測し、輝度 0.9854／RGB 平均 0.9712。§52.5-b）。**G4**（バッチ再読み込みなし）: 増分はジョブ1本目のみで以後横ばい、切替でディスク読み直しなし（ただし増分の大きさが見積りを超える未解明の観測を §52.6 に記録した）。**G5**（降格）: 重みを退避 → `"on->off"` で完走、戻すと**ワーカー再起動なしで** `"on"` へ復帰。**G6**（採否判定）: 4ペアの交互対比較で合計生成時間が **平均 −12.54秒（−10.52%）**・4ペアすべて正・デコード区間 1.608倍・`peak_vram_reserved_mb` −2,635MB、既定側の出力は §50.4 J1 と SHA-256 完全一致（＝off 時の非退行のビット証明）。**G7**（オーナー目視）: 同一シードの A/B をオーナーが視聴し「劣化は肉眼ではほとんど分からない」と判定＝**合格**。
>
> **G9（MCP サーバーへの `vae_mode` 公開）も同日に合格した**（オーナー確定事項 0-8 のとおり G1〜G7 合格後の最終ステップとして実施。§6.3・§9 G9）。`submit_generate` / `submit_chain` の両方に列挙値2つで現れること・実叩きで `"on"` / 省略時 `"off"` になること・ツール本数22が不変であることを実サーバーの `tools/list` と実ジョブ2本で確認済み（§52.10-b）。**したがって本テーマの技術的な作業は完了しており、残るのはフロントエンドの再ビルド／デプロイとコミットだけである。本書の §1〜§12 は着手前の記録としてそのまま残す**——ただし第3版の敵対的レビューが実装前に潰した設計ミス（A-1 の「matcher の無い SDOps は全キーを捨てる」、§2.5 の norm3 非同値）は、実装後の視点でも本テーマ最大の落とし穴だったので、要約を §52.9 に再掲してある。

- 作成: 2026-08-05
- 更新履歴:
  - **2026-08-05（第2版）** — 起票時に未確定として §12 に残していた4件に、オーナーが全件回答した。反映先は §0（確定事項 0-8〜0-11 を追加）・§6.3（MCP サーバーへ**公開する**方針へ転換）・§7（枝刈りデコーダの重みは**必須ダウンロード**で確定）・§9 G6（合格基準を**ジョブ合計生成時間の1秒以上の短縮**へ変更）。§12 は「確認事項」から「オーナー回答済みの記録」へ書き換えた。
  - **2026-08-05（第3版・敵対的レビュー反映）** — 実装プランのオーナー承認後、別エージェントによる敵対的レビューを実施し、全指摘を file:line で照合したうえで監督が重大2件を追認した。**設計そのものが変わった箇所が6件ある**ので、第2版を読んだ状態で着手しないこと。
    - **A-1（致命）**: §4.3 の `SDOps("PRUNAVAED_DECODER_FLAT")` は matcher を1つも持たないため**全キーが捨てられる**（`sd_ops.py:92-97`、`any([])` は `False`）。→ **ジョブ側 SDOps を廃止**し、変換器がモジュール相対の素キーを直接出力、ビルダーは `model_sd_ops=None` で素通しする方式へ変更（§4.1・§4.3）
    - **A-2**: §7 の `Min` しきい値 4.4G は、本書が自ら算出した下限 4,454,420,162 を**下回っていた**。→ **4,500,000,000** へ修正
    - **A-3**: §8 E5 の「パイプライン構築時に検出」は不成立（構築は load オペで1回きり。`fast_video_pipeline.py:287-383`）。→ 重み存在確認を**ジョブ単位**（`_set_vae_mode_job()` 内）へ移動
    - **A-4**: §4.5 の観測は `_keep_resident_used` 型ではなく **`_block_swap_prefetch_used` 型**（`worker.py:549-556`）。降格判定がパイプライン内で起きるため
    - **A-5**: `component_video_vae_pruned_path` には既存 `component_video_vae_path` と同じ**6段配線**が要る。§3 に追加
    - **A-6**: ビルダー生成位置は `_install_component_sources()` 直後ではなく**無条件位置**（`:383`）。コード例の `self._ledger` は誤りで、実体は `self.pipeline.model_ledger`
    - 簡素化（採用）: 保持ビルダーは既定側1本のみ・枝刈り側は都度生成／`_reset_vae_mode_job()` 廃止／G4 の主検証を in-process テストへ／§7 の `Invoke-ModelDownload` 追加は不要
    - 検証の補強: §5.3 の項目5 を**全100テンソルの MD5 総当たり**へ格上げ／G3 の latent は `ledger.video_encoder()` で自作／§4.2 の `__metadata__` に `decoder_blocks` を**含めない**／形状検証は `assert` でなく `raise ValueError`
    - G6 を**オーナー改定の三段判定**（採用／保留／不採用）へ差し替え。ベースラインも §50.4 J1 の **147.60秒**へ訂正
- 由来: フロントエンド台帳 [`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §3-50。**台帳上の項目名は「PruneVAED」だが、これは上流の正式名称の誤記である**（正しくは **PrunaVAED**＝開発元 Pruna AI の名を冠したもの。Hugging Face: `PrunaAI/PrunaVAED`）。本書ではファイル名・本文とも正式名称 **PrunaVAED** に統一する。台帳ファイルそのものは本テーマの着手時に別途更新する（§11）。なお API のフィールド値 `vae_mode="prune_vaed"` は**既存の外部コントラクトなので改名しない**（§0-2）。
- 正本: 本書（仕様・設計判断）／実測値と検証記録＝[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §52（新設）
- 対象リポジトリ: `Nz-LTX23-backend`（本体）＋ `Nz-GGUF-Converter-LTX23`（重み変換ツール）＋ `Nz-LTX23-frontend-AviUtl2`（Settings > Acceleration のパリティ）
- 機能名（内部識別子）: `vae_mode`（既存フィールドを流用。値は `"default"` / `"prune_vaed"`）
- 併読: [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §50.4（映像VAEデコード33.07秒のベースライン実測。本テーマの分母）／同 §51（`fused_gguf_dequant_kernel`＝直近の「モック→実装」転換の前例。配線とゲートの作法はこれを踏襲する。ただし**既定反転は行わない**——本テーマの既定は恒久 OFF である。§0-11）／[`BLOCKSWAP_PREFETCH_WORKORDER.md`](BLOCKSWAP_PREFETCH_WORKORDER.md)（ジョブ単位トグルの配線の原型）／[`ICLORA_DEPTH_DEBLUR_WORKORDER.md`](ICLORA_DEPTH_DEBLUR_WORKORDER.md)（重みの再ホストとインストーラ改修の前例）

---

## 0. オーナー確定事項（最優先・すべてに優先する）

以下は 2026-08-05 にオーナーが確定した事項であり、実装中に本書の他のどの記述と矛盾した場合もこちらを優先する。

| # | 決定 | 補足 |
|---|---|---|
| 0-1 | **ジョブ単位のトグル**として実装する。既存の `vae_mode` フィールド（`"default"` / `"prune_vaed"`、既定 `"default"`）をそのまま使う | sage attention と同じ「設定タブのスイッチ」であり、**サーバー全体のモデル差し替えではない**。既定 OFF は据え置く |
| 0-2 | UI の説明文は「**出力品質がわずかに低下する可能性がある**」とする | オーナー指定の文言。効果を過大に見せない。attention の sage と同じ「絵が変わる」系の注意書きに相当する |
| 0-3 | **バッチでの再読み込み罰則をゼロにする**（ハードゲート） | `keep_resident` を ON にした状態で、既定デコーダと枝刈りデコーダの state_dict が**両方とも** CPU 側の `StateDictRegistry` に同居し、ジョブ間で `vae_mode` を切り替えてもディスクを読み直さないこと |
| 0-4 | 配布は**方式(b)＝事前変換済みのデコーダ単体 safetensors**（約690MB・BF16そのまま・ltx-core native なキー名）を `Rootport/Nz-LTX23-weights` へ再ホストする | 変換スクリプトの置き場所は **`Nz-GGUF-Converter-LTX23` リポジトリ**であって、バックエンドではない |
| 0-5 | バージョンを固定する。**PrunaVAED v2**＝リポジトリ revision `4baacd7ef66a6131439542c1f05872afe042e128`、重みファイル `vae/diffusion_pytorch_model.safetensors` の LFS SHA256 `4cbf0cbe6c185514d62c6c58c35dc42d7ea15924f34391e08be12f44bfccdf1d`、サイズ `1,327,909,418` バイト | v1 と v2 は構造が同一（safetensors ヘッダのJSONがバイト単位で一致）で、値だけが異なる。取り違えを防ぐため上記3点で固定する |
| 0-6 | **GPL-3.0 のコードを持ち込まない**（`ScryptHunter/ComfyUI-PrunaVAED`）。着想の参照のみ | Pruna 配布の `patch_diffusers.py` も Pruna 著作権＋Apache-2.0 の diffusers 由来部分の混合物である。**コピーせず、自前で書き直す** |
| 0-7 | 重みのライセンスは **LTX-2 Community License**（既に再ホスト済みの LTX 系重みと同じ系列）。README と再ホスト先に出所（元リポジトリ・コミット・我々の変換）を明記する | 既存の `LTX23_video_vae_bf16.safetensors` も同ライセンス文を `__metadata__["license"]`（21,393文字）に埋め込んでいる。同じ作法を踏襲する |
| 0-8 | **MCP サーバーへ公開する**（2026-08-05 回答。起票時の推奨案からの方針転換） | オーナー回答「一通り完成したら公開まで進みたい。外出先から操作したいときに便利なので」。ただし**順序を守る**こと——バックエンド実装と実機ゲート G1〜G7 の合格を待ってから、最終ステップとして公開する（§6.3）。`mcp_server/tools/generate.py:7-9` の「意図的に公開しない」というコメントは**撤回対象**である |
| 0-9 | 枝刈りデコーダの重みは **必須ダウンロード**（インストーラが常に取得し、欠けていたら失敗させる）（2026-08-05 回答） | 「入っているか入っていないか分からない」状態を作らない。ただし §4.5 の降格経路（重み欠損時に `vae_mode_used="on->off"` で完走）は**保険として設計に残す**——利用者がファイルを消した場合や、旧環境からの移行時に、生成そのものが止まらないようにするため |
| 0-10 | **G6 は三段判定とする**（2026-08-05 回答＋同日のオーナー改定）。1280x768p の交互対比較で**合計生成時間**と**VAEデコード区間**の両方を測り、①合計の平均短縮 ≥ 1秒 → **採用** ②合計は1秒未満だがデコード区間が上流主張に近い倍率（目安1.5倍以上）→ **保留**（既定OFFのまま残置し、新論点「動画エンコードの高速化は可能か」を台帳へ起票してオーナー判断） ③デコード区間の倍率も出ない → **不採用** | オーナー回答「1280x768pで1秒以上の生成時間短縮が見込めるなら、存在していい機能だと思う」。当初は「区間の倍率は合否に使わない」としていたが、区間が速いのに合計が伸びない場合は**律速がVAEではなく動画エンコード側にある**という別の発見であり、機能を捨てるのは早いという判断で②を足した（§9 G6） |
| 0-11 | 既定値は **恒久 OFF**。後日の既定反転ステップは置かない（2026-08-05 回答） | `block_swap_prefetch` の S4 や `fused_gguf_dequant_kernel` のような「ゲート合格後に既定を ON へ反転する」専用ステップは**設けない**。出力の絵が変わる機能であり、利用者が意図して選ぶべきものだからである |

### 0.1 棄却された設計

| # | 案 | 棄却理由 |
|---|---|---|
| 1 | `.venv-engine` 内の ltx-core wheel を直接書き換える | 再インストールで消える。加えて `vendor/` に置いてあるソースは**実行されていない**（実行されるのは `.venv-engine/Lib/site-packages/ltx_core`）ので、そちらを直しても効かない。wheel は1バイトも触らない方針を維持する |
| 2 | 1.33GB の diffusers 形式ファイルをそのまま配布し、実行時にキー名を読み替える（方式(a)） | (i) 利用者に不要な encoder 分まで二重にダウンロードさせる（encoder は元のものとバイト単位で同一）。(ii) ltx-core の `SDOps` は文字列の `str.replace` の連鎖なので、置換後の文字列が後続ルールに再マッチする事故が起きやすい |
| 3 | 枝刈りされた重みをゼロ埋めして元の形状に戻す | 速度もVRAMも一切改善しない。枝刈りの目的そのものを打ち消す |
| 4 | 既存の `video_vae` モデルレジストリ＋`POST /pipeline/load` によるサーバー全体の差し替え | 切り替え粒度がワーカー再起動になる。オーナー確定事項 0-1（ジョブ単位）に反する |
| 5 | TAEHV（`taeltx2_3.pth`。極小のオートエンコーダ） | 速度は4〜6倍だが品質劣化がはっきり見える別クラスのトレードオフ。本テーマのスコープ外とし、必要になれば台帳に別項目として起票する |
| 6 | 変換後ファイルに `__metadata__["config"]` を埋め込まない | **不可能**。ltx-core のローダーが起動時に必ず読むため、無いと例外で落ちる（§2.4-C。事前想定を覆した確定事実） |

---

## 1. 背景

映像VAEデコード（潜在表現から実際の映像フレームを復元する処理）は、768p/257フレームの標準条件で**実測33.07秒／ジョブ**（7回のタイル分割デコードの合計。[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §50.4 の J1 計測）を占めており、生成側に残った最大の分母である。

Pruna AI が公開した **PrunaVAED** は、LTX-2.3 専用の「枝刈り（pruning＝寄与の小さいチャンネルを削ること）＋蒸留（distillation＝軽くしたモデルに元の出力を教え込むこと）」を施した**デコーダ差し替え品**である。上流のモデルカードによる主張は次のとおり:

- デコード速度 **1.68〜2.08倍**（解像度と尺による。10秒・1080p で最良）
- 品質 **PSNR 39.23〜41.06 / LPIPS 0.0052〜0.0094 / SSIM 0.9811〜0.9876**（PSNR＝ピーク信号対雑音比。値が大きいほど元に近い。40dB前後は「目視でほぼ区別できない」水準）
- ピークVRAM 約50%削減
- VAE 全体のパラメータ数 726.1M → 663.9M（−15%）
- 「元のデコーダとビット単位一致ではない」と明記されている
- エンコーダは変更していない

計測条件は H100 80GB・bfloat16・24fps の4秒クリップであり、我々の実機（RTX 4080相当・16GB）とは異なる。したがって**速度の期待値は実測で置き直す**（§9 G6）。

現状、`vae_mode` は「受理はするがエンジンへ渡していない」モック項目として実装されている（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §43.1 の設計。Acceleration 枠で唯一残ったモック）。本テーマはこれを実機能へ転換する。直前の前例は `fused_gguf_dequant_kernel`（§51、2026-08-04）で、そのときの手順——API・worker・観測フィールド・Gradio・フロントエンドを同時に配線し、実機ゲートで裏を取ってから外へ広げる——をそのまま踏襲する。

ただし前例と2点だけ違う。(i) **既定は恒久 OFF であり、ゲート合格後に既定を ON へ反転するステップは置かない**（オーナー確定事項 0-11）。出力の絵が変わる機能だからである。(ii) **G6 は速度の確認ではなく採否の判定である**（同 0-10）——1280x768p でジョブの合計生成時間が平均1秒以上縮まなければ、この機能は採用しない。

---

## 2. 事前調査の確定事実

以下はすべて 2026-08-05 に実ファイルを読んで確認した事実である。推測は「未確定」と明記した箇所のみ。

### 2.1 バックエンドの既存レール（配線の受け皿）

| 事実 | 根拠 |
|---|---|
| `GenerateRequest.vae_mode` は `Literal["default","prune_vaed"] = "default"`。`Field(...)` も `description` も無い素の宣言 | `api/models.py:210`（説明コメント `:200-209`） |
| `GenerateChainRequest.vae_mode` も同型 | `api/models.py:472`（共通コメント `:456-460`） |
| `to_clip_request()` が各クリップへ転記済み | `api/models.py:734`（理由の docstring `:710-716`） |
| バリデータは無く、`Literal` 型だけで担保されている（不正値は 422） | `api/models.py` 全域に `vae_mode` を見る `field_validator` は無し。テストは `tests/test_validation.py:521-527` |
| **`vae_mode` は意図的にワーカーペイロードへ載せていない**。この配線の穴を塞ぐのが本テーマの核 | `services/ltx_runner.py:1411-1416`（単発）・`:1640-1643`（チェーン） |
| 加算的ペイロードの作法（既定と違うときだけ送る） | `services/ltx_runner.py:1417-1438`（単発）・`:1644-1656`（チェーン） |
| `GenerationOutcome` の `*_used` 群 | `services/ltx_runner.py:251-295`（`attention_used` `:267`／`block_swap_prefetch_used` `:273`／`keep_resident_used` `:282`／`fused_gguf_dequant_kernel_used` `:289`） |
| done イベントからの取り込み | `services/ltx_runner.py:1485-1490`（単発）・`:1689-1693`（チェーン） |
| worker のリゾルバ群 | `engine/worker.py:478`（attention）・`:538-546`（block_swap_prefetch）・`:559-638`（keep_resident）・`:654`（fused dequant） |
| `*_used` の3値規約 `"off"` / `"on"` / `"on->off"` の正典 | `engine/worker.py:641-651`（`_keep_resident_used`。「降格判断が前段で完結するので状態を読まない」型） |
| done イベントの組み立て | `engine/worker.py:792-801`（単発）・`:964-974`（チェーン）。プロトコル記述は `:75-77` |
| metadata.json への転記 | `services/pipeline_manager.py:903-924`（`_write_metadata`）・`:803-808`（`_write_chain_metadata`）。呼び出しは `:866-874` / `:727-742` |
| `GET /status` の acceleration ブロックは3キーのみ（`attention_backends` / `sage_available` / `block_swap_prefetch_available`）。`keep_resident` と `fused_gguf_dequant_kernel` は**意図的に載せていない** | `services/pipeline_manager.py:219-228`、`api/status.py:32-33` |

### 2.2 モデル構築とキャッシュの経路

| 事実 | 根拠 |
|---|---|
| `_LEDGER_BUILDER_ATTRS` は8属性。**ここに載せ忘れると「そのサブモデルだけキャッシュが効かない静かな部分故障」になる**と明記されている | `engine/pipeline/fast_video_pipeline.py:35-44`（警告コメント `:23-34`） |
| `_swap_registry()` は8属性の存在を assert し、`dataclasses.replace(..., registry=new)` で一括差し替えする | 同 `:599-677`（assert `:638-645`、差し替え `:647-657`、docstring の警告 `:624-627`） |
| `_set_keep_resident_job()` は**ジョブ終了時にリセットしない**（キャッシュが次ジョブへ生き残ることが機能そのもの） | 同 `:679-709`（docstring `:680-698`） |
| per-job の呼び出し点 | 同 `:1587-1594`（単発）・`:1731-1735`（チェーン） |
| `_install_component_sources()` が `vae_decoder_builder` を `dataclasses.replace` で作り直す既存前例。**ここに `model_path` と `model_sd_ops` を差し込む作法が既にある** | 同 `:766-771`（docstring `:712-737`） |
| `SingleGPUModelBuilder` は **frozen dataclass**。フィールドは `model_class_configurator` / `model_path` / `model_sd_ops` / `module_ops` / `loras` / `model_loader` / `registry` / `lora_load_device` | `.venv-engine/.../ltx_core/loader/single_gpu_model_builder.py:44-51` |
| `StateDictRegistry` のキャッシュキーは `sha256(解決済みパス群 + sd_ops.name)` | 同 `loader/registry.py:58-64` |
| 同一キーの二重登録は `ValueError` を投げる（`get` で先に確認する規約） | 同 `:66-72` |
| **ledger はジョブ毎にデコーダを新規構築する**（「Models are **not cached**」） | `.venv-engine/.../ltx_pipelines/utils/model_ledger.py:56-60`、`video_decoder()` `:222-228` |
| デコード後にデコーダは解放される（`decode_video` は遅延ジェネレータで、参照を落とすと消える） | `engine/pipeline/chain_pipeline.py:906-908` / `:916-918` / `:1005-1006`、`engine/pipeline/common.py:266-271` |
| **静かな失敗の罠**: `load_state_dict(strict=False, assign=True)`。キーが合わないとパラメータが meta デバイスに残ったまま、`logger.warning("Uninitialized parameters or buffers: ...")` を1行出すだけでモデルが返る | `single_gpu_model_builder.py:98`・`:115`、警告は `:77-84` |
| デコーダの生成点は2箇所。**両方に per-job の選択が届かねばならない** | チェーン: `chain_pipeline.py:870`／単発: `common.py:271`（どちらも `ledger.video_decoder()`） |

### 2.3 チェックポイント構造（実測）

上流 PrunaVAED v2 の `vae/config.json`（revision 固定・§0-5）と、既存の `models/ltx-2.3-components/vae/LTX23_video_vae_bf16.safetensors` の `__metadata__["config"]` を突き合わせて、**枝刈り後のデコーダの構造を完全に確定した**。

既存（無改変）デコーダの構成は `decoder_blocks` 9要素（`__metadata__` に格納された順は encoder 順で、`VideoDecoder.__init__` が `reversed()` して使う。`video_vae.py:637`）。`decoder_base_channels=128` から `feature_channels = base_channels * 8 = 1024`（`video_vae.py:622`）。

**実行順に並べた両者の比較**（幅はブロック通過後の値）:

| 実行順 | 既存（無改変） | PrunaVAED | 一致 |
|---|---|---|---|
| `conv_in` | 128 → 1024 | 128 → 1024 | **ビット一致** |
| `res_x` ×2 | @1024 | @1024 | **ビット一致** |
| `compress_all` m=2 | 1024 → 512（conv `[4096,1024]`） | 同左 | **ビット一致** |
| `res_x` ×2 | @512 | @512 | **ビット一致** |
| — | （無し） | **射影 resnet 512→384** | 新設 |
| `compress_all` m=1 | 512（conv `[4096,512]`） | 384（conv `[3072,384]`） | −25% |
| `res_x` ×4 | @512 | @384 | −25% |
| — | （無し） | **射影 resnet 384→256** | 新設 |
| `compress_time` m=2 | 512 → 256（conv `[512,512]`） | 256 → 128（conv `[256,256]`） | −50% |
| `res_x` ×6 | @256 | @128 | −50% |
| `compress_space` m=2 | 256 → 128（conv `[512,256]`） | 128 → 64（conv `[256,128]`） | −50% |
| `res_x` ×4 | @128 | @64 | −50% |
| `conv_out` | `[48,128,3,3,3]` | `[48,64,3,3,3]` | **再学習済み・値が異なる** |

この削減率（up_blocks.0 が 0%、up_blocks.1 が 25%、up_blocks.2 と up_blocks.3 が 50%）は上流モデルカードの記述と完全に一致する。

**確定した検算**（本書の構造仮説が正しいことの決定的な裏づけ）:

- 上の構成でパラメータ数を積算すると **345,006,256** となり、報告値と**1個の違いも無く一致**した（`per_channel_statistics` の2バッファ計256要素は別勘定）。
- テンソル本数も **100本**（`conv_in` 2＋up_blocks 96＋`conv_out` 2）となり、報告値と一致。`per_channel_statistics` の2本を足して **102本**。
- BF16 で 345,006,256 × 2 バイト ＝ **690,012,512 バイト**。「約690MB」と一致。

`upsample_residual` は上流 config で **全4段とも `false`**（`vae/config.json`）。したがって `DepthToSpaceUpsample(residual=False)` でよく、既存デコーダと同じ設定である（残差モードはパラメータを持たないため、重みファイルからは判別できない＝**明示的に確認する必要があった項目**）。`resnet_norm_eps` は `1e-06`、`timestep_conditioning` は `false`、`decoder_causal` は `false`、`patch_size` は 4、`spatial_padding_mode` は `zeros`——いずれも既存デコーダと同一である。

`latents_mean` / `latents_std`（各 `[128]`）は既存の `per_channel_statistics.mean-of-means` / `std-of-means` と**バイト単位で同一**（MD5照合済み）。エンコーダも元とバイト単位で同一のため配布しない。

### 2.4 ltx-core 側で「できること／できないこと」

**A. アップサンプラは全段そのまま表現できる。**
`_make_decoder_block`（`video_vae.py:469-548`）が持つブロック名は `res_x` / `attn_res_x` / `res_x_y` / `compress_time` / `compress_space` / `compress_all` の6種。`DepthToSpaceUpsample` の出力幅は `prod(stride) * in_channels // out_channels_reduction_factor`（`sampling.py:80`）。枝刈り後の4段を当てはめると:

| 段 | ブロック名 | stride | multiplier | conv 出力 | rearrange 後 |
|---|---|---|---|---|---|
| 1 | `compress_all` | (2,2,2) | 2 | 8×1024÷2 = **4096** ✓ | 512 |
| 2 | `compress_all` | (2,2,2) | 1 | 8×384÷1 = **3072** ✓ | 384 |
| 3 | `compress_time` | (2,1,1) | 2 | 2×256÷2 = **256** ✓ | 128 |
| 4 | `compress_space` | (1,2,2) | 2 | 4×128÷2 = **256** ✓ | 64 |

4段とも実測の重み形状と一致する。

**B. 射影 resnet（512→384、384→256）は表現できない。**
`res_x_y` は `out_channels = in_channels // block_config.get("multiplier", 2)` という整数除算しか持たない（`video_vae.py:505`）。512→384 には 4/3、384→256 には 3/2 が必要で、いずれも整数ではない。**よって専用の Configurator が必須である。**（上流も同じ事情で、Pruna の `patch_diffusers.py` は素の diffusers の `LTX2VideoUpBlock3d.__init__` と `LTX2VideoDecoder3d.__init__` を差し替えて `decoder_block_out_channels` という新しい設定キーを導入している。素の diffusers でもこの構造は組み立てられない。）

**C. `__metadata__["config"]` は必須である（事前想定を覆した確定事実）。**
`SingleGPUModelBuilder.model_config()`（`single_gpu_model_builder.py:56-58`）は必ず `sft_loader.metadata(path)` を呼ぶ。その実体は:

```python
def metadata(self, path: str) -> dict:
    with safetensors.safe_open(path, framework="pt") as f:
        return json.loads(f.metadata()["config"])
```
（`.venv-engine/.../ltx_core/loader/sft_loader.py:58-60`）

ガードが無い。`__metadata__` が無ければ `f.metadata()` が `None` を返して `TypeError`、あっても `"config"` キーが無ければ `KeyError` で落ちる。**カスタム Configurator が呼ばれる前に例外になる。**
したがって変換後ファイルには `__metadata__["config"]` を JSON 文字列として**必ず埋め込む**。当初の「`config` キーは埋めない」という想定は誤りであり、逆に必須である。

**D. タイル分割はチャンネル幅に一切依存しない。**
`self.video_downscale_factors = SpatioTemporalScaleFactors(time=8, width=32, height=32)` は**ハードコードされた定数**であり、ブロック構成から導出されていない（`video_vae.py:601-605`）。`_prepare_tiles` はこの定数だけを使う（`:781-798`）。`default_tiling_config`（`engine/pipeline/common.py:18-50`）も画素数とフレーム数しか扱わない。**枝刈りデコーダでもタイル設定は一切変更不要。**

**E. ブロック構成に依存するのは `up_blocks` と最終幅の2つだけ。**
`VideoDecoder.__init__` を通読した結果、`decoder_blocks` の影響を受けるのは (i) `self.up_blocks` そのものと、(ii) ループ後の `feature_channels`（`conv_norm_out` と `conv_out` の入力幅）だけである（`video_vae.py:620-668`）。`conv_in`・`patch_size`・`video_downscale_factors`・`per_channel_statistics` はすべて `base_channels` / `out_channels` / `in_channels` から決まり、ブロック構成とは無関係。しかも `norm_layer="pixel_norm"` なので `conv_norm_out` は `PixelNorm`＝**パラメータを持たない**（`:657`）。
つまり**ブロック構成に依存する学習パラメータは `up_blocks` と `conv_out` の2つに限られる**。この事実が §4.2 の設計を成立させる。

### 2.5 ★ norm3 の同値性判定（本テーマ最大の落とし穴）

枝刈りデコーダが新設した射影 resnet は `norm3` に**学習済みの weight と bias（形状＝入力チャンネル数）**を持つ。この `norm3` が ltx-core と diffusers で同じ演算かどうかを判定した。

**diffusers 側（重みが学習された実装）** — `LTX2VideoResnetBlock3d`。v0.38.0 タグと main の両方で同一であることを確認済み。

```python
self.norm3 = None
if in_channels != out_channels:
    self.norm3 = nn.LayerNorm(in_channels, eps=eps, elementwise_affine=True, bias=True)
```
```python
if self.norm3 is not None:
    inputs = self.norm3(inputs.movedim(1, -1)).movedim(-1, 1)
if self.conv_shortcut is not None:
    inputs = self.conv_shortcut(inputs)
hidden_states = hidden_states + inputs
```
出典: `https://raw.githubusercontent.com/huggingface/diffusers/v0.38.0/src/diffusers/models/autoencoders/autoencoder_kl_ltx2.py`（main も同一）

**ltx-core 側（我々が実行する実装）** — `ResnetBlock3D`。

```python
# Using GroupNorm with 1 group is equivalent to LayerNorm but works with (B, C, ...) layout
# avoiding the need for dimension rearrangement used in standard nn.LayerNorm
self.norm3 = (
    nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=eps, affine=True)
    if in_channels != out_channels
    else nn.Identity()
)
```
```python
input_tensor = self.norm3(input_tensor)
batch_size = input_tensor.shape[0]
input_tensor = self.conv_shortcut(input_tensor)
output_tensor = input_tensor + hidden_states
```
出典: `.venv-engine/Lib/site-packages/ltx_core/model/video_vae/resnet.py:91-97`（構築）・`:178-184`（forward）

#### 判定: **同値ではない。専用モジュールが必要である。**

- `nn.GroupNorm(num_groups=1)` は、1サンプルにつき **チャンネル・時間・高さ・幅のすべてをまとめて**平均と分散を取る。すなわち `nn.LayerNorm([C, F, H, W])` と等価である。
- diffusers の `nn.LayerNorm(in_channels)` を channel-last で適用したものは、**各 (バッチ, フレーム, 縦, 横) 位置ごとにチャンネル方向だけで**平均と分散を取る。
- 両者は正規化する軸の集合が違うため、**数値として一致しない**。ltx-core のコメントが主張する「GroupNorm(1) は LayerNorm と等価」は、`LayerNorm([C,F,H,W])` に対しては正しいが、diffusers が使う**チャンネルのみの** `LayerNorm(C)` に対しては誤りである。
- さらに悪いことに、**両者の学習パラメータは同じ形状 `[C]` の weight / bias である**。したがって重みは何の警告も出さずに読み込まれ、`strict=False` の警告（§2.2）にも掛からない。**発現するのは映像が壊れる形だけ**であり、静かに間違う典型例である。

#### なぜ今まで問題にならなかったのか

無改変の LTX-2.3 デコーダは `decoder_blocks` が `res_x` と `compress_*` だけで構成されており、`res_x_y`（入出力チャンネル数が異なる resnet）を1つも含まない。したがって**すべての `ResnetBlock3D` で `in_channels == out_channels` が成り立ち、`norm3` は `nn.Identity()`、`conv_shortcut` も `nn.Identity()` になる**（`resnet.py:85-97`）。この不一致が実行経路に乗ったことが一度も無かったため、wheel 側の誤ったコメントが温存されていた。**射影 resnet を持ち込む本テーマが、この経路を初めて踏む。**

#### 付随して確認した項目（いずれも一致）

| 項目 | diffusers | ltx-core | 判定 |
|---|---|---|---|
| `norm1` / `norm2` | `PerChannelRMSNorm(channel_dim=1, eps=1e-8)`：`x / sqrt(mean(x², dim=1) + eps)` | `PixelNorm(dim=1, eps=1e-8)`：同一式（`ResnetBlock3D` は `PixelNorm()` を既定引数で呼ぶため eps=1e-8） | **一致**（パラメータ無し） |
| 活性化 | `get_activation("swish")` | `nn.SiLU()` | **一致**（swish と SiLU は同一関数） |
| `conv_shortcut` | `nn.Conv3d(in, out, kernel_size=1, stride=1)` | `make_linear_nd(dims=3,...)` → `nn.Conv3d(in, out, kernel_size=1, bias=True)` | **一致**。state_dict のキーも `conv_shortcut.weight` `[out,in,1,1,1]` / `conv_shortcut.bias` で同じ |
| shortcut 分岐の順序 | `norm3` → `conv_shortcut` → 加算（どちらも**入力**に適用） | 同一 | **一致** |
| `norm3` の eps | `resnet_norm_eps = 1e-06` | `res_x_y` 経路で `eps=1e-6` | **一致** |

#### アップサンプラの並べ替え順序（確認済み・一致）

- ltx-core: `rearrange(x, "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)")`（`sampling.py:112-118`）
- diffusers: `reshape(B, -1, s0, s1, s2, F, H, W).permute(0,1,5,2,6,3,7,4).flatten(6,7).flatten(4,5).flatten(2,3)`

`reshape` によるチャンネル分解は `(c, p1, p2, p3)` の c 優先であり、`permute` 後の軸並びは `b c (d p1) (h p2) (w p3)` になる。**両者は完全に同一**である。

先頭フレームの切り落としだけ表現が異なる: diffusers は `[:, :, stride[0]-1:]`、ltx-core は `if self.stride[0] == 2: x = x[:, :, 1:, :, :]`。本デコーダの4段の `stride[0]` は 2 / 2 / 2 / 1 であり、この範囲では両者は同一に振る舞う（`stride[0]` が 4 以上なら食い違うが、そのような段は存在しない）。**一致と判定するが、念のため G3 の検証項目に残す。**

---

## 3. 変更ファイル一覧

### バックエンド（`Nz-LTX23-backend`）

- `engine/vae/pruned_video_decoder.py`（**新規**） — `PrunedVideoDecoder`（`VideoDecoder` の派生）＋ `ChannelLayerNorm3d`（norm3 差し替え）＋ `PrunedVideoDecoderConfigurator`
- `engine/pipeline/fast_video_pipeline.py` — `_LEDGER_BUILDER_ATTRS` は変更不要（`vae_decoder_builder` は既に含まれている）。**無条件位置**（`:383` の `_swap_registry(True)` の後）で既定ビルダーを1本だけ保持し、`_set_vae_mode_job()` を追加。`generate()` `:1587-1594` と `generate_chain()` `:1731-1735` に per-job の呼び出しを追加。**`_reset_vae_mode_job()` は作らない**（毎ジョブ明示設定で足りる。§4.3）
- `config.py` — `component_video_vae_pruned_path` を新設（`ModelConfig`、`:98-100` の隣）
- **`component_video_vae_pruned_path` の6段配線**（A-5。既存 `component_video_vae_path` と同じ経路をたどらないと値がワーカーへ届かない）:
  1. `config.py` — フィールド新設
  2. `services/ltx_runner.py:1118` / `:1132` — load ペイロードへ追加
  3. `engine/worker.py:342` — 受け取り
  4. `engine/pipeline/fast_video_pipeline.py` `create()` `:90` / `:115` — 引数として通す
  5. 同 `__init__()` `:141` / `:237` / `:260` — 保持
  6. 設置（§4.3 の既定ビルダー保持と同じ位置）
- `api/models.py` — `vae_mode` の宣言はそのまま。モック扱いの説明コメント（`:200-209` / `:456-460` / `:710-716`）を実装済みの記述へ書き換える
- `services/ltx_runner.py` — 加算的ペイロードへ `vae_mode` を追加（`:1411-1416` のコメント撤去＋送出、`:1640-1643` も同様）。`GenerationOutcome` に `vae_mode_used: str | None = None`。load ペイロード（`:1118` / `:1132`）にも新パスを追加（上記6段配線の2段目）
- `engine/worker.py` — `_resolve_vae_mode()` と `_vae_mode_used()` を追加（**後者は `_block_swap_prefetch_used` `:549-556` と同型**＝パイプラインのアクセサを読む。§4.5）。done イベント（`:792-801` / `:964-974`）へ `vae_mode_used` を加算。`:342` で新パスを受け取る
- `services/pipeline_manager.py` — metadata.json（`:903-924` / `:803-808` と呼び出し `:866-874` / `:727-742`）へ `vae_mode_used` を追加。**`GET /status` は変更しない**（§4.5 末尾）
- `gradio_ui/ui.py` `:1075-1081` — ラジオの `interactive=False` を外し、`info` を新設キーへ差し替え、ハンドラの入力リスト（`:1444-1445` / `:1921-1922`）とシグネチャ（`:1268` / `:1291` / `:1397` / `:1879`）へ追加
- `gradio_ui/i18n.py` — `accel_info_vae`（新設、日英）。`accel_info_unimplemented` は**参照が無くなるので削除**
- `gradio_ui/handlers.py` — 加算的コントラクト（`:512-513` 等と同型）で `vae_mode` を追加。シグネチャ既定は `:430-433` / `:556-569` / `:988-1002`
- `gradio_ui/batch.py` — `:211-214` の Acceleration スナップショットへ `vae_mode: str = "default"` を追加し、`:463-466` で転送
- `mcp_server/tools/generate.py` — **`vae_mode` を公開する**（§6.3・オーナー確定事項 0-8）。`submit_generate` / `submit_chain` の両ツールへ追加し、`:7-9` の「意図的に公開しない」という除外理由コメントを撤回する。**着手は G1〜G7 合格後の最終ステップ**
- `scripts/install_ltx.ps1` — サイズ検証しきい値の引き上げと `INSTALLED_PATHS.txt` の再生成（§7）

### 変換ツール（`Nz-GGUF-Converter-LTX23`）

- `src/converter/convert_vae.py`（**新規**） — デコーダ抽出＋キー改名＋`__metadata__` 生成＋自己検証
- `src/converter/cli.py` — `convert-vae` サブコマンドの登録
- `src/converter/convert.py` — `_SafetensorsRaw` に `shape_of()` を追加（現状ヘッダの `shape` を外へ出す口が無い）
- `config.toml` — `[prunavaed]` セクション（repo_id / revision / filename / expected_size / sha256 / 出力名）
- `tests/test_convert_vae.py`（**新規**）
- `README.md`・`Docs/DESIGN.md`・`Docs/VERIFICATION.md`

### フロントエンド（`Nz-LTX23-frontend-AviUtl2`）

- `webui/src/shell/accelerationSettings.ts` — `StoredAcceleration` へ `vaeMode` を追加（現在は永続化対象外）、`AccelerationRequestFields` へ `vae_mode?` を追加、`accelerationRequestFields()` へ加算的送出を追加
- `webui/src/shell/useAccelerationSettings.ts` — `setVaeMode` を新設し、永続化 `useEffect`（`:97-109`）へ追加
- `webui/src/shell/SettingsPanel.tsx` `:401-420` — `disabled` と `title` を外し、`onClick` を配線し、選択時の `field-hint` 注記を追加
- `webui/src/i18n/strings.ts` — `accelVaeNote` を新設（日英）。`accelNotImplementedTooltip` は参照が無くなるので削除
- `webui/src/api/types.ts` `:134-136` / `:251-252` — MOCK 注記を実装済みの記述へ
- テスト: `accelerationSettings.test.ts` `:172-175`、`SettingsPanel.test.tsx` `:270-273`、`useAccelerationSettings.test.ts`、各 payload テスト

---

## 4. 詳細設計

### 4.1 配布ファイルの形式と平坦キー配置

配布ファイル名は **`PrunaVAED-decoder-bf16.safetensors`** とする。

> **重要な命名上の制約**: ファイル名に **`video` という文字列を含めてはならない**。`services/model_registry.py` の `video_vae` カテゴリは `models/ltx-2.3-components/vae/` を走査し、ファイル名に `video` を含み `audio` を含まない `.safetensors` をすべて「選択可能な映像VAE」として登録してしまう（`:82-87` の `name_hint="video"`、判定は `:182-187`）。デコーダ単体のファイルがそこに現れると、利用者がサーバー全体の映像VAEとして選べてしまい、**エンコーダ側のビルダーが壊れる**。`PrunaVAED-decoder-bf16` は `video` を含まないので自動登録されない。加えて、配置先を専用のサブディレクトリ **`models/ltx-2.3-components/vae/prunavaed/`** とする（`_scan_category` は `recursive=False` なので、サブディレクトリは走査対象外＝二重の防御）。

パスは `config.py` の新フィールドで持つ:

```python
    # PrunaVAED: 枝刈り版の映像VAEデコーダ（デコーダ部のみ・約690MB）。
    # vae_mode="prune_vaed" のジョブでだけ読まれる。モデルレジストリには
    # 参加させない（サブディレクトリ＋"video" を含まないファイル名の二重防御。
    # services/model_registry.py:82-87 の name_hint 走査を参照）。
    component_video_vae_pruned_path: str = (
        "./models/ltx-2.3-components/vae/prunavaed/PrunaVAED-decoder-bf16.safetensors"
    )
```

**平坦キー配置（確定）。** `VideoDecoder` の `up_blocks` は種別混在の平坦な `nn.ModuleList` である。射影 resnet を2つ挿入するため、無改変の9要素に対し**11要素**になる。変換ツールはこの通りのキーを出力しなければならない。

| flat idx | 種別 | 幅（入→出） | state_dict キー |
|---|---|---|---|
| 0 | `res_x` ×2 | 1024 | `up_blocks.0.res_blocks.{0,1}.{conv1,conv2}.conv.{weight,bias}` |
| 1 | `compress_all` m=2 | 1024→512 | `up_blocks.1.conv.conv.{weight,bias}`（`[4096,1024,3,3,3]`） |
| 2 | `res_x` ×2 | 512 | `up_blocks.2.res_blocks.{0,1}.{conv1,conv2}.conv.{weight,bias}` |
| 3 | **射影 resnet** | 512→384 | `up_blocks.3.norm3.{weight,bias}`（`[512]`）／`up_blocks.3.{conv1,conv2}.conv.{weight,bias}`／`up_blocks.3.conv_shortcut.{weight,bias}`（`[384,512,1,1,1]`） |
| 4 | `compress_all` m=1 | 384→384 | `up_blocks.4.conv.conv.{weight,bias}`（`[3072,384,3,3,3]`） |
| 5 | `res_x` ×4 | 384 | `up_blocks.5.res_blocks.{0..3}.{conv1,conv2}.conv.{weight,bias}` |
| 6 | **射影 resnet** | 384→256 | `up_blocks.6.norm3.{weight,bias}`（`[384]`）／`up_blocks.6.{conv1,conv2}.conv.*`／`up_blocks.6.conv_shortcut.*`（`[256,384,1,1,1]`） |
| 7 | `compress_time` m=2 | 256→128 | `up_blocks.7.conv.conv.{weight,bias}`（`[256,256,3,3,3]`） |
| 8 | `res_x` ×6 | 128 | `up_blocks.8.res_blocks.{0..5}.{conv1,conv2}.conv.{weight,bias}` |
| 9 | `compress_space` m=2 | 128→64 | `up_blocks.9.conv.conv.{weight,bias}`（`[256,128,3,3,3]`） |
| 10 | `res_x` ×4 | 64 | `up_blocks.10.res_blocks.{0..3}.{conv1,conv2}.conv.{weight,bias}` |

加えて `conv_in.conv.{weight,bias}`（`[1024,128,3,3,3]`）、`conv_out.conv.{weight,bias}`（`[48,64,3,3,3]`）、`per_channel_statistics.mean-of-means`、`per_channel_statistics.std-of-means`。**合計102本。**

**上表のキー文字列が、配布ファイルに実際に書かれる最終形である**（`decoder.` などの前置は一切無い）。理由は直後の節を参照。

> `CausalConv3d` は実体の畳み込みを `self.conv` に持つ（`convolution.py:292`）ため、`conv1` / `conv2` / `conv_in` / `conv_out` および `DepthToSpaceUpsample` の `conv` はいずれも `.conv` を1段挟む。一方 `conv_shortcut` は `make_linear_nd` が返す**素の `nn.Conv3d`** なので `.conv` を挟まない（`convolution.py` の `make_linear_nd`）。ここを取り違えると §2.2 の「静かな失敗」に直行する。

**diffusers 側キーからの対応表**（変換ツールが実装する写像）:

| diffusers | ltx-core（平坦） |
|---|---|
| `decoder.conv_in.conv.*` | `conv_in.conv.*` |
| `decoder.mid_block.resnets.{0,1}.conv{1,2}.conv.*` | `up_blocks.0.res_blocks.{0,1}.conv{1,2}.conv.*` |
| `decoder.up_blocks.0.upsamplers.0.conv.conv.*` | `up_blocks.1.conv.conv.*` |
| `decoder.up_blocks.0.resnets.{0,1}.*` | `up_blocks.2.res_blocks.{0,1}.*` |
| `decoder.up_blocks.1.conv_in.*` | `up_blocks.3.*` |
| `decoder.up_blocks.1.upsamplers.0.conv.conv.*` | `up_blocks.4.conv.conv.*` |
| `decoder.up_blocks.1.resnets.{0..3}.*` | `up_blocks.5.res_blocks.{0..3}.*` |
| `decoder.up_blocks.2.conv_in.*` | `up_blocks.6.*` |
| `decoder.up_blocks.2.upsamplers.0.conv.conv.*` | `up_blocks.7.conv.conv.*` |
| `decoder.up_blocks.2.resnets.{0..5}.*` | `up_blocks.8.res_blocks.{0..5}.*` |
| `decoder.up_blocks.3.upsamplers.0.conv.conv.*` | `up_blocks.9.conv.conv.*` |
| `decoder.up_blocks.3.resnets.{0..3}.*` | `up_blocks.10.res_blocks.{0..3}.*` |
| `decoder.conv_out.conv.*` | `conv_out.conv.*` |
| `latents_mean` / `latents_std` | `per_channel_statistics.mean-of-means` / `std-of-means` |

> 変換ツールは上流の実キー名を**実測してから**この表を確定すること。上表は diffusers の `LTX2VideoDecoder3d` / `LTX2VideoUpBlock3d` の属性名から導いた期待値であり、`conv` の階層数（`CausalConv3d` 相当のラッパの有無）は実ファイルのヘッダで確認する。ここは G1 の検証項目である。

#### キーは「モジュール相対の素キー」であり、読み込み時の変換は行わない（A-1）

**上表のキーがそのまま配布ファイルに書かれる最終形である。** `decoder.` も `vae.decoder.` も前置しない。`PrunedVideoDecoder` に対する `load_state_dict` がそのまま通る形——すなわちモジュール相対の素キー——を変換器が直接出力する。

既定デコーダは `_install_component_sources()` が「素の `decoder.*` を `vae.decoder.*` へ前置してから `VAE_DECODER_COMFY_KEYS_FILTER` へ渡す」という連鎖 SDOps を組んでいる（`fast_video_pipeline.py:748-771`、フィルタは `model_configurator.py:65-71`）。**枝刈り側はこの機構を一切使わず、ビルダーに `model_sd_ops=None` を渡して素通しする。**

> **第2版の設計は致命的な誤りだった（敵対的レビュー A-1）。** 第2版は `SDOps("PRUNAVAED_DECODER_FLAT")` という名前だけの SDOps を新設して「平坦キーをそのまま使う」つもりでいたが、`SDOps` は**matcher を1つも持たないと全キーを `None` にして捨てる**（`sd_ops.py:92-97`。`any([])` は `False` を返すため、どのキーもマッチせず脱落する）。この設計のままなら state_dict が空になり、§2.2 の「静かな失敗」——`strict=False` で全パラメータが meta のまま、警告1行だけ——に直行していた。名前を付けただけの SDOps は「何もしない」ではなく「全部消す」である。

この変更に伴い、レジストリのキャッシュキーの議論も単純になる。キーは `sha256(解決済みパス群 + sd_ops.name)` だが（`loader/registry.py:58-64`）、**そもそもパスが違えば一意である**ため、SDOps 名に頼る必要が無い（§4.4）。

### 4.2 カスタム Configurator と `PrunedVideoDecoder`

§2.4-E で確定したとおり、`decoder_blocks` に依存する学習パラメータは `up_blocks` と `conv_out` の2つだけである。これを利用して、**wheel を1バイトも触らず、`VideoDecoder.__init__` の大半を再利用する**設計にする。

```python
# engine/vae/pruned_video_decoder.py（新規）

_WIDTH_ONLY_SKELETON = [("compress_space", {"multiplier": 16})]
# ↑ 使い捨ての「幅合わせ専用」ブロック列。1024 // 16 = 64 で、枝刈りデコーダの
#   最終幅と一致する。VideoDecoder.__init__ はブロック列を (1) up_blocks の構築と
#   (2) ループ後の feature_channels（= conv_norm_out と conv_out の入力幅）にしか
#   使わない（video_vae.py:620-668 を通読して確認）。up_blocks は直後に丸ごと差し
#   替えるので、この骨組みが必要なのは conv_out を 64 入力で作らせるためだけ。
#   Configurator は torch.device("meta") の下で呼ばれる（single_gpu_model_builder.py:62）
#   ため、捨てる畳み込み1個の確保コストはゼロである。


class ChannelLayerNorm3d(nn.LayerNorm):
    """diffusers の norm3 と同じ演算。ltx-core の GroupNorm(num_groups=1) では
    チャンネルと空間・時間を**まとめて**正規化してしまい、学習時の演算と一致
    しない（Docs/PRUNAVAED_WORKORDER.md §2.5）。重みの形状は両者とも [C] で
    同一なので、取り違えても警告は一切出ない。ここだけは自前で持つ。

    nn.Module ではなく **nn.LayerNorm を直接継承する**。こうすると学習パラメータ
    が自分自身の weight / bias になり、state_dict のキーが GroupNorm 版と同じ
    ``norm3.weight`` / ``norm3.bias`` のまま（1段深くならない）で済む。
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__(num_channels, eps=eps, elementwise_affine=True, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.movedim(1, -1)).movedim(-1, 1)
```

`nn.LayerNorm` を直接継承する設計により、**state_dict のキーは §4.1 の表のまま**（`up_blocks.3.norm3.weight` / `up_blocks.3.norm3.bias`）であり、diffusers 側のキー名とも一致する。変換ツールは改名も階層追加も不要である。この一致は G1（変換側の期待キー集合）と G2（構築側の警告ゼロ）で相互に検証する。

射影 resnet は `ResnetBlock3D` を構築したうえで `norm3` 属性を差し替える:

```python
def _make_projection_resnet(in_ch: int, out_ch: int, ...) -> ResnetBlock3D:
    block = ResnetBlock3D(
        dims=3, in_channels=in_ch, out_channels=out_ch,
        eps=1e-6, groups=32, norm_layer=NormLayerType.PIXEL_NORM,
        inject_noise=False, timestep_conditioning=False,
        spatial_padding_mode=PaddingModeType.ZEROS,
    )
    block.norm3 = ChannelLayerNorm3d(in_ch, eps=1e-6)
    return block
```

**属性差し替えを選ぶ理由**: `VideoDecoder.forward` のブロック分岐は `isinstance(up_block, ResnetBlock3D)` である（`video_vae.py:734`）。派生クラスでも `isinstance` は通るが、属性差し替えなら**分岐の判定方法に一切依存しない**ため、より安全である。なお分岐の順序は `UNetMidBlock3D` → `ResnetBlock3D` → `else`（＝`DepthToSpaceUpsample`）であり、`ResnetBlock3D` は `causal` と `generator` を受け取る（`:735`）。射影 resnet はこの経路に正しく乗る。

デコーダ本体:

```python
class PrunedVideoDecoder(VideoDecoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(decoder_blocks=_WIDTH_ONLY_SKELETON, **kwargs)
        self.up_blocks = nn.ModuleList([...])  # §4.1 の表どおり 11 要素
        # 骨組みが意図どおり効いたことをここで固定する（黙って形が変わる事故の防止）。
        # assert ではなく raise を使う: assert は python -O で消えるため、
        # 最適化起動された環境で検証が丸ごと無効化される。
        if tuple(self.conv_in.conv.weight.shape) != (1024, 128, 3, 3, 3):
            raise ValueError(f"conv_in shape mismatch: {tuple(self.conv_in.conv.weight.shape)}")
        if tuple(self.conv_out.conv.weight.shape) != (48, 64, 3, 3, 3):
            raise ValueError(f"conv_out shape mismatch: {tuple(self.conv_out.conv.weight.shape)}")
```

> **形状・構造の検証は本書を通じて `raise ValueError` で書くこと。** `assert` は `python -O` で消える。§9 の G2 が数えるチェックも同様である。

`forward` / `tiled_decode` / `_prepare_tiles` はすべて継承のままでよい（§2.4-D・E）。

Configurator は `ModelConfigurator` プロトコルの唯一のメソッドを実装する（`ltx_core/model/model_protocol.py`）:

```python
class PrunedVideoDecoderConfigurator:
    @classmethod
    def from_config(cls, config: dict) -> PrunedVideoDecoder:
        vae = config.get("vae", {})
        # 素性の確認（想定外のファイルを黙って受けない）
        ...
        return PrunedVideoDecoder(
            convolution_dimensions=3, in_channels=128, out_channels=3,
            patch_size=4, norm_layer=NormLayerType.PIXEL_NORM,
            causal=False, timestep_conditioning=False,
            decoder_spatial_padding_mode=PaddingModeType.ZEROS,
            base_channels=128,
        )
```

**構成の単一の正本をどこに置くか。** 上記は「バックエンドのコードに固定値で持つ」案である。変換ツールが埋め込む `__metadata__["config"]`（§2.4-C により必須）は、**同じ値を持つが権威ではない参照情報**として扱う。`from_config` は受け取った config の主要キー（`latent_channels=128`・`patch_size=4`・`norm_layer="pixel_norm"`・`causal_decoder=false`・`timestep_conditioning=false`・`decoder_base_channels=128`）が固定値と食い違ったら**例外で落とす**。こうすると (i) ローダーの必須要求を満たし、(ii) 正本はコード側の1箇所に留まり、(iii) 想定外のファイルを差されたら静かに動くのではなく落ちる。

> **`__metadata__["config"]` に `decoder_blocks` を含めてはならない**（第2版から変更）。ltx-core の語彙では射影 resnet を表現できない（§2.4-B）ので、含めたところで正しい構成にはならない。それどころか、**含めると害がある**——万一このファイルが標準の `VideoDecoderConfigurator` に食わされた場合、`decoder_blocks` があると「読めてしまい」、射影 resnet を欠いた別物のデコーダが**静かに**構築される。省いておけば、そのとき確実に例外で落ちる。**静かに間違うより、確実に落ちるほうを選ぶ。** 併せて `_class_name` は `"PrunaVAEDDecoder"` として既存の `"CausalVideoAutoencoder"` と区別する。
>
> したがって `config` に入れるのは、`PrunedVideoDecoderConfigurator` が突き合わせに使う素性のキーだけ（`latent_channels` / `patch_size` / `norm_layer` / `causal_decoder` / `timestep_conditioning` / `decoder_base_channels` / `_class_name`）とする。

### 4.3 ジョブ単位の切り替え

**保持するのは既定ビルダー1本だけとし、枝刈りビルダーはジョブごとに都度生成する**（敵対的レビューの簡素化提案を採用）。`SingleGPUModelBuilder` は frozen dataclass なので `dataclasses.replace` は安価であり、2本を保持して同期を気にするより、1本から都度作るほうが状態が減る。

**保持位置は無条件位置**（A-6）。`_install_component_sources()` の直後ではなく、**`:383` の `_swap_registry(True)` の後**に置く。理由は `_install_component_sources()` が `use_component_files=True` の構成でしか走らないためで、直後に置くと**単一ファイル構成のサーバーで枝刈りビルダーが作られない**。

```python
# fast_video_pipeline.py  __init__ 内、:383 の _swap_registry(True) の後

# 既定の映像VAEデコーダのビルダーを1本だけ控えておく。枝刈り側はジョブ毎に
# ここから dataclasses.replace で作る（保持しない＝同期ずれの余地を作らない）。
self._default_vae_builder = self.pipeline.model_ledger.vae_decoder_builder
```

```python
def _set_vae_mode_job(self, mode: str) -> None:
    """ジョブ単位の映像VAEデコーダ選択。sage と同じ per-job set 規律。
    reset は持たない（毎ジョブ必ず明示設定するため。§8 E7）。
    """
    ledger = self.pipeline.model_ledger
    want_pruned = (mode == "prune_vaed")
    # 重みの存在確認は「ジョブ単位」で行う。パイプライン構築は load オペで
    # 1回きり（:287-383）なので、そこで確認しても利用者が後からファイルを
    # 消した場合に追随できない（敵対的レビュー A-3）。
    have_pruned = want_pruned and bool(self._pruned_vae_path) and os.path.exists(self._pruned_vae_path)

    if have_pruned:
        ledger.vae_decoder_builder = dataclasses.replace(
            self._default_vae_builder,
            model_path=self._pruned_vae_path,
            model_class_configurator=PrunedVideoDecoderConfigurator,
            model_sd_ops=None,            # 素通し。§4.1（A-1）
            registry=ledger.registry,     # 常に「今の」registry を注入する。§4.4
        )
    else:
        ledger.vae_decoder_builder = dataclasses.replace(
            self._default_vae_builder, registry=ledger.registry
        )
        if want_pruned:
            logger.warning(
                "PrunaVAED decoder not found at %s - falling back to the default "
                "decoder for this job", self._pruned_vae_path,
            )

    self._vae_mode_used = "on" if have_pruned else ("on->off" if want_pruned else "off")
```

呼び出し点は `generate()`（`:1587-1594`）と `generate_chain()`（`:1731-1735`）の**両方**。§2.2 のとおりデコーダ生成点は `chain_pipeline.py:870` と `common.py:271` の2つあり、どちらも `ledger.video_decoder()` 経由なので、ledger 上のビルダーを張り替えれば両方に届く。

**`_LEDGER_BUILDER_ATTRS` について。** `vae_decoder_builder` は既に含まれている（`:36`）ので追加は不要である。

### 4.4 `keep_resident` との併用（オーナー確定事項 0-3 のハードゲート）

`StateDictRegistry` のキャッシュキーは `sha256(解決済みパス群 + sd_ops.name)`（`loader/registry.py:58-64`）である。既定デコーダと枝刈りデコーダは**そもそもパスが違う**ので、SDOps 名に頼らずともキーは一意になる（A-1 で枝刈り側の `sd_ops` は `None` になった）。したがって**キーが衝突せず、両方の state_dict が同じレジストリに同居できる**。ジョブAで既定、ジョブBで枝刈り、ジョブCで再び既定、と切り替えても、B と C はディスクを読み直さない。これが 0-3 を満たす構造的な理由である。

#### 本当の危険は「順序」ではなく「保持したスナップショットの registry 陳腐化」である

第2版は「`_swap_registry()` と `_set_vae_mode_job()` の呼び出し順序」を危険として書いていたが、これは論点がずれていた。真の危険はこうである:

- `_set_keep_resident_job()` は**状態が遷移したときにしか** `_swap_registry()` を呼ばない（`:700-701` の早期 return）。
- 一方 `self._default_vae_builder` は `__init__` の1回きりで撮ったスナップショットであり、**その中の `registry` 参照は撮った時点のもので固定される**。
- したがって、その後に `keep_resident` の ON/OFF が切り替わって `_swap_registry()` がレジストリを差し替えても、スナップショットが抱えている `registry` は**古いオブジェクトのまま**である。これをそのまま ledger へ代入すると、キャッシュが効かない（別レジストリを見る）か、解放済みのレジストリを掴む。

**この設計はその危険を構造的に排除している**——枝刈り側を保持せず都度生成し、**どちらの分岐でも `registry=ledger.registry` を代入時点で注入し直す**からである（§4.3 のコードは `else` 側でも `dataclasses.replace` を掛けているのはこのため）。順序を気にする必要は無くなり、`_set_vae_mode_job()` はいつ呼んでもよい。

`keep_resident` を OFF にすると `_swap_registry(False)` が `old.clear()` を呼ぶ（`:659-672`）。両方の state_dict がまとめて解放される。これは意図どおりである。

**メモリの見積り**: 枝刈りデコーダの state_dict は**約690MB**（345,006,256 パラメータ × BF16）。既定デコーダのデコーダ部は**約814MB**（407,169,072 パラメータ × BF16）。`keep_resident` ON で両方を常駐させると CPU 側に **+約690MB**。`keep_resident` は元々「64GB以上推奨」の機能（`gradio_ui/ui.py:1066-1069` の注記）なので、この増分は許容範囲である。**ただし実測して §52 に記録すること**（G4）。

### 4.5 API・worker・観測

- **ペイロード**（`services/ltx_runner.py`）: 既定と違うときだけ送る加算的コントラクトに従う。`:1411-1416` のコメント（「モックなので絶対に載せない」）を撤去し、実装済みの説明に差し替える。

  ```python
  # vae_mode: 既定 "default" なので、枝刈りを選んだときだけ鍵が乗る。
  # ファイルが無い環境では worker 側が "on->off" へ降格して完走する。
  if request.vae_mode != "default":
      payload["vae_mode"] = request.vae_mode
  ```

- **リゾルバ**（`engine/worker.py`）: `_resolve_block_swap_prefetch`（`:538-546`）と同じ「速度のつまみなので fail-loud にしない」regime を採る。ただし `vae_mode` は列挙値なので、**未知の値は fail-loud**（`_resolve_attention` `:478` 型）とする。API の `Literal` で既に弾かれているため、ここに来る未知値はプロトコル違反であり、黙って既定へ落とすと原因追跡が不能になる。

- **観測**: `vae_mode_used` を3値 `"off"` / `"on"` / `"on->off"` で報告する。踏襲する型は **`_block_swap_prefetch_used`（`:549-556`）**であって `_keep_resident_used`（`:641-651`）**ではない**（A-4）。両者の違いは降格判定がどこで起きるかである——`keep_resident` は判定がワーカー側で完結するので状態を読まないが、**`vae_mode` の降格判定（重みファイルの存在確認）はパイプライン内の `_set_vae_mode_job()` で起きる**（§4.3・A-3）。したがってワーカーは `block_swap_prefetch` と同じく**パイプラインのアクセサを読んで**結果を受け取る。done イベント（`:792-801` / `:964-974`）→ `GenerationOutcome`（`:251-295`）→ metadata.json（`:903-924` / `:803-808`）まで既存の3段はそのまま通す。

- **`GET /status` には載せない。** acceleration ブロックは**環境に由来する利用可否**だけを列挙する場所であり（`services/pipeline_manager.py:219-228`。`keep_resident` と `fused_gguf_dequant_kernel` は載っていない）、重みファイルの有無は「環境の能力」とは性質が違う。`keep_resident` と同じ扱い——**`/status` には出さず、`vae_mode_used` で事後に観測する**——とする。ファイルが無ければ `"on->off"` に落ちて完走する（G5）。

---

## 5. 変換ツール（`Nz-GGUF-Converter-LTX23`）

### 5.1 このリポジトリに置く理由と、置いたことによる看板の変更

オーナー確定事項 0-4 により、変換スクリプトはバックエンドではなくこのリポジトリに置く。理由は既存の資産がそのまま効くこと——`_SafetensorsRaw`（`src/converter/convert.py:82-171`）が safetensors のヘッダだけを読んで `data_offsets` で生バイトを取り出す仕組みを持っており、**BF16 のまま素通しするなら1バイトの変換も要らない**（`get_bf16_bytes` `:151-171` は BF16 源をそのまま返す）。torch 非依存の方針（`Docs/DESIGN.md` §2-5）も維持できる。

ただし現在の `README.md` 冒頭（`:3-5`）は「Sulphur 2 base を Q4_K_M GGUF へ変換する専用ツール」と名乗っている。**冒頭に「本リポジトリは Nz-LTX23 系の重み変換ツール全般を収める場所であり、GGUF 変換だけではない」旨の一文を足す**（オーナー確定事項ではないが、看板と中身の食い違いを残さないための必須整備）。

**併せて既存の不備を1つ直す**: `README.md` のサブコマンド一覧には **`all` が載っていない**（実装はあるのに書かれていない）。`convert-vae` を追記する作業のついでに `all` も補う。

### 5.2 新設サブコマンド `convert-vae`

`src/converter/convert_vae.py` を新設し、`cli.py` の3点セット（`cmd_convert_vae(args, config) -> int` をハンドラ群 `:62-215` へ、`subparsers.add_parser(...)` ＋ `set_defaults(handler=...)` を `build_parser()` `:221-353` へ、`config.toml` の新セクションを `getattr(args, "x", None) or config[...]` で読む）で登録する。モジュール一覧（`Docs/DESIGN.md:103-112`）と `cli.py` の docstring（`:6-13`）も更新する。

入手は既存の `download` サブコマンドが流用できる（`repo_id` / `filename` / `local_dir` / `expected_size` を取る汎用実装。`src/converter/download.py:42-111`）。ただし**現状 revision 固定にも SHA256 検証にも対応していない**（サイズのみ2回照合）。オーナー確定事項 0-5 が revision と SHA256 の固定を要求しているので、**`download` に `revision` 引数と SHA256 照合を追加する**（`hf_hub_download(..., revision=...)` を通し、ダウンロード後にストリーミングで SHA256 を取って照合する）。既存の呼び出しは `revision=None` で従来どおり動く。

**照合の失敗は `raise` で表明する。** 既存の `download.py:106` はサイズ照合を `assert` で書いているが、`assert` は `python -O` で消える。新設する SHA256 照合は `raise ValueError` とし、**ついでに既存のサイズ `assert` も `raise` へ直す**（重みの取り違えを検出する最後の砦が最適化で消えるのは受け入れられない）。

処理そのものは (i) ヘッダ解析、(ii) `decoder.*` と `latents_*` のキーを §4.1 の表で改名、(iii) 生バイトをそのまま連結、(iv) `__metadata__` を付けて書き出す、の4段である。`_SafetensorsRaw` には `shape_of(name)` が無い（ヘッダの `shape` を外へ出す口が無い）ので追加する。出力の書き出しには既存のライターが無いため、`tests/test_convert.py:59-78` の `_write_safetensors` と同型の**ストリーミング版**（テンソルを1本ずつ書き、常駐は1本まで）を実装する。

`__metadata__` に入れるもの:

| キー | 内容 |
|---|---|
| `config` | **必須**（§2.4-C）。`{"vae": {...}}` 形式。`_class_name` は `"PrunaVAEDDecoder"` |
| `model_version` | `"PrunaVAED-v2"` |
| `license` | LTX-2 Community License の全文（既存 `LTX23_video_vae_bf16.safetensors` と同じ作法） |
| `provenance` | 元リポジトリ `PrunaAI/PrunaVAED`・revision `4baacd7e...`・元ファイルの SHA256・変換ツールのバージョン |

### 5.3 変換の自己検証（`convert-vae` が必ず出力するもの）

1. **テンソル本数が 102 であること**
2. **キー集合が期待集合と完全一致すること**（過不足のどちらでも失敗。§4.1 の表から機械的に生成する）
3. **形状表の一致**（全102本）
4. **パラメータ総数が 345,006,256 であること**（`per_channel_statistics` を除く）
5. **バイト単位の素通し確認を全100テンソルで総当たりする**（第2版の「標本」から格上げ）。元ファイルの `data_offsets` が指す領域と、出力ファイルの対応領域の MD5 を**1本ずつ全件**突き合わせる。
   > **格上げの理由（敵対的レビュー）**: 項目1〜4・6〜8 はいずれも「集合」と「形状」しか見ていないため、**同じ形状のテンソルどうしが入れ替わっても全項目が素通しする**。たとえば `conv1` と `conv2`（`up_blocks.5` では両方 `[384,384,3,3,3]`）を取り違えても、本数も形状もパラメータ総数も総サイズも変わらない。標本抽出では取りこぼす。**全件突き合わせだけがこの誤りを捕まえられる。** 690MB を2回読むだけなので費用も知れている
6. **`latents_mean` / `latents_std` が既存 `LTX23_video_vae_bf16.safetensors` の `per_channel_statistics.*` とバイト一致すること**（可能な環境でのみ。バックエンドのファイルが手元にある前提の任意項目）
7. **出力ヘッダの再解析**（書いたものを読み直して round-trip する）
8. **総サイズが 690,012,512 バイト＋ヘッダ＋統計量であること**

### 5.4 テスト

`tests/test_convert_vae.py` を新設し、既存の手組み BF16 フィクスチャの作法（`tests/test_convert.py:36-45` のビット変換ヘルパ、`:59-78` の `_write_safetensors`、`:103-141` の `_build`）を踏襲する。**小さな縮小版**（各幅を 1/64 にしたもの等）を作って、キー改名・本数・形状・バイト素通しを検証する。実行は `PYTHONPATH=src .venv\Scripts\python.exe -m pytest tests/ -v`（`Docs/VERIFICATION.md:101`）。

---

## 6. UI

### 6.1 Gradio UI（バックエンド）

**表示名を `PruneVAED` → `PrunaVAED` に統一する。** `gradio_ui/ui.py:1076` の `choices` のラベル側を正式名称へ直す。**API 値 `"prune_vaed"` は外部コントラクトなので不変**である（ラベルだけを変える）。

`gradio_ui/ui.py:1075-1081` のラジオから `interactive=False` を外し、`info` を新設キー `accel_info_vae` に差し替える。ハンドラの入力リスト（`:1444-1445` / `:1921-1922`）とシグネチャ（`:1268` / `:1291` / `:1397` / `:1879`）へ `accel_vae` を加え、`handlers.py` は既存の加算的コントラクト（`:512-513` 等）と同型で `vae_mode` を積む。`batch.py:211-214` のスナップショットにも `vae_mode: str = "default"` を追加し `:463-466` で転送する。

`accel_info_unimplemented`（`i18n.py:90-91` / `:618-619`）は**参照が無くなるので削除する**（`fused_gguf_dequant_gemm` 撤去時と同じ作法）。`ui.py:1020-1026` と `i18n.py:69-73` / `:600-604` のコメント（「VAE のラジオだけが未実装」）も書き換える。

日本語の文言（オーナー確定事項 0-2 に従う）:

```python
"accel_info_vae": ("枝刈り版（PrunaVAED）を選ぶと映像の復元が速くなります。"
                   "出力品質がわずかに低下する可能性があります。"),
```

英語:

```python
"accel_info_vae": ("PrunaVAED is a pruned decoder that speeds up video "
                   "reconstruction. Output quality may be slightly reduced."),
```

### 6.2 フロントエンド（AviUtl2 webui）

既存の資産がほぼ揃っている。型 `VaeMode`（`accelerationSettings.ts:64-66`）、既定 `VAE_MODE_DEFAULT`（`:131`）、設定型のフィールド（`:93`）、文言 `accelVaeLabel` / `accelVaeDefault` / `accelVaePruneVaed`（`strings.ts:976-980` / `:1700-1702`）はそのまま使える。

**表示名の統一**: `accelVaePruneVaed` の**値**を `"PruneVAED"` → `"PrunaVAED"` に直す（`strings.ts:980` と `:1702` の日英両方。この文言は「両辞書で意図的に同一」と注記されているので、片方だけ直さないこと）。`SettingsPanel.test.tsx:271-273` がこの文字列を PIN しているので同時に更新する。**キー名 `accelVaePruneVaed` と API 値 `"prune_vaed"` は変えない**（前者は改名の必要が無く、後者は外部コントラクトである）。

足りないのは次の4点:

1. `StoredAcceleration`（`:158-172`）へ `vaeMode` を追加し、`readStoredAcceleration()`（`:174-228`）へ per-field のフォールバックを足す（現在は永続化対象外＝毎マウント既定に戻る）
2. `AccelerationRequestFields`（`:318-325`）へ `vae_mode?: VaeMode` を追加し、`accelerationRequestFields()`（`:327-380`）へ加算的送出を追加。**このファイルが唯一の送出点**であり、Create / Chain / Batch の3箇所はこれを spread しているだけなので、ここ1箇所で済む（`:340-345` のコメントが「実装されたらここに足せ。`fusedGgufDequantKernel` が 2026-08-04 にやったのと同じ」と明記している）
3. `useAccelerationSettings.ts` へ `setVaeMode` を新設（`:111-125` の4つの `useCallback` と同型）し、永続化 `useEffect`（`:97-109`）へ追加。**書き出しは必ず `useEffect` 側で行う**（render 中の副作用にしない）
4. `SettingsPanel.tsx:401-420` から `disabled` と `title` を外し、`onClick={() => setVaeMode(mode)}` を配線。選択時に `field-hint` の注記（`accelVaeNote`）を出す（`:291-315` の fused-dequant 行と同型）

`accelNotImplementedTooltip`（`strings.ts:982` / `:1703`）は参照が無くなるので削除する。`api/types.ts:134-136` / `:251-252` の MOCK 注記も書き換える。

型検査は **必ず `npm run typecheck`**（`webui/package.json:14` の `tsc -b --noEmit`）を使う。`npx tsc --noEmit -p .` は偽合格するため使ってはならない。配布は `scripts/deploy.ps1` で2箇所（実機の AviUtl2 Plugin ディレクトリ `:135-147` と、バックエンドリポジトリ内の配布コピー `:158-182`）へ配る。

### 6.3 MCP サーバー

**公開する**（オーナー確定事項 0-8。起票時の推奨案「非公開のまま維持」からの方針転換）。オーナーの利用意図は「外出先から操作したいときに便利」であり、外出先で使う経路にこそ速度のつまみが要る、という理屈である。

**順序を守ること。** MCP への公開は**本テーマの最終ステップ**であり、バックエンド実装と実機ゲート G1〜G7 の合格を待ってから行う。理由は、MCP のツール表面は一度公開するとエージェント側が使い始める外部コントラクトになるためで、実機で裏の取れていない機能を先に見せない。`fused_gguf_dequant_kernel` が §51 で辿った順序（実装 → ゲート → 公開）と同じである。

改修内容:

- `mcp_server/tools/generate.py:7-9` の「`vae_mode` も出さない——Acceleration 機能のうち現状モック（受理のみで効果が無い）の唯一の項目であり…（計画D1）」という**除外理由コメントは撤回する**。実装済みの説明に書き換え、「2026-08-05 のオーナー裁定で公開へ転じた」旨を1行残す（`two_stage_hq` / `pipeline` などの他の隠しフィールドは引き続き非公開であり、その理由は温存する）。
- `submit_generate` / `submit_chain` の両ツールへ `vae_mode` を追加する。既存の `attention_backend` / `block_swap_prefetch` / `keep_resident` / `fused_gguf_dequant_kernel` と同じ素通しの作法に従う。
- ツールの説明文は §6.1 の Gradio と同じ趣旨（「枝刈り版のデコーダで映像の復元が速くなる。出力品質がわずかに低下する可能性がある」）とし、**既定が `"default"` であることと、既定のままなら従来と完全に同じであること**を明記する。

**検証項目（G9 として §9 に定義）**:

- ツールスキーマに `vae_mode` が現れ、列挙値が `"default"` / `"prune_vaed"` の2つであること
- MCP 経由で `vae_mode="prune_vaed"` を指定したジョブが実際に通り、生成後のメタデータの `vae_mode_used` が `"on"` になること（**実叩き**。エージェントが担当する）
- `vae_mode` を省略したジョブが従来どおり動き、`vae_mode_used` が `"off"` になること
- MCP サーバーの既存テスト群がすべて PASS すること

---

## 7. 配布とインストーラ

再ホスト先は `Rootport/Nz-LTX23-weights`（既存）。配置は `models/ltx-2.3-components/vae/prunavaed/PrunaVAED-decoder-bf16.safetensors`。

`scripts/install_ltx.ps1` の改修は**3点だけ**である。

**`Invoke-ModelDownload` への追加は不要**（敵対的レビューで確認）。既存の取得はグロブ `ltx-2.3-components/*` で行われており、**このグロブは `/` を跨いで再帰的にマッチする**とスクリプト自身のコメントが明記している。新しいサブディレクトリ `vae/prunavaed/` に置いたファイルは、追加の呼び出しなしに**自動的に取得対象へ入る**。ここに1件足すと二重取得になる。

1. **サイズ検証しきい値の引き上げ**（`:599-608`）。現在 `models/ltx-2.3-components` の `Min` は `[long] 4000000000`。実サイズは 4,129,262,838 バイトで、しきい値の根拠は「この表が弾く最小のファイル（364,855,188 バイト＝音声VAE）を失っても下回る」という不変条件（`:589-598`）である。約690MB が加わって 4,819,275,350 バイトになるので、**新しい最小ファイルは 364,855,188 バイトのまま**。したがって `Min` は `4,819,275,350 − 364,855,188 = 4,454,420,162` を**上回る**値、かつ完全な状態（4,819,275,350）を通す値でなければならない。
   → **`[long] 4500000000`** とする。
   > **第2版は `4400000000` と書いており、これは誤りだった**（自ら算出した下限 4,454,420,162 を下回っている＝音声VAEが丸ごと欠けても検証を通してしまう）。敵対的レビュー A-2 の指摘による修正である。`:589-598` の根拠コメントも新しい数値で更新すること。
2. **Step 6 の必須ファイル検証表**（`:770-787`）に1行足す:

   ```powershell
       @{ Label = "component_video_vae_pruned"; Rel = "models/ltx-2.3-components/vae/prunavaed/PrunaVAED-decoder-bf16.safetensors"; IsDir = $false; Min = [long]680000000 }
   ```

   **この行は「必須」で確定である**（オーナー確定事項 0-9）。欠けていたら **exit 1** で失敗させ、「入っているか入っていないか分からない」中間状態を作らない。
   > **適用のタイミングに注意**: この行と 1. のしきい値変更は、**G6 の採否が「採用」で確定してから**適用すること。G6 が「不採用」または「保留」に着地した場合、まだ製品に入っていない重みを必須にすると**既存利用者の再インストールが失敗する**。変換・再ホスト・バックエンド実装は先行してよいが、インストーラの必須化だけは採否確定の後である。
3. **`INSTALLED_PATHS.txt` の再生成**（`:819-836`）。ここは config.yaml を読まずにハードコードした here-string なので、**新しいパスを1行足す**。

> **降格経路は保険として設計に残す。** §4.5 と §8 E5 の「重みが無ければ `vae_mode_used="on->off"` で完走する」挙動は**撤去しない**。インストーラが必須にしたことと矛盾しないのは、守る対象が違うからである——インストーラは「正しく導入された環境」を保証し、降格経路は「利用者が後からファイルを消した場合」や「旧バージョンから移行した環境」で**生成そのものが止まらない**ことを保証する。ゲート G5 はこのまま維持する。

---

## 8. エッジケース

| # | 状況 | 挙動と根拠 |
|---|---|---|
| E1 | `keep_resident` **OFF** のとき | ジョブ毎にディスクから読む。これは既定デコーダでも同じ（ledger は「Models are not cached」`model_ledger.py:56-60`）。約690MB の読み込みは既定の**約814MB**より**軽い**ので、退行は無い。0-3 のハードゲートは keep_resident ON のときの要求である |
| E2 | チェーン経路 | デコーダ生成点は `chain_pipeline.py:870` と `common.py:271` の2つ。どちらも `ledger.video_decoder()` を経由するので、ledger 上のビルダー張り替えで両方に届く。**`generate_chain()` 側の `_set_vae_mode_job()` 呼び出しを忘れないこと**（`:1731-1735`）。チェーンでは全クリップ・全ステージで1つの設定が効く（`api/models.py:456-460` の既存規約と同じ） |
| E3 | 2段階アップサンプル（`two_stage` / `chunked_upsample`）との併用 | デコードは stage2 の後段であり、アップサンプルは潜在表現の段階で終わっている。タイル設定はチャンネル幅に依存しない（§2.4-D）ので相互作用は無い |
| E4 | `vae_mode` に不正値 | API の `Literal` が 422 で弾く（`tests/test_validation.py:521-527`）。worker まで届いたら fail-loud（§4.5） |
| E5 | 枝刈りファイルが存在しない | **ジョブ単位**で `os.path.exists` を確認し（`_set_vae_mode_job()` 内。§4.3）、無ければ `vae_mode_used="on->off"` で完走。ログに1行「PrunaVAED decoder not found at ... — falling back to the default decoder for this job」を出す（G5）。**パイプライン構築時の確認では不十分**——構築は load オペで1回きり（`fast_video_pipeline.py:287-383`）なので、利用者が起動後にファイルを消した場合に追随できない（敵対的レビュー A-3）。ジョブ単位なら**ワーカーを再起動せずに**着脱へ追随できる |
| E6 | ダウンロードの中断・サイズ不足 | `Invoke-ModelDownload` の `Test-CheckSet`（`install_ltx.ps1:516-532`）が再検証し、不足なら再取得のうえ throw。既存の仕組みがそのまま効く |
| E7 | 同時実行 | 単一ジョブ思想なので競合は無い。ただし `_set_vae_mode_job()` は sage / prefetch と同じ per-job の set 規律に従い、**ジョブごとに必ず設定し直す**（前ジョブの選択が残らない）。`keep_resident` だけが「リセットしない」例外である（`:680-698`）ので混同しないこと |
| E8 | `keep_resident` の切り替えと同一ジョブ内で `vae_mode` も変わる | §4.4 の順序規律により、代入直前に `registry` を揃えることで解決する |
| E9 | 音声VAE | 完全に別系統（`audio_vae` パッケージ、`config.py:99` の `component_audio_vae_path`）。一切触らない |
| E10 | 既定デコーダとのビット一致 | **期待しない。** `conv_out` が再学習されており値が異なる（§2.3）。上流も「元のデコーダとビット単位一致ではない」と明記している。オーナー確定事項 0-2 の文言はこの事実に対応している |

---

## 9. ゲート定義

house の作法（機械検証 → 実機ゲート → オーナー目視）に従う。

### G1 — 変換ツール（機械検証）

- `tests/test_convert_vae.py` の pytest が全 PASS
- 実ファイル（revision 固定・SHA256 照合済み）に対する `convert-vae` の実行が成功し、§5.3 の自己検証8項目がすべて PASS
- 実行コマンドと**出力全文**を `Docs/VERIFICATION.md` の新節へ記録（§6 の 10Eros の記録が書式の雛形）
- 出力ファイルのサイズ・テンソル本数・パラメータ総数（345,006,256）を明記

### G2 — バックエンドの読み込み（機械検証）

- 枝刈りデコーダが CPU / GPU の両方で構築でき、`load_state_dict` 後に **`"Uninitialized parameters or buffers"` の警告が1件も出ない**（`single_gpu_model_builder.py:81`）。**これが §2.2 の静かな失敗を捕まえる唯一の網である**
- 構築したモジュールのパラメータ総数が **345,006,256** と一致
- `conv_in.conv.weight` が `[1024,128,3,3,3]`、`conv_out.conv.weight` が `[48,64,3,3,3]`
- `up_blocks` が11要素で、種別の並びが §4.1 の表と一致
- `up_blocks.3.norm3` と `up_blocks.6.norm3` が `ChannelLayerNorm3d` であり、`nn.GroupNorm` **ではない**ことを明示的に検証する
- 読み込んだ state_dict のキー本数が 102 で、`load_state_dict` の返り値の `missing_keys` / `unexpected_keys` が**ともに空**であること（`strict=False` でも返り値には出るので、ここで拾う）
- **上記の検証はすべて `raise ValueError` で書く**（`assert` は `python -O` で消える。§4.2）
- **併せて確認する項目**: 第2版で「[推定]」のまま残っていた「diffusers の `PerChannelRMSNorm` の eps が 1e-8 で、ltx-core の `PixelNorm()` 既定と一致する」を、実装時に diffusers のソースで**実値として**確認する。パラメータを持たない層なので実害は小さいが、推定のまま残さない

### G3 — 数値の健全性（実機・catch-net）

**§2.5 の判定が万一誤っていた場合に必ず捕まえる網。**

- **潜在表現は自分で作る**。無改変の `ledger.video_encoder()` に短いクリップ（数フレームで足りる）を通してエンコードし、その出力を使う。第2版は「実ジョブから捕獲する」としていたが、**それには生成経路への一時計装が要り、G3 のためだけに本番コードを触ることになる**。エンコーダは枝刈りの影響を受けない（PrunaVAED はデコーダのみ）ので、自作した潜在表現で何ら問題ない
- **同一の潜在表現**を既定デコーダと枝刈りデコーダの両方でデコードする
- NaN / Inf が1つも出ないこと
- 上流の主張（PSNR 39〜41dB）に照らして **PSNR 35dB 以上**（保守的な基準）。norm3 を取り違えていれば映像が壊れるので、この指標が桁で落ちる
- 併せて §2.5 末尾で「一致」と判定したアップサンプラの先頭フレーム切り落としも、フレーム数が既定と一致することで確認する
- **任意項目**: 隔離した一時 venv（scratchpad 配下のみ。プロジェクトの venv には絶対に触れない）で diffusers 実装と突き合わせる交差検証。実施すれば `norm3` の判定を独立に裏づけられる

### G4 — バッチ・再読み込みなし（オーナー確定事項 0-3 のハードゲート）

**主検証は in-process テストとする**（敵対的レビューの簡素化提案を採用）。実機3ジョブは「常駐メモリの実測」だけに縮小する。実機通しは費用が高いわりに「ディスクを読み直していない」ことの証明が間接的（ログと時間）にしかならないのに対し、in-process なら**レジストリの中身を直接見て**証明できる。

- `tests/test_pipeline_vae_mode_swap.py` — `keep_resident` ON 相当の構成で `_set_vae_mode_job("default")` → `("prune_vaed")` → `("default")` を呼び、**`StateDictRegistry` に2エントリが同居している**こと（`_generate_id` の戻り値が2種類あり、互いに異なること）を検証する。これが 0-3 の直接証明である
- 同テストで、3回目の `("default")` が**1回目と同じレジストリキー**を引くこと（＝読み直しが起きない）を検証する
- `_LEDGER_BUILDER_ATTRS` に `vae_decoder_builder` が含まれることを検証する単体テスト（既に含まれているが、将来の削除を防ぐ回帰テスト）
- 同テストで、`_set_vae_mode_job()` が**どちらの分岐でも** `registry` を注入し直していること（§4.4 の陳腐化対策）を検証する
- **実機で測るのはこれだけ**: `keep_resident` ON で ジョブA（default）→ ジョブB（prune_vaed）→ ジョブC（default）を通し、**CPU 側の常駐メモリ増分（見積り +約690MB）**を実測して §52 に記録する

### G5 — 降格経路

- 枝刈りファイルを一時的に退避した状態で `vae_mode="prune_vaed"` のジョブを投げる
- ジョブが**成功**し、`vae_mode_used` が `"on->off"` になり、ログに明示的な1行が出ること
- チェーン経路でも同様
- **ワーカーの再起動は不要である**（§4.3 のとおり存在確認がジョブ単位だから）。手順としては「サーバーを動かしたままファイルを退避 → ジョブ投入 → `on->off` を確認 → ファイルを戻す → 再度ジョブ投入 → `on` に戻ることを確認」まで通すこと。**再起動を挟んでしまうと、この設計（A-3 の修正点）が効いていることを検証できない**

### G6 — 速度（実機・交互対比較）★このゲートが本機能の採否を決める

**house 規則: 速度計測は交互対比較を必須とする**（default → pruned → default → pruned の順で複数回）。

#### 計測するもの（両方を測る）

条件は **1280x768p の標準ワークロード**。交互対比較（default → pruned → default → pruned）で複数ペアを取り、次の2つを**両方とも**記録する。

| 指標 | ベースライン |
|---|---|
| **ジョブの合計生成時間**（開始〜完了） | **147.60秒**（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §50.4 の J1） |
| **映像VAEデコード区間** | **33.07秒**（同 J1。上流主張の1.68〜2.08倍なら約16〜20秒） |

> **ベースラインの訂正**: 第2版は「ジョブ全体 約165〜270秒」と書いていたが、これは §50.6 が「計測環境の汚染による過大値」と結論づけた数値を引いていた。**正しくは §50.4 J1 の 147.60秒**である。1秒はこれに対して約0.7%にあたる。

#### 三段判定（オーナー確定事項 0-10）

| 判定 | 条件 | 帰結 |
|---|---|---|
| **① 採用** | 合計生成時間の**平均短縮が 1 秒以上** | 製品化する。§7 のインストーラ必須化もここで初めて適用する |
| **② 保留** | 合計は 1 秒未満だが、**デコード区間が上流主張に近い倍率**（目安 **1.5 倍以上**）を出している | 機能は**既定 OFF のまま残置**する（撤去しない）。同時に、新しい論点「**動画エンコードの高速化は可能か**」を台帳へ起票し、オーナー判断を仰ぐ。デコードが速いのに合計が縮まないなら、律速は VAE ではなく動画エンコード側にあるという**発見**であり、その情報ごと捨てるのは惜しい |
| **③ 不採用** | デコード区間の倍率も出ない | 機能を採用しない。台帳へ差し戻す |

- **判定に足る回数を取ること。** 1秒は合計147.60秒に対して約0.7%にすぎず、実行ごとのばらつきに埋もれうる。**ペア数を増やし、ばらつき（各ペアの差の範囲）も併記する**こと。平均が 1 秒を超えていても、ペアごとの差が正負にまたがっているなら「短縮した」とは言えない。
- ピークVRAM（`peak_vram_mb` / `peak_vram_reserved_mb`）も記録する。上流は約50%削減を主張している。

#### デコード区間の計測機構は現存しない — 一時プローブの再導入が必要

**§50 の 33.07秒を出したフェーズ別計測のプローブは、計測後に撤去済みである**（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) `:3556-3564` に「リポジトリのコードは計測前と1バイトも違わない状態に戻してある」と明記）。したがって G6 の作業手順には次の3段を**含めること**:

1. **§50 型の一時プローブを再導入する**（フェーズ別の壁時計を記録するだけの計測用コード）。§50 と同じ形にすれば、ベースライン 33.07秒 / 147.60秒 と直接比較できる
2. 交互対比較を実施し、①②③の判定に必要な数値を採る
3. **プローブを撤去し、`git diff` が空であることを確認して §52 に記録する**（§50 と同じ作法。計測用コードを製品に残さない）

#### 読み方の注意

§50.4 が明記しているとおり、映像VAEデコードの33.07秒は**動画エンコード36.77秒の内側に含まれる**（デコードした分から順に書き出す構造のため）。したがって**デコード区間が速くなっても、その秒数がそのまま合計時間の短縮になるとは限らない**——動画エンコード側が律速になれば頭打ちになる。**②の判定枠はまさにこの事態のために用意してある。**

### G7 — オーナー実機・目視

- 同一シードでの A/B 比較（default 対 prune_vaed）をオーナーが目視し、品質低下が許容範囲であることを確認する
- Gradio UI とフロントエンド Settings > Acceleration の表示・操作・文言を目視で確認する
- 説明文が「出力品質がわずかに低下する可能性がある」になっていること（オーナー確定事項 0-2）

### G8 — 既存機能の回帰

- `vae_mode="default"` のジョブが改修前と **SHA-256 完全一致**の出力を出すこと（IC-LoRA の G4 と同じ作法。[`ICLORA_DEPTH_DEBLUR_WORKORDER.md`](ICLORA_DEPTH_DEBLUR_WORKORDER.md) §7.2）
- ワーカーペイロードが `vae_mode` 既定時にバイト単位で従来どおりであること（§10 の frozen key-set テスト）

### G9 — MCP サーバーへの公開（最終ステップ）

**G1〜G7 の合格後にのみ着手する**（オーナー確定事項 0-8）。検証項目は §6.3 に定義したとおり:

- ツールスキーマに `vae_mode` が現れ、列挙値が `"default"` / `"prune_vaed"` の2つであること
- MCP 経由の**実叩き**で `vae_mode="prune_vaed"` のジョブが通り、メタデータの `vae_mode_used` が `"on"` になること
- `vae_mode` 省略時に従来どおり動き、`vae_mode_used` が `"off"` になること
- MCP サーバーの既存テスト群がすべて PASS すること

---

## 10. テスト計画

### 10.1 変更が必要な既存テスト

| ファイル | 内容 | 変更 |
|---|---|---|
| `tests/test_ltx_runner_payload.py:288-300` | `test_generate_payload_never_carries_mock_acceleration_fields` — `vae_mode` が**乗らない**ことを PIN している | **反転**。`"prune_vaed"` のときだけ乗り、`"default"` のときは乗らないことを検証する形へ |
| `tests/test_ltx_runner_payload.py:319-328` | チェーン版の同じ PIN | 同上 |
| `tests/test_ltx_runner_payload.py:331-360` | frozen key-set テスト。`:332-338` のコメントが「既定リクエストの鍵集合は事前 acceleration のものと厳密に一致（既定が反転した2件を除く）」と述べている | **既定 `"default"` のままなので鍵は増えない**。コメントの「`vae_mode` はモックなので不在」という記述を「既定なので不在」へ書き換えるだけでよい |
| `tests/test_validation.py:429` | 「モック（受理されるが消費されない）」という節コメント | 記述を更新。`:448` `:458` `:472` `:482` `:521-527` `:544-557` の assert 自体は**そのまま通る** |
| `tests/test_smoke.py:165,182` | metadata.json の `request` エコーに `vae_mode` が乗ることを PIN | **そのまま通る**。加えて `vae_mode_used` の新キーを検証する assert を追加 |
| `tests/test_gradio_ui.py:682-694` | `test_acceleration_mock_controls_are_disabled` — VAE ラジオが唯一のモックで `interactive is False` | **書き換え**。「Acceleration にモックは1件も残っていない」ことを検証するテストへ（`interactive is True` と、選択肢が `["default","prune_vaed"]` であること） |
| `tests/test_gradio_ui.py:697-705` | 言語切替の登録チェック | `accel_info_vae` を追加、`accel_info_unimplemented` を削除 |
| `tests/test_model_swap_load.py:51` | **load ペイロードの凍結鍵集合**（A-5） | `component_video_vae_pruned_path` が6段配線で load ペイロードに乗るため、**この凍結集合を更新しないと必ず落ちる**。見落としやすいので実装の早い段階で通しておくこと |

**フロントエンド側で `vae_mode` の非搭載を PIN している6ファイル**（STEP 6＝フロントエンド追随のときの確認一覧）。いずれも**既定値ケースを見ているだけなので通る見込み**だが、加算的送出を入れた直後に必ず全件走らせること:

| ファイル | 該当行 |
|---|---|
| `batchRunner.test.ts` | `:479` |
| `buildA2vChainPayload.test.ts` | `:317` / `:319` |
| `useBatchForm.test.ts` | `:1207` |
| `chainUtils.test.ts` | `:428` / `:430` |
| `useChainForm.test.ts` | `:1137` / `:1139` |
| `useGenerationForm.test.ts` | `:238` / `:311` / `:315` |

### 10.2 新規テスト

- `tests/test_worker_vae_mode_resolve.py`（engine venv 側） — `_resolve_vae_mode` / `_vae_mode_used` の3値
- `tests/test_pruned_video_decoder.py` — G2 の構造 assert 群（meta デバイス上で構築するので GPU 不要）
- `tests/test_pipeline_vae_mode_swap.py` — ビルダー張り替え・`registry` の引き継ぎ・`_LEDGER_BUILDER_ATTRS` の包含
- 変換ツール `tests/test_convert_vae.py`（§5.4）
- フロントエンド: `accelerationSettings.test.ts`（加算的送出）、`useAccelerationSettings.test.ts`（setter と永続化）、`SettingsPanel.test.tsx`（有効化と onClick）
- MCP: 既存の MCP ツールテストへ `vae_mode` のスキーマ検証（列挙値2つ・既定 `"default"`）と素通し検証を追加（§9 G9。**G1〜G7 合格後の最終ステップで着手する**）

### 10.3 実行

- バックエンド `.venv`: `pytest`（直近 §51 時点で 942件中941件 PASS。1件は起動中バックエンド由来の環境要因）
- engine venv: 個別実行（pytest が入っていないモジュールは selfcheck 形式）
- フロントエンド: `npm run test`（直近 1700件 PASS / 109ファイル）＋ `npm run typecheck`
- 変換ツール: `PYTHONPATH=src .venv\Scripts\python.exe -m pytest tests/ -v`

---

## 11. ドキュメント更新一覧

### バックエンド

| ファイル | 内容 |
|---|---|
| [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) | **§52 を新設**（現在の最終節は §51）。§51 の構成を踏襲し、機械検証・G1〜G8・実測値を記録 |
| [`DEVLOG.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/DEVLOG.md) | 新節。`fused_gguf_dequant_kernel` のモック→実装転換（§55 周辺）の記述と対にする |
| `README.md` `:567` | 制限事項の表「VAE（映像の復元処理）｜ Default / PruneVAED ｜ **未実装**（グレーアウト・将来対応予定）」を実装済みへ |
| `README.md` `:699` | 「PruneVAED: 未実装。API 上は `vae_mode` を受け取るが値は生成に一切影響しない」を実装済みの説明へ。**正式名称 PrunaVAED への表記変更もここで行う** |
| `README.md` `:784` | 「Acceleration の残る1項目（PruneVAED）はモック＝未実装のため MCP に公開していない」を**削除**し、**MCP に公開済みである**旨へ（§6.3・オーナー確定事項 0-8）。この書き換えは G9 と同時に行う（それまでは記述と実態が食い違うため） |
| [`MCP_SERVER_DESIGN.md`](MCP_SERVER_DESIGN.md) | 公開フィールド一覧へ `vae_mode` を追加。「モックだから出さない」という設計判断の記述があれば撤回し、2026-08-05 のオーナー裁定で公開へ転じた旨を残す |
| `../LTX23_Backend_Specification.md`（**リポジトリ直下**） | `vae_mode` をモック扱いしている**3か所**を実装済みへ: `:513`（API フィールド表）／`:528`（Acceleration の補足）／`:749`（`/status` の説明）。表示名も `PruneVAED` → `PrunaVAED` へ |
| `scripts/install_ltx.ps1` `:589-598` | サイズ根拠コメントの更新（§7。しきい値は **4,500,000,000**）。**適用は G6 の採否確定後** |
| 本書 | 着手後は冒頭に歴史記録ブロックを追加していく（他の `*_WORKORDER.md` と同じ運用） |

> **訂正（第3版）**: 第2版は「`LTX23_Backend_Specification.md` は存在しない」と書いていたが、**これは誤りだった**。同ファイルは `Docs/` 配下ではなく**リポジトリ直下に実在する**（約160KB）。探す場所が違っていた。下表に更新対象として追加する。

### 変換ツール

| ファイル | 内容 |
|---|---|
| `README.md` | 冒頭に「本リポジトリは Nz-LTX23 系の重み変換ツール全般を収める」旨を追記（§5.1）。`:20-25` のサブコマンド一覧へ `convert-vae` を追加。`:40-49` のディレクトリ説明も更新 |
| `Docs/DESIGN.md` | `:103-112` のモジュール一覧へ `convert_vae.py` を追加。`__metadata__["config"]` が必須である理由（§2.4-C）を1節で説明 |
| `Docs/VERIFICATION.md` | 新節。§6（10Eros）の書式に従い、実行コマンド → 出力全文 → 終了コード → 結論 |

### 台帳（フロントエンドリポジトリ）

| ファイル | 内容 |
|---|---|
| [`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) | §3-50 を「§2 実装済み・ユーザーのテスト待ち」へ移し、完了後に [`PENDING_TASKS_CLOSED.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md) へ。**項目名を「PruneVAED」から「PrunaVAED」へ訂正し、旧表記からの改名である旨を1行残す**（過去のログとの照合可能性を保つため）。§3-53（「次の候補は映像VAEデコード33.1秒」）も更新 |

---

## 12. オーナー回答済みの記録（2026-08-05）

起票時に未確定として残していた4件は、**すべて 2026-08-05 にオーナーが回答した**。以下は質問と回答の対の記録である。決定そのものは §0 の確定事項 0-8〜0-11 に取り込んであり、実装時はそちらを正本とする。**本節は「なぜそう決まったか」を残すための記録であって、未処理の課題ではない。**

| # | 起票時の質問（と本書の推奨案） | オーナー回答（2026-08-05） | 反映先 |
|---|---|---|---|
| 1 | **MCP サーバーへの公開可否**。本書の推奨案は「実装後も公開しない——画質に影響する選択肢を増やす積極的な理由が無い」だった | **公開する**（推奨案を却下）。「一通り完成したら公開まで進みたい。外出先から操作したいときに便利なので」 | §0-8・§6.3・§9 G9・§3・§11 |
| 2 | **インストーラで必須扱いにするか任意扱いにするか**。§4.5 の設計では欠けても完走するので、どちらでも壊れない | **必須ダウンロードで確定**。常に取得し、欠けていたら失敗させる | §0-9・§7 |
| 3 | **G6 の合格基準**。本書はデコード区間で 1.3 倍以上を提案していた。加えて、動画エンコードとの重なりにより、ジョブ全体の壁時計はデコード短縮分そのままにはならない見込みである旨を添えた | **合計生成時間で判定する**。「1280x768pで1秒以上の生成時間短縮が見込めるなら、存在していい機能だと思う」。**その後（同日）オーナーが三段判定へ改定**——合計が1秒に届かなくてもデコード区間が速いなら「保留」とし、機能を残したうえで「動画エンコードの高速化」を新論点として起票する | §0-10・§9 G6 |
| 4 | **既定値の扱い**。ゲート合格後に既定を ON へ反転する専用ステップを置くか、恒久的に OFF 既定とするか。本書は「画質が変わる機能なので恒久 OFF が自然」と推奨した | **恒久 OFF で確定**（推奨案を採用）。既定反転ステップは置かない | §0-11 |

**回答から読み取れる方針**: 3 の回答は、本テーマの採否そのものを「利用者の待ち時間が実際に縮むかどうか」ひとつに絞ったものである。区間の倍率がいくら良くても合計時間が縮まなければ採用しない、という判断であり、§9 G6 はその形に書き直してある。1 の回答（MCP 公開）と 4 の回答（恒久 OFF）は一見逆向きに見えるが、矛盾しない——**選択肢としては広く届けるが、既定では踏ませない**という一貫した立場である。
