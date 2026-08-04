# Async Block Swap Prefetching（先読み block swap）実装ワークオーダー

- 作成: 2026-08-01
- 正本: 本書。承認済みプラン（`groovy-roaming-frog.md`、構成・ステップ・ゲートの正本）＋詳細設計（Planエージェントによる38kトークン設計文書、擬似コード・レイアウト計算・API/UI/フロント配線の出典）＋敵対的レビュー（`prefetch_plan_review_findings.md`、CRITICAL 3件・MAJOR 7件・MINOR 5件の修正指示）を統合した実装用正本。**3者が矛盾する箇所はレビューの修正内容を優先して統合済み**。着手後の状態遷移は本書冒頭に歴史記録ブロックを追加していく運用とする（他の `*_WORKORDER.md` と同じ体裁）。
- 対象: `Nz-LTX23-backend`（メイン）＋ `Nz-LTX23-frontend-AviUtl2`（`webui/src/shell` の Settings > Acceleration のみ）
- 機能名（内部識別子）: `block_swap_prefetch`
- 併読: `Docs/VERIFICATION_LOG.md` §43（Acceleration区画・attention_backend実装の前例。本件はここに §44 として新設）／`../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md`（§3-49 fused GGUF の扱い）

---

## 0. オーナー確定事項（最優先・すべてに優先する）

1. **fused GGUF dequant+GEMM は実測により今回クローズ**（dequant=1パス1.63秒＝stage2ステップ32.76秒の4.97%、Q6_K補正でも5〜7%）。**UIのdisabledトグルは撤去せず現状維持**。台帳（フロント `PENDING_TASKS.md`）の §3-49 は「§3 再訪条件つきでクローズした項目」へ移動する（§3-46と同型）。再訪条件は「さらなる高速化が必要になったとき（実測データを添付）」。
2. `block_swap_prefetch` の既定値は、実機ゲート（ビット一致＋VRAM）合格を条件に **on**。開発中は **off** で実装し、ゲート合格後に既定を反転する専用ステップ（S4）を置く。
3. VRAM追加は先読み分+1ブロック（≈370MB）まで許容する（快適フレーム上限より高速化を優先する）。
4. **off 時は現行挙動を完全温存する**（GPU→CPU退避コピーの廃止も on 時のみ適用。A/B比較のベースラインを保護するため）。

これらは実装中に他のどの記述とも矛盾した場合に優先する。

### 0.1 棄却された設計（詳細設計原案からの変更点・理由付き）

詳細設計文書の原案のうち、以下は敵対的レビューで欠陥が確定し、**本書では採用しない**。各項目の修正版は該当節に記載する。

| # | 原案 | 棄却理由 | 修正版の記載先 |
|---|---|---|---|
| 1 | arena を**転送 stream 上で確保**し、`record_stream` で allocator に使用申告する | PyTorch 2.9 の `CUDACachingAllocator` は stream を第1キーにしており、xfer stream 所有の解放済みブロックは compute stream の割り当てに再利用されない。stage1 最終パス終了時に約3.0〜3.3GBが xfer プールに滞留したまま spatial upsampler（最大VRAM山）を迎え、16GB機で共有メモリスピルを誘発する | §5.2（本体擬似コード）・§5.3（同期点表） |
| 2 | `_block_swap_prefetch_available()` が表示専用の `low_vram.block_swap`（bool）を参照する | real 経路のどこからも読まれておらず、既定構成では false。実ゲートと不整合な判定になる | §8.5 |
| 3 | Gradio/フロントの `fused_gguf` チェックボックスを撤去する | オーナー裁定により UI は現状維持（撤去しない） | §9・§10（該当箇所は「変更しない」と明記） |
| 4 | 後片付け（teardown）を**次ジョブの `install()` 冒頭**に置く | ジョブ N 終了〜N+1 ビルド完了の間、旧 transformer の CPU 正本48ブロックへの強参照と arena 1枚が残留する（`VERIFICATION_LOG.md` §9.6 の +18GB/job リークと同形） | §7 エッジケース E9・§8.4 パイプライン結線 |
| 5 | engine 側ユニットテストを pytest（`tests/test_block_swap_prefetch.py`, `pytest.importorskip("torch")`）として書く | `.venv-engine` に pytest が無い（app の `.venv` には torch が無い）。CPU only の想定自体が成立しない | §11.2（selfcheck に統合） |
| 6 | 常駐数の上限アサートを `<= blocks_on_gpu + 1` とする | ブロック47は前パスから残留するため定常時の `_state` は最大 `bs+2`。`+1` は実態と合わない | §11.2 C3 |

---

## 1. 背景（Context）

block swap（48ブロック中8個常駐のスライディングウィンドウ）は毎forwardパスで47ブロック（1個≈340-370MB、Q4_K_M圧縮のまま）をCPU⇔GPU間で同期転送しており、pinned memory・non_blocking・別CUDA streamは未使用。実測（マイクロベンチ）に基づく見積りで、転送待ちは1パスあたり約2.6〜3.1秒＝768p/257フレーム生成（全体270秒・11パス）の約11〜13%。

これを次の2つで削る:

1. **GPU→CPU退避コピーの廃止**（重みは推論中不変とテスト実証済み。CPU側の正本を保持し、GPU側は捨てるだけでよい）
2. **pinnedステージング＋専用転送stream＋eventによる先読み隠蔽**（H2D転送を計算の裏に隠す）

ジョブ単位パラメータ `block_swap_prefetch` を新設し、Gradio UIとフロントエンドの Settings > Acceleration にトグルを追加する。**攻撃対象は転送方式のみで、生成アルゴリズムは一切変えない**ため、同一シードなら off/on でビット単位一致が期待値になる（`attention_backend` の sage 選択とは違い、「絵が変わる」注意書きは不要）。

期待効果（§13 で再確認）: 768p/257f で 270s → 約236〜240s（1.13〜1.15倍）。ゲート合格基準は保守的に 1.08 倍以上とする。

---

## 2. 前提の裏取り結果（Plan エージェントが実装前に読んで確認した事実）

| 事実 | 根拠 |
|---|---|
| 同期swap本体。`block.to(device)` / `prev.to("cpu")`、pinned/stream一切なし | `engine/transformer/block_swap_service.py:167-218` |
| GPU常駐判定が `list(blk.parameters())` のみ（buffer非考慮） | 同 `:184-196` |
| 退避は `prev.to("cpu")`＝毎回D2H＋CPU側の新規確保 | 同 `:198-205` |
| `blocks_on_gpu >= total` で install 早期return（パッチ自体を張らない） | 同 `:72-77` |
| install時に全ブロックをCPUへ落としてからパッチ | 同 `:84-90` |
| service は**単一常駐インスタンス**、`_installed_transformers` は最新1件のみ保持（リーク対策） | 同 `:92-103` |
| `uninstall()` は本番未使用 | 同 `:105-121`（呼び出し元なし） |
| `GGMLQuantizedTensor.to()` が `_ggml_type`/`_float_shape` と subclass を保持 | `engine/gguf/quant_service.py:473-489` |
| 圧縮バイトは `register_buffer(persistent=True)`（Linear.weight を buffer 化） | 同 `:614-625` |
| `.shape/.size()/.dim()/.numel()` は**float形状を偽装**する（実storageは1D uint8） | 同 `:459-471` |
| subclass識別を落として生バイトを触る作法は `as_subclass(torch.Tensor).view(torch.uint8)` | 同 `:645`（コメント `:636-644` が理由を明記） |
| LoRA A/B は `register_buffer(..., persistent=False)`＝`block.to()` に相乗り | `engine/gguf/ic_lora_common.py:181-182` |
| LoRA attach は block swap install より前（`ledger.transformer` のラップ順） | `quant_service.py:763-800` を `fast_video_pipeline.py:239-244` で先に、`:276-277` で後に install |
| CPU常駐ビルド。非ブロック leaf のみ GPU へ。`_parameters`/`_buffers` の per-module 走査 | `engine/transformer/dit_cpu_load_service.py:39-82` |
| ジョブ毎に transformer 新規構築 →`service.install(t)` | `fast_video_pipeline.py:605-614` |
| chain は transformer を**1回だけ**構築し stage1/stage2 で使い回す | `engine/pipeline/chain_pipeline.py:653` |
| NAG/VSF/sage は install が block swap の**後**、ただし触るのは `forward`/`attention_function` 属性のみ（テンソルを増やさない） | `fast_video_pipeline.py:286,295`／`nag_service.py`・`vsf_service.py` に `register_buffer` なし（grep 0件） |
| attention_backend の配線パターン（踏襲元） | `api/models.py:112,376,634` → `services/ltx_runner.py:1353-1366,1562-1567` → `engine/worker.py:386-427,483,510,616,648` → `fast_video_pipeline.py:367-387,1127,1162-1170,1236,1261-1263` |
| /status の acceleration ブロック | `services/pipeline_manager.py:163-203`、`api/status.py:32-33` |
| Gradio Acceleration 区画 | `gradio_ui/ui.py:1006-1044`、`i18n.py:69-84 / 562-576`、`handlers.py:364,424-428,456,748-752,852,1161-1165`、`batch.py:157-158,191,440` |
| app側は `low_vram.block_swap` / `block_swap_blocks_on_gpu` を知っている | `services/low_vram.py:40,43,86-87`、`services/ltx_runner.py:1086` |
| フロントの加算的コントラクト／localStorage | `webui/src/shell/accelerationSettings.ts:50-152`（`:54-62` に「将来2つ目が来たらJSONへ移行」と明記済み） |

---

## 3. 変更ファイル一覧

### バックエンド

- `engine/transformer/block_swap_service.py` — 既存 `_patch_block`（`:153-218`）は**無改変**。`prefetch_requested`/`last_prefetch_used`/`_patch_block_prefetch`/`teardown_prefetch()` を追加
- `engine/transformer/block_swap_prefetch.py`（新規） — `PrefetchEngine`: CPU正本・pinned・stream・event・arena
- `engine/transformer/block_swap_prefetch_selfcheck.py`（新規） — `.venv-engine` 用の実行可能 selfcheck（pytest ではない。§11.2）
- `engine/pipeline/fast_video_pipeline.py` — `_set_block_swap_prefetch_job()`/finallyでの `_reset_block_swap_prefetch_job()` と `teardown_prefetch()` 呼び出し（`_set_sage_job` :367-387 と同流儀）、`_install_block_swap` :589-624 への配線
- `api/models.py` — `/generate`(:112付近)・`/generate/chain`(:376付近)に `block_swap_prefetch: bool = False`（S4で既定Trueへ反転）、`to_clip_request()`（:634付近）への転記
- `services/ltx_runner.py` — workerペイロード追加（:1365-1366, :1566-1567付近）、`GenerationOutcome` に `block_swap_prefetch_used` と `peak_vram_reserved_mb`
- `engine/worker.py` — パラメータ受け（`_resolve_block_swap_prefetch`）＋ doneイベントに `block_swap_prefetch_used` と `peak_vram_reserved_mb`（`torch.cuda.max_memory_reserved`）を**既存フィールドに加算的に**追加
- `services/pipeline_manager.py` — `/status` の acceleration ブロック（:163-203）に `block_swap_prefetch_available` を追加。**判定式は実ゲートと同一**（§8.5）。metadata.json（`_write_metadata` :825-841／`_write_chain_metadata` :729-746）に `block_swap_prefetch_used` と `peak_vram_reserved_mb` を記録
- `mcp_server/tools/generate.py` — パラメータ素通し追加
- `gradio_ui/ui.py` / `gradio_ui/i18n.py` / `gradio_ui/handlers.py` / `gradio_ui/batch.py` — Accelerationセクションにチェックボックス追加（attention_backendと同じreg/配線/i18nパターン）。**fused GGUF のdisabledチェックボックスは変更しない**

### フロントエンド

- `webui/src/shell/accelerationSettings.ts` — `blockSwapPrefetch: boolean` 追加。localStorage永続化をJSON形式へ移行（旧ベア文字列も受理）、加算的コントラクト（サーバー既定から動かした時だけ送る）維持
- `webui/src/shell/useAccelerationSettings.ts` — 項目追加。**localStorage書き出しは `useEffect` 側**（render中の副作用にしない。§10.2）
- `webui/src/shell/SettingsPanel.tsx` — Accelerationにトグル追加。**disabledは利用不可側（On）のボタンのみ**（sageの `disabled={isSage && sageDisabled}` :281 と同作法）。**fused/VAEのdisabledトグルは現状維持**
- `webui/src/i18n/strings.ts`・`webui/src/api/types.ts`・関連payload builder（`buildA2vChainPayload` 等は素通し確認のみ）
- テスト: `SettingsPanel.test.tsx` 等に追加。型検査は必ず `npm run typecheck`（`tsc -b`。`package.json:14`）

### ドキュメント

- `Nz-LTX23-backend/Docs/VERIFICATION_LOG.md` — §44新設（§43の構成踏襲）
- `Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md` — §3-49を「§3 再訪条件つきでクローズした項目」へ移動（§4への降格ではない。§12.2）
- `Nz-LTX23-backend/README.md` — Acceleration節に先読みblock swapの説明追記

---

## 4. アーキテクチャ

既存 `engine/transformer/block_swap_service.py` の `BlockSwapService` を拡張し、内部に「先読みエンジン」を**別モジュール**として切り出す。

```
engine/transformer/block_swap_service.py           # 既存。install/uninstall/_get_blocks/_patch_block を持つ。
                                                    #  → prefetch ON のときだけ _patch_block_prefetch を張る
engine/transformer/block_swap_prefetch.py  (新規)   # PrefetchEngine: CPU正本・pinned・stream・event・arena
engine/transformer/block_swap_prefetch_selfcheck.py (新規)  # .venv-engine 用 selfcheck
```

### 4.1 なぜ既存クラスを拡張するのか（新規クラスにしない理由）

1. **`_get_blocks()` が単一ソースとして共有されている。** `DitCpuLoadService.build_cpu_resident` が `self.block_swap._get_blocks(t)` と `self.block_swap.blocks_on_gpu` を直接読む（`dit_cpu_load_service.py:49-50`）。別クラスにすると `_install_block_swap`（`fast_video_pipeline.py:589-624`）で「どちらのサービスを渡すか」が分岐し、ブロック集合の食い違いという最悪の不整合を招く。
2. **サービスは常駐1インスタンス**で、`install()` がジョブ毎に呼ばれる（`block_swap_service.py:92-103` のリーク対策コメントが前提）。prefetch を別クラスにすると常駐インスタンスが2つになり、リーク対策ロジックが二重化する。
3. **ON/OFF はジョブ単位**。同一インスタンスが「今回のジョブは OFF」を選べる必要がある。`SageState` と同じ「1インスタンス・per-job set/reset」規律をそのまま使える。
4. 転送機構（CPU正本・pinned・stream・arena レイアウト）は分量が多く独立性も高いので、**そこだけ別モジュール**に出す（テスト容易性）。`BlockSwapService` は「どちらの forward を張るか」を決める薄い分岐に留める。

### 4.2 既存 OFF パスの温存方法

`_patch_block()`（`block_swap_service.py:153-218`）は**1文字も変えない**。ON のときだけ新設の `_patch_block_prefetch()` を張る。

```python
# block_swap_service.py install() 内、_patch_block ループ（現 :88-90）を置換
engine = None
if self.prefetch_requested:
    try:
        engine = PrefetchEngine(blocks, self.device, self.blocks_on_gpu, self._pinned_pool, self._xfer_stream)
        engine.prepare()          # CPU正本のスナップショット + pinned 確保（レイアウト検証含む）
    except Exception as exc:      # pinned 確保失敗 / レイアウト検証失敗など
        logger.warning("BlockSwap prefetch unavailable (%s) — falling back to the "
                       "synchronous path for this job", exc)
        engine = None
self._prefetch_engine = engine
self.last_prefetch_used = ("on" if engine is not None
                            else ("on->off" if self.prefetch_requested else "off"))

for idx, block in enumerate(blocks):
    if engine is None:
        self._patch_block(block, idx, blocks)           # ← 現行コードそのまま
    else:
        self._patch_block_prefetch(block, idx, engine)   # ← 新規
```

OFF のとき、追加で発生するのは `self.prefetch_requested` の bool 参照1回だけ。CPU正本のスナップショットも pinned 確保も起きない。

**`_pinned_pool` と `_xfer_stream` は `BlockSwapService` がジョブ跨ぎで保持し、`PrefetchEngine` へ都度渡す**（§7 E5・E10 参照）。`PrefetchEngine` 自体はジョブごとに使い捨て（`teardown_prefetch()` で参照を切る）。

### 4.3 退避コピー廃止を OFF 時にも適用するか → **しない（v1では ON 時のみ）**

| | ON時のみ適用（採用） | OFF時にも適用 |
|---|---|---|
| 効果 | prefetch ON のジョブだけ D2H 1.76s/pass を削減 | 全ジョブで削減（見込み） |
| リスク | 既存OFFパスが1文字も変わらない＝完全な A/B ベースラインが残る | OFF のはずが挙動変更。実機ゲートの対比較で差分の帰属が不能になる |
| プロジェクト規律 | sage/NAG/VSF が守ってきた「OFF はバイト同一」を維持 | 規律違反 |
| ロールバック | トグルを off にするだけ | コード revert が必要 |

オーナー確定事項4（§0）とも一致。効果が確認できたら台帳へ将来課題として起票する（§12.2 の §4-26相当）。

---

## 5. 先読みアルゴリズム

### 5.1 用語と定数・常駐ウィンドウ

```python
# engine/transformer/block_swap_prefetch.py
_ALIGN = 512                # arena 内の各テンソル開始オフセットのアライン（CUDA caching allocator
                            # の 512B 保証と一致。view(dtype) の storage_offset 可分条件を無条件に満たす）
_NUM_STAGING_SLOTS = 2      # pinned ステージング枠数
_MAX_ISSUES_PER_STEP = 2    # 1ステップで新規発行する転送の上限（cold start のランプ制御）
```

常駐ウィンドウは**循環させない**（現行と同一の `W(idx) = { j | idx <= j < min(idx + blocks_on_gpu, total) }`）。

理由: stage1 → stage2 の間に spatial upsampler が走る（本パイプライン最大の VRAM 山。`chunked_upsample` オプションが存在するのはまさにこのため）。現行の非循環ウィンドウは、パス末尾で自然に `{47}` の1ブロック（約340MB）まで枯れる。循環させると末尾でも常時8ブロック（約2.7GB）が残り、upsample と decode の直前に +2.4GB を積んで OOM を誘発しうる。循環をやめた場合の損失は**パス先頭のブロック0の転送待ち約16ms/pass のみ**（後述のランプで残り7ブロックは隠蔽される）。11パスで約0.18秒＝全体の0.07%。**払う価値がない。**

これにより VRAM プロファイル・末尾の枯れ方・stage 遷移時の常駐量が現行とバイト単位で同じになり、実機ゲートの VRAM 条件が構造的に保証される。定常時の常駐は「ウィンドウ＋前パスのブロック47」で**最大 bs+2**（bs+1 ではない。ブロック47が前パスから残留するため）。

> 将来、循環先読みを試したくなったら `_CYCLIC = True` の1定数で切り替えられる形にしておく。ただしその場合は denoise 終了時に `engine.release_all()` を呼ぶフックが必須になる（現状そのフックは存在しない）ことをモジュール docstring に明記する。

### 5.2 ステートマシンと本体擬似コード（レビュー修正反映済み・最終形）

```python
class _BlockState:
    __slots__ = ("arena", "views", "event", "slot", "acquired")
    # arena   : torch.Tensor  1D uint8 on CUDA（このブロックの全テンソルを詰めた連続領域。
    #           ★ compute stream（current stream）上で確保する＝owning streamはcompute）
    # views   : list[(module, name, is_param, tensor)]  arena 上に構築した device テンソル
    # event   : torch.cuda.Event  H2D 完了（xfer streamでrecord）を示す
    # slot    : int  使用した pinned 枠の index
    # acquired: bool 計算streamが event を wait 済みか
```

```python
def on_block_forward(self, idx: int) -> None:
    """swapped_forward_prefetch の先頭で呼ぶ。ここを抜けた時点で
       ブロック idx の全テンソルは GPU 上にあり、計算stream から安全に読める。"""
    total, bs = len(self._blocks), self.blocks_on_gpu

    # ── (1) 自分がまだ発行されていなければ、今ここで発行（cold start / 先読みミス）
    if idx not in self._state:
        self._stats["sync_misses"] += 1
        self._issue(idx)

    # ── (2) ウィンドウ内の未発行ブロックを最大 _MAX_ISSUES_PER_STEP 本まで発行。
    #        (3) の計算stream waitの *前* に置く＝コピーエンジンに先に仕事を積む。
    issued = 0
    for j in range(idx + 1, min(idx + bs, total)):
        if j in self._state:
            continue
        self._issue(j)
        issued += 1
        if issued >= _MAX_ISSUES_PER_STEP:
            break

    # ── (3) S2: 計算streamが転送完了eventをwait（RAW）。arenaはcompute所有なので
    #        record_stream は不要（解放後の再利用はstream順序で安全。§5.3）
    st = self._state[idx]
    if not st.acquired:
        compute = torch.cuda.current_stream(self.device)
        compute.wait_event(st.event)
        st.acquired = True

    # ── (4) ウィンドウから出たブロックを退避（D2H ゼロ、CPU正本へ付け替え）
    if idx - 1 >= 0:
        self._release(idx - 1)

    # ── (5) パス先頭で前パスの統計をログ（1パス1行）
    if idx == 0:
        self._log_and_reset_stats()
```

```python
def _issue(self, idx: int) -> None:
    """ブロック idx の CPU正本 → pinned → (転送stream) → GPU arena を発行する。
       ホスト側の memcpy はここで同期的に走る（約14ms／ブロック）。
       ★ arena はここで **compute stream（current stream）上に確保**する
       （レビューCRITICAL①反映。転送stream上で確保すると caching allocator の
       stream別プール分断で、stage1最終パス終了時に約3GBがxferプールに滞留し
       spatial upsampler直前の共有メモリスピルを誘発する）。"""
    lay = self._layout[idx]
    slot = self._acquire_slot()                      # 枠の直近H2Dの完了を **host wait**（S1）
    pinned = self._pinned[slot]

    # ── host memcpy: CPU正本 → pinned（全部 uint8 領域で行う＝dtype/alignment 非依存）
    for src_u8, off, nbytes in zip(self._master_u8[idx], lay.offsets, lay.sizes):
        pinned[off:off + nbytes].copy_(src_u8)       # CPU→CPU、約27GB/s

    # ── device 側: arena は compute stream 上で確保する（★修正の核心）
    compute = torch.cuda.current_stream(self.device)
    arena = torch.empty(lay.total_bytes, dtype=torch.uint8, device=self.device)
    alloc_evt = torch.cuda.Event()
    alloc_evt.record(compute)

    # ── S1b: 転送streamが alloc event を wait してから copy_ を発行（WARハザード回避）。
    #        既enqueue計算の追い越し防止＝転送streamが「確保完了」より前にarenaへ書かない。
    self._xfer.wait_event(alloc_evt)
    with torch.cuda.stream(self._xfer):
        arena.copy_(pinned[:lay.total_bytes], non_blocking=True)
    copy_evt = torch.cuda.Event()
    copy_evt.record(self._xfer)

    views = self._build_views(idx, arena)             # arena をスライスして dtype/shape を復元
    self._install_views(views)                        # module._parameters/_buffers を差し替え
    self._state[idx] = _BlockState(arena, views, copy_evt, slot, acquired=False)
    self._slot_event[slot] = copy_evt
    self._stats["issued"] += 1
```

```python
def _acquire_slot(self) -> int:
    slot = self._next_slot
    self._next_slot = (self._next_slot + 1) % _NUM_STAGING_SLOTS
    evt = self._slot_event[slot]
    if evt is not None and not evt.query():
        t0 = time.perf_counter()
        evt.synchronize()                # ★ host-blocking。枠の直前H2Dが pinned を
        self._stats["slot_wait_ms"] += (time.perf_counter() - t0) * 1e3
                                         # 読み終わるまでホストは上書きしてはならない（S1）
    return slot
```

```python
def _release(self, idx: int) -> None:
    """CPU正本へ付け替えるだけ。D2H 転送はゼロ。"""
    st = self._state.pop(idx, None)
    if st is None:
        return
    for (mod, name, is_param), cpu_t in zip(self._layout[idx].slots, self._master[idx]):
        if is_param:
            mod._parameters[name].data = cpu_t
        else:
            mod._buffers[name] = cpu_t
    # st.arena / st.views の最後の参照が消える → allocator へ返却。
    # arena は compute stream 所有＝解放後の再利用は同一 stream 内の順序性で安全
    # （xfer側から見ればS1bのwait_eventがあるので次のarenaの書き込みはこのカーネル完了後）。
    del st
```

### 5.3 同期点の一覧（これがすべて。4点のみ）

| # | 同期 | API | 守るもの |
|---|---|---|---|
| S1 | 枠再利用前の host wait | `slot_event[k].synchronize()` | pinned への host 書き込みが、直前のH2D（DMA読み出し）を追い越さない（WAR、host↔device） |
| S1b | 転送streamがalloc完了を待つ | `xfer.wait_event(alloc_evt)` | arena確保（compute stream）の完了より前に、転送streamがそのメモリへ書き込まない（WAR、device内・別stream） |
| S2 | 計算stream が転送完了を待つ | `compute.wait_event(copy_evt)` | ブロックの重みを読むカーネルが、H2D 完了後に走る（RAW） |
| S4 | install/teardown 時 | `self._xfer.synchronize()` | 前ジョブの in-flight 転送が残ったまま pinned/arena を捨てない |

**`record_stream` は使わない。** arena は compute stream 上で確保するため owning stream が最初から compute であり、free 時に allocator が「同じ owning stream からの再割当は stream 順序で安全」と仮定してよい。S1b の `wait_event` だけで WAR ハザードが解消される。

### 5.4 CUDA caching allocator の stream 安全性（棄却された原案の技術的詳細）

原案（転送stream上でarena確保＋`record_stream`）が破綻する理由を実装者が理解できるよう記録する:

```
t=0  arena_A を xfer 上で確保、H2D
t=1  compute が arena_A を読むカーネルを大量に enqueue（まだ完了していない）
t=2  _release → arena_A の refcount 0 → allocator が xfer プールへ即返却
t=3  次の _issue が同じ物理ブロックを arena_B として受け取り、xfer 上で上書き
     → t=1 のカーネルが読んでいる最中に破壊される（サイレントな数値破損）
```

`record_stream(compute)` でこの t=3 のハザードは消せるが、**allocator のプール自体が xfer 所有のまま**であるため、stage1 最終パス終了時に約8ブロック分（≈3.0-3.3GB）が xfer プールに滞留する。これは compute 側（spatial upsampler・VAE decode）の割り当てには再利用されない（BlockComparator が stream を第1キーにしているため）。16GB機で共有メモリスピルを誘発するのが実測ベースの結論。

**修正版（本書採用）**: arena を compute stream 上で確保する（owning stream = compute）。free 側は「同じ owning stream からの再割当は安全」という allocator の前提どおりに動く。転送stream の書き込みが確保完了を追い越すハザードは `xfer.wait_event(alloc_evt)`（S1b）で解消する。オーバーラップ損失はほぼない——発行は `idx+bs-1` ブロック先（定常時で6〜7ブロック分≒約3.2秒の余裕）に対して、確保イベントの record は即座（マイクロ秒オーダー）だからである。

固定プール方式（bs+1 枠 ×最大ブロックサイズを事前確保）も検討したが、最大サイズ×9 ≒3.3GB を常時確保することになり「+400MB以内」ゲートを割るため不採用。

**断片化について。** arena のサイズはブロック毎に決まっており、48種類の値が周期的に繰り返される。caching allocator は同サイズの解放ブロックを再利用するので、数パス走ればプールは収束する。実機ゲートで `torch.cuda.memory_reserved` の推移を確認する（§14 S6・G12相当）。

### 5.5 なぜ「ウィンドウ末尾を先に発行する」だけで十分隠れるのか

ステップ `idx` で発行するのは `idx+bs-1`（定常時）＝**bs-1 = 7 ブロックぶんの計算時間（約3.2秒）先**に消費される転送である。1ブロックの H2D は約16ms（370MB / 23.4GB/s）。余裕は200倍。ランプ中も `_MAX_ISSUES_PER_STEP=2` で4ステップ以内に窓が埋まる。

パス先頭（idx=0、各パス）: `W(0)={0..7}` の全メンバが未発行。(1)で0を発行、(2)で1,2を発行、(3)で0を待つ。露出するのは「ブロック0の host memcpy 14ms + H2D 16ms + 枠待ち16ms ≒ 46ms」。11パスで0.5秒＝0.19%。

---

## 6. GGMLQuantizedTensor を保ったままの手動デバイス間コピー

### 6.1 S0スパイク: `inference_mode()` 下での `_make_subclass` 可否検証（実装着手前に必須）

`GGMLQuantizedTensor.__new__` は内部で `torch.Tensor._make_subclass(cls, raw_bytes)` を呼ぶ（`quant_service.py:447-457`）。

**元設計は「現行の `GGMLQuantizedTensor.to()` が既に `inference_mode()` 内でこの経路を通っているので新しいリスクではない」としていたが、これはレビューで誤りと判定された。** `GGMLQuantizedTensor.__new__` が実際に呼ばれるのは GGUF ロード時のみで、そこは `inference_mode()` の**外側**である。ジョブ中の `install()`（先読みが `_make_subclass` を呼ぶ場面）は `inference_mode()` **内**で走るため、前例が無い。実装前に以下のスパイクで可否を確定する（リポジトリ変更なし、使い捨てスクリプト）:

```python
# S0スパイク（.venv-engine で実行、10行検証）
import torch
from engine.gguf.quant_service import GGMLQuantizedTensor  # 実際のimportパスは実装時に確認

with torch.inference_mode():
    raw = torch.empty(1024, dtype=torch.uint8, device="cuda")
    try:
        t = torch.Tensor._make_subclass(GGMLQuantizedTensor, raw)
        t._ggml_type = "q4_k"
        t._float_shape = (16, 32)
        print("OK: _make_subclass works under inference_mode")
    except Exception as exc:
        print("NG:", exc)
        t2 = raw.as_subclass(GGMLQuantizedTensor)
        t2._ggml_type = "q4_k"
        t2._float_shape = (16, 32)
        print("fallback as_subclass:", type(t2), t2._ggml_type)
```

- **可なら**: `torch.Tensor._make_subclass(GGMLQuantizedTensor, raw)` を §6.3 の `_build_views` でそのまま使う。
- **不可なら**: `raw.as_subclass(GGMLQuantizedTensor)` ＋ `_ggml_type`/`_float_shape` 属性の直接代入で代替する（`_make_subclass` を経由しないぶん shallow-copy 保証は自前で担保する必要があるが、`view` 操作の結果は sizes/strides/storage_offset を保つのでリスクは小さい）。

**完了条件**: 可否確定と採用経路の決定。

> **S0スパイク結果（2026-08-01実施、engine venv実機、全パターンSUCCESS）**:
> - (a) `torch.inference_mode()`内で `GGMLQuantizedTensor(raw_cuda_uint8, _GGML_Q4_K, (4096,4096))` の構築に成功。`.shape`のfloat形状偽装も正常。生成テンソルは`is_inference()=True`だが推論専用パスでは無害。
> - (b) `raw.as_subclass(GGMLQuantizedTensor)`+属性代入の代替経路も成功（採用は(a)）。
> - (c) meta buffer化したLinearへの `lin._buffers["weight"] = qt` 直接差し替え＋`dequantize_ggml_tensor`→`F.linear` がinference_mode内で動作確認済み。
> - (d) inference_mode内で構築したテンソルへのコンテキスト外アクセス（次ジョブ相当）も問題なし。
> - **採用決定: (a) 既存ロード時と同一のコンストラクタ経路 `GGMLQuantizedTensor(raw, ggml_type, float_shape)` を使用。代替経路(b)は不要。**
> - 検証スクリプト: `scratchpad/s0_spike_inference_mode.py`（リポジトリ外）。

なお `p.data = <inference tensor>` と `_buffers[name] = <inference tensor>` は現行 `blk.to(device)` が毎スワップ実行しているので前例あり・安全（これはレビューでも否定されていない）。

### 6.2 install 時のスナップショット（`PrefetchEngine.prepare()`）

ブロック内の全テンソルを**per-moduleの `_parameters` / `_buffers` 走査**で列挙する。`named_parameters(recurse=True)` を使わない理由:

- `named_parameters` / `named_buffers` は既定 `remove_duplicate=True` で共有テンソルを1回しか返さないため、**復元すべきスロットを取りこぼす**。
- 所有モジュールと属性名が必要（復元は `mod._parameters[name].data = ...` / `mod._buffers[name] = ...`）。
- `dit_cpu_load_service.py:71-78` が同じ作法を採っており、単一の先例に従える。

```python
def _enumerate_slots(block: nn.Module):
    for mod in block.modules():                 # block 自身も含む → scale_shift_table を拾う
        for name, p in list(mod._parameters.items()):
            if p is not None:
                yield mod, name, True, p
        for name, b in list(mod._buffers.items()):
            if b is not None and isinstance(b, torch.Tensor):
                yield mod, name, False, b
```

これで拾えるもの: GGUF圧縮重み（`GGMLQuantizedTensor` buffer, `quant_service.py:621-625`）、Linear の bias（Parameter）、各 norm の weight/bias、`scale_shift_table` / `audio_scale_shift_table`（block 直付け Parameter）、IC-LoRA / Style LoRA の `_ic_lora_A_n` / `_ic_lora_B_n`（`persistent=False` buffer, `ic_lora_common.py:181-182`）。

### 6.3 レイアウト計算（`prepare()` 内、install 1回きり）

```python
def _plan(block) -> _BlockLayout:
    slots, kinds, offs, sizes = [], [], [], []
    off = 0
    for mod, name, is_param, t in _enumerate_slots(block):
        assert t.device.type == "cpu", "prepare() must run with all blocks on CPU"
        if isinstance(t, GGMLQuantizedTensor):
            # ★ .numel()/.shape は float 形状を偽装する（quant_service.py:459-471）。
            #    真のバイト数は subclass を外してから測る（同 :645 の作法）。
            #    ★ MINOR10反映: reshape(-1) を view(uint8) より先に通す（0次元テンソルで
            #    view(dtype) が RuntimeError になるのを避ける）。GGML側は既にuint8実storage
            #    のため通常は影響しないが、他ブランチと統一した作法にする。
            raw = t.as_subclass(torch.Tensor).reshape(-1).view(torch.uint8)
            nbytes = raw.numel()
            kind = ("ggml", t._ggml_type, tuple(t._float_shape))
            src_u8 = raw
        else:
            src = t if t.is_contiguous() else t.contiguous()   # 非連続は install 時に一度だけ実体化
            nbytes = src.numel() * src.element_size()
            kind = ("plain", src.dtype, tuple(src.shape))
            # ★ MINOR10反映: reshape(-1) を先に通す
            src_u8 = src.reshape(-1).view(torch.uint8)
        slots.append((mod, name, is_param)); kinds.append(kind)
        offs.append(off); sizes.append(nbytes)
        off = (off + nbytes + _ALIGN - 1) // _ALIGN * _ALIGN     # 512B アライン
    return _BlockLayout(slots, kinds, offs, sizes, total_bytes=off)
```

CPU正本 `self._master[idx]` には**元のテンソルオブジェクトそのもの**（`GGMLQuantizedTensor` は subclass のまま、Parameter は `.data` の中身）を保持する。`self._master_u8[idx]` には上の `src_u8` をキャッシュ（毎ステップの `as_subclass/reshape/view` を省くため）。

> `.contiguous()` を取ったケースは、`_master` に**その連続化後のテンソル**を入れる（元と別オブジェクトになるが値は同一で、以後は常にそちらを使う）。LTXの重みは全て連続なので実際には発生しない見込みだが、発生したら INFO ログを1行出す。

### 6.4 device 側ビューの構築（`_build_views`）

```python
def _build_views(self, idx, arena):
    lay = self._layout[idx]
    out = []
    for (mod, name, is_param), kind, off, n in zip(lay.slots, lay.kinds, lay.offsets, lay.sizes):
        raw = arena[off:off + n]                       # 1D uint8, storage_offset = off (512の倍数)
        if kind[0] == "ggml":
            _, ggml_type, float_shape = kind
            # quant_service.py:483-489 と同一構成でメタを再付与する。
            # §6.1のS0結果に応じて _make_subclass / as_subclass のどちらかを使う。
            dev_t = GGMLQuantizedTensor.__new__(
                GGMLQuantizedTensor, raw.view(torch.uint8), ggml_type, float_shape,
            )
        else:
            _, dtype, shape = kind
            dev_t = raw.view(dtype).view(shape)        # §6.5 のアライン条件を満たす
        out.append((mod, name, is_param, dev_t))
    return out
```

- こうして作った device 側の `GGMLQuantizedTensor` は、`ggml_linear_forward`（`quant_service.py:634-660`）の `w.as_subclass(torch.Tensor).view(torch.uint8)` / `w._ggml_type` / `w._float_shape` をすべて満たす。

### 6.5 `view(dtype)` の合法性（install 時に一度だけ検証する）

`Tensor.view(dtype)` は「`storage_offset()` が elem-size 比で割り切れること」「`size(-1)` が比で割り切れること」「`stride(-1)==1`」を要求する。`_ALIGN=512` により `off % 2 == off % 4 == 0` は自明、`n` は `numel * element_size` なので比で割り切れる、スライスは連続。よって常に合法。ただし**0次元テンソルは `view(dtype)` が無条件にRuntimeErrorを出す**ため、§6.3 のとおり `reshape(-1)` を先に通す（MINOR10）。

とはいえ**silent に壊れる余地を残さない**ため、`prepare()` の最後に検証を入れる:

```python
def _validate(self):
    """全ブロック分のビュー構築を、実際の GPU 確保なしにダミー arena で1回試す。
       ここで例外が出たら prefetch は無効化して従来同期パスへ落ちる（install 側で捕捉）。
       ★ MINOR11反映: torch.cuda.empty_cache() はここで呼ばない
       （ON経路だけをallocator的に変えてしまい、G3/G5のVRAM比較の帰属を汚す）。"""
    targets = self._layout if _VALIDATE_ALL else self._layout[:1]
    for idx, lay in enumerate(targets):
        dummy = torch.empty(lay.total_bytes, dtype=torch.uint8, device=self.device)
        views = self._build_views(idx, dummy)
        for (mod, name, is_param, dev_t), kind in zip(views, lay.kinds):
            if kind[0] == "ggml":
                assert isinstance(dev_t, GGMLQuantizedTensor)
                assert dev_t._ggml_type == kind[1] and tuple(dev_t._float_shape) == kind[2]
            else:
                assert dev_t.dtype == kind[1] and tuple(dev_t.shape) == kind[2]
        del views, dummy
```

（48ブロック×370MBを順に確保/解放するので約1〜2秒。install はジョブ1回なので許容だが、**推奨は先頭1ブロックのみ**（`_VALIDATE_ALL = False` を既定にする）——全ブロックは同一クラス・同一構造で、違いうるのは ggml_type とバイト数だけであり、レイアウト計算は共通コードだから。）

### 6.6 差し替え / 復元

```python
def _install_views(self, views):
    for mod, name, is_param, dev_t in views:
        if is_param:
            mod._parameters[name].data = dev_t     # Parameter オブジェクトの同一性を保つ
        else:
            mod._buffers[name] = dev_t             # dit_cpu_load_service.py:77 と同じ作法
```

復元（`_release`）は §5.2 の通り、`self._master[idx]` の CPU テンソルを同じスロットへ書き戻すだけ。

**差し替えのタイミングは「発行時」**（転送完了前）。この時点で arena の中身は未定義だが、それを読むカーネルは S2（`compute.wait_event`）の後にしか enqueue されないので安全。denoise 中に module のテンソルを読む他の主体は存在しない（NAG/VSF/sage は forward/attention_function 属性しか触らない）。

---

## 7. エッジケース

| # | ケース | 挙動 | 実装位置 |
|---|---|---|---|
| E1 | `blocks_on_gpu >= total`（swap 不要） | `install()` は `:72-77` で早期 return。**`PrefetchEngine` を作る前**に return するので pinned も確保しない。`last_prefetch_used = "off"` | `block_swap_service.py:72-77` の直後 |
| E2 | `blocks_on_gpu == 0` | `:62-64` で早期 return。同上 | 同 |
| E3 | `_get_blocks()` が空 | `:66-69` で警告 return。同上 | 同 |
| E4 | pinned 確保失敗（`torch.empty(..., pin_memory=True)` が RuntimeError） | `PrefetchEngine.prepare()` 内で捕捉して再送出 → `install()` の try/except が拾い、**その transformer に対しては従来同期パスを張る**。ジョブは落とさない。`last_prefetch_used="on->off"` | `block_swap_service.install()` |
| E5 | サービス常駐インスタンスの pinned 使い回し | **`BlockSwapService` が pinned プールと `torch.cuda.Stream` を保持し、ジョブ跨ぎで再利用する。** 370MB×2 の cudaHostAlloc は100ms級で、長時間稼働後は失敗しうる。ジョブ毎に確保/解放すると「ジョブ7で急に prefetch が効かなくなる」が起きる。プールは**grow-only**（必要サイズが増えたら再確保）。プロセス寿命ぶん約740MBのRAM占有はREADME/VERIFICATION_LOGに明記 | `BlockSwapService._pinned_pool` |
| E6 | ブロックサイズ不揃い | pinned 枠は `max(total_bytes over blocks)` で確保。arena は**ブロック毎の実サイズ**で確保（無駄なし） | `prepare()` |
| E7 | stage1 → stage2 遷移 | 同一 transformer / 同一 `PrefetchEngine` を継続使用。非循環ウィンドウなので stage1 最終パス終了時の常駐は `{47}` の1ブロックのみ＝upsamplerのVRAM山を現行と同条件で通過。stage2 の先頭 pass は cold start 相当（約46ms） | 追加コード不要 |
| E8 | chain 生成（複数クリップ） | `chain_pipeline.py:653` で transformer を1回だけ構築＝`install()` 1回、`PrefetchEngine` 1個。全セグメント・全タイルで共有。各パス末尾でウィンドウが枯れるので、セグメント間の `cleanup_memory()` / `empty_cache()` とも競合しない | 追加コード不要 |
| **E9** | **ジョブ毎の後片付け（リーク防止）★修正** | ~~install()冒頭でteardown~~ は不採用。**`generate()`/`generate_chain()` の `finally`（`_reset_block_swap_prefetch_job()` の隣）で `BlockSwapService.teardown_prefetch()` を呼ぶ**: ①`self._xfer.synchronize()`（in-flight完走を待つ）②旧`PrefetchEngine`の`_state`/`_master`/`_master_u8`/`_layout`参照を切り、`self._prefetch_engine = None`（旧transformer自体はパイプライン側で捨てられるので module 復元は不要。arenaの参照だけ切る）。**pinnedプールと`torch.cuda.Stream`はサービス側に残して再利用**（stream作り直しは不要）。`install()` 冒頭にも**冪等な安全網**として同じ `teardown_prefetch()` 呼び出しを残す（プロセス異常終了などで finally を経由できなかった場合の保険。通常運用では2回目呼び出しはno-op） | `fast_video_pipeline.py` の generate/generate_chain finally、`block_swap_service.py` install() 冒頭（安全網） |
| E10 | stream / event のリーク | `torch.cuda.Stream` はサービス生成時に1本だけ lazily 作る（`self._xfer or torch.cuda.Stream(device=self.device)`）。`Event` はブロック毎に作って `_release` で捨てる（Python GC が cudaEventDestroy する）。イベント数の上限は `_state` のサイズ ≦ bs+2 | `PrefetchEngine.__init__` は stream を受け取るだけ |
| E11 | 例外発生時の stream 状態 | forward 中に例外が飛んでも、転送streamには「完了する転送」しか積んでいない（barrier待ちで詰まる構造がない）。**ジョブの `finally` が呼ぶ `teardown_prefetch()` が `xfer.synchronize()` するので、確実に清算される。** `_release` を finally で呼ぶ必要はない（arenaはPython参照で管理され、`teardown_prefetch()`で切れる） | 同上 |
| E12 | `uninstall()`（本番未使用） | `teardown_prefetch()` を呼んでから既存処理。pinnedは解放する（`self._pinned_pool = None`） | `block_swap_service.py:105-121` |
| E13 | CUDA 非搭載環境 | `install()` で `self.device.type != "cuda"` なら prefetch を無効化（`"on->off"`）。selfcheckがCPUで走る場合の保険 | `install()` |
| E14 | 共有テンソル（weight tying） | per-module走査は重複排除しないので、両スロットがarena内の別領域を指す。値は同一なので出力は不変、VRAMだけ余分。LTXブロックにtyingは無いが、検出したらinstall時にWARNINGを1行出す（`id()`の重複カウント） | `_plan()` |
| E15 | 非連続テンソル | `prepare()` で一度だけ `.contiguous()` 化し、それをCPU正本にする（§6.3） | `_plan()` |
| E16 | LoRA有無の切り替え | LoRA attach は install より前に完了している（`quant_service.py:773-780` → `fast_video_pipeline.py:610-611`）。`_ic_lora_A_n` / `_B_n` は `_buffers` に居るので自動的にレイアウトへ入る。LoRA無しジョブでは `detach_ic_loras` により `_buffers` から消えている（`ic_lora_common.py:220-221`）ので、レイアウトも自然に縮む | 追加コード不要 |
| E17 | FP8混在（`gguf_per_layer_quant=False`のレガシー経路） | 明示ステートマシンにより「一部だけGPU」という状態が構造的に発生しない | 設計上自動 |
| E18 | `blocks_on_gpu` が1 | ウィンドウ=`{idx}`のみ→`_issue(idx)`が毎回(1)で走る＝完全同期（先読みゼロ）。`sync_misses`が全ブロックで立つのでログから一目で分かる。異常ではないので警告のみ | `_log_and_reset_stats` |

---

## 8. API 配線（attention_backend と同じ流儀）

### 8.1 `api/models.py`

`GenerateRequest` の Acceleration ブロック（`:97-126`）に追記。`attention_backend`（`:112`）の直後、モック2件（`:114-126`）の**前**:

```python
    # block_swap_prefetch: block swap（VRAM を節約するために transformer の
    # ブロックを CPU と GPU のあいだで出し入れする仕組み）の転送を、計算とは
    # 別の CUDA stream で先回りさせて待ち時間を隠す。あわせて GPU→CPU の
    # 退避コピーを廃止する（重みは推論中に一切変化しないので、CPU 側の正本を
    # 保持して GPU 側は捨てるだけでよい）。実測 768p/257f で約11〜13%短縮。
    # 【重要】attention_backend と違い、**生成結果は変わらない**（転送の
    # 方式だけを変えるので、同一シードならビット単位で同一になる）。
    # block swap が無効な設定（vram.block_swap=false / blocks_on_gpu=0 /
    # blocks_on_gpu が全ブロック数以上）では黙って no-op になる。実際に
    # 効いたかどうかは metadata.json の block_swap_prefetch_used で確認できる。
    # 利用可否は GET /status の acceleration.block_swap_prefetch_available。
    block_swap_prefetch: bool = False
```

`GenerateChainRequest`（`:371-378`）にも `attention_backend` の直後へ同じフィールドを追加（コメントは「詳細は GenerateRequest の同名フィールドを参照」）。

`to_clip_request()`（`:605-647`）: `attention_backend=self.attention_backend,`（`:634`）の直後に `block_swap_prefetch=self.block_swap_prefetch,` を追加。**必須**——docstring `:616-622` が説明する「転記漏れ = 記録の嘘」の対象になる。

### 8.2 `services/ltx_runner.py`

- 単発: `:1365-1366` の直後に
  ```python
  if request.block_swap_prefetch:
      payload["block_swap_prefetch"] = True
  ```
  （既定 False のときキーを増やさない＝ペイロード完全一致テスト群が無改修で通る）
- chain: `:1566-1567` の直後に同じブロック（`chain.block_swap_prefetch`）
- `GenerationOutcome`（`:231-247`）に `block_swap_prefetch_used: str | None = None` と `peak_vram_reserved_mb: int | None = None` を追加（既存の `peak_vram_mb` は残す。**加算**）
- done 受領: `:1413` / `:1600` の直後に `block_swap_prefetch_used=event.get("block_swap_prefetch_used"), peak_vram_reserved_mb=event.get("peak_vram_reserved_mb"),`

### 8.3 `engine/worker.py`

- プロトコル docstring（`:18-47`, `:52-62`）に `block_swap_prefetch` と `done.block_swap_prefetch_used` / `done.peak_vram_reserved_mb` を追記
- `_resolve_attention`（`:386-427`）の隣に新設:
  ```python
  def _resolve_block_swap_prefetch(msg: dict) -> bool:
      """ジョブの block_swap_prefetch を解決する。キー欠落 -> False。
      不正な型は fail-loud にしない（bool() で丸める）——これは速度の
      つまみであって、生成結果の正しさの前提ではない（_resolve_attention の
      sage 降格と同じ規律。ただし列挙値ではないので未知値という概念がない）。"""
      return bool(msg.get("block_swap_prefetch", False))
  ```
- `_attention_used`（`:430-443`）の隣に:
  ```python
  def _block_swap_prefetch_used() -> str:
      """"off" / "on" / "on->off"（要求したが block swap 未インストール・
      pinned 確保失敗などで降格）。パイプラインの記録を読む。"""
      assert _PIPE is not None
      return _PIPE.block_swap_prefetch_used()

  def _peak_vram_reserved_mb() -> int:
      """torch.cuda.max_memory_reserved() 由来。既存の peak_vram_mb
      （max_memory_allocated）はstream別プール分断・reserved増を検知できないため、
      本改修のVRAMリスクを見るための追加指標として加算する（既存フィールドは置換しない）。"""
      return int(torch.cuda.max_memory_reserved() // (1024 * 1024))
  ```
- `_do_generate`（`:483` 付近）: `attention, attn_degraded = _resolve_attention(msg)` の直後に `bs_prefetch = _resolve_block_swap_prefetch(msg)`、ログ（`:489`）に ` bsprefetch={bs_prefetch}` を追加、`_PIPE.generate(...)`（`:510`）に `block_swap_prefetch=bs_prefetch,`、`_emit("done", ...)`（`:522`）に `block_swap_prefetch_used=_block_swap_prefetch_used(), peak_vram_reserved_mb=_peak_vram_reserved_mb(),`
- `_do_generate_chain`（`:616`, `:624`, `:648`, `:657-663`）に同じ4点

### 8.4 `engine/pipeline/fast_video_pipeline.py`

- `__init__` の Sage 状態の隣（`:142-148`）に:
  ```python
  # ── Block swap 先読み（転送隠蔽）の per-job 状態 ────────────────────
  # SageState と同じ寿命規律：1インスタンスがパイプラインに常駐し、
  # generate()/generate_chain() の set/finally-reset で1ジョブにスコープする。
  # 実体の状態は BlockSwapService が持つ（そこが install() のたびに読む）。
  self._block_swap_prefetch_requested = False
  self._block_swap_prefetch_used = "off"
  ```
- `_set_sage_job`（`:367-387`）の隣に:
  ```python
  def _set_block_swap_prefetch_job(self, enabled: bool) -> None:
      """このジョブの先読み要求を立てる。NEVER raises（_set_sage_job と同じ理由：
      これは try/finally の外側で走るので、ここで例外を出すと reset を飛ばして
      常駐 worker の次ジョブへ要求が漏れる）。block swap が未インストールなら
      この bool はどこからも読まれず、自動的に no-op になる。"""
      self._block_swap_prefetch_requested = bool(enabled)
      svc = getattr(self, "_block_swap_service", None)
      if svc is not None:
          svc.prefetch_requested = self._block_swap_prefetch_requested
          # ★ MINOR13反映: ジョブ開始時にNoneへリセットする。install()前に失敗した
          #   ジョブが前ジョブの last_prefetch_used を読んでしまう事故を防ぐ。
          svc.last_prefetch_used = None

  def block_swap_prefetch_used(self) -> str:
      """直前に完了したジョブで先読みが実際に効いたか："off" / "on" / "on->off"。
      attention_used と同じく、リセット時のスナップショットを返す。"""
      return self._block_swap_prefetch_used

  def _reset_block_swap_prefetch_job(self) -> None:
      svc = getattr(self, "_block_swap_service", None)
      if svc is not None:
          # svc.last_prefetch_used が None のまま＝install()に到達しなかった
          # （例外・早期return）。要求していれば "on->off"、していなければ "off"。
          self._block_swap_prefetch_used = svc.last_prefetch_used or (
              "on->off" if self._block_swap_prefetch_requested else "off")
          svc.teardown_prefetch()   # ★ E9: ここでジョブ単位の後片付け（xfer.synchronize＋参照切り）
      else:
          self._block_swap_prefetch_used = (
              "on->off" if self._block_swap_prefetch_requested else "off")
      self._block_swap_prefetch_requested = False
      if svc is not None:
          svc.prefetch_requested = False
  ```
- `generate()`: シグネチャ（`:1102` の直後）へ `block_swap_prefetch: bool = False`、`self._set_sage_job(attention_backend)`（`:1127`）の直後に `self._set_block_swap_prefetch_job(block_swap_prefetch)`、`finally`（`:1170` の直後）に `self._reset_block_swap_prefetch_job()`
- `generate_chain()`: `:1192` の直後にシグネチャ追加、`:1236` の直後にset、`:1263` の直後にreset。docstring（`:1226-1232`）に「attention_backendと同じく最外の入口で立てる」旨を1段落追記
- `_install_block_swap`（`:589-624`）: `service = BlockSwapService(...)` の直後に
  ```python
  service.prefetch_requested = self._block_swap_prefetch_requested
  ```
  （create時点では常にFalse。per-jobの値は`_set_block_swap_prefetch_job`が入れる）

> **chain の非対称性について**: NAG は `run_chain` の中で立てるが、attention_backend は最外で立てる（`fast_video_pipeline.py:1226-1232` の解説）。先読みも「エンコード順序の制約がない」ので**attention_backend側**に揃える。相互参照コメントを両方に入れる。

### 8.5 `/status`

`services/pipeline_manager.py:163-203`（★CRITICAL②修正版）:

```python
    ATTENTION_BACKENDS = ["sdpa", "sage"]

    def acceleration_status_block(self) -> dict:
        return {
            "attention_backends": list(self.ATTENTION_BACKENDS),
            "sage_available": self._sage_available(),
            # 先読み block swap の利用可否は「実ゲート（block swap が実際に
            # 効くか）」と完全同一の式で判定する。表示専用の low_vram.block_swap
            # （bool）は real 経路のどこからも読まれておらず既定構成では false
            # なので判定式に使わない（レビューCRITICAL②）。
            "block_swap_prefetch_available": self._block_swap_prefetch_available(),
        }

    def _block_swap_prefetch_available(self) -> bool:
        if self.runner.is_mock:
            return False
        # services/ltx_runner.py:1086 と同じ `or 8` の式（実ゲートと同一にする）
        return int(self.low_vram.block_swap_blocks_on_gpu or 8) > 0
```

（`low_vram.block_swap_blocks_on_gpu` は `services/low_vram.py:43` に存在。**凍結済みの `vram_optimization` ブロックには一切触れない**——`low_vram.py:18-29` が明示的に禁じている。）

### 8.6 metadata.json

- `services/pipeline_manager.py` の `_write_metadata`（`:825-841`）: `"attention_used": outcome.attention_used,` の直後に `"block_swap_prefetch_used": outcome.block_swap_prefetch_used,` と `"peak_vram_reserved_mb": outcome.peak_vram_reserved_mb,`
- 同 `_write_chain_metadata`（`:729-746`）: 引数に `block_swap_prefetch_used=None, peak_vram_reserved_mb=None` を追加、`:746` の直後に同キー2つ。呼び出し側 `:686` にも渡す

### 8.7 バッチ経路

- **フロントのバッチ A2V / バッチ i2v-long**: `batchRunner.ts:337` が `settings.acceleration` をそのまま `buildA2vChainPayload` / チェーンビルダへ渡す → `accelerationRequestFields` 経由で自動的に新フィールドが乗る。**追加配線不要**（`buildA2vChainPayload.ts:228`、`chainUtils.ts:470`、`buildI2vLongPayload`）。
- **Gradio のバッチ**: ハンドラ引数ではなく `BatchSnapshot` 経由（`gradio_ui/batch.py:191` に `attention_backend: str = "sdpa"`、`:440` で消費）。**ここが漏れると「バッチだけ永久にオフ」という §43 が踏み抜きかけた罠の再演**になる。`batch.py:191` に `block_swap_prefetch: bool = False` を追加、`:157-158` の docstring に追記、`:440` の隣で `build_a2v_chain_payload(..., block_swap_prefetch=snap.block_swap_prefetch)`、`ui.py:1354` の隣で `block_swap_prefetch=bool(accel_prefetch_v),` をスナップショットへ。
- `gradio_ui/handlers.py`: `build_a2v_chain_payload`（`:341-429`）に kwarg 追加＋`:427-428` の直後に `if block_swap_prefetch: chain_payload["block_swap_prefetch"] = True`（**必ず attention_backend の後**＝キー順契約）。`make_generate_handler.generate`（`:439-456`）と chain 側（`:847-852`）にも同じ末尾 kwarg＋`:751-752` / `:1164-1165` の直後に加算。

### 8.8 MCP

`mcp_server/tools/generate.py`: `submit_generate`（`:56` 付近）と `submit_chain`（`:213` 付近）の引数末尾に `block_swap_prefetch: bool = False` を追加、docstring（`:114`, `:280`）に日本語説明、`:151-152` / `:321-322` の直後に条件付き加算。§43.1-4 の「実装のある項目だけを MCP に公開する」方針に**合致する**（これは実装がある）。

---

## 9. Gradio UI

`gradio_ui/i18n.py`（en: `:74-84` の直後、ja: `:567-576` の直後）に追加:

```python
# en
"accel_lbl_prefetch": "Block-swap prefetch",
"accel_info_prefetch": ("Hides the CPU<->GPU weight-transfer time behind the "
                        "computation (block swap only). The output is "
                        "bit-identical to having it off — only the speed "
                        "changes. Roughly 10-13% faster; no effect when block "
                        "swap is disabled."),
# ja
"accel_lbl_prefetch": "ブロック入れ替えの先読み",
"accel_info_prefetch": ("重みをCPUとGPUのあいだで運ぶ時間を、計算の裏に隠します"
                        "（ブロック入れ替えを使っているときだけ効きます）。"
                        "生成結果はオフのときと完全に同一で、速度だけが変わります。"
                        "約10〜13%短縮。ブロック入れ替えが無効な設定では何も起きません。"),
```

`gradio_ui/ui.py`: attention ラジオ（`:1032-1037`）の直後、VAE ラジオ（`:1038`）の**前**に挿入:

```python
                # 実装のあるつまみは attention に続いてここが2つ目。
                # 利用可否は API 側が黙って no-op にするのでクライアント側で
                # ゲートしない（attention と同じ理由。:1027-1031 参照）。
                accel_prefetch = reg(gr.Checkbox(
                    value=False, label=L("accel_lbl_prefetch"),
                    info=L("accel_info_prefetch"),
                ), "accel_lbl_prefetch")
                reg(accel_prefetch, "accel_info_prefetch", "info")
```

配線（`attention_backend` と完全同型・**末尾に追加**）:

- `dispatch()` シグネチャ（`:1230`）末尾に `, accel_prefetch_v`、`generate(...)` 呼び出し（`:1251`）に `block_swap_prefetch=accel_prefetch_v`、`BatchSnapshot(...)`（`:1354` の直後）に `block_swap_prefetch=bool(accel_prefetch_v),`
- `generate_btn.click(...).then(dispatch, inputs=[...])`（`:1390-1399`）の末尾へ `accel_prefetch` を追加（コメント `:1385-1389` を更新）
- `chain_dispatch`（`:1837-1838`）を2値対応に:
  ```python
  def chain_dispatch(*args):
      yield from chain_generate(*args[:-2],
                                attention_backend=args[-2],
                                block_swap_prefetch=args[-1])
  ```
  と `inputs=[..., attention_backend, accel_prefetch]`（`:1862`）。コメント `:1830-1836`, `:1853-1854` を更新。

**★オーナー確定事項1により、`accel_fused_gguf` チェックボックスは撤去しない。** 詳細設計の原案は §3-49 no-goクローズに合わせて撤去を提案していたが、オーナー裁定でUIは現状維持と確定した。`ui.py:1017-1022` の disabled チェックボックスと `i18n.py` の `accel_lbl_fused_gguf`（en `:77` / ja `:570`）は**変更しない**。`accel_info_unimplemented` もVAEラジオが使い続けるので残す。バックエンドAPIの `fused_gguf_dequant_gemm` フィールドも従来どおり残る（`api/models.py:122`）。

---

## 10. フロントエンド

### 10.1 `webui/src/shell/accelerationSettings.ts`

1. **モジュール docstring（`:1-23`）を更新**: 「3項目のうち実装は attention だけ」→「実装は attention と block-swap prefetch の2つ。fused GGUF は §3-49 の no-go 実測によりバックエンドで既定クローズ扱いだが、**UI（disabledトグル）は撤去せず現状維持**。vaeMode は依然モック」。

2. **型**（`:37-43`）:
```ts
export interface AccelerationSettings {
  attentionBackend: AttentionBackend;
  /** REAL（2026-08-01）: block swap の CPU<->GPU 転送を別 stream で先読みして
   * 隠す。sage と違い **出力はビット単位で不変**（速度だけが変わる）ので、
   * sage のような「同じシードでも絵が変わる」注意書きは不要。
   * 送出キーは `block_swap_prefetch`。 */
  blockSwapPrefetch: boolean;
  /** MOCK, never sent. UI上は disabled トグルとして現状維持（撤去しない。
   * §3-49はno-goでクローズしたが、UI自体は変更対象外というオーナー裁定）。 */
  fusedGgufDequantGemm: boolean;
  /** MOCK, never sent. */
  vaeMode: VaeMode;
}
export const BLOCK_SWAP_PREFETCH_DEFAULT = false;
```
`ACCELERATION_DEFAULTS`（`:71-75`）に `blockSwapPrefetch: BLOCK_SWAP_PREFETCH_DEFAULT,` を追加。

3. **localStorage を JSON へ移行**（`:54-103` を置換）。`:54-62` のコメントが既にこの移行手順を指定している（「非JSON値をこの backend 文字列として扱って移行する」）:

```ts
export const ACCELERATION_STORAGE_KEY = "nzltx23.acceleration";

/** 永続化する部分集合。モック2件は永続化しない（UIに無い／常に既定）。 */
export interface StoredAcceleration {
  attentionBackend: AttentionBackend;
  blockSwapPrefetch: boolean;
}

const STORED_DEFAULTS: StoredAcceleration = {
  attentionBackend: ATTENTION_BACKEND_DEFAULT,
  blockSwapPrefetch: BLOCK_SWAP_PREFETCH_DEFAULT,
};

/** 読み出し。3形態すべてを受ける:
 *  - 新形式 JSON `{"attentionBackend":"sage","blockSwapPrefetch":true}`
 *  - 旧形式のベア文字列 `"sage"` / `"sdpa"`（2026-07-31〜2026-08-01 に書かれたもの）
 *  - 壊れた値 / 欠落 → 既定
 * どの分岐でも未知の値は既定へ落とす（ThemeContext.tsx の readStoredTheme と同じ防御）。 */
export function readStoredAcceleration(): StoredAcceleration {
  let raw: string | null = null;
  try { raw = window.localStorage.getItem(ACCELERATION_STORAGE_KEY); } catch { return { ...STORED_DEFAULTS }; }
  if (raw === null) return { ...STORED_DEFAULTS };
  if (raw === "sdpa" || raw === "sage") {
    return { attentionBackend: raw, blockSwapPrefetch: BLOCK_SWAP_PREFETCH_DEFAULT };
  }
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return { ...STORED_DEFAULTS };
    const rec = parsed as Record<string, unknown>;
    const backend = rec.attentionBackend;
    return {
      attentionBackend: backend === "sdpa" || backend === "sage" ? backend : ATTENTION_BACKEND_DEFAULT,
      blockSwapPrefetch: typeof rec.blockSwapPrefetch === "boolean"
        ? rec.blockSwapPrefetch : BLOCK_SWAP_PREFETCH_DEFAULT,
    };
  } catch { return { ...STORED_DEFAULTS }; }
}

export function writeStoredAcceleration(value: StoredAcceleration): void {
  try { window.localStorage.setItem(ACCELERATION_STORAGE_KEY, JSON.stringify(value)); } catch { /* ignore */ }
}
```
既存の `readStoredAttentionBackend` / `writeStoredAttentionBackend`（`:84-103`）は**削除**し、呼び出し元（`useAccelerationSettings.ts`、`SettingsPanel.test.tsx:16`）を書き換える。

4. **能力読み取り**（`sageAvailability` `:116-119` の隣）:
```ts
/** サーバが block swap 構成で動いているか（backend §44）。判定式は
 * backend の _block_swap_prefetch_available() と同一（§8.5）。三値の理由と
 * 「明示的な false のときだけ無効化する」規律は sageAvailability と同一。 */
export function blockSwapPrefetchAvailability(
  status: StatusResponse | null | undefined,
): boolean | null {
  const v = status?.acceleration?.block_swap_prefetch_available;
  return typeof v === "boolean" ? v : null;
}
```

5. **加算的コントラクト**（`:124-152`）:
```ts
export interface AccelerationRequestFields {
  attention_backend?: AttentionBackend;
  block_swap_prefetch?: boolean;
}

export function accelerationRequestFields(
  acceleration: AccelerationSettings | undefined,
): AccelerationRequestFields {
  if (!acceleration) return {};
  const out: AccelerationRequestFields = {};
  if (acceleration.attentionBackend !== ATTENTION_BACKEND_DEFAULT) {
    out.attention_backend = acceleration.attentionBackend;
  }
  // サーバ既定（false）から動かしたときだけ送る。両方が既定なら {} が返り、
  // 本機能の導入前とリクエストがバイト同一になる（キー順は attention が先）。
  if (acceleration.blockSwapPrefetch !== BLOCK_SWAP_PREFETCH_DEFAULT) {
    out.block_swap_prefetch = acceleration.blockSwapPrefetch;
  }
  return out;
}
```

### 10.2 `webui/src/shell/useAccelerationSettings.ts`（★MINOR12修正版）

```ts
export interface UseAccelerationSettingsResult {
  acceleration: AccelerationSettings;
  setAttentionBackend: (value: AttentionBackend) => void;
  /** write-through で localStorage へ。attentionBackend と同じく「このPCの
   * 性格に対する選択」なのでリロードを跨いで残る。 */
  setBlockSwapPrefetch: (value: boolean) => void;
}

export function useAccelerationSettings(): UseAccelerationSettingsResult {
  const [acceleration, setAcceleration] = useState<AccelerationSettings>(() => {
    const stored = readStoredAcceleration();
    return {
      attentionBackend: stored.attentionBackend,
      blockSwapPrefetch: stored.blockSwapPrefetch,
      fusedGgufDequantGemm: FUSED_GGUF_DEQUANT_GEMM_DEFAULT,
      vaeMode: VAE_MODE_DEFAULT,
    };
  });

  // ★MINOR12反映: 書き出しは render 中の副作用（setState の functional updater内）
  // にせず、effect側へ寄せる。attentionBackend/blockSwapPrefetchのどちらかが
  // 変わるたびに1回だけ書く。React 19 StrictModeの二重実行でも同じ値を2回
  // 書くだけで無害。
  useEffect(() => {
    writeStoredAcceleration({
      attentionBackend: acceleration.attentionBackend,
      blockSwapPrefetch: acceleration.blockSwapPrefetch,
    });
  }, [acceleration.attentionBackend, acceleration.blockSwapPrefetch]);

  const setAttentionBackend = useCallback((value: AttentionBackend) => {
    setAcceleration((prev) => ({ ...prev, attentionBackend: value }));
  }, []);
  const setBlockSwapPrefetch = useCallback((value: boolean) => {
    setAcceleration((prev) => ({ ...prev, blockSwapPrefetch: value }));
  }, []);
  return { acceleration, setAttentionBackend, setBlockSwapPrefetch };
}
```

### 10.3 `webui/src/shell/SettingsPanel.tsx`（★CRITICAL②修正版）

- props（`:38-57`）に `onBlockSwapPrefetchChange: (value: boolean) => void;` を追加
- import（`:17`）に `blockSwapPrefetchAvailability` を追加
- `sageDisabled`（`:98-99`）の隣に:
  ```tsx
  const prefetchDisabled = blockSwapPrefetchAvailability(statusBody) === false;
  ```
- **fused GGUF の行（`:249-268`）は変更しない**（オーナー確定事項1）
- Acceleration 見出しコメント（`:239-246`）を書き換え（「3行のうち中央だけが実物」→「2行が実物、fused/VAEはモック」）
- attention 行（`:270-294`）の**後ろ**に prefetch 行を挿入。**disabledは利用不可側（On）のボタンだけ**（`sage` の `disabled={isSage && sageDisabled}` :281 と同作法。原案は両ボタンをdisabledにしていたがレビューCRITICAL②で修正）:
  ```tsx
  <div className="field">
    <span className="field-label">{strings.settings.accelPrefetchLabel}</span>
    <div className="settings-lang-toggle" role="group" aria-label={strings.settings.accelPrefetchLabel}>
      {[
        { on: true, label: strings.settings.accelPrefetchOn },
        { on: false, label: strings.settings.accelPrefetchOff },
      ].map(({ on, label }) => (
        <button
          key={label}
          type="button"
          className={`mode-tab${acceleration.blockSwapPrefetch === on ? " mode-tab--active" : ""}`}
          aria-pressed={acceleration.blockSwapPrefetch === on}
          disabled={on && prefetchDisabled}
          title={on && prefetchDisabled ? strings.settings.accelPrefetchUnavailableTooltip : undefined}
          onClick={() => onBlockSwapPrefetchChange(on)}
        >
          {label}
        </button>
      ))}
    </div>
  </div>
  {acceleration.blockSwapPrefetch && <p className="field-hint">{strings.settings.accelPrefetchNote}</p>}
  ```
- 呼び出し元 `AppShell.tsx` に `onBlockSwapPrefetchChange={accelerationControls.setBlockSwapPrefetch}` を追加

### 10.4 `webui/src/i18n/strings.ts`

**fused GGUF 関連キー（`accelFusedGgufLabel` / `accelFusedGgufOn` / `accelFusedGgufOff`）は削除しない**（オーナー確定事項1）。

追加（en `:950` の後 / ja `:1639` の後）:

```ts
// en
accelPrefetchLabel: "Block-swap prefetch",
accelPrefetchOn: "On",
accelPrefetchOff: "Off",
accelPrefetchUnavailableTooltip: "The server is not running with block swap, so this has no effect.",
accelPrefetchNote:
  "The output is identical to having it off — only the transfer method changes, so the same seed reproduces the same frames exactly. Roughly 10-13% faster.",
// ja
accelPrefetchLabel: "ブロック入れ替えの先読み",
accelPrefetchOn: "有効",
accelPrefetchOff: "無効",
accelPrefetchUnavailableTooltip: "サーバーがブロック入れ替えを使わない設定のため、効果がありません",
accelPrefetchNote: "生成結果はオフのときと同一です（運び方だけを変えるので、同じシードなら完全に同じ絵になります）。速度は約10〜13%短縮",
```
en/ja の**両辞書に必ず同じキー集合**を置く（`test_gradio_ui.py` 相当の言語切替テストがフロントにもある）。

### 10.5 `webui/src/api/types.ts`

- `GenerateRequest`（`:98-110`）に:
  ```ts
  /** Acceleration（2026-08-01, backend §44）: block swap の転送先読み。
   * accelerationRequestFields が唯一の組み立て元で、サーバ既定（false）の
   * あいだは完全に省略される。attention_backend と違い出力は不変。 */
  block_swap_prefetch?: boolean;
  ```
- `GenerateChainRequest`（`:204-212`）に同じもの
- `StatusResponse.acceleration`（`:347-353`）に `block_swap_prefetch_available?: boolean;`（`sage_available` と同じく optional）

### 10.6 フロントのテスト作法（既存の流儀を踏襲）

`SettingsPanel.test.tsx:1-101` を読んだ範囲での作法:
- `render` は `ThemeProvider > LanguageProvider > PrefillPolicyProvider > Harness`（`:75-89`）。`Harness`（`:45-67`）は**本物の `useAccelerationSettings` を呼ぶ**ので、localStorage 永続化が end-to-end で検証される。新 prop `onBlockSwapPrefetchChange={accelerationControls.setBlockSwapPrefetch}` を Harness に追加。
- `beforeEach` で `window.localStorage.removeItem(ACCELERATION_STORAGE_KEY)`（`:100`）。
- クエリは `screen.getByRole("button", { name: ... })`。同名ボタンが複数ある画面では `within(...)` でスコープを切る流儀（`AppShell.controlLora.test.tsx` / `useBatchForm.test.ts` に例）。**"On"/"Off" は VAE 行などと衝突しうるので、`within(screen.getByRole("group", { name: strings.settings.accelPrefetchLabel }))` でスコープすること。**
- `userEvent.setup()` → `await user.click(...)`。
- 型検査: `npm run typecheck`（`tsc -b`）を**必ず**通す（`package.json:14`）。

---

## 11. テスト計画

### 11.1 実行コマンド（確認済み）

| 対象 | コマンド | 現状ベースライン |
|---|---|---|
| バックエンド pytest（app venv, torch 無し） | `cd Nz-LTX23-backend && .\.venv\Scripts\python.exe -m pytest -q`（README.md:637） | 817 passed / 6 skipped（VERIFICATION_LOG §43.4） |
| エンジン selfcheck（`.venv-engine`＋CUDA） | `.\.venv-engine\Scripts\python.exe -m engine.transformer.block_swap_prefetch_selfcheck` | 新規 |
| 既存 selfcheck 回帰 | `... -m engine.transformer.{nag,vsf,sage}_selfcheck` | 6/6・5/5・3/3 |
| フロント単体 | `cd Nz-LTX23-frontend-AviUtl2/webui && npm run test` | 1585 passed / 10 skipped |
| フロント型検査 | `npm run typecheck` | 0 error |
| ネイティブ doctest | 既存手順 | 267 PASS（本件では不変） |

### 11.2 engine 側 selfcheck（★MAJOR6反映: pytestではなく実行可能スクリプト）

**`engine/transformer/block_swap_prefetch_selfcheck.py`（新規・CUDA必須・pytest非収集）** — `sage_selfcheck.py:1-41` の docstring 規約と `main()`（末尾40行）の PASS/FAIL 集計を踏襲。`.venv-engine` にはpytestが無い（app の `.venv` には torch が無い）ため、engine 側の全チェック（純関数のレイアウト計算・スロット列挙も含む）を**この1本のselfcheckスクリプトに統合する**（原案の別ファイル `tests/test_block_swap_prefetch.py` は不採用）。

| # | チェック | 合格条件 |
|---|---|---|
| C1 | **出力一致**: `GGMLQuantizedTensor` buffer・非persistent buffer（LoRA A/B 相当）・bias（Parameter）・LayerNorm weight・block直付け Parameter（`scale_shift_table` 相当）を持つ 12 ブロックのダミー Sequential を作り、`blocks_on_gpu=3` で prefetch OFF / ON を同一入力で 3 パス回す | `torch.equal(out_off, out_on)`（**ビット同一**。転送方式しか変えていないので当然） |
| C2 | **CPU正本の不変性** | 全ブロックの全 CPU テンソルを install 前に clone し、3パス後に `torch.equal` で全一致。`GGMLQuantizedTensor` は `as_subclass(torch.Tensor)` で生バイト比較 |
| C3 | **常駐数の上限**（★MAJOR9反映） | 各 forward の入口で `len(engine._state) <= blocks_on_gpu + 2` を assert（`+1`ではない。ブロック47の前パス残留分を含む）。かつ末尾ブロック実行後に `len(engine._state) == 1` |
| C4 | **発行スケジュール** | ブロック `idx` を実行する時点で、`idx` は必ず「1ステップ以上前に発行済み」（cold start の idx=0 と ramp 中を除く）。`_stats["sync_misses"]` がパス2以降で 0 |
| C5 | **GGML メタ保持** | 転送後の GPU 側 buffer が `isinstance(..., GGMLQuantizedTensor)` かつ `_ggml_type` / `_float_shape` が CPU 正本と一致し、`.shape` が float 形状を返す。さらに `ggml_linear_forward` を通した結果が CPU 実行と一致 |
| C6 | **pinned 確保失敗のフォールバック** | `torch.empty(..., pin_memory=True)` を RuntimeError を投げるモックに差し替えて install → 従来同期パスが張られ、出力は C1 と同一、`service.last_prefetch_used == "on->off"` |
| C7 | **`blocks_on_gpu >= total`** | install が早期 return し、`_prefetch_engine is None`、pinned が一切確保されていない、`last_prefetch_used == "off"` |
| C8 | **ジョブ跨ぎのリーク** | ベースライン `torch.cuda.memory_allocated()` を取り、`install→3パス→teardown_prefetch()` を3回繰り返して、3回目の値が1回目 ±1MB 以内。`torch.cuda.Stream` の本数と pinned のバイト数が増えていない（E9の finally 経路を模してテストする） |
| C9 | **stream 安全性の負荷テスト**（★CRITICAL①反映） | ブロック計算を意図的に重く（大きめ matmul を 20 回）し、転送を頻繁にして 20 パス回す。C1 のビット一致が維持される。さらに `_SKIP_ALLOC_EVENT_WAIT=True` の変異（`_issue()` 内の `self._xfer.wait_event(alloc_evt)` を意図的にスキップする）で**意図的に破綻することを確認するネガティブケース**を入れる（S1b の必要性を実証する。原案の `_DISABLE_RECORD_STREAM` 変異は不採用——record_stream自体を使わない設計になったため） |
| C10 | **例外後の再 install** | forward で意図的に例外を起こして中断 → 同じサービスで install し直し（teardown_prefetch経由）→ 正常完走・出力一致 |
| C11 | **レイアウト計算** | オフセットが512の倍数、`total_bytes`が全`nbytes`の和以上、0次元テンソルを含めて`view(dtype)`の可分条件が全スロットで成立（§6.3の`reshape(-1)`修正の回帰） |
| C12 | **スロット列挙** | 非persistent buffer / block直付け Parameter / ネストした子モジュールを持つトイ `nn.Module` で、期待どおりの `(module, name, is_param)` 集合が返る |
| C13 | **発行スケジュールの純関数** | `_window(idx, bs, total)` の境界（idx=0 / idx=total-bs / idx=total-1）が期待どおり |
| C14 | **共有テンソル検出** | weight tying を持つダミーモジュールで install 時に WARNING が1回出る |

### 11.3 API/UI 層の pytest（app venv、torch 不要）

| ファイル | 追加テスト |
|---|---|
| `tests/test_ltx_runner_payload.py` | ① 既定リクエストの worker ペイロードに `block_swap_prefetch` キーが**無い**（既存の完全一致 assert がそのまま通ることでも担保） ② `block_swap_prefetch=True` で `payload["block_swap_prefetch"] is True` かつ**キー順の末尾**（`attention_backend` の後） ③ chain 側も同様 |
| `tests/test_validation.py` | `GenerateRequest` / `GenerateChainRequest` の既定が `False`、`to_clip_request()` が転記する |
| `tests/test_smoke.py` or 新規 | `GET /status` の `acceleration` に `block_swap_prefetch_available` があり、mock backend では `False`。凍結済み `vram_optimization` のキー集合が不変 |
| `tests/test_gradio_handlers.py`（`:2947-3090` 帯に追記） | `build_a2v_chain_payload` の完全 dict 一致（既定）／True 時に末尾追加／`generate` ハンドラ・`chain_generate` ハンドラ双方 |
| `tests/test_gradio_ui.py`（`:682-695` 帯に追記。★MINOR14反映のアンカー修正。原案は同帯でfusedチェックボックス撤去も指示していたが、オーナー裁定によりfused UIは現状維持のため撤去テストは不要） | ① prefetch チェックボックスが存在し `value is False` かつ `interactive` ② generate と chain の両 `inputs` に入っている（既存の `test_acceleration_attention_radio_is_wired_into_generate_and_chain` と同型） ③ 言語切替に `accel_lbl_prefetch` / `accel_info_prefetch` が乗る |
| `tests/test_gradio_batch_runner.py` / `test_gradio_batch_fullstack.py` | `BatchSnapshot.block_swap_prefetch` がペイロードへ届く |
| `tests/test_mcp_tools_generate.py` | 既定で `set(body.keys())` が不変、True で1つだけ増える |

### 11.4 フロント vitest

| ファイル | 追加/改訂 |
|---|---|
| `shell/accelerationSettings.test.ts` | ① `readStoredAcceleration` の3形態（新JSON／旧ベア文字列マイグレーション／壊れた値） ② `accelerationRequestFields`: 全既定→`{}`／prefetch のみ→`{block_swap_prefetch:true}`／両方→2キー／モック2件を立てても増えない（`:63-70` を `fusedGgufDequantGemm` 込みのまま維持） |
| `shell/useAccelerationSettings.test.ts` | `setBlockSwapPrefetch` の write-through と再マウント復元、`attentionBackend` との相互非破壊 |
| `shell/SettingsPanel.test.tsx` | ① prefetch の On/Off が `within(getByRole("group",{name:...}))` で押せて `aria-pressed` が入れ替わる ② 押すと localStorage に JSON が入る ③ `block_swap_prefetch_available:false` で**Onボタンのみ** `disabled`、Offは常にクリック可能、`undefined`（旧backend）では両方有効 ④ fused GGUF 行は**変更なし**（既存テストのまま） ⑤ ON のとき note が出る |
| `modes/create/useGenerationForm.test.ts` / `modes/chain/useChainForm.test.ts` / `modes/chain/chainUtils.test.ts` / `modes/batch/buildA2vChainPayload.test.ts` / `modes/batch/batchRunner.test.ts` / `modes/batch/useBatchForm.test.ts` / `modes/batch-i2v-long/buildI2vLongPayload.test.ts` | 各 `describe("acceleration")` に「prefetch=true が本文に乗る」1本ずつ。既存の「既定ならキーなし」assert は無改修で通ること |

### 11.5 実機ゲート実施上の注意（詳細設計より。§14のG1〜G8を実施する際の補足）

1. worker 初回ジョブは約8%速い → **必ず交互対比較**、1本目は捨てる（`off→on→off→on→off→on` の6本を回し、隣接する組で比較する）。
2. 本件は triton JIT のような初回ペナルティは無いが、cudaHostAlloc（pinned 740MB）が prefetch 初回ジョブに一度だけ約0.1〜0.3秒乗る。1本目を捨てる運用でこれも吸収される。
3. `sage` と併用して測らない（要因が2つになる）。速度ゲート（G2/G3）は `attention_backend="sdpa"` 固定で行う。
4. ビット一致確認（G1/G4）は ffmpeg で `-f rawvideo` 抽出しフレーム SHA-256 を比較する。sage と違い転送方式しか変えないので、PSNR ではなく**ビット一致**を要求する。不一致は即FAIL・原因究明（stream同期漏れの疑い）。
5. VRAM 判定（G5）は `peak_vram_reserved_mb`（`torch.cuda.max_memory_reserved`）を主指標にする。`peak_vram_mb`（`max_memory_allocated`）は本改修のリスク（stream別プール分断・reserved増）を検知できないため補助指標に留める。加えて `torch.cuda.memory_reserved` の推移が単調増加しないことを確認する（断片化検知）。

---

## 12. ドキュメント

### 12.1 `Nz-LTX23-backend/Docs/VERIFICATION_LOG.md`

`## 44.` を新設（§43の直後）。§43の構成をそのまま踏襲:

- `44.1 決定事項` — ①既定OFF（理由: 効果は速度のみだが、CPU RAM運用とCUDA stream運用を変えるので実績が溜まるまで既定は動かさない）②切替はジョブ単位③非循環ウィンドウを選んだ理由（§5.1）④D2H廃止はON時のみ（§4.3）⑤pinnedは2枠×最大ブロックサイズ、全ブロックpinは不採用⑥arenaはcompute stream確保・record_stream不使用である理由（§5.4）⑦実際に効いたかは`block_swap_prefetch_used`に記録⑧§3-49（fused GGUF）はno-goでクローズしたがUIは現状維持
- `44.2 実装箇所一覧` — 新規2ファイル＋改修ファイルを§3・§8〜§10の粒度で
- `44.3 計測の根拠` — 実測帯域表（pageable H2D 14.0 / D2H 9.1 / pinned H2D 23.4 / D2H 25.4 / CPU内 27.1 GB/s）、1パスあたりの内訳（H2D 1.14s＋D2H 1.76s＝2.9s）、期待短縮11〜13%
- `44.4 機械検証の結果` — selfcheck 14項目のPASS数、pytest実測値、vitest実測値、typecheck 0
- `44.5 実機ゲート表` — §15の表（実測値を埋める）
- `44.6 計測手順の注意` — §11.5の5点
- `44.7 §43との関係` — Acceleration区画の2つ目の実装項目であること
- `44.8 既知の制約` — pinned 740MBをプロセス寿命ぶん保持する／`vram.block_swap=false`では完全no-op／`blocks_on_gpu=1`では先読みが効かない
- `44.9 残タスク`

### 12.2 `Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md`（★MAJOR5反映：移動先を§4ではなく§3に修正）

1. **§3-49を「§3 再訪条件つきでクローズした項目」内に留めたまま、§3-46と同型の記録に書き換える**（原案の「§4スコープ外へ降格」は不採用。§3が既に「クローズ済みだが再訪条件つき」の区画であるため、この項目の性質（実測でno-go判定した完了項目）と一致するのは§3である）。本文を次の内容で書き換える:
   - 「**判定: no-go（2026-08-01）**。実測により、GGUF Q4_K_M の逆量子化コストは生成時間の支配項ではなく（dequant=1パス1.63秒＝stage2ステップ32.76秒の4.97%、Q6_K補正でも5〜7%）、支配項は block swap の CPU⇔GPU 転送（1パスあたり約2.9秒）であることが判明した。転送側は §44（先読みblock swap）で解消したため、融合カーネルの投資対効果が立たない。」
   - 「**再訪条件**: さらなる高速化が必要になったとき（実測データを添付）。」
   - 台帳の他文書参照はファイル名のみ・行番号は書かない（台帳の既存規則）。
2. **§3-50（PruneVAED）は§3に据え置き**（本テーマの対象外）。ただし出典行の「§43.1・§43.9」に「§44.1」を追記。
3. **新規起票（§4末尾の次番号。実装時に現況の最終番号を確認して採番すること）**: 「先読み block swap を既定 ON にするかの再検討」（`sage` の既存項目と同じ体裁。再訪条件＝フィールドでNジョブ無事故）。
4. **新規起票（同上、次番号）**: 「prefetch OFF 時にも退避コピー廃止（CPU正本方式）を適用するかの再検討」。再訪条件＝§15 G5相当の実機計測でCPU RAMが改善していることが確認できたら。
5. 台帳冒頭の最終更新日（★MINOR14反映のアンカー修正: `:3`）を更新。台帳冒頭のルール（★MINOR14反映のアンカー修正: `:9`）と整合させる。

### 12.3 `Nz-LTX23-backend/config.yaml.example`

**新しい設定キーは追加しない**（§43.1-2と同じ、ジョブ単位のつまみはconfigで固定しない方針）。ただし `block_swap` / `block_swap_blocks_on_gpu` の直下にコメント2行を足す:

```yaml
  block_swap: true
  block_swap_blocks_on_gpu: 8
  # ここが有効（block_swap: true かつ blocks_on_gpu > 0）のときだけ、ジョブ単位の
  # 「先読み block swap」（GenerateRequest.block_swap_prefetch）が意味を持つ。
  # 無効な設定では黙って no-op になる（VERIFICATION_LOG §44）。
```

`config.py` の `VramConfig` は変更なし。

### 12.4 `Nz-LTX23-backend/README.md`

「7. 制限事項」の fused GGUF に関する記述をno-goクローズに合わせて更新（UIは現状維持である旨も明記）し、先読みblock swapの項を追加（pinned 740MBの常駐と、block_swap無効時のno-op）。

---

## 13. 補足: 期待効果の再確認（プランの前提として記録）

- 1パス（48ブロック）あたりの転送: H2D 47×340MB ÷ 14.0GB/s ≈ 1.14s、D2H 47×340MB ÷ 9.1GB/s ≈ 1.76s、計**約2.9s**
- 768p/257f: 全体270s、8+3=11パス→転送**約32s**
- 本改修後: D2Hは**構造的にゼロ**（1.76s/pass消滅）、H2Dはpinned 23.4GB/sで0.68s/passに短縮した上で計算裏に**ほぼ完全隠蔽**。露出はcold startとパス先頭の約46ms/passのみ
- 期待: **270s → 約236〜240s（1.13〜1.15倍）**。ゲート合格基準は保守的に1.08倍

---

## 14. 実装ステップと完了条件（プラン正本転記）

> 各ステップ完了後に次へ進む。実装・テストはOpus/Sonnetのサブエージェント、親は監督専任。

> **設計文書のS1〜S10との対応関係**: 本書の詳細設計（§4〜§13）はより粒度の細かいS1〜S10（selfcheck単独ステップ・実機ゲートを2分割等）で記述されているが、**実行時の正本は以下のプランS0〜S7**である。対応関係の目安: 設計S1+S2（core+selfcheck）→本書S1、設計S4（API配線）→本書S2、設計S5+S6（実機ゲート前半＋後半）→本書S3、設計S7（Gradio）→本書S5、設計S8+S9（フロント＋実機目視）→本書S6、設計S10（ドキュメント）→本書S7。

| STEP | 内容 | 完了条件 |
|---|---|---|
| **S0 スパイク** | engine venvで `torch.inference_mode()`×`_make_subclass` の10行検証（リポジトリ変更なし。§6.1） | 可否確定と採用経路の決定 |
| **S1 バックエンド核心** | block_swap_service.py＋fast_video_pipeline.py | selfcheck全PASS（on/off出力一致・CPU正本不変・常駐数≤bs+2・スロット再利用・例外teardown・0次元/alignment）、off経路のdiffが実質0行であることのレビュー |
| **S2 API配線** | models/runner/worker/status/metadata/MCP＋reservedメトリック | appのpytest全緑（既存736基準）、mock e2eで /status と metadata 反映確認 |
| **S3 実機ゲート G1-G7**（下表） | エージェントがAPI経由で交互実行、GPU占有はオーナーと調整 | 全ゲートPASS |
| **S4 既定on反転** | api/models.pyの既定値・/statusの既定表明を反転、スモーク1本（G8） | ─ |
| **S5 Gradio UI** | チェックボックス（既定はサーバー既定に追随）＋gradioハンドラテスト | ─ |
| **S6 フロントエンド** | 上記変更＋vitest＋typecheck＋deploy.ps1でデプロイ（オーナー目視ゲート） | ─ |
| **S7 ドキュメント・台帳** | §44、WORKORDER、PENDING_TASKS（§3-49の記録更新含む） | コミットはオーナー確認後 |

---

## 15. 実機ゲート（プラン正本転記）

| # | 内容 | 合格条件 |
|---|---|---|
| G1 | 同一シード 768p i2v、off vs on | 全フレーム画素ビット一致（転送方式変更のみなので一致が期待値。不一致=バグ） |
| G2 | 768p/257f i2v 交互対比較 off→on→off→on…3組（§43.6プロトコル: 初回ジョブ除外） | 平均8%以上の短縮（見積り11〜13%） |
| G3 | 1088p/153f 同上1組 | 短縮を確認（見積り8〜10%） |
| G4 | Style LoRA＋IC-LoRA付きジョブ off vs on | 出力ビット一致 |
| G5 | G2実行中のVRAM | `peak_vram_reserved_mb` 差 +400MB以内、共有GPUメモリ溢れ増なし、快適上限257fでclean維持 |
| G6 | chain生成（複数クリップ） | 完走＋出力一致＋短縮傾向 |
| G7 | バッチ経路スモーク | 完走・metadata反映 |
| G8 | 既定on反転後スモーク | 明示指定なしでon動作、/status正常 |

---

## 16. 主要リスクと対策（プラン正本転記）

| リスク | 対策 |
|---|---|
| stream同期バグ（サイレントな数値破損） | G1/G4のビット一致ゲートで検出。同期点はS1/S1b/S2/S4の4点に限定し selfcheckでストレス |
| caching allocatorのstream別プール分断によるVRAM劣化 | arenaは計算stream確保（設計で回避済み）＋G5をreserved側で判定 |
| inference_mode×subclass復元の非互換 | S0スパイクで事前確定、代替経路用意 |
| ジョブ間の参照残留（リーク） | teardownをジョブfinallyに配置、selfcheckで検証 |
| pinned確保失敗等の環境差 | 同期パスへの自動フォールバック＋fail-loudログ＋metadata記録 |
