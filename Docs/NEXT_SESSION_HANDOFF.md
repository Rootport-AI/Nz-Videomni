# 次セッション引き継ぎ書 — LTX 2.3 を 16GB で動かす

---

## ▶▶▶▶ 最新の正本（2026-06-30 後半・このセクションを最初に読む）＝720p 達成 & 連続生成の commit 枯渇を解決

**＝残課題C（720p スケールアップ）完了。本来の機能（16GB で 720p 動画生成）が、先行事例と遜色ないマシンスペックで実現した。** 次は検証用 Gradio GUI での手動動作確認 →（本来の目的）**AviUtl2 拡張機能からこの API を叩く統合**。

### 何ができるようになったか（実機検証済）
- **720p 級（1280×768 を生成→任意で 1280×720 にクロップ）の T2V が本番 API 経路で完走**。生成 ~167–171秒（RTX 4070 Ti SUPER 16GB）。job `684393c6`（crop なし 1280×768 配信）/ `5a3540ba`（crop 1280×720）で ffprobe 確認。OOM なし。
- **連続生成（マルチジョブ）が commit 枯渇クラッシュせず完走**。720p×3本連続・384×256×3本連続いずれも全 PASS（実測）。→ 将来の「5秒クリップを繋いで 20秒動画」（終了フレーム→次の開始フレームの I2V 連結）の土台ができた。
- **✅ `--te-offload`（既定 ON）＝Gemma text-encode ピーク低減 DONE・A/B 実機検証済（2026-07-01）**。機構＝逐次 per-layer ストリーミング（GGUF Gemma 48 層を CPU 常駐→2層ずつ GPU へ・compute は GPU）。encode-phase Dedicated ピークを 15,839→10,540 MB（−33%）に下げ encode 時の shared 溢れを消去・wall-clock ペナルティ無し・出力 OFF とバイト一致（512×320/49f A/B）。720p ON スモークも完走（168.2s）。**toggle＝`--no-te-offload` で無効化**。詳細＝VERIFICATION_LOG §11、commit `96f41c4`。**★スコープ注意：te-offload 自体は全体ジョブ `peak_vram_mb`（両モード 16,944）に触れない**（残課題B の Gemma encode 側＝peak ① だけ解消）。**↳ transformer load 側＝peak ② は ✅ `--dit-cpu-load`（既定 ON, 2026-07-01）で別途 RESOLVED**（transformer ブロックを直接 CPU 構築し full-GPU materialization を経由しない＝transformer-load Dedicated 15.8→1.4GB・512×320 で whole-job ceiling 16,944→~9.2GB・出力バイト一致。VERIFICATION_LOG §12、後述【B】）。

### 確定した本番デフォルト設定（2点・いずれも検証済・コミット対象）
1. **`services/ltx_runner.py`** worker env に `env.setdefault("LTX_KEEP_RESIDENT","0")` ＝ **keep_resident OFF が既定**（明示 env は尊重）。理由＝keep_resident=True だと 720p で Gemma を CPU ビルド→out-of-place `.to(cuda)` 移動する瞬間二重在が 16GB マージンを超え **native crash**（traceback 無し、text-encode 中）。=0 で直接 cuda ビルドになり回避。`LTXFastVideoPipeline.create()` の既定は False（`ltx_fast_video_pipeline.py:75`）。
2. **`config.yaml`** vram に `use_component_files: true`（Path B）＋ `block_swap_blocks_on_gpu: 8` / `vae_spatial_tile_size: 512` / `vae_temporal_tile_size: 64`。理由＝comp=1 が VAE/audio/projection を 46GB モノリスでなく小ファイルから読み、**毎ジョブ再 materialize の commit を束縛**。これが連続生成の鍵。

### なぜこの設定か（実測根拠・GPU＋system commit 両監視）
- **comp=0/keep=0（旧既定）はマルチジョブで残課題A の機構（§8 commit 枯渇）が再発**＝3本連続@384 の job3 で commit **99.2%（135GB）→ Gemma 再構築で native crash**。§9.7 のリーク修正では治らない別機構（毎ジョブ再 materialize）と実証。
- **comp=1 にすると 46GB モノリスを使わず commit が ~90%（102GB）に束縛**＝384/720p とも3本連続 PASS（crash 消滅・commit はジョブ間で累積せず横ばい）。
- 留意（非致命）：keep=0 は毎回再 materialize するため **gen 時間が漸増**（720p で 168→209秒/3本、+24%。アロケータ断片化等）。commit は横ばいなので暴走しない。**長尺連結を多数本回す段で問題化したら、keep=1 を維持したまま Gemma 移動を in-place 化する恒久最適化に着手**（§9.7 の高速 flat 経路＝keep=1 と 720p を両立させる）。

### マシンスペック（公開時の見積り・先行事例と比較）
- 「commit ~102GB」は **ディスクでなく仮想メモリ予約（物理RAM＋ページファイル）** のピーク。先行事例 ComfyUI 16GB レシピも「RAM32GB＋swap64GB」相当（§91）＝**ほぼ同等で極端でない**（旧懸念の ~200GB は 46GB モノリス＋二重 materialize の悪い経路の話で、comp=1 で回避済）。本機は RAM64GB＋pagefile48GB＝commit 上限 ~112GB で 90% 着地。
- **実行に本当に要るモデルは ~28GB**（GGUF transformer 16.5＋GGUF Gemma 6.8＋components 3.85＋upscaler 0.93＋設定）＝ComfyUI GGUF 構成（~25–30GB）と同等。現状 `models/` は 93.83GB あるが、**モノリス 43GB＋qat 重み 22.7GB（計 ~66GB）は今の comp=1/GGUF 経路では不要候補**（落とせば公開フットプリントが先行事例並み。要・読み込み検証→下記残課題）。

### 次のマイルストーン
1. **検証用 Gradio GUI でユーザーが手動動作確認**（UI が 720p/crop トグル/T2V/I2V を出せるか要確認）。
2. （本来の目的）**AviUtl2 拡張機能 → このバックエンド API（FastAPI, port 18620, `/api/v1/*`）統合**。

### 残課題（ディレクトリ掃除を除く・優先度はユーザー判断）
> ※開発段階のゴミ（`outputs/` のテスト出力、未追跡の診断スクリプト等）の**ディレクトリ掃除は、専任の次セッションに方法ごと委ねる**ため、ここには残課題として載せない（ユーザー指示・2026-06-30）。

- **【B】load/encode の一時 shared 溢れ → ✅ load/encode フェーズは 16GB 内達成（2026-07-01）**。(i) **Gemma encode 側（peak ①）は `--te-offload`（既定 ON）で解消**（encode Dedicated 15.9→10.5GB・encode 時 shared 溢れ 0、VERIFICATION_LOG §11）。(ii) **transformer load 側（peak ②）は `--dit-cpu-load`（既定 ON）で解消**＝従来 block-swap が一旦 full-GPU ロード（peak max_alloc 16,944）→退避していたのを、transformer ブロックを**直接 CPU 構築**し full-GPU materialization を経由しないよう変更（transformer-load Dedicated 15.8→1.4GB・load 時 shared 溢れ消失・出力バイト一致。**512×320 で whole-job ceiling 16,944→~9.2GB**、VERIFICATION_LOG §12）。**↳ これで load/encode フェーズ（peak ①＝te-offload・peak ②の load-spike＝dit-cpu-load）は 16GB 内に収まった。** 残るニュアンス＝**大解像度・長尺では denoise stage2 が依然天井へ近づきうる**（これは固定 load スパイクとは別軸＝den2 のトークン数スケーリング。`RESOLUTION_DURATION_CAPABILITY.md §2` の den2 表は不変）。詳細は下記【残課題B の本丸】＋ VERIFICATION_LOG §7.9/§11/§12。
- **【最適化】keep_resident=1 を 720p で使えるようにする恒久修正**＝Gemma の out-of-place `.to(cuda)` 移動を in-place 化／CPU 側を移動前解放。成功すれば keep=1（§9.7 で 6ジョブ flat・高速・gen 時間漸増なし・commit より低い）と 720p を両立。長尺連結や速度が要件化した時の本命。
- **【残課題B の本丸＝peak ② 低減】→ ✅ RESOLVED（`--dit-cpu-load` 既定 ON, 2026-07-01）**。transformer load 段の一時 shared 溢れ（block-swap の full-GPU ロード→退避）を、transformer ブロックの**直接 CPU 構築**（full-GPU materialization 非経由）で除去した＝**全体ジョブ天井を下げるレバーが効いた**：512×320 で whole-job ceiling（torch `max_memory_allocated`）**16,944→~9.2GB**、transformer-load Dedicated 15.8→1.4GB、出力バイト一致、wall-clock 退行なし（VERIFICATION_LOG §12）。Gemma 側（peak ①）は te-offload 済。**↳ 残るのは「den2（denoise stage2）の解像度・尺スケーリング」軸のみ**＝固定 load スパイクではなく高トークン時に天井へ近づく可変成分（`RESOLUTION_DURATION_CAPABILITY.md §2`・VERIFICATION_LOG §12.5 のスコープ注意）。これは別課題で、load/encode の「真の 16GB fit」自体は達成済。
- **【任意・te-offload チューニング露出】**`GemmaLayerOffloadService(layers_on_gpu=2)` は現状ハードコード（`config.yaml` 未露出）。1 に下げると encode ピークさらに低下／3-4 で速度トレード。露出は任意の将来作業（VERIFICATION_LOG §11.6）。
- **【連続生成の上限確認】**keep=0 での gen 時間漸増が、連結機能の実本数（5s→20s＝4本以上）で許容範囲かを、その実装時により長い連続で再計測。
- **【公開フットプリント削減】**comp=1/GGUF 経路で **モノリス 43GB＋qat 重み 22.7GB が実行時に本当に開かれないか**をコード/ログで検証→不要なら required から外す（DL させない/削除可に）。落とせばディスク要求 ~28GB で先行事例並み。[[no-large-pagefile-disk-requirement]] の達成。
- **【未検証パス】**新デフォルト（comp=1/keep=0）での **I2V のマルチジョブ・音声付き連続生成は未検証**（今回は T2V マルチジョブのみ実測）。連結機能は I2V 連続なので、実装前にここを検証。
- **【D】README / `LTX23_Backend_Specification_v04…` の全面改訂**＝pre-pivot のまま。アーキテクチャが固まった今が改訂の好機（残課題サマリ末尾参照）。
- **【真の目的】AviUtl2 拡張機能との統合**（上記マイルストーン2）。

### 一次情報のポインタ
- **解像度×尺の能力リファレンス＝Docs/RESOLUTION_DURATION_CAPABILITY.md**（16GB でどの解像度/尺まで作れるか・den2 推定式・生成時間・AviUtl2 UI 含意の正本）。
- 技術記録＝**VERIFICATION_LOG §10**（720p 二段達成・keep_resident 原因/修正・component_files マルチジョブ修正・commit/ディスク実測・§10.6 解像度×尺スイープ）。
- スケールアップ調査の正本＝Docs/SCALEUP_16GB_RESEARCH.md（✅達成バナー追記済）。
- 設定知識＝Docs/LTX23_REFERENCE.md。memory `[[720p-16gb-verified]]` / `[[ltx-bridge-project]]`。

**（以下は本セクションより前の歴史的経緯。残課題A〜D の元記述・Phase 5 配線記録等は参照用に温存。最新は上記 ▶▶▶▶ が正。）**

---

最終更新: 2026-06-30（**残課題A＝per-job リークを診断確定→修正→6ジョブ実機検証 PASS。フォーク backend を版管理化（commit 済）。音声 Phase 1 実機確認 PASS（✅DONE）。次の一手＝qat-drop(24GB削減)／720p スケールアップ（残課題C）**）/ 想定読者: 次セッションのエージェント

> **このドキュメントが現時点の最新・正本（handoff）です。まず §0 と「★残課題サマリ」を読めば、現状と次の一手が分かる。**
> **【リポジトリ状態 2026-06-30】** フォーク `vendor/LTX-Desktop-LOW-VRAM/backend/` の **.py ソース 169ファイルが版管理対象になった**（commit `4ea5c4b` "Vender"、origin/main 同期済）。`.gitignore` を negation ブロックに変更し backend ソースのみ追跡（`.venv`/weights/フロントエンド/`uv.lock` は除外維持）。フォーク自身の `.git` は **`.git_fork_disabled` にリネームして無効化**（境界を外しファイル追跡を可能にするため・42MB・削除可）。出典は `vendor/LTX-Desktop-LOW-VRAM/backend/VENDOR_NOTICE.md`。**＝以後エンジン改変は通常の git で commit できる**（以前は vendor/ 丸ごと ignore でエンジン全体が版管理外だった）。
> 設計判断の根拠と実機検証の全経緯は [VERIFICATION_LOG.md](VERIFICATION_LOG.md)（**最新＝§9.7（per-job リーク診断確定＋修正＋検証 PASS＝残課題A 完全解決）。次セッションはまず §0 末尾の ▶▶▶ と §9.7 を読め**、§9.6＝Path B component-files、
> §6＝Phase 5(A) 配線、§5＝Gemma GGUF）が一次情報。実装計画は `~/.claude/plans/frolicking-tumbling-hennessy.md`。
> 要約は memory `[[ltx-bridge-project]]` / `[[ltx-desktop-lowvram-fork]]`。⚠️ 古い記述が残るドキュメント（後述 §6）に惑わされないこと。

---

## 次セッション開始点（2026-06-30 更新）

- **完了**: 音声 Phase 1 実機確認 **✅PASS**（VERIFICATION_LOG §9.8）。16GB 高解像度スケールアップの調査を **`Docs/SCALEUP_16GB_RESEARCH.md`** に集約（正本）。残課題A（worker 再利用 crash）も解決済（§9.7）。
- **最初に読むべき**: **`Docs/SCALEUP_16GB_RESEARCH.md`**（720pレバー棚卸し＋コミュニティ実証レシピ＋真の難所3点）。要点＝**720pレバーの 4/5 は我々のフォークに既に配線済・既定OFF**（二段パイプライン / attention tiling=`attention_tile_size` / VAE 空間タイル / VAE 時間タイル / block-swap）。**未実装は FFN チャンキングのみで、それは主に長尺向け＝1280×720 の空間スケールには非必須**。
- **次の一手（推奨A・GPU不要から着手）＝720pスケールアップ**:
  - **Step 1（読解のみ・GPU不要）**: 公式 `ti2vid_two_stages.py`（`vendor/LTX-2/packages/ltx-pipelines/.../ti2vid_two_stages.py`）を精読し、(1) 難所①「stage1→stage2 の VRAM 遷移＝両段を同時保持するか／段間で latent 保存＋モデルアンロードが要るか」、(2) `LTXFastVideoPipeline.create()` の VRAM ノブ（`attention_tile_size` / `vae_spatial_tile_size` / `vae_temporal_tile_size` / `block_swap_blocks_on_gpu`）が **両ステージに効くか** を確定する。
  - **Step 2**: SCALEUP doc §5 の推奨レシピで **計装つき 1 本だけ生成**（dedicated＋shared 両監視・既存 `_gpu_mem_sampler.ps1` 併走・フォーク venv のみ）→ 収まるか／どこでスパイクするか実測。
  - **Step 3（必要時のみ）**: block-swap 深度調整 or 段間アンロード実装。FFN チャンキングは将来の長尺対応で初めて検討。
- **真の難所3点（先行事例に答えが無い・SCALEUP §4）**: (1) 段間(stage1→stage2)遷移スパイク (2) 22B の block-swap 深度 (3) tiling＋GGUF＋block-swap の共存性。
- **代替B**: qat-drop（24GB 削減）。
- **目標解像度＝1280×720 で充分**（より大きいサイズ・長尺の情報は SCALEUP doc 付録に保全）。
- **作業原則の念押し**: 監督役＝編集/テスト/生成はサブエージェントへ委譲、本体 Python 不可触・生成はフォーク venv のみ、**いきなり 720p 生成や闇雲なスイープをしない**（先行事例レシピを写す＋最小再現確認＝[[research-prior-art-first]]）。

---

## 0. いまどこにいるか
- **目標＝LTX-2.3 を 16GB VRAM(RTX 4070 Ti SUPER/Win/torch2.9.1+cu128)で動かし FastAPI で公開**。Phase 1＝T2V＋最小I2V。
- **✅ Phase 5(A) 配線完了・実機検証済**：凍結 REST API が実エンジン（GGUF 低VRAMパイプライン）で生成。`services/ltx_runner.py`
  の `_RealBackend`＝**Approach W（常駐サブプロセスワーカー `_ltx_worker.py`）**。詳細 §3／VERIFICATION_LOG §6。
- **✅ Phase 5(B)＝denoise 工程の VRAM 低減 完了・実装済（2026-06-29）**：真因＝本機 Windows で `expandable_segments` が
  no-op のため、block-swap の transformer ロード後に空きセグメント ~16GB がアロケータに居座り denoise 中 shared へ ~3.9GB
  溢れ（激遅化）。対策＝`_ltx_worker.py` で**各 denoise 直前に `gc.collect(); torch.cuda.empty_cache()`**（monkeypatch）。
  結果＝denoise shared 3969→**742MB(ambient)**・dedicated ~6.4GB・**生成 −46%**、T2V/初回I2V 実機検証・mock pytest 13 passed・
  凍結境界不変。**※「denoise 工程の持続的溢れ解消」であって「全工程 16GB 内」ではない**（load/encode は依然 ~2–2.5GB shared
  一時溢れ＝下記【残課題B】）。詳細＝VERIFICATION_LOG §7。
- **⚠️ 別件の production ブロッカー（Phase 5B 起因でない既存バグ#5系）＝worker 再利用 crash**：常駐 worker の**2ジョブ目以降**で
  native access violation（exit 139）。詳細・修正方針は下記「★残課題サマリ【残課題A】」。
- **▶（超過 → 最新は §0 末尾の ▶▶▶／VERIFICATION_LOG §9.6）旧・起点メモ＝Stage 3**：**残課題A の真因が判明し理解が一新された。最新は
  [VERIFICATION_LOG.md](VERIFICATION_LOG.md) §8（特に §8.3 真因再フレーム・§8.8 commit 確定・§8.9 過剰commit究明・§8.10 Stage1・
  §8.11 Stage3 設計）。** 要点：**再利用 crash は「safetensors mmap ハンドルのバグ」ではなく「Windows commit（仮想メモリ）
  枯渇」**（per-job の全 submodel 再構築が ~70GB のモデルを毎ジョブ re-mmap＋re-materialize＝~125GB commit スパイク→
  上限到達で native 0xC0000005）。**VRAM レバー（empty_cache 等）も非mmap 化も的外れと実証済**。**次の作業＝Stage 3 ＝
  load-once/keep-resident（フォークの未配線 `StateDictRegistry` を配線）を §8.11 の設計ブリーフどおり実装・検証**。
  Stage 1（捨てる bf16 Gemma 24GB を読まずスキップ）は実装済・併存。詳細・着手手順は下記【残課題A】（更新済）と §8.11。
- **▶▶（超過 → 最新は §0 末尾の ▶▶▶／VERIFICATION_LOG §9.6）§9 方針メモ＝Path B 決定**：Stage 3（StateDictRegistry）は実装したが **job1 warmup が commit 152GB/99% で crash**（§8.11 が留保したリスクが顕在化）。リサーチで**真の解決＝モノリス廃止＝コンポーネント・ファイル分離（Path B）**と確定：VAE/audio/projection を 46GB モノリス＋24GB qat でなく **ComfyUI 同様の小単体ファイル**（Video VAE 1.45GB／Audio VAE 365MB／text projection 2.31GB）から読む＝commit bounded・RAM~32GB・巨大ページファイル/DL 不要・凍結API＋パイプライン無改変。**音声は既にフォークが mp4 へ mux 済みと判明→Phase 1 へ格上げ**（`crop_mp4` の音声保持修正だけ要）。唯一の新規実装＝connector 供給（方式選択中）。詳細・理由・棄却した代替（pread/公式streaming/ComfyUI移行）は **§9** が正本。
- **▶▶▶ 最新の正本・次セッションの起点（2026-06-30 最終・最重要）＝VERIFICATION_LOG §9.7：残課題A（worker 再利用 crash／per-job リーク）は完全解決**。経緯：Path B（component-files）が §9.6 で warmup crash を解消したが job5 で別の per-job リークが露呈（commit・VRAM が generate 毎に単調増加→160.5GB/100% native crash、`block_swap_service.py:86`）。§9.7 で **read-only 並列診断→非破壊・計装run で按分実測→標的2パッチ→6ジョブ実機検証 PASS** により解決：
  - **真因＝`BlockSwapService._installed_transformers` が毎 generate の transformer を append し続け解放しない**（仮説a 確定）。registry は HIT・キャッシュ汚染なし（仮説b/c・in-place .to 説は実測で棄却）。計装で `installed_transformers` 1→2→3→4 を直接観測。leaked transformer の CPU 退避ブロック=commit +18GB/job、GPU 窓=VRAM 床 +1GB/job、で両軸を単一原因に統合。
  - **修正（フォーク製品コード・凍結境界外・persistent）**：①`block_swap_service.py install()` で append 前に `_installed_transformers.clear()`（keep-latest）②`_ltx_worker.py _do_generate()` の `_emit("done")` 直後に `gc.collect(); torch.cuda.empty_cache()`。**`model_ledger.py` の in-place `.to` 改修は不要**（キャッシュ非汚染を実測確認＝vendored `.venv` 不触）。
  - **検証 PASS**：6ジョブ全完走（旧 job5 crash 点突破）、`installed_transformers` 1 で一定、torch 床・denoise peak・commit すべて平坦、**出力 mp4 が修正前と SHA256 バイト一致**（計算不変）、mock 13 緑。詳細＝§9.7。
  - **音声 Phase 1 の実機確認は ✅DONE（2026-06-30 PASS、VERIFICATION_LOG §9.8 参照）**＝native joint audio が 16GB で効果音/音楽/発話を生成・AAC mux・decode VRAM 余裕（dedicated ~2.7GB）・crop 音声保持を確認済。**次の一手＝qat-drop(24GB削減)／720p スケールアップ（残課題C）**。gate＝env `LTX_COMPONENT_FILES=1`＋`LTX_KEEP_RESIDENT=1`。

## ★残課題サマリ（次セッション向け・2026-06-29 時点）

Phase 5(B)＝**denoise 工程の VRAM 低減は達成・実装済**（empty_cache fix／§7.5・§7.8、shared 計測で検証）。
以下がオープン項目。優先度はユーザー判断。

### 【残課題A】worker 再利用の native crash → ✅ 解決済（2026-06-30・VERIFICATION_LOG §9.7）

> **✅ 【2026-06-30 解決】残課題A は完全解決した。** 真因＝`BlockSwapService._installed_transformers` が毎 generate の transformer を保持し続け解放しない（registry HIT・キャッシュ非汚染を計装で実証、仮説b/c は棄却）。修正2点（フォーク製品コード・凍結境界外・版管理済）＝① `block_swap_service.py install()` で append 前に `_installed_transformers.clear()`（keep-latest）② `_ltx_worker.py _do_generate()` の `_emit("done")` 直後に `gc.collect(); torch.cuda.empty_cache()`。6ジョブ実機検証 PASS（旧 job5 crash 点突破・`installed_transformers` 1 で一定・床/denoise/commit 平坦・出力 mp4 が修正前と SHA256 バイト一致・mock 13 緑）。詳細＝VERIFICATION_LOG §9.7。**以下（旧）は診断確定前の歴史的記録。**
>
> **【2026-06-29 全面更新・正本＝VERIFICATION_LOG §8】** 以下の「真因＝mmap 反復ロードの native 破損」「修正方向＝非mmap」は
> **誤診で棄却済み**。実機計測（§8.8）で **真因＝Windows commit（仮想メモリ）枯渇**と確定した。要旨：
> - distilled は各 generate で **全 submodel（Gemma/VAE/transformer）を毎回ディスクから再構築**。Windows では safetensors の
>   mmap が「読む量に関係なくファイルサイズ分の commit を予約」＋get_tensor の materialize で **モデルサイズの ~2×** を commit。
>   per-job で ~70GB のモデルが **~125GB commit スパイク**になり、commit 上限（~146→自動拡大160GB）到達で native 0xC0000005。
>   **VRAM は無関係（crash 時 dedicated 9.5GB）／site は非決定的（safetensors VAE / GGUF dequant 等）＝ヒープ破損でなく commit 枯渇**。
> - **試行と棄却**：①完全 non-mmap whole-file→46GB で MemoryError・FAIL（§8.1）。②Gemma CPU キャッシュ→commit を足して逆効果・FAIL（§8.2）。
>   ③**Stage 1＝捨てる bf16 Gemma 24GB を読まずスキップ→実装済・数値不変・commit −10〜16GB だが job3 でまだ crash＝不十分**（§8.10）。
> - **★次セッションの作業＝Stage 3：load-once/keep-resident**。フォークに**存在するが未配線**の `StateDictRegistry`（`registry.py:49-84`）を
>   配線し、submodel を warmup で1度だけロード→全ジョブ HIT 再利用＝per-job 再 materialize を消し commit を steady **~62GB** に。
>   **配線は `ltx_fast_video_pipeline.py:137`（DistilledPipeline 構築直後・service install 前）に ~4行＋Gemma を `_target_device()`
>   準拠（CPU build）にする1修正**。設計・commit試算・VRAM中立・GGUF互換・要確認点は **VERIFICATION_LOG §8.11** に詳細。
>   検証＝reuse_loop（CACHE=0 EC=1 BS=8）＋commit/VRAM サンプラで全5ジョブ完走・commit steady・出力不変・mock 13。
>   **ユーザー作業原則**：実装前に計画提示→承認、計測はサブエージェントに「終了まで block＋同一ターン報告（Monitor 禁止）」を課す。
> - **現コード状態（known-good＋Stage1）**：worker は pre-denoise empty_cache fix のみ＋Stage1 の bf16 Gemma skip。GGUF `copy=True` 温存。
>   `services/gemma_sd_cache.py`・`services/sft_loader_safe_patch.py` は dormant（未配線）。mock pytest 13 passed・凍結境界不変。
>
> 以下（旧）は歴史的記録。

- **症状**：常駐 worker（Approach W）の **2ジョブ目以降の generate 冒頭で Windows native access violation
  （exit 139 / 0xC0000005）**。worker が無言で死亡→`pipeline_manager._cleanup_after_error→unload→次ジョブで再ロード`で
  復帰するが当該ジョブは失敗。warm 運用で断続的に再発（単発/初回は常に成功）。
- **Phase 5(B) 起因ではない**：対照実験（fix 有/無）で **empty_cache fix と無関係**と確定（両方 job2 で crash）＝既存の
  **バグ#5（access-violation）系**。Phase 5A の「reuse 両 PASS」は非決定的な幸運だった。
- **真因**：distilled が各 generate で Gemma テキストエンコーダを `del→全再構築`（VRAM 節約のため）。毎回 **base Gemma
  safetensors を再 mmap ＋ GGUF Gemma を再 dequant**。この反復重ロードが本機 Windows で断続的に native 破損。crash site
  は複数・非決定的：`sft_loader.py:36`（mmap read→`torch storage __getitem__`）と `gguf_quant_service.py:402 _dequant_q6_k`。
- **失敗した試み**：`sft_loader_safe_patch`（copy=True eager 化）配線→**効かず撤去済**（fault は上流の mmap materialize＝
  `get_tensor`、かつ GGUF dequant site は非カバー）。詳細 §7.7。
- **修正方向（優先順位を補正）**：
  1. **［最優先・小・先行事例準拠＝未試行］完全 non-mmap ロード**。コミュニティの実証済み回避策は **mmap をやめる**こと
     （ComfyUI `--disable-mmap`／#13220）。我々の safe-patch は `safe_open`+`get_tensor`（＝mmap のまま）を残し `.to(copy=True)`
     だけ変えたので**効かなかった**（fault は上流の `get_tensor` の mmap 実体化）。→ **safetensors を `safetensors.torch.load_file`
     （非mmap / `backend="pread"`）化**＋**GGUF dequant も mmap 非依存化/計算前コピー**（2つ目の crash site `_dequant_q6_k` は
     city96 #444/#416 と同系の mmap エイリアス疑い）。両 site を non-mmap 化して reuse loop で再検証。
  2. **［大・代替］** 1 で消えなければ **per-job 再構築を廃し Gemma をジョブ間キャッシュ**（VRAM 常駐は 16GB fit と非両立 →
     CPU RAM 側キャッシュ・encode 時のみ GPU）。
  - **要 research**：ComfyUI `--disable-mmap` 実装、フォーク backend が Windows 再利用でどう振る舞うか（Linux 前提か／
    per-request で何を再ロードするか）、公式 CLI が一発実行ゆえ無症状なだけか。**この crash は Windows 固有**
    （safetensors #164 の mmap ハンドル寿命）で、公式/フォークが「解決した」のではなく発症条件（Windows＋同一プロセス反復
    mmap）を踏んでいないだけ、が現時点の理解。詳細 §7.6/§7.7。

### 【残課題B】load/encode の一時 shared 溢れ（「全工程 16GB 内」は未達）
- **症状**：denoise は ambient だが、**Gemma encode（shared ~2.0GB）と transformer load（shared ~2.5GB）は依然 16GB を
  一時超過**（dedicated ~15.9GB＋shared）。ユーザー目視と一致。global shared ピークは 3.9GB→2.5GB に減ったがゼロではない。
- **原因**：transformer load＝block-swap が**一旦 full-GPU ロード(peak 16.9GB)→退避**。Gemma encode は ~15.1GB で本質的にタイト。
- **修正方向**：transformer ブロックを**直接 CPU ロード**（full-GPU を経由しない）。Gemma 側は残課題A の load-path 作り替えと
  地続き。詳細 §7.9。

### 【残課題C】スケールアップ 1280×768→720p クロップ（本来の機能目標）→ ✅ 完了（2026-06-30 後半・冒頭 ▶▶▶▶ と VERIFICATION_LOG §10 が最新）
> ✅ **達成**：1280×768/121f を本番 API で完走（~167–171秒・16GB）、crop で 1280×720 配信、連続3本も commit 枯渇せず PASS。レシピ＝comp=1（Path B）＋keep_resident=0＋bs=8＋vae 512/64。真の難所3点の実測結果は冒頭 ▶▶▶▶／§10 参照。以下（旧）は着手前の調査メモ。
- → 詳細調査は Docs/SCALEUP_16GB_RESEARCH.md に集約（レバー棚卸し＋コミュニティレシピ＋真の難所3点）
- ★訂正(2026-06-30)：以下「修正方向」の **「attention tiling 配線（未配線）」前提は古い**。SCALEUP doc で判明＝attention tiling は既に実装済（`attention_tile_service.py`・`attention_tile_size`、既定OFF）。VAE 空間/時間タイル・二段・block-swap も配線済。**未実装は FFN チャンキングのみ（主に長尺向け・1280×720 には非必須）**。真の難所は移植でなく (1)段間遷移スパイク (2)22B block-swap 深度 (3)tiling＋GGUF＋block-swap 共存。**目標は 1280×720 で充分**。次の一手の手順は本書冒頭「次セッション開始点」＋ SCALEUP doc §5/§6 を参照。
- 384x256 の denoise は解消済だが **production ターゲットは 720p**。高解像度では denoise の**アクティベーションが支配的**に
  なる（384x256 で効かなかった活性軸技法が高解像度で本命）。
- **修正方向**：**attention tiling 配線**（`attention_tile_size`＝フォークに実装済・我々の worker/config で未配線）＋
  **FFN チャンキング移植**（RandomInternetPreson `ChunkedFFN`→`ltx_core/.../feed_forward.py:FeedForward.forward`）。高解像度
  VAE decode には VAE tiling も。empty_cache fix は高解像度でも有効。
- **要・新規診断**：1280×768 で改めてどの軸が支配的かを §7 と同じ手法（perf-counter dedicated/shared フェーズ別）で計測して
  から技法選択。詳細 §7.1、LTX23_REFERENCE §5。

### 【残課題D】後回し（低優先）
- README / `LTX23_Backend_Specification_v04…` は pre-pivot のまま＝全面改訂は**スケールアップ完了後**まで保留。
- 他の `non_blocking=True` 箇所（`fuse_loras.py`／`gemma_gguf_quant_service.py` の owned-tensor 転送）は mmap 非依存で低リスク＝据え置き。
- 品質バンプ：Gemma Q6_K（忠実度↑、encode に余裕あれば）。

**相互関係/推奨着手順（一例）**：A と B は「Gemma/transformer のロード経路」で重なる（A の load-path 作り替えが B も部分解消）。
C は独立の機能拡張。**A（信頼性ブロッカー）→ B → C（720p）** が一例だが優先度はユーザー判断。

---

## 1. アーキテクチャ（二層・これを壊さない）
- **凍結層（不変の資産）**: REST API 形（`api/models.py`・**÷64 解像度契約**）、ジョブ管理、出力 `outputs/{job_id}/output.mp4`+`metadata.json`、`LTXRunner.generate(...)→GenerationOutcome` 契約、**mock backend**（torch 無し app `.venv` のテスト用・温存）。
- **エンジン層（実証済・Phase 5(A) で取り込み済）**: フォーク `vendor/LTX-Desktop-LOW-VRAM/backend/` の `LTXFastVideoPipeline`。公式 `DistilledPipeline`(ltx_core@00dc53d) をラップし、(a) GGUF transformer + block-swap、(b) **GGUF Q4_K_M Gemma**、(c) VAE/attention tiling を配線。常駐 worker `_ltx_worker.py` 経由で `_RealBackend` から呼ばれる（§3）。

## 2. いま整っている資産（実機確認済）
- **フォーク env**: `vendor/LTX-Desktop-LOW-VRAM/backend/.venv`（LTX-2 を git rev `00dc53d` に pin＝ModelLedger API。torch 2.9.1+cu128 / gguf / 全 import OK）。
- **モデル**:
  - transformer GGUF: `models/ltx-2.3-gguf/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf`（17.76GB）
  - **Gemma GGUF**: `models/gemma-3-12b-it-gguf/gemma-3-12b-it-Q4_K_M.gguf`（7.30GB, `ggml-org/gemma-3-12b-it-GGUF`, ungated, arch gemma3, 量子化 {F32,Q4_K,Q6_K}）
  - safetensors（VAE/audio源・projection源・tokenizer/config・vision）: `models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors`, `…-spatial-upscaler-x2-1.1.safetensors`, `models/gemma-3-12b-it-qat/`
- **実装済（フォーク側）**:
  - 新規 `vendor/.../backend/services/gemma_gguf_quant_service.py` — GGUF Q4 Gemma を per-layer dequant で GPU 推論。`gguf_quant_service.py` の bit-exact カーネル再利用＋city96 キーremap＋RMSNorm−1補正＋GGUF Gemma×safetensors projection の merge＋embed CPU offload。
  - `vendor/.../backend/services/fast_video_pipeline/ltx_fast_video_pipeline.py` に `gguf_gemma_path` 引数＋`_install_gemma_gguf`（`cpu_text_encode` と排他）。
- **実装済（我々の backend 側）**: ÷64 契約（`api/models.py`）、`services/ltx_runner.py` の facade＋`_MockBackend`(温存)＋`_RealBackend`(**Phase 5(A) で Approach W 常駐ワーカーへ実体化済＝§3**)、`video_io.crop_mp4`、`config.model.backend`(auto/mock/real)＋`config.model.{gguf_*,fork_*}`。mock pytest 13 passed。

## 3. Phase 5(A)：フォークエンジンを我々の backend へ取り込む

> ## ✅ 完了 (Phase 5A) — 2026-06-28
> **凍結 REST API が実エンジンで生成するようになった。** `services/ltx_runner.py` の `_RealBackend` を
> **Approach W（常駐サブプロセスワーカー）** で実体化。公式 `DistilledPipeline`（本機で native crash）は使わず、
> フォーク `vendor/LTX-Desktop-LOW-VRAM/backend/_ltx_worker.py` を **フォーク env** で常駐起動し、JSON-lines
> プロトコル（`@@LTX@@` フレーム）でジョブを受けてエンジンが `output.mp4` を共有 output dir に直接書く。app `.venv`
> は torch/engine を一切 import しない。**凍結境界は不変・mock pytest 13 passed**。実機検証＝384x256/9 の T2V/I2V
> 完走（VERIFICATION_LOG §6）。実装計画 `~/.claude/plans/a-witty-kazoo.md`。
> **次＝Phase 5(B)（このセクション末尾「次＝Phase 5(B)」を参照）。**
>
> **なぜ Approach W か**（旧ハンドオフの「`services/lowvram/` へコピー」案でも in-process 案でもなく）:
> エンジンは torch＋ltx_core@00dc53d＋gguf を要し、それらは**フォーク env にしか無い**。さらに app と
> フォークは**双方とも `services` という同名トップレベルパッケージ**を持ち、フォーク側 `__init__` が ~18 の
> torch 依存モジュールを eager import する → **同一インタプリタで共存不可**。二プロセス分離なら衝突を完全に
> 回避でき（rename 不要）、**検証済みのフォークコードを無改変で**使える。

**目的**: 我々の FastAPI が、この実証済みパイプラインを実際に配信できるようにする。**凍結層は不変**。

（以下は Phase 5(A) で実際にどう配線したか＝seam の正確な記録。次セッションが実装を追う地図として残す。
詳細は計画書 `a-witty-kazoo.md`、検証は VERIFICATION_LOG §6。）
1. **エンジンの置き場所**: フォークの `services/{fast_video_pipeline, gemma_gguf_quant_service, gguf_quant_service, block_swap_service, ltx_pipeline_common}.py` 等を我々の `services/lowvram/` に取り込む（or フォークを実行基盤として import）。**重要・実行環境**: 我々の backend は **フォーク env(00dc53d) で動かす**必要がある（1.1.6 は ModelLedger 無し＋crash）。足りない依存（pyyaml/pillow/httpx/python-multipart 等）は同 env に追加。gradio は別プロセス（hub 依存衝突回避）。
2. **`services/ltx_runner.py` の `_RealBackend` を作り替え**: 公式 `DistilledPipeline` 直叩き（crash）をやめ、`LTXFastVideoPipeline.create(... gguf_transformer_path=, gguf_gemma_path=, block_swap_blocks_on_gpu=, vae_*_tile_size=)` → `.generate(prompt, seed, height, width, num_frames, frame_rate, images, output_path, num_steps=8)` を呼ぶ。circular import 回避の warm import（`import ltx_core.loader`）を初期化に。
3. **config 追加**: `model.gguf_transformer_path`、`model.gguf_gemma_path`、`vram.block_swap_blocks_on_gpu`、VAE tile sizes 等。`services/low_vram.py` のマッピング更新。`backend` 名は `ltx-distilled-gguf` 等に。
4. **出力**: `LTXFastVideoPipeline.generate(output_path=...)` が mp4 まで書く。`crop_output` 指定時は既存 `video_io.crop_mp4` で後段クロップ。
5. **検証**: ①mock pytest（app `.venv`）が緑のまま ②real smoke 384x256/9 T2V → mp4・peak VRAM フェーズ別・時間 を実測（フォーク env）③phase1_default→最小I2V→phase1_target。落ちたら**勝手に直さず原因分析して報告**。

## 3b. Phase 5 精密ポインタ（backend 凍結契約マップ・2026-06-28 調査済）
次セッションが即着手できるよう、我々の backend 側の正確な契約と差し替え箇所を記す（行番号は目安、シンボルで追うこと）。

**差し替える対象＝`services/ltx_runner.py` の `_RealBackend` のみ**（`load()`/`generate()`/`unload()`, おおよそ L327–500）。現行は公式 `DistilledPipeline` を import・呼び出し（crash する死に筋）。ここを `LTXFastVideoPipeline.create/generate` 呼びに置換。

**変えてはいけない凍結境界**:
- **契約シグネチャ**（`ltx_runner.py`）:
  `LTXRunner.generate(request: GenerateRequest, output_dir: Path, progress_callback: ProgressCallback|None, conditioning_image_paths: list[Path]|None) -> GenerationOutcome`
  - `ProgressCallback = Callable[[int|None,int|None,float],None]`（real は `(None,None,progress)` で良い）
  - `GenerationOutcome(output_path: Path, seed_used: int, peak_vram_mb: int|None, generation_mode: str, backend: str)`。`output_path` は必ず `output_dir/"output.mp4"`。`generation_mode` は `request.generation_mode`（i2v/t2v）。`backend` 文字列は新値（例 `ltx-distilled-gguf`）にして良い。
- **唯一の呼び出し元**＝`services/pipeline_manager.py`（~L135）。`output_dir = config.output_dir/job_id`、`conditioning_image_paths` は `upload_store.path_for(image_id)` で解決済み絶対パス。`metadata.json` は **pipeline_manager 側が `_finalize()` で書く**（runner は `output.mp4` だけ書く）。`outcome.peak_vram_mb` は `low_vram.metadata_block()` に渡る。
- **解像度/フレーム検証**＝`api/models.py`（`GenerateRequest` validator）: **width/height ÷64**、`(num_frames-1)%8==0`、distilled は `num_inference_steps==8`/`guidance_scale==1.0`、I2V は最大1枚・`frame_idx==0`。
- **I2V 型**: `ConditioningImage(image_id, frame_idx=0, strength, crf|None)` → 現行は公式 `ImageConditioningInput(path, frame_idx, strength, crf=33既定)` に変換。フォーク engine 側の画像条件型に合わせて変換し直すこと（フォーク `api_types.ImageConditioningInput`／`generate` の image 引数仕様を次セッションで再確認＝この探索は中断済）。
- **video_io**: `encode_frames_to_mp4(frames, output_path, frame_rate, crop=None,...)`／`crop_mp4(in,out,w,h)`／`save_metadata(path,dict)`。crop は `request.crop_output` 指定時に既存 `crop_mp4` で後段クロップ。
- **テスト固定**: `tests/conftest.py` が `model.backend="mock"` を強制＋`LTX_DISABLE_GRADIO=1`。**mock 経路と pytest は緑のまま**にする（torch 無し app `.venv` で回る）。

**backend 選択 & 可用性**: `_select_backend()` が `config.model.backend`（auto/mock/real）で分岐。`_real_available()` は現状 **`import ltx_pipelines`＋`torch.cuda`＋モデルパス存在**で判定 → **Phase 5 では「フォーク engine が import 可能か＋GGUF パス存在」に更新**が要る（公式 1.1.6 ではなくフォーク env で動かすため）。

**low_vram knob**（`services/low_vram.py` `LowVramSettings`）: 現在 engine に効くのは `low_vram_mode`/`fp8_transformer`/`vae_tiling` のみ。`block_swap`/`attention_tiling`/`cpu_offload_text_encoder` は **定義済みだが未配線** → Phase 5 で `block_swap_blocks_on_gpu` 等として `LTXFastVideoPipeline.create` に渡す配線を追加。

**config 追加（Phase 5）**: `model.gguf_transformer_path`／`model.gguf_gemma_path`／`vram.block_swap_blocks_on_gpu`／VAE tile sizes 等。パス解決は既存 `config._abs`(PROJECT_ROOT 基準) を踏襲。

**フォーク engine 呼び出しの実証済み雛形**: `outputs/phase4_gguf_gemma/run_t2v_bs8.py`（384x256/9 が通った `create(...)+generate(...)` の実値一式）と `vendor/.../backend/_spike_gguf_min.py`。`create()` の全引数は `ltx_fast_video_pipeline.py` 参照（`gguf_transformer_path`/`gguf_gemma_path`/`block_swap_blocks_on_gpu`/`vae_*_tile_size`/`cpu_text_encode`(排他) 等）。Phase 5(A) ではこの呼び出しを **常駐ワーカー `_ltx_worker.py`** に内包した（worker がモデルを1度だけ構築しジョブを使い回す）。

## 3c. （歴史的）Phase 5(B) 起点メモ＝当時の仮説

> **【超過 2026-06-29】このセクションは Phase 5(B) 着手時の仮説メモ（歴史的）。** denoise の shared 溢れは
> **解決済**（§0／VERIFICATION_LOG §7.5）＝真因は当時の想定（block_swap 深度等）ではなく Windows アロケータ居座りで、
> 対策は denoise 直前 `empty_cache`。**現状と次の一手は §0「★残課題サマリ」を参照**。以下は当時の記録。
Phase 5(A) で「配線」は終わったが、**"16GB に本当に収まる" はまだ未達**。denoise フェーズが（384x256 でも）shared メモリへ溢れている（ユーザーがタスクマネージャで目視確認、VERIFICATION_LOG §6/§5 の "denoise shared ~3.6GB @bs8" と整合）。WDDM が溢れ分を system RAM へページングするため**ハード OOM はしないが激遅化**する。これが残った主タスク。

- **ユーザー方針＝これは「Phase 5(B) スケールアップ（1280×768 → crop 720p）」と一つの同じ仕事**。スケールアップ用の denoise-stage VRAM 技法（block_swap 深度を bs4/bs2 へ／VAE・attention タイリング／解像度依存の sequential・streaming）を適用すれば、**現状の 384x256 の shared 溢れも一緒に解消**する見込み。
- **着手の起点＝先行事例の denoise-stage 技法を調査・複製**（独自発明しない、[[research-prior-art-first]]）: フォークの `vendor/.../backend/services/block_swap_service.py` 等、および ComfyUI-GGUF / ComfyUI カスタムノード・ワークフローの解像度対応技法。これらは**まだ我々の backend に反映していない**。
- 計測は perf-counter の "Shared Usage" サンプリングで（§5。worker の `max_memory_allocated` では溢れを検知できない＝§4）。
- **重要・過去ログの読み替え**: VERIFICATION_LOG §2.3 で **bs4 が crash** したのは、当時 Gemma が bf16 で ~17.7GB を共有メモリへ溢れさせ、その圧迫下で遅延ローダが access violation した（バグ#5）ため。**この前提は Phase 4/5(A) の GGUF Q4 Gemma 化で既に解消済**＝「bs4 は落ちる」は**もう当てはまらない**。よって Phase 5(B) は bs8→bs6→bs4→bs2 を**クリーンに再計測**してよい（各深度で dedicated/shared 両ピーク＋wall-clock を §5 のサンプラで測り、溢れ消失と速度のバランス点を探す）。

## 4. （超過）旧・既知の残課題リスト

> **【超過 2026-06-29】この §4 は古い。現行の残課題は冒頭「★残課題サマリ」（A=worker 再利用 crash／B=load/encode
> 一時溢れ／C=720p スケールアップ／D=後回し）と VERIFICATION_LOG §7 を正とする。** 以下は歴史的参照（denoise spill は解決済）。
- **★【最優先・真の 16GB fit】denoise フェーズの shared 溢れ**。これはもはや「block_swap の小ノブ」の後回し課題ではなく、**残った主タスク**であり、上記 **Phase 5(B) スケールアップ（1280×768→crop）と同一の作業**（§3c）。384x256 でもユーザーがタスクマネージャで shared 溢れを目視確認済。**注意: worker が報告する `torch.cuda.max_memory_allocated`（T2V 16913 / I2V 17989 MB）は dedicated と WDDM shared を区別できず、この溢れを検知できない** → 検知には perf-counter の "Shared Usage" 直接サンプリングが要る（§5）。フォーク `block_swap_service.py` 等・ComfyUI-GGUF の denoise-stage 技法を複製して詰める。
- **品質バンプ（任意・将来）**: Gemma を Q6_K に上げると忠実度↑（KL的に Q4 の3倍正確）。同じローダで GGUF 差し替え1つ、encode フェーズに余裕あり（Q6 ~9.6GB→encode ~12GB）。今は実績ある Q4 で完走しているので必要時に。
- **未使用経路の地雷**: フォークの音声/IC-LoRA/dev-HQ は未検証。使う時に個別検証（[[ltx-desktop-lowvram-fork]] 留意点参照）。

## 5. 検証・計測の道具
- **★VRAM 実測（テスト担当サブエージェントへの必須手順）**: 実生成の検証では **専用GPUメモリ(dedicated)だけでなく共有GPUメモリ(WDDM shared)も必ず監視する**。`torch.cuda.max_memory_allocated`（＝worker 報告 `peak_vram_mb`）は dedicated/shared を区別できず **WDDM 共有溢れを見逃す**（Phase 5(A) で「16GBに収まった」と誤判定した原因＝§4/§6.4）。
  - **使う道具＝既存サンプラ `vendor/LTX-Desktop-LOW-VRAM/backend/_gpu_mem_sampler.ps1`**（タスクマネージャと同一ソースの perf-counter `\GPU Adapter Memory(*)\Dedicated Usage` と `\…\Shared Usage` を全アダプタ最大で既定0.5s間隔サンプリング→ `timestamp,dedicated_MB,shared_MB` を CSV 出力。引数 `-OutFile/-IntervalSec/-MaxSeconds`）。
  - **手順**: 生成の直前に**バックグラウンド起動**（例 `powershell -ExecutionPolicy Bypass -File <repo>/vendor/LTX-Desktop-LOW-VRAM/backend/_gpu_mem_sampler.ps1 -OutFile outputs/gpu_mem_<tag>.log -IntervalSec 0.5`）→ 生成後に停止 → ログの **dedicated_MB と shared_MB の両ピークを必ず報告**し、worker のフェーズ print（Gemma encode / denoise / VAE）と wall-clock で突き合わせて溢れフェーズを特定。`torch` 側の `max_memory_allocated` は補助としてのみ併用。過去ログ例: `outputs/gpu_mem_*.log`（§2.2 / §2.4 に読み方）。
- Gemma 単体の正しさ検証スクリプト: `outputs/verify_gemma_gguf/`（dequant bit-exact / キー一致 / norm−1 / forward 49層有限 / bf16 cosine）。
- E2E スパイク: `outputs/phase4_gguf_gemma/`（runner・perf log・出力 mp4・contact_grid・GIF）。フォーク最小呼び出し雛形 `vendor/.../backend/_spike_gguf_min.py`。

## 6. ドキュメントの正本/古い注意（重要）
- **正本**: 本書（handoff）／[VERIFICATION_LOG.md](VERIFICATION_LOG.md)（**§6＝Phase 5(A) 配線・実機検証が最新**、§5＝Gemma GGUF）／計画書 `a-witty-kazoo.md`／memory。本書と VERIFICATION_LOG は **Phase 5(A) の状態を反映済**。
- **LTX 2.3 一般リファレンス（参照URL付き基礎知識）**: [LTX23_REFERENCE.md](LTX23_REFERENCE.md) — 解像度契約(÷32/÷64・2段)・VAE 32×圧縮とトークン数・VRAMスケーリング(重み支配)・16GBレシピ・720pの作り方(1280×768→crop)。タスク非依存の事実集。
- **古い・歴史的（鵜呑み禁止）**:
  - [DESIGN_COMPARISON_and_direction.md](DESIGN_COMPARISON_and_direction.md): fp4_mixed 推奨だったが**不採用**（ComfyUI密結合/cu130前提）。GGUF Q4 採用が結論。比較表の枠組みは有効。
  - [note.md](note.md): 公式 offload+fp8 **戦略は破棄**（本機で native crash）。ただし torch2.9.1/cu128・SDPA(=FlashAttn-2)・xformers任意 の**事実は有効**。
  - `README.md` / `LTX23_Backend_Specification_v04_…md`: **pre-pivot のまま**（公式 DistilledPipeline＋fp8-cast＋xformers、ランナー=モック、一部 ÷32 表記）。**現行の正しい解像度契約は ÷64**（`api/models.py`）。engine の真実は本書/VERIFICATION_LOG。README/spec の全面改訂は **Phase 5(B) 完了後まで保留**（VRAM の真の 16GB fit が実現＝実エンジンの実像が確定してから）。Phase 5(A) で配線は済んだが、VRAM fit が未達のうちは実像が未確定のため。

## 7. 作業原則（ユーザー）
- コード着手前に**実装計画を提示して合意**（plan mode 段階承認）。**テストが落ちたら勝手に直さず原因分析して報告**。
- 編集・テスト実行・生成・DL は**サブエージェントに委譲し、本体は監督**。環境隔離厳守（システム Python を汚さない／全てプロジェクト内 `.venv`/`.python`/`hf_home`/`.uv_cache`）。
- **先行事例のソースを複製し独自発明しない**（[[research-prior-art-first]]）。大きな pivot は相談してから。
