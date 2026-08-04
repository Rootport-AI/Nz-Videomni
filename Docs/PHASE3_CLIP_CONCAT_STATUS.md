# Phase 3 スライス2「クリップ連結」現状ステータス — masked AV-latent 連結で再実装・実機検証PASS

> **（歴史記録・2026-07-06 注記）✅mainマージ済（merge `2cc4cac`）。** 本文の「Chain A/B 本番出力の最終目視/試聴 PENDING」は当時の状態。クリップ連結の映像/試聴ゲートはその後クローズ（720p 級2セグで継ぎ目不可視まで実証・VERIFICATION_LOG §19）。GUI 経由クリップ連結＋結合の症状も解消済（§27）。現在の目視宿題は [`NEXT_SESSION_HANDOFF.md`](NEXT_SESSION_HANDOFF.md) 冒頭が正。

- 更新: 2026-07-03（本セッションで旧アーキから作り直し）
- 併読: [`PHASE3_CLIP_CONCAT_DESIGN.md`](PHASE3_CLIP_CONCAT_DESIGN.md)（研究/設計・訂正バナー＋採用アーキテクチャ確定note）／[`VERIFICATION_LOG.md` §19](VERIFICATION_LOG.md)（本セッションの全ゲート詳細）／[`NEXT_SESSION_HANDOFF.md`](NEXT_SESSION_HANDOFF.md)（引き継ぎ）

---

## ⚠️ 一行結論

**公式パリティ「masked AV-latent 連結」アーキテクチャで再実装・実機検証PASS**（境界の連続性を「境界フレーム目視/画素差分ハーネス」で計測し、ユーザーが known-bad アーティファクトへの較正結果を確認済み）。**ただし Chain A/B 本番出力そのもののユーザー最終目視/試聴はまだ PENDING**（本文書「pending 項目」参照）。旧方式（stage1 latent tail carry を disarm 後の独立 stage2 refine に継がせる方式）は 2026-07-03 冒頭で目視 hard-cut FAIL が確定しており、完全に置き換えた。

## 旧方式と真因（履歴として保持）

2026-07-03 冒頭時点でユーザー目視により確定していた FAIL:
- 2clip は frame24→25、4clip は49→50/89→90/129→130 で hard cut・音声も断絶。クリップ内は滑らかだが境界で完全不連続。
- 根本原因: ①二段パイプラインのミスマッチ（carry=stage1 低解像度 latent／表示=stage2 refine後、各clip独立refine）②carry が弱すぎ短すぎ（K=2 latent・soft strength 0.5）③overlap を concat で trim 破棄④音声は各clip完全独立生成⑤AdaIN/クロスフェード/stage2 overlap 凍結が未実装。
- 検証手法の欠陥: 「テンソル一致・byte-match・VRAM」は配管の検証に過ぎず、映像連続性を一度も目視していなかった。

詳細は git 履歴の本ファイル旧版（commit `876b8c5`）および `VERIFICATION_LOG.md` §18 に温存。

## 新アーキテクチャ（2026-07-03 確定・masked AV-latent concatenation）

構造上の要点（詳細は `chain_math.py` docstring／`engine/pipeline/chain_pipeline.py`／`VERIFICATION_LOG.md` §19.2 の wheel 深掘り）:

1. **root cause の再確認（WP2 read-only wheel 深掘り）**: 旧方式の hard cut は「per-clip VAE decode が causal VAE の "先頭 latent フレーム=1pixel" リセットを毎境界で再トリガーする」＋「音声が各clip完全独立生成」が本質。position はデノイズ呼び出しごとに再構築されるが、VAE decode は position-free かつ既に時間タイル化済み → 「1本の連続 stage1 AV latent を組み立てる → 1回の stage2 → 1回だけ decode」にすれば構造的にシームレスになる。
2. **アーキテクチャ**: per-segment STAGE1（half-res）で video+audio latent tail を carry+freeze → **1本の連続 stage1 AV latent を組み立て**（overlap は線形クロスフェード。各接合部の音声 overlap 長 K_a は「組み立て後の音声長が stage2 ターゲット長にちょうど一致する」よう逆算）→ 全タイムラインを **1回だけ** upsample → **STAGE2 refine を時間タイル分割**で実行（tile=22 latent フレーム／overlap=4 hard-freeze後ブレンド。短いチェーンは単一タイルに縮退）→ **1回だけ** VAE decode → 1本の mp4。
3. **両モダリティの RoPE 20秒天井**: video・audio とも RoPE 位置に20秒の学習上限がある（audio は秒単位の位置・`max_pos=[20]`）。stage2 を常に時間タイル化することで各タイルが位置を0から再スタートし、20秒超のチェーンでも位置レンジ内に収まる。
4. **公式先行事例との整合（WEB調査・github Lightricks/ComfyUI-LTXVideo コード確認）**: 公式 extend/looping サンプラーは video-only（AV latent では明示的に raise）。公式の音声連続性機構は `LTXVSetAudioVideoMaskByTime`（モダリティ別 preserve mask＋線形ランプ）＋標準サンプラー。継ぎ目は `LinearOverlapLatentTransition` クロスフェード。last-frame I2V チェーンは構造上 audio を運べない → 音声連続性必須の本要件では不採用。
5. **wheel 非改変**: engine が wheel 関数を直接呼ぶのみで monkeypatch 不要（旧 `_EXTEND` monkeypatch 機構は現在未使用、下記「保留事項」参照）。

## 検証手法の是正（本セッションの前提）

旧セッションの「配管PASSのみで機能PASS判定」という欠陥を是正するため、本セッションでは:
- **WP1（先行）**: 境界連続性検証ハーネスを実装より先に構築（`services/video_io.py` の `frame_count`/`extract_frame_at`/`extract_last_frame` ＋ `outputs/phase3_clip_concat_spike/verify_boundaries.py`）。
- **既知の bad アーティファクトへ較正**: 旧方式（ユーザー確認済み hard cut）を再解析し、video MAD 比 3倍・audio 30msRMSジャンプ比 6倍のしきい値で **全既知 hard cut を検出**（video 9–11倍・audio 8.8–17.5倍）、intra-clip サンプル11件で偽陽性0。
- 全 WP で「客観PASS」と「ユーザー目視/試聴」を明確に分離し、ユーザー確認済みの箇所とメトリクスのみの箇所を区別（詳細は VERIFICATION_LOG §19）。

## 検証結果サマリ（詳細は VERIFICATION_LOG §19）

| 段階 | 内容 | 客観結果 | 目視/試聴 |
|---|---|---|---|
| ハーネス較正 | 旧hard-cutアーティファクトへの較正 | 既知不連続を全検出・偽陽性0（11サンプル） | ユーザーが判定基準を確認済み |
| S1（2セグ, K_v=3, strength 0.5） | GPU spike | 全境界 continuous・VRAM 8.85GB | ユーザー目視 PASS |
| S2（4セグ 529px/22.04s, stage2タイル） | GPU spike | 30プローブ全 continuous | ユーザー: タイル継ぎ目に破綻なし。継ぎ目直後の subtle drift 2件を発見（チューニング backlog） |
| S3（発話プロンプト） | GPU spike | 音声メトリクス・スペクトログラムとも continuous | 境界での「口が~0.25秒閉じて再度開く」アーティファクトを分析で分離（strength 0.5/0.8/1.0 スイープで strength は原因でないと確認）。ユーザーは視聴時に気づかず＝v1として受容、backlog記録 |
| WP4 本番配線 | byte-match回帰＋pytest | T2V `23844b4e…`／I2V `a511eda4…` 不変・pytest 41 green | — |
| Chain A（実機・2×73f/147s） | 全境界 | continuous | **PENDING**（下記） |
| Chain B（実機・4×145f/22s） | 映像境界7件 | 全 continuous | **PENDING**（下記） |
| Chain B 音声 | J=456で閾値超過（12.63倍） | 調査の結果 speech onset（無音→発話）での既知の誤検知パターンと判定（下記） | **PENDING**（下記） |

### Chain B の J=456 音声フラグ＝調査済み・harness false-positive 判定

- `boundary_metrics.json`: junction 456 の audio ratio=12.63（閾値6超過・`verdict: DISCONTINUOUS`）、同 junction の video ratio=0.84（閾値3内・`verdict: continuous`）。
- 調査: 同一チェーンの非境界地点で 25–46倍のジャンプが見られる一方、この境界では 12.6倍（非境界より低い）。同じ overlap ジオメトリは S2 spike では pass している。境界時刻 t≈19.04s は無音→発話の立ち上がり（onset）に一致。
- **結論（暫定）**: 無音→発話の onset は音声 RMS ジャンプそのものが大きく、境界か否かに関わらず閾値を超えやすい＝ハーネスの speech onset に対する既知の弱点。映像は同境界で continuous。**ユーザーの実試聴での確定判断は PENDING**（下記）。

## ✅ ユーザー最終目視/試聴 結果（2026-07-03・Chain A/B）

- **Chain A: PASS**。71→72境界は自然に繋がっている（背景がピンボケの動画であることもあり大きな問題なし）。
- **Chain B: 継ぎ目・音声すべてPASS**。セグメント継ぎ目 143/271/399・タイル継ぎ目 167/311/455・t≈19.0sの音声（harness偽陽性判定の件）いずれも自然に繋がっていることをユーザーが確認＝**本機能の目視ゲートは消化**。
- ただし継ぎ目とは別の**新規アーティファクト3件**を発見（下記backlog 4-6）。
- ユーザー総評: **512×320級の検証解像度は低すぎて品質判断に不向き**（顔が溶ける・背景がぼやける）。プロンプトの「CM風」も品質を下げた可能性（ネット上のTVCMは低品質が多い）→ 以後の目視検証は**720p級＋映画トレイラー風プロンプト**で行う（運用指針化済み）。

## 既知の品質特性・チューニング backlog（機能は成立、磨き代として記録）

1. **タイル継ぎ目直後の subtle drift**（S2 spike で発見）: 171→172フレーム付近の雲、470→471フレーム付近の波に、継ぎ目そのものではなく**継ぎ目の少し後**で微妙なドリフトが視認された。
2. **発話チェーンの境界での口パウズ**（S3 spike で発見）: チェーン境界で「口が約0.25秒閉じて再度開く」アーティファクト。standalone 生成（seg0単体）には存在しない＝チェーン由来。strength 0.5/0.8/1.0 スイープで strength は原因ではないと判明。おそらく全タイムライン stage2 が「発話→ポーズ→発話」という2つの発話区間を再調停する挙動。ユーザーは視聴時に気づかず、v1として受容。
3. **ハーネスの speech-onset 偽陽性疑い**（Chain B J=456・上記）。
4. **【2026-07-03 目視で追加】Chain B 167-182f: 背景がぐにゃりと歪んで少し変わる**（タイル継ぎ目167直後＝backlog#1のdriftと同族の可能性）。
5. **【同】Chain B 346-370f: 巨大なアーティファクトが画面をワイプし女性が消失、街の光景のみになる**（セグメント3中央部＝継ぎ目位置ではない・最重度）。
6. **【同】Chain B 486→487f: 看板の形が瞬間的に変わる**（人物は連続・背景の一部のみ）。
- backlog 4-6 の切り分け方針（ユーザー決定 2026-07-03）: **低解像度起因の可能性があるため、720p級での再検証結果と合流させて判断**（今すぐの単独調査はしない）。

## ✅ 高解像度チェーン実証（2026-07-03 夜・1280×768で初のチェーン実行）

- job `634f2da5`（`outputs/visual_review/09_hires_chain_2seg_1280x768.mp4`）: 2セグメント（73f+73f・2部構成トレイラー風プロンプト）・overlap 3/strength 0.5・総129f・264s・peak 9578MB・VRAM溢れなし。
- **ユーザー目視: 継ぎ目を発見できず**（「本当に連結したのか」と確認要請→metadata一次情報で `kind: chain`・num_clips 2・seam junction=72 を検証済み＝**本物の連結**）。512×320時代のChain A（継ぎ目は自然だが判別可能）より品質が上がり、**720p級＋トレイラー風では継ぎ目が実質不可視**という強い結果。
- backlog 4-6（背景歪み・ワイプ・看板変化）は本runの範囲（5.4s・2セグ）では**再発せず**。ただし尺・セグメント数がChain B（22s・4セグ）より小さいため、完全な切り分けには長尺での再現確認が必要（将来課題のまま）。

## 実運用で得た経験則（2026-07-14〜15・オーナーの長尺使い込み観察）

> **出自**: Clip Chain拡張（上限24クリップ）とチャンク化アップサンプルの実機ゲート後に、オーナーが8クリップ級の長尺生成を使い込んで得た観察記録である。もとは`LONGFORM_RESEARCH_TOPICS.md`（2026-07-15新設）§1として単独ノートに置いていたが、内容がクリップ連結の品質特性そのものであるため、2026-07-27の整理で本書へ吸収し、同ノートは削除した。将来の研究課題（単発生成へのチャンク化＋タイル化の移植／クリップ毎のキャラクター特徴注入）は[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §3-42・§3-43が正本。

- **チャンク化アップサンプル導入により、768p長尺チェーンのVRAM問題は解決済み。** 8クリップの実運用でピークVRAMは9397〜9529MBに収まり、総尺（クリップ数）に依存しない挙動を確認した。対して一括方式（チャンク化オフ）は同規模の生成で16377MB付近に張り付く（16GBカードの物理上限ぎりぎり）。詳細は[`CHUNKED_UPSAMPLE_WORKORDER.md`](CHUNKED_UPSAMPLE_WORKORDER.md)「GPU実機ゲート結果」節を参照。

- **「10秒おきに景色がぐにっと歪む」現象の正体は、クリップの連結点（クロスフェード連結、240フレーム＝10秒間隔）であり、アップサンプルのチャンク境界ではない。** チャンク化をオフにしたジョブでも同じ現象が発生し、メタデータに記録された連結点の間隔と歪みの発生間隔が一致することから特定した。見えやすさは題材に依存する。カメラが移動し続ける散歩シーンでは歪みが目立つが、静的な舞台（黒猫のオーケストラのような、カメラも被写体もあまり動かない構図）ではほぼ気づかないレベルだった。

- **キャラクター設計のドリフト（連結を重ねるごとに絵柄が少しずつ変質していく現象）を確認した。** 連結を重ねるほど直前のクリップとの差異が累積し、8クリップ級の長尺では最初と最後でかなり別の絵柄になる。画面外に一度消えたキャラクターが、次に画面へ戻ってきたときに違うデザインになっている現象も観察された。これは構造的な制約に起因する。クリップ連結時に前のクリップから持ち越されるのはオーバーラップ3latentフレーム（約0.5秒ぶんの潜在表現＝VAEで圧縮された内部データ）のみであり、それ以外の情報（キャラクターの見た目の全体像）は各クリップが新規に生成し直しているため、少しずつ「解釈」がずれていく。

- **経験則の結論（オーナー判断）。** 長尺を目指すなら、「1クリップをできるだけ長く（現在の上限は481フレーム＝20秒）し、連結数は4〜5個程度に抑える」のが実用上の最適解である。481フレーム×4クリップ（合計79秒）は実機ゲートで完走済み・ピークVRAM 9408MBを実証済み（[`CHUNKED_UPSAMPLE_WORKORDER.md`](CHUNKED_UPSAMPLE_WORKORDER.md)「GPU実機ゲート結果」節のゲートB）。連結数を増やすほどドリフトと連結点の歪みの両方が積み重なるため、同じ総尺を狙うなら「少数の長いクリップ」の方が「多数の短いクリップ」より画質面で有利という経験則になる。

- **1クリップの尺の上限について。** API上の481フレーム上限は、VRAM都合ではなく、モデルの学習上の尺天井（約20秒）に合わせた設計値である。`chain_math.py`のコメントに「trained 20s ceiling（学習時の20秒天井）」と明記されており（stage-2タイル設計＝22latentフレームのタイルが動画・音声のRoPE位置エンコーディング＝時間的な位置情報をこの天井の十分内側に収める設計の根拠として記載）、`api/models.py:84`のコメントにも同じ20秒キャップの由来が記されている。コミュニティの実測報告では、実用上のスイートスポットはさらに手前の10〜15秒とされ、高解像度になるほど早く品質が破綻する。自リポジトリの実測でも、704×1280（縦長）は12〜13秒あたりで崩壊が確認されている（[`LTX23_REFERENCE.md`](LTX23_REFERENCE.md)の1行目付近、該当記述はファイル12行目）。

**上記backlog #1・#4との関係**: 2026-07-03に512×320で記録した「タイル継ぎ目の少し後のドリフト」（backlog #1）と「Chain B 167-182fの背景歪み」（backlog #4）は、当時タイル継ぎ目由来と推定していた。2026-07-14〜15の観察で判明したのは**クリップ連結点（240フレーム間隔）由来の歪み**であり、両者は別地点の現象として切り分けが必要なまま残っている（backlog #4-6の720p級再検証は未実施＝[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §3-45）。

### キャラクター特徴注入研究の一次情報（台帳`PENDING_TASKS.md` §3-43の出典）

上のドリフト観察を受けて2026-07-15に行った調査の一次情報。研究課題そのものの正本は[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §3-43。

- https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients
- https://ltx.io/blog/how-to-use-ic-lora-in-ltx-2
- https://docs.ltx.io/open-source-model/usage-guides/ic-lo-ra
- https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients/discussions/3 （不具合報告）
- https://huggingface.co/LiconStudio/LTX-2.3-Multiple-Subject-Reference
- https://github.com/ID-LoRA/ID-LoRA-LTX2.3-ComfyUI
- https://github.com/ali-vilab/VACE
- https://github.com/Phantom-video/Phantom
- https://github.com/Tencent-Hunyuan/HunyuanCustom
- https://comfyui-wiki.com/en/news/2025-04-06-skyreels-a2-open-source
- https://ip-adapter.github.io/
- https://instantid.github.io/
- https://github.com/Mikubill/sd-webui-controlnet/discussions/1236
- https://fal.ai/models/fal-ai/ltx2-video-trainer
- https://ltx.io/model/ltx-trainer
- https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/looping_sampler.md

## リポジトリ状態

- **main へ merge・push 済**（merge `2cc4cac`）。元 branch `feature/phase3-clip-concat` は merge 済で役目を終えた。
- 本セッションの commit 列: `aebcd08`（engine masked AV-latent chaining）→`c0ed582`（api/services 配線）→`d359e4a`（stage2 単一コンテキスト修正）→`622dd81`（worker protocol ドキュメント）→`194ce44`（境界検証ハーネス用 video_io ヘルパー）→`eb5f7ac`（Gradio ルート `/` 404 修正）→ merge `2cc4cac`。
- 旧 `_EXTEND` monkeypatch 機構（`9fb7111`由来）は**削除済**（同日中に判断確定・`de587b4`「remove dead legacy latent-extend machinery」→ cleanup merge `1ab5e0c`。byte-match検証済＝退行なし）。
- 主要新規モジュール: `chain_math.py`（リポジトリ直下・app/engine両venvで import 可能な純Python junction/tileジオメトリの単一情報源）、`engine/pipeline/chain_pipeline.py`（`run_chain`＝spikeのs1+s2を本番移植）。
- worker: 新 `generate_chain` op（`engine/worker.py`＋`engine/api_types.py`）。app: `services/ltx_runner.py`（real+mock）／`services/pipeline_manager.py`（`run_chain_job`・単一呼び出しに再構成）／`api/models.py`（`overlap_frames`既定=K_v=3・`MAX_CHAIN_TOTAL_PIXEL_FRAMES=8×481`）。
- キャンセルはジョブ境界のみ（チェーン全体が単一の atomic worker op＝ユーザー承認済みの逸脱）。

## Pending 項目 → ✅全消化（2026-07-03 更新）

1. ~~Chain A/B 本番出力の最終目視/試聴~~ → **✅PASS**（本書「ユーザー最終目視/試聴 結果」節。継ぎ目・音声すべて自然・継ぎ目以外のアーティファクト3件はbacklog#4-6へ）。
2. ~~スライス1 キーフレーム目視~~ → **✅消化・ユーザー受容済み**（VERIFICATION_LOG §17.9-17.10）。
3. チューニング backlog（#1-6）＝v1受容済みの磨き候補のまま（非ブロッキング・唯一の将来課題）。
4. 次スコープ＝IC-LoRA Phase A→B は**完了済み**（`IC_LORA_PHASE_B_STATUS.md`）。現在の入口は [`NEXT_SESSION_HANDOFF.md`](NEXT_SESSION_HANDOFF.md) 冒頭ブロック。

## ★フロントエンド開発向けメモ: クリップ入力フレーム数と出力フレーム数は一致しない（2026-07-06 追記・ユーザー指示）

将来のフロントエンド（AviUtl2 統合等）でユーザー向け説明が必須になる算術。GUI 検証（VERIFICATION_LOG §27.7/§27.9）で実ユーザーが「クリップ2が生成されていない」と誤解した実績があるため、ここに正本として記録する。

※総尺上限の現行値は 24×481=11544ピクセルフレーム（api/models.py の MAX_CHAIN_TOTAL_PIXEL_FRAMES、2026-07-14に8×481から拡張）。本文中に登場する 8×481 は2026-07-03当時の歴史記録である。

**原則: クリップの `num_frames` は「生成時の計算単位」であり、出力動画の尺の単純合計にはならない。** 減算は2種類ある。

1. **オーバーラップ融合（全チェーン共通）**: 継ぎ目を滑らかにするため隣接クリップを `overlap_frames`（latent 単位・既定3）だけ重ねて融合する。デコード総フレーム数は `chain_math.compute_chain_layout` が単一情報源。例: 121f+121f・overlap 3 → **総225f**（242 にはならない）。
2. **V2V 継続の参照置換（V2V のみ）**: `source_video.context_frames`（例73）がクリップ1の頭を元動画の末尾で置換・凍結する。この部分は元動画のコピーであり新規生成ではないため、**配信される output.mp4 は「新規部分のみ」＝総フレーム − context_frames**。例: 225 − 73 = **152f ≈ 6.3秒**（クリップ1の新規分48f＋クリップ2の寄与104f）。除かれた元動画部分は `POST /jobs/{id}/join` の結合版（joined.mp4）で戻る。

**フロントエンドが説明・表示すべきこと**:
- 生成前: 予想出力尺 ≈（Σclip − overlap融合 − context_frames）÷ fps（Gradio GUI は `v2v_cap_panel` の説明文で対応済み）。
- 生成中: 進捗の「クリップ n/N」（`JobResponse.clip`/`clip_count`・optional 加算フィールド・§27.10）で「今どのクリップか・全クリップ処理されたか」を文字表示（クリップごとの VAE デコードプレビューは大半を捨てるデコードになるため不採用＝ユーザー決定）。
- 生成後: metadata の `chain.total_frames`・`chain.clip_num_frames`・`v2v.new_frames_px`・`segment_seam_junctions`（クリップ境界のフレーム位置。出力内の境界 ＝ junction − context_frames）で内訳を提示できる。
