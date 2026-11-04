> **このファイルは実機の `outputs/comfort-calib-h3-2026-11-04/progress.md`（git追跡外）のスナップショットである（2026-11-04複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# progress（台帳 §1-274 再較正・2026-11-04）

## 工程 P（10:13〜10:15）
- 作業場所を作成し、11-03 から台本・selftest/・plans/p_warm.json・materials/（5 点。sha256 は原本と一致）・README.md を、11-01 から keyframe_f0.png（sha256 2c9ec216…e698a。期待と一致）を写した。
- 固定パスの書き換え: drive2.sh:9・12、mk.sh:8、rp.sh:2、wait.sh:3、cool.sh:3、commit_table.py:10、digest.py:5。`grep -rn comfort-calib-h3-ref-2026-11-03 *.sh *.py` は該当 0 行（rc 1）。
- plans/p_warm_fl.json を mk.sh で生成（K=1・frame_idx [0]・keyframe_f0.png・default）。preflight は想定どおり rc 1（「K=0 だけ」の誤り）。predict の text_estimate 30 はキーフレームの行を含まない（差分の後に作り直しが要る）。
- D-前の計画 4 本を mk.sh で生成（b1080r・b720r・v720_345_15s・i1080_243_p9）。preflight rc 0（警告はステップ数と seed のみ）。predict: v720_345_15s S 217,167・i1080_243_p9 S 184,512／text 18,462（計画 §5.3 と一致）。p_warm は preflight rc 0。
- 自己試験: parser 281 項目・失敗 0（11-03 の plans/ 19 本を一時的に plans/ へ写して実行し、終わってから消した。selftest は作業場所の plans/ に 11-03 の 17 点があることを前提にしているため）／analyse 43・失敗 0／dryrun_validate 5 本（warm と D-前の 4 本）・失敗 0。parser・analyse の出力は 11-03 と時刻を除いて同一。

## 工程 P0（10:15〜）
- git_state_start.txt: HEAD e85d52f・porcelain 0。
- /status: running・unloaded・キュー空。サーバーの python 8948（親 powershell 2304）／10504（親 8948）、08:28:50 開始。
- **待機時の GPU 使用量 871〜872MiB。前回の値として指示された 1,051MiB と 180MiB 違うため、ここで止めて監督へ報告**（参考: 11-03 の開始時 932MiB、計画 §6.1 の 09:36 実測 875MiB）。server start・commit_monitor・luid・load は未実行。
- 10:17 監督の判断: 待機時 GPU 871〜872MiB。前回 932MiB（11-03 git_state_start.txt）との差 61MiB。計画の 1,051MiB は誤記。P0 の 2 から再開。
- 10:17 server start --pid 2304（rc 0。original_state.json の MiniMaxH3 は pinkcherryMinimaxH3_v18Hybrid_int8）。commit_monitor.ps1 起動（PowerShell PID 2520。commit_log.csv 10:17:40 から 3 秒おき・主メモリ 20.78／138.78GiB）。luid luid_0x00000000_0x0000f2b3_phys_0（882MB）。
- 10:17:50 git_state_before_load_ref2va.txt（e85d52f・0）。10:17:55 load Ref2VA rc 0（8.1 秒。応答と /models の active 一致・transformer_role ref2va）。
- 10:18〜10:21 idle-calib 3 分 rc 0: dedicated 中央値 1,142.3MB（最大 1,145.4）・shared 中央値 169.2MB・182 標本。読み込み後の nvidia-smi 1,128MiB。参考: 11-03 は 933.7／77.0MB、過去の idle は dedicated 574〜1,091MB・shared 54〜103MB（今回の dedicated・shared はどちらも過去の範囲の上を少し超える。判定の基準は基準点の窓なので計測は続行。cap で確かめる）。

| ラベル | 形 | 判定 | cap_mib | 帯（13,999〜14,033）| S（predict／used） | 1 ステップ秒 | 所要秒 | used_check_h3 |
|---|---|---|---|---|---|---|---|---|
| warm | 640×384×56 | smoke（完走・10:21:38〜10:22:39） | 14,025 | 内（11-03 の warm と同じ） | 4,296／4,296 | 1.775 | 56 | 真 |
| b1080r | 1920×1088×56 | baseline（完走・10:25:09〜10:27:26。sequence と predict の差 0。denoise 中央値 1,219.0MB） | 14,001 | 内（11-03 の b1080 と同じ） | 34,896／34,896 | 17.14 | 132 | 真 |
| b720r | 1280×768×56 | baseline（完走・10:29:57〜10:31:14。差 0。denoise 中央値 1,179.8MB） | 14,001 | 内 | 16,536／16,536 | 6.363 | 72 | 真 |
| v720_345_15s | 1280×768×345＋15 秒動画 | **comfortable**（完走・10:33:46〜11:03:08。修正前は OOM）。窓 denoise Δ-38.2（n=1,336）・decode Δ-13.6・encode Δ+16.0。vc 97,920・vcr 200,736・text 15,281（predict と差 0）。reference_encode 203.5 秒・山 4,399MB（予約 5,702）・TE 88.5 秒・山 8,079MB。照合（references）真 | 13,999 | 内 | 217,167／217,167 | 335.73 | 1,756 | 真 |
  - v720_345_15s の事実: アロケータの再試行 num_alloc_retries 8（11-03 の快適な点は 0〜2）・inactive_split の山 3,787MiB・num_ooms 0。判定は変えない（記録のみ）。
| i1080_243_p9 | 1920×1088×243＋画像×9 | **comfortable**（完走・11:07:02〜11:29:11。修正前は OOM）。窓 denoise Δ-49.4（n=995）・decode Δ-9.6・encode Δ+1,165.4（判定外の窓）。vc 146,880・vcr 165,240・text 18,462（差 0）。照合真 | 13,999 | 内 | 184,512／184,512 | 249.98 | 1,324 | 真 |
  - i1080_243_p9 の事実: TE の段の予約 16,074MB（11-03 の i1080_226_p9 は 15,248MB）・山 9,978MB・150.9 秒（同 124.0 秒）。encode の窓の共有メモリ中央値 2,385.1MB（基準 1,219.7、Δ+1,165.4。同 i1080_226_p9 は Δ+80.6）。encode の窓は判定に使わない窓なので判定は変えない。num_alloc_retries 8・inactive_split 山 4,955MiB・num_ooms 0。denoise の予約 13,514MB。
- 11:07 前の点の後に打った確認コマンドに誤って `python -` を混ぜて 3 分止まった（担当の誤り。計測には影響なし。止めた時点で i1080_243_p9 は未開始で、cool.sh の後に改めて開始）。
- 11:29 弾 D-前の 5 点を打ち終えた。/status: pipeline_loaded true・state ready・Ref2VA 読み込み済み・キュー空。弾 A には進まず停止。

# 計測台の差し替えと弾 A（FL2VA・K=1）
- 11:31 calib.py と selftest/parser_selftest.py を ../comfort-calib-h3-2026-11-04-patch から上書き（元の calib.py は calib_before_k.py）。diff で patch 側と一致（差 0）。p_warm_fl.json とその preflight を消して mk.sh で作り直し: preflight rc 0（警告はステップ 4・seed 固定のみ）・predict text 278・S 4,784。parser_selftest（run_with_plans.py で PLANS_DIR を 11-03 の plans へ）288 項目・失敗 0（selftest/parser_selftest_output_k_2026-11-04.txt）。
- 11:32:08 git_state_before_load_fl2va.txt（e85d52f・0）。11:32 load default rc 0（3.9 秒。/models の active default＝models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors・transformer_role fl2va）。idle-calib 3 分 rc 0: dedicated 中央値 1,099.1MB・shared 153.3MB。弾 A の計画 5 本を mk.sh で生成、preflight 全 rc 0（s1080_277 predict text 2,078・S 172,322、s720_481 S 139,882）。

| ラベル | 形 | 判定 | 開始〜終了 | S（predict／実測）| vc | text | 1 ステップ秒 | 所要秒 | cap | 帯 | retries | inactive_split | used_ok | images_used |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| warm_fl | 640×384×56・K=1 | smoke（完走） | 11:36:00〜11:36:55 | 4,784／4,784 | 4,320 | 278 | 1.783 | 50 | 14,023 | 内 | 0 | 92 | 真 | 1 |
| b1080 | 1920×1088×56・K=1 | baseline | 11:39:26〜11:41:58 | 38,984／38,984 | 36,720 | 2,078 | 19.90 | 146 | 13,999 | 内 | 0 | 616 | 真 | 1 |
| s1080_277（門） | 1920×1088×277・K=1 | **comfortable**（修正前は OOM）。窓 denoise Δ-21.6（n=881）・decode Δ+1.9。denoise の予約 13,376MB・割り当ての山 10,670MB | 11:44:30〜12:02:37 | 172,322／172,322 | 169,320 | 2,078 | 221.16 | 1,082 | 13,999 | 内 | 4 | 2,958 | 真 | 1 |
| b720 | 1280×768×56・K=1 | baseline | 12:05:08〜12:06:29 | 18,464／18,464 | 17,280 | 998 | 7.34 | 76 | 13,999 | 内 | 0 | 512 | 真 | 1 |
| s720_481 | 1280×768×481・K=1 | **comfortable**（落ちていない確認）。窓 denoise Δ+2.0（n=610）・decode Δ-0.4。denoise 予約 11,352MB・山 9,080MB | 12:09:01〜12:22:01 | 139,882／139,882 | 137,280 | 998 | 152.95 | 775 | 13,999 | 内 | 4 | 2,450 | 真 | 1 |
| b1440 | 2560×1472×56・K=1 | baseline。復号の段の予約 14,800MB（上限の掛からない段。計画 §5.3 の注記どおり） | 12:24:34〜12:29:26 | 70,144／70,144 | 66,240 | 3,718 | 48.41 | 287 | 13,999 | 内 | 0 | 1,233 | 真 | 1 |
- 12:29 弾 A の 6 点を打ち終えた（全点 used 真・images_used 1・行数の差 0・cap 帯の内）。弾 B へ。

# 弾 B（1920×1088・K=1 の 2 分探索。初手 362）

| ラベル | 形 | 判定 | OOM の段・場所・文言 | 開始〜終了 | S（predict／実測） | vc | text | 1 ステップ秒 | 所要秒 | cap | retries | inactive_split | used_ok | images_used |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1080_362 | 1920×1088×362・K=1 | **OOM** | denoise（線に数える）・`*.py:143 block_forward`（traceback は途中が切られている。`o = torch.empty((s, inner), …)`）・Tried to allocate 2.99 GiB・3.89 GiB free・13.67 GiB allowed・allocated 6.94 GiB・reserved but unallocated 3.84 GiB。1/4 ステップ目（denoise 窓 354 秒） | 12:31:57〜12:38:39 | 223,604／223,604（worker_log） | 220,320 | 2,078 | —（log 354.4） | 397 | — | — | — | — | — |
| （復帰）warm_fl_rec1104124216 | 640×384×56・K=1 | 煙の点（完走）。drive2 の recover（load default → idle 1 分 → p_warm_fl）12:41:10〜12:43:07 | — | 12:42〜12:43 | — | — | — | — | — | 帯の内（gate 合格） | — | — | 真 | 1 |
| s1080_328 | 1920×1088×328・K=1 | **comfortable**。窓 denoise・decode とも退避なし。denoise 予約 12,964MB・山 12,169MB | — | 12:45:37〜13:09:20 | 203,092／203,092 | 199,920 | 2,078 | 297.68 | 1,419 | 13,999 | 9 | 4,566 | 真 | 1 |
| s1080_345 | 1920×1088×345・K=1 | **comfortable**。denoise 予約 13,486MB・山 12,669MB | — | 13:11:52〜13:37:41 | 213,348／213,348 | 210,120 | 2,078 | 326.25 | 1,543 | 13,999 | 9 | 4,778 | 真 | 1 |
| s1080_345_r2 | 1920×1088×345・K=1 | **comfortable**（2 回目）。denoise 予約 13,486MB・山 12,669MB | — | 13:40:14〜14:06:02 | 213,348／213,348 | 210,120 | 2,078 | 326.17 | 1,543 | 13,999 | 9 | 4,778 | 真 | 1 |
- 14:06 弾 B の葉: **c=345（S 213,348・text 2,078・eff 214,179）、U=362（S 223,604・text 2,078・eff 224,435。OOM 1 回・denoise）**。木は s1080_362 OOM → 328 快適 → 345 快適 → 345_r2 快適。弾 C は始めない。読み込み済みは FL2VA（default）・ready・キュー空。

# 弾 C（2560×1472・K=1）
- 14:08 mk.sh s1440_192 → preflight rc 0（警告 2 つ）。predict S 217,798・text 3,718（video 209,760・cond 3,680・audio 640）。監督の値 S 221,486・text 7,406 と 3,688 行違う（キーフレーム 1 枚分の 3,680＋8。監督の検算は 2 枚分）。指示の「差 10 行以内」を超えるので、点は始めずに停止して報告。b1440 の実測 text は 3,718（較正台の値と一致）。
- 14:10 監督の了承（text 3,718 が正しい）。弾 C の木を台本 treeC.sh（scratchpad。gateC.py は退避・降格を 11 で返し同じ点をもう 1 本）で開始。初手 s1440_192（eff 219,285・見積もり 1 ステップ約 337 秒）。

| ラベル | 形 | 判定 | OOM の段・場所・文言 | 開始〜終了 | S（predict／実測） | vc | text | 1 ステップ秒 | 所要秒 | cap | retries | inactive_split | 復号の予約／山 | used_ok | images_used |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1440_192 | 2560×1472×192・K=1 | **comfortable**。窓 denoise Δ-20.8・decode Δ-13.1。denoise 予約 13,764MB・山 12,905MB | — | 14:10:25〜14:36:58 | 217,798／217,798 | 213,440 | 3,718 | 338.34 | 1,587 | 13,999 | 9 | 4,905 | 14,858／10,874 | 真 | 1 |
| s1440_209 | 2560×1472×209・K=1 | **OOM** | denoise（線に数える）・最初のステップの頭（壁時計 40 秒・denoise 窓の標本 0）・`pipeline_h3.py:1580 _denoise → dit.py:653 forward → …`（traceback は切られており最後のフレームは読めない）・Tried to allocate 462.00 MiB・1.22 GiB free・13.67 GiB allowed・allocated 12.80 GiB・reserved but unallocated 552.25 MiB（断片化ではなく割り当て自体が上限に届いた形） | 14:39:30〜14:40:15 | 236,254／236,254（worker_log） | 231,840 | 3,718 | — | 40 | — | — | — | — | — | — |
| （復帰）warm_fl_rec1104144354 | 640×384×56・K=1 | 煙の点（完走・gate 合格）。recover 14:42:47〜14:44:47 | — | — | — | — | — | — | — | 帯の内 | — | — | — | 真 | 1 |
| s1440_192_r2 | 2560×1472×192・K=1 | **comfortable**（2 回目）。窓 denoise Δ-35.8・decode Δ-31.7。denoise 予約 13,764MB・山 12,905MB | — | 14:47:17〜15:13:48 | 217,798／217,798 | 213,440 | 3,718 | 338.39 | 1,585 | 14,001 | 9 | 4,905 | 14,858／10,874 | 真 | 1 |
- 15:13 弾 C の葉: **c=192（S 217,798・text 3,718・eff 219,285）、U=209（S 236,254・text 3,718・eff 237,741。OOM 1 回・denoise の最初のステップの頭）**。

# 切替（FL2VA → Ref2VA）と弾 D-後
- 15:15:06 git_state_before_load_ref2va_2.txt（e85d52f・0）。15:15 load Ref2VA rc 0（応答と /models の active 一致・transformer_role ref2va）。
- 計画: warm2（p_warm と label・purpose 以外同一）・v720_379_15s・r1080_362 を mk.sh で作成、preflight 全 rc 0。
- (a) v720_{上}_15s の F の選び方（較正台の predict。eff＝S＋0.40×text。規則: eff ≧ 224,435＋4,800＝229,235 の最小の F）:

| F | S | text | vc | ref_video | eff |
|---|---|---|---|---|---|
| 362 | 222,023 | 15,281 | 102,720 | 102,816 | 228,135 |
| 379 | 226,881 | 15,281 | 107,520 | 102,816 | 232,993 |
| 396 | 231,737 | 15,281 | 112,320 | 102,816 | 237,849 |

  → F=379（v720_379_15s）。
- (b) v720_{下}_15s は省く: v720_345_15s（eff 223,279）が OOM の最小 224,435（s1080_362）のすぐ下で快適だったので、参照の行の重さの下側の裏付けはこの点で足りている（監督の指示）。
- (c) r1080_362（参考）: predict S 219,516・text 30・eff 219,528（video 218,280・audio 1,206。監督の初めの値 221,556 は K=1 の vc を使った誤りで、監督が訂正済み）。役割は「Ref2VA K=0 の 1080 は、FL2VA K=1 が OOM した 362 の段で快適か（1 段上か）」の確認で、帯の位置の絞り込みではない。
- 順: idle-calib 3 分 → warm2 → (a) → (c)。台本 treeD.sh（scratchpad。Ref2VA の rp.sh は環境変数なし・recover は既定の Ref2VA と p_warm）。

| ラベル | 形 | 判定 | OOM の段・場所・文言 | 開始〜終了 | S（predict／実測） | vc | vcr | text | 1 ステップ秒 | 所要秒 | cap | retries | inactive_split | denoise 予約／山 | decode 予約／山 | 窓 Δ（denoise／decode） | used_ok | images_used |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| warm2 | 640×384×56・K=0 | smoke（完走）。idle-calib 3 分（15:15:44〜）: dedicated 中央値 1,090.3MB・shared 165.9MB | — | 15:18:48〜15:19:40 | 4,296／4,296 | 4,080 | 4,080 | 30 | 1.662 | 46 | 14,025 | 0 | 107 | 3,788／3,776 | 5,370／5,061 | — | 真 | 0 |
| v720_379_15s | 1280×768×379＋15 秒動画・K=0 | **comfortable**（eff 232,993。OOM の最小 224,435 より 8,558 上で快適 → 計画 §5.3 の読み方では「参照はまだ軽い」形 (b)）。reference_encode 203.5 秒・予約 5,702／山 4,399。TE 88.7 秒・予約 9,442／山 8,079 | — | 15:22:12〜15:53:36 | 226,881／226,881 | 107,520 | 210,336 | 15,281 | 363.42 | 1,878 | 13,999 | 8 | 3,940 | 13,910／13,462 | 7,564／6,398 | −35.0／−6.5 | 真 | 0 |
| r1080_362（参考） | 1920×1088×362・K=0 | **comfortable**（FL2VA K=1 の 362 は OOM。Ref2VA K=0 は同じ段で快適＝1 段上）。eff 219,528 | — | 15:56:08〜16:23:06 | 219,516／219,516 | 218,280 | 218,280 | 30 | 341.96 | 1,613 | 13,999 | 9 | 5,014 | 13,888／12,949 | 10,564／8,276 | −74.5／−14.1 | 真 | 0 |
- 16:23 弾 D-後を打ち終えた。形は計画 §8 の (b)（参照の行は映像の行より軽い）。
- **留保（オーナー向け）: 参照つきの上限は 379 で快適まで確認、OOM の段は未測**（v720_413_15s など計画にない点は足していない）。

# 切替（Ref2VA → FL2VA）と弾 E（s1080_345 の 20 ステップ）
- 16:24:26 git_state_before_load_fl2va_2.txt（e85d52f・0）。load default rc 0（応答と /models の active 一致・role fl2va）。warm_fl2（p_warm_fl と label・purpose 以外同一）・s1080_345_20（predict S 213,348・text 2,078＝4 ステップの s1080_345 と一致）を mk.sh で作成、preflight rc 0。台本 treeE.sh（idle 3 分 → warm_fl2 → 17:25 までに開始できれば s1080_345_20）。
| warm_fl2 | 640×384×56・K=1 | smoke（完走・16:27:50〜16:28:44）。idle-calib: dedicated 1,088.2MB・shared 168.3MB | cap 14,023・retries 0・inactive_split 92・used 真・images_used 1 |
- 16:31:17 s1080_345_20 開始（17:25 の停止線の内）。
| s1080_345_20 | 1920×1088×345・K=1・**20 ステップ** | **comfortable**。窓 denoise Δ-24.5（n=6,488）・decode Δ-10.0。各ステップの秒: 1 ステップ目 325.8・2〜20 ステップ目 325.0〜325.9（同じ並び）。denoise 予約 13,488MB（4 ステップの 13,486 と同水準）・山 12,669MB（同 12,669）。復号 予約 10,548／山 8,271 | 16:31:17〜18:23:46 | S 213,348／213,348・vc 210,120・text 2,078 | 1 ステップ 325.44 秒・所要 6,743 秒 | cap 13,999・retries 41・inactive_split 4,780・used 真・images_used 1 |
- 18:24 弾 E 終了。終了処理へ。
- 18:24 restore-state rc 0（Hybrid 版 pinkcherryMinimaxH3_v18Hybrid_int8 を load → state.json 確認）→ POST /pipeline/unload → /status unloaded・キュー空（completed 23・failed 2）。state.json の selections は original_state.json と全一致（MiniMaxH3 も一致）。git status --porcelain 0 行。commit_monitor（PID 2520）停止。自分の台本・待ちループの残りなし（監視用の tail と待ちループ 3 本を止めた）。
- 運転に使った台本（scratchpad の run_fl.sh・run_r.sh・gate*.py・est.py・extras.py・tree{B,C,D,E}.sh・run_with_plans.py）の写しを drivers_scratch/ に置いた（記録用）。
