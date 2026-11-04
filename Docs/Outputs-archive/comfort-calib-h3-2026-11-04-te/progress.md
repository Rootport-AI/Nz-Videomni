> **このファイルは実機の `outputs/comfort-calib-h3-2026-11-04-te/progress.md`（git追跡外）のスナップショットである（2026-11-05複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# progress（台帳 §1-275 再較正 (B)・2026-11-04 夜〜）

位置づけ: TE の注意の変更（第 2 版。塊の境目は変えず、最も高い塊を最初に計算）で快適上限の境界が動いていないことの確認（計画 §6）。

## S5 切替（23:27〜23:33）
- サーバー: run.ps1 の powershell 42140 → python 1160 → python 56952（18620 で待ち受け）。どれも 23:27:29 起動＝fix_v2 の適用（22:53 ごろ）の後。AviUtl2 60632（23:27:36 起動）。
- /status: unloaded・キュー空。`calib.py server start --pid 42140` rc 0。original_state.json の MiniMaxH3 は pinkcherryMinimaxH3_v18Hybrid_int8（Hybrid 版）。
- `calib.py luid` rc 0（chosen luid_0x00000000_0x0000f2b3_phys_0・dedicated 1,006.6MB。11-04 は 882.3）。commit_monitor（PID 59112）→ commit_log.csv 更新を確認。
- git_state_recalib_start.txt: HEAD 7669e01・porcelain 2 行（text_encoder.py・test_h3_engine.py）・text_encoder.py hash-object 75274125…・fix_v2.patch sha256 711ba433…。待機時の GPU 1,006MiB（未読み込み・AviUtl2 と操作パネルあり）。

## FL2VA の弾
- 23:29:48 load default rc 0（4.3 秒。/models transformer_role fl2va・active default）。
- idle-calib 3 分 rc 0: dedicated 中央値 1,275.3MB（最大 1,280.9）・shared 中央値 217.3MB（最大 288.8）。11-04 は 1,099.1／153.3MB で、待機が約 176MB 高い（判定は同日の基準点の窓との差で行うので続行）。

| ラベル | 形 | 判定 | 開始〜終了 | S（predict／実測） | text | cap | 帯 | 窓 denoise／decode の差 | retries | inactive_split | used_ok | 備考 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| warm_fl | 640×384×56・K=1 | smoke（完走） | 23:33:33〜23:34:26 | 4,784／4,784 | 278 | 14,023 | 内（11-04 も 14,023） | — | 0 | 92 | 真 | encode 19.15 秒・山 2,648 |
| b1080 | 1920×1088×56・K=1 | baseline | 23:36:58〜23:39:33 | 38,984／38,984 | 2,078 | 13,999 | 内 | 基準 denoise 中央値 1,260.4MB（11-04 は 1,182 前後） | 0 | 616 | 真 | 1 ステップ 20.5 秒 |
| s1080_345 | 1920×1088×345・K=1 | **comfortable**（11-04 と同じ） | 23:42:05〜00:08:27 | 213,348／213,348 | 2,078 | 13,999 | 内 | denoise Δ-84.8（n=1,331）・decode Δ-65.6 | 9 | 4,778 | 真 | denoise 予約 13,486・山 12,669（11-04 と同値）・decode 予約 10,548・山 8,271・1 ステップ 333.8 秒・commit 47.5% |
| s1080_362 | 1920×1088×362・K=1 | **OOM**（11-04 と同じ。denoise・block_forward・Tried to allocate 2.99 GiB・3.89 GiB free・13.67 GiB allowed・allocated 6.94・reserved but unallocated 3.84。数値も 11-04 と同一） | 00:10:57〜00:17:47 | 223,604／223,604（worker_log） | 2,078 | —（OOM の点は記録なし） | — | denoise Δ-87.9（n=362） | — | — | — | log 1 ステップ 362.9 秒 |
| （復帰）warm_fl_rec1105002124 | 640×384×56・K=1 | smoke（drive2 の recover: load default → idle 1 分 → p_warm_fl） | 00:20:18〜00:22:19 | 4,784／4,784 | 278 | 14,023 | 内 | — | — | — | 真 | |
| b1440 | 2560×1472×56・K=1 | baseline | 00:24:50〜00:29:40 | 70,144／70,144 | 3,718 | 14,001 | 内 | 基準 denoise 中央値 1,201.9MB | 0 | 1,233 | 真 | decode 予約 14,800・山 10,768（11-04 と同じ）・1 ステップ 48.4 秒 |
| s1440_192 | 2560×1472×192・K=1 | **comfortable**（11-04 と同じ） | 00:32:13〜00:58:41 | 217,798／217,798 | 3,718 | 14,001 | 内 | denoise Δ-33.6（n=1,348）・decode Δ+61.4 | 9 | 4,905 | 真 | denoise 予約 13,764・山 12,905・decode 予約 14,858・山 10,874・1 ステップ 338.1 秒 |
| s1440_209 | 2560×1472×209・K=1 | **OOM**（11-04 と同じ。denoise・module.py:1778・Tried to allocate 462.00 MiB・1.23 GiB free（11-04 は 1.22）・13.67 GiB allowed・allocated 12.80・reserved but unallocated 552.25 MiB） | 01:01:12〜01:01:57 | 236,254／236,254（worker_log） | 3,718 | — | — | — | — | — | — | |

- 01:02 FL2VA の弾を打ち終えた。判定は 11-04 と全点同じ（345・192 快適、362・209 は denoise の OOM）。cap は 13,999〜14,023 で帯の内。行数の差 0・used 真（完走した点）。commit の山 47.6%。
  - 事実: s1440_192 の decode の窓の差 +61.4MB（11-04 は −13.1／−31.7）。判定に使う窓で退避の判定は出ていない（comfortable）。待機の dedicated が 11-04 より約 176MB 高い（idle-calib）。

## Ref2VA の弾
- 01:02 git_state_before_load_ref2va.txt（7669e01・porcelain 2 行・hash-object 75274125…）。01:04 load minimax_h3_ref2va_pruned_int8_convrot rc 0（7.5 秒・active 一致）。
- idle-calib 3 分 rc 0: "dedicated_median_mb":1196.7,"shared_median_mb":148.2,（11-04 は dedicated 1,142.3／1,090.3・shared 169.2／165.9）。

| ラベル | 形 | 判定 | 開始〜終了 | S（predict／実測） | text | cap | 帯 | 窓の差 | retries | inactive_split | used_ok | 備考 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| warm | 640×384×56・K=0 | smoke（完走） | 01:08:05〜01:09:02 | 4,296／4,296 | 30 | 14,025 | 内（11-04 も 14,025） | — | 0 | 107 | 真 | |
| b720r | 1280×768×56・K=0 | baseline | 01:11:33〜01:12:52 | 16,536／16,536 | 30 | 14,001 | 内 | 基準 denoise 中央値 1,176.9MB | 0 | 308 | 真 | 1 ステップ 6.36 秒 |
| b1080r | 1920×1088×56・K=0 | baseline | 01:15:22〜01:17:39 | 34,896／34,896 | 30 | 14,001 | 内 | 基準 denoise 中央値 1,192.0MB | 0 | 619 | 真 | 1 ステップ 17.15 秒 |
| v720_379_15s | 1280×768×379＋15 秒動画 | **comfortable**（11-04 と同じ） | 01:20:11〜01:51:31 | 226,881／226,881 | 15,281 | 13,999 | 内 | denoise Δ-31.4（n=1,445）・decode Δ-11.9・encode Δ-10.2 | 8 | 3,940 | 真 | denoise 予約 13,910・山 13,462・TE 段 予約 9,442・山 8,079・88.5 秒・reference_encode 203.1 秒・照合真 |
| i1080_243_p9 | 1920×1088×243＋画像×9 | **comfortable**（11-04 と同じ） | 01:54:02〜02:15:43 | 184,512／184,512 | 18,462 | 13,999 | 内 | denoise Δ-34.4（n=994）・decode Δ-7.8・**encode Δ-0.1（n=140。11-04 は +1,165.4）** | 8 | 4,955 | 真 | **TE 段 予約 12,070（11-04 は 16,074）・山 9,698（9,978）・123.8 秒（150.9）**・denoise 予約 13,514・山 11,425（11-04 と同値）・decode 予約 10,514・山 8,226 |

- 02:16 Ref2VA の弾を打ち終えた。判定は 11-04 と同じ（v720_379_15s・i1080_243_p9 とも快適）。cap 13,999〜14,025 で帯の内。行数の差 0・used 真。i1080_243_p9 の encode 窓の差 −0.1MB（+300MB 未満）・TE 段の予約 12,070MiB（16,074 より低い）。
- 事実: v720_379_15s の TE 段 予約 9,442・山 8,079 は 11-04 と同値（N=15,281 は 512×29＋433 で末尾の塊が 433 行。11-04 も旧規則で同じ形のため変化なし。推測）。

## 復元（02:16〜）
- calib.py restore-state rc 0（pinkcherryMinimaxH3_v18Hybrid_int8 を load → state.json 確認）。state.json の active_base_model と selections は original_state.json と全一致。
- POST /pipeline/unload → /status unloaded・キュー空（completed 11・failed 2＝OOM 2 点）。
- commit_monitor（PID 59112）停止。台本・サンプラ・待ちループの残りなし。主メモリのコミットの山 47.6%（s1080_362）。
- git: porcelain 2 行（fix_v2 適用のまま。コミットは監督）。
