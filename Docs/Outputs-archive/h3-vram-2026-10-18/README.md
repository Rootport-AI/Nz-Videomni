# MiniMax H3 の VRAM 溢れの犯人の計測（台帳 §1-101 の第 1 段）の結果と台本

作成: 2026-10-18。計測台 `outputs/h3-vram-2026-10-18/` は git の管理外なので、結果・台本・採取物をここに複写した。結論と読み方の要約は [`../../VERIFICATION_LOG.md`](../../VERIFICATION_LOG.md) §168。

## 読む順番

1. `RESULTS.md` が結果の正本（結論・腕ごとの表・ピークの瞬間の復元・予測との突き合わせ・T6・説明できない残り・設計へ持ち込む事実）。`summary.json` が機械可読の要約。
2. 腕ごとの生データは `runs/<腕の名前>/`。腕の名前は `T0_B_sdpa_b8_nocap_hist` のように「画角・attention・常駐数・上限の有無・履歴の有無」を並べたもの。
3. 計測台の条件と再実行の手順は `RIG_README.md`（計測台の README を改名して複写。`RESULTS.md` の中で「`README.md`」と呼んでいるものはこれ）。
4. 計画・棚卸し・レビューは `reports/`。

## 複写したもの

- 直下: `RESULTS.md`・`summary.json`・`probe_result.json`（T6 の結果）・`dryrun.json`・`RIG_README.md`、台本 4 本（`measure_h3_vram.py`・`snapshot_report.py`・`sdpa_int32_probe.py`・`vram_sampler2.ps1`）。
- `runs/<腕>/`（T0〜T5b・試運転 `dryrun`）: `run.json`・`memory_stats_step1.json`・`memory_stats_step2.json`・`snapshot_step1_report.md`／`.json`（ある腕だけ）・`wddm.csv`・`dmon.txt`・`gpu_process_memory_*.csv`・`luid.json`・`stdout.log`。
- `runs/T6_probe_*.log` と `runs/probe_result_before_isolated.json`（T6 を別プロセスに分ける前の結果の写し）。
- `reports/`: `PLAN_measure.md`（計画）・`REVIEW_plan_measure.md`（計画のレビュー）・`R1_vram_inventory.md`（VRAM の棚卸し）・`R2_kitchen_sdpa.md`（comfy-kitchen と SDPA の調査）・`EXT_proposal_2026-10-18.md`（外部案）・`REVIEW_results.md`（結果のレビュー）。中身は変えていない。

## 複写していないもの

- メモリの確保と解放の履歴の生データ（`snapshot_step1.pickle`）。数十 MB あり、読み取った結果は `snapshot_step1_report.md`／`.json` に残してある。
- 出力動画 `out.mp4`。

## 追記（2026-10-18 夜）: 対照の腕 F0

- `runs/F0_I2V_896x1152_sage_b8_nocap_keep_det/`: 製品の条件（I2V 896×1152×124・keep_resident・deterministic）を揃えた対照の腕。pickle と mp4 は省いた。結果は `RESULTS.md` の「計測台の検証（対照の腕 F0）」の節と [`../../VERIFICATION_LOG.md`](../../VERIFICATION_LOG.md) §168.9。
- `runs/measure_h3_vram_before_F0.py.txt`: F0 の引数を足す前の台本の写し。直下の `measure_h3_vram.py` は F0 を足した最新版。
- オーナーが操作パネルから行った 2 回目の実行（job `c14379b2`）は、操作パネルのログのみで採取物はない。
