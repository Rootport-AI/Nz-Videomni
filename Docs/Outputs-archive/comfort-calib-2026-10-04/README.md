# comfort-calib-2026-10-04（第 8 弾・§1-79）較正台

- 計測日は **2026-10-05**（84 本・失敗 0）。フォルダ名は準備した日のまま。10-04 に準備したが、自動許可の判定が `calib.py server start --pid` を拒否して計測 0 本で中断し（オーナー裁定で中止）、10-05 に再開した。
- 結果は `RESULTS.md`、点ごとの記録は `progress.md`、集計の出力は `make_results_output.md`・`report_output.txt`・`commit_table.md`・`digest.md`。

## 1. 複製元との差分
複製元は `outputs/comfort-calib-2026-10-03/`。持ち込んだのはコードだけ（`calib.py`・`mkplan.py`・`make_results.py`・`batch_table.py`・`vram_sampler2.ps1`・`drive2.sh`・`commit_table.py`・`digest.py`・`selftest/`・`mk.sh`・`rp.sh`・`cool.sh`・`wait.sh`・`sum.py`）。`commit_monitor.ps1` は監督が先に置いたものをそのまま使う。`original_state.json`・`luid.json`・`calib/`・`runs/`・`manifest.jsonl`・`plans/`・`server.pid`・ログは持ち込んでいない。

変えた箇所（全文は `harness_diff.txt`）:
- `mk.sh`: 変換器を位置引数に（`mk.sh <label> <LTX23|LTX25> <transformer> <allon|default> <W> <H> <frames|-> <基準点|FIRST> [クリップ…]`・`shift 8`）／連結に `--chunked-upsample`／`--tier P2`・`--purpose "P2 <label>"`／事前点検の終了コードを `preflight_<label>.rc` に書く。
- `rp.sh`: `preflight_<label>.rc` が 0 でなければ `ABORT` で exit 5。
- `calib.py` `check_point`: 連結の点が `chunked_upsample=true`・`stage2_window=standard` でなければ事前点検で失敗にする（送る側の既定は変えていない）。
- W5（補足・2026-10-05 夜）: 上の `check_point` のうち `stage2_window` の検査を `errors` から `warnings` に変えた（w46 を意図して測るため・オーナーの明示の指示で監督が変更）。`chunked_upsample` の検査は `errors` のまま。
- `drive2.sh`: 各点の前に監視の最終行が 10 秒より古ければ `ABORT: commit monitor stale` で exit 6。
- `cd` 先・`CSV`・定数 `D` を `10-04` に。
- `sum.py`: 要約に `chunked_upsample`・`stage2_window`・`stage2.seconds`・`wall.seconds`・`s23_gap_mid.seconds` を出す。

## 2. 自己試験
- `selftest/analyse_selftest.py`: 43 項目・失敗 0。
- `selftest/parser_selftest.py`: 節 A〜G は全部 ok。**節 H は飛ばした**（09-27 の計画ファイルに結びついた節で、今回の `plans/` に無い点を探して `KeyError` で止まる）。今回の点の算術は点ごとの `calib.py preflight` で検算する。

## 3. 打った順の手順
（W1 は `calib.py server start --pid 10244` が自動許可の判定に拒否されたところで中断。以下は再開後に追記）
- 2026-10-05（再開）: 監督が `server start --pid 32708`・`luid` を打った後、W1 が 3 点確認 → `load --base-model LTX23 --transformer default` → `idle-calib --minutes 3` → 点ごとに `bash mk.sh …`（`plans/` は手で `mkdir`）→ `plans/p_<label>.json` の `chunked_upsample` を目で確認 → `bash cool.sh` → `bash rp.sh <label> > rp_<label>.txt 2>&1`（裏）→ `bash wait.sh <label>` → `runs/<label>/server_slice.log` の `chain upsample started` と `wddm.csv` の共有の最大を確認 → `progress.md`。腕 qa は `load … Sulphur-2-base-distil-Q6_K` → `idle-calib` → 捨てラン → 基準点。

## 4. 落とし穴（W1 で気づいたもの）
- `plans/` は複製しないので、最初の `mk.sh` の前に作る（無いと `mkplan.py` が `FileNotFoundError`）。
- `wait.sh` は 570 秒で戻る。連結の点（約 11〜12 分）は 2 回呼ぶ。1 回のシェル呼び出しで 2 回続けて呼ぶと 10 分を超えるので、呼び出しを分ける。
- 基準点の異常の条件は監督裁定（2026-10-05）で「同じファイルの過去の基準点の stage-2 中央値と 300 MB 以上違う」に変わった（`idle-calib` の床＋300 MB は撤回）。基準点の共有メモリは重みのファイルごとに違う（Q4_K_M・`default` 約 660〜703 MB、Sulphur Q6_K 約 1,150〜1,190 MB、fp8mixed 約 2,160〜2,200 MB）。
- 単発の点（約 1〜3 分）は `mk.sh`→`cool.sh`→`rp.sh` を 1 回の呼び出しで前で走らせても 10 分に収まる。連結の点（約 10〜12 分）は `rp.sh` を裏で走らせ、`wait.sh` は 1 回の呼び出しに 1 回だけ。

### W2（腕 sb・sd・qb・2026-10-05）で追記
- 手順は W1 と同じ。sb は `load --base-model LTX25 --transformer default`（`state.json` の有効ベースモデルが LTX25 に変わる）→ `idle-calib` → 捨てラン → 基準点 → 梯子。sd は同じ変換器のまま捨てラン `sd_warm`（`mk.sh … default default …`）→ 基準点 → sb の c を 1 回。qb は `load … redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` → `idle-calib` → 捨てラン → 3 幾何の基準点と点（単発は `one.sh` で前で）→ 連結の基準点 → 梯子（連結は `rp.sh` を裏・`wait.sh` を 1 回の呼び出しに 1 回）。
- 落とし穴: 2.5 の連結の 1 点は 10〜16 分（溢れた点ほど stage-2 が長く、2048 幅は 950〜970 秒）。`wait.sh` は 2 回呼ぶ。既定構成（sdpa）の連結は全 on より約 1.3〜1.5 倍遅い。
- 落とし穴: 2.5 でも cu=true の連結は `server_slice.log` に `chain upsample started (3 units)` が 1 行出る（2.3 と同じ）。2.5 の連結の窓は `stage2`・`restore`・`p25_30_decode_encode` の 3 つで、`s23_*` は無い。
- 落とし穴: 単発の基準点（89 コマ）の stage-2 の標本は 1280×768 で 10 ちょうどになることがある（`qb_s_720_b89`）。「10 未満」には当たらないが際どい。
- 落とし穴: 2.5 Q6_K の基準点の stage-2 中央値は 1,157〜1,184 MB（過去の帯 1,140〜1,195 の中）。2.5 `default` は 651〜654 MB（公式 Q4_K_M の帯 660〜703 よりわずかに低いが差は 300 MB よりずっと小さい）。

### W3（腕 ra・rb・ua・2026-10-05）で追記
- 手順: ra は `load --base-model LTX25 --transformer redgraftLTX25Fast2K_ltx25RedgraftNSFW` → `idle-calib` → 捨てラン（`default`）→ 1080p の基準点と梯子 → 1280×768 の基準点と規則 5 の 3 点（単発は `one.sh` で前で）→ 連結の基準点 → 梯子（連結は `rp.sh` を裏・`wait.sh` を 1 回の呼び出しに 1 回）。rb は同じ変換器のまま捨てラン `rb_warm`（`allon`）→ 基準点 → ra の連結の c を 1 回。ua は `load … ltx25_uncensored_v1.1-fp8_scaled` → `idle-calib` → 捨てラン → 1080p の基準点 → c と c＋1 を 1 回ずつ。
- 落とし穴: `GET /status` はサーバーの直下ではなく `http://127.0.0.1:18620/api/v1/status`（直下は 404）。
- 落とし穴: 2.5 重い（REDGraft・fp8）の既定構成の連結は 1 点 13〜16 分（1792×1024 で 794〜796 秒・溢れた 1856 で 975 秒）。全 on は同じ点が約 10 分（615 秒）。`wait.sh` は 2 回呼ぶ。
- 落とし穴: 2.5 重いの基準点の stage-2 中央値は REDGraft 1,149.7〜1,177.0 MB・fp8 1,159.7 MB（過去の帯 1,140〜1,195 の中。単発 1280×768 の基準点の標本は 15）。

### W4（腕 ha・原状復帰・集計・2026-10-05）で追記
- 手順: 3 点確認 → `load --base-model LTX23 --transformer sulphur_distil_fp8mixed`（ベースモデルの切り替えで `state.json` の有効ベースモデルと LTX23 の選択が変わる＝想定どおり）→ `idle-calib --minutes 3` → 捨てラン `ha_warm` → 1080p の基準点と梯子（S＝129 が溢れたので下りて 121）→ 1280×768 の基準点と規則 5 の 3 点（265×2・289×1）→ 連結の基準点 → 梯子（1472×1024 から）→ `restore-state --base-model LTX25` → `restore-state --base-model LTX23` → `state.json` と `original_state.json` の突き合わせ → `POST /api/v1/pipeline/unload` → 3 点確認 → `make_results.py`・`calib.py report`・`commit_table.py`・`digest.py`。
- 落とし穴: fp8mixed は重みの常駐が大きく、基準点の stage-2 中央値は 2,181〜2,184 MB（Q4_K_M の約 3 倍）。基準点の異常の判定は「同じファイルの過去の基準点（2,163〜2,203 MB）」と比べる。
- 落とし穴: fp8mixed・全 on のコミットは計画の見込み（09-26 の GiB ÷ 上限）より約 4〜6 ポイント低かった（捨てラン 59.0%・基準点 68.0%・計測点 70〜71%）。09-26 より 6〜7.5 GiB 低い。
- 落とし穴: c が計画の例示と違ったとき、副の幾何（規則 5）と連結の出発点（規則 8）は c のトークンから計算し直す。連結の出発点は早見表に無い段（1472×1024）になることがある。
- 落とし穴: `restore-state` は担当が打っても自動許可の判定に拒否されなかった（`server start --pid` とは扱いが違った）。

## 5. 落とし穴のまとめ（次回の較正台へ）

前回（10-03）からの引き継ぎ:
- `original_state.json`・`luid.json`・`runs/`・`manifest.jsonl`・`plans/` を持ち込まない。`plans/` は最初の `mk.sh` の前に作る。
- `server stop` は打たない（停止は監督）。`.ps1`／`.bat` は担当が起動しない。
- 腕ごとに基準点を取る（ラベルの頭で腕を分ける・頭に `_` を入れない）。`idle-calib` は `load` の直後に毎回。捨てランは切り替えのあと。
- 冷却は `cool.sh`。`preflight` は 1 回に 1 ファイル。`--base-model`・`--transformer`・`--accel` を省略しない。
- 門は点の最中の最大で見る（`drive2.sh`）。連結は `rp.sh` を裏で走らせ、`wait.sh` は 1 回の呼び出しに 1 回（最大 570 秒・連結は 2 回呼ぶ）。
- 基準点と計測点のクリップの形を揃える（連結は `[361,361]` の 1280×768）。
- `unload` は認証なし。2.5 既定構成の vae の実体は `conv`。
- 切り替え後に `reanalyse` を打たない。残った待ち受けが `load` を打った事故があったので、担当の頭で 3 点確認をする。
- UTF-8（`PYTHONIOENCODING=utf-8`・`PYTHONUTF8=1`）。Windows のコンソールの既定（cp932）では表の記号で落ちる。

今回（10-05）新しく分かったもの:
- **`mk.sh` の変換器名の固定**（複製元は `--transformer default`・`--tier P1` を固定していた）→ 位置引数にした。直さないと `default` 以外の腕の点が `skipped_wrong_transformer` で飛ばされる。
- **cu=true の確認**は 3 か所で行う: `plans/p_<label>.json` の `chunked_upsample`（`preflight` が検査）・`runs/<label>/request.json`・`runs/<label>/server_slice.log` の `chain upsample started`（2.3・2.5 とも 1 行。分割が 1 つのときは「(N units)」が付かないので、この文字列だけで探す）。
- **gap_mid の標本**: cu=true の 2.3 の連結の基準点の `s23_gap_mid` は 12 標本（11.5〜12.0 秒）で、10 未満になる余地は小さいが際どい。計測点は 19〜29 標本。cu=true では `s23_gap_mid` の退避（第 7 弾の cu=false で出た 6〜7 GB）は出なかった。
- **LUID の取り直し**: 再起動で LUID が変わる。`calib.py luid` は監督が打った（`luid_0x00000000_0x0000f29f_phys_0`）。
- **長時間運転のコミットの漂い**: `idle-calib` 直後の待機コミットは 19.04（開始時）→ 19.51・20.00・20.11・20.51・20.77・20.50 GiB で、12 時間で +1.7 GiB 以内。漂いは無視できる。
- **引き継ぎの 3 点確認**: `GET /api/v1/status` の待ち行列／直前の `drive2_*.log` の最終行が `ALL DONE`／`Get-CimInstance Win32_Process -Filter "Name='bash.exe'"` のコマンドライン（確認コマンド自身も引っかかるので、`Get-CimInstance` を含む行は除いて読む）。
- **自動許可の判定**: `calib.py server start --pid` は拒否された（10-04）。拒否されたら担当は回り道をせず止めて報告し、監督が打つ。
- **基準点の異常の条件**は重みのファイルごとの常駐量の帯で見る（Q4_K_M・`default` 650〜703 MB、Q6_K 1,140〜1,195 MB、fp8mixed 2,160〜2,200 MB）。`idle-calib` の床＋300 MB は使わない。
- 単発の基準点（89 コマ）の stage-2 の標本は 1280×768・896×1152 で 10〜11 と際どい。
- 連結の 1 点の所要: 2.3 は 8〜12 分、2.5 全 on は 10〜16 分、2.5 既定構成は 13〜16 分。溢れた点ほど stage-2 が長い。
