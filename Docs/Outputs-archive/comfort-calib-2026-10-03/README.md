# 快適上限の計測台（第 7 弾 P1・2026-10-03）

結果は `RESULTS.md`、経過は `progress.md`。この文書は道具の差分・手順・落とし穴だけを書く。

## 1. 複製元との差分（`harness_diff.txt` に `diff -u` の全文）

複製元は `outputs/comfort-calib-2026-09-27/`。持ち込んだのはコードだけ（`calib.py`・`mkplan.py`・`make_results.py`・`batch_table.py`・`vram_sampler2.ps1`・`commit_monitor.ps1`・`drive2.sh`・`commit_table.py`・`digest.py`・`selftest/`）。`original_state.json`・`luid.json`・`calib/`・`runs/`・`manifest.jsonl`・`plans/` は持ち込んでいない。

- `calib.py`: `cmd_server_start` の、ポートが閉じているときに `run.ps1` を起動する枝を削り、`SystemExit("server is not listening; start it yourself")` に変えた（この較正台はサーバーを起動しない）。冒頭 docstring の 1 行目に今回の折の名を足した。他は 1 バイトも変えていない。
- `drive2.sh`: 絶対パス 2 行（cd 先と `commit_log.csv`）を 10-03 に。行列の確認の `python -c` を `.venv` の python（`"$PY" -c`）に。
- `commit_table.py`・`digest.py`: 定数 `D` を 10-03 に。
- `mkplan.py`・`make_results.py`・`batch_table.py`・`vram_sampler2.ps1`・`commit_monitor.ps1`・`selftest/` は変えていない。
- **`selftest/parser_selftest.py` の節 H は今回は飛ばした。** 節 H は前回（09-27）の計画ファイルに結びついた節で、前回の点（`f_s_1080_121_d` など）が無いので `KeyError` で止まる。節 A〜G は止まる前まで全部 ok。`selftest/analyse_selftest.py` は 43 項目・失敗 0。今回の点の算術は、点ごとに `calib.py preflight --plan` で検算した（全点 0 failed）。

今回足した小道具（較正台の外側。GPU には触らない）:
- `mk.sh`: `mkplan.py` で 1 点 1 ファイルを作り、すぐ `preflight --plan` を走らせる。
- `rp.sh`: 1 点を `drive2.sh` で走らせ、`sum.py` で `run.json` の要約（判定・窓ごとの delta・標本・used）を出す。
- `cool.sh`: 直前の点の終わり（`drive2_*.log` の `done` 行）から 150 秒たつまで待つ。
- `wait.sh`: 裏で走らせた点の要約が出るまで（最大 570 秒）待つ。連結の点は 10 分を超えるので裏で走らせた。
- `sum.py`: `run.json` の要約。

## 2. 手順（実際に打った順）

作業ディレクトリはこのフォルダ。python は `..\..\.venv\Scripts\python.exe`、環境変数は `PYTHONIOENCODING=utf-8`・`PYTHONUTF8=1`。

```
# 監視（別プロセス。PowerShell の Start-Process で起こした）
powershell -NoProfile -ExecutionPolicy Bypass -File commit_monitor.ps1 -Out commit_log.csv -IntervalSec 2

python calib.py server start --pid 27848     # original_state.json を作る
python calib.py luid
python calib.py load --base-model LTX23 --transformer default
python calib.py idle-calib --minutes 3
bash mk.sh a1_warm LTX23 allon 768 512 121 FIRST      # 計画を作って preflight
bash rp.sh a1_warm                                    # 1 点を走らせて要約
bash cool.sh                                          # 冷却 150 秒（別の呼び出し）
...（A1 → A2 → [load LTX25 default・idle-calib] → B1 → B2 → B3）
python calib.py restore-state --base-model LTX25      # LTX25 の選択を戻す
python calib.py restore-state --base-model LTX23      # 有効ベースモデルを最後に
curl -s -X POST http://127.0.0.1:18620/api/v1/pipeline/unload
taskkill /T /F /PID <commit_monitor の PID>
python make_results.py > make_results_output.md ; python commit_table.py > commit_table.md ; python digest.py > digest.md
```

`mk.sh` の引数: `<label> <LTX23|LTX25> <allon|default> <W> <H> <frames または -> <基準点のラベル または FIRST> [連結ならクリップのコマ数…]`。

## 3. 落とし穴

### 前回までから引き継ぐもの（09-26 README 第 5 節・09-27 README 第 4 節）

1. `original_state.json` を前の折から持ち込まない（復元が前の折の選択へ戻る）。今回は新しく作った。
2. `server start --pid` は監督が起動したサーバーにだけ付ける。`server stop` は打たない。
3. 変換器・構成を替えた点には自前の基準点を持たせる。基準点の参照は `manifest.jsonl` の `baseline_label` の最後のレコードで、変換器を照合しない（今回は `a1_`・`a2_`・`b1_`・`b2_`・`b3_` で分けた）。
4. `idle-calib` はモデルを読み込んでから。`calib/calibration.json` が無いと `run` は即座に終わる。
5. 計画ファイルの境目には冷却が入らない。次の点の前に 150 秒空ける（今回は `cool.sh` を別の呼び出しで）。
6. `preflight` は 1 回に 1 ファイル。`preflight.json` は最後の `preflight` で上書きされる（点ごとの記録は `preflight_<label>_output.txt`）。
7. `mkplan.py` では `--base-model`・`--transformer`・`--accel` を省略しない（既定は LTX25・default 構成）。
8. コミットの関門は「点の最中の最大」で見る（`drive2.sh` が `commit_log.csv` から拾う）。95% で次を投入しない。`DELETE /jobs/{id}` は実行中の推論を止めない。
9. Git Bash に `pkill` は無い。裏の道具は `taskkill /T /PID` で子ごと止める。
10. 出力は UTF-8 で（cp932 だと全角で止まる）。
11. 捨てランは変換器の切り替えのあとと、構成の切り替え（全 on ↔ 既定）のあとに置く。

### 今回わかったこと

12. **この機体のコミット上限はいま 110.67 GiB**（前回の記録 113.82 GiB より約 3 GiB 小さい）。P2（重い変換器）を測るときは 95% の線が 105.1 GiB まで下がっている前提で考える。
13. **LTX 2.3 の連結（`[361,361]`）の溢れは stage-2 の窓ではなく、その手前の s23_gap_mid 窓に出る。** 較正台の v3′ は判定可能な窓のどれかが +300 MB で溢れとするので拾えるが、stage-2 の窓だけを見ていると見逃す。連結の点は 1 本 10〜11 分かかる（2.5 の 2112×1088 は 15 分台）ので、1 回 10 分の制限がある実行環境では裏で走らせて待つ。
14. 連結の点は、基準点も計測点も同じクリップ形（今回は `[361,361]`）に揃えた。前回 09-26 の連結は `[25,161]` で、クリップの長さが違うと s23_gap_mid の重さが変わりうる。
15. 実行環境のシェルでは `sleep 150` が拒まれた。冷却は「直前の点の終わりから 150 秒たつまで待つ」ループ（`cool.sh`）で行った。
16. `config.yaml` の `server.api_key` は空で、`pipeline/unload` は認証なしで通る。
17. LTX 2.5 の既定構成の実体は used で `sdpa`・prefetch on・keep_resident off・fused on・vae `conv`・埋め込み常駐 off。
18. **LTX 2.3 の連結は `s23_gap_mid` の退避で判定が割れやすい。** 今回の 1792×1024 以上・`[361,361]` では 6 本すべてで 6〜7 GB の退避が起きていたが、v3′ は窓の中央値で見るので、退避が窓の半分を超えたかどうかだけで溢れ／快適が分かれた（`RESULTS.md` 第 2 節 A2 の追記）。連結の判定では、v3′ の結果だけでなく `wddm.csv` の共有の最大と閾値を超えた標本の数も見ること。また、COMFORT_LIMIT_TABLE 第 4.2 節の規則 v3（stage-2 と復元の窓だけ）と較正台の v3′（2.3 では `s23_*` の 3 窓も入る）は同じではない（`RESULTS.md` 第 1.1 節）。
19. **コミット上限が下がっている**（113.82 → 110.67 GiB）。95% の停止線は 105.1 GiB、90% の警戒線は 99.6 GiB。今回の最大は 2.3 連結の 93.22 GiB（84.2%）で、2.3 の重い変換器（P2）ではこれより厳しくなる前提で計画する（12 項と同じ事実の再掲）。
