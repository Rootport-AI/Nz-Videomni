# Sol-H3-VAE を製品に入れる段 1 の、実機の確認（GPU）の台本

計画 6.2 の 10 項目を回すための台本です。製品のファイルは変えません。オーナーの `config.yaml`・`state.json`・`scripts/manifests/` は読むだけで、バックエンドは `_rt/` の一時の設定と専用のポート 18699 で起動し、項目が終わると止めます。各項目の前に、オーナーのサーバー（ポート 18620）の待ち受け・ほかの python／uvicorn の GPU の使用・残っているエンジンのワーカーを確かめ、引っかかったら何も実行せずに終了します（ほかのプロセスには触りません）。GPU を使う項目は、その都度オーナーの了承を得てから回します。

## ファイル

| ファイル | 役割 |
|---|---|
| `h3_sol_gates.py` | 本体（アプリ側の仮想環境。torch を読み込まない）。バックエンドの起動と停止、関門、記録の書き出し、利用者のフォルダーを含むパスの消去は、pi-H3 の確認の台本 `outputs/h3-pih3-gates-2026-11-23/h3_pih3_gates.py` を読み込んで、このフォルダーに向け直して使います |
| `engine_items.py` | エンジンを直接呼ぶ項目（S1・S6・S9・S10。エンジンの仮想環境で、本体から起動されます） |
| `wddm_sampler.ps1` | 共有 GPU メモリの採取（実験 `h3-sol-spike-2026-11-23` と同じもの） |

## 実行のしかた

リポジトリの最上位（`Nz-Videomni`）で、Git Bash から:

```
export PYTHONUTF8=1
G=outputs/h3-sol-gates-2026-11-23/h3_sol_gates.py
.venv/Scripts/python.exe -X utf8 $G --dry-run           # GPU を使わない確認
.venv/Scripts/python.exe -X utf8 $G --gate-check        # 関門だけ
.venv/Scripts/python.exe -X utf8 $G --live --only S1    # 1 項目（済んだ項目は飛ばす。やり直しは --redo）
.venv/Scripts/python.exe -X utf8 $G --live              # S1〜S10 を順に（途中で止まっても、もう一度で続きから）
.venv/Scripts/python.exe -X utf8 $G --post-only         # summary.json・summary.md を作り直す
```

項目ごとの合否と数値は `runs/<項目>/result.json`、エンジンを直接呼んだ項目の記録は `runs/<項目>/engine.json` にあります。
