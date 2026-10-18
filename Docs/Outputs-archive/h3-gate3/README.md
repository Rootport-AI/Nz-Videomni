# MiniMax H3 第 3 弾 開発機の門 3-G1・3-G2 の台本と結果

作成: 2026-10-18／インストーラ担当 C1。`outputs/h3-gate3/` は git の管理外なので、台本と結果をここに複写した（mp4 と生のログの一部は複写していない）。計画は `Docs/Outputs-archive/h3-stage3-reports-2026-10-18/PLAN_stage3.md` §5。

## 3-G1（構築と冪等）

- 前提: 第 1 弾の `.venv-engine-h3` はリポジトリ外 `_venv-engine-h3.stage1/` へ退避済み。
- 実行: リポジトリ直下で `install-MiniMaxH3.bat`（出力は `g1_run1_console.log`・`g1_run2_console.log`）。
- 1 回目（338 秒）: `tools\uv`（0.11.32）で `.venv-engine-h3` を新規構築。段 (a) は直接ピン 3 本（torch 2.13.0+cu130・torchvision 0.28.0+cu130・SageAttention の cu130 wheel）、段 (b) は freeze の残り（cu130 の index）。重み 4 本と tokenizer は SKIP、LICENSE（17,604 バイト）だけ取得。検証の表は 11 行すべて PASS（H3 の行は python・worker・重み 4・tokenizer・LICENSE）。`INSTALLED_PATHS.txt` に H3 の 6 行。完了の印あり。
- freeze の比較（`compare_freeze.py`。`name==version` の集合、SageAttention は wheel の名前から版を取る）: 第 1 弾の freeze（`engine_h3/venv-engine-h3.freeze.txt`）と 42 件すべて一致（`g1_freeze_compare.json`）。freeze は更新していない。退避した第 1 弾の環境そのものの freeze とも一致（`stage1_vs_new_compare.json`。この JSON の `identical: false` は、取得時に uv の案内行「Using Python … environment at: …」1 行が混入したためで、パッケージの差は 0）。
- 2 回目（1 秒）: 仮想環境・取得 3 項目すべて SKIP、検証の表は全部 PASS（冪等）。

```
python outputs\h3-gate3\compare_freeze.py engine_h3\venv-engine-h3.freeze.txt outputs\h3-gate3\g1_new_freeze.txt outputs\h3-gate3\g1_freeze_compare.json
```

## 3-G2（新しい環境でのテストと指紋）

- テスト: `.venv-engine-h3\Scripts\python.exe scripts\run_h3_pytest.py tests\test_h3_engine.py tests\test_h3_media.py -rs --junitxml ...` → 86 passed・2 skipped（GPU テスト）。
- セルフテスト 1 本（GPU）: プロンプトを渡さない既定のまま実行した結果、指紋は `9161e030…`（基準 `0da6111c…` と不一致）。`quant_backends.cuda.int8_linear` は真、`ready.sage_available` は真。
- 不一致の原因（GPU を使わずに確かめた）: 基準の指紋はセルフテストの既定のプロンプトではなく、「G5 のプロンプト」（`Docs/Outputs-archive/h3-gate2/g4_i2v_resident.py` の `PROMPT`。黄色いレインコートの女性）で取ったもの。既定のプロンプトは 32 トークン、G5 のプロンプトは 41 トークンで、基準の記録（`text_tokens` 41）と今回（32）の差と合う。トークン化は新旧どちらの環境でも同じ 32 個の ID になる（`tok_probe.py`・`tok_probe_new.json`・`tok_probe_stage1.json`）。
- **取り直し（監督の判断で GPU 1 本・3-G2 合格）**: G5 のプロンプトを `--prompt` で渡し、ほかは既定のまま（台本 `g2_selftest_g5prompt.cmd`、結果 `g2_t2v_g5prompt.json`）。指紋 `0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9`（854,334 バイト）で基準と一致。`text_tokens` 41・143.6 秒・VRAM 予約ピーク 7,590MB・コミット量の山 34.53GiB。`quant_backends.cuda.int8_linear` 真・`ready.sage_available` 真。

## 注意（記録 §167 への申し送り）

**セルフテストの既定のプロンプトは基準と違う（既定 32 トークン対 G5 のプロンプト 41 トークン）。** 指紋 `0da6111c…` を取り直すときは、必ず G5 のプロンプトを `--prompt` で渡す（`g2_selftest_g5prompt.cmd`）。既定のプロンプトのままだと指紋は `9161e030…` になり、これは不一致ではなく条件の違い。計画 §5 の 3-G2 とレビュー m-6 の「セルフテストの既定」という書き方はこの点で誤り。
