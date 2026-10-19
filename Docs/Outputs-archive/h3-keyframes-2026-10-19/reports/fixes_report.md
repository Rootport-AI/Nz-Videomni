# 小修正 m1〜m6 の報告
m1 system.py: list_models の説明に keyframes（LTX {8,1}／H3 {1,0}）を追記。
m2 h3/adapter.py: REJECT_TABLE の説明を「strength の既定は None」に直した。
m3 h3/adapter.py: 冒頭の規則一覧にキーフレームの格子を追記。
m4 keyframeUtils.ts: snapFrameIdx に下限（first = offset>0 ? offset : multiple）を追加。docコメント更新。test に (2,0) 格子（1→2 ほか）を追加。(8,1)(1,0)(4,2) の期待値は不変。
m5 worker.py: _selftest_generate で _selftest_images の ValueError を捕まえ、stderr に出して return 2。tests/test_h3_engine.py の test_worker_selftest_keyframe_option に確認を追加（capsys 引数追加）。
m6 pipeline_h3.py: H3LoadOptions.deterministic の説明を「0 と F-1 のキーフレームが画像読み取り部へ渡るときだけ vision tower を読む。中間だけは T2V と同じ」に変更。
## git diff --stat（今回触った 7 ファイル。未コミットの先行差分を含む合計）
keyframeUtils.test.ts 46 / keyframeUtils.ts 28 / pipeline_h3.py 179 / worker.py 50 / system.py 4 / adapter.py 47 / test_h3_engine.py 287
## テスト
- アプリ venv pytest tests: 3449 passed / 65 skipped（変更前と同数。追加テストはエンジン venv 側で動く）
- エンジン venv test_h3_engine.py: 104 passed / 11 skipped（収集 115 件。依頼文の 165 は実際の総数と合わず、失敗は 0。変更前の数は未確認）
- webui: typecheck エラー 0／vitest 166 ファイル 3318 件合格／oxlint（変更 2 ファイル）指摘なし
## できなかったこと
なし。コミットも GPU 使用もしていない。
