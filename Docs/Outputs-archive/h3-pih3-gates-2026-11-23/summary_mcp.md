# pi-H3 段 4: MCP の道具の実機の確認（記録）

台本 `outputs/h3-pih3-gates-2026-11-23/mcp_check.py`。オーナーのサーバー（127.0.0.1:18620）をそのまま使用。本体の重みは Ref2VA 版（`minimax_h3_ref2va_pruned_int8_convrot.safetensors`）。

| 項目 | 結果 |
|---|---|
| 1 list_models | H3 の constraints.pi_h3 = {"4steps": {"num_inference_steps": 4, "shift_video": 12.0, "shift_audio": 3.0}, "2steps": {"num_inference_steps": 2, "shift_video": 12.0, "shift_audio": 3.0}}。LTX23・LTX25 の unsupported_features に pi_h3 あり |
| 2 単発 4steps | MCP job `eeb973ec-e460-4e8a-9ced-38576c191a6f` 完了。pi_h3_used 4steps・steps/shift [4, 12.0, 3.0]・DWD {'chunks': 1, 'seconds': 3.2}。API 直接 job `720957cd-b1d4-4d5e-930d-e10b4cb754d6` と映像・音声のストリームの MD5 が一致（True）。サーバーが記録した要求の差は無し |
| 3 食い違うステップ数 | 422。文: `Error executing tool submit_generate: VALIDATION_ERROR: Request validation failed — [{'loc': ['body'], 'msg': 'Value error, num_inference_steps=20 differs from the value pi_h3="4steps" uses (num_inference_steps=4)', 'type': 'value_error'}]`。ジョブ数 11 → 11 |
| 4 連結 2 クリップ 4steps | MCP job `212351bc-4bce-4011-8b4a-96fd52b29edc` 完了・226 フレーム・音声あり。pi_h3_used 4steps・steps/shift [4, 12.0, 3.0] |
| 5 off の経路 | 道具が送る本文を、送らずに取り出して確認。off の本文に pi_h3 の鍵なし（True）。4steps の本文にステップ数と shift の鍵なし |
