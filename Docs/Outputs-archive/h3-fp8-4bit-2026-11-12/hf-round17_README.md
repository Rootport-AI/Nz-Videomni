# round17-h3-quant-product — MiniMax H3 の fp8・w4a8 の重みを製品で回した記録（2026-11-12）

置き場: `Rootport/Nz-LTX23-eval-clips` の `round17-h3-quant-product/`。

## 目的

製品に入れた MiniMax H3 の fp8（8 ビットの浮動小数点の重み）と w4a8（重みを 4 ビット・入力を 8 ビットにした形式）の読み込みを、利用者と同じ経路（MCP〔Model Context Protocol。外部の道具からアプリを操作する仕組み〕→ アプリのサーバー → エンジン）で回し、完走することと絵が破綻しないことを確かめました（台帳 §1-283）。

## 条件

共通: T2V（文章から動画）・1280×768・73 枚・24fps・種（seed）1234・注意の実装（attention）sage・VRAM 16GB mode on・先読み（prefetch）on。

プロンプト: A busy city street at dusk seen from the sidewalk. Pedestrians walk past the camera, their footsteps clicking on the pavement; cars and a bus drive by with engine hum and tire noise, and a car horn honks in the distance. Steady camera.

| 門 | 条件 | 重みのファイル | ステップ | shift（映像／音声） | LoRA | 読み込み秒 | 秒／ステップ | VRAM の山 割り当て／予約（MiB） | fp8_mode | 形式ごとの層の数 |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | `int8` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20 | 12／3 | なし | 16.1 | 8.760 | 3780／4536 | — | {"int8_tensorwise": 200} |
| G2 | `fp8` | `minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | 20 | 12／3 | なし | 3.2 | 11.725 | 3773／4684 | scaled | {"float8_e4m3fn": 200} |
| G3 | `w4a8_kijai` | `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | 20 | 12／3 | なし | 2.9 | 9.625 | 2810／3638 | — | {"asym_w4a8_int8": 200} |
| G4 | `w4a8_eros` | `10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors` | 8 | 10／3 | なし | 2.9 | 9.575 | 3780／4536 | — | {"asym_w4a8_int8": 180, "int8_tensorwise": 20} |
| G5 | `int8_oldmarker` | `h3ErosMax_beta5_3185154.safetensors` | 8 | 10／3 | なし | 2.9 | 8.762 | 3780／4536 | — | {"int8_tensorwise": 200} |
| G5b | `int8_oldmarker_s20` | `h3ErosMax_beta5_3185154.safetensors` | 20 | 12／3 | なし | 3.0 | 8.815 | 3780／4536 | — | {"int8_tensorwise": 200} |
| G6 | `fp8_turbo` | `minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | 4 | 12／3 | `minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16`（強さ 1.0） | 3.0 | 15.125 | 4072／5096 | scaled | {"float8_e4m3fn": 200} |

## ファイル

- `h3_int8_1280x768_20steps_seed1234.mp4`・`sheet_int8.png`（G1。並び画像はフレーム 0・36・72）
- `h3_fp8_1280x768_20steps_seed1234.mp4`・`sheet_fp8.png`（G2。並び画像はフレーム 0・36・72）
- `h3_w4a8_kijai_1280x768_20steps_seed1234.mp4`・`sheet_w4a8_kijai.png`（G3。並び画像はフレーム 0・36・72）
- `h3_w4a8_eros_1280x768_8steps_seed1234.mp4`・`sheet_w4a8_eros.png`（G4。並び画像はフレーム 0・36・72）
- `h3_int8_oldmarker_1280x768_8steps_seed1234.mp4`・`sheet_int8_oldmarker.png`（G5。並び画像はフレーム 0・36・72）
- `h3_int8_oldmarker_s20_1280x768_20steps_seed1234.mp4`・`sheet_int8_oldmarker_s20.png`（G5b。並び画像はフレーム 0・36・72）
- `h3_fp8_turbo_1280x768_4steps_seed1234.mp4`・`sheet_fp8_turbo.png`（G6。並び画像はフレーム 0・36・72）
- `sheet_all.png`: 全条件の並び画像（行が条件、列がフレーム 0・36・72）

## 所見（監督。視聴の判定はオーナー）

- 7 条件すべて完走し、自己検査はすべて合格（ジョブの完了・1280×768・73 枚・音声あり・使った重み・ステップと shift・読み込みの記録・形式ごとの層の数がヘッダーから数えた数と一致・G6 は LoRA の適用 208 組）。
- 読み込みの記録: fp8_mode は G2・G6 が `scaled`（RTX 40 系の経路。計算能力 8.9）、ほかは無し。fp8 の 200 層のうち 50 層（各ブロックの `mlp.fc2`）は印 `full_precision_matrix_mult` のとおり bf16 に戻して計算した。
- 速度（1 ステップの秒）: 製品の値はスパイク S3 の値と 0.03 秒以内で一致（int8 8.760 ↔ 8.785、fp8 11.725 ↔ 11.753、Kijai の w4a8 9.625 ↔ 9.652、h3ErosMax の w4a8 9.575 ↔ 9.608）。fp8 は int8 の 1.34 倍。G6 の 15.1 秒は 2 ステップ目以降の中央値も 15.1 秒なので立ち上がりでは説明できず、Turbo LoRA の費用（記録簿 §176.5 の実測で 1 ステップ +4.2 秒の階級）によるもの。fp8 の層に固有の増分かどうかは切り分けていない。
- VRAM（ノイズ除去の山）: スパイクと同じ（int8 3780／4536、fp8 3773／4684、Kijai 2810／3638、h3ErosMax 3780／4536）。G6 は LoRA の分だけ高い 4072／5096。
- 並び画像の監督の目視: G1〜G4・G6 は破綻なし。G1（int8）と G2（fp8）は同じ基の重みの別の量子化なので、同じ種で構図がほぼ同じ。G3 と G4 は別の量子化・別のファインチューンで構図が違う。
- G5（h3ErosMax beta5 3185154。旧式の印。8 ステップ・shift 10／3）: 読み込めて完走した（この門の目的）。絵は夜の街で、車のライト・バス・建物の構造は保たれているが、プロンプトの「夕暮れ」より暗く、人物はほとんど黒い影。このファイルは名前に TURBO の印がなく（対の 3178732 は `TURBO_Hybrid`）、Turbo でない版と見られる（推測）のに、G4 と同じ 8 ステップで回していた。
- G5b（同じファイルを公式と同じ 20 ステップ・shift 12／3 で回し直し。オーナーの了承 2026-11-12）: 完走・自己検査合格・速度と VRAM は G1 と同じ。絵は G5 よりはっきり明るく、歩道の白線・マンホール・赤白のバス・車のライト・店の窓が読める。破綻なし。ただし色味は青い薄暮から宵の口で、G1 の金色の夕焼けにはならない。ステップ数の不足が濁りの原因の一部で、光の色はこのファインチューンがプロンプトを読んだ結果と見ている（推測）。
- 読み捨てたテンソル: G4・G5・G5b の h3ErosMax の 2 本は `adaln_basis`・`adaln_mean` を未知として捨てる（スパイクと同じ。研究ノート §13 に既知。ComfyUI も使わない）。
- block swap の先読みの記録（ワーカーのログ）: 1 パスあたりの「pinned の空き待ち」は int8 系で約 1.7 秒、fp8 で約 5.0 秒、w4a8 で約 0.1 秒、G6 で 0.01 秒。秒／ステップはスパイクと一致しているので計算と重なって隠れている待ちと見る（推測）。新しい症状ではなく、製品の既存の先読みの振る舞い。


