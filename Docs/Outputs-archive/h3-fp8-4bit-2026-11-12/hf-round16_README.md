# round16-h3-quant — MiniMax H3 の fp8・w4a8 の試作の検証（2026-11-12）

台帳 §1-283 の第 0 段です。ComfyUI と同じ comfy-kitchen の関数を同じ引数で呼ぶ試作の層を、製品のコードを変えずに台本の中だけで差し込み、同じプロンプト・同じ種で 5 本を生成しました。基準は公式の int8 です。

見ていただきたいこと: 映像と音声が破綻していないか（別のモデル・別の計算経路なので、絵が基準と違うこと自体は想定どおりです）。

プロンプト: A busy city street at dusk seen from the sidewalk. Pedestrians walk past the camera, their footsteps clicking on the pavement; cars and a bus drive by with engine hum and tire noise, and a car horn honks in the distance. Steady camera.

## 条件と数値

| 条件 | 重みのファイル | ステップ | 読み込み秒 | 全体の秒 | 1 ステップの秒 | VRAM の山（割り当て・予約 MiB） | 自己検査 |
|---|---|---|---|---|---|---|---|
| int8 | minimax_h3_fl2va_pruned_int8_convrot.safetensors | 20 | 0.18 | 244.68 | 8.785 | 6332・7484 | True |
| fp8 | minimax_h3_fl2va_pruned_fp8_scaled.safetensors | 20 | 0.14 | 288.98 | 11.753 | 6332・7484 | True |
| w4a8_kijai | minimax_h3_fl2va_pruned_w4a8_mixed.safetensors | 20 | 0.13 | 241.75 | 9.652 | 6332・7484 | True |
| w4a8_eros | 10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors | 8 | 0.13 | 121.91 | 9.608 | 6332・7484 | True |
| fp8_dequant | minimax_h3_fl2va_pruned_fp8_scaled.safetensors | 20 | 0.13 | 359.55 | 15.468 | 6332・7484 | True |

共通: 1280×768・73 フレーム・種 1234・注意の実装 sage・VRAM 16GB モード on・先読み on・LoRA なし。
fp8_dequant は fp8 の全層を bf16 へ戻して計算する経路（RTX 30 系で使う経路）を RTX 40 系で強制したものです。

## ファイル

- `h3_int8_1280x768_20steps_seed1234.mp4`・`sheet_int8.png`（int8。並び画像はフレーム 0・36・72）
- `h3_fp8_1280x768_20steps_seed1234.mp4`・`sheet_fp8.png`（fp8。並び画像はフレーム 0・36・72）
- `h3_w4a8_kijai_1280x768_20steps_seed1234.mp4`・`sheet_w4a8_kijai.png`（w4a8_kijai。並び画像はフレーム 0・36・72）
- `h3_w4a8_eros_1280x768_8steps_seed1234.mp4`・`sheet_w4a8_eros.png`（w4a8_eros。並び画像はフレーム 0・36・72）
- `h3_fp8_dequant_1280x768_20steps_seed1234.mp4`・`sheet_fp8_dequant.png`（fp8_dequant。並び画像はフレーム 0・36・72）
- `sheet_all.png`: 5 本の並び画像（行が条件）
- `summary.md`: 数値の表

## 所見（監督。視聴の判定はオーナー）

- 数値の一致（S1）: 試作の層の出力は、ComfyUI が呼ぶ comfy-kitchen の経路とビット単位で一致した。fp8 の行列積（RTX 40 系の経路。飽和する入力を含む 81 例）、fp8 を bf16 に戻す経路（fc2 と RTX 30 系の経路。90 例）、w4a8（組の大きさ 16・8・4 と符号帳を含む 54 例）。VRAM 16GB モードの行の切り出しも、fp8・w4a8 ともビット単位で一致。w4a8 のカーネルが遅い経路（eager）に落ちた回数は 0。
- 速度（S3・1 ステップ）: int8 8.8 秒、fp8 11.8 秒（1.34 倍）、w4a8 9.6 秒（1.10 倍）、fp8 の全層 bf16 復元 15.5 秒（1.76 倍。RTX 30 系で使う経路の見込み）。fp8 の遅さは ComfyUI と同じカーネル・同じ設定（fp32 累算）に由来し、ComfyUI で fp8 を使うときと同じ階級。
- VRAM（ノイズ除去の段の山）: fp8 は int8 と同程度、Kijai の w4a8 は約 1GB 低い、h3ErosMax の w4a8 は int8 と同じ（block swap の環が int8 のブロックの大きさで作られるため）。5 本とも全体の山は復号の段で同じ。
- 並び画像の監督の目視: 5 本とも破綻なし。fp8 と fp8_dequant は同じ重みなので絵がほぼ同じ。int8 と fp8、Kijai の w4a8 は別の量子化なので構図の細部が違う。h3ErosMax は別のファインチューン。
- オーナーの視聴の判定（2026-11-12）: 5 本とも破綻・映像の乱れ・不自然さは感じなかった → 合格。
