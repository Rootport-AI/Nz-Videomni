全件採用（監督の査定 2026-10-18。観点＝過剰設計・スパゲッティ化の危険／調査で裏付けが取れるか。採用の結果は PLAN_vram16.md に反映済み）

# 敵対的レビュー: MiniMax H3「VRAM 16GB mode」計画（playful-juggling-reef.md・確定前の版）

## 結論

**Critical 1・Major 8・minor 13。**

骨格はおおむね妥当で、コードで裏付けが取れた。確認できた点: `BlockSwapService.blocks_on_gpu` を install の前に代入すれば反映される／受け皿は install のたびに作り直される／`_is_oom` が「out of memory」を含む文言を GPU_OOM（503）に分類しワーカーを unload する／操作パネルの `resets` はサーバーの既定（ACCELERATION_DEFAULTS）へ書き戻す／`int8_linear` は活性化を行ごとに量子化し、m が 1・2 のときだけ別のカーネルを使う。

問題は 3 系統: (1) on の経路が off とビット一致するかを製品の形で端から端まで確かめる門が無い (2) 既存の仕組みに無い前提（真偽値の既定の書き戻し・LTX の 422 の文言）(3) 過剰設計（既存の関数を行の塊ごとに呼べば済むところを `lowvram.py` で書き直している）。

## 指摘

- **C1【Critical】門の穴**: A-G2 は S=512 の合成入力でブロックごとに比べるだけで、S ≤ 4,096 では FFN・AdaLN・最終層の塊化が一度も実行されない。A-G3 は off の指紋だけ、A-G4 は on の VRAM と時間だけ。「on の出力が off と一致する」門が無い（裁定 46 の中心が担保されない）。→ A-G3b（`--vram16` のセルフテストで指紋 `0da6111c…` と一致）を足し、A-G2 は製品の形で 1 ブロックずつ比べる。
- **M1【Major】ビット一致の根拠の穴**: (a) sage は呼び出しごとに `km = k.mean(dim=seq)` を計算する（`sageattention/core.py:724-725`）。ヘッド群に分けると集約の出力数が変わり足し算の順序がずれうる（推測）。(b) 「Ampere 以上は常に CUTLASS」は Python 側から言い切れない（`_C.cutlass_int8_dequant` が偽を返すと cuBLAS へ落ちる。`comfy_kitchen/backends/cuda/__init__.py:2245-2287`）。量子化の経路が m で変わるのは m==1 のときだけ（418-430・2169）という主張は裏付けが取れた。→ 製品の形と幅で門を行う。`k.mean` の切り出し比較の合成テストを 1 本足す。
- **M2【Major】書き戻しの仕組みが無い**: アダプタに `validate_for_family` は無く、書き戻すのは共通の `RequestRules.fill_defaults` だけで、表（`sampling`）は数値専用・`float(default)` に変換し `constraints.sampling` に公開される（`services/engines/__init__.py:215-238`・`ltx/request_rules.py:60-69, 118-125`）。→ `RequestRules` に真偽値の既定の小さな表 `flag_defaults` を足す。
- **M3【Major】LTX の 422 の文言**: 2.3／2.5 の `reject_unsupported` は機能名に関係なく「もう片方の LTX を選んでください」の雛形（`ltx/adapter.py:255-264`・`ltx25/adapter.py:494-503`）。表の 3 要素目は `Callable[[GenerateRequest], bool]` なので `is True` は書ける。→ 機能名で文言を分ける。
- **M4【Major】門の順序と A-G5 の判定規則**: A-G5 で定数が変わりうるのに A-G2／A-G4 がその前。レビューが GPU の後。A-G5 に選ぶ物差しが無い（事前登録の教訓）。→ A-G1→A-G6→A-G5（規則を先に書く）→A-G2・A-G3b・A-G4。
- **M5【Major】B-G3 の OOM の案内文**: 模擬バックエンドにも mockBridge にも OOM の注入は無い。→ pytest（503 `gpu_oom` の `detail` に案内文）と vitest（GPU_OOM の表示に `detail`）で代える。仕組みは足さない。
- **M6【Major】token_refiner の退避の主メモリ**: `move_module_tree` は CPU に新しい 1.5GB を確保する（`sft_load.py:121-150`）。コミット量が約 1.5GB 増える（推測）。→ 非常駐は捨てる、常駐は CPU へ。A-G4 でコミット量を F0 と比べる。
- **M7【Major】二重実装**: `modulate_chunked`・`gate_chunked`・`mlp_row_chunks` は `_modulate`／`_gate`／MLP の手順の書き写し（`dit.py:239-249, 276-296`）。→ 既存の関数を行の範囲ごとに呼ぶ。`VRAM16_FFN_CHUNK_ROWS` と `VRAM16_ROW_CHUNK` は 1 つに。
- **M8【Major】目標の大きさの門が無い**: 台帳 §1-101 の目標（1280×768 で 265〜313f・1920×1088 で 121〜145f・系列およそ 9 万行）で上限の下の OOM と返せない断片が足りるかは未検証（T5b で 4,283MiB）。→ A-G4 に 2 本足す。
- **m1** `sdpa_offset_guard` は on の経路では実行されない（群ごとに連続なら S≈209 万行まで安全）。削除。
- **m2** 端数の併合（3 行未満）は等分 `ceil(S/4096)` に置き換える（Turing の m≤128 の融合経路 `cuda/__init__.py:316-318` も避けられる）。
- **m3** `Vram16Plan.backend` は情報源が 2 つになる（`block.attn.attention_backend` が持つ）。
- **m4** 遅延 import は片側だけで足りる（`lowvram` は先頭で `from engine_h3 import dit as _dit`）。
- **m5** 常駐数のしきい値は、A-G5 で 4 対 8 の差が無ければ削除と事前に決める。全常駐（`n_load ≥ 50`）の扱いを明記。
- **m6** `tokens` の参照の持ち主（`_denoise` の局所変数・`nn.Module._call_impl` の args・forward の引数。swap の包みの args は呼び出しの間だけ）を設計に書き、weakref のテストを足す。
- **m7** token_refiner の CPU テストは常に通ってしまう → `move_module_tree` が呼ばれたことを spy で。
- **m8** `H3GenerateRequest` の項目の追加は手順 7（pipeline）に入れる。
- **m9** qkv の量子化を 1 ブロック 21 回繰り返す（時間への影響の記録）。
- **m10** 上限はほかのプロセスの増加による退避までは防げない（既知の限界）。
- **m11** 設計正本 §8・§11・§2 の振り分けと README §7.6 が更新の対象から漏れている。
- **m12** 連結要求の型だけに `vram16_mode` が入る食い違い（バックエンドの連結スキーマには足さない）。
- **m13** U の見積もりの内訳が無い（数えると注意計算の山は約 5U）。R1 §3.2 と同じ形の表を設計正本に置く。#5・#6 は山の区間ではないので山を下げる効果は持たない。
