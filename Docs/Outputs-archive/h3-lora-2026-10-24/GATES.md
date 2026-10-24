# §1-104 GPU の門（セルフテスト経由）の結果 — 2026-10-24

開発機 RTX 4070 Ti SUPER 16GB・RAM 64GB。`engine_h3.worker --selftest-generate`（sdpa・seed 42・deterministic off・先読み on・常駐ブロック 8〔16GB mode on では 4〕）。プロンプトは基準の指紋 `0da6111c…` を取った G5 のプロンプト。台本は `gates/gates.sh`、JSON は `gates/*.json`、mp4 は `outputs/h3-gate-lora/`（git 管理外）。LoRA 3 本は `models/MiniMaxH3/LoRA/`。

| 門 | 条件 | ステップ | 全体 秒 | ノイズ除去 秒 | 秒／ステップ | VRAM 山 MB（割当／予約） | RSS 山 GiB | コミット山 GiB | LoRA の適用（対／飛ばし） | sha256（先頭 16） |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | LoRA なし・16GB mode off | 20 | 137.3 | 108.0 | 5.40 | 6,677／7,590 | 22.1 | 34.5 | — | `0da6111cdca853ce` ✔ 基準一致 |
| 2 | LoRA なし・16GB mode on | 20 | 143.2 | 113.5 | 5.68 | 5,065／5,382 | 22.1 | 30.8 | — | `0da6111cdca853ce` ✔ 基準一致 |
| 4a | Turbo・4 step・shift 6/3・640×384・off | 4 | 66.7 | 38.2 | 9.56 | 7,292／8,022 | 23.3 | 36.9 | Turbo 208／0 | `0caa05dcde6e48a8` |
| 4b | Turbo・同上・on | 4 | 69.0 | 39.3 | 9.82 | 5,065／5,382 | 23.1 | 32.9 | Turbo 208／0 | `0caa05dcde6e48a8` ✔ off と一致 |
| 4c | Turbo・4 step・**1344×768**×124f・on | 4 | 286.6 | 231.5 | 57.87 | 6,388／7,600 | 23.5 | 35.2 | Turbo 208／0 | `1ff01ebdd9130c33`（目視用） |
| 5a | Style 3dsrx・off | 20 | 153.2 | 124.6 | 6.23 | 6,908／7,650 | 22.4 | 34.9 | Style 200／0 | `c05bca7bcbbc803a` |
| 5b | Style・on | 20 | 159.3 | 129.4 | 6.47 | 5,065／5,382 | 22.4 | 31.0 | Style 200／0 | `c05bca7bcbbc803a` ✔ off と一致 |
| 5c | Motion・off | 20 | 153.9 | 125.1 | 6.25 | 6,887／7,616 | 22.3 | 34.5 | Motion 208／0 | `a9df669270612d50` |
| 5d | Motion・on | 20 | 159.5 | 130.4 | 6.52 | 5,065／5,382 | 22.3 | 31.0 | Motion 208／0 | `a9df669270612d50` ✔ off と一致 |
| 5e | Style＋Motion・off | 20 | 161.9 | 132.2 | 6.61 | 6,947／7,676 | 22.4 | 34.9 | 200／0・208／0 | `f48a4c2a064fc861` |
| 5f | Style＋Motion・on | 20 | 167.5 | 138.0 | 6.90 | 5,065／5,382 | 22.4 | 31.3 | 200／0・208／0 | `f48a4c2a064fc861` ✔ off と一致 |
| 5g | Style・on・**先読み off（同期 swap）** | 20 | 260.2 | 231.7 | 11.58 | 5,065／5,382 | 21.3 | 28.7 | Style 200／0 | `c05bca7bcbbc803a` ✔ 5a/5b と一致 |

16GB mode on の割り当て器の上限はすべて 14,025MiB・常駐 4 ブロック・OOM と再試行の差分 0。`inactive_split` の山は LoRA なし 450MiB → Style／Motion 約 550MiB → Turbo 640 527MiB・768p 785MiB・同期 swap 1,105MiB。

## 読み取り
- **LoRA なしはビット一致**（門 1・2）。LoRA ありでも **off と 16GB mode on・先読み off の sha256 が全組で一致**＝両経路で同じ形を cuBLAS に渡す設計（4096 行×1024 列の塊）が実機で成立。
- **費用**（640×384×124f・系列約 3.2 万行）: 1 ステップ 5.40 秒 → Style／Motion +0.8〜0.9 秒（+15%）・両方で +1.2 秒（+22%）・Turbo（rank 384／128）+4.2 秒（+78%）。1344×768×124f（系列約 13 万行）の Turbo は 57.9 秒／ステップ（LoRA なしの同条件の sdpa 実測は未取得。参考: 896×1152×124f・sage・LoRA なしで約 22 秒／ステップ〔§169.3〕）。rank の大きい Turbo で費用が見積もり（+3 秒）を大きく上回る。原因の切り分け（小さな行列積の起動回数・B の f32 化の繰り返し）と高速化は台帳 §3-250。**それでも Turbo 4 ステップは 20 ステップより十分短い**（768p: 約 4.9 分／本）。
- **VRAM**: 16GB mode on では割り当ての山は LoRA の有無で変わらない（上限つき）。off では Turbo で +615MB（受け皿の増分と f32 の一時）。主メモリのコミット山は Turbo で +2.4GiB（CPU 正本 1.9GB＋固定メモリの中継）。
## 門 3（常駐の清潔さ・アプリの HTTP API 経由）— 合格

台本 `gates/gate3_resident.py`（結果 `gates/g3_resident.json`・`g3_console.log`）。バックエンドは監督が `run.ps1` で起動し、`POST /pipeline/load` で MiniMaxH3 を読み込んだ。条件は門 1 と同じ（640×384×124f・20 ステップ・seed 42・G5 プロンプト・`embed_mp4_metadata: false`・16GB mode on＝アプリの既定）。`GET /loras` は `lora_dir = models/MiniMaxH3/LoRA`・4 件（試験ファイル `_TEST_unsupported_adaln_only` だけ `supported: false`）。

| 順 | 条件 | keep_resident | 秒 | DiT の出どころ | sha256（先頭 16） | 判定 |
|---|---|---|---|---|---|---|
| J0 | LoRA なし | off | 144.3 | file | `0da6111cdca853ce` | セルフテストの指紋と一致 |
| J1 | Turbo・4 step・shift 6/3 | on | 72.1 | file | `0caa05dcde6e48a8` | 門 4 と一致 |
| J2 | Style | on | 150.3 | **resident** | `c05bca7bcbbc803a` | 門 5 と一致・J1 の LoRA を引き継がない |
| J3 | LoRA なし | on | 135.3 | **resident** | `0da6111cdca853ce` | **J0 と一致**＝常駐 DiT は LoRA を持ち越さない |
| J4 | Style | off | 150.3 | resident | `c05bca7bcbbc803a` | **J2 と一致** |

`metadata.json` の `ic_lora.loras[].audio_strength` はすべて `None`（サーバー側の正規化）。`h3.loras` の適用件数は Turbo 208・Style 200・飛ばし 0。

## 門 5 の取り直し（768p・短尺）— オーナーの指摘による

640×384 では H3 の絵が粗く、Style／Motion の効果がノイズに埋もれて判定できなかった（オーナー目視・2026-10-24）。そこで 1344×768×56 フレーム（約 2.3 秒）・20 ステップ・16GB mode on・sdpa で取り直した（台本 `gates/retake768/gates2.sh`）。4 本とも**同じプロンプト**（先頭に Motion LoRA のトリガー語「camera motion, slow push-in.」、続いて Style LoRA のトリガー語「3dsrx,」、以下は夜の濡れた街の女性の描写・台詞なし）を使い、違いは LoRA だけにした。強度は配布元の推奨（Style は T2V 0.8・Motion は 0.8〜1.0）に合わせた。

| 条件 | ノイズ除去 秒 | 秒／ステップ | VRAM 山 MB（割当／予約） | コミット山 GiB | 適用（対／飛ばし） | sha256（先頭 16） |
|---|---|---|---|---|---|---|
| LoRA なし | 288.8 | 14.44 | 6,375／7,580 | 31.0 | — | `257bbfaf20d57240` |
| Style 0.8 | 321.2 | 16.06 | 同 | 31.7 | 200／0 | `93674f7a101626a2` |
| Motion 1.0 | 321.8 | 16.09 | 同 | 31.5 | 208／0 | `1154503ca6a22386` |
| Style 0.8＋Motion 0.8 | 335.3 | 16.76 | 同 | 31.9 | 200／0・208／0 | `95bde2e1f8a59cbb` |

費用（系列約 6 万行）: rank 16 の LoRA 1 本で +1.6 秒／ステップ（+11%）、2 本で +2.3 秒（+16%）。16GB mode on では VRAM の山は LoRA の有無で変わらない。mp4 は `outputs/h3-gate-lora/retake768/`。目視の結果は VERIFICATION_LOG §176。

- 門 6（オーナー目視）は VERIFICATION_LOG §176 に記録。
