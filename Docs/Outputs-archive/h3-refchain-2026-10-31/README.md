# §1-109 MiniMax H3 の Ref2VA の重みでの「固定行の連結＋クリップごとの参照」の計測（2026-10-31）

計画書 `pure-knitting-shell.md`（§3 条件・§4 方式・§5 台本の設計・§6 解析・§7 門と止める条件）の台本と記録です。製品コード（`engine_h3/`・`api/`・`services/`・操作パネル）は変更しません。台本が製品のパイプラインの段をクリップごとに呼び、前のクリップの末尾の潜在を次のクリップの先頭に固定して連結します（固定行方式。設計正本 §3.3・裁定 84）。前回の計測（§183・`Docs/Outputs-archive/h3-chain-2026-10-29/`）の台本を複製し、重みの切り替えとクリップごとの参照画像を足しました。前回の保管庫は記録なので触っていません。数値の正本は記録簿 `Docs/VERIFICATION_LOG.md` §185 です（ここには結果の数値を書きません）。

## 中身

| ファイル | 内容 |
|---|---|
| `cr_chain.py` | 計測台本（GPU。1 条件＝1 プロセス）。前回の `c_chain.py` の複製に、重みの指定（`--transformer`）・クリップごとの参照（`--refs`）・製品の単発生成との一致の門（`--product-recheck`）・継ぎ目の PSNR と潜在の継ぎ目の比を足したもの。強さ 1.0・映像の注入 clean（製品の連結と同じ `video_inject_clean=1.0`）・1 本復号に固定。`--selfcheck` は CPU だけで台本の正しさを確かめる（重みは読まない） |
| `analyze_refchain.py` | 解析台本（CPU。PyAV と numpy）。このフォルダーの `c*_*.json`・`g_*.json` を全部読み、継ぎ目の指標を CSV に書く。前回の `analyze_chain.py` の複製で、条件名の決め打ちと注入の形の潜在の差を外し、台本が記録した PSNR と潜在の比の列を足したもの |
| `run_common.ps1` | 起動の共通部。前回の写しで、フォルダーと台本の名前を変え、事前確認（AviUtl2 の停止・GPU に 2,000MiB 超の他のプロセスが無いこと・コミット上限〔RAM＋ページファイル〕45GB 以上・重み 2 本と画像 2 枚と venv と `weight_identity.json` の存在）で止まるようにしたもの |
| `run_cr.ps1` | 計画 §3.2 の順に条件を並べたもの（事前確認 → 自己検査 → 門 2 つ → C0 → C1 → C2 → C3 → 解析）。止める条件つき |
| `prompts_c0.json` | C0・C1 のプロンプト（3 クリップ分）。共通文は老紳士。クリップ 1・2 は共通文の後ろに 1 文を足したもの（製品の「共通文＋クリップごとの追記」と同じ形）。オーナーが差し替えてよい |
| `prompts_c2.json` | C2 と門 G-R のプロンプト。C0 の 3 本の文頭に `<Picture 1> is the elderly gentleman. ` を置いたもの |
| `prompts_c3.json` | C3 のプロンプト。クリップ 0 は C2 と同じ文（クリップ 0 を使い回す条件）。クリップ 1・2 は文頭に `<Picture 1> is the elderly gentleman. <Picture 2> is a young girl. ` を置き、少女が加わる 1 文を足したもの（少女の服や傘の見た目は文章で言わない。参照が運ぶ） |
| `refs_c2.json` | C2 と門 G-R の参照。3 クリップとも老紳士の三面図（`<Picture 1>`） |
| `refs_c3.json` | C3 の参照。クリップ 0 は老紳士、クリップ 1・2 は老紳士＋赤い番傘の少女の三面図（`<Picture 2>`） |

参照画像は保管庫に追跡済みのものをそのまま指します（複製しません）: `Docs/Outputs-archive/h3-ref2va-2026-10-28/17fde0c1-image.png`（老紳士）・`d80320dd-image.png`（少女）。`refs_*.json` の形は「クリップ数と同じ要素数のリストで、各要素は `{"kind", "slot", "path"}` のリスト（参照なしのクリップは `[]`。パスはリポジトリ直下からの相対）」です。

実行後にこのフォルダーに増えるもの: 各条件の `<名前>.json` と `<名前>.console.log`、自己検査の `selfcheck.*`、解析の `analysis.*` と `analysis_*.csv`、起動の記録 `run_cr.runner.log`。mp4 と潜在（`.pt`）は `outputs\h3-refchain-2026-10-31\<名前>\`（git 管理外）。

## 走らせ方

リポジトリ直下で（GPU を使う。オーナーの了承のうえ、AviUtl2 を閉じ、ほかに GPU を使うプロセスが無い状態で。見込み 約 1 時間 50 分）:

```
powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-refchain-2026-10-31\run_cr.ps1
```

- 先頭で `nvidia-smi` を表示し、事前確認のどれかに当たれば止まる（飛ばす指定はありません）。
- 止める条件（計画 §7）: 自己検査・門 G-F・門 G-R のどれかが終了コード 0 以外のとき（2＝門の照合の不一致、1＝失敗、-1＝時間切れ）。C0・C1 は失敗しても記録して次へ進む。C2 が 0 以外なら C3 を飛ばす（C3 は C2 のクリップ 0 を使い回すため）。解析は、止める条件に掛からず C0 以降へ進んだときに最後に走らせる（自己検査か門で止まったときは走らない）。
- 時間切れのときは `.console.log` の末尾に `TIMEOUT`、`.json` は `{"timeout": true}`。
- 自己検査だけ（CPU のみ。台本が自分で GPU を隠す）:

```
set PYTHONPATH=%CD%
set PYTHONUTF8=1
set PYTHONDONTWRITEBYTECODE=1
.venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-refchain-2026-10-31\cr_chain.py --selfcheck
```

最後の行が `SELFCHECK PASS`。

- 解析だけをやり直す: `.venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-refchain-2026-10-31\analyze_refchain.py`（このフォルダーの `c*_*.json`・`g_*.json` を全部読む）。

## 条件

共通: 20 ステップ・shift 12（映像）／3（音声）・VRAM 16GB mode オン・ブロックの先読みオン・常駐させるブロック数 8（VRAM 16GB mode では実効 4）・Turbo なし・キーフレームなし・音声の参照なし・1 クリップ 124 フレーム × 3 クリップ（継ぎ目 2 か所）・重なり 22 フレーム（潜在 7 個）・固定の強さ 1.0・映像の注入 clean・1 本復号・種はクリップ i で 42＋i。参照画像は製品の `_read_references` が生成の画素数に合わせて縮める（`match`）。

| 順 | 名前 | 重み | 参照（クリップ 0 ／ 1 ／ 2） | プロンプト | 画角・注意の計算 | クリップ 0 | 時間切れ |
|---|---|---|---|---|---|---|---|
| 0 | `selfcheck` | – | – | – | CPU のみ | – | 300 秒 |
| 1 | `g_fl2va_gate` 門 G-F | FL2VA | なし／なし／なし | 指紋のプロンプト（台本の `FP_PROMPT`。3 クリップとも同じ文） | 640×384・sdpa | 生成＋指紋＋作り直し | 1,200 秒 |
| 2 | `g_ref2va_gate` 門 G-R | Ref2VA | 老紳士／老紳士／老紳士 | `prompts_c2.json` | 640×384・sdpa | 生成＋作り直し＋製品の単発生成との一致 | 1,500 秒 |
| 3 | `c0_fl2va` 対照 | FL2VA | なし／なし／なし | `prompts_c0.json` | 1344×768・sage | 生成 | 2,400 秒 |
| 4 | `c1_ref2va_noref` 参考 | Ref2VA | なし／なし／なし | `prompts_c0.json` | 1344×768・sage | 生成 | 2,400 秒 |
| 5 | `c2_ref2va_same` 同じ参照を全クリップに入れる形（本番で最も起こりやすい使い方） | Ref2VA | 老紳士／老紳士／老紳士 | `prompts_c2.json` | 1344×768・sage | 生成 | 2,700 秒 |
| 6 | `c3_ref2va_perclip` クリップごとに違う参照を入れる形（本テーマ固有の機能） | Ref2VA | 老紳士／老紳士＋少女／老紳士＋少女 | `prompts_c3.json` | 1344×768・sage | C2 から再利用 | 2,400 秒 |
| 7 | `analysis` 解析 | – | – | – | CPU のみ | – | 900 秒 |

- 重みは 1 条件 1 プロセスで `--transformer` に渡す（FL2VA＝`minimax_h3_fl2va_pruned_int8_convrot.safetensors`、Ref2VA＝`minimax_h3_ref2va_pruned_int8_convrot.safetensors`）。2 本はバイト数が同じなので、台本はリポジトリ直下の `weight_identity.json` からファイルの大きさと更新時刻が合う項目の sha256 を写して記録する（計算はしない。項目が無ければ記録しないだけ）。
- 参照のある条件では、クリップごとに製品の単発生成と同じ順で参照を処理する: 参照の読み込み → 文章の符号化（参照画像も読む）→ 参照の符号化（映像 VAE の符号化器を作って捨てる）→ ノイズ除去（参照行と固定行を同時に渡す）。
- 門 G-F: クリップ 0 だけを製品の `_decode` で mp4 にし、sha256 を前回と同じ指紋 `0da6111c…` と比べる。FL2VA の重み・指紋の条件・参照なしでないと台本が断る。
- 門 G-R: クリップのループと作り直しの後、DiT を捨てる前に、台本のクリップ 0 の潜在を製品の `_decode` で `clip0_script.mp4` にし、続けて製品の単発生成 `generate()` に同じ要求（参照・プロンプト・種込み）を渡して `clip0_product.mp4` を書かせ、2 つの sha256 を比べる。
- 作り直し（`--recheck-clip0`。門 2 つ）: 常駐の DiT でクリップ 0 を同じ要求（参照込み）で作り直し、潜在がビット一致するかを見る。
- クリップ 0 の再利用（C3 の `--reuse-clip0 c2_ref2va_same.json`）: C2 の JSON の条件（画角・フレーム数・種・ステップ・shift・注意の計算・VRAM 16GB mode・先読み・ブロック数・重みのパスとバイト数と sha256・クリップ 0 の参照の種類と番号とパスとファイルの sha256）とクリップ 0 のプロンプトが同じで、読み込んだ潜在の sha256 が C2 の記録と一致するときだけ使い、違えば断る。使ったときは C2 の `clips[0].references` を写し、`references_source: "reused"` を付ける。
- 終了コード: 0＝完走、1＝失敗（例外・メモリ不足・先頭が前の末尾と一致しない・引数の拒否）、2＝門の照合（指紋・作り直し・製品の単発生成との一致）か、参照と固定行の自己検査（`refs_and_fix`）が一致しない。

## JSON の主な欄（`<名前>.json`）

| 欄 | 意味 |
|---|---|
| `condition` | 全引数と、重みのパス・バイト数（`weights`。transformer には `weight_identity.json` から写した `sha256_cached`）、クリップ 0 の参照（`clip0_references`）、クリップごとの参照（`references_per_clip`）。強さ 1.0・注入 clean・1 本復号は固定値として記録 |
| `geometry` | 継ぎ目の計算: `F`・`T`（潜在フレーム数）・`A`（音声潜在数）・`O`・`k`・`S`・`J`（新しい部分の頭のフレーム）・`J_lat`（連結した映像潜在での固定行の境目＝クリップ i の最初の新しい潜在の番号）・`ka`・`F_total`・`T_total`・`A_concat`・`A_expected`・`audio_latent_excess`・音声の誤差の 3 欄（前回と同じ） |
| `build_report`・`numerics`・`device_name`・`torch` | パイプラインの読み込みの記録と数値の設定 |
| `clips[i]` | 前回の欄（`seed`・`prompt`・`prompt_sha`・`source`・`fix`・`seconds`・`text_tokens`・`dit_source`・`sequence`・`m_per_step`・`mask`・`vram16`・`peak_vram_mb`・`mem`・`phases`・`latents`・`head_equals_prev_tail` など）に加えて、`references`（参照ごとに `kind`・`slot`・`path`〔相対〕・`file_sha256`・`source_size`・`size`〔縮めた後〕・`vision_tokens`・`rows`）、`references_source`（generated／reused）、`refs_and_fix`（台本の自己検査: 参照ありなら `sequence.ref_video` が参照の行数の合計と一致、参照なしなら `sequence` に `ref_video` の鍵が無い、クリップ 1 以降は `mask.video.min` が 0 で `mask.video.inject_clean` が 1.0）。`seconds` と `peak_vram_mb` には参照の符号化（`reference_encode`）も入る |
| `recheck` | 門 2 つ。常駐の DiT でクリップ 0 を作り直したときの `equal_video`・`equal_audio` と、作り直しに渡した参照の数 |
| `product_recheck` | 門 G-R だけ。`match`（2 つの mp4 の sha256 が同じか）、`script`・`product`（それぞれの `path`・`sha256`・`vae_source`・`audio_vae_source`）、製品の `generate` の戻りの参照の記録と `sequence` を台本のクリップ 0 と突き合わせた `references_equal`・`sequence_equal`（不一致のときの切り分け用） |
| `decode.fingerprint` | 門 G-F だけ。`path`・`sha256`・`expected`・`match` |
| `decode.once` | `path`・`sha256`・`size_bytes`・`frames`・`audio_samples`・`audio_samples_muxed`・`seconds`・`peak_vram_mb_video`・`peak_vram_mb_audio`・`seam_metrics`（mp4 に書く前のメモリ上のフレームと連結した潜在から計算した継ぎ目の指標。下の「解析の指標」） |
| `mismatch`・`seconds_total`・`mem`・`oom`・`error`・`traceback`・`ok` | 不一致の種類（`fingerprint`・`recheck`・`product_recheck`・`refs_and_fix`）、全体の秒数、メモリの記録、メモリ不足か、例外の理由、完走して不一致がなければ `ok: true`。有限でない値は書かず `null` にする（その数を `nonfinite_values_nulled` に残す） |

## 解析の指標（`analyze_refchain.py` と `seam_metrics` で共通）

- 輝度 Y＝0.2126R＋0.7152G＋0.0722B（BT.709。mp4 からは Y 面を 0〜255 に戻して使う）。彩度は BT.709 の Cb・Cr の大きさ。
- d(t)＝隣り合うフレームの Y の差の絶対値の平均。跳びの比＝d(J)÷（J の前後 12 フレームから J−1・J・J＋1 を除いた区間の d の中央値）。mp4 から計算した値とメモリ上から計算した値を並べる。比較用にクリップ 0 の内部の最大の比も並べる。
- 継ぎ目の PSNR（台本がメモリ上のフレームの Y で計算。記録のみで合否には使わない）: `psnr_J`＝フレーム J−1 と J の PSNR、`psnr_window_median`＝跳びの比と同じ窓（J の前後 12 フレームから J−1・J・J＋1 を除く）の隣り合うフレームどうしの PSNR の中央値、`psnr_drop`＝中央値 − `psnr_J`（dB）。同じフレームどうしは 99dB で頭打ち。復号器の窓の混ぜ合わせを含む値なので、条件の間で比べる（ほかの実装の数値と絶対値で比べない）。
- 潜在の継ぎ目の比 `ratio_lat_J`（台本が連結した映像潜在から計算。復号器を含まない補助の指標）: d_lat(t)＝隣り合う潜在フレームの差の絶対値の平均。固定行の境目 `J_lat` での d_lat を、同じ窓（前後 12・境目と前後 1 つを除く）の中央値で割ったもの。
- クリップごとの新しい区間の Y の平均・標準偏差・彩度の平均と、クリップ 0 からの差（焼けの目安）。
- 音声: 継ぎ目 J/24 秒の前後 0.5 秒を 50 ミリ秒の窓で切った RMS の列と、継ぎ目の前後 10 ミリ秒の隣接サンプル差の山（クリック音の目安）。

| 出力 | 内容 |
|---|---|
| `analysis_summary.csv` | 条件 × 継ぎ目ごとに 1 行（重みのファイル名・そのクリップの参照の数・S・J・J_lat・跳びの比〔mp4 から・メモリ上から〕・潜在の継ぎ目の比・継ぎ目の PSNR の 3 列・クリップ 0 内部の最大の比・区間の輝度と彩度とその差・クリック音の山・mp4 のフレーム数とサンプル数） |
| `analysis_curve_<名前>.csv` | d(t) の曲線（mp4 から・メモリ上から）とメモリ上の PSNR(t) |
| `analysis_audio_<名前>.csv` | 継ぎ目ごとの RMS の列 |
| `analysis.json` | 解析が読んだ条件・飛ばした JSON（時間切れなど）・書いたファイルの一覧。行数が条件数 × 継ぎ目数とそろえば `ok: true` |

## 結果

- 計測としての合否: 6 条件（門 2 つ＋計測 4 つ）が全て終了コード 0・`ok` 真・メモリ不足と時間切れなしで完走した。門 G-F の指紋の一致、門 G-R の製品の単発生成との mp4 の一致、クリップ 0 の作り直しの一致、全継ぎ目の `head_equals_prev_tail`、参照ありの全クリップの ①②③（系列の参照行・固定行のマスク 0・注入 clean）は、すべて合格した。
- 所要・VRAM・クリップごとの秒・参照 1 枚の増分: 記録簿 `Docs/VERIFICATION_LOG.md` §185.4。
- 継ぎ目の指標と読み取り: 同 §185.5。
- 品質の所見と目視用の材料: 同 §185.6。オーナーの目視と裁定: 同 §185.8（目視済み・裁定 86 済み）。
- 追補（案 B の切り分け）の結果: 同 §185.9（実行済み）。解析は再実行済みで、CSV の行は全条件がそろっている（mp4 を改名して一度欠けたが、元に戻して再解析した）。裁定 86（設計正本）: 対処せず台帳 §3-263 に起票。
- 数値は記録簿だけに書き、ここには書かない。

## 追補（案 B の切り分け）

結果の所見（継ぎ目が C2 でカット・C3 で連続）の原因を、追記の展開と参照の枚数に分けるための 2 条件。台本 `cr_chain.py` は変えず、条件ファイルと起動だけを足した（計画 §12）。

| 名前 | 参照 | クリップ 1・2 のプロンプト | 見るもの |
|---|---|---|---|
| `c4_ref2va_same_stayprompt` | 老紳士 1 枚 ×3（`refs_c2.json`） | 文頭 `<Picture 1>` の札のみ＋共通文＋C3 の追記（少女が来る／並んで歩く。`<Picture 2>` の札は無し） | 参照 1 枚のまま展開を C3 にして連続するか |
| `c5_ref2va_perclip_walkprompt` | クリップ 0 は老紳士 1 枚、クリップ 1・2 は老紳士＋少女（`refs_c3.json`） | 札 2 枚＋共通文＋C2 の追記（歩き去る／ベンチで立ち止まる。少女は行動の文に出ない） | 参照 2 枚で展開を C2 にしてカットするか |

クリップ 0 は両条件とも C2 のものを再利用する（プロンプトは C2 と同一）。

- 走らせ方（GPU の了承が先。約 45 分）: `powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-refchain-2026-10-31\run_cr_b.ps1`。順序は前段の検査、c4、c5、解析。c4・c5 が 0 以外で終わっても続行して記録する。`c2_ref2va_same.json` が無ければ前段で止まる。解析は既存の `c*_*.json` を全部読むので、CSV は上書きされて 2 条件分の行が増える。
- 自己検査と門を省く理由: 台本・重み・参照画像・共通引数が計測 4 条件と変わらず、すでに合格しているため。変えたのはプロンプトと参照の組み合わせだけ。
- 数値は記録簿だけに書き、ここには書かない。
