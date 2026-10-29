# §1-106 MiniMax H3 の Chained（潜在の持ち越し）の継ぎ目の計測（2026-10-29）

計画書 `velvety-hopping-hartmanis.md`（§4 方式・§5 条件・§6 台本の設計・§7 自己検査）の台本と記録です。製品コード（`engine_h3/`・`api/`・`services/`・操作パネル）は変更しません。台本が製品のパイプラインの段（`_encode`・`_denoise`・`_decode`）をクリップごとに呼び、前のクリップの末尾の潜在を次のクリップの先頭に固定して連結します。数値の正本は記録簿 `Docs/VERIFICATION_LOG.md` §183 です（ここには結果の数値を書きません）。

## 中身

| ファイル | 内容 |
|---|---|
| `c_chain.py` | 計測台本（GPU。1 条件＝1 プロセス）。クリップを順に生成し、連結した潜在から目視用の mp4 を作り、JSON に記録する。`--selfcheck` は CPU だけで台本の正しさを確かめる（重みは読まない） |
| `analyze_chain.py` | 解析台本（CPU。PyAV と numpy）。各条件の JSON と mp4 を読み、継ぎ目の指標を CSV に書く |
| `run_common.ps1` | 起動の共通部。前例 `Docs/Outputs-archive/h3-te-vram-2026-10-29/run_common.ps1` の写しで、作業用フォルダーと台本の名前を変え、終了コードを返すようにしたもの（1 条件＝1 プロセス・時間切れでプロセスの木ごと停止・コンソールを `<名前>.console.log` へ） |
| `run_c.ps1` | 計画 §5 の順に条件を並べたもの（自己検査 → C0 → C1 → C3 → C2 → C4 → C5 → C7 → 解析）。止める条件つき |
| `prompts_c.json` | C1 以降のプロンプト（3 クリップ分）。共通文はレインコートの女性（指紋のプロンプト。`Docs/Outputs-archive/h3-gate3/g2_selftest_g5prompt.cmd` の原文）。クリップ 1・2 は共通文の後ろに 1 文を足したもの。オーナーが差し替えてよい |
| `prompts_c0.json` | C0（指紋の門）のプロンプト。3 クリップとも共通文だけ |

実行後にこのフォルダーに増えるもの: 各条件の `<名前>.json` と `<名前>.console.log`、自己検査の `selfcheck.*`、解析の `analysis.*` と `analysis_*.csv`。mp4 と潜在（`.pt`）は `outputs\h3-chain-2026-10-29\<名前>\`（git 管理外）。

## 走らせ方

リポジトリ直下で（GPU を使う。AviUtl2 を閉じ、ほかに GPU を使うプロセスが無い状態で）:

```
powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-chain-2026-10-29\run_c.ps1
```

- 先頭で `nvidia-smi` を表示し、AviUtl2 が起動していれば止まる（`-SkipPreflight` で無視）。
- 止める条件（計画 §5）: 自己検査が失敗したとき、C0 が終了コード 0 以外のとき（2＝指紋か作り直しの不一致、1＝失敗、-1＝時間切れ）、C1 が 0 以外のとき（後の条件は C1 のクリップ 0 を使い回すため）。
- 時間切れのときは `.console.log` の末尾に `TIMEOUT`、`.json` は `{"timeout": true}`。
- 自己検査だけ（CPU のみ。台本が自分で GPU を隠す）:

```
set PYTHONPATH=%CD%
set PYTHONUTF8=1
.venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-chain-2026-10-29\c_chain.py --selfcheck
```

最後の行が `SELFCHECK PASS`。

- 解析だけをやり直す: `.venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-chain-2026-10-29\analyze_chain.py`（このフォルダーの `c*.json` を全部読む）。

## 条件

共通: FL2VA の int8 ConvRot の重み・20 ステップ・shift 12（映像）／3（音声）・VRAM 16GB mode オン・ブロックの先読みオン・常駐させるブロック数 8（VRAM 16GB mode では実効 4）・文章からの生成（キーフレームなし）・1 クリップ 124 フレーム × 3 クリップ・種はクリップ i で 42＋i・全条件で潜在を保存（`--save-latents`）。

| 順 | 名前 | 画角 | 重なり | 固定の強さ | 映像の注入 | 注意の計算 | クリップ 0 | 復号 | 時間切れ |
|---|---|---|---|---|---|---|---|---|---|
| 0 | `selfcheck` | CPU のみ | – | – | – | – | – | – | 300 秒 |
| 1 | `c0_gate` 指紋の門 | 640×384 | 22 | 1.0 | noise | sdpa | 生成（3 クリップとも共通文） | (a)(b)＋クリップ 0 単独 | 1,500 秒 |
| 2 | `c1_base` 基準 | 1344×768 | 22 | 1.0 | noise | sage | 生成 | (a)(b) | 3,600 秒 |
| 3 | `c3_inject_clean` | 1344×768 | 22 | 1.0 | clean | sage | C1 から再利用 | (a)(b) | 3,000 秒 |
| 4 | `c2_strength05` | 1344×768 | 22 | 0.5 | noise | sage | 再利用 | (a)(b)(c) | 3,000 秒 |
| 5 | `c4_overlap5` | 1344×768 | 5 | 1.0 | noise | sage | 再利用 | (a)(b) | 3,000 秒 |
| 6 | `c5_overlap39` | 1344×768 | 39 | 1.0 | noise | sage | 再利用 | (a)(b) | 3,000 秒 |
| 7 | `c7_control` 対照（固定なし） | 1344×768 | なし | 0 | – | sage | 再利用 | (b)（切り落としなし） | 3,000 秒 |
| 8 | `analysis` 解析 | CPU のみ | – | – | – | – | – | – | 1,800 秒 |

- 重なりはフレーム数で、17m＋5 の形だけ受け付ける（5・22・39 は潜在 2・7・12 個）。
- 映像の注入: noise＝製品のまま（固定する行に 0.999×元の潜在＋0.001×初期雑音を与える）。clean＝台本が `engine_h3.denoise_mask.VIDEO_INJECT_CLEAN` を 1.0 にして元の潜在そのものを与える。
- 復号: (a)＝連結した潜在を 1 本として復号（`once.mp4`）。(b)＝クリップごとに復号し、2 本目以降の先頭の重なり分のフレームと対応する音声を捨ててつなぐ（`perclip.mp4`）。(c)＝固定の強さが 1 未満のときだけ、重なり区間の潜在を線形にクロスフェードしてから 1 本として復号（`xfade.mp4`）。C0 はこれに加えてクリップ 0 だけを製品の `_decode` で mp4 にし（`clip0_fingerprint.mp4`）、sha256 を指紋 `0da6111c…` と比べる。
- クリップ 0 の再利用（`--reuse-clip0 c1_base.json`）: C1 の JSON の条件（画角・フレーム数・種・ステップ・shift・注意の計算・VRAM 16GB mode・先読み・ブロック数・重みのパスとバイト数）とクリップ 0 のプロンプトが同じで、読み込んだ潜在の sha256 が C1 の記録と一致するときだけ使い、違えば断る。
- 終了コード: 0＝完走、1＝失敗（例外・メモリ不足・強さ 1.0 で先頭が前の末尾と一致しない・引数の拒否）、2＝指紋か C0 のクリップ 0 の作り直しが一致しない。

## JSON の主な欄（`<名前>.json`）

| 欄 | 意味 |
|---|---|
| `condition` | 全引数と、重みのパス・バイト数（`weights`） |
| `geometry` | 継ぎ目の計算: `F`・`T`（潜在フレーム数）・`A`（音声潜在数）・`O`・`k`・`S`（2 本目以降の重なりの頭のフレーム）・`J`（新しい部分の頭）・`ka`（音声の重なり）・`F_total`・`T_total`・`A_concat`（連結後の音声潜在の実際の長さ）・`A_expected`（合計フレームから計算した長さ）・`audio_latent_excess`（その差。余りは `write_mp4` が切り詰める）・音声の誤差の 3 欄（継ぎ目ごと。単位は音声潜在＝25 ミリ秒。強さ 0 の C7 は重なりが無いので 3 欄とも `null`）: `audio_head_shift_latents`（符号つき。round(S_i×5/3) − S_i×5/3）、`audio_place_error_latents`（その絶対値）、`audio_fixed_end_vs_video_latents`（固定した音声の末尾 − J_i×5/3。計画 H6 の「最大 2/3 潜在」はこれ。C1〜C3 でも +1/3 になる） |
| `inject` | 注入の切り替えの記録（`mode`・`before`／`after`・`video_mask_reads_patched`・`inject_equals_clean`） |
| `build_report`・`numerics`・`device_name`・`torch` | パイプラインの読み込みの記録と数値の設定 |
| `clips[i]` | `seed`・`prompt`・`prompt_sha`・`source`（generated／reused）・`fix`（k・ka・マスクの値）・`seconds`（encode・denoise）・`text_tokens`・`dit_source`・`sequence`・`m_per_step`・`mask`・`vram16`（上限 `cap_mib`・`num_ooms`・`num_alloc_retries` など）・`peak_vram_mb`・`mem`（作業セットとコミット）・`phases`（製品の段の記録そのまま）・`latents`（形・sha256・有限か・平均絶対値・標準偏差）・`head_equals_prev_tail`（先頭が前の末尾とビット一致するか。映像と音声の別も）・`head_vs_prev_tail_max_abs`・`latents_path` |
| `recheck` | C0 だけ。常駐の DiT でクリップ 0 を作り直したときの `equal_video`・`equal_audio` |
| `decode.fingerprint` | C0 だけ。`path`・`sha256`・`expected`・`match` |
| `decode.once`・`decode.perclip`・`decode.xfade` | `path`・`sha256`・`size_bytes`・`frames`・`audio_samples`（復号した長さ）・`audio_samples_muxed`（mp4 に入る長さ）・`seconds`・`peak_vram_mb_video`・`peak_vram_mb_audio`・`seam_metrics`（mp4 に書く前のメモリ上のフレームから計算した継ぎ目の指標。下の「解析の指標」）。`perclip` は `overlap_agreement`（重なり区間の前後のクリップのフレームごとの PSNR。記録のみで合否には使わない） |
| `mismatch`・`seconds_total`・`mem`・`oom`・`error`・`traceback`・`ok` | 不一致の種類、全体の秒数、メモリの記録、メモリ不足か、例外の理由、完走して不一致がなければ `ok: true` |

継ぎ目が音声潜在の境界に乗らないとき、(a) 1 本復号はクリップの音声を round(S×5/3) の位置に置くので映像と最大 1/3 潜在ずれたまま続き、(b) クリップごとの復号はクリップ自身の時間軸で切るので映像とのずれは無いが J で中身が最大 267 サンプル（8.3 ミリ秒）跳ぶ。H4／H6 の試聴の解釈に要る。

## 解析の指標（`analyze_chain.py` と `seam_metrics` で共通）

- 輝度 Y＝0.2126R＋0.7152G＋0.0722B（BT.709。mp4 からは Y 面を 0〜255 に戻して使う）。彩度は BT.709 の Cb・Cr の大きさ。
- d(t)＝隣り合うフレームの Y の差の絶対値の平均。跳びの比＝d(J)÷（J の前後 12 フレームから J−1・J・J＋1 を除いた区間の d の中央値）。比較用にクリップ 0 の内部の最大の比と、C7 の同じ継ぎ目の比を並べる。
- クリップごとの新しい区間の Y の平均・標準偏差・彩度の平均と、クリップ 0 からの差（焼けの目安）。
- 音声: 継ぎ目 J/24 秒の前後 0.5 秒を 50 ミリ秒の窓で切った RMS の列と、継ぎ目の前後 10 ミリ秒の隣接サンプル差の山（クリック音の目安）。
- 注入の形の比較: C1 と C3 の保存した潜在で、2 本目以降の新しい区間（映像 k 個目以降・音声 ka 個目以降）の差の最大と平均。

| 出力 | 内容 |
|---|---|
| `analysis_summary.csv` | 条件 × 復号の仕方 × 継ぎ目ごとに 1 行（S・J・跳びの比〔mp4 から・メモリ上から〕・クリップ 0 内部の最大の比・C7 の比・区間の輝度と彩度とその差・クリック音の山・mp4 のフレーム数とサンプル数） |
| `analysis_curve_<名前>_<復号>.csv` | d(t) の曲線（mp4 から・メモリ上から） |
| `analysis_audio_<名前>_<復号>.csv` | 継ぎ目ごとの RMS の列 |
| `analysis_h2_latents.csv` | C1 と C3 の潜在の差（クリップごと） |
| `analysis.json` | 解析が書いたファイルの一覧 |

## 結果（2026-10-29）

数値の正本は記録簿 `Docs/VERIFICATION_LOG.md` §183 です。ここには要点だけを書きます。

- 計測としては全て合格しました。全 7 条件（C0・C1・C3・C2・C4・C5・C7）と解析が、終了コード 0・メモリ不足なし・時間切れなしで完走しました（§183.4）。
- C0 の指紋 `0da6111c…` が一致し、クリップ 0 の作り直しも一致しました。強さ 1.0 の全条件で、2 本目以降の先頭が前のクリップの末尾とビット一致しました。強さ 0.5 の C2 は一致しません（設計どおり）。
- 跳びの比では、固定なしの対照 C7 のような桁違いの段差は無いことが分かりました（固定のある条件は 1 付近。C0 のクリップごとの復号の 1 か所だけ約 2）（§183.6）。
- 目視で確かめる点: 強さ 0.5 では継ぎ目 1 の区間から既に暗く、さらに暗くなる流れ（§183.5 H1）、注入の形による潜在の差（§183.7）、クリップごとの復号での音声の継ぎ目の山（§183.5 H4）。輝度の標準偏差の増加は、クリップ 0 の末尾を基準にすると焼けの兆しとは言えません（§183.5 H7）。
- 目視・試聴の材料の一覧（mp4 のパス・継ぎ目のフレーム番号と秒）は §183.10。mp4 と潜在は `outputs\h3-chain-2026-10-29\<名前>\`（git 管理外）に残してあります。
- 目視・試聴の所見は、後で §183 の末尾に小節として足します。
- JSON の `audio_samples` と mp4 から読んだサンプル数が違うのは、AAC が 1,024 サンプル単位で末尾を埋めるためです（§183.9）。
