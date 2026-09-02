> **このファイルは実機の `outputs/stage2_window_sweep/SWEEP_RESULTS.md`（git追跡外）のスナップショットである（2026-09-02複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# §3-57 stage-2窓サイズスイープ 横串集計結果

集計日: 2026-08-09。対象: `outputs/stage2_window_sweep/runs/{w22,w19,w16,w13,w10}/`。
本ファイルは既存の生成結果（`result.json` / `jobs.json` / `server.*.log`）を読んで集計しただけの
レポートであり、新規GPU生成もコード変更も行っていない。

---

## 1. 実験条件

`sweep_manifest.json`（`_generated_by` 注記のとおり `chain_math.compute_chain_layout` の実呼び出しで
検証済みの値）より。

- 固定条件（全ジョブ共通、`api/models.py` の現行本番デフォルトのまま）:
  `attention_backend=sdpa`, `vae_mode=default`, `pipeline=distilled`（8ステップ, guidance_scale=1.0）,
  `block_swap_prefetch=true`, `keep_resident=false`, `fused_gguf_dequant_kernel=true`,
  `chunked_upsample=false`, `nag_enabled=false`, `frame_rate=24.0`
- seed: 1234（全9ジョブ共通）
- 品質アーム（quality arm）: 解像度1280×768固定、クリップ構成は3クリップ×121フレーム
  （`overlap_frames=3`）で全5窓共通。`f_total_latent=42`, `total_px=329`,
  `segment_seam_junctions=[120, 224]`（クリップ継ぎ目、窓に依らず不変のはずの対照）。
- spillアーム: 「組み立て後のタイムライン全体が stage-2 タイル1枚ちょうど」になるよう
  クリップ枚数を調整した最小2クリップ構成（`f_total == v_tile`, `n_tiles == 1`）。
  1920×1088は窓22/19、2560×1472は窓13/10のペアのみ実施（窓16には spillアームのジョブ自体が存在しない）。

---

## 2. 横串表 — 品質アーム（1280×768, 継ぎ目MAD比）

出典: 各 `runs\w<N>\result.json`（`analyze_windows.py` 出力）, `runs\w<N>\jobs.json`,
`runs\w<N>\server.err.log`（`peak_vram=`行）。

| 窓(v_tile) | n_tiles | タイル継ぎ目本数 | タイル継ぎ目MAD比（個々） | タイル平均MAD比 | クリップ継ぎ目MAD比（対照,個々） | クリップ平均MAD比 | ベースラインMAD | 品質ジョブ所要時間 | quality peak_vram |
|---|---|---|---|---|---|---|---|---|---|
| 22 | 3 | 2 | J168=0.1266 / J312=1.9797 | **1.0531** | J120=1.6624 / J224=1.2044 | **1.4334** | 9.0832 | 299.9s | 9247MB |
| 19 | 3 | 2 | J144=0.3192 / J264=0.4645 | **0.3918** | J120=1.5479 / J224=1.2682 | **1.4080** | 9.3919 | 258.9s | 9250MB |
| 16 | 4 | 3 | J120=3.1556 / J216=3.1084 / J312=3.4065 | **3.2235** | J120=3.1556（tile継ぎ目と同一フレーム） / J224=2.5556 | **2.8556** | 4.6293 | 259.6s | 9248MB |
| 13 | 5 | 4 | J96=2.0233 / J168=0.3464 / J240=0.4255 / J312=1.9333 | **1.1822** | J120=2.1982 / J224=1.4659 | **1.8321** | 7.4193 | 260.0s | 9248MB |
| 10 | 7 | 6 | J72=2.1100 / J120=3.2718（clip継ぎ目と同一フレーム） / J168=0.3718 / J216=2.7454 / J264=0.7452 / J312=3.5957 | **2.1400** | J120=3.2718（tile継ぎ目と同一フレーム） / J224=2.2840 | **2.7779** | 4.3508 | 262.9s | 9250MB |

タスク指示で挙げられた実測値（w22=1.05, w19=0.39, w16=3.22, w13=1.18）と一致を確認。w10のタイル平均MAD比は
**2.14**（未提示だった値）。

**重要な構造上の事実**: 窓16と窓10では、タイル継ぎ目（`tile_seam_junctions`）とクリップ継ぎ目
（`segment_seam_junctions=[120, 224]`）が**フレーム120で完全に一致**する（`layout.tile_seam_junctions`に
120が含まれる）。`analyze_windows.py`はこの2種類を別レコードとして出力するため（`kind="tile"`と
`kind="clip"`が同じ`junction=120`で並ぶ）、この窓ではタイル継ぎ目と対照であるはずのクリップ継ぎ目が
同一フレームの同一MAD値を指しており、独立した2本の測定ではない。窓22・19・13ではこの重複は発生しない
（タイル継ぎ目とクリップ継ぎ目が別フレームにある）。

## 2b. 横串表 — spillアーム（4ジョブ、単一フルタイル）

出典: `sweep_manifest.json`の`spill_arm.jobs[].verified_layout`（トークン数・仮説）,
`runs\w<N>\jobs.json`（`generation_time_seconds`, 出力`duration_seconds`）,
`runs\w<N>\server.err.log`（`peak_vram=`行）。

| job | 解像度 | v_tile | 1タイルあたりトークン | %ceiling(40,000基準) | 仮説 | 所要時間(generation_time_seconds) | peak_vram |
|---|---|---|---|---|---|---|---|
| spill_1920x1088_v22 | 1920×1088 | 22 | 44,880 | 112.2% | spill想定 | 349.8s | **12984MB** |
| spill_1920x1088_v19 | 1920×1088 | 19 | 38,760 | 96.9% | clean想定 | 276.4s | **11707MB** |
| spill_2560x1472_v13 | 2560×1472 | 13 | 47,840 | 119.6% | spill想定 | 343.5s | **13601MB** |
| spill_2560x1472_v10 | 2560×1472 | 10 | 36,800 | 92.0% | clean想定 | 259.8s | **11292MB** |

ペア差分（同一解像度内、spill想定 − clean想定）:

- 1920×1088（v22 − v19）: peak_vram **+1277MB**（+10.9%）, 所要時間 **+73.4s**（+26.6%）
- 2560×1472（v13 − v10）: peak_vram **+2309MB**（+20.4%）, 所要時間 **+83.7s**（+32.2%）

**spill/fallback/retry/offload/OOM等のログ証跡**: `runs\w*\server.out.log` / `server.err.log` /
`runner.log` / `jobs.json` 全文を`spill|fallback|retry|offload|OOM|out of memory|Traceback|WARNING|Error`
で検索したが、メモリイベントに関する実質的なログ行はゼロ件（ヒットしたのは全て`prompt`文字列や
ファイル名への偶然の部分一致のみ）。つまり **spillの根拠は`server.err.log`の`peak_vram=`行の数値差のみ**
であり、エンジン側が明示的に「spillした」「フォールバックした」と記録するメカニズムは存在しない
（クラッシュもゼロ件 — 全9ジョブとも`completed`で正常終了）。

**仮説の検証結果**: peak_vramはトークン仮説と方向が一致（spill想定側が常に高い、+1.28〜2.3GB）。
一方、所要時間は「約2倍遅い」という`sweep_manifest.json`の`expected`記述ほどの差は出ていない
（実測+27〜32%、フレーム数正規化ではさらに小さい — 下記考察参照）。VRAM面では仮説の「2〜3GB」という
予測レンジの下限〜やや下振れの範囲に収まった。

---

## 3. env var（`LTX_STAGE2_V_TILE`）がengineに届いた証跡

README記載のとおり `GET /status` にも起動ログにも `STAGE2_V_TILE` を直接エコーするフィールドは無いため、
`result.json`の`layout`（実際に生成されたmp4から逆算した継ぎ目位置）と`sweep_manifest.json`の
`verified_layout`（起動前に`chain_math.compute_chain_layout`を直接呼んで検証済みの期待値）を突き合わせる
間接検証のみが可能。5窓すべてで一致を確認した。

| 窓 | manifest期待 `video_tiles` | result.json実測 `video_tiles` | manifest期待 `tile_seam_junctions` | result.json実測 | 一致 |
|---|---|---|---|---|---|
| 22 | [[0,22],[18,22],[36,6]] | [[0,22],[18,22],[36,6]] | [168,312] | [168,312] | ✅ |
| 19 | [[0,19],[15,19],[30,12]] | [[0,19],[15,19],[30,12]] | [144,264] | [144,264] | ✅ |
| 16 | [[0,16],[12,16],[24,16],[36,6]] | [[0,16],[12,16],[24,16],[36,6]] | [120,216,312] | [120,216,312] | ✅ |
| 13 | [[0,13],[9,13],[18,13],[27,13],[36,6]] | [[0,13],[9,13],[18,13],[27,13],[36,6]] | [96,168,240,312] | [96,168,240,312] | ✅ |
| 10 | [[0,10],[6,10],[12,10],[18,10],[24,10],[30,10],[36,6]] | 同左 | [72,120,168,216,264,312] | 同左 | ✅ |

`runner.log`の`env: LTX_STAGE2_V_TILE=<N>`行（サーバ起動直前にセットした環境変数の記録）とも整合。
5窓のいずれも「window 22のレイアウトのまま」になっていない（README §「確認方法」が警告している
典型的な失敗パターン＝ポート使い回しで旧サーバが生き残るケース）ことを確認済み。**5窓すべてで
env varがengineに正しく届いたと判定できる。**

---

## 4. 考察メモ（数値の記述とは分離。断定はせず、オーナー目視に委ねる論点として整理）

### 4.1 タイル継ぎ目MAD比の窓依存傾向

実測: w22=1.05, w19=0.39, w16=3.22, w13=1.18, w10=2.14（いずれも平均, `threshold=3.0`基準）。

単調な「窓を小さくするほど悪化する」傾向ではない。w19が最良（0.39）で、w22（1.05）より良い。
w16が最悪（3.22、かつ`threshold=3.0`を3本中3本ともDISCONTINUOUS判定＝J120=3.16, J216=3.11,
J312=3.41）。w13は持ち直し（1.18）、w10は再び悪化（2.14、6本中2本がDISCONTINUOUS＝J120=3.27,
J312=3.60）。窓サイズと継ぎ目品質の間に単純な線形関係は見られず、窓ごとの再シード結果に依存する
非単調な挙動に見える。

### 4.2 「対照」であるはずのクリップ継ぎ目MAD比が窓間で変動している件

実測: w22=1.43, w19=1.41, w16=2.86, w13=1.83, w10=2.78。

`segment_seam_junctions=[120, 224]`はどの窓でも同一フレーム位置であり、README/`analyze_windows.py`の
設計意図では「窓サイズの影響を受けない対照（control）」のはずである。しかし実測ではclip継ぎ目MAD比も
w19(1.41)〜w16(2.86)の間で約2倍のレンジで変動しており、対照として不変ではない。

考えられる解釈: stage-2は全編をタイル分割し、タイルごとに独立して再シード（denoise）して仕上げ直す
処理であるため、クリップ継ぎ目のすぐ両側のフレームがどのタイルに属し、そのタイルが継ぎ目からどれだけ
離れているか（タイル内のどの位置に該当フレームが来るか）は窓サイズによって変わる。つまりクリップ継ぎ目
も「stage-2のタイル再仕上げ」という同じ処理の影響を受けており、真に窓サイズ非依存な不変対照とは
言えない可能性がある。

さらに`video_baseline_mad`（継ぎ目から離れた場所でサンプリングした通常のフレーム間MAD、いわば動画全体の
「地の動き量」）自体も w22=9.08, w19=9.39, w16=4.63, w13=7.42, w10=4.35 と2倍以上のレンジで変動している。
これはタイル再仕上げによって動画全体の質感・動きの大きさそのものが窓ごとに変わっている可能性を示唆する
（MAD比は`boundary_mad / baseline_mad`の正規化値なので、分母のbaselineが窓ごとに違う動画に対して
算出されている点に注意が必要）。

このため、「継ぎ目でのみ劣化が起きている」のか「動画全体の質感が窓ごとに変わったことの副作用として
継ぎ目のMAD比も動いている」のかを、MAD比の数値だけから機械的に切り分けることは難しい。特にw16・w10は
前述のとおりタイル継ぎ目とクリップ継ぎ目が同一フレームで重複しており、この2窓については「clip対照が
悪化した」という言い方自体、実質的には「同じ1本のtile継ぎ目が悪化した」ことを二重にカウントしている
だけの可能性もある。

→ 数値だけで「窓Nは継ぎ目劣化が大きい／小さい」と断定するのではなく、`runs\w<N>\quality_v<N>.mp4`と
各`boundary_montage_*.png`をオーナーが目視し、(a) 継ぎ目部分で実際に不連続な絵の飛びが見えるか、
(b) 動画全体の質感・ディテール量が窓によって変わって見えるか、を分けて判断することを推奨する。

### 4.3 「continuous」判定しきい値と各値の位置関係

`outputs/phase3_clip_concat_spike/verify_boundaries.py`の実値確認: `DEFAULT_VIDEO_RATIO_THRESHOLD = 3.0`
（`VIDEO_BASELINE_SAMPLES = 10`）。全`result.json`の`video_threshold`フィールドも一貫して`3.0`。

タスク指示にあった「ratio 2.0想定」は実装の実値（3.0）と異なる。ratio>3.0が`DISCONTINUOUS`、
ratio≦3.0が`continuous`。この基準で見ると:

- w22: 4本中0本がDISCONTINUOUS（最大1.98）
- w19: 4本中0本がDISCONTINUOUS（最大1.55）
- w16: 5本中3本がDISCONTINUOUS（J120=3.16, J216=3.11, J312=3.41。いずれも3.0をわずかに超過）
- w13: 6本中0本がDISCONTINUOUS（最大2.20）
- w10: 8本中2本がDISCONTINUOUS（J120=3.27, J312=3.60）

w16とw10のDISCONTINUOUS判定はいずれも僅差（3.0に対して3.11〜3.60）であり、しきい値をまたぐかどうかの
境界線上にある。仮に「ratio 2.0」を基準にした場合はw13(2.20)やw10のJ216(2.75)・clip J224(2.28)なども
DISCONTINUOUS側に転ぶため、しきい値の選び方自体が判定結果を大きく左右する。この点も機械判定のみに
依拠せず、目視で最終判断することが望ましい。

---

## 5. 目視用ファイル一覧

出典ディレクトリ: `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-LTX23-backend\outputs\stage2_window_sweep\runs\`

### 窓22 (`runs\w22\`)
- 品質mp4: `quality_v22.mp4`
- spill mp4: `spill_1920x1088_v22.mp4`
- montage: `boundary_montage_tile_168.png`, `boundary_montage_tile_312.png`,
  `boundary_montage_clip_120.png`, `boundary_montage_clip_224.png`

### 窓19 (`runs\w19\`)
- 品質mp4: `quality_v19.mp4`
- spill mp4: `spill_1920x1088_v19.mp4`
- montage: `boundary_montage_tile_144.png`, `boundary_montage_tile_264.png`,
  `boundary_montage_clip_120.png`, `boundary_montage_clip_224.png`

### 窓16 (`runs\w16\`)（spillアーム無し — manifest上この窓のspillジョブは存在しない）
- 品質mp4: `quality_v16.mp4`
- montage: `boundary_montage_tile_120.png`, `boundary_montage_tile_216.png`,
  `boundary_montage_tile_312.png`, `boundary_montage_clip_120.png`, `boundary_montage_clip_224.png`

### 窓13 (`runs\w13\`)
- 品質mp4: `quality_v13.mp4`
- spill mp4: `spill_2560x1472_v13.mp4`
- montage: `boundary_montage_tile_96.png`, `boundary_montage_tile_168.png`,
  `boundary_montage_tile_240.png`, `boundary_montage_tile_312.png`,
  `boundary_montage_clip_120.png`, `boundary_montage_clip_224.png`

### 窓10 (`runs\w10\`)
- 品質mp4: `quality_v10.mp4`
- spill mp4: `spill_2560x1472_v10.mp4`
- montage: `boundary_montage_tile_72.png`, `boundary_montage_tile_120.png`,
  `boundary_montage_tile_168.png`, `boundary_montage_tile_216.png`,
  `boundary_montage_tile_264.png`, `boundary_montage_tile_312.png`,
  `boundary_montage_clip_120.png`, `boundary_montage_clip_224.png`
