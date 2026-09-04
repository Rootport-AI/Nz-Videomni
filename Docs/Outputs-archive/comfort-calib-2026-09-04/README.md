> **このファイルは実機の `outputs/comfort-calib-2026-09-04/README.md`（git追跡外）のスナップショットである（2026-09-04複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# 快適上限キャリブレーション計測台（第2次・2026-09-04）

このフォルダは、参照動画つきの単発生成（§3-133）と画角拡張＝アウトペインティング
（§3-134）について、動画生成の「快適上限」を実機で測るための計測台です。
製品のコード、`config.yaml`、`state.json`、`Docs/`、`webui/` には一切手を触れません。
書き込むのはこのフォルダの中だけです。

第1次キャンペーン（`outputs/comfort-calib-2026-08-31/`）は**読み取り専用の履歴**です。
そちらのファイルは一行も書き換えません。この計測台は第1次の `calib.py` を複製し、
必要な拡張だけを足したものです。

計測は必ず**製品のHTTPインターフェース経由**で行います。ワーカーを直接叩く方式は
取りません。実際に利用者が通る経路をそのまま測ることが目的だからです。

**結果記録の命名規則（2026-09-04 導入）**: このキャンペーンの一次記録は
`RESULTS_comfort-calib-2026-09-04.md`——`RESULTS_<キャンペーン名>.md` の形にします。
同じ名前のファイルが複数のキャンペーンに散らばると、開いたときにどれなのか分からなくなるためです。
**この規則は旧キャンペーンへ遡及しません**（第1次の `outputs/comfort-calib-2026-08-31/RESULTS.md` は
そのままです）。

---

## 1. 結論から：第1次からの差分

| 追加したもの | 何のためか |
|---|---|
| 計測点の種類 `single_ref`（参照つき単発） | §3-133。参照動画と制御アダプタを載せた単発生成を測る |
| 計測点の種類 `outpaint`（画角拡張） | §3-134。緑のキャンバスを作って外側を描き足す生成を測る |
| `--cooldown`（既定150秒） | 前の点の共有メモリの持ち越しが次の点に混ざらないようにする |
| 判定窓 v3′（`windows_ext` と `v3prime`） | 参照の符号化区間、2.3の画角拡張の隙間区間、2.5の画角拡張の復元区間 |
| `preflight` 副命令 | 計画の全点の算術を、GPUを1秒も使わずに机上で検算する |
| 転送量の90パーセンタイルと毎秒5,000メガバイト超の割合 | 1秒間隔の採取で中央値が0に潰れる問題への対処 |

**第1次の判定列（`axes` / `verdict` / `axes_v2` / `verdict_v2`）は一切変更していません。**
v3′は既存の列を置き換えるのではなく、隣に並べて出します。両方を読み比べられる状態に
しておくのが裁定J1（案B′）の指示だからです。

---

## 2. 計測点の書き方

計測点の一覧（配列）です。第1次と同じ書式に、次の3つの種類が加わりました。

### 2.1 `single`（従来どおりの単発生成・参照なし）

```json
{
  "label": "r_g1_121_t2v_23",
  "kind": "single",
  "base_model": "LTX23",
  "width": 1280, "height": 768, "num_frames": 121,
  "seed": 12345,
  "prompt": "a cinematic tracking shot of a red sports car driving along a coastal road at sunset",
  "accel": { "attention_backend": "sage", "block_swap_prefetch": true,
             "keep_resident": true, "fused_gguf_dequant_kernel": true,
             "vae_mode": "prune_vaed" },
  "baseline_label": null,
  "is_baseline": true
}
```

### 2.2 `single_ref`（参照つき単発）

`single` に `loras` と `reference` が加わります。送信本体には
`reference_video_id` と `loras` が足されます。

```json
{
  "label": "r_g1_361_ref_23",
  "kind": "single_ref",
  "base_model": "LTX23",
  "width": 1280, "height": 768, "num_frames": 361,
  "loras": [ { "name": "canny-control", "strength": 1.0 } ],
  "reference": {
    "video_id": null,
    "source_path": "outputs/99951e90-6b90-4dc7-af40-a78f9d536177/output.mp4",
    "upload_max_frames": 401
  },
  "baseline_label": "r_g1_121_ref_23"
}
```

- `video_id` が `null` のときだけ、`POST /api/v1/upload/video?max_frames=<N>` へ
  素材を1回だけ送ります（項目名は `file`、multipart形式）。返ってきた識別子は
  `reference_uploads.json` に記録され、**同じ素材ファイルと同じ上限フレーム数の
  組み合わせなら、キャンペーン全体で1回しか送りません**。点ごとに送り直すと
  `uploads/videos/` に同じ動画の複製が点の数だけ積み上がるためです。
- `upload_max_frames` は素材の全長（401）にしてあります。エンジン側は
  `frame_cap=num_frames` で必要な分だけ切り出すので（`fast_video_pipeline.py` の
  `_reference_conditioning_for_stage`）、長い素材を1本置いておけば全点で足ります。
- 幅と高さは**128の倍数**でなければサーバーに拒否されます（`api/generate.py`）。

### 2.3 `outpaint`（画角拡張）

`single_ref` に `outpaint` が加わります。

```json
{
  "label": "o_c2_185_23",
  "kind": "outpaint",
  "base_model": "LTX23",
  "width": 1920, "height": 1024, "num_frames": 185,
  "loras": [ { "name": "in-outpainting", "strength": 1.0 } ],
  "reference": { "video_id": null,
                 "source_path": "outputs/99951e90-.../output.mp4",
                 "upload_max_frames": 401 },
  "outpaint": { "pad_left": 320, "pad_right": 320,
                "pad_top": 128, "pad_bottom": 128,
                "freeze_source_audio": true },
  "baseline_label": "o_c2_121_23"
}
```

- `width` / `height` は**最終キャンバス**の大きさです。四辺の余白を引いた残りが
  素材の解像度（1280×768）と一致しなければサーバーに拒否されます。
- `blend_dilation_stage1` / `blend_dilation_stage2` は**送りません**。フロントエンドは
  既定値のままならこの2つを要求から丸ごと省くので、こちらが明示すると
  「利用者が実際に送る要求」ではなくなってしまいます。
- 加速の5項目は、フロントエンドの画角拡張画面が加速の値を一切送らないことが
  確認済みのため（`useOutpaintForm.ts`）、**両系統ともサーバー既定**にしてあります。

### 2.4 共通の決まり

- 加速の5項目は**毎回必ず明示的に送ります**。既定に任せません。
- `baseline_label` には、同じ幾何のより短い基準点のラベルを書きます。
  基準点そのものには `"is_baseline": true` を、動作確認用と暖機の点には
  `"smoke": true` を立てます。**このどちらも無いのに `baseline_label` が空の点は
  `preflight` で落ちます**（書き忘れを机上で捕まえるためです）。
- 土台モデルが計測点の指定と違う場合、その点は実行せず記録だけ残して次へ進みます。
  切り替えは `load` で明示的に行います（計測の途中で勝手に切り替えません）。

---

## 3. 窓の定義（先に決めて書き残す）

数値を見てから窓を決めると、都合のよい窓を選んでしまいます。だから**測る前に**
定義を確定させ、ここに書き残します（§84.7の作法）。

### 3.1 v3 の2窓（第1次から不変・判定の本体）

| 窓 | 定義 |
|---|---|
| 第2段階のノイズ除去 | `logs/server.log` の `generate stage-2 denoise started` から最後の進捗行まで |
| 復元 | 第2段階の最後の行から動画が書かれるまで（2.3）／該当する `PHASE` の実測秒（2.5） |

### 3.2 v3′ で足す窓

| 窓 | 系統 | 定義 |
|---|---|---|
| 参照の符号化 | 2.3 | ワーカー記録の `IC-LoRA reference encode (scale=… tiled=…)` 行について、**その直前のワーカー記録の行が届いた時刻から、当該行が届いた時刻まで**。ワーカーの記録には時刻が無いので、これが唯一誠実に取れる窓であり、**外側の上限値**です。基準にした行そのもの（`anchor_line`）を記録に残します |
| 参照の符号化 | 2.5 | `PHASE 11_reference_encode` の秒数から `[行の到着時刻 − 秒数, 行の到着時刻]`。エンジンが自分で計測しているので**厳密**です |
| 第1段階のノイズ除去 | 2.3 | `generate stage-1 denoise started` から最後の進捗行まで |
| 中間の隙間 | 2.3 | 第1段階の最後の行から第2段階の開始まで。第1段階の復元・1回目の合成・画素の拡大・第2段階の符号化を含む**外側の上限値** |
| 末尾の隙間 | 2.3 | 第2段階の最後の行から動画の書き出しまで。第2段階の復元・2回目の合成・動画の組み立てを含む**外側の上限値** |
| 復元（画角拡張） | 2.5 | **`28_stage2_decode`**。`29_blend2` と `30_decode_encode` は別の列として持ちます |

**参照の符号化の窓は画角拡張の点でも取ります。** 画角拡張は緑のキャンバスを
倍率1の参照として食わせるので、同じ記録行が出るからです。

### 3.3 2.5 の画角拡張で復元窓を差し替える理由

実測（ジョブ `867a5941`）では次のとおりでした。

| 段階 | 秒数 | 予約ピーク |
|---|---:|---:|
| `28_stage2_decode`（第2段階の復元） | 21.92 | 10.822 ギガバイト |
| `29_blend2`（2回目の合成） | 7.29 | 2.887 ギガバイト |
| `30_decode_encode` | **1.78** | 0.48 ギガバイト |

`30_decode_encode` は動画の組み立てだけで、映像が出てくる本体は
`28_stage2_decode` です。`30` を復元窓と呼び続けると、**何も測っていない窓で
判定することになります**。したがって「種類が `outpaint` かつ 2.5」のときだけ
復元窓を `28_stage2_decode` へ切り替えます。他の種類（`single` など）は
第1次と完全に同じ経路を通るので、既存の列は1つも動きません。

### 3.4 2.5 の各段階は転記のみ

2.5 のワーカーは終了時に `GENERATE_REPORT` という一行で、全段階の秒数とピークを
JSON形式で出します。これを `run.json` の `windows_ext.detail.generate_report` に
**そのまま転記**します。段階ごとの窓を到着時刻から作り直すことはしません。
エンジンが正確に測っているものを、より粗い時計で測り直しても、食い違いを
生むだけだからです。

---

## 4. 予測の法則とその根拠（H2′）

参照の符号化の**予約ピーク**は、参照のトークン数に比例します。

```
予約メガバイト ≈ 1086 + 4.902 × 参照トークン数
```

根拠は `logs/ltx_worker.log` の区間ピークの行です。実物を引用します。

```
[ltx_worker] engine.pipeline.fast_video_pipeline: IC-LoRA reference encode
(scale=2 tiled=False channels_last_3d convs=0): allocated 1451 -> 1451 MB,
reserved 1510 -> 10976 MB, interval peak allocated 9256 MB / reserved 10976 MB
(job peak before this interval: allocated 9260 MB, reserved 9722 MB)
```

この行は 1280×768 / 257フレーム・全加速オンの単発生成のものです。
参照トークン数は 10 × 6 × 33 ＝ 1,980、法則の予測は 10,792 メガバイト、
実測は 10,976 メガバイトで、**差は 184 メガバイト（1.7 パーセント）**です。
連鎖生成の4点（±10メガバイト以内）に加えてこの単発の1点も乗ったので、
連鎖から単発への転用は検証済みと扱います。

### 4.1 参照トークン数の出し方

エンジンは参照を「第1段階の解像度をさらに倍率で割った大きさ」で読み込みます
（`fast_video_pipeline.py` の `_reference_pixel_dims`）。第1段階はすでに出力の
半分なので、倍率2なら参照は出力の4分の1の辺の長さになります。したがって

```
参照トークン数 = (幅 ÷ 128) × (高さ ÷ 128) × 潜在フレーム数
```

これは**倍率2かつ幅と高さが128の倍数のときに限り、単発のトークン数のちょうど
16分の1**になります。128の倍数という制約は、まさにこの割り算が割り切れることを
保証するための制約です。

**倍率1（画角拡張）にはこの法則を当てはめません。** 倍率1の符号化はタイル分割の
分岐を通り、同じ記録の実測値が一桁小さいからです（同じ記録の
`scale=1 tiled=True` の行では予約ピーク 2,732 メガバイト）。`predicted_reserved_mb`
は倍率1の点では `null` になり、`prediction_law` にその理由が入ります。

---

## 5. 判定規約 v3′（案B′）

### 5.1 v3 と v3′ の関係

- **v3列**（不変）— 同じ幾何の基準点と比べて、第2段階の窓と復元の窓の共有メモリの
  中央値が持続的に 300 メガバイト以上高いか。
- **v3′列**（新設・別掲）— v3 の2窓に、参照の符号化の窓（§3-133）と
  隙間の窓（§3-134）を加えて同じ物差しで見る。

**両方を併記します。** どちらか一方に丸めません。裁定J1がそう決めているからです。

### 5.2 基準点の取り方

| 窓 | 基準 |
|---|---|
| 第2段階・復元 | 同じ幾何のより短い点（`baseline_label`） |
| 参照の符号化 | 同じ幾何のより短い**参照つき**の点の同じ窓 |
| 隙間（中間・末尾） | 同じ幾何のより短い**画角拡張**の点の同じ窓 |

計測台では `baseline_label` は点につき1つなので、参照つきの点の基準には
**参照つきの点**を、画角拡張の点の基準には**画角拡張の点**を指定してあります。
そうしないと参照の符号化の窓に比べる相手がいなくなります。
（例外は `r_g1_121_ref_23` で、ここは仮説H1の主検定の対にあたるため、
同じ幾何・同じフレーム数の参照なしの点を基準にしています。）

### 5.3 10標本未満の窓は判定不能

窓ごとに `wddm_samples`（GPUメモリの採取標本数）を併記し、
**10標本に満たない窓は判定しません**。`decidable: false` と
`reason_undecidable` を立てて、その窓は結論から外します。

**2.3 の画角拡張の第2段階の窓は、全点で判定不能になる見込みです。**
第2段階が2ステップ・約6.4秒しかなく、1秒間隔の採取では7標本前後にしかならない
からです。これは解析の失敗ではなく**仕様**であり、そのように記録します。
このとき §3-134 の判定は復元の窓と隙間の窓が主役になります。

### 5.4 v3′ の総合判定

- `plateau` — 判定可能な窓のどれかで、基準比 +300 メガバイトを超えた
- `clean` — 判定可能な窓があり、どれも超えなかった
- `undecidable` — 判定可能な窓が1つも無かった
- `baseline` — 基準点そのもの

### 5.5 反転点は必ず2回以上走らせる

境界の近くは非決定的です。判定が反転した点は**必ず2回以上再走**し、
結果が割れたらその点を境界に採りません（§84.6-2 の前例）。
再走用の雛形は `plans/plan_P3b_template.json` にあります。

---

## 6. `metadata.json` の `peak_vram_*` についての注意

**参照つきの点と参照なしの点のあいだで、`peak_vram_mb` と
`peak_vram_reserved_mb` を比べてはいけません。**

`fast_video_pipeline.py:1275` が、参照を通る経路ではジョブの途中で
`reset_peak_memory_stats`（ピーク計数器の初期化）を呼びます。つまり2本の
ジョブで、この2つの数値が数え始めた場所が違います。同じ名前の列に並んで
いても、同じものを測っていません。

`run.json` の `metadata.peak_vram_note` に同じ注意書きを埋め込んであります。
比較には必ず窓ごとの統計（GPUメモリカウンターの実測）を使ってください。

---

## 7. 使い方

Python は必ずバックエンドの仮想環境のものを使ってください。

```
S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\.venv\Scripts\python.exe calib.py <副命令>
```

| 副命令 | 何をするか |
|---|---|
| `preflight` | 計画の全点を机上で検算し `preflight.json` を書く（**GPU不要・最初にこれ**） |
| `preflight --online` | 上に加えて `GET /loras` を叩き、必要なアダプタの有無を確かめる（サーバー起動後） |
| `server start --pid <番号>` | すでに外で起動済みのサーバーを登録する（下の注意を参照） |
| `server status` | 状態確認の結果をそのまま表示する |
| `server stop` | ジョブが走っていないことを確かめてから、プロセス木ごと停止する |
| `load --base-model LTX23\|LTX25` | 土台モデルを切り替え、準備完了になるまで待つ |
| `idle-calib --minutes 3` | 何もしない状態を計測し `calib/calibration.json` を作る |
| `run --plan plans\plan_P2_ltx_ref.json --cooldown 150` | 計測点を1件ずつ順番に実行する |
| `reanalyse [--label ラベル]` | 保存済みの記録から `run.json` を作り直す（**解析の不具合を直したときに、GPUのジョブをやり直さずに済ませるための命令**） |
| `report` | `manifest.jsonl` から一覧表を出力する（v1・v2・v3′の全列） |
| `restore-state` | 計測全体の最後に、元の土台モデルへ戻す |

### 7.1 起動についての注意（重要）

`server start` の切り離し起動は、**保護された（サンドボックス化された）シェルから
呼ぶと子プロセスが即座に落ちます**。保護のかかっていない PowerShell から次のように
起動し、そのプロセス番号を `--pid` で登録してください。

```powershell
$p = Start-Process -FilePath "powershell.exe" `
  -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\run.ps1" `
  -WorkingDirectory "S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni" `
  -RedirectStandardOutput "S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\comfort-calib-2026-09-04\server_console.log" `
  -RedirectStandardError  "S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\comfort-calib-2026-09-04\server_console_err.log" `
  -PassThru -WindowStyle Hidden
$p.Id
```

```
.venv\Scripts\python.exe calib.py server start --pid <上で表示された番号>
```

### 7.2 冷却は150秒

点と点のあいだは既定で150秒待ちます。§84.6-5 で「定常に戻るまで約135秒」と
実測されているので、その**上**に取った値です。最初の点の前と最後の点の後には
待ちません。

### 7.3 実行の順番

```
1. preflight                                   （GPU不要）
2. サーバー起動（上記PowerShell）→ server start --pid <番号>
3. preflight --online                          （GET /loras の確認）
4. load --base-model LTX23
5. idle-calib --minutes 3
6. run --plan plans\plan_smoke.json            （動作確認2本。ここで不合格なら本番に進まない）
7. run --plan plans\plan_P2_ltx_ref.json       （§3-133 の 2.3）
8. run --plan plans\plan_P3_ltx_outpaint.json  （§3-134 の 2.3）
9. load --base-model LTX25
10. idle-calib --minutes 3
11. run --plan plans\plan_P4_warmup.json       （暖機1本）
12. run --plan plans\plan_P5_ltx25_ref.json    （§3-133 の 2.5）
13. run --plan plans\plan_P6_ltx25_outpaint.json（§3-134 の 2.5）
14. report / batch_table.py で集計
15. restore-state                              （キャンペーンの最後に必ず）
```

### 7.4 反転点が出たときの追加測定

判定が反転した点が出たら、`plans/plan_P3b_template.json` を**複製して**名前を変え、
中の `169` を実際に反転した点のフレーム数に置き換えて走らせます。2点入っています。

- `o_c2_169_t2v_23` — 同じキャンバス・同じフレーム数で**拡張なし**の対照。
  こちら側でも復元窓が超えるなら、原因は画角拡張ではなく §4-34 の復号の天井なので、
  画角拡張の予算の判定からは外して別の列で報告します（仮説H4′）。
- `o_c2_169_23_r2` — 反転した点そのものの2回目。再現しなければ境界に採りません。

雛形はそのままでも動く実在の計画です（169フレームを例として埋めてあります）。
`preflight` にもかかります。

---

## 8. 計画ファイル

| ファイル | 点数 | 中身 |
|---|---:|---|
| `plans/plan_smoke.json` | 2 | 動作確認。参照つき単発（768×512/49、2.3全オン）と画角拡張（1536×896/49、2.3既定） |
| `plans/plan_P2_ltx_ref.json` | 7 | §3-133 の 2.3（全加速オン）Tier A |
| `plans/plan_P3_ltx_outpaint.json` | 5 | §3-134 の 2.3（サーバー既定）Tier A |
| `plans/plan_P4_warmup.json` | 1 | 2.5 へ切り替えたあとの暖機。計測点ではない |
| `plans/plan_P5_ltx25_ref.json` | 7 | §3-133 の 2.5（サーバー既定）Tier A |
| `plans/plan_P6_ltx25_outpaint.json` | 5 | §3-134 の 2.5（サーバー既定）Tier A |
| `plans/plan_P3b_template.json` | 2 | 反転点の対照と再走の雛形 |

ラベルには系統の接尾辞（`_23` / `_25`）を付けてあります。両系統が同じ梯子を走るので、
接尾辞が無いと `runs/<ラベル>/` と `manifest.jsonl` が衝突するためです。
計画書 v2 の表の `r_g1_361_ref` は、この計測台では `r_g1_361_ref_23` と
`r_g1_361_ref_25` の2点にあたります。

### 8.1 加速の設定

| 腕 | 注意機構 | 先読み | 常駐 | 融合カーネル | 映像復元器 |
|---|---|---|---|---|---|
| 2.3 全オン（§3-133） | `sage` | on | on | on | `prune_vaed` |
| サーバー既定（他すべて） | `sdpa` | on | off | on | `default` |

全点で `seed` は 12345、指示文は両系統とも 2.3 系の短い車の文で揃えてあります。
素材は全点 `outputs/99951e90-6b90-4dc7-af40-a78f9d536177/output.mp4`
（1280×768 / 401フレーム / 毎秒24フレーム）です。

---

## 9. ファイルの構成

```
comfort-calib-2026-09-04/
├─ calib.py                 計測台の本体（標準ライブラリのみ）
├─ README.md                このファイル
├─ batch_table.py           まとめ表（計画ファイルを渡すと全点の全列を出す）
├─ vram_sampler2.ps1        GPUメモリの採取
├─ preflight.json           机上検算の結果
├─ reference_uploads.json   参照素材の送信記録（識別子の使い回しの台帳）
├─ plans/                   計測計画（上の表）
├─ selftest/                解析器の自己検査（GPU不要・実ログを入力に使う）
│  ├─ parser_selftest.py       記録行の読み取りの検査
│  ├─ analyse_selftest.py      解析全体の通し検査
│  └─ fake_runs/               通し検査が作る作業ファイル
├─ luid.json                選ばれたGPUの識別子と候補一覧
├─ original_state.json      計測開始前の土台モデル（最後に戻すため）
├─ manifest.jsonl           全計測点の結果（1行1件、追記のみ）
├─ calib/calibration.json   待機時の基準値
└─ runs/<ラベル>/
   ├─ point.json            計測点の定義（再解析用）
   ├─ request.json          実際に送った要求
   ├─ job.json              ジョブ照会の最終応答
   ├─ metadata.json         製品が書いた結果情報の写し
   ├─ wddm.csv              GPUメモリの推移
   ├─ dmon.txt              使用率と転送量
   ├─ worker_tail.csv       ワーカー記録＋到着時刻
   ├─ server_slice.log      サーバー記録のこのジョブ分
   └─ run.json              解析結果（v1・v2・v3′の全部）
```

---

## 10. 自己検査

計測台の解析器は、実際の記録ファイルの行を入力にして机上で検査できます。
GPUもサーバーも使いません。

```
.venv\Scripts\python.exe selftest\parser_selftest.py    （記録行の読み取り）
.venv\Scripts\python.exe selftest\analyse_selftest.py   （解析全体の通し）
```

入力は実行のたびに `logs/ltx_worker.log`・`logs/ltx25_worker.log`・`logs/server.log`
から取り直します。写し取った古い行に対して緑になることはありません。
製品側の記録の書式が変わったら、この検査が落ちて知らせます。

---

## 11. 守っている決まり

- 同時に走らせるGPUのジョブは常に1本だけです。
- `outputs/` と `uploads/` の中身は一切消しません。
- 製品のコード、`config.yaml`、`state.json`、テスト、`Docs/`、`webui/` には触れません。
  土台モデルの切り替えは `state.json` を書き換えるため、計測開始前の値を
  `original_state.json` に残し、計測全体の最後に `restore-state` で戻します。
- 第1次キャンペーンのフォルダ（`comfort-calib-2026-08-31/`）は読むだけです。
- `nvidia-smi` は計測台が持つ1秒間隔の1プロセス以外では回しません。
- 加速の設定が実際に効いたかは `metadata.json` の `*_used` と突き合わせ、
  `run.json` の `used_matches_request` に入れます。食い違えば実行中の一行要約に
  `!! acceleration DEGRADED/mismatched` と出ます。値そのものが返らない項目は、
  返らない限り黙って照合を飛ばします。
- `vae_mode_used` だけは、**同じ名前で2つの系統が別のことを答えます**。
  2.3 は「枝刈り版の復元器が走ったか」（`off` / `on`、頼んだのに重みが無くて
  既定に落ちたときは `on->off`）。2.5 は「積んでいる復元器がどちらか」
  （`conv` / `diff`。`engine25/pipeline25.py` の `video_vae_kind`）で、これは
  読み込み時に決まる事実であって、頼み事への返事ではありません。
  2.5 は `vae_mode` を `default` 以外にすると受け付けない
  （`services/engines/ltx25/adapter.py` の `REJECT_TABLE` が 422 を返す）ため、
  2.5 の点で正しい期待値は `conv` の1つだけです。
  照合表はこの系統差を持つように直してあります（`expected_vae_mode_used`）。
  **これは `*_used` の突き合わせに使う語彙の修正であって、判定規則
  （v1 / v2 / v3′）には一切手を入れていません。** 直したあとに 2.5 の13点を
  `reanalyse` し直しても、判定・軸・窓の数値はどれ一つ動きません
  （動いたのは `used_matches_request` の中だけです）。
