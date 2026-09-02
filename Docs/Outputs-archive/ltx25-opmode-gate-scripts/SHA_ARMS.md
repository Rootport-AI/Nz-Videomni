> **このファイルは実機の `outputs/ltx25-opmode-gate/scripts/SHA_ARMS.md`（git追跡外）のスナップショットである（2026-09-02複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# §3-124 G-B(2) 無害証明（SHA一致）— 実行手順書

**結論から**: 画角拡張のジョブを **O1・O5・O8 の3本**だけ回し直し、
`outputs/ltx25-outpaint-gate/evidence/` の恒久コピーと**バイト単位で一致する**ことを
確かめます。3本とも一致すれば「推論モードの統一は納品物を1バイトも変えていない」と
言い切れるので、**目視ゲートは不要**という提案がオーナーへ出せます。

2026-08-30 時点で**GPUの実行は1本もしていません**。以下は手順であって、記録ではありません。
ただし**CPUだけでできる事前確認は済ませてあり、全部一致しています**（§5）。

---

## 1. なぜ3本なのか（O5 を外せない理由）

| 腕 | 条件 | 選んだ理由 |
|---|---|---|
| O1 | 四辺パッド・製品既定・1280×768・73フレーム・音声あり素材 | 基準腕。いちばん普通の画角拡張 |
| O5 | `freeze_source_audio=False` | **ボコーダ経路**。`engine25/pipeline25.py` の `enable_deterministic_convolutions`（:234〜）が名指しする、**唯一の既知の非決定源**（音声VAEのボコーダが `conv_transpose1d` をアトミック加算で畳むため、決定化しないと走るたび波形が変わる）。この経路を通さずに「SHA一致だから目視は要らない」とは言えない |
| O8 | 1920×1152・片側パッドのみ | 合法な最大形状。ステージ2のタイル境界が O1 と別の並びになる |

## 2. 事前確認（CPUのみ・GPU窓の前に済ませる）

```
py = S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/.venv/Scripts/python.exe
$py outputs/ltx25-opmode-gate/scripts/canvas_precheck.py
```

**キャンバスのSHAを先に照合します。** キャンバス生成は ffmpeg（`pad_green_mp4`）なので、
ffmpeg が入れ替わっていれば**入力が変わり、出力も当然変わります**。それは§3-124のせいでは
ないのに、先に出力だけを見ると「案Aで壊れた」に見えてしまいます。

| 腕 | キャンバスの凍結SHA-256 |
|---|---|
| O1 | `b42a5c99e1061972c1596db245c4f46a74f2fbb81e053ba5b602b84ac04099f9` |
| O5 | O1 と**同一ファイル**（O1とO5の違いは生成側のつまみだけ） |
| O8 | `0c3fa886a2b78c8fa64723f5d6afc4a962c2f4102e36b7feb274a1471be41c6c` |

**不一致なら、そこで止めてオーナーへ報告**します。調べる先は入力のドリフト（ffmpeg・素材）で
あって、案Aではありません。

## 3. 本走（実GPU・1腕1プロセス・厳格直列）

ドライバは**画角拡張ゲートが使った `g2_run.py` をそのまま流用**します。

```
cd S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni
.venv-engine-ltx25/Scripts/python.exe outputs/ltx25-outpaint-gate/scripts/g2_run.py --arm O1
.venv-engine-ltx25/Scripts/python.exe outputs/ltx25-outpaint-gate/scripts/g2_run.py --arm O5
.venv-engine-ltx25/Scripts/python.exe outputs/ltx25-outpaint-gate/scripts/g2_run.py --arm O8
```

- **1腕1プロセス。** 各実行の前に `nvidia-smi` で空きを確認します。
- 出力は `S:\OriginalApps\12_Nz-LTX23-AviUtl2\_outpaint_g2_tmp\runs\<腕>\output.mp4` に出ます
  （スクリプトの `TMP` 定数のとおり。**この場所は使い捨てなので、証跡は必ず別に控えること**）。
- **`g2_run.py` の `set_acceleration_job`（:148-153）は1文字も変えません。**
  先読みblock swap・fusedカーネル・常駐オフ・sdpa の4つは、この4値のまま比較しないと
  「同じ条件での比較」でなくなります。scope監査の対象です。
- 画角拡張ゲート当時との違いは**製品コードの1行だけ**（`engine25/outpaint25.py:751` の
  デコレータ削除）である、というのがこの比較の前提です。

## 4. 照合

照合相手は**リポジトリ内の恒久コピー**と、`summary.json` に記録された値の**両方**です。

| 腕 | 出力の凍結SHA-256 | バイト数 | 恒久コピー |
|---|---|---:|---|
| O1 | `b7a1eac5db6af3d272d5f0a989ff65fa9a66b2834d7eee1156525407ae3c1b29` | 602,917 | `outputs/ltx25-outpaint-gate/evidence/O1/output.mp4` |
| O5 | `2d20b0cca1c804114ca528b62cc184161f62d2d256f2348d812978278f4e54be` | 593,165 | `outputs/ltx25-outpaint-gate/evidence/O5/output.mp4` |
| O8 | `f69e59c686256108df13927450b8ff7ff8707e80c3d95f9e3a29d2ae187950c5` | 1,239,449 | `outputs/ltx25-outpaint-gate/evidence/O8/output.mp4` |

```
sha256sum _outpaint_g2_tmp/runs/O1/output.mp4 \
          Nz-Videomni/outputs/ltx25-outpaint-gate/evidence/O1/output.mp4
```

（恒久コピーの中身が `summary.json` の記録値と一致していることは 2026-08-30 に確認済みです。
`op_gate.py check` がこの照合を1コマンドでやります。）

## 5. 判定と、外れたときの手順（先に決めておく）

- **3本とも一致** — 無害証明は成立。G8（目視）は**不要＝報告のみでクローズ可**とオーナーへ提示。
- **O5 だけ不一致** — **即ロールバック判断の前に、O5 をもう1本だけ再走**します。
  ボコーダの run-to-run の揺れか、案Aの影響かの切り分けです。
  （旧ゲートでは O5 と O6 がバイト同一だった実績があり、決定化が効いていれば
  新しいプロセス間でも一致するはずです。だからこそ、揺れた場合の手順を先に決めておきます。）
  再走でも不一致なら、そこで停止しオーナーへ報告。
- **O1 か O8 が不一致** — **即停止・オーナーへ報告・ロールバック判断**。
  切り分けの再走はしません（この2本は音声の非決定源を通らないので、揺れの説明が立ちません）。
- ロールバックは `git revert` 1本で済みます（製品の変更はデコレータ1行）。

## 6. 素材と一時フォルダの現況（2026-08-30 確認）

`g2_run.py` は `S:\OriginalApps\12_Nz-LTX23-AviUtl2\_outpaint_g2_tmp\` を参照します。
**このフォルダは今も残っており、中身も当時のままです。**

| 確認したもの | 結果 |
|---|---|
| `_outpaint_g2_tmp/material/src_audio_full.mp4` | `bb67a464…` — リポジトリ内 `evidence/_material/` の写しと**バイト同一** |
| `_outpaint_g2_tmp/material/src_o8_1536x1152.mp4` | `9c2cf3f2…` — 同上 |
| `_outpaint_g2_tmp/load_payload.json` | `79b27a79…` — `outputs/ltx25-outpaint-gate/scripts/load_payload.json` と**同一** |
| `_outpaint_g2_tmp/runs/O1/outpaint_canvas.mp4` | `b42a5c99…`（当時のキャンバスと一致） |
| `models/LTX23/IC-LoRA/in-outpainting/…-0.9.safetensors` | 存在（1,308,778,338 バイト） |

### 消えていた場合の復元手順

```
mkdir S:\OriginalApps\12_Nz-LTX23-AviUtl2\_outpaint_g2_tmp\material
copy Nz-Videomni\outputs\ltx25-outpaint-gate\evidence\_material\src_audio_full.mp4      ...\material\
copy Nz-Videomni\outputs\ltx25-outpaint-gate\evidence\_material\src_o8_1536x1152.mp4    ...\material\
copy Nz-Videomni\outputs\ltx25-outpaint-gate\scripts\load_payload.json                  ...\_outpaint_g2_tmp\
```

1. 上の4つを置き直す（素材2本は SHA `bb67a464…` / `9c2cf3f2…` を照合してから）。
2. `runs/` は `g2_run.py` が自分で作ります。キャンバスも毎回作り直されるので、
   復元は不要です（§2の先行照合がその検算になります）。
3. `load_payload.json` の5つのモデルパスが実在することを確認します。
   1つでも欠けていれば、そこで止めてオーナーへ報告（重みの入れ替えは案Aとは別件です）。
