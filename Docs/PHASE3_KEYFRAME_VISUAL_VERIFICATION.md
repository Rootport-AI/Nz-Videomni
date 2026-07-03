# Phase 3 多キーフレーム条件付け — 目視検証レシピ（**✅完了＝歴史記録**）

> **✅ 2026-07-03 クローズ**: 本レシピの目視は実画像版（`visual_bookend`/`visual_multikey`）＋高解像度版（1280×768・`outputs/visual_review/07_…`）で実施済み。bookend=PASS・multikeyの挙動（途中キーフレーム=磁石・間の遷移は自由領域でプロンプト支配）は**ユーザーが仕様として受容**。結果と分析の正本＝[`VERIFICATION_LOG.md` §17.9-17.10](VERIFICATION_LOG.md)、ユーザー向け解説＝[`FEATURE_GUIDE_KEYFRAMES_AND_ICLORA.md`](FEATURE_GUIDE_KEYFRAMES_AND_ICLORA.md)。以下は手順の歴史記録。

- 作成: 2026-07-02（監督）
- 対象: ユーザー本人（外出から戻ったら実行）
- 前提: **✅ main 入り済**（merge `7f31935`・2026-07-02）。実 backend＋GPU（RTX 4070 Ti SUPER 16GB）。※旧記述「未 merge」は stale だったため訂正（2026-07-03）。
- 上位: [`VERIFICATION_LOG.md` §17](VERIFICATION_LOG.md) ／ [`PHASE3_API_UNFREEZE_WORKORDER.md`](PHASE3_API_UNFREEZE_WORKORDER.md)

> ## ⚠️ 2026-07-03 注記（この目視はまだ有効に実施されていない）
> 監督が 2026-07-03 に誤って目視用として提示した `outputs/phase3_multikey_smoke/bookend/output.mp4` ・ `.../multikey3/output.mp4` は、**自動スモーク `run_smoke.py` の出力で、全キーフレームに同一の合成テスト画像（`outputs/qat_reclaim_baseline/cond_image_512x320.png`＝青い長方形+グラデ）を渡していた**。同一画像を全端点にロックしたため出力が静止画になり、**機能の成否を目視判断できない無効な検証**だった（機構自体は実 backend で動作）。**有効な目視は、下記 `run_visual.py` を「視覚的に明確に異なる実画像」で実行すること**（出力先＝`visual_bookend/`・`visual_multikey/`）。目視サインオフは依然 PENDING。

> **なぜ目視だけ残っているか**: 客観検証（回帰 byte-match＋新経路がクラッシュせず動く＋VRAM 16GB fit）は**全 PASS 済み**（§17.5）。残るのは「実際に良い映像になっているか」＝**数値 baseline の無い品質判断**で、これは人間の目でしか確定できない（WORKORDER §4 の目視項目）。既定経路（T2V・単一 I2V）は byte 一致で無傷確認済みなので、ここで壊すものは無く、品質が不足なら strength 調整や `8n+1` グリッド切替で後追い可能。

---

## 実行前チェック
```bash
cd s:/OriginalApps/12_Nz-LTX23-backend
git branch --show-current   # feature/phase3-api-unfreeze-conditioning であること
```

## 画像を用意する（重要）
目視の肝は「**指定した端点/キーフレームが実際に反映されるか**」。そのため **first と last は視覚的に明確に異なる 512×320 の PNG** を用意する（例: 明らかに違う色調/被写体）。同一画像だと差が見えない。
- 置き場所は任意。以下ではフルパスで渡す。
- 512×320 以外だと条件付けエンコーダでリサイズ/クロップされ忠実度が落ちるので 512×320 推奨。

---

## A. ブックエンド（first + last）
```bash
./.venv/Scripts/python.exe outputs/phase3_multikey_smoke/run_visual.py bookend  <first.png>  <last.png>
```
- frame_idx: first=0（strength 0.9・強め）／last=48（strength 0.7・やや弱め＝FLF 定石）。**48 はサーバーが公式 8n+1 格子へ 41 にスナップ**（＝最後の latent フレーム開始ピクセル・ComfyUI 準拠）。ハーネスがスナップ後の値を表示する。
- 出力: `outputs/phase3_multikey_smoke/visual_bookend/output.mp4`（コンソールに `OPEN THIS -> ...` が出る）。

**見るべき点（受け入れ基準）**
1. **冒頭フレームが first 画像**に一致しているか。
2. **末尾フレームが last 画像**に（概ね）一致し、末尾が固定されているか。
3. 中間が**両端を尊重した自然な補間**になっているか（破綻・急なジャンプがないか）。
4. 端点ドリフト（端がぼやける/ずれる）が強ければ **last の strength を上げる**（0.7→0.8）か first を下げて再実行。

## B. 多キーフレーム（start / mid / end）
```bash
./.venv/Scripts/python.exe outputs/phase3_multikey_smoke/run_visual.py multikey  <imgA.png>  <imgB.png>  <imgC.png>
```
- frame_idx: 0（s0.9）／24→**17 にスナップ**（s0.8）／48→**41 にスナップ**（s0.7）。
- 出力: `outputs/phase3_multikey_smoke/visual_multikey/output.mp4`。

**見るべき点**
1. 先頭/中間/末尾の各位置で**指定画像がそれぞれ反映**されているか（中間はスナップ後の 17≒時間中央付近）。
2. 中間キーフレームが**ハード置換ではなく guide（ソフト誘導）**として効いているか（周辺との連続性）。

---

## strength チューニングの指針
- 出典 ltx23.org（FLF）: **first 0.8–1.0 / last 0.6–0.9**。
- `run_visual.py` の各 mode の strength は編集して再実行可（ファイル末尾の `run(...)` 呼び出し）。

## 品質が不足だった場合の既知レバー（後追い・低コスト）
- **frame_idx グリッド `8n+1`**: ✅**実施済み（commit `d7a56b1`）**。API は idx>0 を `(f-1)//8*8+1` で公式 latent-frame-start 格子へスナップ（`[1, num_frames-8]` クランプ・ComfyUI `LTXVAddGuide` 準拠）。idx==0 は不変。もし目視で「逆に旧『8 の倍数』の方が良かった」等が出れば、`api/models.py` の当該 3 行を戻すだけで比較可能。
- **strength**: first 0.8–1.0 / last 0.6–0.9 の範囲で `run_visual.py` を再実行。
- **num_pixel_frames / reference-video 条件付け**: 今回スコープ外（将来）。

---

## OK だったら（merge 判断）
目視 3 項目が許容範囲なら、`feature/phase3-api-unfreeze-conditioning` を main へ（`--no-ff` 推奨）。その後:
- `VERIFICATION_LOG.md §17.6` の「目視 PENDING」を解消済みに更新。
- `NEXT_SESSION_HANDOFF.md` / spec §13.4 の Phase 3 チェックリストで「凍結 API 解凍」を done に。
- （任意）`outputs/phase3_multikey_smoke/` の検証出力は一次情報として保全 or 手動整理。

## NG だったら
- どの基準が不足かを記録し、上記レバー（strength / `8n+1`）を試す。engine の条件付け機構自体は公式ハイブリッド（idx0=置換 / idx>0=guide）で LTX-Desktop と同一なので、**機構の作り直しは不要**・パラメータ/グリッドの調整で収束する見込み。
