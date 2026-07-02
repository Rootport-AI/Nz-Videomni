# Phase 3 スライス2「クリップ連結」現状ステータス＆診断 — 次セッションの入口

- 更新: 2026-07-03（監督）
- **次セッションはまずこの文書を読む。** 併読: [`PHASE3_CLIP_CONCAT_DESIGN.md`](PHASE3_CLIP_CONCAT_DESIGN.md)（研究/設計・ただし"成功"トーンは本文書で訂正）／[`PHASE3_CLIP_CONCAT_WORKORDER.md`](PHASE3_CLIP_CONCAT_WORKORDER.md)（原ワークオーダー）
- タスク: **詳しい症状分析＋修正計画の立案は次セッションで行う**（ユーザー指示 2026-07-03）。本文書は"混乱なく引き継ぐ"ための現状固定。

---

## ⚠️ 一行結論
**配管（plumbing）は全部通ったが、肝心の「映像・音声の連続性」は出ていない＝クリップ連結は機能未達。** 監督は配管チェック（テンソル一致・byte-match・VRAM）だけで各フェーズを"PASS"としたが、**デコード後の実映像（特に境界フレーム）を一度も目視していなかった**。ユーザーの目視が失敗を暴いた。

## リポジトリ状態
- branch **`feature/phase3-clip-concat`**（**未 merge**・未 push）。コミット:
  - `760a283` docs（研究/設計）
  - `9fb7111` engine latent-extend 配線（seed/mask/tail・monkeypatch）
  - `5e73e47` chain エンドポイント＋逐次オーケストレーション＋concat
- main には**入れていない**（連結コードは機能未達ゆえ merge 不可）。スライス1 キーフレームは別件で main 入り済（下記）。

## 何が「動く」か（配管・客観 PASS＝ただし連続性は別問題）
- `POST /api/v1/generate/chain`（global 基底＋clip 毎 override・2–8clip）・逐次ループで carry latent を clip 間に受け渡し・concat で単一 mp4 化。
- engine: stage1 に前clip tail を注入し denoise_mask で overlap を凍結・stage1 tail を捕捉/永続化。armed 時のみ動く monkeypatch・finally で disarm（リーク無し検査 PASS）。
- 回帰 byte-match（T2V `23844b4e…`／I2V `a511eda4…`）不変・mock pytest 31・VRAM<16GB・carry がクリップ間を伝播。
- **↑これらは「壊れていない・落ちない・数値が変わらない」の確認であって、連続性の確認ではない。**

## 何が「失敗」か（映像・音声の連続性＝機能の本体）
ユーザー目視（2026-07-03）: 2clip は frame24→25、4clip は 49→50/89→90/129→130 で**hard cut・音声も断絶**。クリップ内は滑らかだが**境界で完全に不連続**（＝各クリップが「同じ被写体の独立シーン」）。

### 根本原因（read-only 調査2本＋監督訂正で確定）
1. **二段パイプラインのミスマッチ（主因）**: carry するのは clip N の **stage1（低解像度・refine前）latent tail**。だが視聴者が見るのは **stage2（refine後）+VAE decode** 映像。次clipは"低解像度の幽霊"を継ぐので実表示と一致しない。かつ stage2 refine は各clip独立。
   - ※注意: stage2 は「fresh noise で stage1 を無視」ではない。`distilled.py:155-185` で stage2 は `initial_video_latent=upscale(stage1出力)` を `noise_scale=stage_2_sigmas[0]` で部分ノイズ後に refine する＝stage1 を継ぐ。ただし disarm ゆえ overlap を再凍結せず・高い noise_scale で stage1 の弱い連続性が薄まる。（初回 agent 分析の "orthogonal/fresh noise" は誤り・訂正済）
2. **carry が弱すぎ/短すぎ（主因）**: K=2 latent(~16px)+overlap_strength=0.5(soft)。ComfyUI looping は overlap ~24フレーム・**単段**サンプラー。K=2 soft では同一seedの独立生成に埋もれる。
3. **overlap を concat で trim 破棄**: 連続性が最も乗る先頭 overlap（先頭 `1+(K-1)*8`=9px）を捨てるので最終映像に stage1 連続性がほぼ残らない。
4. **音声は各clip完全独立生成**（cross-clip conditioning 無し）→音声は原理的に繋がらない。
5. **平滑化処理が未実装**: AdaIN・線形クロスフェード・stage2 overlap 凍結は `9fb7111` が明示的に defer（未実装）。境界を滑らかにするものが何も無い。

**帰結: 現アーキのままでは連続性は原理的に出ない。** ユーザー洞察通り、「clip N の**実デコード最終フレーム**→clip N+1 の I2V 開始条件」という素朴実装の方がむしろ連続的になり得る（実表示フレームを条件にするため）。

## 検証手法の欠陥（次セッションで必ず是正）
- 「テンソル一致・byte-match・VRAM」＝配管の検証。**映像連続性を全く検証していなかった**。スパイクの"GO"は配管のGOに過ぎない。
- **是正: 実装より先に「境界フレームを目視/画素差分する連続性検証」を用意する。** 客観PASSと目視を混同しない。

## 候補となる修正方針（未確定・次セッションでユーザーと確定）
- **D（前提・最優先）**: 連続性の目視/画素ベース検証ハーネスをまず作る。
- **B（有望・簡素）**: clip N の**実デコード最終フレーム→clip N+1 の I2V 開始条件**（素朴 bookend）。表示フレームを条件にするので連続性が出やすく実装小。overlap 併用も可。
- **A（本格・困難）**: latent-extend を真に成立させる＝stage2 も overlap 凍結／full-res latent carry／大きい overlap／AdaIN／CF／音声 cross-clip。二段 distilled ゆえ難度高・payoff 不確実。
- **C（回避）**: 連結せず**長い単一クリップ生成**（能力上 ~481f/20s＝`RESOLUTION_DURATION_CAPABILITY.md`）で長尺化し継ぎ目問題を消す。

## 目視レビュー用アーティファクト（現物パス）
- 2clip: `outputs/b3c8bc8f-c7d5-46b4-a4e8-7f90cc649080/output.mp4`（41f/1.7s・**連続性 NG**）
- 4clip: `outputs/e9e1363e-123d-456b-8e38-8cc987170c9b/output.mp4`（169f/7.04s・**連続性 NG**）
- スパイク/検証プローブ: `outputs/phase3_clip_concat_spike/`（spike.py・phase2_verify.py・rootcause_probe*.py・run_4clip_chain.py）

## 関連: スライス1 キーフレームの目視は別途 PENDING（連結とは別件）
- キーフレーム条件付けは **main 入り済**（merge `7f31935`）。だが**目視サインオフは未了**。
- 私が今セッションで目視用に送った `outputs/phase3_multikey_smoke/bookend|multikey3/output.mp4` は **自動スモーク(`run_smoke.py`)の出力で、全キーフレームに同一の合成テスト画像を使用**＝機能価値を判断できない無効な検証（青い長方形になった理由）。
- **正しい手順**: [`PHASE3_KEYFRAME_VISUAL_VERIFICATION.md`](PHASE3_KEYFRAME_VISUAL_VERIFICATION.md) の `run_visual.py` を**異なる実画像**で実行（出力先 `visual_bookend/`・`visual_multikey/`）。
