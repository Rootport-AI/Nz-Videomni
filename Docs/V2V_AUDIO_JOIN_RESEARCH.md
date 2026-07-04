# V2V 音声接続の先行事例リサーチノート（R0・2026-07-04）

- 位置づけ: G3 試聴で見つかった「音楽のブツ切れ」への対処を、独自発明でなく先行事例に沿わせるための調査（ユーザー指摘による追加ステップ）。実装方針は本書で確定。
- 上位: [`V2V_CONTINUATION_DESIGN.md`](V2V_CONTINUATION_DESIGN.md)・G3 経緯＝VERIFICATION_LOG §24。

## 1. 調査結論（要点）

1. **公式（Lightricks ホスト API / LTX Desktop）に音声境界のブレンド手法の記述は無い**。Extend は「源に音声があれば新規部分の音声を源に整合するよう生成」とだけ述べる。クロスフェード/シーム処理への言及はゼロ（見落としでなく実在しない）。
2. **ComfyUI コミュニティ（2026 時点）でも音声継続は未解決のまま**。RuneXX の「末尾音声を参照音声に使う」構想は未実装の仮説のまま。自動音声連続化や結合後処理の出荷例は確認できず。＝我々のギャップは退行ではなく分野の現在地。
3. **異種レンダリング音声の接合の定石（音響編集の一般実務）**:
   - 非コヒーレントな2素材の重ね合わせクロスフェードは **equal-power カーブ**（linear は中点で音量が落ちる）。ffmpeg `acrossfade`/`afade` では **`qsin`** が equal-power 相当。
   - 尺の相場: セリフ編集 ~10–50ms・**音楽ベッドの転換 ~300–500ms**。
   - 結合前の **LUFS ラウドネス整合**（Adobe が異ソース結合の手順として明記・ffmpeg `loudnorm`）。
   - 補助技法: ノイズフロア／ルームトーン整合（iZotope Ambience Match 相当）＝v1 では不要・残存時の追加手段。
4. **生成側のレバー**: 蒸留パイプラインでは CFG／negative は構造的に使えない（既知）。裏付けがあるのは**条件付けコンテキスト長の最大化**（LTX 公式 Extend も context 最大 20 秒を露出・長文脈が音声一貫性を改善する外部研究あり）。プロンプトへの音楽明示は未検証だが低リスクの試行。
5. **横断的発見**: 主要 AI 動画製品のどれも「実音声→合成音声の接合」の公開手法を持たない。Adobe Premiere の Generative Extend は**実セリフの合成継続を拒否**して環境音のみ延長する設計＝この接合の難しさの傍証。

## 2. 実装方針（本書で確定）

- **クロスフェード採用は先行事例と整合**（矛盾する知見なし）。
- **制約による方式選択**: 教科書的な「重ね合わせ acrossfade」は両素材が継ぎ目を跨ぐ余剰音声（ハンドル）を要するが、届く継続 mp4 はトリム済みでハンドルを持たず、音声だけ重ねると映像と fade 長ぶんズレる。→ **v1 はハンドル不要の標準手法**: 継ぎ目で源の末尾を fade-out・継続の先頭を fade-in（いずれも `qsin`・既定 400ms）＋結合前に継続側を源の実測 LUFS へ `loudnorm`（2パス）。映像はハードカット・総尺不変。
- 実装: `services/video_io.py` の `join_v2v()`（API/エンジン不変・クライアント/検証素材用ヘルパー）。
- **昇格経路（再試聴で不足だった場合）**: ①エンジンが継続 mp4 に音声ハンドル（数百 ms の先行音声）を余分に含める→真の重ね合わせ equal-power クロスフェード ②ノイズフロア整合の追加。いずれも今回はスコープ外。
- **生成側の併用策（G3 再生成に適用）**: context_frames を上限 145（≈6 秒・我々のタイル適合上限）にして音声文脈を最大化＋継続プロンプトに音楽を明示記述＋セリフは源と重複させない。

## 3. 主要出典

- docs.ltx.video Extend API リファレンス（音声は再生成・context ≤20s）
- huggingface.co/RuneXX/LTX-2-Workflows discussions #3（音声継続未解決の明言）
- ffmpeg 公式ドキュメント（afade/acrossfade カーブ一覧・loudnorm）
- Sound on Sound（equal-power クロスフェードの根拠）・Adobe Audition/Premiere ドキュメント（Match Loudness）・iZotope RX（Ambience Match）
- arXiv 2602.20981（長文脈と音声一貫性）・Adobe Premiere Generative Extend の仕様（実セリフ継続の拒否）

## 4. 昇格経路①の実装（ハンドル真クロスフェード・2026-07-04 追記）

R0 §2「昇格経路①」を実装した。V1 のフェードペア（重なり無し＝谷）は**既定のまま不変**、ハンドル方式は**オプトイン**。

### 実装
- **エンジン**（`engine/pipeline/chain_pipeline.py`, `source is not None` 分岐）: トリム前の**全長タイムライン音声**（context＋新規領域、現状トリムで捨てている波形そのもの・フェード無し）を配信 mp4 の隣に `<stem>_audio_handle.wav`（48kHz・元サンプル数のまま・scipy float32）で書き出す。`v2v` done-dict に加算キー `audio_handle_filename` と `handle_context_seconds`（= context 長＝`trim_px/frame_rate`・ハンドル内の継ぎ目オフセット）。**配信 mp4 のパス/mux/トリム/30ms フェードは一切不変**（バイト同一・下記ゲート3で実証）。
- **モック**（`services/ltx_runner.py`）: 同キーを反映＋合成無音の placeholder wav を出力（pytest 契約ピン用・docstring に「合成無音」明記）。
- **`join_v2v`**（`services/video_io.py`）: 加算引数 `handle_audio`, `handle_crossfade_ms=300`。映像はハードカット（不変）。音声は実源音声（A）とハンドル（B）を `acrossfade=d=<ms>:c1=qsin:c2=qsin`（equal-power・既定 tri ではない）でクロスフェード。

### ジオメトリ（要点）
- ハンドル内の継ぎ目は `handle_context_seconds`（hcs）。B を **hcs − クロスフェード長** から始まるようトリム → クロスフェード窓は**完全に継ぎ目より前**（両レンダリングが同じ音楽を描く区間）に収まり、**継ぎ目そのものにはフェードが無い**（B の連続領域＝サンプル連続）。
- hcs は `handle_dur − continuation_dur` で導出（ハンドル＝context 領域＋継続領域なのでサンプルジオメトリ上厳密）。B は源へラウドネス整合（既存と同じ2パス loudnorm）後にフェード。
- 尺: `acrossfade` は合算を d だけ短縮するが、A が継ぎ目で終わり B の先頭 d が継ぎ目前重なりなので、出力音声長＝源音声＋継続音声＝ハードカット映像と一致（出力尺を assert・実測 10.708s で一致）。

### 測定（G3v3・源09 + バイトゲート継続・継ぎ目 J=128 / t=5.375s）
`outputs/v2v_e2e/E2E-A3/`。RMS 谷 depth_ratio＝谷RMS/定常RMS（低いほど深い谷）:

| 方式 | 継ぎ目 audio ratio | 谷 depth_ratio | 谷幅 |
|---|---|---|---|
| フェードペア 400ms（同一継続の参照＝G3v2 相当） | 0.645 continuous | **0.000（ほぼ無音）** | **710ms** |
| ハンドル真クロスフェード **300ms** | 1.238 continuous | **0.421** | 245ms |
| ハンドル真クロスフェード **150ms** | 1.137 continuous | **0.420** | 105ms |

- **ほぼ無音の谷（depth≈0・~710ms、E2E-A2 の G3v2 は ~800ms）は消えた**。ハンドル方式は継ぎ目を跨いでエネルギーを保持し（波形図で継ぎ目が滑らかに連続）、残るのは**クロスフェード窓内の浅い一時的な凹み**（depth≈0.42）のみ。これは源とハンドル context が同一音楽（相関あり）の別レンダで、equal-power でも位相干渉が部分的に残るため（R0 が予見した「phasey」リスク）。
- **150ms は凹み幅を 245ms→105ms へ半減**（深さは同等）。R0「位相っぽければ短縮」に沿ってユーザーへ両方提示。V4（ハードカット・内容起因 ~85ms）が内容の下限。
- 映像 ratio は全方式 1.548 continuous（映像処理は不変）。出力尺 10.708s で A/V 同期維持。
