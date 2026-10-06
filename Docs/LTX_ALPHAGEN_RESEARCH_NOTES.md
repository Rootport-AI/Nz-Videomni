# LTX AlphaGen 事前調査ノート（参考資料）

## 本書の位置づけ

**2026-10-06に実施した、Lightricks の新機能「LTX AlphaGen」を Nz-Videomni に取り込むかどうかの事前調査の記録である。設計資料ではなく、go／no-go の判断材料として収蔵する。**

- 作成日: 2026-10-06（AlphaGen の公開から約 4 日後。数値・状況はすべてこの時点のもの）
- 判断はオーナーが行う。本書は「事実」「選択肢」「監督の推奨とその理由」を分けて書く。監督の推奨には見出しか文頭に「監督の推奨」と明記する。
- 台帳（[`PENDING_TASKS.md`](PENDING_TASKS.md)）への起票は未実施。AlphaGen・マット生成・1 段生成に関する起票は台帳にも文書にも無い。
  > **訂正（2026-10-06）**: 同日中に台帳 §1-83「LTX AlphaGen の実装」として起票した（コミット `88def3a`。ゲート 0 から着手）。
- 調査の内訳: 公式の一次情報／コミュニティと周辺の動向／本製品の LTX 2.5 エンジンのコード棚卸し／AviUtl2 側の受け皿の棚卸し、の 4 系統。
- 本書内では「公式」「実測（第三者）」「コードで確認」「推測」「未確認」を区別して記す。出典は URL か「ファイルパス:行番号」で添え、一覧は付録 A にまとめる。
- コードの場所: `api/`・`engine25/`・`services/` などのサーバー側は `Nz-Videomni/` を起点とする。`native/src/...`（AviUtl2 のプラグイン）と `webui/src/...`（操作パネル）は `Nz-Videomni/AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/` を起点とする（操作パネルの `useGenerationForm.ts` は `webui/src/modes/single/useGenerationForm.ts`）。
- 本書で使う専門用語は初出で短く説明し、まとめて付録 C に置く。

---

## 0. 結論

- **AlphaGen は「透過動画を新しく作る機能」ではない。** 手元の普通の動画を入れると、同じ大きさ・同じフレーム数の白黒の「マット」（白＝見える・黒＝透明・灰＝半透明を表す動画）を返す、LTX 2.5 用の追加部品（LoRA 1 本）である。髪・煙・ガラスなど手作業で抜きにくい縁が得意とされる。
- **本製品との相性は良い。** 必要な土台（LTX 2.5 で参照動画を使う IC-LoRA の経路）は既にある。足りないのは「空のプロンプト」「2 段目を飛ばして原寸で出す経路」「寸法の制約」「劣化しない出力形式」「AviUtl2 への渡し方」などの周辺である。
- **最大の不安は 16GB 級の GPU で実用になるかで、2026-10-06 時点で報告は見つかっていない。** 公式はフル HD・最大尺には H100／B200 級の GPU が要るとしている。
- **監督の推奨は「条件つき Go（段階ゲート）」。** 設計に入る前に小さな現物確認（ゲート 0）を挟み、16GB 級での実用性とマットの品質を見てから本格的に作るかを決める。コードを変えずに試せるのは、LoRA のヘッダに参照の縮小率が書かれている場合に限る（2.2 節の D）。

---

## 1. AlphaGen の正体と公式仕様

### 1.1 何をするものか（公式）

- 正式名は `Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen`（Beta＝試験公開）。
- 中身は **LTX 2.5 用の IC-LoRA** である。
  - LoRA は、本体の重みに小さな差分を足して振る舞いを変える追加部品。
  - IC-LoRA は、その中でも「参照動画を条件として受け取る」種類。本製品では Depth や Pose などで既に使っている形。
- 入力は普通の RGB 動画 1 本。出力はその動画と**同じ大きさ・同じフレーム数のグレースケールのマット動画**（白＝不透明・黒＝透明・灰＝半透明）。
- 公式 X の文言（原文）: 「Give it an ordinary RGB clip and get back an alpha matte that matches your plate frame for frame. No green screen, no masks, no prompt. Built for the edges that are hard to key or roto: hair, fur, smoke, fire, glass, water.」
  - 要旨: グリーンバックもマスクもプロンプトも要らない。髪・毛・煙・火・ガラス・水のような、キーイング（色や明るさで抜く作業）やロトスコープ（手で輪郭をなぞる作業）が難しい縁のために作った。
- 出典: https://x.com/ltx_io/status/2106015208221393339 ／ https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen ／ https://docs.ltx.io/open-source-model/integration-tools/ic-lo-ra-adapters

### 1.2 配布物（公式）

- 増える部品は LoRA 1 本だけ: `ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors`（約 1.31GB・学習 25,000 ステップ。容量の出典は https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen/tree/main ）。VAE（映像の圧縮・復元部品）も transformer（生成の本体）も構造は変わらない。
- Hugging Face では、連絡先の共有に同意してログインしないと取得できない配布（gated＝利用条件つき公開）。
- ベースは LTX-2.5-22B。モデルカードは **distilled 版**（蒸留版＝少ない手順で生成できるよう学習し直した版）を推奨している。
- **LTX 2.3 用は無い。**

### 1.3 公式の使い方（モデルカード）

- 既存の IC-LoRA の処理の流れをそのまま使う。
- **プロンプトは空。**
- LoRA の強さ 1.0、参照動画の強さ 1.0、参照の縮小率 1（等倍）。
- **Stage-2 を省略する。** Stage-2 とは、Stage-1（半分の解像度で描く段）の結果を 2 倍に引き伸ばして仕上げる 2 段目のこと。
- 幅・高さには出したい寸法の 2 倍（例: 3840×2176）を渡す。理由は「Stage-1 は要求の半分で描き、Stage-2 を飛ばすので 1920×1088 で出てくる」ため。
- 前処理は「引き伸ばさずに余白で埋める」。縦横は 32 の倍数、フレーム数は 8n+1（9・17・…・121・145 など）。
- なお「CFG 1（プロンプトへの従い具合の強調を使わない）・STG off（別系統の品質補正を使わない）」は、モデルカードでは確認できておらず、v1.4.2 のリリース文（1.4 節）に書かれている。

### 1.4 公式の中での食い違い（公式）

- LTX-2 v1.4.2（2026-10-02 公開）に専用の処理 `ltx_pipelines.alpha_gen` が入った。
  - リリース文: 「a one-stage IC-LoRA pipeline on the full model with video CFG 1 and STG off」（full 版の上で、1 段の IC-LoRA の処理。映像の CFG 1・STG off）。
  - `alpha_gen.py` のモジュール説明文: 「one-stage IC-LoRA generation with multimodal guidance」（1 段で、複数の手がかりによる誘導つきで生成）。
- full 版は非蒸留版（蒸留していない元の版）のこと。モデルカードは distilled 版を推奨しており、**公式の中で推奨の版が食い違っている**。どちらが品質で勝るかは未確認。
- 出典: https://github.com/Lightricks/LTX-2/releases/tag/v1.4.2 ／ https://raw.githubusercontent.com/Lightricks/LTX-2/main/packages/ltx-pipelines/src/ltx_pipelines/alpha_gen.py

### 1.5 制限（公式）

- 入力は動画 1 本。動画→動画だけで、文章→動画・画像→動画・音声の出力は無い。
- 解像度の上限は 1920×1088。
- フレーム数の上限は 145（8n+1。モデルカード）。モデルカードは「145 を超えると元の映像の中身がマットに漏れる」とする。docs は「121 以下が最も予測しやすい」「longer clips are unsupported」（それより長いものは対象外）としている。
- **フル HD・145 フレームには H100／B200 級（データセンター向け）の GPU が要る。**
- 何を前景（残す側）にするかはモデルが自動で決める。プロンプトやマスクで指定する手段は無い。
- 得意なもの: 髪・毛・煙・火・薄い布・ガラス・水。

### 1.6 出力の形（公式）

- マットは、グレースケールの絵が入った普通の RGB 動画として出てくる。
- 合成の式は `out = rgb*alpha + bg*(1-alpha)`。これは **直線アルファ**（色とは別に透明度を持つ形）の式である。前乗算アルファ（色にあらかじめ透明度を掛けておく形）ではない。
- 公式の Python には透過つき（RGBA）動画の書き出しは無く、「合成ソフト側で当てるか、ProRes 4444 で書き出す」としている。

### 1.7 ライセンス（公式＋推測）

- 重みは LTX-2.x Community License（2026-08-11 発効）。年間売上 1,000 万ドル以上の事業者は有償契約が要り、派生物は同じライセンスになる。
- これは既存の LTX 2.5 の重みと同じライセンス。本製品は重みを再配布せず利用者が自分で取得する運用なので、新しい障害は小さい（推測）。
- 出典: https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x

### 1.8 量子化との相性（推測）

- LoRA を足すだけなので、本製品が使う GGUF（Q4_K_M・Q6_K）や fp8（重みを 8bit の小数で持つ形式）の transformer にもそのまま重ねられる構造である（推測）。GGUF は重みを小さく詰めた配布形式、Q4_K_M などはその詰め方の種類。
- 公式は bf16（詰めていない状態）の distilled 版でしか検証していない。**量子化で半透明の階調がどれだけ落ちるかは未知。**
- 他の LoRA（画風の LoRA など）との同時使用は想定外とみるのが自然（推測）。

---

## 2. 本製品との適合度

### 2.1 できていること（コードで確認）

- **LTX 2.5 エンジンは参照動画つき IC-LoRA に対応済み**（Single・Chained・Outpaint・Inpaint）。
  - API の受け口: `api/models.py:521-538`（`loras[]`・`reference_video_id`・`reference_video_strength` は 0〜1 で既定 1.0・`conditioning_attention_strength`）。
  - 2.5 の変換部品（アダプタ）は IC-LoRA を対応済みに分類: `services/engines/ltx25/adapter.py:247-250`・`:817-850`。
  - 2.5 で使えない機能は `two_stage_hq` と `prune_vaed` だけ: `adapter.py:140-143`。
  - 参照動画は Stage-1 だけに後ろへ連結する: `engine25/reference25.py:458-581`（Stage-1 の判定は `:517`）。
  - 参照の縮小率は LoRA ファイルの見出し部分（ヘッダ）の `reference_downscale_factor` から読む: `reference25.py:102-167`。
  - LoRA は Stage-1・Stage-2 の両方に掛かる（公式は Stage-1 だけ。承知の上での差）: `Docs/LTX25_RESEARCH_NOTES.md:556`。
- **transformer の既定は distilled 版の GGUF**: `scripts/manifests/20-ltx25.json:13`・`:68`。distilled 用の手順設定も組み込み済み: `engine25/pipeline25.py:161-162`・`:194`・`:1256-1257`。
- CFG について: API は `pipeline` が `distilled` のとき `guidance_scale` が 1.0 以外なら 422（入力エラー）を返す（`api/models.py:561-569`）。2.5 のアダプタはこの値を使わずに進む。STG は本製品に無い。**公式の「CFG 1・STG off」は既定のままで満たしている。**
- LoRA の読み込み:
  - `models/LTX23/StyleLoRA` を走査する（既定値は `config.py:144`）か、`config.yaml` の `model.ic_loras` に登録する。
  - 種別の判定: ヘッダに `reference_downscale_factor` があるか、config の `preprocess` が `"none"` 以外なら参照動画を使う種別（control）、どちらでもなければ画風の種別（style）: `services/lora_registry.py:286-312`。
  - A/B 形式と kohya 形式の両方を読める: `engine/gguf/ic_lora_common.py:156-247`。

### 2.2 足りないこと A〜F（コードで確認＋一部推測）

- **A. Stage-2 を飛ばして原寸で出す 1 段の経路が無い。**
  - `pipeline25.generate` は常に 2 段で動く（`engine25/pipeline25.py:1688-1702`）。Stage-1 は要求の半分の解像度。
  - 本製品は公式の `ICLoraPipeline` を使わず、`DistilledPipeline` に参照動画を差し込む作りになっている。しかも「いまが Stage-1 か」の判定を `height != full_height // 2` に頼っている（`engine25/reference25.py:517`）。このため、今の差し込み方のままでは 1 段にできない。
  - 製品コードに公式の `skip_stage_2` の使用は無い。
- **B. 空のプロンプトが通らない。**
  - API が `min_length=1` で 422 を返す: `api/models.py:347`・`:1057`。
  - 操作パネルも `promptEmpty` で生成ボタンを止める: `useGenerationForm.ts:1062`。
  - 空白 1 文字なら通る（前後の空白を削っていない）。ただし文章理解部品（Gemma 4）に空白 1 文字を通したときの影響は未確認。
- **C. 参照動画つきでは幅・高さが 128 の倍数に限られる。**
  - `api/generate.py:94-95`。操作パネルも 128 刻み（`useGenerationForm.ts:44-50`）。1920×1088 は 422 になる。
  - 公式手順どおり 2 倍の寸法（3840×2176）を要求すれば 128 の倍数の検査は通る。ただし本製品では Stage-2 を経るので、要求どおり 3840×2176（1920×1088 の 4 倍の画素数）で出てくる。
  - 参照動画は中央で切り抜く作りで、余白で埋める処理は無い（`engine25/reference25.py:224-226`）。公式の「余白で埋める」と食い違う。
  - 出力の寸法を参照動画から決める経路も無い（リクエストの width／height で決まる）。
- **D. LoRA のヘッダに縮小率が無いと、エンジンが止まる。**
  - AlphaGen の safetensors に `reference_downscale_factor` が入っているかは未確認（手元に現物が無い）。
  - エンジンは、参照動画を使うときにヘッダで縮小率を宣言した LoRA が 1 本も無ければ、`config.yaml` に登録してあっても例外で止まる（`engine25/reference25.py:141-149`。`pipeline25.generate` から必ず呼ばれる `engine25/pipeline25.py:1636`）。
  - さらに `config.yaml` で control 種別にするには `preprocess` を `"none"` 以外にする必要がある。選べるのは `"none"`・`"canny"`・`"dwpose"`・`"depth"` だけ（`config.py:69`・`services/lora_registry.py:288`）なので、登録すると参照動画がエッジ・骨格・深度の絵に前処理されてしまう。
  - **ヘッダに縮小率が無ければ、エンジン側の判定と、登録の選択肢（「前処理なしの control」）の両方に小さな手当てが要る。** ヘッダにあれば手当ては不要。
- **E. 出力が劣化する形式（H.264・色を間引く形式）だけ。**
  - エンジンは `encode_video` を呼んで書き出す（`engine25/pipeline25.py:1707-1713`）。libx264・yuv420p・crf 19 の実体は、エンジン用の仮想環境（venv）に入った公式部品 `ltx_pipelines/utils/media_io/encode.py`（`crf: int = 19` の引数がある）。
  - 劣化しない書き出しは、アプリ側の入力加工用 `services/video_io.py` の `libx264rgb -crf 0` と ffv1 だけ（`:145-231`・`:234-383`）。
  - マットを H.264 で保存すると縁の階調が崩れる（4 節）ので、ここは手当てが要る。
- **F. 快適上限に「1 段・原寸・参照の倍率 1」の負荷の形が無い。**
  - 快適上限とは、本製品が「この大きさ・尺までなら VRAM（GPU のメモリ）から溢れず快適に動く」として画面に示す目安。
  - 単発生成は 1 本の線で表している（`scripts/manifests/20-ltx25.json:45-53`。4bit で 46,920 トークン）。倍率 1 のアダプタはこの線の根拠に含まれていない（`Docs/COMFORT_LIMIT_TABLE.md:314`）。
  - 推測: 1920×1088×145 フレームを 1 段で処理すると、Stage-1 のトークン（処理単位の数）は生成側 38,760 ＋参照側 38,760 ＝約 77,520 相当になり、今の線とは別物の負荷になる。

### 2.3 変更箇所の見込み（設計案ではなく、差分が生じる場所）

- `engine25/pipeline25.py` — 1 段の分岐、64 の倍数を前提にした箇所の見直し（A）
- `engine25/reference25.py` — Stage-1 の判定、余白埋め、縮小率の宣言が無いときの扱い（A・C・D）
- `engine25/worker.py`
- `services/engines/ltx25/adapter.py`
- `api/models.py` — prompt の制約、新しい指定項目（B）
- `api/generate.py` — 128 の倍数、種別の判定（C）
- `services/lora_registry.py`・`config.py` — 「前処理なしの control」の選択肢（D が必要になった場合）
- 出力 — 公式部品の `encode_video` には既存の crf の引数があるが、それだけでは色の間引き（4:2:0）は残る。書き出し側に手を入れるか、`services/video_io.py` を流用する（E）
- WebUI（操作パネル） — `promptEmpty`、128 刻み、参照動画の寸法から出力寸法を決める（B・C）
- 快適上限 — マニフェストと `Docs/COMFORT_LIMIT_TABLE.md`（F）

補足:

- 非蒸留版（`two_stage_hq`）は、「LTX 2.3 でできることを 2.5 でもできるようにする」という要件のもとで、2.5 では対応しない裁定でクローズされている（`Docs/PENDING_TASKS_CLOSED.md:1451-1458`。理由は「2.3 の側がまだ非蒸留に対応していないから」）。恒久的に禁じた裁定ではない。full 版を使う公式の専用処理（1.4 節）を載せる場合はこの裁定と衝突するので、**オーナーの再裁定が要る**。選択肢としては残る。
- 関連する未決事項: LoRA にベースモデルの区別を持たせるかは未決（`Docs/PENDING_TASKS_CLOSED.md:1463`）。AlphaGen は LTX 2.5 専用なので、この論点に触れる可能性がある。

> **訂正（2026-10-06）**: 台帳と CLOSED は行番号で参照しない規則のため、参照先は CLOSED §3-102（クローズ理由の `pipeline` 項と、判断材料の (2)）と読み替える。

---

## 3. AviUtl2 側の受け皿と利用者の操作

### 3.1 (a) マット動画をそのまま渡し、AviUtl2 で当てる

**どの候補も実機の画面では未確認。** 確からしい順に並べる。

- **候補 1（v2.0.54 でも使える・手堅い）**
  - 元動画の上のレイヤーにマット動画を置き、ルミナンスキー（明るさを透明度に変えるフィルタ。基準輝度・輝度範囲・モードを持つ）を掛ける。
  - それを「クリッピングオブジェクト」（下のオブジェクトを自分の形で切り抜く設定）にして、下の元動画を切り抜く（`aviutl2.txt:559-568`）。
  - オブジェクトが 2 つになるので、まとめて動かすにはグループ制御が要る。
  - 半透明の階調がまっすぐ（直線的に）保たれるかは未確認。
- **候補 2（有力だが推測）: 部分フィルタやマスクの「マスク画像」に動画ファイルを選ぶ**
  - 元動画のオブジェクトに部分フィルタかマスクを付け、そのマスク画像の選択肢で動画ファイルを選べば、トラックマット（別素材の明るさで透明度を決める機能）に近い働きになる見込み（推測）。
  - 根拠は 2 種類に分かれる:
    - 更新履歴（beta42。`aviutl2.txt:1168`）: 「オブジェクト設定のマスク画像選択のリストにシーンやファイル選択も表示するようにした」。
    - 本体（exe）の中の文字列: マスク画像の選択肢の並びに「動画ファイルから選択」がある。また「拡張色設定」「前方から合成」などと同じ並びの合成方法の選択肢として「**輝度をアルファ値として乗算／上書き**」がある。ただし、これがどのエフェクトの項目かは特定できていない。
  - 部分フィルタの説明（`Default.aul2:134-135`）に「画像ファイルやシーンをマスク画像として選択」とある。
  - マスク画像の明るさと透明度のどちらが使われるかは未確認。
  - 参考: フィルタ効果「画像合成」（beta22 で追加 `aviutl2.txt:924`）は、説明上は「下位のレイヤーにあるオブジェクトを 1 つの画像オブジェクトとして合成」するもので、マットを当てる道具かどうかは未確認。
- **候補 3（仕組みの上では可能・実機未確認）**
  - Lua スクリプト（.anm2）で作る。`obj.load("movie",file[,time])` で同じ時刻のマットを読み、`obj.pixelshader(...)` で元動画に当てる。
  - v2.0.54 に同梱の `lua.txt` にも `obj.load("movie",file[,time])`（368 行）と `obj.pixelshader`（907 行）があり、v2.0.54 でも組める。最新版の `lua.txt` での行番号は未確認。
  - 注意: `"mask"` の合成は「α値のみを乗算 ※RGB値は利用されません」（v2.0.54 の `lua.txt:931`）。マットは明るさで透明度を表しているので、**明るさを透明度に移す処理（シェーダー）を自分で書く必要がある**。
  - 標準スクリプト `script.anm2:1213` に、動画を読んで掛け合わせる同じ形の前例がある。
- その他（公式サイト https://spring-fragrance.mints.ne.jp/aviutl/ の更新履歴による。手元では未確認）
  - 「上のオブジェクトでクリッピング」は v2.1.5 で復活した（v2.0.54 には無い）。
  - 「画像合成」は v2.1.8 で配置の設定が追加された。
- 仮想バッファ（`*tempbuffer`）を経由する組み方もできる。
- AviUtl2 の内部は、色に透明度を掛け合わせ済みの RGBA16bitFloat で扱っている（`aviutl2.txt:41`）。

本製品の側:

- **オブジェクトのエイリアス（オブジェクトの設定を文字で書き出したもの）を書き換えて置き直す前例がある。** 物体追尾が部分フィルタを書き換える処理: `native/src/bridge.cpp:1960-2000`。
- v2.0.54 に同梱の SDK（プラグインを作るための公式の開発キット）にはエフェクトを追加する仕組みが無い。最新の SDK（2026/7/25。公式サイトによる・手元では未確認）には `create_effect`／`delete_effect`／`set_object_flag(CLIPPING_OBJECT …)` がある（v2.1.x が必要）。

### 3.2 (b) 透過つき動画を作って渡す

- AviUtl2 本体だけで読めるのは AVI・WAV・BMP・PNG・JPG・GIF（`aviutl2.txt:50-52`・`:384`・`:390-392`。最新の v2.1.12 も同じ＝公式サイトによる・手元では未確認）。Media Foundation 経由の読み込み部品は削除済み（`:785`）なので、mp4 などは入力プラグイン頼みになる。
- 開発機の入力プラグインは L-SMASH Works（Mr-Ojii 版 `lwinput.aui2`）。透過つきの 8bit 形式は RGBA32bit で、ProRes 4444 などは PA64（透明度を掛け合わせ済みの 16bit）で AviUtl2 に渡す作り（https://github.com/Mr-Ojii/L-SMASH-Works/tree/master/AviUtl2 の `video_output.c`）。
- ProRes 4444 の MOV・UtVideo RGBA の AVI・qtrle・FFV1・APNG（いずれも透明度を持てる動画形式。付録 C）は透過つきで読める見込み（**実機未確認**）。
- WebM（VP9 の透過つき）は、標準の VP9 の読み取り部品が透明度を捨てるため、L-SMASH の「Preferred decoders」に `libvpx-vp9` を指定する必要がありそう（未確認）。
- 本製品がタイムラインに置く経路は「動画ファイル」＋「映像再生」のエイリアス（`native/src/alias_util.cpp:594-648`。`ファイル=` にパス、`音声付き=1`）。
  - 拡張子の検査は無い（`is_support_media_file` は未使用）。`.mov`／`.avi` でも L-SMASH が読めれば同じ経路を通る見込み。
  - PNG 連番は、画像オブジェクト用のエイリアスを組み立てる処理が無いので、今の経路では置けない。静止画 1 枚なら `create_object_from_media_file` で置く前例がある（Inpainting の黒い背景 PNG）。
  - 拡張子 `.mp4` の決め打ち:
    - 受け取る側（生成結果。透過つき動画にするなら見直し対象）: `api/jobs.py:41,47`・`native/src/bridge_core.cpp:820`。
    - 送る側（AviUtl2 からサーバーへ送る切り出し素材・マスクのファイル名）: `native/src/bridge.cpp:3507,3639`。
- 書き出しの道具:
  - エンジンは PyAV（Python から ffmpeg の部品を使うためのライブラリ）で libx264・yuv420p・aac のみ。
  - アプリ側には同梱の ffmpeg 8.1.2（`tools/ffmpeg/bin`。`run.ps1:35-38` で PATH に追加）があり、prores_ks・utvideo・libvpx-vp9・png・apng・qtrle・ffv1 の書き出し部品があることを実行して確認した。
  - MOV／AVI なら音声を同じファイルに入れられる。PNG 連番は音声を持てない。
- AviUtl2 の最終書き出しで透過を保てるのは「連番ファイル出力」の「PNGファイル(透過付き)」。

### 3.3 利用者の目線の観察（AviUtl2 のプレビューから）

1. 利用者が求めるのは「**同じオブジェクトがそのまま透けること**」で、別素材やレイヤーが増えることではない。一番近いのは、元動画のオブジェクトにフィルタを 1 つ足す形（位置・拡大・グループ制御・分割がそのまま効く）。
2. 今の「右クリック→操作パネル→生成→仮オブジェクト」の流れは「別レイヤーに新しい素材を置く」作り。マットは「元のオブジェクトを書き換える」用途で、性格が違う。物体追尾の部分フィルタ書き換えが前例になる。
3. マット単体は白黒の絵で、それだけでは意味が無い。プレビューで「当てた結果」を見たい。しきい値や縁の調整は、AviUtl2 側のスライダーで後から詰めたくなる。
4. 抜いた後は、下のレイヤーに背景を置く使い方が中心。元動画の音声つきオブジェクトはそのまま残したい。
5. 元動画を後から分割・切り詰め・速度変更したとき、マットが追従するかが実用の分かれ目。同じオブジェクトの中で参照する方式が有利（推測）。

補足（推測）: これまでの利用者は、透過なしの生成動画にクロマキー（特定の色を透明にするフィルタ）やルミナンスキーを掛けて抜いていた可能性が高い。炎・煙のような素材を、加算やスクリーンの合成モードでなじませるエフェクト素材の使い方も想像しやすい。

---

## 4. 実用上の制約とリスク

- **16GB 級の GPU での報告は、2026-10-06 時点で見つかっていない。**
  - 同日時点で見つかった第三者の実測は、Logik Forums の投稿（2026-10-03。https://forum.logik.tv/t/ltx-2-5-alpha-gen/15024 ）だけ。3200×1900・49 フレームを bf16 で処理して約 2.5 分・VRAM 約 90GB。**この解像度は公式の上限 1920×1088 を超えている**。
  - 公式もフル HD・最大尺は H100／B200 級としている。本製品の主な利用者の GPU で、どの解像度・尺までなら実用になるかは誰も測っていない。
- **品質の第三者評価は、2026-10-06 時点で上の 1 投稿だけ。**
  - 同投稿の所見: 「クリップ全体を見渡して処理するので時間方向に安定」。「LTX outperformed both Flame's AutoMatte and Silhouette in output quality」（Flame の AutoMatte や Silhouette より出力の品質で勝った）。
  - 髪・煙・ガラスの縁、ちらつき、前景の選び間違い、複数の被写体、影についての個別の報告は見つかっていない。
- **Beta であり、公式の推奨も揺れている。** モデルカード（distilled）と専用処理（full 版）が食い違う（1.4 節）。
- **前景を指定できない。** モデルが選んだ前景が利用者の意図と違ったとき、直す手段が無い。
- **尺の上限。** 145 フレーム（24fps で約 6 秒）まで。docs は 121 以下を勧める。長い素材は分けて処理するしかない。
  - 上の投稿は「分けると見える継ぎ目が出うるので、切れ目の位置を選ぶ必要がある（strategic seams）」としている。
  - 一般には数フレーム重ねてなめらかに切り替える方法があるが、AlphaGen での実例は見つかっていない。
- **保存形式でマットが劣化する。**（一般的な技術知識）
  - H.264 の mp4（制限レンジ＝明るさを 16〜235 に収める形・色を間引く YUV 4:2:0）で保存すると、0 と 255 に届かず、縁の階調が崩れる。可逆の形式（FFV1・PNG・UtVideo）か、フルレンジで扱う必要がある。
  - 直線アルファと前乗算アルファを取り違えると、縁に色がにじむ。
  - fps・フレーム数・余白を元の動画と揃えてから当てないと、ずれる。
- **量子化の影響が未知。** 本製品の既定は 4bit の GGUF。公式は bf16 でしか検証していない（1.8 節）。2026-10-06 時点で、AlphaGen 向けの GGUF／fp8 の配布物も、GGUF と組み合わせた報告も見つかっていない。
- **周辺の道具がまだ揃っていない。** ComfyUI は既存の LTX 2.5 IC-LoRA（動画→動画）の流れに LoRA を差すだけで、公開版に専用ノードは無い（https://github.com/Lightricks/ComfyUI-LTXVideo/tree/master/example_workflows/2.5 ）。マットの後処理ノード `LTXRefineAlphaMatte`（膨らませる・縮める・ぼかす・色の混ざりを取る・プレビュー）の変更提案（PR #559）は取り込まれずに閉じられた（社内で先に入れる方針。https://github.com/Lightricks/ComfyUI-LTXVideo/pull/559 ）。
- **ライセンス。** 既存の LTX 2.5 と同じなので新しい障害は小さい（推測。1.7 節）。

---

## 5. go／no-go の判断軸と監督の推奨

### 5.1 判断軸

- 16GB 級の GPU で、実用になる解像度と尺が取れるか。
- 本製品の今の 2 段の経路のままでも、使えるマットが出るか（出れば 1 段の経路を新しく作らずに済む）。
- マットの品質が、利用者が手作業でキーイングするより良いか。
- AviUtl2 側で、元のオブジェクトに透過を当てる確実な手段があるか。

### 5.2 監督の推奨: 条件つき Go（段階ゲート）

理由:

- 追加するものが LoRA 1 本だけ。
- 本製品には LTX 2.5 で参照動画を使う IC-LoRA の経路が既にある。
- AviUtl2 側に、マットを当てる道具の候補（3.1 節）がある。
- ライセンスは既存の LTX 2.5 と同じ。

ただし:

- 16GB 級での実績が見つかっていない、Beta、そして「1 段の経路を作るか」が実装の手間の最大の分かれ目になる。
- そのため「設計の前に小さな現物確認」を挟む。

### 5.3 ゲート 0（現物確認・設計の前）

- **(0-a) LoRA の中身を見る。** 取得して safetensors のヘッダに `reference_downscale_factor` があるか、重みの名前の形式は何かを確認する。これで D の手当てが要るかが決まる。
- **(0-b) 今の経路のままで流してみる。** **(0-a) でヘッダに縮小率がある場合に限り**、コードを変えずに試せる。既存の 2.5 IC-LoRA の経路をそのまま使い（2 段のまま・LoRA は両段に掛かる・プロンプトは空白 1 文字で API を通す・参照の強さ 1.0・寸法は 128 の倍数）、AlphaGen を 1〜2 本流す。Stage-2 を通ったマットが使い物になるかを見る。**使えるなら A（1 段の経路の新設）が不要になり、実装は大幅に小さくなる。** ヘッダに縮小率が無い場合は、D の小さな手当てを先に入れてから試す。
- **(0-c) 16GB 級での VRAM と時間を測る。** Q4_K_M・1280×768 前後・121 フレーム前後。
- **GPU を使う実験は、毎回オーナーの了承を得てから行う。**

### 5.4 no-go の条件

次のいずれかに当たれば見送りを推奨する。

- 16GB 級で、実用になる解像度・尺が取れない。
- Stage-2 を通ったマットが使えず、1 段の経路が必須で、その手間に見合わない。
- マットの品質（ちらつき・前景の選び間違い・縁）が、利用者の手作業のキーイングに勝てない。

### 5.5 過去の設計判断との関係

- 物体追尾の設計上の判断「追跡結果から直接マスク動画を作る案は否決。マスクは AviUtl2 側のオブジェクトから作る」（`Docs/OBJECT_TRACKING_DESIGN.md` §13・`:432-440`）とは矛盾しない。
- あの判断は「矩形しか返さない追尾の結果から、サーバーがマスクを作る必然性が無い」という話。AlphaGen は映像の中身から前景を自動で抜く別の道具で、AviUtl2 側では作れないものを作る。
- 効果の付け方（どう当てるか・縁をどう詰めるか）は AviUtl2 側に任せる形にできる。

---

## 6. UI/UX の方向性（AviUtl2 のプレビュー画面を起点に）

### 6.1 利用者が望む流れ（監督の見立て）

タイムライン上の動画オブジェクトを右クリック → 「この動画からマットを作る」 → 生成 → **元のオブジェクトにそのまま透過が付く**。

### 6.2 案 A: マットを渡して、AviUtl2 のフィルタで当てる

- マット動画（劣化しない形式を推奨）を作り、AviUtl2 の標準のフィルタ（部分フィルタやマスクのマスク画像に動画を選び、「輝度をアルファ値として乗算」で当てる形など。**推測・未確認**）を、元のオブジェクトにエイリアスの書き換えで自動で付ける。物体追尾の前例（3.1 節）と同じやり方。
- 利点: AviUtl2 側の自由度を奪わない。縁やしきい値を後から AviUtl2 のスライダーで詰められる。元のオブジェクト（音声も含む）がそのまま残る。
- 欠点: どのフィルタのどの設定で当てるかが実機で未確認（3.1 節の候補 2）。v2.0.54 で動く組み方に限られる可能性がある。元動画を後から分割・切り詰めしたときの追従も要確認。

### 6.3 案 B: 透過つき動画を作って、別のオブジェクトとして置く

- バックエンドで元動画とマットを合成し、透過つき動画（UtVideo RGBA の AVI か ProRes 4444 の MOV。音声も入れられる）を作って、今の生成と同じく別のオブジェクトとして置く。
- 利点: AviUtl2 本体の版に依存しない（ただし L-SMASH Works が透過つきで読むことが前提で、未確認）。今の「仮オブジェクトを置く」流れにそのまま乗る。
- 欠点: 入力プラグイン（L-SMASH Works）頼み。縁を後から詰められない。元のオブジェクトと 2 つ並ぶので、利用者の観察 1（同じオブジェクトが透けてほしい）からは遠い。ファイルが大きくなる。

### 6.4 監督の見立て

- 利用者の観察（3.3 節）からは、案 A のほうが望む形に近い。
- ただし案 A も案 B も、**実機で 1 回見れば決まる未確認点**を抱えている。案 A は「マスク画像に動画ファイルを選び、『輝度をアルファ値として乗算』で当てられるか」（推測・未確認）、案 B は「L-SMASH Works が透過つきで読むか」。どちらにするかは、その確認の後で決めるのがよい。

---

## 7. 実装方針の候補と変更範囲の見込み

### 7.1 最小の形（監督の推奨）

- 既存の IC-LoRA の経路（distilled 版）を使う。
- その上に「マット作成」の専用の動き方を足す:
  - 動画→動画だけ。
  - プロンプト不要（B）。
  - 出力の寸法は参照動画と同じ（C）。
  - フレーム数は 8n+1、寸法は 128 または 32 の倍数に、余白で埋めて合わせる（C）。
- 出力を劣化しない形式にする（E）。
- 案 A か案 B で AviUtl2 に渡す。

### 7.2 ゲート 0 の結果で変わるもの

- (0-a) でヘッダに `reference_downscale_factor` があれば、D の手当ては不要。無ければ、エンジン側の判定と「前処理なしの control」の選択肢の両方に小さな手当てが要る。
- (0-b) で 2 段のままでも使えるマットが出れば、1 段の経路（A）は作らない。
- 1 段の経路を作る場合は、快適上限に新しい線の計測が要る（F）。2 段のままでも、倍率 1 の参照は今の線の根拠に含まれていないので、計測は要る見込み（推測）。

### 7.3 別の選択肢: full 版の公式専用処理を載せる

- 公式の `ltx_pipelines.alpha_gen`（full 版・1 段）を載せる道もある。
- ただし 2.5 の非蒸留版は「対応しない」裁定でクローズされている（2.3 節の補足）ため、**オーナーの再裁定が要る**。full 版の重みの取得・量子化・快適上限の計測も別に要る。

### 7.4 変更範囲

2.3 節の「変更箇所の見込み」のとおり。これに加えて、AviUtl2 側の受け渡し（案 A ならエイリアスの書き換え、案 B なら透過つき動画の書き出しと、受け取り側の拡張子 `.mp4` 決め打ちの見直し。3.2 節）が入る。

---

## 8. 未確認事項と最初の一手

### 8.1 未確認事項

AlphaGen 側:

- distilled 版と full 版のどちらが品質で勝るか（1.4 節）。
- 量子化（Q4_K_M・Q6_K）で半透明の階調がどれだけ落ちるか。
- 16GB 級での VRAM・時間・実用になる解像度と尺。
- Stage-2 を通したマット（LoRA を両段に掛けた状態）が使えるか。
- 空白 1 文字のプロンプトの影響。
- 長い素材を分けて処理したときのつなぎ目。
- LoRA のヘッダに `reference_downscale_factor` があるか。

AviUtl2 側:

- 部分フィルタやマスクのマスク画像に動画ファイルを選べるか。「輝度をアルファ値として乗算」がどのエフェクトの項目か。
- マスク画像の明るさと透明度のどちらが使われるか。
- ルミナンスキー＋クリッピングで半透明の階調が保たれるか。
- L-SMASH Works が ProRes 4444・UtVideo RGBA などを透過つきで読むか。WebM の透過に `libvpx-vp9` の指定が要るか。
- 元動画を分割・切り詰め・速度変更したとき、マットが追従するか。

### 8.2 最初の一手（ゲート 0 の計画案）

所要は数時間規模。GPU を使わない部分を先に置く。

1. **LoRA の取得とヘッダの確認（GPU 不要）。** Hugging Face で利用条件に同意して取得し、safetensors のヘッダを読む（D の判定）。
2. **AviUtl2 の実機確認（GPU 不要）。** 適当なマット動画（ffmpeg で作った白黒の動画でよい）を用意し、3.1 節の候補 1〜2 と、3.2 節の透過つき動画（UtVideo RGBA・ProRes 4444）の読み込みを画面で確かめる。案 A／B の判断材料になる。
3. **今の経路のままで 1〜2 本流す（GPU を使う。オーナーの了承を得てから）。** コードを変えずに試せるのは、手順 1 でヘッダに縮小率があった場合に限る（無ければ D の小さな手当てが先）。2 段のまま・プロンプトは空白 1 文字・参照の強さ 1.0・128 の倍数の寸法・Q4_K_M。髪や煙を含む短い素材を選ぶ。
4. **VRAM と時間の記録（GPU を使う。オーナーの了承を得てから）。** 1280×768 前後・121 フレーム前後。
5. **結果をオーナーに報告し、Go／no-go を判断してもらう。** Go なら、その時点で設計（案 A／B の選択・1 段の経路の要否）に入る。

---

## 付録 A: 出典一覧

公式（Lightricks）:

- https://x.com/ltx_io/status/2106015208221393339
- https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen
- https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen/tree/main （ファイル容量）
- https://docs.ltx.io/open-source-model/integration-tools/ic-lo-ra-adapters
- https://github.com/Lightricks/LTX-2/releases/tag/v1.4.2
- https://raw.githubusercontent.com/Lightricks/LTX-2/main/packages/ltx-pipelines/src/ltx_pipelines/alpha_gen.py
- https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x
- https://github.com/Lightricks/ComfyUI-LTXVideo/pull/559
- https://github.com/Lightricks/ComfyUI-LTXVideo/tree/master/example_workflows/2.5

コミュニティ・周辺:

- https://forum.logik.tv/t/ltx-2-5-alpha-gen/15024
- https://www.youtube.com/watch?v=mzn9R9RJIHs （本文未取得）
- https://www.youtube.com/watch?v=dKbT0x9Se-s （本文未取得）
- https://www.youtube.com/watch?v=J-3rK-8ths4 （本文未取得）
- https://www.pixelsham.com/2026/10/05/ltx-2-5-alpha-gen-designed-to-generate-mattes-for-videos/
- https://github.com/Mr-Ojii/L-SMASH-Works/tree/master/AviUtl2

AviUtl2 の公式サイト（v2.1.5／v2.1.8／v2.1.12・最新 SDK の記述。手元では未確認）:

- https://spring-fragrance.mints.ne.jp/aviutl/

競合（参考）:

- Wan-Alpha（RGB と透明度を同時に作る、本物の透過動画生成。AlphaGen とは別物）: https://huggingface.co/htdong/Wan-Alpha_ComfyUI
- AlphaGen と同じ「既存の動画から抜き出す」系統:
  - VideoMaMa: https://arxiv.org/abs/2601.14255
  - MatAnyone 2: https://arxiv.org/abs/2512.11782
  - Silhouette（市販の合成用ソフト）

本製品のコード・文書（本文で行番号を添えたもの。起点は「本書の位置づけ」を参照）:

- `api/models.py`・`api/generate.py`・`api/jobs.py`
- `services/engines/ltx25/adapter.py`・`services/lora_registry.py`・`services/video_io.py`
- `engine25/pipeline25.py`・`engine25/reference25.py`・`engine/gguf/ic_lora_common.py`
- `scripts/manifests/20-ltx25.json`・`config.py`・`run.ps1`
- エンジン用仮想環境の公式部品 `ltx_pipelines/utils/media_io/encode.py`
- `webui/src/modes/single/useGenerationForm.ts`
- `native/src/alias_util.cpp`・`native/src/bridge.cpp`・`native/src/bridge_core.cpp`
- `Docs/LTX25_RESEARCH_NOTES.md`・`Docs/COMFORT_LIMIT_TABLE.md`・`Docs/PENDING_TASKS_CLOSED.md`・`Docs/OBJECT_TRACKING_DESIGN.md`

AviUtl2 側（同梱文書）:

- `aviutl2.txt`（本体の説明と更新履歴）・`lua.txt`（スクリプトの説明。行番号は v2.0.54 同梱版）・`script.anm2`（標準スクリプト）・`Default.aul2`（本体の説明文）

---

## 付録 B: 透過つき形式の比較と ffmpeg の例

### B.1 形式の比較

- **PNG 連番**
  - 長所: 劣化しない。AviUtl2 本体だけで読める。
  - 短所: 音声を持てない。ファイル数が多い。本製品の今の経路では置けない（画像オブジェクト用のエイリアスが無い）。
  - AviUtl2: 本体で読める（連番の扱いは未確認）。
- **UtVideo RGBA の AVI**
  - 長所: 劣化しない。音声を同じファイルに入れられる。書き出しが速い。
  - 短所: ファイルが大きい。
  - AviUtl2: L-SMASH Works 経由で透過つきで読める見込み（未確認）。
- **ProRes 4444 の MOV**
  - 長所: 映像業界の標準で、他のソフトにも渡しやすい。透明度を 16bit で持てる。音声を入れられる。
  - 短所: わずかに劣化する（見た目にはほぼ分からない）。ファイルが大きい。
  - AviUtl2: L-SMASH Works 経由で読める見込み（掛け合わせ済みの形で渡る。未確認）。
- **WebM（VP9 の透過つき）**
  - 長所: ファイルが小さい。
  - 短所: 劣化する。標準の読み取り部品が透明度を捨てる。
  - AviUtl2: L-SMASH Works で `libvpx-vp9` の指定が要りそう（未確認）。
- **FFV1**
  - 長所: 劣化しない。マット単体の保存に向く。
  - 短所: 対応するソフトが限られる。
  - AviUtl2: L-SMASH Works 経由で読める見込み（未確認）。

### B.2 ffmpeg の例（マット＋元動画→透過つき）

```
# UtVideo RGBA AVI（可逆・音声つき）
ffmpeg -i rgb.mp4 -i matte.mp4 -filter_complex "[1:v]format=gray[a];[0:v][a]alphamerge,format=gbrap" -map 0:a? -c:v utvideo -c:a pcm_s16le out_rgba.avi
# ProRes 4444（アルファつき）
ffmpeg -i rgb.mp4 -i matte.mp4 -filter_complex "[1:v]format=gray[a];[0:v][a]alphamerge" -c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le -alpha_bits 16 -map 0:a? -c:a pcm_s16le out.mov
# PNG 連番（RGBA）
ffmpeg -i rgb.mp4 -i matte.mp4 -filter_complex "[1:v]format=gray[a];[0:v][a]alphamerge" -pix_fmt rgba frames_%05d.png
```

注意: マット側が H.264 の mp4（制限レンジ）だと、白が 255 に届かず完全な不透明にならない。マットは劣化しない形式で受け取るか、フルレンジに直してから合成する（4 節）。

---

## 付録 C: 用語

- **LoRA** — 本体の重みに小さな差分を足して振る舞いを変える追加部品。
- **IC-LoRA** — 参照動画を条件として受け取る種類の LoRA。AlphaGen はこれ。
- **distilled（蒸留版）／full（非蒸留版）** — 蒸留版は少ない手順で生成できるよう学習し直した版。本製品の LTX 2.5 は蒸留版を使う。full 版は元の版。
- **Stage-1／Stage-2** — 本製品の生成は 2 段。Stage-1 で半分の解像度で描き、Stage-2 で 2 倍に引き伸ばして仕上げる。
- **マット** — 透明度を白黒で表した動画。白＝見える、黒＝透明、灰＝半透明。
- **直線アルファ／前乗算アルファ** — 直線アルファは、色と透明度を別々に持つ形。前乗算アルファは、色にあらかじめ透明度を掛けておく形。取り違えると縁に色がにじむ。
- **VRAM** — GPU に載っているメモリ。生成できる大きさと尺はこれで決まる。
- **GGUF** — 重みを小さく詰めた配布形式。Q4_K_M（4bit）・Q6_K（6bit）は詰め方の種類。
- **fp8・bf16** — 重みを 8bit／16bit の小数で持つ形式。bf16 は詰めていない元の状態。
- **快適上限** — 本製品が「この大きさ・尺までなら VRAM から溢れず快適に動く」として画面に示す目安。正本は `scripts/manifests/*.json` の `comfort` と `Docs/COMFORT_LIMIT_TABLE.md`。
- **仮想環境（venv）** — アプリ専用に部品一式を入れておく Python の作業場所。エンジンの公式部品はここに入っている。
- **PyAV** — Python から ffmpeg の部品を使うためのライブラリ。
- **libx264・yuv420p・crf** — libx264 は H.264 形式の書き出し部品。yuv420p は色を間引いて保存する形式（4:2:0）。crf は画質の設定で、数字が小さいほど劣化が少ない（0 で劣化なし）。
- **FFV1・UtVideo** — 劣化しない動画形式。UtVideo は透明度も持てる。
- **ProRes 4444（prores_ks）** — 映像業界で使われる高画質の形式。透明度を持てる。prores_ks はその書き出し部品の名前。
- **qtrle・APNG** — 透明度を持てる劣化しない形式。qtrle は MOV 用の古い形式、APNG は動く PNG。
- **SDK** — AviUtl2 のプラグインを作るための公式の開発キット。
- **PR／merge** — PR（プルリクエスト）は GitHub で変更を提案すること。merge はその変更を本流に取り込むこと。
- **L-SMASH Works** — AviUtl2 に mp4・MOV などを読ませる入力プラグイン。
- **ルミナンスキー** — 明るさをもとに透明にする AviUtl2 のフィルタ。
- **クリッピングオブジェクト** — 下のオブジェクトを自分の形で切り抜く AviUtl2 の設定。
