# IC-LoRA × A2V 併用解禁 ワークオーダー

- 作成: 2026-07-20（オーナー判断により次セッションの課題として起票。着手は2026-07-20夜〜21予定）
- 正本: 本書。着手後の状態遷移は本書の冒頭に歴史記録ブロックを追加していく運用とする（他の `*_WORKORDER.md` と同じ体裁）。
- 併読: [`DEVLOG.md`](DEVLOG.md) §42.8（相互排他の経緯調査の結論）／[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-25（実装完了記録。起票当時は旧§1-1）・同書§3-18（2026-07-19判定の訂正注記）／[`API_REFERENCE.md`](API_REFERENCE.md)（`GenerateChainRequest`の契約）／バックエンド [`../../Nz-LTX23-backend/Docs/VERIFICATION_LOG.md`](../../Nz-LTX23-backend/Docs/VERIFICATION_LOG.md) §34.5・§34.7（実機ゲートの前例）

> **歴史記録（2026-07-21）**: §3.3で「次セッションで判断する」としていたi18n（排他ノートの扱い・ソフト警告の文言）と、モードバッジの併用時表示をオーナーが確定した。確定内容は§2の3〜5項として追記した。実装は未着手（プランモードでの実装計画立案とオーナー承認を待って着手する）。
> **歴史記録（2026-07-21・実装完了）**: 本ワークオーダーを増分I1/I2として**実装完了**した。要点: `useGenerationForm.ts`の三方排他を解禁（`isA2v = audioReady`。参照動画とは独立に判定）／`buildA2vRequest`が`mergedLoras`（制御LoRA＋STYLEタグを`combineLoras`で統合）を使い、N3ゲート（`reference_video_id !== null && loras.length > 0`）付きで参照フィールド（`reference_video_id`・強度2種）を1クリップchainペイロードへ同梱／ペイロード層 [`buildA2vChainPayload.ts`](../webui/src/modes/batch/buildA2vChainPayload.ts) は無改修（既存の`use_adapter`分岐がそのまま効く）。UIは両picker（音声・参照動画）のdisabled排他を撤去し、併用ノート・ソフト警告（生成尺>参照尺）・モードバッジ「IC-LoRA」「A2V」併記を三項式から**独立判定2枚**へ変更。制御LoRAのみ（STYLEタグ無し）＋音声＋参照動画の通しE2E（送信→completed→ホワイトリスト応答）を含めテスト全緑。**オーナー実機ゲートは2026-07-21に全4項目合格した**（結果は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-27）。

> **目的**: Create画面で「参照動画（IC-LoRA）」と「音声（A2V）」を**同時に添付できるように**する。現在フロントエンドは両者を相互排他にしているが、これはフロント専用の設計判断であってバックエンド制約由来ではない。バックエンドは併用対応済み・GPU実機ゲート合格済みであり、フロントの排他を解除して1クリップchain送信経路へ合流させるのが本ワークオーダーの内容である。**本セッションはドキュメント整備のみ。実装は次セッションで行う。**

---

## 1. 背景と経緯

### 1.1 ユースケース（オーナー提示）

「IC-LoRA（In-Context LoRA。参照した動画の動きや輪郭をなぞって生成する仕組み）でポーズを制御しているキャラクターに、A2V（音声から動画を生成する機能）でリップシンクさせる」といった、ポーズ制御とリップシンクの同時適用。参照動画で動き・輪郭をガイドしながら、音声から口の動きを生成したい、という要求である。

### 1.2 バックエンドは併用対応済み（確定事実）

- バックエンド側コミット **3f29776**（2026-07-11、A2V＋LoRA併用解禁）→ **7c50c32**（2026-07-12、A2V＋参照動画つきIC-LoRAのチェーン対応、`clips=1`限定）で、併用が正規に解禁されている。
- 同日、GPU実機ゲート合格済み（pose-control×3＋canny-control×1、1280×768／201フレーム、ピークVRAM 9.5GB）でオーナー受容済み（バックエンド [`../../Nz-LTX23-backend/Docs/VERIFICATION_LOG.md`](../../Nz-LTX23-backend/Docs/VERIFICATION_LOG.md) §34.7）。

### 1.3 スキーマ上の許可（確定事実）

バックエンド `api/models.py` の `GenerateChainRequest` バリデータの排他規則は次のとおり。

- `source_audio` × `source_video` は**排他**。
- `reference_video_id` × `source_video` は**排他**。
- `source_audio` と `reference_video_id` は、**どちらも `clips=1` 限定という同一条件が課されるのみで、相互排他ではない**。同時指定は正規に許可されている。

つまりバックエンドのスキーマ上、「音声＋参照動画」の同時送信は最初から通る（両者に共通するのは「1クリップchainであること」だけ）。

### 1.4 Gradio原典の送信経路（確定事実・従うべき方式）

Gradio原典（バックエンド `gradio_ui/handlers.py` の `generate`、L406-644付近）は、`src_audio` 添付時に `use_adapter`（参照動画IC-LoRA）を**同じ1クリップchainペイロードへ合流**させている（`build_a2v_chain_payload` に `reference_video_id` / `control_adherence` / `reference_strength` を配線）。**併用時の送信経路はこの方式（1クリップchain）に従うこと。** 別モードを新設するのではなく、既存のA2V用1クリップchainペイロードに参照動画フィールドを載せるのが正道である。

### 1.5 フロントの排他は「フロント専用の設計判断」だった（確定事実）

フロントエンドの相互排他は、以下の経緯で入ったものであり、**バックエンド制約由来ではない**。

- **45fc836**（2026-07-15、単発A2VのCreate移設）で、「曖昧な第4モードを作らない」というフロント専用の設計判断として付随的に導入された。
- **80e99bd**（2026-07-17）で双方向化（音声→参照動画／参照動画→音声のどちらの向きでも他方を無効化）された。

### 1.6 注意: バックエンドに残る古いコメント（誤解の元）

バックエンド `gradio_ui/ui.py` L560-567付近に、併用解禁**前**（3f29776時点）の古いコメント（「reference-video CONTROL adapter cannot combine」の趣旨）が放置されている。これを読むと「併用不可」と誤解するが、**7c50c32で制約自体が撤廃済み**である。このコメントは実装の現状を反映していない残骸なので、判断の根拠にしないこと。

---

## 2. 確定要件（オーナー承認済み）

1. **DURATION（生成フレーム数）は A2V 側の計算値（wav長→Frames自動調整）を優先する。**
2. **生成尺が参照動画の尺（`referenceVideoDurationSec`、`fs.probeMediaInfo` で取得済みの値）より長い場合はソフト警告を表示する（送信はブロックしない）。**
3. **併用時の注意ノートを表示する（2026-07-21確定。排他ノート2キーは削除せず、文言変更で残す）。** 参照動画と音声の両方が入力されているとき、参照動画の消し忘れなどを事前に気づかせるため、次のノートを表示する。日本語確定文言: 「参照動画と音声の両方が入力されています。IC-LoRAで制御されたa2vが生成されます。」（英語文言は実装時に用意する。案: "Both a reference video and an audio file are attached. The a2v generation will be guided by the IC-LoRA reference video."）
4. **ソフト警告の確定文言（2026-07-21確定）。** 日本語: 「生成する尺（X.X秒）が参照動画（Y.Y秒）よりも長いため、動画後半はIC-LoRA制御なしで生成されます。」（英語文言は実装時に用意する。案: "The generated duration (X.Xs) is longer than the reference video (Y.Ys), so the latter part of the video will be generated without IC-LoRA control."）
5. **モードバッジは併用時「IC-LoRA」と「A2V」の両方を並べて表示する（2026-07-21確定）。**

### 2.1 この要件の根拠（担当エージェントが必ず理解すべき非対称性）

A2V音声とIC-LoRA参照動画では、「尺が生成フレーム数に対して不足／超過したとき」のバックエンド挙動が**非対称**である。この非対称性が「A2V優先＋不足はソフト警告」という設計の根拠になっている。

| | 不足時（素材が生成尺より短い） | 超過時（素材が生成尺より長い） |
|---|---|---|
| **A2V音声** | 422でハード拒否（フロントの `audioLengthPrecheck` でも事前ブロック） | 切り詰め（動画尺が権威、音声は truncate） |
| **IC-LoRA参照動画** | **エラーなし。参照トークンが短い分だけサイレントに制御が途切れる（品質劣化のみ）** | 先頭から `numFrames` 分に切り詰め |

- IC-LoRA参照動画には**長さのハード制約が一切ない**（バックエンド `api/models.py` に長さバリデーションが存在しない。`ltx_pipelines` の `decode_video_from_file` は `frame_cap` で切り詰めるのみ・不足時はパディングもループもエラーも無し。`VideoConditionByReferenceLatent` は参照潜在を独立トークン列として concat する方式で、フレーム数の一致を要求しない）。
- ゆえに「短いほうの尺へ強制クランプする」処理は**不要**であり、A2V側の計算値を優先し、参照動画が不足するケースは（ハード拒否ではなく）ソフト警告で知らせるのが、バックエンドの実挙動と整合する。

---

## 3. 実装箇所マップ（調査済み、行番号は2026-07-20時点）

### 3.1 変更不要（既に併用をサポート済み）

- **ペイロード層 `webui/src/modes/batch/buildA2vChainPayload.ts`**: `useAdapter` / `referenceVideoId` / `controlAdherence` / `referenceStrength` を**既に完全サポート**している（Python原典 `build_a2v_chain_payload` の忠実移植）。**変更不要**。併用時はこの関数へ参照動画フィールドを渡すだけでよい。

### 3.2 排他を作り込んでいる箇所（解除対象）

- **`webui/src/modes/create/useGenerationForm.ts`**
  - L391-398: `referenceActive` / `isA2v` / `icLoraActive` の**排他の核心**。
  - L829-854: `buildA2vRequest` が参照動画フィールドを落としている（併用時はここで `referenceVideoId` 等を載せる必要がある）。
  - L206-273: 関連するJSDoc群（排他前提の説明。併用解禁に合わせて更新）。
  - L767-773: `isValid`。**`isA2v` と `isReferenceValid` の両立時のバリデーション合成設計が必要**（現在はどちらか一方を前提にしているため、両立を許す合成を新規に設計する）。
- **`webui/src/modes/create/GenerationForm.tsx`**
  - `blockedByAudio`（L299付近）／`blockedByReference`（L480付近）と、それらに連なる `disabled` 連鎖・排他ノート表示。
- **`webui/src/modes/create/CreateScreen.tsx`**
  - L159 `isICLora`（`initialIntent?.intent === "reference-video"` から導出）が右クリックintentによる排他固定の起点。これが `initial.isICLora` として `useGenerationForm` へ渡り、L311 `isICLoraIntent` に固定される（この値が §3.2 冒頭の `referenceActive` を常時 true にして排他を作る）。併用可否に合わせて、intentからの固定と `referenceActive` の合流点を見直す。

### 3.3 i18n（排他ノートとソフト警告）

- 排他ノートのキー: `blockedByAudioNote` / `exclusiveWithReferenceNote`（en L457／L478、ja L1102／L1118付近）。
- **排他解除後の扱いは2026-07-21に確定した（§2の3項）**: 削除ではなく、併用時の注意ノートへ文言を変更して残す。ソフト警告の文言も§2の4項で確定済み。

### 3.4 排他を固定しているテスト3件（併用可へ書き換え対象）

- `webui/src/modes/create/useGenerationForm.test.ts` L654-673
- `webui/src/modes/create/GenerationForm.sourceAudio.test.tsx` L165-173
- `webui/src/modes/create/GenerationForm.referenceVideo.test.tsx` L153-162

これら3件は現在「一方を選ぶと他方が排他で無効になる」ことを固定している。併用解禁に伴い、これらは「両方添付できる／ソフト警告が出る」挙動を検証するテストへ書き換える。

---

## 4. 検証条件

- **実機ゲート**: real バックエンド（実GPU接続の本物の生成エンジン）での**併用生成（音声＋参照動画＋control系LoRA）**まで到達すること。
- **mockでのE2E**: バックエンド [`../../Nz-LTX23-backend/Docs/VERIFICATION_LOG.md`](../../Nz-LTX23-backend/Docs/VERIFICATION_LOG.md) §34.5の前例（`clips=1` ＋ control ＋ `reference_video_id` ＋ `source_audio` → 202 → completed）に準拠する。
- 併用時のDURATION決定（A2V優先）とソフト警告（尺超過時）の挙動を、mockのユニット/コンポーネントテストで固定する。

---

## 5. 着手手順の推奨

1. まず本書§1をよく読み、「排他はフロント専用判断」「バックエンドは併用対応済み」「尺の非対称性」の3点を理解する。特に§2.1の非対称テーブルは、なぜ「A2V優先＋不足はソフト警告」なのかの根幹なので必ず腹落ちさせること。
2. `useGenerationForm.ts` の排他核心（L391-398）とバリデーション合成（L767-773）の設計から着手する。ペイロード層（`buildA2vChainPayload.ts`）は変更不要なので、`buildA2vRequest`（L829-854）が参照動画フィールドを落とさないようにするのが送信側の主要変更点。
3. i18n（排他ノートの扱い＋ソフト警告の新キー）はオーナーへ文言を確認してから確定する。
4. テスト3件を「併用可＋ソフト警告」へ書き換え、mock E2E（§34.5前例準拠）を通す。
5. 最終ゲートとして real バックエンドでの併用生成を実施する。
