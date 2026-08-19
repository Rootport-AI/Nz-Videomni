# バッチi2v-long 作業指示書（旧`PENDING_TASKS.md` §1-7 →現`PENDING_TASKS_CLOSED.md` §3-57 の正本）

起票: 2026-07-30。**同日、chain版として全面改訂のうえ実装完了。** 本書は完成した仕様の正本として残す（以後の改修はここを読んでから行う）。

## 0. 状態（2026-07-31）

> **全項目合格・クローズ（2026-07-31）。** クローズ記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) **§3-57**、実機ゲートの記録は[`REAL_BACKEND_CHECKLIST.md`](REAL_BACKEND_CHECKLIST.md) §4.13（全項目チェック済み）、実装記録は[`DEVLOG.md`](DEVLOG.md) §54（本体・§54.7の行ごとプロンプト再設計・§54.8のSTATUS表示修正・§54.9のバッチ相互排他バグ2件）。

**残作業は無い。** 実機ゲート（[`REAL_BACKEND_CHECKLIST.md`](REAL_BACKEND_CHECKLIST.md) §4.13）は次のとおり全項目が合格した:

1. **基本フロー5点**（2026-07-30）— フォルダ指定・出力フォルダ自動作成・Scan・バッチ生成・Status遷移。
2. **行ごとプロンプト**（2026-07-31）— 出来上がった動画が行ごとに違うプロンプトの内容になる（§54.7の再設計分）。
3. **初回行STATUS表示の修正確認**（2026-07-31・§54.8）。
4. **Stopの中断挙動**（2026-07-31）— 現在生成中の1枚は完走保存され、再Startで中断直後の画像から再開する。
5. **バッチA2Vとの相互ロック両方向**（2026-07-31）— §54.9の実バグ2件の修正後、双方向とも案内文つきでStartが無効化される。
6. **バッチ完走**（2026-07-31）。

派生して残った将来課題は2件——[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-47（バッチA2Vランナーの孤児化とrunLock滞留。本書§6.5）と§3-48（JobLedgerの50件級描画。実運用トリガー待ちへ分離）。

## 改訂履歴

- **初版（2026-07-30午前）は無効**。「大量の画像を1枚ずつi2v生成する」という要件を、エージェント側が**単発生成（`POST /generate`）を1枚ずつ回す機能**と解釈して書いていたが、これは**オーナー意図の誤解**だった。オーナーが求めていたのは「画像1枚ごとに**長尺動画**を作る」機能——つまり1枚の画像から**Clip Chain（`POST /generate/chain`）を1本まるごと**生成し、それを画像の枚数ぶん直列で繰り返すものである。初版の§2「土台は単発`/generate`一択。chainは使わない」という結論は、この誤解の上に立っていたため無効になった（調査自体の事実誤りではない点は§2の訂正注記を参照）。
- **改訂版（2026-07-30、本書）**: 機能名を「簡易バッチi2v」から**「バッチi2v-long」**へ改め、土台を`POST /generate/chain`に差し替えた。設置場所もCreate画面からChain画面へ移した（借用する設定がChain画面のものになるため）。**同日、本書に沿って実装完了**。実装記録は[`DEVLOG.md`](DEVLOG.md) §54、実機ゲートの手順は[`REAL_BACKEND_CHECKLIST.md`](REAL_BACKEND_CHECKLIST.md) §4.13。
- **2026-07-30（実機フィードバック反映）**: オーナーの実機確認で基本フロー（フォルダ指定／出力フォルダ自動作成／Scan／バッチ生成／Status遷移）は**合格**。そのうえで次の4点を改修した。①**プロンプトを画像（行）ごとの指定へ再設計し、バッチ全体プロンプトの入力欄は撤去**——「全画像に同じ文を足す」欄は上の共通プロンプトを書き換えるのと同じで無意味であり、実際に欲しいのは画像ごとの文だったため（バッチA2Vの行プロンプトと同じ作法に揃えた。add/replaceラジオはバッチ全体で1個のまま）。②アコーディオンのタイトルを**「バッチi2v-long（プロトタイプ）」／"Batch i2v-long (prototype)"** へ変更。③パネル冒頭の長い説明文（`notice`・`chainSettingsNote`）を**削除**（内容は本書に記載済み。設定の実効値が見える`chainSummary`の1行だけ残す）。④共通プロンプト欄の下にあった「つなげるモードでも画風LoRAタグは適用されます」の注記（`promptBar.chainNote`）を削除。

## 1. 目的とオーナー確定方針

大量の画像が入ったディレクトリのパスを指定すると、**画像1枚ごとにClip Chain（長尺動画）が直列で生成される**バッチ機能を作る。

オーナー確定事項（2026-07-30）:

- **必要最低限のプロトタイプ**。複雑なことはできなくてよい（単純でも十分便利）。
- 解像度・尺・クリップ構成などの基本設定は操作パネルの設定を**無言借用**（バッチA2Vと同じ考え方）。借用元は**Chain画面**である（バッチA2VはCreate画面から借用しているが、長尺生成の設定はChain画面にしかない）。
- プロンプトのadd/replace程度は付けてよい。
- **clip数・各clipのフレーム数の指定は、Chain画面の設定をそのまま借りることで最初から満たされる**（初版では「将来の拡張」に位置づけていたが、chain版では土台の性質として最初から備わる）。
- **バッチA2Vと同居させない**（独立機能）。ただしUI/UXの見た目・操作感はバッチA2Vに寄せ、「**似たUIで違うことができる**」体験にする。

## 2. APIの結論（chain採用・最重要）

**土台は`POST /generate/chain`。画像1枚＝chain 1本。**

根拠（`Nz-Videomni/api/models.py`の`GenerateChainRequest.validate_chain_constraints`を実読して確認）:

1. **`source_video`・`source_audio`・`reference_video_id`のいずれも無い「素のchain」は、`clips`が2本以上あれば通る。** バリデータの該当節は「3つとも`None`**かつ**`len(self.clips) < 2`」のときだけ`chain requires at least 2 clips`を送出する条件になっている。長尺を作るバッチは当然2本以上のクリップを組むため、この制約はそもそも当たらない。
2. **clip 0だけが`conditioning_images`を持てる。** バリデータは`i > 0 and clip.conditioning_images`で`only clip 0 may carry conditioning_images`を送出する（1クリップあたり最大5枚）。本機能は各行の画像をclip 0の`frame_idx: 0`・`strength: 1.0`の1枚として載せるので、この制約とも整合する。
3. **`source_video`との排他に注意**: `source_video`があるとclip 0の`conditioning_images`は`source_video is mutually exclusive with clips[0].conditioning_images`で拒否される（V2Vでは凍結したソース末尾がclip 0の頭を占めるため）。したがってChain画面がV2Vモード（ソース動画添付済み）のあいだは、バッチを開始させてはならない（§6のガード`sourceVideoAttached`）。

**初版§2の訂正**: 初版が書いていた「`clips`が1個だと422になる」という調査事実そのものは正しい。誤っていたのは**そこから「だからchainは使えない」と結論した推論**である。1クリップchainが通らないことは、2本以上のクリップを組む長尺バッチにとって何の障害でもない。したがって「1-clip chain許可の凍結例外」（初版§7-4）も不要になった。

**バックエンドは無改修**。凍結方針に触れる変更は一切ない。

## 3. 設計の背骨: Chain画面の`buildRequest()`をテンプレートにする

**Chain画面の`useChainForm`が既に持っている`buildRequest()`の出力（1本分の完全な`GenerateChainRequest`）をテンプレートとして受け取り、行ごとに変わる「clip 0の画像1点」だけを差し替える。**

これは「A2V側の`buildA2vChainPayload`のように、ペイロードをもう1本自前で組み立てる」方式を**意図的に採らなかった**結果である。chainには（A2Vの1クリップ固定と違って）ユーザーが選んだクリップ本数・クリップごとのフレーム数・オーバーラップ設定があり、2本目のビルダーを手で保守すると恒久的なドリフト要因になる。Chain画面自身の出力をテンプレートにすれば、**将来Chainに増えるフィールド（crop・NAG/VSF・chunked_upsample・LoRA…）は何もせずに透過する**。

テンプレート化の規則（`modes/batch-i2v-long/buildI2vLongPayload.ts`の`buildI2vLongTemplate`）:

1. **`prompt`（chain全体のプロンプト）はChain画面の値のまま**（2026-07-30の再設計。プロンプトの合成は行ごとに変わるため、テンプレートでは一切触らない）。
2. **`source_video`キーを落とす**（多重防御。ガードで既に止めているが、テンプレートにソースが残ると全行が同じ動画のV2V継続になり、行の画像が黙って無視される）。
3. **`clips[0].conditioning_images`を空にする**（Chain画面に置かれていた冒頭キーフレームは、行ごとの画像で置き換わる）。
4. **`clips[1..]`は完全に無改変で通す**（クリップ別プロンプト・`num_frames`とも）。
5. **その他の全フィールドはそのまま透過**（width/height/crop/fps/seed/overlap/loras/chunked_upsample/NAG/VSF…）。

行ごとの差し替え（同ファイルの`buildRowPayload`）は、テンプレートのコピーに対して**2点だけ**を変える。①clip 0へ`conditioning_images: [{ image_id, frame_idx: 0, strength: 1.0 }]`を載せる。②chain全体の`prompt`を`composeBatchPrompt(テンプレートのprompt, その行のprompt, mode)`へ差し替える（合成規則はバッチA2Vと同一。空白のみの行プロンプトは無視、`replace`は置き換え、`add`は`` `${共通} ${行}` ``をtrim）。**すべてコピーオンライトで、テンプレートも共有部分（`loras`・`crop_output`・clips 1..n）も一切変更しない**——だから同じテンプレートを全行で使い回せる。

**プロンプト合成は「案A（驚き最小）」**: 行のプロンプトはchain全体のプロンプトにだけ適用し、**クリップ個別のプロンプト上書きには触れない**。Chain画面でクリップ別プロンプトを打ったユーザーには、それがそのまま尊重されて見える（`replace`を全クリップへ及ばせると、画面に見えているテキストが無言で消える）。案B（全クリップへ合成）へ移す場合の切替点は`clips.map`内の3行だけであり、コメントで明示してある。

**テンプレートは値でスナップショットする**: `start()`の時点の`GenerateChainRequest`オブジェクトを走行中ずっと保持するため、バッチ実行中にChain画面を編集しても走っているバッチには届かない（メモが再計算されて別オブジェクトになるだけ）。

## 4. 最小スコープ（v1で作るもの）

1. 画像ディレクトリ指定（テキスト入力＋📁ピッカー、手打ちはフォーカス外し＝onBlurで確定）と出力ディレクトリ（自動導出は`{画像フォルダ名}_i2vlong_out`。自分で選んだ／打ち込んだ時点で自動追従は止まり、以後戻らない）
2. Scan → `fs.listFiles`で列挙 → **ファイル名昇順**で行生成（A2Vのmtime昇順は踏襲しない。連番画像が自然な想定のため。数値を意識した自然順で`img2.png`が`img10.png`より前に来る。大文字小文字・アクセントは無視し、同順のときはコード単位で決定的に並べる）
3. 対応拡張子は**バックエンドの許可リストに一致**（`GET /config`の`upload.allowed_image_extensions`を使う。配信が壊れていた場合の保険として`.png/.jpg/.jpeg/.webp`のみのフォールバックを持つが、これは第2の正本ではない）。`*.tmp`は許可リストに関わらず常に除外する。**サイズ超過の画像はスキャン時点で`Failed`＋説明文にする**（実行の何十分後に`POST /upload/image`が4xxを返すのを待たない）
4. 台帳テーブル（列は`#`／画像ファイル名／**プロンプト（行ごとに編集可能な入力欄＋📝で共通プロンプトを流し込むボタン）**／stat（絵文字付き）／output＋🔁ボタン列。**失敗理由はstatセルのツールチップ**にして、長いサーバーメッセージでも1行1行を保つ。🔁は`Done`/`Failed`の行だけ有効で、押すと`Waiting`へ戻る。サムネイルは省略＝最低限方針）
5. プロンプトadd/replaceラジオ1個（**モードはバッチ全体で1つ・文面は行ごと**。バッチA2Vとまったく同じ配置。2026-07-30の実機フィードバックにより、旧「バッチ全体プロンプトのtextarea」は撤去した）。行プロンプトは**再スキャンで消える**（ステートレス設計のため。行の入力欄は走行中・スキャン中は無効）
6. Chain設定の無言借用（§3のテンプレート方式。この節自身はサイズ・尺・seedの欄を一切持たない）
7. 開始/中止・進捗（現在行）・失敗行スキップ継続・未完了行のみの再実行
8. 1行＝1画像＝`upload_image`→`POST /generate/chain`→ポーリング→`backend.downloadVideo`、の直列実行

件数上限は設けない（A2V同様100〜200行想定の素朴なtable）。**状態はメモリのみ**（CSVマニフェストは作らない。バッチA2Vが2026-07-18にステートレス化したのと同じ設計）。したがって**ウィンドウを閉じると行状態は消える**。

**注意文について（2026-07-30変更）**: 当初はこの「ウィンドウを閉じるな」「中止しても現在の1枚は最後まで作られる」をパネル冒頭の説明文で出していたが、オーナー判断で**画面からは削除**した（文章が長すぎる）。これらの挙動の正本は本書であり、UIに残すのは`chainSummary`（クリップ本数・フレーム数・1枚あたり秒数）の1行だけとする。

## 5. ファイル構成（実装済みの実体）

新ディレクトリ `webui/src/modes/batch-i2v-long/` に以下を置いた（各`.ts`/`.tsx`に対応するテストが1本ずつ並ぶ）。

| ファイル | 役割 |
|---|---|
| `chainSnapshot.ts` | Chainフォームから読む最小面の**構造型** `ChainSnapshotSource`（`buildRequest`／`mode`／`clips`／`isValid`／`validityReasons`／`outputFrames`／`outputSeconds`／`minClips`／`minFramesForOverlap`）。`modes/chain/`への`import`をこの型1本で回避しており、Chain画面側は**完全に無改変でこの形を満たす**。`UseChainFormResult`が構造的に代入可能であることは`buildI2vLongPayload.test.ts`のコンパイル時アサーションで固定してあり、将来Chainがフィールド名を変えたら実行時ではなく型検査で落ちる |
| `imageRows.ts` | 行モデル（`I2vLongRow`＝`queue`/`image`/**`prompt`**/`stat`/`output`/`error`、`I2vLongStat`＝`Waiting`/`Generating`/`Done`/`Failed`。A2Vにある`Skip`は無い）＋`scanImagesToRows`（拡張子フィルタ→ファイル名昇順→`queue`採番、`prompt`は常に空、サイズ超過の事前`Failed`化）＋`normalizeImageExtensions`＋`deriveI2vLongOutDir`。**I/Oは一切しない純関数群** |
| `buildI2vLongPayload.ts` | `composeBatchPrompt`（共通プロンプト×行プロンプトのadd/replace合成）＋`buildI2vLongTemplate`（§3の1〜5）＋`buildRowPayload`（clip 0への画像刻印＋行プロンプトの合成）。すべて純粋・コピーオンライト |
| `batchI2vLongRunner.ts` | 実行エンジン`BatchI2vLongRunner`。`modes/batch/batchRunner.ts`からの意図的な複製（A2Vは本改修の編集対象外であり、行モデルとペイロード組み立てが根本的に違うため一本化しない）。1行ずつtry/catch・409（JOB_BUSY）は**最大3回試行**（あいだに3秒バックオフ＝計約6秒で諦め、その行だけ`Failed`）・`GET /jobs/{id}`は1秒間隔で**期限なし**ポーリング（24クリップのchainは数十分走って正常）・アップロードした`image_id`は解決済みパスをキーに走行中キャッシュ |
| `runtime.ts` | **モジュールレベルのシングルトン**。ランナー実体・最新の行・入出力フォルダ・購読機構を持つ（§6の「なぜシングルトンか」参照） |
| `useBatchI2vLongRunner.ts` | `runtime.ts`を`useSyncExternalStore`でReactへ橋渡しする薄いフック |
| `useBatchI2vLongForm.ts` | フォーム状態一式（フォルダ・スキャン・行プロンプトの編集（`setRowPromptLocal`／`copyChainPromptToRow`）・add/replaceモード・テンプレート・ガード・実行制御） |
| `BatchI2vLongSection.tsx` | 表示層（既定は折りたたみ）。`<details>`の外枠・フォルダ行・台帳テーブルの見た目は`modes/batch/BatchSection.css`を**そのまま流用**（このCSSがA2Vから取っている唯一のもの） |
| `BatchI2vLongTable.tsx` | 台帳テーブル（行プロンプト入力欄＋📝・絵文字stat・🔁行リセット） |

**改修した既存ファイル**:

- `webui/src/modes/chain/ChainScreen.tsx`: `<BatchI2vLongSection …/>`を**1段追加**しただけ（`.create-layout`＝Chain画面の2カラムグリッドの**外**に、全幅の兄弟として置く）。加えて§1-6のトリム判定呼び出しが同ファイルに入っている（別テーマ）。
- `webui/src/shell/runLock.ts`（**新規**）: バッチA2V × バッチi2v-longの相互排除ロック（§6）。
- `webui/src/modes/batch/useBatchForm.ts`: 上記ロックの組み込み（実測+61行）。`webui/src/modes/batch/BatchSection.tsx`: ロック中の警告1行を表示（+4行）。**バッチA2Vへの変更はこの2箇所だけ**——ランナーの持ち方（`useRef`）には触れていないため、§6.5末尾の孤児化問題は残る。
- `webui/src/i18n/strings.ts`: `batchI2vLong`名前空間をen/ja両方に新設（キー集合の一致は`strings.test.ts`が自動検証）。

**ブリッジ追加は不要**（初版どおり）: フォルダ選択`ui.pickFolder`・ディレクトリ列挙`fs.listFiles`（拡張子フィルタつき・非再帰）・`backend.uploadFile`・`backend.downloadVideo`の`destDir`/`fileName`/`noClobber`——全部品が既存。

## 6. 開始ガードと相互作用（実装済み）

### 6.1 Startをブロックする理由（11種＋Chain画面の理由の展開）

**1行1理由で表示する**（不透明な「開始できません」だと、就寝中に走らせるつもりだったバッチが動かない原因を誰も特定できない）。`useBatchI2vLongForm`の`I2vLongBlockReason`が安定コードで、表示は`GenerateReasonsNote`＋Chain画面と同じ文言辞書を通す。

| コード | 意味 |
|---|---|
| `imgDirMissing` / `outDirMissing` | フォルダ未指定 |
| `noRows` | スキャンしていない（または0件） |
| `noRunnableRows` | 行はあるが全部`Done`（`Waiting`/`Failed`/`Generating`が1件もない） |
| `sourceVideoAttached` | Chain画面がV2Vモード（§2-3の排他。テンプレート側でも`source_video`を落とすが二重で止める） |
| `clipsTooFew` | クリップが2本未満（素のchainの下限） |
| `promptEmpty` | **合成後のプロンプトが空白のみの実行対象行がある**（＝共通プロンプトが空で、その行のプロンプトも空）。スキャン前は行が無いので共通プロンプトだけで判定する。理由の文言は、行が判明していれば`#1, #2`のように**行番号を並べる**（10件超は`…(+N)`で省略） |
| `promptTooLong` | **合成後のプロンプトが2000字超の実行対象行がある**（サーバー側の`max_length=2000`。Chain画面のPromptBar自身は`maxLength`で守っているが、行プロンプトはその外側で文字を足すので**行ごとに合成後を再検査する**）。こちらも行番号を出す |
| `unknownLoraTag` | テンプレートの`loras[]`に、サーバーに登録の無い名前がある（`<lora:typo>`は全行を404にするので事前に止める。**LoRA一覧が読み込み中／取得失敗のときはこのガードを通す**——一時的な`GET /loras`失敗で夜間バッチ全体を止めるほうが害が大きい） |
| `jobActive` | 単発生成など別のジョブが走っている |
| `lockedByOther` | もう一方のバッチ（Create画面のバッチA2V）が共有ロックを握っている（§6.3） |

これに加えて、**Chain画面自身の`validityReasons`を潰さずそのまま展開する**（`chainBlockReasons`）。「chainの設定が不正です」の1行に丸めると、ユーザーは何を直せばいいか分からない。ただし`sourceVideoAttached`が出ているあいだは、それと同じことを言っているソース動画系のコードだけ間引く。

### 6.2 ブロックしない注意（非ブロック）

「そうなりますが、意図していますか？」という助言であり、Startは止めない。

- `clip0PromptOverride`（**最も強い注意なので警告バナー表示**）: clip 0に個別プロンプトがある。行の画像は必ずclip 0に載るので、`replace`にしても共通プロンプト・行プロンプトが画像の効く区間に届かない。
- `seedFixed`: `seed >= 0`なので全行が同じseedになる。
- `otherClipPromptOverride`: clip 1以降に個別プロンプトがある（案Aにより書き換えない）。
- `startFrameIgnored`: Chain画面のclip 0に冒頭キーフレーム画像がある（行の画像で置き換わる）。
- 同時実行に関する常設の注意（ウィンドウを閉じると行状態が消える／1本ずつしか走らない）。

### 6.3 相互作用: バッチA2Vとの共有ロック（`shell/runLock.ts`）

バックエンドは「1ジョブ＋実行中は409」の設計なので、2つのバッチが同時に走ると互いの409リトライを食い合って無駄になる。そこで**所有者トークン方式の共有ロック**を1つ置いた。

- `acquireRunLock(owner)`は成功時に不透明なトークンを返し、既に誰かが握っていれば`null`を返す（**再入不可**——同じ所有者ラベルでも2本目は取れない）。
- `releaseRunLock(token)`は**そのトークンが今まさに保持者であるときだけ**通る。参照比較なので、同じフィールドを持つ偽造オブジェクトで他人のロックを外すことはできない。
- **なぜ素朴な真偽フラグではないのか**: `AppShell`は全タブを常時マウントするため、「自分のランナーがidleになったら解放する」という素朴なエフェクトは、**もう一方のパネルがマウントした瞬間**（そのランナーは当然idle）に発火して、自分が取っていないロックを解放してしまう。
- 取得は`runtime.run()`が`start()`を呼ぶ**直前**に行う（`start()`のPromiseは走行完了時にしか解決しないため、そこで判定しても遅すぎる）。解放は「running → idle」遷移時、または`start()`が`{started:false}`（走るものが無い／既に走っている）を返した直後。
- **`stop()`ではロックを解放しない**（下記の理由で、いま生成中の1枚はサーバー側で走り続けているため）。

### 6.4 中止の意味: `DELETE /jobs/{id}`は投げない（バッチA2Vとの意図的な差）

中止＝**「以降の行を投入しない」だけ**。いま生成中の1枚はポーリングを続け、完了したら`Done`として保存される（Stopを押さなかった場合とまったく同じ）。理由は2つ:

1. **バックエンドは実行中のchainジョブを途中で止められない**（キャンセルは、ジョブが走り切った**あと**に`cancelled`と印を付けるだけ）。DELETEを投げてもGPU時間は丸ごと消費され、そのうえ完成した動画を捨てることになる。
2. 厳密に直列実行しているので、**安く消せる`queued`のジョブがそもそも存在しない**（走っているのは生成中の1本だけ）。

したがって`BatchI2vLongRunner`には`currentJobId`フィールドが存在しない（A2V側にあるのはDELETEのためだけ）。**Stopの反応には現在の1枚が終わるまでの時間（数十分になり得る）がかかる**ため、画面上の注意文でその旨を伝える必要がある。なお、外部（ジョブパネル）からDELETEされて`cancelled`が観測された場合は、その行を`Waiting`へ巻き戻して再開できるようにする（A2Vと同じ）。

### 6.5 なぜランナーをモジュールレベルのシングルトンにしたか

Chain画面は右クリックのintentルーティングで`remountTokens`による`key`リマウントが起きる（`shell/AppShell.tsx`）。ランナーをフック内の`useRef`に持たせると、**そのリマウントで走行中バッチが孤児化し、UIからは「実行していない」ように見えるのに共有ロックだけが永久に握られたまま**になる。モジュールレベルに置けば、リマウント後のセクションは購読し直すだけで走行中バッチへ再接続できる（フォルダ・行リストも遅延初期化子でスナップショットから復元する）。

**既知の残問題（新規起票対象）**: **バッチA2V側のランナーは`useRef`保持のままなので、走行中にCreate画面が`remountTokens`でリマウントされると孤児化する。** 孤児化自体は従来からある既存問題だが、今回の共有ロックによって「ロックがアプリ再読み込みまで残る」という滞留が上乗せされた。解消はi2v-longと同じ`runtime.ts`シングルトン化で足りる。台帳は`PENDING_TASKS.md`の§3（トリガー待ち）へ起票済み。

## 7. テストの当たり（実装済み）

`modes/batch/*.test.ts`の既存パターンを踏襲した:

1. `imageRows`純関数（列挙→行生成・拡張子フィルタ・`.tmp`除外・ファイル名昇順の自然順・サイズ超過の事前Failed化・出力フォルダ導出）
2. `buildI2vLongPayload`純関数（promptMode合成の3分岐・**テンプレートがpromptに触らないこと**・**`buildRowPayload`が行ごとに別のpromptを作ること**・`source_video`の除去・clip 0の`conditioning_images`置き換え・clips 1..nの無改変・テンプレートと共有部分の非破壊性・`ChainSnapshotSource`への構造的代入可能性のコンパイル時アサーション）
3. `batchI2vLongRunner`クラス（`batchRunner.test.ts`と同型: 未完了行のみ・同時start拒否・アップロードのキャッシュ・失敗行スキップ継続・409リトライ・stopで現在行が完走・**DELETEを投げないこと**・noClobberで実際に書かれた名前を記録すること・**行ごとに別のpromptが送信ボディへ載ること**）
4. `runtime`シングルトン（リマウント後の再接続・ロックの取得と解放・`{started:false}`時の即時解放）
5. `useBatchI2vLongForm`フック（フォルダ選択/scan/11種のガード/**行プロンプトの編集・📝流し込み・再スキャンでの消去・空/長すぎ行の検出**/非ブロック注意/start再判定/**開始時にモードが凍結されること**）
6. `BatchI2vLongSection`/テーブルのUIテスト（**プロンプト列の編集可否・6列構成・長い説明文が無いこと**を含む）
- 検証は`npm run typecheck`（**`tsc -b`**。`npx tsc --noEmit -p .`は偽合格）＋vitest＋oxlint。

## 8. 未決事項（初版§7の決着）

1. ~~`GET /config`が`upload.allowed_image_extensions`を配信しているか~~ → **配信している**。動的取得で実装し、ハードコードは保険のフォールバックのみに留めた。
2. ~~出力ファイル名規約~~ → **画像のstem＋`.mp4`**（A2Vの音声stem踏襲と同型）。衝突時は`noClobber`連番で、表には**実際に書かれた名前**を記録する。
3. ~~seedの扱い~~ → **Chain画面の値をそのまま借用**（A2Vの現行挙動に合わせた既定）。全行が同seedになる場合は非ブロック注意`seedFixed`で伝える。行ごとの乱数化はオーナー判断で後から足せる。
4. ~~1-clip chain許可の凍結例外~~ → **不要**（§2の訂正のとおり、2本以上のクリップを組む本機能には当たらない制約だった）。

## 9. 関連台帳

`PENDING_TASKS_CLOSED.md` §3-57（本件のクローズ記録。起票時は`PENDING_TASKS.md` §1-7）・`PENDING_TASKS.md` §3-1（バッチA2Vのα版省略機能）・同§3-47（バッチA2Vランナーの孤児化とrunLock滞留。本書§6.5で起票）・同§3-48（JobLedgerの多数件描画）・`PENDING_TASKS_CLOSED.md`のバッチA2V設計経緯。
