# E3 webui 棚卸し: MiniMax H3 専用トグル「VRAM 16GB mode」(既定 on) を足すときの触る場所と既存の作法

パスは `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\` を `FE\`、`FE\webui\src\` を `src\` と略す。行番号は読んだ時点のもの。npm / vitest は実行していない。

## 経路 1〜2 Settings の部品、永続化、系統ごとの出し分け

| 触る場所 | 前例の形 |
|---|---|
| `src\shell\accelerationSettings.ts` | 型 `AccelerationSettings`(107)、`StoredAcceleration`(299)、`ACCELERATION_DEFAULTS`(232)、`STORED_DEFAULTS`(310)の 4 か所に項目を足す。 |
| 同上 | `withServerDefaults`(270)は全項目を列挙している。足し忘れは typecheck で落ちる(266 行目のコメントが明言)。 |
| 同上 | `readStoredAcceleration`(344)は旧形式の裸文字列の分岐(約 355)と JSON の分岐(約 370)の 2 か所に、項目ごとに `typeof === "boolean"` の既定フォールバックを書く。 |
| 同上 | 既定値は `*_SERVER_DEFAULT` 定数(160〜214)。既定 on の前例は `FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT = true`(189)と `EMBED_MP4_METADATA_SERVER_DEFAULT = true`(214)。 |
| 同上の冒頭コメント(1〜90) | 「Six items exist」の列挙があり、項目を足すたびに追記されている。 |
| `src\shell\useAccelerationSettings.ts` | 初期化(111〜119)、書き出し effect の中身(126〜134)と依存配列(135〜143)、`setXxx` の `useCallback`(145〜171)、戻り値(192〜202)、戻り値の型(10〜63)に足す。 |
| `src\shell\SettingsPanel.tsx` | props の型(57〜93 付近)、分割代入(135〜143)、JSX に足す。 |
| 同上 | ON/OFF の 2 ボタン行は `div.field > span.field-label + div.settings-lang-toggle[role=group] > button.mode-tab(--active)(aria-pressed)` という形。 |
| 同上 | 一番近い前例は Fused GGUF 行(366〜397)で、既定 on、ゲートなし、注記は ON の間だけ出す。 |
| 同上 | 非表示の前例は `{!xxxUnsupported && (<>行 + 注記</>)}`。`keepResidentEmbeddings` が 498〜531、`vae` が約 545〜571。行と注記を同じ条件で囲む。 |
| `src\i18n\strings.ts` | 言語フォルダや JSON は使っていない。TS の辞書 1 本(`export const en` 14 行目、`type Strings = typeof en` 2381、`export const ja: Strings` 2383)。 |
| 同上 | キーの形は `accel<Name>Label` / `On` / `Off` / `Note`。en が 2182〜2260、ja が 3543〜3573。ja を欠くと型で落ちる。 |
| 同上 | 既存注記に H3 の語が既にある。`accelKeepResidentNote`(2241 / 3568)は「MiniMax H3 keeps its DiT and VAE」。`accelKeepResidentEmbeddingsLabel` のコメント(2244)に「LTX 2.3 and MiniMax H3, which refuse the field」。 |
| `FE\Language\*.aul2` | C++ プラグイン側の文字列。Settings の文字列は無い(grep 0 件)。 |
| 永続化 | `localStorage` の 1 キー `"nzvideomni.acceleration"`(`accelerationSettings.ts` 224)に JSON 1 個。ブリッジは通らない。 |
| 同上 | コメントに「キーを節名にしたので、項目追加でキー名を変えずに済む」とある。 |
| `src\shell\comfortDisplayTable.ts:140-149` | `ALL_ON_ACCELERATION: AccelerationSettings` が全項目を書いたリテラル。足さないと typecheck で落ちる。 |

出し分けの事実:

- **`featureScope.ts` の表(`src\shell\featureScope.ts`)**
  - `UiTarget`(33〜49)、`FEATURE_UI`(127〜147)、`SettingsRowsHidden`(325〜330)、`settingsRowsHiddenFor`(339〜346)がセット。
  - `prune_vaed` の行は `{ disables: ["settingsRow.vae"], resets: "vaeMode" }`。
  - `keep_resident_embeddings` の行は `{ disables: ["settingsRow.keepResidentEmbeddings"], resets: "keepResidentEmbeddings" }`。
  - `chain` の行は `["mode.chained","createSection.batchA2v","createSection.a2v"]`、`reference_video` の行は `["chainPanel.reference","createSection.referenceVideo"]`。Settings の行は持たない。
  - 表は「`unsupported_features` に名前がある → 閉じる」という向きだけ。機能名を知る場所はこのファイルだけ、と冒頭に明記。
- **`resets` の仕組み**
  - `accelerationResetsFor`(227〜)が、閉じた行の保存値を `ACCELERATION_DEFAULTS[field]` に書き戻す対象を返す。
  - `src\shell\AppShell.tsx:1488`(`settingsRowsHidden`)と 1506〜1514(書き戻す effect)が使う。
  - SettingsPanel への配線は `AppShell.tsx:1753-1766`。
  - 既定 on の項目に `resets` を付けると、書き戻し先は on になる。
- **H3 のときだけ表示する前例(逆向き)**
  - 表を使うものでは `keep_resident_embeddings`(LTX 2.5 だけに部品がある)が「逆向き」。ただしこれは「LTX 2.3 と H3 が名前を公開して隠れる」形で、肯定形の表示ではない。
  - H3 専用に表示する行の前例は無い。「機能名で表示する」仕組みも、系統 id で出し分ける仕組みも無い。
  - 作法の明文は「画面に系統 id を書かない。`GET /models` の `unsupported_features` / `constraints` だけを見る」(`AppShell.minimaxH3.test.tsx:12` 付近、`featureScope.ts` 冒頭、DEVLOG §106.2)。
- **H3 の 3 本のスライダー(ステップ数・shift_video・shift_audio)の出し方**
  - サーバーの `constraints.sampling` が非 null かどうかで決めている。
  - 経路は `useBaseModels.ts:301-317` → `generationConstraints.ts:61-75`(`sampling: raw.sampling ?? null`)→ `useGenerationForm.ts:896`(`samplingConstraints`)→ `GenerationForm.tsx:262`。
  - 型は `types.ts:1213-1236`。
  - 系統 id は使っていない。
  - `engineFamily`(H3 は `"h3"`、`mockBridge.ts:323`)は快適上限表の鍵にだけ使う(`comfortTable.ts`)。
  - `SettingsPanel` は `engineFamily` を受け取っているが(props)、`constraints` は受け取っていない。
- **台帳 §1-100(`Nz-Videomni\Docs\PENDING_TASKS.md:36`)**
  - Fused GGUF 行は GGUF を使わない H3 で非表示にしたい、という内容。
  - 今は機能名の受け口が無く、`featureScope.ts` の表にも `SettingsPanel` にも行が無い。H3 でも現に表示される。
  - 提案は「アダプタが `fused_gguf_dequant_kernel` などを公開し、表と部品に受け口を足す」で、未着手。

## 経路 3 要求への載せ方

| 触る場所 | 前例の形 |
|---|---|
| `src\shell\accelerationSettings.ts:481-488, 523-537, 554-` | `AccelerationRequestFields`(481)にキーを足す。 |
| 同上 | `effectiveAccelerationFields`(523)は `Required<...>` を返すので、足すと必ずここも直す必要がある。`comfortTable` の `requires` 照合用のバッグになる(`comfortTable.ts:178`)。未知のキーは `requires` に現れなければ無害。 |
| 同上 | `accelerationRequestFields` は「サーバー既定と違うときだけ送る」規約。最後に追記してキー順を安定させる。 |
| 同上 | 既定 on の前例は `fused_gguf_dequant_kernel`。off のときだけ `false` を送る(約 590)。 |
| `src\api\types.ts` | `GenerateRequest`(約 218〜268)と連結要求(約 541〜573)にキーを足す。 |
| 同上 | `AlphaGenRequestBody`(約 308-312)は 5 キーのみ。 |
| `src\modes\batch\buildA2vChainPayload.ts:210-217` | ローカルの snake_case 型(`A2vChainPayload`)にもキーを足す。足さないとスプレッドが typecheck で落ちる。 |
| `src\modes\batch\useBatchForm.ts:868-876` | 「いずれかが既定から外れていたら `acceleration` を渡す」OR-list に項目を足す。足さないと Batch A2V だけ載らない(コメントが明記)。 |
| 変更なしで載る入口 | Create(`useGenerationForm.ts:1273`)、Chain(`chainUtils.ts:951`)、Batch i2v(`buildI2vGeneratePayload.ts:89`)、Retake / Outpaint / Inpaint(`useRetakeForm.ts:649`、`useOutpaintForm.ts:465`、`useInpaintForm.ts:564`)は `...accelerationRequestFields(acceleration)` をスプレッドしている。 |
| 手書きで選別する入口 | Alpha Gen(`useAlphaGenForm.ts:366-380`)は 5 キーだけを手で選んでいる。`vae_mode` と `embed_mp4_metadata` は意図的に除外。 |
| 同上 | H3 は `alpha_gen` を非対応として公開するので、Alpha Gen は H3 では使えない。 |

要求の流れは、`AppShell` の `useAccelerationSettings` が `acceleration` を受け持ち、各画面に props で配る形。Context は使っていない。

## 経路 4 ブリッジ契約と aux2

- `FE\native`、`FE\Language`、`FE\scripts`、`FE\CMakeLists.txt`、`Docs\BRIDGE_CONTRACT.md` には、`attention_backend`、`block_swap`、`keep_resident`、`vae_mode`、`embed_mp4` が 1 件も無い(grep 0 件)。
- `backend.request` の body は不透明な JSON(`bridge\types.ts:19` 付近)。契約は v16 のままで、前例の追加でも版は上がっていない。
- `localStorage` はブリッジを通らない。
- つまり、Settings に項目を足しても aux2 の変更も契約の版上げも要らない。ただし webui を作り直した `dist-single\index.html` を埋め込んだ aux2 の再ビルドと配置は必要。
- 前例の記録は `FE\Docs\DEVLOG.md` §58(`keep_resident`、2594 行目〜)と §106(`keep_resident_embeddings`、3904 行目〜)。§106.5 に触ったファイルの一覧がある。

## 経路 5 モック(`src\bridge\mockBridge.ts`)

- `*_used` は返していない。`seed_used`(843)以外に出てこない。
- フロントも `*_used` を読まない。`types.ts` のコメントにあるだけ。
- 更新点:
  - 公開リスト `MOCK_UNSUPPORTED_FEATURES`(390〜)。H3 は 476〜493、LTX 2.3 は `keep_resident_embeddings` が 400。
  - 連結側の拒否対応表 `MOCK_CHAIN_FEATURE_FIELDS`(約 518〜551)。ここに行を足すと連結側だけ 422 を返せる。
  - 単発 `/generate` には拒否ループが無い(コメント約 539。`featureUnsupported` の呼び出しは 1927 と 1935 のみ)。
  - Alpha Gen の透過リストの 5 キー(1761〜1765)。
- `comfort_budgets` の `requires`(150〜170)は 5 キーのままで、新キーを足しても照合に影響しない。
- `MOCK_CONSTRAINTS_BY_FAMILY.h3`(342〜352)が、H3 の `constraints.sampling` を非 null で返している。

## 経路 6 テスト

件数は `it(` の数え。vitest 自体は実行していない。

| ファイル | 件数 | 前例(§106.5)で足したもの |
|---|---|---|
| `shell\keepResidentEmbeddings.paths.test.ts` | 3 | 新規。on なら 3 経路に載り、既定なら鍵が無いことを確認。同型に `vaeMode.paths.test.ts`(3)。 |
| `shell\featureScope.test.ts` | 28 | 表のキー名一覧(23〜)、`settingsRowsHiddenFor` の `toEqual`(102、110、117)、`accelerationResetsFor`(128〜141)。 |
| `shell\AppShell.featureScope.test.tsx` | 16 | 非表示 / 書き戻しの各 2 本(472〜589)。 |
| `shell\SettingsPanel.test.tsx` | 29 | harness の props(97〜105)。`toEqual` の保存 JSON(366、475、562、681)は全項目を書いている。 |
| `shell\accelerationSettings.test.ts` | 20 | リテラル(33)、`makeAllOn`、読み書き、`accelerationRequestFields`(189)。 |
| `shell\useAccelerationSettings.test.ts` | 28 | 約 10 か所に項目のリテラル。 |
| `shell\comfortTable.test.ts` | 37 | リテラル(31)。 |
| `modes\batch\useBatchForm.test.ts` | — | 1358〜1409。 |
| `modes\single\useGenerationForm.test.ts` | 93 | 629 のリテラル。 |
| `modes\edit\useAlphaGenForm.test.tsx` | — | 437〜454。 |
| `timeline\prefillSeed.test.ts` | — | 585、627 のリテラル。 |
| `bridge\mockBridge.test.ts` | 82 | 943〜977 のリスト期待値。 |
| `bridge\mockBridge.h3.test.ts` | 4 | 42 の H3 リスト。 |
| `shell\AppShell.minimaxH3.test.tsx` | 7 | H3 を端から端まで確認。 |
| `shell\generationConstraints.test.ts` | — | 58 の H3 リスト、158〜188。 |

- 完全な `AccelerationSettings` リテラルがあるファイルは、項目を足すと typecheck で落ちる。直す先は `accelerationSettings.test.ts`、`comfortTable.test.ts`、`useAccelerationSettings.test.ts`、`useGenerationForm.test.ts`、`useAlphaGenForm.test.tsx`、`prefillSeed.test.ts`、`comfortDisplayTable.ts`(本体)。
- 前例の合計は §106.6 で「136 ファイル 2,696 件全緑」。現在の総数は未取得。
- 門は `package.json` の `typecheck`(`tsc -b --noEmit`)、`test`(`vitest run`)、`lint`(`oxlint`)。今回は実行していない。

## 経路 7 文書と配置

- **`FE\webui\README.md`**
  - Settings の行を列挙する節も、H3 の制限の節も無い(Acceleration の語も 0 件)。
  - `FE\README.md:105-113` の Settings panel の箇条書きにも Acceleration の行は無い。
  - したがって、ここは更新が要らない可能性が高い。
- **`FE\Docs\API_REFERENCE.md`**
  - 477〜527 付近の「高速化フィールドの送信規約(6 フィールド共通)」、520 の表、599 の連結側の表に、フィールドが列挙されている。
  - 追加時は更新が要る(`keep_resident_embeddings` の前例は 483、520、524〜527、569、599)。
- **`FE\Docs\DEVLOG.md`**
  - 末尾に新しい節を足す形。最新は §141(H3 の操作パネル)。
  - 先頭の「最終更新」行(3 行目)も更新が必要になる。
- **`Nz-Videomni\Docs\MULTI_ENGINE_DESIGN.md`**
  - §5.6 は 396〜456。H3 の `unsupported_features` の導出規則と、閉じた機能の項目は要求に載せない、の段落がある。
  - §6 は 505〜569。これは UX の節で、加速トグルの記述は今回の範囲では見つからなかった。
- **`Nz-Videomni\Videomni_Backend_Specification.md` §6.2b**
  - 821 行目〜。(f) が H3 の `unsupported_features` を列挙している。
  - バージョン履歴の表(144 行目〜)に行を足す運用。
  - 項目がサーバー側の機能名を伴う場合だけ、ここが更新対象になる。
- **配置(`FE\scripts\deploy.ps1`)**
  - 2 か所のコピー先がある。
    - `$PluginDir`(既定は `D:\For_Videos\AviUtl2\aviutl2_v2.0.54\data\Plugin`)→ `NzVideomni\NzVideomni.aux2`。150〜181 行目。
    - `$DistDir`(既定はモノレポの `AviUtl2-Plugin\`)→ `NzVideomni.aux2`。193〜206 行目。
  - 埋め込み版でない aux2 は配置を拒否する(140〜148)。
  - `scripts\build.ps1` が `npm run build:single` で `webui\dist-single\index.html` を作り、aux2 に埋め込む。

## 要点

1. 変更が要るのは webui のみ。aux2、`BRIDGE_CONTRACT`、`Language\*.aul2` は不要。永続化は `localStorage` の `nzvideomni.acceleration` に JSON 1 個で、ブリッジは通らない。
2. 最も近い前例は Fused GGUF 行。既定 on、ゲートなし、ON の間だけ注記、`false` のときだけ要求に載せる。
3. 項目を足す本体は `shell\accelerationSettings.ts` の 8 か所前後(型、2 つの defaults、`withServerDefaults`、読み取り 2 分岐、request 型、request 組み立て)。`useAccelerationSettings.ts` の 5 か所、`SettingsPanel.tsx`、`strings.ts` の en / ja も要る。
4. 見落としやすいのは `useBatchForm.ts:868-876` の OR-list、`buildA2vChainPayload.ts:210-217` のローカル型、`comfortDisplayTable.ts:140-149` のリテラル。
5. H3 のときだけ表示する前例は無い。表は「公開された名前で閉じる」向きだけで、`resets` は既定値(既定 on なら on)へ書き戻す。
6. 肯定形の出し分けに近い前例は H3 のスライダーで、サーバーの `constraints.sampling` が非 null かどうかを見る。系統 id は使わない。ただし `SettingsPanel` は今 `constraints` を受け取っていない。
7. 設計の分かれ目として、`unsupported_features` に名前を公開するのは LTX 2.3 / 2.5 の側になり、サーバーのアダプタ変更が要る。台帳 §1-100 の Fused GGUF も同じ「受け口なし」の状態。
8. モックは `*_used` を返さない。更新点は `MOCK_UNSUPPORTED_FEATURES`、`MOCK_CHAIN_FEATURE_FIELDS`、Alpha Gen の 5 キー(1761〜1765)。単発 `/generate` にはモックの拒否ループが無い。
9. テストは完全リテラルの 7 ファイルが typecheck で落ちる。`featureScope.test` の鍵一覧と `toEqual`、`SettingsPanel` の harness と保存 JSON、`mockBridge` のリスト期待値、横断テスト(`*.paths.test.ts` の新規)が要る。
10. 文書は `API_REFERENCE.md` の送信規約(477〜527、599)と DEVLOG の新節が要る。`webui\README.md` と `FE\README.md` には Settings の行の一覧が無い。
11. 配置は `deploy.ps1` の `$PluginDir`(150〜181)と `$DistDir`(193〜206)の 2 か所。埋め込みの再ビルドが先に要る。