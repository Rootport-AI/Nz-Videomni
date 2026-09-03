# V2Vリボン範囲トリム 作業指示書（旧`PENDING_TASKS.md` §1-6 →現`PENDING_TASKS_CLOSED.md` §3-59 の正本）

起票: 2026-07-30。調査完了・仕様確定済み。**フェーズ0〜3は同日実装完了（§0参照）。** 残作業に着手する者は本書を最初から最後まで読んでから始めること。着手時はプランモードで実装計画を出し、オーナー承認を得る（本書は「何を作るか」の確定であり「どう作るか」の細部はプラン時に詰めてよい）。

## 0. 状態（2026-08-01）

> **全ゲート合格・クローズ（2026-08-01）。** 本書は完成した仕様の正本として残す（以後の改修はここを読んでから行う）。クローズ記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) **§3-59**（※§3-58はAcceleration。混同しないこと）、実機ゲートの記録は[`REAL_BACKEND_CHECKLIST.md`](REAL_BACKEND_CHECKLIST.md) §4.12（全5項目チェック済み）、実装記録は[`DEVLOG.md`](DEVLOG.md) §54（骨格）・§56（調査確定・仕上げ・IC-LoRA拡張・クローズ）。

採用案は(b-1)（`/upload/video`への引数加算＋ブリッジ契約v10）。バックエンド・native・WebUIの3層とも実装とテストが完了しており、機械検証はすべて緑である（backend pytest **817 pass/6 skip**〔Accelerationの追加分を含む最新値〕、native doctest **274 pass/6 skip**、webui vitest **1604 pass/10 skip・109ファイル**〔`npm run test`をそのまま流した実測値。skipの10件は`backend.integration.test.ts`が実バックエンド未起動時に自動スキップする分〕）。

**2026-08-01に§4の実機調査が完了し、機能が有効になった。** `再生位置`の単位は**素材時間軸の秒**と確定した（確定事実の全文は§4）。これを受けて、

- nativeが`timeline.getSelection`で再生位置系6フィールドを返すようになり（`playbackStartSec`／`playbackEndSec`／`hasPlaybackRange`／`playbackSpeed`／`loopPlay`／`sectionCount`）、
- `decideSourceTrim`が実際に`{trim:true}`を返すようになった。

**無トリム時のバイト等価は維持されている**：無加工のリボンは`再生位置`に`0.000,<素材全長>`を明示的に持つので「再生位置が読めないから見送る」経路ではなく、「リボンが素材全体を覆っている」という判定条件3で無トリムに落ちる。この経路は`ChainScreen.prefill.test.tsx`のT24（`toHaveBeenCalledWith`でパラメータ全体を厳密比較）で機械的に固定してある。

**残作業は無い。** 実機ゲート（[`REAL_BACKEND_CHECKLIST.md`](REAL_BACKEND_CHECKLIST.md) §4.12）は次のとおり全項目が合格した:

1. **①実機採取・解析**（2026-08-01）— `再生位置`の単位が素材時間軸の秒と確定（§4）。
2. **②非退行**（2026-07-31）— リボン＝素材全体なら従来と同一（無トリム時バイト等価）。
3. **③短尺ゲート**（2026-08-01）— 短すぎるリボンで「This video is too short. A video of about 4 seconds or longer is required.」が出て、冒頭動画が挿入されない。
4. **④トリム後Joinの中身連続性**（2026-08-01・本改修の決め手）— 末尾を削ったリボンのV2Vで「切られた位置から続き」が生成される。
5. **⑤IC-LoRA参照動画の範囲反映**（2026-08-01・同日の拡張分。§7-3）— クリップ区間が正しく参照される。

調査コード（`NZVIDEOMNI_PROBE_VIDEO_ITEMS`のCMakeオプション・`build.ps1`の`-ProbeVideoItems`スイッチ・プローブA/B本体）は2026-08-01に**すべて撤去済み**。`scripts/deploy.ps1`の`[PROBE]`ガード（配置しようとしているDLLに調査コードが残っていないかバイト検索する保険）だけは無害なので残してある。

実装記録は[`DEVLOG.md`](DEVLOG.md) §54（骨格）・§56（実機調査の確定と仕上げ）、バックエンド側の検証記録は[`Nz-Videomni/Docs/VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §42。

## 1. 目的とオーナー確定方針

タイムラインの動画オブジェクト（リボン）から右クリック→V2V（Clip Chainの冒頭クリップ化）を実行するとき、現状はリボンの長さ・範囲に関わらず**元動画ファイル全体**が送られる。これを「リボンが元動画の一部しか占めていない場合は、**その範囲で元動画をトリムしたmp4**を冒頭クリップにする」よう改修する。

オーナー確定事項（2026-07-30）:
- **右クリックメニューは増やさない**。V2V選択時の内部分岐で処理する。
- リボンが元動画全体と等価なら**現行どおり**（無トリム。現在の挙動とバイト等価であること）。
- トリム後が短すぎる場合は、Clip Chain既存のクライアント側ゲートの作法（**Generateボタン無効化＋案内メッセージ**）で弾く。**新しい422経路は作らない**（台帳§3-32のトリガーを発火させない）。
- §3-34の残置コード（`CutoutRangeWorker`＝シーン合成をフレーム毎レンダリングする方式）は**採用しない**。あれは「プレビューに映る合成結果を焼く」機能であり、本件は「元素材ファイルを時間で切る」機能。別物として扱う（§3-34は残置のまま・再訪トリガー不成立）。

## 2. 現状フローの要点（調査で確定）

- 元動画パスの取得点は `webui/src/modes/chain/ChainScreen.tsx:340-341` の1箇所（`selection.selected[0].filePath`）。**リボン範囲（`frameStart`/`frameEnd`）は同じ`item`に既に載っているが未使用**。
- `frameEnd`は**INCLUSIVE**。リボン長＝`frameEnd - frameStart + 1`（根拠: `native/src/bridge.cpp:591`。WebUI側の前提明記: `prefillSeed.ts:66-75`）。秒換算は既存純関数 `spanDurationSec(item, selection)` がそのまま使える。
  - **【2026-09-04追記・決着済み】端点解釈は「包含」で決着し、上の本文の`INCLUSIVE`は正しい**（同日いったん「未決の突き合わせあり」と書いたが、その根拠だった「排他」判断は撤回された。正本は[`SDK_REFERENCE.md`](SDK_REFERENCE.md) §16 (h)、契約側の記述は[`BRIDGE_CONTRACT.md`](BRIDGE_CONTRACT.md) §4.13、経緯は[`DEVLOG.md`](DEVLOG.md) §107.11）。なお上の本文が挙げている根拠の行番号`native/src/bridge.cpp:591`は当時のもので、現物は`GetSelectionEditProc`内の`item.frame_end = lf.end;`（現`:827`）である。（本ワークオーダーは凍結文書のため上の本文は当時のまま残してある）
- アップロードは `useSourceUpload.ts:104-136` → `backend.uploadFile {kind:"video", filePath}` → `native/src/bridge.cpp:1667-1707`（**現状はURLクエリを付ける口が無い**・単一パートmultipart）。
- 音声側の兄弟機能 `timeline.extractAudio`（#3 videoAudioToVideo）は**既にリボン実範囲を渡して出荷済み**（`CreateScreen.tsx:386-449`）。「リボン範囲を素材化に反映する」思想は音声側では前例があり、映像側だけ取り残されているのが現状。

## 3. 設計（確定）

### 3.1 トリムの実行場所 = バックエンド（推奨(b-1)、代替(b-2)）

**フロント側でのトリムは不採用**:
- ffmpeg同梱は`Docs/TIMELINE_ALPHA_REQUIREMENTS.md`で明示的に不採用決定済み。
- Media Foundationでの自前トリムは、AviUtl2の入力プラグインで読める形式（mkv/webm/VP9等）をMFが読めず「AviUtl2では見えるのにトリムできない」という説明困難な失敗を生むため不採用。

**バックエンド側の根拠**（すべて実読で確認済み）:
- ffmpegは同梱済みで可用性保証あり（`services/video_io.py:28-32`の`shutil.which`＋`run.ps1:40-43`のPATH先頭注入）。
- **フレーム精度トリムの実装が既にある**: `cut_tail_mp4`（`services/video_io.py:262-340`）が`-ss`/`-t`を避け`select='gte(n\,{start})'`＋`atrim`でフレーム単位に切る方式を確立済み。任意区間版は`between(n\,a\,b)`に変えるだけのクローン（新設名の例: `cut_range_mp4`）。**リサンプルしない**＝ソース実測fpsで切る（joinのトリムと同方針）。
- **API凍結との整合**: `/upload/video`はBEの凍結契約表（`Videomni_Backend_Specification.md` §6.1）に**未掲載**（Phase B加算エンドポイント）。同型の前例として、Join復活時に`POST /jobs/{id}/join`へ`source_tail_seconds`（元動画を秒数で切る引数）を加算済み。作法はNAGで確立した「未指定時は旧リクエストとバイト単位で同一」。
- **Joinが自然に正しくなる（決め手）**: `joinedInsertFrame`（`webui/src/jobs/fpsConvert.ts:54-65`）は既に**リボンspan基準**で挿入位置を計算しており、フル尺を送る現状はここが潜在的に食い違っている。アップロードするファイル自体をトリムする本方式でのみ、Join・挿入位置・生成文脈の三者が一致する。（逆に「chainリクエストに範囲を渡してサーバーが生成時だけ切る」案は、Joinがフル尺を結合してしまうため**不採用**。`SourceVideoSpec`は凍結表内でもある。）

**(b-1) 第1候補**: `/upload/video`に`trim_start_sec` / `trim_duration_sec`（`Query(None)`・両方Noneなら現行コードパスを素通し）を加算。FE側はブリッジ契約v10で`backend.uploadFile`に`query?: Record<string,string>`を加算（`backend.request`が既に持つ形と同型）。ネイティブは`bridge.cpp:1686`でURLにクエリを連結するだけ。**1リクエストで完結し中間状態が生まれない**。プラグイン再ビルド＋再配置が必要。

**(b-2) 代替**: ネイティブ無改修。アップロード後に`backend.request`で新設RPC（例 `POST /upload/video/{video_id}/trim`、joinと同じatomic renameで`input.mp4`を置換）を叩く。C++変更ゼロだが2リクエストになり、`useSourceUpload`の状態機械に「トリム中」フェーズが要る（`status:"ready"`前に完了させないとGenerateが一瞬有効になる）。**プラグイン再ビルドを避けたい場合のみこちら**。

トリム値の単位は**秒**（Joinの`source_tail_seconds`と同型。クライアントはソースの実fpsを知らないため）。保存はatomic renameで**置換**（`uploads/`にTTL・自動削除は無いため、元＋トリム後の2本を残す設計にしない）。

### 3.2 トリム判定（新規純関数 `webui/src/timeline/sourceTrim.ts`）

「疑わしきはトリムしない」の保守的判定。全条件を満たすときだけトリムし、そうでなければ現行どおり全体を送る:

1. `mediaDurationSec > 0`（0は「不明」＝静止画・取得失敗）
2. `spanSec`が解決できる（`rate>0 && scale>0 && spanFrames>0`）
3. `spanSec < mediaDurationSec − 1/projectFps`（**1プロジェクトフレームの許容誤差**。ソースfps≠プロジェクトfpsのときの丸めで最大1/(2×projectFps)秒ずれるため、厳密一致比較は禁止——「全体なのにトリム扱い」で無トリム時バイト等価が壊れる）
4. 再生速度が1.0、かつ中間点が1区間のみ（→§4の実機調査の結果次第で条項を調整）

トリム範囲: `startSec = 再生位置を秒換算した値`（§4で単位確定後）、`durSec = spanSec`。リボンが元動画より長い場合（引き伸ばし・ループ）は条件3が偽→自動的に無トリム。

### 3.3 短尺ゲート（2箇所・両方入れる）

- **ゲートA（右クリック時）**: `menuSelection.ts`の長さガード。**「実際に上がる素材の尺」で測る方式を採る**——`decideSourceTrim`に判定させ、トリムが成立するときはその窓（`decision.durationSec`）、成立しないときは従来どおりファイル全長（`item.mediaDurationSec`）を測る（実装は `measured = decision.trim ? decision.durationSec : item.mediaDurationSec`）。
  - **訂正（2026-07-30、起票時の§3.3から変更）**: 起票時は「`#2 referenceVideo`と同じ`spanDurationSec`基準へ一本化する（実質1行）」と書いていたが、**敵対的レビューでこれが機能退行になると判明したため不採用にした**。span一本化すると、無トリム経路（＝再生位置が不明・再生速度が1.0以外・中間点が複数など、保守的にトリムを見送るすべての場合。および実機調査が終わるまでの現行動作そのもの）でも短いspanを根拠にブロックしてしまう。実際にアップロードされるのは元動画ファイル全体なのだから、そこで弾くのは誤ブロックである。「上がる素材の尺で測る」方式なら、トリム成立時は窓を測り、それ以外は歴史的な全長判定がそのまま残るため、現行動作が1ミリも変わらない。
  - ~~`#2 referenceVideo`（IC-LoRA参照動画）は起票時のまま`spanDurationSec`基準を維持する（そちらは`spanDurationSec`基準のDURATIONシードに素材が着地する設計で、かつアップロード側にトリムがまだ無いため、意図的に`decideSourceTrim`へ切り替えていない）。~~ → **2026-08-01に統一（§7-3参照）**: #2のアップロードにも同じトリムが入ったため、ゲートAも#1と同じ`decideSourceTrim`基準へ一本化した。「アップロード側にトリムが無い」という保留理由そのものが消えたことによる変更で、span一本化を避ける上記の論拠は#2にもそのまま当てはまる。
- **ゲートB（Chain画面・X3の作法）**: `useChainForm.ts`に理由コード`sourceVideoTooShortForContext`を新設（`trimmedFrames < contextFrames`でGenerate無効化）。作法3点セット＝①理由union（:133-144）②判定push（:795-811）③i18n文言（`strings.ts`の`chain.generateReasons`、雛形は`create.generateReasons.audioTooShort`）。`contextFrames`はユーザーが後から編集できるためゲートAだけでは覆えない。これにより`contextFrames`を上げた場合の422も塞がり、§3-32の「422を起こす操作手段が無い」状態が維持される。

クライアント側のフレーム数計算: `effective = Math.round(spanSec × 生成fps)`（サーバーの`preflight_source_video`と同じ丸め。`services/pipeline_manager.py:406-422`参照）。

### 3.4 DURATIONシードは無改修

`extend-video`のDURATIONは`comfortCeiling`（解像度由来）で素材尺非依存（`prefillSeed.ts:47`, `:95-108`）。触らない。

## 4. 実機調査の確定記録（2026-08-01・完了）

**採取・解析とも完了した。以下が確定事実であり、本節の後半（起票時の「これから調べること」）は歴史記録として残してある。**

### 4.1 `再生位置`の生値の形

- 生値は**4フィールドのCSV**で、`開始,終了,再生範囲,0` の形をとる（例: `"2.000,10.700,再生範囲,0"`）。
- **先頭2つの値は「素材の時間軸上の秒」**である。小数3桁固定・小数点はピリオド。
- **無加工のリボンでも省略されない**。素材全体を使っている状態では`0.000,<素材全長>`が明示的に入る。
- **頭を削ると第1値が動き、末尾を削ると第2値が動く。**
- **中間点は`再生位置`の値には一切現れない**（区間数の判定は`get_object_section_num`だけが頼り）。

### 4.2 単位はプロジェクトfpsに依存しない

同じ「頭を2秒削る」操作が、30fpsのプロジェクトでも24fpsのプロジェクトでも`2.000`と出る。したがって単位は**素材時間軸の秒**であり、プロジェクトのフレームレートによる換算は一切不要である。`sourceTrim.ts`の`playbackStartSec()`は素通しの読み出しになった。

### 4.3 トラックAPIでは第2値が取れない

`get_object_track_value`は`再生位置`について**常に第1値しか返さない**。第2値（終了位置）を取る手段がトラックAPIには無いため、**生文字列のパース一択**である。nativeのパーサ`ParsePlaybackRange`（`bridge_core.cpp`）は、**フィールド数にもモード名にも依存せず先頭2つの数値だけを取る**方針で書いてある——第3フィールドは日本語のモード名なので、将来のAviUtl2で表記が変わったりフィールドが増減したりしてもパースが壊れないようにするためである。読めなかった場合はすべて`hasPlaybackRange:false`に倒れ、WebUIは無トリム（＝v10以前と同じ全体アップロード）になる。

### 4.4 `再生速度`と`ループ再生`

- `再生速度`の生値は**百分率の小数2桁文字列**（`"100.00"`／`"200.00"`）。nativeが100で割って1.0スケールに正規化して返す。
- **`ループ再生`という項目が存在する**（type=3、生値`"0"`/`"1"`）。起票時には想定していなかった項目で、オンだとリボンが窓を繰り返し再生するため単一の切り出しでは再現できない。新しい判定理由`loopEnabled`を設けて無トリムに落とすことにした。

### 4.5 R4型のはみ出しと`min`クランプ（決め手）

実機には**「開始位置をずらしてもリボンの長さが変わらない」状態**（採取ケースR4）が存在する。このときリボンのほうが実際に再生される窓より長く、リボンの末尾は終端フレームの静止画になる。ここで`spanSec`（リボンの秒数）をそのまま切り出し尺にすると、**タイムラインには映っていない映像まで切り出してしまう**。

逆方向のずれも実測されている。素材のフレームがプロジェクトfpsに量子化される都合で、`第2値 − 第1値`が**リボンの秒数を最大1フレーム弱だけ上回る**ケース（24fpsで確認）がある。

したがって切り出し尺は

```
durationSec = min(リボンの秒数, playbackEndSec − playbackStartSec)
```

とする。**`min`の両側がそれぞれ実在のケースに対応している**ので、どちらか一方だけでは破れる。さらに素材末尾を越えないよう`mediaDurationSec − startSec`でもクランプする（従来どおり）。

### 4.6 採取しなかった／できなかったケース

- **R7はUI上そもそも作れない**ため未実施。
- **R8は未実施**。ただしこのケースが生む微小なずれは、判定条件3（`spanCoversWholeMedia`）が持つ**1プロジェクトフレームの余裕**が吸収するので、無トリム側に安全に倒れる。
- 素材のプロジェクトfps量子化誤差があるため、**厳密一致比較は既存方針どおり禁止**（条件3の1フレーム余裕を外してはならない）。

---

## 4-old. 【起票時の前提条件】実機調査2点（歴史記録・§4で解決済み）

`enum_effect_item(L"動画ファイル", …)`（`aviutl2_sdk/plugin2.h:651-660`）の実機1回実行＋実プロジェクトのエイリアスダンプで、以下を確定させる:

1. **`再生位置`の単位**（フレームか秒か・どのfps基準か・省略時=0か）。項目の存在自体は実装で確認済み（`native/src/alias_util.cpp:27-28`の`kItemPlaybackJp`。`PatchAliasReplaceVideoFilePath`が意図的にこの行を削除している）。読み出しは`ParseAliasItemValue(alias, "動画ファイル", "再生位置", …)`が最安。**単位が確定しないなら本機能は保留**（「先頭からリボン長」への縮退は、V2Vが末尾をcontextに使う以上ズレが大きく不可）。
2. **`再生速度`項目の有無**。存在する場合、速度≠1.0は「トリムせず現行どおり」を推奨（トリム秒数と元動画上の消費尺が一致しなくなるため）。中間点（`get_object_section_num`）が2区間以上の場合も同様に無トリム推奨。

確定したら、リボン状態の追加フィールド（`selected[].playbackPositionRaw`等）をブリッジ契約の加算（v10）で`GetSelectionEditProc`（`bridge.cpp:685-707`）に足す。

## 5. 改修ファイル一覧（(b-1)採用時）

**WebUI**: 新規`timeline/sourceTrim.ts`＋test／`ChainScreen.tsx:340-361`（判定呼び出し）／`useChainForm.ts`（`attachSourceByPath`系へのトリム引数貫通・ゲートB）／`useSourceUpload.ts:104-136`（トリム引数→`backend.uploadFile`）／`menuSelection.ts:235-254`（ゲートA）／`i18n/strings.ts` en+ja／`bridge/types.ts`（契約v10: `query?`）／`bridge/mockBridge.ts`追随。
**native**: `bridge_core.h/cpp`（`UploadFileRequest.query`・パース・クエリ連結純関数／**2026-08-01追加**: `SelectionItem`の再生位置系6フィールド、純関数`ParsePlaybackRange`・`ParsePlaybackSpeedPercent`、`MakeSelectionResult`の直列化）／`bridge.cpp:1686`（URL連結）＋`GetSelectionEditProc`の加算フィールド（**2026-08-01**: 項目名定数3つと読み出しヘルパー`ReadVideoFileItem`）／doctest追加。
**backend**: `services/video_io.py`（`cut_range_mp4`新設）／`api/uploads.py:40-55`（Query2引数加算）／`services/video_upload_store.py`（トリム＋atomic rename）／`tests/test_video_io.py`ほか（区間トリムのフレーム数一致・**引数なし時の同一性**）。
**Docs**: `API_REFERENCE.md` §3.10／`BRIDGE_CONTRACT.md`（v10）／`REAL_BACKEND_CHECKLIST.md` **§4.12**（起票時は「§4.5」＝Chain: V2Vの既存節に足す想定だったが、既存項目のチェック済み欄を書き換えないよう独立節として新設した）／backend `VERIFICATION_LOG.md` §42。

## 6. テスト・実機ゲートの当たり

- 純関数`sourceTrim`のユニット（全体一致／fps差1フレーム丸め→**無トリムに落ちる**／短い／長い／`mediaDurationSec=0`／`rate<=0`／再生位置あり／速度≠1.0）。**2026-08-01追加**: 実測したリボン状態をそのままケース化（R1無加工→無トリム／R2頭2秒削り→`{2.0, 8.7}`／R3末尾削り→`{0, 7.042}`／**R4はみ出し→尺が7.367へクランプ**／24fps量子化→尺がリボン側8.667になる）＋`loopEnabled`＋速度の許容差。
- ゲートA: `menuSelection.test.ts`（#2用ケースの複製が近道）。ゲートB: `useChainForm.test.ts`のX3 describe（:1114〜）と同型＋`contextFrames`を後から上げて無効化されるケース。
- 配線: `ChainScreen.prefill.test.tsx`に「トリムありでトリム値が渡る／**全体一致では呼び出しが現行と完全同一**」（後者が無トリム時バイト等価の機械的保証）。
- vitestは`npm run typecheck`（tsc -b）併用（`npx tsc --noEmit -p .`は偽合格）。**doctest基準は実測258本**（`build/ninja-release/NzVideomni_tests.exe --test-suite-exclude=integration`）＋新規9本＝**267 pass/6 skip**。※起票時に「217本」と書いていたのは古い基準値の転記ミスで、実測と合わなかったため訂正した。**2026-08-01の仕上げ実装でさらに7本増え、現在の基準は274 pass/6 skip**（webui vitestは1585→1597→同日のIC-LoRA拡張分7本を足して**1604 pass/10 skip・109ファイル**〔`npm run test`をそのまま流した実測値。skipの10件は`backend.integration.test.ts`が実バックエンド未起動時に自動スキップする分で、同ファイルを除外して数えると「108ファイル・1603 pass」になる〕）。
- 実機ゲート（手順の正本は[`REAL_BACKEND_CHECKLIST.md`](REAL_BACKEND_CHECKLIST.md) §4.12。**2026-08-01に全項目合格**）: ①`enum_effect_item`実測（§4）②リボン=全体で従来と同一結果③短いリボンでGenerate無効化＋案内④**トリム後V2V→Join→🎞挿入で「中身の連続性」が合う**（本改修の決め手の検証）⑤**IC-LoRA参照動画でリボン範囲が反映される**（2026-08-01の拡張分。§7-3）。
  - **④の判定基準の訂正（2026-07-30）**: 起票時は「**挿入位置**が合う」と書いていたが、これは旧実装の症状の取り違えである。`joinedInsertFrame`（`webui/src/jobs/fpsConvert.ts`）はリボンの`frameStart`/`frameEnd`から挿入位置を計算し`frameStart`より前へは行かないようクランプしているため、**位置はもっともらしい値に収まっていた**。ずれていたのは中身で、サーバーには元動画の全長が送られていたため、Joinが残す末尾（`source_tail_seconds`、既定5秒）と生成の文脈（context）が**リボンで選んだ範囲の末尾ではなく元動画ファイルそのものの末尾**になっていた。したがって判定は位置ではなく「継ぎ目の直前に映っている絵が、リボンで選んでいた範囲の末尾のフレームかどうか」で行う。

## 7. 未決事項（プラン時にオーナーへ提示）

1. ~~(b-1)か(b-2)か（プラグイン再ビルドの可否がトレードオフ。推奨は(b-1)）。~~ → **決着（2026-07-30）: (b-1)を採用**。`/upload/video`への引数加算＋ブリッジ契約v10の`query`。
2. `再生速度`・中間点の扱いの最終決定（§4の実機調査結果を見てから。推奨は「1.0以外・複数区間は無トリム」）。→ **推奨どおり実装済み（2026-07-30）**: `decideSourceTrim`は、報告された`再生速度`が1.0以外なら`nonNeutralSpeed`で、区間数が2以上なら`multipleSections`で、いずれも無トリムへ落ちる。**報告が無い場合（項目が存在しない・古いプラグイン）はスキップ扱いにしない**——大半のオブジェクトは速度1.0であり、かつ最終的に`unknownPlaybackPosition`で全体アップロードへ落ちる保険が1段後ろに残っているため。この方針自体は実機採取の結果を見て見直してよい。→ **決着（2026-08-01）: 推奨どおりで確定**。あわせて実機で見つかった`ループ再生`項目についても同型の扱い（`loopEnabled`で無トリム）を追加した。
3. ~~#2 IC-LoRA参照動画も**同型の不整合**（ガードはspan基準・アップロードはフル尺）を抱えている。今回はV2V限定だが、同じ`sourceTrim`関数で将来対応可能な設計にしておく。~~ → **決着（2026-08-01・拡張実装済み＋実機合格済み）**: オーナー承認のうえ、#2 IC-LoRA参照動画へ同じトリムを拡張し、**同日の実機ゲート⑤（[`REAL_BACKEND_CHECKLIST.md`](REAL_BACKEND_CHECKLIST.md) §4.12）も合格した**——リボンをクリップした動画オブジェクトからのIC-LoRAで、正しくクリップ区間を参照した動画が生成されることをオーナー実機で確認済みである。実機症状は「321フレーム素材の31〜120フレームのリボンでIC-LoRAを実行すると、参照が元動画の1〜90フレームになる」で、原因はアップロードがフル尺だったこと（パイプラインはファイル先頭から`frame_cap`分を読む）。設計どおり`sourceTrim.ts`・`useSourceUpload.ts`・バックエンドはいずれも無改修で、**呼び出し側3点だけ**で完結した:
   - `webui/src/modes/create/CreateScreen.tsx`のautoLoad（`reference-video`分岐）が`decideSourceTrim`→`trimQuery`をアップロードへ渡す。無トリム時は`trimQuery`が`undefined`を返すため要求パラメータは従来と**バイト等価**（`CreateScreen.prefill.test.tsx`のR2で機械的に固定）。手動📁ピック経路（`pick()`）はタイムライン情報を持たないので従来どおり無トリム。
   - **ゲートAを#1と統一**（`menuSelection.ts`）: #2専用だった`spanDurationSec`基準を廃し、#1と同じ`measured = decision.trim ? decision.durationSec : item.mediaDurationSec`へ。これにより上記§3.3の「span一本化は誤ブロックになる」という指摘が#2にも適用される——トリムが成立しない#2（再生位置が不明など）は本当にフル尺が上がるので、そこでspanを根拠に弾いてはいけない。
   - `referenceTrimFailed`ゲート（`useGenerationForm.ts` + `create.generateReasons.referenceTrimFailed`）: 切り出しを頼んだのに`trimmed: true`が返らない場合にGenerateを止める。Chain側`sourceTrimFailed`と同型・同趣旨の文面。
4. `Docs/RIGHTCLICK_REDESIGN_SPEC.md`第9節（範囲切り出し→素材化のスコープ外記載）との関係整理——機構は違うが体験は同じ。台帳§3-34の判断材料③「範囲選択が無言で無視される」は本件で**別方式により解消**される旨を§3-34側に追記済み。

## 8. 関連台帳

`PENDING_TASKS_CLOSED.md` §3-59（本件のクローズ記録。旧`PENDING_TASKS.md` §1-6）・`PENDING_TASKS.md` §3-34（cutoutRange残置・別物）・同§3-32（422ゲート維持）・同§4-23（Outputs/Uploadsの一本化＝スコープ外。保存領域の方針の正本は[`Nz-Videomni/Docs/STORAGE_POLICY.md`](../../../Docs/STORAGE_POLICY.md)）・`PENDING_TASKS_CLOSED.md` §3-45（ブリッジ契約の未収録債務）。
