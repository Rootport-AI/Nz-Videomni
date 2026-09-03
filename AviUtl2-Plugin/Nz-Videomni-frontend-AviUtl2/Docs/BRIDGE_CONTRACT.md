# Nz-Videomni ブリッジ契約リファレンス(WebUI ⇔ ネイティブ JSON-RPC, v4.1 + v5抜粋 + v6 + v7 + v8 + v9 + v10 + v11)

最終更新: 2026-09-04(**文書のみの更新で、契約バージョンはv11のまま**——§3-140〔🎞挿入をエイリアス方式へ差し替え〕に伴う注記を§4.5・§4.7・§4.14.1・§5へ足し、あわせて§4.13へ`frameStart`/`frameEnd`の端点解釈が未決である旨を1箇所だけ記した。ワイヤ形式・RPCの引数・応答はいずれも1文字も変わっていない) / その前は 2026-09-01(contract v11＝素材fps——`timeline.getSelection`の`selected[]`へ`mediaFps`を追加した。素材そのもののフレームレートをMedia Foundationで読み、右クリックプリフィルのfps軸「素材に合わせる」へ流し込むためのもので、新規モジュール`native/src/media_fps_probe.{h,cpp}`が担う。§4.13・§8・§10を更新) / 出典: `webui/src/bridge/types.ts` と `native/src/bridge_core.h` / `bridge_core.cpp` / `bridge.h` / `bridge.cpp` / `native/src/webview_host.cpp` を突き合わせて作成(コード一次)。

> **注記(2026-07-17追記)**: ネイティブ側のJSON型を`nlohmann::json`から`nlohmann::ordered_json`(`bridge_core.h`)へ変更した。ワイヤ形式(送受信されるJSON文字列そのもの)には影響しない。変わったのはC++側でのオブジェクトのキー順が挿入順で保持されるようになった点のみ(従来はキーのアルファベット順)で、契約上の意味は無い。

> **注記(2026-07-08追記)**: `webui/src/bridge/types.ts`は既に「contract v5」(タイムライン生成AI機能向けの`timeline.getSelection`／`timeline.cutoutRange`／`timeline.extractAudio`／`timeline.insertProvisional`／`timeline.resolveProvisional`／`timeline.updateProvisionalText`／`timeline.scanProvisionals`と、イベント`timeline.menuInvoked`／`timeline.projectLoaded`)まで拡張されている。本書は右クリック生成の配線セッションで**`timeline.getSelection`と`timeline.menuInvoked`のみ**を§4.13/§4.14として追補した。上記の他のv5メソッド(cutoutRange等)は本書に未収録のままであり、別途の同期作業が必要(既知のドキュメント債務。詳細は本書を参照した親エージェントへの報告を参照)。
>
> **注記(2026-07-15追記、contract v6)**: バッチA2V(CSVマニフェスト駆動の一括生成)とwav長自動調整のため、`webui/src/bridge/types.ts`に「contract v6」として`ui.pickFolder`／`fs.listFiles`／`fs.readTextFile`／`fs.writeTextFileAtomic`／`fs.probeAudioDuration`の5メソッドと、`backend.downloadVideo`への`destDir`/`fileName`/`noClobber`params拡張(既存呼び出しとの後方互換を保った追加専用の拡張)が加わった。本書は§4.15〜§4.20としてこれを追補している。**(`fs.readTextFile`／`fs.writeTextFileAtomic`は2026-07-18撤去・ステートレス化。§4.17/§4.18・DEVLOG §26参照。以下の記述は起票時点の history として残す。)**
5行目時点(起票直後)では`native/`側(C++)が未実装で、本書冒頭の「実装(C++側)を正とする」という原則がv6の6項目に限り逆転し`types.ts`を一次ソースとしていたが、**同じく2026-07-15のフロントエンド追随グループ2セッションでnative実装が完了した**(新規モジュール`native/src/wav_probe.h/.cpp`・`native/src/fs_util.h/.cpp`を追加し、`bridge_core.cpp`/`bridge.cpp`から配線。`native/tests/test_wav_probe.cpp`・`native/tests/test_fs_util.cpp`のdoctestで単体検証済み)。これにより、**「実装(C++側)を正とする」という本書冒頭の原則はv6についても回復した**。§4.15〜§4.20は実装コードと突き合わせ済みの内容に更新している。v5の残り6メソッド(cutoutRange等)の遡及補完は本セッションでは行っていない(上記2026-07-08注記のとおり、既知の債務のまま)。
>
> **注記(2026-07-17追記、contract v7)**: ドラッグ&ドロップ対応のため、`webui/src/bridge/types.ts`に「contract v7」として`ui.resolveDroppedFiles`メソッド(§4.21)が新設され、既存の`ui.pickFile`に第3波のセッションで実装済みだった`"imageOrVideo"` kind(Chainのソース入力欄一本化用、ネイティブ側は`bridge_core.cpp`にすでに実装されていたが本書に未収録だった)を§4.9へ追補した。`ui.resolveDroppedFiles`は`native/src/bridge_core.h/.cpp`が新規純関数`ParseResolveDroppedFiles`/`MakeResolveDroppedFilesResult`として実装し(`HandleRequestJson`から同期ディスパッチ、`native/tests/test_bridge_core.cpp`のdoctestで単体検証済み)、`native/src/webview_host.cpp`の`WebMessageReceived`ハンドラが`ICoreWebView2WebMessageReceivedEventArgs2::get_AdditionalObjects()`経由でドロップされたファイルの実パスを解決して`params.__droppedPaths`へ強制注入する(§4.21のセキュリティ規律参照)。この注入処理自体(WebView2依存)は実機でしか確認できないため、`webview_host.cpp`に`REALDEVICE-VERIFY`コメントを残置している。
>
> **注記(2026-07-17追記、contract v7 BLOCKER修正)**: 敵対的レビューにより、上記実装には注入の発火条件に**BLOCKER**が1件見つかった — 部分文字列チェック(`"__droppedPaths"`を含むか)を**AdditionalObjectsの有無より先に**判定していたため、`params: {}`(空)を送る本物のドロップ経路(`useFileDrop.ts`)では常にfast pathへ素通しされ、注入が一度も実行されずD&D機能全体が実機で機能しない状態だった。`mockBridge.ts`がこの注入機構自体を経由せず`.name`から直接結果を合成していたため、webuiテスト・native側doctestのどちらもこの穴を検出できなかった。判定順序を「AdditionalObjectsの有無を先に確認」へ修正し、注入のJSON書き換え自体を純関数`InjectDroppedPathsIntoRequest`(`bridge_core.h/.cpp`)へ抽出してdoctestで再発防止テストを追加、`mockBridge.ts`もネイティブの注入を模倣する実装へ修正した。詳細は§4.21のセキュリティ規律の節(BLOCKER修正の経緯)を参照。
>
> **注記(2026-07-20追記、contract v9)**: Create画面の「Reference video(IC-LoRA。参照した動画の動きや輪郭をなぞって生成する仕組み)」欄・「Audio to video(A2V。音声から動画を生成する機能)」欄のカードリデザインで、添付ファイルの尺を「12.3s」のように表示するため、`webui/src/bridge/types.ts`に「contract v9」として`fs.probeMediaInfo`メソッド(§4.22)が新設された。単一のローカルメディアファイルの再生時間・解像度を、AviUtl2 SDKの`EDIT_SECTION::get_media_info`([SDK_REFERENCE.md](SDK_REFERENCE.md) §5.1)経由でベストエフォートに取得する同期メソッドである。`fs.probeAudioDuration`(§4.19)と同じく**取得できない場合もrejectせず`{durationSec:0, width:0, height:0}`の成功応答へ穏当に劣化する**設計で、rejectするのは`filePath`が欠落/空/非文字列の`BAD_REQUEST`のみ。native実装は`native/src/bridge_core.h/.cpp`の新規純関数`ParseProbeMediaInfo`/`MakeMediaInfoResult`(パース/整形、doctest対象)と、`native/src/bridge.cpp`の`ProbeMediaInfoEditProc`(`call_edit_section_param`経由でUIスレッド同期に`get_media_info`を呼ぶ実配線)からなる。doctestは`native/tests/test_bridge_core.cpp`に6ケースを追加(全252→258 pass/6 skip、退行なし)。
>
> **注記(2026-07-30追記、contract v10)**: V2Vリボン範囲トリム(タイムライン上のリボンが元動画ファイルの一部しか占めていないとき、その範囲だけを切り出してアップロードする機能。台帳は`PENDING_TASKS_CLOSED.md` §3-59＝2026-08-01にクローズ済み。起票時は`PENDING_TASKS.md` §1-6)のため、「contract v10」として`backend.uploadFile`に任意の`query`(`Record<string,string>`)を加算し、応答に任意の`trimmed`(bool)が加わった(§4.5)。`backend.request`が既に持っていた`query`と同型で、**省略時はURLが完全に不変**(＝v10以前と1バイトも変わらないリクエストになる)。native実装は`bridge_core.h/.cpp`の共有純関数`BuildQueryString`/`AppendQueryToUrl`(`backend.request`側の既存実装を切り出して両者で共有したもの)と、`bridge.cpp`の`UploadFileWorker`でのURL連結からなる。doctestは`native/tests/test_bridge_core.cpp`へ9ケース追加(258→**267 pass/6 skip**、退行なし。実測値は`NzVideomni_tests.exe --test-suite-exclude=integration`で確認)。**v10のもう半分——`timeline.getSelection`への再生位置系6フィールド(`playbackStartSec`／`playbackEndSec`／`hasPlaybackRange`／`playbackSpeed`／`loopPlay`／`sectionCount`)は2026-08-01の実機調査完了をもって実装・収録済みである(§4.13)**。単位は**素材時間軸の秒**で確定した。このときdoctestはさらに7ケース増えて**274 pass/6 skip**になっている。

関連ドキュメント: [API_REFERENCE.md](API_REFERENCE.md) ／ [SDK_REFERENCE.md](SDK_REFERENCE.md) ／ [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) ／ [DEVLOG.md](DEVLOG.md)

対象: WebUI(React/TS, `webui/src/bridge/`)とネイティブ`.aux2`(`native/src/bridge.{h,cpp}` + `bridge_core.{h,cpp}`)の間のJSON-RPCメッセージ契約。**両ソースを実際に読んで突き合わせた結果、実装(C++側)を正とする**。TypeScript側の型定義(`types.ts`)は契約のミラーだが、意図的に narrow している箇所が1点ある(§9参照)。

---

## 0. 全体像

- バージョン: ネイティブ実装(`native/src/bridge_core.cpp`・`bridge.cpp`・`webview_host.cpp`)は**v5(タイムライン生成AI、7メソッド`timeline.getSelection`／`cutoutRange`／`extractAudio`／`insertProvisional`／`resolveProvisional`／`updateProvisionalText`／`scanProvisionals`を全て実装済み)＋v6(バッチA2V + fsブリッジ、起票時6項目のうち現存4項目——`fs.readTextFile`／`fs.writeTextFileAtomic`は2026-07-18撤去・ステートレス化。§4.17/§4.18・DEVLOG §26参照——を実装済み)＋v7(ドラッグ&ドロップ、`ui.resolveDroppedFiles`実装済み)＋v8(右クリック再設計群、`timeline.insertMediaForJob`／`timeline.updateProvisionalReservation`／`timeline.deleteProvisionalByJob`実装済み)＋v9(IC-LoRA／A2Vカードの尺表示、`fs.probeMediaInfo`実装済み)まで到達している**。プラグインの配布バージョン文字列(`bridge_core.h`の`kPluginVersion = "1.0.0-rc1"`)はビルド識別用の文字列であり、本書がここで扱う契約バージョン番号(v1〜v10)とは別体系なので混同しないこと(旧版の本書はこの2つを混同し「ネイティブ実装はv4.1まで」と誤記していたため訂正した)。`types.ts`(WebUI側の契約定義)もv11まで到達しており、実装(C++側)との差は無い。ただし本書がドキュメントとして収録しているのはこのうち§4.13/§4.14(`timeline.getSelection`／`timeline.menuInvoked`のみ)と§4.14.1(v8: `insertMediaForJob`・`updateProvisionalReservation`・`deleteProvisionalByJob`、`insertProvisional`の`numFrames`／`genFps`／配置系統／`textPrefix`契約、`getSelection`の`textContent`／`mediaDurationSec`拡張、`projectLoaded`購読の要点のみ)と§4.15〜§4.21(v6全項目＋v7)と§4.22(v9: `fs.probeMediaInfo`)であり、v5の残り(`cutoutRange`／`extractAudio`／`insertProvisional`／`resolveProvisional`／`updateProvisionalText`／`scanProvisionals`、イベント`timeline.projectLoaded`)は実装済みだが本書には未収録のまま(既知のドキュメント債務、§0冒頭の2026-07-08注記参照)。v1(M1)→v2(M2: backend proxy)→v3(M4: capture/upload/pickFile/thumbnail)→v4(M7a: settings)→v4.1(M7c: `backend.request`の`timeoutMs`)→v5(タイムライン生成AI)→v6(バッチA2V + fsブリッジ)→v7(ドラッグ&ドロップ)→v8(右クリック再設計群: `insertMediaForJob`・`updateProvisionalReservation`・`deleteProvisionalByJob`、`insertProvisional`の`numFrames`/`genFps`/配置系統/`textPrefix`契約、`getSelection`の`textContent`/`mediaDurationSec`拡張、`projectLoaded`購読)→v9(IC-LoRA／A2Vカードの尺表示: `fs.probeMediaInfo`)→v10(V2Vリボン範囲トリム: `backend.uploadFile`の`query`と応答`trimmed`)の順に追加専用で拡張されてきた。**v10は2026-07-30に到達済み**(native・`types.ts`とも実装済み)で、本書は§4.5と§8で収録している。v10として計画されていた**`timeline.getSelection`への再生位置系6フィールドの追加も2026-08-01に完了した**(実機調査で`再生位置`の単位が「素材時間軸の秒」と確定したため。§4.13の拡張注記を参照)。**さらに2026-09-01、`timeline.getSelection`の`selected[]`へ素材そのもののフレームレート`mediaFps`を足したv11へ到達した**(native・`types.ts`とも実装済み。§4.13の拡張注記と§8を参照)。
- トランスポート: WebView2の`postMessage`チャネル1本(**v7**の`ui.resolveDroppedFiles`だけは`postMessageWithAdditionalObjects`でファイルを追加添付する、§4.21参照)。JSON-RPC風だが**バッチ不可・双方向とも単一メッセージ**。
- 全12メソッド(v4.1時点): `ping` / `getEditInfo` / `backend.request` / `backend.downloadVideo` / `backend.uploadFile` / `backend.getBaseUrl` / `timeline.insertMedia` / `timeline.captureFrame` / `ui.pickFile` / `ui.makeThumbnail` / `settings.get` / `settings.set`。(**contract v5**でさらに`timeline.getSelection`ほか7メソッドが追加されているが、本書がv1〜v4.1時点の初版のため未反映だった。§4.13/§4.14で`timeline.getSelection`と`timeline.menuInvoked`イベントのみ追補ずみ。**contract v6**でさらに`ui.pickFolder` / `fs.listFiles` / `fs.readTextFile` / `fs.writeTextFileAtomic` / `fs.probeAudioDuration`の5メソッドが追加され(このうち`fs.readTextFile`/`fs.writeTextFileAtomic`は2026-07-18撤去・ステートレス化、§4.17/§4.18・DEVLOG §26参照。現存するv6メソッドは`ui.pickFolder`/`fs.listFiles`/`fs.probeAudioDuration`の3つ)、`backend.downloadVideo`のparamsが拡張された(`destDir`/`fileName`/`noClobber`に加え、2026-07-18に`reuseIfPresent`も追加。§4.20参照)。こちらは§4.15〜§4.20に全て収録済み。**contract v7**でさらに`ui.resolveDroppedFiles`が追加され(§4.21)、`ui.pickFile`に`"imageOrVideo"` kindが追補された(§4.9)。)
- 一次ソース: `webui/src/bridge/types.ts`(契約の型定義+ JSDoc)、`native/src/bridge_core.h/.cpp`(純粋なパース/ディスパッチ実装、WebView2非依存でdoctest可能)、`native/src/bridge.h/.cpp`(WebView2/WinHTTP/SDKへの実配線、非同期メソッドのワーカー投入)、`native/src/webview_host.cpp`(WebView2のホスティングと`WebMessageReceived`ハンドラ — v7の`__droppedPaths`注入はここで行われる)。

## 1. メッセージエンベロープ

### 1.1 リクエスト(WebUI → ネイティブ)

```json
{ "id": 42, "method": "ping", "params": {} }
```

- `id`: number。WebUI側(`RequestDispatcher`)が単調増加(`nextId++`、1始まり)で採番。
- `method`: v4.1時点で12種(§0参照)。現在は§3のメソッド表(v5・v6分を含む)に列挙する全メソッドのいずれか。
- `params`: メソッドごとのオブジェクト(§4参照)。省略/非オブジェクトは`{}`として扱われる(`bridge_core.cpp`: `req.contains("params") && req["params"].is_object() ? req["params"] : json::object()`)。

送信は`window.chrome.webview.postMessage(obj)`— **`JSON.stringify`しない**(WebView2がシリアライズを担当)。ネイティブ側は`ICoreWebView2::add_WebMessageReceived`相当でUTF-8 JSON文字列として受け取る想定(`Bridge::HandleMessage(const std::string& request_json)`)。

### 1.2 成功レスポンス(ネイティブ → WebUI)

```json
{ "id": 42, "ok": true, "result": { "pong": true, "pluginVersion": "1.0.0-rc1" } }
```

### 1.3 エラーレスポンス

```json
{ "id": 42, "ok": false, "error": { "code": "BAD_REQUEST", "message": "..." } }
```

- `id`は`null`になり得る(リクエストJSONが破損していて`id`すら読めない場合。`ExtractId`は数値でなければ`null`を返す)。
- `code`はREST的な文字列コード(§5)。

### 1.4 イベント(ネイティブ発、v5以降`timeline.menuInvoked`で使用中)

```json
{ "event": "some.event", "data": { } }
```

`id`を持たないメッセージは`isBridgeEvent`(`typeof event === "string" && !("id" in candidate)`)でイベントと判定され、`RequestDispatcher.on(event, handler)`で購読できる配線は存在する。v4.1時点ではネイティブ側は一切イベントを送出していなかった(将来のジョブ進捗プッシュ等のために用意されたプラミングのみ)が、**v5でこの機構が初めて実際に使われるようになった**——右クリックメニューからのアクション通知`timeline.menuInvoked`(§4.14参照)がその第一号で、以後`timeline.projectLoaded`も同機構を使う。

### 1.5 応答なし(no-op)

`id`も`method`も持たないオブジェクトを受け取った場合、`HandleRequestJson`は空文字列を返し**何も投稿しない**(`bridge_core.cpp:66-70`)。

## 2. スレッドモデル

`native/src/bridge.h`冒頭コメントと[SDK_REFERENCE.md §12](SDK_REFERENCE.md)が正式な出典。要約:

| スレッド | 役割 |
|---|---|
| **WebView2 UIスレッド**(≒ホストのメインスレッド) | `Bridge::HandleMessage`が呼ばれる場所。同期メソッド(`ping`/`getEditInfo`/`backend.getBaseUrl`/`timeline.insertMedia`/`settings.get`/`settings.set`)はここで完結して即座にレスポンス文字列を返す。`timeline.insertMedia`は`call_edit_section_param`(SDK)がメインスレッド実行を要求するため、あえて非同期化せずここで同期実行される。`ui.pickFile`もここで`GetOpenFileNameW`のモーダルポンプを回す(再入防止に`pick_dialog_open_`アトミックフラグを使用)。 |
| **HTTPワーカースレッドプール**(`HttpClient`、`http_->Post(...)`) | `backend.request` / `backend.downloadVideo` / `backend.uploadFile` / `timeline.captureFrame`(PNG符号化まで)/ `ui.makeThumbnail`(WICデコード〜JPEG符号化)がここで実行される。`HandleMessage`はこれらに対して即座に空文字列を返し(「今は投稿するものがない」)、ワーカー完了時に`ResponsePoster`(`Bridge::Initialize`で注入)経由でレスポンスJSONをUIスレッドへマーシャリングして`PostWebMessageAsJson`する。 |
| **レンダリング専用スレッド**(ホストが管理) | `EDIT_HANDLE::rendering_scene_video`のコールバック(`CaptureRenderCb`)がここで呼ばれる。バッファ(`PIXEL_RGBA`)はコールバック内でしか有効でないため、`timeline.captureFrame`の実装はコールバック内で**即座にピッチ考慮のメモリコピー**を行い、条件変数でHTTPワーカー側へ引き渡す(タイムアウト10秒、`CaptureFrameWorker`)。タイムアウト時はワーカーが`abandoned`フラグを立てて離脱し、遅れて届いたコールバックが自分で`delete`して後始末する設計。 |

マーシャリングの要点: **`ResponsePoster`はワーカースレッドから呼ばれるので、実装(webview_host側)はUIスレッドへの投げ直しを保証する責務を持つ**(`bridge.h`の`ResponsePoster`コメント: "the implementation must be thread-safe and marshal onto the UI thread before touching WebView2")。

## 3. メソッド一覧(概要)

| メソッド | 同期/非同期 | 実行スレッド | 契約バージョン |
|---|---|---|---|
| `ping` | 同期 | UI | v1 |
| `getEditInfo` | 同期 | UI | v1 |
| `backend.request` | 非同期 | HTTPワーカー | v2(timeoutMsはv4.1) |
| `backend.downloadVideo` | 非同期 | HTTPワーカー | v2 |
| `backend.uploadFile` | 非同期 | HTTPワーカー | v3 |
| `backend.getBaseUrl` | 同期 | UI | v2 |
| `timeline.insertMedia` | 同期(UIスレッド上で`call_edit_section_param`同期呼び出し) | UI | v2 |
| `timeline.captureFrame` | 非同期(内部でレンダリングスレッド→HTTPワーカー) | UI起点→レンダリング→HTTPワーカー | v3 |
| `ui.pickFile` | 同期(モーダルダイアログ) | UI | v3 |
| `ui.makeThumbnail` | 非同期 | HTTPワーカー | v3 |
| `settings.get` | 同期 | UI | v4 |
| `settings.set` | 同期 | UI | v4 |
| `timeline.getSelection` | 同期(`call_edit_section_param`) | UI | v5(§4.13) |
| `timeline.menuInvoked`(イベント、ネイティブ発) | — | UI | v5(§4.14) |
| `ui.pickFolder` | 同期(モーダルダイアログ、`IFileOpenDialog` + `FOS_PICKFOLDERS`) | UI | v6(§4.15) |
| `fs.listFiles` | 非同期(ディスク列挙+wavヘッダ読取) | HTTPワーカー | v6(§4.16) |
| `fs.probeAudioDuration` | 非同期 | HTTPワーカー | v6(§4.19) |
| `backend.downloadVideo`の`destDir`/`fileName`/`noClobber`/`reuseIfPresent`拡張 | 非同期(既存と同じ) | HTTPワーカー | v6(§4.20、`reuseIfPresent`は2026-07-18追加) |
| `ui.resolveDroppedFiles` | 同期(`HandleRequestJson`内の純粋な処理、I/O無し) | UI | v7(§4.21) |
| `ui.pickFile`の`"imageOrVideo"` kind拡張 | 同期(既存と同じ) | UI | v7(§4.9) |
| `fs.probeMediaInfo` | 同期(UIスレッド上で`call_edit_section_param`同期呼び出し) | UI | v9(§4.22) |
| `backend.uploadFile`の`query`拡張／応答`trimmed` | 非同期(既存と同じ) | HTTPワーカー | v10(§4.5) |

v6行の「実行スレッド」列は、2026-07-15のnative実装完了により実装コードで確認済みの値である。`ui.pickFolder`は`ui.pickFile`と同じくUIスレッド上でモーダルダイアログ(`IFileOpenDialog`、`shobjidl.h`)をポンプする。`fs.listFiles`/`fs.probeAudioDuration`はいずれも`backend.uploadFile`等と同様にHTTPワーカースレッドで実行される(`bridge_core.cpp`のコメントに明記)。`ui.resolveDroppedFiles`は`ping`/`getEditInfo`と同じ「`HandleRequestJson`内で完結する同期メソッド」に分類される — ディスク I/O もSDK呼び出しも無い純粋なJSONの読み替えだからである(実際のドロップファイル解決=`AdditionalObjects`の取得自体は、`HandleMessage`が`HandleRequestJson`を呼ぶ**前**に`webview_host.cpp`のWebMessageReceivedハンドラ側で完了している。§4.21参照)。

## 4. メソッド詳細

### 4.1 `ping`

- params: `{}`
- result: `{ pong: true, pluginVersion: string }`(`pluginVersion`は`kPluginVersion` = `"1.0.0-rc1"`)
- エラー: なし(常に成功)。

### 4.2 `getEditInfo`

- params: `{}`
- result: `{ width, height, rate, scale, sampleRate, frame }`(number)。AviUtl2 SDKの`EDIT_INFO`をミラー。
- エラー: `NO_EDIT_HANDLE`(編集ハンドル未取得、またはホストがまだ`EDIT_INFO`を返せない場合)。
- 実装は`layer` / `frameMax` / `layerMax`も追加でJSONに含める。**2026-07-21（Join復活・I4）でWebUI側の型`BridgeResultMap["getEditInfo"]`にもこの3つを宣言追記した**（Join UIが最前面レイヤー挿入に`layerMax`を使うため）。ネイティブは従来からこの3つを常に返しており、**契約バージョンのbumpは不要**（型のミラーを実装へ寄せただけ）。詳細は§9。

### 4.3 `backend.request`

LTX23バックエンドREST APIを1回プロキシする(WinHTTP、CORS回避)。

- params:
  - `method`: `"GET" | "POST" | "DELETE"`(必須)
  - `path`: `/api/v1`プレフィックスを含むフルパス、例 `/api/v1/status`(必須、`/`始まり)
  - `query?`: `Record<string,string>`(URLエンコードされクエリ文字列化)
  - `body?`: object(JSONとしてdumpされ`Content-Type: application/json`で送信。object/arrayのみ許可)
  - `timeoutMs?`: number(**v4.1**。`[1000, 600000]`にクランプ。省略時は既定30秒。数値以外を渡すと`BAD_REQUEST`)
- result: `{ status: number, body: object | null }`。**4xx/5xxもここでは正常結果**(reject されるのはトランスポート層の失敗のみ)。`body`は空/非JSONなら`null`。
- エラー: `BAD_REQUEST`(paramsパース失敗)、`BACKEND_UNREACHABLE`(接続不可/リセット)、`BACKEND_TIMEOUT`(WinHTTPタイムアウト)。
- 役割分担: `backend.request`は呼び出し側(WebUI)が`/api/v1`込みのフルパスを`path`に渡し、ネイティブ側`BuildBackendUrl()`は`base_url + req.path`をそのまま結合する(prefixを追加で付与しない)。一方`backend.downloadVideo`/`backend.uploadFile`はネイティブ側がjobId/kindからprefix無しの相対パスを組み立て、`kBackendApiPrefix`を明示的に付けてURLを作る(§4.4/§4.5)。

### 4.4 `backend.downloadVideo`

完成ジョブのmp4をネイティブ側にダウンロードし、ローカルパスを返す。

- params: `{ jobId: string, joined?: boolean }`。`jobId`に`/`・`\`・`..`が含まれると`BAD_REQUEST`。
- result: `{ filePath: string, sizeBytes: number }`
- 保存先: `%LOCALAPPDATA%\NzVideomni\downloads\<jobId>.mp4`(`joined:true`なら`<jobId>_joined.mp4`)。
- タイムアウト: 既定 **300秒**(`kDefaultDownloadTimeoutMs`、`timeoutMs`オーバーライド不可 — `backend.request`のみが対応)。
- エラー: `BAD_REQUEST`、`BACKEND_UNREACHABLE`、`BACKEND_TIMEOUT`、`DOWNLOAD_FAILED`(HTTP 200以外)。
- **contract v6での拡張(§4.20に詳細)**: `destDir?: string` / `fileName?: string` / `noClobber?: boolean`の3つが追加された。全て省略可能で、省略時はv6以前と完全に同じ挙動(上記の保存先固定パス)になる後方互換の拡張。

### 4.5 `backend.uploadFile`

ローカルファイルをバックエンドのmultipartアップロードエンドポイントへ転送する。

- params: `{ kind: "image" | "video" | "audio", filePath: string, query?: Record<string,string> }`(`query`は**contract v10**。下記)
- result: `{ status: number, body: object | null }`(`backend.request`と同じ「4xx/5xxも正常結果」規約)
- 内部: `UploadPath(kind)` = `/upload/<kind>`、`kBackendApiPrefix`と結合。`Content-Type`は拡張子から推定(`ContentTypeForExtension`: png/jpg/jpeg/webp/mp4/mov/webm/mkv/wav/mp3/m4a/aac/flac/ogg、それ以外は`application/octet-stream`)。
- タイムアウト: 既定 **120秒**(`kUploadTimeoutMs`、`backend.request`と異なりオーバーライド不可)。
- エラー: `BAD_REQUEST`、`FILE_NOT_FOUND`(送信前にネイティブ側で`GetFileAttributesW`チェック)、`BACKEND_UNREACHABLE`、`BACKEND_TIMEOUT`。

**contract v10での拡張(2026-07-30、V2Vリボン範囲トリム)**

- `query?: Record<string, string>`: アップロードURLへ付けるクエリパラメータ。値は文字列のみ(数値・真偽値・null・入れ子オブジェクトは`BAD_REQUEST`)。`backend.request`が既に持っていた`query`と**同じ形・同じ実装を共有**する。
- **省略時の等価性(契約上の保証)**: `query`が無い/`null`/空オブジェクトのとき、組み立てられるURLは**v10以前と完全に同一**になる。これはnative側で`AppendQueryToUrl`が「空クエリならURLをそのまま返す」1箇所に集約されているため構造的に保証されている(`bridge_core.cpp`)。WebUI側も`timeline/sourceTrim.ts`の`trimQuery()`が無トリム時に`undefined`を返し、呼び出し側が`...(query ? { query } : {})`でキーごと落とす作りで、`query: undefined`を送ることがない。
- **nativeの実装関数名**: パースと文字列化は`bridge_core.h/.cpp`の共有純関数 **`BuildQueryString(value, out, err_message)`**(JSONオブジェクト→RFC 3986のパーセントエンコード済み`"a=b&c=d"`。失敗時はメッセージを返し、呼び出し側が自分のメソッド名を前置する)と **`AppendQueryToUrl(url, query)`**(空クエリならURL不変)。`ParseUploadFile`(`bridge_core.cpp`)が`UploadFileRequest.query`(`std::string`。エンコード済みの`"a=b&c=d"`、または空)へ詰め、`bridge.cpp`の`backend.uploadFile`分岐が`AppendQueryToUrl(CurrentBaseUrl() + kBackendApiPrefix + UploadPath(ureq.kind), ureq.query)`でURLを作る。`out->query`は毎回`clear()`されるため、リクエストオブジェクトを再利用しても前回のクエリが漏れない。`HttpClient::UploadFileAsync`・`http_client`側は無改修(`CrackUrl`がクエリをパスへ再結合するため)。なお`BuildQueryString`/`AppendQueryToUrl`は`ParseBackendRequest`／`BuildBackendUrl`の既存実装から切り出したもので、**エラー文字列は1バイトも変えていない**ため既存doctestがそのまま回帰テストとして機能する。
- **応答の`trimmed`**: 今日の唯一の利用者は`POST /upload/video`の`trim_start_sec`/`trim_duration_sec`([`API_REFERENCE.md`](API_REFERENCE.md) §3.10)で、`UploadVideoResponse`に`trimmed: boolean`が加わっている。ブリッジ自身は`body`をそのまま透過するだけなので契約としては`result`の形は変わらないが、`types.ts`の`UploadVideoResponseBody`に任意フィールドとして宣言してある。**トリムを要求したのに`trimmed`が`true`で返らなかった場合、WebUIはGenerateを止める**(`useChainForm`の理由コード`sourceTrimFailed`)——`video_id`自体は使えるが指しているのは元動画全体であり、誤った区間から生成するより止めるほうが安全という判断。
- **transport自体はトリム非依存**: `query`は意図的に汎用で、トリム専用の意味づけをnative側に持たせていない。将来ほかのアップロードエンドポイントがクエリを要求しても、この口をそのまま使える。
- **応答の`frame_count`／`fps`(2026-08-16、素材（末尾）＝End source)**: `POST /upload/video`の応答に実測のフレーム数とフレームレートが加わった([`API_REFERENCE.md`](API_REFERENCE.md) §3.10)。`trimmed`と同じく**ブリッジは`body`を透過するだけで契約の形は変わらず**、`types.ts`の`UploadVideoResponseBody`へ任意フィールド(`number | null`)として宣言してあるだけである。nativeは無改修。WebUIはこの2値から素材（末尾）の帯の長さを決め、取れなかったときは既知の尺からの推定→長さ不明としてブロック、と段階的に退避する。同じ機能で使う`max_frames`クエリも、上記の汎用`query`をそのまま使っている(**この口を作っておいたことの2つ目の配当**)。
- **逆順Chained（End source第2段階バッチ2、2026-08-18）はブリッジ契約に影響しない**: 複数クリップへの拡張はサーバー側の生成方式（`chain_math`のモード判定・Stage-1の生成順）が変わっただけで、アップロード応答の使い道（帯の長さ決定）も系統E（末尾合わせ）の純関数群（`AppShell.tsx`・`ChainedScreen.tsx`）も無改修のままである。nativeは1行も変わっていない。
- doctest: `native/tests/test_bridge_core.cpp`に9ケース追加(258→**267 pass/6 skip**。実測は`build/ninja-release/NzVideomni_tests.exe --test-suite-exclude=integration`)。
- **`timeline.getSelection`のv10追加フィールド(再生位置系)は2026-08-01に収録済み**: トリムの判定には、リボンが元動画のどこから再生を始めているか(AviUtl2の`動画ファイル`エフェクトの`再生位置`項目)が必要である。この項目の**単位**(秒／プロジェクトfpsのフレーム／ソースfpsのフレーム)はSDKのどこにも書かれておらず実機で1回採取する必要があったが、2026-08-01の採取・解析で**素材時間軸の秒**と確定した。これを受けて`playbackStartSec`／`playbackEndSec`／`hasPlaybackRange`／`playbackSpeed`／`loopPlay`／`sectionCount`の6フィールドを`bridge/types.ts`の`timeline.getSelection`結果へ載せ、本書§4.13へ追記した(`webui/src/timeline/sourceTrim.ts`のローカル型に暫定的に置いていた4フィールドはこれに伴い廃止し、同型は契約型の再エクスポートになった)。詳細な確定事実は[`V2V_RIBBON_TRIM_WORKORDER.md`](V2V_RIBBON_TRIM_WORKORDER.md) §4を参照。**実機ゲートは2026-08-01に全5項目合格し、テーマはクローズ済みである**([`REAL_BACKEND_CHECKLIST.md`](REAL_BACKEND_CHECKLIST.md) §4.12、[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-59)。同日、この6フィールドの**2つ目の消費者としてIC-LoRA参照動画のアップロード経路**が加わったが、契約側の変更は無い(`decideSourceTrim`／`trimQuery`という既存の純関数を`CreateScreen.tsx`から呼ぶだけで完結した)。
- **注記(2026-09-04、§3-140の副作用。契約は不変)**: 🎞挿入がエイリアス方式になり`再生位置`を明示するようになったため(§4.7)、**🎞で置いたオブジェクトのリボンを手で詰めてからV2Vへ送ると、これまでの全尺送信ではなくトリム送信になる**。ドラッグ＆ドロップで置いたオブジェクトの挙動と一致させる方向の変化であり、`decideSourceTrim`のロジックも本書の契約も1文字も変えていない(トリムしていない場合は`spanCoversWholeMedia`のゲートで従来どおり全尺のまま)。経緯は[`DEVLOG.md`](DEVLOG.md) §107。

### 4.6 `backend.getBaseUrl`

- params: `{}`
- result: `{ baseUrl: string }`(直接`<video src>`/`<img src>`を組み立てる用途。`settings.get`と値は同じソースだが用途が異なる — §4.11参照)
- エラー: なし。

### 4.7 `timeline.insertMedia`

- params: `{ filePath: string, layer?: number, frame?: number }`(`layer`/`frame`省略時は現在の編集カーソル位置)
- result: `{ inserted: true, layer: number, frame: number }`(実際に使われた解決後の値)
- 実装: `call_edit_section_param`のコールバック内で、メディア生成を一手に引き受けるnative側のヘルパ`CreateMediaObject`(`native/src/bridge.cpp`)を呼ぶ(メインスレッド・更新ロック下、**同期**)。同ヘルパは`get_media_info`1回で素材の実尺と音声トラックの有無を採り、尺は`round(総時間秒 × プロジェクトfps)`で決める(この式は§3-140の前後で変わっていない)。**動画かつ尺が1フレーム以上なら、ドラッグ＆ドロップ相当のエイリアス(`音声付き`キー込み)を組み立てて`create_object_from_alias`へ渡す**——`create_object_from_media_file`は`音声付き`を立てず、できたオブジェクトが青一色になり右クリックの「音声を分離」も出ないためである(§3-140)。静止画・音声のみ・素材情報が読めない・尺が0フレームへ丸まる・パスにCR/LFを含む、のいずれかなら**従来どおり`create_object_from_media_file(file, layer, frame, length)`**を呼ぶ。エイリアス経路が`nullptr`を返した場合と、できたオブジェクトが要求尺より短かった場合(deleteしてから)も同APIへ退避する(**三重の安全網**)。詳細は[`DEVLOG.md`](DEVLOG.md) §107、SDK側の確定知見は[`SDK_REFERENCE.md`](SDK_REFERENCE.md) §16(g)(h)。
- エラー: `BAD_REQUEST`(`filePath`欠落/空)、`FILE_NOT_FOUND`(送信前チェック)、`NO_EDIT_HANDLE`(編集ハンドル未取得、または`call_edit_section_param`が実行されなかった)、`INSERT_FAILED`(**エイリアス経路と従来API経路の両方が失敗**して`nullptr`しか得られなかった — 非対応形式や位置重複。メッセージは§3-140で経路中立な`media object creation failed (unsupported format or overlapping object)`へ改めた)。

### 4.8 `timeline.captureFrame`

現在(または指定)のタイムラインフレームをPNGとしてディスクに書き出す(I2V用の条件画像取得)。

- params: `{ frame?: number }`(省略時は`get_edit_info().frame`)
- result: `{ filePath: string, width: number, height: number, frame: number }`
- 保存先: `%LOCALAPPDATA%\NzVideomni\captures\frame_<frame>_<yyyymmdd_HHMMSS_fff_nnn>.png`
- 内部フロー: `rendering_scene_video(frame, ...)`でレンダリングをキュー→レンダリングスレッドのコールバックで`PIXEL_RGBA`をピッチ考慮でコピー→HTTPワーカーへ引き渡し→WIC PNGエンコード(RGBA→BGRA変換、§ [DEVLOG.md](DEVLOG.md)参照)。
- タイムアウト: レンダリングコールバック到着待ちは**10秒**(ハードコード、`timeoutMs`パラメータ非対応)。
- エラー: `NO_EDIT_HANDLE`(編集ハンドルまたは`rendering_scene_video`関数ポインタが無い)、`BAD_REQUEST`(`frame`が整数でない)、`CAPTURE_FAILED`(`rendering_scene_video`が`false`を返す/コールバックが10秒以内に届かない/バッファが空・短い/PNGエンコード失敗)。

### 4.9 `ui.pickFile`

- params: `{ kind: "image" | "video" | "audio" | "imageOrVideo" }`
  - **`"imageOrVideo"`(contract v7として本書に追補。ネイティブ実装自体はChainのソース入力欄一本化セッションで先行して完了していた)**: Chainの統合ソース入力欄(`SourceInputPanel.tsx`)専用。単一のダイアログで画像・動画どちらも選べる組み合わせフィルタを先頭に出し(`bridge_core.cpp`の`PickFileFilter`)、呼び出し側(`modes/chain/sourceRouting.ts`の`routeSourceByExtension`)が選ばれたファイルの拡張子から事後的に画像/動画のどちらとして扱うかを振り分ける — ダイアログ自身が事前にkindを決め打ちしない設計。
- result: `{ filePath: string, fileName: string }`
- 実装: `GetOpenFileNameW`のネイティブOpenダイアログ(UIスレッド上でモーダルポンプ、フィルタはバックエンドの許可拡張子と一致 — 画像`png/jpg/jpeg/webp`、動画`mp4/mov/webm/mkv`、音声`wav/mp3/m4a/aac/flac/ogg` + 「すべてのファイル」。`"imageOrVideo"`は画像+動画の結合フィルタ→画像単体→動画単体→すべてのファイル、の順)。
- 再入防止: ダイアログが既に開いている間に2件目の`ui.pickFile`が来ると`DIALOG_FAILED`(`pick_dialog_open_`アトミックフラグ)。
- エラー: `BAD_REQUEST`(`kind`不正)、`CANCELLED`(ユーザーがキャンセル — UIでエラー扱いしない)、`DIALOG_FAILED`(ダイアログ自体の失敗、または再入)。
- **WebUI側ローカル待機上限なし**: モーダルダイアログはユーザーが閉じるまで同期的に待つのが正しい動作であり「タイムアウト」という概念が無い。`RequestDispatcher`は`ui.pickFile`に対してローカルタイマーを張らない(`NO_LOCAL_TIMEOUT_METHODS`、§6参照)。ユーザーがダイアログで10秒以上迷っても、WebUI側が先に諦めて選択結果を握りつぶすことはない。

### 4.10 `ui.makeThumbnail`

ローカル画像をダウンスケールしてdata URLとして返す(キーフレームカードのサムネイル用)。

- params: `{ filePath: string, maxDim?: number }`(`maxDim`既定256、`[16, 1024]`にクランプ)
- result: `{ dataUrl: string, width: number, height: number, sourceWidth: number, sourceHeight: number }`(`dataUrl`は`data:image/jpeg;base64,...`)
- サイズ計算: 長辺が`maxDim`以下なら等倍(アップスケールしない)。それ以外は長辺を`maxDim`に合わせ、短辺を四捨五入(最低1px)。
- エラー: `BAD_REQUEST`(`filePath`欠落/`maxDim`が数値でない)、`FILE_NOT_FOUND`(送信前チェック)、`THUMBNAIL_FAILED`(WICデコード/リサイズ/JPEGエンコード失敗)。

### 4.11 `settings.get`

- params: `{}`
- result: `{ baseUrl: string }`(`SettingsStore`の現在値。ストア未初期化時は組み込み既定 `http://127.0.0.1:18620`)
- `backend.getBaseUrl`との違い: `getBaseUrl`は`<video>/<img>`の直接参照URL構築用、`settings.get`は接続設定パネルの表示/編集用。値のソースは同じ(`Bridge::CurrentBaseUrl()`)。

### 4.12 `settings.set`

- params: `{ baseUrl?: string }`。省略または`null`は「読むだけ」の no-op(現在値をそのまま返す)。
- result: `{ baseUrl: string }`(正規化後の値。末尾`/`除去等は`NormalizeBaseUrl`参照)
- バリデーション(`NormalizeBaseUrl`): `http://`または`https://`で始まること(大小文字区別)、スキーム後のホスト部が非空であること。フルRFC 3986パースではない軽量チェック。
- 永続化: 検証成功時、`%LOCALAPPDATA%\NzVideomni\settings.json`へベストエフォートで書き込み(書き込み失敗はログのみでAPI呼び出し自体は成功扱い)。
- エラー: `BAD_REQUEST`(`baseUrl`が文字列でない、正規化失敗、または設定ストア未初期化)。

### 4.13 `timeline.getSelection`(contract v5)

現在のタイムライン選択/カーソル状態のスナップショットを返す。右クリック生成の配線(`useMenuRouter`→`deriveGenerationParams`)が、切り出し範囲や生成解像度の決定に使う。

- params: `{}`
- result:
  - `hasRange: boolean` / `rangeStart: number` / `rangeEnd: number` — フレーム範囲選択(`EDIT_INFO.select_range_start/end`起源。未選択なら`hasRange:false`)。
  - `selected: Array<{ layer, frameStart, frameEnd, effectName, filePath: string|null, objectName: string|null, mediaWidth: number, mediaHeight: number }>` — 選択中の各タイムラインオブジェクト。`filePath`/`objectName`は取得できない場合`null`。
    - **`frameStart`/`frameEnd`の端点解釈は突き合わせ未了である(2026-09-04時点・未決)**: ネイティブは`OBJECT_LAYER_FRAME`の`start`/`end`を**素通しで**載せている(`bridge.cpp`の`GetSelectionEditProc`)。その`end`は2026-09-04の実機ログから**排他**(次フレームの先頭を指す)と確定した([SDK_REFERENCE.md](SDK_REFERENCE.md) §16 (h)が正本)。一方でwebui側は`frameEnd`を**包含**と読み、リボン長を`frameEnd - frameStart + 1`で算出している(`timeline/prefillSeed.ts`・`jobs/fpsConvert.ts`・`timeline/retakeWindow.ts`)。この「包含」の来歴には注意が要る——[DEVLOG.md](DEVLOG.md) §44.4は「ネイティブ現物で`frameEnd`はinclusiveと確定した」と記録しているが、**そこで確認された現物は`next = end + 1`という書き方であって、`end`自体の端点を測ったものではない**（`end + 1`が次の先頭になるのは`end`が包含のときだけ、という読み方に依存している）。つまり§44.4の「確定」は実測ではなく書き方からの推定であり、上記の実機ログによる排他確定と正面から衝突する。**両者の突き合わせは未了で、コードは1行も変えていない**——読むときは2つの解釈が同居している前提で読むこと。
    - **`mediaWidth` / `mediaHeight`(2026-07-08追加)**: 選択オブジェクトが参照するメディアファイルの実解像度(整数)。ネイティブの`GetSelectionEditProc`(`bridge.cpp`)が、選択オブジェクトのファイルパスに対して`EDIT_SECTION::get_media_info`([SDK_REFERENCE.md](SDK_REFERENCE.md) §5.1)を呼び出して取得する。**解決できない場合(音声/図形オブジェクト、`get_media_info`未提供、失敗時)は`0`を返す(`null`にはしない)** — 「不明」は常に`0`として表現される。右クリック生成時の出力解像度を入力素材に合わせるためのseam(`webui/src/timeline/deriveGenerationParams.ts`)がこの値を最優先で参照する。
  - `cursorFrame` / `cursorLayer` — 編集カーソル位置。
  - `rate` / `scale` / `sampleRate` — `getEditInfo`と同じプロジェクトのフレームレート/サンプルレート(frame⇄time換算用)。
- エラー: `NO_EDIT_HANDLE`(編集ハンドル未取得、または`call_edit_section_param`が実行されなかった場合)。
- 実装注記: `get_media_info`の実際の挙動(取得不能ケースの扱いを含む)は**実機で確定済み**(2026-07-19 G1実機ゲート。プロジェクト解像度と異なるpngで実寸一致を確認。[TIMELINE_ALPHA_REQUIREMENTS.md](TIMELINE_ALPHA_REQUIREMENTS.md)第6節・[SDK_REFERENCE.md](SDK_REFERENCE.md) §16(a)〜)。コード上の`// REALDEVICE-VERIFY:`コメントはこのゲート消化に伴い整理対象。
- **拡張(2026-07-19〜20、右クリック再設計)**: `selected[]`の各要素に**`textContent: string|null`**(テキストオブジェクトの本文。`GetSelectionEditProc`が`get_object_item_value("テキスト","テキスト")`で取得。テキスト以外は`null`)と**`mediaDurationSec: number`**(選択メディアの実尺。長さガードに使用)が追加された。実装済みだが本書は詳細をSPEC側へ委譲する(下記§4.14の注記参照)。

- **拡張(2026-08-01、contract v10のもう半分＝再生位置系6フィールド)**: `selected[]`の各要素に、V2Vリボン範囲トリム(台帳[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-59。起票時は`PENDING_TASKS.md` §1-6)の判定に必要な6フィールドが追加された。すべて`動画ファイル`エフェクトのオブジェクトについてのみ意味を持ち、それ以外は既定値のまま返る。

  | フィールド | 型 | 意味 |
  | --- | --- | --- |
  | `playbackStartSec` | `number` | リボンが再生している**素材ファイル側の開始位置(秒)**。 |
  | `playbackEndSec` | `number` | 同・**終了位置(秒)**。 |
  | `hasPlaybackRange` | `boolean` | 上2つが実際に読めたかどうか。`false`なら値は無意味。 |
  | `playbackSpeed` | `number` | AviUtl2の`再生速度`を**1.0スケールに正規化**した値(生値は`"100.00"`のような百分率文字列で、nativeが100で割る)。取得できないときは`1.0`。 |
  | `loopPlay` | `boolean` | AviUtl2の`ループ再生`項目(生値`"0"`/`"1"`)。取得できないときは`false`。 |
  | `sectionCount` | `number` | 中間点で区切られた区間数(`get_object_section_num`)。取得できないときは`1`。 |

  - **単位は秒で確定済み(2026-08-01の実機調査)**。`再生位置`項目の生値は`開始,終了,再生範囲,0`という**4フィールドのCSV**で(例: `"2.000,10.700,再生範囲,0"`)、先頭2つが**素材の時間軸上の秒**(小数3桁固定・ピリオド小数点)である。**プロジェクトのフレームレートには依存しない**(同じ「頭を2秒削る」操作が30fpsでも24fpsでも`2.000`になることを実測)。無加工のリボンでも`0.000,<素材全長>`が省略されずに明示的に入る。第3フィールドは日本語のモード名なので、nativeのパーサ(`ParsePlaybackRange`、`bridge_core.cpp`)は**フィールド数にもモード名にも依存せず先頭2つの数値だけを読む**。
  - **`get_object_track_value`は使えない**: このAPIは`再生位置`について**常に第1値しか返さない**(実機で確認)。第2値はトラックAPIからは取得不能なため、生文字列のパース一択である。
  - **nullableにしない**: `mediaWidth`/`mediaHeight`と同じく「非nullable＋明示のhasフラグ」方式を採る。`0.0`は「先頭から再生」という正当な値でもあるため、値そのものでは「読めなかった」を表現できないからである。
  - **`types.ts`側では6つとも任意フィールド(`?`)として宣言している**。これは**古いプラグインが返さない場合にWebUIが安全側(＝トリムせず全体をアップロード)へ倒れる**ためだけの措置で、現行のnativeは常に6つとも返す。
  - 消費側は`webui/src/timeline/sourceTrim.ts`の`decideSourceTrim`のみ。切り出し尺は**`min(リボンの秒数, playbackEndSec − playbackStartSec)`**で決める(両方向のはみ出しを潰すため。理由は[`V2V_RIBBON_TRIM_WORKORDER.md`](V2V_RIBBON_TRIM_WORKORDER.md) §4を参照)。

- **拡張(2026-09-01、contract v11＝素材fps)**: `selected[]`の各要素に、右クリックプリフィルのfps軸「素材に合わせる」を成立させるためのフィールドが1つ加わった。台帳の記録は[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-13-02(2026-09-01にオーナーの実機ゲート合格でクローズ。起票時は[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md)§3-13で、実機ゲート待ちの間は同書§2-5。どちらも同書では欠番)。`動画ファイル`エフェクトのオブジェクトについてのみ意味を持ち、それ以外は既定値のまま返る。

  | フィールド | 型 | 意味 |
  | --- | --- | --- |
  | `mediaFps` | `number` | 素材そのもののフレームレート。Media Foundationが返す`MF_MT_FRAME_RATE`(分子/分母の`UINT32`対)をdoubleへ直した値で、**取得できないときは`0`**。 |

  - **nullableにしない**: `mediaWidth`/`mediaHeight`とまったく同じ規約で、「不明」は常に`0`として表現する(`null`は返さない)。fpsに`0`という正当な値は存在しないため、値そのもので「読めなかった」を表現できる。
  - **nativeが返すのは生値で、整数へのスナップはwebui側で行う**。29.97(30000/1001)は`29.97…`のまま返り、`29.97→30`・`23.976→24`という丸めは`webui/src/modes/single/paramUtils.ts`の`snapFrameRate`(`Math.round`のあと`[1, 60]`へクランプ。台帳§3-71/§3-72対策で、素材段だけでなくfpsが決まりうる全入口の丸めを担う正本へ統合された)が担当する。`media*`は「素材由来の事実」を運ぶ枠であって方針を混ぜないため、という規約上の理由による分担である。
  - **mkv/webmでの取得失敗は正常系である**。AviUtl2はbeta10でMedia Foundationのファイルリーダーを外しL-SMASH Worksが標準構成になったので、**「AviUtl2が読めるファイル」と「Media Foundationがfpsを答えられるファイル」は同じ集合ではない**(mp4/movは確実、mkv/webmは不確実)。読めなければ`0`が返り、webuiはプロジェクトfpsへ落ちる——これは異常ではないのでnative側はログにも出さない。
  - **`types.ts`側で任意フィールド(`?`)にしているのは、旧nativeビルドとの互換のためだけ**である(v10の再生位置系6フィールドとまったく同じ理由)。現行のnativeは常に返す。
  - 消費側は`prefillSeed.ts`のfps3段決定だけである——①fps軸が`material`なら`snapFrameRate(mediaFps)`(`paramUtils.ts`が正本)、②読めなければ選択のプロジェクトfps(`rate`/`scale`、これも同じ`snapFrameRate`で丸める)、③それも無ければconfig既定、の順に落ちる。
  - native実装は新規モジュール`native/src/media_fps_probe.{h,cpp}`(`MFCreateSourceReaderFromURL`→`GetNativeMediaType`→`MFGetAttributeRatio`。**最初の映像ストリームのネイティブ型を読むだけでデコーダを作らず1フレームも復号しない**ので、コストはファイル尺に比例しない)と、`bridge.cpp`の`GetSelectionEditProc`が動画エフェクト限定ブロック内から呼ぶ配線からなる。`MFStartup`は`std::call_once`でプロセス1回・対応する`MFShutdown`は意図的に呼ばない(プラグインの寿命＝プロセスの寿命であり、毎クリックのStartup/Shutdown往復と`Mp4Writer`稼働中の参照カウント落ちを同時に避けるため。理由はヘッダのコメントが正本)。COM初期化はUIスレッドがSTAなので`COINIT_APARTMENTTHREADED`を要求し、`S_FALSE`と`RPC_E_CHANGED_MODE`のどちらも失敗として扱わない。

### 4.14 `timeline.menuInvoked`(イベント、ネイティブ→WebUI、contract v5)

右クリックメニュー(タイムラインオブジェクト/レイヤーメニュー)からアクションが選ばれた際に、ネイティブがプッシュするイベント(`§1.4`のイベント機構を実際に使う初めてのケース)。

```json
{ "event": "timeline.menuInvoked", "data": { "action": "...", "selection": { /* timeline.getSelectionと同じ形 */ } } }
```

- `data.action: string` — 右クリックで選ばれたアクション種別(例: 続き生成/img2vid/IC-LoRA等。値の一覧は`webui/src/timeline/menuRouting.ts`)。
- `data.selection` — イベント発火時点の`timeline.getSelection`結果のスナップショット(`mediaWidth`/`mediaHeight`を含む)。WebUI側で再取得不要。
- 消費側: `webui/src/timeline/useMenuRouter.ts`がこのイベントを購読して`RoutedMenuCommand`へルーティングし、`webui/src/shell/AppShell.tsx`の`onRoute`が実際にモード切替＋intentプリフィルを行う(2026-07-08の右クリック配線実装で新規に結線。以前は`useMenuRouter`を消費するコンポーネントが存在しなかった)。詳細は[DEVLOG.md](DEVLOG.md) §7参照。
- **注記(2026-07-19〜20、右クリック再設計)**: `data.action`の値集合（メニュー項目）は再設計で刷新された。素材種別ごとの新メニュー（`referenceVideo`／`videoAudioToVideo`／`audioToVideo`／`appendText`ほか）に変わり、旧「全10アクション」「`sendToChain`（生成画面へ送る）」は廃止・再編されている。現行の項目一覧・振る舞いの唯一の正は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md)（第9版）を参照。
- **注記(2026-07-20、第2段階でのアクションID追加)**: レイヤーメニュー（`kLayerMenuItems`）に`addCurrentFrameAsKeyframe`（現在フレームをキーフレームに追加）・`currentFrameToClipChain`（現在フレームから長尺動画を生成、i2v Clip Chain）の2件が加わり、`data.action`の値集合は全12種（オブジェクトメニュー8種＋レイヤーメニュー4種）になった。**この追加は`timeline.menuInvoked`イベント自体のRPC形状（`data.action: string` / `data.selection`の各フィールド）を一切変えていない**——新しい`action`文字列が2つ増えただけで、ペイロードのスキーマは§4.13/§4.14に記載のとおり不変である。そのため契約バージョンは**v8のまま**据え置いている。あわせてネイティブ側`SnapshotSelection`に`use_mouse_frame`という真偽値引数が新設されたが、これは`timeline.menuInvoked`が送る`data.selection`の**値の決め方**（カメラ系3項目のみ`cursorFrame`を右クリック位置ではなく`EDIT_INFO`のカーソル値のまま据え置く）を変えるだけで、`data.selection`が持つフィールド自体の増減は無い。項目の詳細・配置基準は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §3-6・§5-1を参照。
- **注記(2026-07-22、フロントエンド微調整バッチでのアクションID追加)**: オブジェクトメニュー（`kObjectMenuItems`）に`insertProvisionalResult`（⬇ この生成結果を今すぐ挿入）、レイヤーメニュー（`kLayerMenuItems`）に`insertLatestResultHere`（⬇ 最新の生成結果をここに挿入）の2件が加わり、`data.action`の値集合はオブジェクトメニュー9種＋レイヤーメニュー5種の**全14種**になった。**この追加も`timeline.menuInvoked`イベントのRPC形状（`data.action: string` / `data.selection`）を一切変えていない**——新しい`action`文字列が2つ増えただけである。両アクションとも新規RPCを伴わず、WebUI側でジョブ台帳の逆引きと既存RPC（`timeline.insertMediaForJob`〔#13＝仮オブジェクトの🎞置換挿入〕／`timeline.insertMedia`〔#14＝位置指定の素挿入〕）を組み合わせて処理する。そのため契約バージョンは**v9のまま**据え置いている。項目の詳細・振る舞いは[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §3-4・§3-6を参照。
- **注記(2026-07-22、フロントエンド微調整バッチ第2波〔X1〜X6〕は契約変更なし)**: 第1波に続く第2波（[`DEVLOG.md`](DEVLOG.md) §47）はwebuiのみの改修で、**ブリッジ契約には一切変更が無い（v9のまま）**。予約詰まりの修正（X2）で送信失敗時に予約席とプレースホルダを掃除する処理は、既存の`deleteProvisionalByJob`（§4.14.1、v8）の再利用であり、新規メソッドもparams拡張も伴わない。予約席の再構築を台帳認識（reconcile）方式へ変えた変更（X2b）は、ネイティブへ渡すRPCを増やさないWebUI内部のロジックである。その他のX項目（FPS軸のグレーアウト・短クリップのクライアント側ゲート・右クリックv2vの1クリップ化・併用プリフィルのアコーディオン挙動・Settingsラベル）もすべてWebUI内で完結する。
- **注記(2026-08-09、W0〔共通スパイン〕でのアクションID追加)**: オブジェクトメニューに`outpaintVideo`（🎬 この動画に描き足す／Outpainting）・`retakeRange`（🎬 選択範囲を撮り直す／Retake）の2件が加わり、`data.action`の値集合は**全16種（オブジェクトメニュー11種＋レイヤーメニュー5種）**になった。**この追加も`timeline.menuInvoked`イベントのRPC形状を一切変えていない**——新しい`action`文字列が2つ増えただけで、契約バージョンは**v9のまま**据え置いている。項目の詳細は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §3-4・§3-6を参照。
- **注記(2026-08-10、長尺A2Vでのアクション追加／2026-08-11、長尺IC-LoRAでのアクションID追加。記録漏れの遡及補完)**: オブジェクトメニューに、2026-08-10の長尺A2V（台帳[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-74）で`videoAudioToLongA2v`（🎬 この動画の音声でlong a2v）・`audioToLongA2v`（🎵 この音声からlong a2v）の2件、続けて2026-08-11の長尺IC-LoRA（同§3-78）で`referenceVideoChain`（🎬 この動画を参照に長尺IC-LoRA生成／Chained）の1件が加わった。いずれの追加も`timeline.menuInvoked`イベントのRPC形状を一切変えておらず、`action`文字列が増えただけで契約バージョンは**v9のまま**据え置いている。この2回の追加は当時ここに記録されないまま実装・デプロイされていたため、`data.action`の値集合は本注記の時点で**全19種（オブジェクトメニュー14種＋レイヤーメニュー5種）**まで進んでいたことをあわせて補記する。項目の詳細は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §3-4・§3-6を参照。
- **注記(2026-08-16、End sourceでのアクションID追加)**: endWithThis（🎬🖼 これで終わる動画を作る）が加わり、data.action は全20種（オブジェクト15種＋レイヤー5種）。RPC形状は不変で契約バージョンはv10のまま。

### 4.14.1 右クリック再設計(2026-07-19〜20、contract v8)で新設/拡張されたRPC群(注記・本書は詳細をSPECへ委譲)

タイムライン右クリック再設計の第1段階で、仮オブジェクトのライフサイクルを担う**新設ネイティブRPCと既存RPCの契約変更**が加わった。**いずれも`webui/src/bridge/types.ts`とnative(`bridge_core.*`／`bridge.cpp`)に実装済み**だが、本書は§0冒頭のv5ドキュメント債務の方針に倣い、詳細な引数・応答形式の収録は行わず、下記の要点と一次ソースへのポインタにとどめる（**現行の唯一の正は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §5、契約型は`types.ts`**）。

- **`timeline.updateProvisionalReservation`（新設・SPEC §5-3）**: 仮オブジェクトの「削除＋付け替え」を1編集セクション（アンドゥ1件）で原子的に行う。用途は(a)予約移動、(b)旧予約の削除、(c)自己修復（旧予約が見つからなければcreateのみ）。位置衝突時はnative側で`EDIT_INFO.layer_max`を参照し`layer_max+1`へフォールバックcreate。
- **`timeline.insertMediaForJob`（新設・SPEC §5-10）**: 完成動画の🎞挿入を、仮オブジェクトのマーカー位置（レイヤー・フレーム）へ**位置連動で置換**する（現在選択レイヤー・カーソル位置は無視）。1編集セクションで delete＋create（明示尺）を行いアンドゥ1件にまとめる。ロールバック用の機能フラグ`REPLACE_INSERT_ENABLED`（webui側）で旧経路＝`timeline.insertMedia`＋`timeline.deleteProvisionalByJob`へ即時退避できる。**注記（2026-09-04、§3-140）**: このRPCが使う生成プリミティブは`timeline.insertMedia`と共通の`CreateMediaObject`ヘルパへ集約され、動画は`create_object_from_alias`（`音声付き`込み）経由になった（§4.7の実装欄が正本。RPCの引数・応答は不変で、契約バージョンも据え置き）。
- **`timeline.deleteProvisionalByJob`**: jobID指定で仮オブジェクトを逆引き削除する（旧✅マーカー自動削除経路。上記ロールバック経路で使用）。
- **`timeline.insertProvisional`の契約変更**: 長さ指定が旧`lengthFrames`から**`numFrames`（生成フレーム数）＋`genFps`（生成フレームレート）**へ変更された。尺換算はwebuiでは行わず、native側が`ProjectFramesForPixels()`（`timeline_math.h`）でプロジェクトfps基準へ換算する。あわせて配置系統パラメータ（(A)素材直後／(B)素材と同開始で`layer_max+1`／(C)右クリック位置）と`textPrefix`（状況表示テキストの接頭辞）が加わった。`updateProvisionalReservation`も同じ`numFrames`／`genFps`／配置系統を受ける。
- **`timeline.projectLoaded`（イベント）**: §0冒頭2026-07-08注記のとおり本書は未収録のまま（既知の債務）。孤児（セッションまたぎ）仮オブジェクトの予約検出に使う。検出はテキスト本文の`[#id]`マーカーを第一手段、`object_name`の`NzVideomni#<id>`を名前フォールバックとする二重化設計（SPEC §5-7）。

### 4.15〜4.20: contract v6(バッチA2V + fsブリッジ、native実装完了)

以下6項目はバッチA2V(CSVマニフェスト駆動の一括生成)とwav長自動調整機能のために追加された。§0冒頭の注記のとおり、起票時点(2026-07-15の前半)はnative未実装だったが、**同日のフロントエンド追随グループ2セッションでnative(`native/src/bridge.cpp`・`bridge_core.cpp`・新規`wav_probe.h/.cpp`・`fs_util.h/.cpp`)を実装した**。以下は実装コード(`native/src/bridge.cpp`の各`*Worker`関数・`bridge_core.cpp`の`Parse*`関数)を突き合わせて確認済みの内容である。

### 4.15 `ui.pickFolder`(contract v6)

ネイティブのフォルダ選択ダイアログを開く(`ui.pickFile`のフォルダ版)。

- params: `{ title?: string }`(ダイアログのタイトル/プロンプト文字列。省略時はnative既定のタイトル)
- result: `{ folderPath: string }`
- エラー: `CANCELLED`(ユーザーがダイアログをキャンセル — `ui.pickFile`と同じ規約でUIではエラー扱いしない)、`DIALOG_FAILED`(ダイアログ自体の失敗)。
- 実装: `IFileOpenDialog` + `FOS_PICKFOLDERS`(`shobjidl.h`)によるモーダルダイアログ。`ui.pickFile`と同じUIスレッド上のモーダルポンプ方式。
- **WebUI側ローカル待機上限なし**: `ui.pickFile`と同じ理由(§4.9参照)で`RequestDispatcher`はローカルタイマーを張らない(`NO_LOCAL_TIMEOUT_METHODS`、§6参照)。

### 4.16 `fs.listFiles`(contract v6)

指定フォルダ直下(非再帰)のファイル一覧を返す。バッチA2Vのマニフェスト生成画面が、フォルダ選択後にwav素材の候補一覧を作るのに使う。

- params: `{ folderPath: string, extensions?: string[], withAudioDuration?: boolean }`
  - `extensions`: 拡張子フィルタ(`.`込み、例`[".wav", ".csv"]`)。**小文字比較**(大文字小文字を区別しない)。省略時はフィルタなし。
  - `withAudioDuration`: `true`の場合、`.wav`ファイルのみヘッダを実測して`durationSec`を返す。`.wav`以外のファイル、および読み取りに失敗した`.wav`は`durationSec: 0`(**エラーにはしない**)。
- result: `{ files: Array<{ name: string, path: string, sizeBytes: number, mtimeMs: number, durationSec: number }> }`
- **並び順は未規定 — 呼び出し側(WebUI)が`files`を必要な基準(通常は`name`)でソートすること**。native実装は`FindFirstFileW`/`FindNextFileW`によるWindowsファイルシステムの列挙順をそのまま返すため、ソート済みである保証はない(契約上の注意点として維持)。
- エラー: 存在しないフォルダを指定した場合もエラーにせず**空配列を返す**(`FindFirstFileW`が`INVALID_HANDLE_VALUE`を返すケースをそのまま空の結果として扱う実装。native実装確定によりこの挙動が確定した)。ディレクトリ・隠しファイル・システムファイルは列挙結果から除外される。

### 4.17 / 4.18 `fs.readTextFile` / `fs.writeTextFileAtomic`(撤去済み)

> **2026-07-18: `fs.readTextFile` / `fs.writeTextFileAtomic` / エラーコード`WRITE_LOCKED`はバッチA2VのCSV状態管理廃止に伴い撤去した。** contract v6で追加された両メソッド(旧§4.17/§4.18)は、バッチA2Vがマニフェストcsvの読み書きに使っていた唯一の消費者だったが、CSV状態管理そのものが廃止されたため利用者がゼロになった。ネイティブ側の`ParseReadTextFile`/`ParseWriteTextFileAtomic`(`bridge_core.h/.cpp`)、両ワーカー(`bridge.cpp`)、ディスパッチ、doctest、および専用エラーコード`WRITE_LOCKED`を削除した。この撤去自体は既存メソッドの純粋な削除のため単独では契約バージョンを繰り上げていないが、その後2026-07-20に右クリック再設計群の追加(§4.14.1)を理由に契約バージョンはv8へ繰り上げ済みとなっている。BOM処理などの汎用ヘルパー(`fs_util.h`)は他用途向けの汎用ユーティリティとして残置している。

### 4.19 `fs.probeAudioDuration`(contract v6)

単一ファイルのwavヘッダを読んで再生時間を計測する(`fs.listFiles`の`withAudioDuration`の単体版。生成長をアップロードしたwavの長さに自動調整するUIヒント用)。

- params: `{ filePath: string }`
- result: `{ durationSec: number, isWav: boolean }`
- `.wav`以外のファイル、および読み取りに失敗したファイルは`{ durationSec: 0, isWav: false }`を返す(**エラーにはしない** — ベストエフォートのUIヒント用途であり、妥当性検証には使わない設計)。

### 4.20 `backend.downloadVideo`のparams拡張(contract v6)

§4.4の`backend.downloadVideo`に、バッチA2Vが各クリップを呼び出し側指定のフォルダへ保存できるよう、後方互換の追加paramsが計4つ加わった(`destDir`/`fileName`/`noClobber`の3つに加え、2026-07-18に`reuseIfPresent`を追加)。4つとも省略可能で、全て省略すればv6以前と完全に同じ挙動になる。

- 追加params: `destDir?: string` / `fileName?: string` / `noClobber?: boolean` / `reuseIfPresent?: boolean`
  - `destDir`: 指定時はここへ保存する(native既定の一時/キャッシュ場所の代わり)。親フォルダが存在しなければ作成する。
  - `fileName`: 拡張子込みの希望ファイル名(native既定の`<jobId>[_joined].mp4`命名の代わり)。
  - `noClobber`: `true`の場合、解決後のパスに既存ファイルがあっても**絶対に上書きしない** — ファイル名の幹(拡張子を除いた部分)に`_2`, `_3`, ...と付番し、衝突しない名前が見つかるまで続ける。ただし無制限ではなく**上限1000通り**(`bridge.cpp`の`kMaxNoClobberAttempts = 1000`)まで試し、それでも衝突が解消しなければ失敗する。
  - `reuseIfPresent`(2026-07-18追記): `true`かつ解決後の保存先(`noClobber`未指定時のパス)に既にサイズ>0のファイルが存在する場合、ダウンロードを省略して既存のパス/サイズを返す。クリップ単位の不変な出力の再取得を避けるためのオプション。`noClobber`(元々上書きしない)指定時は無視される。
- result(既存と同じ形): `{ filePath: string, sizeBytes: number }`。`filePath`は(`noClobber`による付番後を含め)**実際に書き込まれたパス**を返す。

### 4.21 `ui.resolveDroppedFiles`(contract v7)

エクスプローラーからのドラッグ&ドロップで受け取ったファイルの、実際のローカルパスを解決する。Create画面の参照動画(IC-LoRA)欄・A2V音声欄、Chain画面の統合ソース入力欄(§4.9の`"imageOrVideo"`と同じ振り分け先)の3箇所で使われる。

**背景**: WebView2の`AllowExternalDrop`は既定`true`(本プロジェクトでは変更していない)なので、ページ上の素のDOM `dragover`/`drop`は追加のネイティブ実装なしですでに発火する。ただしJavaScriptの`File`オブジェクトからは実際のローカルパスが取れない(ブラウザのセキュリティ制限で、WebView2もこれを解除しない)。この制限を回避するのが`ui.resolveDroppedFiles`で、通常の`postMessage`ではなく`chrome.webview.postMessageWithAdditionalObjects(request, [file, ...])`(WebUI側は`NativeBridge.requestWithFiles`、`webviewBridge.ts`)でファイルを追加のCOMオブジェクトとして同梱して送る。ネイティブ側は`ICoreWebView2WebMessageReceivedEventArgs2::get_AdditionalObjects()`でこれを取得し、各要素を`ICoreWebView2File`へクエリして`get_Path()`で実パスを得る(`webview_host.cpp`)。

- params: `{}`(呼び出し側が指定できるフィールドは無い。下記の`__droppedPaths`はネイティブが強制的に注入する予約キーで、呼び出し側からは触れない)
- result: `{ files: Array<{ filePath: string, fileName: string }> }`。ドロップされたファイルのうち実パスへ解決できたものだけが、`AdditionalObjects`に並んでいた順のまま並ぶ。何も解決できなければ(あるいはドロップ自体が無ければ)`files: []`。
- エラー: 無し(常に成功。`__droppedPaths`が無い/空/型不正のいずれも単に「何もドロップされなかった」として`files: []`を返す — `bridge_core.cpp`の`ParseResolveDroppedFiles`)。

**セキュリティ規律(敵対的レビュー済み設計、2026-07-17に発火条件のBLOCKERを修正 — 必読)**: `params.__droppedPaths`はページ側が正当な手段で設定できるフィールドでは**ない**。`native/src/webview_host.cpp`のWebMessageReceivedハンドラは、受信した**あらゆる**メッセージについて次の3条件のいずれかに従って`params.__droppedPaths`の扱いを決める(判定は**必ずこの順序**で行う):

1. **このメッセージのAdditionalObjectsが1件以上ある** → `params`が(無ければ空オブジェクトとして生成してでも)存在する前提で、`params.__droppedPaths`を**必ず**実際に解決できたパスの配列(添付オブジェクトが1件もファイルへ解決できなければ空配列`[]`)で上書きする。生のメッセージ文字列が何であるか(`__droppedPaths`という部分文字列を含むかどうか)は一切関係ない。
2. **AdditionalObjectsが0件、かつ生のメッセージ文字列に`"__droppedPaths"`という部分文字列が含まれる** → 同じく`params.__droppedPaths`を上書きするが、こちらは実ドロップが無かったので**空配列`[]`に強制**する(偽装対策 — ページ側が「本物のドロップを装った`__droppedPaths`」を送り込んでも、必ず空配列に潰される)。
3. **AdditionalObjectsが0件、かつ部分文字列も含まれない** → 何もしない(素通し)。この場合`HandleRequestJson`側が`__droppedPaths`というキーを見る経路自体が存在しないため、書き換えの必要が無い(パフォーマンス上のfast path — 大多数の通常メッセージがこれに該当する)。

これにより、ページ側が偽の`__droppedPaths`を送り込んでも、ネイティブに届く前に必ず実際の値(または空配列)へ差し替えられ、`ui.resolveDroppedFiles`を「任意のローカルファイルを読み出せるプリミティブ」に転用することはできない。

- **BLOCKER修正の経緯(2026-07-17)**: 初版の実装は条件1と条件2の順序が逆で、「生のメッセージ文字列に部分文字列が含まれるか」を**先に**判定し、含まれなければAdditionalObjectsを一切確認せずに素通ししていた。ところが実際のドロップ経路(`webui/src/shell/useFileDrop.ts`)は`requestWithFiles("ui.resolveDroppedFiles", {}, [file])`のように**空の`params: {}`**を送るため、本物のドロップのリクエストJSONには`"__droppedPaths"`という文字列がそもそも一度も現れない。その結果、本物のドロップは常にこのfast pathで素通しされてしまい、AdditionalObjectsの解決も注入も一切実行されず、`ui.resolveDroppedFiles`は実機で常に`files: []`(→WebUI側は`resolveFailed`表示)を返すだけで、D&D機能全体が機能しない状態だった。判定順序をAdditionalObjectsの有無優先に直したことで解消している。なお、この抜け穴は`mockBridge.ts`のテスト用実装がネイティブの注入機構を経由せず、渡されたファイルの`.name`から直接`{filePath, fileName}`を合成していたため、webuiのテスト・native側のdoctestのどちらでも検出できなかった(併せて修正済み — 下記参照)。
- 上記の上書き処理(条件1・2)が**免除される**のは、生のメッセージ文字列がJSONオブジェクトとしてパースできない場合だけである — その場合`HandleMessage`側の`HandleRequestJson`も同じ理由で独立にパースへ失敗し`BAD_REQUEST`を返すため、ここで書き換えてもしなくても結果は変わらない。
- `ICoreWebView2WebMessageReceivedEventArgs2`へのクエリ自体が失敗した場合(古いWebView2ランタイム等)は、AdditionalObjectsが0件だったのと同じ扱いになる(条件2または3が適用される)。
- 実装は「発火条件の判定(AdditionalObjects取得・部分文字列チェック — WebView2/COM依存、`webview_host.cpp`)」と「JSON書き換え自体(純粋関数`InjectDroppedPathsIntoRequest(request_json, paths)`、`native/src/bridge_core.h/.cpp`)」に分離されている。後者は「パースできて`params`がオブジェクトなら、`params`が空(`{}`)であっても`__droppedPaths`キーを必ず生やして上書きする」「オブジェクトとしてパースできなければ元の文字列をそのまま返す」という規律を担う純粋関数で、WebView2非依存のためdoctest対象(`native/tests/test_bridge_core.cpp`) — 特に「空`params`のリクエスト＋非空`paths`→注入される」テストは今回のBLOCKERの再発防止テストとして追加した。`ParseResolveDroppedFiles`/`MakeResolveDroppedFilesResult`も同様にWebView2非依存でdoctest対象。AdditionalObjectsの実解決自体(`ExtractDroppedFilePaths`、`webview_host.cpp`、WebView2依存)は実機でしか確認できず、該当箇所に`REALDEVICE-VERIFY`コメントを残置している。
- `webui/src/bridge/mockBridge.ts`の`requestWithFiles`も、渡されたファイルから`params.__droppedPaths`を合成してから`ui.resolveDroppedFiles`のハンドラへ渡すよう修正済み(ネイティブの注入を模倣) — ハンドラ自身は`params.__droppedPaths`を読む実装になり、実物と同じ契約(注入されたキーを読む)をwebuiのテストが検証できるようになった。

- タイムアウト: 他の同期メソッド(`ping`/`getEditInfo`等)と同じ、WebUI側の既定ローカル待機上限**10,000ms**(`DEFAULT_TIMEOUT_MS`)がそのまま適用される。`ui.pickFile`/`ui.pickFolder`と違って**`NO_LOCAL_TIMEOUT_METHODS`には含まれない** — モーダルダイアログを開くわけではなく、ユーザー操作を待たない即時処理だからである(§6参照)。

### 4.22 `fs.probeMediaInfo`(contract v9)

単一のローカルメディアファイルの再生時間と解像度を、ベストエフォートで取得する。Create画面の参照動画(IC-LoRA。参照した動画の動きや輪郭をなぞって生成する仕組み)欄・音声(A2V。音声から動画を生成する機能)欄のカードに、添付したファイルの尺を「12.3s」の形で表示するためのUIヒント用途。`fs.probeAudioDuration`(§4.19、wavヘッダのみを読む音声専用版)の一般化にあたり、動画・画像も含めて`get_media_info`で計測する点が異なる。

- params: `{ filePath: string }`(**非空必須**。欠落/空/非文字列は`BAD_REQUEST`)
- result: `{ durationSec: number, width: number, height: number }`(**3つとも非nullable。不明なときは`0`**)。`durationSec`は秒。静止画は`0`(尺の概念が無い)。
- エラー: `BAD_REQUEST`(`filePath`が欠落/空/非文字列のときのみ)。**それ以外では一切rejectしない**(下記)。
- 実装: `native/src/bridge.cpp`の`ProbeMediaInfoEditProc`が、`call_edit_section_param`のコールバック内で`EDIT_SECTION::get_media_info`([SDK_REFERENCE.md](SDK_REFERENCE.md) §5.1)をUIスレッド同期で呼ぶ(`timeline.insertMedia`/`timeline.getSelection`と同じ流儀)。`get_media_info`が要求するワイド文字列は、UTF-8のfilePathをEditProc内ローカルの`std::wstring`へ変換して渡す(コールバックのライフタイム内でのみ有効にすることでライフタイム安全を担保)。パース/整形は`bridge_core.h/.cpp`の純関数`ParseProbeMediaInfo`/`MakeMediaInfoResult`が担い、こちらはWebView2非依存でdoctest対象(`native/tests/test_bridge_core.cpp`に6ケース追加)。
- **設計上の要点(失敗しても穏当に劣化する)**: `fs.probeAudioDuration`と同じく、取得できないケースはすべて`{ durationSec: 0, width: 0, height: 0 }`の**成功応答**に落とし、rejectしない。具体的には(a)編集ハンドルが無い(プロジェクト未オープン)、(b)`get_media_info`自体が失敗、(c)未対応の動画コンテナ(webm/mkv等)、(d)`MediaInfoProvider`が未配線、のいずれも全ゼロ結果になる。呼び出し側(webui)はこの`0`を「不明」と解釈して尺表示を単に出さない。この「非表示への穏当な劣化」がベストエフォートUIヒントとしての正しい振る舞いであり、妥当性検証には使わない(§4.19の`fs.probeAudioDuration`と同じ設計思想)。**注意**: webui側のprobe用エフェクト(`useGenerationForm.ts`ほか)には`.catch()`を必ず付け、`BAD_REQUEST`(filePath不正)のときの唯一のreject経路がunhandled rejectionにならないようにする規律がある(敵対的レビューで指摘された事項。詳細は[`DEVLOG.md`](DEVLOG.md) §42)。
- ディスパッチ位置: `HandleMessage`の非同期インターセプトには入れず、`HandleRequestJson`の同期分岐(`ui.resolveDroppedFiles`の後)で処理される。§3の表では`timeline.getSelection`と同じ「UIスレッド上で`call_edit_section_param`を同期呼び出しするメソッド」に分類される。
- タイムアウト: 他の同期メソッドと同じWebUI側の既定ローカル待機上限**10,000ms**(`DEFAULT_TIMEOUT_MS`)。`NO_LOCAL_TIMEOUT_METHODS`には含めない(§6参照)。

## 5. エラーコード一覧

`KnownBridgeErrorCode`(`types.ts`)と`bridge_core.h`冒頭コメントは、**v4.1の範囲では**完全一致(§9参照、相違なし)。v6で追加された`WRITE_LOCKED`は`fs.writeTextFileAtomic`とともに2026-07-18に撤去された(§4.17/§4.18の撤去注記参照)。v5で追加されたコード(`CUTOUT_FAILED`/`EXTRACT_FAILED`/`PROVISIONAL_FAILED`等)は、v5の該当メソッド自体が本書に未収録のままのため(§0冒頭注記)、引き続きnative側との突き合わせが済んでいない。

| コード | 意味 | 発生メソッド |
|---|---|---|
| `BAD_REQUEST` | paramsの形式不正・バリデーション失敗 | 全メソッド共通(params検証があるもの全て) |
| `UNKNOWN_METHOD` | 未知の`method` | 全メソッド共通(ディスパッチのフォールバック) |
| `NO_EDIT_HANDLE` | 編集ハンドル未取得/取得直後で使用不可 | `getEditInfo`, `timeline.insertMedia`, `timeline.captureFrame`, `timeline.getSelection`(v5) |
| `BACKEND_UNREACHABLE` | 接続不可・切断等のトランスポート失敗(HTTPステータスを伴う応答はここに含まれない) | `backend.request`, `backend.downloadVideo`, `backend.uploadFile` |
| `BACKEND_TIMEOUT` | WinHTTPタイムアウト | 同上 |
| `DOWNLOAD_FAILED` | ダウンロードHTTPステータスが200以外 | `backend.downloadVideo` |
| `FILE_NOT_FOUND` | 送信前のローカルファイル存在チェック失敗 | `timeline.insertMedia`, `backend.uploadFile`, `ui.makeThumbnail` |
| `INSERT_FAILED` | メディアオブジェクトの生成に失敗(§3-140以降は**エイリアス経路と`create_object_from_media_file`の両方が`nullptr`**を返したとき。メッセージは経路中立な`media object creation failed …`) | `timeline.insertMedia`, `timeline.insertMediaForJob` |
| `CAPTURE_FAILED` | レンダリング/PNGエンコードの失敗(`NO_EDIT_HANDLE`除外後) | `timeline.captureFrame` |
| `CANCELLED` | ユーザーがファイル選択/フォルダ選択をキャンセル(UIではエラー非表示) | `ui.pickFile`, `ui.pickFolder`(v6・§4.15) |
| `DIALOG_FAILED` | ネイティブOpen/フォルダ選択ダイアログ自体の失敗・再入 | `ui.pickFile`, `ui.pickFolder`(v6・§4.15) |
| `THUMBNAIL_FAILED` | デコード/リサイズ/エンコード失敗 | `ui.makeThumbnail` |
| `TIMEOUT`(ローカルのみ) | WebUI側でレスポンスを一定時間受信できなかった | 任意のメソッド(WebUI側`RequestDispatcher`が生成、ネイティブは関与しない) |
| `DISPOSED`(ローカルのみ) | ブリッジ破棄時に未解決のリクエストを一括reject | 同上 |

## 6. タイムアウト既定値

| 項目 | 既定値 | 上書き | 出典 |
|---|---|---|---|
| WebUI `RequestDispatcher`のローカル待機上限 | **10,000ms**(`DEFAULT_TIMEOUT_MS`) | `params.timeoutMs`が数値であればそれを採用(現状`backend.request`のみ`timeoutMs`を持つ)。`ui.pickFile`/`ui.pickFolder`は下記の通り対象外 | `webui/src/bridge/types.ts`, `requestDispatcher.ts` |
| `ui.pickFile`/`ui.pickFolder`のWebUI側ローカル待機上限 | **無し**(`NO_LOCAL_TIMEOUT_METHODS`に列挙されたメソッドはタイマー自体を張らない) | ネイティブのファイルダイアログはユーザーが閉じるまで無期限に待つモーダルで「長すぎる」という概念が無いため。ユーザーが選択に10秒以上かけてもWebUI側が先にTIMEOUTで諦めることはない(ブリッジ破棄時は`DISPOSED`で解決) | `webui/src/bridge/types.ts`(`NO_LOCAL_TIMEOUT_METHODS`), `requestDispatcher.ts` |
| ネイティブ`backend.request`のWinHTTPタイムアウト | **30,000ms**(`kDefaultBackendTimeoutMs`) | `params.timeoutMs`を`[1000, 600000]`にクランプして適用 | `bridge_core.h` |
| `backend.downloadVideo`のダウンロードタイムアウト | **300,000ms**(`kDefaultDownloadTimeoutMs`) | 無し | `http_client.h` |
| `backend.uploadFile`のアップロードタイムアウト | **120,000ms**(`kUploadTimeoutMs`) | 無し | `http_client.h` |
| `timeline.captureFrame`のレンダリングコールバック待機 | **10秒**(ハードコード、`std::chrono::seconds(10)`) | 無し | `bridge.cpp::CaptureFrameWorker` |
| `POST /jobs/{id}/join`呼び出し時のWebUI側`timeoutMs`指定例 | **120,000ms**(`joinJob()`が`backend.request`へ明示的に渡す) | — | `webui/src/api/client.ts`(`bridge/types.ts`のJSDocに記載) |
| `ui.resolveDroppedFiles`のWebUI側ローカル待機上限(**contract v7**) | **10,000ms**(既定の`DEFAULT_TIMEOUT_MS`のまま。`NO_LOCAL_TIMEOUT_METHODS`には含めない) | 無し | `webui/src/bridge/types.ts`, `requestDispatcher.ts` |
| `fs.probeMediaInfo`のWebUI側ローカル待機上限(**contract v9**) | **10,000ms**(既定の`DEFAULT_TIMEOUT_MS`のまま。`NO_LOCAL_TIMEOUT_METHODS`には含めない) | 無し | `webui/src/bridge/types.ts`, `requestDispatcher.ts` |

**重要**: v4.1の`timeoutMs`は「ネイティブのWinHTTPタイムアウト」と「WebUI側のローカル待機上限」の**両方**に同じ値を伝播させる設計。バックエンド処理が長時間かかり得る呼び出し(例: chain join)でWebUI側が先に諦めてネイティブの応答を握りつぶさないようにするため。

## 7. ファイル保存場所一覧(`%LOCALAPPDATA%\NzVideomni\`)

| パス | 内容 | 書き込み元 |
|---|---|---|
| `logs\plugin.log` | UTF-8・タイムスタンプ付きファイルログ。AviUtl2はログをディスクに永続化しないため、自動検証が読める唯一のチャネルとして自前実装。ホストの`LOG_HANDLE`(1024文字/行制限)にも同時出力。 | `native/src/log.cpp` |
| `downloads\<jobId>.mp4` / `<jobId>_joined.mp4` | `backend.downloadVideo`の保存先 | `bridge.cpp` |
| `captures\frame_<frame>_<stamp>.png` | `timeline.captureFrame`の保存先。`<stamp>` = `yyyymmdd_HHMMSS_fff_nnn`(並列キャプチャでも一意) | `bridge.cpp` |
| `settings.json` | `settings.get`/`settings.set`が読み書きする永続設定(現状`baseUrl`のみ)。ファイル欠落/JSON破損/値不正は既定値へ自己修復し、書き込みは行わない(次回`settings.set`成功まで) | `native/src/settings.cpp` |
| `webview2\` | WebView2の`UserDataFolder`(Cookie/キャッシュ等、WebView2 SDKが管理) | `native/src/webview_host.cpp` |
| `buildtools\`(開発環境のみ、配布物には含まれない) | VS2026にC++ CMakeツールが未同梱のため、ビルドスクリプトが自動取得するポータブルCMake+Ninja(per-userキャッシュ、PATH非変更) | `scripts/build.ps1` |

いずれも`AppDataDir()`(`log.h/.cpp`: `%LOCALAPPDATA%\NzVideomni`)を起点に構築される。`LOCALAPPDATA`が取得できない環境では空文字列となり、該当機能は無効化される(例: `settings_`はメモリ上のみで動作し、ファイルへは書き込まない)。

## 8. 契約バージョン変遷(参考)

|版 | マイルストーン | 追加内容 |
|---|---|---|
| v1 | M1 | `ping`, `getEditInfo` |
| v2 | M2 | `backend.request`, `backend.downloadVideo`, `backend.getBaseUrl`, `timeline.insertMedia` |
| v3 | M4 | `timeline.captureFrame`, `backend.uploadFile`, `ui.pickFile`, `ui.makeThumbnail` |
| v4 | M7a | `settings.get`, `settings.set` |
| v4.1 | M7c | `backend.request`の`timeoutMs`パラメータ |
| v5 | タイムライン生成AI機能(α版) | `timeline.getSelection`／`timeline.cutoutRange`／`timeline.extractAudio`／`timeline.insertProvisional`／`timeline.resolveProvisional`／`timeline.updateProvisionalText`／`timeline.scanProvisionals`、イベント`timeline.menuInvoked`／`timeline.projectLoaded`。**本書は`timeline.getSelection`と`timeline.menuInvoked`のみ§4.13/§4.14で追補済み。残り6メソッド＋`timeline.projectLoaded`イベントは本書に未収録(既知のドキュメント債務)**。`timeline.getSelection`の`selected[].mediaWidth`/`mediaHeight`は2026-07-08の右クリック生成配線セッションで追加。 |
| v6 | バッチA2V(CSVマニフェスト駆動の一括生成) + wav長自動調整(フロントエンド追随グループ2) | `ui.pickFolder`／`fs.listFiles`／`fs.readTextFile`／`fs.writeTextFileAtomic`／`fs.probeAudioDuration`の5メソッド新設、および`backend.downloadVideo`への`destDir`/`fileName`/`noClobber`params拡張(後方互換)。エラーコード`WRITE_LOCKED`を新設。**本書は§4.15〜§4.20で全項目を収録済み。native実装は2026-07-15(グループ2セッション)に完了(新規`native/src/wav_probe.h/.cpp`・`native/src/fs_util.h/.cpp`、doctest`test_wav_probe.cpp`/`test_fs_util.cpp`)し、実装コードとの突き合わせも完了している**。(なお`fs.readTextFile`/`fs.writeTextFileAtomic`および`WRITE_LOCKED`は2026-07-18にバッチA2VのCSV状態管理廃止に伴い撤去した——§4.17/§4.18参照。同日、`backend.downloadVideo`に`reuseIfPresent`paramが追加された——§4.20参照。現存するv6メソッドは`ui.pickFolder`/`fs.listFiles`/`fs.probeAudioDuration`の3つ。) |
| v7 | ドラッグ&ドロップ対応(第4波) | `ui.resolveDroppedFiles`メソッド新設(§4.21) — `postMessageWithAdditionalObjects`で同梱されたドロップファイルを実パスへ解決する。`params.__droppedPaths`はネイティブが常時強制上書きする予約キー(セキュリティ規律、§4.21参照)。合わせて、Chainのソース入力欄一本化セッションで先行実装済みだった`ui.pickFile`の`"imageOrVideo"` kindを§4.9へ追補。**本書は§4.9(追補分)/§4.21で全項目を収録済み。native実装(`bridge_core.h/.cpp`の`ParseResolveDroppedFiles`/`MakeResolveDroppedFilesResult`、`webview_host.cpp`の`__droppedPaths`注入)は同セッションで完了、doctest`test_bridge_core.cpp`で単体検証済み(AdditionalObjects解決自体はWebView2依存のためREALDEVICE-VERIFY対象)**。 |
| v8 | 右クリック再設計群(2026-07-19〜20) | `timeline.insertMediaForJob`／`timeline.updateProvisionalReservation`／`timeline.deleteProvisionalByJob`の新設、`timeline.insertProvisional`の契約変更(`numFrames`／`genFps`／配置系統／`textPrefix`)、`timeline.getSelection`への`textContent`／`mediaDurationSec`拡張、レイヤーメニューへの`addCurrentFrameAsKeyframe`／`currentFrameToClipChain`アクション追加。**本書は§4.14.1(要点のみ、詳細は[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md)へ委譲)と§4.13/§4.14の拡張注記で収録。native・`types.ts`とも実装済み**。 |
| v9 | IC-LoRA／A2Vカードの尺表示(2026-07-20) | `fs.probeMediaInfo`メソッド新設(§4.22) — 単一のローカルメディアファイルの再生時間・解像度を`EDIT_SECTION::get_media_info`経由でベストエフォート取得する同期メソッド。IC-LoRA／A2Vカードのリデザインで添付ファイルの尺を「12.3s」表示するために追加。取得不能時は`{durationSec:0, width:0, height:0}`の成功応答へ穏当に劣化し、rejectは`filePath`不正の`BAD_REQUEST`のみ(新規エラーコードは無し)。**本書は§4.22で全項目を収録済み。native実装(`bridge_core.h/.cpp`の`ParseProbeMediaInfo`/`MakeMediaInfoResult`、`bridge.cpp`の`ProbeMediaInfoEditProc`)は完了、doctest`test_bridge_core.cpp`に6ケース追加(252→258 pass/6 skip)。`get_media_info`の実呼び出しは実機でのみ確認可能**。 |
| v10 | V2Vリボン範囲トリム(2026-07-30) | `backend.uploadFile`への任意`query`(`Record<string,string>`)加算と、応答`body`に加わる任意`trimmed`(bool)(§4.5)。タイムライン上のリボンが元動画の一部しか占めていないとき、`POST /upload/video?trim_start_sec=…&trim_duration_sec=…`でその範囲だけを切り出してアップロードするために追加。**省略時のURLはv10以前と完全に同一**(`AppendQueryToUrl`が空クエリでURLを変えないことで構造的に保証)。新規メソッド・新規エラーコードは無し。**本書は§4.5で収録済み。native実装(`bridge_core.h/.cpp`の共有純関数`BuildQueryString`/`AppendQueryToUrl`、`bridge.cpp`の`UploadFileWorker`でのURL連結)は完了、doctest`test_bridge_core.cpp`に9ケース追加(258→267 pass/6 skip)。もう半分の`timeline.getSelection`への再生位置系6フィールド(`playbackStartSec`/`playbackEndSec`/`hasPlaybackRange`/`playbackSpeed`/`loopPlay`/`sectionCount`)は2026-08-01の実機調査完了をもって実装・収録済み(§4.13、単位は素材時間軸の秒。doctestはさらに7ケース増えて274 pass/6 skip)**。 |
| v11 | 素材fpsの取得と操作パネルへの流し込み(2026-09-01) | `timeline.getSelection`の`selected[]`への`mediaFps`(素材そのもののフレームレート、非nullable・`0`＝不明)追加(§4.13)。右クリックプリフィルのfps軸「素材に合わせる」を実際に機能させるために追加した。新規メソッド・新規エラーコードは無く、**追加は1フィールドだけ**である。**取得はMedia Foundation**(`MF_MT_FRAME_RATE`)で、native実装は新規モジュール`native/src/media_fps_probe.h/.cpp`(`FpsFromRatio`／`ProbeMediaFps`)と`bridge.cpp`の`GetSelectionEditProc`からの呼び出し、`bridge_core.h`の`SelectionItem::media_fps`と`MakeSelectionResult`のJSON化からなる。**nativeは生値を返し、整数へのスナップ(29.97→30)はwebuiの`prefillSeed.ts`が行う**。**mkv/webmでの取得失敗は正常系**(webuiがプロジェクトfpsへ落ちる)。doctestは新設`native/tests/test_media_fps_probe.cpp`ほかで**278→282ケース・アサーション1416→1452**(いずれも0失敗。実測値は`build.ps1 -Config Release -RunTests`で確認)。**MFのプローブが実ファイルを開く経路は実機でのみ確認可能**で、オーナーの実機ゲートG1〜G8待ちである(項目は台帳[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md)「2. 実装済み・ユーザーのテスト待ち」)。 |

## 9. `types.ts` と `bridge_core.h`/`.cpp` の相違点

両ソースを突き合わせた結果、以下**1点**の相違を確認した(実装を正とし、WebUI側は意図的にそれを無視する設計):

- **`getEditInfo`の`result`**: `bridge_core.cpp`(`HandleRequestJson`)は`width/height/rate/scale/sampleRate/frame`に加えて **`layer`、`frameMax`、`layerMax`** の3フィールドを常にJSONへ含める(`EDIT_INFO`の対応フィールドをそのまま転写)。かつては`webui/src/bridge/types.ts`の`BridgeResultMap["getEditInfo"]`がこの3つを宣言せず型を絞っていた（「将来これらを使う画面を追加する場合は`BridgeResultMap`側に追記するだけで済む・ネイティブ側の変更は不要」）が、**2026-07-21のJoin復活（I4）でその「将来」が到来し、TS型へ`layer` / `frameMax` / `layerMax`を宣言追記した**（Join UIのjoined挿入が最前面レイヤー算出に`layerMax`を使う）。予告どおりネイティブ側は無変更・契約バージョンのbumpも不要で、型のミラーを実装の現実へ寄せただけである。この項はもはや「相違点」ではなく、解消済みの記録として残す。

それ以外(エラーコード一覧、各メソッドのparams/result形状、`timeoutMs`のクランプ範囲、`settings.set`のno-op挙動等)はコード上完全に一致していた。

## 10. 参照ファイル(絶対パス)

- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\webui\src\bridge\types.ts`
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\webui\src\bridge\requestDispatcher.ts`
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\webui\src\bridge\webviewBridge.ts`
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\native\src\bridge_core.h` / `bridge_core.cpp`
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\native\src\bridge.h` / `bridge.cpp`
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\native\src\http_client.h`
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\native\src\settings.h` / `settings.cpp`
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\native\src\log.h` / `log.cpp`
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\native\src\wic_png.cpp`
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\native\src\wav_probe.h` / `wav_probe.cpp`(contract v6、wavヘッダ計測。2026-07-15新規)
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\native\src\fs_util.h` / `fs_util.cpp`(contract v6、フォルダ列挙・テキスト読み書き・原子的置換・noClobber付番。2026-07-15新規)
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\native\tests\test_bridge_core.cpp`(該当ファイルは`native/tests/`配下)
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\native\tests\test_wav_probe.cpp` / `test_fs_util.cpp`(contract v6のnative実装doctest)
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\native\src\webview_host.cpp`(contract v7、WebView2ホスティング＋`WebMessageReceived`ハンドラでの`__droppedPaths`注入。2026-07-17新規追記)
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\webui\src\bridge\mockBridge.ts`(contract v7、`requestWithFiles`/`ui.resolveDroppedFiles`のdev/testフィクスチャ)
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\webui\src\shell\useFileDrop.ts` / `DropZone.tsx`(contract v7、3箇所のドロップゾーンが共有するD&D受付ロジック/UI)
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\webui\src\timeline\sourceTrim.ts`(contract v10、`query`の唯一の生成元`trimQuery()`と、再生位置系フィールドを消費する唯一の判定関数`decideSourceTrim`。2026-07-30新規／2026-08-01に実機確定値へ更新)
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\webui\src\bridge\mockBridge.v10.test.ts`(contract v10、mockブリッジ側の`query`／`trimmed`の疎通テスト)
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\native\src\media_fps_probe.h` / `media_fps_probe.cpp`(contract v11、Media Foundationによる素材fpsの読み取り。2026-09-01新規。**MFの寿命・COMアパートメント・取得失敗が正常系である理由の正本はこのヘッダのコメント**)
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\native\tests\test_media_fps_probe.cpp`(contract v11のnative実装doctest。純関数`FpsFromRatio`とMF往復)
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\webui\src\timeline\prefillSeed.ts`(contract v11、`mediaFps`の唯一の消費者。fpsの3段フォールバックを実装。整数スナップと`[1, 60]`クランプ自体の正本は`modes/single/paramUtils.ts`の`snapFrameRate`——台帳§3-71/§3-72対策で全入口共通化)
