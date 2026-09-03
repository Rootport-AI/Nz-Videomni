# AviUtl2向けプラグイン開発 WEBリサーチまとめ

最終更新: 2026-07-07 / 出典: 調査エージェント報告(2026-07-07)

関連ドキュメント: [API_REFERENCE.md](API_REFERENCE.md) ／ [SDK_REFERENCE.md](SDK_REFERENCE.md) ／ [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) ／ [BRIDGE_CONTRACT.md](BRIDGE_CONTRACT.md) ／ [DEVLOG.md](DEVLOG.md)

テーマ: AviUtl2向けプラグイン開発(C++、独自UIウィンドウ＋ローカルHTTP通信)に関するWeb上の公開情報の調査。各情報の信頼度を **【公式】【二次情報】【推測】** で付記する。

※現在の実機対象は **v2.0.54ポータブル**（`D:\For_Videos\AviUtl2\aviutl2_v2.0.54\`）です。本調査は2026/7/7時点のbeta52前提でまとめたもので、当時のWeb公開情報の史料として保持しています。

---

## 1. AviUtl2の概要と版事情

- 開発者: KENくん氏。旧AviUtl・拡張編集をゼロから作り直した新版。
- 公式サイト: [AviUtlのお部屋](https://spring-fragrance.mints.ne.jp/aviutl/)【公式】
- AviUtl ExEdit2 beta1は2025年7月7日公開。最新版はbeta53a(2026年7月5日公開、SDKも同日更新)。【公式】
- 旧AviUtlとの違い: 64bit化、拡張編集の本体統合、内部フォーマットRGBA16bitFloat/PCM32bitFloat、プロジェクト形式`.aup2`(旧形式と非互換)、**プラグイン資産は旧新間で完全非互換**。【二次情報】
- 公式ドキュメントサイト [docs.aviutl2.jp](https://docs.aviutl2.jp/) にはプラグイン開発API文書は無い(ユーザー向け操作マニュアルのみ)。

**本プロジェクトへの含意**: 開発対象本体はbeta52に固定するが、SDK自体はbeta53a世代(2026/7/5版)であり、ベータ開発が非常に活発なため両者にAPI差分がある。詳細な整合表は [SDK_REFERENCE.md](SDK_REFERENCE.md) §10 を参照。

## 2. SDK公開状況

- SDKは公式サイトから `AviUtl2_sdk.zip` として配布される。【公式】
- 非公式ミラー: [aviutl2/aviutl2_sdk_mirror](https://github.com/aviutl2/aviutl2_sdk_mirror)(sevenc-nanashi氏、GitHub Actionsで自動更新、MITライセンス)。【二次情報・実体確認済み】
- プラグイン種別は5種: `.aui2` / `.auo2` / `.auf2` / `.mod2` / **`.aux2`(汎用＝独自ウィンドウ)**。種別ごとの詳細は [SDK_REFERENCE.md](SDK_REFERENCE.md) §1参照。
- CHANGELOG解析: `.aux2`向けAPIは2025年10月〜2026年6月に急速に整備された。特に本プロジェクトが使う中核API:
  - `register_window_client` 2025/10/26
  - `call_edit_section_param` 2025/11/22
  - `get_edit_info` / `create_object_from_media_file` 2025/12/2
  - `register_event_listener` 2026/6/14
- **ベータ期間中は破壊的変更・仕様の巻き戻しがあり得るため、ビルドごとにSDKヘッダー差分を確認する必要がある**。実例として`call_edit_section()`コールバックの実行スレッドが2026/4/26に「呼び出し元と同じスレッド」に変更された後、2026/4/28に「メインスレッド」へ差し戻された(コメント「色々問題があるので戻します」)。この種の仕様変更が起こり得ることを開発プロセスに織り込む必要がある。

## 3. 先行事例

### (a) 独自UIウィンドウ

- 公式サンプル `WindowClient.cpp`: `RegisterClassEx`→`CreateWindowEx`→子コントロール→`register_window_client`の流れ。本プロジェクトの雛形として直接利用する([SDK_REFERENCE.md](SDK_REFERENCE.md) §13)。
- 【Delphi】AviUtl2用の汎用プラグイン(.aux2)を自作してみた(vram氏・Qiita): https://qiita.com/vram/items/abfba3e8e8c30ec8f480 — Win32 DLL ABIレベルであればC++以外の言語でも実装可能であることの実証。
- その他の.aux2実装例: メモ帳プラグイン(蛇色氏)、TextEditor/MyAssetManager(なたのこ氏)、ScriptBrowserPlugin(Nao氏)、Gradient Editor(azurite氏)。

### (b) 外部プロセス・ローカル通信

- **GCMZDrops2**(oov氏): https://github.com/oov/aviutl2_gcmzdrops2 — D&Dでタイムラインに追加する汎用プラグイン。C＋Lua。Mutex/FileMapping/**WM_COPYDATA**による外部連携API。ローカルIPC中心の設計。**本プロジェクトの「外部AIバックエンドと連携する.aux2」という要件に最も近い先行事例**。
- Discord RPC連携 .aux2(甘味氏、Nanashi.氏): 外部プロセスとのローカルIPCの実例。

### (c) AI連携

- **AviUtl2-WhisperAutoSub**(nkopikaso氏): https://github.com/nkopikaso/AviUtl2-WhisperAutoSub — Whisper自動字幕。**.aux2＋Pythonサブプロセス**(`whisper_helper.py`経由でfaster-whisper実行)＋ffmpeg前処理という構成。「C++プラグインからPython AIバックエンドを使い、結果をタイムラインへ渡す」構成の実証例であり、本プロジェクトのアーキテクチャ(.aux2 ＋ ローカルHTTPのFastAPIバックエンド)の近縁事例。
- vsr4aviutl(Pachira762氏): RTX Video Super Resolution連携。
- 動画生成AIとの直接連携プラグインは**見つからなかった**(先行者なし＝本プロジェクトが先行事例になる)。
- AI支援開発の知見: 三川おさかな氏 https://osakana4242.hatenablog.com/entry/2026/05/03/235651

## 4. 開発環境ベストプラクティス

- Visual Studio 2022標準(複数記事で一致)。※本機はVS Community 2026で代替可能([DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)の環境確認済み事項を参照)。
- 手順(Qiita vngecjasz氏): https://qiita.com/vngecjasz/items/65f4fff15ebde747eb57
  1. DLLプロジェクトを新規作成
  2. SDKファイルをコピーしてプロジェクトに追加
  3. `dllmain.cpp`を削除
  4. Release構成でビルド
  5. プリコンパイル済みヘッダーは不使用にする
  6. `x64\Release`のDLLを`C:\ProgramData\aviutl2\Plugin`に配置し拡張子を変更(`.dll`→`.aux2`等)
- **x64のみ**(複数の実務記事から強く示唆)。
- 文字コードについて「UTF-8必須」という一次情報は無い。APIはワイド文字(UTF-16)主体で、`/utf-8`コンパイラオプションの使用が推奨される。【推測】(詳細な文字コード規約は [SDK_REFERENCE.md](SDK_REFERENCE.md) §11参照)
- 開発支援CLI: [sevenc-nanashi/aviutl2-cli](https://github.com/sevenc-nanashi/aviutl2-cli)(Rust製。本体ダウンロード、成果物のシンボリックリンク配置、リリースパッケージ作成を自動化)。
- デバッグ: VSプロジェクトの「デバッグ>コマンド」にAviUtl2実行ファイルを指定してF5起動＋アタッチする(Zenn goldsmith氏): https://zenn.dev/goldsmith/articles/21f9292bfd8de6
- C# Native AOTでも開発可能(Zenn yamachu氏 https://zenn.dev/yamachu/articles/f3912ea418f530 、窓の杜 https://forest.watch.impress.co.jp/docs/review/chie/2036691.html)。Rustバインディング: [sevenc-nanashi/aviutl2-rs](https://github.com/sevenc-nanashi/aviutl2-rs)。本プロジェクトはC++＋WebView2を選択しているため参考情報にとどまる。

### 推奨デバッグ手順(まとめ)

1. CMakeでビルドしたDLLの拡張子を`.aux2`に変更し、`C:\ProgramData\aviutl2\Plugin`(または直下のサブフォルダ)へ配置する。
2. VS(または生成した.sln)のデバッグ対象実行ファイルにAviUtl2本体(`aviutl2.exe`)を指定し、F5でアタッチ起動する。
3. C++側のクラッシュはVSデバッガで、WebView2内のJS側はWebView2 DevTools(F12相当)で解析する。

## 5. タイムライン追加・プロジェクト連携の知見

- キーAPI: `EDIT_SECTION::create_object_from_alias()`(**INI風**エイリアス〔`[Object]`／`[Object.N]`セクション＋`key=value`行〕からオブジェクト生成)と`create_object_from_media_file()`。手元SDKヘッダ(`plugin2.h`)で行番号付きで裏取り済み([SDK_REFERENCE.md](SDK_REFERENCE.md) §5参照)。
  - **訂正注記(2026-09-04追記・§3-140)**: 本行は初版で「XML形式エイリアス」と書いていたが**誤りである**。書式はINI風で、根拠と経緯は [SDK_REFERENCE.md](SDK_REFERENCE.md) §5.1の訂正注記が正本。
- 排他制御: `EDIT_HANDLE::call_edit_section()`で編集ロックを取得した状態でコールバックが実行される。マルチスレッドでHTTP結果を受け取り→UIスレッド(メインスレッド)で編集操作、というマーシャリングが必須である点は、GCMZDrops2等の既存IPC実装とも設計思想が共通する。
- 2025年8月時点の「入力・出力プラグインのみ」という記事(guest04氏 https://guest04.hatenablog.com/entry/2025/08/08/221521 )は`.aux2`整備前の古い情報であり、現在は当てはまらない。

## 6. 日本語圏情報源一覧(信頼度付き)

| 種別 | タイトル/著者 | URL | 信頼度 |
|---|---|---|---|
| 公式 | AviUtlのお部屋 | https://spring-fragrance.mints.ne.jp/aviutl/ | 【公式】 |
| 公式Docs | AviUtl2 Modern Docs(操作マニュアル、開発APIなし) | https://docs.aviutl2.jp/ | 【公式】 |
| Qiita | Plugin開発備忘録(vngecjasz) | https://qiita.com/vngecjasz/items/65f4fff15ebde747eb57 | 【二次情報】 |
| Qiita | Delphiで.aux2自作(vram) | https://qiita.com/vram/items/abfba3e8e8c30ec8f480 | 【二次情報・実証済み】 |
| Zenn | 入力プラグインのたたき台(goldsmith) | https://zenn.dev/goldsmith/articles/21f9292bfd8de6 | 【二次情報】 |
| Zenn | C#でNative AOTプラグイン(yamachu) | https://zenn.dev/yamachu/articles/f3912ea418f530 | 【二次情報】 |
| はてな | プラグインを作ろうとしてた話(guest04、2025/8/8) | https://guest04.hatenablog.com/entry/2025/08/08/221521 | 【二次情報・古い】 |
| はてな | プラグイン作成をAIに任せる(三川おさかな) | https://osakana4242.hatenablog.com/entry/2026/05/03/235651 | 【二次情報】 |
| 窓の杜 | C#プラグイン解説 | https://forest.watch.impress.co.jp/docs/review/chie/2036691.html | 【二次情報】 |
| 窓の杜 | AviUtl2カタログ紹介 | https://forest.watch.impress.co.jp/docs/serial/yajiuma/2085118.html | 【二次情報】 |
| 5ch | AviUtl ExEdit2 Part1 | https://egg.5ch.net/test/read.cgi/software/1753157841/l50 | 【二次情報・非一次】 |
| GitHub | SDKミラー(aviutl2_sdk_mirror) | https://github.com/aviutl2/aviutl2_sdk_mirror | 【二次情報・実体確認済み】 |
| GitHub | プラグインカタログ(400+件) | https://github.com/Neosku/aviutl2-catalog | 【二次情報】 |
| GitHub | 開発支援CLI(aviutl2-cli) | https://github.com/sevenc-nanashi/aviutl2-cli | 【二次情報】 |
| GitHub | Rustバインディング(aviutl2-rs) | https://github.com/sevenc-nanashi/aviutl2-rs | 【二次情報】 |
| GitHub | GCMZDrops2(oov氏) | https://github.com/oov/aviutl2_gcmzdrops2 | 【二次情報・実証済み】 |
| GitHub | AviUtl2-WhisperAutoSub(nkopikaso氏) | https://github.com/nkopikaso/AviUtl2-WhisperAutoSub | 【二次情報・実証済み】 |
| ハブ | AviUtl2 Hub | https://lineside0418.github.io/AviUtl2_Plugins/ | 【二次情報】 |

## 7. 総括

1. 独自UI＋HTTP＋タイムライン取込を実現するには`.aux2`が正攻法である。公式サンプル`WindowClient.cpp`と、vram氏(Delphi実装)・nkopikaso氏(Python AIバックエンド連携)の実装例が直接参考になる。
2. HTTPサーバー通信は、AviUtl2の編集ロック機構と整合させる設計が肝である。既存事例(GCMZDrops2等)は非同期通信を別スレッド/別プロセスに逃がす点で共通しており、本プロジェクトの「HTTPワーカースレッド→`call_edit_section_param`でメインスレッドへマーシャリング」という設計方針([SDK_REFERENCE.md](SDK_REFERENCE.md) §12)はこの知見と整合する。
3. SDKは活発に更新中であり、実装前に必ず手元SDKヘッダを一次資料として確認する必要がある(本報告のAPI詳細は手元SDKで裏取り済み)。動画生成AIとの連携プラグインという用途自体には直接の先行事例が無く、本プロジェクトが先行事例になる。
