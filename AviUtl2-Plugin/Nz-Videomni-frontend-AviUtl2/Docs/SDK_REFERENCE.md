# AviUtl2 プラグインSDK リファレンス(.aux2開発向け)

最終更新: 2026-07-07 / 出典: 調査エージェント報告(2026-07-07)

関連ドキュメント: [API_REFERENCE.md](API_REFERENCE.md) ／ [WEB_RESEARCH.md](WEB_RESEARCH.md) ／ [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) ／ [BRIDGE_CONTRACT.md](BRIDGE_CONTRACT.md) ／ [DEVLOG.md](DEVLOG.md)

対象: `aviutl2_sdk/`(本フロントエンド同梱のSDK、2026/7/5版＝beta53a世代)。開発対象の本体は **AviUtl ExEdit2 beta52固定**。両者にバージョン差があるため、本ドキュメント§10「本体バージョン整合表」を必ず確認すること。

**注記（後日追記）**：本書のAPI可否調査はbeta52基準（2026/7/7時点）です。実機ランタイム対象はその後 **v2.0.54ポータブル**（`D:\For_Videos\AviUtl2\aviutl2_v2.0.54\`）へ移行しました。API追加日ベースの可否判定（§10）は引き続き有効な考え方ですが、v2.0.54実機での動作実績は要再確認です。§10表のbeta52基準自体は調査記録として保持します。

注記: SDKヘッダ(`plugin2.h`等)のコメントはShift-JISで書かれている。テキストエディタ/ビルドツールの文字コード設定に注意。本ドキュメントの記述は `plugin2.h` 全文・`config2.h`・`logger2.h`・`aviutl2_plugin_sdk.txt`(更新履歴全文)を実際に読んで裏取り済み。

---

## 1. プラグイン種別と配置

拡張子でプラグイン種別を区別する(`aviutl2_plugin_sdk.txt:13-31`)。

| 種別 | 拡張子 | 用途 |
|---|---|---|
| 入力プラグイン | `.aui2` | 他形式ファイルの読み込み |
| 出力プラグイン | `.auo2` | 他形式への出力 |
| フィルタプラグイン | `.auf2` | メディアオブジェクト・フィルタ効果の追加 |
| スクリプトモジュール | `.mod2` | スクリプトから使う関数群 |
| **汎用プラグイン** | **`.aux2`** | **独自ウィンドウ・インポート/エクスポートメニュー・プロジェクト編集操作の追加** |

**本プロジェクトが実装するのは `.aux2` のみ**。ヘッダは `plugin2.h`。ビルドの雛形は同梱サンプル `WindowClient.cpp`(→§9)。

### インストール・配置(`aviutl2_plugin_sdk.txt:33-81`)

- 単一ファイル(`.aux2`のみ)はAviUtl2のプレビュー画面へD&Dすることで既定フォルダにインストールできる。
- 複数ファイル構成(DLL＋言語ファイル等)は `.au2pkg.zip` としてパッケージ化する。zip内の以下フォルダ配下のみが、アプリケーションデータフォルダからの相対パスで展開される: `Plugin\` `Script\` `Language\` `Alias\` `Figure\` `Transition\` `Preset\` `Default\`。
- パッケージルートに `package.ini`(`[package]` セクションに `id`/`name`/`information`/`uninstallSubFolderFile`)と `package.txt`(インストール時に表示される説明文)を同梱できる。同一`id`のパッケージが既にあれば旧版をアンインストール後に上書きインストールされる。
- 配置例:
  ```
  SamplePackage.au2pkg.zip
  |----Plugin\SamplePackage\SamplePackage.aux2
  |----Language\English.SamplePackage.aul2
  |----package.txt
  |----package.ini
  ```
- 開発時の手動配置先は既定で `C:\ProgramData\aviutl2\Plugin`。プラグインは Plugin フォルダの**1つ下のサブフォルダまで**探索対象(2026/1/25更新で追加された仕様)。

### ビルド前提

- **MSVC専用**(Windows専用、`windows.h`/`commctrl.h`/DirectWrite/D3D11に依存)。**C++17以上**、**x64のみ**。
- DLL拡張子は `.aux2`。エクスポート関数は `EXTERN_C __declspec(dllexport)`。
- APIの文字列型はLPCWSTR(UTF-16)が主。エイリアス・プロジェクトのkey/value・オブジェクト設定値はLPCSTR＝UTF-8(サンプルは `u8R"(...)"` リテラルを使用)。詳細は§11参照。
- `RequiredVersion()` が必要本体バージョン番号を返す。公式サンプル(`WindowClient.cpp`)は `2003300`。**本プロジェクトはbeta52で通る値にM0で実機校正する**(値は本体側の`RequiredVersion`比較ロジックに依存するため事前に断定しない)。

## 2. DLLエクスポート関数一覧(`plugin2.h:7-31`)

汎用プラグインDLLは以下の関数を名前解決可能な形でエクスポートする(先頭2つ以外は任意):

```cpp
COMMON_PLUGIN_TABLE* GetCommonPluginTable(void);          // 任意: プラグイン名と情報
void RegisterPlugin(HOST_APP_TABLE* host);                // 必須: 登録処理
DWORD RequiredVersion();                                  // 任意: 必要本体バージョン
bool InitializePlugin(DWORD version);                     // 任意: DLL初期化(version=本体のバージョン番号)
void UninitializePlugin();                                // 任意: DLL終了
void InitializeLogger(LOG_HANDLE* logger);                // 任意: logger2.h
void InitializeConfig(CONFIG_HANDLE* config);             // 任意: config2.h
void InitializeCache(CACHE_HANDLE* cache);                // 任意: cache2.h
```

呼び出し順序: `InitializeLogger` / `InitializeConfig` / `InitializeCache` は `InitializePlugin()` より**先**に呼ばれる。`RegisterPlugin(host)` が実質的なエントリポイントで、ここで各種`register_*`をまとめて行う。

`COMMON_PLUGIN_TABLE`(`plugin2.h:46-49`): `{ LPCWSTR name; LPCWSTR information; }`。

## 3. HOST_APP_TABLE(`plugin2.h:755-916`)主要API

`RegisterPlugin(HOST_APP_TABLE* host)` で受け取る、本体機能への登録用テーブル。本プロジェクトで使う主要メンバ:

```cpp
void (*register_window_client)(LPCWSTR name, HWND hwnd);                       // 791行
EDIT_HANDLE* (*create_edit_handle)();                                          // 795行
void (*register_import_menu)(LPCWSTR name, void(*cb)(EDIT_SECTION*));          // 780行
void (*register_export_menu)(LPCWSTR name, void(*cb)(EDIT_SECTION*));         // 785行
void (*register_config_menu)(LPCWSTR name, void(*cb)(HWND,HINSTANCE));        // 819行
void (*register_project_load_handler)(void(*)(PROJECT_FILE*));                // 799行
void (*register_project_save_handler)(void(*)(PROJECT_FILE*));                // 803行
void (*register_file_drop_handler)(LPCWSTR name, LPCWSTR filefilter,
        void(*)(EDIT_SECTION*, LPCWSTR file));                                // 873行
void (*register_event_listener)(EVENT_TYPE type, void* param, void(*)(void*)); // 914行
void (*register_layer_menu)(LPCWSTR name, void(*)(EDIT_SECTION*));            // 808行
void (*register_object_menu)(LPCWSTR name, void(*)(EDIT_SECTION*));           // 813行
void (*register_layer_menu_param)(LPCWSTR name, void*, void(*)(void*));       // 853行
void (*register_object_menu_param)(LPCWSTR name, void*, void(*)(void*));      // 860行
void (*register_object_item_menu)(LPCWSTR name, bool allow_effect_only,
        void(*)(EDIT_SECTION*, OBJECT_HANDLE, LPCWSTR effect, LPCWSTR item)); // 888行
void (*register_object_item_menu_param)(LPCWSTR name, bool allow_effect_only,
        void*, void(*)(void*, OBJECT_HANDLE, LPCWSTR effect, LPCWSTR item)); // 897行
```

- **`register_window_client(name, hwnd)`**(791行): 渡したHWNDには自動で `WS_CHILD` が付与され親(ホスト)が設定される(`WS_POPUP`は除去される)。結果として本体パネルとしてドッキング/分離表示される。**本プロジェクトのWebView2ホストウィンドウはこの関数で本体に登録する**。
- **`create_edit_handle()`**(795行): 編集ハンドル(`EDIT_HANDLE*`)を取得する。`RegisterPlugin`内で呼び出し、以後プラグインのライフタイムに渡って保持する(2026/2/23更新で「RegisterPlugin処理内から利用した時の説明」が追記されている＝この時点で呼んでよいことが明文化されている)。
- **`register_event_listener(type, param, cb)`**(914行、**2026/6/14追加**): `EVENT_TYPE`(`UPDATE_OBJECT=1` / `CHANGE_EDIT_FRAME=2` / `CHANGE_EDIT_SCENE=3`、`plugin2.h:121-125`)発生時にコールバックが呼ばれる。**コールバックはイベント専用スレッドから呼ばれ、その中で`call_edit_section()`系は利用できない**(ヘッダ注記どおり)。beta52時点でこのAPIが存在するか未確認のため、**使用前に実機(beta52)で疎通確認すること**(§10参照)。
- **`register_object_menu` / `register_layer_menu`**(813/808行、**2025/11/8追加**)・その `_param` 版(860/853行): 編集画面でオブジェクト／レイヤーを右クリックしたときのコンテキストメニューに項目を追加する。コールバックは `EDIT_SECTION*`(非param版)または任意の`param`(param版)を受け取る。**本プロジェクトで実使用**：`native/src/plugin.cpp`(379-381行の`RegisterTimelineMenu`が`_param`版優先／非param版フォールバックの二経路で登録、582-587行で呼び出し、578行付近に`// REALDEVICE-VERIFY:`コメント)。
- **`register_object_item_menu` / `register_object_item_menu_param`**(888/897行、**2026/5/4追加**): オブジェクト編集の**設定項目単位**の右クリックメニューに項目を追加する。`register_object_menu`(オブジェクト単位)とは別物。コールバック引数に対象の`OBJECT_HANDLE`と`effect`/`item`名(=`get_object_item_value()`と同形式)が渡る。**本プロジェクトでは未使用**：`native/src/`全体を検索したが呼び出し箇所は無い(TIMELINE_ALPHA_REQUIREMENTS.md §3・§6で言及される「設定項目単位」の区別のためSDK上のAPIとして把握されているのみ)。

## 4. EDIT_HANDLE(`plugin2.h:580-709`)— 編集ハンドル

`get_host_app_window()` 以外は `RegisterPlugin()` の実行中には(取得直後の)完全な利用ができない場合がある点に留意しつつ、基本的には取得後いつでも呼び出せる。

```cpp
bool (*call_edit_section)(void (*func_proc_edit)(EDIT_SECTION* edit));                                   // 588行
bool (*call_edit_section_param)(void* param, void (*func_proc_edit)(void* param, EDIT_SECTION* edit));   // 592行
void (*get_edit_info)(EDIT_INFO* info, int info_size);                                                    // 598行
HWND (*get_host_app_window)();                                                                             // 625行
bool (*call_read_section)(void (*func_proc_read_section)(EDIT_SECTION* edit));                            // 640行
bool (*call_read_section_param)(void* param, void (*)(void*, EDIT_SECTION*));                             // 644行
bool (*rendering_scene_video)(int frame, void* param,
    void(*cb)(void* param,int frame,const void* buffer,int width,int height,int pitch));                  // 680行
bool (*rendering_scene_audio)(int frame, void* param, void(*cb)(...));                                     // 692行
void (*wait_rendering_task)();                                                                              // 696行
```

- **`call_edit_section` / `call_edit_section_param`**(588/592行): プロジェクトデータの**編集**用コールバックを呼び出す。編集を排他制御するため**更新ロック状態**でコールバックを実行し、コールバック内で変更したオブジェクトは自動的にUndoへ登録される。**コールバックはメインスレッドから呼ばれる**(下記スレッド規約参照)。`call_edit_section_param`は任意の`param`を引き回せる版。
- **`call_read_section` / `call_read_section_param`**(640/644行): 編集済みデータの**参照専用**アクセス。参照ロック状態でコールバックを実行し、**コールバックは呼び出し元と同じスレッド**で呼ばれる(`call_edit_section`とはスレッドの扱いが異なる点に注意)。
- **`get_edit_info(info, info_size)`**(598行): `EDIT_INFO`を取得。取得のため参照ロックするが、**同一スレッドで既に(更新)ロック状態の場合はそのまま取得できる**(2026/2/14修正で対応済み)。つまり`call_edit_section_param`のコールバック内から呼んでも安全。
- **`rendering_scene_video(frame, param, cb)`**(680行): 現在シーンの指定フレームをレンダリングし、`PIXEL_RGBA`形式の画素バッファ(`width`/`height`/`pitch`)をコールバックへ渡す。**レンダリングはタスクとして積まれるのみで完了は非同期。コールバックはレンダリング専用スレッドから呼ばれる**。バッファはコールバック戻り後に無効になり得るため**即座にコピーする**こと。
- **`wait_rendering_task()`**(696行): レンダリングタスクの完了を待機する。**参照ロック・編集ロック状態(=`call_read_section`/`call_edit_section`系コールバック内)で呼ぶとデッドロックの恐れがある**(ヘッダ注記どおり)。I2Vフレームキャプチャでは、ロック外のワーカースレッドから`rendering_scene_video`を呼び、コールバック内で即コピーしてから後処理する設計にする。

## 5. EDIT_SECTION(`plugin2.h:154-576`)— プロジェクト操作API

`call_edit_section`系/`call_read_section`系コールバックの引数として渡される。「(call_read_section利用不可)」と注記された関数は参照専用コールバック内では呼べない(=編集用)。

### 5.1 動画取り込み・タイムライン追加(核心)

```cpp
OBJECT_HANDLE (*create_object_from_media_file)(LPCWSTR file,int layer,int frame,int length); // 289行
OBJECT_HANDLE (*create_object_from_alias)(LPCSTR alias,int layer,int frame,int length);       // 168行
OBJECT_HANDLE (*create_object)(LPCWSTR effect,int layer,int frame,int length);                // 299行
bool (*is_support_media_file)(LPCWSTR file, bool strict);                                      // 272行
bool (*get_media_info)(LPCWSTR file, MEDIA_INFO* info, int info_size);                         // 279行
OBJECT_HANDLE (*find_object)(int layer,int frame);                                             // 175行
bool (*move_object)(OBJECT_HANDLE,int layer,int frame);                                        // 222行
void (*delete_object)(OBJECT_HANDLE);                                                          // 226行
```

**→ 生成された動画ファイルのパスを `create_object_from_media_file(file, layer, frame, length)` に渡すだけで、指定レイヤー・フレームにオブジェクトとして配置できる。`length=0`指定で長さ自動(メディアの長さから決定)。** 失敗時(サポート外形式/位置の重複)は `nullptr` が返る。

`MEDIA_INFO`(67-72行): `{video_track_num, audio_track_num, total_time, width, height}`。動画/音声トラック数・総時間・解像度。

`create_object_from_alias(alias, layer, frame, length)`(168行): UTF-8の**INI風**エイリアス文字列(`.object`ファイルと同フォーマット。`[Object]`／`[Object.N]`のセクション見出しの下に`key=value`行が並ぶ形式で、各セクションの先頭が`effect.name=<エフェクト名>`)からオブジェクトを生成する。2025/12/6更新で複数オブジェクトのエイリアスデータにも対応。

> **訂正注記（2026-09-04追記・§3-140）**：上記の書式は初版で「XML形式」と書いていたが**誤りである**。根拠はSDK同梱サンプル`aviutl2_sdk\WindowClient.cpp`:69-82のエイリアスリテラル（`[Object]`→`[Object.0]`→`effect.name=テキスト`→`サイズ=150.00`…と続き、XMLタグは1つも現れない）で、実機で採取した`.object`ファイルも同じ書式である。同じ誤記が [`WEB_RESEARCH.md`](WEB_RESEARCH.md) §5にもあり、あわせて訂正した（[`TIMELINE_ALPHA_REQUIREMENTS.md`](TIMELINE_ALPHA_REQUIREMENTS.md) 第0節Bは当初からINI風と正しく書いていた）。

### 5.2 プロジェクト・編集情報

`info`メンバ(`EDIT_SECTION::info`)は`EDIT_INFO*`で、`call_read_section`利用不可の注記がある(=参照専用)。§6参照。

カーソル/表示/選択操作(いずれも編集用、call_read_section不可):

```cpp
void (*set_cursor_layer_frame)(int layer, int frame);      // 305行
void (*set_display_layer_frame)(int layer, int frame);     // 311行
void (*set_select_range)(int start, int end);               // 317行
void (*set_grid_bpm)(float tempo, int beat, float offset); // 323行
```

シーン/レイヤー操作:

```cpp
LPCWSTR (*get_scene_name)();                       // 350行
void (*set_scene_name)(LPCWSTR name);              // 356行
void (*set_scene_size)(int width, int height);     // 362行
void (*set_scene_frame_rate)(int rate, int scale); // 368行
void (*set_scene_sample_rate)(int sample_rate);    // 373行
bool (*get_layer_enable)(int layer);  void (*set_layer_enable)(int, bool); // 378/383行
bool (*get_layer_lock)(int layer);    void (*set_layer_lock)(int, bool);  // 388/393行
LPCWSTR (*get_object_name)(OBJECT_HANDLE);  void (*set_object_name)(OBJECT_HANDLE, LPCWSTR); // 329/334行
```

プロジェクトファイル取得:

```cpp
PROJECT_FILE* (*get_project_file)(EDIT_HANDLE* edit); // 241行 (call_read_section不可)
```

## 6. EDIT_INFO 構造体(`plugin2.h:131-149`)

```cpp
struct EDIT_INFO {
    int width, height;          // シーンの解像度
    int rate, scale;             // シーンのフレームレート(rate/scale)
    int sample_rate;             // シーンのサンプリングレート
    int frame;                    // 現在のカーソルのフレーム番号
    int layer;                    // 現在の選択レイヤー番号
    int frame_max;                // オブジェクトが存在する最大のフレーム番号
    int layer_max;                // オブジェクトが存在する最大のレイヤー番号
    int display_frame_start, display_layer_start;  // 表示開始位置
    int display_frame_num, display_layer_num;       // 表示範囲(目安値)
    int select_range_start, select_range_end;       // フレーム範囲選択(-1=未選択)
    float grid_bpm_tempo; int grid_bpm_beat; float grid_bpm_offset; // 先頭グリッド(BPM)
    int scene_id;                  // シーンID
};
```

**→「AviUtl2の編集中サイズ取得」ボタンは `get_edit_info().width/height` で実装する(64グリッド切り上げ等の丸めはプラグイン側の責務)。** I2Vの現在フレーム番号は `get_edit_info().frame` から取得し、`rendering_scene_video(frame, ...)` に渡す。

## 7. PROJECT_FILE(`plugin2.h:716-750`)— プロジェクト単位の保存

```cpp
LPCSTR (*get_param_string)(LPCSTR key);                       // 721行 (UTF-8)
void (*set_param_string)(LPCSTR key, LPCSTR value);           // 726行 (UTF-8)
bool (*get_param_binary)(LPCSTR key, void* data, int size);   // 733行
void (*set_param_binary)(LPCSTR key, void* data, int size);   // 739行 (4096バイト以下)
void (*clear_params)();                                        // 742行
LPCWSTR (*get_project_file_path)();                            // 748行 (2025/12/2追加)
```

- `register_project_load_handler` / `register_project_save_handler`(HOST_APP_TABLE、799/803行)のコールバック内で読み書きする。
- **プロジェクト単位の設定であり、恒久設定(アプリ全体の設定)には使えない**。恒久設定は`CONFIG_HANDLE::app_data_path`配下に自前ファイルで保持する(§8)。
- `set_param_binary`は**4096バイト以下**という明確な上限がある。大きなデータ(サムネイル等)は保存できない。

## 8. CONFIG_HANDLE(`config2.h`)— 設定・多言語・テーマ

```cpp
struct CONFIG_HANDLE {
    LPCWSTR app_data_path;                                                       // 24行
    LPCWSTR (*translate)(CONFIG_HANDLE* handle, LPCWSTR text);                    // 31行
    LPCWSTR (*get_language_text)(CONFIG_HANDLE*, LPCWSTR section, LPCWSTR text);  // 38行
    FONT_INFO* (*get_font_info)(CONFIG_HANDLE*, LPCSTR key);                      // 44行
    int (*get_color_code)(CONFIG_HANDLE*, LPCSTR key);                            // 49行
    int (*get_layout_size)(CONFIG_HANDLE*, LPCSTR key);                           // 54行
    int (*get_color_code_index)(CONFIG_HANDLE*, LPCSTR key, int index);           // 60行
};
```

- `app_data_path`(24行): アプリデータフォルダのパス。**プラグイン独自設定を書き込むAPIはSDKに無い**ため、恒久設定はここ配下に自前ファイル(JSON等)を作って保存する。
- `translate(handle, text)` / `get_language_text(handle, section, text)`: `.aul2`言語ファイルによる多言語化。`translate`はプラグイン自身のファイル名に対応するセクションを、`get_language_text`は任意セクションを引ける。未定義時は引数の文字列をそのまま返す。
- `get_font_info` / `get_color_code` / `get_color_code_index` / `get_layout_size`: 本体設定ファイル`style.conf`の`[Font]`/`[Color]`/`[Layout]`セクションを参照する。**本体UIとの見た目の統一(ダーク基調・配色・余白)にこれらを使う**。
- `InitializeConfig(CONFIG_HANDLE* config)` は `InitializePlugin()` より先に呼ばれる(§2)。

## 9. LOG_HANDLE(`logger2.h`)

```cpp
struct LOG_HANDLE {
    void (*log)(LOG_HANDLE*, LPCWSTR message);
    void (*info)(LOG_HANDLE*, LPCWSTR message);
    void (*warn)(LOG_HANDLE*, LPCWSTR message);
    void (*error)(LOG_HANDLE*, LPCWSTR message);
    void (*verbose)(LOG_HANDLE*, LPCWSTR message);
};
```

**ログ出力は1024文字で制限される**(ヘッダ冒頭コメントに明記)。長いメッセージは分割するか要約する。

## 10. 本体バージョン整合表

**開発対象の本体は AviUtl ExEdit2 beta52 に固定する。手元SDKは2026/7/5版(beta53a世代のヘッダ)であり本体より新しいため、beta52時点で存在しないAPIを誤って使わないよう注意する。** 関数テーブルは追記式(末尾に追加されるのみ)なので、beta52時点で存在するAPIのみを使えば安全である。`aviutl2_plugin_sdk.txt`の更新履歴から、本プロジェクトが使用する主要APIの追加日を裏取りした結果は以下のとおり。

| API / 機能 | 追加日 | beta52での利用 |
|---|---|---|
| `.aux2`(汎用プラグイン)独自ウィンドウの雛形、`register_window_client` | 2025/10/26 | 安全(beta52 = 2026年公開版はこれより後) |
| `EDIT_HANDLE::call_edit_section_param()` | 2025/11/22 | 安全 |
| `EDIT_HANDLE::get_edit_info()` / `EDIT_SECTION::create_object_from_media_file()` / `create_object()` / `is_support_media_file()` / `get_media_info()` / `get_project_file()` | 2025/12/2 | 安全 |
| `EDIT_SECTION::set_cursor_layer_frame()` | 2025/12/6 | 安全 |
| `EDIT_INFO`のレイヤー編集関連情報、`set_display_layer_frame()`/`set_select_range()`/`set_grid_bpm()` | 2025/12/7 | 安全 |
| `HOST_APP_TABLE::register_object_menu()`/`register_layer_menu()`(非param版・param版とも) | 2025/11/8 | 安全(2025/12追加のAPI群より早い) |
| `RequiredVersion()` の関数定義自体 | 2026/2/14 | **要注意**: この定義追加以降に整備された仕組みなので、beta52がこの仕組みに対応しているかM0で実機確認 |
| `EDIT_HANDLE::call_read_section()`/`call_read_section_param()` | 2026/4/18 | beta52での存在は要確認(比較的新しい追加) |
| `HOST_APP_TABLE::register_object_item_menu()`(param版とも) | 2026/5/4 | beta52での存在は要確認(比較的新しい追加。**本プロジェクトでは未使用**、§3参照) |
| `call_edit_section()`コールバックの実行スレッド仕様 | 2026/4/26に「呼び出し元と同じスレッド」に変更 → 2026/4/28に**「メインスレッド」に差し戻し**(コメント: 「色々問題があるので戻します」) | **現行仕様(メインスレッド)を前提に設計する**。beta52がどちらの版に相当するかは実機ログで要確認 |
| `HOST_APP_TABLE::register_event_listener()` | 2026/6/14 | **未確認。使用前に実機(beta52)で疎通確認すること**。イベントコールバック内では`call_edit_section()`系が使用不可という制約もあわせて要検証 |

**方針**: 中核API(独自ウィンドウ登録・編集コールバック・動画取り込み・EDIT_INFO取得)は2025年10月〜12月に追加済みのため beta52 でも安全に使える可能性が高い。`register_event_listener`のような2026年6月の新しいAPIは、使用前に実機起動して`RegisterPlugin`実行時のログや戻り値で存在確認を行う。SDK呼び出しはネイティブ層の1モジュール(`aviutl_edit`相当)に隔離し、後日のSDK追従を局所化する。

## 11. 文字コード規約

| 対象 | 文字コード | 備考 |
|---|---|---|
| API文字列(`LPCWSTR`引数・戻り値。名前、ファイルパス、翻訳文字列等) | **UTF-16(ワイド文字)** | 大半のAPIがこちら |
| エイリアス文字列(`create_object_from_alias`の`alias`引数)、`get/set_object_item_value`・`get/set_effect_item_value`の`value`、`PROJECT_FILE`の`key`/`value` | **UTF-8(`LPCSTR`)** | サンプルは`u8R"(...)"`リテラルを使用 |
| SDKヘッダファイル自体のコメント | **Shift-JIS** | エディタ/コンパイラの入力エンコーディング設定に注意。関数シグネチャ・構造体名はASCIIなので実害は少ないが、コメントを扱うツール(自動ドキュメント生成等)ではエンコーディング変換が必要 |

変換ユーティリティ(UTF-16↔UTF-8↔Shift-JIS)は1箇所に集約する実装方針とする(詳細は [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) のリスク対策表参照)。

## 12. スレッド規約(重要・厳守)

1. **編集操作は `call_edit_section()` / `call_edit_section_param()` のコールバック内でのみ行う。** コールバックは(現行仕様では)**メインスレッド**から呼ばれ、実行中は更新ロック状態にある。コールバック内で行った変更は自動的にUndoに登録される。
2. **更新ロック下で長時間処理(HTTP通信・ファイルI/O待ち等)を行ってはならない。** コールバックは`create_object_from_media_file`のような短時間の編集APIを呼ぶだけにとどめ、HTTP通信は別スレッド(ワーカースレッド)で完了させてから、その結果(動画ファイルパス等)だけを`call_edit_section_param`へマーシャリングして渡す。
3. **参照専用アクセスは `call_read_section()` / `call_read_section_param()` を使う。** こちらは**呼び出し元と同じスレッド**で同期的に呼ばれる(`call_edit_section`とスレッドの扱いが異なる点に注意)。
4. **`get_edit_info()` は参照ロックを取るが、同一スレッドが既に更新ロック状態(=`call_edit_section_param`コールバック内)であればそのまま取得できる**(2026/2/14修正済み)。よって編集コールバック内から安全に呼べる。
5. **`rendering_scene_video()` / `rendering_scene_audio()` は非同期タスクを積むだけで、コールバックは別スレッド(レンダリング専用スレッド)から呼ばれる。** バッファ(`PIXEL_RGBA`)はコールバック戻り後に無効になり得るため、**コールバック内で即座に自前バッファへコピー**すること。PNG化等の重い後処理はコピー後にワーカースレッドで行う。
6. **`wait_rendering_task()` を参照ロック・編集ロック状態(各コールバック内)で呼んではならない。** デッドロックの恐れがある(ヘッダ注記どおり)。呼ぶ場合はロック外のスレッドで行う。
7. **`register_event_listener()`のコールバックはイベント専用スレッドから呼ばれ、その中では`call_edit_section()`系が利用できない**(ヘッダ注記どおり)。イベントで検知した変更をタイムラインに反映したい場合は、イベントコールバック内ではフラグ/キューへ積むだけにし、別途`call_edit_section_param`を呼び出す設計にする。
8. WebView2およびHTTP通信(WinHTTP)は、いずれもUIスレッド(≒メインスレッド)や編集ロックとは独立したスレッドで動作させ、完了時にのみ`call_edit_section_param`等でメインスレッドへマーシャリングする。

## 13. WindowClient.cpp(独自ウィンドウの雛形)要点

1. `COMMON_PLUGIN_TABLE common_plugin_table = { L"Sample Window Client", L"..." };` をグローバル定義し、`GetCommonPluginTable()`で返す。
2. `RegisterPlugin(HOST_APP_TABLE* host)` 内:
   - `RegisterClassExW` → `CreateWindowEx(..., WS_POPUP, ...)` で自前ウィンドウを生成する(親未指定の状態では`WS_CHILD`のウィンドウを直接作れないため、一旦`WS_POPUP`で作る)。
   - 子コントロールを`CreateWindowEx(WC_BUTTON,...)`等で作成する。高さは`config->get_layout_size(config, "SettingItemHeight")`、ラベルは`config->translate(config, L"...")`で本体UIと統一する。
   - `host->register_window_client(SampleWindowName, hwnd);` で本体に登録する(この時点で`WS_POPUP`が外れ`WS_CHILD`化されドッキング対象になる)。
   - `edit_handle = host->create_edit_handle();` で編集ハンドルを保持しておく。
3. ウィンドウプロシージャでのボタン押下時の実装例(`WindowClient.cpp:93`付近):
   ```cpp
   edit_handle->call_edit_section_param(&msg, [](void*, EDIT_SECTION* edit){
       edit->create_object_from_alias(alias, edit->info->layer, edit->info->frame, 10);
   });
   ```
   `alias`はUTF-8の`u8R"([Object]...)"`リテラル。
4. ログ出力は`logger->log/warn(logger, L"...")`。

**WebView2ホストへの応用**: この自前HWNDを親として`CreateCoreWebView2Controller`を貼り、Reactアプリを表示する。ウィンドウ生成〜`register_window_client`の流れ自体はサンプルと同一で、子コントロールの代わりにWebView2コントローラを配置する。

## 14. 他プラグイン種別(参考・本件では不使用)

- `AviReader.cpp`(`input2.h`): VFW AVI入力。`INPUT_PLUGIN_TABLE`＝flag/name/filefilter/information＋func_open/close/info_get/read_video/read_audio/config。
- `AviSaver.cpp`(`output2.h`): VFW AVI出力。`OUTPUT_PLUGIN_TABLE`。`FLAG_IMAGE=4`で静止画1枚出力。
- `MediaObject.cpp` / `MediaFilter.cpp`(`filter2.h`): `FILTER_ITEM_*`で設定UI宣言、`func_proc_video`で描画。
- `ScriptModule.cpp`(`module2.h`): Lua向け関数群。

## 15. 総合評価

### SDKで実現できる

| 要件 | 手段 |
|---|---|
| 独自UIウィンドウ(WebView2) | `register_window_client` + `WindowClient.cpp`パターン |
| 動画のタイムライン取込 | `create_object_from_media_file` |
| 挿入位置の指定 | `EDIT_INFO::layer/frame`、`set_cursor_layer_frame` |
| 編集中サイズ取得 | `get_edit_info()` |
| I2V用フレーム画像取得 | `rendering_scene_video` |
| 日英切替 | `translate` + `Language/*.aul2` |
| 設定保存 | `app_data_path`配下に自前ファイル / `PROJECT_FILE`(プロジェクト単位のみ) |
| ログ | `LOG_HANDLE`(1024字制限) |
| 本体UIとの統一感 | `get_layout_size`/`get_color_code`/`get_font_info` |

### 自前実装が必要

1. **HTTP通信**: SDKに無し。WinHTTP等で自前実装(→[API_REFERENCE.md](API_REFERENCE.md)参照)。
2. **WebView2**: SDKは素のHWNDのみ提供。WebView2 SDKを別途組み込み、JS↔ネイティブは`PostWebMessage`/`AddHostObjectToScript`で自前ブリッジ。
3. トースト・バナー・進捗UIなど: すべてWeb UI内で描画。
4. 恒久設定ストア: `app_data_path`配下に自前ファイル(`PROJECT_FILE`はプロジェクトを開かないと使えず、かつbinary上限4096バイト)。
5. **スレッド制約の遵守**: 編集操作は`call_edit_section_param`コールバック内(メインスレッド・更新ロック下)のみ。`rendering_scene_video`は別スレッドcb。WebView2/HTTPは別スレッド → 編集反映時は必ず`call_edit_section_param`へマーシャリングする(§12参照)。

### 最小構成(M0の実装目安)

1. `GetCommonPluginTable` / `RequiredVersion` / `InitializeLogger` / `InitializeConfig` を公開。
2. `RegisterPlugin`で自前HWND生成→WebView2貼付→`register_window_client`→`create_edit_handle`保持。
3. WebView2内React UIからバックエンドへHTTP。完了時、JSブリッジ経由でパスを受け取り`call_edit_section_param`内で`create_object_from_media_file`。
4. I2Vは`get_edit_info().frame` + `rendering_scene_video`。
5. サイズ取得は`get_edit_info().width/height`。多言語は`translate`+`.aul2`。設定は`app_data_path`。

## 16. 実機確定知見(2026-07-19〜20 タイムライン右クリック再設計 Phase A / G1・G2・G3実機ゲート)

本節は、右クリック再設計の第1段階でオーナーが**v2.0.54ポータブル実機**で実際に呼び出して確定させたSDK挙動をまとめる。従来の§10は「API追加日 ≦ 本体公開日なら使える」という状況証拠ベースの可否判定だったが、以下は実挙動そのものの一次確認である。一次記録は [`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) 第8節（実機チェックリスト）・[`DEVLOG.md`](DEVLOG.md) §33〜§40。

- **(a) `EDIT_INFO.frame` は赤カーソル（プレビュー表示フレーム）を返す。** 右クリックで移動する灰色カーソルの位置**ではない**。当初は灰色カーソルと一致する見込みだったが、G1実機ゲートで不一致が判明した。§6の「I2Vの現在フレーム番号は `get_edit_info().frame` から取得」という用途はこの意味（表示フレーム）で正しい。
- **(b) `get_mouse_layer_frame` はメニューコールバック内で右クリック位置を返す（正式採用）。** 上記(a)の対策として採用。ポップアップメニュー表示中は親ウィンドウにマウス移動メッセージが届かず、右クリックした地点で位置が凍結されるため、**メニュー内で選択前にマウスを大きく動かしても汚染されず右クリック位置が返る**ことを実機で確認済み。右クリック生成の仮オブジェクト挿入位置はこのAPIで決定する。
- **(c) `create_object_from_alias` と `create_object_from_media_file` は衝突時の挙動が非対称。**
  - `create_object_from_alias(alias, layer, frame, length)` に **`length=0`（長さ自動）** を渡すと、配置位置に既存オブジェクトがあって収まらない場合でも **`null` を返さず、黙って短縮してcreateする**（G1で確定）。当初検討していた「衝突→段階的縮小リトライ」ラダー方式はこの挙動と矛盾するため廃止された。
  - `create_object_from_media_file(file, layer, frame, length)` に **明示尺（`length>0`）** を渡すと、収まらない場合は **`null` を返す**（G3で確定）。完成動画を実尺で置換する経路（🎞挿入）はこちら側の挙動に依存し、`null` 時は `layer_max+1` へ退避する（`usedFallback`）。
- **(d) `layer_max+1` へのcreateはレイヤーが自動確保される。** オブジェクトが存在する最大レイヤー番号＋1（定義上、必ず空いている最前面レイヤー）を指定してcreateすると、そのレイヤーが自動的に確保されてオブジェクトが置ける（フォールバック挿入先として成立することをG2で確認）。
- **(e) `call_edit_section` 1回内の delete＋create はアンドゥ1件にまとまる。** ヘッダ未記載で従来「未確認」だった `call_edit_section` の1トランザクション性を実機で確認した。**テキスト→テキスト**（✨挿入・予約付け替え。G1）と**テキスト→メディア**（🎞置換。G3）の両方で、一連の削除→createがCtrl+Z 1回でまとめて戻せることを確認済み。これにより `updateProvisionalReservation`／`insertMediaForJob`（1セクション内で delete＋create を原子的に行う新設RPC）のアンドゥ設計が成立する。
- **(f) `register_event_listener` は本プラグイン未使用のまま。** §3・§10で「使用前に実機疎通確認」としていた2026/6/14追加のこのAPIは、本再設計でも採用せず未使用のままである。プロジェクトロード検出は `register_project_load_handler`（コールバック内でフラグ／予約検出）で足りており、イベント専用スレッド制約（コールバック内で `call_edit_section` 系が使えない）を負う `register_event_listener` を導入する必要がなかった。

**追記（2026-09-03〜04、§3-140＝🎞挿入オブジェクトが青一色になる症状の調査と修正）**：以下の2点は右クリック再設計とは別の機会に得た知見である。一次記録は [`DEVLOG.md`](DEVLOG.md) §107、実機で採取したエイリアスの突き合わせ結果はバックエンド台帳 [`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §3-140。

- **(g) `動画ファイル`オブジェクトの表示種別は、エイリアス直列化キー`音声付き`（`0`／`1`）で決まる。** 二色リボン（上が映像・下が音声）で描かれるか、右クリックメニューに「音声を分離」が出るかは、この1キーの値で決まる。**`create_object_from_media_file`はこのキーを立てない**ため、同じ動画ファイルでも、このAPIで作ったオブジェクトは`音声付き=0`となり**青一色・「音声を分離」なし**になる（同じファイルをドラッグ＆ドロップで置くと`音声付き=1`）。SDKにはオブジェクトの表示種別を指定・取得するAPIが無いので、**二色リボンを得る唯一の経路は`create_object_from_alias`へこのキーを含むエイリアスを渡すこと**である。§3-140のE1実機採取（2026-09-03、ドロップ由来と🎞挿入由来の`.object`の突き合わせ）で確定した。なお`音声付き`は本体の**UI項目一覧には現れない**が、**エイリアス直列化のキーとしては現役**である（この両立の整理は [`TIMELINE_ALPHA_REQUIREMENTS.md`](TIMELINE_ALPHA_REQUIREMENTS.md) 第0節Bの訂正注記を参照）。
- **(h) `create_object_from_alias`へ明示尺（`length>0`）を渡したときの衝突挙動は未実測のまま。** (c)で確定しているのは`length=0`の側（黙って短縮する）だけで、明示尺を渡した場合に`null`が返るのか黙って短縮されるのかは実機で観測していない。§3-140の実装はどちらかを前提にせず、**エイリアス内の`frame=0,N`ピンと`length`引数の二重ピン＋生成後の実尺検証**（要求より短ければdeleteして`create_object_from_media_file`へ退避）で埋めている。実尺検証は「要求より短いときだけ失敗とみなす」片側判定にしてある。
  - **`OBJECT_LAYER_FRAME.end`は排他的（次フレームの先頭を指す）と確定した（2026-09-04、実機ゲートR1〜R7実行時のログ）。** 実機の`%LOCALAPPDATA%\NzVideomni\logs\plugin.log`に残った`CreateMediaObject`のINFOログ5件（2026-09-04 00:32:14〜00:38:42、layer 0〜3・frame 0/34/81/96・length 120または121の組み合わせ）は、**すべて`end - start`が要求`length`と1の狂いもなく一致した**（例: `requested … frame 34, length 121` → `actual … start 34, end 155`、155-34=121）。コード（`bridge.cpp`の`CreateMediaObject`）が採用している`actual = end - start + 1`という包含前提の計算式では、5件とも`length`より1大きい値になる。**両者が矛盾なく両立するのは`end`が排他（=最後に占有するフレームの次を指す）のときだけ**であり、5件全てで例外なく成立したことから排他と確定できる。`WARN`（`create_object_from_alias`のnull・実尺不足によるdelete＋作り直し）はこの区間に1件も出ておらず、安全網2・3はいずれも実機では発火しなかった。これは本項前段が想定していた「包含として読んだ値が実は排他だった場合は長さを1だけ多く見積もるだけで通過する」という側がそのまま起きていたことの実測的な裏付けでもある。一次記録は[`DEVLOG.md`](DEVLOG.md) §107、台帳クローズはバックエンド[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-140。

## 参照ファイル(絶対パス)

- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\aviutl2_sdk\aviutl2_plugin_sdk.txt`(SDK総論・更新履歴全文)
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\aviutl2_sdk\plugin2.h`(`COMMON_PLUGIN_TABLE:46` / `EDIT_INFO:131` / `EDIT_SECTION:154` / `create_object_from_media_file:289` / `EDIT_HANDLE:580` / `call_edit_section_param:592` / `get_edit_info:598` / `rendering_scene_video:680` / `wait_rendering_task:696` / `PROJECT_FILE:716` / `HOST_APP_TABLE:755` / `register_window_client:791` / `register_event_listener:914`)
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\aviutl2_sdk\WindowClient.cpp` / `English.WindowClient.aul2`
- `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\aviutl2_sdk\config2.h`(`app_data_path:24`) / `logger2.h` / `cache2.h`
- `Mock\AVIUTL2_DESIGN_BRIEF.md`(§5制約・§7実装方針・§11確定デザイン)
