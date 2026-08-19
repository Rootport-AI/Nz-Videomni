#pragma once
//----------------------------------------------------------------------------------
//	スクリプトモジュール ヘッダーファイル for AviUtl ExEdit2
//	By ＫＥＮくん
//----------------------------------------------------------------------------------

//	スクリプトモジュールは下記の関数を外部公開すると呼び出されます
//
//	スクリプトモジュール構造体のポインタを渡す関数 (必須)
//		SCRIPT_MODULE_TABLE* GetScriptModuleTable(void)
//
//	必要とする本体バージョン番号取得関数 (任意)
//		DWORD RequiredVersion() ※必要な本体のバージョン番号を返却します
// 
//	プラグインDLL初期化関数 (任意)
//		bool InitializePlugin(DWORD version) ※versionは本体のバージョン番号
// 
//	プラグインDLL終了関数 (任意)
//		void UninitializePlugin()
// 
//	ログ出力機能初期化関数 (任意) ※logger2.h
//		void InitializeLogger(LOG_HANDLE* logger)
// 
//	設定関連機能初期化関数 (任意) ※config2.h
//		void InitializeConfig(CONFIG_HANDLE* config)
//
//	キャッシュ関連機能初期化関数 ※cache2.h
//		void InitializeCache(CACHE_HANDLE* cache)

//----------------------------------------------------------------------------------

// plugin2.hに定義されています
struct EDIT_SECTION;
struct META_METHOD_FUNCTION;

// 引数の型種別
enum class PARAM_TYPE : int {
	NONE			= -1,
	NIL				= 0,
	BOOLEAN			= 1,
	LIGHTUSERDATA	= 2,
	NUMBER			= 3,
	STRING			= 4,
	TABLE			= 5,
	FUNCTION		= 6,
	USERDATA		= 7,
	THREAD			= 8,
};

// スクリプトモジュール引数構造体
struct SCRIPT_MODULE_PARAM {
	// 引数の数を取得する
	// 戻り値		: 引数の数
	int (*get_param_num)();

	// 引数を整数で取得する
	// index		: 引数の位置(0～)
	// 戻り値		: 引数の値 (取得出来ない場合は0)
	int (*get_param_int)(int index);

	// 引数を浮動小数点で取得する
	// index		: 引数の位置(0～)
	// 戻り値		: 引数の値 (取得出来ない場合は0)
	double (*get_param_double)(int index);

	// 引数を文字列(UTF-8)で取得する
	// index		: 引数の位置(0～)
	// 戻り値		: 引数の値 (取得出来ない場合はnullptr)
	LPCSTR (*get_param_string)(int index);

	// 引数をデータのポインタで取得する ※LightUserData等から取得
	// index		: 引数の位置(0～)
	// 戻り値		: 引数の値 (取得出来ない場合はnullptr)
	void* (*get_param_data)(int index);

	//--------------------------------

	// 引数の連想配列要素を整数で取得する
	// index		: 引数の位置(0～)
	// key			: キー名(UTF-8)
	// 戻り値		: 引数の値 (取得出来ない場合は0)
	int (*get_param_table_int)(int index, LPCSTR key);

	// 引数の連想配列要素を浮動小数点で取得する
	// index		: 引数の位置(0～)
	// key			: キー名(UTF-8)
	// 戻り値		: 引数の値 (取得出来ない場合は0)
	double (*get_param_table_double)(int index, LPCSTR key);

	// 引数の連想配列要素を文字列(UTF-8)で取得する
	// index		: 引数の位置(0～)
	// key			: キー名(UTF-8)
	// 戻り値		: 引数の値 (取得出来ない場合はnullptr)
	LPCSTR (*get_param_table_string)(int index, LPCSTR key);

	//--------------------------------

	// 引数の配列要素の数を取得する
	// index		: 引数の位置(0～)
	// 戻り値		: 配列要素の数
	int (*get_param_array_num)(int index);

	// 引数の配列要素を整数で取得する
	// index		: 引数の位置(0～)
	// key			: 配列のインデックス(0～)
	// 戻り値		: 引数の値 (取得出来ない場合は0)
	int (*get_param_array_int)(int index, int key);

	// 引数の配列要素を浮動小数点で取得する
	// index		: 引数の位置(0～)
	// key			: 配列のインデックス(0～)
	// 戻り値		: 引数の値 (取得出来ない場合は0)
	double (*get_param_array_double)(int index, int key);

	// 引数の配列要素を文字列(UTF-8)で取得する
	// index		: 引数の位置(0～)
	// key			: 配列のインデックス(0～)
	// 戻り値		: 引数の値 (取得出来ない場合はnullptr)
	LPCSTR (*get_param_array_string)(int index, int key);

	//--------------------------------

	// 整数の戻り値を追加する
	// value		: 戻り値
	void (*push_result_int)(int value);

	// 浮動小数点の戻り値を追加する
	// value		: 戻り値
	void (*push_result_double)(double value);

	// 文字列(UTF-8)の戻り値を追加する
	// value		: 戻り値
	void (*push_result_string)(LPCSTR value);

	// データのポインタの戻り値を追加する ※LightUserDataを返却
	// value		: 戻り値
	void (*push_result_data)(void* value);

	//--------------------------------

	// 整数連想配列の戻り値を追加する
	// key			: キー名(UTF-8)の配列
	// value		: 戻り値の配列
	// num			: 配列の要素数
	void (*push_result_table_int)(LPCSTR* key, int* value, int num);

	// 浮動小数点連想配列の戻り値を追加する
	// key			: キー名(UTF-8)の配列
	// value		: 戻り値の配列
	// num			: 配列の要素数
	void (*push_result_table_double)(LPCSTR* key, double* value, int num);

	// 文字列(UTF-8)連想配列の戻り値を追加する
	// key			: キー名(UTF-8)の配列
	// value		: 戻り値の配列
	// num			: 配列の要素数
	void (*push_result_table_string)(LPCSTR* key, LPCSTR* value, int num);

	//--------------------------------

	// 整数配列の戻り値を追加する
	// value		: 戻り値の配列
	// num			: 配列の要素数
	void (*push_result_array_int)(int* value, int num);

	// 浮動小数点配列の戻り値を追加する
	// value		: 戻り値の配列
	// num			: 配列の要素数
	void (*push_result_array_double)(double* value, int num);

	// 文字列(UTF-8)配列の戻り値を追加する
	// value		: 戻り値の配列
	// num			: 配列の要素数
	void (*push_result_array_string)(LPCSTR *value, int num);

	//--------------------------------

	// エラーメッセージを設定する
	// 呼び出された関数をエラー終了する場合に設定します
	// message		: エラーメッセージ(UTF-8)
	void (*set_error)(LPCSTR message);

	//--------------------------------

	// 引数をブール値で取得する
	// index		: 引数の位置(0～)
	// 戻り値		: 引数の値 (取得出来ない場合はfalse)
	bool (*get_param_boolean)(int index);

	// ブール値の戻り値を追加する
	// value		: 戻り値
	void (*push_result_boolean)(bool value);

	// 引数の連想配列要素をブール値で取得する
	// index		: 引数の位置(0～)
	// key			: キー名(UTF-8)
	// 戻り値		: 引数の値 (取得出来ない場合はfalse)
	bool (*get_param_table_boolean)(int index, LPCSTR key);

	// ブール値配列の戻り値を追加する
	// value		: 戻り値の配列
	// num			: 配列の要素数
	void (*push_result_array_boolean)(bool* value, int num);

	// ブール値連想配列の戻り値を追加する
	// key			: キー名(UTF-8)の配列
	// value		: 戻り値の配列
	// num			: 配列の要素数
	void (*push_result_table_boolean)(LPCSTR* key, bool* value, int num);

	//--------------------------------

	// 編集セクション関数
	// スクリプト処理中は参照系の関数が利用出来ます
	EDIT_SECTION* edit;

	//--------------------------------

	// 関数を戻り値として追加する
	// func			: 返却した関数の実行時に呼ばれるコールバック関数
	// userdata		: 任意のユーザーデータのポインタ
	void (*push_result_function)(void (*func)(SCRIPT_MODULE_PARAM*), void* userdata);

	// 新しい関数に差し替えるので廃止します
	void (*deprecated_push_result_meta_table)(void (*func_getter)(SCRIPT_MODULE_PARAM*), void (*func_setter)(SCRIPT_MODULE_PARAM*), void* userdata);

	// 任意のユーザーデータのポインタ
	// push_result_function(),push_result_meta_table()の引数の値が格納されます
	void* userdata;

	// メタテーブルの戻り値を追加する
	// 任意のメタメソッドのコールバック関数を設定したメタテーブルを返却します
	// meta_method_functions	: 登録するメタメソッドの一覧 (META_METHOD_FUNCTIONを列挙してメタメソッド名がnullの要素で終端したリストへのポインタ)
	// userdata					: 任意のユーザーデータのポインタ
	void (*push_result_meta_table)(META_METHOD_FUNCTION* meta_method_functions, void* userdata);

	// 引数のメタテーブルのuserdataのポインタを取得する
	// index					: 引数の位置(0～)
	// meta_method_functions	: 対象のメタテーブルを識別する為のメタメソッドの一覧 ※同一アドレスの場合のみ取得出来ます
	// 戻り値					: userdataのポインタ (取得出来ない場合はnullptr)
	void* (*get_param_meta_table)(int index, META_METHOD_FUNCTION* meta_method_functions);

	// 引数の型を取得します
	// index					: 引数の位置(0～)
	// 戻り値					: 引数の型
	PARAM_TYPE (*get_param_type)(int index);
	
};

// メタメソッド定義構造体
struct META_METHOD_FUNCTION {
	LPCSTR method;						// メタメソッド名 ※luaのメタメソッドを指定出来ます
	void (*func)(SCRIPT_MODULE_PARAM*);	// コールバック関数
};

//----------------------------------------------------------------------------------

// スクリプトモジュール関数定義構造体
struct SCRIPT_MODULE_FUNCTION {
	LPCWSTR name;						// 関数名
	void (*func)(SCRIPT_MODULE_PARAM*);	// 関数へのポインタ
};

// スクリプトモジュール構造体
struct SCRIPT_MODULE_TABLE {
	LPCWSTR information;				// スクリプトモジュールの情報
	SCRIPT_MODULE_FUNCTION* functions;	// 登録する関数の一覧 (SCRIPT_MODULE_FUNCTIONを列挙して関数名がnullの要素で終端したリストへのポインタ)
};
