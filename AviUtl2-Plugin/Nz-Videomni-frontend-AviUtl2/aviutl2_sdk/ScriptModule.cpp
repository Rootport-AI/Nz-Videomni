//----------------------------------------------------------------------------------
//	サンプルスクリプトモジュールプラグイン for AviUtl ExEdit2
//----------------------------------------------------------------------------------
#include <windows.h>
#include <algorithm>

#include "module2.h"
#include "logger2.h"
#include "filter2.h" // PIXEL_RGBA定義用 

LOG_HANDLE* logger;

//---------------------------------------------------------------------
//	ログ出力機能初期化関数 (未定義なら呼ばれません)
//---------------------------------------------------------------------
EXTERN_C __declspec(dllexport) void InitializeLogger(LOG_HANDLE* handle) {
	logger = handle;
}

//---------------------------------------------------------------------
//	プラグインDLL初期化関数 (未定義なら呼ばれません)
//---------------------------------------------------------------------
EXTERN_C __declspec(dllexport) bool InitializePlugin(DWORD version) {
	return true;
}

//---------------------------------------------------------------------
//	プラグインDLL解放関数 (未定義なら呼ばれません)
//---------------------------------------------------------------------
EXTERN_C __declspec(dllexport) void UninitializePlugin() {
}

//---------------------------------------------------------------------
//	合計を計算するサンプル関数
//---------------------------------------------------------------------
void sum(SCRIPT_MODULE_PARAM* param) {
	// 引数の合計を計算
	double total = 0.0;
	auto num = param->get_param_num();
	for (int i = 0; i < num; i++) {
		total += param->get_param_double(i);
	}
	param->push_result_double(total);
}

//---------------------------------------------------------------------
//	明るさを調整するサンプル関数
//---------------------------------------------------------------------
void luminance(SCRIPT_MODULE_PARAM* param) {
	// 引数を取得
	auto n = param->get_param_num();
	if (n != 4) {
		param->set_error(u8"引数の数が正しくありません");
		return;
	}
	auto p = (PIXEL_RGBA*)param->get_param_data(0);
	auto w = param->get_param_int(1);
	auto h = param->get_param_int(2);
	auto v = param->get_param_double(3);
	if (!p || w <= 0 || h <= 0) {
		param->set_error(u8"引数の値が正しくありません");
		return;
	}

	// 明るさを調整
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			p->r = (unsigned char)std::clamp(p->r * v, 0.0, 255.0);
			p->g = (unsigned char)std::clamp(p->g * v, 0.0, 255.0);
			p->b = (unsigned char)std::clamp(p->b * v, 0.0, 255.0);
			p++;
		}
	}
}

//---------------------------------------------------------------------
//	メタテーブルを返却するサンプル関数
//---------------------------------------------------------------------

void meta_table_getter(SCRIPT_MODULE_PARAM* param);
void meta_table_setter(SCRIPT_MODULE_PARAM* param);
void meta_table_add(SCRIPT_MODULE_PARAM* param);
void meta_table_func(SCRIPT_MODULE_PARAM* param);
void meta_table_gc(SCRIPT_MODULE_PARAM* param);

META_METHOD_FUNCTION meta_methods[] = {	// luaのメタメソッドのコールバックを定義出来ます
	{"__index", meta_table_getter },
	{"__newindex", meta_table_setter },
	{"__add", meta_table_add },
	{"__call", meta_table_func },
	{"__gc", meta_table_gc },
	{ nullptr }
};

struct UserData { // メタテーブルに関連付ける任意のユーザーデータ
	double x = 0.0;
	double y = 0.0;
	double z = 0.0;
};

void meta_table_new(SCRIPT_MODULE_PARAM* param, double x = 0.0, double y = 0.0, double z = 0.0) {
	// メタテーブルにUserDataを関連付けて返却
	auto userdata = new UserData();
	userdata->x = x;
	userdata->y = y;
	userdata->z = z;
	param->push_result_meta_table(meta_methods, userdata);
	// デバッグログ出力
	WCHAR buf[256];
	swprintf_s(buf, 256, L"new : %p", userdata);
	logger->log(logger, buf);
}

void meta_table_gc(SCRIPT_MODULE_PARAM* param) {
	// メタテーブルに関連付けたUserDataを破棄する
	delete (UserData*)param->userdata;
	// デバッグログ出力
	WCHAR buf[256];
	swprintf_s(buf, 256, L"gc : %p", param->userdata);
	logger->log(logger, buf);
}

void meta_table_func(SCRIPT_MODULE_PARAM* param) {
	auto userdata = (UserData*)param->userdata;
	// デバッグログ出力
	WCHAR buf[256];
	swprintf_s(buf, 256, L"call func : %lf,%lf,%lf", userdata->x, userdata->y, userdata->z);
	logger->log(logger, buf);
}

void meta_table_getter(SCRIPT_MODULE_PARAM* param) {
	if (param->get_param_num() != 2) return;
	auto key = param->get_param_string(1); // キー名がindex=1に入る
	auto userdata = (UserData*)param->userdata;
	if (strcmp(key, "func") == 0) {
		param->push_result_function(meta_table_func, userdata); // "func"の場合は関数を返却
	} else if (strcmp(key, "x") == 0) {
		param->push_result_double(userdata->x);
	} else if (strcmp(key, "y") == 0) {
		param->push_result_double(userdata->y);
	} else if (strcmp(key, "z") == 0) {
		param->push_result_double(userdata->z);
	}
}

void meta_table_setter(SCRIPT_MODULE_PARAM* param) {
	if (param->get_param_num() != 3) return;
	auto key = param->get_param_string(1); // キー名がindex=1に入る
	auto value = param->get_param_double(2); // 値がindex=2に入る
	auto userdata = (UserData*)param->userdata;
	if (strcmp(key, "x") == 0) {
		userdata->x = value;
	} else if (strcmp(key, "y") == 0) {
		userdata->y = value;
	} else if (strcmp(key, "z") == 0) {
		userdata->z = value;
	}
}

void meta_table_add(SCRIPT_MODULE_PARAM* param) {
	auto a = (UserData*)param->get_param_meta_table(0, meta_methods);
	auto b = (UserData*)param->get_param_meta_table(1, meta_methods);
	if (!a || !b) {
		param->set_error(u8"引数が正しくありません");
		return;
	}
	meta_table_new(param, a->x + b->x, a->y + b->y, a->z + b->z);
}

void meta_table(SCRIPT_MODULE_PARAM* param) {
	if (param->get_param_num() == 3) { // 引数がある場合は初期値を入れる
		meta_table_new(param, param->get_param_double(0), param->get_param_double(1), param->get_param_double(2));
	} else {
		meta_table_new(param);
	}
}

//---------------------------------------------------------------------
//	スクリプトモジュール関数リスト定義
//---------------------------------------------------------------------
SCRIPT_MODULE_FUNCTION functions[] = {
	{ L"sum", sum },
	{ L"luminance", luminance },
	{ L"new", meta_table },
	{ nullptr }
};

//---------------------------------------------------------------------
//	スクリプトモジュール構造体定義
//---------------------------------------------------------------------
SCRIPT_MODULE_TABLE script_module_table = {
	L"Sample ScriptModule version 2.00 By ＫＥＮくん",	// モジュールの情報
	functions
};

//---------------------------------------------------------------------
//	スクリプトモジュール構造体のポインタを渡す関数
//---------------------------------------------------------------------
EXTERN_C __declspec(dllexport) SCRIPT_MODULE_TABLE* GetScriptModuleTable(void) {
	return &script_module_table;
}
