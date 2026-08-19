// webview_host.cpp - WebView2 hosting implementation. ASCII-only source.
#include "webview_host.h"

#include <objbase.h>
#include <shlwapi.h>  // SHCreateMemStream (in-memory IStream for embedded HTML)
#include <wrl.h>

#include <string>
#include <vector>

#include <wil/com.h>
#include <wil/resource.h>  // wil::unique_cotaskmem_string

#include "WebView2.h"
#include "WebView2EnvironmentOptions.h"

// Contract v7: InjectDroppedPathsIntoRequest (the pure, doctest-covered half
// of the __droppedPaths injection below) lives in bridge_core so it can be
// unit-tested without any WebView2/COM dependency.
#include "bridge_core.h"

#include "log.h"
#include "strconv.h"

namespace nzvideomni {

using Microsoft::WRL::Callback;

// Private state kept out of the header so the SDK COM types do not leak.
struct WebViewHost::Impl {
    wil::com_ptr<ICoreWebView2Controller> controller;
    wil::com_ptr<ICoreWebView2> webview;
    wil::com_ptr<ICoreWebView2Environment> environment;  // for embedded serving
    EventRegistrationToken message_token = {};
    EventRegistrationToken resource_token = {};
    EventRegistrationToken navigation_token = {};
    std::string embedded_html;  // non-empty -> serve UI from this DLL resource
    bool com_initialized = false;
};

namespace {

// The virtual origin the Web UI is served from in both modes (folder mapping
// and embedded resource). Keeping a single origin means the bridge and any
// same-origin assumptions in the page behave identically either way.
constexpr wchar_t kVirtualHost[] = L"app.nzvideomni.local";
constexpr wchar_t kVirtualOrigin[] = L"https://app.nzvideomni.local";
constexpr wchar_t kResourceFilter[] = L"https://app.nzvideomni.local/*";
constexpr wchar_t kIndexUrl[] = L"https://app.nzvideomni.local/index.html";

// HMODULE of this DLL (resolved from a function address inside it), used to look
// up the embedded Web UI resource. Returns nullptr on failure.
HMODULE ThisModule() {
    HMODULE module = nullptr;
    ::GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(&ThisModule), &module);
    return module;
}

// Load the embedded single-file Web UI (RCDATA "NZVIDEOMNI_WEBUI_INDEX") from this
// DLL, if one was linked in (NZVIDEOMNI_EMBED_WEBUI=ON at build time). Returns an
// empty string when no such resource exists (development / folder mode).
std::string LoadEmbeddedWebUi() {
    HMODULE module = ThisModule();
    if (module == nullptr) {
        return std::string();
    }
    HRSRC res = ::FindResourceW(module, L"NZVIDEOMNI_WEBUI_INDEX", RT_RCDATA);
    if (res == nullptr) {
        return std::string();
    }
    const DWORD size = ::SizeofResource(module, res);
    HGLOBAL handle = ::LoadResource(module, res);
    if (handle == nullptr || size == 0) {
        return std::string();
    }
    const void* data = ::LockResource(handle);
    if (data == nullptr) {
        return std::string();
    }
    return std::string(static_cast<const char*>(data), size);
}

// Extract the path component of a "https://app.nzvideomni.local/<path>?<query>"
// URI (leading '/', query/fragment stripped). Returns "/" when there is no
// explicit path. An unexpected / foreign URI yields an empty string.
std::wstring UriPath(const std::wstring& uri) {
    const std::wstring host = std::wstring(kVirtualOrigin);
    if (uri.compare(0, host.size(), host) != 0) {
        return std::wstring();
    }
    std::wstring rest = uri.substr(host.size());
    const size_t cut = rest.find_first_of(L"?#");
    if (cut != std::wstring::npos) {
        rest = rest.substr(0, cut);
    }
    if (rest.empty()) {
        return std::wstring(L"/");
    }
    return rest;
}

// WebResourceRequested handler body (embedded mode): serve the single-file UI
// for "/" and "/index.html" from memory; answer 404 for any other path. Runs on
// the UI thread. `env` and `html` come from the host's retained state.
HRESULT ServeEmbedded(ICoreWebView2Environment* env, const std::string& html,
                      ICoreWebView2WebResourceRequestedEventArgs* args) {
    if (env == nullptr || args == nullptr) {
        return S_OK;  // leave the request to default handling
    }
    wil::com_ptr<ICoreWebView2WebResourceRequest> request;
    if (FAILED(args->get_Request(&request)) || !request) {
        return S_OK;
    }
    wil::unique_cotaskmem_string uri;
    if (FAILED(request->get_Uri(&uri)) || !uri) {
        return S_OK;
    }
    const std::wstring path = UriPath(uri.get());
    const bool is_index = (path == L"/" || path == L"/index.html");

    wil::com_ptr<ICoreWebView2WebResourceResponse> response;
    if (is_index) {
        // SHCreateMemStream copies the bytes into a self-owning stream, so the
        // response can outlive this call without pinning `html`.
        wil::com_ptr<IStream> stream;
        stream.attach(::SHCreateMemStream(
            reinterpret_cast<const BYTE*>(html.data()),
            static_cast<UINT>(html.size())));
        if (!stream) {
            return S_OK;
        }
        if (FAILED(env->CreateWebResourceResponse(
                stream.get(), 200, L"OK",
                L"Content-Type: text/html; charset=utf-8", &response)) ||
            !response) {
            return S_OK;
        }
    } else {
        if (FAILED(env->CreateWebResourceResponse(nullptr, 404, L"Not Found",
                                                  L"", &response)) ||
            !response) {
            return S_OK;
        }
    }
    args->put_Response(response.get());
    return S_OK;
}

// ---------------------------------------------------------------------------
// Contract v7: drag-and-drop file resolution.
//
// WebView2's AllowExternalDrop setting defaults to true (unmodified by this
// project), so a plain DOM drop over the page already works without any
// native drop-target plumbing. What it can't give the page is a full local
// path from a dropped file's JS `File` object, so the page instead calls
// `chrome.webview.postMessageWithAdditionalObjects(json, files)`; the extra
// COM objects arrive here as this message's AdditionalObjects, each
// queryable as an ICoreWebView2File for its real path.
// ---------------------------------------------------------------------------

// Resolves every AdditionalObject of this web message that is a file drop
// (ICoreWebView2File) to its UTF-8 local path, in order. Non-file objects
// (e.g. a dragged link/text) are skipped from the returned paths, but still
// count toward `*has_additional_objects` -- that flag reports whether this
// message carried ANY AdditionalObjects at all (independent of how many
// resolved to real files), because THAT raw presence, not "resolved to >=1
// file", is what InjectDroppedPaths below gates the inject-vs-passthrough
// decision on: a message that attached objects but none happen to be files
// must still be treated as "a real drop was attempted" (force an empty-array
// overwrite), not fall through to the substring/anti-spoofing path.
//
// Returns an empty vector (and *has_additional_objects = false) when `args2`
// is null, the runtime has no AdditionalObjects support for this message, or
// the collection is empty -- never throws.
//
// REALDEVICE-VERIFY: this whole path (AdditionalObjects / ICoreWebView2File)
// is WebView2-runtime-dependent and cannot be exercised by doctest (which
// only covers bridge_core.cpp's WebView2-independent parsing); confirm on a
// real beta52 install that a dropped file's AdditionalObjects entry actually
// queries to ICoreWebView2File and that get_Path() returns the real path.
std::vector<std::string> ExtractDroppedFilePaths(
    ICoreWebView2WebMessageReceivedEventArgs2* args2, bool* has_additional_objects) {
    std::vector<std::string> paths;
    *has_additional_objects = false;
    if (args2 == nullptr) {
        return paths;
    }
    wil::com_ptr<ICoreWebView2ObjectCollectionView> objects;
    if (FAILED(args2->get_AdditionalObjects(&objects)) || !objects) {
        return paths;
    }
    UINT32 count = 0;
    if (FAILED(objects->get_Count(&count))) {
        return paths;
    }
    *has_additional_objects = count > 0;
    for (UINT32 i = 0; i < count; ++i) {
        wil::com_ptr<IUnknown> item;
        if (FAILED(objects->GetValueAtIndex(i, &item)) || !item) {
            continue;
        }
        wil::com_ptr<ICoreWebView2File> file = item.try_query<ICoreWebView2File>();
        if (!file) {
            continue;  // not a file drop (e.g. dragged text/link) - skip it
        }
        wil::unique_cotaskmem_string path;
        if (FAILED(file->get_Path(&path)) || !path) {
            continue;
        }
        paths.push_back(WideToUtf8(path.get()));
    }
    return paths;
}

// Security discipline (contract v7, adversarially reviewed design -- see the
// BLOCKER fix note below for why the gating order matters): decides, per
// message, whether and how to force-overwrite params.__droppedPaths, then
// delegates the actual JSON rewrite to bridge_core's pure, doctest-covered
// InjectDroppedPathsIntoRequest.
//
//   1. This message carries >=1 AdditionalObject (checked FIRST, via
//      `has_additional_objects` -- a real drop, regardless of what the raw
//      JSON text happens to contain) -> ALWAYS inject the resolved paths
//      (possibly still an empty array, if none of the attached objects were
//      files).
//   2. Otherwise, if the raw string merely mentions "__droppedPaths"
//      (anti-spoofing: a page trying to smuggle a fake value with no real
//      drop attached) -> also inject, forced to an empty array.
//   3. Otherwise (no AdditionalObjects, no mention of the key at all) ->
//      pass the original string through unchanged; there is no key for
//      HandleRequestJson to ever see, so nothing to spoof or lose, and the
//      overwhelming majority of ordinary messages take this fast path.
//
// BLOCKER fix (post-review): the original implementation checked the
// substring FIRST and returned early if absent, before ever looking at
// AdditionalObjects. Since a real drop's request carries EMPTY params (`{}`
// from `useFileDrop.ts`'s `requestWithFiles("ui.resolveDroppedFiles", {},
// [file])`) with no "__droppedPaths" substring anywhere in the raw JSON, that
// order meant a genuine drop was ALWAYS falling through the fast path and
// never being injected at all -- `ui.resolveDroppedFiles` would silently
// resolve to `files: []` on every real drop. Checking AdditionalObjects
// presence first (step 1, above) closes that gap: it no longer matters
// whether the raw string mentions the key.
std::string InjectDroppedPaths(ICoreWebView2WebMessageReceivedEventArgs* args,
                               const std::string& request_utf8) {
    wil::com_ptr<ICoreWebView2WebMessageReceivedEventArgs2> args2;
    if (args != nullptr) {
        wil::com_ptr<ICoreWebView2WebMessageReceivedEventArgs> args_ptr(args);
        args2 = args_ptr.try_query<ICoreWebView2WebMessageReceivedEventArgs2>();
    }
    bool has_additional_objects = false;
    const std::vector<std::string> paths =
        ExtractDroppedFilePaths(args2.get(), &has_additional_objects);

    if (has_additional_objects) {
        return InjectDroppedPathsIntoRequest(request_utf8, paths);
    }
    if (request_utf8.find("__droppedPaths") == std::string::npos) {
        return request_utf8;  // fast path: nothing dropped, nothing to neutralize
    }
    return InjectDroppedPathsIntoRequest(request_utf8, paths);  // paths is empty here
}

}  // namespace

std::wstring PluginModuleDir() {
    HMODULE module = nullptr;
    if (!::GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&PluginModuleDir), &module)) {
        return std::wstring();
    }
    wchar_t path[MAX_PATH];
    const DWORD length = ::GetModuleFileNameW(module, path, MAX_PATH);
    if (length == 0 || length >= MAX_PATH) {
        return std::wstring();
    }
    std::wstring full(path, length);
    const size_t slash = full.find_last_of(L"\\/");
    if (slash == std::wstring::npos) {
        return std::wstring();
    }
    return full.substr(0, slash);
}

WebViewHost::~WebViewHost() {
    Shutdown();
}

void WebViewHost::ShowFallback(const std::wstring& message) {
    if (fallback_label_ != nullptr) {
        ::SetWindowTextW(fallback_label_, message.c_str());
        ::ShowWindow(fallback_label_, SW_SHOW);
    }
}

std::wstring WebViewHost::Translate(const wchar_t* text) const {
    if (!translate_) {
        return std::wstring(text);
    }
    return translate_(std::wstring(text));
}

void WebViewHost::SetReadyHandler(ReadyHandler handler) {
    on_ready_ = std::move(handler);
}

bool WebViewHost::Initialize(HWND parent, HWND fallback_label,
                             MessageHandler handler, Translator translate) {
    parent_ = parent;
    fallback_label_ = fallback_label;
    handler_ = std::move(handler);
    translate_ = std::move(translate);
    if (impl_ == nullptr) {
        impl_ = new Impl();
    }

    // Prefer the single-file Web UI embedded in this DLL (NZVIDEOMNI_EMBED_WEBUI=ON).
    // When present it is served from memory and no on-disk webui\ folder is
    // needed; otherwise we fall back to the folder mapping (development mode).
    impl_->embedded_html = LoadEmbeddedWebUi();
    const bool embedded = !impl_->embedded_html.empty();

    const std::wstring dir = PluginModuleDir();
    if (dir.empty()) {
        ShowFallback(Translate(L"Could not resolve the plugin module path."));
        LogError(L"WebViewHost: PluginModuleDir() failed");
        return false;
    }

    const std::wstring webui = dir + L"\\webui";
    if (embedded) {
        LogInfo(std::wstring(L"WebViewHost: serving embedded Web UI from DLL resource (") +
                std::to_wstring(impl_->embedded_html.size()) + L" bytes)");
    } else {
        const std::wstring index = webui + L"\\index.html";
        if (::GetFileAttributesW(index.c_str()) == INVALID_FILE_ATTRIBUTES) {
            ShowFallback(Translate(L"webui\\index.html was not found.") + L"\n" +
                         Translate(L"Expected:") + L" " + index);
            LogWarn(std::wstring(L"WebViewHost: index.html not found at ") + index);
            return false;
        }
        LogInfo(L"WebViewHost: serving Web UI from webui\\ folder (development mode)");
    }

    // WebView2 needs COM initialized (STA) on the calling UI thread. S_FALSE
    // means it was already initialized on this thread; both must be balanced
    // with CoUninitialize. RPC_E_CHANGED_MODE means the thread is already MTA;
    // we then leave COM as-is and still attempt creation.
    const HRESULT co = ::CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (SUCCEEDED(co)) {
        impl_->com_initialized = true;
    }

    const std::wstring user_data = AppDataDir() + L"\\webview2";

    const HWND parent_hwnd = parent_;
    const std::wstring webui_folder = webui;

    const HRESULT hr = ::CreateCoreWebView2EnvironmentWithOptions(
        nullptr, user_data.c_str(), nullptr,
        Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>(
            [this, parent_hwnd, webui_folder](
                HRESULT result, ICoreWebView2Environment* env) -> HRESULT {
                if (FAILED(result) || env == nullptr) {
                    ShowFallback(Translate(L"WebView2 environment creation failed. "
                                          L"Is the WebView2 Runtime installed?"));
                    LogError(L"WebViewHost: environment callback reported failure");
                    return result;
                }

                impl_->environment = env;  // retained for embedded resource serving
                const HRESULT chr = env->CreateCoreWebView2Controller(
                    parent_hwnd,
                    Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(
                        [this, webui_folder](
                            HRESULT cresult,
                            ICoreWebView2Controller* controller) -> HRESULT {
                            if (FAILED(cresult) || controller == nullptr) {
                                ShowFallback(
                                    Translate(L"WebView2 controller creation failed."));
                                LogError(L"WebViewHost: controller callback failed");
                                return cresult;
                            }

                            impl_->controller = controller;
                            impl_->controller->get_CoreWebView2(&impl_->webview);
                            if (!impl_->webview) {
                                ShowFallback(
                                    Translate(L"WebView2 core object unavailable."));
                                LogError(L"WebViewHost: get_CoreWebView2 failed");
                                return E_FAIL;
                            }

                            wil::com_ptr<ICoreWebView2Settings> settings;
                            if (SUCCEEDED(impl_->webview->get_Settings(&settings)) &&
                                settings) {
#ifdef NDEBUG
                                settings->put_AreDevToolsEnabled(FALSE);
#else
                                settings->put_AreDevToolsEnabled(TRUE);
#endif
                                settings->put_IsStatusBarEnabled(FALSE);
                            }

                            if (!impl_->embedded_html.empty()) {
                                // Embedded mode: intercept requests to the virtual
                                // origin and serve index.html from memory (404 for
                                // any other path). Same origin as folder mode, so
                                // the bridge behaves identically.
                                impl_->webview->AddWebResourceRequestedFilter(
                                    kResourceFilter,
                                    COREWEBVIEW2_WEB_RESOURCE_CONTEXT_ALL);
                                impl_->webview->add_WebResourceRequested(
                                    Callback<
                                        ICoreWebView2WebResourceRequestedEventHandler>(
                                        [this](ICoreWebView2*,
                                               ICoreWebView2WebResourceRequestedEventArgs*
                                                   args) -> HRESULT {
                                            return ServeEmbedded(
                                                impl_->environment.get(),
                                                impl_->embedded_html, args);
                                        })
                                        .Get(),
                                    &impl_->resource_token);
                            } else {
                                wil::com_ptr<ICoreWebView2_3> wv3 =
                                    impl_->webview.try_query<ICoreWebView2_3>();
                                if (wv3) {
                                    wv3->SetVirtualHostNameToFolderMapping(
                                        kVirtualHost, webui_folder.c_str(),
                                        COREWEBVIEW2_HOST_RESOURCE_ACCESS_KIND_ALLOW);
                                } else {
                                    LogWarn(L"WebViewHost: ICoreWebView2_3 unavailable; "
                                            L"virtual host mapping skipped");
                                }
                            }

                            impl_->webview->add_WebMessageReceived(
                                Callback<ICoreWebView2WebMessageReceivedEventHandler>(
                                    [this](ICoreWebView2*,
                                           ICoreWebView2WebMessageReceivedEventArgs*
                                               args) -> HRESULT {
                                        LPWSTR raw = nullptr;
                                        if (FAILED(args->get_WebMessageAsJson(&raw)) ||
                                            raw == nullptr) {
                                            return S_OK;
                                        }
                                        const std::string raw_request = WideToUtf8(raw);
                                        ::CoTaskMemFree(raw);
                                        // Contract v7: force-inject this message's real
                                        // dropped-file paths (or [] if none) into
                                        // params.__droppedPaths, overwriting anything the
                                        // page itself sent under that key.
                                        const std::string request =
                                            InjectDroppedPaths(args, raw_request);

                                        std::string response;
                                        if (handler_) {
                                            response = handler_(request);
                                        }
                                        if (!response.empty() && impl_->webview) {
                                            const std::wstring wide =
                                                Utf8ToWide(response);
                                            impl_->webview->PostWebMessageAsJson(
                                                wide.c_str());
                                        }
                                        return S_OK;
                                    }).Get(),
                                &impl_->message_token);

                            // Hide the fallback label once the first navigation
                            // actually succeeds (instead of relying on implicit
                            // Z-order), and re-show it with a localized error if
                            // navigation fails. Either way, forward the result to
                            // any caller-supplied ReadyHandler for logging.
                            impl_->webview->add_NavigationCompleted(
                                Callback<ICoreWebView2NavigationCompletedEventHandler>(
                                    [this](ICoreWebView2*,
                                           ICoreWebView2NavigationCompletedEventArgs*
                                               args) -> HRESULT {
                                        BOOL success = FALSE;
                                        if (args != nullptr) {
                                            args->get_IsSuccess(&success);
                                        }
                                        if (success) {
                                            if (fallback_label_ != nullptr) {
                                                ::ShowWindow(fallback_label_, SW_HIDE);
                                            }
                                            LogInfo(L"WebViewHost: navigation completed "
                                                    L"successfully; fallback label hidden");
                                        } else {
                                            ShowFallback(Translate(
                                                L"The Web UI failed to load."));
                                            LogError(L"WebViewHost: navigation completed "
                                                     L"with failure");
                                        }
                                        if (on_ready_) {
                                            on_ready_(success != FALSE);
                                        }
                                        return S_OK;
                                    }).Get(),
                                &impl_->navigation_token);

                            RECT rc = {};
                            ::GetClientRect(parent_, &rc);
                            impl_->controller->put_Bounds(rc);
                            impl_->controller->put_IsVisible(TRUE);
                            LogInfo(L"WebViewHost: bounds set to " +
                                    std::to_wstring(rc.right - rc.left) + L"x" +
                                    std::to_wstring(rc.bottom - rc.top));

                            impl_->webview->Navigate(kIndexUrl);
                            LogInfo(L"WebView2 ready");
                            return S_OK;
                        }).Get());

                if (FAILED(chr)) {
                    ShowFallback(
                        Translate(L"WebView2 controller creation could not start."));
                    LogError(L"WebViewHost: CreateCoreWebView2Controller failed");
                }
                return chr;
            }).Get());

    if (FAILED(hr)) {
        ShowFallback(Translate(L"WebView2 environment creation failed. "
                              L"Is the WebView2 Runtime installed?"));
        LogError(L"WebViewHost: CreateCoreWebView2EnvironmentWithOptions failed");
        return false;
    }

    LogInfo(L"WebViewHost: environment creation requested");
    return true;
}

void WebViewHost::Resize(const RECT& client_rect) {
    if (impl_ != nullptr && impl_->controller) {
        impl_->controller->put_Bounds(client_rect);
        LogInfo(L"WebViewHost: bounds set to " +
                std::to_wstring(client_rect.right - client_rect.left) + L"x" +
                std::to_wstring(client_rect.bottom - client_rect.top));
    }
}

void WebViewHost::PostJson(const std::string& utf8_json) {
    if (impl_ != nullptr && impl_->webview && !utf8_json.empty()) {
        const std::wstring wide = Utf8ToWide(utf8_json);
        impl_->webview->PostWebMessageAsJson(wide.c_str());
    }
}

void WebViewHost::Shutdown() {
    if (impl_ == nullptr) {
        return;
    }
    if (impl_->controller) {
        impl_->controller->Close();
        impl_->controller = nullptr;
    }
    impl_->webview = nullptr;
    impl_->environment = nullptr;
    if (impl_->com_initialized) {
        ::CoUninitialize();
        impl_->com_initialized = false;
    }
    delete impl_;
    impl_ = nullptr;
}

}  // namespace nzvideomni
