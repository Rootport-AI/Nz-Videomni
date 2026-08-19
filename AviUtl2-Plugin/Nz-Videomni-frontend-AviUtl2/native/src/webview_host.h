// webview_host.h - hosts a WebView2 control inside the plugin window.
//
// Responsibilities:
//   * Create the CoreWebView2 environment asynchronously with a per-user data
//     folder under %LOCALAPPDATA%\NzVideomni\webview2.
//   * Attach a CoreWebView2Controller to the plugin's HWND.
//   * Map the virtual host app.nzvideomni.local to the "webui" folder next to the
//     plugin DLL and navigate to https://app.nzvideomni.local/index.html.
//   * Forward WebView2 web messages to a supplied handler and post the
//     handler's JSON response back to the page.
//   * Track the parent's client rect (WM_SIZE) so the control fills the window.
//   * On any failure (missing webui/index.html, environment creation error),
//     show an error string in a fallback STATIC label and keep the plugin
//     alive.
//
// All methods must be called on the UI thread that owns the parent window.
// ASCII-only source.
#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <functional>
#include <string>

namespace nzvideomni {

class WebViewHost {
public:
    // Handler invoked (on the UI thread) with the incoming web message as a
    // UTF-8 JSON string. Returns the UTF-8 JSON response to post back, or an
    // empty string to post nothing.
    using MessageHandler = std::function<std::string(const std::string&)>;

    // Translates a static UI string through CONFIG_HANDLE::translate. Injected
    // so this module (WebView2-only, no SDK config dependency otherwise) does
    // not need to include config2.h; plugin.cpp supplies a lambda that calls
    // g_config->translate(). When unset, or when the callback returns the
    // input unchanged, the original English text is shown (matches the
    // "undefined key" behaviour of translate() itself).
    using Translator = std::function<std::wstring(const std::wstring&)>;

    // Invoked (on the UI thread) once the first navigation to the Web UI
    // completes: true on success (fallback_label has been hidden), false on
    // failure (fallback_label is showing the error). Optional; set via
    // SetReadyHandler before Initialize() so the callback is armed in time
    // for the async NavigationCompleted event.
    using ReadyHandler = std::function<void(bool success)>;

    WebViewHost() = default;
    ~WebViewHost();

    WebViewHost(const WebViewHost&) = delete;
    WebViewHost& operator=(const WebViewHost&) = delete;

    // Begin asynchronous WebView2 initialization. On synchronous failure (no
    // webui folder, etc.) shows the error in fallback_label and returns false.
    // Asynchronous failures are also reported via fallback_label. `translate`
    // is optional (may be an empty std::function) and is used to localize the
    // fallback error strings shown on WebView2 init failure.
    bool Initialize(HWND parent, HWND fallback_label, MessageHandler handler,
                    Translator translate = Translator());

    // Set the handler invoked when the first navigation completes (see
    // ReadyHandler above). Call before Initialize().
    void SetReadyHandler(ReadyHandler handler);

    // Resize the WebView2 controller to the given client rectangle.
    void Resize(const RECT& client_rect);

    // Post a UTF-8 JSON message to the page. Must be called on the UI thread
    // (used to deliver asynchronous RPC responses). No-op if not ready.
    void PostJson(const std::string& utf8_json);

    // Release the controller/environment. Safe to call multiple times.
    void Shutdown();

private:
    struct Impl;
    Impl* impl_ = nullptr;

    void ShowFallback(const std::wstring& message);

    // Translate a static UI string via translate_, or return it unchanged if
    // no Translator was supplied. Called at each fallback-message call site
    // (on the static portion only) before any dynamic text is appended, so
    // the string given to CONFIG_HANDLE::translate is always an exact,
    // context-free key that a Language/*.aul2 file can match.
    std::wstring Translate(const wchar_t* text) const;

    HWND parent_ = nullptr;
    HWND fallback_label_ = nullptr;
    MessageHandler handler_;
    Translator translate_;
    ReadyHandler on_ready_;
};

// Return the directory containing this plugin DLL (no trailing separator).
// Empty on failure.
std::wstring PluginModuleDir();

}  // namespace nzvideomni
