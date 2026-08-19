// bridge.h - connects the WebView2 message channel to the RPC core (contract v2).
//
// The Bridge turns an incoming web message (UTF-8 JSON) into a response
// (UTF-8 JSON). Synchronous methods (ping, getEditInfo, backend.getBaseUrl,
// timeline.insertMedia, settings.get, settings.set) are answered inline and
// returned from HandleMessage. Asynchronous methods (backend.request,
// backend.downloadVideo, backend.uploadFile, timeline.captureFrame,
// timeline.cutoutRange, timeline.extractAudio, and contract v6's
// fs.listFiles / fs.probeAudioDuration) are dispatched to the HTTP worker pool; HandleMessage
// returns an empty string and, when the worker finishes, the response is
// delivered through the ResponsePoster supplied to Initialize() (which is
// expected to marshal onto the UI thread and call PostWebMessageAsJson).
// ui.pickFile / ui.pickFolder (contract v6) are handled inline like the
// synchronous methods, but show a modal Win32 dialog on the UI thread instead
// of consulting the pure RPC core.
//
// The backend base URL used by backend.request / backend.downloadVideo /
// backend.uploadFile / backend.getBaseUrl is read from the persistent
// SettingsStore (settings.h, contract v4) at the time each request is
// dispatched - see CurrentBaseUrl(). A settings.set does not affect a request
// already in flight.
//
// Threading:
//   * HandleMessage runs on the WebView2 UI thread.
//   * HTTP work runs on HttpClient worker threads.
//   * timeline.insertMedia runs on the UI thread (call_edit_section_param must
//     execute on the main thread per the SDK threading rules), so it is handled
//     synchronously inside HandleMessage.
//
// ASCII-only source.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>

struct EDIT_HANDLE;  // forward declared; defined in plugin2.h

namespace nzltx {

class HttpClient;
class SettingsStore;

class Bridge {
public:
    // Delivers a finished async response (UTF-8 JSON) back to the web page.
    // Invoked from an HTTP worker thread; the implementation must be thread-safe
    // and marshal onto the UI thread before touching WebView2.
    using ResponsePoster = std::function<void(const std::string&)>;

    Bridge();
    ~Bridge();

    Bridge(const Bridge&) = delete;
    Bridge& operator=(const Bridge&) = delete;

    // Create the HTTP worker pool and record how async responses are delivered.
    // Call once from RegisterPlugin (never during static init / DllMain).
    void Initialize(ResponsePoster poster);

    // Store the host edit handle (may be nullptr until it is created).
    void SetEditHandle(EDIT_HANDLE* handle);

    // Store the plugin's top-level window handle, used as the owner of the
    // native Open dialog shown by ui.pickFile. Passed as an opaque void* so this
    // header need not pull in <windows.h>. May be nullptr until the window
    // exists.
    void SetPluginHwnd(void* hwnd);

    // Release worker threads. Safe to call multiple times.
    void Shutdown();

    // Handle one incoming web message. Returns the JSON response to post back
    // synchronously, or an empty string when there is nothing to post now
    // (async method in flight, or a non-request event). Never throws.
    std::string HandleMessage(const std::string& request_json);

private:
    // The current backend base URL: the settings store's value once
    // Initialize() has run and loaded it, or the built-in default otherwise.
    std::string CurrentBaseUrl() const;

    EDIT_HANDLE* edit_handle_ = nullptr;
    void* plugin_hwnd_ = nullptr;  // HWND owner for ui.pickFile (opaque here)
    std::unique_ptr<HttpClient> http_;
    // Persistent settings (contract v4: currently just the backend base URL).
    // Created lazily in Initialize() (never during static init), same as
    // http_. Outlives Shutdown() so the value survives a WM_DESTROY/recreate.
    std::unique_ptr<SettingsStore> settings_;
    ResponsePoster poster_;
    // Guards against a re-entrant ui.pickFile: the Open dialog runs its own modal
    // message pump on the UI thread, during which another web message could be
    // delivered. A second pickFile while one is open is rejected as DIALOG_FAILED.
    std::atomic<bool> pick_dialog_open_{false};
};

}  // namespace nzltx
