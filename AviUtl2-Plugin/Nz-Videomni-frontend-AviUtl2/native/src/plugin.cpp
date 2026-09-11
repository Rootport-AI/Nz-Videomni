//----------------------------------------------------------------------------------
//  Nz-Videomni - AviUtl2 general-purpose (.aux2) plugin, milestone M1.
//
//  A dockable window client that hosts a WebView2 control. The Web UI is served
//  from the "webui" folder shipped next to this DLL through the virtual host
//  https://app.nzvideomni.local/index.html, and talks to native code over a
//  postMessage-based JSON-RPC bridge (contract v1: ping, getEditInfo).
//
//  If WebView2 cannot start (runtime missing) or the webui folder is absent,
//  a STATIC fallback label shows the error and the plugin stays alive.
//
//  NOTE: This translation unit is intentionally ASCII-only (including all
//  comments) so it compiles cleanly regardless of the compiler's default
//  code page, while the Shift-JIS SDK headers are parsed under that same
//  default code page (no /utf-8 is used - see CMakeLists.txt).
//----------------------------------------------------------------------------------
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <commctrl.h>

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "plugin2.h"  // COMMON_PLUGIN_TABLE, HOST_APP_TABLE, EDIT_HANDLE, EDIT_SECTION
#include "logger2.h"  // LOG_HANDLE
#include "config2.h"  // CONFIG_HANDLE

#include "json.hpp"        // nlohmann::json (build the timeline.menuInvoked event)
#include "bridge.h"
#include "bridge_core.h"   // SelectionSnapshot, SelectionItem, MakeSelectionResult
#include "log.h"
#include "strconv.h"       // Utf8ToWide / WideToUtf8
#include "webview_host.h"

//---------------------------------------------------------------------
//  Constants
//---------------------------------------------------------------------
#define NZVIDEOMNI_WINDOW_NAME L"Nz-Videomni"

namespace {

constexpr int IDC_STATUS_LABEL = 1002;

// Posted (from an HTTP worker thread) to marshal a finished async RPC response
// onto the UI thread. lparam is a heap-allocated std::string* to be freed here.
constexpr UINT WM_NZVIDEOMNI_POST_RESPONSE = WM_APP + 17;

// Host-provided handles (retained for the lifetime of the plugin).
EDIT_HANDLE*   g_edit_handle = nullptr;
LOG_HANDLE*    g_logger      = nullptr;
CONFIG_HANDLE* g_config      = nullptr;

// Child controls / hosted components.
HWND                g_status_label = nullptr;  // fallback error/status label
// Plugin top-level window, retained so the non-capturing timeline menu / project
// callbacks (plain function pointers, no captures allowed) can PostMessage a
// native-initiated event onto the UI thread the same way async RPC replies do.
HWND                g_plugin_hwnd = nullptr;
nzvideomni::Bridge       g_bridge;
nzvideomni::WebViewHost  g_webview_host;

// Backing storage for translated menu labels passed to
// register_object_menu[_param] / register_layer_menu[_param]. The host does
// NOT copy the LPCWSTR it is given (it keeps the raw pointer and reads it
// back later when the menu is actually shown), so each translated label must
// live for the plugin's lifetime. std::deque is used (not std::vector)
// because push_back on a deque never reallocates/moves existing elements, so
// pointers handed out by earlier push_back calls (across the object- and
// layer-menu registration passes) stay valid.
std::deque<std::wstring> g_menu_label_storage;

// Translate a static UI string through CONFIG_HANDLE::translate (contract v4
// Language files, see Language/*.NzVideomni.aul2). Falls back to the original
// English text when no config handle is available yet or the host returns
// null, matching translate()'s own "undefined key" behaviour.
std::wstring Translate(LPCWSTR text) {
    if (g_config != nullptr && g_config->translate != nullptr) {
        LPCWSTR translated = g_config->translate(g_config, text);
        if (translated != nullptr) {
            return std::wstring(translated);
        }
    }
    return std::wstring(text);
}

//---------------------------------------------------------------------
//  General-purpose plugin table returned to the host.
//---------------------------------------------------------------------
COMMON_PLUGIN_TABLE g_common_plugin_table = {
    L"Nz-Videomni",                                                      // plugin name
    L"LTX 2.3 video generation frontend (1.0.0)",                // plugin information
};

// Apply the current client rect to the fallback label and the WebView2
// controller. Shared by WM_SIZE and WM_WINDOWPOSCHANGED so the panel keeps
// following the parent's client area even when the host resizes/docks it
// without sending WM_SIZE (e.g. some panel-visibility transitions).
void ApplyClientBounds(HWND hwnd) {
    RECT rc = {};
    ::GetClientRect(hwnd, &rc);
    if (g_status_label != nullptr) {
        ::MoveWindow(g_status_label, 0, 0, rc.right, rc.bottom, TRUE);
    }
    g_webview_host.Resize(rc);
}

//---------------------------------------------------------------------
//  Window procedure.
//---------------------------------------------------------------------
LRESULT CALLBACK WndProc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
    switch (message) {
        case WM_SIZE: {
            ApplyClientBounds(hwnd);
            return 0;
        }
        case WM_WINDOWPOSCHANGED: {
            ApplyClientBounds(hwnd);
            // Fall through to DefWindowProc (not `return 0`) so the default
            // handling that derives WM_SIZE / WM_MOVE from this message is
            // not suppressed.
            break;
        }
        case WM_NZVIDEOMNI_POST_RESPONSE: {
            // Take ownership of the response string posted by a worker thread and
            // deliver it to the page on this (UI) thread.
            std::unique_ptr<std::string> payload(
                reinterpret_cast<std::string*>(lparam));
            if (payload != nullptr) {
                g_webview_host.PostJson(*payload);
            }
            return 0;
        }
        case WM_DESTROY:
            g_bridge.Shutdown();
            g_webview_host.Shutdown();
            break;
        default:
            break;
    }
    return ::DefWindowProcW(hwnd, message, wparam, lparam);
}

//---------------------------------------------------------------------
//  Contract v5: timeline context-menu + project-load wiring.
//
//  These are pure ADDITIONS to the existing window-client / captureFrame /
//  insertMedia / backend plumbing above - none of that is changed. The menu and
//  project-load callbacks must be non-capturing plain function pointers (the
//  host stores the address), so they route everything through file-scope
//  globals and helpers rather than lambdas with state.
//---------------------------------------------------------------------

// Marshal a native-initiated JSON message (an event here, but any UTF-8 JSON)
// onto the UI thread and hand it to the page, reusing the exact
// WM_NZVIDEOMNI_POST_RESPONSE path the async RPC responses already use
// (WndProc -> WebViewHost::PostJson). Safe to call from any thread; delivery is
// deferred to the UI message queue, so it never re-enters WebView2 while the
// host is still processing the menu click. No-op before the window exists; if
// WebView2 is not ready yet, PostJson itself is a documented no-op.
void PostJsonToPage(std::string json_utf8) {
    if (g_plugin_hwnd == nullptr) {
        return;
    }
    auto* payload = new std::string(std::move(json_utf8));
    if (!::PostMessageW(g_plugin_hwnd, WM_NZVIDEOMNI_POST_RESPONSE, 0,
                        reinterpret_cast<LPARAM>(payload))) {
        delete payload;
    }
}

// Fill a SelectionSnapshot from a live EDIT_SECTION the same way the bridge's
// timeline.getSelection provider does (GetSelectionEditProc in bridge.cpp),
// EXCEPT it does not parse media file paths out of the object alias: that step
// needs bridge.cpp's private ExtractMediaFilePath plus the Japanese
// effect-name byte tables, deliberately NOT duplicated in this ASCII-only
// translation unit. has_file_path therefore stays false (serialized as
// filePath:null); the WebUI calls timeline.getSelection to obtain authoritative
// media paths when a flow needs them. Everything done here is a cheap
// read-only snapshot - no HTTP / render / encode - so it is safe to run inside a
// menu callback per the SDK thread rules.
void SnapshotSelection(EDIT_SECTION* edit, nzvideomni::SelectionSnapshot* snap,
                       bool use_mouse_frame) {
    snap->available = true;
    if (edit->info != nullptr) {
        snap->rate = edit->info->rate;
        snap->scale = edit->info->scale;
        snap->sample_rate = edit->info->sample_rate;
        snap->cursor_frame = edit->info->frame;
        snap->cursor_layer = edit->info->layer;
        // Use get_mouse_layer_frame to obtain the right-click position (verified
        // on real hardware 2026-07-19): while a popup menu is up the parent
        // window receives no further WM_MOUSEMOVE, so get_mouse_layer_frame -
        // which reports "the last mouse-move window message" coordinate - stays
        // FROZEN at the right-click point. Confirmed to return the right-click
        // spot even when the mouse is moved far across the menu before a choice
        // is made, so the placeholder lands exactly where the user clicked
        // (EDIT_INFO.frame/layer instead track the RED displayed-frame caret,
        // which is the wrong spot for a right-click insert). On false (or a null
        // function pointer) we fall back to the EDIT_INFO cursor.
        //
        // use_mouse_frame lets the caller keep the RED displayed-frame caret as
        // cursor_frame while still taking the mouse LAYER: the current-frame
        // actions (imageFromCurrentFrame / addCurrentFrameAsKeyframe /
        // currentFrameToClipChain) operate on the frame shown in the preview
        // (EDIT_INFO.frame), not the right-click column, so for them only the
        // layer is snapped to the mouse and cursor_frame stays the caret.
        if (edit->get_mouse_layer_frame != nullptr) {
            int mouse_layer = 0;
            int mouse_frame = 0;
            if (edit->get_mouse_layer_frame(&mouse_layer, &mouse_frame)) {
                snap->cursor_layer = mouse_layer;
                if (use_mouse_frame) {
                    snap->cursor_frame = mouse_frame;
                }
            }
        }
        if (edit->info->select_range_start >= 0 && edit->info->select_range_end >= 0) {
            snap->has_range = true;
            snap->range_start = edit->info->select_range_start;
            snap->range_end = edit->info->select_range_end;
        }
    }
    OBJECT_HANDLE focus =
        edit->get_focus_object != nullptr ? edit->get_focus_object() : nullptr;
    std::vector<OBJECT_HANDLE> objects;
    if (focus != nullptr) {
        objects.push_back(focus);
    } else if (edit->get_selected_object_num != nullptr &&
               edit->get_selected_object != nullptr) {
        const int n = edit->get_selected_object_num();
        for (int i = 0; i < n; ++i) {
            OBJECT_HANDLE o = edit->get_selected_object(i);
            if (o != nullptr) objects.push_back(o);
        }
    }
    for (OBJECT_HANDLE o : objects) {
        nzvideomni::SelectionItem item;
        if (edit->get_object_layer_frame != nullptr) {
            const OBJECT_LAYER_FRAME lf = edit->get_object_layer_frame(o);
            item.layer = lf.layer;
            item.frame_start = lf.start;
            item.frame_end = lf.end;
        }
        if (edit->get_object_name != nullptr) {
            const wchar_t* nm = edit->get_object_name(o);  // valid during callback
            if (nm != nullptr && nm[0] != L'\0') {
                item.has_object_name = true;
                item.object_name = nzvideomni::WideToUtf8(nm);  // copy immediately
            }
        }
        snap->selected.push_back(std::move(item));
    }
}

// Build and push the timeline.menuInvoked event (contract v5, see
// webui/src/bridge/types.ts). `action` is a stable ASCII identifier for the
// invoked command; `selection` is serialized through the shared public
// MakeSelectionResult() so its JSON shape can never drift from
// timeline.getSelection. Never throws into the host callback.
void EmitMenuInvoked(const char* action, EDIT_SECTION* edit) {
    try {
        nzvideomni::SelectionSnapshot snap;
        if (edit != nullptr) {
            // The current-frame (camera) actions operate on the RED
            // displayed-frame caret (EDIT_INFO.frame), not the right-click
            // column, so they freeze cursor_frame to the caret (mouse LAYER is
            // still used). All other items keep the historical right-click frame.
            const std::string a(action);
            const bool use_mouse_frame =
                !(a == "imageFromCurrentFrame" ||
                  a == "addCurrentFrameAsKeyframe" ||
                  a == "currentFrameToClipChain");
            SnapshotSelection(edit, &snap, use_mouse_frame);
        }
        nzvideomni::json_t data;
        data["action"] = action;
        data["selection"] = nzvideomni::MakeSelectionResult(snap);
        nzvideomni::json_t ev;
        ev["event"] = "timeline.menuInvoked";
        ev["data"] = std::move(data);
        PostJsonToPage(ev.dump());
        nzvideomni::LogInfo(std::wstring(L"timeline.menuInvoked pushed: action=") +
                       nzvideomni::Utf8ToWide(action));
    } catch (...) {
        // A C++ catch (not SEH) so object unwinding is allowed here (no C2712);
        // nothing may propagate out of a host-invoked plain callback.
        nzvideomni::LogError(L"EmitMenuInvoked failed (exception swallowed)");
    }
}

// --- Non-capturing menu callbacks (plain function pointers for the host). ----
// The alpha feature set exposes several timeline commands. Every menu item does
// the SAME lightweight thing: snapshot the selection and post a
// timeline.menuInvoked event carrying its own stable `action` id. No HTTP /
// render / encode is started here (SDK thread rules forbid heavy work in these
// callbacks); the WebUI turns the action + selection into a flow.
//
// Two host registration paths exist (see plugin2.h) and we support BOTH:
//
//   * register_object_menu / register_layer_menu  -- the callback is handed a
//     live EDIT_SECTION* directly. One plain (non-capturing) function per
//     action carries the fixed action id (NZVIDEOMNI_MENU_FALLBACK_FN below).
//
//   * register_object_menu_param / register_layer_menu_param  -- one shared
//     callback plus a per-item void* param, so all items register in a loop.
//     IMPORTANT SDK DETAIL: the _param callback signature is void(void* param)
//     and, per plugin2.h, the host invokes it WITHOUT entering an edit section
//     (the Shift-JIS comment there states it calls back "without an edit
//     section", passing only the param). So, unlike the non-param path which is
//     handed a live EDIT_SECTION*, the _param callback does NOT receive one; the shared
//     callback enters one itself via EDIT_HANDLE::call_edit_section_param -- the
//     same cheap read-only pattern timeline.getSelection already uses.
//
// REALDEVICE-VERIFY: object/layer menu callbacks fire on the main (UI) thread
// with a live/obtainable EDIT_SECTION, and the selection getters used in
// SnapshotSelection are valid there.

// POD context handed through call_edit_section_param on the _param path (POD so
// it is safe even under an SEH wrapper; see C2712 note in EmitMenuInvoked).
struct MenuEmitContext {
    const char* action;
};

// Runs INSIDE an edit section that OnMenuInvokedParam enters on our behalf; the
// live EDIT_SECTION is finally available here so we can snapshot + emit.
void MenuEmitEditProc(void* param, EDIT_SECTION* edit) {
    const MenuEmitContext* ctx = static_cast<const MenuEmitContext*>(param);
    EmitMenuInvoked(ctx->action, edit);
}

// Shared _param menu callback for BOTH object and layer menus (behaviour is
// identical: snapshot + emit the action). `param` is the action id C-string we
// passed to register_*_menu_param (a string literal, static storage). Because
// the host does not give param callbacks an edit section, we enter one via
// call_edit_section_param. If no edit handle is available (or the host declines
// to run the section) we still deliver the action with an empty selection so the
// WebUI can react (it can re-query timeline.getSelection).
// REALDEVICE-VERIFY: register_*_menu_param is populated by beta52 AND a nested
// call_edit_section_param from within a menu-param callback is permitted / runs
// synchronously on the UI thread.
void OnMenuInvokedParam(void* param) {
    const char* action = static_cast<const char*>(param);
    MenuEmitContext ctx{action};
    if (g_edit_handle != nullptr && g_edit_handle->call_edit_section_param != nullptr) {
        if (g_edit_handle->call_edit_section_param(&ctx, &MenuEmitEditProc)) {
            return;
        }
        nzvideomni::LogWarn(L"menu-param: call_edit_section_param did not run; "
                       L"emitting action with an empty selection");
    }
    EmitMenuInvoked(action, nullptr);
}

// One plain non-capturing function per action for the non-param registration
// path (each simply emits its fixed action from the host-provided EDIT_SECTION).
#define NZVIDEOMNI_MENU_FALLBACK_FN(fn, action_id) \
    void fn(EDIT_SECTION* edit) { EmitMenuInvoked(action_id, edit); }

// Object right-click menu actions (right-click redesign, 15 items; see
// Docs/RIGHTCLICK_REDESIGN_SPEC.md 3-4 and the 7-1 old/new mapping). The
// insertProvisionalResult item (W2) is the down-arrow quick-insert: when the
// right-clicked object is a Nz-Videomni provisional whose job is already complete,
// the WebUI runs the same replace-insert the panel's film-strip button does.
// outpaintVideo / retakeRange (2026-08-09 W0) route to the Edit tab's
// Outpainting / Retake sub-panels; both require a video, and retakeRange
// additionally requires a selected frame range (the WebUI enforces both in
// menuSelection.ts - this file only registers and emits).
// videoAudioToLongA2v / audioToLongA2v (ledger 1-16, long a2v) are the Chained
// tab's counterparts of videoAudioToVideo / audioToVideo: the WebUI extracts
// the ribbon's audio (mix for the video item, solo for the audio item) and
// attaches it to the whole clip chain instead of a single generation.
// referenceVideoChain (ledger 1-15 W4, 2026-08-11) is the same idea applied to
// the IC-LoRA reference: referenceVideo sends the clip to the Single tab's
// reference slot, referenceVideoChain sends it to the Chained tab's, so one
// reference can condition a whole clip chain.
// endWithThis (end source, 2026-08-15) is the mirror of extendVideo: instead of
// generating what comes AFTER the selected material, it generates a video that
// ENDS with it. It is the first item that accepts EITHER a video or an image
// (the WebUI's menuSelection.ts holds that two-kind rule), which is why its
// display key carries both emoji instead of one category word.
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_ExtendVideo,       "extendVideo")
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_ReferenceVideo,    "referenceVideo")
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_ReferenceVideoChain, "referenceVideoChain")
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_VideoAudioToVideo, "videoAudioToVideo")
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_VideoAudioToLongA2v, "videoAudioToLongA2v")
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_AudioToLongA2v,    "audioToLongA2v")
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_ImageToVideo,      "imageToVideo")
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_AddImageKeyframe,  "addImageKeyframe")
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_ImageToClipChain,  "imageToClipChain")
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_EndWithThis,       "endWithThis")
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_AudioToVideo,      "audioToVideo")
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_AppendText,        "appendText")
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_InsertProvisionalResult, "insertProvisionalResult")
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_OutpaintVideo,     "outpaintVideo")
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_RetakeRange,       "retakeRange")
// Object tracking (section 3-54, SPEC #21): the first object item that targets
// something other than a generation - it makes a partial filter's box follow a
// subject. Enabled only when the selection IS a partial filter (the WebUI's
// menuSelection.ts holds that rule, as for every other item here).
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_TrackObject,       "trackObject")

// Layer right-click menu actions (5 items; see SPEC 3-6). The final item (W3,
// insertLatestResultHere) inserts the latest completed generation result at the
// right-click position as a plain insert (use_mouse_frame, like #9 below).
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_TextToVideoHere,             "textToVideoHere")
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_ImageFromCurrentFrame,      "imageFromCurrentFrame")
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_AddCurrentFrameKeyframe,    "addCurrentFrameAsKeyframe")
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_CurrentFrameToClipChain,    "currentFrameToClipChain")
NZVIDEOMNI_MENU_FALLBACK_FN(OnMenu_InsertLatestResultHere,     "insertLatestResultHere")

#undef NZVIDEOMNI_MENU_FALLBACK_FN

// A registerable timeline menu item: ASCII display key (run through Translate()
// into the Language/*.NzVideomni.aul2 Japanese label), the stable ASCII action id
// emitted to the WebUI, and the per-action plain callback for the non-param
// path. The display key keeps this translation unit ASCII-only (see file header);
// the embedded '\' makes the host render an "Nz-Videomni" submenu.
struct TimelineMenuItem {
    const wchar_t* name_key;
    const char*    action;
    void (*fallback)(EDIT_SECTION* edit);
};

// Emoji are embedded as universal-character-name escapes (file stays ASCII;
// see the header note). Each is followed by a single space, then the English
// display key. The '\' after "Nz-Videomni" makes the host render an "Nz-Videomni"
// submenu. The display keys are translated through Language/*.NzVideomni.aul2.
// Emoji codepoints (all pre-2020, safe on older Windows glyph sets): U+1F3AC
// clapper, U+1F5BC framed picture, U+1F3B5 musical note, U+1F4DD memo,
// U+2728 sparkles, U+1F4F7 camera, U+2B07 down arrow (W2/W3 quick-insert).
//
// ORDER (owner, 2026-08-09): the host renders the submenu in exactly this array
// order, so the array IS the menu order. The seven Video entries come first as
// one block - Outpainting and Retake were appended at the end when they were
// added, and have now been moved up into it - then the three Image entries,
// then the two audio entries, text, and the quick-insert. ONLY the order was
// changed: every name_key / action / fallback triple is byte-identical to
// before, so the WebUI's routing table (timeline/menuRouting.ts) and the
// translation keys in Language/*.NzVideomni.aul2 are untouched. (Keep this comment
// ASCII - see the file header; a non-ASCII byte anywhere here raises MSVC
// C4819.)
//
// Ledger 1-16 (long a2v): the two entries sit directly beside the
// single-shot a2v items they mirror - videoAudioToLongA2v right after
// videoAudioToVideo inside the Video block, audioToLongA2v right after
// audioToVideo - so the pair "single shot / long chain" reads together.
//
// Ledger 1-15 W4 (2026-08-11, 14 items): referenceVideoChain follows the same
// rule - it sits directly after referenceVideo, the single-shot IC-LoRA item it
// mirrors, so that pair reads together too.
//
// End source (2026-08-17, back to 15 items): the endWithThis entry is
// restored, sitting right after the Image block as before, now that the
// windowed-mode implementation resolves the crossfade issue that had it
// pulled from the user's reach.
const TimelineMenuItem kObjectMenuItems[] = {
    { L"Nz-Videomni\\\U0001F3AC Video: continue this video (v2v)",                             "extendVideo",       &OnMenu_ExtendVideo },
    { L"Nz-Videomni\\\U0001F3AC Video: generate using this video as reference (IC-LoRA)",      "referenceVideo",    &OnMenu_ReferenceVideo },
    { L"Nz-Videomni\\\U0001F3AC Video: long IC-LoRA reference for a clip chain",               "referenceVideoChain", &OnMenu_ReferenceVideoChain },
    { L"Nz-Videomni\\\U0001F3AC Video: audio-to-video from this video's sound",                "videoAudioToVideo", &OnMenu_VideoAudioToVideo },
    { L"Nz-Videomni\\\U0001F3AC Video: long a2v from this video's sound",                      "videoAudioToLongA2v", &OnMenu_VideoAudioToLongA2v },
    { L"Nz-Videomni\\\U0001F3AC Video: expand this video's canvas (Outpainting)",              "outpaintVideo",     &OnMenu_OutpaintVideo },
    { L"Nz-Videomni\\\U0001F3AC Video: redo the selected range (Retake)",                      "retakeRange",       &OnMenu_RetakeRange },
    { L"Nz-Videomni\\\U0001F5BC Image: generate a video from this image (i2v)",                "imageToVideo",      &OnMenu_ImageToVideo },
    { L"Nz-Videomni\\\U0001F5BC Image: add this image as a keyframe",                          "addImageKeyframe",  &OnMenu_AddImageKeyframe },
    { L"Nz-Videomni\\\U0001F5BC Image: generate a long video from this image (i2v Clip Chain)", "imageToClipChain", &OnMenu_ImageToClipChain },
    { L"Nz-Videomni\\\U0001F3AC\U0001F5BC: Generate a video ending with this",                 "endWithThis",       &OnMenu_EndWithThis },
    { L"Nz-Videomni\\\U0001F3B5 Audio: audio-to-video from this audio",                        "audioToVideo",      &OnMenu_AudioToVideo },
    { L"Nz-Videomni\\\U0001F3B5 Audio: long a2v from this audio",                              "audioToLongA2v",    &OnMenu_AudioToLongA2v },
    { L"Nz-Videomni\\\U0001F4DD Text: append to the main prompt",                              "appendText",        &OnMenu_AppendText },
    { L"Nz-Videomni\\\U00002B07 Insert this generated result now",                             "insertProvisionalResult", &OnMenu_InsertProvisionalResult },
    { L"Nz-Videomni\\\U0001F3AF Track: follow this partial filter's box",                      "trackObject",       &OnMenu_TrackObject },
};

const TimelineMenuItem kLayerMenuItems[] = {
    { L"Nz-Videomni\\\U00002728 Insert AI generation here (t2v)",                                    "textToVideoHere",           &OnMenu_TextToVideoHere },
    { L"Nz-Videomni\\\U0001F4F7 Generate a video from the current frame (i2v)",                      "imageFromCurrentFrame",     &OnMenu_ImageFromCurrentFrame },
    { L"Nz-Videomni\\\U0001F4F7 Add the current frame as a keyframe",                                "addCurrentFrameAsKeyframe", &OnMenu_AddCurrentFrameKeyframe },
    { L"Nz-Videomni\\\U0001F4F7 Generate a long video from the current frame (i2v Clip Chain)",      "currentFrameToClipChain",   &OnMenu_CurrentFrameToClipChain },
    { L"Nz-Videomni\\\U00002B07 Insert the latest generation result here",                           "insertLatestResultHere",    &OnMenu_InsertLatestResultHere },
};

// Register one menu table onto the host, preferring the _param path (single
// shared callback, loop registration) and falling back to per-action plain
// functions when the host does not expose the _param entry. Both paths are
// null-checked and logged; if neither entry is populated the feature still works
// through the WebView2 panel calling timeline.getSelection (the I4 fallback).
// `is_object` selects the object vs layer host entry points (their signatures
// are identical, so this stays a single helper).
void RegisterTimelineMenu(HOST_APP_TABLE* host, const TimelineMenuItem* items,
                          size_t count, bool is_object) {
    void (*const reg_param)(LPCWSTR, void*, void (*)(void*)) =
        is_object ? host->register_object_menu_param : host->register_layer_menu_param;
    void (*const reg_plain)(LPCWSTR, void (*)(EDIT_SECTION*)) =
        is_object ? host->register_object_menu : host->register_layer_menu;
    const wchar_t* const kind = is_object ? L"object" : L"layer";

    if (reg_param != nullptr) {
        for (size_t i = 0; i < count; ++i) {
            // Stored in g_menu_label_storage (not a local std::wstring): the host
            // keeps the raw LPCWSTR without copying it, so the string must outlive
            // this function call (see g_menu_label_storage's declaration comment).
            g_menu_label_storage.push_back(Translate(items[i].name_key));
            // The action id is a string literal (static storage); the host only
            // reads it back as the param, so dropping const here is safe.
            reg_param(g_menu_label_storage.back().c_str(),
                      const_cast<char*>(items[i].action), &OnMenuInvokedParam);
        }
        nzvideomni::LogInfo(std::wstring(L"register_") + kind + L"_menu_param: registered " +
                       std::to_wstring(count) + L" item(s)");
    } else if (reg_plain != nullptr) {
        for (size_t i = 0; i < count; ++i) {
            g_menu_label_storage.push_back(Translate(items[i].name_key));
            reg_plain(g_menu_label_storage.back().c_str(), items[i].fallback);
        }
        nzvideomni::LogInfo(std::wstring(L"register_") + kind +
                       L"_menu (non-param fallback): registered " +
                       std::to_wstring(count) + L" item(s)");
    } else {
        nzvideomni::LogWarn(std::wstring(L"register_") + kind + L"_menu[_param] unavailable "
                       L"(beta52) - using WebView2 panel + timeline.getSelection fallback");
    }
}

// Project-load handler: the host calls this after loading a project (and,
// per the SDK, also at project initialization). We only push a lightweight
// trigger with no payload; the WebUI reacts by calling timeline.scanProvisionals
// to find orphaned provisional placeholder objects and offer to re-generate
// them. Never throws into the host callback.
// REALDEVICE-VERIFY: register_project_load_handler firing timing (project load
// AND initialization), and that no PROJECT_FILE data is needed here.
void OnProjectLoad(PROJECT_FILE* project) {
    (void)project;
    try {
        // NOTE: "timeline.projectLoaded" is a native->WebUI trigger event with an
        // empty payload. Adding the matching WebUI subscriber and a
        // TIMELINE_PROJECT_LOADED constant in bridge/types.ts is a follow-up,
        // out of this change's scope (which is "plugin.cpp emits the trigger
        // only"). It is harmless while unsubscribed. TODO(webui): subscribe and
        // call timeline.scanProvisionals on this event.
        PostJsonToPage("{\"event\":\"timeline.projectLoaded\",\"data\":{}}");
        nzvideomni::LogInfo(L"timeline.projectLoaded pushed (project load handler)");
    } catch (...) {
        nzvideomni::LogError(L"OnProjectLoad failed (exception swallowed)");
    }
}

} // namespace

//---------------------------------------------------------------------
//  Exported entry points.
//---------------------------------------------------------------------
EXTERN_C __declspec(dllexport) DWORD RequiredVersion() {
    // Same minimum host version as the SDK WindowClient sample.
    return 2003300;
}

EXTERN_C __declspec(dllexport) void InitializeLogger(LOG_HANDLE* handle) {
    g_logger = handle;
    nzvideomni::SetLogHandle(handle);
}

EXTERN_C __declspec(dllexport) void InitializeConfig(CONFIG_HANDLE* handle) {
    g_config = handle;
}

EXTERN_C __declspec(dllexport) bool InitializePlugin(DWORD version) {
    (void)version;
    nzvideomni::LogInfo(L"Nz-Videomni InitializePlugin");
    return true;
}

EXTERN_C __declspec(dllexport) void UninitializePlugin() {
    nzvideomni::LogInfo(L"Nz-Videomni UninitializePlugin");
    g_bridge.Shutdown();
    g_webview_host.Shutdown();
}

EXTERN_C __declspec(dllexport) COMMON_PLUGIN_TABLE* GetCommonPluginTable(void) {
    return &g_common_plugin_table;
}

EXTERN_C __declspec(dllexport) void RegisterPlugin(HOST_APP_TABLE* host) {
    const HINSTANCE hinst = ::GetModuleHandleW(nullptr);

    // Register the window class for our own top-level window.
    WNDCLASSEXW wcex = {};
    wcex.cbSize        = sizeof(WNDCLASSEXW);
    wcex.lpszClassName = NZVIDEOMNI_WINDOW_NAME;
    wcex.lpfnWndProc   = WndProc;
    wcex.hInstance     = hinst;
    wcex.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    wcex.hCursor       = ::LoadCursorW(nullptr, IDC_ARROW);
    if (!::RegisterClassExW(&wcex)) {
        nzvideomni::LogError(L"RegisterClassExW failed");
        return;
    }

    // Create the window as WS_POPUP first: the host converts it to a docked
    // WS_CHILD when register_window_client() is called.
    HWND hwnd = ::CreateWindowExW(
        0,
        NZVIDEOMNI_WINDOW_NAME,
        NZVIDEOMNI_WINDOW_NAME,
        WS_POPUP,
        CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
        nullptr, nullptr, hinst, nullptr);
    if (hwnd == nullptr) {
        nzvideomni::LogError(L"CreateWindowExW failed");
        return;
    }

    // Fallback status label filling the client area. Hidden behind the WebView2
    // control on success; brought forward (with an error message) on failure.
    RECT client = {};
    ::GetClientRect(hwnd, &client);
    g_status_label = ::CreateWindowExW(
        0,
        WC_STATICW,
        Translate(L"Loading Nz-Videomni UI...").c_str(),
        WS_VISIBLE | WS_CHILD | SS_LEFT,
        0, 0, client.right, client.bottom,
        hwnd,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(IDC_STATUS_LABEL)),
        hinst,
        nullptr);

    // Register the window with the host (docks it into the main UI). The
    // display name is translated (Language/*.NzVideomni.aul2); the window class
    // name / creation title above stay the fixed NZVIDEOMNI_WINDOW_NAME identifier.
    const std::wstring display_name = Translate(NZVIDEOMNI_WINDOW_NAME);
    host->register_window_client(display_name.c_str(), hwnd);

    // Acquire the edit handle used to read project data and drive timeline
    // edits, and hand it to the bridge.
    g_edit_handle = host->create_edit_handle();
    g_bridge.SetEditHandle(g_edit_handle);

    // Retain the plugin window for the non-capturing timeline / project
    // callbacks (they PostMessage native-initiated events onto this window).
    g_plugin_hwnd = hwnd;

    // Give the bridge the plugin window so ui.pickFile can parent its modal
    // Open dialog to it.
    g_bridge.SetPluginHwnd(hwnd);

    // Start the bridge's HTTP worker pool and give it a thread-safe way to
    // deliver asynchronous RPC responses: a worker thread posts the response
    // (a heap std::string) to this window, and WndProc forwards it to the page
    // on the UI thread.
    const HWND plugin_hwnd = hwnd;
    g_bridge.Initialize([plugin_hwnd](const std::string& response) {
        auto* payload = new std::string(response);
        if (!::PostMessageW(plugin_hwnd, WM_NZVIDEOMNI_POST_RESPONSE, 0,
                            reinterpret_cast<LPARAM>(payload))) {
            delete payload;
        }
    });

    // Log the NavigationCompleted outcome (success/failure). Visibility of the
    // fallback label vs. the WebView2 control is handled entirely inside
    // WebViewHost; this callback only logs, per the plan.
    g_webview_host.SetReadyHandler([](bool success) {
        if (success) {
            nzvideomni::LogInfo(L"Nz-Videomni WebView2 panel ready (navigation succeeded)");
        } else {
            nzvideomni::LogError(L"Nz-Videomni WebView2 panel failed to become ready "
                            L"(navigation failed)");
        }
    });

    // Start hosting the Web UI. On failure this reports through g_status_label
    // (localized via the Translate() lambda below, contract v4 Language files).
    g_webview_host.Initialize(
        hwnd, g_status_label,
        [](const std::string& request) -> std::string {
            return g_bridge.HandleMessage(request);
        },
        [](const std::wstring& text) -> std::wstring {
            return Translate(text.c_str());
        });

    // ---- Contract v5: timeline context menus + project-load trigger --------
    // Registered as ADDITIONS to the window-client wiring above; the existing
    // captureFrame / insertMedia / backend paths are untouched. RegisterTimelineMenu
    // null-checks each host entry (preferring the _param path, falling back to the
    // per-action non-param path): if beta52 populates neither, it logs and skips,
    // and the feature still works through the WebView2 panel calling
    // timeline.getSelection (the I4 "generate from selection" fallback). Menu
    // names are ASCII keys run through Translate() (Language/*.aul2); the embedded
    // '\' makes the host render an "Nz-Videomni" submenu.
    // REALDEVICE-VERIFY: register_object_menu[_param] / register_layer_menu[_param] /
    // register_project_load_handler are populated by AviUtl2 beta52 and fire as
    // documented (right-click menus on object / empty layer; project load+init),
    // and every alpha item below actually appears/fires under the "Nz-Videomni" submenu.
    RegisterTimelineMenu(host, kObjectMenuItems,
                         sizeof(kObjectMenuItems) / sizeof(kObjectMenuItems[0]),
                         /*is_object=*/true);
    RegisterTimelineMenu(host, kLayerMenuItems,
                         sizeof(kLayerMenuItems) / sizeof(kLayerMenuItems[0]),
                         /*is_object=*/false);
    if (host->register_project_load_handler != nullptr) {
        host->register_project_load_handler(&OnProjectLoad);
        nzvideomni::LogInfo(L"register_project_load_handler: registered");
    } else {
        nzvideomni::LogWarn(L"register_project_load_handler unavailable (beta52) - orphan "
                       L"provisional scan must be user-triggered");
    }

    nzvideomni::LogInfo(L"Nz-Videomni plugin registered (WebView2 host + edit handle)");
}
