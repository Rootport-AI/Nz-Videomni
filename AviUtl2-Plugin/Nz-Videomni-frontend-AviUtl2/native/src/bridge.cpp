// bridge.cpp - implementation of the WebView2 <-> RPC-core bridge (contract v2).
// ASCII-only source.
#include "bridge.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>  // plugin2.h relies on the Win32 typedefs (LPCWSTR, HWND)

#include <commdlg.h>   // GetOpenFileNameW / OPENFILENAMEW (ui.pickFile)
#include <shobjidl.h>  // IFileOpenDialog / FOS_PICKFOLDERS (ui.pickFolder, contract v6)

#include <algorithm>  // std::sort (the trackObject score-distribution log)
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <vector>

#include "json.hpp"
#include "plugin2.h"  // EDIT_HANDLE, EDIT_INFO, EDIT_SECTION, PIXEL_RGBA

#include "alias_util.h"  // BuildMediaObjectAlias, NormalizeAliasObjectFrameHeader (3-140)
#include "bridge_core.h"
#include "fs_util.h"    // MatchesAnyExtension, MakeNumberedName, ExtensionLower, JoinPath (v6)
#include "http_client.h"
#include "log.h"
#include "media_fps_probe.h"  // ProbeMediaFps (contract v11: getSelection mediaFps)
#include "mf_mp4_writer.h"  // Mp4Writer for timeline.cutoutRange (I7/I9)
#include "settings.h"
#include "strconv.h"
#include "timeline_math.h"  // ProjectFramesForSeconds (Insert: media real length)
#include "track_postprocess.h"  // PostProcessTrack (timeline.trackObject, 3-54)
#include "wav_probe.h"  // ProbeWavFile (contract v6: fs.listFiles / fs.probeAudioDuration)
#include "wic_png.h"

namespace nzvideomni {

namespace {

using json = json_t;

// Context passed through call_edit_section_param into the (non-capturing) edit
// callback. The callback fills in the outputs.
struct InsertContext {
    const InsertMediaParams* in = nullptr;
    OBJECT_HANDLE object = nullptr;
    int layer = 0;
    int frame = 0;
    bool ran = false;
};

// Create one media object at layer/frame - the single funnel every live media
// insert in this file goes through (section 3-140).
//
// Why not just call create_object_from_media_file: that API never writes the
// "audio present" item, so the object it makes renders as a ONE-tone ribbon and
// its context menu offers no "separate audio" - the mp4 plays fine, but the
// object is not what dropping the same file on the timeline produces. So the
// drag-and-drop-equivalent alias is built here instead (BuildMediaObjectAlias,
// fed by one get_media_info probe for the audio flag and the source duration)
// and handed to create_object_from_alias.
//
// The length is resolved with the SAME formula as before 3-140: length =
// round(total_time_seconds * project_fps), project_fps = rate/scale, and any
// failure (no get_media_info, unsupported/still media with total_time <= 0,
// degenerate rate/scale, or a sub-frame clip that rounds to 0) leaves length 0,
// i.e. the host's automatic length adjustment.
//
// The legacy API remains the safety net, three times over:
//   1. the alias is only used for a real video with a usable length - a still,
//      an audio-only file, an unreadable one or a path with CR/LF takes the
//      legacy call with exactly the arguments it received before 3-140;
//   2. if create_object_from_alias returns null, the legacy call is tried once
//      with the same layer/frame/length, so an alias-specific failure (an
//      effect name that does not resolve, say) costs nothing;
//   3. the created object's real span is measured afterwards, and one SHORTER
//      than requested is deleted and rebuilt with the legacy call.
// Requested vs actual is logged either way: alias + explicit-length collision
// behaviour is not yet observed on the real device (gate R7).
//
// Returns the created object, or nullptr when every attempt failed (callers
// treat that exactly as they treated a null create_object_from_media_file).
OBJECT_HANDLE CreateMediaObject(EDIT_SECTION* edit, const std::string& path_utf8,
                                int layer, int frame) {
    if (edit == nullptr || edit->create_object_from_media_file == nullptr) {
        return nullptr;
    }
    const std::wstring wide = Utf8ToWide(path_utf8);

    // One probe feeds both the length and the alias (audio flag + source span).
    MEDIA_INFO info{};
    bool have_info = false;
    if (edit->get_media_info != nullptr) {
        have_info = edit->get_media_info(wide.c_str(), &info, sizeof(info));
    }
    int length = 0;
    if (have_info && info.total_time > 0.0 && edit->info != nullptr &&
        edit->info->scale != 0) {
        const double project_fps = static_cast<double>(edit->info->rate) /
                                   static_cast<double>(edit->info->scale);
        const int frames = ProjectFramesForSeconds(info.total_time, project_fps);
        if (frames >= 1) {
            length = frames;
        }
    }

    // Safety net 1. video_track_num == 0 rules out stills and audio-only files
    // (whose ribbons are already right today); length >= 1 is required because
    // an explicit length of 0 is exactly where the SDK silently shortens a
    // colliding create instead of failing (D2 finding).
    std::string alias;
    if (have_info && info.video_track_num > 0 && length >= 1) {
        alias = BuildMediaObjectAlias(path_utf8, info.total_time,
                                      info.audio_track_num > 0);
    }
    if (alias.empty() || edit->create_object_from_alias == nullptr) {
        return edit->create_object_from_media_file(wide.c_str(), layer, frame, length);
    }

    // An alias' own frame range OVERRIDES the length argument, so pin both to
    // the same span - the double pin (the frame= header inside the alias PLUS
    // the explicit length argument) is the shape insertProvisional already
    // proved on the real device. The two are written in DIFFERENT units: the
    // header's second value is an INCLUSIVE end frame (so the helper writes
    // length-1), while the length argument is a frame COUNT. Conflating them is
    // what made every inserted object one frame too long until 2026-09-04.
    const std::string pinned = NormalizeAliasObjectFrameHeader(alias, length);
    OBJECT_HANDLE created =
        edit->create_object_from_alias(pinned.c_str(), layer, frame, length);
    if (created == nullptr) {
        // Safety net 2: alias-specific failure only - this is byte-for-byte the
        // pre-3-140 call, so the outcome here is exactly today's.
        LogWarn(std::wstring(L"CreateMediaObject: create_object_from_alias returned null "
                             L"(layer ") +
                std::to_wstring(layer) + L", frame " + std::to_wstring(frame) +
                L", length " + std::to_wstring(length) +
                L") - retrying with create_object_from_media_file");
        return edit->create_object_from_media_file(wide.c_str(), layer, frame, length);
    }

    // Safety net 3. OBJECT_LAYER_FRAME.end is INCLUSIVE - measured 2026-09-04,
    // see Docs\SDK_REFERENCE.md section 16 (h) - so "end - start + 1" below is
    // the real frame count and, with the header now pinned to length-1, the
    // expected reading is actual == length.
    //
    // The check stays deliberately one-sided (only a span SHORTER than requested
    // is a failure) rather than an equality test. The reason is a dependency, not
    // a proof: actual == length holds only while the alias' frame header keeps
    // overriding the length argument (itself an observed behaviour, same day).
    // Should that ever stop holding, an equality test would make every insert
    // fall back silently and undo the whole 3-140 fix, whereas a one-sided test
    // simply passes.
    if (edit->get_object_layer_frame != nullptr) {
        const OBJECT_LAYER_FRAME lf = edit->get_object_layer_frame(created);
        const int actual = lf.end - lf.start + 1;
        LogInfo(std::wstring(L"CreateMediaObject: alias create requested layer ") +
                std::to_wstring(layer) + L", frame " + std::to_wstring(frame) +
                L", length " + std::to_wstring(length) + L"; actual layer " +
                std::to_wstring(lf.layer) + L", start " + std::to_wstring(lf.start) +
                L", end " + std::to_wstring(lf.end));
        if (actual < length) {
            LogWarn(std::wstring(L"CreateMediaObject: alias object is shorter than "
                                 L"requested (") +
                    std::to_wstring(actual) + L" < " + std::to_wstring(length) + L")");
            if (edit->delete_object != nullptr) {
                edit->delete_object(created);
                return edit->create_object_from_media_file(wide.c_str(), layer, frame,
                                                           length);
            }
        }
    }
    return created;
}

// Non-capturing callback invoked by the host on the main thread under the edit
// lock. Must be a plain function pointer, so it captures nothing.
void InsertMediaEditProc(void* param, EDIT_SECTION* edit) {
    auto* ctx = static_cast<InsertContext*>(param);
    ctx->ran = true;
    int layer = ctx->in->has_layer ? ctx->in->layer
                                    : (edit->info != nullptr ? edit->info->layer : 0);
    int frame = ctx->in->has_frame ? ctx->in->frame
                                    : (edit->info != nullptr ? edit->info->frame : 0);
    ctx->layer = layer;
    ctx->frame = frame;
    // Sizing to the media's real length (so the ribbon reflects the actual video
    // duration instead of the host's default add-position guess) and the
    // "audio present" flag both live in CreateMediaObject.
    ctx->object = CreateMediaObject(edit, ctx->in->file_path, layer, frame);
}

// True if a file exists on disk (a plain file or directory; used to give the
// caller a precise FILE_NOT_FOUND before touching the edit lock).
bool FileExists(const std::string& utf8_path) {
    const std::wstring wide = Utf8ToWide(utf8_path);
    const DWORD attrs = ::GetFileAttributesW(wide.c_str());
    return attrs != INVALID_FILE_ATTRIBUTES;
}

// Map an HttpClient transport failure onto an RPC error code.
const char* TransportCode(TransportError e) {
    switch (e) {
        case TransportError::kTimeout:
            return "BACKEND_TIMEOUT";
        case TransportError::kUnreachable:
        default:
            return "BACKEND_UNREACHABLE";
    }
}

// ---------------------------------------------------------------------------
// Contract v6 (batch A2V + fs bridge) shared Win32 helpers.
// ---------------------------------------------------------------------------

// Minimal RAII for a COM interface pointer (same shape as wic_png.cpp's; kept
// as a separate copy per translation unit so bridge_core/tests never need to
// pull in COM headers - see that file's comment for the rationale).
template <typename T>
class ComPtr {
public:
    ComPtr() = default;
    ~ComPtr() { reset(); }
    ComPtr(const ComPtr&) = delete;
    ComPtr& operator=(const ComPtr&) = delete;

    T** put() { reset(); return &p_; }
    T* get() const { return p_; }
    T* operator->() const { return p_; }
    explicit operator bool() const { return p_ != nullptr; }

    void reset() {
        if (p_ != nullptr) {
            p_->Release();
            p_ = nullptr;
        }
    }

private:
    T* p_ = nullptr;
};

// Recursively create every path component of `dir` (the directory itself, not
// a file within it). Best effort - mirrors the segment-by-segment loop
// http_client.cpp's DownloadSync already uses for its own default destination.
void EnsureDirectoryTree(const std::wstring& dir) {
    for (size_t i = 0; i < dir.size(); ++i) {
        if ((dir[i] == L'\\' || dir[i] == L'/') && i > 0) {
            ::CreateDirectoryW(dir.substr(0, i).c_str(), nullptr);
        }
    }
    if (!dir.empty()) {
        ::CreateDirectoryW(dir.c_str(), nullptr);
    }
}

// Recursively create the parent directory of `file_path`. Best effort.
void EnsureParentDirectoryOf(const std::wstring& file_path) {
    const size_t slash = file_path.find_last_of(L"\\/");
    if (slash == std::wstring::npos) {
        return;
    }
    EnsureDirectoryTree(file_path.substr(0, slash));
}

// FILETIME (100-ns intervals since 1601-01-01) -> milliseconds since the Unix
// epoch (1970-01-01), as a double (fs.listFiles' 'mtimeMs').
double FileTimeToUnixMs(const FILETIME& ft) {
    ULARGE_INTEGER uli;
    uli.LowPart = ft.dwLowDateTime;
    uli.HighPart = ft.dwHighDateTime;
    constexpr unsigned long long kTicksPerMs = 10000ULL;         // 100ns -> ms
    constexpr unsigned long long kEpochDiffMs = 11644473600000ULL;  // 1601->1970
    const unsigned long long total_ms = uli.QuadPart / kTicksPerMs;
    return static_cast<double>(total_ms) - static_cast<double>(kEpochDiffMs);
}

// Contract v6 noClobber (backend.downloadVideo): claim a free file name under
// `dir` via an exclusive CREATE_NEW loop - never opens an existing file for
// writing, so two concurrent claims can never collide on the same name. Tries
// MakeNumberedName(desired_name, 1) (unchanged), then (,2), (,3), ... The
// empty placeholder file left behind is immediately overwritten by
// HttpClient::DownloadSync's own CREATE_ALWAYS open on the very same path
// (safe: we still own the just-created empty file; no other file is ever
// touched). `dir` must already exist (see EnsureDirectoryTree).
constexpr int kMaxNoClobberAttempts = 1000;

bool ClaimNoClobberPath(const std::wstring& dir, const std::string& desired_name_utf8,
                        std::wstring* out_path, std::string* err_message) {
    for (int n = 1; n <= kMaxNoClobberAttempts; ++n) {
        const std::string candidate_name = MakeNumberedName(desired_name_utf8, n);
        const std::wstring candidate_path = dir + L"\\" + Utf8ToWide(candidate_name);
        HANDLE h = ::CreateFileW(candidate_path.c_str(), GENERIC_WRITE, 0, nullptr,
                                 CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h != INVALID_HANDLE_VALUE) {
            ::CloseHandle(h);
            *out_path = candidate_path;
            return true;
        }
        const DWORD e = ::GetLastError();
        if (e != ERROR_FILE_EXISTS) {
            *err_message = "Could not claim destination file name (winhttp error " +
                           std::to_string(e) + ")";
            return false;
        }
    }
    *err_message = "No available file name found after " +
                   std::to_string(kMaxNoClobberAttempts) + " attempts";
    return false;
}

// Atomically publish `tmp` (already fully written) as `target`. Uses
// ReplaceFileW when target already exists (keeps target's ACL/attributes) and
// MoveFileExW(MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) for a
// first-time write. A sharing violation / access denial is retried with
// exponential backoff (50ms, 100ms, 200ms, 400ms - 5 attempts total); after
// the last failed attempt the leftover tmp file is deleted and this reports
// failure. Retained as a general-purpose atomic-publish helper: its former
// caller (fs.writeTextFileAtomic) was removed on 2026-07-18, so it (and
// EnsureParentDirectoryOf above) is currently unreferenced by a live method.
bool AtomicReplaceOrMove(const std::wstring& target, const std::wstring& tmp,
                         std::string* err_message) {
    const bool target_exists =
        ::GetFileAttributesW(target.c_str()) != INVALID_FILE_ATTRIBUTES;
    DWORD backoff_ms = 50;
    for (int attempt = 0; attempt < 5; ++attempt) {
        const BOOL ok =
            target_exists
                ? ::ReplaceFileW(target.c_str(), tmp.c_str(), nullptr,
                                 REPLACEFILE_WRITE_THROUGH, nullptr, nullptr)
                : ::MoveFileExW(tmp.c_str(), target.c_str(),
                                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
        if (ok) {
            return true;
        }
        const DWORD e = ::GetLastError();
        const bool lockish = (e == ERROR_SHARING_VIOLATION || e == ERROR_ACCESS_DENIED ||
                              e == ERROR_LOCK_VIOLATION || e == ERROR_USER_MAPPED_FILE);
        if (!lockish) {
            break;  // not a lock: retrying will not help.
        }
        if (attempt < 4) {
            ::Sleep(backoff_ms);
            backoff_ms *= 2;
        }
    }
    ::DeleteFileW(tmp.c_str());  // best-effort cleanup of the leftover temp file
    *err_message = "target file is locked by another process";
    return false;
}

// ---------------------------------------------------------------------------
// timeline.captureFrame - asynchronous render -> copy -> WIC PNG encode flow.
//
// Threading (SDK_REFERENCE.md section 12): rendering_scene_video() only queues a
// task and returns; its callback fires on the host's rendering thread with a
// PIXEL_RGBA buffer valid only for the duration of the callback. We therefore
// copy the pixels (pitch-aware) inside the callback and hand them to a worker
// via a condition variable. The worker runs on the HTTP pool (never the UI
// thread and never under an edit/reference lock), so blocking there is safe.
// ---------------------------------------------------------------------------

// Shared state between the render callback and the waiting worker. Allocated on
// the heap so its lifetime can outlive a timed-out worker: on timeout the worker
// marks it abandoned and the (late) callback frees it, avoiding a dangling copy.
struct CaptureState {
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    bool abandoned = false;
    int width = 0;
    int height = 0;
    std::vector<unsigned char> rgba;  // tightly packed, stride == width*4
};

// Rendering-thread callback. Must be a plain function pointer (captures nothing).
// PIXEL_RGBA is { unsigned char r, g, b, a; } (filter2.h) == DXGI R8G8B8A8_UNORM,
// so the bytes are copied straight through with no channel swap. `pitch` is the
// source row stride in bytes and may exceed width*4, hence the per-row copy.
void CaptureRenderCb(void* param, int frame, const void* buffer, int width,
                     int height, int pitch) {
    (void)frame;
    auto* st = static_cast<CaptureState*>(param);
    bool abandoned = false;
    {
        std::lock_guard<std::mutex> lock(st->m);
        st->width = width;
        st->height = height;
        if (buffer != nullptr && width > 0 && height > 0 && pitch > 0) {
            const size_t row_bytes = static_cast<size_t>(width) * 4u;
            st->rgba.resize(row_bytes * static_cast<size_t>(height));
            const unsigned char* src = static_cast<const unsigned char*>(buffer);
            for (int y = 0; y < height; ++y) {
                std::memcpy(st->rgba.data() + static_cast<size_t>(y) * row_bytes,
                            src + static_cast<size_t>(y) * static_cast<size_t>(pitch),
                            row_bytes);
            }
        }
        st->done = true;
        abandoned = st->abandoned;
    }
    st->cv.notify_one();
    if (abandoned) {
        delete st;  // worker gave up; we own the state now.
    }
}

// A "yyyymmdd_HHMMSS_fff_nnn" stamp, unique across concurrent captures.
std::string UniqueStamp() {
    static std::atomic<unsigned> counter{0};
    SYSTEMTIME st = {};
    ::GetLocalTime(&st);
    const unsigned seq = counter.fetch_add(1) & 0xFFFu;
    char buffer[64];
    ::_snprintf_s(buffer, sizeof(buffer), _TRUNCATE,
                  "%04d%02d%02d_%02d%02d%02d_%03d_%03u", st.wYear, st.wMonth, st.wDay,
                  st.wHour, st.wMinute, st.wSecond, st.wMilliseconds, seq);
    return std::string(buffer);
}

// Runs on an HTTP worker thread: drive the render, copy, PNG-encode and post the
// captureFrame response. `handle->rendering_scene_video` is guaranteed non-null
// by the caller.
void CaptureFrameWorker(EDIT_HANDLE* handle, int frame, json_t id,
                        Bridge::ResponsePoster poster) {
    auto post_error = [&poster, &id](const char* code, const std::string& msg) {
        poster(MakeErrorResponse(id, code, msg));
    };

    auto* st = new CaptureState();
    const bool queued = handle->rendering_scene_video(frame, st, &CaptureRenderCb);
    if (!queued) {
        delete st;
        LogWarn(L"timeline.captureFrame: rendering_scene_video returned false");
        post_error("CAPTURE_FAILED", "rendering_scene_video returned false");
        return;
    }

    int width = 0;
    int height = 0;
    std::vector<unsigned char> rgba;
    {
        std::unique_lock<std::mutex> lock(st->m);
        const bool ok = st->cv.wait_for(lock, std::chrono::seconds(10),
                                        [st] { return st->done; });
        if (!ok) {
            // Hand ownership to the (late) callback and walk away.
            st->abandoned = true;
            lock.unlock();
            LogWarn(L"timeline.captureFrame: render callback timed out after 10s");
            post_error("CAPTURE_FAILED",
                       "Render callback did not arrive within 10 seconds");
            return;
        }
        width = st->width;
        height = st->height;
        rgba = std::move(st->rgba);
    }
    delete st;  // callback has completed and cannot fire again.

    if (width <= 0 || height <= 0 || rgba.size() < static_cast<size_t>(width) *
                                                        static_cast<size_t>(height) * 4u) {
        LogWarn(L"timeline.captureFrame: render produced an empty/short buffer");
        post_error("CAPTURE_FAILED", "Render produced an empty buffer");
        return;
    }

    const std::wstring dir = AppDataDir() + L"\\captures";
    const std::string file_name = CaptureFileName(frame, UniqueStamp());
    const std::wstring dest = dir + L"\\" + Utf8ToWide(file_name);

    std::string enc_err;
    if (!EncodeRgbaToPngFile(dest, rgba.data(), width, height, &enc_err)) {
        LogWarn(std::wstring(L"timeline.captureFrame: PNG encode failed: ") +
                Utf8ToWide(enc_err));
        post_error("CAPTURE_FAILED", "PNG encode failed: " + enc_err);
        return;
    }

    json result;
    result["filePath"] = WideToUtf8(dest);
    result["width"] = width;
    result["height"] = height;
    result["frame"] = frame;
    LogInfo(std::wstring(L"timeline.captureFrame: saved ") + std::to_wstring(width) +
            L"x" + std::to_wstring(height) + L" frame " + std::to_wstring(frame) +
            L" -> " + dest + L" (first RGBA " + std::to_wstring(rgba[0]) + L"," +
            std::to_wstring(rgba[1]) + L"," + std::to_wstring(rgba[2]) + L"," +
            std::to_wstring(rgba[3]) + L")");
    poster(MakeSuccessResponse(id, std::move(result)));
}

// ---------------------------------------------------------------------------
// ui.makeThumbnail - runs on an HTTP worker thread: WIC decode -> scale -> JPEG
// -> base64 -> data URL. Never touches the UI thread.
// ---------------------------------------------------------------------------
void MakeThumbnailWorker(MakeThumbnailRequest req, json_t id,
                         Bridge::ResponsePoster poster) {
    std::vector<unsigned char> bytes;
    int width = 0;
    int height = 0;
    int src_w = 0;
    int src_h = 0;
    std::string enc_err;
    const bool ok = MakeJpegThumbnail(Utf8ToWide(req.file_path), req.max_dim, &bytes,
                                      &width, &height, &src_w, &src_h, &enc_err);
    if (!ok) {
        LogWarn(std::wstring(L"ui.makeThumbnail: failed: ") + Utf8ToWide(enc_err));
        poster(MakeErrorResponse(id, "THUMBNAIL_FAILED",
                                 "Thumbnail generation failed: " + enc_err));
        return;
    }

    const std::string base64 = Base64Encode(bytes.data(), bytes.size());
    json result;
    result["dataUrl"] = MakeDataUrl(kThumbnailMimeType, base64);
    result["width"] = width;
    result["height"] = height;
    result["sourceWidth"] = src_w;
    result["sourceHeight"] = src_h;
    LogInfo(std::wstring(L"ui.makeThumbnail: ") + std::to_wstring(src_w) + L"x" +
            std::to_wstring(src_h) + L" -> " + std::to_wstring(width) + L"x" +
            std::to_wstring(height) + L" JPEG (" + std::to_wstring(bytes.size()) +
            L" bytes)");
    poster(MakeSuccessResponse(id, std::move(result)));
}

// ---------------------------------------------------------------------------
// fs.listFiles (contract v6) - runs on an HTTP worker thread: non-recursive
// FindFirstFileW/FindNextFileW enumeration, extension filter, optional .wav
// duration probe. Never touches the UI thread.
// ---------------------------------------------------------------------------
void ListFilesWorker(ListFilesRequest req, json_t id, Bridge::ResponsePoster poster) {
    std::vector<ListedFile> files;
    std::wstring folder_w = Utf8ToWide(req.folder_path);
    std::wstring pattern = folder_w;
    if (!pattern.empty() && pattern.back() != L'\\' && pattern.back() != L'/') {
        pattern += L'\\';
    }
    pattern += L'*';

    WIN32_FIND_DATAW fd = {};
    HANDLE find = ::FindFirstFileW(pattern.c_str(), &fd);
    if (find != INVALID_HANDLE_VALUE) {
        do {
            // Skip directories (including "." / "..") and hidden/system entries;
            // fs.listFiles is a non-recursive FILE listing.
            if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0 ||
                (fd.dwFileAttributes & (FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM)) != 0) {
                continue;
            }
            const std::string name = WideToUtf8(fd.cFileName);
            if (!MatchesAnyExtension(name, req.extensions)) {
                continue;
            }
            ListedFile lf;
            lf.name = name;
            lf.path = JoinPath(req.folder_path, name);
            lf.size_bytes = (static_cast<long long>(fd.nFileSizeHigh) << 32) |
                            static_cast<long long>(fd.nFileSizeLow);
            lf.mtime_ms = FileTimeToUnixMs(fd.ftLastWriteTime);
            lf.duration_sec = 0.0;
            if (req.with_audio_duration && ExtensionLower(name) == ".wav") {
                const std::optional<double> d = ProbeWavFile(Utf8ToWide(lf.path));
                if (d.has_value()) {
                    lf.duration_sec = *d;
                }
            }
            files.push_back(std::move(lf));
        } while (::FindNextFileW(find, &fd));
        ::FindClose(find);
    }
    // A missing/inaccessible folder resolves to an empty list, never an error
    // (contract v6, Docs/BRIDGE_CONTRACT.md Sec.4.16 - unregulated, mirrors the
    // webui mock's behavior).
    LogInfo(std::wstring(L"fs.listFiles: ") + std::to_wstring(files.size()) +
            L" entries -> " + folder_w);
    poster(MakeSuccessResponse(id, MakeListFilesResult(files)));
}

// ---------------------------------------------------------------------------
// fs.probeAudioDuration (contract v6) - runs on an HTTP worker thread: never
// errors, resolves {0, false} for a non-.wav or unreadable file.
// ---------------------------------------------------------------------------
void ProbeAudioDurationWorker(ProbeAudioDurationRequest req, json_t id,
                              Bridge::ResponsePoster poster) {
    double duration = 0.0;
    bool is_wav = false;
    if (ExtensionLower(FileNameFromPath(req.file_path)) == ".wav") {
        const std::optional<double> d = ProbeWavFile(Utf8ToWide(req.file_path));
        if (d.has_value()) {
            duration = *d;
            is_wav = true;
        }
    }
    json result;
    result["durationSec"] = duration;
    result["isWav"] = is_wav;
    poster(MakeSuccessResponse(id, std::move(result)));
}

// ===========================================================================
// Contract v5 timeline providers (I6/I7/I8/I9/I11).
//
// Design mirrors the two existing shapes in this file:
//   * The synchronous providers (getSelection, insertProvisional, scanObjects,
//     replaceObject, updateObjectText) do all their AviUtl2 SDK object work
//     inside a call_edit_section_param callback - a plain (non-capturing)
//     function driven by a POD-pointer context, exactly like InsertMediaEditProc.
//   * The asynchronous intercepts (cutoutRange, extractAudio) render off the UI
//     thread on the HTTP worker pool, collecting one frame at a time through the
//     CaptureState/condvar/10s-timeout/abandoned pattern of CaptureFrameWorker.
//     wait_rendering_task() is deliberately NOT used (it can deadlock under the
//     reference/edit lock; see plugin2.h).
//
// Handles are never retained (they go stale across undo/reload): every re-find
// happens inside the callback via find_object + provisional::AliasMatchesJob.
//
// Media object creation is funnelled through ONE helper, CreateMediaObject
// (defined near the top of this file, next to InsertMediaEditProc). It goes via
// create_object_from_alias rather than create_object_from_media_file because
// only the alias can carry the "audio present" item - without it the ribbon is
// one-tone and the object's context menu has no "separate audio" (section
// 3-140). The legacy API stays as the fallback for every case the alias cannot
// or must not cover, so a failure there is never worse than before.
//
// Character encoding: alias text is UTF-8 (get_object_alias / LPCSTR), effect &
// item names are UTF-16 (built at runtime from the UTF-8 byte escapes below so
// this stays an ASCII-only translation unit), and paths are wide. All string
// conversions reuse strconv (Utf8ToWide / WideToUtf8).
// ===========================================================================

// UTF-8 byte escapes for the AviUtl2 Japanese effect / item names this file
// needs. The ones it shares with alias_util.cpp are kept byte-for-byte
// identical to that copy so the two stay in sync (alias_util.cpp also carries
// names used only by its builders, which are deliberately not duplicated here).
const char kEffectTextJp[] =
    "\xe3\x83\x86\xe3\x82\xad\xe3\x82\xb9\xe3\x83\x88";  // "text" effect + text item
const char kEffectVideoFileJp[] =
    "\xe5\x8b\x95\xe7\x94\xbb\xe3\x83\x95\xe3\x82\xa1\xe3\x82\xa4\xe3\x83\xab";  // "video file"
const char kEffectImageFileJp[] =
    "\xe7\x94\xbb\xe5\x83\x8f\xe3\x83\x95\xe3\x82\xa1\xe3\x82\xa4\xe3\x83\xab";  // "image file"
const char kEffectAudioFileJp[] =
    "\xe9\x9f\xb3\xe5\xa3\xb0\xe3\x83\x95\xe3\x82\xa1\xe3\x82\xa4\xe3\x83\xab";  // "audio file"
const char kItemFileJp[] =
    "\xe3\x83\x95\xe3\x82\xa1\xe3\x82\xa4\xe3\x83\xab";  // "file" item (path)
// Contract v10 (source trim, section 1-6): the three "video file" effect items
// the trim decision reads off a selected object.
const char kItemPlaybackPositionJp[] =
    "\xe5\x86\x8d\xe7\x94\x9f\xe4\xbd\x8d\xe7\xbd\xae";  // "playback position"
const char kItemPlaybackSpeedJp[] =
    "\xe5\x86\x8d\xe7\x94\x9f\xe9\x80\x9f\xe5\xba\xa6";  // "playback speed"
const char kItemLoopPlayJp[] =
    "\xe3\x83\xab\xe3\x83\xbc\xe3\x83\x97\xe5\x86\x8d\xe7\x94\x9f";  // "loop playback"

// Read one item of the "video file" effect for a selected object (contract v10).
//
// Two sources, in this order:
//  1. get_object_item_value - the LIVE value, present even for items the alias
//     serializer happens to leave at their default;
//  2. the object's alias text, already copied by the caller - the fallback for a
//     host build where the getter is missing.
// Returns false when neither yields anything, which every caller turns into
// "keep the conservative default" (no playback range / neutral speed / no loop).
//
// String-lifetime rule (plugin2.h): every pointer an SDK getter returns is only
// valid until the next string-returning call, so the value is copied into the
// std::string BEFORE anything else runs.
bool ReadVideoFileItem(EDIT_SECTION* edit, OBJECT_HANDLE o, const std::string& alias,
                       const char* item_jp, std::string* out) {
    if (edit->get_object_item_value != nullptr) {
        const std::wstring effect = Utf8ToWide(kEffectVideoFileJp);
        const std::wstring item = Utf8ToWide(item_jp);
        const char* v = edit->get_object_item_value(o, effect.c_str(), item.c_str());
        if (v != nullptr) {
            out->assign(v);  // copy immediately (lifetime rule)
            if (!out->empty()) {
                return true;
            }
        }
    }
    if (!alias.empty() && ParseAliasItemValue(alias, kEffectVideoFileJp, item_jp, out)) {
        return !out->empty();
    }
    return false;
}

// Upper bound on find_object hops per layer, so a host that reports a
// never-advancing end frame can never spin us forever. Objects on one layer
// cannot exceed the timeline's frame count; +16 covers empty/edge cases.
int ScanGuardMax(int frame_max) {
    long long g = static_cast<long long>(frame_max) + 16;
    if (g > 2000000) g = 2000000;
    if (g < 16) g = 16;
    return static_cast<int>(g);
}

// Pull the media file path out of an object alias (UTF-8), trying each media
// effect and each known file-key spelling. On the first hit returns true and
// fills *effect_out (the matched effect name) / *path_out (the raw value).
bool ExtractMediaFilePath(const std::string& alias, std::string* effect_out,
                          std::string* path_out) {
    const char* effects[] = {kEffectVideoFileJp, kEffectImageFileJp, kEffectAudioFileJp};
    const char* items[] = {kItemFileJp, "File", "file", "path", "Path"};
    for (const char* eff : effects) {
        for (const char* it : items) {
            std::string value;
            if (ParseAliasItemValue(alias, eff, it, &value)) {
                *effect_out = eff;
                *path_out = value;
                return true;
            }
        }
    }
    return false;
}

// Re-find the provisional object for job_id on `layer` by exact alias match.
// Advances strictly through find_object hops with a loop guard. Callback-only
// (uses the live EDIT_SECTION); the returned handle is used immediately and
// never stored. REALDEVICE-VERIFY: find_object's layer/frame search semantics.
OBJECT_HANDLE FindObjectByJob(EDIT_SECTION* edit, int layer, const std::string& job_id,
                              int frame_max) {
    if (edit->find_object == nullptr || edit->get_object_alias == nullptr ||
        edit->get_object_layer_frame == nullptr) {
        return nullptr;
    }
    const int guard_max = ScanGuardMax(frame_max);
    int frame = 0;
    int guard = 0;
    while (guard++ < guard_max) {
        OBJECT_HANDLE o = edit->find_object(layer, frame);
        if (o == nullptr) {
            break;
        }
        const OBJECT_LAYER_FRAME lf = edit->get_object_layer_frame(o);
        const char* a = edit->get_object_alias(o);
        const std::string alias = a != nullptr ? std::string(a) : std::string();
        if (AliasMatchesJob(alias, job_id)) {
            return o;
        }
        int next = lf.end + 1;               // strictly past this object
        if (next <= frame) next = frame + 1;  // guarantee forward progress
        frame = next;
    }
    return nullptr;
}

// D2 (G1 fallback fix): true when the INCLUSIVE span [frame, frame+length-1] on
// `layer` is free of any existing object, so create_object_from_alias can place
// a full-length object there without the SDK silently SHORTENING it to fit a
// smaller gap. The G1 real-device finding was that create_object_from_alias
// with a literal length 0 never returns null on a collision - it just truncates
// the object to the free space - so the old "create returned null -> fall back
// to layer_max+1" path never fired. Pre-scanning the span lets us jump straight
// to layer_max+1 when it will not fit. Scans the layer with find_object from
// frame 0 (so an object that STARTS before `frame` but reaches into the span is
// also seen) and stops once the scan passes the span's end, reusing
// FindObjectByJob's hop/guard pattern and ScanGuardMax bound. A missing
// find_object / get_object_layer_frame or a non-positive length reports "has
// room" - i.e. no behavior change from before this pre-check existed (create +
// the SDK auto-adjust decide). REALDEVICE-VERIFY (spec section 8): find_object's
// per-layer scan semantics and the exact free/occupied boundary - space exactly
// enough, one frame short, and zero length are the three cases to confirm.
bool HasRoomForLength(EDIT_SECTION* edit, int layer, int frame, int length,
                      int frame_max) {
    if (edit->find_object == nullptr || edit->get_object_layer_frame == nullptr) {
        return true;
    }
    if (length <= 0) {
        return true;
    }
    const int req_end = frame + length - 1;  // inclusive last frame we need free
    const int guard_max = ScanGuardMax(frame_max);
    int scan = 0;
    int guard = 0;
    while (guard++ < guard_max) {
        OBJECT_HANDLE o = edit->find_object(layer, scan);
        if (o == nullptr) {
            break;
        }
        const OBJECT_LAYER_FRAME lf = edit->get_object_layer_frame(o);
        if (SpansOverlap(lf.start, lf.end, frame, req_end)) {
            return false;  // the requested span is (partly) occupied
        }
        int next = lf.end + 1;              // strictly past this object
        if (next <= scan) next = scan + 1;  // guarantee forward progress
        scan = next;
        if (scan > req_end) {
            break;  // every later object starts past our span -> cannot overlap
        }
    }
    return true;
}

// --- timeline.getSelection (I6) --------------------------------------------
// param -> SelectionSnapshot*. REALDEVICE-VERIFY: call_edit_section_param is
// used (per the SDK handoff) even though this only reads; if it pollutes undo,
// switch to call_read_section_param (the getters are valid there too).
void GetSelectionEditProc(void* param, EDIT_SECTION* edit) {
    auto* snap = static_cast<SelectionSnapshot*>(param);
    snap->available = true;
    if (edit->info != nullptr) {
        snap->rate = edit->info->rate;
        snap->scale = edit->info->scale;
        snap->sample_rate = edit->info->sample_rate;
        snap->cursor_frame = edit->info->frame;
        snap->cursor_layer = edit->info->layer;
        if (edit->info->select_range_start >= 0 && edit->info->select_range_end >= 0) {
            snap->has_range = true;
            snap->range_start = edit->info->select_range_start;
            snap->range_end = edit->info->select_range_end;
        }
    }
    std::vector<OBJECT_HANDLE> objects;
    OBJECT_HANDLE focus = edit->get_focus_object != nullptr ? edit->get_focus_object()
                                                            : nullptr;
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
        SelectionItem item;
        if (edit->get_object_layer_frame != nullptr) {
            const OBJECT_LAYER_FRAME lf = edit->get_object_layer_frame(o);
            item.layer = lf.layer;
            item.frame_start = lf.start;
            item.frame_end = lf.end;
        }
        std::string alias;
        if (edit->get_object_alias != nullptr) {
            const char* a = edit->get_object_alias(o);  // valid only during callback
            if (a != nullptr) alias.assign(a);            // copy immediately
        }
        std::string effect;
        std::string path;
        if (!alias.empty() && ExtractMediaFilePath(alias, &effect, &path)) {
            item.effect_name = effect;
            item.has_file_path = true;
            item.file_path = path;
            if (!path.empty() && edit->get_media_info != nullptr) {
                const std::wstring wide_path = Utf8ToWide(path);
                MEDIA_INFO info{};
                // REALDEVICE-VERIFY: get_media_info (actual return value and the
                // 0-fallback for unsupported formats confirmed on beta52)
                if (edit->get_media_info(wide_path.c_str(), &info, sizeof(info))) {
                    item.media_width = info.width;
                    item.media_height = info.height;
                    // total_time is 0 for a still image (correct: no duration).
                    item.media_duration_sec = info.total_time;
                }
            }
            // Contract v10 (source trim, section 1-6): the played window of the
            // backing file, plus the two "the mapping is not 1:1" flags the
            // WebUI uses to decline the trim. Only meaningful for a video
            // object, so it rides inside the media branch; every failure below
            // leaves the conservative default in place (no range reported ->
            // decideSourceTrim skips -> the whole file is uploaded exactly as
            // before v10). get_object_track_value is NOT usable here: it only
            // ever returns the FIRST of the item's values (real-hardware
            // finding 2026-08-01), so the raw string is parsed instead.
            if (effect == kEffectVideoFileJp) {
                std::string raw;
                if (ReadVideoFileItem(edit, o, alias, kItemPlaybackPositionJp, &raw)) {
                    double start_sec = 0.0;
                    double end_sec = 0.0;
                    if (ParsePlaybackRange(raw, &start_sec, &end_sec)) {
                        item.playback_start_sec = start_sec;
                        item.playback_end_sec = end_sec;
                        item.has_playback_range = true;
                    }
                }
                std::string speed_raw;
                if (ReadVideoFileItem(edit, o, alias, kItemPlaybackSpeedJp, &speed_raw)) {
                    double speed = 1.0;
                    if (ParsePlaybackSpeedPercent(speed_raw, &speed)) {
                        item.playback_speed = speed;
                    }
                }
                std::string loop_raw;
                if (ReadVideoFileItem(edit, o, alias, kItemLoopPlayJp, &loop_raw)) {
                    // The raw value is "0" / "1"; anything else stays false.
                    item.loop_play = (loop_raw == "1");
                }
                if (edit->get_object_section_num != nullptr) {
                    const int sections = edit->get_object_section_num(o);
                    // A host that cannot answer reports <= 0; keep the
                    // single-section default rather than reading that as "many".
                    if (sections > 0) item.section_count = sections;
                }
                // Contract v11 (material fps, section 3-13): the material's own
                // frame rate, for the prefill "match the material" fps axis.
                // It rides inside the video branch so Media Foundation is never
                // opened for an image / audio object. `path` is converted again
                // here because get_media_info's wide_path is local to the
                // sibling branch above. No cache: one right-click resolves one
                // file, so there is nothing for a cache to hit.
                if (!path.empty()) {
                    double fps = 0.0;
                    // The error string is discarded on purpose: a failed probe
                    // is a normal outcome (a .mkv / .webm has no Media
                    // Foundation source), and media_fps then stays 0 so the
                    // webui falls back to the project fps.
                    if (ProbeMediaFps(Utf8ToWide(path), &fps, nullptr)) {
                        item.media_fps = fps;
                    }
                }
            }
        } else if (!alias.empty()) {
            // Not a media object: try the text effect. A text object stores its
            // body under the "text" effect + "text" item (kEffectTextJp for
            // both), the same key set_object_item_value writes for provisional
            // placeholders. Filling effect_name with "text" lets the webui's
            // 4-way type check (video/image/audio/text) classify it; text_content
            // carries the body for #8 ("append to the main prompt").
            std::string text;
            if (ParseAliasItemValue(alias, kEffectTextJp, kEffectTextJp, &text)) {
                item.effect_name = kEffectTextJp;
                item.has_text_content = true;
                item.text_content = text;
            } else {
                // Section 3-54: neither media nor text - report the alias' FIRST
                // effect name so the webui can recognise a partial filter (and,
                // as a side effect, name any other effect object instead of
                // reporting nothing). This has to live in THIS else, not one
                // level up: the outer `else if (!alias.empty())` branch is only
                // entered when there is an alias, so an outer else would be
                // unreachable. classifySelectionKind still drops every name
                // outside its known set to "unknown", so nothing else moves.
                item.effect_name = FirstEffectName(alias);
            }
        }
        if (edit->get_object_name != nullptr) {
            const wchar_t* nm = edit->get_object_name(o);
            if (nm != nullptr && nm[0] != L'\0') {
                item.has_object_name = true;
                item.object_name = WideToUtf8(nm);
            }
        }
        snap->selected.push_back(std::move(item));
    }
}

// --- fs.probeMediaInfo ------------------------------------------------------
// param -> ProbeMediaInfoCtx*. Best-effort media probe: a null get_media_info
// or a failed lookup leaves ok == false (the caller reports an all-zero result).
// The UTF-8 path is held as a std::string and converted to a local std::wstring
// inside the callback (get_media_info wants the wide c_str only for the call's
// duration), mirroring InsertMediaEditProc - never stash a c_str() in the ctx.
struct ProbeMediaInfoCtx {
    std::string file_path;  // UTF-8 (owned by this ctx for the call duration)
    bool ok = false;
    double duration_sec = 0.0;
    int width = 0;
    int height = 0;
};
void ProbeMediaInfoEditProc(void* param, EDIT_SECTION* edit) {
    auto* c = static_cast<ProbeMediaInfoCtx*>(param);
    if (edit->get_media_info == nullptr) {
        return;
    }
    const std::wstring wide = Utf8ToWide(c->file_path);
    MEDIA_INFO info{};
    if (edit->get_media_info(wide.c_str(), &info, sizeof(info))) {
        c->ok = true;
        // total_time is 0 for a still image (correct: no duration); width/height
        // are the real resolution. Passed through verbatim.
        c->duration_sec = info.total_time;
        c->width = info.width;
        c->height = info.height;
    }
}

// --- timeline.insertProvisional (I6; I3 collision fallback) ----------------
struct InsertProvisionalCtx {
    const char* alias;      // UTF-8 alias (owned by caller for the call duration)
    const wchar_t* name_w;  // object name (UTF-16), or nullptr
    int layer;
    int frame;
    int length;     // D2: requested length (frames) - drives the room pre-check
                    // and the explicit create_object_from_alias length arg
    int layer_max;  // I3: EDIT_INFO.layer_max, for the collision fallback
    bool ok;
    int out_layer;
    int out_frame;
    bool used_fallback;  // I3: create collided -> retried at layer_max+1
};
void InsertProvisionalEditProc(void* param, EDIT_SECTION* edit) {
    auto* c = static_cast<InsertProvisionalCtx*>(param);
    if (edit->create_object_from_alias == nullptr) {
        return;
    }
    const int frame_max = edit->info != nullptr ? edit->info->frame_max : 0;
    // D2 (G1 fallback fix): decide the target layer BEFORE creating. The old
    // code passed length 0 and relied on create_object_from_alias returning null
    // on a collision, but the SDK silently SHORTENS the object to fit the free
    // gap instead of failing, so the fallback never fired (G1 gate finding).
    // Pre-scan the requested span and jump straight to layer_max+1 when it will
    // not fit; either way create with the EXPLICIT requested length so a partial
    // gap can never quietly truncate the placeholder.
    int target_layer = c->layer;
    if (!HasRoomForLength(edit, c->layer, c->frame, c->length, frame_max)) {
        target_layer = c->layer_max + 1;
        c->used_fallback = true;
    }
    OBJECT_HANDLE o =
        edit->create_object_from_alias(c->alias, target_layer, c->frame, c->length);
    if (o == nullptr && !c->used_fallback) {
        // Defensive: create still failed on the (pre-scan) free slot - retry on
        // layer_max+1 (guaranteed empty, so the explicit length can't shorten).
        // Stays in this one edit section so the whole insert is a single undo
        // step. REALDEVICE-VERIFY: layer_max+1 create auto-provisions the layer.
        target_layer = c->layer_max + 1;
        o = edit->create_object_from_alias(c->alias, target_layer, c->frame, c->length);
        if (o != nullptr) {
            c->used_fallback = true;
        }
    }
    if (o == nullptr) {
        return;
    }
    if (edit->set_object_name != nullptr && c->name_w != nullptr) {
        edit->set_object_name(o, c->name_w);
    }
    if (edit->get_object_layer_frame != nullptr) {
        const OBJECT_LAYER_FRAME lf = edit->get_object_layer_frame(o);
        c->out_layer = lf.layer;
        c->out_frame = lf.start;
    } else {
        c->out_layer = c->used_fallback ? c->layer_max + 1 : c->layer;
        c->out_frame = c->frame;
    }
    c->ok = true;
}

// --- timeline.updateProvisionalReservation (I3) ----------------------------
// Atomic delete-old + create-new in one edit section (single undo step). The
// old placeholder is searched across every layer (FindObjectByJob is per-layer)
// by exact job-id match; only the FIRST match is deleted (any duplicates are
// left as harmless orphan text). The new placeholder is created at the resolved
// target, with the same layer_max+1 collision fallback as insertProvisional.
struct UpdateReservationCtx {
    const UpdateReservationRequest* req;
    const wchar_t* name_w;  // object name (UTF-16), or nullptr
    int layer_max;
    int frame_max;
    bool ok;
    bool deleted_old;
    int out_layer;
    int out_frame;
    bool used_fallback;
};
void UpdateProvisionalReservationEditProc(void* param, EDIT_SECTION* edit) {
    auto* c = static_cast<UpdateReservationCtx*>(param);
    const UpdateReservationRequest& r = *c->req;
    // 1-2. Find and delete the old reservation (first match across all layers).
    if (!r.old_job_id.empty() && edit->delete_object != nullptr) {
        for (int layer = 0; layer <= c->layer_max; ++layer) {
            OBJECT_HANDLE old = FindObjectByJob(edit, layer, r.old_job_id, c->frame_max);
            if (old != nullptr) {
                edit->delete_object(old);
                c->deleted_old = true;
                break;
            }
        }
    }
    // 3. Create the new placeholder at the resolved target.
    if (edit->create_object_from_alias == nullptr) {
        return;
    }
    const int frame_max = edit->info != nullptr ? edit->info->frame_max : 0;
    // D2 (G1 fallback fix, same rationale as InsertProvisionalEditProc): pre-scan
    // the requested span so a collision jumps straight to layer_max+1 instead of
    // relying on a null return that never comes, and create with the EXPLICIT
    // requested length so a partial gap can never silently truncate.
    int target_layer = r.layer;
    if (!HasRoomForLength(edit, r.layer, r.frame, r.length_frames, frame_max)) {
        target_layer = c->layer_max + 1;
        c->used_fallback = true;
    }
    OBJECT_HANDLE o = edit->create_object_from_alias(r.alias.c_str(), target_layer,
                                                     r.frame, r.length_frames);
    if (o == nullptr && !c->used_fallback) {
        // Defensive: create still failed on the (pre-scan) free slot - retry at
        // layer_max+1 (guaranteed empty, so the explicit length can't shorten).
        target_layer = c->layer_max + 1;
        o = edit->create_object_from_alias(r.alias.c_str(), target_layer, r.frame,
                                           r.length_frames);
        if (o != nullptr) {
            c->used_fallback = true;
        }
    }
    if (o == nullptr) {
        return;
    }
    if (edit->set_object_name != nullptr && c->name_w != nullptr) {
        edit->set_object_name(o, c->name_w);
    }
    if (edit->get_object_layer_frame != nullptr) {
        const OBJECT_LAYER_FRAME lf = edit->get_object_layer_frame(o);
        c->out_layer = lf.layer;
        c->out_frame = lf.start;
    } else {
        c->out_layer = c->used_fallback ? c->layer_max + 1 : r.layer;
        c->out_frame = r.frame;
    }
    c->ok = true;
}

// --- timeline.scanProvisionals / resolve* re-discovery (I6/I11) ------------
struct ScanObjectsCtx {
    std::vector<ScannedObject>* out;
    int layer_max;
    int frame_max;
};
void ScanObjectsEditProc(void* param, EDIT_SECTION* edit) {
    auto* c = static_cast<ScanObjectsCtx*>(param);
    if (edit->find_object == nullptr || edit->get_object_layer_frame == nullptr) {
        return;
    }
    const int guard_max = ScanGuardMax(c->frame_max);
    for (int layer = 0; layer <= c->layer_max; ++layer) {
        int frame = 0;
        int guard = 0;
        while (guard++ < guard_max) {
            OBJECT_HANDLE o = edit->find_object(layer, frame);
            if (o == nullptr) {
                break;
            }
            const OBJECT_LAYER_FRAME lf = edit->get_object_layer_frame(o);
            ScannedObject so;
            so.layer = lf.layer;
            so.frame_start = lf.start;
            so.frame_end = lf.end;
            if (edit->get_object_alias != nullptr) {
                const char* a = edit->get_object_alias(o);
                if (a != nullptr) so.alias.assign(a);
            }
            c->out->push_back(std::move(so));
            int next = lf.end + 1;
            if (next <= frame) next = frame + 1;
            frame = next;
        }
    }
}

// --- timeline.resolveProvisional replace path (I11) ------------------------
struct ReplaceObjectCtx {
    const ReplaceObjectRequest* req;
    int frame_max;
    bool ok;
};
void ReplaceObjectEditProc(void* param, EDIT_SECTION* edit) {
    auto* c = static_cast<ReplaceObjectCtx*>(param);
    const ReplaceObjectRequest& r = *c->req;
    // Reliable path: delete our own placeholder text object, then create the
    // finished video from the media file pinned to length_frames. This never
    // touches the user's real media (only the self-authored placeholder is
    // deleted). REALDEVICE-VERIFY: undo coalesces delete+create into one step;
    // in-place set_object_item_value on the file item is left as an optional
    // optimization, not the default.
    // This path is NOT used by the webui, so the alias-based media create
    // (CreateMediaObject, section 3-140) is deliberately not applied here.
    OBJECT_HANDLE placeholder = FindObjectByJob(edit, r.layer, r.job_id, c->frame_max);
    if (placeholder != nullptr && edit->delete_object != nullptr) {
        edit->delete_object(placeholder);
    }
    if (edit->create_object_from_media_file == nullptr) {
        return;
    }
    const std::wstring wpath = Utf8ToWide(r.video_file_path);
    OBJECT_HANDLE created =
        edit->create_object_from_media_file(wpath.c_str(), r.layer, r.frame, r.length_frames);
    c->ok = (created != nullptr);
}

// --- timeline.insertMediaForJob replace-insert path -------------------------
// The "found a provisional marker" branch of timeline.insertMediaForJob: delete
// our own placeholder text object and create the finished media in its place,
// atomically (one edit section = one undo step). Unlike ReplaceObjectEditProc
// (resolveProvisional), the length is the media's REAL length (get_media_info ->
// ProjectFramesForSeconds, resolved inside CreateMediaObject exactly like the
// plain insert path), not a caller-supplied frame count.
//
// Deliberately NO HasRoomForLength pre-check here (unlike InsertProvisional /
// UpdateReservation): the placeholder we are about to delete occupies the exact
// slot we are about to create in, so a room pre-check would false-positive
// against our own about-to-be-deleted marker, and the section-visibility of that
// delete is itself an unverified dependency. Whether the SDK truncates a
// colliding create (D2 finding: create_object_from_media_file with an explicit
// length may silently shorten to fit a partial gap) or returns null against a
// THIRD object is left to be observed on the real device (gate G3). On a null
// create at the marker slot (a genuine failure - broken file, or the slot really
// is blocked) we retry ONCE at layer_max+1 / same frame / same explicit length
// (guaranteed empty). If that too returns null we report failure; a single undo
// then restores the just-deleted marker as the user's retreat path.
struct ReplaceMediaForJobCtx {
    const ReplaceMediaForJobRequest* req;
    int layer_max;
    int frame_max;
    bool ok;
    bool used_fallback;
    int out_layer;
    int out_frame;
};
void ReplaceMediaForJobEditProc(void* param, EDIT_SECTION* edit) {
    auto* c = static_cast<ReplaceMediaForJobCtx*>(param);
    const ReplaceMediaForJobRequest& r = *c->req;
    // 1. Re-find and delete our own placeholder at the marker's layer.
    OBJECT_HANDLE placeholder = FindObjectByJob(edit, r.layer, r.job_id, c->frame_max);
    if (placeholder != nullptr && edit->delete_object != nullptr) {
        edit->delete_object(placeholder);
    }
    if (edit->create_object_from_media_file == nullptr) {
        return;
    }
    // 2. Create the media in the marker's slot (no room pre-check - see above).
    // CreateMediaObject resolves the media's REAL length itself (the same math
    // as before section 3-140) and prefers the alias path so the finished object
    // carries the "audio present" flag.
    int placed_layer = r.layer;
    OBJECT_HANDLE created = CreateMediaObject(edit, r.file_path, r.layer, r.frame);
    if (created == nullptr) {
        // Genuine failure at the marker slot -> retry once on a guaranteed-empty
        // layer_max+1 (same frame, same real length). One undo restores the
        // deleted marker if this also fails.
        placed_layer = c->layer_max + 1;
        created = CreateMediaObject(edit, r.file_path, placed_layer, r.frame);
        if (created != nullptr) {
            c->used_fallback = true;
        }
    }
    if (created == nullptr) {
        return;
    }
    if (edit->get_object_layer_frame != nullptr) {
        const OBJECT_LAYER_FRAME lf = edit->get_object_layer_frame(created);
        c->out_layer = lf.layer;
        c->out_frame = lf.start;
    } else {
        c->out_layer = placed_layer;
        c->out_frame = r.frame;
    }
    c->ok = true;
}

// --- timeline.updateProvisionalText (I11) ----------------------------------
struct UpdateTextCtx {
    const UpdateObjectTextRequest* req;
    const wchar_t* effect_w;  // L"text" effect name (UTF-16)
    const wchar_t* item_w;    // L"text" item name (UTF-16)
    int frame_max;
    bool ok;
};
void UpdateObjectTextEditProc(void* param, EDIT_SECTION* edit) {
    auto* c = static_cast<UpdateTextCtx*>(param);
    const UpdateObjectTextRequest& r = *c->req;
    OBJECT_HANDLE target = FindObjectByJob(edit, r.layer, r.job_id, c->frame_max);
    if (target == nullptr || edit->set_object_item_value == nullptr) {
        return;
    }
    // set_object_item_value applies at the current cursor position, so move the
    // cursor into the target object's span before writing and restore it after.
    // REALDEVICE-VERIFY: cursor-move-before-write requirement and restore.
    const int saved_frame = edit->info != nullptr ? edit->info->frame : 0;
    const int saved_layer = edit->info != nullptr ? edit->info->layer : 0;
    OBJECT_LAYER_FRAME lf{r.layer, r.frame, r.frame};
    if (edit->get_object_layer_frame != nullptr) {
        lf = edit->get_object_layer_frame(target);
    }
    if (edit->set_cursor_layer_frame != nullptr) {
        edit->set_cursor_layer_frame(lf.layer, lf.start);
    }
    c->ok = edit->set_object_item_value(target, c->effect_w, c->item_w, r.text.c_str());
    if (edit->set_cursor_layer_frame != nullptr) {
        edit->set_cursor_layer_frame(saved_layer, saved_frame);
    }
}

// --- timeline.deleteProvisionalByJob (I13, spec 5-10) ----------------------
// Delete the ✅ placeholder for a job in one edit section (one undo step). The
// object is re-found by exact job-id match on the layer bridge_core located it
// on (via ctx.scan_objects + FindProvisionalIndex), reusing the same
// FindObjectByJob machinery as resolve/updateText. A missing object is a no-op
// (deleted stays false); the dispatcher already treats that as success.
struct DeleteProvisionalCtx {
    const DeleteProvisionalRequest* req;
    int frame_max;
    bool deleted;
};
void DeleteProvisionalByJobEditProc(void* param, EDIT_SECTION* edit) {
    auto* c = static_cast<DeleteProvisionalCtx*>(param);
    const DeleteProvisionalRequest& r = *c->req;
    if (edit->delete_object == nullptr) {
        return;
    }
    OBJECT_HANDLE o = FindObjectByJob(edit, r.layer, r.job_id, c->frame_max);
    if (o != nullptr) {
        edit->delete_object(o);
        c->deleted = true;
    }
}

// ---------------------------------------------------------------------------
// Async range rendering (cutoutRange I7/I9, extractAudio I8).
// ---------------------------------------------------------------------------

// Blocking single-frame video render (mirrors CaptureFrameWorker's condvar +
// 10s timeout + abandoned handoff). Returns false on timeout / empty buffer.
// `caller` only names the method in the timeout warning: two methods now use
// this helper, and a line that always said "cutoutRange" would send the reader
// to the wrong place.
bool RenderSceneVideoFrame(EDIT_HANDLE* handle, int frame, int* out_w, int* out_h,
                           std::vector<unsigned char>* out_rgba,
                           const wchar_t* caller = L"timeline.cutoutRange") {
    auto* st = new CaptureState();
    if (!handle->rendering_scene_video(frame, st, &CaptureRenderCb)) {
        delete st;
        return false;
    }
    std::unique_lock<std::mutex> lock(st->m);
    if (!st->cv.wait_for(lock, std::chrono::seconds(10), [st] { return st->done; })) {
        st->abandoned = true;  // hand ownership to the late callback
        lock.unlock();
        LogWarn(std::wstring(caller) +
                L": video render callback timed out after 10s");
        return false;
    }
    *out_w = st->width;
    *out_h = st->height;
    *out_rgba = std::move(st->rgba);
    lock.unlock();
    delete st;  // callback completed and cannot fire again
    return *out_w > 0 && *out_h > 0 &&
           out_rgba->size() >=
               static_cast<size_t>(*out_w) * static_cast<size_t>(*out_h) * 4u;
}

// Shared state for a single-frame audio render (planar float32, mirrors
// CaptureState). Heap-owned so a timed-out worker can abandon it to the late
// callback rather than dangle.
struct AudioCaptureState {
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    bool abandoned = false;
    int sample_num = 0;
    std::vector<float> left;
    std::vector<float> right;
};
void AudioRenderCb(void* param, int frame, const float* b0, const float* b1,
                   int sample_num) {
    (void)frame;
    auto* st = static_cast<AudioCaptureState*>(param);
    bool abandoned = false;
    {
        std::lock_guard<std::mutex> lock(st->m);
        st->sample_num = sample_num;
        if (sample_num > 0) {
            if (b0 != nullptr) st->left.assign(b0, b0 + sample_num);
            if (b1 != nullptr) st->right.assign(b1, b1 + sample_num);
        }
        st->done = true;
        abandoned = st->abandoned;
    }
    st->cv.notify_one();
    if (abandoned) {
        delete st;
    }
}
bool RenderSceneAudioFrame(EDIT_HANDLE* handle, int frame, std::vector<float>* left,
                           std::vector<float>* right, int* sample_num) {
    if (handle->rendering_scene_audio == nullptr) {
        return false;
    }
    auto* st = new AudioCaptureState();
    if (!handle->rendering_scene_audio(frame, st, &AudioRenderCb)) {
        delete st;
        return false;
    }
    std::unique_lock<std::mutex> lock(st->m);
    if (!st->cv.wait_for(lock, std::chrono::seconds(10), [st] { return st->done; })) {
        st->abandoned = true;
        lock.unlock();
        LogWarn(L"timeline audio render callback timed out after 10s");
        return false;
    }
    *sample_num = st->sample_num;
    *left = std::move(st->left);
    *right = std::move(st->right);
    lock.unlock();
    delete st;
    return true;
}

// --- audioMode == "solo": disable every layer not in solo_keep_layers, render,
// then ALWAYS restore (I9). REALDEVICE-VERIFY: solo isolation has no community
// reference implementation; the "mix" path works with solo disabled.
struct SoloDisableCtx {
    const std::vector<int>* keep;
    int layer_max;
    std::vector<char>* saved;  // out: original enable state per layer index
    bool ran;
};
void SoloDisableEditProc(void* param, EDIT_SECTION* edit) {
    auto* c = static_cast<SoloDisableCtx*>(param);
    c->ran = true;
    c->saved->assign(static_cast<size_t>(c->layer_max) + 1, 1);
    for (int layer = 0; layer <= c->layer_max; ++layer) {
        const bool enabled =
            edit->get_layer_enable != nullptr ? edit->get_layer_enable(layer) : true;
        (*c->saved)[static_cast<size_t>(layer)] = enabled ? 1 : 0;
        bool keep = false;
        for (int k : *c->keep) {
            if (k == layer) {
                keep = true;
                break;
            }
        }
        if (!keep && edit->set_layer_enable != nullptr) {
            edit->set_layer_enable(layer, false);
        }
    }
}
struct SoloRestoreCtx {
    const std::vector<char>* saved;
};
void SoloRestoreEditProc(void* param, EDIT_SECTION* edit) {
    auto* c = static_cast<SoloRestoreCtx*>(param);
    if (edit->set_layer_enable == nullptr) {
        return;
    }
    for (size_t layer = 0; layer < c->saved->size(); ++layer) {
        edit->set_layer_enable(static_cast<int>(layer), (*c->saved)[layer] != 0);
    }
}
// RAII: restores the solo layer-enable snapshot even on an early return / throw.
struct SoloGuard {
    EDIT_HANDLE* handle;
    std::vector<char>* saved;
    bool* applied;
    ~SoloGuard() {
        if (applied != nullptr && *applied && handle != nullptr &&
            handle->call_edit_section_param != nullptr) {
            SoloRestoreCtx rc{saved};
            handle->call_edit_section_param(&rc, &SoloRestoreEditProc);
            *applied = false;
        }
    }
};

// Apply solo isolation if requested; records whether it took effect so the
// SoloGuard restores it. Returns via *applied.
void MaybeApplySolo(EDIT_HANDLE* handle, const std::vector<int>& keep, int layer_max,
                    std::vector<char>* saved, bool* applied) {
    *applied = false;
    if (handle->call_edit_section_param == nullptr) {
        return;
    }
    SoloDisableCtx sc;
    sc.keep = &keep;
    sc.layer_max = layer_max;
    sc.saved = saved;
    sc.ran = false;
    if (handle->call_edit_section_param(&sc, &SoloDisableEditProc) && sc.ran) {
        *applied = true;
    }
}

// Clamp a normalized float sample to signed 16-bit PCM.
int16_t FloatToPcm16(float v) {
    float x = v;
    if (x > 1.0f) x = 1.0f;
    if (x < -1.0f) x = -1.0f;
    int s = static_cast<int>(x * 32767.0f);
    if (s > 32767) s = 32767;
    if (s < -32768) s = -32768;
    return static_cast<int16_t>(s);
}

// Write a minimal 16-bit PCM WAV (audio-only extract output). mf_mp4_writer has
// no audio-only mode (it always adds a video stream), and the contract permits
// wav, so extractAudio emits a self-contained PCM wav via Win32 file I/O.
bool WriteWavFile(const std::wstring& path, const std::vector<int16_t>& interleaved,
                  int sample_rate, int channels) {
    HANDLE hf = ::CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hf == INVALID_HANDLE_VALUE) {
        return false;
    }
    const uint32_t data_bytes =
        static_cast<uint32_t>(interleaved.size() * sizeof(int16_t));
    const uint16_t block_align = static_cast<uint16_t>(channels * 2);
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * block_align;
    unsigned char h[44];
    auto put32 = [](unsigned char* p, uint32_t v) {
        p[0] = static_cast<unsigned char>(v & 0xff);
        p[1] = static_cast<unsigned char>((v >> 8) & 0xff);
        p[2] = static_cast<unsigned char>((v >> 16) & 0xff);
        p[3] = static_cast<unsigned char>((v >> 24) & 0xff);
    };
    auto put16 = [](unsigned char* p, uint16_t v) {
        p[0] = static_cast<unsigned char>(v & 0xff);
        p[1] = static_cast<unsigned char>((v >> 8) & 0xff);
    };
    std::memcpy(h + 0, "RIFF", 4);
    put32(h + 4, 36 + data_bytes);
    std::memcpy(h + 8, "WAVE", 4);
    std::memcpy(h + 12, "fmt ", 4);
    put32(h + 16, 16);
    put16(h + 20, 1);  // PCM
    put16(h + 22, static_cast<uint16_t>(channels));
    put32(h + 24, static_cast<uint32_t>(sample_rate));
    put32(h + 28, byte_rate);
    put16(h + 32, block_align);
    put16(h + 34, 16);
    std::memcpy(h + 36, "data", 4);
    put32(h + 40, data_bytes);
    DWORD written = 0;
    bool ok = ::WriteFile(hf, h, 44, &written, nullptr) && written == 44;
    if (ok && data_bytes > 0) {
        ok = ::WriteFile(hf, interleaved.data(), data_bytes, &written, nullptr) &&
             written == data_bytes;
    }
    ::CloseHandle(hf);
    return ok;
}

// Runs on an HTTP worker thread: render frame_start..frame_start+frame_count-1 to
// an mp4 (video + optional audio) via Mp4Writer, then post the cutoutRange reply.
void CutoutRangeWorker(EDIT_HANDLE* handle, CutoutRangeRequest req, int width, int height,
                       int rate, int scale, int sample_rate, int layer_max,
                       std::string dest_utf8, json_t id,
                       Bridge::ResponsePoster poster) {
    auto post_error = [&poster, &id](const char* code, const std::string& msg) {
        poster(MakeErrorResponse(id, code, msg));
    };

    std::vector<char> saved_enable;
    bool solo_applied = false;
    SoloGuard guard{handle, &saved_enable, &solo_applied};
    if (req.audio_mode == "solo" && req.with_audio) {
        MaybeApplySolo(handle, req.solo_keep_layers, layer_max, &saved_enable,
                       &solo_applied);
    }

    Mp4WriterConfig cfg;
    cfg.output_path = Utf8ToWide(dest_utf8);
    cfg.width = width;
    cfg.height = height;
    cfg.fps_num = static_cast<uint32_t>(rate > 0 ? rate : 30);
    cfg.fps_den = static_cast<uint32_t>(scale > 0 ? scale : 1);
    cfg.pixel_order = PixelOrder::kRGBA;
    cfg.has_audio = req.with_audio && handle->rendering_scene_audio != nullptr;
    cfg.audio_sample_rate = static_cast<uint32_t>(sample_rate > 0 ? sample_rate : 48000);
    cfg.audio_channels = 2;

    Mp4Writer writer;
    std::string enc_err;
    if (!writer.Initialize(cfg, &enc_err)) {
        LogWarn(std::wstring(L"timeline.cutoutRange: mp4 init failed: ") +
                Utf8ToWide(enc_err));
        post_error("CUTOUT_FAILED", "mp4 init failed: " + enc_err);
        return;
    }
    for (int i = 0; i < req.frame_count; ++i) {
        const int frame = req.frame_start + i;
        int w = 0;
        int h = 0;
        std::vector<unsigned char> rgba;
        if (!RenderSceneVideoFrame(handle, frame, &w, &h, &rgba)) {
            post_error("CUTOUT_FAILED",
                       "video render failed at frame " + std::to_string(frame));
            return;
        }
        if (!writer.WriteVideoFrame(rgba.data(), w * 4, &enc_err)) {
            post_error("CUTOUT_FAILED", "WriteVideoFrame failed: " + enc_err);
            return;
        }
        if (cfg.has_audio) {
            std::vector<float> left;
            std::vector<float> right;
            int sn = 0;
            if (RenderSceneAudioFrame(handle, frame, &left, &right, &sn) && sn > 0) {
                const float* rp = right.empty() ? nullptr : right.data();
                const float* lp = left.empty() ? nullptr : left.data();
                writer.WriteAudioSamples(lp, rp, sn, &enc_err);
            }
        }
    }
    if (!writer.Finalize(&enc_err)) {
        LogWarn(std::wstring(L"timeline.cutoutRange: mp4 finalize failed: ") +
                Utf8ToWide(enc_err));
        post_error("CUTOUT_FAILED", "mp4 finalize failed: " + enc_err);
        return;
    }
    LogInfo(std::wstring(L"timeline.cutoutRange: wrote ") +
            std::to_wstring(req.frame_count) + L" frames -> " + Utf8ToWide(dest_utf8));
    poster(MakeSuccessResponse(
        id, MakeCutoutResult(dest_utf8, width, height, req.frame_count, cfg.has_audio)));
}

// Runs on an HTTP worker thread: render the audio of the range, mux to a wav,
// and post the extractAudio reply. Silence (no real stream) -> EXTRACT_FAILED.
void ExtractAudioWorker(EDIT_HANDLE* handle, ExtractAudioRequest req, int sample_rate,
                        int layer_max, std::string dest_utf8, json_t id,
                        Bridge::ResponsePoster poster) {
    auto post_error = [&poster, &id](const char* code, const std::string& msg) {
        poster(MakeErrorResponse(id, code, msg));
    };
    if (handle->rendering_scene_audio == nullptr) {
        post_error("EXTRACT_FAILED", "rendering_scene_audio is not available");
        return;
    }

    std::vector<char> saved_enable;
    bool solo_applied = false;
    SoloGuard guard{handle, &saved_enable, &solo_applied};
    if (req.audio_mode == "solo") {
        MaybeApplySolo(handle, req.solo_keep_layers, layer_max, &saved_enable,
                       &solo_applied);
    }

    std::vector<int16_t> pcm;
    double peak = 0.0;
    long long total_samples = 0;
    for (int i = 0; i < req.frame_count; ++i) {
        const int frame = req.frame_start + i;
        std::vector<float> left;
        std::vector<float> right;
        int sn = 0;
        if (!RenderSceneAudioFrame(handle, frame, &left, &right, &sn)) {
            post_error("EXTRACT_FAILED",
                       "audio render failed at frame " + std::to_string(frame));
            return;
        }
        const int lsz = static_cast<int>(left.size());
        const int rsz = static_cast<int>(right.size());
        for (int s = 0; s < sn; ++s) {
            const float l = s < lsz ? left[s] : 0.0f;
            const float r = s < rsz ? right[s] : (s < lsz ? left[s] : 0.0f);
            const double al = std::fabs(static_cast<double>(l));
            const double ar = std::fabs(static_cast<double>(r));
            if (al > peak) peak = al;
            if (ar > peak) peak = ar;
            pcm.push_back(FloatToPcm16(l));
            pcm.push_back(FloatToPcm16(r));
        }
        total_samples += sn;
    }
    const int clamped_samples =
        total_samples > INT_MAX ? INT_MAX : static_cast<int>(total_samples);
    if (!IsAudioStreamPresent(clamped_samples, peak)) {
        LogWarn(L"timeline.extractAudio: no audio stream present (silent range)");
        post_error("EXTRACT_FAILED", "no audio stream present (silent range)");
        return;
    }
    const int sr = sample_rate > 0 ? sample_rate : 48000;
    if (!WriteWavFile(Utf8ToWide(dest_utf8), pcm, sr, 2)) {
        post_error("EXTRACT_FAILED", "failed to write wav file");
        return;
    }
    const double duration = static_cast<double>(total_samples) / static_cast<double>(sr);
    LogInfo(std::wstring(L"timeline.extractAudio: wrote ") +
            std::to_wstring(total_samples) + L" samples -> " + Utf8ToWide(dest_utf8));
    poster(MakeSuccessResponse(id, MakeExtractAudioResult(dest_utf8, duration, sr, true)));
}

// ---------------------------------------------------------------------------
// Object tracking (timeline.trackObject / timeline.cancelTracking, 3-54).
// Docs\OBJECT_TRACKING_DESIGN.md sections 5.4-5.7 are the canonical procedure;
// the comments below only record what is specific to this translation unit.
// ---------------------------------------------------------------------------

// One tracking run at a time (design 5.2). g_track_running is claimed by the
// dispatcher with exchange(true) and released by TrackRunningGuard on every
// worker exit path; g_track_cancel is the stop button's cooperative flag,
// cleared at the start of each run and read at the top of every loop iteration.
std::atomic<bool> g_track_running{false};
std::atomic<bool> g_track_cancel{false};

// Raised by Bridge::Shutdown BEFORE the HTTP worker pool is torn down, and
// never cleared - once the host is taking the plugin down there is nothing to
// go back to, so the start of a run does not reset it (nor could there be one).
// Why it exists: Shutdown runs on the UI thread and waits for the worker to
// finish, while every call_edit_section_param the worker still wants to make
// has to run ON that same UI thread - so the two would wait for each other for
// ever. The tracking worker therefore SKIPS its remaining edit-section calls
// once this is set.
// The price, deliberately paid: an exit in the middle of a run can leave the
// tracked object's layer switched off in the project (the user switches it back
// on by hand), and a write-back that had not started yet is dropped.
// NOT covered: a worker already INSIDE call_edit_section_param when the flag
// goes up. No flag can close that window; it is known and accepted.
std::atomic<bool> g_shutting_down{false};

// Opening a tracking session is the only request that may have to wait for the
// backend to load the tracker model, for which the utility worker allows itself
// 180 s (Docs\OBJECT_TRACKING_DESIGN.md section 9). This has to expire AFTER
// that budget, or a cold start would always look like a timeout here; every
// other request in the run keeps kDefaultRequestTimeoutMs.
constexpr int kTrackSessionOpenTimeoutMs = 190000;

// RAII release of g_track_running - the same shape as SoloGuard, for the same
// reason: the worker has a dozen early returns and every one of them must leave
// the feature usable again.
struct TrackRunningGuard {
    ~TrackRunningGuard() { g_track_running.store(false); }
};

// The object that OWNS `frame` on `layer`, or nullptr. find_object searches
// "from this frame ONWARDS" (SDK plugin2.h line 170), so on its own it would
// happily return a LATER object when the requested frame is empty. The
// trackObject contract guarantees `frame` is the selection snapshot's
// frameStart, i.e. the target's own first frame - so the containment check
// below is an assertion that the contract held, and a refusal to edit a
// stranger's object when it did not (plan risk 7).
OBJECT_HANDLE FindObjectAt(EDIT_SECTION* edit, int layer, int frame,
                           OBJECT_LAYER_FRAME* out_lf) {
    if (edit->find_object == nullptr || edit->get_object_layer_frame == nullptr) {
        return nullptr;
    }
    OBJECT_HANDLE o = edit->find_object(layer, frame);
    if (o == nullptr) {
        return nullptr;
    }
    const OBJECT_LAYER_FRAME lf = edit->get_object_layer_frame(o);
    if (frame < lf.start || frame > lf.end) {
        return nullptr;  // the hit starts LATER - not the object we were told about
    }
    if (out_lf != nullptr) {
        *out_lf = lf;
    }
    return o;
}

// Step 1: read the seed object's alias, span and name. The alias is copied
// IMMEDIATELY because get_object_alias' buffer only survives until the next
// string-returning SDK call on this thread - which get_object_name, called
// right after, is (SDK plugin2.h lines 192 and 329).
struct ReadObjectAliasCtx {
    int layer = 0;
    int frame = 0;
    bool ok = false;
    std::string alias;     // UTF-8, owned copy
    bool has_name = false;
    std::string name;      // UTF-8 object name (empty when the host uses its default)
    int start = 0;
    int end = 0;
};
void ReadObjectAliasEditProc(void* param, EDIT_SECTION* edit) {
    auto* c = static_cast<ReadObjectAliasCtx*>(param);
    OBJECT_LAYER_FRAME lf{};
    OBJECT_HANDLE o = FindObjectAt(edit, c->layer, c->frame, &lf);
    if (o == nullptr || edit->get_object_alias == nullptr) {
        return;
    }
    const char* a = edit->get_object_alias(o);
    if (a == nullptr) {
        return;
    }
    c->alias.assign(a);  // copy BEFORE any other string-returning SDK call
    c->start = lf.start;
    c->end = lf.end;
    if (edit->get_object_name != nullptr) {
        const wchar_t* nm = edit->get_object_name(o);
        if (nm != nullptr && nm[0] != L'\0') {
            c->has_name = true;
            c->name = WideToUtf8(nm);
        }
    }
    c->ok = true;
}

// Step 3: hide the ONE layer the partial filter sits on while the frames are
// rendered, so effects the user already attached to it are not baked into the
// pixels the tracker sees (design 5.4 step 3). This is the opposite selection
// from SoloDisableEditProc, which disables everything EXCEPT a keep list; the
// RAII shape is borrowed from SoloGuard unchanged.
struct LayerDisableOneCtx {
    int layer = 0;
    bool saved_enabled = true;  // out: the state to restore
    bool ran = false;
};
void LayerDisableOneEditProc(void* param, EDIT_SECTION* edit) {
    auto* c = static_cast<LayerDisableOneCtx*>(param);
    if (edit->set_layer_enable == nullptr) {
        // Run without the isolation: the tracker then sees the user's own
        // effects baked into the frames, which is worth a line in the log.
        LogWarn(L"timeline.trackObject: the host cannot toggle layers - tracking "
                L"without hiding the object's own layer");
        return;
    }
    c->saved_enabled =
        edit->get_layer_enable != nullptr ? edit->get_layer_enable(c->layer) : true;
    edit->set_layer_enable(c->layer, false);
    c->ran = true;
}
struct LayerRestoreOneCtx {
    int layer = 0;
    bool enabled = true;
};
void LayerRestoreOneEditProc(void* param, EDIT_SECTION* edit) {
    auto* c = static_cast<LayerRestoreOneCtx*>(param);
    if (edit->set_layer_enable == nullptr) {
        return;
    }
    edit->set_layer_enable(c->layer, c->enabled);
}
// RAII: re-enables the layer on an early return, an exception, or a cancel.
struct LayerEnableGuard {
    EDIT_HANDLE* handle = nullptr;
    int layer = 0;
    bool* saved_enabled = nullptr;
    bool* applied = nullptr;
    ~LayerEnableGuard() {
        if (applied == nullptr || !*applied || handle == nullptr ||
            handle->call_edit_section_param == nullptr) {
            return;
        }
        if (g_shutting_down.load()) {
            // See g_shutting_down: calling into the edit section here would
            // deadlock against the UI thread that is shutting us down. The
            // layer stays switched off - the user can switch it back on.
            LogWarn(L"timeline.trackObject: shutting down - the hidden layer is "
                    L"left as it is");
            *applied = false;
            return;
        }
        LayerRestoreOneCtx rc;
        rc.layer = layer;
        rc.enabled = saved_enabled != nullptr ? *saved_enabled : true;
        if (!handle->call_edit_section_param(&rc, &LayerRestoreOneEditProc)) {
            LogWarn(L"timeline.trackObject: the layer could not be switched back "
                    L"on (the edit section did not run)");
        }
        *applied = false;
    }
};

// Step 8: delete the seed object and recreate it from the keyframed alias, in
// ONE edit section so a single undo takes the user back (design 5.7). The SDK
// has no "add a keyframe" call and no way to change an existing object's
// sections, so delete + create is the only route. No layer_max+1 fallback: the
// object has to end up exactly where it was, and the slot it vacated a moment
// ago is by definition free. If the new alias will not create, the ORIGINAL
// alias is put back immediately so the worst case is "nothing changed".
struct ReplacePartialFilterCtx {
    int layer = 0;
    int frame = 0;
    int length = 0;
    const std::string* new_alias = nullptr;
    const std::string* orig_alias = nullptr;
    const wchar_t* name_w = nullptr;  // nullptr when the object had no custom name
    bool ok = false;
    bool restored = false;   // create failed but the original came back
    bool not_found = false;  // the slot is empty now
    bool mismatch = false;   // the slot holds a DIFFERENT object than we read
};
void ReplacePartialFilterEditProc(void* param, EDIT_SECTION* edit) {
    auto* c = static_cast<ReplacePartialFilterCtx*>(param);
    if (edit->delete_object == nullptr || edit->create_object_from_alias == nullptr) {
        return;
    }
    OBJECT_HANDLE target = FindObjectAt(edit, c->layer, c->frame, nullptr);
    if (target == nullptr) {
        c->not_found = true;
        return;  // moved or deleted while we were tracking - do nothing
    }
    // A run takes minutes, and nothing stops the user from deleting the seed
    // object and dropping a different one into the same layer and frame while
    // it runs. delete_object would then destroy a stranger's work, so the alias
    // is read back and compared with the one the keyframes were built from.
    // Copied IMMEDIATELY, like every other get_object_alias caller here: the
    // buffer only survives until the next string-returning SDK call.
    if (edit->get_object_alias == nullptr || c->orig_alias == nullptr) {
        c->mismatch = true;
        return;
    }
    const char* current = edit->get_object_alias(target);
    if (current == nullptr) {
        c->mismatch = true;
        return;
    }
    const std::string current_alias(current);
    if (current_alias != *c->orig_alias) {
        c->mismatch = true;
        return;
    }
    edit->delete_object(target);
    OBJECT_HANDLE created = edit->create_object_from_alias(c->new_alias->c_str(),
                                                           c->layer, c->frame, c->length);
    if (created == nullptr) {
        // Put the user's object back exactly as it was. The alias' own frame
        // header decides the length either way (SDK line 161), so the length
        // argument here is the same redundant second pin as above.
        created = edit->create_object_from_alias(c->orig_alias->c_str(), c->layer,
                                                 c->frame, c->length);
        c->restored = (created != nullptr);
    } else {
        c->ok = true;
    }
    // The object name is NOT part of the alias, so it is reapplied to whichever
    // object now stands in the slot (UpdateProvisionalReservationEditProc does
    // the same after its create).
    if (created != nullptr && edit->set_object_name != nullptr && c->name_w != nullptr) {
        edit->set_object_name(created, c->name_w);
    }
}

// Lift the backend's error envelope ({"error":{"code","message","detail"}},
// api/errors.py) out of a non-2xx body. Any TRACK_* code is forwarded verbatim
// so the webui can tell "not installed" (TRACK_UNAVAILABLE) from "the worker
// died" (TRACK_FAILED); anything else collapses to TRACK_FAILED.
std::string BackendTrackError(const HttpResponse& resp, std::string* message) {
    std::string code = "TRACK_FAILED";
    const json j = json::parse(resp.body, nullptr, /*allow_exceptions=*/false);
    if (!j.is_discarded() && j.is_object() && j.contains("error") &&
        j["error"].is_object()) {
        const json& e = j["error"];
        if (e.contains("message") && e["message"].is_string()) {
            *message = e["message"].get<std::string>();
        }
        if (e.contains("detail") && e["detail"].is_string()) {
            const std::string detail = e["detail"].get<std::string>();
            if (!detail.empty()) {
                *message += (message->empty() ? "" : ": ") + detail;
            }
        }
        if (e.contains("code") && e["code"].is_string()) {
            const std::string c = e["code"].get<std::string>();
            if (c.rfind("TRACK_", 0) == 0) {
                code = c;
            }
        }
    }
    if (message->empty()) {
        *message = "tracking backend returned HTTP " + std::to_string(resp.status);
    }
    return code;
}

// Closes a tracking session on EVERY exit from the scope that owns it. A failed
// DELETE is logged and otherwise ignored: the backend expires an idle session
// by itself, and there is nothing useful the user could do about it.
struct TrackSessionGuard {
    HttpClient* http;
    std::string url;
    ~TrackSessionGuard() {
        const HttpResponse r =
            http->RequestSync(url, "DELETE", std::string(), kDefaultRequestTimeoutMs);
        if (r.transport != TransportError::kNone || r.status < 200 || r.status >= 300) {
            LogWarn(L"timeline.trackObject: closing the tracking session failed");
        }
    }
};

// Nearest-rank quantile of an ALREADY SORTED, non-empty sample: the value at the
// rounded rank q*(n-1), clamped to both ends. Pure, and deliberately without
// interpolation - this only feeds a log line, and a value some frame really
// scored reads better than an average of two. A short sample therefore just
// yields an end value (n == 1 gives that one value for every q).
double SortedQuantile(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) {
        return 0.0;
    }
    const size_t n = sorted.size();
    if (q <= 0.0) {
        return sorted.front();
    }
    if (q >= 1.0) {
        return sorted.back();
    }
    size_t index = static_cast<size_t>(q * static_cast<double>(n - 1) + 0.5);
    if (index >= n) {
        index = n - 1;
    }
    return sorted[index];
}

// A score as exactly three decimals, without going through the C locale's
// decimal point: the host process may have called setlocale, and "%.3f" would
// then log "0,412" (alias_util.cpp's FixedDecimals avoids snprintf for the same
// reason). Scores live in [0, 1], so the magnitude guard only has to catch a
// value that is not finite.
std::wstring FormatScore(double value) {
    if (!std::isfinite(value)) {
        return L"nan";
    }
    const bool negative = value < 0.0;
    double magnitude = negative ? -value : value;
    if (!(magnitude < 1.0e9)) {
        magnitude = 1.0e9;
    }
    const long long scaled = static_cast<long long>(magnitude * 1000.0 + 0.5);
    std::wstring frac = std::to_wstring(scaled % 1000);
    while (frac.size() < 3) {
        frac.insert(frac.begin(), L'0');
    }
    std::wstring out;
    if (negative && scaled != 0) {
        out += L'-';
    }
    out += std::to_wstring(scaled / 1000);
    out += L'.';
    out += frac;
    return out;
}

// Runs on an HTTP worker thread. Renders the seed object's span frame by frame,
// posts each frame to the tracking session, post-processes the raw boxes and
// writes the result back as keyframes. See design 5.4 for the nine steps; the
// numbered comments below line up with it one to one.
void TrackObjectWorker(EDIT_HANDLE* handle, HttpClient* http, TrackObjectRequest req,
                       std::string base_url, json_t id, Bridge::ResponsePoster poster) {
    TrackRunningGuard running_guard;
    const auto started_at = std::chrono::steady_clock::now();
    auto elapsed_ms = [&started_at]() -> double {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - started_at)
            .count();
    };
    auto post_error = [&poster, &id](const char* code, const std::string& msg) {
        LogWarn(std::wstring(L"timeline.trackObject: ") + Utf8ToWide(code) + L" - " +
                Utf8ToWide(msg));
        poster(MakeErrorResponse(id, code, msg));
    };
    if (handle == nullptr || handle->call_edit_section_param == nullptr ||
        handle->rendering_scene_video == nullptr) {
        post_error("NO_EDIT_HANDLE", "Edit handle is not available");
        return;
    }

    // --- 1. read the seed object -------------------------------------------
    ReadObjectAliasCtx seed;
    seed.layer = req.layer;
    seed.frame = req.frame;
    if (!handle->call_edit_section_param(&seed, &ReadObjectAliasEditProc) || !seed.ok) {
        post_error("TRACK_FAILED", "No object found at layer " +
                                       std::to_string(req.layer) + ", frame " +
                                       std::to_string(req.frame));
        return;
    }
    const int length = seed.end - seed.start + 1;
    if (length <= 0) {
        post_error("TRACK_FAILED", "The object at layer " + std::to_string(req.layer) +
                                       " has no frames");
        return;
    }

    // --- 2. read the seed box's four values (the rectangle needs the rendered
    //        frame's size, so the conversion waits for step 4) ---------------
    PartialFilterValues seed_values;
    if (!ParsePartialFilterValues(seed.alias, &seed_values)) {
        post_error("TRACK_SEED_INVALID",
                   "The selected object is not a partial filter with a readable box");
        return;
    }

    std::vector<TrackSample> samples;
    bool cancelled = false;
    int scene_w = 0;
    int scene_h = 0;
    {
        // --- 3. hide the partial filter's own layer while rendering ---------
        bool layer_disabled = false;
        bool saved_enabled = true;
        LayerEnableGuard layer_guard{handle, req.layer, &saved_enabled, &layer_disabled};
        {
            LayerDisableOneCtx dc;
            dc.layer = req.layer;
            if (handle->call_edit_section_param(&dc, &LayerDisableOneEditProc) && dc.ran) {
                saved_enabled = dc.saved_enabled;
                layer_disabled = true;
            }
        }

        // --- 4. render the first frame, open the session with ITS size ------
        std::vector<unsigned char> rgba;
        if (!RenderSceneVideoFrame(handle, seed.start, &scene_w, &scene_h, &rgba,
                                   L"timeline.trackObject")) {
            post_error("TRACK_FAILED",
                       "Scene render failed at frame " + std::to_string(seed.start));
            return;
        }
        const size_t frame_bytes =
            static_cast<size_t>(scene_w) * static_cast<size_t>(scene_h) * 4u;

        double sx = 0.0;
        double sy = 0.0;
        double sw = 0.0;
        double sh = 0.0;
        if (!PartialFilterToRect(seed_values, scene_w, scene_h, &sx, &sy, &sw, &sh) ||
            !(sw > 0.0) || !(sh > 0.0)) {
            post_error("TRACK_SEED_INVALID",
                       "The partial filter's box has no positive size");
            return;
        }

        json open_body;
        open_body["width"] = scene_w;
        open_body["height"] = scene_h;
        open_body["box"] = json{{"x", sx}, {"y", sy}, {"w", sw}, {"h", sh}};
        open_body["search_factor"] = req.search_factor;
        const std::string sessions_url =
            base_url + kBackendApiPrefix + "/utils/track/sessions";
        // Opening a session is the one request that can block on the backend
        // loading the tracker model, which it allows itself 180 s for; 30 s
        // would time out on every cold start.
        const HttpResponse open_resp = http->RequestSync(
            sessions_url, "POST", open_body.dump(), kTrackSessionOpenTimeoutMs);
        if (open_resp.transport != TransportError::kNone) {
            post_error(TransportCode(open_resp.transport), open_resp.transport_message);
            return;
        }
        if (open_resp.status < 200 || open_resp.status >= 300) {
            std::string msg;
            const std::string code = BackendTrackError(open_resp, &msg);
            post_error(code.c_str(), msg);
            return;
        }
        std::string session_id;
        {
            const json j = json::parse(open_resp.body, nullptr, false);
            if (j.is_object() && j.contains("session_id") && j["session_id"].is_string()) {
                session_id = j["session_id"].get<std::string>();
            }
        }
        if (session_id.empty()) {
            post_error("TRACK_FAILED",
                       "The tracking backend returned no usable session id");
            return;
        }
        const std::string session_url =
            sessions_url + "/" + session_id;
        TrackSessionGuard session_guard{http, session_url};
        const std::string frame_url_base = session_url + "/frame?frame=";

        // Progress throttling (design 5.3): one event per 200 ms, but the first
        // and the last frame always go out. last_index lets the forced final
        // event skip itself when the throttle happened to fire on it anyway.
        std::chrono::steady_clock::time_point last_emit{};
        bool have_emitted = false;
        int last_index = 0;
        auto emit_progress = [&](int abs_frame, int index, double score, bool force) {
            const auto now = std::chrono::steady_clock::now();
            if (!force && have_emitted &&
                std::chrono::duration_cast<std::chrono::milliseconds>(now - last_emit)
                        .count() < 200) {
                return;
            }
            last_emit = now;
            have_emitted = true;
            last_index = index;
            poster(MakeTrackProgressEvent(abs_frame, index, length, score,
                                          score < req.lost_score_threshold));
        };

        // One frame out, one box back. The pixel buffer is trimmed to exactly
        // width*height*4 - CaptureRenderCb already repacks each row to that
        // stride, so this only drops a trailing slack the vector may carry.
        auto send_frame = [&](int abs_frame, const std::vector<unsigned char>& pixels,
                              TrackSample* out, std::string* code,
                              std::string* msg) -> bool {
            if (pixels.size() < frame_bytes) {
                *code = "TRACK_FAILED";
                *msg = "Scene render produced a short frame at " +
                       std::to_string(abs_frame);
                return false;
            }
            const std::string body(reinterpret_cast<const char*>(pixels.data()),
                                   frame_bytes);
            const HttpResponse resp = http->RequestSync(
                frame_url_base + std::to_string(abs_frame), "POST", body,
                kDefaultRequestTimeoutMs, "application/octet-stream");
            if (resp.transport != TransportError::kNone) {
                *code = TransportCode(resp.transport);
                *msg = resp.transport_message;
                return false;
            }
            if (resp.status < 200 || resp.status >= 300) {
                *code = BackendTrackError(resp, msg);
                return false;
            }
            const json j = json::parse(resp.body, nullptr, false);
            if (!j.is_object() || !j.contains("box") || !j["box"].is_object()) {
                *code = "TRACK_FAILED";
                *msg = "The tracking backend returned no box for frame " +
                       std::to_string(abs_frame);
                return false;
            }
            const json& b = j["box"];
            if (!b.contains("x") || !b["x"].is_number() || !b.contains("y") ||
                !b["y"].is_number() || !b.contains("w") || !b["w"].is_number() ||
                !b.contains("h") || !b["h"].is_number()) {
                *code = "TRACK_FAILED";
                *msg = "The tracking backend returned an incomplete box for frame " +
                       std::to_string(abs_frame);
                return false;
            }
            out->frame = abs_frame - seed.start;  // 0-based, relative to the object
            out->x = b["x"].get<double>();
            out->y = b["y"].get<double>();
            out->w = b["w"].get<double>();
            out->h = b["h"].get<double>();
            // Defaulting a missing score to 0.0 would silently mark the frame
            // lost - and, with "hold", freeze the box - on a backend that has
            // simply changed its reply shape. Stop instead and say so.
            if (!j.contains("score") || !j["score"].is_number()) {
                *code = "TRACK_FAILED";
                *msg = "score missing from the tracking backend's reply for frame " +
                       std::to_string(abs_frame);
                return false;
            }
            out->score = j["score"].get<double>();
            return true;
        };

        {
            TrackSample first;
            std::string code;
            std::string msg;
            if (!send_frame(seed.start, rgba, &first, &code, &msg)) {
                post_error(code.c_str(), msg);
                return;
            }
            samples.push_back(first);
            emit_progress(seed.start, 1, first.score, /*force=*/true);
        }

        // --- 5. the loop ----------------------------------------------------
        // A render or transport failure mid-run is NOT fatal to the samples
        // already collected: the run stops there and the frames that did work
        // are still post-processed and written back, which is exactly what the
        // stop button does. The user sees a short result rather than nothing.
        for (int f = seed.start + 1; f <= seed.end; ++f) {
            if (g_track_cancel.load()) {
                cancelled = true;
                break;
            }
            int w = 0;
            int h = 0;
            std::vector<unsigned char> pixels;
            if (!RenderSceneVideoFrame(handle, f, &w, &h, &pixels,
                                       L"timeline.trackObject") ||
                w != scene_w || h != scene_h) {
                LogWarn(std::wstring(L"timeline.trackObject: render stopped at frame ") +
                        std::to_wstring(f));
                break;
            }
            TrackSample s;
            std::string code;
            std::string msg;
            if (!send_frame(f, pixels, &s, &code, &msg)) {
                LogWarn(std::wstring(L"timeline.trackObject: tracking stopped at frame ") +
                        std::to_wstring(f) + L" (" + Utf8ToWide(code) + L")");
                break;
            }
            samples.push_back(s);
            emit_progress(f, static_cast<int>(samples.size()), s.score, /*force=*/false);
        }
        if (!samples.empty() && last_index != static_cast<int>(samples.size())) {
            const TrackSample& last = samples.back();
            emit_progress(last.frame + seed.start, static_cast<int>(samples.size()),
                          last.score, /*force=*/true);
        }
        // --- 6. the session closes and the layer comes back as this scope ends
    }

    if (samples.empty()) {
        post_error("TRACK_FAILED", "No frames were tracked");
        return;
    }

    // The run's score distribution, so the lost-score threshold's default can be
    // picked from measured numbers instead of a guess. Every ending that collected
    // at least one sample logs it - a clean finish, the stop button, or a render or
    // transport failure part way through - which is why it sits here, ahead of the
    // post-processing and the write-back (both of which can still bail out).
    {
        std::vector<double> scores;
        scores.reserve(samples.size());
        int below = 0;
        for (const TrackSample& s : samples) {
            scores.push_back(s.score);
            if (s.score < req.lost_score_threshold) {  // PostProcessTrack's own test
                ++below;
            }
        }
        std::sort(scores.begin(), scores.end());
        LogInfo(std::wstring(L"timeline.trackObject: scores n=") +
                std::to_wstring(scores.size()) + L" min=" + FormatScore(scores.front()) +
                L" p10=" + FormatScore(SortedQuantile(scores, 0.10)) +
                L" median=" + FormatScore(SortedQuantile(scores, 0.50)) +
                L" max=" + FormatScore(scores.back()) +
                L" below_threshold=" + std::to_wstring(below) +
                L" (threshold=" + FormatScore(req.lost_score_threshold) + L")");
    }

    // --- 7. post-process (pure) --------------------------------------------
    TrackPostOptions opt;
    opt.lost_score_threshold = req.lost_score_threshold;
    opt.hold_on_lost = (req.lost_behavior == "hold");
    opt.smoothing = req.smoothing;
    opt.follow_size = req.follow_size;
    opt.keyframe_stride = req.keyframe_stride;
    const TrackPostResult processed = PostProcessTrack(samples, opt);

    // --- 8. write back ------------------------------------------------------
    // The alias is built and checked FIRST: an empty result means the timeline
    // is never touched, so a failure here costs the user nothing but the wait.
    const std::string new_alias = PatchAliasPartialFilterKeyframes(
        seed.alias, processed.keyframes, scene_w, scene_h, length);
    if (new_alias.empty()) {
        post_error("TRACK_WRITEBACK_FAILED",
                   "Could not build the keyframed alias; the object was left untouched");
        return;
    }
    const std::wstring name_w = seed.has_name ? Utf8ToWide(seed.name) : std::wstring();
    ReplacePartialFilterCtx rep;
    rep.layer = req.layer;
    rep.frame = seed.start;
    rep.length = length;
    rep.new_alias = &new_alias;
    rep.orig_alias = &seed.alias;
    rep.name_w = seed.has_name ? name_w.c_str() : nullptr;
    if (g_shutting_down.load()) {
        // See g_shutting_down: the edit section runs on the UI thread that is
        // waiting for this worker, so asking for one now would deadlock. The
        // keyframes are dropped; the user's object is untouched.
        post_error("TRACK_WRITEBACK_FAILED",
                   "The plugin is shutting down; the object was left untouched");
        return;
    }
    if (!handle->call_edit_section_param(&rep, &ReplacePartialFilterEditProc) || !rep.ok) {
        std::string detail;
        if (rep.not_found) {
            detail = "the object is no longer at that layer and frame";
        } else if (rep.mismatch) {
            detail = "the object had been replaced by a different one";
        } else if (rep.restored) {
            detail = "recreating the object failed, so the original was restored";
        } else {
            detail = "recreating the object failed";
        }
        post_error("TRACK_WRITEBACK_FAILED",
                   "Writing the keyframes failed: " + detail);
        return;
    }

    // --- 9. reply -----------------------------------------------------------
    LogInfo(std::wstring(L"timeline.trackObject: ") + std::to_wstring(samples.size()) +
            L" frame(s), " + std::to_wstring(processed.keyframes.size()) +
            L" keyframe(s)" + (cancelled ? L" (cancelled)" : L""));
    // PostProcessTrack works in the object-relative offsets it was fed, and the
    // keyframes stay that way (the alias is written relative to the object). The
    // lost ranges are for the user to read, so they leave in ABSOLUTE AviUtl2
    // frame numbers - the same rule as the progress event's 'frame'.
    std::vector<TrackRange> lost_abs = processed.lost_ranges;
    for (TrackRange& r : lost_abs) {
        r.start += seed.start;
        r.end += seed.start;
    }
    poster(MakeSuccessResponse(
        id, MakeTrackObjectResult(true, static_cast<int>(samples.size()),
                                  processed.keyframes, lost_abs, elapsed_ms(),
                                  cancelled)));
}

}  // namespace

Bridge::Bridge() = default;

Bridge::~Bridge() {
    Shutdown();
}

void Bridge::Initialize(ResponsePoster poster) {
    poster_ = std::move(poster);
    if (!http_) {
        http_ = std::make_unique<HttpClient>();
    }
    if (!settings_) {
        const std::wstring dir = AppDataDir();
        const std::wstring path = dir.empty() ? std::wstring() : (dir + L"\\settings.json");
        settings_ = std::make_unique<SettingsStore>(path);
        settings_->Load();
        LogInfo(std::wstring(L"Bridge: settings loaded, baseUrl=") +
                Utf8ToWide(settings_->GetBaseUrl()));
    }
    LogInfo(L"Bridge: initialized (HTTP worker pool ready)");
}

// Current backend base URL: the settings store's value once Initialize() has
// run, or the built-in default before that / if settings_ could not be
// created. Read once per request at dispatch time (see bridge_core.h's
// comment on RequestContext::base_url); an in-flight request is not affected
// by a later settings.set.
std::string Bridge::CurrentBaseUrl() const {
    return settings_ ? settings_->GetBaseUrl() : std::string(kBackendBaseUrl);
}

void Bridge::SetEditHandle(EDIT_HANDLE* handle) {
    edit_handle_ = handle;
}

void Bridge::SetPluginHwnd(void* hwnd) {
    plugin_hwnd_ = hwnd;
}

void Bridge::Shutdown() {
    // Order matters: ~HttpClient waits for the worker threads, and a tracking
    // worker still inside its loop would want the UI thread that is running
    // this. Both flags go up FIRST so the worker stops at its next check and
    // skips the edit-section calls it has left (see g_shutting_down).
    g_track_cancel.store(true);
    g_shutting_down.store(true);
    http_.reset();
    poster_ = nullptr;
}

std::string Bridge::HandleMessage(const std::string& request_json) {
    json req = json::parse(request_json, nullptr, /*allow_exceptions=*/false);

    std::string method;
    if (!req.is_discarded() && req.is_object() && req.contains("method") &&
        req["method"].is_string()) {
        method = req["method"].get<std::string>();
    }
    LogInfo(std::wstring(L"bridge request: ") +
            Utf8ToWide(method.empty() ? std::string("(none)") : method));

    // --- Asynchronous backend methods (dispatched to the HTTP worker pool) ---
    if (method == "backend.request" || method == "backend.downloadVideo") {
        const json id = ExtractId(req);
        const json params =
            (req.contains("params") && req["params"].is_object()) ? req["params"]
                                                                   : json::object();

        if (http_ == nullptr || !poster_) {
            return MakeErrorResponse(id, "BACKEND_UNREACHABLE",
                                     "HTTP client is not initialized");
        }

        if (method == "backend.request") {
            BackendRequest breq;
            std::string err;
            if (!ParseBackendRequest(params, &breq, &err)) {
                return MakeErrorResponse(id, "BAD_REQUEST", err);
            }
            const std::string url = BuildBackendUrl(CurrentBaseUrl(), breq);
            ResponsePoster poster = poster_;
            http_->RequestAsync(
                url, breq.http_method, breq.body, breq.timeout_ms,
                [poster, id](HttpResponse resp) {
                    std::string out;
                    if (resp.transport != TransportError::kNone) {
                        out = MakeErrorResponse(id, TransportCode(resp.transport),
                                                resp.transport_message);
                        LogWarn(std::wstring(L"backend.request transport error: ") +
                                Utf8ToWide(TransportCode(resp.transport)));
                    } else {
                        out = MakeSuccessResponse(
                            id, MakeBackendRequestResult(resp.status, resp.body));
                        LogInfo(std::wstring(L"backend.request done: HTTP ") +
                                std::to_wstring(resp.status));
                    }
                    poster(out);
                });
            return std::string();  // response delivered asynchronously
        }

        // backend.downloadVideo
        DownloadVideoRequest dreq;
        std::string err;
        if (!ParseDownloadVideo(params, &dreq, &err)) {
            return MakeErrorResponse(id, "BAD_REQUEST", err);
        }
        std::string url = CurrentBaseUrl() + kBackendApiPrefix + DownloadVideoPath(dreq);

        // Contract v6: destDir/fileName/noClobber, all optional and additive.
        // Omitting all three reproduces the exact pre-v6 destination
        // (AppDataDir()\downloads\<jobId>[_joined].mp4) and pre-v6
        // overwrite-in-place behavior - see ClaimNoClobberPath's comment for how
        // noClobber claims a free name without ever opening an existing file.
        const std::wstring dest_dir = dreq.has_dest_dir ? Utf8ToWide(dreq.dest_dir)
                                                        : (AppDataDir() + L"\\downloads");
        const std::string dest_name =
            dreq.has_file_name ? dreq.file_name : DownloadFileName(dreq);
        const bool no_clobber = dreq.no_clobber;
        // reuseIfPresent only makes sense on the overwrite-in-place (non-
        // noClobber) path; noClobber already claims a fresh name so there is
        // never an existing file to reuse.
        const bool reuse_if_present = dreq.reuse_if_present && !no_clobber;

        ResponsePoster poster = poster_;
        HttpClient* http = http_.get();
        http_->Post([http, url, dest_dir, dest_name, no_clobber, reuse_if_present, id, poster]() {
            std::wstring dest;
            std::wstring claimed;  // non-empty when noClobber pre-claimed this path
            if (no_clobber) {
                EnsureDirectoryTree(dest_dir);
                std::string claim_err;
                if (!ClaimNoClobberPath(dest_dir, dest_name, &dest, &claim_err)) {
                    LogWarn(std::wstring(L"downloadVideo: noClobber claim failed: ") +
                            Utf8ToWide(claim_err));
                    poster(MakeErrorResponse(id, "DOWNLOAD_FAILED", claim_err));
                    return;
                }
                claimed = dest;
            } else {
                dest = dest_dir + L"\\" + Utf8ToWide(dest_name);
                // reuseIfPresent: a completed job's clip is immutable, so if the
                // destination already exists non-empty, skip the download rather
                // than re-open it with CREATE_ALWAYS|share=0 (which fails with a
                // shared-write violation while AviUtl2 still holds the file from
                // a prior insert). Return the existing path/size, same response
                // shape as a fresh download below.
                if (reuse_if_present) {
                    WIN32_FILE_ATTRIBUTE_DATA fad{};
                    if (::GetFileAttributesExW(dest.c_str(), GetFileExInfoStandard, &fad) &&
                        (fad.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
                        ULARGE_INTEGER sz;
                        sz.LowPart = fad.nFileSizeLow;
                        sz.HighPart = fad.nFileSizeHigh;
                        if (sz.QuadPart > 0) {
                            json r;
                            r["filePath"] = WideToUtf8(dest);
                            r["sizeBytes"] = static_cast<long long>(sz.QuadPart);
                            LogInfo(std::wstring(L"downloadVideo: reuseIfPresent hit, "
                                                 L"skipping download -> ") +
                                    dest);
                            poster(MakeSuccessResponse(id, std::move(r)));
                            return;
                        }
                    }
                }
            }
            const DownloadResult result = http->DownloadSync(url, dest, kDefaultDownloadTimeoutMs);
            std::string out;
            if (result.transport != TransportError::kNone) {
                if (!claimed.empty()) {
                    ::DeleteFileW(claimed.c_str());
                }
                out = MakeErrorResponse(id, TransportCode(result.transport),
                                        result.transport_message);
                LogWarn(std::wstring(L"downloadVideo transport error: ") +
                        Utf8ToWide(TransportCode(result.transport)));
            } else if (result.status != 200) {
                if (!claimed.empty()) {
                    ::DeleteFileW(claimed.c_str());
                }
                std::string msg = "Download failed: HTTP " +
                                  std::to_string(result.status);
                if (!result.response_summary.empty()) {
                    msg += " - " + result.response_summary;
                }
                out = MakeErrorResponse(id, "DOWNLOAD_FAILED", msg);
                LogWarn(std::wstring(L"downloadVideo failed: HTTP ") +
                        std::to_wstring(result.status));
            } else {
                json r;
                r["filePath"] = WideToUtf8(result.file_path);
                r["sizeBytes"] = result.size_bytes;
                out = MakeSuccessResponse(id, std::move(r));
                LogInfo(std::wstring(L"downloadVideo done: ") +
                        std::to_wstring(result.size_bytes) + L" bytes -> " +
                        result.file_path);
            }
            poster(out);
        });
        return std::string();  // response delivered asynchronously
    }

    // --- backend.uploadFile (async multipart upload) -------------------------
    if (method == "backend.uploadFile") {
        const json id = ExtractId(req);
        const json params =
            (req.contains("params") && req["params"].is_object()) ? req["params"]
                                                                   : json::object();
        if (http_ == nullptr || !poster_) {
            return MakeErrorResponse(id, "BACKEND_UNREACHABLE",
                                     "HTTP client is not initialized");
        }
        UploadFileRequest ureq;
        std::string err;
        if (!ParseUploadFile(params, &ureq, &err)) {
            return MakeErrorResponse(id, "BAD_REQUEST", err);
        }
        if (!FileExists(ureq.file_path)) {
            return MakeErrorResponse(id, "FILE_NOT_FOUND",
                                     "Upload file not found: " + ureq.file_path);
        }
        // Contract v10: ureq.query is empty unless the caller sent a 'query'
        // object, and AppendQueryToUrl leaves the URL untouched in that case -
        // so a query-less upload is byte-for-byte the pre-v10 request.
        // UploadFileAsync / http_client need no change: CrackUrl re-joins the
        // query onto the request path.
        const std::string url = AppendQueryToUrl(
            CurrentBaseUrl() + kBackendApiPrefix + UploadPath(ureq.kind), ureq.query);
        const std::string filename = FileNameFromPath(ureq.file_path);
        const std::string content_type = ContentTypeForExtension(ureq.file_path);
        ResponsePoster poster = poster_;
        http_->UploadFileAsync(
            url, Utf8ToWide(ureq.file_path), "file", filename, content_type,
            kUploadTimeoutMs, [poster, id](HttpResponse resp) {
                std::string out;
                if (resp.transport != TransportError::kNone) {
                    out = MakeErrorResponse(id, TransportCode(resp.transport),
                                            resp.transport_message);
                    LogWarn(std::wstring(L"backend.uploadFile transport error: ") +
                            Utf8ToWide(TransportCode(resp.transport)));
                } else {
                    out = MakeSuccessResponse(
                        id, MakeBackendRequestResult(resp.status, resp.body));
                    LogInfo(std::wstring(L"backend.uploadFile done: HTTP ") +
                            std::to_wstring(resp.status));
                }
                poster(out);
            });
        return std::string();  // response delivered asynchronously
    }

    // --- timeline.captureFrame (async render -> PNG) -------------------------
    if (method == "timeline.captureFrame") {
        const json id = ExtractId(req);
        const json params =
            (req.contains("params") && req["params"].is_object()) ? req["params"]
                                                                   : json::object();
        if (http_ == nullptr || !poster_) {
            return MakeErrorResponse(id, "BACKEND_UNREACHABLE",
                                     "HTTP client is not initialized");
        }
        CaptureFrameRequest creq;
        std::string err;
        if (!ParseCaptureFrame(params, &creq, &err)) {
            return MakeErrorResponse(id, "BAD_REQUEST", err);
        }
        EDIT_HANDLE* handle = edit_handle_;
        if (handle == nullptr || handle->rendering_scene_video == nullptr) {
            return MakeErrorResponse(id, "NO_EDIT_HANDLE",
                                     "Edit handle is not available");
        }
        // Resolve the frame on the UI thread. get_edit_info() takes a reference
        // lock but we are not under an edit lock here, so this is safe.
        int frame = creq.frame;
        if (!creq.has_frame) {
            if (handle->get_edit_info == nullptr) {
                return MakeErrorResponse(id, "NO_EDIT_HANDLE",
                                         "Edit handle is not available");
            }
            EDIT_INFO info = {};
            handle->get_edit_info(&info, sizeof(info));
            frame = info.frame;
        }
        ResponsePoster poster = poster_;
        http_->Post([handle, frame, id, poster]() {
            CaptureFrameWorker(handle, frame, id, poster);
        });
        return std::string();  // response delivered asynchronously
    }

    // --- ui.pickFile (modal Open dialog, UI thread) --------------------------
    if (method == "ui.pickFile") {
        const json id = ExtractId(req);
        const json params =
            (req.contains("params") && req["params"].is_object()) ? req["params"]
                                                                   : json::object();
        std::string kind;
        std::string err;
        if (!ParsePickFileKind(params, &kind, &err)) {
            return MakeErrorResponse(id, "BAD_REQUEST", err);
        }
        // Reject a re-entrant request while a dialog is already open. The dialog's
        // modal pump can deliver another web message on this same UI thread.
        bool expected = false;
        if (!pick_dialog_open_.compare_exchange_strong(expected, true)) {
            LogWarn(L"ui.pickFile: rejected re-entrant request (dialog already open)");
            return MakeErrorResponse(id, "DIALOG_FAILED", "dialog already open");
        }
        struct OpenGuard {
            std::atomic<bool>& flag;
            ~OpenGuard() { flag.store(false); }
        } open_guard{pick_dialog_open_};

        // Build the wide "label\0pattern\0...\0\0" filter buffer the dialog wants.
        const std::vector<PickFileFilterEntry> entries = PickFileFilter(kind);
        std::vector<wchar_t> filter;
        for (const PickFileFilterEntry& e : entries) {
            const std::wstring label = Utf8ToWide(e.label);
            const std::wstring pattern = Utf8ToWide(e.pattern);
            filter.insert(filter.end(), label.begin(), label.end());
            filter.push_back(L'\0');
            filter.insert(filter.end(), pattern.begin(), pattern.end());
            filter.push_back(L'\0');
        }
        filter.push_back(L'\0');  // final terminator for the filter list

        std::vector<wchar_t> file_buf(4096, L'\0');
        OPENFILENAMEW ofn = {};
        ofn.lStructSize = sizeof(ofn);
        ofn.hwndOwner = static_cast<HWND>(plugin_hwnd_);
        ofn.lpstrFilter = filter.data();
        ofn.nFilterIndex = 1;
        ofn.lpstrFile = file_buf.data();
        ofn.nMaxFile = static_cast<DWORD>(file_buf.size());
        ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST |
                    OFN_NOCHANGEDIR;

        LogInfo(std::wstring(L"ui.pickFile: opening Open dialog (kind ") +
                Utf8ToWide(kind) + L")");
        const BOOL picked = ::GetOpenFileNameW(&ofn);
        if (!picked) {
            const DWORD ext = ::CommDlgExtendedError();
            if (ext == 0) {
                LogInfo(L"ui.pickFile: cancelled by user");
                return MakeErrorResponse(id, "CANCELLED", "File selection cancelled");
            }
            LogWarn(std::wstring(L"ui.pickFile: dialog failed, CommDlgExtendedError=") +
                    std::to_wstring(ext));
            return MakeErrorResponse(id, "DIALOG_FAILED", "Open dialog failed");
        }
        const std::string path_utf8 = WideToUtf8(std::wstring(file_buf.data()));
        json result;
        result["filePath"] = path_utf8;
        result["fileName"] = FileNameFromPath(path_utf8);
        LogInfo(std::wstring(L"ui.pickFile: selected ") + Utf8ToWide(path_utf8));
        return MakeSuccessResponse(id, std::move(result));
    }

    // --- ui.pickFolder (modal IFileOpenDialog + FOS_PICKFOLDERS, UI thread,
    //     contract v6) ----------------------------------------------------------
    if (method == "ui.pickFolder") {
        const json id = ExtractId(req);
        const json params =
            (req.contains("params") && req["params"].is_object()) ? req["params"]
                                                                   : json::object();
        PickFolderRequest preq;
        std::string err;
        if (!ParsePickFolder(params, &preq, &err)) {
            return MakeErrorResponse(id, "BAD_REQUEST", err);
        }
        // Reuses ui.pickFile's re-entrancy guard: the dialog's modal pump can
        // deliver another web message on this same UI thread.
        bool expected = false;
        if (!pick_dialog_open_.compare_exchange_strong(expected, true)) {
            LogWarn(L"ui.pickFolder: rejected re-entrant request (dialog already open)");
            return MakeErrorResponse(id, "DIALOG_FAILED", "dialog already open");
        }
        struct OpenGuard {
            std::atomic<bool>& flag;
            ~OpenGuard() { flag.store(false); }
        } open_guard{pick_dialog_open_};

        ComPtr<IFileOpenDialog> dialog;
        HRESULT hr = ::CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                        IID_PPV_ARGS(dialog.put()));
        if (FAILED(hr) || !dialog) {
            LogWarn(L"ui.pickFolder: CoCreateInstance(FileOpenDialog) failed");
            return MakeErrorResponse(id, "DIALOG_FAILED", "Could not create folder dialog");
        }
        DWORD opts = 0;
        dialog->GetOptions(&opts);
        dialog->SetOptions(opts | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);
        std::wstring title_w;
        if (preq.has_title) {
            title_w = Utf8ToWide(preq.title);
            dialog->SetTitle(title_w.c_str());
        }

        LogInfo(L"ui.pickFolder: opening folder dialog");
        hr = dialog->Show(static_cast<HWND>(plugin_hwnd_));
        if (hr == HRESULT_FROM_WIN32(ERROR_CANCELLED)) {
            LogInfo(L"ui.pickFolder: cancelled by user");
            return MakeErrorResponse(id, "CANCELLED", "Folder selection cancelled");
        }
        if (FAILED(hr)) {
            LogWarn(L"ui.pickFolder: Show failed");
            return MakeErrorResponse(id, "DIALOG_FAILED", "Folder dialog failed");
        }
        ComPtr<IShellItem> item;
        hr = dialog->GetResult(item.put());
        if (FAILED(hr) || !item) {
            LogWarn(L"ui.pickFolder: GetResult failed");
            return MakeErrorResponse(id, "DIALOG_FAILED", "Could not read dialog result");
        }
        PWSTR path_w = nullptr;
        hr = item->GetDisplayName(SIGDN_FILESYSPATH, &path_w);
        std::string path_utf8;
        if (SUCCEEDED(hr) && path_w != nullptr) {
            path_utf8 = WideToUtf8(path_w);
        }
        if (path_w != nullptr) {
            ::CoTaskMemFree(path_w);
        }
        if (path_utf8.empty()) {
            LogWarn(L"ui.pickFolder: dialog returned no filesystem path");
            return MakeErrorResponse(id, "DIALOG_FAILED", "Folder dialog returned no path");
        }
        json result;
        result["folderPath"] = path_utf8;
        LogInfo(std::wstring(L"ui.pickFolder: selected ") + Utf8ToWide(path_utf8));
        return MakeSuccessResponse(id, std::move(result));
    }

    // --- fs.listFiles (async folder enumeration, contract v6) ----------------
    if (method == "fs.listFiles") {
        const json id = ExtractId(req);
        const json params =
            (req.contains("params") && req["params"].is_object()) ? req["params"]
                                                                   : json::object();
        if (http_ == nullptr || !poster_) {
            return MakeErrorResponse(id, "BACKEND_UNREACHABLE",
                                     "HTTP worker pool is not initialized");
        }
        ListFilesRequest lreq;
        std::string err;
        if (!ParseListFiles(params, &lreq, &err)) {
            return MakeErrorResponse(id, "BAD_REQUEST", err);
        }
        ResponsePoster poster = poster_;
        http_->Post([lreq, id, poster]() { ListFilesWorker(lreq, id, poster); });
        return std::string();  // response delivered asynchronously
    }

    // --- fs.probeAudioDuration (async wav header probe, contract v6) ---------
    if (method == "fs.probeAudioDuration") {
        const json id = ExtractId(req);
        const json params =
            (req.contains("params") && req["params"].is_object()) ? req["params"]
                                                                   : json::object();
        if (http_ == nullptr || !poster_) {
            return MakeErrorResponse(id, "BACKEND_UNREACHABLE",
                                     "HTTP worker pool is not initialized");
        }
        ProbeAudioDurationRequest preq;
        std::string err;
        if (!ParseProbeAudioDuration(params, &preq, &err)) {
            return MakeErrorResponse(id, "BAD_REQUEST", err);
        }
        ResponsePoster poster = poster_;
        http_->Post([preq, id, poster]() { ProbeAudioDurationWorker(preq, id, poster); });
        return std::string();  // response delivered asynchronously
    }

    // --- ui.makeThumbnail (async WIC decode -> scale -> JPEG -> data URL) -----
    if (method == "ui.makeThumbnail") {
        const json id = ExtractId(req);
        const json params =
            (req.contains("params") && req["params"].is_object()) ? req["params"]
                                                                   : json::object();
        if (http_ == nullptr || !poster_) {
            return MakeErrorResponse(id, "THUMBNAIL_FAILED",
                                     "HTTP worker pool is not initialized");
        }
        MakeThumbnailRequest treq;
        std::string err;
        if (!ParseMakeThumbnail(params, &treq, &err)) {
            return MakeErrorResponse(id, "BAD_REQUEST", err);
        }
        if (!FileExists(treq.file_path)) {
            LogWarn(std::wstring(L"ui.makeThumbnail: FILE_NOT_FOUND ") +
                    Utf8ToWide(treq.file_path));
            return MakeErrorResponse(id, "FILE_NOT_FOUND",
                                     "Thumbnail source not found: " + treq.file_path);
        }
        ResponsePoster poster = poster_;
        http_->Post([treq, id, poster]() { MakeThumbnailWorker(treq, id, poster); });
        return std::string();  // response delivered asynchronously
    }

    // --- timeline.cutoutRange (async render+mux of a frame range -> mp4) ------
    if (method == "timeline.cutoutRange") {
        const json id = ExtractId(req);
        const json params =
            (req.contains("params") && req["params"].is_object()) ? req["params"]
                                                                   : json::object();
        if (http_ == nullptr || !poster_) {
            return MakeErrorResponse(id, "BACKEND_UNREACHABLE",
                                     "HTTP worker pool is not initialized");
        }
        CutoutRangeRequest creq;
        std::string err;
        if (!ParseCutoutRange(params, &creq, &err)) {
            return MakeErrorResponse(id, "BAD_REQUEST", err);
        }
        EDIT_HANDLE* handle = edit_handle_;
        if (handle == nullptr || handle->rendering_scene_video == nullptr ||
            handle->get_edit_info == nullptr) {
            return MakeErrorResponse(id, "NO_EDIT_HANDLE",
                                     "Edit handle is not available");
        }
        // Resolve rate/scale/resolution on the UI thread (get_edit_info takes a
        // reference lock; we are not under an edit lock here, so this is safe).
        EDIT_INFO info = {};
        handle->get_edit_info(&info, sizeof(info));
        if (info.width <= 0 || info.height <= 0) {
            return MakeErrorResponse(id, "CUTOUT_FAILED",
                                     "Scene has no valid resolution");
        }
        const std::wstring dir = AppDataDir() + L"\\cutouts";
        const std::string name = "cutout_" + std::to_string(creq.frame_start) + "_" +
                                 std::to_string(creq.frame_count) + "_" + UniqueStamp() +
                                 ".mp4";
        const std::string dest = WideToUtf8(dir + L"\\" + Utf8ToWide(name));
        const int w = info.width;
        const int h = info.height;
        const int rate = info.rate;
        const int scale = info.scale;
        const int sr = info.sample_rate;
        const int lm = info.layer_max;
        ResponsePoster poster = poster_;
        http_->Post([handle, creq, w, h, rate, scale, sr, lm, dest, id, poster]() {
            CutoutRangeWorker(handle, creq, w, h, rate, scale, sr, lm, dest, id, poster);
        });
        return std::string();  // response delivered asynchronously
    }

    // --- timeline.extractAudio (async render of a range's audio -> wav) -------
    if (method == "timeline.extractAudio") {
        const json id = ExtractId(req);
        const json params =
            (req.contains("params") && req["params"].is_object()) ? req["params"]
                                                                   : json::object();
        if (http_ == nullptr || !poster_) {
            return MakeErrorResponse(id, "BACKEND_UNREACHABLE",
                                     "HTTP worker pool is not initialized");
        }
        ExtractAudioRequest ereq;
        std::string err;
        if (!ParseExtractAudio(params, &ereq, &err)) {
            return MakeErrorResponse(id, "BAD_REQUEST", err);
        }
        EDIT_HANDLE* handle = edit_handle_;
        if (handle == nullptr || handle->rendering_scene_audio == nullptr ||
            handle->get_edit_info == nullptr) {
            return MakeErrorResponse(id, "NO_EDIT_HANDLE",
                                     "Edit handle is not available");
        }
        EDIT_INFO info = {};
        handle->get_edit_info(&info, sizeof(info));
        const std::wstring dir = AppDataDir() + L"\\audio";
        ::CreateDirectoryW(AppDataDir().c_str(), nullptr);
        ::CreateDirectoryW(dir.c_str(), nullptr);
        const std::string name = "audio_" + std::to_string(ereq.frame_start) + "_" +
                                 std::to_string(ereq.frame_count) + "_" + UniqueStamp() +
                                 ".wav";
        const std::string dest = WideToUtf8(dir + L"\\" + Utf8ToWide(name));
        const int sr = info.sample_rate;
        const int lm = info.layer_max;
        ResponsePoster poster = poster_;
        http_->Post([handle, ereq, sr, lm, dest, id, poster]() {
            ExtractAudioWorker(handle, ereq, sr, lm, dest, id, poster);
        });
        return std::string();  // response delivered asynchronously
    }

    // --- timeline.trackObject (async object tracking, section 3-54) ----------
    if (method == "timeline.trackObject") {
        const json id = ExtractId(req);
        const json params =
            (req.contains("params") && req["params"].is_object()) ? req["params"]
                                                                   : json::object();
        if (http_ == nullptr || !poster_) {
            return MakeErrorResponse(id, "BACKEND_UNREACHABLE",
                                     "HTTP worker pool is not initialized");
        }
        TrackObjectRequest treq;
        std::string err;
        if (!ParseTrackObject(params, &treq, &err)) {
            return MakeErrorResponse(id, "BAD_REQUEST", err);
        }
        EDIT_HANDLE* handle = edit_handle_;
        if (handle == nullptr || handle->rendering_scene_video == nullptr ||
            handle->call_edit_section_param == nullptr) {
            return MakeErrorResponse(id, "NO_EDIT_HANDLE",
                                     "Edit handle is not available");
        }
        // Claim the single tracking slot LAST, so every validation failure above
        // returns without having to hand the flag back. From here on the flag is
        // owned by TrackObjectWorker's TrackRunningGuard.
        if (g_track_running.exchange(true)) {
            return MakeErrorResponse(id, "TRACK_BUSY",
                                     "A tracking run is already in progress");
        }
        g_track_cancel.store(false);
        HttpClient* http = http_.get();
        const std::string base = CurrentBaseUrl();
        ResponsePoster poster = poster_;
        http_->Post([handle, http, treq, base, id, poster]() {
            TrackObjectWorker(handle, http, treq, base, id, poster);
        });
        return std::string();  // response delivered asynchronously
    }

    // --- timeline.cancelTracking (synchronous flag raise, section 5.4) -------
    // Raising the flag is all this does: the worker checks it at the top of each
    // loop iteration and then finishes the run normally - post-processing and
    // writing back whatever it collected (design 3.2), so a stop is a short
    // result, not a discarded one. "cancelled": false means there was nothing
    // running to stop, which the webui treats as success, not as an error.
    if (method == "timeline.cancelTracking") {
        const json id = ExtractId(req);
        if (!g_track_running.load()) {
            return MakeSuccessResponse(id, json{{"cancelled", false}});
        }
        g_track_cancel.store(true);
        LogInfo(L"timeline.cancelTracking: stop requested");
        return MakeSuccessResponse(id, json{{"cancelled", true}});
    }

    // --- Synchronous methods handled by the pure core ------------------------
    EDIT_HANDLE* handle = edit_handle_;
    RequestContext ctx;
    ctx.base_url = CurrentBaseUrl();
    SettingsStore* settings = settings_.get();
    ctx.settings_get = [settings]() -> std::string {
        return settings ? settings->GetBaseUrl() : std::string(kBackendBaseUrl);
    };
    ctx.settings_set = [settings](const std::string& candidate) -> SettingsSetOutcome {
        SettingsSetOutcome outcome;
        if (settings == nullptr) {
            outcome.ok = false;
            outcome.err_message = "settings store is not available";
            return outcome;
        }
        std::string normalized;
        std::string err;
        outcome.ok = settings->SetBaseUrl(candidate, &normalized, &err);
        if (outcome.ok) {
            outcome.base_url = normalized;
            LogInfo(std::wstring(L"settings.set: baseUrl -> ") + Utf8ToWide(normalized));
        } else {
            outcome.err_message = err;
            LogWarn(std::wstring(L"settings.set: rejected candidate (") +
                    Utf8ToWide(err) + L")");
        }
        return outcome;
    };
    ctx.edit_info = [handle]() -> EditInfoResult {
        EditInfoResult result;
        if (handle == nullptr || handle->get_edit_info == nullptr) {
            result.available = false;
            return result;
        }
        EDIT_INFO info = {};
        handle->get_edit_info(&info, sizeof(info));
        result.available = true;
        result.width = info.width;
        result.height = info.height;
        result.rate = info.rate;
        result.scale = info.scale;
        result.sample_rate = info.sample_rate;
        result.frame = info.frame;
        result.layer = info.layer;
        result.frame_max = info.frame_max;
        result.layer_max = info.layer_max;
        return result;
    };
    ctx.insert_media = [handle](const InsertMediaParams& p) -> InsertMediaResult {
        InsertMediaResult r;
        if (!FileExists(p.file_path)) {
            r.status = InsertMediaResult::Status::kFileNotFound;
            return r;
        }
        if (handle == nullptr || handle->call_edit_section_param == nullptr) {
            r.status = InsertMediaResult::Status::kNoEditHandle;
            return r;
        }
        InsertContext ic;
        ic.in = &p;
        const bool ok = handle->call_edit_section_param(&ic, &InsertMediaEditProc);
        if (!ok || !ic.ran) {
            LogWarn(L"timeline.insertMedia: call_edit_section_param did not run");
            r.status = InsertMediaResult::Status::kNoEditHandle;
            return r;
        }
        if (ic.object == nullptr) {
            // Every create attempt failed - the alias path AND the legacy
            // create_object_from_media_file fallback (see CreateMediaObject).
            LogWarn(std::wstring(L"timeline.insertMedia: media object creation failed "
                                 L"(layer ") +
                    std::to_wstring(ic.layer) + L", frame " + std::to_wstring(ic.frame) +
                    L")");
            r.status = InsertMediaResult::Status::kInsertFailed;
            return r;
        }
        LogInfo(std::wstring(L"timeline.insertMedia: inserted at layer ") +
                std::to_wstring(ic.layer) + L", frame " + std::to_wstring(ic.frame));
        r.status = InsertMediaResult::Status::kOk;
        r.layer = ic.layer;
        r.frame = ic.frame;
        return r;
    };

    // --- Contract v5 timeline providers (appended; existing fields above are
    // unchanged). Each runs its SDK work inside a call_edit_section_param
    // callback on this (UI) thread. See the callbacks defined at file scope.
    ctx.get_selection = [handle]() -> SelectionSnapshot {
        SelectionSnapshot snap;
        if (handle == nullptr || handle->call_edit_section_param == nullptr) {
            snap.available = false;
            return snap;
        }
        const bool ran = handle->call_edit_section_param(&snap, &GetSelectionEditProc);
        if (!ran) {
            snap.available = false;
            LogWarn(L"timeline.getSelection: call_edit_section_param did not run");
        }
        return snap;
    };
    // fs.probeMediaInfo: best-effort media duration / resolution probe. Any
    // failure (no edit handle, no call_edit_section_param, callback didn't run,
    // or get_media_info returned false) resolves an unavailable snapshot, which
    // bridge_core reports as the all-zero success result - this method never
    // errors except on bad params.
    ctx.probe_media_info = [handle](const std::string& path) -> MediaInfoSnapshot {
        MediaInfoSnapshot out;
        if (handle == nullptr || handle->call_edit_section_param == nullptr) {
            return out;
        }
        ProbeMediaInfoCtx c;
        c.file_path = path;
        const bool ran = handle->call_edit_section_param(&c, &ProbeMediaInfoEditProc);
        if (ran && c.ok) {
            out.available = true;
            out.duration_sec = c.duration_sec;
            out.width = c.width;
            out.height = c.height;
        }
        return out;
    };
    ctx.insert_provisional = [handle](const std::string& alias,
                                      const std::string& object_name, int layer,
                                      int frame, int length) -> InsertProvisionalOutcome {
        InsertProvisionalOutcome out;
        if (handle == nullptr || handle->call_edit_section_param == nullptr) {
            return out;
        }
        // I3: read layer_max up front so the edit callback can fall back to
        // layer_max+1 when the requested slot is occupied.
        int layer_max = 0;
        if (handle->get_edit_info != nullptr) {
            EDIT_INFO info = {};
            handle->get_edit_info(&info, sizeof(info));
            layer_max = info.layer_max;
        }
        const std::wstring name_w = Utf8ToWide(object_name);
        InsertProvisionalCtx c;
        c.alias = alias.c_str();
        c.name_w = name_w.c_str();
        c.layer = layer;
        c.frame = frame;
        c.length = length;
        c.layer_max = layer_max;
        c.ok = false;
        c.out_layer = layer;
        c.out_frame = frame;
        c.used_fallback = false;
        const bool ran = handle->call_edit_section_param(&c, &InsertProvisionalEditProc);
        if (ran && c.ok) {
            out.ok = true;
            out.layer = c.out_layer;
            out.frame = c.out_frame;
            out.used_fallback = c.used_fallback;
        } else {
            LogWarn(L"timeline.insertProvisional: create_object_from_alias failed");
        }
        return out;
    };
    ctx.scan_objects = [handle]() -> std::vector<ScannedObject> {
        std::vector<ScannedObject> objs;
        if (handle == nullptr || handle->call_edit_section_param == nullptr) {
            return objs;
        }
        int layer_max = 0;
        int frame_max = 0;
        if (handle->get_edit_info != nullptr) {
            EDIT_INFO info = {};
            handle->get_edit_info(&info, sizeof(info));
            layer_max = info.layer_max;
            frame_max = info.frame_max;
        }
        ScanObjectsCtx c{&objs, layer_max, frame_max};
        handle->call_edit_section_param(&c, &ScanObjectsEditProc);
        return objs;
    };
    ctx.replace_object = [handle](const ReplaceObjectRequest& rr) -> bool {
        if (!FileExists(rr.video_file_path)) {
            LogWarn(std::wstring(L"timeline.resolveProvisional: video not found: ") +
                    Utf8ToWide(rr.video_file_path));
            return false;
        }
        if (handle == nullptr || handle->call_edit_section_param == nullptr) {
            return false;
        }
        int frame_max = 0;
        if (handle->get_edit_info != nullptr) {
            EDIT_INFO info = {};
            handle->get_edit_info(&info, sizeof(info));
            frame_max = info.frame_max;
        }
        ReplaceObjectCtx c{&rr, frame_max, false};
        const bool ran = handle->call_edit_section_param(&c, &ReplaceObjectEditProc);
        return ran && c.ok;
    };
    ctx.update_object_text = [handle](const UpdateObjectTextRequest& ur) -> bool {
        if (handle == nullptr || handle->call_edit_section_param == nullptr) {
            return false;
        }
        int frame_max = 0;
        if (handle->get_edit_info != nullptr) {
            EDIT_INFO info = {};
            handle->get_edit_info(&info, sizeof(info));
            frame_max = info.frame_max;
        }
        const std::wstring effect_w = Utf8ToWide(kEffectTextJp);
        const std::wstring item_w = Utf8ToWide(kEffectTextJp);
        UpdateTextCtx c{&ur, effect_w.c_str(), item_w.c_str(), frame_max, false};
        const bool ran = handle->call_edit_section_param(&c, &UpdateObjectTextEditProc);
        return ran && c.ok;
    };
    // --- I3: timeline.updateProvisionalReservation (atomic delete+create) ----
    ctx.update_reservation =
        [handle](const UpdateReservationRequest& r) -> UpdateReservationOutcome {
        UpdateReservationOutcome out;
        if (handle == nullptr || handle->call_edit_section_param == nullptr) {
            return out;
        }
        int layer_max = 0;
        int frame_max = 0;
        if (handle->get_edit_info != nullptr) {
            EDIT_INFO info = {};
            handle->get_edit_info(&info, sizeof(info));
            layer_max = info.layer_max;
            frame_max = info.frame_max;
        }
        const std::wstring name_w = Utf8ToWide(r.object_name);
        UpdateReservationCtx c;
        c.req = &r;
        c.name_w = name_w.c_str();
        c.layer_max = layer_max;
        c.frame_max = frame_max;
        c.ok = false;
        c.deleted_old = false;
        c.out_layer = r.layer;
        c.out_frame = r.frame;
        c.used_fallback = false;
        const bool ran =
            handle->call_edit_section_param(&c, &UpdateProvisionalReservationEditProc);
        if (ran && c.ok) {
            out.ok = true;
            out.deleted_old = c.deleted_old;
            out.placed_layer = c.out_layer;
            out.placed_frame = c.out_frame;
            out.used_fallback = c.used_fallback;
        } else {
            LogWarn(L"timeline.updateProvisionalReservation: create/delete failed");
        }
        return out;
    };
    // --- I13: timeline.deleteProvisionalByJob (✅ marker cleanup on 🎞 insert) -
    ctx.delete_provisional = [handle](const DeleteProvisionalRequest& r) -> bool {
        if (handle == nullptr || handle->call_edit_section_param == nullptr) {
            return false;
        }
        int frame_max = 0;
        if (handle->get_edit_info != nullptr) {
            EDIT_INFO info = {};
            handle->get_edit_info(&info, sizeof(info));
            frame_max = info.frame_max;
        }
        DeleteProvisionalCtx c{&r, frame_max, false};
        const bool ran =
            handle->call_edit_section_param(&c, &DeleteProvisionalByJobEditProc);
        return ran && c.deleted;
    };
    // --- Replace-insert (place-and-replace 🎞): timeline.insertMediaForJob's
    // "found a marker" branch. The "not found" branch reuses ctx.insert_media.
    ctx.replace_media_for_job =
        [handle](const ReplaceMediaForJobRequest& r) -> ReplaceMediaForJobOutcome {
        ReplaceMediaForJobOutcome out;
        if (!FileExists(r.file_path)) {
            LogWarn(std::wstring(L"timeline.insertMediaForJob: media not found: ") +
                    Utf8ToWide(r.file_path));
            return out;  // ok == false -> INSERT_FAILED
        }
        if (handle == nullptr || handle->call_edit_section_param == nullptr) {
            return out;
        }
        int layer_max = 0;
        int frame_max = 0;
        if (handle->get_edit_info != nullptr) {
            EDIT_INFO info = {};
            handle->get_edit_info(&info, sizeof(info));
            layer_max = info.layer_max;
            frame_max = info.frame_max;
        }
        ReplaceMediaForJobCtx c{&r, layer_max, frame_max, false, false, r.layer, r.frame};
        const bool ran =
            handle->call_edit_section_param(&c, &ReplaceMediaForJobEditProc);
        if (ran && c.ok) {
            out.ok = true;
            out.layer = c.out_layer;
            out.frame = c.out_frame;
            out.used_fallback = c.used_fallback;
        } else {
            LogWarn(L"timeline.insertMediaForJob: replace-insert failed");
        }
        return out;
    };

    return HandleRequestJson(request_json, ctx);
}

}  // namespace nzvideomni
