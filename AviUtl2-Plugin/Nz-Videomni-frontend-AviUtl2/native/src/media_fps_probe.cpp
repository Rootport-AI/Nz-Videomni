// media_fps_probe.cpp - implementation. See media_fps_probe.h for the contract.
// ASCII-only source.
#include "media_fps_probe.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfreadwrite.h>

#include <mutex>

namespace nzvideomni {

namespace {

// Minimal RAII COM pointer (mirrors mf_mp4_writer.cpp / wic_png.cpp; keeps this
// TU free of wil so it builds in the plain test executable too).
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

// COM init for the duration of one probe. The caller is AviUtl2's UI thread, an
// STA, so COINIT_APARTMENTTHREADED is what we ask for; only a ref we actually
// took is released. S_OK / S_FALSE are both ref-counted (balance them);
// RPC_E_CHANGED_MODE means the thread already lives in the other apartment, so
// no ref was taken and none may be released - MF works either way.
class ComScope {
public:
    ComScope() {
        const HRESULT hr = ::CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
        needs_uninit_ = SUCCEEDED(hr);
        ok_ = needs_uninit_ || hr == RPC_E_CHANGED_MODE;
    }
    ~ComScope() {
        if (needs_uninit_) ::CoUninitialize();
    }
    ComScope(const ComScope&) = delete;
    ComScope& operator=(const ComScope&) = delete;
    bool ok() const { return ok_; }

private:
    bool ok_ = false;
    bool needs_uninit_ = false;
};

// MFStartup exactly once per process; the matching MFShutdown is deliberately
// never called (see the header for the two reasons).
std::once_flag g_mf_once;
bool g_mf_started = false;

// True when `path` ends in an extension the Windows Matroska source is
// registered for with video (.mkv / .webm / .mk3d), compared case-insensitively.
// The extension is taken with rfind(L'.') - not PathFindExtensionW, since shlwapi
// is not linked into the test executable. A dot inside a directory name is not
// mistaken for one: the dot must come after the last path separator.
bool HasMatroskaExtension(const std::wstring& path) {
    const std::wstring::size_type dot = path.rfind(L'.');
    if (dot == std::wstring::npos) return false;
    const std::wstring::size_type sep = path.find_last_of(L"\\/");
    if (sep != std::wstring::npos && sep > dot) return false;
    std::wstring ext = path.substr(dot);
    for (wchar_t& c : ext) {
        if (c >= L'A' && c <= L'Z') c = static_cast<wchar_t>(c - L'A' + L'a');
    }
    return ext == L".mkv" || ext == L".webm" || ext == L".mk3d";
}

}  // namespace

bool EnsureMediaFoundation() {
    std::call_once(g_mf_once, []() {
        g_mf_started = SUCCEEDED(::MFStartup(MF_VERSION, MFSTARTUP_LITE));
    });
    return g_mf_started;
}

double FpsFromRatio(std::uint32_t num, std::uint32_t den) {
    if (num == 0 || den == 0) {
        return 0.0;
    }
    return static_cast<double>(num) / static_cast<double>(den);
}

bool ProbeMediaFps(const std::wstring& path, double* fps_out, std::string* err) {
    auto fail = [err](const char* msg) -> bool {
        if (err != nullptr) *err = msg;
        return false;
    };

    if (fps_out == nullptr) {
        return fail("fps_out is null");
    }
    if (path.empty()) {
        return fail("empty path");
    }
    // Windows' Matroska source declares half the real rate (see the header), so
    // a Matroska file is "unknown" by rule and Media Foundation is not opened.
    if (HasMatroskaExtension(path)) {
        return fail("Matroska: Media Foundation's MKV source misreports the frame rate; treated as unknown");
    }

    ComScope com;
    if (!com.ok()) {
        return fail("CoInitializeEx failed");
    }
    if (!EnsureMediaFoundation()) {
        return fail("MFStartup failed");
    }

    // No attributes: only the container's declared type is read, so neither a
    // decoder nor the video processor needs to be instantiated.
    ComPtr<IMFSourceReader> reader;
    HRESULT hr = ::MFCreateSourceReaderFromURL(path.c_str(), nullptr, reader.put());
    if (FAILED(hr) || !reader) {
        // An EXPECTED failure: the file is missing / unreadable, or no installed
        // MF source handles its container. (Matroska never gets here.)
        return fail("MFCreateSourceReaderFromURL failed");
    }

    ComPtr<IMFMediaType> type;
    hr = reader->GetNativeMediaType(
        static_cast<DWORD>(MF_SOURCE_READER_FIRST_VIDEO_STREAM),
        static_cast<DWORD>(MF_SOURCE_READER_CURRENT_TYPE_INDEX), type.put());
    if (FAILED(hr) || !type) {
        return fail("no native video media type");  // audio-only file, etc.
    }

    UINT32 num = 0;
    UINT32 den = 0;
    hr = ::MFGetAttributeRatio(type.get(), MF_MT_FRAME_RATE, &num, &den);
    if (FAILED(hr)) {
        return fail("MF_MT_FRAME_RATE is not set");
    }

    const double fps = FpsFromRatio(num, den);
    if (fps <= 0.0) {
        return fail("frame rate ratio has a zero component");
    }
    *fps_out = fps;
    return true;
}

}  // namespace nzvideomni
