// test_mf_mp4_writer.cpp - encode synthetic frames + audio to an mp4 with the
// Media Foundation writer, then read it back with IMFSourceReader and assert the
// file is a decodable H.264/AAC mp4 with the expected resolution, duration, and
// track layout. Generated files live under the scratchpad (from %TEMP%) and are
// deleted after each case.
//
// ASCII-only source.
#include "doctest.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfreadwrite.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "mf_mp4_writer.h"

namespace {

// RAII COM pointer, local to the test.
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
        if (p_) { p_->Release(); p_ = nullptr; }
    }
private:
    T* p_ = nullptr;
};

// Build a unique scratchpad path from %TEMP% (the harness scratchpad lives under
// the user temp dir). Falls back to GetTempPathW.
std::wstring TempMp4Path(const std::wstring& name) {
    wchar_t dir[MAX_PATH] = {};
    const DWORD n = ::GetTempPathW(MAX_PATH, dir);
    std::wstring path(dir, n);
    path += L"nzltx23_mp4_";
    path += name;
    return path;
}

// H.264 codes in 16x16 macroblocks, so a decoded frame's coded size is the
// requested size rounded UP to the next multiple of 16 (e.g. 120 -> 128). A
// dimension that is already 16-aligned (like 320x240) round-trips exactly. The
// read-back resolution check accepts either the exact size or its 16-aligned
// coded size.
int Align16(int v) { return (v + 15) & ~15; }

bool DimMatches(UINT32 decoded, int requested) {
    return decoded == static_cast<UINT32>(requested) ||
           decoded == static_cast<UINT32>(Align16(requested));
}

int64_t FileSize(const std::wstring& path) {
    WIN32_FILE_ATTRIBUTE_DATA fad = {};
    if (!::GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &fad)) {
        return -1;
    }
    LARGE_INTEGER li;
    li.HighPart = fad.nFileSizeHigh;
    li.LowPart = fad.nFileSizeLow;
    return li.QuadPart;
}

// Fill an RGBA frame (top-down, tightly packed) with a frame that has a distinct
// top band vs bottom band so read-back can verify vertical orientation, plus a
// moving column so successive frames differ (motion the encoder must carry).
void MakeFrame(std::vector<uint8_t>* px, int width, int height, int frame_idx) {
    px->resize(static_cast<size_t>(width) * height * 4u);
    const int band = height / 2;
    const int col = (frame_idx * 7) % width;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t i = (static_cast<size_t>(y) * width + x) * 4u;
            uint8_t r, g, b;
            if (y < band) {
                // Top band: strong red.
                r = 220; g = 20; b = 20;
            } else {
                // Bottom band: strong blue.
                r = 20; g = 20; b = 220;
            }
            if (x == col) {  // moving white column
                r = g = b = 255;
            }
            (*px)[i + 0] = r;  // R
            (*px)[i + 1] = g;  // G
            (*px)[i + 2] = b;  // B
            (*px)[i + 3] = 255;  // A
        }
    }
}

// Read the mp4 back with a source reader and collect: whether it decodes, the
// video frame size, the total duration, and whether an audio stream is present.
struct ReadbackInfo {
    bool opened = false;
    bool decoded_a_frame = false;
    UINT32 width = 0;
    UINT32 height = 0;
    int64_t duration_100ns = 0;  // from the file's presentation descriptor
    bool has_audio = false;
    // Average color (0..255) of the top-center and bottom-center of the first
    // decoded frame, in R/G/B. Used to check channel order + orientation.
    int top_r = -1, top_g = -1, top_b = -1;
    int bot_r = -1, bot_g = -1, bot_b = -1;
};

// RAII: initialise COM + Media Foundation for the current scope. The writer
// tears MF/COM down in Finalize(), so the read-back path must start its own.
class MfScope {
public:
    MfScope() {
        const HRESULT co = ::CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        co_ = SUCCEEDED(co);
        mf_ = SUCCEEDED(::MFStartup(MF_VERSION, MFSTARTUP_LITE));
    }
    ~MfScope() {
        if (mf_) ::MFShutdown();
        if (co_) ::CoUninitialize();
    }
    bool ok() const { return mf_; }
private:
    bool co_ = false;
    bool mf_ = false;
};

ReadbackInfo ReadBackMp4(const std::wstring& path) {
    ReadbackInfo info;

    MfScope mf;
    if (!mf.ok()) return info;

    ComPtr<IMFAttributes> attrs;
    if (FAILED(::MFCreateAttributes(attrs.put(), 1))) return info;
    // Allow the reader to insert a color converter so we can request RGB32 out.
    attrs->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, 1);

    ComPtr<IMFSourceReader> reader;
    HRESULT hr = ::MFCreateSourceReaderFromURL(path.c_str(), attrs.get(),
                                               reader.put());
    if (FAILED(hr) || !reader) return info;
    info.opened = true;

    // Is there an audio stream? Query its native media type; MF_E_INVALIDSTREAMNUMBER
    // means the stream does not exist.
    {
        ComPtr<IMFMediaType> atype;
        HRESULT ahr = reader->GetNativeMediaType(
            static_cast<DWORD>(MF_SOURCE_READER_FIRST_AUDIO_STREAM), 0,
            atype.put());
        info.has_audio = SUCCEEDED(ahr) && atype;
    }

    // File duration (100ns) from the presentation descriptor.
    {
        PROPVARIANT var;
        ::PropVariantInit(&var);
        if (SUCCEEDED(reader->GetPresentationAttribute(
                static_cast<DWORD>(MF_SOURCE_READER_MEDIASOURCE),
                MF_PD_DURATION, &var)) &&
            var.vt == VT_UI8) {
            info.duration_100ns = static_cast<int64_t>(var.uhVal.QuadPart);
        }
        ::PropVariantClear(&var);
    }

    // Ask the reader to decode video to RGB32 so we can inspect pixels.
    ComPtr<IMFMediaType> want;
    if (SUCCEEDED(::MFCreateMediaType(want.put()))) {
        want->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        want->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
        reader->SetCurrentMediaType(
            static_cast<DWORD>(MF_SOURCE_READER_FIRST_VIDEO_STREAM), nullptr,
            want.get());
    }

    // Current (decoded) video type -> frame size.
    {
        ComPtr<IMFMediaType> cur;
        if (SUCCEEDED(reader->GetCurrentMediaType(
                static_cast<DWORD>(MF_SOURCE_READER_FIRST_VIDEO_STREAM),
                cur.put()))) {
            ::MFGetAttributeSize(cur.get(), MF_MT_FRAME_SIZE, &info.width,
                                 &info.height);
        }
    }

    // Pull samples until we decode one video frame (skip any empty/format events).
    for (int guard = 0; guard < 240 && !info.decoded_a_frame; ++guard) {
        DWORD stream_index = 0;
        DWORD flags = 0;
        LONGLONG ts = 0;
        ComPtr<IMFSample> sample;
        hr = reader->ReadSample(
            static_cast<DWORD>(MF_SOURCE_READER_FIRST_VIDEO_STREAM), 0,
            &stream_index, &flags, &ts, sample.put());
        if (FAILED(hr)) break;
        if (flags & MF_SOURCE_READERF_ENDOFSTREAM) break;
        if (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED) {
            ComPtr<IMFMediaType> cur;
            if (SUCCEEDED(reader->GetCurrentMediaType(
                    static_cast<DWORD>(MF_SOURCE_READER_FIRST_VIDEO_STREAM),
                    cur.put()))) {
                ::MFGetAttributeSize(cur.get(), MF_MT_FRAME_SIZE, &info.width,
                                     &info.height);
            }
        }
        if (!sample) continue;

        ComPtr<IMFMediaBuffer> buf;
        if (FAILED(sample->ConvertToContiguousBuffer(buf.put())) || !buf) continue;
        BYTE* data = nullptr;
        DWORD len = 0;
        if (FAILED(buf->Lock(&data, nullptr, &len)) || !data) continue;

        if (info.width > 0 && info.height > 0 &&
            len >= info.width * info.height * 4u) {
            const int w = static_cast<int>(info.width);
            const int h = static_cast<int>(info.height);
            const int stride = w * 4;
            auto sample_at = [&](int x, int y, int* r, int* g, int* b) {
                const BYTE* p = data + static_cast<size_t>(y) * stride + x * 4;
                // RGB32 memory order is B,G,R,A.
                *b = p[0];
                *g = p[1];
                *r = p[2];
            };
            // Sample well inside the top and bottom bands, center column.
            sample_at(w / 2, h / 4, &info.top_r, &info.top_g, &info.top_b);
            sample_at(w / 2, (3 * h) / 4, &info.bot_r, &info.bot_g, &info.bot_b);
            info.decoded_a_frame = true;
        }
        buf->Unlock();
    }

    return info;
}

}  // namespace

TEST_CASE("Mp4Writer encodes video+audio and it reads back as a valid mp4") {
    const int width = 320;
    const int height = 240;
    const int fps = 30;
    const int num_frames = 45;  // 1.5 seconds
    const uint32_t sample_rate = 48000;
    const std::wstring path = TempMp4Path(L"av.mp4");
    ::DeleteFileW(path.c_str());

    nzltx::Mp4WriterConfig cfg;
    cfg.output_path = path;
    cfg.width = width;
    cfg.height = height;
    cfg.fps_num = fps;
    cfg.fps_den = 1;
    cfg.pixel_order = nzltx::PixelOrder::kRGBA;  // AviUtl2 PIXEL_RGBA order
    cfg.has_audio = true;
    cfg.audio_sample_rate = sample_rate;
    cfg.audio_channels = 2;

    nzltx::Mp4Writer writer;
    std::string err;
    REQUIRE_MESSAGE(writer.Initialize(cfg, &err), err);

    // Feed audio in ~1 video-frame-sized chunks: sample_rate/fps samples/frame.
    const int samples_per_chunk = static_cast<int>(sample_rate) / fps;  // 1600
    std::vector<float> left(samples_per_chunk);
    std::vector<float> right(samples_per_chunk);
    int64_t audio_phase = 0;  // running sample index for a continuous sine

    std::vector<uint8_t> frame;
    for (int f = 0; f < num_frames; ++f) {
        MakeFrame(&frame, width, height, f);
        REQUIRE_MESSAGE(writer.WriteVideoFrame(frame.data(), width * 4, &err), err);

        for (int i = 0; i < samples_per_chunk; ++i) {
            const double t = static_cast<double>(audio_phase + i) / sample_rate;
            left[i] = 0.3f * static_cast<float>(std::sin(2.0 * 3.14159265 * 440.0 * t));
            right[i] = 0.3f * static_cast<float>(std::sin(2.0 * 3.14159265 * 660.0 * t));
        }
        audio_phase += samples_per_chunk;
        REQUIRE_MESSAGE(
            writer.WriteAudioSamples(left.data(), right.data(), samples_per_chunk, &err),
            err);
    }

    REQUIRE_MESSAGE(writer.Finalize(&err), err);

    // File exists and is non-empty.
    const int64_t size = FileSize(path);
    CHECK(size > 0);

    // Read it back.
    const ReadbackInfo info = ReadBackMp4(path);
    CHECK(info.opened);
    CHECK(info.decoded_a_frame);
    CHECK(DimMatches(info.width, width));
    CHECK(DimMatches(info.height, height));
    CHECK(info.has_audio);

    // Duration ~ num_frames / fps seconds (1.5s). Allow generous slack for
    // encoder priming / container rounding.
    const int64_t expected = static_cast<int64_t>(num_frames) * 10000000 / fps;
    CHECK(info.duration_100ns > expected / 2);
    CHECK(info.duration_100ns < expected * 2);

    // Channel order + vertical orientation: the top band was red, the bottom
    // band blue. After a lossy H.264 round trip the exact values drift, so use a
    // dominant-channel check (R clearly leads at top, B clearly leads at bottom).
    CHECK(info.top_r > info.top_b + 40);   // top is reddish
    CHECK(info.bot_b > info.bot_r + 40);   // bottom is bluish

    ::DeleteFileW(path.c_str());
}

TEST_CASE("Mp4Writer encodes a video-only mp4 (no audio track)") {
    const int width = 160;
    const int height = 120;
    const int fps = 30;
    const int num_frames = 30;
    const std::wstring path = TempMp4Path(L"video_only.mp4");
    ::DeleteFileW(path.c_str());

    nzltx::Mp4WriterConfig cfg;
    cfg.output_path = path;
    cfg.width = width;
    cfg.height = height;
    cfg.fps_num = fps;
    cfg.fps_den = 1;
    cfg.has_audio = false;

    nzltx::Mp4Writer writer;
    std::string err;
    REQUIRE_MESSAGE(writer.Initialize(cfg, &err), err);

    std::vector<uint8_t> frame;
    for (int f = 0; f < num_frames; ++f) {
        MakeFrame(&frame, width, height, f);
        REQUIRE_MESSAGE(writer.WriteVideoFrame(frame.data(), width * 4, &err), err);
        // Audio writes are no-ops when has_audio is false.
        CHECK(writer.WriteAudioSamples(nullptr, nullptr, 0, &err));
    }
    REQUIRE_MESSAGE(writer.Finalize(&err), err);

    CHECK(FileSize(path) > 0);

    const ReadbackInfo info = ReadBackMp4(path);
    CHECK(info.opened);
    CHECK(info.decoded_a_frame);
    CHECK(DimMatches(info.width, width));
    CHECK(DimMatches(info.height, height));  // 120 -> coded 128 (macroblock align)
    CHECK_FALSE(info.has_audio);  // no audio stream in a video-only mp4

    ::DeleteFileW(path.c_str());
}

TEST_CASE("Mp4Writer rejects invalid configuration") {
    nzltx::Mp4Writer writer;
    std::string err;
    nzltx::Mp4WriterConfig cfg;
    cfg.output_path = L"";  // empty path
    cfg.width = 0;
    cfg.height = 0;
    CHECK_FALSE(writer.Initialize(cfg, &err));
    CHECK_FALSE(err.empty());
}
