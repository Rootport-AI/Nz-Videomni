// test_video_thumbnail.cpp - MakeJpegThumbnailFromVideo (bridge contract v17:
// ui.makeThumbnail on a video returns its first frame).
//
// Each case writes a small mp4 to %TEMP% with Mp4Writer (top half red, bottom
// half blue), makes a thumbnail of it, decodes the JPEG back with WIC and checks
// the size, the vertical order of the colors and that the bottom rows are not
// decoder padding. The heights are deliberately NOT multiples of 16, so H.264
// decodes them taller than the clip (40 -> 48, 1080 -> 1088) and the display
// area crop is exercised. A text file saved as .mp4 and a missing file must
// fail. Files are deleted after each case.
//
// ASCII-only source.
#include "doctest.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <wincodec.h>

#include <cstdint>
#include <string>
#include <vector>

#include "mf_mp4_writer.h"
#include "wic_png.h"

namespace {

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

std::wstring TempVideoPath(const std::wstring& name) {
    wchar_t dir[MAX_PATH] = {};
    const DWORD n = ::GetTempPathW(MAX_PATH, dir);
    std::wstring path(dir, n);
    path += L"nzvideomni_vthumb_";
    path += name;
    return path;
}

// Write a short video-only mp4 whose top half is red and bottom half blue.
bool WriteTwoBandMp4(const std::wstring& path, int width, int height, int frames,
                     std::string* err) {
    nzvideomni::Mp4WriterConfig cfg;
    cfg.output_path = path;
    cfg.width = width;
    cfg.height = height;
    cfg.fps_num = 30;
    cfg.fps_den = 1;
    cfg.pixel_order = nzvideomni::PixelOrder::kRGBA;
    cfg.has_audio = false;

    std::vector<uint8_t> px(static_cast<size_t>(width) * height * 4u);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t* p = &px[(static_cast<size_t>(y) * width + x) * 4u];
            const bool top = y < height / 2;
            p[0] = top ? 220 : 20;   // R
            p[1] = 20;               // G
            p[2] = top ? 20 : 220;   // B
            p[3] = 255;              // A
        }
    }

    nzvideomni::Mp4Writer writer;
    if (!writer.Initialize(cfg, err)) return false;
    for (int f = 0; f < frames; ++f) {
        if (!writer.WriteVideoFrame(px.data(), width * 4, err)) return false;
    }
    return writer.Finalize(err);
}

// A decoded thumbnail: top-down BGRA, stride = width * 4.
struct Decoded {
    bool ok = false;
    int width = 0;
    int height = 0;
    std::vector<uint8_t> bgra;

    // Average R/G/B over rows [y0, y1) and the middle half of the columns.
    void Average(int y0, int y1, int* r, int* g, int* b) const {
        long long sr = 0, sg = 0, sb = 0, n = 0;
        for (int y = y0; y < y1; ++y) {
            for (int x = width / 4; x < (3 * width) / 4; ++x) {
                const uint8_t* p = &bgra[(static_cast<size_t>(y) * width + x) * 4u];
                sb += p[0];
                sg += p[1];
                sr += p[2];
                ++n;
            }
        }
        *r = static_cast<int>(sr / n);
        *g = static_cast<int>(sg / n);
        *b = static_cast<int>(sb / n);
    }
};

Decoded DecodeJpeg(const std::vector<unsigned char>& jpeg) {
    Decoded out;
    const HRESULT co = ::CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    do {
        ComPtr<IWICImagingFactory> factory;
        if (FAILED(::CoCreateInstance(CLSID_WICImagingFactory, nullptr,
                                      CLSCTX_INPROC_SERVER,
                                      IID_PPV_ARGS(factory.put())))) break;
        ComPtr<IWICStream> stream;
        if (FAILED(factory->CreateStream(stream.put()))) break;
        if (FAILED(stream->InitializeFromMemory(
                const_cast<BYTE*>(jpeg.data()), static_cast<DWORD>(jpeg.size())))) break;
        ComPtr<IWICBitmapDecoder> decoder;
        if (FAILED(factory->CreateDecoderFromStream(
                stream.get(), nullptr, WICDecodeMetadataCacheOnDemand,
                decoder.put()))) break;
        GUID container = {};
        if (FAILED(decoder->GetContainerFormat(&container)) ||
            container != GUID_ContainerFormatJpeg) break;
        ComPtr<IWICBitmapFrameDecode> frame;
        if (FAILED(decoder->GetFrame(0, frame.put()))) break;
        UINT w = 0, h = 0;
        if (FAILED(frame->GetSize(&w, &h)) || w == 0 || h == 0) break;
        ComPtr<IWICFormatConverter> conv;
        if (FAILED(factory->CreateFormatConverter(conv.put()))) break;
        if (FAILED(conv->Initialize(frame.get(), GUID_WICPixelFormat32bppBGRA,
                                    WICBitmapDitherTypeNone, nullptr, 0.0,
                                    WICBitmapPaletteTypeCustom))) break;
        out.width = static_cast<int>(w);
        out.height = static_cast<int>(h);
        out.bgra.resize(static_cast<size_t>(w) * h * 4u);
        if (FAILED(conv->CopyPixels(nullptr, w * 4u,
                                    static_cast<UINT>(out.bgra.size()),
                                    out.bgra.data()))) break;
        out.ok = true;
    } while (false);
    if (SUCCEEDED(co)) ::CoUninitialize();
    return out;
}

// Make the thumbnail of a freshly written two-band clip and check it.
void CheckTwoBandClip(const std::wstring& name, int width, int height, int max_dim,
                      int expect_w, int expect_h) {
    const std::wstring path = TempVideoPath(name);
    ::DeleteFileW(path.c_str());
    std::string err;
    REQUIRE_MESSAGE(WriteTwoBandMp4(path, width, height, 3, &err), err);

    std::vector<unsigned char> bytes;
    int tw = 0, th = 0, sw = 0, sh = 0;
    const bool ok = nzvideomni::MakeJpegThumbnailFromVideo(path, max_dim, &bytes, &tw,
                                                           &th, &sw, &sh, &err);
    ::DeleteFileW(path.c_str());
    INFO("thumbnail error: ", err);
    REQUIRE(ok);

    // The source size is the clip's own size, not the 16-aligned decoded size.
    CHECK(sw == width);
    CHECK(sh == height);
    CHECK(tw == expect_w);
    CHECK(th == expect_h);

    const Decoded d = DecodeJpeg(bytes);
    REQUIRE(d.ok);
    CHECK(d.width == expect_w);
    CHECK(d.height == expect_h);

    // Top quarter red, bottom quarter blue (JPEG + H.264 drift: dominant-channel
    // checks with a margin, not exact values).
    int r = 0, g = 0, b = 0;
    d.Average(0, d.height / 4, &r, &g, &b);
    INFO("top avg r=", r, " g=", g, " b=", b);
    CHECK(r > b + 80);
    CHECK(r > g + 80);
    d.Average((3 * d.height) / 4, d.height, &r, &g, &b);
    INFO("bottom avg r=", r, " g=", g, " b=", b);
    CHECK(b > r + 80);
    CHECK(b > g + 80);
    // The very last row is still the blue band, not decoder padding (which
    // would read as black or green).
    d.Average(d.height - 1, d.height, &r, &g, &b);
    INFO("last row avg r=", r, " g=", g, " b=", b);
    CHECK(b > 120);
    CHECK(b > r + 60);
    CHECK(b > g + 60);
}

}  // namespace

TEST_CASE("MakeJpegThumbnailFromVideo: first frame of a 64x40 mp4, no upscale") {
    // 40 lines decode as 48; the 8 padding rows must be cropped away.
    CheckTwoBandClip(L"64x40.mp4", 64, 40, 256, 64, 40);
}

TEST_CASE("MakeJpegThumbnailFromVideo: 1920x1080 mp4 is cropped and downscaled") {
    // 1080 lines decode as 1088 (the case the crop exists for).
    CheckTwoBandClip(L"1920x1080.mp4", 1920, 1080, 256, 256, 144);
}

TEST_CASE("MakeJpegThumbnailFromVideo fails cleanly on a non-video and a missing file") {
    std::vector<unsigned char> bytes;
    int tw = 0, th = 0, sw = 0, sh = 0;
    std::string err;

    const std::wstring fake = TempVideoPath(L"not_a_video.mp4");
    {
        HANDLE h = ::CreateFileW(fake.c_str(), GENERIC_WRITE, 0, nullptr,
                                 CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        REQUIRE(h != INVALID_HANDLE_VALUE);
        const char text[] = "this is plain text, not an mp4\r\n";
        DWORD written = 0;
        ::WriteFile(h, text, sizeof(text) - 1, &written, nullptr);
        ::CloseHandle(h);
    }
    CHECK_FALSE(nzvideomni::MakeJpegThumbnailFromVideo(fake, 256, &bytes, &tw, &th,
                                                       &sw, &sh, &err));
    CHECK_FALSE(err.empty());
    CHECK(bytes.empty());
    CHECK(tw == 0);
    CHECK(sw == 0);
    ::DeleteFileW(fake.c_str());

    err.clear();
    const std::wstring missing = TempVideoPath(L"does_not_exist.mp4");
    ::DeleteFileW(missing.c_str());
    CHECK_FALSE(nzvideomni::MakeJpegThumbnailFromVideo(missing, 256, &bytes, &tw, &th,
                                                       &sw, &sh, &err));
    CHECK_FALSE(err.empty());

    // Argument checks shared with MakeJpegThumbnail.
    CHECK_FALSE(nzvideomni::MakeJpegThumbnailFromVideo(missing, 0, &bytes, &tw, &th,
                                                       &sw, &sh, &err));
    CHECK_FALSE(nzvideomni::MakeJpegThumbnailFromVideo(missing, 256, nullptr, &tw,
                                                       &th, &sw, &sh, &err));
}
