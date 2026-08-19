// test_wic_png.cpp - WIC PNG encoder unit test plus an end-to-end upload test.
//
// The encoder test runs anywhere (no backend needed). The upload test is in the
// "integration" suite and requires the mock LTX23 backend at
// http://127.0.0.1:18620; it encodes a small PNG and POSTs it through
// UploadFileSync, mirroring the backend.uploadFile code path.
//
// ASCII-only source.
#include "doctest.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <string>
#include <vector>

#include "json.hpp"

#include "bridge_core.h"
#include "http_client.h"
#include "wic_png.h"

using json = nlohmann::json;

namespace {

std::wstring TempPath(const std::wstring& name) {
    wchar_t dir[MAX_PATH] = {};
    const DWORD n = ::GetTempPathW(MAX_PATH, dir);
    std::wstring path(dir, n);
    path += L"nzltx23_test_";
    path += name;
    return path;
}

// Read a whole file into memory.
std::vector<unsigned char> ReadFileBytes(const std::wstring& path) {
    std::vector<unsigned char> out;
    HANDLE h = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return out;
    }
    LARGE_INTEGER size = {};
    ::GetFileSizeEx(h, &size);
    out.resize(static_cast<size_t>(size.QuadPart));
    DWORD read = 0;
    if (!out.empty()) {
        ::ReadFile(h, out.data(), static_cast<DWORD>(out.size()), &read, nullptr);
    }
    ::CloseHandle(h);
    out.resize(read);
    return out;
}

// Build a width x height RGBA gradient (fully opaque).
std::vector<unsigned char> MakeGradient(int width, int height) {
    std::vector<unsigned char> px(static_cast<size_t>(width) * height * 4u);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t i = (static_cast<size_t>(y) * width + x) * 4u;
            px[i + 0] = static_cast<unsigned char>((x * 255) / (width - 1));   // R
            px[i + 1] = static_cast<unsigned char>((y * 255) / (height - 1));  // G
            px[i + 2] = 64;                                                    // B
            px[i + 3] = 255;                                                   // A
        }
    }
    return px;
}

}  // namespace

TEST_CASE("EncodeRgbaToPngFile writes a valid PNG with the right dimensions") {
    const int width = 12;
    const int height = 8;
    const std::vector<unsigned char> px = MakeGradient(width, height);
    const std::wstring dest = TempPath(L"wic_encode.png");
    ::DeleteFileW(dest.c_str());

    std::string err;
    const bool encoded = nzltx::EncodeRgbaToPngFile(dest, px.data(), width, height, &err);
    INFO("encode error: ", err);
    REQUIRE(encoded);

    const std::vector<unsigned char> bytes = ReadFileBytes(dest);
    REQUIRE(bytes.size() > 33);  // 8-byte signature + IHDR
    // PNG signature.
    const unsigned char sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    for (int i = 0; i < 8; ++i) {
        CHECK(bytes[i] == sig[i]);
    }
    // IHDR width/height are big-endian 32-bit integers at offsets 16 and 20.
    const unsigned int w = (bytes[16] << 24) | (bytes[17] << 16) | (bytes[18] << 8) |
                           bytes[19];
    const unsigned int h = (bytes[20] << 24) | (bytes[21] << 16) | (bytes[22] << 8) |
                           bytes[23];
    CHECK(w == static_cast<unsigned int>(width));
    CHECK(h == static_cast<unsigned int>(height));
    ::DeleteFileW(dest.c_str());
}

TEST_CASE("EncodeRgbaToPngFile rejects invalid arguments") {
    std::string err;
    const std::wstring dest = TempPath(L"wic_invalid.png");
    CHECK_FALSE(nzltx::EncodeRgbaToPngFile(dest, nullptr, 4, 4, &err));
    const std::vector<unsigned char> px(16, 0);
    CHECK_FALSE(nzltx::EncodeRgbaToPngFile(dest, px.data(), 0, 4, &err));
}

TEST_CASE("MakeJpegThumbnail decodes a PNG and downscales it to a JPEG") {
    // Encode a 320x200 source PNG, then request a 64px thumbnail of it.
    const int width = 320;
    const int height = 200;
    const std::vector<unsigned char> px = MakeGradient(width, height);
    const std::wstring src = TempPath(L"thumb_source.png");
    ::DeleteFileW(src.c_str());
    std::string enc_err;
    REQUIRE(nzltx::EncodeRgbaToPngFile(src, px.data(), width, height, &enc_err));

    std::vector<unsigned char> bytes;
    int tw = 0;
    int th = 0;
    int sw = 0;
    int sh = 0;
    std::string err;
    const bool ok =
        nzltx::MakeJpegThumbnail(src, 64, &bytes, &tw, &th, &sw, &sh, &err);
    INFO("thumbnail error: ", err);
    REQUIRE(ok);
    CHECK(sw == width);
    CHECK(sh == height);
    // Long edge (width) scaled to 64; short edge preserves the 320:200 ratio.
    CHECK(tw == 64);
    CHECK(th == 40);
    // JPEG SOI magic bytes.
    REQUIRE(bytes.size() > 3);
    CHECK(bytes[0] == 0xFF);
    CHECK(bytes[1] == 0xD8);
    CHECK(bytes[2] == 0xFF);

    ::DeleteFileW(src.c_str());
}

TEST_CASE("MakeJpegThumbnail does not upscale a small source") {
    const int width = 24;
    const int height = 16;
    const std::vector<unsigned char> px = MakeGradient(width, height);
    const std::wstring src = TempPath(L"thumb_small.png");
    ::DeleteFileW(src.c_str());
    std::string enc_err;
    REQUIRE(nzltx::EncodeRgbaToPngFile(src, px.data(), width, height, &enc_err));

    std::vector<unsigned char> bytes;
    int tw = 0;
    int th = 0;
    int sw = 0;
    int sh = 0;
    std::string err;
    REQUIRE(nzltx::MakeJpegThumbnail(src, 256, &bytes, &tw, &th, &sw, &sh, &err));
    CHECK(tw == width);   // already smaller than max_dim -> unchanged
    CHECK(th == height);
    ::DeleteFileW(src.c_str());
}

TEST_CASE("MakeJpegThumbnail fails cleanly on a missing/undecodable file") {
    std::vector<unsigned char> bytes;
    int tw = 0;
    int th = 0;
    int sw = 0;
    int sh = 0;
    std::string err;
    const std::wstring missing = TempPath(L"thumb_does_not_exist.png");
    ::DeleteFileW(missing.c_str());
    CHECK_FALSE(
        nzltx::MakeJpegThumbnail(missing, 128, &bytes, &tw, &th, &sw, &sh, &err));
    CHECK_FALSE(err.empty());
}

TEST_SUITE("integration") {

TEST_CASE("uploadFile code path: encode a PNG and POST it to /upload/image") {
    // 1. Create a small PNG on disk via the same encoder captureFrame uses.
    const int width = 16;
    const int height = 16;
    const std::vector<unsigned char> px = MakeGradient(width, height);
    const std::wstring png = TempPath(L"upload_probe.png");
    ::DeleteFileW(png.c_str());
    std::string enc_err;
    REQUIRE(nzltx::EncodeRgbaToPngFile(png, px.data(), width, height, &enc_err));

    // 2. Derive the URL / filename / content-type exactly like the bridge.
    const std::string url = std::string(nzltx::kBackendBaseUrl) +
                            nzltx::kBackendApiPrefix + nzltx::UploadPath("image");
    const std::string filename = "upload_probe.png";
    const std::string content_type = nzltx::ContentTypeForExtension("upload_probe.png");
    CHECK(content_type == "image/png");

    // 3. Upload through the multipart client.
    nzltx::HttpClient client;
    const nzltx::HttpResponse resp = client.UploadFileSync(
        url, png, "file", filename, content_type, nzltx::kUploadTimeoutMs);
    REQUIRE(resp.transport == nzltx::TransportError::kNone);
    CHECK(resp.status == 200);

    const json body = json::parse(resp.body, nullptr, false);
    REQUIRE_FALSE(body.is_discarded());
    CHECK(body.contains("image_id"));
    CHECK(body["image_id"].is_string());
    CHECK_FALSE(body["image_id"].get<std::string>().empty());
    CHECK(body.value("width", 0) == width);
    CHECK(body.value("height", 0) == height);

    ::DeleteFileW(png.c_str());
}

}  // TEST_SUITE("integration")
