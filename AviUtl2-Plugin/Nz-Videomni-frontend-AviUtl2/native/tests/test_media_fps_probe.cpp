// test_media_fps_probe.cpp - FpsFromRatio's pure ratio arithmetic, plus one
// Media Foundation round trip: encode a small 30 fps mp4 with Mp4Writer, probe
// it back with ProbeMediaFps, and assert both that the probe SUCCEEDED and that
// it reports ~30. Generated files live under %TEMP% (the harness scratchpad) and
// are deleted after each case, following test_mf_mp4_writer.cpp.
//
// ASCII-only source.
#include "doctest.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <cstdint>
#include <string>
#include <vector>

#include "media_fps_probe.h"
#include "mf_mp4_writer.h"

namespace {

// Unique scratchpad path from %TEMP% (mirrors TempMp4Path in
// test_mf_mp4_writer.cpp, with its own prefix so the two never collide).
std::wstring TempProbePath(const std::wstring& name) {
    wchar_t dir[MAX_PATH] = {};
    const DWORD n = ::GetTempPathW(MAX_PATH, dir);
    std::wstring path(dir, n);
    path += L"nzvideomni_fpsprobe_";
    path += name;
    return path;
}

// A flat grey frame whose brightness changes per frame. The probe only reads the
// container's declared media type, so the pixel content is irrelevant - it just
// has to be something the H.264 encoder accepts.
void MakeFlatFrame(std::vector<uint8_t>* px, int width, int height, int frame_idx) {
    px->resize(static_cast<size_t>(width) * height * 4u);
    const uint8_t v = static_cast<uint8_t>(16 + (frame_idx * 8) % 200);
    for (size_t i = 0; i < px->size(); i += 4) {
        (*px)[i + 0] = v;
        (*px)[i + 1] = v;
        (*px)[i + 2] = v;
        (*px)[i + 3] = 255;
    }
}

}  // namespace

TEST_CASE("FpsFromRatio converts an MF_MT_FRAME_RATE ratio to fps") {
    // Integer rates round-trip exactly.
    CHECK(nzvideomni::FpsFromRatio(24, 1) == doctest::Approx(24.0));
    CHECK(nzvideomni::FpsFromRatio(30, 1) == doctest::Approx(30.0));
    CHECK(nzvideomni::FpsFromRatio(60, 1) == doctest::Approx(60.0));

    // NTSC rates are reported RAW - snapping 29.97 to 30 is the webui's job,
    // never this module's.
    CHECK(nzvideomni::FpsFromRatio(30000, 1001) == doctest::Approx(29.97));
    CHECK(nzvideomni::FpsFromRatio(24000, 1001) == doctest::Approx(23.976));
    CHECK(nzvideomni::FpsFromRatio(60000, 1001) == doctest::Approx(59.94));
    // ...and they really are not rounded up on the way out.
    CHECK(nzvideomni::FpsFromRatio(30000, 1001) < 30.0);
    CHECK(nzvideomni::FpsFromRatio(24000, 1001) < 24.0);
}

TEST_CASE("FpsFromRatio reports 0 for a ratio with a zero component") {
    // 0 is SelectionItem::media_fps's "unknown" value; it is never a real rate.
    CHECK(nzvideomni::FpsFromRatio(30000, 0) == 0.0);  // no denominator
    CHECK(nzvideomni::FpsFromRatio(0, 1001) == 0.0);   // no numerator
    CHECK(nzvideomni::FpsFromRatio(0, 0) == 0.0);      // neither
}

TEST_CASE("ProbeMediaFps reads the frame rate back from a real mp4") {
    const int width = 160;
    const int height = 96;
    const uint32_t fps_num = 30;
    const int num_frames = 12;
    const std::wstring path = TempProbePath(L"fps30.mp4");
    ::DeleteFileW(path.c_str());

    nzvideomni::Mp4WriterConfig cfg;
    cfg.output_path = path;
    cfg.width = width;
    cfg.height = height;
    cfg.fps_num = fps_num;
    cfg.fps_den = 1;
    cfg.has_audio = false;

    nzvideomni::Mp4Writer writer;
    std::string err;
    REQUIRE_MESSAGE(writer.Initialize(cfg, &err), err);
    std::vector<uint8_t> frame;
    for (int f = 0; f < num_frames; ++f) {
        MakeFlatFrame(&frame, width, height, f);
        REQUIRE_MESSAGE(writer.WriteVideoFrame(frame.data(), width * 4, &err), err);
    }
    REQUIRE_MESSAGE(writer.Finalize(&err), err);

    double probed = -1.0;
    std::string probe_err;
    // Assert the probe SUCCEEDED, not only the value: on a silent false `probed`
    // would keep its sentinel and a value-only check would say nothing about
    // whether the Media Foundation path works at all.
    const bool ok = nzvideomni::ProbeMediaFps(path, &probed, &probe_err);
    CHECK_MESSAGE(ok, probe_err);
    CHECK(probed == doctest::Approx(30.0).epsilon(0.001));

    ::DeleteFileW(path.c_str());
}

TEST_CASE("ProbeMediaFps fails cleanly when the file cannot be opened") {
    // A missing file stands in for the "Media Foundation cannot open it" family
    // - a NORMAL outcome the bridge answers with mediaFps 0. (Matroska is NOT
    // in this family any more: Windows does open it, and the Matroska rule
    // below answers it before Media Foundation is reached.)
    const std::wstring missing = TempProbePath(L"no_such_file.mp4");
    ::DeleteFileW(missing.c_str());

    double fps = -1.0;
    std::string err;
    CHECK_FALSE(nzvideomni::ProbeMediaFps(missing, &fps, &err));
    CHECK_FALSE(err.empty());
    CHECK(fps == -1.0);  // *fps_out is left untouched on failure

    // An empty path is rejected before any Media Foundation work happens.
    std::string empty_err;
    CHECK_FALSE(nzvideomni::ProbeMediaFps(L"", &fps, &empty_err));
    CHECK_FALSE(empty_err.empty());

    // err may be null - the plugin passes nullptr, since a failed probe is an
    // expected outcome it deliberately does not log.
    CHECK_FALSE(nzvideomni::ProbeMediaFps(missing, &fps, nullptr));
}

TEST_CASE("ProbeMediaFps reports a Matroska file as unknown by rule") {
    // Windows' Matroska source declares HALF the real rate (29.97 -> 14.985,
    // see media_fps_probe.h), so .mkv / .webm / .mk3d are refused by extension,
    // in any case, before Media Foundation is opened. A missing file fails too,
    // so the REASON string is what proves the Matroska rule (and not the
    // missing file) made the probe fail.
    const std::string matroska_reason =
        "Matroska: Media Foundation's MKV source misreports the frame rate; treated as unknown";
    const wchar_t* names[] = {L"clip.mkv", L"clip.webm", L"CLIP.MKV", L"clip.Mk3d"};
    for (const wchar_t* name : names) {
        const std::wstring path = TempProbePath(name);
        double fps = -1.0;
        std::string err;
        CHECK_FALSE(nzvideomni::ProbeMediaFps(path, &fps, &err));
        CHECK(err == matroska_reason);
        CHECK(fps == -1.0);  // *fps_out is left untouched on failure
    }

    // Only the file's own extension counts: a dot in a directory name, or an
    // mkv-looking name with another extension, is not Matroska (these fail for
    // the ordinary "cannot open" reason instead).
    const wchar_t* others[] = {L"dir.mkv\\clip", L"clip.mkv.mp4"};
    for (const wchar_t* name : others) {
        double fps = -1.0;
        std::string err;
        CHECK_FALSE(nzvideomni::ProbeMediaFps(TempProbePath(name), &fps, &err));
        CHECK(err != matroska_reason);
    }
}
