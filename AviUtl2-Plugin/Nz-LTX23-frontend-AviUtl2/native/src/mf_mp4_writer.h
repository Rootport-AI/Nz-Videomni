// mf_mp4_writer.h - encode a sequence of RGBA frames (+ optional PCM audio) to an
// mp4 (H.264 video + AAC audio) using Media Foundation's IMFSinkWriter and the
// OS-bundled software H.264 / AAC encoders. No AviUtl2 SDK dependency: this is a
// pure encoder that takes raw pixel rows and planar float32 audio.
//
// --- Pixel byte order --------------------------------------------------------
// AviUtl2's rendering callback hands out a PIXEL_RGBA buffer, defined in the SDK
// (filter2.h) as `struct PIXEL_RGBA { unsigned char r, g, b, a; }`, i.e. the
// bytes in memory are R, G, B, A (== DXGI_FORMAT_R8G8B8A8_UNORM). Media
// Foundation's uncompressed 32-bit RGB format, MFVideoFormat_RGB32, is instead
// BGRA in memory (a 0xAARRGGBB little-endian DWORD: byte0=B, byte1=G, byte2=R,
// byte3=A). WriteVideoFrame therefore swaps R and B when the input order is the
// default kRGBA, and copies straight through for kBGRA. See PixelOrder below.
//
// --- Vertical orientation ----------------------------------------------------
// The AviUtl2 buffer is top-down (row 0 is the top row). MFVideoFormat_RGB32 is
// historically bottom-up, so the input media type sets a POSITIVE
// MF_MT_DEFAULT_STRIDE (= width*4) to declare the frame top-down, and
// WriteVideoFrame copies rows in natural (top-to-bottom) order. This keeps the
// decoded video upright. (Verified by the unit test, which encodes a frame with
// a distinct top vs. bottom band and reads the corners back.)
//
// --- COM / MF lifecycle ------------------------------------------------------
// Initialize() calls CoInitializeEx(COINIT_MULTITHREADED) and MFStartup(); it
// balances CoUninitialize()/MFShutdown() in Finalize() (or the destructor). COM
// init tolerates an already-initialised thread (S_FALSE) and a prior init in a
// different apartment (RPC_E_CHANGED_MODE): in both cases the writer does NOT
// balance the ref it did not take, so it composes safely with a host that has
// already called CoInitialize on the calling thread. MFStartup is internally
// ref-counted, so several writers (or a host that also started MF) coexist.
//
// Failures are reported as `bool` + a short ASCII diagnostic via the err string,
// matching wic_png.{h,cpp}; nothing throws. All comments are ASCII.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace nzltx {

// Byte order of the caller's source pixels (4 bytes/pixel).
enum class PixelOrder {
    kRGBA,  // byte0=R, byte1=G, byte2=B, byte3=A  (AviUtl2 PIXEL_RGBA; default)
    kBGRA,  // byte0=B, byte1=G, byte2=R, byte3=A  (native MF RGB32 order)
};

struct Mp4WriterConfig {
    std::wstring output_path;             // native wide path to the .mp4 to write
    int width = 0;                        // frame width in pixels (> 0)
    int height = 0;                       // frame height in pixels (> 0)
    uint32_t fps_num = 30;                // frame rate numerator   (rate)
    uint32_t fps_den = 1;                 // frame rate denominator (scale)
    uint32_t video_bitrate = 0;           // avg H.264 bitrate in bps; 0 => auto
    PixelOrder pixel_order = PixelOrder::kRGBA;

    bool has_audio = false;               // false => video-only mp4
    uint32_t audio_sample_rate = 48000;   // AAC supports 44100 or 48000 reliably
    uint32_t audio_channels = 2;          // 1 or 2 (stereo assumed)
    uint32_t audio_bitrate = 128000;      // avg AAC bitrate in bps
};

// Stateful mp4 writer. Usage:
//   Mp4Writer w;
//   w.Initialize(cfg, &err);
//   for each frame: w.WriteVideoFrame(rgba, pitch, &err);
//   for each audio block: w.WriteAudioSamples(l, r, n, &err);   // if has_audio
//   w.Finalize(&err);
// Video and audio samples may be interleaved in any order; the sink writer times
// them from the fps / sample-rate counters this class maintains internally.
class Mp4Writer {
public:
    Mp4Writer();
    ~Mp4Writer();
    Mp4Writer(const Mp4Writer&) = delete;
    Mp4Writer& operator=(const Mp4Writer&) = delete;

    // Configure the sink writer and begin writing. Returns false + *err on any
    // failure. Must be called exactly once before the Write* methods.
    bool Initialize(const Mp4WriterConfig& config, std::string* err);

    // Append one video frame. `pixels` points at the top-left pixel; `pitch` is
    // the source row stride in bytes (>= width*4; may include row padding). The
    // presentation timestamp is derived from the frame counter and fps.
    bool WriteVideoFrame(const uint8_t* pixels, int pitch, std::string* err);

    // Append `sample_num` audio frames. `left`/`right` are planar float32 in the
    // shape rendering_scene_audio delivers (normalised roughly to [-1, 1]).
    // For a mono config only `left` is read; for stereo, `right == nullptr`
    // duplicates the left channel. Ignored (returns true) if has_audio is false.
    bool WriteAudioSamples(const float* left, const float* right, int sample_num,
                           std::string* err);

    // Flush and close the file (IMFSinkWriter::Finalize), then release MF/COM.
    // Safe to call once; the destructor calls it if the caller did not.
    bool Finalize(std::string* err);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nzltx
