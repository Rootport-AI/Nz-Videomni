// mf_mp4_writer.cpp - Media Foundation mp4 (H.264 + AAC) writer. See header.
// ASCII-only source.
#include "mf_mp4_writer.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <codecapi.h>  // eAVEncH264VProfile_* enums

#include <algorithm>
#include <cstring>
#include <vector>

namespace nzvideomni {

namespace {

// Minimal RAII COM pointer (mirrors wic_png.cpp; keeps this TU free of wil so it
// builds in the plain test executable).
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

// Recursively create every component of a directory path. Best effort.
void EnsureParentDir(const std::wstring& path) {
    const size_t slash = path.find_last_of(L"\\/");
    if (slash == std::wstring::npos) {
        return;
    }
    const std::wstring dir = path.substr(0, slash);
    for (size_t i = 1; i < dir.size(); ++i) {
        if (dir[i] == L'\\' || dir[i] == L'/') {
            ::CreateDirectoryW(dir.substr(0, i).c_str(), nullptr);
        }
    }
    ::CreateDirectoryW(dir.c_str(), nullptr);
}

// 100-nanosecond ticks per second (Media Foundation timeline unit).
constexpr int64_t kTicksPerSecond = 10000000LL;

}  // namespace

struct Mp4Writer::Impl {
    Mp4WriterConfig cfg;

    bool com_needs_uninit = false;  // we took a COM ref that we must balance
    bool mf_started = false;        // MFStartup succeeded -> owe an MFShutdown
    bool writing = false;           // BeginWriting done, Finalize not yet called
    bool failed = false;            // a Write* call failed; stop feeding samples

    ComPtr<IMFSinkWriter> writer;
    DWORD video_stream = 0;
    DWORD audio_stream = 0;
    bool has_audio_stream = false;

    int64_t video_frame_index = 0;  // number of frames written so far
    int64_t video_frame_duration = 0;  // per-frame duration in 100ns ticks
    int64_t audio_samples_written = 0;  // running audio frame (sample) count

    // Scratch buffer for interleaved PCM16 audio (reused across calls).
    std::vector<int16_t> pcm_scratch;
};

Mp4Writer::Mp4Writer() : impl_(std::make_unique<Impl>()) {}

Mp4Writer::~Mp4Writer() {
    if (impl_) {
        std::string ignored;
        Finalize(&ignored);
    }
}

namespace {

// Set major/subtype + common video attributes shared by in/out media types.
HRESULT ConfigureVideoCommon(IMFMediaType* mt, const Mp4WriterConfig& cfg,
                             const GUID& subtype) {
    HRESULT hr = mt->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    if (SUCCEEDED(hr)) hr = mt->SetGUID(MF_MT_SUBTYPE, subtype);
    if (SUCCEEDED(hr)) {
        hr = mt->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
    }
    if (SUCCEEDED(hr)) {
        hr = MFSetAttributeSize(mt, MF_MT_FRAME_SIZE,
                                static_cast<UINT32>(cfg.width),
                                static_cast<UINT32>(cfg.height));
    }
    if (SUCCEEDED(hr)) {
        hr = MFSetAttributeRatio(mt, MF_MT_FRAME_RATE, cfg.fps_num, cfg.fps_den);
    }
    if (SUCCEEDED(hr)) {
        hr = MFSetAttributeRatio(mt, MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
    }
    return hr;
}

}  // namespace

bool Mp4Writer::Initialize(const Mp4WriterConfig& config, std::string* err) {
    auto fail = [err](const char* msg) -> bool {
        if (err != nullptr) *err = msg;
        return false;
    };

    if (impl_->writing || impl_->writer) {
        return fail("Mp4Writer already initialized");
    }
    if (config.output_path.empty() || config.width <= 0 || config.height <= 0) {
        return fail("invalid output path or dimensions");
    }
    if (config.fps_num == 0 || config.fps_den == 0) {
        return fail("invalid frame rate");
    }
    if (config.has_audio &&
        (config.audio_sample_rate == 0 || config.audio_channels == 0 ||
         config.audio_channels > 2)) {
        return fail("invalid audio parameters (channels must be 1 or 2)");
    }

    impl_->cfg = config;

    // --- COM init (tolerant of an already-initialised host thread) -----------
    // S_OK: we initialised COM and must balance it. S_FALSE: already init on
    // this thread in a compatible mode; still ref-counted, so balance it.
    // RPC_E_CHANGED_MODE: the thread is already an STA and we asked for MTA;
    // MF works either way, so proceed WITHOUT taking (or later releasing) a ref.
    const HRESULT co = ::CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (co == RPC_E_CHANGED_MODE) {
        impl_->com_needs_uninit = false;
    } else if (SUCCEEDED(co)) {
        impl_->com_needs_uninit = true;
    } else {
        return fail("CoInitializeEx failed");
    }

    HRESULT hr = ::MFStartup(MF_VERSION, MFSTARTUP_LITE);
    if (FAILED(hr)) {
        return fail("MFStartup failed");
    }
    impl_->mf_started = true;

    EnsureParentDir(config.output_path);

    // --- Sink writer: force software transforms for portability --------------
    ComPtr<IMFAttributes> sink_attrs;
    hr = ::MFCreateAttributes(sink_attrs.put(), 2);
    if (SUCCEEDED(hr)) {
        // Do NOT require a hardware encoder: fall back to the OS software H.264
        // MFT so this works on machines with no GPU encode block.
        hr = sink_attrs->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, 0);
    }
    if (SUCCEEDED(hr)) {
        hr = sink_attrs->SetUINT32(MF_SINK_WRITER_DISABLE_THROTTLING, 1);
    }
    if (FAILED(hr)) {
        return fail("MFCreateAttributes(sink) failed");
    }

    hr = ::MFCreateSinkWriterFromURL(config.output_path.c_str(), nullptr,
                                     sink_attrs.get(), impl_->writer.put());
    if (FAILED(hr) || !impl_->writer) {
        return fail("MFCreateSinkWriterFromURL failed");
    }

    // --- Video output (H.264) ------------------------------------------------
    uint32_t bitrate = config.video_bitrate;
    if (bitrate == 0) {
        // Rough default: ~0.12 bits/pixel/frame, clamped to a sane band.
        const double bpp = 0.12;
        const double fps =
            static_cast<double>(config.fps_num) / config.fps_den;
        double b = bpp * config.width * config.height * fps;
        b = std::clamp(b, 1000000.0, 24000000.0);
        bitrate = static_cast<uint32_t>(b);
    }

    ComPtr<IMFMediaType> vout;
    hr = ::MFCreateMediaType(vout.put());
    if (SUCCEEDED(hr)) hr = ConfigureVideoCommon(vout.get(), config, MFVideoFormat_H264);
    if (SUCCEEDED(hr)) hr = vout->SetUINT32(MF_MT_AVG_BITRATE, bitrate);
    if (SUCCEEDED(hr)) {
        hr = vout->SetUINT32(MF_MT_MPEG2_PROFILE, eAVEncH264VProfile_Main);
    }
    if (FAILED(hr)) {
        return fail("configure H.264 output media type failed");
    }
    hr = impl_->writer->AddStream(vout.get(), &impl_->video_stream);
    if (FAILED(hr)) {
        return fail("AddStream(video) failed");
    }

    // --- Video input (uncompressed BGRA, top-down positive stride) -----------
    ComPtr<IMFMediaType> vin;
    hr = ::MFCreateMediaType(vin.put());
    if (SUCCEEDED(hr)) hr = ConfigureVideoCommon(vin.get(), config, MFVideoFormat_RGB32);
    if (SUCCEEDED(hr)) {
        // Positive stride => top-down frame. Our RGBA source has row 0 at the
        // top, so this keeps the decoded video upright (no vertical flip).
        hr = vin->SetUINT32(MF_MT_DEFAULT_STRIDE,
                            static_cast<UINT32>(config.width * 4));
    }
    if (FAILED(hr)) {
        return fail("configure RGB32 input media type failed");
    }
    hr = impl_->writer->SetInputMediaType(impl_->video_stream, vin.get(), nullptr);
    if (FAILED(hr)) {
        return fail("SetInputMediaType(video) failed");
    }

    // --- Audio streams (optional) --------------------------------------------
    if (config.has_audio) {
        // AAC output.
        ComPtr<IMFMediaType> aout;
        hr = ::MFCreateMediaType(aout.put());
        if (SUCCEEDED(hr)) hr = aout->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
        if (SUCCEEDED(hr)) hr = aout->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_AAC);
        if (SUCCEEDED(hr)) {
            hr = aout->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 16);
        }
        if (SUCCEEDED(hr)) {
            hr = aout->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND,
                                 config.audio_sample_rate);
        }
        if (SUCCEEDED(hr)) {
            hr = aout->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, config.audio_channels);
        }
        if (SUCCEEDED(hr)) {
            hr = aout->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND,
                                 config.audio_bitrate / 8);
        }
        if (SUCCEEDED(hr)) {
            hr = aout->SetUINT32(MF_MT_AAC_PAYLOAD_TYPE, 0);  // raw AAC
        }
        if (FAILED(hr)) {
            return fail("configure AAC output media type failed");
        }
        hr = impl_->writer->AddStream(aout.get(), &impl_->audio_stream);
        if (FAILED(hr)) {
            return fail("AddStream(audio) failed");
        }

        // PCM16 interleaved input. rendering_scene_audio hands us planar float32
        // (L in buffer0, R in buffer1); WriteAudioSamples interleaves + quantises
        // to signed 16-bit PCM, which the OS AAC encoder MFT accepts directly.
        const uint32_t ch = config.audio_channels;
        const uint32_t block_align = ch * 2;  // 2 bytes/sample (16-bit)
        ComPtr<IMFMediaType> ain;
        hr = ::MFCreateMediaType(ain.put());
        if (SUCCEEDED(hr)) hr = ain->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
        if (SUCCEEDED(hr)) hr = ain->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_PCM);
        if (SUCCEEDED(hr)) hr = ain->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 16);
        if (SUCCEEDED(hr)) {
            hr = ain->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND,
                                config.audio_sample_rate);
        }
        if (SUCCEEDED(hr)) {
            hr = ain->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, ch);
        }
        if (SUCCEEDED(hr)) {
            hr = ain->SetUINT32(MF_MT_AUDIO_BLOCK_ALIGNMENT, block_align);
        }
        if (SUCCEEDED(hr)) {
            hr = ain->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND,
                                config.audio_sample_rate * block_align);
        }
        if (SUCCEEDED(hr)) {
            hr = ain->SetUINT32(MF_MT_ALL_SAMPLES_INDEPENDENT, 1);
        }
        if (FAILED(hr)) {
            return fail("configure PCM input media type failed");
        }
        hr = impl_->writer->SetInputMediaType(impl_->audio_stream, ain.get(),
                                              nullptr);
        if (FAILED(hr)) {
            return fail("SetInputMediaType(audio) failed");
        }
        impl_->has_audio_stream = true;
    }

    hr = impl_->writer->BeginWriting();
    if (FAILED(hr)) {
        return fail("IMFSinkWriter::BeginWriting failed");
    }

    // Per-frame duration in 100ns ticks: fps_den / fps_num seconds.
    impl_->video_frame_duration =
        ::MFllMulDiv(kTicksPerSecond, config.fps_den, config.fps_num, 0);
    impl_->writing = true;
    return true;
}

bool Mp4Writer::WriteVideoFrame(const uint8_t* pixels, int pitch,
                                std::string* err) {
    auto fail = [err, this](const char* msg) -> bool {
        if (err != nullptr) *err = msg;
        impl_->failed = true;
        return false;
    };

    if (!impl_->writing || !impl_->writer) {
        return fail("WriteVideoFrame before Initialize / after Finalize");
    }
    if (impl_->failed) {
        return fail("writer already in a failed state");
    }
    if (pixels == nullptr) {
        return fail("null pixel buffer");
    }
    const int width = impl_->cfg.width;
    const int height = impl_->cfg.height;
    const int dst_stride = width * 4;
    if (pitch < dst_stride) {
        return fail("pitch smaller than width*4");
    }

    const DWORD buffer_size = static_cast<DWORD>(dst_stride) *
                              static_cast<DWORD>(height);
    ComPtr<IMFMediaBuffer> buffer;
    HRESULT hr = ::MFCreateMemoryBuffer(buffer_size, buffer.put());
    if (FAILED(hr) || !buffer) {
        return fail("MFCreateMemoryBuffer(video) failed");
    }

    BYTE* dst = nullptr;
    hr = buffer->Lock(&dst, nullptr, nullptr);
    if (FAILED(hr) || dst == nullptr) {
        return fail("IMFMediaBuffer::Lock(video) failed");
    }

    // Convert into MF's RGB32 (memory BGRA) layout, top-down (row 0 first). The
    // input media type declared a positive stride, so no vertical flip is done.
    const bool swap_rb = (impl_->cfg.pixel_order == PixelOrder::kRGBA);
    for (int y = 0; y < height; ++y) {
        const uint8_t* s = pixels + static_cast<size_t>(y) * pitch;
        uint8_t* d = dst + static_cast<size_t>(y) * dst_stride;
        if (swap_rb) {
            // src = R,G,B,A -> dst = B,G,R,A (swap channels 0 and 2).
            for (int x = 0; x < width; ++x) {
                d[0] = s[2];  // B
                d[1] = s[1];  // G
                d[2] = s[0];  // R
                d[3] = s[3];  // A
                s += 4;
                d += 4;
            }
        } else {
            // src already B,G,R,A -> straight copy.
            std::memcpy(d, s, static_cast<size_t>(dst_stride));
        }
    }
    buffer->Unlock();
    hr = buffer->SetCurrentLength(buffer_size);
    if (FAILED(hr)) {
        return fail("SetCurrentLength(video) failed");
    }

    ComPtr<IMFSample> sample;
    hr = ::MFCreateSample(sample.put());
    if (SUCCEEDED(hr)) hr = sample->AddBuffer(buffer.get());
    if (SUCCEEDED(hr)) {
        hr = sample->SetSampleTime(impl_->video_frame_index *
                                   impl_->video_frame_duration);
    }
    if (SUCCEEDED(hr)) {
        hr = sample->SetSampleDuration(impl_->video_frame_duration);
    }
    if (FAILED(hr)) {
        return fail("build video sample failed");
    }
    hr = impl_->writer->WriteSample(impl_->video_stream, sample.get());
    if (FAILED(hr)) {
        return fail("IMFSinkWriter::WriteSample(video) failed");
    }
    ++impl_->video_frame_index;
    return true;
}

bool Mp4Writer::WriteAudioSamples(const float* left, const float* right,
                                  int sample_num, std::string* err) {
    auto fail = [err, this](const char* msg) -> bool {
        if (err != nullptr) *err = msg;
        impl_->failed = true;
        return false;
    };

    if (!impl_->cfg.has_audio || !impl_->has_audio_stream) {
        return true;  // audio disabled: silently ignore.
    }
    if (!impl_->writing || !impl_->writer) {
        return fail("WriteAudioSamples before Initialize / after Finalize");
    }
    if (impl_->failed) {
        return fail("writer already in a failed state");
    }
    if (sample_num <= 0) {
        return true;  // nothing to do
    }
    if (left == nullptr) {
        return fail("null left audio buffer");
    }

    const uint32_t ch = impl_->cfg.audio_channels;
    const size_t total = static_cast<size_t>(sample_num) * ch;
    impl_->pcm_scratch.resize(total);

    // Quantise float [-1,1] -> int16 with clamping. Planar L/R -> interleaved.
    auto to_pcm = [](float v) -> int16_t {
        float s = v * 32767.0f;
        if (s > 32767.0f) s = 32767.0f;
        if (s < -32768.0f) s = -32768.0f;
        return static_cast<int16_t>(s < 0 ? s - 0.5f : s + 0.5f);
    };

    int16_t* out = impl_->pcm_scratch.data();
    if (ch == 1) {
        for (int i = 0; i < sample_num; ++i) {
            *out++ = to_pcm(left[i]);
        }
    } else {
        // right == nullptr -> duplicate the left channel (mono source upmix).
        const float* r = (right != nullptr) ? right : left;
        for (int i = 0; i < sample_num; ++i) {
            *out++ = to_pcm(left[i]);
            *out++ = to_pcm(r[i]);
        }
    }

    const DWORD bytes = static_cast<DWORD>(total * sizeof(int16_t));
    ComPtr<IMFMediaBuffer> buffer;
    HRESULT hr = ::MFCreateMemoryBuffer(bytes, buffer.put());
    if (FAILED(hr) || !buffer) {
        return fail("MFCreateMemoryBuffer(audio) failed");
    }
    BYTE* dst = nullptr;
    hr = buffer->Lock(&dst, nullptr, nullptr);
    if (FAILED(hr) || dst == nullptr) {
        return fail("IMFMediaBuffer::Lock(audio) failed");
    }
    std::memcpy(dst, impl_->pcm_scratch.data(), bytes);
    buffer->Unlock();
    hr = buffer->SetCurrentLength(bytes);
    if (FAILED(hr)) {
        return fail("SetCurrentLength(audio) failed");
    }

    const int64_t pts = ::MFllMulDiv(impl_->audio_samples_written,
                                     kTicksPerSecond,
                                     impl_->cfg.audio_sample_rate, 0);
    const int64_t dur = ::MFllMulDiv(sample_num, kTicksPerSecond,
                                     impl_->cfg.audio_sample_rate, 0);
    ComPtr<IMFSample> sample;
    hr = ::MFCreateSample(sample.put());
    if (SUCCEEDED(hr)) hr = sample->AddBuffer(buffer.get());
    if (SUCCEEDED(hr)) hr = sample->SetSampleTime(pts);
    if (SUCCEEDED(hr)) hr = sample->SetSampleDuration(dur);
    if (FAILED(hr)) {
        return fail("build audio sample failed");
    }
    hr = impl_->writer->WriteSample(impl_->audio_stream, sample.get());
    if (FAILED(hr)) {
        return fail("IMFSinkWriter::WriteSample(audio) failed");
    }
    impl_->audio_samples_written += sample_num;
    return true;
}

bool Mp4Writer::Finalize(std::string* err) {
    auto fail = [err](const char* msg) -> bool {
        if (err != nullptr) *err = msg;
        return false;
    };

    bool ok = true;
    if (impl_->writer && impl_->writing && !impl_->failed) {
        HRESULT hr = impl_->writer->Finalize();
        if (FAILED(hr)) {
            ok = fail("IMFSinkWriter::Finalize failed");
        }
    } else if (impl_->failed) {
        ok = fail("writer was in a failed state; file may be incomplete");
    }
    impl_->writing = false;
    impl_->writer.reset();

    if (impl_->mf_started) {
        ::MFShutdown();
        impl_->mf_started = false;
    }
    if (impl_->com_needs_uninit) {
        ::CoUninitialize();
        impl_->com_needs_uninit = false;
    }
    return ok;
}

}  // namespace nzvideomni
