// wic_png.cpp - WIC PNG/JPEG encoder implementation, plus the Media Foundation
// first-frame grab for video thumbnails (contract v17). See wic_png.h. ASCII-only.
#include "wic_png.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <objidl.h>  // IStream, GetHGlobalFromStream
#include <wincodec.h>

#include <mfapi.h>
#include <mfidl.h>
#include <mfobjects.h>
#include <mfreadwrite.h>

#include <cstddef>
#include <cstring>
#include <new>  // std::bad_alloc
#include <string>
#include <vector>

#include "bridge_core.h"  // ComputeThumbnailSize (pure sizing logic)
#include "media_fps_probe.h"  // EnsureMediaFoundation (contract v17 video path)

namespace nzvideomni {

namespace {

// Minimal RAII for a COM interface pointer (avoids pulling in wil here so this
// module stays buildable in the plain test executable).
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
    for (size_t i = 0; i < dir.size(); ++i) {
        if ((dir[i] == L'\\' || dir[i] == L'/') && i > 0) {
            ::CreateDirectoryW(dir.substr(0, i).c_str(), nullptr);
        }
    }
    ::CreateDirectoryW(dir.c_str(), nullptr);
}

}  // namespace

bool EncodeRgbaToPngFile(const std::wstring& dest_path, const unsigned char* rgba,
                         int width, int height, std::string* err_message) {
    auto fail = [err_message](const char* msg) -> bool {
        if (err_message != nullptr) {
            *err_message = msg;
        }
        return false;
    };

    if (rgba == nullptr || width <= 0 || height <= 0) {
        return fail("invalid pixel buffer or dimensions");
    }

    EnsureParentDir(dest_path);

    // WIC lives in the multi-threaded apartment for our worker thread. S_FALSE
    // means COM was already initialised on this thread; both must be balanced.
    const HRESULT co = ::CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool need_uninit = SUCCEEDED(co);
    bool ok = false;
    std::string err = "unknown WIC failure";

    do {
        ComPtr<IWICImagingFactory> factory;
        HRESULT hr = ::CoCreateInstance(CLSID_WICImagingFactory, nullptr,
                                        CLSCTX_INPROC_SERVER,
                                        IID_PPV_ARGS(factory.put()));
        if (FAILED(hr) || !factory) {
            err = "CoCreateInstance(WICImagingFactory) failed";
            break;
        }

        ComPtr<IWICStream> stream;
        hr = factory->CreateStream(stream.put());
        if (FAILED(hr) || !stream) {
            err = "IWICImagingFactory::CreateStream failed";
            break;
        }
        hr = stream->InitializeFromFilename(dest_path.c_str(), GENERIC_WRITE);
        if (FAILED(hr)) {
            err = "IWICStream::InitializeFromFilename failed";
            break;
        }

        ComPtr<IWICBitmapEncoder> encoder;
        hr = factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, encoder.put());
        if (FAILED(hr) || !encoder) {
            err = "CreateEncoder(PNG) failed";
            break;
        }
        hr = encoder->Initialize(stream.get(), WICBitmapEncoderNoCache);
        if (FAILED(hr)) {
            err = "IWICBitmapEncoder::Initialize failed";
            break;
        }

        ComPtr<IWICBitmapFrameEncode> frame;
        ComPtr<IPropertyBag2> props;
        hr = encoder->CreateNewFrame(frame.put(), props.put());
        if (FAILED(hr) || !frame) {
            err = "IWICBitmapEncoder::CreateNewFrame failed";
            break;
        }
        hr = frame->Initialize(props.get());
        if (FAILED(hr)) {
            err = "IWICBitmapFrameEncode::Initialize failed";
            break;
        }
        hr = frame->SetSize(static_cast<UINT>(width), static_cast<UINT>(height));
        if (FAILED(hr)) {
            err = "IWICBitmapFrameEncode::SetSize failed";
            break;
        }

        // The PNG encoder's native 32-bit format is 32bppBGRA, not RGBA, so the
        // frame is asked for BGRA and our RGBA source is run through a WIC format
        // converter (a channel swap) rather than written raw.
        WICPixelFormatGUID format = GUID_WICPixelFormat32bppBGRA;
        hr = frame->SetPixelFormat(&format);
        if (FAILED(hr)) {
            err = "IWICBitmapFrameEncode::SetPixelFormat failed";
            break;
        }

        // Wrap our tightly-packed RGBA bytes as a source bitmap. WIC copies the
        // pixels, so `rgba` need not outlive this call.
        const UINT stride = static_cast<UINT>(width) * 4u;
        const UINT buffer_size = stride * static_cast<UINT>(height);
        ComPtr<IWICBitmap> source;
        hr = factory->CreateBitmapFromMemory(
            static_cast<UINT>(width), static_cast<UINT>(height),
            GUID_WICPixelFormat32bppRGBA, stride, buffer_size,
            const_cast<BYTE*>(rgba), source.put());
        if (FAILED(hr) || !source) {
            err = "IWICImagingFactory::CreateBitmapFromMemory failed";
            break;
        }

        ComPtr<IWICFormatConverter> converter;
        hr = factory->CreateFormatConverter(converter.put());
        if (FAILED(hr) || !converter) {
            err = "IWICImagingFactory::CreateFormatConverter failed";
            break;
        }
        hr = converter->Initialize(source.get(), format, WICBitmapDitherTypeNone,
                                   nullptr, 0.0, WICBitmapPaletteTypeCustom);
        if (FAILED(hr)) {
            err = "IWICFormatConverter::Initialize failed";
            break;
        }

        hr = frame->WriteSource(converter.get(), nullptr);
        if (FAILED(hr)) {
            err = "IWICBitmapFrameEncode::WriteSource failed";
            break;
        }
        hr = frame->Commit();
        if (FAILED(hr)) {
            err = "IWICBitmapFrameEncode::Commit failed";
            break;
        }
        hr = encoder->Commit();
        if (FAILED(hr)) {
            err = "IWICBitmapEncoder::Commit failed";
            break;
        }
        ok = true;
    } while (false);

    if (need_uninit) {
        ::CoUninitialize();
    }
    if (!ok) {
        return fail(err.c_str());
    }
    return true;
}

namespace {

// Shared tail of both thumbnail makers: scale `source` (src_w x src_h) so its
// long edge is at most `max_dim`, convert to 24bppBGR and JPEG-encode (quality
// 0.85) into *out_bytes. The caller owns COM and the factory. On failure sets
// *err and returns false. (Split out of MakeJpegThumbnail for contract v17's
// video path; the error strings are the ones MakeJpegThumbnail always used.)
bool EncodeSourceToJpegThumbnail(IWICImagingFactory* factory,
                                 IWICBitmapSource* source, int src_w, int src_h,
                                 int max_dim, std::vector<unsigned char>* out_bytes,
                                 int* out_dst_w, int* out_dst_h, std::string* err) {
    int dst_w = 0;
    int dst_h = 0;
    ComputeThumbnailSize(src_w, src_h, max_dim, &dst_w, &dst_h);
    if (dst_w <= 0 || dst_h <= 0) {
        *err = "computed thumbnail size is empty";
        return false;
    }
    bool ok = false;
    do {
        HRESULT hr = S_OK;
        // Scale (if downsizing) then convert to the JPEG-native 24bppBGR (drops
        // any alpha, which JPEG cannot store). When dst == src the scaler is a
        // pass-through, so it is always safe to route through it.
        ComPtr<IWICBitmapScaler> scaler;
        hr = factory->CreateBitmapScaler(scaler.put());
        if (FAILED(hr) || !scaler) {
            *err = "IWICImagingFactory::CreateBitmapScaler failed";
            break;
        }
        hr = scaler->Initialize(source, static_cast<UINT>(dst_w),
                                static_cast<UINT>(dst_h),
                                WICBitmapInterpolationModeFant);
        if (FAILED(hr)) {
            *err = "IWICBitmapScaler::Initialize failed";
            break;
        }

        WICPixelFormatGUID format = GUID_WICPixelFormat24bppBGR;
        ComPtr<IWICFormatConverter> converter;
        hr = factory->CreateFormatConverter(converter.put());
        if (FAILED(hr) || !converter) {
            *err = "IWICImagingFactory::CreateFormatConverter failed";
            break;
        }
        hr = converter->Initialize(scaler.get(), format, WICBitmapDitherTypeNone,
                                   nullptr, 0.0, WICBitmapPaletteTypeCustom);
        if (FAILED(hr)) {
            *err = "IWICFormatConverter::Initialize failed";
            break;
        }

        // Encode to an in-memory stream so the caller can base64 it.
        ComPtr<IStream> mem_stream;
        hr = ::CreateStreamOnHGlobal(nullptr, TRUE, mem_stream.put());
        if (FAILED(hr) || !mem_stream) {
            *err = "CreateStreamOnHGlobal failed";
            break;
        }

        ComPtr<IWICBitmapEncoder> encoder;
        hr = factory->CreateEncoder(GUID_ContainerFormatJpeg, nullptr,
                                    encoder.put());
        if (FAILED(hr) || !encoder) {
            *err = "CreateEncoder(JPEG) failed";
            break;
        }
        hr = encoder->Initialize(mem_stream.get(), WICBitmapEncoderNoCache);
        if (FAILED(hr)) {
            *err = "IWICBitmapEncoder::Initialize failed";
            break;
        }

        ComPtr<IWICBitmapFrameEncode> frame;
        ComPtr<IPropertyBag2> props;
        hr = encoder->CreateNewFrame(frame.put(), props.put());
        if (FAILED(hr) || !frame) {
            *err = "IWICBitmapEncoder::CreateNewFrame failed";
            break;
        }
        // Request JPEG quality 0.85 before initialising the frame. A missing
        // property is non-fatal (older encoders), so failures here are ignored.
        if (props) {
            PROPBAG2 option = {};
            option.pstrName = const_cast<LPOLESTR>(L"ImageQuality");
            VARIANT value;
            ::memset(&value, 0, sizeof(value));
            value.vt = VT_R4;
            value.fltVal = 0.85f;
            props->Write(1, &option, &value);
        }
        hr = frame->Initialize(props.get());
        if (FAILED(hr)) {
            *err = "IWICBitmapFrameEncode::Initialize failed";
            break;
        }
        hr = frame->SetSize(static_cast<UINT>(dst_w), static_cast<UINT>(dst_h));
        if (FAILED(hr)) {
            *err = "IWICBitmapFrameEncode::SetSize failed";
            break;
        }
        hr = frame->SetPixelFormat(&format);
        if (FAILED(hr)) {
            *err = "IWICBitmapFrameEncode::SetPixelFormat failed";
            break;
        }
        hr = frame->WriteSource(converter.get(), nullptr);
        if (FAILED(hr)) {
            *err = "IWICBitmapFrameEncode::WriteSource failed";
            break;
        }
        hr = frame->Commit();
        if (FAILED(hr)) {
            *err = "IWICBitmapFrameEncode::Commit failed";
            break;
        }
        hr = encoder->Commit();
        if (FAILED(hr)) {
            *err = "IWICBitmapEncoder::Commit failed";
            break;
        }

        // Pull the encoded bytes out of the HGLOBAL backing the stream.
        HGLOBAL hglobal = nullptr;
        hr = ::GetHGlobalFromStream(mem_stream.get(), &hglobal);
        if (FAILED(hr) || hglobal == nullptr) {
            *err = "GetHGlobalFromStream failed";
            break;
        }
        const SIZE_T size = ::GlobalSize(hglobal);
        void* ptr = ::GlobalLock(hglobal);
        if (ptr == nullptr || size == 0) {
            if (ptr != nullptr) {
                ::GlobalUnlock(hglobal);
            }
            *err = "encoded JPEG stream is empty";
            break;
        }
        const unsigned char* bytes = static_cast<const unsigned char*>(ptr);
        out_bytes->assign(bytes, bytes + size);
        ::GlobalUnlock(hglobal);
        ok = true;
    } while (false);
    if (ok) {
        *out_dst_w = dst_w;
        *out_dst_h = dst_h;
    }
    return ok;
}

}  // namespace

bool MakeJpegThumbnail(const std::wstring& src_path, int max_dim,
                       std::vector<unsigned char>* out_bytes, int* out_width,
                       int* out_height, int* out_source_width,
                       int* out_source_height, std::string* err_message) {
    auto fail = [err_message](const char* msg) -> bool {
        if (err_message != nullptr) {
            *err_message = msg;
        }
        return false;
    };

    if (out_bytes == nullptr || out_width == nullptr || out_height == nullptr ||
        out_source_width == nullptr || out_source_height == nullptr) {
        return fail("invalid output pointers");
    }
    out_bytes->clear();
    *out_width = *out_height = *out_source_width = *out_source_height = 0;
    if (max_dim <= 0) {
        return fail("invalid max_dim");
    }

    const HRESULT co = ::CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool need_uninit = SUCCEEDED(co);
    bool ok = false;
    std::string err = "unknown WIC failure";
    int src_w = 0;
    int src_h = 0;
    int dst_w = 0;
    int dst_h = 0;

    do {
        ComPtr<IWICImagingFactory> factory;
        HRESULT hr = ::CoCreateInstance(CLSID_WICImagingFactory, nullptr,
                                        CLSCTX_INPROC_SERVER,
                                        IID_PPV_ARGS(factory.put()));
        if (FAILED(hr) || !factory) {
            err = "CoCreateInstance(WICImagingFactory) failed";
            break;
        }

        ComPtr<IWICBitmapDecoder> decoder;
        hr = factory->CreateDecoderFromFilename(
            src_path.c_str(), nullptr, GENERIC_READ,
            WICDecodeMetadataCacheOnDemand, decoder.put());
        if (FAILED(hr) || !decoder) {
            err = "CreateDecoderFromFilename failed (unsupported or unreadable)";
            break;
        }

        ComPtr<IWICBitmapFrameDecode> src_frame;
        hr = decoder->GetFrame(0, src_frame.put());
        if (FAILED(hr) || !src_frame) {
            err = "IWICBitmapDecoder::GetFrame failed";
            break;
        }

        UINT uw = 0;
        UINT uh = 0;
        hr = src_frame->GetSize(&uw, &uh);
        if (FAILED(hr) || uw == 0 || uh == 0) {
            err = "IWICBitmapFrameDecode::GetSize failed";
            break;
        }
        src_w = static_cast<int>(uw);
        src_h = static_cast<int>(uh);
        ok = EncodeSourceToJpegThumbnail(factory.get(), src_frame.get(), src_w,
                                         src_h, max_dim, out_bytes, &dst_w,
                                         &dst_h, &err);
    } while (false);

    if (need_uninit) {
        ::CoUninitialize();
    }
    if (!ok) {
        return fail(err.c_str());
    }
    *out_width = dst_w;
    *out_height = dst_h;
    *out_source_width = src_w;
    *out_source_height = src_h;
    return true;
}

namespace {

// Read the first decodable video frame of `path` with a Media Foundation Source
// Reader (video processing on, RGB32 out) and copy its DISPLAY area into a
// tightly-packed top-down BGRX buffer (*out_px, stride = *out_w * 4). COM and
// MFStartup must already be done by the caller. On failure sets *err.
//
// Row layout: the buffer is read with IMFMediaBuffer::Lock only (valid for 2D
// buffers too, just less efficient - one path, no IMF2DBuffer branch). The
// signed stride comes from MF_MT_DEFAULT_STRIDE, else from
// MFGetStrideForBitmapInfoHeader(RGB32, width), else w*4 top-down; a negative
// stride means bottom-up (top row at the end of the buffer).
//
// Frames larger than 8192x8192 are refused, and an allocation failure (here or
// in WIC) is an ordinary failure rather than an exception on the HTTP worker.
//
// Display area: H.264 decodes in 16-pixel macroblocks, so a 1080-line clip can
// come out as 1088 lines whose bottom 8 are padding. The area is, in order:
// MF_MT_MINIMUM_DISPLAY_APERTURE of the decoded type, the same attribute of the
// file's native type, or the native type's MF_MT_FRAME_SIZE (anchored at the
// top-left) - each clamped to the decoded frame. Rotation and pixel aspect
// ratio are not applied (a thumbnail shows the stored pixels).
bool ReadFirstVideoFrameBgrx(const std::wstring& path,
                             std::vector<unsigned char>* out_px, int* out_w,
                             int* out_h, std::string* err) {
    ComPtr<IMFAttributes> attrs;
    HRESULT hr = ::MFCreateAttributes(attrs.put(), 1);
    if (FAILED(hr) || !attrs) {
        *err = "MFCreateAttributes failed";
        return false;
    }
    // Lets the reader insert the software YUV -> RGB32 converter. Microsoft
    // documents it for "a small number of frames, such as a thumbnail".
    attrs->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);

    ComPtr<IMFSourceReader> reader;
    hr = ::MFCreateSourceReaderFromURL(path.c_str(), attrs.get(), reader.put());
    if (FAILED(hr) || !reader) {
        *err = "MFCreateSourceReaderFromURL failed (unsupported or unreadable)";
        return false;
    }

    const DWORD kVideo = static_cast<DWORD>(MF_SOURCE_READER_FIRST_VIDEO_STREAM);
    // Only the video stream is read; audio would just be decoded and dropped.
    reader->SetStreamSelection(static_cast<DWORD>(MF_SOURCE_READER_ALL_STREAMS), FALSE);
    hr = reader->SetStreamSelection(kVideo, TRUE);
    if (FAILED(hr)) {
        *err = "no video stream";
        return false;
    }

    ComPtr<IMFMediaType> native_type;
    hr = reader->GetNativeMediaType(
        kVideo, static_cast<DWORD>(MF_SOURCE_READER_CURRENT_TYPE_INDEX),
        native_type.put());
    if (FAILED(hr) || !native_type) {
        *err = "no native video media type";
        return false;
    }

    {
        ComPtr<IMFMediaType> want;
        hr = ::MFCreateMediaType(want.put());
        if (FAILED(hr) || !want) {
            *err = "MFCreateMediaType failed";
            return false;
        }
        want->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        want->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
        hr = reader->SetCurrentMediaType(kVideo, nullptr, want.get());
        if (FAILED(hr)) {
            *err = "SetCurrentMediaType(RGB32) failed (no decoder for this codec)";
            return false;
        }
    }

    // Pull until the first non-empty sample (decoders may emit format-change
    // events or empty ticks first). The guard bounds a stream that never yields.
    ComPtr<IMFSample> sample;
    for (int guard = 0; guard < 256; ++guard) {
        DWORD stream_index = 0;
        DWORD flags = 0;
        LONGLONG ts = 0;
        hr = reader->ReadSample(kVideo, 0, &stream_index, &flags, &ts, sample.put());
        if (FAILED(hr)) {
            *err = "IMFSourceReader::ReadSample failed";
            return false;
        }
        if ((flags & MF_SOURCE_READERF_ERROR) != 0) {
            *err = "IMFSourceReader::ReadSample reported a stream error";
            return false;
        }
        if (sample) break;
        if ((flags & MF_SOURCE_READERF_ENDOFSTREAM) != 0) break;
    }
    if (!sample) {
        *err = "no decodable video frame";
        return false;
    }

    // The decoded type is read AFTER the first sample: the decoder may only
    // settle its output size (e.g. 1080 -> 1088) once it has seen the stream.
    ComPtr<IMFMediaType> cur_type;
    hr = reader->GetCurrentMediaType(kVideo, cur_type.put());
    if (FAILED(hr) || !cur_type) {
        *err = "IMFSourceReader::GetCurrentMediaType failed";
        return false;
    }
    UINT32 fw = 0;
    UINT32 fh = 0;
    hr = ::MFGetAttributeSize(cur_type.get(), MF_MT_FRAME_SIZE, &fw, &fh);
    if (FAILED(hr) || fw == 0 || fh == 0 || fw > 8192 || fh > 8192) {
        *err = "decoded MF_MT_FRAME_SIZE is missing or out of range";
        return false;
    }
    const int frame_w = static_cast<int>(fw);
    const int frame_h = static_cast<int>(fh);

    // Display area (see the comment above the function).
    int ax = 0;
    int ay = 0;
    int aw = frame_w;
    int ah = frame_h;
    {
        MFVideoArea area = {};
        UINT32 got = 0;
        bool have_area =
            SUCCEEDED(cur_type->GetBlob(MF_MT_MINIMUM_DISPLAY_APERTURE,
                                        reinterpret_cast<UINT8*>(&area),
                                        sizeof(area), &got)) &&
            got == sizeof(area);
        if (!have_area) {
            got = 0;
            have_area =
                SUCCEEDED(native_type->GetBlob(MF_MT_MINIMUM_DISPLAY_APERTURE,
                                               reinterpret_cast<UINT8*>(&area),
                                               sizeof(area), &got)) &&
                got == sizeof(area);
        }
        if (have_area && area.Area.cx > 0 && area.Area.cy > 0) {
            ax = area.OffsetX.value;
            ay = area.OffsetY.value;
            aw = static_cast<int>(area.Area.cx);
            ah = static_cast<int>(area.Area.cy);
        } else {
            UINT32 nw = 0;
            UINT32 nh = 0;
            if (SUCCEEDED(::MFGetAttributeSize(native_type.get(), MF_MT_FRAME_SIZE,
                                               &nw, &nh)) &&
                nw > 0 && nh > 0) {
                aw = static_cast<int>(nw);
                ah = static_cast<int>(nh);
            }
        }
        // Clamp into the decoded frame.
        if (ax < 0 || ax >= frame_w) ax = 0;
        if (ay < 0 || ay >= frame_h) ay = 0;
        if (aw > frame_w - ax) aw = frame_w - ax;
        if (ah > frame_h - ay) ah = frame_h - ay;
        if (aw <= 0 || ah <= 0) {
            ax = ay = 0;
            aw = frame_w;
            ah = frame_h;
        }
    }

    ComPtr<IMFMediaBuffer> buf;
    hr = sample->ConvertToContiguousBuffer(buf.put());
    if (FAILED(hr) || !buf) {
        *err = "IMFSample::ConvertToContiguousBuffer failed";
        return false;
    }

    const size_t row_bytes = static_cast<size_t>(aw) * 4u;
    out_px->assign(row_bytes * static_cast<size_t>(ah), 0);  // bad_alloc: caught by the caller

    // Signed stride: MF_MT_DEFAULT_STRIDE, else what Media Foundation computes
    // for RGB32 at this width, else w*4 top-down. Negative = bottom-up.
    LONG stride = 0;
    {
        UINT32 raw = 0;
        if (SUCCEEDED(cur_type->GetUINT32(MF_MT_DEFAULT_STRIDE, &raw)) && raw != 0) {
            stride = static_cast<LONG>(raw);
        } else if (FAILED(::MFGetStrideForBitmapInfoHeader(
                       MFVideoFormat_RGB32.Data1, fw, &stride)) ||
                   stride == 0) {
            stride = static_cast<LONG>(frame_w) * 4;
        }
    }
    const LONG abs_stride = stride < 0 ? -stride : stride;
    if (abs_stride < static_cast<LONG>(frame_w) * 4) {
        *err = "video stride is narrower than the frame";
        return false;
    }

    BYTE* data = nullptr;
    DWORD len = 0;
    hr = buf->Lock(&data, nullptr, &len);
    if (FAILED(hr) || data == nullptr) {
        *err = "IMFMediaBuffer::Lock failed";
        return false;
    }
    const size_t needed = static_cast<size_t>(abs_stride) * static_cast<size_t>(frame_h);
    const bool enough = static_cast<size_t>(len) >= needed;
    if (enough) {
        // `top` is the first displayed row. With a negative stride (bottom-up)
        // it is the LAST row in memory, and `stride` then steps upward from
        // there. Copy rows [ay, ay+ah) starting at column ax.
        const BYTE* top = stride < 0
                              ? data + static_cast<size_t>(abs_stride) * (frame_h - 1)
                              : data;
        for (int y = 0; y < ah; ++y) {
            const BYTE* src = top + static_cast<ptrdiff_t>(ay + y) * stride +
                              static_cast<ptrdiff_t>(ax) * 4;
            ::memcpy(out_px->data() + static_cast<size_t>(y) * row_bytes, src,
                     row_bytes);
        }
    }
    buf->Unlock();
    if (!enough) {
        *err = "decoded frame buffer is smaller than its declared size";
        return false;
    }
    *out_w = aw;
    *out_h = ah;
    return true;
}

}  // namespace

bool MakeJpegThumbnailFromVideo(const std::wstring& src_path, int max_dim,
                                std::vector<unsigned char>* out_bytes,
                                int* out_width, int* out_height,
                                int* out_source_width, int* out_source_height,
                                std::string* err_message) {
    auto fail = [err_message](const char* msg) -> bool {
        if (err_message != nullptr) {
            *err_message = msg;
        }
        return false;
    };

    if (out_bytes == nullptr || out_width == nullptr || out_height == nullptr ||
        out_source_width == nullptr || out_source_height == nullptr) {
        return fail("invalid output pointers");
    }
    out_bytes->clear();
    *out_width = *out_height = *out_source_width = *out_source_height = 0;
    if (max_dim <= 0) {
        return fail("invalid max_dim");
    }

    // Same COM handling as MakeJpegThumbnail: the HTTP worker thread does not
    // initialise COM itself, so take an MTA ref for this call and balance it.
    const HRESULT co = ::CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool need_uninit = SUCCEEDED(co);
    bool ok = false;
    std::string err = "unknown Media Foundation failure";
    int src_w = 0;
    int src_h = 0;
    int dst_w = 0;
    int dst_h = 0;

    // A huge frame can make the pixel copy (or WIC / the JPEG bytes) fail to
    // allocate; that must be THUMBNAIL_FAILED, not std::terminate on the HTTP
    // worker, so the whole read-and-encode runs under one catch. ComPtrs inside
    // are released by unwinding before CoUninitialize below.
    try {
    do {
        if (!EnsureMediaFoundation()) {
            err = "MFStartup failed";
            break;
        }
        std::vector<unsigned char> px;
        if (!ReadFirstVideoFrameBgrx(src_path, &px, &src_w, &src_h, &err)) {
            break;
        }

        ComPtr<IWICImagingFactory> factory;
        HRESULT hr = ::CoCreateInstance(CLSID_WICImagingFactory, nullptr,
                                        CLSCTX_INPROC_SERVER,
                                        IID_PPV_ARGS(factory.put()));
        if (FAILED(hr) || !factory) {
            err = "CoCreateInstance(WICImagingFactory) failed";
            break;
        }
        // 32bppBGR = RGB32's memory order (B, G, R, X); the X byte is ignored.
        // WIC copies the pixels, so `px` need not outlive the bitmap.
        const UINT stride = static_cast<UINT>(src_w) * 4u;
        ComPtr<IWICBitmap> bitmap;
        hr = factory->CreateBitmapFromMemory(
            static_cast<UINT>(src_w), static_cast<UINT>(src_h),
            GUID_WICPixelFormat32bppBGR, stride, static_cast<UINT>(px.size()),
            px.data(), bitmap.put());
        if (FAILED(hr) || !bitmap) {
            err = "IWICImagingFactory::CreateBitmapFromMemory failed";
            break;
        }
        ok = EncodeSourceToJpegThumbnail(factory.get(), bitmap.get(), src_w, src_h,
                                         max_dim, out_bytes, &dst_w, &dst_h, &err);
    } while (false);
    } catch (const std::bad_alloc&) {
        ok = false;
        err = "out of memory while making the video thumbnail";
    }

    if (need_uninit) {
        ::CoUninitialize();
    }
    if (!ok) {
        out_bytes->clear();
        return fail(err.c_str());
    }
    *out_width = dst_w;
    *out_height = dst_h;
    *out_source_width = src_w;
    *out_source_height = src_h;
    return true;
}

}  // namespace nzvideomni
