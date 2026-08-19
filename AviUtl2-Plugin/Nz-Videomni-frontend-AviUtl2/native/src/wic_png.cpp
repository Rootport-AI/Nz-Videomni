// wic_png.cpp - WIC PNG/JPEG encoder implementation. See wic_png.h. ASCII-only.
#include "wic_png.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <objidl.h>  // IStream, GetHGlobalFromStream
#include <wincodec.h>

#include <string>
#include <vector>

#include "bridge_core.h"  // ComputeThumbnailSize (pure sizing logic)

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
        ComputeThumbnailSize(src_w, src_h, max_dim, &dst_w, &dst_h);
        if (dst_w <= 0 || dst_h <= 0) {
            err = "computed thumbnail size is empty";
            break;
        }

        // Scale (if downsizing) then convert to the JPEG-native 24bppBGR (drops
        // any alpha, which JPEG cannot store). When dst == src the scaler is a
        // pass-through, so it is always safe to route through it.
        ComPtr<IWICBitmapScaler> scaler;
        hr = factory->CreateBitmapScaler(scaler.put());
        if (FAILED(hr) || !scaler) {
            err = "IWICImagingFactory::CreateBitmapScaler failed";
            break;
        }
        hr = scaler->Initialize(src_frame.get(), static_cast<UINT>(dst_w),
                                static_cast<UINT>(dst_h),
                                WICBitmapInterpolationModeFant);
        if (FAILED(hr)) {
            err = "IWICBitmapScaler::Initialize failed";
            break;
        }

        WICPixelFormatGUID format = GUID_WICPixelFormat24bppBGR;
        ComPtr<IWICFormatConverter> converter;
        hr = factory->CreateFormatConverter(converter.put());
        if (FAILED(hr) || !converter) {
            err = "IWICImagingFactory::CreateFormatConverter failed";
            break;
        }
        hr = converter->Initialize(scaler.get(), format, WICBitmapDitherTypeNone,
                                   nullptr, 0.0, WICBitmapPaletteTypeCustom);
        if (FAILED(hr)) {
            err = "IWICFormatConverter::Initialize failed";
            break;
        }

        // Encode to an in-memory stream so the caller can base64 it.
        ComPtr<IStream> mem_stream;
        hr = ::CreateStreamOnHGlobal(nullptr, TRUE, mem_stream.put());
        if (FAILED(hr) || !mem_stream) {
            err = "CreateStreamOnHGlobal failed";
            break;
        }

        ComPtr<IWICBitmapEncoder> encoder;
        hr = factory->CreateEncoder(GUID_ContainerFormatJpeg, nullptr,
                                    encoder.put());
        if (FAILED(hr) || !encoder) {
            err = "CreateEncoder(JPEG) failed";
            break;
        }
        hr = encoder->Initialize(mem_stream.get(), WICBitmapEncoderNoCache);
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
            err = "IWICBitmapFrameEncode::Initialize failed";
            break;
        }
        hr = frame->SetSize(static_cast<UINT>(dst_w), static_cast<UINT>(dst_h));
        if (FAILED(hr)) {
            err = "IWICBitmapFrameEncode::SetSize failed";
            break;
        }
        hr = frame->SetPixelFormat(&format);
        if (FAILED(hr)) {
            err = "IWICBitmapFrameEncode::SetPixelFormat failed";
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

        // Pull the encoded bytes out of the HGLOBAL backing the stream.
        HGLOBAL hglobal = nullptr;
        hr = ::GetHGlobalFromStream(mem_stream.get(), &hglobal);
        if (FAILED(hr) || hglobal == nullptr) {
            err = "GetHGlobalFromStream failed";
            break;
        }
        const SIZE_T size = ::GlobalSize(hglobal);
        void* ptr = ::GlobalLock(hglobal);
        if (ptr == nullptr || size == 0) {
            if (ptr != nullptr) {
                ::GlobalUnlock(hglobal);
            }
            err = "encoded JPEG stream is empty";
            break;
        }
        const unsigned char* bytes = static_cast<const unsigned char*>(ptr);
        out_bytes->assign(bytes, bytes + size);
        ::GlobalUnlock(hglobal);
        ok = true;
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

}  // namespace nzvideomni
