// wic_png.h - encode a tightly-packed 32-bit RGBA buffer to a PNG file via WIC.
//
// The AviUtl2 rendering callback hands us a PIXEL_RGBA buffer, whose bytes are
// laid out r, g, b, a per pixel (filter2.h: `struct PIXEL_RGBA { unsigned char
// r, g, b, a; }`, documented as DXGI_FORMAT_R8G8B8A8_UNORM). WIC's PNG encoder's
// native 32-bit format is 32bppBGRA, so the buffer is wrapped as a 32bppRGBA
// source bitmap and run through a WIC format converter (a channel swap) into the
// frame. Alpha is treated as straight (unassociated) alpha, which PNG stores.
//
// The encoder manages COM for its own scope: it calls CoInitializeEx with
// COINIT_MULTITHREADED on the calling (worker) thread and balances it before
// returning, so callers do not need COM initialised. All comments are ASCII.
#pragma once

#include <string>
#include <vector>

namespace nzltx {

// Encode `width` x `height` pixels of tightly-packed RGBA (4 bytes/pixel, no row
// padding: stride == width*4) to a PNG at `dest_path` (native wide path). Parent
// directories are created as needed. Returns true on success; on failure returns
// false and sets *err_message to a short ASCII diagnostic.
bool EncodeRgbaToPngFile(const std::wstring& dest_path, const unsigned char* rgba,
                         int width, int height, std::string* err_message);

// Decode the image at `src_path` (any WIC-supported codec: PNG/JPEG/WebP/...),
// scale it (aspect-preserving) so its long edge is at most `max_dim` (no upscale
// when it already fits), and JPEG-encode (quality 0.85) the result into
// *out_bytes. The scaled dimensions are written to *out_width / *out_height and
// the original dimensions to *out_source_width / *out_source_height. `max_dim`
// should already be clamped by the caller. Returns true on success; on failure
// returns false and sets *err_message to a short ASCII diagnostic (used to map
// to THUMBNAIL_FAILED). Like EncodeRgbaToPngFile, this manages COM for its own
// scope, so it is safe to call from any worker thread.
bool MakeJpegThumbnail(const std::wstring& src_path, int max_dim,
                       std::vector<unsigned char>* out_bytes, int* out_width,
                       int* out_height, int* out_source_width,
                       int* out_source_height, std::string* err_message);

}  // namespace nzltx
