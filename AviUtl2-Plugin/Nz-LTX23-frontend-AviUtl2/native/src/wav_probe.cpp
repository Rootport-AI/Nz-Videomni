// wav_probe.cpp - implementation. See wav_probe.h for the contract.
#include "wav_probe.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <cstring>

namespace nzltx {

namespace {

// Little-endian field readers. Callers must have already bounds-checked that
// [offset, offset + N) lies within the buffer.
std::uint16_t ReadU16LE(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0]) | (static_cast<std::uint16_t>(p[1]) << 8);
}

std::uint32_t ReadU32LE(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

bool TagIs(const std::uint8_t* p, const char* tag) {
    return std::memcmp(p, tag, 4) == 0;
}

constexpr std::uint16_t kWaveFormatPcm = 1;
constexpr std::uint16_t kWaveFormatIeeeFloat = 3;

}  // namespace

std::optional<double> WavDurationSeconds(const std::uint8_t* data, size_t size) {
    // "RIFF" (4) + size (4) + "WAVE" (4) = 12 bytes minimum before any chunk.
    if (size < 12 || data == nullptr) {
        return std::nullopt;
    }
    if (!TagIs(data, "RIFF") || !TagIs(data + 8, "WAVE")) {
        return std::nullopt;
    }

    bool found_fmt = false;
    std::uint16_t audio_format = 0;
    std::uint16_t num_channels = 0;
    std::uint32_t sample_rate = 0;
    std::uint16_t bits_per_sample = 0;

    bool found_data = false;
    std::uint32_t data_chunk_size = 0;

    size_t offset = 12;
    // Each chunk header is 8 bytes (4-byte id + 4-byte little-endian size).
    while (offset + 8 <= size) {
        const std::uint8_t* chunk_id = data + offset;
        const std::uint32_t chunk_size = ReadU32LE(data + offset + 4);
        const size_t chunk_data_offset = offset + 8;

        if (TagIs(chunk_id, "fmt ")) {
            // Need the 16 canonical PCM/IEEE-float fields, fully present in
            // the buffer - never read past `size` even if chunk_size claims
            // there is more (truncated-file safety).
            if (chunk_size < 16 || chunk_data_offset + 16 > size) {
                return std::nullopt;
            }
            const std::uint8_t* f = data + chunk_data_offset;
            audio_format = ReadU16LE(f + 0);
            num_channels = ReadU16LE(f + 2);
            sample_rate = ReadU32LE(f + 4);
            // f + 8 (byteRate, u32) and f + 12 (blockAlign, u16) are part of
            // the canonical layout but are not needed for the duration
            // computation below (recomputed from channels/bitsPerSample).
            bits_per_sample = ReadU16LE(f + 14);
            if (audio_format != kWaveFormatPcm && audio_format != kWaveFormatIeeeFloat) {
                return std::nullopt;  // extensible / unsupported format tag
            }
            found_fmt = true;
        } else if (TagIs(chunk_id, "data")) {
            if (!found_fmt) {
                return std::nullopt;  // 'data' before 'fmt ' - malformed
            }
            // Only the 8-byte chunk header needs to be in the buffer: the
            // declared size is trusted as-is, since a header-only probe
            // (ProbeWavFile) intentionally never loads the audio payload.
            data_chunk_size = chunk_size;
            found_data = true;
            break;
        }
        // Skip this chunk's data (RIFF chunks are word-aligned: an odd-sized
        // chunk has one pad byte after it). size_t is 64-bit on this target,
        // so widening a uint32 chunk_size here cannot overflow the addition.
        const size_t advance = 8 + static_cast<size_t>(chunk_size) +
                               (static_cast<size_t>(chunk_size) % 2);
        offset += advance;
    }

    if (!found_fmt || !found_data) {
        return std::nullopt;
    }
    if (sample_rate == 0 || num_channels == 0 || bits_per_sample == 0) {
        return std::nullopt;
    }

    const std::uint32_t bytes_per_sample = bits_per_sample / 8;
    if (bytes_per_sample == 0) {
        return std::nullopt;
    }
    const std::uint32_t block_align = static_cast<std::uint32_t>(num_channels) * bytes_per_sample;
    if (block_align == 0) {
        return std::nullopt;
    }

    const double nframes = static_cast<double>(data_chunk_size / block_align);
    return nframes / static_cast<double>(sample_rate);
}

std::optional<double> ProbeWavFile(const std::wstring& path) {
    HANDLE h = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return std::nullopt;
    }
    std::string buf(kWavProbeHeaderCapBytes, '\0');
    DWORD read = 0;
    const BOOL ok = ::ReadFile(h, buf.data(), static_cast<DWORD>(buf.size()), &read, nullptr);
    ::CloseHandle(h);
    if (!ok || read == 0) {
        return std::nullopt;
    }
    return WavDurationSeconds(reinterpret_cast<const std::uint8_t*>(buf.data()),
                              static_cast<size_t>(read));
}

}  // namespace nzltx
