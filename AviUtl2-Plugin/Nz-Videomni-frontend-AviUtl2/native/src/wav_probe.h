// wav_probe.h - pure RIFF/WAVE header parsing for Nz-Videomni (contract v6
// fs.probeAudioDuration / fs.listFiles' withAudioDuration).
//
// Reads only the RIFF chunk structure (never the audio samples themselves) to
// answer "how many seconds does this .wav play for". This mirrors the backend
// reference implementation, which uses Python's stdlib `wave` module purely to
// read `getnframes() / getframerate()`
// (Nz-Videomni/gradio_ui/manifest.py:271-284) - no sample data is ever
// decoded there either.
//
// Only PCM (audioFormat == 1) and IEEE-float (audioFormat == 3) 'fmt ' chunks
// are supported; WAVE_FORMAT_EXTENSIBLE (0xFFFE) and any other format tag are
// treated as a probe failure (std::nullopt), per contract: an unreadable /
// unsupported file resolves durationSec: 0 / isWav: false at the bridge layer,
// this module never throws.
//
// WavDurationSeconds is a pure function over an already-loaded byte buffer, so
// it is fully unit-testable with hand-built RIFF byte strings (see
// native/tests/test_wav_probe.cpp) without touching a real file. All chunk
// offsets are bounds-checked against `size` before every multi-byte read, so a
// truncated/malformed buffer can never cause an out-of-bounds access - it just
// yields std::nullopt. All comments are ASCII/English.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace nzvideomni {

// Parse a RIFF/WAVE byte buffer and compute its playback duration in seconds.
//
// Requires, in order: a "RIFF"..."WAVE" header, a 'fmt ' chunk (PCM or IEEE
// float, >=16 bytes, fully present within [data, data+size)) appearing before
// a 'data' chunk. Only the 'data' chunk's 8-byte header (id + declared size)
// needs to be present in the buffer - its declared size is trusted as-is and
// is NOT required to fit within `size`, since callers are expected to pass
// only a header-sized prefix of the file (see ProbeWavFile below), not the
// full audio payload.
//
// Returns std::nullopt when: the buffer is not a well-formed "RIFF....WAVE"
// header; no 'fmt ' chunk is found before a 'data' chunk (or none at all); the
// 'fmt ' chunk's declared size is too small (<16) or its fields are not fully
// present within the buffer (truncated input); the format is not PCM/IEEE
// float; sampleRate is 0; bitsPerSample or numChannels is 0; or no 'data'
// chunk is found. data may be nullptr only when size == 0.
std::optional<double> WavDurationSeconds(const std::uint8_t* data, size_t size);

// Header-prefix size ProbeWavFile reads from disk. Generous for the 'fmt '
// chunk plus any typical metadata (LIST/INFO/JUNK/etc.) chunks that precede
// 'data' in a real-world .wav; a file whose 'data' chunk header does not
// appear within this many bytes is reported as unreadable (std::nullopt).
inline constexpr size_t kWavProbeHeaderCapBytes = 64 * 1024;

// Read up to kWavProbeHeaderCapBytes from the start of the file at `path`
// (native wide path) and probe it with WavDurationSeconds. Returns
// std::nullopt if the file cannot be opened/read at all, or if the header
// prefix does not parse as a valid .wav (see WavDurationSeconds). Never
// throws; never reads more than kWavProbeHeaderCapBytes regardless of the
// file's actual size.
std::optional<double> ProbeWavFile(const std::wstring& path);

}  // namespace nzvideomni
