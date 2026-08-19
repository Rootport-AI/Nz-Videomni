// test_wav_probe.cpp - unit tests for the pure RIFF/WAVE header parser.
// All fixtures are hand-built byte strings; no real files are touched here
// (ProbeWavFile's disk-I/O wrapper is exercised in test_fs_util.cpp's spirit
// via a temp file instead, see the dedicated file-based test case below).
#include "doctest.h"

#include <cstdint>
#include <string>

#include "wav_probe.h"

using namespace nzvideomni;

namespace {

void AppendU16LE(std::string* s, std::uint16_t v) {
    s->push_back(static_cast<char>(v & 0xFF));
    s->push_back(static_cast<char>((v >> 8) & 0xFF));
}

void AppendU32LE(std::string* s, std::uint32_t v) {
    s->push_back(static_cast<char>(v & 0xFF));
    s->push_back(static_cast<char>((v >> 8) & 0xFF));
    s->push_back(static_cast<char>((v >> 16) & 0xFF));
    s->push_back(static_cast<char>((v >> 24) & 0xFF));
}

// Builds a minimal, well-formed "RIFF....WAVEfmt data" byte string.
//   channels/sample_rate/bits_per_sample: canonical fmt fields.
//   audio_format: 1 == PCM, 3 == IEEE float, 0xFFFE == extensible.
//   data_bytes: raw payload bytes for the 'data' chunk (actually appended, so
//   the buffer is fully self-contained - a realistic whole-file load).
std::string MakeWav(std::uint16_t channels, std::uint32_t sample_rate,
                    std::uint16_t bits_per_sample, std::uint16_t audio_format,
                    const std::string& data_bytes) {
    std::string fmt_chunk;
    AppendU16LE(&fmt_chunk, audio_format);
    AppendU16LE(&fmt_chunk, channels);
    AppendU32LE(&fmt_chunk, sample_rate);
    const std::uint32_t byte_rate =
        sample_rate * channels * (bits_per_sample / 8);
    AppendU32LE(&fmt_chunk, byte_rate);
    AppendU16LE(&fmt_chunk, static_cast<std::uint16_t>(channels * (bits_per_sample / 8)));
    AppendU16LE(&fmt_chunk, bits_per_sample);
    REQUIRE(fmt_chunk.size() == 16);

    std::string body;  // everything after "RIFF"+size, i.e. starting at "WAVE"
    body += "WAVE";
    body += "fmt ";
    AppendU32LE(&body, static_cast<std::uint32_t>(fmt_chunk.size()));
    body += fmt_chunk;
    body += "data";
    AppendU32LE(&body, static_cast<std::uint32_t>(data_bytes.size()));
    body += data_bytes;

    std::string out;
    out += "RIFF";
    AppendU32LE(&out, static_cast<std::uint32_t>(body.size()));
    out += body;
    return out;
}

const std::uint8_t* Bytes(const std::string& s) {
    return reinterpret_cast<const std::uint8_t*>(s.data());
}

}  // namespace

TEST_CASE("WavDurationSeconds: normal 16-bit PCM mono") {
    // 4 sample frames (8 bytes) at 44100 Hz, 16-bit mono -> 4/44100 s.
    const std::string data(8, '\x11');
    const std::string wav = MakeWav(1, 44100, 16, 1, data);
    const std::optional<double> d = WavDurationSeconds(Bytes(wav), wav.size());
    REQUIRE(d.has_value());
    CHECK(*d == doctest::Approx(4.0 / 44100.0));
}

TEST_CASE("WavDurationSeconds: IEEE float stereo is supported") {
    // 2 channels * 4 bytes (32-bit float) = 8 bytes/frame; 16 bytes -> 2 frames.
    const std::string data(16, '\x00');
    const std::string wav = MakeWav(2, 48000, 32, 3, data);
    const std::optional<double> d = WavDurationSeconds(Bytes(wav), wav.size());
    REQUIRE(d.has_value());
    CHECK(*d == doctest::Approx(2.0 / 48000.0));
}

TEST_CASE("WavDurationSeconds: extensible format tag is rejected") {
    const std::string data(8, '\x00');
    const std::string wav = MakeWav(1, 44100, 16, 0xFFFE, data);
    CHECK_FALSE(WavDurationSeconds(Bytes(wav), wav.size()).has_value());
}

TEST_CASE("WavDurationSeconds: sampleRate 0 yields nullopt") {
    const std::string data(8, '\x00');
    const std::string wav = MakeWav(1, 0, 16, 1, data);
    CHECK_FALSE(WavDurationSeconds(Bytes(wav), wav.size()).has_value());
}

TEST_CASE("WavDurationSeconds: fmt chunk missing yields nullopt") {
    // "RIFF" + size + "WAVE" + only a 'data' chunk, no 'fmt '.
    std::string body = "WAVE";
    body += "data";
    const std::string payload(8, '\x00');
    AppendU32LE(&body, static_cast<std::uint32_t>(payload.size()));
    body += payload;
    std::string wav = "RIFF";
    AppendU32LE(&wav, static_cast<std::uint32_t>(body.size()));
    wav += body;

    CHECK_FALSE(WavDurationSeconds(Bytes(wav), wav.size()).has_value());
}

TEST_CASE("WavDurationSeconds: data chunk missing yields nullopt") {
    // "RIFF" + size + "WAVE" + only a 'fmt ' chunk, no 'data'.
    std::string fmt_chunk;
    AppendU16LE(&fmt_chunk, 1);
    AppendU16LE(&fmt_chunk, 1);
    AppendU32LE(&fmt_chunk, 44100);
    AppendU32LE(&fmt_chunk, 44100 * 2);
    AppendU16LE(&fmt_chunk, 2);
    AppendU16LE(&fmt_chunk, 16);

    std::string body = "WAVE";
    body += "fmt ";
    AppendU32LE(&body, static_cast<std::uint32_t>(fmt_chunk.size()));
    body += fmt_chunk;
    std::string wav = "RIFF";
    AppendU32LE(&wav, static_cast<std::uint32_t>(body.size()));
    wav += body;

    CHECK_FALSE(WavDurationSeconds(Bytes(wav), wav.size()).has_value());
}

TEST_CASE("WavDurationSeconds: declared fmt size exceeds the truncated buffer") {
    // The 'fmt ' chunk header CLAIMS a valid 16-byte payload, but the buffer
    // itself is chopped off after only a few of those bytes - a truncated
    // file. Must return nullopt, not read out of bounds.
    std::string wav = "RIFF";
    std::string body = "WAVE";
    body += "fmt ";
    AppendU32LE(&body, 16);  // declares 16 bytes of fmt data
    body += std::string(5, '\x00');  // but only 5 are actually present
    AppendU32LE(&wav, static_cast<std::uint32_t>(body.size()));
    wav += body;

    CHECK_FALSE(WavDurationSeconds(Bytes(wav), wav.size()).has_value());
}

TEST_CASE("WavDurationSeconds: non-RIFF buffer yields nullopt") {
    const std::string not_riff = "This is not a RIFF file at all, ignore.";
    CHECK_FALSE(WavDurationSeconds(Bytes(not_riff), not_riff.size()).has_value());
}

TEST_CASE("WavDurationSeconds: empty / too-short buffer yields nullopt") {
    CHECK_FALSE(WavDurationSeconds(nullptr, 0).has_value());
    const std::string tiny = "RIFF";
    CHECK_FALSE(WavDurationSeconds(Bytes(tiny), tiny.size()).has_value());
}

TEST_CASE("WavDurationSeconds: data chunk's declared size may exceed the buffer") {
    // A header-only probe (ProbeWavFile) never loads the audio payload, so the
    // 'data' chunk's declared size legitimately exceeds what is physically in
    // the buffer. Only the 8-byte chunk header needs to be present.
    std::string fmt_chunk;
    AppendU16LE(&fmt_chunk, 1);
    AppendU16LE(&fmt_chunk, 1);
    AppendU32LE(&fmt_chunk, 44100);
    AppendU32LE(&fmt_chunk, 44100 * 2);
    AppendU16LE(&fmt_chunk, 2);
    AppendU16LE(&fmt_chunk, 16);

    std::string wav = "RIFF";
    std::string body = "WAVE";
    body += "fmt ";
    AppendU32LE(&body, static_cast<std::uint32_t>(fmt_chunk.size()));
    body += fmt_chunk;
    body += "data";
    AppendU32LE(&body, 44100 * 2 * 10);  // "10 seconds", but no payload follows
    AppendU32LE(&wav, static_cast<std::uint32_t>(body.size()));
    wav += body;

    const std::optional<double> d = WavDurationSeconds(Bytes(wav), wav.size());
    REQUIRE(d.has_value());
    CHECK(*d == doctest::Approx(10.0));
}

// ---------------------------------------------------------------------------
// ProbeWavFile - thin disk-I/O wrapper (kept minimal; the core logic above is
// exhaustively covered without touching the filesystem).
// ---------------------------------------------------------------------------

TEST_CASE("ProbeWavFile: a nonexistent path yields nullopt") {
    CHECK_FALSE(ProbeWavFile(L"Z:\\this\\path\\does\\not\\exist_nzvideomni.wav").has_value());
}
