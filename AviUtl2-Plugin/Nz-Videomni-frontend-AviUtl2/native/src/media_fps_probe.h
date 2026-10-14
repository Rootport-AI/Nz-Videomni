// media_fps_probe.h - read a video file's native frame rate with Media
// Foundation (bridge contract v11 / section 3-13: timeline.getSelection's
// mediaFps, which feeds the right-click prefill "match the material" fps axis;
// and contract v15: fs.probeMediaInfo's fps).
//
// --- What it reads -----------------------------------------------------------
// An IMFSourceReader is opened over the file and the FIRST VIDEO STREAM's
// NATIVE media type - the container's own declaration, so no decoder is created
// and not a single frame is decoded - is queried for MF_MT_FRAME_RATE, a UINT32
// numerator/denominator pair. Only the container header is parsed, so the cost
// does not grow with the file's length.
//
// --- Raw fps, no rounding ----------------------------------------------------
// The value is returned RAW: an NTSC clip reports 30000/1001 = 29.97..., not 30.
// mediaFps is a "fact about the material" field like mediaWidth/mediaHeight, so
// the snap-to-integer policy (29.97 -> 30, 23.976 -> 24) belongs to the webui's
// prefill layer (webui/src/.../prefillSeed.ts), not here.
//
// --- Matroska is reported as unknown, by rule ---------------------------------
// Windows DOES open Matroska: the "MKV Byte Stream Handler" (mfmkvsrcsnk.dll) is
// registered for .mkv / .webm / .mk3d (and the audio-only .mka / .mks / .weba).
// But its native video type declares HALF the real rate: measured 2026-10-14 on
// Windows 11 (mfmkvsrcsnk.dll 10.0.26100.9549), a 29.97 fps file reports
// MF_MT_FRAME_RATE 1498502247/100000000 (14.985) and 30 / 24 fps files report
// 15 / 12 - for H.264 with or without B-frames, whatever the SPS timing says,
// and for VP9 too, while the same H.264 stream in .mp4 / .mov reads 30000/1001.
// No other attribute of that type carries the right rate (the range attributes
// and MF_MT_INTERLACE_MODE are unset). A wrong rate does more harm than none, so
// ProbeMediaFps returns false for a Matroska extension WITHOUT opening Media
// Foundation, and the webui treats the file like any other unknown fps (the
// right-click prefill falls back to the project fps; Alpha Gen shows "unknown"
// and lets the server measure). Every Matroska file is treated this way, even
// one Media Foundation might read correctly, because the two cannot be told
// apart. VERIFICATION_LOG section 154 has the measurement table.
//
// --- A failed probe is a NORMAL outcome --------------------------------------
// Matroska (above), a missing file, an audio-only file or a codec Media
// Foundation has no type for all end in false, and the webui simply falls back
// as described above. The plugin therefore DISCARDS `err`: logging it would
// turn an everyday outcome into log noise. `err` exists so the unit test can
// tell "the probe failed" apart from "the probe returned a nonsense number" (and
// which rule made it fail), and so a future diagnostic build has something to
// print.
//
// --- COM / MF lifecycle ------------------------------------------------------
// ProbeMediaFps runs on AviUtl2's UI thread (inside call_edit_section_param),
// which is an STA, so it asks for COINIT_APARTMENTTHREADED and tolerates both
// S_FALSE (already initialised on this thread - still ref-counted, so the ref is
// balanced) and RPC_E_CHANGED_MODE (the host already chose the other apartment -
// no ref taken, so none is released). CoUninitialize is called only for a ref we
// actually took.
//
// MFStartup is done ONCE per process (std::call_once) and the matching
// MFShutdown is deliberately NEVER called. Two reasons:
//   1. The plugin's lifetime IS the process lifetime. Media Foundation is torn
//      down by the OS at process exit anyway, so there is nothing that leaks for
//      any observable length of time.
//   2. MFStartup/MFShutdown are internally ref-counted. A Startup/Shutdown round
//      trip on every right-click would be pure overhead, and worse, it could
//      drop the refcount to zero while an Mp4Writer (which also starts MF, on an
//      HTTP worker thread) is mid-encode.
//
// Failures are reported as `bool` + a short ASCII diagnostic via `err`, matching
// wic_png / mf_mp4_writer; nothing throws. All comments are ASCII/English.
#pragma once

#include <cstdint>
#include <string>

namespace nzvideomni {

// Frames per second for an MF_MT_FRAME_RATE numerator/denominator pair.
//
// Returns 0.0 when either component is 0 (a media type that declares no usable
// frame rate), which is also SelectionItem::media_fps's "unknown" value. Both
// inputs are UINT32, so a division with two non-zero components is always
// finite - the two zero checks are exhaustive. Pure function; its doctest cases
// live in native/tests/test_media_fps_probe.cpp.
double FpsFromRatio(std::uint32_t num, std::uint32_t den);

// Probe `path` (native wide path) for its video stream's native frame rate.
//
// On success writes the RAW fps (29.97 stays 29.97 - see above) to *fps_out and
// returns true. Returns false, leaving *fps_out untouched, when: `path` is empty
// or `fps_out` is null; the extension is a Matroska one (.mkv / .webm / .mk3d,
// any case - Media Foundation is not opened, see above); COM/MF cannot be
// initialised; no installed Media Foundation source can open the file (missing
// file - a NORMAL outcome); the file has no video stream Media Foundation can
// type; or its media type declares no usable MF_MT_FRAME_RATE. `err` may be
// nullptr; when non-null it is set to a short ASCII reason on failure and left
// untouched on success. Never throws.
bool ProbeMediaFps(const std::wstring& path, double* fps_out, std::string* err);

}  // namespace nzvideomni
