// timeline_math.h - pure generation-length pre-computation for Nz-LTX23.
//
// Given a generate request (single clip, chain, or vid2vid continuation) this
// module answers, WITHOUT touching the backend, "how many pixel frames will the
// finished mp4 be, how many seconds is that, and how long a provisional object
// should be placed on the AviUtl2 timeline (in the PROJECT fps)". It is a
// faithful pure-C++ replica of the backend single-source-of-truth geometry in
//   Nz-LTX23-backend/chain_math.py   (v_latent_frames / px_from_v_latent /
//                                     compute_chain_layout / a_frames_for_px)
//   Nz-LTX23-backend/api/models.py   (the 8n+1 / range / cap validators)
// so the timeline placeholder matches the delivered video frame-for-frame.
//
// No AviUtl2 SDK, WebView2, WinHTTP or Win32 dependency: unit-tested with doctest
// in isolation (see native/tests/test_timeline_math.cpp). Invalid input is
// reported through the result structs' TimelineError code (never thrown), so the
// caller can surface a user-facing message. All comments are ASCII/English.
#pragma once

#include <string>
#include <vector>

namespace nzltx {

// --- Latent geometry constants (mirror chain_math.py) ----------------------
inline constexpr int kVideoTimeFactor = 8;         // temporal VAE compression
inline constexpr double kAudioLatentsPerSec = 25.0;  // 16000/160/4

// --- Single-clip request bounds (mirror api/models.py GenerateRequest) ------
inline constexpr int kMinNumFrames = 9;            // 8*1 + 1
inline constexpr int kMaxNumFrames = 481;          // 8*60 + 1 (20s @ 24fps)
inline constexpr double kDefaultGenerationFps = 24.0;
inline constexpr double kMinFrameRate = 1.0;
inline constexpr double kMaxFrameRate = 60.0;

// --- Chain request bounds (mirror GenerateChainRequest / chain_math) --------
inline constexpr int kMaxChainClips = 8;
inline constexpr int kMinOverlapFrames = 1;        // K_v
inline constexpr int kMaxOverlapFrames = 8;
inline constexpr int kMaxChainTotalPixelFrames = 8 * 481;  // 3848 (sanity cap)

// Stage-2 tile size in video-latent frames (chain_math.STAGE2_V_TILE). The
// vid2vid frozen context head must fit inside stage-2 tile 0, i.e.
// v_latent_frames(source_context_px) <= kStage2VTile.
//
// NOTE (§1-14/§3-57, 2026-08-09): 22 is the "standard" stage-2 window, which is
// still the server default and the only geometry this plugin can produce. The
// WebUI may now opt a chain into the narrower "high_resolution" window
// (GenerateChainRequest.stage2_window -> chain_math.STAGE2_WINDOW_PRESETS =
// 19 latents), whose context ceiling is 137 rather than 161. That choice lives
// entirely in the WebUI form, which does its own client-side check
// (webui/src/shell/tokenBudget.ts's stage2MaxContextFrames) and the server
// 422s past it either way. Do NOT make this constant track that setting: the
// plugin has no window control, and 22 is the correct bound for everything it
// itself emits.
// NOTE (§1-19, 2026-08-12): a third preset "full_length" (61,61) also exists in
// chain_math.STAGE2_WINDOW_PRESETS now. It is a fixed value used only by
// Single/Batch's a2v request builders (one clip, whole-length stage-2, no
// tiling); the plugin does not know about it and this constant does not track
// it either.
inline constexpr int kStage2VTile = 22;
// px_from_v_latent(22) = 169 -> largest 8n+1 source context that FITS tile 0 on
// the standard window; the UI config layer bounds context_frames further to
// [25, 145] (SourceVideoSpec), NOT enforced here (this replicates chain_math's
// invariant).
inline constexpr int kV2VContextConfigMin = 25;
inline constexpr int kV2VContextConfigMax = 145;

// --- Join (continuation<->source) audio crossfade (JoinRequest) -------------
inline constexpr int kDefaultHandleCrossfadeMs = 300;
inline constexpr int kMaxHandleCrossfadeMs = 2000;

// Error codes for the result structs below (kNone == success).
enum class TimelineError {
    kNone = 0,
    kNumFramesRange,                  // num_frames outside [9, 481]
    kNumFramesNotAligned,             // (num_frames - 1) % 8 != 0
    kFrameRateRange,                  // frame_rate outside [1, 60]
    kNoClips,                         // chain with zero clips
    kTooManyClips,                    // chain with > 8 clips
    kOverlapRange,                    // overlap_frames outside [1, 8]
    kOverlapTooLarge,                 // K_v >= some clip's stage-1 latent frames
    kAudioDegenerate,                 // clips too short for a continuous xfade
    kChainTooLong,                    // total_px > kMaxChainTotalPixelFrames
    kSourceContextRange,              // source_context_px < 1
    kSourceContextNotAligned,         // source_context_px not 8n+1
    kSourceContextTooLong,            // frozen head exceeds stage-2 tile 0
    kSourceContextNotLessThanClip0,   // context_px >= clip 0 -> no new tail
};

// Stable ASCII message for an error code (empty string for kNone).
const char* TimelineErrorMessage(TimelineError e);

// --- 8n+1 helpers -----------------------------------------------------------

// True when num_frames is a valid single-clip count: 8n+1 and within [9, 481].
bool IsValidNumFrames(int num_frames);

// Snap num_frames to the nearest valid 8n+1 value in [9, 481]. Ties (exactly
// between two grid points) round up. Values below/above the range clamp to
// kMinNumFrames / kMaxNumFrames.
int NearestValidNumFrames(int num_frames);

// Video latent-frame count for a pixel-frame span: (P - 1) / 8 + 1.
int VLatentFrames(int pixel_frames);

// Inverse of VLatentFrames: pixel frames (8n+1) for N video latent frames.
int PxFromVLatent(int n_latent);

// Audio latent-frame count for a pixel span at fps: round(P / fps * 25), using
// Python's round-half-to-even so it matches chain_math.a_frames_for_px exactly
// (only the chain audio-degenerate check depends on this). fps must be > 0.
int AudioLatentFrames(int pixel_frames, double fps);

// --- Generation fps <-> project fps -----------------------------------------

// Convert a pixel-frame span produced at gen_fps into the number of frames that
// span occupies on a timeline running at project_fps, preserving DURATION:
//   project_frames = round(pixel_frames * project_fps / gen_fps)   (ties up).
// This is the length a provisional object should be given so it covers exactly
// the same wall-clock duration as the finished video. gen_fps must be > 0;
// a non-positive gen_fps yields 0.
int ProjectFramesForPixels(int pixel_frames, double gen_fps, double project_fps);

// Convert a duration in SECONDS into the number of frames it occupies on a
// timeline running at project_fps, preserving DURATION:
//   project_frames = round(seconds * project_fps)   (ties up).
// Used to size an inserted media object to its real length (get_media_info's
// MEDIA_INFO.total_time, project fps = rate/scale). A non-positive or NaN
// project_fps/seconds yields 0 (caller falls back to the host's auto length).
// An out-of-int-range product (a corrupt/huge total_time, or +inf) is clamped
// to 2000000000 so the double->int cast never hits undefined behavior.
int ProjectFramesForSeconds(double seconds, double project_fps);

// --- Timeline span occupancy (provisional collision pre-check, D2) ----------

// True when two INCLUSIVE integer frame spans overlap, i.e. share at least one
// frame. The provisional-insert collision pre-check (bridge.cpp's
// HasRoomForLength) uses this to decide, BEFORE calling create_object_from_alias,
// whether the requested [frame, frame+length-1] span is free on a layer -
// needed because create_object_from_alias with length 0 silently SHORTENS the
// object to fit a smaller gap instead of failing (the G1 real-device finding),
// so the old "create returned null -> fall back" path never fired. Pure integer
// geometry, doctest-tested; the SDK object scan around it is real-device
// verified. A span whose start > end is treated as empty (never overlaps).
bool SpansOverlap(int a_start, int a_end, int b_start, int b_end);

// --- Single clip (txt2vid / img2vid) ----------------------------------------

struct SingleClipResult {
    TimelineError error = TimelineError::kNone;
    int pixel_frames = 0;   // output video pixel frames (== num_frames)
    double seconds = 0.0;   // num_frames / frame_rate
};

// Pre-compute a single-clip generate. Validates num_frames (range + 8n+1) and
// frame_rate ([1, 60]); on error the numeric fields are 0.
SingleClipResult ComputeSingleClip(int num_frames, double frame_rate);

// --- Chain (clip concatenation), optionally vid2vid --------------------------

struct ChainInput {
    std::vector<int> clip_frames;             // per-clip num_frames (each 8n+1)
    double frame_rate = kDefaultGenerationFps;
    int overlap_frames = 3;                   // K_v (video latent overlap)
    bool has_source_context = false;          // vid2vid continuation source
    int source_context_px = 0;                // frozen source tail (8n+1)
};

struct ChainResult {
    TimelineError error = TimelineError::kNone;
    int f_total_latent = 0;      // assembled stage-1 video latent frames
    int total_px = 0;            // full decoded timeline pixel frames (context incl.)
    int delivered_px = 0;        // pixel frames actually delivered in the mp4
                                 //   = total_px - source_context_px (vid2vid),
                                 //     else == total_px
    double total_seconds = 0.0;  // total_px / frame_rate (whole decoded timeline)
    double delivered_seconds = 0.0;  // delivered_px / frame_rate (== placeholder len)
};

// Pre-compute a chain (>=1 clip). Mirrors chain_math.compute_chain_layout,
// including the K_v-fits check, the audio crossfade degeneracy check, the total
// timeline cap, and (when has_source_context) the vid2vid frozen-context
// invariants. Validation order matches the backend so the first surfaced error
// is the same one the backend would raise. On error the numeric fields are 0.
ChainResult ComputeChain(const ChainInput& in);

// --- Join (server-side continuation onto uploaded source) --------------------
//
// APPROXIMATE / partially unresolved. The join endpoint (POST /jobs/{id}/join)
// hard-concats the source video's frames with the continuation's frames, so the
// joined VIDEO track length is simply source_frames + continuation_frames. The
// handle_crossfade_ms (default 300 ms, [0, 2000]) is an AUDIO-ONLY equal-power
// crossfade at the junction and does NOT shorten the video track, so it is not
// reflected in the frame count here. The uploaded source's own length is not
// part of a generate request, so the caller must supply source_frames; this is
// why the estimate lives outside ComputeChain. See report note on join.

// True when a handle crossfade duration is within [0, kMaxHandleCrossfadeMs].
bool IsValidHandleCrossfadeMs(int ms);

struct JoinLengthEstimate {
    int joined_frames = 0;      // source_frames + continuation_frames (video: hard concat)
    double joined_seconds = 0.0;
};

// Approximate joined length: video is a hard concat (audio crossfade does not
// change the frame count). Negative inputs are treated as 0. fps must be > 0.
JoinLengthEstimate EstimateJoinLength(int source_frames, int continuation_frames,
                                      double fps);

}  // namespace nzltx
