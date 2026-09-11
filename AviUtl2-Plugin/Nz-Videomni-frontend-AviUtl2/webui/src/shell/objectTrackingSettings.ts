/**
 * Object-tracking (物体追尾) settings — the six persisted knobs the Toolbox
 * tab's tracking panel sends with every `timeline.trackObject` call (§3-54,
 * 2026-09-11; design doc `Nz-Videomni/Docs/OBJECT_TRACKING_DESIGN.md` §3.1).
 * This module is the pure, UI-free layer: the types, the defaults, and the
 * `localStorage` read/write — modelled directly on `shell/accelerationSettings.
 * ts`, which is the house pattern for "a settings object AppShell owns once and
 * hands down as props" (`shell/nagSettings.ts` documents the D1 arrangement).
 *
 * ## Why these live in the WebUI and not in `config.yaml`
 *
 * Native holds NO defaults for any of them (`bridge_core.cpp`'s
 * `ParseTrackObject` requires all eight fields and range-checks each), and the
 * backend never sees them at all — it answers "frame in, raw box + score out",
 * and every knob here is applied by the plugin's post-processing
 * (`native/src/track_postprocess.cpp`). So this file is the ONLY source of the
 * values, and the ranges below must stay in step with the native parser's.
 *
 * ## The seventh UI item is not here
 *
 * The panel shows SEVEN controls; only six are stored. The model radio
 * (`UETrack` / `Other`) is a permanently disabled mock — a seat kept visible
 * for a second tracker — so there is no choice to persist and no field for it
 * in {@link ObjectTrackingSettings}. Storing it would invent a preference the
 * user cannot actually express.
 */

import type { StatusResponse } from "../api/types";

/** What a lost frame does to the written path. Mirrors native's
 * `lostBehavior` literal (`bridge/types.ts`'s `timeline.trackObject` params):
 *  - `"hold"` freezes the last good box until the tracker recovers, so the
 *    filter stays put rather than wandering off with a bad guess;
 *  - `"continue"` writes the tracker's raw box anyway, which is what you want
 *    when the object really did keep moving behind an occlusion. */
export type LostBehavior = "hold" | "continue";

/** The tracking panel's persisted state — one object owned by `AppShell` and
 * handed to the Toolbox screen, the same single-owner arrangement
 * `AccelerationSettings` uses (a per-screen copy would silently reset on every
 * right-click remount's `key` bump). */
export interface ObjectTrackingSettings {
  /** How far around the current box the tracker looks in the next frame, as a
   * multiple of the box. Sent as `searchFactor`. This is the ONE knob the
   * backend sees (it opens the session with it); everything else below is the
   * plugin's own post-processing. */
  searchFactor: number;
  /** Frames whose tracker score falls below this are treated as lost. Sent as
   * `lostScoreThreshold`. */
  lostScoreThreshold: number;
  /** What a lost frame does — see {@link LostBehavior}. Sent as
   * `lostBehavior`. */
  lostBehavior: LostBehavior;
  /** Exponential smoothing of the path, `0` = the tracker's raw boxes. Sent as
   * `smoothing`. */
  smoothing: number;
  /** When false, only the position follows and the box keeps the seed's size.
   * Sent as `followSize`. */
  followSize: boolean;
  /** Keep one keyframe every N frames; the first and last are always kept
   * whatever this says (native's post-processing guarantees that, not this
   * value). Sent as `keyframeStride`. Must be a whole number — native REJECTS
   * a fractional one rather than rounding it. */
  keyframeStride: number;
}

/** Slider bounds, shared by the panel's `<input>`s, by
 * {@link readStoredObjectTracking}'s range check, and by the tests — one
 * declaration so a widened slider cannot silently outgrow the validation that
 * is supposed to guard it. These MIRROR native's own accepted ranges
 * (`bridge_core.cpp`'s `ParseTrackObject`), except `searchFactor`, where the
 * panel deliberately offers 2.0–6.0 of the server's wider 8.0 ceiling
 * (design doc §3.1). */
export const SEARCH_FACTOR_MIN = 2.0;
export const SEARCH_FACTOR_MAX = 6.0;
export const SEARCH_FACTOR_STEP = 0.1;
export const LOST_THRESHOLD_MIN = 0;
export const LOST_THRESHOLD_MAX = 1;
export const LOST_THRESHOLD_STEP = 0.01;
export const SMOOTHING_MIN = 0;
export const SMOOTHING_MAX = 1;
export const SMOOTHING_STEP = 0.05;
export const KEYFRAME_STRIDE_MIN = 1;
export const KEYFRAME_STRIDE_MAX = 10;
export const KEYFRAME_STRIDE_STEP = 1;

/** `localStorage` key every tracking choice is persisted under. Named for the
 * SECTION rather than any field, exactly as `ACCELERATION_STORAGE_KEY` is, so
 * a seventh knob never forces a storage-key rename on existing installs. */
export const OBJECT_TRACKING_STORAGE_KEY = "nzvideomni.objectTracking";

/** The starting values (design doc §3.1). `lostScoreThreshold` is the one that
 * is still provisional: 0.35 came out of the M1 spike's score distribution
 * (good frames clustered near 0.79 median / 0.41 p10, occlusions down at a 0.57
 * median) and is expected to be re-pinned once the real-hardware measurement
 * lands. Frozen so an accidental in-place mutation of this shared reference
 * fails immediately, matching `ACCELERATION_DEFAULTS`. */
export const OBJECT_TRACKING_DEFAULTS: Readonly<ObjectTrackingSettings> = Object.freeze({
  searchFactor: 4.0,
  lostScoreThreshold: 0.35,
  lostBehavior: "hold" as LostBehavior,
  smoothing: 0.3,
  followSize: true,
  keyframeStride: 1,
});

/** Pure: a finite number inside `[min, max]`, else the default. Used for every
 * numeric field below, so an out-of-range stored value (a hand-edited
 * `localStorage`, or a build whose slider range has since narrowed) can never
 * reach native — which would answer `BAD_REQUEST` minutes into the user's
 * workflow rather than at read time. */
function numberInRange(value: unknown, min: number, max: number, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  return value >= min && value <= max ? value : fallback;
}

/**
 * Reads the persisted tracking choices, falling back FIELD BY FIELD.
 *
 * Wrapped in a `try` because `localStorage` can throw outright in a restricted
 * embedding (a WebView2 host with storage disabled), where the defaults are
 * plainly better than a crash — the same posture `readStoredAcceleration` and
 * `ThemeContext`'s `readStoredTheme` take.
 *
 * Per-field, rather than "validate the whole blob or discard it", for the
 * reason that pattern exists here: a blob written by an older build simply has
 * no key for a knob added later, and one bad field is no reason to throw away
 * five good ones. EVERY field added to {@link ObjectTrackingSettings} needs its
 * own line here — without one the whole-object write below would persist an
 * `undefined` that reads back as the default forever.
 */
export function readStoredObjectTracking(): ObjectTrackingSettings {
  let raw: string | null = null;
  try {
    raw = window.localStorage.getItem(OBJECT_TRACKING_STORAGE_KEY);
  } catch {
    return { ...OBJECT_TRACKING_DEFAULTS };
  }
  if (raw === null) return { ...OBJECT_TRACKING_DEFAULTS };
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { ...OBJECT_TRACKING_DEFAULTS };
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return { ...OBJECT_TRACKING_DEFAULTS };
  }
  const rec = parsed as Record<string, unknown>;
  const behavior = rec.lostBehavior;
  const stride = rec.keyframeStride;
  return {
    searchFactor: numberInRange(
      rec.searchFactor,
      SEARCH_FACTOR_MIN,
      SEARCH_FACTOR_MAX,
      OBJECT_TRACKING_DEFAULTS.searchFactor,
    ),
    lostScoreThreshold: numberInRange(
      rec.lostScoreThreshold,
      LOST_THRESHOLD_MIN,
      LOST_THRESHOLD_MAX,
      OBJECT_TRACKING_DEFAULTS.lostScoreThreshold,
    ),
    lostBehavior:
      behavior === "hold" || behavior === "continue" ? behavior : OBJECT_TRACKING_DEFAULTS.lostBehavior,
    smoothing: numberInRange(
      rec.smoothing,
      SMOOTHING_MIN,
      SMOOTHING_MAX,
      OBJECT_TRACKING_DEFAULTS.smoothing,
    ),
    followSize:
      typeof rec.followSize === "boolean" ? rec.followSize : OBJECT_TRACKING_DEFAULTS.followSize,
    // The one field with an extra condition: native rejects a fractional
    // stride outright (it would mean the caller computed it wrong), so a
    // stored `1.5` has to become the default here rather than travel.
    keyframeStride: Number.isInteger(stride)
      ? numberInRange(stride, KEYFRAME_STRIDE_MIN, KEYFRAME_STRIDE_MAX, OBJECT_TRACKING_DEFAULTS.keyframeStride)
      : OBJECT_TRACKING_DEFAULTS.keyframeStride,
  };
}

/** Persists the tracking choices as JSON. Wrapped in a `try` for
 * {@link readStoredObjectTracking}'s reason; a failed write is swallowed
 * (matching `writeStoredAcceleration`) — the session keeps working, it just
 * starts from the defaults next time. */
export function writeStoredObjectTracking(value: ObjectTrackingSettings): void {
  try {
    window.localStorage.setItem(OBJECT_TRACKING_STORAGE_KEY, JSON.stringify(value));
  } catch {
    // Ignore — localStorage unavailable (e.g. private mode).
  }
}

/** The `tracking` block to believe right now, given the `/status` body in hand
 * (`null` when the current server state carries none) and the last block that
 * was observed.
 *
 * Pure and last-value-carrying, because the states that carry no body are
 * ordinary and long: `checking` before the first poll lands, `offline` and
 * `error` while the backend is restarting, and a `loading-models` raised by
 * this WebUI's own `POST /pipeline/load` (`status: null` — there is no body
 * behind that claim). Tracking is installed or it is not; none of those
 * windows is news about it, so the last observation stands. Falling back to
 * "unavailable" for them would put "run install-UETrack.bat" in front of a
 * user whose tracking is installed and working, and refuse the 追尾 right-click
 * for the duration.
 *
 * A body that HAS arrived always wins, including when its `tracking` is
 * absent: that is a real observation (a server too old to report the block, or
 * one that stopped reporting it), not a gap. The caller keeps the last value
 * in a ref and feeds it back in — see `AppShell`'s `lastTrackingRef`. */
export function resolveTrackingStatus(
  status: Pick<StatusResponse, "tracking"> | null | undefined,
  lastSeen: StatusResponse["tracking"],
): StatusResponse["tracking"] {
  return status ? status.tracking : lastSeen;
}
