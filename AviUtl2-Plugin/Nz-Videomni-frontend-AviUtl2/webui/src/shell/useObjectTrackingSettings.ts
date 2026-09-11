import { useCallback, useEffect, useState } from "react";
import {
  OBJECT_TRACKING_DEFAULTS,
  readStoredObjectTracking,
  writeStoredObjectTracking,
} from "./objectTrackingSettings";
import type { LostBehavior, ObjectTrackingSettings } from "./objectTrackingSettings";

export interface UseObjectTrackingSettingsResult {
  /** The current settings. Unlike `useAccelerationSettings`, there is no
   * "effective" fold on top of the stored object: no tracking knob is gated on
   * a server capability (the `/status.tracking` block answers "can this server
   * track at all", which disables the whole section rather than rewriting one
   * field), so what is stored IS what is sent. */
  tracking: ObjectTrackingSettings;
  setSearchFactor: (value: number) => void;
  setLostScoreThreshold: (value: number) => void;
  setLostBehavior: (value: LostBehavior) => void;
  setSmoothing: (value: number) => void;
  setFollowSize: (value: boolean) => void;
  setKeyframeStride: (value: number) => void;
}

/**
 * Object-tracking settings (§3-54, 2026-09-11), owned by `AppShell` and called
 * exactly ONCE — the same D1 arrangement `useAccelerationSettings` and
 * `useNagSettings` document, and for the same reason: a per-screen copy would
 * silently reset every time a right-click remounts the Toolbox screen, and the
 * `trackObject` RPC would then go out with defaults the user never chose.
 *
 * One `useState` with a lazy initializer (so the `localStorage` read happens
 * once, not on every render) plus functional updates in every setter (so none
 * captures a stale value). The WRITE lives in a `useEffect` keyed off the six
 * fields rather than inside the setters, for the reason
 * `useAccelerationSettings` spells out: `localStorage` is I/O, and React may
 * invoke a `setState` updater more than once per commit (StrictMode's
 * double-invoke), so synchronizing with a system outside React belongs in an
 * effect.
 *
 * These are persisted — not session-local like NAG's per-generation parameters
 * — because they describe how the user wants THIS KIND OF WORK done ("my
 * subjects move fast, search wider"), which is a standing preference rather
 * than a property of one clip.
 */
export function useObjectTrackingSettings(): UseObjectTrackingSettingsResult {
  const [tracking, setTracking] = useState<ObjectTrackingSettings>(() => readStoredObjectTracking());

  useEffect(() => {
    writeStoredObjectTracking({
      searchFactor: tracking.searchFactor,
      lostScoreThreshold: tracking.lostScoreThreshold,
      lostBehavior: tracking.lostBehavior,
      smoothing: tracking.smoothing,
      followSize: tracking.followSize,
      keyframeStride: tracking.keyframeStride,
    });
  }, [
    tracking.searchFactor,
    tracking.lostScoreThreshold,
    tracking.lostBehavior,
    tracking.smoothing,
    tracking.followSize,
    tracking.keyframeStride,
  ]);

  const setSearchFactor = useCallback((value: number) => {
    setTracking((prev) => ({ ...prev, searchFactor: value }));
  }, []);

  const setLostScoreThreshold = useCallback((value: number) => {
    setTracking((prev) => ({ ...prev, lostScoreThreshold: value }));
  }, []);

  const setLostBehavior = useCallback((value: LostBehavior) => {
    setTracking((prev) => ({ ...prev, lostBehavior: value }));
  }, []);

  const setSmoothing = useCallback((value: number) => {
    setTracking((prev) => ({ ...prev, smoothing: value }));
  }, []);

  const setFollowSize = useCallback((value: boolean) => {
    setTracking((prev) => ({ ...prev, followSize: value }));
  }, []);

  const setKeyframeStride = useCallback((value: number) => {
    // A range `<input>` can only ever hand back a number, but an empty number
    // box yields `NaN` — which native rejects. Fall back to the default rather
    // than letting it through, the same posture the stored-value reader takes.
    //
    // Rounded because native rejects a FRACTIONAL stride outright rather than
    // rounding it (`bridge_core.cpp`'s `ParseTrackObject`), and the number box
    // will happily accept `1.5` whatever its `step` says. Rounding here rather
    // than refusing keeps the typed value close to what was meant; the panel's
    // own clamp still holds the range, so this only ever moves it within it.
    setTracking((prev) => ({
      ...prev,
      keyframeStride: Number.isFinite(value)
        ? Math.round(value)
        : OBJECT_TRACKING_DEFAULTS.keyframeStride,
    }));
  }, []);

  return {
    tracking,
    setSearchFactor,
    setLostScoreThreshold,
    setLostBehavior,
    setSmoothing,
    setFollowSize,
    setKeyframeStride,
  };
}
