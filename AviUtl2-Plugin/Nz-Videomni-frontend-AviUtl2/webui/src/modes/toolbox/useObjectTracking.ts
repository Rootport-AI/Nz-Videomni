import { useCallback, useEffect, useRef, useState } from "react";
import { BridgeError, TIMELINE_TRACK_PROGRESS_EVENT, bridge as defaultBridge } from "../../bridge";
import type { NativeBridge, ResultOf, TimelineTrackProgressData } from "../../bridge";
import type { ObjectTrackingSettings } from "../../shell/objectTrackingSettings";

/**
 * The Toolbox tracking panel's whole runtime: fire `timeline.trackObject`,
 * follow its `timeline.trackProgress` pushes, derive a rate from their arrival
 * times, offer a stop, and land on one of three terminal states (§3-54,
 * 2026-09-11).
 *
 * ## Why a hook rather than the panel's own `useState`s
 *
 * Four pieces of state (phase, progress, result, error) move together and are
 * driven by two asynchronous sources (one promise, one event stream) — the
 * shape that reliably rots when it is spread across a component body. Keeping
 * it here also makes the ONE thing worth testing in isolation testable: that
 * the RPC goes out with the stored settings and `frame = frameStart`.
 *
 * ## Independence from generation (design doc §18)
 *
 * Nothing in here consults `serverBusy`, the reservation seat, or the job
 * ledger, and nothing in those consults this. Tracking runs on the CPU inside
 * a separate worker, so it neither waits for a generation nor blocks one —
 * this is the ONLY long operation in the app that is not part of the
 * one-job-at-a-time discipline, and that is deliberate, not an oversight.
 */

/** What `AppShell` hands the screen when a 追尾 right-click routes here: the
 * guarded selection snapshot, and nothing besides.
 *
 * Deliberately carries no discriminator (a timestamp, a counter) to make a
 * second right-click on the same object "look different". Nothing here watches
 * this object for changes: the run is fired ON MOUNT, and `AppShell` bumps
 * `remountTokens.toolbox` for every routed 追尾, so the remount IS the second
 * run's signal. */
export interface ObjectTrackRequest {
  selection: ResultOf<"timeline.getSelection">;
}

/** idle -> running -> (done | cancelled | error). There is no path back to
 * `idle`: the panel keeps the last run's numbers on screen until the next
 * right-click remounts the screen with a new request. */
export type ObjectTrackingPhase = "idle" | "running" | "done" | "cancelled" | "error";

export interface ObjectTrackingProgress {
  /** 1-based position in this run. */
  index: number;
  total: number;
  /** AviUtl2 absolute frame number (native already added the object's head). */
  frame: number;
  score: number;
  lost: boolean;
  /** Frames per second, measured by this hook from the progress events' own
   * arrival times — `null` until a second event gives something to measure
   * against (see {@link useObjectTracking}'s rate note). */
  fps: number | null;
}

export interface ObjectTrackingRunResult {
  frames: number;
  keyframes: number;
  elapsedMs: number;
  cancelled: boolean;
  /** AviUtl2 absolute frame numbers, both ends inclusive — passed through from
   * the RPC result verbatim. Native already works on that axis for everything
   * it sends the WebUI (contract v12), so there is nothing to convert here. */
  lostRanges: Array<{ start: number; end: number }>;
}

export interface UseObjectTrackingResult {
  phase: ObjectTrackingPhase;
  /** The latest progress push, or `null` before the first one. */
  progress: ObjectTrackingProgress | null;
  /** The finished run's numbers, present on `done` and on `cancelled` alike —
   * a stopped run still writes back what it tracked. */
  result: ObjectTrackingRunResult | null;
  /** The failure's bridge error code (`""` when the rejection carried none),
   * for the panel to look up its sentence. */
  errorCode: string;
  /** The failure's raw message, shown when the code has no sentence and is
   * itself empty. */
  errorMessage: string;
  /** Raise the stop flag. A no-op unless a run is in flight. */
  cancel: () => void;
}

export interface UseObjectTrackingParams {
  /** Test/integration seam, threaded down from `AppShell`; production omits it
   * and the app-wide singleton is used. */
  nativeBridge?: NativeBridge | undefined;
  /** The routed request. `undefined` = the screen was opened by hand (a tab
   * click), which starts nothing. */
  trackRequest?: ObjectTrackRequest | undefined;
  /** The stored settings, read ONCE at fire time (see the ref below). */
  settings: ObjectTrackingSettings;
  /** Test seam for the rate measurement. Defaults to `Date.now`. */
  now?: () => number;
}

export function useObjectTracking({
  nativeBridge,
  trackRequest,
  settings,
  now = Date.now,
}: UseObjectTrackingParams): UseObjectTrackingResult {
  const [phase, setPhase] = useState<ObjectTrackingPhase>("idle");
  const [progress, setProgress] = useState<ObjectTrackingProgress | null>(null);
  const [result, setResult] = useState<ObjectTrackingRunResult | null>(null);
  const [errorCode, setErrorCode] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  // The settings are read at FIRE TIME and never again: a run already in
  // flight is not re-parameterized by a slider the user nudges while watching
  // it. Mirroring them into a ref (rather than taking them as an effect
  // dependency) is what keeps the fire effect from re-running on every nudge.
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const nowRef = useRef(now);
  nowRef.current = now;

  // The rate baseline: the arrival time and index of the FIRST progress event
  // of this run. Everything after is measured against that one point rather
  // than against its immediate predecessor, so the number on screen is a
  // running average that settles instead of a figure that jumps with every
  // 200ms thinning interval. (Native thins to "at most one per 200ms but
  // always the first and last", so consecutive gaps are uneven by design —
  // a last-two-events rate would read as noise.)
  const baselineRef = useRef<{ at: number; index: number } | null>(null);

  // `phase` for the progress handler to read. The subscription below is made
  // ONCE per mount, so it would otherwise close over the phase its first
  // render saw ("idle") forever. Written at the transition rather than mirrored
  // during render so it is already `"running"` when the first push lands, with
  // no dependence on React having re-rendered in between.
  const phaseRef = useRef<ObjectTrackingPhase>("idle");
  const enterPhase = useCallback((next: ObjectTrackingPhase) => {
    phaseRef.current = next;
    setPhase(next);
  }, []);

  // Subscribing FIRST (this effect is declared before the fire effect below,
  // and effects run in declaration order) so no push can land between the RPC
  // going out and the handler being attached.
  useEffect(() => {
    const bridge = nativeBridge ?? defaultBridge;
    return bridge.on(TIMELINE_TRACK_PROGRESS_EVENT, (data) => {
      const push = data as TimelineTrackProgressData | null;
      if (!push || typeof push.index !== "number") return;
      // Only a RUNNING panel takes progress. A push that arrives after the run
      // has settled — native's stream and the RPC's answer are two channels,
      // and a stop in particular lands its last push around the result — would
      // otherwise overwrite the final readout with a count from mid-run, and
      // one that arrives before the run starts would show a stranger's numbers
      // (the subscription is live from mount, tracking is one session per
      // host, and this panel may be a second one opened while the first runs).
      if (phaseRef.current !== "running") return;
      const at = nowRef.current();
      const baseline = baselineRef.current;
      let fps: number | null = null;
      if (baseline === null) {
        baselineRef.current = { at, index: push.index };
      } else {
        const elapsedSec = (at - baseline.at) / 1000;
        const advanced = push.index - baseline.index;
        // Both guards matter: two pushes inside the same millisecond give a
        // zero denominator, and a `total`-forced final push that repeats the
        // last index gives a zero numerator. Neither is a rate.
        if (elapsedSec > 0 && advanced > 0) fps = advanced / elapsedSec;
      }
      setProgress({
        index: push.index,
        total: push.total,
        frame: push.frame,
        score: push.score,
        lost: push.lost,
        fps,
      });
    });
  }, [nativeBridge]);

  // Fire once per mount. `AppShell` bumps `remountTokens.toolbox` on every
  // routed 追尾, so a fresh mount IS the "start a run" signal — the same
  // one-shot `initialIntent` + remount arrangement every other screen uses,
  // and the reason this needs no prop-following effect. The ref additionally
  // makes a StrictMode double-invoke (or any future re-run of this effect)
  // harmless: a second `trackObject` would only earn a `TRACK_BUSY`.
  const startedRef = useRef(false);
  useEffect(() => {
    if (startedRef.current) return;
    if (!trackRequest) return;
    const item = trackRequest.selection.selected[0];
    // `guardMenuSelection` already refused an empty/multiple/wrong-kind
    // selection before `AppShell` set this request, so this is a belt-and-
    // braces check rather than a live path — but firing with no object would
    // mean sending `frame: undefined`, which is worth never doing.
    if (!item) return;
    startedRef.current = true;

    const bridge = nativeBridge ?? defaultBridge;
    const stored = settingsRef.current;
    enterPhase("running");
    void (async () => {
      try {
        const run = await bridge.request("timeline.trackObject", {
          layer: item.layer,
          // THE INVARIANT (contract v12 / design doc §5.2): the object's own
          // head frame, never the playback cursor. Native's `find_object`
          // searches "this frame and onward", so a cursor-derived frame could
          // bind the write-back to a completely different object.
          frame: item.frameStart,
          searchFactor: stored.searchFactor,
          smoothing: stored.smoothing,
          followSize: stored.followSize,
          lostScoreThreshold: stored.lostScoreThreshold,
          lostBehavior: stored.lostBehavior,
          keyframeStride: stored.keyframeStride,
        });
        setResult({
          frames: run.frames,
          keyframes: run.keyframes,
          elapsedMs: run.elapsedMs,
          cancelled: run.cancelled,
          // Verbatim: `lostRanges` are already AviUtl2 absolute frame numbers,
          // the same axis the progress event's `frame` uses (contract v12).
          lostRanges: run.lostRanges,
        });
        enterPhase(run.cancelled ? "cancelled" : "done");
      } catch (err) {
        setErrorCode(err instanceof BridgeError ? String(err.code) : "");
        setErrorMessage(err instanceof Error ? err.message : String(err));
        enterPhase("error");
      }
    })();
  }, [enterPhase, nativeBridge, trackRequest]);

  const cancel = useCallback(() => {
    const bridge = nativeBridge ?? defaultBridge;
    // Fire-and-forget: the answer only says whether a run was in flight, and
    // the terminal state comes from the `trackObject` promise either way. A
    // `false` here (the run finished between render and click) is an ordinary
    // outcome, not something to report.
    void bridge.request("timeline.cancelTracking", {}).catch(() => {
      // A failed stop leaves the run going; its own result still settles the
      // panel, so there is nothing useful to say here.
    });
  }, [nativeBridge]);

  return { phase, progress, result, errorCode, errorMessage, cancel };
}
