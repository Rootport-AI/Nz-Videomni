import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import type { RefObject } from "react";
import type { JobResponse } from "../../api/types";
import type { SubmitState } from "./useGeneration";
import {
  clampRepeatCount,
  decideAfterSettle,
  getRepeatCountText,
  isTerminal,
  setRepeatCountText,
  shouldNotifyChange,
  subscribeRepeatCountText,
} from "./repeatRun";
import type { RepeatScreen } from "./repeatRun";

/** §1-80 Repeat count — the state half. Called at the top of the Single /
 * Chained screen body (before `onSubmitted`/`onFailed`, which need
 * `noteSubmitted`/`stop`). The completion watcher that sends the next run is
 * the separate {@link useRepeatRunDriver}, called once `handleGenerate` exists.
 *
 * "Active" means `remaining > 0`: that one condition drives the Stop button,
 * the disabled count field, and the settings-changed toast. The run lives in
 * this hook's state, so a screen remount (right-click `key` rebuild) ends it —
 * the job already running finishes on its own. Only the field text survives
 * (module variable in `repeatRun.ts`). */
export interface RepeatRun {
  /** Raw field text (kept as a string so the field can be emptied and retyped). */
  countText: string;
  setCountText: (text: string) => void;
  /** Blur: normalize the field text to the clamped count. */
  commitCountText: () => void;
  /** Submissions still to send after the current one. */
  remaining: number;
  remainingRef: RefObject<number>;
  /** The job whose settling triggers the next submission (null = none). */
  awaitingJobIdRef: RefObject<string | null>;
  /** Fingerprint of the last submission (for the settings-changed toast). */
  baselineRef: RefObject<string | null>;
  /** The settings-changed toast already fired since the last submission. */
  notifiedRef: RefObject<boolean>;
  start: (args: { send: () => void; fingerprint: string }) => void;
  /** User Stop and every abort path: no further submissions. */
  stop: () => void;
  /** Call first thing in `onSubmitted`. No-op unless runs remain, so the
   * N=1 path is untouched. */
  noteSubmitted: (jobId: string) => void;
  /** Account for one automatic submission about to be sent. */
  consumeOne: (fingerprint: string) => void;
}

export function useRepeatRun(screen: RepeatScreen): RepeatRun {
  const countText = useSyncExternalStore(subscribeRepeatCountText, () => getRepeatCountText(screen));
  const [remaining, setRemainingState] = useState(0);
  const remainingRef = useRef(0);
  const awaitingJobIdRef = useRef<string | null>(null);
  const baselineRef = useRef<string | null>(null);
  const notifiedRef = useRef(false);

  const setRemaining = useCallback((n: number) => {
    remainingRef.current = n;
    setRemainingState(n);
  }, []);

  const setCountText = useCallback((text: string) => setRepeatCountText(screen, text), [screen]);
  const commitCountText = useCallback(
    () => setRepeatCountText(screen, String(clampRepeatCount(getRepeatCountText(screen)))),
    [screen],
  );

  const start = useCallback(
    ({ send, fingerprint }: { send: () => void; fingerprint: string }) => {
      const n = clampRepeatCount(getRepeatCountText(screen));
      setRepeatCountText(screen, String(n));
      setRemaining(n - 1);
      awaitingJobIdRef.current = null;
      baselineRef.current = fingerprint;
      notifiedRef.current = false;
      send();
    },
    [screen, setRemaining],
  );

  const stop = useCallback(() => {
    setRemaining(0);
    awaitingJobIdRef.current = null;
  }, [setRemaining]);

  const noteSubmitted = useCallback((jobId: string) => {
    if (remainingRef.current > 0) awaitingJobIdRef.current = jobId;
  }, []);

  const consumeOne = useCallback(
    (fingerprint: string) => {
      setRemaining(remainingRef.current - 1);
      baselineRef.current = fingerprint;
      notifiedRef.current = false;
    },
    [setRemaining],
  );

  return useMemo(
    () => ({
      countText,
      setCountText,
      commitCountText,
      remaining,
      remainingRef,
      awaitingJobIdRef,
      baselineRef,
      notifiedRef,
      start,
      stop,
      noteSubmitted,
      consumeOne,
    }),
    [countText, setCountText, commitCountText, remaining, start, stop, noteSubmitted, consumeOne],
  );
}

export interface RepeatRunDriverOptions {
  jobs: readonly JobResponse[];
  submitPhase: SubmitState["phase"];
  /** The panel could submit right now (Single: no generate reasons; Chained:
   * `form.isValid`). `serverBusy` is deliberately not part of it — a busy
   * server answers 409 and `onFailed` ends the repeat. */
  canSubmit: boolean;
  /** Fingerprint of what `send` would submit right now. */
  getFingerprint: () => string;
  /** The screen's `handleGenerate` (reads the panel's current values). */
  send: () => void;
  onSettingsChanged: () => void;
}

/** §1-80 Repeat count — the driver half: watches `jobs` for the awaited job's
 * terminal transition and sends the next run, and raises the settings-changed
 * notice. Takes `jobs` as an argument (no `useJobsContext` inside) so it can be
 * exercised with a plain `renderHook`. */
export function useRepeatRunDriver(
  run: RepeatRun,
  { jobs, submitPhase, canSubmit, getFingerprint, send, onSettingsChanged }: RepeatRunDriverOptions,
): void {
  useEffect(() => {
    const id = run.awaitingJobIdRef.current;
    // Wait for `onSubmitted`'s ledger refresh to finish (the submit hook holds
    // "submitting" until then).
    if (id === null || submitPhase !== "idle") return;
    const job = jobs.find((j) => j.job_id === id);
    if (!job || !isTerminal(job.status)) return;
    // One shot per job: clear before deciding.
    run.awaitingJobIdRef.current = null;
    if (decideAfterSettle(job.status, run.remainingRef.current, canSubmit) === "stop") {
      run.stop();
      return;
    }
    run.consumeOne(getFingerprint());
    send();
  }, [jobs, submitPhase, canSubmit, getFingerprint, send, run]);

  const active = run.remaining > 0;
  const current = useMemo(() => (active ? getFingerprint() : null), [active, getFingerprint]);
  useEffect(() => {
    if (
      shouldNotifyChange({
        active,
        notified: run.notifiedRef.current,
        baseline: run.baselineRef.current,
        current,
      })
    ) {
      run.notifiedRef.current = true;
      onSettingsChanged();
    }
  }, [active, current, onSettingsChanged, run]);
}
