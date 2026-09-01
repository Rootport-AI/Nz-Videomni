import { useCallback, useRef, useState } from "react";
import type { ConditioningImage } from "../../api/types";
import { bridge as defaultBridge } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import { BatchRunner } from "./batchRunner";
import type { BatchRunnerSettings, BatchRunnerState, BatchRunnerStartResult } from "./batchRunner";
import type { BatchRow } from "./manifestMerge";

export interface RunBatchParams {
  wavDir: string;
  imgDir?: string;
  outDir: string;
  settings: BatchRunnerSettings;
  rows: BatchRow[];
  /** N7 (PENDING §6): the Generate-tab's shared i2v keyframe(s), already
   * resolved to `image_id`s (`useBatchForm`'s own `useKeyframes` instance) —
   * snapshotted once here and passed straight through to
   * `BatchRunner.start()`'s `BatchRunnerStartParams.sharedConditioningImages`.
   * Omitted (undefined) defaults to `[]` there, matching every `Shared` row
   * resolving to no keyframe (the pre-N7 behavior) when the caller doesn't
   * pass this. */
  sharedConditioningImages?: ConditioningImage[];
  /** Test-only pass-through to `BatchRunner.start()` — see
   * `batchRunner.ts`'s `BatchRunnerStartParams` doc comments. Production
   * call sites should omit both. */
  pollIntervalMs?: number;
  jobBusyBackoffMs?: number;
}

export interface UseBatchRunnerDeps {
  nativeBridge?: NativeBridge;
}

export interface UseBatchRunnerResult {
  state: BatchRunnerState;
  /** The outcome of the most recently *attempted* `start()` — in particular
   * `{started:false, reason:"no rows to process"}` / `"batch already
   * running"`, which never produce a single row update and would otherwise
   * be invisible to the caller. `null` before the first attempt. */
  lastStartResult: BatchRunnerStartResult | null;
  /** Fires the run. `onRowsChanged` is called after every `stat` transition
   * with a fresh row array — the caller (`useBatchForm`) is expected to feed
   * it straight into its own row state. That React state IS the persistence:
   * nothing is flushed to disk (owner decision, 2026-07-18: stateless batch).
   * Fire-and-forget: this never throws and
   * the caller should read `state`/`lastStartResult` for the outcome.
   *
   * `onSettled` is called once, when `BatchRunner.start()`'s promise resolves
   * — i.e. when the run has finished (or was refused outright with
   * `{started:false}`). It fires from the promise itself, NOT from a React
   * effect, so it still runs when this hook has already been unmounted (a
   * `remountTokens` remount of the Create screen mid-run): that is what makes
   * it safe to release the shared run lock from there
   * (`useBatchForm`), instead of stranding it until the app is reloaded. */
  run: (
    params: RunBatchParams,
    onRowsChanged: (rows: BatchRow[]) => void,
    onSettled?: (result: BatchRunnerStartResult) => void,
  ) => void;
  stop: () => void;
}

/**
 * Thin React wrapper around the framework-agnostic {@link BatchRunner} class
 * — owns exactly one `BatchRunner` instance per hook mount (via `useRef`, so
 * it survives re-renders) and mirrors its 3-state lifecycle
 * (`idle`/`running`/`stopping`) into React state at the two points it can
 * actually change: right after `start()`/`stop()` return (both flip
 * `BatchRunner`'s internal state synchronously, before/without awaiting
 * anything — see `batchRunner.ts`) and once the run's promise settles.
 */
export function useBatchRunner(deps: UseBatchRunnerDeps = {}): UseBatchRunnerResult {
  const nativeBridge = deps.nativeBridge ?? defaultBridge;
  const runnerRef = useRef<BatchRunner | null>(null);
  if (runnerRef.current === null) runnerRef.current = new BatchRunner(nativeBridge);

  const [state, setState] = useState<BatchRunnerState>("idle");
  const [lastStartResult, setLastStartResult] = useState<BatchRunnerStartResult | null>(null);

  const run = useCallback(
    (
      params: RunBatchParams,
      onRowsChanged: (rows: BatchRow[]) => void,
      onSettled?: (result: BatchRunnerStartResult) => void,
    ) => {
      const runner = runnerRef.current;
      if (!runner) return;

      void runner
        .start({
          ...params,
          onRowsChanged,
        })
        .then((result) => {
          setLastStartResult(result);
          setState(runner.state);
          // Last, and outside React's state bookkeeping: this is the run-lock
          // handback, and it must happen even if the two setState calls above
          // land on an unmounted hook (both are no-ops in that case).
          onSettled?.(result);
        });

      // `BatchRunner.start()` flips its internal state synchronously before
      // its first `await` — reading it right back here keeps `state` truthful
      // even before the returned promise settles (matters for `idle` ->
      // `running` showing up on the very same tick as the click).
      setState(runner.state);
    },
    [],
  );

  const stop = useCallback(() => {
    const runner = runnerRef.current;
    if (!runner) return;
    runner.stop();
    setState(runner.state);
  }, []);

  return { state, lastStartResult, run, stop };
}
