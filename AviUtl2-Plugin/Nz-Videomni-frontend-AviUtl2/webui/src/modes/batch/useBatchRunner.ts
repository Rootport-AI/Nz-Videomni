/**
 * バッチA2VのReactラッパー。バックエンドの `Docs/PENDING_TASKS_CLOSED.md` §3-47-02（2026-09-02）以降、**ランナーを`useRef`で
 * 持たない**——実体は`runtime.ts`のモジュールレベル・シングルトンで、このフックは
 * 購読するだけ。そのためCreate画面が`key`リマウントされても、マウント時に現在の
 * 状態と行へ自動的に再接続する（新しいランナーは作られない）。
 *
 * The React face of Batch A2V. The runner is NOT owned by a `useRef` here: it
 * lives in `runtime.ts`'s module-level singleton and this hook merely
 * subscribes, so a `key`-remounted Create screen re-attaches to a batch that is
 * already running (state, rows and the folders it runs against included)
 * instead of silently orphaning it.
 *
 * `useSyncExternalStore` is the right primitive for exactly that: the store is
 * outside React, the snapshot object is referentially stable between changes
 * (`runtime.ts` replaces it wholesale on every update), and a fresh mount reads
 * the current value on its very first render — no effect, no flash of "idle".
 */
import { useCallback, useSyncExternalStore } from "react";
import type { ConditioningImage } from "../../api/types";
import { bridge as defaultBridge } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import type { BatchRunnerSettings, BatchRunnerState, BatchRunnerStartResult } from "./batchRunner";
import type { BatchRow } from "./manifestMerge";
import { getBatchA2vRuntime } from "./runtime";

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
  /** Scan-derived state frozen with the rows so a mid-run remount can restore
   * the panel exactly as it stood at `start()` — the per-row image `<select>`
   * options and the fps/DURATION drift hint. */
  imageFileNames: string[];
  scannedFps: number | null;
  scannedMaxFrames: number | null;
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
  /** The running (or most recent) batch's live rows — served from the runtime
   * singleton, so this is already correct on a remount's first render. */
  rows: BatchRow[];
  /** The folders the running (or most recent) batch was started against. */
  wavDir: string | null;
  imgDir: string | null;
  outDir: string | null;
  /** The scan-derived state frozen at that run's `start()`. */
  imageFileNames: string[];
  scannedFps: number | null;
  scannedMaxFrames: number | null;
  /** The outcome of the most recently *attempted* `run()` — in particular
   * `{started:false, reason:"no rows to process"}` / `"batch already
   * running"` / `"run lock held by …"`, which never produce a single row
   * update and would otherwise be invisible to the caller. `null` before the
   * first attempt. */
  lastStartResult: BatchRunnerStartResult | null;
  /** Fires the run. `onRowsChanged` is an OPTIONAL extra sink — the primary
   * channel is `rows` above, which is what survives a remount. Fire-and-forget:
   * this never throws; read `state`/`lastStartResult` for the outcome.
   *
   * The shared run lock is taken inside `runtime.run()` (before the runner's
   * `start()`) and handed back from the run promise itself, NOT from a React
   * effect — so it still comes back when this hook has already been unmounted
   * (a `remountTokens` remount of the Create screen mid-run). */
  run: (params: RunBatchParams, onRowsChanged?: (rows: BatchRow[]) => void) => void;
  /** Graceful stop: no further rows are submitted, but the clip generating
   * right now runs to completion and is still saved. */
  stop: () => void;
}

export function useBatchRunner(deps: UseBatchRunnerDeps = {}): UseBatchRunnerResult {
  const nativeBridge = deps.nativeBridge ?? defaultBridge;
  const runtime = getBatchA2vRuntime();

  const snapshot = useSyncExternalStore(runtime.subscribe, runtime.getSnapshot, runtime.getSnapshot);

  const run = useCallback(
    (params: RunBatchParams, onRowsChanged?: (rows: BatchRow[]) => void) => {
      runtime.run({
        bridge: nativeBridge,
        wavDir: params.wavDir,
        outDir: params.outDir,
        settings: params.settings,
        rows: params.rows,
        imageFileNames: params.imageFileNames,
        scannedFps: params.scannedFps,
        scannedMaxFrames: params.scannedMaxFrames,
        ...(params.imgDir !== undefined ? { imgDir: params.imgDir } : {}),
        ...(params.sharedConditioningImages !== undefined
          ? { sharedConditioningImages: params.sharedConditioningImages }
          : {}),
        ...(onRowsChanged ? { onRowsChanged } : {}),
        ...(params.pollIntervalMs !== undefined ? { pollIntervalMs: params.pollIntervalMs } : {}),
        ...(params.jobBusyBackoffMs !== undefined ? { jobBusyBackoffMs: params.jobBusyBackoffMs } : {}),
      });
    },
    [runtime, nativeBridge],
  );

  const stop = useCallback(() => {
    runtime.stop();
  }, [runtime]);

  return {
    state: snapshot.state,
    rows: snapshot.rows,
    wavDir: snapshot.wavDir,
    imgDir: snapshot.imgDir,
    outDir: snapshot.outDir,
    imageFileNames: snapshot.imageFileNames,
    scannedFps: snapshot.scannedFps,
    scannedMaxFrames: snapshot.scannedMaxFrames,
    lastStartResult: snapshot.lastStartResult,
    run,
    stop,
  };
}
