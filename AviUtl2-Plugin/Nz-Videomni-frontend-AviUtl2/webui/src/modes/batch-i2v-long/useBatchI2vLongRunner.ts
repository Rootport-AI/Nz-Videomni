/**
 * バッチi2v-longのReactラッパー。出典は`modes/batch/useBatchRunner.ts`だが、
 * 最大の違いは**ランナーを`useRef`で持たないこと**——実体は`runtime.ts`の
 * モジュールレベル・シングルトンで、このフックは購読するだけ。そのため
 * ChainedScreenが`key`リマウントされても、マウント時に現在の状態と行へ自動的に
 * 再接続する（新しいランナーは作られない）。
 *
 * The React face of Batch i2v-long. Ported from `modes/batch/useBatchRunner.ts`
 * with one structural change: the runner is NOT owned by a `useRef` here. It
 * lives in `runtime.ts`'s module-level singleton and this hook merely
 * subscribes, so a `key`-remounted Chain screen re-attaches to a batch that is
 * already running (state, rows and folders included) instead of silently
 * orphaning it.
 *
 * `useSyncExternalStore` is the right primitive for exactly that: the store is
 * outside React, the snapshot object is referentially stable between changes
 * (`runtime.ts` replaces it wholesale on every update), and a fresh mount reads
 * the current value on its very first render — no effect, no flash of "idle".
 */
import { useCallback, useSyncExternalStore } from "react";
import type { GenerateChainRequest } from "../../api/types";
import { bridge as defaultBridge } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import type { BatchI2vLongRunnerState, I2vLongRunnerSettings, I2vLongRunnerStartResult } from "./batchI2vLongRunner";
import type { I2vLongRow } from "./imageRows";
import { getI2vLongRuntime } from "./runtime";

export interface RunI2vLongBatchParams {
  imgDir: string;
  outDir: string;
  /** The already-snapshotted template (`buildI2vLongTemplate`) — frozen for
   * the whole run, so later Chain-form edits cannot affect it. */
  template: GenerateChainRequest;
  /** Frozen with the template — today just the row-prompt add/replace mode. */
  settings: I2vLongRunnerSettings;
  rows: I2vLongRow[];
  /** Test-only pass-throughs to `BatchI2vLongRunner.start()`. Production call
   * sites should omit both. */
  pollIntervalMs?: number;
  jobBusyBackoffMs?: number;
}

export interface UseBatchI2vLongRunnerDeps {
  nativeBridge?: NativeBridge;
}

export interface UseBatchI2vLongRunnerResult {
  state: BatchI2vLongRunnerState;
  /** The running (or most recent) batch's live rows — served from the runtime
   * singleton, so this is already correct on a remount's first render. */
  rows: I2vLongRow[];
  /** The folders the running (or most recent) batch was started against. */
  imgDir: string | null;
  outDir: string | null;
  /** The outcome of the most recently attempted `run()` — in particular the
   * `{started:false}` cases (nothing runnable, already running, or the run lock
   * held by Batch A2V), which produce no row updates at all. `null` before the
   * first attempt. */
  lastStartResult: I2vLongRunnerStartResult | null;
  /** Fires the run. `onRowsChanged` is an OPTIONAL extra sink — the primary
   * channel is `rows` above, which is what survives a remount. Fire-and-forget:
   * this never throws; read `state`/`lastStartResult` for the outcome. */
  run: (params: RunI2vLongBatchParams, onRowsChanged?: (rows: I2vLongRow[]) => void) => void;
  /** Graceful stop: no further rows are submitted, but the image generating
   * right now runs to completion and is still saved (no DELETE is sent — see
   * `batchI2vLongRunner.stop()`). */
  stop: () => void;
}

export function useBatchI2vLongRunner(deps: UseBatchI2vLongRunnerDeps = {}): UseBatchI2vLongRunnerResult {
  const nativeBridge = deps.nativeBridge ?? defaultBridge;
  const runtime = getI2vLongRuntime();

  const snapshot = useSyncExternalStore(runtime.subscribe, runtime.getSnapshot, runtime.getSnapshot);

  const run = useCallback(
    (params: RunI2vLongBatchParams, onRowsChanged?: (rows: I2vLongRow[]) => void) => {
      runtime.run({
        bridge: nativeBridge,
        imgDir: params.imgDir,
        outDir: params.outDir,
        template: params.template,
        settings: params.settings,
        rows: params.rows,
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
    imgDir: snapshot.imgDir,
    outDir: snapshot.outDir,
    lastStartResult: snapshot.lastStartResult,
    run,
    stop,
  };
}
