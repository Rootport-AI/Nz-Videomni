/**
 * バッチi2v-longのReactラッパー。ランナーを`useRef`で持たないのが要点——実体は
 * `runtime.ts`のモジュールレベル・シングルトンで、このフックは購読するだけ。
 * そのためChainedScreenが`key`リマウントされても、マウント時に現在の状態と行へ
 * 自動的に再接続する（新しいランナーは作られない）。
 *
 * この形は当初`modes/batch/useBatchRunner.ts`（バッチA2V）から移植したうえで、
 * 「ランナーをフックの寿命に縛らない」という一点だけを変えたものだった。
 * **2026-09-02（バックエンドの `Docs/PENDING_TASKS_CLOSED.md` §3-47-02）にA2V側も同じシングルトン方式へ移した**ので、いまは
 * 双子の構造になっている（A2V側の`modes/batch/runtime.ts`を参照）。
 *
 * The React face of Batch i2v-long. The runner is NOT owned by a `useRef` here:
 * it lives in `runtime.ts`'s module-level singleton and this hook merely
 * subscribes, so a `key`-remounted Chain screen re-attaches to a batch that is
 * already running (state, rows and folders included) instead of silently
 * orphaning it.
 *
 * This shape was originally ported from `modes/batch/useBatchRunner.ts` (Batch
 * A2V) with exactly that one structural change. Batch A2V moved to the same
 * singleton shape on 2026-09-02, so the two are now twins — see
 * `modes/batch/runtime.ts`.
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
