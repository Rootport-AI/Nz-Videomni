/**
 * Batch v2v-ref-long: the run's module-level home (runner instance and lock
 * token), so a closed panel or a `key`-remounted Chain screen does not orphan
 * a running batch. The rows and the run state live in `v2vRefLongStore` only.
 *
 * Lock protocol (`shell/runLock.ts`), same as `batch-i2v-long/runtime.ts`:
 * acquired before the runner starts, released with this run's own token when
 * the runner's `start()` resolves (finished or `started:false`), and NOT on
 * `stop()` — the row generating now still occupies the server.
 */
import type { GenerateChainRequest } from "../../api/types";
import type { NativeBridge } from "../../bridge";
import { RUN_LOCK_OWNER_BATCH_V2V_REF_LONG, acquireRunLock, getRunLockOwner, releaseRunLock } from "../../shell/runLock";
import type { RunLockToken } from "../../shell/runLock";
import { V2vRefLongRunner } from "./runner";
import type { V2vRefLongRunnerSettings, V2vRefLongRunnerStartResult } from "./runner";
import { v2vRefLongStore } from "./store";
import type { RowPlan } from "./types";

export interface RunV2vRefLongParams {
  bridge: NativeBridge;
  template: GenerateChainRequest;
  settings: V2vRefLongRunnerSettings;
  plans: RowPlan[];
  outDir: string;
  /** The Chain form's clip count at Start (shown while running). */
  clipCount: number;
  /** Test-only. */
  pollIntervalMs?: number;
  /** Test-only. */
  jobBusyBackoffMs?: number;
}

let runner: V2vRefLongRunner | null = null;
let lockToken: RunLockToken | null = null;

function releaseOwnLock(): void {
  const token = lockToken;
  lockToken = null;
  releaseRunLock(token);
}

/** Fire-and-forget: the outcome lands in the store (`run`). */
export function runV2vRefLong(params: RunV2vRefLongParams): void {
  const token = acquireRunLock(RUN_LOCK_OWNER_BATCH_V2V_REF_LONG);
  if (token === null) {
    v2vRefLongStore.dispatch({
      type: "runStateChanged",
      patch: { lastStartResult: { started: false, reason: `run lock held by ${getRunLockOwner() ?? "another run"}` } },
    });
    return;
  }
  lockToken = token;
  const instance = new V2vRefLongRunner(params.bridge);
  runner = instance;

  // Before `start()`: it runs synchronously up to its first await, and the
  // first row's Generating patch happens inside that window.
  v2vRefLongStore.dispatch({
    type: "runStateChanged",
    patch: { state: "running", clipCount: params.clipCount, lastStartResult: null },
  });

  const finish = (result: V2vRefLongRunnerStartResult) => {
    // Our token is still `lockToken`: no other run can take the lock meanwhile.
    releaseOwnLock();
    v2vRefLongStore.dispatch({ type: "runStateChanged", patch: { state: "idle", lastStartResult: result } });
  };

  void instance
    .start({
      template: params.template,
      settings: params.settings,
      plans: params.plans,
      outDir: params.outDir,
      onRowPatched: (row, patch) => v2vRefLongStore.dispatch({ type: "rowPatched", row, patch }),
      ...(params.pollIntervalMs !== undefined ? { pollIntervalMs: params.pollIntervalMs } : {}),
      ...(params.jobBusyBackoffMs !== undefined ? { jobBusyBackoffMs: params.jobBusyBackoffMs } : {}),
    })
    .then(finish, (err: unknown) => finish({ started: false, reason: String(err) }));
}

/** Graceful stop: no further rows; the lock stays held until the run ends. */
export function stopV2vRefLong(): void {
  const instance = runner;
  if (!instance || instance.state === "idle") return;
  instance.stop();
  v2vRefLongStore.dispatch({ type: "runStateChanged", patch: { state: "stopping" } });
}

/** Test-only: forget the runner and give back a held lock. Does not touch the
 * store (`__resetV2vRefLongStoreForTests`) or the lock module itself. */
export function __resetV2vRefLongRuntimeForTests(): void {
  releaseOwnLock();
  runner = null;
}
