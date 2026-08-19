/**
 * バッチi2v-longの「走行中バッチの生存点」。ランナー実体・最新の行・入出力
 * フォルダ・購読機構をすべてモジュールレベルで持つシングルトン。
 *
 * こうする理由: Chain画面は右クリックのintentルーティングで`remountTokens`に
 * よる`key`リマウントが起きる（`shell/AppShell.tsx`）。フック内の`useRef`に
 * ランナーを持たせると、そのリマウントで走行中バッチが孤児化し、UIからは
 * 「実行していない」ように見えるのに共有ランロックだけが永久に握られたままに
 * なる。モジュールレベルに置けば、リマウント後のセクションは購読し直すだけで
 * 走行中バッチへ再接続できる。
 *
 * The single live home of a running Batch i2v-long: the runner instance, the
 * latest row snapshot, the folders it is running against and the subscriber
 * list all live at module level, not inside a hook.
 *
 * Why: the Chain screen is `key`-remounted by the right-click intent routing
 * (`AppShell`'s `remountTokens`). With a `useRef`-owned runner, that remount
 * would orphan an in-flight batch — the UI would show "not running" while the
 * shared run lock stayed held forever. Module-level state means a remounted
 * section merely re-subscribes and is back in sync with the running batch.
 *
 * Lock protocol (see `shell/runLock.ts`):
 * - {@link I2vLongRuntime.run} acquires BEFORE calling the runner (that is the
 *   only moment a token can be obtained), and releases IMMEDIATELY when the
 *   runner reports `started:false` (already running / nothing to run) — that
 *   case never reaches a `running` state, so no later transition would free it;
 * - a real run releases on its running -> idle transition, with the token it
 *   took, so it can never free a lock somebody else owns;
 * - `stop()` does NOT release: the current image keeps generating server-side
 *   until it finishes (`batchI2vLongRunner.stop()`'s doc comment), and the lock
 *   must stay held for exactly that long.
 */
import type { GenerateChainRequest } from "../../api/types";
import type { NativeBridge } from "../../bridge";
import {
  RUN_LOCK_OWNER_BATCH_I2V_LONG,
  acquireRunLock,
  getRunLockOwner,
  releaseRunLock,
} from "../../shell/runLock";
import type { RunLockToken } from "../../shell/runLock";
import { BatchI2vLongRunner } from "./batchI2vLongRunner";
import type { BatchI2vLongRunnerState, I2vLongRunnerSettings, I2vLongRunnerStartResult } from "./batchI2vLongRunner";
import type { I2vLongRow } from "./imageRows";

/** Everything a (re)mounted section needs to render the current run. Replaced
 * wholesale on every change so it can be a `useSyncExternalStore` snapshot
 * (referentially stable while nothing changes — never build it on the fly). */
export interface I2vLongRuntimeSnapshot {
  state: BatchI2vLongRunnerState;
  /** The live row list of the current/most recent run, or `[]` before the
   * first one. This is what makes a remount re-attach to a running batch. */
  rows: I2vLongRow[];
  /** The folders the current/most recent run was started with (`null` before
   * the first run) — a remounted section can show what is actually running,
   * not what its freshly-initialized form state happens to hold. */
  imgDir: string | null;
  outDir: string | null;
  /** Outcome of the most recently ATTEMPTED run — notably the
   * `{started:false}` cases (nothing runnable, already running, lock held by
   * Batch A2V), which produce no row updates at all and would otherwise be
   * invisible. `null` before the first attempt. */
  lastStartResult: I2vLongRunnerStartResult | null;
}

/** `run()`'s parameters. The bridge is passed per call (rather than baked into
 * the module) so tests can drive the singleton with their own mock bridge, and
 * so production keeps injecting the same default bridge every other hook uses. */
export interface RunI2vLongParams {
  bridge: NativeBridge;
  imgDir: string;
  outDir: string;
  /** A snapshot from `buildI2vLongTemplate` — held by value for the whole run
   * (`I2vLongRunnerStartParams.template`). */
  template: GenerateChainRequest;
  /** Frozen alongside `template` — today just the row-prompt add/replace mode
   * (`I2vLongRunnerStartParams.settings`). */
  settings: I2vLongRunnerSettings;
  rows: I2vLongRow[];
  /** Optional extra sink for row updates, on top of the snapshot store (which
   * is the primary channel and the one that survives a remount). */
  onRowsChanged?: (rows: I2vLongRow[]) => void;
  /** Test-only pass-throughs to the runner. */
  pollIntervalMs?: number;
  jobBusyBackoffMs?: number;
}

export interface I2vLongRuntime {
  getSnapshot(): I2vLongRuntimeSnapshot;
  subscribe(listener: () => void): () => void;
  /** Fire-and-forget: never throws, never awaited by the caller. Read
   * `getSnapshot()` for the outcome. */
  run(params: RunI2vLongParams): void;
  /** Graceful stop — no further rows are submitted; the image being generated
   * right now still finishes and is saved. */
  stop(): void;
}

const IDLE_SNAPSHOT: I2vLongRuntimeSnapshot = {
  state: "idle",
  rows: [],
  imgDir: null,
  outDir: null,
  lastStartResult: null,
};

let snapshot: I2vLongRuntimeSnapshot = IDLE_SNAPSHOT;
let runner: BatchI2vLongRunner | null = null;
let lockToken: RunLockToken | null = null;
const listeners = new Set<() => void>();

function notify(): void {
  for (const listener of listeners) listener();
}

function setSnapshot(patch: Partial<I2vLongRuntimeSnapshot>): void {
  snapshot = { ...snapshot, ...patch };
  notify();
}

/** Releases the run lock if this runtime is the one holding it. Idempotent —
 * a second call after the token was already given back is a no-op (the token
 * is cleared first, and `releaseRunLock` rejects a stale token anyway). */
function releaseOwnLock(): void {
  const token = lockToken;
  lockToken = null;
  releaseRunLock(token);
}

const runtime: I2vLongRuntime = {
  getSnapshot() {
    return snapshot;
  },

  subscribe(listener) {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  },

  run(params) {
    // The lock has to be taken BEFORE `start()`, because `start()`'s promise
    // only resolves when the whole run is over — by then it is far too late to
    // decide whether we were allowed to begin.
    const token = acquireRunLock(RUN_LOCK_OWNER_BATCH_I2V_LONG);
    if (token === null) {
      setSnapshot({
        lastStartResult: { started: false, reason: `run lock held by ${getRunLockOwner() ?? "another run"}` },
      });
      return;
    }
    lockToken = token;

    // Holding the lock proves no run is in flight (a second `run()` during one
    // never gets here), so a fresh runner bound to the CURRENT bridge is always
    // safe — and keeps a stale bridge from a previous run out of this one.
    const instance = new BatchI2vLongRunner(params.bridge);
    runner = instance;

    // 開始時の入力(フォルダ・行)は`start()`の「前」に公開する。
    // `start()`は最初のawaitまで同期実行され、その中で1行目が既に
    // `Generating`へ遷移して`onRowsChanged`が飛ぶため、後から
    // `rows: params.rows`を書くとその更新を開始時の行で上書きしてしまう
    // （1行目だけWaitingのまま止まって見え、完了した瞬間にDoneへ飛ぶ）。
    //
    // Publish the run's inputs BEFORE `start()`: it executes synchronously up
    // to its first `await`, and row 1's `Waiting -> Generating` flush happens
    // inside that window. Writing `rows` afterwards would overwrite exactly
    // that update — the first row would appear stuck on `Waiting` until it
    // finished and then jump straight to `Done` (owner-reported, 2026-07-30).
    setSnapshot({ imgDir: params.imgDir, outDir: params.outDir, rows: params.rows });

    void instance
      .start({
        imgDir: params.imgDir,
        outDir: params.outDir,
        template: params.template,
        settings: params.settings,
        rows: params.rows,
        onRowsChanged: (rows) => {
          setSnapshot({ rows });
          params.onRowsChanged?.(rows);
        },
        ...(params.pollIntervalMs !== undefined ? { pollIntervalMs: params.pollIntervalMs } : {}),
        ...(params.jobBusyBackoffMs !== undefined ? { jobBusyBackoffMs: params.jobBusyBackoffMs } : {}),
      })
      .then((result) => {
        // Either the run finished (running -> idle) or it was rejected outright
        // (`started:false`, which never left `idle`). Both end with the lock
        // going back, and both are the same code path because `start()` only
        // resolves at one of those two points.
        releaseOwnLock();
        setSnapshot({ state: instance.state, lastStartResult: result });
      });

    // `BatchI2vLongRunner.start()` flips its internal state synchronously
    // before its first `await`, so reading it right back here makes `running`
    // visible on the same tick as the click. A rejected start is still `idle`
    // here — and its lock is released by the `.then` above, on the microtask
    // that immediately follows. Only `state` is written here: everything the
    // run was started WITH was already published above, and re-publishing it
    // now would clobber the row updates `start()` has already flushed.
    setSnapshot({ state: instance.state });
  },

  stop() {
    const instance = runner;
    if (!instance) return;
    instance.stop();
    // NOTE: the lock is deliberately NOT released here — the image currently
    // generating is still occupying the backend's single job slot.
    setSnapshot({ state: instance.state });
  },
};

/** The one and only runtime. Always returns the same object. */
export function getI2vLongRuntime(): I2vLongRuntime {
  return runtime;
}

/**
 * Test-only: forget the runner, the snapshot and any held lock token, so
 * module-level state cannot leak between tests. Does NOT stop an in-flight run
 * (nothing can, mid-image) — call it from `beforeEach`, where any previous
 * test's run has already been awaited. Subscribers are notified so a still-
 * mounted hook re-reads the cleared snapshot. The run lock itself is reset via
 * `shell/runLock.ts`'s own `__resetRunLockForTests`.
 */
export function __resetI2vLongRuntimeForTests(): void {
  releaseOwnLock();
  runner = null;
  snapshot = IDLE_SNAPSHOT;
  notify();
}
