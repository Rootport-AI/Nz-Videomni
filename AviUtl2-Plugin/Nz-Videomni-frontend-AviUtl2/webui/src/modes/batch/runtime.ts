/**
 * バッチA2Vの「走行中バッチの生存点」。ランナー実体・最新の行・走行に使った
 * 入力（フォルダ・スキャン由来の派生状態）・購読機構をすべてモジュールレベルで
 * 持つシングルトン。
 *
 * こうする理由: Create画面は右クリックのintentルーティングで`remountTokens`に
 * よる`key`リマウントが起きる（`shell/AppShell.tsx`）。フック内の`useRef`に
 * ランナーを持たせると、そのリマウントで走行中バッチが孤児化し、走行は裏で続く
 * のにUIからは「実行していない」ように見え、Stopも行の進捗も消える。
 * モジュールレベルに置けば、リマウント後のセクションは購読し直すだけで走行中
 * バッチへ再接続できる（§3-47・2026-09-02）。出典は
 * `modes/batch-i2v-long/runtime.ts`——同じ形をA2Vへ移植したもの。
 *
 * The single live home of a running Batch A2V: the runner instance, the latest
 * row snapshot, the inputs the run was started with and the subscriber list all
 * live at module level, not inside a hook.
 *
 * Why: the Create screen is `key`-remounted by the right-click intent routing
 * (`AppShell`'s `remountTokens`). With a `useRef`-owned runner, that remount
 * orphaned an in-flight batch — the run kept submitting rows while the UI
 * showed "not running", with no Stop button and no row progress. Module-level
 * state means a remounted section merely re-subscribes and is back in sync.
 *
 * Lock protocol (see `shell/runLock.ts`) — identical to Batch i2v-long's:
 * - {@link BatchA2vRuntime.run} acquires BEFORE calling the runner (that is the
 *   only moment a token can be obtained);
 * - the token goes back on the run promise's `.then`, which covers BOTH endings
 *   `BatchRunner.start()` has (a finished run, and a `{started:false}` refusal
 *   that never left `idle`). It hangs off the promise, not a React effect, so
 *   an unmount mid-run can never strand the lock;
 * - `stop()` does NOT release: the clip currently generating still occupies the
 *   backend's single job slot until it finishes.
 */
import type { ConditioningImage } from "../../api/types";
import type { NativeBridge } from "../../bridge";
import {
  RUN_LOCK_OWNER_BATCH_A2V,
  acquireRunLock,
  getRunLockOwner,
  releaseRunLock,
} from "../../shell/runLock";
import type { RunLockToken } from "../../shell/runLock";
import { BatchRunner } from "./batchRunner";
import type { BatchRunnerSettings, BatchRunnerState, BatchRunnerStartResult } from "./batchRunner";
import type { BatchRow } from "./manifestMerge";

/** Everything a (re)mounted section needs to render — and keep driving — the
 * current run. Replaced wholesale on every change so it can be a
 * `useSyncExternalStore` snapshot (referentially stable while nothing changes —
 * never build it on the fly). */
export interface BatchA2vRuntimeSnapshot {
  state: BatchRunnerState;
  /** The live row list of the current/most recent run, or `[]` before the
   * first one. This is what makes a remount re-attach to a running batch. */
  rows: BatchRow[];
  /** The folders the current/most recent run was started with (`null` before
   * the first run) — a remounted section can show what is actually running,
   * not what its freshly-initialized form state happens to hold. */
  wavDir: string | null;
  imgDir: string | null;
  outDir: string | null;
  /** Scan-derived state frozen at `run()` time, so a remount restores the
   * per-row image `<select>` options and the fps/DURATION drift hint exactly as
   * they stood when the run began. Not live: the row table is disabled for the
   * whole run (`BatchSection`'s `disabled` gate), so nothing can edit them. */
  imageFileNames: string[];
  scannedFps: number | null;
  scannedMaxFrames: number | null;
  /** Outcome of the most recently ATTEMPTED run — notably the
   * `{started:false}` cases (nothing runnable, already running, lock held by
   * Batch i2v-long), which produce no row updates at all and would otherwise be
   * invisible. `null` before the first attempt. */
  lastStartResult: BatchRunnerStartResult | null;
}

/** `run()`'s parameters. The bridge is passed per call (rather than baked into
 * the module) so tests can drive the singleton with their own mock bridge, and
 * so production keeps injecting the same default bridge every other hook uses. */
export interface RunBatchA2vParams {
  bridge: NativeBridge;
  wavDir: string;
  imgDir?: string;
  outDir: string;
  settings: BatchRunnerSettings;
  rows: BatchRow[];
  /** The Create-owned KEYFRAMES panel's shared i2v keyframe(s), already
   * narrowed and snapshotted by the caller — frozen for the whole run. */
  sharedConditioningImages?: ConditioningImage[];
  /** Frozen alongside the rows: the scan-time derived state a remount has to
   * restore (see {@link BatchA2vRuntimeSnapshot}). */
  imageFileNames: string[];
  scannedFps: number | null;
  scannedMaxFrames: number | null;
  /** Optional extra sink for row updates, on top of the snapshot store (which
   * is the primary channel and the one that survives a remount). */
  onRowsChanged?: (rows: BatchRow[]) => void;
  /** Test-only pass-throughs to the runner. */
  pollIntervalMs?: number;
  jobBusyBackoffMs?: number;
}

export interface BatchA2vRuntime {
  getSnapshot(): BatchA2vRuntimeSnapshot;
  subscribe(listener: () => void): () => void;
  /** Fire-and-forget: never throws, never awaited by the caller. Read
   * `getSnapshot()` for the outcome. */
  run(params: RunBatchA2vParams): void;
  /** Graceful stop — no further rows are submitted; the clip being generated
   * right now still finishes and is saved. */
  stop(): void;
}

const IDLE_SNAPSHOT: BatchA2vRuntimeSnapshot = {
  state: "idle",
  rows: [],
  wavDir: null,
  imgDir: null,
  outDir: null,
  imageFileNames: [],
  scannedFps: null,
  scannedMaxFrames: null,
  lastStartResult: null,
};

let snapshot: BatchA2vRuntimeSnapshot = IDLE_SNAPSHOT;
let runner: BatchRunner | null = null;
let lockToken: RunLockToken | null = null;
const listeners = new Set<() => void>();

function notify(): void {
  for (const listener of listeners) listener();
}

function setSnapshot(patch: Partial<BatchA2vRuntimeSnapshot>): void {
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

const runtime: BatchA2vRuntime = {
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
    const token = acquireRunLock(RUN_LOCK_OWNER_BATCH_A2V);
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
    const instance = new BatchRunner(params.bridge);
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
    // finished and then jump straight to `Done` (owner-reported on the
    // i2v-long twin of this file, 2026-07-30).
    setSnapshot({
      wavDir: params.wavDir,
      imgDir: params.imgDir ?? null,
      outDir: params.outDir,
      rows: params.rows,
      imageFileNames: params.imageFileNames,
      scannedFps: params.scannedFps,
      scannedMaxFrames: params.scannedMaxFrames,
    });

    void instance
      .start({
        wavDir: params.wavDir,
        outDir: params.outDir,
        settings: params.settings,
        rows: params.rows,
        ...(params.imgDir !== undefined ? { imgDir: params.imgDir } : {}),
        ...(params.sharedConditioningImages !== undefined
          ? { sharedConditioningImages: params.sharedConditioningImages }
          : {}),
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

    // `BatchRunner.start()` flips its internal state synchronously before its
    // first `await`, so reading it right back here makes `running` visible on
    // the same tick as the click. A rejected start is still `idle` here — and
    // its lock is released by the `.then` above, on the microtask that
    // immediately follows. Only `state` is written here: everything the run was
    // started WITH was already published above, and re-publishing it now would
    // clobber the row updates `start()` has already flushed.
    setSnapshot({ state: instance.state });
  },

  stop() {
    const instance = runner;
    if (!instance) return;
    instance.stop();
    // NOTE: the lock is deliberately NOT released here — the clip currently
    // generating is still occupying the backend's single job slot.
    setSnapshot({ state: instance.state });
  },
};

/** The one and only runtime. Always returns the same object. */
export function getBatchA2vRuntime(): BatchA2vRuntime {
  return runtime;
}

/**
 * Test-only: forget the runner, the snapshot and any held lock token, so
 * module-level state cannot leak between tests. Does NOT stop an in-flight run
 * (nothing can, mid-clip) — call it from `beforeEach`, where any previous
 * test's run has already been awaited. Subscribers are notified so a still-
 * mounted hook re-reads the cleared snapshot. The run lock itself is reset via
 * `shell/runLock.ts`'s own `__resetRunLockForTests`.
 */
export function __resetBatchRuntimeForTests(): void {
  releaseOwnLock();
  runner = null;
  snapshot = IDLE_SNAPSHOT;
  notify();
}
