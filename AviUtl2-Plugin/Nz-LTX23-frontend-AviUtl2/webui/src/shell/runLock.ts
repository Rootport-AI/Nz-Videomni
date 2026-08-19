/**
 * バッチA2V（Create画面）とバッチi2v-long（Chain画面）の相互排除ロック。
 *
 * 所有者トークン方式なのは、両画面が常時マウント（`AppShell`は全タブを同時に
 * マウントする）で、マウント直後の初期idle遷移により他方のロックを誤解放する
 * 事故を防ぐため。素朴な「runner.state が idle になったら解放する」監視だと、
 * 後から生えた画面がマウントされた瞬間（state は当然 idle）に走行中の他方の
 * ロックを解放してしまう。解放は `acquireRunLock` が返したトークンと現在の
 * 保持者が同一のときだけ通る。
 *
 * The shared "only one batch may run at a time" lock between Batch A2V and
 * Batch i2v-long. It is an OWNER-TOKEN lock (acquire hands back an opaque
 * token; release only succeeds when that exact token is still the holder)
 * rather than a plain boolean flag, because both panels are permanently
 * mounted: a naive "release when my runner goes idle" effect would fire on the
 * OTHER panel's very first mount — its runner is idle at that moment — and free
 * a lock it never took.
 *
 * Module-level state, one lock per app (same shape as
 * `timeline/provisionalReservation.ts`'s single reservation seat). Volatile: a
 * reload starts unlocked. `__resetRunLockForTests()` exists purely so vitest's
 * `beforeEach` can clear the leak between tests.
 */

/** Display label of whoever holds (or wants) the lock. Kept as a plain string
 * union rather than an enum so the value is directly usable in tests/logs. */
export type RunLockOwner = "batch-a2v" | "batch-i2v-long";

export const RUN_LOCK_OWNER_BATCH_A2V: RunLockOwner = "batch-a2v";
export const RUN_LOCK_OWNER_BATCH_I2V_LONG: RunLockOwner = "batch-i2v-long";

/**
 * The opaque proof of ownership {@link acquireRunLock} returns. Only the object
 * IDENTITY matters — {@link releaseRunLock} compares by reference, so a
 * hand-built object with the same fields can never release someone else's lock.
 * The fields are readable purely for diagnostics/tests.
 */
export interface RunLockToken {
  readonly owner: string;
  /** Monotonic serial of this acquisition (1-based), for diagnostics. */
  readonly serial: number;
}

let holder: RunLockToken | null = null;
let serial = 0;
const listeners = new Set<() => void>();

function notify(): void {
  for (const listener of listeners) listener();
}

/**
 * Takes the lock for `ownerLabel`, or returns `null` when it is already held
 * (by anyone, INCLUDING the same owner label — the lock is not reentrant: one
 * run at a time, full stop). Notifies subscribers only on a successful take.
 */
export function acquireRunLock(ownerLabel: string): RunLockToken | null {
  if (holder !== null) return null;
  serial += 1;
  holder = { owner: ownerLabel, serial };
  notify();
  return holder;
}

/**
 * Releases the lock if — and only if — `token` is the very token currently
 * holding it. Returns whether anything was actually released, so a caller can
 * tell a genuine release apart from a no-op (a stale token from a previous run,
 * a `null` token, or an already-free lock). Notifies subscribers only on a real
 * release; a rejected release is completely inert, which is exactly what makes
 * the "other panel's first idle" case harmless.
 */
export function releaseRunLock(token: RunLockToken | null | undefined): boolean {
  if (!token || holder === null || holder !== token) return false;
  holder = null;
  notify();
  return true;
}

/** Whether any run currently holds the lock. */
export function isRunLockHeld(): boolean {
  return holder !== null;
}

/** The current holder's label, or `null` while the lock is free. Stable enough
 * to be a `useSyncExternalStore` snapshot (a primitive that only changes when
 * the lock does). */
export function getRunLockOwner(): string | null {
  return holder === null ? null : holder.owner;
}

/**
 * Subscribes to lock changes and returns the unsubscribe — the
 * `useSyncExternalStore(subscribeRunLock, getRunLockOwner)` pair is how a React
 * panel watches the OTHER panel's lock. Same listener-set shape as
 * `provisionalReservation.ts`'s `subscribeInserted`.
 */
export function subscribeRunLock(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Test-only: force the lock free (module-level state leaks between tests in
 * the same file otherwise). Notifies so any live subscriber re-reads. */
export function __resetRunLockForTests(): void {
  holder = null;
  serial = 0;
  notify();
}
