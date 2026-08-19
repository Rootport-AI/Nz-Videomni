import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  RUN_LOCK_OWNER_BATCH_A2V,
  RUN_LOCK_OWNER_BATCH_I2V_LONG,
  __resetRunLockForTests,
  acquireRunLock,
  getRunLockOwner,
  isRunLockHeld,
  releaseRunLock,
  subscribeRunLock,
} from "./runLock";

describe("runLock", () => {
  beforeEach(() => {
    __resetRunLockForTests();
  });

  it("初期状態は未保持で、所有者もいない", () => {
    expect(isRunLockHeld()).toBe(false);
    expect(getRunLockOwner()).toBeNull();
  });

  it("取得するとトークンが返り、所有者ラベルが読める", () => {
    const token = acquireRunLock(RUN_LOCK_OWNER_BATCH_A2V);
    expect(token).not.toBeNull();
    expect(token?.owner).toBe("batch-a2v");
    expect(isRunLockHeld()).toBe(true);
    expect(getRunLockOwner()).toBe("batch-a2v");
  });

  it("保持中の二重取得はnull（再入不可）——所有者ラベルが同じでも取れない", () => {
    const first = acquireRunLock(RUN_LOCK_OWNER_BATCH_A2V);
    expect(first).not.toBeNull();

    expect(acquireRunLock(RUN_LOCK_OWNER_BATCH_I2V_LONG)).toBeNull();
    expect(acquireRunLock(RUN_LOCK_OWNER_BATCH_A2V)).toBeNull();
    // 保持者は最初の取得者のまま。
    expect(getRunLockOwner()).toBe("batch-a2v");
  });

  it("正しいトークンで解放でき、解放後は再取得できる", () => {
    const token = acquireRunLock(RUN_LOCK_OWNER_BATCH_A2V);
    expect(releaseRunLock(token)).toBe(true);
    expect(isRunLockHeld()).toBe(false);
    expect(getRunLockOwner()).toBeNull();

    const second = acquireRunLock(RUN_LOCK_OWNER_BATCH_I2V_LONG);
    expect(second).not.toBeNull();
    expect(getRunLockOwner()).toBe("batch-i2v-long");
  });

  // 本方式の存在理由そのもの: 常時マウントの反対側が「自分はidleだから解放」と
  // 誤って走行中のロックを解放する事故を、トークン不一致で必ず弾く。
  it("誤ったトークンでの解放は拒否され、ロックは保持されたまま", () => {
    const held = acquireRunLock(RUN_LOCK_OWNER_BATCH_I2V_LONG);
    const forged = { owner: "batch-i2v-long", serial: held?.serial ?? 1 };

    expect(releaseRunLock(forged)).toBe(false);
    expect(releaseRunLock(null)).toBe(false);
    expect(releaseRunLock(undefined)).toBe(false);
    expect(isRunLockHeld()).toBe(true);
    expect(getRunLockOwner()).toBe("batch-i2v-long");

    // 本物のトークンなら通る。
    expect(releaseRunLock(held)).toBe(true);
  });

  it("解放済みトークンでの二度目の解放は、その後の他所有者のロックを奪わない", () => {
    const stale = acquireRunLock(RUN_LOCK_OWNER_BATCH_A2V);
    expect(releaseRunLock(stale)).toBe(true);

    const fresh = acquireRunLock(RUN_LOCK_OWNER_BATCH_I2V_LONG);
    expect(releaseRunLock(stale)).toBe(false);
    expect(getRunLockOwner()).toBe("batch-i2v-long");

    expect(releaseRunLock(fresh)).toBe(true);
  });

  it("空きロックへの解放呼び出しは何も起きない（通知もしない）", () => {
    const listener = vi.fn();
    subscribeRunLock(listener);

    expect(releaseRunLock(null)).toBe(false);
    expect(listener).not.toHaveBeenCalled();
  });

  it("取得・解放それぞれで購読者へ通知し、購読解除後は通知しない", () => {
    const listener = vi.fn();
    const unsubscribe = subscribeRunLock(listener);

    const token = acquireRunLock(RUN_LOCK_OWNER_BATCH_A2V);
    expect(listener).toHaveBeenCalledTimes(1);

    // 取得失敗（保持中）は状態が変わらないので通知しない。
    acquireRunLock(RUN_LOCK_OWNER_BATCH_I2V_LONG);
    expect(listener).toHaveBeenCalledTimes(1);

    // 誤トークンの解放も状態が変わらないので通知しない。
    releaseRunLock({ owner: "batch-a2v", serial: 999 });
    expect(listener).toHaveBeenCalledTimes(1);

    releaseRunLock(token);
    expect(listener).toHaveBeenCalledTimes(2);

    unsubscribe();
    acquireRunLock(RUN_LOCK_OWNER_BATCH_A2V);
    expect(listener).toHaveBeenCalledTimes(2);
  });

  it("__resetRunLockForTestsは保持中でも解放し、購読者へ通知する", () => {
    const listener = vi.fn();
    subscribeRunLock(listener);
    acquireRunLock(RUN_LOCK_OWNER_BATCH_I2V_LONG);
    listener.mockClear();

    __resetRunLockForTests();

    expect(isRunLockHeld()).toBe(false);
    expect(getRunLockOwner()).toBeNull();
    expect(listener).toHaveBeenCalledTimes(1);
    // リセット後は誰でも取得できる。
    expect(acquireRunLock(RUN_LOCK_OWNER_BATCH_A2V)).not.toBeNull();
  });
});
