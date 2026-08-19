import { beforeEach, describe, expect, it, vi } from "vitest";
import type { GenerateChainRequest } from "../../api/types";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import {
  RUN_LOCK_OWNER_BATCH_A2V,
  __resetRunLockForTests,
  acquireRunLock,
  getRunLockOwner,
  isRunLockHeld,
  releaseRunLock,
} from "../../shell/runLock";
import type { I2vLongRunnerSettings } from "./batchI2vLongRunner";
import type { I2vLongRow, I2vLongStat } from "./imageRows";
import { __resetI2vLongRuntimeForTests, getI2vLongRuntime } from "./runtime";

const POLL_MS = 4;
const BACKOFF_MS = 4;
const IMG_DIR = "C:\\batch\\img";
const OUT_DIR = "C:\\batch\\out";

function template(): GenerateChainRequest {
  return {
    prompt: "a cat walking",
    width: 768,
    height: 512,
    frame_rate: 24,
    seed: 42,
    overlap_frames: 3,
    overlap_strength: 0.5,
    clips: [{ num_frames: 121 }, { num_frames: 121 }],
    chunked_upsample: true,
  };
}

function makeRow(queue: number, image: string, stat: I2vLongStat): I2vLongRow {
  return { queue, image, prompt: "", stat, output: "", error: "" };
}

/** The run's non-template settings (row-prompt add/replace mode) — irrelevant
 * to this file's concerns, so every call passes the panel's own default. */
const SETTINGS: I2vLongRunnerSettings = { promptMode: "add" };

async function waitUntil(predicate: () => boolean, timeoutMs = 2_000, intervalMs = 2): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error("waitUntil: timed out");
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

describe("batch i2v-long runtime", () => {
  beforeEach(() => {
    __resetI2vLongRuntimeForTests();
    __resetRunLockForTests();
  });

  it("getI2vLongRuntime()は常に同じインスタンスを返す（モジュールレベル・シングルトン）", () => {
    expect(getI2vLongRuntime()).toBe(getI2vLongRuntime());
  });

  it("初期スナップショットはidle・行なし・フォルダ未設定で、参照も安定している", () => {
    const runtime = getI2vLongRuntime();
    expect(runtime.getSnapshot()).toEqual({
      state: "idle",
      rows: [],
      imgDir: null,
      outDir: null,
      lastStartResult: null,
    });
    // useSyncExternalStoreの要件: 変化がない間はスナップショットの参照が変わらない。
    expect(runtime.getSnapshot()).toBe(runtime.getSnapshot());
  });

  it("run()はランロックを取得し、走行完了(idle遷移)で解放する", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
    const runtime = getI2vLongRuntime();

    runtime.run({
      bridge,
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      template: template(),
      settings: SETTINGS,
      rows: [makeRow(1, "a.png", "Waiting")],
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    // 走行中はロック保持（所有者はi2v-long側）。
    expect(runtime.getSnapshot().state).toBe("running");
    expect(getRunLockOwner()).toBe("batch-i2v-long");

    await waitUntil(() => runtime.getSnapshot().state === "idle");
    expect(runtime.getSnapshot().lastStartResult).toEqual({ started: true, reason: "completed" });
    expect(runtime.getSnapshot().rows[0]).toMatchObject({ stat: "Done", output: "a.mp4" });
    // 完了と同時に解放されている。
    expect(isRunLockHeld()).toBe(false);
  });

  // 実機報告(2026-07-30): 1行目だけSTATUSがWaitingのまま動かず、完了した瞬間に
  // Doneへ飛ぶ。原因は`run()`が`start()`の「後」で`rows: params.rows`を書いて
  // いたこと——`start()`は最初のawaitまで同期実行され、その中で1行目の
  // `Generating`が既にflushされているため、開始時の行で上書きしてしまっていた。
  it("run()から戻った同一tickで、1行目は既にGeneratingとして見えている（開始時の行で上書きしない）", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs, holdUploads: true });
    const runtime = getI2vLongRuntime();

    runtime.run({
      bridge,
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      template: template(),
      settings: SETTINGS,
      rows: [makeRow(1, "a.png", "Waiting"), makeRow(2, "b.png", "Waiting")],
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    const snap = runtime.getSnapshot();
    expect(snap.state).toBe("running");
    expect(snap.imgDir).toBe(IMG_DIR);
    expect(snap.outDir).toBe(OUT_DIR);
    expect(snap.rows[0]?.stat).toBe("Generating");
    expect(snap.rows[1]?.stat).toBe("Waiting");

    // 走行を最後まで進めてから終わる（走りっぱなしのランナーを残さない）。
    bridge.releaseUploads();
    await waitUntil(() => runtime.getSnapshot().state === "idle", 10_000);
  }, 15_000);

  // 購読者側から見ても同じこと: 通知のたびにスナップショットを記録すると、
  // 1行目のGeneratingを一度も見ないまま終わる、ということが起きてはならない。
  it("購読者は1行目のGeneratingを必ず一度は観測する", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
    const runtime = getI2vLongRuntime();

    const seen: I2vLongStat[][] = [];
    runtime.subscribe(() => {
      seen.push(runtime.getSnapshot().rows.map((r) => r.stat));
    });

    runtime.run({
      bridge,
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      template: template(),
      settings: SETTINGS,
      rows: [makeRow(1, "a.png", "Waiting"), makeRow(2, "b.png", "Waiting")],
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });
    await waitUntil(() => runtime.getSnapshot().state === "idle", 10_000);

    // 「1行目がGeneratingで2行目がWaiting」というスナップショットが、
    // 1行目がDoneになるより前に存在していること。
    const firstDoneAt = seen.findIndex((stats) => stats[0] === "Done");
    const generatingAt = seen.findIndex((stats) => stats[0] === "Generating" && stats[1] === "Waiting");
    expect(generatingAt).toBeGreaterThanOrEqual(0);
    expect(firstDoneAt).toBeGreaterThan(generatingAt);
    // そのスナップショットが「後から開始時の行に巻き戻される」ことも許されない。
    const lastWaitingAt = seen.map((s) => s[0]).lastIndexOf("Waiting");
    expect(lastWaitingAt).toBeLessThan(generatingAt);
  }, 15_000);

  it("started:false（対象行ゼロ）ならロックは即座に返される", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const runtime = getI2vLongRuntime();

    runtime.run({
      bridge,
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      template: template(),
      settings: SETTINGS,
      rows: [{ queue: 1, image: "a.png", prompt: "", stat: "Done", output: "a.mp4", error: "" }],
    });

    expect(runtime.getSnapshot().state).toBe("idle");
    await waitUntil(() => runtime.getSnapshot().lastStartResult !== null);
    expect(runtime.getSnapshot().lastStartResult).toEqual({ started: false, reason: "no rows to process" });
    expect(isRunLockHeld()).toBe(false);
    // ロックが返っているので、直後に他方が取得できる。
    expect(acquireRunLock(RUN_LOCK_OWNER_BATCH_A2V)).not.toBeNull();
  });

  it("他方(バッチA2V)がロックを保持中なら開始せず、理由だけを残す", () => {
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1 });
    const runtime = getI2vLongRuntime();
    const otherToken = acquireRunLock(RUN_LOCK_OWNER_BATCH_A2V);

    runtime.run({
      bridge,
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      template: template(),
      settings: SETTINGS,
      rows: [makeRow(1, "a.png", "Waiting")],
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    const snap = runtime.getSnapshot();
    expect(snap.state).toBe("idle");
    expect(snap.rows).toEqual([]);
    expect(snap.lastStartResult?.started).toBe(false);
    expect(snap.lastStartResult?.reason).toContain("batch-a2v");
    // 他方のロックは横取りも解放もされていない。
    expect(getRunLockOwner()).toBe("batch-a2v");
    expect(releaseRunLock(otherToken)).toBe(true);
  });

  it("走行中の重複run()はロック取得に失敗して二重実行しない", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 3, fs });
    const runtime = getI2vLongRuntime();
    const params = {
      bridge,
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      template: template(),
      settings: SETTINGS,
      rows: [makeRow(1, "a.png", "Waiting")],
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    };

    runtime.run(params);
    expect(runtime.getSnapshot().state).toBe("running");

    runtime.run(params);
    expect(runtime.getSnapshot().state).toBe("running");
    expect(runtime.getSnapshot().lastStartResult?.started).toBe(false);

    await waitUntil(() => runtime.getSnapshot().state === "idle");
    expect(isRunLockHeld()).toBe(false);
  });

  // リマウント相当: ChainedScreenが`key`リマウントされてもランナーは生き続け、
  // 購読し直したセクションが走行中の状態と行へそのまま再接続できること。
  it("購読解除→再購読（リマウント相当）で、走行中バッチの状態と行へ再接続できる", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 3, fs });
    const runtime = getI2vLongRuntime();

    const firstListener = vi.fn();
    const unsubscribeFirst = runtime.subscribe(firstListener);

    runtime.run({
      bridge,
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      template: template(),
      settings: SETTINGS,
      rows: [makeRow(1, "a.png", "Waiting"), makeRow(2, "b.png", "Waiting")],
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });
    expect(runtime.getSnapshot().state).toBe("running");

    // リマウント: 旧セクションが購読解除され、新セクションが購読し直す。
    unsubscribeFirst();
    const firstCallsAtUnsubscribe = firstListener.mock.calls.length;
    const secondListener = vi.fn();
    runtime.subscribe(secondListener);

    // 再購読直後の初回読み出しだけで、走行中であることも行も分かる。
    const afterRemount = runtime.getSnapshot();
    expect(afterRemount.state).toBe("running");
    expect(afterRemount.rows).toHaveLength(2);
    expect(afterRemount.imgDir).toBe(IMG_DIR);
    expect(afterRemount.outDir).toBe(OUT_DIR);
    // ロックは走行中のまま保持されており、孤児化していない。
    expect(getRunLockOwner()).toBe("batch-i2v-long");

    await waitUntil(() => runtime.getSnapshot().state === "idle");
    // 新しい購読者にも更新が届いている（＝走行中バッチへ実際に繋がっている）。
    expect(secondListener).toHaveBeenCalled();
    // 購読解除した旧セクションにはもう届かない（リークしていない）。
    expect(firstListener.mock.calls.length).toBe(firstCallsAtUnsubscribe);
    expect(runtime.getSnapshot().rows.every((r) => r.stat === "Done")).toBe(true);
    expect(isRunLockHeld()).toBe(false);
  }, 10_000);

  it("stop()ではロックを解放しない（生成中の1枚がまだサーバで走っているため）", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 3, fs });
    const runtime = getI2vLongRuntime();

    runtime.run({
      bridge,
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      template: template(),
      settings: SETTINGS,
      rows: [makeRow(1, "a.png", "Waiting"), makeRow(2, "b.png", "Waiting")],
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    runtime.stop();
    expect(runtime.getSnapshot().state).toBe("stopping");
    expect(isRunLockHeld()).toBe(true);

    await waitUntil(() => runtime.getSnapshot().state === "idle");
    // 走り切ってから初めて解放される。
    expect(isRunLockHeld()).toBe(false);
    expect(runtime.getSnapshot().rows[0]?.stat).toBe("Done");
    expect(runtime.getSnapshot().rows[1]?.stat).toBe("Waiting");
  }, 10_000);

  it("__resetI2vLongRuntimeForTestsはスナップショットを初期化し、保持中のロックも返す", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
    const runtime = getI2vLongRuntime();

    runtime.run({
      bridge,
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      template: template(),
      settings: SETTINGS,
      rows: [makeRow(1, "a.png", "Waiting")],
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });
    await waitUntil(() => runtime.getSnapshot().state === "idle");

    __resetI2vLongRuntimeForTests();
    expect(runtime.getSnapshot()).toEqual({
      state: "idle",
      rows: [],
      imgDir: null,
      outDir: null,
      lastStartResult: null,
    });
    expect(isRunLockHeld()).toBe(false);
  });
});
