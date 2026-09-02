import { beforeEach, describe, expect, it, vi } from "vitest";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import {
  RUN_LOCK_OWNER_BATCH_I2V_LONG,
  __resetRunLockForTests,
  acquireRunLock,
  getRunLockOwner,
  isRunLockHeld,
  releaseRunLock,
} from "../../shell/runLock";
import type { BatchRunnerSettings } from "./batchRunner";
import { IMAGE_SHARED } from "./manifestMerge";
import type { BatchRow, BatchStat } from "./manifestMerge";
import { __resetBatchRuntimeForTests, getBatchA2vRuntime } from "./runtime";

const POLL_MS = 4;
const BACKOFF_MS = 4;
const WAV_DIR = "C:\\batch\\wav";
const IMG_DIR = "C:\\batch\\img";
const OUT_DIR = "C:\\batch\\out";

const SETTINGS: BatchRunnerSettings = {
  promptCommon: "a cat",
  promptMode: "add",
  width: 512,
  height: 320,
  frameRate: 24,
  seed: -1,
  chunkedUpsample: true,
};

/** The scan-derived state a run freezes into the snapshot. Most tests do not
 * care about the values themselves, only that they survive a remount. */
const SCANNED = { imageFileNames: ["cover.png"], scannedFps: 24, scannedMaxFrames: 481 };

function makeRow(queue: number, wav: string, stat: BatchStat, overrides: Partial<BatchRow> = {}): BatchRow {
  return {
    queue,
    wav,
    duration: 3.0,
    image: IMAGE_SHARED,
    prompt: "",
    stat,
    output: "",
    frames: 49,
    skipReason: "",
    error: "",
    ...overrides,
  };
}

async function waitUntil(predicate: () => boolean, timeoutMs = 2_000, intervalMs = 2): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error("waitUntil: timed out");
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

describe("batch A2V runtime", () => {
  beforeEach(() => {
    __resetBatchRuntimeForTests();
    __resetRunLockForTests();
  });

  it("getBatchA2vRuntime()は常に同じインスタンスを返す（モジュールレベル・シングルトン）", () => {
    expect(getBatchA2vRuntime()).toBe(getBatchA2vRuntime());
  });

  it("初期スナップショットはidle・行なし・入力未設定で、参照も安定している", () => {
    const runtime = getBatchA2vRuntime();
    expect(runtime.getSnapshot()).toEqual({
      state: "idle",
      rows: [],
      wavDir: null,
      imgDir: null,
      outDir: null,
      imageFileNames: [],
      scannedFps: null,
      scannedMaxFrames: null,
      lastStartResult: null,
    });
    // useSyncExternalStoreの要件: 変化がない間はスナップショットの参照が変わらない。
    expect(runtime.getSnapshot()).toBe(runtime.getSnapshot());
  });

  it("run()はランロックを取得し、走行完了(idle遷移)で解放する", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
    const runtime = getBatchA2vRuntime();

    runtime.run({
      bridge,
      wavDir: WAV_DIR,
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      settings: SETTINGS,
      rows: [makeRow(1, "a.wav", "Waiting")],
      ...SCANNED,
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    // 走行中はロック保持（所有者はA2V側）。
    expect(runtime.getSnapshot().state).toBe("running");
    expect(getRunLockOwner()).toBe("batch-a2v");

    await waitUntil(() => runtime.getSnapshot().state === "idle");
    expect(runtime.getSnapshot().lastStartResult).toEqual({ started: true, reason: "completed" });
    expect(runtime.getSnapshot().rows[0]).toMatchObject({ stat: "Done", output: "a.mp4" });
    // 完了と同時に解放されている。
    expect(isRunLockHeld()).toBe(false);
  });

  // 順序規律（i2v-long `runtime.ts:161-172`の実害と同じもの）: 開始時の入力は
  // `start()`の「前」に公開しなければならない。後から書くと、`start()`が最初の
  // awaitまでの同期区間で飛ばした1行目のGenerating更新を、開始時の行で
  // 上書きしてしまう（1行目だけWaitingのまま止まって見える）。
  it("run()から戻った同一tickで、1行目は既にGeneratingとして見えている（開始時の行で上書きしない）", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs, holdUploads: true });
    const runtime = getBatchA2vRuntime();

    runtime.run({
      bridge,
      wavDir: WAV_DIR,
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      settings: SETTINGS,
      rows: [makeRow(1, "a.wav", "Waiting"), makeRow(2, "b.wav", "Waiting")],
      ...SCANNED,
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    const snap = runtime.getSnapshot();
    expect(snap.state).toBe("running");
    expect(snap.wavDir).toBe(WAV_DIR);
    expect(snap.imgDir).toBe(IMG_DIR);
    expect(snap.outDir).toBe(OUT_DIR);
    expect(snap.imageFileNames).toEqual(["cover.png"]);
    expect(snap.scannedFps).toBe(24);
    expect(snap.scannedMaxFrames).toBe(481);
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
    const runtime = getBatchA2vRuntime();

    const seen: BatchStat[][] = [];
    runtime.subscribe(() => {
      seen.push(runtime.getSnapshot().rows.map((r) => r.stat));
    });

    runtime.run({
      bridge,
      wavDir: WAV_DIR,
      outDir: OUT_DIR,
      settings: SETTINGS,
      rows: [makeRow(1, "a.wav", "Waiting"), makeRow(2, "b.wav", "Waiting")],
      ...SCANNED,
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

  it("started:false（対象行ゼロ）ならロックは返される", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const runtime = getBatchA2vRuntime();

    runtime.run({
      bridge,
      wavDir: WAV_DIR,
      outDir: OUT_DIR,
      settings: SETTINGS,
      rows: [makeRow(1, "a.wav", "Done", { output: "a.mp4" })],
      ...SCANNED,
    });

    expect(runtime.getSnapshot().state).toBe("idle");
    await waitUntil(() => runtime.getSnapshot().lastStartResult !== null);
    expect(runtime.getSnapshot().lastStartResult).toEqual({ started: false, reason: "no rows to process" });
    expect(isRunLockHeld()).toBe(false);
    // ロックが返っているので、直後に他方が取得できる。
    expect(acquireRunLock(RUN_LOCK_OWNER_BATCH_I2V_LONG)).not.toBeNull();
  });

  it("他方(バッチi2v-long)がロックを保持中なら開始せず、理由だけを残す", () => {
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1 });
    const runtime = getBatchA2vRuntime();
    const otherToken = acquireRunLock(RUN_LOCK_OWNER_BATCH_I2V_LONG);

    runtime.run({
      bridge,
      wavDir: WAV_DIR,
      outDir: OUT_DIR,
      settings: SETTINGS,
      rows: [makeRow(1, "a.wav", "Waiting")],
      ...SCANNED,
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    const snap = runtime.getSnapshot();
    expect(snap.state).toBe("idle");
    expect(snap.rows).toEqual([]);
    expect(snap.wavDir).toBeNull();
    expect(snap.lastStartResult?.started).toBe(false);
    expect(snap.lastStartResult?.reason).toContain("batch-i2v-long");
    // 他方のロックは横取りも解放もされていない。
    expect(getRunLockOwner()).toBe("batch-i2v-long");
    expect(releaseRunLock(otherToken)).toBe(true);
  });

  it("走行中の重複run()はロック取得に失敗して二重実行しない", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 3, fs });
    const runtime = getBatchA2vRuntime();
    const params = {
      bridge,
      wavDir: WAV_DIR,
      outDir: OUT_DIR,
      settings: SETTINGS,
      rows: [makeRow(1, "a.wav", "Waiting")],
      ...SCANNED,
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

  // §3-47の直接証明（購読者レベル）: Create画面が`key`リマウントされても
  // ランナーは生き続け、購読し直した側が走行中の状態・行・走行入力へそのまま
  // 再接続できること。
  it("購読解除→再購読（リマウント相当）で、走行中バッチの状態と行へ再接続できる", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 3, fs });
    const runtime = getBatchA2vRuntime();

    const firstListener = vi.fn();
    const unsubscribeFirst = runtime.subscribe(firstListener);

    runtime.run({
      bridge,
      wavDir: WAV_DIR,
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      settings: SETTINGS,
      rows: [makeRow(1, "a.wav", "Waiting"), makeRow(2, "b.wav", "Waiting")],
      ...SCANNED,
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });
    expect(runtime.getSnapshot().state).toBe("running");

    // リマウント: 旧セクションが購読解除され、新セクションが購読し直す。
    unsubscribeFirst();
    const firstCallsAtUnsubscribe = firstListener.mock.calls.length;
    const secondListener = vi.fn();
    runtime.subscribe(secondListener);

    // 再購読直後の初回読み出しだけで、走行中であることも行も走行入力も分かる。
    const afterRemount = runtime.getSnapshot();
    expect(afterRemount.state).toBe("running");
    expect(afterRemount.rows).toHaveLength(2);
    expect(afterRemount.wavDir).toBe(WAV_DIR);
    expect(afterRemount.imgDir).toBe(IMG_DIR);
    expect(afterRemount.outDir).toBe(OUT_DIR);
    expect(afterRemount.imageFileNames).toEqual(["cover.png"]);
    // ロックは走行中のまま保持されており、孤児化していない。
    expect(getRunLockOwner()).toBe("batch-a2v");

    await waitUntil(() => runtime.getSnapshot().state === "idle");
    // 新しい購読者にも更新が届いている（＝走行中バッチへ実際に繋がっている）。
    expect(secondListener).toHaveBeenCalled();
    // 購読解除した旧セクションにはもう届かない（リークしていない）。
    expect(firstListener.mock.calls.length).toBe(firstCallsAtUnsubscribe);
    expect(runtime.getSnapshot().rows.every((r) => r.stat === "Done")).toBe(true);
    expect(isRunLockHeld()).toBe(false);
  }, 10_000);

  // §3-47の直接証明（操作レベル）: 走行中に購読者が入れ替わっても、あとから
  // 購読した側のStopがその走行に効く。修正前は新しいマウントが新しいランナーを
  // 作っていたため、Stopは「自分の空のランナー」を止めるだけで、実際に走って
  // いる方は止められなかった（＝孤児化）。
  it("走行中に購読者が入れ替わっても、新しい購読者のstop()がその走行に効く", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 3, fs });
    const runtime = getBatchA2vRuntime();

    const unsubscribeFirst = runtime.subscribe(vi.fn());
    runtime.run({
      bridge,
      wavDir: WAV_DIR,
      outDir: OUT_DIR,
      settings: SETTINGS,
      rows: [makeRow(1, "a.wav", "Waiting"), makeRow(2, "b.wav", "Waiting")],
      ...SCANNED,
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });
    expect(runtime.getSnapshot().state).toBe("running");

    // リマウント相当の購読者入れ替え。
    unsubscribeFirst();
    const secondListener = vi.fn();
    runtime.subscribe(secondListener);

    // 新しい購読者から止められる（＝同じランナーに繋がっている証拠）。
    runtime.stop();
    expect(runtime.getSnapshot().state).toBe("stopping");
    expect(secondListener).toHaveBeenCalled();
    // 中止要求だけではロックを返さない（生成中の1本がジョブ枠を占有中）。
    expect(getRunLockOwner()).toBe("batch-a2v");

    await waitUntil(() => runtime.getSnapshot().state === "idle");
    // 実際に止まっている: 1本目は完走し、2本目には進んでいない。
    expect(runtime.getSnapshot().rows[0]?.stat).toBe("Done");
    expect(runtime.getSnapshot().rows[1]?.stat).toBe("Waiting");
    expect(isRunLockHeld()).toBe(false);
  }, 10_000);

  it("stop()ではロックを解放しない（生成中の1本がまだサーバで走っているため）", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 3, fs });
    const runtime = getBatchA2vRuntime();

    runtime.run({
      bridge,
      wavDir: WAV_DIR,
      outDir: OUT_DIR,
      settings: SETTINGS,
      rows: [makeRow(1, "a.wav", "Waiting"), makeRow(2, "b.wav", "Waiting")],
      ...SCANNED,
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

  it("__resetBatchRuntimeForTestsはスナップショットを初期化し、保持中のロックも返す", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
    const runtime = getBatchA2vRuntime();

    runtime.run({
      bridge,
      wavDir: WAV_DIR,
      outDir: OUT_DIR,
      settings: SETTINGS,
      rows: [makeRow(1, "a.wav", "Waiting")],
      ...SCANNED,
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });
    await waitUntil(() => runtime.getSnapshot().state === "idle");

    __resetBatchRuntimeForTests();
    expect(runtime.getSnapshot()).toEqual({
      state: "idle",
      rows: [],
      wavDir: null,
      imgDir: null,
      outDir: null,
      imageFileNames: [],
      scannedFps: null,
      scannedMaxFrames: null,
      lastStartResult: null,
    });
    expect(isRunLockHeld()).toBe(false);
  });
});
