import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import { __resetRunLockForTests, isRunLockHeld } from "../../shell/runLock";
import type { BatchRunnerSettings } from "./batchRunner";
import { IMAGE_SHARED } from "./manifestMerge";
import type { BatchRow, BatchStat } from "./manifestMerge";
import { __resetBatchRuntimeForTests } from "./runtime";
import { useBatchRunner } from "./useBatchRunner";

const WAV_DIR = "C:\\batch\\in";
const OUT_DIR = "C:\\batch\\out";

const BASE_SETTINGS: BatchRunnerSettings = {
  promptCommon: "a cat",
  promptMode: "add",
  width: 512,
  height: 320,
  frameRate: 24,
  seed: -1,
  chunkedUpsample: true,
};

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

/** The scan-derived state every `run()` freezes into the runtime snapshot —
 * irrelevant to this file's concerns, so every call passes the same stub. */
const SCANNED = { imageFileNames: [] as string[], scannedFps: 24, scannedMaxFrames: 481 };

describe("useBatchRunner", () => {
  beforeEach(() => {
    __resetBatchRuntimeForTests();
    __resetRunLockForTests();
  });

  it("何も走っていなければidleで始まる", () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useBatchRunner({ nativeBridge: bridge }));
    expect(result.current.state).toBe("idle");
    expect(result.current.rows).toEqual([]);
    expect(result.current.lastStartResult).toBeNull();
    expect(result.current.wavDir).toBeNull();
    expect(result.current.outDir).toBeNull();
  });

  it("run()で同一tickにrunningへ、完走でidleへ戻り、行の更新が転送される", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
    const { result } = renderHook(() => useBatchRunner({ nativeBridge: bridge }));

    let forwarded: BatchRow[] = [];
    act(() => {
      result.current.run(
        {
          wavDir: WAV_DIR,
          outDir: OUT_DIR,
          settings: BASE_SETTINGS,
          rows: [makeRow(1, "a.wav", "Waiting")],
          ...SCANNED,
          pollIntervalMs: 4,
          jobBusyBackoffMs: 4,
        },
        (rows) => {
          forwarded = rows;
        },
      );
    });

    expect(result.current.state).toBe("running");
    expect(result.current.wavDir).toBe(WAV_DIR);
    expect(result.current.outDir).toBe(OUT_DIR);
    // 走行中はロック保持（取得は`runtime.ts`の`run()`が`start()`の前に行う）。
    expect(isRunLockHeld()).toBe(true);

    await waitFor(() => {
      expect(result.current.state).toBe("idle");
    });
    expect(result.current.lastStartResult).toEqual({ started: true, reason: "completed" });
    // 任意の転送コールバックにもフックのrowsにも、同じ結果が届く。
    expect(forwarded[0]).toMatchObject({ stat: "Done", output: "a.mp4" });
    expect(result.current.rows[0]).toMatchObject({ stat: "Done", output: "a.mp4" });
    expect(isRunLockHeld()).toBe(false);
  });

  it("対象行がゼロならrunningにならず、started:falseだけが残る", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useBatchRunner({ nativeBridge: bridge }));

    act(() => {
      result.current.run({
        wavDir: WAV_DIR,
        outDir: OUT_DIR,
        settings: BASE_SETTINGS,
        rows: [makeRow(1, "a.wav", "Done", { output: "a.mp4" })],
        ...SCANNED,
      });
    });

    expect(result.current.state).toBe("idle");
    await waitFor(() => {
      expect(result.current.lastStartResult).toEqual({ started: false, reason: "no rows to process" });
    });
    expect(isRunLockHeld()).toBe(false);
  });

  it("stop()は即座にstoppingを反映する", () => {
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 50 });
    const { result } = renderHook(() => useBatchRunner({ nativeBridge: bridge }));

    act(() => {
      result.current.run({
        wavDir: WAV_DIR,
        outDir: OUT_DIR,
        settings: BASE_SETTINGS,
        rows: [makeRow(1, "a.wav", "Waiting")],
        ...SCANNED,
        pollIntervalMs: 4,
        jobBusyBackoffMs: 4,
      });
    });
    expect(result.current.state).toBe("running");

    act(() => {
      result.current.stop();
    });
    expect(result.current.state).toBe("stopping");
  });

  // 本フックが`useRef`でランナーを持たない最大の理由（Create画面の`key`
  // リマウント耐性・バックエンドの `Docs/PENDING_TASKS_CLOSED.md` §3-47-02）を、アンマウント→再マウントで直接ピン留めする。
  it("アンマウント→再マウントしても走行中バッチへ再接続し、初回レンダーから状態が正しい", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 3, fs });
    const first = renderHook(() => useBatchRunner({ nativeBridge: bridge }));

    act(() => {
      first.result.current.run({
        wavDir: WAV_DIR,
        outDir: OUT_DIR,
        settings: BASE_SETTINGS,
        rows: [makeRow(1, "a.wav", "Waiting"), makeRow(2, "b.wav", "Waiting")],
        imageFileNames: ["cover.png"],
        scannedFps: 24,
        scannedMaxFrames: 481,
        pollIntervalMs: 4,
        jobBusyBackoffMs: 4,
      });
    });
    expect(first.result.current.state).toBe("running");

    // リマウント（AppShellのremountTokensによる`key`差し替え相当）。
    first.unmount();
    const second = renderHook(() => useBatchRunner({ nativeBridge: bridge }));

    // 再マウント直後の初回レンダーで、走行中であることも行も入力も見えている。
    expect(second.result.current.state).toBe("running");
    expect(second.result.current.rows).toHaveLength(2);
    expect(second.result.current.wavDir).toBe(WAV_DIR);
    expect(second.result.current.outDir).toBe(OUT_DIR);
    expect(second.result.current.imageFileNames).toEqual(["cover.png"]);
    expect(second.result.current.scannedFps).toBe(24);
    expect(second.result.current.scannedMaxFrames).toBe(481);
    // ロックも保たれている（孤児化していない）。
    expect(isRunLockHeld()).toBe(true);

    // 以降の進捗も新しいインスタンスへ届く（＝実際に繋がっている）。
    await waitFor(
      () => {
        expect(second.result.current.state).toBe("idle");
      },
      { timeout: 5_000 },
    );
    expect(second.result.current.rows.every((r) => r.stat === "Done")).toBe(true);
    expect(isRunLockHeld()).toBe(false);
  }, 10_000);
});
