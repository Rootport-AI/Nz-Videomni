import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import type { BatchRunnerSettings } from "./batchRunner";
import { IMAGE_SHARED } from "./manifestMerge";
import type { BatchRow, BatchStat } from "./manifestMerge";
import { useBatchRunner } from "./useBatchRunner";

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

describe("useBatchRunner", () => {
  it("starts idle", () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useBatchRunner({ nativeBridge: bridge }));
    expect(result.current.state).toBe("idle");
    expect(result.current.lastStartResult).toBeNull();
  });

  it("flips to running synchronously on run(), then back to idle once the batch completes, forwarding row updates", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
    const { result } = renderHook(() => useBatchRunner({ nativeBridge: bridge }));

    const rows: BatchRow[] = [makeRow(1, "a.wav", "Waiting")];
    let latestRows: BatchRow[] = rows;

    act(() => {
      result.current.run(
        { wavDir: "C:\\batch\\in", outDir: "C:\\batch\\out", settings: BASE_SETTINGS, rows, pollIntervalMs: 4, jobBusyBackoffMs: 4 },
        (r) => {
          latestRows = r;
        },
      );
    });

    expect(result.current.state).toBe("running");

    await waitFor(() => {
      expect(result.current.state).toBe("idle");
    });
    expect(result.current.lastStartResult).toEqual({ started: true, reason: "completed" });
    expect(latestRows[0]).toMatchObject({ stat: "Done", output: "a.mp4" });
  });

  it("surfaces lastStartResult:{started:false} when every row is Done/Skip, without ever going running", () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useBatchRunner({ nativeBridge: bridge }));
    const rows: BatchRow[] = [makeRow(1, "a.wav", "Done", { output: "a.mp4" })];

    act(() => {
      result.current.run({ wavDir: "C:\\batch\\in", outDir: "C:\\batch\\out", settings: BASE_SETTINGS, rows }, () => {});
    });

    expect(result.current.state).toBe("idle");
  });

  it("stop() flips state to stopping immediately", () => {
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 50 });
    const { result } = renderHook(() => useBatchRunner({ nativeBridge: bridge }));
    const rows: BatchRow[] = [makeRow(1, "a.wav", "Waiting")];

    act(() => {
      result.current.run(
        { wavDir: "C:\\batch\\in", outDir: "C:\\batch\\out", settings: BASE_SETTINGS, rows, pollIntervalMs: 4, jobBusyBackoffMs: 4 },
        () => {},
      );
    });
    expect(result.current.state).toBe("running");

    act(() => {
      result.current.stop();
    });
    expect(result.current.state).toBe("stopping");
  });
});
