import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import type { GenerateChainRequest } from "../../api/types";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import { __resetRunLockForTests, isRunLockHeld } from "../../shell/runLock";
import type { I2vLongRunnerSettings } from "./batchI2vLongRunner";
import type { I2vLongRow, I2vLongStat } from "./imageRows";
import { __resetI2vLongRuntimeForTests } from "./runtime";
import { useBatchI2vLongRunner } from "./useBatchI2vLongRunner";

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

describe("useBatchI2vLongRunner", () => {
  beforeEach(() => {
    __resetI2vLongRuntimeForTests();
    __resetRunLockForTests();
  });

  it("何も走っていなければidleで始まる", () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useBatchI2vLongRunner({ nativeBridge: bridge }));
    expect(result.current.state).toBe("idle");
    expect(result.current.rows).toEqual([]);
    expect(result.current.lastStartResult).toBeNull();
    expect(result.current.imgDir).toBeNull();
  });

  it("run()で同一tickにrunningへ、完走でidleへ戻り、行の更新が転送される", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
    const { result } = renderHook(() => useBatchI2vLongRunner({ nativeBridge: bridge }));

    let forwarded: I2vLongRow[] = [];
    act(() => {
      result.current.run(
        {
          imgDir: IMG_DIR,
          outDir: OUT_DIR,
          template: template(),
        settings: SETTINGS,
          rows: [makeRow(1, "a.png", "Waiting")],
          pollIntervalMs: 4,
          jobBusyBackoffMs: 4,
        },
        (rows) => {
          forwarded = rows;
        },
      );
    });

    expect(result.current.state).toBe("running");
    expect(result.current.imgDir).toBe(IMG_DIR);
    expect(result.current.outDir).toBe(OUT_DIR);

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
    const { result } = renderHook(() => useBatchI2vLongRunner({ nativeBridge: bridge }));

    act(() => {
      result.current.run({
        imgDir: IMG_DIR,
        outDir: OUT_DIR,
        template: template(),
        settings: SETTINGS,
        rows: [{ queue: 1, image: "a.png", prompt: "", stat: "Done", output: "a.mp4", error: "" }],
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
    const { result } = renderHook(() => useBatchI2vLongRunner({ nativeBridge: bridge }));

    act(() => {
      result.current.run({
        imgDir: IMG_DIR,
        outDir: OUT_DIR,
        template: template(),
        settings: SETTINGS,
        rows: [makeRow(1, "a.png", "Waiting")],
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

  // 本フックが`useRef`でランナーを持たない最大の理由（ChainedScreenの`key`
  // リマウント耐性）を、アンマウント→再マウントで直接ピン留めする。
  it("アンマウント→再マウントしても走行中バッチへ再接続し、初回レンダーから状態が正しい", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 3, fs });
    const first = renderHook(() => useBatchI2vLongRunner({ nativeBridge: bridge }));

    act(() => {
      first.result.current.run({
        imgDir: IMG_DIR,
        outDir: OUT_DIR,
        template: template(),
        settings: SETTINGS,
        rows: [makeRow(1, "a.png", "Waiting"), makeRow(2, "b.png", "Waiting")],
        pollIntervalMs: 4,
        jobBusyBackoffMs: 4,
      });
    });
    expect(first.result.current.state).toBe("running");

    // リマウント（AppShellのremountTokensによる`key`差し替え相当）。
    first.unmount();
    const second = renderHook(() => useBatchI2vLongRunner({ nativeBridge: bridge }));

    // 再マウント直後の初回レンダーで、走行中であることも行も見えている。
    expect(second.result.current.state).toBe("running");
    expect(second.result.current.rows).toHaveLength(2);
    expect(second.result.current.imgDir).toBe(IMG_DIR);

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
