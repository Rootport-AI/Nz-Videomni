import { beforeEach, describe, expect, it } from "vitest";
import type { GenerateChainRequest } from "../../api/types";
import { createMockBridge } from "../../bridge/mockBridge";
import {
  RUN_LOCK_OWNER_BATCH_I2V_LONG,
  RUN_LOCK_OWNER_BATCH_V2V_REF_LONG,
  __resetRunLockForTests,
  acquireRunLock,
  getRunLockOwner,
  releaseRunLock,
} from "../../shell/runLock";
import { createInitialState } from "../single/referenceRules";
import type { ReferenceRules } from "../single/referenceRules";
import { __resetV2vRefLongRuntimeForTests, runV2vRefLong, stopV2vRefLong } from "./runtime";
import type { RunV2vRefLongParams } from "./runtime";
import { __resetV2vRefLongStoreForTests, v2vRefLongStore } from "./store";
import type { ResolvedClip, RowPlan, ScanResult } from "./types";

const RULES: ReferenceRules = {
  maxImages: 9,
  maxVideos: 3,
  maxAudios: 3,
  maxTotal: 12,
  videoTotalSec: 15,
  audioTotalSec: 15,
  clipMinSec: 1,
};

const TEMPLATE: GenerateChainRequest = {
  prompt: "a cat",
  width: 768,
  height: 512,
  frame_rate: 24,
  seed: 1,
  overlap_frames: 22,
  overlap_strength: 1,
  clips: [{ num_frames: 49 }, { num_frames: 49 }],
};

function scanWithRows(count: number): ScanResult {
  const source = Array.from({ length: count }, (_, i) => ({
    name: `s${i}.mp4`,
    path: `C:\\src\\s${i}.mp4`,
    sizeBytes: 1,
    durationSec: 5,
  }));
  return { inputs: { videoDir: "C:\\src", frames: [] }, source, frames: [], rules: RULES };
}

function clip(): ResolvedClip {
  return { promptMode: "add", prompt: "", refs: { ...createInitialState(RULES), enabled: true } };
}

function plan(index: number): RowPlan {
  return { index, sourcePath: `C:\\src\\s${index}.mp4`, mainPrompt: "", clips: [clip(), clip()], outFileName: `s${index}.mp4` };
}

function params(patch: Partial<RunV2vRefLongParams> = {}): RunV2vRefLongParams {
  return {
    bridge: createMockBridge({ delayMs: 0, runningPollCount: 1 }),
    template: TEMPLATE,
    settings: {
      mainPromptMode: "add",
      contextFrames: 22,
      rules: RULES,
      role: "ref2va",
      joined: false,
      joinFailedText: (m) => m,
    },
    plans: [plan(0), plan(1)],
    outDir: "C:\\out",
    clipCount: 2,
    pollIntervalMs: 2,
    jobBusyBackoffMs: 2,
    ...patch,
  };
}

async function waitUntil(predicate: () => boolean, timeoutMs = 3_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error("waitUntil: timed out");
    await new Promise((resolve) => setTimeout(resolve, 2));
  }
}

const store = () => v2vRefLongStore.getSnapshot();

beforeEach(() => {
  __resetV2vRefLongRuntimeForTests();
  __resetRunLockForTests();
  __resetV2vRefLongStoreForTests();
  v2vRefLongStore.dispatch({ type: "scanSucceeded", result: scanWithRows(2) });
});

describe("runV2vRefLong", () => {
  it("holds the lock while running and gives it back at the end", async () => {
    runV2vRefLong(params({ clipCount: 3 }));
    expect(getRunLockOwner()).toBe(RUN_LOCK_OWNER_BATCH_V2V_REF_LONG);
    expect(store().run).toEqual({ state: "running", clipCount: 3, lastStartResult: null });
    expect(store().rows[0]?.stat).toBe("Generating");

    await waitUntil(() => store().run.state === "idle");
    expect(getRunLockOwner()).toBeNull();
    expect(store().run).toEqual({ state: "idle", clipCount: null, lastStartResult: { started: true, reason: "completed" } });
    expect(store().rows.map((r) => [r.stat, r.output])).toEqual([
      ["Done", "s0.mp4"],
      ["Done", "s1.mp4"],
    ]);
  });

  it("does not start while another batch holds the lock", async () => {
    const other = acquireRunLock(RUN_LOCK_OWNER_BATCH_I2V_LONG);
    runV2vRefLong(params());
    expect(store().run.state).toBe("idle");
    expect(store().run.lastStartResult).toEqual({ started: false, reason: `run lock held by ${RUN_LOCK_OWNER_BATCH_I2V_LONG}` });
    expect(store().rows.every((r) => r.stat === "Waiting")).toBe(true);
    expect(getRunLockOwner()).toBe(RUN_LOCK_OWNER_BATCH_I2V_LONG);
    releaseRunLock(other);
  });

  it("gives the lock back when the runner does not start (no plan)", async () => {
    runV2vRefLong(params({ plans: [] }));
    await waitUntil(() => store().run.lastStartResult !== null);
    expect(store().run).toEqual({ state: "idle", clipCount: null, lastStartResult: { started: false, reason: "no rows to process" } });
    expect(getRunLockOwner()).toBeNull();
  });

  it("rowPatched leaves the row's other fields alone", async () => {
    v2vRefLongStore.dispatch({ type: "setRowMainPrompt", row: 0, prompt: "kept" });
    v2vRefLongStore.dispatch({ type: "setClipPrompt", row: 0, clip: 1, prompt: "clip text" });
    const [before, untouched] = store().rows;
    runV2vRefLong(params({ plans: [plan(0)] }));
    await waitUntil(() => store().run.state === "idle");
    const after = store().rows[0];
    expect(after?.mainPrompt).toBe("kept");
    expect(after?.clipEdits).toBe(before?.clipEdits);
    expect(after?.stat).toBe("Done");
    expect(store().rows[1]).toBe(untouched);
  });

  it("stop marks stopping and keeps the lock until the current row is done", async () => {
    runV2vRefLong(params());
    stopV2vRefLong();
    expect(store().run.state).toBe("stopping");
    expect(store().run.clipCount).toBe(2);
    expect(getRunLockOwner()).toBe(RUN_LOCK_OWNER_BATCH_V2V_REF_LONG);

    await waitUntil(() => store().run.state === "idle");
    expect(getRunLockOwner()).toBeNull();
    expect(store().rows.map((r) => r.stat)).toEqual(["Done", "Waiting"]);
  });

  it("stop while idle does nothing", () => {
    stopV2vRefLong();
    expect(store().run.state).toBe("idle");
  });

  it("a second run while one is in flight is refused by the lock", async () => {
    runV2vRefLong(params());
    runV2vRefLong(params());
    expect(store().run.state).toBe("running");
    expect(store().run.lastStartResult).toEqual({
      started: false,
      reason: `run lock held by ${RUN_LOCK_OWNER_BATCH_V2V_REF_LONG}`,
    });
    await waitUntil(() => store().run.state === "idle");
    expect(getRunLockOwner()).toBeNull();
  });
});
