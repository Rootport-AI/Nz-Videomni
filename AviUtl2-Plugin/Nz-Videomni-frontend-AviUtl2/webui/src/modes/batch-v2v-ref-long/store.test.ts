import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ReferenceRules } from "../single/referenceRules";
import {
  __resetV2vRefLongStoreForTests,
  createInitialV2vRefLongState,
  reduceV2vRefLong,
  v2vRefLongStore,
} from "./store";
import type { V2vRefLongAction } from "./store";
import type { ScanResult, ScannedFile, V2vRefLongState } from "./types";

const RULES: ReferenceRules = {
  maxImages: 9,
  maxVideos: 3,
  maxAudios: 3,
  maxTotal: 12,
  videoTotalSec: 15,
  audioTotalSec: 15,
  clipMinSec: 1,
};

function f(dir: string, name: string): ScannedFile {
  return { name, path: `${dir}\\${name}`, sizeBytes: 1, durationSec: null };
}

function reduce(state: V2vRefLongState, ...actions: V2vRefLongAction[]): V2vRefLongState {
  return actions.reduce(reduceV2vRefLong, state);
}

/** Picture 1 of clip 1 from D:\work\pics (2 files). */
function scanResult(state: V2vRefLongState): ScanResult {
  return {
    inputs: { videoDir: state.inputs.videoDir, frames: state.inputs.frames },
    source: [],
    frames: [[{ kind: "picture", slot: 1, path: "D:\\work\\pics", files: [f("D:\\work\\pics", "1.png"), f("D:\\work\\pics", "2.png")] }]],
    rules: RULES,
  };
}

function scanned(): V2vRefLongState {
  const s = reduce(createInitialV2vRefLongState(), { type: "setFieldPath", frame: 0, fieldId: 1, path: "D:\\work\\pics" });
  return reduce(s, { type: "scanStarted" }, { type: "scanSucceeded", result: scanResult(s) });
}

describe("reduceV2vRefLong: inputs", () => {
  it("starts with one default frame, fillFollowing and joined on, auto output folder, add mode", () => {
    const s = createInitialV2vRefLongState();
    expect(s.inputs.frames).toHaveLength(1);
    expect(s.inputs.frames[0]?.fields).toHaveLength(4);
    expect(s.inputs.fillFollowing).toBe(true);
    expect(s.inputs.joined).toBe(true);
    expect(s.inputs.outDirIsAuto).toBe(true);
    expect(s.mainPromptMode).toBe("add");
    expect(s.run).toEqual({ state: "idle", lastStartResult: null, clipCount: null });
  });

  it("setVideoDir / setJoined / setFillFollowing / setOutDir", () => {
    const s = reduce(
      createInitialV2vRefLongState(),
      { type: "setVideoDir", path: "D:\\v" },
      { type: "setJoined", joined: false },
      { type: "setFillFollowing", value: false },
      { type: "setOutDir", path: "D:\\out" },
    );
    expect(s.inputs).toMatchObject({ videoDir: "D:\\v", joined: false, fillFollowing: false, outDir: "D:\\out", outDirIsAuto: false });
  });

  it("addField only within the limits", () => {
    let s = createInitialV2vRefLongState();
    s = reduce(s, { type: "addField", frame: 0, kind: "audio", rules: RULES }, { type: "addField", frame: 0, kind: "audio", rules: RULES });
    expect(s.inputs.frames[0]?.fields.filter((x) => x.kind === "audio")).toHaveLength(3);
    const same = reduce(s, { type: "addField", frame: 0, kind: "audio", rules: RULES });
    expect(same).toBe(s);
    expect(reduce(s, { type: "addField", frame: 5, kind: "picture", rules: RULES })).toBe(s);
  });

  it("setFieldPath writes one field", () => {
    const s = reduce(createInitialV2vRefLongState(), { type: "setFieldPath", frame: 0, fieldId: 3, path: "D:\\vid" });
    expect(s.inputs.frames[0]?.fields.map((x) => x.path)).toEqual(["", "", "D:\\vid", ""]);
  });

  it("addFrame / removeLastFrame (never below one frame)", () => {
    let s = reduce(createInitialV2vRefLongState(), { type: "addFrame" }, { type: "addFrame" });
    expect(s.inputs.frames.map((x) => x.id)).toEqual([1, 2, 3]);
    s = reduce(s, { type: "removeLastFrame" }, { type: "removeLastFrame" });
    expect(s.inputs.frames.map((x) => x.id)).toEqual([1]);
    expect(reduce(s, { type: "removeLastFrame" })).toBe(s);
  });
});

describe("reduceV2vRefLong: scan", () => {
  it("progress, success (rows and the auto output folder) and failure", () => {
    let s = reduce(createInitialV2vRefLongState(), { type: "scanStarted" }, { type: "scanProgress", done: 1, total: 3 });
    expect(s.scanning).toBe(true);
    expect(s.scanProgress).toEqual({ done: 1, total: 3 });
    s = reduce(s, { type: "scanFailed", message: "boom" });
    expect(s).toMatchObject({ scanning: false, scanProgress: null, scanError: "boom" });

    const ok = scanned();
    expect(ok.scanning).toBe(false);
    expect(ok.scanError).toBeNull();
    expect(ok.rows.map((r) => r.index)).toEqual([0, 1]);
    expect(ok.rows[0]).toEqual({ index: 0, mainPrompt: "", clipEdits: {}, stat: "Waiting", output: "", error: "" });
    expect(ok.inputs.outDir).toBe("D:\\work\\pics_out");
  });

  it("a hand-set output folder is kept by a scan", () => {
    const s = reduce(createInitialV2vRefLongState(), { type: "setOutDir", path: "D:\\mine" });
    const after = reduce(s, { type: "scanSucceeded", result: scanResult(s) });
    expect(after.inputs.outDir).toBe("D:\\mine");
  });

  it("a re-scan drops the row edits", () => {
    const edited = reduce(scanned(), { type: "setRowMainPrompt", row: 0, prompt: "x" });
    expect(edited.rows[0]?.mainPrompt).toBe("x");
    const again = reduce(edited, { type: "scanSucceeded", result: scanResult(edited) });
    expect(again.rows[0]?.mainPrompt).toBe("");
  });
});

describe("reduceV2vRefLong: row edits", () => {
  it("setMainPromptMode / setRowMainPrompt", () => {
    const s = reduce(scanned(), { type: "setMainPromptMode", mode: "replace" }, { type: "setRowMainPrompt", row: 1, prompt: "m" });
    expect(s.mainPromptMode).toBe("replace");
    expect(s.rows[1]?.mainPrompt).toBe("m");
  });

  it("a prompt edit keeps refs null (the folder assignment stays live)", () => {
    const s = reduce(scanned(), { type: "setClipPrompt", row: 1, clip: 1, prompt: "p" });
    const edit = s.rows[1]?.clipEdits[1];
    expect(edit).toEqual({ promptMode: "add", prompt: "p", refs: null });

    const mode = reduce(s, { type: "setClipPromptMode", row: 1, clip: 1, mode: "replace" });
    expect(mode.rows[1]?.clipEdits[1]).toEqual({ promptMode: "replace", prompt: "p", refs: null });
  });

  it("the first card touch copies the folder assignment in, and later prompt edits keep it", () => {
    const file = { fileName: "v.mp4", filePath: "D:\\v.mp4", durationSec: 4 };
    let s = reduce(scanned(), { type: "setClipPrompt", row: 1, clip: 0, prompt: "p" });
    s = reduce(s, { type: "setClipSlot", row: 1, clip: 0, kind: "video", slot: 1, file });
    const refs = s.rows[1]?.clipEdits[0]?.refs;
    expect(refs?.slots.picture[0]?.fileName).toBe("2.png");
    expect(refs?.slots.video[0]?.fileName).toBe("v.mp4");
    expect(refs?.enabled).toBe(true);
    s = reduce(s, { type: "setClipPrompt", row: 1, clip: 0, prompt: "q" });
    expect(s.rows[1]?.clipEdits[0]).toMatchObject({ prompt: "q", refs });
  });

  it("setClipSlot puts a card (next addedAt) and clears one", () => {
    const file = { fileName: "v.mp4", filePath: "D:\\v.mp4", durationSec: 4 };
    const s = reduce(scanned(), { type: "setClipSlot", row: 0, clip: 0, kind: "video", slot: 1, file });
    const card = s.rows[0]?.clipEdits[0]?.refs?.slots.video[0];
    expect(card).toMatchObject({ status: "ready", id: null, fileName: "v.mp4", filePath: "D:\\v.mp4", durationSec: 4, addedAt: 2 });
    const cleared = reduce(s, { type: "setClipSlot", row: 0, clip: 0, kind: "picture", slot: 1, file: null });
    expect(cleared.rows[0]?.clipEdits[0]?.refs?.slots.picture[0]?.status).toBe("empty");
  });

  it("clip edits before a scan do nothing", () => {
    const s = createInitialV2vRefLongState();
    expect(reduce(s, { type: "setClipPrompt", row: 0, clip: 0, prompt: "p" })).toBe(s);
  });

  it("resetRow puts only Done/Failed back to Waiting", () => {
    let s = reduce(scanned(), { type: "rowPatched", row: 0, patch: { stat: "Done", output: "1.mp4" } });
    s = reduce(s, { type: "resetRow", row: 0 });
    expect(s.rows[0]).toMatchObject({ stat: "Waiting", output: "", error: "" });
    const generating = reduce(s, { type: "rowPatched", row: 0, patch: { stat: "Generating" } });
    expect(reduce(generating, { type: "resetRow", row: 0 })).toBe(generating);
  });
});

describe("reduceV2vRefLong: run", () => {
  it("rowPatched writes stat/output/error and leaves the edits alone", () => {
    const s = reduce(
      scanned(),
      { type: "setRowMainPrompt", row: 0, prompt: "keep" },
      { type: "setClipPrompt", row: 0, clip: 1, prompt: "clip" },
    );
    const patched = reduce(s, { type: "rowPatched", row: 0, patch: { stat: "Failed", error: "e" } });
    expect(patched.rows[0]).toMatchObject({ mainPrompt: "keep", stat: "Failed", error: "e", output: "" });
    expect(patched.rows[0]?.clipEdits).toBe(s.rows[0]?.clipEdits);
    expect(patched.rows[1]).toBe(s.rows[1]);
  });

  it("runStateChanged patches the run; idle clears the frozen clip count", () => {
    let s = reduce(createInitialV2vRefLongState(), { type: "runStateChanged", patch: { state: "running", clipCount: 3 } });
    expect(s.run).toEqual({ state: "running", clipCount: 3, lastStartResult: null });
    s = reduce(s, { type: "runStateChanged", patch: { state: "idle", lastStartResult: { started: true, reason: "" } } });
    expect(s.run).toEqual({ state: "idle", clipCount: null, lastStartResult: { started: true, reason: "" } });
  });
});

function deepFreeze<T>(value: T): T {
  if (value !== null && typeof value === "object" && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const child of Object.values(value as Record<string, unknown>)) deepFreeze(child);
  }
  return value;
}

describe("reduceV2vRefLong: purity", () => {
  it("never writes to its argument (every action on a deep-frozen state)", () => {
    const file = { fileName: "v.mp4", filePath: "D:\\v.mp4", durationSec: 4 };
    const withEdit = reduce(scanned(), { type: "setClipSlot", row: 0, clip: 0, kind: "video", slot: 1, file });
    const actions: V2vRefLongAction[] = [
      { type: "setVideoDir", path: "D:\\v" },
      { type: "setJoined", joined: false },
      { type: "addField", frame: 0, kind: "video", rules: RULES },
      { type: "setFieldPath", frame: 0, fieldId: 2, path: "D:\\x" },
      { type: "addFrame" },
      { type: "removeLastFrame" },
      { type: "setFillFollowing", value: false },
      { type: "setOutDir", path: "D:\\o" },
      { type: "scanStarted" },
      { type: "scanProgress", done: 1, total: 2 },
      { type: "scanSucceeded", result: scanResult(withEdit) },
      { type: "scanFailed", message: "m" },
      { type: "setMainPromptMode", mode: "replace" },
      { type: "setRowMainPrompt", row: 0, prompt: "m" },
      { type: "setClipPrompt", row: 0, clip: 0, prompt: "p" },
      { type: "setClipPromptMode", row: 0, clip: 1, mode: "replace" },
      { type: "setClipSlot", row: 0, clip: 0, kind: "picture", slot: 1, file: null },
      { type: "setClipSlot", row: 1, clip: 1, kind: "audio", slot: 1, file },
      { type: "resetRow", row: 0 },
      { type: "rowPatched", row: 0, patch: { stat: "Done" } },
      { type: "runStateChanged", patch: { state: "running", clipCount: 2 } },
    ];
    for (const action of actions) {
      const frozen = deepFreeze(structuredClone(withEdit));
      expect(() => reduceV2vRefLong(frozen, action), action.type).not.toThrow();
    }
  });
});

describe("v2vRefLongStore", () => {
  beforeEach(() => {
    __resetV2vRefLongStoreForTests();
  });

  it("dispatch replaces the snapshot and notifies; a no-op action does neither", () => {
    const listener = vi.fn();
    const unsubscribe = v2vRefLongStore.subscribe(listener);
    const first = v2vRefLongStore.getSnapshot();
    expect(v2vRefLongStore.getSnapshot()).toBe(first);

    v2vRefLongStore.dispatch({ type: "setJoined", joined: false });
    expect(v2vRefLongStore.getSnapshot()).not.toBe(first);
    expect(v2vRefLongStore.getSnapshot().inputs.joined).toBe(false);
    expect(listener).toHaveBeenCalledTimes(1);

    v2vRefLongStore.dispatch({ type: "removeLastFrame" });
    expect(listener).toHaveBeenCalledTimes(1);

    unsubscribe();
    v2vRefLongStore.dispatch({ type: "setJoined", joined: true });
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("the reset goes back to the initial state and notifies", () => {
    v2vRefLongStore.dispatch({ type: "setVideoDir", path: "D:\\v" });
    const listener = vi.fn();
    v2vRefLongStore.subscribe(listener);
    __resetV2vRefLongStoreForTests();
    expect(v2vRefLongStore.getSnapshot()).toEqual(createInitialV2vRefLongState());
    expect(listener).toHaveBeenCalledTimes(1);
  });
});
