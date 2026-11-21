import { describe, expect, it } from "vitest";
import type { ChainValidityReason } from "../chained/useChainForm";
import type { ReferenceRules } from "../single/referenceRules";
import type { RowEnv } from "./rowModel";
import {
  CHAIN_REASON_POLICY,
  chainSettingReasons,
  clipsTooShortForOverlap,
  runnableRowIndices,
  startBlockReasons,
} from "./startGuards";
import type { StartGuardInput } from "./startGuards";
import { createInitialV2vRefLongState } from "./store";
import type { ScanResult, ScannedFile, V2vRefLongState, V2vRow } from "./types";

// Compile-time: the policy names every ChainValidityReason and nothing else.
type PolicyMissing = Exclude<ChainValidityReason, keyof typeof CHAIN_REASON_POLICY>;
type PolicyExtra = Exclude<keyof typeof CHAIN_REASON_POLICY, ChainValidityReason>;
const policyComplete: [PolicyMissing, PolicyExtra] extends [never, never] ? true : never = true;
void policyComplete;

const RULES: ReferenceRules = {
  maxImages: 9,
  maxVideos: 3,
  maxAudios: 3,
  maxTotal: 12,
  videoTotalSec: 15,
  audioTotalSec: 15,
  clipMinSec: 1,
};

const ENV: RowEnv = {
  clipCount: 2,
  clip0Frames: 49,
  minFramesWithSource: 39,
  contextFrames: 22,
  frameRate: 24,
  maxBytes: { image: 1e9, video: 1e9, audio: 1e9 },
  fillFollowing: true,
};

function f(dir: string, name: string): ScannedFile {
  return { name, path: `${dir}\\${name}`, sizeBytes: 1, durationSec: null };
}

function row(index: number, patch: Partial<V2vRow> = {}): V2vRow {
  return { index, mainPrompt: "", clipEdits: {}, stat: "Waiting", output: "", error: "", ...patch };
}

/** Video folder V (2 files), Picture 1 of clip 1 from P (1 file). */
function scannedState(withPicture = true): V2vRefLongState {
  const base = createInitialV2vRefLongState();
  const frames = base.inputs.frames.map((frame) => ({
    ...frame,
    fields: frame.fields.map((field) => (field.id === 1 && withPicture ? { ...field, path: "P" } : field)),
  }));
  const inputs = { ...base.inputs, videoDir: "V", frames, outDir: "V_out" };
  const scan: ScanResult = {
    inputs: { videoDir: inputs.videoDir, frames: inputs.frames },
    source: [f("V", "a.mp4"), f("V", "b.mp4")],
    frames: [withPicture ? [{ kind: "picture", slot: 1, path: "P", files: [f("P", "1.png")] }] : []],
    rules: RULES,
  };
  return { ...base, inputs, scan, rows: [row(0), row(1)] };
}

function input(patch: Partial<StartGuardInput> = {}): StartGuardInput {
  return {
    state: scannedState(),
    chain: { clips: [{ numFrames: 49 }, { numFrames: 49 }], minFramesForOverlap: 39, hasSourceAudio: false },
    env: ENV,
    commonPrompt: "a cat",
    role: "ref2va",
    hasUnknownLoraTag: false,
    serverBusy: false,
    lockedByOther: false,
    ...patch,
  };
}

describe("startBlockReasons: the scan branch", () => {
  it("a scanned, valid batch has no reason", () => {
    expect(startBlockReasons(input())).toEqual({ reasons: [], promptEmptyRows: [], runnableRows: [0, 1] });
  });

  it("not scanned: only notScanned plus the always-on checks", () => {
    const state = { ...scannedState(), scan: null, rows: [] };
    expect(startBlockReasons(input({ state, serverBusy: true })).reasons).toEqual(["notScanned", "jobActive"]);
  });

  it("the folders changed after the scan: scanStale, and the row checks are skipped", () => {
    const state = scannedState();
    const stale = { ...state, inputs: { ...state.inputs, videoDir: "W", outDir: "" } };
    expect(startBlockReasons(input({ state: stale })).reasons).toEqual(["scanStale"]);
    // changing it back clears it
    expect(startBlockReasons(input({ state: { ...stale, inputs: { ...stale.inputs, videoDir: "V", outDir: "x" } } })).reasons).toEqual([]);
  });

  it("outDirMissing", () => {
    const state = scannedState();
    expect(startBlockReasons(input({ state: { ...state, inputs: { ...state.inputs, outDir: " " } } })).reasons).toEqual([
      "outDirMissing",
    ]);
  });

  it("noRunnableRows: every row done or 🚫", () => {
    const state = scannedState();
    const done = { ...state, rows: [row(0, { stat: "Done" }), row(1, { stat: "Done" })] };
    expect(startBlockReasons(input({ state: done })).reasons).toEqual(["noRunnableRows"]);
    const blocked = { ...state, rows: [row(0, { stat: "Done" }), row(1)] };
    // clip 1 too short for an opening video -> row 2 is 🚫
    expect(startBlockReasons(input({ state: blocked, env: { ...ENV, clip0Frames: 33 } })).reasons).toEqual([
      "noRunnableRows",
    ]);
  });

  it("runnable rows: Waiting, Failed and Generating without 🚫", () => {
    const state = scannedState();
    const rows = { ...state, rows: [row(0, { stat: "Failed" }), row(1, { stat: "Generating" })] };
    expect(runnableRowIndices(rows, ENV)).toEqual([0, 1]);
    expect(runnableRowIndices({ ...state, rows: [row(0, { stat: "Done" }), row(1)] }, ENV)).toEqual([1]);
    expect(runnableRowIndices({ ...state, scan: null }, ENV)).toEqual([]);
  });

  it("fl2vaLoaded only when a runnable row uses a reference", () => {
    expect(startBlockReasons(input({ role: "fl2va" })).reasons).toEqual(["fl2vaLoaded"]);
    expect(startBlockReasons(input({ role: "fl2va", state: scannedState(false) })).reasons).toEqual([]);
    // the row with the picture is done; the runnable one has none
    const state = scannedState();
    const onlyPlain = { ...state, rows: [row(0, { stat: "Done" }), row(1)] };
    expect(startBlockReasons(input({ role: "fl2va", state: onlyPlain })).reasons).toEqual([]);
  });

  it("fl2vaLoaded ignores a reference in a clip past the clip count", () => {
    const plain = scannedState(false);
    const frame2 = { id: 2, fields: [{ id: 1, kind: "picture" as const, path: "P" }] };
    const inputs = { ...plain.inputs, frames: [...plain.inputs.frames, frame2] };
    const scan: ScanResult = {
      ...(plain.scan as ScanResult),
      inputs: { videoDir: inputs.videoDir, frames: inputs.frames },
      frames: [[], [{ kind: "picture", slot: 1, path: "P", files: [f("P", "1.png")] }]],
    };
    const state = { ...plain, inputs, scan };
    const oneClip = { clips: [{ numFrames: 49 }], minFramesForOverlap: 39, hasSourceAudio: false };
    expect(startBlockReasons(input({ role: "fl2va", state, chain: oneClip, env: { ...ENV, clipCount: 1 } })).reasons).toEqual([]);
    expect(startBlockReasons(input({ role: "fl2va", state })).reasons).toEqual(["fl2vaLoaded"]);
  });

  it("promptEmpty names the runnable rows whose composed main prompt is empty", () => {
    const state = scannedState();
    const rows = { ...state, rows: [row(0, { mainPrompt: "dog" }), row(1)] };
    expect(startBlockReasons(input({ state: rows, commonPrompt: "  " }))).toEqual({
      reasons: ["promptEmpty"],
      promptEmptyRows: [1],
      runnableRows: [0, 1],
    });
    const replace = { ...rows, mainPromptMode: "replace" as const };
    expect(startBlockReasons(input({ state: replace })).promptEmptyRows).toEqual([]);
  });
});

describe("startBlockReasons: the always-on checks", () => {
  it("each check, in order", () => {
    const result = startBlockReasons(
      input({
        chain: { clips: [{ numFrames: 49 }, { numFrames: 33 }], minFramesForOverlap: 39, hasSourceAudio: true },
        hasUnknownLoraTag: true,
        serverBusy: true,
        lockedByOther: true,
      }),
    );
    expect(result.reasons).toEqual([
      "sourceAudioAttached",
      "unknownLoraTag",
      "clipTooShortForOverlap",
      "jobActive",
      "lockedByOther",
    ]);
  });

  it("clipTooShortForOverlap looks at clips 2+ only", () => {
    expect(clipsTooShortForOverlap({ clips: [{ numFrames: 25 }, { numFrames: 49 }], minFramesForOverlap: 39 })).toBe(false);
    expect(clipsTooShortForOverlap({ clips: [{ numFrames: 49 }, { numFrames: 38 }], minFramesForOverlap: 39 })).toBe(true);
    expect(clipsTooShortForOverlap({ clips: [{ numFrames: 25 }], minFramesForOverlap: 39 })).toBe(false);
  });
});

describe("chainSettingReasons", () => {
  it("keeps only the row-independent settings", () => {
    const all = Object.keys(CHAIN_REASON_POLICY) as ChainValidityReason[];
    expect(chainSettingReasons({ validityReasons: all })).toEqual([
      "dimensionsOffGrid",
      "clipFramesOffGrid",
      "totalFramesExceeded",
      "controlLoraNeedsReference",
      "referenceDimensionsOffGrid",
      "cropInvalid",
      "nagNegativeEmpty",
    ]);
    expect(chainSettingReasons({ validityReasons: ["promptEmpty", "minClips", "clipTooShortForOverlap", "sourceNotReady"] })).toEqual([]);
  });
});
