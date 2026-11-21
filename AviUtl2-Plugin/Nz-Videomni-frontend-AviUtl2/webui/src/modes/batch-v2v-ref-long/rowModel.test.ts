import { describe, expect, it } from "vitest";
import { createInitialState, emptySlot } from "../single/referenceRules";
import type { ReferenceKind, ReferenceRules, ReferenceState } from "../single/referenceRules";
import {
  buildRowPlan,
  effectiveClip,
  initialClipRefs,
  rowBlockReasons,
  rowHasUsedReference,
  rowIgnoredClips,
} from "./rowModel";
import type { RowEnv } from "./rowModel";
import type { ScanResult, ScannedField, ScannedFile, V2vRow } from "./types";

const RULES: ReferenceRules = {
  maxImages: 9,
  maxVideos: 3,
  maxAudios: 3,
  maxTotal: 12,
  videoTotalSec: 15,
  audioTotalSec: 15,
  clipMinSec: 1,
};

const MB = 1024 * 1024;

function file(dir: string, name: string, extra: Partial<ScannedFile> = {}): ScannedFile {
  return { name, path: `${dir}\\${name}`, sizeBytes: 1000, durationSec: null, ...extra };
}

function field(kind: ReferenceKind, slot: number, dir: string, list: ScannedFile[]): ScannedField {
  return { kind, slot, path: dir, files: list };
}

function makeScan(opts: { source?: ScannedFile[]; frames?: ScannedField[][]; rules?: ReferenceRules } = {}): ScanResult {
  return {
    inputs: { videoDir: "S", frames: [] },
    source: opts.source ?? [],
    frames: opts.frames ?? [],
    rules: opts.rules ?? RULES,
  };
}

function row(index: number, patch: Partial<V2vRow> = {}): V2vRow {
  return { index, mainPrompt: "", clipEdits: {}, stat: "Waiting", output: "", error: "", ...patch };
}

const ENV: RowEnv = {
  clipCount: 2,
  clip0Frames: 49,
  minFramesWithSource: 39,
  contextFrames: 22,
  frameRate: 24,
  maxBytes: { image: 20 * MB, video: 200 * MB, audio: 200 * MB },
  fillFollowing: true,
};

/** Frame 1: Picture 1 (2 files), Video 1 (1 file, 4 s). Frame 2: Picture 1 (2 files). */
function standardScan(source: ScannedFile[] = [file("S", "a.mp4", { durationSec: 5 }), file("S", "b.mp4", { durationSec: 5 })]): ScanResult {
  return makeScan({
    source,
    frames: [
      [
        field("picture", 1, "P", [file("P", "1.png"), file("P", "2.png")]),
        field("video", 1, "V", [file("V", "v1.mp4", { durationSec: 4 })]),
      ],
      [field("picture", 1, "Q", [file("Q", "q1.png"), file("Q", "q2.png")])],
    ],
  });
}

function refsWith(cards: { kind: ReferenceKind; slot: number; path: string; durationSec?: number | null }[]): ReferenceState {
  const state = { ...createInitialState(RULES), enabled: true };
  const slots = { picture: [...state.slots.picture], video: [...state.slots.video], audio: [...state.slots.audio] };
  cards.forEach((c, i) => {
    slots[c.kind][c.slot - 1] = {
      ...emptySlot(c.kind, c.slot),
      status: "ready",
      fileName: c.path,
      filePath: c.path,
      durationSec: c.durationSec ?? null,
      addedAt: i + 1,
    };
  });
  return { ...state, slots };
}

describe("initialClipRefs", () => {
  it("puts the row's file of each field into the card of the same kind and position", () => {
    const refs = initialClipRefs(standardScan(), 0, 0, true);
    expect(refs.enabled).toBe(true);
    expect(refs.a2vSlot).toBeNull();
    expect(refs.slots.picture[0]).toMatchObject({
      kind: "picture",
      slot: 1,
      status: "ready",
      id: null,
      fileName: "1.png",
      filePath: "P\\1.png",
      durationSec: null,
      addedAt: 1,
    });
    expect(refs.slots.video[0]).toMatchObject({ slot: 1, status: "ready", filePath: "V\\v1.mp4", durationSec: 4, addedAt: 2 });
    expect(refs.slots.picture[1]?.status).toBe("empty");
  });

  it("the position follows the field's slot (Picture 2 goes into card 2)", () => {
    const scan = makeScan({ frames: [[field("picture", 2, "P2", [file("P2", "x.png")])]] });
    const refs = initialClipRefs(scan, 0, 0, true);
    expect(refs.slots.picture[0]?.status).toBe("empty");
    expect(refs.slots.picture[1]).toMatchObject({ slot: 2, fileName: "x.png", addedAt: 1 });
  });

  it("a folder without the row's file leaves the card empty", () => {
    const refs = initialClipRefs(standardScan(), 1, 0, true);
    expect(refs.slots.picture[0]?.fileName).toBe("2.png");
    expect(refs.slots.video[0]?.status).toBe("empty");
  });

  it("clips past the frames take the last frame only while fillFollowing", () => {
    expect(initialClipRefs(standardScan(), 0, 2, true).slots.picture[0]?.fileName).toBe("q1.png");
    expect(initialClipRefs(standardScan(), 0, 2, false).slots.picture[0]?.status).toBe("empty");
  });
});

describe("effectiveClip", () => {
  it("is the edit when there is one, else the folder assignment with an empty add prompt", () => {
    const scan = standardScan();
    const plain = effectiveClip(row(0), scan, 1, true);
    expect(plain.promptMode).toBe("add");
    expect(plain.prompt).toBe("");
    expect(plain.refs.slots.picture[0]?.fileName).toBe("q1.png");

    const edit = { promptMode: "replace" as const, prompt: "x", refs: createInitialState(RULES) };
    expect(effectiveClip(row(0, { clipEdits: { 1: edit } }), scan, 1, true)).toEqual(edit);
  });

  it("a prompt-only edit (refs null) follows fillFollowing; a card-touched edit does not", () => {
    const scan = standardScan();
    const promptOnly = row(0, { clipEdits: { 2: { promptMode: "replace", prompt: "p", refs: null } } });
    const on = effectiveClip(promptOnly, scan, 2, true);
    expect(on).toMatchObject({ promptMode: "replace", prompt: "p" });
    expect(on.refs.slots.picture[0]?.fileName).toBe("q1.png");
    expect(effectiveClip(promptOnly, scan, 2, false).refs.slots.picture[0]?.status).toBe("empty");

    const touchedRefs = initialClipRefs(scan, 0, 2, true);
    const touched = row(0, { clipEdits: { 2: { promptMode: "add", prompt: "", refs: touchedRefs } } });
    expect(effectiveClip(touched, scan, 2, false).refs).toBe(touchedRefs);
    expect(effectiveClip(touched, scan, 2, true).refs).toBe(touchedRefs);
  });
});

describe("rowBlockReasons", () => {
  it("a runnable row has none", () => {
    expect(rowBlockReasons(row(0), standardScan(), ENV)).toEqual([]);
  });

  it("noSourceSingleClip: one clip and no opening video", () => {
    const scan = standardScan([file("S", "a.mp4")]);
    expect(rowBlockReasons(row(1), scan, { ...ENV, clipCount: 1 }).map((r) => r.code)).toEqual(["noSourceSingleClip"]);
    expect(rowBlockReasons(row(1), scan, ENV)).toEqual([]);
  });

  it("sourceTooShort: only when the length is known", () => {
    const short = standardScan([file("S", "a.mp4", { durationSec: 0.5 })]);
    expect(rowBlockReasons(row(0), short, ENV)).toEqual([{ code: "sourceTooShort", frames: 22 }]);
    // 22/24 s rounds to exactly 22 frames: not short
    const edge = standardScan([file("S", "a.mp4", { durationSec: 22 / 24 })]);
    expect(rowBlockReasons(row(0), edge, ENV)).toEqual([]);
    const unknown = standardScan([file("S", "a.mp4", { durationSec: null })]);
    expect(rowBlockReasons(row(0), unknown, ENV)).toEqual([]);
  });

  it("clip0TooShort: with an opening video, clip 1 below the floor", () => {
    expect(rowBlockReasons(row(0), standardScan(), { ...ENV, clip0Frames: 33 })).toEqual([
      { code: "clip0TooShort", minFrames: 39 },
    ]);
    const noSource = standardScan([]);
    expect(rowBlockReasons(row(0), noSource, { ...ENV, clip0Frames: 33 })).toEqual([]);
  });

  it("fileTooLarge: scanned files over their kind's cap; Audio fields use the audio cap", () => {
    const scan = makeScan({
      source: [file("S", "big.mp4", { sizeBytes: 300 * MB })],
      frames: [
        [
          field("picture", 1, "P", [file("P", "big.png", { sizeBytes: 21 * MB })]),
          field("audio", 1, "A", [file("A", "track.mp4", { sizeBytes: 150 * MB })]),
        ],
      ],
    });
    const env = { ...ENV, maxBytes: { image: 20 * MB, video: 200 * MB, audio: 100 * MB } };
    expect(rowBlockReasons(row(0), scan, env)).toEqual([
      { code: "fileTooLarge", fileName: "big.mp4", filePath: "S\\big.mp4", limitMb: 200 },
      { code: "fileTooLarge", fileName: "big.png", filePath: "P\\big.png", limitMb: 20 },
      { code: "fileTooLarge", fileName: "track.mp4", filePath: "A\\track.mp4", limitMb: 100 },
    ]);
  });

  it("fileTooLarge: a greyed card's file is not judged", () => {
    const scan = makeScan({
      source: [file("S", "a.mp4")],
      frames: [
        [
          field("video", 1, "V1", [file("V1", "long.mp4", { durationSec: 16 })]),
          field("video", 2, "V2", [file("V2", "grey.mp4", { durationSec: 3, sizeBytes: 500 * MB })]),
        ],
      ],
    });
    expect(rowIgnoredClips(row(0), scan, 2, true)).toEqual([0, 1]);
    expect(rowBlockReasons(row(0), scan, ENV)).toEqual([]);
  });

  it("fileTooLarge: a file put in by hand (not in the scan) is not judged", () => {
    const scan = standardScan();
    const edit = { promptMode: "add" as const, prompt: "", refs: refsWith([{ kind: "picture", slot: 1, path: "X\\hand.png" }]) };
    expect(rowBlockReasons(row(0, { clipEdits: { 0: edit } }), scan, ENV)).toEqual([]);
  });

  it("noMaterial: no opening video and no used reference in clips 1..N", () => {
    // the base folder is frame 3's, past a clip count of 2
    const scan = makeScan({
      frames: [[field("picture", 1, "P", [file("P", "1.png")])], [], [field("video", 1, "V", [file("V", "a"), file("V", "b")])]],
    });
    const env = { ...ENV, fillFollowing: false };
    expect(rowBlockReasons(row(1), scan, env)).toEqual([{ code: "noMaterial" }]);
    expect(rowBlockReasons(row(0), scan, env)).toEqual([]);
  });

  it("an edit of a clip past the clip count is neither judged nor dropped", () => {
    // frame 3 (clip 3) holds an oversized picture; the edit of clip 3 empties its cards
    const scan = standardScan();
    const big: ScanResult = {
      ...scan,
      frames: [...scan.frames, [], [field("picture", 1, "Z", [file("Z", "z.png", { sizeBytes: 99 * MB })])]],
    };
    const env = { ...ENV, fillFollowing: false };
    const edit = { promptMode: "add" as const, prompt: "late", refs: createInitialState(RULES) };
    const edited = row(0, { clipEdits: { 3: edit } });
    const plain = row(0);

    // below the clip count: neither row is judged on clip 3, the edit is kept
    expect(rowBlockReasons(edited, big, env)).toEqual([]);
    expect(rowBlockReasons(plain, big, env)).toEqual([]);
    expect(buildRowPlan(edited, big, 2, false).clips).toHaveLength(2);
    expect(edited.clipEdits[3]).toBe(edit);

    // with the clip count raised, the edit is used again
    const env4 = { ...env, clipCount: 4 };
    expect(rowBlockReasons(plain, big, env4).map((x) => x.code)).toEqual(["fileTooLarge"]);
    expect(rowBlockReasons(edited, big, env4)).toEqual([]);
    expect(buildRowPlan(edited, big, 4, false).clips[3]?.prompt).toBe("late");
  });

  it("fileTooLarge counts a file once, even when two fields use it", () => {
    const shared = file("P", "same.png", { sizeBytes: 30 * MB });
    const scan = makeScan({
      source: [file("S", "a.mp4")],
      frames: [[field("picture", 1, "P", [shared]), field("picture", 2, "P", [shared])], [field("picture", 1, "P", [shared])]],
    });
    expect(rowBlockReasons(row(0), scan, ENV)).toEqual([
      { code: "fileTooLarge", fileName: "same.png", filePath: "P\\same.png", limitMb: 20 },
    ]);
  });

  it("the same video in a Video and an Audio field is judged against each kind's cap", () => {
    const clip = file("M", "clip.mp4", { sizeBytes: 150 * MB, durationSec: 5 });
    const scan = makeScan({
      source: [file("S", "a.mp4")],
      frames: [[field("video", 1, "M", [clip]), field("audio", 1, "M", [clip])]],
    });
    const audioTight = { ...ENV, maxBytes: { image: 20 * MB, video: 200 * MB, audio: 100 * MB } };
    expect(rowBlockReasons(row(0), scan, audioTight)).toEqual([
      { code: "fileTooLarge", fileName: "clip.mp4", filePath: "M\\clip.mp4", limitMb: 100 },
    ]);
    const bothTight = { ...ENV, maxBytes: { image: 20 * MB, video: 100 * MB, audio: 100 * MB } };
    expect(rowBlockReasons(row(0), scan, bothTight)).toHaveLength(2);
  });
});

describe("rowIgnoredClips (⚠)", () => {
  it("a clip whose videos go past the 15 s total", () => {
    const scan = makeScan({
      frames: [
        [
          field("video", 1, "V1", [file("V1", "long.mp4", { durationSec: 16 })]),
          field("video", 2, "V2", [file("V2", "more.mp4", { durationSec: 3 })]),
        ],
      ],
    });
    expect(rowIgnoredClips(row(0), scan, 2, true)).toEqual([0, 1]);
    expect(rowIgnoredClips(row(0), scan, 2, false)).toEqual([0]);
  });

  it("a clip with more filled cards than the mixed total", () => {
    const rules = { ...RULES, maxTotal: 2 };
    const scan = makeScan({
      rules,
      frames: [
        [
          field("picture", 1, "P1", [file("P1", "a.png")]),
          field("picture", 2, "P2", [file("P2", "b.png")]),
          field("picture", 3, "P3", [file("P3", "c.png")]),
        ],
        [field("picture", 1, "Q", [file("Q", "q.png")])],
      ],
    });
    expect(rowIgnoredClips(row(0), scan, 2, true)).toEqual([0]);
  });
});

describe("rowHasUsedReference / buildRowPlan", () => {
  it("rowHasUsedReference looks at clips 1..N only", () => {
    const scan = makeScan({ source: [file("S", "a.mp4")], frames: [[], [field("picture", 1, "Q", [file("Q", "q.png")])]] });
    expect(rowHasUsedReference(row(0), scan, 1, true)).toBe(false);
    expect(rowHasUsedReference(row(0), scan, 2, true)).toBe(true);
  });

  it("buildRowPlan carries the source, the main prompt, N clips and the output name", () => {
    const scan = standardScan();
    const edit = { promptMode: "replace" as const, prompt: "p", refs: createInitialState(RULES) };
    const plan = buildRowPlan(row(1, { mainPrompt: "m", clipEdits: { 0: edit } }), scan, 3, true);
    expect(plan.index).toBe(1);
    expect(plan.sourcePath).toBe("S\\b.mp4");
    expect(plan.mainPrompt).toBe("m");
    expect(plan.clips).toHaveLength(3);
    expect(plan.clips[0]).toEqual({ promptMode: "replace", prompt: "p", refs: edit.refs });
    expect(plan.clips[2]?.refs.slots.picture[0]?.fileName).toBe("q2.png");
    expect(plan.outFileName).toBe("b.mp4");
  });

  it("buildRowPlan without an opening video for the row", () => {
    const plan = buildRowPlan(row(1), standardScan([file("S", "a.mp4")]), 2, true);
    expect(plan.sourcePath).toBeNull();
    expect(plan.outFileName).toBe("2.mp4");
  });
});
