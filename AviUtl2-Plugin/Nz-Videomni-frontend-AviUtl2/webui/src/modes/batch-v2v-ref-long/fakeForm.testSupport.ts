/**
 * Test support for the screen parts: a fake `UseV2vRefLongFormResult` (every
 * operation a `vi.fn`) over a real initial state.
 */
import { vi } from "vitest";
import { createInitialState } from "../single/referenceRules";
import type { ReferenceKind, ReferenceRules, ReferenceSlot, ReferenceState } from "../single/referenceRules";
import { createInitialV2vRefLongState } from "./store";
import type { ResolvedClip, UseV2vRefLongFormResult, V2vRefLongState, V2vRow, V2vRowView } from "./types";

export const RULES: ReferenceRules = {
  maxImages: 9,
  maxVideos: 3,
  maxAudios: 3,
  maxTotal: 12,
  videoTotalSec: 15,
  audioTotalSec: 15,
  clipMinSec: 2,
};

export function emptyRefs(rules: ReferenceRules = RULES): ReferenceState {
  return { ...createInitialState(rules), enabled: true };
}

/** A ready card at `slot` (1-based) holding `filePath`. */
export function withCard(
  refs: ReferenceState,
  kind: ReferenceKind,
  slot: number,
  filePath: string,
  durationSec: number | null = null,
): ReferenceState {
  const card: ReferenceSlot = {
    kind,
    slot,
    status: "ready",
    id: null,
    fileName: filePath.split("\\").pop() ?? filePath,
    filePath,
    durationSec,
    thumbnailDataUrl: null,
    addedAt: slot,
    trimFailed: false,
    errorCode: null,
  };
  return { ...refs, slots: { ...refs.slots, [kind]: refs.slots[kind].map((s) => (s.slot === slot ? card : s)) } };
}

export function makeRow(index: number, patch: Partial<V2vRow> = {}): V2vRow {
  return { index, mainPrompt: "", clipEdits: {}, stat: "Waiting", output: "", error: "", ...patch };
}

export function makeView(row: V2vRow, patch: Partial<Omit<V2vRowView, "row">> = {}): V2vRowView {
  return { row, baseFileName: `file${row.index + 1}.mp4`, hasSource: true, blockReasons: [], ignoredClips: [], ...patch };
}

export interface FakeFormOptions extends Partial<Omit<UseV2vRefLongFormResult, "state">> {
  state?: Partial<V2vRefLongState>;
  /** `row:clip` -> the clip; others resolve to an empty "add" clip. */
  clips?: Record<string, ResolvedClip>;
}

export function makeForm(options: FakeFormOptions = {}): UseV2vRefLongFormResult {
  const { state: statePatch, clips = {}, ...overrides } = options;
  const state: V2vRefLongState = { ...createInitialV2vRefLongState(), ...statePatch };
  const rules = overrides.rules ?? RULES;
  return {
    state,
    rules,
    rowViews: [],
    baseField: null,
    clipCount: 2,
    emptyFields: [],
    allFieldsEmpty: false,
    unusedFrameStart: null,
    filledFolderCount: 0,
    blockReasons: [],
    chainBlockReasons: [],
    promptEmptyRows: [],
    minFramesForOverlap: 39,
    fixedOverlapFrames: 22,
    seedFixed: false,
    summary: { done: 0, total: 0 },
    currentRow: null,
    canStart: false,
    canScan: true,
    inputsLocked: false,
    effectiveClipOf: (row, clip) => clips[`${row}:${clip}`] ?? { promptMode: "add", prompt: "", refs: emptyRefs(rules) },
    setVideoDir: vi.fn(),
    pickVideoDir: vi.fn().mockResolvedValue(undefined),
    setJoined: vi.fn(),
    addField: vi.fn(),
    setFieldPath: vi.fn(),
    pickFieldDir: vi.fn().mockResolvedValue(undefined),
    removeField: vi.fn(),
    addFrame: vi.fn(),
    removeLastFrame: vi.fn(),
    setFillFollowing: vi.fn(),
    setOutDir: vi.fn(),
    pickOutDir: vi.fn().mockResolvedValue(undefined),
    scan: vi.fn().mockResolvedValue(undefined),
    setMainPromptMode: vi.fn(),
    setRowMainPrompt: vi.fn(),
    copyChainPromptToRow: vi.fn(),
    setClipPrompt: vi.fn(),
    setClipPromptMode: vi.fn(),
    attachClipFile: vi.fn().mockResolvedValue(undefined),
    pickClipFile: vi.fn().mockResolvedValue(undefined),
    clearClipSlot: vi.fn(),
    appendClipTag: vi.fn(),
    resetRow: vi.fn(),
    start: vi.fn(),
    stop: vi.fn(),
    ...overrides,
  };
}
