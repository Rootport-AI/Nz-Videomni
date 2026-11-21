/**
 * Batch v2v-ref-long: the panel's whole state, kept at module level so it
 * survives the panel closing and the Chain screen's `key` remount. The reducer
 * is pure; the store replaces its snapshot on every change, so it can feed
 * `useSyncExternalStore` directly.
 */
import { REFERENCE_KINDS, emptySlot } from "../single/referenceRules";
import type { ReferenceKind, ReferenceSlot, ReferenceState } from "../single/referenceRules";
import {
  canAddField,
  createDefaultFrame,
  deriveV2vRefLongOutDir,
  baseFolderPath,
  filledPath,
  isDefaultField,
  rowCountOf,
  withAddedField,
} from "./folderRules";
import { effectiveClip } from "./rowModel";
import type {
  ClipEdit,
  ClipSlotFile,
  FolderField,
  PromptMode,
  ScanResult,
  V2vRefLongState,
  V2vRow,
  V2vRowStat,
  V2vRunState,
} from "./types";
import type { ReferenceRules } from "../single/referenceRules";

export type V2vRefLongAction =
  // inputs
  | { type: "setVideoDir"; path: string }
  | { type: "setJoined"; joined: boolean }
  | { type: "addField"; frame: number; kind: ReferenceKind; rules: ReferenceRules }
  | { type: "setFieldPath"; frame: number; fieldId: number; path: string }
  /** ❌: a default field's path is cleared, an added field is removed. */
  | { type: "removeField"; frame: number; fieldId: number }
  | { type: "addFrame" }
  | { type: "removeLastFrame" }
  | { type: "setFillFollowing"; value: boolean }
  | { type: "setOutDir"; path: string }
  // scan
  | { type: "scanStarted" }
  | { type: "scanProgress"; done: number; total: number }
  | { type: "scanSucceeded"; result: ScanResult }
  | { type: "scanFailed"; message: string }
  // rows
  | { type: "setMainPromptMode"; mode: PromptMode }
  | { type: "setRowMainPrompt"; row: number; prompt: string }
  | { type: "setClipPrompt"; row: number; clip: number; prompt: string }
  | { type: "setClipPromptMode"; row: number; clip: number; mode: PromptMode }
  | { type: "setClipSlot"; row: number; clip: number; kind: ReferenceKind; slot: number; file: ClipSlotFile | null }
  | { type: "resetRow"; row: number }
  // run
  | { type: "rowPatched"; row: number; patch: { stat?: V2vRowStat; output?: string; error?: string } }
  | { type: "runStateChanged"; patch: Partial<V2vRunState> };

export function createInitialV2vRefLongState(): V2vRefLongState {
  return {
    inputs: {
      videoDir: "",
      frames: [createDefaultFrame(1)],
      fillFollowing: true,
      joined: true,
      outDir: "",
      outDirIsAuto: true,
    },
    scan: null,
    scanning: false,
    scanProgress: null,
    scanError: null,
    rows: [],
    mainPromptMode: "add",
    run: { state: "idle", lastStartResult: null, clipCount: null },
  };
}

function freshRow(index: number): V2vRow {
  return { index, mainPrompt: "", clipEdits: {}, stat: "Waiting", output: "", error: "" };
}

function updateRow(state: V2vRefLongState, index: number, update: (row: V2vRow) => V2vRow): V2vRefLongState {
  const current = state.rows.find((r) => r.index === index);
  if (!current) return state;
  const next = update(current);
  if (next === current) return state;
  return { ...state, rows: state.rows.map((r) => (r === current ? next : r)) };
}

/** Applies `update` to the row's clip edit (an empty "add" edit with
 * `refs: null` when there is none). A no-op before a scan. */
function updateClip(
  state: V2vRefLongState,
  rowIndex: number,
  clip: number,
  update: (edit: ClipEdit, row: V2vRow, scan: ScanResult) => ClipEdit,
): V2vRefLongState {
  const scan = state.scan;
  if (!scan) return state;
  return updateRow(state, rowIndex, (row) => {
    const current: ClipEdit = row.clipEdits[clip] ?? { promptMode: "add", prompt: "", refs: null };
    return { ...row, clipEdits: { ...row.clipEdits, [clip]: update(current, row, scan) } };
  });
}

function nextAddedAt(refs: ReferenceState): number {
  let max = 0;
  for (const kind of REFERENCE_KINDS) {
    for (const s of refs.slots[kind]) if (s.addedAt !== null && s.addedAt > max) max = s.addedAt;
  }
  return max + 1;
}

function withSlot(refs: ReferenceState, kind: ReferenceKind, slot: number, file: ClipSlotFile | null): ReferenceState {
  const row = refs.slots[kind];
  if (!row[slot - 1]) return refs;
  const card: ReferenceSlot = file
    ? {
        ...emptySlot(kind, slot),
        status: "ready",
        fileName: file.fileName,
        filePath: file.filePath,
        durationSec: kind === "picture" ? null : file.durationSec,
        addedAt: nextAddedAt(refs),
      }
    : emptySlot(kind, slot);
  return { ...refs, slots: { ...refs.slots, [kind]: row.map((s, i) => (i === slot - 1 ? card : s)) } };
}

function withInputs(state: V2vRefLongState, patch: Partial<V2vRefLongState["inputs"]>): V2vRefLongState {
  return { ...state, inputs: { ...state.inputs, ...patch } };
}

export function reduceV2vRefLong(state: V2vRefLongState, action: V2vRefLongAction): V2vRefLongState {
  switch (action.type) {
    case "setVideoDir":
      return withInputs(state, { videoDir: action.path });
    case "setJoined":
      return withInputs(state, { joined: action.joined });
    case "addField": {
      const frame = state.inputs.frames[action.frame];
      if (!frame || !canAddField(frame, action.kind, action.rules)) return state;
      const next = withAddedField(frame, action.kind);
      return withInputs(state, { frames: state.inputs.frames.map((f) => (f === frame ? next : f)) });
    }
    case "setFieldPath": {
      const frame = state.inputs.frames[action.frame];
      if (!frame || !frame.fields.some((f) => f.id === action.fieldId)) return state;
      const next = {
        ...frame,
        fields: frame.fields.map((f) => (f.id === action.fieldId ? { ...f, path: action.path } : f)),
      };
      return withInputs(state, { frames: state.inputs.frames.map((f) => (f === frame ? next : f)) });
    }
    case "removeField": {
      const frame = state.inputs.frames[action.frame];
      const field = frame?.fields.find((f) => f.id === action.fieldId);
      if (!frame || !field) return state;
      let fields: FolderField[];
      if (isDefaultField(frame, field.id)) {
        if (field.path === "") return state;
        fields = frame.fields.map((f) => (f === field ? { ...f, path: "" } : f));
      } else {
        // The later fields of the kind move up a tag number.
        fields = frame.fields.filter((f) => f !== field);
      }
      const next = { ...frame, fields };
      return withInputs(state, { frames: state.inputs.frames.map((f) => (f === frame ? next : f)) });
    }
    case "addFrame": {
      const id = state.inputs.frames.reduce((max, f) => Math.max(max, f.id), 0) + 1;
      return withInputs(state, { frames: [...state.inputs.frames, createDefaultFrame(id)] });
    }
    case "removeLastFrame":
      if (state.inputs.frames.length < 2) return state;
      return withInputs(state, { frames: state.inputs.frames.slice(0, -1) });
    case "setFillFollowing":
      return withInputs(state, { fillFollowing: action.value });
    case "setOutDir":
      // Emptied (blank) = back to auto: the next scan fills it in again.
      return withInputs(state, { outDir: action.path, outDirIsAuto: filledPath(action.path) === "" });

    case "scanStarted":
      return { ...state, scanning: true, scanProgress: null, scanError: null };
    case "scanProgress":
      return { ...state, scanProgress: { done: action.done, total: action.total } };
    case "scanSucceeded": {
      const result = action.result;
      const rows = Array.from({ length: rowCountOf(result) }, (_, i) => freshRow(i));
      let inputs = state.inputs;
      if (inputs.outDirIsAuto) {
        const base = baseFolderPath(result);
        inputs = { ...inputs, outDir: base ? deriveV2vRefLongOutDir(base) : "" };
      }
      return { ...state, inputs, scan: result, rows, scanning: false, scanProgress: null, scanError: null };
    }
    case "scanFailed":
      return { ...state, scanning: false, scanProgress: null, scanError: action.message };

    case "setMainPromptMode":
      return state.mainPromptMode === action.mode ? state : { ...state, mainPromptMode: action.mode };
    case "setRowMainPrompt":
      return updateRow(state, action.row, (row) => ({ ...row, mainPrompt: action.prompt }));
    case "setClipPrompt":
      return updateClip(state, action.row, action.clip, (edit) => ({ ...edit, prompt: action.prompt }));
    case "setClipPromptMode":
      return updateClip(state, action.row, action.clip, (edit) => ({ ...edit, promptMode: action.mode }));
    case "setClipSlot":
      // The first card touch copies the folder assignment in.
      return updateClip(state, action.row, action.clip, (edit, row, scan) => ({
        ...edit,
        refs: withSlot(
          edit.refs ?? effectiveClip(row, scan, action.clip, state.inputs.fillFollowing).refs,
          action.kind,
          action.slot,
          action.file,
        ),
      }));
    case "resetRow":
      return updateRow(state, action.row, (row) =>
        row.stat === "Done" || row.stat === "Failed" ? { ...row, stat: "Waiting", output: "", error: "" } : row,
      );

    case "rowPatched":
      return updateRow(state, action.row, (row) => ({ ...row, ...action.patch }));
    case "runStateChanged": {
      const run = { ...state.run, ...action.patch };
      if (run.state === "idle") run.clipCount = null;
      return { ...state, run };
    }
  }
}

// --- the module-level store ----------------------------------------------------

let snapshot: V2vRefLongState = createInitialV2vRefLongState();
const listeners = new Set<() => void>();

function notify(): void {
  for (const listener of listeners) listener();
}

export const v2vRefLongStore = {
  getSnapshot(): V2vRefLongState {
    return snapshot;
  },
  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  },
  dispatch(action: V2vRefLongAction): void {
    const next = reduceV2vRefLong(snapshot, action);
    if (next === snapshot) return;
    snapshot = next;
    notify();
  },
};

/** Test-only: back to the initial state (subscribers are notified). */
export function __resetV2vRefLongStoreForTests(): void {
  snapshot = createInitialV2vRefLongState();
  notify();
}
