/**
 * Batch v2v-ref-long: the types every part of the feature shares.
 *
 * Numbering: rows, clips and frames are 0-based indices everywhere in code
 * (`V2vRow.index`, `clipEdits` keys, `BaseField.frame`, the clip lists the
 * judgements return). Only the field position `slot` is 1-based — it is the
 * tag number (`<Picture 1>`), as in `ReferenceSlot.slot`. The copy in
 * `strings.batchV2vRefLong` takes display numbers (index + 1).
 */
import type { GenerateChainRequest } from "../../api/types";
import type { PromptMode } from "../batch-i2v-long/buildI2vLongPayload";
import type { ChainValidityReason } from "../chained/useChainForm";
import type { ReferenceKind, ReferenceRules, ReferenceState } from "../single/referenceRules";
import type { V2vBlockReason } from "./startGuards";

export type { PromptMode };

// --- inputs ------------------------------------------------------------------

/** One folder field of a frame. `path === ""` (after trim) = not used. */
export interface FolderField {
  /** Unique within its frame. */
  id: number;
  kind: ReferenceKind;
  path: string;
}

/** The folder fields of one clip. */
export interface FolderFrame {
  id: number;
  fields: FolderField[];
}

/** What a scan reads. */
export interface ScanInputs {
  videoDir: string;
  frames: FolderFrame[];
}

export interface V2vRefLongInputs extends ScanInputs {
  /** One value for the whole panel (shown on the last frame only). Default
   * `true`. Not a scan input: toggling it takes effect without a re-scan. */
  fillFollowing: boolean;
  /** Default `true`. */
  joined: boolean;
  outDir: string;
  outDirIsAuto: boolean;
}

// --- scan result ---------------------------------------------------------------

export interface ScannedFile {
  name: string;
  path: string;
  sizeBytes: number;
  /** Measured for videos and audios; `null` for a picture or when unmeasurable. */
  durationSec: number | null;
}

/** One filled field's listing (natural name order). */
export interface ScannedField {
  kind: ReferenceKind;
  /** 1-based position among the frame's fields of the same kind (= tag number). */
  slot: number;
  path: string;
  files: ScannedFile[];
}

export interface ScanResult {
  /** A copy of the inputs the scan used. */
  inputs: ScanInputs;
  /** The opening videos (natural name order); `[]` without a video folder. */
  source: ScannedFile[];
  /** Per frame (0-based), the filled fields in the frame's field order. */
  frames: ScannedField[][];
  rules: ReferenceRules;
}

/** The field whose file count decides the rows. `frame` is 0-based, `slot`
 * 1-based. */
export type BaseField = { type: "source" } | { type: "ref"; frame: number; kind: ReferenceKind; slot: number };

// --- rows ----------------------------------------------------------------------

/** A clip the user edited in a row. `refs === null` = the folder assignment
 * (copied in only when a card is touched). */
export interface ClipEdit {
  promptMode: PromptMode;
  prompt: string;
  refs: ReferenceState | null;
}

/** A clip as the row uses it: the edit with its cards resolved. */
export interface ResolvedClip {
  promptMode: PromptMode;
  prompt: string;
  refs: ReferenceState;
}

export type V2vRowStat = "Waiting" | "Generating" | "Done" | "Failed";

export interface V2vRow {
  /** 0-based; = the i-th file of every folder. The table's `#` is index + 1. */
  index: number;
  mainPrompt: string;
  /** Only the clips the user touched, keyed by 0-based clip index. Edits of
   * clips past the Chain form's clip count are kept but not used. */
  clipEdits: Readonly<Record<number, ClipEdit>>;
  stat: V2vRowStat;
  output: string;
  error: string;
}

/** Why a row is 🚫 (skipped). */
export type V2vRowBlockReason =
  | { code: "noSourceSingleClip" }
  | { code: "sourceTooShort"; frames: number }
  | { code: "clip0TooShort"; minFrames: number }
  | { code: "fileTooLarge"; fileName: string; filePath: string; limitMb: number }
  | { code: "noMaterial" };

// --- the whole state -------------------------------------------------------------

export type V2vRunStateName = "idle" | "running" | "stopping";

export interface V2vRunState {
  state: V2vRunStateName;
  lastStartResult: { started: boolean; reason: string } | null;
  /** The clip count frozen at Start; `null` while idle. */
  clipCount: number | null;
}

export interface V2vRefLongState {
  inputs: V2vRefLongInputs;
  scan: ScanResult | null;
  scanning: boolean;
  scanProgress: { done: number; total: number } | null;
  scanError: string | null;
  rows: V2vRow[];
  /** Default `"add"`. */
  mainPromptMode: PromptMode;
  run: V2vRunState;
}

// --- what the panel reads from the Chain form ------------------------------------

/** `UseChainFormResult` satisfies this structurally (pinned in
 * `buildPayload.test.ts`). */
export interface V2vRefLongChainSource {
  buildRequest(): GenerateChainRequest;
  clips: readonly { readonly prompt: string; readonly numFrames: number }[];
  minFramesForOverlap: number;
  fixedOverlapFrames: number | null;
  common: { frameRate: number };
  hasSourceAudio: boolean;
  validityReasons: readonly ChainValidityReason[];
}

// --- what the runner gets per row -------------------------------------------------

export interface RowPlan {
  index: number;
  sourcePath: string | null;
  mainPrompt: string;
  /** Clips 0..N-1 of the run's clip count. */
  clips: ResolvedClip[];
  outFileName: string;
}

// --- what the panel's hook hands the screen ----------------------------------------

/** One row's derived display values. */
export interface V2vRowView {
  row: V2vRow;
  /** The base folder's file name for this row (the table's name column). */
  baseFileName: string | null;
  /** The row has an opening video (🎬). */
  hasSource: boolean;
  /** Non-empty = 🚫 (only shown on Waiting/Failed rows). */
  blockReasons: readonly V2vRowBlockReason[];
  /** 0-based clips (within the clip count) with a greyed card = ⚠. */
  ignoredClips: readonly number[];
}

/** A file picked into a row's clip card by hand (no upload until Start). */
export interface ClipSlotFile {
  fileName: string;
  filePath: string;
  durationSec: number | null;
}

export interface UseV2vRefLongFormResult {
  // --- state ---
  state: V2vRefLongState;
  /** The rules the cards and the field limits are sized from. */
  rules: ReferenceRules;

  // --- derived ---
  rowViews: readonly V2vRowView[];
  /** The field the name column is named after; `null` before a scan or when
   * every folder came back empty. */
  baseField: BaseField | null;
  /** The clip count shown ("Chained clips: N") and used for the row cards —
   * the frozen one while running, else the Chain form's current one. */
  clipCount: number;
  /** Scanned fields (and the opening-video field) with no usable file. */
  emptyFields: readonly BaseField[];
  /** A scan was made and no folder had a usable file. */
  allFieldsEmpty: boolean;
  /** 0-based index of the first frame past the clip count, or `null` when
   * every frame is used. */
  unusedFrameStart: number | null;
  /** Filled folders across all frames (the accordion heading). */
  filledFolderCount: number;
  /** This panel's own Start blockers, in display order. */
  blockReasons: readonly V2vBlockReason[];
  /** The Chain form's own reasons this panel keeps (`chainSettingReasons`). */
  chainBlockReasons: readonly ChainValidityReason[];
  /** 0-based rows behind `promptEmpty`. */
  promptEmptyRows: readonly number[];
  /** For `buildChainReasonMessages`. */
  minFramesForOverlap: number;
  fixedOverlapFrames: number | null;
  /** Seed is fixed (not -1) — the advisory note. */
  seedFixed: boolean;
  summary: { done: number; total: number };
  currentRow: V2vRow | null;
  canStart: boolean;
  canScan: boolean;
  /** Scanning or running: every input and row edit is disabled. */
  inputsLocked: boolean;
  /** The clip as the row would send it (an edit, else the folder assignment). */
  effectiveClipOf(rowIndex: number, clip: number): ResolvedClip | null;

  // --- inputs ---
  setVideoDir(path: string): void;
  pickVideoDir(title?: string): Promise<void>;
  setJoined(joined: boolean): void;
  addField(frame: number, kind: ReferenceKind): void;
  setFieldPath(frame: number, fieldId: number, path: string): void;
  pickFieldDir(frame: number, fieldId: number, title?: string): Promise<void>;
  addFrame(): void;
  removeLastFrame(): void;
  setFillFollowing(value: boolean): void;
  setOutDir(path: string): void;
  pickOutDir(title?: string): Promise<void>;
  scan(): Promise<void>;

  // --- rows ---
  setMainPromptMode(mode: PromptMode): void;
  setRowMainPrompt(rowIndex: number, prompt: string): void;
  /** 📝: the Chain form's shared prompt into the row's main prompt. */
  copyChainPromptToRow(rowIndex: number): void;
  setClipPrompt(rowIndex: number, clip: number, prompt: string): void;
  setClipPromptMode(rowIndex: number, clip: number, mode: PromptMode): void;
  /** Puts a file into a card (measures its length, no upload). */
  attachClipFile(rowIndex: number, clip: number, kind: ReferenceKind, slot: number, filePath: string, fileName: string): Promise<void>;
  /** 📁 then {@link attachClipFile}; a cancelled dialog does nothing. */
  pickClipFile(rowIndex: number, clip: number, kind: ReferenceKind, slot: number): Promise<void>;
  clearClipSlot(rowIndex: number, clip: number, kind: ReferenceKind, slot: number): void;
  /** ✒: appends the card's tag (as the Reference accordion hands it) to the
   * row's clip prompt. */
  appendClipTag(rowIndex: number, clip: number, tag: string): void;
  /** 🔁: Done/Failed back to Waiting. */
  resetRow(rowIndex: number): void;

  // --- run ---
  start(): void;
  stop(): void;
}
