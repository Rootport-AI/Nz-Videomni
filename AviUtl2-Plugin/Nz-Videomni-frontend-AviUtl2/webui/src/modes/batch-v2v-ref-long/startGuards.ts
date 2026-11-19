/**
 * Batch v2v-ref-long: why Start is blocked. Pure.
 */
import type { TransformerRole } from "../../api/types";
import { composeBatchPrompt } from "../batch-i2v-long/buildI2vLongPayload";
import type { ChainValidityReason } from "../chained/useChainForm";
import { sameScanInputs } from "./folderRules";
import { rowBlockReasons, rowHasUsedReference } from "./rowModel";
import type { RowEnv } from "./rowModel";
import type { V2vRefLongChainSource, V2vRefLongState } from "./types";

export type V2vBlockReason =
  | "notScanned"
  | "scanStale"
  | "outDirMissing"
  | "noRunnableRows"
  | "fl2vaLoaded"
  | "sourceAudioAttached"
  | "promptEmpty"
  | "unknownLoraTag"
  | "clipTooShortForOverlap"
  | "jobActive"
  | "lockedByOther";

/**
 * Which of the Chain form's own reasons the batch keeps. `keep` = a setting
 * that applies to every row. `drop` = re-judged per row (`promptEmpty`, the
 * clip-1 floor, the clip count → 🚫), re-computed for clips 2+
 * (`clipTooShortForOverlap`), or about Chain-screen material the template
 * strips (opening material, audio, IC-LoRA reference video, end source).
 */
export const CHAIN_REASON_POLICY: Record<ChainValidityReason, "keep" | "drop"> = {
  promptEmpty: "drop",
  dimensionsOffGrid: "keep",
  clipFramesOffGrid: "keep",
  clipTooShortForOverlap: "drop",
  minClips: "drop",
  totalFramesExceeded: "keep",
  sourceUploading: "drop",
  sourceNotReady: "drop",
  sourceVideoTooShortForContext: "drop",
  sourceTrimFailed: "drop",
  contextFramesTooLongForWindow: "drop",
  audioConflictsWithSourceVideo: "drop",
  audioUploading: "drop",
  audioNotReady: "drop",
  audioTooLong: "drop",
  audioTooShortForChain: "drop",
  audioTooShort: "drop",
  chainLayoutInvalid: "drop",
  referenceNeedsLoras: "drop",
  controlLoraNeedsReference: "keep",
  referenceUploading: "drop",
  referenceNotReady: "drop",
  referenceTrimFailed: "drop",
  referenceConflictsWithSourceVideo: "drop",
  referenceDimensionsOffGrid: "keep",
  depthChainUnsupported: "drop",
  endSourceUploading: "drop",
  endSourceNotReady: "drop",
  endSourceTrimFailed: "drop",
  endSourceConflictsWithAudio: "drop",
  endSourceConflictsWithReference: "drop",
  endSourceTooShort: "drop",
  endSourceLengthUnknown: "drop",
  endSourceNeedsOverlap: "drop",
  endSourceAudioOverlapBudget: "drop",
  materialInClosedPanel: "drop",
  cropInvalid: "keep",
  nagNegativeEmpty: "keep",
};

/** The Chain form's reasons the batch keeps, in the form's order. */
export function chainSettingReasons(chain: Pick<V2vRefLongChainSource, "validityReasons">): ChainValidityReason[] {
  return chain.validityReasons.filter((code) => CHAIN_REASON_POLICY[code] === "keep");
}

/** The part of the Chain form the clip-floor check reads. */
export interface ClipFloorSource {
  clips: readonly { readonly numFrames: number }[];
  minFramesForOverlap: number;
}

/** Clips 2+ (index ≥ 1) shorter than the fixed-overlap floor. Clip 1's own
 * floor is a per-row 🚫 (`clip0TooShort`). */
export function clipsTooShortForOverlap(chain: ClipFloorSource): boolean {
  return chain.clips.some((clip, index) => index >= 1 && clip.numFrames < chain.minFramesForOverlap);
}

const RUNNABLE_STATS = new Set(["Waiting", "Failed", "Generating"]);

/** 0-based rows that a run would process: Waiting/Failed/Generating, no 🚫. */
export function runnableRowIndices(state: V2vRefLongState, env: RowEnv): number[] {
  const scan = state.scan;
  if (!scan) return [];
  return state.rows
    .filter((row) => RUNNABLE_STATS.has(row.stat) && rowBlockReasons(row, scan, env).length === 0)
    .map((row) => row.index);
}

export interface StartGuardInput {
  state: V2vRefLongState;
  chain: ClipFloorSource & Pick<V2vRefLongChainSource, "hasSourceAudio">;
  env: RowEnv;
  /** The template's shared prompt (`<lora:>` already stripped). */
  commonPrompt: string;
  role: TransformerRole | null;
  hasUnknownLoraTag: boolean;
  serverBusy: boolean;
  lockedByOther: boolean;
}

export interface StartGuardResult {
  reasons: V2vBlockReason[];
  /** 0-based runnable rows whose composed main prompt is empty. */
  promptEmptyRows: number[];
  runnableRows: number[];
}

/** The panel's own Start blockers (the kept Chain reasons are
 * {@link chainSettingReasons}). */
export function startBlockReasons(input: StartGuardInput): StartGuardResult {
  const { state, chain, env } = input;
  const reasons: V2vBlockReason[] = [];
  const promptEmptyRows: number[] = [];
  let runnableRows: number[] = [];

  const scan = state.scan;
  if (scan === null) {
    reasons.push("notScanned");
  } else if (!sameScanInputs(scan.inputs, state.inputs)) {
    reasons.push("scanStale");
  } else {
    runnableRows = runnableRowIndices(state, env);
    if (state.inputs.outDir.trim() === "") reasons.push("outDirMissing");
    if (runnableRows.length === 0) reasons.push("noRunnableRows");
    const rowsByIndex = new Map(state.rows.map((row) => [row.index, row]));
    const runnable = runnableRows.flatMap((index) => rowsByIndex.get(index) ?? []);
    if (
      input.role === "fl2va" &&
      runnable.some((row) => rowHasUsedReference(row, scan, env.clipCount, env.fillFollowing))
    ) {
      reasons.push("fl2vaLoaded");
    }
    for (const row of runnable) {
      if (composeBatchPrompt(input.commonPrompt, row.mainPrompt, state.mainPromptMode).trim() === "") {
        promptEmptyRows.push(row.index);
      }
    }
    if (promptEmptyRows.length > 0) reasons.push("promptEmpty");
  }

  if (chain.hasSourceAudio) reasons.push("sourceAudioAttached");
  if (input.hasUnknownLoraTag) reasons.push("unknownLoraTag");
  if (clipsTooShortForOverlap(chain)) reasons.push("clipTooShortForOverlap");
  if (input.serverBusy) reasons.push("jobActive");
  if (input.lockedByOther) reasons.push("lockedByOther");

  return { reasons, promptEmptyRows, runnableRows };
}
