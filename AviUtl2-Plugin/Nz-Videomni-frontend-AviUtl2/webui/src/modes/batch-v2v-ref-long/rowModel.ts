/**
 * Batch v2v-ref-long: the rules of one row — its clips' references, its 🚫
 * and ⚠ judgements and the plan the runner sends. Pure. Clip indices are
 * 0-based; only clips below the clip count are ever judged or sent (edits of
 * later clips stay in the row, unused).
 */
import {
  REFERENCE_KINDS,
  countUsedReferences,
  createInitialState,
  summarizeReference,
} from "../single/referenceRules";
import type { ReferenceRules, ReferenceSlot, ReferenceState } from "../single/referenceRules";
import { frameIndexForClip, outFileNameOf } from "./folderRules";
import type { ResolvedClip, RowPlan, ScanResult, ScannedFile, V2vRow, V2vRowBlockReason } from "./types";

/** The clip's cards as the folders assign them: Enable on, no A2V, the
 * `rowIndex`-th file of each field of the clip's frame in the card of the same
 * kind and position (`ready`, no upload id yet). */
export function initialClipRefs(scan: ScanResult, rowIndex: number, clip: number, fillFollowing: boolean): ReferenceState {
  const base = createInitialState(scan.rules);
  const state: ReferenceState = { ...base, enabled: true, a2vSlot: null };
  const frame = frameIndexForClip(scan.frames.length, fillFollowing, clip);
  if (frame === null) return state;
  const slots = { picture: [...state.slots.picture], video: [...state.slots.video], audio: [...state.slots.audio] };
  let addedAt = 0;
  for (const field of scan.frames[frame] ?? []) {
    const file = field.files[rowIndex];
    const current = slots[field.kind][field.slot - 1];
    if (!file || !current) continue;
    addedAt += 1;
    const card: ReferenceSlot = {
      kind: field.kind,
      slot: field.slot,
      status: "ready",
      id: null,
      fileName: file.name,
      filePath: file.path,
      durationSec: field.kind === "picture" ? null : file.durationSec,
      thumbnailDataUrl: null,
      addedAt,
      trimFailed: false,
      errorCode: null,
    };
    slots[field.kind][field.slot - 1] = card;
  }
  return { ...state, slots };
}

/** The clip as the row uses it: the edit's prompt (empty "add" without one)
 * and its cards, or the folder assignment while the cards were never touched. */
export function effectiveClip(row: V2vRow, scan: ScanResult, clip: number, fillFollowing: boolean): ResolvedClip {
  const edit = row.clipEdits[clip];
  return {
    promptMode: edit?.promptMode ?? "add",
    prompt: edit?.prompt ?? "",
    refs: edit?.refs ?? initialClipRefs(scan, row.index, clip, fillFollowing),
  };
}

/** The row's opening video, if any. */
export function rowSourceFile(scan: ScanResult, rowIndex: number): ScannedFile | null {
  return scan.source[rowIndex] ?? null;
}

/** Everything a row judgement reads besides the row and the scan. */
export interface RowEnv {
  clipCount: number;
  /** The Chain form's clip 1 `numFrames`. */
  clip0Frames: number;
  /** `minFramesForOverlap` — clip 1's floor with an opening video. */
  minFramesWithSource: number;
  /** `fixedOverlapFrames` (22 on H3). */
  contextFrames: number;
  frameRate: number;
  /** Upload caps in bytes. */
  maxBytes: { image: number; video: number; audio: number };
  fillFollowing: boolean;
}

function clipRange(clipCount: number): number[] {
  return Array.from({ length: Math.max(0, clipCount) }, (_, i) => i);
}

/** The ready cards the generation uses (greyed cards and the A2V slot left
 * out) — the ones that are uploaded and sent. */
export function usedReadyCards(refs: ReferenceState, rules: ReferenceRules): ReferenceSlot[] {
  const summary = summarizeReference(refs, rules);
  const out: ReferenceSlot[] = [];
  for (const kind of REFERENCE_KINDS) {
    refs.slots[kind].forEach((s, i) => {
      const use = summary.slotUse[kind][i];
      if (use?.used && !use.isA2v && s.status === "ready" && s.filePath !== null) out.push(s);
    });
  }
  return out;
}

function usedCount(refs: ReferenceState, scan: ScanResult): number {
  return countUsedReferences(summarizeReference(refs, scan.rules));
}

function sizeByPath(scan: ScanResult): Map<string, number> {
  const sizes = new Map<string, number>();
  for (const file of scan.source) sizes.set(file.path, file.sizeBytes);
  for (const fields of scan.frames) {
    for (const field of fields) for (const file of field.files) sizes.set(file.path, file.sizeBytes);
  }
  return sizes;
}

const BYTES_PER_MB = 1024 * 1024;

/** The 🚫 reasons of a row (empty = runnable). */
export function rowBlockReasons(row: V2vRow, scan: ScanResult, env: RowEnv): V2vRowBlockReason[] {
  const reasons: V2vRowBlockReason[] = [];
  const source = rowSourceFile(scan, row.index);

  if (!source && env.clipCount === 1) reasons.push({ code: "noSourceSingleClip" });
  if (
    source &&
    source.durationSec !== null &&
    source.durationSec > 0 &&
    Math.round(source.durationSec * env.frameRate) < env.contextFrames
  ) {
    reasons.push({ code: "sourceTooShort", frames: env.contextFrames });
  }
  if (source && env.clip0Frames < env.minFramesWithSource) {
    reasons.push({ code: "clip0TooShort", minFrames: env.minFramesWithSource });
  }

  const sizes = sizeByPath(scan);
  const seen = new Set<string>();
  const checkSize = (category: "image" | "video" | "audio", fileName: string, filePath: string) => {
    const key = `${category}:${filePath}`;
    if (seen.has(key)) return;
    seen.add(key);
    const size = sizes.get(filePath);
    const limit = env.maxBytes[category];
    if (size !== undefined && size > limit) {
      reasons.push({ code: "fileTooLarge", fileName, filePath, limitMb: limit / BYTES_PER_MB });
    }
  };
  if (source) checkSize("video", source.name, source.path);
  let anyUsed = false;
  for (const clip of clipRange(env.clipCount)) {
    const { refs } = effectiveClip(row, scan, clip, env.fillFollowing);
    const used = usedReadyCards(refs, scan.rules);
    for (const s of used) {
      if (s.filePath === null) continue;
      checkSize(s.kind === "picture" ? "image" : s.kind, s.fileName ?? s.filePath, s.filePath);
    }
    if (usedCount(refs, scan) > 0) anyUsed = true;
  }

  if (!source && !anyUsed) reasons.push({ code: "noMaterial" });
  return reasons;
}

/** ⚠: the 0-based clips (below the clip count) with a card left out. */
export function rowIgnoredClips(row: V2vRow, scan: ScanResult, clipCount: number, fillFollowing: boolean): number[] {
  return clipRange(clipCount).filter((clip) => {
    const { refs } = effectiveClip(row, scan, clip, fillFollowing);
    const summary = summarizeReference(refs, scan.rules);
    return REFERENCE_KINDS.some((kind) => summary.slotUse[kind].some((use) => use.ignored !== null));
  });
}

/** Some clip below the clip count uses a reference (the FL2VA judgement). */
export function rowHasUsedReference(row: V2vRow, scan: ScanResult, clipCount: number, fillFollowing: boolean): boolean {
  return clipRange(clipCount).some((clip) => usedCount(effectiveClip(row, scan, clip, fillFollowing).refs, scan) > 0);
}

/** What the runner sends for the row. */
export function buildRowPlan(row: V2vRow, scan: ScanResult, clipCount: number, fillFollowing: boolean): RowPlan {
  return {
    index: row.index,
    sourcePath: rowSourceFile(scan, row.index)?.path ?? null,
    mainPrompt: row.mainPrompt,
    clips: clipRange(clipCount).map((clip) => effectiveClip(row, scan, clip, fillFollowing)),
    outFileName: outFileNameOf(scan, row.index) ?? `row_${row.index + 1}.mp4`,
  };
}
