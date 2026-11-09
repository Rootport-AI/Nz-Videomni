/**
 * §1-120: MiniMax H3's long A2V on Chained — the pure parts.
 *
 * ONE long audio is uploaded on the Chained screen's A2V card and
 * "distributed" to every clip: each clip's Reference accordion gets the SAME
 * audio id in one of its Audios cards, with "Audio to video" checked, and the
 * request sends `clips[i].source_audio = {audio_id, strength, offset_frames}`
 * with `offset_frames = S_i` (the clip's start on the chain). Nothing is cut or
 * re-uploaded. Whether a card holds the distributed audio is never stored: it
 * is `slot.id === longAudioId` (every upload returns a fresh id, so a file the
 * user puts in by hand never matches).
 *
 * Everything here is pure: the screen (`ChainedScreen.tsx`) owns the state and
 * calls these.
 */
import type { ChainClipSourceAudio } from "../../api/types";
import type { FrameGrid } from "../single/paramUtils";
import { floorToGrid } from "../single/paramUtils";
import { singleAudioPrecheck } from "../single/audioFit";
import type { ReferenceKind, ReferenceSlot, ReferenceSlotStatus, ReferenceSummary } from "../single/referenceRules";

/** Same tolerance `audioFit.ts` uses when turning seconds into frames. */
const FRAME_EPSILON = 1e-6;

// --- the distribution plan ------------------------------------------------------

export type H3AudioDistributionOutcome =
  | "alreadyFits"
  | "extended"
  | "shrunk"
  | "cappedAtMaxClips"
  | "tooShortForChain";

export interface H3AudioDistributionInput {
  /** The current clips' `numFrames`, in order. */
  clipFrames: readonly number[];
  /** Parallel to {@link clipFrames}: the card's `intact` flag
   * (`chainUtils.isClipIntact` — `false` once the user touched it). */
  intact: readonly boolean[];
  audioSec: number;
  fps: number;
  /** The clip grid with its bounds (`17n+5`, `[22, 481]` on H3). */
  grid: FrameGrid & { min: number; max: number };
  /** The server-fixed seam (22). */
  overlap: number;
  /** The floor of every clip after the first (`overlap + grid.step` = 39). */
  laterMin: number;
  /** An added clip never runs longer than this many seconds (10). */
  addedMaxSec: number;
  maxClips: number;
  minClips: number;
}

export interface H3AudioDistributionPlan {
  /** The new clip lengths, in order (existing clips first, added ones after). */
  frames: number[];
  /** How many clips were appended. */
  added: number;
  /** Existing clips whose length went down (0-based `index`). */
  shrunk: { index: number; from: number; to: number }[];
  /** Existing clips removed (0-based, contiguous from the end, ascending). */
  removed: number[];
  /** A clip the user touched (`intact` false) is shortened or removed. */
  needsConfirm: boolean;
  /** Audio frames left past the end of the chain (never negative). */
  unusedFrames: number;
  outcome: H3AudioDistributionOutcome;
}

/** The output length of a chain: `ΣF − overlap × (n − 1)`. */
export function chainTotal(frames: readonly number[], overlap: number): number {
  if (frames.length === 0) return 0;
  return frames.reduce((sum, f) => sum + f, 0) - overlap * (frames.length - 1);
}

/**
 * Lays the clip list over the long audio (plan §7.3):
 * - the chain is SHORTER than the audio → clips are appended, each the largest
 *   grid value within `addedMaxSec` (226 at 24 fps on 17n+5) and the last one
 *   the largest grid value not past the audio's end; existing clips are left
 *   as they are;
 * - LONGER → clips are shortened from the back, and a later clip that would
 *   fall under `laterMin` is removed (down to `minClips`).
 * `A = floor(audioSec × fps)` — at most 0.001 s stricter than the server's
 * `duration + 0.001 ≥ end / fps`.
 */
export function planH3AudioDistribution(input: H3AudioDistributionInput): H3AudioDistributionPlan {
  const { clipFrames, intact, audioSec, fps, grid, overlap, laterMin, addedMaxSec, maxClips, minClips } = input;
  const A = audioFramesOf(audioSec, fps);
  const frames = [...clipFrames];
  const unchanged = (outcome: H3AudioDistributionOutcome): H3AudioDistributionPlan => ({
    frames,
    added: 0,
    shrunk: [],
    removed: [],
    needsConfirm: false,
    unusedFrames: Math.max(0, A - chainTotal(frames, overlap)),
    outcome,
  });

  if (A < grid.min + laterMin - overlap) return unchanged("tooShortForChain");

  let n = frames.length;
  let T = chainTotal(frames, overlap);

  if (T <= A) {
    const addMax = Math.min(floorToGrid(Math.floor(addedMaxSec * fps + FRAME_EPSILON), grid), grid.max);
    let added = 0;
    while (n < maxClips) {
      const room = A - T + overlap;
      const F = floorToGrid(Math.min(room, addMax), grid);
      if (F < laterMin || F > room) break;
      frames.push(F);
      n += 1;
      T += F - overlap;
      added += 1;
    }
    const unusedFrames = Math.max(0, A - T);
    const outcome: H3AudioDistributionOutcome =
      n >= maxClips && unusedFrames >= grid.step ? "cappedAtMaxClips" : added === 0 ? "alreadyFits" : "extended";
    return { frames, added, shrunk: [], removed: [], needsConfirm: false, unusedFrames, outcome };
  }

  // Shorten from the back.
  const original = [...clipFrames];
  const removed: number[] = [];
  let i = n - 1;
  while (T > A && i >= 0) {
    const current = frames[i] as number;
    const minF = i === 0 ? grid.min : laterMin;
    const target = current - (T - A);
    if (target >= minF) {
      const next = floorToGrid(target, grid);
      T -= current - next;
      frames[i] = next;
      break;
    } else if (n > minClips && i > 0) {
      T -= current - overlap;
      frames.splice(i, 1);
      removed.unshift(i);
      n -= 1;
      i -= 1;
    } else {
      T -= current - minF;
      frames[i] = minF;
      i -= 1;
    }
  }
  const shrunk: { index: number; from: number; to: number }[] = [];
  frames.forEach((f, index) => {
    const from = original[index] as number;
    if (f < from) shrunk.push({ index, from, to: f });
  });
  const touched = (index: number) => intact[index] === false;
  const needsConfirm = shrunk.some((s) => touched(s.index)) || removed.some(touched);
  return {
    frames,
    added: 0,
    shrunk,
    removed,
    needsConfirm,
    unusedFrames: Math.max(0, A - T),
    outcome: "shrunk",
  };
}

// --- the slots -------------------------------------------------------------------

/** What one clip's Reference cards tell the screen (`ClipReferenceEntry`'s
 * readable half). */
export interface H3ClipSlotsView {
  summary: ReferenceSummary;
  slots: Record<ReferenceKind, ReferenceSlot[]>;
}

/** The declared "Audio to video" slot (1-based), filled or not, or `null`. */
export function declaredA2vSlotOf(entry: H3ClipSlotsView): number | null {
  const index = entry.summary.slotUse.audio.findIndex((use) => use.isA2v);
  if (index < 0) return null;
  return entry.slots.audio[index]?.slot ?? null;
}

/** The declared "Audio to video" card itself (filled or not), or `null`
 * (also for a clip with no registered cards). */
export function declaredA2vCardOf(entry: H3ClipSlotsView | undefined): ReferenceSlot | null {
  if (!entry) return null;
  const declared = declaredA2vSlotOf(entry);
  return declared === null ? null : (entry.slots.audio[declared - 1] ?? null);
}

/**
 * Whether a clip's ready A2V audio is too short — the ONE judgement behind both
 * the card's warning and the Generate gate:
 * - `"long"`: the card holds the long audio and the clip ends past it
 *   (`start + numFrames > longAudioFrames`, per clip);
 * - `"hand"`: an audio put in by hand that does not cover the clip
 *   (`!singleAudioPrecheck`);
 * - `null`: long enough, not ready, or a length that is unknown.
 */
export function clipA2vTooShort(input: {
  card: { status: ReferenceSlotStatus; id: string | null; durationSec: number | null } | null;
  start: number;
  numFrames: number;
  longAudioId: string | null;
  longAudioFrames: number | null;
  fps: number;
}): "long" | "hand" | null {
  const { card, start, numFrames, longAudioId, longAudioFrames, fps } = input;
  if (!card || card.status !== "ready" || card.id === null) return null;
  if (longAudioId !== null && card.id === longAudioId) {
    return longAudioFrames !== null && start + numFrames > longAudioFrames ? "long" : null;
  }
  return card.durationSec !== null && !singleAudioPrecheck(card.durationSec, numFrames, fps) ? "hand" : null;
}

/** Where the distribution puts the long audio in one clip: the slot that
 * already holds it → the first free slot (`empty | error`) → the declared A2V
 * slot (its hand-put audio is replaced) → `null` (nothing is overwritten). */
export function pickDistributionSlot(entry: H3ClipSlotsView, longAudioId: string): number | null {
  const audio = entry.slots.audio;
  const same = audio.find((s) => s.id === longAudioId && s.status === "ready");
  if (same) return same.slot;
  const free = audio.find((s) => s.status === "empty" || s.status === "error");
  if (free) return free.slot;
  return declaredA2vSlotOf(entry);
}

/** Whether any audio card of the clip holds the long audio. */
export function holdsLongAudio(entry: H3ClipSlotsView, longAudioId: string): boolean {
  return entry.slots.audio.some((s) => s.id === longAudioId);
}

/**
 * `clips[i].source_audio` for one clip: a declared A2V slot that is ready with
 * an id, else `null`. `offset_frames` is the clip's start `S_i` when the slot
 * holds the long audio, `0` for an audio put in by hand.
 */
export function clipSourceAudioBody(
  entry: H3ClipSlotsView,
  opts: { longAudioId: string | null; start: number; strength: number },
): ChainClipSourceAudio | null {
  const slot = declaredA2vCardOf(entry);
  if (!slot || slot.status !== "ready" || slot.id === null) return null;
  return {
    audio_id: slot.id,
    strength: opts.strength,
    offset_frames: opts.longAudioId !== null && slot.id === opts.longAudioId ? opts.start : 0,
  };
}

// --- the Generate gate -------------------------------------------------------------

export type H3ChainA2vReasonCode =
  | "chainA2vReferencesOff"
  | "chainA2vNotDistributed"
  | "chainA2vAudioTooShort"
  | "clipA2vAudioTooShort"
  | "clipA2vAudioMissing"
  | "clipA2vConflictsWithSourceVideo";

export interface H3ChainA2vReasonClip {
  numFrames: number;
  /** `S_i` (`chainUtils.clipStartFrames`). */
  start: number;
  /** The declared A2V slot's content, or `null` when none is declared. */
  a2v: { status: ReferenceSlotStatus; id: string | null; durationSec: number | null } | null;
  /** Some audio card of the clip holds the long audio. */
  holdsLongAudio: boolean;
}

export interface H3ChainA2vReasonInput {
  clips: readonly H3ChainA2vReasonClip[];
  referencesOn: boolean;
  /** The long audio's id while its upload is ready, else `null`. */
  longAudioReadyId: string | null;
  /** `floor(duration × fps)` of the long audio, `null` when unknown. */
  audioFrames: number | null;
  hasSourceVideo: boolean;
  fps: number;
}

export interface H3ChainA2vReasons {
  codes: H3ChainA2vReasonCode[];
  /** 1-based clip numbers behind `clipA2vAudioTooShort`. */
  tooShortClips: number[];
  /** 1-based clip numbers behind `clipA2vAudioMissing`. */
  missingClips: number[];
}

/**
 * The reasons that stop Generate in the distribute mode (plan §7.8). With
 * References off, only the first line can fire for the hidden cards.
 */
export function h3ChainA2vReasons(input: H3ChainA2vReasonInput): H3ChainA2vReasons {
  const { clips, referencesOn, longAudioReadyId, audioFrames, hasSourceVideo, fps } = input;
  const codes: H3ChainA2vReasonCode[] = [];
  const tooShortClips: number[] = [];
  const missingClips: number[] = [];
  const anyDeclared = clips.some((c) => c.a2v !== null);

  if (!referencesOn && anyDeclared) codes.push("chainA2vReferencesOff");
  if (longAudioReadyId !== null && !clips.some((c) => c.holdsLongAudio)) codes.push("chainA2vNotDistributed");
  if (referencesOn) {
    let longTooShort = false;
    clips.forEach((c, index) => {
      const a2v = c.a2v;
      if (a2v === null) return;
      if (a2v.status === "empty" || a2v.status === "error") {
        missingClips.push(index + 1);
        return;
      }
      const tooShort = clipA2vTooShort({
        card: a2v,
        start: c.start,
        numFrames: c.numFrames,
        longAudioId: longAudioReadyId,
        longAudioFrames: audioFrames,
        fps,
      });
      if (tooShort === "long") longTooShort = true;
      else if (tooShort === "hand") tooShortClips.push(index + 1);
    });
    if (longTooShort) codes.push("chainA2vAudioTooShort");
    if (tooShortClips.length > 0) codes.push("clipA2vAudioTooShort");
    if (missingClips.length > 0) codes.push("clipA2vAudioMissing");
  }
  if (hasSourceVideo && anyDeclared) codes.push("clipA2vConflictsWithSourceVideo");
  return { codes, tooShortClips, missingClips };
}

/** `floor(audioSec × fps)` — the `A` both the plan and the gate measure in. */
export function audioFramesOf(audioSec: number, fps: number): number {
  return Math.floor(audioSec * fps + FRAME_EPSILON);
}
