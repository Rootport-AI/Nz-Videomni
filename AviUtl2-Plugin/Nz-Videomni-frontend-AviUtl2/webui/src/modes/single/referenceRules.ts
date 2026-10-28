/**
 * §1-107: the MiniMax H3 Reference accordion's state, its reducer, the pure
 * rules that decide which reference inputs a generation uses, and the
 * `/generate` `references` body built from them.
 *
 * Every limit (slot counts, the mixed-total cap, the per-kind total seconds,
 * the shortest clip) comes from the server's `constraints.reference`
 * ({@link ReferenceRules}); the slot arrays are built from those counts, so no
 * number such as 9/3/3 is written here. What IS fixed: the three kinds
 * (picture/video/audio) and the tag names ({@link referenceTag}).
 *
 * The used cards reach `/generate` as `references` ({@link buildReferenceBody});
 * the A2V slot reaches it through the existing `form.sourceAudio` path instead.
 */

import type { ReferenceInputBody, TransformerRole } from "../../api/types";

export type ReferenceKind = "picture" | "video" | "audio";

/** The three kinds in row order (Pictures, Videos, Audios). */
export const REFERENCE_KINDS: readonly ReferenceKind[] = ["picture", "video", "audio"];

/** `constraints.reference`, camelCased (`shell/generationConstraints.ts`). */
export interface ReferenceRules {
  maxImages: number;
  maxVideos: number;
  maxAudios: number;
  /** Mixed total of filled slots across the three kinds. */
  maxTotal: number;
  /** Total seconds of the videos used (per kind, not shared with audio). */
  videoTotalSec: number;
  audioTotalSec: number;
  /** A clip shorter than this is warned about (still accepted). */
  clipMinSec: number;
}

export type ReferenceSlotStatus = "empty" | "uploading" | "ready" | "error";

export interface ReferenceSlot {
  kind: ReferenceKind;
  /** 1-based, = the tag number (`<Picture 1>`). */
  slot: number;
  status: ReferenceSlotStatus;
  fileName: string | null;
  filePath: string | null;
  /** The upload's `image_id` / `video_id` / `audio_id`. */
  id: string | null;
  /** Seconds; `null` for a picture, or when the length could not be measured. */
  durationSec: number | null;
  thumbnailDataUrl: string | null;
  /** Input-order number, assigned when an attach STARTS (a re-attach gets a
   * new, larger one). Orders the mixed-total rule. `null` while empty. */
  addedAt: number | null;
  /** The video upload was asked to trim but the server did not (`trimmed`). */
  trimFailed: boolean;
  errorCode: string | null;
}

export interface ReferenceState {
  /** Default off. Off = every card greyed (content kept, nothing used). */
  enabled: boolean;
  /** The audio slot whose "Audio to video" box is checked, or `null`. */
  a2vSlot: number | null;
  /** One array per kind; lengths come from the rules. */
  slots: Record<ReferenceKind, ReferenceSlot[]>;
}

// --- derived values --------------------------------------------------------

/** Why a filled slot is not used, in the order the rules are applied. */
export type ReferenceIgnoreReason = "disabled" | "a2vExclusive" | "overTotal" | "overDuration";

export interface ReferenceSlotUse {
  /** The slot's content goes into the generation: filled, not ignored (the
   * effective A2V slot counts as used — through the A2V path). */
  used: boolean;
  /** Set only on a filled slot (`uploading | ready`) that is left out. */
  ignored: ReferenceIgnoreReason | null;
  /** Seconds actually used (after truncation). `null` for a picture, an
   * unknown length, an ignored slot or the A2V slot. */
  usedSec: number | null;
  /** Only the head (`usedSec`) of the clip fits the per-kind total. */
  truncated: boolean;
  /** `usedSec < clipMinSec` — a warning, never a reason to ignore. */
  tooShort: boolean;
  /** A ready video/audio whose length could not be measured: not counted in
   * the seconds, warned about. */
  durationUnknown: boolean;
  /** The effective A2V slot ({@link effectiveA2vSlot}): outside every count
   * and rule here (its shortness is `audioPrecheck`'s business). */
  isA2v: boolean;
}

export interface ReferenceSummary {
  /** Filled slots (`uploading | ready`) excluding the effective A2V slot —
   * ignored ones included. The `n` of `n/maxTotal` and each row's `n/max…`. */
  filled: { total: number; picture: number; video: number; audio: number };
  /** Raw total seconds of the filled videos / audios (ignored ones included,
   * the A2V slot and unknown lengths excluded). The `n` of `n/15s`. */
  seconds: { video: number; audio: number };
  /** Numerator > denominator, for the red bold counters. */
  over: {
    total: boolean;
    picture: boolean;
    video: boolean;
    audio: boolean;
    videoSec: boolean;
    audioSec: boolean;
  };
  /** Per slot, parallel to `state.slots[kind]`. */
  slotUse: Record<ReferenceKind, ReferenceSlotUse[]>;
}

// --- the hook's return type (implemented in `useReferenceSlots.ts`) ----------

export interface ReferenceAttachOptions {
  /** Extra upload query (the right-click #2 trim pair, `trimQuery()`). */
  query?: Record<string, string>;
  /** A known length (right-click #2: the length AFTER the trim), used instead
   * of probing the file. */
  durationSec?: number;
}

/** What `useReferenceSlots(rules)` returns — the Reference accordion's props
 * read only this. */
export interface ReferenceSlotsApi {
  state: ReferenceState;
  summary: ReferenceSummary;
  /** The rules the state is currently sized for. */
  rules: ReferenceRules;
  /** Uploads `filePath` into the slot (replacing what was there). */
  attach(
    kind: ReferenceKind,
    slot: number,
    filePath: string,
    fileName: string,
    opts?: ReferenceAttachOptions,
  ): Promise<void>;
  /** `ui.pickFile` then {@link attach}; a CANCELLED dialog does nothing. */
  pick(kind: ReferenceKind, slot: number): Promise<void>;
  /** Empties the slot (numbers are not packed); unchecks A2V if it was it. */
  clear(kind: ReferenceKind, slot: number): void;
  /** Attaches into {@link firstEmptySlot}; returns that slot, or `null` (and
   * attaches nothing) when every slot of the kind is filled. */
  attachToFirstEmpty(
    kind: ReferenceKind,
    filePath: string,
    fileName: string,
    opts?: ReferenceAttachOptions,
  ): number | null;
  setEnabled(enabled: boolean): void;
  /** Checks "Audio to video" on one audio slot (`null` unchecks). */
  setA2vSlot(slot: number | null): void;
}

// --- construction ------------------------------------------------------------

function slotCount(kind: ReferenceKind, rules: ReferenceRules): number {
  if (kind === "picture") return rules.maxImages;
  if (kind === "video") return rules.maxVideos;
  return rules.maxAudios;
}

export function emptySlot(kind: ReferenceKind, slot: number): ReferenceSlot {
  return {
    kind,
    slot,
    status: "empty",
    fileName: null,
    filePath: null,
    id: null,
    durationSec: null,
    thumbnailDataUrl: null,
    addedAt: null,
    trimFailed: false,
    errorCode: null,
  };
}

function emptyRow(kind: ReferenceKind, count: number): ReferenceSlot[] {
  return Array.from({ length: Math.max(0, count) }, (_, i) => emptySlot(kind, i + 1));
}

/** Every slot empty, Enable off, no A2V. */
export function createInitialState(rules: ReferenceRules): ReferenceState {
  return {
    enabled: false,
    a2vSlot: null,
    slots: {
      picture: emptyRow("picture", rules.maxImages),
      video: emptyRow("video", rules.maxVideos),
      audio: emptyRow("audio", rules.maxAudios),
    },
  };
}

// --- reducer -----------------------------------------------------------------

export type ReferenceAction =
  | { type: "attachStart"; kind: ReferenceKind; slot: number; filePath: string; fileName: string }
  | {
      type: "attachDone";
      kind: ReferenceKind;
      slot: number;
      id: string;
      durationSec: number | null;
      thumbnailDataUrl: string | null;
      trimFailed: boolean;
    }
  | { type: "attachFail"; kind: ReferenceKind; slot: number; errorCode: string }
  | { type: "clear"; kind: ReferenceKind; slot: number }
  | { type: "setEnabled"; enabled: boolean }
  | { type: "setA2vSlot"; slot: number | null }
  | { type: "resize"; rules: ReferenceRules };

function replaceSlot(
  state: ReferenceState,
  kind: ReferenceKind,
  slot: number,
  update: (current: ReferenceSlot) => ReferenceSlot,
): ReferenceState {
  const row = state.slots[kind];
  const current = row[slot - 1];
  if (!current) return state;
  const next = update(current);
  if (next === current) return state;
  return { ...state, slots: { ...state.slots, [kind]: row.map((s, i) => (i === slot - 1 ? next : s)) } };
}

function nextAddedAt(state: ReferenceState): number {
  let max = 0;
  for (const kind of REFERENCE_KINDS) {
    for (const s of state.slots[kind]) {
      if (s.addedAt !== null && s.addedAt > max) max = s.addedAt;
    }
  }
  return max + 1;
}

/** Pure. Stale async results are the hook's to drop (per-slot generation);
 * as a backstop, `attachDone`/`attachFail` only land on an `uploading` slot. */
export function referenceReducer(state: ReferenceState, action: ReferenceAction): ReferenceState {
  switch (action.type) {
    case "attachStart": {
      const addedAt = nextAddedAt(state);
      return replaceSlot(state, action.kind, action.slot, (s) => ({
        ...emptySlot(s.kind, s.slot),
        status: "uploading",
        fileName: action.fileName,
        filePath: action.filePath,
        addedAt,
      }));
    }
    case "attachDone":
      return replaceSlot(state, action.kind, action.slot, (s) =>
        s.status !== "uploading"
          ? s
          : {
              ...s,
              status: "ready",
              id: action.id,
              durationSec: action.kind === "picture" ? null : action.durationSec,
              thumbnailDataUrl: action.thumbnailDataUrl,
              trimFailed: action.trimFailed,
              errorCode: null,
            },
      );
    case "attachFail":
      return replaceSlot(state, action.kind, action.slot, (s) =>
        s.status !== "uploading" ? s : { ...s, status: "error", errorCode: action.errorCode },
      );
    case "clear": {
      const cleared = replaceSlot(state, action.kind, action.slot, (s) => emptySlot(s.kind, s.slot));
      const dropA2v = action.kind === "audio" && cleared.a2vSlot === action.slot;
      return dropA2v ? { ...cleared, a2vSlot: null } : cleared;
    }
    case "setEnabled":
      return state.enabled === action.enabled ? state : { ...state, enabled: action.enabled };
    case "setA2vSlot":
      return state.a2vSlot === action.slot ? state : { ...state, a2vSlot: action.slot };
    case "resize": {
      const slots = {} as Record<ReferenceKind, ReferenceSlot[]>;
      for (const kind of REFERENCE_KINDS) {
        const count = Math.max(0, slotCount(kind, action.rules));
        const row = state.slots[kind];
        slots[kind] = Array.from({ length: count }, (_, i) => row[i] ?? emptySlot(kind, i + 1));
      }
      const a2vSlot = state.a2vSlot !== null && state.a2vSlot <= slots.audio.length ? state.a2vSlot : null;
      return { ...state, a2vSlot, slots };
    }
  }
}

// --- small helpers -------------------------------------------------------------

/** `uploading | ready` (an `error` slot is not filled). */
export function isFilled(slot: ReferenceSlot): boolean {
  return slot.status === "uploading" || slot.status === "ready";
}

/** `empty | error` — a slot an attach may go into without overwriting. */
export function isFree(slot: ReferenceSlot): boolean {
  return slot.status === "empty" || slot.status === "error";
}

/** The lowest-numbered free slot (1-based), or `null` when all are filled. */
export function firstEmptySlot(slots: readonly ReferenceSlot[]): number | null {
  const found = slots.find(isFree);
  return found ? found.slot : null;
}

/** The audio slot that rides the A2V path: Enable on, its box checked, and
 * its upload ready. Otherwise `null`. */
export function effectiveA2vSlot(state: ReferenceState): number | null {
  if (!state.enabled || state.a2vSlot === null) return null;
  return state.slots.audio[state.a2vSlot - 1]?.status === "ready" ? state.a2vSlot : null;
}

const TAG_NAMES: Record<ReferenceKind, string> = { picture: "Picture", video: "Video", audio: "Audio" };

/** `<Picture 1>` / `<Video 2>` / `<Audio 3>`. */
export function referenceTag(kind: ReferenceKind, slot: number): string {
  return `<${TAG_NAMES[kind]} ${slot}>`;
}

/** Appends `tag` after the prompt's trailing whitespace is trimmed, with one
 * space between (the tag alone for an empty prompt). Duplicates are allowed. */
export function appendReferenceTag(prompt: string, tag: string): string {
  const base = prompt.trimEnd();
  return base.length === 0 ? tag : `${base} ${tag}`;
}

// --- the summary -----------------------------------------------------------------

/** Float slack for the seconds budget (10.2 + 4.8 must fill 15 exactly). */
const SECONDS_EPSILON = 1e-6;

function blankUse(): ReferenceSlotUse {
  return {
    used: false,
    ignored: null,
    usedSec: null,
    truncated: false,
    tooShort: false,
    durationUnknown: false,
    isA2v: false,
  };
}

/**
 * Applies the rules in order; a slot left out at one step is not counted by
 * the later ones:
 *  0. the effective A2V slot is outside every count and rule;
 *  1. `disabled` — Enable off leaves out every filled slot;
 *  2. `a2vExclusive` — an effective A2V slot leaves out every other one;
 *  3. `overTotal` — by input order (`addedAt`), slots past `maxTotal`;
 *  4. `overDuration` — videos and audios separately, in slot-number order,
 *     cumulating seconds: nothing left → left out, less left than the clip →
 *     `truncated` to what is left (an unknown length is skipped);
 *  5. `tooShort` — warning only, on the seconds actually used (on the raw
 *     length for a slot left out at 1–4).
 * `durationUnknown` and `tooShort` are set on every counted slot, left out or
 * not (never on the A2V slot); `truncated` only on slots that survive 1–4.
 */
export function summarizeReference(state: ReferenceState, rules: ReferenceRules): ReferenceSummary {
  const a2v = effectiveA2vSlot(state);
  const isA2vSlot = (s: ReferenceSlot) => s.kind === "audio" && s.slot === a2v;

  const uses = new Map<ReferenceSlot, ReferenceSlotUse>();
  const slotUse = {} as Record<ReferenceKind, ReferenceSlotUse[]>;
  for (const kind of REFERENCE_KINDS) {
    slotUse[kind] = state.slots[kind].map((s) => {
      const use = blankUse();
      uses.set(s, use);
      return use;
    });
  }
  const slotUseOf = (s: ReferenceSlot): ReferenceSlotUse => uses.get(s) ?? blankUse();

  // 0. Counted slots: filled, not the A2V slot.
  const counted: ReferenceSlot[] = [];
  for (const kind of REFERENCE_KINDS) {
    for (const s of state.slots[kind]) {
      if (!isFilled(s)) continue;
      if (isA2vSlot(s)) {
        Object.assign(slotUseOf(s), { used: true, isA2v: true });
        continue;
      }
      counted.push(s);
    }
  }

  const filled = { total: counted.length, picture: 0, video: 0, audio: 0 };
  const seconds = { video: 0, audio: 0 };
  for (const s of counted) {
    filled[s.kind] += 1;
    if (s.kind !== "picture" && s.durationSec !== null) seconds[s.kind] += s.durationSec;
  }

  // 1–2. Everything out when disabled, or when an A2V slot is in effect.
  const wholesale: ReferenceIgnoreReason | null = !state.enabled ? "disabled" : a2v !== null ? "a2vExclusive" : null;
  if (wholesale !== null) {
    for (const s of counted) slotUseOf(s).ignored = wholesale;
  } else {
    // 3. The mixed total, by input order.
    const byInput = [...counted].sort((a, b) => (a.addedAt ?? 0) - (b.addedAt ?? 0));
    byInput.forEach((s, i) => {
      if (i >= rules.maxTotal) slotUseOf(s).ignored = "overTotal";
    });

    // 4. The per-kind seconds, by slot number.
    for (const kind of ["video", "audio"] as const) {
      let remaining = kind === "video" ? rules.videoTotalSec : rules.audioTotalSec;
      for (const s of state.slots[kind]) {
        const use = slotUseOf(s);
        if (!counted.includes(s) || use.ignored !== null || s.durationSec === null) continue;
        if (remaining <= SECONDS_EPSILON) {
          use.ignored = "overDuration";
        } else if (s.durationSec > remaining + SECONDS_EPSILON) {
          use.truncated = true;
          use.usedSec = remaining;
          remaining = 0;
        } else {
          use.usedSec = s.durationSec;
          remaining -= s.durationSec;
        }
      }
    }

  }

  // 5. Used, and the clip's own warnings. The warnings are facts about the
  // clip, so a left-out card shows them too (on its raw length; a used card
  // on the seconds actually used). The A2V slot is not in `counted`.
  for (const s of counted) {
    const use = slotUseOf(s);
    if (use.ignored === null) use.used = true;
    if (s.kind === "picture") continue;
    if (s.status === "ready" && s.durationSec === null) use.durationUnknown = true;
    const sec = use.ignored === null ? use.usedSec : s.durationSec;
    if (sec !== null && sec < rules.clipMinSec) use.tooShort = true;
  }

  return {
    filled,
    seconds,
    over: {
      total: filled.total > rules.maxTotal,
      picture: filled.picture > rules.maxImages,
      video: filled.video > rules.maxVideos,
      audio: filled.audio > rules.maxAudios,
      videoSec: seconds.video > rules.videoTotalSec,
      audioSec: seconds.audio > rules.audioTotalSec,
    },
    slotUse,
  };
}

// --- the /generate body and the notices --------------------------------------------

/**
 * The cards a generation would send, in body order (pictures, videos, audios;
 * card-number order within a kind), whatever their upload status. Empty off
 * H3 (`isH3` false — the state survives a trip through LTX, Enable included,
 * so this is what keeps LTX from receiving `references` and answering 422)
 * and while an FL2VA transformer is loaded (its notice says the references
 * are ignored, so they are not sent). A Hybrid or an unrecognised file
 * (`role === null`) sends them. Enable off and an effective A2V slot are
 * already folded into `used` / `isA2v` by {@link summarizeReference}.
 */
function sentSlots(
  summary: ReferenceSummary,
  slots: Record<ReferenceKind, ReferenceSlot[]>,
  role: TransformerRole | null,
  isH3: boolean,
): { slot: ReferenceSlot; use: ReferenceSlotUse }[] {
  if (!isH3 || role === "fl2va") return [];
  const out: { slot: ReferenceSlot; use: ReferenceSlotUse }[] = [];
  for (const kind of REFERENCE_KINDS) {
    slots[kind].forEach((s, i) => {
      const use = summary.slotUse[kind][i];
      if (use?.used && !use.isA2v) out.push({ slot: s, use });
    });
  }
  return out;
}

/**
 * `GenerateRequest.references`: the ready cards of {@link sentSlots}, with
 * `max_sec` only on a card truncated to fit its kind's total seconds.
 */
export function buildReferenceBody(
  summary: ReferenceSummary,
  slots: Record<ReferenceKind, ReferenceSlot[]>,
  role: TransformerRole | null,
  isH3: boolean,
): ReferenceInputBody[] {
  const body: ReferenceInputBody[] = [];
  for (const { slot: s, use } of sentSlots(summary, slots, role, isH3)) {
    if (s.status !== "ready" || s.id === null) continue;
    const entry: ReferenceInputBody = { kind: s.kind, id: s.id, slot: s.slot };
    if (use.truncated && use.usedSec !== null) entry.max_sec = use.usedSec;
    body.push(entry);
  }
  return body;
}

/** A card {@link buildReferenceBody} would send is still uploading — the
 * generation waits for it (like a keyframe upload). Cards that would not be
 * sent (LTX, FL2VA, Enable off, left out by a rule) never block. */
export function referenceSendUploading(
  summary: ReferenceSummary,
  slots: Record<ReferenceKind, ReferenceSlot[]>,
  role: TransformerRole | null,
  isH3: boolean,
): boolean {
  return sentSlots(summary, slots, role, isH3).some(({ slot }) => slot.status === "uploading");
}

/** The non-blocking notices under Generate (`strings.single.reference.notices`). */
export type ReferenceNoticeCode = "fl2vaLoaded" | "ref2vaLoaded" | "noReferences";

export interface ReferenceNoticeInput {
  role: TransformerRole | null;
  /** The accordion's Enable. */
  enabled: boolean;
  /** An effective "Audio to video" slot ({@link effectiveA2vSlot}). */
  a2vActive: boolean;
  /** Used cards, the A2V slot excluded. */
  usedCount: number;
}

/**
 * Each notice is decided on its own (no rule suppresses another):
 *  - `fl2vaLoaded`: FL2VA loaded, Enable on, no A2V (A2V works on FL2VA, so
 *    the "references are ignored" sentence would not hold);
 *  - `ref2vaLoaded`: Ref2VA loaded, Enable off;
 *  - `noReferences`: Enable on, no A2V, no card used.
 */
export function referenceNotices(input: ReferenceNoticeInput): ReferenceNoticeCode[] {
  const codes: ReferenceNoticeCode[] = [];
  if (input.role === "fl2va" && input.enabled && !input.a2vActive) codes.push("fl2vaLoaded");
  if (input.role === "ref2va" && !input.enabled) codes.push("ref2vaLoaded");
  if (input.enabled && !input.a2vActive && input.usedCount === 0) codes.push("noReferences");
  return codes;
}

/** Used cards excluding the A2V slot — {@link ReferenceNoticeInput.usedCount}. */
export function countUsedReferences(summary: ReferenceSummary): number {
  let n = 0;
  for (const kind of REFERENCE_KINDS) {
    for (const use of summary.slotUse[kind]) if (use.used && !use.isA2v) n += 1;
  }
  return n;
}
