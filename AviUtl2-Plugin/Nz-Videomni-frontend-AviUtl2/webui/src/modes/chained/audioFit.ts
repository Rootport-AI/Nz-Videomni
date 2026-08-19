/**
 * §1-16 長尺A2V — the "↔️ 再生時間の自動調整" resolver.
 *
 * ONE pure function (`planAudioFit`) that answers: given the clip cards the
 * Chained screen currently holds and an attached audio track, what clip list
 * covers that audio? Pure and framework-free (no React, no `i18n/strings`) for
 * the same reason `chainUtils.ts` is — `useChainForm` turns an outcome into a
 * toast, this file only decides.
 *
 * The rules the owner fixed for it (2026-08-10):
 *
 *  * A card the user has TOUCHED is untouchable. `intact: false` (see
 *    `ChainClipInput.intact`) pins a card's LENGTH, its POSITION and its very
 *    EXISTENCE. The resolver only ever resizes / adds / drops intact cards, and
 *    resizing one does not make it non-intact.
 *  * Every intact card is set to the slider's "追加クリップの長さ" value `S`,
 *    except the LAST intact card in array order, which absorbs the remainder so
 *    the timeline lands on the audio instead of overshooting it.
 *  * The clip count stays within `[minClips, maxClips]` (2..24). Running into
 *    the 24 ceiling with audio left over is reported, not an error.
 *  * THE INVARIANT: any clip list this returns as a CHANGE is guaranteed to
 *    satisfy `chainLayoutError(...) === null`. The auto-fit button must never
 *    produce a construction the server rejects with a 422 — see
 *    `chainLayoutError`'s doc comment for why that is a real risk at 29.97fps
 *    and friends, not a theoretical one. (`cannotFit`/`noFlexibleClips` return
 *    the caller's own clips untouched and therefore carry no such guarantee:
 *    nothing was proposed.)
 */
import { clamp } from "../single/paramUtils";
import {
  audioLatentsAvailable,
  audioLatentsRequired,
  chainLayoutError,
  MAX_CLIPS,
  MAX_CLIP_NUM_FRAMES,
  minClipFramesForKv,
  MIN_CLIPS_NO_SOURCE,
  resolveA2vFps,
  snapClipNumFrames,
  AUDIO_LATENTS_PER_SEC,
} from "./chainUtils";

/** The subset of a clip card `planAudioFit` reads. */
export interface AudioFitClip {
  id: string;
  numFrames: number;
  /** `false` = the user touched this card; it is frozen. Callers holding a
   * `ChainClipInput` should pass `isClipIntact(clip)`. */
  intact: boolean;
}

export interface AudioFitParams {
  clips: readonly AudioFitClip[];
  /** Measured length of the attached track, in seconds. */
  audioDurationSec: number;
  /** The form's `frame_rate`. `0`/`NaN` falls back to 24, the same way
   * `chainUtils.resolveA2vFps` (and the backend's `_resolve_fps`) does. */
  fps: number;
  /** The form's ACTUAL `overlap_frames` (K_v) — never hard-code 3 here. The
   * seam overlap changes both the audio requirement and the per-clip floor. */
  overlapFrames: number;
  /** The "追加クリップの長さ" slider's value `S` (8n+1, 9..481). Clamped and
   * snapped internally, so a raw slider value is fine. */
  targetClipFrames: number;
  /** Stage-2 window `(vTile, vAdv)` — one of `shell/tokenBudget.ts`'s
   * `STAGE2_WINDOW_PRESETS` entries. Only used for the layout check. */
  stage2Window: { vTile: number; vAdv: number };
  /** Floor on the resulting clip count. Default {@link MIN_CLIPS_NO_SOURCE}
   * (2) — a source-less chain needs two clips to concatenate anything. */
  minClips?: number;
  /** Ceiling on the resulting clip count. Default {@link MAX_CLIPS} (24). */
  maxClips?: number;
}

/**
 * - `adjusted` — a new clip list is proposed (and `changed` is `true`).
 * - `alreadyFits` — the current clip list is already the answer.
 * - `cappedAtMaxClips` — the audio is longer than `maxClips` cards of `S`
 *   frames can cover; the returned list is that maximum and the leftover is in
 *   `unusedAudioSec`. Still a valid, submittable chain.
 * - `cannotFit` — no arrangement works (audio too short even for the smallest
 *   legal chain, or every candidate fails the layout check). `clips` is the
 *   caller's own list, untouched.
 * - `noFlexibleClips` — every card is `intact: false`, so there is nothing the
 *   resolver is allowed to move. `clips` is untouched.
 */
export type AudioFitOutcome =
  | "adjusted"
  | "alreadyFits"
  | "cappedAtMaxClips"
  | "cannotFit"
  | "noFlexibleClips";

export interface AudioFitPlan {
  outcome: AudioFitOutcome;
  /** The proposed clip list, in order. `sourceId` is the existing card's `id`,
   * or `null` for a card that has to be created. */
  clips: ReadonlyArray<{ sourceId: string | null; numFrames: number }>;
  /** `false` when {@link clips} is card-for-card identical to the input. */
  changed: boolean;
  /** `round(audioDurationSec * 25)` — the server's own `available`, WITHOUT
   * the safety margin (see {@link AUDIO_FIT_SAFETY_MARGIN_LATENTS}). */
  audioLatentsAvailable: number;
  /** Audio-latent frames {@link clips} consumes, or `null` when that cannot be
   * computed (a geometrically impossible input on the untouched-return paths). */
  audioLatentsUsed: number | null;
  /** Tail of the track no clip covers, in seconds. `0` when unknown. */
  unusedAudioSec: number;
}

/**
 * Audio-latent frames (1 = 0.04s) held back from what the resolver is allowed
 * to consume.
 *
 * The server measures the upload with ffprobe and the front end measures it
 * with `fs.probeAudioDuration`/`fs.probeMediaInfo`/`timeline.extractAudio`;
 * those two numbers agree to within a container-header rounding, not exactly.
 * A plan that consumes the LAST latent frame therefore lands on a 422
 * `SOURCE_AUDIO_TOO_SHORT` whenever the server's measurement comes back one
 * frame shorter than ours. One frame of slack costs 0.04s of unused audio and
 * removes that entire failure mode.
 */
export const AUDIO_FIT_SAFETY_MARGIN_LATENTS = 1;

/** One entry of the assembled candidate, plus whether it came from an intact
 * card (which is what may be resized / is eligible to absorb the remainder). */
interface Assembled {
  sourceId: string | null;
  numFrames: number;
  fromIntact: boolean;
}

/** The `n` of an 8n+1 frame count — the index the tail search binary-searches
 * over so every candidate it tries is already on the grid. */
function gridIndex(frames: number): number {
  return (frames - 1) / 8;
}

/**
 * Resolves the clip list that covers `audioDurationSec` seconds of audio.
 *
 * Never throws and never mutates its input; on any "nothing to propose" outcome
 * it echoes the caller's clips back unchanged. See {@link AudioFitOutcome} for
 * what each result means and the module header for the rules being applied.
 */
export function planAudioFit(params: AudioFitParams): AudioFitPlan {
  const {
    clips,
    audioDurationSec,
    stage2Window,
    minClips = MIN_CLIPS_NO_SOURCE,
    maxClips = MAX_CLIPS,
  } = params;
  const fps = resolveA2vFps(params.fps);
  const kv = params.overlapFrames;

  const realAvail = audioDurationSec > 0 ? audioLatentsAvailable(audioDurationSec) : 0;
  const currentFrames = clips.map((clip) => clip.numFrames);
  const unchanged = (outcome: AudioFitOutcome): AudioFitPlan => {
    const used = clips.length > 0 ? audioLatentsRequired(currentFrames, fps, kv) : null;
    return {
      outcome,
      clips: clips.map((clip) => ({ sourceId: clip.id, numFrames: clip.numFrames })),
      changed: false,
      audioLatentsAvailable: realAvail,
      audioLatentsUsed: used,
      unusedAudioSec: used == null ? 0 : Math.max(0, (realAvail - used) / AUDIO_LATENTS_PER_SEC),
    };
  };

  // Nothing the resolver is allowed to move.
  const flexCount = clips.reduce((count, clip) => count + (clip.intact ? 1 : 0), 0);
  if (flexCount === 0) return unchanged("noFlexibleClips");
  if (!(audioDurationSec > 0)) return unchanged("cannotFit");

  const floor = minClipFramesForKv(kv);
  const target = snapClipNumFrames(clamp(params.targetClipFrames, floor, MAX_CLIP_NUM_FRAMES));
  const bigFrames = Math.max(floor, target);

  const pinnedCount = clips.length - flexCount;
  const mMin = Math.max(1, minClips - pinnedCount);
  const mMax = maxClips - pinnedCount;
  if (mMax < mMin) return unchanged("cannotFit");

  const avail = realAvail - AUDIO_FIT_SAFETY_MARGIN_LATENTS;

  /** Builds the candidate list for `m` intact cards whose last one is `tail`
   * frames long. Pinned cards keep their length AND their position; intact
   * cards are taken in array order and the surplus is dropped from the end;
   * a shortfall is appended as brand-new cards (`sourceId: null`). */
  const assemble = (m: number, tail: number): Assembled[] => {
    const out: Assembled[] = [];
    let kept = 0;
    for (const clip of clips) {
      if (!clip.intact) {
        out.push({ sourceId: clip.id, numFrames: clip.numFrames, fromIntact: false });
        continue;
      }
      if (kept < m) {
        out.push({ sourceId: clip.id, numFrames: bigFrames, fromIntact: true });
        kept += 1;
      }
      // else: this intact card is surplus and is dropped.
    }
    for (; kept < m; kept += 1) {
      out.push({ sourceId: null, numFrames: bigFrames, fromIntact: true });
    }
    // The LAST intact card in ARRAY order absorbs the remainder — deliberately
    // "last intact", not "last card", so a pinned card sitting at the end of
    // the chain does not steal the job it is not allowed to do.
    for (let i = out.length - 1; i >= 0; i -= 1) {
      const entry = out[i];
      if (entry !== undefined && entry.fromIntact) {
        entry.numFrames = tail;
        break;
      }
    }
    return out;
  };

  const requirementFor = (candidate: Assembled[]): number | null =>
    audioLatentsRequired(candidate.map((entry) => entry.numFrames), fps, kv);

  const fits = (m: number, tail: number): boolean => {
    const required = requirementFor(assemble(m, tail));
    return required !== null && required <= avail;
  };

  // ── how many intact cards? ────────────────────────────────────────────────
  // Adding one more card of the minimum length adds exactly one video-latent
  // frame to the timeline (`v_latent(8*kv+1) - kv == 1`), so the requirement is
  // strictly increasing in `m` — a plain climb finds the largest workable count.
  if (!fits(mMin, floor)) return unchanged("cannotFit");
  let mGrown = mMin;
  while (mGrown < mMax && fits(mGrown + 1, floor)) mGrown += 1;

  const kFloor = gridIndex(floor);
  const kTarget = gridIndex(bigFrames);

  /** Largest 8n+1 tail in `[floor, bigFrames]` whose chain still fits `avail`,
   * or `null` when even `floor` overflows. Binary search — the requirement is
   * monotonically non-decreasing in the tail length. */
  const bestTail = (m: number): number | null => {
    if (!fits(m, floor)) return null;
    let lo = kFloor;
    let hi = kTarget;
    while (lo < hi) {
      const mid = Math.ceil((lo + hi) / 2);
      if (fits(m, mid * 8 + 1)) lo = mid;
      else hi = mid - 1;
    }
    return lo * 8 + 1;
  };

  // ── settle on a construction the SERVER will also accept ──────────────────
  // `chainLayoutError` rejects whole families of otherwise-sane clip lists at
  // non-integer frame rates, so the best-fitting candidate is only a starting
  // point: walk the tail down the 8n+1 grid (then back up, for the case where
  // the tail was capped by the slider rather than by the audio), and if the
  // whole tail range fails, give up one card and try again.
  let chosen: { m: number; tail: number; clips: Assembled[]; used: number } | null = null;
  for (let m = mGrown; m >= mMin && chosen === null; m -= 1) {
    const start = bestTail(m);
    if (start === null) continue;
    const kStart = gridIndex(start);
    const order: number[] = [];
    for (let k = kStart; k >= kFloor; k -= 1) order.push(k);
    for (let k = kStart + 1; k <= kTarget; k += 1) order.push(k);
    for (const k of order) {
      const tail = k * 8 + 1;
      const candidate = assemble(m, tail);
      const frames = candidate.map((entry) => entry.numFrames);
      const required = audioLatentsRequired(frames, fps, kv);
      if (required === null || required > avail) continue;
      if (chainLayoutError(frames, fps, kv, stage2Window) !== null) continue;
      chosen = { m, tail, clips: candidate, used: required };
      break;
    }
  }
  if (chosen === null) return unchanged("cannotFit");

  const resultClips = chosen.clips.map((entry) => ({ sourceId: entry.sourceId, numFrames: entry.numFrames }));
  const changed =
    resultClips.length !== clips.length ||
    resultClips.some((entry, i) => {
      const before = clips[i];
      return before === undefined || entry.sourceId !== before.id || entry.numFrames !== before.numFrames;
    });

  // Capped: the maximum card count, every card at the full slider length, and
  // audio STILL left over (strictly — `used === avail` is an exact fit, and the
  // one-frame safety margin alone must not read as "leftover").
  const capped = chosen.m === mMax && chosen.tail === bigFrames && chosen.used < avail;

  let outcome: AudioFitOutcome;
  if (capped) outcome = "cappedAtMaxClips";
  else if (!changed) outcome = "alreadyFits";
  else outcome = "adjusted";

  return {
    outcome,
    clips: resultClips,
    changed,
    audioLatentsAvailable: realAvail,
    audioLatentsUsed: chosen.used,
    unusedAudioSec: Math.max(0, (realAvail - chosen.used) / AUDIO_LATENTS_PER_SEC),
  };
}

/**
 * Owner spec #9 — "is this audio so short that a CHAIN is the wrong screen?".
 *
 * A chain needs at least two clips, and the first one is whatever the user
 * already has. So the shortest chain that can exist right now is
 * `[firstClipFrames, minClipFramesForKv(kv)]`; if the track cannot even cover
 * that, no amount of auto-fitting helps and the UI should point at the Single
 * screen instead of at the ↔️ button.
 *
 * Returns `false` (defer to the server) when the requirement cannot be computed
 * at all, matching `audioLatentsRequired`'s `null` contract everywhere else.
 */
export function isAudioTooShortForChain(
  firstClipFrames: number,
  audioDurationSec: number,
  fps: number,
  overlapFrames: number,
): boolean {
  const fpsV = resolveA2vFps(fps);
  const required = audioLatentsRequired(
    [firstClipFrames, minClipFramesForKv(overlapFrames)],
    fpsV,
    overlapFrames,
  );
  if (required === null) return false;
  return audioLatentsAvailable(audioDurationSec) < required;
}
