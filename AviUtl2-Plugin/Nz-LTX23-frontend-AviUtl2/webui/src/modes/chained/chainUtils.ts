/**
 * Pure, framework/i18n-agnostic helpers backing the Chain screen
 * (Docs/API_REFERENCE.md §3.13/§5.2 `POST /generate/chain`). The screen has a
 * single, auto-detected mode: attaching a `source_video` makes it a V2V
 * continuation, otherwise it's a from-scratch clip concatenation. Kept
 * dependency-free (no React, no `i18n/strings`) so every rule here is trivially
 * unit-testable — `modes/chained/useChainForm.ts` is what turns a `false` result
 * into a rendered warning banner with actual copy.
 *
 * The audio-latent geometry helpers further down (`suggestFramesForAudio`/
 * `audioLengthPrecheck` et al.) stay here as the shared, pure single source of
 * truth that Create (`modes/single/useGenerationForm.ts`) and Batch A2V import.
 *
 * `num_frames`/`context_frames`'s "8n+1" grid constraint is identical to
 * Create's single `num_frames` field, so this reuses `../single/paramUtils`
 * rather than re-deriving it.
 */
import type { ChainClip, ConditioningImage, CropOutput, GenerateChainRequest, LoraSpec } from "../../api/types";
import { nagRequestFields } from "../../shell/nagSettings";
import type { NagSettings } from "../../shell/nagSettings";
import { accelerationRequestFields } from "../../shell/accelerationSettings";
import type { AccelerationSettings } from "../../shell/accelerationSettings";
import { STAGE2_WINDOW_DEFAULT } from "../../shell/tokenBudget";
import type { Stage2Window } from "../../shell/tokenBudget";
import { clamp, snapNumFrames } from "../single/paramUtils";

/** Total-duration ceiling across every clip in a chain request
 * (Docs/API_REFERENCE.md §5.2: `MAX_CHAIN_TOTAL_PIXEL_FRAMES = 24*481 = 11544`). */
export const MAX_CHAIN_TOTAL_FRAMES = 11544;

/** `clips` array bounds (Docs/API_REFERENCE.md §5.2: "1-24本"). */
export const MAX_CLIPS = 24;
/** From-scratch (no source video) minimum ("source無しなら最低2本") — at
 * least two clips are needed to have anything to concatenate. */
export const MIN_CLIPS_NO_SOURCE = 2;
/** V2V may chain with a single clip, since the source itself supplies the
 * "previous" content. */
export const MIN_CLIPS_WITH_SOURCE = 1;

export const MIN_OVERLAP_FRAMES = 1;
export const MAX_OVERLAP_FRAMES = 8;
export const DEFAULT_OVERLAP_FRAMES = 3;

export const MIN_OVERLAP_STRENGTH = 0.0;
export const MAX_OVERLAP_STRENGTH = 1.0;
export const DEFAULT_OVERLAP_STRENGTH = 0.5;

/** 素材（末尾）の錨の固定強度 (`GenerateChainRequest.end_source.strength`,
 * バッチ1・2026-08-18). 1.0＝ハード凍結（既定・従来と完全同値）、下げると
 * Stage-1のマスク値が緩む。`overlapStrength`（生成物同士の継ぎ目）とは別物 —
 * こちらは素材そのものへの凍結強度で、Stage-2は常にハード凍結するため
 * `strength`の値にかかわらず最終フレームは常に素材どおりになる。 */
export const MIN_END_SOURCE_STRENGTH = 0.0;
export const MAX_END_SOURCE_STRENGTH = 1.0;
export const DEFAULT_END_SOURCE_STRENGTH = 1.0;

export const MIN_CLIP_NUM_FRAMES = 9;
export const MAX_CLIP_NUM_FRAMES = 481;

/** §1-16 (long-form A2V): the "追加クリップの長さ" slider's opening value — the
 * per-clip length the "↔️ 再生時間の自動調整" resolver gives every card it is
 * allowed to resize (`audioFit.ts`'s `targetClipFrames`).
 *
 * 257 is the owner-fixed default AND the recommendation the default preset
 * already produces via {@link recommendedClipFrames}, so a form that never
 * touches a preset opens on exactly the value the preset would have re-seeded it
 * with (judgement point 7 of the §1-16 plan: applying a preset re-seeds this
 * slider to THAT preset's recommendation, since a fixed 257 would otherwise make
 * ↔️ silently override a 768p preset's comfort ceiling). */
export const ADDED_CLIP_FRAMES_DEFAULT = 257;

/** `conditioning_attention_strength`/`reference_video_strength` share the
 * same `[0, 1]` range (`api/models.py` `ge=0.0, le=1.0`) and the same
 * "unset until the user opts in" UX in `useChainForm` — this is the value a
 * slider is seeded with the moment its enable checkbox is first ticked. */
export const MIN_REFERENCE_STRENGTH = 0.0;
export const MAX_REFERENCE_STRENGTH = 1.0;
/** IC-LoRA UI redesign (2026-07-17, 第5波), owner decision #3: seeded at 1.0
 * (was 0.5) to match Gradio/the official reference-conditioning guidance —
 * the prior 0.5 default under-drove the effect for most users. Still just the
 * SEED for the enable checkbox's first tick; the slider covers the full
 * `[MIN_REFERENCE_STRENGTH, MAX_REFERENCE_STRENGTH]` range regardless. */
export const DEFAULT_REFERENCE_STRENGTH = 1.0;

/** N1 (`crop_output`, Docs/API_REFERENCE.md §5.1/§5.2): the backend accepts
 * ANY integer for each crop dimension — Gradio's own crop_output field has no
 * fixed-grid constraint, unlike width/height's own 64-pixel grid.
 * `CROP_OUTPUT_MIN` is the floor for either dimension; the ceiling is always
 * the CURRENT generation width/height (never `config.limits.max_width/
 * max_height`) — a crop can only ever be smaller than or equal to what's
 * actually being generated. */
export const CROP_OUTPUT_MIN = 32;

/** Clamps a candidate `crop_output` to an integer within
 * `[CROP_OUTPUT_MIN, maxWidth/maxHeight]` (the current generation
 * width/height, passed in by the caller — see {@link CROP_OUTPUT_MIN}'s doc
 * comment). No grid-snapping — any integer in range is valid
 * (Gradio-faithful). Shared by Create's `useGenerationForm`/Chain's
 * `useChainForm` (state setters) and `CommonGenerationFields.tsx`'s
 * `CropOutputField` (live input clamping while the user types). */
export function clampCropOutput(raw: CropOutput, maxWidth: number, maxHeight: number): CropOutput {
  return {
    width: clamp(Math.round(raw.width), CROP_OUTPUT_MIN, Math.max(CROP_OUTPUT_MIN, maxWidth)),
    height: clamp(Math.round(raw.height), CROP_OUTPUT_MIN, Math.max(CROP_OUTPUT_MIN, maxHeight)),
  };
}

/** `true` when `crop` is `null` (nothing to validate — the field is simply
 * omitted) or is a valid crop: an integer at least `CROP_OUTPUT_MIN` and no
 * larger than the current generation `width`/`height` (no grid constraint —
 * see {@link CROP_OUTPUT_MIN}'s doc comment). Folded into both
 * `useGenerationForm.isValid` and `useChainForm.isValid` — mirrors the
 * width/height "manual entry passes through unclamped, `isValid` gates at
 * submit time" pattern rather than continuously re-clamping a stored crop
 * whenever width/height changes afterward. */
export function isCropOutputValid(crop: CropOutput | null, width: number, height: number): boolean {
  if (crop === null) return true;
  return (
    Number.isInteger(crop.width) &&
    crop.width >= CROP_OUTPUT_MIN &&
    crop.width <= width &&
    Number.isInteger(crop.height) &&
    crop.height >= CROP_OUTPUT_MIN &&
    crop.height <= height
  );
}

/** The minimum number of clips required, given whether a V2V source video is
 * attached. With a source video the chain may be a single clip (the source
 * itself supplies the "previous" content); a from-scratch chain needs at
 * least two clips to have anything to concatenate. The ceiling is always
 * {@link MAX_CLIPS} regardless — callers reference that constant directly. */
export function minClipsForChain(hasSourceVideo: boolean): number {
  return hasSourceVideo ? MIN_CLIPS_WITH_SOURCE : MIN_CLIPS_NO_SOURCE;
}

export function isClipCountValid(hasSourceVideo: boolean, count: number): boolean {
  return count >= minClipsForChain(hasSourceVideo) && count <= MAX_CLIPS;
}

/** Sum of every clip's `num_frames` — the value compared against
 * `MAX_CHAIN_TOTAL_FRAMES`. */
export function computeTotalFrames(clips: ReadonlyArray<{ numFrames: number }>): number {
  return clips.reduce((sum, clip) => sum + clip.numFrames, 0);
}

export function isTotalFramesValid(totalFrames: number): boolean {
  return totalFrames > 0 && totalFrames <= MAX_CHAIN_TOTAL_FRAMES;
}

/** Video latent-frame count for a pixel frame count (8n+1). Copy of
 * `chain_math.v_latent_frames` (single source of truth:
 * `Nz-LTX23-backend/chain_math.py`): `(P-1)//8 + 1`, integer division. */
export function vLatentFrames(pxFrames: number): number {
  return Math.floor((pxFrames - 1) / 8) + 1;
}

/** Inverse of {@link vLatentFrames} — pixel frames (8n+1) for a video
 * latent-frame count. Copy of `chain_math.px_from_v_latent`. */
export function pxFromVLatent(latentFrames: number): number {
  return (latentFrames - 1) * 8 + 1;
}

/** Video-latent frames a TAIL band of `tailPx` pixel frames occupies — copy of
 * `chain_math.v_tail_latents` (`Nz-LTX23-backend/chain_math.py:349-365`;
 * `n_end_v` in that module's own naming). Deliberately NOT
 * {@link vLatentFrames}: the causal VAE's head/tail grids differ (a tail band
 * counts back from the end in whole groups of 8 and never touches the lone
 * keyframe latent a HEAD band's `+1` belongs to), so this is plain integer
 * division rather than `vLatentFrames`'s `+1`. Used by
 * {@link endSourceLastClipHasFreeLatents} (素材（末尾）reverse mode, 2.6(f)). */
export function vTailLatents(tailPx: number): number {
  return Math.floor(tailPx / 8);
}

/** Predicts the delivered `output.mp4`'s pixel-frame count for a chain —
 * a copy of `chain_math.compute_chain_layout`'s `f_total`/`total_px` (and,
 * for V2V, the context-trim) arithmetic (single source of truth:
 * `Nz-LTX23-backend/chain_math.py`; see also `Docs/PHASE3_CLIP_CONCAT_STATUS.md`
 * "★フロントエンド開発向けメモ"). Clip `num_frames` don't sum directly to the
 * output length: adjacent clips fuse `overlapFrames` (video-latent, K_v)
 * worth of overlap at every seam, and V2V continuation further trims
 * `contextFrames` pixel frames off the front (that span is a frozen copy of
 * the source video, not newly generated content — see `SourceVideoInput`).
 *
 * ```
 * v_latent(P) = (P-1)/8 + 1        (integer division)
 * f_total     = Σ v_latent(clip_i) − (n−1) × overlapFrames
 * total_px    = (f_total−1) × 8 + 1
 * output      = total_px                  (Clips)
 *             = total_px − contextFrames  (V2V)
 * ```
 *
 * Geometrically-impossible inputs (e.g. `overlapFrames` too large for the
 * shortest clip, or `contextFrames` exceeding `total_px`) raise a
 * `ValueError` server-side (`compute_chain_layout`); this client-side copy
 * instead clamps to `0` so it stays a total function safe to call from a
 * live preview while the user is still mid-edit.
 *
 * 素材（末尾, `end_source`）は**この式に一切入らない**, and that is all a
 * caller needs to know whether the chain is one clip (窓内モード, 2026-08-17)
 * or several (reverse mode, 2026-08-18): the anchor freezes the last
 * `END_SOURCE_CONTEXT_FRAMES` (8) frames OF THE LAST CLIP ITSELF, INSIDE the
 * length this function returns, so the delivered file is exactly `total_px`
 * whether or not the end slot holds material and however many clips it has.
 * `useChainForm`'s `outputFrames` therefore adds nothing on top of this call.
 *
 * (The backend's older multi-clip path — `internal_segment` mode, dead code
 * kept only for rollback — appends the band as an internal segment AFTER the
 * clips instead: `出力長 = クリップ合計 + 帯`. That addition turned out to be
 * unnecessary here even once the UI's one-clip limit was lifted (2026-08-18,
 * Docs/PENDING_TASKS_CLOSED.md §3-84 second stage): reverse mode — 2 or
 * more clips with an end source —
 * freezes the band inside the LAST clip's own tail exactly like the
 * single-clip window-internal case, appending nothing, so `total_px ==
 * clips_total_px` holds for every clip count and this function needed no
 * caller-side addition after all.) */
export function computeOutputFrames(
  clipNumFrames: number[],
  overlapFrames: number,
  contextFrames: number | null,
): number {
  if (clipNumFrames.length === 0) return 0;

  const segLatent = clipNumFrames.map(vLatentFrames);
  const sumLatent = segLatent.reduce((sum, frames) => sum + frames, 0);
  const fTotal = Math.max(0, sumLatent - (segLatent.length - 1) * overlapFrames);
  const totalPx = fTotal > 0 ? pxFromVLatent(fTotal) : 0;

  const output = contextFrames != null ? totalPx - contextFrames : totalPx;
  return Math.max(0, output);
}

/** Snaps a per-clip `num_frames` value onto the 8n+1 grid, clamped to
 * `[MIN_CLIP_NUM_FRAMES, MAX_CLIP_NUM_FRAMES]` (or a narrower caller-supplied
 * max, e.g. what's left of the total-frames budget). */
export function snapClipNumFrames(raw: number, max: number = MAX_CLIP_NUM_FRAMES): number {
  return snapNumFrames(raw, MIN_CLIP_NUM_FRAMES, Math.max(MIN_CLIP_NUM_FRAMES, max));
}

export function clampOverlapFrames(raw: number): number {
  return clamp(Math.round(raw), MIN_OVERLAP_FRAMES, MAX_OVERLAP_FRAMES);
}

export function clampOverlapStrength(raw: number): number {
  return clamp(raw, MIN_OVERLAP_STRENGTH, MAX_OVERLAP_STRENGTH);
}

export function clampEndSourceStrength(raw: number): number {
  return clamp(raw, MIN_END_SOURCE_STRENGTH, MAX_END_SOURCE_STRENGTH);
}

/** Recommended per-clip `num_frames` for a chosen chain preset (Sprint 2
 * item 9) — pure mirror of `gradio_ui/presets.py`'s
 * `_chain_preset_clip_recommendation` (line ~264-272,
 * `Nz-LTX23-backend/gradio_ui/presets.py`): the resolution's
 * `limits.spill_free_frames["{width}x{height}"]` comfortable cap when the
 * server publishes one for this EXACT `(width, height)`, else the preset's
 * own `num_frames`. The result is additionally snapped onto the 8n+1 grid via
 * {@link snapClipNumFrames} (defensive — server presets/spill thresholds are
 * already valid 8n+1 values in practice, but this keeps the function a total,
 * safe-to-call-anytime helper like every other snap* in this file). */
export function recommendedClipFrames(
  width: number,
  height: number,
  presetNumFrames: number,
  spillFreeFrames: Record<string, number> | undefined,
): number {
  const threshold = spillFreeFrames?.[`${width}x${height}`];
  const raw = threshold ?? presetNumFrames;
  return snapClipNumFrames(raw);
}

// ── A2V wav-duration geometry (Sprint 2 item 10) ────────────────────────────
// Pure transcriptions of `Nz-LTX23-backend/chain_math.py` (the audio-latent
// geometry single source of truth) and `Nz-LTX23-backend/gradio_ui/handlers.py`
// (the A2V "suggest Frames from an attached wav" + length-precheck UX built on
// top of it). Kept in this file rather than duplicated ad hoc in
// `useChainForm.ts` so they're independently unit-testable against the same
// worked examples the backend's own tests use.

/** `sample_rate / hop_length / audio_latent_downsample_factor` = `16000/160/4`
 * (`chain_math.py:42`) — audio-latent frames per second of decoded audio. */
export const AUDIO_LATENTS_PER_SEC = 25.0;

/** Python's built-in `round()` — round-half-to-EVEN ("banker's rounding"), NOT
 * JavaScript's `Math.round` (which is round-half-UP for positive values).
 *
 * Every `round(...)` in `Nz-LTX23-backend/chain_math.py` is this function, and
 * the two disagree on exact `.5` values: `pyRound(12.5) === 12` where
 * `Math.round(12.5) === 13`. That is not a theoretical difference — the
 * `a_frames_for_px` grid lands exactly on `.5` for whole families of real
 * inputs (every 8n+1 clip length at 50fps, for instance: `49/50*25 = 24.5`),
 * and a single off-by-one audio-latent frame is the difference between a
 * request the server accepts and a 422. So EVERY mirror of a backend `round()`
 * in this file goes through here, never through `Math.round`.
 *
 * (`Math.round` is still correct for the UI-side clamps in this file —
 * `clampCropOutput`/`clampOverlapFrames` mirror no backend arithmetic, they
 * just tidy a slider value.)
 *
 * Implementation note: `x - Math.floor(x)` is exact for every |x| < 2^52, so
 * the `=== 0.5` test really does only fire on true half-way values. */
export function pyRound(x: number): number {
  const lower = Math.floor(x);
  const frac = x - lower;
  if (frac > 0.5) return lower + 1;
  if (frac < 0.5) return lower;
  // Exact half: pick the EVEN neighbour. `%` keeps the sign of the dividend in
  // JS, so `-1 % 2` is `-1` (odd) — which is what we want: round(-0.5) === 0.
  return lower % 2 === 0 ? lower : lower + 1;
}

/** Audio-latent-frame count for a pixel-frame span at `fps`. Copy of
 * `chain_math.a_frames_for_px` (`chain_math.py:274-276`): `round(pixel_frames
 * / fps * AUDIO_LATENTS_PER_SEC)` — through {@link pyRound}, see its doc
 * comment for why `Math.round` is wrong here. Not exported — only the helpers
 * in this section need it, mirroring the Python module's own scoping. */
function audioFramesForPx(pixelFrames: number, fps: number): number {
  return pyRound((pixelFrames / fps) * AUDIO_LATENTS_PER_SEC);
}

/** Audio-latent frames an uploaded `durationSec`-long track VAE-encodes to —
 * mirror of `services/pipeline_manager.py`'s `preflight_source_audio`
 * (`available = round(duration * chain_math.AUDIO_LATENTS_PER_SEC)`), the
 * number the server compares against {@link audioLatentsRequired} before it
 * raises `SOURCE_AUDIO_TOO_SHORT` (422). Uses {@link pyRound} for the same
 * reason everything else in this section does. */
export function audioLatentsAvailable(durationSec: number): number {
  return pyRound(durationSec * AUDIO_LATENTS_PER_SEC);
}

/** Mirrors `gradio_ui/handlers.py`'s `_resolve_fps`: falls back to 24 when
 * `fps` is falsy (0/NaN/undefined-as-number) — the same fallback
 * {@link suggestFramesForAudio}/{@link rawFramesForAudio}, the length
 * precheck and `audioFit.ts`'s resolver all use for a missing/zero frame
 * rate. */
export function resolveA2vFps(fps: number): number {
  return fps || 24;
}

/** Total audio-latent frames the assembled chain timeline requires — pure
 * copy of `chain_math.audio_latents_required` (`chain_math.py:412-424`),
 * itself `compute_chain_layout(...).a_total` for the given clip lengths.
 * Reproduces just the pieces of `compute_chain_layout` (`chain_math.py:224
 * onward`) needed for `a_total`: `seg_latent = clip_frames.map(vLatentFrames)`,
 * `f_total = sum(seg_latent) - (n-1)*kv`, `total_px = pxFromVLatent(f_total)`,
 * `a_total = audioFramesForPx(total_px, fps)`.
 *
 * Returns `null` instead of throwing when the geometry is impossible — the
 * server's `any(kv >= L for L in seg_latent)` guard (`chain_math.py:289-293`,
 * "overlap_frames (K_v) must be < every clip's stage-1 latent frames") would
 * raise a `ValueError` there; this client-side mirror surfaces that as "cannot
 * determine a requirement" so callers (the A2V suggestion loop and the
 * length precheck) can treat it as "nothing to gate on yet" rather than
 * crashing a live UI. */
export function audioLatentsRequired(clipFrames: number[], fps: number, kv: number = DEFAULT_OVERLAP_FRAMES): number | null {
  const segLatent = clipFrames.map(vLatentFrames);
  if (segLatent.some((latent) => kv >= latent)) return null;
  const fTotal = segLatent.reduce((sum, latent) => sum + latent, 0) - (clipFrames.length - 1) * kv;
  const totalPx = pxFromVLatent(fTotal);
  return audioFramesForPx(totalPx, fps);
}

/** The largest 8n+1 `num_frames` a `durationSec`-long clip at `fps` covers,
 * BEFORE the shrink-to-fit loop {@link suggestFramesForAudio} runs — exported
 * for the batch A2V flow's own "would this file even need shrinking" skip
 * check. Pure copy of the first line of `handlers.suggest_frames_for_audio`
 * (`gradio_ui/handlers.py:94-95`): `((floor(dur*fps) - 1) // 8) * 8 + 1`,
 * clamped only at the `9` floor (no `481` ceiling — that clamp belongs to
 * {@link suggestFramesForAudio} alone). */
export function rawFramesForAudio(durationSec: number, fps: number): number {
  const fpsV = resolveA2vFps(fps);
  const nf = Math.floor((Math.floor(durationSec * fpsV) - 1) / 8) * 8 + 1;
  return Math.max(nf, MIN_CLIP_NUM_FRAMES);
}

/** Suggests a `num_frames` (8n+1) that fits `durationSec` seconds of audio at
 * `fps` — pure copy of `handlers.suggest_frames_for_audio`
 * (`gradio_ui/handlers.py:75-108`). Starts from {@link rawFramesForAudio},
 * then shrinks by 8 while the audio would VAE-encode to fewer latent frames
 * than that `num_frames` requires (checked via {@link audioLatentsRequired}
 * with the SAME fixed `kv=3` the A2V chain payload always uses, regardless of
 * whatever the Chain form's own seam-blend `overlapFrames` slider currently
 * holds — mirrors the Python docstring's "kv=3 ... mirroring the A2V chain
 * payload's fixed overlap_frames"), finally clamped to
 * `[MIN_CLIP_NUM_FRAMES, MAX_CLIP_NUM_FRAMES]`. */
export function suggestFramesForAudio(durationSec: number, fps: number): number {
  const fpsV = resolveA2vFps(fps);
  let nf = rawFramesForAudio(durationSec, fpsV);
  while (nf > MIN_CLIP_NUM_FRAMES) {
    const required = audioLatentsRequired([nf], fpsV, 3);
    if (required === null) break;
    const available = audioLatentsAvailable(durationSec);
    if (available >= required) break;
    nf -= 8;
  }
  return clamp(nf, MIN_CLIP_NUM_FRAMES, MAX_CLIP_NUM_FRAMES);
}

export interface AudioLengthPrecheckResult {
  /** `true` when the attached audio's measured duration VAE-encodes to at
   * least as many audio-latent frames as `numFrames` requires. */
  ok: boolean;
  /** Seconds of audio the current `numFrames` needs — meaningful whether or
   * not `ok` (0 when the requirement couldn't be determined, e.g. `numFrames`
   * too small for the fixed `kv=3` guard, mirroring
   * {@link audioLatentsRequired}'s `null`). */
  requiredSeconds: number;
}

/** Client-side mirror of the A2V length precheck in
 * `handlers.make_generate_handler` (`gradio_ui/handlers.py:462-487`,
 * `a2v_msg_too_short`): rejects BEFORE any upload/API call when the attached
 * wav's measured duration would VAE-encode to fewer audio-latent frames than
 * the chain's `numFrames` requires. `required` uses the SAME fixed `kv=3` as
 * {@link suggestFramesForAudio}; `available = round(durationSec *
 * AUDIO_LATENTS_PER_SEC)` mirrors the Python precheck's own `available`.
 * `requiredSeconds = required / AUDIO_LATENTS_PER_SEC`, matching the Python
 * precheck's `need_s = required / chain_math.AUDIO_LATENTS_PER_SEC`. When the
 * requirement can't be computed at all (`audioLatentsRequired` returns
 * `null`), this defers to the server rather than blocking submission —
 * `{ ok: true, requiredSeconds: 0 }`. */
export function audioLengthPrecheck(durationSec: number, numFrames: number, fps: number): AudioLengthPrecheckResult {
  const fpsV = resolveA2vFps(fps);
  const required = audioLatentsRequired([numFrames], fpsV, 3);
  if (required === null) return { ok: true, requiredSeconds: 0 };
  const available = audioLatentsAvailable(durationSec);
  return { ok: available >= required, requiredSeconds: required / AUDIO_LATENTS_PER_SEC };
}

// ── Long-form A2V geometry (§1-16: one audio track across a whole chain) ─────
// The helpers below exist because a chain that carries a `source_audio` is
// checked by the server against MORE of `chain_math.compute_chain_layout` than
// `audioLatentsRequired` alone covers: the layout function raises on several
// geometrically-impossible configurations, and every one of those is a 422 the
// user sees only after pressing Generate. `chainLayoutError` mirrors those
// rejections so the UI (and `audioFit.ts`'s auto-fit resolver) can never
// propose a chain the server would refuse.

/** Total audio-latent frames the LONGEST possible chain needs: 24 clips of the
 * maximum 481 frames each. The ceiling behind the "this audio is far too long
 * for any chain" hard error at attach time — no arrangement of clips can ever
 * consume more than this. Reuses {@link audioLatentsRequired} rather than
 * re-deriving the arithmetic. */
export function maxChainAudioLatents(fps: number, kv: number = DEFAULT_OVERLAP_FRAMES): number {
  const required = audioLatentsRequired(Array<number>(MAX_CLIPS).fill(MAX_CLIP_NUM_FRAMES), resolveA2vFps(fps), kv);
  // Unreachable in practice (`kv` is clamped to [1, 8] and `vLatentFrames(481)`
  // is 61), but a `null` here must NOT collapse the ceiling to 0 and start
  // rejecting every attachment — fail open.
  return required ?? Number.POSITIVE_INFINITY;
}

/** Smallest `num_frames` a clip may have for a given seam overlap `kv`.
 * `compute_chain_layout` (`chain_math.py:516-520`) rejects any chain where
 * `kv >= v_latent_frames(clip)`, so a clip needs at least `kv + 1` video-latent
 * frames — `px_from_v_latent(kv + 1)` = `8*kv + 1` pixel frames — on top of the
 * absolute {@link MIN_CLIP_NUM_FRAMES} floor. 25 at the default `kv=3`. */
export function minClipFramesForKv(kv: number): number {
  return Math.max(MIN_CLIP_NUM_FRAMES, 8 * kv + 1);
}

/** One stage-1 segment's slice of the GLOBAL audio timeline. Mirror of one
 * `(start, len)` pair from `chain_math.audio_segment_windows`, plus the same
 * span expressed in seconds (`/ AUDIO_LATENTS_PER_SEC`) for the clip card's
 * "🎵 担当時間帯" badge. */
export interface AudioSegmentWindow {
  /** Start offset on the global audio-latent timeline. */
  startLatent: number;
  /** Length in audio-latent frames (== `seg_audio[i]`). */
  lenLatent: number;
  /** `startLatent / 25`. */
  startSec: number;
  /** `(startLatent + lenLatent) / 25`. */
  endSec: number;
}

/** Per-clip window on the uploaded audio track — verbatim mirror of
 * `chain_math.audio_segment_windows` (`chain_math.py:690-707`) plus the
 * `ka_list` derivation from `compute_chain_layout` (`chain_math.py:552-566`)
 * that feeds it.
 *
 * Each stage-1 segment hard-freezes the slice of the uploaded audio latent that
 * lands under it. Segment `i` covers `seg_audio[i]` audio-latent frames and
 * consecutive segments overlap by `ka_list[i]` (the per-join audio crossfade),
 * chosen so `sum(seg_audio) - sum(ka) == a_total` EXACTLY. A single clip
 * therefore reduces to `[(0, a_total)]` and the last window always ends at
 * `a_total`.
 *
 * Returns `null` — never throws — on the two geometries the Python raises a
 * `ValueError` for (`kv >= v_latent_frames(clip)`; `sum_ka < joins`), so this
 * is safe to call from a live render while the user is mid-edit. Note this
 * covers only the audio-window rejections: use {@link chainLayoutError} for the
 * full set the server enforces. */
export function audioSegmentWindows(
  clipFrames: number[],
  fps: number,
  kv: number = DEFAULT_OVERLAP_FRAMES,
): AudioSegmentWindow[] | null {
  const n = clipFrames.length;
  if (n < 1) return null;
  const fpsV = resolveA2vFps(fps);

  const segLatent = clipFrames.map(vLatentFrames);
  const segAudio = clipFrames.map((frames) => audioFramesForPx(frames, fpsV));
  if (segLatent.some((latent) => kv >= latent)) return null;

  const fTotal = segLatent.reduce((sum, latent) => sum + latent, 0) - (n - 1) * kv;
  const aTotal = audioFramesForPx(pxFromVLatent(fTotal), fpsV);

  // Per-join audio overlap K_a so the assembled audio lands on a_total exactly.
  const kaList: number[] = [];
  if (n > 1) {
    const sumKa = segAudio.reduce((sum, len) => sum + len, 0) - aTotal;
    const nJoin = n - 1;
    if (sumKa < nJoin) return null;
    const baseKa = Math.floor(sumKa / nJoin);
    const remKa = sumKa % nJoin;
    for (let j = 0; j < nJoin; j += 1) kaList.push(baseKa + (j < remKa ? 1 : 0));
  }

  const windows: AudioSegmentWindow[] = [];
  let start = 0;
  segAudio.forEach((alen, i) => {
    windows.push({
      startLatent: start,
      lenLatent: alen,
      startSec: start / AUDIO_LATENTS_PER_SEC,
      endSec: (start + alen) / AUDIO_LATENTS_PER_SEC,
    });
    const ka = kaList[i];
    if (ka !== undefined) start += alen - ka;
  });
  return windows;
}

/** 素材（末尾）reverse mode (2+ clips, 2.6(f)): mirrors `chain_math.py`'s
 * degenerate-audio-overlap guard (`Nz-LTX23-backend/chain_math.py:1120-1122`)
 * that {@link chainLayoutError}'s `"degenerateAudioOverlap"` already covers,
 * but WITHOUT that function's `audioReady` precondition — `end_source` and
 * `source_audio` are mutually exclusive server-side, so a chain ending on
 * material never has a `source_audio` to make `chainLayoutError` reachable,
 * and this is the same arithmetic freed of that gate:
 *
 * ```
 * sum_ka = Σ seg_audio[i] − a_total
 * n_join = n_clips − 1
 * ok     = sum_ka >= n_join
 * ```
 *
 * `reverse` mode appends no internal-segment band, so `seg_frames ==
 * clipFrames` and this is exactly {@link audioSegmentWindows}'s own internal
 * `sumKa`/`nJoin` — this just isolates the ONE boolean that function's `null`
 * return conflates with the separate `kv >= clip latent` failure (already
 * `clipTooShortForOverlap`'s job, not this gate's — hence `true`, not a
 * block, whenever that other failure applies). Also `true` for `n < 2`
 * (nothing to join — the 窓内モード single-clip case never reaches this
 * check server-side either). */
export function endSourceAudioOverlapOk(clipFrames: number[], fps: number, kv: number): boolean {
  const n = clipFrames.length;
  if (n < 2) return true;
  const fpsV = resolveA2vFps(fps);
  const segLatent = clipFrames.map(vLatentFrames);
  if (segLatent.some((latent) => kv >= latent)) return true;
  const segAudio = clipFrames.map((frames) => audioFramesForPx(frames, fpsV));
  const fTotal = segLatent.reduce((sum, latent) => sum + latent, 0) - (n - 1) * kv;
  const aTotal = audioFramesForPx(pxFromVLatent(fTotal), fpsV);
  const sumKa = segAudio.reduce((sum, len) => sum + len, 0) - aTotal;
  const nJoin = n - 1;
  return sumKa >= nJoin;
}

/** Why `compute_chain_layout` would reject a chain, as a stable identifier the
 * caller maps to copy. One per `raise ValueError` the source-less/retake-less
 * path of `chain_math.compute_chain_layout` can hit, named after its message.
 *
 * `audioTileOutOfBounds`/`audioOverlapMismatch`/`lastAudioTileShort` are the
 * three siblings of `audioReassemblyMismatch` in the same stage-2 verification
 * block; they are mirrored for completeness (and checked in the same order the
 * Python checks them) even though the reassembly check is the one that fires in
 * practice. */
export type ChainLayoutError =
  | "noClips"
  | "kvTooLarge"
  | "degenerateAudioOverlap"
  | "audioTileOutOfBounds"
  | "audioOverlapMismatch"
  | "audioReassemblyMismatch"
  | "lastAudioTileShort";

/** `null` when `chain_math.compute_chain_layout(clipFrames, fps, kv=kv,
 * v_tile=…, v_adv=…)` would succeed, otherwise the identifier of the FIRST
 * `ValueError` it would raise — a verbatim mirror of that function's
 * source-less, retake-less path (`chain_math.py:392-608`), in the same order.
 *
 * Why the front end needs this at all: the geometry is not a smooth function of
 * the inputs. At non-integer frame rates the stage-2 audio tiling re-assembles
 * to `a_total ± 1` for whole families of otherwise-reasonable chains — e.g.
 * `[257, 257] @ 29.97fps, kv=3` is rejected (`audio reassembly 414 != a_total
 * 415`) while the identical clip list at 24fps is fine. That is an EXISTING
 * chain-wide constraint, not an A2V one; §1-16 mirrors it only so the auto-fit
 * resolver can never hand the user a construction the server will 422.
 *
 * `window` is `{ vTile, vAdv }` — pass one of `shell/tokenBudget.ts`'s
 * `STAGE2_WINDOW_PRESETS` entries (`standard` 22/18, `high_resolution` 19/12).
 *
 * NOT mirrored (deliberately): the `source_context_px` (V2V) and
 * `retake_glue_px` branches. Both are mutually exclusive with `source_audio`
 * server-side, so a long-form A2V chain never takes them. */
export function chainLayoutError(
  clipFrames: number[],
  fps: number,
  kv: number,
  window: { vTile: number; vAdv: number },
): ChainLayoutError | null {
  const n = clipFrames.length;
  if (n < 1) return "noClips";
  const fpsV = resolveA2vFps(fps);
  const { vTile, vAdv } = window;
  const ktV = vTile - vAdv;

  const segLatent = clipFrames.map(vLatentFrames);
  const segAudio = clipFrames.map((frames) => audioFramesForPx(frames, fpsV));
  if (segLatent.some((latent) => kv >= latent)) return "kvTooLarge";

  const fTotal = segLatent.reduce((sum, latent) => sum + latent, 0) - (n - 1) * kv;
  const totalPx = pxFromVLatent(fTotal);
  const aTotal = audioFramesForPx(totalPx, fpsV);

  if (n > 1) {
    const sumKa = segAudio.reduce((sum, len) => sum + len, 0) - aTotal;
    if (sumKa < n - 1) return "degenerateAudioOverlap";
  }

  // ── stage-2 tile layout (video-latent domain) ──────────────────────────────
  // `v_starts = list(range(0, max(1, f_total - kt_v), v_adv))`, then a final
  // start is appended (and the list re-`sorted(set(...))`) whenever the tiles so
  // far do not reach the end. The `max(1, …)` guarantees at least the start `0`,
  // so the Python's `not v_starts` arm is dead — this mirror simply reads the
  // last element, which therefore always exists.
  const vStarts: number[] = [];
  for (let s = 0; s < Math.max(1, fTotal - ktV); s += vAdv) vStarts.push(s);
  const lastStart = vStarts[vStarts.length - 1] ?? 0;
  if (lastStart + vTile < fTotal) {
    const tailStart = Math.max(0, fTotal - vTile);
    // `set(...)`: the loop above produces strictly increasing values, so the
    // only possible duplicate is `tailStart` itself.
    if (!vStarts.includes(tailStart)) vStarts.push(tailStart);
    vStarts.sort((a, b) => a - b);
  }
  const nTiles = vStarts.length;

  // Audio tiles, time-aligned to the video advance. NOTE `audio_adv` derives
  // from `v_adv * 8` (a pure delta), NOT from `px_from_v_latent(v_adv)` — the
  // causal keyframe belongs to a LENGTH, never to an advance.
  const audioAdv = pyRound(((vAdv * 8) / fpsV) * AUDIO_LATENTS_PER_SEC);
  const aLenFull = audioFramesForPx(pxFromVLatent(vTile), fpsV);
  const ktA = aLenFull - audioAdv;
  const aTiles = vStarts.map((vs, i) => {
    const vLen = Math.min(vs + vTile, fTotal) - vs;
    return { start: i * audioAdv, len: audioFramesForPx(pxFromVLatent(vLen), fpsV) };
  });

  if (nTiles > 1) {
    let aReassembled = 0;
    let prevEnd = 0;
    for (const [i, tile] of aTiles.entries()) {
      if (i === 0) {
        aReassembled = tile.len;
      } else {
        aReassembled = aReassembled + tile.len - ktA;
        if (tile.start + tile.len > aTotal) return "audioTileOutOfBounds";
        if (prevEnd - tile.start !== ktA) return "audioOverlapMismatch";
      }
      prevEnd = tile.start + tile.len;
    }
    if (aReassembled !== aTotal) return "audioReassemblyMismatch";
    if (prevEnd !== aTotal) return "lastAudioTileShort";
  }

  return null;
}

/** V2V `context_frames` validity (Docs/API_REFERENCE.md §5.2
 * `SourceVideoSpec`): 8n+1, within `[min, max]` from
 * `config.limits.v2v_context_frames_min/max`, and strictly less than
 * `clips[0].num_frames` — a source can't "carry over" more frames than the
 * very first clip is going to produce. */
export function isContextFramesValid(
  contextFrames: number,
  min: number,
  max: number,
  firstClipNumFrames: number,
): boolean {
  return (
    Number.isInteger(contextFrames) &&
    (contextFrames - 1) % 8 === 0 &&
    contextFrames >= min &&
    contextFrames <= max &&
    contextFrames < firstClipNumFrames
  );
}

/** Snaps a `context_frames` candidate onto the 8n+1 grid within `[min, max]`
 * (the `firstClipNumFrames < ` constraint is validated, not snapped, since
 * clamping it there could silently produce a value below `min`). */
export function snapContextFrames(raw: number, min: number, max: number): number {
  return snapNumFrames(raw, min, Math.max(min, max));
}

// ── 素材（末尾）end source geometry ────────────────────────
// 窓内モード (single clip) still asks no geometry question of its own — this
// section used to hold the geometry the v1 "末尾フレーム数" slider needed
// (`MIN_END_CONTEXT_FRAMES` / `END_CONTEXT_FRAMES_STEP` / `snapEndContextFrames`
// / `isEndContextFramesValid` / `endContextRequiredLastClipFrames` /
// `endContextFitsChain`), and every one of them answered a question 窓内モード
// (2026-08-17) does not ask:
//
//  - there is no slider to snap or validate: the anchor is the CONSTANT
//    `timeline/tailAlign.ts`'s `END_SOURCE_CONTEXT_FRAMES` (8), sent verbatim,
//    so an out-of-range value is unreachable;
//  - "does the freeze fit the ONE clip?" is answered by construction: 8 frames
//    inside a clip that is at least `MIN_CLIP_NUM_FRAMES` long always fit, and
//    the backend's own version of the check went away with the v1 geometry —
//    keeping a client mirror of a rule the server no longer has would block
//    chains the server accepts.
//
// reverse mode (2+ clips, second stage, 2026-08-18) reintroduces exactly ONE
// real acceptance check — {@link endSourceLastClipHasFreeLatents} below — plus
// the audio-budget mirror ({@link endSourceAudioOverlapOk} above): the anchor
// is the LAST clip's own tail there too (nothing is appended), but that clip's
// HEAD is also spoken for by the previous clip's carried-forward のりしろ, so a
// short-enough last clip can run out of latents to generate at all.
//
// What the UI additionally bounds is quality, not geometry: a clip longer than
// one stage-2 tile (`pxFromVLatent(vTile)` — 169 / 145) puts the anchor and the
// frames that must blend into it in different tiles, which smears the join.
// That is a WARNING computed in `useChainForm` (`endSourceQualityLimitFrames`,
// single-clip only — the quality study behind it never ran a multi-clip
// chain), never a block, and the right-click route rounds its seed below it
// (`timeline/prefillSeed.ts`'s `END_SOURCE_SEED_MAX_FRAMES`).

/** 素材（末尾）reverse mode (2+ clips, 2.6(f)): mirrors `chain_math.py`'s
 * reverse-mode reception check (`Nz-LTX23-backend/chain_math.py:1076-1088`:
 * rejects when `kv + n_end_v >= clip_latent[-1]`). The LAST clip's tail holds
 * the end-source band (`nEndV` latents, {@link vTailLatents}) and its HEAD is
 * what the PREVIOUS clip's carried-forward のりしろ freezes (`kv` latents), so
 * if those two together already fill the clip there is nothing left for it to
 * generate — the reverse carry would just be crossfading the upload into its
 * own middle. `true` (ok) whenever the clip has MORE video-latent frames than
 * `kv + nEndV`. */
export function endSourceLastClipHasFreeLatents(lastClipFrames: number, kv: number, nEndV: number): boolean {
  return kv + nEndV < vLatentFrames(lastClipFrames);
}

/** One clip card's editable state, owned by `useChainForm`. `prompt` empty
 * means "use the chain-wide shared prompt" (task brief §1: "空=共通プロンプ
 * トを使用"); `conditioningImages` is only ever read for index 0 by
 * `buildChainRequest`. */
export interface ChainClipInput {
  id: string;
  prompt: string;
  numFrames: number;
  conditioningImages?: ConditioningImage[];
  /** §1-16 (long-form A2V): `false` once the user has TOUCHED this card — an
   * edit to its prompt or a manual change to its length. `undefined` (the
   * out-of-the-box state) and `true` both mean "untouched", which is what
   * {@link isClipIntact} encodes.
   *
   * The flag is IRREVERSIBLE (nothing ever sets it back to `true`) and it is
   * the ONLY thing the "↔️ 再生時間の自動調整" resolver (`audioFit.ts`) is
   * allowed to look at: it may resize, add or drop intact cards, while a card
   * with `intact: false` keeps its length, its position AND its existence, no
   * matter what the audio needs. Applying a preset does NOT clear it (the user
   * did not touch this card), and neither does the resolver's own resizing. */
  intact?: boolean;
}

/** Whether the auto-fit resolver may touch this card — `intact !== false`, so
 * an absent flag counts as intact. See {@link ChainClipInput.intact}. */
export function isClipIntact(clip: Pick<ChainClipInput, "intact">): boolean {
  return clip.intact !== false;
}

export interface SourceVideoInput {
  videoId: string;
  contextFrames: number;
}

/** 素材（末尾）: the ONE image or video the chain must END with
 * (`GenerateChainRequest.end_source`). `kind` decides which of the server's two
 * mutually-exclusive id fields carries {@link EndSourceInput.id} — the server
 * takes exactly one, and the form knows which upload store the id came from, so
 * the branch is made here rather than by guessing at the id's shape.
 *
 * `contextFrames` is a MULTIPLE OF 8 (not 8n+1 — the causal VAE's `+1` keyframe
 * belongs to the START of a span and a tail span carries none). v2: it is no
 * longer a user setting at all — 8 for an image, and for a video the band
 * `timeline/tailAlign.ts` derives from the length the server measured. Mutually
 * exclusive with {@link SourceAudioInput} and the reference video server-side;
 * combinable with {@link SourceVideoInput} (that is the "start here, end there"
 * interpolation case). */
export interface EndSourceInput {
  /** `"video"` -> `end_source.video_id`, `"image"` -> `end_source.image_id`. */
  kind: "image" | "video";
  /** The `video_id` from `POST /upload/video`, or the `image_id` from
   * `POST /upload/image`, per {@link EndSourceInput.kind}. */
  id: string;
  contextFrames: number;
  /** The anchor's fixed strength (`end_source.strength`, バッチ1・2026-08-18):
   * 1.0 = hard freeze (default, byte-identical to before this field existed),
   * lower softens Stage-1's freeze mask. Always sent — see
   * {@link DEFAULT_END_SOURCE_STRENGTH}'s own doc comment for why the value
   * still matters even though Stage-2 re-hard-freezes regardless. */
  strength: number;
}

/** §1-16 (long-form A2V): the ONE uploaded audio track a chain drives its whole
 * timeline from (`GenerateChainRequest.source_audio`). No per-clip split — the
 * server assigns each stage-1 segment its own window of the same upload (see
 * {@link audioSegmentWindows}). Mutually exclusive with {@link SourceVideoInput}
 * server-side. */
export interface SourceAudioInput {
  /** The `audio_id` from `POST /upload/audio`. */
  audioId: string;
}

export interface BuildChainRequestParams {
  /** Already stripped of `<lora:...>` tags by the caller — Chain now sends
   * `loras` too (Docs/API_REFERENCE.md §5.2 grew the field 2026-07-03/07-11),
   * but as a separate array, not inline in the prompt text. The caller is
   * expected to run the same `parseLoraPrompt` split Create's
   * `toGenerateRequest` does (`modes/single/useGenerationForm.ts`) and pass
   * the two halves through as `prompt` (stripped) and `loras` below. */
  prompt: string;
  width: number;
  height: number;
  /** N1: opt-in output crop (`GenerateChainRequest.crop_output`). `null`/
   * `undefined` omits the field entirely, matching every other optional field
   * on this params object. See `clampCropOutput`/`isCropOutputValid` above
   * for the 32-pixel-grid constraint this is expected to already satisfy. */
  cropOutput?: CropOutput | null;
  frameRate: number;
  seed: number;
  overlapFrames: number;
  overlapStrength: number;
  clips: ChainClipInput[];
  sourceVideo?: SourceVideoInput | null;
  /** §1-16 long-form A2V (`GenerateChainRequest.source_audio`). `null`/
   * `undefined` omits the field entirely, matching every other optional field
   * on this params object. The caller (`useChainForm`) is responsible for never
   * passing this alongside {@link sourceVideo} — the server rejects the pair. */
  sourceAudio?: SourceAudioInput | null;
  /** 素材（末尾）(`GenerateChainRequest.end_source`). `null`/`undefined` omits
   * the field entirely, matching every other optional field on this params
   * object — an end-source-less request stays byte-identical to before the
   * field existed. The caller (`useChainForm`) is responsible for never passing
   * this alongside {@link sourceAudio} or a reference video; the server rejects
   * those pairs. */
  endSource?: EndSourceInput | null;
  /** Style/control IC-LoRAs parsed from the shared prompt's
   * `<lora:name:strength>` tags. Omitted (not sent as `[]`) when empty,
   * matching Create's `toGenerateRequest` pattern. */
  loras?: LoraSpec[];
  /** Opt into temporal-chunk upsampling (Docs/API_REFERENCE.md §5.2,
   * `GenerateChainRequest.chunked_upsample`). Required, not optional: the
   * server defaults to `false` (the one-pass path) when the field is
   * omitted, so this must always be sent explicitly or long/high-res chains
   * permanently fall back to the OOM-prone one-shot upsample. */
  chunkedUpsample: boolean;
  /** Stage-2 window preset (`GenerateChainRequest.stage2_window`, owner decision
   * 2026-08-09). Optional and OMITTED at its `"standard"` default — the same
   * treatment `crop_output`/`loras`/`source_video` get, and deliberately NOT
   * {@link chunkedUpsample}'s "always explicit" treatment: the server default IS
   * what we want when the user hasn't opted in, so staying silent keeps the
   * request byte-identical to before this field existed. */
  stage2Window?: Stage2Window;
  /** Reference-video CONTROL IC-LoRA source (sprint 2, Clips-submode-only,
   * `GenerateChainRequest.reference_video_id`) — the `video_id` from
   * `POST /upload/video`. `null`/`undefined` omits the field entirely,
   * matching every other optional field on this params object; the caller
   * (`useChainForm`) is responsible for only ever passing a value while in
   * the Clips submode (structurally keeps this mutually exclusive with
   * `sourceVideo`, matching `api/models.py:512-517`). */
  referenceVideoId?: string | null;
  /** 0.0-1.0; requires `loras` server-side (`GenerateChainRequest.conditioning_attention_strength`).
   * `null`/`undefined` omits the field — `0` is a valid, distinct value and
   * must still be sent (checked with `!= null`, not truthiness). */
  conditioningAttentionStrength?: number | null;
  /** 0.0-1.0; requires `loras` server-side (`GenerateChainRequest.reference_video_strength`).
   * Same `!= null` (not truthy) omission rule as {@link conditioningAttentionStrength}. */
  referenceVideoStrength?: number | null;
  /** NAG (2026-07-28)/VSF (2026-07-29): the shared Negative Prompt accordion's
   * settings (`shell/nagSettings.ts`). `undefined`, or `enabled: false`, omits
   * all 7 additive fields entirely — see `nagRequestFields`'s own doc comment
   * (D2: the single place this additive contract is built). */
  nag?: NagSettings;
  /** Acceleration (2026-07-31, backend §43): the Settings panel's shared
   * attention-backend choice (`shell/accelerationSettings.ts`).
   * `undefined`, or the server-default backend, omits the single additive
   * field entirely — see `accelerationRequestFields`'s own doc comment
   * (the one place this additive contract is built). */
  acceleration?: AccelerationSettings;
}

/** Assembles a `GenerateChainRequest` body from the Chain form's current
 * state. Per-clip `prompt` is omitted entirely (not sent as `""`) when the
 * user left it blank, and `conditioning_images` is only ever attached to
 * clip 0, matching `ChainClip`'s documented shape
 * (Docs/API_REFERENCE.md §5.2). */
export function buildChainRequest(params: BuildChainRequestParams): GenerateChainRequest {
  const clips: ChainClip[] = params.clips.map((clip, index) => {
    const out: ChainClip = { num_frames: clip.numFrames };
    if (clip.prompt.trim().length > 0) out.prompt = clip.prompt;
    if (index === 0 && clip.conditioningImages && clip.conditioningImages.length > 0) {
      out.conditioning_images = clip.conditioningImages;
    }
    return out;
  });

  const request: GenerateChainRequest = {
    prompt: params.prompt,
    width: params.width,
    height: params.height,
    frame_rate: params.frameRate,
    seed: params.seed,
    overlap_frames: params.overlapFrames,
    overlap_strength: params.overlapStrength,
    clips,
    // Always sent explicitly — see `chunkedUpsample`'s JSDoc above; omitting
    // it silently reverts to the server's one-pass upsample default.
    chunked_upsample: params.chunkedUpsample,
    // NAG (2026-07-28): `{}` (no keys) whenever NAG is off/unset, keeping this
    // request byte-identical to before NAG existed.
    ...nagRequestFields(params.nag),
    // Acceleration (2026-07-31): `{}` (no keys) unless the user moved off the
    // server default, keeping this request byte-identical to before.
    ...accelerationRequestFields(params.acceleration),
  };

  // Omitted at the default, like every other optional field below — see
  // `stage2Window`'s own JSDoc for why this does NOT follow
  // `chunked_upsample`'s always-explicit rule.
  if (params.stage2Window && params.stage2Window !== STAGE2_WINDOW_DEFAULT) {
    request.stage2_window = params.stage2Window;
  }
  if (params.cropOutput) {
    request.crop_output = params.cropOutput;
  }
  if (params.sourceVideo) {
    request.source_video = { video_id: params.sourceVideo.videoId, context_frames: params.sourceVideo.contextFrames };
  }
  if (params.sourceAudio) {
    request.source_audio = { audio_id: params.sourceAudio.audioId };
  }
  // 素材（末尾）: exactly ONE of `video_id`/`image_id` is sent, decided by the
  // slot the material came from (`EndSourceInput.kind`) — never both, and never
  // an `undefined` sibling key.
  if (params.endSource) {
    // `strength` rides along explicitly EVEN AT its 1.0 default — same
    // "always explicit" rule `contextFrames` follows here, so the request JSON
    // shows exactly what happened rather than relying on the server's default.
    request.end_source =
      params.endSource.kind === "video"
        ? { video_id: params.endSource.id, context_frames: params.endSource.contextFrames, strength: params.endSource.strength }
        : { image_id: params.endSource.id, context_frames: params.endSource.contextFrames, strength: params.endSource.strength };
  }
  if (params.loras && params.loras.length > 0) {
    request.loras = params.loras;
  }
  // `!= null` (not truthiness): `0.0` is a valid, distinct
  // conditioning_attention_strength/reference_video_strength value that must
  // still be sent, not treated as "unset".
  if (params.referenceVideoId != null) {
    request.reference_video_id = params.referenceVideoId;
  }
  if (params.conditioningAttentionStrength != null) {
    request.conditioning_attention_strength = params.conditioningAttentionStrength;
  }
  if (params.referenceVideoStrength != null) {
    request.reference_video_strength = params.referenceVideoStrength;
  }

  return request;
}
