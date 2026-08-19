/**
 * CHAIN-ONLY token-budget geometry: the §1-14 stage-2 window budget AND the
 * §1-15 stage-1-with-reference budget. Mirror of `Nz-Videomni/chain_math.py`
 * (`STAGE2_WINDOW_PRESETS` / `CHAIN_COMFORT_TOKEN_BUDGET` / `chain_window_tokens`
 * / `stage2_max_context_px` / `CHAIN_STAGE1_COMFORT_TOKEN_BUDGET` /
 * `chain_stage1_tokens`), which is the single source of truth — every number
 * here is pinned against the same expected-value tables the backend's
 * `tests/test_stage2_window.py` and `tests/test_chain_math_reference.py` use.
 *
 * ⚠ SCOPE: this is about ONE STAGE-2 WINDOW OF A CHAIN, and ONE STAGE-1
 * SEGMENT OF A CHAIN, nothing else.
 *
 * A single `/generate` refines its whole clip in one pass, so "how many
 * attention tokens is comfortable" is a DIFFERENT number there with a different
 * derivation. Do not import this module from Create/single-generate code, and do
 * not generalise `CHAIN_COMFORT_TOKEN_BUDGET` / `CHAIN_STAGE1_COMFORT_TOKEN_BUDGET`
 * into app-wide constants — every function below is named `chain*`/`stage2*` for
 * exactly that reason.
 *
 * This is also a SEPARATE axis from `config.limits.spill_free_frames`
 * (`modes/single/spillUtils.ts`): that one is the server-published comfortable
 * per-clip FRAME count for a resolution — how LONG one clip may be. This one is
 * how WIDE one stage-2 window (or one referenced stage-1 segment) is. A chain
 * can satisfy one and violate the other.
 */

/** The stage-2 window presets a chain request may ask for
 * (`GenerateChainRequest.stage2_window`).
 *
 * This type represents only the windows the UI lets a user PICK. The backend
 * also has a third preset, `"full_length"` (61, 61) — one stage-2 tile
 * spanning the whole clip, no seams — but it is a fixed wire value the
 * Single/Batch A2V builders always send (`modes/batch/buildA2vChainPayload.ts`),
 * never a choice, so it is deliberately NOT added here (§1-19, 2026-08-11).
 * Widening this union to 3 values would leak into places that assume exactly
 * 2: `modes/edit/useRetakeForm.ts`'s `retakeMaxWindowPx` window-size ceiling
 * (which would silently jump from 169 to 481 px) and the Chained screen's
 * stage-2 window dropdown (which would type-check accepting a value it must
 * never offer). See `api/types.ts`'s `GenerateChainRequest.stage2_window` for
 * the wire type that does include `"full_length"`. */
export type Stage2Window = "standard" | "high_resolution";

export const STAGE2_WINDOW_DEFAULT: Stage2Window = "standard";

/** Resolved `(vTile, vAdv)` per preset, in video-latent frames. Mirror of
 * `chain_math.STAGE2_WINDOW_PRESETS`.
 *
 * - `standard` (22, 18) — the frozen S2-spike layout; overlap ("のり代") 4.
 * - `high_resolution` (19, 12) — opt-in only; overlap 7. A shorter window costs
 *   ~14% fewer attention tokens per tile, which is what keeps high resolutions
 *   inside {@link CHAIN_COMFORT_TOKEN_BUDGET}; the wider overlap is what the
 *   §3-57 follow-up run added to suppress the drift the bare 19/4 window showed.
 *   It advances LESS per tile, so the same timeline gets MORE seams — hence
 *   opt-in, never the default (owner decision 2026-08-09).
 */
export const STAGE2_WINDOW_PRESETS: Readonly<Record<Stage2Window, { vTile: number; vAdv: number }>> = {
  standard: { vTile: 22, vAdv: 18 },
  high_resolution: { vTile: 19, vAdv: 12 },
};

/** The stage-2 window order the UI offers, default first. */
export const STAGE2_WINDOW_OPTIONS: readonly Stage2Window[] = ["standard", "high_resolution"];

/** Comfortable attention-token ceiling for ONE stage-2 window of a chain.
 * Mirror of `chain_math.CHAIN_COMFORT_TOKEN_BUDGET`.
 *
 * Derived from the §3-57 spill arm's measurements
 * (`outputs/stage2_window_sweep/SWEEP_RESULTS.md` §2b): at the same resolution,
 * 44,880 tokens cost +1,277MB peak VRAM and +27% wall clock against 38,760;
 * 47,840 cost +2,309MB and +32% against 36,800. 40,000 sits between the clean
 * and the spilling measurement of both pairs. Exceeding it is not an error —
 * it is a "this will get slow and memory-hungry" warning. */
export const CHAIN_COMFORT_TOKEN_BUDGET = 40000;

/** The served `config.limits.chain_comfort_token_budget`, or the mirrored
 * {@link CHAIN_COMFORT_TOKEN_BUDGET} when the server did not send a usable one.
 *
 * The budget is a served number (2026-08-12) precisely so it can be tuned per
 * machine without a rebuild, so nothing may read the raw field directly: an
 * older backend omits it entirely, and the fallback config used while the
 * server is unreachable is not the machine's real config either. Anything that
 * is not a positive finite number (missing, 0, negative, NaN) falls back — a
 * budget of 0 would otherwise put every resolution "over budget" and pin every
 * guide at 0px. */
export function resolveChainComfortBudget(published: number | null | undefined): number {
  return typeof published === "number" && Number.isFinite(published) && published > 0
    ? published
    : CHAIN_COMFORT_TOKEN_BUDGET;
}

/** The video VAE's spatial compression factor: a 32x32 pixel block is one
 * latent position, and each latent position of each window frame is one
 * attention token. */
const VAE_SPATIAL_FACTOR = 32;

/** Attention tokens ONE stage-2 window of a CHAIN spans at `width x height`.
 * Mirror of `chain_math.chain_window_tokens`:
 * `(width / 32) * (height / 32) * vTile`, integer division on each factor
 * (width/height are always multiples of 64 in practice, so the flooring only
 * matters for defensive calls). */
export function chainWindowTokens(width: number, height: number, vTile: number): number {
  return Math.floor(width / VAE_SPATIAL_FACTOR) * Math.floor(height / VAE_SPATIAL_FACTOR) * vTile;
}

/** `true` when a chain at `width x height` on `window` would exceed the
 * comfortable budget — the condition behind the Chain screen's resolution
 * warning. `budget` defaults to the mirrored {@link CHAIN_COMFORT_TOKEN_BUDGET};
 * a caller that has a served config passes
 * {@link resolveChainComfortBudget}'s result so the warning line moves with the
 * machine's own budget. */
export function isChainWindowOverBudget(
  width: number,
  height: number,
  window: Stage2Window,
  budget: number = CHAIN_COMFORT_TOKEN_BUDGET,
): boolean {
  return chainWindowTokens(width, height, STAGE2_WINDOW_PRESETS[window].vTile) > budget;
}

// ── Resolution guides drawn on the Chain screen's width/height sliders ───────
//
// ⚠ The three functions below are UI-ONLY and have NO counterpart in
// `chain_math.py`. They answer "where do we DRAW the budget on a slider?",
// which is a question the backend never asks — it only ever compares a finished
// resolution against the budget. Their absence from the Python side is
// therefore NOT a missing mirror: do not "restore" them there, and do not treat
// a diff between the two files here as a bug. The only mirrored numbers they
// consume are `STAGE2_WINDOW_PRESETS` and the budget itself.

/** The width/height grid the sliders actually step on: 64 normally, 128 while a
 * reference video / control LoRA is active (`useChainForm`'s `active.multiple`).
 * Guides are floored onto it so every guide sits on a position the slider can
 * actually reach — flooring only ever moves a guide to the safe side. */
const GUIDE_GRID_DEFAULT = 64;

function floorToGrid(value: number, grid: number): number {
  return Math.floor(value / grid) * grid;
}

/** Largest value for ONE axis that keeps a stage-2 window inside `budget` while
 * the OTHER axis stays at `other` pixels, floored onto `grid`.
 *
 * `floorGrid(32 * floor(budget / (vTile * floor(other / 32))))` — the closed
 * form of "how many 32px cells may this axis still spend". One formula serves
 * all three guides: the comfortable height opposite the comfortable width, and
 * both red per-axis maxima opposite the user's actual current value.
 *
 * Returns 0 for an `other` below one full cell (< 32px), which no slider can
 * produce but a defensive/mid-edit call could — the alternative is a division
 * by zero, and a 0 guide is simply out of every slider's range and not drawn. */
export function chainComfortAxisMax(
  other: number,
  window: Stage2Window,
  budget: number = CHAIN_COMFORT_TOKEN_BUDGET,
  grid: number = GUIDE_GRID_DEFAULT,
): number {
  const otherCells = Math.floor(other / VAE_SPATIAL_FACTOR);
  if (otherCells < 1) return 0;
  const cells = Math.floor(budget / (STAGE2_WINDOW_PRESETS[window].vTile * otherCells));
  return floorToGrid(VAE_SPATIAL_FACTOR * cells, grid);
}

/** The comfortable RECOMMENDED size for `window`: the largest roughly-16:9 pair
 * that still fits `budget`. At the default budget/grid that is 1792x1024 on
 * `standard` (39,424 tokens) and 1920x1088 on `high_resolution` (38,760) —
 * every neighbour one 64-step up is over budget.
 *
 * The width comes from solving `cells_w * cells_h * vTile = budget` at a 16:9
 * ratio (`cells * 1024` is `(32 px)²` per cell, so the square root lands in
 * pixels directly); the height is then {@link chainComfortAxisMax} of that
 * width, which is what makes the pair EXACT rather than merely near-16:9.
 *
 * ⚠ This is a recommended POINT, not a per-axis ceiling: lowering the other
 * dimension legitimately allows an axis above its own guide. The per-axis
 * ceiling is {@link chainComfortAxisMax}. */
export function chainComfortSize16x9(
  window: Stage2Window,
  budget: number = CHAIN_COMFORT_TOKEN_BUDGET,
  grid: number = GUIDE_GRID_DEFAULT,
): { width: number; height: number } {
  const cells = Math.floor(budget / STAGE2_WINDOW_PRESETS[window].vTile);
  const width = floorToGrid(Math.sqrt((cells * 1024 * 16) / 9), grid);
  return { width, height: chainComfortAxisMax(width, window, budget, grid) };
}

/** Everything the width/height sliders draw for `window` at the current
 * `width x height`. */
export interface ChainWindowBudgetMarkers {
  /** The recommended roughly-16:9 point, one guide per axis. Always present —
   * it depends on the window and the budget, never on the current values. */
  comfortWidth: number;
  comfortHeight: number;
  /** The per-axis ceiling given the OTHER axis's current value; `null` while
   * there is nothing to warn about. */
  limitWidth: number | null;
  limitHeight: number | null;
}

/** The guides for both sliders.
 *
 * An axis's red ceiling appears only once the OTHER axis has gone ABOVE its own
 * recommended point — strictly `>`, never `>=`: sitting exactly on the
 * recommended pair is the intended comfortable state and must stay free of red.
 *
 * The ceiling is computed from the other axis's ACTUAL value, not from the
 * recommended pair, so it may well land ABOVE the current value (e.g. the width
 * is past its guide while the total is still inside the budget). That is left
 * as-is on purpose rather than hidden by a special case — the copy calls it
 * "the comfortable maximum at the other dimension's current value", which is
 * exactly what it is. */
export function chainWindowBudgetMarkers(
  width: number,
  height: number,
  window: Stage2Window,
  budget: number = CHAIN_COMFORT_TOKEN_BUDGET,
  grid: number = GUIDE_GRID_DEFAULT,
): ChainWindowBudgetMarkers {
  const comfort = chainComfortSize16x9(window, budget, grid);
  return {
    comfortWidth: comfort.width,
    comfortHeight: comfort.height,
    limitWidth: height > comfort.height ? chainComfortAxisMax(height, window, budget, grid) : null,
    limitHeight: width > comfort.width ? chainComfortAxisMax(width, window, budget, grid) : null,
  };
}

/** The video VAE's temporal compression factor (`chain_math.VIDEO_TIME_FACTOR`). */
const VIDEO_TIME_FACTOR = 8;

/** Pixel frames one stage-2 window ADVANCES on `window` — the span the UI
 * renders as a duration ("how far the finishing pass moves each step"). Divide
 * by the frame rate for seconds; that is why the API field is named for the
 * geometry, not for a duration (the same preset is 6.0s at 24fps and 4.8s at
 * 30fps).
 *
 * ⚠ An ADVANCE converts as `vAdv * 8`, NOT via `px_from_v_latent` (`(n-1)*8+1`).
 * The `+1` in that formula is the causal keyframe at the START of a span, so it
 * belongs to a window's LENGTH (22 latents = 169 pixel frames) — an advance is a
 * pure delta between two window starts and carries no keyframe of its own.
 * `chain_math.compute_chain_layout` makes the same distinction: it derives
 * `audio_adv` from `v_adv * VIDEO_TIME_FACTOR` while sizing tiles with
 * `px_from_v_latent`. Getting this wrong shifts every label by ~0.3s. */
export function stage2AdvancePixelFrames(window: Stage2Window): number {
  return STAGE2_WINDOW_PRESETS[window].vAdv * VIDEO_TIME_FACTOR;
}

/** Seconds one stage-2 window advances at `frameRate`. Returns 0 for a
 * non-positive frame rate rather than dividing by zero (this feeds a live
 * readout while the user may still be mid-edit). */
export function stage2AdvanceSeconds(window: Stage2Window, frameRate: number): number {
  if (!(frameRate > 0)) return 0;
  return stage2AdvancePixelFrames(window) / frameRate;
}

/** Largest 8n+1 V2V `context_frames` that leaves stage-2 tile 0 something to
 * generate on `window`. Mirror of `chain_math.stage2_max_context_px`:
 * `px_from_v_latent(vTile - 1)` — 161 for `standard` (above the server's
 * published `v2v_context_frames_max` of 145, so it never binds) and 137 for
 * `high_resolution` (below 145, so it DOES bind and the server 422s past it). */
export function stage2MaxContextFrames(window: Stage2Window): number {
  // A LENGTH, so this one really is `px_from_v_latent` — see
  // `stage2AdvancePixelFrames` for why an advance is not.
  return (STAGE2_WINDOW_PRESETS[window].vTile - 1 - 1) * VIDEO_TIME_FACTOR + 1;
}

// 素材（末尾）: `stage2MaxEndContextFrames` was deleted here. It mirrored
// `chain_math.stage2_max_end_context_px`, i.e. "how much of the LAST stage-2
// tile may the tail freeze claim" — a question the backend stopped asking, and
// one 窓内モード (2026-08-17) makes doubly moot: the anchor is a constant 8
// frames sitting INSIDE the clip, not a window-sized band.
//
// The stage-2 window still bounds the feature, but on a different axis: the
// CLIP wants to stay within one tile, or the anchor and the frames blending
// into it land in different tiles. That length is plain `pxFromVLatent(vTile)`
// (169 / 145) — `modes/chained/chainUtils.ts` already exports it, and
// `useChainForm`'s `endSourceQualityLimitFrames` calls it there rather than
// growing a third copy of the same formula in this module.

// ── Comfortable stage-1 attention-token budget with a reference (§1-15) ──────

/** Comfortable attention-token ceiling for ONE stage-1 segment WITH an IC-LoRA
 * reference attached. Mirror of `chain_math.CHAIN_STAGE1_COMFORT_TOKEN_BUDGET`.
 *
 * A DIFFERENT axis from {@link CHAIN_COMFORT_TOKEN_BUDGET} above: that one
 * bounds ONE stage-2 tile, this one bounds ONE stage-1 segment. Stage 1 runs at
 * half resolution and denoises a whole clip in a single pass (no tiling), so
 * without a reference it is never the bottleneck — but an IC-LoRA reference is
 * patchified alongside the clip and ADDS tokens to that same single pass, which
 * is what makes stage 1 bind first on a referenced chain.
 *
 * PROVISIONAL, derived from the two measured statements in
 * `Docs/CHAIN_STAGE2_RESEARCH_NOTES.md` §6: at 1152x1536 a referenced clip tops
 * out around 361 frames (= 24,840 tokens by the formula below at scale 2), and
 * keeping 481 frames requires dropping to roughly 328 stage-1 patches
 * (= 25,010 tokens). Both land just either side of 25,000. Calibrate it on the
 * real-hardware gate (plan G4) and update this constant with the measured knee. */
export const CHAIN_STAGE1_COMFORT_TOKEN_BUDGET = 25000;

/** Attention tokens ONE stage-1 segment spans at `width x height`. Mirror of
 * `chain_math.chain_stage1_tokens` verbatim.
 *
 * `width`/`height` are the OUTPUT (full) resolution; stage 1 runs at half of
 * it, and the video VAE plus patchifier put one latent token per 32 output
 * pixels of that halved frame — hence `(width//2//32) * (height//2//32)`
 * spatial patches, times `vLatent` (the segment's stage-1 video latent
 * frames, i.e. `vLatentFrames(clipFrames[i])`).
 *
 * `refScale` is the adapter's `reference_downscale_factor` (`undefined`/`0` =
 * no IC-LoRA reference on this segment). A reference is patchified from its
 * OWN shape (`engine/pipeline/reference_video_cond.py`) and appended to the
 * same attention sequence, so it adds `refSpatial * vLatent` tokens on top:
 * `refScale=2` (the union-control adapters) contributes a quarter of the
 * spatial patches, `refScale=1` (deblur) contributes exactly as many as the
 * clip itself — i.e. it doubles the segment.
 *
 * The integer divisions are applied in that EXACT order and are NOT simplified
 * away on the "resolutions are multiples of 128 anyway" assumption: the budget
 * is also consulted while the user is still dragging a resolution slider.
 * Compare the result against {@link CHAIN_STAGE1_COMFORT_TOKEN_BUDGET}.
 * `chain_math.chain_stage1_tokens`'s own docstring notes this module mirrors it
 * verbatim, so any change there must be mirrored here. */
export function chainStage1Tokens(width: number, height: number, vLatent: number, refScale?: number): number {
  const halfW = Math.floor(width / 2);
  const halfH = Math.floor(height / 2);
  const spatial = Math.floor(halfW / VAE_SPATIAL_FACTOR) * Math.floor(halfH / VAE_SPATIAL_FACTOR);
  let tokens = spatial * vLatent;
  if (refScale) {
    const refSpatial =
      Math.floor(Math.floor(halfW / refScale) / VAE_SPATIAL_FACTOR) *
      Math.floor(Math.floor(halfH / refScale) / VAE_SPATIAL_FACTOR);
    tokens += refSpatial * vLatent;
  }
  return tokens;
}

/** Video latent-frame count for a pixel frame count (8n+1). PRIVATE local copy
 * of `chain_math.v_latent_frames` — the CHAIN-ONLY copy importable from other
 * modules is `modes/chained/chainUtils.ts`'s `vLatentFrames`, which itself
 * imports FROM this module (`STAGE2_WINDOW_DEFAULT`); importing it back here
 * would be circular, hence the duplicate one-liner. */
function vLatentFramesLocal(pixelFrames: number): number {
  return Math.floor((pixelFrames - 1) / VIDEO_TIME_FACTOR) + 1;
}

/** `true` when a chain stage-1 segment of `maxClipFrames` pixel frames at
 * `width x height` (with an optional IC-LoRA reference at `refScale`) would
 * exceed {@link CHAIN_STAGE1_COMFORT_TOKEN_BUDGET} — the condition behind the
 * Chain screen's stage-1 reference warning banner (plan F5). */
export function isChainStage1OverBudget(
  width: number,
  height: number,
  maxClipFrames: number,
  refScale?: number,
): boolean {
  return (
    chainStage1Tokens(width, height, vLatentFramesLocal(maxClipFrames), refScale) >
    CHAIN_STAGE1_COMFORT_TOKEN_BUDGET
  );
}
