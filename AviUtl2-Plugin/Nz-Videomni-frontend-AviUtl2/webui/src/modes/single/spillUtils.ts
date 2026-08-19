import { clamp } from "./paramUtils";

/**
 * Resolves the "comfortable" `num_frames` ceiling for a given resolution
 * from `AppConfig.limits.spill_free_frames` (Docs/API_REFERENCE.md §3.2/§8):
 * a map of `"WIDTHxHEIGHT" -> frame count` beyond which generation slows
 * 2-4x (without OOMing). Looks up an exact "WxH" match first; if the
 * current resolution isn't a key (the width/height sliders move in
 * 64px steps, independent of the handful of resolutions the backend
 * happens to have measured), falls back to whichever key's pixel area is
 * closest to the current resolution's area, since the VRAM-spill slowdown
 * is driven by total pixel count rather than the exact aspect ratio.
 *
 * Returns `null` if the map is empty (or has no parseable keys at all),
 * so callers can skip the warning/tick-mark entirely rather than showing a
 * bogus threshold.
 */
export function resolveSpillFreeFrames(
  spillFreeFrames: Record<string, number>,
  width: number,
  height: number,
): number | null {
  const exactKey = `${width}x${height}`;
  const exactValue = spillFreeFrames[exactKey];
  if (exactValue !== undefined) return exactValue;

  const targetArea = width * height;
  let bestKey: string | null = null;
  let bestDiff = Infinity;

  for (const key of Object.keys(spillFreeFrames)) {
    const area = parseAreaFromKey(key);
    if (area === null) continue;
    const diff = Math.abs(area - targetArea);
    if (diff < bestDiff) {
      bestDiff = diff;
      bestKey = key;
    }
  }

  if (bestKey === null) return null;
  return spillFreeFrames[bestKey] ?? null;
}

function parseAreaFromKey(key: string): number | null {
  const match = /^(\d+)x(\d+)$/.exec(key);
  if (!match) return null;
  return Number(match[1]) * Number(match[2]);
}

// ── Smart comfort marker (2026-08-18) ────────────────────────────────────────
//
// The functions below answer the SAME question `resolveSpillFreeFrames` above
// does — "what num_frames is comfortable at this resolution?" — but with a
// closed-form inverse of the token formula instead of a 5-key nearest-area
// lookup, so it moves smoothly with every 64px resolution step instead of
// jumping between five measured points. It is used ONLY while all five
// Acceleration toggles are on (`shell/accelerationSettings.isFullAcceleration`)
// — `useGenerationForm` falls back to `resolveSpillFreeFrames` above the
// instant even one toggle is off, since the budget below was calibrated
// exclusively against that all-on configuration.
//
// ⚠ SCOPE: this is about ONE Create (`/generate`) request refining its WHOLE
// clip in ONE pass — a DIFFERENT workload from `shell/tokenBudget.ts`'s
// `CHAIN_COMFORT_TOKEN_BUDGET` (one stage-2 window of a CHAIN) and from
// `modes/edit/outpaintGeometry.ts`'s `COMFORT_TOKEN_BUDGET` (Outpainting's
// single-pass budget, but a video-VAE-encode-plus-mask workload the 44,880
// anchor below was never measured against). All three share the same token
// formula and the same "purely advisory, the server judges nothing by it"
// contract, but are three separately-calibrated numbers — never substitute
// one for another.

/** Mirror of `config.LimitsConfig.single_comfort_token_budget`
 * (`Docs/COMFORT_LIMIT_TABLE.md`), used when the server hasn't published a
 * usable one — see {@link resolveSingleComfortBudget}.
 *
 * Calibrated 2026-08-18 from a 4-stage/21-job real-device run across 3
 * resolutions x both orientations, ALL FIVE Acceleration toggles on. 44,880 is
 * the largest common comfortable value, anchored at M2 = 1920x1088, 169
 * frames (exactly 44,880 tokens by {@link singleComfortFrames}'s own formula
 * run in reverse). */
export const SINGLE_COMFORT_TOKEN_BUDGET = 44880;

/** The served `config.limits.single_comfort_token_budget`, or the mirrored
 * {@link SINGLE_COMFORT_TOKEN_BUDGET} when the server did not send a usable
 * one. Same guard as `shell/tokenBudget.ts`'s `resolveChainComfortBudget`:
 * anything that is not a positive finite number (missing, 0, negative, NaN)
 * falls back — a budget of 0 would otherwise pin every resolution's marker at
 * the floor. */
export function resolveSingleComfortBudget(published: number | null | undefined): number {
  return typeof published === "number" && Number.isFinite(published) && published > 0
    ? published
    : SINGLE_COMFORT_TOKEN_BUDGET;
}

/** The video VAE's spatial compression factor: a 32x32 pixel block is one
 * latent position, and each latent position of each frame is one attention
 * token. Deliberately a LOCAL copy of `shell/tokenBudget.ts`'s own constant of
 * the same name/value rather than an import — that module's doc comment
 * explicitly scopes it to chain code and forbids Create/single-generate code
 * from importing it, so this file keeps its own copy instead. */
const VAE_SPATIAL_FACTOR = 32;

/** The comfortable `num_frames` ceiling for a Create request at `width x
 * height`, by inverting the token formula `(width//32) * (height//32) *
 * latentFrames` against `budget`, then converting the resulting latent-frame
 * count back to pixel frames (`8n+1`) and clamping into `[minFrames,
 * maxFrames]`.
 *
 * `frames = 8 * (nLatentMax - 1) + 1` is on the 8n+1 grid by construction —
 * unlike `paramUtils.snapNumFrames`, no separate rounding step is applied
 * before the clamp (there is nothing to round: the formula only ever
 * produces grid values). `maxFrames` is assumed to already be 8n+1 itself
 * (the same assumption `snapNumFrames` makes of its own `max`).
 *
 * `width`/`height` are the GENERATION size (not `crop_output`, which is
 * irrelevant to the token count). They are read RAW — no rounding — because a
 * hand-typed width/height passes through `setWidth`/`setHeight`'s `snap:
 * false` path unrounded, so `Math.floor` here is load-bearing, not
 * defensive-only.
 *
 * Returns `null` only when `width`/`height` don't even reach one full 32px
 * cell on either axis (`cells < 1` — a hand-typed 0, negative, or empty-string
 * value, since `Number("") === 0`) — a division-by-zero guard, not a
 * real-world case: every served resolution limit is comfortably above one
 * cell.
 *
 * The `minFrames` clamp is likewise unreachable through any resolution the
 * server actually publishes limits for — `budget` is calibrated large enough
 * that `nLatentMax` never drops low enough to need it — but it is kept
 * (rather than treated as dead code and deleted) because a hand-typed
 * out-of-range resolution can still reach this function, and `clamp`'s lower
 * bound is what keeps that case from returning a negative frame count. */
export function singleComfortFrames(
  width: number,
  height: number,
  budget: number,
  minFrames: number,
  maxFrames: number,
): number | null {
  const cells = Math.floor(width / VAE_SPATIAL_FACTOR) * Math.floor(height / VAE_SPATIAL_FACTOR);
  if (cells < 1) return null;
  const nLatentMax = Math.floor(budget / cells);
  const frames = 8 * (nLatentMax - 1) + 1;
  return clamp(frames, minFrames, maxFrames);
}
