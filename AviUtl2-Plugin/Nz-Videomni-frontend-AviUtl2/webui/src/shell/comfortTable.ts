/**
 * 快適上限マーカーの配信テーブル解決（2026-08-31）——「この構成でこの解像度なら
 * 何フレームまでが快適か」の**エンジン系統ごとの**答えを、サーバが配信する
 * `AppConfig.limits.comfort_budgets` から1箇所で決めるモジュール。
 *
 * なぜ `shell/` に置くか: Create 専用の `modes/single/spillUtils.ts` と Chained
 * 専用の `shell/tokenBudget.ts` は互いに import してはならない（両者の doc が
 * 明示的に禁じている——同じ「トークン数」でも別々に較正された別の量だから）。
 * この解決器は Single の予算と Chain の予算を**同じ1行から**取り出す横断的な
 * 存在なので、どちらにも属さない `shell/` の独立モジュールとして置く。
 *
 * ⚠ このモジュールは `modes/single/spillUtils.ts` を import しない（R-7）。
 * レガシー表 `spill_free_frames` の解決はあちらの
 * `resolveSpillFreeFrames` が唯一の正で、こちらが `null` を返したとき
 * **呼び手が**そちらへ落ちる、という役割分担にしてある。
 *
 * 判定は「実効の高速化設定を**サーバ語彙**（リクエストのフィールド名）へ写し、
 * `rows` を上から順に見て `requires` の全鍵が一致した最初の行を採る」だけ。
 * 条件はフロントにハードコードされておらず、将来のモデル追加・高速化トグル
 * 追加はサーバ側の**行の追加だけ**で済む。
 */

import type { AppLimits } from "../api/types";
import { clamp } from "../modes/single/paramUtils";
import type { AccelerationRequestFields, AccelerationSettings } from "./accelerationSettings";
import { effectiveAccelerationFields } from "./accelerationSettings";
import { CHAIN_COMFORT_TOKEN_BUDGET, resolveChainComfortBudget } from "./tokenBudget";

/** Mirror of `config.LimitsConfig.single_comfort_token_budget`
 * (`Docs/COMFORT_LIMIT_TABLE.md`), used when neither a served table row nor a
 * served scalar key produced a usable budget — see
 * {@link resolveSingleComfortBudget}.
 *
 * Calibrated 2026-08-18 from a 4-stage/21-job real-device run across 3
 * resolutions x both orientations, ALL FIVE Acceleration toggles on. 44,880 is
 * the largest common comfortable value, anchored at M2 = 1920x1088, 169
 * frames (exactly 44,880 tokens by {@link comfortFramesForBudget}'s own formula
 * run in reverse). Re-confirmed 2026-08-31 at two points, and adopted
 * unconditionally by LTX 2.5 (which is comfortable at the same line whatever
 * the acceleration settings are). */
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

/** The video VAE's spatial compression factor a served profile falls back to:
 * a 32x32 pixel block is one latent position. */
const DEFAULT_SPATIAL_FACTOR = 32;

/** The pixel-frames-per-latent-frame factor a served profile falls back to
 * (the `8n+1` grid's 8). */
const DEFAULT_TEMPORAL_FACTOR = 8;

/** The five-toggles-all-on condition, written in the SAME server vocabulary a
 * served row's `requires` uses — the compatibility shim's single row (B-4).
 * This is the ONLY place the old hard-coded "smart marker needs everything on"
 * rule still lives, and it applies exclusively to a server that publishes no
 * `comfort_budgets` at all (or before `GET /models` has told us which engine
 * is loaded), so such a session behaves EXACTLY as it did before 2026-08-31. */
const COMPAT_SHIM_REQUIRES: Readonly<Record<string, string | boolean>> = Object.freeze({
  attention_backend: "sage",
  block_swap_prefetch: true,
  keep_resident: true,
  fused_gguf_dequant_kernel: true,
  vae_mode: "prune_vaed",
});

/** The row {@link resolveComfortRow} settled on, normalised and ready to use. */
export interface ResolvedComfortRow {
  /** Create (single `/generate`) token budget for this configuration. */
  singleBudget: number;
  /** Chained stage-2 window token budget for this configuration. */
  chainBudget: number;
  /** The engine's spatial compression factor (32 unless a future engine
   * publishes something else). */
  spatialFactor: number;
  /** The engine's pixel-frames-per-latent-frame factor (8, likewise). */
  temporalFactor: number;
  /** Which `rows[]` entry matched, or `-1` for the compatibility shim (no
   * served table, or the engine family is not known yet). Diagnostic — no
   * caller branches on it, but it is what the tests pin the match ORDER with. */
  rowIndex: number;
}

/** Every `requires` key must be present in `fields` AND equal — so a row that
 * demands a key this WebUI version does not know about never matches, which is
 * the safe side (fall through to the next row, ultimately to the legacy
 * `spill_free_frames` table). */
function matchesRequires(
  requires: Record<string, string | boolean>,
  fields: Required<AccelerationRequestFields>,
): boolean {
  const bag: Record<string, string | boolean | undefined> = fields;
  for (const [key, want] of Object.entries(requires)) {
    if (bag[key] !== want) return false;
  }
  return true;
}

/** A served budget, or the mirrored constant when it is not a positive finite
 * number (missing/0/negative/NaN) — the same guard the two scalar keys get,
 * applied per FIELD so one bad number never discards an otherwise good row. */
function usableBudget(published: number | undefined, fallback: number): number {
  return typeof published === "number" && Number.isFinite(published) && published > 0 ? published : fallback;
}

/**
 * The comfort-budget row that applies to the CURRENT engine + acceleration
 * configuration, or `null` when there is none — in which case the caller falls
 * back to the legacy `AppLimits.spill_free_frames` lookup
 * (`modes/single/spillUtils.resolveSpillFreeFrames` for Create,
 * `shell/tokenBudget.resolveChainComfortBudget` for Chained).
 *
 * **`null` is a normal, expected outcome, not an error.** LTX 2.3's DEFAULT
 * configuration deliberately has no row: its plain VAE decoder makes the
 * comfortable boundary non-monotone in token count (the boundaries line up
 * with the decoder's chunk-count steps 7→8 / 4→5 / 2→3), so no single token
 * line can describe it and the five measured `spill_free_frames` points are
 * the truth there. Adding a `requires: {}` row to `ltx` to "make every
 * configuration smart" would be wrong.
 *
 * Order of resolution:
 *  1. No served table at all, OR `engineFamily` unknown (`undefined`/`""` —
 *     `GET /models` has not landed, or an offline session): the
 *     {@link COMPAT_SHIM_REQUIRES} shim, whose budgets come from the two
 *     legacy scalar keys. Matching yields `rowIndex: -1`; not matching yields
 *     `null`. Either way the session behaves exactly as it did before this
 *     table existed — no startup flicker, no offline regression.
 *  2. The table has no profile for this engine family: `null`.
 *  3. Otherwise the first `rows[]` entry whose `requires` all match.
 *
 * `acceleration` MUST already be the EFFECTIVE settings object (see
 * `effectiveAccelerationFields`). The field bag is built ONCE here rather than
 * by the caller, precisely so it never ends up in a React dependency array
 * (it is a fresh object every call — R-9).
 */
export function resolveComfortRow(
  limits: AppLimits,
  engineFamily: string | undefined,
  acceleration: AccelerationSettings,
  sageAvailable: boolean | null,
): ResolvedComfortRow | null {
  const fields = effectiveAccelerationFields(acceleration, sageAvailable);
  const table = limits.comfort_budgets;

  if (!table || engineFamily === undefined || engineFamily === "") {
    if (!matchesRequires(COMPAT_SHIM_REQUIRES, fields)) return null;
    return {
      singleBudget: resolveSingleComfortBudget(limits.single_comfort_token_budget),
      chainBudget: resolveChainComfortBudget(limits.chain_comfort_token_budget),
      spatialFactor: DEFAULT_SPATIAL_FACTOR,
      temporalFactor: DEFAULT_TEMPORAL_FACTOR,
      rowIndex: -1,
    };
  }

  const profile = table[engineFamily];
  if (!profile) return null;

  const rows = profile.rows ?? [];
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    if (!row) continue;
    if (!matchesRequires(row.requires ?? {}, fields)) continue;
    const spatialFactor = profile.spatial_factor;
    const temporalFactor = profile.temporal_factor;
    return {
      singleBudget: usableBudget(row.single_budget, SINGLE_COMFORT_TOKEN_BUDGET),
      chainBudget: usableBudget(row.chain_budget, CHAIN_COMFORT_TOKEN_BUDGET),
      // A nonsensical factor is normalised to the default rather than rejected
      // (there is no `tf <= 0` special case any more — an invalid factor is
      // simply treated as "the engine didn't say", per the owner's decision).
      spatialFactor: typeof spatialFactor === "number" && spatialFactor >= 1 ? spatialFactor : DEFAULT_SPATIAL_FACTOR,
      temporalFactor:
        typeof temporalFactor === "number" && temporalFactor >= 1 ? temporalFactor : DEFAULT_TEMPORAL_FACTOR,
      rowIndex: index,
    };
  }

  return null;
}

/** The comfortable `num_frames` ceiling at `width x height`, by inverting the
 * token formula `(width//sf) * (height//sf) * latentFrames` against `budget`,
 * then converting the resulting latent-frame count back to pixel frames
 * (`tf*(n-1)+1`) and clamping into `[minFrames, maxFrames]`.
 *
 * Moved here from `modes/single/spillUtils.singleComfortFrames` on 2026-08-31,
 * with the two engine factors (`spatialFactor`/`temporalFactor`) promoted from
 * file constants to arguments so a future engine with a different VAE geometry
 * needs a served row, not a code change. Both come from
 * {@link ResolvedComfortRow} and are already normalised there.
 *
 * `frames = tf * (nLatentMax - 1) + 1` is on the `8n+1` grid by construction at
 * the current `tf = 8` — unlike `paramUtils.snapNumFrames`, no separate
 * rounding step is applied before the clamp (there is nothing to round: the
 * formula only ever produces grid values). `maxFrames` is assumed to already be
 * on that grid itself (the same assumption `snapNumFrames` makes of its own
 * `max`).
 *
 * `width`/`height` are the GENERATION size (not `crop_output`, which is
 * irrelevant to the token count). They are read RAW — no rounding — because a
 * hand-typed width/height passes through `setWidth`/`setHeight`'s `snap: false`
 * path unrounded, so `Math.floor` here is load-bearing, not defensive-only.
 *
 * Returns `null` only when `width`/`height` don't even reach one full cell on
 * either axis (`cells < 1` — a hand-typed 0, negative, or empty-string value,
 * since `Number("") === 0`) — a division-by-zero guard, not a real-world case:
 * every served resolution limit is comfortably above one cell. The caller
 * treats that `null` exactly like "no row matched" and falls back to the legacy
 * table, so a blank width box never blanks the marker.
 *
 * The `minFrames` clamp is likewise unreachable through any resolution the
 * server actually publishes limits for — `budget` is calibrated large enough
 * that `nLatentMax` never drops low enough to need it — but it is kept (rather
 * than treated as dead code and deleted) because a hand-typed out-of-range
 * resolution can still reach this function, and `clamp`'s lower bound is what
 * keeps that case from returning a negative frame count. */
export function comfortFramesForBudget(
  width: number,
  height: number,
  budget: number,
  minFrames: number,
  maxFrames: number,
  spatialFactor: number,
  temporalFactor: number,
): number | null {
  const cells = Math.floor(width / spatialFactor) * Math.floor(height / spatialFactor);
  if (cells < 1) return null;
  const nLatentMax = Math.floor(budget / cells);
  const frames = temporalFactor * (nLatentMax - 1) + 1;
  return clamp(frames, minFrames, maxFrames);
}
