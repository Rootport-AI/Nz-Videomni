/**
 * The α-version resolution decision seam (Docs/TIMELINE_ALPHA_REQUIREMENTS.md
 * §7-5, owner-approved hybrid). Right-click generation must derive
 * width/height from the selected input material (falling back to the
 * project's own resolution) and round/clamp them to the backend's shape
 * constraints — this function is the single place that does that.
 *
 * fps/num_frames are deliberately *not* decided here: §7-5 scopes this seam to
 * width/height only (the "match output resolution to input" plan), while
 * duration/frame-rate stay on the existing form defaults + user adjustment
 * (§7-3-B option 3). Folding them in would need the same seam treatment
 * (`getEditInfo`/selection length -> num_frames rounding to 8n+1) but that is
 * a separate, not-yet-decided design question — see §7-3-B options 1/2.
 *
 * β's automatic best-resolution + letterbox/reflect-pad plumbing
 * (`resize_and_reflect_pad`/`align_resolution`, §7-5 "β版でやること") is meant
 * to plug into this exact seam, so keep the signature stable.
 */

import { ceilToMultiple, floorToMultiple } from "../modes/single/paramUtils";

/** The subset of `AppConfig.limits` (api/types.ts) this seam needs, renamed to
 * the camelCase min/max pairs the derivation logic reads naturally. */
export interface GenerationParamLimits {
  minWidth: number;
  maxWidth: number;
  minHeight: number;
  maxHeight: number;
}

export interface DeriveParamsInput {
  /** The selected material's actual resolution, when known. Takes priority
   * over the project resolution when both dimensions are positive. */
  mediaWidth?: number | null;
  mediaHeight?: number | null;
  /** `getEditInfo` fallback: the AviUtl2 project's resolution. */
  projectWidth: number;
  projectHeight: number;
  /** true for the IC-LoRA (`referenceVideo`) flow, which requires width/height
   * to be multiples of 128 instead of the general-mode 64
   * (Docs/TIMELINE_ALPHA_REQUIREMENTS.md §2). */
  isICLora: boolean;
  limits: GenerationParamLimits;
}

export interface DerivedGenerationParams {
  width: number;
  height: number;
  /** The multiple width/height were rounded to (128 for IC-LoRA, else 64). */
  multiple: number;
  /** Which input the resolution was derived from. */
  source: "media" | "project";
  /** True if either dimension had to be pulled in from its naively-rounded
   * value to satisfy `limits` (min or max). */
  clamped: boolean;
  /** Human-readable (Japanese) reasons for the chosen value, in the spirit of
   * LTX's "surface the reasoning behind an adjustment" good practice. */
  notes: string[];
}

/**
 * Aligns a configured minimum up to the nearest multiple, so the derived
 * value never requests a size the server would reject as too small. Reuses
 * `ceilToMultiple`'s rounding; the wide-open [min, +Infinity] range means this
 * call only ever rounds, it never itself clamps away from that rounded value.
 */
function alignedMinBound(min: number, multiple: number): number {
  return ceilToMultiple(min, multiple, min, Number.MAX_SAFE_INTEGER);
}

/**
 * Aligns a configured maximum down to the nearest multiple. This is the case
 * called out by the task brief: IC-LoRA's `max_height` (1088) is not a
 * multiple of 128, so this floors it to 1024 rather than letting a rounded
 * value sneak past the real server-enforced ceiling.
 */
function alignedMaxBound(max: number, multiple: number): number {
  return floorToMultiple(max, multiple, 0, max);
}

/**
 * Derives the width/height to send for a right-click generation, per the
 * §7-5 α plan: prefer the selected material's real resolution, fall back to
 * the project resolution, round to the mode's required multiple (64, or 128
 * for IC-LoRA), and clamp into the server's actual limits (themselves aligned
 * to that multiple first, since `max_height` is not always a clean multiple).
 */
export function deriveGenerationParams(input: DeriveParamsInput): DerivedGenerationParams {
  const { mediaWidth, mediaHeight, projectWidth, projectHeight, isICLora, limits } = input;
  const multiple = isICLora ? 128 : 64;
  const notes: string[] = [];

  let rawWidth: number;
  let rawHeight: number;
  let source: "media" | "project";
  if (mediaWidth != null && mediaWidth > 0 && mediaHeight != null && mediaHeight > 0) {
    rawWidth = mediaWidth;
    rawHeight = mediaHeight;
    source = "media";
    notes.push(`選択素材の解像度 ${mediaWidth}×${mediaHeight} に合わせて決定`);
  } else {
    rawWidth = projectWidth;
    rawHeight = projectHeight;
    source = "project";
    notes.push(`選択素材の解像度が無いため、プロジェクト設定 ${projectWidth}×${projectHeight} に合わせて決定`);
  }

  const minWidth = alignedMinBound(limits.minWidth, multiple);
  const maxWidth = alignedMaxBound(limits.maxWidth, multiple);
  const minHeight = alignedMinBound(limits.minHeight, multiple);
  const maxHeight = alignedMaxBound(limits.maxHeight, multiple);

  // The naive (unclamped) rounding, used only to detect whether the final
  // clamp actually changed anything.
  const naiveWidth = Math.ceil(rawWidth / multiple) * multiple;
  const naiveHeight = Math.ceil(rawHeight / multiple) * multiple;

  const width = ceilToMultiple(rawWidth, multiple, minWidth, maxWidth);
  const height = ceilToMultiple(rawHeight, multiple, minHeight, maxHeight);

  if (isICLora) {
    notes.push(`IC-LoRAのため${multiple}の倍数に丸め`);
  } else if (width !== rawWidth || height !== rawHeight) {
    notes.push(`${multiple}の倍数に丸め`);
  }

  let clamped = false;
  if (naiveWidth > maxWidth || naiveHeight > maxHeight) {
    clamped = true;
    notes.push(`上限 ${maxWidth}×${maxHeight} にクランプ`);
  }
  if (naiveWidth < minWidth || naiveHeight < minHeight) {
    clamped = true;
    notes.push(`下限 ${minWidth}×${minHeight} にクランプ`);
  }

  return { width, height, multiple, source, clamped, notes };
}
