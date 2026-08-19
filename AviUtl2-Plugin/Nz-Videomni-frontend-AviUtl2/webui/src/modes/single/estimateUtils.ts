/**
 * Rough duration estimate for the "Est. ~X min" label that sits next to the
 * Generate button (Mock/AVIUTL2_DESIGN_BRIEF.md §11 / §4-2: "生成ボタンに所要
 * 時間の見積りを常時表示"). Deliberately a *linear* model — proportional to
 * (width*height)*num_frames, per the task brief — calibrated against a
 * single reference point from Docs/API_REFERENCE.md §8's performance table
 * (720p/1280x768, ~121 frames = 5s@24fps, ~2.8 min).
 *
 * This will not match real generation times at other resolutions (the real
 * relationship is super-linear, especially past the spill-free threshold —
 * see `spillUtils.ts`) — the point of this estimate is to give the user
 * *some* signal before they commit to a multi-minute wait, not to be a
 * precise predictor. That tradeoff is explicit in the task brief ("mockでは
 * 実測と合わないが表示自体が目的").
 */
const CALIBRATION = {
  width: 1280,
  height: 768,
  numFrames: 121, // 5s @ 24fps, snapped to the nearest 8n+1
  seconds: 168, // ~2.8 min
};

const SECONDS_PER_PIXEL_FRAME = CALIBRATION.seconds / (CALIBRATION.width * CALIBRATION.height * CALIBRATION.numFrames);

/** Estimated wall-clock generation time in seconds, linear in pixel area
 * (width*height) and num_frames. */
export function estimateGenerationSeconds(width: number, height: number, numFrames: number): number {
  if (width <= 0 || height <= 0 || numFrames <= 0) return 0;
  return SECONDS_PER_PIXEL_FRAME * width * height * numFrames;
}

/** Formats an estimate as "~Ns" under a minute, otherwise "~X min" (one
 * decimal place below 10 minutes, whole minutes at/above 10). */
export function formatEstimate(seconds: number): string {
  if (seconds < 60) return `~${Math.max(1, Math.round(seconds))}s`;
  const minutes = seconds / 60;
  const rounded = minutes < 10 ? Math.round(minutes * 10) / 10 : Math.round(minutes);
  return `~${rounded} min`;
}
