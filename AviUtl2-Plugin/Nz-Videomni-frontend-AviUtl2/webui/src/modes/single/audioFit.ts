/**
 * Single-clip A2V's audio fit — the `/generate` route (an engine that
 * publishes `constraints.source_audio`, MiniMax H3).
 *
 * The rule is plain time: the clip needs `numFrames / fps` seconds of audio,
 * and the server refuses a shorter one (422 `SOURCE_AUDIO_TOO_SHORT`). The
 * chain route keeps `chained/chainUtils.ts`'s audio-latent arithmetic (LTX:
 * 25 latents/s, 8n+1), which does not apply here.
 *
 * Pure functions only.
 */

/** Absorbs float noise in `sec * fps` (e.g. `0.29 * 100 = 28.999…`). Shared
 * by both functions so a suggested count always passes the precheck. */
const FRAME_EPSILON = 1e-6;

/** The family's frame grid (`step * n + offset`) and the form's bounds. */
export interface SingleAudioFrameGrid {
  step: number;
  offset: number;
  min: number;
  max: number;
}

/** The largest frame count on the grid that the audio covers
 * (`<= floor(sec * fps)`), within `[min, max]`. Audio shorter than `min`
 * frames gives `min` — {@link singleAudioPrecheck} then flags it.
 * H3 (17n+5, 24 fps): 3.886 s → 93 frames' worth → 90. */
export function suggestFramesForSingleAudio(sec: number, fps: number, grid: SingleAudioFrameGrid): number {
  const covered = Math.min(Math.floor(sec * fps + FRAME_EPSILON), grid.max);
  const n = Math.floor((covered - grid.offset) / grid.step);
  return Math.max(grid.min, n * grid.step + grid.offset);
}

/** Whether `sec` seconds of audio cover `numFrames` at `fps`
 * (`numFrames / fps <= sec`). */
export function singleAudioPrecheck(sec: number, numFrames: number, fps: number): boolean {
  return numFrames <= sec * fps + FRAME_EPSILON;
}
