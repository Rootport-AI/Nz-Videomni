/**
 * MiniMax H3's DiT sequence length (S, the total number of rows the denoiser
 * sees) for a Single `/generate`, and the comfort marker derived from it
 * (ledger §1-119, 2026-11-07).
 *
 * H3's comfort boundary is an out-of-VRAM failure, and the recalibration
 * (CLOSED 1-274 / 1-275) found it separated by ONE line in S, whatever the
 * resolution. The server publishes that line as the profile's
 * `single_rows_budget` (`shell/comfortTable.ts`'s `resolveSequenceRowsBudget`);
 * this module computes S from the form's composition — size, frame count,
 * keyframe count and the Reference cards — so the marker follows the inputs.
 *
 * The formula is a mirror of the backend's (`engine_h3/constants.py`,
 * `engine_h3/reference.py`, `engine_h3/text_encoder.py`); the canonical
 * description is the backend's `Docs/COMFORT_LIMIT_TABLE.md` §17.3 / §18.2
 * (the formula) and §1.1 (the representative-value counting).
 * Reference inputs are counted by REPRESENTATIVE values instead of their
 * measured sizes: a picture as the generation grid (exact for a large image,
 * an upper bound for a small one), a video as 1,008 rows per latent frame
 * (exact for a 16:9 clip; an extreme aspect ratio can need up to about 4%
 * more), an audio by its seconds. The prompt's own tokens are not counted —
 * the line already includes the 30 tokens of the calibration prompt.
 *
 * Synchronous and pure on purpose: the right-click seeds and the A2V cap read
 * the marker as a plain value.
 */

/** Pixels per latent cell on each axis (`engine_h3/constants.py`). */
const H3_CELL = 32;
/** The `num_frames` grid `17n+5` — its step and offset. Same values as the
 * served `constraints.num_frames` (change both together). */
const H3_CLIP = 17;
const H3_OFFSET = 5;
/** H3's fixed frame rate, and the generated audio's latent rate per second.
 * The 24 fps is the served `constraints.frame_rate` (change both together). */
const H3_FPS = 24;
const H3_AUDIO_LATENTS_PER_SEC = 40;
/** Text-encoder rows a picture adds beyond its cell count. */
const H3_IMAGE_TEXT_EXTRA = 8;
/** DiT rows per latent frame of a reference video (the 768x1344 canvas grid;
 * exact for a 16:9 clip — see the file comment). */
const H3_REF_VIDEO_ROWS_PER_LATENT = 1008;
/** The longest reference video the server reads, in frames (15 s on the grid). */
const H3_REF_VIDEO_MAX_FRAMES = 362;
/** The longest reference clip, in seconds; also the length assumed for a clip
 * whose length is unknown. The served `constraints.reference` total seconds
 * (change both together). */
const H3_REF_CLIP_MAX_SEC = 15;
/** A reference video goes to the text encoder one frame in this many. */
const H3_REF_TE_FRAME_STRIDE = 12;
/** Text-encoder rows of a reference's label (`<Video 1>` …) — an upper bound. */
const H3_REF_LABEL_TOKENS = 6;
/** Text-encoder rows per encoded frame pair of a reference video beyond the
 * 1,008 image rows (the timestamp tokens; never below the measurement). */
const H3_REF_VIDEO_PAIR_TEXT_EXTRA = 9;

export type H3ReferenceKind = "image" | "video" | "audio";

/** One Reference card that rides the generation. */
export interface H3SequenceReference {
  kind: H3ReferenceKind;
  /** Seconds actually used (after truncation); `null` for an image, or when the
   * length is unknown (counted as {@link H3_REF_CLIP_MAX_SEC}). */
  seconds: number | null;
}

/** The conditioning half of the composition: keyframes and Reference cards. */
export interface H3SequenceShape {
  /** Keyframe cards the generation sends (or is still uploading). */
  keyframeCount: number;
  references: readonly H3SequenceReference[];
}

export interface H3SequenceInput extends H3SequenceShape {
  /** The GENERATION size, read raw (a hand-typed value is floored per cell). */
  width: number;
  height: number;
}

/** No keyframes and no Reference cards — the right-click seeds' composition
 * apart from their keyframe count, and the form's default. */
export const H3_NO_CONDITIONING: H3SequenceShape = Object.freeze({
  keyframeCount: 0,
  references: Object.freeze([]) as readonly H3SequenceReference[],
});

/** S broken into its parts (rows). `text` is the text-encoder output without
 * the prompt's own tokens. */
export interface H3SequenceRows {
  video: number;
  cond: number;
  audio: number;
  ref: number;
  text: number;
  total: number;
}

/** Latent frames of `frames` pixel frames on the 17n+5 grid. */
function latentFrames(frames: number): number {
  return ((frames - H3_OFFSET) / H3_CLIP) * 5 + 2;
}

/** The largest grid value not above `frames` (5 at the least). */
function floorToH3Grid(frames: number): number {
  if (frames <= H3_OFFSET) return H3_OFFSET;
  return H3_OFFSET + Math.floor((frames - H3_OFFSET) / H3_CLIP) * H3_CLIP;
}

/** The smallest grid value not below `frames` (5 at the least). */
function ceilToH3Grid(frames: number): number {
  if (frames <= H3_OFFSET) return H3_OFFSET;
  return H3_OFFSET + Math.ceil((frames - H3_OFFSET) / H3_CLIP) * H3_CLIP;
}

/** Mirror of `engine_h3/reference.py`'s `frames_for_seconds`: the clip's frame
 * count at 24 fps, rounded UP onto the grid. `Math.round` rounds .5 up where
 * Python rounds to even, which only ever makes this side larger (the safe
 * side). The server reads an untrimmed clip's frame count rounded DOWN, so an
 * untrimmed clip whose length is off the grid can be counted one step higher
 * here — an upper bound. */
function framesForSeconds(seconds: number): number {
  return ceilToH3Grid(Math.round(seconds * H3_FPS));
}

/**
 * S for `input` at `numFrames`, or `null` when `numFrames` is not on the 17n+5
 * grid or the size does not reach one cell on either axis.
 *
 * - video: lat(F) x cells; cond: K x cells; generated audio: 2 x round(F x 40 / 24)
 *   (the fractions are .0 / .333 / .667 only, so `Math.round` agrees with Python).
 * - a picture: cells, text cells + 8; a video: lat(Fr) x 1,008 with
 *   Fr = min(F, frames of its seconds, 362), text 6 + pairs x (1,008 + 9) with
 *   pairs = ceil(ceil(Fr / 12) / 2); an audio: 2 x ceil(min(seconds, 15) x 40),
 *   text 6.
 * - keyframes reach the text encoder only while no Reference card rides
 *   (`pipeline_h3.py`), counted as min(K, 2) x (cells + 8) without telling the
 *   first and last apart (an upper bound).
 */
export function h3SequenceRows(input: H3SequenceInput, numFrames: number): H3SequenceRows | null {
  if (!Number.isInteger(numFrames) || numFrames < H3_OFFSET || (numFrames - H3_OFFSET) % H3_CLIP !== 0) return null;
  const cells = Math.floor(input.width / H3_CELL) * Math.floor(input.height / H3_CELL);
  if (!(cells >= 1)) return null;

  const video = latentFrames(numFrames) * cells;
  const cond = input.keyframeCount * cells;
  const audio = 2 * Math.round((numFrames * H3_AUDIO_LATENTS_PER_SEC) / H3_FPS);

  let ref = 0;
  let text = input.references.length === 0 ? Math.min(input.keyframeCount, 2) * (cells + H3_IMAGE_TEXT_EXTRA) : 0;
  for (const reference of input.references) {
    const seconds = Math.min(reference.seconds ?? H3_REF_CLIP_MAX_SEC, H3_REF_CLIP_MAX_SEC);
    if (reference.kind === "image") {
      ref += cells;
      text += cells + H3_IMAGE_TEXT_EXTRA;
    } else if (reference.kind === "video") {
      const refFrames = floorToH3Grid(
        Math.min(numFrames, framesForSeconds(reference.seconds ?? H3_REF_CLIP_MAX_SEC), H3_REF_VIDEO_MAX_FRAMES),
      );
      ref += latentFrames(refFrames) * H3_REF_VIDEO_ROWS_PER_LATENT;
      const pairs = Math.ceil(Math.ceil(refFrames / H3_REF_TE_FRAME_STRIDE) / 2);
      text += H3_REF_LABEL_TOKENS + pairs * (H3_REF_VIDEO_ROWS_PER_LATENT + H3_REF_VIDEO_PAIR_TEXT_EXTRA);
    } else {
      ref += 2 * Math.ceil(seconds * H3_AUDIO_LATENTS_PER_SEC);
      text += H3_REF_LABEL_TOKENS;
    }
  }

  return { video, cond, audio, ref, text, total: video + cond + audio + ref + text };
}

/**
 * The comfort marker: walking down from the largest grid value not above
 * `maxFrames` in steps of 17, the first frame count whose S fits `budget`.
 * When none does, the smallest grid value not below `minFrames`. `null` when
 * the size does not reach one cell on either axis.
 */
export function h3ComfortFrames(
  input: H3SequenceInput,
  budget: number,
  minFrames: number,
  maxFrames: number,
): number | null {
  const floor = ceilToH3Grid(minFrames);
  if (h3SequenceRows(input, floor) === null) return null;
  for (let frames = floorToH3Grid(maxFrames); frames >= floor; frames -= H3_CLIP) {
    const rows = h3SequenceRows(input, frames);
    if (rows !== null && rows.total <= budget) return frames;
  }
  return floor;
}
