/**
 * 素材（末尾）系統E「末尾合わせ」の純関数群（2026-08-17 窓内モード改修）。
 *
 * The pure arithmetic behind the END-source flow's placement question, kept free
 * of React/i18n/bridge so both consumers can share ONE copy:
 *
 *  - `shell/AppShell.tsx` Step 8 — the right-click reservation, which has only a
 *    timeline selection to go on;
 *  - `modes/chained/ChainedScreen.tsx` — the Generate-time re-place, which has
 *    the form's confirmed output length.
 *
 * ## 窓内モード: the output does NOT get longer
 *
 * The backend freezes the LAST {@link END_SOURCE_CONTEXT_FRAMES} frames OF THE
 * CLIP ITSELF onto the head of the material — the anchor lives INSIDE the
 * generated window rather than being appended after it. So there is no "band"
 * length to derive and no per-material arithmetic left: the delivered file is
 * exactly as long as the clip settings say, and the frozen tail is a fixed 8
 * frames whether the material is a still or a two-hour film.
 *
 * That is why v2's `END_BAND_STEP`/`END_BAND_MAX_FRAMES`/
 * `endBandFromEffectiveFrames`/`endSourceBandFrames` are gone: every one of them
 * existed to answer "how many frames does THIS material contribute", a question
 * the window-internal mode does not ask.
 *
 * 逆順Chained (2026-08-18, second stage): a chain of 2+ clips ending on
 * material (`chain_math`'s `reverse` mode) freezes the SAME 8 frames onto the
 * SAME material, still carved out of the LAST clip's own tail rather than
 * appended — nothing here changes with the clip count. Every function below is
 * therefore verbatim-correct for a multi-clip chain too.
 *
 * ## Why "tail alignment" still needs its own module
 *
 * The provisional object for 「これで終わる動画を作る」 has to be placed so its
 * frozen TAIL lands on the material the user right-clicked — i.e. the object
 * starts `出力長 − 8` frames BEFORE the material. Native knows no such
 * placement系統 (`bridge_core.cpp`'s `ResolveProvisionalPlacement` has A/B/C
 * only), so the shift is computed HERE and handed to native as an ordinary
 * `"B"` head-aligned placement with an already-shifted `materialFrameStart` —
 * the exact same discipline 系統D (Retake) uses. That is what keeps this改修
 * free of any C++ change (`timeline/menuRouting.ts`'s `MenuPlacement` doc).
 */

/**
 * The anchor: how many frames at the END of the generated clip are frozen onto
 * the START of the end material. A CONSTANT, for both images and videos.
 *
 * 8 is the API's own `end_context_frames` minimum and the length the 2026-08-16
 * real-device comparison settled on: longer anchors were what produced the
 * crossfade-into-the-material artefact the feature was withdrawn for, because
 * they forced the model to spend a large part of the window reproducing footage
 * it was also being asked to arrive at. At 8 the result reaches the material
 * cleanly. The UI therefore always sends `context_frames: 8` explicitly rather
 * than letting the server's default decide.
 */
export const END_SOURCE_CONTEXT_FRAMES = 8;

/**
 * The shortest end-source VIDEO the UI accepts, in frames AT THE GENERATION
 * FRAME RATE: the anchor's 8 plus the causal VAE's 1-frame keyframe primer, i.e.
 * the true technical floor and nothing more.
 *
 * v2 guarded at 25 (≈1 second) because the anchor was derived from the
 * material's own length and a short video produced a uselessly short band. With
 * a fixed 8-frame anchor a 9-frame material supplies exactly what the server
 * reads, so refusing anything longer would be refusing footage that works.
 *
 * Two callers reach this floor through a conversion biased by one frame
 * (`framesAtGenFps`, and `menuSelection.ts`'s duration guard), which makes the
 * EFFECTIVE floor 10 frames on those paths. That is the same conservative bias
 * every other length gate on this screen carries and is left in place
 * deliberately: one frame of slack at a 9-frame floor is not a limitation a user
 * can run into on purpose.
 */
export const END_SOURCE_MIN_FRAMES = 9;

/**
 * How many frames of a `durationSec`-long piece of material survive the
 * conversion to `genFps`, biased CONSERVATIVELY by one frame.
 *
 * The `− 1` mirrors the same one-frame safety margin the source-video length
 * gate has always used (`useChainForm`'s `sourceVideoTooShortForContext`): the
 * server counts frames through a double rounding whose source frame rate the
 * WebUI does not know, so an exact match is not reproducible here and the
 * boundary is biased to the blocking side. Returns `0` for a non-positive
 * duration / frame rate rather than a negative count.
 */
export function framesAtGenFps(durationSec: number, genFps: number): number {
  if (!(durationSec > 0) || !(genFps > 0)) return 0;
  return Math.max(0, Math.floor(durationSec * genFps) - 1);
}

export interface TailAlignedStartFrameOpts {
  /** The right-clicked material's own first frame, in PROJECT frames. */
  materialFrameStart: number;
  /** The part of the output that does NOT overlap the material — `出力長 − 8`,
   * in GENERATION (pixel) frames. The frozen tail is deliberately excluded: it
   * is the part that lands ON the material. */
  leadPixelFrames: number;
  /** The generation frame rate the output will be produced at. */
  genFps: number;
  /** The project's `rate`/`scale` (the timeline's own frame rate), straight off
   * the selection snapshot. */
  projectRate: number;
  projectScale: number;
}

/**
 * Where a 「これで終わる動画を作る」 provisional object should START, in PROJECT
 * frames, so that its frozen tail lands exactly on the material's head.
 *
 * ```
 * start = max(0, materialFrameStart − ProjectFramesForPixels(leadPixelFrames))
 * ProjectFramesForPixels(p) = floor(p × projectFps / genFps + 0.5)   (ties UP)
 * ```
 *
 * The conversion is a VERBATIM mirror of native's
 * `timeline_math.cpp::ProjectFramesForPixels` — including its round-HALF-UP tie
 * rule, which is `Math.floor(v + 0.5)` and NOT `Math.round` on negatives (the
 * input here is never negative, but keeping the identical expression is what
 * makes the mirror checkable line by line). Native uses that very function to
 * size the placeholder, so using anything else here would leave the ribbon's
 * head and tail computed on two different rounding rules.
 *
 * Two deliberate degradations, both "place it head-aligned instead of nowhere":
 *  - an unresolvable frame rate (a project without `rate`/`scale`, or a
 *    non-positive `genFps`) returns `materialFrameStart` unchanged, i.e. the
 *    plain 系統B head-aligned position;
 *  - a computed start before frame 0 is CLAMPED to 0 (owner confirmation point C
 *    of the v2 plan: no note — the user can see where it landed).
 */
export function tailAlignedStartFrame({
  materialFrameStart,
  leadPixelFrames,
  genFps,
  projectRate,
  projectScale,
}: TailAlignedStartFrameOpts): number {
  const projectFps = projectScale > 0 ? projectRate / projectScale : 0;
  if (!(genFps > 0) || !(projectFps > 0)) return materialFrameStart;
  if (!(leadPixelFrames > 0)) return materialFrameStart;
  const lead = Math.floor((leadPixelFrames * projectFps) / genFps + 0.5);
  return Math.max(0, materialFrameStart - lead);
}
