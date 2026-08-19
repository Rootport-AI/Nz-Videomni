import type { NativeBridge, ParamsOf, ResultOf } from "../bridge";

/**
 * The input-side symmetry of `jobs/downloadAndInsert.ts` (contract v5): where
 * that module hands a *finished* job's mp4 to the timeline
 * (`backend.downloadVideo` -> `timeline.insertMedia`), this one takes a *range
 * of the timeline* and feeds it into the generation flow — bounce it to a temp
 * file on disk (`timeline.cutoutRange` / `timeline.extractAudio`) then stream
 * that file to the backend's multipart upload endpoint (`backend.uploadFile`),
 * so the bridge call sequence lives in exactly one place.
 *
 * Kept as a near-pure orchestration function (bridge injected as the first
 * argument, no module-level singletons) so it is unit-testable against the
 * mock bridge, exactly like `downloadAndInsertVideo`.
 */

export type CutoutAndUploadPhase = "cutting-out" | "uploading";

export interface CutoutRangeAndUploadResult {
  /** The `timeline.cutoutRange` result (local file path + dimensions). */
  cutout: ResultOf<"timeline.cutoutRange">;
  /** The `backend.uploadFile` result (HTTP status + backend upload body). */
  upload: ResultOf<"backend.uploadFile">;
}

export interface ExtractAudioAndUploadResult {
  /** The `timeline.extractAudio` result (local file path + duration). */
  extract: ResultOf<"timeline.extractAudio">;
  /** The `backend.uploadFile` result (HTTP status + backend upload body). */
  upload: ResultOf<"backend.uploadFile">;
}

/**
 * Cut a video range out of the timeline and upload it to the backend as a
 * video. The cutout's local `filePath` is handed straight to
 * `backend.uploadFile` with `kind: "video"`.
 *
 * Note (same two-channel contract as `api/client.ts`): a reachable backend
 * that answers with an HTTP error status is a *normal* resolved result here
 * (inspect `upload.status`); only bridge transport failures reject.
 */
export async function cutoutRangeAndUpload(
  nativeBridge: NativeBridge,
  params: ParamsOf<"timeline.cutoutRange">,
  onPhase?: (phase: CutoutAndUploadPhase) => void,
): Promise<CutoutRangeAndUploadResult> {
  onPhase?.("cutting-out");
  const cutout = await nativeBridge.request("timeline.cutoutRange", params);

  onPhase?.("uploading");
  const upload = await nativeBridge.request("backend.uploadFile", {
    kind: "video",
    filePath: cutout.filePath,
  });

  return { cutout, upload };
}

/**
 * Extract the audio for a timeline range and upload it to the backend as
 * audio (for an audio-conditioned generation). Symmetric to
 * `cutoutRangeAndUpload` but with `kind: "audio"`.
 */
export async function extractAudioAndUpload(
  nativeBridge: NativeBridge,
  params: ParamsOf<"timeline.extractAudio">,
  onPhase?: (phase: CutoutAndUploadPhase) => void,
): Promise<ExtractAudioAndUploadResult> {
  onPhase?.("cutting-out");
  const extract = await nativeBridge.request("timeline.extractAudio", params);

  onPhase?.("uploading");
  const upload = await nativeBridge.request("backend.uploadFile", {
    kind: "audio",
    filePath: extract.filePath,
  });

  return { extract, upload };
}
