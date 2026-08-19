/**
 * Pure routing for Chain's unified source-input picker (task brief
 * "Chainのソース入力欄一本化"): `ui.pickFile({kind:"imageOrVideo"})` returns one
 * file regardless of type, and `useChainForm.pickSource` uses this to decide
 * whether it's an image (start-frame / scratch mode) or a video (source
 * video / V2V mode).
 *
 * The extension lists below MUST stay in sync with native's combined
 * "imageOrVideo" filter (`native/src/bridge_core.cpp`'s `PickFileFilter`) —
 * that filter's extensions are exactly this file's `IMAGE_EXTENSIONS` union
 * `VIDEO_EXTENSIONS`, in the same order.
 */

const IMAGE_EXTENSIONS: ReadonlySet<string> = new Set(["png", "jpg", "jpeg", "webp"]);
const VIDEO_EXTENSIONS: ReadonlySet<string> = new Set(["mp4", "mov", "webm", "mkv"]);

/** Routes a picked file name to `"image"` or `"video"` by its extension
 * (case-insensitive). Returns `null` for an unrecognised or missing
 * extension — callers surface that as an "unsupported file type" error
 * rather than guessing. */
export function routeSourceByExtension(fileName: string): "image" | "video" | null {
  const dotIdx = fileName.lastIndexOf(".");
  if (dotIdx < 0 || dotIdx === fileName.length - 1) return null;
  const ext = fileName.slice(dotIdx + 1).toLowerCase();
  if (IMAGE_EXTENSIONS.has(ext)) return "image";
  if (VIDEO_EXTENSIONS.has(ext)) return "video";
  return null;
}
