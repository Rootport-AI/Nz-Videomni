/**
 * バッチi2v-long（§1-7）の行モデルと、画像フォルダのスキャン結果を行に
 * 変換する純関数。I/Oは一切しない（`fs.listFiles`の呼び出しはフック側）。
 *
 * Batch i2v-long's row model plus the pure "folder listing -> row list"
 * transform. No I/O at all — the caller does the `fs.listFiles` bridge call
 * and hands the result in.
 *
 * Deliberately much smaller than Batch A2V's `manifestMerge.ts`: there is no
 * CSV, no `Shared` image column and no per-row duration — every row is "one
 * image -> one clip chain", and everything else (size, clip layout, seed, fps)
 * comes from the Chain form's own settings. The one thing it DOES share with
 * A2V is the per-row `prompt` column (2026-07-30 owner feedback: a batch-wide
 * extra prompt was pointless, since it was the same as editing the shared
 * prompt — the prompt has to vary per image).
 */

/** A row's lifecycle state. No `Skip`: unlike A2V (whose rows can be
 * ineligible for reasons found at scan time, e.g. a non-wav file or an
 * over-cap duration), every image that survives the extension filter here is
 * runnable — the one scan-time rejection, an oversized file, lands directly
 * in `Failed` with an explanatory `error` so 🔁 can re-queue it after the
 * user shrinks/replaces the file. */
export type I2vLongStat = "Waiting" | "Generating" | "Done" | "Failed";

export interface I2vLongRow {
  /** 1-based position in the run order (= the table's `#` column). */
  queue: number;
  /** The image's bare file name (not its full path) — the run-time lookup
   * back to a real path is `joinPath(imgDir, image)`, exactly like A2V's
   * `wav` column. */
  image: string;
  /** This image's own extra prompt, combined with the Chain form's shared
   * prompt at submit time (`composeBatchPrompt`, add/replace per the panel's
   * one mode radio) — exactly like Batch A2V's `BatchRow.prompt`. Always `""`
   * from a scan: a re-scan rebuilds the row list from scratch, so hand-typed
   * row prompts are deliberately lost with it (stateless batch — there is no
   * manifest to carry them). Never `<lora:>`-tag-parsed; LoRAs can only come
   * from the shared prompt, which the Chain form already stripped into
   * `loras[]` before the template was built. */
  prompt: string;
  stat: I2vLongStat;
  /** The saved output file's name once `Done`, else `""`. */
  output: string;
  /** Failure detail once `Failed`, else `""`. */
  error: string;
}

/** One entry of `fs.listFiles`'s response (bridge contract v6,
 * `bridge/types.ts`'s `"fs.listFiles"` result element). Structurally copied
 * rather than imported so this module stays a leaf with no bridge dependency;
 * the real response type is assignable to this one. `durationSec` is always
 * `0` here (it is only populated for `.wav` files when the request asks for
 * it), kept in the shape purely so a raw listing can be passed straight in. */
export interface ScannedImageFile {
  name: string;
  path: string;
  sizeBytes: number;
  mtimeMs: number;
  durationSec: number;
}

/** Fallback extension list used when the caller's list (normally
 * `config.upload.allowed_image_extensions`, served by `GET /config`) is
 * missing or normalizes to nothing. Intentionally NARROWER than Batch A2V's
 * hard-coded `IMAGE_EXTENSIONS`: this is only a safety net for a config that
 * failed to deliver the real list, not a second source of truth. */
export const DEFAULT_IMAGE_EXTENSIONS: readonly string[] = [".png", ".jpg", ".jpeg", ".webp"];

/** Native writes partially-downloaded/being-written files as `*.tmp`; those
 * are never valid inputs, so they are dropped regardless of what the allowed
 * extension list says. */
const TMP_EXTENSION = ".tmp";

/** Lower-cased extension of `name`, INCLUDING the leading dot (`""` when the
 * name has no dot, or ends with one). */
function extensionOf(name: string): string {
  const idx = name.lastIndexOf(".");
  if (idx < 0 || idx === name.length - 1) return "";
  return name.slice(idx).toLowerCase();
}

/**
 * Normalizes an extension list into the exact form `extensionOf` produces:
 * lower-cased, dot-prefixed, blanks dropped, duplicates removed (first
 * occurrence wins). An empty result falls back to
 * {@link DEFAULT_IMAGE_EXTENSIONS}, so a broken/empty config can never make
 * the scan silently return zero rows.
 *
 * Exported for tests; the scan calls it itself.
 */
export function normalizeImageExtensions(raw: readonly string[] | null | undefined): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const item of raw ?? []) {
    if (typeof item !== "string") continue;
    const trimmed = item.trim().toLowerCase();
    if (trimmed.length === 0) continue;
    const ext = trimmed.startsWith(".") ? trimmed : `.${trimmed}`;
    // A bare "." normalizes to nothing usable.
    if (ext === ".") continue;
    if (seen.has(ext)) continue;
    seen.add(ext);
    out.push(ext);
  }
  return out.length > 0 ? out : [...DEFAULT_IMAGE_EXTENSIONS];
}

/** Human-readable megabytes for {@link formatOversizeError} (1 decimal, no
 * trailing `.0`). */
function toMegabytes(bytes: number): string {
  const mb = bytes / (1024 * 1024);
  return (Math.round(mb * 10) / 10).toString();
}

/**
 * The `error` text a scan-time oversize rejection gets. Pre-empting the
 * server here (rather than letting `POST /upload/image` 4xx at run time) is
 * the whole point: an over-limit image would otherwise fail tens of minutes
 * into a run, after every earlier row already burned GPU time.
 */
export function formatOversizeError(sizeBytes: number, maxImageBytes: number): string {
  return `Image is too large: ${toMegabytes(sizeBytes)} MB (limit ${toMegabytes(maxImageBytes)} MB).`;
}

/**
 * Sorts by file name ASCENDING, numeric-aware (`img2.png` before `img10.png`)
 * and case/accent-insensitive, with a plain code-unit tiebreak so names that
 * compare equal under that collation still get a stable, deterministic order.
 * `fs.listFiles`'s order is not guaranteed (and in practice is filesystem
 * order, not name order), so this is what makes a scan reproducible.
 */
function compareByName(a: ScannedImageFile, b: ScannedImageFile): number {
  const collated = a.name.localeCompare(b.name, "en", { numeric: true, sensitivity: "base" });
  if (collated !== 0) return collated;
  if (a.name < b.name) return -1;
  if (a.name > b.name) return 1;
  return 0;
}

/**
 * Turns a raw `fs.listFiles` listing into the batch's row list:
 * filter by extension (case-insensitive, `.tmp` always dropped) -> sort by
 * file name ascending -> number `queue` from 1.
 *
 * `maxImageBytes` (normally `config.upload.max_image_size_mb * 1024 * 1024`)
 * is optional: when given, a file over the limit still gets a row, but starts
 * in `Failed` with {@link formatOversizeError}'s text instead of `Waiting`,
 * so the user sees the problem before starting rather than mid-run. Omitted
 * (or non-finite/non-positive) disables the check entirely.
 *
 * Pure: `files` is never mutated (it is copied before sorting).
 */
export function scanImagesToRows(
  files: readonly ScannedImageFile[],
  allowedExtensions: readonly string[] | null | undefined,
  maxImageBytes?: number,
): I2vLongRow[] {
  const allowed = new Set(normalizeImageExtensions(allowedExtensions));
  const limit = typeof maxImageBytes === "number" && Number.isFinite(maxImageBytes) && maxImageBytes > 0 ? maxImageBytes : null;

  const matched = files.filter((file) => {
    const ext = extensionOf(file.name);
    if (ext === TMP_EXTENSION) return false;
    return allowed.has(ext);
  });

  const sorted = [...matched].sort(compareByName);

  return sorted.map((file, index) => {
    const oversize = limit !== null && file.sizeBytes > limit;
    return {
      queue: index + 1,
      image: file.name,
      prompt: "",
      stat: oversize ? ("Failed" as const) : ("Waiting" as const),
      output: "",
      error: oversize ? formatOversizeError(file.sizeBytes, limit) : "",
    };
  });
}

/** Joins a folder path with a file name using a backslash, unless `dir`
 * already ends with a path separator. Local copy of
 * `modes/batch/useBatchForm.ts`'s own `joinPath` (:194-197) — that one is not
 * exported, and both are tiny/pure. */
function joinPath(dir: string, name: string): string {
  const sep = dir.endsWith("\\") || dir.endsWith("/") ? "" : "\\";
  return `${dir}${sep}${name}`;
}

/** Splits a folder path into its parent folder and its own last-segment name,
 * tolerating a trailing separator. `parent === ""` means `path` had no
 * separator at all (a bare name, or a drive root like `C:`). Local copy of
 * `modes/batch/useBatchForm.ts`'s `splitDir` (:202-207) — same reason as
 * {@link joinPath}. */
function splitDir(path: string): { parent: string; base: string } {
  const trimmed = path.replace(/[\\/]+$/, "");
  const idx = Math.max(trimmed.lastIndexOf("\\"), trimmed.lastIndexOf("/"));
  if (idx === -1) return { parent: "", base: trimmed };
  return { parent: trimmed.slice(0, idx), base: trimmed.slice(idx + 1) };
}

/**
 * Auto-derives the batch's output folder: a sibling of `imgDir` named
 * `{imgDir's own folder name}_i2vlong_out`. Same construction as Batch A2V's
 * `deriveOutDir` (`modes/batch/useBatchForm.ts`:211-215, also unexported and
 * therefore duplicated), with the distinct `_i2vlong_out` suffix that keeps
 * the two features' outputs from ever landing in the same folder.
 */
export function deriveI2vLongOutDir(imgDir: string): string {
  const { parent, base } = splitDir(imgDir);
  const name = `${base}_i2vlong_out`;
  return parent ? joinPath(parent, name) : name;
}
