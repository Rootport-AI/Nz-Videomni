/**
 * Batch A2V — folder scan -> fresh in-memory `BatchRow`s. There is NO CSV
 * persistence anymore (owner decision, 2026-07-18): the batch is stateless —
 * a folder scan produces a plain in-memory row list that lives only as long
 * as the window stays open. This module owns the row shape (`BatchRow`/
 * `BatchStat`/`IMAGE_SHARED`) and the pure `scanToRows` folder-listing -> rows
 * mapping; assembling a generation request lives in `./buildA2vChainPayload`.
 *
 * `scanToRows` accepts the array shape the WebView2 bridge's `fs.listFiles`
 * resolves with (`{name, path, sizeBytes, mtimeMs, durationSec}[]`, see
 * `webui/src/bridge/types.ts` `BridgeResultMap["fs.listFiles"]`)
 * structurally — this module does not import the bridge itself, so it stays
 * dependency-free; any object with that shape works. The frame-count
 * arithmetic itself is injected as two callbacks (`framesFor`/
 * `rawFramesFor`) rather than imported from `modes/chained/chainUtils.ts`, per
 * this sprint's parallel-edit rule (that file's frame-count helpers are
 * being actively changed by another agent working the Chain screen).
 */

export type BatchStat = "Waiting" | "Generating" | "Done" | "Failed" | "Skip";

/** One batch queue row (the 10 fields the batch runner reads/writes in
 * memory). `wav`/`image`/`output` are filenames only (no directory
 * component) — the audio/image/output folders are resolved separately by the
 * caller. */
export interface BatchRow {
  queue: number;
  wav: string;
  duration: number;
  image: string;
  prompt: string;
  stat: BatchStat;
  output: string;
  frames: number;
  skipReason: string;
  error: string;
}

/** `image` column sentinel meaning "use the Generate tab's common i2v
 * keyframe(s)". Every fresh-scanned row defaults its `image` to this;
 * `./buildA2vChainPayload` keys off this same constant. */
export const IMAGE_SHARED = "Shared";

/** One entry as returned by the bridge's `fs.listFiles` (contract v6,
 * `withAudioDuration: true`). Declared locally (not imported) so this module
 * stays bridge-agnostic. */
export interface ScannedFile {
  name: string;
  path: string;
  sizeBytes: number;
  mtimeMs: number;
  /** Wav header duration in seconds; `0` for non-wav or unreadable files
   * (bridge contract — never an error, see `fs.listFiles`'s doc comment). */
  durationSec: number;
}

export interface ScanToRowsOptions {
  fps: number;
  /** Final frame count for a row that survives the 481-frame Skip test —
   * the shrink/clamp-against-VRAM-budget policy lives with the caller
   * (mirrors `gradio_ui.handlers.suggest_frames_for_audio`, injected by
   * `gradio_ui.manifest.scan_wav_folder` the same way; this module stays
   * policy-free). */
  framesFor: (durationSec: number, fps: number) => number;
  /** The raw (unclamped) 8n+1 frame count, used ONLY for the 481-frame Skip
   * test — mirrors `gradio_ui.manifest.raw_frame_count`
   * (`((floor(dur*fps)-1)//8)*8+1`). */
  rawFramesFor: (durationSec: number, fps: number) => number;
  /** Skip判定の上限フレーム数。省略時481。CreateタブのDURATION値が注入される。
   * API絶対上限481を超える値は内部でクランプ */
  maxFrames?: number;
}

/** Same extension allowlist as `manifest.py`'s `ALLOWED_AUDIO_EXTENSIONS` —
 * files outside this set are not even considered (excluded from the scan
 * entirely, never turned into a Skip row). Only `.wav` durations are
 * actually probed in the v1 alpha; every other allowed extension always
 * becomes a `wav-only-alpha` Skip row (no length-probing support yet). */
const ALLOWED_AUDIO_EXTENSIONS = [".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"];

/** Server hard cap on `num_frames` (`api/models.py`), mirrored from
 * `gradio_ui.manifest.MAX_FRAMES`. */
const MAX_FRAMES = 481;

function extOf(name: string): string {
  const idx = name.lastIndexOf(".");
  return idx === -1 ? "" : name.slice(idx).toLowerCase();
}

function baseRow(queue: number, wav: string, duration: number, stat: BatchStat, skipReason: string): BatchRow {
  return {
    queue,
    wav,
    duration,
    image: IMAGE_SHARED,
    prompt: "",
    stat,
    output: "",
    frames: 0,
    skipReason,
    error: "",
  };
}

/** Scans a folder listing into fresh `BatchRow`s, sorted by `mtimeMs`
 * ascending (spec §6.1 — VOICEROID writes files in script order, so mtime
 * order approximates script order better than filename order does).
 * Mirrors `gradio_ui.manifest.scan_wav_folder`:
 * - anything outside `ALLOWED_AUDIO_EXTENSIONS` (and `.tmp` scratch files) is
 *   excluded entirely;
 * - a non-`.wav` file, or a `.wav` whose duration couldn't be read
 *   (`durationSec <= 0`), becomes `stat=Skip, skipReason="wav-only-alpha"`;
 * - a `.wav` whose raw (unclamped) frame count exceeds the effective cap
 *   (`opts.maxFrames`, clamped to the 481 server hard cap) becomes
 *   `stat=Skip, skipReason="over-cap"`;
 * - everything else is `stat=Waiting` with `frames` from `opts.framesFor`.
 */
export function scanToRows(files: ScannedFile[], opts: ScanToRowsOptions): BatchRow[] {
  const cap = Math.min(opts.maxFrames ?? MAX_FRAMES, MAX_FRAMES);
  const candidates = files
    .filter((f) => !f.name.endsWith(".tmp"))
    .filter((f) => ALLOWED_AUDIO_EXTENSIONS.includes(extOf(f.name)))
    .slice()
    .sort((a, b) => a.mtimeMs - b.mtimeMs);

  return candidates.map((f, idx) => {
    const queue = idx + 1;
    const isWav = extOf(f.name) === ".wav";
    const dur = f.durationSec;

    if (!isWav || !(dur > 0)) {
      return baseRow(queue, f.name, 0, "Skip", "wav-only-alpha");
    }
    if (opts.rawFramesFor(dur, opts.fps) > cap) {
      return baseRow(queue, f.name, dur, "Skip", "over-cap");
    }
    const row = baseRow(queue, f.name, dur, "Waiting", "");
    row.frames = Math.trunc(opts.framesFor(dur, opts.fps));
    return row;
  });
}

/** Start-time re-judgment (frames / 481-frame Skip), recomputed at the
 * snapshot's current frame rate. Mirrors `gradio_ui.batch._plan_rejudgement`
 * + `_apply_plan` (`batch.py:594-641`) — those two are a plan/apply split
 * (project first, mutate only after every validation check passes) because
 * Python's rows are mutated in place under a lock; this module has no
 * mutable state to protect, so the two collapse into one pure
 * map — same judgment order, same outcomes.
 *
 * Only `"Waiting"`/`"Failed"` rows are re-judged (the original's
 * `_UNFINISHED` set minus `"Generating"` — see intentional difference #3
 * below); every other row is returned as the exact same object reference,
 * unchanged.
 *
 * Judgment order per targeted row (identical to `scanToRows`'s, so a row
 * re-scanned by folder-scan or by this function always lands the same way):
 * 1. `duration <= 0` -> `Skip("wav-only-alpha")`, `frames: 0`.
 * 2. `rawFramesFor(duration, fps) > cap` (the effective cap: `opts.maxFrames`
 *    clamped to the 481 server hard cap) -> `Skip("over-cap")`, `frames: 0`.
 *    The gate MUST use `rawFramesFor`, never `framesFor` — the latter is
 *    clamped to `[9, 481]` (see `scanToRows` line ~129) and can never itself
 *    exceed 481, which would make this branch permanently dead.
 * 3. Otherwise -> `frames: Math.trunc(framesFor(duration, fps))`, `stat`
 *    left untouched (a `Failed` row that re-validates stays `Failed`,
 *    matching the original's "frames" branch, which never touches `stat`).
 *
 * `error`/`output` are never touched by this function either way.
 *
 * Intentional differences from `batch.py:594-641`:
 * 1. `duration <= 0` is forced to `Skip` here; the Python original silently
 *    leaves such rows alone (they're assumed to already be non-`_UNFINISHED`
 *    Skip remnants). The frontend's manual "retry" affordance (re-arming a
 *    `Skip` row back to `Waiting`) can produce a `Waiting` row with
 *    `duration <= 0`, which this function must catch and re-Skip.
 * 2. The "frames" branch here also clears a stale `skipReason` back to
 *    `""`; the original never needs to because a `_UNFINISHED` row that
 *    reaches that branch never carried a `skipReason` to begin with.
 * 3. `"Generating"` is excluded from the targeted set entirely (the original
 *    includes it in `_UNFINISHED` to catch a crash-remnant row still marked
 *    mid-flight on process restart). The frontend has no such crash-remnant
 *    case — a `Generating` row cannot structurally survive into an idle,
 *    re-judgment-eligible state — so there is nothing to re-judge here.
 */
export function rejudgeRows(rows: BatchRow[], opts: ScanToRowsOptions): BatchRow[] {
  const cap = Math.min(opts.maxFrames ?? MAX_FRAMES, MAX_FRAMES);
  return rows.map((row) => {
    if (row.stat !== "Waiting" && row.stat !== "Failed") {
      return row;
    }
    if (!(row.duration > 0)) {
      return { ...row, stat: "Skip", skipReason: "wav-only-alpha", frames: 0 };
    }
    if (opts.rawFramesFor(row.duration, opts.fps) > cap) {
      return { ...row, stat: "Skip", skipReason: "over-cap", frames: 0 };
    }
    return { ...row, frames: Math.trunc(opts.framesFor(row.duration, opts.fps)), skipReason: "" };
  });
}
