/**
 * Batch A2V — the in-browser execution engine, a mirror of
 * `Nz-Videomni/gradio_ui/batch.py`'s `BatchRunner` adapted to run
 * entirely inside the WebUI (no daemon thread — the Gradio version needed one
 * because a Gradio event handler dies when the SSE stream drops; here the
 * "thread" is simply an `async` method the caller doesn't have to await, and
 * the run only survives as long as this browser tab/window is open — see the
 * required on-screen notice in `BatchSection.tsx`).
 *
 * Source of truth for a row's re-run eligibility is `BatchRow.stat` alone
 * (`Docs/BATCH_A2V_CSV_SPEC.md` §3.2/§4): `Waiting`/`Failed`/`Generating` are
 * "unfinished" and get (re)processed; `Done`/`Skip` never trigger an API
 * call. Every row is processed inside its own try/catch so one bad row can
 * never take down the rest of the run (mirrors `batch.py`'s "Design
 * invariants" doc comment).
 *
 * Deliberate simplifications vs. the Python reference (see this sprint's
 * report for the full list):
 * - The start-time "re-judge frames / Skip" pass (`batch.py`'s
 *   `_plan_rejudgement`) now lives in the CALLER, not here: `useBatchForm`'s
 *   `start()` runs `rejudgeRows` over its row list at the current fps, with
 *   the skip cap = min(Single tab's DURATION, 481) (commit 9324f05), and
 *   hands this runner the already-re-judged rows. The image/prompt "foolproof"
 *   preflight (`batch.py`'s `_validate`) is still NOT implemented anywhere —
 *   an invalid row (e.g. an empty composed prompt) simply fails at the backend
 *   and is caught by this module's own per-row try/catch, landing on
 *   `Failed` with the server's error message. The "one bad row never kills
 *   the batch" invariant already covers that without a separate preflight.
 * - N7 (PENDING §6): the `image` column's `Shared` sentinel (the Generate
 *   tab's common i2v keyframe(s)) IS now resolved — {@link resolveConditioning}
 *   uses `start()`'s `sharedConditioningImages` snapshot (already resolved to
 *   `image_id`s by the caller, `useBatchForm`'s `useKeyframes` instance) for
 *   any `Shared` row, via the same pure `resolveConditioningImages` a row
 *   naming its own image file also goes through (`buildA2vChainPayload.ts`).
 *   A run started without `sharedConditioningImages` defaults every `Shared`
 *   row to `[]` (the pre-N7 behavior), so this stays backward compatible.
 * - No reference-video (control IC-LoRA) adapter support — `batch.py`'s
 *   `use_adapter`/`ref_video_path`/`control_adherence`/`reference_strength`
 *   have no counterpart in {@link BatchRunnerSettings}; the task brief's
 *   "必要最小限" scope for the batch generation-settings panel doesn't call
 *   for them.
 */
import type { ConditioningImage, CropOutput, LoraSpec } from "../../api/types";
import { BridgeError } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import type { NagSettings } from "../../shell/nagSettings";
import type { AccelerationSettings } from "../../shell/accelerationSettings";
import { buildA2vChainPayload, composeRowPrompt, resolveConditioningImages } from "./buildA2vChainPayload";
import type { PromptMode } from "./buildA2vChainPayload";
import { IMAGE_SHARED } from "./manifestMerge";
import type { BatchRow, BatchStat } from "./manifestMerge";

/** Rows in one of these `stat`s are "unfinished" and get (re)processed on the
 * next `start()` — spec §4. `Generating` is included so a crash/close-tab
 * remnant (a row mid-flight when the browser tab was closed) is retried on
 * the next run, matching the required on-screen notice. */
const UNFINISHED_STATS: ReadonlySet<BatchStat> = new Set<BatchStat>(["Waiting", "Failed", "Generating"]);

/** `POST /generate/chain` 409 (JOB_BUSY) retry policy — mirrors
 * `batch.py`'s `_JOB_BUSY_MAX_ATTEMPTS`/`_JOB_BUSY_BACKOFF_S`. Overridable per
 * `start()` call (see {@link BatchRunnerStartParams.jobBusyBackoffMs}) purely
 * for test speed — production call sites should leave it at the default. */
const JOB_BUSY_MAX_ATTEMPTS = 3;
const DEFAULT_JOB_BUSY_BACKOFF_MS = 3_000;

/** GET /jobs/{id} poll interval. Mirrors `modes/single/useGeneration.ts`'s
 * own (module-private, not exported) `POLL_INTERVAL_MS` — duplicated here
 * rather than imported since that file is out of this sprint's edit scope
 * and doesn't export the constant. Overridable per `start()` call (see
 * {@link BatchRunnerStartParams.pollIntervalMs}) purely for test speed. */
const DEFAULT_POLL_INTERVAL_MS = 1_000;

export type BatchRunnerState = "idle" | "running" | "stopping";

/** The batch's shared (Generate-tab-equivalent) generation settings — the
 * fields {@link import("./buildA2vChainPayload").buildA2vChainPayload}
 * requires that are the same for every row in the run, plus the common
 * prompt + Add/Replace mode `composeRowPrompt` combines with each row's own
 * `prompt` column. Frozen for the whole run at `start()` time (mirrors
 * `batch.py`'s `BatchSnapshot` — "immune to later UI edits"). */
export interface BatchRunnerSettings {
  promptCommon: string;
  promptMode: PromptMode;
  width: number;
  height: number;
  cropOutput?: CropOutput | null;
  frameRate: number;
  seed: number;
  loras?: LoraSpec[];
  chunkedUpsample: boolean;
  /** NAG (2026-07-28)/VSF (2026-07-29): the shared Negative Prompt accordion's
   * settings — Batch has no NAG UI of its own (owner decision: it silently
   * inherits Create's accordion, exactly like every other Generate-tab-
   * equivalent field on this settings object). `undefined`/disabled sends
   * none of the 7 additive fields, via `buildA2vChainPayload`'s own
   * `nagRequestFields` call. */
  nag?: NagSettings;
  /** Acceleration (2026-07-31): the Settings panel's shared
   * attention-backend choice — Batch has no acceleration UI of its own and
   * silently inherits it, exactly like `nag` above. `undefined`, or the
   * server-default backend, adds nothing to any row's payload (via
   * `buildA2vChainPayload`'s own `accelerationRequestFields` call). */
  acceleration?: AccelerationSettings;
}

export interface BatchRunnerStartParams {
  /** Absolute path to the audio folder. Each row's `wav` is resolved against
   * this; the manifest CSV also lives directly under this folder (spec §1). */
  wavDir: string;
  /** Absolute path to the image folder a row's own (non-`Shared`) `image`
   * column is resolved against. Empty/omitted falls back to `wavDir`
   * (spec §3.1's backwards-compat rule). */
  imgDir?: string;
  /** Absolute path each completed row's mp4 is saved into. */
  outDir: string;
  settings: BatchRunnerSettings;
  /** The full row list (every row, not just the unfinished ones) — mutated
   * only through this runner's internal copy-on-write; the array/objects
   * passed in are never mutated in place. */
  rows: BatchRow[];
  /** N7 (PENDING §6): the Generate-tab's shared i2v keyframe(s), already
   * resolved to `image_id`s — frozen for the whole run (mirrors `settings`'s
   * own "immune to later UI edits" contract). Every `Shared`-image row's
   * `conditioning_images` comes from this snapshot, via
   * `resolveConditioningImages` (`buildA2vChainPayload.ts`). Omitted/`undefined`
   * defaults to `[]` (see {@link BatchRunner.resolveConditioning}) — a `Shared`
   * row then resolves to no keyframe, matching the pre-N7 behavior. */
  sharedConditioningImages?: ConditioningImage[];
  /** Called after every `stat` transition with a fresh array reflecting the
   * runner's current row state, so a React caller can feed it straight into
   * `setState`. This is the ONLY persistence the batch has — rows live in
   * memory, never in a CSV (owner decision, 2026-07-18: stateless batch). */
  onRowsChanged?: (rows: BatchRow[]) => void;
  /** Test-only override of the GET /jobs/{id} poll interval. Production call
   * sites should omit this (defaults to {@link DEFAULT_POLL_INTERVAL_MS}). */
  pollIntervalMs?: number;
  /** Test-only override of the 409 JOB_BUSY retry backoff. Production call
   * sites should omit this (defaults to {@link DEFAULT_JOB_BUSY_BACKOFF_MS}). */
  jobBusyBackoffMs?: number;
}

export interface BatchRunnerStartResult {
  started: boolean;
  reason: string;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Joins a folder path with a file name using a backslash, unless `dir`
 * already ends with a path separator — mirrors the Windows-style paths this
 * app always deals in (`bridge/mockBridge.ts`'s own `joinMockPath`). */
function joinPath(dir: string, name: string): string {
  const sep = dir.endsWith("\\") || dir.endsWith("/") ? "" : "\\";
  return `${dir}${sep}${name}`;
}

function fileNameOf(path: string): string {
  const idx = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return idx >= 0 ? path.slice(idx + 1) : path;
}

/** Strips a file name's extension (last `.` onward); a name with no `.`
 * (or one that starts with it, e.g. `.wav` alone) is returned unchanged. */
function stemOf(name: string): string {
  const idx = name.lastIndexOf(".");
  return idx > 0 ? name.slice(0, idx) : name;
}

/** Mirrors `batch.py`'s `_summarize_exc`: a short, readable string for the
 * CSV `error` column, truncated to 500 characters. */
function summarizeError(err: unknown): string {
  if (err instanceof BridgeError) return `${err.code}: ${err.message}`.slice(0, 500);
  if (err instanceof Error) return `${err.name}: ${err.message}`.slice(0, 500);
  return String(err).slice(0, 500);
}

/** Mirrors `batch.py`'s `_resp_error`: renders a `backend.request`/
 * `backend.uploadFile` error body as a short string. */
function summarizeBody(body: object | null): string {
  if (body === null) return "";
  try {
    return JSON.stringify(body).slice(0, 500);
  } catch {
    return String(body).slice(0, 500);
  }
}

/**
 * Runs a batch A2V queue against the given `NativeBridge`, one row at a time,
 * in `queue` order. One instance handles one run at a time — `start()`
 * rejects (returns `{started:false}`) while already `running`/`stopping`.
 *
 * Unlike `batch.py`'s `BatchRunner`, `start()` itself is the whole run: it
 * `await`s every row to completion and only resolves once the run is fully
 * finished (idle again) or stopped. A React caller fires it with `void
 * runner.start(...)` and relies on `onRowsChanged`/`state` for live updates,
 * exactly as `batch.py`'s `gr.Timer`-polled `snapshot_rows()`/`summary()`
 * stand in for the (here, nonexistent) daemon thread.
 */
export class BatchRunner {
  private readonly bridge: NativeBridge;
  private _state: BatchRunnerState = "idle";
  private rowsByQueue = new Map<number, BatchRow>();
  private params: BatchRunnerStartParams | null = null;
  private stopRequested = false;
  private currentJobId: string | null = null;
  /** Uploaded image ids cached by resolved absolute-ish local path for the
   * whole run — mirrors `batch.py`'s `_image_id_cache` ("no duplicate
   * uploads"). */
  private imageIdCache = new Map<string, string>();

  constructor(bridge: NativeBridge) {
    this.bridge = bridge;
  }

  get state(): BatchRunnerState {
    return this._state;
  }

  /** Begins a batch run. See {@link BatchRunnerStartResult}: `started:false`
   * when already running, or when no row's `stat` is currently one of
   * {@link UNFINISHED_STATS} (mirrors `batch.py`'s "no rows to process"
   * start-time rejection — nothing is touched in that case). */
  async start(params: BatchRunnerStartParams): Promise<BatchRunnerStartResult> {
    if (this._state !== "idle") {
      return { started: false, reason: "batch already running" };
    }
    const hasTarget = params.rows.some((r) => UNFINISHED_STATS.has(r.stat));
    if (!hasTarget) {
      return { started: false, reason: "no rows to process" };
    }

    this._state = "running";
    this.rowsByQueue = new Map(params.rows.map((r) => [r.queue, { ...r }] as const));
    this.params = params;
    this.stopRequested = false;
    this.currentJobId = null;
    this.imageIdCache = new Map();

    try {
      await this.processAll();
    } finally {
      this._state = "idle";
      this.currentJobId = null;
      this.params = null;
    }
    return { started: true, reason: "completed" };
  }

  /** Requests a graceful stop (spec §4's "explicit stop" rule): no further
   * rows are submitted after the one currently in flight, and a best-effort
   * `DELETE /jobs/{id}` is sent for that row. A queued job cancels
   * immediately — the next poll sees `status:"cancelled"` and rewinds the
   * row to `Waiting` for a clean resume; a running job may finish anyway
   * (`Done`/`Failed`). Any DELETE failure is swallowed (best-effort, mirrors
   * `batch.py`'s `request_stop`). No-op while idle. */
  stop(): void {
    this.stopRequested = true;
    if (this._state === "running") this._state = "stopping";
    const jobId = this.currentJobId;
    if (jobId) {
      void this.bridge
        .request("backend.request", { method: "DELETE", path: `/api/v1/jobs/${jobId}` })
        .catch(() => undefined);
    }
  }

  private currentRows(): BatchRow[] {
    return [...this.rowsByQueue.values()].sort((a, b) => a.queue - b.queue);
  }

  private updateRow(queue: number, patch: Partial<BatchRow>): void {
    const prev = this.rowsByQueue.get(queue);
    if (!prev) return;
    this.rowsByQueue.set(queue, { ...prev, ...patch });
  }

  /** Notifies `onRowsChanged` with the full current row list on every `stat`
   * transition. Stateless batch (owner decision, 2026-07-18): rows are held
   * in memory only — there is no manifest CSV to write, so this is purely the
   * live-update callback into the React caller. Kept `async` (a no-op await)
   * so the many `await this.flush()` call sites throughout `processRow`/
   * `pollRow` stay unchanged. */
  private async flush(): Promise<void> {
    const params = this.params;
    if (!params) return;
    params.onRowsChanged?.(this.currentRows());
  }

  private async processAll(): Promise<void> {
    const queueOrder = this.currentRows().map((r) => r.queue);
    for (const queue of queueOrder) {
      if (this.stopRequested) break;
      const row = this.rowsByQueue.get(queue);
      if (!row || !UNFINISHED_STATS.has(row.stat)) continue;
      await this.processRow(row);
    }
  }

  private async processRow(row: BatchRow): Promise<void> {
    const queue = row.queue;
    const params = this.params;
    if (!params) return;
    try {
      this.updateRow(queue, { stat: "Generating", error: "" });
      await this.flush();

      const wavPath = joinPath(params.wavDir, row.wav);
      const audioUpload = await this.bridge.request("backend.uploadFile", { kind: "audio", filePath: wavPath });
      if (audioUpload.status >= 400) {
        throw new Error(`upload audio ${audioUpload.status}: ${summarizeBody(audioUpload.body)}`);
      }
      const audioId = (audioUpload.body as { audio_id?: string } | null)?.audio_id;
      if (!audioId) throw new Error("upload audio: response missing audio_id");

      const conditioningImages = await this.resolveConditioning(row, params);

      const { settings } = params;
      const prompt = composeRowPrompt(settings.promptCommon, row.prompt, settings.promptMode);
      const payload = buildA2vChainPayload({
        audioId,
        numFrames: row.frames,
        prompt,
        width: settings.width,
        height: settings.height,
        ...(settings.cropOutput !== undefined ? { cropOutput: settings.cropOutput } : {}),
        frameRate: settings.frameRate,
        seed: settings.seed,
        conditioningImages,
        ...(settings.loras !== undefined ? { loras: settings.loras } : {}),
        chunkedUpsample: settings.chunkedUpsample,
        // NAG (2026-07-28): omitted entirely (not even as `undefined`) while
        // unset, matching every other optional field on this call.
        ...(settings.nag ? { nag: settings.nag } : {}),
        // Acceleration (2026-07-31): same omit-while-unset shape as `nag`.
        ...(settings.acceleration ? { acceleration: settings.acceleration } : {}),
      });

      const jobId = await this.submitWithRetry(payload as unknown as object, params.jobBusyBackoffMs ?? DEFAULT_JOB_BUSY_BACKOFF_MS);
      if (jobId === null) {
        this.updateRow(queue, { stat: "Failed", error: "server busy (409) after retries" });
        await this.flush();
        return;
      }
      this.currentJobId = jobId;

      await this.pollRow(queue, jobId, params.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS);
    } catch (err) {
      this.updateRow(queue, { stat: "Failed", error: summarizeError(err) });
      await this.flush();
    } finally {
      this.currentJobId = null;
    }
  }

  /** Resolves a row's `conditioning_images` (spec §3.1, N7-extended per
   * PENDING §6). Guard order matters — `Shared`/empty must never trigger an
   * upload:
   * - an empty `image` -> `[]`, no upload;
   * - `IMAGE_SHARED` -> `params.sharedConditioningImages` as-is (defaults to
   *   `[]` when the caller omitted it), no upload — the sentinel itself is
   *   never handed to `joinPath`/`uploadImageCached`;
   * - anything else (a row naming its own image file) -> uploads it (cached
   *   per absolute-ish path for the run) and resolves a single
   *   frame-0/strength-1.0 keyframe.
   * The actual `conditioning_images` shape for each case is delegated to the
   * pure `resolveConditioningImages` (`buildA2vChainPayload.ts`) so this stays
   * in sync with its documented contract instead of re-deriving the shape
   * here. */
  private async resolveConditioning(row: BatchRow, params: BatchRunnerStartParams): Promise<ConditioningImage[]> {
    const sharedImages = params.sharedConditioningImages ?? [];
    if (!row.image) return [];
    if (row.image === IMAGE_SHARED) {
      return resolveConditioningImages({ image: row.image, sharedImages });
    }
    const imgDir = params.imgDir || params.wavDir;
    const imagePath = joinPath(imgDir, row.image);
    const rowImageId = await this.uploadImageCached(imagePath);
    return resolveConditioningImages({ image: row.image, sharedImages, rowImageId });
  }

  private async uploadImageCached(imagePath: string): Promise<string> {
    const cached = this.imageIdCache.get(imagePath);
    if (cached) return cached;
    const result = await this.bridge.request("backend.uploadFile", { kind: "image", filePath: imagePath });
    if (result.status >= 400) {
      throw new Error(`upload image ${result.status}: ${summarizeBody(result.body)}`);
    }
    const imageId = (result.body as { image_id?: string } | null)?.image_id;
    if (!imageId) throw new Error("upload image: response missing image_id");
    this.imageIdCache.set(imagePath, imageId);
    return imageId;
  }

  /** `POST /generate/chain`, retrying a 409 (JOB_BUSY) up to
   * {@link JOB_BUSY_MAX_ATTEMPTS} times with a fixed backoff — mirrors
   * `batch.py`'s `_submit_with_retry`. Returns the `job_id` on success,
   * `null` if 409 persisted through every retry; any other >=400 status
   * throws (caught by {@link processRow} -> `Failed`). */
  private async submitWithRetry(payload: object, backoffMs: number): Promise<string | null> {
    for (let attempt = 0; attempt < JOB_BUSY_MAX_ATTEMPTS; attempt += 1) {
      const resp = await this.bridge.request("backend.request", {
        method: "POST",
        path: "/api/v1/generate/chain",
        body: payload,
      });
      if (resp.status === 409) {
        if (attempt < JOB_BUSY_MAX_ATTEMPTS - 1) await sleep(backoffMs);
        continue;
      }
      if (resp.status >= 400) {
        throw new Error(`generate/chain ${resp.status}: ${summarizeBody(resp.body)}`);
      }
      const jobId = (resp.body as { job_id?: string } | null)?.job_id;
      if (!jobId) throw new Error("generate/chain: response missing job_id");
      return jobId;
    }
    return null;
  }

  /** Polls `GET /jobs/{id}` until a terminal status, with no deadline (task
   * brief: "デッドライン無し" — an overnight batch's individual clip can
   * legitimately take a long time). A transport error or a non-200 status is
   * treated as transient and simply retried on the next tick, mirroring
   * `batch.py`'s `except Exception: continue` around `get_job`. */
  private async pollRow(queue: number, jobId: string, intervalMs: number): Promise<void> {
    for (;;) {
      await sleep(intervalMs);
      let body: Record<string, unknown>;
      try {
        const resp = await this.bridge.request("backend.request", { method: "GET", path: `/api/v1/jobs/${jobId}` });
        if (resp.status !== 200 || resp.body === null) continue;
        body = resp.body as Record<string, unknown>;
      } catch {
        continue;
      }

      const status = body.status;
      if (status === "completed") {
        await this.finishCompleted(queue, jobId);
        return;
      }
      if (status === "failed") {
        const error = typeof body.error === "string" && body.error ? body.error : "failed";
        this.updateRow(queue, { stat: "Failed", error: error.slice(0, 500) });
        await this.flush();
        return;
      }
      if (status === "cancelled") {
        // Stop-driven cancel: rewind to Waiting for a clean resume (spec §4).
        this.updateRow(queue, { stat: "Waiting", error: "" });
        await this.flush();
        return;
      }
      // queued / running -> keep polling.
    }
  }

  private async finishCompleted(queue: number, jobId: string): Promise<void> {
    const params = this.params;
    if (!params) return;
    const row = this.rowsByQueue.get(queue);
    if (!row) return;

    const fileName = `${stemOf(row.wav)}.mp4`;
    const result = await this.bridge.request("backend.downloadVideo", {
      jobId,
      destDir: params.outDir,
      fileName,
      noClobber: true,
    });
    const actualName = fileNameOf(result.filePath);
    this.updateRow(queue, { stat: "Done", output: actualName, error: "" });
    await this.flush();
  }
}
