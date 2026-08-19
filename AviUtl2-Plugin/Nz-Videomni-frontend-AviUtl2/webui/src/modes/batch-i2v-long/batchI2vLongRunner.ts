/**
 * バッチi2v-long（§1-7）の実行エンジン。
 *
 * 出典: `modes/batch/batchRunner.ts`（バッチA2Vの`BatchRunner`）。A2V側は本改修の
 * 編集対象外のため、共通化ではなく意図的に複製している。両者は行の型（A2Vは
 * 音声＋Skip列あり、こちらは画像1列のみ）とペイロード組み立て（A2Vは
 * `buildA2vChainPayload`の自己完結ミラー、こちらはChain画面の`buildRequest()`
 * テンプレート）が根本的に異なり、無理に一本化すると両機能が相互に壊れる。
 *
 * Batch i2v-long's execution engine. Ported (copied) from
 * `modes/batch/batchRunner.ts` — Batch A2V is deliberately NOT refactored into
 * a shared engine here, because it is outside this sprint's edit scope and the
 * two runners differ in their row model (A2V has audio + a `Skip` state; this
 * one has a single image column and no `Skip`) and in how a row's request body
 * is built (A2V re-derives one from primitives; this one stamps a row's image
 * onto a pre-built Chain-form template).
 *
 * Structure kept identical to the source on purpose, so a future fix to either
 * can be transplanted by inspection:
 * - one row at a time, in `queue` order, each inside its own try/catch, so a
 *   bad row can never take down the rest of the run;
 * - `Waiting`/`Failed`/`Generating` are "unfinished" and get (re)processed;
 *   `Done` never triggers an API call;
 * - `POST /generate/chain` 409 (JOB_BUSY) is retried 3× with a backoff, then
 *   that row alone fails;
 * - `GET /jobs/{id}` polling has NO deadline (a 24-clip chain legitimately runs
 *   for tens of minutes).
 *
 * The run only survives as long as this browser tab/window is open (same
 * caveat as A2V's, and the reason its on-screen notice says so).
 */
import type { GenerateChainRequest } from "../../api/types";
import { BridgeError } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import { buildRowPayload } from "./buildI2vLongPayload";
import type { PromptMode } from "./buildI2vLongPayload";
import type { I2vLongRow, I2vLongStat } from "./imageRows";

/** Rows in one of these `stat`s are "unfinished" and get (re)processed on the
 * next `start()`. `Generating` is included so a close-tab remnant (a row
 * mid-flight when the window was closed) is retried on the next run. Mirrors
 * `batchRunner.ts`'s `UNFINISHED_STATS` minus the `Skip` state, which this
 * feature's row model does not have (`imageRows.ts`). */
const UNFINISHED_STATS: ReadonlySet<I2vLongStat> = new Set<I2vLongStat>(["Waiting", "Failed", "Generating"]);

/** `POST /generate/chain` 409 (JOB_BUSY) retry policy — same numbers as Batch
 * A2V (which mirrors `gradio_ui/batch.py`'s `_JOB_BUSY_MAX_ATTEMPTS`/
 * `_JOB_BUSY_BACKOFF_S`). Overridable per `start()` call purely for test speed;
 * production call sites leave it at the default. */
const JOB_BUSY_MAX_ATTEMPTS = 3;
const DEFAULT_JOB_BUSY_BACKOFF_MS = 3_000;

/** `GET /jobs/{id}` poll interval, same as Batch A2V's. Overridable per
 * `start()` call purely for test speed. */
const DEFAULT_POLL_INTERVAL_MS = 1_000;

export type BatchI2vLongRunnerState = "idle" | "running" | "stopping";

/**
 * The run's non-template settings — everything that is the same for every row
 * but is NOT part of the Chain-form template snapshot. Today that is exactly
 * one field, the add/replace mode each row's own prompt is combined with the
 * template's shared prompt under.
 *
 * Modelled on Batch A2V's `BatchRunnerSettings` (`modes/batch/batchRunner.ts`)
 * and frozen for the whole run in the same way: the panel may keep editing its
 * radio while the batch runs, and none of that reaches the run in flight.
 */
export interface I2vLongRunnerSettings {
  promptMode: PromptMode;
}

export interface I2vLongRunnerStartParams {
  /** Absolute path to the image folder; each row's `image` is resolved against
   * it (`joinPath(imgDir, row.image)`). */
  imgDir: string;
  /** Absolute path each completed row's mp4 is saved into. */
  outDir: string;
  /**
   * The row-neutral `POST /generate/chain` body every row is derived from —
   * already snapshotted by the caller from the Chain form
   * (`buildI2vLongPayload.buildI2vLongTemplate`). Held BY VALUE for the whole
   * run: later Chain-screen edits cannot affect a batch in flight.
   *
   * Treated as deeply immutable. Sub-objects (`loras`, `crop_output`, clips
   * 1..n) are shared references with the Chain form's own state, so nothing in
   * this runner may mutate them — `buildRowPayload` is copy-on-write for
   * exactly that reason.
   */
  template: GenerateChainRequest;
  /** Frozen for the whole run, exactly like `template` — see
   * {@link I2vLongRunnerSettings}. */
  settings: I2vLongRunnerSettings;
  /** The full row list (every row, not just the unfinished ones). Never
   * mutated in place — the runner keeps its own copy-on-write map. Each row's
   * `prompt` column is combined with `template.prompt` at submit time. */
  rows: I2vLongRow[];
  /** Called after every `stat` transition with a fresh array reflecting the
   * runner's current row state, so a React caller can feed it straight into
   * `setState`. In-memory only: this batch has no manifest file of any kind. */
  onRowsChanged?: (rows: I2vLongRow[]) => void;
  /** Test-only override of the `GET /jobs/{id}` poll interval. */
  pollIntervalMs?: number;
  /** Test-only override of the 409 JOB_BUSY retry backoff. */
  jobBusyBackoffMs?: number;
}

export interface I2vLongRunnerStartResult {
  started: boolean;
  reason: string;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Joins a folder path with a file name using a backslash, unless `dir`
 * already ends with a path separator — the Windows-style paths this app always
 * deals in. */
function joinPath(dir: string, name: string): string {
  const sep = dir.endsWith("\\") || dir.endsWith("/") ? "" : "\\";
  return `${dir}${sep}${name}`;
}

function fileNameOf(path: string): string {
  const idx = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return idx >= 0 ? path.slice(idx + 1) : path;
}

/** Strips a file name's extension (last `.` onward); a name with no `.` (or
 * one that starts with it) is returned unchanged. */
function stemOf(name: string): string {
  const idx = name.lastIndexOf(".");
  return idx > 0 ? name.slice(0, idx) : name;
}

/** A short, readable failure string for a row's `error` cell (≤500 chars). */
function summarizeError(err: unknown): string {
  if (err instanceof BridgeError) return `${err.code}: ${err.message}`.slice(0, 500);
  if (err instanceof Error) return `${err.name}: ${err.message}`.slice(0, 500);
  return String(err).slice(0, 500);
}

/** Renders a `backend.request`/`backend.uploadFile` error body as a short
 * string. */
function summarizeBody(body: object | null): string {
  if (body === null) return "";
  try {
    return JSON.stringify(body).slice(0, 500);
  } catch {
    return String(body).slice(0, 500);
  }
}

/**
 * Runs one batch i2v-long queue against the given `NativeBridge`, one image at
 * a time, in `queue` order. One instance handles one run at a time — `start()`
 * returns `{started:false}` while already `running`/`stopping`.
 *
 * `start()` IS the whole run: it awaits every row and only resolves once the
 * run is fully finished (idle again) or stopped. A React caller fires it with
 * `void runner.start(...)` and relies on `onRowsChanged`/`state` for live
 * updates (see `runtime.ts`, which owns the singleton instance).
 */
export class BatchI2vLongRunner {
  private readonly bridge: NativeBridge;
  private _state: BatchI2vLongRunnerState = "idle";
  private rowsByQueue = new Map<number, I2vLongRow>();
  private params: I2vLongRunnerStartParams | null = null;
  private stopRequested = false;
  // NOTE: there is deliberately no `currentJobId` field here, unlike
  // `batchRunner.ts` — its only purpose there is the best-effort
  // `DELETE /jobs/{id}` on stop, which this runner never sends (see `stop()`).
  /** Uploaded image ids cached by resolved local path for the whole run, so a
   * folder containing the same picture twice uploads it once. */
  private imageIdCache = new Map<string, string>();

  constructor(bridge: NativeBridge) {
    this.bridge = bridge;
  }

  get state(): BatchI2vLongRunnerState {
    return this._state;
  }

  /** Begins a run. `{started:false}` when already running, or when no row's
   * `stat` is currently one of {@link UNFINISHED_STATS} — nothing is touched
   * in either case. */
  async start(params: I2vLongRunnerStartParams): Promise<I2vLongRunnerStartResult> {
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
    this.imageIdCache = new Map();

    try {
      await this.processAll();
    } finally {
      this._state = "idle";
      this.params = null;
    }
    return { started: true, reason: "completed" };
  }

  /**
   * 中止＝「以降の行を投入しない」だけ。**`DELETE /jobs/{id}`は投げない**
   * （バッチA2Vとの意図的な差）。
   *
   * Requests a graceful stop. Unlike `batchRunner.ts`'s `stop()`, this sends NO
   * `DELETE /jobs/{id}` — deliberately:
   *
   * - the backend cannot interrupt a chain job mid-flight
   *   (`pipeline_manager.py`'s cancel only marks the job `cancelled` AFTER it
   *   has run to completion), so a DELETE would burn the full GPU time anyway
   *   and then throw the finished video away;
   * - with strictly serial execution there is never a QUEUED job to cancel
   *   cheaply either — the only job in flight is the one being generated.
   *
   * So the row currently generating keeps polling and, when it completes, is
   * saved as `Done` exactly as if Stop had never been pressed; only the rows
   * after it are left `Waiting` for a later resume. Stopping therefore takes as
   * long as the current image needs (possibly tens of minutes) — the panel's
   * on-screen notice must say so. No-op while idle.
   */
  stop(): void {
    this.stopRequested = true;
    if (this._state === "running") this._state = "stopping";
  }

  private currentRows(): I2vLongRow[] {
    return [...this.rowsByQueue.values()].sort((a, b) => a.queue - b.queue);
  }

  private updateRow(queue: number, patch: Partial<I2vLongRow>): void {
    const prev = this.rowsByQueue.get(queue);
    if (!prev) return;
    this.rowsByQueue.set(queue, { ...prev, ...patch });
  }

  /** Notifies `onRowsChanged` with the full current row list on every `stat`
   * transition. Kept `async` (a no-op await) so the many `await this.flush()`
   * call sites read the same as the source runner's. */
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

  private async processRow(row: I2vLongRow): Promise<void> {
    const queue = row.queue;
    const params = this.params;
    if (!params) return;
    try {
      this.updateRow(queue, { stat: "Generating", error: "" });
      await this.flush();

      const imagePath = joinPath(params.imgDir, row.image);
      const imageId = await this.uploadImageCached(imagePath);
      // The row's own prompt is composed here, per row — the template carries
      // the Chain form's shared prompt untouched (2026-07-30 redesign).
      const payload = buildRowPayload(params.template, {
        imageId,
        rowPrompt: row.prompt,
        promptMode: params.settings.promptMode,
      });

      const jobId = await this.submitWithRetry(payload as unknown as object, params.jobBusyBackoffMs ?? DEFAULT_JOB_BUSY_BACKOFF_MS);
      if (jobId === null) {
        this.updateRow(queue, { stat: "Failed", error: "server busy (409) after retries" });
        await this.flush();
        return;
      }

      await this.pollRow(queue, jobId, params.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS);
    } catch (err) {
      this.updateRow(queue, { stat: "Failed", error: summarizeError(err) });
      await this.flush();
    }
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
   * {@link JOB_BUSY_MAX_ATTEMPTS} times with a fixed backoff. Returns the
   * `job_id` on success, `null` if 409 persisted through every retry; any other
   * >=400 status throws (caught by {@link processRow} -> `Failed`). */
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

  /** Polls `GET /jobs/{id}` until a terminal status, with no deadline. A
   * transport error or a non-200 status is treated as transient and simply
   * retried on the next tick. */
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
        // This runner never cancels anything itself (see `stop()`), so a
        // `cancelled` here can only come from outside (the Jobs panel's own
        // DELETE). Rewind to `Waiting` for a clean resume, same as A2V.
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

    const fileName = `${stemOf(row.image)}.mp4`;
    const result = await this.bridge.request("backend.downloadVideo", {
      jobId,
      destDir: params.outDir,
      fileName,
      noClobber: true,
    });
    // `noClobber` may have renamed the file (`img_2.mp4`) — record what was
    // actually written, never the name we asked for.
    const actualName = fileNameOf(result.filePath);
    this.updateRow(queue, { stat: "Done", output: actualName, error: "" });
    await this.flush();
  }
}
