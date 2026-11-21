/**
 * Batch v2v-ref-long: runs the planned rows one at a time (upload → generate →
 * poll → join → save). Copied from `batch-i2v-long/batchI2vLongRunner.ts`
 * rather than shared, like that runner was copied from Batch A2V's.
 *
 * - The rows are planned and frozen by the caller at Start; the runner keeps
 *   no row copies and only reports `stat`/`output`/`error` per row index.
 * - Each row runs inside its own try/catch: a failure marks that row Failed
 *   and the run moves on.
 * - `POST /generate/chain` 409 is retried 3× with a backoff; `GET /jobs/{id}`
 *   is polled with no deadline.
 * - `stop()` sends no `DELETE`: the row generating now finishes and is saved.
 */
import type { GenerateChainRequest } from "../../api/types";
import { BridgeError } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import { buildRowRequest, rowUploadTargets } from "./buildPayload";
import type { RowRequestSettings, UploadKind } from "./buildPayload";
import type { RowPlan, V2vRowStat } from "./types";

const JOB_BUSY_MAX_ATTEMPTS = 3;
const DEFAULT_JOB_BUSY_BACKOFF_MS = 3_000;
const DEFAULT_POLL_INTERVAL_MS = 1_000;
/** `POST /generate/chain` checks the opening video's frames before it creates
 * the job (≈13 s for a 2-minute 1080p video), past the 10 s default. */
const SUBMIT_TIMEOUT_MS = 120_000;

/** The join settings of the Jobs panel's JOIN button (`jobs/JoinControls.tsx`
 * defaults). */
export const JOIN_HANDLE_CROSSFADE_MS = 300;
export const JOIN_SOURCE_TAIL_SECONDS = 5;
const JOIN_TIMEOUT_MS = 120_000;

const ERROR_MAX_CHARS = 500;

const UPLOAD_ID_FIELD: Record<UploadKind, "image_id" | "video_id" | "audio_id"> = {
  image: "image_id",
  video: "video_id",
  audio: "audio_id",
};

export type V2vRefLongRunnerState = "idle" | "running" | "stopping";

export interface V2vRefLongRunnerSettings extends RowRequestSettings {
  joined: boolean;
  /** The row's error text for a failed join (`strings.batchV2vRefLong.joinFailed`). */
  joinFailedText: (message: string) => string;
}

export interface V2vRowPatch {
  stat?: V2vRowStat;
  output?: string;
  error?: string;
}

export interface V2vRefLongRunnerStartParams {
  /** `buildV2vRefLongTemplate(chain.buildRequest())` at Start; never mutated. */
  template: GenerateChainRequest;
  settings: V2vRefLongRunnerSettings;
  /** The rows to run, in order (frozen at Start). */
  plans: RowPlan[];
  outDir: string;
  onRowPatched: (index: number, patch: V2vRowPatch) => void;
  /** Test-only. */
  pollIntervalMs?: number;
  /** Test-only. */
  jobBusyBackoffMs?: number;
}

export interface V2vRefLongRunnerStartResult {
  started: boolean;
  reason: string;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function fileNameOf(path: string): string {
  const idx = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return idx >= 0 ? path.slice(idx + 1) : path;
}

function clip(text: string): string {
  return text.slice(0, ERROR_MAX_CHARS);
}

function summarizeError(err: unknown): string {
  if (err instanceof BridgeError) return clip(`${err.code}: ${err.message}`);
  if (err instanceof Error) return clip(`${err.name}: ${err.message}`);
  return clip(String(err));
}

function summarizeBody(body: object | null): string {
  if (body === null) return "";
  try {
    return clip(JSON.stringify(body));
  } catch {
    return clip(String(body));
  }
}

export class V2vRefLongRunner {
  private readonly bridge: NativeBridge;
  private _state: V2vRefLongRunnerState = "idle";
  private stopRequested = false;
  /** Upload ids of this run, keyed by `uploadKey` (kind + path). */
  private uploadIds = new Map<string, string>();

  constructor(bridge: NativeBridge) {
    this.bridge = bridge;
  }

  get state(): V2vRefLongRunnerState {
    return this._state;
  }

  /** Runs every plan; resolves when the run is over. `{started:false}` while
   * already running or with no plan. */
  async start(params: V2vRefLongRunnerStartParams): Promise<V2vRefLongRunnerStartResult> {
    if (this._state !== "idle") return { started: false, reason: "batch already running" };
    if (params.plans.length === 0) return { started: false, reason: "no rows to process" };

    this._state = "running";
    this.stopRequested = false;
    this.uploadIds = new Map();
    try {
      for (const plan of params.plans) {
        if (this.stopRequested) break;
        await this.processRow(plan, params);
      }
    } finally {
      this._state = "idle";
    }
    return { started: true, reason: "completed" };
  }

  /** Submits no further rows; the row generating now still finishes. */
  stop(): void {
    this.stopRequested = true;
    if (this._state === "running") this._state = "stopping";
  }

  private async processRow(plan: RowPlan, params: V2vRefLongRunnerStartParams): Promise<void> {
    const index = plan.index;
    const patch = (p: V2vRowPatch) => params.onRowPatched(index, p);
    try {
      patch({ stat: "Generating", error: "" });

      for (const target of rowUploadTargets(plan, params.settings.rules)) {
        if (this.uploadIds.has(target.key)) continue;
        const id = await this.upload(target.uploadKind, target.path);
        this.uploadIds.set(target.key, id);
      }

      const body = buildRowRequest(params.template, plan, this.uploadIds, params.settings);
      const jobId = await this.submitWithRetry(body, params.jobBusyBackoffMs ?? DEFAULT_JOB_BUSY_BACKOFF_MS);
      if (jobId === null) {
        patch({ stat: "Failed", error: "server busy (409) after retries" });
        return;
      }

      const outcome = await this.poll(jobId, params.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS);
      if (outcome.status === "failed") {
        patch({ stat: "Failed", error: clip(outcome.error) });
        return;
      }
      if (outcome.status === "cancelled") {
        // Only the Jobs panel can cancel (this runner never does): back to Waiting.
        patch({ stat: "Waiting", error: "" });
        return;
      }

      const joined = params.settings.joined && plan.sourcePath !== null;
      if (joined) {
        const joinError = await this.join(jobId);
        if (joinError !== null) {
          patch({ stat: "Failed", error: clip(params.settings.joinFailedText(joinError)) });
          return;
        }
      }
      const saved = await this.bridge.request("backend.downloadVideo", {
        jobId,
        ...(joined ? { kind: "joined" as const } : {}),
        destDir: params.outDir,
        fileName: plan.outFileName,
        noClobber: true,
      });
      // `noClobber` may have renamed the file (`name_2.mp4`).
      patch({ stat: "Done", output: fileNameOf(saved.filePath), error: "" });
    } catch (err) {
      patch({ stat: "Failed", error: summarizeError(err) });
    }
  }

  private async upload(kind: UploadKind, filePath: string): Promise<string> {
    const result = await this.bridge.request("backend.uploadFile", { kind, filePath });
    if (result.status >= 400) throw new Error(`upload ${kind} ${result.status}: ${summarizeBody(result.body)}`);
    const field = UPLOAD_ID_FIELD[kind];
    const id = (result.body as Partial<Record<typeof field, unknown>> | null)?.[field];
    if (typeof id !== "string" || id === "") throw new Error(`upload ${kind}: response missing ${field}`);
    return id;
  }

  /** The `job_id`, or `null` when 409 persisted through every attempt. */
  private async submitWithRetry(body: GenerateChainRequest, backoffMs: number): Promise<string | null> {
    for (let attempt = 0; attempt < JOB_BUSY_MAX_ATTEMPTS; attempt += 1) {
      const resp = await this.bridge.request("backend.request", {
        method: "POST",
        path: "/api/v1/generate/chain",
        body: body as unknown as object,
        timeoutMs: SUBMIT_TIMEOUT_MS,
      });
      if (resp.status === 409) {
        if (attempt < JOB_BUSY_MAX_ATTEMPTS - 1) await sleep(backoffMs);
        continue;
      }
      if (resp.status >= 400) throw new Error(`generate/chain ${resp.status}: ${summarizeBody(resp.body)}`);
      const jobId = (resp.body as { job_id?: unknown } | null)?.job_id;
      if (typeof jobId !== "string" || jobId === "") throw new Error("generate/chain: response missing job_id");
      return jobId;
    }
    return null;
  }

  /** Polls until a terminal status (no deadline; transport errors and non-200
   * answers are retried on the next tick). */
  private async poll(
    jobId: string,
    intervalMs: number,
  ): Promise<{ status: "completed" } | { status: "cancelled" } | { status: "failed"; error: string }> {
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
      if (body.status === "completed") return { status: "completed" };
      if (body.status === "cancelled") return { status: "cancelled" };
      if (body.status === "failed") {
        return { status: "failed", error: typeof body.error === "string" && body.error ? body.error : "failed" };
      }
    }
  }

  /** `POST /jobs/{id}/join`; `null` on 200, else a short reason. */
  private async join(jobId: string): Promise<string | null> {
    try {
      const resp = await this.bridge.request("backend.request", {
        method: "POST",
        path: `/api/v1/jobs/${jobId}/join`,
        body: { handle_crossfade_ms: JOIN_HANDLE_CROSSFADE_MS, source_tail_seconds: JOIN_SOURCE_TAIL_SECONDS },
        timeoutMs: JOIN_TIMEOUT_MS,
      });
      if (resp.status === 200) return null;
      return `${resp.status}: ${summarizeBody(resp.body)}`;
    } catch (err) {
      return summarizeError(err);
    }
  }
}
