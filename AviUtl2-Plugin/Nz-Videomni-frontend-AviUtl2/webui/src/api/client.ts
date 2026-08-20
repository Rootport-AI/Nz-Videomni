/**
 * Typed LTX23 backend API client. Every call goes through
 * `bridge.request("backend.request", ...)` (see `bridge/types.ts` v2) so it
 * works unmodified against both the dev mock bridge and the real WebView2
 * bridge, which proxies to the backend over WinHTTP.
 *
 * Two failure channels, deliberately kept separate:
 *  - Transport failure (bridge itself errors, e.g. BACKEND_UNREACHABLE /
 *    BACKEND_TIMEOUT, or the bridge is missing entirely): the `bridge.request`
 *    promise rejects with a `BridgeError`.
 *  - Backend-reported failure (any HTTP status outside 2xx, body is the
 *    `{error:{code,message}}` envelope from Docs/API_REFERENCE.md §2): the
 *    `bridge.request` promise *resolves* with `{status, body}`.
 *
 * Both are normalized into a single `BackendApiError` here so callers never
 * need to know which channel produced the failure.
 */
import { bridge as defaultBridge, BridgeError } from "../bridge";
import type { NativeBridge } from "../bridge";
import type {
  AppConfig,
  BackendErrorEnvelope,
  DeleteJobResponse,
  GenerateAcceptedResponse,
  GenerateChainAcceptedResponse,
  GenerateChainRequest,
  GenerateRequest,
  JobResponse,
  JoinRequestBody,
  JoinResponse,
  LorasListResponse,
  LorasReloadResponse,
  ModelCategory,
  ModelsResponse,
  PipelineLoadResponse,
  PipelineUnloadResponse,
  StatusResponse,
} from "./types";

/** Base path every backend endpoint lives under (Docs/API_REFERENCE.md §1). */
const API_PREFIX = "/api/v1";

export class BackendApiError extends Error {
  readonly code: string;
  /** HTTP status the backend returned, or 0 for a pure transport failure
   * (bridge unreachable/timeout/missing) that never reached the backend. */
  readonly httpStatus: number;
  /** The envelope's optional `detail` string — the SPECIFIC reason behind a
   * generic `message` (`api/errors.py`'s `APIError.to_envelope`, which omits
   * the key entirely when unset). `message` alone is often a template
   * ("selected model 'transformer/default' failed the compatibility
   * precheck") while `detail` carries the sentence actually worth showing
   * ("このtransformerはltxv 2.5.0です。…"), so `shell/useBaseModels.ts` shows
   * `detail` verbatim for a 422 rather than paraphrasing it client-side.
   *
   * `undefined` when absent OR when the backend sent the LIST form (the
   * `VALIDATION_ERROR` envelope's `detail` is an array of
   * `{loc,msg,type}` — see {@link BackendErrorEnvelope}); only the string
   * form is captured, so a reader never has to type-test it. */
  readonly detail: string | undefined;

  constructor(code: string, message: string, httpStatus: number, detail?: string) {
    super(message);
    this.name = "BackendApiError";
    this.code = code;
    this.httpStatus = httpStatus;
    this.detail = detail;
  }
}

function isErrorEnvelope(body: unknown): body is BackendErrorEnvelope {
  if (typeof body !== "object" || body === null || !("error" in body)) return false;
  const err = (body as { error: unknown }).error;
  return typeof err === "object" && err !== null && "code" in err && "message" in err;
}

interface CallOptions {
  query?: Record<string, string>;
  body?: object;
  /** v4.1: forwarded as `backend.request`'s `timeoutMs` param (see
   * `bridge/types.ts`) — used by `joinJob` below, whose backend-side
   * processing can legitimately outlast native's default WinHTTP timeout. */
  timeoutMs?: number;
}

export interface ApiClient {
  getStatus(): Promise<StatusResponse>;
  getConfig(): Promise<AppConfig>;
  generate(request: GenerateRequest): Promise<GenerateAcceptedResponse>;
  /** M6: `POST /generate/chain` — clip chaining / V2V continuation / A2V
   * (Docs/API_REFERENCE.md §3.13/§5.2). */
  generateChain(request: GenerateChainRequest): Promise<GenerateChainAcceptedResponse>;
  getJob(jobId: string): Promise<JobResponse>;
  listJobs(): Promise<JobResponse[]>;
  deleteJob(jobId: string): Promise<DeleteJobResponse>;
  /** M6: `POST /jobs/{id}/join` — V2V result + source clip join
   * (Docs/API_REFERENCE.md §3.17). Synchronous on the real backend; M7b adds
   * an explicit `timeoutMs: 120_000` (contract v4.1) since this call's
   * backend-side processing can legitimately outlast native's default
   * WinHTTP timeout. */
  joinJob(jobId: string, body?: JoinRequestBody): Promise<JoinResponse>;
  getLoras(): Promise<LorasListResponse>;
  reloadLoras(): Promise<LorasReloadResponse>;
  /** Model management (S1): `GET /models` (Docs/API_REFERENCE.md §3.5) — the
   * per-category (transformer/text_encoder/video_vae/audio) selectable model
   * list. Rescans server-side on every call, so refetching this is how the
   * WebUI detects a newly downloaded file. */
  getModels(): Promise<ModelsResponse>;
  /** Model management (S1): `POST /pipeline/load` (Docs/API_REFERENCE.md
   * §3.3) with an explicit per-category selection. A selection differing
   * from the live one forces a worker rebuild (unload -> load), which can
   * legitimately take minutes — `timeoutMs: 600_000` is set unconditionally
   * (mirrors `gradio_ui/api_client.py`'s 600s httpx timeout for the same
   * call), well past the bridge's default ~10s WinHTTP timeout.
   *
   * Multi-engine (§3-97 P6): `baseModel` switches the BASE MODEL and is
   * omitted from the body entirely when not given, so every pre-existing call
   * site keeps sending a byte-identical request. `loadPipeline({}, "LTX25")`
   * — an empty selection plus a base model — is the header dropdown's call:
   * the server resolves that base's own categories itself. */
  loadPipeline(
    models: Partial<Record<ModelCategory, string>>,
    baseModel?: string,
  ): Promise<PipelineLoadResponse>;
  /** N4 "danger zone": `POST /pipeline/unload` — tears down the loaded engine
   * without loading a replacement, freeing its VRAM. Same active-job guard as
   * `loadPipeline` (409 `JOB_BUSY` while a generation job is running); no
   * body, no extended timeout (mirrors `reloadLoras`'s minimal shape). */
  unloadPipeline(): Promise<PipelineUnloadResponse>;
}

/** Builds an `ApiClient` bound to the given bridge instance. The app uses the
 * default export below (bound to the app-wide bridge singleton); tests use
 * this factory directly to bind a purpose-built mock bridge (e.g. one
 * configured with `backendUnreachable: true`) without touching the shared
 * singleton. */
export function createApiClient(nativeBridge: NativeBridge): ApiClient {
  async function call<T>(method: "GET" | "POST" | "DELETE", path: string, options: CallOptions = {}): Promise<T> {
    let result;
    try {
      result = await nativeBridge.request("backend.request", {
        method,
        path: `${API_PREFIX}${path}`,
        ...(options.query ? { query: options.query } : {}),
        ...(options.body ? { body: options.body } : {}),
        ...(options.timeoutMs !== undefined ? { timeoutMs: options.timeoutMs } : {}),
      });
    } catch (err) {
      if (err instanceof BridgeError) {
        throw new BackendApiError(err.code, err.message, 0);
      }
      throw err;
    }

    const { status, body } = result;
    if (status >= 200 && status < 300) {
      return body as T;
    }
    if (isErrorEnvelope(body)) {
      const detail = typeof body.error.detail === "string" ? body.error.detail : undefined;
      throw new BackendApiError(body.error.code, body.error.message, status, detail);
    }
    throw new BackendApiError("UNKNOWN_ERROR", `Unexpected backend response (HTTP ${status})`, status);
  }

  return {
    getStatus: () => call<StatusResponse>("GET", "/status"),
    getConfig: () => call<AppConfig>("GET", "/config"),
    generate: (request) => call<GenerateAcceptedResponse>("POST", "/generate", { body: request }),
    generateChain: (request) => call<GenerateChainAcceptedResponse>("POST", "/generate/chain", { body: request }),
    getJob: (jobId) => call<JobResponse>("GET", `/jobs/${encodeURIComponent(jobId)}`),
    listJobs: () => call<JobResponse[]>("GET", "/jobs"),
    deleteJob: (jobId) => call<DeleteJobResponse>("DELETE", `/jobs/${encodeURIComponent(jobId)}`),
    joinJob: (jobId, body) =>
      call<JoinResponse>("POST", `/jobs/${encodeURIComponent(jobId)}/join`, { body: body ?? {}, timeoutMs: 120_000 }),
    getLoras: () => call<LorasListResponse>("GET", "/loras"),
    reloadLoras: () => call<LorasReloadResponse>("POST", "/loras/reload"),
    getModels: () => call<ModelsResponse>("GET", "/models"),
    loadPipeline: (models, baseModel) =>
      call<PipelineLoadResponse>("POST", "/pipeline/load", {
        body: { models, ...(baseModel ? { base_model: baseModel } : {}) },
        timeoutMs: 600_000,
      }),
    unloadPipeline: () => call<PipelineUnloadResponse>("POST", "/pipeline/unload"),
  };
}

/** Direct, bridge-bypassing thumbnail URL for a LoRA card's `<img src>`
 * (Docs/API_REFERENCE.md §3.8: "WebView2 の `<img src>` から直接叩ける…CORS
 * 対象外のリソース取得なのでプロキシ不要"). `baseUrl` comes from
 * `backend.getBaseUrl` (`modes/single/useBaseUrl.ts`); 404
 * (`LORA_THUMBNAIL_NOT_FOUND`) is handled by the `<img>`'s `onError`, not
 * here (`modes/inventory/LoraCard.tsx`). */
export function loraThumbnailUrl(baseUrl: string, name: string): string {
  return `${baseUrl}${API_PREFIX}/loras/${encodeURIComponent(name)}/thumbnail`;
}

/** App-wide client bound to the app-wide bridge singleton. Hooks default to
 * this but accept an override (see `useServerStatus`/`useGeneration`) so
 * tests can bind a purpose-configured mock bridge instead. */
export const apiClient: ApiClient = createApiClient(defaultBridge);

export const getStatus: ApiClient["getStatus"] = () => apiClient.getStatus();
export const getConfig: ApiClient["getConfig"] = () => apiClient.getConfig();
export const generate: ApiClient["generate"] = (request) => apiClient.generate(request);
export const generateChain: ApiClient["generateChain"] = (request) => apiClient.generateChain(request);
export const getJob: ApiClient["getJob"] = (jobId) => apiClient.getJob(jobId);
export const listJobs: ApiClient["listJobs"] = () => apiClient.listJobs();
export const deleteJob: ApiClient["deleteJob"] = (jobId) => apiClient.deleteJob(jobId);
export const joinJob: ApiClient["joinJob"] = (jobId, body) => apiClient.joinJob(jobId, body);
export const getLoras: ApiClient["getLoras"] = () => apiClient.getLoras();
export const reloadLoras: ApiClient["reloadLoras"] = () => apiClient.reloadLoras();
export const getModels: ApiClient["getModels"] = () => apiClient.getModels();
export const loadPipeline: ApiClient["loadPipeline"] = (models, baseModel) =>
  apiClient.loadPipeline(models, baseModel);
export const unloadPipeline: ApiClient["unloadPipeline"] = () => apiClient.unloadPipeline();
