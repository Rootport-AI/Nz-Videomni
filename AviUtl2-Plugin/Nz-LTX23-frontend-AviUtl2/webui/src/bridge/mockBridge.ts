import { BridgeError, type BridgeMethod, type NativeBridge, type ParamsOf, type ResultOf } from "./types";

/** Default `getEditInfo` payload, matching the values noted in the M1 spec. */
const DEFAULT_EDIT_INFO: ResultOf<"getEditInfo"> = {
  width: 1920,
  height: 1080,
  rate: 30,
  scale: 1,
  sampleRate: 44100,
  frame: 120,
  // Native always sends these three (BRIDGE_CONTRACT.md §9): current layer,
  // project max frame, project max layer. Fixture-only values; no dev/test
  // consumer reads them yet.
  layer: 1,
  frameMax: 300,
  layerMax: 100,
};

/** Fixture mirroring `AppConfig.model_dump()` as documented in
 * Docs/API_REFERENCE.md §3.2 / §5.1. Only the parts the WebUI actually reads
 * (generation_presets/defaults/limits/upload) need to be realistic; the rest
 * is included for shape-fidelity but ignored by the client.
 *
 * Exported (B-1, 2026-08-12 adversarial review) solely so
 * `mockBridge.test.ts` can assert `limits` stays in parity with
 * `modes/single/defaultConfig.ts`'s `FALLBACK_APP_CONFIG.limits` — the two
 * are independently hand-maintained fixtures for the same real `/config`
 * contract ("mocks walk the real contract path" lesson), and this file's own
 * `limits` had silently drifted behind the real backend's
 * `chain_comfort_token_budget` key once before being caught. */
export const MOCK_CONFIG_BODY = {
  server: { host: "127.0.0.1", port: 18620, allow_all_cors: false, api_key: null },
  model: {
    backend: "mock",
    pipeline_type: "distilled",
    // N2: control IC-LoRAs for Create's dropdown (`GenerationForm.tsx`) —
    // full `{path, preprocess}` entries plus one legacy bare-path-shaped
    // entry (`deblur`), so dev/mock exercises both shapes
    // `AppConfig.model.ic_loras` accepts.
    ic_loras: {
      "canny-control": { path: "mock/canny.safetensors", preprocess: "canny" },
      "pose-control": { path: "mock/pose.safetensors", preprocess: "dwpose" },
      "depth-control": { path: "mock/depth.safetensors", preprocess: "depth" },
      "deblur": "mock/deblur.safetensors",
      // §1-13 Outpainting: the Edit tab's pinned adapter (`lora/controlLoras.ts`'s
      // `OUTPAINT_LORA_NAME`). Must be present for `useOutpaintForm`'s
      // `hasOutpaintLora` gate to clear in dev/mock/tests.
      "in-outpainting": "mock/in-outpainting.safetensors",
    },
  },
  generation_presets: {
    smoke_test: { width: 384, height: 256, crop_output: null, num_frames: 17 },
    minimal: { width: 512, height: 320, crop_output: null, num_frames: 481 },
    small: {
      width: 960,
      height: 576,
      crop_output: { width: 960, height: 540 },
      num_frames: 481,
    },
    standard_720p: {
      width: 1280,
      height: 768,
      crop_output: { width: 1280, height: 720 },
      num_frames: 361,
    },
    FHD_1080p: {
      width: 1920,
      height: 1088,
      crop_output: { width: 1920, height: 1080 },
      num_frames: 169,
    },
    WQHD_1440p: {
      width: 2560,
      height: 1472,
      crop_output: { width: 2560, height: 1440 },
      num_frames: 89,
    },
  },
  generation_defaults: {
    width: 1280,
    height: 768,
    crop_output: null,
    num_frames: 361,
    frame_rate: 24.0,
    num_inference_steps: 8,
    guidance_scale: 1.0,
    seed: -1,
    pipeline: "distilled",
    conditioning_images: [],
  },
  upload: {
    dir: "./uploads",
    max_image_size_mb: 20,
    allowed_image_extensions: [".png", ".jpg", ".jpeg", ".webp"],
    normalize_to_png: true,
    max_video_size_mb: 200,
    allowed_video_extensions: [".mp4", ".mov", ".webm", ".mkv"],
    max_audio_size_mb: 50,
    allowed_audio_extensions: [".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"],
  },
  limits: {
    max_width: 1920,
    max_height: 1088,
    max_num_frames: 481,
    max_conditioning_images: 5,
    conditioning_frame_idx_multiple: 8,
    conditioning_keyframe_grid_offset: 1,
    phase1_max_concurrent_jobs: 1,
    low_vram_disabled_required: false,
    spill_free_frames: { "512x320": 481, "960x576": 481, "1280x768": 257, "1920x1088": 153, "2560x1472": 81 },
    v2v_context_frames_default: 73,
    v2v_context_frames_min: 25,
    v2v_context_frames_max: 145,
    retake_window_min_frames: 73,
    retake_window_max_frames: 169,
    end_context_frames_default: 72,
    end_context_frames_min: 8,
    end_context_frames_max: 136,
    chain_comfort_token_budget: 40000,
    single_comfort_token_budget: 44880,
  },
  output: { dir: "./outputs", format: "mp4", save_metadata_json: true, keep_raw_frames: false },
};

/** Fixture for `GET /loras` (Docs/API_REFERENCE.md §3.6). Deliberately mixes
 * `has_thumbnail: true/false` across both kinds so the Library screen's
 * thumbnail-vs-placeholder fallback path is exercised in dev/tests without
 * needing a real image behind the (never actually fetched, in this mock)
 * thumbnail URL — the `true` entries below rely on their `<img>`'s
 * `onError` handler to fall back, exactly as it would for a genuinely
 * missing file.
 *
 * The `control` rows mirror `MOCK_CONFIG_BODY.model.ic_loras` ONE-FOR-ONE on
 * purpose: `lora/controlLoras.ts`'s `resolveControlLoraNames` reads this list
 * once `/loras` has landed but falls back to the `/config` keys before then, so
 * a row missing here would make the Create dropdown silently SHRINK the moment
 * `/loras` resolved. Keep the two lists in step whenever either gains an
 * adapter (`depth-control`/`deblur` were added on 2026-08-04 for exactly this
 * reason — see ledger §4-8(A)). */
const MOCK_LORAS: Array<{ name: string; kind: "style" | "control"; has_thumbnail: boolean; exists: boolean; source: string }> = [
  { name: "Pixar_Toon", kind: "style", has_thumbnail: true, exists: true, source: "scan" },
  { name: "LTX-2.3-Henshin", kind: "style", has_thumbnail: false, exists: true, source: "scan" },
  { name: "canny-control", kind: "control", has_thumbnail: true, exists: true, source: "config" },
  { name: "pose-control", kind: "control", has_thumbnail: false, exists: true, source: "config" },
  { name: "depth-control", kind: "control", has_thumbnail: false, exists: true, source: "config" },
  { name: "deblur", kind: "control", has_thumbnail: false, exists: true, source: "config" },
  // §1-13 Outpainting: see the matching `ic_loras` entry above for why this
  // one must stay in step too. Deliberately EXCLUDED from Create/Chain's own
  // dropdown by `lora/controlLoras.ts`'s `UI_HIDDEN_CONTROL_LORA_NAMES` — see
  // `AppShell.controlLora.test.tsx`'s "lists every control adapter" test,
  // which pins the dropdown's set to the four names ABOVE this one on purpose.
  { name: "in-outpainting", kind: "control", has_thumbnail: false, exists: true, source: "config" },
];

/** Model management (S1) fixed category order, mirroring
 * `services/model_registry.py:69-97` `CATEGORIES`. */
const MOCK_MODEL_CATEGORIES = ["transformer", "text_encoder", "video_vae", "audio"] as const;
type MockModelCategory = (typeof MOCK_MODEL_CATEGORIES)[number];

interface MockModelEntryFixture {
  name: string;
  path: string;
  is_default: boolean;
  exists: boolean;
  source: "config" | "scan";
}

/** Fixture for `GET /models` (Docs/API_REFERENCE.md §3.5): each category gets
 * the always-present `"default"` entry plus one realistic alternate. The
 * `text_encoder` alternate has `exists: false` on purpose — it exercises the
 * "registered but the file is gone" (`MODEL_FILE_MISSING`) path and the
 * ModelsPanel's "(file missing)" annotation without needing a real missing
 * file on disk. */
const MOCK_MODEL_ENTRIES: Record<MockModelCategory, MockModelEntryFixture[]> = {
  transformer: [
    {
      name: "default",
      path: "models/ltx-2.3-gguf/ltx-2.3-13b-distilled-q8_0.gguf",
      is_default: true,
      exists: true,
      source: "config",
    },
    {
      name: "ltx-2.3-13b-distilled-q4_k_m",
      path: "models/ltx-2.3-gguf/ltx-2.3-13b-distilled-q4_k_m.gguf",
      is_default: false,
      exists: true,
      source: "scan",
    },
  ],
  text_encoder: [
    {
      name: "default",
      path: "models/ltx-2.3-components/gemma-2-2b-it-Q8_0.gguf",
      is_default: true,
      exists: true,
      source: "config",
    },
    {
      name: "gemma-2-2b-it-Q4_K_M",
      path: "models/ltx-2.3-components/gemma-2-2b-it-Q4_K_M.gguf",
      is_default: false,
      exists: false,
      source: "scan",
    },
  ],
  video_vae: [
    {
      name: "default",
      path: "models/ltx-2.3-components/vae/ltx-2.3-video-vae.safetensors",
      is_default: true,
      exists: true,
      source: "config",
    },
    {
      name: "video_vae_fp16",
      path: "models/ltx-2.3-components/vae/video_vae_fp16.safetensors",
      is_default: false,
      exists: true,
      source: "scan",
    },
  ],
  audio: [
    {
      name: "default",
      path: "models/ltx-2.3-components/vae/ltx-2.3-audio-vae.safetensors",
      is_default: true,
      exists: true,
      source: "config",
    },
    {
      name: "audio_vae_alt",
      path: "models/ltx-2.3-components/vae/audio_vae_alt.safetensors",
      is_default: false,
      exists: true,
      source: "scan",
    },
  ],
};

/** Progress "stage" labels a job cycles through while running, per
 * Docs/API_REFERENCE.md §4. */
const MOCK_STAGES = ["encode", "stage1_denoise", "denoise", "decode"] as const;

const MOCK_TOTAL_STEPS = 8;

type MockJobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

interface MockJobRecord {
  id: string;
  request: Record<string, unknown>;
  createdAt: string;
  /** Incremented on every `GET /jobs/{id}`; drives the queued->running->
   * completed progression. Peeked (not incremented) elsewhere (list, busy
   * check, delete, download). */
  pollCount: number;
  /** Set at creation time (via magic prompt substrings) to make the job
   * settle on `failed`/`cancelled` instead of `completed`, for testing error
   * paths deterministically. */
  forcedTerminal?: "failed" | "cancelled";
  /** M6: set only for `/generate/chain` jobs, to `clips.length` from the
   * request — drives the `clip`/`clip_count` fields in `JobResponse`
   * (Docs/API_REFERENCE.md §6). `undefined` for plain `/generate` jobs, which
   * always report `clip: null, clip_count: null`. */
  numClips?: number;
  /** I4: mirrors the real server's `JobResponse.is_v2v` — true when this is a
   * chain job that carried a `source_video` spec at submit time. Set once at
   * creation from the request; the echoed `request` never carries
   * `source_video`, so this flag is the only signal (see `jobResponseBody`). */
  isV2V: boolean;
  /** I4: ISO timestamp of the last successful `POST /jobs/{id}/join`, or `null`
   * before any join has run. Stands in for the real server's file-existence
   * check for `joined.mp4` — drives `JobResponse.joined` and the
   * `GET /jobs/{id}/joined` 200-vs-404 route. */
  joinedAt: string | null;
}

interface DerivedJobFields {
  status: MockJobStatus;
  progress: number;
  currentStep: number | null;
  stage: string | null;
}

function isNonTerminal(status: MockJobStatus): boolean {
  return status === "queued" || status === "running";
}

/** Pure function: derives a job's current status/progress from its poll
 * count without mutating anything, so it can be reused for both the actual
 * polling endpoint (which increments first) and read-only peeks (list,
 * busy-check, delete, download). */
function deriveJobFields(pollCount: number, runningPolls: number, forcedTerminal?: "failed" | "cancelled"): DerivedJobFields {
  if (pollCount <= 1) {
    return { status: "queued", progress: 0, currentStep: null, stage: null };
  }
  const runningIndex = pollCount - 1; // 1..runningPolls while "running"
  if (runningIndex <= runningPolls) {
    const progress = runningIndex / runningPolls;
    const stageIdx = Math.min(MOCK_STAGES.length - 1, Math.floor(progress * MOCK_STAGES.length));
    return {
      status: "running",
      progress,
      currentStep: Math.round(progress * MOCK_TOTAL_STEPS),
      stage: MOCK_STAGES[stageIdx] ?? null,
    };
  }
  return {
    status: forcedTerminal ?? "completed",
    progress: 1,
    currentStep: forcedTerminal === "failed" ? Math.round(0.5 * MOCK_TOTAL_STEPS) : MOCK_TOTAL_STEPS,
    stage: null,
  };
}

/** M6: pure derivation of `clip`/`clip_count` (Docs/API_REFERENCE.md §6),
 * mirroring `deriveJobFields`'s progress formula so the two stay in lockstep
 * (clip N becomes "current" at the same poll where overall progress crosses
 * N/numClips). Returns `{clip: null, clip_count: null}` for a non-chain job
 * (`numClips` undefined) — the documented shape for `/generate` jobs. */
function deriveClipFields(
  pollCount: number,
  runningPolls: number,
  numClips: number | undefined,
): { clip: number | null; clip_count: number | null } {
  if (!numClips || numClips <= 0) return { clip: null, clip_count: null };
  if (pollCount <= 1) return { clip: 1, clip_count: numClips };
  const runningIndex = pollCount - 1;
  if (runningIndex <= runningPolls) {
    const progress = runningIndex / runningPolls;
    const clip = Math.min(numClips, 1 + Math.floor(progress * numClips));
    return { clip, clip_count: numClips };
  }
  return { clip: numClips, clip_count: numClips };
}

function nowIso(): string {
  return new Date().toISOString().replace(/\.\d+Z$/, "Z");
}

/** Lowercased extension (including the leading `.`) of a mock file name, or
 * `""` if there is none. Shared by `fs.listFiles`'s extension filter,
 * `fs.probeAudioDuration`'s `.wav` check, and `backend.downloadVideo`'s
 * `noClobber` stem/extension split. */
function extnameLower(name: string): string {
  const idx = name.lastIndexOf(".");
  return idx > 0 ? name.slice(idx).toLowerCase() : "";
}

/** Joins a mock folder path with a file name using a backslash, unless `dir`
 * already ends with a path separator. Mirrors the Windows-style paths used
 * throughout this mock (e.g. `C:\\Users\\mock\\...`). */
function joinMockPath(dir: string, name: string): string {
  const sep = dir.endsWith("\\") || dir.endsWith("/") ? "" : "\\";
  return `${dir}${sep}${name}`;
}

/** Contract v6 `backend.downloadVideo` `noClobber` support: returns
 * `desiredName` unchanged if no entry in `entries` already has that name,
 * otherwise appends `_2`, `_3`, ... to the name's stem until a free name is
 * found. Preserves the original casing of the extension (unlike
 * `extnameLower`, which is only used for case-insensitive comparisons). */
function dedupeFileName(entries: MockFsFileEntry[] | undefined, desiredName: string): string {
  const existingNames = new Set((entries ?? []).map((e) => e.name));
  if (!existingNames.has(desiredName)) return desiredName;
  const dotIdx = desiredName.lastIndexOf(".");
  const stem = dotIdx > 0 ? desiredName.slice(0, dotIdx) : desiredName;
  const ext = dotIdx > 0 ? desiredName.slice(dotIdx) : "";
  let n = 2;
  let candidate = `${stem}_${n}${ext}`;
  while (existingNames.has(candidate)) {
    n += 1;
    candidate = `${stem}_${n}${ext}`;
  }
  return candidate;
}

/** Builds the echoed `JobResponse.request` exactly as the real server does:
 * the whitelisted `GenerateRequest` field set only (never the raw submitted
 * body). This is the structural fix for the "test-green / device-invisible"
 * Join bug — the real `GET /jobs` runs each record through `to_clip_request`
 * (`api/models.py`), so a chain job's `source_video`/`source_audio`/`clips`/
 * `overlap_frames` never reach the response, and a chain job's per-clip
 * `GenerateRequest` also drops `loras`/`reference_video_id`/the IC-LoRA
 * strengths (`to_clip_request` doesn't forward them). Mirroring that here
 * keeps the mock honest so an `is_v2v`-based Join gate is exercised the same
 * way it behaves on a real server. */
function buildEchoedRequest(request: Record<string, unknown>): Record<string, unknown> {
  const str = (v: unknown, fallback: string): string => (typeof v === "string" ? v : fallback);
  const num = (v: unknown, fallback: number): number => (typeof v === "number" ? v : fallback);
  const clipsArr = Array.isArray(request.clips)
    ? (request.clips as Array<Record<string, unknown>>)
    : null;
  const isChain = clipsArr !== null;
  const clip0: Record<string, unknown> = clipsArr?.[0] ?? {};

  // Chain jobs echo clip-0's `to_clip_request(0)` GenerateRequest: prompt /
  // num_frames / conditioning_images come from clip 0, crop is deferred to the
  // final concat (always null per-clip), and the IC-LoRA fields are NOT
  // forwarded per-clip (default []/null). Plain /generate jobs echo their own
  // submitted GenerateRequest fields, with server-only defaults filled in.
  const prompt = isChain
    ? (typeof clip0.prompt === "string" && clip0.prompt.length > 0 ? clip0.prompt : str(request.prompt, ""))
    : str(request.prompt, "");
  const numFrames = isChain ? num(clip0.num_frames, 49) : num(request.num_frames, 49);
  const conditioningImages = isChain
    ? (Array.isArray(clip0.conditioning_images) ? clip0.conditioning_images : [])
    : (Array.isArray(request.conditioning_images) ? request.conditioning_images : []);
  const cropOutput = isChain ? null : (request.crop_output ?? null);
  const loras = isChain ? [] : (Array.isArray(request.loras) ? request.loras : []);
  const referenceVideoId = isChain ? null : (request.reference_video_id ?? null);
  const condAttn = isChain ? null : (request.conditioning_attention_strength ?? null);
  const refStrength = isChain ? null : (request.reference_video_strength ?? null);

  return {
    prompt,
    negative_prompt: str(request.negative_prompt, ""),
    width: num(request.width, 512),
    height: num(request.height, 320),
    crop_output: cropOutput,
    num_frames: numFrames,
    frame_rate: num(request.frame_rate, 24.0),
    num_inference_steps: num(request.num_inference_steps, MOCK_TOTAL_STEPS),
    guidance_scale: num(request.guidance_scale, 1.0),
    seed: num(request.seed, -1),
    pipeline: str(request.pipeline, "distilled"),
    conditioning_images: conditioningImages,
    loras,
    reference_video_id: referenceVideoId,
    conditioning_attention_strength: condAttn,
    reference_video_strength: refStrength,
  };
}

function jobResponseBody(job: MockJobRecord, fields: DerivedJobFields, runningPolls: number): Record<string, unknown> {
  const request = job.request;
  const width = typeof request.width === "number" ? request.width : 512;
  const height = typeof request.height === "number" ? request.height : 320;
  // M6: a chain request has no top-level `num_frames` — sum the per-clip
  // `num_frames` instead, for a realistic-ish `result.duration_seconds`.
  const clipsArr = Array.isArray(request.clips) ? (request.clips as Array<{ num_frames?: unknown }>) : null;
  const chainTotalFrames = clipsArr
    ? clipsArr.reduce((sum, c) => sum + (typeof c.num_frames === "number" ? c.num_frames : 0), 0)
    : 0;
  const numFrames =
    chainTotalFrames > 0 ? chainTotalFrames : typeof request.num_frames === "number" ? request.num_frames : 49;
  const frameRate = typeof request.frame_rate === "number" ? request.frame_rate : 24.0;
  const seed = typeof request.seed === "number" ? request.seed : -1;
  const clipFields = deriveClipFields(job.pollCount, runningPolls, job.numClips);

  const echoedRequest = buildEchoedRequest(request);

  const base: Record<string, unknown> = {
    job_id: job.id,
    status: fields.status,
    progress: fields.progress,
    current_step: fields.currentStep,
    total_steps: MOCK_TOTAL_STEPS,
    stage: fields.stage,
    clip: clipFields.clip,
    clip_count: clipFields.clip_count,
    is_v2v: job.isV2V,
    joined: job.joinedAt != null,
    created_at: job.createdAt,
    started_at: fields.status === "queued" ? null : job.createdAt,
    completed_at: fields.status === "completed" || fields.status === "failed" || fields.status === "cancelled" ? nowIso() : null,
    error:
      fields.status === "failed"
        ? "GENERATION_FAILED: mock bridge forced failure (prompt contained __MOCK_FAIL__)"
        : null,
    request: echoedRequest,
    result: null,
  };

  if (fields.status === "completed") {
    base.result = {
      video_url: `/api/v1/jobs/${job.id}/video`,
      duration_seconds: Math.round((numFrames / frameRate) * 1000) / 1000,
      resolution: `${width}x${height}`,
      file_size_bytes: 4_096 + numFrames * 512,
      generation_time_seconds: 0.24,
      seed_used: seed === -1 ? 1_000_000 + (job.id.length % 999) : seed,
      output_path: `outputs/${job.id}/output.mp4`,
      metadata_path: `outputs/${job.id}/metadata.json`,
    };
  }

  return base;
}

// --- contract v6 (batch A2V + fs bridge) virtual filesystem -----------------

/** A single entry in a `MockFs` folder — mirrors the shape `fs.listFiles`
 * reports (minus `path`, which is derived by joining the folder path the
 * entry lives under with `name`). */
export interface MockFsFileEntry {
  name: string;
  sizeBytes: number;
  mtimeMs: number;
  /** Wav header duration in seconds, for `fs.listFiles`'s
   * `withAudioDuration`/`fs.probeAudioDuration` fixtures. Only meaningful for
   * `.wav`-named entries; leave unset to simulate a `.wav` native fails to
   * read (both resolve `durationSec: 0`/`isWav: false` rather than erroring,
   * per contract). */
  durationSec?: number;
}

/** Dev/test-only virtual filesystem backing the contract v6 `fs.*` mock
 * methods (`listFiles`/`probeAudioDuration`) and `backend.downloadVideo`'s
 * `destDir`/`noClobber` extension. Construct one with `createMockFs()` to
 * seed initial state and pass it via `MockBridgeOptions.fs`; because the mock
 * mutates the same object in place (e.g. registering each
 * `backend.downloadVideo` write), a test can inspect `folders`/`files`
 * afterward without re-querying the bridge. */
export interface MockFs {
  /** folderPath -> entries directly inside it (non-recursive, matching the
   * native contract). */
  folders: Map<string, MockFsFileEntry[]>;
  /** filePath -> UTF-8 text content. No current `fs.*` mock handler reads or
   * writes this map; kept as part of the virtual filesystem model for
   * callers (e.g. `backend.downloadVideo`'s `reuseIfPresent`) to seed and
   * inspect directly. */
  files: Map<string, string>;
}

/** Builds a `MockFs` from plain-object initial state, for concise test setup
 * (`createMockFs({ folders: { "C:\\in": [...] } })` instead of hand-building
 * `Map`s). */
export function createMockFs(init?: {
  folders?: Record<string, MockFsFileEntry[]>;
  files?: Record<string, string>;
}): MockFs {
  return {
    folders: new Map(Object.entries(init?.folders ?? {})),
    files: new Map(Object.entries(init?.files ?? {})),
  };
}

export interface MockBridgeOptions {
  /** Simulated round-trip latency in ms, applied to every call. Default 200. */
  delayMs?: number;
  /** Plugin version string reported by `ping`. */
  pluginVersion?: string;
  /** When true, `ping` rejects instead of resolving (simulates a dead/unreachable bridge). */
  failPing?: boolean;
  /** When true, `getEditInfo` rejects with NO_EDIT_HANDLE (simulates no open AviUtl2 project). */
  failEditInfo?: boolean;
  /** Overrides the returned edit info payload (merged over the default). */
  editInfo?: Partial<ResultOf<"getEditInfo">>;
  /** When true, every `backend.request` call rejects with BACKEND_UNREACHABLE
   * (simulates the backend HTTP server being down while the native bridge
   * itself is fine). */
  backendUnreachable?: boolean;
  /** Number of "running" polls a job spends before settling on a terminal
   * status. Default 2 (so: queued -> running -> running -> completed). */
  runningPollCount?: number;
  /** When true, `timeline.insertMedia` rejects with NO_EDIT_HANDLE. */
  failInsertMedia?: boolean;
  /** Initial base URL, returned by `backend.getBaseUrl`/`settings.get` until
   * changed via `settings.set` (contract v4, M7b). */
  baseUrl?: string;
  /** When set, `timeline.captureFrame` rejects with this code instead of
   * resolving (contract v3; M4 keyframe panel error-path tests). */
  failCaptureFrame?: "NO_EDIT_HANDLE" | "CAPTURE_FAILED";
  /** When set, `ui.pickFile` rejects with this code instead of resolving
   * (contract v3). `"CANCELLED"` simulates the user dismissing the dialog. */
  failPickFile?: "CANCELLED" | "DIALOG_FAILED";
  /** Overrides the `fileName` (and derived `filePath`) `ui.pickFile` resolves
   * with, when it isn't forced to fail. */
  pickFileName?: string;
  /** When set, `ui.makeThumbnail` rejects with this code instead of resolving
   * (contract v3). */
  failMakeThumbnail?: "FILE_NOT_FOUND" | "THUMBNAIL_FAILED";
  /** When set, `backend.uploadFile` rejects with this code instead of
   * resolving (contract v3). */
  failUploadFile?: "FILE_NOT_FOUND" | "BACKEND_UNREACHABLE" | "BACKEND_TIMEOUT";
  /** 素材（末尾）v2: the `frame_count` a MEASURED `/upload/video` answers with —
   * i.e. one whose query carried `max_frames` (the end-source / reference
   * slots). Defaults to 300. Set `null` to model a server that could not measure
   * the file at all, which is what pushes `useChainForm` onto its
   * duration-based fallback. Uploads WITHOUT `max_frames` always answer `null`
   * regardless of this option, mirroring the real endpoint. */
  uploadVideoFrameCount?: number | null;
  /** 素材（末尾）v2: the `fps` that accompanies {@link uploadVideoFrameCount},
   * under exactly the same conditions. Defaults to 24 (the generation rate), so
   * a default-config test's conversion is the identity. */
  uploadVideoFps?: number | null;
  /** Contract v5: when true, `timeline.getSelection` rejects with
   * NO_EDIT_HANDLE (no open AviUtl2 project). */
  failGetSelection?: boolean;
  /** Contract v5: overrides the returned selection snapshot (merged over the
   * default fixture). */
  selection?: Partial<ResultOf<"timeline.getSelection">>;
  /** Contract v5: when set, `timeline.cutoutRange` rejects with this code. */
  failCutoutRange?: "NO_EDIT_HANDLE" | "CUTOUT_FAILED";
  /** Contract v5: when set, `timeline.extractAudio` rejects with this code. */
  failExtractAudio?: "NO_EDIT_HANDLE" | "EXTRACT_FAILED";
  /** I10 §3-4 #3: overrides the thrown `BridgeError` message when
   * `failExtractAudio` is set, so a test can reproduce native's distinct
   * `EXTRACT_FAILED` reason strings ("…silent range" / "audio render failed at
   * frame N" / a setup failure) that the WebUI classifies into the silence /
   * timeout / generic failure notes. Defaults to the generic mock message. */
  extractAudioErrorMessage?: string;
  /** Contract v5: when true, the provisional-placeholder methods
   * (`insertProvisional`/`resolveProvisional`/`updateProvisionalText`) reject
   * with PROVISIONAL_FAILED. */
  failProvisional?: boolean;
  /** Contract v5: orphan placeholders `timeline.scanProvisionals` reports
   * (default: none). */
  provisionalOrphans?: Array<{ jobId: string; layer: number; frame: number }>;
  /** Contract v6: the virtual filesystem backing `fs.listFiles`/
   * `fs.probeAudioDuration` and `backend.downloadVideo`'s `destDir`/
   * `noClobber` extension. Defaults to a fresh empty `MockFs` when omitted —
   * pass one built with `createMockFs()` to seed initial folders/files. */
  fs?: MockFs;
  /** Contract v6: when set, `ui.pickFolder` rejects with this code instead of
   * resolving. `"CANCELLED"` simulates the user dismissing the dialog. */
  failPickFolder?: "CANCELLED" | "DIALOG_FAILED";
  /** Contract v6: overrides the `folderPath` `ui.pickFolder` resolves with,
   * when it isn't forced to fail. */
  pickFolderPath?: string;
  /** Test-determinism knob: when true, `backend.uploadFile` calls don't
   * resolve after `delayMs` like every other call does — they block
   * indefinitely until the test calls the returned bridge's
   * `releaseUploads()`. This lets a test observe the "uploading" in-flight
   * state deterministically (no real-timer race) instead of picking a
   * `delayMs` and hoping a `waitFor` lands inside the window. Every other
   * bridge call (including `ui.pickFile`, which normally precedes an upload)
   * keeps its ordinary `delayMs`-based timing. Defaults to false, in which
   * case `backend.uploadFile` behaves exactly as before this option existed. */
  holdUploads?: boolean;
  /** Contract v9: the `durationSec` `fs.probeMediaInfo` resolves with (for the
   * IC-LoRA/A2V source cards' `12.3s` readout). Omitted -> `0` (the "unknown"
   * value native reports when there's no edit handle / `get_media_info` fails).
   * `probeMediaInfoWidth`/`probeMediaInfoHeight` likewise default to `0`. */
  probeMediaInfoDurationSec?: number;
  probeMediaInfoWidth?: number;
  probeMediaInfoHeight?: number;
  /** Contract v7: when true, `requestWithFiles(..., "ui.resolveDroppedFiles", ...)`
   * rejects instead of resolving (simulates the native drop-resolution round
   * trip failing — `useFileDrop`'s `dnd.resolveFailed` error path). Resolving
   * to an empty `files: []` (the other failure shape `useFileDrop` must
   * handle) needs no option — it happens naturally when the dropped `files`
   * array itself is empty or contains no `.name`-bearing entries. */
  failResolveDroppedFiles?: boolean;
  /** V2V Join: the `source_normalized` flag `POST /jobs/{id}/join` returns
   * (default true, matching the previous fixed response). Set false to model a
   * join where the source needed no re-encode. */
  joinSourceNormalized?: boolean;
  /** V2V Join: the `source_fps` `POST /jobs/{id}/join` returns (default 24,
   * matching the previous fixed response). `null` models an unprobeable source. */
  joinSourceFps?: number | null;
}

/** Contract v5 default `timeline.getSelection` snapshot — a single selected
 * video object spanning frames 0..120 on layer 1, with project rate/scale/
 * sampleRate matching `DEFAULT_EDIT_INFO` so frame<->time math lines up.
 * `mediaWidth`/`mediaHeight` carry the object's real resolution (a video here,
 * so a concrete 1920×1080); an audio/shape object would report `0` (unknown). */
const DEFAULT_SELECTION: ResultOf<"timeline.getSelection"> = {
  hasRange: true,
  rangeStart: 0,
  rangeEnd: 120,
  selected: [
    {
      layer: 1,
      frameStart: 0,
      frameEnd: 120,
      effectName: "動画ファイル",
      filePath: "C:\\Users\\mock\\Videos\\clip.mp4",
      objectName: "clip",
      textContent: null,
      mediaWidth: 1920,
      mediaHeight: 1080,
      mediaDurationSec: 0,
      // Contract v10 (§1-6). The defaults mirror native's: no playback range
      // read (so `decideSourceTrim` skips and the upload stays whole-file),
      // neutral speed, no loop, a single section. A test that wants a trim to
      // fire overrides these through `MockBridgeOptions.selection`.
      playbackStartSec: 0,
      playbackEndSec: 0,
      hasPlaybackRange: false,
      playbackSpeed: 1,
      loopPlay: false,
      sectionCount: 1,
    },
  ],
  cursorFrame: DEFAULT_EDIT_INFO.frame,
  cursorLayer: 1,
  rate: DEFAULT_EDIT_INFO.rate,
  scale: DEFAULT_EDIT_INFO.scale,
  sampleRate: DEFAULT_EDIT_INFO.sampleRate,
};

/** Dev/test-only extension of `NativeBridge`: `emit` lets a test drive a
 * native-initiated event (e.g. `timeline.menuInvoked`) through the same `on`
 * subscription path the real WebView2 bridge uses. */
export interface MockBridge extends NativeBridge {
  emit(event: string, data: unknown): void;
  /** Test-determinism knob paired with `MockBridgeOptions.holdUploads`:
   * resolves the pending `backend.uploadFile` gate, letting any in-flight
   * (and all future) upload calls proceed. A harmless no-op if
   * `holdUploads` wasn't set. */
  releaseUploads(): void;
}

/** 1x1 transparent PNG, base64-encoded — the mock's canned `ui.makeThumbnail`
 * output (real thumbnails come from native's WIC-based downscale; the mock
 * only needs to exercise the data-URL-shaped contract, not real pixels). */
/** 素材（末尾）: what a MEASURED `/upload/video` reports when a test doesn't say
 * otherwise — 300 frames at 24fps, i.e. a comfortable 12.5-second material.
 * Since the window-internal mode (2026-08-17) THERE IS NO BAND DERIVATION any
 * more: the anchor is a fixed 8 frames and the only length rule left is the
 * 9-frame minimum (8 + the causal VAE's primer). 300 stays because it is an
 * ordinary material that clears that minimum with room to spare, keeping the
 * out-of-the-box dev experience (drag a video onto the end card) the valid one. */
const DEFAULT_UPLOAD_VIDEO_FRAME_COUNT = 300;
const DEFAULT_UPLOAD_VIDEO_FPS = 24;

const MOCK_THUMBNAIL_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

/**
 * Dev-time stand-in for the native bridge, used when the app runs in a plain
 * browser (no `window.chrome.webview`), e.g. `npm run dev` or unit tests.
 * Mirrors contract v1/v2's success/error shapes but skips the postMessage
 * transport entirely — it resolves/rejects promises directly.
 *
 * `backend.request` is a self-contained fixture store (queued->running->
 * completed job simulation, `/status`/`/config` fixtures, a `JOB_BUSY` 409
 * while a job is in flight) — it never calls the real backend, so `npm run
 * dev` works with no backend process running at all.
 */
export function createMockBridge(options: MockBridgeOptions = {}): MockBridge {
  const delayMs = options.delayMs ?? 200;
  const pluginVersion = options.pluginVersion ?? "0.0.0-mock";
  const runningPolls = options.runningPollCount ?? 2;

  const jobs = new Map<string, MockJobRecord>();
  let activeJobId: string | null = null;
  let jobCounter = 0;
  /** Model management (S1): the NAME used by the last successful
   * `/pipeline/load` per category, mirroring `PipelineManager.active_models`
   * server-side. All-default until a `POST /pipeline/load` with a `models`
   * block succeeds. */
  const activeModels: Record<MockModelCategory, string> = {
    transformer: "default",
    text_encoder: "default",
    video_vae: "default",
    audio: "default",
  };
  /** Mutable backend URL, seeded from `options.baseUrl` — `settings.set`
   * (contract v4, M7b) updates this in place, and `backend.getBaseUrl`
   * reflects the current value so a saved settings change is immediately
   * visible to anything building a direct `<video src>`/`<img src>` URL. */
  let currentBaseUrl = options.baseUrl ?? "http://127.0.0.1:18620";
  /** Contract v6: virtual filesystem backing `fs.*`/`backend.downloadVideo`'s
   * `destDir`/`noClobber` extension. Defaults to a fresh empty `MockFs` when
   * `options.fs` is omitted. */
  const mockFs: MockFs = options.fs ?? createMockFs();

  const wait = (): Promise<void> =>
    new Promise((resolve) => setTimeout(resolve, delayMs));

  /** Test-determinism knob: when `options.holdUploads` is set, this promise
   * never settles on its own — only `releaseUploads()` (exposed on the
   * returned bridge, below) resolves it. `request()` awaits it instead of
   * `wait()` for `backend.uploadFile` calls specifically, so a test can hold
   * an upload open indefinitely rather than racing a real setTimeout. */
  let resolveUploadGate: (() => void) | null = null;
  const uploadGate: Promise<void> | null = options.holdUploads
    ? new Promise<void>((resolve) => {
        resolveUploadGate = resolve;
      })
    : null;

  async function handlePing(): Promise<ResultOf<"ping">> {
    if (options.failPing) {
      throw new BridgeError(
        "NO_EDIT_HANDLE",
        "Mock bridge: ping forced to fail (failPing option set)",
      );
    }
    return { pong: true, pluginVersion };
  }

  async function handleGetEditInfo(): Promise<ResultOf<"getEditInfo">> {
    if (options.failEditInfo) {
      throw new BridgeError(
        "NO_EDIT_HANDLE",
        "Mock bridge: no active AviUtl2 edit session",
      );
    }
    return { ...DEFAULT_EDIT_INFO, ...options.editInfo };
  }

  function handleStatus(): ResultOf<"backend.request"> {
    const active = activeJobId ? jobs.get(activeJobId) : undefined;
    const activeNonTerminal =
      active !== undefined && isNonTerminal(deriveJobFields(active.pollCount, runningPolls, active.forcedTerminal).status);

    let completed = 0;
    let failed = 0;
    for (const job of jobs.values()) {
      const status = deriveJobFields(job.pollCount, runningPolls, job.forcedTerminal).status;
      if (status === "completed") completed += 1;
      if (status === "failed" || status === "cancelled") failed += 1;
    }

    return {
      status: 200,
      body: {
        server: "running",
        version: "0.4.0-mock",
        host: "127.0.0.1",
        port: 18620,
        pipeline_loaded: true,
        pipeline_type: "distilled",
        gpu: {
          available: true,
          name: "NVIDIA GeForce RTX 4070 Ti SUPER (mock)",
          vram_total_mb: 16375,
          vram_used_mb: 1288,
          vram_free_mb: 15087,
        },
        vram_optimization: {
          low_vram_mode: true,
          low_vram_profile: "16gb_safe",
          fp8_transformer: true,
          cpu_offload_text_encoder: true,
          vae_tiling: true,
          attention_tiling: false,
          block_swap: true,
          low_vram_disabled_required: false,
        },
        queue: {
          mode: "single_job_in_memory",
          pending: 0,
          running: activeNonTerminal ? 1 : 0,
          completed,
          failed,
        },
        // Acceleration capability block (backend §43, 2026-07-31). The mock
        // reports sage as AVAILABLE — that is the interesting state for the
        // UI (the Settings panel's sage button is selectable), and without it
        // here every unit test would only ever exercise the "unknown" branch
        // of `sageAvailability`. Tests that need the unavailable state
        // construct their own /status body.
        acceleration: {
          attention_backends: ["sdpa", "sage"],
          sage_available: true,
        },
      },
    };
  }

  function handleConfig(): ResultOf<"backend.request"> {
    return { status: 200, body: MOCK_CONFIG_BODY };
  }

  function handleLoras(): ResultOf<"backend.request"> {
    return { status: 200, body: { loras: MOCK_LORAS } };
  }

  function handleLorasReload(): ResultOf<"backend.request"> {
    const styles = MOCK_LORAS.filter((l) => l.kind === "style").length;
    const controls = MOCK_LORAS.filter((l) => l.kind === "control").length;
    return { status: 200, body: { total: MOCK_LORAS.length, styles, controls } };
  }

  /** `GET /models` fixture (Docs/API_REFERENCE.md §3.5). Always rescans in
   * spirit (the fixture list is static, but `active` reflects whatever the
   * last successful `/pipeline/load` set), matching the real endpoint's
   * "rescans on every call" contract closely enough for UI testing. */
  function handleModels(): ResultOf<"backend.request"> {
    const categories = Object.fromEntries(
      MOCK_MODEL_CATEGORIES.map((category) => [
        category,
        {
          default: "default",
          active: activeModels[category],
          entries: MOCK_MODEL_ENTRIES[category],
        },
      ]),
    );
    return { status: 200, body: { categories } };
  }

  /** `POST /pipeline/load` fixture (Docs/API_REFERENCE.md §3.3,
   * `api/pipeline.py:35-93`). No/empty `models` block is the legacy bodyless
   * path (response omits `models`); a non-empty block validates each
   * category/name against the `GET /models` fixture (`MODEL_NOT_FOUND` for
   * an unknown category/name, `MODEL_FILE_MISSING` for a registered name
   * whose fixture entry has `exists: false`) before committing it to
   * `activeModels`, mirroring the server's fail-loud resolve. Swapping while
   * a job is active mirrors `handleGenerate`'s `JOB_BUSY` check
   * (`context.job_store.has_active()` server-side). */
  function handlePipelineLoad(body: object | undefined): ResultOf<"backend.request"> {
    const req = (body ?? {}) as Record<string, unknown>;
    const requested = (req.models ?? {}) as Record<string, string>;

    if (Object.keys(requested).length === 0) {
      // Legacy bodyless load: response shape unchanged, no `models` key.
      return { status: 200, body: { pipeline_loaded: true, pipeline_type: "distilled", state: "loaded" } };
    }

    if (activeJobId) {
      const active = jobs.get(activeJobId);
      if (active && isNonTerminal(deriveJobFields(active.pollCount, runningPolls, active.forcedTerminal).status)) {
        return {
          status: 409,
          body: { error: { code: "JOB_BUSY", message: "cannot swap models while a job is running" } },
        };
      }
    }

    for (const [category, name] of Object.entries(requested)) {
      if (!(MOCK_MODEL_CATEGORIES as readonly string[]).includes(category)) {
        return {
          status: 404,
          body: { error: { code: "MODEL_NOT_FOUND", message: `unknown category '${category}'` } },
        };
      }
      const entries = MOCK_MODEL_ENTRIES[category as MockModelCategory];
      const entry = entries.find((e) => e.name === name);
      if (name !== "default" && !entry) {
        return {
          status: 404,
          body: { error: { code: "MODEL_NOT_FOUND", message: `unknown model name '${name}' in category '${category}'` } },
        };
      }
      if (entry && !entry.exists) {
        return {
          status: 422,
          body: {
            error: {
              code: "MODEL_FILE_MISSING",
              message: `registered model file for '${category}/${name}' is missing on disk`,
            },
          },
        };
      }
    }

    for (const [category, name] of Object.entries(requested)) {
      activeModels[category as MockModelCategory] = name;
    }

    return {
      status: 200,
      body: { pipeline_loaded: true, pipeline_type: "distilled", state: "loaded", models: { ...activeModels } },
    };
  }

  /** `POST /pipeline/unload` fixture (N4 "danger zone"). Reuses
   * `handlePipelineLoad`'s active-job guard verbatim — unloading out from
   * under a running job would be just as unsafe as swapping models under it
   * — and otherwise always reports the engine as torn down. Deliberately
   * stateless beyond that guard (no persisted "is it actually loaded right
   * now" flag exists in this fixture store, mirroring `handlePipelineLoad`'s
   * own bodyless-path simplicity). */
  function handlePipelineUnload(): ResultOf<"backend.request"> {
    if (activeJobId) {
      const active = jobs.get(activeJobId);
      if (active && isNonTerminal(deriveJobFields(active.pollCount, runningPolls, active.forcedTerminal).status)) {
        return {
          status: 409,
          body: { error: { code: "JOB_BUSY", message: "cannot unload the pipeline while a job is running" } },
        };
      }
    }

    return { status: 200, body: { pipeline_loaded: false, state: "unloaded" } };
  }

  function validationError(loc: string, msg: string): ResultOf<"backend.request"> {
    return {
      status: 422,
      body: {
        error: {
          code: "VALIDATION_ERROR",
          message: "Request validation failed",
          detail: [{ loc: ["body", loc], msg, type: "value_error" }],
        },
      },
    };
  }

  function handleGenerate(body: object | undefined): ResultOf<"backend.request"> {
    if (activeJobId) {
      const active = jobs.get(activeJobId);
      if (active && isNonTerminal(deriveJobFields(active.pollCount, runningPolls, active.forcedTerminal).status)) {
        return {
          status: 409,
          body: { error: { code: "JOB_BUSY", message: "Another job is already running" } },
        };
      }
    }

    const req = (body ?? {}) as Record<string, unknown>;
    const prompt = typeof req.prompt === "string" ? req.prompt : "";
    if (prompt.length < 1 || prompt.length > 2000) {
      return validationError("prompt", "String should have at least 1 character");
    }
    const width = req.width;
    if (typeof width === "number" && (width % 64 !== 0 || width < 256 || width > 4096)) {
      return validationError("width", "width must be a multiple of 64 in [256, 4096]");
    }
    const height = req.height;
    if (typeof height === "number" && (height % 64 !== 0 || height < 128 || height > 4096)) {
      return validationError("height", "height must be a multiple of 64 in [128, 4096]");
    }
    const numFrames = req.num_frames;
    if (typeof numFrames === "number" && ((numFrames - 1) % 8 !== 0 || numFrames < 9 || numFrames > 481)) {
      return validationError("num_frames", "num_frames must be 8n+1 in [9, 481]");
    }
    // NAG (2026-07-28): mirrors the real backend's 422 for "enabled but no
    // negative prompt" (`shell/nagSettings.ts`'s `isNagNegativeEmpty` is the
    // client-side pre-check for the same rule; this is the server-side echo).
    if (req.nag_enabled === true && String(req.negative_prompt ?? "").trim() === "") {
      return validationError("negative_prompt", "negative_prompt must not be empty when nag_enabled is true");
    }

    jobCounter += 1;
    const id = `mock-job-${jobCounter}-${Math.random().toString(36).slice(2, 8)}`;
    const forcedTerminal = prompt.includes("__MOCK_FAIL__")
      ? "failed"
      : prompt.includes("__MOCK_CANCEL__")
        ? "cancelled"
        : undefined;
    const record: MockJobRecord = {
      id,
      request: req,
      createdAt: nowIso(),
      pollCount: 0,
      isV2V: false,
      joinedAt: null,
      ...(forcedTerminal ? { forcedTerminal } : {}),
    };
    jobs.set(id, record);
    activeJobId = id;

    return { status: 202, body: { job_id: id, status: "queued", created_at: record.createdAt } };
  }

  /** M6: `POST /generate/chain` fixture (Docs/API_REFERENCE.md §3.13). Keeps
   * validation intentionally light (prompt + clip-count shape only) — the
   * WebUI's own client-side validation (`modes/chained/chainUtils.ts`) is what
   * the M6 task brief asks to unit-test in depth; this fixture only needs to
   * prove the 202/`num_clips` contract and drive a believable clip/clip_count
   * progression via `deriveClipFields` above. */
  function handleGenerateChain(body: object | undefined): ResultOf<"backend.request"> {
    if (activeJobId) {
      const active = jobs.get(activeJobId);
      if (active && isNonTerminal(deriveJobFields(active.pollCount, runningPolls, active.forcedTerminal).status)) {
        return {
          status: 409,
          body: { error: { code: "JOB_BUSY", message: "Another job is already running" } },
        };
      }
    }

    const req = (body ?? {}) as Record<string, unknown>;
    const prompt = typeof req.prompt === "string" ? req.prompt : "";
    if (prompt.length < 1 || prompt.length > 2000) {
      return validationError("prompt", "String should have at least 1 character");
    }
    const clips = Array.isArray(req.clips) ? req.clips : [];
    if (clips.length < 1 || clips.length > 24) {
      return validationError("clips", "clips must have between 1 and 24 items");
    }
    const hasSource = req.source_video != null || req.source_audio != null;
    if (!hasSource && clips.length < 2) {
      return validationError("clips", "at least 2 clips are required when no source_video/source_audio is set");
    }
    // NAG (2026-07-28): mirrors the real backend's 422 for "enabled but no
    // negative prompt" (`shell/nagSettings.ts`'s `isNagNegativeEmpty` is the
    // client-side pre-check for the same rule; this is the server-side echo).
    if (req.nag_enabled === true && String(req.negative_prompt ?? "").trim() === "") {
      return validationError("negative_prompt", "negative_prompt must not be empty when nag_enabled is true");
    }

    jobCounter += 1;
    const id = `mock-chain-job-${jobCounter}-${Math.random().toString(36).slice(2, 8)}`;
    const forcedTerminal = prompt.includes("__MOCK_FAIL__")
      ? "failed"
      : prompt.includes("__MOCK_CANCEL__")
        ? "cancelled"
        : undefined;
    const record: MockJobRecord = {
      id,
      request: req,
      createdAt: nowIso(),
      pollCount: 0,
      numClips: clips.length,
      // V2V iff a source_video spec was submitted (mirrors the real server's
      // JobResponse.is_v2v derivation from the chain request). A2V (source_audio)
      // and plain multi-clip chains are not V2V.
      isV2V: req.source_video != null,
      joinedAt: null,
      ...(forcedTerminal ? { forcedTerminal } : {}),
    };
    jobs.set(id, record);
    activeJobId = id;

    return {
      status: 202,
      body: { job_id: id, status: "queued", created_at: record.createdAt, num_clips: clips.length },
    };
  }

  /** M6/I4: `POST /jobs/{id}/join` fixture (Docs/API_REFERENCE.md §3.17). A
   * 200 for any known job (the real backend's `422 JOB_NOT_JOINABLE` for a
   * non-V2V job isn't reproduced — the WebUI only ever shows the Join control
   * for an `is_v2v` job via `jobs/joinUtils.ts`). Records `joinedAt` so
   * subsequent `JobResponse.joined` reads and the `GET /jobs/{id}/joined` route
   * flip to "ready". Honors the I4 body fields (`source_tail_seconds` /
   * `handle_crossfade_ms`) and returns the I3 `JoinResponse` shape
   * (`trimmed_source_seconds` / `source_fps`); the trim value is a plausible
   * pseudo-computation from an assumed source length, not a real measurement. */
  function handleJoinJob(jobId: string, body: object | undefined): ResultOf<"backend.request"> {
    const job = jobs.get(jobId);
    if (!job) return jobNotFound(jobId);
    const req = (body ?? {}) as Record<string, unknown>;
    const sourceTailSeconds = typeof req.source_tail_seconds === "number" ? req.source_tail_seconds : 5.0;
    const crossfadeMs = typeof req.handle_crossfade_ms === "number" ? req.handle_crossfade_ms : 300;
    job.joinedAt = nowIso();
    // Pseudo tail-keep trim: pretend the source ran ~12s and we kept the last
    // `source_tail_seconds`, so the continuation begins at max(0, 12 - tail).
    const ASSUMED_SOURCE_SECONDS = 12.0;
    const trimmedSourceSeconds =
      sourceTailSeconds > 0
        ? Math.max(0, Math.round((ASSUMED_SOURCE_SECONDS - sourceTailSeconds) * 1000) / 1000)
        : 0;
    return {
      status: 200,
      body: {
        job_id: jobId,
        joined_path: `outputs/${jobId}/joined.mp4`,
        join_mode: crossfadeMs > 0 ? "handle_crossfade" : "hard_concat",
        source_normalized: options.joinSourceNormalized ?? true,
        trimmed_source_seconds: trimmedSourceSeconds,
        source_fps: options.joinSourceFps === undefined ? 24 : options.joinSourceFps,
      },
    };
  }

  /** I4: `GET /jobs/{id}/joined` fixture (Docs/API_REFERENCE.md §3.18,
   * `api/jobs.py`). 200 once a join has run (`joinedAt` set), else 404
   * `JOINED_NOT_READY`; 404 `JOB_NOT_FOUND` for an unknown job. Like
   * `GET /jobs/{id}/video`, the mock serves no bytes here (the WebUI reaches
   * the joined mp4 via a direct `<video src>` against the backend origin, and
   * downloads it through native `backend.downloadVideo({joined:true})`, never
   * through this proxy path) — only the status/shape are modelled. */
  function handleGetJoined(jobId: string): ResultOf<"backend.request"> {
    const job = jobs.get(jobId);
    if (!job) return jobNotFound(jobId);
    if (job.joinedAt == null) {
      return {
        status: 404,
        body: {
          error: {
            code: "JOINED_NOT_READY",
            message: `joined video has not been created yet (POST /jobs/${jobId}/join first)`,
          },
        },
      };
    }
    return { status: 200, body: null };
  }

  function jobNotFound(jobId: string): ResultOf<"backend.request"> {
    return { status: 404, body: { error: { code: "JOB_NOT_FOUND", message: `Unknown job_id "${jobId}"` } } };
  }

  function handleGetJob(jobId: string): ResultOf<"backend.request"> {
    const job = jobs.get(jobId);
    if (!job) return jobNotFound(jobId);

    job.pollCount += 1;
    const fields = deriveJobFields(job.pollCount, runningPolls, job.forcedTerminal);
    if (!isNonTerminal(fields.status) && activeJobId === job.id) {
      activeJobId = null;
    }
    return { status: 200, body: jobResponseBody(job, fields, runningPolls) };
  }

  function handleListJobs(): ResultOf<"backend.request"> {
    // GET /jobs returns a bare JSON array (Docs/API_REFERENCE.md §3.14), not
    // wrapped in an envelope object.
    //
    // Unlike the real backend (where job progress is driven by an actual
    // async worker, independent of who's polling), this fixture's
    // queued->running->completed progression is a stand-in driven entirely
    // by *how many times a job has been polled* (`pollCount`). M3's job
    // rail polls only this endpoint (never `GET /jobs/{id}`) so it can see
    // real progress on its own — without also running a redundant
    // per-job poll — this also advances each listed job's simulated clock,
    // exactly like `GET /jobs/{id}` does. This is additive: nothing in the
    // M1/M2 test suite calls `GET /jobs`, so existing `GET /jobs/{id}`-only
    // flows are unaffected.
    const body = Array.from(jobs.values()).map((job) => {
      job.pollCount += 1;
      const fields = deriveJobFields(job.pollCount, runningPolls, job.forcedTerminal);
      if (!isNonTerminal(fields.status) && activeJobId === job.id) {
        activeJobId = null;
      }
      return jobResponseBody(job, fields, runningPolls);
    });
    return { status: 200, body: body as unknown as object };
  }

  function handleDeleteJob(jobId: string): ResultOf<"backend.request"> {
    const job = jobs.get(jobId);
    if (!job) return jobNotFound(jobId);

    const fields = deriveJobFields(job.pollCount, runningPolls, job.forcedTerminal);
    if (isNonTerminal(fields.status)) {
      job.forcedTerminal = "cancelled";
      return { status: 200, body: { job_id: jobId, cancel_requested: true, status: fields.status } };
    }

    jobs.delete(jobId);
    if (activeJobId === jobId) activeJobId = null;
    return { status: 200, body: { job_id: jobId, deleted: true } };
  }

  const JOB_PATH = /^\/api\/v1\/jobs\/(?<id>[^/]+)$/;
  const JOB_JOIN_PATH = /^\/api\/v1\/jobs\/(?<id>[^/]+)\/join$/;
  const JOB_JOINED_PATH = /^\/api\/v1\/jobs\/(?<id>[^/]+)\/joined$/;

  async function handleBackendRequest(params: ParamsOf<"backend.request">): Promise<ResultOf<"backend.request">> {
    if (options.backendUnreachable) {
      throw new BridgeError(
        "BACKEND_UNREACHABLE",
        "Mock bridge: backend forced unreachable (backendUnreachable option set)",
      );
    }

    const { method, path, body } = params;

    if (method === "GET" && path === "/api/v1/status") return handleStatus();
    if (method === "GET" && path === "/api/v1/config") return handleConfig();
    if (method === "POST" && path === "/api/v1/generate") return handleGenerate(body);
    if (method === "POST" && path === "/api/v1/generate/chain") return handleGenerateChain(body);
    if (method === "GET" && path === "/api/v1/jobs") return handleListJobs();
    if (method === "GET" && path === "/api/v1/loras") return handleLoras();
    if (method === "POST" && path === "/api/v1/loras/reload") return handleLorasReload();
    if (method === "GET" && path === "/api/v1/models") return handleModels();
    if (method === "POST" && path === "/api/v1/pipeline/load") return handlePipelineLoad(body);
    if (method === "POST" && path === "/api/v1/pipeline/unload") return handlePipelineUnload();

    const joinId = JOB_JOIN_PATH.exec(path)?.groups?.id;
    if (joinId && method === "POST") return handleJoinJob(joinId, body);

    const joinedId = JOB_JOINED_PATH.exec(path)?.groups?.id;
    if (joinedId && method === "GET") return handleGetJoined(joinedId);

    const jobId = JOB_PATH.exec(path)?.groups?.id;
    if (jobId && method === "GET") return handleGetJob(jobId);
    if (jobId && method === "DELETE") return handleDeleteJob(jobId);

    return {
      status: 404,
      body: { error: { code: "NOT_FOUND", message: `Mock bridge: no fixture for ${method} ${path}` } },
    };
  }

  async function handleDownloadVideo(params: ParamsOf<"backend.downloadVideo">): Promise<ResultOf<"backend.downloadVideo">> {
    const job = jobs.get(params.jobId);
    if (!job) {
      throw new BridgeError("DOWNLOAD_FAILED", `Mock bridge: unknown job_id "${params.jobId}"`);
    }
    const fields = deriveJobFields(job.pollCount, runningPolls, job.forcedTerminal);
    if (fields.status !== "completed") {
      throw new BridgeError(
        "DOWNLOAD_FAILED",
        `Mock bridge: job "${params.jobId}" is not completed (status=${fields.status})`,
      );
    }
    const suffix = params.joined ? "joined" : "output";
    // Contract v6: destDir/fileName/noClobber. Omitting all three reproduces
    // the exact pre-v6 path/behavior.
    const destDir = params.destDir ?? "C:\\Users\\mock\\AppData\\Local\\Temp\\Nz-LTX23";
    const desiredName = params.fileName ?? `${job.id}-${suffix}.mp4`;
    const sizeBytes = 4_096 + 512;
    const entries = mockFs.folders.get(destDir) ?? [];
    const finalName = params.noClobber ? dedupeFileName(entries, desiredName) : desiredName;
    const writtenEntry: MockFsFileEntry = { name: finalName, sizeBytes, mtimeMs: Date.now() };
    const existingIdx = entries.findIndex((e) => e.name === finalName);
    if (existingIdx >= 0) {
      // Non-noClobber re-download of the same name: simulate an overwrite in
      // place rather than a duplicate folder entry.
      entries[existingIdx] = writtenEntry;
    } else {
      entries.push(writtenEntry);
    }
    mockFs.folders.set(destDir, entries);
    return {
      filePath: joinMockPath(destDir, finalName),
      sizeBytes,
    };
  }

  async function handleInsertMedia(params: ParamsOf<"timeline.insertMedia">): Promise<ResultOf<"timeline.insertMedia">> {
    if (options.failInsertMedia) {
      throw new BridgeError(
        "NO_EDIT_HANDLE",
        "Mock bridge: insertMedia forced to fail (failInsertMedia option set)",
      );
    }
    return {
      inserted: true,
      layer: params.layer ?? 1,
      frame: params.frame ?? 0,
    };
  }

  /** Replace-insert (the 🎞 "place & replace"): when a provisional marker for
   * `jobId` is present in the table, remove it and report `mode:"replaced"` at
   * the marker's recorded position — modelling the finished media taking the
   * placeholder's slot (the marker disappears from `scanProvisionals`, and the
   * video lands where it was). When absent, behave like `insertMedia`
   * (`mode:"inserted"`, cursor fallback layer 1 / frame 0). `usedFallback` is
   * always false in the mock: the marker-slot collision -> `layer_max+1` retreat
   * is a real-device-only path with no timeline geometry to model here. The
   * `provisionals` Map is declared lower down (function hoisting keeps this
   * closure valid). */
  async function handleInsertMediaForJob(
    params: ParamsOf<"timeline.insertMediaForJob">,
  ): Promise<ResultOf<"timeline.insertMediaForJob">> {
    if (options.failInsertMedia) {
      throw new BridgeError(
        "NO_EDIT_HANDLE",
        "Mock bridge: insertMediaForJob forced to fail (failInsertMedia option set)",
      );
    }
    const marker = provisionals.get(params.jobId);
    if (marker) {
      provisionals.delete(params.jobId);
      return { ok: true, mode: "replaced", layer: marker.layer, frame: marker.frame, usedFallback: false };
    }
    return { ok: true, mode: "inserted", layer: 1, frame: 0, usedFallback: false };
  }

  async function handleGetBaseUrl(): Promise<ResultOf<"backend.getBaseUrl">> {
    return { baseUrl: currentBaseUrl };
  }

  /** Strips a trailing slash so `${baseUrl}${API_PREFIX}/...` concatenation
   * (`api/client.ts`) never ends up with a doubled slash. */
  function normalizeBaseUrl(raw: string): string {
    return raw.trim().replace(/\/+$/, "");
  }

  async function handleSettingsGet(): Promise<ResultOf<"settings.get">> {
    return { baseUrl: currentBaseUrl };
  }

  /** `settings.set` (contract v4, M7b's connection-settings panel): omitting
   * `baseUrl` is a no-op read; an unparseable URL or a non-http(s) scheme
   * rejects with `BAD_REQUEST`, matching what the settings panel needs to
   * show inline (task brief: "BAD_REQUESTはインラインエラー表示"). */
  async function handleSettingsSet(params: ParamsOf<"settings.set">): Promise<ResultOf<"settings.set">> {
    if (params.baseUrl !== undefined) {
      const trimmed = params.baseUrl.trim();
      let parsed: URL;
      try {
        parsed = new URL(trimmed);
      } catch {
        throw new BridgeError("BAD_REQUEST", `Mock bridge: invalid backend URL "${params.baseUrl}"`);
      }
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
        throw new BridgeError("BAD_REQUEST", `Mock bridge: backend URL must be http(s) ("${params.baseUrl}")`);
      }
      currentBaseUrl = normalizeBaseUrl(trimmed);
    }
    return { baseUrl: currentBaseUrl };
  }

  let captureCounter = 0;

  async function handleCaptureFrame(
    params: ParamsOf<"timeline.captureFrame">,
  ): Promise<ResultOf<"timeline.captureFrame">> {
    if (options.failCaptureFrame) {
      throw new BridgeError(
        options.failCaptureFrame,
        `Mock bridge: timeline.captureFrame forced to fail (${options.failCaptureFrame})`,
      );
    }
    captureCounter += 1;
    const frame = params.frame ?? DEFAULT_EDIT_INFO.frame;
    const width = options.editInfo?.width ?? DEFAULT_EDIT_INFO.width;
    const height = options.editInfo?.height ?? DEFAULT_EDIT_INFO.height;
    return {
      filePath: `C:\\Users\\mock\\AppData\\Local\\Temp\\Nz-LTX23\\frame_${frame}_${captureCounter}.png`,
      width,
      height,
      frame,
    };
  }

  let pickFileCounter = 0;

  async function handlePickFile(params: ParamsOf<"ui.pickFile">): Promise<ResultOf<"ui.pickFile">> {
    if (options.failPickFile) {
      throw new BridgeError(
        options.failPickFile,
        options.failPickFile === "CANCELLED"
          ? "Mock bridge: ui.pickFile forced CANCELLED (user dismissed the dialog)"
          : "Mock bridge: ui.pickFile forced to fail (DIALOG_FAILED)",
      );
    }
    pickFileCounter += 1;
    // "imageOrVideo" (Chain's unified source-input picker) has no unambiguous
    // default extension when the caller doesn't override `pickFileName` —
    // default to an image ("png") like the plain "image" kind. Tests that need
    // deterministic image/video routing always pass `pickFileName` explicitly
    // (e.g. "clip.mp4") rather than relying on this default.
    const ext = params.kind === "video" ? "mp4" : params.kind === "audio" ? "wav" : "png";
    const fileName = options.pickFileName ?? `picked-${params.kind}-${pickFileCounter}.${ext}`;
    return {
      filePath: `C:\\Users\\mock\\Pictures\\${fileName}`,
      fileName,
    };
  }

  async function handleMakeThumbnail(
    params: ParamsOf<"ui.makeThumbnail">,
  ): Promise<ResultOf<"ui.makeThumbnail">> {
    if (options.failMakeThumbnail) {
      throw new BridgeError(
        options.failMakeThumbnail,
        `Mock bridge: ui.makeThumbnail forced to fail (${options.failMakeThumbnail})`,
      );
    }
    const maxDim = params.maxDim ?? 256;
    return {
      dataUrl: MOCK_THUMBNAIL_DATA_URL,
      width: Math.min(1, maxDim),
      height: Math.min(1, maxDim),
      sourceWidth: 1,
      sourceHeight: 1,
    };
  }

  let uploadCounter = 0;

  /** Contract v10 (§1-6): did this upload actually ask for a range trim? Both
   * `trim_start_sec` AND `trim_duration_sec` must be present — the server
   * treats a lone parameter as "no trim requested" and passes the file
   * through untouched, so the mock does too. */
  function hasTrimQuery(query: Record<string, string> | undefined): boolean {
    if (!query) return false;
    return typeof query["trim_start_sec"] === "string" && typeof query["trim_duration_sec"] === "string";
  }

  /** Mirrors `POST /upload/{kind}`'s response shape (Docs/API_REFERENCE.md
   * §3.9-3.11) without touching the real mock backend process — the native
   * bridge's `backend.uploadFile` streams the file there itself; this fixture
   * stands in for that whole round trip so `npm run dev`/tests never need a
   * real file on disk. */
  function uploadResponseBody(
    kind: "image" | "video" | "audio",
    filePath: string,
    query?: Record<string, string> | undefined,
  ): Record<string, unknown> {
    const fileName = fileNameFromPathMock(filePath);
    uploadCounter += 1;
    const id = `mock-${kind}-${uploadCounter}`;
    if (kind === "image") {
      return {
        image_id: id,
        original_filename: fileName,
        stored_path: `uploads/${id}/input.png`,
        width: 512,
        height: 512,
        content_type: "image/png",
      };
    }
    if (kind === "video") {
      // 素材（末尾）v2 (2026-08-15): the real `/upload/video` measures the stored
      // file's frame count / fps ONLY for uploads that asked for a `max_frames`
      // cap (the end-source and reference slots) — a plain upload pays for no
      // extra probe and answers `null` for both. The mock mirrors exactly that
      // rule, so a test (and `npm run dev`) exercises the same two branches the
      // form has to handle: measured, and not measured.
      const measured = query !== undefined && "max_frames" in query;
      return {
        video_id: id,
        original_filename: fileName,
        stored_path: `uploads/${id}/input.mp4`,
        content_type: "video/mp4",
        size_bytes: 4_096,
        // Contract v10 (§1-6): the real `/upload/video` ALWAYS returns
        // `trimmed`, and sets it true only when both trim query parameters
        // arrived and the cut succeeded. The mock mirrors that: both keys
        // present -> `true`, anything else (no query, one key only) -> `false`.
        trimmed: hasTrimQuery(query),
        // `!== undefined` (not `??`): an EXPLICIT `null` is the "the server could
        // not measure it" case a test needs to be able to ask for, and `??`
        // would silently swap it back for the default.
        frame_count: measured
          ? options.uploadVideoFrameCount !== undefined
            ? options.uploadVideoFrameCount
            : DEFAULT_UPLOAD_VIDEO_FRAME_COUNT
          : null,
        fps: measured ? (options.uploadVideoFps !== undefined ? options.uploadVideoFps : DEFAULT_UPLOAD_VIDEO_FPS) : null,
      };
    }
    return {
      audio_id: id,
      original_filename: fileName,
      stored_path: `uploads/${id}/input.wav`,
      content_type: "audio/wav",
      size_bytes: 2_048,
    };
  }

  function fileNameFromPathMock(path: string): string {
    const idx = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
    return idx >= 0 ? path.slice(idx + 1) : path;
  }

  async function handleUploadFile(params: ParamsOf<"backend.uploadFile">): Promise<ResultOf<"backend.uploadFile">> {
    if (options.failUploadFile) {
      throw new BridgeError(
        options.failUploadFile,
        `Mock bridge: backend.uploadFile forced to fail (${options.failUploadFile})`,
      );
    }
    return { status: 200, body: uploadResponseBody(params.kind, params.filePath, params.query) };
  }

  // --- contract v5 (timeline generation-AI bridge) --------------------------

  async function handleGetSelection(): Promise<ResultOf<"timeline.getSelection">> {
    if (options.failGetSelection) {
      throw new BridgeError("NO_EDIT_HANDLE", "Mock bridge: no active AviUtl2 edit session");
    }
    return { ...DEFAULT_SELECTION, ...options.selection };
  }

  let cutoutCounter = 0;

  async function handleCutoutRange(
    params: ParamsOf<"timeline.cutoutRange">,
  ): Promise<ResultOf<"timeline.cutoutRange">> {
    if (options.failCutoutRange) {
      throw new BridgeError(
        options.failCutoutRange,
        `Mock bridge: timeline.cutoutRange forced to fail (${options.failCutoutRange})`,
      );
    }
    cutoutCounter += 1;
    const width = options.editInfo?.width ?? DEFAULT_EDIT_INFO.width;
    const height = options.editInfo?.height ?? DEFAULT_EDIT_INFO.height;
    return {
      filePath: `C:\\Users\\mock\\AppData\\Local\\Temp\\Nz-LTX23\\cutout_${params.layer}_${params.frameStart}_${cutoutCounter}.mp4`,
      width,
      height,
      frameCount: params.frameCount,
      hasAudio: params.withAudio,
    };
  }

  let extractCounter = 0;

  async function handleExtractAudio(
    params: ParamsOf<"timeline.extractAudio">,
  ): Promise<ResultOf<"timeline.extractAudio">> {
    if (options.failExtractAudio) {
      throw new BridgeError(
        options.failExtractAudio,
        options.extractAudioErrorMessage ??
          `Mock bridge: timeline.extractAudio forced to fail (${options.failExtractAudio})`,
      );
    }
    extractCounter += 1;
    const rate = options.editInfo?.rate ?? DEFAULT_EDIT_INFO.rate;
    const scale = options.editInfo?.scale ?? DEFAULT_EDIT_INFO.scale;
    const sampleRate = options.editInfo?.sampleRate ?? DEFAULT_EDIT_INFO.sampleRate;
    const durationSec = Math.round((params.frameCount * (scale / rate)) * 1000) / 1000;
    return {
      filePath: `C:\\Users\\mock\\AppData\\Local\\Temp\\Nz-LTX23\\extract_${params.layer}_${params.frameStart}_${extractCounter}.wav`,
      durationSec,
      sampleRate,
      hasAudioStream: true,
    };
  }

  /** In-memory record of provisional placeholders, keyed by jobId, so
   * `resolveProvisional`/`updateProvisionalText`/`updateProvisionalReservation`
   * can report whether the placeholder is still present (drives
   * `mode`/`updated`/`deletedOld`). */
  const provisionals = new Map<string, { layer: number; frame: number; lengthFrames: number; text: string }>();

  /** The mock's stand-in for `EDIT_INFO.layer_max`, used by placement `"B"`
   * (same start, `layer_max+1`) and would-be collision fallbacks. */
  const MOCK_LAYER_MAX = 10;

  /** Mirrors native's length resolution: `numFrames`+`genFps` re-scaled to the
   * project fps (`round(numFrames * projectFps / genFps)`). */
  function resolveMockLength(p: { numFrames: number; genFps: number }): number {
    const rate = options.editInfo?.rate ?? DEFAULT_EDIT_INFO.rate;
    const scale = options.editInfo?.scale ?? DEFAULT_EDIT_INFO.scale;
    if (p.genFps > 0 && scale > 0) {
      const projectFps = rate / scale;
      return Math.max(1, Math.round((p.numFrames * projectFps) / p.genFps));
    }
    return Math.max(1, p.numFrames);
  }

  /** Mirrors native's placement resolution (spec §5-9): (A) directly after the
   * material (`materialFrameEnd` is inclusive, so the next free frame is `+1`),
   * (B) same start on `layer_max+1`, (C) the cursor position. */
  function resolveMockPlacement(p: {
    placement: "A" | "B" | "C";
    materialLayer?: number;
    materialFrameStart?: number;
    materialFrameEnd?: number;
    cursorLayer?: number;
    cursorFrame?: number;
  }): { layer: number; frame: number } {
    switch (p.placement) {
      case "A":
        return { layer: p.materialLayer ?? 0, frame: (p.materialFrameEnd ?? 0) + 1 };
      case "B":
        return { layer: MOCK_LAYER_MAX + 1, frame: p.materialFrameStart ?? 0 };
      case "C":
        return { layer: p.cursorLayer ?? 0, frame: p.cursorFrame ?? 0 };
    }
  }

  async function handleInsertProvisional(
    params: ParamsOf<"timeline.insertProvisional">,
  ): Promise<ResultOf<"timeline.insertProvisional">> {
    if (options.failProvisional) {
      throw new BridgeError("PROVISIONAL_FAILED", "Mock bridge: insertProvisional forced to fail");
    }
    // Length from numFrames+genFps, position from the placement system; native
    // resolves them identically.
    const lengthFrames = resolveMockLength(params);
    const pos = resolveMockPlacement(params);
    provisionals.set(params.jobId, {
      layer: pos.layer,
      frame: pos.frame,
      lengthFrames,
      text: params.displayText,
    });
    return {
      inserted: true,
      layer: pos.layer,
      frame: pos.frame,
      objectName: `provisional-${params.jobId}`,
      placedLayer: pos.layer,
      placedFrame: pos.frame,
      usedFallback: false,
    };
  }

  /** I3 (spec §5-3): atomic delete-old + create-new. Deletes the `oldJobId`
   * entry when present (empty `oldJobId` = create-only), then records the new
   * `newJobId` placeholder at the placement-resolved position. */
  async function handleUpdateProvisionalReservation(
    params: ParamsOf<"timeline.updateProvisionalReservation">,
  ): Promise<ResultOf<"timeline.updateProvisionalReservation">> {
    if (options.failProvisional) {
      throw new BridgeError("PROVISIONAL_FAILED", "Mock bridge: updateProvisionalReservation forced to fail");
    }
    let deletedOld = false;
    if (params.oldJobId && provisionals.has(params.oldJobId)) {
      provisionals.delete(params.oldJobId);
      deletedOld = true;
    }
    const lengthFrames = resolveMockLength(params);
    const pos = resolveMockPlacement(params);
    provisionals.set(params.newJobId, {
      layer: pos.layer,
      frame: pos.frame,
      lengthFrames,
      text: params.displayText,
    });
    return {
      ok: true,
      deletedOld,
      placedLayer: pos.layer,
      placedFrame: pos.frame,
      usedFallback: false,
    };
  }

  async function handleResolveProvisional(
    params: ParamsOf<"timeline.resolveProvisional">,
  ): Promise<ResultOf<"timeline.resolveProvisional">> {
    if (options.failProvisional) {
      throw new BridgeError("PROVISIONAL_FAILED", "Mock bridge: resolveProvisional forced to fail");
    }
    const existed = provisionals.delete(params.jobId);
    return {
      mode: existed ? "replaced" : "insertedReserved",
      layer: params.reservedLayer,
      frame: params.reservedFrame,
    };
  }

  async function handleUpdateProvisionalText(
    params: ParamsOf<"timeline.updateProvisionalText">,
  ): Promise<ResultOf<"timeline.updateProvisionalText">> {
    if (options.failProvisional) {
      throw new BridgeError("PROVISIONAL_FAILED", "Mock bridge: updateProvisionalText forced to fail");
    }
    const existing = provisionals.get(params.jobId);
    if (existing) existing.text = params.text;
    return { updated: existing !== undefined };
  }

  /** I13 (spec §5-10): the ✅ marker cleanup on a successful 🎞 insert. Removes
   * the `jobId` entry from the provisional table when present; a missing entry
   * is an idempotent no-op success (`deleted: false`). After this, a
   * `scanProvisionals` no longer reports the removed id — modelling the ✅
   * placeholder being deleted from the timeline. */
  async function handleDeleteProvisionalByJob(
    params: ParamsOf<"timeline.deleteProvisionalByJob">,
  ): Promise<ResultOf<"timeline.deleteProvisionalByJob">> {
    if (options.failProvisional) {
      throw new BridgeError("PROVISIONAL_FAILED", "Mock bridge: deleteProvisionalByJob forced to fail");
    }
    const deleted = provisionals.delete(params.jobId);
    return { ok: true, deleted };
  }

  async function handleScanProvisionals(): Promise<ResultOf<"timeline.scanProvisionals">> {
    // A faithful scan reports every NzLTX23-tagged object actually on the
    // timeline. Two sources contribute: (1) placeholders inserted this session
    // via `insertProvisional`/`updateProvisionalReservation` (the live
    // `provisionals` map), and (2) `options.provisionalOrphans` — placeholders
    // seeded as "survived a prior session's reload" (the mock never ran their
    // insert, so they only exist as seeded state). Merging both is what lets an
    // on-demand `reconcileFromTimeline` re-observe a still-present reservation
    // (keeping the seat `reserved` so a re-click MOVES it) while a seeded
    // job-bound orphan still restores `generating`. A `resolveProvisional` /
    // `updateProvisionalReservation` that deletes an id drops it from the live
    // map, so the scan stops reporting it — modelling a placeholder that was
    // removed from the timeline. Seeded orphans are filtered out if the live
    // map already carries the same id, so a re-inserted id isn't double-counted.
    const live = Array.from(provisionals.entries()).map(([jobId, p]) => ({
      jobId,
      layer: p.layer,
      frame: p.frame,
    }));
    const seeded = (options.provisionalOrphans ?? []).filter((o) => !provisionals.has(o.jobId));
    return { orphans: [...live, ...seeded] };
  }

  // --- contract v6 (batch A2V + fs bridge) -----------------------------

  let pickFolderCounter = 0;

  async function handlePickFolder(
    // `title` isn't reflected in the fixture — native only uses it to label
    // the dialog — but the param is still validated by the type system.
    _params: ParamsOf<"ui.pickFolder">,
  ): Promise<ResultOf<"ui.pickFolder">> {
    if (options.failPickFolder) {
      throw new BridgeError(
        options.failPickFolder,
        options.failPickFolder === "CANCELLED"
          ? "Mock bridge: ui.pickFolder forced CANCELLED (user dismissed the dialog)"
          : "Mock bridge: ui.pickFolder forced to fail (DIALOG_FAILED)",
      );
    }
    pickFolderCounter += 1;
    const folderPath = options.pickFolderPath ?? `C:\\Users\\mock\\Videos\\picked-folder-${pickFolderCounter}`;
    return { folderPath };
  }

  async function handleListFiles(params: ParamsOf<"fs.listFiles">): Promise<ResultOf<"fs.listFiles">> {
    const entries = mockFs.folders.get(params.folderPath) ?? [];
    const extFilter = params.extensions?.map((e) => e.toLowerCase());
    const filtered = extFilter ? entries.filter((e) => extFilter.includes(extnameLower(e.name))) : entries;
    const files = filtered.map((e) => {
      const isWav = extnameLower(e.name) === ".wav";
      const durationSec = params.withAudioDuration && isWav ? (e.durationSec ?? 0) : 0;
      return {
        name: e.name,
        path: joinMockPath(params.folderPath, e.name),
        sizeBytes: e.sizeBytes,
        mtimeMs: e.mtimeMs,
        durationSec,
      };
    });
    return { files };
  }

  /** Finds the `MockFs` entry backing `filePath` by scanning every folder for
   * a `name` that joins back to it. `fs.probeAudioDuration` is keyed by full
   * path (unlike `fs.listFiles`, which is scoped to one folder), so this scan
   * is the mock's stand-in for a real filesystem lookup. */
  function findMockFsEntry(filePath: string): MockFsFileEntry | undefined {
    for (const [folderPath, entries] of mockFs.folders) {
      for (const entry of entries) {
        if (joinMockPath(folderPath, entry.name) === filePath) return entry;
      }
    }
    return undefined;
  }

  async function handleProbeAudioDuration(
    params: ParamsOf<"fs.probeAudioDuration">,
  ): Promise<ResultOf<"fs.probeAudioDuration">> {
    const isWavExt = extnameLower(params.filePath) === ".wav";
    const entry = isWavExt ? findMockFsEntry(params.filePath) : undefined;
    if (!isWavExt || !entry || entry.durationSec === undefined) {
      return { durationSec: 0, isWav: false };
    }
    return { durationSec: entry.durationSec, isWav: true };
  }

  /** Contract v9: mirrors native's `fs.probeMediaInfo` — a best-effort probe
   * that never rejects for a bad/unreadable file. The values come straight
   * from `options.probeMediaInfoDurationSec`/`Width`/`Height` (all default
   * `0`, matching native's "unknown" response when there's no edit handle or
   * `get_media_info` fails). */
  async function handleProbeMediaInfo(
    _params: ParamsOf<"fs.probeMediaInfo">,
  ): Promise<ResultOf<"fs.probeMediaInfo">> {
    return {
      durationSec: options.probeMediaInfoDurationSec ?? 0,
      width: options.probeMediaInfoWidth ?? 0,
      height: options.probeMediaInfoHeight ?? 0,
    };
  }

  // --- contract v7 (drag-and-drop) -------------------------------------------

  /** Mirrors native's real contract (`bridge_core.cpp`'s
   * `ParseResolveDroppedFiles`/`MakeResolveDroppedFilesResult`): reads
   * `params.__droppedPaths` (a plain string array) and synthesizes one
   * `{filePath, fileName}` per non-empty string entry, in order, under a
   * fixed fake folder. A missing/malformed key, or a key whose array has no
   * usable string entries, simply excludes everything — never an error,
   * exactly like native's "nothing usable was dropped" case.
   *
   * This function is fed `params` (not a raw `files` array) on purpose: this
   * mock's `requestWithFiles` (below) populates `params.__droppedPaths`
   * itself, mirroring native's real `webview_host.cpp` injection, so this
   * handler exercises the EXACT SAME shape the real `ParseResolveDroppedFiles`
   * reads. An earlier version synthesized the result directly from the
   * `files` array, bypassing the injection step entirely — which is exactly
   * what let a real regression (native's own injection never firing for a
   * genuine drop, because it checked for a "__droppedPaths" substring BEFORE
   * checking whether any files were actually attached) slip past every webui
   * test: the mock's shortcut couldn't fail the same way the bypassed
   * injection did. */
  async function handleResolveDroppedFiles(params: unknown): Promise<ResultOf<"ui.resolveDroppedFiles">> {
    if (options.failResolveDroppedFiles) {
      throw new BridgeError(
        "BAD_REQUEST",
        "Mock bridge: ui.resolveDroppedFiles forced to fail (failResolveDroppedFiles option set)",
      );
    }
    const droppedPaths = (params as { __droppedPaths?: unknown } | null)?.__droppedPaths;
    const resolvedFiles = (Array.isArray(droppedPaths) ? droppedPaths : [])
      .filter((name): name is string => typeof name === "string" && name.length > 0)
      .map((name) => ({ filePath: joinMockPath("C:\\Users\\mock\\Downloads", name), fileName: name }));
    return { files: resolvedFiles };
  }

  /** Real event registry (contract v5): unlike M1/M2, the mock now routes
   * `on`/`emit` through a live handler map so tests (and `npm run dev` tooling)
   * can drive native-initiated events like `timeline.menuInvoked`. */
  const eventHandlers = new Map<string, Set<(data: unknown) => void>>();

  return {
    async request<M extends BridgeMethod>(
      method: M,
      params: ParamsOf<M>,
    ): Promise<ResultOf<M>> {
      if (method === "backend.uploadFile" && uploadGate) {
        await uploadGate;
      } else {
        await wait();
      }

      switch (method) {
        case "ping":
          return (await handlePing()) as ResultOf<M>;
        case "getEditInfo":
          return (await handleGetEditInfo()) as ResultOf<M>;
        case "backend.request":
          return (await handleBackendRequest(params as ParamsOf<"backend.request">)) as ResultOf<M>;
        case "backend.downloadVideo":
          return (await handleDownloadVideo(params as ParamsOf<"backend.downloadVideo">)) as ResultOf<M>;
        case "timeline.insertMedia":
          return (await handleInsertMedia(params as ParamsOf<"timeline.insertMedia">)) as ResultOf<M>;
        case "timeline.insertMediaForJob":
          return (await handleInsertMediaForJob(
            params as ParamsOf<"timeline.insertMediaForJob">,
          )) as ResultOf<M>;
        case "backend.getBaseUrl":
          return (await handleGetBaseUrl()) as ResultOf<M>;
        case "timeline.captureFrame":
          return (await handleCaptureFrame(params as ParamsOf<"timeline.captureFrame">)) as ResultOf<M>;
        case "ui.pickFile":
          return (await handlePickFile(params as ParamsOf<"ui.pickFile">)) as ResultOf<M>;
        case "ui.makeThumbnail":
          return (await handleMakeThumbnail(params as ParamsOf<"ui.makeThumbnail">)) as ResultOf<M>;
        case "backend.uploadFile":
          return (await handleUploadFile(params as ParamsOf<"backend.uploadFile">)) as ResultOf<M>;
        case "settings.get":
          return (await handleSettingsGet()) as ResultOf<M>;
        case "settings.set":
          return (await handleSettingsSet(params as ParamsOf<"settings.set">)) as ResultOf<M>;
        case "timeline.getSelection":
          return (await handleGetSelection()) as ResultOf<M>;
        case "timeline.cutoutRange":
          return (await handleCutoutRange(params as ParamsOf<"timeline.cutoutRange">)) as ResultOf<M>;
        case "timeline.extractAudio":
          return (await handleExtractAudio(params as ParamsOf<"timeline.extractAudio">)) as ResultOf<M>;
        case "timeline.insertProvisional":
          return (await handleInsertProvisional(params as ParamsOf<"timeline.insertProvisional">)) as ResultOf<M>;
        case "timeline.resolveProvisional":
          return (await handleResolveProvisional(params as ParamsOf<"timeline.resolveProvisional">)) as ResultOf<M>;
        case "timeline.updateProvisionalText":
          return (await handleUpdateProvisionalText(params as ParamsOf<"timeline.updateProvisionalText">)) as ResultOf<M>;
        case "timeline.updateProvisionalReservation":
          return (await handleUpdateProvisionalReservation(
            params as ParamsOf<"timeline.updateProvisionalReservation">,
          )) as ResultOf<M>;
        case "timeline.deleteProvisionalByJob":
          return (await handleDeleteProvisionalByJob(
            params as ParamsOf<"timeline.deleteProvisionalByJob">,
          )) as ResultOf<M>;
        case "timeline.scanProvisionals":
          return (await handleScanProvisionals()) as ResultOf<M>;
        case "ui.pickFolder":
          return (await handlePickFolder(params as ParamsOf<"ui.pickFolder">)) as ResultOf<M>;
        case "fs.listFiles":
          return (await handleListFiles(params as ParamsOf<"fs.listFiles">)) as ResultOf<M>;
        case "fs.probeAudioDuration":
          return (await handleProbeAudioDuration(params as ParamsOf<"fs.probeAudioDuration">)) as ResultOf<M>;
        case "fs.probeMediaInfo":
          return (await handleProbeMediaInfo(params as ParamsOf<"fs.probeMediaInfo">)) as ResultOf<M>;
        case "ui.resolveDroppedFiles":
          // Reached only if a caller uses plain `request()` instead of
          // `requestWithFiles()` — `params` carries no `__droppedPaths` key
          // in that case, so this mirrors native's own "no key" -> empty
          // result via the same handler `requestWithFiles` uses.
          return (await handleResolveDroppedFiles(params)) as ResultOf<M>;
        default:
          throw new BridgeError(
            "UNKNOWN_METHOD",
            `Mock bridge: unknown method "${String(method)}"`,
          );
      }
    },

    async requestWithFiles<M extends BridgeMethod>(
      method: M,
      params: ParamsOf<M>,
      files: unknown[],
    ): Promise<ResultOf<M>> {
      // Mirrors the real contract (webview_host.cpp's InjectDroppedPaths):
      // dropped files are merged into params.__droppedPaths BEFORE dispatch,
      // exactly like native force-injects them server-side — see
      // `handleResolveDroppedFiles`'s doc comment for why bypassing this step
      // (as an earlier version of this mock did) is exactly what let a real
      // native regression slip past every webui test.
      const droppedPaths = files
        .map((f) => (f as { name?: unknown } | null)?.name)
        .filter((name): name is string => typeof name === "string" && name.length > 0);
      const paramsWithDropped = { ...(params as Record<string, unknown>), __droppedPaths: droppedPaths };

      if (method === "ui.resolveDroppedFiles") {
        await wait();
        return (await handleResolveDroppedFiles(paramsWithDropped)) as ResultOf<M>;
      }
      // No other mock handler needs the attached files — fall back to the
      // ordinary dispatch (this delegates to `this.request`, so the caller
      // still gets its own `wait()`/dispatch behavior unchanged).
      return this.request(method, params);
    },

    on(event, handler) {
      let handlers = eventHandlers.get(event);
      if (!handlers) {
        handlers = new Set();
        eventHandlers.set(event, handlers);
      }
      handlers.add(handler);
      return () => {
        handlers?.delete(handler);
      };
    },

    emit(event, data) {
      const handlers = eventHandlers.get(event);
      if (!handlers) return;
      // Copy before iterating so a handler that unsubscribes mid-dispatch
      // doesn't disturb the walk.
      for (const handler of [...handlers]) {
        handler(data);
      }
    },

    releaseUploads() {
      resolveUploadGate?.();
    },
  };
}
