import { afterEach, describe, expect, it, vi } from "vitest";
import { FALLBACK_APP_CONFIG } from "../modes/single/defaultConfig";
import { createMockBridge, MOCK_CONFIG_BODY } from "./mockBridge";
import { BridgeError } from "./types";

describe("createMockBridge", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("resolves ping with pong=true and the configured plugin version, after the simulated delay", async () => {
    vi.useFakeTimers();
    const bridge = createMockBridge({ delayMs: 200, pluginVersion: "0.1.0-M1" });

    let settled = false;
    const promise = bridge.request("ping", {}).then((result) => {
      settled = true;
      return result;
    });

    await vi.advanceTimersByTimeAsync(199);
    expect(settled).toBe(false);

    await vi.advanceTimersByTimeAsync(1);
    await expect(promise).resolves.toEqual({
      pong: true,
      pluginVersion: "0.1.0-M1",
    });
  });

  it("resolves getEditInfo with the documented M1 default payload", async () => {
    const bridge = createMockBridge({ delayMs: 0 });

    await expect(bridge.request("getEditInfo", {})).resolves.toEqual({
      width: 1920,
      height: 1080,
      rate: 30,
      scale: 1,
      sampleRate: 44100,
      frame: 120,
      // Native always sends these three (BRIDGE_CONTRACT.md §9); the mock
      // mirrors that so the payload shape matches a real edit session.
      layer: 1,
      frameMax: 300,
      layerMax: 100,
    });
  });

  it("merges editInfo overrides over the default payload", async () => {
    const bridge = createMockBridge({
      delayMs: 0,
      editInfo: { width: 3840, height: 2160 },
    });

    await expect(bridge.request("getEditInfo", {})).resolves.toMatchObject({
      width: 3840,
      height: 2160,
      rate: 30,
    });
  });

  it("rejects ping with a BridgeError when failPing is set", async () => {
    const bridge = createMockBridge({ delayMs: 0, failPing: true });

    await expect(bridge.request("ping", {})).rejects.toBeInstanceOf(
      BridgeError,
    );
  });

  it("rejects getEditInfo with NO_EDIT_HANDLE when failEditInfo is set", async () => {
    const bridge = createMockBridge({ delayMs: 0, failEditInfo: true });

    await expect(bridge.request("getEditInfo", {})).rejects.toMatchObject({
      code: "NO_EDIT_HANDLE",
    });
  });

  it("on() returns an unsubscribe function and never invokes the handler", () => {
    const bridge = createMockBridge();
    const handler = vi.fn();

    const unsubscribe = bridge.on("job.progress", handler);
    unsubscribe();

    expect(handler).not.toHaveBeenCalled();
  });

  // --- contract v3 (M4 I2V keyframes) ---------------------------------------

  describe("timeline.captureFrame", () => {
    it("resolves with a filePath/width/height/frame, using the current edit info by default", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await bridge.request("timeline.captureFrame", {});
      expect(result.filePath).toMatch(/\.png$/);
      expect(result.width).toBe(1920);
      expect(result.height).toBe(1080);
      expect(result.frame).toBe(120); // DEFAULT_EDIT_INFO.frame
    });

    it("echoes back an explicit frame number", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await bridge.request("timeline.captureFrame", { frame: 42 });
      expect(result.frame).toBe(42);
    });

    it("rejects with the configured error code when failCaptureFrame is set", async () => {
      const bridge = createMockBridge({ delayMs: 0, failCaptureFrame: "CAPTURE_FAILED" });
      await expect(bridge.request("timeline.captureFrame", {})).rejects.toMatchObject({
        code: "CAPTURE_FAILED",
      });
    });
  });

  describe("ui.pickFile", () => {
    it("resolves with a filePath/fileName for the requested kind", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await bridge.request("ui.pickFile", { kind: "image" });
      expect(result.fileName).toMatch(/\.png$/);
      expect(result.filePath).toContain(result.fileName);
    });

    it("honors a custom pickFileName override", async () => {
      const bridge = createMockBridge({ delayMs: 0, pickFileName: "chosen.jpg" });
      const result = await bridge.request("ui.pickFile", { kind: "image" });
      expect(result.fileName).toBe("chosen.jpg");
    });

    it("rejects with CANCELLED when failPickFile is 'CANCELLED'", async () => {
      const bridge = createMockBridge({ delayMs: 0, failPickFile: "CANCELLED" });
      await expect(bridge.request("ui.pickFile", { kind: "image" })).rejects.toMatchObject({
        code: "CANCELLED",
      });
    });

    it("rejects with DIALOG_FAILED when failPickFile is 'DIALOG_FAILED'", async () => {
      const bridge = createMockBridge({ delayMs: 0, failPickFile: "DIALOG_FAILED" });
      await expect(bridge.request("ui.pickFile", { kind: "image" })).rejects.toMatchObject({
        code: "DIALOG_FAILED",
      });
    });
  });

  describe("ui.makeThumbnail", () => {
    it("resolves with a data: URL and dimensions", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await bridge.request("ui.makeThumbnail", { filePath: "C:\\mock\\frame.png" });
      expect(result.dataUrl).toMatch(/^data:image\/png;base64,/);
      expect(result.sourceWidth).toBeGreaterThan(0);
      expect(result.sourceHeight).toBeGreaterThan(0);
    });

    it("rejects with the configured error code when failMakeThumbnail is set", async () => {
      const bridge = createMockBridge({ delayMs: 0, failMakeThumbnail: "FILE_NOT_FOUND" });
      await expect(bridge.request("ui.makeThumbnail", { filePath: "nope.png" })).rejects.toMatchObject({
        code: "FILE_NOT_FOUND",
      });
    });
  });

  describe("backend.uploadFile", () => {
    it("resolves status 200 with an UploadImageResponse-shaped body for kind='image'", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await bridge.request("backend.uploadFile", { kind: "image", filePath: "C:\\mock\\tiny.png" });
      expect(result.status).toBe(200);
      const body = result.body as Record<string, unknown>;
      expect(typeof body.image_id).toBe("string");
      expect(body.original_filename).toBe("tiny.png");
      expect(typeof body.width).toBe("number");
      expect(typeof body.height).toBe("number");
    });

    it("mints a fresh image_id per call", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const first = await bridge.request("backend.uploadFile", { kind: "image", filePath: "a.png" });
      const second = await bridge.request("backend.uploadFile", { kind: "image", filePath: "b.png" });
      const firstBody = first.body as { image_id: string };
      const secondBody = second.body as { image_id: string };
      expect(firstBody.image_id).not.toBe(secondBody.image_id);
    });

    it("returns video_id/audio_id shapes for the other kinds", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const video = await bridge.request("backend.uploadFile", { kind: "video", filePath: "clip.mp4" });
      expect((video.body as { video_id: string }).video_id).toBeDefined();
      const audio = await bridge.request("backend.uploadFile", { kind: "audio", filePath: "sound.wav" });
      expect((audio.body as { audio_id: string }).audio_id).toBeDefined();
    });

    it("rejects with the configured error code when failUploadFile is set (transport failure)", async () => {
      const bridge = createMockBridge({ delayMs: 0, failUploadFile: "BACKEND_UNREACHABLE" });
      await expect(
        bridge.request("backend.uploadFile", { kind: "image", filePath: "tiny.png" }),
      ).rejects.toMatchObject({ code: "BACKEND_UNREACHABLE" });
    });
  });

  // NAG (2026-07-28): the mock's 422 mirror of the backend's "enabled but
  // empty negative prompt" rule for a plain /generate request.
  describe("POST /generate (NAG)", () => {
    const GENERATE_BODY = {
      prompt: "a cat riding a skateboard",
      width: 512,
      height: 320,
      num_frames: 49,
      frame_rate: 24,
      seed: -1,
    };

    async function postGenerate(bridge: ReturnType<typeof createMockBridge>, body: object) {
      return bridge.request("backend.request", { method: "POST", path: "/api/v1/generate", body });
    }

    it("rejects nag_enabled=true with an empty negative_prompt as 422 VALIDATION_ERROR", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await postGenerate(bridge, { ...GENERATE_BODY, nag_enabled: true, negative_prompt: "   " });
      expect(result.status).toBe(422);
      expect((result.body as { error: { code: string } }).error.code).toBe("VALIDATION_ERROR");
    });

    it("accepts nag_enabled=true with a non-empty negative_prompt", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await postGenerate(bridge, {
        ...GENERATE_BODY,
        nag_enabled: true,
        negative_prompt: "blurry, low quality",
        nag_scale: 11.0,
        nag_tau: 2.5,
        nag_alpha: 0.25,
      });
      expect(result.status).toBe(202);
    });
  });

  // M6: POST /generate/chain, GET /jobs/{id}'s clip/clip_count, and
  // POST /jobs/{id}/join (Docs/API_REFERENCE.md §3.13/§3.17).
  describe("POST /generate/chain", () => {
    async function postChain(bridge: ReturnType<typeof createMockBridge>, body: object) {
      return bridge.request("backend.request", { method: "POST", path: "/api/v1/generate/chain", body });
    }

    const CHAIN_BODY = {
      prompt: "a cat riding a skateboard, then a dog joins in",
      width: 512,
      height: 320,
      frame_rate: 24,
      seed: -1,
      overlap_frames: 3,
      overlap_strength: 0.5,
      clips: [{ num_frames: 49 }, { num_frames: 33 }],
    };

    it("accepts a 2-clip chain request with 202 + job_id + num_clips", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await postChain(bridge, CHAIN_BODY);
      expect(result.status).toBe(202);
      const body = result.body as { job_id: string; status: string; num_clips: number };
      expect(typeof body.job_id).toBe("string");
      expect(body.status).toBe("queued");
      expect(body.num_clips).toBe(2);
    });

    it("rejects an empty prompt with 422 VALIDATION_ERROR", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await postChain(bridge, { ...CHAIN_BODY, prompt: "" });
      expect(result.status).toBe(422);
      expect((result.body as { error: { code: string } }).error.code).toBe("VALIDATION_ERROR");
    });

    it("rejects fewer than 2 clips when there's no source_video/source_audio", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await postChain(bridge, { ...CHAIN_BODY, clips: [{ num_frames: 49 }] });
      expect(result.status).toBe(422);
    });

    // NAG (2026-07-28): the mock's 422 mirror of the backend's "enabled but
    // empty negative prompt" rule, on the chain endpoint too.
    it("rejects nag_enabled=true with an empty negative_prompt as 422 VALIDATION_ERROR", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await postChain(bridge, { ...CHAIN_BODY, nag_enabled: true, negative_prompt: "" });
      expect(result.status).toBe(422);
      expect((result.body as { error: { code: string } }).error.code).toBe("VALIDATION_ERROR");
    });

    it("accepts nag_enabled=true with a non-empty negative_prompt", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await postChain(bridge, {
        ...CHAIN_BODY,
        nag_enabled: true,
        negative_prompt: "blurry, low quality",
        nag_scale: 11.0,
        nag_tau: 2.5,
        nag_alpha: 0.25,
      });
      expect(result.status).toBe(202);
    });

    it("accepts a single clip when source_video is present (V2V)", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await postChain(bridge, {
        ...CHAIN_BODY,
        clips: [{ num_frames: 49 }],
        source_video: { video_id: "mock-video-1", context_frames: 25 },
      });
      expect(result.status).toBe(202);
      expect((result.body as { num_clips: number }).num_clips).toBe(1);
    });

    it("returns 409 JOB_BUSY while another job (chain or plain) is already in flight", async () => {
      const bridge = createMockBridge({ delayMs: 0, runningPollCount: 50 });
      const first = await postChain(bridge, CHAIN_BODY);
      expect(first.status).toBe(202);

      const second = await postChain(bridge, CHAIN_BODY);
      expect(second.status).toBe(409);
      expect((second.body as { error: { code: string } }).error.code).toBe("JOB_BUSY");
    });

    it("GET /jobs/{id} reports clip/clip_count that progress toward num_clips, and null/null for a plain /generate job", async () => {
      const bridge = createMockBridge({ delayMs: 0, runningPollCount: 4 });
      const accepted = await postChain(bridge, CHAIN_BODY);
      const jobId = (accepted.body as { job_id: string }).job_id;

      const getJob = () => bridge.request("backend.request", { method: "GET", path: `/api/v1/jobs/${jobId}` });

      const first = await getJob();
      const firstBody = first.body as { clip: number | null; clip_count: number | null };
      expect(firstBody.clip_count).toBe(2);
      expect(firstBody.clip).toBeGreaterThanOrEqual(1);
      expect(firstBody.clip).toBeLessThanOrEqual(2);

      // Drain to completion; clip should have reached the final clip index.
      let last = firstBody;
      for (let i = 0; i < 10; i += 1) {
        const res = await getJob();
        last = res.body as { clip: number | null; clip_count: number | null; status?: string };
        if ((last as { status?: string }).status === "completed") break;
      }
      expect(last.clip).toBe(2);
      expect(last.clip_count).toBe(2);

      // A plain (non-chain) job never reports clip/clip_count.
      const plainAccepted = await bridge.request("backend.request", {
        method: "POST",
        path: "/api/v1/generate",
        body: { prompt: "a cat", width: 384, height: 256, num_frames: 17, frame_rate: 24, seed: 1 },
      });
      const plainJobId = (plainAccepted.body as { job_id: string }).job_id;
      const plainJob = await bridge.request("backend.request", {
        method: "GET",
        path: `/api/v1/jobs/${plainJobId}`,
      });
      const plainBody = plainJob.body as { clip: number | null; clip_count: number | null };
      expect(plainBody.clip).toBeNull();
      expect(plainBody.clip_count).toBeNull();
    });
  });

  describe("POST /jobs/{id}/join", () => {
    async function makeV2VJob(bridge: ReturnType<typeof createMockBridge>): Promise<string> {
      const generated = await bridge.request("backend.request", {
        method: "POST",
        path: "/api/v1/generate/chain",
        body: {
          prompt: "continue this video",
          width: 512,
          height: 320,
          frame_rate: 24,
          seed: -1,
          overlap_frames: 3,
          overlap_strength: 0.5,
          clips: [{ num_frames: 49 }],
          source_video: { video_id: "mock-video-1", context_frames: 25 },
        },
      });
      return (generated.body as { job_id: string }).job_id;
    }

    it("returns a 200 join response (I4 shape: trimmed_source_seconds / source_fps) for a known job", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const jobId = await makeV2VJob(bridge);

      const joined = await bridge.request("backend.request", {
        method: "POST",
        path: `/api/v1/jobs/${jobId}/join`,
        body: { source_tail_seconds: 5 },
      });
      expect(joined.status).toBe(200);
      const body = joined.body as {
        job_id: string;
        joined_path: string;
        join_mode: string;
        source_normalized: boolean;
        trimmed_source_seconds: number;
        source_fps: number | null;
      };
      expect(body.job_id).toBe(jobId);
      expect(typeof body.joined_path).toBe("string");
      expect(body.source_normalized).toBe(true);
      // Pseudo tail-keep: assumed ~12s source, kept last 5s -> trimmed 7s.
      expect(body.trimmed_source_seconds).toBe(7);
      expect(body.source_fps).toBe(24);
    });

    it("source_tail_seconds=0 joins the source at full length (no trim)", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const jobId = await makeV2VJob(bridge);
      const joined = await bridge.request("backend.request", {
        method: "POST",
        path: `/api/v1/jobs/${jobId}/join`,
        body: { source_tail_seconds: 0 },
      });
      expect((joined.body as { trimmed_source_seconds: number }).trimmed_source_seconds).toBe(0);
    });

    it("flips JobResponse.joined true and unlocks GET /jobs/{id}/joined (404 JOINED_NOT_READY -> 200)", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const jobId = await makeV2VJob(bridge);

      // Before any join: JobResponse.joined is false and the joined route 404s.
      const before = await bridge.request("backend.request", { method: "GET", path: `/api/v1/jobs/${jobId}` });
      expect((before.body as { is_v2v: boolean; joined: boolean }).is_v2v).toBe(true);
      expect((before.body as { joined: boolean }).joined).toBe(false);
      const joinedBefore = await bridge.request("backend.request", { method: "GET", path: `/api/v1/jobs/${jobId}/joined` });
      expect(joinedBefore.status).toBe(404);
      expect((joinedBefore.body as { error: { code: string } }).error.code).toBe("JOINED_NOT_READY");

      // Run the join, then both flip to "ready".
      await bridge.request("backend.request", { method: "POST", path: `/api/v1/jobs/${jobId}/join`, body: {} });
      const after = await bridge.request("backend.request", { method: "GET", path: `/api/v1/jobs/${jobId}` });
      expect((after.body as { joined: boolean }).joined).toBe(true);
      const joinedAfter = await bridge.request("backend.request", { method: "GET", path: `/api/v1/jobs/${jobId}/joined` });
      expect(joinedAfter.status).toBe(200);
    });

    it("returns 404 JOB_NOT_FOUND for an unknown job", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await bridge.request("backend.request", {
        method: "POST",
        path: "/api/v1/jobs/does-not-exist/join",
        body: {},
      });
      expect(result.status).toBe(404);
      expect((result.body as { error: { code: string } }).error.code).toBe("JOB_NOT_FOUND");
    });
  });

  // I4: the echoed JobResponse.request is the whitelisted per-clip
  // GenerateRequest — never the raw chain body (the "test-green / device-
  // invisible" Join bug's structural fix).
  describe("JobResponse.request whitelist (I4)", () => {
    it("a V2V chain job's echoed request drops source_video and the chain-only fields", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const generated = await bridge.request("backend.request", {
        method: "POST",
        path: "/api/v1/generate/chain",
        body: {
          prompt: "continue this video",
          width: 512,
          height: 320,
          frame_rate: 24,
          seed: 7,
          overlap_frames: 3,
          overlap_strength: 0.5,
          clips: [{ num_frames: 49 }],
          source_video: { video_id: "mock-video-1", context_frames: 25 },
        },
      });
      const jobId = (generated.body as { job_id: string }).job_id;
      const got = await bridge.request("backend.request", { method: "GET", path: `/api/v1/jobs/${jobId}` });
      const req = (got.body as { request: Record<string, unknown> }).request;

      // Chain-only fields are absent; is_v2v is the only V2V signal.
      expect(req.source_video).toBeUndefined();
      expect(req.clips).toBeUndefined();
      expect(req.overlap_frames).toBeUndefined();
      expect(req.overlap_strength).toBeUndefined();
      // Per-clip GenerateRequest shape (num_frames from clip 0, crop deferred).
      expect(req.num_frames).toBe(49);
      expect(req.crop_output).toBeNull();
      expect(req.loras).toEqual([]);
      expect(req.reference_video_id).toBeNull();
      expect(req.pipeline).toBe("distilled");
      expect(req.negative_prompt).toBe("");
      expect((got.body as { is_v2v: boolean }).is_v2v).toBe(true);
    });

    it("a plain /generate job echoes its own GenerateRequest fields", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const generated = await bridge.request("backend.request", {
        method: "POST",
        path: "/api/v1/generate",
        body: { prompt: "a cat", width: 384, height: 256, num_frames: 17, frame_rate: 24, seed: 5 },
      });
      const jobId = (generated.body as { job_id: string }).job_id;
      const got = await bridge.request("backend.request", { method: "GET", path: `/api/v1/jobs/${jobId}` });
      const req = (got.body as { request: Record<string, unknown> }).request;
      expect(req.prompt).toBe("a cat");
      expect(req.num_frames).toBe(17);
      expect(req.seed).toBe(5);
      expect((got.body as { is_v2v: boolean }).is_v2v).toBe(false);
    });
  });

  describe("settings.get / settings.set (contract v4, M7b)", () => {
    it("settings.get returns the configured baseUrl option by default", async () => {
      const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:9999" });
      await expect(bridge.request("settings.get", {})).resolves.toEqual({ baseUrl: "http://127.0.0.1:9999" });
    });

    it("settings.set with no baseUrl is a no-op read", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const before = await bridge.request("settings.get", {});
      await expect(bridge.request("settings.set", {})).resolves.toEqual(before);
    });

    it("settings.set updates the baseUrl, reflected by both settings.get and backend.getBaseUrl", async () => {
      const bridge = createMockBridge({ delayMs: 0 });

      await expect(bridge.request("settings.set", { baseUrl: "http://192.168.1.50:18620" })).resolves.toEqual({
        baseUrl: "http://192.168.1.50:18620",
      });
      await expect(bridge.request("settings.get", {})).resolves.toEqual({ baseUrl: "http://192.168.1.50:18620" });
      await expect(bridge.request("backend.getBaseUrl", {})).resolves.toEqual({ baseUrl: "http://192.168.1.50:18620" });
    });

    it("settings.set strips a trailing slash", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      await expect(bridge.request("settings.set", { baseUrl: "http://127.0.0.1:18620/" })).resolves.toEqual({
        baseUrl: "http://127.0.0.1:18620",
      });
    });

    it("settings.set rejects with BAD_REQUEST for an unparseable URL", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      await expect(bridge.request("settings.set", { baseUrl: "not a url" })).rejects.toMatchObject({
        code: "BAD_REQUEST",
      });
    });

    it("settings.set rejects with BAD_REQUEST for a non-http(s) scheme", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      await expect(bridge.request("settings.set", { baseUrl: "ftp://127.0.0.1:18620" })).rejects.toMatchObject({
        code: "BAD_REQUEST",
      });
    });

    it("a failed settings.set leaves the previously configured baseUrl untouched", async () => {
      const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
      await expect(bridge.request("settings.set", { baseUrl: "nonsense" })).rejects.toBeInstanceOf(Error);
      await expect(bridge.request("settings.get", {})).resolves.toEqual({ baseUrl: "http://127.0.0.1:18620" });
    });
  });
});

// B-1 (2026-08-12, adversarial review): `MOCK_CONFIG_BODY.limits` (this
// file's `GET /config` fixture) and `modes/single/defaultConfig.ts`'s
// `FALLBACK_APP_CONFIG.limits` (the built-in fallback used when the real
// `/config` can't be reached) are two independently hand-maintained mirrors
// of the same real backend contract — the mock had silently drifted behind
// it once already (missing `chain_comfort_token_budget`, caught only by
// manual review, not by any test). The `AppShell.controlLora.test.tsx`
// `MOCK_LORAS` <-> `MOCK_CONFIG_BODY.model.ic_loras` guard is the precedent
// for catching this class of drift.
//
// The comparison is deliberately ONE-DIRECTIONAL (mock has every fallback
// key, not "the two key sets are equal"): `chain_comfort_token_budget` is
// optional on `AppLimits` (`api/types.ts`), and `MOCK_CONFIG_BODY.limits`
// legitimately carries `low_vram_disabled_required` — a mock-only fixture
// key with no `FALLBACK_APP_CONFIG` counterpart, since the fallback only
// ever needs to seed the form, never gate low-VRAM mode. Requiring exact
// equality would force one side to grow a key it has no use for just to
// satisfy the test.
describe("MOCK_CONFIG_BODY.limits <-> FALLBACK_APP_CONFIG.limits parity (B-1)", () => {
  it("the mock /config fixture carries every key the fallback config declares", () => {
    const mockKeys = new Set(Object.keys(MOCK_CONFIG_BODY.limits));
    const missing = Object.keys(FALLBACK_APP_CONFIG.limits).filter((key) => !mockKeys.has(key));
    expect(missing).toEqual([]);
  });
});
