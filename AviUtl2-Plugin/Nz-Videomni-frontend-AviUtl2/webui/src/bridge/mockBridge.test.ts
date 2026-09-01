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

    it("rejects fewer than 2 clips when none of source_video/source_audio/retake/reference_video_id/end_source is set", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await postChain(bridge, { ...CHAIN_BODY, clips: [{ num_frames: 49 }] });
      expect(result.status).toBe(422);
    });

    // §3-116: the real schema's clip-count floor exempts FIVE fields, not just
    // source_video/source_audio (see the `hasSource` comment in mockBridge.ts
    // for the exact citation of `api/models.py`'s `GenerateChainRequest`).
    // Table-driven: one 1-clip case per exception (202), plus the "none of
    // them" case that must still 422. Backend-side pytest counterparts:
    // `tests/test_chain.py::test_chain_rejects_single_clip` (plain 1-clip
    // chain, 422), `tests/test_chain_reference.py::
    // test_depth_reference_with_one_clip_accepted` (reference_video_id + a
    // 1-clip chain, 202), and `tests/test_end_source_chain.py::
    // test_end_source_allows_a_single_clip` (end_source + a 1-clip chain,
    // 202) — retake and source_video/source_audio have no dedicated
    // cross-reference here, but are covered by the same `model_validator`.
    const CLIP_COUNT_EXCEPTION_CASES: ReadonlyArray<{ name: string; body: object; status: 202 | 422 }> = [
      { name: "source_video", body: { source_video: { video_id: "vid-1", context_frames: 25 } }, status: 202 },
      { name: "source_audio", body: { source_audio: { audio_id: "aud-1" } }, status: 202 },
      { name: "retake", body: { retake: { video_id: "vid-1", window_start_sec: 2 } }, status: 202 },
      {
        name: "reference_video_id",
        // The real schema also requires `reference_video_id` to carry at
        // least one lora ("reference_video_id requires at least one lora",
        // `api/models.py`) — `loras` is added so this case matches what the
        // real API actually accepts, not just this fixture's own (looser)
        // `hasSource` check.
        body: { reference_video_id: "vid-3", loras: [{ name: "style-a", strength: 0.8 }] },
        status: 202,
      },
      { name: "end_source", body: { end_source: { video_id: "vid-2", context_frames: 24 } }, status: 202 },
      { name: "none of the five", body: {}, status: 422 },
    ];

    for (const { name, body, status } of CLIP_COUNT_EXCEPTION_CASES) {
      it(`1-clip chain with '${name}' set -> ${status}`, async () => {
        const bridge = createMockBridge({ delayMs: 0 });
        const result = await postChain(bridge, { ...CHAIN_BODY, clips: [{ num_frames: 49 }], ...body });
        expect(result.status).toBe(status);
      });
    }

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

  // 2026-08-31: the one-directional key check above cannot see a CONTENT drift
  // inside `comfort_budgets`, and that block is a nested structure both
  // fixtures now carry. This pins the two FRONTEND mirrors against each other
  // (both engine families, both budget values, both `requires` shapes) — it
  // does NOT pin against the backend's own `_default_comfort_budgets()`; that
  // parity is checked manually, or by the backend's own tests, not from here.
  it("both fixtures publish an identical comfort_budgets block", () => {
    expect(FALLBACK_APP_CONFIG.limits.comfort_budgets).toEqual(MOCK_CONFIG_BODY.limits.comfort_budgets);
  });

  it("the ltx row requires exactly the five all-on acceleration fields, and ltx25 requires nothing", () => {
    // ⚠ `ltx`'s single row is CONDITIONAL on purpose: LTX 2.3's default
    // configuration has no smart line at all (its comfort boundary tracks the
    // VAE decoder's chunk-count steps, not the token count), so it falls to
    // `spill_free_frames`. A `requires: {}` row here would be a regression,
    // not an improvement — hence the exact key-set assertion.
    const ltxRow = MOCK_CONFIG_BODY.limits.comfort_budgets.ltx.rows[0];
    expect(Object.keys(ltxRow?.requires ?? {}).sort()).toEqual([
      "attention_backend",
      "block_swap_prefetch",
      "fused_gguf_dequant_kernel",
      "keep_resident",
      "vae_mode",
    ]);
    expect(MOCK_CONFIG_BODY.limits.comfort_budgets.ltx25.rows[0]?.requires).toEqual({});
  });
});

// §3-98 P5: `GET /models` publishes, per base model, the feature names that
// base model's ENGINE cannot run. The fixture must carry it the same way the
// real endpoint does — the WebUI's greying reads these exact strings, so a
// fixture that omitted the key (or invented its own names) would let the whole
// feature ship broken while every UI test passed. Same "mock mirrors the real
// contract" discipline as the config-parity guard above.
describe("GET /models — unsupported_features (§3-98 P5)", () => {
  async function getModels() {
    const bridge = createMockBridge({ delayMs: 0 });
    const result = (await bridge.request("backend.request", {
      method: "GET",
      path: "/api/v1/models",
    })) as { status: number; body: { base_models: { id: string; unsupported_features?: string[] }[] } };
    expect(result.status).toBe(200);
    return result.body;
  }

  it("emits the key for EVERY base model, empty where there is nothing to report", async () => {
    const body = await getModels();
    // Present-and-empty is a different fact from absent: absent means "this
    // server is too old to say", which the WebUI must read as "no limits".
    for (const base of body.base_models) {
      expect(Array.isArray(base.unsupported_features), base.id).toBe(true);
    }
    expect(body.base_models.find((b) => b.id === "LTX23")?.unsupported_features).toEqual([]);
  });

  it("names LTX 2.5's v1 scope with the server's own feature names", async () => {
    const body = await getModels();
    const features = body.base_models.find((b) => b.id === "LTX25")?.unsupported_features ?? [];
    // The chain family (whole endpoints) and the request-field half — the two
    // halves `services/engines/ltx25/adapter.py` builds `UNSUPPORTED_FEATURES`
    // from. Spelt out rather than counted so a rename on either side shows up.
    expect(features).toEqual(
      expect.arrayContaining([
        "two_stage_hq",
        "prune_vaed",
      ]),
    );
    // §3-102 (LTX 2.5 Chained, first stage): `chain` is GONE — the engine
    // chains now, and its absence is what un-greys the Chained tab. §3-102
    // second stage: `v2v` and `a2v` left with it — the engine takes a source
    // video and a source audio track now, which is what un-greys the two Chain
    // material panels and the Batch A2V section. §3-102 third stage: `loras`
    // and `reference_video` left too — Style LoRA and IC-LoRA (long IC-LoRA
    // included) run on this engine now, which is what un-greys the reference
    // panels on Single and Chained. All five asserted negatively, because
    // `arrayContaining` above would not notice them coming back.
    // Retake increment: `retake` left too, and it is asserted NEGATIVELY for
    // the reason the five below are -- dropping it from the `arrayContaining`
    // list alone would keep this test green whether the fixture was updated or
    // not, and a stale fixture would go on greying out the Edit tab's 撮り直し
    // sub-tab (and the timeline right-click route into it) for a mode that runs.
    expect(features).not.toContain("retake");
    // End source increment: `end_source` left too, and it was the LAST
    // chain-family MODE name on this list — so what LTX 2.5 publishes now is
    // engine-level features only. Asserted NEGATIVELY for the reason the
    // others are: dropping it from the `arrayContaining` list alone would keep
    // this test green whether the fixture was updated or not, and a stale
    // fixture would go on greying out the Chained tab's 素材（末尾） panel for a
    // mode that runs.
    expect(features).not.toContain("end_source");
    // Outpainting increment: `outpaint` left too, and it was the LAST MODE name
    // of ANY family on this list — chain or single. Asserted NEGATIVELY for the
    // reason all the others are, and a stale fixture would go on greying out
    // the Edit tab's 画角拡張 sub-tab for a mode that runs.
    expect(features).not.toContain("outpaint");
    expect(features).not.toContain("chain");
    expect(features).not.toContain("v2v");
    expect(features).not.toContain("a2v");
    expect(features).not.toContain("loras");
    expect(features).not.toContain("reference_video");
    // 高速化第2弾: `keep_resident` left too — the engine keeps its text encoder
    // resident between jobs now. Asserted NEGATIVELY for the reason above and
    // for one more: dropping it from the `arrayContaining` list alone would
    // have kept this test green whether the fixture was updated or not, and a
    // stale fixture would go on greying out a Settings control that works.
    expect(features).not.toContain("keep_resident");
    // 高速化第3弾: and `sage_attention` left, from the same Settings panel and
    // pinned the same way. The engine runs 2.3's sage kernels now.
    expect(features).not.toContain("sage_attention");
    // NAG/VSF increment: `nag` left, and it is asserted NEGATIVELY for the
    // reason every name above it is -- removing it from the `arrayContaining`
    // list alone would keep this test green whether the fixture was updated or
    // not, and a stale fixture would go on greying out the negative-prompt
    // panel (and the method switch and three knobs behind it) on both the
    // Single and the Chained tab, for a feature the server runs.
    expect(features).not.toContain("nag");
  });

  it("declares LTX 2.5 as its own engine family", async () => {
    // It stopped being an `ltx` base model when engine25 shipped (§3-98 P3c:
    // `FAMILY_BY_KV[("ltxv","2.5")] === "ltx25"`).
    const body = await getModels();
    const base = body.base_models.find((b) => b.id === "LTX25") as { engine_family?: string } | undefined;
    expect(base?.engine_family).toBe("ltx25");
  });
});

// §3-102 (LTX 2.5 Chained, first stage): the fixture's chain endpoint now
// refuses the individual MATERIALS the active base model declares it cannot
// use, instead of refusing the whole endpoint. The two properties that matter
// are opposite ones, so both are pinned here:
//
//   * LTX 2.5 — a declared feature arriving non-empty is 422 FEATURE_UNSUPPORTED
//     (the fail-loud backstop for material attached before the switch, which
//     the greyed panels cannot retroactively remove);
//   * LTX 2.3 — declares NOTHING, so every one of those same fields sails
//     through exactly as it always did. That half is load-bearing: the WebUI's
//     whole V2V / A2V / End source / reference Chain suite runs on this
//     fixture, and a refusal keyed on anything but the declared list would take
//     all of it down.
describe("POST /generate/chain — engine feature scope (§3-102)", () => {
  const CHAIN_BODY = {
    prompt: "a cat riding a skateboard, then a dog joins in",
    width: 512,
    height: 320,
    frame_rate: 24,
    seed: -1,
    clips: [{ num_frames: 49 }, { num_frames: 33 }],
  };

  /** Every field the fixture can refuse, with a value that counts as "the user
   * asked for this" — one case per row of `MOCK_CHAIN_FEATURE_FIELDS`. */
  const CASES: ReadonlyArray<{ feature: string; body: object }> = [
    // `end_source` LEFT this table with the End-source increment, together
    // with its row in `MOCK_CHAIN_FEATURE_FIELDS` — the acceptance twin is
    // below (`accepts end_source ...`), which is what makes the move checkable
    // rather than merely an absence.
    //
    // `nag` LEFT WITH THE NAG/VSF INCREMENT, and it was the last row here. The
    // table is NOT left empty: `prune_vaed` takes its place, which is the same
    // move the backend's own test tables made (`vae_mode` is the subject the
    // 2.5 suite switched every refusal test onto). An empty CASES would have
    // silently retired both loops below — the 422 on 2.5 AND the pass-through
    // on 2.3 — leaving the fixture's whole refusal mechanism untested.
    //
    // It is also the first row whose FIELD only became refusable in that same
    // commit: `vae_mode` was declared unsupported all along but had no row in
    // `MOCK_CHAIN_FEATURE_FIELDS` to enforce it with.
    { feature: "prune_vaed", body: { vae_mode: "prune_vaed" } },
  ];

  async function postChain(bridge: ReturnType<typeof createMockBridge>, body: object) {
    return bridge.request("backend.request", { method: "POST", path: "/api/v1/generate/chain", body });
  }

  /** A fixture with LTX 2.5 both installed and loadable, switched to it. */
  async function ltx25Bridge() {
    const bridge = createMockBridge({
      delayMs: 0,
      ltx25Install: "full",
      supportedBaseModels: ["LTX23", "LTX25"],
    });
    const loaded = await bridge.request("backend.request", {
      method: "POST",
      path: "/api/v1/pipeline/load",
      body: { base_model: "LTX25" },
    });
    expect(loaded.status).toBe(200);
    return bridge;
  }

  it("accepts a plain multi-clip chain on LTX 2.5 — the endpoint itself is open now", async () => {
    const bridge = await ltx25Bridge();
    const result = await postChain(bridge, CHAIN_BODY);
    expect(result.status).toBe(202);
  });

  // §3-102 second stage: the two materials that just LEFT the declared list.
  // Asserted as acceptances rather than by their absence from `CASES` above,
  // because a row quietly dropped from a table proves nothing — only a request
  // that actually carries the field and comes back 202 does.
  it("accepts a V2V chain on LTX 2.5 — `source_video` is in scope now", async () => {
    const bridge = await ltx25Bridge();
    const result = await postChain(bridge, {
      ...CHAIN_BODY,
      source_video: { video_id: "vid-1", context_frames: 73 },
    });
    expect(result.status).toBe(202);
  });

  it("accepts an A2V chain on LTX 2.5 — `source_audio` is in scope now", async () => {
    const bridge = await ltx25Bridge();
    const result = await postChain(bridge, {
      ...CHAIN_BODY,
      source_audio: { audio_id: "aud-1" },
    });
    expect(result.status).toBe(202);
  });

  // §3-102 THIRD stage: the two that just left the declared list this time.
  // Same reasoning as the two acceptances above — a row quietly dropped from
  // `CASES` proves nothing, only a 202 on a request that carries the field does.
  it("accepts a Style LoRA chain on LTX 2.5 — `loras` is in scope now", async () => {
    const bridge = await ltx25Bridge();
    const result = await postChain(bridge, {
      ...CHAIN_BODY,
      loras: [{ name: "style-a", strength: 0.8 }],
    });
    expect(result.status).toBe(202);
  });

  it("accepts an IC-LoRA chain on LTX 2.5 — `reference_video_id` is in scope now", async () => {
    const bridge = await ltx25Bridge();
    const result = await postChain(bridge, {
      ...CHAIN_BODY,
      // 384 rather than the shared body's 320: the real server requires a
      // 128-multiple resolution of every request that carries a reference
      // (`api/generate_chain.py`). The fixture does not check it, so spelling
      // it out here keeps the case from being one no real server would take.
      height: 384,
      reference_video_id: "vid-3",
    });
    expect(result.status).toBe(202);
  });

  it("accepts the Single-tab A2V shape on LTX 2.5 — one clip, full_length", async () => {
    // The shape the Single tab and Batch A2V both submit: a ONE-clip chain
    // carrying an audio track. It is only legal because a source is attached
    // (the fixture's own `hasSource` rule), so it is the case that would break
    // first if `source_audio` were still being refused.
    const bridge = await ltx25Bridge();
    const result = await postChain(bridge, {
      ...CHAIN_BODY,
      clips: [{ num_frames: 121 }],
      stage2_window: "full_length",
      source_audio: { audio_id: "aud-1" },
    });
    expect(result.status).toBe(202);
  });

  it("accepts end_source alongside a V2V source on LTX 2.5", async () => {
    // This case used to be the 422 that proved opening V2V had not opened the
    // material still out of scope. The End-source increment INVERTS it, and
    // this particular pairing is the one worth keeping: start material plus end
    // material on one chain is the interpolation the API explicitly allows, and
    // it is exactly what a refusal keyed on the wrong field would break.
    const bridge = await ltx25Bridge();
    const result = await postChain(bridge, {
      ...CHAIN_BODY,
      clips: [{ num_frames: 121 }],
      source_video: { video_id: "vid-1", context_frames: 25 },
      end_source: { video_id: "vid-2", context_frames: 24 },
    });
    expect(result.status).toBe(202);
  });

  it("accepts a NAG chain on LTX 2.5 — the negative prompt is in scope now", async () => {
    // The NAG/VSF increment's acceptance twin, and the reason the removal from
    // `CASES` above proves anything: a row quietly dropped from a table is not
    // evidence, a 202 on a request that actually carries the field is.
    const bridge = await ltx25Bridge();
    const result = await postChain(bridge, {
      ...CHAIN_BODY,
      nag_enabled: true,
      negative_prompt: "blurry, low quality",
      nag_scale: 7.5,
      nag_tau: 3.5,
      nag_alpha: 0.5,
    });
    expect(result.status).toBe(202);
  });

  it("accepts a VSF chain on LTX 2.5 — the method switch is in scope too", async () => {
    const bridge = await ltx25Bridge();
    const result = await postChain(bridge, {
      ...CHAIN_BODY,
      nag_enabled: true,
      negative_prompt: "blurry, low quality",
      neg_method: "vsf",
      vsf_scale: 3.0,
    });
    expect(result.status).toBe(202);
  });

  it("accepts end_source on a single-clip chain (the `in_window` geometry)", async () => {
    // The headline of the End-source increment on the fixture: what used to be
    // this suite's example of a refusal now passes the ruling. §3-116 aligned
    // this fixture's clip-count floor with the real schema (`api/models.py`),
    // so a bare one-clip end-source chain earns 202 here too — see the
    // 5-exception table-driven test above (`CLIP_COUNT_EXCEPTION_CASES`) for
    // the full parity check across all five fields.
    const bridge = await ltx25Bridge();
    const result = await postChain(bridge, {
      ...CHAIN_BODY,
      clips: [{ num_frames: 121 }],
      end_source: { video_id: "vid-2", context_frames: 24 },
    });
    expect(result.status).toBe(202);
  });

  for (const { feature, body } of CASES) {
    it(`refuses '${feature}' on LTX 2.5 with 422 FEATURE_UNSUPPORTED`, async () => {
      const bridge = await ltx25Bridge();
      const result = await postChain(bridge, { ...CHAIN_BODY, ...body });
      expect(result.status).toBe(422);
      const error = (result.body as { error: { code: string; message: string } }).error;
      expect(error.code).toBe("FEATURE_UNSUPPORTED");
      expect(error.message).toContain(feature);
    });

    it(`lets '${feature}' through on LTX 2.3, which declares no restrictions`, async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await postChain(bridge, { ...CHAIN_BODY, ...body });
      expect(result.status).toBe(202);
    });
  }

  it("ignores a field that is present but empty/default", async () => {
    // The WebUI sends the whole schema every time, so `null`, `[]` and
    // `nag_enabled: false` are NOT requests for those features — refusing them
    // would 422 an ordinary chain.
    const bridge = await ltx25Bridge();
    const result = await postChain(bridge, {
      ...CHAIN_BODY,
      source_video: null,
      source_audio: null,
      end_source: null,
      reference_video_id: null,
      loras: [],
      nag_enabled: false,
      // THE TWO ROWS THE NAG/VSF INCREMENT ADDED, at their schema defaults.
      // These are the first refusable fields whose default is a NON-EMPTY
      // STRING, so they are the ones that would 422 an ordinary chain if the
      // rows had been added without predicates of their own — which is exactly
      // what this assertion is here to catch.
      pipeline: "distilled",
      vae_mode: "default",
    });
    expect(result.status).toBe(202);
  });

  it("refuses BEFORE request validation — an unrunnable request is not a fixable one", async () => {
    // A 1-clip chain is also invalid, but telling the user to add a clip to a
    // request that could never have run sends them to fix the wrong thing.
    // The subject was `end_source` until that mode was implemented, then `nag`
    // until the NAG/VSF increment; `vae_mode` is now the field the fixture
    // still refuses on this base model — and it needs no companion, so the body
    // under test is the plain chain plus one value.
    const bridge = await ltx25Bridge();
    const result = await postChain(bridge, {
      ...CHAIN_BODY,
      clips: [{ num_frames: 49 }],
      vae_mode: "prune_vaed",
    });
    expect(result.status).toBe(422);
    expect((result.body as { error: { code: string } }).error.code).toBe("FEATURE_UNSUPPORTED");
  });

  it("creates no job when it refuses", async () => {
    const bridge = await ltx25Bridge();
    await postChain(bridge, {
      ...CHAIN_BODY,
      // Followed the test above onto `vae_mode` for the same reason.
      vae_mode: "prune_vaed",
    });
    // The refusal must not take the single-job slot with it: a following, valid
    // chain has to be accepted rather than earning a 409 JOB_BUSY.
    const next = await postChain(bridge, CHAIN_BODY);
    expect(next.status).toBe(202);
  });
});
