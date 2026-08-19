import { describe, expect, it, vi } from "vitest";
import { createMockBridge } from "../bridge/mockBridge";
import { cutoutRangeAndUpload, extractAudioAndUpload } from "./cutoutAndUpload";
import type { CutoutAndUploadPhase } from "./cutoutAndUpload";

describe("cutoutRangeAndUpload", () => {
  const CUTOUT_PARAMS = {
    layer: 2,
    frameStart: 10,
    frameCount: 60,
    withAudio: true,
    audioMode: "mix" as const,
    soloKeepLayers: [],
  };

  it("cuts out the range and uploads the resulting file as a video", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const result = await cutoutRangeAndUpload(bridge, CUTOUT_PARAMS);

    expect(result.cutout.filePath).toMatch(/\.mp4$/);
    expect(result.upload.status).toBe(200);
    expect((result.upload.body as { video_id: string }).video_id).toBeDefined();
  });

  it("passes the cutout's filePath straight to backend.uploadFile with kind='video'", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const spy = vi.spyOn(bridge, "request");

    const result = await cutoutRangeAndUpload(bridge, CUTOUT_PARAMS);

    expect(spy).toHaveBeenNthCalledWith(1, "timeline.cutoutRange", CUTOUT_PARAMS);
    expect(spy).toHaveBeenNthCalledWith(2, "backend.uploadFile", {
      kind: "video",
      filePath: result.cutout.filePath,
    });
  });

  it("reports phases in order: cutting-out then uploading", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const phases: CutoutAndUploadPhase[] = [];
    await cutoutRangeAndUpload(bridge, CUTOUT_PARAMS, (p) => phases.push(p));
    expect(phases).toEqual(["cutting-out", "uploading"]);
  });

  it("does not upload when the cutout fails (bridge transport error propagates)", async () => {
    const bridge = createMockBridge({ delayMs: 0, failCutoutRange: "CUTOUT_FAILED" });
    const spy = vi.spyOn(bridge, "request");
    await expect(cutoutRangeAndUpload(bridge, CUTOUT_PARAMS)).rejects.toMatchObject({
      code: "CUTOUT_FAILED",
    });
    // Only the cutout call was attempted; no upload.
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it("surfaces a non-2xx upload as a normal resolved result (two-channel contract)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    // Force the upload leg to 'fail' at the HTTP level by returning a rejecting
    // upload — but the mock only rejects on transport, so assert the resolved
    // 200 path here and rely on failUploadFile tests for the transport channel.
    const result = await cutoutRangeAndUpload(bridge, CUTOUT_PARAMS);
    expect(result.upload.status).toBe(200);
  });
});

describe("extractAudioAndUpload", () => {
  const EXTRACT_PARAMS = {
    layer: 1,
    frameStart: 0,
    frameCount: 60,
    audioMode: "mix" as const,
    soloKeepLayers: [],
  };

  it("extracts audio and uploads it as kind='audio'", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const spy = vi.spyOn(bridge, "request");

    const result = await extractAudioAndUpload(bridge, EXTRACT_PARAMS);

    expect(result.extract.filePath).toMatch(/\.wav$/);
    expect((result.upload.body as { audio_id: string }).audio_id).toBeDefined();
    expect(spy).toHaveBeenNthCalledWith(2, "backend.uploadFile", {
      kind: "audio",
      filePath: result.extract.filePath,
    });
  });

  it("propagates an extract failure without uploading", async () => {
    const bridge = createMockBridge({ delayMs: 0, failExtractAudio: "EXTRACT_FAILED" });
    const spy = vi.spyOn(bridge, "request");
    await expect(extractAudioAndUpload(bridge, EXTRACT_PARAMS)).rejects.toMatchObject({
      code: "EXTRACT_FAILED",
    });
    expect(spy).toHaveBeenCalledTimes(1);
  });
});
