import { describe, expect, it, vi } from "vitest";
import { createMockBridge } from "../bridge/mockBridge";
import type { NativeBridge } from "../bridge";
import { BackendApiError, createApiClient } from "./client";
import type { GenerateRequest } from "./types";

const BASE_REQUEST: GenerateRequest = {
  prompt: "a cat riding a skateboard",
  width: 384,
  height: 256,
  num_frames: 17,
  frame_rate: 24,
  seed: 42,
};

describe("api/client", () => {
  it("getStatus() returns the status fixture with a queue block", async () => {
    const client = createApiClient(createMockBridge({ delayMs: 0 }));

    const status = await client.getStatus();

    expect(status.server).toBe("running");
    expect(status.queue).toMatchObject({ pending: 0, running: 0 });
  });

  it("getConfig() returns generation presets/defaults/limits/upload", async () => {
    const client = createApiClient(createMockBridge({ delayMs: 0 }));

    const config = await client.getConfig();

    expect(config.generation_defaults).toMatchObject({ width: 1280, height: 768, num_frames: 361 });
    expect(config.generation_presets.smoke_test).toMatchObject({ width: 384, height: 256, num_frames: 17 });
    expect(config.limits.max_num_frames).toBe(481);
    expect(config.upload.allowed_image_extensions).toContain(".png");
  });

  it("generate() returns 202 accepted with a job_id, then getJob() progresses queued -> running -> completed", async () => {
    const client = createApiClient(createMockBridge({ delayMs: 0, runningPollCount: 2 }));

    const accepted = await client.generate(BASE_REQUEST);
    expect(accepted.status).toBe("queued");
    expect(accepted.job_id).toBeTruthy();

    const poll1 = await client.getJob(accepted.job_id);
    expect(poll1.status).toBe("queued");

    const poll2 = await client.getJob(accepted.job_id);
    expect(poll2.status).toBe("running");
    expect(poll2.progress).toBeGreaterThan(0);
    expect(poll2.progress).toBeLessThan(1);

    const poll3 = await client.getJob(accepted.job_id);
    expect(poll3.status).toBe("running");

    const poll4 = await client.getJob(accepted.job_id);
    expect(poll4.status).toBe("completed");
    expect(poll4.progress).toBe(1);
    expect(poll4.result).not.toBeNull();
    expect(poll4.result?.video_url).toBe(`/api/v1/jobs/${accepted.job_id}/video`);
  });

  it("generate() rejects with JOB_BUSY (409) while another job is in flight", async () => {
    const client = createApiClient(createMockBridge({ delayMs: 0, runningPollCount: 5 }));

    await client.generate(BASE_REQUEST);

    await expect(client.generate(BASE_REQUEST)).rejects.toMatchObject({
      code: "JOB_BUSY",
      httpStatus: 409,
    });
  });

  it("generate() normalizes a validation error (empty prompt) into BackendApiError", async () => {
    const client = createApiClient(createMockBridge({ delayMs: 0 }));

    await expect(client.generate({ ...BASE_REQUEST, prompt: "" })).rejects.toBeInstanceOf(BackendApiError);
    await expect(client.generate({ ...BASE_REQUEST, prompt: "" })).rejects.toMatchObject({
      code: "VALIDATION_ERROR",
      httpStatus: 422,
    });
  });

  it("getJob() with an unknown id normalizes to JOB_NOT_FOUND (404)", async () => {
    const client = createApiClient(createMockBridge({ delayMs: 0 }));

    await expect(client.getJob("no-such-job")).rejects.toMatchObject({
      code: "JOB_NOT_FOUND",
      httpStatus: 404,
    });
  });

  it("normalizes a bridge transport failure (BACKEND_UNREACHABLE) to BackendApiError with httpStatus 0", async () => {
    const client = createApiClient(createMockBridge({ delayMs: 0, backendUnreachable: true }));

    await expect(client.getStatus()).rejects.toBeInstanceOf(BackendApiError);
    await expect(client.getStatus()).rejects.toMatchObject({
      code: "BACKEND_UNREACHABLE",
      httpStatus: 0,
    });
  });

  it("deleteJob() on a completed job returns deleted:true", async () => {
    const client = createApiClient(createMockBridge({ delayMs: 0, runningPollCount: 1 }));

    const accepted = await client.generate(BASE_REQUEST);
    await client.getJob(accepted.job_id); // queued
    await client.getJob(accepted.job_id); // running
    const job = await client.getJob(accepted.job_id); // completed
    expect(job.status).toBe("completed");

    const result = await client.deleteJob(accepted.job_id);
    expect(result).toMatchObject({ job_id: accepted.job_id, deleted: true });
  });

  it("deleteJob() on a still-running job returns cancel_requested:true instead of deleting it, and the job later settles on cancelled", async () => {
    const client = createApiClient(createMockBridge({ delayMs: 0, runningPollCount: 1 }));

    const accepted = await client.generate(BASE_REQUEST);
    await client.getJob(accepted.job_id); // queued

    const result = await client.deleteJob(accepted.job_id);
    expect(result.cancel_requested).toBe(true);

    await client.getJob(accepted.job_id); // still running for one more poll
    const afterCancel = await client.getJob(accepted.job_id);
    expect(afterCancel.status).toBe("cancelled");
  });

  it("joinJob() forwards timeoutMs: 120_000 to backend.request (contract v4.1)", async () => {
    const requestSpy = vi.fn(() =>
      Promise.resolve({
        status: 200,
        body: { job_id: "job-1", joined_path: "outputs/job-1/joined.mp4", join_mode: "concat_crossfade", source_normalized: true },
      }),
    );
    const spyBridge = { request: requestSpy, on: () => () => {} } as unknown as NativeBridge;
    const client = createApiClient(spyBridge);

    await client.joinJob("job-1");

    expect(requestSpy).toHaveBeenCalledWith(
      "backend.request",
      expect.objectContaining({ method: "POST", path: "/api/v1/jobs/job-1/join", timeoutMs: 120_000 }),
    );
  });

  it("other calls (e.g. getStatus) do not set a timeoutMs override", async () => {
    const requestSpy = vi.fn((_method: string, _params: unknown) => Promise.resolve({ status: 200, body: { server: "running" } }));
    const spyBridge = { request: requestSpy, on: () => () => {} } as unknown as NativeBridge;
    const client = createApiClient(spyBridge);

    await client.getStatus();

    const call = requestSpy.mock.calls[0];
    if (!call) throw new Error("expected requestSpy to have been called");
    const params = call[1] as Record<string, unknown>;
    expect("timeoutMs" in params).toBe(false);
  });

  it("a prompt containing the __MOCK_FAIL__ marker settles on failed with an error message", async () => {
    const client = createApiClient(createMockBridge({ delayMs: 0, runningPollCount: 1 }));

    const accepted = await client.generate({ ...BASE_REQUEST, prompt: "__MOCK_FAIL__ trigger" });
    await client.getJob(accepted.job_id); // queued
    await client.getJob(accepted.job_id); // running
    const finalJob = await client.getJob(accepted.job_id); // failed
    expect(finalJob.status).toBe("failed");
    expect(finalJob.error).toContain("GENERATION_FAILED");
  });
});
