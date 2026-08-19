import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { createMockBridge } from "../../bridge/mockBridge";
import { createApiClient } from "../../api/client";
import type { ApiClient } from "../../api/client";
import { useGenerationSubmit } from "./useGeneration";
import type { GenerateChainRequest, GenerateRequest } from "../../api/types";

// U-R1: `useGenerationSubmit` is submit-only — it fires POST /generate(/chain)
// and hands the accepted job_id to `onSubmitted`; polling/insert/completion
// now live in the job ledger (`jobs/JobLedger.tsx` + `JobsContext`), covered by
// their own tests. Real timers throughout (mock bridge delay is 0ms here).

const REQUEST: GenerateRequest = {
  prompt: "a cat riding a skateboard",
  width: 384,
  height: 256,
  num_frames: 17,
  frame_rate: 24,
  seed: 1,
};

const CHAIN_REQUEST: GenerateChainRequest = {
  prompt: "a cat riding a skateboard, then a dog joins in",
  width: 384,
  height: 256,
  frame_rate: 24,
  seed: 1,
  overlap_frames: 3,
  overlap_strength: 0.5,
  clips: [{ num_frames: 17 }, { num_frames: 17 }],
};

function setup(runningPollCount = 1) {
  const mockBridge = createMockBridge({ delayMs: 0, runningPollCount });
  const apiClient = createApiClient(mockBridge);
  return { mockBridge, apiClient };
}

describe("useGenerationSubmit", () => {
  it("starts idle", () => {
    const { apiClient } = setup();
    const { result } = renderHook(() => useGenerationSubmit({ apiClient }));
    expect(result.current.submitState.phase).toBe("idle");
  });

  it("submits /generate, calls onSubmitted with the job_id, then returns to idle", async () => {
    const { apiClient } = setup();
    const onSubmitted = vi.fn();
    const { result } = renderHook(() => useGenerationSubmit({ apiClient, onSubmitted }));

    act(() => result.current.submit(REQUEST));
    expect(result.current.submitState.phase).toBe("submitting");

    await waitFor(() => expect(result.current.submitState.phase).toBe("idle"));
    expect(onSubmitted).toHaveBeenCalledTimes(1);
    expect(onSubmitted).toHaveBeenCalledWith(expect.any(String));
  });

  it("submitChain goes through POST /generate/chain", async () => {
    const { apiClient } = setup();
    const generateChainSpy = vi.spyOn(apiClient, "generateChain");
    const onSubmitted = vi.fn();
    const { result } = renderHook(() => useGenerationSubmit({ apiClient, onSubmitted }));

    act(() => result.current.submitChain(CHAIN_REQUEST));

    await waitFor(() => expect(result.current.submitState.phase).toBe("idle"));
    expect(generateChainSpy).toHaveBeenCalledWith(CHAIN_REQUEST);
    expect(onSubmitted).toHaveBeenCalledTimes(1);
  });

  it("holds 'submitting' until onSubmitted resolves (MN-6 double-click window)", async () => {
    const { apiClient } = setup();
    let release!: () => void;
    const onSubmitted = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          release = resolve;
        }),
    );
    const { result } = renderHook(() => useGenerationSubmit({ apiClient, onSubmitted }));

    act(() => result.current.submit(REQUEST));
    await waitFor(() => expect(onSubmitted).toHaveBeenCalled());
    // Still submitting while onSubmitted (the ledger refresh) is pending.
    expect(result.current.submitState.phase).toBe("submitting");

    act(() => release());
    await waitFor(() => expect(result.current.submitState.phase).toBe("idle"));
  });

  it("surfaces phase 'error' with code JOB_BUSY on a 409", async () => {
    const { apiClient } = setup(50); // never settles — keeps the single slot occupied
    await apiClient.generate(REQUEST); // occupy the slot directly

    const onSubmitted = vi.fn();
    const { result } = renderHook(() => useGenerationSubmit({ apiClient, onSubmitted }));

    act(() => result.current.submit(REQUEST));

    await waitFor(() => expect(result.current.submitState.phase).toBe("error"));
    if (result.current.submitState.phase !== "error") throw new Error("unreachable");
    expect(result.current.submitState.code).toBe("JOB_BUSY");
    expect(onSubmitted).not.toHaveBeenCalled();
  });

  it("fires onFailed (not onSubmitted) when the submit rejects — X2(a)", async () => {
    const { apiClient } = setup(50); // never settles — keeps the single slot occupied
    await apiClient.generate(REQUEST); // occupy the slot so the next submit 409s

    const onSubmitted = vi.fn();
    const onFailed = vi.fn();
    const { result } = renderHook(() => useGenerationSubmit({ apiClient, onSubmitted, onFailed }));

    act(() => result.current.submit(REQUEST));

    await waitFor(() => expect(result.current.submitState.phase).toBe("error"));
    // The rollback hook fires on the failure; the success hook never does.
    expect(onFailed).toHaveBeenCalledTimes(1);
    expect(onSubmitted).not.toHaveBeenCalled();
  });

  it("does not fire onFailed on a successful submit — X2(a)", async () => {
    const { apiClient } = setup();
    const onFailed = vi.fn();
    const { result } = renderHook(() => useGenerationSubmit({ apiClient, onFailed }));

    act(() => result.current.submit(REQUEST));

    await waitFor(() => expect(result.current.submitState.phase).toBe("idle"));
    expect(onFailed).not.toHaveBeenCalled();
  });

  it("fires onFailed even when the hook unmounted mid-submit (runs before the mounted guard) — X2(a)", async () => {
    // The rollback cleans up MODULE-level reservation state, so it must run even
    // if the screen unmounted (a tab switch) while the submit was in flight —
    // that's exactly why `onFailed` fires ahead of the `mountedRef` guard.
    let rejectFire!: (err: unknown) => void;
    const apiClient = {
      generate: () => new Promise<{ job_id: string }>((_, reject) => {
        rejectFire = reject;
      }),
      generateChain: () => Promise.reject(new Error("unused")),
    } as unknown as ApiClient;

    const onFailed = vi.fn();
    const { result, unmount } = renderHook(() => useGenerationSubmit({ apiClient, onFailed }));

    act(() => result.current.submit(REQUEST));
    // Unmount BEFORE the submit rejects.
    unmount();
    await act(async () => {
      rejectFire(new Error("boom"));
      await Promise.resolve();
    });

    expect(onFailed).toHaveBeenCalledTimes(1);
  });

  it("reset clears an error back to idle", async () => {
    const { apiClient } = setup(50);
    await apiClient.generate(REQUEST);
    const { result } = renderHook(() => useGenerationSubmit({ apiClient }));

    act(() => result.current.submit(REQUEST));
    await waitFor(() => expect(result.current.submitState.phase).toBe("error"));

    act(() => result.current.reset());
    expect(result.current.submitState.phase).toBe("idle");
  });
});
