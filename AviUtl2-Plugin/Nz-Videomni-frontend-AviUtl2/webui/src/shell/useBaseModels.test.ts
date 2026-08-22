import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { BackendApiError, createApiClient } from "../api/client";
import type { ApiClient } from "../api/client";
import { createMockBridge } from "../bridge/mockBridge";
import { useBaseModels } from "./useBaseModels";
import type { BaseModelSwitchOutcome } from "./useBaseModels";

/** A client that answers `GET /models` from the mock bridge (so the option
 * list is realistic) but fails `POST /pipeline/load` with a specific backend
 * error. Used for the two 409 guards, which the fixture server cannot produce
 * on its own: its loads are instantaneous (no `PIPELINE_LOADING` window) and
 * driving a real job in just to reach `JOB_BUSY` would test the fixture's job
 * bookkeeping rather than this hook's classification. */
function clientRejectingLoadWith(err: unknown): ApiClient {
  const real = createApiClient(createMockBridge({ delayMs: 0 }));
  return {
    ...real,
    loadPipeline: () => Promise.reject(err),
  };
}

async function renderReady(apiClient: ApiClient) {
  const view = renderHook(() => useBaseModels({ apiClient }));
  await waitFor(() => expect(view.result.current.options.length).toBeGreaterThan(0));
  return view;
}

describe("useBaseModels", () => {
  it("lists base_models[] in server order and reports the active one", async () => {
    const { result } = await renderReady(createApiClient(createMockBridge({ delayMs: 0 })));

    expect(result.current.options.map((o) => o.id)).toEqual(["LTX23", "LTX25"]);
    expect(result.current.options[0]).toMatchObject({
      displayName: "LTX 2.3",
      installed: true,
      present: true,
      missingCategories: [],
    });
    // The default fixture install of LTX 2.5 is partial: selectable (present),
    // but not complete.
    expect(result.current.options[1]).toMatchObject({ installed: false, present: true });
    expect(result.current.options[1]?.missingCategories).toContain("audio");
    expect(result.current.current).toBe("LTX23");
    expect(result.current.switching).toBe(false);
  });

  it("a successful switch moves `current` and re-reads GET /models", async () => {
    const apiClient = createApiClient(
      createMockBridge({ delayMs: 0, ltx25Install: "full", supportedBaseModels: ["LTX23", "LTX25"] }),
    );
    const loadPipeline = vi.spyOn(apiClient, "loadPipeline");
    const { result } = await renderReady(apiClient);

    await act(async () => {
      await expect(result.current.switchBaseModel("LTX25")).resolves.toEqual({ kind: "switched", id: "LTX25" });
    });

    expect(result.current.current).toBe("LTX25");
    expect(result.current.switching).toBe(false);
    // Switching IS loading (§6.1), with no per-category selection of its own.
    expect(loadPipeline).toHaveBeenCalledWith({}, "LTX25");
    // The refetch is what makes the `active` flag follow the swap.
    expect(result.current.options.find((o) => o.id === "LTX25")).toBeDefined();
  });

  it("a base model with nothing on disk is refused locally — no request, no movement", async () => {
    const apiClient = createApiClient(createMockBridge({ delayMs: 0, ltx25Install: "none" }));
    const loadPipeline = vi.spyOn(apiClient, "loadPipeline");
    const { result } = await renderReady(apiClient);

    await act(async () => {
      await expect(result.current.switchBaseModel("LTX25")).resolves.toEqual({
        kind: "not-installed",
        id: "LTX25",
        displayName: "LTX 2.5",
      });
    });

    expect(loadPipeline).not.toHaveBeenCalled();
    expect(result.current.current).toBe("LTX23");
  });

  it("a partial install is NOT short-circuited — the server gets to say why", async () => {
    const apiClient = createApiClient(createMockBridge({ delayMs: 0 }));
    const loadPipeline = vi.spyOn(apiClient, "loadPipeline");
    const { result } = await renderReady(apiClient);

    let outcome: BaseModelSwitchOutcome | undefined;
    await act(async () => {
      outcome = await result.current.switchBaseModel("LTX25");
    });

    expect(loadPipeline).toHaveBeenCalledWith({}, "LTX25");
    // 422 carries the server's `detail` through untouched.
    expect(outcome?.kind).toBe("rejected");
    expect(outcome?.kind === "rejected" ? outcome.reason : "").toContain("LTX 2.5エンジンは次段階");
    expect(result.current.current).toBe("LTX23");
  });

  it("409 JOB_BUSY is its own outcome and reverts the selection", async () => {
    const apiClient = clientRejectingLoadWith(
      new BackendApiError("JOB_BUSY", "A job is already running (Phase 1 allows one concurrent job)", 409),
    );
    const { result } = await renderReady(apiClient);

    await act(async () => {
      await expect(result.current.switchBaseModel("LTX25")).resolves.toEqual({ kind: "busy" });
    });

    expect(result.current.current).toBe("LTX23");
    expect(result.current.switching).toBe(false);
  });

  it("409 PIPELINE_LOADING is distinguished from JOB_BUSY", async () => {
    const apiClient = clientRejectingLoadWith(
      new BackendApiError(
        "PIPELINE_LOADING",
        "The pipeline is already loading (モデルの読み込み中です)",
        409,
        "現在モデルを読み込んでいます。完了までお待ちください。",
      ),
    );
    const { result } = await renderReady(apiClient);

    await act(async () => {
      await expect(result.current.switchBaseModel("LTX25")).resolves.toEqual({ kind: "loading" });
    });

    expect(result.current.current).toBe("LTX23");
  });

  it("422 falls back to the envelope's message when it carries no detail", async () => {
    const apiClient = clientRejectingLoadWith(
      new BackendApiError("MODEL_FILE_MISSING", "registered model file is missing on disk", 422),
    );
    const { result } = await renderReady(apiClient);

    await act(async () => {
      await expect(result.current.switchBaseModel("LTX25")).resolves.toEqual({
        kind: "rejected",
        reason: "registered model file is missing on disk",
      });
    });
  });

  it("an unknown id (404) and a transport failure both land on the generic outcome", async () => {
    const notFound = clientRejectingLoadWith(
      new BackendApiError("MODEL_NOT_FOUND", "unknown model name 'NOPE' in category 'base_model'", 404),
    );
    const { result: r1 } = await renderReady(notFound);
    await act(async () => {
      await expect(r1.current.switchBaseModel("NOPE")).resolves.toMatchObject({ kind: "failed" });
    });
    expect(r1.current.current).toBe("LTX23");

    const offline = clientRejectingLoadWith(new Error("bridge exploded"));
    const { result: r2 } = await renderReady(offline);
    await act(async () => {
      await expect(r2.current.switchBaseModel("LTX23")).resolves.toEqual({
        kind: "failed",
        message: "bridge exploded",
      });
    });
  });

  it("a backend that declares no base models leaves the caller with an empty list", async () => {
    // An older server: `GET /models` still answers, just without the
    // multi-engine block. Every field is optional for exactly this reason.
    const real = createApiClient(createMockBridge({ delayMs: 0 }));
    const apiClient: ApiClient = {
      ...real,
      getModels: async () => {
        const models = await real.getModels();
        return { categories: models.categories };
      },
    };
    const { result } = renderHook(() => useBaseModels({ apiClient }));

    await waitFor(() => expect(result.current.current).toBe(""));
    expect(result.current.options).toEqual([]);
  });

});
