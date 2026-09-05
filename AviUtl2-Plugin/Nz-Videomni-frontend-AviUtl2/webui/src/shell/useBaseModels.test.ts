import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { BackendApiError, createApiClient } from "../api/client";
import type { ApiClient } from "../api/client";
import { createMockBridge } from "../bridge/mockBridge";
import { withExtraUnsupportedFeatures } from "../test/unsupportedFeatures";
// The four pure feature→UI mappers moved out to `shell/featureScope.ts` (§3-135)
// and are tested there; this file keeps only the hook. `editSubTabsDisabledFor`
// is still imported because two of the hook's own tests read a switch's effect
// through it.
import { editSubTabsDisabledFor } from "./featureScope";
import { baseModelInstaller, useBaseModels } from "./useBaseModels";
import type { BaseModelSwitchOutcome } from "./useBaseModels";
import type { TrackPipelineLoad } from "../jobs/JobsContext";

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

/** A pass-through `TrackPipelineLoad` with a plain counter beside it —
 * `vi.fn` cannot carry the generic signature, and the tests only ever ask how
 * many times it ran. */
function countingTrackLoad(): { spy: ReturnType<typeof vi.fn>; trackLoad: TrackPipelineLoad } {
  const spy = vi.fn();
  return {
    spy,
    trackLoad: (promise) => {
      spy();
      return promise;
    },
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

  it("trackLoad wraps exactly one request per switch, and none for a local refusal", async () => {
    const apiClient = createApiClient(
      createMockBridge({ delayMs: 0, ltx25Install: "full", supportedBaseModels: ["LTX23", "LTX25"] }),
    );
    const { spy: tracked, trackLoad } = countingTrackLoad();
    const view = renderHook(() => useBaseModels({ apiClient, trackLoad }));
    await waitFor(() => expect(view.result.current.options.length).toBeGreaterThan(0));

    // The `GET /models` on mount is NOT a pipeline load.
    expect(tracked).not.toHaveBeenCalled();

    await act(async () => {
      await view.result.current.switchBaseModel("LTX25");
    });
    expect(tracked).toHaveBeenCalledTimes(1);
  });

  it("trackLoad is not called for a base model with nothing on disk (no request is made)", async () => {
    const apiClient = createApiClient(createMockBridge({ delayMs: 0, ltx25Install: "none" }));
    const { spy: tracked, trackLoad } = countingTrackLoad();
    const view = renderHook(() => useBaseModels({ apiClient, trackLoad }));
    await waitFor(() => expect(view.result.current.options.length).toBeGreaterThan(0));

    await act(async () => {
      await view.result.current.switchBaseModel("LTX25");
    });
    expect(tracked).not.toHaveBeenCalled();
  });

  it("names the installer batch file after the descriptor id", () => {
    expect(baseModelInstaller("LTX25")).toBe("install-LTX25.bat");
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

  // ── §3-98 P5: unsupported_features ───────────────────────────────────────

  it("carries each base model's unsupported_features and reports the ACTIVE one's", async () => {
    const { result } = await renderReady(createApiClient(createMockBridge({ delayMs: 0 })));

    // 台帳 §3-114 (2026-09-03): LTX 2.3 declares exactly ONE name now, where it
    // declared none for this feature's whole life. `keep_resident_embeddings`
    // belongs to a component only LTX 2.5 has, so for the first time it is the
    // OLDER engine that publishes a refusal — and it greys/hides nothing but a
    // Settings row, which is why the ordinary case is still untouched here.
    expect(result.current.options[0]?.unsupportedFeatures).toEqual(["keep_resident_embeddings"]);
    // §3-102: `chain` is no longer among them — LTX 2.5 chains now — and its
    // second stage took `v2v` and `a2v` with it. The End-source increment took
    // the LAST chain-family name, `end_source`, and the Outpainting increment
    // took `outpaint` — the last MODE name of any kind. What LTX 2.5 declares
    // now is engine-level features only, and NONE of them greys a tab or a
    // sub-tab, which is why the next test drives a synthetic name instead.
    expect(result.current.options[1]?.unsupportedFeatures).not.toContain("chain");
    expect(result.current.options[1]?.unsupportedFeatures).not.toContain("v2v");
    expect(result.current.options[1]?.unsupportedFeatures).not.toContain("a2v");
    expect(result.current.options[1]?.unsupportedFeatures).not.toContain("end_source");
    expect(result.current.options[1]?.unsupportedFeatures).not.toContain("outpaint");
    // …and one POSITIVE assertion, so this stays a test that the list travels
    // at all rather than a list of things that are absent from an empty array.
    expect(result.current.options[1]?.unsupportedFeatures).toContain("two_stage_hq");
    // LTX 2.3 is what is loaded, so nothing is disabled — its one declared name
    // (§3-114) hides a Settings row, not a mode.
    expect(result.current.unsupportedFeatures).toEqual(["keep_resident_embeddings"]);
    expect(result.current.disabledModes).toEqual([]);
  });

  // ── 2026-08-31: activeEngineFamily (comfort-budget table key) ──────────────

  it("activeEngineFamily reflects the ACTIVE base model, and stays on the OLD family while a switch is in flight", async () => {
    // Mirrors "carries each base model's unsupported_features and reports the
    // ACTIVE one's" above, for the sibling field threaded to
    // `shell/comfortTable.ts`'s `resolveComfortRow`.
    const real = createApiClient(
      createMockBridge({ delayMs: 0, ltx25Install: "full", supportedBaseModels: ["LTX23", "LTX25"] }),
    );
    let releaseLoad: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      releaseLoad = resolve;
    });
    const apiClient: ApiClient = {
      ...real,
      loadPipeline: async (models, baseModel) => {
        await gate;
        return real.loadPipeline(models, baseModel);
      },
    };
    const { result } = await renderReady(apiClient);

    expect(result.current.options[0]?.engineFamily).toBe("ltx");
    expect(result.current.options[1]?.engineFamily).toBe("ltx25");
    // LTX 2.3 (the loaded/default model) is what is ACTIVE.
    expect(result.current.activeEngineFamily).toBe("ltx");

    let outcome: Promise<BaseModelSwitchOutcome>;
    act(() => {
      outcome = result.current.switchBaseModel("LTX25");
    });

    // `current` (the DISPLAYED selection) moves optimistically to `pending`...
    await waitFor(() => expect(result.current.current).toBe("LTX25"));
    // ...but `activeEngineFamily` (read off `active`, not `current` — see its
    // own doc comment) must NOT: the pipeline has not actually loaded yet, so
    // moving the comfort marker's line early would advertise a ceiling that
    // does not apply.
    expect(result.current.activeEngineFamily).toBe("ltx");

    releaseLoad();
    await act(async () => {
      await outcome;
    });

    // The switch completed — `active` (and therefore the family) followed.
    expect(result.current.activeEngineFamily).toBe("ltx25");
  });

  it("switching to a restricted base model moves the restrictions with it", async () => {
    // THE RESTRICTION IS SYNTHETIC (`withExtraUnsupportedFeatures`), and since
    // the Outpainting increment it has to be: LTX 2.5's real list no longer
    // greys anything at all — no tab, and no Edit sub-tab — so a round trip
    // driven by it would compare "nothing disabled" with "nothing disabled"
    // and pass however broken this hook was. What is under test here is the
    // MOVEMENT of a restriction across a switch, not today's feature list;
    // that list is the test above's job, and `bridge/mockBridge.test.ts`'s.
    //
    // `retake` is the name added for the same reason `AppShell.featureScope`
    // adds it: it greys exactly ONE Edit sub-tab, so the assertions below can
    // tell "moved" from "greyed everything".
    const apiClient = createApiClient(
      withExtraUnsupportedFeatures(
        createMockBridge({
          delayMs: 0,
          ltx25Install: "full",
          supportedBaseModels: ["LTX23", "LTX25"],
        }),
        "LTX25",
        ["retake"],
      ),
    );
    const { result } = await renderReady(apiClient);
    expect(result.current.disabledModes).toEqual([]);
    // §3-114: LTX 2.3's own one name is the starting list here. What matters
    // for this test is that it is not `retake` — the switch below is what has
    // to move `retake` in, and 2.3's `keep_resident_embeddings` out.
    expect(result.current.unsupportedFeatures).toEqual(["keep_resident_embeddings"]);

    await act(async () => {
      await result.current.switchBaseModel("LTX25");
    });

    expect(result.current.unsupportedFeatures).toContain("retake");
    expect(result.current.unsupportedFeatures).not.toContain("keep_resident_embeddings");
    // The engine's OWN list travelled too, and what it no longer contains is
    // the point of this increment: `outpaint` left it, so the 画角拡張 sub-tab
    // below is expected FALSE where it used to be true.
    expect(result.current.unsupportedFeatures).not.toContain("end_source");
    expect(result.current.unsupportedFeatures).not.toContain("outpaint");
    // §3-102 took `chain` off the list and the Chained tab came back; the
    // Retake increment took `retake` off it and the Edit tab came back too —
    // Edit hosts Retake AND Outpainting, so one of the two running is enough
    // to keep the tab. NO WHOLE TAB is greyed for this engine any more, which
    // is why the restriction has to be read one level down.
    expect(result.current.disabledModes).toEqual([]);
    expect(editSubTabsDisabledFor(result.current.unsupportedFeatures)).toEqual({
      retake: true,
      outpainting: false,
    });

    // …and back. A restriction that never lifts is not a restriction, it is a
    // broken build.
    await act(async () => {
      await result.current.switchBaseModel("LTX23");
    });
    expect(result.current.disabledModes).toEqual([]);
    expect(editSubTabsDisabledFor(result.current.unsupportedFeatures)).toEqual({
      retake: false,
      outpainting: false,
    });
  });

  it("a REFUSED switch leaves the disabled modes where they were", async () => {
    // The 422 path (fixture default: the engine cannot run LTX25). Guard 4
    // reverts the displayed selection; the feature scope must revert with it,
    // or the user is left with tabs greyed for a base model that never loaded.
    const apiClient = createApiClient(createMockBridge({ delayMs: 0, ltx25Install: "full" }));
    const { result } = await renderReady(apiClient);

    await act(async () => {
      await expect(result.current.switchBaseModel("LTX25")).resolves.toMatchObject({ kind: "rejected" });
    });

    expect(result.current.current).toBe("LTX23");
    expect(result.current.disabledModes).toEqual([]);
  });

  it("a backend that omits unsupported_features means NO restrictions", async () => {
    const real = createApiClient(createMockBridge({ delayMs: 0 }));
    const apiClient: ApiClient = {
      ...real,
      getModels: async () => {
        const models = await real.getModels();
        return {
          ...models,
          base_models: (models.base_models ?? []).map(({ unsupported_features: _drop, ...rest }) => rest),
        };
      },
    };
    const { result } = await renderReady(apiClient);

    expect(result.current.options.every((o) => o.unsupportedFeatures.length === 0)).toBe(true);
    expect(result.current.disabledModes).toEqual([]);
  });
});
