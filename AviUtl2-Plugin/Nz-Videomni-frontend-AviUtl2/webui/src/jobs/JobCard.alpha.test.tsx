import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ApiClient } from "../api/client";
import type { AlphaChromaRequestBody, AlphaOutputResponse, AlphaThresholds, JobResponse } from "../api/types";
import type { NativeBridge } from "../bridge";
import { LanguageProvider } from "../i18n/LanguageContext";
import { en } from "../i18n/strings";
import { getSourceLocation, recordSourceLocation, resetSourceLocationMap } from "../timeline/sourceLocationMap";
import { resetProvisionalReservation } from "../timeline/provisionalReservation";
import { JobCard } from "./JobCard";

/**
 * Alpha Gen 第 3 弾（出口）: what `JobCard` does with `AlphaControls`' view —
 * the preview URL, and 🎞 inserting what is shown (Docs/ALPHAGEN_DESIGN.md
 * §10.3): the kind / color — the transparent MOV and the fill under a fresh
 * name (`noClobber`), the matte reused (`reuseIfPresent`) — at the right-clicked
 * object's start frame on a new layer when that location is known (whatever
 * the view), at the cursor otherwise.
 */
const t = en.jobs.alpha;
const BASE = "http://127.0.0.1:18620";
const JOB_ID = "job-alpha-1";
const LAYER_MAX = 100;

function alphaJob(): JobResponse {
  return {
    job_id: JOB_ID,
    status: "completed",
    progress: 1,
    current_step: null,
    total_steps: null,
    stage: null,
    clip: null,
    clip_count: null,
    is_v2v: false,
    joined: false,
    alpha: { rgba: null, chroma: {} },
    created_at: "2026-10-15T00:00:00Z",
    started_at: "2026-10-15T00:00:01Z",
    completed_at: "2026-10-15T00:00:09Z",
    error: null,
    request: {},
    result: {
      video_url: `/api/v1/jobs/${JOB_ID}/video`,
      duration_seconds: 5,
      resolution: "1280x768",
      file_size_bytes: 1024,
      generation_time_seconds: 8,
      seed_used: 1,
      output_path: "/out/output.mp4",
      metadata_path: "/out/metadata.json",
    },
  };
}

function answer(kind: "rgba" | "chroma", thresholds: Partial<AlphaThresholds>, color: string | null): AlphaOutputResponse {
  return {
    job_id: JOB_ID,
    kind,
    color,
    transparent_threshold: thresholds.transparent_threshold ?? 7,
    opaque_threshold: thresholds.opaque_threshold ?? 248,
    path: "outputs/x",
    reused: false,
  };
}

function fakeApi() {
  const exportAlphaRgba = vi.fn(async (_id: string, body?: Partial<AlphaThresholds>) => answer("rgba", body ?? {}, null));
  const exportAlphaChroma = vi.fn(async (_id: string, body: AlphaChromaRequestBody) =>
    answer("chroma", body, body.color),
  );
  return { exportAlphaRgba, exportAlphaChroma } as unknown as ApiClient;
}

/** Records every bridge call. `downloadVideo` can be held open with `hold`. */
function stubBridge({ failEditInfo = false }: { failEditInfo?: boolean } = {}) {
  let release: () => void = () => {};
  let holding = false;
  const request = vi.fn((method: string, _params?: unknown) => {
    if (method === "backend.downloadVideo") {
      const done = Promise.resolve({ filePath: "C:/dl/x.mov", sizeBytes: 1 });
      if (!holding) return done;
      return new Promise((r) => {
        release = () => r({ filePath: "C:/dl/x.mov", sizeBytes: 1 });
      });
    }
    if (method === "getEditInfo") {
      return failEditInfo
        ? Promise.reject(new Error("NO_EDIT_HANDLE"))
        : Promise.resolve({ layerMax: LAYER_MAX, rate: 30, scale: 1 });
    }
    if (method === "timeline.insertMediaForJob")
      return Promise.resolve({ ok: true, mode: "inserted", layer: 1, frame: 0, usedFallback: false });
    if (method === "timeline.insertMedia")
      return Promise.resolve({ inserted: true, layer: LAYER_MAX + 1, frame: 120, usedFallback: false });
    return Promise.resolve({});
  });
  const bridge = { request, requestWithFiles: vi.fn(), on: vi.fn(() => () => {}) } as unknown as NativeBridge;
  return {
    bridge,
    request,
    hold: () => {
      holding = true;
    },
    release: () => release(),
  };
}

function renderCard(apiClient: ApiClient, bridge: NativeBridge) {
  return render(
    <LanguageProvider>
      <JobCard
        job={alphaJob()}
        baseUrl={BASE}
        isCancelling={false}
        isDeleting={false}
        onCancel={() => {}}
        onDelete={() => {}}
        nativeBridge={bridge}
        apiClient={apiClient}
      />
    </LanguageProvider>,
  );
}

const videoSrc = () => document.querySelector("video")?.getAttribute("src") ?? null;
const insertButton = () => screen.getByRole("button", { name: /^insert$/i });
const insertedButton = () => screen.findByRole("button", { name: /^inserted$/i });
const cut = () => screen.getByRole("button", { name: t.cut });
const bucket = () => screen.getByRole("button", { name: t.bucket });
const reset = () => screen.getByRole("button", { name: t.reset });

function calls(request: ReturnType<typeof stubBridge>["request"], method: string): Record<string, unknown>[] {
  return request.mock.calls.filter((c) => c[0] === method).map((c) => (c[1] ?? {}) as Record<string, unknown>);
}

beforeEach(() => {
  resetSourceLocationMap();
  resetProvisionalReservation();
});

describe("JobCard × AlphaControls — the preview", () => {
  it("✂ opens the checkerboard preview, 🪣 the fill, 🔁 the matte; the thresholds ride as ?v=L-R", async () => {
    const { bridge } = stubBridge();
    renderCard(fakeApi(), bridge);
    expect(videoSrc()).toBeNull(); // collapsed until something is made

    fireEvent.click(cut());
    await waitFor(() => expect(videoSrc()).toBe(`${BASE}/api/v1/jobs/${JOB_ID}/alpha/preview?v=7-248`));

    fireEvent.change(screen.getByRole("combobox", { name: t.colorLabel }), { target: { value: "blue" } });
    fireEvent.click(bucket());
    await waitFor(() => expect(videoSrc()).toBe(`${BASE}/api/v1/jobs/${JOB_ID}/alpha/chroma/blue?v=7-248`));

    fireEvent.click(reset());
    await waitFor(() => expect(videoSrc()).toBe(`${BASE}/api/v1/jobs/${JOB_ID}/video`));
  });

  it("a composite made again with other thresholds gets a new URL", async () => {
    const { bridge } = stubBridge();
    renderCard(fakeApi(), bridge);
    fireEvent.click(cut());
    await waitFor(() => expect(videoSrc()).toContain("?v=7-248"));
    const field = screen.getByRole("spinbutton", { name: new RegExp(t.opaqueThreshold) });
    fireEvent.change(field, { target: { value: "200" } });
    fireEvent.blur(field);
    // Only the thresholds changed: the preview still shows what was made.
    expect(videoSrc()).toContain("?v=7-248");
    fireEvent.click(cut());
    await waitFor(() => expect(videoSrc()).toBe(`${BASE}/api/v1/jobs/${JOB_ID}/alpha/preview?v=7-200`));
  });
});

describe("JobCard × AlphaControls — 🎞", () => {
  it("rgba view, no recorded location: downloads kind rgba under a fresh name and inserts at the cursor", async () => {
    const { bridge, request } = stubBridge();
    renderCard(fakeApi(), bridge);
    fireEvent.click(cut());
    await waitFor(() => expect(videoSrc()).toContain("/alpha/preview"));
    fireEvent.click(insertButton());
    await insertedButton();
    expect(calls(request, "backend.downloadVideo")).toEqual([{ jobId: JOB_ID, kind: "rgba", noClobber: true }]);
    expect(calls(request, "timeline.insertMediaForJob")).toHaveLength(1);
    expect(calls(request, "timeline.insertMedia")).toHaveLength(0);
  });

  it("chroma view: kind chroma with the shown color", async () => {
    const { bridge, request } = stubBridge();
    renderCard(fakeApi(), bridge);
    fireEvent.click(bucket());
    await waitFor(() => expect(videoSrc()).toContain("/alpha/chroma/green"));
    // Changing the dropdown alone does not change what is shown, nor what 🎞 places.
    fireEvent.change(screen.getByRole("combobox", { name: t.colorLabel }), { target: { value: "red" } });
    fireEvent.click(insertButton());
    await insertedButton();
    expect(calls(request, "backend.downloadVideo")).toEqual([
      { jobId: JOB_ID, kind: "chroma", color: "green", noClobber: true },
    ]);
  });

  it("right-click job: at the source object's start frame on layerMax+1, on every view, and the location is kept", async () => {
    recordSourceLocation(JOB_ID, { layer: 3, frameStart: 120, frameEnd: 419, filePath: "C:\\src.mp4", rate: 30, scale: 1 });
    const { bridge, request } = stubBridge();
    renderCard(fakeApi(), bridge);

    // The matte first (the plain view).
    fireEvent.click(insertButton());
    await insertedButton();
    // The matte never changes: reused, not a fresh name.
    expect(calls(request, "backend.downloadVideo")).toEqual([{ jobId: JOB_ID, reuseIfPresent: true }]);
    expect(calls(request, "timeline.insertMedia")).toEqual([
      { filePath: "C:/dl/x.mov", layer: LAYER_MAX + 1, frame: 120 },
    ]);
    expect(calls(request, "timeline.insertMediaForJob")).toHaveLength(0);
    expect(getSourceLocation(JOB_ID)).not.toBeNull();

    // Then the transparent MOV: same spot, location still kept.
    fireEvent.click(cut());
    await waitFor(() => expect(insertButton()).toBeEnabled());
    fireEvent.click(insertButton());
    await insertedButton();
    expect(calls(request, "timeline.insertMedia")[1]).toEqual({
      filePath: "C:/dl/x.mov",
      layer: LAYER_MAX + 1,
      frame: 120,
    });
    expect(calls(request, "backend.downloadVideo")[1]).toEqual({ jobId: JOB_ID, kind: "rgba", noClobber: true });
    expect(getSourceLocation(JOB_ID)).not.toBeNull();
    expect(request.mock.calls.some((c) => c[0] === "timeline.deleteProvisionalByJob")).toBe(false);
  });

  it("right-click job whose getEditInfo fails: falls back to the cursor insert (insertMediaForJob)", async () => {
    recordSourceLocation(JOB_ID, { layer: 3, frameStart: 120, frameEnd: 419, filePath: "C:\\src.mp4", rate: 30, scale: 1 });
    const { bridge, request } = stubBridge({ failEditInfo: true });
    renderCard(fakeApi(), bridge);
    fireEvent.click(cut());
    await waitFor(() => expect(videoSrc()).toContain("/alpha/preview"));
    fireEvent.click(insertButton());
    await insertedButton();
    expect(calls(request, "timeline.insertMediaForJob")).toEqual([{ jobId: JOB_ID, filePath: "C:/dl/x.mov" }]);
    expect(calls(request, "timeline.insertMedia")).toHaveLength(0);
  });

  it("re-inserting the same view downloads again (no remembered path for an Alpha Gen job)", async () => {
    const { bridge, request } = stubBridge();
    renderCard(fakeApi(), bridge);
    fireEvent.click(insertButton());
    await insertedButton();
    fireEvent.click(screen.getByRole("button", { name: /^inserted$/i })); // ✅ -> 🎞
    fireEvent.click(insertButton());
    await insertedButton();
    expect(calls(request, "backend.downloadVideo")).toHaveLength(2);
  });

  it("switching the view puts ✅ back to 🎞", async () => {
    const { bridge } = stubBridge();
    renderCard(fakeApi(), bridge);
    fireEvent.click(insertButton());
    await insertedButton();
    fireEvent.click(cut());
    await waitFor(() => expect(insertButton()).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: /^inserted$/i })).not.toBeInTheDocument();
  });

  it("while 🎞 runs the view cannot change (✂ 🪣 💾 🔁 disabled)", async () => {
    const stub = stubBridge();
    stub.hold();
    renderCard(fakeApi(), stub.bridge);
    fireEvent.click(insertButton());
    await waitFor(() => expect(cut()).toBeDisabled());
    for (const name of [t.bucket, t.download, t.reset]) expect(screen.getByRole("button", { name })).toBeDisabled();
    await act(async () => {
      stub.release();
    });
    await insertedButton();
    expect(cut()).toBeEnabled();
  });

  it("while ✂ is making, 🎞 is disabled", async () => {
    let resolve: (value: AlphaOutputResponse) => void = () => {};
    const apiClient = {
      exportAlphaRgba: vi.fn(
        () =>
          new Promise<AlphaOutputResponse>((r) => {
            resolve = r;
          }),
      ),
      exportAlphaChroma: vi.fn(),
    } as unknown as ApiClient;
    const { bridge } = stubBridge();
    renderCard(apiClient, bridge);
    fireEvent.click(cut());
    await waitFor(() => expect(insertButton()).toBeDisabled());
    await act(async () => {
      resolve(answer("rgba", {}, null));
    });
    expect(insertButton()).toBeEnabled();
  });
});
