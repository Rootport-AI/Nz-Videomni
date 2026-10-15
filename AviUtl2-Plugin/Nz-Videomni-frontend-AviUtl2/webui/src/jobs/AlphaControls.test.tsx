import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ReactElement } from "react";
import { BackendApiError } from "../api/client";
import type { ApiClient } from "../api/client";
import type { AlphaChromaRequestBody, AlphaOutputResponse, AlphaThresholds, JobResponse } from "../api/types";
import { LanguageProvider } from "../i18n/LanguageContext";
import { en } from "../i18n/strings";
import { AlphaControls } from "./AlphaControls";
import type { AlphaControlsProps } from "./AlphaControls";

/**
 * Alpha Gen 第 3 弾（出口）: the card's ✂🪣色💾🔁 row, the threshold slider and
 * its number fields (Docs/ALPHAGEN_DESIGN.md §10.2 / §10.3). The preview and
 * the 🎞 insert are `JobCard`'s (see `JobCard.alpha.test.tsx`).
 */
const t = en.jobs.alpha;
const BASE = "http://127.0.0.1:18620";

function alphaJob(overrides: Partial<JobResponse> = {}): JobResponse {
  return {
    job_id: "job-alpha-1",
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
      video_url: "/api/v1/jobs/job-alpha-1/video",
      duration_seconds: 5,
      resolution: "1280x768",
      file_size_bytes: 1024,
      generation_time_seconds: 8,
      seed_used: 1,
      output_path: "/out/output.mp4",
      metadata_path: "/out/metadata.json",
    },
    ...overrides,
  };
}

function answer(kind: "rgba" | "chroma", thresholds: Partial<AlphaThresholds>, color: string | null): AlphaOutputResponse {
  return {
    job_id: "job-alpha-1",
    kind,
    color,
    transparent_threshold: thresholds.transparent_threshold ?? 7,
    opaque_threshold: thresholds.opaque_threshold ?? 248,
    path: kind === "rgba" ? "outputs/job-alpha-1/rgba.mov" : `outputs/job-alpha-1/chroma_${color}.mp4`,
    reused: false,
  };
}

/** An `ApiClient` whose two exit calls echo the body back as the server would. */
function fakeApi() {
  const exportAlphaRgba = vi.fn(async (_id: string, body?: Partial<AlphaThresholds>) => answer("rgba", body ?? {}, null));
  const exportAlphaChroma = vi.fn(async (_id: string, body: AlphaChromaRequestBody) =>
    answer("chroma", body, body.color),
  );
  const apiClient = { exportAlphaRgba, exportAlphaChroma } as unknown as ApiClient;
  return { apiClient, exportAlphaRgba, exportAlphaChroma };
}

function renderControls(props: Partial<AlphaControlsProps> & { apiClient: ApiClient }) {
  const onViewChange = vi.fn();
  const onBusyChange = vi.fn();
  const ui = (p: Partial<AlphaControlsProps>): ReactElement => (
    <LanguageProvider>
      <AlphaControls
        job={alphaJob()}
        baseUrl={BASE}
        onViewChange={onViewChange}
        onBusyChange={onBusyChange}
        {...props}
        {...p}
      />
    </LanguageProvider>
  );
  const view = render(ui({}));
  return { ...view, onViewChange, onBusyChange, rerenderWith: (p: Partial<AlphaControlsProps>) => view.rerender(ui(p)) };
}

const cut = () => screen.getByRole("button", { name: t.cut });
const bucket = () => screen.getByRole("button", { name: t.bucket });
const colorSelect = () => screen.getByRole("combobox", { name: t.colorLabel });
const save = () => screen.getByRole("button", { name: t.download });
const reset = () => screen.getByRole("button", { name: t.reset });
const transparentField = () => screen.getByRole("spinbutton", { name: new RegExp(t.transparentThreshold) });
const opaqueField = () => screen.getByRole("spinbutton", { name: new RegExp(t.opaqueThreshold) });

function setField(field: HTMLElement, value: string) {
  fireEvent.change(field, { target: { value } });
  fireEvent.blur(field);
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("AlphaControls — when it shows", () => {
  it("renders nothing for a job that is not an Alpha Gen job (alpha null or absent)", () => {
    const { apiClient } = fakeApi();
    const { container, rerenderWith } = renderControls({ apiClient, job: alphaJob({ alpha: null }) });
    expect(container).toBeEmptyDOMElement();
    const { alpha: _omit, ...withoutAlpha } = alphaJob();
    rerenderWith({ job: withoutAlpha as JobResponse });
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing for a failed Alpha Gen job", () => {
    const { apiClient } = fakeApi();
    const { container } = renderControls({ apiClient, job: alphaJob({ status: "failed", result: null }) });
    expect(container).toBeEmptyDOMElement();
  });

  it("shows ✂ 🪣 color 💾 🔁, the slider and the two fields on the defaults (7 / 248, green)", () => {
    const { apiClient } = fakeApi();
    renderControls({ apiClient });
    for (const button of [cut(), bucket(), save(), reset()]) expect(button).toBeEnabled();
    expect(cut()).toHaveAttribute("title", t.cut);
    expect((colorSelect() as HTMLSelectElement).value).toBe("green");
    expect(Array.from((colorSelect() as HTMLSelectElement).options).map((o) => o.value)).toEqual([
      "white",
      "red",
      "yellow",
      "green",
      "aqua",
      "blue",
      "magenta",
      "black",
    ]);
    expect(transparentField()).toHaveValue(7);
    expect(opaqueField()).toHaveValue(248);
    expect(screen.getAllByRole("slider")).toHaveLength(2);
  });
});

describe("AlphaControls — ✂", () => {
  it("POSTs the current thresholds and switches to the transparent view with the 💾 note", async () => {
    const { apiClient, exportAlphaRgba } = fakeApi();
    const { onViewChange } = renderControls({ apiClient });
    expect(onViewChange).toHaveBeenLastCalledWith("matte", null);
    fireEvent.click(cut());
    await screen.findByText(t.downloadHint);
    expect(exportAlphaRgba).toHaveBeenCalledWith("job-alpha-1", { transparent_threshold: 7, opaque_threshold: 248 });
    expect(onViewChange).toHaveBeenLastCalledWith("rgba", {
      thresholds: { transparent_threshold: 7, opaque_threshold: 248 },
      color: null,
    });
  });

  it("does nothing on the transparent view when the thresholds are the shown ones", async () => {
    const { apiClient, exportAlphaRgba } = fakeApi();
    renderControls({ apiClient });
    fireEvent.click(cut());
    await screen.findByText(t.downloadHint);
    fireEvent.click(cut());
    await act(async () => {});
    expect(exportAlphaRgba).toHaveBeenCalledTimes(1);
  });

  it("makes it again when the thresholds changed (the banner shows, then clears)", async () => {
    const { apiClient, exportAlphaRgba } = fakeApi();
    const { onViewChange } = renderControls({ apiClient });
    fireEvent.click(cut());
    await screen.findByText(t.downloadHint);
    setField(transparentField(), "20");
    expect(screen.getByText(t.thresholdsChanged)).toBeInTheDocument();
    fireEvent.click(cut());
    await waitFor(() => expect(screen.queryByText(t.thresholdsChanged)).not.toBeInTheDocument());
    expect(exportAlphaRgba).toHaveBeenLastCalledWith("job-alpha-1", { transparent_threshold: 20, opaque_threshold: 248 });
    expect(onViewChange).toHaveBeenLastCalledWith("rgba", {
      thresholds: { transparent_threshold: 20, opaque_threshold: 248 },
      color: null,
    });
  });
});

describe("AlphaControls — the threshold banner", () => {
  it("shows only on a composite view, and moving the values back clears it (no flag)", async () => {
    const { apiClient } = fakeApi();
    renderControls({ apiClient });
    // On the matte the thresholds apply to nothing shown: no banner.
    setField(opaqueField(), "200");
    expect(screen.queryByText(t.thresholdsChanged)).not.toBeInTheDocument();
    fireEvent.click(cut());
    await screen.findByText(t.downloadHint);
    setField(opaqueField(), "210");
    expect(screen.getByText(t.thresholdsChanged)).toBeInTheDocument();
    setField(opaqueField(), "200");
    expect(screen.queryByText(t.thresholdsChanged)).not.toBeInTheDocument();
    // The slider moves the same values.
    fireEvent.change(screen.getByRole("slider", { name: t.transparentThreshold }), { target: { value: "30" } });
    expect(screen.getByText(t.thresholdsChanged)).toBeInTheDocument();
    expect(transparentField()).toHaveValue(30);
  });
});

describe("AlphaControls — 🪣 and the color", () => {
  it("POSTs the chosen color and shows the fill (no 💾 note)", async () => {
    const { apiClient, exportAlphaChroma } = fakeApi();
    const { onViewChange } = renderControls({ apiClient });
    fireEvent.click(bucket());
    await waitFor(() =>
      expect(onViewChange).toHaveBeenLastCalledWith("chroma", {
        thresholds: { transparent_threshold: 7, opaque_threshold: 248 },
        color: "green",
      }),
    );
    expect(exportAlphaChroma).toHaveBeenCalledWith("job-alpha-1", {
      color: "green",
      transparent_threshold: 7,
      opaque_threshold: 248,
    });
    expect(screen.queryByText(t.downloadHint)).not.toBeInTheDocument();
  });

  it("same color and thresholds: nothing; a new color alone: nothing until 🪣; same color, new thresholds: made again", async () => {
    const { apiClient, exportAlphaChroma } = fakeApi();
    const { onViewChange } = renderControls({ apiClient });
    fireEvent.click(bucket());
    await waitFor(() => expect(onViewChange).toHaveBeenLastCalledWith("chroma", expect.anything()));
    fireEvent.click(bucket());
    await act(async () => {});
    expect(exportAlphaChroma).toHaveBeenCalledTimes(1);

    fireEvent.change(colorSelect(), { target: { value: "blue" } });
    await act(async () => {});
    expect(exportAlphaChroma).toHaveBeenCalledTimes(1);
    fireEvent.click(bucket());
    await waitFor(() => expect(exportAlphaChroma).toHaveBeenCalledTimes(2));
    expect(exportAlphaChroma).toHaveBeenLastCalledWith("job-alpha-1", expect.objectContaining({ color: "blue" }));

    setField(transparentField(), "0");
    setField(opaqueField(), "255");
    fireEvent.click(bucket());
    await waitFor(() => expect(exportAlphaChroma).toHaveBeenCalledTimes(3));
    expect(exportAlphaChroma).toHaveBeenLastCalledWith("job-alpha-1", {
      color: "blue",
      transparent_threshold: 0,
      opaque_threshold: 255,
    });
  });

  it("from the transparent view 🪣 switches to the fill", async () => {
    const { apiClient, exportAlphaChroma } = fakeApi();
    const { onViewChange } = renderControls({ apiClient });
    fireEvent.click(cut());
    await screen.findByText(t.downloadHint);
    fireEvent.click(bucket());
    await waitFor(() => expect(onViewChange).toHaveBeenLastCalledWith("chroma", expect.anything()));
    expect(exportAlphaChroma).toHaveBeenCalledTimes(1);
  });
});

describe("AlphaControls — 💾", () => {
  it("from the matte, with no MOV yet: POST, then an <a> click on the MOV's URL, then the transparent view", async () => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    const { apiClient, exportAlphaRgba } = fakeApi();
    const { onViewChange } = renderControls({ apiClient });
    fireEvent.click(save());
    await screen.findByText(t.downloadHint);
    expect(exportAlphaRgba).toHaveBeenCalledTimes(1);
    expect(click).toHaveBeenCalledTimes(1);
    const anchor = click.mock.contexts[0] as HTMLAnchorElement;
    expect(anchor.href).toBe(`${BASE}/api/v1/jobs/job-alpha-1/alpha/rgba?v=7-248`);
    // `download` would do nothing on another origin; the server's header does it.
    expect(anchor.hasAttribute("download")).toBe(false);
    // The temporary anchor does not stay in the page.
    expect(document.querySelector(`a[href*="/alpha/rgba"]`)).toBeNull();
    expect(onViewChange).toHaveBeenLastCalledWith("rgba", expect.objectContaining({ color: null }));
  });

  it("even when job.alpha says the MOV exists, 💾 POSTs first (the server decides) and the URL uses the answer's thresholds", async () => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    // The server reuses the file it has: `reused: true`, made with 7 / 248.
    const exportAlphaRgba = vi.fn(async () => ({ ...answer("rgba", {}, null), reused: true }));
    const apiClient = { exportAlphaRgba, exportAlphaChroma: vi.fn() } as unknown as ApiClient;
    const { onViewChange } = renderControls({
      apiClient,
      job: alphaJob({ alpha: { rgba: { transparent_threshold: 7, opaque_threshold: 248 }, chroma: {} } }),
    });
    fireEvent.click(save());
    expect(click).not.toHaveBeenCalled(); // not before the answer
    await waitFor(() => expect(click).toHaveBeenCalledTimes(1));
    expect(exportAlphaRgba).toHaveBeenCalledTimes(1);
    expect((click.mock.contexts[0] as HTMLAnchorElement).href).toBe(
      `${BASE}/api/v1/jobs/job-alpha-1/alpha/rgba?v=7-248`,
    );
    await waitFor(() =>
      expect(onViewChange).toHaveBeenLastCalledWith("rgba", {
        thresholds: { transparent_threshold: 7, opaque_threshold: 248 },
        color: null,
      }),
    );
  });

  it("on the transparent view with the same thresholds, every 💾 still POSTs before it downloads", async () => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    const { apiClient, exportAlphaRgba } = fakeApi();
    renderControls({ apiClient });
    fireEvent.click(cut());
    await screen.findByText(t.downloadHint);
    fireEvent.click(save());
    await waitFor(() => expect(click).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(save()).toBeEnabled());
    fireEvent.click(save());
    await waitFor(() => expect(click).toHaveBeenCalledTimes(2));
    expect(exportAlphaRgba).toHaveBeenCalledTimes(3);
  });

  it("job.alpha with other thresholds: makes the MOV first", async () => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    const { apiClient, exportAlphaRgba } = fakeApi();
    renderControls({
      apiClient,
      job: alphaJob({ alpha: { rgba: { transparent_threshold: 0, opaque_threshold: 255 }, chroma: {} } }),
    });
    fireEvent.click(save());
    await waitFor(() => expect(click).toHaveBeenCalledTimes(1));
    expect(exportAlphaRgba).toHaveBeenCalledWith("job-alpha-1", { transparent_threshold: 7, opaque_threshold: 248 });
  });
});

describe("AlphaControls — 🔁, busy and failure", () => {
  it("🔁 goes back to the matte", async () => {
    const { apiClient } = fakeApi();
    const { onViewChange } = renderControls({ apiClient });
    fireEvent.click(cut());
    await screen.findByText(t.downloadHint);
    fireEvent.click(reset());
    expect(screen.queryByText(t.downloadHint)).not.toBeInTheDocument();
    expect(onViewChange).toHaveBeenLastCalledWith("matte", null);
  });

  it("while making, the five controls are disabled and 作成中 shows; the card hears busy true then false", async () => {
    let resolve: (value: AlphaOutputResponse) => void = () => {};
    const exportAlphaRgba = vi.fn(
      () =>
        new Promise<AlphaOutputResponse>((r) => {
          resolve = r;
        }),
    );
    const apiClient = { exportAlphaRgba, exportAlphaChroma: vi.fn() } as unknown as ApiClient;
    const { onBusyChange } = renderControls({ apiClient });
    fireEvent.click(cut());
    expect(await screen.findByText(t.busy)).toBeInTheDocument();
    for (const control of [cut(), bucket(), colorSelect(), save(), reset()]) expect(control).toBeDisabled();
    expect(onBusyChange).toHaveBeenLastCalledWith(true);
    await act(async () => {
      resolve(answer("rgba", {}, null));
    });
    expect(screen.queryByText(t.busy)).not.toBeInTheDocument();
    for (const control of [cut(), bucket(), colorSelect(), save(), reset()]) expect(control).toBeEnabled();
    expect(onBusyChange).toHaveBeenLastCalledWith(false);
  });

  it("a failure shows one red line (code: message) and leaves the view as it was", async () => {
    const exportAlphaRgba = vi.fn(async () => {
      throw new BackendApiError("ALPHA_OUTPUT_FAILED", "Alpha Gen output failed", 503);
    });
    const apiClient = { exportAlphaRgba, exportAlphaChroma: vi.fn() } as unknown as ApiClient;
    const { onViewChange } = renderControls({ apiClient });
    fireEvent.click(cut());
    const line = await screen.findByText("ALPHA_OUTPUT_FAILED: Alpha Gen output failed");
    expect(line).toHaveClass("field-hint-error");
    expect(screen.queryByText(t.downloadHint)).not.toBeInTheDocument();
    expect(onViewChange).toHaveBeenLastCalledWith("matte", null);
    expect(cut()).toBeEnabled();
  });

  it("while 🎞 runs (insertBusy), ✂ 🪣 💾 🔁 are disabled; the color stays free", () => {
    const { apiClient } = fakeApi();
    renderControls({ apiClient, insertBusy: true });
    for (const button of [cut(), bucket(), save(), reset()]) expect(button).toBeDisabled();
    expect(colorSelect()).toBeEnabled();
  });
});

describe("AlphaControls — the number fields", () => {
  it("transparent is held to 0..opaque, opaque to transparent..255, rounded on blur; equal is allowed", () => {
    const { apiClient } = fakeApi();
    renderControls({ apiClient });
    setField(transparentField(), "300");
    expect(transparentField()).toHaveValue(248);
    setField(transparentField(), "-5");
    expect(transparentField()).toHaveValue(0);
    setField(transparentField(), "12.6");
    expect(transparentField()).toHaveValue(13);
    setField(opaqueField(), "999");
    expect(opaqueField()).toHaveValue(255);
    setField(opaqueField(), "1");
    expect(opaqueField()).toHaveValue(13);
    expect(screen.getByRole("slider", { name: t.opaqueThreshold })).toHaveValue("13");
  });

  it("text that is not a number goes back to the current value", () => {
    const { apiClient } = fakeApi();
    renderControls({ apiClient });
    setField(transparentField(), "");
    expect(transparentField()).toHaveValue(7);
  });

  it("typing alone does not move the slider until blur", () => {
    const { apiClient } = fakeApi();
    renderControls({ apiClient });
    fireEvent.change(transparentField(), { target: { value: "50" } });
    expect(screen.getByRole("slider", { name: t.transparentThreshold })).toHaveValue("7");
    fireEvent.blur(transparentField());
    expect(screen.getByRole("slider", { name: t.transparentThreshold })).toHaveValue("50");
  });
});
