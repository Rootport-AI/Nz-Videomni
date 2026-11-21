import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import type { ApiClient } from "../../api/client";
import type { GenerateChainRequest, TransformerRole } from "../../api/types";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import type { MockFsFileEntry } from "../../bridge/mockBridge";
import type { BridgeMethod, NativeBridge, ParamsOf, ResultOf } from "../../bridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { RUN_LOCK_OWNER_BATCH_A2V, __resetRunLockForTests, acquireRunLock } from "../../shell/runLock";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import { BatchV2vRefLongSection } from "./BatchV2vRefLongSection";
import { RULES } from "./fakeForm.testSupport";
import { __resetV2vRefLongRuntimeForTests } from "./runtime";
import { __resetV2vRefLongStoreForTests } from "./store";
import type { V2vRefLongChainSource } from "./types";

const t = en.batchV2vRefLong;
const VIDS = "C:\\work\\vids";
const PICS = "C:\\work\\pics";

function entry(name: string, durationSec?: number): MockFsFileEntry {
  return { name, sizeBytes: 1000, mtimeMs: 0, ...(durationSec !== undefined ? { durationSec } : {}) };
}

function makeChain(clips = 2): V2vRefLongChainSource {
  const list = Array.from({ length: clips }, () => ({ prompt: "", numFrames: 49 }));
  return {
    buildRequest: (): GenerateChainRequest => ({
      prompt: "a cat",
      width: 768,
      height: 512,
      frame_rate: 24,
      seed: -1,
      overlap_frames: 22,
      overlap_strength: 1,
      clips: list.map((c) => ({ num_frames: c.numFrames })),
    }),
    clips: list,
    minFramesForOverlap: 39,
    fixedOverlapFrames: 22,
    common: { frameRate: 24 },
    hasSourceAudio: false,
    validityReasons: [],
  };
}

const API_CLIENT = { getLoras: async () => ({ loras: [] }) } as unknown as ApiClient;

interface Call {
  method: string;
  params: Record<string, unknown>;
}

/** 2 opening videos, 3 pictures; H3 loaded (the fixture takes references only then). */
async function makeBridge(): Promise<{ bridge: NativeBridge; calls: Call[] }> {
  const base = createMockBridge({
    delayMs: 0,
    runningPollCount: 1,
    fs: createMockFs({
      folders: {
        [VIDS]: [entry("b.mp4", 5), entry("a.mp4", 5)],
        [PICS]: [entry("1.png"), entry("2.png"), entry("3.png")],
      },
    }),
  });
  await base.request("backend.request", { method: "POST", path: "/api/v1/pipeline/load", body: { base_model: "MiniMaxH3" } });
  const calls: Call[] = [];
  const bridge: NativeBridge = {
    request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
      calls.push({ method, params: params as unknown as Record<string, unknown> });
      return base.request(method, params);
    },
    requestWithFiles: (method, params, files) => base.requestWithFiles(method, params, files),
    on: (event, handler) => base.on(event, handler),
  };
  return { bridge, calls };
}

interface Opts {
  chain?: V2vRefLongChainSource;
  role?: TransformerRole | null;
}

function section(bridge: NativeBridge, opts: Opts = {}) {
  return (
    <LanguageProvider>
      <BatchV2vRefLongSection
        config={FALLBACK_APP_CONFIG}
        chain={opts.chain ?? makeChain()}
        rules={RULES}
        role={opts.role === undefined ? "ref2va" : opts.role}
        serverBusy={false}
        nativeBridge={bridge}
        apiClient={API_CLIENT}
        pollIntervalMs={2}
        jobBusyBackoffMs={2}
      />
    </LanguageProvider>
  );
}

function details(container: HTMLElement): HTMLDetailsElement {
  return container.querySelector("details.batch-section") as HTMLDetailsElement;
}

/** jsdom does not toggle on a summary click: write `open` and send the event. */
function setOpen(container: HTMLElement, open: boolean) {
  const el = details(container);
  act(() => {
    el.open = open;
    fireEvent(el, new Event("toggle"));
  });
}

function typePath(name: string, value: string, scope: HTMLElement = document.body) {
  const input = within(scope).getByRole("textbox", { name });
  fireEvent.change(input, { target: { value } });
  fireEvent.blur(input);
}

/** Opening videos and Picture 1 of clip 1 (either may be left out), then a scan. */
async function fillAndScan(opts: { videos?: boolean; pictures?: boolean } = {}) {
  if (opts.videos ?? true) typePath(t.videoDir.label, VIDS);
  if (opts.pictures ?? true) typePath("Clip 1 <Picture 1>", PICS, screen.getByTestId("v2v-frame-0"));
  fireEvent.click(screen.getByRole("button", { name: t.scanButton }));
  await waitFor(() => {
    expect(screen.getAllByTestId(/^v2v-row-\d+$/).length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: t.scanButton })).toBeEnabled();
  });
}

function chainPosts(calls: Call[]): Call[] {
  return calls.filter((c) => c.method === "backend.request" && c.params.method === "POST" && c.params.path === "/api/v1/generate/chain");
}

function joinPosts(calls: Call[]): Call[] {
  return calls.filter(
    (c) => c.method === "backend.request" && c.params.method === "POST" && /\/join$/.test(String(c.params.path)),
  );
}

function downloads(calls: Call[]): Record<string, unknown>[] {
  return calls.filter((c) => c.method === "backend.downloadVideo").map((c) => c.params);
}

function startButton(): HTMLElement {
  return screen.getByRole("button", { name: new RegExp(`^(${t.startButton}|${t.runningButton}|${t.stoppingButton})$`) });
}

beforeEach(() => {
  __resetV2vRefLongRuntimeForTests();
  __resetRunLockForTests();
  __resetV2vRefLongStoreForTests();
});

describe("BatchV2vRefLongSection", () => {
  it("closed by default with no body; opening draws it", async () => {
    const { bridge } = await makeBridge();
    const { container } = render(section(bridge));
    expect(details(container).open).toBe(false);
    expect(screen.getByText(t.heading)).toBeInTheDocument();
    expect(screen.queryByRole("textbox")).toBeNull();
    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.queryByRole("table")).toBeNull();

    setOpen(container, true);
    expect(screen.getByRole("textbox", { name: t.videoDir.label })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: t.scanButton })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: t.startButton })).toBeInTheDocument();

    setOpen(container, false);
    expect(screen.queryByRole("textbox")).toBeNull();
  });

  it("keeps the inputs, the scan and the row edits across a close/reopen and a remount", async () => {
    const { bridge } = await makeBridge();
    const first = render(section(bridge));
    setOpen(first.container, true);
    await fillAndScan();
    expect(screen.getAllByTestId(/^v2v-row-\d+$/)).toHaveLength(3);
    fireEvent.change(within(screen.getByTestId("v2v-row-1")).getByRole("textbox", { name: t.table.mainPromptInputLabel }), {
      target: { value: "kept" },
    });

    setOpen(first.container, false);
    setOpen(first.container, true);
    expect(screen.getByRole("textbox", { name: t.videoDir.label })).toHaveValue(VIDS);
    expect(screen.getAllByTestId(/^v2v-row-\d+$/)).toHaveLength(3);
    expect(within(screen.getByTestId("v2v-row-1")).getByRole("textbox", { name: t.table.mainPromptInputLabel })).toHaveValue("kept");
    first.unmount();

    const second = render(section(bridge));
    setOpen(second.container, true);
    expect(within(screen.getByTestId("v2v-frame-0")).getByRole("textbox", { name: "Clip 1 <Picture 1>" })).toHaveValue(PICS);
    expect(screen.getAllByTestId(/^v2v-row-\d+$/)).toHaveLength(3);
    expect(within(screen.getByTestId("v2v-row-1")).getByRole("textbox", { name: t.table.mainPromptInputLabel })).toHaveValue("kept");
    expect(startButton()).toBeEnabled();
  });

  it("scan → Start → every row Done; Joined rows with an opening video are joined and saved as joined", async () => {
    const { bridge, calls } = await makeBridge();
    const view = render(section(bridge));
    setOpen(view.container, true);
    await fillAndScan();

    const headers = screen.getAllByRole("columnheader").map((th) => th.textContent);
    expect(headers).toContain("<Picture 1> file name");
    const cells = (i: number) => within(screen.getByTestId(`v2v-row-${i}`)).getAllByRole("cell").map((c) => c.textContent);
    expect([0, 1, 2].map((i) => [cells(i)[2], cells(i)[3]])).toEqual([
      ["1.png", "🎬"],
      ["2.png", "🎬"],
      ["3.png", ""],
    ]);
    expect(screen.getByRole("textbox", { name: t.outDir.label })).toHaveValue("C:\\work\\pics_out");

    fireEvent.click(screen.getByRole("button", { name: t.startButton }));
    expect(startButton()).toHaveTextContent(t.runningButton);
    expect(screen.getByRole("textbox", { name: t.videoDir.label })).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: t.joined })).toBeDisabled();
    // The clip count shown stays the one frozen at Start.
    view.rerender(section(bridge, { chain: makeChain(3) }));
    expect(screen.getByText(t.table.clipCount(2))).toBeInTheDocument();

    await waitFor(() => expect(startButton()).toHaveTextContent(t.startButton), { timeout: 5_000 });
    expect(screen.getByText(t.table.clipCount(3))).toBeInTheDocument();
    for (const i of [0, 1, 2]) {
      expect(within(screen.getByTestId(`v2v-row-${i}`)).getByText(/✅/)).toBeInTheDocument();
      expect(cells(i)[6]).toBe(`${i + 1}.mp4`);
    }
    expect(chainPosts(calls)).toHaveLength(3);
    expect(joinPosts(calls)).toHaveLength(2);
    const saved = downloads(calls);
    expect(saved.filter((d) => d.kind === "joined")).toHaveLength(2);
    expect(saved.filter((d) => d.kind === undefined)).toHaveLength(1);
    expect(saved.every((d) => d.destDir === "C:\\work\\pics_out")).toBe(true);
  });

  it("Joined off: no join, every row saved as the plain output", async () => {
    const { bridge, calls } = await makeBridge();
    const view = render(section(bridge));
    setOpen(view.container, true);
    await fillAndScan();
    fireEvent.click(screen.getByRole("checkbox", { name: t.joined }));
    fireEvent.click(screen.getByRole("button", { name: t.startButton }));
    await waitFor(() => expect(startButton()).toHaveTextContent(t.startButton), { timeout: 5_000 });
    expect(screen.getAllByText(/✅/)).toHaveLength(3);
    expect(joinPosts(calls)).toHaveLength(0);
    expect(downloads(calls).map((d) => d.kind)).toEqual([undefined, undefined, undefined]);
  });

  it("another batch holding the run lock blocks Start with the lock sentence", async () => {
    const { bridge } = await makeBridge();
    const view = render(section(bridge));
    setOpen(view.container, true);
    await fillAndScan();
    expect(startButton()).toBeEnabled();
    act(() => {
      acquireRunLock(RUN_LOCK_OWNER_BATCH_A2V);
    });
    expect(startButton()).toBeDisabled();
    expect(within(screen.getByRole("note")).getByText(t.blockReasons.lockedByOther)).toBeInTheDocument();
  });

  it("FL2VA blocks Start while a row has a reference, and not without one", async () => {
    const { bridge } = await makeBridge();
    const withRefs = render(section(bridge, { role: "fl2va" }));
    setOpen(withRefs.container, true);
    await fillAndScan();
    expect(startButton()).toBeDisabled();
    expect(within(screen.getByRole("note")).getByText(t.blockReasons.fl2vaLoaded)).toBeInTheDocument();
    withRefs.unmount();

    __resetV2vRefLongStoreForTests();
    const videosOnly = render(section(bridge, { role: "fl2va" }));
    setOpen(videosOnly.container, true);
    await fillAndScan({ pictures: false });
    expect(screen.getAllByTestId(/^v2v-row-\d+$/)).toHaveLength(2);
    expect(screen.queryByText(t.blockReasons.fl2vaLoaded)).toBeNull();
    expect(startButton()).toBeEnabled();
  });
});
