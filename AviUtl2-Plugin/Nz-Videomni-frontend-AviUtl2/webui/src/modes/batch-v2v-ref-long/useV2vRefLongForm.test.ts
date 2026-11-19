import { act, renderHook, waitFor } from "@testing-library/react";
import { createElement } from "react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it } from "vitest";
import type { ApiClient } from "../../api/client";
import type { GenerateChainRequest, LoraEntry, TransformerRole } from "../../api/types";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import type { MockBridgeOptions, MockFsFileEntry } from "../../bridge/mockBridge";
import type { BridgeMethod, NativeBridge, ParamsOf, ResultOf } from "../../bridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import {
  RUN_LOCK_OWNER_BATCH_A2V,
  __resetRunLockForTests,
  acquireRunLock,
  releaseRunLock,
} from "../../shell/runLock";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import type { ReferenceRules } from "../single/referenceRules";
import { __resetV2vRefLongRuntimeForTests } from "./runtime";
import { __resetV2vRefLongStoreForTests, v2vRefLongStore } from "./store";
import type { V2vRefLongChainSource } from "./types";
import { useV2vRefLongForm } from "./useV2vRefLongForm";
import type { UseV2vRefLongFormDeps } from "./useV2vRefLongForm";

const RULES: ReferenceRules = {
  maxImages: 9,
  maxVideos: 3,
  maxAudios: 3,
  maxTotal: 12,
  videoTotalSec: 15,
  audioTotalSec: 15,
  clipMinSec: 1,
};

const VIDS = "C:\\work\\vids";
const PICS = "C:\\work\\pics";

function entry(name: string, durationSec?: number): MockFsFileEntry {
  return { name, sizeBytes: 1000, mtimeMs: 0, ...(durationSec !== undefined ? { durationSec } : {}) };
}

/** 2 opening videos, 3 pictures. */
function folders() {
  return createMockFs({
    folders: {
      [VIDS]: [entry("b.mp4", 5), entry("a.mp4", 5)],
      [PICS]: [entry("1.png"), entry("2.png"), entry("3.png")],
    },
  });
}

interface ChainOpts {
  clips?: number;
  prompt?: string;
  hasSourceAudio?: boolean;
  loras?: { name: string; strength: number }[];
}

function makeChain(opts: ChainOpts = {}): V2vRefLongChainSource {
  const count = opts.clips ?? 2;
  const clips = Array.from({ length: count }, () => ({ prompt: "", numFrames: 49 }));
  const buildRequest = (): GenerateChainRequest => ({
    prompt: opts.prompt ?? "a cat",
    width: 768,
    height: 512,
    frame_rate: 24,
    seed: -1,
    overlap_frames: 22,
    overlap_strength: 1,
    clips: clips.map((c) => ({ num_frames: c.numFrames })),
    ...(opts.loras ? { loras: opts.loras } : {}),
  });
  return {
    buildRequest,
    clips,
    minFramesForOverlap: 39,
    fixedOverlapFrames: 22,
    common: { frameRate: 24 },
    hasSourceAudio: opts.hasSourceAudio ?? false,
    validityReasons: [],
  };
}

function lorasClient(names: string[]): ApiClient {
  const loras: LoraEntry[] = names.map((name) => ({ name, kind: "style", has_thumbnail: false, exists: true, source: "scan" }));
  return { getLoras: async () => ({ loras }) } as unknown as ApiClient;
}

interface Call {
  method: string;
  params: Record<string, unknown>;
}

async function makeBridge(options: MockBridgeOptions = {}): Promise<{ bridge: NativeBridge; calls: Call[] }> {
  const base = createMockBridge({ delayMs: 0, runningPollCount: 1, fs: folders(), ...options });
  // The fixture server takes `references` only with H3 loaded.
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

interface Props {
  chain: V2vRefLongChainSource;
  role: TransformerRole | null;
  serverBusy: boolean;
}

function wrapper({ children }: { children: ReactNode }) {
  return createElement(LanguageProvider, null, children);
}

function render(bridge: NativeBridge, initial: Partial<Props> = {}, apiClient: ApiClient = lorasClient([])) {
  const initialProps: Props = { chain: makeChain(), role: "ref2va", serverBusy: false, ...initial };
  return renderHook(
    (props: Props) => {
      const deps: UseV2vRefLongFormDeps = {
        config: FALLBACK_APP_CONFIG,
        chain: props.chain,
        rules: RULES,
        role: props.role,
        serverBusy: props.serverBusy,
        nativeBridge: bridge,
        apiClient,
        pollIntervalMs: 2,
        jobBusyBackoffMs: 2,
      };
      return useV2vRefLongForm(deps);
    },
    { initialProps, wrapper },
  );
}

type Rendered = ReturnType<typeof render>;

/** Opening videos and Picture 1 of clip 1, then a scan. */
async function scanned(r: Rendered, opts: { videos?: boolean; pictures?: boolean } = {}) {
  act(() => {
    if (opts.videos ?? true) r.result.current.setVideoDir(VIDS);
    if (opts.pictures ?? true) r.result.current.setFieldPath(0, 1, PICS);
  });
  await act(async () => {
    await r.result.current.scan();
  });
}

function chainPosts(calls: Call[]): GenerateChainRequest[] {
  return calls
    .filter((c) => c.method === "backend.request" && c.params.method === "POST" && c.params.path === "/api/v1/generate/chain")
    .map((c) => c.params.body as GenerateChainRequest);
}

beforeEach(() => {
  __resetV2vRefLongRuntimeForTests();
  __resetRunLockForTests();
  __resetV2vRefLongStoreForTests();
});

describe("useV2vRefLongForm", () => {
  it("scan builds the rows from the base folder and fills the automatic output folder", async () => {
    const { bridge } = await makeBridge();
    const r = render(bridge);
    expect(r.result.current.canScan).toBe(false);
    await scanned(r);

    const { current } = r.result;
    expect(current.state.scan).not.toBeNull();
    expect(current.baseField).toEqual({ type: "ref", frame: 0, kind: "picture", slot: 1 });
    expect(current.rowViews.map((v) => [v.baseFileName, v.hasSource])).toEqual([
      ["1.png", true],
      ["2.png", true],
      ["3.png", false],
    ]);
    expect(current.state.inputs.outDir).toBe("C:\\work\\pics_out");
    expect(current.state.scan?.source.map((f) => f.name)).toEqual(["a.mp4", "b.mp4"]);
    expect(current.summary).toEqual({ done: 0, total: 3 });
    expect(current.filledFolderCount).toBe(1);
    expect(current.canScan).toBe(true);
  });

  it("a scan failure lands in scanError", async () => {
    const { bridge } = await makeBridge();
    const failing: NativeBridge = {
      ...bridge,
      request: (method, params) =>
        method === "fs.listFiles" ? Promise.reject(new Error("disk gone")) : bridge.request(method, params),
    };
    const r = render(failing);
    act(() => r.result.current.setVideoDir(VIDS));
    await act(async () => {
      await r.result.current.scan();
    });
    expect(r.result.current.state.scanError).toBe("disk gone");
    expect(r.result.current.state.scanning).toBe(false);
  });

  it("blocks Start before a scan and after the folders change, until restored", async () => {
    const { bridge } = await makeBridge();
    const r = render(bridge);
    expect(r.result.current.blockReasons).toContain("notScanned");
    expect(r.result.current.canStart).toBe(false);

    await scanned(r);
    expect(r.result.current.blockReasons).toEqual([]);
    expect(r.result.current.canStart).toBe(true);

    act(() => r.result.current.setFieldPath(0, 1, "C:\\elsewhere"));
    expect(r.result.current.blockReasons).toContain("scanStale");
    expect(r.result.current.state.rows).toHaveLength(3);
    act(() => r.result.current.setFieldPath(0, 1, PICS));
    expect(r.result.current.blockReasons).toEqual([]);
  });

  it("FL2VA blocks only when a used reference is assigned", async () => {
    const { bridge } = await makeBridge();
    const withRefs = render(bridge, { role: "fl2va" });
    await scanned(withRefs);
    expect(withRefs.result.current.blockReasons).toContain("fl2vaLoaded");
    withRefs.unmount();

    __resetV2vRefLongStoreForTests();
    const videosOnly = render(bridge, { role: "fl2va" });
    await scanned(videosOnly, { pictures: false });
    expect(videosOnly.result.current.blockReasons).not.toContain("fl2vaLoaded");
    expect(videosOnly.result.current.canStart).toBe(true);
  });

  it("blocks on long audio, an unknown LoRA, a busy server and another batch's lock", async () => {
    const { bridge } = await makeBridge();
    const r = render(bridge, { chain: makeChain({ hasSourceAudio: true, loras: [{ name: "nope", strength: 1 }] }), serverBusy: true });
    await scanned(r);
    await waitFor(() => expect(r.result.current.blockReasons).toContain("unknownLoraTag"));
    expect(r.result.current.blockReasons).toEqual(
      expect.arrayContaining(["sourceAudioAttached", "unknownLoraTag", "jobActive"]),
    );
    expect(r.result.current.blockReasons).not.toContain("lockedByOther");

    let token: ReturnType<typeof acquireRunLock> = null;
    act(() => {
      token = acquireRunLock(RUN_LOCK_OWNER_BATCH_A2V);
    });
    expect(r.result.current.blockReasons).toContain("lockedByOther");
    act(() => {
      releaseRunLock(token);
    });
    expect(r.result.current.blockReasons).not.toContain("lockedByOther");
  });

  it("while running, the clip count stays the one frozen at Start and inputs are locked", async () => {
    const { bridge } = await makeBridge();
    const r = render(bridge);
    await scanned(r);
    expect(r.result.current.inputsLocked).toBe(false);

    act(() => r.result.current.start());
    expect(r.result.current.state.run.state).toBe("running");
    expect(r.result.current.inputsLocked).toBe(true);
    expect(r.result.current.canScan).toBe(false);
    expect(r.result.current.canStart).toBe(false);
    r.rerender({ chain: makeChain({ clips: 3 }), role: "ref2va", serverBusy: false });
    expect(r.result.current.clipCount).toBe(2);

    await waitFor(() => expect(r.result.current.state.run.state).toBe("idle"), { timeout: 5_000 });
    expect(r.result.current.clipCount).toBe(3);
    expect(r.result.current.inputsLocked).toBe(false);
    expect(r.result.current.summary).toEqual({ done: 3, total: 3 });
  });

  it("does not hand 🚫 rows to the run", async () => {
    const { bridge, calls } = await makeBridge();
    // One clip: the row without an opening video is 🚫.
    const r = render(bridge, { chain: makeChain({ clips: 1 }) });
    await scanned(r);
    expect(r.result.current.rowViews.map((v) => v.blockReasons.map((b) => b.code))).toEqual([[], [], ["noSourceSingleClip"]]);

    act(() => r.result.current.start());
    await waitFor(() => expect(r.result.current.state.run.state).toBe("idle"), { timeout: 5_000 });
    expect(chainPosts(calls)).toHaveLength(2);
    expect(r.result.current.state.rows.map((row) => row.stat)).toEqual(["Done", "Done", "Waiting"]);
    const [first] = chainPosts(calls);
    expect(first?.source_video?.context_frames).toBe(22);
    expect(first?.clips[0]?.references).toEqual([{ kind: "picture", id: expect.any(String), slot: 1 }]);
  });

  it("attachClipFile measures the file and uploads nothing", async () => {
    const { bridge, calls } = await makeBridge({ probeMediaInfoDurationSec: 7 });
    const r = render(bridge);
    await scanned(r);
    await act(async () => {
      await r.result.current.attachClipFile(0, 1, "video", 2, "C:\\hand\\x.mp4", "x.mp4");
    });
    expect(calls.some((c) => c.method === "backend.uploadFile")).toBe(false);
    const card = r.result.current.effectiveClipOf(0, 1)?.refs.slots.video[1];
    expect(card).toMatchObject({ status: "ready", filePath: "C:\\hand\\x.mp4", fileName: "x.mp4", durationSec: 7, id: null });

    act(() => r.result.current.clearClipSlot(0, 1, "video", 2));
    expect(r.result.current.effectiveClipOf(0, 1)?.refs.slots.video[1]?.status).toBe("empty");
  });

  it("an older attach that resolves after a newer one is dropped", async () => {
    const { bridge } = await makeBridge();
    let releaseFirst: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });
    const slow: NativeBridge = {
      ...bridge,
      request: async (method, params) => {
        if (method === "fs.probeMediaInfo" && (params as { filePath?: string }).filePath === "C:\\old.wav") await gate;
        return bridge.request(method, params);
      },
    };
    const r = render(slow);
    await scanned(r);
    let first: Promise<void> = Promise.resolve();
    act(() => {
      first = r.result.current.attachClipFile(1, 0, "audio", 1, "C:\\old.wav", "old.wav");
    });
    await act(async () => {
      await r.result.current.attachClipFile(1, 0, "audio", 1, "C:\\new.wav", "new.wav");
    });
    await act(async () => {
      releaseFirst();
      await first;
    });
    expect(r.result.current.effectiveClipOf(1, 0)?.refs.slots.audio[0]?.fileName).toBe("new.wav");
  });

  it("pickClipFile attaches the picked file; a cancelled dialog does nothing", async () => {
    const { bridge } = await makeBridge({ pickFileName: "picked.png" });
    const r = render(bridge);
    await scanned(r);
    await act(async () => {
      await r.result.current.pickClipFile(0, 0, "picture", 3);
    });
    expect(r.result.current.effectiveClipOf(0, 0)?.refs.slots.picture[2]?.fileName).toBe("picked.png");

    const cancelled = await makeBridge({ failPickFile: "CANCELLED" });
    const r2 = render(cancelled.bridge);
    const before = v2vRefLongStore.getSnapshot();
    await act(async () => {
      await r2.result.current.pickClipFile(0, 0, "picture", 4);
    });
    expect(v2vRefLongStore.getSnapshot()).toBe(before);
  });

  it("✒ appends the tag to the clip prompt; 📝 copies the shared prompt", async () => {
    const { bridge } = await makeBridge();
    const r = render(bridge, { chain: makeChain({ prompt: "shared prompt" }) });
    await scanned(r);
    act(() => r.result.current.setClipPrompt(0, 1, "walks"));
    act(() => r.result.current.appendClipTag(0, 1, "<Picture 1>"));
    expect(r.result.current.state.rows[0]?.clipEdits[1]?.prompt).toBe("walks <Picture 1>");

    act(() => r.result.current.copyChainPromptToRow(2));
    // The template's prompt (the Chain form has already taken the LoRA tags out).
    expect(r.result.current.state.rows[2]?.mainPrompt).toBe("shared prompt");
  });

  it("keeps everything across an unmount and a remount", async () => {
    const { bridge } = await makeBridge();
    const first = render(bridge);
    await scanned(first);
    act(() => first.result.current.setRowMainPrompt(1, "kept"));
    first.unmount();

    const second = render(bridge);
    expect(second.result.current.state.rows).toHaveLength(3);
    expect(second.result.current.state.rows[1]?.mainPrompt).toBe("kept");
    expect(second.result.current.canStart).toBe(true);
  });

  it("a scan started before an unmount still lands in the store", async () => {
    const { bridge } = await makeBridge();
    const r = render(bridge);
    act(() => r.result.current.setVideoDir(VIDS));
    let pending: Promise<void> = Promise.resolve();
    act(() => {
      pending = r.result.current.scan();
    });
    expect(v2vRefLongStore.getSnapshot().scanning).toBe(true);
    r.unmount();
    await pending;
    expect(v2vRefLongStore.getSnapshot().scanning).toBe(false);
    expect(v2vRefLongStore.getSnapshot().rows).toHaveLength(2);
  });

  it("a run started before an unmount goes on and a remount sees it", async () => {
    const { bridge } = await makeBridge();
    const first = render(bridge);
    await scanned(first);
    act(() => first.result.current.start());
    first.unmount();

    const second = render(bridge);
    expect(second.result.current.state.run.state).not.toBe("idle");
    await waitFor(() => expect(second.result.current.state.run.state).toBe("idle"), { timeout: 5_000 });
    expect(second.result.current.summary.done).toBe(3);
  });

  it("stop marks the run stopping and the current row still finishes", async () => {
    const { bridge, calls } = await makeBridge();
    const r = render(bridge);
    await scanned(r);
    act(() => r.result.current.start());
    act(() => r.result.current.stop());
    expect(r.result.current.state.run.state).toBe("stopping");
    await waitFor(() => expect(r.result.current.state.run.state).toBe("idle"), { timeout: 5_000 });
    expect(r.result.current.state.rows.map((row) => row.stat)).toEqual(["Done", "Waiting", "Waiting"]);
    expect(calls.some((c) => c.method === "backend.request" && c.params.method === "DELETE")).toBe(false);
  });

  it("a failed folder dialog shows in scanError; a cancelled one does nothing", async () => {
    const failed = await makeBridge({ failPickFolder: "DIALOG_FAILED" });
    const r = render(failed.bridge);
    await act(async () => {
      await r.result.current.pickVideoDir();
    });
    expect(r.result.current.state.scanError).toMatch(/^DIALOG_FAILED: /);
    expect(r.result.current.state.inputs.videoDir).toBe("");
    r.unmount();

    __resetV2vRefLongStoreForTests();
    const cancelled = await makeBridge({ failPickFolder: "CANCELLED" });
    const r2 = render(cancelled.bridge);
    const before = v2vRefLongStore.getSnapshot();
    await act(async () => {
      await r2.result.current.pickOutDir();
    });
    expect(v2vRefLongStore.getSnapshot()).toBe(before);
  });

  it("a failed file dialog in a card does nothing", async () => {
    const { bridge } = await makeBridge({ failPickFile: "DIALOG_FAILED" });
    const r = render(bridge);
    await scanned(r);
    const before = v2vRefLongStore.getSnapshot();
    await act(async () => {
      await r.result.current.pickClipFile(0, 0, "picture", 3);
    });
    expect(v2vRefLongStore.getSnapshot()).toBe(before);
  });

  it("a measure that ends after a re-scan does not reach the rebuilt rows", async () => {
    const { bridge } = await makeBridge();
    let release: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const slow: NativeBridge = {
      ...bridge,
      request: async (method, params) => {
        if (method === "fs.probeMediaInfo" && (params as { filePath?: string }).filePath === "C:\\late.wav") await gate;
        return bridge.request(method, params);
      },
    };
    const r = render(slow);
    await scanned(r);
    let pending: Promise<void> = Promise.resolve();
    act(() => {
      pending = r.result.current.attachClipFile(1, 0, "audio", 1, "C:\\late.wav", "late.wav");
    });
    await act(async () => {
      await r.result.current.scan();
    });
    await act(async () => {
      release();
      await pending;
    });
    expect(r.result.current.state.rows[1]?.clipEdits).toEqual({});
    expect(r.result.current.effectiveClipOf(1, 0)?.refs.slots.audio[0]?.status).toBe("empty");
  });

  it("changing the shared prompt and the role mid-run does not change the rows still to send", async () => {
    const { bridge, calls } = await makeBridge();
    const r = render(bridge, { chain: makeChain({ prompt: "start prompt" }) });
    await scanned(r);
    act(() => r.result.current.start());
    r.rerender({ chain: makeChain({ prompt: "changed prompt" }), role: "fl2va", serverBusy: false });
    await waitFor(() => expect(r.result.current.state.run.state).toBe("idle"), { timeout: 5_000 });

    const posts = chainPosts(calls);
    expect(posts).toHaveLength(3);
    const [first, ...rest] = posts;
    for (const body of rest) {
      expect(body.prompt).toBe("start prompt");
      expect(body.clips[0]?.references).toEqual([{ kind: "picture", id: expect.any(String), slot: 1 }]);
      expect({ ...body, clips: undefined, source_video: undefined }).toEqual({ ...first, clips: undefined, source_video: undefined });
    }
  });

  it("reports the frames past the clip count and the empty fields", async () => {
    const { bridge } = await makeBridge();
    const r = render(bridge, { chain: makeChain({ clips: 1 }) });
    act(() => {
      r.result.current.addFrame();
      r.result.current.setFieldPath(0, 3, "C:\\work\\nothing");
    });
    expect(r.result.current.unusedFrameStart).toBe(1);
    await scanned(r);
    expect(r.result.current.emptyFields).toEqual([{ type: "ref", frame: 0, kind: "video", slot: 1 }]);
    expect(r.result.current.allFieldsEmpty).toBe(false);
  });
});
