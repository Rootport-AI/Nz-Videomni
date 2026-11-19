import { describe, expect, it } from "vitest";
import type { GenerateChainRequest } from "../../api/types";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import type { MockFs } from "../../bridge/mockBridge";
import type { BridgeMethod, NativeBridge, ParamsOf, ResultOf } from "../../bridge";
import { createInitialState, emptySlot } from "../single/referenceRules";
import type { ReferenceKind, ReferenceRules, ReferenceState } from "../single/referenceRules";
import { JOIN_HANDLE_CROSSFADE_MS, JOIN_SOURCE_TAIL_SECONDS, V2vRefLongRunner } from "./runner";
import type { V2vRefLongRunnerSettings, V2vRowPatch } from "./runner";
import type { ResolvedClip, RowPlan } from "./types";

// Real timers; the poll interval and the 409 backoff are a few ms.
const POLL_MS = 3;
const BACKOFF_MS = 3;
const OUT_DIR = "C:\\out";

const RULES: ReferenceRules = {
  maxImages: 9,
  maxVideos: 3,
  maxAudios: 3,
  maxTotal: 12,
  videoTotalSec: 15,
  audioTotalSec: 15,
  clipMinSec: 1,
};

function template(): GenerateChainRequest {
  return {
    prompt: "a cat",
    width: 768,
    height: 512,
    frame_rate: 24,
    seed: 1,
    overlap_frames: 22,
    overlap_strength: 1,
    clips: [{ num_frames: 49 }, { num_frames: 49 }],
  };
}

function refs(cards: { kind: ReferenceKind; slot: number; path: string; durationSec?: number }[]): ReferenceState {
  const state: ReferenceState = { ...createInitialState(RULES), enabled: true, a2vSlot: null };
  const slots = { picture: [...state.slots.picture], video: [...state.slots.video], audio: [...state.slots.audio] };
  cards.forEach((c, i) => {
    slots[c.kind][c.slot - 1] = {
      ...emptySlot(c.kind, c.slot),
      status: "ready",
      fileName: c.path.slice(c.path.lastIndexOf("\\") + 1),
      filePath: c.path,
      durationSec: c.kind === "picture" ? null : (c.durationSec ?? 3),
      addedAt: i + 1,
    };
  });
  return { ...state, slots };
}

function clip(cards: Parameters<typeof refs>[0] = []): ResolvedClip {
  return { promptMode: "add", prompt: "", refs: refs(cards) };
}

function plan(index: number, patch: Partial<RowPlan> = {}): RowPlan {
  return {
    index,
    sourcePath: `C:\\src\\s${index}.mp4`,
    mainPrompt: "",
    clips: [clip(), clip()],
    outFileName: `s${index}.mp4`,
    ...patch,
  };
}

function settings(patch: Partial<V2vRefLongRunnerSettings> = {}): V2vRefLongRunnerSettings {
  return {
    mainPromptMode: "add",
    contextFrames: 22,
    rules: RULES,
    role: "ref2va",
    joined: true,
    joinFailedText: (message) => `JOIN FAILED: ${message}`,
    ...patch,
  };
}

interface Call {
  method: string;
  params: Record<string, unknown>;
}

type Override = (method: string, params: Record<string, unknown>) => unknown;

/** Records every call; `override` may answer a call instead of the mock. */
async function harness(opts: { fs?: MockFs; override?: Override } = {}) {
  const base = createMockBridge({ delayMs: 0, runningPollCount: 1, fs: opts.fs ?? createMockFs() });
  // The fixture server refuses `references` unless H3 is the loaded base model.
  await base.request("backend.request", { method: "POST", path: "/api/v1/pipeline/load", body: { base_model: "MiniMaxH3" } });
  const calls: Call[] = [];
  const bridge: NativeBridge = {
    async request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
      const p = params as unknown as Record<string, unknown>;
      calls.push({ method, params: p });
      const answer = opts.override?.(method, p);
      if (answer !== undefined) return (await answer) as ResultOf<M>;
      return base.request(method, params);
    },
    requestWithFiles: (method, params, files) => base.requestWithFiles(method, params, files),
    on: (event, handler) => base.on(event, handler),
  };
  const rows = new Map<number, V2vRowPatch>();
  const patches: Array<[number, V2vRowPatch]> = [];
  const onRowPatched = (index: number, patch: V2vRowPatch) => {
    patches.push([index, patch]);
    rows.set(index, { ...rows.get(index), ...patch });
  };
  return { bridge, calls, rows, patches, onRowPatched };
}

function isRequest(c: Call, method: string, path: RegExp): boolean {
  return c.method === "backend.request" && c.params.method === method && path.test(String(c.params.path));
}

async function runAll(
  h: Awaited<ReturnType<typeof harness>>,
  plans: RowPlan[],
  s: V2vRefLongRunnerSettings = settings(),
  runner = new V2vRefLongRunner(h.bridge),
) {
  return runner.start({
    template: template(),
    settings: s,
    plans,
    outDir: OUT_DIR,
    onRowPatched: h.onRowPatched,
    pollIntervalMs: POLL_MS,
    jobBusyBackoffMs: BACKOFF_MS,
  });
}

function chainBodies(calls: Call[]): GenerateChainRequest[] {
  return calls
    .filter((c) => isRequest(c, "POST", /\/generate\/chain$/))
    .map((c) => c.params.body as GenerateChainRequest);
}

describe("V2vRefLongRunner", () => {
  it("uploads a key once across rows and clips, and a path used as Video and Audio twice", async () => {
    const h = await harness();
    const shared = "C:\\ref\\shared.mp4";
    const plans = [
      plan(0, {
        clips: [
          clip([
            { kind: "picture", slot: 1, path: "C:\\ref\\p.png" },
            { kind: "video", slot: 1, path: shared },
            { kind: "audio", slot: 1, path: shared },
          ]),
          clip([{ kind: "picture", slot: 1, path: "C:\\ref\\p.png" }]),
        ],
      }),
      plan(1, { sourcePath: "C:\\src\\s0.mp4", clips: [clip([{ kind: "picture", slot: 1, path: "C:\\ref\\p.png" }]), clip()] }),
    ];
    const result = await runAll(h, plans);
    expect(result).toEqual({ started: true, reason: "completed" });

    const uploads = h.calls.filter((c) => c.method === "backend.uploadFile").map((c) => `${String(c.params.kind)}:${String(c.params.filePath)}`);
    expect(uploads).toEqual([
      "video:C:\\src\\s0.mp4",
      "image:C:\\ref\\p.png",
      `video:${shared}`,
      `audio:${shared}`,
    ]);
    expect(h.rows.get(0)?.stat).toBe("Done");
    expect(h.rows.get(1)?.stat).toBe("Done");
  });

  it("sends source_video with the context frames and each clip's references", async () => {
    const h = await harness();
    await runAll(h, [
      plan(0, {
        mainPrompt: "row text",
        clips: [clip([{ kind: "picture", slot: 2, path: "C:\\ref\\p.png" }]), clip([{ kind: "audio", slot: 1, path: "C:\\ref\\a.wav", durationSec: 4 }])],
      }),
    ]);
    const [body] = chainBodies(h.calls);
    const ids = new Map(
      h.calls
        .filter((c) => c.method === "backend.uploadFile")
        .map((c, i) => [String(c.params.filePath), `mock-${String(c.params.kind)}-${i + 1}`]),
    );
    expect(body?.prompt).toBe("a cat row text");
    expect(body?.source_video).toEqual({ video_id: ids.get("C:\\src\\s0.mp4"), context_frames: 22 });
    expect(body?.clips[0]?.references).toEqual([{ kind: "picture", id: ids.get("C:\\ref\\p.png"), slot: 2 }]);
    expect(body?.clips[1]?.references).toEqual([{ kind: "audio", id: ids.get("C:\\ref\\a.wav"), slot: 1 }]);
  });

  it("leaves source_video out for a row without an opening video", async () => {
    const h = await harness();
    await runAll(h, [plan(0, { sourcePath: null, clips: [clip([{ kind: "picture", slot: 1, path: "C:\\ref\\p.png" }]), clip()] })]);
    expect(chainBodies(h.calls)[0]?.source_video).toBeUndefined();
  });

  it("retries a 409 and then submits", async () => {
    let busy = 2;
    const h = await harness({
      override: (method, p) => {
        if (method === "backend.request" && p.method === "POST" && p.path === "/api/v1/generate/chain" && busy > 0) {
          busy -= 1;
          return { status: 409, body: { error: { code: "JOB_BUSY" } } };
        }
        return undefined;
      },
    });
    await runAll(h, [plan(0)]);
    expect(chainBodies(h.calls)).toHaveLength(3);
    expect(h.rows.get(0)).toMatchObject({ stat: "Done" });
  });

  it("fails the row after three 409s", async () => {
    const h = await harness({
      override: (method, p) =>
        method === "backend.request" && p.path === "/api/v1/generate/chain" ? { status: 409, body: null } : undefined,
    });
    await runAll(h, [plan(0)]);
    expect(chainBodies(h.calls)).toHaveLength(3);
    expect(h.rows.get(0)).toMatchObject({ stat: "Failed", error: "server busy (409) after retries" });
  });

  it("Joined on with an opening video: joins, then saves the joined video", async () => {
    const h = await harness();
    await runAll(h, [plan(0)]);
    const join = h.calls.find((c) => isRequest(c, "POST", /\/join$/));
    expect(join?.params).toMatchObject({
      body: { handle_crossfade_ms: JOIN_HANDLE_CROSSFADE_MS, source_tail_seconds: JOIN_SOURCE_TAIL_SECONDS },
      timeoutMs: 120_000,
    });
    expect(JOIN_HANDLE_CROSSFADE_MS).toBe(300);
    expect(JOIN_SOURCE_TAIL_SECONDS).toBe(5);
    const download = h.calls.find((c) => c.method === "backend.downloadVideo");
    expect(download?.params).toEqual({
      jobId: expect.any(String),
      kind: "joined",
      destDir: OUT_DIR,
      fileName: "s0.mp4",
      noClobber: true,
    });
    expect(h.rows.get(0)).toEqual({ stat: "Done", output: "s0.mp4", error: "" });

    // Joined first, then saved — the same job.
    const joinAt = h.calls.findIndex((c) => isRequest(c, "POST", /\/join$/));
    const saveAt = h.calls.findIndex((c) => c.method === "backend.downloadVideo");
    expect(joinAt).toBeGreaterThanOrEqual(0);
    expect(joinAt).toBeLessThan(saveAt);
    expect(join?.params.path).toBe(`/api/v1/jobs/${String(download?.params.jobId)}/join`);
  });

  it("Joined off, or no opening video: no join, the plain output is saved", async () => {
    for (const [s, p] of [
      [settings({ joined: false }), plan(0)],
      [settings(), plan(0, { sourcePath: null, clips: [clip([{ kind: "picture", slot: 1, path: "C:\\ref\\p.png" }]), clip()] })],
    ] as const) {
      const h = await harness();
      await runAll(h, [p], s);
      expect(h.calls.some((c) => isRequest(c, "POST", /\/join$/))).toBe(false);
      const download = h.calls.find((c) => c.method === "backend.downloadVideo");
      expect(download?.params).toEqual({ jobId: expect.any(String), destDir: OUT_DIR, fileName: "s0.mp4", noClobber: true });
      expect(h.rows.get(0)?.stat).toBe("Done");
    }
  });

  it("a failed join fails that row with the join text and the run goes on", async () => {
    let first = true;
    const h = await harness({
      override: (method, p) => {
        if (method === "backend.request" && /\/join$/.test(String(p.path)) && first) {
          first = false;
          return { status: 422, body: { error: { code: "JOB_NOT_JOINABLE" } } };
        }
        return undefined;
      },
    });
    await runAll(h, [plan(0), plan(1)]);
    expect(h.rows.get(0)?.stat).toBe("Failed");
    expect(h.rows.get(0)?.error).toMatch(/^JOIN FAILED: 422: .*JOB_NOT_JOINABLE/);
    expect(h.rows.get(1)).toMatchObject({ stat: "Done", output: "s1.mp4" });
    expect(h.calls.filter((c) => c.method === "backend.downloadVideo")).toHaveLength(1);
  });

  it("a join that throws is a failed join too", async () => {
    const h = await harness({
      override: (method, p) =>
        method === "backend.request" && /\/join$/.test(String(p.path)) ? Promise.reject(new Error("timeout")) : undefined,
    });
    await runAll(h, [plan(0)]);
    expect(h.rows.get(0)).toMatchObject({ stat: "Failed", error: "JOIN FAILED: Error: timeout" });
  });

  it("an upload failure fails only that row", async () => {
    const h = await harness({
      override: (method, p) =>
        method === "backend.uploadFile" && p.filePath === "C:\\src\\s0.mp4"
          ? { status: 413, body: { error: { code: "FILE_TOO_LARGE" } } }
          : undefined,
    });
    await runAll(h, [plan(0), plan(1)]);
    expect(h.rows.get(0)?.stat).toBe("Failed");
    expect(h.rows.get(0)?.error).toMatch(/upload video 413/);
    expect(h.rows.get(1)?.stat).toBe("Done");
    expect(chainBodies(h.calls)).toHaveLength(1);
  });

  it("a failed upload is not remembered: the next row using the file uploads it again", async () => {
    let failed = false;
    const h = await harness({
      override: (method) => {
        if (method === "backend.uploadFile" && !failed) {
          failed = true;
          return { status: 500, body: null };
        }
        return undefined;
      },
    });
    await runAll(h, [plan(0), plan(1, { sourcePath: "C:\\src\\s0.mp4" })]);
    const uploads = h.calls.filter((c) => c.method === "backend.uploadFile").map((c) => c.params.filePath);
    expect(uploads).toEqual(["C:\\src\\s0.mp4", "C:\\src\\s0.mp4"]);
    expect(h.rows.get(0)?.stat).toBe("Failed");
    expect(h.rows.get(1)?.stat).toBe("Done");
  });

  it("an upload answer without an id fails the row", async () => {
    const h = await harness({ override: (method) => (method === "backend.uploadFile" ? { status: 200, body: {} } : undefined) });
    await runAll(h, [plan(0)]);
    expect(h.rows.get(0)).toMatchObject({ stat: "Failed", error: "Error: upload video: response missing video_id" });
  });

  it("a failed job keeps its reason; a cancelled job goes back to Waiting", async () => {
    const h = await harness();
    await runAll(h, [plan(0, { mainPrompt: "__MOCK_FAIL__" }), plan(1, { mainPrompt: "__MOCK_CANCEL__" })]);
    expect(h.rows.get(0)?.stat).toBe("Failed");
    expect(h.rows.get(0)?.error).not.toBe("");
    expect(h.rows.get(1)).toEqual({ stat: "Waiting", error: "" });
  });

  it("stop sends no DELETE and lets the current row finish", async () => {
    const h = await harness();
    const runner = new V2vRefLongRunner(h.bridge);
    const done = runAll(h, [plan(0), plan(1)], settings(), runner);
    expect(runner.state).toBe("running");
    runner.stop();
    expect(runner.state).toBe("stopping");
    await done;
    expect(runner.state).toBe("idle");
    expect(h.calls.some((c) => c.method === "backend.request" && c.params.method === "DELETE")).toBe(false);
    expect(h.rows.get(0)?.stat).toBe("Done");
    expect(h.rows.has(1)).toBe(false);
  });

  it("records the name actually written (noClobber's _2)", async () => {
    const fs = createMockFs({ folders: { [OUT_DIR]: [{ name: "s0.mp4", sizeBytes: 1, mtimeMs: 0 }] } });
    const h = await harness({ fs });
    await runAll(h, [plan(0)]);
    expect(h.rows.get(0)).toMatchObject({ stat: "Done", output: "s0_2.mp4" });
  });

  it("marks each row Generating first, and reports only stat/output/error", async () => {
    const h = await harness();
    await runAll(h, [plan(3)]);
    expect(h.patches[0]).toEqual([3, { stat: "Generating", error: "" }]);
    for (const [, patch] of h.patches) {
      expect(Object.keys(patch).every((k) => k === "stat" || k === "output" || k === "error")).toBe(true);
    }
  });

  it("does not start with no plan or while running", async () => {
    const h = await harness();
    const runner = new V2vRefLongRunner(h.bridge);
    expect(await runAll(h, [], settings(), runner)).toEqual({ started: false, reason: "no rows to process" });
    const first = runAll(h, [plan(0)], settings(), runner);
    expect(await runAll(h, [plan(1)], settings(), runner)).toEqual({ started: false, reason: "batch already running" });
    await first;
  });
});
