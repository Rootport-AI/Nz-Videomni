import { describe, expect, it } from "vitest";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import type { BridgeMethod, NativeBridge, ParamsOf, ResultOf } from "../../bridge";
import type { NagSettings } from "../../shell/nagSettings";
import { ACCELERATION_DEFAULTS } from "../../shell/accelerationSettings";
import { BatchRunner } from "./batchRunner";
import type { BatchRunnerSettings } from "./batchRunner";
import { IMAGE_SHARED } from "./manifestMerge";
import type { BatchRow, BatchStat } from "./manifestMerge";

// Real timers throughout, like `useGeneration.test.ts` — `pollIntervalMs`/
// `jobBusyBackoffMs` are overridden to a few ms per test so the whole suite
// stays fast; production call sites leave both at their real (1s/3s)
// defaults.
const POLL_MS = 4;
const BACKOFF_MS = 4;

const BASE_SETTINGS: BatchRunnerSettings = {
  promptCommon: "a cat",
  promptMode: "add",
  width: 512,
  height: 320,
  frameRate: 24,
  seed: -1,
  chunkedUpsample: true,
};

function makeRow(queue: number, wav: string, stat: BatchStat, overrides: Partial<BatchRow> = {}): BatchRow {
  return {
    queue,
    wav,
    duration: 3.0,
    image: IMAGE_SHARED,
    prompt: "",
    stat,
    output: "",
    frames: 49,
    skipReason: "",
    error: "",
    ...overrides,
  };
}

interface RecordedCall {
  method: string;
  params: unknown;
}

/** Wraps a `NativeBridge` to record every call, for assertions like "zero
 * calls ever referenced this row's wav file" without needing `vi.fn`
 * spies on the mock bridge's internals. */
function spyBridge(bridge: NativeBridge): { wrapped: NativeBridge; calls: RecordedCall[] } {
  const calls: RecordedCall[] = [];
  const wrapped: NativeBridge = {
    async request(method, params) {
      calls.push({ method, params });
      return bridge.request(method, params);
    },
    async requestWithFiles(method, params, files) {
      calls.push({ method, params });
      return bridge.requestWithFiles(method, params, files);
    },
    on(event, handler) {
      return bridge.on(event, handler);
    },
  };
  return { wrapped, calls };
}

/** Polls `predicate` until it returns true or `timeoutMs` elapses (pure
 * setTimeout polling — no `@testing-library/react` dependency, since
 * `batchRunner.ts` is deliberately React-agnostic and its tests should be
 * too). */
async function waitUntil(predicate: () => boolean, timeoutMs = 2_000, intervalMs = 2): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error("waitUntil: timed out");
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

describe("BatchRunner", () => {
  it("processes only Waiting/Failed/Generating rows; Done/Skip are left untouched with zero bridge calls", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
    const { wrapped, calls } = spyBridge(bridge);
    const runner = new BatchRunner(wrapped);

    const rows: BatchRow[] = [
      makeRow(1, "a.wav", "Waiting"),
      makeRow(2, "b.wav", "Failed", { error: "old error" }),
      makeRow(3, "c.wav", "Generating"),
      makeRow(4, "d.wav", "Done", { output: "d.mp4" }),
      makeRow(5, "e.wav", "Skip", { skipReason: "over-cap" }),
    ];
    let latest: BatchRow[] = rows;

    const result = await runner.start({
      wavDir: "C:\\batch\\in",
      outDir: "C:\\batch\\out",
      settings: BASE_SETTINGS,
      rows,
      onRowsChanged: (r) => {
        latest = r;
      },
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    expect(result.started).toBe(true);
    const byWav = new Map(latest.map((r) => [r.wav, r]));
    expect(byWav.get("a.wav")?.stat).toBe("Done");
    expect(byWav.get("b.wav")?.stat).toBe("Done");
    expect(byWav.get("c.wav")?.stat).toBe("Done");
    // Untouched: no processing, no row field changes.
    expect(byWav.get("d.wav")).toMatchObject({ stat: "Done", output: "d.mp4" });
    expect(byWav.get("e.wav")).toMatchObject({ stat: "Skip", skipReason: "over-cap" });

    // No bridge call (upload/submit/download) ever referenced the Done/Skip
    // rows' files — they are skipped entirely, never processed.
    const serialized = JSON.stringify(calls);
    expect(serialized).not.toContain("d.wav");
    expect(serialized).not.toContain("e.wav");
  });

  it("returns started:false and touches nothing when every row is Done/Skip", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const { wrapped, calls } = spyBridge(bridge);
    const runner = new BatchRunner(wrapped);
    const rows: BatchRow[] = [makeRow(1, "a.wav", "Done", { output: "a.mp4" }), makeRow(2, "b.wav", "Skip")];

    const result = await runner.start({
      wavDir: "C:\\batch\\in",
      outDir: "C:\\batch\\out",
      settings: BASE_SETTINGS,
      rows,
    });

    expect(result).toEqual({ started: false, reason: "no rows to process" });
    expect(calls).toHaveLength(0);
    expect(runner.state).toBe("idle");
  });

  it("rejects a second concurrent start() while already running", async () => {
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1 });
    const runner = new BatchRunner(bridge);
    const rows: BatchRow[] = [makeRow(1, "a.wav", "Waiting")];

    const first = runner.start({
      wavDir: "C:\\batch\\in",
      outDir: "C:\\batch\\out",
      settings: BASE_SETTINGS,
      rows,
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });
    const second = await runner.start({
      wavDir: "C:\\batch\\in",
      outDir: "C:\\batch\\out",
      settings: BASE_SETTINGS,
      rows,
    });
    expect(second).toEqual({ started: false, reason: "batch already running" });

    await first;
  });

  it("completes normally: Done + output filename recorded, downloaded into the output folder", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
    const runner = new BatchRunner(bridge);
    const rows: BatchRow[] = [makeRow(1, "line01.wav", "Waiting")];
    let latest: BatchRow[] = rows;

    await runner.start({
      wavDir: "C:\\batch\\in",
      outDir: "C:\\batch\\out",
      settings: BASE_SETTINGS,
      rows,
      onRowsChanged: (r) => {
        latest = r;
      },
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    expect(latest[0]).toMatchObject({ stat: "Done", output: "line01.mp4", error: "" });
    expect(fs.folders.get("C:\\batch\\out")?.some((e) => e.name === "line01.mp4")).toBe(true);
  });

  it("uploads a per-row image once per run, cached by path, across rows that share the same image file", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
    const { wrapped, calls } = spyBridge(bridge);
    const runner = new BatchRunner(wrapped);
    const rows: BatchRow[] = [
      makeRow(1, "a.wav", "Waiting", { image: "face.png" }),
      makeRow(2, "b.wav", "Waiting", { image: "face.png" }),
    ];

    await runner.start({
      wavDir: "C:\\batch\\in",
      imgDir: "C:\\batch\\img",
      outDir: "C:\\batch\\out",
      settings: BASE_SETTINGS,
      rows,
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    const imageUploads = calls.filter(
      (c) => c.method === "backend.uploadFile" && (c.params as { kind: string }).kind === "image",
    );
    expect(imageUploads).toHaveLength(1);
  });

  it("marks a row Failed with the server's error summary when the job fails", async () => {
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1 });
    const runner = new BatchRunner(bridge);
    const rows: BatchRow[] = [makeRow(1, "a.wav", "Waiting", { prompt: "__MOCK_FAIL__" })];
    let latest: BatchRow[] = rows;

    await runner.start({
      wavDir: "C:\\batch\\in",
      outDir: "C:\\batch\\out",
      settings: { ...BASE_SETTINGS, promptCommon: "" },
      rows,
      onRowsChanged: (r) => {
        latest = r;
      },
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    expect(latest[0]?.stat).toBe("Failed");
    expect(latest[0]?.error).toContain("GENERATION_FAILED");
  });

  it("marks a row Failed after exhausting 409 JOB_BUSY retries", async () => {
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 50 });
    // Occupy the single job slot so every /generate/chain attempt 409s.
    await bridge.request("backend.request", {
      method: "POST",
      path: "/api/v1/generate",
      body: { prompt: "occupying the slot" },
    });

    const runner = new BatchRunner(bridge);
    const rows: BatchRow[] = [makeRow(1, "a.wav", "Waiting")];
    let latest: BatchRow[] = rows;

    await runner.start({
      wavDir: "C:\\batch\\in",
      outDir: "C:\\batch\\out",
      settings: BASE_SETTINGS,
      rows,
      onRowsChanged: (r) => {
        latest = r;
      },
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    expect(latest[0]).toMatchObject({ stat: "Failed", error: "server busy (409) after retries" });
  }, 10_000);

  it("stop() best-effort cancels the in-flight job and rewinds it to Waiting", async () => {
    // runningPollCount:0 makes the mock settle non-queued statuses almost
    // immediately once polled, so the cancel (forced via DELETE) surfaces on
    // the run's 2nd poll rather than requiring a long "running" phase.
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 0 });
    const { wrapped, calls } = spyBridge(bridge);
    const runner = new BatchRunner(wrapped);
    const rows: BatchRow[] = [makeRow(1, "a.wav", "Waiting")];
    let latest: BatchRow[] = rows;

    const startPromise = runner.start({
      wavDir: "C:\\batch\\in",
      outDir: "C:\\batch\\out",
      settings: BASE_SETTINGS,
      rows,
      onRowsChanged: (r) => {
        latest = r;
      },
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    // Wait until the first GET /jobs/{id} poll has actually fired — at that
    // point `currentJobId` is guaranteed set (it's assigned synchronously
    // right after the submit resolves, strictly before `pollRow` issues its
    // first GET). Stopping any earlier would race the DELETE ahead of the
    // job even existing, and the run would simply complete normally.
    await waitUntil(() =>
      calls.some(
        (c) =>
          c.method === "backend.request" &&
          (c.params as { method?: string; path?: string }).method === "GET" &&
          (c.params as { path?: string }).path?.startsWith("/api/v1/jobs/"),
      ),
    );
    runner.stop();

    const result = await startPromise;
    expect(result.started).toBe(true);
    expect(latest[0]).toMatchObject({ stat: "Waiting", error: "" });
    expect(runner.state).toBe("idle");
  }, 10_000);

  it("reflects noClobber's renamed output file in the output column on a collision", async () => {
    const fs = createMockFs({
      folders: { "C:\\batch\\out": [{ name: "line01.mp4", sizeBytes: 1, mtimeMs: 1 }] },
    });
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
    const runner = new BatchRunner(bridge);
    const rows: BatchRow[] = [makeRow(1, "line01.wav", "Waiting")];
    let latest: BatchRow[] = rows;

    await runner.start({
      wavDir: "C:\\batch\\in",
      outDir: "C:\\batch\\out",
      settings: BASE_SETTINGS,
      rows,
      onRowsChanged: (r) => {
        latest = r;
      },
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    expect(latest[0]).toMatchObject({ stat: "Done", output: "line01_2.mp4" });
  });

  it("re-runs a Failed row on the next start() and can succeed", async () => {
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1 });
    const runner = new BatchRunner(bridge);
    const rows: BatchRow[] = [makeRow(1, "a.wav", "Failed", { error: "previous attempt failed" })];
    let latest: BatchRow[] = rows;

    await runner.start({
      wavDir: "C:\\batch\\in",
      outDir: "C:\\batch\\out",
      settings: BASE_SETTINGS,
      rows,
      onRowsChanged: (r) => {
        latest = r;
      },
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    expect(latest[0]).toMatchObject({ stat: "Done", error: "" });
  });

  /** Wraps a bridge to capture every `POST /generate/chain` body, then
   * returns a 500 so the row fails immediately (no job polling needed) —
   * mirrors `useBatchForm.test.ts`'s `captureChainBridge`. Hoisted out of the
   * `settings.nag` block 2026-07-31 so the acceleration block below can share
   * it verbatim rather than keeping a second copy in sync. */
  function captureChainBridge(base: NativeBridge): {
    wrapped: NativeBridge;
    chainBodies: Array<Record<string, unknown>>;
  } {
    const chainBodies: Array<Record<string, unknown>> = [];
    const wrapped: NativeBridge = {
      async request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
        if (method === "backend.request") {
          const p = params as { method?: string; path?: string; body?: Record<string, unknown> };
          if (p.method === "POST" && p.path === "/api/v1/generate/chain") {
            chainBodies.push(p.body ?? {});
            return {
              status: 500,
              body: { error: { code: "MOCK_STOP", message: "captured by test" } },
            } as unknown as ResultOf<M>;
          }
        }
        return base.request(method, params);
      },
      requestWithFiles: (method, params, files) => base.requestWithFiles(method, params, files),
      on: (event, handler) => base.on(event, handler),
    };
    return { wrapped, chainBodies };
  }

  // NAG (2026-07-28): `settings.nag` threads through `buildA2vChainPayload`
  // into every row's `/generate/chain` body.
  describe("settings.nag", () => {
    const ENABLED_NAG: NagSettings = {
      text: "blurry, low quality, distorted, watermark, text",
      enabled: true,
      scale: 11.0,
      tau: 2.5,
      alpha: 0.25,
      method: "nag",
      vsfScale: 1.5,
    };

    it("present (enabled): the captured chain body carries the 7 additive NAG fields", async () => {
      const fs = createMockFs();
      const base = createMockBridge({ delayMs: 0, fs });
      const { wrapped, chainBodies } = captureChainBridge(base);
      const runner = new BatchRunner(wrapped);
      const rows: BatchRow[] = [makeRow(1, "a.wav", "Waiting")];

      await runner.start({
        wavDir: "C:\\batch\\in",
        outDir: "C:\\batch\\out",
        settings: { ...BASE_SETTINGS, nag: ENABLED_NAG },
        rows,
      });

      expect(chainBodies).toHaveLength(1);
      const body = chainBodies[0]!;
      expect(body.negative_prompt).toBe(ENABLED_NAG.text);
      expect(body.nag_enabled).toBe(true);
      expect(body.nag_scale).toBe(11.0);
      expect(body.nag_tau).toBe(2.5);
      expect(body.nag_alpha).toBe(0.25);
      expect(body.neg_method).toBe("nag");
      expect(body.vsf_scale).toBe(1.5);
    });

    it("absent: the captured chain body carries none of the 7 additive NAG fields", async () => {
      const fs = createMockFs();
      const base = createMockBridge({ delayMs: 0, fs });
      const { wrapped, chainBodies } = captureChainBridge(base);
      const runner = new BatchRunner(wrapped);
      const rows: BatchRow[] = [makeRow(1, "a.wav", "Waiting")];

      await runner.start({
        wavDir: "C:\\batch\\in",
        outDir: "C:\\batch\\out",
        settings: BASE_SETTINGS,
        rows,
      });

      expect(chainBodies).toHaveLength(1);
      const body = chainBodies[0]!;
      // `negative_prompt` is a pre-NAG REQUIRED field on the A2V payload
      // (`buildA2vChainPayload` always sends it, defaulting to `""`) — NAG
      // being absent leaves it at that pre-existing default rather than
      // omitting the key; only the 6 `nag_*`/`neg_method`/`vsf_scale` keys
      // are genuinely absent.
      expect(body.negative_prompt).toBe("");
      expect(body).not.toHaveProperty("nag_enabled");
      expect(body).not.toHaveProperty("nag_scale");
      expect(body).not.toHaveProperty("nag_tau");
      expect(body).not.toHaveProperty("nag_alpha");
      expect(body).not.toHaveProperty("neg_method");
      expect(body).not.toHaveProperty("vsf_scale");
    });
  });

  // Acceleration (2026-07-31): `settings.acceleration` threads through
  // `buildA2vChainPayload` into every row's `/generate/chain` body — Batch has
  // no acceleration UI of its own, it inherits the Settings panel's choice.
  describe("settings.acceleration", () => {
    async function runOneRow(settings: BatchRunnerSettings): Promise<Record<string, unknown>> {
      const fs = createMockFs();
      const base = createMockBridge({ delayMs: 0, fs });
      const { wrapped, chainBodies } = captureChainBridge(base);
      const runner = new BatchRunner(wrapped);
      await runner.start({
        wavDir: "C:\\batch\\in",
        outDir: "C:\\batch\\out",
        settings,
        rows: [makeRow(1, "a.wav", "Waiting")],
      });
      expect(chainBodies).toHaveLength(1);
      return chainBodies[0]!;
    }

    it("absent / all-defaults: the row body is byte-identical and carries no acceleration key", async () => {
      const baseline = await runOneRow(BASE_SETTINGS);
      const withDefaults = await runOneRow({ ...BASE_SETTINGS, acceleration: ACCELERATION_DEFAULTS });
      expect(JSON.stringify(withDefaults)).toBe(JSON.stringify(baseline));
      expect(withDefaults).not.toHaveProperty("attention_backend");
      expect(withDefaults).not.toHaveProperty("fused_gguf_dequant_kernel");
      expect(withDefaults).not.toHaveProperty("vae_mode");
    });

    it("sage: exactly one key more on every row body", async () => {
      const baseline = await runOneRow(BASE_SETTINGS);
      const sage = await runOneRow({
        ...BASE_SETTINGS,
        acceleration: { ...ACCELERATION_DEFAULTS, attentionBackend: "sage" },
      });
      expect(sage.attention_backend).toBe("sage");
      expect(Object.keys(sage)).toEqual([...Object.keys(baseline), "attention_backend"]);
    });

    it("block-swap prefetch off: exactly one key more on every row body", async () => {
      // S4: default is now true, so OFF is what diverges.
      const baseline = await runOneRow(BASE_SETTINGS);
      const prefetch = await runOneRow({
        ...BASE_SETTINGS,
        acceleration: { ...ACCELERATION_DEFAULTS, blockSwapPrefetch: false },
      });
      expect(prefetch.block_swap_prefetch).toBe(false);
      expect(Object.keys(prefetch)).toEqual([...Object.keys(baseline), "block_swap_prefetch"]);
    });

    it("keep-resident on: exactly one key more on every row body", async () => {
      // §48: server default is false, so ON is what diverges.
      const baseline = await runOneRow(BASE_SETTINGS);
      const keepResident = await runOneRow({
        ...BASE_SETTINGS,
        acceleration: { ...ACCELERATION_DEFAULTS, keepResident: true },
      });
      expect(keepResident.keep_resident).toBe(true);
      expect(Object.keys(keepResident)).toEqual([...Object.keys(baseline), "keep_resident"]);
    });

    it("fused GGUF dequant kernel off: exactly one key more on every row body (§1-11)", async () => {
      // §51 (2026-08-04): server default flipped to true, so OFF is what
      // diverges.
      const baseline = await runOneRow(BASE_SETTINGS);
      const fused = await runOneRow({
        ...BASE_SETTINGS,
        acceleration: { ...ACCELERATION_DEFAULTS, fusedGgufDequantKernel: false },
      });
      expect(fused.fused_gguf_dequant_kernel).toBe(false);
      expect(Object.keys(fused)).toEqual([...Object.keys(baseline), "fused_gguf_dequant_kernel"]);
    });

    it("PrunaVAED: exactly one key more on every row body (§52)", async () => {
      // §52 (2026-08-05): server default is "default", so PrunaVAED is what
      // diverges — the same direction as keep-resident.
      const baseline = await runOneRow(BASE_SETTINGS);
      const pruned = await runOneRow({
        ...BASE_SETTINGS,
        acceleration: { ...ACCELERATION_DEFAULTS, vaeMode: "prune_vaed" },
      });
      expect(pruned.vae_mode).toBe("prune_vaed");
      expect(Object.keys(pruned)).toEqual([...Object.keys(baseline), "vae_mode"]);
    });
  });

  // §1-19 (2026-08-11): Batch a2v is one of the two full-length-window
  // callers of `buildA2vChainPayload` — every row's `/generate/chain` body
  // must carry the fixed `stage2_window: "full_length"` key, unconditionally
  // (there is no batch setting that could turn it off).
  describe("stage2_window (§1-19)", () => {
    it("every row body carries the fixed full-length stage-2 window", async () => {
      const fs = createMockFs();
      const base = createMockBridge({ delayMs: 0, fs });
      const { wrapped, chainBodies } = captureChainBridge(base);
      const runner = new BatchRunner(wrapped);
      const rows: BatchRow[] = [makeRow(1, "a.wav", "Waiting")];

      await runner.start({
        wavDir: "C:\\batch\\in",
        outDir: "C:\\batch\\out",
        settings: BASE_SETTINGS,
        rows,
      });

      expect(chainBodies).toHaveLength(1);
      expect(chainBodies[0]!.stage2_window).toBe("full_length");
    });
  });
});
