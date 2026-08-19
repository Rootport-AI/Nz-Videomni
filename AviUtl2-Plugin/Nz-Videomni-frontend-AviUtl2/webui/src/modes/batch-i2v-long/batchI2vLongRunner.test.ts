import { describe, expect, it } from "vitest";
import type { GenerateChainRequest } from "../../api/types";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import type { BridgeMethod, NativeBridge, ParamsOf, ResultOf } from "../../bridge";
import { BatchI2vLongRunner } from "./batchI2vLongRunner";
import type { I2vLongRunnerSettings } from "./batchI2vLongRunner";
import type { I2vLongRow, I2vLongStat } from "./imageRows";

// Real timers throughout, like `modes/batch/batchRunner.test.ts` —
// `pollIntervalMs`/`jobBusyBackoffMs` are overridden to a few ms per test so
// the suite stays fast; production call sites leave both at their real 1s/3s
// defaults.
const POLL_MS = 4;
const BACKOFF_MS = 4;

const IMG_DIR = "C:\\batch\\img";
const OUT_DIR = "C:\\batch\\out";

/** A realistic scratch-mode template: what `buildI2vLongTemplate` hands the
 * runner (2 clips, no `source_video`, clip 0's start frame already stripped). */
function template(overrides: Partial<GenerateChainRequest> = {}): GenerateChainRequest {
  return {
    prompt: "a cat walking",
    width: 768,
    height: 512,
    frame_rate: 24,
    seed: 42,
    overlap_frames: 3,
    overlap_strength: 0.5,
    clips: [{ num_frames: 121 }, { num_frames: 121, prompt: "clip 1 override" }],
    chunked_upsample: true,
    loras: [{ name: "style", strength: 1.0 }],
    ...overrides,
  };
}

function makeRow(queue: number, image: string, stat: I2vLongStat, overrides: Partial<I2vLongRow> = {}): I2vLongRow {
  return { queue, image, prompt: "", stat, output: "", error: "", ...overrides };
}

/** The run's non-template settings. Every test that does not care about prompt
 * composition uses this default (add mode, matching the panel's own default). */
const SETTINGS: I2vLongRunnerSettings = { promptMode: "add" };

interface RecordedCall {
  method: string;
  params: unknown;
}

/** Wraps a `NativeBridge` to record every call — same helper (and reason) as
 * `modes/batch/batchRunner.test.ts`'s. */
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

/** Wraps a bridge to capture every `POST /generate/chain` body and return a 500
 * so the row fails immediately (no job polling needed) — mirrors
 * `modes/batch/batchRunner.test.ts`'s `captureChainBridge`. */
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

/** Wraps a bridge so `backend.uploadFile` returns a 4xx for one specific image
 * file — the "one bad row must not take down the run" fixture. */
function failUploadFor(base: NativeBridge, badFileName: string): NativeBridge {
  return {
    async request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
      if (method === "backend.uploadFile") {
        const p = params as { filePath?: string };
        if (p.filePath?.endsWith(badFileName)) {
          return {
            status: 413,
            body: { error: { code: "FILE_TOO_LARGE", message: "image is too large" } },
          } as unknown as ResultOf<M>;
        }
      }
      return base.request(method, params);
    },
    requestWithFiles: (method, params, files) => base.requestWithFiles(method, params, files),
    on: (event, handler) => base.on(event, handler),
  };
}

async function waitUntil(predicate: () => boolean, timeoutMs = 2_000, intervalMs = 2): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error("waitUntil: timed out");
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

/** Every `DELETE /api/v1/jobs/...` the runner issued (must always be empty —
 * this runner never cancels anything). */
function deleteCalls(calls: RecordedCall[]): RecordedCall[] {
  return calls.filter(
    (c) => c.method === "backend.request" && (c.params as { method?: string }).method === "DELETE",
  );
}

describe("BatchI2vLongRunner", () => {
  it("未完了行(Waiting/Failed/Generating)だけを処理し、Done行は一切触らない", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
    const { wrapped, calls } = spyBridge(bridge);
    const runner = new BatchI2vLongRunner(wrapped);

    const rows: I2vLongRow[] = [
      makeRow(1, "a.png", "Waiting"),
      makeRow(2, "b.png", "Failed", { error: "old error" }),
      makeRow(3, "c.png", "Generating"),
      makeRow(4, "d.png", "Done", { output: "d.mp4" }),
    ];
    let latest: I2vLongRow[] = rows;

    const result = await runner.start({
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      template: template(),
      settings: SETTINGS,
      rows,
      onRowsChanged: (r) => {
        latest = r;
      },
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    expect(result).toEqual({ started: true, reason: "completed" });
    const byImage = new Map(latest.map((r) => [r.image, r]));
    expect(byImage.get("a.png")?.stat).toBe("Done");
    expect(byImage.get("b.png")?.stat).toBe("Done");
    expect(byImage.get("c.png")?.stat).toBe("Done");
    expect(byImage.get("d.png")).toMatchObject({ stat: "Done", output: "d.mp4" });

    // Done行のファイル名はどのブリッジ呼び出しにも現れない。
    expect(JSON.stringify(calls)).not.toContain("d.png");
    expect(runner.state).toBe("idle");
  });

  it("対象行ゼロならstarted:falseで、ブリッジ呼び出しは一切ない", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const { wrapped, calls } = spyBridge(bridge);
    const runner = new BatchI2vLongRunner(wrapped);

    const result = await runner.start({
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      template: template(),
      settings: SETTINGS,
      rows: [makeRow(1, "a.png", "Done", { output: "a.mp4" })],
    });

    expect(result).toEqual({ started: false, reason: "no rows to process" });
    expect(calls).toHaveLength(0);
    expect(runner.state).toBe("idle");
  });

  it("実行中の二重start()は拒否される", async () => {
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1 });
    const runner = new BatchI2vLongRunner(bridge);
    const rows: I2vLongRow[] = [makeRow(1, "a.png", "Waiting")];

    const first = runner.start({
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      template: template(),
      settings: SETTINGS,
      rows,
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });
    const second = await runner.start({ imgDir: IMG_DIR, outDir: OUT_DIR, template: template(), settings: SETTINGS, rows });

    expect(second).toEqual({ started: false, reason: "batch already running" });
    await first;
  });

  it("正常完了で Done + 出力ファイル名を記録し、出力フォルダへ保存する（画像stem.mp4）", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
    const runner = new BatchI2vLongRunner(bridge);
    const rows: I2vLongRow[] = [makeRow(1, "shot01.png", "Waiting")];
    let latest: I2vLongRow[] = rows;

    await runner.start({
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      template: template(),
      settings: SETTINGS,
      rows,
      onRowsChanged: (r) => {
        latest = r;
      },
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    expect(latest[0]).toMatchObject({ stat: "Done", output: "shot01.mp4", error: "" });
    expect(fs.folders.get(OUT_DIR)?.some((e) => e.name === "shot01.mp4")).toBe(true);
  });

  it("noClobberで別名保存された場合、実際のファイル名を出力列に記録する", async () => {
    const fs = createMockFs({ folders: { [OUT_DIR]: [{ name: "shot01.mp4", sizeBytes: 1, mtimeMs: 1 }] } });
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
    const runner = new BatchI2vLongRunner(bridge);
    const rows: I2vLongRow[] = [makeRow(1, "shot01.png", "Waiting")];
    let latest: I2vLongRow[] = rows;

    await runner.start({
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      template: template(),
      settings: SETTINGS,
      rows,
      onRowsChanged: (r) => {
        latest = r;
      },
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    expect(latest[0]).toMatchObject({ stat: "Done", output: "shot01_2.mp4" });
  });

  it("アップロード失敗の行はFailedになり、後続の行は処理が続く", async () => {
    const fs = createMockFs();
    const base = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
    const runner = new BatchI2vLongRunner(failUploadFor(base, "bad.png"));
    const rows: I2vLongRow[] = [makeRow(1, "bad.png", "Waiting"), makeRow(2, "good.png", "Waiting")];
    let latest: I2vLongRow[] = rows;

    await runner.start({
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      template: template(),
      settings: SETTINGS,
      rows,
      onRowsChanged: (r) => {
        latest = r;
      },
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    expect(latest[0]?.stat).toBe("Failed");
    expect(latest[0]?.error).toContain("upload image 413");
    expect(latest[1]).toMatchObject({ stat: "Done", output: "good.mp4" });
  });

  it("409(JOB_BUSY)が3回続いた行はFailedになる", async () => {
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 50 });
    // 単一ジョブ枠を占有して、以降の /generate/chain を必ず409にする。
    await bridge.request("backend.request", {
      method: "POST",
      path: "/api/v1/generate",
      body: { prompt: "occupying the slot" },
    });

    const runner = new BatchI2vLongRunner(bridge);
    const rows: I2vLongRow[] = [makeRow(1, "a.png", "Waiting")];
    let latest: I2vLongRow[] = rows;

    await runner.start({
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      template: template(),
      settings: SETTINGS,
      rows,
      onRowsChanged: (r) => {
        latest = r;
      },
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    expect(latest[0]).toMatchObject({ stat: "Failed", error: "server busy (409) after retries" });
  }, 10_000);

  it("ジョブがfailedになった行は、サーバのエラー文言つきでFailedになる", async () => {
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1 });
    const runner = new BatchI2vLongRunner(bridge);
    const rows: I2vLongRow[] = [makeRow(1, "a.png", "Waiting")];
    let latest: I2vLongRow[] = rows;

    await runner.start({
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      template: template({ prompt: "__MOCK_FAIL__" }),
      settings: SETTINGS,
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

  it("ジョブがcancelledになった行はWaitingへ巻き戻る（外部からのDELETE想定）", async () => {
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1 });
    const runner = new BatchI2vLongRunner(bridge);
    const rows: I2vLongRow[] = [makeRow(1, "a.png", "Waiting")];
    let latest: I2vLongRow[] = rows;

    await runner.start({
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      template: template({ prompt: "__MOCK_CANCEL__" }),
      settings: SETTINGS,
      rows,
      onRowsChanged: (r) => {
        latest = r;
      },
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    expect(latest[0]).toMatchObject({ stat: "Waiting", error: "" });
  });

  // A2Vとの意図的な差（正本§1-7）: 中止はDELETEを投げず、生成中の1枚は完走して
  // Doneとして保存される。以降の行だけが投入されない。
  it("stop()はDELETEを投げず、生成中の行は完走してDone保存・以降の行は未処理のまま", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 3, fs });
    const { wrapped, calls } = spyBridge(bridge);
    const runner = new BatchI2vLongRunner(wrapped);
    const rows: I2vLongRow[] = [makeRow(1, "first.png", "Waiting"), makeRow(2, "second.png", "Waiting")];
    let latest: I2vLongRow[] = rows;

    const startPromise = runner.start({
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      template: template(),
      settings: SETTINGS,
      rows,
      onRowsChanged: (r) => {
        latest = r;
      },
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    // 1行目のポーリングが始まった＝ジョブが実際に走っているタイミングで中止。
    await waitUntil(() =>
      calls.some(
        (c) =>
          c.method === "backend.request" &&
          (c.params as { method?: string }).method === "GET" &&
          ((c.params as { path?: string }).path ?? "").startsWith("/api/v1/jobs/"),
      ),
    );
    runner.stop();
    expect(runner.state).toBe("stopping");

    const result = await startPromise;
    expect(result.started).toBe(true);
    expect(runner.state).toBe("idle");

    // 生成中だった1枚は完走してDone、ファイルも保存済み。
    expect(latest[0]).toMatchObject({ stat: "Done", output: "first.mp4", error: "" });
    expect(fs.folders.get(OUT_DIR)?.some((e) => e.name === "first.mp4")).toBe(true);
    // 2行目は投入されず、Waitingのまま（再開でそのまま処理できる）。
    expect(latest[1]).toMatchObject({ stat: "Waiting", output: "" });
    expect(JSON.stringify(calls)).not.toContain("second.png");
    // 最重要: DELETE /jobs/{id} は一度も投げていない。
    expect(deleteCalls(calls)).toHaveLength(0);
  }, 10_000);

  it("同じ画像パスの行が複数あっても、アップロードは1回だけ（実行中キャッシュ）", async () => {
    const fs = createMockFs();
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
    const { wrapped, calls } = spyBridge(bridge);
    const runner = new BatchI2vLongRunner(wrapped);
    const rows: I2vLongRow[] = [makeRow(1, "same.png", "Waiting"), makeRow(2, "same.png", "Waiting")];

    await runner.start({
      imgDir: IMG_DIR,
      outDir: OUT_DIR,
      template: template(),
      settings: SETTINGS,
      rows,
      pollIntervalMs: POLL_MS,
      jobBusyBackoffMs: BACKOFF_MS,
    });

    const imageUploads = calls.filter(
      (c) => c.method === "backend.uploadFile" && (c.params as { kind: string }).kind === "image",
    );
    expect(imageUploads).toHaveLength(1);
  });

  describe("送信ボディ", () => {
    it("clipsは2本以上で、行の画像はclip0のみ・clip1以降には画像が付かない", async () => {
      const fs = createMockFs();
      const base = createMockBridge({ delayMs: 0, fs });
      const { wrapped, chainBodies } = captureChainBridge(base);
      const runner = new BatchI2vLongRunner(wrapped);

      await runner.start({
        imgDir: IMG_DIR,
        outDir: OUT_DIR,
        template: template(),
        settings: SETTINGS,
        rows: [makeRow(1, "hero.png", "Waiting")],
      });

      expect(chainBodies).toHaveLength(1);
      const body = chainBodies[0]!;
      const clips = body.clips as Array<Record<string, unknown>>;
      expect(clips.length).toBeGreaterThanOrEqual(2);

      const conditioning = clips[0]!.conditioning_images as Array<Record<string, unknown>>;
      expect(conditioning).toHaveLength(1);
      // モックの image_id（`mock-image-N`）がclip0に乗っている。
      expect(String(conditioning[0]!.image_id)).toMatch(/^mock-image-\d+$/);
      expect(conditioning[0]!.frame_idx).toBe(0);
      expect(conditioning[0]!.strength).toBe(1.0);

      for (const clip of clips.slice(1)) {
        expect(clip).not.toHaveProperty("conditioning_images");
      }
    });

    it("テンプレート由来のフィールドはすべて素通しで送られる", async () => {
      const fs = createMockFs();
      const base = createMockBridge({ delayMs: 0, fs });
      const { wrapped, chainBodies } = captureChainBridge(base);
      const runner = new BatchI2vLongRunner(wrapped);
      const tpl = template({ negative_prompt: "worst quality", nag_enabled: true, nag_scale: 5, neg_method: "vsf", vsf_scale: 1.5 });

      await runner.start({
        imgDir: IMG_DIR,
        outDir: OUT_DIR,
        template: tpl,
        settings: SETTINGS,
        rows: [makeRow(1, "hero.png", "Waiting")],
      });

      const body = chainBodies[0]!;
      expect(body.prompt).toBe(tpl.prompt);
      expect(body.width).toBe(768);
      expect(body.height).toBe(512);
      expect(body.frame_rate).toBe(24);
      expect(body.seed).toBe(42);
      expect(body.overlap_frames).toBe(3);
      expect(body.overlap_strength).toBe(0.5);
      expect(body.chunked_upsample).toBe(true);
      expect(body.loras).toEqual([{ name: "style", strength: 1.0 }]);
      expect(body.negative_prompt).toBe("worst quality");
      expect(body.nag_enabled).toBe(true);
      expect(body.nag_scale).toBe(5);
      expect(body.neg_method).toBe("vsf");
      expect(body.vsf_scale).toBe(1.5);
      // clip個別のプロンプト/フレーム数も手つかずで通る。
      const clips = body.clips as Array<Record<string, unknown>>;
      expect(clips[1]).toMatchObject({ num_frames: 121, prompt: "clip 1 override" });
    });

    // 2026-07-30 行ごとプロンプト: テンプレートのpromptは共通プロンプトのまま、
    // 合成はランナーが行ごとに行う。
    it("行ごとのプロンプトがadd/replaceに従って合成され、行ごとに別々の本文が送られる", async () => {
      const fs = createMockFs();
      const base = createMockBridge({ delayMs: 0, fs });
      const { wrapped, chainBodies } = captureChainBridge(base);
      const runner = new BatchI2vLongRunner(wrapped);

      await runner.start({
        imgDir: IMG_DIR,
        outDir: OUT_DIR,
        template: template(),
        settings: { promptMode: "add" },
        rows: [
          makeRow(1, "a.png", "Waiting", { prompt: "in the rain" }),
          makeRow(2, "b.png", "Waiting", { prompt: "on the beach" }),
          makeRow(3, "c.png", "Waiting"),
        ],
      });

      expect(chainBodies.map((b) => b.prompt)).toEqual([
        "a cat walking in the rain",
        "a cat walking on the beach",
        // 行プロンプトが空の行は共通プロンプトそのまま。
        "a cat walking",
      ]);
    });

    it("replaceモードでは行プロンプトが共通プロンプトを丸ごと置き換える（空の行は除く）", async () => {
      const fs = createMockFs();
      const base = createMockBridge({ delayMs: 0, fs });
      const { wrapped, chainBodies } = captureChainBridge(base);
      const runner = new BatchI2vLongRunner(wrapped);

      await runner.start({
        imgDir: IMG_DIR,
        outDir: OUT_DIR,
        template: template(),
        settings: { promptMode: "replace" },
        rows: [makeRow(1, "a.png", "Waiting", { prompt: "a dog running" }), makeRow(2, "b.png", "Waiting")],
      });

      expect(chainBodies.map((b) => b.prompt)).toEqual(["a dog running", "a cat walking"]);
    });

    it("行プロンプトはclip個別プロンプトを書き換えない（案A）", async () => {
      const fs = createMockFs();
      const base = createMockBridge({ delayMs: 0, fs });
      const { wrapped, chainBodies } = captureChainBridge(base);
      const runner = new BatchI2vLongRunner(wrapped);

      await runner.start({
        imgDir: IMG_DIR,
        outDir: OUT_DIR,
        template: template(),
        settings: { promptMode: "replace" },
        rows: [makeRow(1, "a.png", "Waiting", { prompt: "a dog running" })],
      });

      const clips = chainBodies[0]!.clips as Array<Record<string, unknown>>;
      expect(clips[1]).toMatchObject({ prompt: "clip 1 override" });
    });

    it("テンプレートは行ごとに複製され、共有参照(loras/clips)は書き換えられない", async () => {
      const fs = createMockFs();
      const base = createMockBridge({ delayMs: 0, fs });
      const { wrapped, chainBodies } = captureChainBridge(base);
      const runner = new BatchI2vLongRunner(wrapped);
      const tpl = template();
      const lorasRef = tpl.loras;
      const clip1Ref = tpl.clips[1];

      await runner.start({
        imgDir: IMG_DIR,
        outDir: OUT_DIR,
        template: tpl,
        settings: SETTINGS,
        rows: [makeRow(1, "a.png", "Waiting"), makeRow(2, "b.png", "Waiting")],
      });

      expect(chainBodies).toHaveLength(2);
      // テンプレート自身は無傷（clip0に画像が焼き付いていない）。
      expect(tpl.clips[0]).not.toHaveProperty("conditioning_images");
      expect(tpl.loras).toBe(lorasRef);
      expect(tpl.clips[1]).toBe(clip1Ref);
      // 2行の画像は互いに独立。
      const idOf = (body: Record<string, unknown>): string => {
        const clips = body.clips as Array<{ conditioning_images?: Array<{ image_id: string }> }>;
        return clips[0]?.conditioning_images?.[0]?.image_id ?? "";
      };
      expect(idOf(chainBodies[0]!)).not.toBe(idOf(chainBodies[1]!));
    });
  });
});
