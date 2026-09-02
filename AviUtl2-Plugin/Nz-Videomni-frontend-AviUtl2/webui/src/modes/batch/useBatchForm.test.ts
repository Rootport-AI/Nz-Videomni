import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import type { MockFsFileEntry } from "../../bridge/mockBridge";
import { BridgeError } from "../../bridge";
import type { BridgeMethod, NativeBridge, ParamsOf, ResultOf } from "../../bridge";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import { suggestFramesForAudio } from "../chained/chainUtils";
import { useKeyframes } from "../single/useKeyframes";
import type { NagSettings } from "../../shell/nagSettings";
import { ACCELERATION_DEFAULTS } from "../../shell/accelerationSettings";
import type { AccelerationSettings } from "../../shell/accelerationSettings";
import {
  RUN_LOCK_OWNER_BATCH_A2V,
  RUN_LOCK_OWNER_BATCH_I2V_LONG,
  __resetRunLockForTests,
  acquireRunLock,
  getRunLockOwner,
  isRunLockHeld,
  releaseRunLock,
} from "../../shell/runLock";
import { __resetBatchRuntimeForTests } from "./runtime";
import { useBatchForm } from "./useBatchForm";
import type { BatchGenerationValues } from "./useBatchForm";

const WAV_DIR = "C:\\voice\\ep01";
const AUTO_OUT_DIR = "C:\\voice\\ep01_a2v_out";

// U4: width/height/frameRate/seed/numFrames are now owned by the Create form
// and passed in. On-grid defaults (512/320 are multiples of 64) so
// `resolutionValid` holds and the guard tests can flip individual values
// off-grid deliberately. `numFrames: 481` keeps the DURATION Skip cap at the
// server hard cap (the pre-DURATION-wiring semantics), so every existing
// over-cap/fps regression test still trips on the 481-frame boundary exactly
// as before (adversarial review M4).
const GEN_VALUES: BatchGenerationValues = { width: 512, height: 320, frameRate: 24, seed: -1, numFrames: 481 };

function wavFolder(entries: MockFsFileEntry[]) {
  return createMockFs({ folders: { [WAV_DIR]: entries } });
}

/**
 * Batch A2V Shared spec (2026-07-18): `useBatchForm` no longer owns its own
 * `useKeyframes` — the Create screen's shared instance is injected. These tests
 * wire a real `useKeyframes` (against the same bridge) INSIDE the rendered hook
 * and pass it in, exactly like `SingleScreen`/`BatchSection` do in production.
 * `useBatchForm` re-exports it as a pass-through (`result.current.keyframes`),
 * so the existing tests that drive `result.current.keyframes.*` keep working
 * against the very instance the form reads its `conditioningImages` from.
 */
function renderBatchForm(
  bridge: NativeBridge,
  opts: {
    prompt?: string;
    gen?: BatchGenerationValues;
    nag?: NagSettings;
    acceleration?: AccelerationSettings;
    serverBusy?: boolean;
  } = {},
) {
  const prompt = opts.prompt ?? "a prompt";
  const gen = opts.gen ?? GEN_VALUES;
  return renderHook(() => {
    const keyframes = useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: bridge });
    return useBatchForm(FALLBACK_APP_CONFIG, prompt, gen, keyframes, {
      nativeBridge: bridge,
      nag: opts.nag,
      acceleration: opts.acceleration,
      ...(opts.serverBusy !== undefined ? { serverBusy: opts.serverBusy } : {}),
    });
  });
}

/** Wraps a bridge to capture (and short-circuit) every `POST /generate/chain`:
 * records the request body and returns a 500 so `BatchRunner.processRow`'s
 * try/catch marks the row `Failed` and the run ends immediately — no 1s job
 * polling, no dangling timers. The captured body's `clips[0].conditioning_images`
 * is exactly what a `Shared` row resolved to at `start()` time, which is what
 * the Shared-spec tests assert on. */
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

/** Pulls the single clip's `conditioning_images` out of a captured
 * `/generate/chain` body (an empty array when the clip omitted the key). */
function conditioningOf(body: Record<string, unknown>): Array<Record<string, unknown>> {
  const clips = body.clips as Array<{ conditioning_images?: Array<Record<string, unknown>> }> | undefined;
  return clips?.[0]?.conditioning_images ?? [];
}

/** Pulls the single clip's `num_frames` out of a captured `/generate/chain`
 * body — the frame count a row was submitted with (used by the M-1 start-time
 * re-judgment tests to prove a row was recomputed at the CURRENT fps). */
function framesOf(body: Record<string, unknown>): number {
  const clips = body.clips as Array<{ num_frames: number }> | undefined;
  return clips?.[0]?.num_frames ?? 0;
}

// A 25s clip exceeds the 481-frame cap at 24fps (raw ~593f) but is well under
// it at 12fps — the fixture the start-time re-judgment tests lean on to flip a
// row between Skip(over-cap) and Waiting purely by changing the Create fps.
const OVER_CAP_DURATION_SEC = 25.0;

describe("useBatchForm", () => {
  // §1-7 相互ロック: `shell/runLock.ts` is module-level state, so a test whose
  // run is still in flight when it ends would leak a held lock into the next
  // test (`canStart` would be false for no visible reason). Clear it up front.
  //
  // §3-47（2026-09-02）: ランナー実体と走行中の行も`runtime.ts`のモジュール
  // レベル・シングルトンになったので、同じ理由でこちらもリセットする。これが
  // 無いと、前のテストの走行状態が残ったまま次のテストがマウントされ、
  // 「走行中なら復元する」初期化子が前のテストのフォルダと行を復元してしまう。
  // ※これはテスト間の後始末であって、走り残ったランナー自体は止まらない
  //   （止められない）——実際に走らせるテストは末尾でidleまで待ち切ること。
  beforeEach(() => {
    __resetBatchRuntimeForTests();
    __resetRunLockForTests();
  });

  it("pickWavDir sets wavDir and auto-derives outDir as a sibling '_a2v_out' folder", async () => {
    const fs = wavFolder([]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result } = renderBatchForm(bridge);

    await act(async () => {
      await result.current.pickWavDir();
    });

    expect(result.current.wavDir).toBe(WAV_DIR);
    expect(result.current.outDir).toBe(AUTO_OUT_DIR);
    expect(result.current.outDirIsAuto).toBe(true);
  });

  it("pickOutDir overrides the auto-derived output folder and disables further auto-derivation", async () => {
    const fs = wavFolder([]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result } = renderBatchForm(bridge);

    await act(async () => {
      await result.current.pickWavDir();
    });
    expect(result.current.outDir).toBe(AUTO_OUT_DIR);

    const bridge2 = createMockBridge({ delayMs: 0, fs, pickFolderPath: "C:\\custom\\out" });
    const { result: result2 } = renderBatchForm(bridge2);
    await act(async () => {
      await result2.current.pickWavDir();
    });
    await act(async () => {
      await result2.current.pickOutDir();
    });
    expect(result2.current.outDir).toBe("C:\\custom\\out");
    expect(result2.current.outDirIsAuto).toBe(false);

    // A second wav-folder pick must NOT clobber the user's manual choice.
    await act(async () => {
      await result2.current.pickWavDir();
    });
    expect(result2.current.outDir).toBe("C:\\custom\\out");
  });

  it("scan(): a fresh folder populates in-memory rows (mtime order, all Waiting) — no CSV is written", async () => {
    const fs = wavFolder([
      { name: "b.wav", sizeBytes: 100, mtimeMs: 2000, durationSec: 2.0 },
      { name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 },
    ]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result } = renderBatchForm(bridge);

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });

    expect(result.current.scanError).toBeNull();
    expect(result.current.rows.map((r) => r.wav)).toEqual(["a.wav", "b.wav"]); // mtime order
    expect(result.current.rows.every((r) => r.stat === "Waiting")).toBe(true);
    // Stateless batch (owner decision 2026-07-18): scanning never writes a
    // manifest CSV (or anything else) to disk.
    expect(fs.files.size).toBe(0);
  });

  it("scan() is stateless: a re-scan regenerates every row from scratch, discarding in-memory edits", async () => {
    const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result } = renderBatchForm(bridge);

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });
    expect(result.current.rows[0]).toMatchObject({ wav: "a.wav", stat: "Waiting", prompt: "" });

    // Edit a row in memory, then re-scan: the edit is gone (nothing persists).
    act(() => {
      result.current.setRowPromptLocal(0, "an edit that should not survive a re-scan");
    });
    expect(result.current.rows[0]?.prompt).toBe("an edit that should not survive a re-scan");

    await act(async () => {
      await result.current.scan();
    });
    expect(result.current.rows[0]).toMatchObject({ wav: "a.wav", stat: "Waiting", prompt: "" });
  });

  it("resetRowToWaiting rewinds an eligible (Skip) row to Waiting, in memory", async () => {
    // A non-wav file scans to a Skip row (wav-only-alpha) — Skip is one of the
    // reset-eligible stats, so this exercises the reset path without needing a
    // full run to produce a Done row (rows are in-memory only now).
    const fs = wavFolder([{ name: "note.txt", sizeBytes: 100, mtimeMs: 1000 }, { name: "clip.mp3", sizeBytes: 100, mtimeMs: 2000, durationSec: 3.0 }]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result } = renderBatchForm(bridge);

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });
    // .txt is not an allowed audio ext (excluded); .mp3 is allowed but non-wav
    // -> Skip.
    expect(result.current.rows.map((r) => [r.wav, r.stat])).toEqual([["clip.mp3", "Skip"]]);

    act(() => {
      result.current.resetRowToWaiting(1);
    });
    expect(result.current.rows[0]).toMatchObject({ wav: "clip.mp3", stat: "Waiting", output: "", error: "" });
  });

  it("resetRowToWaiting is a no-op for a Waiting/Generating row", async () => {
    const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result } = renderBatchForm(bridge);

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });
    expect(result.current.rows[0]?.stat).toBe("Waiting");

    await act(async () => {
      await result.current.resetRowToWaiting(1);
    });
    expect(result.current.rows[0]?.stat).toBe("Waiting");
  });

  // M-1 start-time re-judgment (the original over-cap bug) -------------------

  it("regression (main bug): re-arming a Skip(over-cap) row and starting at the SAME fps re-judges it back to Skip and sends nothing", async () => {
    const fs = wavFolder([{ name: "long.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: OVER_CAP_DURATION_SEC }]);
    const base = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { wrapped, chainBodies } = captureChainBridge(base);
    const { result } = renderBatchForm(wrapped);

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });
    // A 25s clip at 24fps blows past the 481-frame cap -> Skip(over-cap).
    expect(result.current.rows[0]).toMatchObject({ wav: "long.wav", stat: "Skip", skipReason: "over-cap", frames: 0 });

    // Manually re-arm it to Waiting (the 🔁 row action).
    act(() => {
      result.current.resetRowToWaiting(1);
    });
    expect(result.current.rows[0]?.stat).toBe("Waiting");

    // Start at the SAME fps: the start-time re-judgment (rejudgeRows in
    // `start()`) must send it straight back to Skip(over-cap) and emit no
    // /generate/chain at all. Passing the closure's stale `rows` here (the
    // original bug) would have submitted the over-cap row and 422'd.
    act(() => {
      result.current.start();
    });

    expect(chainBodies).toHaveLength(0);
    expect(result.current.rows[0]).toMatchObject({ stat: "Skip", skipReason: "over-cap", frames: 0 });
    expect(result.current.runnerState).toBe("idle");
  });

  it("M-1: lowering the fps no longer blocks canStart, and re-arming a Skip(over-cap) row starts it with frames recomputed at the current fps", async () => {
    const fs = wavFolder([
      { name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }, // short -> Waiting
      { name: "long.wav", sizeBytes: 100, mtimeMs: 2000, durationSec: OVER_CAP_DURATION_SEC }, // -> Skip(over-cap) @24fps
    ]);
    const base = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { wrapped, chainBodies } = captureChainBridge(base);
    const { result, rerender } = renderHook(
      ({ gen }: { gen: BatchGenerationValues }) => {
        const keyframes = useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: wrapped });
        return useBatchForm(FALLBACK_APP_CONFIG, "a prompt", gen, keyframes, { nativeBridge: wrapped });
      },
      { initialProps: { gen: GEN_VALUES } },
    );

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });
    expect(result.current.rows.map((r) => [r.wav, r.stat])).toEqual([
      ["a.wav", "Waiting"],
      ["long.wav", "Skip"],
    ]);

    // Supply the shared keyframe so a Shared runnable row can start; only the
    // fps behavior is under test here.
    act(() => {
      void result.current.keyframes.addFromCapture();
    });
    await waitFor(() => expect(result.current.keyframes.items[0]?.status).toBe("ready"));
    expect(result.current.canStart).toBe(true);

    // Lower the Create-form fps AFTER the scan baked the old value into rows.
    // Under M-1 this must NOT block canStart (the pre-M-1 `fpsMismatch` guard
    // did); the flag is now informational only and the rows are re-judged at
    // start time instead.
    rerender({ gen: { ...GEN_VALUES, frameRate: 12 } });
    expect(result.current.fpsMismatch).toBe(true);
    expect(result.current.canStart).toBe(true);

    // Re-arm the over-cap Skip row (valid again at the lower fps) and start.
    act(() => {
      result.current.resetRowToWaiting(2);
    });
    act(() => {
      result.current.start();
    });

    // Both rows submit; the long row's num_frames is the value recomputed at
    // the CURRENT (12) fps, not the stale 24-fps scan value.
    await waitFor(() => expect(chainBodies).toHaveLength(2));
    const expectedFrames = suggestFramesForAudio(OVER_CAP_DURATION_SEC, 12);
    expect(chainBodies.map(framesOf)).toContain(expectedFrames);
    // §3-47: ランナーは`runtime.ts`のシングルトンなので、走らせたら走り切るまで
    // 待つ（走り残しが次のテストのスナップショットへ書き込むのを防ぐ）。
    await waitFor(() => expect(result.current.runnerState).toBe("idle"));
  });

  it("all runnable rows fall to Skip on re-judge: rows reflect the re-judged state, runner stays idle (0 targets)", async () => {
    const fs = wavFolder([{ name: "long.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: OVER_CAP_DURATION_SEC }]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result } = renderBatchForm(bridge);

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });
    act(() => {
      result.current.resetRowToWaiting(1);
    });
    expect(result.current.rows[0]?.stat).toBe("Waiting");

    act(() => {
      result.current.start();
    });

    // Re-judged back to Skip(over-cap); nothing runnable, so the runner never
    // leaves idle — but the row list still reflects the re-judgment.
    expect(result.current.rows[0]).toMatchObject({ stat: "Skip", skipReason: "over-cap", frames: 0 });
    expect(result.current.runnerState).toBe("idle");
  });

  it("re-arming a wav-only-alpha Skip row always re-judges back to Skip on start and sends nothing", async () => {
    const fs = wavFolder([{ name: "clip.mp3", sizeBytes: 100, mtimeMs: 1000, durationSec: 3.0 }]);
    const base = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { wrapped, chainBodies } = captureChainBridge(base);
    const { result } = renderBatchForm(wrapped);

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });
    // A non-wav audio file is always a wav-only-alpha Skip (no length probing).
    expect(result.current.rows[0]).toMatchObject({ wav: "clip.mp3", stat: "Skip", skipReason: "wav-only-alpha" });

    act(() => {
      result.current.resetRowToWaiting(1);
    });
    expect(result.current.rows[0]?.stat).toBe("Waiting");

    // rejudgeRows forces any duration<=0 row (a wav-only-alpha carries
    // duration 0) straight back to Skip, so it can never submit a headless job.
    act(() => {
      result.current.start();
    });

    expect(chainBodies).toHaveLength(0);
    expect(result.current.rows[0]).toMatchObject({ stat: "Skip", skipReason: "wav-only-alpha", frames: 0 });
  });

  it("summary/canStart reflect the current rows and runner idle state", async () => {
    const fs = wavFolder([
      { name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 },
      { name: "b.wav", sizeBytes: 100, mtimeMs: 2000, durationSec: 1.0 },
    ]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result } = renderBatchForm(bridge);

    expect(result.current.canStart).toBe(false); // no wavDir/outDir yet

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });

    expect(result.current.summary).toMatchObject({ total: 2, waiting: 2, done: 0, failed: 0, skip: 0 });
    // Batch A2V Shared spec (2026-07-18): every fresh-scanned row defaults its
    // `image` column to the `Shared` sentinel (`manifestMerge.baseRow`) — so
    // canStart stays blocked until the Create-owned KEYFRAMES panel holds at
    // least one READY image (its slider position no longer matters).
    expect(result.current.sharedKeyframeMissing).toBe(true);
    expect(result.current.canStart).toBe(false);
    expect(result.current.currentRow).toBeNull();

    act(() => {
      void result.current.keyframes.addFromCapture();
    });
    await waitFor(() => expect(result.current.keyframes.items[0]?.status).toBe("ready"));

    expect(result.current.sharedKeyframeMissing).toBe(false);
    expect(result.current.canStart).toBe(true);
  });

  it("Shared spec (2026-07-18): a ready image at ANY slider position clears the start gate", async () => {
    const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result } = renderBatchForm(bridge);

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });
    expect(result.current.sharedKeyframeMissing).toBe(true);
    expect(result.current.canStart).toBe(false);

    // Add a ready keyframe, then park it well away from frame 0 — the old gate
    // required a `frame_idx===0` entry; the new one only needs a ready image.
    act(() => {
      void result.current.keyframes.addFromCapture();
    });
    await waitFor(() => expect(result.current.keyframes.items[0]?.status).toBe("ready"));
    act(() => {
      const id = result.current.keyframes.items[0]!.id;
      result.current.keyframes.setFrameIdx(id, 24);
    });
    expect(result.current.keyframes.items[0]?.frameIdx).toBe(24);
    expect(result.current.keyframes.conditioningImages.some((c) => c.frame_idx === 0)).toBe(false);

    // Still eligible to start — no frame-0 keyframe required anymore.
    expect(result.current.sharedKeyframeMissing).toBe(false);
    expect(result.current.canStart).toBe(true);
  });

  it("Shared spec (2026-07-18): a Shared row sends ONLY the first (lowest-frame_idx) ready image, force-pinned to frame 0 with its strength preserved; later keyframes are ignored", async () => {
    const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
    const base = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { wrapped, chainBodies } = captureChainBridge(base);
    const { result } = renderBatchForm(wrapped);

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });
    expect(result.current.rows[0]?.image).toBe("Shared");

    // Two ready keyframes. The FIRST captured (image_id mock-image-1) is parked
    // at frame_idx 40; the SECOND (image_id mock-image-2) at frame_idx 16 — so
    // sort order (not insertion order) makes mock-image-2 the leading image.
    act(() => {
      void result.current.keyframes.addFromCapture();
    });
    await waitFor(() => expect(result.current.keyframes.items[0]?.status).toBe("ready"));
    const firstId = result.current.keyframes.items[0]!.id;
    act(() => {
      result.current.keyframes.setFrameIdx(firstId, 40);
      result.current.keyframes.setStrength(firstId, 0.9);
    });

    act(() => {
      void result.current.keyframes.addFromCapture();
    });
    await waitFor(() => expect(result.current.keyframes.items.filter((i) => i.status === "ready")).toHaveLength(2));
    const secondId = result.current.keyframes.items.map((i) => i.id).find((id) => id !== firstId)!;
    act(() => {
      result.current.keyframes.setFrameIdx(secondId, 16);
      result.current.keyframes.setStrength(secondId, 0.35);
    });

    // Sanity: two ready images, one at 40 (mock-image-1), one at 16 (mock-image-2).
    const ready = result.current.keyframes.conditioningImages;
    expect(ready).toHaveLength(2);
    expect(new Set(ready.map((c) => c.image_id))).toEqual(new Set(["mock-image-1", "mock-image-2"]));

    expect(result.current.canStart).toBe(true);
    act(() => {
      result.current.start();
    });

    await waitFor(() => expect(chainBodies).toHaveLength(1));
    const conditioning = conditioningOf(chainBodies[0]!);
    // Exactly one keyframe: the lowest-frame_idx image (mock-image-2), forced to
    // frame_idx 0, strength preserved (0.35). mock-image-1 (frame_idx 40) is
    // dropped entirely — no error, no warning.
    expect(conditioning).toEqual([{ image_id: "mock-image-2", frame_idx: 0, strength: 0.35 }]);
    // §3-47: 走らせたテストは走り切るまで待つ（シングルトンへの走り残り防止）。
    await waitFor(() => expect(result.current.runnerState).toBe("idle"));
  });

  it("Shared spec (2026-07-18): with no ready image the run is blocked (canStart false); a Shared run therefore never sends a headless job", async () => {
    const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
    const base = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { wrapped, chainBodies } = captureChainBridge(base);
    const { result } = renderBatchForm(wrapped);

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });

    // No keyframes at all -> the Shared row's start gate is closed.
    expect(result.current.keyframes.conditioningImages).toHaveLength(0);
    expect(result.current.sharedKeyframeMissing).toBe(true);
    expect(result.current.canStart).toBe(false);

    // Even if `start()` is invoked, `BatchSection` gates it on `canStart`; here
    // we assert the underlying invariant that no /generate/chain is emitted
    // synchronously — the submit is several awaits away, and the UI gate is
    // what actually keeps a headless job from ever being queued.
    act(() => {
      result.current.start();
    });
    expect(chainBodies).toHaveLength(0);
    // §3-47: この呼び出しは（`canStart`を迂回しているので）実際に走り出す。
    // シングルトンへの走り残りを残さないよう、ここで走り切らせる。
    await waitFor(() => expect(result.current.runnerState).toBe("idle"));
  });

  it("sharedKeyframeMissing only inspects runnable rows: a non-runnable (Skip) Shared row doesn't block when every still-to-run row has its own image", async () => {
    // Parity with Gradio `batch.py` `_validate()`, which only inspects its
    // `targets` (unfinished rows). Here a non-wav file scans to a Skip/Shared
    // row (excluded from runnable rows, exactly like a Done row would be), and
    // a real .wav scans to a Waiting/Shared row that we then reassign its own
    // image — so no shared keyframe is required and the run can start, even
    // with the KEYFRAMES panel empty.
    const fs = wavFolder([
      { name: "skipme.mp3", sizeBytes: 100, mtimeMs: 1000, durationSec: 3.0 },
      { name: "b.wav", sizeBytes: 100, mtimeMs: 2000, durationSec: 1.0 },
    ]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result } = renderBatchForm(bridge);

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });
    // skipme.mp3 -> Skip/Shared (non-wav); b.wav -> Waiting/Shared.
    expect(result.current.rows.map((r) => [r.wav, r.stat, r.image])).toEqual([
      ["skipme.mp3", "Skip", "Shared"],
      ["b.wav", "Waiting", "Shared"],
    ]);

    // Give the only runnable row (b.wav) its own image.
    act(() => {
      result.current.updateRowImage(1, "b.png");
    });
    expect(result.current.rows[1]).toMatchObject({ wav: "b.wav", stat: "Waiting", image: "b.png" });

    // The only runnable row (b.wav) supplies its own image — no shared
    // keyframe needed even though a Skip Shared row is still present, and the
    // KEYFRAMES panel is empty.
    expect(result.current.keyframes.conditioningImages).toHaveLength(0);
    expect(result.current.sharedKeyframeMissing).toBe(false);
    expect(result.current.canStart).toBe(true);
  });

  // N5 "バッチ表の行内編集" ---------------------------------------------------

  it("setRowPromptLocal edits only the target row's prompt (by array index), in memory", async () => {
    const fs = wavFolder([
      { name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 },
      { name: "b.wav", sizeBytes: 100, mtimeMs: 2000, durationSec: 1.0 },
    ]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result } = renderBatchForm(bridge);

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });
    expect(result.current.rows.map((r) => r.wav)).toEqual(["a.wav", "b.wav"]);

    // onChange-equivalent (controlled input): updates only the target row's
    // in-memory prompt immediately. Nothing is persisted (stateless batch).
    act(() => {
      result.current.setRowPromptLocal(1, "row b's own prompt");
    });
    expect(result.current.rows[0]).toMatchObject({ wav: "a.wav", prompt: "" });
    expect(result.current.rows[1]).toMatchObject({ wav: "b.wav", prompt: "row b's own prompt" });
    expect(fs.files.size).toBe(0);
  });

  it("copyCommonPromptToRow overwrites a row's prompt with the common prompt verbatim, never through parseLoraPrompt", async () => {
    const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    // Deliberately includes a <lora:> tag: if this were routed through
    // `parseLoraPrompt` (like `start()`'s own prompt does), the tag would be
    // stripped out of the stored prompt — it must survive untouched here.
    const commonPrompt = "<lora:my-style:0.8> a shared common prompt";
    const { result } = renderBatchForm(bridge, { prompt: commonPrompt });

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });
    expect(result.current.rows[0]?.prompt).toBe("");

    act(() => {
      result.current.copyCommonPromptToRow(0, commonPrompt);
    });

    expect(result.current.rows[0]?.prompt).toBe(commonPrompt);
  });

  it("updateRowImage normalizes an empty selection to Shared and no-ops when the value doesn't change", async () => {
    const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result } = renderBatchForm(bridge);

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });
    expect(result.current.rows[0]?.image).toBe("Shared"); // fresh-scan default (manifestMerge's baseRow)

    // Re-selecting the row's own current value is a no-op (same reference).
    const before = result.current.rows;
    act(() => {
      result.current.updateRowImage(0, "Shared");
    });
    expect(result.current.rows).toBe(before);

    // A genuine change updates the in-memory row.
    act(() => {
      result.current.updateRowImage(0, "cat.png");
    });
    expect(result.current.rows[0]?.image).toBe("cat.png");

    // An empty string (e.g. a blank <select> commit) normalizes to Shared.
    act(() => {
      result.current.updateRowImage(0, "");
    });
    expect(result.current.rows[0]?.image).toBe("Shared");
  });

  it("scan() fetches the image folder's file list once and sorts it into imageOptions, leading with Shared", async () => {
    const fs = wavFolder([
      { name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 },
      { name: "z.png", sizeBytes: 10, mtimeMs: 1 },
      { name: "a.png", sizeBytes: 10, mtimeMs: 1 },
      { name: "m.jpg", sizeBytes: 10, mtimeMs: 1 },
    ]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result } = renderBatchForm(bridge);

    expect(result.current.imageOptions).toEqual(["Shared"]); // no imgDir yet

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.pickImgDir();
    });
    expect(result.current.imgDir).toBe(WAV_DIR);
    expect(result.current.imageOptions).toEqual(["Shared"]); // not fetched until scan()

    await act(async () => {
      await result.current.scan();
    });

    // The image files never leak into the audio-row scan (extension allowlist).
    expect(result.current.rows.map((r) => r.wav)).toEqual(["a.wav"]);
    expect(result.current.imageOptions).toEqual(["Shared", "a.png", "m.jpg", "z.png"]);
  });

  it("switching the image folder (pickImgDir) immediately collapses imageOptions to just Shared, discarding the previous folder's stale file names (M1 remediation)", async () => {
    const IMG_DIR_2 = "C:\\voice\\ep01-images-v2";
    const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
    fs.folders.set(WAV_DIR, [
      { name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 },
      { name: "old-cat.png", sizeBytes: 10, mtimeMs: 1 },
    ]);
    fs.folders.set(IMG_DIR_2, [{ name: "new-cat.png", sizeBytes: 10, mtimeMs: 1 }]);
    // `options` is a plain object the mock reads `pickFolderPath` from on
    // every `ui.pickFolder` call (not snapshotted at bridge-creation time) —
    // mutating it between `pickImgDir()` calls stands in for the user
    // re-opening the folder dialog and choosing a different folder.
    const bridgeOptions = { delayMs: 0, fs, pickFolderPath: WAV_DIR };
    const bridge = createMockBridge(bridgeOptions);
    const { result } = renderBatchForm(bridge);

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.pickImgDir(); // -> WAV_DIR (has old-cat.png)
    });
    await act(async () => {
      await result.current.scan();
    });
    expect(result.current.imageOptions).toEqual(["Shared", "old-cat.png"]);

    // Switch the image folder to IMG_DIR_2 WITHOUT scanning yet: the old
    // folder's file names must be dropped immediately, collapsing the
    // options back to just Shared rather than continuing to offer
    // "old-cat.png" (which doesn't exist in the new folder).
    bridgeOptions.pickFolderPath = IMG_DIR_2;
    await act(async () => {
      await result.current.pickImgDir();
    });
    expect(result.current.imgDir).toBe(IMG_DIR_2);
    expect(result.current.imageOptions).toEqual(["Shared"]);

    // The next scan() repopulates it with the NEW folder's files.
    await act(async () => {
      await result.current.scan();
    });
    expect(result.current.imageOptions).toEqual(["Shared", "new-cat.png"]);
  });

  it("setImgDir('') (blank onBlur commit) clears the image folder, collapsing imageOptions to just Shared and discarding the folder's stale file names", async () => {
    // The dedicated Clear button was removed (2026-07-19): the image folder is
    // now cleared by blanking the hand-typed input, which routes through the
    // same `setImgDir` onBlur path as any other edit. The M1-remediation
    // invariant (stale file names dropped immediately) must still hold.
    const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
    fs.folders.set(WAV_DIR, [
      { name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 },
      { name: "cat.png", sizeBytes: 10, mtimeMs: 1 },
    ]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result } = renderBatchForm(bridge);

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.pickImgDir();
    });
    await act(async () => {
      await result.current.scan();
    });
    expect(result.current.imageOptions).toEqual(["Shared", "cat.png"]);

    await act(async () => {
      await result.current.setImgDir("");
    });
    expect(result.current.imgDir).toBeNull();
    expect(result.current.imageOptions).toEqual(["Shared"]);
  });

  // U4 "手打ちフォルダ欄" (hand-typed folder setters) --------------------------

  it("setWavDir (hand-typed) mirrors pickWavDir: sets wavDir, auto-derives outDir, clears rows", async () => {
    const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result } = renderBatchForm(bridge);

    await act(async () => {
      await result.current.setWavDir(WAV_DIR);
    });
    expect(result.current.wavDir).toBe(WAV_DIR);
    expect(result.current.outDir).toBe(AUTO_OUT_DIR);
    expect(result.current.outDirIsAuto).toBe(true);

    await act(async () => {
      await result.current.scan();
    });
    expect(result.current.rows).toHaveLength(1);

    // Typing a different audio folder clears the old rows and re-derives outDir.
    await act(async () => {
      await result.current.setWavDir("C:\\voice\\ep02");
    });
    expect(result.current.wavDir).toBe("C:\\voice\\ep02");
    expect(result.current.rows).toHaveLength(0);
    expect(result.current.outDir).toBe("C:\\voice\\ep02_a2v_out");
  });

  it("setOutDir (hand-typed) sets a manual output folder, disables auto-derivation, and survives a later wav change", async () => {
    const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result } = renderBatchForm(bridge);

    await act(async () => {
      await result.current.setWavDir(WAV_DIR);
    });
    await act(async () => {
      await result.current.setOutDir("D:\\renders\\out");
    });
    expect(result.current.outDir).toBe("D:\\renders\\out");
    expect(result.current.outDirIsAuto).toBe(false);

    await act(async () => {
      await result.current.setWavDir("C:\\voice\\ep02");
    });
    expect(result.current.outDir).toBe("D:\\renders\\out"); // manual choice preserved
  });

  it("setImgDir (hand-typed) sets the image folder and only populates imageOptions on the next scan", async () => {
    const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
    fs.folders.set(WAV_DIR, [
      { name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 },
      { name: "cat.png", sizeBytes: 10, mtimeMs: 1 },
    ]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result } = renderBatchForm(bridge);

    await act(async () => {
      await result.current.setWavDir(WAV_DIR);
    });
    await act(async () => {
      await result.current.setImgDir(WAV_DIR);
    });
    expect(result.current.imgDir).toBe(WAV_DIR);
    expect(result.current.imageOptions).toEqual(["Shared"]); // not fetched until scan()

    await act(async () => {
      await result.current.scan();
    });
    expect(result.current.imageOptions).toEqual(["Shared", "cat.png"]);
  });

  it("setWavDir surfaces a scan error when the probed folder doesn't exist (onBlur existence probe)", async () => {
    const base = createMockBridge({ delayMs: 0 });
    // A bridge whose fs.listFiles rejects, standing in for native's
    // FOLDER_NOT_FOUND on a hand-typed path that doesn't exist.
    const bridge: NativeBridge = {
      async request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
        if (method === "fs.listFiles") throw new BridgeError("FOLDER_NOT_FOUND", "no such folder");
        return base.request(method, params);
      },
      requestWithFiles: (method, params, files) => base.requestWithFiles(method, params, files),
      on: (event, handler) => base.on(event, handler),
    };
    const { result } = renderBatchForm(bridge);

    await act(async () => {
      await result.current.setWavDir("C:\\does\\not\\exist");
    });
    expect(result.current.scanError).toContain("FOLDER_NOT_FOUND");
  });

  // U4 guards (shared generation values) --------------------------------------

  it("guard 1: canStart is blocked when the shared width/height fall off the 64 grid", async () => {
    const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result, rerender } = renderHook(
      ({ gen }: { gen: BatchGenerationValues }) => {
        const keyframes = useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: bridge });
        return useBatchForm(FALLBACK_APP_CONFIG, "a prompt", gen, keyframes, { nativeBridge: bridge });
      },
      { initialProps: { gen: GEN_VALUES } },
    );

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });
    // Supply the shared keyframe so only the resolution guard can block.
    act(() => {
      void result.current.keyframes.addFromCapture();
    });
    await waitFor(() => expect(result.current.keyframes.items[0]?.status).toBe("ready"));

    expect(result.current.resolutionValid).toBe(true);
    expect(result.current.canStart).toBe(true);

    // The Create form now holds an off-64-grid width (free-typed passthrough).
    rerender({ gen: { ...GEN_VALUES, width: 500 } });
    expect(result.current.resolutionValid).toBe(false);
    expect(result.current.canStart).toBe(false);
  });

  it("guard 2 (M-1 new spec): an fps change flags fpsMismatch but NO LONGER blocks canStart; a re-scan clears the flag", async () => {
    // Pre-M-1 this guard blocked `canStart` until a manual re-scan. M-1 moved
    // the re-judgment into `start()` (rejudgeRows), so a stale fps is now
    // corrected automatically at run time — the flag survives only as a
    // display hint and canStart stays true.
    const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result, rerender } = renderHook(
      ({ gen }: { gen: BatchGenerationValues }) => {
        const keyframes = useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: bridge });
        return useBatchForm(FALLBACK_APP_CONFIG, "a prompt", gen, keyframes, { nativeBridge: bridge });
      },
      { initialProps: { gen: GEN_VALUES } },
    );

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });
    act(() => {
      void result.current.keyframes.addFromCapture();
    });
    await waitFor(() => expect(result.current.keyframes.items[0]?.status).toBe("ready"));

    expect(result.current.fpsMismatch).toBe(false);
    expect(result.current.canStart).toBe(true);

    // Change the Create form's fps after the scan baked the old value into rows.
    rerender({ gen: { ...GEN_VALUES, frameRate: 30 } });
    expect(result.current.fpsMismatch).toBe(true);
    // The key M-1 change: the mismatch is informational and does NOT block start.
    expect(result.current.canStart).toBe(true);

    // Re-scanning at the new fps clears the (now purely cosmetic) flag.
    await act(async () => {
      await result.current.scan();
    });
    expect(result.current.fpsMismatch).toBe(false);
    expect(result.current.canStart).toBe(true);
  });

  // DURATION (numFrames) -> Skip cap wiring ----------------------------------
  // The Create-form DURATION is threaded through as scanToRows/rejudgeRows'
  // `maxFrames`. These tests prove BOTH wiring points (scan side AND start
  // side) actually pass it — a plain "renders" test would pass even if the
  // value were dropped on the floor (adversarial review M5).

  it("DURATION wiring (scan side): scanning with a low numFrames Skips a row whose raw frame count exceeds it as over-cap", async () => {
    // A 5s clip is 113 raw frames at 24fps — comfortably under the 481 server
    // hard cap (so it would scan to Waiting at the default numFrames) but over
    // a DURATION cap of 100. Scanning with numFrames=100 must therefore Skip it
    // as over-cap, which only happens if scan() threads
    // generationValues.numFrames into scanToRows's maxFrames.
    const fs = wavFolder([{ name: "mid.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 5.0 }]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result } = renderBatchForm(bridge, { gen: { ...GEN_VALUES, numFrames: 100 } });

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });

    expect(result.current.rows[0]).toMatchObject({
      wav: "mid.wav",
      stat: "Skip",
      skipReason: "over-cap",
      frames: 0,
    });
  });

  it("DURATION wiring (start side): a row scanned Waiting at numFrames=481 is re-judged to Skip(over-cap) and sends nothing after lowering numFrames to 100", async () => {
    // Same 5s/113-raw-frame clip. It scans to Waiting at numFrames=481, but
    // once the Create form lowers DURATION to 100 the start-time re-judgment
    // (rejudgeRows in start()) must send it back to Skip(over-cap) and emit no
    // /generate/chain at all — which only happens if start() threads the CURRENT
    // generationValues.numFrames into rejudgeRows's maxFrames.
    const fs = wavFolder([{ name: "mid.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 5.0 }]);
    const base = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { wrapped, chainBodies } = captureChainBridge(base);
    const { result, rerender } = renderHook(
      ({ gen }: { gen: BatchGenerationValues }) => {
        const keyframes = useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: wrapped });
        return useBatchForm(FALLBACK_APP_CONFIG, "a prompt", gen, keyframes, { nativeBridge: wrapped });
      },
      { initialProps: { gen: GEN_VALUES } }, // numFrames: 481
    );

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });
    // Under the 481 cap the 5s clip is a normal runnable row.
    expect(result.current.rows[0]).toMatchObject({ wav: "mid.wav", stat: "Waiting" });

    // Lower the Create-form DURATION AFTER the scan baked the old cap in.
    rerender({ gen: { ...GEN_VALUES, numFrames: 100 } });

    act(() => {
      result.current.start();
    });

    // rejudgeRows at the new cap (100) demotes the 113-frame row to Skip; the
    // runner therefore has nothing runnable and emits no /generate/chain.
    expect(result.current.rows[0]).toMatchObject({ stat: "Skip", skipReason: "over-cap", frames: 0 });
    expect(chainBodies).toHaveLength(0);
    expect(result.current.runnerState).toBe("idle");
  });

  it("DURATION wiring: changing numFrames flags fpsMismatch (informational), and a re-scan clears it", async () => {
    // `fpsMismatch`'s semantics were widened to also cover a DURATION drift
    // (field name kept for API stability). A 1s clip stays Waiting at either
    // cap, so this isolates the flag from any over-cap re-judgment.
    const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
    const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
    const { result, rerender } = renderHook(
      ({ gen }: { gen: BatchGenerationValues }) => {
        const keyframes = useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: bridge });
        return useBatchForm(FALLBACK_APP_CONFIG, "a prompt", gen, keyframes, { nativeBridge: bridge });
      },
      { initialProps: { gen: GEN_VALUES } }, // numFrames: 481
    );

    await act(async () => {
      await result.current.pickWavDir();
    });
    await act(async () => {
      await result.current.scan();
    });
    expect(result.current.fpsMismatch).toBe(false);

    // A DURATION change alone (fps unchanged) trips the flag.
    rerender({ gen: { ...GEN_VALUES, numFrames: 100 } });
    expect(result.current.fpsMismatch).toBe(true);

    // Re-scanning at the new DURATION cap clears it (scannedMaxFrames catches up).
    await act(async () => {
      await result.current.scan();
    });
    expect(result.current.fpsMismatch).toBe(false);
  });

  // NAG (2026-07-28): Batch has no accordion of its own — it silently
  // inherits `deps.nag` (the Create-owned shared state).
  describe("deps.nag", () => {
    const ENABLED_NAG: NagSettings = {
      text: "blurry, low quality, distorted, watermark, text",
      enabled: true,
      scale: 11.0,
      tau: 2.5,
      alpha: 0.25,
      method: "nag",
      vsfScale: 1.5,
    };

    it("enabled + empty body: nagInvalid is true and canStart is false", async () => {
      const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
      const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
      const { result } = renderBatchForm(bridge, { nag: { ...ENABLED_NAG, text: "   " } });

      await act(async () => {
        await result.current.pickWavDir();
      });
      await act(async () => {
        await result.current.scan();
      });
      // Clear every OTHER canStart blocker so nagInvalid is isolated.
      act(() => {
        void result.current.keyframes.addFromCapture();
      });
      await waitFor(() => expect(result.current.keyframes.items[0]?.status).toBe("ready"));

      expect(result.current.nagInvalid).toBe(true);
      expect(result.current.canStart).toBe(false);
    });

    it("enabled + non-empty body: nagInvalid is false, and start()'s captured settings carry nag", async () => {
      const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
      const base = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
      const { wrapped, chainBodies } = captureChainBridge(base);
      const { result } = renderBatchForm(wrapped, { nag: ENABLED_NAG });

      await act(async () => {
        await result.current.pickWavDir();
      });
      await act(async () => {
        await result.current.scan();
      });
      act(() => {
        void result.current.keyframes.addFromCapture();
      });
      await waitFor(() => expect(result.current.keyframes.items[0]?.status).toBe("ready"));

      expect(result.current.nagInvalid).toBe(false);
      expect(result.current.canStart).toBe(true);

      act(() => {
        result.current.start();
      });
      await waitFor(() => expect(chainBodies).toHaveLength(1));
      const body = chainBodies[0]!;
      expect(body.negative_prompt).toBe(ENABLED_NAG.text);
      expect(body.nag_enabled).toBe(true);
      expect(body.nag_scale).toBe(11.0);
      expect(body.nag_tau).toBe(2.5);
      expect(body.nag_alpha).toBe(0.25);
      // §3-47: 走らせたテストは走り切るまで待つ（シングルトンへの走り残り防止）。
      await waitFor(() => expect(result.current.runnerState).toBe("idle"));
    });

    // Dependency-array regression guard (adversarial review, 2026-07-28): the
    // `start` useCallback MUST list `nag` in its dependency array — omitting
    // it would let a stale closure fire an overnight batch against an old NAG
    // state. Changes `nag` BETWEEN renders (no re-mount) and asserts the
    // captured request reflects the LATEST value at the moment `start()` is
    // actually invoked.
    it("captures the LATEST nag value at start() time, even after it changed since the last render that didn't call start()", async () => {
      const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
      const base = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
      const { wrapped, chainBodies } = captureChainBridge(base);
      const { result, rerender } = renderHook(
        ({ nag }: { nag: NagSettings }) => {
          const keyframes = useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: wrapped });
          return useBatchForm(FALLBACK_APP_CONFIG, "a prompt", GEN_VALUES, keyframes, { nativeBridge: wrapped, nag });
        },
        { initialProps: { nag: { ...ENABLED_NAG, text: "stale negative prompt" } } },
      );

      await act(async () => {
        await result.current.pickWavDir();
      });
      await act(async () => {
        await result.current.scan();
      });
      act(() => {
        void result.current.keyframes.addFromCapture();
      });
      await waitFor(() => expect(result.current.keyframes.items[0]?.status).toBe("ready"));

      // Change nag AFTER the last render `start` would have closed over, then
      // call start() without any further render in between.
      rerender({ nag: { ...ENABLED_NAG, text: "fresh negative prompt" } });
      act(() => {
        result.current.start();
      });

      await waitFor(() => expect(chainBodies).toHaveLength(1));
      expect(chainBodies[0]?.negative_prompt).toBe("fresh negative prompt");
      // §3-47: 走らせたテストは走り切るまで待つ（シングルトンへの走り残り防止）。
      await waitFor(() => expect(result.current.runnerState).toBe("idle"));
    });
  });

  // Acceleration (2026-07-31): Batch has no acceleration UI of its own — it
  // silently inherits the Settings panel's choice via `deps.acceleration`,
  // exactly like `deps.nag` above.
  describe("deps.acceleration", () => {
    /** Picks/scans the wav folder, adds one ready shared keyframe and starts —
     * returning the single captured `/generate/chain` body. */
    async function startAndCapture(acceleration?: AccelerationSettings): Promise<Record<string, unknown>> {
      const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
      const base = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
      const { wrapped, chainBodies } = captureChainBridge(base);
      const { result } = renderBatchForm(wrapped, { ...(acceleration ? { acceleration } : {}) });

      await act(async () => {
        await result.current.pickWavDir();
      });
      await act(async () => {
        await result.current.scan();
      });
      act(() => {
        void result.current.keyframes.addFromCapture();
      });
      await waitFor(() => expect(result.current.keyframes.items[0]?.status).toBe("ready"));
      act(() => {
        result.current.start();
      });
      await waitFor(() => expect(chainBodies).toHaveLength(1));
      // §3-47: ランナーは`runtime.ts`のシングルトン。このヘルパは1つのテスト内で
      // 2回呼ばれることもある（all-defaults）ので、走り切らせてから返さないと
      // 次のマウントが走行中の状態を復元し、2回目のstart()はロックを取れない。
      await waitFor(() => expect(result.current.runnerState).toBe("idle"));
      return chainBodies[0]!;
    }

    it("omitted / all-defaults: no acceleration key reaches the row body", async () => {
      expect(await startAndCapture()).not.toHaveProperty("attention_backend");
      const withDefaults = await startAndCapture(ACCELERATION_DEFAULTS);
      expect(withDefaults).not.toHaveProperty("attention_backend");
      expect(withDefaults).not.toHaveProperty("fused_gguf_dequant_kernel");
      expect(withDefaults).not.toHaveProperty("vae_mode");
    });

    it("sage: every row body carries attention_backend", async () => {
      const body = await startAndCapture({ ...ACCELERATION_DEFAULTS, attentionBackend: "sage" });
      expect(body.attention_backend).toBe("sage");
    });

    it("block-swap prefetch off: every row body carries block_swap_prefetch", async () => {
      // S4: default is now true, so OFF is what diverges.
      const body = await startAndCapture({ ...ACCELERATION_DEFAULTS, blockSwapPrefetch: false });
      expect(body.block_swap_prefetch).toBe(false);
    });

    it("keep-resident on: every row body carries keep_resident", async () => {
      // §48: server default is false, so ON is what diverges. This is the
      // test that catches a missed entry in `useBatchForm`'s hand-written
      // "is any acceleration field off its default" OR-list.
      const body = await startAndCapture({ ...ACCELERATION_DEFAULTS, keepResident: true });
      expect(body.keep_resident).toBe(true);
    });

    it("fused GGUF dequant kernel off: every row body carries fused_gguf_dequant_kernel (§1-11)", async () => {
      // §51 (2026-08-04): the server default flipped to true, so OFF is what
      // diverges. Same role as the keep-resident test above — this is THE test
      // that catches a missed entry in `useBatchForm`'s hand-written OR-list,
      // which would otherwise make the toggle work everywhere except Batch A2V.
      const body = await startAndCapture({ ...ACCELERATION_DEFAULTS, fusedGgufDequantKernel: false });
      expect(body.fused_gguf_dequant_kernel).toBe(false);
    });

    it("PrunaVAED: every row body carries vae_mode (§52)", async () => {
      // §52 (2026-08-05): server default is "default", so PrunaVAED is what
      // diverges. Same role as the two tests above — THE test that catches a
      // missed entry in `useBatchForm`'s hand-written OR-list, which would
      // otherwise make the toggle work everywhere except Batch A2V.
      const body = await startAndCapture({ ...ACCELERATION_DEFAULTS, vaeMode: "prune_vaed" });
      expect(body.vae_mode).toBe("prune_vaed");
    });
  });

  // §1-7 相互ロック（`shell/runLock.ts`）: Batch A2V and Batch i2v-long share
  // the backend's single job slot, so only one of them may run at a time. This
  // panel takes the lock at `start()` and gives it back on its runner's
  // running -> idle transition, with the token it took.
  describe("run lock", () => {
    /** Renders the form against `bridge`, picks/scans the wav folder and adds
     * one ready shared keyframe — the minimum state where `canStart` is true. */
    async function readyToStart(bridge: NativeBridge) {
      const rendered = renderBatchForm(bridge);
      const { result } = rendered;
      await act(async () => {
        await result.current.pickWavDir();
      });
      await act(async () => {
        await result.current.scan();
      });
      act(() => {
        void result.current.keyframes.addFromCapture();
      });
      await waitFor(() => expect(result.current.keyframes.items[0]?.status).toBe("ready"));
      return rendered;
    }

    it("start() takes the lock synchronously and releases it when the run finishes", async () => {
      const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
      const base = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
      // The capture bridge 500s the submit, so the run ends after one row.
      const { wrapped, chainBodies } = captureChainBridge(base);
      const { result } = await readyToStart(wrapped);

      expect(isRunLockHeld()).toBe(false);
      expect(result.current.canStart).toBe(true);

      act(() => {
        result.current.start();
      });
      // Taken before `start()` returns — `BatchRunner.start()` flips to
      // `running` synchronously, so there is no window where a run is in
      // flight without the lock.
      expect(getRunLockOwner()).toBe(RUN_LOCK_OWNER_BATCH_A2V);

      await waitFor(() => expect(chainBodies).toHaveLength(1));
      await waitFor(() => expect(result.current.runnerState).toBe("idle"));
      expect(isRunLockHeld()).toBe(false);
    });

    it("blocks Start while Batch i2v-long holds the lock, and never submits anything", async () => {
      const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
      const base = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
      const { wrapped, chainBodies } = captureChainBridge(base);
      const { result } = await readyToStart(wrapped);

      const otherToken = acquireRunLock(RUN_LOCK_OWNER_BATCH_I2V_LONG);
      await waitFor(() => expect(result.current.lockedByOther).toBe(true));
      expect(result.current.canStart).toBe(false);

      // Even a direct call (bypassing the disabled button) submits nothing.
      act(() => {
        result.current.start();
      });
      expect(chainBodies).toHaveLength(0);
      expect(result.current.runnerState).toBe("idle");
      // The other feature's lock was neither stolen nor released.
      expect(getRunLockOwner()).toBe(RUN_LOCK_OWNER_BATCH_I2V_LONG);

      // Releasing it re-enables Start (the panel subscribes to the lock).
      act(() => {
        releaseRunLock(otherToken);
      });
      await waitFor(() => expect(result.current.lockedByOther).toBe(false));
      expect(result.current.canStart).toBe(true);
    });

    it("holding the lock itself is not 'lockedByOther' (that case is covered by runnerState)", async () => {
      const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
      const base = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
      const { wrapped } = captureChainBridge(base);
      const { result } = await readyToStart(wrapped);

      act(() => {
        result.current.start();
      });
      expect(result.current.lockedByOther).toBe(false);

      await waitFor(() => expect(result.current.runnerState).toBe("idle"));
      expect(isRunLockHeld()).toBe(false);
    });

    it("releases the lock when there was nothing to run (start() never goes running)", async () => {
      // An empty audio folder scans to zero rows, so `BatchRunner.start()`
      // returns {started:false} without ever leaving `idle`. The handback still
      // happens, on the run promise's `.then` — the ONE code path that covers
      // both endings `start()` has.
      //
      // §3-47（2026-09-02）: 以前はこの「対象ゼロ」だけ同期で即時解放する分岐を
      // 持っていた（1マイクロタスクぶんのロック点滅を避けるため）。その作り込みは
      // 引き算した——離散イベントのReact同期フラッシュ内で解放マイクロタスクが
      // 完了するため描画上は見えず、A2V固有の分岐を1つ増やすだけだったため。
      // よってここは`waitFor`で待つ。
      const fs = wavFolder([]);
      const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
      const { result } = renderBatchForm(bridge);

      await act(async () => {
        await result.current.pickWavDir();
      });
      await act(async () => {
        await result.current.scan();
      });
      expect(result.current.rows).toHaveLength(0);

      act(() => {
        result.current.start();
      });

      // 走行状態には一度もならない（対象行ゼロは`start()`の最初のawaitより前に
      // 判定される）。ロックだけが1マイクロタスク遅れて返る。
      expect(result.current.runnerState).toBe("idle");
      await waitFor(() => expect(isRunLockHeld()).toBe(false));
      expect(result.current.runnerState).toBe("idle");
    });

    // 2026-07-31 オーナー実機報告の修正: ロックの返却は実行Promise側
    // （§3-47以降は`runtime.ts`の`.then`）にぶら下がっている。Reactのeffectでは
    // ないので、走行中にこのフックがアンマウントされても（Create画面の
    // `remountTokens`リマウント）ロックが取り残されない。
    // ※アンマウント後も走行そのものは`runtime.ts`のシングルトンが持ち続ける。
    //   リマウント後のパネルがその走行へ再接続できることは
    //   `shell/runLock.crossPanel.test.tsx`と`runtime.test.ts`がピンしている。
    it("releases the lock even when the panel is unmounted mid-run (remount no longer strands it)", async () => {
      const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
      const base = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
      const { wrapped, chainBodies } = captureChainBridge(base);
      const { result, unmount } = await readyToStart(wrapped);

      act(() => {
        result.current.start();
      });
      expect(getRunLockOwner()).toBe(RUN_LOCK_OWNER_BATCH_A2V);

      // The panel goes away while the run is still in flight.
      unmount();
      await waitFor(() => expect(chainBodies).toHaveLength(1));
      // ...and the lock still comes back when that run ends, with no panel
      // mounted behind it to notice.
      await waitFor(() => expect(isRunLockHeld()).toBe(false));
    });

    // §1-7 相互ロック 第2段: the run lock is browser-volatile, so it cannot
    // protect the job slot across a reload — nor against a job no batch panel
    // started. `serverBusy` is the server-derived gate that does.
    it("blocks Start while a job is active, even with the run lock free, and submits nothing if called anyway", async () => {
      const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
      const base = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });
      const { wrapped, chainBodies } = captureChainBridge(base);
      const rendered = renderBatchForm(wrapped, { serverBusy: true });
      const { result } = rendered;
      await act(async () => {
        await result.current.pickWavDir();
      });
      await act(async () => {
        await result.current.scan();
      });
      act(() => {
        void result.current.keyframes.addFromCapture();
      });
      await waitFor(() => expect(result.current.keyframes.items[0]?.status).toBe("ready"));

      expect(isRunLockHeld()).toBe(false);
      expect(result.current.jobActive).toBe(true);
      expect(result.current.canStart).toBe(false);

      // Even a direct call (bypassing the disabled button) submits nothing and
      // never takes the lock.
      act(() => {
        result.current.start();
      });
      expect(chainBodies).toHaveLength(0);
      expect(isRunLockHeld()).toBe(false);
      expect(result.current.runnerState).toBe("idle");
    });

    it("a freshly mounted panel never releases a lock held by the other feature", async () => {
      // 所有者トークン方式の存在理由そのもの（両画面が常時マウントのため）:
      // mounting this panel while i2v-long is running must not free its lock.
      const otherToken = acquireRunLock(RUN_LOCK_OWNER_BATCH_I2V_LONG);
      const fs = wavFolder([{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }]);
      const bridge = createMockBridge({ delayMs: 0, fs, pickFolderPath: WAV_DIR });

      const { result, unmount } = renderBatchForm(bridge);
      await act(async () => {
        await result.current.pickWavDir();
      });
      expect(getRunLockOwner()).toBe(RUN_LOCK_OWNER_BATCH_I2V_LONG);

      // Unmounting must not release it either.
      unmount();
      expect(getRunLockOwner()).toBe(RUN_LOCK_OWNER_BATCH_I2V_LONG);
      expect(releaseRunLock(otherToken)).toBe(true);
    });
  });

  // §3-47（2026-09-02）: ランナー実体は`runtime.ts`のモジュールレベル・
  // シングルトン。設計判断1の両側——「復元するのは走行中のときだけ」——を、
  // アンマウント→再マウント（Create画面の`remountTokens`リマウント相当）で
  // 両方向ともピン留めする。
  describe("リマウント時の復元（§3-47）", () => {
    const IMG_DIR = "C:\\voice\\ep01-images";

    /** 音声フォルダと画像フォルダの両方を持つmock fs。行に自前の画像を割り当てて
     * 走らせるので、Shared用のキーフレーム（＝画像アップロード）が要らない。 */
    function bothFolders() {
      return createMockFs({
        folders: {
          [WAV_DIR]: [{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 1.0 }],
          [IMG_DIR]: [{ name: "cover.png", sizeBytes: 10, mtimeMs: 1 }],
        },
      });
    }

    /** 両フォルダを手入力で確定し、スキャンして、行に自前の画像を割り当てる
     * ——`canStart`が立つ最小状態まで持っていく。 */
    async function readyWithOwnImage(bridge: NativeBridge) {
      const rendered = renderBatchForm(bridge);
      const { result } = rendered;
      await act(async () => {
        await result.current.setWavDir(WAV_DIR);
      });
      await act(async () => {
        await result.current.setImgDir(IMG_DIR);
      });
      await act(async () => {
        await result.current.scan();
      });
      act(() => {
        result.current.updateRowImage(0, "cover.png");
      });
      expect(result.current.sharedKeyframeMissing).toBe(false);
      expect(result.current.canStart).toBe(true);
      return rendered;
    }

    it("走行中のリマウントは、フォルダ・行・画像選択肢を復元して走行へ再接続する", async () => {
      // `holdUploads`で音声アップロード（`processRow`の最初のawait）に走行を
      // 停め、その間にリマウントする。解放後は`captureChainBridge`が投入を
      // 500で弾くので、ジョブのポーリング待ちなしで走り切る。
      const base = createMockBridge({ delayMs: 0, fs: bothFolders(), holdUploads: true });
      const { wrapped } = captureChainBridge(base);
      const { result, unmount } = await readyWithOwnImage(wrapped);

      act(() => {
        result.current.start();
      });
      expect(result.current.runnerState).toBe("running");

      unmount();
      const { result: remounted } = renderBatchForm(wrapped);

      // 再マウント直後の初回レンダーで、走行中であることも入力も見えている。
      expect(remounted.current.runnerState).toBe("running");
      expect(remounted.current.wavDir).toBe(WAV_DIR);
      expect(remounted.current.imgDir).toBe(IMG_DIR);
      expect(remounted.current.outDir).toBe(AUTO_OUT_DIR);
      // 行と、行の画像`<select>`の選択肢（スキャン由来の派生状態）も戻る。
      expect(remounted.current.rows[0]).toMatchObject({ wav: "a.wav", image: "cover.png", stat: "Generating" });
      expect(remounted.current.imageOptions).toEqual(["Shared", "cover.png"]);
      // 走行中はStopが出せる状態（行編集はUI側で無効化される）。
      expect(remounted.current.canStart).toBe(false);

      // 復元は初回レンダーだけの話ではない: 以降の行の進捗も、リマウント後の
      // パネルへ届き続ける（＝走行中のランナーに実際に繋がっている）。
      base.releaseUploads();
      await waitFor(() => expect(remounted.current.runnerState).toBe("idle"));
      // `captureChainBridge`が投入を500で弾くので、行はFailedで終わる。その
      // 遷移はリマウント後に起きているので、届いていれば復元が生きている証拠。
      expect(remounted.current.rows[0]).toMatchObject({ wav: "a.wav", stat: "Failed" });
    }, 15_000);

    it("走行が終わった後のリマウントは、従来どおり白紙から始まる（走行中だけ復元する）", async () => {
      // A2Vはステートレスバッチ（行はメモリのみ・CSVなし、オーナー裁定
      // 2026-07-18）。走り終わった後まで前回の入力や行を復活させると、その設計に
      // 反するうえ「フォルダ欄は空なのに前回の行だけ残っている」という半端な
      // 状態になる。
      const base = createMockBridge({ delayMs: 0, fs: bothFolders() });
      const { wrapped, chainBodies } = captureChainBridge(base);
      const { result, unmount } = await readyWithOwnImage(wrapped);

      act(() => {
        result.current.start();
      });
      await waitFor(() => expect(chainBodies).toHaveLength(1));
      await waitFor(() => expect(result.current.runnerState).toBe("idle"));

      unmount();
      const { result: remounted } = renderBatchForm(wrapped);

      expect(remounted.current.runnerState).toBe("idle");
      expect(remounted.current.wavDir).toBeNull();
      expect(remounted.current.imgDir).toBeNull();
      expect(remounted.current.outDir).toBeNull();
      expect(remounted.current.rows).toEqual([]);
      expect(remounted.current.imageOptions).toEqual(["Shared"]);
      // 出力フォルダの自動導出も生きたまま（設計判断3: `outDirIsAuto`は
      // 走行入力から導出しない）——音声フォルダを選べばちゃんと再導出される。
      expect(remounted.current.outDirIsAuto).toBe(true);
      await act(async () => {
        await remounted.current.setWavDir(WAV_DIR);
      });
      expect(remounted.current.outDir).toBe(AUTO_OUT_DIR);
    }, 15_000);
  });
});
