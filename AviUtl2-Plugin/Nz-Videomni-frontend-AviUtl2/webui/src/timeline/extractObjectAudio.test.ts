import { beforeEach, describe, expect, it, vi } from "vitest";
import { BridgeError } from "../bridge/types";
import type { ResultOf } from "../bridge/types";
import { createMockBridge } from "../bridge/mockBridge";
import { en } from "../i18n/strings";
import { classifyExtractAudioError, extractAudioFailureNote, extractObjectAudio, resetExtractObjectAudioChain } from "./extractObjectAudio";
import type { SelectionItem } from "./menuSelection";

beforeEach(() => {
  resetExtractObjectAudioChain();
});

function item(layer: number, frameStart: number, frameEnd: number): SelectionItem {
  return {
    layer,
    frameStart,
    frameEnd,
    effectName: "動画ファイル",
    filePath: null,
    objectName: "x",
    textContent: null,
    mediaWidth: 1920,
    mediaHeight: 1080,
    mediaDurationSec: 10,
  };
}

type ExtractResult = ResultOf<"timeline.extractAudio">;

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (err: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const wav = (name: string): ExtractResult => ({
  filePath: `C:/tmp/${name}.wav`,
  durationSec: 2,
  sampleRate: 48000,
  hasAudioStream: true,
});

describe("extractObjectAudio", () => {
  it("requests a solo render of exactly the ribbon range (frameEnd inclusive) and names the wav", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const spy = vi.spyOn(bridge, "request");

    const result = await extractObjectAudio(bridge, item(3, 10, 130));

    expect(spy).toHaveBeenCalledWith("timeline.extractAudio", {
      layer: 3,
      frameStart: 10,
      frameCount: 121,
      audioMode: "solo",
      soloKeepLayers: [3],
    });
    expect(result.fileName).toMatch(/^extract_3_10_[0-9]+[.]wav$/);
    expect(result.filePath.endsWith(result.fileName)).toBe(true);
    // The mock reports frameCount / project fps (30/1) as the wav's length.
    expect(result.durationSec).toBeCloseTo(121 / 30, 2);
  });

  it("serializes calls: the second request waits until the first has settled", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const first = deferred<ExtractResult>();
    const order: string[] = [];
    const spy = vi.spyOn(bridge, "request").mockImplementation(((_method: string, params: { layer: number }) => {
      order.push(`start ${params.layer}`);
      if (params.layer === 1) return first.promise;
      return Promise.resolve(wav("second"));
    }) as typeof bridge.request);

    const a = extractObjectAudio(bridge, item(1, 0, 10));
    const b = extractObjectAudio(bridge, item(2, 0, 10));
    await new Promise((r) => setTimeout(r, 0));
    // Only the first has started; the second is queued behind it.
    expect(order).toEqual(["start 1"]);

    first.resolve(wav("first"));
    await expect(a).resolves.toMatchObject({ fileName: "first.wav" });
    await expect(b).resolves.toMatchObject({ fileName: "second.wav" });
    expect(order).toEqual(["start 1", "start 2"]);
    expect(spy).toHaveBeenCalledTimes(2);
  });

  it("a rejected call does not poison the chain: the next call still runs", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const first = deferred<ExtractResult>();
    const order: string[] = [];
    vi.spyOn(bridge, "request").mockImplementation(((_method: string, params: { layer: number }) => {
      order.push(`start ${params.layer}`);
      if (params.layer === 1) return first.promise;
      return Promise.resolve(wav("after"));
    }) as typeof bridge.request);

    const a = extractObjectAudio(bridge, item(1, 0, 10));
    const b = extractObjectAudio(bridge, item(2, 0, 10));
    await new Promise((r) => setTimeout(r, 0));
    expect(order).toEqual(["start 1"]);

    first.reject(new BridgeError("EXTRACT_FAILED", "no audio stream present (silent range)"));
    await expect(a).rejects.toThrow(/silent range/);
    await expect(b).resolves.toMatchObject({ fileName: "after.wav" });
    expect(order).toEqual(["start 1", "start 2"]);
  });
});

describe("classifyExtractAudioError", () => {
  it.each([
    ["no audio stream present (silent range)", "silent"],
    ["Silent Range", "silent"],
    ["audio render failed at frame 3", "timeout"],
    ["rendering_scene_audio is not available", "failed"],
    ["failed to write wav file", "failed"],
  ] as const)("%s -> %s", (message, kind) => {
    expect(classifyExtractAudioError(new BridgeError("EXTRACT_FAILED", message))).toBe(kind);
  });

  it("classifies a BridgeError TIMEOUT as timeout", () => {
    expect(classifyExtractAudioError(new BridgeError("TIMEOUT", "timed out"))).toBe("timeout");
  });

  it("treats a non-Error rejection as a generic failure", () => {
    expect(classifyExtractAudioError("boom")).toBe("failed");
    expect(classifyExtractAudioError(undefined)).toBe("failed");
  });

  it("maps each kind to its note", () => {
    expect(extractAudioFailureNote(new Error("silent range"), en.notes)).toBe(en.notes.audioExtractSilent);
    expect(extractAudioFailureNote(new Error("audio render failed at frame 1"), en.notes)).toBe(
      en.notes.audioExtractTimeout,
    );
    expect(extractAudioFailureNote(new Error("other"), en.notes)).toBe(en.notes.audioExtractFailed);
  });
});
