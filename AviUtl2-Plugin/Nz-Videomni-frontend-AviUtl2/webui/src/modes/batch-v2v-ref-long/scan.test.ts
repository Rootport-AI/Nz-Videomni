import { describe, expect, it } from "vitest";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import type { MockFsFileEntry } from "../../bridge/mockBridge";
import type { NativeBridge } from "../../bridge";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import type { ReferenceRules } from "../single/referenceRules";
import { createDefaultFrame } from "./folderRules";
import { scanFolders } from "./scan";
import type { FolderFrame, ScanInputs } from "./types";

const RULES: ReferenceRules = {
  maxImages: 9,
  maxVideos: 3,
  maxAudios: 3,
  maxTotal: 12,
  videoTotalSec: 15,
  audioTotalSec: 15,
  clipMinSec: 1,
};
const UPLOAD = FALLBACK_APP_CONFIG.upload;

function entry(name: string, durationSec?: number): MockFsFileEntry {
  return { name, sizeBytes: 1000, mtimeMs: 0, ...(durationSec !== undefined ? { durationSec } : {}) };
}

/** Sets the default frame's Picture 1 / Picture 2 / Video 1 / Audio 1 paths. */
function frame(id: number, paths: { p1?: string; p2?: string; v1?: string; a1?: string }): FolderFrame {
  const base = createDefaultFrame(id);
  const byId: Record<number, string | undefined> = { 1: paths.p1, 2: paths.p2, 3: paths.v1, 4: paths.a1 };
  return { ...base, fields: base.fields.map((f) => ({ ...f, path: byId[f.id] ?? "" })) };
}

interface Call {
  method: string;
  params: unknown;
}

function spy(bridge: NativeBridge): { wrapped: NativeBridge; calls: Call[] } {
  const calls: Call[] = [];
  return {
    calls,
    wrapped: {
      request: (method, params) => {
        calls.push({ method, params });
        return bridge.request(method, params);
      },
      requestWithFiles: (method, params, files) => bridge.requestWithFiles(method, params, files),
      on: (event, handler) => bridge.on(event, handler),
    },
  };
}

function run(bridge: NativeBridge, inputs: ScanInputs, progress: Array<[number, number]> = []) {
  return scanFolders(bridge, inputs, UPLOAD, RULES, (done, total) => progress.push([done, total]));
}

describe("scanFolders", () => {
  it("picks up only the formats each field's card accepts (dotted lists, real mock matching)", async () => {
    const fs = createMockFs({
      folders: {
        "C:\\mix": [
          entry("a.png"),
          entry("b.mp4", 3),
          entry("c.wav", 2),
          entry("d.txt"),
          entry("e.JPG"),
        ],
      },
    });
    const bridge = createMockBridge({ delayMs: 0, fs });
    const result = await run(bridge, { videoDir: "C:\\mix", frames: [frame(1, { p1: "C:\\mix", v1: "C:\\mix", a1: "C:\\mix" })] });

    expect(result.source.map((f) => f.name)).toEqual(["b.mp4"]);
    const [pic, vid, aud] = result.frames[0] ?? [];
    expect(pic).toMatchObject({ kind: "picture", slot: 1, path: "C:\\mix" });
    expect(pic?.files.map((f) => f.name)).toEqual(["a.png", "e.JPG"]);
    expect(vid?.files.map((f) => f.name)).toEqual(["b.mp4"]);
    // Audio takes video containers too.
    expect(aud?.files.map((f) => f.name)).toEqual(["b.mp4", "c.wav"]);
  });

  it("sorts in natural name order", async () => {
    const fs = createMockFs({ folders: { "C:\\p": [entry("img10.png"), entry("img2.png"), entry("img1.png")] } });
    const result = await run(createMockBridge({ delayMs: 0, fs }), { videoDir: "", frames: [frame(1, { p1: "C:\\p" })] });
    expect(result.frames[0]?.[0]?.files.map((f) => f.name)).toEqual(["img1.png", "img2.png", "img10.png"]);
    expect(result.frames[0]?.[0]?.files[0]).toEqual({ name: "img1.png", path: "C:\\p\\img1.png", sizeBytes: 1000, durationSec: null });
  });

  it("lists each (folder, kind) once; the opening video and Video share one listing", async () => {
    const fs = createMockFs({ folders: { "C:\\v": [entry("a.mp4", 4)], "C:\\p": [entry("x.png")] } });
    const { wrapped, calls } = spy(createMockBridge({ delayMs: 0, fs }));
    const result = await run(wrapped, {
      videoDir: "C:\\v",
      frames: [frame(1, { p1: "C:\\p", p2: "C:\\p", v1: "C:\\v", a1: "C:\\v" }), frame(2, { p1: "C:\\p" })],
    });
    const listings = calls.filter((c) => c.method === "fs.listFiles").map((c) => c.params as { folderPath: string; extensions: string[] });
    expect(listings).toHaveLength(3);
    expect(listings).toContainEqual({ folderPath: "C:\\v", extensions: UPLOAD.allowed_video_extensions });
    expect(listings).toContainEqual({ folderPath: "C:\\v", extensions: UPLOAD.allowed_audio_extensions });
    expect(listings).toContainEqual({ folderPath: "C:\\p", extensions: UPLOAD.allowed_image_extensions });
    // Every field still gets its own entry, with its slot.
    expect(result.frames[0]?.map((f) => [f.kind, f.slot])).toEqual([
      ["picture", 1],
      ["picture", 2],
      ["video", 1],
      ["audio", 1],
    ]);
    expect(result.frames[1]?.map((f) => [f.kind, f.slot])).toEqual([["picture", 1]]);
  });

  it("measures only videos and audios, once per path, and reports progress", async () => {
    const fs = createMockFs({
      folders: {
        "C:\\v": [entry("a.mp4", 4), entry("b.mp4", 6)],
        "C:\\p": [entry("x.png", 9)],
        "C:\\au": [entry("s.wav", 2.5)],
      },
    });
    const { wrapped, calls } = spy(createMockBridge({ delayMs: 0, fs }));
    const progress: Array<[number, number]> = [];
    const result = await run(
      wrapped,
      { videoDir: "C:\\v", frames: [frame(1, { p1: "C:\\p", v1: "C:\\v", a1: "C:\\au" })] },
      progress,
    );
    const probed = calls.filter((c) => c.method === "fs.probeMediaInfo").map((c) => (c.params as { filePath: string }).filePath);
    expect(probed).toEqual(["C:\\v\\a.mp4", "C:\\v\\b.mp4", "C:\\au\\s.wav"]);
    expect(progress).toEqual([
      [0, 3],
      [1, 3],
      [2, 3],
      [3, 3],
    ]);
    expect(result.source.map((f) => f.durationSec)).toEqual([4, 6]);
    expect(result.frames[0]?.[1]?.files.map((f) => f.durationSec)).toEqual([4, 6]);
    expect(result.frames[0]?.[2]?.files[0]?.durationSec).toBe(2.5);
    expect(result.frames[0]?.[0]?.files[0]?.durationSec).toBeNull();
  });

  it("an unmeasurable file keeps a null length", async () => {
    const fs = createMockFs({ folders: { "C:\\v": [entry("a.mp4")] } });
    const result = await run(createMockBridge({ delayMs: 0, fs }), { videoDir: "C:\\v", frames: [frame(1, {})] });
    expect(result.source[0]?.durationSec).toBeNull();
  });

  it("an empty field and a missing folder both come back with no files", async () => {
    const fs = createMockFs({ folders: { "C:\\p": [entry("notes.txt")] } });
    const progress: Array<[number, number]> = [];
    const result = await run(
      createMockBridge({ delayMs: 0, fs }),
      { videoDir: "C:\\nowhere", frames: [frame(1, { p1: "C:\\p", v1: "C:\\gone" })] },
      progress,
    );
    expect(result.source).toEqual([]);
    expect(result.frames[0]?.map((f) => f.files.length)).toEqual([0, 0]);
    expect(progress).toEqual([]);
  });

  it("skips blank paths, trims the rest, and copies the inputs", async () => {
    const fs = createMockFs({ folders: { "C:\\p": [entry("x.png")] } });
    const inputs: ScanInputs = { videoDir: "  ", frames: [frame(1, { p1: "  C:\\p  " }), frame(2, {})] };
    const result = await run(createMockBridge({ delayMs: 0, fs }), inputs);
    expect(result.frames).toHaveLength(2);
    expect(result.frames[0]?.[0]?.path).toBe("C:\\p");
    expect(result.frames[1]).toEqual([]);
    expect(result.rules).toBe(RULES);

    expect(result.inputs).toEqual(inputs);
    inputs.frames[0]!.fields[0]!.path = "changed";
    expect(result.inputs.frames[0]?.fields[0]?.path).toBe("  C:\\p  ");
  });

  it("rethrows an fs.listFiles failure", async () => {
    const base = createMockBridge({ delayMs: 0 });
    const failing: NativeBridge = {
      request: (method, params) =>
        method === "fs.listFiles" ? Promise.reject(new Error("disk gone")) : base.request(method, params),
      requestWithFiles: (method, params, files) => base.requestWithFiles(method, params, files),
      on: (event, handler) => base.on(event, handler),
    };
    await expect(run(failing, { videoDir: "C:\\v", frames: [frame(1, {})] })).rejects.toThrow("disk gone");
  });
});
