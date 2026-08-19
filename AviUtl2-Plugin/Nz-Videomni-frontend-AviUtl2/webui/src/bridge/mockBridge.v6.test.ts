import { describe, expect, it } from "vitest";
import { createMockBridge, createMockFs } from "./mockBridge";

describe("createMockBridge — contract v6 (batch A2V + fs bridge)", () => {
  describe("ui.pickFolder", () => {
    it("resolves a canned folderPath by default", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await bridge.request("ui.pickFolder", {});
      expect(result.folderPath).toMatch(/picked-folder-1$/);
    });

    it("resolves the overridden pickFolderPath, ignoring title", async () => {
      const bridge = createMockBridge({ delayMs: 0, pickFolderPath: "C:\\batch\\in" });
      const result = await bridge.request("ui.pickFolder", { title: "入力フォルダを選択" });
      expect(result.folderPath).toBe("C:\\batch\\in");
    });

    it("mints a fresh default folderPath per call", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const a = await bridge.request("ui.pickFolder", {});
      const b = await bridge.request("ui.pickFolder", {});
      expect(a.folderPath).not.toBe(b.folderPath);
    });

    it("rejects with CANCELLED when failPickFolder is 'CANCELLED'", async () => {
      const bridge = createMockBridge({ delayMs: 0, failPickFolder: "CANCELLED" });
      await expect(bridge.request("ui.pickFolder", {})).rejects.toMatchObject({ code: "CANCELLED" });
    });

    it("rejects with DIALOG_FAILED when failPickFolder is 'DIALOG_FAILED'", async () => {
      const bridge = createMockBridge({ delayMs: 0, failPickFolder: "DIALOG_FAILED" });
      await expect(bridge.request("ui.pickFolder", {})).rejects.toMatchObject({ code: "DIALOG_FAILED" });
    });
  });

  describe("fs.listFiles", () => {
    it("lists all entries in a folder when extensions is omitted", async () => {
      const fs = createMockFs({
        folders: {
          "C:\\batch\\in": [
            { name: "a.wav", sizeBytes: 100, mtimeMs: 1000 },
            { name: "b.csv", sizeBytes: 50, mtimeMs: 2000 },
            { name: "c.mp3", sizeBytes: 75, mtimeMs: 3000 },
          ],
        },
      });
      const bridge = createMockBridge({ delayMs: 0, fs });
      const result = await bridge.request("fs.listFiles", { folderPath: "C:\\batch\\in" });
      expect(result.files).toHaveLength(3);
      expect(result.files.map((f) => f.name).sort()).toEqual(["a.wav", "b.csv", "c.mp3"]);
      // path is the folder joined with the entry name.
      expect(result.files.find((f) => f.name === "a.wav")?.path).toBe("C:\\batch\\in\\a.wav");
    });

    it("filters by extension case-insensitively", async () => {
      const fs = createMockFs({
        folders: {
          "C:\\batch\\in": [
            { name: "a.WAV", sizeBytes: 100, mtimeMs: 1000 },
            { name: "b.csv", sizeBytes: 50, mtimeMs: 2000 },
            { name: "c.mp3", sizeBytes: 75, mtimeMs: 3000 },
          ],
        },
      });
      const bridge = createMockBridge({ delayMs: 0, fs });
      const result = await bridge.request("fs.listFiles", {
        folderPath: "C:\\batch\\in",
        extensions: [".wav"],
      });
      expect(result.files).toHaveLength(1);
      expect(result.files[0]?.name).toBe("a.WAV");
    });

    it("returns an empty list for an unknown folder", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await bridge.request("fs.listFiles", { folderPath: "C:\\nope" });
      expect(result.files).toEqual([]);
    });

    it("reports durationSec 0 for every file when withAudioDuration is omitted", async () => {
      const fs = createMockFs({
        folders: {
          "C:\\batch\\in": [{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 3.5 }],
        },
      });
      const bridge = createMockBridge({ delayMs: 0, fs });
      const result = await bridge.request("fs.listFiles", { folderPath: "C:\\batch\\in" });
      expect(result.files[0]?.durationSec).toBe(0);
    });

    it("reports the real durationSec for a readable .wav when withAudioDuration is true", async () => {
      const fs = createMockFs({
        folders: {
          "C:\\batch\\in": [{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 3.5 }],
        },
      });
      const bridge = createMockBridge({ delayMs: 0, fs });
      const result = await bridge.request("fs.listFiles", {
        folderPath: "C:\\batch\\in",
        withAudioDuration: true,
      });
      expect(result.files[0]?.durationSec).toBe(3.5);
    });

    it("reports durationSec 0 for a non-wav file even when withAudioDuration is true", async () => {
      const fs = createMockFs({
        folders: {
          "C:\\batch\\in": [{ name: "a.mp3", sizeBytes: 100, mtimeMs: 1000, durationSec: 3.5 }],
        },
      });
      const bridge = createMockBridge({ delayMs: 0, fs });
      const result = await bridge.request("fs.listFiles", {
        folderPath: "C:\\batch\\in",
        withAudioDuration: true,
      });
      expect(result.files[0]?.durationSec).toBe(0);
    });

    it("reports durationSec 0 for an unreadable .wav (no durationSec seeded) when withAudioDuration is true", async () => {
      const fs = createMockFs({
        folders: {
          "C:\\batch\\in": [{ name: "broken.wav", sizeBytes: 10, mtimeMs: 1000 }],
        },
      });
      const bridge = createMockBridge({ delayMs: 0, fs });
      const result = await bridge.request("fs.listFiles", {
        folderPath: "C:\\batch\\in",
        withAudioDuration: true,
      });
      expect(result.files[0]?.durationSec).toBe(0);
    });
  });

  describe("fs.probeAudioDuration", () => {
    it("resolves the real duration for a readable .wav registered in fs.folders", async () => {
      const fs = createMockFs({
        folders: {
          "C:\\batch\\in": [{ name: "a.wav", sizeBytes: 100, mtimeMs: 1000, durationSec: 4.25 }],
        },
      });
      const bridge = createMockBridge({ delayMs: 0, fs });
      const result = await bridge.request("fs.probeAudioDuration", { filePath: "C:\\batch\\in\\a.wav" });
      expect(result).toEqual({ durationSec: 4.25, isWav: true });
    });

    it("resolves {0, false} for a non-wav file (never rejects)", async () => {
      const fs = createMockFs({
        folders: {
          "C:\\batch\\in": [{ name: "a.mp3", sizeBytes: 100, mtimeMs: 1000 }],
        },
      });
      const bridge = createMockBridge({ delayMs: 0, fs });
      const result = await bridge.request("fs.probeAudioDuration", { filePath: "C:\\batch\\in\\a.mp3" });
      expect(result).toEqual({ durationSec: 0, isWav: false });
    });

    it("resolves {0, false} for a .wav path with no matching fs entry (unreadable)", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await bridge.request("fs.probeAudioDuration", { filePath: "C:\\ghost\\a.wav" });
      expect(result).toEqual({ durationSec: 0, isWav: false });
    });
  });

  describe("backend.downloadVideo — contract v6 destDir/fileName/noClobber extension", () => {
    async function completeAJob(bridge: ReturnType<typeof createMockBridge>): Promise<string> {
      const created = await bridge.request("backend.request", {
        method: "POST",
        path: "/api/v1/generate",
        body: { prompt: "a cat" },
      });
      const jobId = (created.body as { job_id: string }).job_id;
      // runningPollCount defaults to 2; poll until completed.
      for (let i = 0; i < 4; i += 1) {
        await bridge.request("backend.request", { method: "GET", path: `/api/v1/jobs/${jobId}` });
      }
      return jobId;
    }

    it("preserves the exact pre-v6 path when destDir/fileName/noClobber are omitted", async () => {
      const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1 });
      const jobId = await completeAJob(bridge);
      const result = await bridge.request("backend.downloadVideo", { jobId });
      expect(result.filePath).toBe(`C:\\Users\\mock\\AppData\\Local\\Temp\\Nz-Videomni\\${jobId}-output.mp4`);
    });

    it("saves to destDir with a custom fileName", async () => {
      const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1 });
      const jobId = await completeAJob(bridge);
      const result = await bridge.request("backend.downloadVideo", {
        jobId,
        destDir: "C:\\batch\\out",
        fileName: "clip_001.mp4",
      });
      expect(result.filePath).toBe("C:\\batch\\out\\clip_001.mp4");
    });

    it("noClobber leaves the name unchanged when there is no collision", async () => {
      const fs = createMockFs();
      const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
      const jobId = await completeAJob(bridge);
      const result = await bridge.request("backend.downloadVideo", {
        jobId,
        destDir: "C:\\batch\\out",
        fileName: "clip_001.mp4",
        noClobber: true,
      });
      expect(result.filePath).toBe("C:\\batch\\out\\clip_001.mp4");
    });

    it("noClobber appends _2 on a single collision", async () => {
      const fs = createMockFs({
        folders: { "C:\\batch\\out": [{ name: "clip_001.mp4", sizeBytes: 1, mtimeMs: 1 }] },
      });
      const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
      const jobId = await completeAJob(bridge);
      const result = await bridge.request("backend.downloadVideo", {
        jobId,
        destDir: "C:\\batch\\out",
        fileName: "clip_001.mp4",
        noClobber: true,
      });
      expect(result.filePath).toBe("C:\\batch\\out\\clip_001_2.mp4");
      // The pre-existing file must remain untouched (not overwritten).
      const original = fs.folders.get("C:\\batch\\out")?.find((e) => e.name === "clip_001.mp4");
      expect(original).toMatchObject({ name: "clip_001.mp4", sizeBytes: 1, mtimeMs: 1 });
    });

    it("noClobber appends increasing suffixes across consecutive collisions", async () => {
      const fs = createMockFs({
        folders: {
          "C:\\batch\\out": [
            { name: "clip_001.mp4", sizeBytes: 1, mtimeMs: 1 },
            { name: "clip_001_2.mp4", sizeBytes: 1, mtimeMs: 1 },
            { name: "clip_001_3.mp4", sizeBytes: 1, mtimeMs: 1 },
          ],
        },
      });
      const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
      const jobId = await completeAJob(bridge);
      const result = await bridge.request("backend.downloadVideo", {
        jobId,
        destDir: "C:\\batch\\out",
        fileName: "clip_001.mp4",
        noClobber: true,
      });
      expect(result.filePath).toBe("C:\\batch\\out\\clip_001_4.mp4");
    });

    it("never overwrites an existing file's recorded metadata when noClobber is true", async () => {
      const fs = createMockFs({
        folders: { "C:\\batch\\out": [{ name: "clip_001.mp4", sizeBytes: 999, mtimeMs: 12345 }] },
      });
      const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
      const jobId = await completeAJob(bridge);
      await bridge.request("backend.downloadVideo", {
        jobId,
        destDir: "C:\\batch\\out",
        fileName: "clip_001.mp4",
        noClobber: true,
      });
      const entries = fs.folders.get("C:\\batch\\out") ?? [];
      expect(entries).toHaveLength(2);
      const original = entries.find((e) => e.name === "clip_001.mp4");
      expect(original).toEqual({ name: "clip_001.mp4", sizeBytes: 999, mtimeMs: 12345 });
    });

    it("without noClobber, a same-named re-download overwrites the recorded entry in place", async () => {
      const fs = createMockFs({
        folders: { "C:\\batch\\out": [{ name: "clip_001.mp4", sizeBytes: 1, mtimeMs: 1 }] },
      });
      const bridge = createMockBridge({ delayMs: 0, runningPollCount: 1, fs });
      const jobId = await completeAJob(bridge);
      const result = await bridge.request("backend.downloadVideo", {
        jobId,
        destDir: "C:\\batch\\out",
        fileName: "clip_001.mp4",
      });
      expect(result.filePath).toBe("C:\\batch\\out\\clip_001.mp4");
      const entries = fs.folders.get("C:\\batch\\out") ?? [];
      expect(entries).toHaveLength(1);
      expect(entries[0]?.sizeBytes).toBe(4_096 + 512);
    });
  });
});
