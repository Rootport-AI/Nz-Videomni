import { describe, expect, it } from "vitest";
import { createMockBridge } from "./mockBridge";

/**
 * Contract v7 (drag-and-drop). Post-review fidelity fix: `requestWithFiles`
 * must merge the dropped files into `params.__droppedPaths` BEFORE dispatch
 * — mirroring native's real injection (`webview_host.cpp`'s
 * `InjectDroppedPaths`) — rather than synthesizing the result directly from
 * the `files` array and bypassing the injection step entirely. An earlier
 * version of this mock did the latter, which is exactly why a real native
 * regression (the injection never firing for a genuine drop at all) slipped
 * past every webui test: the mock's shortcut couldn't fail the same way.
 * These tests exercise the SAME params.__droppedPaths shape the real
 * `ParseResolveDroppedFiles` reads.
 */
describe("createMockBridge — contract v7 (drag-and-drop)", () => {
  describe("requestWithFiles / ui.resolveDroppedFiles", () => {
    it("merges dropped files' names into params.__droppedPaths and resolves them", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const file = new File([], "clip.mp4");
      const result = await bridge.requestWithFiles("ui.resolveDroppedFiles", {}, [file]);
      expect(result.files).toHaveLength(1);
      expect(result.files[0]?.fileName).toBe("clip.mp4");
      expect(result.files[0]?.filePath).toContain("clip.mp4");
    });

    it("overwrites any pre-existing __droppedPaths in params with the real dropped files "
      + "(mirrors native never letting a page-authored value survive)", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const file = new File([], "real.mp4");
      // A caller could never actually construct this in practice (the type
      // declares empty params), but the underlying merge is `{...params,
      // __droppedPaths: ...}` — verifying the REAL value wins even if some
      // caller's params object already carried the key proves the injection
      // always overwrites rather than only filling a gap.
      const spoofedParams = { __droppedPaths: ["C:\\spoofed\\fake.png"] } as unknown as Record<string, never>;
      const result = await bridge.requestWithFiles("ui.resolveDroppedFiles", spoofedParams, [file]);
      expect(result.files).toHaveLength(1);
      expect(result.files[0]?.fileName).toBe("real.mp4");
    });

    it("resolves only the files actually passed in (first-file-only slots pass a single file)", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const file = new File([], "one.png");
      const result = await bridge.requestWithFiles("ui.resolveDroppedFiles", {}, [file]);
      expect(result.files).toHaveLength(1);
      expect(result.files[0]?.fileName).toBe("one.png");
    });

    it("excludes a File-like object with no usable name, resolving to files: []", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await bridge.requestWithFiles("ui.resolveDroppedFiles", {}, [{}]);
      expect(result.files).toEqual([]);
    });

    it("rejects when failResolveDroppedFiles is set", async () => {
      const bridge = createMockBridge({ delayMs: 0, failResolveDroppedFiles: true });
      const file = new File([], "clip.mp4");
      await expect(bridge.requestWithFiles("ui.resolveDroppedFiles", {}, [file])).rejects.toMatchObject({
        code: "BAD_REQUEST",
      });
    });

    it("plain request() (no files attached) resolves to files: [], mirroring native's "
      + "'no __droppedPaths key' case", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await bridge.request("ui.resolveDroppedFiles", {});
      expect(result.files).toEqual([]);
    });

    it("requestWithFiles for any OTHER method still dispatches normally, ignoring the files", async () => {
      const bridge = createMockBridge({ delayMs: 0, pickFileName: "photo.png" });
      const file = new File([], "unrelated.mp4");
      const result = await bridge.requestWithFiles("ui.pickFile", { kind: "image" }, [file]);
      expect(result.fileName).toBe("photo.png");
    });
  });
});
