import { describe, expect, it } from "vitest";
import { createMockBridge } from "./mockBridge";

/**
 * 契約v10（§1-6 元動画の範囲トリム）のモック追随。
 *
 * Contract v10's `backend.uploadFile.query`: native URL-encodes the map onto the
 * upload endpoint's URL. The real `/upload/video` returns `trimmed: true` only
 * when BOTH trim parameters arrived and the cut actually succeeded, so the mock
 * mirrors exactly that rule — otherwise a WebUI test could never distinguish "the
 * trim applied" from "the trim was silently dropped", which is the whole point of
 * the `sourceTrimFailed` gate (`modes/chained/useChainForm.ts`).
 *
 * The no-query cases double as the mock-side half of the "an untrimmed upload is
 * unchanged from before §1-6" guarantee.
 */
describe("createMockBridge — contract v10 (source trim query)", () => {
  it("accepts a query on backend.uploadFile and reports trimmed:true for the full pair", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const { status, body } = await bridge.request("backend.uploadFile", {
      kind: "video",
      filePath: "C:\\v\\clip.mp4",
      query: { trim_start_sec: "12.500", trim_duration_sec: "5.000" },
    });

    expect(status).toBe(200);
    const video = body as { video_id?: string; trimmed?: boolean };
    expect(typeof video.video_id).toBe("string");
    expect(video.trimmed).toBe(true);
  });

  it("reports trimmed:false when no query is passed at all (the pre-v10 call shape)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const { status, body } = await bridge.request("backend.uploadFile", {
      kind: "video",
      filePath: "C:\\v\\clip.mp4",
    });

    expect(status).toBe(200);
    expect((body as { trimmed?: boolean }).trimmed).toBe(false);
  });

  it("reports trimmed:false for a HALF pair — one parameter alone is not a trim request", async () => {
    // The server treats a lone parameter as "no trim requested" and stores the
    // file untouched, so a caller that somehow sent only one must see `false`.
    const bridge = createMockBridge({ delayMs: 0 });
    const { body } = await bridge.request("backend.uploadFile", {
      kind: "video",
      filePath: "C:\\v\\clip.mp4",
      query: { trim_start_sec: "1.000" },
    });

    expect((body as { trimmed?: boolean }).trimmed).toBe(false);
  });

  it("leaves the other upload kinds' response bodies untouched (video-only field)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const image = await bridge.request("backend.uploadFile", { kind: "image", filePath: "C:\\i\\a.png" });
    const audio = await bridge.request("backend.uploadFile", { kind: "audio", filePath: "C:\\a\\a.wav" });

    expect(image.body).not.toHaveProperty("trimmed");
    expect(audio.body).not.toHaveProperty("trimmed");
    expect((image.body as { image_id?: string }).image_id).toBeTypeOf("string");
    expect((audio.body as { audio_id?: string }).audio_id).toBeTypeOf("string");
  });

  it("a query does not disturb the id/round-trip contract (the upload still succeeds normally)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const withQuery = await bridge.request("backend.uploadFile", {
      kind: "video",
      filePath: "C:\\v\\clip.mp4",
      query: { trim_start_sec: "0.000", trim_duration_sec: "2.000" },
    });
    const withoutQuery = await bridge.request("backend.uploadFile", {
      kind: "video",
      filePath: "C:\\v\\clip.mp4",
    });

    // Same status, same shape, different ids (a fresh upload each time) — the
    // only difference is `trimmed`.
    expect(withQuery.status).toBe(withoutQuery.status);
    expect(Object.keys(withQuery.body as object).sort()).toEqual(Object.keys(withoutQuery.body as object).sort());
    expect((withQuery.body as { trimmed?: boolean }).trimmed).toBe(true);
    expect((withoutQuery.body as { trimmed?: boolean }).trimmed).toBe(false);
  });
});

/**
 * 契約v10のもう半分（`timeline.getSelection` の再生位置系6項目）。実機調査
 * （2026-08-01）で確定してnative側が常に返すようになったので、モックも同じ既定値を
 * 返す。既定はすべて「保守側」＝トリムしない方向であることを固定する。
 */
describe("createMockBridge — contract v10 (getSelection playback fields)", () => {
  it("reports all six playback fields with native's conservative defaults", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const selection = await bridge.request("timeline.getSelection", {});
    const item = selection.selected[0]!;

    // hasPlaybackRange false is what makes `decideSourceTrim` skip, so the mock's
    // default right-click flow uploads the whole file exactly as it did pre-v10.
    expect(item.hasPlaybackRange).toBe(false);
    expect(item.playbackStartSec).toBe(0);
    expect(item.playbackEndSec).toBe(0);
    // Neutral speed / no loop / one section: never a skip reason on their own.
    expect(item.playbackSpeed).toBe(1);
    expect(item.loopPlay).toBe(false);
    expect(item.sectionCount).toBe(1);
  });

  it("lets a test override the playback fields to drive a real trim", async () => {
    const bridge = createMockBridge({
      delayMs: 0,
      selection: {
        selected: [
          {
            layer: 1,
            frameStart: 0,
            frameEnd: 260,
            effectName: "動画ファイル",
            filePath: "C:\\v\\clip.mp4",
            objectName: "clip",
            textContent: null,
            mediaWidth: 1920,
            mediaHeight: 1080,
            mediaDurationSec: 10.7,
            hasPlaybackRange: true,
            playbackStartSec: 2,
            playbackEndSec: 10.7,
            playbackSpeed: 1,
            loopPlay: false,
            sectionCount: 1,
          },
        ],
      },
    });
    const selection = await bridge.request("timeline.getSelection", {});
    const item = selection.selected[0]!;
    expect(item.hasPlaybackRange).toBe(true);
    expect(item.playbackStartSec).toBe(2);
    expect(item.playbackEndSec).toBe(10.7);
  });
});
