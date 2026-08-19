import { describe, expect, it } from "vitest";
import { IMAGE_SHARED, rejudgeRows, scanToRows, type BatchRow, type ScannedFile } from "./manifestMerge";

const FPS = 24;

/** Same raw 8n+1 formula as `gradio_ui.manifest.raw_frame_count` /
 * `modes/chained/chainUtils.ts`'s `vLatentFrames` derivation — written out
 * locally (not imported) per this sprint's rule against importing
 * chainUtils.ts's frame-count helpers while it's under concurrent edit. */
function rawFramesFor(durationSec: number, fps: number): number {
  return Math.floor((Math.floor(durationSec * fps) - 1) / 8) * 8 + 1;
}

/** Stand-in "final frame count" policy for these tests — identical to
 * `rawFramesFor` here. The real shrink/clamp policy is deliberately out of
 * this module's scope (spec §9 / `gradio_ui.handlers.suggest_frames_for_audio`),
 * so scanToRows only needs SOME injected callback to exercise the Waiting
 * path; its exact arithmetic is irrelevant to what this module is
 * responsible for. */
function framesFor(durationSec: number, fps: number): number {
  return rawFramesFor(durationSec, fps);
}

function file(name: string, mtimeMs: number, durationSec = 0): ScannedFile {
  return { name, path: `C:\\wav\\${name}`, sizeBytes: 1000, mtimeMs, durationSec };
}

describe("scanToRows", () => {
  it("mtime昇順でWaiting行を生成し、queueを1から振る", () => {
    const files = [file("b.wav", 200, 2.0), file("a.wav", 100, 1.0)];
    const rows = scanToRows(files, { fps: FPS, framesFor, rawFramesFor });
    expect(rows.map((r) => r.wav)).toEqual(["a.wav", "b.wav"]);
    expect(rows.map((r) => r.queue)).toEqual([1, 2]);
    expect(rows.every((r) => r.stat === "Waiting")).toBe(true);
    expect(rows.every((r) => r.image === IMAGE_SHARED)).toBe(true);
  });

  it("非wav、または長さ取得不能(duration<=0)はwav-only-alphaでSkip", () => {
    const files = [file("x.mp3", 100, 3.0), file("y.wav", 200, 0)];
    const rows = scanToRows(files, { fps: FPS, framesFor, rawFramesFor });
    expect(rows).toHaveLength(2);
    expect(rows.every((r) => r.stat === "Skip" && r.skipReason === "wav-only-alpha")).toBe(true);
    expect(rows.every((r) => r.duration === 0)).toBe(true);
  });

  it("生フレーム数が481を超えるとover-capでSkip", () => {
    // 24fps・25秒 -> raw frame count 593 > 481.
    const files = [file("long.wav", 100, 25.0)];
    const rows = scanToRows(files, { fps: FPS, framesFor, rawFramesFor });
    expect(rows[0]?.stat).toBe("Skip");
    expect(rows[0]?.skipReason).toBe("over-cap");
    expect(rows[0]?.duration).toBe(25.0);
  });

  it("maxFrames指定時は可変上限でSkip判定される(raw=297 vs raw=185, maxFrames=257)", () => {
    // 24fps・12.5s -> raw frame count 297 > 257 (over cap).
    // 24fps・8.0s  -> raw frame count 185 <= 257 (still valid).
    const files = [file("over.wav", 100, 12.5), file("under.wav", 200, 8.0)];
    const rows = scanToRows(files, { fps: FPS, framesFor, rawFramesFor, maxFrames: 257 });
    expect(rows[0]?.stat).toBe("Skip");
    expect(rows[0]?.skipReason).toBe("over-cap");
    expect(rows[1]?.stat).toBe("Waiting");
  });

  it("maxFrames省略時は従来どおり481が上限", () => {
    // 24fps・25s -> raw frame count 593 > 481 (no maxFrames passed).
    const files = [file("long.wav", 100, 25.0)];
    const rows = scanToRows(files, { fps: FPS, framesFor, rawFramesFor });
    expect(rows[0]?.stat).toBe("Skip");
    expect(rows[0]?.skipReason).toBe("over-cap");
  });

  it("maxFrames=9999のような過大値は481へ内部クランプされる", () => {
    // 24fps・25s -> raw frame count 593 > 481, still Skip even though
    // maxFrames itself is far above the server hard cap.
    const files = [file("long.wav", 100, 25.0)];
    const rows = scanToRows(files, { fps: FPS, framesFor, rawFramesFor, maxFrames: 9999 });
    expect(rows[0]?.stat).toBe("Skip");
    expect(rows[0]?.skipReason).toBe("over-cap");
  });

  it(".tmp・許可外拡張子(.csv/.png等)は除外される", () => {
    const files = [
      file("batch_a2v_manifest.csv", 50),
      file("batch_a2v_manifest.csv.tmp", 70),
      file("cover.png", 80),
      file("keep.wav", 90, 1.0),
    ];
    const rows = scanToRows(files, { fps: FPS, framesFor, rawFramesFor });
    expect(rows.map((r) => r.wav)).toEqual(["keep.wav"]);
  });
});

/** Builds a fully-populated `BatchRow` for `rejudgeRows` tests, with
 * everything defaulted except the fields a given test cares about. */
function row(overrides: Partial<BatchRow> & Pick<BatchRow, "queue">): BatchRow {
  return {
    wav: `${overrides.queue}.wav`,
    duration: 1.0,
    image: IMAGE_SHARED,
    prompt: "",
    stat: "Waiting",
    output: "",
    frames: 0,
    skipReason: "",
    error: "",
    ...overrides,
  };
}

describe("rejudgeRows", () => {
  it("有効なWaiting行はWaitingのままframesが更新される", () => {
    // 24fps・1.0s -> raw/framesFor both well under 481.
    const rows = [row({ queue: 1, stat: "Waiting", duration: 1.0, frames: 999 })];
    const out = rejudgeRows(rows, { fps: FPS, framesFor, rawFramesFor });
    expect(out[0]?.stat).toBe("Waiting");
    expect(out[0]?.frames).toBe(framesFor(1.0, FPS));
    expect(out[0]?.skipReason).toBe("");
  });

  it("481超のWaiting行(🔁で戻したstale skipReason付き)はSkipへ強制されframes=0になる", () => {
    // 24fps・25s -> raw frame count 593 > 481.
    const rows = [
      row({ queue: 1, stat: "Waiting", duration: 25.0, frames: 481, skipReason: "over-cap" }),
    ];
    const out = rejudgeRows(rows, { fps: FPS, framesFor, rawFramesFor });
    expect(out[0]?.stat).toBe("Skip");
    expect(out[0]?.skipReason).toBe("over-cap");
    expect(out[0]?.frames).toBe(0);
  });

  it("Failed行が481超でも同様にSkipへ強制される", () => {
    const rows = [row({ queue: 1, stat: "Failed", duration: 25.0, error: "boom" })];
    const out = rejudgeRows(rows, { fps: FPS, framesFor, rawFramesFor });
    expect(out[0]?.stat).toBe("Skip");
    expect(out[0]?.skipReason).toBe("over-cap");
    expect(out[0]?.frames).toBe(0);
  });

  it("Failed行が有効ならFailedのままframesだけ更新される", () => {
    const rows = [row({ queue: 1, stat: "Failed", duration: 1.0, error: "boom", frames: 999 })];
    const out = rejudgeRows(rows, { fps: FPS, framesFor, rawFramesFor });
    expect(out[0]?.stat).toBe("Failed");
    expect(out[0]?.error).toBe("boom");
    expect(out[0]?.frames).toBe(framesFor(1.0, FPS));
    expect(out[0]?.skipReason).toBe("");
  });

  it("duration<=0はWaiting/Failedどちらでもwav-only-alphaでSkipへ強制される", () => {
    const rows = [
      row({ queue: 1, stat: "Waiting", duration: 0 }),
      row({ queue: 2, stat: "Failed", duration: 0, error: "boom" }),
    ];
    const out = rejudgeRows(rows, { fps: FPS, framesFor, rawFramesFor });
    expect(out.every((r) => r.stat === "Skip" && r.skipReason === "wav-only-alpha")).toBe(true);
    expect(out.every((r) => r.frames === 0)).toBe(true);
  });

  it("Done/Skip/Generating行は完全無変更(参照等価)のまま返る", () => {
    const done = row({ queue: 1, stat: "Done", output: "1.mp4" });
    const skip = row({ queue: 2, stat: "Skip", skipReason: "wav-only-alpha" });
    const generating = row({ queue: 3, stat: "Generating" });
    const rows = [done, skip, generating];
    const out = rejudgeRows(rows, { fps: FPS, framesFor, rawFramesFor });
    expect(out[0]).toBe(done);
    expect(out[1]).toBe(skip);
    expect(out[2]).toBe(generating);
  });

  it("順序・件数・queueが保持される", () => {
    const rows = [
      row({ queue: 1, stat: "Done" }),
      row({ queue: 2, stat: "Waiting", duration: 1.0 }),
      row({ queue: 3, stat: "Failed", duration: 25.0 }),
      row({ queue: 4, stat: "Skip", skipReason: "wav-only-alpha" }),
    ];
    const out = rejudgeRows(rows, { fps: FPS, framesFor, rawFramesFor });
    expect(out).toHaveLength(4);
    expect(out.map((r) => r.queue)).toEqual([1, 2, 3, 4]);
  });

  it("【差別化】481判定はframesForではなくrawFramesForを見る", () => {
    // Deliberately divergent mocks: rawFramesFor reports over-limit (593),
    // framesFor reports a clamped in-range value (481). If the gate ever
    // mistakenly switched to framesFor, this row would stay Waiting instead
    // of being forced to Skip.
    const divergentRawFramesFor = () => 593;
    const divergentFramesFor = () => 481;
    const rows = [row({ queue: 1, stat: "Waiting", duration: 25.0 })];
    const out = rejudgeRows(rows, {
      fps: FPS,
      framesFor: divergentFramesFor,
      rawFramesFor: divergentRawFramesFor,
    });
    expect(out[0]?.stat).toBe("Skip");
    expect(out[0]?.skipReason).toBe("over-cap");
    expect(out[0]?.frames).toBe(0);
  });

  it("maxFrames指定時は可変上限でSkip判定される(raw=297 vs raw=185, maxFrames=257)", () => {
    // 24fps・12.5s -> raw frame count 297 > 257 (over cap, though well under 481).
    // 24fps・8.0s  -> raw frame count 185 <= 257 (still valid).
    const rows = [
      row({ queue: 1, stat: "Waiting", duration: 12.5 }),
      row({ queue: 2, stat: "Waiting", duration: 8.0 }),
    ];
    const out = rejudgeRows(rows, { fps: FPS, framesFor, rawFramesFor, maxFrames: 257 });
    expect(out[0]?.stat).toBe("Skip");
    expect(out[0]?.skipReason).toBe("over-cap");
    expect(out[0]?.frames).toBe(0);
    expect(out[1]?.stat).toBe("Waiting");
  });

  it("maxFrames省略時は従来どおり481が上限(raw=297はmaxFrames省略なら有効なまま)", () => {
    // Same 12.5s/raw=297 row as above, but with no maxFrames -> default 481
    // cap keeps it Waiting instead of Skip.
    const rows = [row({ queue: 1, stat: "Waiting", duration: 12.5 })];
    const out = rejudgeRows(rows, { fps: FPS, framesFor, rawFramesFor });
    expect(out[0]?.stat).toBe("Waiting");
    expect(out[0]?.skipReason).toBe("");
  });

  it("maxFrames=9999のような過大値は481へ内部クランプされる", () => {
    // 24fps・25s -> raw frame count 593 > 481, still forced to Skip even
    // though maxFrames itself is far above the server hard cap.
    const rows = [row({ queue: 1, stat: "Waiting", duration: 25.0 })];
    const out = rejudgeRows(rows, { fps: FPS, framesFor, rawFramesFor, maxFrames: 9999 });
    expect(out[0]?.stat).toBe("Skip");
    expect(out[0]?.skipReason).toBe("over-cap");
    expect(out[0]?.frames).toBe(0);
  });

  it("【差別化】maxFrames併用時もSkip判定はframesForではなくrawFramesForを見る", () => {
    // Same divergent-mock setup as above, but with a custom maxFrames to
    // confirm the cap and the raw-vs-clamped gate compose correctly.
    const divergentRawFramesFor = () => 300;
    const divergentFramesFor = () => 257;
    const rows = [row({ queue: 1, stat: "Waiting", duration: 12.5 })];
    const out = rejudgeRows(rows, {
      fps: FPS,
      framesFor: divergentFramesFor,
      rawFramesFor: divergentRawFramesFor,
      maxFrames: 257,
    });
    expect(out[0]?.stat).toBe("Skip");
    expect(out[0]?.skipReason).toBe("over-cap");
    expect(out[0]?.frames).toBe(0);
  });
});
