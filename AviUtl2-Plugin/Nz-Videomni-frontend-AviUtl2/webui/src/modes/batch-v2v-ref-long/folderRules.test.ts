import { describe, expect, it } from "vitest";
import type { UploadLimits } from "../../api/types";
import type { ReferenceRules } from "../single/referenceRules";
import {
  baseFolderPath,
  canAddField,
  createDefaultFrame,
  deriveV2vRefLongOutDir,
  emptyFieldsOf,
  fieldSlot,
  filledFolderCount,
  frameCounts,
  frameIndexForClip,
  outFileNameOf,
  pickBaseField,
  rowCountOf,
  sameScanInputs,
  uploadRuleOf,
  withAddedField,
} from "./folderRules";
import type { FolderFrame, ScanInputs, ScanResult, ScannedField, ScannedFile } from "./types";

const RULES: ReferenceRules = {
  maxImages: 9,
  maxVideos: 3,
  maxAudios: 3,
  maxTotal: 12,
  videoTotalSec: 15,
  audioTotalSec: 15,
  clipMinSec: 1,
};

function files(dir: string, names: string[]): ScannedFile[] {
  return names.map((name) => ({ name, path: `${dir}\\${name}`, sizeBytes: 100, durationSec: null }));
}

function field(kind: ScannedField["kind"], slot: number, path: string, names: string[]): ScannedField {
  return { kind, slot, path, files: files(path, names) };
}

function scanOf(source: string[], frames: ScannedField[][], videoDir = "D:\\src"): ScanResult {
  return {
    inputs: { videoDir, frames: [] },
    source: files(videoDir, source),
    frames,
    rules: RULES,
  };
}

describe("folderRules: frames and fields", () => {
  it("the default frame is Picture, Picture, Video, Audio", () => {
    expect(createDefaultFrame(1).fields.map((f) => f.kind)).toEqual(["picture", "picture", "video", "audio"]);
  });

  it("fieldSlot is the position among the same kind (= tag number)", () => {
    const frame = createDefaultFrame(1);
    expect(fieldSlot(frame, 1)).toBe(1);
    expect(fieldSlot(frame, 2)).toBe(2);
    expect(fieldSlot(frame, 3)).toBe(1);
    expect(fieldSlot(frame, 4)).toBe(1);
    expect(fieldSlot(frame, 99)).toBeNull();
  });

  it("an added field goes after the last field of its kind", () => {
    const frame = withAddedField(withAddedField(createDefaultFrame(1), "video"), "picture");
    expect(frame.fields.map((f) => f.kind)).toEqual(["picture", "picture", "picture", "video", "video", "audio"]);
    const added = frame.fields[2];
    expect(added && fieldSlot(frame, added.id)).toBe(3);
  });

  it("frameCounts counts only fields with a path", () => {
    const frame: FolderFrame = {
      id: 1,
      fields: [
        { id: 1, kind: "picture", path: "A" },
        { id: 2, kind: "picture", path: "  " },
        { id: 3, kind: "video", path: "V" },
        { id: 4, kind: "audio", path: "" },
      ],
    };
    expect(frameCounts(frame)).toEqual({ picture: 1, video: 1, audio: 0, total: 2 });
    expect(filledFolderCount([frame, { ...frame, id: 2 }])).toBe(4);
  });

  it("canAddField reads the per-kind and total limits from the rules", () => {
    let frame = createDefaultFrame(1);
    expect(canAddField(frame, "video", RULES)).toBe(true);
    frame = withAddedField(withAddedField(frame, "video"), "video");
    expect(canAddField(frame, "video", RULES)).toBe(false);
    expect(canAddField(frame, "video", { ...RULES, maxVideos: 4 })).toBe(true);
    // total: 6 fields now; a total limit of 6 stops any kind
    expect(canAddField(frame, "picture", { ...RULES, maxTotal: 6 })).toBe(false);
    expect(canAddField(frame, "picture", RULES)).toBe(true);
  });
});

describe("folderRules: frameIndexForClip", () => {
  it("clip c uses frame c; past the frames, the last one only while fillFollowing", () => {
    expect(frameIndexForClip(2, true, 0)).toBe(0);
    expect(frameIndexForClip(2, true, 1)).toBe(1);
    expect(frameIndexForClip(2, true, 3)).toBe(1);
    expect(frameIndexForClip(2, false, 1)).toBe(1);
    expect(frameIndexForClip(2, false, 2)).toBeNull();
    expect(frameIndexForClip(0, true, 0)).toBeNull();
  });
});

describe("folderRules: the base field and the row count", () => {
  it("picks the field with the most files", () => {
    const scan = scanOf(["a.mp4"], [[field("picture", 1, "P1", ["1.png", "2.png"])]]);
    expect(pickBaseField(scan)).toEqual({ type: "ref", frame: 0, kind: "picture", slot: 1 });
    expect(rowCountOf(scan)).toBe(2);
    expect(baseFolderPath(scan)).toBe("P1");
  });

  it("on a tie: opening video, then Picture 1, 2…, Video, Audio, then the next frame", () => {
    const two = ["1", "2"];
    expect(pickBaseField(scanOf(two, [[field("picture", 1, "P1", two)]]))).toEqual({ type: "source" });
    expect(
      pickBaseField(
        scanOf([], [[field("audio", 1, "A1", two), field("video", 1, "V1", two), field("picture", 2, "P2", two)]]),
      ),
    ).toEqual({ type: "ref", frame: 0, kind: "picture", slot: 2 });
    expect(
      pickBaseField(scanOf([], [[field("audio", 1, "A1", two), field("video", 1, "V1", two)]])),
    ).toEqual({ type: "ref", frame: 0, kind: "video", slot: 1 });
    expect(
      pickBaseField(scanOf([], [[field("audio", 1, "A1", two)], [field("picture", 1, "P", two)]])),
    ).toEqual({ type: "ref", frame: 0, kind: "audio", slot: 1 });
  });

  it("counts every frame, whatever the clip count (a later frame can be the base)", () => {
    const scan = scanOf(["a"], [[field("picture", 1, "P", ["1"])], [], [field("video", 1, "D:\\clip3", ["1", "2", "3"])]]);
    expect(pickBaseField(scan)).toEqual({ type: "ref", frame: 2, kind: "video", slot: 1 });
    expect(rowCountOf(scan)).toBe(3);
  });

  it("every field empty: no base, no rows", () => {
    const scan = scanOf([], [[field("picture", 1, "P", [])]]);
    expect(pickBaseField(scan)).toBeNull();
    expect(rowCountOf(scan)).toBe(0);
    expect(baseFolderPath(scan)).toBeNull();
    expect(outFileNameOf(scan, 0)).toBeNull();
  });

  it("emptyFieldsOf lists the empty fields, the opening video only when given", () => {
    const scan = scanOf([], [[field("picture", 1, "P", []), field("video", 1, "V", ["x"])]]);
    expect(emptyFieldsOf(scan)).toEqual([{ type: "source" }, { type: "ref", frame: 0, kind: "picture", slot: 1 }]);
    const noVideoDir = { ...scan, inputs: { videoDir: "", frames: [] } };
    expect(emptyFieldsOf(noVideoDir)).toEqual([{ type: "ref", frame: 0, kind: "picture", slot: 1 }]);
  });
});

describe("folderRules: output names", () => {
  it("the output file is the base file's name with .mp4", () => {
    const scan = scanOf(["clip.a.mov", "b.mp4"], []);
    expect(outFileNameOf(scan, 0)).toBe("clip.a.mp4");
    expect(outFileNameOf(scan, 1)).toBe("b.mp4");
    expect(outFileNameOf(scan, 2)).toBeNull();
  });

  it("the output folder is a `_out` sibling of the base folder", () => {
    expect(deriveV2vRefLongOutDir("D:\\work\\videos")).toBe("D:\\work\\videos_out");
    expect(deriveV2vRefLongOutDir("D:\\work\\videos\\")).toBe("D:\\work\\videos_out");
    expect(deriveV2vRefLongOutDir("videos")).toBe("videos_out");
  });
});

describe("folderRules: sameScanInputs", () => {
  const inputs = (): ScanInputs => ({
    videoDir: "D:\\v",
    frames: [
      {
        id: 1,
        fields: [
          { id: 1, kind: "picture", path: "D:\\p" },
          { id: 2, kind: "picture", path: "" },
          { id: 3, kind: "video", path: "" },
          { id: 4, kind: "audio", path: "" },
        ],
      },
    ],
  });

  it("adding an empty field does not change the inputs", () => {
    const b = inputs();
    const frame = b.frames[0];
    if (!frame) throw new Error("frame");
    b.frames[0] = withAddedField(frame, "video");
    expect(sameScanInputs(inputs(), b)).toBe(true);
  });

  it("surrounding spaces are not a change", () => {
    const b = inputs();
    b.videoDir = " D:\\v ";
    expect(sameScanInputs(inputs(), b)).toBe(true);
  });

  it("the frame count, a path, a field position and the video folder are changes", () => {
    const more = inputs();
    more.frames.push(createDefaultFrame(2));
    expect(sameScanInputs(inputs(), more)).toBe(false);

    const path = inputs();
    path.frames[0]?.fields.splice(0, 1, { id: 1, kind: "picture", path: "D:\\q" });
    expect(sameScanInputs(inputs(), path)).toBe(false);

    const moved = inputs();
    moved.frames[0]?.fields.splice(0, 2, { id: 1, kind: "picture", path: "" }, { id: 2, kind: "picture", path: "D:\\p" });
    expect(sameScanInputs(inputs(), moved)).toBe(false);

    const video = inputs();
    video.videoDir = "D:\\w";
    expect(sameScanInputs(inputs(), video)).toBe(false);
  });
});

describe("folderRules: uploadRuleOf", () => {
  const upload: UploadLimits = {
    max_image_size_mb: 20,
    allowed_image_extensions: [".png"],
    max_video_size_mb: 200,
    allowed_video_extensions: [".mp4"],
    max_audio_size_mb: 150,
    allowed_audio_extensions: [".wav", ".mp4"],
  };
  it("maps each field kind to its upload list and cap", () => {
    expect(uploadRuleOf("source", upload)).toEqual({ extensions: [".mp4"], maxMb: 200 });
    expect(uploadRuleOf("video", upload)).toEqual({ extensions: [".mp4"], maxMb: 200 });
    expect(uploadRuleOf("picture", upload)).toEqual({ extensions: [".png"], maxMb: 20 });
    expect(uploadRuleOf("audio", upload)).toEqual({ extensions: [".wav", ".mp4"], maxMb: 150 });
  });
});
