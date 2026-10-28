import { afterEach, describe, expect, it, vi } from "vitest";
import {
  dispatchCreateCommand,
  resetCreateLiveCommands,
  subscribeCreateCommands,
  type CreateLiveCommand,
} from "./createLiveCommands";

// The live-command channel from AppShell.handleRoute to the mounted Create
// screen (RIGHTCLICK_REDESIGN_SPEC.md §3-6 #10). A tiny module-level pub/sub —
// these tests pin its dispatch/subscribe/unsubscribe contract in isolation.

afterEach(() => {
  resetCreateLiveCommands();
});

describe("createLiveCommands", () => {
  it("delivers a dispatched command to a subscriber", () => {
    const received: CreateLiveCommand[] = [];
    subscribeCreateCommands((command) => received.push(command));

    dispatchCreateCommand({ type: "captureFrameToKeyframe" });

    expect(received).toEqual([{ type: "captureFrameToKeyframe" }]);
  });

  it("delivers to every current subscriber (fan-out)", () => {
    const a = vi.fn();
    const b = vi.fn();
    subscribeCreateCommands(a);
    subscribeCreateCommands(b);

    dispatchCreateCommand({ type: "captureFrameToKeyframe" });

    expect(a).toHaveBeenCalledTimes(1);
    expect(b).toHaveBeenCalledTimes(1);
    expect(a).toHaveBeenCalledWith({ type: "captureFrameToKeyframe" });
  });

  it("stops delivering after unsubscribe", () => {
    const listener = vi.fn();
    const unsubscribe = subscribeCreateCommands(listener);

    dispatchCreateCommand({ type: "captureFrameToKeyframe" });
    expect(listener).toHaveBeenCalledTimes(1);

    unsubscribe();
    dispatchCreateCommand({ type: "captureFrameToKeyframe" });
    expect(listener).toHaveBeenCalledTimes(1); // no further calls
  });

  it("a dispatch with no subscribers is a harmless no-op", () => {
    expect(() => dispatchCreateCommand({ type: "captureFrameToKeyframe" })).not.toThrow();
  });

  it("unsubscribing one listener leaves the others attached", () => {
    const a = vi.fn();
    const b = vi.fn();
    const unsubscribeA = subscribeCreateCommands(a);
    subscribeCreateCommands(b);

    unsubscribeA();
    dispatchCreateCommand({ type: "captureFrameToKeyframe" });

    expect(a).not.toHaveBeenCalled();
    expect(b).toHaveBeenCalledTimes(1);
  });

  it("resetCreateLiveCommands drops all subscribers", () => {
    const listener = vi.fn();
    subscribeCreateCommands(listener);

    resetCreateLiveCommands();
    dispatchCreateCommand({ type: "captureFrameToKeyframe" });

    expect(listener).not.toHaveBeenCalled();
  });

  // I11 §3-4 #5: the addImageKeyframe variant carries the resolved file path and
  // display name for the live keyframe append.
  it("delivers the #5 addImageKeyframe variant with its filePath/fileName payload", () => {
    const received: CreateLiveCommand[] = [];
    subscribeCreateCommands((command) => received.push(command));

    dispatchCreateCommand({ type: "addImageKeyframe", filePath: "C:\\i\\a.png", fileName: "a.png" });

    expect(received).toEqual([{ type: "addImageKeyframe", filePath: "C:\\i\\a.png", fileName: "a.png" }]);
  });

  it("delivers a #5 addImageKeyframe with a null filePath (unresolved source)", () => {
    const received: CreateLiveCommand[] = [];
    subscribeCreateCommands((command) => received.push(command));

    dispatchCreateCommand({ type: "addImageKeyframe", filePath: null, fileName: "" });

    expect(received).toEqual([{ type: "addImageKeyframe", filePath: null, fileName: "" }]);
  });

  // W5 #11: the addCaptureAsKeyframe variant is payload-free (the subscriber
  // captures the frame and picks the tail slot itself) and is DISTINCT from the
  // #10 captureFrameToKeyframe (which lands at frame 0).
  it("delivers the #11 addCaptureAsKeyframe variant (payload-free, distinct from #10)", () => {
    const received: CreateLiveCommand[] = [];
    subscribeCreateCommands((command) => received.push(command));

    dispatchCreateCommand({ type: "addCaptureAsKeyframe" });

    expect(received).toEqual([{ type: "addCaptureAsKeyframe" }]);
  });

  // W8 #9/#10: the setDuration variant carries the target num_frames for the
  // keep-panel-state DURATION write.
  it("delivers the W8 setDuration variant with its numFrames payload", () => {
    const received: CreateLiveCommand[] = [];
    subscribeCreateCommands((command) => received.push(command));

    dispatchCreateCommand({ type: "setDuration", numFrames: 481 });

    expect(received).toEqual([{ type: "setDuration", numFrames: 481 }]);
  });

  // §1-107: the two Reference variants (H3's #2 / #7) carry the file and, for a
  // video, the ribbon's trim window and trimmed length.
  it("delivers the §1-107 addReferenceVideo variant with its trim payload", () => {
    const received: CreateLiveCommand[] = [];
    subscribeCreateCommands((command) => received.push(command));

    const command: CreateLiveCommand = {
      type: "addReferenceVideo",
      filePath: "C:/v/a.mp4",
      fileName: "a.mp4",
      trimQuery: { trim_start_sec: "1.000", trim_duration_sec: "3.000" },
      durationSec: 3,
    };
    dispatchCreateCommand(command);

    expect(received).toEqual([command]);
  });

  it("delivers the §1-107 addReferenceAudio variant (null filePath allowed)", () => {
    const received: CreateLiveCommand[] = [];
    subscribeCreateCommands((command) => received.push(command));

    dispatchCreateCommand({ type: "addReferenceAudio", filePath: null, fileName: "" });

    expect(received).toEqual([{ type: "addReferenceAudio", filePath: null, fileName: "" }]);
  });
});
