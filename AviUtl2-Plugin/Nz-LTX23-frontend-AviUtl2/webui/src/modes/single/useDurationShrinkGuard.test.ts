import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { KeyframeItem } from "./keyframeUtils";
import { useDurationShrinkGuard, type DurationShrinkGuardDeps } from "./useDurationShrinkGuard";

// multiple=8, offset=1 throughout (the server default — see keyframeUtils.ts's
// snapFrameIdx doc). maxPlaceablePosition(numFrames, 8, 1) values used below:
//   numFrames=41 -> 33   numFrames=25 -> 17   numFrames=17 -> 9
//   numFrames=9  -> 1    numFrames=49 -> 41   numFrames=81 -> 73
// A keyframe pinned at frameIdx=33 is therefore in-range at 41/49/81 and
// overflows at 25/17/9.
const MULTIPLE = 8;
const OFFSET = 1;

function makeItem(id: string, frameIdx: number): KeyframeItem {
  return {
    id,
    filePath: `C:/kf/${id}.png`,
    fileName: `${id}.png`,
    imageId: `img-${id}`,
    thumbnailDataUrl: null,
    frameIdx,
    strength: 0.8,
    status: "ready",
    errorCode: null,
  };
}

function makeDeps(overrides: Partial<DurationShrinkGuardDeps> = {}): DurationShrinkGuardDeps {
  return {
    numFrames: 41,
    setNumFrames: vi.fn(),
    applyPreset: vi.fn(),
    getPresetNumFrames: vi.fn(() => null),
    items: [],
    removeKeyframe: vi.fn(),
    multiple: MULTIPLE,
    offset: OFFSET,
    ...overrides,
  };
}

describe("useDurationShrinkGuard", () => {
  it("no-overflow commit: no pendingShrink is raised, and a subsequent cancel is a no-op", () => {
    const deps = makeDeps({ numFrames: 41, items: [] });
    const { result } = renderHook((props: DurationShrinkGuardDeps) => useDurationShrinkGuard(props), {
      initialProps: deps,
    });

    act(() => result.current.onDurationChange(33, true));
    expect(deps.setNumFrames).toHaveBeenCalledTimes(1);
    expect(deps.setNumFrames).toHaveBeenNthCalledWith(1, 33, true);

    act(() => result.current.onDurationCommit(33, true));
    expect(result.current.pendingShrink).toBeNull();

    // Per spec: verify indirectly — cancelShrink after a no-overflow commit
    // must not issue any restore call (preEditRef was cleared by the commit).
    act(() => result.current.cancelShrink());
    expect(deps.setNumFrames).toHaveBeenCalledTimes(1);
    expect(result.current.pendingShrink).toBeNull();
  });

  it("m2: a pre-existing out-of-range pin does NOT raise the modal on an unchanged-value commit (focus->blur with no edit)", () => {
    // frameIdx=33 is already stranded at numFrames=17 (maxPlaceable(17)=9) —
    // left over from something unrelated to this commit (e.g. a missed
    // arrow-key nudge elsewhere). No onDurationChange ever ran, so there is
    // no edit session in flight when the field is simply focused and blurred
    // without changing its value.
    const deps = makeDeps({ numFrames: 17, items: [makeItem("a", 33)] });
    const { result } = renderHook((props: DurationShrinkGuardDeps) => useDurationShrinkGuard(props), {
      initialProps: deps,
    });

    act(() => result.current.onDurationCommit(17, true));

    expect(result.current.pendingShrink).toBeNull();
    expect(deps.removeKeyframe).not.toHaveBeenCalled();
  });

  it("m2: a commit that doesn't shrink the value within its own edit session does NOT raise the modal, even with a stranded pin", () => {
    // An edit session runs (onDurationChange fires), but the committed value
    // is not smaller than the pre-edit value — e.g. the user nudges the
    // field down and back up before blurring. The stray pin at 33 already
    // overflows at 17, but this commit isn't what shrank anything.
    const deps = makeDeps({ numFrames: 17, items: [makeItem("a", 33)] });
    const { result } = renderHook((props: DurationShrinkGuardDeps) => useDurationShrinkGuard(props), {
      initialProps: deps,
    });

    act(() => result.current.onDurationChange(17, true));
    act(() => result.current.onDurationCommit(17, true));

    expect(result.current.pendingShrink).toBeNull();
    expect(deps.removeKeyframe).not.toHaveBeenCalled();
  });

  it("m2: a commit that DOES shrink the value this session still raises the modal as before", () => {
    const deps = makeDeps({ numFrames: 41, items: [makeItem("a", 33)] });
    const { result } = renderHook((props: DurationShrinkGuardDeps) => useDurationShrinkGuard(props), {
      initialProps: deps,
    });

    act(() => result.current.onDurationChange(17, true));
    act(() => result.current.onDurationCommit(17, true));

    expect(result.current.pendingShrink).toEqual({
      kind: "duration",
      pendingNumFrames: 17,
      overflowIds: ["a"],
    });
  });

  it("duration increase (value grows) passes through with no overflow", () => {
    const deps = makeDeps({ numFrames: 41, items: [makeItem("a", 33)] });
    const { result } = renderHook((props: DurationShrinkGuardDeps) => useDurationShrinkGuard(props), {
      initialProps: deps,
    });

    act(() => result.current.onDurationChange(81, true));
    act(() => result.current.onDurationCommit(81, true));

    expect(result.current.pendingShrink).toBeNull();
    expect(deps.removeKeyframe).not.toHaveBeenCalled();
  });

  it("shrink commit raises pendingShrink; confirmShrink removes the overflowing keyframes and clears state", () => {
    const deps = makeDeps({ numFrames: 41, items: [makeItem("a", 33)] });
    const { result } = renderHook((props: DurationShrinkGuardDeps) => useDurationShrinkGuard(props), {
      initialProps: deps,
    });

    act(() => result.current.onDurationChange(17, true));
    act(() => result.current.onDurationCommit(17, true));

    expect(result.current.pendingShrink).toEqual({
      kind: "duration",
      pendingNumFrames: 17,
      overflowIds: ["a"],
    });

    act(() => result.current.confirmShrink());

    expect(deps.removeKeyframe).toHaveBeenCalledTimes(1);
    expect(deps.removeKeyframe).toHaveBeenCalledWith("a");
    expect(result.current.pendingShrink).toBeNull();
    // Duration is already live-applied by onDurationChange; confirmShrink
    // must not issue a second setNumFrames call.
    expect(deps.setNumFrames).toHaveBeenCalledTimes(1);
  });

  it("shrink commit + cancelShrink restores numFrames to the pre-edit value", () => {
    const deps = makeDeps({ numFrames: 41, items: [makeItem("a", 33)] });
    const { result } = renderHook((props: DurationShrinkGuardDeps) => useDurationShrinkGuard(props), {
      initialProps: deps,
    });

    act(() => result.current.onDurationChange(17, true));
    act(() => result.current.onDurationCommit(17, true));
    act(() => result.current.cancelShrink());

    expect(deps.setNumFrames).toHaveBeenCalledTimes(2);
    expect(deps.setNumFrames).toHaveBeenNthCalledWith(1, 17, true);
    expect(deps.setNumFrames).toHaveBeenNthCalledWith(2, 41, false);
    expect(result.current.pendingShrink).toBeNull();
    expect(deps.removeKeyframe).not.toHaveBeenCalled();
  });

  it("two consecutive shrink-then-cancel sessions each restore to their OWN pre-edit value (preEditRef is re-captured, not stuck)", () => {
    const deps = makeDeps({ numFrames: 41, items: [makeItem("a", 33)] });
    const { result, rerender } = renderHook((props: DurationShrinkGuardDeps) => useDurationShrinkGuard(props), {
      initialProps: deps,
    });

    // Session 1: 41 -> 17 (overflow) -> cancel -> restores to 41.
    act(() => result.current.onDurationChange(17, true));
    act(() => result.current.onDurationCommit(17, true));
    act(() => result.current.cancelShrink());
    expect(deps.setNumFrames).toHaveBeenNthCalledWith(2, 41, false);

    // Simulate the restored value flowing back into the live form, AND the
    // baseline moving on to something else entirely (49) before the next
    // session starts — distinct from session 1's pre-edit value (41) so a
    // stale (uncleared) preEditRef would be caught red-handed.
    const deps2 = { ...deps, numFrames: 49 };
    rerender(deps2);

    // Session 2: 49 -> 9 (still overflows: maxPlaceable(9)=1 < 33) -> cancel
    // -> must restore to 49 (session 2's own pre-edit value), NOT 41.
    act(() => result.current.onDurationChange(9, true));
    act(() => result.current.onDurationCommit(9, true));
    expect(result.current.pendingShrink).toEqual({
      kind: "duration",
      pendingNumFrames: 9,
      overflowIds: ["a"],
    });
    act(() => result.current.cancelShrink());

    expect(deps.setNumFrames).toHaveBeenNthCalledWith(3, 9, true);
    expect(deps.setNumFrames).toHaveBeenNthCalledWith(4, 49, false);
    expect(result.current.pendingShrink).toBeNull();
  });

  it("frameIdx=0 is never treated as overflowing, even at the smallest duration", () => {
    const deps = makeDeps({ numFrames: 41, items: [makeItem("zero", 0), makeItem("b", 33)] });
    const { result } = renderHook((props: DurationShrinkGuardDeps) => useDurationShrinkGuard(props), {
      initialProps: deps,
    });

    act(() => result.current.onDurationChange(9, true));
    act(() => result.current.onDurationCommit(9, true));

    expect(result.current.pendingShrink).toEqual({
      kind: "duration",
      pendingNumFrames: 9,
      overflowIds: ["b"],
    });
  });

  it("duplicate commits (e.g. pointerup followed by blur) are idempotent", () => {
    const deps = makeDeps({ numFrames: 41, items: [makeItem("a", 33)] });
    const { result } = renderHook((props: DurationShrinkGuardDeps) => useDurationShrinkGuard(props), {
      initialProps: deps,
    });

    act(() => result.current.onDurationChange(17, true));
    act(() => result.current.onDurationCommit(17, true));
    const first = result.current.pendingShrink;
    act(() => result.current.onDurationCommit(17, true));
    const second = result.current.pendingShrink;

    expect(first).toEqual(second);
    expect(deps.removeKeyframe).not.toHaveBeenCalled();
    // Still only the one live-apply call from onDurationChange — re-running
    // onDurationCommit never itself calls setNumFrames.
    expect(deps.setNumFrames).toHaveBeenCalledTimes(1);
  });

  it("preset with no overflow applies immediately (no confirmation)", () => {
    const deps = makeDeps({
      numFrames: 41,
      items: [makeItem("a", 33)],
      getPresetNumFrames: vi.fn(() => 41),
    });
    const { result } = renderHook((props: DurationShrinkGuardDeps) => useDurationShrinkGuard(props), {
      initialProps: deps,
    });

    act(() => result.current.onApplyPreset("cinematic"));

    expect(deps.applyPreset).toHaveBeenCalledTimes(1);
    expect(deps.applyPreset).toHaveBeenCalledWith("cinematic");
    expect(result.current.pendingShrink).toBeNull();
  });

  it("preset that shrinks past a keyframe withholds applyPreset until confirmShrink, which then removes the stranded keyframes", () => {
    const deps = makeDeps({
      numFrames: 41,
      items: [makeItem("a", 33)],
      getPresetNumFrames: vi.fn(() => 17),
    });
    const { result } = renderHook((props: DurationShrinkGuardDeps) => useDurationShrinkGuard(props), {
      initialProps: deps,
    });

    act(() => result.current.onApplyPreset("short"));

    expect(deps.applyPreset).not.toHaveBeenCalled();
    expect(result.current.pendingShrink).toEqual({
      kind: "preset",
      presetName: "short",
      pendingNumFrames: 17,
      overflowIds: ["a"],
    });

    act(() => result.current.confirmShrink());

    expect(deps.applyPreset).toHaveBeenCalledTimes(1);
    expect(deps.applyPreset).toHaveBeenCalledWith("short");
    expect(deps.removeKeyframe).toHaveBeenCalledTimes(1);
    expect(deps.removeKeyframe).toHaveBeenCalledWith("a");
    expect(result.current.pendingShrink).toBeNull();
  });

  it("cancelling a pending preset shrink never applies the preset", () => {
    const deps = makeDeps({
      numFrames: 41,
      items: [makeItem("a", 33)],
      getPresetNumFrames: vi.fn(() => 17),
    });
    const { result } = renderHook((props: DurationShrinkGuardDeps) => useDurationShrinkGuard(props), {
      initialProps: deps,
    });

    act(() => result.current.onApplyPreset("short"));
    act(() => result.current.cancelShrink());

    expect(deps.applyPreset).not.toHaveBeenCalled();
    expect(deps.removeKeyframe).not.toHaveBeenCalled();
    expect(deps.setNumFrames).not.toHaveBeenCalled();
    expect(result.current.pendingShrink).toBeNull();
  });

  it("an unknown preset (getPresetNumFrames -> null) is a no-op", () => {
    const deps = makeDeps({ numFrames: 41, items: [makeItem("a", 33)], getPresetNumFrames: vi.fn(() => null) });
    const { result } = renderHook((props: DurationShrinkGuardDeps) => useDurationShrinkGuard(props), {
      initialProps: deps,
    });

    act(() => result.current.onApplyPreset("nonexistent"));

    expect(deps.applyPreset).not.toHaveBeenCalled();
    expect(result.current.pendingShrink).toBeNull();
  });
});
