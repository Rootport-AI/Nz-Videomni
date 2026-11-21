import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { createMockBridge } from "../../bridge/mockBridge";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import { useChainForm } from "./useChainForm";

// Batch v2v-ref-long: `batchClipFloorRelaxed` lowers only the remove button's
// floor (`clipRemovalFloor`). The manual Generate's count check (`minClips`,
// `isClipCountValid`, the `minClips` reason) still reads the screen's own
// material, and turning the flag off never re-pads the clip list.

function setup(initialRelaxed: boolean | undefined) {
  const mockBridge = createMockBridge({ delayMs: 0 });
  return renderHook(
    ({ relaxed }: { relaxed: boolean | undefined }) =>
      useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", {
        nativeBridge: mockBridge,
        ...(relaxed !== undefined ? { batchClipFloorRelaxed: relaxed } : {}),
      }),
    { initialProps: { relaxed: initialRelaxed } },
  );
}

function removeSecondClip(result: { current: ReturnType<typeof useChainForm> }) {
  act(() => {
    const id = result.current.clips[1]?.id;
    if (id) result.current.removeClip(id);
  });
}

describe("useChainForm — batchClipFloorRelaxed (Batch v2v-ref-long)", () => {
  it("relaxed with no material: the removal floor is 1, the Generate floor stays 2", () => {
    const { result } = setup(true);
    expect(result.current.clipRemovalFloor).toBe(1);
    expect(result.current.minClips).toBe(2);
    expect(result.current.clips).toHaveLength(2);

    removeSecondClip(result);
    expect(result.current.clips).toHaveLength(1);
    expect(result.current.isClipCountValid).toBe(false);
    expect(result.current.validityReasons).toContain("minClips");
    expect(result.current.isValid).toBe(false);
  });

  it.each([
    ["omitted", undefined],
    ["false", false],
  ] as const)("%s: the removal floor equals minClips (2 without material, 1 with a source video)", async (_label, relaxed) => {
    const { result } = setup(relaxed);
    expect(result.current.clipRemovalFloor).toBe(2);
    expect(result.current.minClips).toBe(2);

    act(() => {
      void result.current.sourceVideo.pick();
    });
    await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
    expect(result.current.minClips).toBe(1);
    expect(result.current.clipRemovalFloor).toBe(1);
  });

  it("turning the flag off with a single clip leaves the clip count alone", () => {
    const { result, rerender } = setup(true);
    removeSecondClip(result);
    expect(result.current.clips).toHaveLength(1);

    rerender({ relaxed: false });
    expect(result.current.clips).toHaveLength(1);
    expect(result.current.clipRemovalFloor).toBe(2);
    expect(result.current.isClipCountValid).toBe(false);
  });

  it("while relaxed, attaching then clearing a source video does not pad the clips back to 2", async () => {
    const { result } = setup(true);
    removeSecondClip(result);
    expect(result.current.clips).toHaveLength(1);

    act(() => {
      void result.current.sourceVideo.pick();
    });
    await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
    expect(result.current.clips).toHaveLength(1);

    act(() => result.current.sourceVideo.clear());
    expect(result.current.hasSourceVideo).toBe(false);
    expect(result.current.clips).toHaveLength(1);
  });

  it("not relaxed, clearing a source video still pads the clips back to 2", async () => {
    const { result } = setup(false);
    act(() => {
      void result.current.sourceVideo.pick();
    });
    await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
    removeSecondClip(result);
    expect(result.current.clips).toHaveLength(1);

    act(() => result.current.sourceVideo.clear());
    expect(result.current.clips).toHaveLength(2);
  });
});
