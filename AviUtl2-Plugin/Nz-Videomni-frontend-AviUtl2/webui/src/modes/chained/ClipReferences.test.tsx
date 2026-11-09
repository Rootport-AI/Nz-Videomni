import { act, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { createMockBridge } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import type { ReferenceRules } from "../single/referenceRules";
import { ClipReferences } from "./ClipReferences";
import type { ClipReferenceEntry, ClipReferencesProps } from "./ClipReferences";

// §1-109: one Chained clip's Reference cards — registered with the screen
// under the clip id (Enable switched on once at mount), unregistered on
// unmount, and drawn only while `referencesOn`.

const RULES: ReferenceRules = {
  maxImages: 9,
  maxVideos: 3,
  maxAudios: 3,
  maxTotal: 12,
  videoTotalSec: 15,
  audioTotalSec: 15,
  clipMinSec: 2,
};

function renderClip(props: Partial<ClipReferencesProps> = {}) {
  const register = vi.fn<(clipId: string, entry: ClipReferenceEntry) => void>();
  const unregister = vi.fn<(clipId: string) => void>();
  const base: ClipReferencesProps = {
    clipId: "clip-a",
    rules: RULES,
    referencesOn: false,
    nativeBridge: createMockBridge({ delayMs: 0 }),
    disabled: false,
    onAppendTag: vi.fn(),
    register,
    unregister,
    ...props,
  };
  const view = render(
    <LanguageProvider>
      <ClipReferences {...base} />
    </LanguageProvider>,
  );
  const rerender = (next: Partial<ClipReferencesProps>) =>
    view.rerender(
      <LanguageProvider>
        <ClipReferences {...base} {...next} />
      </LanguageProvider>,
    );
  return { ...view, register, unregister, rerender };
}

describe("ClipReferences", () => {
  it("registers its slots under the clip id, and unregisters on unmount", async () => {
    const { register, unregister, unmount } = renderClip();
    await waitFor(() => expect(register).toHaveBeenCalled());
    const [clipId, entry] = register.mock.calls.at(-1)!;
    expect(clipId).toBe("clip-a");
    expect(entry.slots.picture).toHaveLength(9);
    expect(entry.slots.video).toHaveLength(3);
    expect(entry.slots.audio).toHaveLength(3);
    expect(entry.summary.filled.total).toBe(0);
    // A state change (Enable switched on at mount) re-registers without
    // unregistering first.
    expect(unregister).not.toHaveBeenCalled();

    unmount();
    expect(unregister).toHaveBeenLastCalledWith("clip-a");
  });

  it("draws nothing while off, and the embedded accordion (no Enable; no ON badge while empty) while on", () => {
    const { container, rerender } = renderClip();
    expect(container.querySelector(".ref-accordion")).toBeNull();
    rerender({ referencesOn: true });
    expect(container.querySelector(".ref-accordion")).not.toBeNull();
    expect(screen.queryByRole("checkbox", { name: /enable/i })).toBeNull();
    expect(screen.queryByText("【🔴ON】")).toBeNull();
    // Enable was switched on at mount: the cards are not greyed out.
    expect(container.querySelector(".ref-slot-ignored")).toBeNull();
  });

  it("does not register again when re-rendered with the same state", async () => {
    const { register, rerender } = renderClip();
    await waitFor(() => expect(register).toHaveBeenCalled());
    const calls = register.mock.calls.length;
    rerender({ disabled: true });
    rerender({ disabled: false });
    expect(register.mock.calls.length).toBe(calls);
  });

  // §1-120: the stable slot mutations the screen's distribution uses.
  it("registers stable ops; adopting through them re-registers the new state with the same ops", async () => {
    const { register, rerender } = renderClip();
    await waitFor(() => expect(register).toHaveBeenCalled());
    const first = register.mock.calls.at(-1)![1];
    rerender({ disabled: true });
    act(() => {
      first.ops.adopt("audio", 1, { id: "long", fileName: "long.wav", filePath: "C:\\long.wav", durationSec: 30 });
      first.ops.setA2vSlot(1);
    });
    await waitFor(() => expect(register.mock.calls.at(-1)![1].slots.audio[0]?.id).toBe("long"));
    const last = register.mock.calls.at(-1)![1];
    expect(last.ops).toBe(first.ops);
    expect(last.summary.slotUse.audio[0]?.isA2v).toBe(true);
  });

  it("draws Audio to video only when the screen passes `a2v`", () => {
    const a2v = {
      available: true,
      strength: 0.5,
      onStrengthChange: vi.fn(),
      strengthMin: 0,
      strengthMax: 1,
      strengthStep: 0.05,
      tooShortWarning: null,
    };
    const { rerender } = renderClip({ referencesOn: true });
    expect(screen.queryByRole("checkbox", { name: "Audio to video" })).toBeNull();
    rerender({ referencesOn: true, a2v });
    expect(screen.getAllByRole("checkbox", { name: "Audio to video" })).toHaveLength(3);
  });
});
