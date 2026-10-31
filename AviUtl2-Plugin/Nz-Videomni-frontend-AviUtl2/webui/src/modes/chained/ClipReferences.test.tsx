import { render, screen, waitFor } from "@testing-library/react";
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

  it("draws nothing while off, and the embedded accordion (no Enable) while on", () => {
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
});
