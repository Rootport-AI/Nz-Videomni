import { useState } from "react";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { NativeBridge } from "../../bridge";
import { createMockBridge } from "../../bridge/mockBridge";
import { LanguageProvider, useStrings } from "../../i18n/LanguageContext";
import { DurationField } from "./CommonGenerationFields";
import { FALLBACK_APP_CONFIG } from "./defaultConfig";
import { findOverflowFrameIdxs } from "./keyframeGrid";
import { KeyframeShrinkModal } from "./KeyframeShrinkModal";
import { KeyframesPanel } from "./KeyframesPanel";
import { useDurationShrinkGuard } from "./useDurationShrinkGuard";
import { useKeyframes } from "./useKeyframes";

// Integration harness: the REAL `useKeyframes` hook driving the REAL
// `KeyframesPanel` (timeline bar + cards + add row) over a mock bridge — the
// same wiring `GenerationForm` gives it, minus the surrounding form. `multiple`/
// `offset` are the fixture config's 8/1, so with numFrames=49 the placeable
// grid is {0, 1, 9, 17, 25, 33, 41} (maxPlaceable = 41).
const MULTIPLE = FALLBACK_APP_CONFIG.limits.conditioning_frame_idx_multiple;
const OFFSET = FALLBACK_APP_CONFIG.limits.conditioning_keyframe_grid_offset;

function PanelHarness({
  nativeBridge,
  numFrames = 49,
}: {
  nativeBridge: NativeBridge;
  numFrames?: number;
}) {
  const keyframes = useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge });
  return (
    <KeyframesPanel
      keyframes={keyframes}
      numFrames={numFrames}
      frameRate={24}
      frameIdxMultiple={MULTIPLE}
      gridOffset={OFFSET}
      disabled={false}
    />
  );
}

function renderPanel(numFrames = 49) {
  const nativeBridge = createMockBridge({ delayMs: 0 });
  const utils = render(
    <LanguageProvider>
      <PanelHarness nativeBridge={nativeBridge} numFrames={numFrames} />
    </LanguageProvider>,
  );
  return { ...utils, nativeBridge };
}

/** The add row is down to a single full-width "Add keyframe" button — capture
 * and choose-file moved onto each `KeyframeCard` itself. */
function addButton(): HTMLButtonElement {
  return screen.getByRole("button", { name: /add keyframe/i }) as HTMLButtonElement;
}

/** Every card's FRAME `<input type="number">` in DOM (sorted) order. */
function cardFrameInputs(container: HTMLElement): HTMLInputElement[] {
  return Array.from(container.querySelectorAll<HTMLInputElement>('.kfc-card input[type="number"]'));
}

/** Every timeline pin's `aria-valuenow`, in DOM order. The dedicated
 * start-frame slot is gone (2026-07-18 follow-up) — every pin, including
 * position 0, is just a `role="slider"` on the shared bar, so this asserts
 * against the semantic role rather than any location-specific class name.
 * Scoped to `.kf-timeline` (via its `role="group"`) rather than `screen`
 * because each card's own STRENGTH `<input type="range">` is *also* an
 * implicit `role="slider"` and would otherwise get swept in too. */
function pinValues(container: HTMLElement): string[] {
  const timeline = container.querySelector(".kf-timeline") as HTMLElement;
  return within(timeline)
    .getAllByRole("slider")
    .map((pin) => pin.getAttribute("aria-valuenow") ?? "");
}

describe("KeyframesPanel (integration)", () => {
  it("empty state: 'Add keyframe' places an empty card at position 0 and shows a pin there", () => {
    const { container } = renderPanel();
    // Nothing placed yet: no cards, no pins.
    expect(container.querySelector(".kfc-card")).toBeNull();
    expect(screen.queryAllByRole("slider")).toHaveLength(0);

    fireEvent.click(addButton());

    // One empty card (its "no image yet" hint), and the timeline now has
    // exactly one pin, at position 0.
    expect(container.querySelectorAll(".kfc-card")).toHaveLength(1);
    expect(screen.getByText(/no image yet/i)).toBeInTheDocument();
    expect(pinValues(container)).toEqual(["0"]);
  });

  it("second add lands at the rounded midpoint (25) while the first pin stays at 0", () => {
    const { container } = renderPanel();

    fireEvent.click(addButton()); // -> 0
    fireEvent.click(addButton()); // occupied {0}: midpoint of [0,48] rounds to 25

    // Cards render sorted ascending by frameIdx: [0, 25].
    expect(cardFrameInputs(container).map((i) => i.value)).toEqual(["0", "25"]);
    // Both pins are present, at 0 and 25 (order on the bar, not assertion order).
    expect(pinValues(container).slice().sort()).toEqual(["0", "25"]);
  });

  it("no room after the last placeable slot: the add button disables and shows the no-room hint", () => {
    const { container } = renderPanel();

    // Add one card and push it to the last placeable slot (41). From there the
    // grid has no free position after it, so `nextAddPosition` is null even
    // though the item-count cap (10) isn't reached.
    fireEvent.click(addButton());
    const frameInput = cardFrameInputs(container)[0]!;
    fireEvent.change(frameInput, { target: { value: "41" } });
    fireEvent.blur(frameInput);
    expect(frameInput.value).toBe("41");

    expect(addButton()).toBeDisabled();
    expect(screen.getByText(/no room left/i)).toBeInTheDocument();
  });

  it("counter clamps 'max' to the placeable slot count (numFrames=9 -> only {0,1} -> '0 / 2')", () => {
    renderPanel(9);
    // maxItems is 10, but numFrames=9 leaves only positions {0, 1}, so the
    // effective cap the counter shows is 2, not 10.
    expect(screen.getByText("0 / 2")).toBeInTheDocument();
  });

  it("editing a card's FRAME re-places its timeline pin (two-way sync)", () => {
    const { container } = renderPanel();
    fireEvent.click(addButton());

    // Newly added at 0 -> retarget it to 17.
    const frameInput = cardFrameInputs(container)[0]!;
    fireEvent.change(frameInput, { target: { value: "17" } });
    fireEvent.keyDown(frameInput, { key: "Enter" });

    // Still exactly one pin, now at 17.
    expect(pinValues(container)).toEqual(["17"]);
  });

  it("a card's capture button (📸) round-trips through keyframes.captureIntoCard, clearing the empty-card hint", async () => {
    const { container } = renderPanel();
    fireEvent.click(addButton());
    expect(screen.getByText(/no image yet/i)).toBeInTheDocument();

    const card = container.querySelector(".kfc-card") as HTMLElement;
    fireEvent.click(within(card).getByRole("button", { name: /capture/i }));

    // `captureIntoCard` uploads + thumbnails through the mock bridge and
    // settles the card into "ready", clearing the empty-card hint — proof the
    // Panel wired `onCaptureInto`/`isCapturing`/`isPicking` to the real
    // `useKeyframes` fields rather than dead props.
    await waitFor(() => expect(screen.queryByText(/no image yet/i)).not.toBeInTheDocument());
  });
});

// ---------------------------------------------------------------------------
// SingleScreen-level integration, verified via a thin harness rather than the
// full screen (which needs Jobs/Toast/config-fetch providers + a polling
// ledger). This wires the REAL `useKeyframes` + `useDurationShrinkGuard` +
// `KeyframeShrinkModal` + `DurationField` + the same `findOverflowFrameIdxs`
// Generate防壁 expression `SingleScreen` uses, exercising the guard→modal→
// apply/cancel flow and the out-of-range Generate blocker end to end.
// ---------------------------------------------------------------------------

function GuardHarness({ nativeBridge }: { nativeBridge: NativeBridge }) {
  const strings = useStrings();
  const [numFrames, setNumFrames] = useState(49);
  const keyframes = useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge });

  const guard = useDurationShrinkGuard({
    numFrames,
    setNumFrames: (value) => setNumFrames(value),
    applyPreset: () => {},
    getPresetNumFrames: () => null,
    items: keyframes.items,
    removeKeyframe: keyframes.remove,
    multiple: MULTIPLE,
    offset: OFFSET,
  });

  const overflow = findOverflowFrameIdxs(keyframes.items, numFrames, MULTIPLE, OFFSET);
  const hasOverflow = overflow.length > 0;

  return (
    <>
      <KeyframesPanel
        keyframes={keyframes}
        numFrames={numFrames}
        frameRate={24}
        frameIdxMultiple={MULTIPLE}
        gridOffset={OFFSET}
        disabled={false}
      />
      <div data-testid="duration">
        <DurationField
          label="Duration"
          value={numFrames}
          min={9}
          max={481}
          disabled={false}
          onChange={guard.onDurationChange}
          onCommit={guard.onDurationCommit}
          hint=""
        />
      </div>
      <button type="button" disabled={hasOverflow}>
        Generate
      </button>
      {hasOverflow && <p className="warning-banner">{strings.single.keyframes.outOfRangeGenerateHint}</p>}
      {guard.pendingShrink && (
        <KeyframeShrinkModal
          pendingNumFrames={guard.pendingShrink.pendingNumFrames}
          overflowCount={guard.pendingShrink.overflowIds.length}
          onApply={guard.confirmShrink}
          onCancel={guard.cancelShrink}
        />
      )}
    </>
  );
}

function renderGuardHarness() {
  const nativeBridge = createMockBridge({ delayMs: 0 });
  const utils = render(
    <LanguageProvider>
      <GuardHarness nativeBridge={nativeBridge} />
    </LanguageProvider>,
  );
  return { ...utils, nativeBridge };
}

/** Adds one keyframe and moves it to frame 41 (the last placeable slot for
 * numFrames=49), so a later shrink to 9 strands it off the grid. */
function seedKeyframeAt41(container: HTMLElement) {
  fireEvent.click(screen.getByRole("button", { name: /add keyframe/i }));
  const frameInput = container.querySelector('.kfc-card input[type="number"]') as HTMLInputElement;
  fireEvent.change(frameInput, { target: { value: "41" } });
  fireEvent.blur(frameInput);
  expect(frameInput.value).toBe("41");
}

/** Commits a new duration through the DurationField's number input (a
 * fireEvent.change carries no `inputType`, so it reads as a stepper edit ->
 * snaps + commits immediately). */
function commitDuration(container: HTMLElement, value: string) {
  const durationInput = within(container.querySelector('[data-testid="duration"]') as HTMLElement).getByRole("spinbutton");
  fireEvent.change(durationInput, { target: { value } });
}

describe("KeyframesPanel + duration-shrink guard (SingleScreen wiring)", () => {
  it("shrinking below a keyframe raises the modal, blocks Generate, and Apply deletes the stranded keyframe", () => {
    const { container } = renderGuardHarness();
    seedKeyframeAt41(container);
    // At numFrames=49 the pin at 41 is in range: Generate is enabled.
    expect(screen.getByRole("button", { name: /generate/i })).toBeEnabled();

    commitDuration(container, "9");

    // The shrink guard raised its confirmation modal…
    expect(screen.getByText(/shorten duration/i)).toBeInTheDocument();
    // …and the final防壁 blocks Generate with its reason while the pin is
    // out of range (the duration was live-applied to 9 already).
    expect(screen.getByRole("button", { name: /generate/i })).toBeDisabled();
    expect(screen.getByText(/beyond the current duration/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /^apply$/i }));

    // The stranded keyframe is gone, the modal dismissed, Generate re-enabled.
    expect(container.querySelector(".kfc-card")).toBeNull();
    expect(screen.queryByText(/shorten duration/i)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /generate/i })).toBeEnabled();
  });

  it("cancelling the shrink restores the pre-edit duration and keeps the keyframe", () => {
    const { container } = renderGuardHarness();
    seedKeyframeAt41(container);

    commitDuration(container, "9");
    expect(screen.getByText(/shorten duration/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /^cancel$/i }));

    // Duration is restored to 49 (so the pin at 41 is in range again), the
    // keyframe survives, the modal is gone, and Generate is enabled.
    const durationInput = within(container.querySelector('[data-testid="duration"]') as HTMLElement).getByRole("spinbutton");
    expect((durationInput as HTMLInputElement).value).toBe("49");
    expect(container.querySelectorAll(".kfc-card")).toHaveLength(1);
    expect(screen.queryByText(/shorten duration/i)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /generate/i })).toBeEnabled();
  });
});
