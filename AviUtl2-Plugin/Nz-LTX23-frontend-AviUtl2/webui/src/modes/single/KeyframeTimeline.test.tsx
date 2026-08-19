import { fireEvent, render } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { KeyframeTimeline, type KeyframeTimelineProps } from "./KeyframeTimeline";
import type { KeyframeItem } from "./keyframeUtils";

// Grid: multiple=8, offset=1. With numFrames=41 the valid positions are
// {0, 1, 9, 17, 25, 33}; numFrames-1 = 40 and maxPlaceable = 33. The track is
// pinned to left=0/width=400, so a frame `f` maps to clientX `f*10` and
// positionToRatio(f) = f/40 (e.g. 9 -> "22.5%", 25 -> "62.5%", 33 -> "82.5%").
// The left-edge snap-to-zero zone is max(offsetPx/2, 14) = max(5, 14) = 14px
// here, so any clientX <= 14 lands on position 0.
const NUM_FRAMES = 41;

function makeItem(id: string, frameIdx: number, withThumb = true): KeyframeItem {
  return {
    id,
    filePath: `C:/tmp/${id}.png`,
    fileName: `${id}.png`,
    imageId: `img-${id}`,
    thumbnailDataUrl: withThumb ? "data:image/png;base64,AAAA" : null,
    frameIdx,
    strength: 0.8,
    status: "ready",
    errorCode: null,
  };
}

function renderTimeline(overrides: Partial<KeyframeTimelineProps> = {}) {
  const onCommitFrameIdx = vi.fn();
  const props: KeyframeTimelineProps = {
    items: [],
    numFrames: NUM_FRAMES,
    frameRate: 24,
    multiple: 8,
    offset: 1,
    disabled: false,
    onCommitFrameIdx,
    ...overrides,
  };
  const utils = render(
    <LanguageProvider>
      <KeyframeTimeline {...props} />
    </LanguageProvider>,
  );
  // Fix the track geometry so clientX <-> frame conversion is deterministic.
  const track = utils.container.querySelector(".kf-track") as HTMLElement;
  vi.spyOn(track, "getBoundingClientRect").mockReturnValue({
    left: 0,
    right: 400,
    top: 0,
    bottom: 8,
    width: 400,
    height: 8,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  } as DOMRect);
  return { ...utils, onCommitFrameIdx, track };
}

function barPins(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(".kf-pin-bar"));
}

describe("KeyframeTimeline", () => {
  it("draws bar pins at their positionToRatio left%", () => {
    const { container } = renderTimeline({ items: [makeItem("a", 9), makeItem("b", 25)] });
    const pins = barPins(container);
    expect(pins).toHaveLength(2);
    expect(pins[0]?.style.left).toBe("22.5%");
    expect(pins[1]?.style.left).toBe("62.5%");
  });

  it("draws the position-0 pin on the bar at left 0% with the square zero class", () => {
    const { container } = renderTimeline({ items: [makeItem("z", 0), makeItem("a", 9)] });
    const pins = barPins(container);
    expect(pins).toHaveLength(2); // position 0 is a bar pin, not a separate slot
    const zero = container.querySelector<HTMLElement>(".kf-pin-zero");
    expect(zero).not.toBeNull();
    expect(zero?.getAttribute("data-kf-pin-id")).toBe("z");
    expect(zero?.style.left).toBe("0%");
    expect(zero?.getAttribute("aria-valuenow")).toBe("0");
    // The non-zero pin keeps the ordinary (round) shape — no zero class.
    const nonZero = pins.find((p) => !p.classList.contains("kf-pin-zero"));
    expect(nonZero?.getAttribute("aria-valuenow")).toBe("9");
  });

  it("shows the resolved landing label mid-drag while other pins stay put", () => {
    const { container } = renderTimeline({ items: [makeItem("a", 9), makeItem("b", 25)] });
    const pins = barPins(container);
    const dragged = pins[0]!;
    const otherLeftBefore = pins[1]!.style.left;

    fireEvent.pointerDown(dragged, { pointerId: 1 });
    // Aim at frame 25 (occupied by "b") -> resolves forward to 33.
    fireEvent.pointerMove(dragged, { pointerId: 1, clientX: 250 });

    const draggingLabel = container.querySelector(".kf-pin-dragging .kf-pin-label");
    expect(draggingLabel?.textContent).toContain("33f");
    // The dragged pin itself has advanced to the resolved slot (33 -> 82.5%)…
    expect(container.querySelector<HTMLElement>(".kf-pin-dragging")?.style.left).toBe("82.5%");
    // …but the untouched neighbour hasn't moved.
    expect(barPins(container)[1]?.style.left).toBe(otherLeftBefore);
  });

  it("commits the forward-slid landing value when dropped onto an occupied tick", () => {
    const items = [makeItem("a", 9), makeItem("b", 17)];
    const { container, onCommitFrameIdx } = renderTimeline({ items });
    const dragged = barPins(container)[0]!;

    fireEvent.pointerDown(dragged, { pointerId: 1 });
    fireEvent.pointerMove(dragged, { pointerId: 1, clientX: 170 }); // frame 17 (taken) -> 25
    fireEvent.pointerUp(dragged, { pointerId: 1, clientX: 170 });

    expect(onCommitFrameIdx).toHaveBeenCalledWith("a", 25);
  });

  it("allows a pin to be dragged past a later pin (overtaking)", () => {
    const items = [makeItem("a", 9), makeItem("b", 25)];
    const { container, onCommitFrameIdx } = renderTimeline({ items });
    const dragged = barPins(container)[0]!;

    fireEvent.pointerDown(dragged, { pointerId: 1 });
    fireEvent.pointerMove(dragged, { pointerId: 1, clientX: 330 }); // frame 33, past "b"@25
    fireEvent.pointerUp(dragged, { pointerId: 1, clientX: 330 });

    expect(onCommitFrameIdx).toHaveBeenCalledWith("a", 33);
  });

  it("snaps a drop within the left-edge zone to 0 when position 0 is free", () => {
    const items = [makeItem("a", 9)];
    const { container, onCommitFrameIdx } = renderTimeline({ items });
    const dragged = barPins(container)[0]!;

    fireEvent.pointerDown(dragged, { pointerId: 1 });
    // clientX 12 is inside the 14px snap-to-zero zone (not merely <= the edge).
    fireEvent.pointerMove(dragged, { pointerId: 1, clientX: 12 });
    fireEvent.pointerUp(dragged, { pointerId: 1, clientX: 12 });

    expect(onCommitFrameIdx).toHaveBeenCalledWith("a", 0);
  });

  it("resolves a left-edge drop to 1 when position 0 is already taken", () => {
    const items = [makeItem("z", 0), makeItem("a", 9)];
    const { container, onCommitFrameIdx } = renderTimeline({ items });
    // Drag the non-zero pin toward the left edge; desired 0 is taken -> 1.
    const dragged = barPins(container).find((p) => !p.classList.contains("kf-pin-zero"))!;

    fireEvent.pointerDown(dragged, { pointerId: 1 });
    fireEvent.pointerMove(dragged, { pointerId: 1, clientX: 8 }); // desired 0 (taken) -> 1
    fireEvent.pointerUp(dragged, { pointerId: 1, clientX: 8 });

    expect(onCommitFrameIdx).toHaveBeenCalledWith("a", 1);
  });

  it("draws both the 0 and 1 pins with the zero pin distinguished, each operable", () => {
    const { container, onCommitFrameIdx } = renderTimeline({ items: [makeItem("z", 0), makeItem("a", 1)] });
    const pins = barPins(container);
    expect(pins).toHaveLength(2);

    const zero = container.querySelector<HTMLElement>(".kf-pin-zero")!;
    const one = pins.find((p) => !p.classList.contains("kf-pin-zero"))!;
    // Both occupy the near-overlapping left edge; the zero pin carries the class
    // (and thus the raised z-index / square shape) that keeps it on top.
    expect(zero.style.left).toBe("0%");
    expect(one.style.left).toBe("2.5%"); // 1/40
    expect(one.classList.contains("kf-pin-zero")).toBe(false);

    // The zero pin is independently steppable (0 -> jumps over occupied 1 -> 9).
    fireEvent.keyDown(zero, { key: "ArrowRight" });
    expect(onCommitFrameIdx).toHaveBeenLastCalledWith("z", 9);
    // …and so is the position-1 pin (1 -> 9, since 0 is behind it).
    fireEvent.keyDown(one, { key: "ArrowRight" });
    expect(onCommitFrameIdx).toHaveBeenLastCalledWith("a", 9);
  });

  it("steps one grid tick per arrow key and reaches 0", () => {
    const { container, onCommitFrameIdx } = renderTimeline({ items: [makeItem("a", 9)] });
    const pin = barPins(container)[0]!;

    fireEvent.keyDown(pin, { key: "ArrowRight" });
    expect(onCommitFrameIdx).toHaveBeenLastCalledWith("a", 17);

    fireEvent.keyDown(pin, { key: "ArrowLeft" });
    expect(onCommitFrameIdx).toHaveBeenLastCalledWith("a", 1);

    // From frame 1 a left step reaches the start position 0.
    const { container: c2, onCommitFrameIdx: commit2 } = renderTimeline({ items: [makeItem("a", 1)] });
    fireEvent.keyDown(barPins(c2)[0]!, { key: "ArrowLeft" });
    expect(commit2).toHaveBeenLastCalledWith("a", 0);
  });

  it("jumps an arrow-key step over a run of occupied ticks", () => {
    const items = [makeItem("a", 1), makeItem("b", 9), makeItem("c", 17)];
    const { container, onCommitFrameIdx } = renderTimeline({ items });
    const pinA = barPins(container)[0]!; // frame 1

    fireEvent.keyDown(pinA, { key: "ArrowRight" }); // 9 & 17 taken -> 25
    expect(onCommitFrameIdx).toHaveBeenCalledWith("a", 25);
  });

  it("does not commit when a pin is already at the last placeable tick", () => {
    const { container, onCommitFrameIdx } = renderTimeline({ items: [makeItem("a", 33)] });
    fireEvent.keyDown(barPins(container)[0]!, { key: "ArrowRight" });
    expect(onCommitFrameIdx).not.toHaveBeenCalled();
  });

  it("renders the non-placeable grey band starting at maxPlaceable's ratio", () => {
    const { container } = renderTimeline({ items: [] });
    const deadZone = container.querySelector<HTMLElement>(".kf-deadzone");
    expect(deadZone).not.toBeNull();
    expect(deadZone?.style.left).toBe("82.5%"); // maxPlaceable = 33 -> 33/40
  });

  it("aborts a drag on pointercancel without committing (M1)", () => {
    const { container, onCommitFrameIdx } = renderTimeline({ items: [makeItem("a", 9), makeItem("b", 25)] });
    const dragged = barPins(container)[0]!;

    fireEvent.pointerDown(dragged, { pointerId: 1 });
    fireEvent.pointerMove(dragged, { pointerId: 1, clientX: 250 });
    expect(container.querySelector(".kf-pin-dragging")).not.toBeNull();

    fireEvent.pointerCancel(dragged, { pointerId: 1 });

    expect(container.querySelector(".kf-pin-dragging")).toBeNull(); // drag state cleared
    expect(onCommitFrameIdx).not.toHaveBeenCalled(); // reverted, not committed
    // The pin snapped back to its pre-drag slot.
    expect(barPins(container)[0]?.style.left).toBe("22.5%");
  });

  it("keeps labels visible for well-spaced pins including position 0 (M2)", () => {
    // numFrames 41 with pins at 0 and 33 — far apart at the fallback width.
    const { container } = renderTimeline({ items: [makeItem("z", 0), makeItem("b", 33)] });
    for (const pin of barPins(container)) {
      expect(pin.classList.contains("kf-pin-labelled")).toBe(true);
    }
    // The zero pin is a first-class label target now (was excluded before).
    expect(container.querySelector(".kf-pin-zero")?.classList.contains("kf-pin-labelled")).toBe(true);
  });

  it("degrades labels to hover/focus/drag for a crowded cluster (M2)", () => {
    // numFrames 81 packs adjacent grid ticks well under the 64px threshold.
    const { container } = renderTimeline({
      numFrames: 81,
      items: [makeItem("a", 1), makeItem("b", 9), makeItem("c", 17)],
    });
    for (const pin of barPins(container)) {
      expect(pin.classList.contains("kf-pin-labelled")).toBe(false);
    }
  });

  it("keeps focus on a pin as the arrow keys move it 0<->1 within the bar (m6)", () => {
    // A controlled harness so committing actually moves the pin between frames.
    // With every pin on one keyed list, 0<->1 is an in-place reorder (React
    // reuses the DOM node by `item.id`), so focus survives without any restore
    // hack — this test guards that the removed focusRestoreId stays unnecessary.
    function Harness() {
      const [items, setItems] = useState<KeyframeItem[]>([makeItem("a", 1)]);
      return (
        <KeyframeTimeline
          items={items}
          numFrames={NUM_FRAMES}
          frameRate={24}
          multiple={8}
          offset={1}
          disabled={false}
          onCommitFrameIdx={(id, frameIdx) =>
            setItems((prev) => prev.map((i) => (i.id === id ? { ...i, frameIdx } : i)))
          }
        />
      );
    }
    const { container } = render(
      <LanguageProvider>
        <Harness />
      </LanguageProvider>,
    );

    const pin = container.querySelector<HTMLElement>('[data-kf-pin-id="a"]')!;
    pin.focus();

    fireEvent.keyDown(pin, { key: "ArrowLeft" }); // 1 -> 0
    const atZero = container.querySelector<HTMLElement>('[data-kf-pin-id="a"]')!;
    expect(atZero.getAttribute("aria-valuenow")).toBe("0");
    expect(atZero.classList.contains("kf-pin-zero")).toBe(true);
    expect(document.activeElement).toBe(atZero);

    fireEvent.keyDown(atZero, { key: "ArrowRight" }); // 0 -> 1
    const atOne = container.querySelector<HTMLElement>('[data-kf-pin-id="a"]')!;
    expect(atOne.getAttribute("aria-valuenow")).toBe("1");
    expect(atOne.classList.contains("kf-pin-zero")).toBe(false);
    expect(document.activeElement).toBe(atOne);
  });

  it("ignores drag and keyboard entirely when disabled", () => {
    const { container, onCommitFrameIdx } = renderTimeline({
      items: [makeItem("a", 9), makeItem("b", 25)],
      disabled: true,
    });
    const pin = barPins(container)[0]!;
    expect(pin.tabIndex).toBe(-1);

    fireEvent.pointerDown(pin, { pointerId: 1 });
    fireEvent.pointerMove(pin, { pointerId: 1, clientX: 250 });
    fireEvent.pointerUp(pin, { pointerId: 1, clientX: 250 });
    fireEvent.keyDown(pin, { key: "ArrowRight" });

    expect(onCommitFrameIdx).not.toHaveBeenCalled();
    // No pin entered the dragging state.
    expect(container.querySelector(".kf-pin-dragging")).toBeNull();
  });
});
