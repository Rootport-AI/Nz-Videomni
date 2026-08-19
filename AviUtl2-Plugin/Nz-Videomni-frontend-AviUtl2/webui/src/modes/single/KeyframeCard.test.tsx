import { useState } from "react";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { KeyframeCard } from "./KeyframeCard";
import type { KeyframeCardProps } from "./KeyframeCard";
import type { KeyframeItem } from "./keyframeUtils";

function makeItem(overrides: Partial<KeyframeItem> = {}): KeyframeItem {
  return {
    id: "kf-1",
    filePath: "C:\\Users\\mock\\Pictures\\a.png",
    fileName: "a.png",
    imageId: "mock-image-1",
    thumbnailDataUrl: null,
    frameIdx: 9,
    strength: 0.8,
    status: "ready",
    errorCode: null,
    ...overrides,
  };
}

/** Wraps `KeyframeCard` with just enough local state to mirror what a real
 * parent (`useKeyframes` + the timeline bar) does: apply a committed
 * `frameIdx` back onto the item before the next render. Without this, a bare
 * static `item` prop never advances past its initial value, and the card's
 * own optimistic `setFrameDraft` in `commitFrame` gets clobbered by its
 * resync effect on the very next render (the effect can't tell "the parent
 * caught up" apart from "someone else changed it out from under us" if the
 * prop never moves at all). */
function Harness({
  initialItem,
  onCommitFrameIdx,
  ...rest
}: { initialItem: KeyframeItem } & Omit<KeyframeCardProps, "item">) {
  const [item, setItem] = useState(initialItem);
  return (
    <ul>
      <KeyframeCard
        {...rest}
        item={item}
        onCommitFrameIdx={(id, frameIdx) => {
          onCommitFrameIdx(id, frameIdx);
          setItem((prev) => ({ ...prev, frameIdx }));
        }}
      />
    </ul>
  );
}

function renderCard(overrides: Partial<KeyframeCardProps> = {}) {
  const onCommitFrameIdx = vi.fn();
  const onStrengthChange = vi.fn();
  const onRemove = vi.fn();
  const onReplaceFile = vi.fn();
  const onPickFileFor = vi.fn();
  const onCaptureInto = vi.fn();
  const { item: itemOverride, ...rest } = overrides;
  const item = itemOverride ?? makeItem();
  const cardProps: Omit<KeyframeCardProps, "item"> = {
    numFrames: 49,
    frameRate: 24,
    multiple: 8,
    offset: 1,
    occupiedOthers: new Set<number>(),
    disabled: false,
    isCapturing: false,
    isPicking: false,
    onCommitFrameIdx,
    onStrengthChange,
    onRemove,
    onReplaceFile,
    onPickFileFor,
    onCaptureInto,
    ...rest,
  };
  const utils = render(
    <LanguageProvider>
      <Harness initialItem={item} {...cardProps} />
    </LanguageProvider>,
  );
  return {
    ...utils,
    onCommitFrameIdx,
    onStrengthChange,
    onRemove,
    onReplaceFile,
    onPickFileFor,
    onCaptureInto,
    item,
  };
}

describe("KeyframeCard", () => {
  it("typing a FRAME value shows a willLandAt hint and commits the resolved value on Enter", () => {
    const { container, onCommitFrameIdx, item } = renderCard();
    const frameInput = container.querySelector('input[type="number"]')!;

    // multiple=8, offset=1: snapFrameIdx(20) -> floor(19/8)*8+1 = 17, and no
    // other card occupies 17, so it lands exactly there.
    fireEvent.change(frameInput, { target: { value: "20" } });
    expect(screen.getByText(/17/)).toBeInTheDocument();

    fireEvent.keyDown(frameInput, { key: "Enter" });
    expect(onCommitFrameIdx).toHaveBeenCalledWith(item.id, 17);
    expect((frameInput as HTMLInputElement).value).toBe("17");
  });

  it("resolves a collision by sliding forward to the next free grid position, and commits on blur", () => {
    const { container, onCommitFrameIdx, item } = renderCard({ occupiedOthers: new Set([17]) });
    const frameInput = container.querySelector('input[type="number"]')!;

    // Desired (17) is taken by another card; the next free grid position
    // (offset=1, multiple=8) is 25.
    fireEvent.change(frameInput, { target: { value: "20" } });
    expect(screen.getByText(/25/)).toBeInTheDocument();

    fireEvent.blur(frameInput);
    expect(onCommitFrameIdx).toHaveBeenCalledWith(item.id, 25);
    expect((frameInput as HTMLInputElement).value).toBe("25");
  });

  it("STRENGTH slider change calls onStrengthChange with the item id", () => {
    const { container, onStrengthChange, item } = renderCard();
    const strengthInput = container.querySelector('input[type="range"]')!;

    fireEvent.change(strengthInput, { target: { value: "0.55" } });
    expect(onStrengthChange).toHaveBeenCalledWith(item.id, 0.55);
  });

  it("renders the three action buttons in ❌📁📸 order (remove, choose, capture)", () => {
    renderCard();
    const buttons = screen.getAllByRole("button");
    expect(buttons).toHaveLength(3);
    expect(buttons[0]).toHaveAccessibleName(/remove/i);
    expect(buttons[1]).toHaveAccessibleName(/choose image file/i);
    expect(buttons[2]).toHaveAccessibleName(/capture/i);
  });

  it("remove button calls onRemove with the item id", () => {
    const { onRemove, item } = renderCard();

    fireEvent.click(screen.getByRole("button", { name: /remove/i }));
    expect(onRemove).toHaveBeenCalledWith(item.id);
  });

  it("choose-file button calls onPickFileFor with the item id", () => {
    const { onPickFileFor, item } = renderCard();

    fireEvent.click(screen.getByRole("button", { name: /choose image file/i }));
    expect(onPickFileFor).toHaveBeenCalledWith(item.id);
  });

  it("capture button calls onCaptureInto with the item id (never adds a new card)", () => {
    const { onCaptureInto, item } = renderCard();

    fireEvent.click(screen.getByRole("button", { name: /capture/i }));
    expect(onCaptureInto).toHaveBeenCalledWith(item.id);
  });

  it("disables choose and capture while the item is uploading", () => {
    renderCard({ item: makeItem({ status: "uploading", imageId: null }) });

    expect(screen.getByRole("button", { name: /choose image file/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /capture/i })).toBeDisabled();
  });

  it("disables only the capture button while isCapturing", () => {
    renderCard({ isCapturing: true });

    expect(screen.getByRole("button", { name: /capture/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /choose image file/i })).not.toBeDisabled();
  });

  it("disables only the choose-file button while isPicking", () => {
    renderCard({ isPicking: true });

    expect(screen.getByRole("button", { name: /choose image file/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /capture/i })).not.toBeDisabled();
  });

  it("shows the start-frame-fixed badge only when frameIdx === 0", () => {
    const { rerender, container } = render(
      <LanguageProvider>
        <ul>
          <KeyframeCard
            item={makeItem({ frameIdx: 0 })}
            numFrames={49}
            frameRate={24}
            multiple={8}
            offset={1}
            occupiedOthers={new Set<number>()}
            disabled={false}
            isCapturing={false}
            isPicking={false}
            onCommitFrameIdx={vi.fn()}
            onStrengthChange={vi.fn()}
            onRemove={vi.fn()}
            onReplaceFile={vi.fn()}
            onPickFileFor={vi.fn()}
            onCaptureInto={vi.fn()}
          />
        </ul>
      </LanguageProvider>,
    );
    expect(container.querySelector(".kfc-badge-fixed")).toBeInTheDocument();

    rerender(
      <LanguageProvider>
        <ul>
          <KeyframeCard
            item={makeItem({ frameIdx: 9 })}
            numFrames={49}
            frameRate={24}
            multiple={8}
            offset={1}
            occupiedOthers={new Set<number>()}
            disabled={false}
            isCapturing={false}
            isPicking={false}
            onCommitFrameIdx={vi.fn()}
            onStrengthChange={vi.fn()}
            onRemove={vi.fn()}
            onReplaceFile={vi.fn()}
            onPickFileFor={vi.fn()}
            onCaptureInto={vi.fn()}
          />
        </ul>
      </LanguageProvider>,
    );
    expect(container.querySelector(".kfc-badge-fixed")).not.toBeInTheDocument();
  });

  it("shows the empty-card hint for status:'empty'", () => {
    renderCard({ item: makeItem({ status: "empty", imageId: null, fileName: "", filePath: "" }) });

    expect(screen.getByText(/no image yet/i)).toBeInTheDocument();
  });

  it("shows the error code for status:'error'", () => {
    renderCard({ item: makeItem({ status: "error", errorCode: "BACKEND_UNREACHABLE", imageId: null }) });

    expect(screen.getByText("BACKEND_UNREACHABLE")).toBeInTheDocument();
  });

  it("accepts a supported image dropped anywhere on the card and calls onReplaceFile with the resolved path", async () => {
    const { container, onReplaceFile, item } = renderCard();
    const card = container.querySelector(".kfc-card")!;

    fireEvent.drop(card, { dataTransfer: { files: [new File([], "new-keyframe.png")] } });

    await waitFor(() =>
      expect(onReplaceFile).toHaveBeenCalledWith(item.id, expect.stringContaining("new-keyframe.png"), "new-keyframe.png"),
    );
  });

  it("rejects an unsupported dropped extension, shows an inline error, and never calls onReplaceFile", async () => {
    const { container, onReplaceFile } = renderCard();
    const card = container.querySelector(".kfc-card")!;

    fireEvent.drop(card, { dataTransfer: { files: [new File([], "notes.txt")] } });

    await screen.findByText(/unsupported file type/i);
    expect(onReplaceFile).not.toHaveBeenCalled();
    // The error must render inside the card itself (not delegated away).
    expect(within(card as HTMLElement).getByText(/unsupported file type/i)).toBeInTheDocument();
  });
});
