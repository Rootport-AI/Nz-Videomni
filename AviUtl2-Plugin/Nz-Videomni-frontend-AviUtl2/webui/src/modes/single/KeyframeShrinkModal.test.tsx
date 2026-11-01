import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { KeyframeShrinkModal } from "./KeyframeShrinkModal";
import type { KeyframeShrinkModalProps } from "./KeyframeShrinkModal";

function renderModal(overrides: Partial<KeyframeShrinkModalProps> = {}) {
  const onApply = vi.fn();
  const onCancel = vi.fn();
  const props: KeyframeShrinkModalProps = {
    pendingNumFrames: 121,
    overflowCount: 3,
    onApply,
    onCancel,
    ...overrides,
  };
  const utils = render(
    <LanguageProvider>
      <KeyframeShrinkModal {...props} />
    </LanguageProvider>,
  );
  return { ...utils, onApply, onCancel };
}

describe("KeyframeShrinkModal", () => {
  it("embeds pendingNumFrames and overflowCount in the body text", () => {
    renderModal({ pendingNumFrames: 121, overflowCount: 3 });
    expect(screen.getByText(/121/)).toBeInTheDocument();
    expect(screen.getByText(/3/)).toBeInTheDocument();
  });

  // 2026-11-01 (§1-114 G-7, ruling B): a right-click prefill's shrink asks
  // about the automatic frame-count adjustment instead of "shorten?".
  it('origin="prefill": shows the automatic-adjustment title and body', () => {
    renderModal({ pendingNumFrames: 25, overflowCount: 1, origin: "prefill" });
    expect(screen.getByRole("heading", { name: "Adjust the frame count automatically?" })).toBeInTheDocument();
    expect(
      screen.getByText(
        "The right-click will set the frame count to 25 frames. 1 keyframe(s) outside the new range will be deleted.",
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText("Shorten duration?")).not.toBeInTheDocument();
  });

  it("without origin: keeps the manual-edit title and body unchanged", () => {
    renderModal({ pendingNumFrames: 25, overflowCount: 1 });
    expect(screen.getByRole("heading", { name: "Shorten duration?" })).toBeInTheDocument();
    expect(
      screen.getByText("Shortening the duration to 25 frames will delete 1 keyframe(s) that fall outside the new range."),
    ).toBeInTheDocument();
    expect(screen.queryByText("Adjust the frame count automatically?")).not.toBeInTheDocument();
  });

  it("calls onApply when the apply button is clicked", () => {
    const { onApply, onCancel } = renderModal();
    fireEvent.click(screen.getByRole("button", { name: "Apply" }));
    expect(onApply).toHaveBeenCalledTimes(1);
    expect(onCancel).not.toHaveBeenCalled();
  });

  it("calls onCancel when the cancel button is clicked", () => {
    const { onApply, onCancel } = renderModal();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onApply).not.toHaveBeenCalled();
  });

  it("calls onCancel when the overlay background is clicked", () => {
    const { container, onCancel } = renderModal();
    const overlay = container.querySelector(".kfm-overlay")!;
    fireEvent.click(overlay);
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("does not call onCancel when the dialog card itself is clicked", () => {
    const { onCancel } = renderModal();
    fireEvent.click(screen.getByRole("dialog"));
    expect(onCancel).not.toHaveBeenCalled();
  });

  it("calls onCancel when Escape is pressed", () => {
    const { onCancel, onApply } = renderModal();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onApply).not.toHaveBeenCalled();
  });

  it("focuses the apply button on mount", () => {
    renderModal();
    expect(screen.getByRole("button", { name: "Apply" })).toHaveFocus();
  });

  it("renders a modal dialog with the correct ARIA attributes", () => {
    renderModal();
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAttribute("aria-labelledby");
  });
});
