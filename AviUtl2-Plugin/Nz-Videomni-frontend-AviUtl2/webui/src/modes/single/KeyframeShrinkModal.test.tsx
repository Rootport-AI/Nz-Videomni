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
