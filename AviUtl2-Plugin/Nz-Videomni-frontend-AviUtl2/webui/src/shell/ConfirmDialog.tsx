import { useEffect, useRef } from "react";
import "./ConfirmDialog.css";

export interface ConfirmDialogProps {
  /** Short heading. */
  title: string;
  /** The question/explanation body. */
  body: string;
  /** Primary (confirm) button label. */
  confirmLabel: string;
  /** Secondary (cancel) button label. */
  cancelLabel: string;
  onConfirm: () => void;
  onCancel: () => void;
}

/** A generic yes/no confirmation modal (I8 §3-4 ※: the Chain-discard prompt).
 * Modeled on `modes/single/KeyframeShrinkModal` — the same dependency-light
 * overlay/dialog pair, Escape-to-cancel, and autoFocus on the primary action —
 * but decoupled from that screen's specific copy so `AppShell` can await a
 * boolean answer from it (via a Promise the parent resolves in the button
 * handlers). Visibility is the caller's responsibility: it always renders once
 * mounted; the parent conditionally mounts/unmounts it. */
export function ConfirmDialog({ title, body, confirmLabel, cancelLabel, onConfirm, onCancel }: ConfirmDialogProps) {
  const titleId = "confirm-dialog-title";
  const confirmButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    confirmButtonRef.current?.focus();
  }, []);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onCancel();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onCancel]);

  return (
    <div className="confirm-overlay" role="presentation" onClick={onCancel}>
      <div
        className="confirm-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id={titleId} className="confirm-title">
          {title}
        </h2>
        <p className="confirm-body">{body}</p>
        <div className="confirm-actions">
          <button type="button" className="confirm-cancel-button" onClick={onCancel}>
            {cancelLabel}
          </button>
          <button type="button" className="confirm-apply-button" ref={confirmButtonRef} onClick={onConfirm}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
