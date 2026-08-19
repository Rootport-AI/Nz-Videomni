import { useEffect, useRef } from "react";
import { useStrings } from "../../i18n/LanguageContext";
import "./KeyframeShrinkModal.css";

export interface KeyframeShrinkModalProps {
  /** The new (shortened) duration, in frames, that the user is about to apply. */
  pendingNumFrames: number;
  /** How many keyframes fall outside `[0, pendingNumFrames)` and would be deleted. */
  overflowCount: number;
  onApply: () => void;
  onCancel: () => void;
}

/** Confirmation modal shown when shortening the Create screen's DURATION would
 * push one or more keyframes out of range, deleting them. Owner requirement:
 * "ユーザーが気付かないリスクの排除" — a destructive, easy-to-miss side effect
 * of a plain duration edit must be surfaced with a blocking, explicit dialog
 * rather than a silent delete or a passive inline hint.
 *
 * Modeled on `SettingsPanel.tsx`'s overlay/dialog pair (this app's only other
 * modal precedent — no portal/focus-trap library, matching its
 * dependency-light approach), but adds two things that precedent doesn't
 * need: Escape-to-cancel and autoFocus on the primary action, since this
 * dialog can be triggered by a keyboard-driven duration edit and must be
 * dismissable/confirmable without reaching for the mouse.
 *
 * Visibility is entirely the caller's responsibility — this component always
 * renders itself once mounted; the parent conditionally mounts/unmounts it. */
export function KeyframeShrinkModal({ pendingNumFrames, overflowCount, onApply, onCancel }: KeyframeShrinkModalProps) {
  const strings = useStrings();
  const t = strings.single.keyframes.shrinkModal;
  const titleId = "kfm-shrink-modal-title";
  const applyButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    applyButtonRef.current?.focus();
  }, []);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        onCancel();
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onCancel]);

  return (
    <div className="kfm-overlay" role="presentation" onClick={onCancel}>
      <div className="kfm-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId} onClick={(e) => e.stopPropagation()}>
        <h2 id={titleId} className="kfm-title">
          {t.title}
        </h2>
        <p className="kfm-body">{t.body(pendingNumFrames, overflowCount)}</p>
        <div className="kfm-actions">
          <button type="button" className="kfm-cancel-button" onClick={onCancel}>
            {t.cancelButton}
          </button>
          <button type="button" className="kfm-apply-button" ref={applyButtonRef} onClick={onApply}>
            {t.applyButton}
          </button>
        </div>
      </div>
    </div>
  );
}
