import { createContext, useContext } from "react";
import type { ReactNode } from "react";
import { useStrings } from "../i18n/LanguageContext";

/**
 * The shared "note area" (RIGHTCLICK_REDESIGN_SPEC.md §6-1): a single, always-
 * visible slot at the top of the operation panel that carries every right-click
 * feedback message (fallback-insert notices, mismatch guidance, receipt notes,
 * audio-extraction progress, …). Unlike the transient `Toasts` stack it does
 * NOT auto-dismiss — it stays until the user closes it with the × button or a
 * newer note replaces it. Only one note is ever shown at a time (§6-1: "次の
 * 通知が来たら前の通知を置き換える / 同時に複数は積まない"), so the state is a
 * single `AppNote | null` owned by `AppShell` rather than a queue.
 *
 * I5 scope: this is the container + the fallback-insert note only. The mismatch
 * guidance (I6) and receipt notes (I7) reuse this same area via `useShowNote`.
 */

/** Note severity — drives a light tonal distinction only (§6-1), reusing the
 * existing `--warning`/`--danger`/`--accent` palette. */
export type NoteKind = "info" | "warning" | "error";

export interface AppNote {
  kind: NoteKind;
  message: string;
}

/** Imperative "show a note" handle handed down to the mode screens. Replaces
 * whatever note is currently shown (single-seat, §6-1). */
export type ShowNote = (kind: NoteKind, message: string) => void;

const ShowNoteContext = createContext<ShowNote | null>(null);

/** Provides `showNote` to any descendant screen (light context rather than
 * prop-drilling through all three mode screens). `AppShell` still owns the note
 * STATE (so the single `NoteArea` renders in its fixed tab-bar-level slot); this
 * only shares the setter. */
export function ShowNoteProvider({ showNote, children }: { showNote: ShowNote; children: ReactNode }) {
  return <ShowNoteContext.Provider value={showNote}>{children}</ShowNoteContext.Provider>;
}

/** Access the shared `showNote` handle from a mode screen. */
export function useShowNote(): ShowNote {
  const ctx = useContext(ShowNoteContext);
  if (!ctx) throw new Error("useShowNote must be used within a ShowNoteProvider");
  return ctx;
}

export interface NoteAreaProps {
  /** The single note to show, or `null` to render nothing. */
  note: AppNote | null;
  /** Close the current note (the × button). */
  onDismiss: () => void;
}

/** Presentational note slot. Renders nothing when there is no note, so it takes
 * no vertical space until one appears. `role="status"` + `aria-live="polite"`
 * mirror the `Toasts` pattern; the × button reuses `common.dismissNotification`
 * for its accessible name. */
export function NoteArea({ note, onDismiss }: NoteAreaProps) {
  const strings = useStrings();
  if (!note) return null;

  return (
    <div className={`note-area note-area-${note.kind}`} role="status" aria-live="polite">
      <p className="note-area-message">{note.message}</p>
      <button
        type="button"
        className="note-area-dismiss"
        aria-label={strings.common.dismissNotification}
        onClick={onDismiss}
      >
        ×
      </button>
    </div>
  );
}
