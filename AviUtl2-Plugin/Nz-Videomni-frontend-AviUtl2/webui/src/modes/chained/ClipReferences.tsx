import { useLayoutEffect } from "react";
import type { NativeBridge } from "../../bridge";
import { ReferenceAccordion } from "../single/ReferenceAccordion";
import type { ReferenceKind, ReferenceRules, ReferenceSlot, ReferenceSummary } from "../single/referenceRules";
import { useReferenceSlots } from "../single/useReferenceSlots";

/** §1-109: what one clip's cards tell the Chained screen — enough for
 * `buildReferenceBody` / `referenceSendUploading` / `countUsedReferences`. */
export interface ClipReferenceEntry {
  summary: ReferenceSummary;
  slots: Record<ReferenceKind, ReferenceSlot[]>;
}

export interface ClipReferencesProps {
  clipId: string;
  /** The last `constraints.reference` the screen has seen (kept through LTX). */
  rules: ReferenceRules;
  /** Draw the accordion: the screen's References checkbox is on and the
   * loaded base model takes references. Off = hidden, the cards are kept. */
  referencesOn: boolean;
  nativeBridge?: NativeBridge | undefined;
  disabled: boolean;
  /** The pen button: appends the card's tag to THIS clip's own prompt. */
  onAppendTag: (tag: string) => void;
  /** Stable callbacks from the screen (`useCallback`). */
  register: (clipId: string, entry: ClipReferenceEntry) => void;
  unregister: (clipId: string) => void;
}

/**
 * §1-109: one Chained clip's Reference cards. Mounted for every clip (keyed by
 * the clip id) once MiniMax H3 has been seen, so the cards survive the
 * References checkbox going off and a trip through LTX; removing the clip
 * unmounts this and drops its cards.
 *
 * The state is Create's own `useReferenceSlots`, with Enable switched on once
 * at mount (the embedded accordion draws no Enable checkbox — the screen-wide
 * References checkbox decides whether anything is sent). The state is handed
 * up in a layout effect, so the screen never renders a body older than the
 * cards on screen (an upload just started, a card just cleared).
 */
export function ClipReferences({
  clipId,
  rules,
  referencesOn,
  nativeBridge,
  disabled,
  onAppendTag,
  register,
  unregister,
}: ClipReferencesProps) {
  const api = useReferenceSlots(rules, { nativeBridge });
  const { setEnabled, state, summary } = api;

  useLayoutEffect(() => {
    setEnabled(true);
  }, [setEnabled]);

  // Registering and unregistering are two effects, so a state change only
  // re-registers (the screen keeps its object when nothing changed) and the
  // key is removed when the clip goes away.
  useLayoutEffect(() => {
    register(clipId, { summary, slots: state.slots });
  }, [clipId, register, state, summary]);

  useLayoutEffect(() => () => unregister(clipId), [clipId, unregister]);

  if (!referencesOn) return null;
  return (
    <ReferenceAccordion
      api={api}
      embedded
      openSignal={0}
      onAppendTag={onAppendTag}
      nativeBridge={nativeBridge}
      disabled={disabled}
    />
  );
}
