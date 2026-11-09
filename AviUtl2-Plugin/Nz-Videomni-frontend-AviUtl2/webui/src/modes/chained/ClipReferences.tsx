import { useLayoutEffect, useMemo } from "react";
import type { NativeBridge } from "../../bridge";
import { ReferenceAccordion } from "../single/ReferenceAccordion";
import type { ReferenceAccordionA2v } from "../single/ReferenceAccordion";
import type {
  ReferenceKind,
  ReferenceRules,
  ReferenceSlot,
  ReferenceSlotsApi,
  ReferenceSummary,
} from "../single/referenceRules";
import { useReferenceSlots } from "../single/useReferenceSlots";

/** §1-109: what one clip's cards tell the Chained screen — enough for
 * `buildReferenceBody` / `referenceSendUploading` / `countUsedReferences`. */
export interface ClipReferenceEntry {
  summary: ReferenceSummary;
  slots: Record<ReferenceKind, ReferenceSlot[]>;
  /** §1-120: the clip's stable slot mutations, for the screen's distribution
   * of the long audio (inject, clear, check). Built once per clip
   * (`useMemo` over `useCallback`s), never compared by the screen's
   * no-change check — only `summary` and `slots` are. */
  ops: ClipReferenceOps;
}

/** §1-120: see {@link ClipReferenceEntry.ops}. */
export type ClipReferenceOps = Pick<ReferenceSlotsApi, "adopt" | "clear" | "setA2vSlot">;

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
  /** §1-120: the Audios cards' "Audio to video" side (MiniMax H3 with
   * `constraints.chain.source_audio`). Omitted = no box (LTX, or a backend
   * without the key) — the accordion then draws as before. */
  a2v?: ReferenceAccordionA2v | undefined;
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
  a2v,
  register,
  unregister,
}: ClipReferencesProps) {
  const api = useReferenceSlots(rules, { nativeBridge });
  const { setEnabled, state, summary, adopt, clear, setA2vSlot } = api;
  // §1-120: stable (each member is a `useCallback` of the hook), so the
  // registration below never changes because of it.
  const ops = useMemo<ClipReferenceOps>(() => ({ adopt, clear, setA2vSlot }), [adopt, clear, setA2vSlot]);

  useLayoutEffect(() => {
    setEnabled(true);
  }, [setEnabled]);

  // Registering and unregistering are two effects, so a state change only
  // re-registers (the screen keeps its object when nothing changed) and the
  // key is removed when the clip goes away.
  useLayoutEffect(() => {
    register(clipId, { summary, slots: state.slots, ops });
  }, [clipId, register, state, summary, ops]);

  useLayoutEffect(() => () => unregister(clipId), [clipId, unregister]);

  if (!referencesOn) return null;
  return (
    <ReferenceAccordion
      api={api}
      embedded
      openSignal={0}
      onAppendTag={onAppendTag}
      a2v={a2v}
      nativeBridge={nativeBridge}
      disabled={disabled}
    />
  );
}
