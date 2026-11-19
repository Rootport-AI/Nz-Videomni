/**
 * Batch v2v-ref-long: one row clip's cards as the Reference accordion's
 * `ReferenceSlotsApi`. The cards live in the store (as file paths, no upload);
 * this hook only adds the thumbnails, which it keeps while it is mounted (the
 * open card) and drops with it.
 */
import { useEffect, useRef, useState } from "react";
import { bridge as defaultBridge } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import { REFERENCE_KINDS, createInitialState, summarizeReference } from "../single/referenceRules";
import type { ReferenceKind, ReferenceSlot, ReferenceSlotsApi, ReferenceState } from "../single/referenceRules";
import { makeThumbnail } from "../single/useReferenceSlots";
import type { UseV2vRefLongFormResult } from "./types";

/** Ready picture/video cards without a thumbnail, as `kind\tpath` keys. */
function wantedThumbnails(refs: ReferenceState): string[] {
  const keys: string[] = [];
  for (const kind of REFERENCE_KINDS) {
    if (kind === "audio") continue;
    for (const slot of refs.slots[kind]) {
      if (slot.status === "ready" && slot.filePath !== null && slot.thumbnailDataUrl === null) {
        keys.push(`${kind}\t${slot.filePath}`);
      }
    }
  }
  return keys;
}

export function useClipReferenceApi(
  form: UseV2vRefLongFormResult,
  rowIndex: number,
  clip: number,
  nativeBridge?: NativeBridge,
): ReferenceSlotsApi {
  const bridge = nativeBridge ?? defaultBridge;
  const { rules } = form;
  const refs = form.effectiveClipOf(rowIndex, clip)?.refs ?? { ...createInitialState(rules), enabled: true };

  // File path -> data URL (`null` = the request failed; not asked again).
  const [thumbs, setThumbs] = useState<ReadonlyMap<string, string | null>>(() => new Map());
  const requested = useRef(new Set<string>());
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const wantKey = wantedThumbnails(refs).join("\n");
  useEffect(() => {
    if (wantKey === "") return;
    for (const key of wantKey.split("\n")) {
      const [kind, filePath] = key.split("\t") as [ReferenceKind, string];
      if (requested.current.has(filePath)) continue;
      requested.current.add(filePath);
      void makeThumbnail(bridge, kind, filePath).then((dataUrl) => {
        if (!mounted.current) return;
        setThumbs((prev) => new Map(prev).set(filePath, dataUrl));
      });
    }
  }, [wantKey, bridge]);

  const withThumbs = (slots: ReferenceSlot[]): ReferenceSlot[] =>
    slots.map((slot) => {
      const thumb = slot.filePath !== null ? thumbs.get(slot.filePath) : undefined;
      return thumb && slot.thumbnailDataUrl === null ? { ...slot, thumbnailDataUrl: thumb } : slot;
    });
  const state: ReferenceState = {
    ...refs,
    slots: { picture: withThumbs(refs.slots.picture), video: withThumbs(refs.slots.video), audio: refs.slots.audio },
  };

  return {
    state,
    summary: summarizeReference(state, rules),
    rules,
    attach: (kind, slot, filePath, fileName) => form.attachClipFile(rowIndex, clip, kind, slot, filePath, fileName),
    pick: (kind, slot) => form.pickClipFile(rowIndex, clip, kind, slot),
    clear: (kind, slot) => form.clearClipSlot(rowIndex, clip, kind, slot),
    // Unused by the embedded accordion without `a2v` (no Enable, no Audio to
    // video) and by this panel (no right-click delivery).
    adopt: () => {},
    attachToFirstEmpty: () => null,
    setEnabled: () => {},
    setA2vSlot: () => {},
  };
}
