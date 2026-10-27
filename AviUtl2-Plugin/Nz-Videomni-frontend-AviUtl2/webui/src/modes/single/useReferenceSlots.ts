/**
 * §1-107 (UI first): the MiniMax H3 Reference accordion's state hook
 * (`useReferenceSlots`) and the hook that mirrors its A2V audio slot into the
 * form's existing `sourceAudio` upload (`useReferenceA2vSync`).
 *
 * The state, the reducer and every rule live in `referenceRules.ts`; this file
 * only adds the bridge calls (upload, thumbnail, duration probe) and drops
 * stale async results with a per-slot generation number — the same idea as
 * `useSourceUpload`'s `generationRef`.
 */
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { bridge as defaultBridge, BridgeError } from "../../bridge";
import type { NativeBridge, UploadVideoResponseBody } from "../../bridge";
import { requestedTrim, type UseSourceUploadResult } from "../chained/useSourceUpload";
import {
  createInitialState,
  effectiveA2vSlot,
  firstEmptySlot,
  referenceReducer,
  summarizeReference,
  type ReferenceAction,
  type ReferenceAttachOptions,
  type ReferenceKind,
  type ReferenceRules,
  type ReferenceSlotsApi,
  type ReferenceState,
} from "./referenceRules";

export interface UseReferenceSlotsDeps {
  nativeBridge?: NativeBridge | undefined;
}

const UPLOAD_KIND: Record<ReferenceKind, "image" | "video" | "audio"> = {
  picture: "image",
  video: "video",
  audio: "audio",
};

const ID_FIELD: Record<ReferenceKind, "image_id" | "video_id" | "audio_id"> = {
  picture: "image_id",
  video: "video_id",
  audio: "audio_id",
};

function errorCodeOf(err: unknown): string {
  return err instanceof BridgeError ? err.code : "UNKNOWN";
}

function positiveSeconds(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
}

/** The three counts the slot arrays are built from — a change means `resize`. */
function sizeKey(rules: ReferenceRules): string {
  return `${rules.maxImages}/${rules.maxVideos}/${rules.maxAudios}`;
}

/** The clip's length: the caller's known value, else `fs.probeMediaInfo`, else
 * (audio only) the wav header. `null` for a picture or an unmeasurable file.
 * Never rejects. */
async function measureDuration(
  nativeBridge: NativeBridge,
  kind: ReferenceKind,
  filePath: string,
  known: number | undefined,
): Promise<number | null> {
  if (kind === "picture") return null;
  if (known !== undefined) return positiveSeconds(known);
  try {
    const info = await nativeBridge.request("fs.probeMediaInfo", { filePath });
    const sec = positiveSeconds(info.durationSec);
    if (sec !== null) return sec;
  } catch {
    // best-effort — fall through
  }
  if (kind !== "audio") return null;
  try {
    const wav = await nativeBridge.request("fs.probeAudioDuration", { filePath });
    return wav.isWav ? positiveSeconds(wav.durationSec) : null;
  } catch {
    return null;
  }
}

/** Best-effort thumbnail for a picture/video card; `null` on failure (the card
 * then shows its fixed placeholder). Audio cards never ask. */
async function makeThumbnail(nativeBridge: NativeBridge, kind: ReferenceKind, filePath: string): Promise<string | null> {
  if (kind === "audio") return null;
  try {
    const result = await nativeBridge.request("ui.makeThumbnail", { filePath });
    return result.dataUrl;
  } catch {
    return null;
  }
}

/**
 * The Reference accordion's slots (counts from `rules`, i.e. the server's
 * `constraints.reference`). When the counts change (another base model), the
 * state is resized and the slots that still fit keep their content.
 */
export function useReferenceSlots(rules: ReferenceRules, deps: UseReferenceSlotsDeps = {}): ReferenceSlotsApi {
  const nativeBridge = deps.nativeBridge ?? defaultBridge;
  const [state, dispatch] = useReducer(referenceReducer, rules, createInitialState);

  // The latest state, advanced synchronously with every action this hook sends,
  // so `attachToFirstEmpty` called twice before a re-render picks two slots.
  const stateRef = useRef<ReferenceState>(state);
  const send = useCallback((action: ReferenceAction) => {
    stateRef.current = referenceReducer(stateRef.current, action);
    dispatch(action);
  }, []);

  // Resize during render (React's "adjust state when a prop changes" pattern),
  // so no render ever pairs the new rules with the old slot arrays.
  const [sizedFor, setSizedFor] = useState(() => sizeKey(rules));
  if (sizedFor !== sizeKey(rules)) {
    setSizedFor(sizeKey(rules));
    send({ type: "resize", rules });
  }

  // Per-slot generation: bumped by every attach and clear of that slot; an
  // async result whose generation is no longer current is dropped.
  const generationRef = useRef(new Map<string, number>());
  const bump = useCallback((kind: ReferenceKind, slot: number): (() => boolean) => {
    const key = `${kind}:${slot}`;
    const generation = (generationRef.current.get(key) ?? 0) + 1;
    generationRef.current.set(key, generation);
    return () => generationRef.current.get(key) === generation;
  }, []);

  const attach = useCallback(
    async (kind: ReferenceKind, slot: number, filePath: string, fileName: string, opts?: ReferenceAttachOptions) => {
      const isCurrent = bump(kind, slot);
      send({ type: "attachStart", kind, slot, filePath, fileName });
      const query = opts?.query;

      const [upload, thumbnailDataUrl, durationSec] = await Promise.all([
        nativeBridge
          .request("backend.uploadFile", { kind: UPLOAD_KIND[kind], filePath, ...(query ? { query } : {}) })
          .then(
            (value) => ({ ok: true as const, value }),
            (err: unknown) => ({ ok: false as const, code: errorCodeOf(err) }),
          ),
        makeThumbnail(nativeBridge, kind, filePath),
        measureDuration(nativeBridge, kind, filePath, opts?.durationSec),
      ]);
      if (!isCurrent()) return; // cleared or re-attached meanwhile — discard

      if (!upload.ok) {
        send({ type: "attachFail", kind, slot, errorCode: upload.code });
        return;
      }
      const { status, body } = upload.value;
      const raw = body as Record<string, unknown> | null;
      const id = status >= 200 && status < 300 && raw && typeof raw[ID_FIELD[kind]] === "string"
        ? (raw[ID_FIELD[kind]] as string)
        : null;
      if (id === null) {
        const envelope = body as { error?: { code?: string } } | null;
        send({ type: "attachFail", kind, slot, errorCode: envelope?.error?.code ?? `HTTP_${status}` });
        return;
      }
      // Same rule as `useSourceUpload`: a trim we asked for but did not get.
      const trimFailed = kind === "video" && requestedTrim(query) && (body as UploadVideoResponseBody).trimmed !== true;
      send({ type: "attachDone", kind, slot, id, durationSec, thumbnailDataUrl, trimFailed });
    },
    [nativeBridge, bump, send],
  );

  const pick = useCallback(
    async (kind: ReferenceKind, slot: number) => {
      let picked: { filePath: string; fileName: string };
      try {
        picked = await nativeBridge.request("ui.pickFile", { kind: UPLOAD_KIND[kind] });
      } catch {
        // CANCELLED (or a dialog failure): the slot is left as it was.
        return;
      }
      await attach(kind, slot, picked.filePath, picked.fileName);
    },
    [nativeBridge, attach],
  );

  const clear = useCallback(
    (kind: ReferenceKind, slot: number) => {
      bump(kind, slot);
      send({ type: "clear", kind, slot });
    },
    [bump, send],
  );

  const attachToFirstEmpty = useCallback(
    (kind: ReferenceKind, filePath: string, fileName: string, opts?: ReferenceAttachOptions): number | null => {
      const slot = firstEmptySlot(stateRef.current.slots[kind]);
      if (slot === null) return null;
      void attach(kind, slot, filePath, fileName, opts);
      return slot;
    },
    [attach],
  );

  const setEnabled = useCallback((enabled: boolean) => send({ type: "setEnabled", enabled }), [send]);
  const setA2vSlot = useCallback((slot: number | null) => send({ type: "setA2vSlot", slot }), [send]);

  const summary = useMemo(() => summarizeReference(state, rules), [state, rules]);

  return { state, summary, rules, attach, pick, clear, attachToFirstEmpty, setEnabled, setA2vSlot };
}

/**
 * Mirrors the Reference accordion's effective A2V audio slot into the form's
 * `sourceAudio` (which stays the one source of truth for the A2V path:
 * `audioPrecheck`, the wav length auto-adjust, the `/generate` body).
 *
 * - While `referenceAvailable` (the current constraints carry `reference`), the
 *   Reference decides: an effective A2V slot is `adopt`ed (when `sourceAudio`
 *   is not ready or holds a different id), otherwise `sourceAudio` is cleared — including an audio that did
 *   not come from the Reference (e.g. one attached under LTX before switching).
 * - Once the Reference goes away (switch to LTX), `sourceAudio` is cleared only
 *   if it still holds the id this hook adopted.
 *
 * Call it once, next to `useReferenceSlots`, in the component that owns both:
 * `useReferenceA2vSync(reference, form.sourceAudio, constraints.reference !== null)`
 * (the current constraints, not the retained `rules`).
 */
export function useReferenceA2vSync(
  ref: ReferenceSlotsApi,
  sourceAudio: UseSourceUploadResult,
  referenceAvailable: boolean,
): void {
  const adoptedIdRef = useRef<string | null>(null);

  const a2vSlot = effectiveA2vSlot(ref.state);
  const target = a2vSlot !== null ? ref.state.slots.audio[a2vSlot - 1] : undefined;
  const targetId = target?.id ?? null;
  const targetFileName = target?.fileName ?? "";
  const targetFilePath = target?.filePath ?? "";

  const { adopt, clear } = sourceAudio;
  const currentId = sourceAudio.state.id;
  const currentStatus = sourceAudio.state.status;

  useEffect(() => {
    if (referenceAvailable) {
      if (targetId !== null) {
        if (currentStatus !== "ready" || currentId !== targetId) {
          adopt({ id: targetId, fileName: targetFileName, filePath: targetFilePath });
        }
        adoptedIdRef.current = targetId;
      } else {
        if (currentStatus !== "idle") clear();
        adoptedIdRef.current = null;
      }
      return;
    }
    if (adoptedIdRef.current !== null) {
      if (currentId === adoptedIdRef.current) clear();
      adoptedIdRef.current = null;
    }
  }, [referenceAvailable, targetId, targetFileName, targetFilePath, currentId, currentStatus, adopt, clear]);
}
