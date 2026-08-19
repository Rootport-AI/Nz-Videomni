import { useCallback, useMemo, useRef, useState } from "react";
import { bridge as defaultBridge, BridgeError } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import type { AppConfig } from "../../api/types";
import {
  buildConditioningImages,
  DEFAULT_STRENGTH,
  clampStrength,
  fileNameFromPath,
  sortByFrameIdx,
  type KeyframeItem,
} from "./keyframeUtils";

export interface UseKeyframesDeps {
  nativeBridge?: NativeBridge | undefined;
}

export interface UseKeyframesOptions {
  /** Caps the panel below `config.limits.max_conditioning_images` — e.g. Chain
   * mode's single "start frame" slot passes `1`. Omitted (the Create/Batch
   * default) leaves the cap at the server's `max_conditioning_images`, so
   * existing callers are unaffected. Clamped to `[1, max_conditioning_images]`
   * so it can only ever tighten the limit, never exceed the server's. */
  maxItemsOverride?: number;
}

export interface UseKeyframesResult {
  /** Sorted by ascending `frame_idx` (display order only). */
  items: KeyframeItem[];
  maxItems: number;
  canAdd: boolean;
  /** Non-null (and disables both add actions) once `items.length >= maxItems`. */
  addDisabledReason: string | null;
  /** True while a "Capture current frame" round trip is in flight (the source
   * pick itself, before a card even exists). */
  isCapturing: boolean;
  /** True while a "Choose image file..." dialog is open. */
  isPicking: boolean;
  /** True while any card is still uploading — callers should block Generate
   * on this (task brief §3: "アップロード中の生成実行はブロック"). */
  isUploading: boolean;
  /** `initialFrameIdx` seeds the new card's `frameIdx` (default `0`, matching
   * pre-rework behavior exactly — Chain's `maxItemsOverride:1` single-slot
   * usage never passes it and must see zero change here). */
  addFromCapture: (initialFrameIdx?: number) => Promise<void>;
  /** See `addFromCapture`'s `initialFrameIdx` note. */
  addFromFile: (initialFrameIdx?: number) => Promise<void>;
  /** Adds a card from an already-resolved local file path, skipping
   * `ui.pickFile` — the shared "create card, then finalizeUpload" half of
   * `addFromFile`, reused directly by Chain's unified source-input picker
   * (`useChainForm.pickSource`), which resolves the file itself via a single
   * `ui.pickFile({kind:"imageOrVideo"})` call before routing here. Always
   * adds a new card with no capacity check — callers own their own
   * capacity/replace policy (mirroring how `addFromFile` checks `canAdd`
   * itself before ever calling this). `initialFrameIdx` see `addFromCapture`. */
  addFromPath: (filePath: string, fileName: string, initialFrameIdx?: number) => Promise<void>;
  /** Keyframe timeline rework (2026-07-18): adds an image-less card (`status:
   * "empty"`) at `initialFrameIdx` — the timeline bar's "add keyframe"
   * button creates a placeholder pin this way before the user attaches an
   * image via `replaceFile`/drag-and-drop. A no-op once `items.length >=
   * maxItems`, mirroring `addFromCapture`/`addFromFile`'s own cap check. */
  addEmpty: (initialFrameIdx: number) => void;
  /** Keyframe timeline rework (2026-07-18): replaces an existing card's
   * source file in place — used both for a plain image-swap on a `"ready"`
   * card and for filling in a `"empty"` card's first image. `frameIdx`/
   * `strength` are preserved; `imageId`/`thumbnailDataUrl` are cleared and
   * `status` drops to `"uploading"` before re-running the same
   * upload+thumbnail round trip `finalizeUpload` uses elsewhere, so a failed
   * replace leaves `"error"` with no stale `imageId` left over from the
   * previous file. */
  replaceFile: (id: string, filePath: string, fileName: string) => Promise<void>;
  /** Keyframe timeline rework (2026-07-18): opens `ui.pickFile` and, on a
   * successful pick, swaps the chosen image onto the EXISTING card `id` via
   * `replaceFile` (never adds a new card). A `CANCELLED` dialog leaves the
   * card exactly as it was; any other pick failure marks the card `"error"`
   * with the bridge's code (mirroring `addFromFile`'s own DIALOG_FAILED
   * visibility, adapted to an in-place replace). Shares the `isPicking` flag
   * with `addFromFile`. Additive — Chain's single-slot `useKeyframes` usage
   * never calls it, so existing callers are unaffected. */
  pickReplaceFile: (id: string) => Promise<void>;
  /** Keyframe timeline rework: captures the current timeline frame straight
   * into an EXISTING card `id` — the KEYFRAMES panel's per-card 📸 button
   * (as opposed to `addFromCapture`, which always creates a brand-new card).
   * Marks the card `"uploading"` synchronously, before the
   * `timeline.captureFrame` round trip even starts (mirroring how
   * `addFromCapture` creates its new card already `"uploading"`), which
   * doubles as a busy indicator AND a re-entrancy guard: calling this again
   * on a card that's already `"uploading"` (its own in-flight capture, or
   * any other in-flight replace) is a silent no-op. A `CANCELLED` capture
   * restores the card's pre-call status (nothing visibly changes, no
   * confirmation needed); any other capture failure marks it `"error"` with
   * the bridge's code. A successful capture hands off to `replaceFile` (the
   * same `finalizeUpload` upload+thumbnail round trip `pickReplaceFile`
   * uses), so `frameIdx`/`strength` survive untouched. Shares the
   * `isCapturing` flag with `addFromCapture`. */
  captureIntoCard: (id: string) => Promise<void>;
  remove: (id: string) => void;
  setFrameIdx: (id: string, value: number) => void;
  setStrength: (id: string, value: number) => void;
  /** `conditioning_images` payload for the current items, or `[]` if none
   * are ready yet. */
  conditioningImages: ReturnType<typeof buildConditioningImages>;
}

function errorCodeOf(err: unknown): string {
  if (err instanceof BridgeError) return err.code;
  if (err instanceof Error) return "UNKNOWN";
  return "UNKNOWN";
}

/**
 * Owns the M4 I2V keyframe (conditioning image) panel's state: adding a
 * keyframe from either `timeline.captureFrame` or `ui.pickFile`, uploading it
 * via `backend.uploadFile` to get an `image_id`, best-effort rendering a
 * thumbnail via `ui.makeThumbnail`, and per-card frame_idx/strength editing.
 *
 * Once a card exists it's always created in `"uploading"` state, so the
 * spinner shows for the rest of the round trip through `finalizeUpload` — but
 * WHEN the card gets created differs by entry point: `addFromCapture` creates
 * it immediately, before the `timeline.captureFrame` call itself even starts,
 * so its spinner covers the capture round trip too. `addFromFile`/
 * `addFromPath` only create a card once a file path is already known —
 * `addFromFile` awaits `ui.pickFile` first and creates nothing while the
 * dialog is still open; a `CANCELLED` dismissal never creates a card at all
 * (nothing to roll back), while any other pick failure (e.g. `DIALOG_FAILED`)
 * synthesizes its own one-off `"error"` card so the failure is still visible.
 * Every other failure (the upload itself) leaves the card in `"error"` with
 * the bridge's error code, per the task brief: "追加中はスピナー、失敗はエラー
 * コード表示（CANCELLEDは無表示で無視）".
 */
export function useKeyframes(
  config: AppConfig,
  deps: UseKeyframesDeps = {},
  options: UseKeyframesOptions = {},
): UseKeyframesResult {
  const nativeBridge = deps.nativeBridge ?? defaultBridge;
  const maxItems =
    options.maxItemsOverride != null
      ? Math.max(1, Math.min(options.maxItemsOverride, config.limits.max_conditioning_images))
      : config.limits.max_conditioning_images;

  const [items, setItems] = useState<KeyframeItem[]>([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const [isPicking, setIsPicking] = useState(false);
  const nextId = useRef(0);

  const canAdd = items.length < maxItems;
  const addDisabledReason = canAdd ? null : `Maximum ${maxItems} keyframes reached.`;

  const updateItem = useCallback((id: string, patch: Partial<KeyframeItem>) => {
    setItems((prev) => prev.map((item) => (item.id === id ? { ...item, ...patch } : item)));
  }, []);

  const removeItem = useCallback((id: string) => {
    setItems((prev) => prev.filter((item) => item.id !== id));
  }, []);

  /** Uploads + thumbnails a keyframe whose source file is already known,
   * settling the card into `"ready"` or `"error"`. Runs both bridge calls
   * concurrently: the thumbnail is best-effort (failure just leaves the
   * fileName-only fallback) while the upload result is what actually decides
   * the card's terminal status. */
  const finalizeUpload = useCallback(
    async (id: string, filePath: string) => {
      const [uploadOutcome, thumbnailOutcome] = await Promise.allSettled([
        nativeBridge.request("backend.uploadFile", { kind: "image", filePath }),
        nativeBridge.request("ui.makeThumbnail", { filePath }),
      ]);

      if (thumbnailOutcome.status === "fulfilled") {
        updateItem(id, { thumbnailDataUrl: thumbnailOutcome.value.dataUrl });
      }
      // A thumbnail failure is silently ignored (fileName-only fallback per
      // the task brief) — it never affects the card's upload status below.

      if (uploadOutcome.status === "rejected") {
        updateItem(id, { status: "error", errorCode: errorCodeOf(uploadOutcome.reason) });
        return;
      }

      const { status, body } = uploadOutcome.value;
      const imageId =
        status >= 200 && status < 300 && body && typeof (body as Record<string, unknown>).image_id === "string"
          ? ((body as Record<string, unknown>).image_id as string)
          : null;
      if (imageId) {
        updateItem(id, { imageId, status: "ready", errorCode: null });
      } else {
        const errorEnvelope = body as { error?: { code?: string } } | null;
        updateItem(id, { status: "error", errorCode: errorEnvelope?.error?.code ?? `HTTP_${status}` });
      }
    },
    [nativeBridge, updateItem],
  );

  const addFromCapture = useCallback(
    async (initialFrameIdx = 0) => {
      if (items.length >= maxItems) return;
      nextId.current += 1;
      const id = `kf-${nextId.current}`;
      setItems((prev) => [
        ...prev,
        {
          id,
          filePath: "",
          fileName: "",
          imageId: null,
          thumbnailDataUrl: null,
          frameIdx: initialFrameIdx,
          strength: DEFAULT_STRENGTH,
          status: "uploading",
          errorCode: null,
        },
      ]);

      setIsCapturing(true);
      try {
        const captured = await nativeBridge.request("timeline.captureFrame", {});
        updateItem(id, { filePath: captured.filePath, fileName: fileNameFromPath(captured.filePath) });
        await finalizeUpload(id, captured.filePath);
      } catch (err) {
        updateItem(id, { status: "error", errorCode: errorCodeOf(err) });
      } finally {
        setIsCapturing(false);
      }
    },
    [items.length, maxItems, nativeBridge, updateItem, finalizeUpload],
  );

  const addFromPath = useCallback(
    async (filePath: string, fileName: string, initialFrameIdx = 0) => {
      nextId.current += 1;
      const id = `kf-${nextId.current}`;
      setItems((prev) => [
        ...prev,
        {
          id,
          filePath,
          fileName,
          imageId: null,
          thumbnailDataUrl: null,
          frameIdx: initialFrameIdx,
          strength: DEFAULT_STRENGTH,
          status: "uploading",
          errorCode: null,
        },
      ]);
      await finalizeUpload(id, filePath);
    },
    [finalizeUpload],
  );

  const addFromFile = useCallback(
    async (initialFrameIdx = 0) => {
      if (items.length >= maxItems) return;
      setIsPicking(true);
      try {
        const picked = await nativeBridge.request("ui.pickFile", { kind: "image" });
        await addFromPath(picked.filePath, picked.fileName, initialFrameIdx);
      } catch (err) {
        const code = errorCodeOf(err);
        // A CANCELLED dialog never created a card in the first place (unlike
        // the old pre-pick placeholder), so there's nothing to roll back here —
        // net-visible behavior is unchanged (no card either way). Any other
        // failure (e.g. DIALOG_FAILED) gets its own error card, mirroring what
        // `addFromPath`/`finalizeUpload` would have left behind had the pick
        // itself succeeded.
        if (code !== "CANCELLED") {
          nextId.current += 1;
          const id = `kf-${nextId.current}`;
          setItems((prev) => [
            ...prev,
            {
              id,
              filePath: "",
              fileName: "",
              imageId: null,
              thumbnailDataUrl: null,
              frameIdx: initialFrameIdx,
              strength: DEFAULT_STRENGTH,
              status: "error",
              errorCode: code,
            },
          ]);
        }
      } finally {
        setIsPicking(false);
      }
    },
    [items.length, maxItems, nativeBridge, addFromPath],
  );

  const addEmpty = useCallback(
    (initialFrameIdx: number) => {
      if (items.length >= maxItems) return;
      nextId.current += 1;
      const id = `kf-${nextId.current}`;
      setItems((prev) => [
        ...prev,
        {
          id,
          filePath: "",
          fileName: "",
          imageId: null,
          thumbnailDataUrl: null,
          frameIdx: initialFrameIdx,
          strength: DEFAULT_STRENGTH,
          status: "empty",
          errorCode: null,
        },
      ]);
    },
    [items.length, maxItems],
  );

  const replaceFile = useCallback(
    async (id: string, filePath: string, fileName: string) => {
      // frameIdx/strength are left untouched by this patch (only the
      // image-identifying fields reset) — the whole point of "replace" vs.
      // "remove + add" is that the card's position/strength survive.
      updateItem(id, {
        filePath,
        fileName,
        imageId: null,
        thumbnailDataUrl: null,
        status: "uploading",
        errorCode: null,
      });
      await finalizeUpload(id, filePath);
    },
    [updateItem, finalizeUpload],
  );

  const pickReplaceFile = useCallback(
    async (id: string) => {
      setIsPicking(true);
      try {
        const picked = await nativeBridge.request("ui.pickFile", { kind: "image" });
        await replaceFile(id, picked.filePath, picked.fileName);
      } catch (err) {
        const code = errorCodeOf(err);
        // A CANCELLED dialog leaves the card untouched (no image change) —
        // the in-place analog of `addFromFile`'s "no card either way". Any
        // other pick failure (e.g. DIALOG_FAILED) surfaces on the card
        // itself, since — unlike a fresh add — there's already a card here to
        // carry the error.
        if (code !== "CANCELLED") {
          updateItem(id, { status: "error", errorCode: code });
        }
      } finally {
        setIsPicking(false);
      }
    },
    [nativeBridge, replaceFile, updateItem],
  );

  const captureIntoCard = useCallback(
    async (id: string) => {
      const target = items.find((item) => item.id === id);
      // Also covers "no such card" (target undefined) — nothing to capture into.
      if (!target || target.status === "uploading") return;
      const previousStatus = target.status;

      // Flip to "uploading" immediately, before the captureFrame round trip
      // even starts — this is both the busy indicator and the re-entrancy
      // guard the check above relies on for any second call that lands while
      // this one is still in flight.
      updateItem(id, { status: "uploading" });

      setIsCapturing(true);
      try {
        const captured = await nativeBridge.request("timeline.captureFrame", {});
        await replaceFile(id, captured.filePath, fileNameFromPath(captured.filePath));
      } catch (err) {
        const code = errorCodeOf(err);
        if (code === "CANCELLED") {
          // No capture happened — put the card back exactly as it was
          // before this call, same as pickReplaceFile's CANCELLED handling.
          updateItem(id, { status: previousStatus });
        } else {
          updateItem(id, { status: "error", errorCode: code });
        }
      } finally {
        setIsCapturing(false);
      }
    },
    [items, nativeBridge, replaceFile, updateItem],
  );

  const setFrameIdx = useCallback(
    (id: string, value: number) => {
      const frameIdx = Math.max(0, Math.trunc(value));
      updateItem(id, { frameIdx });
    },
    [updateItem],
  );

  const setStrength = useCallback(
    (id: string, value: number) => {
      updateItem(id, { strength: clampStrength(value) });
    },
    [updateItem],
  );

  const isUploading = items.some((item) => item.status === "uploading");
  const sortedItems = useMemo(() => sortByFrameIdx(items), [items]);
  const conditioningImages = useMemo(() => buildConditioningImages(items), [items]);

  return {
    items: sortedItems,
    maxItems,
    canAdd,
    addDisabledReason,
    isCapturing,
    isPicking,
    isUploading,
    addFromCapture,
    addFromFile,
    addFromPath,
    addEmpty,
    replaceFile,
    pickReplaceFile,
    captureIntoCard,
    remove: removeItem,
    setFrameIdx,
    setStrength,
    conditioningImages,
  };
}
