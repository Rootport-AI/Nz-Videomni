import { useCallback, useEffect, useRef, useState } from "react";
import type { DragEvent } from "react";
import { bridge as defaultBridge } from "../bridge";
import type { NativeBridge } from "../bridge";

/** The last path component's extension, lowercased and without the leading
 * dot (e.g. "mp4"). `null` when there is no extension (no dot, or the dot is
 * the file name's last character). */
function extensionOf(fileName: string): string | null {
  const dot = fileName.lastIndexOf(".");
  if (dot < 0 || dot === fileName.length - 1) return null;
  return fileName.slice(dot + 1).toLowerCase();
}

/**
 * The audio extensions every audio slot accepts — Create's A2V card
 * (`modes/single/GenerationForm.tsx`) and Chain's long-A2V card
 * (`modes/chained/ChainAudioPanel.tsx`), for both drag-and-drop
 * ({@link UseFileDropOptions.accept}) and the native picker.
 *
 * Mirrors native's `ui.pickFile({kind: "audio"})` filter exactly
 * (`bridge_core.cpp`'s `PickFileFilter`) and must stay in sync with it. It
 * lives here — next to the `accept` contract it is passed to — rather than in
 * either card, so §1-16 長尺A2V did not have to give one screen's component
 * module a non-component export just to share the list with the other.
 */
export const SOURCE_AUDIO_EXTENSIONS = ["wav", "mp3", "m4a", "aac", "flac", "ogg"];

export interface UseFileDropOptions {
  /** Accepted extensions, lowercase and without the leading dot (e.g.
   * `["mp4", "mov", "webm", "mkv"]`). A dropped file whose extension isn't in
   * this list (or has none) sets `error: "unsupported"` instead of calling
   * `onFile`. */
  accept: readonly string[];
  /** Called once a dropped file resolves to a real local path with an
   * accepted extension. */
  onFile: (filePath: string, fileName: string) => void;
  /** Ignores drag/drop entirely while true (mirrors the surrounding form's
   * own disabled state). Defaults to false. */
  disabled?: boolean | undefined;
  /** Test/DI seam — defaults to the app-wide bridge singleton. */
  nativeBridge?: NativeBridge | undefined;
  /** Bump this (e.g. a counter) whenever the caller's own "clear" action
   * fires, to force-clear a stale `error` that a plain `clear()` on the
   * caller's own state can't reach — `error` otherwise only resets at the
   * start of the NEXT drop (see `onDrop` below), so an "Unsupported file
   * type" message from a rejected drop would otherwise survive a
   * same-slot "Remove"/"Clear" click. Any value works as long as it
   * changes; identity, not meaning, is what matters. */
  resetErrorSignal?: unknown;
}

export type FileDropError = "unsupported" | "resolveFailed";

export interface UseFileDropResult {
  /** True for the whole span the pointer is dragging a file over the zone
   * (dragenter..dragleave/drop), for the caller's highlight styling. */
  isDragOver: boolean;
  /** Non-null after a drop attempt failed — cleared at the start of the next
   * drop. `"unsupported"` is an extension `accept` doesn't cover;
   * `"resolveFailed"` is `ui.resolveDroppedFiles` erroring, timing out, or
   * resolving to no usable file (empty `files[]`). */
  error: FileDropError | null;
  /** Spread directly onto the drop target element. */
  handlers: {
    onDragEnter: (e: DragEvent<HTMLElement>) => void;
    onDragOver: (e: DragEvent<HTMLElement>) => void;
    onDragLeave: (e: DragEvent<HTMLElement>) => void;
    onDrop: (e: DragEvent<HTMLElement>) => void;
  };
}

/**
 * Drag-and-drop file acceptance for a single-file input slot (contract v7).
 * Only the first dropped file is ever used — every slot this wires into
 * (Create's reference-video/source-audio, Chain's unified source input) is a
 * single-file slot, so a multi-file drop's remaining entries are simply
 * ignored rather than surfaced as an error.
 *
 * The dropped `File` object never carries a real local path (a browser
 * security restriction WebView2 doesn't lift), so resolving it is a native
 * round trip: `nativeBridge.requestWithFiles("ui.resolveDroppedFiles", {},
 * [file])` attaches the file as a WebView2 `AdditionalObject`; native
 * resolves it to `{filePath, fileName}` (or excludes it if it couldn't).
 */
export function useFileDrop(options: UseFileDropOptions): UseFileDropResult {
  const { accept, onFile, disabled = false } = options;
  const nativeBridge = options.nativeBridge ?? defaultBridge;
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<FileDropError | null>(null);

  // Fires whenever `resetErrorSignal`'s IDENTITY changes (including the
  // mount-time null→null "change" and, under React StrictMode, the dev-only
  // double-invoke of this effect) — every one of those is a no-op `setError`
  // call when there's nothing to clear, so it's safe to run unconditionally
  // rather than only wiring it up once a real error is present.
  useEffect(() => {
    setError(null);
  }, [options.resetErrorSignal]);

  // MINOR fix (post-review): dragenter/dragleave fire on every child-element
  // boundary crossing inside the zone (a DropZone's children include the
  // choose/change/clear button row), not just on the zone's own outer edge —
  // without a counter, hovering from the zone onto one of those children
  // fires a dragleave (clearing the highlight) immediately followed by a
  // dragenter (re-setting it), flickering the highlight off and on. A ref
  // counter incremented on every dragenter and decremented on every
  // dragleave only clears `isDragOver` once it returns to 0 — i.e. once the
  // pointer has actually left every nested element, not just one of them.
  const dragCounterRef = useRef(0);
  // MINOR fix (post-review): guards the async resolve/apply tail of `onDrop`
  // against firing `setError`/`onFile` after this hook's owner has already
  // unmounted (e.g. the user switched Create/Chain/Library mid-resolve) —
  // without it, a slow `ui.resolveDroppedFiles` round trip could call back
  // into a stale form's setters after React has torn it down.
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // preventDefault unconditionally (even while disabled): without it the
  // browser's default handling takes over on dragover/drop (see main.tsx's
  // module-level fallback for anywhere outside every DropZone). Only the
  // VISIBLE state (highlight, resolving a file) is gated on `disabled`.
  const onDragEnter = useCallback(
    (e: DragEvent<HTMLElement>) => {
      e.preventDefault();
      if (disabled) return;
      dragCounterRef.current += 1;
      setIsDragOver(true);
    },
    [disabled],
  );

  const onDragOver = useCallback(
    (e: DragEvent<HTMLElement>) => {
      e.preventDefault();
      // dragover fires continuously while hovering (not just once per
      // enter/leave pair) — it must never touch the enter/leave counter
      // above, only defensively ensure the highlight is on in case a
      // dragenter was somehow missed.
      if (disabled) return;
      setIsDragOver(true);
    },
    [disabled],
  );

  const onDragLeave = useCallback((e: DragEvent<HTMLElement>) => {
    e.preventDefault();
    dragCounterRef.current = Math.max(0, dragCounterRef.current - 1);
    if (dragCounterRef.current === 0) {
      setIsDragOver(false);
    }
  }, []);

  const onDrop = useCallback(
    (e: DragEvent<HTMLElement>) => {
      e.preventDefault();
      dragCounterRef.current = 0;
      setIsDragOver(false);
      if (disabled) return;

      const file = e.dataTransfer.files[0];
      if (!file) return; // nothing file-shaped was dropped (e.g. plain text)
      setError(null);

      void (async () => {
        let resolvedFiles: Array<{ filePath: string; fileName: string }>;
        try {
          const result = await nativeBridge.requestWithFiles("ui.resolveDroppedFiles", {}, [file]);
          resolvedFiles = result.files;
        } catch {
          if (mountedRef.current) setError("resolveFailed");
          return;
        }
        if (!mountedRef.current) return; // unmounted while resolving — never touch a stale form
        const resolved = resolvedFiles[0];
        if (!resolved) {
          setError("resolveFailed"); // native couldn't resolve a real path
          return;
        }
        const ext = extensionOf(resolved.fileName);
        if (!ext || !accept.includes(ext)) {
          setError("unsupported");
          return;
        }
        onFile(resolved.filePath, resolved.fileName);
      })();
    },
    [disabled, accept, onFile, nativeBridge],
  );

  return { isDragOver, error, handlers: { onDragEnter, onDragOver, onDragLeave, onDrop } };
}
