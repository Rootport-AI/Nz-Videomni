import { useCallback, useMemo, useRef, useState } from "react";
import { BackendApiError, apiClient as defaultApiClient, createApiClient } from "../../api/client";
import type { ApiClient } from "../../api/client";
import { BridgeError, bridge as defaultBridge } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import { useStrings } from "../../i18n/LanguageContext";
import { useFileDrop } from "../../shell/useFileDrop";
import "./Mp4InfoSection.css";

/** The video extensions the drop card accepts — the same four as native's
 * `ui.pickFile({kind: "video"})` filter and the other video slots. */
const MP4_INFO_EXTENSIONS = ["mp4", "mov", "webm", "mkv"];

/** idle -> loading -> (found | notFound | error). A new file restarts from
 * `loading`; there is no way back to `idle`. */
type Mp4InfoState =
  | { kind: "idle" }
  | { kind: "loading"; fileName: string }
  | { kind: "found"; fileName: string; comment: string }
  | { kind: "notFound"; fileName: string }
  | { kind: "error"; fileName: string; message: string; code?: string | undefined };

export interface Mp4InfoSectionProps {
  /** Test/DI seam — defaults to the app-wide bridge singleton. */
  nativeBridge?: NativeBridge | undefined;
}

/**
 * The Toolbox tab's "mp4 info" tool (§3-164, 2026-09-24): the mp4 counterpart
 * of A1111's PNG Info. A video is dropped (or picked) on the left card; its
 * LOCAL PATH goes to `POST /utils/mp4-info`, whose answer is the file's
 * `comment` tag — for a video this app generated, the same JSON as its
 * metadata.json. The right column shows that string verbatim in a read-only
 * textarea: no reformatting (the backend already writes it indented), and no
 * copy button (select-and-copy is the intended way, owner ruling 2026-09-24).
 *
 * Only the path crosses the bridge, never the file's bytes, so this works only
 * while the backend runs on this machine — the endpoint refuses non-loopback
 * callers anyway.
 */
export function Mp4InfoSection({ nativeBridge }: Mp4InfoSectionProps) {
  const strings = useStrings();
  const t = strings.toolbox.mp4info;
  const bridge = nativeBridge ?? defaultBridge;
  const client: ApiClient = useMemo(
    () => (nativeBridge ? createApiClient(nativeBridge) : defaultApiClient),
    [nativeBridge],
  );
  const [state, setState] = useState<Mp4InfoState>({ kind: "idle" });
  // Only the latest file's answer may land: a slow read of an earlier file
  // must not overwrite the result of a later one.
  const requestSeqRef = useRef(0);

  const readFile = useCallback(
    (filePath: string, fileName: string) => {
      requestSeqRef.current += 1;
      const seq = requestSeqRef.current;
      setState({ kind: "loading", fileName });
      void (async () => {
        let next: Mp4InfoState;
        try {
          const res = await client.getMp4Info(filePath);
          next =
            res.comment === null
              ? { kind: "notFound", fileName }
              : { kind: "found", fileName, comment: res.comment };
        } catch (err) {
          next = {
            kind: "error",
            fileName,
            message: err instanceof Error ? err.message : String(err),
            code: err instanceof BackendApiError ? err.code : undefined,
          };
        }
        if (seq === requestSeqRef.current) setState(next);
      })();
    },
    [client],
  );

  const pick = useCallback(async () => {
    let picked;
    try {
      picked = await bridge.request("ui.pickFile", { kind: "video" });
    } catch (err) {
      // Dismissing the dialog is not an error worth showing.
      if (err instanceof BridgeError && err.code === "CANCELLED") return;
      setState({ kind: "error", fileName: "", message: err instanceof Error ? err.message : String(err) });
      return;
    }
    readFile(picked.filePath, picked.fileName);
  }, [bridge, readFile]);

  const {
    isDragOver,
    error: dropError,
    handlers: dropHandlers,
  } = useFileDrop({ accept: MP4_INFO_EXTENSIONS, onFile: readFile, nativeBridge });

  const loading = state.kind === "loading";
  const fileName = state.kind === "idle" ? "" : state.fileName;

  let statusText: string;
  if (state.kind === "idle") statusText = t.idle;
  else if (state.kind === "loading") statusText = t.loading;
  else if (state.kind === "notFound") statusText = t.notFound;
  // The two codes this endpoint defines get their own translated sentence;
  // anything else (other codes, transport failures) keeps the server's message.
  else if (state.kind === "error" && state.code === "MEDIA_NOT_FOUND") statusText = t.errorNotFound;
  else if (state.kind === "error" && state.code === "MEDIA_UNREADABLE") statusText = t.errorUnreadable;
  else if (state.kind === "error") statusText = t.error(state.message);
  else statusText = "";

  const cardClassName = ["toolbox-section", "mp4info-drop", isDragOver ? "mp4info-drop-dragover" : ""]
    .filter(Boolean)
    .join(" ");

  return (
    <div className="mp4info-layout">
      <section
        className={cardClassName}
        aria-label={t.heading}
        onDragEnter={dropHandlers.onDragEnter}
        onDragOver={dropHandlers.onDragOver}
        onDragLeave={dropHandlers.onDragLeave}
        onDrop={dropHandlers.onDrop}
      >
        <h3 className="toolbox-section-heading">{t.heading}</h3>
        <p className="field-hint">{t.dropHint}</p>
        <button
          type="button"
          className="secondary-button mp4info-pick"
          disabled={loading}
          onClick={() => void pick()}
        >
          {t.pickButton}
        </button>
        {fileName && (
          <p className="field-hint mp4info-file">
            {t.fileLabel}: {fileName}
          </p>
        )}
        {dropError && (
          <p className="field-hint field-hint-error">
            {dropError === "unsupported" ? strings.dnd.unsupportedVideo : strings.dnd.resolveFailed}
          </p>
        )}
      </section>

      <section className="toolbox-section mp4info-result">
        <label className="field mp4info-result-field">
          <span className="field-label">{t.resultLabel}</span>
          <textarea
            className="prompt-input mp4info-textarea"
            readOnly
            spellCheck={false}
            value={state.kind === "found" ? state.comment : ""}
          />
        </label>
        {statusText && (
          <p
            className={`field-hint${state.kind === "error" ? " field-hint-error" : ""}`}
            role="status"
          >
            {statusText}
          </p>
        )}
      </section>
    </div>
  );
}
