import { useCallback, useRef, useState } from "react";
import { bridge as defaultBridge, BridgeError } from "../../bridge";
import type { NativeBridge, UploadVideoResponseBody } from "../../bridge";

export type SourceUploadStatus = "idle" | "uploading" | "ready" | "error";

export interface SourceUploadState {
  status: SourceUploadStatus;
  fileName: string | null;
  /** The resolved local file path this upload came from (from `ui.pickFile`,
   * a dropped-file resolution, or a direct `uploadPath` caller) — null until
   * an upload starts, and null again after `clear()`. Added so consumers that
   * need the path itself (e.g. Create's `fs.probeMediaInfo` duration readout)
   * can read it off the state; Chain's own run flow only ever needs the
   * `id`/`fileName`, so this is purely additive to that path. */
  filePath: string | null;
  /** `video_id`/`audio_id` from `POST /upload/video|audio` — null until
   * `status === "ready"`. */
  id: string | null;
  errorCode: string | null;
  /** §1-6 (source trim): this upload ASKED for a range trim (its `query`
   * carried the trim keys — see {@link TRIM_QUERY_KEYS}) but the server did not
   * report `trimmed: true`. A query that carries only NON-trim keys (§1-15's
   * `max_frames` cap) never arms this.
   *
   * It is deliberately a flag on an otherwise ordinary `ready` state rather
   * than a new `status`: the upload itself genuinely succeeded and the
   * `video_id` is perfectly usable — what failed is only the range narrowing
   * (an older plugin that drops the query, a backend without ffmpeg, an
   * unprobeable container). But USING that id would silently generate from the
   * wrong part of the footage, so `useChainForm` turns this into a Generate
   * block (`sourceTrimFailed`) with a "re-pick the file to use it whole"
   * escape hatch. Always `false` when no trim was requested — which is every
   * upload in production until the playback-position probe lands. */
  trimFailed: boolean;
  /** 素材（末尾）v2 (2026-08-15): the STORED video's own frame count, as the
   * SERVER measured it (`UploadVideoResponseBody.frame_count`) — `null` for an
   * audio slot, for any upload the server did not measure (see that field's
   * doc: only `max_frames`-carrying uploads are), and whenever the slot is not
   * `ready`.
   *
   * This is the single input that let the v1 「末尾フレーム数」 slider disappear:
   * the band is DERIVED from the material's real length now instead of being
   * guessed at by the user, and the server is the only party that can measure a
   * file it has just stored. Deliberately RAW — the count is at {@link fps}, not
   * at the generation rate — so the conversion lives in exactly one place
   * (`useChainForm`'s `endContextFrames` memo). */
  frames: number | null;
  /** 素材（末尾）v2: the frame rate {@link frames} was counted at, from the same
   * response, `null` under exactly the same conditions. Without it a frame count
   * is unusable: a 30fps material and a 24fps generation do not share a frame
   * axis. */
  fps: number | null;
}

export interface UseSourceUploadDeps {
  nativeBridge?: NativeBridge;
  /** W5 (反対スロット保持): an already-uploaded file to seed the slot with as a
   * ready state from mount, instead of the usual `idle`. Used by Create's
   * opposite-slot carry-over across a right-click remount (#2 carries the
   * existing audio; #3/#7 carry the existing reference video): the upload id is
   * still valid server-side, so the material survives the remount with no
   * re-upload. Applied ONLY in the lazy `useState` initializer (a later
   * re-render never re-applies it), so it behaves exactly like the frozen
   * `idle` start for every existing caller that omits it. `null`/`undefined` =
   * the ordinary `idle` start. */
  initial?: { id: string; fileName: string; filePath: string } | null | undefined;
}

export interface UseSourceUploadResult {
  state: SourceUploadState;
  /** Opens the native file picker for `kind`, then uploads whatever was
   * chosen. A `CANCELLED` pick (user dismissed the dialog) is silently
   * ignored — the previous state (if any) is left untouched, mirroring
   * `useKeyframes.addFromFile`'s CANCELLED handling. */
  pick: () => Promise<void>;
  /** Uploads an already-resolved local file path, skipping `ui.pickFile` —
   * used when the file was chosen through a different entry point, e.g.
   * Chain's unified source-input picker (`useChainForm.pickSource`), which
   * calls `ui.pickFile({kind:"imageOrVideo"})` itself and routes a video
   * extension here. Mirrors `pick()`'s upload half exactly, including the
   * generation-token guard against a stale in-flight upload winning a race
   * against a subsequent `clear()`/re-pick.
   *
   * §1-6 (source trim), contract v10: `query` adds URL query parameters to the
   * upload request — `/upload/video`'s `trim_start_sec`/`trim_duration_sec`
   * pair, produced by `timeline/sourceTrim.ts`'s `trimQuery()`, and (§1-15)
   * the chain reference video's `max_frames` cap. OMITTING it (every pre-§1-6
   * caller, and every no-trim decision) leaves the request params
   * byte-identical to before: the key is spread away, never sent as
   * `undefined`. Passing one whose keys include a TRIM key additionally arms
   * the {@link SourceUploadState.trimFailed} check on the response; a
   * non-trim query (`max_frames`) does not. */
  uploadPath: (filePath: string, fileName: string, query?: Record<string, string>) => Promise<void>;
  clear: () => void;
}

const INITIAL_STATE: SourceUploadState = {
  status: "idle",
  fileName: null,
  filePath: null,
  id: null,
  errorCode: null,
  trimFailed: false,
  frames: null,
  fps: null,
};

/** 素材（末尾）v2: reads the server's measured `frame_count`/`fps` off an upload
 * response body. Both are additive and OPTIONAL — an older backend omits them,
 * a plain (no-`max_frames`) upload answers `null` — so anything that is not a
 * finite positive number normalises to `null` ("not measured"), which is the
 * value every consumer's fallback path keys on. Kept as one helper so the two
 * fields can never be normalised differently. */
function measuredNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
}

function errorCodeOf(err: unknown): string {
  if (err instanceof BridgeError) return err.code;
  return "UNKNOWN";
}

function idFieldFor(kind: "video" | "audio"): "video_id" | "audio_id" {
  return kind === "video" ? "video_id" : "audio_id";
}

/** The query keys that ask `/upload/video` for a RANGE TRIM — the only ones
 * whose absence from the response's `trimmed` flag means the material is wrong.
 *
 * §1-15 (chain reference video): the query object is no longer trim-exclusive.
 * The chain's reference-video slot uploads with `max_frames` (a "cut only if the
 * file is longer than the whole chain could ever consume" cap), for which
 * `trimmed: false` is the NORMAL, correct answer — the file was already short
 * enough. Keying {@link SourceUploadState.trimFailed} off "a query was passed at
 * all" therefore produced a permanent false positive there. It is keyed off the
 * trim keys themselves instead, which leaves the V2V/#2 trim path's meaning
 * exactly as it was (`timeline/sourceTrim.ts`'s `trimQuery()` always emits
 * `trim_start_sec` together with `trim_duration_sec`). */
const TRIM_QUERY_KEYS = ["trim_start_sec"] as const;

function requestedTrim(query: Record<string, string> | undefined): boolean {
  if (query === undefined) return false;
  return TRIM_QUERY_KEYS.some((key) => key in query);
}

/**
 * Owns the "pick a local video/audio file -> upload it -> get back a
 * video_id/audio_id" flow shared by Chain mode's V2V (`source_video`) and
 * A2V (`source_audio`) submodes (Docs/API_REFERENCE.md §3.10/§3.11):
 * `ui.pickFile({kind})` -> `backend.uploadFile({kind, filePath})`. One
 * instance covers exactly one source slot — V2V and A2V each only ever need
 * a single file, unlike Create's `useKeyframes` which manages a *list* of
 * images, so this is deliberately simpler (no thumbnail, no per-item id
 * management).
 */
export function useSourceUpload(kind: "video" | "audio", deps: UseSourceUploadDeps = {}): UseSourceUploadResult {
  const nativeBridge = deps.nativeBridge ?? defaultBridge;
  // W5 (反対スロット保持): seed a carried-over ready state when `deps.initial` is
  // supplied — same field shape as a genuine post-upload `ready` state, so the
  // downstream duration-probe / gate logic can't tell it apart from a fresh
  // upload. Lazy initializer, so it applies once at mount only.
  const [state, setState] = useState<SourceUploadState>(() =>
    deps.initial
      ? {
          status: "ready",
          fileName: deps.initial.fileName,
          filePath: deps.initial.filePath,
          id: deps.initial.id,
          errorCode: null,
          // A carried-over slot was never (re)uploaded here, so no trim was
          // requested for it — see W5 above.
          trimFailed: false,
          // …and no response body was seen here either, so nothing was measured.
          // The carry-over is only ever used by Create's opposite-slot flows,
          // which have no band to derive.
          frames: null,
          fps: null,
        }
      : INITIAL_STATE,
  );
  // Generation token: bumped by `clear()` and at the start of every fresh
  // `uploadPath()` call. Each call captures its own generation and checks it
  // against the ref before ever calling `setState` after an `await` — so a
  // slow upload whose result arrives after the user has since cleared (or
  // started replacing) the source can't win the race and resurrect a stale
  // `ready`/`error` state. Needed once Chain's unified source-input picker can
  // route straight to `uploadPath` mid-upload (see `useChainForm.pickSource`);
  // a plain `clear()` racing an in-flight `pick()` was already a latent bug
  // this closes too.
  const generationRef = useRef(0);

  const uploadPath = useCallback(
    async (filePath: string, fileName: string, query?: Record<string, string>) => {
      generationRef.current += 1;
      const generation = generationRef.current;
      setState({
        status: "uploading",
        fileName,
        filePath,
        id: null,
        errorCode: null,
        trimFailed: false,
        frames: null,
        fps: null,
      });

      try {
        // §1-6: the spread is what guarantees byte-equality for every no-trim
        // upload — with `query` omitted the params object is EXACTLY
        // `{kind, filePath}`, the same shape (and the same native URL) as
        // before contract v10. Never `query: undefined`.
        const { status, body } = await nativeBridge.request("backend.uploadFile", {
          kind,
          filePath,
          ...(query ? { query } : {}),
        });
        if (generationRef.current !== generation) return; // superseded — discard
        const idField = idFieldFor(kind);
        const id =
          status >= 200 && status < 300 && body && typeof (body as Record<string, unknown>)[idField] === "string"
            ? ((body as Record<string, unknown>)[idField] as string)
            : null;
        if (id) {
          // §1-6: a trim we asked for but did not get. The upload still
          // succeeded (this stays a `ready` state with a usable id) — the flag
          // tells the form the material is the FULL file, so it can block
          // Generate instead of continuing from the wrong footage.
          const videoBody = body as UploadVideoResponseBody;
          const trimFailed = requestedTrim(query) && videoBody.trimmed !== true;
          setState({
            status: "ready",
            fileName,
            filePath,
            id,
            errorCode: null,
            trimFailed,
            // 素材（末尾）v2: additive, and normalised to `null` for every
            // response that does not carry a real measurement (an audio upload,
            // an older backend, a plain upload the server did not probe).
            frames: measuredNumber(videoBody.frame_count),
            fps: measuredNumber(videoBody.fps),
          });
        } else {
          const errorEnvelope = body as { error?: { code?: string } } | null;
          setState({
            status: "error",
            fileName,
            filePath,
            id: null,
            errorCode: errorEnvelope?.error?.code ?? `HTTP_${status}`,
            trimFailed: false,
            frames: null,
            fps: null,
          });
        }
      } catch (err) {
        if (generationRef.current !== generation) return; // superseded — discard
        setState({
          status: "error",
          fileName,
          filePath,
          id: null,
          errorCode: errorCodeOf(err),
          trimFailed: false,
          frames: null,
          fps: null,
        });
      }
    },
    [nativeBridge, kind],
  );

  const pick = useCallback(async () => {
    let picked: { filePath: string; fileName: string };
    try {
      picked = await nativeBridge.request("ui.pickFile", { kind });
    } catch (err) {
      const code = errorCodeOf(err);
      // A CANCELLED dialog leaves whatever was previously selected (if any)
      // untouched — this is a "change source" retry, not a fresh add, so
      // there's no pending card to roll back the way `useKeyframes` does.
      if (code !== "CANCELLED") {
        setState({
          status: "error",
          fileName: null,
          filePath: null,
          id: null,
          errorCode: code,
          trimFailed: false,
          frames: null,
          fps: null,
        });
      }
      return;
    }

    await uploadPath(picked.filePath, picked.fileName);
  }, [nativeBridge, kind, uploadPath]);

  const clear = useCallback(() => {
    generationRef.current += 1; // invalidate any in-flight uploadPath()
    setState(INITIAL_STATE);
  }, []);

  return { state, pick, uploadPath, clear };
}
