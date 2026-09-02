/**
 * Bridge RPC contract v1/v2/v3 — the single source of truth for the WebUI <->
 * native (.aux2) message shapes. Native code implements the mirror image of
 * this contract; if either side changes, this file must change first and the
 * other side must be updated to match.
 *
 * Transport (see native/src for the C++ side):
 *  - WebUI -> Native: `window.chrome.webview.postMessage(obj)` with a plain
 *    object (NOT JSON.stringify'd — WebView2 does the serialization).
 *  - Native -> WebUI: `window.chrome.webview.addEventListener('message', e => ...)`.
 *    `e.data` is either a BridgeResponse (has `id`) or a BridgeEvent (no `id`,
 *    has `event`/`data`, used for native-initiated pushes — unused in M1 but
 *    the receiving plumbing exists so later milestones can add events without
 *    changing the transport).
 *
 * v2 (M2) adds the methods native uses to proxy the LTX23 backend REST API
 * (WinHTTP, avoiding WebView2 CORS restrictions) and to hand a downloaded mp4
 * off to the AviUtl2 timeline. See Docs/API_REFERENCE.md for the backend
 * schema that `backend.request` carries as an opaque JSON body.
 *
 * v3 (M4) adds the methods the Create screen's I2V keyframe panel needs:
 * capturing the current timeline frame to a PNG (`timeline.captureFrame`),
 * uploading a local file to the backend's multipart upload endpoints
 * (`backend.uploadFile`), picking a local file via a native file dialog
 * (`ui.pickFile`), and rendering a downscaled preview of a local image as a
 * data URL for a keyframe card's thumbnail (`ui.makeThumbnail`).
 *
 * v4 (M7b) adds the settings panel's connection-settings methods:
 * `settings.get`/`settings.set` read and update the backend URL the native
 * host proxies `backend.request`/`backend.uploadFile` to. v4.1 additionally
 * gives `backend.request` an optional per-call `timeoutMs`, for requests
 * whose backend-side processing can legitimately run longer than native's
 * default WinHTTP timeout (e.g. `POST /jobs/{id}/join`, which `joinJob()`
 * calls with `timeoutMs: 120_000` — see `api/client.ts`). `RequestDispatcher`
 * also honors this as its own local wait ceiling for that specific call (see
 * `bridge/requestDispatcher.ts`), so the WebUI doesn't give up on the
 * WebView2 round trip before native's own (longer) timeout has a chance to
 * fire.
 *
 * v5 (timeline generation-AI bridge) adds the `timeline.*` methods the WebUI
 * needs to feed the AviUtl2 timeline into the generation flow and reserve
 * space for pending results: `timeline.getSelection` (read the current
 * selection/cursor), `timeline.cutoutRange`/`timeline.extractAudio` (bounce a
 * range to a temp video/audio file for `backend.uploadFile`), and the
 * provisional-placeholder lifecycle (`insertProvisional`/`resolveProvisional`/
 * `updateProvisionalText`/`scanProvisionals`). It also adds the native->WebUI
 * `timeline.menuInvoked` event (see `TIMELINE_MENU_INVOKED_EVENT`), pushed
 * when the user invokes one of the plugin's timeline context-menu actions.
 *
 * v6 (batch A2V + wav-duration bridge) adds the filesystem-oriented methods
 * the batch A2V flow and its wav-length auto-adjustment need: `ui.pickFolder`
 * (native folder-picker dialog), `fs.listFiles` (list a folder's entries,
 * optionally with `.wav` header durations), and `fs.probeAudioDuration` (a
 * single file's wav duration, never erroring for non-wav/unreadable files).
 * It also extends `backend.downloadVideo` with optional
 * `destDir`/`fileName`/`noClobber` params so a batch run can save each clip
 * to a caller-chosen folder without clobbering same-named outputs — all
 * three are additive and backward compatible; omitting them preserves exact
 * pre-v6 behavior.
 *
 * v7 (drag-and-drop) adds two things. First, `ui.pickFile`'s `"imageOrVideo"`
 * kind (Chain's unified source-input picker) had already shipped natively
 * before this doc section was written up — see that field's own comment.
 * Second, `ui.resolveDroppedFiles`: WebView2's `AllowExternalDrop` defaults
 * to true, so a plain DOM `dragover`/`drop` over the page already fires
 * without any extra native drop-target plumbing — but a dropped file's JS
 * `File` object never carries a real local path. The fix is
 * `chrome.webview.postMessageWithAdditionalObjects(request, files)`
 * (`NativeBridge.requestWithFiles`, `webviewBridge.ts`): the extra COM
 * objects arrive on the native side as this message's AdditionalObjects,
 * each resolved to an `ICoreWebView2File`'s real path, which native then
 * force-injects into the request's `params.__droppedPaths` (a reserved key)
 * before dispatching it. `ui.resolveDroppedFiles`'s own declared params take
 * no visible fields (`{}`) — the caller never has to (and cannot
 * meaningfully) set `__droppedPaths` itself. Native checks this message's
 * AdditionalObjects FIRST: if it carries any, the key is always overwritten
 * with whatever they resolve to (even an empty array, when passed a real
 * request with none) — this is what makes a genuine drop's empty `params: {}`
 * still work, since nothing about the raw JSON text needs to mention the key
 * at all. Only when there are no AdditionalObjects does native fall back to
 * checking whether the raw message merely *mentions* `__droppedPaths` (a
 * cheap substring pre-check) and, if so, force it to `[]` — this is the
 * anti-spoofing net: a page-authored `__droppedPaths` with no real drop
 * behind it can never survive and turn this method into an
 * arbitrary-local-file-read primitive. Ordinary messages that have neither
 * AdditionalObjects nor any mention of the key are left untouched entirely.
 * This is a quick, in-memory, effectively-synchronous method — it keeps the
 * default 10s local timeout (it is NOT added to `NO_LOCAL_TIMEOUT_METHODS`).
 *
 * v8 (replace-insert) adds `timeline.insertMediaForJob`: a single atomic RPC for
 * the 🎞 "place & replace" insert that either swaps the finished media into a
 * job's still-present provisional marker (mode `"replaced"`) or, when no marker
 * remains, inserts it exactly like `timeline.insertMedia` (mode `"inserted"`).
 * It subsumes the old two-call `insertMedia` + `deleteProvisionalByJob` sequence
 * on the per-clip insert path. `timeline.insertMedia` stays for the V2V *joined*
 * insert (which must never replace a per-clip marker) and
 * `timeline.deleteProvisionalByJob` stays for the legacy rollback path.
 *
 * v9 (media-info probe) adds `fs.probeMediaInfo`: a best-effort, synchronous
 * probe of a single local media file's duration/resolution, for the IC-LoRA /
 * A2V source cards' duration readout (`12.3s`). Like `fs.probeAudioDuration`
 * it never rejects for a bad/unreadable/unsupported file — a missing edit
 * handle, a `get_media_info` failure, or an unsupported container all resolve
 * to `{durationSec: 0, width: 0, height: 0}` (all three non-nullable, `0` =
 * "unknown"; `durationSec` is seconds, a still image reports `0`). Only a
 * missing/empty/non-string `filePath` rejects (`BAD_REQUEST`).
 *
 * v10 (source trim, §1-6) adds `backend.uploadFile`'s optional `query`: a flat
 * `Record<string, string>` native URL-encodes and appends to the upload
 * endpoint's URL. It exists for `/upload/video`'s `trim_start_sec`/
 * `trim_duration_sec` pair (upload only the ribbon's range of a source video
 * instead of the whole file), but is deliberately generic — nothing about the
 * transport is trim-specific. Strictly additive in both directions: an OMITTED
 * (or empty) `query` produces the exact same URL as before v10, and an older
 * native build ignores the key entirely, so the upload merely stays untrimmed.
 * The response gains an equally optional `trimmed` flag so the caller can tell
 * "the trim was applied" from "the trim was silently ignored" (the WebUI blocks
 * Generate on the latter rather than generating from the wrong footage).
 *
 * v10's OTHER half — the `timeline.getSelection` playback fields the trim
 * decision needs (`playbackStartSec`/`playbackEndSec`/`hasPlaybackRange`/
 * `playbackSpeed`/`loopPlay`/`sectionCount`) — landed with the real-hardware
 * probe on 2026-08-01 and is now part of this contract (see the
 * `timeline.getSelection` entry below). The unit is SECONDS on the source's own
 * time axis, independent of the project fps. They are declared OPTIONAL for one
 * reason only: an older plugin build simply does not emit them, and the WebUI
 * must then fall back to the pre-v10 whole-file upload instead of trimming from
 * a position it never learned.
 *
 * v11 (§3-13) adds `mediaFps` to `timeline.getSelection`'s `selected[]` entries:
 * the selected object's material framerate, probed from the backing file via
 * Media Foundation (`MFCreateSourceReaderFromURL` + `GetNativeMediaType` +
 * `MF_MT_FRAME_RATE`). It is the RAW rate the container reports (`29.97`, not
 * pre-snapped to `30`) — the same non-nullable, `0` = "unknown" contract as
 * `mediaWidth`/`mediaHeight` above, so `mediaFps` is never `null`, only absent
 * (older build) or `0` (probe failed, or an unsupported container — mkv/webm
 * commonly land here, and that is a NORMAL outcome, not an error). Integer
 * snapping (`29.97` -> `30`, `23.976` -> `24`) is deliberately NOT native's
 * job: it happens on the WebUI side, via `modes/single/paramUtils.ts`'s
 * `snapFrameRate` (§3-71/§3-72's one source of truth for the rounding rule),
 * so the raw value stays available unrounded here.
 * `mediaFps` is declared OPTIONAL for the same single reason the six v10
 * fields are: an older native build simply does not emit it yet.
 */

/** All RPC methods defined as of contract v6. */
export type BridgeMethod = keyof BridgeParamsMap;

/** Request parameter shape for each method. M1 methods take no parameters. */
export interface BridgeParamsMap {
  ping: Record<string, never>;
  getEditInfo: Record<string, never>;
  /** Proxies one LTX23 backend REST call through native WinHTTP. `path` is
   * the full request path including the `/api/v1` prefix (e.g.
   * `/api/v1/status`), matching the shape of `JobResponse.result.video_url`
   * so paths never need re-deriving on either side. */
  "backend.request": {
    method: "GET" | "POST" | "DELETE";
    path: string;
    query?: Record<string, string>;
    body?: object;
    /** v4.1: caps how long this specific call may take, overriding native's
     * default WinHTTP timeout. Omitted -> native's default. */
    timeoutMs?: number;
  };
  /** Downloads a completed job's mp4 to a native temp/cache location and
   * returns the local file path, ready for `timeline.insertMedia`.
   *
   * Contract v6 adds three optional, backward-compatible params for the
   * batch A2V flow, which needs each clip saved to a caller-chosen folder
   * instead of native's default temp/cache location. Omitting all three
   * preserves the exact pre-v6 behavior. */
  "backend.downloadVideo": {
    jobId: string;
    joined?: boolean;
    /** Contract v6: save to this folder instead of native's default temp/
     * cache location. Missing parent folders are created. */
    destDir?: string;
    /** Contract v6: desired file name (including extension) instead of
     * native's default `<jobId>[_joined].mp4` naming. */
    fileName?: string;
    /** Contract v6: when true, never overwrite an existing file at the
     * resolved path — append `_2`, `_3`, ... to the name's stem instead until
     * a free name is found. The result's `filePath` reflects the name
     * actually written. */
    noClobber?: boolean;
    /** When true, if the resolved destination already exists and its size is
     * `> 0`, skip the download entirely and return that existing path/size.
     * A job's generated video is immutable, so re-inserting a completed job's
     * clip must not re-download over a file AviUtl2 may still hold open (a
     * shared-write violation — "Could not open destination file"). Intended
     * for the plain (per-clip) path only; do NOT use with `joined`, whose
     * output changes with the crossfade setting and must always be refetched.
     * Ignored together with `noClobber` (which already never overwrites). */
    reuseIfPresent?: boolean;
  };
  /** Inserts a local media file into the current AviUtl2 timeline at the given
   * layer/frame (or the current selection layer / cursor frame when omitted).
   *
   * Still current for the V2V *joined* insert path (`downloadAndInsert` with
   * `joined: true`): the joined clip is a distinct, longer artifact that must
   * NOT replace any single per-clip reservation's provisional marker, so it is
   * appended at the cursor via this method rather than through
   * `timeline.insertMediaForJob`. The per-clip / reservation-backed 🎞 insert
   * goes through `timeline.insertMediaForJob` instead. */
  "timeline.insertMedia": {
    filePath: string;
    layer?: number;
    frame?: number;
  };
  /** Replace-insert tied to a job's provisional marker (the 🎞 "place & replace"
   * insert). Native locates the still-present provisional placeholder for
   * `jobId` (exact job-id match across all layers, never a position fallback):
   *  - found -> the finished media at `filePath` REPLACES it in place (same
   *    layer/frame, media's real length, one undo step) — `mode: "replaced"`;
   *  - not found -> native inserts exactly like `timeline.insertMedia` (current
   *    selection layer / cursor frame / real length) so normal-generation 🎞
   *    behavior is unchanged — `mode: "inserted"`.
   * See the result's `mode`/`usedFallback`. Subsumes the old two-call
   * `insertMedia` + `deleteProvisionalByJob` sequence in one atomic native call. */
  "timeline.insertMediaForJob": {
    jobId: string;
    filePath: string;
  };
  /** Returns the backend's origin (scheme+host+port, no path) for building
   * direct `<video src>` / `<img src>` URLs that bypass the bridge. */
  "backend.getBaseUrl": Record<string, never>;
  /** Captures the current (or a given) timeline frame to a PNG file on disk
   * (contract v3, M4), for use as an I2V conditioning image. `frame` omitted
   * means "the current edit cursor frame". */
  "timeline.captureFrame": {
    frame?: number;
  };
  /** Uploads a local file to the LTX23 backend's multipart upload endpoint
   * for the given `kind` (`/upload/image|video|audio`) via native WinHTTP,
   * bypassing WebView2 CORS restrictions (contract v3, M4). Mirrors
   * `backend.request`'s result shape: any HTTP status the backend returns is
   * a normal (non-throwing) result — only transport-level failures reject. */
  "backend.uploadFile": {
    kind: "image" | "video" | "audio";
    filePath: string;
    /** Contract v10 (source trim, §1-6): extra query parameters to append to
     * the upload URL, URL-encoded by native. Omit (the pre-v10 shape) for the
     * plain "upload the whole file" call — the resulting URL is then
     * byte-identical to before v10. Today's only producer is
     * `timeline/sourceTrim.ts`'s `trimQuery()`, which returns `undefined`
     * (so the key is spread away entirely) whenever no trim applies. */
    query?: Record<string, string>;
  };
  /** Opens a native file-picker dialog scoped to the given media kind
   * (contract v3, M4). `"imageOrVideo"` (Chain's unified source-input picker,
   * task brief "Chainのソース入力欄一本化") shows a single dialog whose default
   * filter accepts both image and video extensions — the caller then routes
   * the picked file by extension (`modes/chained/sourceRouting.ts`) rather than
   * the dialog itself picking a kind up front. */
  "ui.pickFile": {
    kind: "image" | "video" | "audio" | "imageOrVideo";
  };
  /** Renders a downscaled preview of a local image file as a data URL, for a
   * keyframe card's thumbnail (contract v3, M4). `maxDim` bounds the longer
   * side in pixels (native default 256 when omitted). */
  "ui.makeThumbnail": {
    filePath: string;
    maxDim?: number;
  };
  /** Returns the currently configured backend URL (contract v4, M7b's
   * connection-settings panel). Distinct from `backend.getBaseUrl`, which
   * exists for building direct `<video src>`/`<img src>` URLs — `settings.get`
   * is for showing/editing the value itself. */
  "settings.get": Record<string, never>;
  /** Updates the backend URL native proxies `backend.request`/
   * `backend.uploadFile` to (contract v4, M7b). Omitting `baseUrl` (or
   * passing `{}`) is a no-op read — the response always echoes the
   * (possibly unchanged) resulting value. An invalid URL rejects with
   * `BAD_REQUEST`. */
  "settings.set": {
    baseUrl?: string;
  };
  /** Contract v5 (timeline generation-AI bridge). Returns a snapshot of the
   * current AviUtl2 timeline selection/cursor plus project rate/scale, used by
   * the WebUI to decide what range to cut out / hand to the generation flow.
   * Takes no parameters. */
  "timeline.getSelection": Record<string, never>;
  /** Contract v5. Renders the given layer/frame range to a temp mp4 on disk
   * (optionally muxing audio per `audioMode`), returning the local file path
   * ready for `backend.uploadFile`. `audioMode:"mix"` bounces the full audio
   * mixdown; `"solo"` keeps only `soloKeepLayers`. */
  "timeline.cutoutRange": {
    layer: number;
    frameStart: number;
    frameCount: number;
    withAudio: boolean;
    audioMode: "mix" | "solo";
    soloKeepLayers: number[];
  };
  /** Contract v5. Extracts just the audio for the given range to a temp file
   * on disk (no video), for an audio-conditioned generation. */
  "timeline.extractAudio": {
    layer: number;
    frameStart: number;
    frameCount: number;
    audioMode: "mix" | "solo";
    soloKeepLayers: number[];
  };
  /** Contract v5 (right-click redesign spec §5). Inserts a placeholder
   * ("provisional") text object reserving the layer/frame span a not-yet-
   * finished generation job (`jobId`) will later occupy, so the timeline shows
   * progress immediately. Resolved later by `timeline.resolveProvisional`.
   *
   * - Length: native computes the timeline length with
   *   `ProjectFramesForPixels(numFrames, genFps, projectFps)` (projectFps from
   *   the edit info's rate/scale). `genFps` must be `> 0`.
   * - Position: native resolves the target from the selected material's frame
   *   range (`"A"`/`"B"`) or the cursor (`"C"`) per spec §5-9. Placement `"A"`
   *   needs `materialLayer`+`materialFrameEnd`, `"B"` needs `materialFrameStart`,
   *   `"C"` needs `cursorLayer`+`cursorFrame`. */
  "timeline.insertProvisional": {
    jobId: string;
    displayText: string;
    /** 4-stage label prefix (spec §5-5). Omitted/undefined -> native's default
     * "⏳生成中："; the stage-1 reservation insert passes the ASCII, emoji-free
     * "AI video will be placed here" prefix (with an empty `displayText`, since
     * native truncates the body but not the prefix) so the placeholder reads
     * "AI video will be placed here…" (I13 owner decision). */
    textPrefix?: string;
    /** Generation frame count; with `genFps`, native resolves the timeline
     * length. */
    numFrames: number;
    /** Generation fps for the `numFrames` length resolution (must be `> 0`). */
    genFps: number;
    /** Placement system (spec §5-9). */
    placement: "A" | "B" | "C";
    /** Selected material's layer / frame range (inclusive end), for placement
     * `"A"`/`"B"`. */
    materialLayer?: number;
    materialFrameStart?: number;
    materialFrameEnd?: number;
    /** Right-click cursor position, for placement `"C"`. */
    cursorLayer?: number;
    cursorFrame?: number;
  };
  /** I3 (right-click redesign spec §5-3). Atomically re-places a provisional
   * reservation: deletes the placeholder for `oldJobId` (if found — pass an
   * empty string to skip the search and create only) and creates a fresh one
   * for `newJobId` at the position resolved from `placement` (spec §5-9), all
   * in one edit section (one undo step). The length is always resolved natively
   * from `numFrames`+`genFps` (no legacy `lengthFrames` on this RPC). Used for
   * Generate-time length/jobId hand-off and the "one reservation seat" move on
   * re-clicking a generation-origin menu item. */
  "timeline.updateProvisionalReservation": {
    /** Job id of the placeholder to delete; empty string = create only. */
    oldJobId: string;
    newJobId: string;
    displayText: string;
    /** 4-stage label prefix (spec §5-5), same contract as
     * `insertProvisional.textPrefix`. Omitted -> native's default "Generating: "
     * (ASCII since I14; the emoji default was retired with the tofu fix). In
     * production the prefix is always passed explicitly: the Generate-time bind
     * sends "Generating: " (stage 2) and a still-unbound reservation MOVE sends
     * the stage-1 "AI video will be placed here" prefix. */
    textPrefix?: string;
    numFrames: number;
    genFps: number;
    placement: "A" | "B" | "C";
    /** Selected material's layer / frame range (inclusive end), for placement
     * `"A"`/`"B"`. */
    materialLayer?: number;
    materialFrameStart?: number;
    materialFrameEnd?: number;
    /** Right-click cursor position, for placement `"C"`. */
    cursorLayer?: number;
    cursorFrame?: number;
  };
  /** Contract v5. Replaces the provisional placeholder for `jobId` with the
   * finished video at `videoFilePath`. `mode:"replaced"` when the placeholder
   * was still present and swapped in place; `"insertedReserved"` when the
   * placeholder was gone (e.g. user deleted it) and the video was inserted at
   * the originally reserved layer/frame instead. */
  "timeline.resolveProvisional": {
    jobId: string;
    videoFilePath: string;
    reservedLayer: number;
    reservedFrame: number;
    lengthFrames: number;
  };
  /** Contract v5. Updates the visible text of a still-pending provisional
   * placeholder (progress %, or a failure message when the job fails). */
  "timeline.updateProvisionalText": {
    jobId: string;
    layer: number;
    frame: number;
    text: string;
  };
  /** Contract v5. Lists provisional placeholders whose backing job is no
   * longer known to the WebUI ("orphans"), so they can be cleaned up. Takes
   * no parameters. */
  "timeline.scanProvisionals": Record<string, never>;
  /** I13 (right-click redesign spec §5-10). Deletes the still-present ✅
   * provisional placeholder for `jobId` — the cleanup tied to the user's
   * explicit successful 🎞 insert (`timeline.insertMedia`). Only the first
   * match is removed; a missing placeholder is a no-op success (idempotent),
   * so `deleted` reports whether anything was actually removed. Never removes a
   * ❌ failed marker: a 🎞 insert is only possible for a *completed* job while
   * the ❌ marker only exists for a *failed* one, so they never collide. */
  "timeline.deleteProvisionalByJob": {
    jobId: string;
  };
  /** Contract v6 (batch A2V bridge). Opens a native folder-picker dialog.
   * `title` is an optional dialog title/prompt shown to the user; native uses
   * a default title when omitted. */
  "ui.pickFolder": {
    title?: string;
  };
  /** Contract v6. Lists a folder's entries (non-recursive). `extensions`,
   * when given, filters to file names whose extension (including the leading
   * `.`, e.g. `.wav`) case-insensitively matches one of the list; omitted
   * means no filtering. `withAudioDuration:true` additionally measures each
   * `.wav` entry's duration by reading its header (`durationSec`); non-`.wav`
   * files and `.wav` files native fails to read report `durationSec: 0` —
   * this never turns into an error. **Native's result order is unspecified/
   * implementation-dependent — callers must sort `files` themselves** (e.g.
   * by `name`) rather than relying on any particular order. */
  "fs.listFiles": {
    folderPath: string;
    extensions?: string[];
    withAudioDuration?: boolean;
  };
  /** Contract v6. Probes a single local file's audio duration by reading its
   * wav header. Non-`.wav` files and files native fails to read resolve
   * (never reject) with `{durationSec: 0, isWav: false}` — this is meant for
   * best-effort UI hints (e.g. auto-adjusting a generation length to match an
   * uploaded wav), not as a validity check. */
  "fs.probeAudioDuration": {
    filePath: string;
  };
  /** Contract v9. Probes a single local media file's duration/resolution via
   * native's `get_media_info` (best-effort UI hint for the IC-LoRA/A2V source
   * cards' `12.3s` readout). A missing edit handle, a `get_media_info`
   * failure, or an unsupported container all resolve (never reject) with
   * `{durationSec: 0, width: 0, height: 0}`; only a missing/empty/non-string
   * `filePath` rejects with `BAD_REQUEST`. */
  "fs.probeMediaInfo": {
    filePath: string;
  };
  /** Contract v7. Resolves the file(s) attached via `requestWithFiles` (i.e.
   * sent through `postMessageWithAdditionalObjects`) to their real local
   * paths. Declared params are empty (`{}`) — `__droppedPaths` is a reserved
   * key native force-injects itself, keyed off this message's own
   * AdditionalObjects (see the v7 note in this file's header comment); a
   * caller has no field to set here. */
  "ui.resolveDroppedFiles": Record<string, never>;
}

/** Result shape for each method's successful response. */
export interface BridgeResultMap {
  ping: {
    pong: true;
    pluginVersion: string;
  };
  /** Mirrors AviUtl2 SDK's EDIT_INFO. Native (`bridge_core.cpp`) always also
   * sends `layer`/`frameMax`/`layerMax` (the current layer and the project's
   * max frame / max layer), declared here now that consumers exist; any other
   * extra fields from native are ignored (BRIDGE_CONTRACT.md §9). */
  getEditInfo: {
    width: number;
    height: number;
    rate: number;
    scale: number;
    sampleRate: number;
    frame: number;
    layer: number;
    frameMax: number;
    layerMax: number;
  };
  "backend.request": {
    /** HTTP status returned by the backend. 4xx/5xx are a normal (non-error)
     * result here — only transport-level failures reject the promise. */
    status: number;
    body: object | null;
  };
  "backend.downloadVideo": {
    filePath: string;
    sizeBytes: number;
  };
  "timeline.insertMedia": {
    inserted: true;
    layer: number;
    frame: number;
  };
  /** Result of the replace-insert (🎞 "place & replace"). `mode:"replaced"` when
   * a provisional marker for `jobId` was found and the media swapped into its
   * slot; `"inserted"` when no marker was found and the media was inserted with
   * the plain `timeline.insertMedia` behavior. `layer`/`frame` are where the
   * media landed. `usedFallback` is true only when a `replaced` create collided
   * at the marker slot and native retried on `layer_max+1`. */
  "timeline.insertMediaForJob": {
    ok: boolean;
    mode: "replaced" | "inserted";
    layer: number;
    frame: number;
    usedFallback: boolean;
  };
  "backend.getBaseUrl": {
    baseUrl: string;
  };
  "timeline.captureFrame": {
    filePath: string;
    width: number;
    height: number;
    frame: number;
  };
  "backend.uploadFile": {
    /** HTTP status returned by the backend. 4xx/5xx are a normal (non-error)
     * result here — only transport-level failures reject the promise. */
    status: number;
    /** The endpoint's JSON body, verbatim. Deliberately left as the opaque
     * `object` (callers narrow the one field they need — `video_id`,
     * `image_id`, … — with a cast), so a backend that adds a response field
     * never has to be mirrored here. Contract v10's `trimmed` flag is one such
     * additive field: see {@link UploadVideoResponseBody}. */
    body: object | null;
  };
  "ui.pickFile": {
    filePath: string;
    fileName: string;
  };
  "ui.makeThumbnail": {
    dataUrl: string;
    width: number;
    height: number;
    sourceWidth: number;
    sourceHeight: number;
  };
  "settings.get": {
    baseUrl: string;
  };
  "settings.set": {
    baseUrl: string;
  };
  /** Contract v5. `selected` is the set of currently-selected timeline objects;
   * `hasRange` is true when the user has a time range (not just a cursor)
   * selected, bounded by `rangeStart`/`rangeEnd` (frames). `cursorFrame`/
   * `cursorLayer` are the edit cursor position; `rate`/`scale`/`sampleRate`
   * mirror `getEditInfo`'s project fields for frame<->time conversions. */
  "timeline.getSelection": {
    hasRange: boolean;
    rangeStart: number;
    rangeEnd: number;
    selected: Array<{
      layer: number;
      frameStart: number;
      frameEnd: number;
      effectName: string;
      filePath: string | null;
      objectName: string | null;
      /** A text object's body, for the #8 "append to the main prompt" action.
       * `null` for any non-text object (video/image/audio), matching native's
       * has_text_content nullable pattern. */
      textContent: string | null;
      /** The selected object's actual media resolution, when native could
       * resolve it (video/image objects). `0` means "unknown" (audio/shape
       * objects, or a lookup that failed) — never `null`. Consumed by the
       * right-click generation flow to match output resolution to the input
       * material (`timeline/deriveGenerationParams.ts`). */
      mediaWidth: number;
      mediaHeight: number;
      /** The selected object's real media duration in seconds, from
       * `get_media_info`'s `total_time`. `0` means "unknown" (still image, an
       * audio-less object, or a lookup that failed) — never `null`. Consumed by
       * the #1/#2 length guard (see the redesign spec §4-5). */
      mediaDurationSec: number;
      /** Contract v10 (source trim, §1-6): the window of the BACKING FILE this
       * object plays, in SECONDS on the source's own time axis — the unit was
       * established on real hardware on 2026-08-01 (AviUtl2's 再生位置 item is a
       * 4-field CSV whose first two fields are these seconds, independent of the
       * project fps). `hasPlaybackRange` is the explicit "was it really read"
       * flag, because `0` is both a perfectly normal start and the natural
       * "unavailable" default — the same non-nullable + has-flag shape native
       * already uses for `mediaWidth`/`mediaHeight`.
       *
       * All six v10 fields are optional here ONLY to tolerate an older plugin
       * build that does not emit them; the current native always does. Consumed
       * by `timeline/sourceTrim.ts` — see `decideSourceTrim`. */
      playbackStartSec?: number;
      playbackEndSec?: number;
      hasPlaybackRange?: boolean;
      /** AviUtl2's 再生速度, normalized to a 1.0 scale by native (the raw value is
       * a percentage string like `"100.00"`). A REPORTED non-neutral speed makes
       * the trim decision skip; an unreported one does not. */
      playbackSpeed?: number;
      /** AviUtl2's ループ再生. `true` makes the object replay its window, so a
       * single start/duration pair no longer describes what is on screen. */
      loopPlay?: boolean;
      /** Number of 中間点-delimited sections (`get_object_section_num`). `>= 2`
       * means the time mapping is piecewise; native reports `1` when it cannot
       * ask. */
      sectionCount?: number;
      /** Contract v11 (§3-13): the object's material framerate, probed via
       * Media Foundation. RAW (unsnapped) — `29.97` stays `29.97`, never
       * pre-rounded to `30`. Non-nullable, `0` = "unknown" (the same contract
       * as `mediaWidth`/`mediaHeight`); a failed probe (mkv/webm are the
       * common case) is a normal `0`, not an error. Optional only because an
       * older native build does not emit it. Integer snapping happens on the
       * WebUI side, in `modes/single/paramUtils.ts`'s `snapFrameRate`, not
       * here. */
      mediaFps?: number;
    }>;
    cursorFrame: number;
    cursorLayer: number;
    rate: number;
    scale: number;
    sampleRate: number;
  };
  "timeline.cutoutRange": {
    filePath: string;
    width: number;
    height: number;
    frameCount: number;
    hasAudio: boolean;
  };
  "timeline.extractAudio": {
    filePath: string;
    durationSec: number;
    sampleRate: number;
    hasAudioStream: boolean;
  };
  "timeline.insertProvisional": {
    inserted: boolean;
    layer: number;
    frame: number;
    objectName: string;
    /** The position the placeholder actually landed at (mirrors `layer`/`frame`)
     * and whether native had to fall back to `layer_max+1` because the requested
     * slot was occupied (spec §5-4). */
    placedLayer?: number;
    placedFrame?: number;
    usedFallback?: boolean;
  };
  /** I3 (spec §5-3). `ok` is true when the new placeholder was created;
   * `deletedOld` reports whether an old placeholder was found and removed;
   * `placedLayer`/`placedFrame` are where the new placeholder actually landed;
   * `usedFallback` is true when native fell back to `layer_max+1` (spec §5-4). */
  "timeline.updateProvisionalReservation": {
    ok: boolean;
    deletedOld: boolean;
    placedLayer: number;
    placedFrame: number;
    usedFallback: boolean;
  };
  "timeline.resolveProvisional": {
    mode: "replaced" | "insertedReserved";
    layer: number;
    frame: number;
  };
  "timeline.updateProvisionalText": {
    updated: boolean;
  };
  "timeline.scanProvisionals": {
    orphans: Array<{ jobId: string; layer: number; frame: number }>;
  };
  /** I13 (spec §5-10). `ok` is always true (idempotent); `deleted` reports
   * whether a placeholder was actually found and removed. */
  "timeline.deleteProvisionalByJob": {
    ok: boolean;
    deleted: boolean;
  };
  "ui.pickFolder": {
    folderPath: string;
  };
  "fs.listFiles": {
    files: Array<{
      name: string;
      path: string;
      sizeBytes: number;
      mtimeMs: number;
      /** Wav header duration in seconds when the request set
       * `withAudioDuration: true` and this is a readable `.wav` file; `0`
       * otherwise (non-`.wav`, or native failed to read it — never an
       * error). */
      durationSec: number;
    }>;
  };
  "fs.probeAudioDuration": {
    durationSec: number;
    isWav: boolean;
  };
  /** Contract v9. `durationSec` is seconds (a still image or an unknown
   * duration reports `0`); `width`/`height` are the media's pixel resolution
   * (`0` when unknown). All three are non-nullable — a probe that couldn't
   * resolve anything reports `{durationSec: 0, width: 0, height: 0}` rather
   * than rejecting. */
  "fs.probeMediaInfo": {
    durationSec: number;
    width: number;
    height: number;
  };
  /** Contract v7. One entry per dropped file that resolved to a real local
   * path (in AdditionalObjects order); a drop native couldn't resolve at all
   * (or a message with no dropped files) yields `files: []`. */
  "ui.resolveDroppedFiles": {
    files: Array<{ filePath: string; fileName: string }>;
  };
}

/**
 * The fields the WebUI reads out of `POST /upload/video`'s JSON body (the
 * `backend.uploadFile` result's opaque `body`, narrowed at the one call site
 * that needs it — `modes/chained/useSourceUpload.ts`).
 *
 * Contract v10 (§1-6) adds `trimmed`: the server sets it `true` only when it
 * actually cut the uploaded video down to the requested
 * `trim_start_sec`/`trim_duration_sec` window. It is `false` (or, against an
 * older backend, absent) when no trim was requested — and also when a trim WAS
 * requested but could not be performed (no ffmpeg, an unprobeable container, an
 * out-of-range window): the server logs a warning and stores the untrimmed file
 * rather than failing the upload. The caller must therefore treat
 * `trimmed !== true` after requesting a trim as "the trim did not happen", which
 * is exactly what `useSourceUpload`'s `trimFailed` flag reports.
 */
export interface UploadVideoResponseBody {
  video_id?: string;
  /** Contract v10. See this interface's doc comment — `!== true` after a trim
   * request means the upload holds the FULL source file. */
  trimmed?: boolean;
  /** 素材（末尾）v2 (2026-08-15), additive: the number of frames the STORED
   * video actually has, as the server measured it. Filled ONLY for uploads that
   * carried a `max_frames` query (the end-source and reference slots) — a plain
   * upload does not pay for the extra probe and gets `null`/an absent key, and
   * so does an upload whose requested trim failed. `useSourceUpload` copies it
   * onto its state and `useChainForm` derives the end band from it, which is
   * what removed the v1 "末尾フレーム数" slider entirely. */
  frame_count?: number | null;
  /** 素材（末尾）v2, additive: the STORED video's own frame rate, alongside
   * {@link frame_count} and filled under exactly the same conditions. Needed
   * because `frame_count` is counted at the SOURCE rate: a 30fps material and a
   * 24fps generation do not share a frame axis, so the count has to be converted
   * before it can bound a band. `null` means "not measured" and sends the caller
   * down its duration-based fallback. */
  fps?: number | null;
}

/** Contract v5 native->WebUI event name: fired when the user invokes one of
 * the plugin's timeline context-menu actions. */
export const TIMELINE_MENU_INVOKED_EVENT = "timeline.menuInvoked";

/** Payload of the `timeline.menuInvoked` event. `selection` has the same shape
 * `timeline.getSelection` resolves with, but it is only a lightweight snapshot
 * taken inside the menu callback: it does NOT carry `effectName`, `filePath`,
 * the media dimensions, or `textContent` (those are left null/empty/0 to avoid
 * the costly per-object lookups in the callback). Before the WebUI actually
 * needs the material — to classify the object type, load a file, or read a text
 * body — it must re-query `timeline.getSelection` to fill those in (see
 * `timeline/menuSelection.ts`'s `resolveMenuSelection`). */
export interface TimelineMenuInvokedData {
  action: string;
  selection: BridgeResultMap["timeline.getSelection"];
}

/** Contract v5 native->WebUI event name: fired by the plugin's project-load
 * handler after a project is loaded (and, per the SDK, at project
 * initialization). The WebUI reacts by calling `timeline.scanProvisionals` to
 * find orphaned provisional placeholders left by a prior session and offer to
 * re-generate them. Native pushes this with an empty payload (see
 * `plugin.cpp`'s `OnProjectLoad`). */
export const TIMELINE_PROJECT_LOADED_EVENT = "timeline.projectLoaded";

/** Payload of the `timeline.projectLoaded` event: intentionally empty — the
 * event is a bare trigger, carrying no data (the WebUI queries
 * `timeline.scanProvisionals` itself in response). */
export type TimelineProjectLoadedData = Record<string, never>;

export type ParamsOf<M extends BridgeMethod> = BridgeParamsMap[M];
export type ResultOf<M extends BridgeMethod> = BridgeResultMap[M];

/** Error codes defined by contract v1/v2/v3. Native may add new codes over
 * time; unrecognized codes are treated as opaque strings by the WebUI. */
export type KnownBridgeErrorCode =
  | "BAD_REQUEST"
  | "UNKNOWN_METHOD"
  | "NO_EDIT_HANDLE"
  /** `backend.request`/`backend.uploadFile` transport failures only — a
   * reachable backend that answers with an HTTP error status is a normal
   * (non-throwing) result. */
  | "BACKEND_UNREACHABLE"
  | "BACKEND_TIMEOUT"
  | "DOWNLOAD_FAILED"
  | "FILE_NOT_FOUND"
  | "INSERT_FAILED"
  /** `timeline.captureFrame` (contract v3): the render/encode pipeline
   * failed after `NO_EDIT_HANDLE` was already ruled out. */
  | "CAPTURE_FAILED"
  /** `ui.pickFile` (contract v3): the user dismissed the file dialog without
   * choosing a file. Never surfaced as an error in the UI. */
  | "CANCELLED"
  /** `ui.pickFile` (contract v3): the native file dialog itself failed to open. */
  | "DIALOG_FAILED"
  /** `ui.makeThumbnail` (contract v3): decode/resize/encode failed. */
  | "THUMBNAIL_FAILED"
  /** `timeline.cutoutRange` (contract v5): the range render/mux pipeline
   * failed after `NO_EDIT_HANDLE` was already ruled out. */
  | "CUTOUT_FAILED"
  /** `timeline.extractAudio` (contract v5): audio extraction/encode failed,
   * or the range contains no audio to extract. */
  | "EXTRACT_FAILED"
  /** `timeline.insertProvisional`/`resolveProvisional`/`updateProvisionalText`
   * (contract v5): the placeholder insert/replace/update operation failed. */
  | "PROVISIONAL_FAILED";

/** Local-only error code used when a request never receives a response. */
export type LocalBridgeErrorCode = "TIMEOUT" | "DISPOSED";

export type BridgeErrorCode = KnownBridgeErrorCode | LocalBridgeErrorCode | string;

export interface BridgeRequest<M extends BridgeMethod = BridgeMethod> {
  id: number;
  method: M;
  params: ParamsOf<M>;
}

export interface BridgeSuccessResponse<M extends BridgeMethod = BridgeMethod> {
  id: number;
  ok: true;
  result: ResultOf<M>;
}

export interface BridgeErrorPayload {
  code: BridgeErrorCode;
  message: string;
}

export interface BridgeErrorResponse {
  id: number;
  ok: false;
  error: BridgeErrorPayload;
}

export type BridgeResponse<M extends BridgeMethod = BridgeMethod> =
  | BridgeSuccessResponse<M>
  | BridgeErrorResponse;

/** Native-initiated, unsolicited message. Not used by any M1 feature, but the
 * dispatch plumbing accepts and routes these so future milestones can add
 * events (e.g. job progress) without a transport change. */
export interface BridgeEvent<T = unknown> {
  event: string;
  data: T;
}

/** Narrow an unknown incoming message to a BridgeResponse. */
export function isBridgeResponse(message: unknown): message is BridgeResponse {
  if (typeof message !== "object" || message === null) return false;
  const candidate = message as Record<string, unknown>;
  return typeof candidate.id === "number" && typeof candidate.ok === "boolean";
}

/** Narrow an unknown incoming message to a BridgeEvent. */
export function isBridgeEvent(message: unknown): message is BridgeEvent {
  if (typeof message !== "object" || message === null) return false;
  const candidate = message as Record<string, unknown>;
  return typeof candidate.event === "string" && !("id" in candidate);
}

/** Thrown by NativeBridge implementations for both native-reported errors
 * (code is one of KnownBridgeErrorCode) and local failures (timeout/dispose). */
export class BridgeError extends Error {
  readonly code: BridgeErrorCode;

  constructor(code: BridgeErrorCode, message: string) {
    super(message);
    this.name = "BridgeError";
    this.code = code;
  }
}

/** Default time to wait for a response before rejecting with a TIMEOUT BridgeError. */
export const DEFAULT_TIMEOUT_MS = 10_000;

/**
 * Methods `RequestDispatcher` must not apply any local wait ceiling to
 * (neither `DEFAULT_TIMEOUT_MS` nor a per-call `timeoutMs`, unlike the
 * `backend.request` override described above): `ui.pickFile`/`ui.pickFolder`
 * open a native file dialog that blocks synchronously until the user closes
 * it, so there is no "too long" — a still-open dialog is expected, correct
 * behavior, not a hang. `dispose()` still rejects these with `DISPOSED` if
 * the bridge itself is torn down while one is pending.
 */
export const NO_LOCAL_TIMEOUT_METHODS: ReadonlySet<BridgeMethod> = new Set([
  "ui.pickFile",
  "ui.pickFolder",
]);

/** Abstraction implemented by both the production WebView2 bridge and the
 * dev-time mock bridge. UI code depends only on this interface. */
export interface NativeBridge {
  request<M extends BridgeMethod>(
    method: M,
    params: ParamsOf<M>,
  ): Promise<ResultOf<M>>;

  /** Contract v7: like `request`, but attaches `files` (raw drag-and-drop
   * `File` objects) as WebView2 `AdditionalObjects` — the production bridge
   * sends them via `postMessageWithAdditionalObjects`; the mock bridge
   * synthesizes `__droppedPaths` from each file's `.name`. Used exclusively
   * by `shell/useFileDrop.ts`'s `ui.resolveDroppedFiles` call; every other
   * caller keeps using plain `request`. */
  requestWithFiles<M extends BridgeMethod>(
    method: M,
    params: ParamsOf<M>,
    files: unknown[],
  ): Promise<ResultOf<M>>;

  /** Subscribe to a native-initiated event. Returns an unsubscribe function. */
  on(event: string, handler: (data: unknown) => void): () => void;
}
