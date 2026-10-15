import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AlphaGenRequestBody, AppConfig } from "../../api/types";
import { bridge as defaultBridge, BridgeError } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import { ACCELERATION_DEFAULTS, accelerationRequestFields } from "../../shell/accelerationSettings";
import type { AccelerationSettings } from "../../shell/accelerationSettings";
import { comfortFramesForBudget, resolveComfortRow } from "../../shell/comfortTable";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import { snapRetakeWindowFrames } from "../../timeline/retakeWindow";
import { formatTrimSeconds } from "../../timeline/sourceTrim";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import { fileNameFromPath } from "../single/keyframeUtils";
import { ceilToMultiple, snapNumFrames } from "../single/paramUtils";
// クロスモード import。`useOutpaintForm` / `useRetakeForm` と同じ理由で、
// `modes/chained/` のフックをそのまま使う。
import { useSourceUpload } from "../chained/useSourceUpload";

/** The frame-count range `POST /generate/alpha` accepts (`AlphaGenRequest.num_frames`). */
export const ALPHA_GEN_MIN_FRAMES = 9;
export const ALPHA_GEN_MAX_FRAMES = 145;
/** The slider's starting value (設計書 §9). */
export const ALPHA_GEN_DEFAULT_FRAMES = 81;

/** Where the size notes start (設計書 §9): above the official maximum the matte
 * gets less accurate, and so does light mode below its floor. */
const OFFICIAL_MAX_WIDTH = 1920;
const OFFICIAL_MAX_HEIGHT = 1088;
const LIGHT_MODE_MIN_WIDTH = 1280;
const LIGHT_MODE_MIN_HEIGHT = 768;

/** The grid the server pads the canvas onto (`chain_math.ceil64`). */
const CANVAS_GRID = 64;

/** How the source arrived. Only a right-click carries a 再生位置; a drop and 📁
 * start at the head of the file (設計書 §9.2). */
export type AlphaGenSourceOrigin = "rightClick" | "drop" | "pick";

/** Where a right-clicked source object sits on the timeline, with the
 * project's rate/scale at that moment (`timeline.getSelection`'s
 * `selected[0]` and top-level `rate` / `scale`). Alpha Gen 第 3 弾: recorded
 * against the job at submit (`EditScreen` → `recordSourceLocation`) so the
 * card's 🎞 can drop the output at the object's start frame
 * (Docs/ALPHAGEN_DESIGN.md §10.3). */
export interface AlphaGenSourceLocation {
  layer: number;
  frameStart: number;
  /** Inclusive, like `getSelection`'s `frameEnd`. */
  frameEnd: number;
  rate: number;
  scale: number;
}

export interface AlphaGenSource {
  filePath: string;
  fileName: string;
  /** Seconds on the source's own time axis where the matte starts. */
  startSec: number;
  origin: AlphaGenSourceOrigin;
  /** Only for `origin: "rightClick"`; a drop / 📁 has no timeline position. */
  location?: AlphaGenSourceLocation;
}

/** `fs.probeMediaInfo`'s answer for the source (contract v15). Every field is
 * `0` when unknown — `fps` included (always `0` for Matroska mkv/webm/mk3d,
 * by native's rule). */
export interface AlphaGenMediaInfo {
  durationSec: number;
  width: number;
  height: number;
  fps: number;
}

const UNKNOWN_MEDIA_INFO: AlphaGenMediaInfo = { durationSec: 0, width: 0, height: 0, fps: 0 };

/** Why Generate cannot be pressed (`editReasonMessages.buildAlphaGenReasonMessages`). */
export type AlphaGenReasonCode = "sourceMissing" | "sourceTooShort" | "uploading";

export interface UseAlphaGenFormDeps {
  /** Test/integration seam (probe, pick, upload). Production omits it. */
  nativeBridge?: NativeBridge | undefined;
  /** The one-shot right-click payload. Read once, on mount, and only when its
   * intent is `"alpha-gen"` — `EditScreen` keeps every sub-panel mounted, so
   * another sub-tab's right-click reaches this hook too. */
  initialIntent?: GenerationPrefill | undefined;
  /** `GET /config` — only its comfort table is read. */
  config?: AppConfig | undefined;
  /** The four inputs that pick the comfort row, exactly as Retake takes them. */
  engineFamily?: string | undefined;
  acceleration?: AccelerationSettings | undefined;
  sageAvailable?: boolean | null | undefined;
  weightClass?: string | undefined;
  /** Called once when {@link UseAlphaGenFormResult.prepareRequest} had to cut
   * the frame count down to what the source actually has (the caller shows a
   * toast). */
  onAdjusted?: (() => void) | undefined;
}

export interface UseAlphaGenFormResult {
  source: AlphaGenSource | null;
  mediaInfo: AlphaGenMediaInfo;
  /** Drag-and-drop entry (`useFileDrop`'s `onFile`). Starts at the head. */
  attachDropped: (filePath: string, fileName: string) => void;
  /** 📁 entry: `ui.pickFile({kind: "video"})`. Starts at the head. */
  pick: () => Promise<void>;
  clear: () => void;

  /** 8n+1 in [9, 145] once committed; free typing passes through until then. */
  numFrames: number;
  setNumFrames: (value: number, snap: boolean) => void;
  /** `numFrames / fps`, or `null` while the fps is unknown. */
  seconds: number | null;
  lightMode: boolean;
  setLightMode: (value: boolean) => void;

  /** The comfort line on the slider, or `null` for no line (no row, no budget,
   * size unknown, or the line would fall below the slider's 9-frame minimum). */
  comfortFrames: number | null;
  /** The frame count is past the comfort line (always `false` with no line). */
  isOverComfort: boolean;
  /** The source is larger than 1920*1088 (a note, never a block). */
  aboveOfficialMax: boolean;
  /** Light mode on a source smaller than 1280*768 (a note; the checkbox stays). */
  belowLightFloor: boolean;

  /** The window upload is in flight (it starts at the Generate click). */
  uploading: boolean;
  /** The last upload failed. Cleared by a new source or the next attempt. */
  uploadError: boolean;
  validityReasons: AlphaGenReasonCode[];
  isValid: boolean;
  /**
   * Uploads the window and builds the request body; `null` when the upload
   * failed (see {@link uploadError}) or the source turned out too short.
   * Handed to `useGenerationSubmit.submitAlpha` at Generate time.
   */
  prepareRequest: () => Promise<AlphaGenRequestBody | null>;
}

/**
 * Owns the Alpha Gen panel's form (Alpha Gen 第 2 弾, 設計書 §9): the source
 * (by right-click, drop or 📁), its measured size / length / fps, the frame
 * count, light mode, and the request built at Generate time.
 *
 * ## The upload happens at Generate, and only the window goes up
 *
 * The three entries only RECORD the source. `prepareRequest` uploads exactly
 * the window that will be used (`trim_start_sec` / `trim_duration_sec` +
 * `max_frames`), so `uploads/` never holds a whole long file for a short
 * matte, and nothing is converted (the server cuts at the source's own fps).
 * When the window's time cannot be known (fps or length unknown — an mkv,
 * say) the file goes up whole with no query at all and the server cuts from
 * `window_start_sec`: a `max_frames` alone would cut from the HEAD and lose
 * the playback position.
 */
export function useAlphaGenForm(deps: UseAlphaGenFormDeps = {}): UseAlphaGenFormResult {
  const { nativeBridge, initialIntent, config, engineFamily, acceleration, sageAvailable, weightClass, onAdjusted } =
    deps;
  const bridge = nativeBridge ?? defaultBridge;

  const [source, setSourceState] = useState<AlphaGenSource | null>(null);
  const [mediaInfo, setMediaInfo] = useState<AlphaGenMediaInfo>(UNKNOWN_MEDIA_INFO);
  const [numFrames, setNumFramesState] = useState(ALPHA_GEN_DEFAULT_FRAMES);
  const [lightMode, setLightMode] = useState(false);
  const [uploadError, setUploadError] = useState(false);
  // The server measured fewer than 9 usable frames (known only after the
  // upload). Sits beside the estimate from the probe, which is all there is
  // before that.
  const [measuredTooShort, setMeasuredTooShort] = useState(false);

  const upload = useSourceUpload("video", nativeBridge ? { nativeBridge } : {});
  const { uploadPath, clear: clearUpload } = upload;

  const setSource = useCallback(
    (next: AlphaGenSource | null) => {
      setSourceState(next);
      setUploadError(false);
      setMeasuredTooShort(false);
      clearUpload();
    },
    [clearUpload],
  );

  const attachDropped = useCallback(
    (filePath: string, fileName: string) => setSource({ filePath, fileName, startSec: 0, origin: "drop" }),
    [setSource],
  );

  const pick = useCallback(async () => {
    try {
      const picked = await bridge.request("ui.pickFile", { kind: "video" });
      setSource({ filePath: picked.filePath, fileName: picked.fileName, startSec: 0, origin: "pick" });
    } catch (err) {
      // A dismissed dialog (or any other picker failure) leaves the current
      // source as it is — there is nothing new to show.
      if (!(err instanceof BridgeError)) throw err;
    }
  }, [bridge, setSource]);

  const clear = useCallback(() => setSource(null), [setSource]);

  // --- one-shot right-click consume (the `useOutpaintForm` arrangement) -----
  const autoLoadRef = useRef(false);
  useEffect(() => {
    if (autoLoadRef.current) return;
    autoLoadRef.current = true;
    if (initialIntent?.intent !== "alpha-gen") return;
    const item = initialIntent.selection.selected[0];
    const filePath = item?.filePath;
    if (!item || !filePath) return;
    // The 再生位置 is on the source's own time axis in seconds (`sourceTrim.ts`),
    // so it is the window start as it stands. No reported range = the head.
    const startSec = item.hasPlaybackRange === true ? Math.max(0, item.playbackStartSec ?? 0) : 0;
    const { rate, scale } = initialIntent.selection;
    setSource({
      filePath,
      fileName: fileNameFromPath(filePath),
      startSec,
      origin: "rightClick",
      location: { layer: item.layer, frameStart: item.frameStart, frameEnd: item.frameEnd, rate, scale },
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps -- one-shot mount consume
  }, []);

  // --- media probe (contract v15: size, length AND fps from one call) -------
  const sourcePath = source?.filePath ?? null;
  useEffect(() => {
    setMediaInfo(UNKNOWN_MEDIA_INFO);
    if (sourcePath === null) return;
    let cancelled = false;
    bridge
      .request("fs.probeMediaInfo", { filePath: sourcePath })
      .then((info) => {
        if (cancelled) return;
        setMediaInfo({ durationSec: info.durationSec, width: info.width, height: info.height, fps: info.fps ?? 0 });
      })
      .catch(() => {
        // It never rejects for a bad file (only for an empty path): unknown it is.
      });
    return () => {
      cancelled = true;
    };
  }, [bridge, sourcePath]);

  // --- frame count ------------------------------------------------------------
  const setNumFrames = useCallback((value: number, snap: boolean) => {
    // Free typing passes through (Create's field does the same) and the slider /
    // commit snaps, so "121" can be typed without the "1" snapping to 9 first.
    setNumFramesState(snap ? snapNumFrames(value, ALPHA_GEN_MIN_FRAMES, ALPHA_GEN_MAX_FRAMES) : value);
  }, []);

  const fps = mediaInfo.fps;
  const seconds = fps > 0 ? numFrames / fps : null;

  // The frames the source has from the start onwards — known only when both
  // the fps and the length are. `null` = unknown: generation is still allowed
  // (the server measures, and answers 422 if it is too short).
  const startSec = source?.startSec ?? 0;
  const usableFrames =
    fps > 0 && mediaInfo.durationSec > 0 ? Math.floor((mediaInfo.durationSec - startSec) * fps + 1e-6) : null;

  // --- comfort line -----------------------------------------------------------
  const limits = (config ?? FALLBACK_APP_CONFIG).limits;
  const comfortRow = useMemo(
    () =>
      resolveComfortRow(limits, engineFamily, acceleration ?? ACCELERATION_DEFAULTS, sageAvailable ?? null, weightClass),
    [limits, engineFamily, acceleration, sageAvailable, weightClass],
  );
  const known = mediaInfo.width > 0 && mediaInfo.height > 0;
  const budget = comfortRow === null ? null : lightMode ? comfortRow.singleBudget : comfortRow.alphaGenBudget;
  // The server measures the CANVAS (the source padded onto the 64 grid). The
  // lower clamp is 1, not 9, so that a line below 9 frames stays visible here
  // and becomes "no line": a line outside the slider's range (9-145) is not
  // drawn.
  const rawComfortFrames =
    known && comfortRow !== null && budget !== null
      ? comfortFramesForBudget(
          ceilToMultiple(mediaInfo.width, CANVAS_GRID, 0, Number.POSITIVE_INFINITY),
          ceilToMultiple(mediaInfo.height, CANVAS_GRID, 0, Number.POSITIVE_INFINITY),
          budget,
          1,
          ALPHA_GEN_MAX_FRAMES,
          comfortRow.spatialFactor,
          comfortRow.temporalFactor,
        )
      : null;
  const comfortFrames = rawComfortFrames !== null && rawComfortFrames >= ALPHA_GEN_MIN_FRAMES ? rawComfortFrames : null;
  // Past the line: the warning under the slider (Create's `DurationField` banner).
  const isOverComfort = comfortFrames !== null && numFrames > comfortFrames;

  const aboveOfficialMax = known && (mediaInfo.width > OFFICIAL_MAX_WIDTH || mediaInfo.height > OFFICIAL_MAX_HEIGHT);
  const belowLightFloor =
    lightMode && known && (mediaInfo.width < LIGHT_MODE_MIN_WIDTH || mediaInfo.height < LIGHT_MODE_MIN_HEIGHT);

  // --- gates ------------------------------------------------------------------
  const uploading = upload.state.status === "uploading";
  const validityReasons: AlphaGenReasonCode[] = [];
  if (source === null) {
    validityReasons.push("sourceMissing");
  } else if ((usableFrames !== null && usableFrames < ALPHA_GEN_MIN_FRAMES) || measuredTooShort) {
    validityReasons.push("sourceTooShort");
  }
  if (uploading) validityReasons.push("uploading");

  // --- the request, built at Generate time ------------------------------------
  // Read through a ref so the callback the submit hook runs never closes over
  // an older render's form.
  const latestRef = useRef({ source, mediaInfo, numFrames, lightMode, acceleration, onAdjusted });
  latestRef.current = { source, mediaInfo, numFrames, lightMode, acceleration, onAdjusted };

  const prepareRequest = useCallback(async (): Promise<AlphaGenRequestBody | null> => {
    const current = latestRef.current;
    const src = current.source;
    if (src === null) return null;
    setUploadError(false);

    // 1. What the source has, and the frame count we can ask for.
    const S = src.startSec;
    const F = current.mediaInfo.fps;
    const D = current.mediaInfo.durationSec;
    const requested = snapNumFrames(current.numFrames, ALPHA_GEN_MIN_FRAMES, ALPHA_GEN_MAX_FRAMES);
    const windowKnown = F > 0 && D > 0;
    const available = windowKnown ? Math.floor((D - S) * F + 1e-6) : null;
    let N = available === null ? requested : snapRetakeWindowFrames(Math.min(requested, available));

    // 2./3. Only the window goes up when its time is known (one frame of slack;
    // `max_frames` both caps and MEASURES it). From the head (S = 0) the cap
    // alone does the job — the server ignores a duration without a start
    // (`_trim_window`), so none is sent. Otherwise the whole file goes up with
    // no query, and the server cuts from S.
    const query = windowKnown
      ? S > 0
        ? {
            trim_start_sec: formatTrimSeconds(S),
            trim_duration_sec: formatTrimSeconds((N + 1) / F),
            max_frames: String(N + 1),
          }
        : { max_frames: String(N + 1) }
      : undefined;
    const outcome = await uploadPath(src.filePath, src.fileName, query);
    if (outcome === null) {
      setUploadError(true);
      return null;
    }
    // A trim that did not happen leaves the whole file: cut from S on it. From
    // the head, cut or not, the upload starts at the window (S = 0 either way).
    const windowStart = query !== undefined && outcome.trimmed ? 0 : S;

    // 4. The server's own count wins over the estimate.
    if (outcome.frame_count !== null && F > 0) {
      N = Math.min(N, snapRetakeWindowFrames(outcome.frame_count - Math.round(windowStart * F)));
    }
    if (N < ALPHA_GEN_MIN_FRAMES) {
      setMeasuredTooShort(true);
      return null;
    }

    // 5. Tell the user, once, when the count went down.
    if (N < requested) {
      setNumFramesState(N);
      current.onAdjusted?.();
    }

    // 6. The body. Only the five pass-through acceleration keys the endpoint
    // takes (`AlphaGenRequest`) — `vae_mode` / `embed_mp4_metadata` are not
    // among them — and no `frame_rate`: the server measures the source's own.
    const fields = accelerationRequestFields(current.acceleration);
    return {
      reference_video_id: outcome.video_id,
      window_start_sec: windowStart,
      num_frames: N,
      seed: -1,
      light_mode: current.lightMode,
      ...(fields.attention_backend !== undefined ? { attention_backend: fields.attention_backend } : {}),
      ...(fields.block_swap_prefetch !== undefined ? { block_swap_prefetch: fields.block_swap_prefetch } : {}),
      ...(fields.keep_resident !== undefined ? { keep_resident: fields.keep_resident } : {}),
      ...(fields.keep_resident_embeddings !== undefined
        ? { keep_resident_embeddings: fields.keep_resident_embeddings }
        : {}),
      ...(fields.fused_gguf_dequant_kernel !== undefined
        ? { fused_gguf_dequant_kernel: fields.fused_gguf_dequant_kernel }
        : {}),
    };
  }, [uploadPath]);

  return {
    source,
    mediaInfo,
    attachDropped,
    pick,
    clear,
    numFrames,
    setNumFrames,
    seconds,
    lightMode,
    setLightMode,
    comfortFrames,
    isOverComfort,
    aboveOfficialMax,
    belowLightFloor,
    uploading,
    uploadError,
    validityReasons,
    isValid: validityReasons.length === 0,
    prepareRequest,
  };
}
