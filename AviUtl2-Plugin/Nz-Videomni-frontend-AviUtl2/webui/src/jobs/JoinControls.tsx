import { useCallback, useEffect, useId, useRef, useState } from "react";
import { apiClient as defaultApiClient, BackendApiError } from "../api/client";
import type { ApiClient } from "../api/client";
import { bridge as defaultBridge } from "../bridge";
import type { NativeBridge } from "../bridge";
import type { JobResponse, JoinResponse } from "../api/types";
import { useStrings } from "../i18n/LanguageContext";
import { isV2VJob } from "./joinUtils";
import { editInfoFps, formatFps, framesToSeconds, jobFrameRate } from "./fpsConvert";

export type JoinState =
  | { phase: "idle" }
  | { phase: "joining" }
  | { phase: "joined" }
  | { phase: "error"; code: string; message: string };

/** N6: crossfade length choices offered by the join dropdown. The server
 * default is 300ms (`JoinRequestBody.handle_crossfade_ms`,
 * Docs/API_REFERENCE.md §3.17) — kept as this component's default too so an
 * untouched dropdown reproduces the previous body-less `joinJob` call. */
const CROSSFADE_CHOICES_MS = [150, 300, 500] as const;
const DEFAULT_CROSSFADE_MS = 300;

/** I5: source tail-keep trim lengths, expressed in frames and converted to
 * `source_tail_seconds` at the generated video's fps on submit. 120f is the
 * default (≈5s at 24fps, matching the server's own 5.0s default). */
const TRIM_FRAME_CHOICES = [120, 240] as const;
const DEFAULT_TRIM_FRAMES = 120;

/** The trim info the joined-clip insert needs (JOIN_FEATURE_RESEARCH.md §4.5):
 * only ever known for a join performed THIS session — a joined view merely
 * restored from `job.joined` at mount reports `null`, so its insert degrades to
 * position-less (the pre-existing behavior). */
export interface JoinedInfo {
  trimmedSourceSeconds: number;
}

export interface JoinControlsProps {
  job: JobResponse;
  baseUrl: string | null;
  apiClient?: ApiClient;
  nativeBridge?: NativeBridge;
  /** I6 Join: lifted so `JobCard` owns the single preview area and the 🎞
   * insert branch (JOIN_FEATURE_RESEARCH.md §4.5). Fired whenever the
   * "showing joined?" state changes: `true` + the just-joined trim info on a
   * successful join, `true` + `null` when the joined view was restored from
   * `job.joined` at mount (trim unknown), `false` + `null` on Unjoin / a
   * true->false poll collapse / a `unjoinToken` bump. */
  onJoinedStateChange?: (showingJoined: boolean, joinedInfo: JoinedInfo | null) => void;
  /** I6 Join: `JobCard` bumps this to force the joined view back to idle — the
   * reverse channel for the joined-preview error fallback, which lives on the
   * (now `JobCard`-owned) `<video>`. Ignored on its initial value; each change
   * collapses a `joined` view to `idle` (reappearing the [Join] button). */
  unjoinToken?: number;
}

function describeError(err: unknown): { code: string; message: string } {
  if (err instanceof BackendApiError) return { code: err.code, message: err.message };
  if (err instanceof Error) return { code: "UNKNOWN", message: err.message };
  return { code: "UNKNOWN", message: String(err) };
}

/**
 * M6's "Join with source" flow for a completed V2V chain job
 * (Docs/API_REFERENCE.md §3.17/§3.18): `POST /jobs/{id}/join` -> on success the
 * joined view is entered, and the actual `<video>` preview + 🎞 insert of the
 * joined clip live in `JobCard` (I6: single preview area, 🎞 inserts the
 * currently-displayed clip). This component is now control-only — the trim-length
 * radio, audio crossfade, Join/Unjoin buttons, the fps-mismatch note and the
 * error state — and reports the joined display state up via `onJoinedStateChange`.
 *
 * Renders nothing for a job that isn't both `completed` and V2V
 * (`isV2VJob`) — safe to mount unconditionally from both `JobCard` (the
 * always-on job rail) and Chain mode's own result panel.
 *
 * I5 raised the join UI to its settled shape:
 *  - "joined" is seeded from the server-authoritative `JobResponse.joined`
 *    (the `joined.mp4` exists) at mount only, then followed one-way (below).
 *  - a trim-length radio (120/240 frames) sets `source_tail_seconds`, and
 *  - an [Unjoin] button reverts the local view without touching the server.
 */
export function JoinControls({
  job,
  apiClient = defaultApiClient,
  nativeBridge = defaultBridge,
  onJoinedStateChange,
  unjoinToken,
}: JoinControlsProps) {
  const strings = useStrings();
  // Per-instance radio group name: `useId` is unique per mounted component, so
  // the trim radios of two `JoinControls` for the SAME job (three panels are
  // always mounted) never merge into one document-wide group — which would
  // leave the visible instance rendering as unselected.
  const radioGroupId = useId();
  // Mount-time only: restore the joined view from the server's file-existence
  // flag. Polling updates are handled one-way by the effect below, never by
  // re-reading `job.joined` into this initial state on later renders.
  const [joinState, setJoinState] = useState<JoinState>(() =>
    job.joined === true ? { phase: "joined" } : { phase: "idle" },
  );
  const [crossfadeMs, setCrossfadeMs] = useState<number>(DEFAULT_CROSSFADE_MS);
  const [trimFrames, setTrimFrames] = useState<number>(DEFAULT_TRIM_FRAMES);
  // The `JoinResponse` from the join we performed this session (null when the
  // joined view was merely restored from `job.joined` at mount) — drives the
  // fps-mismatch note AND the joined-insert trim length (both only make sense
  // for a join we just ran).
  const [joinResponse, setJoinResponse] = useState<JoinResponse | null>(null);
  // The project's generation fps (getEditInfo rate/scale), or null when it
  // couldn't be read — in which case the fps note stays hidden (graceful
  // fallback, no numbers to show and no reliable comparison to make).
  const [projectFps, setProjectFps] = useState<number | null>(null);

  const videoFps = jobFrameRate(job);

  // One-way polling follow: only a true->false transition of `job.joined` (e.g.
  // the joined.mp4 was deleted out from under us in Explorer) collapses the
  // joined view back to idle. A `job.joined === true` arriving from polling
  // MUST NOT re-assert the joined phase — that would stomp a local [Unjoin] the
  // user just performed, rewinding it on every poll.
  useEffect(() => {
    if (job.joined === false) {
      setJoinState((prev) => (prev.phase === "joined" ? { phase: "idle" } : prev));
    }
  }, [job.joined]);

  // I6 Join: force the joined view back to idle when `JobCard` bumps
  // `unjoinToken` (the joined-preview error fallback). Ignore the initial value
  // (seeded into the ref) so a first render never spuriously unjoins.
  const unjoinTokenSeenRef = useRef(unjoinToken);
  useEffect(() => {
    if (unjoinToken === unjoinTokenSeenRef.current) return;
    unjoinTokenSeenRef.current = unjoinToken;
    setJoinState((prev) => (prev.phase === "joined" ? { phase: "idle" } : prev));
  }, [unjoinToken]);

  // Read the project fps once for the fps-mismatch note. `.catch()` is
  // mandatory: getEditInfo rejects (NO_EDIT_HANDLE) when no project is open,
  // which must not surface as an error here — the note simply stays hidden.
  useEffect(() => {
    if (job.status !== "completed" || !isV2VJob(job)) return;
    let cancelled = false;
    void nativeBridge
      .request("getEditInfo", {})
      .then((info) => {
        if (!cancelled) setProjectFps(editInfoFps(info.rate, info.scale));
      })
      .catch(() => {
        /* no project / probe failed -> leave projectFps null, note hidden */
      });
    return () => {
      cancelled = true;
    };
  }, [nativeBridge, job]);

  // I6 Join: publish the "showing joined?" state (and, for a join run this
  // session, its trim info) up to `JobCard`, which owns the preview src swap and
  // the 🎞 insert branch. Runs on every relevant change; `JobCard` wraps its
  // handler in `useCallback` so this never loops.
  const showingJoined = joinState.phase === "joined";
  useEffect(() => {
    const trimmed = joinResponse?.trimmed_source_seconds;
    onJoinedStateChange?.(
      showingJoined,
      showingJoined && trimmed !== undefined ? { trimmedSourceSeconds: trimmed } : null,
    );
  }, [showingJoined, joinResponse, onJoinedStateChange]);

  const handleJoin = useCallback(() => {
    setJoinState({ phase: "joining" });
    void (async () => {
      try {
        const response = await apiClient.joinJob(job.job_id, {
          handle_crossfade_ms: crossfadeMs,
          source_tail_seconds: framesToSeconds(trimFrames, videoFps),
        });
        setJoinResponse(response);
        setJoinState({ phase: "joined" });
      } catch (err) {
        const { code, message } = describeError(err);
        setJoinState({ phase: "error", code, message });
      }
    })();
  }, [apiClient, job.job_id, crossfadeMs, trimFrames, videoFps]);

  const handleUnjoin = useCallback(() => {
    // Local view toggle only — no server call, the joined.mp4 stays on disk.
    // Back to idle the [Join] button reappears; re-joining just re-runs
    // handleJoin (the server overwrites the existing joined.mp4).
    setJoinState({ phase: "idle" });
  }, []);

  if (job.status !== "completed" || !isV2VJob(job)) return null;

  const trimSecondsLabel = (frames: number): string =>
    strings.jobs.trimLengthOption(String(frames), framesToSeconds(frames, videoFps).toFixed(1));

  // The note shown after a join performed this session (never for a joined view
  // merely restored from `job.joined`, where `joinResponse` is null). The body
  // reports what actually happened to the source — its own fps was converted, or
  // it was re-encoded to match the generated format — and an optional advice
  // line flags a project/generated fps mismatch. The body shows even when the
  // project fps is unknown; advice needs a known project fps to compare.
  const fpsNote = (() => {
    if (joinState.phase !== "joined" || joinResponse == null) return null;
    const srcFps = joinResponse.source_fps;
    const sourceFpsDiff = srcFps != null && Math.abs(srcFps - videoFps) > 0.001;
    const body =
      sourceFpsDiff && srcFps != null
        ? strings.jobs.joinSourceFpsNote(formatFps(srcFps), formatFps(videoFps))
        : joinResponse.source_normalized === true
          ? strings.jobs.joinNormalizedNote
          : null;
    const advice =
      projectFps != null && Math.abs(projectFps - videoFps) > 0.001
        ? strings.jobs.joinProjectFpsAdvice(formatFps(projectFps), formatFps(videoFps))
        : null;
    if (body == null && advice == null) return null;
    return [body, advice].filter(Boolean).join(" ");
  })();

  return (
    <div className="join-controls">
      {joinState.phase === "idle" && (
        <>
          <label className="field">
            <span className="field-label">{strings.jobs.crossfadeLabel}</span>
            <select
              className="field-select"
              value={crossfadeMs}
              onChange={(e) => setCrossfadeMs(Number(e.target.value))}
            >
              {CROSSFADE_CHOICES_MS.map((ms) => (
                <option key={ms} value={ms}>
                  {strings.jobs.crossfadeOption(ms)}
                </option>
              ))}
            </select>
          </label>
          <div className="field">
            <span className="field-label">{strings.jobs.trimLengthLabel}</span>
            <div className="join-trim-row">
              {TRIM_FRAME_CHOICES.map((frames) => (
                <label key={frames} className="field field-inline">
                  <input
                    type="radio"
                    name={`join-trim-${radioGroupId}`}
                    value={frames}
                    checked={trimFrames === frames}
                    onChange={() => setTrimFrames(frames)}
                  />
                  <span>{trimSecondsLabel(frames)}</span>
                </label>
              ))}
            </div>
          </div>
          <button type="button" className="secondary-button" onClick={handleJoin}>
            {strings.jobs.join}
          </button>
        </>
      )}
      {joinState.phase === "joining" && (
        <button type="button" className="secondary-button" disabled>
          {strings.jobs.joining}
        </button>
      )}
      {joinState.phase === "error" && (
        <>
          <p className="hint field-hint-error">
            {joinState.code}: {joinState.message}
          </p>
          <button type="button" className="secondary-button" onClick={handleJoin}>
            {strings.jobs.joinRetry}
          </button>
        </>
      )}
      {joinState.phase === "joined" && (
        <div className="join-result">
          {fpsNote != null && (
            <p className="warning-banner warning-banner-mild">{fpsNote}</p>
          )}
          <button type="button" className="secondary-button" onClick={handleUnjoin}>
            {strings.jobs.unjoin}
          </button>
        </div>
      )}
    </div>
  );
}
