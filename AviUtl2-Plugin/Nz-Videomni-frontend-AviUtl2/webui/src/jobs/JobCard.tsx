import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { apiClient as defaultApiClient } from "../api/client";
import type { ApiClient } from "../api/client";
import { bridge as defaultBridge } from "../bridge";
import type { NativeBridge } from "../bridge";
import type { JobResponse, JobStatus } from "../api/types";
import { useStrings } from "../i18n/LanguageContext";
import type { Strings } from "../i18n/LanguageContext";
import { downloadAndInsertVideo } from "./downloadAndInsert";
import { JoinControls } from "./JoinControls";
import type { JoinedInfo } from "./JoinControls";
import { editInfoFps, joinedInsertFrame } from "./fpsConvert";
import { getSourceLocation, releaseSourceLocation } from "../timeline/sourceLocationMap";
import {
  clearJobInserted,
  getInsertedFilePath,
  isJobInserted,
  subscribeInserted,
} from "../timeline/provisionalReservation";
import { jobSeed } from "./seedUtils";

export interface JobCardProps {
  job: JobResponse;
  baseUrl: string | null;
  isCancelling: boolean;
  isDeleting: boolean;
  onCancel: () => void;
  onDelete: () => void;
  nativeBridge?: NativeBridge | undefined;
  apiClient?: ApiClient;
  /** U-R1 MJ-4: auto-expand the completed-clip preview once this job reaches
   * `completed` (the ledger sets it on the just-submitted, highlighted job, so
   * the finished result opens itself — the old `GenerationPanel`'s "show the
   * result big" moment). Unset elsewhere (e.g. `HistorySection`), preserving
   * the play-to-expand default. */
  autoOpenPreview?: boolean;
}

type LocalInsertState = "idle" | "downloading" | "inserting" | "done" | "error";

function statusLabels(strings: Strings): Record<JobStatus, string> {
  return {
    queued: strings.jobs.status.queued,
    running: strings.jobs.status.running,
    completed: strings.jobs.status.completed,
    failed: strings.jobs.status.failed,
    cancelled: strings.jobs.status.cancelled,
  };
}

/**
 * Resolve the timeline slot for a *joined* V2V insert (JOIN_FEATURE_RESEARCH.md
 * §4.5): the source object's tail minus the kept trim length, on the frontmost
 * layer (`getEditInfo.layerMax + 1`). Returns `undefined` — degrading to the
 * position-less joined insert (native's cursor placement) — whenever anything is
 * missing: no trim info (a mount-restored joined view, not a join run this
 * session), no recorded source location (reloaded / manual v2v), an unresolvable
 * project fps, or a failed `getEditInfo` (can't read `layerMax`).
 */
async function resolveJoinedInsertPosition(
  nativeBridge: NativeBridge,
  jobId: string,
  joinedInfo: JoinedInfo | null,
): Promise<{ layer: number; frame: number } | undefined> {
  if (joinedInfo === null) return undefined;
  const loc = getSourceLocation(jobId);
  if (loc === null) return undefined;
  const projectFps = editInfoFps(loc.rate, loc.scale);
  if (projectFps === null) return undefined;
  const frame = joinedInsertFrame({
    frameStart: loc.frameStart,
    frameEnd: loc.frameEnd,
    projectFps,
    trimmedSourceSeconds: joinedInfo.trimmedSourceSeconds,
  });
  try {
    const info = await nativeBridge.request("getEditInfo", {});
    return { layer: info.layerMax + 1, frame };
  } catch {
    // Can't read the frontmost layer -> give up on the position and let the
    // joined insert fall back to native's cursor placement.
    return undefined;
  }
}

/** One job's card in the always-on job rail (task brief §2). Status badge,
 * progress+stage while running, a lightweight play-to-expand preview once
 * completed, and Insert/Cancel/Delete actions scoped to the states they're
 * valid in. */
export function JobCard({
  job,
  baseUrl,
  isCancelling,
  isDeleting,
  onCancel,
  onDelete,
  nativeBridge = defaultBridge,
  apiClient = defaultApiClient,
  autoOpenPreview = false,
}: JobCardProps) {
  const strings = useStrings();
  const STATUS_LABELS = statusLabels(strings);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [insertState, setInsertState] = useState<LocalInsertState>("idle");
  const [insertError, setInsertError] = useState<string | null>(null);

  // I6 Join: the single preview area shows either the job's own clip or its
  // *joined* V2V result. `JoinControls` owns the join lifecycle and pushes the
  // "showing joined?" state (plus, for a join run this session, its trim info)
  // up here via `onJoinedStateChange`; this card is the sole owner of the
  // preview `<video>` and the 🎞 insert branch (JOIN_FEATURE_RESEARCH.md §4.5).
  const [showingJoined, setShowingJoined] = useState(false);
  const [joinedInfo, setJoinedInfo] = useState<JoinedInfo | null>(null);
  // Bumped to ask `JoinControls` to leave the joined view (the reverse channel
  // for the joined-preview error fallback below).
  const [unjoinToken, setUnjoinToken] = useState(0);
  const handleJoinedStateChange = useCallback((showing: boolean, info: JoinedInfo | null) => {
    setShowingJoined(showing);
    setJoinedInfo(info);
  }, []);

  // MJ-4 (fixed): force the preview open only the moment an auto-open job is
  // *observed transitioning into* `completed` — never on mount, so reopening
  // this card (e.g. re-render after a ledger refresh) for a job that was
  // already completed doesn't yank the preview back open out from under a
  // user who closed it. `prevStatusRef` seeds from the mount-time status, so
  // a job that's already `completed` at mount never satisfies the
  // "prevStatus !== completed" transition check below.
  const prevStatusRef = useRef(job.status);
  useEffect(() => {
    const prevStatus = prevStatusRef.current;
    prevStatusRef.current = job.status;
    if (autoOpenPreview && job.status === "completed" && prevStatus !== "completed") {
      setPreviewOpen(true);
    }
  }, [autoOpenPreview, job.status]);

  // The local path the first successful insert downloaded to. On a re-insert
  // (after ✅/⚠️ reset the button back to 🎞️) we hand this back to
  // `downloadAndInsertVideo` so it skips the download — re-downloading over a
  // file AviUtl2 may still hold open from the first insert fails with a
  // shared-write violation ("Could not open destination file"). Deliberately
  // NOT cleared on reset: clearing it would resurrect that re-download and
  // reintroduce the bug. Used ONLY on the plain (non-joined) path — the joined
  // output changes with the crossfade setting and must always be refetched, so
  // the joined branch never reads or writes this.
  const rememberedPathRef = useRef<string | null>(null);

  // I6 Join: the joined-preview `<video>` (moved here from `JoinControls`) and
  // its one-shot reload budget, keyed by URL so it resets when the joined source
  // changes. `preload="auto"` prefetches the moment the join succeeds, so a
  // transient hiccup would otherwise tear the whole joined view down — reload
  // the same source once before falling back.
  const joinedVideoRef = useRef<HTMLVideoElement | null>(null);
  const retriedJoinedUrlRef = useRef<string | null>(null);
  const joinedVideoUrl = baseUrl ? `${baseUrl}/api/v1/jobs/${job.job_id}/joined` : null;

  // I6 Join: when a join is entered (showingJoined false->true, this session or a
  // mount restore), open the preview so the swapped-in joined clip is visible
  // (the old `JoinControls` always rendered the joined video on join) and reset
  // the one-shot reload budget for the fresh source.
  const prevShowingJoinedRef = useRef(showingJoined);
  useEffect(() => {
    const prev = prevShowingJoinedRef.current;
    prevShowingJoinedRef.current = showingJoined;
    if (showingJoined && !prev) {
      setPreviewOpen(true);
      retriedJoinedUrlRef.current = null;
    }
  }, [showingJoined]);

  // Y3 (W2 shared ✅, 非対称案): W2 (AppShell `insertProvisionalResult`) runs the
  // SAME replace-insert this card's manual 🎞 does, but its ✅ lives in AppShell,
  // out of reach of this card's LOCAL `insertState`. Subscribe to the shared
  // inserted-store so a W2 insert of THIS job flips the card to ✅ too. The
  // manual 🎞 never publishes to that store (it stays instance-local), so the
  // asymmetry is intentional: only W2 shares. `prevSharedRef` mirrors the
  // `prevShowingJoinedRef` guard above — act ONLY on the one-shot false->true
  // transition, never on every render.
  const sharedInserted = useSyncExternalStore(subscribeInserted, () => isJobInserted(job.job_id));
  const prevSharedRef = useRef(sharedInserted);
  useEffect(() => {
    const prev = prevSharedRef.current;
    prevSharedRef.current = sharedInserted;
    if (sharedInserted && !prev) {
      // Rare same-job race: a manual insert of THIS job is already mid-flight
      // (⏳ downloading/inserting). Don't stomp its in-progress state — let its
      // own settle drive the ✅; the W2 mark is redundant here.
      if (insertState === "downloading" || insertState === "inserting") return;
      // Reuse the file W2 already downloaded so a ✅→🎞 re-insert never
      // re-downloads over a destination AviUtl2 may still hold open.
      const fp = getInsertedFilePath(job.job_id);
      if (fp !== null) rememberedPathRef.current = fp;
      setInsertState("done");
    }
  }, [sharedInserted, insertState, job.job_id]);

  const handleJoinedVideoError = useCallback(() => {
    const video = joinedVideoRef.current;
    if (video && retriedJoinedUrlRef.current !== joinedVideoUrl) {
      // First error for this joined URL: reload once before giving up.
      retriedJoinedUrlRef.current = joinedVideoUrl;
      video.load();
      return;
    }
    // Retry spent (or no element): the joined mp4 really isn't reachable — ask
    // `JoinControls` to leave the joined view (its [Join] button reappears), and
    // the preview falls back to the plain clip.
    setUnjoinToken((t) => t + 1);
  }, [joinedVideoUrl]);

  const handleInsert = useCallback(() => {
    setInsertError(null);
    setInsertState("downloading");
    void (async () => {
      try {
        if (showingJoined) {
          // I6 Join: insert the *joined* clip at the source tail − trim length,
          // frontmost layer (position-less when it can't be resolved), then
          // remove the continuation's provisional marker and free the recorded
          // source location now that the joined clip has landed.
          const position = await resolveJoinedInsertPosition(nativeBridge, job.job_id, joinedInfo);
          await downloadAndInsertVideo(
            nativeBridge,
            job.job_id,
            (phase) => setInsertState(phase),
            position ? { joined: true, position } : { joined: true },
          );
          void nativeBridge
            .request("timeline.deleteProvisionalByJob", { jobId: job.job_id })
            .catch(() => {
              // No marker (reloaded / manual v2v) or a benign native hiccup —
              // the joined clip is already placed, so this cleanup is best-effort.
            });
          releaseSourceLocation(job.job_id);
          setInsertState("done");
        } else {
          const result = await downloadAndInsertVideo(
            nativeBridge,
            job.job_id,
            (phase) => setInsertState(phase),
            rememberedPathRef.current !== null ? { knownFilePath: rememberedPathRef.current } : {},
          );
          rememberedPathRef.current = result.filePath;
          setInsertState("done");
        }
      } catch (err) {
        setInsertState("error");
        setInsertError(err instanceof Error ? err.message : String(err));
      }
    })();
  }, [job.job_id, nativeBridge, showingJoined, joinedInfo]);

  const isNonTerminal = job.status === "queued" || job.status === "running";
  const isTerminal = !isNonTerminal;
  const progressPct = Math.round((job.progress ?? 0) * 100);
  const videoSrc = baseUrl && job.result ? `${baseUrl}${job.result.video_url}` : null;
  // I6 Join: the single preview swaps to the joined mp4 while the joined view is
  // active; otherwise it shows the job's own clip (unchanged behavior).
  const activeVideoSrc = showingJoined ? joinedVideoUrl : videoSrc;

  // Insert button appearance per state (③ Insert✅-cycle + error). The
  // accessible name comes solely from `aria-label` (the emoji sits in an
  // aria-hidden span) — `idle`/`error` keep the exact `strings.jobs.insert` name
  // and `done` keeps `strings.jobs.inserted` so App/App.chain/InventoryScreen
  // tests that fetch this button by /^insert$/i and /^inserted$/i stay green.
  // The click handler (below) is state-branched instead: neither `done` (✅)
  // nor `error` (⚠️) inserts on click — both just reset the button back to idle
  // (🎞️) without touching the native bridge, so the emoji cycles 🎞️ → ⏳ →
  // ✅/⚠️ → 🎞️ and the *next* 🎞️ click is what re-inserts. (⚠️'s `title` still
  // surfaces the failure message on hover; `aria-label` stays "Insert".) The
  // reset does NOT clear `rememberedPathRef`, so that re-insert reuses the
  // already-downloaded file instead of re-downloading over a possibly-open one.
  const insertLabel =
    insertState === "downloading"
      ? strings.jobs.downloading
      : insertState === "inserting"
        ? strings.jobs.inserting
        : insertState === "done"
          ? strings.jobs.inserted
          : strings.jobs.insert;
  const insertEmoji =
    insertState === "downloading" || insertState === "inserting"
      ? "⏳"
      : insertState === "done"
        ? "✅"
        : insertState === "error"
          ? "⚠️"
          : "🎞️";
  const insertTitle = insertState === "error" ? (insertError ?? insertLabel) : insertLabel;
  const insertDisabled = insertState === "downloading" || insertState === "inserting";
  const onInsertClick =
    insertState === "done" || insertState === "error"
      ? () => {
          setInsertState("idle");
          setInsertError(null);
          // Y3: clear any shared W2 flag for this job so the store's `done`
          // snapshot doesn't immediately re-flip this card back to ✅ on the
          // next render (and a later W2 re-insert can transition false->true
          // again). No-op for a manual/error reset (nothing was ever shared).
          clearJobInserted(job.job_id);
        }
      : handleInsert;
  const seed = jobSeed(job);

  return (
    <div className={`job-card job-card--${job.status}`} data-job-id={job.job_id}>
      <div className="job-card-head">
        <span className={`badge badge-${job.status}`}>{STATUS_LABELS[job.status]}</span>
        {job.clip !== null && job.clip_count !== null && (
          <span className="badge badge-clip">{strings.jobs.clipBadge(job.clip, job.clip_count)}</span>
        )}
        <span className="job-card-id">{job.job_id.slice(0, 8)}</span>
        <span className="job-card-seed">{seed !== null ? `#${seed}` : strings.jobs.seedRandom}</span>
        <div className="job-card-head-actions">
          {job.status === "completed" && (
            <button
              type="button"
              className="icon-action-button"
              title={insertTitle}
              aria-label={insertLabel}
              disabled={insertDisabled}
              onClick={onInsertClick}
            >
              <span aria-hidden="true">{insertEmoji}</span>
            </button>
          )}
          {isNonTerminal && (
            <button
              type="button"
              className="icon-action-button"
              title={isCancelling ? strings.jobs.cancelling : strings.jobs.cancel}
              aria-label={isCancelling ? strings.jobs.cancelling : strings.jobs.cancel}
              disabled={isCancelling}
              onClick={onCancel}
            >
              <span aria-hidden="true">🚫</span>
            </button>
          )}
          {isTerminal && (
            <button
              type="button"
              className="icon-action-button"
              title={isDeleting ? strings.jobs.deleting : strings.jobs.delete}
              aria-label={isDeleting ? strings.jobs.deleting : strings.jobs.delete}
              disabled={isDeleting}
              onClick={onDelete}
            >
              <span aria-hidden="true">❌</span>
            </button>
          )}
        </div>
      </div>

      {job.status === "running" && (
        <>
          <div
            className="progress-bar"
            role="progressbar"
            aria-valuenow={progressPct}
            aria-valuemin={0}
            aria-valuemax={100}
          >
            <div className="progress-bar-fill" style={{ width: `${progressPct}%` }} />
          </div>
          <p className="hint">
            {job.stage ? `${job.stage} — ` : ""}
            {progressPct}%
          </p>
        </>
      )}

      {job.status === "completed" && (
        <div className="job-card-preview">
          <button
            type="button"
            className="job-card-play"
            onClick={() => setPreviewOpen((open) => !open)}
            aria-label={previewOpen ? strings.jobs.closePreview : strings.jobs.preview}
            aria-expanded={previewOpen}
          >
            {previewOpen ? "▾" : "▶"}
          </button>
          {previewOpen &&
            (activeVideoSrc ? (
              // eslint-disable-next-line jsx-a11y/media-has-caption
              // preload="auto" (not "metadata"): Chromium leaves the initial
              // frame undrawn / playback stalled with "metadata", so we prefetch
              // fully to get a painted thumbnail on tab return. The joined src
              // gets a one-shot reload on error before falling back (I6).
              <video
                ref={joinedVideoRef}
                src={activeVideoSrc}
                controls
                preload="auto"
                className="job-card-video"
                onError={showingJoined ? handleJoinedVideoError : undefined}
              />
            ) : (
              <p className="hint">{strings.common.loadingPreview}</p>
            ))}
        </div>
      )}

      {job.status === "failed" && <p className="hint field-hint-error">{job.error ?? strings.jobs.genericFailure}</p>}

      <JoinControls
        job={job}
        baseUrl={baseUrl}
        apiClient={apiClient}
        nativeBridge={nativeBridge}
        onJoinedStateChange={handleJoinedStateChange}
        unjoinToken={unjoinToken}
      />

      {insertState === "error" && insertError && <p className="hint field-hint-error">{insertError}</p>}
    </div>
  );
}
