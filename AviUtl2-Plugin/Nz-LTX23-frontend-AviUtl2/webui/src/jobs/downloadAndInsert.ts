import type { NativeBridge } from "../bridge";
import { releaseIfSettled } from "../timeline/provisionalReservation";

export type DownloadAndInsertPhase = "downloading" | "inserting";

export interface DownloadAndInsertOptions {
  /** M6: when true, downloads the *joined* V2V result
   * (`GET /jobs/{id}/joined` via `backend.downloadVideo`'s `joined` param)
   * instead of the plain per-clip output. Only meaningful after a
   * successful `POST /jobs/{id}/join` (see `jobs/JoinControls.tsx`). */
  joined?: boolean;
  /** When set, skip `backend.downloadVideo` entirely and insert this
   * already-downloaded local path straight into the timeline. Used by
   * `JobCard`'s re-insert (after a first successful insert, AviUtl2 may still
   * hold the file open — re-downloading over it fails with a shared-write
   * violation), which remembers the `filePath` returned by the first insert
   * and hands it back here. Mutually exclusive with `joined` in practice
   * (only the plain per-clip path is ever re-inserted this way). */
  knownFilePath?: string;
  /** V2V joined-insert only: the resolved timeline slot the joined clip should
   * land at — the source object's tail minus the kept trim length, on the
   * frontmost layer (JOIN_FEATURE_RESEARCH.md §4.5). Passed straight to
   * `timeline.insertMedia`'s `layer`/`frame`. Omitted — the source location is
   * unknown (reloaded / manual v2v) or the edit info couldn't be read — falls
   * back to the position-less joined insert (native's cursor placement). Only
   * meaningful together with `joined`. */
  position?: { layer: number; frame: number };
  /** W3 (⬇ "insert the latest generation result here"): insert the downloaded
   * clip as a PLAIN positioned `timeline.insertMedia` at this layer/frame,
   * bypassing the per-clip replace-RPC (`insertForJob`) entirely. Because it
   * never goes through the replace path, a provisional reservation object at the
   * cursor is left in place — this is an ADDITIVE insert (the latest result
   * dropped at the right-click spot), not the fulfillment of a specific job's
   * reservation. Mutually exclusive with `joined`/`knownFilePath` in practice
   * (the W3 layer command always downloads a plain completed clip). */
  plainInsertAt?: { layer: number; frame: number };
}

/**
 * The two-step "hand a completed job's mp4 to the AviUtl2 timeline" flow
 * (`backend.downloadVideo` -> `timeline.insertMedia`, Docs/API_REFERENCE.md
 * §3.16 / SDK_REFERENCE.md §7). Shared by both the Create screen's own
 * result panel (`modes/single/useGeneration.ts`) and the job rail's
 * per-card "Insert" button (`jobs/JobCard.tsx`), so the bridge call
 * sequence lives in exactly one place.
 */
export async function downloadAndInsertVideo(
  nativeBridge: NativeBridge,
  jobId: string,
  onPhase?: (phase: DownloadAndInsertPhase) => void,
  options: DownloadAndInsertOptions = {},
): Promise<{ layer: number; frame: number; filePath: string }> {
  let filePath: string;
  if (options.knownFilePath !== undefined) {
    // Re-insert of an already-downloaded clip: skip the download so we never
    // reopen a destination AviUtl2 may still hold from the first insert.
    filePath = options.knownFilePath;
  } else {
    onPhase?.("downloading");
    // The plain per-clip path is immutable, so let native reuse an existing
    // download instead of re-writing over a possibly-open file. The `joined`
    // output changes with the crossfade setting, so it must always refetch —
    // never pass `reuseIfPresent` there.
    const download = await nativeBridge.request(
      "backend.downloadVideo",
      options.joined ? { jobId, joined: true } : { jobId, reuseIfPresent: true },
    );
    filePath = download.filePath;
  }

  onPhase?.("inserting");

  // W3 (⬇ layer quick-insert): a PLAIN positioned insert that never touches any
  // provisional marker. It does NOT go through `insertForJob`'s replace-RPC, so
  // a reservation object sitting at the cursor is left intact — the layer-menu
  // ⬇ is an additive "drop the latest result here", not a job fulfillment.
  if (options.plainInsertAt) {
    const result = await nativeBridge.request("timeline.insertMedia", {
      filePath,
      layer: options.plainInsertAt.layer,
      frame: options.plainInsertAt.frame,
    });
    return { layer: result.layer, frame: result.frame, filePath };
  }

  // The V2V *joined* insert must NEVER replace a job's provisional marker
  // (adversarial-review Med 6): the joined clip is a distinct, longer artifact
  // that does not belong in any single per-clip reservation's slot. Route it
  // through the plain `insertMedia` (append at the cursor), so the replace-insert
  // can never swap a joined result into a provisional placeholder.
  if (options.joined) {
    const result = await nativeBridge.request(
      "timeline.insertMedia",
      options.position
        ? { filePath, layer: options.position.layer, frame: options.position.frame }
        : { filePath },
    );
    return { layer: result.layer, frame: result.frame, filePath };
  }

  const { layer, frame } = await insertForJob(nativeBridge, jobId, filePath);
  return { layer, frame, filePath };
}

/**
 * The per-clip / reservation-backed insert step, split out so it is directly
 * testable. Replace-insert: one atomic RPC that either replaces this job's
 * provisional marker in place (`mode:"replaced"`) or, when no marker remains,
 * inserts exactly like the plain `insertMedia` (`mode:"inserted"`). This
 * subsumes the old "insertMedia + deleteProvisionalByJob" two-call sequence.
 */
export async function insertForJob(
  nativeBridge: NativeBridge,
  jobId: string,
  filePath: string,
): Promise<{ layer: number; frame: number }> {
  const result = await nativeBridge.request("timeline.insertMediaForJob", { jobId, filePath });
  // On any success (replaced OR inserted), sync-release the reservation seat
  // for this job if it is still tracked (adversarial-review Med 7). After a
  // project reload, `reconcileFromTimeline` restores the seat to `generating`
  // from the surviving marker; the 🎞 insert then removes that marker, but
  // nothing else would free the seat, leaving a false ⏳生成中 busy-guard.
  // `releaseIfSettled` is a no-op unless jobId is the currently-tracked job, so
  // the normal (already-settled) path is unaffected. This only touches the
  // in-memory seat state — it issues no bridge call and edits nothing.
  releaseIfSettled(jobId);
  return { layer: result.layer, frame: result.frame };
}
