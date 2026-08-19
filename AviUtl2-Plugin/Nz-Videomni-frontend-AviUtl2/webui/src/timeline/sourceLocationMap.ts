/**
 * Volatile map from a job id to the timeline position of the SOURCE video a V2V
 * continuation (#1 `extendVideo`) was generated from (JOIN_FEATURE_RESEARCH.md
 * §4.5). The Join flow's 🎞 insert of the *joined* clip needs to drop it at
 * "the source's tail − the trim length actually cut", on the frontmost layer —
 * which requires knowing where the original source object sat on the timeline.
 *
 * The right-click that starts a v2v continuation records the selected source
 * object's location here keyed by the reservation's `pending-…` id
 * (`recordSourceLocation`); the Generate-time hand-off re-keys that entry to the
 * real backend job id (`rekeySourceLocation`), so the completed job's joined
 * insert can look it up (`getSourceLocation`). The same `rekeySourceLocation` is
 * used when the single reservation seat MOVES (`reservePlacement`'s 用途b mints a
 * new `pending-…` id), so the memo follows the seat instead of being stranded
 * under the retired id.
 *
 * Deliberately in-memory only (one `Map`, module-level, like
 * `provisionalReservation`'s reservation seat): a project reload wipes it, and
 * that is the DESIGN — a job whose source location is unknown (reloaded, or a
 * plain Chain-screen manual v2v that never went through the right-click) simply
 * degrades to a position-less joined insert (the pre-existing behavior). No
 * persistence is attempted; the map is a best-effort convenience, never a
 * correctness dependency.
 */

/** Where the v2v SOURCE object sat on the timeline when the continuation was
 * reserved, plus the project rate/scale captured at that moment (for the
 * frame<->seconds conversion the joined-insert position needs). */
export interface SourceLocation {
  layer: number;
  /** First covered frame of the source object (inclusive). */
  frameStart: number;
  /** Last covered frame of the source object (INCLUSIVE — matches native's
   * `timeline.getSelection` `frameEnd`, see `bridge_core.cpp`
   * `ResolveProvisionalPlacement`). */
  frameEnd: number;
  filePath: string | null;
  rate: number;
  scale: number;
}

/** The single volatile store. Keyed first by the reservation's `pending-…` id,
 * then re-keyed to the real job id at Generate-time hand-off. */
let sourceLocations = new Map<string, SourceLocation>();

/** Record the source object's location under a reservation's `pending-…` id.
 * Overwrites any existing entry for the same key. Note a moved reservation does
 * NOT reuse its key: `provisionalReservation`'s `reservePlacement` mints a fresh
 * `newPendingId()` on every move and carries the memo across with
 * `rekeySourceLocation`, so the entry ends up under the NEW pending id (a
 * re-record then simply overwrites that carried-over entry). */
export function recordSourceLocation(pendingId: string, loc: SourceLocation): void {
  sourceLocations.set(pendingId, loc);
}

/** Re-key the entry stored under `pendingId` to `jobId` (Generate-time hand-off,
 * mirroring `bindToJob`). No-op when no entry exists for `pendingId` (e.g. a
 * chain flow that never recorded a source location, like #6 image-to-clip-chain
 * or a plain manual v2v). */
export function rekeySourceLocation(pendingId: string, jobId: string): void {
  const loc = sourceLocations.get(pendingId);
  if (loc === undefined) return;
  sourceLocations.delete(pendingId);
  sourceLocations.set(jobId, loc);
}

/** The source location recorded for `jobId`, or `null` when none is known
 * (reloaded, manual v2v, or already released) — the caller then degrades to a
 * position-less joined insert. */
export function getSourceLocation(jobId: string): SourceLocation | null {
  return sourceLocations.get(jobId) ?? null;
}

/** Drop the entry for `jobId` once the joined insert has consumed it. No-op when
 * absent. */
export function releaseSourceLocation(jobId: string): void {
  sourceLocations.delete(jobId);
}

/** Test-only: clear the whole map so each test starts from a clean module. */
export function resetSourceLocationMap(): void {
  sourceLocations = new Map<string, SourceLocation>();
}
