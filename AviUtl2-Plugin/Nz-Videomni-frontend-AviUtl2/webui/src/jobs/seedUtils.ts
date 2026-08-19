import type { JobResponse } from "../api/types";

/**
 * The seed a job actually ran with (⑤). Prefers the server-reported
 * `result.seed_used` (authoritative — a `-1` "random" request is resolved to a
 * concrete seed there once the job completes). Falls back to the echoed
 * `request.seed` when there's no result yet, but only when it's a real,
 * non-negative seed: `request` is a loose `Record<string, unknown>`, so the
 * value is type-guarded, and a `-1` (or any negative "random" sentinel) is
 * treated as "not yet known" → `null`. Returns `null` when neither is present.
 */
export function jobSeed(job: JobResponse): number | null {
  const used = job.result?.seed_used;
  if (typeof used === "number") return used;

  const requested = job.request.seed;
  if (typeof requested === "number" && requested >= 0) return requested;

  return null;
}

/**
 * The seed of the most recent job — the one the ♻ "reuse last seed" button
 * offers. Uses the same `created_at`-descending order as `JobLedger.tsx`'s
 * card list (newest first) and applies {@link jobSeed} to the head. Returns
 * `null` for an empty list or when the newest job has no resolvable seed.
 */
export function latestJobSeed(jobs: JobResponse[]): number | null {
  const sorted = [...jobs].sort((a, b) => b.created_at.localeCompare(a.created_at));
  const newest = sorted[0];
  return newest ? jobSeed(newest) : null;
}
