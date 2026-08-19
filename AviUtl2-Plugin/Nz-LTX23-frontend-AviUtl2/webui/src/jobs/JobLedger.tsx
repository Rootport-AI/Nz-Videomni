import { useEffect, useRef } from "react";
import type { NativeBridge } from "../bridge";
import { useStrings } from "../i18n/LanguageContext";
import { JobCard } from "./JobCard";
import { useJobsContext } from "./JobsContext";
import "./jobs.css";

export interface JobLedgerProps {
  baseUrl: string | null;
  /** The most recently submitted job (from `AppShell` via Create/Chain): its
   * card is outlined and scrolled into view, and — once it completes — has its
   * preview auto-expanded (U-R1 MJ-4, the replacement for the old
   * `GenerationPanel`'s "show the finished result big" moment). */
  highlightedJobId: string | null;
  /** Passed through from Create/Chain (which hold the shell's injected bridge,
   * `undefined` in production) straight to each `JobCard` — U-R1 MN-2: the
   * app-wide singleton fallback lives in exactly one place (`JobCard`'s default
   * param), not repeated here the way the old `JobRail` did. */
  nativeBridge?: NativeBridge | undefined;
}

/** The always-on job ledger (U-R1): every job the backend knows about, newest
 * first, as `JobCard`s (Cancel/Delete/Insert/Join/progress all built in). It
 * replaces the old right-hand `JobRail` sidebar and the per-Create
 * `GenerationPanel` at once — the single generation panel now stacks the
 * Generate button, a submit-error row, and this ledger, so there is nothing to
 * the right of the panel (the owner's core requirement). Always expanded — the
 * rail's narrow-width collapse mechanism is gone; `jobs.css` instead caps the
 * body height so the list scrolls in place. */
export function JobLedger({ baseUrl, highlightedJobId, nativeBridge }: JobLedgerProps) {
  const strings = useStrings();
  const { jobs, cancellingIds, deletingIds, cancelJob, deleteJob } = useJobsContext();
  const highlightRef = useRef<HTMLLIElement | null>(null);

  useEffect(() => {
    if (highlightedJobId && highlightRef.current) {
      highlightRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [highlightedJobId]);

  const sorted = [...jobs].sort((a, b) => b.created_at.localeCompare(a.created_at));

  return (
    <section className="job-ledger">
      <h2 className="job-ledger-heading">
        {strings.jobs.heading} ({jobs.length})
      </h2>
      <div className="job-ledger-body">
        {sorted.length === 0 ? (
          <p className="hint job-ledger-empty">{strings.jobs.empty}</p>
        ) : (
          <ul className="job-list">
            {sorted.map((job) => {
              const isHighlighted = job.job_id === highlightedJobId;
              return (
                <li
                  key={job.job_id}
                  ref={isHighlighted ? highlightRef : undefined}
                  className={isHighlighted ? "job-list-item job-list-item--highlighted" : "job-list-item"}
                >
                  <JobCard
                    job={job}
                    baseUrl={baseUrl}
                    isCancelling={cancellingIds.has(job.job_id)}
                    isDeleting={deletingIds.has(job.job_id)}
                    onCancel={() => void cancelJob(job.job_id)}
                    onDelete={() => void deleteJob(job.job_id)}
                    nativeBridge={nativeBridge}
                    autoOpenPreview={isHighlighted}
                  />
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </section>
  );
}
