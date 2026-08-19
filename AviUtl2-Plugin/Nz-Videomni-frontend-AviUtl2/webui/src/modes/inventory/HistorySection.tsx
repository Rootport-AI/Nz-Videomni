import type { JobResponse } from "../../api/types";
import { useStrings } from "../../i18n/LanguageContext";
import { JobCard } from "../../jobs/JobCard";
import { useJobsContext } from "../../jobs/JobsContext";

export interface HistorySectionProps {
  baseUrl: string | null;
}

const PROMPT_EXCERPT_LENGTH = 80;

/** `JobResponse.request` is a `Record<string, unknown>` echo of whatever was
 * submitted (Docs/API_REFERENCE.md §6) — `prompt` is always a string in
 * practice, but this is read defensively since nothing enforces that at the
 * type level. `noPromptLabel` is threaded in by the caller (rather than this
 * pure function importing `i18n` itself) since it comes from `useStrings()`. */
function promptExcerpt(job: JobResponse, noPromptLabel: string): string {
  const raw = job.request.prompt;
  const text = typeof raw === "string" ? raw.trim() : "";
  if (text.length === 0) return noPromptLabel;
  return text.length > PROMPT_EXCERPT_LENGTH ? `${text.slice(0, PROMPT_EXCERPT_LENGTH)}…` : text;
}

/** The M5 "history" half of the Library screen: every completed job so far,
 * as a grid of cards — reusing the exact same job rail data source
 * (`JobsContext`) and `JobCard` (preview-to-expand, Insert, Delete) the M3
 * job rail already has, just laid out as a grid with a prompt/resolution/
 * duration header per card instead of the rail's single-column list. */
export function HistorySection({ baseUrl }: HistorySectionProps) {
  const strings = useStrings();
  const { jobs, cancellingIds, deletingIds, cancelJob, deleteJob } = useJobsContext();
  const completed = jobs
    .filter((job) => job.status === "completed")
    .sort((a, b) => b.created_at.localeCompare(a.created_at));

  return (
    <section className="inventory-section">
      <div className="inventory-section-head">
        <h2>{strings.inventory.historyHeading}</h2>
      </div>
      <p className="hint">{strings.inventory.historyNote}</p>

      {completed.length === 0 ? (
        <p className="hint">{strings.inventory.historyEmpty}</p>
      ) : (
        <ul className="history-grid">
          {completed.map((job) => (
            <li key={job.job_id} className="history-card">
              <p className="history-card-prompt" title={promptExcerpt(job, strings.inventory.noPromptPlaceholder)}>
                {promptExcerpt(job, strings.inventory.noPromptPlaceholder)}
              </p>
              {job.result && (
                <p className="hint history-card-meta">
                  {job.result.resolution} · {job.result.duration_seconds.toFixed(1)}s
                </p>
              )}
              <JobCard
                job={job}
                baseUrl={baseUrl}
                isCancelling={cancellingIds.has(job.job_id)}
                isDeleting={deletingIds.has(job.job_id)}
                onCancel={() => void cancelJob(job.job_id)}
                onDelete={() => void deleteJob(job.job_id)}
              />
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
