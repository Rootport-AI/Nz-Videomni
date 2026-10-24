export interface TokenCounterProps {
  /** `null`/`undefined` (not counted, or counting failed) renders nothing. */
  count: number | null | undefined;
  /** `null`/`undefined` = the engine has no limit: the count alone, never red. */
  limit: number | null | undefined;
}

/** Whether `count` is past `limit` (the text-encoder cuts the excess
 * silently). Also drives the textarea's `input-warning`. */
export function isOverTokenLimit(count: number | null | undefined, limit: number | null | undefined): boolean {
  return count != null && limit != null && count > limit;
}

/** §1-111: `n / limit` (or `n`) beside a prompt field; red past the limit.
 * Informational only — it never blocks generation. */
export function TokenCounter({ count, limit }: TokenCounterProps) {
  if (count == null) return null;
  const over = isOverTokenLimit(count, limit);
  return (
    <span className={`token-counter${over ? " field-hint-error" : ""}`}>
      {limit == null ? count : `${count} / ${limit}`}
    </span>
  );
}
