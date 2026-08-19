/**
 * W7: the "why can't I press Generate?" note shown directly under the Generate
 * button on both Create and Chain. Given the current list of failing-gate
 * reason codes plus a code→message map, it renders one imperative line per
 * DISTINCT message (de-duped by the resolved string, not by code) as a mild
 * warning banner. Renders nothing when there are no reasons.
 *
 * Pure display: it owns no state and never reads the form directly — the caller
 * (`SingleScreen`/`ChainedScreen`) composes the reason list (hook `validityReasons`
 * + any screen-level codes) and supplies the localized `messages` map. Kept in
 * `modes/single/` and imported by Chain too, mirroring how `CommonGenerationFields`
 * / `GenerateButtonBar` are already shared across the two screens.
 *
 * De-dup rationale (adversarial review R2): two different codes can resolve to
 * the SAME line — e.g. Create's `controlNeedsReference` and `referenceNotReady`
 * both say "Attach a reference video." and can hold simultaneously. Collapsing
 * on the rendered string keeps the list from repeating an instruction. A code
 * with no entry in `messages` is skipped rather than shown blank.
 */
export function GenerateReasonsNote({
  reasons,
  messages,
  className,
}: {
  reasons: readonly string[];
  /** Code → localized imperative line. Missing codes are skipped. */
  messages: Readonly<Record<string, string>>;
  /** Extra class(es) appended to the banner (e.g. a test hook / spacing). */
  className?: string | undefined;
}) {
  const lines: string[] = [];
  const seen = new Set<string>();
  for (const code of reasons) {
    const message = messages[code];
    if (message === undefined || seen.has(message)) continue;
    seen.add(message);
    lines.push(message);
  }
  if (lines.length === 0) return null;

  return (
    <div
      className={["warning-banner", "warning-banner-mild", "generate-reasons-note", className].filter(Boolean).join(" ")}
      role="note"
    >
      <ul className="generate-reasons-list">
        {lines.map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>
    </div>
  );
}
