import type { AppConfig } from "../api/types";

/** Fixed placeholder for a redacted secret. Deliberately not derived from
 * the original value's length (a length-preserving mask like `"*".repeat(n)`
 * would still leak the key's length), and language-independent so it needs
 * no i18n string. */
const REDACTED = "***";

/** N10 raw-config viewer: returns a deep copy of `config` with known secret
 * fields replaced by {@link REDACTED}, safe to `JSON.stringify` and render
 * on screen. Never mutates `config` — callers elsewhere in the app keep
 * reading the original object.
 *
 * `AppConfig` (api/types.ts) now types `server?.api_key` too (N13's
 * API-key-status badge needs it), but this redactor still reads it off the
 * post-`JSON.parse` runtime `copy` rather than trusting the type — `copy` is
 * the exact object about to be rendered on screen, and reading it
 * defensively there is the correctness-relevant guard, independent of
 * whatever the type says. Add further secret fields here the same way, one
 * explicit `if` per field — this stays a short, named list rather than a
 * generic recursive redactor, matching the rest of this app's "simplicity
 * over blanket coverage" style. */
export function redactConfigForDisplay(config: AppConfig): unknown {
  // `config` is plain data straight out of a `GET /config` JSON response
  // (no functions/class instances/circular refs), so a JSON round-trip is a
  // safe, simple deep clone that leaves the original untouched.
  const copy: any = JSON.parse(JSON.stringify(config));

  const apiKey = copy?.server?.api_key;
  if (typeof apiKey === "string" && apiKey.length > 0) {
    copy.server.api_key = REDACTED;
  }

  return copy;
}
