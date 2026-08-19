import type { AppConfig } from "../api/types";

/** N13: Settings panel's API-key status badge (set/unset/unknown). Reuses
 * the same "present" test as `redactConfig.ts`'s display-time redaction
 * (`typeof api_key === "string" && length > 0`) so the two stay in lockstep
 * — the badge should never claim "set" for a value the raw-config viewer
 * would treat as absent, or vice versa. */
export type ApiKeyStatus = "set" | "unset" | "unknown";

/** Derives the badge state from a fetched `AppConfig` plus `useConfig()`'s
 * `usingFallback` flag. `usingFallback` wins over everything else: the
 * fallback config (`defaultConfig.ts`) carries no `server` section at all,
 * so treating that as "unset" would misreport a server we never actually
 * reached as having no API key configured. */
export function apiKeyStatus(config: AppConfig, usingFallback: boolean): ApiKeyStatus {
  if (usingFallback) return "unknown";

  const apiKey = config.server?.api_key;
  if (typeof apiKey === "string" && apiKey.length > 0) return "set";

  return "unset";
}
