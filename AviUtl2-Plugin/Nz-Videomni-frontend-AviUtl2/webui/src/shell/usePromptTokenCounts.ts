import { useEffect, useState } from "react";
import { apiClient } from "../api/client";
import type { ApiClient } from "../api/client";
import type { PromptTokensRequest, PromptTokensResponse } from "../api/types";

/** Quiet time after the last keystroke before the count is requested. The
 * server counts in a few ms (G0 measurement), so this is a feel value. */
export const PROMPT_TOKEN_DEBOUNCE_MS = 250;

/**
 * §1-111: the token counts of the prompt fields (`POST /utils/prompt-tokens`).
 *
 * - `request === null` = disabled: nothing is sent (the Chained tab is
 *   always-mounted, so its clip counting runs only while it is shown).
 * - Re-requests when the request's content or `modelKey` (active base model +
 *   pipeline-loading flag, so a text-encoder swap is picked up by the load
 *   that follows it) changes, after {@link PROMPT_TOKEN_DEBOUNCE_MS}.
 * - Latest wins: a response whose request was superseded is dropped, so an
 *   out-of-order reply never overwrites a newer one.
 * - Keeps the previous value while debouncing / in flight (no flicker); a
 *   failed request clears it to `null` (the counter then shows nothing).
 */
export function usePromptTokenCounts(
  request: PromptTokensRequest | null,
  modelKey: string,
  client: Pick<ApiClient, "countPromptTokens"> = apiClient,
): PromptTokensResponse | null {
  const [counts, setCounts] = useState<PromptTokensResponse | null>(null);
  const key = request === null ? null : JSON.stringify(request);

  useEffect(() => {
    if (key === null) return undefined;
    let superseded = false;
    const timer = setTimeout(() => {
      client.countPromptTokens(JSON.parse(key) as PromptTokensRequest).then(
        (res) => {
          if (!superseded) setCounts(res);
        },
        () => {
          if (!superseded) setCounts(null);
        },
      );
    }, PROMPT_TOKEN_DEBOUNCE_MS);
    return () => {
      superseded = true;
      clearTimeout(timer);
    };
  }, [key, modelKey, client]);

  return counts;
}
