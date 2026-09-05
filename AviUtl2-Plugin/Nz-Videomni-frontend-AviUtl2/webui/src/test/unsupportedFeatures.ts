import type { MockBridge } from "../bridge/mockBridge";
import type { NativeBridge } from "../bridge/types";

/**
 * A mock bridge whose `GET /api/v1/models` reports EXTRA `unsupported_features`
 * for one base model.
 *
 * Why this exists. The shell's tab-level greying (`shell/featureScope.ts`'s
 * `disabledModesFor`) and the right-click gate that rides on it
 * (`AppShell.handleRoute`) can only be
 * exercised end to end while SOME base model in the fixture actually declares
 * enough limitations to take a whole tab down — and the fixture's LTX 2.5 keeps
 * declaring fewer of them, one increment at a time. Edit needs BOTH `retake`
 * and `outpaint` refused before its tab greys, and since the Retake increment
 * LTX 2.5 refuses only `outpaint`, so the tab is live and there is nothing left
 * in the fixture world that disables a mode.
 *
 * Pinning those tests to whatever LTX 2.5 happens to declare this month is what
 * made them need rewriting at every increment. This wraps the fixture instead:
 * the response is the real one plus the names a test names, so what is being
 * tested is the MECHANISM (a published feature name greys a tab, and the gate
 * refuses a route into it) rather than today's feature list. The list itself is
 * asserted where it belongs — `bridge/mockBridge.test.ts`, against the server's
 * own `UNSUPPORTED_FEATURES`.
 *
 * Everything but that one response passes through untouched, `emit` included,
 * so a test can still drive native events through the returned object.
 */
export function withExtraUnsupportedFeatures(
  bridge: MockBridge,
  baseModelId: string,
  extra: readonly string[],
): MockBridge {
  const inner = bridge.request.bind(bridge) as (method: string, params: unknown) => Promise<unknown>;

  const request = async (method: string, params: unknown): Promise<unknown> => {
    const result = await inner(method, params);
    const req = params as { method?: string; path?: string } | undefined;
    if (method !== "backend.request" || req?.method !== "GET" || req?.path !== "/api/v1/models") {
      return result;
    }
    const response = result as {
      body?: { base_models?: Array<{ id: string; unsupported_features?: string[] }> };
    };
    const models = response.body?.base_models;
    if (!models) return result;
    return {
      ...response,
      body: {
        ...response.body,
        base_models: models.map((entry) =>
          entry.id === baseModelId
            ? { ...entry, unsupported_features: [...(entry.unsupported_features ?? []), ...extra] }
            : entry,
        ),
      },
    };
  };

  return {
    ...bridge,
    request: request as unknown as NativeBridge["request"],
    emit: bridge.emit.bind(bridge),
    releaseUploads: bridge.releaseUploads.bind(bridge),
  };
}
