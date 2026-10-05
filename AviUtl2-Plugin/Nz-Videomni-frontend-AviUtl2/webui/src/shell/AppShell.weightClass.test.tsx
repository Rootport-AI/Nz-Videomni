import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { bridge as appBridge, createMockBridge } from "../bridge";
import type { NativeBridge } from "../bridge";
import { AppShell } from "./AppShell";

// §1-31 (2026-10-06): the comfort rows are keyed on the LOADED transformer's
// weight class (`GET /models` → `base_models[].transformer_weight_class`). A
// transformer swapped in Settings' Models panel does not go through the header
// base-model switch, so `AppShell` re-reads `GET /models` on the falling edge
// of `pipelineLoading` — this pins that wiring end to end, through the marker
// Create actually draws.

/** True once a `POST /pipeline/load` carrying a per-category selection has
 * gone through. The Models panel talks to the app-wide bridge (`useModels`
 * falls back to the singleton `apiClient`), so that is where it is watched. */
let transformerLoaded = false;

function watchAppBridgeLoads(): void {
  const original = appBridge.request.bind(appBridge);
  vi.spyOn(appBridge, "request").mockImplementation((async (method: string, params?: unknown) => {
    const result = await original(method as never, params as never);
    const request = params as { path?: string; body?: { models?: Record<string, string> } } | undefined;
    if (method === "backend.request" && request?.path === "/api/v1/pipeline/load") {
      if (request.body?.models && Object.keys(request.body.models).length > 0) transformerLoaded = true;
    }
    return result;
  }) as NativeBridge["request"]);
}

/** The injected bridge `AppShell` (and so `useBaseModels`) reads `GET /models`
 * from: it names the active base model's class as `before` until the Models
 * panel has loaded a transformer, and as `after` from then on — the server's
 * answer once the user has loaded a different transformer file. */
function bridgeWithClassChangeOnLoad(before: string, after: string): NativeBridge {
  const bridge = createMockBridge({ delayMs: 0 });
  const original = bridge.request.bind(bridge);
  bridge.request = (async (method: string, params?: unknown) => {
    const result = await original(method as never, params as never);
    const path = (params as { path?: string } | undefined)?.path;
    if (method === "backend.request" && path === "/api/v1/models") {
      const response = result as { status: number; body: { base_models?: Record<string, unknown>[] } };
      const weightClass = transformerLoaded ? after : before;
      return {
        ...response,
        body: {
          ...response.body,
          base_models: (response.body.base_models ?? []).map((b) =>
            b.active ? { ...b, transformer_weight_class: weightClass } : b,
          ),
        },
      };
    }
    return result;
  }) as NativeBridge["request"];
  return bridge;
}

/** The value of Create's comfort-marker tick (`spill-tick` datalist) in the
 * visible tab panel, or `null` when none is drawn. */
function createMarker(): string | null {
  const option = document.querySelector('[role="tabpanel"]:not([hidden]) datalist#spill-tick option');
  return option?.getAttribute("value") ?? null;
}

afterEach(() => {
  vi.restoreAllMocks();
  transformerLoaded = false;
});

describe("AppShell — weight class follows a transformer swap (§1-31)", () => {
  it(
    "re-reads GET /models when a Models-panel load finishes, so Create's marker moves to the new class's line",
    async () => {
      // Default acceleration (not all-on): the 2.3 4bit row is conditional, so
      // the marker sits on the measured table (273 at 1280x768); the 8bit row
      // needs the class alone, so after the swap it moves to 265.
      watchAppBridgeLoads();
      render(<AppShell nativeBridge={bridgeWithClassChangeOnLoad("4bit", "8bit")} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(createMarker()).toBe("273"));

      const user = userEvent.setup();
      await user.click(screen.getByRole("button", { name: "Settings" }));
      const dialog = screen.getByRole("dialog", { name: "Settings" });
      const transformer = (await within(dialog).findByRole("combobox", {
        name: /video model \(checkpoint\)/i,
      })) as HTMLSelectElement;
      await waitFor(
        () =>
          expect(
            within(transformer).getByRole("option", { name: "Sulphur-2-base-distil-Q4_K_M" }),
          ).toBeInTheDocument(),
        { timeout: 5_000 },
      );
      await user.selectOptions(transformer, "Sulphur-2-base-distil-Q4_K_M");
      await user.click(within(dialog).getByRole("button", { name: "Load selected models" }));

      await waitFor(() => expect(transformerLoaded).toBe(true), { timeout: 5_000 });
      await waitFor(() => expect(createMarker()).toBe("265"), { timeout: 5_000 });
    },
    20_000,
  );
});
