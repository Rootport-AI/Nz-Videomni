import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { AppConfig } from "../api/types";
import type { LorasState } from "../modes/inventory/useLoras";
import { useControlLoraNames } from "./useControlLoraNames";

// Only `model` is ever read by `resolveControlLoraNames` — the rest of
// `AppConfig` is irrelevant to these tests, so this stub only needs to
// satisfy the type, not represent a fully-populated config (mirrors
// `lora/controlLoras.test.ts`'s own `configWith` helper).
const CONFIG = {
  model: { ic_loras: { "canny-control": { path: "x", preprocess: "canny" } } },
} as unknown as AppConfig;

const READY_LORAS = [
  { name: "canny-control", kind: "control" as const, has_thumbnail: true, exists: true, source: "config" },
];

/** Mirrors `useLoras`'s own return shape (`modes/inventory/useLoras.ts:52`,
 * `return { ...state, refresh }`) — a BRAND-NEW object literal every call,
 * even though the underlying `loras` array and `status` haven't changed.
 * This is exactly the shape that made the pre-fix `useMemo(..., [config,
 * lorasState])` recompute (and return a new `Set`) on every single render
 * (M-1, post-implementation adversarial review, 2026-07-17). */
function freshLorasStateWrapper(status: "ready", loras: typeof READY_LORAS): LorasState & { refresh: () => Promise<void> } {
  return { status, loras, refresh: async () => {} };
}

describe("useControlLoraNames (M-1 regression guard)", () => {
  it("returns a referentially-stable Set across renders where lorasState is a NEW wrapper object but its status/loras array are unchanged", () => {
    const { result, rerender } = renderHook(
      ({ lorasState }: { lorasState: LorasState }) => useControlLoraNames(CONFIG, lorasState),
      { initialProps: { lorasState: freshLorasStateWrapper("ready", READY_LORAS) } },
    );
    const first = result.current;
    expect(first).toEqual(new Set(["canny-control"]));

    // A fresh `{ ...state, refresh }` wrapper (new object identity, same
    // `status` and same `loras` ARRAY reference) — mirrors what happens on
    // every unrelated re-render of a component that calls `useLoras()`.
    rerender({ lorasState: freshLorasStateWrapper("ready", READY_LORAS) });

    expect(result.current).toBe(first); // same Set instance, not just equal
  });

  it("recomputes (a new Set) once the loras ARRAY reference actually changes (e.g. a refresh() resolved with new data)", () => {
    const { result, rerender } = renderHook(
      ({ lorasState }: { lorasState: LorasState }) => useControlLoraNames(CONFIG, lorasState),
      { initialProps: { lorasState: freshLorasStateWrapper("ready", READY_LORAS) } },
    );
    const first = result.current;

    const refreshedLoras = [...READY_LORAS, { name: "pose-control", kind: "control" as const, has_thumbnail: false, exists: true, source: "config" }];
    rerender({ lorasState: freshLorasStateWrapper("ready", refreshedLoras) });

    expect(result.current).not.toBe(first);
    expect(result.current).toEqual(new Set(["canny-control", "pose-control"]));
  });

  it("recomputes when status transitions (loading -> ready), even with a stable config", () => {
    const { result, rerender } = renderHook(
      ({ lorasState }: { lorasState: LorasState }) => useControlLoraNames(CONFIG, lorasState),
      { initialProps: { lorasState: { status: "loading" } as LorasState } },
    );
    // Falls back to config.model.ic_loras while loading.
    expect(result.current).toEqual(new Set(["canny-control"]));

    rerender({ lorasState: freshLorasStateWrapper("ready", []) });

    // GET /loras resolved with no control entries -> now empty, distinct
    // from the loading-time fallback.
    expect(result.current).toEqual(new Set());
  });
});
