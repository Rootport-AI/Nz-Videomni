import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import {
  PrefillPolicyProvider,
  usePrefillPolicy,
  PREFILL_SIZE_POLICY_STORAGE_KEY,
  PREFILL_FPS_POLICY_STORAGE_KEY,
  DEFAULT_PREFILL_SIZE_POLICY,
  DEFAULT_PREFILL_FPS_POLICY,
  readStoredSizePolicy,
  readStoredFpsPolicy,
} from "./PrefillPolicyContext";

function wrapper({ children }: { children: ReactNode }) {
  return <PrefillPolicyProvider>{children}</PrefillPolicyProvider>;
}

// W1: the former single `nzvideomni.prefillResolutionPolicy` key is deliberately
// no longer read — kept here only to assert it is IGNORED (no migration).
const LEGACY_STORAGE_KEY = "nzvideomni.prefillResolutionPolicy";

describe("PrefillPolicyContext", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    window.localStorage.clear();
  });

  it("defaults size to material and fps to project when nothing is stored", () => {
    const { result } = renderHook(() => usePrefillPolicy(), { wrapper });
    expect(result.current.sizePolicy).toBe("material");
    expect(result.current.fpsPolicy).toBe("project");
    expect(DEFAULT_PREFILL_SIZE_POLICY).toBe("material");
    expect(DEFAULT_PREFILL_FPS_POLICY).toBe("project");
  });

  it("restores each axis from its own persisted key on mount", () => {
    window.localStorage.setItem(PREFILL_SIZE_POLICY_STORAGE_KEY, "project");
    window.localStorage.setItem(PREFILL_FPS_POLICY_STORAGE_KEY, "defaults");
    const { result } = renderHook(() => usePrefillPolicy(), { wrapper });
    expect(result.current.sizePolicy).toBe("project");
    expect(result.current.fpsPolicy).toBe("defaults");
  });

  it("§3-13: restores a persisted fps=material unchanged (the X1 coercion is gone)", () => {
    // Both axes honour "material" now: contract v11 gives the fps axis a real
    // material framerate to read (`mediaFps`, snapped in `prefillSeed.ts`), so a
    // stored "material" is a valid choice rather than a stale value to correct.
    window.localStorage.setItem(PREFILL_SIZE_POLICY_STORAGE_KEY, "material");
    window.localStorage.setItem(PREFILL_FPS_POLICY_STORAGE_KEY, "material");
    const { result } = renderHook(() => usePrefillPolicy(), { wrapper });
    expect(result.current.sizePolicy).toBe("material");
    expect(result.current.fpsPolicy).toBe("material");
    expect(readStoredFpsPolicy()).toBe("material");
  });

  it("ignores the legacy single key entirely (no migration)", () => {
    window.localStorage.setItem(LEGACY_STORAGE_KEY, "defaults");
    const { result } = renderHook(() => usePrefillPolicy(), { wrapper });
    // The legacy value is neither read nor migrated — each axis stays at its default.
    expect(result.current.sizePolicy).toBe("material");
    expect(result.current.fpsPolicy).toBe("project");
  });

  it("ignores an unrecognized stored value and falls back to each axis default", () => {
    window.localStorage.setItem(PREFILL_SIZE_POLICY_STORAGE_KEY, "banana");
    window.localStorage.setItem(PREFILL_FPS_POLICY_STORAGE_KEY, "banana");
    const { result } = renderHook(() => usePrefillPolicy(), { wrapper });
    expect(result.current.sizePolicy).toBe("material");
    expect(result.current.fpsPolicy).toBe("project");
    expect(readStoredSizePolicy()).toBe("material");
    expect(readStoredFpsPolicy()).toBe("project");
  });

  it("setSizePolicy/setFpsPolicy update state and persist each choice independently", () => {
    const { result } = renderHook(() => usePrefillPolicy(), { wrapper });

    act(() => {
      result.current.setSizePolicy("defaults");
    });
    expect(result.current.sizePolicy).toBe("defaults");
    // The fps axis is untouched by a size-axis change.
    expect(result.current.fpsPolicy).toBe("project");
    expect(window.localStorage.getItem(PREFILL_SIZE_POLICY_STORAGE_KEY)).toBe("defaults");

    act(() => {
      result.current.setFpsPolicy("project");
    });
    expect(result.current.fpsPolicy).toBe("project");
    expect(result.current.sizePolicy).toBe("defaults");
    expect(window.localStorage.getItem(PREFILL_FPS_POLICY_STORAGE_KEY)).toBe("project");
    expect(readStoredSizePolicy()).toBe("defaults");
    expect(readStoredFpsPolicy()).toBe("project");
  });

  it("§3-13: setFpsPolicy('material') is kept and persisted as material (no coercion)", () => {
    const { result } = renderHook(() => usePrefillPolicy(), { wrapper });

    act(() => {
      result.current.setFpsPolicy("material");
    });
    // The fps axis stores "material" verbatim — the X1 二重防御 that rewrote it
    // to "project" is gone now that the choice does something.
    expect(result.current.fpsPolicy).toBe("material");
    expect(window.localStorage.getItem(PREFILL_FPS_POLICY_STORAGE_KEY)).toBe("material");
    expect(readStoredFpsPolicy()).toBe("material");

    // The SIZE axis accepts "material" too, and the two stay independent.
    act(() => {
      result.current.setSizePolicy("material");
    });
    expect(result.current.sizePolicy).toBe("material");
    expect(window.localStorage.getItem(PREFILL_SIZE_POLICY_STORAGE_KEY)).toBe("material");
  });

  it("falls back to each axis default when localStorage throws", () => {
    const getItemSpy = vi.spyOn(window.localStorage.__proto__, "getItem").mockImplementation(() => {
      throw new Error("storage disabled");
    });

    const { result } = renderHook(() => usePrefillPolicy(), { wrapper });
    expect(result.current.sizePolicy).toBe("material");
    expect(result.current.fpsPolicy).toBe("project");

    getItemSpy.mockRestore();
  });

  it("usePrefillPolicy throws when used outside a PrefillPolicyProvider", () => {
    expect(() => renderHook(() => usePrefillPolicy())).toThrow(/PrefillPolicyProvider/);
  });
});
