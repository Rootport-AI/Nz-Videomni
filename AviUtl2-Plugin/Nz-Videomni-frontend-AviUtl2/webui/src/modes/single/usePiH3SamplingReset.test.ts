import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { PiH3Mode } from "../../shell/accelerationSettings";
import { usePiH3SamplingReset } from "./usePiH3SamplingReset";

/** Ruling 107 ②: the one shared "pi-H3 changed → reset the sampling values"
 * hook the five forms call. */
describe("usePiH3SamplingReset", () => {
  function renderReset(initial: PiH3Mode) {
    const reset = vi.fn();
    const view = renderHook(({ piH3 }: { piH3: PiH3Mode }) => usePiH3SamplingReset(piH3, reset), {
      initialProps: { piH3: initial },
    });
    return { ...view, reset };
  }

  it("does not call reset on the first render, whatever the choice", () => {
    for (const piH3 of ["off", "4steps", "2steps"] as const) {
      expect(renderReset(piH3).reset).not.toHaveBeenCalled();
    }
  });

  it("calls reset once per change of the choice, and not on a re-render with the same one", () => {
    const { rerender, reset } = renderReset("off");
    rerender({ piH3: "off" });
    expect(reset).not.toHaveBeenCalled();
    rerender({ piH3: "4steps" });
    expect(reset).toHaveBeenCalledTimes(1);
    rerender({ piH3: "4steps" });
    expect(reset).toHaveBeenCalledTimes(1);
    rerender({ piH3: "2steps" });
    expect(reset).toHaveBeenCalledTimes(2);
    rerender({ piH3: "off" });
    expect(reset).toHaveBeenCalledTimes(3);
  });

  it("a re-created reset function alone does not call it", () => {
    const first = vi.fn();
    const second = vi.fn();
    const { rerender } = renderHook(
      ({ reset }: { reset: () => void }) => usePiH3SamplingReset("4steps", reset),
      { initialProps: { reset: first } },
    );
    rerender({ reset: second });
    expect(first).not.toHaveBeenCalled();
    expect(second).not.toHaveBeenCalled();
  });
});
