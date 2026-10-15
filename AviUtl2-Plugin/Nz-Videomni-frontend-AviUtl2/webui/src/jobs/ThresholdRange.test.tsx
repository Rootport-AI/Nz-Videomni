import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "../i18n/LanguageContext";
import { en } from "../i18n/strings";
import { ThresholdRange } from "./ThresholdRange";

/**
 * Alpha Gen 第 3 弾: the two-thumb threshold slider. The thumbs never cross
 * (held on the value side), equal values are allowed, and both stay ordinary
 * `<input type="range">` so the keyboard is the browser's own (jsdom does not
 * move a range on arrow keys, so a keyboard step is driven as the `change` it
 * produces).
 */
const t = en.jobs.alpha;

function Harness({ initial, onChange }: { initial: [number, number]; onChange?: (l: number, r: number) => void }) {
  const [[l, r], setValues] = useState(initial);
  return (
    <ThresholdRange
      transparent={l}
      opaque={r}
      onChange={(nextL, nextR) => {
        setValues([nextL, nextR]);
        onChange?.(nextL, nextR);
      }}
    />
  );
}

function renderRange(initial: [number, number], onChange?: (l: number, r: number) => void) {
  render(
    <LanguageProvider>
      <Harness initial={initial} {...(onChange ? { onChange } : {})} />
    </LanguageProvider>,
  );
  return {
    left: () => screen.getByRole("slider", { name: t.transparentThreshold }) as HTMLInputElement,
    right: () => screen.getByRole("slider", { name: t.opaqueThreshold }) as HTMLInputElement,
  };
}

describe("ThresholdRange", () => {
  it("two native 0..255 integer sliders with their own labels and tooltips", () => {
    const { left, right } = renderRange([7, 248]);
    for (const input of [left(), right()]) {
      expect(input.type).toBe("range");
      expect(input.min).toBe("0");
      expect(input.max).toBe("255");
      expect(input.step).toBe("1");
    }
    expect(left().value).toBe("7");
    expect(right().value).toBe("248");
    expect(left().title).toBe(t.transparentThresholdTitle);
    expect(right().title).toBe(t.opaqueThresholdTitle);
  });

  it("the left thumb stops at the right one, and the right thumb at the left one (they never cross)", () => {
    const onChange = vi.fn();
    const { left, right } = renderRange([50, 100], onChange);
    fireEvent.change(left(), { target: { value: "180" } });
    expect(onChange).toHaveBeenLastCalledWith(100, 100);
    expect(left().value).toBe("100");
    fireEvent.change(right(), { target: { value: "20" } });
    expect(onChange).toHaveBeenLastCalledWith(100, 100);
    expect(right().value).toBe("100");
  });

  it("equal values are allowed and either thumb can still move away (a keyboard step)", () => {
    const onChange = vi.fn();
    const { left, right } = renderRange([128, 128], onChange);
    // Arrow-right on the left thumb: held at the right one.
    fireEvent.change(left(), { target: { value: "129" } });
    expect(onChange).toHaveBeenLastCalledWith(128, 128);
    // Arrow-left on the left thumb / arrow-right on the right thumb: free.
    fireEvent.change(left(), { target: { value: "127" } });
    expect(onChange).toHaveBeenLastCalledWith(127, 128);
    fireEvent.change(right(), { target: { value: "129" } });
    expect(onChange).toHaveBeenLastCalledWith(127, 129);
  });

  it("in the upper half the left input is lifted on top, so a pair stuck at 255 can be pulled apart", () => {
    const { left } = renderRange([255, 255]);
    expect(left().className).toContain("threshold-range-input--top");
    fireEvent.change(left(), { target: { value: "200" } });
    expect(left().value).toBe("200");
  });

  it("in the lower half the right input stays on top", () => {
    const { left } = renderRange([0, 0]);
    expect(left().className).not.toContain("threshold-range-input--top");
  });
});
