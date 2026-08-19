import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { FloatSliderField } from "./FloatSliderField";

/** `FloatSliderField` is a controlled `value`/`onChange` pair — this harness
 * closes the loop with a local `useState` (mirroring
 * `ReferenceStrengthField.test.tsx`'s own `Harness`) so a change on either
 * input is actually reflected back into the OTHER one, letting the "stay in
 * sync" test assert on both rendered values rather than just `onChange`'s
 * call args. */
function Harness({ initial = 5 }: { initial?: number }) {
  const [value, setValue] = useState(initial);
  return (
    <FloatSliderField label="NAG scale" value={value} min={1} max={20} step={0.5} disabled={false} onChange={setValue} />
  );
}

describe("FloatSliderField", () => {
  it("the range and number inputs both fire onChange and stay in sync with each other", () => {
    render(<Harness />);
    const range = screen.getByRole("slider") as HTMLInputElement;
    const number = screen.getByRole("spinbutton") as HTMLInputElement;
    expect(range.valueAsNumber).toBe(5);
    expect(number.valueAsNumber).toBe(5);

    fireEvent.change(range, { target: { value: "10" } });
    expect(range.valueAsNumber).toBe(10);
    expect(number.valueAsNumber).toBe(10);

    fireEvent.change(number, { target: { value: "15" } });
    expect(range.valueAsNumber).toBe(15);
    expect(number.valueAsNumber).toBe(15);
  });

  it("clamps a typed value above max down to max", () => {
    const onChange = vi.fn();
    render(<FloatSliderField label="x" value={5} min={1} max={20} step={0.5} disabled={false} onChange={onChange} />);

    fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "999" } });

    expect(onChange).toHaveBeenCalledWith(20);
  });

  it("clamps a typed value below min up to min", () => {
    const onChange = vi.fn();
    render(<FloatSliderField label="x" value={5} min={1} max={20} step={0.5} disabled={false} onChange={onChange} />);

    fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "-999" } });

    expect(onChange).toHaveBeenCalledWith(1);
  });

  it("ignores a non-finite typed value — onChange is not called", () => {
    const onChange = vi.fn();
    render(<FloatSliderField label="x" value={5} min={1} max={20} step={0.5} disabled={false} onChange={onChange} />);

    // A real `<input type="number">` sanitizes any non-numeric assignment
    // down to `""` itself (jsdom included) — `Number("")` is `0`, which IS
    // finite, so it can never reach this component's `!Number.isFinite`
    // guard. To actually exercise that guard we bypass the element's own
    // number-input value setter with `Object.defineProperty`, forcing
    // `e.target.value` to read a literal non-numeric string the way a
    // synthetic/malformed event could.
    const spinbutton = screen.getByRole("spinbutton");
    Object.defineProperty(spinbutton, "value", { value: "not-a-number", writable: true, configurable: true });
    fireEvent.change(spinbutton);

    expect(onChange).not.toHaveBeenCalled();
  });

  it("disabled: both inputs are disabled", () => {
    const onChange = vi.fn();
    render(<FloatSliderField label="x" value={5} min={1} max={20} step={0.5} disabled onChange={onChange} />);

    expect(screen.getByRole("slider")).toBeDisabled();
    expect(screen.getByRole("spinbutton")).toBeDisabled();
  });
});
