import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { DEFAULT_REFERENCE_STRENGTH, MAX_REFERENCE_STRENGTH, MIN_REFERENCE_STRENGTH } from "../chained/chainUtils";
import { ReferenceStrengthField } from "./ReferenceStrengthField";

/** `ReferenceStrengthField` is a controlled `value`/`onChange` pair — this
 * harness closes the loop with a local `useState` so ticking the checkbox
 * (which calls `onChange(DEFAULT_REFERENCE_STRENGTH)`) is actually reflected
 * back into the rendered `value`, letting the test assert on the range
 * input's resulting numeric value rather than just the `onChange` call args. */
function Harness({ initial = null }: { initial?: number | null }) {
  const [value, setValue] = useState<number | null>(initial);
  return <ReferenceStrengthField label="Reference video strength" value={value} onChange={setValue} disabled={false} />;
}

describe("ReferenceStrengthField", () => {
  it("DEFAULT_REFERENCE_STRENGTH is 1.0 (owner decision #3: raised from 0.5)", () => {
    // Guards the constant itself, independent of the component — a future
    // regression back to 0.5 (or any other value) fails here even if the
    // component-level test below were somehow skipped.
    expect(DEFAULT_REFERENCE_STRENGTH).toBe(1.0);
  });

  it("starts unchecked with no range input when value is null", () => {
    render(<Harness />);
    const checkbox = screen.getByRole("checkbox");
    expect(checkbox).not.toBeChecked();
    expect(screen.queryByRole("slider")).not.toBeInTheDocument();
  });

  it("ticking the enable checkbox seeds the range input at 1.0, not the old 0.5", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    await user.click(screen.getByRole("checkbox"));

    const slider = screen.getByRole("slider") as HTMLInputElement;
    expect(slider.valueAsNumber).toBe(1.0);
    expect(screen.getByText("1.00")).toBeInTheDocument();
  });

  it("the range covers the full [0, 1] span regardless of the seeded default", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(screen.getByRole("checkbox"));

    const slider = screen.getByRole("slider") as HTMLInputElement;
    expect(Number(slider.min)).toBe(MIN_REFERENCE_STRENGTH);
    expect(Number(slider.max)).toBe(MAX_REFERENCE_STRENGTH);
  });

  it("unticking clears back to null and hides the range input", async () => {
    const user = userEvent.setup();
    render(<Harness initial={0.7} />);
    expect(screen.getByRole("slider")).toBeInTheDocument();

    await user.click(screen.getByRole("checkbox"));

    expect(screen.queryByRole("slider")).not.toBeInTheDocument();
  });
});
