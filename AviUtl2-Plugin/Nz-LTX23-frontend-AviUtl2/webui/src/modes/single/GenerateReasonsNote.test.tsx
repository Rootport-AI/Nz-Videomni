import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { GenerateReasonsNote } from "./GenerateReasonsNote";

// W7: `GenerateReasonsNote` is the pure display component shared by Create and
// Chain — it turns a list of failing-gate reason codes + a code→message map
// into one imperative line per DISTINCT message.

const MESSAGES = {
  promptEmpty: "Fill the main prompt.",
  cropInvalid: "Fix the output crop size.",
  controlNeedsReference: "Attach a reference video.",
  referenceNotReady: "Attach a reference video.", // same line as controlNeedsReference
};

describe("GenerateReasonsNote", () => {
  it("renders nothing when there are no reasons", () => {
    const { container } = render(<GenerateReasonsNote reasons={[]} messages={MESSAGES} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders one line per reason, in order, as a mild warning banner", () => {
    render(<GenerateReasonsNote reasons={["promptEmpty", "cropInvalid"]} messages={MESSAGES} />);
    const items = screen.getAllByRole("listitem").map((li) => li.textContent);
    expect(items).toEqual(["Fill the main prompt.", "Fix the output crop size."]);
    expect(screen.getByRole("note")).toHaveClass("warning-banner", "warning-banner-mild");
  });

  it("de-dups codes that resolve to the same message (R2: control-needs-reference + reference-not-ready)", () => {
    render(
      <GenerateReasonsNote reasons={["controlNeedsReference", "referenceNotReady"]} messages={MESSAGES} />,
    );
    // Both codes map to "Attach a reference video." — it must appear exactly once.
    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(1);
    expect(items[0]?.textContent).toBe("Attach a reference video.");
  });

  it("skips codes with no entry in the message map rather than rendering a blank line", () => {
    render(<GenerateReasonsNote reasons={["promptEmpty", "unknownCode"]} messages={MESSAGES} />);
    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(1);
    expect(items[0]?.textContent).toBe("Fill the main prompt.");
  });

  it("renders nothing when every reason code is unknown", () => {
    const { container } = render(<GenerateReasonsNote reasons={["nope", "nada"]} messages={MESSAGES} />);
    expect(container.firstChild).toBeNull();
  });
});
