import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { GenerateButtonBar } from "./GenerateButtonBar";

// §1-80: `belowButton` is optional. Edit's three Generate bars never pass it,
// so their DOM must stay byte-for-byte what it was before the prop existed.
describe("GenerateButtonBar", () => {
  it("renders exactly the button + hint when belowButton is omitted", () => {
    const { container } = render(<GenerateButtonBar label="Generate" disabled={false} onGenerate={() => {}} hint="5.0s" />);
    expect(container.innerHTML).toBe(
      '<div class="generate-bar">' +
        '<button type="button" class="primary-button generate-button">Generate</button>' +
        '<span class="field-hint generate-bar-hint">5.0s</span>' +
        "</div>",
    );
  });

  it("renders only the button when neither belowButton nor hint is given", () => {
    const { container } = render(<GenerateButtonBar label="Generate" disabled onGenerate={() => {}} />);
    expect(container.innerHTML).toBe(
      '<div class="generate-bar">' +
        '<button type="button" class="primary-button generate-button" disabled="">Generate</button>' +
        "</div>",
    );
  });

  it("renders belowButton between the button and the hint", () => {
    const { container } = render(
      <GenerateButtonBar
        label="Generate"
        disabled={false}
        onGenerate={() => {}}
        belowButton={<span data-testid="below">below</span>}
        hint="hint"
      />,
    );
    const children = Array.from(container.querySelector(".generate-bar")!.children);
    expect(children.map((el) => el.tagName)).toEqual(["BUTTON", "SPAN", "SPAN"]);
    expect(children[1]).toHaveAttribute("data-testid", "below");
    expect(children[2]).toHaveClass("generate-bar-hint");
  });
});
