import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { isOverTokenLimit, TokenCounter } from "./TokenCounter";

describe("TokenCounter", () => {
  it("shows n / limit, not red at the limit itself", () => {
    const { container } = render(<TokenCounter count={1024} limit={1024} />);
    expect(container.textContent).toBe("1024 / 1024 tokens");
    expect(container.querySelector(".field-hint-error")).toBeNull();
  });

  it("turns red one past the limit", () => {
    const { container } = render(<TokenCounter count={1025} limit={1024} />);
    expect(container.textContent).toBe("1025 / 1024 tokens");
    expect(container.querySelector(".token-counter.field-hint-error")).not.toBeNull();
  });

  it("shows the bare count and never goes red without a limit", () => {
    const { container } = render(<TokenCounter count={99999} limit={null} />);
    expect(container.textContent).toBe("99999 tokens");
    expect(container.querySelector(".field-hint-error")).toBeNull();
  });

  it("renders nothing without a count", () => {
    expect(render(<TokenCounter count={null} limit={1024} />).container.innerHTML).toBe("");
    expect(render(<TokenCounter count={undefined} limit={1024} />).container.innerHTML).toBe("");
  });

  it("isOverTokenLimit needs both a count and a limit", () => {
    expect(isOverTokenLimit(1025, 1024)).toBe(true);
    expect(isOverTokenLimit(1024, 1024)).toBe(false);
    expect(isOverTokenLimit(5000, null)).toBe(false);
    expect(isOverTokenLimit(null, 1024)).toBe(false);
  });
});
