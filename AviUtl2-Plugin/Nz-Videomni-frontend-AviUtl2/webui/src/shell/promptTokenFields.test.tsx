import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LanguageProvider } from "../i18n/LanguageContext";
import { ClipCard } from "../modes/chained/ClipCard";
import { NagAccordion } from "./NagAccordion";
import { NAG_OFF } from "./nagSettings";
import { PromptBar } from "./PromptBar";

// §1-111: where the token counter sits in each prompt field and how the field
// reacts past the limit (red frame via `input-warning`, never a block).

const noop = () => {};
const controls = {
  setEnabled: noop,
  setText: noop,
  setScale: noop,
  setTau: noop,
  setAlpha: noop,
  setMethod: noop,
  setVsfScale: noop,
  resetParams: noop,
};

describe("PromptBar", () => {
  it("shows the token counter instead of n/2000, and flags the textarea past the limit", () => {
    const { container, rerender } = render(
      <LanguageProvider>
        <PromptBar value="hello" onChange={noop} tokenCount={10} tokenLimit={1024} />
      </LanguageProvider>,
    );
    expect(container.textContent).toContain("10 / 1024 tokens");
    expect(container.textContent).not.toContain("/2000");
    expect(container.querySelector("textarea.input-warning")).toBeNull();
    rerender(
      <LanguageProvider>
        <PromptBar value="hello" onChange={noop} tokenCount={1025} tokenLimit={1024} />
      </LanguageProvider>,
    );
    expect(container.querySelector("textarea.input-warning")).not.toBeNull();
    expect(container.querySelector(".token-counter.field-hint-error")).not.toBeNull();
  });

  it("shows no counter and no hint text without a count or an error", () => {
    const { container } = render(
      <LanguageProvider>
        <PromptBar value="hello" onChange={noop} />
      </LanguageProvider>,
    );
    expect(container.querySelector(".token-counter")).toBeNull();
    expect(container.querySelector(".field-hint")).toBeNull();
  });
});

describe("NagAccordion", () => {
  it("shows the counter even while the field is greyed (engine without NAG)", () => {
    const { container } = render(
      <LanguageProvider>
        <NagAccordion nag={NAG_OFF} controls={controls} unavailable tokenCount={42} tokenLimit={null} />
      </LanguageProvider>,
    );
    expect(container.querySelector(".token-counter")?.textContent).toBe("42 tokens");
    expect(screen.getByRole("textbox", { name: /negative prompt/i })).toBeDisabled();
  });
});

describe("ClipCard", () => {
  const clip = { id: "c1", prompt: "x", numFrames: 97 };

  it("shows the clip's composed count and flags the textarea past the limit", () => {
    const { container } = render(
      <LanguageProvider>
        <ul>
          <ClipCard
            clip={clip}
            index={0}
            frameRate={24}
            disabled={false}
            canRemove={false}
            onPromptChange={noop}
            onNumFramesChange={noop}
            onRemove={noop}
            tokenCount={1100}
            tokenLimit={1024}
          />
        </ul>
      </LanguageProvider>,
    );
    expect(container.querySelector(".token-counter")?.textContent).toBe("1100 / 1024 tokens");
    expect(container.querySelector("textarea.clip-prompt-input.input-warning")).not.toBeNull();
  });
});
