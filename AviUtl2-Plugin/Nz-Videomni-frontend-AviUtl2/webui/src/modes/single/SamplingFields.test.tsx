import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { SamplingConstraints } from "../../api/types";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { LANG_STORAGE_KEY } from "../../i18n/strings";
import { SamplingFields } from "./CommonGenerationFields";
import type { SamplingValues } from "./samplingValues";

/** Ruling 107: `SamplingFields`' fixed-values input (the pi-H3 lock). */
const CONSTRAINTS: SamplingConstraints = {
  num_inference_steps: { min: 1, max: 50, step: 1, default: 20 },
  shift_video: { min: 1, max: 20, step: 0.1, default: 12 },
  shift_audio: { min: 1, max: 20, step: 0.1, default: 3 },
};
const VALUES: SamplingValues = { numInferenceSteps: 30, shiftVideo: 10, shiftAudio: 4 };
const FIXED: SamplingValues = { numInferenceSteps: 4, shiftVideo: 12, shiftAudio: 3 };

function renderFields(fixedValues: SamplingValues | null, onChange = vi.fn()) {
  render(
    <LanguageProvider>
      <SamplingFields
        constraints={CONSTRAINTS}
        values={VALUES}
        fixedValues={fixedValues}
        disabled={false}
        onChange={onChange}
      />
    </LanguageProvider>,
  );
  return onChange;
}

describe("SamplingFields — fixed values (ruling 107)", () => {
  beforeEach(() => {
    window.localStorage.removeItem(LANG_STORAGE_KEY);
  });

  it("null: shows the form's values, every input usable, no tooltip (as before)", () => {
    const onChange = renderFields(null);
    const sliders = screen.getAllByRole("slider") as HTMLInputElement[];
    const numbers = screen.getAllByRole("spinbutton") as HTMLInputElement[];
    expect(sliders.map((s) => s.value)).toEqual(["30", "10", "4"]);
    for (const input of [...sliders, ...numbers]) {
      expect(input).toBeEnabled();
      expect(input.closest("label")).not.toHaveAttribute("title");
    }
    fireEvent.change(sliders[0]!, { target: { value: "25" } });
    expect(onChange).toHaveBeenCalledWith("numInferenceSteps", 25);
  });

  it("set: shows the fixed values, every input disabled, each row carries the reason tooltip", () => {
    renderFields(FIXED);
    const sliders = screen.getAllByRole("slider") as HTMLInputElement[];
    const numbers = screen.getAllByRole("spinbutton") as HTMLInputElement[];
    expect(sliders.map((s) => s.value)).toEqual(["4", "12", "3"]);
    expect(numbers.map((s) => s.value)).toEqual(["4", "12", "3"]);
    for (const input of [...sliders, ...numbers]) {
      expect(input).toBeDisabled();
      expect(input.closest("label")).toHaveAttribute("title", "Fixed because pi-H3 is enabled in Settings");
    }
  });

  it("the tooltip reads the Japanese wording in Japanese", () => {
    window.localStorage.setItem(LANG_STORAGE_KEY, "ja");
    renderFields(FIXED);
    for (const slider of screen.getAllByRole("slider")) {
      expect(slider.closest("label")).toHaveAttribute("title", "Settings の pi-H3 が有効のため固定");
    }
  });
});
