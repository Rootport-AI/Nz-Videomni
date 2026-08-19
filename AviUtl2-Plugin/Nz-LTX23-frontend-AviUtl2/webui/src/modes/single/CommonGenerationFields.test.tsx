import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { JobResponse } from "../../api/types";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en as strings } from "../../i18n/strings";
import { DurationField, FrameRateSeedFields, ReservedFields, SizeFields } from "./CommonGenerationFields";
import type { DurationFieldProps, SizeFieldsProps } from "./CommonGenerationFields";

// FrameRateSeedFields reads the job ledger via useJobsContext for the ♻ button;
// stub it so these unit tests don't need a full JobsProvider.
const mockJobs = vi.fn<() => JobResponse[]>(() => []);
vi.mock("../../jobs/JobsContext", () => ({
  useJobsContext: () => ({ jobs: mockJobs() }),
}));

function seedJob(seedUsed: number, createdAt = "2026-07-17T00:00:00Z"): JobResponse {
  return {
    job_id: `job-${seedUsed}`,
    status: "completed",
    progress: 1,
    current_step: null,
    total_steps: null,
    stage: null,
    clip: null,
    clip_count: null,
    is_v2v: false,
    joined: false,
    created_at: createdAt,
    started_at: null,
    completed_at: null,
    error: null,
    request: {},
    result: {
      video_url: "/v",
      duration_seconds: 2,
      resolution: "1x1",
      file_size_bytes: 1,
      generation_time_seconds: 1,
      seed_used: seedUsed,
      output_path: "/o",
      metadata_path: "/m",
    },
  };
}

function renderField(overrides: Partial<DurationFieldProps> = {}) {
  const onChange = vi.fn();
  const props: DurationFieldProps = {
    label: "Duration",
    value: 81,
    min: 9,
    max: 161,
    disabled: false,
    onChange,
    hint: "81 frames",
    ...overrides,
  };
  render(
    <LanguageProvider>
      <DurationField {...props} />
    </LanguageProvider>,
  );
  const range = screen.getByRole("slider") as HTMLInputElement;
  const number = screen.getByRole("spinbutton") as HTMLInputElement;
  return { onChange, range, number };
}

describe("DurationField onCommit", () => {
  it("range: fires onCommit(value, true) on pointerup", () => {
    const onCommit = vi.fn();
    const { range } = renderField({ onCommit });

    fireEvent.pointerUp(range, { target: { value: "97" } });

    expect(onCommit).toHaveBeenCalledWith(97, true);
  });

  it("range: fires onCommit(value, true) on keyup", () => {
    const onCommit = vi.fn();
    const { range } = renderField({ onCommit });

    fireEvent.keyUp(range, { target: { value: "89" } });

    expect(onCommit).toHaveBeenCalledWith(89, true);
  });

  it("range: fires onCommit(value, true) on blur", () => {
    const onCommit = vi.fn();
    const { range } = renderField({ onCommit });

    fireEvent.blur(range, { target: { value: "105" } });

    expect(onCommit).toHaveBeenCalledWith(105, true);
  });

  it("number: fires onCommit(value, false) on blur", () => {
    const onCommit = vi.fn();
    const { number } = renderField({ onCommit });

    fireEvent.blur(number, { target: { value: "73" } });

    expect(onCommit).toHaveBeenCalledWith(73, false);
  });

  it("number: fires onCommit(value, false) on Enter keydown", () => {
    const onCommit = vi.fn();
    const { number } = renderField({ onCommit });

    fireEvent.keyDown(number, { key: "Enter", target: { value: "121" } });

    expect(onCommit).toHaveBeenCalledWith(121, false);
  });

  it("number: does not fire onCommit on non-Enter keydown", () => {
    const onCommit = vi.fn();
    const { number } = renderField({ onCommit });

    fireEvent.keyDown(number, { key: "a", target: { value: "121" } });

    expect(onCommit).not.toHaveBeenCalled();
  });

  it("number: stepper-driven onChange (isStepEvent) immediately fires onCommit(value, true)", () => {
    const onCommit = vi.fn();
    const onChange = vi.fn();
    const { number } = renderField({ onCommit, onChange });

    // A stepper/spinner-driven change carries no InputEvent.inputType,
    // matching paramUtils.isStepEvent's detection of non-typed input.
    fireEvent.change(number, { target: { value: "89" } });

    expect(onChange).toHaveBeenCalledWith(89, true);
    expect(onCommit).toHaveBeenCalledWith(89, true);
  });

  it("number: manually-typed onChange (has inputType) does not fire onCommit", () => {
    const onCommit = vi.fn();
    const onChange = vi.fn();
    const { number } = renderField({ onCommit, onChange });

    fireEvent.input(number, { target: { value: "89" }, inputType: "insertText" });

    expect(onChange).toHaveBeenCalledWith(89, false);
    expect(onCommit).not.toHaveBeenCalled();
  });

  it("onCommit unspecified: no error, onChange still fires as before", () => {
    const { onChange, range, number } = renderField();

    expect(() => {
      fireEvent.pointerUp(range, { target: { value: "97" } });
      fireEvent.keyUp(range, { target: { value: "97" } });
      fireEvent.blur(range, { target: { value: "97" } });
      fireEvent.blur(number, { target: { value: "73" } });
      fireEvent.keyDown(number, { key: "Enter", target: { value: "73" } });
      fireEvent.change(range, { target: { value: "105" } });
    }).not.toThrow();

    expect(onChange).toHaveBeenCalledWith(105, true);
  });
});

describe("FrameRateSeedFields seed buttons (⑤)", () => {
  beforeEach(() => {
    mockJobs.mockReturnValue([]);
  });

  function renderSeed() {
    const onSeedChange = vi.fn();
    render(
      <LanguageProvider>
        <FrameRateSeedFields
          frameRate={16}
          seed={42}
          disabled={false}
          onFrameRateChange={vi.fn()}
          onSeedChange={onSeedChange}
        />
      </LanguageProvider>,
    );
    return { onSeedChange };
  }

  it("🎲 randomizes the seed via onSeedChange(-1)", () => {
    const { onSeedChange } = renderSeed();
    fireEvent.click(screen.getByRole("button", { name: /randomize seed/i }));
    expect(onSeedChange).toHaveBeenCalledWith(-1);
  });

  it("♻ reuses the newest job's seed", () => {
    mockJobs.mockReturnValue([
      seedJob(1, "2026-07-17T00:00:00Z"),
      seedJob(555, "2026-07-17T09:00:00Z"),
    ]);
    const { onSeedChange } = renderSeed();
    fireEvent.click(screen.getByRole("button", { name: /reuse last seed/i }));
    expect(onSeedChange).toHaveBeenCalledWith(555);
  });

  it("♻ is disabled when there are no jobs to reuse a seed from", () => {
    const { onSeedChange } = renderSeed();
    const reuse = screen.getByRole("button", { name: /reuse last seed/i });
    expect(reuse).toBeDisabled();
    fireEvent.click(reuse);
    expect(onSeedChange).not.toHaveBeenCalled();
  });
});

describe("SizeFields budgetMarkers (2026-08-12)", () => {
  function renderSizeFields(overrides: Partial<SizeFieldsProps> = {}) {
    const props: SizeFieldsProps = {
      width: 1280,
      height: 768,
      minWidth: 128,
      maxWidth: 1920,
      minHeight: 128,
      maxHeight: 1088,
      disabled: false,
      onWidthChange: vi.fn(),
      onHeightChange: vi.fn(),
      ...overrides,
    };
    return render(
      <LanguageProvider>
        <SizeFields {...props} />
      </LanguageProvider>,
    );
  }

  it("draws no guide at all when the caller passes none (Create/Retake stay untouched)", () => {
    const { container } = renderSizeFields();
    expect(container.querySelector("[data-size-marker]")).toBeNull();
    // The sliders themselves are still the same two operable controls.
    expect(screen.getAllByRole("slider")).toHaveLength(2);
  });

  it("keeps each slider reachable by its label through the new wrapper span", () => {
    // The guide overlay put a `<span>` between the `<label>` and its range
    // input. The implicit label association is "the first labelable DESCENDANT",
    // so the name must survive that extra level — this is the one real
    // regression risk the wrapper carries for Create/Retake.
    renderSizeFields({ budgetMarkers: { comfortWidth: 1792, comfortHeight: 1024, limitWidth: null, limitHeight: null } });
    expect(screen.getByRole("slider", { name: /Width/ })).toBeInTheDocument();
    expect(screen.getByRole("slider", { name: /Height/ })).toBeInTheDocument();
  });

  it("skips a guide that falls outside its own slider's range, and keeps the ones inside", () => {
    const { container } = renderSizeFields({
      // Width guide above the 1920 ceiling -> not drawn; height guide inside.
      budgetMarkers: { comfortWidth: 2560, comfortHeight: 1024, limitWidth: null, limitHeight: null },
    });
    const drawn = Array.from(container.querySelectorAll<HTMLElement>("[data-size-marker]"));
    expect(drawn).toHaveLength(1);
    expect(drawn[0]!.dataset.markerValue).toBe("1024");
  });

  it("draws a guide sitting exactly ON the ceiling (closed interval)", () => {
    const { container } = renderSizeFields({
      budgetMarkers: { comfortWidth: 1920, comfortHeight: 1088, limitWidth: null, limitHeight: null },
    });
    expect(container.querySelectorAll("[data-size-marker]")).toHaveLength(2);
  });

  it("puts the red ceiling after the neutral guide so it wins an overlap", () => {
    const { container } = renderSizeFields({
      budgetMarkers: { comfortWidth: 1792, comfortHeight: 1024, limitWidth: 1792, limitHeight: null },
    });
    const widthField = container.querySelectorAll<HTMLElement>(".size-field-stack .field")[0]!;
    const kinds = Array.from(widthField.querySelectorAll<HTMLElement>("[data-size-marker]")).map(
      (el) => el.dataset.sizeMarker,
    );
    expect(kinds).toEqual(["comfort", "limit"]);
  });

  it("hides every guide from assistive technology (it duplicates the slider's own value)", () => {
    const { container } = renderSizeFields({
      budgetMarkers: { comfortWidth: 1792, comfortHeight: 1024, limitWidth: null, limitHeight: null },
    });
    for (const marker of container.querySelectorAll("[data-size-marker]")) {
      expect(marker).toHaveAttribute("aria-hidden", "true");
    }
  });

  // A-1 (2026-08-12, adversarial review): the tooltip was previously
  // unreachable because `.size-marker` carried `pointer-events: none` — the
  // `title` attribute was present in the DOM but no hover could ever land on
  // it. This asserts the attribute itself carries the exact copy
  // `strings.single.size.comfortMarkerTitle`/`limitMarkerTitle` produce, so a
  // future regression that drops or mismatches `title` (rather than the CSS
  // hit area) is still caught here even though jsdom can't verify hover.
  it("gives each guide a title tooltip with the comfort/limit copy", () => {
    const { container } = renderSizeFields({
      budgetMarkers: { comfortWidth: 1792, comfortHeight: 1024, limitWidth: 1600, limitHeight: null },
    });
    const widthField = container.querySelectorAll<HTMLElement>(".size-field-stack .field")[0]!;
    const comfortMarker = widthField.querySelector<HTMLElement>('[data-size-marker="comfort"]')!;
    const limitMarker = widthField.querySelector<HTMLElement>('[data-size-marker="limit"]')!;
    expect(comfortMarker).toHaveAttribute("title", strings.single.size.comfortMarkerTitle(1792));
    expect(limitMarker).toHaveAttribute("title", strings.single.size.limitMarkerTitle(1600));
  });
});

describe("ReservedFields (D9: negative-prompt textarea removed by NAG, 2026-07-28)", () => {
  it("no longer renders a negative-prompt textarea — shell/NagAccordion.tsx owns that field now", () => {
    render(
      <LanguageProvider>
        <ReservedFields />
      </LanguageProvider>,
    );
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("still shows the permanently-disabled CFG scale display, fixed at 1.0", () => {
    render(
      <LanguageProvider>
        <ReservedFields />
      </LanguageProvider>,
    );
    const cfgSlider = screen.getByRole("slider") as HTMLInputElement;
    expect(cfgSlider).toBeDisabled();
    expect(cfgSlider.valueAsNumber).toBe(1);
    expect(screen.getByText("1.0")).toBeInTheDocument();
  });
});
