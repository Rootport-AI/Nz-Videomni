/**
 * §1-14/§3-57 — the Chain screen's stage-2 window opt-in ("finishing pass step")
 * and the comfortable-resolution warning, at both the hook and the rendered-UI
 * level.
 *
 * The geometry itself (token table, preset numbers, the advance-vs-length
 * conversion) is pinned in `shell/tokenBudget.test.ts` against the same table
 * the backend uses. This file is about the FORM behaviour built on top: what
 * gets sent, what blocks Generate, and what the user actually reads on screen.
 */
import { act, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { ReactNode } from "react";
import type { AppConfig } from "../../api/types";
import { createMockBridge } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { JobsProvider } from "../../jobs/JobsContext";
import { ToastProvider } from "../../shell/ToastContext";
import { PrefillPolicyProvider } from "../../shell/PrefillPolicyContext";
import { ShowNoteProvider } from "../../shell/NoteArea";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import { ChainedScreen } from "./ChainedScreen";
import { useChainForm } from "./useChainForm";

function setup(config: AppConfig = FALLBACK_APP_CONFIG) {
  const mockBridge = createMockBridge({ delayMs: 0 });
  const { result } = renderHook(() =>
    useChainForm(config, "a cat riding a skateboard", { nativeBridge: mockBridge }),
  );
  return { result, mockBridge };
}

describe("useChainForm — stage2Window", () => {
  it("opens on the server default, so an untouched form sends no stage2_window key", () => {
    const { result } = setup();
    expect(result.current.stage2Window).toBe("standard");
    expect(result.current.buildRequest()).not.toHaveProperty("stage2_window");
  });

  it("sends the field once the user opts into the shorter step", () => {
    const { result } = setup();
    act(() => result.current.setStage2Window("high_resolution"));
    expect(result.current.stage2Window).toBe("high_resolution");
    expect(result.current.buildRequest().stage2_window).toBe("high_resolution");
  });

  it("goes back to omitting the key when the user returns to the default", () => {
    const { result } = setup();
    act(() => result.current.setStage2Window("high_resolution"));
    act(() => result.current.setStage2Window("standard"));
    expect(result.current.buildRequest()).not.toHaveProperty("stage2_window");
  });

  it("never blocks Generate by itself — the window is a choice, not a constraint", () => {
    const { result } = setup();
    const before = result.current.isValid;
    act(() => result.current.setStage2Window("high_resolution"));
    expect(result.current.isValid).toBe(before);
    expect(result.current.validityReasons).toEqual([]);
  });
});

describe("useChainForm — chainWindowOverBudget (§1-14 advisory)", () => {
  it("is false at the default resolution", () => {
    const { result } = setup();
    expect(result.current.chainWindowOverBudget).toBe(false);
  });

  it("turns on for a resolution over the budget, and off again on the shorter step", () => {
    const { result } = setup();
    act(() => {
      result.current.setWidth(1216);
      result.current.setHeight(1664);
    });
    // 1216x1664 on the standard window = 43,472 tokens > 40,000.
    expect(result.current.chainWindowOverBudget).toBe(true);
    act(() => result.current.setStage2Window("high_resolution"));
    // ...and 37,544 on the shorter one.
    expect(result.current.chainWindowOverBudget).toBe(false);
  });

  it("is ADVISORY — crossing the budget adds no validity reason of its own", () => {
    const { result } = setup();
    act(() => {
      result.current.setWidth(1216);
      result.current.setHeight(1664);
    });
    // Whatever else this resolution may or may not satisfy, the budget itself
    // contributes nothing: flipping the window flips `chainWindowOverBudget`
    // without moving a single validity reason.
    expect(result.current.chainWindowOverBudget).toBe(true);
    const reasonsWhenOver = [...result.current.validityReasons];

    act(() => result.current.setStage2Window("high_resolution"));
    expect(result.current.chainWindowOverBudget).toBe(false);
    expect(result.current.validityReasons).toEqual(reasonsWhenOver);
  });
});

// ── the served budget + the slider guides (2026-08-12) ──────────────────────
// Only the HOOK can be given a different budget: `ChainedScreen` reads its
// config through `useConfig()` (the app-wide bridge), which has no per-render
// injection point — so everything about a NON-default budget is asserted here,
// through `setup(config)`, and the rendered tests below stay on the default.
describe("useChainForm — the served comfort budget", () => {
  function withBudget(budget: number): AppConfig {
    return {
      ...FALLBACK_APP_CONFIG,
      limits: { ...FALLBACK_APP_CONFIG.limits, chain_comfort_token_budget: budget },
    };
  }

  /** An older backend simply has no such key in its `/config` payload. */
  function withoutBudget(): AppConfig {
    const { chain_comfort_token_budget: _omitted, ...limits } = FALLBACK_APP_CONFIG.limits;
    return { ...FALLBACK_APP_CONFIG, limits };
  }

  it("moves the warning line with the served budget", () => {
    const { result } = setup(withBudget(15_000));
    // 1280x768 on standard = 21,120 tokens: comfortable at 40,000, not at 15,000.
    expect(result.current.chainWindowOverBudget).toBe(true);
  });

  it("warns on the SHORTER step too once the budget is tight enough (2026-08-12 reversal)", () => {
    // The old copy only ever warned on `standard`, on the theory that the
    // shorter step always rescues it. It does not: a small enough budget (or a
    // large enough resolution) overshoots on both, and staying silent there
    // would hide the very slowdown the banner exists to announce.
    // 1280x768 is 21,120 tokens on standard and still 18,240 on the shorter
    // step — both over a 15,000 budget.
    const { result } = setup(withBudget(15_000));
    act(() => result.current.setStage2Window("high_resolution"));
    expect(result.current.stage2Window).toBe("high_resolution");
    expect(result.current.chainWindowOverBudget).toBe(true);
    // Still advisory on either step.
    expect(result.current.validityReasons).toEqual([]);
  });

  it("moves the slider guides with the served budget", () => {
    const { result } = setup(withBudget(15_000));
    expect(result.current.chainWindowMarkers.comfortWidth).toBeLessThan(1792);
  });

  it("falls back to the mirrored 40,000 when the server publishes no budget at all", () => {
    const { result } = setup(withoutBudget());
    expect(result.current.chainWindowMarkers.comfortWidth).toBe(1792);
    expect(result.current.chainWindowMarkers.comfortHeight).toBe(1024);
    expect(result.current.chainWindowOverBudget).toBe(false);
  });

  it("floors the guides onto the 128 grid while a reference video is active (§1-15)", async () => {
    const { result } = setup();
    act(() => result.current.setStage2Window("high_resolution"));
    expect(result.current.chainWindowMarkers.comfortHeight).toBe(1088);

    await act(async () => {
      await result.current.referenceVideo.uploadPath("C:\\v\\ref.mp4", "ref.mp4");
    });
    await waitFor(() => expect(result.current.isReferenceActive).toBe(true));
    // 1088 is not on the 128 grid the sliders switch to, so the guide steps
    // DOWN to the nearest value the slider can actually reach — flooring only
    // ever moves a guide to the safe side.
    expect(result.current.sizeMultiple).toBe(128);
    expect(result.current.chainWindowMarkers.comfortHeight).toBe(1024);
    expect(result.current.chainWindowMarkers.comfortWidth).toBe(1920);
  });
});

describe("useChainForm — contextFramesTooLongForWindow (the V2V 422 pre-block)", () => {
  it("reports the ceiling of the CURRENT window", () => {
    const { result } = setup();
    expect(result.current.stage2MaxContextFrames).toBe(161);
    act(() => result.current.setStage2Window("high_resolution"));
    expect(result.current.stage2MaxContextFrames).toBe(137);
  });

  it("never fires without a source video attached (contextFrames is meaningless then)", () => {
    const { result } = setup();
    act(() => result.current.setStage2Window("high_resolution"));
    act(() => result.current.setContextFrames(145));
    expect(result.current.hasSourceVideo).toBe(false);
    expect(result.current.validityReasons).not.toContain("contextFramesTooLongForWindow");
  });

  it("blocks Generate for a 145-frame carry-over on the shorter step, once a source is attached", async () => {
    const { result } = setup();
    await act(async () => {
      await result.current.attachSourceByPath("C:\\v\\clip.mp4", "clip.mp4");
    });
    await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));

    act(() => result.current.setContextFrames(145));
    // Still fine on the standard window (the server's own cap is 145).
    expect(result.current.validityReasons).not.toContain("contextFramesTooLongForWindow");

    act(() => result.current.setStage2Window("high_resolution"));
    expect(result.current.validityReasons).toContain("contextFramesTooLongForWindow");
    expect(result.current.isValid).toBe(false);

    // 137 is the documented ceiling and must pass.
    act(() => result.current.setContextFrames(137));
    expect(result.current.validityReasons).not.toContain("contextFramesTooLongForWindow");
  });
});

// ── rendered UI ─────────────────────────────────────────────────────────────
function Providers({ children, nativeBridge }: { children: ReactNode; nativeBridge: ReturnType<typeof createMockBridge> }) {
  return (
    <LanguageProvider>
      <ToastProvider>
        <PrefillPolicyProvider>
          <ShowNoteProvider showNote={() => {}}>
            <JobsProvider nativeBridge={nativeBridge}>{children}</JobsProvider>
          </ShowNoteProvider>
        </PrefillPolicyProvider>
      </ToastProvider>
    </LanguageProvider>
  );
}

function renderChain() {
  const nativeBridge = createMockBridge({ delayMs: 0 });
  const rendered = render(
    <Providers nativeBridge={nativeBridge}>
      <ChainedScreen
        prompt="a cat riding a skateboard"
        baseUrl={null}
        nativeBridge={nativeBridge}
        highlightedJobId={null}
        onJobSubmitted={() => {}}
        controlLoraNames={new Set()}
      />
    </Providers>,
  );
  return { ...rendered, nativeBridge };
}

describe("ChainedScreen — the finishing-pass step control", () => {
  it("renders a step selector whose options are the vTile latent-frame lengths, with fixed approximate seconds", async () => {
    const { container } = renderChain();
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });

    // Copy rework (owner decision, 2026-08-09): the option text is now a fixed
    // "N latent frames (≈S s)" string (pxFromVLatent(vTile)/24fps), not a
    // duration read live off the form's frame rate. vTile=22 -> 169px/24fps
    // ≈ 7.0s; vTile=19 -> 145px/24fps ≈ 6.0s.
    const option7 = screen.getByRole("option", { name: /7\.0/ });
    const option6 = screen.getByRole("option", { name: /6\.0/ });
    expect(option7).toHaveValue("standard");
    expect(option6).toHaveValue("high_resolution");

    const select = option7.closest("select");
    expect(select).not.toBeNull();
    expect(select).toHaveValue("standard");
    expect(container).toBeTruthy();
  });

  it("never shows the words 窓 / タイル to the user", async () => {
    // "latent" is no longer avoided here (owner decision, 2026-08-09): the
    // reworked copy deliberately says "N latent frames" / "潜在Nフレーム" —
    // see `strings.chained.stage2Window`'s JSDoc. Only 窓/タイル stay banned.
    // The 2026-08-12 guide tooltips live in `title` attributes, which are not
    // part of `textContent`; they keep the same ban by hand (`strings.ts`).
    const { container } = renderChain();
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
    const text = container.textContent ?? "";
    expect(text).not.toMatch(/窓/);
    expect(text).not.toMatch(/タイル/);
  });
});

// ── the guides drawn on the width/height sliders (2026-08-12) ───────────────
// The mock backend's own ceiling is 1920x1088, which is EXACTLY the shorter
// step's recommended pair — so these tests also cover the "a guide may sit on
// the very end of the range" case that the closed-interval draw condition
// exists for. Positions in `%` are never asserted: jsdom computes no layout,
// so only the pixel values carried in `data-marker-value` are checkable here
// (the visual alignment is an owner gate on real hardware).
describe("ChainedScreen — the comfortable-resolution guides", () => {
  /** `{comfort?: px, limit?: px}` for the width (0) or height (1) slider. */
  function guidesOf(container: HTMLElement, axis: 0 | 1): Record<string, number> {
    // Queried through the DOM rather than by role: `SizeFields` wraps a range
    // AND a number input in one `<label>`, so both live values end up in every
    // accessible name (`ChainedScreen.referenceVideo.test.tsx` has the same
    // note on its own helper).
    const field = container.querySelectorAll<HTMLElement>(".size-field-stack .field")[axis]!;
    return Object.fromEntries(
      Array.from(field.querySelectorAll<HTMLElement>("[data-size-marker]")).map((el) => [
        el.dataset.sizeMarker,
        Number(el.dataset.markerValue),
      ]),
    );
  }

  function sliderOf(container: HTMLElement, axis: 0 | 1): HTMLInputElement {
    const field = container.querySelectorAll<HTMLElement>(".size-field-stack .field")[axis]!;
    return field.querySelector<HTMLInputElement>('input[type="range"]')!;
  }

  function selectWindow(value: string) {
    const select = screen.getByRole("option", { name: /7\.0/ }).closest("select")!;
    fireEvent.change(select, { target: { value } });
  }

  it("marks the recommended pair — 1792 wide, 1024 tall — on the default step", async () => {
    const { container } = renderChain();
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });

    expect(guidesOf(container, 0)).toEqual({ comfort: 1792 });
    expect(guidesOf(container, 1)).toEqual({ comfort: 1024 });
  });

  it("moves the pair to 1920x1088 on the shorter step — both sitting on the slider's own ceiling", async () => {
    const { container } = renderChain();
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });

    selectWindow("high_resolution");
    await waitFor(() => expect(guidesOf(container, 0)).toEqual({ comfort: 1920 }));
    expect(guidesOf(container, 1)).toEqual({ comfort: 1088 });
    expect(sliderOf(container, 0).max).toBe("1920");
    expect(sliderOf(container, 1).max).toBe("1088");
  });

  it("adds the red ceiling to the OTHER slider once the width passes its guide", async () => {
    const { container } = renderChain();
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });

    fireEvent.change(sliderOf(container, 0), { target: { value: "1920" } });
    // Height may go up to 960 while the width stays at 1920.
    await waitFor(() => expect(guidesOf(container, 1)).toEqual({ comfort: 1024, limit: 960 }));
    // ...and the width itself gains nothing red: the guide it passed is its own.
    expect(guidesOf(container, 0)).toEqual({ comfort: 1792 });
  });
});

describe("ChainedScreen — the over-budget advisory", () => {
  function setSize(container: HTMLElement, width: number, height: number) {
    const fields = container.querySelectorAll<HTMLElement>(".size-field-stack .field");
    fireEvent.change(fields[0]!.querySelector<HTMLInputElement>('input[type="range"]')!, {
      target: { value: String(width) },
    });
    fireEvent.change(fields[1]!.querySelector<HTMLInputElement>('input[type="range"]')!, {
      target: { value: String(height) },
    });
  }

  it("stays silent at a comfortable resolution", async () => {
    renderChain();
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
    expect(screen.queryByText(en.chained.stage2Window.overBudgetWarning)).not.toBeInTheDocument();
  });

  it("states the slowdown AND the shorter-step nudge on the default step, without blocking Generate", async () => {
    const { container } = renderChain();
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });

    // 1920x1088 on the standard step = 44,880 tokens > 40,000.
    setSize(container, 1920, 1088);
    const banner = await screen.findByText(
      `${en.chained.stage2Window.overBudgetWarning} ${en.chained.stage2Window.overBudgetShorterWindowHint}`,
    );
    expect(banner).toBeInTheDocument();
    // §1-14: advisory, never a gate.
    expect(screen.getByRole("button", { name: en.chained.generateButton })).toBeEnabled();
  });

  it("never nudges towards the shorter step while the shorter step is already selected", async () => {
    // The nudge is the ONLY part that is step-specific (2026-08-12): the
    // slowdown sentence itself is shown on both steps, which is why it is a
    // separate string. At the mock backend's 1920x1088 ceiling the shorter step
    // is back inside the budget, so here the whole banner goes away — and the
    // nudge must not survive it.
    const { container } = renderChain();
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });
    setSize(container, 1920, 1088);
    await screen.findByText(new RegExp(en.chained.stage2Window.overBudgetShorterWindowHint));

    const select = screen.getByRole("option", { name: /7\.0/ }).closest("select")!;
    fireEvent.change(select, { target: { value: "high_resolution" } });
    await waitFor(() =>
      expect(screen.queryByText(new RegExp(en.chained.stage2Window.overBudgetShorterWindowHint))).not.toBeInTheDocument(),
    );
    expect(screen.queryByText(new RegExp(en.chained.stage2Window.overBudgetWarning))).not.toBeInTheDocument();
  });

  // B-2 (2026-08-12, adversarial review): the shorter step (high_resolution)
  // can still be over budget at a large enough width — this pins the
  // single-sentence rendering (`ChainedScreen.tsx`'s
  // `form.stage2Window === "standard" && ...` guard means the nudge ONLY ever
  // joins the slowdown sentence on the default step). No IN-BOUNDS
  // width/height pair can reach this state on this mock backend: its own
  // ceiling (1920x1088) is exactly high_resolution's recommended point
  // (38,760 tokens, under the 40,000 line — see this file's own "moves the
  // pair to 1920x1088 on the shorter step" test and DEVLOG.md §76's note on
  // the coincidence), so reaching over-budget here requires a width past
  // that ceiling. A free-typed 4096 (height stays the default 768) clears
  // 40,000 tokens at vTile=19 with room to spare (4096x768x19 = 58,368).
  //
  // `fireEvent.change` cannot be used to get the raw 4096 into state: it
  // synthesizes a plain `Event` with no `inputType`, which `paramUtils.ts`'s
  // `isStepEvent` reads as `undefined -> true` (a stepper/slider edit) —
  // exactly the same "snap" path a spinner click takes — so the value would
  // be silently rounded/CLAMPED to the 1920 ceiling by `useChainForm`'s own
  // snapping before ever reaching the over-budget check. `userEvent.type`
  // fires real keystrokes carrying `inputType: "insertText"`, so
  // `isStepEvent` correctly reads it as free typing and passes 4096 through
  // untouched (`setWidth`'s `snap=false` branch, verbatim).
  //
  // Going past the server's own `max_width` this way also (correctly) trips
  // the UNRELATED off-grid validity gate (`dimensionsOnGrid` in
  // `useChainForm.ts`, the same one Create's width/height free-typing gate
  // mirrors) and disables Generate — that is expected and is not what this
  // test is about. The point being pinned is that the budget ADVISORY itself
  // never contributes a validity reason (already covered, at the hook level,
  // by "is ADVISORY — crossing the budget adds no validity reason of its
  // own" above); this test only adds the one thing that describe block
  // cannot reach through the hook alone — the RENDERED banner text with the
  // shorter step already selected.
  it("shows only the slowdown sentence (no shorter-step nudge) when the shorter step ITSELF is still over budget", async () => {
    const user = userEvent.setup();
    const { container } = renderChain();
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });

    const select = screen.getByRole("option", { name: /7\.0/ }).closest("select")!;
    fireEvent.change(select, { target: { value: "high_resolution" } });
    await screen.findByRole("option", { name: /6\.0/ });

    const widthField = container.querySelectorAll<HTMLElement>(".size-field-stack .field")[0]!;
    const widthNumberInput = widthField.querySelector<HTMLInputElement>('input[type="number"]')!;
    await user.clear(widthNumberInput);
    await user.type(widthNumberInput, "4096");

    // 4096 x 768 x 19 latent frames = 58,368 tokens > 40,000 — still over
    // budget even on the shorter step.
    const banner = await screen.findByText(en.chained.stage2Window.overBudgetWarning);
    expect(banner).toBeInTheDocument();
    expect(banner.textContent).not.toContain(en.chained.stage2Window.overBudgetShorterWindowHint);
  });
});
