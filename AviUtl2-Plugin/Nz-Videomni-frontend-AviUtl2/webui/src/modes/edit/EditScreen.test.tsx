import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "../../i18n/LanguageContext";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import { EditScreen } from "./EditScreen";

// 2026-08-09: the Edit tab was promoted from a disabled mock to a real mode,
// and gained a sub-tab row of its own (Retake / Outpainting / Inpainting).
// Inpainting is the mock here — `<button disabled>` with no `onClick` at all,
// exactly like `shell/ModeTabs.tsx`'s Toolbox. These tests cover the SKELETON
// only: order, the disabled one, the switch, and the deliberate ABSENCE of
// `role="tabpanel"` (7 existing app-level test files take
// `getByRole("tabpanel")` in the singular). The Outpainting panel's own
// behaviour lives in `OutpaintingPanel.test.tsx`.
//
// Outpainting (2026-08-09) turned that panel from a placeholder into a real
// form, which brought two of its dependencies into this file's render tree:
// the seed field's ♻ button reads the job ledger through `useJobsContext`
// (stubbed below, exactly as `CommonGenerationFields.test.tsx` and the
// GenerationForm tests do, so no `JobsProvider` is needed), and the form hook
// fetches `GET /loras` through the app-wide mock bridge, which resolves
// harmlessly and is guarded by `useLoras`'s own mounted-ref.
//
// The screen now also mounts the shared `jobs/JobLedger`, which reads FOUR more
// members off the same context — hence the fuller stub. An empty `jobs` array
// makes the ledger render its "no jobs yet" line, so the cancel/delete halves
// are never actually invoked; they exist so the destructuring finds them.
vi.mock("../../jobs/JobsContext", () => ({
  useJobsContext: () => ({
    jobs: [],
    hasActiveJob: false,
    cancellingIds: new Set<string>(),
    deletingIds: new Set<string>(),
    cancelJob: () => Promise.resolve(),
    deleteJob: () => Promise.resolve(),
  }),
}));

function renderEdit(initialIntent?: GenerationPrefill, prompt?: string) {
  return render(
    <LanguageProvider>
      <EditScreen initialIntent={initialIntent} prompt={prompt} />
    </LanguageProvider>,
  );
}

/** W0: a routed right-click payload, exactly as `AppShell` builds it (the
 * screen only reads `intent`, but the whole `GenerationPrefill` is what W2/W3's
 * panels will consume, so the fixture is the real shape). */
function prefill(intent: string): GenerationPrefill {
  return {
    intent,
    targetMode: "edit",
    selection: {
      hasRange: true,
      rangeStart: 48,
      rangeEnd: 96,
      selected: [
        {
          layer: 1, frameStart: 0, frameEnd: 120, effectName: "動画ファイル",
          filePath: "C:\\v\\a.mp4", objectName: "a", textContent: null,
          mediaWidth: 1920, mediaHeight: 1080, mediaDurationSec: 5,
        },
      ],
      cursorFrame: 48,
      cursorLayer: 1,
      rate: 30,
      scale: 1,
      sampleRate: 44100,
    },
  };
}

describe("EditScreen sub-tabs", () => {
  it("renders 3 sub-tabs in order Retake/Outpainting/Inpainting", () => {
    renderEdit();
    const tabs = screen.getAllByRole("tab");
    expect(tabs).toHaveLength(3);
    expect(tabs.map((tab) => tab.textContent)).toEqual(["Retake", "Outpainting", "Inpainting"]);
  });

  it("disables the Inpainting sub-tab only", () => {
    renderEdit();
    expect(screen.getByRole("tab", { name: "Inpainting" })).toBeDisabled();
    expect(screen.getByRole("tab", { name: "Retake" })).not.toBeDisabled();
    expect(screen.getByRole("tab", { name: "Outpainting" })).not.toBeDisabled();
  });

  // A `hidden` panel is out of the accessibility tree, so role queries need
  // `{ hidden: true }` to reach it at all — `toBeVisible()` is what actually
  // asserts which of the two is on screen.
  const heading = (name: RegExp) => screen.getByRole("heading", { name, hidden: true });

  it("starts on Retake, with the Outpainting panel hidden", () => {
    renderEdit();
    expect(screen.getByRole("tab", { name: "Retake" })).toHaveAttribute("aria-selected", "true");
    expect(heading(/^retake/i)).toBeVisible();
    expect(heading(/^outpainting/i)).not.toBeVisible();
  });

  it("switches to the Outpainting panel when its sub-tab is clicked", async () => {
    const user = userEvent.setup();
    renderEdit();
    await user.click(screen.getByRole("tab", { name: "Outpainting" }));
    expect(heading(/^outpainting/i)).toBeVisible();
    expect(heading(/^retake/i)).not.toBeVisible();
    expect(screen.getByRole("tab", { name: "Outpainting" })).toHaveAttribute("aria-selected", "true");
  });

  it("does nothing when the disabled Inpainting sub-tab is clicked", async () => {
    const user = userEvent.setup();
    renderEdit();
    await user.click(screen.getByRole("tab", { name: "Inpainting" }));
    // Still on Retake — no panel of its own was revealed, and no sub-tab moved.
    expect(heading(/^retake/i)).toBeVisible();
    expect(screen.getByRole("tab", { name: "Retake" })).toHaveAttribute("aria-selected", "true");
    expect(screen.queryByRole("heading", { name: /^inpainting/i, hidden: true })).not.toBeInTheDocument();
  });

  it("does NOT mark its sub-panels with role=tabpanel", () => {
    // Load-bearing: the app-level tests scope queries with a singular
    // `getByRole("tabpanel")` (the one visible mode screen). A second visible
    // tabpanel inside Edit would make every one of those queries ambiguous.
    renderEdit();
    expect(screen.queryAllByRole("tabpanel", { hidden: true })).toHaveLength(0);
  });

  // W0 (2026-08-09): the Edit tab became a right-click destination, so a routed
  // intent pre-selects the sub-tab. Consumed ONCE in the lazy `useState`
  // initializer — `AppShell` bumps `remountTokens.edit` per route, so a fresh
  // right-click always arrives as a fresh mount.
  it("auto-selects the Outpainting sub-tab for an 'outpaint' intent", () => {
    renderEdit(prefill("outpaint"));
    expect(screen.getByRole("tab", { name: "Outpainting" })).toHaveAttribute("aria-selected", "true");
    expect(heading(/^outpainting/i)).toBeVisible();
    expect(heading(/^retake/i)).not.toBeVisible();
  });

  it("auto-selects the Retake sub-tab for a 'retake' intent", () => {
    renderEdit(prefill("retake"));
    expect(screen.getByRole("tab", { name: "Retake" })).toHaveAttribute("aria-selected", "true");
    expect(heading(/^retake/i)).toBeVisible();
    expect(heading(/^outpainting/i)).not.toBeVisible();
  });

  it("falls back to Retake for an unrecognized intent", () => {
    // Defensive: an intent this screen does not know (e.g. a future Inpainting
    // route landing before its panel exists) must not blank the screen.
    renderEdit(prefill("totally-unknown"));
    expect(screen.getByRole("tab", { name: "Retake" })).toHaveAttribute("aria-selected", "true");
    expect(heading(/^retake/i)).toBeVisible();
  });

  it("lets the user override the auto-selected sub-tab", async () => {
    // The intent SEEDS the selection; it does not pin it (a lazy initializer,
    // never a prop-following effect), so a manual click still wins.
    const user = userEvent.setup();
    renderEdit(prefill("outpaint"));
    expect(heading(/^outpainting/i)).toBeVisible();
    await user.click(screen.getByRole("tab", { name: "Retake" }));
    expect(heading(/^retake/i)).toBeVisible();
    expect(heading(/^outpainting/i)).not.toBeVisible();
  });

  // 2026-08-09: Edit gained a job ledger of its own — the SAME `jobs/JobLedger`
  // Create/Chain mount, fed by the same provider. Before this, the panel's copy
  // told the user to go and look at the Single screen, and `AppShell`'s toast
  // click had to switch tabs to find a ledger to highlight in.
  it("mounts the shared job ledger below the panels", () => {
    renderEdit();
    expect(screen.getByRole("heading", { name: /^jobs/i })).toBeVisible();
    expect(screen.getByText(/no jobs yet/i)).toBeVisible();
  });

  it("keeps the ledger visible on either sub-tab", async () => {
    // It sits OUTSIDE the sub-tab switch, so it is not hidden with a panel.
    const user = userEvent.setup();
    renderEdit();
    await user.click(screen.getByRole("tab", { name: "Outpainting" }));
    expect(screen.getByRole("heading", { name: /^jobs/i })).toBeVisible();
    await user.click(screen.getByRole("tab", { name: "Retake" }));
    expect(screen.getByRole("heading", { name: /^jobs/i })).toBeVisible();
  });

  // The shared prompt (`AppShell`'s `PromptBar`) reaches the Outpainting panel
  // as a prop — the panel has no prompt box of its own to fall back on, so a
  // broken thread here shows up as a permanent "fill the main prompt" block.
  it("threads the shared prompt down into the Outpainting panel", () => {
    renderEdit(prefill("outpaint"), "a wide city street");
    const note = screen.getByRole("note");
    expect(within(note).queryByText(/fill the main prompt/i)).not.toBeInTheDocument();
    // And no second prompt box was introduced anywhere under Edit.
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("blocks on the empty shared prompt when none was handed down", () => {
    renderEdit(prefill("outpaint"));
    expect(within(screen.getByRole("note")).getByText(/fill the main prompt/i)).toBeInTheDocument();
  });

  // 2カラム化（オーナー指示 第2バッチ、2026-08-09）: the panels used to be one
  // full-width stack with the ledger at the very bottom, where it was
  // effectively invisible. Edit now uses Create/Chain's own `.single-layout` +
  // `.generation-column` pair, so the Generate button and the ledger sit
  // together in the right-hand column and the form keeps the left one. Asserted
  // through the class names because that IS the change — the two columns are
  // exactly the shared classes, not Edit-local copies of them.
  it("puts the Generate button and the job ledger together in the generation column", () => {
    const { container } = renderEdit(prefill("outpaint"));
    const layout = container.querySelector(".single-layout");
    expect(layout).not.toBeNull();

    const column = container.querySelector<HTMLElement>(".generation-column");
    expect(column).not.toBeNull();
    expect(within(column!).getByRole("button", { name: /^generate$/i })).toBeInTheDocument();
    expect(within(column!).getByRole("heading", { name: /^jobs/i })).toBeInTheDocument();

    // …and the form column no longer ends in a Generate button of its own.
    const formColumn = container.querySelector<HTMLElement>(".edit-form-column");
    expect(formColumn).not.toBeNull();
    expect(within(formColumn!).queryByRole("button", { name: /^generate$/i })).not.toBeInTheDocument();
  });

  it("drops the Generate group — but not the ledger — on the Retake sub-tab", async () => {
    // Retake has nothing to generate yet, so only the ledger is left in the
    // right-hand column. The ledger itself sits OUTSIDE the sub-tab switch.
    const user = userEvent.setup();
    const { container } = renderEdit(prefill("outpaint"));
    expect(screen.getByRole("button", { name: /^generate$/i })).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Retake" }));
    expect(screen.queryByRole("button", { name: /^generate$/i })).not.toBeInTheDocument();
    const column = container.querySelector<HTMLElement>(".generation-column");
    expect(within(column!).getByRole("heading", { name: /^jobs/i })).toBeVisible();
  });

  it("keeps the Retake panel mounted while Outpainting is shown", async () => {
    // Both sub-panels stay mounted and are merely `hidden` — the same
    // arrangement `AppShell` uses for the mode screens, so a future panel's
    // form state will survive a sub-tab switch.
    const user = userEvent.setup();
    renderEdit();
    await user.click(screen.getByRole("tab", { name: "Outpainting" }));
    expect(screen.getByRole("heading", { name: /^retake/i, hidden: true })).toBeInTheDocument();
  });
});
