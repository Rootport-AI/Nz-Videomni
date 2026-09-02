import { describe, expect, it } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";

// All three mode screens are always mounted now (only the active one is
// visible), so labels/text that exist on more than one screen — e.g. the
// "Presets" control on both Create and Chain — appear twice in the DOM.
// `getByRole("tabpanel")` returns the single *visible* panel (hidden subtrees
// are skipped by role queries), so `within(panel())` scopes such ambiguous
// text/label/display-value queries to the screen actually on screen.
const panel = () => screen.getByRole("tabpanel");

// jsdom has no `window.chrome.webview`, so `src/bridge/index.ts` selects the
// dev mock bridge automatically — these tests exercise the whole M2 Create
// screen wired to that mock end-to-end, on top of the unit-level tests in
// `src/bridge/`, `src/api/` and `src/modes/single/`.

describe("App / Create screen", () => {
  it("shows the connecting badge, then the online badge once the mock bridge answers ping+status", async () => {
    render(<App />);

    expect(screen.getByText(/connecting/i)).toBeInTheDocument();

    expect(await screen.findByText(/server online/i, {}, { timeout: 5_000 })).toBeInTheDocument();
  });

  it("renders the Create form seeded from the mock backend's /config defaults", async () => {
    render(<App />);

    // The prompt bar (task brief §11: shared, above the mode tabs) renders
    // immediately; the form below it waits on `GET /config` before showing
    // presets/generate, so wait for that separately rather than assuming
    // one implies the other.
    await waitFor(
      () => {
        expect(screen.getByPlaceholderText(/describe the video/i)).toBeInTheDocument();
      },
      { timeout: 5_000 },
    );

    expect(await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 })).toBeInTheDocument();
    // U2: presets moved from a tag-button row to a persistent `<select>`.
    expect(screen.getByRole("option", { name: /smoke_test/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /get size from aviutl2/i })).toBeInTheDocument();
  });

  it(
    "runs the full T2V flow: prompt -> generate -> job appears in the ledger -> completes -> insert",
    async () => {
      const user = userEvent.setup();
      render(<App />);

      const promptBox = await screen.findByPlaceholderText(/describe the video/i, {}, { timeout: 5_000 });
      await user.type(promptBox, "a cat riding a skateboard");

      // Use the smallest preset so the (fixed) mock progression finishes
      // quickly. U2: presets are now a persistent `<select>` (`PresetDropdown`).
      // Chain also has a "Presets" label (both screens are mounted), so scope
      // to the visible Create panel.
      const presetSelect = await within(panel()).findByLabelText(/presets/i, {}, { timeout: 5_000 });
      await user.selectOptions(presetSelect, "smoke_test");

      const generateButton = screen.getByRole("button", { name: /^generate$/i });
      await user.click(generateButton);

      // The button reads "Generating…" while the submit is in flight.
      expect(await screen.findByText(/generating/i)).toBeInTheDocument();

      // U-R1: completion/insert now surface through the job ledger's `JobCard`
      // (`strings.jobs.insert`/`inserted`), fed by the 2s `GET /jobs` poll —
      // the old inline `GenerationPanel` "Insert into timeline" is gone.
      const insertButton = await screen.findByRole("button", { name: /^insert$/i }, { timeout: 15_000 });
      await user.click(insertButton);

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /^inserted$/i })).toBeInTheDocument();
      });
    },
    25_000,
  );

  it(
    "preserves an edited Create form value across a tab switch to Chain and back (state-persistence acceptance)",
    async () => {
      // The whole point of always-mounting the mode screens: switching tabs no
      // longer unmounts (and resets) the form. Edit Create's width, bounce to
      // Chain and back, and the edit must still be there.
      const user = userEvent.setup();
      render(<App />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // The "Width" <label> wraps a range slider then a number input; the label
      // names the slider (first labelable descendant), so query it by role and
      // set a clean multiple of 64 in one deterministic change — both the
      // slider and its twin number input track the same form state.
      const widthInput = within(panel()).getByRole("slider", { name: /width/i }) as HTMLInputElement;
      fireEvent.change(widthInput, { target: { value: "832" } });
      expect(widthInput.value).toBe("832");

      await user.click(screen.getByRole("tab", { name: /^chained$/i }));
      await screen.findByRole("tab", { name: /^chained$/i });
      await user.click(screen.getByRole("tab", { name: /^single$/i }));
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // Create was never unmounted, so the edited width survives.
      const widthAfter = within(panel()).getByRole("slider", { name: /width/i }) as HTMLInputElement;
      expect(widthAfter.value).toBe("832");
    },
    20_000,
  );
});

// 台帳§3-71/§3-72 (2026-09-02): the fps box only ever holds a whole frame rate.
// Driven through the REAL input with a single `fireEvent.change` — the
// paste-equivalent route, which is how 29.97 actually reaches the field (typing
// it keystroke by keystroke passes through "29." first, which an
// `<input type="number">` reports as "" and the setter lands on 1fps, exactly as
// it did before this change).
describe("App / frame rate is whole numbers only (台帳§3-71/§3-72)", () => {
  const fpsInput = () => within(panel()).getByRole("spinbutton", { name: /fps/i }) as HTMLInputElement;

  it(
    "Create: pasting 29.97 leaves the field on 30",
    async () => {
      render(<App />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      const fps = fpsInput();
      // The bounds come from `paramUtils`' FRAME_RATE_MIN/MAX, the same pair the
      // snap clamps to.
      expect(fps).toHaveAttribute("min", "1");
      expect(fps).toHaveAttribute("max", "60");

      fireEvent.change(fps, { target: { value: "29.97" } });
      expect(fpsInput().value).toBe("30");
    },
    20_000,
  );

  it(
    "Chain: same, in the common parameters section",
    async () => {
      const user = userEvent.setup();
      render(<App />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      await user.click(screen.getByRole("tab", { name: /^chained$/i }));
      await waitFor(() =>
        expect(screen.getByRole("tab", { name: /^chained$/i })).toHaveAttribute("aria-selected", "true"),
      );

      const fps = fpsInput();
      expect(fps.value).toBe("24");
      fireEvent.change(fps, { target: { value: "29.97" } });
      expect(fpsInput().value).toBe("30");
    },
    20_000,
  );
});
