import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { createMockBridge } from "../bridge/mockBridge";
import { AppShell } from "./AppShell";

// IC-LoRA UI redesign (2026-07-17, 第5波): integration coverage for the part
// of the redesign that can only be observed at the `AppShell` level — the
// control-LoRA selection state living ABOVE Create/Chain (so it survives a
// tab switch), the auto-migration effect that moves a hand-typed
// `<lora:name:strength>` control tag into the panel, and the
// selection-preserving dropdown that was the owner's original complaint
// ("選択後にplaceholderへ戻り、有効になったことに気づきにくい"). Unit-level
// coverage for the pure merge/gate logic lives in
// `lora/controlLoras.test.ts` / `modes/single/useGenerationForm.test.ts` /
// `modes/chained/useChainForm.test.ts` — this file only covers the wiring that
// requires the whole shell (state ownership, effect timing, cross-mode
// persistence).
//
// The mock bridge's default `GET /loras` fixture (`bridge/mockBridge.ts`)
// already includes five `kind: "control"` entries (`canny-control`,
// `pose-control`, `depth-control`, `deblur`, `in-outpainting`), so no fixture
// override is needed for any test here. The "lists every control adapter"
// test below asserts on only FOUR of those five — `in-outpainting` is
// deliberately excluded from this (Create's) dropdown by
// `lora/controlLoras.ts`'s `UI_HIDDEN_CONTROL_LORA_NAMES` (T2, 2026-08-11: it
// has its own dedicated place, the Edit tab's Outpainting panel), so that
// test doubles as this exclusion's regression guard.

async function renderAppAndWaitForCreate() {
  const bridge = createMockBridge({ delayMs: 0 });
  render(<AppShell nativeBridge={bridge} />);
  await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
  return bridge;
}

/** Opens the "Reference video (IC-LoRA)" `<details>` accordion — its content
 * (the dropdown, the strength slider, the warnings) isn't exposed to
 * role-based queries while collapsed, same as any real browser.
 *
 * Y1: scoped to the accordion `<summary>` specifically — once a control LoRA is
 * selected the form is in IC-LoRA mode, so the Batch section also renders a "A
 * reference video (IC-LoRA) is active…" `<p>` banner that otherwise matches the
 * same text (a plain `getByText` then finds two elements).
 *
 * W3 (2026-08-11): Chain's own reference-video accordion now uses the exact
 * same summary wording (`chained.referenceVideo.heading`, deliberately
 * matched to Single's), and every screen is always mounted
 * (`App.test.tsx`'s doc comment) — so a bare `screen.getByText` would find
 * TWO `<summary>`s once Chain exists in the tree. Scoped to
 * `getByRole("tabpanel")`, which (per that same convention) returns only the
 * single *visible* panel — Single's, since every call site here runs while
 * that tab is active. */
async function openReferenceVideoAccordion(user: ReturnType<typeof userEvent.setup>) {
  const panel = screen.getByRole("tabpanel");
  await user.click(within(panel).getByText(/reference video \(ic-lora\)/i, { selector: "summary" }));
}

describe("AppShell — IC-LoRA control-LoRA panel (第5波)", () => {
  // Guards the `MOCK_LORAS` <-> `MOCK_CONFIG_BODY.model.ic_loras` parity that
  // `bridge/mockBridge.ts` documents: `depth-control`/`deblur` were registered
  // in the mock `/config` long before they reached the `/loras` fixture, so the
  // dropdown listed four names while `/config` was the source and silently fell
  // back to two the moment `/loras` resolved. Asserting the post-`/loras` set
  // is what catches that direction.
  it("lists every control adapter the mock backend registers", async () => {
    const user = userEvent.setup();
    await renderAppAndWaitForCreate();
    await openReferenceVideoAccordion(user);

    const select = await screen.findByRole("combobox", { name: /^control lora$/i });
    const names = Array.from((select as HTMLSelectElement).options)
      .map((option) => option.value)
      .filter((value) => value !== "");
    expect(names.sort()).toEqual(["canny-control", "deblur", "depth-control", "pose-control"]);
  });

  it(
    "the dropdown selection is preserved (not reset to a placeholder) after choosing a name",
    async () => {
      const user = userEvent.setup();
      await renderAppAndWaitForCreate();
      await openReferenceVideoAccordion(user);

      const select = await screen.findByRole("combobox", { name: /^control lora$/i });
      await user.selectOptions(select, "canny-control");

      expect((select as HTMLSelectElement).value).toBe("canny-control");
      // The old one-shot dropdown reset itself to "" (its placeholder) the
      // instant a name was picked — this is the regression guard for that.
      expect((select as HTMLSelectElement).value).not.toBe("");

      // The weight slider appears once a selection exists, seeded at 1.0.
      const slider = await screen.findByRole("slider", { name: /control lora strength/i });
      expect((slider as HTMLInputElement).valueAsNumber).toBe(1.0);
    },
    15_000,
  );

  it(
    "selecting a name never writes a prompt tag — the prompt stays untouched",
    async () => {
      const user = userEvent.setup();
      await renderAppAndWaitForCreate();
      await openReferenceVideoAccordion(user);

      const select = await screen.findByRole("combobox", { name: /^control lora$/i });
      await user.selectOptions(select, "canny-control");

      const promptBox = screen.getByPlaceholderText(/describe the video/i) as HTMLTextAreaElement;
      expect(promptBox.value).toBe("");
      expect(screen.queryByText(/<lora:canny-control/i)).not.toBeInTheDocument();
    },
    15_000,
  );

  it(
    "choosing \"none\" clears the selection and hides the strength slider",
    async () => {
      const user = userEvent.setup();
      await renderAppAndWaitForCreate();
      await openReferenceVideoAccordion(user);

      const select = await screen.findByRole("combobox", { name: /^control lora$/i });
      await user.selectOptions(select, "canny-control");
      expect(await screen.findByRole("slider", { name: /control lora strength/i })).toBeInTheDocument();

      await user.selectOptions(select, "");

      expect((select as HTMLSelectElement).value).toBe("");
      expect(screen.queryByRole("slider", { name: /control lora strength/i })).not.toBeInTheDocument();
    },
    15_000,
  );

  it(
    "a hand-typed control-LoRA tag auto-migrates into the panel, strips the tag from the prompt, and toasts",
    async () => {
      const user = userEvent.setup();
      await renderAppAndWaitForCreate();

      const promptBox = screen.getByPlaceholderText(/describe the video/i);
      await user.type(promptBox, "a cat <lora:canny-control:0.8>");

      await screen.findByText(/moved the.*canny-control.*panel above/i, {}, { timeout: 5_000 });

      expect((promptBox as HTMLTextAreaElement).value).toBe("a cat");

      await openReferenceVideoAccordion(user);
      const select = await screen.findByRole("combobox", { name: /^control lora$/i });
      expect((select as HTMLSelectElement).value).toBe("canny-control");
      const slider = await screen.findByRole("slider", { name: /control lora strength/i });
      expect((slider as HTMLInputElement).valueAsNumber).toBeCloseTo(0.8);
    },
    15_000,
  );

  it(
    "having multiple control-LoRA tags appear at once (e.g. a paste) keeps only the LAST one and shows the 'multiple' toast",
    async () => {
      // A single paste (one onChange with the full text) — NOT `user.type`,
      // which simulates real per-keystroke typing: the FIRST tag would
      // already be complete (and migrated by the effect) before the second
      // tag's closing `>` is even typed, so it'd never see both at once.
      // A paste (or any programmatic prompt replace) is the realistic way
      // two tags land in the prompt simultaneously.
      const user = userEvent.setup();
      await renderAppAndWaitForCreate();

      const promptBox = screen.getByPlaceholderText(/describe the video/i);
      await user.click(promptBox);
      await user.paste("<lora:canny-control:1.0> a cat <lora:pose-control:0.5> riding a skateboard");

      await screen.findByText(/multiple control lora tags/i, {}, { timeout: 5_000 });

      await openReferenceVideoAccordion(user);
      const select = await screen.findByRole("combobox", { name: /^control lora$/i });
      expect((select as HTMLSelectElement).value).toBe("pose-control");
    },
    15_000,
  );

  it(
    "the selection survives a tab switch away from Create and back",
    async () => {
      const user = userEvent.setup();
      await renderAppAndWaitForCreate();
      await openReferenceVideoAccordion(user);

      const select = await screen.findByRole("combobox", { name: /^control lora$/i });
      await user.selectOptions(select, "canny-control");

      await user.click(screen.getByRole("tab", { name: /^chained$/i }));
      await screen.findByRole("tab", { name: /^chained$/i });
      await user.click(screen.getByRole("tab", { name: /^single$/i }));
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // State-persistence is exactly what this test checks: Create is no longer
      // unmounted on a tab switch, so the accordion stays OPEN and the
      // selection is still in place — no need to re-open anything. Re-opening
      // here would in fact toggle the still-open <details> shut and hide the
      // combobox, so we assert directly on the persisted control.
      const selectAfter = await screen.findByRole("combobox", { name: /^control lora$/i });
      expect((selectAfter as HTMLSelectElement).value).toBe("canny-control");
    },
    20_000,
  );

  it(
    "does NOT auto-migrate a control-LoRA tag typed on the Chain tab — it asks for a reference video and blocks Generate",
    async () => {
      const user = userEvent.setup();
      await renderAppAndWaitForCreate();

      await user.click(screen.getByRole("tab", { name: /^chained$/i }));
      const promptBox = await screen.findByPlaceholderText(/describe the video/i);
      await user.type(promptBox, " <lora:canny-control:1.0>");

      // §1-15: a control tag is no longer forbidden here — Chain accepts one
      // long reference video for the whole chain now. What it still cannot do
      // is run a control adapter WITHOUT that reference, so the Generate-reasons
      // note asks for one (Create's own `attachReferenceVideo` line, reused).
      await waitFor(
        () => expect(screen.getAllByText(/attach a reference video/i).length).toBeGreaterThanOrEqual(1),
        { timeout: 5_000 },
      );
      // The old "not supported in Chain" copy is gone for good.
      expect(screen.queryByText(/control lora tags aren't supported in chain/i)).toBeNull();
      // The tag is still in the prompt (no migration happened — AppShell's
      // auto-migration effect is `mode === "single"` only) and the Generate
      // button is disabled by the gate.
      expect((promptBox as HTMLTextAreaElement).value).toContain("<lora:canny-control:1.0>");
      expect(screen.getByRole("button", { name: /^generate$/i })).toBeDisabled();
    },
    15_000,
  );
});
