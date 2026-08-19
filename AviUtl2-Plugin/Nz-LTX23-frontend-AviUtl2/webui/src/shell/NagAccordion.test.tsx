import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { LanguageProvider } from "../i18n/LanguageContext";
import { NagAccordion } from "./NagAccordion";
import {
  DEFAULT_NEGATIVE_PROMPT,
  NAG_ALPHA_DEFAULT,
  NAG_SCALE_DEFAULT,
  NAG_TAU_DEFAULT,
  NEG_METHOD_DEFAULT,
  VSF_SCALE_DEFAULT,
} from "./nagSettings";
import type { NagSettings } from "./nagSettings";

// `NagAccordion` is pure display, driven entirely by its `nag`/`controls`
// props (D4) — this harness closes the loop with a local
// `useState<NagSettings>` mirroring the update shape `useNagSettings.ts`
// itself uses, so a click on the checkbox/textarea/🔄 button is actually
// reflected back into what's rendered. Same pattern as
// `ReferenceStrengthField.test.tsx`'s `Harness`.

function Harness({ initial }: { initial?: Partial<NagSettings> | undefined }) {
  const [nag, setNag] = useState<NagSettings>({
    text: DEFAULT_NEGATIVE_PROMPT,
    enabled: false,
    scale: NAG_SCALE_DEFAULT,
    tau: NAG_TAU_DEFAULT,
    alpha: NAG_ALPHA_DEFAULT,
    method: NEG_METHOD_DEFAULT,
    vsfScale: VSF_SCALE_DEFAULT,
    ...initial,
  });
  return (
    <NagAccordion
      nag={nag}
      controls={{
        setEnabled: (value) => setNag((prev) => ({ ...prev, enabled: value })),
        setText: (value) => setNag((prev) => ({ ...prev, text: value })),
        setScale: (value) => setNag((prev) => ({ ...prev, scale: value })),
        setTau: (value) => setNag((prev) => ({ ...prev, tau: value })),
        setAlpha: (value) => setNag((prev) => ({ ...prev, alpha: value })),
        setMethod: (value) => setNag((prev) => ({ ...prev, method: value })),
        setVsfScale: (value) => setNag((prev) => ({ ...prev, vsfScale: value })),
        resetParams: () =>
          setNag((prev) => ({
            ...prev,
            scale: NAG_SCALE_DEFAULT,
            tau: NAG_TAU_DEFAULT,
            alpha: NAG_ALPHA_DEFAULT,
            vsfScale: VSF_SCALE_DEFAULT,
          })),
      }}
    />
  );
}

function renderAccordion(initial?: Partial<NagSettings>) {
  return render(
    <LanguageProvider>
      <Harness initial={initial} />
    </LanguageProvider>,
  );
}

function accordionDetails(): HTMLDetailsElement {
  return screen.getByText(/^negative prompt$/i, { selector: "summary" }).closest("details") as HTMLDetailsElement;
}

async function openAccordion(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByText(/^negative prompt$/i, { selector: "summary" }));
}

async function openAdvanced(user: ReturnType<typeof userEvent.setup>) {
  await openAccordion(user);
  await user.click(screen.getByText(/^advanced$/i, { selector: "summary" }));
}

/** A `FloatSliderField`'s number input, found via its own `<label>` — three
 * separate scale/tau/alpha instances render at once, so a bare
 * `getByRole("spinbutton")` would be ambiguous. */
function numberInputFor(fieldLabelText: string): HTMLInputElement {
  const label = screen.getByText(fieldLabelText).closest("label") as HTMLElement;
  return within(label).getByRole("spinbutton") as HTMLInputElement;
}

describe("NagAccordion", () => {
  it("defaults closed", () => {
    renderAccordion();
    expect(accordionDetails().open).toBe(false);
  });

  it("shows the ON badge only once the checkbox is ticked (nothing extra while off)", async () => {
    const user = userEvent.setup();
    renderAccordion();
    expect(screen.queryByText("【🔴ON】")).not.toBeInTheDocument();

    await openAccordion(user);
    await user.click(screen.getByRole("checkbox", { name: /non-cfg negative/i }));

    expect(screen.getByText("【🔴ON】")).toBeInTheDocument();
  });

  it("the negative-prompt textarea is disabled until the checkbox is ticked, then editable", async () => {
    const user = userEvent.setup();
    renderAccordion();
    await openAccordion(user);

    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    expect(textarea).toBeDisabled();

    await user.click(screen.getByRole("checkbox", { name: /non-cfg negative/i }));

    expect(textarea).toBeEnabled();
  });

  it("NAG and VSF are both always-operable radios, with NAG checked by default", async () => {
    const user = userEvent.setup();
    renderAccordion();
    await openAccordion(user);

    const nagRadio = screen.getByRole("radio", { name: /^nag$/i });
    const vsfRadio = screen.getByRole("radio", { name: /^vsf$/i });
    expect(nagRadio).toBeChecked();
    expect(vsfRadio).not.toBeChecked();
    expect(nagRadio).toBeEnabled();
    expect(vsfRadio).toBeEnabled();
  });

  it("clicking the VSF radio switches the checked state and shows the VSF scale slider instead of NAG's 3 sliders", async () => {
    const user = userEvent.setup();
    renderAccordion();
    await openAccordion(user);

    expect(screen.getByText("NAG scale")).toBeInTheDocument();
    expect(screen.getByText(/^advanced$/i)).toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: /^vsf$/i }));

    expect(screen.getByRole("radio", { name: /^vsf$/i })).toBeChecked();
    expect(screen.getByRole("radio", { name: /^nag$/i })).not.toBeChecked();
    expect(screen.getByText("VSF scale")).toBeInTheDocument();
    expect(screen.queryByText("NAG scale")).not.toBeInTheDocument();
    expect(screen.queryByText(/^advanced$/i)).not.toBeInTheDocument();

    // Switching back to NAG restores the 3-slider view.
    await user.click(screen.getByRole("radio", { name: /^nag$/i }));
    expect(screen.getByText("NAG scale")).toBeInTheDocument();
    expect(screen.queryByText("VSF scale")).not.toBeInTheDocument();
  });

  it("the radios stay operable even while the checkbox is OFF, but the VSF slider stays disabled", async () => {
    const user = userEvent.setup();
    renderAccordion();
    await openAccordion(user);

    expect(screen.getByRole("checkbox", { name: /non-cfg negative/i })).not.toBeChecked();

    await user.click(screen.getByRole("radio", { name: /^vsf$/i }));
    expect(screen.getByRole("radio", { name: /^vsf$/i })).toBeChecked();

    const vsfSlider = numberInputFor("VSF scale");
    expect(vsfSlider).toBeDisabled();
  });

  it("🔄 resets scale/tau/alpha to their defaults and leaves the negative-prompt text untouched", async () => {
    const user = userEvent.setup();
    renderAccordion({ enabled: true, text: "custom negative text" });
    await openAdvanced(user);

    fireEvent.change(numberInputFor("NAG scale"), { target: { value: "3" } });
    fireEvent.change(numberInputFor("NAG tau"), { target: { value: "9" } });
    fireEvent.change(numberInputFor("NAG alpha"), { target: { value: "0.9" } });
    expect(numberInputFor("NAG scale").valueAsNumber).toBe(3);

    await user.click(screen.getByRole("button", { name: /reset/i }));

    expect(numberInputFor("NAG scale").valueAsNumber).toBe(NAG_SCALE_DEFAULT);
    expect(numberInputFor("NAG tau").valueAsNumber).toBe(NAG_TAU_DEFAULT);
    expect(numberInputFor("NAG alpha").valueAsNumber).toBe(NAG_ALPHA_DEFAULT);
    expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toBe("custom negative text");
  });

  it("🔄 also resets vsfScale to its default while VSF is the selected method", async () => {
    const user = userEvent.setup();
    renderAccordion({ enabled: true, method: "vsf" });
    await openAccordion(user);

    fireEvent.change(numberInputFor("VSF scale"), { target: { value: "6" } });
    expect(numberInputFor("VSF scale").valueAsNumber).toBe(6);

    await user.click(screen.getByRole("button", { name: /reset/i }));

    expect(numberInputFor("VSF scale").valueAsNumber).toBe(VSF_SCALE_DEFAULT);
  });

  it("the 🔄 reset button lives next to the scale slider, NOT inside the Advanced sub-accordion (2026-07-28 slim-down)", async () => {
    const user = userEvent.setup();
    renderAccordion();
    await openAccordion(user);

    const resetBtn = screen.getByRole("button", { name: /reset/i });
    expect(resetBtn.closest("details.nag-subaccordion")).toBeNull();
    expect(resetBtn).toBeDisabled();

    await user.click(screen.getByRole("checkbox", { name: /non-cfg negative/i }));
    expect(resetBtn).toBeEnabled();
  });

  it("soft disable: unticking the checkbox keeps the text/scale values intact (nothing is cleared)", async () => {
    const user = userEvent.setup();
    renderAccordion();
    await openAdvanced(user);

    await user.click(screen.getByRole("checkbox", { name: /non-cfg negative/i }));
    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: "keep me" } });
    fireEvent.change(numberInputFor("NAG scale"), { target: { value: "7" } });

    // Untick — soft disable (owner decision §UX 5).
    await user.click(screen.getByRole("checkbox", { name: /non-cfg negative/i }));

    expect(textarea).toBeDisabled();
    expect(textarea.value).toBe("keep me");
    expect(numberInputFor("NAG scale").valueAsNumber).toBe(7);
  });
});
