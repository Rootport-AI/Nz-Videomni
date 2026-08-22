import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "../i18n/LanguageContext";
import { ModeTabs } from "./ModeTabs";

// Phase 6 added Toolbox and Edit as disabled mock tabs (visual placeholders
// only, no `AppMode` value behind them). 2026-08-09: Edit was promoted to a
// real tab, so **Toolbox is the only remaining mock** — see `ModeTabs.tsx`'s
// `MockTabId`/`ModeTabSpec`. This file covers the tab order, which tabs are
// disabled, and that clicking the disabled one never calls `onChange`.

type Mode = "single" | "chained" | "edit" | "inventory";

function renderTabs(onChange: (mode: Mode) => void, disabledModes?: readonly Mode[]) {
  return render(
    <LanguageProvider>
      <ModeTabs mode="single" onChange={onChange} {...(disabledModes ? { disabledModes } : {})} />
    </LanguageProvider>,
  );
}

describe("ModeTabs", () => {
  it("renders 5 tabs in order Toolbox/Single/Chained/Edit/Inventory", () => {
    renderTabs(vi.fn());
    const tabs = screen.getAllByRole("tab");
    expect(tabs).toHaveLength(5);
    expect(tabs.map((tab) => tab.textContent)).toEqual(["Toolbox", "Single", "Chained", "Edit", "Inventory"]);
  });

  it("disables the Toolbox tab only", () => {
    renderTabs(vi.fn());
    expect(screen.getByRole("tab", { name: "Toolbox" })).toBeDisabled();
  });

  it("leaves Single/Chained/Edit/Inventory enabled", () => {
    renderTabs(vi.fn());
    expect(screen.getByRole("tab", { name: "Single" })).not.toBeDisabled();
    expect(screen.getByRole("tab", { name: "Chained" })).not.toBeDisabled();
    expect(screen.getByRole("tab", { name: "Edit" })).not.toBeDisabled();
    expect(screen.getByRole("tab", { name: "Inventory" })).not.toBeDisabled();
  });

  it("does not call onChange when the disabled Toolbox tab is clicked", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderTabs(onChange);
    await user.click(screen.getByRole("tab", { name: "Toolbox" }));
    expect(onChange).not.toHaveBeenCalled();
  });

  it("calls onChange with the mode id when an enabled tab is clicked", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderTabs(onChange);
    await user.click(screen.getByRole("tab", { name: "Chained" }));
    expect(onChange).toHaveBeenCalledWith("chained");
  });

  it("calls onChange('edit') when the newly-enabled Edit tab is clicked", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderTabs(onChange);
    await user.click(screen.getByRole("tab", { name: "Edit" }));
    expect(onChange).toHaveBeenCalledWith("edit");
  });

  // §3-98 P5: a second reason a tab can be greyed — the loaded base model's
  // engine cannot run it. `ModeTabs` knows nothing about engines; it is handed
  // a list of mode ids (`shell/useBaseModels.ts`'s `disabledModesFor`).

  it("greys the modes it is told the base model cannot run", () => {
    renderTabs(vi.fn(), ["chained", "edit"]);
    expect(screen.getByRole("tab", { name: "Chained" })).toBeDisabled();
    expect(screen.getByRole("tab", { name: "Edit" })).toBeDisabled();
    // Single is the baseline every engine serves, and Inventory issues no
    // generation at all — neither is ever disabled this way.
    expect(screen.getByRole("tab", { name: "Single" })).not.toBeDisabled();
    expect(screen.getByRole("tab", { name: "Inventory" })).not.toBeDisabled();
  });

  it("does not call onChange when a base-model-disabled tab is clicked", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderTabs(onChange, ["chained"]);
    await user.click(screen.getByRole("tab", { name: "Chained" }));
    expect(onChange).not.toHaveBeenCalled();
  });

  it("explains WHY a base-model-disabled tab is greyed, but leaves the mock tab alone", () => {
    renderTabs(vi.fn(), ["chained"]);
    expect(screen.getByRole("tab", { name: "Chained" }).getAttribute("title")).toContain(
      "base model",
    );
    // The Toolbox mock is "not built yet", which needs no sentence — and
    // borrowing the base-model wording for it would be a lie.
    expect(screen.getByRole("tab", { name: "Toolbox" })).not.toHaveAttribute("title");
  });

  it("an omitted disabledModes leaves everything exactly as before", () => {
    renderTabs(vi.fn());
    for (const name of ["Single", "Chained", "Edit", "Inventory"]) {
      expect(screen.getByRole("tab", { name })).not.toBeDisabled();
    }
    expect(screen.getByRole("tab", { name: "Toolbox" })).toBeDisabled();
  });
});
