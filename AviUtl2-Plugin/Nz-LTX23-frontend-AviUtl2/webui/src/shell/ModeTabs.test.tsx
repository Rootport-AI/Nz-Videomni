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

function renderTabs(onChange: (mode: "single" | "chained" | "edit" | "inventory") => void) {
  return render(
    <LanguageProvider>
      <ModeTabs mode="single" onChange={onChange} />
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
});
