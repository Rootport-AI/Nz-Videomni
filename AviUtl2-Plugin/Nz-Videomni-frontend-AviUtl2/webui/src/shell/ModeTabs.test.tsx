import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "../i18n/LanguageContext";
import { ModeTabs } from "./ModeTabs";

// Phase 6 added Toolbox and Edit as disabled mock tabs (visual placeholders
// only, no `AppMode` value behind them). Edit was promoted to a real tab on
// 2026-08-09 and **Toolbox on 2026-09-11** (§3-54 物体追尾), which retired
// `ModeTabs.tsx`'s `MockTabId`/`ModeTabSpec` union along with it — so this file
// now covers the tab order, the fact that NOTHING is disabled by default, and
// the one remaining reason a tab greys (the loaded base model's engine).

type Mode = "single" | "chained" | "edit" | "inventory" | "toolbox";

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

  // §3-54 (2026-09-11): the assertion this file used to make — "disables the
  // Toolbox tab only" — inverted on the day Toolbox became a real tab. Keeping
  // a test for the inverse matters: nothing should be greyed unless a base
  // model says so.
  it("disables nothing by default", () => {
    renderTabs(vi.fn());
    for (const tab of screen.getAllByRole("tab")) {
      expect(tab).not.toBeDisabled();
    }
  });

  it("calls onChange('toolbox') when the newly-enabled Toolbox tab is clicked", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderTabs(onChange);
    await user.click(screen.getByRole("tab", { name: "Toolbox" }));
    expect(onChange).toHaveBeenCalledWith("toolbox");
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
  // a list of mode ids (`shell/featureScope.ts`'s `disabledModesFor`).

  it("greys the modes it is told the base model cannot run", () => {
    renderTabs(vi.fn(), ["chained", "edit"]);
    expect(screen.getByRole("tab", { name: "Chained" })).toBeDisabled();
    expect(screen.getByRole("tab", { name: "Edit" })).toBeDisabled();
    // Single is the baseline every engine serves; Inventory only browses
    // finished files and Toolbox runs a CPU tool that is not a generation at
    // all — none of the three is ever disabled this way
    // (`shell/featureScope.ts` gives them no row).
    expect(screen.getByRole("tab", { name: "Single" })).not.toBeDisabled();
    expect(screen.getByRole("tab", { name: "Inventory" })).not.toBeDisabled();
    expect(screen.getByRole("tab", { name: "Toolbox" })).not.toBeDisabled();
  });

  it("does not call onChange when a base-model-disabled tab is clicked", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderTabs(onChange, ["chained"]);
    await user.click(screen.getByRole("tab", { name: "Chained" }));
    expect(onChange).not.toHaveBeenCalled();
  });

  it("explains WHY a base-model-disabled tab is greyed, and says nothing on the live ones", () => {
    renderTabs(vi.fn(), ["chained"]);
    expect(screen.getByRole("tab", { name: "Chained" }).getAttribute("title")).toContain(
      "base model",
    );
    // A tab that is not greyed has nothing to explain.
    expect(screen.getByRole("tab", { name: "Toolbox" })).not.toHaveAttribute("title");
  });

  it("an omitted disabledModes leaves every tab live", () => {
    renderTabs(vi.fn());
    for (const name of ["Toolbox", "Single", "Chained", "Edit", "Inventory"]) {
      expect(screen.getByRole("tab", { name })).not.toBeDisabled();
    }
  });
});
