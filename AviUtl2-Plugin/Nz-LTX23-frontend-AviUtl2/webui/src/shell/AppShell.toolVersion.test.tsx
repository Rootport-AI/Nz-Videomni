import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { createMockBridge } from "../bridge/mockBridge";
import { AppShell } from "./AppShell";

// Tool-version header dropdown (2026-08-19, mock, owner-directed): replaces
// the old static "Nz-LTX23" `<h1>` title with a `<select>` offering
// "LTX 2.3"/"LTX 2.5". There is no real 2.5 model behind this yet — picking it
// must silently snap back to "LTX 2.3" (no toast, no warning), and the value
// is plain component state (no persistence across remounts).

async function renderAppAndWaitForCreate() {
  const bridge = createMockBridge({ delayMs: 0 });
  const utils = render(<AppShell nativeBridge={bridge} />);
  await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
  return utils;
}

describe("AppShell — tool-version header dropdown (mock)", () => {
  it("defaults to LTX 2.3 and no longer shows the old static Nz-LTX23 title", async () => {
    await renderAppAndWaitForCreate();

    const select = screen.getByRole("combobox", { name: /tool version/i }) as HTMLSelectElement;
    expect(select.value).toBe("LTX 2.3");
    expect(screen.queryByText("Nz-LTX23")).not.toBeInTheDocument();
  });

  it("silently reverts to LTX 2.3 after selecting LTX 2.5 — no toast/warning", async () => {
    const user = userEvent.setup();
    const { container } = await renderAppAndWaitForCreate();

    const select = screen.getByRole("combobox", { name: /tool version/i }) as HTMLSelectElement;
    await user.selectOptions(select, "LTX 2.5");

    expect(select.value).toBe("LTX 2.3");
    // No toast/warning is raised by the mock reversion.
    expect(container.querySelector(".toast-stack .toast-message")).not.toBeInTheDocument();
  });
});
