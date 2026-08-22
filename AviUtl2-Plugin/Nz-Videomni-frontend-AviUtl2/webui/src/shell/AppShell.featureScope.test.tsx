import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { createMockBridge } from "../bridge/mockBridge";
import type { MockBridgeOptions } from "../bridge/mockBridge";
import { AppShell } from "./AppShell";

// §3-98 Phase 5 — the WebUI half of the LTX 2.5 v1 feature scope.
//
// The server refuses what its engine cannot run (422 `FEATURE_UNSUPPORTED`)
// and PUBLISHES the same list on `GET /models` so the WebUI can grey the
// controls first. This file covers the greying end-to-end, through the fixture
// server rather than by handing components props — the whole point of the
// design is that the names come from the backend, and a test that supplies
// them itself would pass even if the response were never read.
//
// The fixture models the world after §3-98 ships: `ltx25Install: "full"` puts
// the weights on disk and `supportedBaseModels` lets the engine actually run
// them, which together make LTX 2.5 selectable AND loadable.

const AS_LTX25: MockBridgeOptions = {
  delayMs: 0,
  ltx25Install: "full",
  supportedBaseModels: ["LTX23", "LTX25"],
};

async function renderApp(options: MockBridgeOptions = {}) {
  const bridge = createMockBridge({ delayMs: 0, ...options });
  const utils = render(<AppShell nativeBridge={bridge} />);
  await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
  const select = (await screen.findByRole("combobox", { name: /base model/i })) as HTMLSelectElement;
  await waitFor(() => expect(select.value).toBe("LTX23"));
  return { ...utils, select, bridge };
}

/** Switch the header dropdown to LTX 2.5 and wait for the load to settle. */
async function switchToLtx25(select: HTMLSelectElement) {
  const user = userEvent.setup();
  await user.selectOptions(select, "LTX25");
  await waitFor(() => expect(select.value).toBe("LTX25"));
}

function tab(name: string) {
  return screen.getByRole("tab", { name });
}

describe("AppShell — base-model feature scope", () => {
  it("leaves every tab enabled on a base model that declares no restrictions", async () => {
    await renderApp();

    // The regression that matters most: LTX 2.3 must look exactly as it did
    // before this feature existed.
    for (const name of ["Single", "Chained", "Edit", "Inventory"]) {
      expect(tab(name)).not.toBeDisabled();
    }
  });

  it("greys Chained and Edit once LTX 2.5 is the loaded base model", async () => {
    const { select } = await renderApp(AS_LTX25);

    await switchToLtx25(select);

    await waitFor(() => expect(tab("Chained")).toBeDisabled());
    expect(tab("Edit")).toBeDisabled();
    // Single is the one thing this engine's v1 CAN do, and Inventory only
    // browses finished files.
    expect(tab("Single")).not.toBeDisabled();
    expect(tab("Inventory")).not.toBeDisabled();
  });

  it("switching back to LTX 2.3 restores the tabs", async () => {
    const { select } = await renderApp(AS_LTX25);
    const user = userEvent.setup();

    await switchToLtx25(select);
    await waitFor(() => expect(tab("Chained")).toBeDisabled());

    await user.selectOptions(select, "LTX23");
    await waitFor(() => expect(select.value).toBe("LTX23"));

    await waitFor(() => expect(tab("Chained")).not.toBeDisabled());
    expect(tab("Edit")).not.toBeDisabled();
  });

  it("bounces out of a mode the new base model cannot run", async () => {
    // The case greying alone cannot cover: the user is ALREADY on Chained when
    // they switch. Leaving that panel open behind a disabled tab would let them
    // fill in a form whose every submission comes back 422.
    const { select } = await renderApp(AS_LTX25);
    const user = userEvent.setup();

    await user.click(tab("Chained"));
    await waitFor(() => expect(tab("Chained")).toHaveAttribute("aria-selected", "true"));

    await switchToLtx25(select);

    await waitFor(() => expect(tab("Single")).toHaveAttribute("aria-selected", "true"));
    expect(tab("Chained")).toBeDisabled();
  });

  it("leaves the current mode alone when the new base model can run it", async () => {
    // The corollary: the bounce must be caused by the RESTRICTION, not by the
    // switch. A user on Inventory stays on Inventory.
    const { select } = await renderApp(AS_LTX25);
    const user = userEvent.setup();

    await user.click(tab("Inventory"));
    await waitFor(() => expect(tab("Inventory")).toHaveAttribute("aria-selected", "true"));

    await switchToLtx25(select);

    await waitFor(() => expect(tab("Chained")).toBeDisabled());
    expect(tab("Inventory")).toHaveAttribute("aria-selected", "true");
  });

  it("disables the Batch A2V panel and says why (M4)", async () => {
    // Batch A2V is not a mode — it is a `<details>` section on Create — so the
    // tab greying above cannot reach it, yet every row it queues is a
    // `POST /generate/chain`.
    const { select, container } = await renderApp(AS_LTX25);

    // The panel is a collapsed `<details>`, but jsdom keeps its children in
    // the DOM either way — no click needed to reach them.
    //
    // The Start button is NOT the probe here: it is already disabled on a
    // fresh panel (no folder scanned yet, so `canStart` is false), which would
    // make the assertion pass for the wrong reason. The folder inputs are the
    // honest one — they are live until something disables them.
    const section = container.querySelector(".batch-section") as HTMLDetailsElement;
    const inputs = () => within(section).getAllByRole("textbox") as HTMLInputElement[];
    expect(inputs().length).toBeGreaterThan(0);
    expect(inputs().every((i) => i.disabled)).toBe(false);

    await switchToLtx25(select);

    // Editing a queue that can never run is worse than not offering it.
    await waitFor(() => expect(inputs().every((i) => i.disabled)).toBe(true));
    // A greyed control with no stated reason is exactly what this panel's other
    // block-reason lines already exist to prevent.
    expect(within(section).getByText(/not available on the selected base model/i)).toBeInTheDocument();
  });

  it("leaves the Batch A2V panel usable on LTX 2.3", async () => {
    const { container } = await renderApp();

    const section = container.querySelector(".batch-section") as HTMLDetailsElement;
    expect(within(section).queryByText(/not available on the selected base model/i)).not.toBeInTheDocument();
    const inputs = within(section).getAllByRole("textbox") as HTMLInputElement[];
    expect(inputs.length).toBeGreaterThan(0);
    expect(inputs.every((i) => i.disabled)).toBe(false);
  });
});
