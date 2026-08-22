import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { createMockBridge } from "../bridge/mockBridge";
import type { MockBridgeOptions } from "../bridge/mockBridge";
import { AppShell } from "./AppShell";

// Header BASE MODEL dropdown (§3-97 P7, `Docs/MULTI_ENGINE_DESIGN.md` §6).
//
// THE SPEC HERE IS THE INVERSE of what this file tested until 2026-08-20. The
// dropdown used to be a mock: two hard-coded literals, and picking "LTX 2.5"
// snapped back to "LTX 2.3" SILENTLY, on purpose — there was nothing behind it
// to explain. It is now really wired, so the same click has to do the opposite:
// say what happened and why. "Silently reverts" is no longer the guarantee;
// "reverts AND tells you" is (guard 4 — the display must never disagree with
// what is actually loaded, and a wordless revert looks like a broken widget).
//
// The options are the server's `base_models[]`, not client-side literals, so
// none of these assertions name a model the fixture did not declare.

async function renderApp(options: MockBridgeOptions = {}) {
  const bridge = createMockBridge({ delayMs: 0, ...options });
  const requests = vi.spyOn(bridge, "request");
  const utils = render(<AppShell nativeBridge={bridge} />);
  await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
  const select = await screen.findByRole("combobox", { name: /base model/i });
  await waitFor(() => expect((select as HTMLSelectElement).value).toBe("LTX23"));
  return { ...utils, select: select as HTMLSelectElement, requests };
}

function loadCalls(requests: { mock: { calls: unknown[][] } }) {
  return requests.mock.calls.filter(
    (call) => call[0] === "backend.request" && (call[1] as { path?: string } | undefined)?.path === "/api/v1/pipeline/load",
  );
}

async function toastText(container: HTMLElement) {
  const stack = await waitFor(() => {
    const el = container.querySelector(".toast-stack .toast-message");
    if (!el) throw new Error("no toast yet");
    return el;
  });
  return stack.textContent ?? "";
}

describe("AppShell — header base-model dropdown", () => {
  it("builds its options from base_models[], annotating anything not fully installed", async () => {
    const { select } = await renderApp();

    const options = within(select).getAllByRole("option") as HTMLOptionElement[];
    expect(options.map((o) => o.value)).toEqual(["LTX23", "LTX25"]);
    // Display names come from the server; the partial install is called out so
    // the entry's failure is predictable before it is clicked.
    expect(options[0]?.textContent).toBe("LTX 2.3");
    expect(options[1]?.textContent).toContain("LTX 2.5");
    expect(options[1]?.textContent).toContain("partly installed");
    // The old static title is gone for good.
    expect(screen.queryByText("Nz-Videomni")).not.toBeInTheDocument();
  });

  it("switching to an installed base model loads it and the value moves there", async () => {
    // Models the world after §3-98: LTX 2.5 fully installed AND runnable.
    const { select, container, requests } = await renderApp({
      ltx25Install: "full",
      supportedBaseModels: ["LTX23", "LTX25"],
    });
    const user = userEvent.setup();

    await user.selectOptions(select, "LTX25");

    await waitFor(() => expect(select.value).toBe("LTX25"));
    expect(await toastText(container)).toContain("LTX 2.5");
    // Picking it IS loading it — one `POST /pipeline/load` carrying the base
    // model and no per-category selection (§6.1/§6.3: the server resolves that
    // base's own categories).
    expect(loadCalls(requests)).toHaveLength(1);
    expect(loadCalls(requests)[0]?.[1]).toMatchObject({ body: { base_model: "LTX25", models: {} } });
  });

  it("a 422 shows the server's own reason and puts the selection back", async () => {
    const { select, container, requests } = await renderApp({ ltx25Install: "full" });
    const user = userEvent.setup();

    await user.selectOptions(select, "LTX25");

    // The server's `detail` verbatim — the WebUI must not paraphrase the
    // incompatibility, since only the server knows what it actually found.
    expect(await toastText(container)).toContain("LTX 2.5エンジンは次段階");
    // Guard 4: back to the base model that is still loaded.
    await waitFor(() => expect(select.value).toBe("LTX23"));
    expect(loadCalls(requests)).toHaveLength(1);
  });

  it("an uninstalled base model is answered with setup guidance, without calling the API", async () => {
    const { select, container, requests } = await renderApp({ ltx25Install: "none" });
    const user = userEvent.setup();

    const options = within(select).getAllByRole("option") as HTMLOptionElement[];
    expect(options[1]?.textContent).toContain("not installed");

    await user.selectOptions(select, "LTX25");

    expect(await toastText(container)).toContain("LTX 2.5 is not installed");
    expect(select.value).toBe("LTX23");
    // Guard 2: nothing is on disk, so the server is never asked. Putting the
    // weights in place is the setup procedure's job, never the server's (§6.2).
    expect(loadCalls(requests)).toHaveLength(0);
  });

  it("is disabled while the switch it started is still in flight", async () => {
    // A real load takes minutes; a second pick during it would only earn a 409
    // PIPELINE_LOADING, so the control locks itself for the duration. (The
    // sibling guard — disabled while a generation job holds the queue — is
    // driven by the `/status` poll, which in this harness runs against the
    // app-wide singleton bridge rather than the injected one, so it is not
    // reachable from here.)
    const { select, container } = await renderApp({ delayMs: 40 });
    const user = userEvent.setup();

    await user.selectOptions(select, "LTX25");

    expect(select).toBeDisabled();
    // …and released once the attempt settles, whichever way it went.
    await toastText(container);
    await waitFor(() => expect(select).toBeEnabled());
  });
});
