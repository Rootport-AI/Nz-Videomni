import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { createMockBridge } from "../bridge/mockBridge";
import type { MockBridgeOptions } from "../bridge/mockBridge";
import { withExtraUnsupportedFeatures } from "../test/unsupportedFeatures";
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

async function renderApp(options: MockBridgeOptions = {}, extraUnsupported: readonly string[] = []) {
  const base = createMockBridge({ delayMs: 0, ...options });
  const bridge = extraUnsupported.length
    ? withExtraUnsupportedFeatures(base, "LTX25", extraUnsupported)
    : base;
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

/** One of the Edit tab's SUB-tabs.
 *
 * Reached through the sub-tab strip rather than by role name alone, because the
 * main mode tabs and these are both `role="tab"` and every mode panel stays
 * mounted (hidden) — so a bare `getByRole` would be ambiguous. `container` is
 * used for the same reason the Chain-panel tests below use it: a hidden panel
 * is out of the accessibility tree but still in the DOM. */
function editSubTab(container: HTMLElement, name: string) {
  const strip = container.querySelector(".edit-subtabs");
  if (!strip) throw new Error("the Edit sub-tab strip is not mounted");
  return within(strip as HTMLElement).getByRole("tab", { name, hidden: true });
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

  it("leaves every TAB enabled on LTX 2.5 — the greying moved into Edit", async () => {
    // The Retake increment is what moved it. §3-102 gave the engine `chain`, so
    // the Chained tab came back; Retake gives it 撮り直し, and the Edit tab
    // hosts Retake AND Outpainting, so ONE of the two being runnable is enough
    // to keep the tab. What is still out of scope (画角拡張) therefore has to
    // grey ONE LEVEL DOWN, on its own sub-tab — which is exactly the machinery
    // C0 put in place for this moment.
    const { select, container } = await renderApp(AS_LTX25);

    await switchToLtx25(select);

    await waitFor(() => expect(editSubTab(container, "Outpainting")).toBeDisabled());
    expect(editSubTab(container, "Retake")).not.toBeDisabled();
    for (const name of ["Single", "Chained", "Edit", "Inventory"]) {
      expect(tab(name)).not.toBeDisabled();
    }
  });

  it("switching back to LTX 2.3 restores the Edit sub-tab", async () => {
    // The same round trip the tab-level test used to make, one level down:
    // a restriction that never lifts is not a restriction, it is a broken build.
    const { select, container } = await renderApp(AS_LTX25);
    const user = userEvent.setup();

    await switchToLtx25(select);
    await waitFor(() => expect(editSubTab(container, "Outpainting")).toBeDisabled());

    await user.selectOptions(select, "LTX23");
    await waitFor(() => expect(select.value).toBe("LTX23"));

    await waitFor(() => expect(editSubTab(container, "Outpainting")).not.toBeDisabled());
    expect(editSubTab(container, "Retake")).not.toBeDisabled();
    for (const name of ["Single", "Chained", "Edit", "Inventory"]) {
      expect(tab(name)).not.toBeDisabled();
    }
  });

  it("bounces out of a mode the new base model cannot run", async () => {
    // The case greying alone cannot cover: the user is ALREADY on Edit when
    // they switch. Leaving that panel open behind a disabled tab would let them
    // fill in a form whose every submission comes back 422.
    //
    // THE BASE MODEL HERE IS SYNTHETIC. Edit greys only when BOTH of its
    // sub-modes are refused, and the fixture's LTX 2.5 refuses only 画角拡張
    // since the Retake increment — so the extra name is added to the published
    // list rather than the test being re-pointed at a different tab every time
    // the engine grows. What is under test is the BOUNCE, not today's feature
    // list (that is `bridge/mockBridge.test.ts`'s job).
    const { select } = await renderApp(AS_LTX25, ["retake"]);
    const user = userEvent.setup();

    await user.click(tab("Edit"));
    await waitFor(() => expect(tab("Edit")).toHaveAttribute("aria-selected", "true"));

    await switchToLtx25(select);

    await waitFor(() => expect(tab("Single")).toHaveAttribute("aria-selected", "true"));
    expect(tab("Edit")).toBeDisabled();
  });

  it("leaves the current mode alone when the new base model can run it", async () => {
    // The corollary: the bounce must be caused by the RESTRICTION, not by the
    // switch. A user on Inventory stays on Inventory — and the switch really
    // did land, which the greyed Outpainting sub-tab is the positive signal for.
    const { select, container } = await renderApp(AS_LTX25);
    const user = userEvent.setup();

    await user.click(tab("Inventory"));
    await waitFor(() => expect(tab("Inventory")).toHaveAttribute("aria-selected", "true"));

    await switchToLtx25(select);

    await waitFor(() => expect(editSubTab(container, "Outpainting")).toBeDisabled());
    expect(tab("Inventory")).toHaveAttribute("aria-selected", "true");
  });


  // -- §3-102: the four Chain material panels --------------------------------
  //
  // With `chain` gone from LTX 2.5's scope the Chained TAB is live, so the
  // greying moves one level down: the panels that attach material the engine
  // still cannot use grey individually, each with its own stated reason. The
  // Chained tabpanel is `hidden` while another mode is selected, but jsdom keeps
  // its children in the DOM either way — `container.querySelector` reaches them
  // with no tab switch, exactly as the Batch A2V tests below do.
  //
  // §3-102 SECOND stage (V2V + A2V, incl. long A2V): the split moved again.
  // `v2v` and `a2v` left the declared list, so the A2V track panel and the V2V
  // half of the source card come BACK.
  //
  // §3-102 THIRD stage (Style LoRA + IC-LoRA, long IC-LoRA included):
  // `reference_video` left too, so the reference panel came back as well and
  // 素材（末尾） was the ONLY Chain material still greyed.
  //
  // END-SOURCE increment: `end_source` left as well, and with it the LAST name
  // that greyed ANY Chain material panel on this engine. The suite below is
  // therefore split in two, which is the same bargain the Edit sub-tab tests
  // struck one increment earlier:
  //
  //   * the MECHANISM — a published feature name greys ITS panel and states
  //     ITS reason — is driven with a SYNTHETIC name added to the response
  //     (`withExtraUnsupportedFeatures`), so it keeps testing the wiring
  //     whatever the engine grows next;
  //   * TODAY'S ANSWER — LTX 2.5 greys none of the four — is asserted against
  //     the real fixture. Both halves are needed: the first alone would pass on
  //     a build that greys everything, the second alone on a build that greys
  //     nothing ever.

  /** The 📁 choose buttons of the four Chain material panels — the honest
   * probe, the way the folder inputs are for Batch A2V: each is live until
   * something disables it. (The 🔁 clear buttons are not: they start disabled
   * with nothing attached, which would make the assertion pass for the wrong
   * reason.) `.source-input-pick` has its own helper below because what it
   * says in its title is part of what V2V being open means. */
  const CHAIN_PANEL_PICKS = [".chain-end-source-pick"];

  /** The panels greyed by nothing on either base model. Since the End-source
   * increment `CHAIN_PANEL_PICKS` is in the same position — it is kept as a
   * list of its own only because the synthetic-name test still needs to name
   * that ONE panel and read its reason line. */
  const CHAIN_OPEN_PICKS = [".chain-audio-pick", ".chain-reference-pick"];

  function chainForm(container: HTMLElement): HTMLElement {
    return container.querySelector(".chained-form") as HTMLElement;
  }

  function picksOf(container: HTMLElement, selectors: readonly string[]): HTMLButtonElement[] {
    return selectors.map((sel) => chainForm(container).querySelector(sel) as HTMLButtonElement);
  }

  function chainPicks(container: HTMLElement): HTMLButtonElement[] {
    return picksOf(container, CHAIN_PANEL_PICKS);
  }

  function openPicks(container: HTMLElement): HTMLButtonElement[] {
    return picksOf(container, CHAIN_OPEN_PICKS);
  }

  function sourcePick(container: HTMLElement): HTMLButtonElement {
    return chainForm(container).querySelector(".source-input-pick") as HTMLButtonElement;
  }

  it("greys a Chain material panel the base model declares, with its own reason", async () => {
    // THE FEATURE NAME HERE IS SYNTHETIC. The fixture's LTX 2.5 declares none
    // of the four Chain materials since the End-source increment, so pinning
    // this test to whatever it happens to refuse this month is exactly what
    // made it need rewriting at every increment. What is under test is the
    // WIRING — a published name reaches `chainPanelsDisabledFor`, greys ITS
    // panel and prints ITS reason — not today's list (`bridge/mockBridge.test.ts`
    // owns that, against the server's own `UNSUPPORTED_FEATURES`).
    const { select, container } = await renderApp(AS_LTX25, ["end_source"]);

    expect(chainPicks(container).every((b) => b != null)).toBe(true);
    expect(chainPicks(container).every((b) => b.disabled)).toBe(false);

    await switchToLtx25(select);

    await waitFor(() => expect(chainPicks(container).every((b) => b.disabled)).toBe(true));

    // A greyed control with no stated reason is what these lines exist to
    // prevent — one per panel, each naming ITS OWN material rather than a
    // generic "unsupported".
    const form = within(chainForm(container));
    expect(form.getByText(/End source is not available/i)).toBeInTheDocument();
    // …and NOT the other three. A reason line that appeared for a material
    // nobody refused would tell the user to switch base models for something
    // that works — the greying must be keyed on the NAME, not on "some
    // restriction exists".
    expect(form.queryByText(/source VIDEO cannot be used on the selected base model/i)).toBeNull();
    expect(form.queryByText(/Generating from an audio track \(A2V\) is not available/i)).toBeNull();
    expect(form.queryByText(/Reference video \(control IC-LoRA\) is not available/i)).toBeNull();

    // …and the rest of the Chain form is untouched: the tab is live because the
    // engine CAN chain, so the clip list must stay usable.
    // `getByRole` is off the table here: the Chained tabpanel is `hidden`, and
    // role queries skip hidden subtrees. The text is the way in.
    const addClip = form.getByText(/add clip/i).closest("button") as HTMLButtonElement;
    expect(addClip.disabled).toBe(false);
  });

  it("greys NO Chain material panel on LTX 2.5 — the End-source increment took the last", async () => {
    // The other half, and the increment's actual headline: with `end_source`
    // off the published list every one of the four materials is attachable on
    // this engine, so the Chained form looks the same on both base models. A
    // stale fixture would go on greying 素材（末尾） for a mode that runs, which
    // is the one failure this pair exists to catch.
    const { select, container } = await renderApp(AS_LTX25);

    await switchToLtx25(select);
    // The greyed Outpainting sub-tab is the positive signal that the switch
    // landed — `outpaint` is what LTX 2.5 still refuses, and settling on an
    // assertion about the thing under test would be no signal at all.
    await waitFor(() => expect(editSubTab(container, "Outpainting")).toBeDisabled());

    expect(chainPicks(container).every((b) => b != null)).toBe(true);
    expect(chainPicks(container).every((b) => b.disabled)).toBe(false);
    expect(openPicks(container).every((b) => b.disabled)).toBe(false);
    expect(sourcePick(container).disabled).toBe(false);

    const form = within(chainForm(container));
    expect(form.queryByText(/is not available on the selected base model/i)).toBeNull();
    expect(form.queryByText(/cannot be used on the selected base model/i)).toBeNull();
  });

  it("keeps the V2V source, the A2V track and the reference video attachable on LTX 2.5", async () => {
    // The second stage's headline: continuing an existing video (V2V) and
    // generating from an audio track (A2V) both run on this engine now, so the
    // source card keeps BOTH halves and the audio panel stays live. Greying
    // either would make a supported path unreachable — the same mistake the
    // first stage had to correct for clip 1's opening image. §3-102 third
    // stage adds the reference video to `CHAIN_OPEN_PICKS` for the same reason.
    const { select, container } = await renderApp(AS_LTX25);

    await switchToLtx25(select);
    // The settle signal is the greyed Outpainting sub-tab, not a greyed Chain
    // panel: since the End-source increment there is no longer such a panel.
    await waitFor(() => expect(editSubTab(container, "Outpainting")).toBeDisabled());

    expect(openPicks(container).every((b) => b != null)).toBe(true);
    expect(openPicks(container).every((b) => b.disabled)).toBe(false);

    expect(sourcePick(container).disabled).toBe(false);
    // The source card offers images AND video again — the wording is what tells
    // the user the video half is open, so it is asserted rather than assumed.
    expect(sourcePick(container).getAttribute("title")).toMatch(/choose image or video/i);
  });

  it("keeps the reference-video panels attachable on LTX 2.5 — Create AND Chained", async () => {
    // §3-102 third stage's headline, asserted on BOTH screens: `loras` and
    // `reference_video` left LTX 2.5's declared list, so Style LoRA and IC-LoRA
    // (long IC-LoRA included) run here now. Chained's reference panel is the one
    // the scope actually greyed (`referenceUnavailable`); Create's never was,
    // and is asserted anyway so a prop later wired to the same feature name
    // could not close it unnoticed.
    //
    // Since the End-source increment there is no CONTRAST material left to
    // check against on this engine, so the settle signal moved to the Edit
    // sub-tab and the end-source panel is asserted OPEN below with the rest.
    const { select, container } = await renderApp(AS_LTX25);

    // The Create-tab reference block has no class of its own — its 📁 button
    // is named by `strings.single.referenceVideo.chooseButton`, and scoping to
    // the Create form is what keeps the identically-worded Chained one out.
    const singleForm = container.querySelector(".generation-form:not(.chained-form)") as HTMLElement;
    const singleRefPick = () =>
      within(singleForm).getByRole("button", { name: /choose reference video/i }) as HTMLButtonElement;
    expect(singleRefPick().disabled).toBe(false);

    await switchToLtx25(select);

    // The Outpainting sub-tab settling into its greyed state is what proves the
    // switch took effect before the assertions below run.
    await waitFor(() => expect(editSubTab(container, "Outpainting")).toBeDisabled());

    const chainRefPick = chainForm(container).querySelector(".chain-reference-pick") as HTMLButtonElement;
    expect(chainRefPick).not.toBeNull();
    expect(chainRefPick.disabled).toBe(false);
    expect(singleRefPick().disabled).toBe(false);

    // No reason line for a material the engine can take — which since the
    // End-source increment is all four of them.
    const form = within(chainForm(container));
    expect(form.queryByText(/Reference video \(control IC-LoRA\) is not available/i)).toBeNull();
    expect(form.queryByText(/End source is not available/i)).toBeNull();
  });

  it("leaves every Chain material panel usable on LTX 2.3", async () => {
    const { container } = await renderApp();

    expect(chainPicks(container).every((b) => b.disabled)).toBe(false);
    expect(openPicks(container).every((b) => b.disabled)).toBe(false);
    expect(sourcePick(container).disabled).toBe(false);
    expect(sourcePick(container).getAttribute("title")).toMatch(/choose image or video/i);
    expect(within(chainForm(container)).queryByText(/is not available on the selected base model/i)).toBeNull();
    expect(within(chainForm(container)).queryByText(/cannot be used on the selected base model/i)).toBeNull();
  });

  it("leaves the Batch A2V panel usable on LTX 2.5 (M4)", async () => {
    // Batch A2V is not a mode — it is a `<details>` section on Create — so the
    // tab greying above cannot reach it, yet every row it queues is a
    // `POST /generate/chain` carrying an audio track. §3-102 second stage: that
    // is exactly what LTX 2.5 can run now, so the panel must survive the switch.
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

    // The switch settles on the greyed Outpainting sub-tab; once it has, the
    // Batch inputs must still be live and no block reason may have appeared.
    await waitFor(() => expect(editSubTab(container, "Outpainting")).toBeDisabled());
    expect(inputs().every((i) => i.disabled)).toBe(false);
    expect(within(section).queryByText(/not available on the selected base model/i)).not.toBeInTheDocument();
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
