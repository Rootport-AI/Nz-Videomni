import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createMockBridge } from "../bridge/mockBridge";
import type { MockBridgeOptions } from "../bridge/mockBridge";
import { withExtraUnsupportedFeatures } from "../test/unsupportedFeatures";
import { ACCELERATION_STORAGE_KEY, readStoredAcceleration } from "./accelerationSettings";
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
  // backend Docs/PENDING_TASKS_CLOSED.md's old §1-26 (hiding PrunaVAED on LTX 2.5,
  // closed 2026-09-01): the two PrunaVAED tests below SEED this key, and every
  // `AppShell` mount writes it back through `useAccelerationSettings` — so a
  // leftover would decide the next test's starting choice. Same precaution
  // `SettingsPanel.test.tsx` takes for the same key.
  beforeEach(() => {
    window.localStorage.removeItem(ACCELERATION_STORAGE_KEY);
  });

  afterEach(() => {
    window.localStorage.removeItem(ACCELERATION_STORAGE_KEY);
  });

  it("leaves every tab enabled on a base model that declares no restrictions", async () => {
    await renderApp();

    // The regression that matters most: LTX 2.3 must look exactly as it did
    // before this feature existed.
    for (const name of ["Single", "Chained", "Edit", "Inventory"]) {
      expect(tab(name)).not.toBeDisabled();
    }
  });

  it("leaves every TAB enabled on LTX 2.5 — a restriction greys one level down", async () => {
    // §3-102 gave the engine `chain`, so the Chained tab came back; the Retake
    // increment gave it 撮り直し, and the Edit tab hosts Retake AND Outpainting,
    // so ONE of the two being runnable is enough to keep the tab. A restriction
    // that remains therefore has to grey ONE LEVEL DOWN, on its own sub-tab.
    //
    // THE RESTRICTION HERE IS SYNTHETIC (`withExtraUnsupportedFeatures`), and
    // since the Outpainting increment it has to be: LTX 2.5's real list greys
    // NOTHING any more — 画角拡張 was the last mode it refused — so there would
    // be no way to tell "the response was read and nothing greys" from "the
    // response was never read". The synthetic `retake` is this test's settle
    // signal, which is what makes the `outpaint` assertion after it meaningful:
    // by then the new feature list has demonstrably been applied.
    const { select, container } = await renderApp(AS_LTX25, ["retake"]);

    await switchToLtx25(select);

    await waitFor(() => expect(editSubTab(container, "Retake")).toBeDisabled());
    // THE INCREMENT'S OWN EVIDENCE. Until Outpainting opened, this sub-tab was
    // the one that greyed on LTX 2.5 and 撮り直し was the one that did not —
    // the two have swapped, and only the synthetic name greys now.
    expect(editSubTab(container, "Outpainting")).not.toBeDisabled();
    for (const name of ["Single", "Chained", "Edit", "Inventory"]) {
      expect(tab(name)).not.toBeDisabled();
    }
  });

  it("switching back to LTX 2.3 restores the Edit sub-tab", async () => {
    // The same round trip the tab-level test used to make, one level down:
    // a restriction that never lifts is not a restriction, it is a broken build.
    // Synthetic for the same reason the test above is — and here the need is
    // sharper still, because a round trip driven by a list that greys nothing
    // would compare "not disabled" with "not disabled" at both ends.
    const { select, container } = await renderApp(AS_LTX25, ["retake"]);
    const user = userEvent.setup();

    await switchToLtx25(select);
    await waitFor(() => expect(editSubTab(container, "Retake")).toBeDisabled());
    expect(editSubTab(container, "Outpainting")).not.toBeDisabled();

    await user.selectOptions(select, "LTX23");
    await waitFor(() => expect(select.value).toBe("LTX23"));

    await waitFor(() => expect(editSubTab(container, "Retake")).not.toBeDisabled());
    expect(editSubTab(container, "Outpainting")).not.toBeDisabled();
    for (const name of ["Single", "Chained", "Edit", "Inventory"]) {
      expect(tab(name)).not.toBeDisabled();
    }
  });

  it("bounces out of a mode the new base model cannot run", async () => {
    // The case greying alone cannot cover: the user is ALREADY on Edit when
    // they switch. Leaving that panel open behind a disabled tab would let them
    // fill in a form whose every submission comes back 422.
    //
    // THE BASE MODEL HERE IS SYNTHETIC, and since the Outpainting increment
    // BOTH halves of it are. Edit greys only when BOTH of its sub-modes are
    // refused; the Retake increment gave the engine 撮り直し and the Outpainting
    // increment gave it 画角拡張, so the fixture's LTX 2.5 refuses neither now.
    // The two names are added to the published list rather than the test being
    // re-pointed at a different tab every time the engine grows. What is under
    // test is the BOUNCE, not today's feature list (that is
    // `bridge/mockBridge.test.ts`'s job).
    const { select } = await renderApp(AS_LTX25, ["retake", "outpaint"]);
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
    // did land, which is proven with a SYNTHETIC `retake` addition
    // (`withExtraUnsupportedFeatures`) rather than the real `outpaint`
    // refusal: once Outpainting opens up there is nothing left in the real
    // fixture to grey, and this waitFor would hang forever. The synthetic name
    // settles on the same `active`-derived recompute (`editSubTabsDisabledFor`)
    // regardless of what the real feature list says.
    const { select, container } = await renderApp(AS_LTX25, ["retake"]);
    const user = userEvent.setup();

    await user.click(tab("Inventory"));
    await waitFor(() => expect(tab("Inventory")).toHaveAttribute("aria-selected", "true"));

    await switchToLtx25(select);

    await waitFor(() => expect(editSubTab(container, "Retake")).toBeDisabled());
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
    const { select, container } = await renderApp(AS_LTX25, ["retake"]);

    await switchToLtx25(select);
    // The settle signal is a SYNTHETIC `retake` addition
    // (`withExtraUnsupportedFeatures`), not the real `outpaint` refusal:
    // settling on an assertion about the very thing under test below would be
    // no signal at all, and the real refusal stops existing the moment
    // Outpainting opens up.
    await waitFor(() => expect(editSubTab(container, "Retake")).toBeDisabled());

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
    const { select, container } = await renderApp(AS_LTX25, ["retake"]);

    await switchToLtx25(select);
    // The settle signal is a SYNTHETIC `retake` addition
    // (`withExtraUnsupportedFeatures`), not a greyed Chain panel (none exists
    // to grey since the End-source increment) and not the real `outpaint`
    // refusal, which stops existing the moment Outpainting opens up.
    await waitFor(() => expect(editSubTab(container, "Retake")).toBeDisabled());

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
    const { select, container } = await renderApp(AS_LTX25, ["retake"]);

    // The Create-tab reference block has no class of its own — its 📁 button
    // is named by `strings.single.referenceVideo.chooseButton`, and scoping to
    // the Create form is what keeps the identically-worded Chained one out.
    const singleForm = container.querySelector(".generation-form:not(.chained-form)") as HTMLElement;
    const singleRefPick = () =>
      within(singleForm).getByRole("button", { name: /choose reference video/i }) as HTMLButtonElement;
    expect(singleRefPick().disabled).toBe(false);

    await switchToLtx25(select);

    // A SYNTHETIC `retake` addition (`withExtraUnsupportedFeatures`) settling
    // into its greyed state is what proves the switch took effect before the
    // assertions below run — not the real `outpaint` refusal, which stops
    // existing the moment Outpainting opens up.
    await waitFor(() => expect(editSubTab(container, "Retake")).toBeDisabled());

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
    const { select, container } = await renderApp(AS_LTX25, ["retake"]);

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

    // The switch settles on a SYNTHETIC `retake` addition
    // (`withExtraUnsupportedFeatures`) rather than the real `outpaint`
    // refusal, which stops existing the moment Outpainting opens up; once it
    // has, the Batch inputs must still be live and no block reason may have
    // appeared.
    await waitFor(() => expect(editSubTab(container, "Retake")).toBeDisabled());
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

  // -- backend Docs/PENDING_TASKS_CLOSED.md's old §1-26 (closed 2026-09-01): PrunaVAED,
  // -- a Settings row rather than a mode ---------------------------------------
  //
  // The name `prune_vaed` is a REAL entry on LTX 2.5's published list (unlike
  // the synthetic ones above), and unlike every other feature in this file it
  // greys nothing: an engine that publishes it answers `vae_mode` with a 422
  // instead of degrading, so the row is HIDDEN and the stored choice is written
  // back to the server default.

  /** Seeds the persisted Acceleration choices with PrunaVAED selected — the
   * leftover a user leaves behind by picking it on LTX 2.3. Written as the
   * whole JSON object `readStoredAcceleration` reads, so the other four fields
   * land on their own defaults rather than on `undefined`. */
  function storePruneVaed() {
    window.localStorage.setItem(
      ACCELERATION_STORAGE_KEY,
      JSON.stringify({
        attentionBackend: "sdpa",
        blockSwapPrefetch: true,
        keepResident: false,
        fusedGgufDequantKernel: true,
        vaeMode: "prune_vaed",
      }),
    );
  }

  /** The PrunaVAED row itself — its `role="group"` button pair, named by
   * `strings.settings.accelVaeLabel`. */
  function vaeRow() {
    return screen.queryByRole("group", { name: /^VAE \(video decode\)$/ });
  }

  /** The row's warning note (`strings.settings.accelVaeNote`), a SIBLING of the
   * row rather than a child — which is the whole reason the two are wrapped by
   * one condition, and why both are asserted here. */
  function vaeNote() {
    return screen.queryByText(/a pruned decoder speeds up video reconstruction/i);
  }

  it("hides the PrunaVAED row AND its note on LTX 2.5", async () => {
    // PrunaVAED is stored as the choice, so BOTH halves are on screen to begin
    // with — without that the note would be absent either way and the
    // disappearance of the row alone would prove nothing about it.
    storePruneVaed();
    const { select } = await renderApp(AS_LTX25);
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: "Settings" }));
    expect(vaeRow()).toBeInTheDocument();
    expect(vaeNote()).toBeInTheDocument();

    // The panel stays open across the switch: the header dropdown is outside
    // the modal, and watching the row vanish in place is exactly the moment
    // this hiding exists for.
    await switchToLtx25(select);

    await waitFor(() => expect(vaeRow()).toBeNull());
    expect(vaeNote()).toBeNull();
  });

  it("writes a leftover PrunaVAED choice back to the server default on LTX 2.5", async () => {
    // Hiding the row is not enough on its own: the choice PERSISTS, so a
    // `vae_mode: "prune_vaed"` picked on LTX 2.3 would keep riding along on
    // every request to an engine that 422s it. The normalization effect is what
    // this asserts, through `localStorage` rather than through the UI — the row
    // it would have shown up in is gone by then.
    storePruneVaed();
    const { select } = await renderApp(AS_LTX25);
    expect(readStoredAcceleration().vaeMode).toBe("prune_vaed");

    await switchToLtx25(select);

    await waitFor(() => expect(readStoredAcceleration().vaeMode).toBe("default"));
  });

  // -- 台帳 §3-114 (2026-09-03): keep the embeddings processor resident, the
  // -- same mechanism RUNNING BACKWARDS ------------------------------------
  //
  // `keep_resident_embeddings` is a REAL entry too, but on LTX 2.3's published
  // list rather than LTX 2.5's — it names LTX 2.5's embeddings processor, a
  // component 2.3's pipeline does not have. So this is the first feature in
  // this file whose row is hidden BEFORE the base-model switch and appears
  // AFTER it, and the first one LTX 2.3 has ever refused. Everything else about
  // it matches PrunaVAED above: a hard 422 rather than a downgrade, so the row
  // is hidden outright and a leftover choice is written back.

  /** Seeds the persisted Acceleration choices with the embeddings processor
   * kept resident — the leftover a user leaves behind by turning it on while
   * LTX 2.5 is loaded. Whole JSON object, same as {@link storePruneVaed}. */
  function storeKeepResidentEmbeddings() {
    window.localStorage.setItem(
      ACCELERATION_STORAGE_KEY,
      JSON.stringify({
        attentionBackend: "sdpa",
        blockSwapPrefetch: true,
        keepResident: false,
        fusedGgufDequantKernel: true,
        vaeMode: "default",
        keepResidentEmbeddings: true,
      }),
    );
  }

  /** The row itself — its `role="group"` button pair, named by
   * `strings.settings.accelKeepResidentEmbeddingsLabel`. The `^…$` anchors keep
   * it from also matching the plain keep-resident row above it. */
  function embeddingsRow() {
    return screen.queryByRole("group", { name: /^Keep embeddings processor resident \(LTX 2\.5\)$/ });
  }

  /** The row's note (`strings.settings.accelKeepResidentEmbeddingsNote`), a
   * SIBLING of the row — asserted alongside it for the same reason the
   * PrunaVAED note is: the two are wrapped by one condition. */
  function embeddingsNote() {
    return screen.queryByText(/keeps the part that arranges the prompt's reading between jobs/i);
  }

  it("hides the keep-embeddings-resident row AND its note on LTX 2.3, and shows both on LTX 2.5", async () => {
    // The REVERSE of the PrunaVAED case: absent at startup (LTX 2.3 is what
    // loads first) and present after the switch. Seeded ON so the note is on
    // screen the moment the row is — without that, the note would be absent
    // either way and the row's appearance would prove nothing about it.
    storeKeepResidentEmbeddings();
    const { select } = await renderApp(AS_LTX25);
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: "Settings" }));
    // LTX 2.3 publishes the name, so the row is not there at all — and the
    // normalization effect below has already turned the stored choice off.
    expect(embeddingsRow()).toBeNull();
    expect(embeddingsNote()).toBeNull();

    // The panel stays open across the switch, exactly as in the PrunaVAED test.
    await switchToLtx25(select);

    await waitFor(() => expect(embeddingsRow()).toBeInTheDocument());
    // The note follows the row's OWN state, and the normalization above turned
    // the choice off, so it is the row that has to be turned on for the note to
    // return. That is the sibling relationship this assertion is really about.
    expect(embeddingsNote()).toBeNull();
    await user.click(within(embeddingsRow()!).getByRole("button", { name: "ON" }));
    await waitFor(() => expect(embeddingsNote()).toBeInTheDocument());
  });

  it("writes a leftover keep-embeddings-resident choice back to the server default on LTX 2.3", async () => {
    // Hiding the row is not enough on its own, same as PrunaVAED: the choice
    // PERSISTS, so a `true` picked on LTX 2.5 would keep riding along on every
    // request to LTX 2.3, which 422s it. Here the write-back fires at STARTUP
    // rather than after a switch — LTX 2.3 is the model that loads first — so
    // this is the "leftover from a previous session" half of the effect, which
    // the PrunaVAED pair covers only through a switch.
    storeKeepResidentEmbeddings();
    await renderApp(AS_LTX25);

    await waitFor(() => expect(readStoredAcceleration().keepResidentEmbeddings).toBe(false));
  });
});
