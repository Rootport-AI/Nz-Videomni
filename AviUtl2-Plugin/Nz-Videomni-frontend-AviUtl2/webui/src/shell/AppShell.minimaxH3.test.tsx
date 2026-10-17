import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { bridge as appBridge } from "../bridge";
import { createMockBridge } from "../bridge/mockBridge";
import type { NativeBridge } from "../bridge";
import { AppShell } from "./AppShell";

// MiniMax H3 第 2 弾 (2026-10-17) — 2c-G1, end to end through the fixture
// server. Picking "MiniMax H3" in the header must, with NO engine id read
// anywhere on screen (only `GET /models`' `constraints` and
// `unsupported_features`):
//  - replace the CFG placeholder with the three sampling sliders,
//  - grey the Chained and Edit tabs,
//  - hide Create's reference video, grey its A2V card, the Negative Prompt
//    accordion, the second keyframe and the keyframe strength,
//  - drop the comfort marker,
//  - snap DURATION onto 17n+5 and pin fps at 24,
// and the request it sends must carry the three sampling values and no NAG.

const SINGLE_PANEL = '[role="tabpanel"]:not([hidden])';

function singlePanel(): HTMLElement {
  const el = document.querySelector(SINGLE_PANEL);
  if (!el) throw new Error("no visible tab panel");
  return el as HTMLElement;
}

/** Create's comfort-marker tick value, or `null` when none is drawn. */
function createMarker(): string | null {
  return document.querySelector(`${SINGLE_PANEL} datalist#spill-tick option`)?.getAttribute("value") ?? null;
}

/** Create's DURATION number input — the only number input whose `step` is the
 * frame grid's (8 on LTX, 17 on MiniMax H3) inside the Duration label. */
function durationNumber(): HTMLInputElement {
  const label = within(singlePanel()).getByText("Duration").closest("label");
  if (!label) throw new Error("no Duration field");
  return label.querySelector('input[type="number"]') as HTMLInputElement;
}

/** The number input inside the `.field` label whose label text is `text`. */
function numberInLabel(text: string): HTMLInputElement {
  const label = within(singlePanel()).getByText(text, { selector: ".field-label" }).closest("label");
  if (!label) throw new Error(`no field labelled ${text}`);
  return label.querySelector('input[type="number"]') as HTMLInputElement;
}

/** `bridge` defaults to a fresh fixture server. The submit test passes the
 * app-wide singleton instead: Create's submit goes through the singleton
 * `apiClient`, so the header switch has to reach the SAME fixture server for
 * the generate request to be judged against MiniMax H3's rules. */
async function renderApp(bridge: NativeBridge = createMockBridge({ delayMs: 0 })): Promise<NativeBridge> {
  render(<AppShell nativeBridge={bridge} />);
  await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
  const select = (await screen.findByRole("combobox", { name: /base model/i })) as HTMLSelectElement;
  await waitFor(() => expect(select.value).toBe("LTX23"));
  return bridge;
}

async function switchTo(id: string) {
  const user = userEvent.setup();
  const select = screen.getByRole("combobox", { name: /base model/i }) as HTMLSelectElement;
  await user.selectOptions(select, id);
  await waitFor(() => expect(select.value).toBe(id));
}

describe("AppShell — MiniMax H3 (第 2 弾, 2c-G1)", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });
  afterEach(() => {
    window.localStorage.clear();
    vi.restoreAllMocks();
  });

  it(
    "LTX 2.3 keeps its screen: CFG placeholder, marker, 8n+1 step, editable fps, every tab",
    async () => {
      await renderApp();
      const panel = singlePanel();
      // The regression that matters: none of the new parts reach LTX.
      expect(within(panel).getByText("CFG scale")).toBeInTheDocument();
      expect(within(panel).queryByText("Steps")).not.toBeInTheDocument();
      await waitFor(() => expect(createMarker()).not.toBeNull());
      expect(durationNumber().step).toBe("8");
      expect(within(panel).getByRole("spinbutton", { name: /fps/i })).not.toBeDisabled();
      expect(within(panel).getByText("Reference video (IC-LoRA)")).toBeInTheDocument();
      for (const name of ["Single", "Chained", "Edit"]) {
        expect(screen.getByRole("tab", { name })).not.toBeDisabled();
      }
    },
    15_000,
  );

  it(
    "switching to MiniMax H3 shows the sliders and closes / greys exactly what its engine cannot do",
    async () => {
      await renderApp();
      await switchTo("MiniMaxH3");

      // Sliders in place of the CFG placeholder, at the served defaults.
      await waitFor(() => expect(within(singlePanel()).getByText("Steps")).toBeInTheDocument());
      const panel = singlePanel();
      expect(within(panel).queryByText("CFG scale")).not.toBeInTheDocument();
      expect(numberInLabel("Steps").value).toBe("20");
      expect(numberInLabel("Shift (video)").value).toBe("12");
      expect(numberInLabel("Shift (audio)").value).toBe("3");

      // The Chained tab greys, and so does the Edit tab — see the dedicated
      // case below for the four sub-tabs behind it.
      expect(screen.getByRole("tab", { name: "Chained" })).toBeDisabled();
      expect(screen.getByRole("tab", { name: "Edit" })).toBeDisabled();
      expect(screen.getByRole("tab", { name: "Single" })).not.toBeDisabled();

      // Reference video hidden; A2V greyed with its note.
      expect(within(panel).queryByText("Reference video (IC-LoRA)")).not.toBeInTheDocument();
      expect(within(panel).getByText(/Generating from an audio track \(A2V\) is not available/)).toBeInTheDocument();
      // Code review m-4: the A2V card's controls themselves are disabled.
      const a2vCard = within(panel).getByText("Audio to video (A2V)").closest("details") as HTMLElement;
      const a2vButtons = within(a2vCard).getAllByRole("button", { hidden: true });
      expect(a2vButtons.length).toBeGreaterThan(0);
      for (const button of a2vButtons) expect(button).toBeDisabled();

      // Negative Prompt accordion greyed with its note; the switch is disabled.
      expect(screen.getByText(/Negative prompts \(NAG\/VSF\) are not available/)).toBeInTheDocument();
      expect(screen.getByRole("checkbox", { name: /non-cfg negative/i })).toBeDisabled();

      // No comfort marker; DURATION on 17n+5 with a 17 step; fps fixed at 24.
      expect(createMarker()).toBeNull();
      const frames = Number(durationNumber().value);
      expect((frames - 5) % 17).toBe(0);
      expect(frames).toBeGreaterThanOrEqual(22);
      expect(durationNumber().step).toBe("17");
      const fps = within(panel).getByRole("spinbutton", { name: /fps/i }) as HTMLInputElement;
      expect(fps).toBeDisabled();
      expect(fps.value).toBe("24");
    },
    15_000,
  );

  it(
    "MiniMax H3: all four Edit sub-tabs close, so the Edit tab itself greys",
    async () => {
      // Supervisor ruling 2026-10-17: the H3 adapter publishes `retake` (a
      // Retake is a chain request it refuses); `chain` itself still reaches no
      // Edit sub-tab. With Outpainting / Inpainting / Alpha Gen closed too,
      // `CONTAINER_TARGETS` closes the whole tab.
      await renderApp();
      expect(screen.getByRole("tab", { name: "Edit" })).not.toBeDisabled();
      await switchTo("MiniMaxH3");
      await waitFor(() => expect(screen.getByRole("tab", { name: "Edit" })).toBeDisabled());
      const strip = document.querySelector(".edit-subtabs") as HTMLElement;
      for (const name of ["Retake", "Outpainting", "Inpainting", "Alpha Gen"]) {
        expect(within(strip).getByRole("tab", { name: new RegExp(`^${name}`), hidden: true })).toBeDisabled();
      }
    },
    15_000,
  );

  it(
    "MiniMax H3: one keyframe only, and its strength is greyed",
    async () => {
      await renderApp();
      await switchTo("MiniMaxH3");
      const user = userEvent.setup();
      const panel = singlePanel();
      await waitFor(() => expect(within(panel).getByText(/uses only one keyframe/)).toBeInTheDocument());

      const add = within(panel).getByRole("button", { name: /add keyframe \( 0 \/ 1 \)/i });
      expect(add).not.toBeDisabled();
      await user.click(add);
      await waitFor(() =>
        expect(within(panel).getByRole("button", { name: /add keyframe \( 1 \/ 1 \)/i })).toBeDisabled(),
      );
      const strength = panel.querySelector(".keyframe-list .kfc-field input[type='range']") as HTMLInputElement | null;
      expect(strength).not.toBeNull();
      expect(strength).toBeDisabled();
      expect(within(panel).getByText(/does not use keyframe strength/)).toBeInTheDocument();
    },
    15_000,
  );

  it(
    "MiniMax H3: the submitted body carries the three sampling values, fps 24, a 17n+5 length and no NAG",
    async () => {
      const bridge = await renderApp(appBridge);
      const requestSpy = vi.spyOn(bridge, "request");
      const user = userEvent.setup();

      // Turn NAG ON while LTX is loaded — the H3 request must still carry none
      // of it (the accordion greys and the shell stops sending it).
      await user.click(screen.getByText("Negative Prompt"));
      await user.click(screen.getByRole("checkbox", { name: /non-cfg negative/i }));

      await switchTo("MiniMaxH3");
      await waitFor(() => expect(within(singlePanel()).getByText("Steps")).toBeInTheDocument());

      await user.type(screen.getByPlaceholderText(/describe the video/i), "a cat riding a skateboard");
      const generate = screen.getByRole("button", { name: /^generate$/i });
      await waitFor(() => expect(generate).not.toBeDisabled());
      await user.click(generate);

      await waitFor(() =>
        expect(
          requestSpy.mock.calls.some(
            (c) => c[0] === "backend.request" && (c[1] as { path?: string }).path === "/api/v1/generate",
          ),
        ).toBe(true),
      );
      const call = requestSpy.mock.calls.find(
        (c) => c[0] === "backend.request" && (c[1] as { path?: string }).path === "/api/v1/generate",
      );
      const body = (call?.[1] as { body?: Record<string, unknown> } | undefined)?.body ?? {};
      expect(body.num_inference_steps).toBe(20);
      expect(body.shift_video).toBe(12);
      expect(body.shift_audio).toBe(3);
      expect(body.frame_rate).toBe(24);
      expect(((body.num_frames as number) - 5) % 17).toBe(0);
      expect((body.width as number) % 32).toBe(0);
      expect(body.nag_enabled).toBeUndefined();
      expect(body.negative_prompt).toBeUndefined();
      // The fixture validated it against H3's own rules and accepted it.
      await screen.findByText(/generating/i);
    },
    20_000,
  );

  it(
    "code review m-8: NAG on with an EMPTY body on LTX does not block Generate once on MiniMax H3",
    async () => {
      await renderApp();
      const user = userEvent.setup();
      await user.click(screen.getByText("Negative Prompt"));
      await user.click(screen.getByRole("checkbox", { name: /non-cfg negative/i }));
      await user.clear(screen.getByRole("textbox", { name: /negative prompt/i }));
      await user.type(screen.getByPlaceholderText(/describe the video/i), "a cat riding a skateboard");
      const generate = screen.getByRole("button", { name: /^generate$/i });
      // On LTX the empty body blocks Generate…
      await waitFor(() => expect(generate).toBeDisabled());
      // …and on MiniMax H3 (NAG closed) it no longer does.
      await switchTo("MiniMaxH3");
      await waitFor(() => expect(generate).not.toBeDisabled());
    },
    20_000,
  );

  it(
    "switching back to LTX 2.3 restores the placeholder, the marker and the 8n+1 grid",
    async () => {
      await renderApp();
      await switchTo("MiniMaxH3");
      await waitFor(() => expect(within(singlePanel()).getByText("Steps")).toBeInTheDocument());
      await switchTo("LTX23");
      await waitFor(() => expect(within(singlePanel()).getByText("CFG scale")).toBeInTheDocument());
      expect(within(singlePanel()).queryByText("Steps")).not.toBeInTheDocument();
      await waitFor(() => expect(createMarker()).not.toBeNull());
      expect(durationNumber().step).toBe("8");
      expect((Number(durationNumber().value) - 1) % 8).toBe(0);
      expect(within(singlePanel()).getByRole("spinbutton", { name: /fps/i })).not.toBeDisabled();
      expect(screen.getByRole("tab", { name: "Chained" })).not.toBeDisabled();
      expect(screen.getByRole("tab", { name: "Edit" })).not.toBeDisabled();
    },
    20_000,
  );
});
