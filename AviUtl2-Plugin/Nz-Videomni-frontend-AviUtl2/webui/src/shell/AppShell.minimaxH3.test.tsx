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
//  - hide Create's reference video and the keyframe strength, grey its A2V
//    card and the Negative Prompt accordion (台帳 §1-97: keyframes at any
//    frame, so the second keyframe stays open),
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

/** Adds `count` keyframe cards through Create's "Add keyframe" button and
 * fills each one with its own 📸 capture, so every card ends up "ready". The
 * capture and the upload are two fixture round trips (200 ms each on the
 * app-wide fixture), hence the longer wait. */
async function addCapturedKeyframes(user: ReturnType<typeof userEvent.setup>, panel: HTMLElement, count: number) {
  for (let n = 1; n <= count; n++) {
    await user.click(within(panel).getByRole("button", { name: /add keyframe/i }));
    await waitFor(() => expect(panel.querySelectorAll(".keyframe-list .kfc-card")).toHaveLength(n));
    const empty = Array.from(panel.querySelectorAll<HTMLElement>(".keyframe-list .kfc-card")).find((card) =>
      within(card).queryByText(/no image yet/i),
    );
    if (!empty) throw new Error("no empty card to capture into");
    await user.click(within(empty).getByRole("button", { name: /capture/i }));
    // Ready = neither the empty hint nor the upload spinner (the card goes
    // "uploading" the moment the capture starts, so the hint alone clears early).
    await waitFor(
      () => {
        expect(within(panel).queryByText(/no image yet/i)).not.toBeInTheDocument();
        expect(panel.querySelector(".keyframe-list .kfc-spinner")).toBeNull();
      },
      { timeout: 5_000 },
    );
  }
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
    "LTX 2.3: a keyframe card shows its strength row",
    async () => {
      await renderApp();
      const user = userEvent.setup();
      const panel = singlePanel();
      await user.click(within(panel).getByRole("button", { name: /add keyframe/i }));
      await waitFor(() => expect(panel.querySelectorAll(".keyframe-list .kfc-card")).toHaveLength(1));
      const strength = panel.querySelector(".keyframe-list .kfc-field input[type='range']");
      expect(strength).not.toBeNull();
      expect(strength).not.toBeDisabled();
    },
    15_000,
  );

  it(
    "MiniMax H3 (台帳 §1-97): a second keyframe can be added, and no card has a strength row",
    async () => {
      await renderApp();
      await switchTo("MiniMaxH3");
      await waitFor(() => expect(within(singlePanel()).getByText("Steps")).toBeInTheDocument());
      const user = userEvent.setup();
      const panel = singlePanel();
      await addCapturedKeyframes(user, panel, 2);
      // The strength row is not greyed but absent.
      expect(panel.querySelector(".keyframe-list input[type='range']")).toBeNull();
      expect(within(panel).queryByText("Strength")).not.toBeInTheDocument();
      expect(within(panel).getByRole("button", { name: /add keyframe/i })).not.toBeDisabled();
    },
    20_000,
  );

  it(
    "台帳 §1-104: LTX 2.3 keeps the chip's audio toggle and sends the tag's audio strength",
    async () => {
      // The app-wide fixture (Create submits through the singleton client).
      // The generate call is recorded and answered with a 422 here instead of
      // reaching the fixture, so no job is left running for the next test on
      // this same fixture (a running job would refuse its base-model switch).
      const bridge = await renderApp(appBridge);
      const original = bridge.request.bind(bridge);
      const requestSpy = vi.spyOn(bridge, "request").mockImplementation(((method: string, params: unknown) => {
        if (method === "backend.request" && (params as { path?: string }).path === "/api/v1/generate") {
          return Promise.resolve({
            status: 422,
            body: { error: { code: "VALIDATION_ERROR", message: "recorded by the test", detail: [] } },
          });
        }
        return (original as (m: string, p: unknown) => Promise<unknown>)(method, params);
      }) as typeof bridge.request);
      const user = userEvent.setup();

      await user.type(screen.getByPlaceholderText(/describe the video/i), "a cat <lora:Pixar_Toon:1.0:0.3>");
      const chip = document.querySelector('.lora-chip[data-lora-name="Pixar_Toon"]') as HTMLElement | null;
      expect(chip?.querySelector(".lora-chip-audio-toggle")).not.toBeNull();
      const generate = screen.getByRole("button", { name: /^generate$/i });
      await waitFor(() => expect(generate).not.toBeDisabled());
      await user.click(generate);

      const isGenerate = (c: unknown[]) =>
        c[0] === "backend.request" && (c[1] as { path?: string }).path === "/api/v1/generate";
      await waitFor(() => expect(requestSpy.mock.calls.some(isGenerate)).toBe(true));
      const call = requestSpy.mock.calls.find(isGenerate);
      const body = (call?.[1] as { body?: Record<string, unknown> } | undefined)?.body ?? {};
      expect(body.loras).toEqual([{ name: "Pixar_Toon", strength: 1, audio_strength: 0.3 }]);
    },
    20_000,
  );

  it(
    "MiniMax H3: the submitted body carries the three sampling values, fps 24, a 17n+5 length, no NAG, no keyframe strength and no LoRA audio strength",
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
      // 台帳 §1-97: two keyframes, whose `strength` key must not be sent.
      await addCapturedKeyframes(user, singlePanel(), 2);

      // 台帳 §1-104: a Library-style 3-argument tag — its chip shows no audio
      // side on H3, and the third number is not sent.
      await user.type(
        screen.getByPlaceholderText(/describe the video/i),
        "a cat riding a skateboard <lora:Pixar_Toon:1.0:1.0>",
      );
      const chip = document.querySelector('.lora-chip[data-lora-name="Pixar_Toon"]') as HTMLElement | null;
      expect(chip).not.toBeNull();
      expect(chip?.querySelector(".lora-chip-audio-toggle")).toBeNull();
      expect(chip?.querySelector(".lora-chip-video-icon")).toBeNull();
      expect(chip?.querySelector(".lora-chip-audio")).toBeNull();
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
      const images = body.conditioning_images as Record<string, unknown>[];
      expect(images).toHaveLength(2);
      expect(images[0]?.frame_idx).toBe(0);
      for (const image of images) expect(image).not.toHaveProperty("strength");
      const loras = body.loras as Record<string, unknown>[];
      expect(loras).toEqual([{ name: "Pixar_Toon", strength: 1 }]);
      expect(loras[0]).not.toHaveProperty("audio_strength");
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
    "台帳 §1-104: an unsupported LoRA in the prompt shows the yellow note, and Generate stays enabled",
    async () => {
      await renderApp();
      await switchTo("MiniMaxH3");
      await waitFor(() => expect(within(singlePanel()).getByText("Steps")).toBeInTheDocument());
      const user = userEvent.setup();
      const note = "This LoRA's format is not supported. It may have no effect.";

      await user.type(screen.getByPlaceholderText(/describe the video/i), "a cat <lora:Pixar_Toon:1.0>");
      const generate = screen.getByRole("button", { name: /^generate$/i });
      await waitFor(() => expect(generate).not.toBeDisabled());
      expect(screen.queryByText(note)).not.toBeInTheDocument();

      // The fixture marks `LTX-2.3-Henshin` `supported: false`.
      await user.type(screen.getByPlaceholderText(/describe the video/i), " <lora:LTX-2.3-Henshin:1.0>");
      const banner = await within(singlePanel()).findByText(note);
      expect(banner).toHaveAttribute("role", "status");
      expect(banner).toHaveClass("warning-banner", "warning-banner-mild");
      expect(generate).not.toBeDisabled();
    },
    20_000,
  );

  it(
    "台帳 §1-104 (plan §7-6): the unsupported-format note shows on LTX 2.3 too, Generate still enabled",
    async () => {
      await renderApp();
      const user = userEvent.setup();
      await user.type(screen.getByPlaceholderText(/describe the video/i), "a cat <lora:LTX-2.3-Henshin:1.0>");
      const banner = await within(singlePanel()).findByText(
        "This LoRA's format is not supported. It may have no effect.",
      );
      expect(banner).toHaveAttribute("role", "status");
      await waitFor(() => expect(screen.getByRole("button", { name: /^generate$/i })).not.toBeDisabled());
    },
    20_000,
  );

  it(
    "台帳 §1-104: a completed base-model switch refetches GET /loras (the list follows the base model)",
    async () => {
      const bridge = await renderApp();
      const requestSpy = vi.spyOn(bridge, "request");
      const loraCalls = () =>
        requestSpy.mock.calls.filter(
          (c) =>
            c[0] === "backend.request" &&
            (c[1] as { method?: string; path?: string }).method === "GET" &&
            (c[1] as { path?: string }).path === "/api/v1/loras",
        ).length;
      expect(loraCalls()).toBe(0);

      await switchTo("MiniMaxH3");
      await waitFor(() => expect(within(singlePanel()).getByText("Steps")).toBeInTheDocument());
      await waitFor(() => expect(loraCalls()).toBeGreaterThan(0));
      // …and what came back is H3's directory, not LTX's.
      const index = requestSpy.mock.calls.findIndex(
        (c) => c[0] === "backend.request" && (c[1] as { path?: string }).path === "/api/v1/loras",
      );
      const result = (await requestSpy.mock.results[index]?.value) as { body?: { lora_dir?: string } };
      expect(result.body?.lora_dir).toBe("models/MiniMaxH3/LoRA");
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
