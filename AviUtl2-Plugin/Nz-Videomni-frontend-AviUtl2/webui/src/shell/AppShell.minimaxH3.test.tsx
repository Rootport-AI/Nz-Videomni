import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { bridge as appBridge, TIMELINE_MENU_INVOKED_EVENT } from "../bridge";
import type { ResultOf } from "../bridge";
import { resetCreateLiveCommands } from "../timeline/createLiveCommands";
import { resetExtractObjectAudioChain } from "../timeline/extractObjectAudio";
import { resetProvisionalReservation } from "../timeline/provisionalReservation";
import { createMockBridge } from "../bridge/mockBridge";
import type { NativeBridge } from "../bridge";
import type { ModelsResponse } from "../api/types";
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
  await waitFor(() => expect(select.value).toBe(id), { timeout: 5_000 });
}

describe("AppShell — MiniMax H3 (第 2 弾, 2c-G1)", () => {
  beforeEach(() => {
    window.localStorage.clear();
    resetExtractObjectAudioChain();
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

      // The Edit tab greys — see the dedicated case below for the four
      // sub-tabs behind it. §1-106: the Chained tab stays live now (H3
      // chains from text; see the Chained block at the end of this file).
      expect(screen.getByRole("tab", { name: "Chained" })).not.toBeDisabled();
      expect(screen.getByRole("tab", { name: "Edit" })).toBeDisabled();
      expect(screen.getByRole("tab", { name: "Single" })).not.toBeDisabled();

      // Reference video hidden. §1-107: H3 publishes `reference`, so the A2V
      // accordion gives way to the Reference accordion, whose Audios cards carry
      // the "Audio to video" tick; ticking one brings up the Strength slider
      // (served `constraints.source_audio`) under that card.
      expect(within(panel).queryByText("Reference video (IC-LoRA)")).not.toBeInTheDocument();
      expect(within(panel).queryByText("Audio to video (A2V)")).not.toBeInTheDocument();
      expect(within(panel).getByText("Reference", { selector: "summary" })).toBeInTheDocument();
      const audioCard = within(panel).getByTestId("ref-slot-audio-1");
      expect(within(audioCard).queryByTestId("source-audio-strength")).not.toBeInTheDocument();
      await userEvent.setup().click(within(audioCard).getByRole("checkbox", { name: "Audio to video", hidden: true }));
      expect(within(audioCard).getByTestId("source-audio-strength")).toBeInTheDocument();

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
      // Retake is a chain request it refuses). With Outpainting / Inpainting / Alpha Gen closed too,
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
      // (Scoped to the keyframe cards: the A2V card has its own Strength slider.)
      expect(within(panel.querySelector(".keyframe-list") as HTMLElement).queryByText("Strength")).not.toBeInTheDocument();
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
      await waitFor(() => expect(generate).not.toBeDisabled(), { timeout: 5_000 });
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
      await waitFor(() => expect(generate).not.toBeDisabled(), { timeout: 5_000 });
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

  // ── §1-107: right-click #2 / #7 / #3 on MiniMax H3 ──────────────
  describe("§1-107 right-click into the Reference accordion", () => {
    beforeEach(() => {
      resetProvisionalReservation();
      resetCreateLiveCommands();
      resetExtractObjectAudioChain();
    });

    /** A video object whose ribbon is frames 30..119 @30fps of a 10.7 s file,
     * played from 1.0 s to 4.0 s — `decideSourceTrim` cuts a 3.000 s window. */
    function trimmedVideoSelection(filePath: string): ResultOf<"timeline.getSelection"> {
      return {
        hasRange: true,
        rangeStart: 30,
        rangeEnd: 119,
        selected: [
          {
            layer: 1,
            frameStart: 30,
            frameEnd: 119,
            effectName: "動画ファイル",
            filePath,
            objectName: "a",
            textContent: null,
            mediaWidth: 1280,
            mediaHeight: 720,
            mediaDurationSec: 10.7,
            hasPlaybackRange: true,
            playbackStartSec: 1.0,
            playbackEndSec: 4.0,
          },
        ],
        cursorFrame: 0,
        cursorLayer: 1,
        rate: 30,
        scale: 1,
        sampleRate: 44100,
      };
    }

    function audioSelection(filePath: string): ResultOf<"timeline.getSelection"> {
      const sel = trimmedVideoSelection(filePath);
      return {
        ...sel,
        selected: [{ ...sel.selected[0]!, effectName: "音声ファイル", mediaWidth: 0, mediaHeight: 0 }],
      };
    }

    const card = (id: string) => within(singlePanel()).getByTestId(id);

    async function waitForCardReady(id: string) {
      await waitFor(
        () => {
          expect(within(card(id)).getByRole("button", { name: /add this tag/i, hidden: true })).toBeEnabled();
          expect(card(id).querySelector(".keyframe-spinner")).toBeNull();
        },
        { timeout: 5_000 },
      );
    }

    it(
      "H3: #2 and #7 fill the first free card without remounting Create (prompt and other cards survive); #2 carries the trimmed length",
      async () => {
        const bridge = createMockBridge({ delayMs: 0 });
        const requestSpy = vi.spyOn(bridge, "request");
        await renderApp(bridge);
        await switchTo("MiniMaxH3");
        await waitFor(() => expect(card("ref-slot-video-1")).toBeInTheDocument());
        const user = userEvent.setup();
        await user.type(screen.getByPlaceholderText(/describe the video/i), "a cat on a skateboard");

        // #2 → Videos card 1, trimmed to the ribbon; the card shows the trimmed 3.0 s.
        act(() => {
          bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
            action: "referenceVideo",
            selection: trimmedVideoSelection("C:\\v\\one.mp4"),
          });
        });
        await waitForCardReady("ref-slot-video-1");
        expect(within(card("ref-slot-video-1")).getByText("one.mp4")).toBeInTheDocument();
        expect(within(card("ref-slot-video-1")).getByText("3.0s")).toBeInTheDocument();
        expect(requestSpy).toHaveBeenCalledWith("backend.uploadFile", {
          kind: "video",
          filePath: "C:\\v\\one.mp4",
          query: { trim_start_sec: "1.000", trim_duration_sec: "3.000" },
        });

        // A second #2 lands in card 2; card 1 is untouched.
        act(() => {
          bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
            action: "referenceVideo",
            selection: trimmedVideoSelection("C:\\v\\two.mp4"),
          });
        });
        await waitForCardReady("ref-slot-video-2");
        expect(within(card("ref-slot-video-2")).getByText("two.mp4")).toBeInTheDocument();
        expect(within(card("ref-slot-video-1")).getByText("one.mp4")).toBeInTheDocument();

        // #7 → the object's ribbon range is extracted (solo) and the wav lands in
        // Audios card 1 (台帳 §1-113).
        act(() => {
          bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
            action: "audioToVideo",
            selection: audioSelection("C:\\a\\voice.wav"),
          });
        });
        await waitForCardReady("ref-slot-audio-1");
        expect(requestSpy).toHaveBeenCalledWith("timeline.extractAudio", {
          layer: 1,
          frameStart: 30,
          frameCount: 90,
          audioMode: "solo",
          soloKeepLayers: [1],
        });
        expect(within(card("ref-slot-audio-1")).getByText(/^extract_1_30_\d+\.wav$/)).toBeInTheDocument();
        expect(screen.getByText(/loaded the object's sound/i)).toBeInTheDocument();
        expect(within(card("ref-slot-video-1")).getByText("one.mp4")).toBeInTheDocument();
        expect((screen.getByPlaceholderText(/describe the video/i) as HTMLTextAreaElement).value).toBe(
          "a cat on a skateboard",
        );
        // Enable is left off (owner ruling 14); still on Single.
        expect(within(singlePanel()).getByRole("checkbox", { name: "Enable", hidden: true })).not.toBeChecked();
        expect(screen.getByRole("tab", { name: "Single" })).toHaveAttribute("aria-selected", "true");
      },
      25_000,
    );

    it(
      "H3 → LTX 2.3 → H3: the Reference accordion goes away on LTX and the filled card is still there on return",
      async () => {
        const bridge = createMockBridge({ delayMs: 0 });
        await renderApp(bridge);
        await switchTo("MiniMaxH3");
        await waitFor(() => expect(card("ref-slot-audio-1")).toBeInTheDocument());
        act(() => {
          bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
            action: "audioToVideo",
            selection: audioSelection("C:\\a\\voice.wav"),
          });
        });
        await waitForCardReady("ref-slot-audio-1");

        await switchTo("LTX23");
        await waitFor(() =>
          expect(within(singlePanel()).queryByTestId("ref-slot-audio-1")).not.toBeInTheDocument(),
        );
        expect(within(singlePanel()).queryByText("Reference", { selector: "summary" })).not.toBeInTheDocument();

        await switchTo("MiniMaxH3");
        await waitForCardReady("ref-slot-audio-1");
        expect(within(card("ref-slot-audio-1")).getByText(/^extract_1_30_\d+\.wav$/)).toBeInTheDocument();
        expect(screen.getByText(/loaded the object's sound/i)).toBeInTheDocument();
      },
      25_000,
    );

    it(
      "H3: #7 on a VIDEO object extracts its own sound and lands in the Audios card (never the Videos card)",
      async () => {
        const bridge = createMockBridge({ delayMs: 0 });
        const requestSpy = vi.spyOn(bridge, "request");
        await renderApp(bridge);
        await switchTo("MiniMaxH3");
        await waitFor(() => expect(card("ref-slot-audio-1")).toBeInTheDocument());

        act(() => {
          bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
            action: "audioToVideo",
            selection: trimmedVideoSelection("C:\\v\\one.mp4"),
          });
        });
        await waitForCardReady("ref-slot-audio-1");
        expect(within(card("ref-slot-audio-1")).getByText(/^extract_1_30_\d+\.wav$/)).toBeInTheDocument();
        expect(screen.getByText(/loaded the object's sound/i)).toBeInTheDocument();
        expect(requestSpy).toHaveBeenCalledWith(
          "backend.uploadFile",
          expect.objectContaining({ kind: "audio", filePath: expect.stringContaining("extract_1_30_") }),
        );
        expect(requestSpy).not.toHaveBeenCalledWith("backend.uploadFile", expect.objectContaining({ kind: "video" }));
      },
      20_000,
    );

    it(
      "H3: #7 shows the silence note and fills no card when the extraction is silent",
      async () => {
        const bridge = createMockBridge({
          delayMs: 0,
          failExtractAudio: "EXTRACT_FAILED",
          extractAudioErrorMessage: "no audio stream present (silent range)",
        });
        const requestSpy = vi.spyOn(bridge, "request");
        await renderApp(bridge);
        await switchTo("MiniMaxH3");
        await waitFor(() => expect(card("ref-slot-audio-1")).toBeInTheDocument());

        act(() => {
          bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
            action: "audioToVideo",
            selection: trimmedVideoSelection("C:\\v\\one.mp4"),
          });
        });
        await screen.findByText(/no audible sound/i, undefined, { timeout: 5_000 });
        expect(requestSpy).not.toHaveBeenCalledWith("backend.uploadFile", expect.anything());
      },
      20_000,
    );

    it(
      "H3: #4 keeps the Reference cards and the prompt, and reserves the seat at the panel's own DURATION (台帳 §1-114: no remount)",
      async () => {
        const bridge = createMockBridge({ delayMs: 0 });
        const requestSpy = vi.spyOn(bridge, "request");
        await renderApp(bridge);
        await switchTo("MiniMaxH3");
        await waitFor(() => expect(card("ref-slot-video-1")).toBeInTheDocument());
        const user = userEvent.setup();
        await user.type(screen.getByPlaceholderText(/describe the video/i), "a cat on a skateboard");

        act(() => {
          bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
            action: "referenceVideo",
            selection: trimmedVideoSelection("C:\\v\\one.mp4"),
          });
        });
        await waitForCardReady("ref-slot-video-1");

        // Hand-set DURATION to a value on H3's 17n+5 grid that no seed produces.
        // H3 has no comfort line, so #4's seed leaves DURATION alone and the
        // provisional must be reserved at the panel's own value.
        fireEvent.change(durationNumber(), { target: { value: "277" } });
        fireEvent.blur(durationNumber());
        await waitFor(() => expect(durationNumber().value).toBe("277"));

        // #4 on an image object: the image becomes the frame-0 keyframe, and the
        // Reference card and the prompt are untouched.
        const image = trimmedVideoSelection("C:\\i\\a.png");
        act(() => {
          bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
            action: "imageToVideo",
            selection: { ...image, selected: [{ ...image.selected[0]!, effectName: "画像ファイル" }] },
          });
        });
        await screen.findByText(/loaded from a right-click: a\.png/i, undefined, { timeout: 5_000 });
        await waitFor(() => expect(within(singlePanel()).getByAltText("a.png")).toBeInTheDocument(), {
          timeout: 5_000,
        });
        expect(within(card("ref-slot-video-1")).getByText("one.mp4")).toBeInTheDocument();
        expect((screen.getByPlaceholderText(/describe the video/i) as HTMLTextAreaElement).value).toBe(
          "a cat on a skateboard",
        );
        expect(screen.getByRole("tab", { name: "Single" })).toHaveAttribute("aria-selected", "true");
        await waitFor(() =>
          expect(requestSpy).toHaveBeenCalledWith(
            "timeline.insertProvisional",
            expect.objectContaining({ placement: "A", numFrames: 277 }),
          ),
        );
        expect(durationNumber().value).toBe("277");
      },
      25_000,
    );

    it(
      "LTX 2.3: #2 still takes the IC-LoRA road (no Reference accordion)",
      async () => {
        const bridge = createMockBridge({ delayMs: 0 });
        await renderApp(bridge);
        act(() => {
          bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
            action: "referenceVideo",
            selection: trimmedVideoSelection("C:\\v\\one.mp4"),
          });
        });
        await screen.findByText(/ic-lora is \/generate-only/i, {}, { timeout: 5_000 });
        expect(within(singlePanel()).queryByText("Reference", { selector: "summary" })).not.toBeInTheDocument();
        expect(within(singlePanel()).queryByTestId("ref-slot-video-1")).not.toBeInTheDocument();
      },
      20_000,
    );
  });
});

// ── §1-106: MiniMax H3's Chained tab, from the fixture's GET /models alone ──
// No engine id is read anywhere: the screen follows `constraints` (17n+5 clips,
// 24fps, the sampling sliders, `chain.overlap_frames`) and the booleans
// `unsupported_features` turns into (v2v / end_source / a2v / reference_video /
// chain_start_image).
describe("AppShell — MiniMax H3 Chained (§1-106)", () => {
  beforeEach(() => {
    window.localStorage.clear();
    resetProvisionalReservation();
    resetCreateLiveCommands();
  });
  afterEach(() => {
    window.localStorage.clear();
    vi.restoreAllMocks();
  });

  async function openChained() {
    await userEvent.setup().click(screen.getByRole("tab", { name: "Chained" }));
    await waitFor(() => expect(within(singlePanel()).getByText("Clips")).toBeInTheDocument());
    return singlePanel();
  }

  /** Every clip card's DURATION number input, in order. */
  function clipDurations(panel: HTMLElement): HTMLInputElement[] {
    return Array.from(panel.querySelectorAll<HTMLInputElement>(".clip-card input[type='number']"));
  }

  it(
    "LTX 2.3 keeps its Chained screen: seam blend, chunked upsample, stage-2 window, CFG row, 8n+1 clips",
    async () => {
      await renderApp();
      const panel = await openChained();
      expect(within(panel).getByText("Seam blend width")).toBeInTheDocument();
      expect(within(panel).getByText("Chunked upsample")).toBeInTheDocument();
      expect(within(panel).getByText("CFG scale")).toBeInTheDocument();
      expect(within(panel).queryByText("Steps")).not.toBeInTheDocument();
      expect(within(panel).getByText(/Total input:/)).toBeInTheDocument();
      expect(within(panel).getByText("Batch i2v-long (prototype)")).toBeInTheDocument();
      for (const input of clipDurations(panel)) expect(input.step).toBe("8");
      expect(panel.querySelector(".clip-note")).toBeNull();
    },
    20_000,
  );

  it(
    "MiniMax H3: the Chained tab opens in the fixed-overlap mode and closes the material it cannot take",
    async () => {
      await renderApp();
      await switchTo("MiniMaxH3");
      await waitFor(() => expect(screen.getByRole("tab", { name: "Chained" })).not.toBeDisabled());
      const panel = await openChained();

      // Fixed-overlap mode: no seam blend, no chunked upsample, no stage-2 window,
      // no "Total input / Est.", and the sampling sliders in place of the CFG row.
      await waitFor(() => expect(within(panel).getByText("Steps")).toBeInTheDocument());
      expect(within(panel).queryByText("Seam blend width")).not.toBeInTheDocument();
      expect(within(panel).queryByText("Seam blend strength")).not.toBeInTheDocument();
      expect(within(panel).queryByText("Chunked upsample")).not.toBeInTheDocument();
      expect(within(panel).queryByText("CFG scale")).not.toBeInTheDocument();
      expect(within(panel).queryByText(/Total input:/)).not.toBeInTheDocument();
      expect(within(panel).queryByText(/Est\./)).not.toBeInTheDocument();

      // 17n+5 clips with a 17 step; fps pinned at 24.
      const durations = clipDurations(panel);
      expect(durations).toHaveLength(2);
      for (const input of durations) {
        expect(input.step).toBe("17");
        expect((Number(input.value) - 5) % 17).toBe(0);
      }
      const fps = within(panel).getByRole("spinbutton", { name: /fps/i }) as HTMLInputElement;
      expect(fps).toBeDisabled();
      expect(fps.value).toBe("24");

      // The material it cannot take: the opening image (the SOURCE card greys
      // whole) and Batch i2v-long (not rendered); the three accordions grey.
      expect(within(panel).getByText(/An opening image for clip 1 cannot be used/)).toBeInTheDocument();
      expect(within(panel).queryByText("Batch i2v-long (prototype)")).not.toBeInTheDocument();
      expect(within(panel).getByText(/End source is not available/)).toBeInTheDocument();
      expect(within(panel).getByText(/Reference video \(control IC-LoRA\) is not available/)).toBeInTheDocument();
      expect(within(panel).getByText(/Generating from an audio track \(A2V\) is not available/)).toBeInTheDocument();
    },
    20_000,
  );

  it(
    "MiniMax H3: the audio note follows the seam (141 shows it, 124 does not) and the output length is Σ − 22 × (n − 1)",
    async () => {
      await renderApp();
      await switchTo("MiniMaxH3");
      const panel = await openChained();
      await waitFor(() => expect(within(panel).getByText("Steps")).toBeInTheDocument());
      const [first, second] = clipDurations(panel);
      fireEvent.change(first!, { target: { value: "124" } });
      fireEvent.change(second!, { target: { value: "124" } });
      await waitFor(() => expect(first!.value).toBe("124"));
      expect(panel.querySelector(".clip-note")).toBeNull();
      // 124 + 124 − 22 = 226 frames.
      expect(within(panel).getByText(/\(226 frames\)/)).toBeInTheDocument();

      fireEvent.change(first!, { target: { value: "141" } });
      await waitFor(() => expect(panel.querySelectorAll(".clip-note")).toHaveLength(1));
      const note = panel.querySelector(".clip-note") as HTMLElement;
      expect(note.textContent).toBe("The audio may drift slightly.");
      // On the SECOND card — the seam in front of it.
      expect(panel.querySelectorAll(".clip-card")[1]?.contains(note)).toBe(true);
      expect(within(panel).getByText(/\(243 frames\)/)).toBeInTheDocument();
    },
    20_000,
  );

  it(
    "MiniMax H3: the chain request carries the sampling values and none of the four LTX seam fields",
    async () => {
      // The app-wide fixture is shared with the Create submit test above, which
      // leaves it on MiniMax H3 — put it back on LTX 2.3 so `renderApp` starts
      // where every other case does.
      await appBridge.request("backend.request", {
        method: "POST",
        path: "/api/v1/pipeline/load",
        body: { base_model: "LTX23" },
      });
      const bridge = await renderApp(appBridge);
      const requestSpy = vi.spyOn(bridge, "request");
      await switchTo("MiniMaxH3");
      const panel = await openChained();
      await waitFor(() => expect(within(panel).getByText("Steps")).toBeInTheDocument());
      const user = userEvent.setup();
      await user.type(
        screen.getByPlaceholderText(/describe the video/i),
        "a cat riding a skateboard <lora:Pixar_Toon:1.0:1.0>",
      );
      const generate = within(panel).getByRole("button", { name: /^generate$/i });
      await waitFor(() => expect(generate).not.toBeDisabled());
      await user.click(generate);

      const isChain = (c: unknown[]) =>
        c[0] === "backend.request" && (c[1] as { path?: string }).path === "/api/v1/generate/chain";
      await waitFor(() => expect(requestSpy.mock.calls.some(isChain)).toBe(true));
      const call = requestSpy.mock.calls.find(isChain);
      const body = (call?.[1] as { body?: Record<string, unknown> } | undefined)?.body ?? {};
      expect(body.num_inference_steps).toBe(20);
      expect(body.shift_video).toBe(12);
      expect(body.shift_audio).toBe(3);
      expect(body.frame_rate).toBe(24);
      expect((body.width as number) % 32).toBe(0);
      for (const key of ["overlap_frames", "overlap_strength", "stage2_window", "chunked_upsample"]) {
        expect(body).not.toHaveProperty(key);
      }
      for (const clip of body.clips as Record<string, unknown>[]) {
        expect(((clip.num_frames as number) - 5) % 17).toBe(0);
        expect(clip).not.toHaveProperty("conditioning_images");
      }
      expect(body.loras).toEqual([{ name: "Pixar_Toon", strength: 1 }]);
      // The fixture accepted it (no FEATURE_UNSUPPORTED card).
      await waitFor(() => expect(within(panel).queryByText("FEATURE_UNSUPPORTED")).not.toBeInTheDocument());
      // Leave the shared app-wide fixture on LTX 2.3 for whoever uses it next.
      await appBridge.request("backend.request", {
        method: "POST",
        path: "/api/v1/pipeline/load",
        body: { base_model: "LTX23" },
      });
    },
    20_000,
  );

  it(
    "MiniMax H3: Chained right-click entries into closed panels are refused with one note and nothing else",
    async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const requestSpy = vi.spyOn(bridge, "request");
      await renderApp(bridge);
      await switchTo("MiniMaxH3");
      await waitFor(() => expect(screen.getByRole("tab", { name: "Chained" })).not.toBeDisabled());
      const selection: ResultOf<"timeline.getSelection"> = {
        hasRange: false,
        rangeStart: 0,
        rangeEnd: 0,
        selected: [
          {
            layer: 1,
            frameStart: 0,
            frameEnd: 59,
            effectName: "動画ファイル",
            filePath: "C:\\v\\one.mp4",
            objectName: "a",
            textContent: null,
            mediaWidth: 1280,
            mediaHeight: 720,
            mediaDurationSec: 2,
            hasPlaybackRange: false,
            playbackStartSec: 0,
            playbackEndSec: 0,
          },
        ],
        cursorFrame: 0,
        cursorLayer: 1,
        rate: 30,
        scale: 1,
        sampleRate: 44100,
      };
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "extendVideo", selection });
      });
      await screen.findByText(/A source VIDEO cannot be used on the selected base model/, {}, { timeout: 5_000 });
      // Nothing else: still on Single, nothing uploaded, no provisional placed.
      expect(screen.getByRole("tab", { name: "Single" })).toHaveAttribute("aria-selected", "true");
      expect(requestSpy.mock.calls.some((c) => c[0] === "backend.uploadFile")).toBe(false);
      expect(requestSpy.mock.calls.some((c) => String(c[0]).startsWith("timeline.place"))).toBe(false);

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "currentFrameToClipChain", selection });
      });
      await screen.findByText(/An opening image for clip 1 cannot be used/, {}, { timeout: 5_000 });
      expect(requestSpy.mock.calls.some((c) => c[0] === "timeline.captureFrame")).toBe(false);
      expect(screen.getByRole("tab", { name: "Single" })).toHaveAttribute("aria-selected", "true");
    },
    20_000,
  );
});

// ── §1-109: the per-clip References on H3's Chained tab, end to end ──
// `transformerRole` reaches the Chained screen from `GET /models` (the
// fixture's H3 transformer is "default", so the role is patched into the
// response here): Ref2VA with References off shows ②, on with no card ③.
describe("AppShell — MiniMax H3 Chained References (§1-109)", () => {
  beforeEach(() => {
    window.localStorage.clear();
    resetProvisionalReservation();
    resetCreateLiveCommands();
  });
  afterEach(() => {
    window.localStorage.clear();
    vi.restoreAllMocks();
  });

  const REF2VA_NOTICE =
    "A Ref2VA model is loaded. When no references are used, an FL2VA model or a Hybrid model is more accurate.";
  const NONE_NOTICE = "There are no reference files.";

  it(
    "LTX 2.3 has no References checkbox; H3 (Ref2VA) has it off with ②, and on with ③",
    async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const realRequest = bridge.request.bind(bridge);
      vi.spyOn(bridge, "request").mockImplementation((async (method: string, params: unknown) => {
        const res = await (realRequest as (m: string, p: unknown) => Promise<unknown>)(method, params);
        if (method === "backend.request" && (params as { path?: string }).path === "/api/v1/models") {
          const body = (res as { body: ModelsResponse }).body;
          for (const block of body.base_models ?? []) {
            if (block.active && block.engine_family === "h3") block.transformer_role = "ref2va";
          }
        }
        return res;
      }) as typeof bridge.request);
      await renderApp(bridge);
      await userEvent.setup().click(screen.getByRole("tab", { name: "Chained" }));
      await waitFor(() => expect(within(singlePanel()).getByText("Clips")).toBeInTheDocument(), { timeout: 5_000 });
      expect(within(singlePanel()).queryByRole("checkbox", { name: "References" })).toBeNull();
      expect(within(singlePanel()).queryByText(REF2VA_NOTICE)).toBeNull();

      await switchTo("MiniMaxH3");
      const box = await within(singlePanel()).findByRole("checkbox", { name: "References" }, { timeout: 5_000 });
      expect(box).not.toBeChecked();
      await waitFor(() => expect(within(singlePanel()).getByText(REF2VA_NOTICE)).toBeInTheDocument(), { timeout: 5_000 });
      expect(within(singlePanel()).queryByText(NONE_NOTICE)).toBeNull();

      fireEvent.click(box);
      await waitFor(() => expect(within(singlePanel()).getByText(NONE_NOTICE)).toBeInTheDocument(), { timeout: 5_000 });
      expect(within(singlePanel()).queryByText(REF2VA_NOTICE)).toBeNull();
      expect(singlePanel().querySelectorAll(".clip-card .ref-accordion")).toHaveLength(2);
    },
    20_000,
  );
});
