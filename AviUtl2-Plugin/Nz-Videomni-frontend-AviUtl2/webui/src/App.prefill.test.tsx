import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMockBridge } from "./bridge/mockBridge";
import { bridge as singletonBridge, TIMELINE_MENU_INVOKED_EVENT, TIMELINE_PROJECT_LOADED_EVENT } from "./bridge";
import type { MockBridge, MockBridgeOptions, ResultOf } from "./bridge";
import { AppShell } from "./shell/AppShell";
import {
  PREFILL_SIZE_POLICY_STORAGE_KEY,
  PREFILL_FPS_POLICY_STORAGE_KEY,
} from "./shell/PrefillPolicyContext";
import {
  getChainDirty,
  getPublishedKeyframeCount,
  getPublishedSourceSlots,
  getReservationPhase,
  resetProvisionalReservation,
} from "./timeline/provisionalReservation";
import { resetCreateLiveCommands } from "./timeline/createLiveCommands";
import { END_SOURCE_SEED_MAX_FRAMES } from "./timeline/prefillSeed";
import { END_SOURCE_CONTEXT_FRAMES } from "./timeline/tailAlign";

// Integration coverage for the right-click "routing -> prefill" host wiring
// (task: ルーティング→プリフィル). A mock bridge is injected into `AppShell`
// so `useMenuRouter` subscribes to *it*; emitting `timeline.menuInvoked`
// through that bridge should switch mode and prefill the target form's
// resolution from `selection.mediaWidth/Height` (via `deriveGenerationParams`).
//
// (`useConfig`/`useServerStatus` still talk to the app-wide singleton mock —
// jsdom has no WebView2 host — which is what supplies the form's `/config`
// limits; only the menu-routing seam is exercised against the injected bridge.)

function selectionWith(mediaWidth: number, mediaHeight: number): ResultOf<"timeline.getSelection"> {
  return {
    hasRange: true,
    rangeStart: 0,
    rangeEnd: 120,
    selected: [
      {
        layer: 1,
        frameStart: 0,
        frameEnd: 120,
        effectName: "動画ファイル",
        filePath: "C:\\v\\a.mp4",
        objectName: "a",
        textContent: null,
        mediaWidth,
        mediaHeight,
        mediaDurationSec: 0,
      },
    ],
    cursorFrame: 0,
    cursorLayer: 1,
    rate: 30,
    scale: 1,
    sampleRate: 44100,
  };
}

// A layer-menu selection (no selected object; ✨/📷 generate at a cursor
// position rather than off an existing object). `cursorLayer`/`cursorFrame` are
// SDK-0-based; the ✨ G1 toast echoes them 1-based (L=+1, F=+1).
function layerSelection(cursorLayer: number, cursorFrame: number): ResultOf<"timeline.getSelection"> {
  return {
    hasRange: false,
    rangeStart: 0,
    rangeEnd: 0,
    selected: [],
    cursorFrame,
    cursorLayer,
    rate: 30,
    scale: 1,
    sampleRate: 44100,
  };
}

// An image object selection (for the image-required items #4/#5/#6). `layer`/
// `frameStart`/`frameEnd` are the material range placement A/B derive from.
function imageSelection(
  layer = 2,
  frameStart = 10,
  frameEnd = 130,
): ResultOf<"timeline.getSelection"> {
  return {
    hasRange: true,
    rangeStart: frameStart,
    rangeEnd: frameEnd,
    selected: [
      {
        layer,
        frameStart,
        frameEnd,
        effectName: "画像ファイル",
        filePath: "C:\\i\\a.png",
        objectName: "a",
        textContent: null,
        mediaWidth: 800,
        mediaHeight: 600,
        mediaDurationSec: 0,
      },
    ],
    cursorFrame: 0,
    cursorLayer: 0,
    rate: 30,
    scale: 1,
    sampleRate: 44100,
  };
}

// A text object selection (for #8 append-text). Text objects have no backing
// file (`filePath: null`), so `resolveMenuSelection` re-queries `getSelection`;
// tests pass this same object as the mock's `selection` so the re-query returns
// the text body. `body` defaults to "hello"; pass "" for the empty-body branch.
function textSelection(body = "hello"): ResultOf<"timeline.getSelection"> {
  const sel = imageSelection();
  return {
    ...sel,
    selected: [{ ...sel.selected[0]!, effectName: "テキスト", filePath: null, textContent: body }],
  };
}

// The shared prompt textarea (owned by AppShell's PromptBar, above the tabs).
// Queried by its unique placeholder — the wrapping <label> folds the char-count
// hint into the accessible name, so a name match isn't exact.
const promptBox = () => screen.getByPlaceholderText(/describe the video/i) as HTMLTextAreaElement;

// Two selected objects (single-selection-only guard, §4-2).
function multiSelection(): ResultOf<"timeline.getSelection"> {
  const one = imageSelection().selected[0]!;
  return { ...imageSelection(), selected: [one, { ...one, layer: 3 }] };
}

// An audio object selection (for #7 audio-to-video).
function audioSelection(): ResultOf<"timeline.getSelection"> {
  const sel = imageSelection();
  return {
    ...sel,
    selected: [
      {
        ...sel.selected[0]!,
        effectName: "音声ファイル",
        filePath: "C:\\a\\voice.wav",
        mediaWidth: 0,
        mediaHeight: 0,
      },
    ],
  };
}

// W8: an image object of a specific media resolution (for #4 imageToVideo's
// DURATION comfort-ceiling seed). `mediaDurationSec` is irrelevant to the
// comfortCeiling policy, so it's left 0.
function imageSelectionSized(mediaWidth: number, mediaHeight: number): ResultOf<"timeline.getSelection"> {
  const sel = imageSelection();
  return { ...sel, selected: [{ ...sel.selected[0]!, mediaWidth, mediaHeight, mediaDurationSec: 0 }] };
}

// W8: a video object with an explicit media resolution + real-time duration +
// project rate/scale (for #2 referenceVideo's materialClampedToCeiling seed —
// the DURATION derives from `mediaDurationSec * (rate/scale)`).
function videoSelectionSized(
  mediaWidth: number,
  mediaHeight: number,
  mediaDurationSec: number,
  rate = 24,
  scale = 1,
): ResultOf<"timeline.getSelection"> {
  return {
    hasRange: true,
    rangeStart: 0,
    rangeEnd: 120,
    selected: [
      {
        layer: 1,
        frameStart: 0,
        frameEnd: 120,
        effectName: "動画ファイル",
        filePath: "C:\\v\\a.mp4",
        objectName: "a",
        textContent: null,
        mediaWidth,
        mediaHeight,
        mediaDurationSec,
      },
    ],
    cursorFrame: 0,
    cursorLayer: 1,
    rate,
    scale,
    sampleRate: 44100,
  };
}

// Every mode screen is mounted at once now (only the active one visible), and
// Create/Chain both carry width/height inputs that can share a default value,
// so display-value queries are scoped to the single visible panel via
// `within(panel())` — `getByRole("tabpanel")` skips the hidden screens.
const panel = () => screen.getByRole("tabpanel");

// W1/X1: pin the SIZE policy to "material" and the FPS policy to "defaults" so
// neither axis triggers the async getEditInfo overwrite — a few DURATION
// assertions here isolate the W4 span-based seed and would otherwise be a beat
// behind the project-fps recompute (a divergence covered by App.prefillPolicy.test.tsx).
// X1 note: the fps axis no longer offers "material" (a stored "material" would be
// coerced to the default "project", re-introducing the very async overwrite this
// helper avoids), so the fps axis is pinned to the other synchronous choice,
// "defaults". The config default fps (24) equals these selections' own rate (24),
// so the span-derived DURATION values below are unchanged by the switch.
function pinMaterialPolicies(): void {
  window.localStorage.setItem(PREFILL_SIZE_POLICY_STORAGE_KEY, "material");
  window.localStorage.setItem(PREFILL_FPS_POLICY_STORAGE_KEY, "defaults");
}

describe("App / right-click routing -> prefill", () => {
  // localStorage persists across tests within a file (jsdom is per-file), so
  // clear it before each test — otherwise a policy pinned by one test would leak
  // into the next. Cleared state = the product defaults (size material, fps project).
  beforeEach(() => {
    window.localStorage.clear();
  });
  afterEach(() => {
    window.localStorage.clear();
  });

  it(
    "referenceVideo routes to Create in IC-LoRA mode, prefilling resolution snapped to a multiple of 128",
    async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      render(<AppShell nativeBridge={bridge} />);

      // Wait for the Create form (default mode) to finish loading /config.
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // media 1000x700 with the 128 grid -> ceil to 1024 x 768.
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "referenceVideo",
          selection: selectionWith(1000, 700),
        });
      });

      await waitFor(
        () => {
          expect(within(panel()).getAllByDisplayValue("1024").length).toBeGreaterThan(0);
          expect(within(panel()).getAllByDisplayValue("768").length).toBeGreaterThan(0);
        },
        { timeout: 5_000 },
      );

      // The IC-LoRA reference-video block is visible with its /generate-only note.
      expect(screen.getByText(/ic-lora is \/generate-only/i)).toBeInTheDocument();
      // Still on Create (its tab stays selected).
      expect(screen.getByRole("tab", { name: /^single$/i })).toHaveAttribute("aria-selected", "true");
    },
    15_000,
  );

  it(
    "Y1: removing a #2-routed reference video relaxes IC-LoRA mode and restores Generate (was permanently blocked)",
    async () => {
      // Regression for Y1 (参照動画を外すとGenerate復帰不能になるバグの根治・案A):
      // a routed #2 used to FREEZE IC-LoRA mode at mount, so removing the
      // reference video left `isICLora` stuck true and the `referenceNotReady`
      // reason / disabled Generate permanent. Now the mode is dynamic, so
      // clearing the reference restores Generate.
      const bridge = createMockBridge({ delayMs: 0 });
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // A valid shared prompt so IC-LoRA state is the only remaining blocker.
      const user = userEvent.setup();
      await user.type(promptBox(), "a cat riding a skateboard");

      // Route #2 — media 1300x720 ceils on the 128 grid to 1408x768, which keeps
      // the default crop (1280x720) valid, and auto-loads the reference video.
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "referenceVideo",
          selection: selectionWith(1300, 720),
        });
      });

      // Reference video auto-loads to ready (IC-LoRA active). A lora-less
      // reference blocks Generate on `referenceNeedsLoras`.
      await waitFor(
        () => {
          expect(within(panel()).getAllByDisplayValue("1408").length).toBeGreaterThan(0);
        },
        { timeout: 5_000 },
      );
      const generateButton = () => screen.getByRole("button", { name: /^generate$/i }) as HTMLButtonElement;
      await waitFor(() => expect(generateButton()).toBeDisabled(), { timeout: 5_000 });

      // Remove the reference video via the ❌ "Remove IC-LoRA" button (clears the
      // reference + any control LoRA + strengths in one click).
      const removeButton = within(panel()).getByRole("button", { name: /remove ic-lora/i });
      await user.click(removeButton);

      // Generate is restored — the dynamic mode relaxed out of IC-LoRA, so the
      // reference gates cleared and nothing else blocks a plain T2V submit.
      await waitFor(() => expect(generateButton()).toBeEnabled(), { timeout: 5_000 });
    },
    20_000,
  );

  it(
    "extendVideo routes to Chain, prefilling the common resolution snapped to a multiple of 64 and auto-loading the source video (#1)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // media 800x600 with the general 64 grid -> ceil to 832 x 640.
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "extendVideo",
          selection: selectionWith(800, 600),
        });
      });

      // The Chain tab should become the active one.
      await waitFor(
        () => {
          expect(screen.getByRole("tab", { name: /^chained$/i })).toHaveAttribute("aria-selected", "true");
        },
        { timeout: 5_000 },
      );

      await waitFor(
        () => {
          expect(within(panel()).getAllByDisplayValue("832").length).toBeGreaterThan(0);
          expect(within(panel()).getAllByDisplayValue("640").length).toBeGreaterThan(0);
        },
        { timeout: 5_000 },
      );

      // #1 (I8 §3-4): the selected video's filePath is auto-uploaded into the
      // source_video slot.
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "backend.uploadFile",
            expect.objectContaining({ kind: "video", filePath: "C:\\v\\a.mp4" }),
          );
        },
        { timeout: 5_000 },
      );

      // §6 receipt note names the loaded file.
      expect(screen.getByText(/loaded from a right-click: a\.mp4/i)).toBeInTheDocument();
    },
    15_000,
  );

  it(
    "imageToClipChain routes to Chain and auto-loads the image as clip 0's opening keyframe (#6)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "imageToClipChain",
          selection: imageSelection(2, 10, 130),
        });
      });

      await waitFor(
        () => {
          expect(screen.getByRole("tab", { name: /^chained$/i })).toHaveAttribute("aria-selected", "true");
        },
        { timeout: 5_000 },
      );

      // #6: the image (imageSelection's C:\i\a.png) is uploaded as an image (the
      // start-frame keyframe), not a source video.
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "backend.uploadFile",
            expect.objectContaining({ kind: "image", filePath: "C:\\i\\a.png" }),
          );
        },
        { timeout: 5_000 },
      );
      expect(spy).not.toHaveBeenCalledWith("backend.uploadFile", expect.objectContaining({ kind: "video" }));
      expect(screen.getByText(/loaded from a right-click: a\.png/i)).toBeInTheDocument();
    },
    15_000,
  );

  it(
    "shows the §6-2 missing-material note when the auto-load upload fails (#1)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0, failUploadFile: "FILE_NOT_FOUND" });
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "extendVideo",
          selection: selectionWith(800, 600),
        });
      });

      await screen.findByText(/source file was not found/i, undefined, { timeout: 5_000 });
    },
    15_000,
  );

  it(
    "keeps a prefilled + edited Create screen intact after the pending intent expires on a manual tab switch (M-1)",
    async () => {
      // Adversarial-review M-1: with screens always mounted, the Screen `key`
      // must NOT be derived from the transient `pendingIntent`. A right-click
      // prefill applies, then a manual tab switch nulls `pendingIntent`; under
      // the old key that flipped the key back to a "base" value and remounted
      // (wiping) the exact screen the user was editing. The per-mode remount
      // counter never regresses, so nothing here should remount Create.
      const bridge = createMockBridge({ delayMs: 0 });
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // Prefill Create to 1024x768 via a right-click reference-video route.
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "referenceVideo",
          selection: selectionWith(1000, 700),
        });
      });
      await waitFor(
        () => {
          expect(within(panel()).getAllByDisplayValue("1024").length).toBeGreaterThan(0);
        },
        { timeout: 5_000 },
      );

      // Now hand-edit the width on top of the prefill. The "Width" <label>
      // names its range slider (first labelable descendant), so target that by
      // role; the number input twin tracks the same state. This route is
      // IC-LoRA (128-px grid), so pick a clean multiple of 128 distinct from
      // the prefilled 1024.
      const widthInput = within(panel()).getByRole("slider", { name: /width/i }) as HTMLInputElement;
      fireEvent.change(widthInput, { target: { value: "896" } });
      expect(widthInput.value).toBe("896");

      // Manual tab switch (this nulls the pending intent) and back.
      const user = userEvent.setup();
      await user.click(screen.getByRole("tab", { name: /^chained$/i }));
      await screen.findByRole("tab", { name: /^chained$/i });
      await user.click(screen.getByRole("tab", { name: /^single$/i }));
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // The edited value survives — Create was never remounted by the expiring
      // intent.
      const widthAfter = within(panel()).getByRole("slider", { name: /width/i }) as HTMLInputElement;
      expect(widthAfter.value).toBe("896");
    },
    20_000,
  );

  it(
    "✨ text-to-video keeps the Create panel's edited form state (no remount, D3) and reserves a provisional at the cursor",
    async () => {
      // D3 owner decision: ✨ (textToVideoHere) must NOT remount Create — the
      // panel's current form state survives — and it only reserves a provisional
      // + switches to Create. This differs from the object-menu routes above,
      // which deliberately remount-and-prefill.
      //
      // I5: the throwaway "G1 insert" toast is gone. A NORMAL insert
      // (usedFallback=false) shows NO note (§6 思想: the provisional appearing
      // on the timeline is the feedback), so this asserts the provisional RPC
      // was actually issued at placement "C"/the cursor instead.
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      const requestSpy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // Hand-edit the width (plain Create is the 64-px grid; 896 = 64*14).
      const widthInput = within(panel()).getByRole("slider", { name: /width/i }) as HTMLInputElement;
      fireEvent.change(widthInput, { target: { value: "896" } });
      expect(widthInput.value).toBe("896");

      // Fire ✨ at layer 4 / frame 360 (SDK 0-based).
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "textToVideoHere",
          selection: layerSelection(4, 360),
        });
      });

      // The reservation issues `timeline.insertProvisional` at placement "C"
      // with the cursor it was handed (SDK 0-based layer 4 / frame 360). I13
      // §5-5 段階1: the placeholder carries the FIXED ASCII phrase (NOT the
      // prompt), split prefix + body so native's 16-codepoint body truncation
      // never clips the "placed here" tail — native renders
      // "AI video will be placed here…".
      await waitFor(
        () => {
          expect(requestSpy).toHaveBeenCalledWith(
            "timeline.insertProvisional",
            expect.objectContaining({
              placement: "C",
              cursorLayer: 4,
              cursorFrame: 360,
              textPrefix: "AI video will be ",
              displayText: "placed here",
            }),
          );
        },
        { timeout: 5_000 },
      );

      // A plain insert shows NO note.
      expect(screen.queryByText(/frontmost layer/i)).not.toBeInTheDocument();

      // Still on Create, and the edited width was NOT wiped by a remount.
      expect(screen.getByRole("tab", { name: /^single$/i })).toHaveAttribute("aria-selected", "true");
      const widthAfter = within(panel()).getByRole("slider", { name: /width/i }) as HTMLInputElement;
      expect(widthAfter.value).toBe("896");
    },
    15_000,
  );

  it(
    "✨ shows the §5-4 fallback note (layer shown +1) when native falls back to layer_max+1",
    async () => {
      // §5-4 / §6-3: when the cursor spot is occupied, native places the
      // provisional on layer_max+1 and reports usedFallback=true; the note area
      // then explains it, with the layer number shown +1 (1-based, §6-3).
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      const original = bridge.request.bind(bridge);
      // Force just the provisional insert to report a fallback placement
      // (SDK-0-based layer 11 -> shown as 12); everything else is unchanged.
      vi.spyOn(bridge, "request").mockImplementation((method, params) => {
        if (method === "timeline.insertProvisional") {
          return Promise.resolve({
            inserted: true,
            layer: 11,
            frame: 360,
            objectName: "provisional-fallback",
            placedLayer: 11,
            placedFrame: 360,
            usedFallback: true,
          }) as ReturnType<typeof original>;
        }
        return original(method, params);
      });
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "textToVideoHere",
          selection: layerSelection(4, 360),
        });
      });

      // The fallback note appears in the shared note area, layer shown +1 (12).
      const note = await screen.findByText(/frontmost layer 12/i, undefined, { timeout: 5_000 });
      expect(note).toBeInTheDocument();
    },
    15_000,
  );

  // --- I8: #1/#6 Chain-discard confirm (§3-4 ※) ----------------------------

  // Loads a source into the Chain form (via #1) so it becomes dirty, leaving the
  // app on the Chain tab with a reserved seat. Shared setup for the two dialog
  // tests below.
  async function makeChainDirtyViaExtendVideo(bridge: ReturnType<typeof createMockBridge>) {
    act(() => {
      bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "extendVideo", selection: selectionWith(800, 600) });
    });
    await waitFor(
      () => {
        expect(screen.getByRole("tab", { name: /^chained$/i })).toHaveAttribute("aria-selected", "true");
      },
      { timeout: 5_000 },
    );
    // The published dirty flag is what the #6 route's confirm gate reads.
    await waitFor(() => expect(getChainDirty()).toBe(true), { timeout: 5_000 });
  }

  it(
    "a dirty Chain shows the discard dialog on a #6 route; Cancel does nothing (no load, no remount)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      await makeChainDirtyViaExtendVideo(bridge);

      const spy = vi.spyOn(bridge, "request");
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "imageToClipChain", selection: imageSelection(2, 10, 130) });
      });

      // The confirm dialog appears (Chain form is dirty from the #1 source).
      const dialog = await screen.findByRole("dialog", undefined, { timeout: 5_000 });
      expect(within(dialog).getByText(/discard the current edits on the chain screen/i)).toBeInTheDocument();

      // Cancel: nothing else happens — no image auto-load, no reservation move.
      const user = userEvent.setup();
      await user.click(within(dialog).getByRole("button", { name: /^cancel$/i }));
      await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

      // Give the (aborted) handler a beat; the #6 image was never uploaded and
      // the seat was never moved.
      await new Promise((r) => setTimeout(r, 50));
      expect(spy).not.toHaveBeenCalledWith("backend.uploadFile", expect.objectContaining({ kind: "image" }));
      expect(spy).not.toHaveBeenCalledWith("timeline.updateProvisionalReservation", expect.anything());
    },
    20_000,
  );

  it(
    "a dirty Chain's #6 route continues the load when the discard dialog is confirmed (OK)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      await makeChainDirtyViaExtendVideo(bridge);

      const spy = vi.spyOn(bridge, "request");
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "imageToClipChain", selection: imageSelection(2, 10, 130) });
      });

      const dialog = await screen.findByRole("dialog", undefined, { timeout: 5_000 });
      const user = userEvent.setup();
      await user.click(within(dialog).getByRole("button", { name: /^ok$/i }));

      // OK: the #6 image now auto-loads (remount + prefill proceeds).
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "backend.uploadFile",
            expect.objectContaining({ kind: "image", filePath: "C:\\i\\a.png" }),
          );
        },
        { timeout: 5_000 },
      );
      expect(screen.getByText(/loaded from a right-click: a\.png/i)).toBeInTheDocument();
    },
    20_000,
  );

  // --- I7: unified handleRoute (guard-first, busy-guard, placement) ---------

  it(
    "a type-mismatch guard shows a note and does NOTHING else (no tab switch, no reservation)",
    async () => {
      // §4-3: extendVideo requires a video; an image selection mismatches. The
      // guard must fire before anything destructive — no Chain switch, no
      // provisional insert.
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(getReservationPhase()).toBe("idle"));

      const spy = vi.spyOn(bridge, "request");
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "extendVideo", selection: imageSelection() });
      });

      // The mismatch note appears…
      await screen.findByText(/requires a video/i, undefined, { timeout: 5_000 });
      // …and we stayed on Create (no Chain switch) and issued no provisional RPC.
      expect(screen.getByRole("tab", { name: /^single$/i })).toHaveAttribute("aria-selected", "true");
      expect(spy).not.toHaveBeenCalledWith("timeline.insertProvisional", expect.anything());
      expect(spy).not.toHaveBeenCalledWith("timeline.updateProvisionalReservation", expect.anything());
    },
    15_000,
  );

  it(
    "a multiple-selection guard shows a note and does nothing else",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(getReservationPhase()).toBe("idle"));

      const spy = vi.spyOn(bridge, "request");
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "imageToVideo", selection: multiSelection() });
      });

      await screen.findByText(/just one object/i, undefined, { timeout: 5_000 });
      expect(spy).not.toHaveBeenCalledWith("timeline.insertProvisional", expect.anything());
    },
    15_000,
  );

  it(
    "refuses a new reservation while the seat is ⏳生成中 (busy-guard, §5-6)",
    async () => {
      // The reconcile-on-mount finds a job-bound placeholder and restores the
      // seat to `generating`; a generation-origin right-click is then refused
      // with the 予約不可 note and issues no provisional RPC.
      resetProvisionalReservation();
      const bridge = createMockBridge({
        delayMs: 0,
        provisionalOrphans: [{ jobId: "job-busy", layer: 0, frame: 0 }],
      });
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(getReservationPhase()).toBe("generating"));

      const spy = vi.spyOn(bridge, "request");
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "imageToVideo", selection: imageSelection() });
      });

      await screen.findByText(/new reservation cannot be made/i, undefined, { timeout: 5_000 });
      expect(spy).not.toHaveBeenCalledWith("timeline.insertProvisional", expect.anything());
      expect(spy).not.toHaveBeenCalledWith("timeline.updateProvisionalReservation", expect.anything());
    },
    15_000,
  );

  it(
    "a material item (imageToVideo) inserts a placement-A provisional with the material range",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(getReservationPhase()).toBe("idle"));

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "imageToVideo",
          selection: imageSelection(2, 10, 130),
        });
      });

      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "timeline.insertProvisional",
            expect.objectContaining({
              placement: "A",
              materialLayer: 2,
              materialFrameStart: 10,
              materialFrameEnd: 130,
            }),
          );
        },
        { timeout: 5_000 },
      );
      // Still routed to Create (imageToVideo's target).
      expect(screen.getByRole("tab", { name: /^single$/i })).toHaveAttribute("aria-selected", "true");
    },
    15_000,
  );

  it(
    "re-clicking a material item MOVES the single seat (updateProvisionalReservation)",
    async () => {
      // First ✨ reserves at the cursor (seat -> reserved); then a material item
      // re-uses the one seat by MOVING it to placement A (§5-6 用途b), not a 2nd
      // insert.
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(getReservationPhase()).toBe("idle"));

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "textToVideoHere",
          selection: layerSelection(4, 360),
        });
      });
      await waitFor(() => expect(getReservationPhase()).toBe("reserved"), { timeout: 5_000 });

      const spy = vi.spyOn(bridge, "request");
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "imageToVideo",
          selection: imageSelection(2, 10, 130),
        });
      });

      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "timeline.updateProvisionalReservation",
            expect.objectContaining({ placement: "A", materialFrameEnd: 130 }),
          );
        },
        { timeout: 5_000 },
      );
      expect(spy).not.toHaveBeenCalledWith("timeline.insertProvisional", expect.anything());
    },
    15_000,
  );

  // --- I11: #8 appendText (early-return append + placement B) ----------------

  it(
    "#8 appends the text body to an empty prompt, shows the receipt note, reserves a placement-B provisional, and never remounts Create",
    async () => {
      resetProvisionalReservation();
      // The text object's filePath is null, so resolveMenuSelection re-queries
      // getSelection — point the mock at the same text selection.
      const bridge = createMockBridge({ delayMs: 0, selection: textSelection() });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(getReservationPhase()).toBe("idle"));

      // Hand-edit the width first — #8 must not remount Create (§3-4 #8).
      const widthInput = within(panel()).getByRole("slider", { name: /width/i }) as HTMLInputElement;
      fireEvent.change(widthInput, { target: { value: "896" } });
      expect(widthInput.value).toBe("896");

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "appendText", selection: textSelection() });
      });

      // The body is appended to the (empty) shared prompt — no leading newline.
      await waitFor(() => expect(promptBox().value).toBe("hello"), { timeout: 5_000 });
      // §3-4 #8 receipt note.
      expect(screen.getByText(/appended to the prompt: hello/i)).toBeInTheDocument();
      // §5-9 system B: a provisional head-aligned with the text object (layer 2,
      // frames 10..130 from imageSelection()'s range).
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "timeline.insertProvisional",
            expect.objectContaining({
              placement: "B",
              materialLayer: 2,
              materialFrameStart: 10,
              materialFrameEnd: 130,
            }),
          );
        },
        { timeout: 5_000 },
      );
      // No remount: the edited width survives, and #8 never switched tabs.
      expect((within(panel()).getByRole("slider", { name: /width/i }) as HTMLInputElement).value).toBe("896");
      expect(screen.getByRole("tab", { name: /^single$/i })).toHaveAttribute("aria-selected", "true");
    },
    15_000,
  );

  it(
    "#8 appends after an existing prompt with a single newline separator",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0, selection: textSelection() });
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // Seed the shared prompt with existing text.
      const user = userEvent.setup();
      await user.type(promptBox(), "existing");
      expect(promptBox().value).toBe("existing");

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "appendText", selection: textSelection() });
      });

      // §3-4 #8 区切り: exactly one newline between the old text and the appended body.
      await waitFor(() => expect(promptBox().value).toBe("existing\nhello"), { timeout: 5_000 });
    },
    15_000,
  );

  it(
    "#8 with an empty text body shows the empty note and appends/reserves nothing",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0, selection: textSelection("") });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(getReservationPhase()).toBe("idle"));

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "appendText", selection: textSelection("") });
      });

      await screen.findByText(/the text is empty/i, undefined, { timeout: 5_000 });
      // Nothing appended, nothing reserved.
      expect(promptBox().value).toBe("");
      expect(spy).not.toHaveBeenCalledWith("timeline.insertProvisional", expect.anything());
      expect(spy).not.toHaveBeenCalledWith("timeline.updateProvisionalReservation", expect.anything());
    },
    15_000,
  );

  it(
    "#8 is refused entirely while the seat is ⏳生成中 (no append, no reservation)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({
        delayMs: 0,
        selection: textSelection(),
        provisionalOrphans: [{ jobId: "job-busy", layer: 0, frame: 0 }],
      });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(getReservationPhase()).toBe("generating"));

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "appendText", selection: textSelection() });
      });

      await screen.findByText(/new reservation cannot be made/i, undefined, { timeout: 5_000 });
      expect(promptBox().value).toBe("");
      expect(spy).not.toHaveBeenCalledWith("timeline.insertProvisional", expect.anything());
      expect(spy).not.toHaveBeenCalledWith("timeline.updateProvisionalReservation", expect.anything());
    },
    15_000,
  );

  // --- I11: #5 addImageKeyframe (live append, count-branched placement) -------

  it(
    "#5 (count 0) appends the image as the 1st keyframe, switches to Create, reserves placement A, and keeps the form",
    async () => {
      resetProvisionalReservation();
      resetCreateLiveCommands();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(getReservationPhase()).toBe("idle"));

      // Hand-edit the width — #5 keeps the panel state (no remount).
      const widthInput = within(panel()).getByRole("slider", { name: /width/i }) as HTMLInputElement;
      fireEvent.change(widthInput, { target: { value: "896" } });
      expect(widthInput.value).toBe("896");

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "addImageKeyframe", selection: imageSelection(2, 10, 130) });
      });

      // The image is appended into KEYFRAMES (an image upload) and becomes the
      // 1st keyframe -> mode badge I2V.
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "backend.uploadFile",
            expect.objectContaining({ kind: "image", filePath: "C:\\i\\a.png" }),
          );
        },
        { timeout: 5_000 },
      );
      // §5-9 system A: a provisional placed off the image's tail (frames 10..130).
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "timeline.insertProvisional",
            expect.objectContaining({ placement: "A", materialFrameStart: 10, materialFrameEnd: 130 }),
          );
        },
        { timeout: 5_000 },
      );
      await waitFor(
        () => expect(within(panel()).getByText(/i2v \(1 keyframe\)/i)).toBeInTheDocument(),
        { timeout: 5_000 },
      );
      expect(screen.getByText(/appended the image to a keyframe: a\.png/i)).toBeInTheDocument();
      // Switched to Create, and the edited width survives (no remount).
      expect(screen.getByRole("tab", { name: /^single$/i })).toHaveAttribute("aria-selected", "true");
      expect((within(panel()).getByRole("slider", { name: /width/i }) as HTMLInputElement).value).toBe("896");
    },
    15_000,
  );

  it(
    "#5 (count>0) appends another keyframe but touches no reservation",
    async () => {
      resetProvisionalReservation();
      resetCreateLiveCommands();
      const bridge = createMockBridge({ delayMs: 0 });
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(getReservationPhase()).toBe("idle"));

      // First #5 makes the image the 1st keyframe (count 0 -> placement A).
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "addImageKeyframe", selection: imageSelection(2, 10, 130) });
      });
      // Wait for BOTH the append (count 1) and the placement-A insert to settle
      // (reserved phase) before spying, so the first #5's own insertProvisional
      // can't leak into the second #5's assertions.
      await waitFor(() => expect(getReservationPhase()).toBe("reserved"), { timeout: 5_000 });
      await waitFor(() => expect(getPublishedKeyframeCount()).toBe(1), { timeout: 5_000 });

      // Now spy — the SECOND #5 (count 1) must append only, no reservation RPC.
      const spy = vi.spyOn(bridge, "request");
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "addImageKeyframe", selection: imageSelection(2, 10, 130) });
      });

      // A 2nd keyframe is appended (image upload) …
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "backend.uploadFile",
            expect.objectContaining({ kind: "image", filePath: "C:\\i\\a.png" }),
          );
        },
        { timeout: 5_000 },
      );
      await waitFor(() => expect(getPublishedKeyframeCount()).toBe(2), { timeout: 5_000 });
      // … but the seat was never inserted-into or moved.
      expect(spy).not.toHaveBeenCalledWith("timeline.insertProvisional", expect.anything());
      expect(spy).not.toHaveBeenCalledWith("timeline.updateProvisionalReservation", expect.anything());
    },
    15_000,
  );

  it(
    "#5 (count 0) is refused entirely while ⏳生成中 (no append, no reservation)",
    async () => {
      resetProvisionalReservation();
      resetCreateLiveCommands();
      const bridge = createMockBridge({
        delayMs: 0,
        provisionalOrphans: [{ jobId: "job-busy", layer: 0, frame: 0 }],
      });
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(getReservationPhase()).toBe("generating"));

      // Start on Chain so a (forbidden) switch-to-Create would be observable.
      const user = userEvent.setup();
      await user.click(screen.getByRole("tab", { name: /^chained$/i }));
      await waitFor(() =>
        expect(screen.getByRole("tab", { name: /^chained$/i })).toHaveAttribute("aria-selected", "true"),
      );

      const spy = vi.spyOn(bridge, "request");
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "addImageKeyframe", selection: imageSelection() });
      });

      await screen.findByText(/new reservation cannot be made/i, undefined, { timeout: 5_000 });
      // Full reject: no append, no reservation, no tab switch.
      expect(spy).not.toHaveBeenCalledWith("backend.uploadFile", expect.objectContaining({ kind: "image" }));
      expect(spy).not.toHaveBeenCalledWith("timeline.insertProvisional", expect.anything());
      expect(spy).not.toHaveBeenCalledWith("timeline.updateProvisionalReservation", expect.anything());
      expect(screen.getByRole("tab", { name: /^chained$/i })).toHaveAttribute("aria-selected", "true");
    },
    15_000,
  );

  it(
    "#5 (count>0) still appends while ⏳生成中 (append passes; seat untouched)",
    async () => {
      resetProvisionalReservation();
      resetCreateLiveCommands();
      const bridge = createMockBridge({
        delayMs: 0,
        provisionalOrphans: [{ jobId: "job-busy", layer: 0, frame: 0 }],
      });
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(getReservationPhase()).toBe("generating"));

      // Add an (image-less) keyframe via the panel's own button so the count is
      // >0 while the seat stays ⏳生成中. A running job does not disable the panel.
      const user = userEvent.setup();
      await user.click(screen.getByRole("button", { name: /add keyframe/i }));
      await waitFor(() => expect(getPublishedKeyframeCount()).toBe(1), { timeout: 5_000 });

      const spy = vi.spyOn(bridge, "request");
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "addImageKeyframe", selection: imageSelection(2, 10, 130) });
      });

      // The append proceeds despite ⏳生成中 (count>0 skips the busy-guard)…
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "backend.uploadFile",
            expect.objectContaining({ kind: "image", filePath: "C:\\i\\a.png" }),
          );
        },
        { timeout: 5_000 },
      );
      // … and it never refuses or touches the seat.
      expect(screen.queryByText(/new reservation cannot be made/i)).not.toBeInTheDocument();
      expect(spy).not.toHaveBeenCalledWith("timeline.insertProvisional", expect.anything());
      expect(spy).not.toHaveBeenCalledWith("timeline.updateProvisionalReservation", expect.anything());
    },
    15_000,
  );

  // --- W5: #11 addCurrentFrameAsKeyframe (capture -> tail keyframe, no remount) ---

  it(
    "addCurrentFrameAsKeyframe (count 0) captures the frame as the 1st keyframe, reserves placement C at the cursor, and keeps the form (#11)",
    async () => {
      resetProvisionalReservation();
      resetCreateLiveCommands();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(getReservationPhase()).toBe("idle"));

      // Hand-edit the width — #11 keeps the panel state (no remount).
      const widthInput = within(panel()).getByRole("slider", { name: /width/i }) as HTMLInputElement;
      fireEvent.change(widthInput, { target: { value: "896" } });
      expect(widthInput.value).toBe("896");

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "addCurrentFrameAsKeyframe",
          selection: layerSelection(4, 360),
        });
      });

      // The current frame is captured (native RPC) into KEYFRAMES and becomes the
      // 1st keyframe -> mode badge I2V.
      await waitFor(() => expect(spy).toHaveBeenCalledWith("timeline.captureFrame", {}), { timeout: 5_000 });
      await waitFor(
        () => expect(within(panel()).getByText(/i2v \(1 keyframe\)/i)).toBeInTheDocument(),
        { timeout: 5_000 },
      );

      // §5-9 system C: a provisional reserved at the right-click cursor
      // (SDK 0-based layer 4 / frame 360).
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "timeline.insertProvisional",
            expect.objectContaining({ placement: "C", cursorLayer: 4, cursorFrame: 360 }),
          );
        },
        { timeout: 5_000 },
      );

      // Receipt note (capture-based, so it fires once the card reaches "ready"),
      // and the edited width survives (Create was NOT remounted).
      await screen.findByText(/captured the current frame into a keyframe/i, undefined, { timeout: 5_000 });
      const widthAfter = within(panel()).getByRole("slider", { name: /width/i }) as HTMLInputElement;
      expect(widthAfter.value).toBe("896");
      expect(screen.getByRole("tab", { name: /^single$/i })).toHaveAttribute("aria-selected", "true");
    },
    15_000,
  );

  it(
    "addCurrentFrameAsKeyframe (count>0) appends another captured keyframe but touches no reservation (#11)",
    async () => {
      resetProvisionalReservation();
      resetCreateLiveCommands();
      const bridge = createMockBridge({ delayMs: 0 });
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(getReservationPhase()).toBe("idle"));

      // Seed one keyframe via the panel's own button (count -> 1) without touching
      // the reservation seat.
      const user = userEvent.setup();
      await user.click(screen.getByRole("button", { name: /add keyframe/i }));
      await waitFor(() => expect(getPublishedKeyframeCount()).toBe(1), { timeout: 5_000 });

      // Now spy — the #11 append (count 1) must capture + append only, no reservation.
      const spy = vi.spyOn(bridge, "request");
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "addCurrentFrameAsKeyframe",
          selection: layerSelection(4, 360),
        });
      });

      await waitFor(() => expect(spy).toHaveBeenCalledWith("timeline.captureFrame", {}), { timeout: 5_000 });
      await waitFor(() => expect(getPublishedKeyframeCount()).toBe(2), { timeout: 5_000 });
      // The seat was never inserted-into or moved.
      expect(spy).not.toHaveBeenCalledWith("timeline.insertProvisional", expect.anything());
      expect(spy).not.toHaveBeenCalledWith("timeline.updateProvisionalReservation", expect.anything());
    },
    15_000,
  );

  it(
    "addCurrentFrameAsKeyframe (count 0) is refused entirely while ⏳生成中 (no capture, no reservation) (#11)",
    async () => {
      resetProvisionalReservation();
      resetCreateLiveCommands();
      const bridge = createMockBridge({
        delayMs: 0,
        provisionalOrphans: [{ jobId: "job-busy", layer: 0, frame: 0 }],
      });
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(getReservationPhase()).toBe("generating"));

      // Start on Chain so a (forbidden) switch-to-Create would be observable.
      const user = userEvent.setup();
      await user.click(screen.getByRole("tab", { name: /^chained$/i }));
      await waitFor(() =>
        expect(screen.getByRole("tab", { name: /^chained$/i })).toHaveAttribute("aria-selected", "true"),
      );

      const spy = vi.spyOn(bridge, "request");
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "addCurrentFrameAsKeyframe",
          selection: layerSelection(4, 360),
        });
      });

      await screen.findByText(/new reservation cannot be made/i, undefined, { timeout: 5_000 });
      // Full reject: no capture, no reservation, no tab switch.
      expect(spy).not.toHaveBeenCalledWith("timeline.captureFrame", {});
      expect(spy).not.toHaveBeenCalledWith("timeline.insertProvisional", expect.anything());
      expect(spy).not.toHaveBeenCalledWith("timeline.updateProvisionalReservation", expect.anything());
      expect(screen.getByRole("tab", { name: /^chained$/i })).toHaveAttribute("aria-selected", "true");
    },
    15_000,
  );

  // --- W5: #12 currentFrameToClipChain (capture -> Chain clip 0 opening frame) ---

  it(
    "currentFrameToClipChain shows the discard confirm on a dirty Chain, then captures the frame into clip 0's opening keyframe (#12)",
    async () => {
      resetProvisionalReservation();
      resetCreateLiveCommands();
      const bridge = createMockBridge({ delayMs: 0 });
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // Make Chain dirty first (via #1) so the §3-4 ※ discard gate is exercised.
      await makeChainDirtyViaExtendVideo(bridge);

      const spy = vi.spyOn(bridge, "request");
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "currentFrameToClipChain",
          selection: layerSelection(4, 360),
        });
      });

      // The confirm dialog appears (Chain form is dirty from the #1 source); OK
      // continues the remount + capture load.
      const dialog = await screen.findByRole("dialog", undefined, { timeout: 5_000 });
      const user = userEvent.setup();
      await user.click(within(dialog).getByRole("button", { name: /^ok$/i }));

      // The current frame is captured and uploaded as an IMAGE (clip 0's opening
      // keyframe). (No "never video" assertion here: the just-dirtied #1 source
      // video's own upload can still be settling on the shared bridge — the #6
      // dirty-OK test omits it for the same reason.)
      await waitFor(() => expect(spy).toHaveBeenCalledWith("timeline.captureFrame", {}), { timeout: 5_000 });
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "backend.uploadFile",
            expect.objectContaining({ kind: "image" }),
          );
        },
        { timeout: 5_000 },
      );
      // §6 receipt and still on Chain (#12's target).
      expect(screen.getByText(/captured the current frame into a keyframe/i)).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /^chained$/i })).toHaveAttribute("aria-selected", "true");
    },
    20_000,
  );

  it(
    "currentFrameToClipChain shows the capture-failure note when captureFrame rejects (#12)",
    async () => {
      resetProvisionalReservation();
      resetCreateLiveCommands();
      const bridge = createMockBridge({ delayMs: 0, failCaptureFrame: "CAPTURE_FAILED" });
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // Pristine Chain -> no discard dialog; the route remounts Chain and the
      // capture attempt fails outright.
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "currentFrameToClipChain",
          selection: layerSelection(4, 360),
        });
      });

      await waitFor(
        () => expect(screen.getByRole("tab", { name: /^chained$/i })).toHaveAttribute("aria-selected", "true"),
        { timeout: 5_000 },
      );
      await screen.findByText(/could not capture the current frame/i, undefined, { timeout: 5_000 });
    },
    15_000,
  );

  // --- 6-b: reload-time seat rebuild (projectLoaded push + on-demand re-sync) ---

  it(
    "rebuilds the reservation seat when a projectLoaded event arrives (§6-b push)",
    async () => {
      // A provisional that survived a project reopen isn't in memory yet (the
      // seat starts idle after the mount reconcile finds nothing). When native
      // fires `timeline.projectLoaded`, the WebUI re-runs
      // `timeline.scanProvisionals` and rebuilds the seat from the surviving
      // `pending-…` placeholder.
      resetProvisionalReservation();
      const opts: MockBridgeOptions = { delayMs: 0, provisionalOrphans: [] };
      const bridge = createMockBridge(opts);
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(getReservationPhase()).toBe("idle"));

      // A provisional now "exists" on the reopened timeline; the projectLoaded
      // push should make the WebUI re-scan and restore the seat.
      opts.provisionalOrphans = [{ jobId: "pending-survivor", layer: 2, frame: 20 }];
      const spy = vi.spyOn(bridge, "request");
      act(() => {
        bridge.emit(TIMELINE_PROJECT_LOADED_EVENT, {});
      });

      await waitFor(() => expect(getReservationPhase()).toBe("reserved"), { timeout: 5_000 });
      expect(spy).toHaveBeenCalledWith("timeline.scanProvisionals", {});
    },
    15_000,
  );

  it(
    "right-click re-syncs from the timeline: an idle-in-memory seat with a live ⏳生成中 still trips the busy-guard",
    async () => {
      // Insurance path (§6-b): even if `projectLoaded` never fired, a
      // generation-origin right-click re-runs the reconcile *before* the
      // busy-guard, so a job-bound `NzVideomni#<jobId>` present on the timeline
      // (memory still idle) makes the guard refuse the new reservation.
      resetProvisionalReservation();
      const opts: MockBridgeOptions = { delayMs: 0, provisionalOrphans: [] };
      const bridge = createMockBridge(opts);
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(getReservationPhase()).toBe("idle"));

      // A real generation (non-pending id -> job-bound) is now on the timeline,
      // but the in-memory seat was never told about it.
      opts.provisionalOrphans = [{ jobId: "live-job-77", layer: 0, frame: 0 }];
      const spy = vi.spyOn(bridge, "request");
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "imageToVideo", selection: imageSelection() });
      });

      await screen.findByText(/new reservation cannot be made/i, undefined, { timeout: 5_000 });
      expect(spy).not.toHaveBeenCalledWith("timeline.insertProvisional", expect.anything());
      expect(spy).not.toHaveBeenCalledWith("timeline.updateProvisionalReservation", expect.anything());
    },
    15_000,
  );

  it(
    "right-click re-syncs from the timeline: a reserved seat whose placeholder was manually deleted falls back to a fresh insert (not a move)",
    async () => {
      // §6-b: the seat is `reserved` from a survived placeholder, but the user
      // then deletes that placeholder by hand. The next right-click's reconcile
      // sees the timeline is empty, flips the seat back to idle, and
      // reservePlacement performs a fresh insert rather than moving a phantom
      // reservation.
      resetProvisionalReservation();
      const opts: MockBridgeOptions = {
        delayMs: 0,
        provisionalOrphans: [{ jobId: "pending-old", layer: 1, frame: 5 }],
      };
      const bridge = createMockBridge(opts);
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      // Mount reconcile restored the seat as reserved from the survived placeholder.
      await waitFor(() => expect(getReservationPhase()).toBe("reserved"), { timeout: 5_000 });

      // The user manually deletes the placeholder: the timeline no longer has it.
      opts.provisionalOrphans = [];
      const spy = vi.spyOn(bridge, "request");
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "imageToVideo",
          selection: imageSelection(2, 10, 130),
        });
      });

      // A fresh insert (not a move) is issued at placement A.
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "timeline.insertProvisional",
            expect.objectContaining({ placement: "A", materialFrameEnd: 130 }),
          );
        },
        { timeout: 5_000 },
      );
      expect(spy).not.toHaveBeenCalledWith("timeline.updateProvisionalReservation", expect.anything());
    },
    15_000,
  );

  // --- I9: Create-side material wiring (#2 referenceVideo / #4 imageToVideo) ---

  it(
    "referenceVideo auto-loads the selected video into the IC-LoRA reference slot with the filePath (#2)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // media 1000x700 with the IC-LoRA 128 grid -> ceil to 1024 x 768.
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "referenceVideo",
          selection: selectionWith(1000, 700),
        });
      });

      // #2: the selected video's filePath is auto-uploaded into the reference
      // (IC-LoRA) slot — a VIDEO upload, path-carried.
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "backend.uploadFile",
            expect.objectContaining({ kind: "video", filePath: "C:\\v\\a.mp4" }),
          );
        },
        { timeout: 5_000 },
      );

      // 128-rounding still in force (prefilled resolution), and the §6 receipt note.
      expect(within(panel()).getAllByDisplayValue("1024").length).toBeGreaterThan(0);
      expect(within(panel()).getAllByDisplayValue("768").length).toBeGreaterThan(0);
      expect(screen.getByText(/loaded from a right-click: a\.mp4/i)).toBeInTheDocument();
      // Stayed on Create.
      expect(screen.getByRole("tab", { name: /^single$/i })).toHaveAttribute("aria-selected", "true");
    },
    15_000,
  );

  it(
    "imageToVideo auto-loads the image as KEYFRAMES' 1st card and the mode badge becomes I2V (#4)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "imageToVideo",
          selection: imageSelection(2, 10, 130),
        });
      });

      // #4: the image (C:\i\a.png) is uploaded as an IMAGE (the frame-0 keyframe).
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "backend.uploadFile",
            expect.objectContaining({ kind: "image", filePath: "C:\\i\\a.png" }),
          );
        },
        { timeout: 5_000 },
      );
      expect(spy).not.toHaveBeenCalledWith("backend.uploadFile", expect.objectContaining({ kind: "video" }));

      // Once the keyframe is ready, the mode badge reads I2V (1 keyframe).
      await waitFor(
        () => {
          expect(within(panel()).getByText(/i2v \(1 keyframe\)/i)).toBeInTheDocument();
        },
        { timeout: 5_000 },
      );
      expect(screen.getByText(/loaded from a right-click: a\.png/i)).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /^single$/i })).toHaveAttribute("aria-selected", "true");
    },
    15_000,
  );

  it(
    "imageToVideo shows the §6-2 missing-material note when the keyframe upload fails (#4)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0, failUploadFile: "FILE_NOT_FOUND" });
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "imageToVideo",
          selection: imageSelection(2, 10, 130),
        });
      });

      await screen.findByText(/source file was not found/i, undefined, { timeout: 5_000 });
    },
    15_000,
  );

  // --- I9: #10 imageFromCurrentFrame (live-command channel, no remount) ------

  it(
    "imageFromCurrentFrame captures the current frame into KEYFRAMES without remounting Create (#10)",
    async () => {
      // D3: 📷 keeps the panel state (no remount) and arrives over the live
      // command channel, not a pendingIntent. It captures the current timeline
      // frame into KEYFRAMES via addFromCapture and preserves the edited form.
      resetProvisionalReservation();
      resetCreateLiveCommands();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // Hand-edit the width first (plain Create is the 64-px grid; 896 = 64*14).
      const widthInput = within(panel()).getByRole("slider", { name: /width/i }) as HTMLInputElement;
      fireEvent.change(widthInput, { target: { value: "896" } });
      expect(widthInput.value).toBe("896");

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "imageFromCurrentFrame",
          selection: layerSelection(4, 360),
        });
      });

      // The current frame was captured (native RPC) and the resulting card
      // becomes the 1st keyframe -> mode badge I2V.
      await waitFor(() => expect(spy).toHaveBeenCalledWith("timeline.captureFrame", {}), { timeout: 5_000 });
      await waitFor(
        () => {
          expect(within(panel()).getByText(/i2v \(1 keyframe\)/i)).toBeInTheDocument();
        },
        { timeout: 5_000 },
      );

      // Receipt note, and the edited width survives (Create was NOT remounted).
      // `findByText` (not `getByText`): the I2V badge appears the moment the card
      // is APPENDED (status still "uploading"), but the receipt note only fires
      // once that card reaches "ready" (the capture status-watch), so a sync read
      // here raced the note ~40% of runs. Awaiting the note closes that window.
      await screen.findByText(/captured the current frame into a keyframe/i, undefined, { timeout: 5_000 });
      const widthAfter = within(panel()).getByRole("slider", { name: /width/i }) as HTMLInputElement;
      expect(widthAfter.value).toBe("896");
      expect(screen.getByRole("tab", { name: /^single$/i })).toHaveAttribute("aria-selected", "true");
    },
    15_000,
  );

  it(
    "imageFromCurrentFrame shows the capture-failure note when captureFrame rejects (#10)",
    async () => {
      resetProvisionalReservation();
      resetCreateLiveCommands();
      const bridge = createMockBridge({ delayMs: 0, failCaptureFrame: "CAPTURE_FAILED" });
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "imageFromCurrentFrame",
          selection: layerSelection(4, 360),
        });
      });

      await screen.findByText(/could not capture the current frame/i, undefined, { timeout: 5_000 });
    },
    15_000,
  );

  // W6: #10's frame-0 collision bugfix — when a frame-0 card already exists,
  // captureFrameToKeyframe must REPLACE its image in place (captureIntoCard)
  // instead of always appending a second frame-0 card (the old always-append
  // `addFromCapture(0)` behavior, which left two cards stacked at frame 0).

  it(
    "imageFromCurrentFrame replaces an existing frame-0 card in place instead of appending a duplicate (#10 W6 bugfix)",
    async () => {
      resetProvisionalReservation();
      resetCreateLiveCommands();
      const bridge = createMockBridge({ delayMs: 0 });
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // Seed a frame-0 card via #5 (addImageKeyframe) — it lands at frame 0
      // (count 0 -> nextAddPosition 0), the SAME slot #10 targets.
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "addImageKeyframe", selection: imageSelection(2, 10, 130) });
      });
      await waitFor(() => expect(getPublishedKeyframeCount()).toBe(1), { timeout: 5_000 });
      await waitFor(
        () => expect(within(panel()).getByText(/i2v \(1 keyframe\)/i)).toBeInTheDocument(),
        { timeout: 5_000 },
      );
      await waitFor(() => expect(within(panel()).getByAltText("a.png")).toBeInTheDocument(), { timeout: 5_000 });

      // #10: a frame-0 card is already there, so this must REPLACE its image
      // (captureIntoCard) rather than append a second frame-0 card.
      const spy = vi.spyOn(bridge, "request");
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "imageFromCurrentFrame",
          selection: layerSelection(4, 360),
        });
      });

      await waitFor(() => expect(spy).toHaveBeenCalledWith("timeline.captureFrame", {}), { timeout: 5_000 });
      await screen.findByText(/captured the current frame into a keyframe/i, undefined, { timeout: 5_000 });

      // Card count is unchanged (still 1) — no collision, no duplicate card —
      // and the frame-0 card's image was swapped for the capture.
      expect(getPublishedKeyframeCount()).toBe(1);
      expect(within(panel()).getByText(/i2v \(1 keyframe\)/i)).toBeInTheDocument();
      expect(within(panel()).getAllByRole("img")).toHaveLength(1);
      expect(within(panel()).queryByAltText("a.png")).not.toBeInTheDocument();
      expect(within(panel()).getByAltText(/^frame_\d+_\d+\.png$/)).toBeInTheDocument();
    },
    15_000,
  );

  it(
    "imageFromCurrentFrame still replaces the frame-0 card at the keyframe cap (the cap only blocks appends) (#10 W6 bugfix)",
    async () => {
      resetProvisionalReservation();
      resetCreateLiveCommands();
      const bridge = createMockBridge({ delayMs: 0 });
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // Fill KEYFRAMES to the cap (mock config: max_conditioning_images 5) via
      // repeated #5 dispatches — the first lands at frame 0, #10's target slot.
      for (let i = 0; i < 5; i++) {
        act(() => {
          bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "addImageKeyframe", selection: imageSelection(2, 10, 130) });
        });
        await waitFor(() => expect(getPublishedKeyframeCount()).toBe(i + 1), { timeout: 5_000 });
      }
      await waitFor(
        () => expect(within(panel()).getByText(/i2v \(5 keyframes\)/i)).toBeInTheDocument(),
        { timeout: 5_000 },
      );

      // At the cap, #10 must still succeed by REPLACING the frame-0 card — the
      // cap (canAdd) only guards the append path, never this in-place swap.
      const spy = vi.spyOn(bridge, "request");
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "imageFromCurrentFrame",
          selection: layerSelection(4, 360),
        });
      });

      await waitFor(() => expect(spy).toHaveBeenCalledWith("timeline.captureFrame", {}), { timeout: 5_000 });
      await screen.findByText(/captured the current frame into a keyframe/i, undefined, { timeout: 5_000 });

      // Not rejected by the cap (the failure note never appears), and the
      // count is still exactly the cap — a replace, not a blocked append.
      expect(screen.queryByText(/could not capture the current frame/i)).not.toBeInTheDocument();
      expect(getPublishedKeyframeCount()).toBe(5);
      expect(within(panel()).getByText(/i2v \(5 keyframes\)/i)).toBeInTheDocument();
      expect(within(panel()).getAllByRole("img")).toHaveLength(5);
    },
    20_000,
  );

  // --- I10: audio wiring (#7 audioToVideo / #3 videoAudioToVideo) -------------

  it(
    "audioToVideo auto-loads the selected audio file into the source-audio slot and enables A2V (#7)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "audioToVideo",
          selection: audioSelection(),
        });
      });

      // #7: the selected audio's filePath is auto-uploaded into the source-audio
      // slot via attachSourceAudioByPath -> backend.uploadFile({kind:"audio"}).
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "backend.uploadFile",
            expect.objectContaining({ kind: "audio", filePath: "C:\\a\\voice.wav" }),
          );
        },
        { timeout: 5_000 },
      );
      // Never uploaded as a video (that would be the wrong slot).
      expect(spy).not.toHaveBeenCalledWith("backend.uploadFile", expect.objectContaining({ kind: "video" }));
      // §6 receipt note names the loaded file, and we stayed on Create.
      expect(screen.getByText(/loaded from a right-click: voice\.wav/i)).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /^single$/i })).toHaveAttribute("aria-selected", "true");
    },
    15_000,
  );

  it(
    "videoAudioToVideo extracts the video's audio (mix, inclusive frame range) and loads the wav into the source-audio slot (#3)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // selectionWith(): video, layer 1, frameStart 0, frameEnd 120 -> frameCount
      // 121 (frameEnd inclusive). audioMode "mix", empty solo.
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "videoAudioToVideo",
          selection: selectionWith(800, 600),
        });
      });

      // The extracting note shows up front (mix-path explanation bundled).
      await screen.findByText(/extracting audio/i, undefined, { timeout: 5_000 });

      // #3: timeline.extractAudio is called DIRECTLY with the exact params.
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith("timeline.extractAudio", {
            layer: 1,
            frameStart: 0,
            frameCount: 121,
            audioMode: "mix",
            soloKeepLayers: [],
          });
        },
        { timeout: 5_000 },
      );

      // The extracted wav (mock's extract_*.wav) is then uploaded to the audio
      // slot — a single upload, funnelled through the source-audio channel.
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "backend.uploadFile",
            expect.objectContaining({ kind: "audio", filePath: expect.stringContaining("extract_1_0_") }),
          );
        },
        { timeout: 5_000 },
      );
      // The success receipt replaces the extracting note.
      await screen.findByText(/loaded the video's audio/i, undefined, { timeout: 5_000 });
      expect(screen.getByRole("tab", { name: /^single$/i })).toHaveAttribute("aria-selected", "true");
    },
    15_000,
  );

  it(
    "videoAudioToVideo shows the silence note when the range is silent (#3)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({
        delayMs: 0,
        failExtractAudio: "EXTRACT_FAILED",
        extractAudioErrorMessage: "no audio stream present (silent range)",
      });
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "videoAudioToVideo",
          selection: selectionWith(800, 600),
        });
      });

      await screen.findByText(/no audible sound/i, undefined, { timeout: 5_000 });
    },
    15_000,
  );

  it(
    "videoAudioToVideo shows the timeout note when extraction runs too long (#3)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({
        delayMs: 0,
        failExtractAudio: "EXTRACT_FAILED",
        extractAudioErrorMessage: "audio render failed at frame 3",
      });
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "videoAudioToVideo",
          selection: selectionWith(800, 600),
        });
      });

      await screen.findByText(/took too long and was aborted/i, undefined, { timeout: 5_000 });
    },
    15_000,
  );

  it(
    "videoAudioToVideo shows the generic failure note for an unclassified EXTRACT_FAILED (#3)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({
        delayMs: 0,
        failExtractAudio: "EXTRACT_FAILED",
        extractAudioErrorMessage: "rendering_scene_audio is not available",
      });
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "videoAudioToVideo",
          selection: selectionWith(800, 600),
        });
      });

      await screen.findByText(/audio extraction failed/i, undefined, { timeout: 5_000 });
    },
    15_000,
  );

  // --- 台帳§1-16 長尺A2V: the two Chain-targeted right-click audio items -------
  //
  // Both cut the SELECTED OBJECT'S RIBBON RANGE with `timeline.extractAudio` and
  // hand the wav to Chained's audio slot. What they must prove here is exactly
  // what a unit test cannot: that the right-click really lands on the Chained
  // tab, that the extraction arguments describe the ribbon (frameCount is
  // INCLUSIVE) with the correct mix/solo semantics, that the wav reaches the
  // AUDIO upload channel, and that the once-per-track auto-fit fires once.
  //
  // The mock's `timeline.extractAudio` reports `durationSec = frameCount * scale
  // / rate` off the project rate (30/1), so a 1200-frame ribbon is a 40s track —
  // long enough for a real multi-clip fit rather than a degenerate one.

  /** A video object occupying [frameStart, frameEnd] on `layer`. */
  function videoSelectionSpan(layer = 1, frameStart = 0, frameEnd = 1199): ResultOf<"timeline.getSelection"> {
    const sel = selectionWith(512, 320);
    return { ...sel, selected: [{ ...sel.selected[0]!, layer, frameStart, frameEnd }] };
  }

  /** An audio object occupying [frameStart, frameEnd] on `layer`. */
  function audioSelectionSpan(layer = 4, frameStart = 30, frameEnd = 1229): ResultOf<"timeline.getSelection"> {
    const sel = audioSelection();
    return { ...sel, selected: [{ ...sel.selected[0]!, layer, frameStart, frameEnd }] };
  }

  /** The auto-fit success toast, WITH the length baked into it. The 40.00s is
   * `timeline.extractAudio`'s own `durationSec` (1200 frames / 30fps) travelling
   * through `attachAudioByPath`'s `knownDurationSec` — tier 1 of the hook's
   * three-tier length resolution — so this one regex proves both that the fit
   * ran and that it ran on the EXACT measured length rather than a probe's
   * guess. Counting matches is how "exactly once" is asserted. */
  const AUDIO_FIT_TOAST = /adjusted the clips to match the 40\.00s audio/i;

  it(
    "🎬 videoAudioToLongA2v routes to Chained and mixes the ribbon's audio into the chain's audio slot",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "videoAudioToLongA2v",
          selection: videoSelectionSpan(),
        });
      });

      // Unlike #3 (which stays on Create), this one lands on Chained.
      await waitFor(
        () => expect(screen.getByRole("tab", { name: /^chained$/i })).toHaveAttribute("aria-selected", "true"),
        { timeout: 5_000 },
      );

      // Mix path, ribbon range, frameCount = frameEnd - frameStart + 1 = 1200.
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith("timeline.extractAudio", {
            layer: 1,
            frameStart: 0,
            frameCount: 1200,
            audioMode: "mix",
            soloKeepLayers: [],
          });
        },
        { timeout: 5_000 },
      );

      // The extracted wav goes to the AUDIO upload channel (never the video one).
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "backend.uploadFile",
            expect.objectContaining({ kind: "audio", filePath: expect.stringContaining("extract_1_0_") }),
          );
        },
        { timeout: 5_000 },
      );
      expect(spy).not.toHaveBeenCalledWith("backend.uploadFile", expect.objectContaining({ kind: "video" }));

      await screen.findByText(/loaded the video's audio/i, undefined, { timeout: 5_000 });

      // The wav is visibly in the Chained panel's audio card…
      await waitFor(
        () => expect(within(panel()).getByText(/extract_1_0_\d+\.wav/)).toBeInTheDocument(),
        { timeout: 5_000 },
      );
      // …and the once-per-track auto-fit fired exactly once (one toast, one
      // extraction — the mount effect is StrictMode-double-invoked).
      await waitFor(() => expect(screen.getAllByText(AUDIO_FIT_TOAST)).toHaveLength(1), { timeout: 5_000 });
      expect(spy.mock.calls.filter(([method]) => method === "timeline.extractAudio")).toHaveLength(1);
      // …without any probe round trip: `knownDurationSec` short-circuits all
      // three tiers of the length resolution.
      expect(spy).not.toHaveBeenCalledWith("fs.probeAudioDuration", expect.anything());
      expect(spy).not.toHaveBeenCalledWith("fs.probeMediaInfo", expect.anything());
    },
    20_000,
  );

  it(
    "🎵 audioToLongA2v extracts the audio object's OWN range in solo mode (its layer only)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "audioToLongA2v",
          selection: audioSelectionSpan(),
        });
      });

      await waitFor(
        () => expect(screen.getByRole("tab", { name: /^chained$/i })).toHaveAttribute("aria-selected", "true"),
        { timeout: 5_000 },
      );

      // Solo, keeping ONLY the object's own layer — and the object's own ribbon
      // (frameStart 30, 1200 frames), not the whole backing file.
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith("timeline.extractAudio", {
            layer: 4,
            frameStart: 30,
            frameCount: 1200,
            audioMode: "solo",
            soloKeepLayers: [4],
          });
        },
        { timeout: 5_000 },
      );

      // The uploaded file is the CUT wav, never the audio object's backing file
      // (that whole-file upload is #7's single-shot behaviour, not this one's).
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "backend.uploadFile",
            expect.objectContaining({ kind: "audio", filePath: expect.stringContaining("extract_4_30_") }),
          );
        },
        { timeout: 5_000 },
      );
      expect(spy).not.toHaveBeenCalledWith(
        "backend.uploadFile",
        expect.objectContaining({ filePath: "C:\\a\\voice.wav" }),
      );

      await screen.findByText(/loaded the audio object's sound/i, undefined, { timeout: 5_000 });
      await waitFor(
        () => expect(within(panel()).getByText(/extract_4_30_\d+\.wav/)).toBeInTheDocument(),
        { timeout: 5_000 },
      );
      await waitFor(() => expect(screen.getAllByText(AUDIO_FIT_TOAST)).toHaveLength(1), { timeout: 5_000 });
    },
    20_000,
  );

  it(
    "🎵 audioToLongA2v surfaces the silence note and attaches nothing when the range is silent",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({
        delayMs: 0,
        failExtractAudio: "EXTRACT_FAILED",
        extractAudioErrorMessage: "no audio stream present (silent range)",
      });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "audioToLongA2v",
          selection: audioSelectionSpan(),
        });
      });

      await screen.findByText(/no audible sound/i, undefined, { timeout: 5_000 });
      // A failed extraction must not fall back to uploading anything.
      expect(spy).not.toHaveBeenCalledWith("backend.uploadFile", expect.anything());
    },
    15_000,
  );

  // --- 台帳§1-15 W4: 🎬 referenceVideoChain (right-click -> Chained's reference) --
  //
  // The Chain-targeted twin of #2. What only an integration test can prove is
  // the chain of four hand-offs the unit tests each see one end of: the menu
  // event routes to the CHAINED tab (not Create, where #2 goes), the selected
  // video's path reaches the REFERENCE upload channel (not the source-video
  // one), the accordion W3 collapsed that slot into is opened by the mount-time
  // `useLayoutEffect` so the material is visible where it landed, and the
  // upload carries the slot's `max_frames` cap.

  /** The Chained tab's reference accordion `<details>`, found through its
   * summary. Scoped to the visible tabpanel — Create carries an accordion with
   * the identical heading, and every tab is mounted at once. */
  function referenceAccordion(): HTMLDetailsElement {
    const summary = within(panel()).getByText(/^reference video \(ic-lora\)$/i);
    const details = summary.closest("details");
    expect(details).not.toBeNull();
    return details as HTMLDetailsElement;
  }

  it(
    "🎬 referenceVideoChain routes to Chained, uploads to the reference slot, and opens its accordion",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "referenceVideoChain",
          selection: selectionWith(512, 320),
        });
      });

      // Unlike #2 (which stays on Create), this one lands on Chained.
      await waitFor(
        () => expect(screen.getByRole("tab", { name: /^chained$/i })).toHaveAttribute("aria-selected", "true"),
        { timeout: 5_000 },
      );

      // The selected video goes up as a VIDEO upload carrying the reference
      // slot's standing cap. `selectionWith` reports no playback position, so
      // `decideSourceTrim` declines and no trim keys ride along — the cap is the
      // whole query, exactly as for a manual 📁 attach.
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith("backend.uploadFile", {
            kind: "video",
            filePath: "C:\\v\\a.mp4",
            query: { max_frames: "11544" },
          });
        },
        { timeout: 5_000 },
      );

      // It landed in the REFERENCE card (the source card would show it too, so
      // the accordion scope is what distinguishes them)…
      await waitFor(
        () => expect(within(referenceAccordion()).getByText("a.mp4")).toBeInTheDocument(),
        { timeout: 5_000 },
      );
      // …and that accordion was opened on mount, so the material is not hidden
      // behind a collapsed summary. The audio one beside it stays shut.
      expect(referenceAccordion().open).toBe(true);
      const audioSummary = within(panel()).getByText(/^audio to video \(a2v\)$/i);
      expect((audioSummary.closest("details") as HTMLDetailsElement).open).toBe(false);
    },
    20_000,
  );

  // --- Replace-insert E2E: ✨ -> Generate -> 完了 -> 🎞 replaces the marker ------

  it(
    "✨ reserve -> Generate -> complete -> 🎞 replaces the job's marker in place (mode replaced, marker gone)",
    async () => {
      // The full place-and-replace loop, end to end through the real app UI. This
      // one must run on a SINGLE bridge: the job ledger / JobCard I/O go through
      // the app-wide singleton mock (the injected `nativeBridge` prop only wires
      // the menu-routing seam + direct timeline calls, not the ledger's apiClient),
      // so we render AppShell WITHOUT an injected bridge (everything -> singleton)
      // and drive the ✨ menu event + assertions against that same singleton. A ✨
      // reservation binds to the real job on Generate; once the job completes, the
      // card's 🎞 insert must REPLACE the job's provisional marker in place
      // (`timeline.insertMediaForJob` mode "replaced") — the marker disappears from
      // the timeline table and the reservation seat is freed.
      resetProvisionalReservation();
      const bridge = singletonBridge as unknown as MockBridge;
      const spy = vi.spyOn(bridge, "request");
      const user = userEvent.setup();
      render(<AppShell />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(getReservationPhase()).toBe("idle"), { timeout: 5_000 });

      // ✨ reserves a provisional at the cursor (seat -> reserved).
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "textToVideoHere",
          selection: layerSelection(0, 0),
        });
      });
      await waitFor(() => expect(getReservationPhase()).toBe("reserved"), { timeout: 5_000 });

      // Prompt + smallest preset, then Generate. The reservation binds to the real
      // job id (seat -> generating), so its marker is now keyed by that job id.
      await user.type(promptBox(), "a cat riding a skateboard");
      const presetSelect = await within(panel()).findByLabelText(/presets/i, {}, { timeout: 5_000 });
      await user.selectOptions(presetSelect, "smoke_test");
      await user.click(screen.getByRole("button", { name: /^generate$/i }));

      // The job runs to completion and its card exposes the 🎞 Insert button.
      const insertButton = await screen.findByRole("button", { name: /^insert$/i }, { timeout: 20_000 });
      await user.click(insertButton);
      await screen.findByRole("button", { name: /^inserted$/i }, { timeout: 10_000 });

      // The 🎞 went through the replace-insert RPC, and it REPLACED (not appended).
      const insertIdx = spy.mock.calls.findIndex((c) => c[0] === "timeline.insertMediaForJob");
      expect(insertIdx).toBeGreaterThanOrEqual(0);
      const jobId = (spy.mock.calls[insertIdx]![1] as { jobId: string }).jobId;
      expect(await spy.mock.results[insertIdx]!.value).toMatchObject({ mode: "replaced" });

      // The marker is gone from the timeline table (the video took its slot), and
      // the reservation seat is freed.
      const after = await bridge.request("timeline.scanProvisionals", {});
      expect(after.orphans.map((o) => o.jobId)).not.toContain(jobId);
      expect(getReservationPhase()).toBe("idle");
    },
    40_000,
  );

  // --- W8: right-click DURATION decision engine ------------------------------

  it(
    "#4 imageToVideo seeds DURATION to the resolution's comfort ceiling (512x320 -> 481, a raise from the 257 default)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "imageToVideo", selection: imageSelectionSized(512, 320) });
      });

      // Width/height applied (512x320) AND DURATION raised to the 512x320 ceiling
      // (481) — not the config default 257.
      await waitFor(
        () => {
          expect(within(panel()).getAllByDisplayValue("512").length).toBeGreaterThan(0);
          expect(
            (within(panel()).getByRole("slider", { name: /duration/i }) as HTMLInputElement).value,
          ).toBe("481");
        },
        { timeout: 5_000 },
      );
    },
    15_000,
  );

  it(
    "#4 imageToVideo lowers DURATION to the ceiling for a large resolution (1920x1088 -> 153)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "imageToVideo", selection: imageSelectionSized(1920, 1088) });
      });

      await waitFor(
        () => {
          expect(
            (within(panel()).getByRole("slider", { name: /duration/i }) as HTMLInputElement).value,
          ).toBe("153");
        },
        { timeout: 5_000 },
      );
    },
    15_000,
  );

  it(
    "#2 referenceVideo clamps DURATION to the trimmed SPAN when under the ceiling (121-frame span -> 121)",
    async () => {
      resetProvisionalReservation();
      // Pin synchronous size=material/fps=defaults so the span seed is what the
      // form shows, without the project-fps overwrite recomputing it (W1/X1). The
      // config default fps (24) equals the selection's rate (24), so the DURATION
      // resolves to the same 121 either way.
      pinMaterialPolicies();
      const bridge = createMockBridge({ delayMs: 0 });
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // 1280x768 (already on the IC-LoRA 128 grid) -> comfort ceiling 257. W4: the
      // reference now follows the object's TRIMMED span, not the full file: frames
      // 0..120 (inclusive span 121) @ selection fps 24 -> 121 frames (< 257), so
      // the span wins over the ceiling. mediaDurationSec (5) is deliberately ignored.
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "referenceVideo",
          selection: videoSelectionSized(1280, 768, 5, 24, 1),
        });
      });

      await waitFor(
        () => {
          expect(
            (within(panel()).getByRole("slider", { name: /duration/i }) as HTMLInputElement).value,
          ).toBe("121");
        },
        { timeout: 5_000 },
      );
    },
    15_000,
  );

  it(
    "#2 referenceVideo clamps DURATION to the ceiling when the trimmed SPAN is longer (721-frame span, 1920x1152 -> 153)",
    async () => {
      resetProvisionalReservation();
      pinMaterialPolicies();
      const bridge = createMockBridge({ delayMs: 0 });
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // 1920x1152 (128-aligned) -> comfort ceiling 153. W4: a long TRIMMED span
      // (frames 0..720, inclusive span 721) @24fps floors well past 153, so it
      // clamps DOWN to the 153 ceiling — a value distinct from the 257 default.
      const sel = videoSelectionSized(1920, 1152, 30, 24, 1);
      sel.selected[0]!.frameEnd = 720;
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "referenceVideo", selection: sel });
      });

      await waitFor(
        () => {
          expect(
            (within(panel()).getByRole("slider", { name: /duration/i }) as HTMLInputElement).value,
          ).toBe("153");
        },
        { timeout: 5_000 },
      );
    },
    15_000,
  );

  it(
    "#9 ✨ live-sets DURATION to the published resolution's comfort ceiling and reserves the same length (no remount)",
    async () => {
      resetProvisionalReservation();
      resetCreateLiveCommands();
      const bridge = createMockBridge({ delayMs: 0 });
      const requestSpy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // Edit width to 512 (published resolution 512x768 -> nearest-area comfort
      // ceiling is the "small" 960x576 bucket, 481 — see DEVLOG.md §41.1a).
      const widthInput = within(panel()).getByRole("slider", { name: /width/i }) as HTMLInputElement;
      fireEvent.change(widthInput, { target: { value: "512" } });
      expect(widthInput.value).toBe("512");
      // DURATION starts at the config default (361).
      expect((within(panel()).getByRole("slider", { name: /duration/i }) as HTMLInputElement).value).toBe("361");

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "textToVideoHere", selection: layerSelection(4, 360) });
      });

      // DURATION is live-set to 481 (512x768's ceiling), with no remount — the
      // edited width survives.
      await waitFor(
        () => {
          expect(
            (within(panel()).getByRole("slider", { name: /duration/i }) as HTMLInputElement).value,
          ).toBe("481");
        },
        { timeout: 5_000 },
      );
      expect((within(panel()).getByRole("slider", { name: /width/i }) as HTMLInputElement).value).toBe("512");

      // The provisional reservation uses the SAME length (481), not the old 361.
      await waitFor(
        () => {
          expect(requestSpy).toHaveBeenCalledWith(
            "timeline.insertProvisional",
            expect.objectContaining({ placement: "C", cursorLayer: 4, cursorFrame: 360, numFrames: 481 }),
          );
        },
        { timeout: 5_000 },
      );
    },
    15_000,
  );

  it(
    "#10 📸 live-sets DURATION to the published resolution's comfort ceiling (keep-panel-state, setDuration)",
    async () => {
      resetProvisionalReservation();
      resetCreateLiveCommands();
      const bridge = createMockBridge({ delayMs: 0 });
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // Edit width to 512 (published 512x768 -> ceiling 481, the "small" bucket's
      // value — see DEVLOG.md §41.1a).
      const widthInput = within(panel()).getByRole("slider", { name: /width/i }) as HTMLInputElement;
      fireEvent.change(widthInput, { target: { value: "512" } });
      expect((within(panel()).getByRole("slider", { name: /duration/i }) as HTMLInputElement).value).toBe("361");

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "imageFromCurrentFrame", selection: layerSelection(4, 360) });
      });

      // DURATION live-set to 481; the edited width survives (no remount).
      await waitFor(
        () => {
          expect(
            (within(panel()).getByRole("slider", { name: /duration/i }) as HTMLInputElement).value,
          ).toBe("481");
        },
        { timeout: 5_000 },
      );
      expect((within(panel()).getByRole("slider", { name: /width/i }) as HTMLInputElement).value).toBe("512");
    },
    15_000,
  );

  // --- W8 (owner requirement 2026-07-21 #1): the provisional reservation for a
  // MATERIAL item is placed at the DURATION-engine length from the outset (via
  // the shared `resolvePrefillSeed`), NOT the config default — so its ribbon
  // matches the remounted form the instant it appears. -----------------------

  it(
    "#4 imageToVideo reserves the provisional at the comfort-ceiling DURATION (512x320 -> numFrames 481, not the 257 default)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "imageToVideo", selection: imageSelectionSized(512, 320) });
      });

      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "timeline.insertProvisional",
            expect.objectContaining({ placement: "A", numFrames: 481 }),
          );
        },
        { timeout: 5_000 },
      );
    },
    15_000,
  );

  it(
    "#2 referenceVideo reserves the provisional at the span-derived DURATION (121-frame span @24fps -> numFrames 121)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "referenceVideo",
          selection: videoSelectionSized(1280, 768, 5, 24, 1),
        });
      });

      // #2 is placement B (head-aligned). W4: the reservation seed follows the
      // TRIMMED span (frames 0..120, inclusive span 121) @ the selection's fps 24
      // -> 121 frames (< the 1280x768 ceiling 257). The reservation always uses
      // the selection fps (the seed), independent of the fps policy.
      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "timeline.insertProvisional",
            expect.objectContaining({ placement: "B", numFrames: 121 }),
          );
        },
        { timeout: 5_000 },
      );
    },
    15_000,
  );

  it(
    "#3 videoAudioToVideo reserves the provisional at the object's SPAN-derived DURATION (121 frames @30fps span)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // selectionWith(): frames 0..120 (inclusive span 121) @ rate 30/scale 1 ->
      // 121/30 s * 30fps = 121 frames (on the 8n+1 grid). media 800x600 -> ceiling
      // 481, so no clamp. Placement B (§5-9). The A2V wav auto-adjust later
      // supersedes the FORM value, but the RESERVATION is placed at this seed.
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "videoAudioToVideo",
          selection: selectionWith(800, 600),
        });
      });

      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "timeline.insertProvisional",
            expect.objectContaining({ placement: "B", numFrames: 121 }),
          );
        },
        { timeout: 5_000 },
      );
    },
    15_000,
  );

  it(
    "#7 audioToVideo reserves the provisional at the span-derived DURATION (121-frame span @30fps -> numFrames 121)",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // W4: #7 now follows the audio object's TRIMMED timeline span, not the full
      // file. audioSelection() frames 10..130 (inclusive span 121) @ rate 30 ->
      // 121/30 s * 30fps = 121 frames; the audio has no media resolution, so
      // DURATION resolves off the project-default 1280x768 (ceiling 257), leaving
      // 121 unclamped. mediaDurationSec (3, full file) is deliberately ignored.
      // Placement B.
      const sized = (() => {
        const sel = audioSelection();
        return { ...sel, selected: [{ ...sel.selected[0]!, mediaDurationSec: 3 }] };
      })();
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "audioToVideo", selection: sized });
      });

      await waitFor(
        () => {
          expect(spy).toHaveBeenCalledWith(
            "timeline.insertProvisional",
            expect.objectContaining({ placement: "B", numFrames: 121 }),
          );
        },
        { timeout: 5_000 },
      );
    },
    15_000,
  );

  // --- W5: 反対スロット保持 (carry the opposite source slot across a remount) ---

  // The mode badges (`.mode-badge`) are the cleanest live signal of which source
  // slots are populated: "IC-LoRA" appears while a reference video is attached,
  // "A2V" while a source audio is. Reading them off the visible panel proves a
  // carried slot survived the full form-resetting remount.
  const modeBadges = () => Array.from(panel().querySelectorAll(".mode-badge")).map((el) => el.textContent);
  const uploadsOfKind = (spy: ReturnType<typeof vi.spyOn>, kind: string) =>
    spy.mock.calls.filter(
      (c: unknown[]) => c[0] === "backend.uploadFile" && (c[1] as { kind?: string } | undefined)?.kind === kind,
    );

  it(
    "#2 referenceVideo carries the existing A2V audio across the remount, without re-uploading it (W5)",
    async () => {
      resetProvisionalReservation();
      pinMaterialPolicies();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // First fill the A2V audio slot via a #7 route (its own remount).
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "audioToVideo", selection: audioSelection() });
      });
      await waitFor(() => expect(modeBadges()).toContain("A2V"), { timeout: 5_000 });
      expect(uploadsOfKind(spy, "audio")).toHaveLength(1);

      // Now #2 reference-video: the whole Create form resets, but the OPPOSITE
      // slot (the audio) must carry over.
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "referenceVideo", selection: selectionWith(1000, 700) });
      });

      // The new reference video loads (IC-LoRA badge) AND the carried audio
      // survives the remount (A2V badge still present).
      await waitFor(
        () => expect(modeBadges()).toEqual(expect.arrayContaining(["IC-LoRA", "A2V"])),
        { timeout: 5_000 },
      );

      // The carried audio id was reused — NOT re-uploaded across the remount…
      expect(uploadsOfKind(spy, "audio")).toHaveLength(1);
      // …while the new reference video itself WAS uploaded.
      expect(spy).toHaveBeenCalledWith(
        "backend.uploadFile",
        expect.objectContaining({ kind: "video", filePath: "C:\\v\\a.mp4" }),
      );
    },
    20_000,
  );

  it(
    "#7 audioToVideo carries the existing IC-LoRA reference video + keeps the 128 grid across the remount (W5, 回帰対策1)",
    async () => {
      resetProvisionalReservation();
      pinMaterialPolicies();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      // First fill the IC-LoRA reference slot via a #2 route. The IC-LoRA badge
      // is INTENT-driven (shows before the upload settles), so wait on the
      // published slot instead — that is the exact source AppShell reads at #7.
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "referenceVideo", selection: selectionWith(1000, 700) });
      });
      await waitFor(() => expect(getPublishedSourceSlots().referenceVideo).not.toBeNull(), { timeout: 5_000 });
      expect(uploadsOfKind(spy, "video")).toHaveLength(1);

      // Now #7 audio-to-video: the form resets, but the reference video carries.
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "audioToVideo", selection: audioSelection() });
      });

      // The new audio loads (A2V badge) AND the carried reference video survives
      // (IC-LoRA badge still present).
      await waitFor(
        () => expect(modeBadges()).toEqual(expect.arrayContaining(["IC-LoRA", "A2V"])),
        { timeout: 5_000 },
      );

      // The reference video was NOT re-uploaded; only the new audio uploaded.
      expect(uploadsOfKind(spy, "video")).toHaveLength(1);
      expect(spy).toHaveBeenCalledWith(
        "backend.uploadFile",
        expect.objectContaining({ kind: "audio", filePath: "C:\\a\\voice.wav" }),
      );

      // 回帰対策1: the carried reference video forces the 128 grid, so the width
      // stays a multiple of 128 and Generate is never blocked as `dimensionsOffGrid`.
      const widthInput = within(panel()).getByRole("slider", { name: /width/i }) as HTMLInputElement;
      expect(Number(widthInput.value) % 128).toBe(0);
    },
    20_000,
  );

  // --- 素材（末尾）系統E: 末尾合わせの予約 ----------------------------------
  //
  // The result ENDS with the material, so the provisional has to start
  // `出力長 − 凍結分` frames BEFORE it. Native knows no such placement, so
  // `AppShell` Step 8 computes the shift and `placementParams` maps "E" onto
  // native's "B" — which is why the assertion below is about "B" plus a
  // SHIFTED `materialFrameStart`, not about an "E" ever reaching the wire.
  it(
    "endWithThis reserves TAIL-ALIGNED: placement B on the wire, with the start shifted back off the material",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(getReservationPhase()).toBe("idle"));

      // A 10-second 1280x720 video sitting at frames 600..720 of a 24fps project.
      const selection = videoSelectionSized(1280, 720, 10, 24, 1);
      const material = { ...selection.selected[0]!, layer: 1, frameStart: 600, frameEnd: 720 };
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, {
          action: "endWithThis",
          selection: { ...selection, selected: [material], rangeStart: 600, rangeEnd: 720 },
        });
      });

      await waitFor(
        () => expect(spy).toHaveBeenCalledWith("timeline.insertProvisional", expect.anything()),
        { timeout: 5_000 },
      );
      const call = spy.mock.calls.find(([method]) => method === "timeline.insertProvisional")!;
      const params = call[1] as {
        placement: string;
        materialLayer: number;
        materialFrameStart: number;
        numFrames: number;
        genFps: number;
      };

      // "E" is a WebUI-only系統 — it must never reach native.
      expect(params.placement).toBe("B");
      // 窓内モード: the reserved LENGTH is the clip length itself — the material
      // is frozen into the last 8 frames of it, never appended after it — and
      // the seed is rounded down to one stage-2 window (169).
      expect(params.numFrames).toBe(END_SOURCE_SEED_MAX_FRAMES);
      expect(END_SOURCE_SEED_MAX_FRAMES).toBe(169);
      // The material's own start is 600; the provisional starts EARLIER, by the
      // part that does NOT overlap the material, converted to project frames.
      // Both rates are 24 here, so the shift is `numFrames − 9` exactly — the
      // extra 1 is the causal VAE's keyframe primer (材料の2〜9フレーム目が
      // 錨8fに重なるため、素材の1フレーム目はさらに1フレーム手前にある).
      expect(params.materialFrameStart).toBe(
        600 - (params.numFrames - (END_SOURCE_CONTEXT_FRAMES + 1)),
      );
      expect(params.materialFrameStart).toBeLessThan(600);
      expect(params.materialLayer).toBe(1);
      expect(getReservationPhase()).toBe("reserved");
    },
    20_000,
  );

  it(
    "endWithThis clamps to frame 0 for material near the head of the timeline",
    async () => {
      resetProvisionalReservation();
      const bridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(bridge, "request");
      render(<AppShell nativeBridge={bridge} />);
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await waitFor(() => expect(getReservationPhase()).toBe("idle"));

      // The same material, but at frames 0..120: the tail-aligned start would be
      // deeply negative, and a negative frame is not a place.
      const selection = videoSelectionSized(1280, 720, 10, 24, 1);
      act(() => {
        bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "endWithThis", selection });
      });

      await waitFor(
        () => expect(spy).toHaveBeenCalledWith("timeline.insertProvisional", expect.anything()),
        { timeout: 5_000 },
      );
      const call = spy.mock.calls.find(([method]) => method === "timeline.insertProvisional")!;
      expect(call[1]).toMatchObject({ placement: "B", materialFrameStart: 0 });
    },
    20_000,
  );
});
