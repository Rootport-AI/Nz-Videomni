import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { NativeBridge } from "../../bridge";
import { createMockBridge } from "../../bridge/mockBridge";
import type { MockBridgeOptions } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { OUTPAINT_LORA_NAME } from "../../lora/controlLoras";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import { ChainReferencePanel } from "./ChainReferencePanel";
import { useChainForm } from "./useChainForm";
import type { UseChainFormResult } from "./useChainForm";

/**
 * §1-15 (clip-wise IC-LoRA reference), plan F3: Chain's own reference-video
 * card. Same two-harness recipe as `ChainAudioPanel.test.tsx`: the default
 * harness drives the REAL `useChainForm` behind the real panel through the
 * mock bridge (attach/drop/clear/pick, the control-LoRA select, the two
 * opt-in strength sliders); the note-priority matrix spreads a handful of
 * overrides over the same real hook result, since driving a V2V conflict or a
 * failed probe through the real hook would make the SETUP — not the
 * assertion — the thing under test.
 */

const CONTROL_NAMES = new Set(["canny-control", "pose-control"]);

function Harness({
  nativeBridge,
  overrides,
  disabled = false,
  controlLoraNames = CONTROL_NAMES,
}: {
  nativeBridge: NativeBridge;
  overrides?: Partial<UseChainFormResult>;
  disabled?: boolean;
  controlLoraNames?: ReadonlySet<string>;
}) {
  const form = useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", {
    nativeBridge,
    controlLoraNames,
  });
  return (
    <LanguageProvider>
      <ChainReferencePanel form={{ ...form, ...overrides }} disabled={disabled} nativeBridge={nativeBridge} />
    </LanguageProvider>
  );
}

const t = en.chained.referenceVideo;

function setup(options: MockBridgeOptions = {}, overrides?: Partial<UseChainFormResult>) {
  const nativeBridge = createMockBridge({ delayMs: 0, ...options });
  const view = render(<Harness nativeBridge={nativeBridge} {...(overrides ? { overrides } : {})} />);
  return { ...view, nativeBridge };
}

/** Drops one video file onto the card (the whole card is the drop target). */
function dropReference(container: HTMLElement, fileName = "ref.mp4") {
  fireEvent.drop(container.querySelector(".chain-reference-section")!, {
    dataTransfer: { files: [new File([], fileName)] },
  });
}

function noteText(container: HTMLElement): string {
  return container.querySelector(".chain-reference-note")!.textContent ?? "";
}

describe("ChainReferencePanel (§1-15 参照動画)", () => {
  describe("the three attach states", () => {
    it("renders the unattached state: choose button, 'no reference' note, 🔁 disabled, no thumbnail", () => {
      const { container } = setup();

      expect(screen.getByText(t.heading)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: t.chooseButton })).toBeEnabled();
      expect(screen.getByRole("button", { name: t.clearButton })).toBeDisabled();
      expect(noteText(container)).toBe(t.none);
      expect(container.querySelector(".keyframe-thumb img")).toBeNull();
    });

    it("the fps reminder is mounted regardless of attach state", () => {
      setup();
      expect(screen.getByText(t.fpsNote)).toBeInTheDocument();
    });

    it("shows the spinner and the 'Uploading…' label while the upload is in flight", async () => {
      const { container, nativeBridge } = setup({ holdUploads: true });

      dropReference(container);

      await waitFor(() => expect(screen.getByRole("button", { name: t.uploadingButton })).toBeDisabled());
      expect(screen.getByRole("status")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: t.clearButton })).toBeDisabled();

      (nativeBridge as ReturnType<typeof createMockBridge>).releaseUploads();
      await waitFor(() => expect(screen.getByRole("button", { name: t.changeButton })).toBeInTheDocument());
    });

    it("shows the thumbnail, the file name and the measured resolution once ready", async () => {
      const { container } = setup({ probeMediaInfoWidth: 1280, probeMediaInfoHeight: 768 });

      dropReference(container);

      await waitFor(() => expect(screen.getByAltText(t.videoPlaceholderAlt)).toBeInTheDocument());
      expect(screen.getByText("ref.mp4")).toBeInTheDocument();
      expect(await screen.findByText(t.resolutionLabel(1280, 768))).toBeInTheDocument();
      expect(screen.getByRole("button", { name: t.clearButton })).toBeEnabled();
      expect(screen.getByRole("button", { name: t.changeButton })).toBeInTheDocument();
    });

    it("never shows a duration — the reference is sliced by frame number, not seconds", async () => {
      const { container } = setup({ probeMediaInfoWidth: 1280, probeMediaInfoHeight: 768, probeMediaInfoDurationSec: 42 });

      dropReference(container);

      await waitFor(() => expect(screen.getByAltText(t.videoPlaceholderAlt)).toBeInTheDocument());
      expect(screen.queryByText(/42/)).not.toBeInTheDocument();
    });

    it("📁 attaches whatever the native dialog returned", async () => {
      const user = userEvent.setup();
      setup({ pickFileName: "control.mov", probeMediaInfoWidth: 640, probeMediaInfoHeight: 480 });

      await user.click(screen.getByRole("button", { name: t.chooseButton }));

      await waitFor(() => expect(screen.getByText("control.mov")).toBeInTheDocument());
      expect(screen.getByAltText(t.videoPlaceholderAlt)).toBeInTheDocument();
    });

    it("🔁 detaches the video and returns the card to the unattached state", async () => {
      const user = userEvent.setup();
      const { container } = setup({ probeMediaInfoWidth: 1280, probeMediaInfoHeight: 768 });

      dropReference(container);
      await waitFor(() => expect(screen.getByRole("button", { name: t.clearButton })).toBeEnabled());

      await user.click(screen.getByRole("button", { name: t.clearButton }));

      expect(noteText(container)).toBe(t.none);
      expect(screen.getByRole("button", { name: t.clearButton })).toBeDisabled();
      expect(container.querySelector(".keyframe-thumb img")).toBeNull();
    });

    it("rejects a non-video drop inline and leaves the slot empty", async () => {
      const { container } = setup();

      dropReference(container, "notes.txt");

      await waitFor(() =>
        expect(container.querySelector(".field-hint-error")?.textContent).toBe(en.dnd.unsupportedVideo),
      );
      expect(screen.getByRole("button", { name: t.clearButton })).toBeDisabled();
    });
  });

  describe("the control-LoRA select and its strength slider", () => {
    it("lists the form's controlLoraNames, 'select' first", () => {
      setup();

      const select = screen.getByRole("combobox") as HTMLSelectElement;
      const optionLabels = Array.from(select.options).map((o) => o.textContent);
      expect(optionLabels).toEqual([t.controlLoraNoneOption, "canny-control", "pose-control"]);
      // No strength slider until a control LoRA is actually selected.
      expect(screen.queryByText(t.controlLoraStrengthLabel)).not.toBeInTheDocument();
    });

    it("excludes in-outpainting from the dropdown even when the form reports it as a control LoRA", () => {
      // §1-13/T2 (2026-08-11): `in-outpainting` has its own dedicated place —
      // the Edit tab's Outpainting panel — so Chain's own dropdown must never
      // offer it, even though `form.controlLoraNames` (unfiltered, per
      // `lora/controlLoras.ts`'s doc comment) legitimately contains it.
      const nativeBridge = createMockBridge({ delayMs: 0 });
      render(
        <Harness
          nativeBridge={nativeBridge}
          controlLoraNames={new Set(["canny-control", OUTPAINT_LORA_NAME, "pose-control"])}
        />,
      );

      const select = screen.getByRole("combobox") as HTMLSelectElement;
      const optionLabels = Array.from(select.options).map((o) => o.textContent);
      expect(optionLabels).toEqual([t.controlLoraNoneOption, "canny-control", "pose-control"]);
      expect(optionLabels).not.toContain(OUTPAINT_LORA_NAME);
    });

    it("choosing a name reveals the strength slider, seeded at the default", async () => {
      const user = userEvent.setup();
      setup();

      await user.selectOptions(screen.getByRole("combobox"), "canny-control");

      expect(screen.getByText(t.controlLoraStrengthLabel)).toBeInTheDocument();
      expect(screen.getByText("1.00")).toBeInTheDocument();
    });

    it("calls setControlLoraStrength when the slider moves", () => {
      const setControlLoraStrength = vi.fn();
      setup({}, { controlLora: { name: "canny-control", strength: 1.0 }, setControlLoraStrength });

      fireEvent.change(screen.getByRole("slider"), { target: { value: "0.5" } });

      expect(setControlLoraStrength).toHaveBeenCalledWith(0.5);
    });
  });

  // One always-mounted line, one message at a time — see the panel's own doc
  // comment. Each case below pins BOTH the message and that it won.
  describe("the priority note's order", () => {
    it("1. a V2V + reference conflict is a red error, beating every other case", () => {
      const { container } = setup(
        {},
        { hasSourceVideo: true, hasReferenceVideo: true, referenceMediaSize: null },
      );

      expect(noteText(container)).toBe(t.conflictsWithSourceVideo);
      expect(container.querySelector(".chain-reference-note")).toHaveClass("field-hint-error");
    });

    it("2. an unreadable resolution beats the short-reference reminder", () => {
      const { container } = setup(
        {},
        { referenceVideo: { state: { status: "ready", fileName: "r.mp4", filePath: "C:\\r.mp4", id: "v1", errorCode: null, trimFailed: false, frames: null, fps: null }, pick: async () => {}, uploadPath: async () => {}, clear: () => {} }, hasReferenceVideo: true, referenceMediaSize: null },
      );

      expect(noteText(container)).toBe(t.probeFailed);
      expect(container.querySelector(".chain-reference-note")).not.toHaveClass("field-hint-error");
    });

    it("3. attached, measured, no conflict — the static short-reference reminder", () => {
      const { container } = setup(
        {},
        { hasReferenceVideo: true, referenceMediaSize: { width: 1280, height: 768 } },
      );

      expect(noteText(container)).toBe(t.shortReferenceNote);
    });

    it("4. nothing attached — the plain 'no reference video' note", () => {
      const { container } = setup();
      expect(noteText(container)).toBe(t.none);
    });

    it("is mounted in every state, so the card never reflows", () => {
      const { container } = setup();
      expect(container.querySelectorAll(".chain-reference-note")).toHaveLength(1);

      const attached = setup({}, { hasSourceVideo: true, hasReferenceVideo: true });
      expect(attached.container.querySelectorAll(".chain-reference-note")).toHaveLength(1);
    });
  });

  describe("the two opt-in strength sliders", () => {
    it("are absent until the reference is ready", () => {
      setup();
      expect(screen.queryByText(t.conditioningAttentionStrengthLabel)).not.toBeInTheDocument();
      expect(screen.queryByText(t.referenceVideoStrengthLabel)).not.toBeInTheDocument();
    });

    it("appear once ready, and ticking one calls the form's setter", async () => {
      const user = userEvent.setup();
      const { container } = setup({ probeMediaInfoWidth: 1280, probeMediaInfoHeight: 768 });

      dropReference(container);
      await waitFor(() => expect(screen.getByAltText(t.videoPlaceholderAlt)).toBeInTheDocument());

      expect(screen.getByText(t.conditioningAttentionStrengthLabel)).toBeInTheDocument();
      expect(screen.getByText(t.referenceVideoStrengthLabel)).toBeInTheDocument();

      await user.click(screen.getByRole("checkbox", { name: t.conditioningAttentionStrengthLabel }));

      expect(screen.getByRole("checkbox", { name: t.conditioningAttentionStrengthLabel })).toBeChecked();
    });
  });
});
