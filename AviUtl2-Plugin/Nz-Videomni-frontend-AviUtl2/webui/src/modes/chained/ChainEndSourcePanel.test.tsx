import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { NativeBridge } from "../../bridge";
import { createMockBridge } from "../../bridge/mockBridge";
import type { MockBridgeOptions } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import { ChainEndSourcePanel } from "./ChainEndSourcePanel";
import { useChainForm } from "./useChainForm";
import type { UseChainFormResult } from "./useChainForm";

/**
 * 素材（末尾）: the end-source card. Same two-harness recipe as
 * `ChainReferencePanel.test.tsx` — the default harness drives the REAL
 * `useChainForm` behind the real panel through the mock bridge (drop/pick/clear,
 * the derived band readout, the image-only note), while the note-priority matrix
 * spreads a handful of overrides over that same real result, since driving an
 * audio/reference conflict through the real hook would make the SETUP, not the
 * assertion, the thing under test.
 *
 * 窓内モード (2026-08-17): the card has NO control left — the "末尾フレーム数"
 * slider is gone and the anchor is the fixed 8 — so the slider cases below are
 * "there is no slider" cases, the note matrix carries the two length failures,
 * and the clip-length quality warning gets its own block (it is a SECOND line
 * below the note, not part of the priority rotation).
 */
function Harness({
  nativeBridge,
  overrides,
  disabled = false,
}: {
  nativeBridge: NativeBridge;
  overrides?: Partial<UseChainFormResult>;
  disabled?: boolean;
}) {
  const form = useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge });
  return (
    <LanguageProvider>
      <ChainEndSourcePanel form={{ ...form, ...overrides }} disabled={disabled} nativeBridge={nativeBridge} />
    </LanguageProvider>
  );
}

const t = en.chained.endSource;

function setup(options: MockBridgeOptions = {}, overrides?: Partial<UseChainFormResult>) {
  const nativeBridge = createMockBridge({ delayMs: 0, ...options });
  const view = render(<Harness nativeBridge={nativeBridge} {...(overrides ? { overrides } : {})} />);
  return { ...view, nativeBridge };
}

/** Drops one file onto the card (the whole card is the drop target). */
function dropEndSource(container: HTMLElement, fileName: string) {
  fireEvent.drop(container.querySelector(".chain-end-source-section")!, {
    dataTransfer: { files: [new File([], fileName)] },
  });
}

function noteText(container: HTMLElement): string {
  return container.querySelector(".chain-end-source-note")!.textContent ?? "";
}

/** The strength slider — the only control this card has (バッチ1・2026-08-18).
 * Scoped to its own class rather than "any range input in the controls
 * column" so this stays pointed at the strength control specifically if a
 * future control is ever added alongside it. */
function strengthSlider(container: HTMLElement): HTMLInputElement | null {
  return container.querySelector<HTMLInputElement>(".chain-end-source-strength input[type=\"range\"]");
}

describe("ChainEndSourcePanel (素材（末尾）)", () => {
  describe("the attach states", () => {
    it("renders the unattached state: choose button, 'none' note, 🔁 disabled, strength slider at its 1.0 default", () => {
      const { container } = setup();

      expect(screen.getByText(t.heading)).toBeInTheDocument();
      expect(screen.getByText(t.note)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: t.chooseButton })).toBeEnabled();
      expect(screen.getByRole("button", { name: t.clearButton })).toBeDisabled();
      expect(noteText(container)).toBe(t.none);
      // バッチ1 (2026-08-18): unconditional, like `overlapFrames`/`overlapStrength`
      // — it seeds the request value whether or not material is attached yet.
      expect(strengthSlider(container)?.value).toBe("1");
    });

    it("shows the spinner and the 'Uploading…' label while the upload is in flight", async () => {
      const { container, nativeBridge } = setup({ holdUploads: true });

      dropEndSource(container, "tail.mp4");

      await waitFor(() => expect(screen.getByRole("button", { name: t.uploadingButton })).toBeDisabled());
      expect(screen.getByRole("status")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: t.clearButton })).toBeDisabled();

      (nativeBridge as ReturnType<typeof createMockBridge>).releaseUploads();
      await waitFor(() => expect(screen.getByRole("button", { name: t.changeButton })).toBeInTheDocument());
    });

    it("a dropped VIDEO shows the placeholder, the file name and the 8-frame anchor — the anchor LENGTH has no control", async () => {
      const { container } = setup();

      dropEndSource(container, "tail.mp4");

      await waitFor(() => expect(screen.getByAltText(t.videoPlaceholderAlt)).toBeInTheDocument());
      expect(screen.getByText("tail.mp4")).toBeInTheDocument();
      // Nothing to set for the LENGTH — the anchor is reported, never chosen.
      // The strength slider is a separate, unconditional control (see above).
      await waitFor(() => expect(noteText(container)).toBe(t.contextFramesHint(8)));
      expect(strengthSlider(container)).not.toBeNull();
    });

    it("reports the SAME 8 whatever the material's own length is", async () => {
      // 窓内モード: a 73-frame material and a 300-frame one are indistinguishable
      // here — the readout is about the OUTPUT's tail, not about the material.
      const { container } = setup({ uploadVideoFrameCount: 73, uploadVideoFps: 24 });

      dropEndSource(container, "tail.mp4");

      await waitFor(() => expect(noteText(container)).toBe(t.contextFramesHint(8)));
    });

    it("says the output does NOT get longer", () => {
      // The one sentence the owner's real-device check reads first: attaching
      // material must not be understood as extending the video.
      expect(t.contextFramesHint(8)).toContain("does not get longer");
    });

    it("shows the too-short error for a material below the 9-frame floor", async () => {
      const { container } = setup({ uploadVideoFrameCount: 8, uploadVideoFps: 24 });

      dropEndSource(container, "tail.mp4");

      await waitFor(() => expect(noteText(container)).toBe(t.tooShortNote));
      expect(container.querySelector(".chain-end-source-note")).toHaveClass("field-hint-error");
    });

    it("shows the unknown-length error when the server measured nothing", async () => {
      const { container } = setup({ uploadVideoFrameCount: null, uploadVideoFps: null });

      dropEndSource(container, "tail.mp4");

      await waitFor(() => expect(noteText(container)).toBe(t.lengthUnknownNote));
      expect(container.querySelector(".chain-end-source-note")).toHaveClass("field-hint-error");
    });

    it("a dropped IMAGE shows the still-image warning; the strength slider is still there (unconditional)", async () => {
      const { container } = setup();

      dropEndSource(container, "tail.png");

      await waitFor(() => expect(noteText(container)).toBe(t.stillImageNote));
      // The exact owner-agreed sentence, not a paraphrase.
      expect(t.stillImageNote).toBe("The last 8 frames will be a still image.");
      expect(strengthSlider(container)).not.toBeNull();
    });

    it("🔁 detaches the material and returns the card to the unattached state", async () => {
      const user = userEvent.setup();
      const { container } = setup();

      dropEndSource(container, "tail.mp4");
      await waitFor(() => expect(screen.getByRole("button", { name: t.clearButton })).toBeEnabled());

      await user.click(screen.getByRole("button", { name: t.clearButton }));

      expect(noteText(container)).toBe(t.none);
      expect(screen.getByRole("button", { name: t.clearButton })).toBeDisabled();
      expect(strengthSlider(container)).not.toBeNull();
    });

    it("📁 attaches whatever the native dialog returned", async () => {
      const user = userEvent.setup();
      setup({ pickFileName: "ending.mov" });

      await user.click(screen.getByRole("button", { name: t.chooseButton }));

      await waitFor(() => expect(screen.getByText("ending.mov")).toBeInTheDocument());
      expect(screen.getByAltText(t.videoPlaceholderAlt)).toBeInTheDocument();
    });

    it("rejects an unsupported drop inline and leaves the slot empty", async () => {
      const { container } = setup();

      dropEndSource(container, "notes.txt");

      await waitFor(() =>
        expect(container.querySelector(".field-hint-error")?.textContent).toBe(en.dnd.unsupportedSource),
      );
      expect(screen.getByRole("button", { name: t.clearButton })).toBeDisabled();
    });
  });

  // バッチ1 (2026-08-18): v1's inoperable audio-mode radio pair (a
  // `<fieldset disabled>`) is gone — the one sentence below now covers the
  // same ground on its own.
  describe("the audio note", () => {
    it("renders no audio radios at all (regression: the v1 mock is gone for good)", () => {
      setup();
      expect(screen.queryAllByRole("radio")).toHaveLength(0);
    });

    it("says in words that the material's audio is carried into the tail when present", () => {
      setup();
      expect(screen.getByText(t.audioModeNote)).toBeInTheDocument();
    });
  });

  // バッチ1 (2026-08-18): the anchor's fixed-strength slider — the one control
  // this card has. Unconditional (see the "unattached state" test above), so
  // these cases don't need any material attached either.
  describe("the strength slider", () => {
    it("opens at the 1.0 default", () => {
      const { container } = setup();
      expect(strengthSlider(container)?.value).toBe("1");
    });

    it("calls setEndSourceStrength with the slider's new value", () => {
      const setEndSourceStrength = vi.fn();
      const { container } = setup({}, { setEndSourceStrength });

      fireEvent.change(strengthSlider(container)!, { target: { value: "0.5" } });

      expect(setEndSourceStrength).toHaveBeenCalledWith(0.5);
    });

    it("shows the current value to two decimal places", () => {
      const { container } = setup({}, { endSourceStrength: 0.25 });
      expect(strengthSlider(container)?.value).toBe("0.25");
      expect(screen.getByText("0.25")).toBeInTheDocument();
    });
  });

  // One always-mounted line, one message at a time — see the panel's own doc
  // comment. Each case pins BOTH the message and that it won.
  describe("the priority note's order", () => {
    it("1. an end-source + audio conflict is a red error, beating everything else", () => {
      const { container } = setup({}, { hasEndSource: true, hasSourceAudio: true, endSourceKind: "image" });

      expect(noteText(container)).toBe(t.conflictsWithAudio);
      expect(container.querySelector(".chain-end-source-note")).toHaveClass("field-hint-error");
    });

    it("2. an end-source + reference conflict is the other red error", () => {
      const { container } = setup({}, { hasEndSource: true, hasReferenceVideo: true, endSourceKind: "video" });

      expect(noteText(container)).toBe(t.conflictsWithReference);
      expect(container.querySelector(".chain-end-source-note")).toHaveClass("field-hint-error");
    });

    it("3. an image's still-image warning beats the video readout", () => {
      const { container } = setup({}, { hasEndSource: true, endSourceKind: "image" });
      expect(noteText(container)).toBe(t.stillImageNote);
      expect(container.querySelector(".chain-end-source-note")).not.toHaveClass("field-hint-error");
    });

    it("3b. a too-short material is a red error that beats the frame readout", () => {
      const { container } = setup(
        {},
        { hasEndSource: true, endSourceKind: "video", endContextFrames: null, endSourceLengthIssue: "tooShort" },
      );
      expect(noteText(container)).toBe(t.tooShortNote);
      expect(container.querySelector(".chain-end-source-note")).toHaveClass("field-hint-error");
    });

    it("3c. an unmeasurable material is the other red length error", () => {
      const { container } = setup(
        {},
        { hasEndSource: true, endSourceKind: "video", endContextFrames: null, endSourceLengthIssue: "unknown" },
      );
      expect(noteText(container)).toBe(t.lengthUnknownNote);
      expect(container.querySelector(".chain-end-source-note")).toHaveClass("field-hint-error");
    });

    it("4. an attached video reports the anchor", () => {
      const { container } = setup({}, { hasEndSource: true, endSourceKind: "video", endContextFrames: 8 });
      expect(noteText(container)).toBe(t.contextFramesHint(8));
    });

    it("5. nothing attached — the plain 'no end source' note", () => {
      const { container } = setup();
      expect(noteText(container)).toBe(t.none);
    });

    it("is mounted in every state, so the card never reflows", () => {
      const { container } = setup();
      expect(container.querySelectorAll(".chain-end-source-note")).toHaveLength(1);

      const attached = setup({}, { hasEndSource: true, endSourceKind: "video" });
      expect(attached.container.querySelectorAll(".chain-end-source-note")).toHaveLength(1);
    });
  });

  // 窓内モード品質警告: a SECOND line, below the priority note. It is about the
  // CLIP rather than the material, so it coexists with whatever the note says
  // instead of competing for the same slot.
  describe("the clip-length quality warning", () => {
    function warning(container: HTMLElement): HTMLElement | null {
      return container.querySelector<HTMLElement>(".warning-banner-mild");
    }

    it("is absent while the hook reports no limit", () => {
      const { container } = setup({}, { hasEndSource: true, endSourceKind: "video", endContextFrames: 8 });
      expect(warning(container)).toBeNull();
    });

    it("renders the limit the hook reports, as a mild (non-blocking) banner", () => {
      const { container } = setup(
        {},
        { hasEndSource: true, endSourceKind: "video", endContextFrames: 8, endSourceQualityLimitFrames: 169 },
      );
      expect(warning(container)?.textContent).toBe(t.qualityWarning(169));
      expect(warning(container)).toHaveClass("warning-banner");
    });

    it("follows the hook down to the high-resolution window's 145", () => {
      const { container } = setup(
        {},
        { hasEndSource: true, endSourceKind: "video", endContextFrames: 8, endSourceQualityLimitFrames: 145 },
      );
      expect(warning(container)?.textContent).toBe(t.qualityWarning(145));
    });

    it("coexists with the priority note instead of replacing it", () => {
      const { container } = setup(
        {},
        { hasEndSource: true, endSourceKind: "image", endSourceQualityLimitFrames: 169 },
      );
      expect(noteText(container)).toBe(t.stillImageNote);
      expect(warning(container)?.textContent).toBe(t.qualityWarning(169));
    });
  });

  // §1-22 素材（末尾）複数クリップ品質警告: a THIRD line, the multi-clip
  // counterpart of the clip-length warning above. The two never fire
  // together in practice (`useChainForm`'s own trigger conditions are
  // mutually exclusive over `clips.length`), so each is exercised here in
  // isolation via its own override.
  describe("the multi-clip quality warning", () => {
    function multiClipWarning(container: HTMLElement): HTMLElement | null {
      return container.querySelector<HTMLElement>(".chain-end-source-multiclip-warning");
    }

    it("is absent while the hook reports no warning", () => {
      const { container } = setup(
        {},
        { hasEndSource: true, endSourceKind: "video", endSourceMultiClipQualityWarning: false },
      );
      expect(multiClipWarning(container)).toBeNull();
    });

    it("renders the owner-worded warning as a mild (non-blocking) banner", () => {
      const { container } = setup(
        {},
        { hasEndSource: true, endSourceKind: "video", endSourceMultiClipQualityWarning: true },
      );
      expect(multiClipWarning(container)?.textContent).toBe(t.multiClipQualityWarning);
      expect(multiClipWarning(container)).toHaveClass("warning-banner");
      expect(multiClipWarning(container)).toHaveClass("warning-banner-mild");
    });

    it("coexists with the priority note instead of replacing it", () => {
      const { container } = setup(
        {},
        { hasEndSource: true, endSourceKind: "image", endSourceMultiClipQualityWarning: true },
      );
      expect(noteText(container)).toBe(t.stillImageNote);
      expect(multiClipWarning(container)?.textContent).toBe(t.multiClipQualityWarning);
    });
  });

  it("its Clear button is NOT named the same as the START slot's (accessibility)", () => {
    expect(t.clearButton).not.toBe(en.chained.sourceInput.clearButton);
  });
});
