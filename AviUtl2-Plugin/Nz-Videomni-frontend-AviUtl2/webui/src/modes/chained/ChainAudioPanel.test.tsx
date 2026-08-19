import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { NativeBridge } from "../../bridge";
import { createMockBridge } from "../../bridge/mockBridge";
import type { MockBridgeOptions } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import { ChainAudioPanel } from "./ChainAudioPanel";
import { useChainForm } from "./useChainForm";
import type { UseChainFormResult } from "./useChainForm";

/**
 * §1-16 長尺A2V: Chain's own audio card.
 *
 * Two harnesses, on purpose. The default one is `SourceInputPanel.test.tsx`'s
 * exactly — the REAL `useChainForm` behind the real panel, driven through the
 * mock bridge — and covers everything the user can actually do (pick, drop,
 * clear, press ↔️, move the slider).
 *
 * The note-priority matrix cannot be driven that way: several of those states
 * (an over-long attach, the 24-clip ceiling, a leftover tail) require an audio
 * file of a specific length AND a clip list the once-per-track auto-fit has
 * already rearranged, which makes the setup — not the assertion — the thing
 * under test. Those cases spread a handful of overrides over the SAME real
 * hook result (`overrides`), so the object is still a genuine
 * `UseChainFormResult` with no casts: only the field being exercised is
 * synthetic. The states' own derivation is pinned in `useChainForm.test.ts`.
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
      <ChainAudioPanel form={{ ...form, ...overrides }} disabled={disabled} nativeBridge={nativeBridge} />
    </LanguageProvider>
  );
}

const t = en.chained.sourceAudio;

function setup(options: MockBridgeOptions = {}, overrides?: Partial<UseChainFormResult>) {
  const nativeBridge = createMockBridge({ delayMs: 0, ...options });
  const view = render(
    <Harness nativeBridge={nativeBridge} {...(overrides ? { overrides } : {})} />,
  );
  return { ...view, nativeBridge };
}

/** Drops one audio file onto the card (the whole card is the drop target). */
function dropAudio(container: HTMLElement, fileName = "track.mp3") {
  fireEvent.drop(container.querySelector(".chain-audio-section")!, {
    dataTransfer: { files: [new File([], fileName)] },
  });
}

function noteText(container: HTMLElement): string {
  return container.querySelector(".chain-audio-note")!.textContent ?? "";
}

describe("ChainAudioPanel (§1-16 長尺A2V)", () => {
  describe("the three attach states", () => {
    it("renders the unattached state: choose button, 'no audio' note, 🔁 disabled, no thumbnail", () => {
      const { container } = setup();

      expect(screen.getByText(t.heading)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: t.chooseButton })).toBeEnabled();
      expect(screen.getByRole("button", { name: t.clearButton })).toBeDisabled();
      expect(noteText(container)).toBe(t.none);
      expect(container.querySelector(".keyframe-thumb img")).toBeNull();
      // The auto-fit button needs a measured track; there isn't one yet.
      expect(screen.getByRole("button", { name: t.adjustButton })).toBeDisabled();
    });

    it("shows the spinner and the 'Uploading…' label while the upload is in flight", async () => {
      const { container, nativeBridge } = setup({ holdUploads: true });

      dropAudio(container);

      await waitFor(() => expect(screen.getByRole("button", { name: t.uploadingButton })).toBeDisabled());
      expect(screen.getByRole("status")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: t.clearButton })).toBeDisabled();
      expect(screen.getByRole("button", { name: t.adjustButton })).toBeDisabled();

      (nativeBridge as ReturnType<typeof createMockBridge>).releaseUploads();
      await waitFor(() => expect(screen.getByRole("button", { name: t.changeButton })).toBeInTheDocument());
    });

    it("shows the ♫ thumbnail, the file name and the measured length once ready", async () => {
      const { container } = setup({ probeMediaInfoDurationSec: 20.72 });

      dropAudio(container);

      await waitFor(() => expect(screen.getByAltText(t.audioPlaceholderAlt)).toBeInTheDocument());
      expect(screen.getByText("track.mp3")).toBeInTheDocument();
      // `formatSecondsLabel` — one decimal, same readout Create's A2V card uses.
      expect(await screen.findByText("20.7s")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: t.clearButton })).toBeEnabled();
      expect(screen.getByRole("button", { name: t.changeButton })).toBeInTheDocument();
    });

    it("📁 attaches whatever the native dialog returned", async () => {
      const user = userEvent.setup();
      setup({ pickFileName: "voice.m4a", probeMediaInfoDurationSec: 12 });

      await user.click(screen.getByRole("button", { name: t.chooseButton }));

      await waitFor(() => expect(screen.getByText("voice.m4a")).toBeInTheDocument());
      expect(screen.getByAltText(t.audioPlaceholderAlt)).toBeInTheDocument();
    });

    it("🔁 detaches the track and returns the card to the unattached state", async () => {
      const user = userEvent.setup();
      const { container } = setup({ probeMediaInfoDurationSec: 20.72 });

      dropAudio(container);
      await waitFor(() => expect(screen.getByRole("button", { name: t.clearButton })).toBeEnabled());

      await user.click(screen.getByRole("button", { name: t.clearButton }));

      expect(noteText(container)).toBe(t.none);
      expect(screen.getByRole("button", { name: t.clearButton })).toBeDisabled();
      expect(container.querySelector(".keyframe-thumb img")).toBeNull();
    });

    it("rejects a non-audio drop inline and leaves the slot empty", async () => {
      const { container } = setup();

      dropAudio(container, "notes.txt");

      await waitFor(() =>
        expect(container.querySelector(".field-hint-error")?.textContent).toBe(en.dnd.unsupportedAudio),
      );
      expect(screen.getByRole("button", { name: t.clearButton })).toBeDisabled();
    });
  });

  describe("↔️ the auto-fit button", () => {
    it("stays disabled while the whole form is disabled, even with a measured track", async () => {
      const nativeBridge = createMockBridge({ delayMs: 0, probeMediaInfoDurationSec: 20.72 });
      const { container, rerender } = render(<Harness nativeBridge={nativeBridge} />);

      dropAudio(container);
      await waitFor(() => expect(screen.getByRole("button", { name: t.adjustButton })).toBeEnabled());

      rerender(<Harness nativeBridge={nativeBridge} disabled />);
      expect(screen.getByRole("button", { name: t.adjustButton })).toBeDisabled();
    });

    it("stays disabled when the length could not be measured, and says why", async () => {
      // No `probeMediaInfoDurationSec` -> the mock answers 0 ("unknown"), and the
      // file isn't a .wav either, so all three tiers fail.
      const { container } = setup();

      dropAudio(container);

      await waitFor(() => expect(noteText(container)).toBe(t.probeFailed));
      expect(screen.getByRole("button", { name: t.adjustButton })).toBeDisabled();
    });

    it("calls adjustClipsForAudio when pressed", async () => {
      const user = userEvent.setup();
      const adjustClipsForAudio = vi.fn();
      const { container } = setup({ probeMediaInfoDurationSec: 20.72 }, { adjustClipsForAudio });

      dropAudio(container);
      await waitFor(() => expect(screen.getByRole("button", { name: t.adjustButton })).toBeEnabled());

      await user.click(screen.getByRole("button", { name: t.adjustButton }));

      expect(adjustClipsForAudio).toHaveBeenCalledTimes(1);
    });
  });

  describe("the 'added clip length' slider", () => {
    it("opens on the 257-frame default, over the whole 9..481 clip range on the 8n+1 grid", () => {
      setup();

      const slider = screen.getByRole("slider");
      expect(slider).toHaveValue("257");
      expect(slider).toHaveAttribute("min", "9");
      expect(slider).toHaveAttribute("max", "481");
      expect(slider).toHaveAttribute("step", "8");
      expect(screen.getByText("257")).toBeInTheDocument();
    });

    it("moving it updates the form value and the readout", () => {
      setup();

      fireEvent.change(screen.getByRole("slider"), { target: { value: "121" } });

      expect(screen.getByRole("slider")).toHaveValue("121");
      expect(screen.getByText("121")).toBeInTheDocument();
    });
  });

  // One always-mounted line, one message at a time — see the panel's own doc
  // comment. Each case below pins BOTH the message and that it won.
  describe("the note line's priority order", () => {
    it("1. an over-long attach is a red error naming the ceiling in seconds", () => {
      // 24fps / seam width 3 -> maxChainAudioLatents = 11618 -> 464s.
      const { container } = setup({}, { audioAttachError: "tooLong" });

      expect(noteText(container)).toBe(t.tooLongError(464));
      expect(container.querySelector(".chain-audio-note")).toHaveClass("field-hint-error");
    });

    it("2. V2V + A2V beats every other note", () => {
      const { container } = setup(
        {},
        {
          audioAttachError: null,
          hasSourceVideo: true,
          hasSourceAudio: true,
          audioProbeFailed: true,
          audioAtMaxClips: true,
          audioSurplusSec: 9,
        },
      );

      expect(noteText(container)).toBe(t.conflictsWithSourceVideo);
      expect(container.querySelector(".chain-audio-note")).toHaveClass("chain-audio-note-warning");
    });

    it("3. an unmeasurable track beats the leftover-tail notes", () => {
      const { container } = setup(
        {},
        { hasSourceAudio: true, audioProbeFailed: true, audioAtMaxClips: true, audioSurplusSec: 9 },
      );

      expect(noteText(container)).toBe(t.probeFailed);
    });

    it("4. the 24-clip ceiling names the leftover and how to shrink it", () => {
      const { container } = setup({}, { hasSourceAudio: true, audioAtMaxClips: true, audioSurplusSec: 9.25 });

      expect(noteText(container)).toBe(t.atMaxClipsNote("9.3"));
      expect(container.querySelector(".chain-audio-note")).toHaveClass("chain-audio-note-warning");
    });

    it("5. a plain leftover tail is a note, not a warning", () => {
      const { container } = setup({}, { hasSourceAudio: true, audioSurplusSec: 1.5 });

      expect(noteText(container)).toBe(t.surplusNote("1.5"));
      expect(container.querySelector(".chain-audio-note")).not.toHaveClass("chain-audio-note-warning");
    });

    it("stays quiet for the one audio-latent frame every plan leaves behind by design", () => {
      const { container } = setup({}, { hasSourceAudio: true, audioSurplusSec: 0.04 });

      expect(noteText(container).trim()).toBe("");
    });

    it("is mounted in every state, so the card never reflows", () => {
      const { container } = setup({}, { hasSourceAudio: true, audioSurplusSec: 0.04 });
      expect(container.querySelectorAll(".chain-audio-note")).toHaveLength(1);

      const attached = setup({}, { audioAttachError: "tooLong" });
      expect(attached.container.querySelectorAll(".chain-audio-note")).toHaveLength(1);
    });
  });
});
