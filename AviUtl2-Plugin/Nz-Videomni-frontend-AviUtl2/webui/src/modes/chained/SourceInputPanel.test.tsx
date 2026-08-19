import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { NativeBridge } from "../../bridge";
import { createMockBridge } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import { SourceInputPanel } from "./SourceInputPanel";
import { useChainForm } from "./useChainForm";

// `nativeBridge` is created once in each test body and threaded in as a prop
// (rather than inside `Harness`) so its identity — and the mock's internal
// state (upload gate, fs, etc.) — survives every re-render `useChainForm`
// triggers, instead of being rebuilt from scratch each time.
function Harness({ nativeBridge }: { nativeBridge: NativeBridge }) {
  const form = useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge });
  return (
    <LanguageProvider>
      {/* `nativeBridge` is threaded into the panel too (not just the form
          hook) so the drag-and-drop tests' `ui.resolveDroppedFiles` round
          trip goes through the SAME mock instance as everything else,
          instead of silently falling back to the (unrelated) default bridge
          singleton `useFileDrop` would otherwise pick up. */}
      <SourceInputPanel
        form={form}
        disabled={false}
        nativeBridge={nativeBridge}
        mediaSize={form.sourceMediaSize}
      />
    </LanguageProvider>
  );
}

describe("SourceInputPanel", () => {
  it("renders the unselected state: choose button, 'no source' hint, Clear disabled", () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    render(<Harness nativeBridge={mockBridge} />);

    // Owner redesign (2026-07-18): choose/change and clear are now emoji
    // icon buttons (🔁/📁) with the same choose/change/clear text carried
    // over as their `aria-label`/`title` — role-name lookups stay unchanged.
    expect(screen.getByRole("button", { name: /choose image or video/i })).toBeInTheDocument();
    expect(screen.getByText(/no source selected/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^clear$/i })).toBeDisabled();
    // Nothing is attached yet, so the controls area defaults to the STRENGTH
    // slider (disabled) rather than CONTEXT FRAMES — same panel footprint as
    // once an image IS attached, so the very first pick never reflows it.
    expect(screen.getByText(/^strength$/i)).toBeInTheDocument();
    expect(screen.queryByText(/context frames/i)).not.toBeInTheDocument();
  });

  it("picking an image shows the STRENGTH slider (same panel footprint as video mode) and enables Clear", async () => {
    const user = userEvent.setup();
    const mockBridge = createMockBridge({ delayMs: 0, pickFileName: "photo.png" });
    render(<Harness nativeBridge={mockBridge} />);

    await user.click(screen.getByRole("button", { name: /choose image or video/i }));

    await waitFor(() => expect(screen.getByText(/^strength$/i)).toBeInTheDocument());
    expect(screen.queryByText(/context frames/i)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^clear$/i })).toBeEnabled();
    // The button now reads "Change…", not "Choose image or video…".
    expect(screen.getByRole("button", { name: /^change/i })).toBeInTheDocument();
  });

  it("picking a video shows the CONTEXT FRAMES slider instead of STRENGTH, and enables Clear", async () => {
    const user = userEvent.setup();
    const mockBridge = createMockBridge({ delayMs: 0, pickFileName: "clip.mp4" });
    render(<Harness nativeBridge={mockBridge} />);

    await user.click(screen.getByRole("button", { name: /choose image or video/i }));

    await waitFor(() => expect(screen.getByText(/context frames/i)).toBeInTheDocument());
    expect(screen.queryByText(/^strength$/i)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^clear$/i })).toBeEnabled();
  });

  it("Clear returns to the unselected state from either mode", async () => {
    const user = userEvent.setup();
    const mockBridge = createMockBridge({ delayMs: 0, pickFileName: "clip.mp4" });
    render(<Harness nativeBridge={mockBridge} />);

    await user.click(screen.getByRole("button", { name: /choose image or video/i }));
    await waitFor(() => expect(screen.getByText(/context frames/i)).toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: /^clear$/i }));

    expect(screen.getByText(/no source selected/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^clear$/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /choose image or video/i })).toBeInTheDocument();
  });

  it("an unrecognised extension surfaces the unsupported-file-type warning", async () => {
    const user = userEvent.setup();
    const mockBridge = createMockBridge({ delayMs: 0, pickFileName: "notes.txt" });
    render(<Harness nativeBridge={mockBridge} />);

    await user.click(screen.getByRole("button", { name: /choose image or video/i }));

    await waitFor(() => expect(screen.getByText(/unsupported file type/i)).toBeInTheDocument());
    expect(screen.getByRole("button", { name: /^clear$/i })).toBeDisabled();
  });

  describe("structural symmetry between image mode / video mode / unselected (post-review M-1/m-4 fix)", () => {
    it("keeps exactly one `.source-input-note` line mounted across unselected -> video -> cleared, never appearing/disappearing", async () => {
      const user = userEvent.setup();
      const mockBridge = createMockBridge({ delayMs: 0, pickFileName: "clip.mp4" });
      const { container } = render(<Harness nativeBridge={mockBridge} />);

      expect(container.querySelectorAll(".source-input-note")).toHaveLength(1);

      await user.click(screen.getByRole("button", { name: /choose image or video/i }));
      await waitFor(() => expect(screen.getByText(/context frames/i)).toBeInTheDocument());
      expect(container.querySelectorAll(".source-input-note")).toHaveLength(1);

      await user.click(screen.getByRole("button", { name: /^clear$/i }));
      expect(container.querySelectorAll(".source-input-note")).toHaveLength(1);
    });

    it("video mode's inline slider row shows only a short numeric readout — the descriptive sentence lives only in the separate note line", async () => {
      const user = userEvent.setup();
      const mockBridge = createMockBridge({ delayMs: 0, pickFileName: "clip.mp4" });
      const { container } = render(<Harness nativeBridge={mockBridge} />);

      await user.click(screen.getByRole("button", { name: /choose image or video/i }));
      await waitFor(() => expect(screen.getByText(/context frames/i)).toBeInTheDocument());

      const inlineRow = container.querySelector(".source-input-controls .field-inline");
      expect(inlineRow).not.toBeNull();
      expect(inlineRow?.textContent).not.toMatch(/carried over/i);

      const note = container.querySelector(".source-input-note");
      expect(note).not.toBeNull();
      expect(note?.textContent).toMatch(/carried over/i);
    });

    it("image mode's inline slider row never contains the video's descriptive sentence either", async () => {
      const user = userEvent.setup();
      const mockBridge = createMockBridge({ delayMs: 0, pickFileName: "photo.png" });
      const { container } = render(<Harness nativeBridge={mockBridge} />);

      await user.click(screen.getByRole("button", { name: /choose image or video/i }));
      await waitFor(() => expect(screen.getByText(/^strength$/i)).toBeInTheDocument());

      const inlineRow = container.querySelector(".source-input-controls .field-inline");
      expect(inlineRow?.textContent).not.toMatch(/carried over/i);
    });
  });

  describe("m-3: the STRENGTH slider is disabled once the image card errors out", () => {
    it("disables the slider when the start-frame upload fails", async () => {
      const user = userEvent.setup();
      const mockBridge = createMockBridge({ delayMs: 0, pickFileName: "photo.png", failUploadFile: "BACKEND_UNREACHABLE" });
      render(<Harness nativeBridge={mockBridge} />);

      await user.click(screen.getByRole("button", { name: /choose image or video/i }));

      await waitFor(() => expect(screen.getByRole("slider")).toBeDisabled());
    });
  });

  // Owner request 2026-08-10: the attached material's pixel size is appended to
  // the file-name line, so the width/height sliders can be matched against it.
  // The size half is dropped entirely whenever it isn't known (Retake's
  // `sourceReadout` precedent) — never "0 x 0", never "unknown".
  describe("resolution readout on the file-name line", () => {
    it("shows the probed size next to a picked VIDEO's file name", async () => {
      const user = userEvent.setup();
      const mockBridge = createMockBridge({
        delayMs: 0,
        pickFileName: "clip.mp4",
        probeMediaInfoWidth: 1280,
        probeMediaInfoHeight: 768,
      });
      render(<Harness nativeBridge={mockBridge} />);

      await user.click(screen.getByRole("button", { name: /choose image or video/i }));

      await waitFor(() => expect(screen.getByText(/clip\.mp4 — 1280 x 768/)).toBeInTheDocument());
    });

    it("shows the probed size next to a picked IMAGE's file name", async () => {
      const user = userEvent.setup();
      const mockBridge = createMockBridge({
        delayMs: 0,
        pickFileName: "photo.png",
        probeMediaInfoWidth: 1920,
        probeMediaInfoHeight: 1080,
      });
      render(<Harness nativeBridge={mockBridge} />);

      await user.click(screen.getByRole("button", { name: /choose image or video/i }));

      await waitFor(() => expect(screen.getByText(/photo\.png — 1920 x 1080/)).toBeInTheDocument());
    });

    it("leaves the bare file name when a dimension comes back unknown (0)", async () => {
      const user = userEvent.setup();
      const mockBridge = createMockBridge({
        delayMs: 0,
        pickFileName: "clip.mp4",
        probeMediaInfoWidth: 1280,
        probeMediaInfoHeight: 0,
      });
      render(<Harness nativeBridge={mockBridge} />);

      await user.click(screen.getByRole("button", { name: /choose image or video/i }));

      await waitFor(() => expect(screen.getByText("clip.mp4")).toBeInTheDocument());
      expect(screen.queryByText(/1280 x/)).not.toBeInTheDocument();
    });

    it("leaves the bare file name when the probe itself fails", async () => {
      const user = userEvent.setup();
      const inner = createMockBridge({
        delayMs: 0,
        pickFileName: "clip.mp4",
        probeMediaInfoWidth: 1280,
        probeMediaInfoHeight: 768,
      });
      // The mock's own `fs.probeMediaInfo` never rejects; native's DOES (e.g.
      // BAD_REQUEST on an unreadable path), so that path is injected here.
      const mockBridge: NativeBridge = {
        ...inner,
        request: ((method: string, params: unknown) =>
          method === "fs.probeMediaInfo"
            ? Promise.reject(new Error("Mock: unreadable path"))
            : (inner.request as (m: string, p: unknown) => Promise<unknown>)(method, params)) as NativeBridge["request"],
      };
      render(<Harness nativeBridge={mockBridge} />);

      await user.click(screen.getByRole("button", { name: /choose image or video/i }));

      await waitFor(() => expect(screen.getByText("clip.mp4")).toBeInTheDocument());
      expect(screen.queryByText(/1280 x 768/)).not.toBeInTheDocument();
    });

    it("drops the whole readout line again once the source is cleared", async () => {
      const user = userEvent.setup();
      const mockBridge = createMockBridge({
        delayMs: 0,
        pickFileName: "clip.mp4",
        probeMediaInfoWidth: 1280,
        probeMediaInfoHeight: 768,
      });
      render(<Harness nativeBridge={mockBridge} />);

      await user.click(screen.getByRole("button", { name: /choose image or video/i }));
      await waitFor(() => expect(screen.getByText(/clip\.mp4 — 1280 x 768/)).toBeInTheDocument());

      await user.click(screen.getByRole("button", { name: /^clear$/i }));

      expect(screen.queryByText(/clip\.mp4/)).not.toBeInTheDocument();
      expect(screen.queryByText(/1280 x 768/)).not.toBeInTheDocument();
    });
  });

  // Contract v7 (drag-and-drop): the same unified source input additionally
  // accepts a drop, routed through `useChainForm.attachSourceByPath`. Owner
  // redesign (2026-07-18): the drop target is now the whole `.source-section`
  // card itself (no wrapping `DropZone`), so every drop event below fires on
  // that root element rather than a `.drop-zone` child.
  describe("drag-and-drop (contract v7)", () => {
    it("dropping an image keeps scratch mode and shows the STRENGTH slider", async () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { container } = render(<Harness nativeBridge={mockBridge} />);
      const zone = container.querySelector(".source-section")!;

      fireEvent.drop(zone, { dataTransfer: { files: [new File([], "photo.png")] } });

      // Unlike the STRENGTH slider (also shown, disabled, before anything is
      // attached — see the unselected-state test above), Clear only enables
      // once the image is actually attached, so this is the real "it worked"
      // signal to wait on.
      await waitFor(() => expect(screen.getByRole("button", { name: /^clear$/i })).toBeEnabled());
      expect(screen.getByText(/^strength$/i)).toBeInTheDocument();
      expect(screen.queryByText(/context frames/i)).not.toBeInTheDocument();
    });

    it("dropping a video switches to v2v mode and shows the CONTEXT FRAMES slider", async () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { container } = render(<Harness nativeBridge={mockBridge} />);
      const zone = container.querySelector(".source-section")!;

      fireEvent.drop(zone, { dataTransfer: { files: [new File([], "clip.mp4")] } });

      await waitFor(() => expect(screen.getByText(/context frames/i)).toBeInTheDocument());
      expect(screen.queryByText(/^strength$/i)).not.toBeInTheDocument();
      expect(screen.getByRole("button", { name: /^clear$/i })).toBeEnabled();
    });

    it("dragging over the card highlights it with .source-section-dragover", () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { container } = render(<Harness nativeBridge={mockBridge} />);
      const zone = container.querySelector(".source-section")!;

      fireEvent.dragEnter(zone, { dataTransfer: { files: [] } });

      expect(zone).toHaveClass("source-section-dragover");

      fireEvent.dragLeave(zone, { dataTransfer: { files: [] } });

      expect(zone).not.toHaveClass("source-section-dragover");
    });

    it("dropping an unsupported extension shows the unsupported-source message inline and leaves the source untouched", async () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { container } = render(<Harness nativeBridge={mockBridge} />);
      const zone = container.querySelector(".source-section")!;

      fireEvent.drop(zone, { dataTransfer: { files: [new File([], "notes.txt")] } });

      // Scoped to `.field-hint-error` (rather than a bare `screen.findByText`)
      // because the always-rendered `.field-hint` note above it ("Choose an
      // image to start a new chain, or a video…") also happens to satisfy a
      // loose "choose an image ... or a video" match.
      await waitFor(() => {
        const errorLine = container.querySelector(".field-hint-error");
        expect(errorLine).not.toBeNull();
        expect(errorLine?.textContent).toMatch(/choose an image .* or a video/i);
      });
      expect(screen.getByText(/no source selected/i)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /^clear$/i })).toBeDisabled();
    });
  });
});
