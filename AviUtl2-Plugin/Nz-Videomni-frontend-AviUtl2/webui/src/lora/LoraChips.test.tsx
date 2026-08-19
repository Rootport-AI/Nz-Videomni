import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "../i18n/LanguageContext";
import { LoraChips } from "./LoraChips";

/** `LoraChips` calls `useStrings()` (M7b) — every render needs a
 * `LanguageProvider` ancestor. */
function renderWithLanguage(ui: ReactElement) {
  return render(<LanguageProvider>{ui}</LanguageProvider>);
}

describe("LoraChips", () => {
  it("renders nothing when the prompt has no LoRA tags", () => {
    const { container } = renderWithLanguage(<LoraChips prompt="a cat riding a skateboard" onChange={vi.fn()} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders a chip per valid tag, with name and strength", () => {
    renderWithLanguage(<LoraChips prompt="a cat <lora:Pixar_Toon:1.0> riding a skateboard" onChange={vi.fn()} />);
    expect(screen.getByText("Pixar_Toon")).toBeInTheDocument();
    expect(screen.getAllByText("1.00").length).toBeGreaterThan(0);
  });

  it("does not render a chip for an invalid-name tag", () => {
    renderWithLanguage(<LoraChips prompt="<lora:../bad:1.0> a cat" onChange={vi.fn()} />);
    expect(screen.queryByText("../bad")).not.toBeInTheDocument();
  });

  it("+ button increases strength by 0.1 and rewrites the prompt tag", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithLanguage(<LoraChips prompt="<lora:Pixar_Toon:1.0> a cat" onChange={onChange} />);

    await user.click(screen.getByRole("button", { name: /increase pixar_toon strength/i }));

    expect(onChange).toHaveBeenCalledWith("<lora:Pixar_Toon:1.1> a cat");
  });

  it("− button decreases strength by 0.1", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithLanguage(<LoraChips prompt="<lora:Pixar_Toon:1.0> a cat" onChange={onChange} />);

    await user.click(screen.getByRole("button", { name: /decrease pixar_toon strength/i }));

    expect(onChange).toHaveBeenCalledWith("<lora:Pixar_Toon:0.9> a cat");
  });

  it("× button removes the tag entirely", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithLanguage(<LoraChips prompt="a cat <lora:Pixar_Toon:1.0> riding a skateboard" onChange={onChange} />);

    await user.click(screen.getByRole("button", { name: /remove pixar_toon/i }));

    expect(onChange).toHaveBeenCalledWith("a cat riding a skateboard");
  });

  it("renders one chip per tag when multiple LoRAs are present", () => {
    renderWithLanguage(<LoraChips prompt="<lora:A:1.0> text <lora:B:0.5>" onChange={vi.fn()} />);
    expect(screen.getByText("A")).toBeInTheDocument();
    expect(screen.getByText("B")).toBeInTheDocument();
  });

  describe("hiddenNames (IC-LoRA UI redesign, 第5波)", () => {
    it("hides only the chip whose name is in hiddenNames, leaving other chips visible", () => {
      renderWithLanguage(
        <LoraChips
          prompt="<lora:canny-control:1.0> a cat <lora:Pixar_Toon:1.0> riding a skateboard"
          onChange={vi.fn()}
          hiddenNames={new Set(["canny-control"])}
        />,
      );
      expect(screen.queryByText("canny-control")).not.toBeInTheDocument();
      expect(screen.getByText("Pixar_Toon")).toBeInTheDocument();
    });

    it("renders nothing when every tag's name is hidden", () => {
      const { container } = renderWithLanguage(
        <LoraChips prompt="<lora:canny-control:1.0> a cat" onChange={vi.fn()} hiddenNames={new Set(["canny-control"])} />,
      );
      expect(container).toBeEmptyDOMElement();
    });

    it("does not touch the prompt itself — hiding is a display filter only", () => {
      const onChange = vi.fn();
      renderWithLanguage(
        <LoraChips
          prompt="<lora:canny-control:1.0> a cat"
          onChange={onChange}
          hiddenNames={new Set(["canny-control"])}
        />,
      );
      expect(onChange).not.toHaveBeenCalled();
    });

    it("renders every chip as usual when hiddenNames is omitted", () => {
      renderWithLanguage(<LoraChips prompt="<lora:canny-control:1.0> a cat" onChange={vi.fn()} />);
      expect(screen.getByText("canny-control")).toBeInTheDocument();
    });
  });

  describe("audio strength badge + mute toggle (Style LoRA音声強度制御, 2026-08-02; reordered 2026-08-02)", () => {
    it("shows the audio badge for a plain 2-argument tag too, with the effective (video-following) value", () => {
      const { container } = renderWithLanguage(<LoraChips prompt="<lora:Pixar_Toon:1.0> a cat" onChange={vi.fn()} />);
      const badge = container.querySelector(".lora-chip-audio");
      expect(badge).toBeInTheDocument();
      expect(badge).toHaveTextContent("1.00");
      expect(badge).not.toHaveClass("lora-chip-audio-muted");
    });

    it("shows the audio badge with the value for a 3-argument tag", () => {
      const { container } = renderWithLanguage(<LoraChips prompt="<lora:X:0.8:0.30> a cat" onChange={vi.fn()} />);
      // The mute toggle button renders the bare emoji with no value; only the
      // badge span carries the formatted number, with no emoji of its own.
      expect(container.querySelector(".lora-chip-audio")).toHaveTextContent("0.30");
    });

    it("shows the muted badge with 0.00 (still visible, not hidden) when audio_strength is 0", () => {
      const { container } = renderWithLanguage(<LoraChips prompt="<lora:X:0.8:0> a cat" onChange={vi.fn()} />);
      const badge = container.querySelector(".lora-chip-audio");
      expect(badge).toHaveTextContent("0.00");
      expect(badge).toHaveClass("lora-chip-audio-muted");
    });

    it("orders the chip's children: name, 🎥 video strength, −/+, mute toggle, audio badge, remove", () => {
      const { container } = renderWithLanguage(<LoraChips prompt="<lora:X:0.8:0.30> a cat" onChange={vi.fn()} />);
      const chip = container.querySelector(".lora-chip");
      const classNames = Array.from(chip?.children ?? []).map((el) => el.className);
      expect(classNames).toEqual([
        "lora-chip-name",
        "lora-chip-video-icon",
        "lora-chip-strength",
        "lora-chip-btn",
        "lora-chip-btn",
        "lora-chip-audio-toggle",
        "lora-chip-audio",
        "lora-chip-remove",
      ]);
    });

    it("the mute toggle is present even on a 2-argument tag, starting as audible (aria-pressed=false)", () => {
      renderWithLanguage(<LoraChips prompt="<lora:Pixar_Toon:1.0> a cat" onChange={vi.fn()} />);
      const toggle = screen.getByRole("button", { name: /mute pixar_toon audio/i });
      expect(toggle).toHaveAttribute("aria-pressed", "false");
    });

    it("clicking the toggle on an audible (undefined) tag sets audio_strength to 0", async () => {
      const user = userEvent.setup();
      const onChange = vi.fn();
      renderWithLanguage(<LoraChips prompt="<lora:X:0.8> a cat" onChange={onChange} />);

      await user.click(screen.getByRole("button", { name: /mute x audio/i }));

      expect(onChange).toHaveBeenCalledWith("<lora:X:0.8:0.0> a cat");
    });

    it("clicking the toggle on a muted (0) tag restores audio_strength to 1.0", async () => {
      const user = userEvent.setup();
      const onChange = vi.fn();
      renderWithLanguage(<LoraChips prompt="<lora:X:0.8:0> a cat" onChange={onChange} />);

      const toggle = screen.getByRole("button", { name: /unmute x audio/i });
      expect(toggle).toHaveAttribute("aria-pressed", "true");
      await user.click(toggle);

      expect(onChange).toHaveBeenCalledWith("<lora:X:0.8:1.0> a cat");
    });

    it("adjusting strength via +/- preserves an existing audio badge", async () => {
      const user = userEvent.setup();
      const onChange = vi.fn();
      renderWithLanguage(<LoraChips prompt="<lora:X:0.8:0.3> a cat" onChange={onChange} />);

      await user.click(screen.getByRole("button", { name: /increase x strength/i }));

      expect(onChange).toHaveBeenCalledWith("<lora:X:0.9:0.3> a cat");
    });
  });
});
