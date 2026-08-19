import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { AppConfig, GenerationPreset } from "../../api/types";
import { PresetDropdown, sortPresets } from "./PresetDropdown";

const ORIENTATION_LABELS = { landscape: "Landscape", square: "Square", portrait: "Portrait" };

function preset(width: number, height: number, num_frames = 1): GenerationPreset {
  return { width, height, crop_output: null, num_frames };
}

function makeConfig(presets: Record<string, GenerationPreset>): AppConfig {
  return {
    generation_presets: presets,
  } as unknown as AppConfig;
}

describe("sortPresets", () => {
  it("sorts the current (all-landscape) config presets by ascending area", () => {
    // Same 6 entries as `defaultConfig.ts`'s FALLBACK_APP_CONFIG, given in a
    // shuffled order — all landscape (w>h), so orientation never breaks the tie.
    const entries: [string, GenerationPreset][] = [
      ["WQHD_1440p", preset(2560, 1472)],
      ["standard_720p", preset(1280, 768)],
      ["smoke_test", preset(384, 256)],
      ["FHD_1080p", preset(1920, 1088)],
      ["small", preset(960, 576)],
      ["minimal", preset(512, 320)],
    ];

    const sorted = sortPresets(entries).map(([name]) => name);

    expect(sorted).toEqual(["smoke_test", "minimal", "small", "standard_720p", "FHD_1080p", "WQHD_1440p"]);
  });

  it("groups landscape before square before portrait, area-ascending within each group", () => {
    const entries: [string, GenerationPreset][] = [
      ["tall", preset(576, 960)],
      ["wide_big", preset(1280, 768)],
      ["squarish", preset(512, 512)],
      ["wide_small", preset(384, 256)],
    ];

    const sorted = sortPresets(entries).map(([name]) => name);

    expect(sorted).toEqual(["wide_small", "wide_big", "squarish", "tall"]);
  });

  it("breaks ties on equal orientation and area by name (localeCompare)", () => {
    const entries: [string, GenerationPreset][] = [
      ["beta", preset(512, 320)],
      ["alpha", preset(320, 512)],
      ["gamma", preset(320, 200)],
    ];

    // "beta"/"gamma" are both landscape with different areas (gamma smaller);
    // "alpha" is portrait. Add a genuine tie to exercise the name key.
    const tied: [string, GenerationPreset][] = [
      ["zeta", preset(400, 200)],
      ["alpha2", preset(400, 200)],
    ];

    expect(sortPresets(entries).map(([name]) => name)).toEqual(["gamma", "beta", "alpha"]);
    expect(sortPresets(tied).map(([name]) => name)).toEqual(["alpha2", "zeta"]);
  });
});

describe("PresetDropdown rendering", () => {
  it("renders a flat option list (no optgroup) when only one orientation is present", () => {
    const config = makeConfig({
      smoke_test: preset(384, 256),
      minimal: preset(512, 320),
    });

    render(
      <PresetDropdown
        config={config}
        onSelect={vi.fn()}
        disabled={false}
        label="Presets"
        placeholder="Choose a preset…"
        orientationLabels={ORIENTATION_LABELS}
      />,
    );

    expect(screen.queryByRole("group")).not.toBeInTheDocument();
    expect(screen.getByRole("option", { name: /smoke_test/ })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /minimal/ })).toBeInTheDocument();
  });

  it("groups into optgroups with the given labels when orientations are mixed", () => {
    const config = makeConfig({
      wide: preset(1280, 768),
      squarish: preset(512, 512),
      tall: preset(576, 960),
    });

    render(
      <PresetDropdown
        config={config}
        onSelect={vi.fn()}
        disabled={false}
        label="Presets"
        placeholder="Choose a preset…"
        orientationLabels={ORIENTATION_LABELS}
      />,
    );

    const groups = screen.getAllByRole("group");
    expect(groups).toHaveLength(3);
    expect(groups.map((g) => g.getAttribute("label"))).toEqual(["Landscape", "Square", "Portrait"]);
  });

  it("renders nothing when there are no presets", () => {
    const config = makeConfig({});

    const { container } = render(
      <PresetDropdown
        config={config}
        onSelect={vi.fn()}
        disabled={false}
        label="Presets"
        placeholder="Choose a preset…"
        orientationLabels={ORIENTATION_LABELS}
      />,
    );

    expect(container).toBeEmptyDOMElement();
  });
});

describe("PresetDropdown auto-apply on selection", () => {
  it("selecting a different preset fires onSelect via onChange, and routing through a third preset and back (A→B→A) re-applies the original", () => {
    const onSelect = vi.fn();
    const config = makeConfig({ smoke_test: preset(384, 256), minimal: preset(512, 320) });
    render(
      <PresetDropdown
        config={config}
        onSelect={onSelect}
        disabled={false}
        label="Presets"
        placeholder="Choose a preset…"
        orientationLabels={ORIENTATION_LABELS}
      />,
    );

    const select = screen.getByRole("combobox");

    fireEvent.change(select, { target: { value: "minimal" } });
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenLastCalledWith("minimal");

    // A native <select> change event doesn't fire again for re-selecting the
    // SAME already-selected option — but there is no dedicated control for
    // that case anymore (the Apply button was removed 2026-07-18). Instead,
    // going via a different preset first (A→B→A) fires onChange → onSelect
    // twice more, and since `applyPreset` unconditionally overwrites its
    // governed fields on every call, landing back on "minimal" fully
    // restores it regardless of what "smoke_test" changed in between.
    fireEvent.change(select, { target: { value: "smoke_test" } });
    expect(onSelect).toHaveBeenCalledTimes(2);
    expect(onSelect).toHaveBeenLastCalledWith("smoke_test");

    fireEvent.change(select, { target: { value: "minimal" } });
    expect(onSelect).toHaveBeenCalledTimes(3);
    expect(onSelect).toHaveBeenLastCalledWith("minimal");
  });

  it("does not fire onSelect when the placeholder (empty value) is selected", () => {
    const onSelect = vi.fn();
    const config = makeConfig({ smoke_test: preset(384, 256) });
    render(
      <PresetDropdown
        config={config}
        onSelect={onSelect}
        disabled={false}
        label="Presets"
        placeholder="Choose a preset…"
        orientationLabels={ORIENTATION_LABELS}
      />,
    );

    fireEvent.change(screen.getByRole("combobox"), { target: { value: "" } });
    expect(onSelect).not.toHaveBeenCalled();
  });
});
