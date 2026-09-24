import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { createMockBridge } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { OBJECT_TRACKING_DEFAULTS } from "../../shell/objectTrackingSettings";
import { ToolboxScreen } from "./ToolboxScreen";

// §3-164 (2026-09-24): the Toolbox tab's two sub-tabs, "Tracking" and
// "mp4 info". Both panels stay mounted and are toggled with `hidden`, the same
// arrangement as Edit's sub-panels; neither carries role="tabpanel".

function renderScreen() {
  const nativeBridge = createMockBridge({ delayMs: 0 });
  const noop = vi.fn();
  return render(
    <LanguageProvider>
      <ToolboxScreen
        trackingAvailable
        nativeBridge={nativeBridge}
        settings={OBJECT_TRACKING_DEFAULTS}
        onSearchFactorChange={noop}
        onLostScoreThresholdChange={noop}
        onLostBehaviorChange={noop}
        onSmoothingChange={noop}
        onFollowSizeChange={noop}
        onKeyframeStrideChange={noop}
        onStartTracking={noop}
      />
    </LanguageProvider>,
  );
}

/** The `.toolbox-subpanel` ancestor of an element — the node that carries
 * `hidden`. */
function subpanelOf(el: HTMLElement): HTMLElement {
  const panel = el.closest(".toolbox-subpanel");
  if (!(panel instanceof HTMLElement)) throw new Error("not inside a toolbox sub-panel");
  return panel;
}

describe("ToolboxSubTabs", () => {
  it("renders Tracking and mp4 info, with Tracking selected by default", () => {
    renderScreen();
    const tracking = screen.getByRole("tab", { name: en.toolbox.subTabs.tracking });
    const mp4info = screen.getByRole("tab", { name: en.toolbox.subTabs.mp4info });
    expect(tracking).toHaveAttribute("aria-selected", "true");
    expect(mp4info).toHaveAttribute("aria-selected", "false");

    expect(subpanelOf(screen.getByText(en.toolbox.tracking.heading))).not.toHaveAttribute("hidden");
    expect(subpanelOf(screen.getByText(en.toolbox.mp4info.dropHint))).toHaveAttribute("hidden");
  });

  it("switching to mp4 info hides the tracking panel without unmounting it, and back again", async () => {
    renderScreen();
    const user = userEvent.setup();
    const trackingHeading = screen.getByText(en.toolbox.tracking.heading);

    await user.click(screen.getByRole("tab", { name: en.toolbox.subTabs.mp4info }));

    expect(screen.getByRole("tab", { name: en.toolbox.subTabs.mp4info })).toHaveAttribute("aria-selected", "true");
    expect(subpanelOf(trackingHeading)).toHaveAttribute("hidden");
    expect(subpanelOf(screen.getByText(en.toolbox.mp4info.dropHint))).not.toHaveAttribute("hidden");

    await user.click(screen.getByRole("tab", { name: en.toolbox.subTabs.tracking }));

    // Same DOM node: the panel was hidden, never unmounted.
    expect(screen.getByText(en.toolbox.tracking.heading)).toBe(trackingHeading);
    expect(subpanelOf(trackingHeading)).not.toHaveAttribute("hidden");
  });

  it('puts no role="tabpanel" on the sub-panels (app tests read the ONE visible mode panel by that role)', () => {
    renderScreen();
    expect(screen.queryAllByRole("tabpanel", { hidden: true })).toHaveLength(0);
  });
});
