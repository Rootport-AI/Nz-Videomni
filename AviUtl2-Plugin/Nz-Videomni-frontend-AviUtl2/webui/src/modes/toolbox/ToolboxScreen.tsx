import type { NativeBridge } from "../../bridge";
import type { LostBehavior, ObjectTrackingSettings } from "../../shell/objectTrackingSettings";
import type { TimelineSelection } from "../../timeline/menuSelection";
import { useState } from "react";
import { ObjectTrackingSection } from "./ObjectTrackingSection";
import { Mp4InfoSection } from "./Mp4InfoSection";
import { ToolboxSubTabs } from "./ToolboxSubTabs";
import type { ToolboxSubMode } from "./ToolboxSubTabs";
import type { ObjectTrackRequest } from "./useObjectTracking";
import "./ToolboxScreen.css";

/**
 * The Toolbox mode screen (§3-54, 2026-09-11). Until this day the Toolbox tab
 * was a disabled mock in `shell/ModeTabs.tsx` — the last one left after Edit
 * was promoted on 2026-08-09 — so this is the same promotion, done the same
 * way: an `AppMode` value, a remount token, a `MenuTargetMode` entry, and a
 * panel that is always mounted and merely hidden when another mode is showing
 * (see `AppShell`'s `.app-mode-body`).
 *
 * It is deliberately a THIN container. Unlike `EditScreen` — which owns its
 * panels' form state because the Generate button lives in a separate column and
 * needs it — a Toolbox tool has no shared column to reach across: it owns its
 * own controls, its own run and its own readout. The only state this screen
 * holds is WHICH tool is showing.
 *
 * Since §3-164 (2026-09-24) there are two tools, and they sit under sub-tabs
 * (`ToolboxSubTabs`, the same shape as `EditSubTabs`): "Tracking"
 * (`ObjectTrackingSection`, unchanged) and "mp4 info" (`Mp4InfoSection`). Both
 * panels stay mounted and are hidden with the plain `hidden` attribute, like
 * Edit's sub-panels, so switching tabs never loses a tool's state or cuts off
 * a running tracking run. They carry NO `role="tabpanel"` — see
 * `ToolboxSubTabs`'s doc comment. A routed 追尾 remounts this screen
 * (`remountTokens.toolbox`), which also brings the selection back to its
 * default, "tracking" — the tab that run needs.
 *
 * A further tool is one more sub-tab id and one more hidden panel here, with
 * its own section and hook.
 */
export interface ToolboxScreenProps {
  /** From `AppShell`'s `/status` poll: whether the server can run tracking at
   * all. */
  trackingAvailable: boolean;
  /** `/status.tracking.reason` when it cannot. */
  trackingReason?: string | undefined;
  /** The routed 追尾 request. Delivered with a bumped `remountTokens.toolbox`,
   * so a fresh mount is what starts a run (see `useObjectTracking`). */
  trackRequest?: ObjectTrackRequest | undefined;
  nativeBridge?: NativeBridge | undefined;
  settings: ObjectTrackingSettings;
  onSearchFactorChange: (value: number) => void;
  onLostScoreThresholdChange: (value: number) => void;
  onLostBehaviorChange: (value: LostBehavior) => void;
  onSmoothingChange: (value: number) => void;
  onFollowSizeChange: (value: boolean) => void;
  onKeyframeStrideChange: (value: number) => void;
  /** Passed straight through to the section: it raises this while its run is
   * going so `AppShell` can refuse a SECOND 追尾 right-click (or button press)
   * instead of remounting this screen out from under the run (owner gate
   * 2026-09-11). */
  onRunningChange?: ((running: boolean) => void) | undefined;
  /** `AppShell`'s `startTrackingFromSelection` — the one road a 追尾 run starts
   * down, shared with the timeline right-click (2026-09-16). Passed straight
   * through: this screen decides nothing about it, and the section below calls
   * it only after it has read the selection and found a single 部分フィルタ. */
  onStartTracking: (selection: TimelineSelection) => void;
}

export function ToolboxScreen({
  trackingAvailable,
  trackingReason,
  trackRequest,
  nativeBridge,
  settings,
  onSearchFactorChange,
  onLostScoreThresholdChange,
  onLostBehaviorChange,
  onSmoothingChange,
  onFollowSizeChange,
  onKeyframeStrideChange,
  onRunningChange,
  onStartTracking,
}: ToolboxScreenProps) {
  const [subMode, setSubMode] = useState<ToolboxSubMode>("tracking");
  return (
    <div className="toolbox-screen">
      <ToolboxSubTabs mode={subMode} onChange={setSubMode} />
      <div className="toolbox-subpanel" hidden={subMode !== "tracking"}>
        <ObjectTrackingSection
          trackingAvailable={trackingAvailable}
          trackingReason={trackingReason}
          trackRequest={trackRequest}
          nativeBridge={nativeBridge}
          settings={settings}
          onSearchFactorChange={onSearchFactorChange}
          onLostScoreThresholdChange={onLostScoreThresholdChange}
          onLostBehaviorChange={onLostBehaviorChange}
          onSmoothingChange={onSmoothingChange}
          onFollowSizeChange={onFollowSizeChange}
          onKeyframeStrideChange={onKeyframeStrideChange}
          onRunningChange={onRunningChange}
          onStartTracking={onStartTracking}
        />
      </div>
      <div className="toolbox-subpanel" hidden={subMode !== "mp4info"}>
        <Mp4InfoSection nativeBridge={nativeBridge} />
      </div>
    </div>
  );
}
