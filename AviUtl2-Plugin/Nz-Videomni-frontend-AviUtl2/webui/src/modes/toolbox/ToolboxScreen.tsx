import type { NativeBridge } from "../../bridge";
import type { LostBehavior, ObjectTrackingSettings } from "../../shell/objectTrackingSettings";
import { ObjectTrackingSection } from "./ObjectTrackingSection";
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
 * own controls, its own run and its own readout. So this screen holds no state
 * at all; it passes `AppShell`'s props through to the one section below.
 *
 * That also settles where a SECOND tool would go: another `<section>` here,
 * with its own hook, needing nothing from this file but a row of props. The
 * tab is the container; the sections are the tools.
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
}: ToolboxScreenProps) {
  return (
    <div className="toolbox-screen">
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
      />
    </div>
  );
}
