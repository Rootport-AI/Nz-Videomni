import { useStrings } from "../../i18n/LanguageContext";
import type { NativeBridge } from "../../bridge";
import {
  KEYFRAME_STRIDE_MAX,
  KEYFRAME_STRIDE_MIN,
  KEYFRAME_STRIDE_STEP,
  LOST_THRESHOLD_MAX,
  LOST_THRESHOLD_MIN,
  LOST_THRESHOLD_STEP,
  SEARCH_FACTOR_MAX,
  SEARCH_FACTOR_MIN,
  SEARCH_FACTOR_STEP,
  SMOOTHING_MAX,
  SMOOTHING_MIN,
  SMOOTHING_STEP,
} from "../../shell/objectTrackingSettings";
import type { LostBehavior, ObjectTrackingSettings } from "../../shell/objectTrackingSettings";
import { useObjectTracking } from "./useObjectTracking";
import type { ObjectTrackRequest } from "./useObjectTracking";

/**
 * 物体追尾 / Object tracking — the Toolbox tab's first real occupant (§3-54,
 * 2026-09-11; design doc `Nz-Videomni/Docs/OBJECT_TRACKING_DESIGN.md` §3).
 *
 * ## What this panel is NOT
 *
 * It draws no rectangle and shows no video. The box comes from AviUtl2's own
 * preview: the user puts a 部分フィルタ on a layer under the video, fits its box
 * there, and right-clicks the object (design doc §11). This panel owns the
 * SETTINGS for the run and the READOUT of it — nothing about the geometry.
 *
 * ## Tracking does not share the generation lock
 *
 * Nothing here reads `serverBusy` or the reservation seat, and neither of those
 * reads this. Tracking runs on the CPU in a separate worker, so it never waits
 * for a generation to finish and never stops the user from starting one
 * (design doc §18). Every other long operation in this app is GPU work under a
 * one-at-a-time discipline; this one is the deliberate exception, so the panel
 * says so in as many words rather than leaving the user to wonder.
 *
 * ## Parts are copies, on purpose
 *
 * The slider+number pair, the radio group and the checkbox are local copies of
 * the shapes `modes/edit/OutpaintingPanel.tsx`, `modes/edit/RetakePanel.tsx`
 * and `modes/chained/ChainedScreen.tsx` use, styled with the global `.field*`
 * classes (`modes/single/SingleScreen.css`). That duplication is this
 * codebase's standing arrangement for mode-local form parts — see those files —
 * and it is what keeps a Toolbox tweak from rippling into the generation forms.
 *
 * ## Where failures are shown
 *
 * In this section, not in the shared note area. `NoteArea` is the right-click
 * flow's channel — a one-seat slot above the tabs that the next route replaces
 * — whereas a tracking failure arrives minutes after the click, quite possibly
 * while the user is looking at another tab. It belongs next to the run it
 * describes.
 */
export interface ObjectTrackingSectionProps {
  /** Whether the server can track at all, derived by `AppShell` from
   * `/status.tracking.available`. False greys every control and shows the
   * install guidance. */
  trackingAvailable: boolean;
  /** `/status.tracking.reason` when unavailable — `"not installed"` or
   * `"worker failed"`. Anything else (including absent) falls back to the
   * not-installed guidance, which is the overwhelmingly common case. */
  trackingReason?: string | undefined;
  /** The routed request, delivered alongside a bumped `remountTokens.toolbox`;
   * `undefined` when the tab was opened by hand. */
  trackRequest?: ObjectTrackRequest | undefined;
  nativeBridge?: NativeBridge | undefined;
  /** The persisted settings and their setters, owned by `AppShell`
   * (`shell/useObjectTrackingSettings.ts`). */
  settings: ObjectTrackingSettings;
  onSearchFactorChange: (value: number) => void;
  onLostScoreThresholdChange: (value: number) => void;
  onLostBehaviorChange: (value: LostBehavior) => void;
  onSmoothingChange: (value: number) => void;
  onFollowSizeChange: (value: boolean) => void;
  onKeyframeStrideChange: (value: number) => void;
  /** Test seam for the rate measurement, passed straight to
   * `useObjectTracking`. */
  now?: (() => number) | undefined;
}

/** A slider paired with a number box, both bound to one value — the shape
 * `OutpaintingPanel`'s `PadSlider` uses, including its `aria-label` discipline:
 * an implicit label binds to the FIRST control inside it, so without explicit
 * names the number box would have none. Both carry the same name because they
 * are two views of one value.
 *
 * Values are clamped here rather than in the setter: the RANGE input cannot
 * leave its bounds, but the NUMBER box can be typed to anything, and an
 * out-of-range value would not fail until native answered `BAD_REQUEST` minutes
 * into a run. (`PadSlider` clamps for the same reason, one layer down in
 * `useOutpaintForm`.) */
function TrackNumberField({
  label,
  hint,
  value,
  min,
  max,
  step,
  disabled,
  onChange,
}: {
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled: boolean;
  onChange: (value: number) => void;
}) {
  const commit = (raw: string) => {
    // An EMPTY box is "mid-edit", not a value: the user cleared it to retype.
    // `Number("")` is 0, not NaN, so without this line clearing the box would
    // silently snap the setting to its minimum.
    if (raw.trim() === "") return;
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) return;
    onChange(Math.min(max, Math.max(min, parsed)));
  };
  return (
    <div className="toolbox-field">
      <label className="field">
        <span className="field-label">{label}</span>
        <input
          type="range"
          aria-label={label}
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(e) => commit(e.target.value)}
        />
        <input
          type="number"
          aria-label={label}
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(e) => commit(e.target.value)}
        />
      </label>
      <p className="field-hint">{hint}</p>
    </div>
  );
}

export function ObjectTrackingSection({
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
  now,
}: ObjectTrackingSectionProps) {
  const strings = useStrings();
  const t = strings.toolbox.tracking;
  const run = useObjectTracking({
    nativeBridge,
    trackRequest,
    settings,
    ...(now ? { now } : {}),
  });

  const running = run.phase === "running";
  // Two reasons a control is dead, one treatment: the server cannot track, or
  // a run is already using these values. Combining them keeps the panel from
  // growing a second greyed-out look for what the user reads as the same fact
  // ("not now") — the same judgment `ModeTabs` makes about its two reasons.
  const settingsDisabled = !trackingAvailable || running;

  const lostRanges = run.result?.lostRanges ?? [];
  const elapsedSec = (run.result?.elapsedMs ?? 0) / 1000;

  // A code with no sentence of its own is shown verbatim — the backend's codes
  // (`TRACK_UNAVAILABLE` and friends) travel through this method too, and a
  // bare code the user can search for beats a wrong guess at what it meant.
  const errorSentences: Record<string, string | undefined> = t.errors;
  const errorText =
    errorSentences[run.errorCode] ?? (run.errorCode !== "" ? run.errorCode : run.errorMessage);

  return (
    <section className="toolbox-section">
      <h2 className="toolbox-section-heading">{t.heading}</h2>
      <p className="field-hint">{t.intro}</p>
      <p className="field-hint">{t.independenceHint}</p>

      {!trackingAvailable && (
        <div className="toolbox-unavailable">
          <p className="field-label">{t.unavailableHeading}</p>
          <p className="field-hint">
            {trackingReason === "worker failed" ? t.unavailableWorkerFailed : t.unavailableNotInstalled}
          </p>
        </div>
      )}

      {/* Model. Both buttons are permanently disabled: this is a seat kept
          visible for a second tracker, not a choice — so nothing is stored and
          nothing reacts. Greyed with the SAME treatment a disabled mode tab
          gets (`.mode-tab:disabled`), so "not a real option" looks the same
          everywhere in the app. */}
      <fieldset className="toolbox-group">
        <legend className="field-label">{t.model.label}</legend>
        <label className="field field-inline">
          <input type="radio" name="track-model" checked readOnly disabled />
          <span className="field-label">{t.model.uetrack}</span>
        </label>
        <label className="field field-inline">
          <input type="radio" name="track-model" checked={false} readOnly disabled />
          <span className="field-label">{t.model.other}</span>
        </label>
        <p className="field-hint">{t.model.hint}</p>
      </fieldset>

      <TrackNumberField
        label={t.searchFactor.label}
        hint={t.searchFactor.hint}
        value={settings.searchFactor}
        min={SEARCH_FACTOR_MIN}
        max={SEARCH_FACTOR_MAX}
        step={SEARCH_FACTOR_STEP}
        disabled={settingsDisabled}
        onChange={onSearchFactorChange}
      />

      <TrackNumberField
        label={t.lostThreshold.label}
        hint={t.lostThreshold.hint}
        value={settings.lostScoreThreshold}
        min={LOST_THRESHOLD_MIN}
        max={LOST_THRESHOLD_MAX}
        step={LOST_THRESHOLD_STEP}
        disabled={settingsDisabled}
        onChange={onLostScoreThresholdChange}
      />

      <fieldset className="toolbox-group">
        <legend className="field-label">{t.lostBehavior.label}</legend>
        {(
          [
            ["hold", t.lostBehavior.hold],
            ["continue", t.lostBehavior.continueOn],
          ] as Array<[LostBehavior, string]>
        ).map(([value, label]) => (
          <label key={value} className="field field-inline">
            <input
              type="radio"
              name="track-lost-behavior"
              checked={settings.lostBehavior === value}
              disabled={settingsDisabled}
              onChange={() => onLostBehaviorChange(value)}
            />
            <span className="field-label">{label}</span>
          </label>
        ))}
      </fieldset>

      <TrackNumberField
        label={t.smoothing.label}
        hint={t.smoothing.hint}
        value={settings.smoothing}
        min={SMOOTHING_MIN}
        max={SMOOTHING_MAX}
        step={SMOOTHING_STEP}
        disabled={settingsDisabled}
        onChange={onSmoothingChange}
      />

      <div className="toolbox-field">
        <label className="field field-inline">
          <input
            type="checkbox"
            checked={settings.followSize}
            disabled={settingsDisabled}
            onChange={(e) => onFollowSizeChange(e.target.checked)}
          />
          <span className="field-label">{t.followSize.label}</span>
        </label>
        <p className="field-hint">{t.followSize.hint}</p>
      </div>

      <TrackNumberField
        label={t.keyframeStride.label}
        hint={t.keyframeStride.hint}
        value={settings.keyframeStride}
        min={KEYFRAME_STRIDE_MIN}
        max={KEYFRAME_STRIDE_MAX}
        step={KEYFRAME_STRIDE_STEP}
        disabled={settingsDisabled}
        onChange={onKeyframeStrideChange}
      />

      <h3 className="toolbox-section-heading">{t.progressHeading}</h3>
      {/* `aria-live` so the count is announced as it moves without the user
          having to go hunting for it — the readout changes on its own. */}
      <p className="toolbox-progress" aria-live="polite">
        {run.progress
          ? `${t.progressFrames(run.progress.index, run.progress.total)} ・ ${
              run.progress.fps === null ? t.progressFpsUnknown : t.progressFps(run.progress.fps)
            }`
          : t.progressIdle}
      </p>
      {/* Pressable in one case where THIS panel is not running: `TRACK_BUSY`,
          which means a run started by an earlier right-click is still going —
          native holds one tracking session at a time, so the stop this button
          sends reaches that run. Leaving it greyed would show the user the
          sentence "stop it first" beside the only control that could. */}
      <button
        type="button"
        className="toolbox-stop"
        disabled={!running && run.errorCode !== "TRACK_BUSY"}
        onClick={run.cancel}
      >
        {t.stop}
      </button>

      {run.result !== null && (
        <>
          <p className="toolbox-result">
            {run.result.cancelled
              ? t.resultCancelled(run.result.frames, run.result.keyframes, elapsedSec)
              : t.resultDone(run.result.frames, run.result.keyframes, elapsedSec)}
          </p>
          <h3 className="toolbox-section-heading">{t.lostHeading}</h3>
          {lostRanges.length === 0 ? (
            <p className="field-hint">{t.lostNone}</p>
          ) : (
            <ul className="toolbox-lost-list">
              {lostRanges.map((range) => (
                <li key={`${range.start}-${range.end}`}>{t.lostRange(range.start, range.end)}</li>
              ))}
            </ul>
          )}
        </>
      )}

      {run.phase === "error" && (
        <div className="toolbox-error" role="alert">
          <p className="field-label">{t.errorHeading}</p>
          <p className="field-hint field-hint-error">{errorText}</p>
        </div>
      )}
    </section>
  );
}
