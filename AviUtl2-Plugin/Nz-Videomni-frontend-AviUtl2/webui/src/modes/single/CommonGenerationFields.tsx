import type { CSSProperties } from "react";
import type { CropOutput } from "../../api/types";
import { useStrings } from "../../i18n/LanguageContext";
import { useJobsContext } from "../../jobs/JobsContext";
import { latestJobSeed } from "../../jobs/seedUtils";
import { clampCropOutput, CROP_OUTPUT_MIN } from "../chained/chainUtils";
import { FRAME_RATE_MAX, FRAME_RATE_MIN, isStepEvent } from "./paramUtils";

/** Guide positions drawn on the width/height sliders, in pixels of the very
 * same axis the slider controls.
 *
 * Structurally identical to `shell/tokenBudget.ts`'s `ChainWindowBudgetMarkers`
 * and deliberately NOT imported from it: that module is chain-only by its own
 * scope note ("do not import this from Create/single-generate code"), while
 * this file is shared with Create and Retake. The duplication is one interface
 * of four numbers and keeps the shared component free of any opinion about
 * WHERE the guides came from — it only draws them.
 *
 * `limitWidth`/`limitHeight` are `null` when there is no ceiling to warn about;
 * the comfort pair is always present. Any value outside its slider's
 * `[min, max]` is simply not drawn. */
export interface SizeBudgetMarkers {
  comfortWidth: number;
  comfortHeight: number;
  limitWidth: number | null;
  limitHeight: number | null;
}

interface SizeSliderProps {
  value: number;
  min: number;
  max: number;
  disabled: boolean;
  onChange: (value: number) => void;
  /** The neutral "recommended point" guide, or `null`/omitted for no guide.
   * The explicit `| undefined` is `exactOptionalPropertyTypes` (the same reason
   * `SizeFieldsProps.onGetSize` spells it out): the caller forwards
   * `budgetMarkers?.…`, which is `undefined` whenever the group is omitted. */
  comfort?: number | null | undefined;
  /** The red "ceiling" guide, or `null`/omitted for no guide. */
  limit?: number | null | undefined;
  comfortTitle: (px: number) => string;
  limitTitle: (px: number) => string;
}

/** One width/height range input, plus the optional guide overlay.
 *
 * The overlay is absolutely-positioned inside a relative wrapper span: a
 * `<datalist>` cannot be coloured and painting the track with a gradient would
 * cost the native thumb, so the guides are a visible 1px line each — matching
 * the hairline of the Single tab's native `<datalist>` ticks (owner's verdict
 * on G-D1, 2026-08-12). CSS widens the actual hit area to 11px via
 * `background-clip: content-box` — `SingleScreen.css`'s `.size-marker` — since
 * a bare 1px target is too small to hover reliably (A-1, 2026-08-12); note the
 * colour rules there MUST use `background-color`, never the `background`
 * shorthand, which resets that clip and paints the whole hit area as a band.
 * They sit inside the 10px band the wrapper reserves ABOVE the slider track
 * (`.size-slider-wrap-marked`'s `padding-top`), so hovering/clicking them never
 * lands on the track itself and the control stays fully operable either way.
 * `title` carries the tooltip text (`comfortTitle`/`limitTitle`); they remain
 * `aria-hidden` because the over-limit state they'd otherwise announce is
 * already conveyed as text by the warning banner elsewhere on the same screen,
 * so a duplicate announcement here would add noise for screen-reader users
 * without new information. They carry their pixel value in `data-marker-value`
 * so tests can assert POSITION-INDEPENDENTLY (jsdom computes no layout, so the
 * `%` maths is only ever verified on real hardware).
 *
 * A guide is drawn on the closed interval `min <= value <= max`. Both ends are
 * INCLUSIVE on purpose: a guide can legitimately coincide with the slider's own
 * ceiling (the shorter step's recommended 1920x1088 is exactly the mock/1080p
 * ceiling), and excluding it there would silently hide the guide precisely
 * where it matters most. A degenerate range (`max <= min`) draws nothing rather
 * than dividing by zero. The red guide comes second in DOM order so it wins the
 * overlap when both land on the same position. */
function SizeSlider({
  value,
  min,
  max,
  disabled,
  onChange,
  comfort,
  limit,
  comfortTitle,
  limitTitle,
}: SizeSliderProps) {
  const span = max - min;
  const drawable = (px: number | null | undefined): px is number =>
    span > 0 && px != null && px >= min && px <= max;
  const markers: { kind: "comfort" | "limit"; px: number; title: string }[] = [];
  if (drawable(comfort)) markers.push({ kind: "comfort", px: comfort, title: comfortTitle(comfort) });
  if (drawable(limit)) markers.push({ kind: "limit", px: limit, title: limitTitle(limit) });

  return (
    <span className={`size-slider-wrap${markers.length > 0 ? " size-slider-wrap-marked" : ""}`}>
      <input
        type="range"
        min={min}
        max={max}
        step={64}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {markers.map((marker) => (
        <span
          key={marker.kind}
          className={`size-marker size-marker-${marker.kind}`}
          data-size-marker={marker.kind}
          data-marker-value={marker.px}
          title={marker.title}
          aria-hidden="true"
          // React's CSSProperties has no room for custom properties, hence the
          // cast — `--marker-pos` is a unitless 0..1 the stylesheet multiplies
          // by the track length.
          style={{ "--marker-pos": (marker.px - min) / span } as CSSProperties}
        />
      ))}
    </span>
  );
}

export interface SizeFieldsProps {
  width: number;
  height: number;
  minWidth: number;
  maxWidth: number;
  minHeight: number;
  maxHeight: number;
  disabled: boolean;
  onWidthChange: (value: number, snap: boolean) => void;
  onHeightChange: (value: number, snap: boolean) => void;
  /** The "Get size from AviUtl2" button's three props. OPTIONAL as a group
   * (2026-08-10, §1-17 Retake): omit `onGetSize` and the button — with its
   * spinner label and its error line — is not rendered at all. Retake's
   * width/height come from the material it was invoked on, so importing the
   * project's own resolution there is meaningless; every other caller
   * (Create/Chain) passes all three and is unchanged. */
  gettingSize?: boolean;
  getSizeError?: string | null;
  onGetSize?: (() => void) | undefined;
  /** Comfortable-resolution guides drawn on the two sliders (2026-08-12).
   * OPTIONAL, following `onGetSize`'s precedent above: omit it and no guide —
   * and no wrapper class change beyond the plain span — is rendered, so Create
   * and Retake keep their exact previous DOM. Only Chain passes it
   * (`useChainForm`'s `chainWindowMarkers`), because the budget being drawn is
   * chain-specific (one stage-2 window of a chain). */
  budgetMarkers?: SizeBudgetMarkers;
}

/** Width/height sliders + numeric inputs, plus the "Get size from AviUtl2"
 * button. Extracted out of Create's `GenerationForm` (M6) so Chain mode's
 * common-parameters section can share the exact same width/height controls
 * instead of re-implementing them — both take primitive
 * value/limit/callback props rather than the Create-specific
 * `useGenerationForm` hook, so this has no opinion about *how* the caller
 * derives or snaps those values (Create uses `roundToMultiple`; Chain uses
 * the same helper directly in `useChainForm`). */
export function SizeFields({
  width,
  height,
  minWidth,
  maxWidth,
  minHeight,
  maxHeight,
  disabled,
  onWidthChange,
  onHeightChange,
  gettingSize = false,
  getSizeError = null,
  onGetSize,
  budgetMarkers,
}: SizeFieldsProps) {
  const strings = useStrings();
  return (
    <>
      <div className="size-field-stack">
        <label className="field">
          <span className="field-label">{strings.single.size.width}</span>
          <SizeSlider
            min={minWidth}
            max={maxWidth}
            value={width}
            disabled={disabled}
            onChange={(value) => onWidthChange(value, true)}
            comfort={budgetMarkers?.comfortWidth}
            limit={budgetMarkers?.limitWidth}
            comfortTitle={strings.single.size.comfortMarkerTitle}
            limitTitle={strings.single.size.limitMarkerTitle}
          />
          <input
            type="number"
            min={minWidth}
            max={maxWidth}
            step={64}
            value={width}
            disabled={disabled}
            onChange={(e) => onWidthChange(Number(e.target.value), isStepEvent(e.nativeEvent))}
          />
        </label>

        <label className="field">
          <span className="field-label">{strings.single.size.height}</span>
          <SizeSlider
            min={minHeight}
            max={maxHeight}
            value={height}
            disabled={disabled}
            onChange={(value) => onHeightChange(value, true)}
            comfort={budgetMarkers?.comfortHeight}
            limit={budgetMarkers?.limitHeight}
            comfortTitle={strings.single.size.comfortMarkerTitle}
            limitTitle={strings.single.size.limitMarkerTitle}
          />
          <input
            type="number"
            min={minHeight}
            max={maxHeight}
            step={64}
            value={height}
            disabled={disabled}
            onChange={(e) => onHeightChange(Number(e.target.value), isStepEvent(e.nativeEvent))}
          />
        </label>
      </div>

      {onGetSize && (
        <div>
          <button
            type="button"
            className="secondary-button"
            disabled={disabled || gettingSize}
            onClick={onGetSize}
          >
            {gettingSize ? strings.single.size.gettingFromAviUtl2 : strings.single.size.getFromAviUtl2}
          </button>
          {getSizeError && <span className="field-hint field-hint-error"> {getSizeError}</span>}
        </div>
      )}
    </>
  );
}

export interface CropOutputFieldProps {
  /** `null` = "not set" (the field is omitted from the request); a
   * `{width,height}` while the user has opted in. */
  value: CropOutput | null;
  onChange: (value: CropOutput | null) => void;
  /** The CURRENT generation width/height — the ceiling for crop width/height
   * (never `config.limits.max_width/max_height`; a crop can only ever be
   * smaller than or equal to what's actually being generated). See
   * `chainUtils.CROP_OUTPUT_MIN`'s doc comment — unlike width/height, a crop
   * dimension has no fixed-grid constraint (matching Gradio), just the
   * `[CROP_OUTPUT_MIN, max]` range. */
  maxWidth: number;
  maxHeight: number;
  disabled: boolean;
}

/** N1: opt-in output crop (`crop_output`) — a checkbox that flips `value`
 * between `null` ("not set") and a `{width,height}` seeded from the current
 * generation size, plus two number inputs (any integer in range — no
 * fixed-grid snapping, matching Gradio) shown only while enabled. Mirrors
 * `ReferenceStrengthField`'s null<->value toggle shape. Shared by Create
 * (`GenerationForm.tsx`) and Chain (`ChainedScreen.tsx`), both placed directly
 * after `SizeFields`. Every input passes through `chainUtils.clampCropOutput`
 * so a stored crop can never itself be out of range or over the *current*
 * width/height ceiling — but if the user later shrinks width/height below a
 * previously-set crop, this field does NOT reactively re-clamp;
 * `useGenerationForm`/`useChainForm`'s `isValid` (via
 * `chainUtils.isCropOutputValid`) catches that stale combination instead,
 * mirroring how width/height's own free-typed grid violations are only
 * ever caught at submit time, not silently corrected mid-edit. */
export function CropOutputField({ value, onChange, maxWidth, maxHeight, disabled }: CropOutputFieldProps) {
  const strings = useStrings();
  const t = strings.single.crop;
  return (
    <div className="field">
      <label className="field field-inline">
        <input
          type="checkbox"
          checked={value !== null}
          disabled={disabled}
          onChange={(e) =>
            onChange(e.target.checked ? clampCropOutput({ width: maxWidth, height: maxHeight }, maxWidth, maxHeight) : null)
          }
        />
        <span className="field-label">{t.label}</span>
      </label>
      <p className="field-hint">{t.hint}</p>
      {value !== null && (
        <div className="field-row">
          <label className="field">
            <span className="field-label">{t.width}</span>
            <input
              type="number"
              min={CROP_OUTPUT_MIN}
              max={maxWidth}
              step={1}
              value={value.width}
              disabled={disabled}
              onChange={(e) => onChange(clampCropOutput({ width: Number(e.target.value), height: value.height }, maxWidth, maxHeight))}
            />
          </label>
          <label className="field">
            <span className="field-label">{t.height}</span>
            <input
              type="number"
              min={CROP_OUTPUT_MIN}
              max={maxHeight}
              step={1}
              value={value.height}
              disabled={disabled}
              onChange={(e) => onChange(clampCropOutput({ width: value.width, height: Number(e.target.value) }, maxWidth, maxHeight))}
            />
          </label>
        </div>
      )}
    </div>
  );
}

export interface DurationFieldProps {
  label: string;
  value: number;
  min: number;
  max: number;
  disabled: boolean;
  /** `snap` mirrors `SizeFields`' `onWidthChange`/`onHeightChange`: `true`
   * for the slider/stepper (snap onto the 8n+1 grid), `false` for free
   * keyboard typing (pass through untouched — `isStepEvent` tells them
   * apart). */
  onChange: (value: number, snap: boolean) => void;
  /** Optional "value just got committed" signal, separate from `onChange`'s
   * per-keystroke/per-drag stream: fires once the user has settled on a
   * value (slider release/blur/keyup, or number-field blur/Enter/stepper),
   * so a caller can react to "duration just changed" without reacting to
   * every intermediate drag position. `snap` mirrors `onChange`'s meaning
   * (`true` for slider/stepper-driven commits, `false` for free typing).
   * Omitted entirely by Chain's `ClipCard` — when unset this is a strict
   * no-op with zero other behavior change, so Chain's duration field is
   * unaffected. */
  onCommit?: (value: number, snap: boolean) => void;
  /** Ready-to-render "N frames ≈ X.Xs @ fps" hint (`paramUtils.formatDurationHint`). */
  hint: string;
  /** Create-only: the comfortable `num_frames` ceiling for the current
   * width/height (`spillUtils.resolveSpillFreeFrames`), rendered as a tick on
   * the slider via a `<datalist>`. `null`/omitted hides the tick entirely —
   * Chain's per-clip duration has no equivalent threshold. */
  spillThresholdFrames?: number | null;
  /** Create-only: `true` once `value` exceeds `spillThresholdFrames` — shows
   * `spillWarningText` as a mild warning banner. */
  isOverSpillThreshold?: boolean;
  spillWarningText?: string;
}

/** Duration (`num_frames`) range+number pair, following `SizeFields`'
 * "range+number both editable, `isStepEvent` snaps stepper/slider input while
 * free typing passes through untouched" pattern. Shared by Create
 * (`GenerationForm.tsx`, one instance for `values.numFrames`) and Chain
 * (`ClipCard.tsx`, one instance per clip) — the optional spill-threshold
 * datalist tick + warning banner are Create-only (Chain's per-clip duration
 * has no such threshold), so they're plain optional props rather than a
 * second component. */
export function DurationField({
  label,
  value,
  min,
  max,
  disabled,
  onChange,
  onCommit,
  hint,
  spillThresholdFrames,
  isOverSpillThreshold,
  spillWarningText,
}: DurationFieldProps) {
  const spillTickId = "spill-tick";
  // Duplicate commits (e.g. a keyup immediately followed by blur) are left
  // unfiltered on purpose — the receiver is expected to treat `onCommit` as
  // idempotent, and de-duping here would add state for no real benefit.
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={8}
        value={value}
        disabled={disabled}
        list={spillThresholdFrames != null ? spillTickId : undefined}
        onChange={(e) => onChange(Number(e.target.value), true)}
        onPointerUp={(e) => onCommit?.(Number(e.currentTarget.value), true)}
        onKeyUp={(e) => onCommit?.(Number(e.currentTarget.value), true)}
        onBlur={(e) => onCommit?.(Number(e.currentTarget.value), true)}
      />
      <input
        type="number"
        min={min}
        max={max}
        step={8}
        value={value}
        disabled={disabled}
        onChange={(e) => {
          const snap = isStepEvent(e.nativeEvent);
          onChange(Number(e.target.value), snap);
          // A stepper click/arrow-key edit is a self-contained "commit" —
          // there's no separate release/blur gesture to wait for.
          if (snap) onCommit?.(Number(e.target.value), true);
        }}
        onBlur={(e) => onCommit?.(Number(e.currentTarget.value), false)}
        onKeyDown={(e) => {
          if (e.key === "Enter") onCommit?.(Number(e.currentTarget.value), false);
        }}
      />
      {spillThresholdFrames != null && (
        <datalist id={spillTickId}>
          <option value={spillThresholdFrames} />
        </datalist>
      )}
      <span className="field-hint">{hint}</span>
      {isOverSpillThreshold && spillWarningText && (
        <p className="warning-banner warning-banner-mild">{spillWarningText}</p>
      )}
    </label>
  );
}

export interface FrameRateSeedFieldsProps {
  frameRate: number;
  seed: number;
  disabled: boolean;
  onFrameRateChange: (value: number) => void;
  onSeedChange: (value: number) => void;
}

/** Frame-rate + seed inputs, extracted out of Create's `GenerationForm` (M6)
 * for the same reason as `SizeFields` above — Chain's common-parameters
 * section reuses it verbatim. */
export function FrameRateSeedFields({
  frameRate,
  seed,
  disabled,
  onFrameRateChange,
  onSeedChange,
}: FrameRateSeedFieldsProps) {
  const strings = useStrings();
  // ⑤ seed 🎲/♻ helpers. `latestJobSeed` reads the same job ledger the panel
  // shows (`JobsContext`), so ♻ offers the seed of the most recent job and is
  // disabled when there is none to reuse. Both screens that render this
  // (Create/Chain) are already inside a `JobsProvider`.
  const { jobs } = useJobsContext();
  const lastSeed = latestJobSeed(jobs);
  return (
    <>
      <label className="field field-inline">
        <span className="field-label">{strings.single.duration.fps}</span>
        <input
          type="number"
          min={FRAME_RATE_MIN}
          max={FRAME_RATE_MAX}
          value={frameRate}
          disabled={disabled}
          onChange={(e) => onFrameRateChange(Number(e.target.value))}
        />
      </label>

      <label className="field field-inline">
        <span className="field-label">{strings.single.seed.label}</span>
        <input
          type="number"
          value={seed}
          disabled={disabled}
          onChange={(e) => onSeedChange(Number(e.target.value))}
        />
        <span className="seed-actions">
          <button
            type="button"
            className="icon-action-button"
            title={strings.single.seed.randomTooltip}
            aria-label={strings.single.seed.randomTooltip}
            disabled={disabled}
            onClick={() => onSeedChange(-1)}
          >
            <span aria-hidden="true">🎲</span>
          </button>
          <button
            type="button"
            className="icon-action-button"
            title={strings.single.seed.reuseTooltip}
            aria-label={strings.single.seed.reuseTooltip}
            disabled={disabled || lastSeed === null}
            onClick={() => {
              if (lastSeed !== null) onSeedChange(lastSeed);
            }}
          >
            <span aria-hidden="true">♻</span>
          </button>
        </span>
        <span className="field-hint">{strings.single.seed.hint}</span>
      </label>
    </>
  );
}

/**
 * Permanently disabled CFG-scale placeholder for a future non-distilled
 * backend (Mock/AVIUTL2_DESIGN_BRIEF.md §5-2 / §11). Shared verbatim by
 * Create (`GenerationForm.tsx`) and Chain (`ChainedScreen.tsx`) — never
 * enabled, read, or included in any request body; there is no corresponding
 * state anywhere in `useGenerationForm`/`useChainForm` on purpose (task brief
 * M7b: "値はどこにも送らない"). Always shown fixed at "1.0", matching the
 * server's own hardcoded default for the distilled pipeline
 * (Docs/API_REFERENCE.md, `api/types.ts`'s `GenerateRequest` doc comment).
 *
 * The negative-prompt textarea this used to also render was removed
 * (2026-07-28) once NAG (Normalized Attention Guidance) unlocked a real,
 * sent negative prompt — see `shell/NagAccordion.tsx` instead
 * (`strings.single.negativePrompt` was deleted along with it).
 */
export function ReservedFields() {
  const strings = useStrings();
  const note = strings.single.reservedNote;

  return (
    <label className="field field-inline field-reserved" title={note}>
      <span className="field-label">{strings.single.cfgScale.label}</span>
      <input
        type="range"
        min={1}
        max={1}
        value={1}
        disabled
        title={note}
        aria-label={`${strings.single.cfgScale.label} — ${note}`}
      />
      <span className="field-hint">1.0</span>
    </label>
  );
}
