import { useStrings } from "../i18n/LanguageContext";
import { FloatSliderField } from "../modes/single/FloatSliderField";
import {
  DEFAULT_NEGATIVE_PROMPT,
  NAG_ALPHA_MAX,
  NAG_ALPHA_MIN,
  NAG_ALPHA_STEP,
  NAG_SCALE_MAX,
  NAG_SCALE_MIN,
  NAG_SCALE_STEP,
  NAG_TAU_MAX,
  NAG_TAU_MIN,
  NAG_TAU_STEP,
  VSF_SCALE_MAX,
  VSF_SCALE_MIN,
  VSF_SCALE_STEP,
} from "./nagSettings";
import type { NagSettings } from "./nagSettings";
import { isOverTokenLimit, TokenCounter } from "./TokenCounter";
import "../styles/formAccordion.css";

export interface NagAccordionControls {
  setEnabled: (value: boolean) => void;
  setText: (value: string) => void;
  setScale: (value: number) => void;
  setTau: (value: number) => void;
  setAlpha: (value: number) => void;
  setMethod: (value: NagSettings["method"]) => void;
  setVsfScale: (value: number) => void;
  resetParams: () => void;
}

export interface NagAccordionProps {
  nag: NagSettings;
  controls: NagAccordionControls;
  /** The loaded base model's engine cannot run NAG/VSF
   * (`featureScope.nagDisabledFor`). The accordion stays visible but GREYED —
   * every control disabled, a note on top, no ON badge — and the shell sends
   * none of its fields meanwhile (`nagSettings.effectiveNag`), so a NAG left
   * on from another base model neither 422s a job nor hides out of reach. The
   * user's settings are kept as they are for when a NAG-capable base model
   * comes back. Omitted = available. */
  unavailable?: boolean | undefined;
  /** §1-111: token count of the negative prompt text and the engine's limit
   * (`TokenCounter`). Shown even while the field is greyed (incl. H3). */
  tokenCount?: number | null | undefined;
  tokenLimit?: number | null | undefined;
}

/**
 * The shared "Negative Prompt" accordion (D4), rendered ONCE by `AppShell`
 * directly below `PromptBar` and outside every mode tab — Create/Chain/Batch
 * all read the identical `NagSettings` (`shell/useNagSettings.ts`, through
 * `nagSettings.effectiveNag`) this way,
 * mirroring `PromptBar` itself. Presentation-only: every value comes from
 * `nag`, every mutation goes through `controls` — this component owns no
 * state of its own.
 *
 * `<details>` is uncontrolled and defaults closed (owner decision §UX 1/2):
 * the summary shows nothing extra while off, and only the ON badge while
 * `nag.enabled` (never any other indicator).
 */
export function NagAccordion({ nag, controls, unavailable = false, tokenCount, tokenLimit }: NagAccordionProps) {
  const strings = useStrings();
  const t = strings.nag;
  // While the engine cannot run NAG every control greys,
  // including the ones that are otherwise always operable (checkbox, radios).
  const off = !nag.enabled || unavailable;

  return (
    <details className="form-accordion nag-accordion">
      <summary className="form-accordion-summary">
        {t.heading}
        {nag.enabled && !unavailable && <span className="nag-on-badge">{t.onBadge}</span>}
      </summary>
      <div className={`form-accordion-body${unavailable ? " field-reserved" : ""}`}>
        {unavailable && <p className="warning-banner">{t.unavailableOnBaseModel}</p>}
        {/* No visible label (2026-07-28 slim-down: `PROMPT`-style items
            already say what this field is via the accordion heading itself,
            so a repeated "NEGATIVE PROMPT" caption above the textarea was
            pure duplication) — `aria-label` keeps the accessible name intact
            for `App.nag.test.tsx`'s `getByRole("textbox", { name: /negative
            prompt/i })` queries (same fix-up `BatchTable.tsx:106-114` used
            when it dropped a visible column label). */}
        <div className="prompt-field-head">
          <TokenCounter count={tokenCount} limit={tokenLimit} />
        </div>
        <textarea
          className={`prompt-input${isOverTokenLimit(tokenCount, tokenLimit) ? " input-warning" : ""}`}
          rows={2}
          placeholder={DEFAULT_NEGATIVE_PROMPT}
          value={nag.text}
          disabled={off}
          aria-label={t.textLabel}
          onChange={(e) => controls.setText(e.target.value)}
        />

        {/* Checkbox + both method radios share one row now (2026-07-28
            slim-down): the standalone `METHOD` caption and the two
            explanatory paragraphs it used to sit above are gone, and
            `.nag-toggle-row` (flex-wrap) lets the whole group share a single
            line when there's room, wrapping naturally at narrow widths
            without a media query — same recipe as `.batch-radio-row`.

            The radios are ALWAYS operable, independent of `nag.enabled`
            (2026-07-29, owner-confirmed design: picking a method is a
            declaration of "what to use next", not a live parameter — it's
            harmless to change while off, unlike the textarea/sliders below
            which soft-disable). `name="nag-method"` must stay unique
            app-wide — a duplicate radio `name` elsewhere in the mounted tree
            would silently link the two groups (DEVLOG §45's name-collision
            precedent). */}
        <div className="nag-toggle-row">
          <label className="field field-inline">
            <input
              type="checkbox"
              checked={nag.enabled}
              disabled={unavailable}
              onChange={(e) => controls.setEnabled(e.target.checked)}
            />
            <span className="field-label">{t.enableLabel}</span>
          </label>
          <label className="field field-inline">
            <input
              type="radio"
              name="nag-method"
              checked={nag.method === "nag"}
              disabled={unavailable}
              onChange={() => controls.setMethod("nag")}
            />
            <span>{t.methodNag}</span>
          </label>
          {/* VSF (Value Sign Flip, 2026-07-29) replaces the old permanently-
              disabled "Other" placeholder — `title` carries the one-line
              "0 doesn't mean off" caveat that used to be `methodOtherDisabledHint`
              (same spot: on the wrapping `<label>`, not the `<input>`, since a
              radio's own `title` tooltip is unreliable on hover). */}
          <label className="field field-inline" title={t.methodVsfHint}>
            <input
              type="radio"
              name="nag-method"
              checked={nag.method === "vsf"}
              disabled={unavailable}
              onChange={() => controls.setMethod("vsf")}
            />
            <span>{t.methodVsf}</span>
          </label>
        </div>

        {/* Scale + 🔄 share a row now (2026-07-28 slim-down: the reset button
            moved out of the Advanced sub-accordion so it's reachable without
            opening it). Which slider renders here depends on `nag.method`
            (2026-07-29): NAG's own scale, or VSF's — the 🔄 button resets
            both plus tau/alpha regardless of which is showing (`resetParams`
            touches all four numeric params at once), so its meaning doesn't
            change across a method switch. */}
        <div className="nag-scale-row">
          {nag.method === "nag" ? (
            <FloatSliderField
              label={t.scaleLabel}
              value={nag.scale}
              min={NAG_SCALE_MIN}
              max={NAG_SCALE_MAX}
              step={NAG_SCALE_STEP}
              disabled={off}
              onChange={controls.setScale}
            />
          ) : (
            <FloatSliderField
              label={t.vsfScaleLabel}
              value={nag.vsfScale}
              min={VSF_SCALE_MIN}
              max={VSF_SCALE_MAX}
              step={VSF_SCALE_STEP}
              disabled={off}
              onChange={controls.setVsfScale}
            />
          )}
          <button
            type="button"
            className="icon-action-button"
            title={t.resetTooltip}
            aria-label={t.resetTooltip}
            disabled={off}
            onClick={controls.resetParams}
          >
            <span aria-hidden="true">🔄</span>
          </button>
        </div>

        {/* Advanced (tau/alpha) only applies to NAG — VSF has no equivalent
            pair, so the whole sub-accordion is hidden while `method === "vsf"`
            (2026-07-29). Its open/closed state is lost across a method switch
            since `<details>` is uncontrolled here — accepted as-is (see the
            module doc comment's own note on `<details>` being uncontrolled). */}
        {nag.method === "nag" && (
          <details className="form-accordion nag-subaccordion">
            <summary className="form-accordion-summary">{t.advancedHeading}</summary>
            <div className="form-accordion-body">
              <FloatSliderField
                label={t.tauLabel}
                value={nag.tau}
                min={NAG_TAU_MIN}
                max={NAG_TAU_MAX}
                step={NAG_TAU_STEP}
                disabled={off}
                onChange={controls.setTau}
              />
              <FloatSliderField
                label={t.alphaLabel}
                value={nag.alpha}
                min={NAG_ALPHA_MIN}
                max={NAG_ALPHA_MAX}
                step={NAG_ALPHA_STEP}
                disabled={off}
                onChange={controls.setAlpha}
              />
            </div>
          </details>
        )}
      </div>
    </details>
  );
}
