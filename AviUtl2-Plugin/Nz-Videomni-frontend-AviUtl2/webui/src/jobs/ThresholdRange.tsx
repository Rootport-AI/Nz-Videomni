import { useStrings } from "../i18n/LanguageContext";

/** The slider's ends (8-bit matte values, Docs/ALPHAGEN_DESIGN.md §10.2). */
export const THRESHOLD_MIN = 0;
export const THRESHOLD_MAX = 255;

export interface ThresholdRangeProps {
  /** The transparent threshold (left thumb). */
  transparent: number;
  /** The opaque threshold (right thumb). */
  opaque: number;
  /** Fired with both values whenever either thumb moves; the moved one is
   * already held on its own side of the other (they never cross). */
  onChange: (transparent: number, opaque: number) => void;
}

/**
 * Alpha Gen 第 3 弾: the two-thumb 0..255 slider for the matte thresholds
 * (Docs/ALPHAGEN_DESIGN.md §10.2). Two plain `<input type="range">` laid on top
 * of each other, so keyboard and screen-reader handling stay the browser's
 * own. The thumbs never cross: the left one stops at the right one's value and
 * vice versa (held on the VALUE side, so the keyboard obeys it too); equal
 * values are allowed (the black-and-white case). `RangeBand` is not reused: it
 * carries Retake's own seconds / 8-step / window-drag display, and its thumbs
 * lock up when they meet. The paired number fields live in the caller
 * (`AlphaControls`).
 */
export function ThresholdRange({ transparent, opaque, onChange }: ThresholdRangeProps) {
  const strings = useStrings();
  const t = strings.jobs.alpha;
  const span = THRESHOLD_MAX - THRESHOLD_MIN;
  const leftPct = ((transparent - THRESHOLD_MIN) / span) * 100;
  const rightPct = ((THRESHOLD_MAX - opaque) / span) * 100;
  // Where the thumbs meet, only the upper input can be grabbed. Lift the left
  // one in the upper half of the range, so a pair stuck together at 255 can
  // still be pulled apart (and the right one, on top by default, at 0).
  const leftOnTop = transparent > span / 2;
  return (
    <div className="threshold-range">
      <div className="threshold-range-track" aria-hidden="true">
        <div className="threshold-range-fill" style={{ left: `${leftPct}%`, right: `${rightPct}%` }} />
      </div>
      <input
        type="range"
        className={`threshold-range-input${leftOnTop ? " threshold-range-input--top" : ""}`}
        min={THRESHOLD_MIN}
        max={THRESHOLD_MAX}
        step={1}
        value={transparent}
        aria-label={t.transparentThreshold}
        title={t.transparentThresholdTitle}
        onChange={(e) => onChange(Math.min(Number(e.target.value), opaque), opaque)}
      />
      <input
        type="range"
        className="threshold-range-input"
        min={THRESHOLD_MIN}
        max={THRESHOLD_MAX}
        step={1}
        value={opaque}
        aria-label={t.opaqueThreshold}
        title={t.opaqueThresholdTitle}
        onChange={(e) => onChange(transparent, Math.max(Number(e.target.value), transparent))}
      />
    </div>
  );
}
