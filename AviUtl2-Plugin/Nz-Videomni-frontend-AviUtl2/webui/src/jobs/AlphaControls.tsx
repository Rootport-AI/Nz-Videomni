import { useCallback, useEffect, useState } from "react";
import { apiClient as defaultApiClient, BackendApiError } from "../api/client";
import type { ApiClient } from "../api/client";
import type { AlphaChromaColor, AlphaOutputResponse, AlphaThresholds, JobResponse } from "../api/types";
import { useStrings } from "../i18n/LanguageContext";
import { ALPHA_CHROMA_COLORS, DEFAULT_ALPHA_CHROMA_COLOR } from "./alphaColors";
import { THRESHOLD_MAX, THRESHOLD_MIN, ThresholdRange } from "./ThresholdRange";

/** What the card's single preview shows (Docs/ALPHAGEN_DESIGN.md §10.3). */
export type AlphaView = "matte" | "rgba" | "chroma";

/** The composite on show: the thresholds (and, for `chroma`, the color) the
 * server actually made it with — taken from the POST's answer. */
export interface AlphaShown {
  thresholds: AlphaThresholds;
  /** `null` for `rgba`. */
  color: AlphaChromaColor | null;
}

/** The thresholds the card starts on (§10.1; the server's defaults too). */
const DEFAULT_ALPHA_THRESHOLDS: AlphaThresholds = { transparent_threshold: 7, opaque_threshold: 248 };

type AlphaBusy = "rgba" | "chroma" | "download";

export interface AlphaControlsProps {
  job: JobResponse;
  baseUrl: string | null;
  apiClient?: ApiClient;
  /** The card's 🎞 is downloading / inserting: the four buttons that change
   * what is shown (✂🪣💾🔁) are disabled, so the ✅ that insert lands on always
   * belongs to the view it started from. */
  insertBusy?: boolean;
  /** Fired whenever the view (or the composite shown) changes. `JobCard` owns
   * the preview `<video>` and builds its URL from these; `shown` is `null` for
   * the matte. */
  onViewChange?: (view: AlphaView, shown: AlphaShown | null) => void;
  /** Fired when a POST starts / ends — `JobCard` disables 🎞 meanwhile. */
  onBusyChange?: (busy: boolean) => void;
}

function sameThresholds(a: AlphaThresholds | null | undefined, b: AlphaThresholds): boolean {
  return (
    a != null && a.transparent_threshold === b.transparent_threshold && a.opaque_threshold === b.opaque_threshold
  );
}

function describeError(err: unknown): { code: string; message: string } {
  if (err instanceof BackendApiError) return { code: err.code, message: err.message };
  if (err instanceof Error) return { code: "UNKNOWN", message: err.message };
  return { code: "UNKNOWN", message: String(err) };
}

function shownFrom(response: AlphaOutputResponse, color: AlphaChromaColor | null): AlphaShown {
  return {
    thresholds: {
      transparent_threshold: response.transparent_threshold,
      opaque_threshold: response.opaque_threshold,
    },
    color,
  };
}

/** A typed threshold, rounded and held inside `[lo, hi]`; `null` when it is
 * not a number at all (the field then goes back to the current value). */
function commitThreshold(text: string, lo: number, hi: number): number | null {
  const parsed = Number(text);
  if (text.trim() === "" || !Number.isFinite(parsed)) return null;
  return Math.min(hi, Math.max(lo, Math.round(parsed)));
}

/** Hands the MOV to the browser's own download (WebView2's default). A real
 * `<a>` click: `download` is not set — the API is another origin, where the
 * attribute does nothing — and the server's `Content-Disposition: attachment`
 * is what makes it a download (Docs/ALPHAGEN_DESIGN.md §10.3). */
function startDownload(url: string): void {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.rel = "noopener";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

/**
 * Alpha Gen 第 3 弾（出口）: the ✂🪣色💾🔁 row and the threshold slider on a
 * completed Alpha Gen job's card (Docs/ALPHAGEN_DESIGN.md §10.2 / §10.3). The
 * `JoinControls` arrangement: this component owns the controls, the POSTs and
 * the view state and reports the view up (`onViewChange`); `JobCard` owns the
 * one preview `<video>` (and builds its URL) and the 🎞 insert of whatever is
 * shown.
 *
 * Renders nothing unless the job is `completed` and an Alpha Gen job
 * (`job.alpha != null` — the server's answer, from the `matte.mkv` next to its
 * output). Whether an output already exists, and with which thresholds, is the
 * server's to say: every ✂ / 🪣 POSTs, and the server reuses the file when the
 * thresholds (and color) match (`reused: true`). The only no-ops are pressing
 * ✂ on the transparent view, or 🪣 on the same color's fill, when the shown
 * composite was made with the thresholds the slider holds now. Changing the
 * color or the thresholds alone never makes anything. Nothing is kept across a
 * remount: the view starts on the matte and the controls on their defaults.
 */
export function AlphaControls({
  job,
  baseUrl,
  apiClient = defaultApiClient,
  insertBusy = false,
  onViewChange,
  onBusyChange,
}: AlphaControlsProps) {
  const strings = useStrings();
  const t = strings.jobs.alpha;
  const [view, setView] = useState<AlphaView>("matte");
  const [shown, setShown] = useState<AlphaShown | null>(null);
  const [thresholds, setThresholds] = useState<AlphaThresholds>(DEFAULT_ALPHA_THRESHOLDS);
  // The two number fields hold text while typing and commit on blur (the
  // `RepeatCountField` arrangement); the slider writes both at once.
  const [transparentText, setTransparentText] = useState(String(DEFAULT_ALPHA_THRESHOLDS.transparent_threshold));
  const [opaqueText, setOpaqueText] = useState(String(DEFAULT_ALPHA_THRESHOLDS.opaque_threshold));
  const [color, setColor] = useState<AlphaChromaColor>(DEFAULT_ALPHA_CHROMA_COLOR);
  const [busy, setBusy] = useState<AlphaBusy | null>(null);
  const [error, setError] = useState<{ code: string; message: string } | null>(null);

  useEffect(() => {
    onViewChange?.(view, view === "matte" ? null : shown);
  }, [view, shown, onViewChange]);

  const isBusy = busy !== null;
  useEffect(() => {
    onBusyChange?.(isBusy);
  }, [isBusy, onBusyChange]);

  const setBoth = useCallback((transparent: number, opaque: number) => {
    setThresholds({ transparent_threshold: transparent, opaque_threshold: opaque });
    setTransparentText(String(transparent));
    setOpaqueText(String(opaque));
  }, []);

  // Transparent: 0..opaque. Opaque: transparent..255. Equal is allowed.
  const commitTransparent = useCallback(() => {
    const value = commitThreshold(transparentText, THRESHOLD_MIN, thresholds.opaque_threshold);
    setBoth(value ?? thresholds.transparent_threshold, thresholds.opaque_threshold);
  }, [transparentText, thresholds, setBoth]);

  const commitOpaque = useCallback(() => {
    const value = commitThreshold(opaqueText, thresholds.transparent_threshold, THRESHOLD_MAX);
    setBoth(thresholds.transparent_threshold, value ?? thresholds.opaque_threshold);
  }, [opaqueText, thresholds, setBoth]);

  const jobId = job.job_id;

  /** One POST at a time; on failure the view stays as it was. */
  const run = useCallback(
    (kind: AlphaBusy, post: () => Promise<AlphaOutputResponse>, onDone: (response: AlphaOutputResponse) => void) => {
      setError(null);
      setBusy(kind);
      void (async () => {
        try {
          onDone(await post());
        } catch (err) {
          setError(describeError(err));
        } finally {
          setBusy(null);
        }
      })();
    },
    [],
  );

  const handleCut = useCallback(() => {
    if (view === "rgba" && sameThresholds(shown?.thresholds, thresholds)) return;
    run(
      "rgba",
      () => apiClient.exportAlphaRgba(jobId, thresholds),
      (response) => {
        setShown(shownFrom(response, null));
        setView("rgba");
      },
    );
  }, [view, shown, thresholds, run, apiClient, jobId]);

  const handleBucket = useCallback(() => {
    if (view === "chroma" && shown?.color === color && sameThresholds(shown.thresholds, thresholds)) return;
    run(
      "chroma",
      () => apiClient.exportAlphaChroma(jobId, { color, ...thresholds }),
      (response) => {
        setShown(shownFrom(response, color));
        setView("chroma");
      },
    );
  }, [view, shown, color, thresholds, run, apiClient, jobId]);

  const handleDownload = useCallback(() => {
    if (baseUrl === null) return;
    // The thresholds ride along as a query the server ignores, so a MOV made
    // again with other thresholds is never served from an old cache entry.
    const url = (made: AlphaThresholds) =>
      `${baseUrl}/api/v1/jobs/${encodeURIComponent(jobId)}/alpha/rgba?v=${made.transparent_threshold}-${made.opaque_threshold}`;
    // Whether the MOV already exists is the server's to say (`reused`): always
    // POST first. A reuse answers in milliseconds, well inside the browser's
    // transient user activation (5 seconds in Chromium) for the click below.
    run(
      "download",
      () => apiClient.exportAlphaRgba(jobId, thresholds),
      (response) => {
        const made = shownFrom(response, null);
        startDownload(url(made.thresholds));
        setShown(made);
        setView("rgba");
      },
    );
  }, [baseUrl, thresholds, run, apiClient, jobId]);

  const handleReset = useCallback(() => {
    setError(null);
    setView("matte");
    setShown(null);
  }, []);

  if (job.status !== "completed" || job.alpha == null) return null;

  // 作成中 locks all five; a running 🎞 locks the four that change the view.
  const viewLocked = isBusy || insertBusy;
  // No flag: the banner is the difference itself, so moving the thumbs back
  // clears it as surely as making the composite again does.
  const thresholdsChanged = view !== "matte" && shown !== null && !sameThresholds(shown.thresholds, thresholds);

  return (
    <div className="alpha-controls">
      <div className="alpha-controls-row">
        <button
          type="button"
          className="icon-action-button"
          title={t.cut}
          aria-label={t.cut}
          disabled={viewLocked}
          onClick={handleCut}
        >
          <span aria-hidden="true">✂</span>
        </button>
        <span className="alpha-controls-gap" aria-hidden="true" />
        <button
          type="button"
          className="icon-action-button"
          title={t.bucket}
          aria-label={t.bucket}
          disabled={viewLocked}
          onClick={handleBucket}
        >
          <span aria-hidden="true">🪣</span>
        </button>
        <select
          className="field-select alpha-controls-color"
          title={t.colorLabel}
          aria-label={t.colorLabel}
          value={color}
          disabled={isBusy}
          onChange={(e) => setColor(e.target.value as AlphaChromaColor)}
        >
          {ALPHA_CHROMA_COLORS.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
        <span className="alpha-controls-gap" aria-hidden="true" />
        <button
          type="button"
          className="icon-action-button"
          title={t.download}
          aria-label={t.download}
          disabled={viewLocked || baseUrl === null}
          onClick={handleDownload}
        >
          <span aria-hidden="true">💾</span>
        </button>
        <button
          type="button"
          className="icon-action-button"
          title={t.reset}
          aria-label={t.reset}
          disabled={viewLocked}
          onClick={handleReset}
        >
          <span aria-hidden="true">🔁</span>
        </button>
      </div>
      <ThresholdRange
        transparent={thresholds.transparent_threshold}
        opaque={thresholds.opaque_threshold}
        onChange={setBoth}
      />
      <div className="alpha-controls-fields">
        <label className="field field-inline" title={t.transparentThresholdTitle}>
          <span className="field-label">{t.transparentThreshold}:</span>
          <input
            type="number"
            min={THRESHOLD_MIN}
            max={thresholds.opaque_threshold}
            step={1}
            value={transparentText}
            onChange={(e) => setTransparentText(e.target.value)}
            onBlur={commitTransparent}
          />
        </label>
        <label className="field field-inline" title={t.opaqueThresholdTitle}>
          <span className="field-label">{t.opaqueThreshold}:</span>
          <input
            type="number"
            min={thresholds.transparent_threshold}
            max={THRESHOLD_MAX}
            step={1}
            value={opaqueText}
            onChange={(e) => setOpaqueText(e.target.value)}
            onBlur={commitOpaque}
          />
        </label>
      </div>
      {isBusy && <p className="hint">{t.busy}</p>}
      {view === "rgba" && <p className="warning-banner warning-banner-mild">{t.downloadHint}</p>}
      {thresholdsChanged && <p className="warning-banner warning-banner-mild">{t.thresholdsChanged}</p>}
      {error !== null && (
        <p className="hint field-hint-error">
          {error.code}: {error.message}
        </p>
      )}
    </div>
  );
}
