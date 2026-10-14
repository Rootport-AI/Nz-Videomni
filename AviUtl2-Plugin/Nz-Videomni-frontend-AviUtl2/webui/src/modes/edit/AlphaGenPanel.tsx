import type { NativeBridge } from "../../bridge";
import { useStrings } from "../../i18n/LanguageContext";
import { VIDEO_PLACEHOLDER_DATA_URL } from "../../shell/thumbnailPlaceholders";
import { useFileDrop } from "../../shell/useFileDrop";
import { DurationField } from "../single/CommonGenerationFields";
import { ALPHA_GEN_MAX_FRAMES, ALPHA_GEN_MIN_FRAMES } from "./useAlphaGenForm";
import type { UseAlphaGenFormResult } from "./useAlphaGenForm";
import { OUTPAINT_VIDEO_EXTENSIONS } from "./useOutpaintForm";

/** The frame slider's comfort tick, as a `<datalist>` id of its own. Every
 * screen stays mounted, so reusing Create's `"spill-tick"` would point this
 * slider at CREATE's tick (`DurationField`'s `tickId`). */
const ALPHA_GEN_TICK_ID = "alpha-gen-comfort-tick";

/** A number with at most 3 decimals and no trailing zeros (`29.97`, `24`). */
function formatShort(value: number): string {
  return String(Number(value.toFixed(3)));
}

export interface AlphaGenPanelProps {
  /** The whole form, owned by `EditScreen` (whose generation column holds the
   * Generate button) — the `OutpaintingPanel` arrangement. */
  form: UseAlphaGenFormResult;
  /** `true` while a submit (upload + send) is in flight. */
  disabled: boolean;
  /** Test/integration seam for the drop target. */
  nativeBridge?: NativeBridge | undefined;
}

/**
 * Alpha Gen（透過用マットの作成）のパネル（Alpha Gen 第 2 弾、設計書 §9）。
 *
 * 上から: 入力動画のカード（右クリック送付・ドロップ・📁。fps と開始位置は表示
 * のみ）→ フレーム数のスライダー（9〜145・8 刻み、快適上限の線つき）→ 軽量
 * モード → 寸法の注意。プロンプト・LoRA・fps の欄は無い —— どれもサーバーが
 * 決める（fps は元動画のまま変えない）。
 */
export function AlphaGenPanel({ form, disabled, nativeBridge }: AlphaGenPanelProps) {
  const strings = useStrings();
  const t = strings.edit.alphaGen;

  const hint = [t.framesRecommend, form.seconds !== null ? t.secondsText(form.seconds.toFixed(1)) : null]
    .filter((part): part is string => part !== null)
    .join(" / ");

  return (
    <section className="edit-panel alpha-gen-panel">
      <h2>{t.heading}</h2>
      <p className="field-hint">{t.summary}</p>

      <SourceCard form={form} disabled={disabled} nativeBridge={nativeBridge} />

      <DurationField
        label={t.framesLabel}
        value={form.numFrames}
        min={ALPHA_GEN_MIN_FRAMES}
        max={ALPHA_GEN_MAX_FRAMES}
        disabled={disabled}
        onChange={form.setNumFrames}
        onCommit={(value) => form.setNumFrames(value, true)}
        hint={hint}
        spillThresholdFrames={form.comfortFrames}
        tickId={ALPHA_GEN_TICK_ID}
      />
      {form.comfortFrames !== null && (
        <p className="field-hint">{form.lightMode ? t.comfortHintLight : t.comfortHintFull}</p>
      )}

      <label className="field field-inline">
        <input
          type="checkbox"
          checked={form.lightMode}
          disabled={disabled}
          onChange={(e) => form.setLightMode(e.target.checked)}
        />
        <span className="field-label">{t.lightModeLabel}</span>
      </label>
      {form.belowLightFloor && <p className="warning-banner warning-banner-mild">{t.lightModeSmallWarning}</p>}
      {form.aboveOfficialMax && <p className="warning-banner warning-banner-mild">{t.oversizeNote}</p>}
    </section>
  );
}

/** The source card: the whole card is the drop target, with ❌ / 📁 icon
 * buttons — the look of Outpainting's card (`OutpaintingPanel.tsx`). Unlike
 * there, choosing a file does not upload it: the window goes up at Generate. */
function SourceCard({
  form,
  disabled,
  nativeBridge,
}: {
  form: UseAlphaGenFormResult;
  disabled: boolean;
  nativeBridge?: NativeBridge | undefined;
}) {
  const strings = useStrings();
  const t = strings.edit.alphaGen;
  const { source, mediaInfo } = form;
  const chooseLabel = source !== null ? t.changeButton : t.chooseButton;

  const {
    isDragOver,
    error: dropError,
    handlers,
  } = useFileDrop({
    accept: OUTPAINT_VIDEO_EXTENSIONS,
    onFile: form.attachDropped,
    disabled,
    nativeBridge,
  });

  const className = ["field", "source-section", isDragOver ? "source-section-dragover" : ""].filter(Boolean).join(" ");

  return (
    <div
      className={className}
      onDragEnter={handlers.onDragEnter}
      onDragOver={handlers.onDragOver}
      onDragLeave={handlers.onDragLeave}
      onDrop={handlers.onDrop}
    >
      <span className="field-label">{t.sourceHeading}</span>
      <div className="source-input-actions">
        <button
          type="button"
          className="icon-action-button"
          title={t.clearButton}
          aria-label={t.clearButton}
          disabled={disabled || source === null}
          onClick={form.clear}
        >
          <span aria-hidden="true">❌</span>
        </button>
        <button
          type="button"
          className="icon-action-button"
          title={chooseLabel}
          aria-label={chooseLabel}
          disabled={disabled}
          onClick={() => void form.pick()}
        >
          <span aria-hidden="true">📁</span>
        </button>
      </div>

      <div className="source-input-body">
        <div className="keyframe-thumb">
          {form.uploading ? (
            <span className="keyframe-spinner" role="status" aria-label={t.uploadingButton} />
          ) : source !== null ? (
            <img src={VIDEO_PLACEHOLDER_DATA_URL} alt={t.sourceThumbAlt} />
          ) : (
            <span className="keyframe-thumb-filename">—</span>
          )}
        </div>

        <div className="source-input-controls">
          <p className="field-hint">{t.dropHint}</p>
          {source !== null && (
            <>
              <p className="field-hint">{source.fileName}</p>
              {mediaInfo.width > 0 && (
                <p className="field-hint">
                  {strings.edit.outpainting.sourceReadout(
                    mediaInfo.width,
                    mediaInfo.height,
                    mediaInfo.durationSec.toFixed(1),
                  )}
                </p>
              )}
              <p className="field-hint">
                {t.fpsLabel}: {mediaInfo.fps > 0 ? formatShort(mediaInfo.fps) : t.fpsUnknown}
              </p>
              <p className="field-hint">
                {source.origin === "rightClick" ? t.startFromPlayback(formatShort(source.startSec)) : t.startFromHead}
              </p>
            </>
          )}
          {dropError && (
            <p className="field-hint field-hint-error">
              {dropError === "unsupported" ? strings.dnd.unsupportedVideo : strings.dnd.resolveFailed}
            </p>
          )}
          {form.uploadError && <p className="field-hint field-hint-error">{t.uploadFailed}</p>}
        </div>
      </div>
    </div>
  );
}
