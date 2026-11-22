import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useStrings } from "../../i18n/LanguageContext";
import { REFERENCE_KINDS, referenceTag } from "../single/referenceRules";
import type { ReferenceKind } from "../single/referenceRules";
import { canAddField, fieldSlot, filledPath, frameCounts, isDefaultField, maxOfKind } from "./folderRules";
import type { FolderFrame, UseV2vRefLongFormResult } from "./types";

export interface FolderInputsProps {
  form: UseV2vRefLongFormResult;
}

/**
 * Batch v2v-ref-long's folder inputs: the opening-video folder (with Joined),
 * the per-clip reference folders and the output folder. Rendering only — every
 * value and mutation is `form`'s.
 */
export function FolderInputs({ form }: FolderInputsProps) {
  const t = useStrings().batchV2vRefLong;
  const { inputs } = form.state;
  const disabled = form.inputsLocked;
  const frameCount = inputs.frames.length;
  const lastFrame = frameCount - 1;

  // An added frame lands at the strip's right end: a rise in the count scrolls
  // the strip there (before paint). A removal, the first render and a remount
  // leave the position alone.
  const stripRef = useRef<HTMLDivElement>(null);
  const prevFrameCount = useRef(frameCount);
  useLayoutEffect(() => {
    const strip = stripRef.current;
    if (strip && frameCount > prevFrameCount.current) strip.scrollLeft = strip.scrollWidth;
    prevFrameCount.current = frameCount;
  }, [frameCount]);

  return (
    <>
      <div className="field source-section">
        <PathRow
          label={t.videoDir.label}
          value={inputs.videoDir}
          disabled={disabled}
          pickTitle={t.videoDir.button}
          onPick={() => void form.pickVideoDir(t.videoDir.button)}
          onCommit={(value) => form.setVideoDir(value)}
          hint={inputs.videoDir.trim() === "" ? t.videoDir.none : null}
        />
        <label className="field field-inline">
          <input
            type="checkbox"
            checked={inputs.joined}
            disabled={disabled}
            onChange={(e) => form.setJoined(e.target.checked)}
          />
          <span>{t.joined}</span>
        </label>
      </div>

      <details className="form-accordion v2v-ref-folders">
        <summary className="form-accordion-summary">{t.refFolders.heading(form.filledFolderCount)}</summary>
        <div className="form-accordion-body">
          <div className="v2v-frames-strip" ref={stripRef}>
            {inputs.frames.map((frame, index) => (
              <FrameBox
                key={frame.id}
                form={form}
                frame={frame}
                index={index}
                isLast={index === lastFrame}
                canRemove={index === lastFrame && frameCount >= 2}
              />
            ))}
          </div>
          <div>
            <button type="button" className="secondary-button" disabled={disabled} onClick={() => form.addFrame()}>
              {t.refFolders.addFrame}
            </button>
          </div>
        </div>
      </details>
      {form.unusedFrameStart !== null && (
        <p className="warning-banner warning-banner-mild">{t.refFolders.unusedFrames(form.unusedFrameStart + 1)}</p>
      )}

      <div className="field source-section">
        <PathRow
          label={t.outDir.label}
          value={inputs.outDir}
          disabled={disabled}
          pickTitle={t.outDir.button}
          onPick={() => void form.pickOutDir(t.outDir.button)}
          onCommit={(value) => form.setOutDir(value)}
          hint={
            filledPath(inputs.outDir) === ""
              ? t.outDir.none
              : inputs.outDirIsAuto
                ? t.outDir.auto(inputs.outDir)
                : null
          }
        />
      </div>
    </>
  );
}

function FrameBox({
  form,
  frame,
  index,
  isLast,
  canRemove,
}: {
  form: UseV2vRefLongFormResult;
  frame: FolderFrame;
  index: number;
  isLast: boolean;
  canRemove: boolean;
}) {
  const all = useStrings().batchV2vRefLong;
  const t = all.refFolders;
  const { rules } = form;
  const disabled = form.inputsLocked;
  const counts = frameCounts(frame);
  const kindCount: Record<ReferenceKind, (filled: number, max: number) => string> = {
    picture: t.pictureCount,
    video: t.videoCount,
    audio: t.audioCount,
  };
  const addLabel: Record<ReferenceKind, string> = { picture: t.addPicture, video: t.addVideo, audio: t.addAudio };

  return (
    <div className="clip-card v2v-frame" data-testid={`v2v-frame-${index}`}>
      <div className="clip-card-head">
        <span className="field-label">
          {t.frameTitle(index + 1)} <span className="v2v-count">{t.frameTotal(counts.total, rules.maxTotal)}</span>
        </span>
        {canRemove && (
          <button
            type="button"
            className="icon-action-button clip-remove"
            disabled={disabled}
            title={t.removeFrame}
            aria-label={t.removeFrame}
            onClick={() => form.removeLastFrame()}
          >
            <span aria-hidden="true">❌</span>
          </button>
        )}
      </div>

      {REFERENCE_KINDS.map((kind) => (
        <div className="v2v-kind-group" key={kind} data-testid={`v2v-frame-${index}-${kind}`}>
          <span className="v2v-kind-title">{kindCount[kind](counts[kind], maxOfKind(kind, rules))}</span>
          {frame.fields
            .filter((field) => field.kind === kind)
            .map((field) => {
              const tag = referenceTag(kind, fieldSlot(frame, field.id) ?? 0);
              const name = all.table.clipFieldLabel(index + 1, tag);
              const removeName = isDefaultField(frame, field.id) ? t.clearField(name) : t.removeField(name);
              return (
                <PathRow
                  key={field.id}
                  label={tag}
                  inputName={name}
                  value={field.path}
                  disabled={disabled}
                  pickTitle={t.fieldButton}
                  pickName={`${name} ${t.fieldButton}`}
                  onPick={() => void form.pickFieldDir(index, field.id, t.fieldButton)}
                  onCommit={(value) => form.setFieldPath(index, field.id, value)}
                  removeName={removeName}
                  onRemove={() => form.removeField(index, field.id)}
                  inline
                />
              );
            })}
          <div>
            <button
              type="button"
              className="secondary-button"
              disabled={disabled || !canAddField(frame, kind, rules)}
              onClick={() => form.addField(index, kind)}
            >
              {addLabel[kind]}
            </button>
          </div>
        </div>
      ))}

      {isLast && (
        <label className="field field-inline">
          <input
            type="checkbox"
            checked={form.state.inputs.fillFollowing}
            disabled={disabled}
            onChange={(e) => form.setFillFollowing(e.target.checked)}
          />
          <span>{t.fillFollowing}</span>
        </label>
      )}
    </div>
  );
}

interface PathRowProps {
  label: string;
  value: string;
  disabled: boolean;
  pickTitle: string;
  onPick: () => void;
  onCommit: (value: string) => void;
  hint?: ReactNode;
  /** The label sits on the row's left (the reference fields). */
  inline?: boolean;
  /** Accessible names when the visible label alone is ambiguous. Default:
   * `label` and `pickTitle`. */
  inputName?: string;
  pickName?: string;
  /** ❌ after 📁 (the reference fields); `removeName` is its title and
   * accessible name. */
  onRemove?: () => void;
  removeName?: string;
}

/** "label + hand-typable path + 📁", committed on blur — the same shape as
 * `batch-i2v-long/BatchI2vLongSection.tsx`'s local `FolderRow`. */
function PathRow({
  label,
  value,
  disabled,
  pickTitle,
  onPick,
  onCommit,
  hint,
  inline = false,
  inputName = label,
  pickName = pickTitle,
  onRemove,
  removeName,
}: PathRowProps) {
  const [draft, setDraft] = useState(value);
  useEffect(() => {
    setDraft(value);
  }, [value]);

  const input = (
    <div className="folder-row">
      {inline && <span className="field-label v2v-field-label">{label}</span>}
      <input
        type="text"
        className="folder-input"
        value={draft}
        disabled={disabled}
        aria-label={inputName}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => {
          if (draft !== value) onCommit(draft);
        }}
      />
      <button
        type="button"
        className="secondary-button folder-pick"
        disabled={disabled}
        title={pickTitle}
        aria-label={pickName}
        onClick={onPick}
      >
        📁
      </button>
      {onRemove && (
        <button
          type="button"
          className="icon-action-button clip-remove"
          disabled={disabled}
          title={removeName}
          aria-label={removeName}
          onClick={onRemove}
        >
          <span aria-hidden="true">❌</span>
        </button>
      )}
    </div>
  );
  if (inline) return input;
  return (
    <div className="field">
      <span className="field-label">{label}</span>
      {input}
      {hint && <p className="field-hint">{hint}</p>}
    </div>
  );
}
