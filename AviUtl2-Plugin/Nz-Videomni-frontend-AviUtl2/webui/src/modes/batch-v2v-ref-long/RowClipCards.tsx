import { useLayoutEffect, useRef, useState } from "react";
import type { NativeBridge } from "../../bridge";
import { useStrings } from "../../i18n/LanguageContext";
import { ReferenceAccordion } from "../single/ReferenceAccordion";
import { createInitialState, summarizeReference } from "../single/referenceRules";
import { useClipReferenceApi } from "./useClipReferenceApi";
import type { PromptMode, UseV2vRefLongFormResult } from "./types";

export interface RowClipCardsProps {
  form: UseV2vRefLongFormResult;
  /** 0-based. */
  rowIndex: number;
  nativeBridge?: NativeBridge | undefined;
}

/**
 * One row's clip cards (the opened row). One Reference is open at a time: its
 * card moves to the top, full width; the others stay in the row below.
 */
export function RowClipCards({ form, rowIndex, nativeBridge }: RowClipCardsProps) {
  const [openClip, setOpenClip] = useState<number | null>(null);
  const clips = Array.from({ length: Math.max(0, form.clipCount) }, (_, i) => i);
  // A clip count that dropped below the open card closes it.
  const open = openClip !== null && openClip < form.clipCount ? openClip : null;

  // The opened card shows at the top: a switch scrolls the box back up to it
  // (before paint, so the old position is never drawn).
  const boxRef = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    if (boxRef.current) boxRef.current.scrollTop = 0;
  }, [open]);

  return (
    <div className="v2v-row-cards" ref={boxRef}>
      {open !== null && (
        <div className="v2v-row-cards-top">
          <ClipCardBox
            key={open}
            form={form}
            rowIndex={rowIndex}
            clip={open}
            nativeBridge={nativeBridge}
            expanded
            onClose={() => setOpenClip(null)}
          />
        </div>
      )}
      <div className="v2v-row-cards-strip">
        {clips
          .filter((clip) => clip !== open)
          .map((clip) => (
            <ClipCardBox key={clip} form={form} rowIndex={rowIndex} clip={clip} onOpen={() => setOpenClip(clip)} />
          ))}
      </div>
    </div>
  );
}

function ClipCardBox({
  form,
  rowIndex,
  clip,
  nativeBridge,
  expanded = false,
  onOpen,
  onClose,
}: {
  form: UseV2vRefLongFormResult;
  rowIndex: number;
  clip: number;
  nativeBridge?: NativeBridge | undefined;
  expanded?: boolean;
  onOpen?: () => void;
  onClose?: () => void;
}) {
  const t = useStrings().batchV2vRefLong;
  const disabled = form.inputsLocked;
  const resolved = form.effectiveClipOf(rowIndex, clip);
  const promptMode: PromptMode = resolved?.promptMode ?? "add";
  const ignored = form.rowViews.find((view) => view.row.index === rowIndex)?.ignoredClips.includes(clip) ?? false;
  const radioName = `v2v-clip-prompt-mode-${rowIndex}-${clip}`;

  return (
    <div
      className={`clip-card v2v-clip-card${expanded ? " v2v-clip-card-expanded" : ""}`}
      data-testid={`v2v-clip-card-${clip}`}
    >
      <span className="field-label">{t.clipCard.title(clip + 1)}</span>
      <div className="batch-radio-row">
        <span className="field-label">{t.clipCard.promptMode.label}</span>
        {(["add", "replace"] as const).map((mode) => (
          <label className="field field-inline" key={mode}>
            <input
              type="radio"
              name={radioName}
              checked={promptMode === mode}
              disabled={disabled}
              onChange={() => form.setClipPromptMode(rowIndex, clip, mode)}
            />
            <span>{t.clipCard.promptMode[mode]}</span>
          </label>
        ))}
      </div>
      <textarea
        className="clip-prompt-input"
        rows={2}
        aria-label={t.clipCard.promptInputLabel}
        value={resolved?.prompt ?? ""}
        disabled={disabled}
        onChange={(e) => form.setClipPrompt(rowIndex, clip, e.target.value)}
      />
      {expanded ? (
        <OpenReference form={form} rowIndex={rowIndex} clip={clip} nativeBridge={nativeBridge} onClose={onClose} />
      ) : (
        <ReferenceButton form={form} rowIndex={rowIndex} clip={clip} ignored={ignored} onOpen={onOpen} />
      )}
    </div>
  );
}

function ReferenceButton({
  form,
  rowIndex,
  clip,
  ignored,
  onOpen,
}: {
  form: UseV2vRefLongFormResult;
  rowIndex: number;
  clip: number;
  ignored: boolean;
  onOpen: (() => void) | undefined;
}) {
  const t = useStrings().batchV2vRefLong;
  const refs = form.effectiveClipOf(rowIndex, clip)?.refs ?? { ...createInitialState(form.rules), enabled: true };
  const filled = summarizeReference(refs, form.rules).filled.total;
  return (
    <div>
      <button type="button" className="secondary-button v2v-reference-button" onClick={onOpen}>
        <span aria-hidden="true">▶</span>
        {t.clipCard.referenceButton(filled, form.rules.maxTotal)}
        {ignored && (
          <span className="v2v-ignored-mark" title={t.ignoredClips([clip + 1])}>
            {" ⚠"}
          </span>
        )}
      </button>
    </div>
  );
}

/** The open card's accordion; mounted only while open, so the thumbnails it
 * fetches go away with it. */
function OpenReference({
  form,
  rowIndex,
  clip,
  nativeBridge,
  onClose,
}: {
  form: UseV2vRefLongFormResult;
  rowIndex: number;
  clip: number;
  nativeBridge: NativeBridge | undefined;
  onClose: (() => void) | undefined;
}) {
  const api = useClipReferenceApi(form, rowIndex, clip, nativeBridge);
  return (
    <ReferenceAccordion
      api={api}
      embedded
      defaultOpen
      openSignal={0}
      onToggle={(isOpen) => {
        if (!isOpen) onClose?.();
      }}
      onAppendTag={(tag) => form.appendClipTag(rowIndex, clip, tag)}
      nativeBridge={nativeBridge}
      disabled={form.inputsLocked}
    />
  );
}
