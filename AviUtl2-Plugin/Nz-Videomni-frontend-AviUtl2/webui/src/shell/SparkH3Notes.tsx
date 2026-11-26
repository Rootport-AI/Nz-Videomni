import { useStrings } from "../i18n/LanguageContext";

export interface SparkH3NotesProps {
  /** `shell/accelerationSettings.ts`'s `sparkH3StepsTooFew`. */
  stepsTooFew: boolean;
  /** `shell/accelerationSettings.ts`'s `sparkH3VideoTooShort`. */
  videoTooShort: boolean;
}

/**
 * Spark-H3's two notes under a Generate button (裁定 113 ③,
 * MINIMAX_H3_DESIGN §6.11) — Single, Chained and Edit's Retake / Inpainting /
 * Outpainting. The same yellow paragraphs as the LoRA and Reference notes
 * (`warning-banner warning-banner-mild`), one per sentence. Informational
 * only: the screens never count these among the reasons that disable
 * Generate. The rules are decided by the two pure functions; this only draws.
 */
export function SparkH3Notes({ stepsTooFew, videoTooShort }: SparkH3NotesProps) {
  const strings = useStrings();
  return (
    <>
      {stepsTooFew && (
        <p className="warning-banner warning-banner-mild" role="status">
          {strings.sparkH3Notes.stepsTooFew}
        </p>
      )}
      {videoTooShort && (
        <p className="warning-banner warning-banner-mild" role="status">
          {strings.sparkH3Notes.videoTooShort}
        </p>
      )}
    </>
  );
}
