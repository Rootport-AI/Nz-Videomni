import { useState } from "react";
import type { ApiClient } from "../../api/client";
import type { AppConfig, TransformerRole } from "../../api/types";
import type { NativeBridge } from "../../bridge";
import { useStrings } from "../../i18n/LanguageContext";
import type { ReferenceRules } from "../single/referenceRules";
import { V2vRefLongBody } from "./V2vRefLongBody";
import type { V2vRefLongChainSource } from "./types";
// The `.batch-section*` styles come in with `V2vRefLongBody`'s imports.
import { useV2vRefLongForm } from "./useV2vRefLongForm";

export interface BatchV2vRefLongSectionProps {
  config: AppConfig;
  /** The Chain screen's `useChainForm` result. */
  chain: V2vRefLongChainSource;
  /** The live `constraints.reference`. */
  rules: ReferenceRules;
  role: TransformerRole | null;
  serverBusy: boolean;
  nativeBridge?: NativeBridge | undefined;
  loraRefreshKey?: string | undefined;
  /** Test-only seams, passed through to the hook. */
  apiClient?: ApiClient | undefined;
  pollIntervalMs?: number | undefined;
  jobBusyBackoffMs?: number | undefined;
}

/**
 * The "Batch v2v-ref-long" panel (MiniMax H3, Chain screen): a collapsed
 * `<details>` whose body (and hook) is mounted only while open. Every value
 * lives in the store, so closing the panel or rebuilding the Chain screen
 * keeps it. Rendered outside `.single-layout`, below Batch i2v-long.
 */
export function BatchV2vRefLongSection(props: BatchV2vRefLongSectionProps) {
  const t = useStrings().batchV2vRefLong;
  const [open, setOpen] = useState(false);
  return (
    <details className="batch-section" onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary className="batch-section-summary">{t.heading}</summary>
      {open && (
        <div className="batch-section-body">
          <SectionBody {...props} />
        </div>
      )}
    </details>
  );
}

/** The props are the hook's deps as they are. */
function SectionBody(props: BatchV2vRefLongSectionProps) {
  const form = useV2vRefLongForm(props);
  return <V2vRefLongBody form={form} nativeBridge={props.nativeBridge} />;
}
