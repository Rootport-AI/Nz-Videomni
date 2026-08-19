import { useState } from "react";
import { loraThumbnailUrl } from "../../api/client";
import type { LoraEntry } from "../../api/types";

export interface LoraCardProps {
  lora: LoraEntry;
  /** From `useBaseUrl` — thumbnails are fetched directly (`<img src>`,
   * Docs/API_REFERENCE.md §3.8), bypassing the bridge entirely, so nothing
   * renders until this resolves. */
  baseUrl: string | null;
  /** IC-LoRA UI redesign (第5波), Step 6: always present now — `InventoryScreen`
   * only ever renders STYLE entries (`lora.kind === "style"`), so every card
   * is clickable (appends a prompt tag). No more disabled/Control variant. */
  onClick: () => void;
}

/** One LoRA in the Library browser's grid: thumbnail (or a placeholder when
 * `has_thumbnail` is false, or the `<img>` itself fails to load — a 404
 * `LORA_THUMBNAIL_NOT_FOUND` is entirely plausible even when the registry
 * *thinks* there's a thumbnail) and name. The kind badge is gone along with
 * the Control tab (第5波 Step 6) — every card `InventoryScreen` renders is a
 * style LoRA now, so a per-card kind label would be redundant. */
export function LoraCard({ lora, baseUrl, onClick }: LoraCardProps) {
  const [thumbnailFailed, setThumbnailFailed] = useState(false);
  const showImage = lora.has_thumbnail && baseUrl !== null && !thumbnailFailed;

  return (
    <li className="lora-card lora-card--clickable">
      <button type="button" className="lora-card-button" onClick={onClick}>
        <span className="lora-card-thumb">
          {showImage ? (
            // eslint-disable-next-line jsx-a11y/alt-text
            <img
              src={loraThumbnailUrl(baseUrl, lora.name)}
              alt={lora.name}
              onError={() => setThumbnailFailed(true)}
            />
          ) : (
            <span className="lora-card-thumb-placeholder" aria-hidden="true">
              ▦
            </span>
          )}
        </span>
        <span className="lora-card-name" title={lora.name}>
          {lora.name}
        </span>
      </button>
    </li>
  );
}
