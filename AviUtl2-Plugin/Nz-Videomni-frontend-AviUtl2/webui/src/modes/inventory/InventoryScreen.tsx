import { useCallback, useState } from "react";
import { reloadLoras as defaultReloadLoras } from "../../api/client";
import type { ApiClient } from "../../api/client";
import type { NativeBridge } from "../../bridge";
import { useStrings } from "../../i18n/LanguageContext";
import { appendLoraTag } from "../../lora/loraTags";
import { useToasts } from "../../shell/ToastContext";
import { HistorySection } from "./HistorySection";
import { LoraCard } from "./LoraCard";
import { useLoras } from "./useLoras";
import "./InventoryScreen.css";

export interface InventoryScreenProps {
  /** Owned by `AppShell`'s shared `PromptBar` — the same state Create/Chain
   * read/write, so a style-card click here shows up as a `<lora:...>` tag
   * there immediately. */
  prompt: string;
  onPromptChange: (value: string) => void;
  baseUrl: string | null;
  /** Test/integration seam threaded from `AppShell`; production omits it.
   * Unused today (Library makes no bridge calls of its own yet). */
  nativeBridge?: NativeBridge | undefined;
  /** Test-only override; production call sites omit it (see `useLoras`). */
  apiClient?: ApiClient;
  /** Test-only override for `POST /loras/reload`, since it isn't part of
   * `ApiClient` (Reload's result shape `{total,styles,controls}` doesn't
   * match any other `ApiClient` method and isn't otherwise reused). */
  reloadLoras?: () => Promise<{ total: number; styles: number; controls: number }>;
}

/** The M5 "Library" mode (task brief §2, Mock/AVIUTL2_DESIGN_BRIEF.md §11's
 * "素材棚"): the STYLE LoRA browser (`GET /loras` + thumbnails + reload)
 * above the completed-job history grid (`HistorySection`). IC-LoRA UI
 * redesign (第5波), Step 6: the Control tab is gone — it was a permanently
 * click-disabled placeholder ("coming in M6") from before the IC-LoRA panel
 * existed; now that CONTROL LoRAs have their own selection UI on Create
 * (`GenerationForm.tsx`'s `ReferenceVideoSection`), this screen is Style-only,
 * matching Gradio's own "Style browser" scope. */
export function InventoryScreen({ prompt, onPromptChange, baseUrl, apiClient, reloadLoras = defaultReloadLoras }: InventoryScreenProps) {
  const strings = useStrings();
  const { push } = useToasts();
  const lorasState = useLoras(apiClient ? { apiClient } : {});
  const [reloading, setReloading] = useState(false);

  const handleReload = useCallback(() => {
    setReloading(true);
    void (async () => {
      try {
        const result = await reloadLoras();
        push({ kind: "success", message: strings.inventory.reloadToast(result.total, result.styles, result.controls) });
        await lorasState.refresh();
      } catch (err) {
        push({ kind: "error", message: err instanceof Error ? err.message : String(err) });
      } finally {
        setReloading(false);
      }
    })();
  }, [reloadLoras, lorasState, push, strings]);

  const handleStyleClick = useCallback(
    (name: string) => {
      const next = appendLoraTag(prompt, name);
      if (next === prompt) return; // already tagged — no-op, no toast
      onPromptChange(next);
      push({ kind: "success", message: strings.inventory.addedToast(name) });
    },
    [prompt, onPromptChange, push, strings],
  );

  const loras = lorasState.status === "ready" ? lorasState.loras : [];
  const styles = loras.filter((l) => l.kind === "style");

  return (
    <div className="inventory-screen">
      <section className="inventory-section">
        <div className="inventory-section-head">
          <h2>{strings.inventory.heading}</h2>
          <button type="button" className="secondary-button" disabled={reloading} onClick={handleReload}>
            {reloading ? strings.inventory.reloading : strings.inventory.reload}
          </button>
        </div>

        {lorasState.status === "loading" && <p className="hint">{strings.inventory.loadingLoras}</p>}
        {lorasState.status === "error" && <p className="hint field-hint-error">{strings.inventory.loadError}</p>}

        {lorasState.status === "ready" &&
          (styles.length === 0 ? (
            <p className="hint">{strings.inventory.emptyLoras}</p>
          ) : (
            <ul className="lora-grid">
              {styles.map((lora) => (
                <LoraCard key={lora.name} lora={lora} baseUrl={baseUrl} onClick={() => handleStyleClick(lora.name)} />
              ))}
            </ul>
          ))}
      </section>

      <HistorySection baseUrl={baseUrl} />
    </div>
  );
}
