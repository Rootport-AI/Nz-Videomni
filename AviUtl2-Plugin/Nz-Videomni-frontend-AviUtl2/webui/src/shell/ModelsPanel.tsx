import { useStrings } from "../i18n/LanguageContext";
import type { ApiClient } from "../api/client";
import type { ModelCategory, ModelEntry } from "../api/types";
import { MODEL_DEFAULT_NAME } from "../api/types";
import { MODEL_CATEGORIES, useModels } from "./useModels";
import "./SettingsPanel.css";

export interface ModelsPanelProps {
  /** Test-only override; production call sites omit it (`useModels` falls
   * back to the app-wide `apiClient` singleton, matching `SettingsPanel`'s
   * own `nativeBridge` test-hook convention). */
  apiClient?: ApiClient;
}

/** Fallback single-choice list before `GET /models` has resolved, so the
 * dropdown never renders empty while `list.status === "loading"`. */
const PENDING_ENTRIES: ModelEntry[] = [
  { name: MODEL_DEFAULT_NAME, path: "", is_default: true, exists: true, source: "config" },
];

/** Mirrors `gradio_ui/adapters.py`'s `build_model_choices`: the injected
 * default entry gets a descriptive `"default — <filename>"` label built from
 * its `path` (display-only — the `<option>` value stays the bare NAME), and
 * any entry whose file is missing on disk gets a trailing "(file missing)"
 * annotation but stays selectable (a load against it fails server-side with
 * `MODEL_FILE_MISSING`, which the panel surfaces after the fact). */
function formatEntryLabel(entry: ModelEntry, missingSuffix: string): string {
  let label = entry.name;
  if (entry.name === MODEL_DEFAULT_NAME && entry.path) {
    const filename = entry.path.replace(/\\/g, "/").split("/").pop();
    if (filename) label = `${entry.name} — ${filename}`;
  }
  return entry.exists ? label : `${label} (${missingSuffix})`;
}

function ModelCategoryField({
  category,
  entries,
  value,
  disabled,
  missingSuffix,
  label,
  onChange,
}: {
  category: ModelCategory;
  entries: ModelEntry[];
  value: string;
  disabled: boolean;
  missingSuffix: string;
  label: string;
  onChange: (name: string) => void;
}) {
  const options = entries.length > 0 ? entries : PENDING_ENTRIES;
  return (
    <label className="field" data-category={category}>
      <span className="field-label">{label}</span>
      <select
        className="field-select"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((entry) => (
          <option key={entry.name} value={entry.name}>
            {formatEntryLabel(entry, missingSuffix)}
          </option>
        ))}
      </select>
    </label>
  );
}

/** Model management (S1): the Settings panel's "Models" section —
 * per-category (transformer/text_encoder/video_vae/audio) selection dropdown
 * + Refresh (`GET /models`, server-side rescan) + Load (`POST
 * /pipeline/load`) (Docs/API_REFERENCE.md §3.3/§3.5). Independent of the
 * connection-settings save flow in `SettingsPanel` above it — mirrors
 * `gradio_ui/ui.py`'s "Independent section" ruling for the same feature
 * ("shares NO closure or outputs with on_page_load / the top-bar load
 * buttons"). A model swap rebuilds the engine and can take several minutes,
 * so both buttons are disabled for the whole in-flight `POST` and the panel
 * shows an explicit "this can take a while" notice instead of a spinner. */
export function ModelsPanel({ apiClient }: ModelsPanelProps) {
  const strings = useStrings();
  const models = useModels({ ...(apiClient !== undefined ? { apiClient } : {}) });

  const listLoading = models.list.status === "loading";
  const load = models.load;
  const busy = load.status === "loading";

  let loadSummary: string | null = null;
  if (load.status === "done") {
    loadSummary = strings.models.loadSuccess(MODEL_CATEGORIES.map((c) => `${c}=${load.models[c]}`).join(", "));
  }

  let errorMessage: string | null = null;
  if (load.status === "error") {
    const hint = strings.models.errorHints[load.code as keyof typeof strings.models.errorHints] ?? load.message;
    errorMessage = strings.models.loadFailed(hint);
  }

  return (
    <div className="models-section">
      <h3>{strings.models.sectionTitle}</h3>
      <p className="field-hint">{strings.models.hint}</p>

      {models.list.status === "error" && (
        <p className="field-hint field-hint-error">{strings.models.fetchError}</p>
      )}

      {MODEL_CATEGORIES.map((category) => (
        <ModelCategoryField
          key={category}
          category={category}
          entries={models.list.status === "ready" ? models.list.models.categories[category].entries : []}
          value={models.selection[category]}
          disabled={busy}
          missingSuffix={strings.models.missingSuffix}
          label={strings.models.categories[category]}
          onChange={(name) => models.setSelection(category, name)}
        />
      ))}

      <div className="settings-panel-actions models-actions">
        <button
          type="button"
          className="secondary-button"
          disabled={busy || listLoading}
          onClick={() => void models.refresh()}
        >
          {strings.models.refreshButton}
        </button>
        <button
          type="button"
          className="primary-button"
          disabled={busy || listLoading}
          onClick={() => models.loadPipeline()}
        >
          {strings.models.loadButton}
        </button>
      </div>

      {busy && <p className="hint">{strings.models.loadingNotice}</p>}
      {loadSummary && <p className="field-hint">{loadSummary}</p>}
      {load.status === "busy" && <p className="field-hint field-hint-error">{strings.models.jobBusy}</p>}
      {errorMessage && <p className="field-hint field-hint-error">{errorMessage}</p>}
    </div>
  );
}
