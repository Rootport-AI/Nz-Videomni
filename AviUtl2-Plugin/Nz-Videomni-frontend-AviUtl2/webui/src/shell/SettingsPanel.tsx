import { useCallback, useMemo } from "react";
import type { NativeBridge } from "../bridge";
import { createApiClient } from "../api/client";
import type { ApiClient } from "../api/client";
import { useLanguage, useStrings } from "../i18n/LanguageContext";
import type { Lang } from "../i18n/LanguageContext";
import { ModelsPanel } from "./ModelsPanel";
import { DangerZonePanel } from "./DangerZonePanel";
import { useSettings } from "./useSettings";
import { useConfig } from "../modes/single/useConfig";
import { redactConfigForDisplay } from "./redactConfig";
import { apiKeyStatus } from "./apiKeyStatus";
import { useTheme } from "./ThemeContext";
import type { Theme } from "./ThemeContext";
import { usePrefillPolicy } from "./PrefillPolicyContext";
import type { PrefillResolutionPolicy } from "./PrefillPolicyContext";
import {
  blockSwapPrefetchAvailability,
  blockSwapPrefetchEffective,
  keepResidentEffective,
  sageAvailability,
} from "./accelerationSettings";
import type { AccelerationSettings, AttentionBackend, VaeMode } from "./accelerationSettings";
import type { ServerStatusState } from "../modes/single/useServerStatus";
import "./SettingsPanel.css";

const LANGUAGE_OPTIONS: ReadonlyArray<{ lang: Lang; label: string }> = [
  { lang: "en", label: "English" },
  { lang: "ja", label: "日本語" },
];

/** Acceleration → Attention. Deliberately NOT translated (and so not built
 * from `strings`, unlike every other option list in this panel): "sdpa" and
 * "sage attention" are the implementations' own fixed technical names, and the
 * backend's Gradio UI renders the identical untranslated pair — a user
 * comparing the two UIs, a log line, or the plan's own wording all say the
 * same thing this way. */
const ATTENTION_OPTIONS: ReadonlyArray<{ backend: AttentionBackend; label: string }> = [
  { backend: "sdpa", label: "sdpa" },
  { backend: "sage", label: "sage attention" },
];

export interface SettingsPanelProps {
  onClose: () => void;
  /** Called right after a successful `settings.set` (task brief §2: "保存で
   * settings.set→成功したら/statusを即時再確認") — `AppShell` wires this to
   * `useServerStatus`'s `retry()`. */
  onSaved: () => void;
  /** Acceleration (2026-07-31): the shared settings object owned by
   * `AppShell` (D1: `AppShell` useState + props, no Context — see
   * `shell/accelerationSettings.ts`). This panel is the only place any of it
   * is edited; every submission path merely reads it. */
  acceleration: AccelerationSettings;
  onAttentionBackendChange: (value: AttentionBackend) => void;
  /** Acceleration (2026-08-01, backend §44): block-swap prefetch toggle. */
  onBlockSwapPrefetchChange: (value: boolean) => void;
  /** Acceleration (2026-08-02, backend §48): keep-model-skeleton-resident
   * toggle. Gated on block-swap prefetch since §1-10 — see the row itself. */
  onKeepResidentChange: (value: boolean) => void;
  /** Acceleration (2026-08-04, backend §51 / ledger §1-11): fused GGUF
   * dequantization kernel. An INDEPENDENT toggle — unlike keep-resident it is
   * gated on nothing at all (no `/status` capability flag, no relationship to
   * prefetch). */
  onFusedGgufDequantKernelChange: (value: boolean) => void;
  /** Acceleration (2026-08-05, backend §52): the PrunaVAED decoder choice.
   * INDEPENDENT like the fused-kernel row — no `/status` capability flag and
   * no relationship to any other row (a server without the pruned weights
   * degrades per job on its own). */
  onVaeModeChange: (value: VaeMode) => void;
  /** §1-26 (2026-09-01): whether the LOADED base model's engine refuses
   * `prune_vaed` — `AppShell` derives it from the `unsupported_features` list
   * `GET /models` publishes, never from a base-model id. Unlike sage and
   * block-swap prefetch (greyed out, since the server downgrades on its own),
   * this one HIDES the row outright: an engine that publishes the name 422s
   * the field, so there is no degraded-but-working choice left to offer. */
  vaeUnsupported: boolean;
  /** `AppShell`'s existing `useServerStatus` state (the same `/status` poll the
   * header badge reads) — the ONLY source of sage availability. No separate
   * capability hook/fetch exists on purpose; see
   * `shell/useAccelerationSettings.ts`. */
  serverStatus: ServerStatusState;
  /** Test-only override; production call sites omit it. */
  nativeBridge?: NativeBridge;
}

/** The M7b connection-settings panel: backend URL (bridge contract v4
 * `settings.get`/`settings.set`) plus the language toggle, opened from the
 * gear icon in `AppShell`'s header. A lightweight modal — no portal/focus
 * trap library, matching the rest of this app's dependency-light approach
 * (`Toasts.tsx` is the only other floating-overlay precedent). */
export function SettingsPanel({
  onClose,
  onSaved,
  acceleration,
  onAttentionBackendChange,
  onBlockSwapPrefetchChange,
  onKeepResidentChange,
  onFusedGgufDequantKernelChange,
  onVaeModeChange,
  vaeUnsupported,
  serverStatus,
  nativeBridge,
}: SettingsPanelProps) {
  const strings = useStrings();
  const { lang, setLang } = useLanguage();
  const { theme, setTheme } = useTheme();
  // N8: unlike `LANGUAGE_OPTIONS` (deliberately not translated — see its own
  // comment below), the theme toggle's own labels ("Dark"/"Light") are
  // ordinary UI copy, so this is built from `strings` instead of a
  // module-level constant.
  const themeOptions: ReadonlyArray<{ theme: Theme; label: string }> = [
    { theme: "dark", label: strings.settings.themeDark },
    { theme: "light", label: strings.settings.themeLight },
  ];
  // W1: right-click prefill size/fps policies — two independent toggles now
  // (localStorage-only, like the theme toggle). Each offers the same three
  // choices; order mirrors the spec's ①②③ reading (defaults → project →
  // material) rather than the storage/default order.
  const { sizePolicy, setSizePolicy, fpsPolicy, setFpsPolicy } = usePrefillPolicy();
  const prefillPolicyOptions: ReadonlyArray<{ policy: PrefillResolutionPolicy; label: string }> = [
    { policy: "defaults", label: strings.settings.prefillPolicyDefaults },
    { policy: "project", label: strings.settings.prefillPolicyProject },
    { policy: "material", label: strings.settings.prefillPolicyMaterial },
  ];
  // Acceleration: sage is greyed out ONLY on an explicit `sage_available:
  // false` from `GET /status`. Unknown (still checking / offline / a backend
  // older than §43) leaves it selectable — the server downgrades to sdpa by
  // itself when sage is asked for and unavailable, so this is a courtesy, not
  // a safety gate (see `sageAvailability`'s own doc comment).
  const statusBody = serverStatus.kind === "online" || serverStatus.kind === "busy" ? serverStatus.status : null;
  const sageDisabled = sageAvailability(statusBody) === false;
  // Block-swap prefetch (2026-08-01, backend §44): same "explicit false only"
  // disabling rule as sage — see `blockSwapPrefetchAvailability`'s own doc
  // comment.
  const prefetchAvailable = blockSwapPrefetchAvailability(statusBody);
  const prefetchDisabled = prefetchAvailable === false;
  // Keep-resident (§1-10, 2026-08-03): usable only while prefetch is
  // EFFECTIVELY on — the local toggle plus the same "explicit false only"
  // capability rule as the row above. While it is not, both buttons are
  // disabled and the row RENDERS Off (`keepResidentShown`), never "On but
  // greyed out": the stored choice is kept and returns on its own once
  // prefetch does. `acceleration` already arrives folded from
  // `useAccelerationSettings`; deriving it here too is what makes this panel
  // correct on its own terms (and testable without `AppShell`).
  const keepResidentUsable = blockSwapPrefetchEffective(acceleration, prefetchAvailable);
  const keepResidentShown = keepResidentEffective(acceleration, prefetchAvailable);
  const settings = useSettings(nativeBridge);
  // N11/N10: fetched once here and shared by both the spill-free table and
  // the raw /config viewer below — `useConfig()` already caches within its
  // own effect, but calling it twice would still mean two separate `GET
  // /config` requests, so both new sections read from this single call.
  const configState = useConfig();
  // Models section (S1): only build a bespoke ApiClient when a test override
  // is supplied — production call sites pass no `nativeBridge`, so
  // `ModelsPanel`/`useModels` fall back to the app-wide `apiClient`
  // singleton, same as every other bridge-backed hook in this app.
  const modelsApiClient: ApiClient | undefined = useMemo(
    () => (nativeBridge ? createApiClient(nativeBridge) : undefined),
    [nativeBridge],
  );

  const handleSave = useCallback(() => {
    void settings.submit().then(
      () => onSaved(),
      () => {
        // Already captured in `settings.save` and rendered below; nothing
        // else to do here (task brief: BAD_REQUEST is shown inline).
      },
    );
  }, [settings, onSaved]);

  const busy = settings.save.status === "saving" || settings.load.status === "loading";

  return (
    <div className="settings-overlay" role="presentation" onClick={onClose}>
      <div
        className="settings-panel"
        role="dialog"
        aria-modal="true"
        aria-label={strings.settings.title}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="settings-panel-head">
          <h2>{strings.settings.title}</h2>
          <button type="button" className="toast-dismiss" aria-label={strings.settings.closeDialogAriaLabel} onClick={onClose}>
            ×
          </button>
        </div>

        <div className="field">
          <span className="field-label">{strings.settings.languageLabel}</span>
          <div className="settings-lang-toggle" role="group" aria-label={strings.settings.languageLabel}>
            {/* Each option always shows its own language's native name (task
             * brief: "ヘッダに言語トグル（EN/JA）") — deliberately not run
             * through `useStrings()`, so a user who can't read the current
             * language can still find their own. */}
            {LANGUAGE_OPTIONS.map(({ lang: option, label }) => (
              <button
                key={option}
                type="button"
                className={`mode-tab${lang === option ? " mode-tab--active" : ""}`}
                aria-pressed={lang === option}
                onClick={() => setLang(option)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="field">
          <span className="field-label">{strings.settings.themeLabel}</span>
          <div className="settings-lang-toggle" role="group" aria-label={strings.settings.themeLabel}>
            {themeOptions.map(({ theme: option, label }) => (
              <button
                key={option}
                type="button"
                className={`mode-tab${theme === option ? " mode-tab--active" : ""}`}
                aria-pressed={theme === option}
                onClick={() => setTheme(option)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* X6: a shared heading over both the size and fps policy toggles —
         * they are two facets of the one "right-click menu" prefill choice.
         * Y2: `settings-group-heading` (SettingsPanel.css) pulls it closer to
         * its own group below than to THEME above — this bare span sits
         * directly in `.settings-panel`'s flex column, so plain `gap`
         * treated it as equidistant from both neighbors. */}
        <span className="field-label settings-group-heading">{strings.settings.rightClickMenuHeading}</span>

        <div className="field">
          <span className="field-label">{strings.settings.prefillSizePolicyLabel}</span>
          <div className="settings-lang-toggle" role="group" aria-label={strings.settings.prefillSizePolicyLabel}>
            {/* Size axis: all three choices stay active (X1 only retired the fps
             * axis's material option). */}
            {prefillPolicyOptions.map(({ policy: option, label }) => (
              <button
                key={option}
                type="button"
                className={`mode-tab${sizePolicy === option ? " mode-tab--active" : ""}`}
                aria-pressed={sizePolicy === option}
                title={option === "defaults" ? strings.settings.prefillPolicyDevTooltip : undefined}
                onClick={() => setSizePolicy(option)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="field">
          <span className="field-label">{strings.settings.prefillFpsPolicyLabel}</span>
          <div className="settings-lang-toggle" role="group" aria-label={strings.settings.prefillFpsPolicyLabel}>
            {/* Fps axis: X1 disables the "materials" button (the SDK can't read a
             * material's real fps yet), leaving Dev/project selectable. */}
            {prefillPolicyOptions.map(({ policy: option, label }) => {
              const isMaterial = option === "material";
              return (
                <button
                  key={option}
                  type="button"
                  className={`mode-tab${fpsPolicy === option ? " mode-tab--active" : ""}`}
                  aria-pressed={fpsPolicy === option}
                  disabled={isMaterial}
                  title={
                    isMaterial
                      ? strings.settings.prefillFpsMaterialDisabledTooltip
                      : option === "defaults"
                        ? strings.settings.prefillPolicyDevTooltip
                        : undefined
                  }
                  onClick={() => setFpsPolicy(option)}
                >
                  {label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Directly after the right-click menu group and BEFORE Acceleration
         * (owner instruction 2026-09-01: move Models up here). It used to sit
         * below the Save/Close actions, away from the connection settings'
         * primary button; the owner asked for the base-model controls to be
         * reachable without scrolling past every acceleration row instead. */}
        <ModelsPanel {...(modelsApiClient !== undefined ? { apiClient: modelsApiClient } : {})} />

        {/* Acceleration (2026-07-31, backend §43; block-swap prefetch added
            2026-08-01, backend §44; keep-resident added 2026-08-02, backend
            §48; fused GGUF dequant kernel added 2026-08-04, backend §51;
            PrunaVAED added 2026-08-05, backend §52). Five rows under one
            heading, ALL of them REAL — the VAE row was the last mock and its
            buttons are live as of 2026-08-05. Nothing here renders a
            live-looking control for a no-op, which is the discipline the
            section has followed since it was written. */}
        <span className="field-label settings-group-heading">{strings.settings.accelerationHeading}</span>

        {/* Fused GGUF dequantization kernel (2026-08-04, backend §51). Keeps
            the position the old `fused_gguf_dequant_gemm` MOCK held at the
            head of this section — that mock was removed outright (owner
            ruling 2026-08-04) and this REAL toggle took its place. Same
            two-button shape as prefetch/keep-resident, but with NO gate at
            all: the backend publishes no availability flag for it (same
            discipline as `keep_resident`) and it depends on no other row. */}
        <div className="field">
          <span className="field-label">{strings.settings.accelFusedGgufKernelLabel}</span>
          <div className="settings-lang-toggle" role="group" aria-label={strings.settings.accelFusedGgufKernelLabel}>
            {[
              { on: true, label: strings.settings.accelFusedGgufKernelOn },
              { on: false, label: strings.settings.accelFusedGgufKernelOff },
            ].map(({ on, label }) => (
              <button
                key={label}
                type="button"
                className={`mode-tab${acceleration.fusedGgufDequantKernel === on ? " mode-tab--active" : ""}`}
                aria-pressed={acceleration.fusedGgufDequantKernel === on}
                onClick={() => onFusedGgufDequantKernelChange(on)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        {/* Shown only while it is ON, like the prefetch/keep-resident notes —
            and like them it is reassurance, not a warning: the output is
            bit-identical either way. */}
        {acceleration.fusedGgufDequantKernel && (
          <p className="field-hint">{strings.settings.accelFusedGgufKernelNote}</p>
        )}

        <div className="field">
          <span className="field-label">{strings.settings.accelAttentionLabel}</span>
          <div className="settings-lang-toggle" role="group" aria-label={strings.settings.accelAttentionLabel}>
            {ATTENTION_OPTIONS.map(({ backend, label }) => {
              const isSage = backend === "sage";
              return (
                <button
                  key={backend}
                  type="button"
                  className={`mode-tab${acceleration.attentionBackend === backend ? " mode-tab--active" : ""}`}
                  aria-pressed={acceleration.attentionBackend === backend}
                  disabled={isSage && sageDisabled}
                  title={isSage && sageDisabled ? strings.settings.accelSageUnavailableTooltip : undefined}
                  onClick={() => onAttentionBackendChange(backend)}
                >
                  {label}
                </button>
              );
            })}
          </div>
        </div>
        {/* The one piece of copy the owner asked to be unmissable: sage is not
            bit-identical to sdpa, so a fixed seed no longer reproduces the
            exact same frames. Shown only while sage is actually selected. */}
        {acceleration.attentionBackend === "sage" && <p className="field-hint">{strings.settings.accelSageNote}</p>}

        <div className="field">
          <span className="field-label">{strings.settings.accelPrefetchLabel}</span>
          <div className="settings-lang-toggle" role="group" aria-label={strings.settings.accelPrefetchLabel}>
            {[
              { on: true, label: strings.settings.accelPrefetchOn },
              { on: false, label: strings.settings.accelPrefetchOff },
            ].map(({ on, label }) => (
              <button
                key={label}
                type="button"
                className={`mode-tab${acceleration.blockSwapPrefetch === on ? " mode-tab--active" : ""}`}
                aria-pressed={acceleration.blockSwapPrefetch === on}
                disabled={on && prefetchDisabled}
                title={on && prefetchDisabled ? strings.settings.accelPrefetchUnavailableTooltip : undefined}
                onClick={() => onBlockSwapPrefetchChange(on)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        {/* Unlike sage's note above, this one is not a warning — the output
            is bit-identical either way, only the transfer method changes. */}
        {acceleration.blockSwapPrefetch && <p className="field-hint">{strings.settings.accelPrefetchNote}</p>}

        {/* Keep-resident (2026-08-02, backend §48). Same two-button shape as
            the prefetch row above. §1-10 (2026-08-03) added the dependency on
            prefetch: BOTH buttons go disabled while prefetch is effectively
            off (the backend's guard would silently turn keep-resident off
            anyway, so offering the choice there was the "the UI says it's on"
            trap in miniature). Unlike the prefetch row's own gate, the OFF
            button is disabled too — there is nothing left to choose. */}
        <div className="field">
          <span className="field-label">{strings.settings.accelKeepResidentLabel}</span>
          <div className="settings-lang-toggle" role="group" aria-label={strings.settings.accelKeepResidentLabel}>
            {[
              { on: true, label: strings.settings.accelKeepResidentOn },
              { on: false, label: strings.settings.accelKeepResidentOff },
            ].map(({ on, label }) => (
              <button
                key={label}
                type="button"
                className={`mode-tab${keepResidentShown === on ? " mode-tab--active" : ""}`}
                aria-pressed={keepResidentShown === on}
                disabled={!keepResidentUsable}
                title={!keepResidentUsable ? strings.settings.accelKeepResidentPrefetchOffTooltip : undefined}
                onClick={() => onKeepResidentChange(on)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        {/* Shown only while it is ON — the RAM cost is the thing the reader
            needs in front of them at that moment. Not a warning about output
            quality: like prefetch, the frames are bit-identical either way. */}
        {keepResidentShown && <p className="field-hint">{strings.settings.accelKeepResidentNote}</p>}

        {/* PrunaVAED (2026-08-05, backend §52). Same two-button shape as the
            fused-kernel row above and, like it, gated on no `/status`
            capability flag: the backend publishes none and instead degrades per
            job (`vae_mode_used: "on->off"`) when the pruned weights are absent.

            §1-26 (2026-09-01): the row and its note are HIDDEN OUTRIGHT — not
            greyed — on an engine whose published `unsupported_features` name
            `prune_vaed`, because there the field is a hard 422 rather than a
            silent downgrade. Both are wrapped by the ONE condition on purpose:
            they are siblings, and condition-ing only the row would leave the
            note stranded. */}
        {!vaeUnsupported && (
          <>
            <div className="field">
              <span className="field-label">{strings.settings.accelVaeLabel}</span>
              <div className="settings-lang-toggle" role="group" aria-label={strings.settings.accelVaeLabel}>
                {[
                  { mode: "default" as const, label: strings.settings.accelVaeDefault },
                  { mode: "prune_vaed" as const, label: strings.settings.accelVaePruneVaed },
                ].map(({ mode, label }) => (
                  <button
                    key={mode}
                    type="button"
                    className={`mode-tab${acceleration.vaeMode === mode ? " mode-tab--active" : ""}`}
                    aria-pressed={acceleration.vaeMode === mode}
                    onClick={() => onVaeModeChange(mode)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
            {/* Shown only while PrunaVAED is selected. Unlike the three
                bit-identical rows above, this note is a WARNING in the same
                spirit as sage's: a pruned decoder is a different decoder, so
                the output can differ slightly. */}
            {acceleration.vaeMode === "prune_vaed" && <p className="field-hint">{strings.settings.accelVaeNote}</p>}
          </>
        )}

        <label className="field">
          <span className="field-label">{strings.settings.backendUrlLabel}</span>
          <input
            type="text"
            className="settings-url-input"
            value={settings.baseUrlInput}
            placeholder={strings.settings.backendUrlPlaceholder}
            disabled={settings.load.status === "loading"}
            onChange={(e) => settings.setBaseUrlInput(e.target.value)}
          />
        </label>
        {settings.load.status === "loading" && <p className="hint">{strings.settings.loadingCurrent}</p>}
        {settings.load.status === "error" && <p className="field-hint field-hint-error">{strings.settings.loadError}</p>}
        {settings.save.status === "error" && (
          <p className="field-hint field-hint-error">
            {settings.save.code === "BAD_REQUEST" ? strings.settings.invalidUrl : settings.save.message}
          </p>
        )}

        <div className="settings-panel-actions">
          <button type="button" className="secondary-button" onClick={onClose}>
            {strings.settings.close}
          </button>
          <button type="button" className="primary-button" disabled={busy} onClick={handleSave}>
            {settings.save.status === "saving" ? strings.settings.saving : strings.settings.save}
          </button>
        </div>

        {/* N11: spill-free frame-count table, and N10: raw /config JSON
         * viewer. Both new sections share the single `configState` fetched
         * above — mirrors `ModelsPanel`'s own "independent section" placement
         * (task brief: settings has no config today, so this is the panel's
         * first consumer of it). */}
        {configState.status === "loading" && <p className="hint">{strings.settings.configLoading}</p>}
        {configState.status === "ready" && (
          <>
            <div className="models-section">
              <h3>{strings.settings.spillFreeSectionTitle}</h3>
              <p className="field-hint">{strings.settings.spillFreeHint}</p>
              <table className="spill-free-table">
                <thead>
                  <tr>
                    <th>{strings.settings.spillFreeResolutionHeader}</th>
                    <th>{strings.settings.spillFreeFramesHeader}</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(configState.config.limits.spill_free_frames).map(([resolution, frames]) => (
                    <tr key={resolution}>
                      <td>{resolution}</td>
                      <td>{frames}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="models-section">
              <h3>{strings.settings.apiKeySectionTitle}</h3>
              <p className="field-hint">{strings.settings.apiKeyHint}</p>
              {/* N13: presence/absence only — the key's value must never be
               * rendered (see `apiKeyStatus.ts`'s doc comment). */}
              {(() => {
                const status = apiKeyStatus(configState.config, configState.usingFallback);
                if (status === "set") {
                  return <span className="badge badge-connected">{strings.settings.apiKeySet}</span>;
                }
                if (status === "unset") {
                  return <span className="badge badge-neutral">{strings.settings.apiKeyUnset}</span>;
                }
                return <span className="badge badge-checking">{strings.settings.apiKeyUnknown}</span>;
              })()}
            </div>

            <div className="models-section">
              <h3>{strings.settings.rawConfigSectionTitle}</h3>
              {configState.usingFallback && (
                <p className="field-hint field-hint-error">{strings.settings.rawConfigFallbackNote}</p>
              )}
              {/* N10 follow-up: secrets like `server.api_key` must never hit
               * the screen in plaintext — display a redacted copy, never
               * `configState.config` directly (see `redactConfig.ts`). */}
              <pre className="raw-config-json">{JSON.stringify(redactConfigForDisplay(configState.config), null, 2)}</pre>
            </div>
          </>
        )}

        {/* N4: "danger zone" — pipeline unload + bulk purge of terminal jobs.
         * Placed last on purpose: these are slow-to-notice, consequential
         * actions that must never be reachable by whatever muscle-memory click
         * lands on the connection settings' primary button. (`ModelsPanel`
         * shared that reasoning until 2026-09-01, when the owner moved it up
         * above Acceleration.) Reuses the same apiClient instance (test
         * override only; production omits it and both fall back to the
         * app-wide singleton). */}
        <DangerZonePanel {...(modelsApiClient !== undefined ? { apiClient: modelsApiClient } : {})} />
      </div>
    </div>
  );
}
