import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createMockBridge } from "../bridge/mockBridge";
import { LanguageProvider } from "../i18n/LanguageContext";
import { LANG_STORAGE_KEY } from "../i18n/strings";
import { JobsProvider } from "../jobs/JobsContext";
import { SettingsPanel } from "./SettingsPanel";
import { ToastProvider } from "./ToastContext";
import { ThemeProvider } from "./ThemeContext";
import {
  PrefillPolicyProvider,
  PREFILL_SIZE_POLICY_STORAGE_KEY,
  PREFILL_FPS_POLICY_STORAGE_KEY,
  readStoredSizePolicy,
  readStoredFpsPolicy,
} from "./PrefillPolicyContext";
import { ACCELERATION_STORAGE_KEY, readStoredAcceleration } from "./accelerationSettings";
import { useAccelerationSettings } from "./useAccelerationSettings";
import type { ServerStatusState } from "../modes/single/useServerStatus";
import type { StatusResponse } from "../api/types";

/** A minimal `GET /status` body — only the acceleration block matters for
 * these tests, but `StatusResponse`'s required fields still have to be there. */
function statusWithSage(sageAvailable: boolean | undefined): StatusResponse {
  return {
    server: "running",
    version: "0.4.0-test",
    host: "127.0.0.1",
    port: 18620,
    pipeline_loaded: false,
    pipeline_type: null,
    gpu: { available: true, name: "test", vram_total_mb: 16000, vram_used_mb: 0, vram_free_mb: 16000 },
    queue: { mode: "single_job_in_memory", pending: 0, running: 0, completed: 0, failed: 0 },
    // `undefined` models a backend older than §43 (no acceleration block at
    // all) — the "unknown" case, which must NOT disable the sage button.
    ...(sageAvailable === undefined
      ? {}
      : { acceleration: { attention_backends: ["sdpa", "sage"], sage_available: sageAvailable } }),
  };
}

/** Same shape as {@link statusWithSage}, for the block-swap prefetch
 * availability flag (backend §44). `undefined` models a backend older than
 * §44 with no `block_swap_prefetch_available` field at all (still "unknown",
 * must NOT disable the On button). */
function statusWithPrefetch(prefetchAvailable: boolean | undefined): StatusResponse {
  return {
    server: "running",
    version: "0.4.0-test",
    host: "127.0.0.1",
    port: 18620,
    pipeline_loaded: false,
    pipeline_type: null,
    gpu: { available: true, name: "test", vram_total_mb: 16000, vram_used_mb: 0, vram_free_mb: 16000 },
    queue: { mode: "single_job_in_memory", pending: 0, running: 0, completed: 0, failed: 0 },
    ...(prefetchAvailable === undefined
      ? {}
      : { acceleration: { sage_available: true, block_swap_prefetch_available: prefetchAvailable } }),
  };
}

/** Acceleration is owned by `AppShell` in production (D1: useState + props, no
 * Context), so this direct-render test owns it the same way — a tiny harness
 * that calls the REAL `useAccelerationSettings`, which is what makes the
 * localStorage-persistence assertion below end-to-end rather than a mock. */
function Harness({
  bridge,
  onSaved,
  onClose,
  serverStatus,
}: {
  bridge: ReturnType<typeof createMockBridge>;
  onSaved: () => void;
  onClose: () => void;
  serverStatus: ServerStatusState;
}) {
  const accelerationControls = useAccelerationSettings();
  return (
    <SettingsPanel
      onClose={onClose}
      onSaved={onSaved}
      nativeBridge={bridge}
      acceleration={accelerationControls.acceleration}
      onAttentionBackendChange={accelerationControls.setAttentionBackend}
      onBlockSwapPrefetchChange={accelerationControls.setBlockSwapPrefetch}
      onKeepResidentChange={accelerationControls.setKeepResident}
      onFusedGgufDequantKernelChange={accelerationControls.setFusedGgufDequantKernel}
      onVaeModeChange={accelerationControls.setVaeMode}
      // backend Docs/PENDING_TASKS_CLOSED.md's old §1-26 (closed 2026-09-01): the
      // PrunaVAED row is hidden only on an engine that publishes `prune_vaed`;
      // these tests render the panel directly, with no base model in play, so
      // they take the ordinary "supported" case. The hiding itself is covered
      // end-to-end through the fixture server in `AppShell.featureScope.test.tsx`.
      vaeUnsupported={false}
      onKeepResidentEmbeddingsChange={accelerationControls.setKeepResidentEmbeddings}
      // 台帳 §3-114: the mirror image of `vaeUnsupported` above — this row is
      // hidden on an engine that publishes `keep_resident_embeddings`, which is
      // LTX 2.3 rather than LTX 2.5. Same reasoning applies here: these tests
      // render the panel directly with no base model in play, so they take the
      // ordinary "supported" case, and the hiding itself is covered end-to-end
      // through the fixture server in `AppShell.featureScope.test.tsx`.
      keepResidentEmbeddingsUnsupported={false}
      serverStatus={serverStatus}
    />
  );
}

function renderPanel(
  bridge: ReturnType<typeof createMockBridge>,
  onSaved = vi.fn(),
  onClose = vi.fn(),
  serverStatus: ServerStatusState = { kind: "online", status: statusWithSage(true) },
) {
  render(
    // N8: `SettingsPanel` now also calls `useTheme()` (theme toggle, next to
    // the language toggle) — `ThemeProvider` must wrap it here since this
    // test renders `SettingsPanel` directly instead of through `AppShell`.
    // W7: it also calls `usePrefillPolicy()`, so `PrefillPolicyProvider` is
    // needed here for the same reason.
    <ThemeProvider>
      <LanguageProvider>
        <PrefillPolicyProvider>
          {/* `ModelsPanel` (rendered inside `SettingsPanel`) reads
              `useJobsContext()` directly — it is one of the two issuers of
              `POST /pipeline/load`, so it takes the flag from its owner rather
              than through two layers of props. `JobsProvider` needs a
              `ToastProvider` above it for the job-settled toasts. */}
          <ToastProvider>
            <JobsProvider nativeBridge={bridge}>
              <Harness bridge={bridge} onSaved={onSaved} onClose={onClose} serverStatus={serverStatus} />
            </JobsProvider>
          </ToastProvider>
        </PrefillPolicyProvider>
      </LanguageProvider>
    </ThemeProvider>,
  );
  return { onSaved, onClose };
}

describe("SettingsPanel", () => {
  beforeEach(() => {
    // Keep the language toggle deterministic across tests (defaults to English).
    window.localStorage.removeItem(LANG_STORAGE_KEY);
    window.localStorage.removeItem(PREFILL_SIZE_POLICY_STORAGE_KEY);
    window.localStorage.removeItem(PREFILL_FPS_POLICY_STORAGE_KEY);
    // Acceleration (2026-07-31): the attention backend PERSISTS, so a leftover
    // key from a previous test would otherwise decide the next test's default.
    window.localStorage.removeItem(ACCELERATION_STORAGE_KEY);
  });

  it("shows the current backend URL from settings.get", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    renderPanel(bridge);

    const input = await screen.findByDisplayValue("http://127.0.0.1:18620");
    expect(input).toBeInTheDocument();
  });

  it("saves a new backend URL and calls onSaved (task brief: re-check /status immediately)", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    const user = userEvent.setup();
    const { onSaved } = renderPanel(bridge);

    const input = await screen.findByDisplayValue("http://127.0.0.1:18620");
    await user.clear(input);
    await user.type(input, "http://192.168.1.30:18620");

    const saveButton = screen.getByRole("button", { name: /^save$/i });
    await user.click(saveButton);

    await waitFor(() => {
      expect(onSaved).toHaveBeenCalledTimes(1);
    });

    await expect(bridge.request("settings.get", {})).resolves.toEqual({ baseUrl: "http://192.168.1.30:18620" });
  });

  it("shows an inline error for BAD_REQUEST and does not call onSaved", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    const user = userEvent.setup();
    const { onSaved } = renderPanel(bridge);

    const input = await screen.findByDisplayValue("http://127.0.0.1:18620");
    await user.clear(input);
    await user.type(input, "not a url");

    const saveButton = screen.getByRole("button", { name: /^save$/i });
    await user.click(saveButton);

    expect(await screen.findByText(/invalid backend url/i)).toBeInTheDocument();
    expect(onSaved).not.toHaveBeenCalled();

    // The backend URL itself must not have been clobbered by the failed save.
    await expect(bridge.request("settings.get", {})).resolves.toEqual({ baseUrl: "http://127.0.0.1:18620" });
  });

  it("toggling the language button switches the panel's own copy to Japanese", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    const user = userEvent.setup();
    renderPanel(bridge);

    await screen.findByDisplayValue("http://127.0.0.1:18620");
    const japaneseButton = screen.getByRole("button", { name: "日本語" });
    await user.click(japaneseButton);

    expect(await screen.findByRole("button", { name: "保存" })).toBeInTheDocument();
  });

  it("shows the size/fps policy groups under the shared right-click heading, defaulting size=material, fps=project", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    renderPanel(bridge);

    await screen.findByDisplayValue("http://127.0.0.1:18620");

    // X6: the shared heading over both policy axes renders.
    expect(screen.getByText("Right-click menu:")).toBeInTheDocument();

    const sizeGroup = within(screen.getByRole("group", { name: "Match Gen video size to the..." }));
    const fpsGroup = within(screen.getByRole("group", { name: "Match Gen video FPS to the..." }));

    // The renamed "Dev" (defaults) label appears in both groups.
    expect(sizeGroup.getByRole("button", { name: "Dev" })).toBeInTheDocument();
    expect(fpsGroup.getByRole("button", { name: "Dev" })).toBeInTheDocument();

    // Size default is "material"; fps default is "project" — each group's own
    // pressed button reflects its own axis, independently.
    expect(sizeGroup.getByRole("button", { name: "materials" })).toHaveAttribute("aria-pressed", "true");
    expect(sizeGroup.getByRole("button", { name: "project" })).toHaveAttribute("aria-pressed", "false");
    expect(fpsGroup.getByRole("button", { name: "project" })).toHaveAttribute("aria-pressed", "true");
    expect(fpsGroup.getByRole("button", { name: "materials" })).toHaveAttribute("aria-pressed", "false");

    // §3-13: BOTH axes' "materials" buttons are active. The fps one was disabled
    // under X1 while native could not read a material's fps; contract v11's
    // `mediaFps` gives it a real value, so the two axes are the same shape now.
    expect(fpsGroup.getByRole("button", { name: "materials" })).toBeEnabled();
    expect(sizeGroup.getByRole("button", { name: "materials" })).toBeEnabled();
  });

  it("selecting each prefill policy persists it to its own localStorage key", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    const user = userEvent.setup();
    renderPanel(bridge);

    await screen.findByDisplayValue("http://127.0.0.1:18620");
    const sizeGroup = within(screen.getByRole("group", { name: "Match Gen video size to the..." }));
    const fpsGroup = within(screen.getByRole("group", { name: "Match Gen video FPS to the..." }));

    // Change the size axis to "project"...
    await user.click(sizeGroup.getByRole("button", { name: "project" }));
    await waitFor(() => {
      expect(sizeGroup.getByRole("button", { name: "project" })).toHaveAttribute("aria-pressed", "true");
    });
    // ...and the fps axis to "Dev" (defaults).
    await user.click(fpsGroup.getByRole("button", { name: "Dev" }));
    await waitFor(() => {
      expect(fpsGroup.getByRole("button", { name: "Dev" })).toHaveAttribute("aria-pressed", "true");
    });

    expect(window.localStorage.getItem(PREFILL_SIZE_POLICY_STORAGE_KEY)).toBe("project");
    expect(window.localStorage.getItem(PREFILL_FPS_POLICY_STORAGE_KEY)).toBe("defaults");
    expect(readStoredSizePolicy()).toBe("project");
    expect(readStoredFpsPolicy()).toBe("defaults");

    // §3-13: the fps axis's "materials" is a real, clickable choice now — it
    // presses and persists like any other, where X1 left it inert.
    await user.click(fpsGroup.getByRole("button", { name: "materials" }));
    await waitFor(() => {
      expect(fpsGroup.getByRole("button", { name: "materials" })).toHaveAttribute("aria-pressed", "true");
    });
    expect(window.localStorage.getItem(PREFILL_FPS_POLICY_STORAGE_KEY)).toBe("material");
    expect(readStoredFpsPolicy()).toBe("material");
  });

  it("renders the Acceleration section: sdpa selected, the one remaining mock row disabled", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    renderPanel(bridge);

    await screen.findByDisplayValue("http://127.0.0.1:18620");

    expect(screen.getByText("Acceleration:")).toBeInTheDocument();

    const attention = within(screen.getByRole("group", { name: "Attention" }));
    expect(attention.getByRole("button", { name: "sdpa" })).toHaveAttribute("aria-pressed", "true");
    expect(attention.getByRole("button", { name: "sage attention" })).toHaveAttribute("aria-pressed", "false");
    // sage is available in this fixture, so the button is selectable.
    expect(attention.getByRole("button", { name: "sage attention" })).toBeEnabled();
    // The "same seed no longer reproduces the same details" note only shows
    // while sage is actually selected.
    expect(screen.queryByText(/Fine details of the output change/)).not.toBeInTheDocument();

    // §1-11: the fused-dequant row is REAL — it sits where the old mock did,
    // defaults to On since 2026-08-04 (backend §51 flipped the server default
    // after gates G1-G8), and BOTH buttons are enabled (no availability gate of
    // any kind).
    const fused = within(screen.getByRole("group", { name: "Fused GGUF Dequantization Kernel" }));
    expect(fused.getByRole("button", { name: "ON" })).toHaveAttribute("aria-pressed", "true");
    expect(fused.getByRole("button", { name: "OFF" })).toHaveAttribute("aria-pressed", "false");
    expect(fused.getByRole("button", { name: "ON" })).toBeEnabled();
    expect(fused.getByRole("button", { name: "OFF" })).toBeEnabled();
    // Its note shows while it is ON — which, like block-swap prefetch, is now
    // the resting state.
    expect(screen.getByText(/how the model's compressed weights are unpacked/)).toBeInTheDocument();

    // §52 (2026-08-05): the VAE row is REAL now — the last mock in this
    // section. It defaults to the unmodified decoder (the server default), and
    // BOTH buttons are enabled: like the fused-kernel row it has no
    // availability gate of any kind. The display name is "PrunaVAED"; the API
    // value it sends stays `"prune_vaed"`.
    const vae = within(screen.getByRole("group", { name: "VAE (video decode)" }));
    expect(vae.getByRole("button", { name: "Default" })).toHaveAttribute("aria-pressed", "true");
    expect(vae.getByRole("button", { name: "PrunaVAED" })).toHaveAttribute("aria-pressed", "false");
    expect(vae.getByRole("button", { name: "Default" })).toBeEnabled();
    expect(vae.getByRole("button", { name: "PrunaVAED" })).toBeEnabled();
    // Its quality note only shows while PrunaVAED is actually selected.
    expect(screen.queryByText(/Output quality may be slightly reduced/)).not.toBeInTheDocument();
  });

  it("selecting PrunaVAED moves aria-pressed, shows the quality note, and persists to localStorage (§52)", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    const user = userEvent.setup();
    renderPanel(bridge);

    await screen.findByDisplayValue("http://127.0.0.1:18620");
    const vae = within(screen.getByRole("group", { name: "VAE (video decode)" }));
    await user.click(vae.getByRole("button", { name: "PrunaVAED" }));

    await waitFor(() => {
      expect(vae.getByRole("button", { name: "PrunaVAED" })).toHaveAttribute("aria-pressed", "true");
    });
    expect(vae.getByRole("button", { name: "Default" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByText(/Output quality may be slightly reduced/)).toBeInTheDocument();

    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!)).toEqual({
        attentionBackend: "sdpa",
        blockSwapPrefetch: true,
        keepResident: false,
        fusedGgufDequantKernel: true,
        vaeMode: "prune_vaed",
        keepResidentEmbeddings: false,
      });
    });
    expect(readStoredAcceleration().vaeMode).toBe("prune_vaed");

    // Clicking Default again hides the note and flips aria-pressed back.
    await user.click(vae.getByRole("button", { name: "Default" }));
    await waitFor(() => {
      expect(vae.getByRole("button", { name: "Default" })).toHaveAttribute("aria-pressed", "true");
    });
    expect(screen.queryByText(/Output quality may be slightly reduced/)).not.toBeInTheDocument();
  });

  it("keeps the VAE row usable while block-swap prefetch is unavailable (no gate at all, §52)", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    // Prefetch reported unavailable — which DOES grey keep-resident out.
    renderPanel(bridge, vi.fn(), vi.fn(), { kind: "online", status: statusWithPrefetch(false) });

    await screen.findByDisplayValue("http://127.0.0.1:18620");
    const vae = within(screen.getByRole("group", { name: "VAE (video decode)" }));
    expect(vae.getByRole("button", { name: "Default" })).toBeEnabled();
    expect(vae.getByRole("button", { name: "PrunaVAED" })).toBeEnabled();
  });

  it("selecting sage moves aria-pressed, shows the seed-detail note, and persists to localStorage", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    const user = userEvent.setup();
    renderPanel(bridge);

    await screen.findByDisplayValue("http://127.0.0.1:18620");
    const attention = within(screen.getByRole("group", { name: "Attention" }));
    await user.click(attention.getByRole("button", { name: "sage attention" }));

    await waitFor(() => {
      expect(attention.getByRole("button", { name: "sage attention" })).toHaveAttribute("aria-pressed", "true");
    });
    expect(attention.getByRole("button", { name: "sdpa" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByText(/Fine details of the output change/)).toBeInTheDocument();

    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!).attentionBackend).toBe("sage");
    });
    expect(readStoredAcceleration().attentionBackend).toBe("sage");
  });

  it("disables the sage button only when /status explicitly reports sage_available:false", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    renderPanel(bridge, vi.fn(), vi.fn(), { kind: "online", status: statusWithSage(false) });

    await screen.findByDisplayValue("http://127.0.0.1:18620");
    const attention = within(screen.getByRole("group", { name: "Attention" }));
    expect(attention.getByRole("button", { name: "sage attention" })).toBeDisabled();
    expect(attention.getByRole("button", { name: "sdpa" })).toBeEnabled();
  });

  it("leaves the sage button enabled while availability is unknown (no status / old backend)", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    // No /status at all yet.
    renderPanel(bridge, vi.fn(), vi.fn(), { kind: "checking" });
    await screen.findByDisplayValue("http://127.0.0.1:18620");
    expect(
      within(screen.getByRole("group", { name: "Attention" })).getByRole("button", { name: "sage attention" }),
    ).toBeEnabled();
  });

  it("leaves the sage button enabled against a pre-§43 backend with no acceleration block", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    renderPanel(bridge, vi.fn(), vi.fn(), { kind: "online", status: statusWithSage(undefined) });
    await screen.findByDisplayValue("http://127.0.0.1:18620");
    expect(
      within(screen.getByRole("group", { name: "Attention" })).getByRole("button", { name: "sage attention" }),
    ).toBeEnabled();
  });

  it("renders the block-swap prefetch row: on selected by default (post-S4), note shown", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    renderPanel(bridge);

    await screen.findByDisplayValue("http://127.0.0.1:18620");
    const prefetch = within(screen.getByRole("group", { name: "Block-swap prefetch" }));
    expect(prefetch.getByRole("button", { name: "ON" })).toHaveAttribute("aria-pressed", "true");
    expect(prefetch.getByRole("button", { name: "OFF" })).toHaveAttribute("aria-pressed", "false");
    expect(prefetch.getByRole("button", { name: "ON" })).toBeEnabled();
    expect(prefetch.getByRole("button", { name: "OFF" })).toBeEnabled();
    expect(screen.getByText(/only the transfer method changes/)).toBeInTheDocument();
  });

  it("selecting Off for block-swap prefetch moves aria-pressed, hides the note, and persists to localStorage", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    const user = userEvent.setup();
    renderPanel(bridge);

    await screen.findByDisplayValue("http://127.0.0.1:18620");
    const prefetch = within(screen.getByRole("group", { name: "Block-swap prefetch" }));
    await user.click(prefetch.getByRole("button", { name: "OFF" }));

    await waitFor(() => {
      expect(prefetch.getByRole("button", { name: "OFF" })).toHaveAttribute("aria-pressed", "true");
    });
    expect(prefetch.getByRole("button", { name: "ON" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.queryByText(/only the transfer method changes/)).not.toBeInTheDocument();

    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!)).toEqual({
        attentionBackend: "sdpa",
        blockSwapPrefetch: false,
        keepResident: false,
        fusedGgufDequantKernel: true,
        vaeMode: "default",
        keepResidentEmbeddings: false,
      });
    });
    expect(readStoredAcceleration().blockSwapPrefetch).toBe(false);

    // Clicking On again restores the note and flips aria-pressed back.
    await user.click(prefetch.getByRole("button", { name: "ON" }));
    await waitFor(() => {
      expect(prefetch.getByRole("button", { name: "ON" })).toHaveAttribute("aria-pressed", "true");
    });
    expect(screen.getByText(/only the transfer method changes/)).toBeInTheDocument();
  });

  it("disables only the On button when /status explicitly reports block_swap_prefetch_available:false", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    renderPanel(bridge, vi.fn(), vi.fn(), { kind: "online", status: statusWithPrefetch(false) });

    await screen.findByDisplayValue("http://127.0.0.1:18620");
    const prefetch = within(screen.getByRole("group", { name: "Block-swap prefetch" }));
    expect(prefetch.getByRole("button", { name: "ON" })).toBeDisabled();
    expect(prefetch.getByRole("button", { name: "ON" })).toHaveAttribute(
      "title",
      "The server is not running with block swap, so this has no effect.",
    );
    // Off must always stay clickable — it never has a "this has no effect" state.
    expect(prefetch.getByRole("button", { name: "OFF" })).toBeEnabled();
  });

  it("leaves both block-swap prefetch buttons enabled while availability is unknown (no status / old backend)", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    renderPanel(bridge, vi.fn(), vi.fn(), { kind: "checking" });
    await screen.findByDisplayValue("http://127.0.0.1:18620");
    const prefetch = within(screen.getByRole("group", { name: "Block-swap prefetch" }));
    expect(prefetch.getByRole("button", { name: "ON" })).toBeEnabled();
    expect(prefetch.getByRole("button", { name: "OFF" })).toBeEnabled();
  });

  it("leaves both block-swap prefetch buttons enabled against a pre-§44 backend with no availability field", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    renderPanel(bridge, vi.fn(), vi.fn(), { kind: "online", status: statusWithPrefetch(undefined) });
    await screen.findByDisplayValue("http://127.0.0.1:18620");
    const prefetch = within(screen.getByRole("group", { name: "Block-swap prefetch" }));
    expect(prefetch.getByRole("button", { name: "ON" })).toBeEnabled();
    expect(prefetch.getByRole("button", { name: "OFF" })).toBeEnabled();
  });

  // Keep-resident (2026-08-02, backend §48). The mirror image of the prefetch
  // row above: the server default is OFF, so the note is HIDDEN at rest and
  // appears only once the user opts in.
  const KEEP_RESIDENT_GROUP = "Keep the model skeleton resident (cache between jobs)";
  /** §1-11 (2026-08-04): the fused GGUF dequantization kernel row's label. */
  const FUSED_KERNEL_GROUP = "Fused GGUF Dequantization Kernel";

  it("renders the keep-resident row: off selected by default, no note shown", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    renderPanel(bridge);

    await screen.findByDisplayValue("http://127.0.0.1:18620");
    const keepResident = within(screen.getByRole("group", { name: KEEP_RESIDENT_GROUP }));
    expect(keepResident.getByRole("button", { name: "OFF" })).toHaveAttribute("aria-pressed", "true");
    expect(keepResident.getByRole("button", { name: "ON" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.queryByText(/64GB or more of memory recommended/)).not.toBeInTheDocument();
  });

  it("selecting On for keep-resident moves aria-pressed, shows the note, and persists to localStorage", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    const user = userEvent.setup();
    renderPanel(bridge);

    await screen.findByDisplayValue("http://127.0.0.1:18620");
    const keepResident = within(screen.getByRole("group", { name: KEEP_RESIDENT_GROUP }));
    await user.click(keepResident.getByRole("button", { name: "ON" }));

    await waitFor(() => {
      expect(keepResident.getByRole("button", { name: "ON" })).toHaveAttribute("aria-pressed", "true");
    });
    expect(keepResident.getByRole("button", { name: "OFF" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByText(/64GB or more of memory recommended/)).toBeInTheDocument();

    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!)).toEqual({
        attentionBackend: "sdpa",
        blockSwapPrefetch: true,
        keepResident: true,
        fusedGgufDequantKernel: true,
        vaeMode: "default",
        keepResidentEmbeddings: false,
      });
    });
    expect(readStoredAcceleration().keepResident).toBe(true);

    // Clicking Off again hides the note and flips aria-pressed back.
    await user.click(keepResident.getByRole("button", { name: "OFF" }));
    await waitFor(() => {
      expect(keepResident.getByRole("button", { name: "OFF" })).toHaveAttribute("aria-pressed", "true");
    });
    expect(screen.queryByText(/64GB or more of memory recommended/)).not.toBeInTheDocument();
  });

  // §1-10 (2026-08-03): keep-resident is greyed out — and RENDERS Off, never
  // "On but greyed out" — while block-swap prefetch is effectively off.
  const KEEP_RESIDENT_PREFETCH_TOOLTIP = "Available only while block-swap prefetch is enabled.";

  it("greys keep-resident out, shows Off, and restores the stored On when the prefetch toggle comes back", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    const user = userEvent.setup();
    renderPanel(bridge);

    await screen.findByDisplayValue("http://127.0.0.1:18620");
    const keepResident = within(screen.getByRole("group", { name: KEEP_RESIDENT_GROUP }));
    const prefetch = within(screen.getByRole("group", { name: "Block-swap prefetch" }));

    await user.click(keepResident.getByRole("button", { name: "ON" }));
    await waitFor(() => {
      expect(keepResident.getByRole("button", { name: "ON" })).toHaveAttribute("aria-pressed", "true");
    });

    await user.click(prefetch.getByRole("button", { name: "OFF" }));

    await waitFor(() => {
      expect(keepResident.getByRole("button", { name: "ON" })).toBeDisabled();
    });
    expect(keepResident.getByRole("button", { name: "OFF" })).toBeDisabled();
    // Off is what the row SHOWS, matching what the request now carries.
    expect(keepResident.getByRole("button", { name: "OFF" })).toHaveAttribute("aria-pressed", "true");
    expect(keepResident.getByRole("button", { name: "ON" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.queryByText(/64GB or more of memory recommended/)).not.toBeInTheDocument();
    expect(keepResident.getByRole("button", { name: "ON" })).toHaveAttribute(
      "title",
      KEEP_RESIDENT_PREFETCH_TOOLTIP,
    );
    expect(keepResident.getByRole("button", { name: "OFF" })).toHaveAttribute(
      "title",
      KEEP_RESIDENT_PREFETCH_TOOLTIP,
    );
    // The STORED choice is kept — that is what makes the restore below
    // automatic rather than something the user has to redo.
    await waitFor(() => {
      expect(readStoredAcceleration().keepResident).toBe(true);
    });

    await user.click(prefetch.getByRole("button", { name: "ON" }));
    await waitFor(() => {
      expect(keepResident.getByRole("button", { name: "ON" })).toHaveAttribute("aria-pressed", "true");
    });
    expect(keepResident.getByRole("button", { name: "ON" })).toBeEnabled();
    expect(keepResident.getByRole("button", { name: "OFF" })).toBeEnabled();
    expect(keepResident.getByRole("button", { name: "ON" })).not.toHaveAttribute("title");
    expect(screen.getByText(/64GB or more of memory recommended/)).toBeInTheDocument();
  });

  it("greys keep-resident out when /status explicitly reports block_swap_prefetch_available:false", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    // The capability half of the rule: the local prefetch toggle is still at
    // its default (on), but the server says prefetch does nothing here.
    renderPanel(bridge, vi.fn(), vi.fn(), { kind: "online", status: statusWithPrefetch(false) });

    await screen.findByDisplayValue("http://127.0.0.1:18620");
    const keepResident = within(screen.getByRole("group", { name: KEEP_RESIDENT_GROUP }));
    expect(keepResident.getByRole("button", { name: "ON" })).toBeDisabled();
    expect(keepResident.getByRole("button", { name: "OFF" })).toBeDisabled();
    expect(keepResident.getByRole("button", { name: "OFF" })).toHaveAttribute("aria-pressed", "true");
    expect(keepResident.getByRole("button", { name: "ON" })).toHaveAttribute(
      "title",
      KEEP_RESIDENT_PREFETCH_TOOLTIP,
    );
  });

  it("leaves keep-resident clickable while prefetch availability is unknown (offline / pre-§44 backend)", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    renderPanel(bridge, vi.fn(), vi.fn(), { kind: "checking" });

    await screen.findByDisplayValue("http://127.0.0.1:18620");
    const keepResident = within(screen.getByRole("group", { name: KEEP_RESIDENT_GROUP }));
    expect(keepResident.getByRole("button", { name: "ON" })).toBeEnabled();
    expect(keepResident.getByRole("button", { name: "OFF" })).toBeEnabled();
    expect(keepResident.getByRole("button", { name: "ON" })).not.toHaveAttribute("title");
  });

  // §1-11 (2026-08-04): the fused GGUF dequantization kernel row. Unlike
  // keep-resident above it is gated on NOTHING, so there is no "greyed out"
  // case to test — only the click/persist round-trip.
  it("selecting Off for the fused dequant kernel moves aria-pressed, hides the note, and persists", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    const user = userEvent.setup();
    renderPanel(bridge);

    await screen.findByDisplayValue("http://127.0.0.1:18620");
    const fused = within(screen.getByRole("group", { name: FUSED_KERNEL_GROUP }));
    await user.click(fused.getByRole("button", { name: "OFF" }));

    await waitFor(() => {
      expect(fused.getByRole("button", { name: "OFF" })).toHaveAttribute("aria-pressed", "true");
    });
    expect(fused.getByRole("button", { name: "ON" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.queryByText(/how the model's compressed weights are unpacked/)).not.toBeInTheDocument();

    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!)).toEqual({
        attentionBackend: "sdpa",
        blockSwapPrefetch: true,
        keepResident: false,
        fusedGgufDequantKernel: false,
        vaeMode: "default",
        keepResidentEmbeddings: false,
      });
    });
    expect(readStoredAcceleration().fusedGgufDequantKernel).toBe(false);

    // Clicking On again restores the note and flips aria-pressed back.
    await user.click(fused.getByRole("button", { name: "ON" }));
    await waitFor(() => {
      expect(fused.getByRole("button", { name: "ON" })).toHaveAttribute("aria-pressed", "true");
    });
    expect(screen.getByText(/how the model's compressed weights are unpacked/)).toBeInTheDocument();
  });

  it("keeps the fused dequant kernel row usable while block-swap prefetch is off (no gate at all)", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    const user = userEvent.setup();
    // Prefetch reported unavailable — which DOES grey keep-resident out.
    renderPanel(bridge, vi.fn(), vi.fn(), { kind: "online", status: statusWithPrefetch(false) });

    await screen.findByDisplayValue("http://127.0.0.1:18620");
    expect(
      within(screen.getByRole("group", { name: KEEP_RESIDENT_GROUP })).getByRole("button", { name: "ON" }),
    ).toBeDisabled();

    const fused = within(screen.getByRole("group", { name: FUSED_KERNEL_GROUP }));
    expect(fused.getByRole("button", { name: "OFF" })).toBeEnabled();
    expect(fused.getByRole("button", { name: "OFF" })).not.toHaveAttribute("title");
    // Click the button that actually MOVES the row (the default is On), so a
    // silently-frozen control cannot pass this test.
    await user.click(fused.getByRole("button", { name: "OFF" }));
    await waitFor(() => {
      expect(fused.getByRole("button", { name: "OFF" })).toHaveAttribute("aria-pressed", "true");
    });
  });

  it("closing calls onClose", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    const user = userEvent.setup();
    const { onClose } = renderPanel(bridge);

    await screen.findByDisplayValue("http://127.0.0.1:18620");
    const closeButton = screen.getByRole("button", { name: /^close$/i });
    await user.click(closeButton);

    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
