import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { createApiClient } from "../../api/client";
import { createMockBridge } from "../../bridge/mockBridge";
import type { MockBridgeOptions } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { JobsProvider } from "../../jobs/JobsContext";
import { ToastProvider, useToasts } from "../../shell/ToastContext";
import { InventoryScreen } from "./InventoryScreen";

const BASE_URL = "http://127.0.0.1:18620";

function setup(overrides: MockBridgeOptions = {}) {
  const mockBridge = createMockBridge({ delayMs: 0, ...overrides });
  const apiClient = createApiClient(mockBridge);
  return { mockBridge, apiClient };
}

function Wrapper({ apiClient, children }: { apiClient: ReturnType<typeof createApiClient>; children: ReactNode }) {
  return (
    <LanguageProvider>
      <ToastProvider>
        <JobsProvider apiClient={apiClient} intervalMs={20}>
          {children}
        </JobsProvider>
      </ToastProvider>
    </LanguageProvider>
  );
}

/** Renders a small harness exposing the latest toast messages alongside the
 * screen, so tests can assert on the "added to prompt" / reload toasts
 * without threading a separate render pass. */
function renderInventory(props: {
  apiClient: ReturnType<typeof createApiClient>;
  prompt?: string;
  onPromptChange?: (value: string) => void;
}) {
  const onPromptChange = props.onPromptChange ?? vi.fn();

  function ToastPeek() {
    const { toasts } = useToasts();
    return (
      <ul data-testid="toast-peek">
        {toasts.map((t) => (
          <li key={t.id}>{t.message}</li>
        ))}
      </ul>
    );
  }

  const utils = render(
    <Wrapper apiClient={props.apiClient}>
      <InventoryScreen
        prompt={props.prompt ?? ""}
        onPromptChange={onPromptChange}
        baseUrl={BASE_URL}
        apiClient={props.apiClient}
        reloadLoras={() => props.apiClient.reloadLoras()}
      />
      <ToastPeek />
    </Wrapper>,
  );
  return { ...utils, onPromptChange };
}

describe("InventoryScreen — LoRA browser", () => {
  it("shows only the mock's style LoRAs — Control LoRAs never appear (第5波 Step 6: Control tab removed)", async () => {
    const { apiClient } = setup();
    renderInventory({ apiClient });

    expect(await screen.findByText("Pixar_Toon")).toBeInTheDocument();
    expect(screen.getByText("LTX-2.3-Henshin")).toBeInTheDocument();
    expect(screen.queryByText("canny-control")).not.toBeInTheDocument();
    expect(screen.queryByText("pose-control")).not.toBeInTheDocument();
  });

  it("renders no tablist at all (there is only one kind of card shown)", async () => {
    const { apiClient } = setup();
    renderInventory({ apiClient });

    await screen.findByText("Pixar_Toon");
    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
    expect(screen.queryByRole("tab")).not.toBeInTheDocument();
  });

  it("every card is a clickable, enabled button — no disabled/Control variant remains", async () => {
    const { apiClient } = setup();
    renderInventory({ apiClient });

    const styleButton = (await screen.findByText("Pixar_Toon")).closest("button");
    expect(styleButton).not.toBeDisabled();
  });

  it("clicking a style card appends a default-strength tag to the prompt and shows a toast", async () => {
    const { apiClient } = setup();
    const user = userEvent.setup();
    const { onPromptChange } = renderInventory({ apiClient, prompt: "a cat riding a skateboard" });

    await user.click((await screen.findByText("Pixar_Toon")).closest("button")!);

    expect(onPromptChange).toHaveBeenCalledWith("a cat riding a skateboard <lora:Pixar_Toon:1.0:1.0>");
    expect(await screen.findByText(/added <lora:pixar_toon:1\.0:1\.0>/i)).toBeInTheDocument();
  });

  it("clicking a style card that's already tagged is a no-op (no onPromptChange call)", async () => {
    const { apiClient } = setup();
    const user = userEvent.setup();
    const { onPromptChange } = renderInventory({
      apiClient,
      prompt: "a cat <lora:Pixar_Toon:0.7> riding a skateboard",
    });

    await user.click((await screen.findByText("Pixar_Toon")).closest("button")!);

    expect(onPromptChange).not.toHaveBeenCalled();
  });

  it("Reload calls POST /loras/reload, shows a summary toast, and re-fetches the list", async () => {
    const { apiClient } = setup();
    const getLorasSpy = vi.spyOn(apiClient, "getLoras");
    const user = userEvent.setup();
    renderInventory({ apiClient });

    await screen.findByText("Pixar_Toon");
    const callsBeforeReload = getLorasSpy.mock.calls.length;

    await user.click(screen.getByRole("button", { name: /^🔁Reload$/ }));

    expect(await screen.findByText(/reloaded loras: 7 total \(2 style, 5 control\)/i)).toBeInTheDocument();
    await waitFor(() => expect(getLorasSpy.mock.calls.length).toBeGreaterThan(callsBeforeReload));
  });

  it("renders a thumbnail <img> for has_thumbnail=true entries, and falls back to a placeholder on load error", async () => {
    const { apiClient } = setup();
    renderInventory({ apiClient });

    await screen.findByText("Pixar_Toon");
    const pixarCard = screen.getByText("Pixar_Toon").closest("li")!;
    const img = within(pixarCard).getByRole("img") as HTMLImageElement;
    expect(img.src).toBe(`${BASE_URL}/api/v1/loras/Pixar_Toon/thumbnail`);

    fireEvent.error(img);

    await waitFor(() => {
      expect(within(pixarCard).queryByRole("img")).not.toBeInTheDocument();
    });
    expect(within(pixarCard).getByText("▦")).toBeInTheDocument();
  });

  it("renders the placeholder directly (no <img> at all) for has_thumbnail=false entries", async () => {
    const { apiClient } = setup();
    renderInventory({ apiClient });

    const henshinCard = (await screen.findByText("LTX-2.3-Henshin")).closest("li")!;
    expect(within(henshinCard).queryByRole("img")).not.toBeInTheDocument();
    expect(within(henshinCard).getByText("▦")).toBeInTheDocument();
  });
});

describe("InventoryScreen — history", () => {
  it("shows a completed job in the history grid with a prompt excerpt and resolution/duration", async () => {
    const { apiClient } = setup({ runningPollCount: 1 });
    await apiClient.generate({
      prompt: "a cat riding a skateboard",
      width: 384,
      height: 256,
      num_frames: 17,
      frame_rate: 24,
      seed: 1,
    });

    renderInventory({ apiClient });

    expect(await screen.findByText(/a cat riding a skateboard/i, {}, { timeout: 8_000 })).toBeInTheDocument();
    expect(await screen.findByText(/384x256/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^insert$/i })).toBeInTheDocument();
  }, 10_000);

  it("shows the empty-history hint when there are no completed jobs", async () => {
    const { apiClient } = setup();
    renderInventory({ apiClient });

    expect(await screen.findByText(/no completed jobs yet/i)).toBeInTheDocument();
  });

  it("shows the in-memory/restart-clears-history note", async () => {
    const { apiClient } = setup();
    renderInventory({ apiClient });

    expect(await screen.findByText(/cleared on restart/i)).toBeInTheDocument();
  });
});
