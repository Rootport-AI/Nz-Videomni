import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import { apiClient } from "../../api/client";
import { bridge } from "../../bridge";
import { createMockBridge } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { JobsProvider } from "../../jobs/JobsContext";
import { ToastProvider } from "../../shell/ToastContext";
import { PrefillPolicyProvider } from "../../shell/PrefillPolicyContext";
import { ShowNoteProvider } from "../../shell/NoteArea";
import { __resetRepeatCountForTests } from "../single/repeatRun";
import { ChainedScreen } from "./ChainedScreen";

// §1-80 Repeat count, wired through a real ChainedScreen. Submissions and the
// job ledger poll both go through the app-wide `bridge` singleton (the dev
// mock); `afterEach` drains every job to a terminal state.

const CHAIN_PATH = "/api/v1/generate/chain";

function Providers({ children, nativeBridge }: { children: ReactNode; nativeBridge: ReturnType<typeof createMockBridge> }) {
  return (
    <LanguageProvider>
      <ToastProvider>
        <PrefillPolicyProvider>
          <ShowNoteProvider showNote={() => {}}>
            <JobsProvider nativeBridge={nativeBridge} intervalMs={20}>
              {children}
            </JobsProvider>
          </ShowNoteProvider>
        </PrefillPolicyProvider>
      </ToastProvider>
    </LanguageProvider>
  );
}

function renderChain() {
  const nativeBridge = createMockBridge({ delayMs: 0 });
  return render(
    <Providers nativeBridge={nativeBridge}>
      <ChainedScreen
        prompt="a cat riding a skateboard"
        baseUrl={null}
        nativeBridge={nativeBridge}
        highlightedJobId={null}
        onJobSubmitted={() => {}}
        controlLoraNames={new Set()}
      />
    </Providers>,
  );
}

function isChainCall(call: Parameters<typeof bridge.request>): boolean {
  return call[0] === "backend.request" && (call[1] as { path?: string }).path === CHAIN_PATH;
}

function generateColumn(container: HTMLElement): HTMLElement {
  const column = container.querySelector(".generation-column");
  if (!column) throw new Error(".generation-column not found");
  return column as HTMLElement;
}

function mainButton(container: HTMLElement): HTMLButtonElement {
  const button = generateColumn(container).querySelector(".generate-button");
  if (!button) throw new Error("Generate button not found");
  return button as HTMLButtonElement;
}

async function drainJobs(): Promise<void> {
  for (let i = 0; i < 50; i += 1) {
    const jobs = await apiClient.listJobs();
    if (jobs.every((j) => j.status !== "queued" && j.status !== "running")) return;
  }
}

beforeEach(() => {
  __resetRepeatCountForTests();
});

afterEach(async () => {
  vi.restoreAllMocks();
  await drainJobs();
  __resetRepeatCountForTests();
});

describe("ChainedScreen — Repeat count (§1-80)", () => {
  it(
    "N=2 sends /generate/chain twice, with Stop (1 left) and a disabled field in between",
    async () => {
      const user = userEvent.setup();
      const requestSpy = vi.spyOn(bridge, "request");
      const { container } = renderChain();
      await waitFor(
        () => expect(within(generateColumn(container)).getByRole("button", { name: /^generate$/i })).toBeEnabled(),
        { timeout: 10_000 },
      );

      const field = within(generateColumn(container)).getByLabelText("Repeat count") as HTMLInputElement;
      expect(field.value).toBe("1");
      await user.clear(field);
      await user.type(field, "2");
      await user.click(mainButton(container));

      await waitFor(() => expect(mainButton(container)).toHaveTextContent("Stop (1 left)"));
      expect(mainButton(container)).toBeEnabled();
      expect(field).toBeDisabled();

      await waitFor(() => expect(requestSpy.mock.calls.filter(isChainCall)).toHaveLength(2), { timeout: 15_000 });
      await waitFor(
        () => expect(within(generateColumn(container)).getByRole("button", { name: /^generate$/i })).toBeEnabled(),
        { timeout: 15_000 },
      );
      expect(field).toBeEnabled();
      expect(requestSpy.mock.calls.filter(isChainCall)).toHaveLength(2);
      // Neither submission was rejected.
      expect(screen.queryByText(/JOB_BUSY/)).not.toBeInTheDocument();
    },
    30_000,
  );
});
