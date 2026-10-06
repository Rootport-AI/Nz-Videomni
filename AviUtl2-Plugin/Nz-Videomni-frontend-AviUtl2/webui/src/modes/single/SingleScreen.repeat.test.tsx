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
import { __resetRepeatCountForTests } from "./repeatRun";
import { SingleScreen } from "./SingleScreen";

// §1-80 Repeat count, wired through a real SingleScreen. Submissions and the
// job ledger poll both go through the app-wide `bridge` singleton (the dev
// mock), so this file shares one simulated backend across its tests —
// `afterEach` drains every job to a terminal state before the next test.

const GENERATE_PATH = "/api/v1/generate";

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

function renderSingle() {
  const nativeBridge = createMockBridge({ delayMs: 0 });
  return render(
    <Providers nativeBridge={nativeBridge}>
      <SingleScreen
        prompt="a cat riding a skateboard"
        baseUrl={null}
        nativeBridge={nativeBridge}
        highlightedJobId={null}
        onJobSubmitted={() => {}}
        controlLora={null}
        setControlLora={() => {}}
        controlLoraNames={new Set()}
      />
    </Providers>,
  );
}

type RequestFn = typeof bridge.request;

function isGenerateCall(call: Parameters<RequestFn>): boolean {
  return call[0] === "backend.request" && (call[1] as { path?: string }).path === GENERATE_PATH;
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

async function waitForReady(container: HTMLElement): Promise<void> {
  await waitFor(
    () => {
      expect(within(generateColumn(container)).getByRole("button", { name: /^generate$/i })).toBeEnabled();
    },
    { timeout: 10_000 },
  );
}

/** Advance the shared mock backend until no job is queued/running. */
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

describe("SingleScreen — Repeat count (§1-80)", () => {
  it(
    "N=2 sends /generate twice; while one run remains the button is Stop and the field is disabled",
    async () => {
      const user = userEvent.setup();
      const requestSpy = vi.spyOn(bridge, "request");
      const { container } = renderSingle();
      await waitForReady(container);

      const field = screen.getByLabelText("Repeat count") as HTMLInputElement;
      expect(field.value).toBe("1");
      await user.clear(field);
      await user.type(field, "2");
      await user.click(mainButton(container));

      await waitFor(() => expect(mainButton(container)).toHaveTextContent("Stop (1 left)"));
      expect(mainButton(container)).toBeEnabled();
      expect(field).toBeDisabled();

      await waitFor(() => expect(requestSpy.mock.calls.filter(isGenerateCall)).toHaveLength(2), { timeout: 15_000 });
      // Nothing remains: back to the ordinary button, field editable again.
      await waitFor(() => expect(mainButton(container)).not.toHaveTextContent(/stop/i));
      expect(field).toBeEnabled();
      expect(field.value).toBe("2");

      // Both runs finish and no third one is sent.
      await waitFor(() => expect(within(generateColumn(container)).getByRole("button", { name: /^generate$/i })).toBeEnabled(), {
        timeout: 15_000,
      });
      expect(requestSpy.mock.calls.filter(isGenerateCall)).toHaveLength(2);
    },
    30_000,
  );

  it(
    "Stop ends the repeat: only the first run is sent",
    async () => {
      const user = userEvent.setup();
      const requestSpy = vi.spyOn(bridge, "request");
      const { container } = renderSingle();
      await waitForReady(container);

      const field = screen.getByLabelText("Repeat count");
      await user.clear(field);
      await user.type(field, "3");
      await user.click(mainButton(container));

      await waitFor(() => expect(mainButton(container)).toHaveTextContent("Stop (2 left)"));
      await user.click(mainButton(container));
      expect(mainButton(container)).not.toHaveTextContent(/stop/i);

      // The running job finishes on its own; nothing more is sent.
      await waitFor(() => expect(within(generateColumn(container)).getByRole("button", { name: /^generate$/i })).toBeEnabled(), {
        timeout: 15_000,
      });
      expect(requestSpy.mock.calls.filter(isGenerateCall)).toHaveLength(1);
    },
    30_000,
  );

  it(
    "a 409 on the automatic submission ends the repeat and shows the error card",
    async () => {
      const user = userEvent.setup();
      const original = bridge.request.bind(bridge) as RequestFn;
      let generateCount = 0;
      // On the 2nd /generate (the automatic one), first put another job on the
      // mock backend so the server is genuinely busy, then let the automatic
      // submission through — the mock answers 409 JOB_BUSY. Both calls are
      // started in the same tick: the mock delays every request by one equal
      // timer, so the busy job is registered right before the automatic one
      // is handled, with no ledger poll (which advances mock jobs) in between.
      const requestSpy = vi.spyOn(bridge, "request").mockImplementation((async (...args: Parameters<RequestFn>) => {
        if (isGenerateCall(args)) {
          generateCount += 1;
          if (generateCount === 2) {
            const busyJob = original(...args);
            const automatic = original(...args);
            await busyJob;
            return automatic;
          }
        }
        return original(...args);
      }) as RequestFn);
      const { container } = renderSingle();
      await waitForReady(container);

      const field = screen.getByLabelText("Repeat count");
      await user.clear(field);
      await user.type(field, "3");
      await user.click(mainButton(container));
      await waitFor(() => expect(mainButton(container)).toHaveTextContent("Stop (2 left)"));

      expect(await screen.findByText("JOB_BUSY", {}, { timeout: 15_000 })).toBeInTheDocument();
      expect(mainButton(container)).not.toHaveTextContent(/stop/i);
      expect(field).toBeEnabled();

      // The busy job finishes; the repeat stays ended.
      await waitFor(() => expect(within(generateColumn(container)).getByRole("button", { name: /^generate$/i })).toBeEnabled(), {
        timeout: 15_000,
      });
      expect(generateCount).toBe(2);
      expect(requestSpy.mock.calls.filter(isGenerateCall)).toHaveLength(2);
    },
    30_000,
  );
});
