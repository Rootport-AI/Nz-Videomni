import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import { bridge } from "./bridge";
import {
  DEFAULT_NEGATIVE_PROMPT,
  NAG_ALPHA_DEFAULT,
  NAG_SCALE_DEFAULT,
  NAG_TAU_DEFAULT,
  NAG_TEXT_STORAGE_KEY,
  NEG_METHOD_DEFAULT,
  VSF_SCALE_DEFAULT,
} from "./shell/nagSettings";

// NAG (2026-07-28) integration coverage: the shared "Negative Prompt"
// accordion lives in `AppShell` directly below `PromptBar`, OUTSIDE every mode
// tab (D1/D4) — unlike most per-mode coverage in this codebase, these queries
// run against the whole document (`screen.*`), never `within(panel())` (that
// only scopes Create/Chain's own tabpanel, which does NOT contain this
// accordion). The final two tests mirror `App.chained.test.tsx`'s bridge-capture
// idiom: `POST /generate` always goes through the app-wide singleton `bridge`
// (`api/client.ts`'s `apiClient`), independent of any injected `nativeBridge`,
// so spying on the singleton is the only way to see the actual request body.

async function renderReady() {
  render(<App />);
  await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
}

async function openNagAccordion(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByText(/^negative prompt$/i, { selector: "summary" }));
}

describe("App / NAG shared negative-prompt accordion", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });
  afterEach(() => {
    window.localStorage.clear();
  });

  it(
    "restores the persisted negative-prompt text into the accordion on mount",
    async () => {
      window.localStorage.setItem(NAG_TEXT_STORAGE_KEY, "no cats, no dogs");
      const user = userEvent.setup();
      await renderReady();
      await openNagAccordion(user);

      const textarea = screen.getByRole("textbox", { name: /negative prompt/i }) as HTMLTextAreaElement;
      expect(textarea.value).toBe("no cats, no dogs");
    },
    15_000,
  );

  it(
    "writes an edit through to localStorage immediately",
    async () => {
      const user = userEvent.setup();
      await renderReady();
      await openNagAccordion(user);
      await user.click(screen.getByRole("checkbox", { name: /non-cfg negative/i }));

      const textarea = screen.getByRole("textbox", { name: /negative prompt/i }) as HTMLTextAreaElement;
      await user.clear(textarea);
      await user.type(textarea, "blurry only");

      expect(window.localStorage.getItem(NAG_TEXT_STORAGE_KEY)).toBe("blurry only");
    },
    15_000,
  );

  it(
    "the checkbox starts OFF on every fresh mount, even with negative-prompt text left from a 'previous session'",
    async () => {
      window.localStorage.setItem(NAG_TEXT_STORAGE_KEY, "leftover text");
      const user = userEvent.setup();
      await renderReady();
      await openNagAccordion(user);

      expect(screen.getByRole("checkbox", { name: /non-cfg negative/i })).not.toBeChecked();
      expect(screen.queryByText("【🔴ON】")).not.toBeInTheDocument();
    },
    15_000,
  );

  it(
    "the accordion's ON state is shared across a Create <-> Chain tab switch (it lives outside both tabpanels)",
    async () => {
      const user = userEvent.setup();
      await renderReady();
      await openNagAccordion(user);
      await user.click(screen.getByRole("checkbox", { name: /non-cfg negative/i }));
      expect(screen.getByText("【🔴ON】")).toBeInTheDocument();

      await user.click(screen.getByRole("tab", { name: /^chained$/i }));
      await screen.findByRole("tab", { name: /^chained$/i });

      // Still on screen (and still ON) after switching tabs — the accordion is
      // a shell-level sibling of the tab panels, not inside either.
      expect(screen.getByText("【🔴ON】")).toBeInTheDocument();
      expect(screen.getByRole("checkbox", { name: /non-cfg negative/i })).toBeChecked();
    },
    20_000,
  );

  it(
    "enabled + empty negative text blocks Create's Generate button with a reason line",
    async () => {
      const user = userEvent.setup();
      await renderReady();
      await openNagAccordion(user);
      await user.click(screen.getByRole("checkbox", { name: /non-cfg negative/i }));

      const textarea = screen.getByRole("textbox", { name: /negative prompt/i }) as HTMLTextAreaElement;
      await user.clear(textarea);

      const promptBox = screen.getByPlaceholderText(/describe the video/i);
      await user.type(promptBox, "a cat riding a skateboard");

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /^generate$/i })).toBeDisabled();
      });
      // Batch silently inherits Create's nag settings (D7/owner decision §UX
      // 1) and shows its OWN warning-banner line for the identical condition,
      // so this text legitimately appears twice at once (Create's
      // `GenerateReasonsNote` + `BatchSection`'s banner) — assert presence,
      // not uniqueness.
      expect(screen.getAllByText(/fill the negative prompt, or turn off/i).length).toBeGreaterThanOrEqual(1);
    },
    15_000,
  );

  it(
    "bridge capture — NAG off: the submitted body carries none of the nag_* fields",
    async () => {
      const requestSpy = vi.spyOn(bridge, "request");
      const user = userEvent.setup();
      await renderReady();

      const promptBox = screen.getByPlaceholderText(/describe the video/i);
      await user.type(promptBox, "a cat riding a skateboard");

      await user.click(screen.getByRole("button", { name: /^generate$/i }));
      await screen.findByText(/generating/i);

      const call = requestSpy.mock.calls.find(
        (c) => c[0] === "backend.request" && (c[1] as { path?: string }).path === "/api/v1/generate",
      );
      expect(call).toBeDefined();
      const body = (call?.[1] as { body?: Record<string, unknown> }).body ?? {};
      expect(body.negative_prompt).toBeUndefined();
      expect(body.nag_enabled).toBeUndefined();
      expect(body.nag_scale).toBeUndefined();
      expect(body.nag_tau).toBeUndefined();
      expect(body.nag_alpha).toBeUndefined();
      expect(body.neg_method).toBeUndefined();
      expect(body.vsf_scale).toBeUndefined();

      requestSpy.mockRestore();
    },
    15_000,
  );

  it(
    "bridge capture — NAG on: the submitted body carries all seven fields",
    async () => {
      const requestSpy = vi.spyOn(bridge, "request");
      const user = userEvent.setup();
      await renderReady();
      await openNagAccordion(user);
      await user.click(screen.getByRole("checkbox", { name: /non-cfg negative/i }));

      const promptBox = screen.getByPlaceholderText(/describe the video/i);
      await user.type(promptBox, "a cat riding a skateboard");

      await user.click(screen.getByRole("button", { name: /^generate$/i }));
      await screen.findByText(/generating/i);

      const call = requestSpy.mock.calls.find(
        (c) => c[0] === "backend.request" && (c[1] as { path?: string }).path === "/api/v1/generate",
      );
      expect(call).toBeDefined();
      const body = (call?.[1] as { body?: Record<string, unknown> }).body ?? {};
      expect(body.negative_prompt).toBe(DEFAULT_NEGATIVE_PROMPT);
      expect(body.nag_enabled).toBe(true);
      expect(body.nag_scale).toBe(NAG_SCALE_DEFAULT);
      expect(body.nag_tau).toBe(NAG_TAU_DEFAULT);
      expect(body.nag_alpha).toBe(NAG_ALPHA_DEFAULT);
      expect(body.neg_method).toBe(NEG_METHOD_DEFAULT);
      expect(body.vsf_scale).toBe(VSF_SCALE_DEFAULT);

      requestSpy.mockRestore();
    },
    15_000,
  );

  it(
    "bridge capture — VSF method selected via its radio: the submitted body carries neg_method=\"vsf\" and vsf_scale",
    async () => {
      // Regression guard: clicking the VSF radio must actually flow through
      // `useNagSettings.setMethod` -> `AppShell` -> `nagRequestFields` -> the
      // bridge request body. Asserting against `NEG_METHOD_DEFAULT`
      // ("nag", see `bridge capture — NAG on` above) would pass even if that
      // wiring were entirely broken, since "nag" is also the untouched
      // default — this test only means something because it drives the
      // actual radio click.
      const requestSpy = vi.spyOn(bridge, "request");
      const user = userEvent.setup();
      await renderReady();
      await openNagAccordion(user);
      await user.click(screen.getByRole("radio", { name: /^vsf$/i }));
      await user.click(screen.getByRole("checkbox", { name: /non-cfg negative/i }));

      const promptBox = screen.getByPlaceholderText(/describe the video/i);
      await user.type(promptBox, "a cat riding a skateboard");

      await user.click(screen.getByRole("button", { name: /^generate$/i }));
      await screen.findByText(/generating/i);

      const call = requestSpy.mock.calls.find(
        (c) => c[0] === "backend.request" && (c[1] as { path?: string }).path === "/api/v1/generate",
      );
      expect(call).toBeDefined();
      if (!call) throw new Error("unreachable");
      const body = (call[1] as { body?: Record<string, unknown> }).body ?? {};
      expect(body.neg_method).toBe("vsf");
      expect(body.vsf_scale).toBe(VSF_SCALE_DEFAULT);

      requestSpy.mockRestore();
    },
    15_000,
  );
});
