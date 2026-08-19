import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import { bridge } from "./bridge";

// Create and Chain are both mounted at once now (only the active tab is
// visible), and they share one JobsContext, so the running job's
// `Clip x/y` badge renders in *both* ledgers. `getByRole("tabpanel")` returns
// the single visible panel; `within(panel())` scopes such duplicated text to
// the screen that's actually on screen.
const panel = () => screen.getByRole("tabpanel");

// M6 end-to-end coverage for the Chain screen, mirroring App.test.tsx's
// structure for Create: exercises the whole ChainedScreen wired to the dev
// mock bridge (`src/bridge/mockBridge.ts`), on top of the unit-level
// coverage in `src/modes/chained/*.test.ts`.

describe("App / Chain screen", () => {
  it(
    "runs a 2-clip from-scratch chain: prompt -> generate -> poll (clip/clip_count visible) -> completed -> insert",
    async () => {
      const user = userEvent.setup();
      render(<App />);

      const promptBox = await screen.findByPlaceholderText(/describe the video/i, {}, { timeout: 5_000 });
      await user.type(promptBox, "a cat riding a skateboard, then a dog joins in");

      const chainedTab = await screen.findByRole("tab", { name: /^chained$/i }, { timeout: 5_000 });
      await user.click(chainedTab);

      // No source video attached => the auto-detected "scratch" mode, which
      // starts with the 2-clip minimum already satisfied, so Generate should
      // be enabled immediately.
      const generateButton = await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      expect(generateButton).toBeEnabled();

      await user.click(generateButton);

      expect(await screen.findByText(/generating/i)).toBeInTheDocument();

      // The clip/clip_count badge (Docs/API_REFERENCE.md §6 `clip`/
      // `clip_count`) should surface on the ledger's `JobCard` while running,
      // before the job completes.
      await waitFor(
        () => {
          // Scoped to the visible Chain panel: the same job's badge also
          // renders in the hidden Create ledger (shared JobsContext).
          expect(within(panel()).getByText(/Clip \d\/\d/)).toBeInTheDocument();
        },
        { timeout: 12_000 },
      );

      // U-R1: insert now runs from the ledger's `JobCard` (`strings.jobs.insert`
      // /`inserted`), not the removed inline `GenerationPanel`.
      const insertButton = await screen.findByRole("button", { name: /^insert$/i }, { timeout: 15_000 });
      await user.click(insertButton);

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /^inserted$/i })).toBeInTheDocument();
      });
    },
    30_000,
  );

  it(
    "strips <lora:...> tags from the shared prompt into a loras[] array on submit, with no ignored-tags toast",
    async () => {
      const user = userEvent.setup();
      // Chain submission always goes through the app-wide `bridge` singleton
      // (`useGeneration`'s default `apiClient`/`client`, `modes/single/
      // useGeneration.ts` — unrelated to the `nativeBridge` prop threaded
      // for upload/timeline calls), so spying on it lets us inspect the
      // actual `POST /generate/chain` body the same way `api/client.ts`'s
      // `call()` builds it (`backend.request` with
      // `path: "/api/v1/generate/chain"`).
      const requestSpy = vi.spyOn(bridge, "request");

      render(<App />);

      const promptBox = await screen.findByPlaceholderText(/describe the video/i, {}, { timeout: 5_000 });
      await user.type(promptBox, "a cat <lora:Pixar_Toon:1.0> riding a skateboard");

      const chainedTab = await screen.findByRole("tab", { name: /^chained$/i }, { timeout: 5_000 });
      await user.click(chainedTab);

      const generateButton = await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });
      await user.click(generateButton);

      expect(await screen.findByText(/generating/i)).toBeInTheDocument();

      // Docs/API_REFERENCE.md §5.2/§5.3: the tag never reaches the backend as
      // prompt text — `buildChainRequest` (`modes/chained/chainUtils.ts`) splits
      // it into the stripped `prompt` and a `loras[]` entry, alongside the
      // always-explicit `chunked_upsample` (Chain's UI default is on;
      // `useChainForm`'s `chunkedUpsample` JSDoc explains why it can never be
      // omitted).
      const chainCall = requestSpy.mock.calls.find(
        (call) => call[0] === "backend.request" && (call[1] as { path?: string }).path === "/api/v1/generate/chain",
      );
      expect(chainCall).toBeDefined();
      const body = (
        chainCall?.[1] as {
          body?: {
            prompt?: string;
            loras?: Array<{ name: string; strength: number }>;
            chunked_upsample?: boolean;
          };
        }
      ).body;
      expect(body?.prompt).toBe("a cat riding a skateboard");
      expect(body?.loras).toEqual([{ name: "Pixar_Toon", strength: 1.0 }]);
      expect(body?.chunked_upsample).toBe(true);

      // The old "Style LoRA tags are ignored in chain mode" warning is gone —
      // loras are honored now, not discarded.
      expect(screen.queryByText(/ignored/i)).not.toBeInTheDocument();

      requestSpy.mockRestore();
    },
    15_000,
  );
});
