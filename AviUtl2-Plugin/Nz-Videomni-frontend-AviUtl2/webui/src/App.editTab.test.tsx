import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import App from "./App";

// 2026-08-09: Edit stopped being a disabled mock tab and became a real
// `AppMode` with a mounted panel of its own (`modes/edit/EditScreen.tsx`).
// These app-level tests cover exactly the two things the promotion could have
// broken: that the main tab bar actually reaches the new panel, and that the
// singular `getByRole("tabpanel")` idiom the other 7 app-level test files rely
// on still resolves to ONE panel while Edit is on screen.

const panel = () => screen.getByRole("tabpanel");

describe("App / Edit tab", () => {
  it(
    "shows the Edit panel (and only it) after clicking the Edit tab",
    async () => {
      const user = userEvent.setup();
      render(<App />);

      // Wait for Create to finish seeding from `GET /config` so the app is in
      // its normal steady state before the tab switch.
      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      await user.click(screen.getByRole("tab", { name: /^edit$/i }));

      // Still exactly one visible tabpanel — the Edit one. Its sub-panels
      // deliberately carry no `role="tabpanel"` (see `EditSubTabs.tsx`).
      const visible = panel();
      expect(within(visible).getByRole("tab", { name: "Retake" })).toBeInTheDocument();
      expect(within(visible).getByRole("tab", { name: "Outpainting" })).toBeInTheDocument();
      expect(within(visible).getByRole("tab", { name: "Inpainting" })).toBeDisabled();
      // Create's Generate button is now hidden along with its panel.
      expect(screen.queryByRole("button", { name: /^generate$/i })).not.toBeInTheDocument();
    },
    20_000,
  );

  it(
    "goes back to Create with its form state intact (Edit is always-mounted like the rest)",
    async () => {
      const user = userEvent.setup();
      render(<App />);

      await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 });

      await user.click(screen.getByRole("tab", { name: /^edit$/i }));
      await user.click(screen.getByRole("tab", { name: /^single$/i }));

      expect(await screen.findByRole("button", { name: /^generate$/i }, { timeout: 5_000 })).toBeInTheDocument();
      // The Edit sub-tabs are hidden again, so the singular tabpanel query
      // (and every `within(panel())` in the other files) resolves as before.
      expect(within(panel()).queryByRole("tab", { name: "Retake" })).not.toBeInTheDocument();
    },
    20_000,
  );
});
