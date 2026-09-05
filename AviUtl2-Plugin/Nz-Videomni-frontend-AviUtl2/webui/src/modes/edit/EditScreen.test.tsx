import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "../../i18n/LanguageContext";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import { EditScreen } from "./EditScreen";

// 2026-08-09: the Edit tab was promoted from a disabled mock to a real mode,
// and gained a sub-tab row of its own (Retake / Outpainting / Inpainting).
// Inpainting is the mock here — `<button disabled>` with no `onClick` at all,
// exactly like `shell/ModeTabs.tsx`'s Toolbox. These tests cover the SKELETON
// only: order, the disabled one, the switch, and the deliberate ABSENCE of
// `role="tabpanel"` (7 existing app-level test files take
// `getByRole("tabpanel")` in the singular). The Outpainting panel's own
// behaviour lives in `OutpaintingPanel.test.tsx`.
//
// Outpainting (2026-08-09) turned that panel from a placeholder into a real
// form, which brought two of its dependencies into this file's render tree:
// the seed field's ♻ button reads the job ledger through `useJobsContext`
// (stubbed below, exactly as `CommonGenerationFields.test.tsx` and the
// GenerationForm tests do, so no `JobsProvider` is needed), and the form hook
// fetches `GET /loras` through the app-wide mock bridge, which resolves
// harmlessly and is guarded by `useLoras`'s own mounted-ref.
//
// The screen now also mounts the shared `jobs/JobLedger`, which reads FOUR more
// members off the same context — hence the fuller stub. An empty `jobs` array
// makes the ledger render its "no jobs yet" line, so the cancel/delete halves
// are never actually invoked; they exist so the destructuring finds them.
//
// `serverBusy` is the one member a test needs to move (the Generate gate), so
// it is read through a mutable box; `vi.hoisted` because the `vi.mock` factory
// is hoisted above this file's own statements.
const jobsStub = vi.hoisted(() => ({ serverBusy: false }));

vi.mock("../../jobs/JobsContext", () => ({
  useJobsContext: () => ({
    jobs: [],
    serverBusy: jobsStub.serverBusy,
    cancellingIds: new Set<string>(),
    deletingIds: new Set<string>(),
    cancelJob: () => Promise.resolve(),
    deleteJob: () => Promise.resolve(),
  }),
}));

afterEach(() => {
  jobsStub.serverBusy = false;
});

function renderEdit(
  initialIntent?: GenerationPrefill,
  prompt?: string,
  subTabsDisabled?: { retake: boolean; outpainting: boolean },
) {
  return render(
    <LanguageProvider>
      <EditScreen initialIntent={initialIntent} prompt={prompt} subTabsDisabled={subTabsDisabled} />
    </LanguageProvider>,
  );
}

/** W0: a routed right-click payload, exactly as `AppShell` builds it (the
 * screen only reads `intent`, but the whole `GenerationPrefill` is what W2/W3's
 * panels will consume, so the fixture is the real shape). */
function prefill(intent: string): GenerationPrefill {
  return {
    intent,
    targetMode: "edit",
    selection: {
      hasRange: true,
      rangeStart: 48,
      rangeEnd: 96,
      selected: [
        {
          layer: 1, frameStart: 0, frameEnd: 120, effectName: "動画ファイル",
          filePath: "C:\\v\\a.mp4", objectName: "a", textContent: null,
          mediaWidth: 1920, mediaHeight: 1080, mediaDurationSec: 5,
        },
      ],
      cursorFrame: 48,
      cursorLayer: 1,
      rate: 30,
      scale: 1,
      sampleRate: 44100,
    },
  };
}

describe("EditScreen sub-tabs", () => {
  it("renders 3 sub-tabs in order Retake/Outpainting/Inpainting", () => {
    renderEdit();
    const tabs = screen.getAllByRole("tab");
    expect(tabs).toHaveLength(3);
    expect(tabs.map((tab) => tab.textContent)).toEqual(["Retake", "Outpainting", "Inpainting"]);
  });

  it("disables the Inpainting sub-tab only", () => {
    renderEdit();
    expect(screen.getByRole("tab", { name: "Inpainting" })).toBeDisabled();
    expect(screen.getByRole("tab", { name: "Retake" })).not.toBeDisabled();
    expect(screen.getByRole("tab", { name: "Outpainting" })).not.toBeDisabled();
  });

  // A `hidden` panel is out of the accessibility tree, so role queries need
  // `{ hidden: true }` to reach it at all — `toBeVisible()` is what actually
  // asserts which of the two is on screen.
  const heading = (name: RegExp) => screen.getByRole("heading", { name, hidden: true });

  it("starts on Retake, with the Outpainting panel hidden", () => {
    renderEdit();
    expect(screen.getByRole("tab", { name: "Retake" })).toHaveAttribute("aria-selected", "true");
    expect(heading(/^retake/i)).toBeVisible();
    expect(heading(/^outpainting/i)).not.toBeVisible();
  });

  it("switches to the Outpainting panel when its sub-tab is clicked", async () => {
    const user = userEvent.setup();
    renderEdit();
    await user.click(screen.getByRole("tab", { name: "Outpainting" }));
    expect(heading(/^outpainting/i)).toBeVisible();
    expect(heading(/^retake/i)).not.toBeVisible();
    expect(screen.getByRole("tab", { name: "Outpainting" })).toHaveAttribute("aria-selected", "true");
  });

  it("does nothing when the disabled Inpainting sub-tab is clicked", async () => {
    const user = userEvent.setup();
    renderEdit();
    await user.click(screen.getByRole("tab", { name: "Inpainting" }));
    // Still on Retake — no panel of its own was revealed, and no sub-tab moved.
    expect(heading(/^retake/i)).toBeVisible();
    expect(screen.getByRole("tab", { name: "Retake" })).toHaveAttribute("aria-selected", "true");
    expect(screen.queryByRole("heading", { name: /^inpainting/i, hidden: true })).not.toBeInTheDocument();
  });

  it("does NOT mark its sub-panels with role=tabpanel", () => {
    // Load-bearing: the app-level tests scope queries with a singular
    // `getByRole("tabpanel")` (the one visible mode screen). A second visible
    // tabpanel inside Edit would make every one of those queries ambiguous.
    renderEdit();
    expect(screen.queryAllByRole("tabpanel", { hidden: true })).toHaveLength(0);
  });

  // W0 (2026-08-09): the Edit tab became a right-click destination, so a routed
  // intent pre-selects the sub-tab. Consumed ONCE in the lazy `useState`
  // initializer — `AppShell` bumps `remountTokens.edit` per route, so a fresh
  // right-click always arrives as a fresh mount.
  it("auto-selects the Outpainting sub-tab for an 'outpaint' intent", () => {
    renderEdit(prefill("outpaint"));
    expect(screen.getByRole("tab", { name: "Outpainting" })).toHaveAttribute("aria-selected", "true");
    expect(heading(/^outpainting/i)).toBeVisible();
    expect(heading(/^retake/i)).not.toBeVisible();
  });

  it("auto-selects the Retake sub-tab for a 'retake' intent", () => {
    renderEdit(prefill("retake"));
    expect(screen.getByRole("tab", { name: "Retake" })).toHaveAttribute("aria-selected", "true");
    expect(heading(/^retake/i)).toBeVisible();
    expect(heading(/^outpainting/i)).not.toBeVisible();
  });

  it("falls back to Retake for an unrecognized intent", () => {
    // Defensive: an intent this screen does not know (e.g. a future Inpainting
    // route landing before its panel exists) must not blank the screen.
    renderEdit(prefill("totally-unknown"));
    expect(screen.getByRole("tab", { name: "Retake" })).toHaveAttribute("aria-selected", "true");
    expect(heading(/^retake/i)).toBeVisible();
  });

  it("lets the user override the auto-selected sub-tab", async () => {
    // The intent SEEDS the selection; it does not pin it (a lazy initializer,
    // never a prop-following effect), so a manual click still wins.
    const user = userEvent.setup();
    renderEdit(prefill("outpaint"));
    expect(heading(/^outpainting/i)).toBeVisible();
    await user.click(screen.getByRole("tab", { name: "Retake" }));
    expect(heading(/^retake/i)).toBeVisible();
    expect(heading(/^outpainting/i)).not.toBeVisible();
  });

  // 2026-08-09: Edit gained a job ledger of its own — the SAME `jobs/JobLedger`
  // Create/Chain mount, fed by the same provider. Before this, the panel's copy
  // told the user to go and look at the Single screen, and `AppShell`'s toast
  // click had to switch tabs to find a ledger to highlight in.
  it("mounts the shared job ledger below the panels", () => {
    renderEdit();
    expect(screen.getByRole("heading", { name: /^jobs/i })).toBeVisible();
    expect(screen.getByText(/no jobs yet/i)).toBeVisible();
  });

  it("keeps the ledger visible on either sub-tab", async () => {
    // It sits OUTSIDE the sub-tab switch, so it is not hidden with a panel.
    const user = userEvent.setup();
    renderEdit();
    await user.click(screen.getByRole("tab", { name: "Outpainting" }));
    expect(screen.getByRole("heading", { name: /^jobs/i })).toBeVisible();
    await user.click(screen.getByRole("tab", { name: "Retake" }));
    expect(screen.getByRole("heading", { name: /^jobs/i })).toBeVisible();
  });

  // The shared prompt (`AppShell`'s `PromptBar`) reaches the Outpainting panel
  // as a prop — the panel has no prompt box of its own to fall back on, so a
  // broken thread here shows up as a permanent "fill the main prompt" block.
  it("threads the shared prompt down into the Outpainting panel", () => {
    renderEdit(prefill("outpaint"), "a wide city street");
    const note = screen.getByRole("note");
    expect(within(note).queryByText(/fill the main prompt/i)).not.toBeInTheDocument();
    // And no second prompt box was introduced anywhere under Edit.
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("blocks on the empty shared prompt when none was handed down", () => {
    renderEdit(prefill("outpaint"));
    expect(within(screen.getByRole("note")).getByText(/fill the main prompt/i)).toBeInTheDocument();
  });

  // 2カラム化（オーナー指示 第2バッチ、2026-08-09）: the panels used to be one
  // full-width stack with the ledger at the very bottom, where it was
  // effectively invisible. Edit now uses Create/Chain's own `.single-layout` +
  // `.generation-column` pair, so the Generate button and the ledger sit
  // together in the right-hand column and the form keeps the left one. Asserted
  // through the class names because that IS the change — the two columns are
  // exactly the shared classes, not Edit-local copies of them.
  it("puts the Generate button and the job ledger together in the generation column", () => {
    const { container } = renderEdit(prefill("outpaint"));
    const layout = container.querySelector(".single-layout");
    expect(layout).not.toBeNull();

    const column = container.querySelector<HTMLElement>(".generation-column");
    expect(column).not.toBeNull();
    expect(within(column!).getByRole("button", { name: /^generate$/i })).toBeInTheDocument();
    expect(within(column!).getByRole("heading", { name: /^jobs/i })).toBeInTheDocument();

    // …and the form column no longer ends in a Generate button of its own.
    const formColumn = container.querySelector<HTMLElement>(".edit-form-column");
    expect(formColumn).not.toBeNull();
    expect(within(formColumn!).queryByRole("button", { name: /^generate$/i })).not.toBeInTheDocument();
  });

  it("drops the Generate group — but not the ledger — on the Retake sub-tab", async () => {
    // Retake has nothing to generate yet, so only the ledger is left in the
    // right-hand column. The ledger itself sits OUTSIDE the sub-tab switch.
    const user = userEvent.setup();
    const { container } = renderEdit(prefill("outpaint"));
    expect(screen.getByRole("button", { name: /^generate$/i })).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Retake" }));
    expect(screen.queryByRole("button", { name: /^generate$/i })).not.toBeInTheDocument();
    const column = container.querySelector<HTMLElement>(".generation-column");
    expect(within(column!).getByRole("heading", { name: /^jobs/i })).toBeVisible();
  });

  it("keeps the Retake panel mounted while Outpainting is shown", async () => {
    // Both sub-panels stay mounted and are merely `hidden` — the same
    // arrangement `AppShell` uses for the mode screens, so a future panel's
    // form state will survive a sub-tab switch.
    const user = userEvent.setup();
    renderEdit();
    await user.click(screen.getByRole("tab", { name: "Outpainting" }));
    expect(screen.getByRole("heading", { name: /^retake/i, hidden: true })).toBeInTheDocument();
  });
});

// §3-98 P5 / §3-102 — the sub-tab half of the base-model feature scope.
//
// Until the Retake/End source 開通, LTX 2.5 declared BOTH `retake` and
// `outpaint`, so `disabledModesFor`'s `needsAnyOf` greyed the whole Edit TAB and
// no sub-tab was ever reachable to grey. That made the tab-level rule look as
// though it covered this case; it did so by accident. The moment ONE of the two
// opens, the tab goes live and the other sub-tab has to grey on its own.
//
// These tests therefore drive the prop directly rather than through a base
// model: the state they describe is not reachable from any fixture yet, and the
// translation from a server feature name to these booleans is tested where it
// lives (`shell/featureScope.test.ts`'s `editSubTabsDisabledFor`).
describe("EditScreen sub-tabs — base-model feature scope", () => {
  const subTab = (name: string) => screen.getByRole("tab", { name });
  const heading = (name: RegExp) => screen.getByRole("heading", { name, hidden: true });

  it("leaves both real sub-tabs live when the prop is omitted", () => {
    // The regression that matters most: a build with this feature must look
    // exactly like the build before it on a base model with no restrictions.
    renderEdit();
    expect(subTab("Retake")).not.toBeDisabled();
    expect(subTab("Outpainting")).not.toBeDisabled();
    expect(subTab("Inpainting")).toBeDisabled();
  });

  it("greys the Outpainting sub-tab — and only it — when the engine cannot outpaint", () => {
    renderEdit(undefined, undefined, { retake: false, outpainting: true });
    expect(subTab("Outpainting")).toBeDisabled();
    expect(subTab("Retake")).not.toBeDisabled();
  });

  it("greys the Retake sub-tab — and only it — when the engine cannot retake", () => {
    renderEdit(undefined, undefined, { retake: true, outpainting: false });
    expect(subTab("Retake")).toBeDisabled();
    expect(subTab("Outpainting")).not.toBeDisabled();
  });

  it("gives a greyed sub-tab its OWN reason, and the Inpainting mock none", () => {
    // One greyed treatment, two reasons — the tooltip is what tells them apart
    // (`shell/ModeTabs.tsx` makes the same call for its Toolbox mock). A greyed
    // control with no stated reason is what this line exists to prevent; a mock
    // that grew one would be telling the user to switch base models for
    // something no base model has.
    renderEdit(undefined, undefined, { retake: false, outpainting: true });
    expect(subTab("Outpainting")).toHaveAttribute(
      "title",
      expect.stringMatching(/Outpainting is not available/i),
    );
    expect(subTab("Retake")).not.toHaveAttribute("title");
    expect(subTab("Inpainting")).not.toHaveAttribute("title");
  });

  it("does nothing when a greyed real sub-tab is clicked", async () => {
    // The same treatment the Inpainting mock already gets: `<button disabled>`
    // with no `onClick` attached at all, so there is no handler to reach even if
    // the disabled attribute were somehow bypassed.
    const user = userEvent.setup();
    renderEdit(undefined, undefined, { retake: false, outpainting: true });
    await user.click(subTab("Outpainting"));
    expect(subTab("Retake")).toHaveAttribute("aria-selected", "true");
    expect(heading(/^retake/i)).toBeVisible();
    expect(heading(/^outpainting/i)).not.toBeVisible();
  });

  it("falls back to Retake for an 'outpaint' route when Outpainting is greyed", () => {
    // The case C2 creates: the engine gains Retake but still has no Outpainting,
    // and a right-click 画角拡張 lands here anyway (the timeline's context menu is
    // outside this app's greying entirely). Landing ON the greyed sub-tab would
    // show a panel behind a tab the user cannot click back to.
    renderEdit(prefill("outpaint"), undefined, { retake: false, outpainting: true });
    expect(subTab("Retake")).toHaveAttribute("aria-selected", "true");
    expect(heading(/^retake/i)).toBeVisible();
    expect(heading(/^outpainting/i)).not.toBeVisible();
  });

  it("falls back to Outpainting for the default/'retake' route when Retake is greyed", () => {
    // The mirror. Both greyed at once cannot reach this screen: `disabledModesFor`
    // takes the whole Edit tab in that case, so it is never mounted.
    renderEdit(prefill("retake"), undefined, { retake: true, outpainting: false });
    expect(subTab("Outpainting")).toHaveAttribute("aria-selected", "true");
    expect(heading(/^outpainting/i)).toBeVisible();
    expect(heading(/^retake/i)).not.toBeVisible();
  });

  it("still lets a routed intent pick a sub-tab that IS available", () => {
    // The corollary: the fallback must be caused by the RESTRICTION, not by the
    // prop merely being present. An 'outpaint' route on an engine that CAN
    // outpaint lands on Outpainting exactly as it always did.
    renderEdit(prefill("outpaint"), undefined, { retake: true, outpainting: false });
    expect(subTab("Outpainting")).toHaveAttribute("aria-selected", "true");
  });

  // R7 (実装計画 §リスク): Edit 配下のミューテーションを**全部数える**。7 本あり、
  // それ以外は無い:
  //   送信 2 本      — Outpainting の `submit` (POST /generate) と
  //                    Retake の `submitChain` (POST /generate/chain)
  //   アップロード 2 本 — 両パネルの 📁（`useOutpaintForm` / `useRetakeForm` の
  //                    `uploadPath` = POST /upload）
  //   予約系 3 本    — `reservePlacement` / `bindToJob` /
  //                    `rollbackReservedPlacement`。**Retake 側にしか無い**:
  //                    Outpainting のルートは席を取らない（placement null）ので、
  //                    専用の `useGenerationSubmit` インスタンスを分けて
  //                    bind/rollback を共有させていない（EditScreen の doc 参照）。
  //
  // 灰色化が入口をちゃんと塞ぐのは、この 7 本が例外なく
  //  (a) `subMode === <その側>` のときだけ描かれる生成群の中か、
  //  (b) `hidden` になるサブパネルの中か、
  // のどちらかにしか無いからで、サブタブを押せなければ `subMode` はそこへ行かない。
  // 下の 2 本はその事実を両向きで押さえる。
  it("leaves no Outpainting-side mutation reachable while it is greyed (R7)", async () => {
    const user = userEvent.setup();
    const { container } = renderEdit(undefined, "a wide city street", {
      retake: false,
      outpainting: true,
    });

    // 入口の入口: 灰色のサブタブは選べない。
    await user.click(subTab("Outpainting"));
    expect(subTab("Retake")).toHaveAttribute("aria-selected", "true");

    // (a) 送信: Outpainting の Generate は `subMode === "outpainting"` のときだけ
    //     描かれる。Retake 側も右クリック由来のスナップショットが無いので生成群を
    //     出さない — つまり Generate ボタンは画面に 1 つも無い。
    expect(screen.queryByRole("button", { name: /^generate$/i })).not.toBeInTheDocument();

    // (b) アップロード: Outpainting パネルごと `hidden`。
    expect(heading(/^outpainting/i)).not.toBeVisible();

    // …そして残った側は普通に使える（灰色化が巻き添えにしていない）。
    expect(heading(/^retake/i)).toBeVisible();
    expect(container.querySelector(".edit-subpanel")).not.toBeNull();
  });

  it("leaves no Retake-side mutation — the 予約系 3 included — reachable while it is greyed (R7)", async () => {
    const user = userEvent.setup();
    renderEdit(prefill("retake"), "a wide city street", { retake: true, outpainting: false });

    // ルートは Retake 行きだったが、灰色なので Outpainting へ逃げている。
    expect(subTab("Outpainting")).toHaveAttribute("aria-selected", "true");
    await user.click(subTab("Retake"));
    expect(subTab("Outpainting")).toHaveAttribute("aria-selected", "true");

    // 予約系 3 本はすべて Retake の Generate 押下か、Retake パネル内の ❌ からしか
    // 始まらない。前者は生成群ごと出ていない（見えている Generate は Outpainting の
    // 1 本だけ）、後者はパネルごと `hidden`。
    expect(screen.getAllByRole("button", { name: /^generate$/i })).toHaveLength(1);
    expect(heading(/^outpainting/i)).toBeVisible();
    expect(heading(/^retake/i)).not.toBeVisible();
  });

  // 2026-08-31: Outpainting was the one Generate button in the app with no
  // server-busy gate — it stayed pressable while a job ran or a model loaded,
  // and the press came back as a failed job in the ledger. It now follows the
  // same convention as Retake/Create/Chain: the button alone is frozen, the
  // form stays editable, and the label says why.
  it("freezes the Outpainting Generate while the server is busy, with the busy label", () => {
    jobsStub.serverBusy = true;
    renderEdit(prefill("outpaint"), "a wide city street");

    expect(screen.queryByRole("button", { name: /^generate$/i })).not.toBeInTheDocument();
    const button = screen.getByRole("button", { name: /^busy…$/i });
    expect(button).toBeDisabled();
  });

  it("labels the Outpainting Generate normally when the server is idle", () => {
    renderEdit(prefill("outpaint"), "a wide city street");

    expect(screen.getByRole("button", { name: /^generate$/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^busy…$/i })).not.toBeInTheDocument();
  });
});
