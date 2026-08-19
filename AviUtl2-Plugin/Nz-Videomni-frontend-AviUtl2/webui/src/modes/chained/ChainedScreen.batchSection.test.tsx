import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ReactNode } from "react";
import { createMockBridge } from "../../bridge/mockBridge";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { JobsProvider } from "../../jobs/JobsContext";
import { ToastProvider } from "../../shell/ToastContext";
import { PrefillPolicyProvider } from "../../shell/PrefillPolicyContext";
import { ShowNoteProvider } from "../../shell/NoteArea";
import { ChainedScreen } from "./ChainedScreen";

/**
 * §1-7 回帰ピン: Batch i2v-long のセクションを Chain 画面に足したことで、
 * 既存レイアウト（`.single-layout` の2列グリッド）と Generate/JobLedger が
 * 壊れていないことを固定する。
 *
 * Regression pin for the §1-7 wiring: the Batch i2v-long `<details>` must be a
 * FULL-WIDTH SIBLING BELOW `.single-layout`, never a third grid child — dropping
 * it inside the two-column grid would silently squeeze the batch panel (and the
 * form/ledger columns) into a broken layout that no unit test would otherwise
 * notice. The Generate button and the job ledger must also still be exactly
 * where they were.
 */
function Providers({ children, nativeBridge }: { children: ReactNode; nativeBridge: ReturnType<typeof createMockBridge> }) {
  return (
    <LanguageProvider>
      <ToastProvider>
        <PrefillPolicyProvider>
          <ShowNoteProvider showNote={() => {}}>
            <JobsProvider nativeBridge={nativeBridge}>{children}</JobsProvider>
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

describe("ChainedScreen — Batch i2v-long section wiring (§1-7)", () => {
  it("バッチセクションは.single-layoutの外・直後の兄弟として置かれる", async () => {
    const { container } = renderChain();
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });

    const layout = container.querySelector(".single-layout");
    const section = container.querySelector("details.batch-section");
    expect(layout).not.toBeNull();
    expect(section).not.toBeNull();
    // グリッドの子ではない。
    expect(layout!.contains(section)).toBe(false);
    // 同じ親の、レイアウト直後の兄弟。
    expect(section!.parentElement).toBe(layout!.parentElement);
    expect(layout!.nextElementSibling).toBe(section);
    // `.single-layout` の直接の子は従来どおり2つ（フォーム列と生成列）。
    expect(layout!.children).toHaveLength(2);
  });

  it("既定で閉じており、見出しだけが見えている", async () => {
    const { container } = renderChain();
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });

    const details = container.querySelector("details.batch-section") as HTMLDetailsElement;
    expect(details.open).toBe(false);
    expect(screen.getByText(en.batchI2vLong.heading)).toBeInTheDocument();
    // 閉じている間はバッチの操作系がDOM上に出ていても、Chainの生成導線とは
    // 別のボタンであること（名前が衝突していない）を確認する。
    expect(screen.getByRole("button", { name: /^generate$/i })).toBeInTheDocument();
  });

  it("Generateボタンとジョブ台帳は従来どおり存在する", async () => {
    const { container } = renderChain();
    await screen.findByText(/^clip 1$/i, {}, { timeout: 5_000 });

    const generate = screen.getByRole("button", { name: /^generate$/i });
    expect(generate).toBeEnabled();
    // 生成列の中に残っている（バッチセクションへ吸い込まれていない）。
    expect(container.querySelector(".generation-column")!.contains(generate)).toBe(true);
    expect(container.querySelector(".job-ledger")).not.toBeNull();
    expect(container.querySelector(".generation-column")!.contains(container.querySelector(".job-ledger"))).toBe(true);
  });
});
