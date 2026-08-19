import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "../../i18n/LanguageContext";
import { en } from "../../i18n/strings";
import { BatchI2vLongTable } from "./BatchI2vLongTable";
import type { I2vLongRow, I2vLongStat } from "./imageRows";

function makeRow(overrides: Partial<I2vLongRow> = {}): I2vLongRow {
  return { queue: 1, image: "a.png", prompt: "", stat: "Waiting", output: "", error: "", ...overrides };
}

interface RenderOpts {
  disabled?: boolean;
  onResetRow?: (queue: number) => void;
  onUpdateRowPrompt?: (index: number, value: string) => void;
  onCopyChainPrompt?: (index: number) => void;
}

function renderTable(rows: I2vLongRow[], opts: RenderOpts = {}) {
  const onResetRow = opts.onResetRow ?? vi.fn();
  const onUpdateRowPrompt = opts.onUpdateRowPrompt ?? vi.fn();
  const onCopyChainPrompt = opts.onCopyChainPrompt ?? vi.fn();
  const utils = render(
    <LanguageProvider>
      <BatchI2vLongTable
        rows={rows}
        disabled={opts.disabled ?? false}
        onResetRow={onResetRow}
        onUpdateRowPrompt={onUpdateRowPrompt}
        onCopyChainPrompt={onCopyChainPrompt}
      />
    </LanguageProvider>,
  );
  return { ...utils, onResetRow, onUpdateRowPrompt, onCopyChainPrompt };
}

describe("BatchI2vLongTable", () => {
  it("行が無いときは空メッセージだけを出す", () => {
    renderTable([]);
    expect(screen.getByText(en.batchI2vLong.table.empty)).toBeInTheDocument();
    expect(screen.queryByRole("table")).toBeNull();
  });

  it("6列（#/画像/プロンプト/状態/出力/🔁）で描画する — A2Vのduration・画像selectは無い", () => {
    const { container } = renderTable([makeRow({ output: "a.mp4", stat: "Done" })]);

    const headers = Array.from(container.querySelectorAll("thead th"));
    expect(headers).toHaveLength(6);
    expect(headers.slice(0, 5).map((th) => th.textContent)).toEqual([
      en.batchI2vLong.table.queue,
      en.batchI2vLong.table.image,
      en.batchI2vLong.table.prompt,
      en.batchI2vLong.table.stat,
      en.batchI2vLong.table.output,
    ]);
    // 行別画像の`<select>`はこの機能には無い（画像は1行1枚で固定）。
    expect(container.querySelector("select")).toBeNull();
    expect(container.querySelectorAll("tbody tr td")).toHaveLength(6);
  });

  // 2026-07-30 行ごとプロンプト（オーナー実機フィードバック）。
  it("プロンプト列は編集可能で、入力すると行indexと値を返す", async () => {
    const user = userEvent.setup();
    const onUpdateRowPrompt = vi.fn();
    renderTable([makeRow({ queue: 1 }), makeRow({ queue: 2, image: "b.png" })], { onUpdateRowPrompt });

    const inputs = screen.getAllByLabelText(en.batchI2vLong.table.promptInputLabel);
    expect(inputs).toHaveLength(2);
    expect(inputs[1]).toHaveAttribute("maxlength", "2000");

    await user.type(inputs[1]!, "x");
    expect(onUpdateRowPrompt).toHaveBeenCalledWith(1, "x");
  });

  it("行のプロンプト値がそのまま表示される", () => {
    renderTable([makeRow({ prompt: "in the rain" })]);
    expect(screen.getByLabelText(en.batchI2vLong.table.promptInputLabel)).toHaveValue("in the rain");
  });

  it("📝は行indexを返す", async () => {
    const user = userEvent.setup();
    const onCopyChainPrompt = vi.fn();
    renderTable([makeRow({ queue: 1 }), makeRow({ queue: 2, image: "b.png" })], { onCopyChainPrompt });

    const buttons = screen.getAllByRole("button", { name: en.batchI2vLong.table.copyChainPromptButton });
    await user.click(buttons[1]!);
    expect(onCopyChainPrompt).toHaveBeenCalledWith(1);
  });

  it("disabled中はプロンプト入力と📝も無効", () => {
    renderTable([makeRow()], { disabled: true });
    expect(screen.getByLabelText(en.batchI2vLong.table.promptInputLabel)).toBeDisabled();
    expect(screen.getByRole("button", { name: en.batchI2vLong.table.copyChainPromptButton })).toBeDisabled();
  });

  it("各状態のアイコンと語を出す", () => {
    const stats: I2vLongStat[] = ["Waiting", "Generating", "Done", "Failed"];
    const { container } = renderTable(stats.map((stat, i) => makeRow({ queue: i + 1, image: `${i}.png`, stat })));

    const cells = Array.from(container.querySelectorAll(".batch-stat-cell")).map((el) => el.textContent);
    expect(cells).toEqual([
      `⚪ ${en.batchI2vLong.stat.waiting}`,
      `⏳ ${en.batchI2vLong.stat.generating}`,
      `✅ ${en.batchI2vLong.stat.done}`,
      `❌ ${en.batchI2vLong.stat.failed}`,
    ]);
  });

  it("Failed行はerrorをtitleに出し、それ以外の行はtitleを持たない", () => {
    const { container } = renderTable([
      makeRow({ queue: 1, stat: "Failed", error: "generate/chain 422: bad clip" }),
      makeRow({ queue: 2, image: "b.png", stat: "Done", output: "b.mp4" }),
    ]);

    const cells = Array.from(container.querySelectorAll(".batch-stat-cell"));
    expect(cells[0]).toHaveAttribute("title", "generate/chain 422: bad clip");
    expect(cells[1]).not.toHaveAttribute("title");
  });

  it("🔁はDone/Failedのみ活性で、押すとその行のqueueを返す", async () => {
    const user = userEvent.setup();
    const onResetRow = vi.fn();
    renderTable(
      [
        makeRow({ queue: 1, stat: "Waiting" }),
        makeRow({ queue: 2, image: "b.png", stat: "Generating" }),
        makeRow({ queue: 3, image: "c.png", stat: "Done", output: "c.mp4" }),
        makeRow({ queue: 4, image: "d.png", stat: "Failed", error: "x" }),
      ],
      { onResetRow },
    );

    const buttons = screen.getAllByRole("button", { name: en.batchI2vLong.table.resetButton });
    expect(buttons[0]).toBeDisabled();
    expect(buttons[1]).toBeDisabled();
    expect(buttons[2]).toBeEnabled();
    expect(buttons[3]).toBeEnabled();

    await user.click(buttons[2]!);
    expect(onResetRow).toHaveBeenCalledWith(3);
  });

  it("disabled中は🔁が全行で無効", () => {
    renderTable([makeRow({ queue: 1, stat: "Done", output: "a.mp4" }), makeRow({ queue: 2, image: "b.png", stat: "Failed", error: "x" })], {
      disabled: true,
    });

    for (const button of screen.getAllByRole("button", { name: en.batchI2vLong.table.resetButton })) {
      expect(button).toBeDisabled();
    }
  });
});
