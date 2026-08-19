import { describe, expect, it } from "vitest";
import {
  DEFAULT_IMAGE_EXTENSIONS,
  deriveI2vLongOutDir,
  normalizeImageExtensions,
  scanImagesToRows,
  type ScannedImageFile,
} from "./imageRows";

/** Minimal `fs.listFiles` entry factory — only `name`/`sizeBytes` ever matter
 * to `scanImagesToRows`; `path`/`mtimeMs`/`durationSec` are carried along
 * because the real bridge response has them. */
function file(name: string, overrides: Partial<ScannedImageFile> = {}): ScannedImageFile {
  return {
    name,
    path: `C:\\imgs\\${name}`,
    sizeBytes: 1024,
    mtimeMs: 0,
    durationSec: 0,
    ...overrides,
  };
}

const PNG_ONLY = [".png"];

describe("normalizeImageExtensions", () => {
  it("小文字化・先頭ドット補完・前後空白の除去を行う", () => {
    expect(normalizeImageExtensions(["PNG", " .JPG ", "webp"])).toEqual([".png", ".jpg", ".webp"]);
  });

  it("空文字・空白のみ・ドットだけの要素は落とす", () => {
    expect(normalizeImageExtensions(["", "   ", ".", ".png"])).toEqual([".png"]);
  });

  it("重複は最初の1つだけ残す（正規化後で判定）", () => {
    expect(normalizeImageExtensions([".png", "PNG", "png", ".jpg"])).toEqual([".png", ".jpg"]);
  });

  it("空リスト/未指定/nullは既定の拡張子リストにフォールバックする", () => {
    expect(normalizeImageExtensions([])).toEqual([...DEFAULT_IMAGE_EXTENSIONS]);
    expect(normalizeImageExtensions(undefined)).toEqual([...DEFAULT_IMAGE_EXTENSIONS]);
    expect(normalizeImageExtensions(null)).toEqual([...DEFAULT_IMAGE_EXTENSIONS]);
    // 全要素が捨てられた場合も同じ（"実質空"の扱い）
    expect(normalizeImageExtensions(["", "  ", "."])).toEqual([...DEFAULT_IMAGE_EXTENSIONS]);
  });
});

describe("scanImagesToRows: 拡張子フィルタ", () => {
  it("拡張子は大文字小文字を区別せずに判定する", () => {
    const rows = scanImagesToRows([file("a.PNG"), file("b.png"), file("c.PnG")], PNG_ONLY);
    expect(rows.map((row) => row.image)).toEqual(["a.PNG", "b.png", "c.PnG"]);
  });

  it("許可リストにない拡張子・拡張子なしのファイルは除外する", () => {
    const rows = scanImagesToRows([file("a.png"), file("b.txt"), file("noext"), file("trailing.")], PNG_ONLY);
    expect(rows.map((row) => row.image)).toEqual(["a.png"]);
  });

  it(".tmp は許可リストに含まれていても常に除外する", () => {
    const rows = scanImagesToRows([file("a.png"), file("half.tmp"), file("HALF.TMP")], [".png", ".tmp"]);
    expect(rows.map((row) => row.image)).toEqual(["a.png"]);
  });

  it("許可リストが空なら既定の拡張子で判定する", () => {
    const rows = scanImagesToRows([file("a.png"), file("b.jpeg"), file("c.bmp")], []);
    expect(rows.map((row) => row.image)).toEqual(["a.png", "b.jpeg"]);
  });

  it("空フォルダ・全除外なら空配列を返す", () => {
    expect(scanImagesToRows([], PNG_ONLY)).toEqual([]);
    expect(scanImagesToRows([file("a.txt")], PNG_ONLY)).toEqual([]);
  });
});

describe("scanImagesToRows: 並び順と採番", () => {
  it("ファイル名の昇順に並べ替える（数値を数として比較する）", () => {
    const rows = scanImagesToRows([file("img10.png"), file("img2.png"), file("img1.png")], PNG_ONLY);
    expect(rows.map((row) => row.image)).toEqual(["img1.png", "img2.png", "img10.png"]);
  });

  it("入力順が更新日時順でも、結果はファイル名昇順で同一になる", () => {
    const byName = [file("a.png", { mtimeMs: 300 }), file("b.png", { mtimeMs: 200 }), file("c.png", { mtimeMs: 100 })];
    const byMtime = [byName[2]!, byName[1]!, byName[0]!];
    expect(scanImagesToRows(byMtime, PNG_ONLY)).toEqual(scanImagesToRows(byName, PNG_ONLY));
    expect(scanImagesToRows(byMtime, PNG_ONLY).map((row) => row.image)).toEqual(["a.png", "b.png", "c.png"]);
  });

  it("大文字小文字だけが違う名前も安定した順序になる（コードユニット順のタイブレーク）", () => {
    const rows = scanImagesToRows([file("B.png"), file("a.png"), file("A.png")], PNG_ONLY);
    expect(rows.map((row) => row.image)).toEqual(["A.png", "a.png", "B.png"]);
  });

  // 行プロンプト（2026-07-30）はスキャン時点では常に空 — 再スキャンで
  // 手入力が消えるのは仕様（ステートレスなバッチ）。
  it("queue は1始まりの連番、初期状態は Waiting・プロンプト/出力/エラーは空", () => {
    const rows = scanImagesToRows([file("b.png"), file("a.png")], PNG_ONLY);
    expect(rows).toEqual([
      { queue: 1, image: "a.png", prompt: "", stat: "Waiting", output: "", error: "" },
      { queue: 2, image: "b.png", prompt: "", stat: "Waiting", output: "", error: "" },
    ]);
  });

  it("入力配列を書き換えない（並べ替えはコピー上で行う）", () => {
    const input = [file("b.png"), file("a.png")];
    const snapshot = input.map((entry) => entry.name);
    scanImagesToRows(input, PNG_ONLY);
    expect(input.map((entry) => entry.name)).toEqual(snapshot);
  });
});

describe("scanImagesToRows: サイズ上限", () => {
  const limit = 10 * 1024 * 1024;

  it("上限超過の行はスキャン時点で Failed になり、理由が error に入る", () => {
    const rows = scanImagesToRows([file("big.png", { sizeBytes: limit + 1 }), file("ok.png", { sizeBytes: limit })], PNG_ONLY, limit);
    const big = rows.find((row) => row.image === "big.png");
    const ok = rows.find((row) => row.image === "ok.png");
    expect(big?.stat).toBe("Failed");
    expect(big?.error).toContain("too large");
    // 境界ちょうど（== 上限）は通す
    expect(ok?.stat).toBe("Waiting");
    expect(ok?.error).toBe("");
  });

  it("超過行も行としては残り、queue の連番は詰めない", () => {
    const rows = scanImagesToRows(
      [file("a.png"), file("b.png", { sizeBytes: limit + 1 }), file("c.png")],
      PNG_ONLY,
      limit,
    );
    expect(rows.map((row) => [row.queue, row.stat])).toEqual([
      [1, "Waiting"],
      [2, "Failed"],
      [3, "Waiting"],
    ]);
  });

  it("上限を渡さない/0/負値/非有限値なら判定そのものを行わない", () => {
    const huge = [file("big.png", { sizeBytes: Number.MAX_SAFE_INTEGER })];
    expect(scanImagesToRows(huge, PNG_ONLY)[0]?.stat).toBe("Waiting");
    expect(scanImagesToRows(huge, PNG_ONLY, 0)[0]?.stat).toBe("Waiting");
    expect(scanImagesToRows(huge, PNG_ONLY, -1)[0]?.stat).toBe("Waiting");
    expect(scanImagesToRows(huge, PNG_ONLY, Number.NaN)[0]?.stat).toBe("Waiting");
    expect(scanImagesToRows(huge, PNG_ONLY, Number.POSITIVE_INFINITY)[0]?.stat).toBe("Waiting");
  });
});

describe("deriveI2vLongOutDir", () => {
  it("画像フォルダの隣に `{フォルダ名}_i2vlong_out` を作る", () => {
    expect(deriveI2vLongOutDir("C:\\work\\images")).toBe("C:\\work\\images_i2vlong_out");
  });

  it("末尾の区切り文字は無視する", () => {
    expect(deriveI2vLongOutDir("C:\\work\\images\\")).toBe("C:\\work\\images_i2vlong_out");
    expect(deriveI2vLongOutDir("C:\\work\\images\\\\")).toBe("C:\\work\\images_i2vlong_out");
  });

  it("スラッシュ区切りでも親フォルダを正しく切り出す（結合はバックスラッシュ）", () => {
    expect(deriveI2vLongOutDir("C:/work/images")).toBe("C:/work\\images_i2vlong_out");
  });

  it("区切りのない裸の名前は親なしで扱う", () => {
    expect(deriveI2vLongOutDir("images")).toBe("images_i2vlong_out");
  });

  it("ドライブ直下は親が `C:` になる", () => {
    expect(deriveI2vLongOutDir("C:\\images")).toBe("C:\\images_i2vlong_out");
  });

  it("空文字は空の名前のまま接尾辞だけを返す（呼び出し側が未選択を弾く前提）", () => {
    expect(deriveI2vLongOutDir("")).toBe("_i2vlong_out");
  });
});
