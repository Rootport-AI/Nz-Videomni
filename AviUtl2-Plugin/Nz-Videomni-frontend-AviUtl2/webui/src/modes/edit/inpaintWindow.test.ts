import { describe, expect, it } from "vitest";
import {
  INPAINT_FRAMES_MAX,
  INPAINT_FRAMES_MIN,
  defaultInpaintFrames,
  placeInpaintWindow,
  roundUpTo128,
  roundUpTo8n1,
} from "./inpaintWindow";

// 台帳 §3-55 / オーナー裁定 D3: 窓の規則そのもの。純関数なので、実機ゲート
// G4（右クリックでフレーム数が埋まる／端で頭側へずれる／収まらない）の判定条件は
// 全部ここで固定できる。

describe("roundUpTo8n1", () => {
  it("leaves a value already on the grid alone", () => {
    for (const frames of [9, 17, 25, 121, 481]) {
      expect(roundUpTo8n1(frames)).toBe(frames);
    }
  });

  it("rounds UP, never down", () => {
    // 切り下げにすると部分フィルタの尻尾がマスクの外へ出る、というのがこの
    // 関数が `snapRetakeWindowFrames`（切り下げ）を借りない理由そのもの。
    expect(roundUpTo8n1(10)).toBe(17);
    expect(roundUpTo8n1(16)).toBe(17);
    expect(roundUpTo8n1(18)).toBe(25);
    expect(roundUpTo8n1(120)).toBe(121);
  });

  it("clamps up to the floor, and survives nonsense", () => {
    expect(roundUpTo8n1(1)).toBe(INPAINT_FRAMES_MIN);
    expect(roundUpTo8n1(0)).toBe(INPAINT_FRAMES_MIN);
    expect(roundUpTo8n1(-40)).toBe(INPAINT_FRAMES_MIN);
    expect(roundUpTo8n1(Number.NaN)).toBe(INPAINT_FRAMES_MIN);
  });

  it("does NOT cap at the ceiling — that is the default's job alone", () => {
    expect(roundUpTo8n1(600)).toBe(601);
  });
});

describe("defaultInpaintFrames", () => {
  it("is the partial filter's own length, rounded up onto the grid", () => {
    // 100..340 は閉区間なので 241 フレーム。241 = 8*30+1 でちょうど格子の上。
    expect(defaultInpaintFrames(100, 340)).toBe(241);
    // 100..339 は 240 フレーム → 241 へ切り上げ。
    expect(defaultInpaintFrames(100, 339)).toBe(241);
  });

  it("clamps to the API ceiling (D12: 481 is the hard limit)", () => {
    expect(defaultInpaintFrames(0, 999)).toBe(INPAINT_FRAMES_MAX);
  });

  it("never goes below the floor for a one-frame filter", () => {
    expect(defaultInpaintFrames(10, 10)).toBe(INPAINT_FRAMES_MIN);
  });
});

describe("placeInpaintWindow", () => {
  /** 部分フィルタ 100..340（241 フレーム）／リボン 0..600。 */
  const base = { filterStart: 100, filterEnd: 340, ribbonStart: 0, ribbonEnd: 600 };

  it("starts at the partial filter's head when there is room", () => {
    expect(placeInpaintWindow({ ...base, frames: 241 })).toEqual({
      windowStart: 100,
      windowEnd: 340,
      frames: 241,
      shiftedHead: false,
      coversFilter: true,
    });
  });

  it("keeps the requested length exactly — it never shortens the window", () => {
    // 長さはフレーム数欄が持つ唯一の正。ここで黙って縮めると、画面の数字と
    // 生成される長さが食い違う。
    const placed = placeInpaintWindow({ ...base, frames: 401 });
    expect(placed?.frames).toBe(401);
    expect(placed?.windowEnd).toBe(500);
  });

  it("shifts the window earlier when its tail would leave the ribbon", () => {
    // 部分フィルタが尻尾近く（500..560）で、窓 241 は 740 まで要る → 頭側へ。
    const placed = placeInpaintWindow({ ...base, filterStart: 500, filterEnd: 560, frames: 241 });
    expect(placed).toEqual({
      windowStart: 360, // 600 - 241 + 1
      windowEnd: 600,
      frames: 241,
      shiftedHead: true,
      coversFilter: true,
    });
  });

  it("returns null when the window cannot fit in the ribbon at all", () => {
    // リボンが 0..100（101 フレーム）しかないのに 241 を要求した。
    expect(placeInpaintWindow({ ...base, ribbonEnd: 100, frames: 241 })).toBeNull();
  });

  it("returns null when the partial filter does not overlap the ribbon", () => {
    // 対象動画の上に載っていない部分フィルタからはマスクを作れない。
    expect(placeInpaintWindow({ ...base, ribbonStart: 400, ribbonEnd: 600, frames: 9 })).toBeNull();
    expect(placeInpaintWindow({ ...base, ribbonStart: 0, ribbonEnd: 50, frames: 9 })).toBeNull();
  });

  it("reports coversFilter=false when the window is shorter than the filter", () => {
    // D3: 短くするのは自由。ブロックはせず、注意文を出すための印だけ返す。
    const placed = placeInpaintWindow({ ...base, frames: 121 });
    expect(placed).toMatchObject({ windowStart: 100, windowEnd: 220, coversFilter: false });
  });

  it("reports coversFilter=false when a short window leaves the filter's tail out", () => {
    // 部分フィルタ 560..600（41 フレーム）に対し、窓は 9 フレームしかない。
    // 起点は部分フィルタの頭のままなので 560..568 ——**頭は覆えるが尻尾が外**。
    // ずらし（shiftedHead）は起きていないので、覆えない理由は長さだけ。
    const placed = placeInpaintWindow({ ...base, filterStart: 560, filterEnd: 600, frames: 9 });
    expect(placed).toMatchObject({ windowStart: 560, windowEnd: 568, coversFilter: false });
  });

  it("clamps to the ribbon's head when the filter starts before it", () => {
    const placed = placeInpaintWindow({
      filterStart: 10,
      filterEnd: 200,
      ribbonStart: 50,
      ribbonEnd: 600,
      frames: 121,
    });
    expect(placed).toMatchObject({ windowStart: 50, windowEnd: 170, shiftedHead: false });
  });

  it("refuses broken input rather than guessing", () => {
    expect(placeInpaintWindow({ ...base, frames: 0 })).toBeNull();
    expect(placeInpaintWindow({ ...base, filterEnd: 50, frames: 9 })).toBeNull(); // 逆転
    expect(placeInpaintWindow({ ...base, ribbonEnd: -1, frames: 9 })).toBeNull();
    expect(placeInpaintWindow({ ...base, frames: Number.NaN })).toBeNull();
  });
});

describe("roundUpTo128", () => {
  it("rounds a real resolution up onto the canvas grid", () => {
    expect(roundUpTo128(1920)).toBe(1920);
    expect(roundUpTo128(1080)).toBe(1152);
    expect(roundUpTo128(768)).toBe(768);
    expect(roundUpTo128(1)).toBe(128);
  });

  it("keeps 'unknown' as 0 rather than inventing a size", () => {
    // 0 のまま返すからこそ `mediaInfoUnknown` のゲートまで届く。
    expect(roundUpTo128(0)).toBe(0);
    expect(roundUpTo128(-10)).toBe(0);
    expect(roundUpTo128(Number.NaN)).toBe(0);
  });
});
