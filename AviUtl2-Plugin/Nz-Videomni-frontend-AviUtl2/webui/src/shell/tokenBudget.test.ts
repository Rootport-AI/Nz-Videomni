import { describe, expect, it } from "vitest";
import {
  CHAIN_COMFORT_TOKEN_BUDGET,
  CHAIN_STAGE1_COMFORT_TOKEN_BUDGET,
  chainComfortAxisMax,
  chainComfortSize16x9,
  chainStage1Tokens,
  chainWindowBudgetMarkers,
  chainWindowTokens,
  isChainStage1OverBudget,
  isChainWindowOverBudget,
  resolveChainComfortBudget,
  STAGE2_WINDOW_OPTIONS,
  STAGE2_WINDOW_PRESETS,
  stage2AdvancePixelFrames,
  stage2AdvanceSeconds,
  stage2MaxContextFrames,
} from "./tokenBudget";

describe("tokenBudget", () => {
  // ── pre-existing stage-2 window coverage (must stay green) ─────────────────
  describe("chainWindowTokens / isChainWindowOverBudget", () => {
    it("computes (width//32) * (height//32) * vTile", () => {
      expect(chainWindowTokens(1152, 1536, STAGE2_WINDOW_PRESETS.standard.vTile)).toBe(
        Math.floor(1152 / 32) * Math.floor(1536 / 32) * 22,
      );
    });

    it("flags a resolution/window pair over CHAIN_COMFORT_TOKEN_BUDGET", () => {
      // 1600x1600 standard: (1600//32)*(1600//32)*22 = 50*50*22 = 55000 > 40000.
      expect(isChainWindowOverBudget(1600, 1600, "standard")).toBe(true);
      expect(chainWindowTokens(1600, 1600, STAGE2_WINDOW_PRESETS.standard.vTile)).toBeGreaterThan(
        CHAIN_COMFORT_TOKEN_BUDGET,
      );
      // 512x512 standard: (512//32)*(512//32)*22 = 16*16*22 = 5632 < 40000.
      expect(isChainWindowOverBudget(512, 512, "standard")).toBe(false);
    });

    it("compares against a served budget when one is passed", () => {
      // 1280x768 standard = 21,120 tokens: comfortable at 40,000, not at 20,000.
      expect(isChainWindowOverBudget(1280, 768, "standard")).toBe(false);
      expect(isChainWindowOverBudget(1280, 768, "standard", 20_000)).toBe(true);
      expect(isChainWindowOverBudget(1600, 1600, "standard", 60_000)).toBe(false);
    });
  });

  // ── the served budget (2026-08-12) ──────────────────────────────────────────
  describe("resolveChainComfortBudget", () => {
    it("takes any positive finite number the server sent", () => {
      expect(resolveChainComfortBudget(30_000)).toBe(30_000);
      expect(resolveChainComfortBudget(60_000)).toBe(60_000);
    });

    it("falls back to the mirrored constant for anything unusable", () => {
      // An older backend simply has no such key; 0 would put EVERY resolution
      // over budget and pin every guide at 0px, so it is a fallback too.
      for (const published of [undefined, null, 0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
        expect(resolveChainComfortBudget(published)).toBe(CHAIN_COMFORT_TOKEN_BUDGET);
      }
      expect(CHAIN_COMFORT_TOKEN_BUDGET).toBe(40_000);
    });
  });

  // ── resolution guides (2026-08-12) ─────────────────────────────────────────
  describe("chainComfortSize16x9", () => {
    it("is 1792x1024 on standard and 1920x1088 on the shorter step", () => {
      expect(chainComfortSize16x9("standard")).toEqual({ width: 1792, height: 1024 });
      expect(chainComfortSize16x9("high_resolution")).toEqual({ width: 1920, height: 1088 });
    });

    it("lands inside the budget — 39,424 and 38,760 tokens", () => {
      // 38,760 is the backend TOKEN_TABLE's own high_resolution row; 39,424 was
      // added to it alongside this guide (tests/test_stage2_window.py).
      expect(chainWindowTokens(1792, 1024, STAGE2_WINDOW_PRESETS.standard.vTile)).toBe(39_424);
      expect(chainWindowTokens(1920, 1088, STAGE2_WINDOW_PRESETS.high_resolution.vTile)).toBe(38_760);
      expect(isChainWindowOverBudget(1792, 1024, "standard")).toBe(false);
      expect(isChainWindowOverBudget(1920, 1088, "high_resolution")).toBe(false);
    });

    it("every neighbouring 64-step is over budget — the guide really is the last comfortable pair", () => {
      expect(isChainWindowOverBudget(1856, 1024, "standard")).toBe(true);
      expect(isChainWindowOverBudget(1792, 1088, "standard")).toBe(true);
      expect(isChainWindowOverBudget(1984, 1088, "high_resolution")).toBe(true);
      expect(isChainWindowOverBudget(1920, 1152, "high_resolution")).toBe(true);
    });

    it("moves with the budget", () => {
      const tight = chainComfortSize16x9("standard", 30_000);
      expect(isChainWindowOverBudget(tight.width, tight.height, "standard", 30_000)).toBe(false);
      expect(tight.width).toBeLessThan(1792);
    });
  });

  describe("chainComfortAxisMax", () => {
    it("inverts the budget for one axis given the other one's value", () => {
      // (1088//32)=34 cells opposite -> floor(40000/(22*34)) = 53 cells -> 1696
      // -> floored onto the 64 grid = 1664.
      expect(chainComfortAxisMax(1088, "standard")).toBe(1664);
      // The manual-QA case: width 1920 on standard leaves height 960.
      expect(chainComfortAxisMax(1920, "standard")).toBe(960);
    });

    it("is self-consistent with the recommended pair (on the 64 grid)", () => {
      // The height guide inverted back gives the width guide again — the pair
      // is a fixed point. TRUE ONLY on the default 64 grid: with a reference
      // video active (128) the two floorings no longer commute, e.g.
      // high_resolution's 1024 inverts to 2048, not 1920. That asymmetry is
      // harmless (flooring only ever moves a guide to the safe side) but it
      // must not be asserted, hence the grid argument stays omitted here.
      for (const window of STAGE2_WINDOW_OPTIONS) {
        const comfort = chainComfortSize16x9(window);
        expect(chainComfortAxisMax(comfort.height, window)).toBe(comfort.width);
      }
    });

    it("always returns a multiple of the grid", () => {
      for (const other of [256, 512, 768, 1024, 1088, 1600, 1920, 2560]) {
        expect(chainComfortAxisMax(other, "standard") % 64).toBe(0);
        expect(chainComfortAxisMax(other, "high_resolution") % 64).toBe(0);
        expect(chainComfortAxisMax(other, "standard", CHAIN_COMFORT_TOKEN_BUDGET, 128) % 128).toBe(0);
      }
    });

    it("floors onto the 128 grid a reference video puts the form on", () => {
      // §1-15: the shorter step's 1088 height guide becomes a reachable 1024.
      expect(chainComfortSize16x9("high_resolution", CHAIN_COMFORT_TOKEN_BUDGET, 128)).toEqual({
        width: 1920,
        height: 1024,
      });
    });

    it("returns 0 below one full 32px cell instead of dividing by zero", () => {
      expect(chainComfortAxisMax(31, "standard")).toBe(0);
      expect(chainComfortAxisMax(0, "standard")).toBe(0);
    });
  });

  describe("chainWindowBudgetMarkers", () => {
    it("carries the recommended pair regardless of the current size", () => {
      const markers = chainWindowBudgetMarkers(1280, 768, "standard");
      expect(markers.comfortWidth).toBe(1792);
      expect(markers.comfortHeight).toBe(1024);
      // Comfortable on both axes -> nothing red.
      expect(markers.limitWidth).toBeNull();
      expect(markers.limitHeight).toBeNull();
    });

    it("shows the OTHER axis's ceiling once an axis passes its own guide", () => {
      const wide = chainWindowBudgetMarkers(1920, 768, "standard");
      expect(wide.limitHeight).toBe(960);
      expect(wide.limitWidth).toBeNull();
      const tall = chainWindowBudgetMarkers(1280, 1088, "standard");
      expect(tall.limitWidth).toBe(1664);
      expect(tall.limitHeight).toBeNull();
    });

    it("appears STRICTLY above the guide, never on it (`>`, not `>=`)", () => {
      // Sitting exactly on the recommended pair is the intended comfortable
      // state — a `>=` here would paint red over it. One 64-step past it does
      // warn. Both axes, so neither direction can regress alone.
      expect(chainWindowBudgetMarkers(1792, 768, "standard").limitHeight).toBeNull();
      expect(chainWindowBudgetMarkers(1856, 768, "standard").limitHeight).not.toBeNull();
      expect(chainWindowBudgetMarkers(1280, 1024, "standard").limitWidth).toBeNull();
      expect(chainWindowBudgetMarkers(1280, 1088, "standard").limitWidth).not.toBeNull();
    });

    it("follows the selected step", () => {
      expect(chainWindowBudgetMarkers(1280, 768, "high_resolution")).toEqual({
        comfortWidth: 1920,
        comfortHeight: 1088,
        limitWidth: null,
        limitHeight: null,
      });
    });

    it("follows a served budget", () => {
      const markers = chainWindowBudgetMarkers(1280, 768, "standard", 30_000);
      expect(markers.comfortWidth).toBeLessThan(1792);
      expect(markers.comfortWidth).toBe(chainComfortSize16x9("standard", 30_000).width);
    });
  });

  describe("stage2AdvancePixelFrames / stage2AdvanceSeconds", () => {
    it("advances vAdv * 8 pixel frames", () => {
      expect(stage2AdvancePixelFrames("standard")).toBe(18 * 8);
      expect(stage2AdvancePixelFrames("high_resolution")).toBe(12 * 8);
    });

    it("converts to seconds at a given frame rate, 0 for non-positive rates", () => {
      expect(stage2AdvanceSeconds("standard", 24)).toBeCloseTo((18 * 8) / 24, 6);
      expect(stage2AdvanceSeconds("standard", 0)).toBe(0);
      expect(stage2AdvanceSeconds("standard", -1)).toBe(0);
    });
  });

  describe("stage2MaxContextFrames", () => {
    it("matches the documented per-preset ceilings", () => {
      expect(stage2MaxContextFrames("standard")).toBe(161);
      expect(stage2MaxContextFrames("high_resolution")).toBe(137);
    });
  });

  // ── §1-15 stage-1-with-reference budget (F6) ────────────────────────────────
  // Pinned against `Nz-Videomni/tests/test_chain_math_reference.py`'s
  // `test_chain_stage1_tokens_*` cases so the TS mirror stays numerically
  // identical to `chain_math.chain_stage1_tokens`.
  describe("chainStage1Tokens", () => {
    it("1152x1536, vLatent=46, refScale=2 -> 24840 (research-note anchor: 361f ceiling)", () => {
      expect(chainStage1Tokens(1152, 1536, 46, 2)).toBe(24_840);
    });

    it("1152x1536, vLatent=61, refScale=2 -> 32940 (research-note anchor: 481f/scale2)", () => {
      expect(chainStage1Tokens(1152, 1536, 61, 2)).toBe(32_940);
    });

    it("refScale=1 (deblur) contributes exactly the bare spatial cost, doubling the segment", () => {
      for (const [width, height, vLatent] of [
        [1152, 1536, 46],
        [768, 512, 22],
        [1024, 1024, 61],
      ] as const) {
        const bare = chainStage1Tokens(width, height, vLatent);
        expect(chainStage1Tokens(width, height, vLatent, 1)).toBe(2 * bare);
      }
    });

    it("no refScale -> no reference contribution at all", () => {
      // 1152x1536 -> (576//32)*(768//32) = 18*24 = 432 spatial patches.
      expect(chainStage1Tokens(1152, 1536, 46)).toBe(432 * 46);
      expect(chainStage1Tokens(1152, 1536, 46)).toBe(19_872);
      expect(chainStage1Tokens(1152, 1536, 46, undefined)).toBe(19_872);
      expect(chainStage1Tokens(1152, 1536, 46, 0)).toBe(19_872);
    });

    it("pins the integer-division order at an off-128-grid width (backend: test_chain_stage1_tokens_division_order_is_floor_at_each_step)", () => {
      // 1150//2 = 575; 575//32 = 17 (not 1150/64 = 17.97 rounded).
      expect(chainStage1Tokens(1150, 1536, 10)).toBe(17 * 24 * 10);
      // ref: (575//2)//32 = 287//32 = 8, (768//2)//32 = 12.
      expect(chainStage1Tokens(1150, 1536, 10, 2)).toBe((17 * 24 + 8 * 12) * 10);
    });
  });

  describe("CHAIN_STAGE1_COMFORT_TOKEN_BUDGET", () => {
    it("is the provisional 25000, bracketing the two research-note anchors", () => {
      expect(CHAIN_STAGE1_COMFORT_TOKEN_BUDGET).toBe(25_000);
      expect(chainStage1Tokens(1152, 1536, 46, 2)).toBeLessThan(25_000);
      // 328 stage-1 spatial patches x 61 latent frames x 1.25 = 25,010.
      expect(Math.trunc(328 * 61 * 1.25)).toBeGreaterThan(CHAIN_STAGE1_COMFORT_TOKEN_BUDGET);
    });
  });

  describe("isChainStage1OverBudget", () => {
    it("converts maxClipFrames to vLatent internally and compares against the budget", () => {
      // v_latent_frames(361) = 46 -> 24,840 < 25,000 -> not over budget.
      expect(isChainStage1OverBudget(1152, 1536, 361, 2)).toBe(false);
      // v_latent_frames(481) = 61 -> 32,940 > 25,000 -> over budget.
      expect(isChainStage1OverBudget(1152, 1536, 481, 2)).toBe(true);
      // No reference at all: a much shorter clip stays comfortably under budget
      // (432 spatial patches x vLatentFrames(121)=16 = 6,912).
      expect(isChainStage1OverBudget(1152, 1536, 121)).toBe(false);
    });
  });

  // §1-19 (2026-08-11): exposure-prevention guard. The backend gained a third
  // stage-2 preset, `"full_length"`, that Single/Batch A2V send as a fixed
  // wire literal (`modes/batch/buildA2vChainPayload.ts`) — it must NEVER show
  // up here, because `Stage2Window`/`STAGE2_WINDOW_OPTIONS`/
  // `STAGE2_WINDOW_PRESETS` are what the Chained screen's dropdown and
  // `useRetakeForm.ts`'s `retakeMaxWindowPx` window-size ceiling are built
  // from. If someone "helpfully" mirrors the new backend preset into this
  // file, one of these two assertions goes red.
  describe("full_length exposure guard (§1-19)", () => {
    it("STAGE2_WINDOW_OPTIONS stays exactly the 2 UI-selectable presets", () => {
      expect(STAGE2_WINDOW_OPTIONS).toEqual(["standard", "high_resolution"]);
    });

    it("STAGE2_WINDOW_PRESETS stays exactly 2 entries (no full_length key)", () => {
      expect(Object.keys(STAGE2_WINDOW_PRESETS)).toHaveLength(2);
      expect(Object.keys(STAGE2_WINDOW_PRESETS).sort()).toEqual(["high_resolution", "standard"]);
    });
  });
});
