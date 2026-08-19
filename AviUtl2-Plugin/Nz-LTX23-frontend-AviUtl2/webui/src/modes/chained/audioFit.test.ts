import { describe, expect, it } from "vitest";
import { AUDIO_FIT_SAFETY_MARGIN_LATENTS, isAudioTooShortForChain, planAudioFit } from "./audioFit";
import type { AudioFitClip, AudioFitPlan } from "./audioFit";
import { audioLatentsAvailable, audioLatentsRequired, chainLayoutError } from "./chainUtils";
import { STAGE2_WINDOW_PRESETS } from "../../shell/tokenBudget";

const STANDARD = STAGE2_WINDOW_PRESETS.standard;

/** Every expected clip list / latent count below was cross-checked against the
 * backend (`Nz-LTX23-backend/.venv/Scripts/python.exe` importing `chain_math`):
 * `audio_latents_required([...], fps, kv=...)` for the requirement and
 * `compute_chain_layout(...)` for the layout verdict. The resolver's own
 * bookkeeping (which card absorbs the remainder, what gets dropped) is the only
 * part derived here. */
const intact = (id: string, numFrames: number): AudioFitClip => ({ id, numFrames, intact: true });
const pinned = (id: string, numFrames: number): AudioFitClip => ({ id, numFrames, intact: false });

const shape = (plan: AudioFitPlan) => plan.clips.map((c) => [c.sourceId, c.numFrames]);

const fit = (clips: AudioFitClip[], audioDurationSec: number, over: Partial<Parameters<typeof planAudioFit>[0]> = {}) =>
  planAudioFit({
    clips,
    audioDurationSec,
    fps: 24,
    overlapFrames: 3,
    targetClipFrames: 257,
    stage2Window: STANDARD,
    ...over,
  });

describe("planAudioFit", () => {
  // ① Worked example: two untouched cards, S=257, 22s of audio @24fps, kv=3.
  //
  //   avail  = round(22*25) - 1 margin        = 550 - 1 = 549
  //   m=2 at the floor: [257, 25]   -> 276  <= 549, so 2 fits
  //   m=3 at the floor: [257,257,25]-> 526  <= 549, so grow to 3
  //   m=4 at the floor: [257,257,257,25] -> 776 > 549, stop at 3
  //   tail: [257,257,41] -> 543 <= 549, [257,257,49] -> 551 > 549
  // (276 / 526 / 776 / 543 / 551 all read off the backend.)
  it("aligns intact cards to the slider value and lets the LAST one absorb the remainder", () => {
    const plan = fit([intact("c0", 257), intact("c1", 257)], 22.0);

    expect(plan.outcome).toBe("adjusted");
    expect(plan.changed).toBe(true);
    expect(shape(plan)).toEqual([
      ["c0", 257],
      ["c1", 257],
      [null, 41],
    ]);
    expect(plan.audioLatentsAvailable).toBe(550);
    expect(plan.audioLatentsUsed).toBe(543);
    expect(plan.unusedAudioSec).toBeCloseTo(7 / 25, 10);
    expect(chainLayoutError([257, 257, 41], 24, 3, STANDARD)).toBeNull();
  });

  it("leaves the one-latent safety margin unconsumed (the 422-avoidance rule)", () => {
    // 543 is the largest requirement <= 549; without the margin the resolver
    // would have been allowed to reach 550 exactly and a server measuring the
    // upload one frame shorter would 422.
    const plan = fit([intact("c0", 257), intact("c1", 257)], 22.0);
    expect(plan.audioLatentsUsed).toBeLessThanOrEqual(
      plan.audioLatentsAvailable - AUDIO_FIT_SAFETY_MARGIN_LATENTS,
    );
    expect(AUDIO_FIT_SAFETY_MARGIN_LATENTS).toBe(1);
  });

  // ② A touched card is frozen: length, position and existence.
  it("never touches a non-intact card — [pinned 49, intact 257] keeps the 49 exactly where it is", () => {
    //   avail = 549;  m=1 floor [49,25] -> 59;  m=2 floor [49,257,25] -> 309;
    //   m=3 floor [49,257,257,25] -> 559 > 549, so m=2.
    //   tail sweep: [49,257,249] -> 543 <= 549, [49,257,257] -> 551 > 549.
    const plan = fit([pinned("c0", 49), intact("c1", 257)], 22.0);

    expect(plan.outcome).toBe("adjusted");
    expect(shape(plan)).toEqual([
      ["c0", 49],
      ["c1", 257],
      [null, 249],
    ]);
    expect(plan.audioLatentsUsed).toBe(543);
    // The pinned card is still first and still 49 frames.
    expect(plan.clips[0]).toEqual({ sourceId: "c0", numFrames: 49 });
  });

  it("keeps a pinned card that sits at the END of the chain, and appends new cards after it", () => {
    const plan = fit([intact("c0", 257), pinned("c1", 49)], 22.0);
    // The remainder is absorbed by the last INTACT card, never by the pinned
    // one — here that is the newly appended card at the tail of the array.
    expect(plan.clips.some((c) => c.sourceId === "c1" && c.numFrames === 49)).toBe(true);
    expect(plan.clips.map((c) => c.sourceId)).toContain("c1");
    expect(chainLayoutError(plan.clips.map((c) => c.numFrames), 24, 3, STANDARD)).toBeNull();
  });

  // ③ Audio too short for even the smallest legal chain -> nothing proposed.
  it("cannotFit leaves the clip list completely untouched", () => {
    const clips = [intact("c0", 257), intact("c1", 257)];
    const plan = fit(clips, 1.0); // avail 25-1 = 24; the 2-clip minimum needs 276

    expect(plan.outcome).toBe("cannotFit");
    expect(plan.changed).toBe(false);
    expect(shape(plan)).toEqual([
      ["c0", 257],
      ["c1", 257],
    ]);
    expect(plan.unusedAudioSec).toBe(0);
  });

  it("cannotFit also covers 'the slider value alone is longer than the track' (8s @ S=257)", () => {
    // The smallest chain the rules allow here is [S=257, floor=25] = 276
    // latents = 11.04s of audio; 8s (available 200) cannot reach it. Lowering
    // the slider is the user's way out — the resolver never shrinks S itself.
    const plan = fit([intact("c0", 257), intact("c1", 257)], 8.0);
    expect(plan.outcome).toBe("cannotFit");
    expect(plan.changed).toBe(false);
    // ...and with a smaller slider value the very same audio DOES fit.
    const smaller = fit([intact("c0", 257), intact("c1", 257)], 8.0, { targetClipFrames: 89 });
    expect(smaller.outcome).toBe("adjusted");
    expect(smaller.audioLatentsUsed).toBeLessThanOrEqual(199);
    expect(chainLayoutError(smaller.clips.map((c) => c.numFrames), 24, 3, STANDARD)).toBeNull();
  });

  it("cannotFit when the audio duration is unknown/zero rather than guessing", () => {
    const plan = fit([intact("c0", 257), intact("c1", 257)], 0);
    expect(plan.outcome).toBe("cannotFit");
    expect(plan.changed).toBe(false);
  });

  // ④ Nothing the resolver is allowed to move.
  it("noFlexibleClips when every card has been touched", () => {
    const clips = [pinned("c0", 257), pinned("c1", 121)];
    const plan = fit(clips, 60.0);

    expect(plan.outcome).toBe("noFlexibleClips");
    expect(plan.changed).toBe(false);
    expect(shape(plan)).toEqual([
      ["c0", 257],
      ["c1", 121],
    ]);
    // The report is still populated so the UI can explain itself.
    expect(plan.audioLatentsAvailable).toBe(1500);
    expect(plan.audioLatentsUsed).toBe(audioLatentsRequired([257, 121], 24, 3));
  });

  // ⑤ 24-card ceiling.
  it("cappedAtMaxClips fills all 24 cards and reports the leftover", () => {
    // [257]*24 @24fps kv=3 needs 6018 latents (240.72s, backend-verified);
    // 300s of audio provides 7500.
    const plan = fit([intact("c0", 257), intact("c1", 257)], 300.0);

    expect(plan.outcome).toBe("cappedAtMaxClips");
    expect(plan.clips).toHaveLength(24);
    expect(plan.clips.every((c) => c.numFrames === 257)).toBe(true);
    expect(plan.clips[0]!.sourceId).toBe("c0");
    expect(plan.clips[1]!.sourceId).toBe("c1");
    expect(plan.clips.slice(2).every((c) => c.sourceId === null)).toBe(true);
    expect(plan.audioLatentsUsed).toBe(6018);
    expect(plan.unusedAudioSec).toBeCloseTo((7500 - 6018) / 25, 10);
  });

  it("the 24-card ceiling counts pinned cards too", () => {
    const clips = [pinned("p0", 121), ...Array.from({ length: 3 }, (_, i) => intact(`c${i}`, 257))];
    const plan = fit(clips, 300.0);
    expect(plan.clips).toHaveLength(24);
    expect(plan.clips[0]!).toEqual({ sourceId: "p0", numFrames: 121 });
  });

  // ⑥ Shrinking: surplus intact cards are dropped from the END, order kept.
  it("drops surplus intact cards from the end and preserves the order of the rest", () => {
    const clips = Array.from({ length: 8 }, (_, i) => intact(`c${i}`, 257));
    const plan = fit(clips, 22.0);

    expect(plan.outcome).toBe("adjusted");
    expect(shape(plan)).toEqual([
      ["c0", 257],
      ["c1", 257],
      ["c2", 41],
    ]);
  });

  // ⑦ A remainder smaller than one minimum-length clip stays unused.
  it("does not add a card for a remainder below the per-clip floor", () => {
    // avail = round(21*25) - 1 = 524. [257,257] needs 518 (fits);
    // [257,257,25] — the cheapest 3-card chain — needs 526 > 524, so the
    // leftover 0.28s has nowhere to go and is reported instead.
    const plan = fit([intact("c0", 257), intact("c1", 25)], 21.0);

    expect(plan.outcome).toBe("adjusted");
    expect(shape(plan)).toEqual([
      ["c0", 257],
      ["c1", 257],
    ]);
    expect(plan.audioLatentsUsed).toBe(518);
    expect(plan.unusedAudioSec).toBeCloseTo(7 / 25, 10);
  });

  it("alreadyFits when the current list is already the answer", () => {
    const plan = fit([intact("c0", 257), intact("c1", 257)], 21.0);
    expect(plan.outcome).toBe("alreadyFits");
    expect(plan.changed).toBe(false);
    expect(shape(plan)).toEqual([
      ["c0", 257],
      ["c1", 257],
    ]);
  });

  // ⑨ Idempotence: re-running on the resolver's own output changes nothing.
  it("is idempotent — re-applying the plan reports alreadyFits", () => {
    const first = fit(Array.from({ length: 8 }, (_, i) => intact(`c${i}`, 257)), 22.0);
    const materialised = first.clips.map((c, i) => intact(c.sourceId ?? `new${i}`, c.numFrames));
    const second = fit(materialised, 22.0);

    expect(second.outcome).toBe("alreadyFits");
    expect(second.changed).toBe(false);
    expect(shape(second)).toEqual(shape(first).map(([, frames], i) => [materialised[i]!.id, frames]));
  });

  it("re-applying a capped plan stays capped and stays unchanged", () => {
    const first = fit([intact("c0", 257), intact("c1", 257)], 300.0);
    const materialised = first.clips.map((c, i) => intact(c.sourceId ?? `new${i}`, c.numFrames));
    const second = fit(materialised, 300.0);
    expect(second.outcome).toBe("cappedAtMaxClips");
    expect(second.changed).toBe(false);
  });

  it("honours a caller-supplied minClips/maxClips", () => {
    const plan = fit([intact("c0", 257), intact("c1", 257)], 300.0, { maxClips: 5 });
    expect(plan.clips).toHaveLength(5);
    expect(plan.outcome).toBe("cappedAtMaxClips");
  });

  it("uses the form's ACTUAL overlap_frames, not a hard-coded 3", () => {
    // kv=8 raises the per-clip floor to 65 and changes the requirement, so the
    // same audio produces a different construction.
    const kv3 = fit([intact("c0", 257), intact("c1", 257)], 22.0, { overlapFrames: 3 });
    const kv8 = fit([intact("c0", 257), intact("c1", 257)], 22.0, { overlapFrames: 8 });
    expect(kv8.clips.map((c) => c.numFrames)).not.toEqual(kv3.clips.map((c) => c.numFrames));
    expect(chainLayoutError(kv8.clips.map((c) => c.numFrames), 24, 8, STANDARD)).toBeNull();
  });

  it("snaps and clamps the slider value onto the 8n+1 grid above the kv floor", () => {
    // 30 is not on the grid and 20 is below the kv=3 floor of 25.
    const snapped = fit([intact("c0", 257), intact("c1", 257)], 60.0, { targetClipFrames: 30 });
    expect(snapped.clips.every((c) => (c.numFrames - 1) % 8 === 0)).toBe(true);
    expect(snapped.clips.every((c) => c.numFrames >= 25)).toBe(true);

    const clamped = fit([intact("c0", 257), intact("c1", 257)], 60.0, { targetClipFrames: 5 });
    expect(clamped.clips.every((c) => c.numFrames >= 25)).toBe(true);
  });

  it("rejects a chain the SERVER would reject, rather than proposing it (29.97fps)", () => {
    // At 29.97 the stage-2 audio tiling makes most multi-clip constructions
    // illegal; whatever comes back must still pass the layout mirror.
    const plan = fit([intact("c0", 257), intact("c1", 257)], 40.0, { fps: 30000 / 1001 });
    if (plan.outcome !== "cannotFit" && plan.outcome !== "noFlexibleClips") {
      expect(chainLayoutError(plan.clips.map((c) => c.numFrames), 30000 / 1001, 3, STANDARD)).toBeNull();
    }
    // ...and the naive "just fill it" answer at this fps is NOT legal, so the
    // resolver really did have to work around the layout check.
    expect(chainLayoutError([257, 257], 30000 / 1001, 3, STANDARD)).not.toBeNull();
  });

  // ⑧ Property sweep.
  describe("properties across frame rates, overlaps and audio lengths", () => {
    const FPS_LIST = [24000 / 1001, 24, 25, 30000 / 1001, 30, 50, 60];
    const KV_LIST = [1, 3, 8];
    const DURATIONS = [5, 12, 22, 47, 90, 240];
    const TARGETS = [65, 121, 257];
    const CONFIGS: Array<{ name: string; clips: AudioFitClip[] }> = [
      { name: "2 intact", clips: [intact("a0", 257), intact("a1", 257)] },
      { name: "pinned head", clips: [pinned("b0", 49), intact("b1", 257)] },
      { name: "4 intact", clips: Array.from({ length: 4 }, (_, i) => intact(`d${i}`, 121)) },
      {
        name: "pinned head + tail",
        clips: [pinned("e0", 121), intact("e1", 257), intact("e2", 257), pinned("e3", 89)],
      },
    ];

    for (const window of ["standard", "high_resolution"] as const) {
      it(`holds for every fps x kv x duration x target combination (${window} window)`, () => {
        const stage2Window = STAGE2_WINDOW_PRESETS[window];
        let checked = 0;
        const outcomes: Record<string, number> = {};
        for (const fps of FPS_LIST) {
          for (const overlapFrames of KV_LIST) {
            for (const audioDurationSec of DURATIONS) {
              for (const targetClipFrames of TARGETS) {
                for (const config of CONFIGS) {
                  const plan = planAudioFit({
                    clips: config.clips,
                    audioDurationSec,
                    fps,
                    overlapFrames,
                    targetClipFrames,
                    stage2Window,
                  });
                  const label = `${config.name} fps=${fps} kv=${overlapFrames} dur=${audioDurationSec} S=${targetClipFrames} ${window}`;
                  checked += 1;
                  outcomes[plan.outcome] = (outcomes[plan.outcome] ?? 0) + 1;

                  // Pinned cards survive with their id, length and relative order.
                  const pinnedIn = config.clips.filter((c) => !c.intact);
                  const pinnedOut = plan.clips.filter((c) =>
                    pinnedIn.some((p) => p.id === c.sourceId),
                  );
                  expect(
                    pinnedOut.map((c) => [c.sourceId, c.numFrames]),
                    label,
                  ).toEqual(pinnedIn.map((c) => [c.id, c.numFrames]));

                  // Card count stays inside [2, 24].
                  expect(plan.clips.length, label).toBeGreaterThanOrEqual(2);
                  expect(plan.clips.length, label).toBeLessThanOrEqual(24);

                  if (plan.outcome === "cannotFit" || plan.outcome === "noFlexibleClips") {
                    // Nothing proposed -> the caller's own list comes straight back.
                    expect(
                      plan.clips.map((c) => [c.sourceId, c.numFrames]),
                      label,
                    ).toEqual(config.clips.map((c) => [c.id, c.numFrames]));
                    expect(plan.changed, label).toBe(false);
                    continue;
                  }

                  const frames = plan.clips.map((c) => c.numFrames);
                  // THE invariant: never propose something the server 422s.
                  expect(chainLayoutError(frames, fps, overlapFrames, stage2Window), label).toBeNull();
                  // ...and never propose something longer than the audio.
                  const required = audioLatentsRequired(frames, fps, overlapFrames);
                  expect(required, label).not.toBeNull();
                  expect(required!, label).toBeLessThanOrEqual(
                    audioLatentsAvailable(audioDurationSec) - AUDIO_FIT_SAFETY_MARGIN_LATENTS,
                  );
                  expect(plan.audioLatentsUsed, label).toBe(required);
                  expect(plan.unusedAudioSec, label).toBeGreaterThanOrEqual(0);
                  // Every card is a legal 8n+1 num_frames within [9, 481].
                  for (const nf of frames) {
                    expect((nf - 1) % 8, label).toBe(0);
                    expect(nf, label).toBeGreaterThanOrEqual(9);
                    expect(nf, label).toBeLessThanOrEqual(481);
                  }
                }
              }
            }
          }
        }
        expect(checked).toBe(
          FPS_LIST.length * KV_LIST.length * DURATIONS.length * TARGETS.length * CONFIGS.length,
        );
        // Non-vacuity: the invariants above are only worth anything if the
        // sweep actually PRODUCES plans. Measured distribution over the 1,512
        // combinations per window: ~900 adjusted, ~280 capped, ~330 cannotFit —
        // and both non-integer frame rates contribute successful plans, which
        // is exactly the case the layout-check retry loop exists for.
        expect(outcomes.adjusted ?? 0).toBeGreaterThan(500);
        expect(outcomes.cappedAtMaxClips ?? 0).toBeGreaterThan(100);
      });
    }

    it("every successful plan is a fixed point (re-running it changes nothing)", () => {
      for (const fps of FPS_LIST) {
        for (const overlapFrames of KV_LIST) {
          for (const audioDurationSec of DURATIONS) {
            const base = {
              audioDurationSec,
              fps,
              overlapFrames,
              targetClipFrames: 257,
              stage2Window: STANDARD,
            };
            const first = planAudioFit({ clips: [intact("c0", 257), intact("c1", 257)], ...base });
            if (first.outcome === "cannotFit" || first.outcome === "noFlexibleClips") continue;
            const materialised = first.clips.map((c, i) => intact(c.sourceId ?? `new${i}`, c.numFrames));
            const second = planAudioFit({ clips: materialised, ...base });
            const label = `fps=${fps} kv=${overlapFrames} dur=${audioDurationSec}`;
            expect(second.changed, label).toBe(false);
            expect(second.clips.map((c) => c.numFrames), label).toEqual(first.clips.map((c) => c.numFrames));
          }
        }
      }
    });
  });
});

describe("isAudioTooShortForChain", () => {
  it("is true only when the audio cannot cover [current first clip, one minimum clip]", () => {
    // [257, 25] @24fps kv=3 needs 276 latents = 11.04s (backend-verified).
    expect(isAudioTooShortForChain(257, 11.0, 24, 3)).toBe(true); // available 275
    expect(isAudioTooShortForChain(257, 11.04, 24, 3)).toBe(false); // available 276
    expect(isAudioTooShortForChain(257, 30.0, 24, 3)).toBe(false);
  });

  it("uses the CURRENT first clip's length, so a short first clip needs less audio", () => {
    // [49, 25] @24fps kv=3 needs 59 latents = 2.36s.
    expect(isAudioTooShortForChain(49, 2.3, 24, 3)).toBe(true);
    expect(isAudioTooShortForChain(49, 2.4, 24, 3)).toBe(false);
  });

  it("gives the same verdict for every seam overlap — the growing floor exactly cancels the growing seam", () => {
    // Not an accident: the floor is `8*kv+1` px = `kv+1` video latents, and the
    // seam eats `kv` of them, so the minimum second clip always contributes
    // exactly ONE latent frame no matter what kv is. Verified against the
    // backend: `audio_latents_required([257, 8*kv+1], 24.0, kv=kv)` is 276 for
    // every kv in 1..8.
    for (const kv of [1, 2, 3, 4, 5, 6, 7, 8]) {
      expect(isAudioTooShortForChain(257, 11.0, 24, kv)).toBe(true); // available 275 < 276
      expect(isAudioTooShortForChain(257, 11.04, 24, kv)).toBe(false); // available 276
    }
  });

  it("defers to the server (false) when the requirement cannot be computed", () => {
    // A 17-frame first clip is below the kv=3 floor, so there is no
    // requirement to compare against — this is not the check that should block.
    expect(isAudioTooShortForChain(17, 0.1, 24, 3)).toBe(false);
  });

  it("falls back to 24fps for a falsy frame rate", () => {
    expect(isAudioTooShortForChain(257, 11.0, 0, 3)).toBe(isAudioTooShortForChain(257, 11.0, 24, 3));
  });
});
