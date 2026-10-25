import { describe, expect, it } from "vitest";
import { singleAudioPrecheck, suggestFramesForSingleAudio } from "./audioFit";

// MiniMax H3's grid (17n+5, 22..481).
const H3_GRID = { step: 17, offset: 5, min: 22, max: 481 };

describe("suggestFramesForSingleAudio", () => {
  it("picks the largest grid count the audio covers (3.886 s at 24 fps = 93 frames -> 90)", () => {
    expect(suggestFramesForSingleAudio(3.886, 24, H3_GRID)).toBe(90);
  });

  it("an exact grid fit is kept (90 frames = 3.75 s)", () => {
    expect(suggestFramesForSingleAudio(3.75, 24, H3_GRID)).toBe(90);
  });

  it("audio shorter than the minimum gives the minimum; longer than the maximum is capped", () => {
    expect(suggestFramesForSingleAudio(0.5, 24, H3_GRID)).toBe(22);
    expect(suggestFramesForSingleAudio(60, 24, H3_GRID)).toBe(481);
  });

  it("the suggestion always passes the precheck unless the audio is below the minimum", () => {
    for (let sec = 1; sec <= 20; sec += 0.137) {
      const frames = suggestFramesForSingleAudio(sec, 24, H3_GRID);
      expect(singleAudioPrecheck(sec, frames, 24), `sec=${sec}`).toBe(true);
    }
    expect(singleAudioPrecheck(0.5, suggestFramesForSingleAudio(0.5, 24, H3_GRID), 24)).toBe(false);
  });
});

describe("singleAudioPrecheck", () => {
  it("passes when the audio is exactly as long as the clip, fails a hair short", () => {
    expect(singleAudioPrecheck(3.75, 90, 24)).toBe(true);
    expect(singleAudioPrecheck(3.74, 90, 24)).toBe(false);
    expect(singleAudioPrecheck(4, 90, 24)).toBe(true);
  });
});
