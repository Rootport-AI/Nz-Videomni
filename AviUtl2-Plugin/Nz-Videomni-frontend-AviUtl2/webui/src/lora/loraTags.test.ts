import { describe, expect, it } from "vitest";
import {
  appendLoraTag,
  clampLoraAudioStrength,
  clampLoraStrength,
  formatLoraStrength,
  formatLoraTag,
  isValidLoraName,
  parseLoraPrompt,
  removeLoraTag,
  setLoraTagAudioStrength,
  setLoraTagStrength,
  stripLoraTagsByName,
} from "./loraTags";
import type { ParsedLoraTag } from "./loraTags";

/** `tags` is a plain array, so TS sees `tags[0]` as possibly-undefined
 * (`noUncheckedIndexedAccess`) even though every call site here parses a
 * prompt it just wrote and knows has exactly the tag(s) it's asking for. */
function firstTag(prompt: string): ParsedLoraTag {
  const tag = parseLoraPrompt(prompt).tags[0];
  if (!tag) throw new Error(`expected at least one tag in "${prompt}"`);
  return tag;
}

describe("parseLoraPrompt", () => {
  it("parses a single tag with an explicit strength and strips it from the body", () => {
    const parsed = parseLoraPrompt("a cat <lora:Pixar_Toon:1.5> riding a skateboard");
    expect(parsed.loras).toEqual([{ name: "Pixar_Toon", strength: 1.5 }]);
    expect(parsed.strippedPrompt).toBe("a cat riding a skateboard");
    expect(parsed.tags).toHaveLength(1);
    expect(parsed.tags[0]).toMatchObject({ name: "Pixar_Toon", strength: 1.5, valid: true });
  });

  it("defaults strength to 1.0 when omitted", () => {
    const parsed = parseLoraPrompt("<lora:Pixar_Toon> a cat");
    expect(parsed.loras).toEqual([{ name: "Pixar_Toon", strength: 1.0 }]);
    expect(parsed.strippedPrompt).toBe("a cat");
  });

  it("defaults strength to 1.0 when the strength segment is present but empty", () => {
    const parsed = parseLoraPrompt("<lora:Pixar_Toon:> a cat");
    expect(parsed.loras).toEqual([{ name: "Pixar_Toon", strength: 1.0 }]);
  });

  it("clamps strength above 2.0 down to 2.0", () => {
    const parsed = parseLoraPrompt("<lora:Foo:5.0> x");
    expect(parsed.loras).toEqual([{ name: "Foo", strength: 2.0 }]);
  });

  it("clamps strength below 0.05 up to 0.05", () => {
    const parsed = parseLoraPrompt("<lora:Foo:0.0> x");
    expect(parsed.loras).toEqual([{ name: "Foo", strength: 0.05 }]);
  });

  it("clamps a negative strength up to 0.05", () => {
    const parsed = parseLoraPrompt("<lora:Foo:-3> x");
    expect(parsed.loras).toEqual([{ name: "Foo", strength: 0.05 }]);
  });

  it.each(["../secret", "a/b", "a\\b"])(
    "leaves a tag with an invalid name (%s) untouched in the prompt body, and excludes it from loras[]",
    (badName) => {
      const prompt = `a cat <lora:${badName}:1.0> riding a skateboard`;
      const parsed = parseLoraPrompt(prompt);
      expect(parsed.loras).toEqual([]);
      expect(parsed.strippedPrompt).toBe(prompt);
      expect(parsed.tags).toHaveLength(1);
      expect(parsed.tags[0]).toMatchObject({ valid: false });
    },
  );

  it("parses multiple tags and strips all valid ones, keeping plain text intact", () => {
    const parsed = parseLoraPrompt("<lora:A:1.0> a cat <lora:B:0.5> riding a skateboard");
    expect(parsed.loras).toEqual([
      { name: "A", strength: 1.0 },
      { name: "B", strength: 0.5 },
    ]);
    expect(parsed.strippedPrompt).toBe("a cat riding a skateboard");
  });

  it("returns the exact same prompt reference when there are no tags at all", () => {
    const prompt = "a cat riding a skateboard";
    const parsed = parseLoraPrompt(prompt);
    expect(parsed.strippedPrompt).toBe(prompt);
    expect(parsed.loras).toEqual([]);
    expect(parsed.tags).toEqual([]);
  });

  it("mixes a valid and an invalid tag: only the valid one is stripped/collected", () => {
    const parsed = parseLoraPrompt("<lora:Good:1.0> text <lora:../bad:1.0> more text");
    expect(parsed.loras).toEqual([{ name: "Good", strength: 1.0 }]);
    expect(parsed.strippedPrompt).toBe("text <lora:../bad:1.0> more text");
  });
});

describe("parseLoraPrompt — audio_strength (3rd tag argument)", () => {
  it("parses a 3-argument tag's audio_strength alongside the video-side strength", () => {
    const parsed = parseLoraPrompt("<lora:X:0.8:0.3> a cat");
    expect(parsed.loras).toEqual([{ name: "X", strength: 0.8, audio_strength: 0.3 }]);
    expect(parsed.tags[0]).toMatchObject({ name: "X", strength: 0.8, audio_strength: 0.3, valid: true });
  });

  it("a 2-argument tag has no audio_strength key at all (not undefined-valued, absent)", () => {
    const parsed = parseLoraPrompt("<lora:Pixar_Toon:1.0> a cat");
    expect(parsed.loras).toEqual([{ name: "Pixar_Toon", strength: 1.0 }]);
    expect(parsed.loras[0]).not.toHaveProperty("audio_strength");
  });

  it("a bare (no strength at all) tag also has no audio_strength key", () => {
    const parsed = parseLoraPrompt("<lora:Pixar_Toon> a cat");
    expect(parsed.loras).toEqual([{ name: "Pixar_Toon", strength: 1.0 }]);
    expect(parsed.loras[0]).not.toHaveProperty("audio_strength");
  });

  it("clamps an audio_strength above 2.0 down to 2.0", () => {
    const parsed = parseLoraPrompt("<lora:X:1.0:5.0> a cat");
    expect(parsed.loras).toEqual([{ name: "X", strength: 1.0, audio_strength: 2.0 }]);
  });

  it("clamps a negative audio_strength up to 0.0 — 0 IS the floor, unlike video strength's 0.05", () => {
    const parsed = parseLoraPrompt("<lora:X:1.0:-1> a cat");
    expect(parsed.loras).toEqual([{ name: "X", strength: 1.0, audio_strength: 0.0 }]);
  });

  it("accepts audio_strength=0 (mute) as a valid, distinct value — not clamped away", () => {
    const parsed = parseLoraPrompt("<lora:X:0.8:0> a cat");
    expect(parsed.loras).toEqual([{ name: "X", strength: 0.8, audio_strength: 0.0 }]);
  });

  it("a non-numeric 3rd argument yields no audio_strength (new regex behavior: <lora:A:1.0:junk>)", () => {
    const parsed = parseLoraPrompt("<lora:A:1.0:junk> a cat");
    // The 2nd group ("1.0") is still the video strength; the unparseable 3rd
    // group ("junk") is dropped rather than blocking the match — the tag as a
    // whole still strips from the prompt body.
    expect(parsed.loras).toEqual([{ name: "A", strength: 1.0 }]);
    expect(parsed.loras[0]).not.toHaveProperty("audio_strength");
    expect(parsed.strippedPrompt).toBe("a cat");
  });

  it("BLOCKER regression: tag.index stays correct across multiple 3-argument tags (offset shifted by the new capture group)", () => {
    const prompt = "<lora:A:1.0:0.5> mid <lora:B:0.8:0.2> end";
    const parsed = parseLoraPrompt(prompt);
    expect(parsed.tags).toHaveLength(2);
    const [first, second] = parsed.tags;
    expect(first!.index).toBe(prompt.indexOf("<lora:A:1.0:0.5>"));
    expect(second!.index).toBe(prompt.indexOf("<lora:B:0.8:0.2>"));
    // Prove the indices are actionable, not just numerically plausible: editing
    // the SECOND tag via its reported index/raw must touch only that tag.
    const updated = setLoraTagStrength(prompt, second!, 0.5);
    expect(updated).toBe("<lora:A:1.0:0.5> mid <lora:B:0.5:0.2> end");
  });
});

describe("isValidLoraName / clampLoraStrength / formatLoraStrength", () => {
  it("rejects names with path separators or ..", () => {
    expect(isValidLoraName("Pixar_Toon")).toBe(true);
    expect(isValidLoraName("a/b")).toBe(false);
    expect(isValidLoraName("a\\b")).toBe(false);
    expect(isValidLoraName("../etc")).toBe(false);
    expect(isValidLoraName("")).toBe(false);
  });

  it("clamps to [0.05, 2.0]", () => {
    expect(clampLoraStrength(1.0)).toBe(1.0);
    expect(clampLoraStrength(2.5)).toBe(2.0);
    expect(clampLoraStrength(0.0)).toBe(0.05);
    expect(clampLoraStrength(-1)).toBe(0.05);
  });

  it("formats whole/one-decimal strengths with one decimal place", () => {
    expect(formatLoraStrength(1.0)).toBe("1.0");
    expect(formatLoraStrength(0.9)).toBe("0.9");
    expect(formatLoraStrength(0.1)).toBe("0.1");
  });

  it("formats the 0.05 clamp floor with two decimal places", () => {
    expect(formatLoraStrength(0.05)).toBe("0.05");
  });
});

describe("clampLoraAudioStrength", () => {
  it("clamps to [0.0, 2.0] — 0 IS a valid floor, unlike clampLoraStrength's 0.05", () => {
    expect(clampLoraAudioStrength(1.0)).toBe(1.0);
    expect(clampLoraAudioStrength(2.5)).toBe(2.0);
    expect(clampLoraAudioStrength(0.0)).toBe(0.0);
    expect(clampLoraAudioStrength(-1)).toBe(0.0);
  });
});

describe("formatLoraTag", () => {
  it("emits <lora:name:strength> when audioStrength is undefined", () => {
    expect(formatLoraTag("Pixar_Toon", 1.0)).toBe("<lora:Pixar_Toon:1.0>");
    expect(formatLoraTag("Pixar_Toon", 0.8, undefined)).toBe("<lora:Pixar_Toon:0.8>");
  });

  it("emits <lora:name:strength:audioStrength> when audioStrength is a number, video strength always explicit", () => {
    expect(formatLoraTag("X", 0.8, 0.3)).toBe("<lora:X:0.8:0.3>");
    expect(formatLoraTag("X", 1.0, 0)).toBe("<lora:X:1.0:0.0>");
  });

  it("never produces an empty video-strength segment (no '::') when audio is present", () => {
    const tag = formatLoraTag("X", 1.0, 1.0);
    expect(tag).not.toContain("::");
  });
});

describe("appendLoraTag", () => {
  it("appends a default-strength tag (3-arg, owner decision: card click always shows the audio slot) to a non-empty prompt", () => {
    const result = appendLoraTag("a cat riding a skateboard", "Pixar_Toon");
    expect(result).toBe("a cat riding a skateboard <lora:Pixar_Toon:1.0:1.0>");
  });

  it("appends without a leading double space when the prompt already ends in whitespace", () => {
    const result = appendLoraTag("a cat \n", "Pixar_Toon");
    expect(result).toBe("a cat \n<lora:Pixar_Toon:1.0:1.0>");
  });

  it("appends to an empty prompt with no leading space", () => {
    const result = appendLoraTag("", "Pixar_Toon");
    expect(result).toBe("<lora:Pixar_Toon:1.0:1.0>");
  });

  it("is a no-op (returns the same string) when a tag for that name is already present", () => {
    const prompt = "a cat <lora:Pixar_Toon:0.7> riding a skateboard";
    const result = appendLoraTag(prompt, "Pixar_Toon");
    expect(result).toBe(prompt);
  });
});

describe("chip <-> prompt sync (setLoraTagStrength / removeLoraTag)", () => {
  it("setLoraTagStrength rewrites just the targeted tag's strength, clamped", () => {
    const prompt = "a cat <lora:Pixar_Toon:1.0> riding a skateboard";
    const tag = firstTag(prompt);
    const updated = setLoraTagStrength(prompt, tag, 0.9);
    expect(updated).toBe("a cat <lora:Pixar_Toon:0.9> riding a skateboard");

    // Re-parsing the updated prompt reflects the new strength — the prompt
    // string is the single source of truth, not any retained chip state.
    expect(parseLoraPrompt(updated).loras).toEqual([{ name: "Pixar_Toon", strength: 0.9 }]);
  });

  it("setLoraTagStrength clamps beyond [0.05, 2.0]", () => {
    const prompt = "<lora:Pixar_Toon:1.0>";
    const tag = firstTag(prompt);
    expect(setLoraTagStrength(prompt, tag, 5)).toBe("<lora:Pixar_Toon:2.0>");
    expect(setLoraTagStrength(prompt, tag, -5)).toBe("<lora:Pixar_Toon:0.05>");
  });

  it("targets the correct occurrence among two tags with the same name+strength (disambiguated by index)", () => {
    const prompt = "<lora:Dup:1.0> middle <lora:Dup:1.0> end";
    const tags = parseLoraPrompt(prompt).tags;
    const first = tags[0]!;
    const second = tags[1]!;
    const updated = setLoraTagStrength(prompt, second, 0.5);
    expect(updated).toBe("<lora:Dup:1.0> middle <lora:Dup:0.5> end");
    // The first occurrence is untouched.
    expect(updated.indexOf("<lora:Dup:1.0>")).toBe(prompt.indexOf(first.raw));
  });

  it("removeLoraTag deletes the tag and collapses leftover whitespace", () => {
    const prompt = "a cat <lora:Pixar_Toon:1.0> riding a skateboard";
    const tag = firstTag(prompt);
    expect(removeLoraTag(prompt, tag)).toBe("a cat riding a skateboard");
  });

  it("removeLoraTag on the only content leaves an empty string", () => {
    const prompt = "<lora:Pixar_Toon:1.0>";
    const tag = firstTag(prompt);
    expect(removeLoraTag(prompt, tag)).toBe("");
  });

  it("BLOCKER fix: setLoraTagStrength PRESERVES an existing 3rd (audio) argument instead of dropping it", () => {
    const prompt = "<lora:X:0.8:0.3> a cat";
    const tag = firstTag(prompt);
    const updated = setLoraTagStrength(prompt, tag, 1.2);
    expect(updated).toBe("<lora:X:1.2:0.3> a cat");
    expect(parseLoraPrompt(updated).loras).toEqual([{ name: "X", strength: 1.2, audio_strength: 0.3 }]);
  });

  it("setLoraTagStrength on a 2-argument tag stays 2-argument (no audio introduced)", () => {
    const prompt = "<lora:Pixar_Toon:1.0>";
    const tag = firstTag(prompt);
    expect(setLoraTagStrength(prompt, tag, 0.5)).toBe("<lora:Pixar_Toon:0.5>");
  });
});

describe("setLoraTagAudioStrength (mute toggle primitive)", () => {
  it("sets a fresh audio strength on a 2-argument tag, adding the 3rd argument", () => {
    const prompt = "<lora:Pixar_Toon:0.8> a cat";
    const tag = firstTag(prompt);
    const updated = setLoraTagAudioStrength(prompt, tag, 0);
    expect(updated).toBe("<lora:Pixar_Toon:0.8:0.0> a cat");
  });

  it("clamps the value to [0.0, 2.0]", () => {
    const prompt = "<lora:X:1.0:0.5>";
    const tag = firstTag(prompt);
    expect(setLoraTagAudioStrength(prompt, tag, 5)).toBe("<lora:X:1.0:2.0>");
    expect(setLoraTagAudioStrength(prompt, tag, -5)).toBe("<lora:X:1.0:0.0>");
  });

  it("undefined removes the 3rd argument entirely — audio strength then follows the video strength", () => {
    const prompt = "<lora:X:0.8:0> a cat";
    const tag = firstTag(prompt);
    const updated = setLoraTagAudioStrength(prompt, tag, undefined);
    expect(updated).toBe("<lora:X:0.8> a cat");
    expect(parseLoraPrompt(updated).loras[0]).not.toHaveProperty("audio_strength");
  });

  it("roundtrips: mute (0) then unmute (1.0) then remove (undefined) returns to the plain 2-argument tag", () => {
    const original = "<lora:X:0.8> a cat";
    const muted = setLoraTagAudioStrength(original, firstTag(original), 0);
    expect(muted).toBe("<lora:X:0.8:0.0> a cat");

    const unmuted = setLoraTagAudioStrength(muted, firstTag(muted), 1.0);
    expect(unmuted).toBe("<lora:X:0.8:1.0> a cat");

    const cleared = setLoraTagAudioStrength(unmuted, firstTag(unmuted), undefined);
    expect(cleared).toBe(original);
  });

  it("leaves the video-side strength untouched", () => {
    const prompt = "<lora:X:0.8> a cat";
    const tag = firstTag(prompt);
    const updated = setLoraTagAudioStrength(prompt, tag, 0.4);
    expect(parseLoraPrompt(updated).loras).toEqual([{ name: "X", strength: 0.8, audio_strength: 0.4 }]);
  });
});

describe("stripLoraTagsByName (IC-LoRA UI redesign, 第5波)", () => {
  it("removes every valid tag whose name is in the set, collapsing leftover whitespace", () => {
    const prompt = "a cat <lora:canny-control:1.0> riding a skateboard";
    expect(stripLoraTagsByName(prompt, new Set(["canny-control"]))).toBe("a cat riding a skateboard");
  });

  it("removes multiple occurrences of names in the set, leaving others untouched", () => {
    const prompt = "<lora:canny-control:1.0> a cat <lora:Pixar_Toon:1.0> riding <lora:pose-control:0.8> a skateboard";
    const result = stripLoraTagsByName(prompt, new Set(["canny-control", "pose-control"]));
    expect(result).toBe("a cat <lora:Pixar_Toon:1.0> riding a skateboard");
  });

  it("is a no-op (returns the exact same string reference) when nothing matches", () => {
    const prompt = "a cat <lora:Pixar_Toon:1.0> riding a skateboard";
    expect(stripLoraTagsByName(prompt, new Set(["canny-control"]))).toBe(prompt);
  });

  it("is a no-op when the name set is empty", () => {
    const prompt = "a cat <lora:Pixar_Toon:1.0> riding a skateboard";
    expect(stripLoraTagsByName(prompt, new Set())).toBe(prompt);
  });

  it("never strips an invalid-named tag even if its raw text contains a matching substring", () => {
    const prompt = "a cat <lora:../canny-control:1.0> riding a skateboard";
    expect(stripLoraTagsByName(prompt, new Set(["canny-control"]))).toBe(prompt);
  });

  it("is idempotent — stripping the already-stripped result is a no-op", () => {
    const prompt = "a cat <lora:canny-control:1.0> riding a skateboard";
    const names = new Set(["canny-control"]);
    const once = stripLoraTagsByName(prompt, names);
    expect(stripLoraTagsByName(once, names)).toBe(once);
  });

  it("removes a 3-argument (video+audio strength) tag just as readily as a 2-argument one", () => {
    const prompt = "a cat <lora:canny-control:1.0:0.5> riding a skateboard";
    expect(stripLoraTagsByName(prompt, new Set(["canny-control"]))).toBe("a cat riding a skateboard");
  });
});
