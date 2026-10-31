import type { NativeBridge } from "../bridge";
import { BridgeError } from "../bridge/types";
import { fileNameFromPath } from "../modes/single/keyframeUtils";
import type { NotesStrings, SelectionItem } from "./menuSelection";

/**
 * §3-4 #7 / #18 (台帳 §1-113, 2026-10-31): render the selected audio OR video
 * object's own sound over its ribbon range to a temp wav, through native's
 * `timeline.extractAudio` in SOLO mode (only the object's layer kept audible).
 * The three receivers — Create's A2V card (LTX), the Reference Audios card
 * (MiniMax H3) and Chain's audio card — all call this one function, so the
 * request shape and the failure classification live in exactly one place.
 *
 * The extracted wav covers exactly the ribbon range, so its length is the
 * ribbon's length: a trimmed object contributes the seconds its ribbon shows.
 */

/** The three user-facing failure kinds of an extraction. */
export type ExtractAudioFailure = "silent" | "timeout" | "failed";

/**
 * Pure: classify an extraction error. Native returns a single `EXTRACT_FAILED`
 * code for every failure and only the message tells them apart
 * (`native/src/bridge.cpp`'s `ExtractAudioWorker`): "…silent range" for a
 * muted/disabled range (or a video object whose sound is off or split to
 * another layer — solo then hears nothing); "audio render failed at frame N"
 * for a per-frame render failure whose dominant cause is the worker's render
 * timeout; setup failures ("rendering_scene_audio is not available" / "failed
 * to write wav file") and anything unclassifiable fall through to "failed".
 */
export function classifyExtractAudioError(err: unknown): ExtractAudioFailure {
  if (err instanceof BridgeError && err.code === "TIMEOUT") return "timeout";
  const message = err instanceof Error ? err.message : "";
  if (/silent range|no audio stream/i.test(message)) return "silent";
  if (/render failed/i.test(message)) return "timeout";
  return "failed";
}

/** Pure: the localized note for an extraction error (the three receivers
 * show the same note for the same failure). */
export function extractAudioFailureNote(err: unknown, notes: NotesStrings): string {
  switch (classifyExtractAudioError(err)) {
    case "silent":
      return notes.audioExtractSilent;
    case "timeout":
      return notes.audioExtractTimeout;
    case "failed":
      return notes.audioExtractFailed;
  }
}

/**
 * Serialization chain. Solo extraction temporarily disables every other layer
 * and restores a snapshot of their state afterwards; two overlapping runs could
 * restore each other's snapshot (the second one snapshots the first one's
 * "everything else disabled" state and puts it back last), leaving layers
 * disabled. So each call waits for the previous one to settle — success or
 * failure — before it starts. A rejected call never poisons the chain: the
 * tail is always a promise that resolves.
 */
let chainTail: Promise<void> = Promise.resolve();

export function extractObjectAudio(
  bridge: NativeBridge,
  item: SelectionItem,
): Promise<{ filePath: string; fileName: string; durationSec: number }> {
  const run = chainTail.then(async () => {
    // frameEnd is INCLUSIVE: native's `FindObjectByJob` walks objects with
    // `next = frameEnd + 1`, so the ribbon spans [frameStart, frameEnd].
    const extracted = await bridge.request("timeline.extractAudio", {
      layer: item.layer,
      frameStart: item.frameStart,
      frameCount: item.frameEnd - item.frameStart + 1,
      audioMode: "solo",
      soloKeepLayers: [item.layer],
    });
    return {
      filePath: extracted.filePath,
      fileName: fileNameFromPath(extracted.filePath),
      durationSec: extracted.durationSec,
    };
  });
  chainTail = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
}

/** Test-only: forget any in-flight chain so a hung run in one test cannot block
 * the next (same idea as `resetCreateLiveCommands()`). */
export function resetExtractObjectAudioChain(): void {
  chainTail = Promise.resolve();
}
