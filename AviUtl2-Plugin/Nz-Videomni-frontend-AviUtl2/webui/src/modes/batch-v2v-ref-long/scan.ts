/**
 * Batch v2v-ref-long: the folder scan — one `fs.listFiles` per (folder, kind)
 * and a length probe per video/audio file, one at a time.
 */
import type { UploadLimits } from "../../api/types";
import type { NativeBridge } from "../../bridge";
import { compareByName } from "../batch-i2v-long/imageRows";
import type { ReferenceKind, ReferenceRules } from "../single/referenceRules";
import { measureDuration } from "../single/useReferenceSlots";
import { fieldSlot, filledPath, uploadRuleOf } from "./folderRules";
import type { ScanInputs, ScanResult, ScannedField, ScannedFile } from "./types";

type ListingKind = ReferenceKind | "source";

/** A deep copy, so later edits of the inputs never reach the result. */
function copyInputs(inputs: ScanInputs): ScanInputs {
  return {
    videoDir: inputs.videoDir,
    frames: inputs.frames.map((frame) => ({ ...frame, fields: frame.fields.map((field) => ({ ...field })) })),
  };
}

/**
 * Lists the opening-video folder and every filled reference field (natural
 * name order), then measures the videos' and audios' lengths. A missing
 * folder lists as empty; an `fs.listFiles` rejection is rethrown.
 */
export async function scanFolders(
  bridge: NativeBridge,
  inputs: ScanInputs,
  upload: UploadLimits,
  rules: ReferenceRules,
  onProgress: (done: number, total: number) => void,
): Promise<ScanResult> {
  const used = copyInputs(inputs);

  // The source and Video share the video extension list, so they share a listing.
  const listings = new Map<string, Promise<ScannedFile[]>>();
  const list = (kind: ListingKind, folderPath: string): Promise<ScannedFile[]> => {
    const category = kind === "source" ? "video" : kind;
    const key = `${category}|${folderPath}`;
    let pending = listings.get(key);
    if (!pending) {
      pending = bridge
        .request("fs.listFiles", { folderPath, extensions: [...uploadRuleOf(kind, upload).extensions] })
        .then((listing) =>
          [...listing.files]
            .sort(compareByName)
            .map((f): ScannedFile => ({ name: f.name, path: f.path, sizeBytes: f.sizeBytes, durationSec: null })),
        );
      listings.set(key, pending);
    }
    return pending;
  };

  // Listed one after another (in field order), so a rejection surfaces once.
  const videoDir = filledPath(used.videoDir);
  const sourceFiles = videoDir === "" ? [] : await list("source", videoDir);
  const frameFields: ScannedField[][] = [];
  for (const frame of used.frames) {
    const fields: ScannedField[] = [];
    for (const field of frame.fields) {
      const path = filledPath(field.path);
      if (path === "") continue;
      fields.push({ kind: field.kind, slot: fieldSlot(frame, field.id) ?? 0, path, files: await list(field.kind, path) });
    }
    frameFields.push(fields);
  }

  // One probe per path; a path used as an Audio is measured as one (it adds
  // the wav-header fallback).
  const toMeasure = new Map<string, "video" | "audio">();
  for (const f of sourceFiles) if (!toMeasure.has(f.path)) toMeasure.set(f.path, "video");
  for (const fields of frameFields) {
    for (const field of fields) {
      if (field.kind === "picture") continue;
      for (const f of field.files) {
        if (field.kind === "audio" || !toMeasure.has(f.path)) toMeasure.set(f.path, field.kind);
      }
    }
  }
  const durations = new Map<string, number | null>();
  const total = toMeasure.size;
  let done = 0;
  if (total > 0) onProgress(done, total);
  for (const [path, kind] of toMeasure) {
    durations.set(path, await measureDuration(bridge, kind, path, undefined));
    done += 1;
    onProgress(done, total);
  }

  const withDuration = (f: ScannedFile): ScannedFile => ({ ...f, durationSec: durations.get(f.path) ?? null });
  return {
    inputs: used,
    source: sourceFiles.map(withDuration),
    frames: frameFields.map((fields) =>
      fields.map((field) => ({ ...field, files: field.kind === "picture" ? field.files : field.files.map(withDuration) })),
    ),
    rules,
  };
}
