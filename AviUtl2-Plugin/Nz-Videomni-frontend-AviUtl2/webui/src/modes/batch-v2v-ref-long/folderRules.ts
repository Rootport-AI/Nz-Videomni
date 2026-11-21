/**
 * Batch v2v-ref-long: the rules of the folder inputs and of a scan result.
 * Pure — the scan itself (`fs.listFiles`, length probes) lives elsewhere.
 */
import type { UploadLimits } from "../../api/types";
import { REFERENCE_KINDS } from "../single/referenceRules";
import type { ReferenceKind, ReferenceRules } from "../single/referenceRules";
import type { BaseField, FolderField, FolderFrame, ScanInputs, ScanResult, ScannedField, ScannedFile } from "./types";

/** The trimmed path, `""` = the field is not used. */
export function filledPath(path: string): string {
  return path.trim();
}

/** Picture, Picture, Video, Audio. */
export function createDefaultFrame(id: number): FolderFrame {
  return {
    id,
    fields: [
      { id: 1, kind: "picture", path: "" },
      { id: 2, kind: "picture", path: "" },
      { id: 3, kind: "video", path: "" },
      { id: 4, kind: "audio", path: "" },
    ],
  };
}

/** The field's 1-based position among the frame's fields of its kind (= its
 * tag number), or `null` when the id is not in the frame. */
export function fieldSlot(frame: FolderFrame, fieldId: number): number | null {
  const field = frame.fields.find((f) => f.id === fieldId);
  if (!field) return null;
  let slot = 0;
  for (const f of frame.fields) {
    if (f.kind !== field.kind) continue;
    slot += 1;
    if (f.id === fieldId) return slot;
  }
  return null;
}

/** Fields per kind a new frame starts with ({@link createDefaultFrame}). */
const DEFAULT_FIELDS_OF_KIND: Record<ReferenceKind, number> = { picture: 2, video: 1, audio: 1 };

/** Whether the field is one a new frame starts with, by its position among
 * the frame's fields of its kind (Picture 1–2, Video 1, Audio 1). Its ❌ clears
 * the path; an added field's ❌ removes the field. */
export function isDefaultField(frame: FolderFrame, fieldId: number): boolean {
  const field = frame.fields.find((f) => f.id === fieldId);
  const slot = fieldSlot(frame, fieldId);
  return field !== undefined && slot !== null && slot <= DEFAULT_FIELDS_OF_KIND[field.kind];
}

export interface FrameCounts {
  picture: number;
  video: number;
  audio: number;
  total: number;
}

/** Fields with a path, per kind and in total. */
export function frameCounts(frame: FolderFrame): FrameCounts {
  const counts: FrameCounts = { picture: 0, video: 0, audio: 0, total: 0 };
  for (const f of frame.fields) {
    if (filledPath(f.path) === "") continue;
    counts[f.kind] += 1;
    counts.total += 1;
  }
  return counts;
}

export function maxOfKind(kind: ReferenceKind, rules: ReferenceRules): number {
  if (kind === "picture") return rules.maxImages;
  if (kind === "video") return rules.maxVideos;
  return rules.maxAudios;
}

/** Another field of `kind` keeps the frame within the per-kind and the total
 * limits (counted on fields, filled or not). */
export function canAddField(frame: FolderFrame, kind: ReferenceKind, rules: ReferenceRules): boolean {
  const ofKind = frame.fields.filter((f) => f.kind === kind).length;
  return ofKind < maxOfKind(kind, rules) && frame.fields.length < rules.maxTotal;
}

/** Adds a field of `kind` after the frame's last field of that kind (kind
 * order when there is none), so the array stays grouped by kind. */
export function withAddedField(frame: FolderFrame, kind: ReferenceKind): FolderFrame {
  const id = frame.fields.reduce((max, f) => Math.max(max, f.id), 0) + 1;
  const field: FolderField = { id, kind, path: "" };
  const order = REFERENCE_KINDS.indexOf(kind);
  let at = 0;
  frame.fields.forEach((f, i) => {
    if (REFERENCE_KINDS.indexOf(f.kind) <= order) at = i + 1;
  });
  const fields = [...frame.fields.slice(0, at), field, ...frame.fields.slice(at)];
  return { ...frame, fields };
}

/** Filled folders across every frame (the accordion heading). */
export function filledFolderCount(frames: readonly FolderFrame[]): number {
  return frames.reduce((sum, frame) => sum + frameCounts(frame).total, 0);
}

/** The frame clip `clip` (0-based) takes its references from: its own, else
 * the last one while `fillFollowing`, else none. */
export function frameIndexForClip(frameCount: number, fillFollowing: boolean, clip: number): number | null {
  if (clip < frameCount) return clip;
  if (fillFollowing && frameCount > 0) return frameCount - 1;
  return null;
}

/** The scanned field at (frame, kind, slot), or null when the frame has none. */
export function scannedFieldOf(scan: ScanResult, frame: number, kind: ReferenceKind, slot: number): ScannedField | null {
  return scan.frames[frame]?.find((f) => f.kind === kind && f.slot === slot) ?? null;
}

/** The files of a base field. */
export function filesOfField(scan: ScanResult, field: BaseField): readonly ScannedFile[] {
  if (field.type === "source") return scan.source;
  return scannedFieldOf(scan, field.frame, field.kind, field.slot)?.files ?? [];
}

/** The scanned fields in tie-break order: the opening video, then per frame
 * Picture 1, 2…, Video 1…, Audio 1…. */
export function fieldsInOrder(scan: ScanResult): BaseField[] {
  const out: BaseField[] = [{ type: "source" }];
  scan.frames.forEach((fields, frame) => {
    for (const kind of REFERENCE_KINDS) {
      const ofKind = fields.filter((f) => f.kind === kind).sort((a, b) => a.slot - b.slot);
      for (const f of ofKind) out.push({ type: "ref", frame, kind, slot: f.slot });
    }
  });
  return out;
}

/** The field with the most files (the first in {@link fieldsInOrder} on a
 * tie), or `null` when every field is empty. Every frame counts, whatever the
 * Chain form's clip count. */
export function pickBaseField(scan: ScanResult): BaseField | null {
  let best: BaseField | null = null;
  let bestCount = 0;
  for (const field of fieldsInOrder(scan)) {
    const count = filesOfField(scan, field).length;
    if (count > bestCount) {
      best = field;
      bestCount = count;
    }
  }
  return best;
}

export function rowCountOf(scan: ScanResult): number {
  const base = pickBaseField(scan);
  return base ? filesOfField(scan, base).length : 0;
}

/** The fields (opening video included, when given) that came back with no
 * usable file. */
export function emptyFieldsOf(scan: ScanResult): BaseField[] {
  return fieldsInOrder(scan).filter((field) => {
    if (field.type === "source" && filledPath(scan.inputs.videoDir) === "") return false;
    return filesOfField(scan, field).length === 0;
  });
}

/** The base folder's path, or `null` without a base field. */
export function baseFolderPath(scan: ScanResult): string | null {
  const base = pickBaseField(scan);
  if (!base) return null;
  if (base.type === "source") return filledPath(scan.inputs.videoDir);
  return scannedFieldOf(scan, base.frame, base.kind, base.slot)?.path ?? null;
}

function stripExtension(name: string): string {
  const idx = name.lastIndexOf(".");
  return idx > 0 ? name.slice(0, idx) : name;
}

/** The base folder's `rowIndex`-th file name without its extension, plus
 * `.mp4`; `null` when there is no such file. */
export function outFileNameOf(scan: ScanResult, rowIndex: number): string | null {
  const base = pickBaseField(scan);
  if (!base) return null;
  const file = filesOfField(scan, base)[rowIndex];
  return file ? `${stripExtension(file.name)}.mp4` : null;
}

/** Same as `batch-i2v-long/imageRows.ts`'s local copy. */
function joinPath(dir: string, name: string): string {
  const sep = dir.endsWith("\\") || dir.endsWith("/") ? "" : "\\";
  return `${dir}${sep}${name}`;
}

/** Same as `batch-i2v-long/imageRows.ts`'s local copy. */
function splitDir(path: string): { parent: string; base: string } {
  const trimmed = path.replace(/[\\/]+$/, "");
  const idx = Math.max(trimmed.lastIndexOf("\\"), trimmed.lastIndexOf("/"));
  if (idx === -1) return { parent: "", base: trimmed };
  return { parent: trimmed.slice(0, idx), base: trimmed.slice(idx + 1) };
}

/** A sibling of the base folder named `{its name}_out`. */
export function deriveV2vRefLongOutDir(baseFolder: string): string {
  const { parent, base } = splitDir(baseFolder);
  const name = `${base}_out`;
  return parent ? joinPath(parent, name) : name;
}

/** `frame:kind:slot:path` for every filled field. */
function filledFieldKeys(inputs: ScanInputs): string[] {
  const keys: string[] = [];
  inputs.frames.forEach((frame, index) => {
    for (const field of frame.fields) {
      const path = filledPath(field.path);
      if (path === "") continue;
      keys.push(`${index}:${field.kind}:${fieldSlot(frame, field.id) ?? 0}:${path}`);
    }
  });
  return keys.sort();
}

/** Whether a scan of `b` would read the same folders as one of `a`: the
 * opening-video folder, the frame count and every filled field (frame, kind,
 * position, path). Empty fields, `fillFollowing`, `joined` and the output
 * folder are not compared. */
export function sameScanInputs(a: ScanInputs, b: ScanInputs): boolean {
  if (filledPath(a.videoDir) !== filledPath(b.videoDir)) return false;
  if (a.frames.length !== b.frames.length) return false;
  const ka = filledFieldKeys(a);
  const kb = filledFieldKeys(b);
  return ka.length === kb.length && ka.every((key, i) => key === kb[i]);
}

/** The extensions a field picks up and its upload size cap (MB), from
 * `config.upload`. The opening video and Video use the video list; an Audio
 * field uses the audio list (which includes video containers). */
export function uploadRuleOf(
  field: ReferenceKind | "source",
  upload: UploadLimits,
): { extensions: readonly string[]; maxMb: number } {
  if (field === "picture") return { extensions: upload.allowed_image_extensions, maxMb: upload.max_image_size_mb };
  if (field === "audio") return { extensions: upload.allowed_audio_extensions, maxMb: upload.max_audio_size_mb };
  return { extensions: upload.allowed_video_extensions, maxMb: upload.max_video_size_mb };
}
