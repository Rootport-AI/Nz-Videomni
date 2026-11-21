/**
 * Batch v2v-ref-long: the `POST /generate/chain` body of one row. Pure and
 * copy-on-write — the template is never mutated.
 */
import type { ChainClip, GenerateChainRequest, TransformerRole } from "../../api/types";
import { buildI2vLongTemplate, composeBatchPrompt } from "../batch-i2v-long/buildI2vLongPayload";
import type { PromptMode } from "../batch-i2v-long/buildI2vLongPayload";
import { REFERENCE_KINDS, buildReferenceBody, summarizeReference } from "../single/referenceRules";
import type { ReferenceKind, ReferenceRules, ReferenceState } from "../single/referenceRules";
import { usedReadyCards } from "./rowModel";
import type { RowPlan } from "./types";

/** The Chain form's request without the screen's material (opening material,
 * long audio, IC-LoRA reference video, end source). */
export function buildV2vRefLongTemplate(base: GenerateChainRequest): GenerateChainRequest {
  return buildI2vLongTemplate({ base });
}

export type UploadKind = "image" | "video" | "audio";

export function uploadKindOf(kind: ReferenceKind): UploadKind {
  return kind === "picture" ? "image" : kind;
}

/** The upload cache key: the same file used as a Video and as an Audio is two
 * uploads (the server keeps the ids per kind). */
export function uploadKey(uploadKind: UploadKind, path: string): string {
  return `${uploadKind}:${path}`;
}

export interface UploadTarget {
  key: string;
  uploadKind: UploadKind;
  path: string;
  fileName: string;
}

/** The row's files to upload: the opening video, then the used ready cards
 * of the planned clips (greyed cards are not uploaded); one entry per key. */
export function rowUploadTargets(plan: RowPlan, rules: ReferenceRules): UploadTarget[] {
  const out: UploadTarget[] = [];
  const seen = new Set<string>();
  const add = (uploadKind: UploadKind, path: string, fileName: string) => {
    const key = uploadKey(uploadKind, path);
    if (seen.has(key)) return;
    seen.add(key);
    out.push({ key, uploadKind, path, fileName });
  };
  if (plan.sourcePath !== null) add("video", plan.sourcePath, fileNameOf(plan.sourcePath));
  for (const clip of plan.clips) {
    for (const s of usedReadyCards(clip.refs, rules)) {
      if (s.filePath === null) continue;
      add(uploadKindOf(s.kind), s.filePath, s.fileName ?? fileNameOf(s.filePath));
    }
  }
  return out;
}

function fileNameOf(path: string): string {
  const idx = Math.max(path.lastIndexOf("\\"), path.lastIndexOf("/"));
  return idx === -1 ? path : path.slice(idx + 1);
}

function idFor(ids: ReadonlyMap<string, string>, uploadKind: UploadKind, path: string): string {
  const id = ids.get(uploadKey(uploadKind, path));
  if (id === undefined) throw new Error(`No upload id for ${uploadKind} ${path}`);
  return id;
}

/** A copy of `refs` with the used ready cards' upload ids filled in (greyed
 * cards are never sent, so they keep `id: null`). */
function withUploadIds(refs: ReferenceState, ids: ReadonlyMap<string, string>, rules: ReferenceRules): ReferenceState {
  const used = new Set(usedReadyCards(refs, rules));
  const slots = {} as ReferenceState["slots"];
  for (const kind of REFERENCE_KINDS) {
    slots[kind] = refs.slots[kind].map((s) =>
      used.has(s) && s.filePath !== null ? { ...s, id: idFor(ids, uploadKindOf(kind), s.filePath) } : s,
    );
  }
  return { ...refs, slots };
}

export interface RowRequestSettings {
  mainPromptMode: PromptMode;
  /** `fixedOverlapFrames` (22 on H3). */
  contextFrames: number;
  rules: ReferenceRules;
  role: TransformerRole | null;
}

/**
 * One row's request: the main prompt composed onto the shared one, each
 * planned clip's prompt composed onto the Chain card's (key left out when
 * empty), each clip's `references` from its cards (left out when empty; the
 * template clips' own `references`/`source_audio` are dropped), and
 * `source_video` when the row has an opening video. Throws when a ready card
 * or the opening video has no upload id in `ids`.
 */
export function buildRowRequest(
  template: GenerateChainRequest,
  plan: RowPlan,
  ids: ReadonlyMap<string, string>,
  settings: RowRequestSettings,
): GenerateChainRequest {
  const clips = template.clips.map((templateClip, i): ChainClip => {
    const planned = plan.clips[i];
    const clip: ChainClip = { ...templateClip };
    delete clip.references;
    delete clip.source_audio;
    if (!planned) return clip;
    const prompt = composeBatchPrompt(templateClip.prompt ?? "", planned.prompt, planned.promptMode);
    if (prompt.trim() === "") delete clip.prompt;
    else clip.prompt = prompt;
    const refs = withUploadIds(planned.refs, ids, settings.rules);
    const references = buildReferenceBody(summarizeReference(refs, settings.rules), refs.slots, settings.role, true);
    if (references.length > 0) clip.references = references;
    return clip;
  });

  const request: GenerateChainRequest = {
    ...template,
    prompt: composeBatchPrompt(template.prompt, plan.mainPrompt, settings.mainPromptMode),
    clips,
  };
  if (plan.sourcePath !== null) {
    request.source_video = {
      video_id: idFor(ids, "video", plan.sourcePath),
      context_frames: settings.contextFrames,
    };
  }
  return request;
}
