/**
 * Shared placeholder artwork for source-input thumbnail slots (the 64×64
 * `.keyframe-thumb` square). Inlined as SVG data URIs so no asset file or
 * network fetch is needed, and so the same glyph can back both Chain's
 * unified source input (`SourceInputPanel.tsx`) and Create's IC-LoRA / A2V
 * source cards (`GenerationForm.tsx`).
 *
 * The dark palette (#2a2a33 / #3d3d47 / #c7cad1) is hardcoded to match the
 * original `SourceInputPanel` placeholder — a light-theme polish pass is a
 * future task (see the IC-LoRA/A2V redesign plan's "risks" note).
 */

/** A "play" glyph for an attached video's thumbnail slot (IC-LoRA reference
 * video, Chain's V2V source video). Moved here verbatim from
 * `SourceInputPanel.tsx`. */
export const VIDEO_PLACEHOLDER_DATA_URL =
  "data:image/svg+xml;utf8," +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">' +
      '<rect width="64" height="64" fill="#2a2a33"/>' +
      '<circle cx="32" cy="32" r="22" fill="#3d3d47"/>' +
      '<polygon points="26,20 26,44 46,32" fill="#c7cad1"/>' +
      "</svg>",
  );

/** A "♫" (beamed eighth notes) glyph for an attached audio file's thumbnail
 * slot (Create's A2V source card). Same 64×64 SVG-data-URI recipe and palette
 * as {@link VIDEO_PLACEHOLDER_DATA_URL}. */
export const AUDIO_PLACEHOLDER_DATA_URL =
  "data:image/svg+xml;utf8," +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">' +
      '<rect width="64" height="64" fill="#2a2a33"/>' +
      '<circle cx="32" cy="32" r="22" fill="#3d3d47"/>' +
      '<text x="32" y="42" font-size="30" text-anchor="middle" fill="#c7cad1">♫</text>' +
      "</svg>",
  );
