/**
 * Single source for all UI copy, in two dictionaries (English/Japanese, M7b:
 * "日英2言語、既定英語"). `en` is the historical M1-M6 copy verbatim; `ja` is
 * its Japanese translation, following the terminology fixed by
 * `Mock/AVIUTL2_DESIGN_BRIEF.md` §1 (プロンプト/シード/クリップ/画風LoRA/
 * 制御LoRA/予約, etc.). Both dictionaries must expose exactly the same set of
 * keys — enforced by `strings.test.ts` — so `i18n/LanguageContext.tsx`'s
 * `useStrings()` can swap between them without any call site needing to
 * change.
 */

import type { MaterialKind } from "../timeline/menuRouting";

export const en = {
  /** Header BASE MODEL dropdown, in the old static "Nz-Videomni" title's spot.
   * Really wired since the multi-engine groundwork (§3-97 P7): picking an entry
   * runs `POST /pipeline/load` for that base model right away, and on any
   * failure the selection returns to the base model still loaded
   * (`shell/useBaseModels.ts`, `Docs/MULTI_ENGINE_DESIGN.md` §6).
   *
   * The OPTION LABELS are not in here — they are the server's
   * `base_models[].display_name` ("LTX 2.3", …), so adding a base model is a
   * descriptor change on the server and nothing else. The `ltx23`/`ltx25`
   * literals this block used to carry were removed with the mock they served. */
  toolVersion: {
    ariaLabel: "Base model",
    /** Stand-in label while the list has not arrived, or on a backend too old
     * to declare any base model. */
    unknown: "Base model",
    /** Option label for a base model with NO weights on disk. */
    optionNotInstalled: (displayName: string): string => `${displayName} (not installed)`,
    /** Option label for a partial install — some categories are still missing,
     * so it is selectable but the server will refuse it with the specifics. */
    optionPartial: (displayName: string): string => `${displayName} (partly installed)`,
    switched: (displayName: string): string => `Switched to ${displayName}.`,
    /** Guard 2: no request was made — putting weights on disk is a batch
     * file's job, never the server's (§6.2). Names the batch file: it really
     * exists now (`install-<id>.bat`, 台帳 §3-111), so the vaguer
     * "follow the setup guide" wording would only make the user hunt for a
     * name this screen already knows. */
    notInstalled: (displayName: string, installer: string): string =>
      `${displayName} is not installed. Double-click ${installer} in the backend folder to download its weight files, then reopen this screen and pick it again.`,
    /** Guard 1: 409 JOB_BUSY. */
    switchFailedBusy: "Cannot switch the base model while a job is running.",
    /** Guard 3: 409 PIPELINE_LOADING. */
    switchFailedLoading: "The server is still loading a model. Wait for it to finish, then try again.",
    /** 422: `reason` is the server's own `detail`, shown verbatim. */
    switchFailedRejected: (reason: string): string => `Could not switch the base model: ${reason}`,
    /** Anything else (unknown id, transport failure, unexpected status). */
    switchFailed: (message: string): string => `Could not switch the base model: ${message}`,
  },
  connection: {
    checking: "Connecting…",
    connected: (pluginVersion: string): string => `Connected (${pluginVersion})`,
    disconnected: "Native bridge not available",
  },
  editInfo: {
    button: "Get edit info",
    buttonBusy: "Fetching…",
    heading: "Edit info",
    empty: "No edit info fetched yet.",
    fields: {
      width: "Width",
      height: "Height",
      rate: "Frame rate",
      scale: "Scale",
      sampleRate: "Sample rate",
      frame: "Frame",
    },
  },
  error: {
    heading: "Error",
  },
  serverStatus: {
    checking: "Connecting…",
    bridgeUnavailable: "Native bridge not available",
    offline: "Server offline",
    online: "Server online",
    busy: "Busy",
    /** `GET /status`'s `state === "loading"` — the server is rebuilding its
     * worker, which takes minutes and must not read as "ready" (§6.5). */
    loadingModels: "Loading models…",
    error: "Status error",
    retry: "Retry",
  },
  modes: {
    toolbox: "Toolbox",
    single: "Single",
    chained: "Chained",
    edit: "Edit",
    inventory: "Inventory",
    /** §3-98 P5: tooltip on a tab greyed out because the LOADED base model's
     * engine cannot run it. Names no specific feature on purpose — the tab is
     * the unit the user sees, and the server's own 422 spells out the detail
     * for anything that still gets submitted. */
    unsupportedByBaseModel: "Not available on the selected base model. Switch the base model in the header to use this tab.",
  },
  /** Edit mode (2026-08-09): the tab was promoted from a disabled mock to a
   * real one, and now hosts its own sub-tab row (Retake / Outpainting /
   * Inpainting). Outpainting is a real panel (`Docs/PENDING_TASKS_CLOSED.md`
   * §3-70, filed as §1-13 at the time);
   * Retake is still a placeholder and Inpainting is a disabled mock sub-tab.
   * The Outpainting panel has NO prompt field of its own — it reads the shared
   * `promptBar` above the tabs, exactly as Create/Chain do. The sub-tab LABELS are
   * deliberately identical in en/ja, matching the `modes.*` main-tab labels
   * (owner decision, 2026-08 tab rename); only the prose below is translated.
   * Groundwork for `Docs/PENDING_TASKS_CLOSED.md` §3-70 (Outpainting, filed as
   * §1-13 at the time). */
  edit: {
    subTabsAriaLabel: "Edit tool",
    subTabs: {
      retake: "Retake",
      outpainting: "Outpainting",
      inpainting: "Inpainting",
    },
    /** The tooltip on a sub-tab the LOADED base model's engine cannot run, one
     * line per sub-tab — the same shape `chained.unavailableOnBaseModel` uses
     * for its material panels, and for the same reason: the Edit TAB is live
     * here (the engine can run the OTHER sub-tab), so a generic "unsupported"
     * would leave the user guessing which half is out of scope.
     *
     * The Inpainting mock deliberately gets NO tooltip — it is "not built yet"
     * on every base model, which its own absence of a panel already says
     * (`shell/ModeTabs.tsx` treats the Toolbox mock exactly this way). */
    unavailableOnBaseModel: {
      retake:
        "Retake is not available on the selected base model. Switch the base model in the header to use it.",
      outpainting:
        "Outpainting is not available on the selected base model. Switch the base model in the header to use it.",
    },
    retake: {
      heading: "Retake",
      /** Rewritten 2026-08-10 (owner feedback ⑥): now names the 73-frame floor,
       * because a window at (or near) that floor barely differs from the
       * original — the single most common "why did nothing change?" report. */
      summary:
        "Regenerates the selected section of a video. Anything outside the selection is left untouched. (73 frames minimum. Too short a section barely differs from the original.)",
      /** §1-17 F2: the range band (`modes/edit/RangeBand.tsx`). Four layers —
       * the whole material, the retake window, the two glue strips inside the
       * window, and the two drag handles — plus the readout line, the
       * at-the-limit note and the one-line glue explanation. */
      rangeBand: {
        label: "Section to retake",
        startHandle: "Start of the retake",
        endHandle: "End of the retake",
        startValueText: (frame: number, seconds: string): string => `Start: frame ${frame} (${seconds}s)`,
        endValueText: (frame: number, seconds: string): string => `End: frame ${frame} (${seconds}s)`,
        /** The readout's MAIN line (owner feedback ③, 2026-08-10): where on the
         * TIMELINE the retake lands, which is what the user actually asked for.
         * Frame numbers are 1-origin, matching AviUtl2's own UI
         * (`Docs/RIGHTCLICK_REDESIGN_SPEC.md` §6-3) — the caller adds the +1.
         * Built by `RetakePanel` (it needs the material-seconds ↔ project-frames
         * mapping that only `useRetakeForm` has) and handed to `RangeBand` as a
         * function, so it keeps updating mid-drag. */
        /** The trailing note spells out the 8n+1 rule (owner feedback,
         * 2026-08-10): the window snaps to those lengths, so the numbers here
         * rarely match the range that was selected frame for frame. */
        placementReadout: (startFrame: number, endFrame: number): string =>
          `Redoing frames ${startFrame} to ${endFrame} of the timeline. (Window lengths are always a multiple of 8, plus 1.)`,
        /** The readout's SUB line, in brackets under the main one. */
        readout: (frames: number, seconds: string): string => `(a ${frames}-frame window, ${seconds}s)`,
        maxNote: (maxFrames: number): string =>
          `The window can be at most ${maxFrames} frames long, so it cannot be stretched any further.`,
        glueNote:
          "The striped strips at both ends of the window are the join: the original footage stays there so the retaken part blends into what comes before and after. Only the part between them is regenerated, and the strips cannot be moved on their own.",
      },
      /** F5: the panel proper. `idle` is the manual-tab-switch state — Retake
       * has no material picker of its own, so without a right-click there is
       * nothing to show. */
      idle: "Right-click the object on the timeline, select the range you want to redo, and choose “Redo the selected range”.",
      /** 🔁 (owner feedback batch 2, 2026-08-10): the material and the range
       * were dropped, but every setting below is still on screen and waiting
       * for the next right-click. Width/height are called out because they are
       * the ONE pair that does NOT carry over — they follow the new material's
       * own size (owner decision ①). */
      awaitingSource:
        "The material was cleared. Right-click the object on the timeline, select the range you want to redo, and choose “Redo the selected range”. The settings below are kept as they are (width and height will follow the new material).",
      /** ❌ / 🔁, the two icon buttons on the material card. ❌ throws the whole
       * retake away (settings included); 🔁 keeps the settings and waits for the
       * next right-click. */
      clearButton: "Clear this retake and reset the settings",
      resetSourceButton: "Redo a different section (keeps these settings)",
      sourceHeading: "Material",
      sourceThumbAlt: "Video attached",
      uploading: "Loading the material…",
      /** Material card (owner feedback ②, 2026-08-10). `seconds` is the length
       * of the part of the file this ribbon actually PLAYS (not the file's whole
       * duration — those differ on a head-trimmed object, and the band above
       * draws the played part). `frames` is that length at the panel's own frame
       * rate, which is why the "at Nfps" is spelled out: the material's true
       * frame rate is not obtainable from AviUtl2 (known limitation), so this is
       * a conversion, not a reading. `width`/`height` are `0` when unknown
       * (`bridge/types.ts`) — the resolution is then left off entirely. */
      sourceReadout: (
        fileName: string,
        seconds: string,
        frames: number,
        fps: number,
        width: number,
        height: number,
      ): string => {
        const size = width > 0 && height > 0 ? `, ${width} x ${height}` : "";
        return `${fileName} — ${seconds}s, ${frames} frames (at ${fps}fps)${size}`;
      },
      /** The material is fixed at right-click time — there is deliberately no
       * way to swap it here, because the window was measured against THAT
       * object's place on the timeline. */
      sourceFixedNote: "The material comes from the object you right-clicked and cannot be swapped here.",
      loadFailed: "The material could not be loaded. Please close this tab and try the right-click again.",
      /** The upload came back without the trim actually applied — the window
       * would be measured against the wrong part of the file, so Generate is
       * blocked rather than silently redoing the wrong section. */
      trimFailed:
        "Only part of this object is used on the timeline, but the material could not be cut to that part. Redoing it would work on the wrong section, so this cannot continue.",
      audioHeading: "What to redo",
      audioBoth: "Picture and sound",
      audioVideoOnly: "Picture only (keep the original sound)",
      /** のりしろ hint (1 line, plan §2). The "these are calibrated values and
       * cannot be changed" half was dropped 2026-08-10 (owner feedback ⑥) — the
       * absence of a knob already says it, and `rangeBand.glueNote` covers the
       * "you cannot move them yourself" part where it is actually relevant. */
      glueHint: (headFrames: number, tailFrames: number): string =>
        `The join keeps ${headFrames} frames at the front and ${tailFrames} at the back.`,
      noticesHeading: "Before you press Generate",
      noticeDifferent:
        "The redone section becomes a different take. It will not match the original frame for frame, even with the same prompt.",
      noticeGlueQuality:
        "The join keeps the original picture, but it is decoded and re-encoded once, so its quality changes very slightly.",
      noticeSilentAudio:
        "If the material has no sound in this section, choosing “Picture and sound” will invent sound for it. Choose “Picture only” to keep it silent.",
      /** checkStale: the timeline moved between the right-click and Generate.
       * A warning only — it never blocks. */
      staleWarning:
        "The timeline selection has changed since this panel opened. The range shown here is the one that will be redone.",
      generateButton: "Redo the selected range",
      generatingButton: "Submitting…",
      /** Gate messages, rendered by the shared `GenerateReasonsNote`. */
      generateReasons: {
        sourceMissing: "The material has not loaded yet.",
        sourceUploading: "Waiting for the material to finish loading.",
        sourceUploadFailed: "The material could not be loaded.",
        sourceTrimFailed: "The material could not be cut to the part used on the timeline.",
        rangeUnusable:
          "The selected range cannot be turned into a window on this object. Please select a range inside the part of the object that actually plays.",
        outOfMaterial: "The material is too short for the shortest redo window.",
      },
    },
    outpainting: {
      heading: "Outpainting",
      sourceHeading: "Material",
      sourceHint: "Drag and drop a video file here.",
      /** Alt text for the play-glyph placeholder shown once a video is attached
       * (`shell/thumbnailPlaceholders.VIDEO_PLACEHOLDER_DATA_URL`). */
      sourceThumbAlt: "Video attached",
      chooseButton: "Choose source video",
      changeButton: "Change source video",
      uploadingButton: "Uploading…",
      clearButton: "Clear source video",
      none: "No source video selected.",
      sourceReadout: (width: number, height: number, seconds: string): string =>
        `${width} x ${height} px, ${seconds}s`,
      padsHeading: "How much to add",
      padLeft: "Left",
      padRight: "Right",
      padTop: "Top",
      padBottom: "Bottom",
      /** 設計方針書 §5-B: the note that goes directly under the four sliders. */
      gridNote: "The width and height must both be multiples of 128 pixels.",
      /** 設計方針書 §5-D (2026-08-12): the opt-in that mirrors each pad onto its
       * opposite side, so the original video stays centred. Label only — no hint
       * line, keeping §5-C-6's「画面の文字を増やさない」intact. */
      centeringLabel: "Keep centered",
      /** マスクブラー (2026-08-09). The slider is in dilation STEPS internally
       * (0-15, the server's own range) but every number the user sees is in
       * full-resolution pixels — `steps x canvas long side / 64`. */
      blurLabel: "Mask blur",
      blurZero: "0 px (no dilation)",
      /** Mandatory copy (blend band), now quoting the REAL width for this
       * canvas rather than the old fixed "150 px" (which was only ever right at
       * 1920 px). Users must know that band is NOT untouched original footage.
       *
       * ONE sentence (owner, 2026-08-09): the "raise this when the mask's green
       * survives" advice used to be a second line of its own (`blurGreenNote`)
       * directly below, which read as two separate paragraphs about the same
       * slider. It is now a parenthetical here, and that key is gone. */
      blendNote: (px: number): string =>
        `About ${px} px inward from the edge of the extension, the mask blur blends the original video and the added area together so that they merge (if the green of the mask is left showing in the added area, raise the mask blur).`,
      /** Non-blocking warning: at this width the two opposing bands meet, so
       * essentially the whole original video is inside the blend. */
      blurWarning: "At this value almost the whole of the original video becomes part of the blend.",
      canvasReadout: (width: number, height: number): string => `Generating at ${width} x ${height} px`,
      /** Frame count, shown next to the frame rate under the duration slider. */
      numFramesLabel: "Frames",
      previewAlt: (
        canvasWidth: number,
        canvasHeight: number,
        sourceWidth: number,
        sourceHeight: number,
      ): string =>
        `Preview: an extended canvas of ${canvasWidth} by ${canvasHeight} pixels, holding the original ${sourceWidth} by ${sourceHeight} pixel video.`,
      previewLegendSource: "Blue rectangle: the original video's size",
      previewLegendCanvas: "Blue line: the size of the video after extension",
      previewLegendBlend: (px: number): string =>
        `Dashed line: the area where the pixels are kept (about ${px} px inside the edge)`,
      /** 設計方針書 §4-5: a warning, never a block. */
      comfortWarning: (tokens: number): string =>
        `At this size and length the generation comes to roughly ${tokens} units of work, which is past the point where this machine usually starts running short of video memory. It will still run, but it may be slow. Reduce the amount added, or shorten the length.`,
      /** Edit専用のspill警告文言（修正1, 2026-08-09）。Single側の
       * `single.spillWarning` とは独立のキー — Single の文言は変更しない方針
       * のため分離した。中身は「2〜4倍」の事実自体は共通だが、断定を避けた
       * 言い回しにしている（オーナー指定）。 */
      spillWarning: "Generation may slow down 2-4x.",
      generateButton: "Generate",
      generatingButton: "Generating…",
      generateReasons: {
        sourceMissing: "Attach the video you want to extend.",
        sourceUploading: "Wait for the source video upload to finish.",
        sourceUploadFailed: "The source video could not be uploaded. Choose the file again.",
        sourceTrimFailed:
          "The source video's range could not be cut out. The plugin may be out of date — update it, or pick the file again to run with the whole video.",
        mediaInfoUnknown:
          "The source video's size and length could not be read. Choose the file again, or use a video in a different format.",
        padsZero: "Add at least some amount on one of the four sides.",
        /** 2026-08-11: the pad sliders moved to 1px steps, so the canvas can now
         * land off the 128 grid. Width and height are deliberately SEPARATE
         * lines — `GenerateReasonsNote` de-dupes on the resolved sentence, so a
         * single shared line would stay put after only one of the two was
         * fixed (owner decision; see `outpaintGeometry.outpaintReasons`). */
        canvasWidthOffGrid: "The width must be a multiple of 128.",
        canvasHeightOffGrid: "The height must be a multiple of 128.",
        /** 2026-08-11 (第2弾): the 4096 ceiling is a message too, now that the
         * pad inputs no longer cap themselves against the opposite side. Split
         * into width and height for the same reason as the 128 lines above. */
        canvasWidthTooLarge: "The width can be at most 4096 pixels.",
        canvasHeightTooLarge: "The height can be at most 4096 pixels.",
        innerTooSmall: (minSide: number): string =>
          `The original video must be at least ${minSide} pixels on both sides. Use a larger video.`,
        loraMissing: (loraName: string): string =>
          `The "${loraName}" control adapter is not installed on the server. Install it, then reload the list on the Inventory screen.`,
      },
    },
    /** Shared by every placeholder panel under Edit. */
    comingSoon: "Under construction. This panel is only the frame of the screen — there is nothing to operate here yet.",
  },
  promptBar: {
    label: "Prompt",
    placeholder: "Describe the video you want to generate…",
  },
  /** NAG (Normalized Attention Guidance, 2026-07-28): the shared negative-
   * prompt accordion rendered once below the prompt bar, outside any tab —
   * Create/Chain/Batch all read the same `NagSettings` (`shell/NagAccordion.tsx`).
   * `onBadge` is intentionally identical in both languages (owner decision). */
  nag: {
    heading: "Negative Prompt",
    onBadge: "【🔴ON】",
    textLabel: "Negative prompt",
    enableLabel: "non-CFG Negative",
    methodNag: "NAG",
    methodVsf: "VSF",
    methodVsfHint: "Practical range is 1.5-5. Note: 0 does not disable VSF - uncheck the non-CFG Negative box to turn it off.",
    scaleLabel: "NAG scale",
    vsfScaleLabel: "VSF scale",
    advancedHeading: "Advanced",
    tauLabel: "NAG tau",
    alphaLabel: "NAG alpha",
    resetTooltip: "Reset the sliders to their defaults.",
  },
  chained: {
    heading: "Chained",
    /** Auto-detected mode badge shown at the top of the Chain tab — reflects
     * whether a V2V source video is attached vs. a from-scratch chain (no
     * source video, `chain.sourceInput.none`). Read-only status, not a
     * clickable tab: the mode follows source-video attachment. */
    modeBadge: {
      v2v: "Continue video (V2V)",
      scratch: "New chain (no source)",
    },
    configFallbackWarning:
      "Could not reach the server to load presets/limits — using built-in defaults. Values may not match the running server.",
    /** §3-102 (LTX 2.5 Chained, first stage): one line per material panel the
     * LOADED base model's engine cannot use, shown next to the greyed panel.
     * The Chained tab itself is live here — only these attachments are out of
     * scope — so every line names the ONE thing that is unavailable and the one
     * way out (switch the base model in the header), in the same shape
     * `batch.unavailableOnBaseModel` already uses for its whole panel. */
    unavailableOnBaseModel: {
      sourceVideo:
        "A source VIDEO cannot be used on the selected base model (continuing an existing video, V2V). Everything else about this card still works — clip 1's opening IMAGE is unaffected. To attach a video, switch the base model in the header.",
      sourceAudio:
        "Generating from an audio track (A2V) is not available on the selected base model. Chained generation itself still works — switch the base model in the header to attach audio.",
      endSource:
        "End source is not available on the selected base model. Chained generation itself still works — switch the base model in the header to attach an end material.",
      referenceVideo:
        "Reference video (control IC-LoRA) is not available on the selected base model. Chained generation itself still works — switch the base model in the header to attach a reference.",
    },
    /** Sprint 2 item 9: chain preset dropdown (`config.generation_presets`),
     * mirroring Create's `strings.single.presets` naming. */
    presets: {
      label: "Presets",
      placeholder: "Choose a preset…",
    },
    clipsHeading: "Clips",
    addClipButton: "Add clip",
    removeClipButton: "Remove clip",
    clipLabel: (index: number): string => `Clip ${index + 1}`,
    clipPromptLabel: "Prompt override",
    clipPromptPlaceholder: "Leave empty to use the shared prompt above",
    clipNumFramesLabel: "Duration",
    /** §1-16 長尺A2V: the "🎵 which part of the track does this clip cover?"
     * badge in a clip card's header, shown only while an audio file is attached
     * (`useChainForm.audioSegmentWindowsForClips` supplies one window per clip).
     * Seconds arrive already rounded to two decimals by `ClipCard`. */
    clipAudioWindow: {
      label: (startSec: string, endSec: string): string => `🎵 Audio ${startSec}s – ${endSec}s`,
      tooltip:
        "The part of the audio this clip covers. It overlaps slightly with the neighbouring clip (the crossfade at the seam).",
    },
    minClipsError: (min: number): string => `At least ${min} clips are required (no source video attached).`,
    maxClipsReached: (max: number): string => `Maximum ${max} clips reached.`,
    totalFramesLabel: (total: number, max: number): string => `Total input: ${total} / ${max} frames`,
    totalFramesExceeded: (max: number): string => `Total frames across all clips must not exceed ${max}.`,
    outputFramesLabel: (frames: number, seconds: number): string =>
      `Predicted output: ≈${seconds.toFixed(1)}s (${frames} frames)`,
    /** 素材（末尾）v2 (2026-08-15): the BREAKDOWN form of `outputFramesLabel`,
     * used instead of it while an end source contributes a band. v2 appends the
     * material after the clips, so the predicted output is longer than the clip
     * settings alone suggest — saying only the total would look like a bug. The
     * screen falls back to the plain label when `tail` is 0, so a chain without
     * an end source reads exactly as it always has. */
    outputFramesWithTailLabel: (frames: number, seconds: number, tailFrames: number, tailSeconds: number): string =>
      `Predicted output: ≈${seconds.toFixed(1)}s (${frames} frames) — the last ${tailSeconds.toFixed(1)}s (${tailFrames} frames) are the material`,
    overlapHeading: "Seam blending",
    overlapFramesLabel: "Seam blend width",
    overlapStrengthLabel: "Seam blend strength",
    chunkedUpsampleLabel: "Chunked upsample",
    chunkedUpsampleHint:
      "Upsamples in memory-saving chunks so long chains don't run out of GPU memory. Recommended to leave on.",
    /** §1-14/§3-57 stage-2 window (`GenerateChainRequest.stage2_window`).
     * Copy rework (owner decision, 2026-08-09): earlier this was DELIBERATELY
     * free of the words "window"/"tile"/"latent", presenting the choice as a
     * duration the finishing pass steps by (seconds derived live from the
     * form's frame rate). The owner has since decided the opposite: the user
     * should see this as the LENGTH of the clip stage-2 cuts the draft into
     * (`vTile`, in latent frames — not the `vAdv` advance the old copy used),
     * spelled out plainly as "latent frames", with fixed approximate seconds
     * (24fps assumption, `Math.round`-free — see the literals below) rather
     * than a live frameRate-derived readout. `shell/tokenBudget.ts` still owns
     * the underlying geometry (`STAGE2_WINDOW_PRESETS`); only the display
     * copy stopped reading it dynamically. All four leaves below are now
     * plain strings, not template functions — `ChainedScreen.tsx` no longer
     * imports `stage2AdvanceSeconds` at all as a result. */
    stage2Window: {
      label: "Stage-2 (upscale pass) clip length",
      /** vTile=22 -> pxFromVLatent(22) = 169px / 24fps ≈ 7.0s. */
      standardOption: "22 latent frames (≈7.0 s)",
      /** vTile=19 -> pxFromVLatent(19) = 145px / 24fps ≈ 6.0s. */
      highResolutionOption: "19 latent frames (≈6.0 s, reduces VRAM overflow)",
      /** Two paragraphs, separated by a blank line (`\n\n`) — rendered as two
       * stacked `.field-hint` <p>s by `ChainedScreen.tsx` so the break survives
       * (that element doesn't set `white-space`, so a literal `\n` alone would
       * collapse). */
      hint:
        "22 latent frames is recommended. 19 latent frames makes VRAM overflow less likely even on high-resolution video, but increases drift and artifacts in the result.\n\n[How it works] LTX 2.3 generation goes through three stages: Stage-1 (generate a low-resolution draft), Stage-2 (upscale the draft), then VAE decode. In chained-clip generation, Stage-1 joins your specified clip lengths into one long draft video. That draft is too long to upscale all at once in Stage-2, so it is split into fixed-length segments from the start, each upscaled separately, then rejoined. This dropdown is where you set the length used for that split — measured in latent-representation frames, not real time.",
      /** Shown when the current resolution exceeds the comfortable budget on
       * the currently-selected step. Advisory — Generate stays enabled.
       * Rewritten (修正3, 2026-08-09) to drop the seconds-based phrasing (the
       * option labels above no longer read as durations either), then split in
       * two (owner decision, 2026-08-12): this sentence now states ONLY the
       * consequence and is shown on BOTH steps, because 19 latent frames can go
       * over budget too. The "switch to 19" nudge moved to
       * {@link overBudgetShorterWindowHint} below. */
      overBudgetWarning: "Generation may slow down because of VRAM overflow.",
      /** Appended to {@link overBudgetWarning} with a single space, and only on
       * the `standard` step — on the shorter one there is nothing shorter left
       * to suggest. */
      overBudgetShorterWindowHint: "Choosing 19 latent frames may reduce it.",
    },
    /** Chain's unified source-input slot (task brief "Chainのソース入力欄一本
     * 化"): one picker button that accepts either an image (start frame for a
     * from-scratch chain) or a video (V2V continuation), auto-routed by
     * extension (`modes/chained/sourceRouting.ts`) — replaces the old separate
     * `startFrame`/`sourceVideo` choose buttons. Rendered by
     * `modes/chained/SourceInputPanel.tsx`. */
    sourceInput: {
      /** 素材（末尾）追加に伴う改名: this slot is the chain's START material, so
       * it is named for that now that a matching END slot
       * ({@link chained.endSource}) sits below the clip list. Only consumer:
       * `modes/chained/SourceInputPanel.tsx`. */
      heading: "Start source",
      note: "Generates a video that continues from the attached material (video or image).",
      chooseButton: "Choose image or video…",
      /** §3-102: the same button while the loaded base model cannot take a
       * source video — the dialog it opens is filtered to images, so the label
       * must not keep offering both. */
      chooseImageOnlyButton: "Choose image…",
      changeButton: "Change…",
      uploadingButton: "Uploading…",
      /** Detaches whichever is currently attached, returning the screen to
       * from-scratch mode if it was a video. The only way back once a source
       * video is attached. */
      clearButton: "Clear",
      none: "No source selected.",
      videoPlaceholderAlt: "Source video",
      strengthLabel: "Strength",
      /** The attached file's name line, with the material's pixel size appended
       * (owner request 2026-08-10: it is what the width/height sliders below
       * are being matched against). Same shape as Retake's `sourceReadout`: the
       * size is dropped entirely — name only — when it isn't known, which is
       * `null`/`0` from `useChainForm.sourceMediaSize` (probe failed, or the
       * file reports no dimensions). */
      sourceReadout: (fileName: string, width: number, height: number): string =>
        width > 0 && height > 0 ? `${fileName} — ${width} x ${height}` : fileName,
      unsupportedType:
        "Unsupported file type. Choose an image (.png, .jpg, .jpeg, .webp) or a video (.mp4, .mov, .webm, .mkv).",
    },
    sourceVideo: {
      contextFramesLabel: "Context frames",
      contextFramesHint: (frames: number): string => `${frames} frames carried over from the source video.`,
      contextFramesError: "Context frames must be less than the first clip's duration.",
    },
    /** 素材（末尾）: the mirror image of `sourceInput` above — ONE image or video
     * the generated chain must END with (`GenerateChainRequest.end_source`),
     * rendered by `modes/chained/ChainEndSourcePanel.tsx` inside its own
     * accordion directly below the clip list.
     *
     * 窓内モード (2026-08-17): the card came back after the 2026-08-16
     * 非露出化. What changed underneath is that the material is frozen onto the
     * LAST 8 FRAMES OF THE CLIP rather than appended after it, so the output no
     * longer gets longer — which is what every readout below now says.
     *
     * The card shows AT MOST ONE note at a time (a single always-mounted line,
     * so it never reflows) — the keys below are listed in that priority order:
     * the two conflicts, the two length failures, the still-image warning, the
     * attached video's anchor readout, then the unattached hint. The quality
     * warning is NOT part of that rotation: it is its own banner below the note,
     * because it is about the CLIP rather than about the material. */
    endSource: {
      /** Also the `<summary>` text of the accordion `ChainedScreen` wraps the
       * panel in (`showHeading={false}` on the panel itself then). */
      heading: "End source",
      /** 2026-08-16 (owner wording): one sentence for what the card does, the
       * exact mirror of `sourceInput.note`'s own new sentence. */
      note: "Generates a video that ends with the attached material (video or image).",
      chooseButton: "Choose image or video…",
      changeButton: "Change…",
      uploadingButton: "Uploading…",
      /** Deliberately NOT "Clear" — `chained.sourceInput.clearButton` uses that
       * exact text and both cards are mounted on the same screen at once, so a
       * screen reader (and a `getByRole` query) would see two identically-named
       * buttons. Same reasoning as `referenceVideo`'s own "Remove reference
       * video". */
      clearButton: "Remove end source",
      none: "No end source attached.",
      videoPlaceholderAlt: "End source video",
      /** A READOUT of the fixed anchor, not a control's label — the v1
       * `contextFramesLabel` slider caption is gone with the slider. 窓内モード:
       * the parenthesis is the headline, since the whole point of the change is
       * that attaching material no longer lengthens the result. */
      contextFramesHint: (frames: number): string =>
        `The last ${frames} frames of the generated video are frozen onto the start of the material (the video does not get longer).`,
      /** Image-only. The exact sentence agreed with the owner — the still image
       * is frozen for the API minimum of 8 frames. */
      stillImageNote: "The last 8 frames will be a still image.",
      /** Priority 3: the attached VIDEO is below the 9-frame floor (the 8-frame
       * anchor plus the causal VAE's keyframe primer), so it cannot supply the
       * anchor at all. */
      tooShortNote: "The end source needs to be 9 frames or longer.",
      /** 窓内モード品質警告: shown BELOW the priority note, not inside it — it is
       * about the CLIP's length rather than the material's, so it can be true at
       * the same time as any of the readouts above. `maxFrames` is one stage-2
       * window in pixel frames (169 standard / 145 high-resolution), and the
       * sentence names it twice on purpose: once as the boundary that was
       * crossed, once as the value to type in.
       *
       * 逆順Chained (2026-08-18, second stage): SINGLE-CLIP ONLY
       * (`useChainForm`'s `endSourceQualityLimitFrames` now gates on
       * `clips.length === 1`) — the ceiling is based on クリップ1本のときの
       * 実験結果 (§62's reconnaissance experiment ran the window-internal case
       * exclusively), so it says nothing about a multi-clip reverse chain and
       * is silent there rather than mis-warning. */
      qualityWarning: (maxFrames: number): string =>
        `Past ${maxFrames} frames, the generated video loses quality.`,
      /** §1-22 (owner wording, 2026-08-18): the multi-clip counterpart of
       * `qualityWarning` above — shown while material is attached to a 2+ clip
       * chain (逆順Chained), regardless of any single clip's length. Mutually
       * exclusive with `qualityWarning`'s own trigger (`endSourceQualityLimitFrames`
       * only ever fires at `clips.length === 1`) — see
       * `useChainForm.endSourceMultiClipQualityWarning`'s doc comment. */
      multiClipQualityWarning:
        "Chaining multiple clips together with an end source attached lowers the quality of the generated result.",
      /** のりしろ既定値1 (2.6(e), 逆順Chained second stage): shown while 素材
       * （末尾）meets a 2nd (or later) clip — the moment `overlapFrames` gets
       * nudged to 1 on the user's behalf. States that the value is still a
       * plain slider, not a locked-in setting. */
      reverseOverlapHint: "The seam blend width was set to 1 for this multi-clip end source. You can still change it.",
      /** Priority 4: the material's length could not be established at all
       * (the server reported none and the attach carried no measurement). Says
       * what to DO, since there is nothing to fix about the file itself. */
      lengthUnknownNote: "The length of the end source could not be determined. Attach it again, or use a different file.",
      /** If the material has an audio track, it is now always frozen into the
       * tail along with the video (バッチ3・2026-08-18: the anchor's audio is
       * no longer generated independently — it's carried in whenever present).
       * Materials without audio, and still images, fall back to independent
       * generation for the tail, same as before. Unconditionally rendered, so
       * the wording must stay correct for every material type. */
      audioModeNote:
        "If the material has audio, it is carried into the tail as well (materials without audio, and still images, generate audio independently).",
      /** バッチ1 (2026-08-18): the anchor's fixed-strength slider
       * (`end_source.strength`). `overlapStrength`'s exact label pattern. */
      strengthLabel: "End source strength",
      /** Stage-2 always re-hard-freezes regardless of `strength`, so the
       * sentence can flatly promise the final frame — only the APPROACH to it
       * (Stage-1's blend) is what softens. Audio (バッチ3・2026-08-18) is
       * hard-frozen unconditionally and never follows `strength` at all. */
      strengthHint:
        "At 1.00 the ending matches the material exactly. Lower values blend into it more loosely (the final frame is still exactly the material — and audio always stays exactly the material, regardless of strength).",
      /** Priority 1/2 (errors). Deliberately the SHORT form, matching
       * `sourceAudio`/`referenceVideo`'s own conflict lines; the
       * Generate-reasons note below the button carries the remedy. */
      conflictsWithAudio: "An end source and a2v can't be combined.",
      conflictsWithReference: "An end source and IC-LoRA can't be combined.",
    },
    /** §1-16 長尺A2V: Chain's own audio slot (`modes/chained/ChainAudioPanel.tsx`),
     * the multi-clip counterpart of Create's `create.sourceAudio`. One track is
     * attached to the WHOLE chain and the server assigns each clip its own slice
     * of it; the ↔️ button re-lays the clip list over the measured length.
     *
     * The card shows AT MOST ONE note at a time (a single always-mounted line,
     * so the panel never reflows) — the keys below are listed in that priority
     * order: hard attach error, V2V conflict, unmeasurable length, the 24-clip
     * ceiling, then a plain leftover-tail note. */
    sourceAudio: {
      /** W3 レイアウト再構成 (2026-08-11): this is now also the `<summary>`
       * text of the accordion `ChainedScreen` wraps `ChainAudioPanel` in
       * (`showHeading={false}` on the panel itself then, so the text isn't
       * doubled) — matches `single.sourceAudio.heading`'s own wording
       * verbatim rather than the previous chain-specific "(long A2V)" gloss. */
      heading: "Audio to video (A2V)",
      note: "Attach one audio file and each clip is automatically assigned its own part of it. Can't be combined with a source video (V2V).",
      audioPlaceholderAlt: "Audio file",
      chooseButton: "Choose audio file…",
      changeButton: "Change audio file…",
      uploadingButton: "Uploading…",
      clearButton: "Remove audio",
      none: "No audio file attached.",
      /** The ↔️ auto-fit button. Named in `chain.generateReasons.audioTooShort`/
       * `chainLayoutInvalid` and in the two failure toasts, so all four must be
       * changed together if this label ever changes. */
      adjustButton: "↔️Adjust the duration",
      /** The slider feeding the auto-fit: the length every clip it is free to
       * resize gets (8n+1, `chainUtils.ADDED_CLIP_FRAMES_DEFAULT` = 257). */
      addedClipFramesLabel: "Added clip length",
      /** Hard attach rejection (`audioAttachError === "tooLong"`): longer than the
       * biggest chain that can exist. The seconds are derived from the current
       * frame rate / seam width (`chainUtils.maxChainAudioLatents`). */
      tooLongError: (maxSeconds: number): string =>
        `The audio is too long. Chained can handle about ${maxSeconds} seconds at most (24 clips x 481 frames). Use a shorter file.`,
      /** Both slots filled. Deliberately the SHORT form — the Generate-reasons
       * note below the button carries the "remove one of them" remedy. */
      conflictsWithSourceVideo: "V2V and A2V can't be combined.",
      probeFailed:
        "Couldn't read the audio length. Auto-adjust is unavailable (the server checks the length instead).",
      /** Owner feedback 2026-08-10: both notes open with the SAME "why" clause
       * (the video is shorter than the audio) — the 24-clip-ceiling variant only
       * adds the remedy. Keeping the openings identical is deliberate: seeing
       * one note replaced by the other must never read as two different
       * diagnoses. */
      atMaxClipsNote: (unusedSeconds: string): string =>
        `The video is not long enough, so about ${unusedSeconds}s at the end of the audio won't be used. Raise the added clip length, or use a shorter audio file.`,
      surplusNote: (unusedSeconds: string): string =>
        `The video is not long enough, so about ${unusedSeconds}s at the end of the audio won't be used.`,
      /** One per `audioFit.AudioFitOutcome`. `adjusted`/`cappedAtMaxClips` share
       * the success line; the two failures each name a concrete way out —
       * `cannotFit` MUST mention lowering the slider, because the resolver's
       * smallest possible chain is the slider value (Step F1 hand-off). */
      fitToast: {
        adjusted: (durationSec: number): string =>
          `Adjusted the clips to match the ${durationSec.toFixed(2)}s audio.`,
        alreadyFits: "The clips already match the audio length.",
        cannotFit:
          "Couldn't adjust. The clips are too long for the audio, or the added clip length is too large. Remove a clip, or lower the added clip length.",
        noFlexibleClips:
          "Couldn't adjust. Every clip has been edited by hand, so there is no card left to change. Add a clip.",
      },
    },
    /** §1-15 (clip-wise IC-LoRA reference), plan F3/F7: Chain's own reference-
     * video card (`modes/chained/ChainReferencePanel.tsx`) — ONE video attached
     * to the whole chain, sliced by frame number so each clip's stage-1 pass
     * gets its own window of it. Structured like `sourceAudio` above (a
     * card-wide drop target, 🔁 clear / 📁 choose, a thumbnail, one
     * always-mounted priority note) but with no auto-fit button and no seconds
     * readout — the slicing is by FRAME NUMBER, so a duration would only invite
     * the wrong mental model (`fpsNote` says so directly).
     *
     * The always-mounted note (`chain-reference-note`) shows AT MOST ONE line,
     * in this priority order: `conflictsWithSourceVideo` (error) >
     * `probeFailed` > `shortReferenceNote` (attached and fine) > `none`
     * (unattached). Labels/hints deliberately reuse `single.referenceVideo`'s
     * wording where the two panels say the exact same thing (the strength
     * sliders, the control-LoRA picker) — duplicated here rather than shared
     * cross-namespace since the two forms hold independent material. */
    referenceVideo: {
      /** W3 レイアウト再構成 (2026-08-11): this is now also the `<summary>`
       * text of the accordion `ChainedScreen` wraps `ChainReferencePanel` in
       * (`showHeading={false}` on the panel itself then) — matches
       * `single.referenceVideo.heading`'s own wording verbatim rather than
       * the previous chain-specific "(multi-clip IC-LoRA)" gloss. */
      heading: "Reference video (IC-LoRA)",
      /** Always-mounted, regardless of attach state — the slicing is by FRAME
       * NUMBER (not seconds), which is easy to get wrong if the reference and
       * the generated video run at different frame rates. */
      fpsNote:
        "We recommend using the same frame rate for the reference video and the generated video: the reference is read frame by frame, not by seconds.",
      videoPlaceholderAlt: "Reference video",
      chooseButton: "Choose reference video…",
      changeButton: "Change reference video…",
      uploadingButton: "Uploading…",
      /** Deliberately NOT "Clear" — `chained.sourceInput.clearButton` uses
       * that exact text and both cards are mounted on the same screen at
       * once, so a screen reader (and a `getByRole` query) would see two
       * identically-named buttons. Same reasoning as `sourceAudio`'s own
       * "Remove audio". */
      clearButton: "Remove reference video",
      /** The attached file's measured pixel size, e.g. "1280×768". Only shown
       * once `referenceMediaSize` is known — no duration counterpart (see the
       * card's own doc comment for why). */
      resolutionLabel: (width: number, height: number): string => `${width}×${height}`,
      controlLoraLabel: "Control LoRA",
      controlLoraNoneOption: "Select a control LoRA…",
      controlLoraStrengthLabel: "Control LoRA strength",
      conditioningAttentionStrengthLabel: "Conditioning attention strength",
      referenceVideoStrengthLabel: "Reference video strength",
      /** Priority 1 (error): the V2V source video and the reference video are
       * mutually exclusive — deliberately the SHORT form, matching
       * `sourceAudio.conflictsWithSourceVideo`; the Generate-reasons note below
       * the button (`referenceConflictsWithSourceVideo`) carries the remedy.
       * "v2v (video extension)" / "IC-LoRA" is the shared terminology for this
       * conflict pair on this screen (2026-08-11 owner feedback) — both the
       * short form here and the remedy form below use it, so the same
       * conflict is never described in two different vocabularies. */
      conflictsWithSourceVideo: "v2v (video extension) and IC-LoRA can't be combined.",
      /** Priority 2: the ready upload's resolution could not be read
       * (`referenceMediaSize === null` while `status === "ready"`). */
      probeFailed: "Couldn't read the reference video's resolution.",
      /** Priority 3: attached, measured, no conflict — a static reminder (no
       * numbers to fill in, since the server does the actual frame-count
       * accounting; the FE only knows the clip layout, not the reference's own
       * frame count until it is probed for a duration this panel doesn't
       * probe for). */
      shortReferenceNote:
        "If the reference video is shorter than the generated duration, the remainder generates without a reference.",
      /** Priority 4: nothing attached yet. */
      none: "No reference video attached.",
      /** Plan F5: `.warning-banner-mild` (amber, non-blocking, never added to
       * `validityReasons`) shown next to the stage-2 `overBudgetWarning` above
       * when a reference is active AND the longest clip would exceed the
       * stage-1 comfort token budget (`shell/tokenBudget.ts`'s
       * `isChainStage1OverBudget`) at the current resolution. */
      stage1OverBudgetWarning:
        "A clip is too long: VRAM spill may slow down generation. We recommend shortening each clip's frame count or lowering the video resolution.",
    },
    /** W7: Chain-only Generate-disabled reasons (`GenerateReasonsNote`, below the
     * button). `promptEmpty`/`dimensionsOffGrid`/`cropInvalid` reuse Create's
     * `create.generateReasons`; the count/total-frames/source-not-ready reasons
     * reuse the existing field-adjacent banners (`minClipsError`/
     * `totalFramesExceeded`/`sourceVideo.contextFramesError`). `uploadInFlight`
     * is the single reason the three overlapping source gates normalize to
     * (R3). §1-15's reference lines that Create words identically
     * (`referenceNeedsLoras`/`attachReferenceVideo`/`referenceUploading`) are
     * likewise NOT duplicated here — only the Chain-specific ones are. */
    generateReasons: {
      clipFramesOffGrid: "Set each clip's frame count to a valid value.",
      uploadInFlight: "Wait for the upload to finish.",
      clipTooShortForOverlap: (n: number): string =>
        `With the current SEAM BLEND settings, each clip requires at least ${n} frames.`,
      /** §1-6 ゲートB: the range of the source video that will actually be used
       * is shorter than `context_frames` (the server would 422). */
      sourceVideoTooShortForContext:
        "The range used from the source video is too short for the number of carried-over frames. Use a longer range, or lower the carried-over frame count.",
      /** §1-6: the trim was requested but not applied — the stored upload is the
       * whole file, so generating would continue from the wrong footage. */
      sourceTrimFailed:
        "The source video's range could not be cut out. The plugin may be out of date — update it, or pick the file again to run with the whole video.",
      /** §1-14/§3-57: the shorter finishing-pass step cannot hold this many
       * carried-over frames (the server would 422). */
      contextFramesTooLongForWindow: (maxFrames: number): string =>
        `With the current Stage-2 clip length, at most ${maxFrames} frames can be carried over from the source video. Lower the carried-over frame count, or return the clip length to 22 latent frames (standard).`,
      /** §1-16 長尺A2V. `audioUploading` is deliberately absent — that code reuses
       * `create.generateReasons.audioUploading` verbatim, since it is the exact
       * same sentence about the exact same upload. */
      audioConflictsWithSourceVideo:
        "V2V and A2V can't be combined. Remove either the source video or the audio.",
      audioNotReady: "Attach the audio file again (the upload failed).",
      audioTooShort: "The audio is too short. Shorten the video, or press ↔️Adjust the duration.",
      audioTooShortForChain: "Use Single for short audio — Chained needs at least 2 clips.",
      audioTooLong: "The audio is too long for Chained. Use a shorter file.",
      chainLayoutInvalid:
        "This clip layout doesn't align with the audio timeline at this frame rate. Press ↔️Adjust the duration, or change a clip length.",
      /** §1-15 参照動画: the Chain-specific reference lines. The "choose a
       * control LoRA" / "attach a reference video" / "wait for the upload"
       * cases reuse `create.generateReasons` verbatim (same rule, same
       * remedy) — see `modes/chained/generateReasonMessages.ts`. */
      referenceNotReady: "Attach the reference video again (the upload failed).",
      /** §1-15 W4 (2026-08-11): the reference-slot twin of `sourceTrimFailed`
       * above — the right-click route asked for the object's ribbon range but
       * the response did not confirm the cut, so the stored upload is the whole
       * file. Worded as `referenceNotReady`'s sibling (same slot, same remedy)
       * rather than reusing Create's longer sentence. */
      referenceTrimFailed: "Attach the reference video again (its range could not be cut out).",
      referenceConflictsWithSourceVideo:
        "v2v (video extension) and IC-LoRA can't be combined. Remove either one.",
      referenceDimensionsOffGrid:
        "With a reference video attached, the width and height must both be multiples of 128. Set them to valid values.",
      /** Owner decision 2026-08-11: depth adapters are blocked on Chain
       * UNCONDITIONALLY (not only past one clip) — deliberately stricter than
       * the server, which still accepts depth on a single-clip chain. The
       * dropdown still lists depth adapters; this is a generate-time block,
       * not a removed option. */
      depthChainUnsupported: "An unimplemented IC-LoRA adapter is selected.",
      /** 素材（末尾）: one line per end-source code. Every one of them names a
       * concrete way out, like the audio/reference blocks above. */
      endSourceUploading: "Wait for the end source upload to finish.",
      endSourceNotReady: "Attach the end source again (the upload failed).",
      endSourceTrimFailed: "Attach the end source again (its range could not be cut out).",
      endSourceConflictsWithAudio: "An end source and a2v can't be combined. Remove either one.",
      endSourceConflictsWithReference: "An end source and IC-LoRA can't be combined. Remove either one.",
      /** 逆順Chained (2026-08-18, second stage): a V2V source video and an end
       * source on 2+ clips — the interpolation case (start+end on ONE clip) is
       * unaffected. Deliberately names the future so the user doesn't read it
       * as a permanent limitation. */
      endSourceWithSourceVideoMultiClip:
        "A source video and an end source together need a single clip. Remove one, or drop to one clip (support for more is planned).",
      /** The v1 geometry lines (`endContextFramesInvalid` /
       * `…TooLongForWindow` / `…TooLongForClip`) are gone with the rules they
       * explained; these replace them. The first two say the same thing as the
       * CARD's own note — deliberately, per the existing convention that the card
       * states the fact and this note below the button states the fix. */
      endSourceTooShort: "The end source needs to be 9 frames or longer. Use a longer video.",
      endSourceLengthUnknown:
        "The length of the end source could not be determined. Attach it again, or use a different file.",
      /** 窓内モード: only reachable on a SINGLE-clip chain (逆順Chained,
       * 2026-08-18, exempts 2+ clips from this floor — see
       * `endSourceAudioOverlapBudget` below for what replaces it there). */
      endSourceNeedsOverlap:
        "An end source can't be used with a seam blend width of 1. Set the seam blend width to 2 or more.",
      /** 逆順Chained (2026-08-18, second stage): the 2+-clip counterpart of
       * `endSourceNeedsOverlap` — the audio-overlap budget mirror
       * (`chainUtils.endSourceAudioOverlapOk`). Same remedy, worded generically
       * since the exact width that fixes it depends on frame rate and clip
       * length. */
      endSourceAudioOverlapBudget:
        "This end source's clip lengths leave no audio overlap budget for the joins. Raise the seam blend width.",
    },
    generateButton: "Generate",
    generatingButton: "Generating…",
    /** U-R1: Generate button label while another job is already running. */
    busyButton: "Busy…",
  },
  /** Batch A2V (webui-B): the collapsible "run A2V overnight against a whole
   * folder of audio files" panel embedded in the Create screen. Mirrors
   * `Nz-Videomni/gradio_ui/ui.py`'s batch accordion copy (`L("batch_*")`
   * keys), condensed to what `BatchSection.tsx`/`BatchTable.tsx` need. Width/
   * height/fps/seed field labels are intentionally NOT duplicated here — the
   * panel reuses `create.size`/`create.duration`/`create.seed`/
   * `chain.chunkedUpsampleLabel` verbatim. */
  batch: {
    heading: "Batch A2V",
    /** Required on-screen notice (owner decision 2026-07-19): summarizes what
     * the batch actually does and where its output lands. */
    notice:
      "Creates one video for each audio file in the selected folder, all in a single run. (output folder will be automatically created next to the audio data folder)",
    wavDir: {
      label: "Audio folder",
      button: "Choose audio folder…",
      none: "No audio folder selected.",
    },
    imgDir: {
      label: "Image folder (optional)",
      button: "Choose image folder…",
      none: "No image folder selected — a row's own image falls back to the audio folder.",
    },
    outDir: {
      label: "Output folder",
      button: "Choose output folder…",
      auto: (path: string): string => `Auto: ${path}`,
      none: "No output folder yet — choose an audio folder first.",
    },
    promptMode: {
      label: "Row prompt mode",
      add: "Add to the shared prompt",
      replace: "Replace the shared prompt",
    },
    /** N7 (PENDING §6): shared keyframe(s) backing `Shared`-image rows —
     * mirrors Create/Chain's `create.keyframes` panel, reused verbatim via
     * `KeyframesPanel`. */
    sharedKeyframes: {
      /** Batch A2V Shared spec change (2026-07-18): rows whose Image column
       * is "Shared" now always use the first image from the Create screen's
       * KEYFRAMES panel as frame 0; any further keyframes there are ignored. */
      note: "Rows whose Image column is \"Shared\" use the first image in the Create screen's KEYFRAMES panel as frame 0. Any images after the first are not used.",
      missingWarning:
        "At least one row uses the Shared image, but the Create screen's KEYFRAMES panel has no image yet. Add at least one image there to use Shared.",
    },
    scanButton: "Scan folder",
    scanningButton: "Scanning…",
    scanError: (message: string): string => `Scan failed: ${message}`,
    table: {
      queue: "#",
      wav: "Audio file",
      duration: "Duration",
      image: "Image",
      prompt: "Prompt",
      stat: "Status",
      output: "Output",
      resetButton: "Reset to Waiting",
      empty: "No rows yet — choose an audio folder and scan it.",
      /** N5 "A1: プロンプト直接編集" — the row prompt `<input>`'s aria-label. */
      promptInputLabel: "Row prompt",
      /** N5 "A2: 共通プロンプト流し込み" — button that overwrites a row's
       * prompt with the Create screen's shared prompt verbatim. */
      copyCommonPromptButton: "Use shared prompt",
      /** N5 "A3: 行別画像割当" — the row image `<select>`'s aria-label. */
      imageSelectLabel: "Row image",
    },
    stat: {
      waiting: "Waiting",
      generating: "Generating",
      done: "Done",
      failed: "Failed",
      skip: "Skip",
    },
    summary: (done: number, total: number): string => `${done} / ${total} done`,
    currentRow: (wav: string): string => `Processing: ${wav}`,
    startButton: "Start batch",
    runningButton: "Running…",
    stoppingButton: "Stopping…",
    stopButton: "Stop",
    /** U4 guard 1: the shared width/height (owned by the Create form now) must
     * be multiples of 64, or every row 422s. Blocks Start. */
    resolutionOffGrid: "Width and height must be multiples of 64. Fix the size on the Create form above before starting.",
    /** U4 guard 2: the FPS and DURATION in effect at scan time were baked into
     * each row's frame count — if either changed since, the rows are stale.
     * Blocks Start. */
    fpsMismatch: "The FPS or DURATION differs from when the folder was scanned. Frame counts will be recalculated automatically with the current values when the batch starts.",
    /** U4 guard 3: a reference video (IC-LoRA, 128 grid) is active on the
     * Create form. Warning only — does not block Start. */
    icLoraActiveWarning: "A reference video (IC-LoRA) is active on the Create form, so the batch uses its 128-grid resolution.",
    /** §1-7 相互ロック: the shared run lock (`shell/runLock.ts`) is held by
     * Batch i2v-long on the Clip Chain screen. Blocks Start. */
    lockedByOther: "Batch i2v-long is running on the Clip Chain screen. Only one batch runs at a time — wait for it to finish, or stop it there.",
    /** §1-7 相互ロック 第2段: the backend is occupied — its single job slot is
     * taken, or a model load is in flight — the same gate Batch i2v-long's
     * `blockReasons.jobActive` states. The KEY keeps its original name (it is
     * the block-reason code the form pushes); only the copy covers both. */
    jobActive: "The server is busy: another job is running, or a model is being loaded. Only one runs at a time — wait for it to finish before starting the batch.",
    /** §3-98 P5: the loaded base model's engine cannot run chained generation,
     * which is what every batch row is. Disables the whole panel. */
    unavailableOnBaseModel: "Batch A2V is not available on the selected base model (it uses chained generation). Switch the base model in the header to use it.",
    /** Skip badge tooltips (owner decision 2026-07-19), keyed by
     * `BatchRow.skipReason` (`manifestMerge.ts`). Skip rows are excluded from
     * re-judging on Start, so raising DURATION/lowering FPS alone does not
     * bring an `over-cap` row back — the row must be manually re-queued
     * with 🔁 first, which is why that step is spelled out here. */
    skipReasons: {
      "wav-only-alpha": "Not a .wav file, or its length could not be read.",
      "over-cap":
        "Audio is longer than the current DURATION limit. Raise DURATION (or lower FPS), press 🔁 to re-queue the row, then start.",
    },
  },
  /** Batch i2v-long (§1-7): the collapsible "one long video per image" panel
   * embedded in the CHAIN screen (Batch A2V, above, lives in Create — the two
   * are deliberately kept apart). Every generation parameter comes from the
   * Clip Chain form itself, so — exactly like `batch` — no size/duration/seed
   * field labels are duplicated here. */
  batchI2vLong: {
    /** 2026-07-30 owner feedback: the panel is explicitly labelled a prototype,
     * and the two long explanatory paragraphs that used to sit here (`notice` /
     * `chainSettingsNote`) were removed — the behaviour they described (do not
     * close the window, what Stop does, "settings come from the Clip Chain
     * form") is documented in `Docs/BATCH_I2V_WORKORDER.md` instead. */
    heading: "Batch i2v-long (prototype)",
    /** Per-image output length readout. NEVER use a `Clip n/m` wording here —
     * `App.chained.test.tsx` selects clip cards with `/Clip \d\/\d/`. */
    chainSummary: (clips: number, frames: number, seconds: number): string =>
      `${clips} clips · ${frames} frames · ~${Math.round(seconds)}s per image`,
    imgDir: {
      label: "Image folder",
      button: "Choose image folder…",
      none: "No image folder selected.",
    },
    outDir: {
      label: "Output folder",
      button: "Choose output folder…",
      auto: (path: string): string => `Auto: ${path}`,
      none: "No output folder yet — choose an image folder first.",
    },
    /** 2026-07-30 owner feedback: there is no batch-wide extra prompt any more
     * (writing one was no different from editing the shared prompt itself) —
     * each ROW has its own prompt column, and this one radio pair says how that
     * row prompt combines with the Clip Chain's shared prompt. Same wording
     * shape as Batch A2V's `batch.promptMode`. */
    promptMode: {
      label: "Row prompt mode",
      add: "Add to the Clip Chain prompt",
      replace: "Replace the Clip Chain prompt",
    },
    scanButton: "Scan folder",
    scanningButton: "Scanning…",
    scanError: (message: string): string => `Scan failed: ${message}`,
    table: {
      queue: "#",
      image: "Image",
      prompt: "Prompt",
      stat: "Status",
      output: "Output",
      resetButton: "Reset to Waiting",
      empty: "No rows yet — choose an image folder and scan it.",
      /** The row prompt `<input>`'s aria-label (Batch A2V's
       * `batch.table.promptInputLabel` equivalent). */
      promptInputLabel: "Row prompt",
      /** 📝: overwrites a row's prompt with the Clip Chain's shared prompt
       * verbatim — the counterpart of Batch A2V's "Use shared prompt" button. */
      copyChainPromptButton: "Use the Clip Chain prompt",
    },
    stat: {
      waiting: "Waiting",
      generating: "Generating",
      done: "Done",
      failed: "Failed",
    },
    summary: (done: number, total: number): string => `${done} / ${total} done`,
    currentRow: (image: string): string => `Processing: ${image}`,
    startButton: "Start batch",
    runningButton: "Running…",
    stoppingButton: "Stopping…",
    stopButton: "Stop",
    /** Every reason Start is disabled, listed one by one — an opaque "the
     * chain settings are invalid" would leave the user with no way to find out
     * what to fix. Chain's own `validityReasons` are expanded alongside these
     * through `chain.generateReasons`/`create.generateReasons`. */
    blockReasons: {
      imgDirMissing: "Choose the image folder to run.",
      outDirMissing: "Choose an output folder.",
      noRows: "Scan the image folder first.",
      noRunnableRows: "Every row has finished. Press 🔁 on a row to run it again.",
      sourceVideoAttached:
        "A source video is attached to the Clip Chain form. Clear it — each image has to start its own chain, so the video would replace every image.",
      /** §1-16 長尺A2V: the audio counterpart of `sourceVideoAttached`. The batch
       * strips `source_audio` from every row it sends, so an attached track would
       * silently do nothing — better to say so than to run the whole batch
       * without the audio the user attached. */
      sourceAudioAttached:
        "An audio file is attached to the Clip Chain form. Remove it — this batch generates from images only and does not send the audio.",
      /** §1-15 参照動画: the reference-video counterpart of the two lines above.
       * The batch strips `reference_video_id` (and both strengths) from every
       * row it sends, so an attached reference would silently do nothing. */
      referenceVideoAttached:
        "Remove the IC-LoRA reference video — this batch generates from images only.",
      /** 素材（末尾）v2: the end-slot counterpart of the three lines above. The
       * batch strips `end_source` from every row it sends, so an attached
       * material would silently do nothing — and every result would be shorter
       * than the Clip Chain form's predicted output claimed. */
      endSourceAttached:
        "Remove the end source — this batch generates from images only, and does not send the end material.",
      clipsTooFew: "The Clip Chain needs at least 2 clips when no source video is attached. Add a clip.",
      promptEmpty: "Write a prompt — either on the Clip Chain form or in a row's own Prompt column.",
      promptTooLong: "The combined prompt is too long. Shorten it to 2000 characters or less.",
      unknownLoraTag: "The prompt has a LoRA name that does not exist. Fix or remove the tag.",
      jobActive: "The server is busy: another job is running, or a model is being loaded. Wait for it to finish.",
      /** The shared run lock (`shell/runLock.ts`) is held by the OTHER batch
       * panel (Batch A2V, on the Create screen) — only one batch runs at a
       * time. */
      lockedByOther: "Batch A2V is running on the Create screen. Only one batch runs at a time — stop it, or wait for it to finish.",
    },
    /** Row-aware variants of the two prompt block reasons above: once the
     * folder has been scanned the panel knows exactly WHICH rows are at fault,
     * so it names them. Kept out of `blockReasons` (a `Record<string, string>`
     * consumed by `GenerateReasonsNote`) because these are functions. */
    promptRowIssues: {
      empty: (queues: string): string =>
        `These rows would be sent with an empty prompt: ${queues}. Write a prompt on the Clip Chain form, or in each row's Prompt column.`,
      tooLong: (queues: string): string =>
        `The combined prompt is over 2000 characters on these rows: ${queues}. Shorten the Clip Chain prompt, or those rows' own prompts.`,
    },
    /** Advisory notes — none of these blocks Start. */
    notes: {
      seedFixed: "The seed is fixed, so every image is generated with the same seed. Set the seed to -1 on the Clip Chain form to vary it per image.",
      clip0PromptOverride:
        "The first clip has a prompt of its own. Each image goes on that first clip, so the shared and row prompts never reach the part of the video the image controls.",
      clipPromptOverride: "Some clips have a prompt of their own. The shared and row prompts do not change those.",
      startFrameIgnored: "The start frame on the Clip Chain form is not used — each row's own image takes its place.",
      concurrency: "Only one batch runs at a time, and Generate on the Clip Chain form is unavailable while this batch is running.",
    },
  },
  inventory: {
    heading: "LoRA",
    reload: "🔁Reload",
    reloading: "Reloading…",
    reloadToast: (total: number, styles: number, controls: number): string =>
      `Reloaded LoRAs: ${total} total (${styles} style, ${controls} control).`,
    emptyLoras: "No LoRAs found. Drop files into models/LTX23/StyleLoRA and Reload.",
    loadError: "Could not load the LoRA list.",
    loadingLoras: "Loading LoRAs…",
    addedToast: (name: string): string => `Added <lora:${name}:1.0:1.0> to the prompt.`,
    historyHeading: "History",
    historyNote: "Job history is in-memory on the server — it's cleared on restart.",
    historyEmpty: "No completed jobs yet.",
    noPromptPlaceholder: "(no prompt)",
  },
  jobs: {
    heading: "Jobs",
    empty: "No jobs yet — generated clips will appear here.",
    status: {
      queued: "Queued",
      running: "Running",
      completed: "Completed",
      failed: "Failed",
      cancelled: "Cancelled",
    },
    preview: "Preview",
    closePreview: "Close preview",
    /** ⑤ seed shown in a job card header when the job's seed is unknown. */
    seedRandom: "random",
    insert: "Insert",
    downloading: "Downloading…",
    inserting: "Inserting…",
    inserted: "Inserted",
    cancel: "Cancel",
    cancelling: "Cancelling…",
    delete: "Delete",
    deleting: "Deleting…",
    toastCompleted: (jobId: string): string => `Job ${jobId.slice(0, 8)} completed.`,
    toastFailed: (jobId: string): string => `Job ${jobId.slice(0, 8)} failed.`,
    toastCancelled: (jobId: string): string => `Job ${jobId.slice(0, 8)} was cancelled.`,
    /** M6: chain progress badge, `clip`/`clip_count` from `JobResponse` (1-based). */
    clipBadge: (clip: number, count: number): string => `Clip ${clip}/${count}`,
    /** M6: V2V join flow (Docs/API_REFERENCE.md §3.17/§3.18). */
    join: "Join with source",
    /** N6/I5: crossfade-length dropdown shown next to the join button. Only
     * the *audio* is crossfaded (`JoinRequestBody.handle_crossfade_ms`,
     * Docs/API_REFERENCE.md §3.17) — the label says so honestly. */
    crossfadeLabel: "Audio crossfade",
    crossfadeOption: (ms: number): string => `${ms} ms`,
    /** I5: source tail-keep trim length, offered as a frame count (converted to
     * `source_tail_seconds` at the generated video's fps). */
    trimLengthLabel: "Source tail to keep",
    trimLengthOption: (frames: string, sec: string): string => `${frames} frames (≈${sec}s)`,
    joining: "Joining…",
    joinRetry: "Retry join",
    /** I5: revert the local joined view back to the pre-join buttons without
     * touching the server (the joined.mp4 is left in place). */
    unjoin: "Unjoin",
    /** V2V Join: the source's own fps (`sourceFps`) genuinely differed from the
     * generated video (`videoFps`) and was converted during the join. */
    joinSourceFpsNote: (sourceFps: string, videoFps: string): string =>
      `The source video's frame rate (${sourceFps}) did not match the generated video (${videoFps}), so it was converted during the join.`,
    /** V2V Join: the source was re-encoded to the generated video's format
     * (e.g. a resolution difference) even though the fps matched. */
    joinNormalizedNote:
      "The source video was re-encoded to match the generated video's format during the join.",
    /** V2V Join: advisory shown when the project fps (`projectFps`) differs from
     * the generated video (`videoFps`), recommending they be kept aligned. */
    joinProjectFpsAdvice: (projectFps: string, videoFps: string): string =>
      `Keeping the project and generated-video frame rates aligned is recommended (project ${projectFps} / video ${videoFps}).`,
    insertJoined: "Insert joined video",
    genericFailure: "Generation failed.",
  },
  /** W2/W3 (⬇ right-click quick-insert): toasts for the two right-click
   * "insert a generated result" commands. W2 (`insertProvisionalResult`) inserts
   * a reservation object's OWN finished result via the same replace-insert the
   * 🎞 button uses; W3 (`insertLatestResultHere`) inserts the most recent
   * completed result at the right-click position as a plain insert (the
   * provisional stays). Kept beside the job toasts (`jobs.toast*`) since they are
   * likewise transient job-result feedback. */
  menuInsert: {
    /** W2 (a): the right-clicked object is not a Nz-Videomni reservation object. */
    notProvisional: "Right-click a Nz-Videomni reservation object.",
    /** W2 (b): no job in the ledger matches this reservation object (e.g. a
     * never-generated reservation, or a job cleared on the server). */
    jobNotFound: "No job matching this reservation object was found.",
    /** W2 (c) success: the completed result was inserted (replace-insert). */
    inserted: "Inserted the generated result.",
    /** W2 (c) failure: the download/insert threw. */
    insertFailed: "Could not insert the generated result.",
    /** W2 (d): the job is still queued/running. */
    notReady: "This job is still generating. Please try again after it completes.",
    /** W2 (e): the job failed or was cancelled, so there is no result. */
    jobFailed: "This job did not finish successfully, so there is no result to insert.",
    /** W3 (0 completed): there is no completed job to insert. */
    noCompleted: "There is no completed generation result to insert yet.",
    /** W3 success: the latest completed result was inserted at the cursor. */
    insertedLatest: "Inserted the latest generation result.",
    /** W3 failure: the download/insert threw. */
    insertLatestFailed: "Could not insert the latest generation result.",
    /** W3: the right-click position could not be resolved, so nothing was
     * inserted (avoids dropping the clip somewhere unexpected). */
    noCursor: "Could not determine the insert position.",
  },
  single: {
    heading: "Single",
    configFallbackWarning:
      "Could not reach the server to load presets/limits — using built-in defaults. Values may not match the running server.",
    loadingConfig: "Loading configuration…",
    spillWarning: "Generation slows down 2-4x beyond this point.",
    /** Smart comfort marker (2026-08-18): shown INSTEAD of `spillWarning`
     * above whenever the marker itself is the smart per-resolution ceiling
     * (`useGenerationForm`'s `isComfortMarkerSmart`) rather than the coarse
     * `spill_free_frames` fallback — the "2-4x" figure above was measured
     * against LTX 2.3's DEFAULT configuration, and does not hold in the
     * configurations where a smart line can be drawn at all (LTX 2.5
     * unconditionally, LTX 2.3 only with every acceleration toggle on — see
     * `shell/comfortTable.ts`'s `resolveComfortRow`). Named with a `Smart`
     * suffix specifically to avoid two unrelated existing keys: `single.size.
     * comfortMarkerTitle` (the Chained screen's resolution-slider guide
     * tooltip — a different feature entirely) and `edit.comfortWarning`
     * (Outpainting's own token-budget warning, a different workload/budget —
     * see `modes/edit/outpaintGeometry.ts`'s `COMFORT_TOKEN_BUDGET`).
     * The text is DELIBERATELY identical to Chained's `overBudgetWarning`
     * below (same wording for the same kind of warning) — a text-anchored
     * search for this string will hit both keys. */
    comfortWarningSmart: "Generation may slow down because of VRAM overflow.",
    size: {
      label: "Size",
      width: "Width",
      height: "Height",
      getFromAviUtl2: "Get size from AviUtl2",
      gettingFromAviUtl2: "Fetching…",
      /** Tooltips for the two resolution guides the Chain screen draws on these
       * sliders (2026-08-12). They live here, next to `width`/`height`, because
       * `CommonGenerationFields.SizeFields` renders them — Create and Retake
       * pass no guides and never show either.
       *
       * The two must not be confused: the first is a RECOMMENDED POINT (the
       * balanced near-16:9 pair), so a dimension above it can still be
       * comfortable once the other one comes down; the second is the real
       * per-axis ceiling AT THE OTHER DIMENSION'S CURRENT VALUE and moves as
       * that value moves. The first also names the finishing pass explicitly,
       * because a reference video's own load falls on Stage-1 and has a
       * separate advisory of its own. */
      comfortMarkerTitle: (px: number): string =>
        `Recommended limit for the finishing pass (Stage-2): ${px}px — the balanced, near-16:9 point. Lowering the other dimension can keep you comfortable above this line. It does not cover the Stage-1 load of a reference video.`,
      limitMarkerTitle: (px: number): string =>
        `Comfortable maximum for this dimension at the other dimension's current value: ${px}px.`,
    },
    /** N1: opt-in output crop (`crop_output`). Shared verbatim by Create
     * (`GenerationForm.tsx`) and Chain (`ChainedScreen.tsx`) — both render
     * `CommonGenerationFields.CropOutputField`, which reads this namespace
     * directly rather than a separate `chain.crop`, mirroring how
     * `SizeFields`/`FrameRateSeedFields` already reuse `create.size`/
     * `create.duration`/`create.seed` across both screens. */
    crop: {
      label: "Crop output",
      hint: "Crops the final video to a smaller region after generation. Must be 32 or more, and no larger than the current width/height.",
      width: "Crop width",
      height: "Crop height",
    },
    duration: {
      label: "Duration",
      fps: "FPS",
    },
    seed: {
      label: "Seed",
      hint: "-1 = random",
      randomTooltip: "Randomize seed",
      reuseTooltip: "Reuse last seed",
    },
    presets: {
      label: "Presets",
      placeholder: "Select a preset…",
      /** U-R2: `<optgroup>` headings shown when the preset list spans 2+
       * orientations (see `modes/single/PresetDropdown.tsx`'s `sortPresets`). */
      orientation: {
        landscape: "Landscape",
        square: "Square",
        portrait: "Portrait",
      },
    },
    /** M7b: permanently disabled CFG-scale placeholder reserved for a future
     * non-distilled backend (Mock/AVIUTL2_DESIGN_BRIEF.md §5-2 / §11). Never
     * enabled, never submitted — see `modes/single/CommonGenerationFields.tsx`'s
     * `ReservedFields`. (`negativePrompt` lived here too until NAG unlocked a
     * real, sent negative prompt, 2026-07-28 — see `nag.*` above instead.) */
    cfgScale: {
      label: "CFG scale",
    },
    reservedNote: "CFG is not supported by the current backend (reserved for future use)",
    keyframes: {
      label: "Keyframes (I2V)",
      /** Live "N / max" counter shown next to the keyframes heading, e.g.
       * "2 / 5" — purely numeric, so the same format works for both
       * languages. */
      count: (count: number, max: number): string => `${count} / ${max}`,
      captureButton: "Capture current frame",
      capturingButton: "Capturing…",
      chooseFileButton: "Choose image file…",
      choosingButton: "Choosing…",
      maxReached: (max: number): string => `Maximum ${max} keyframes reached.`,
      frameIdxLabel: "Frame",
      willSnapTo: (frame: number): string => `will snap to ${frame}`,
      outOfRangeWarning: "Out of range for the current duration.",
      strengthLabel: "Strength",
      removeButton: "Remove",
      uploading: "Uploading…",
      modeT2V: "T2V",
      modeI2V: (count: number): string => `I2V (${count} keyframe${count === 1 ? "" : "s"})`,
      modeA2V: "A2V",
      /** U2: shown on the mode badge while a reference video (IC-LoRA) is
       * active — explains the switch to the 128-multiple resolution grid. */
      modeICLora: "IC-LoRA",
      /** Keyframe timeline rework (2026-07-18): aria-label for the whole
       * timeline bar. */
      timelineLabel: "Keyframe timeline",
      /** Keyframe timeline rework (2026-07-18): aria-label for an individual
       * pin on the timeline (1-based index). */
      pinAriaLabel: (index: number): string => `Keyframe ${index}`,
      /** Keyframe timeline rework (2026-07-18): a pin's on-screen label and
       * `aria-valuetext` — `seconds` arrives pre-formatted (one decimal) from
       * the caller. */
      pinValueText: (frame: number, seconds: string): string => `${frame}f / ${seconds}s`,
      /** Keyframe timeline rework (2026-07-18): the gap label shown between
       * two adjacent pins on the timeline. */
      intervalLabel: (frames: number, seconds: string): string => `${frames}f / ${seconds}s`,
      /** Keyframe card polish (2026-07-18): badge shown on the keyframe card
       * at position 0. LTX 2.3's engine treats position 0 differently from
       * every later keyframe — position 0 replaces the initial latent, so the
       * start frame comes out pixel-exact, while position 1+ only guides the
       * generation (no exact-match guarantee). This badge communicates that
       * asymmetry to the user. */
      startFrameFixedBadge: "Start frame (fixed)",
      /** Keyframe timeline rework (2026-07-18): the "Add keyframe" button's
       * label, with a live count/max. */
      addKeyframeButton: (count: number, max: number): string => `Add keyframe ( ${count} / ${max} )`,
      /** Keyframe timeline rework (2026-07-18): shown when Add is disabled
       * because there's no free frame left after the last keyframe. */
      addNoRoomHint: "No room left after the last keyframe. Move or remove a keyframe to add another.",
      /** Keyframe timeline rework (2026-07-18): hint on a keyframe card that
       * has no image attached yet. */
      emptyCardHint:
        "No image yet — drop an image here or use the buttons in the top-right corner. This keyframe will be skipped when generating.",
      /** Keyframe timeline rework (2026-07-18): shown while dragging a pin
       * that will be snapped to a different frame once collisions with other
       * pins are resolved. */
      willLandAt: (frame: number): string => `will land at ${frame}`,
      /** Keyframe timeline rework (2026-07-18): reason text next to a
       * disabled Generate button when one or more keyframes sit past the
       * current duration. */
      outOfRangeGenerateHint:
        "Some keyframes are beyond the current duration. Shorten, move, or remove them to generate.",
      /** Keyframe timeline rework (2026-07-18): confirmation modal shown when
       * shortening the duration would delete out-of-range keyframes. */
      shrinkModal: {
        title: "Shorten duration?",
        body: (numFrames: number, count: number): string =>
          `Shortening the duration to ${numFrames} frames will delete ${count} keyframe(s) that fall outside the new range.`,
        applyButton: "Apply",
        cancelButton: "Cancel",
      },
    },
    /** IC-LoRA (`referenceVideo` right-click) reference-clip block. N3 adds
     * the loras-required warning + the two opt-in strength sliders; N2 adds
     * the control-LoRA dropdown + its "requires a reference video" warning —
     * both mirror Chain's `chain.referenceVideo` copy of the same shape. */
    referenceVideo: {
      heading: "Reference video (IC-LoRA)",
      icLoraNote: "IC-LoRA is /generate-only. Please choose a reference video.",
      /** depth-control/deblur adapters (§4-8): static hints — no adapter-
       * conditional show/hide mechanism exists here (only the enabled state
       * of the reference-video/control-LoRA fields tracks the selection), so
       * these stay always-visible notes next to `icLoraNote` above rather
       * than adding a new UI mechanism. */
      controlAdapterAspectHint:
        "The reference video is simply resized to the output resolution, so a mismatched aspect ratio will distort it (most noticeable with Depth control).",
      controlAdapterDepthHint:
        "Depth control: the official recommendation is Conditioning attention strength = 0.6. Keep Control LoRA strength at 1.0 — lowering it can make the reference bleed through (these are two different sliders).",
      controlAdapterDeblurHint:
        'Deblur: write the prompt in two parts, e.g. "Reference shows <scene>, heavily out of focus with soft defocused blur and no fine detail. Edited shows the same scene in sharp focus with crisp detail and clean edges. DEBLUR <scene>. Subject identity, framing, and background geometry are identical to the reference; only focus and sharpness differ between reference and edited." It removes defocus blur (out-of-focus shots), not motion blur. The reference video is used for conditioning without downscaling.',
      /** IC-LoRA/A2V card redesign: alt text for the attached reference
       * video's placeholder thumbnail. */
      videoPlaceholderAlt: "Reference video",
      chooseButton: "Choose reference video…",
      changeButton: "Change reference video…",
      uploadingButton: "Uploading…",
      clearButton: "Clear",
      /** W6: the ❌ button next to 🔁 — fully disables IC-LoRA in one click
       * (clears the reference video AND the control LoRA AND both opt-in
       * strengths), so no orphan strength/selection survives a bare 🔁 clear.
       * 🔁 only clears the reference video; this clears the whole IC-LoRA setup. */
      removeIcLoraButton: "Remove IC-LoRA (clear reference & control LoRA)",
      none: "No reference video selected.",
      lorasRequiredWarning: "Choose CONTROL LORA.",
      conditioningAttentionStrengthLabel: "Conditioning attention strength",
      referenceVideoStrengthLabel: "Reference video strength",
      controlLoraLabel: "Control LoRA",
      /** IC-LoRA UI redesign (第5波): first `<option>` of the selection-
       * preserving dropdown — replaces the old one-shot `controlLoraPlaceholder`
       * (a `value=""` placeholder the `<select>` snapped back to after every
       * pick, which is exactly what made the selection look like it hadn't
       * "stuck"). */
      controlLoraNoneOption: "None",
      /** IC-LoRA UI redesign (第5波): the panel's weight slider, shown once a
       * control LoRA is selected. */
      controlLoraStrengthLabel: "Control LoRA strength",
      referenceVideoRequiredWarning: "This control LoRA requires a reference video. Please choose one above.",
      /** IC-LoRA UI redesign (第5波): `AppShell`'s auto-migration toast — a
       * single hand-typed control-LoRA tag moved into the panel above. */
      controlLoraMigratedToast: (name: string): string =>
        `Moved the <lora:${name}:...> tag to the Control LoRA panel above.`,
      /** Same migration, but more than one control-LoRA tag was found in the
       * prompt at once — only the LAST one is kept (mirrors correcting a
       * choice by retyping it); the rest are discarded along with it. */
      controlLoraMigratedMultiple: (name: string): string =>
        `Found multiple control LoRA tags — kept ${name} in the panel above and removed the rest.`,
      /** A2V+IC-LoRA combined-mode note (kept as this key by
       * A2V_ICLORA_COMBO_WORKORDER §2-3 rather than deleted): shown in BOTH
       * cards when a reference video and an audio file are attached together —
       * they now combine instead of excluding each other. */
      blockedByAudioNote:
        "Both a reference video and an audio file are attached. The a2v generation will be guided by the IC-LoRA reference video.",
      /** A2V+IC-LoRA combined-mode soft warning (non-blocking): the generated
       * duration outruns the reference video, so the tail generates without
       * IC-LoRA guidance. Seconds arrive pre-formatted (one decimal). */
      icLoraSpillWarning: (genSec: string, refSec: string): string =>
        `The generated duration (${genSec}s) is longer than the reference video (${refSec}s), so the latter part of the video will be generated without IC-LoRA control.`,
    },
    /** A2V (audio-to-video), moved here from Chain — group 3 item 11. Attach an
     * audio file to drive a single distilled clip's generation; width/height
     * and keyframe images still apply. Mutually exclusive with `referenceVideo`. */
    sourceAudio: {
      heading: "Audio to video (A2V)",
      note: "Attach an audio file to generate a video timed to it. Uses a single distilled clip; width/height and keyframe images still apply.",
      /** IC-LoRA/A2V card redesign: alt text for the attached audio file's
       * ♫ placeholder thumbnail. */
      audioPlaceholderAlt: "Audio file",
      chooseButton: "Choose audio file…",
      uploadingButton: "Uploading…",
      changeButton: "Change audio file…",
      clearButton: "Remove audio",
      none: "No audio file attached.",
      /** 台帳§1-16 (長尺A2V) rewrite: Create's A2V still uses exactly one clip,
       * but that is no longer the whole story now that Chained accepts one track
       * across up to 24 clips — so this line points at the other screen instead
       * of restating a limit the user cannot do anything about. */
      singleClipNote: "Long audio can use Chained's long a2v.",
      framesAdjustedToast: (frames: number, durationSec: number): string =>
        `Adjusted Duration to ${frames} frames to match the ${durationSec.toFixed(2)}s audio file.`,
      tooShortWarning: (requiredSeconds: number): string =>
        `The audio file is too short for the current duration. At least ${requiredSeconds.toFixed(2)}s of audio is required.`,
      /** A2V+IC-LoRA combined-mode note (same copy as
       * `referenceVideo.blockedByAudioNote`; kept as this key by
       * A2V_ICLORA_COMBO_WORKORDER §2-3 rather than deleted). */
      exclusiveWithReferenceNote:
        "Both a reference video and an audio file are attached. The a2v generation will be guided by the IC-LoRA reference video.",
    },
    /** W7: imperative one-line reasons the Generate button is disabled, shown
     * directly below it (`GenerateReasonsNote`). Each corresponds to a
     * `validityReasons` code from `useGenerationForm` (or a Create-screen-level
     * code composed in `SingleScreen`). `overflowKeyframes` reuses
     * `keyframes.outOfRangeGenerateHint` (not duplicated here); `referenceNotReady`
     * shares `attachReferenceVideo` with `controlNeedsReference` (they can both
     * hold at once, and the note de-dups identical lines). */
    generateReasons: {
      promptEmpty: "Fill the main prompt.",
      dimensionsOffGrid: "Set the width and height to valid values.",
      numFramesOffGrid: "Set the frame count to a valid value.",
      audioTooShort: "Use a longer audio clip, or lower the frame count.",
      referenceNeedsLoras: "Choose CONTROL LORA.",
      attachReferenceVideo: "Attach a reference video.",
      cropInvalid: "Fix the output crop size.",
      keyframeUploading: "Wait for the keyframe upload to finish.",
      audioUploading: "Wait for the audio upload to finish.",
      referenceUploading: "Wait for the reference video upload to finish.",
      /** NAG (2026-07-28): shown when "non-CFG Negative" is enabled but the
       * shared negative-prompt text is empty/whitespace-only — mirrors the
       * backend's own 422 for the same combination. */
      nagNegativeEmpty: "Fill the negative prompt, or turn off \"non-CFG Negative\".",
      /** §1-6 拡張 (2026-08-01): the reference video's ribbon range was requested
       * but not applied — the stored upload is the whole file, so generating
       * would condition on the wrong footage. Same remedy as Chain's
       * `chain.generateReasons.sourceTrimFailed`. */
      referenceTrimFailed:
        "The reference video's range could not be cut out. The plugin may be out of date — update it, or pick the file again to run with the whole video.",
    },
    generateButton: "Generate",
    generatingButton: "Generating…",
    /** U-R1: Generate button label while the server is occupied — another job
     * is running, or a model is being loaded (`JobsContext.serverBusy`). The
     * button is disabled, not queued (reservation removed 2026-07-17). */
    busyButton: "Busy…",
  },
  /** Shared note area (RIGHTCLICK_REDESIGN_SPEC.md §6): persistent, single-seat
   * right-click feedback shown above the operation panel. I5 seeds only the
   * fallback-insert note; the mismatch guidance (I6) and receipt notes (I7)
   * add their copy under this namespace next. Layer/frame numbers in note copy
   * are always shown +1 (1-based) to match AviUtl2's UI (§6-3). */
  notes: {
    /** §5-4 fallback notice: the cursor spot was occupied, so the provisional
     * was placed on the guaranteed-free frontmost layer instead. `layerNumber`
     * arrives already +1'd (1-based) by the caller. */
    insertedOnFrontmostLayer: (layerNumber: number): string =>
      `There was an object at the cursor, so it was inserted on the frontmost layer ${layerNumber}.`,
    /** I6 §4-3: the localized display name for an object kind, used inside the
     * mismatch template's 〔種別〕 / 〔required type〕 slots. */
    kindName: (kind: MaterialKind): string => {
      switch (kind) {
        case "video":
          return "video";
        case "image":
          return "image";
        case "audio":
          return "audio";
        case "text":
          return "text";
      }
    },
    /** I6 §4-2: more than one object is selected (single-selection only in α). */
    multipleSelection: "Please select just one object.",
    /** I6 §4-3: the object's kind could not be determined (unsupported type). */
    unsupportedType: "This object type is not supported.",
    /** I6 §4-3 mismatch template. `selected`/`required` are already-localized
     * kind display names (via `kindName`). Verbatim per spec ("a audio"). */
    typeMismatch: (selected: string, required: string): string =>
      `The selected object is a ${selected}. This action requires a ${required}.`,
    /** §4-3 mismatch template for actions that accept SEVERAL kinds (素材（末尾）
     * 2026-08-15). `required` is the already-localized list of accepted kind
     * names, joined with " or " — "a video or image". Same sentence shape as
     * `typeMismatch` above so the two read alike. */
    typeMismatchAny: (selected: string, required: readonly string[]): string =>
      `The selected object is a ${selected}. This action requires a ${required.join(" or ")}.`,
    /** I6 §4-3 near-neighbor guidance: a video was selected for the audio-object
     * a2v (#7); point the user at the video-audio a2v (#3) instead. */
    useVideoAudioInstead:
      `To use this video's sound, choose "🎬 Audio-to-video from this video's sound".`,
    /** §4-3 near-neighbor guidance, 台帳§1-16: the same mistake on the LONG-form
     * audio item — point at its own video sibling, not at the single-shot #3. */
    useVideoAudioLongInstead:
      `To use this video's sound, choose "🎬 Long a2v from this video's sound".`,
    /** I6 §4-5: the #1/#2 source/reference video is shorter than the length
     * floor. `requiredSeconds` is the exact threshold; shown rounded up. */
    videoTooShort: (requiredSeconds: number): string =>
      `This video is too short. A video of about ${Math.ceil(requiredSeconds)} seconds or longer is required.`,
    /** W0 §4 範囲系: Retake regenerates the SELECTED frame range, so with no
     * range selected there is nothing to retake. */
    rangeNotSelected:
      "Please select the range you want to redo on the timeline first, then run this command again.",
    /** W0 §4 範囲系: the selected range is shorter than the shortest window the
     * retake can regenerate. Reserved by W0; W3 owns the fps-dependent
     * judgement that actually raises it. */
    rangeTooShort: (requiredFrames: number): string =>
      `The selected range is too short. Please select at least ${requiredFrames} frames.`,
    /** I7 §5-6: a ⏳生成中 provisional already holds the single reservation seat,
     * so a new generation reservation is refused (busy-guard). */
    reservationBusy:
      "A new reservation cannot be made until the current generation finishes. Please try again after it completes.",
    /** §3-98 P5 / §3-102: the right-click command routes to a mode the LOADED
     * base model's engine cannot run (`shell/useBaseModels.ts`'s
     * `disabledModes`). Its tab is already greyed, so this note is what covers
     * the OTHER way into that mode — the timeline's context menu, which the
     * greying cannot reach.
     *
     * Worded as "this command", not "this tab": the user right-clicked an
     * object on the timeline and never saw a tab. Deliberately distinct from
     * `modes.unsupportedByBaseModel` (the tab tooltip) and from
     * `chained.unavailableOnBaseModel.*` (the material-panel lines) for the
     * same reason. */
    modeUnsupportedByBaseModel:
      "This command cannot be run on the selected base model. Switch the base model in the header and try again.",
    /** I8 §6: receipt note shown at the START of a #1/#6 auto-load — the
     * routed material's file name is now being loaded into the source slot. */
    loadedFromRightClick: (fileName: string): string => `Loaded from a right-click: ${fileName}`,
    /** I8 §6-2: the routed material's file couldn't be resolved or uploaded
     * (moved/deleted/unreadable). Shared by every material item (#1–#7). */
    sourceFileMissing: "The source file was not found. It may have been moved or deleted.",
    /** I9 §3-6 #10: 📷 imageFromCurrentFrame captured the current timeline
     * frame into a keyframe (the visible change is subtle, so a receipt note
     * confirms it). */
    capturedFrameToKeyframe: "Captured the current frame into a keyframe.",
    /** I9 §3-6 #10: the `timeline.captureFrame` round trip failed (or KEYFRAMES
     * was already full), so no frame could be captured. */
    captureFrameFailed: "Could not capture the current frame.",
    /** I10 §3-4 #3: shown while the video's sound is being extracted to a wav
     * (a few seconds). Bundles the mix-path explanation so the user knows the
     * timeline mixdown — not just the selected object — is what gets captured
     * (the note area shows one message at a time). */
    extractingAudio: "Extracting audio… This extracts the sound as it actually plays on the timeline.",
    /** I10 §3-4 #3 receipt: the extracted wav is now loaded into the audio slot
     * (the temp wav's own name is opaque, so this reads more naturally than the
     * shared loadedFromRightClick copy). */
    loadedVideoAudio: "Loaded the video's audio.",
    /** 台帳§1-16 🎵 audio-to-long-a2v: shown while the AUDIO OBJECT's own ribbon
     * range is being cut to a wav. Distinct from `extractingAudio` because this
     * one runs in SOLO mode — only the selected object's layer is audible — so
     * the mix-path sentence that note bundles would be actively wrong here. */
    extractingObjectAudio: "Extracting audio… This cuts the selected audio object's range.",
    /** 台帳§1-16 🎵 receipt: the solo-extracted wav is now in Chain's audio slot
     * (the temp wav's own name is opaque, hence not `loadedFromRightClick`). */
    loadedObjectAudio: "Loaded the audio object's sound.",
    /** I10 §3-4 #3 failure — the range was silent (EXTRACT_FAILED, "…silent
     * range"): a muted/disabled layer usually explains it. */
    audioExtractSilent:
      "The selected range has no audible sound, so nothing could be extracted. Check for muted or disabled layers.",
    /** I10 §3-4 #3 failure — extraction ran too long (EXTRACT_FAILED, per-frame
     * "audio render failed…", whose dominant cause is the worker's render
     * timeout). */
    audioExtractTimeout: "Audio extraction took too long and was aborted.",
    /** I10 §3-4 #3 failure — any other EXTRACT_FAILED (setup failure, wav write
     * failure, or an unclassifiable message). */
    audioExtractFailed: "Audio extraction failed.",
    /** I11 §3-4 #8: the right-clicked text object's body is empty, so nothing is
     * appended to the prompt (and no provisional is reserved). */
    promptTextEmpty: "The text is empty.",
    /** I11 §3-4 #8 receipt: the text was appended to the shared prompt. `head`
     * is the first ≤20 chars of the appended body; `truncated` adds the ellipsis
     * only when the body ran past 20 chars (§3-4 #8 追記細目). */
    appendedToPrompt: (head: string, truncated: boolean): string =>
      `Appended to the prompt: ${head}${truncated ? "…" : ""}`,
    /** I11 §3-4 #5: the keyframe panel is full (item cap reached or no free grid
     * slot), so the right-clicked image could not be appended. Mirrors the "Add
     * keyframe" button's disabled condition (atCapacity || noRoom). */
    keyframeLimitReached: "The keyframe limit has been reached.",
    /** I11 §3-4 #5 receipt: the right-clicked image was appended to a keyframe.
     * `fileName` names the loaded file (the visible new card is the primary
     * feedback; this reinforces it). */
    appendedImageToKeyframe: (fileName: string): string =>
      `Appended the image to a keyframe: ${fileName}`,
  },
  /** I12 §5-5 (revised I13, owner decision 2026-07-19): the 4-stage provisional
   * placeholder labels burned onto the timeline text object. Stages 1/2 are built
   * natively (`BuildProvisionalTextAlias` = prefix + display-head + "…"); the WebUI
   * supplies stage 1's fixed prefix/body (the reservation insert), stage 2's
   * "Generating: " prefix (`bindToJob`, no longer relying on native's default),
   * and the stage 3/4 TERMINAL text (overwritten whole on settle via
   * `timeline.updateProvisionalText`).
   *
   * These strings are DELIBERATELY emoji-free, plain ASCII English — and are the
   * SAME in `en` and `ja` (see `ja.provisional`). The AviUtl2 default font has no
   * emoji glyphs, so ⏳✅❌ rendered as tofu on the real timeline; the burned text
   * must therefore be font/locale-independent regardless of the UI language. The
   * PANEL note copy (`notes.*`) stays per-language — only what lands on the
   * timeline object is pinned to this English. */
  provisional: {
    /** Stage 1 prefix (段階1), passed as `insertProvisional.textPrefix`. Native
     * renders prefix + display-head + "…", where ONLY the display-head (the body
     * below) is truncated to 16 codepoints — the prefix is verbatim. The phrase
     * is split "AI video will be " (prefix, trailing space) + "placed here" (body,
     * 11 codepoints ≤ 16 so it is never clipped), giving the burned text
     * "AI video will be placed here…". Keep the body ≤16 codepoints if you reword
     * it, or move more of the phrase into the prefix. */
    reservedPrefix: "AI video will be ",
    /** Stage 1 body (段階1) — the tail of the stage-1 phrase; see `reservedPrefix`.
     * Must stay ≤16 codepoints so native's display-head truncation never clips it. */
    reservedBody: "placed here",
    /** Stage 2 prefix (段階2), passed as `updateProvisionalReservation.textPrefix`
     * by `bindToJob`. Burned text reads "Generating: 〔prompt head〕…". */
    generatingPrefix: "Generating: ",
    /** Stage 4 (段階4): the finished result is ready to insert by hand. Kept short
     * (owner decision): the 🎞-insert explanation was over-verbose. */
    done: "Done",
    /** Stage 3 (段階3): failure, with a human-readable `reason`. */
    failed: (reason: string): string => `Failed: ${reason}`,
    /** Stage 3 fallback reason when the job carries no error message. */
    failedGeneric: "unknown error",
    /** Stage 3 reason for a cancelled job — reads "Failed: canceled". */
    cancelled: "canceled",
  },
  /** 台帳§3-71/§3-72 (2026-09-02): copy for the right-click PREFILL itself —
   * messages about how a seeded value was adjusted on the way into the form,
   * as opposed to `settings`' copy about which policy decides the seed. */
  prefill: {
    /** Shown once, as a warning toast, when a right-click prefill had to round
     * a non-integer frame rate (an NTSC project's 29.97, or a 23.976 material)
     * to the whole frame rate the generation actually runs at. `from` arrives
     * pre-formatted by `jobs/fpsConvert.ts`'s `formatFps` (29.97 -> "29.97",
     * 23.976 -> "23.98"); `to` is the integer now in the field.
     *
     * Only the PREFILL route says anything — typing a rate by hand rounds
     * silently, because the field visibly shows the rounded value the same
     * instant (owner ruling 2026-09-02 ②). */
    fpsSnappedToast: (from: string, to: number): string =>
      `Frame rate ${from} was rounded to ${to} fps. Generation only runs at whole frame rates.`,
  },
  /** I8 §3-4 (※): confirmation dialogs. `chainDiscard` guards a #1/#6 Chain
   * remount that would throw away in-progress chain editing (only shown when
   * the Chain form is actually dirty). */
  dialogs: {
    chainDiscard: {
      title: "Discard Chain edits?",
      body: "Discard the current edits on the Chain screen and load this?",
      confirmButton: "OK",
      cancelButton: "Cancel",
    },
  },
  /** Copy shared across more than one mode/component, plus accessibility-only
   * text (aria-label/title) that never appeared in `strings.ts` before M7b
   * (task brief: "aria-labelやtitle属性も" must be covered). */
  common: {
    loadingPreview: "Loading preview…",
    promptLengthError: "Prompt must be 1-2000 characters.",
    dismissNotification: "Dismiss notification",
    modeAriaLabel: "Mode",
    loraTagsAriaLabel: "LoRA tags",
    decreaseStrength: (name: string): string => `Decrease ${name} strength`,
    increaseStrength: (name: string): string => `Increase ${name} strength`,
    removeLora: (name: string): string => `Remove ${name}`,
    /** Style LoRA audio strength control (2026-08-02): the chip's mute
     * toggle button, shown while the LoRA is currently audible (undefined or
     * >0 `audio_strength`) — clicking it mutes (sets audio strength to 0). */
    muteLoraAudio: (name: string): string => `Mute ${name} audio`,
    /** The same toggle button's label while the LoRA is currently muted
     * (`audio_strength === 0`) — clicking it restores audio strength to 1.0. */
    unmuteLoraAudio: (name: string): string => `Unmute ${name} audio`,
  },
  /** Contract v7: drag-and-drop file acceptance (`shell/useFileDrop.ts`),
   * shared by Create's reference-video/source-audio cards and Chain's
   * unified source input — each card is its own whole-card drop target now
   * (the old `DropZone` wrapper was retired in the IC-LoRA/A2V redesign).
   * `unsupportedVideo`/`unsupportedAudio`/
   * `unsupportedSource` each enumerate the extensions THAT slot accepts, so
   * they stay separate copy rather than one generic message. */
  dnd: {
    unsupportedVideo: "Unsupported file type. Choose a video (.mp4, .mov, .webm, .mkv).",
    unsupportedAudio: "Unsupported file type. Choose an audio file (.wav, .mp3, .m4a, .aac, .flac, .ogg).",
    unsupportedSource:
      "Unsupported file type. Choose an image (.png, .jpg, .jpeg, .webp) or a video (.mp4, .mov, .webm, .mkv).",
    /** Keyframe timeline rework (2026-07-18): image-only drop target's
     * unsupported-file message (each keyframe pin accepts a single image). */
    unsupportedImage: "Unsupported file type. Please drop an image file (.png, .jpg, .jpeg, .webp).",
    resolveFailed: "Could not resolve the dropped file. Please use the button above instead.",
  },
  /** M7b: connection settings panel (gear icon in the header). Bridge
   * contract v4's `settings.get`/`settings.set` (see `bridge/types.ts`). */
  settings: {
    title: "Settings",
    /** Accessible name for the header "×" dismiss button — distinct from
     * `close` (the footer button's own visible text) so the two buttons
     * don't share one accessible name (task brief: covers aria-labels too). */
    closeDialogAriaLabel: "Close settings",
    languageLabel: "Language",
    /** N8: theme toggle (dark/light), next to the language toggle. Persisted
     * client-side only (`shell/ThemeContext.tsx`'s `localStorage`), no
     * native/settings.json involvement. */
    themeLabel: "Theme",
    themeDark: "Dark",
    themeLight: "Light",
    /** W1/X6/§3-13: how a right-click prefill decides the generation size
     * (width/height) and, independently, the fps (`shell/PrefillPolicyContext.tsx`).
     * Persisted client-side only (`localStorage`), like the theme toggle above.
     * Affects only the right-click prefill's initial values — ordinary panel
     * edits, presets and "Get size from AviUtl2" are unchanged. X6: a shared
     * heading sits above both axes. §3-13: all three choices are live on BOTH
     * axes — the fps axis's "materials" reads the material's own framerate and
     * falls back to the project's fps when it can't be read. §3-71/§3-72
     * (2026-09-02): every fps stage this prefill can land on — material,
     * project, and the generation defaults fallback — is rounded to a whole
     * number by `modes/single/paramUtils.ts`'s `snapFrameRate` (the one source
     * of truth), not just the material tier. */
    rightClickMenuHeading: "Right-click menu:",
    prefillSizePolicyLabel: "Match Gen video size to the...",
    prefillFpsPolicyLabel: "Match Gen video FPS to the...",
    prefillPolicyDefaults: "Dev",
    prefillPolicyProject: "project",
    prefillPolicyMaterial: "materials",
    /** X6: tooltip on both axes' "Dev" button (its label is intentionally terse). */
    prefillPolicyDevTooltip: "Developmental defaults",
    /** Acceleration (2026-07-31, backend §43): five rows, all of them REAL as
     * of 2026-08-05, when the VAE row (the last mock) went live with backend
     * §52 (see `shell/accelerationSettings.ts`). The attention buttons' own
     * labels are NOT here: "sdpa"/"sage attention" are fixed technical names,
     * left untranslated on both this UI and the backend's Gradio one. */
    accelerationHeading: "Acceleration:",
    /** Acceleration (2026-08-04, backend §51, ledger §1-11): the fused GGUF
     * dequantization Triton kernel. Replaced the old `accelFusedGguf*` mock
     * keys ("Fused GGUF dequant + GEMM"), which were removed with the mock. */
    accelFusedGgufKernelLabel: "Fused GGUF Dequantization Kernel",
    accelFusedGgufKernelOn: "ON",
    accelFusedGgufKernelOff: "OFF",
    /** Shown under the row while it is on. Like `accelPrefetchNote` this is
     * reassurance, not a warning: the output is bit-identical either way. */
    accelFusedGgufKernelNote:
      "The output is identical to having it off — only how the model's compressed weights are unpacked changes, so the same seed reproduces the same frames exactly. It shortens each generation by roughly 20 seconds. If it cannot run on this machine, the previous method is used automatically and generation is unaffected.",
    accelAttentionLabel: "Attention",
    accelVaeLabel: "VAE (video decode)",
    accelVaeDefault: "Default",
    /** Technical name of the pruned VAE decoder — kept identical in both
     * dictionaries on purpose. The spelling settled on "PrunaVAED" on
     * 2026-08-05 (backend §52); the API VALUE stays `"prune_vaed"`, which is
     * an external contract and must not follow the display name. */
    accelVaePruneVaed: "PrunaVAED",
    /** Shown under the row while PrunaVAED is selected. UNLIKE the
     * prefetch/keep-resident/fused-kernel notes this IS a warning, in the same
     * spirit as `accelSageNote`: a different decoder is a different picture. */
    accelVaeNote:
      "A pruned decoder speeds up video reconstruction. Output quality may be slightly reduced. If the pruned decoder is missing on the server, the usual one is used automatically and generation is unaffected.",
    /** Tooltip on the sage button while `GET /status` explicitly reports
     * SageAttention as missing (never shown while availability is unknown). */
    accelSageUnavailableTooltip: "SageAttention is not installed on the server, so it cannot be selected.",
    /** Shown under the attention row while sage is selected. Both halves
     * matter: identical seeds no longer reproduce identical output down to
     * the last detail, and the payoff is roughly 1.2-1.6x. */
    accelSageNote:
      "Fine details of the output change even with the same seed (the numeric precision differs). Speed is about 1.2-1.6x.",
    /** Acceleration (2026-08-01, backend §44): block-swap prefetch. */
    accelPrefetchLabel: "Block-swap prefetch",
    accelPrefetchOn: "ON",
    accelPrefetchOff: "OFF",
    /** Tooltip on the On button while `GET /status` explicitly reports
     * block-swap prefetch as unavailable (never shown while unknown). */
    accelPrefetchUnavailableTooltip: "The server is not running with block swap, so this has no effect.",
    /** Shown under the row while prefetch is on. Unlike `accelSageNote`, this
     * is reassurance, not a warning: the output is bit-identical either way. */
    accelPrefetchNote:
      "The output is identical to having it off — only the transfer method changes, so the same seed reproduces the same frames exactly. Roughly 10-13% faster.",
    /** Acceleration (2026-08-02, backend §48): keep the model skeleton
     * resident between jobs. */
    accelKeepResidentLabel: "Keep the model skeleton resident (cache between jobs)",
    accelKeepResidentOn: "ON",
    accelKeepResidentOff: "OFF",
    /** §1-10 (2026-08-03): tooltip on BOTH buttons while block-swap prefetch
     * is off or unavailable — keep-resident only pays off alongside it, so the
     * row is greyed out and shows Off until prefetch comes back. */
    accelKeepResidentPrefetchOffTooltip: "Available only while block-swap prefetch is enabled.",
    /** Shown under the row while keep-resident is on. Like
     * `accelPrefetchNote` this is not an output warning — the RAM cost is
     * what the reader needs at that moment, so it sits in the very first
     * sentence next to the recommendation. Wording set by the owner on
     * 2026-08-06 after the real-device gate: the measured "70s -> 10s" pair
     * and the bit-identical aside were dropped as noise for the reader. */
    accelKeepResidentNote:
      "64GB or more of memory recommended: LTX 2.3 keeps about 20GB of main memory resident, LTX 2.5 about 8GB (2.5 only keeps the part that reads the prompt). Keeps the model's CPU-side skeleton between jobs, greatly shortening the preprocessing of the second and later generations. The output does not change.",
    backendUrlLabel: "Backend URL",
    backendUrlPlaceholder: "http://127.0.0.1:18620",
    loadingCurrent: "Loading current settings…",
    loadError: "Could not load the current backend URL.",
    save: "Save",
    saving: "Saving…",
    invalidUrl: "Invalid backend URL. Use a URL such as http://127.0.0.1:18620.",
    close: "Close",
    /** N11/N10: config-derived sections at the bottom of the panel, both
     * fed by a single `useConfig()` call (`GET /config`,
     * Docs/API_REFERENCE.md §3.2). */
    configLoading: "Loading configuration…",
    spillFreeSectionTitle: "Spill-free frame limits",
    spillFreeHint:
      "The frame count per resolution that avoids VRAM overflow and keeps generation from slowing down 2-4x. (Provisional)",
    spillFreeResolutionHeader: "Resolution",
    spillFreeFramesHeader: "Frames",
    rawConfigSectionTitle: "Raw /config",
    rawConfigFallbackNote: "Showing the built-in fallback configuration — the server could not be reached.",
    /** N13: API-key status badge — shows whether the server has one
     * configured, never the key itself. */
    apiKeySectionTitle: "API key",
    apiKeyHint: "Whether the server has an API key configured. The key itself is never shown here.",
    apiKeySet: "Configured",
    apiKeyUnset: "Not configured",
    apiKeyUnknown: "Unknown",
    /** N4: "danger zone" section — an explicit pipeline unload and a bulk
     * purge of every finished (completed/failed/cancelled) job. Both actions
     * require a second, explicit confirmation click (see
     * `DangerZonePanel.tsx`) — there is no undo for either. */
    dangerZoneSectionTitle: "Danger zone",
    dangerZoneHint: "These actions are immediate and cannot be undone.",
    unloadButton: "Unload pipeline",
    unloadConfirm: "This frees the loaded engine's VRAM. Unload now?",
    unloadConfirmButton: "Unload now",
    dangerCancelButton: "Cancel",
    unloadDone: "Pipeline unloaded.",
    unloadBusy: "Cannot unload while a generation job is running.",
    unloadError: "Failed to unload the pipeline.",
    purgeButton: "Purge finished jobs",
    purgeConfirm: "This permanently deletes every finished job (completed, failed, or cancelled). Purge now?",
    purgeConfirmButton: "Purge now",
    purgeDone: (count: number): string => `Deleted ${count} finished job${count === 1 ? "" : "s"}.`,
    purgeNone: "No finished jobs to delete.",
    /** Some (or all) of the terminal jobs found by the sweep failed to
     * delete — distinct from `purgeNone` (there was nothing to purge at
     * all). */
    purgePartial: (deleted: number, attempted: number): string =>
      `Deleted ${deleted} of ${attempted} finished job(s); the rest could not be removed.`,
    purgeError: "Failed to fetch the job list.",
  },
  /** Model management (S1): the Settings panel's "Models" section —
   * per-category (transformer/text_encoder/video_vae/audio) selection +
   * `POST /pipeline/load` (Docs/API_REFERENCE.md §3.3/§3.5). Independent of
   * the connection-settings save flow above (mirrors `gradio_ui/ui.py`'s
   * "Independent section" ruling for the same feature). */
  models: {
    sectionTitle: "Models",
    hint: "Selections apply when you press Load. \"default\" is the stock configuration.",
    /** Owner ruling 2026-08-20 on the wording: "checkpoint" rather than the
     * internal component name "transformer", because that is what the model
     * sites (CivitAI et al.) call the file the user downloads; and no engine
     * name on the text encoder, so a base model that ships something other
     * than Gemma needs no string change here. */
    categories: {
      transformer: "Video model (checkpoint)",
      text_encoder: "Text encoder",
      video_vae: "Video VAE",
      audio: "Audio model (audio VAE + vocoder)",
    },
    refreshButton: "Refresh model list",
    loadButton: "Load selected models",
    loadingNotice: "Loading models… switching rebuilds the engine and can take several minutes.",
    fetchError: "Failed to fetch the model list.",
    loadSuccess: (summary: string): string => `Models loaded: ${summary}`,
    loadFailed: (detail: string): string => `Model load failed: ${detail}`,
    missingSuffix: "file missing",
    jobBusy: "Cannot switch models while a generation job is running.",
    errorHints: {
      MODEL_NOT_FOUND: "Unknown model name. Refresh the model list and choose again.",
      MODEL_FILE_MISSING: "The model file is missing on disk. Re-download it or choose another model.",
      MODEL_INCOMPATIBLE: "The selected file is not valid for this category. Choose another model.",
      PIPELINE_LOAD_FAILED: "Failed to load the model pipeline. Check the server's VRAM and logs, then retry.",
    },
  },
};

/** Structural shape both dictionaries must satisfy — every leaf in `ja` must
 * exist at the exact same path as in `en` (checked by `strings.test.ts`).
 * Deliberately *not* `typeof en` from an `as const` object: without a const
 * assertion, plain string properties widen to `string` (not their English
 * literal value), so `ja`'s translations aren't forced to reuse `en`'s text. */
export type Strings = typeof en;

export const ja: Strings = {
  toolVersion: {
    ariaLabel: "ベースモデル",
    unknown: "ベースモデル",
    optionNotInstalled: (displayName: string): string => `${displayName}（未導入）`,
    optionPartial: (displayName: string): string => `${displayName}（一部未導入）`,
    switched: (displayName: string): string => `${displayName}へ切り替えました。`,
    notInstalled: (displayName: string, installer: string): string =>
      `${displayName}はまだ導入されていません。バックエンドのフォルダにある${installer}をダブルクリックして重みファイルを取得したあと、この画面を開き直してから選び直してください。`,
    switchFailedBusy: "生成中はベースモデルを切り替えられません。",
    switchFailedLoading: "モデルの読み込み中です。完了してから切り替えてください。",
    switchFailedRejected: (reason: string): string => `ベースモデルを切り替えられませんでした: ${reason}`,
    switchFailed: (message: string): string => `ベースモデルを切り替えられませんでした: ${message}`,
  },
  connection: {
    checking: "接続中…",
    connected: (pluginVersion: string): string => `接続済み (${pluginVersion})`,
    disconnected: "ネイティブブリッジに接続できません",
  },
  editInfo: {
    button: "編集情報を取得",
    buttonBusy: "取得中…",
    heading: "編集情報",
    empty: "まだ編集情報を取得していません。",
    fields: {
      width: "幅",
      height: "高さ",
      rate: "フレームレート",
      scale: "スケール",
      sampleRate: "サンプルレート",
      frame: "フレーム",
    },
  },
  error: {
    heading: "エラー",
  },
  serverStatus: {
    checking: "接続中…",
    bridgeUnavailable: "ネイティブブリッジに接続できません",
    offline: "サーバー未起動",
    online: "サーバー稼働中",
    busy: "生成中",
    loadingModels: "モデル読み込み中…",
    error: "状態取得エラー",
    retry: "再試行",
  },
  modes: {
    toolbox: "Toolbox",
    single: "Single",
    chained: "Chained",
    edit: "Edit",
    inventory: "Inventory",
    unsupportedByBaseModel: "選択中のベースモデルでは使えません。このタブを使うには、上のベースモデルを切り替えてください。",
  },
  edit: {
    subTabsAriaLabel: "編集ツールの切り替え",
    subTabs: {
      retake: "Retake",
      outpainting: "Outpainting",
      inpainting: "Inpainting",
    },
    unavailableOnBaseModel: {
      retake:
        "Retake（撮り直し）は、選択中のベースモデルでは使えません。使うには、上のベースモデルを切り替えてください。",
      outpainting:
        "Outpainting（画角拡張）は、選択中のベースモデルでは使えません。使うには、上のベースモデルを切り替えてください。",
    },
    retake: {
      heading: "Retake（リテイク：撮り直し）",
      summary:
        "動画の選択された区間を再生成する機能です。選択されていない部分はそのまま残ります。（最小73f。短すぎると元動画からほとんど変化しません）",
      rangeBand: {
        label: "撮り直す区間",
        startHandle: "撮り直しの開始位置",
        endHandle: "撮り直しの終了位置",
        startValueText: (frame: number, seconds: string): string => `開始 ${frame}フレーム（${seconds}秒）`,
        endValueText: (frame: number, seconds: string): string => `終了 ${frame}フレーム（${seconds}秒）`,
        placementReadout: (startFrame: number, endFrame: number): string =>
          `タイムラインの ${startFrame}〜${endFrame} フレーム目を撮り直します。（※フレーム数は「8の倍数+1」です）`,
        readout: (frames: number, seconds: string): string => `（窓 ${frames}フレーム・${seconds}秒）`,
        maxNote: (maxFrames: number): string =>
          `撮り直せる長さは最大 ${maxFrames} フレームです。これより長くは伸ばせません。`,
        glueNote:
          "区間の両端にある縞模様は、前後の映像となめらかにつなぐための「のりしろ」です。ここには元の映像がそのまま残り、撮り直されるのはその内側だけです。のりしろだけを動かすことはできません。",
      },
      idle: "タイムライン上で動画を右クリックし、撮り直したい範囲を選んでから「選択範囲を撮り直す」を選んでください。",
      awaitingSource:
        "素材を外しました。タイムライン上で動画を右クリックし、撮り直したい範囲を選んでから「選択範囲を撮り直す」を選んでください。下の設定はこのまま残ります（幅と高さは新しい素材に合わせ直されます）。",
      clearButton: "撮り直しを取りやめて設定を戻す",
      resetSourceButton: "別の場所を撮り直す（この設定のまま）",
      sourceHeading: "素材",
      sourceThumbAlt: "動画を設定済み",
      uploading: "素材を読み込んでいます…",
      sourceReadout: (
        fileName: string,
        seconds: string,
        frames: number,
        fps: number,
        width: number,
        height: number,
      ): string => {
        const size = width > 0 && height > 0 ? `／${width}×${height}` : "";
        return `${fileName} ${seconds}秒 ${frames}フレーム（${fps}fps換算）${size}`;
      },
      sourceFixedNote: "素材は右クリックした動画で固定されています。ここでは差し替えられません。",
      loadFailed: "素材を読み込めませんでした。いったんこのタブを離れて、右クリックからやり直してください。",
      trimFailed:
        "この動画はタイムライン上で一部だけを使っていますが、その部分を切り出せませんでした。このまま進むと違う場所を撮り直してしまうため、続行できません。",
      audioHeading: "撮り直す対象",
      audioBoth: "映像と音声",
      audioVideoOnly: "映像のみ（音声は元のまま）",
      glueHint: (headFrames: number, tailFrames: number): string =>
        `のりしろは前 ${headFrames} フレーム・後ろ ${tailFrames} フレームです。`,
      noticesHeading: "生成する前に",
      noticeDifferent:
        "撮り直した区間は別のテイクになります。同じ指示文でも、元の映像と1コマ単位で同じにはなりません。",
      noticeGlueQuality:
        "のりしろの部分は元の映像を保ちますが、一度だけ再エンコードされるため、画質はごく僅かに変化します。",
      noticeSilentAudio:
        "この区間に音が入っていない素材で「映像と音声」を選ぶと、無音だった部分に音が付きます。無音のままにしたいときは「映像のみ」を選んでください。",
      staleWarning:
        "このパネルを開いてから、タイムラインの選択が変わりました。撮り直されるのは、ここに表示されている範囲です。",
      generateButton: "選択範囲を撮り直す",
      generatingButton: "送信中…",
      generateReasons: {
        sourceMissing: "素材がまだ読み込まれていません。",
        sourceUploading: "素材の読み込みが終わるのを待っています。",
        sourceUploadFailed: "素材を読み込めませんでした。",
        sourceTrimFailed: "タイムラインで使っている部分を切り出せませんでした。",
        rangeUnusable:
          "選んだ範囲からは撮り直す区間を作れません。動画が実際に再生されている部分の中で範囲を選び直してください。",
        outOfMaterial: "素材が、撮り直せる最短の区間よりも短いです。",
      },
    },
    outpainting: {
      heading: "Outpainting",
      sourceHeading: "素材",
      sourceHint: "動画ファイルをドラッグ＆ドロップしてください。",
      sourceThumbAlt: "動画を設定済み",
      chooseButton: "元になる動画を選ぶ",
      changeButton: "元になる動画を選び直す",
      uploadingButton: "アップロード中…",
      clearButton: "元になる動画を外す",
      none: "元になる動画がまだ設定されていません。",
      sourceReadout: (width: number, height: number, seconds: string): string =>
        `${width}×${height}ピクセル／${seconds}秒`,
      padsHeading: "描き足す量",
      padLeft: "左",
      padRight: "右",
      padTop: "上",
      padBottom: "下",
      gridNote: "縦横は128ピクセルの倍数でなければならない。",
      centeringLabel: "センタリング",
      blurLabel: "マスクブラー",
      blurZero: "0px（膨張なし）",
      blendNote: (px: number): string =>
        `拡張の境界から内側約${px}pxは、元動画と拡張部分が馴染むようにマスクブラーがかかる（※拡張部分にマスクの緑色が残る場合は、マスクブラーを強めること）`,
      blurWarning: "この値では元動画のほぼ全体が混ぜ合わせの対象になります。",
      canvasReadout: (width: number, height: number): string => `${width} × ${height} pxサイズに拡張して生成`,
      numFramesLabel: "フレーム数",
      previewAlt: (
        canvasWidth: number,
        canvasHeight: number,
        sourceWidth: number,
        sourceHeight: number,
      ): string =>
        `プレビュー：${canvasWidth}×${canvasHeight}ピクセルに広げたキャンバスの中に、元の${sourceWidth}×${sourceHeight}ピクセルの動画が置かれています。`,
      previewLegendSource: "青い四角：元動画のサイズ",
      previewLegendCanvas: "青線：拡大後の動画サイズ",
      previewLegendBlend: (px: number): string => `破線：ピクセルが維持される範囲（境界から約${px}px内側）`,
      comfortWarning: (tokens: number): string =>
        `この大きさと長さでは、処理量がおよそ${tokens}単位になります。これは、このパソコンで映像用メモリー（VRAM）が足りなくなり始める目安を超えています。実行はできますが遅くなることがあります。描き足す量を減らすか、尺を短くしてください。`,
      spillWarning: "生成速度が2〜4倍遅くなる可能性があります。",
      generateButton: "生成",
      generatingButton: "生成中…",
      generateReasons: {
        sourceMissing: "広げたい動画を設定してください。",
        sourceUploading: "元になる動画のアップロード完了を待ってください。",
        sourceUploadFailed: "元になる動画をアップロードできませんでした。ファイルを選び直してください。",
        sourceTrimFailed:
          "元動画の範囲切り出しが適用されませんでした。プラグインが古い可能性があります。プラグインを更新するか、ファイルを選び直すと全体を使って実行できます。",
        mediaInfoUnknown:
          "元になる動画の大きさと長さを読み取れませんでした。ファイルを選び直すか、別の形式の動画を使ってください。",
        padsZero: "上下左右のどれかに、描き足す量を設定してください。",
        canvasWidthOffGrid: "幅は128の倍数でなければなりません。",
        canvasHeightOffGrid: "高さは128の倍数でなければなりません。",
        canvasWidthTooLarge: "幅の上限は4096ピクセルです。",
        canvasHeightTooLarge: "高さの上限は4096ピクセルです。",
        innerTooSmall: (minSide: number): string =>
          `元の動画は縦横とも${minSide}ピクセル以上が必要です。もっと大きい動画を使ってください。`,
        loraMissing: (loraName: string): string =>
          `制御用の追加学習データ「${loraName}」がサーバーに入っていません。導入したうえで、Inventory（在庫）の画面で一覧を読み込み直してください。`,
      },
    },
    comingSoon: "この機能はまだ準備中です。ここは画面の枠だけを先に用意したもので、操作できる項目はまだありません。",
  },
  promptBar: {
    label: "プロンプト",
    placeholder: "生成したい映像を文章で説明してください…",
  },
  nag: {
    heading: "ネガティブプロンプト",
    onBadge: "【🔴ON】",
    textLabel: "ネガティブプロンプト",
    enableLabel: "non-CFG Negative",
    methodNag: "NAG",
    methodVsf: "VSF",
    methodVsfHint: "実用域はscale 1.5〜5。0にしても無効にはなりません（無効化はチェックを外してください）。",
    scaleLabel: "NAG scale",
    vsfScaleLabel: "VSF scale",
    advancedHeading: "詳細設定",
    tauLabel: "NAG tau（正規化しきい値）",
    alphaLabel: "NAG alpha（混合率）",
    resetTooltip: "スライダーを既定値に戻します。",
  },
  chained: {
    heading: "Chained",
    modeBadge: {
      v2v: "V2V継続（続きを生成）",
      scratch: "ゼロから連結",
    },
    configFallbackWarning:
      "サーバーからプリセット・上限値を取得できませんでした — 内蔵の既定値を使用しています。実際のサーバーと値が異なる場合があります。",
    unavailableOnBaseModel: {
      sourceVideo:
        "素材に動画を使うこと（動画の続きを生成する＝V2V）は、選択中のベースモデルでは使えません。この欄のほかの機能はそのまま使えます——クリップ1の開始画像は影響を受けません。動画を読み込むには、上のベースモデルを切り替えてください。",
      sourceAudio:
        "音声から生成（A2V）は、選択中のベースモデルでは使えません。連結生成そのものは使えます。音声を読み込むには、上のベースモデルを切り替えてください。",
      endSource:
        "素材（末尾）は、選択中のベースモデルでは使えません。連結生成そのものは使えます。末尾の素材を読み込むには、上のベースモデルを切り替えてください。",
      referenceVideo:
        "参照動画（制御系IC-LoRA）は、選択中のベースモデルでは使えません。連結生成そのものは使えます。参照動画を読み込むには、上のベースモデルを切り替えてください。",
    },
    presets: {
      label: "プリセット",
      placeholder: "プリセットを選択…",
    },
    clipsHeading: "クリップ",
    addClipButton: "クリップを追加",
    removeClipButton: "クリップを削除",
    clipLabel: (index: number): string => `クリップ ${index + 1}`,
    clipPromptLabel: "プロンプトの上書き",
    clipPromptPlaceholder: "空欄の場合は上の共通プロンプトを使用します",
    clipNumFramesLabel: "尺",
    clipAudioWindow: {
      label: (startSec: string, endSec: string): string => `🎵 担当時間帯 ${startSec}〜${endSec}秒`,
      tooltip: "このクリップが担当する音声の区間です。隣のクリップと少し重なります（継ぎ目のクロスフェード）。",
    },
    minClipsError: (min: number): string => `最低 ${min} 本のクリップが必要です（元動画が未指定のため）。`,
    maxClipsReached: (max: number): string => `クリップは最大 ${max} 本までです。`,
    totalFramesLabel: (total: number, max: number): string => `合計入力: ${total} / ${max} フレーム`,
    totalFramesExceeded: (max: number): string => `全クリップの合計フレーム数は ${max} を超えられません。`,
    outputFramesLabel: (frames: number, seconds: number): string =>
      `予想出力: ≈${seconds.toFixed(1)}秒（${frames} フレーム）`,
    outputFramesWithTailLabel: (frames: number, seconds: number, tailFrames: number, tailSeconds: number): string =>
      `予想出力: ≈${seconds.toFixed(1)}秒（${frames} フレーム）・うち末尾${tailSeconds.toFixed(1)}秒（${tailFrames} フレーム）は素材`,
    overlapHeading: "つなぎ目のブレンド",
    overlapFramesLabel: "ブレンド幅",
    overlapStrengthLabel: "ブレンド強度",
    chunkedUpsampleLabel: "チャンク化アップサンプル",
    chunkedUpsampleHint:
      "長尺でもGPUメモリを節約するチャンク方式でアップサンプルする。通常はオンのまま推奨。",
    stage2Window: {
      label: "Stage-2（アップスケール工程）のクリップ長",
      standardOption: "潜在22フレーム（約7.0秒）",
      highResolutionOption: "潜在19フレーム（約6.0秒／VRAM溢れ軽減）",
      hint:
        "潜在22フレーム推奨。潜在19フレームでは高解像度動画でもVRAM溢れが起こりにくくなりますが、動画のドリフトやアーティファクトが増えます。\n\n【解説】LTX 2.3の動画生成は、Stage-1（低解像度で仮動画を生成） → Stage-2（仮動画をアップスケール） → VAEデコードという工程を経ます。連結クリップ生成では、Stage-1ではユーザー指定のクリップ長を繋げて、長い仮動画を生成します。この仮動画は長すぎるため、Stage-2で丸ごとアップスケールできません。そこで、動画冒頭から一定の長さで区切ってアップスケールを行い、後から再連結します。この再連結時の長さを指定するためのドロップダウンリストです。（実時間ではなく、潜在表現でのフレーム数を指定します）",
      overBudgetWarning: "VRAM溢れにより生成が遅くなる可能性があります。",
      overBudgetShorterWindowHint: "潜在19フレームで軽減できる可能性があります。",
    },
    sourceInput: {
      heading: "素材（冒頭）",
      note: "入力された素材（動画 or 画像）に続く動画を生成します。",
      chooseButton: "画像または動画を選択…",
      chooseImageOnlyButton: "画像を選択…",
      changeButton: "変更…",
      uploadingButton: "アップロード中…",
      clearButton: "クリア",
      none: "素材が選択されていません。",
      videoPlaceholderAlt: "元動画",
      strengthLabel: "強さ",
      sourceReadout: (fileName: string, width: number, height: number): string =>
        width > 0 && height > 0 ? `${fileName}／${width}×${height}` : fileName,
      unsupportedType:
        "対応していないファイル形式です。画像（.png, .jpg, .jpeg, .webp）または動画（.mp4, .mov, .webm, .mkv）を選択してください。",
    },
    sourceVideo: {
      contextFramesLabel: "引き継ぎフレーム数",
      contextFramesHint: (frames: number): string => `元動画から ${frames} フレームを引き継ぎます。`,
      contextFramesError: "引き継ぎフレーム数は、先頭クリップの尺より少なくしてください。",
    },
    endSource: {
      heading: "素材（末尾）",
      note: "入力された素材（動画 or 画像）で終わる動画を生成します。",
      chooseButton: "画像または動画を選択…",
      changeButton: "変更…",
      uploadingButton: "アップロード中…",
      clearButton: "素材（末尾）を外す",
      none: "素材（末尾）は未添付です。",
      videoPlaceholderAlt: "素材（末尾）の動画",
      contextFramesHint: (frames: number): string =>
        `生成される動画の末尾 ${frames} フレームが、素材の冒頭で固定されます（動画の長さは変わりません）。`,
      stillImageNote: "末尾8フレームは静止画になります。",
      tooShortNote: "素材（末尾）は9フレーム以上必要です。",
      qualityWarning: (maxFrames: number): string =>
        `クリップが ${maxFrames} フレームを超えると、生成される動画の品質が下がります。`,
      multiClipQualityWarning: "素材（末尾）つきで複数クリップを連結すると、生成結果の品質が低下します。",
      reverseOverlapHint: "複数クリップの素材（末尾）に合わせて、つなぎ目のブレンド幅を1に設定しました。手動で変更できます。",
      lengthUnknownNote: "素材（末尾）の長さを取得できませんでした。添付し直すか、別のファイルを使ってください。",
      audioModeNote: "素材に音声があれば、その音声も末尾に取り込まれます（音声の無い素材・静止画では、音声は独立して生成されます）。",
      strengthLabel: "素材の固定強度",
      strengthHint:
        "1.00で素材のとおりに終わります。下げると素材へのなじみ方がゆるくなります（最終フレームは素材のまま・音声は強度によらず常に素材のままです）。",
      conflictsWithAudio: "素材（末尾）とa2vは併用できません。",
      conflictsWithReference: "素材（末尾）とIC-LoRAは併用できません。",
    },
    sourceAudio: {
      heading: "音声から動画（A2V）",
      note: "音声を1本添付すると、各クリップに担当区間が自動で割り当てられます。ソース動画（V2V）とは併用できません。",
      audioPlaceholderAlt: "音声ファイル",
      chooseButton: "音声ファイルを選ぶ…",
      changeButton: "音声ファイルを変更…",
      uploadingButton: "アップロード中…",
      clearButton: "音声を外す",
      none: "音声ファイルは未添付です。",
      adjustButton: "↔️再生時間の自動調整",
      addedClipFramesLabel: "追加クリップの長さ",
      tooLongError: (maxSeconds: number): string =>
        `音声が長すぎます。Chainedで扱える上限は約${maxSeconds}秒（24クリップ×481フレーム）です。短い音声を使ってください。`,
      conflictsWithSourceVideo: "v2vとa2vは併用できません",
      probeFailed: "音声の長さを取得できませんでした。自動調整は使えません（長さはサーバー側で検査されます）。",
      atMaxClipsNote: (unusedSeconds: string): string =>
        `動画の尺が足りないため、音声の末尾 約${unusedSeconds}秒は使われません。追加クリップの長さを上げるか、音声を短くしてください。`,
      surplusNote: (unusedSeconds: string): string =>
        `動画の尺が足りないため、音声の末尾 約${unusedSeconds}秒は使われません。`,
      fitToast: {
        adjusted: (durationSec: number): string =>
          `音声の長さ（${durationSec.toFixed(2)}秒）に合わせてクリップ構成を調整しました。`,
        alreadyFits: "クリップ構成はすでに音声の長さに合っています。",
        cannotFit:
          "調整できませんでした。音声に対してクリップ構成が長すぎるか、追加クリップの長さが大きすぎます。クリップを減らすか、追加クリップの長さを下げてください。",
        noFlexibleClips:
          "調整できませんでした。すべてのクリップが手動編集済みのため、変更できるカードがありません。クリップを追加してください。",
      },
    },
    referenceVideo: {
      heading: "参照動画（IC-LoRA）",
      fpsNote:
        "fpsは、参照動画と生成動画で一致させることを推奨。参照動画は（秒数ではなく）フレーム単位で参照されるため。",
      videoPlaceholderAlt: "参照動画",
      chooseButton: "参照動画を選ぶ…",
      changeButton: "参照動画を変更…",
      uploadingButton: "アップロード中…",
      clearButton: "参照動画を外す",
      resolutionLabel: (width: number, height: number): string => `${width}×${height}`,
      controlLoraLabel: "制御LoRA",
      controlLoraNoneOption: "選択してください",
      controlLoraStrengthLabel: "制御LoRAの強さ",
      conditioningAttentionStrengthLabel: "制御強度（conditioning attention strength）",
      referenceVideoStrengthLabel: "参照強度（reference video strength）",
      conflictsWithSourceVideo: "v2v（動画延長）とIC-LoRAは併用できません。",
      probeFailed: "参照動画の解像度を取得できませんでした。",
      shortReferenceNote:
        "参照動画が生成の尺より短い場合、足りない分は参照なしで生成されます。",
      none: "参照動画は未添付です。",
      stage1OverBudgetWarning:
        "長すぎるクリップがあるため、VRAM溢れにより生成速度が低下するリスクがあります。各クリップのフレーム数を短くするか、動画の解像度を下げることを推奨します。",
    },
    generateReasons: {
      clipFramesOffGrid: "各クリップのフレーム数を有効な値にしてください。",
      uploadInFlight: "アップロード完了を待ってください。",
      clipTooShortForOverlap: (n: number): string =>
        `現在のつなぎ目ブレンド設定では、各クリップは${n}フレーム以上が必要です`,
      sourceVideoTooShortForContext:
        "元動画の使用範囲が、引き継ぎフレーム数に対して短すぎます。範囲を長くするか、引き継ぎフレーム数を下げてください。",
      sourceTrimFailed:
        "元動画の範囲切り出しが適用されませんでした。プラグインが古い可能性があります。プラグインを更新するか、ファイルを選び直すと全体を使って実行できます。",
      contextFramesTooLongForWindow: (maxFrames: number): string =>
        `現在のStage-2のクリップ長では、元動画から引き継げるのは最大${maxFrames}フレームです。引き継ぎフレーム数を下げるか、クリップ長を潜在22フレーム（標準）に戻してください。`,
      audioConflictsWithSourceVideo: "v2vとa2vは併用できません。元動画か音声のどちらかを外してください。",
      audioNotReady: "音声を添付し直してください（アップロードに失敗しました）。",
      audioTooShort: "音声が短すぎます。動画を短くするか、↔️再生時間の自動調整を押してください。",
      audioTooShortForChain: "短い音声はSingleで生成してください。Chainedには2クリップ以上が必要です。",
      audioTooLong: "音声が長すぎます。短い音声を使ってください。",
      chainLayoutInvalid:
        "このフレームレートではクリップ構成が音声タイムラインと整合しません。↔️再生時間の自動調整を押すか、クリップの長さを変えてください。",
      referenceNotReady: "参照動画を添付し直してください（アップロードに失敗しました）。",
      referenceTrimFailed: "参照動画を添付し直してください（範囲の切り出しに失敗しました）。",
      referenceConflictsWithSourceVideo:
        "v2v（動画延長）とIC-LoRAは併用できません。どちらかを外してください。",
      referenceDimensionsOffGrid:
        "参照動画を使う場合、幅と高さはどちらも128の倍数にしてください。",
      depthChainUnsupported: "未実装のIC-LoRAアダプタが選択されています。",
      endSourceUploading: "素材（末尾）のアップロード完了を待ってください。",
      endSourceNotReady: "素材（末尾）を添付し直してください（アップロードに失敗しました）。",
      endSourceTrimFailed: "素材（末尾）を添付し直してください（範囲の切り出しに失敗しました）。",
      endSourceConflictsWithAudio: "素材（末尾）とa2vは併用できません。どちらかを外してください。",
      endSourceConflictsWithReference: "素材（末尾）とIC-LoRAは併用できません。どちらかを外してください。",
      endSourceWithSourceVideoMultiClip:
        "素材（冒頭）と素材（末尾）を両方使うときは、クリップは1本だけです（複数クリップでの併用は今後対応予定です）。",
      endSourceTooShort: "素材（末尾）は9フレーム以上必要です。長い動画を使ってください。",
      endSourceLengthUnknown:
        "素材（末尾）の長さを取得できませんでした。添付し直すか、別のファイルを使ってください。",
      endSourceNeedsOverlap: "素材（末尾）はのりしろ1では使えません。ブレンド幅を2以上にしてください。",
      endSourceAudioOverlapBudget:
        "このクリップ構成では、素材（末尾）のつなぎ目に必要な音声のりしろが足りません。ブレンド幅を上げてください。",
    },
    generateButton: "生成",
    generatingButton: "生成中…",
    busyButton: "処理中…",
  },
  batch: {
    heading: "バッチA2V（一括生成）",
    notice:
      "選んだフォルダ内のすべての音声から、1ファイルにつき1本の動画をまとめて生成します。（出力フォルダは音声フォルダの隣に自動作成されます）",
    wavDir: {
      label: "音声フォルダ",
      button: "音声フォルダを選択…",
      none: "音声フォルダが選択されていません。",
    },
    imgDir: {
      label: "画像フォルダ（任意）",
      button: "画像フォルダを選択…",
      none: "画像フォルダが選択されていません — 行ごとの画像は音声フォルダを代わりに使用します。",
    },
    outDir: {
      label: "出力フォルダ",
      button: "出力フォルダを選択…",
      auto: (path: string): string => `自動: ${path}`,
      none: "出力フォルダは未設定です — まず音声フォルダを選択してください。",
    },
    promptMode: {
      label: "行プロンプトの合成方法",
      add: "共通プロンプトに追記",
      replace: "共通プロンプトを置き換え",
    },
    sharedKeyframes: {
      /** バッチA2VのShared仕様変更（2026-07-18）：画像列が「Shared」の行は、
       * Create画面のKEYFRAMES欄にある1枚目の画像を常に冒頭フレーム（frame 0）
       * として使うようになった。2枚目以降の画像は使われない。 */
      note: "画像列が「Shared」の行には、Create画面のKEYFRAMES欄にある1枚目の画像が冒頭フレーム（フレーム0）として使われます。2枚目以降の画像は使われません。",
      missingWarning:
        "画像列が「Shared」の行がありますが、Create画面のKEYFRAMES欄に画像がまだ設定されていません。Sharedを使うには、そちらに画像を1枚以上追加してください。",
    },
    scanButton: "フォルダをスキャン",
    scanningButton: "スキャン中…",
    scanError: (message: string): string => `スキャンに失敗しました: ${message}`,
    table: {
      queue: "#",
      wav: "音声ファイル",
      duration: "長さ",
      image: "画像",
      prompt: "プロンプト",
      stat: "状態",
      output: "出力",
      resetButton: "待機に戻す",
      empty: "行がまだありません — 音声フォルダを選んでスキャンしてください。",
      promptInputLabel: "行のプロンプト",
      copyCommonPromptButton: "共通プロンプトを流し込む",
      imageSelectLabel: "行の画像",
    },
    stat: {
      waiting: "待機中",
      generating: "生成中",
      done: "完了",
      failed: "失敗",
      skip: "対象外",
    },
    summary: (done: number, total: number): string => `${done} / ${total} 件完了`,
    currentRow: (wav: string): string => `処理中: ${wav}`,
    startButton: "バッチを開始",
    runningButton: "実行中…",
    stoppingButton: "停止処理中…",
    stopButton: "停止",
    resolutionOffGrid: "幅と高さは64の倍数にしてください。開始する前に、上の「作る」フォームでサイズを修正してください。",
    fpsMismatch: "FPSまたはDURATIONがスキャン時と異なります。フレーム数はバッチ開始時に現在の値で自動的に再計算されます。",
    icLoraActiveWarning: "「作る」フォームで参照動画（IC-LoRA）が有効なため、バッチはその128グリッドの解像度を使用します。",
    lockedByOther: "クリップチェーン画面でバッチi2v-longが実行中です。バッチは同時に1つしか実行できません。終わるまで待つか、そちらで停止してください。",
    jobActive: "サーバーが処理中です（ほかの生成が実行中か、モデルを読み込み中です）。同時に1つしか動かせないため、終わるまで待ってから開始してください。",
    unavailableOnBaseModel: "バッチA2Vは選択中のベースモデルでは使えません（連結生成を使うため）。使うには、上のベースモデルを切り替えてください。",
    skipReasons: {
      "wav-only-alpha": "wav形式でないか、長さを読み取れませんでした",
      "over-cap":
        "音声が現在のDURATION上限を超えています。DURATIONを上げる（またはFPSを下げる）→🔁で待機に戻す→開始、の順で再実行できます",
    },
  },
  batchI2vLong: {
    heading: "バッチi2v-long（プロトタイプ）",
    chainSummary: (clips: number, frames: number, seconds: number): string =>
      `クリップ${clips}本 ・ ${frames}フレーム ・ 画像1枚あたり約${Math.round(seconds)}秒`,
    imgDir: {
      label: "画像フォルダ",
      button: "画像フォルダを選択…",
      none: "画像フォルダが選択されていません。",
    },
    outDir: {
      label: "出力フォルダ",
      button: "出力フォルダを選択…",
      auto: (path: string): string => `自動: ${path}`,
      none: "出力フォルダは未設定です — まず画像フォルダを選択してください。",
    },
    promptMode: {
      label: "行プロンプトの合成方法",
      add: "クリップチェーンのプロンプトに追記",
      replace: "クリップチェーンのプロンプトを置き換え",
    },
    scanButton: "フォルダをスキャン",
    scanningButton: "スキャン中…",
    scanError: (message: string): string => `スキャンに失敗しました: ${message}`,
    table: {
      queue: "#",
      image: "画像",
      prompt: "プロンプト",
      stat: "状態",
      output: "出力",
      resetButton: "待機に戻す",
      empty: "行がまだありません — 画像フォルダを選んでスキャンしてください。",
      promptInputLabel: "行のプロンプト",
      copyChainPromptButton: "クリップチェーンのプロンプトを流し込む",
    },
    stat: {
      waiting: "待機中",
      generating: "生成中",
      done: "完了",
      failed: "失敗",
    },
    summary: (done: number, total: number): string => `${done} / ${total} 件完了`,
    currentRow: (image: string): string => `処理中: ${image}`,
    startButton: "バッチを開始",
    runningButton: "実行中…",
    stoppingButton: "停止処理中…",
    stopButton: "停止",
    blockReasons: {
      imgDirMissing: "実行する画像フォルダを選択してください。",
      outDirMissing: "出力フォルダを選択してください。",
      noRows: "先に画像フォルダをスキャンしてください。",
      noRunnableRows: "すべての行が完了しています。もう一度実行するには、行の🔁を押してください。",
      sourceVideoAttached:
        "クリップチェーンに元動画が設定されています。解除してください — 各画像がそれぞれチェーンの先頭になるため、元動画があるとすべての画像が無視されます。",
      sourceAudioAttached:
        "クリップチェーンに音声が設定されています。外してください — このバッチは画像だけから生成し、音声は送信しません。",
      referenceVideoAttached:
        "IC-LoRAの参照動画を外してください。このバッチは画像だけから生成します。",
      endSourceAttached:
        "素材（末尾）を外してください。このバッチは画像だけから生成し、末尾の素材は送信しません。",
      clipsTooFew: "元動画を使わない場合、クリップチェーンにはクリップが2本以上必要です。クリップを追加してください。",
      promptEmpty: "プロンプトを入力してください（クリップチェーンの共通プロンプトか、行ごとのプロンプト欄のどちらかで）。",
      promptTooLong: "合成後のプロンプトが長すぎます。2000文字以内にしてください。",
      unknownLoraTag: "プロンプトに存在しないLoRA名が含まれています。タグを修正するか削除してください。",
      jobActive: "サーバーが処理中です（他のジョブが実行中か、モデルを読み込み中です）。終わるまで待ってください。",
      lockedByOther: "作成画面のバッチA2Vが実行中です。バッチは同時に1つしか実行できません。停止するか、終わるまで待ってください。",
    },
    promptRowIssues: {
      empty: (queues: string): string =>
        `プロンプトが空のまま送信される行があります: ${queues}。クリップチェーンの共通プロンプトか、その行のプロンプト欄に入力してください。`,
      tooLong: (queues: string): string =>
        `合成後のプロンプトが2000文字を超えている行があります: ${queues}。クリップチェーンの共通プロンプトか、その行のプロンプトを短くしてください。`,
    },
    notes: {
      seedFixed: "シードが固定されているため、すべての画像が同じシードで生成されます。画像ごとに変えたい場合は、クリップチェーンのシードを-1にしてください。",
      clip0PromptOverride:
        "1本目のクリップに個別のプロンプトが入力されています。画像は必ず1本目のクリップに乗るため、共通プロンプトも行ごとのプロンプトも、画像が効く部分にはまったく反映されません。",
      clipPromptOverride: "個別のプロンプトが入力されたクリップがあります。共通プロンプトも行ごとのプロンプトも、それらを書き換えません。",
      startFrameIgnored: "クリップチェーンの開始フレーム画像は使われません。各行の画像がその位置に入ります。",
      concurrency: "バッチは同時に1つしか実行できません。実行中はクリップチェーンの生成ボタンも使えなくなります。",
    },
  },
  inventory: {
    heading: "LoRA",
    reload: "🔁リロード",
    reloading: "再読み込み中…",
    reloadToast: (total: number, styles: number, controls: number): string =>
      `LoRAを再読み込みしました: 合計 ${total} 件（画風 ${styles} 件、制御 ${controls} 件）。`,
    emptyLoras: "LoRAが見つかりません。models/LTX23/StyleLoRA にファイルを置いてから再読み込みしてください。",
    loadError: "LoRA一覧を取得できませんでした。",
    loadingLoras: "LoRAを読み込み中…",
    addedToast: (name: string): string => `プロンプトに <lora:${name}:1.0:1.0> を追加しました。`,
    historyHeading: "履歴",
    historyNote: "ジョブ履歴はサーバー上のメモリに保持されており、再起動で消去されます。",
    historyEmpty: "完成したジョブはまだありません。",
    noPromptPlaceholder: "（プロンプトなし）",
  },
  jobs: {
    heading: "ジョブ",
    empty: "ジョブはまだありません — 生成されたクリップはここに表示されます。",
    status: {
      queued: "待機中",
      running: "生成中",
      completed: "完了",
      failed: "失敗",
      cancelled: "キャンセル済み",
    },
    preview: "プレビュー",
    closePreview: "プレビューを閉じる",
    seedRandom: "ランダム",
    insert: "挿入",
    downloading: "ダウンロード中…",
    inserting: "挿入中…",
    inserted: "挿入済み",
    cancel: "キャンセル",
    cancelling: "キャンセル中…",
    delete: "削除",
    deleting: "削除中…",
    toastCompleted: (jobId: string): string => `ジョブ ${jobId.slice(0, 8)} が完了しました。`,
    toastFailed: (jobId: string): string => `ジョブ ${jobId.slice(0, 8)} が失敗しました。`,
    toastCancelled: (jobId: string): string => `ジョブ ${jobId.slice(0, 8)} はキャンセルされました。`,
    clipBadge: (clip: number, count: number): string => `クリップ ${clip}/${count}`,
    join: "元動画とつなぐ",
    crossfadeLabel: "音声クロスフェード",
    crossfadeOption: (ms: number): string => `${ms} ミリ秒`,
    trimLengthLabel: "元動画の残し幅",
    trimLengthOption: (frames: string, sec: string): string => `${frames}フレーム（≒${sec}秒）`,
    joining: "結合中…",
    joinRetry: "結合をやり直す",
    unjoin: "元に戻す (Unjoin)",
    joinSourceFpsNote: (sourceFps: string, videoFps: string): string =>
      `元動画のfps（${sourceFps}）が生成動画（${videoFps}）と一致しなかったため、結合時に変換しました。`,
    joinNormalizedNote: "生成動画の形式に合わせて元動画を再変換しました。",
    joinProjectFpsAdvice: (projectFps: string, videoFps: string): string =>
      `プロジェクトと生成動画のfpsは揃えることを推奨します（project ${projectFps} / video ${videoFps}）。`,
    insertJoined: "結合済み動画を挿入",
    genericFailure: "生成に失敗しました。",
  },
  menuInsert: {
    notProvisional: "Nz-Videomniの予約オブジェクトを右クリックしてください。",
    jobNotFound: "この予約オブジェクトに対応するジョブが見つかりませんでした。",
    inserted: "生成結果を挿入しました。",
    insertFailed: "生成結果の挿入に失敗しました。",
    notReady: "このジョブはまだ生成中です。完了してからもう一度お試しください。",
    jobFailed: "このジョブは正常に完了していないため、挿入できる結果がありません。",
    noCompleted: "挿入できる完了済みの生成結果がまだありません。",
    insertedLatest: "最新の生成結果を挿入しました。",
    insertLatestFailed: "最新の生成結果の挿入に失敗しました。",
    noCursor: "挿入位置を特定できませんでした。",
  },
  single: {
    heading: "Single",
    configFallbackWarning:
      "サーバーからプリセット・上限値を取得できませんでした — 内蔵の既定値を使用しています。実際のサーバーと値が異なる場合があります。",
    loadingConfig: "設定を読み込み中…",
    spillWarning: "これを超えると生成速度が2〜4倍遅くなります。",
    // 賢い快適上限マーカー（2026-08-18）: マーカー自体が賢い解像度別上限
    // （`useGenerationForm` の `isComfortMarkerSmart`）のときは上の
    // `spillWarning` の代わりにこちらを出す——「2〜4倍」はLTX 2.3の既定構成の
    // 実測であり、賢い線を引ける構成（LTX 2.5は無条件・LTX 2.3は加速全オンのみ
    // ——`shell/comfortTable.ts` の `resolveComfortRow` 参照）では成り立たないため。
    // キー名に `Smart` を付けたのは、別機能の既存キー2つと混同しないため:
    // `single.size.comfortMarkerTitle`（Chained画面の解像度スライダー用ガイドの
    // ツールチップ、全く別機能）と `edit.comfortWarning`（Outpaintingの別軸トークン
    // 予算の警告——`modes/edit/outpaintGeometry.ts` の `COMFORT_TOKEN_BUDGET`
    // 参照）。
    comfortWarningSmart: "VRAM溢れにより生成が遅くなる可能性があります。",
    size: {
      label: "サイズ",
      width: "幅",
      height: "高さ",
      getFromAviUtl2: "AviUtl2からサイズを取得",
      gettingFromAviUtl2: "取得中…",
      comfortMarkerTitle: (px: number): string =>
        `Stage-2（アップスケール工程）の快適上限の目安：${px}px。16:9に近い形で釣り合う推奨点です。もう一方の値を下げれば、この線より上でも快適な範囲に収まることがあります。参照動画を使うときのStage-1側の負荷は含みません。`,
      limitMarkerTitle: (px: number): string =>
        `もう一方の現在値のままで、この項目が快適な範囲に収まる上限：${px}px。`,
    },
    crop: {
      label: "出力をクロップ",
      hint: "生成後の映像を指定サイズにクロップします（※最小32px）",
      width: "クロップ幅",
      height: "クロップ高さ",
    },
    duration: {
      label: "尺",
      fps: "フレームレート（fps）",
    },
    seed: {
      label: "シード",
      hint: "-1 = ランダム",
      randomTooltip: "シードをランダムにする（-1）",
      reuseTooltip: "直近のジョブのシードを再利用",
    },
    presets: {
      label: "プリセット",
      placeholder: "プリセットを選択…",
      orientation: {
        landscape: "横長",
        square: "正方形",
        portrait: "縦長",
      },
    },
    cfgScale: {
      label: "CFGスケール",
    },
    reservedNote: "CFGは現在のバックエンドでは未対応です（将来の使用のために予約されています）",
    keyframes: {
      label: "キーフレーム（I2V）",
      count: (count: number, max: number): string => `${count} / ${max}`,
      captureButton: "現在のフレームを取り込む",
      capturingButton: "取り込み中…",
      chooseFileButton: "画像ファイルを選択…",
      choosingButton: "選択中…",
      maxReached: (max: number): string => `キーフレームは最大 ${max} 枚までです。`,
      frameIdxLabel: "フレーム",
      willSnapTo: (frame: number): string => `${frame} に補正されます`,
      outOfRangeWarning: "現在の尺の範囲外です。",
      strengthLabel: "強さ",
      removeButton: "削除",
      uploading: "アップロード中…",
      modeT2V: "T2V",
      modeI2V: (count: number): string => `I2V（キーフレーム${count}枚）`,
      modeA2V: "A2V",
      modeICLora: "IC-LoRA",
      timelineLabel: "キーフレームのタイムライン",
      pinAriaLabel: (index: number): string => `キーフレーム${index}`,
      pinValueText: (frame: number, seconds: string): string => `${frame}f / ${seconds}s`,
      intervalLabel: (frames: number, seconds: string): string => `${frames}f / ${seconds}s`,
      /** Keyframe card polish (2026-07-18): 位置0のキーフレームカードに出す
       * バッジ。LTX 2.3のエンジンは位置0だけ特別扱いで、位置0はlatent置換に
       * より開始フレームが完全固定（ピクセル一致）になる一方、位置1以降は
       * あくまで誘導（完全一致は保証されない）という挙動差がある。このバッジ
       * はその差をユーザーに伝えるためのもの。 */
      startFrameFixedBadge: "開始フレーム（固定）",
      addKeyframeButton: (count: number, max: number): string => `キーフレームを追加（ ${count} / ${max} ）`,
      addNoRoomHint: "最後のキーフレームより後ろに空きがありません。ピンを動かすか削除すると追加できます。",
      emptyCardHint:
        "画像が未設定です。ここに画像をドロップするか、右上のボタンで選んでください。未設定のまま生成すると、このキーフレームは送信されません。",
      willLandAt: (frame: number): string => `位置 ${frame} に配置されます`,
      outOfRangeGenerateHint: "現在の尺の範囲外にあるキーフレームがあります。位置を調整するか削除すると生成できます。",
      shrinkModal: {
        title: "尺を短くしますか？",
        body: (numFrames: number, count: number): string =>
          `尺を ${numFrames} フレームに縮めると、範囲外になるキーフレーム ${count} 枚が削除されます。`,
        applyButton: "適用",
        cancelButton: "キャンセル",
      },
    },
    referenceVideo: {
      heading: "参照動画（IC-LoRA）",
      icLoraNote: "IC-LoRAは /generate 専用です。参照動画を選択してください。",
      controlAdapterAspectHint:
        "参照動画は出力解像度へ単純にリサイズされるため、アスペクト比が異なると映像が歪みます（特にDepth controlで目立ちます）。",
      controlAdapterDepthHint:
        "Depth control（深度制御）: 公式推奨は制御強度（conditioning attention strength）=0.6です。制御LoRAの強さは1.0のままにしてください——下げると参照が滲み込みます（この2つは別のつまみです）。",
      controlAdapterDeblurHint:
        "Deblur（ぼけ除去）: プロンプトは2段構成で書きます。例（公式モデルカードより）: 「Reference shows <場面の説明>, heavily out of focus with soft defocused blur and no fine detail. Edited shows the same scene in sharp focus with crisp detail and clean edges. DEBLUR <場面の説明>. Subject identity, framing, and background geometry are identical to the reference; only focus and sharpness differ between reference and edited.」対象はデフォーカスぼけ（ピンぼけ）のみで、モーションブラーには効きません。参照動画は縮小せずに条件付けに使われます。",
      videoPlaceholderAlt: "参照動画",
      chooseButton: "参照動画を選択…",
      changeButton: "参照動画を変更…",
      uploadingButton: "アップロード中…",
      clearButton: "クリア",
      removeIcLoraButton: "IC-LoRAを解除（参照動画と制御LoRAをクリア）",
      none: "参照動画が選択されていません。",
      lorasRequiredWarning: "CONTROL LORAを選択してください",
      conditioningAttentionStrengthLabel: "制御強度（conditioning attention strength）",
      referenceVideoStrengthLabel: "参照強度（reference video strength）",
      controlLoraLabel: "制御LoRA",
      controlLoraNoneOption: "なし",
      controlLoraStrengthLabel: "制御LoRAの強さ",
      referenceVideoRequiredWarning: "この制御LoRAには参照動画が必要です。上で選択してください。",
      controlLoraMigratedToast: (name: string): string =>
        `<lora:${name}:...> タグを上の制御LoRAパネルへ移動しました。`,
      controlLoraMigratedMultiple: (name: string): string =>
        `複数の制御LoRAタグが見つかったため、${name} のみを上のパネルに残し、残りは削除しました。`,
      blockedByAudioNote: "参照動画と音声の両方が入力されています。IC-LoRAで制御されたa2vが生成されます。",
      icLoraSpillWarning: (genSec: string, refSec: string): string =>
        `生成する尺（${genSec}秒）が参照動画（${refSec}秒）よりも長いため、動画後半はIC-LoRA制御なしで生成されます。`,
    },
    sourceAudio: {
      heading: "音声から動画（A2V）",
      note: "音声ファイルを添付すると、その音声に合わせた動画を生成します。単一のdistilledクリップを使用し、幅・高さやキーフレーム画像は通常どおり適用されます。",
      audioPlaceholderAlt: "音声ファイル",
      chooseButton: "音声ファイルを選ぶ…",
      uploadingButton: "アップロード中…",
      changeButton: "音声ファイルを変更…",
      clearButton: "音声を外す",
      none: "音声ファイルは未添付です。",
      singleClipNote: "長い音声はChainedのlong a2vが使えます。",
      framesAdjustedToast: (frames: number, durationSec: number): string =>
        `音声の長さ（${durationSec.toFixed(2)}秒）に合わせて尺を${frames}フレームに調整しました。`,
      tooShortWarning: (requiredSeconds: number): string =>
        `音声ファイルが現在の尺に対して短すぎます。少なくとも${requiredSeconds.toFixed(2)}秒の音声が必要です。`,
      exclusiveWithReferenceNote: "参照動画と音声の両方が入力されています。IC-LoRAで制御されたa2vが生成されます。",
    },
    generateReasons: {
      promptEmpty: "メインプロンプトを入力してください。",
      dimensionsOffGrid: "幅と高さを有効な値にしてください。",
      numFramesOffGrid: "フレーム数を有効な値にしてください。",
      audioTooShort: "音声が短すぎます。音声を長くするかフレーム数を減らしてください。",
      referenceNeedsLoras: "制御LoRAを選んでください。",
      attachReferenceVideo: "参照動画を設定してください。",
      cropInvalid: "出力クロップのサイズを直してください。",
      keyframeUploading: "キーフレームのアップロード完了を待ってください。",
      audioUploading: "音声のアップロード完了を待ってください。",
      referenceUploading: "参照動画のアップロード完了を待ってください。",
      nagNegativeEmpty: "ネガティブプロンプトを入力するか、「non-CFG Negative」をオフにしてください。",
      referenceTrimFailed:
        "参照動画の範囲切り出しが適用されませんでした。プラグインが古い可能性があります。プラグインを更新するか、ファイルを選び直すと全体を使って実行できます。",
    },
    generateButton: "生成",
    generatingButton: "生成中…",
    busyButton: "処理中…",
  },
  notes: {
    insertedOnFrontmostLayer: (layerNumber: number): string =>
      `カーソル位置に既存オブジェクトがあったため、最前面のレイヤー${layerNumber}に挿入しました。`,
    kindName: (kind: MaterialKind): string => {
      switch (kind) {
        case "video":
          return "動画";
        case "image":
          return "画像";
        case "audio":
          return "音声";
        case "text":
          return "テキスト";
      }
    },
    multipleSelection: "対象のオブジェクトを1つだけ選んでください。",
    unsupportedType: "対応していない種類のオブジェクトです。",
    typeMismatch: (selected: string, required: string): string =>
      `選択されているのは${selected}です。この操作には${required}が必要です。`,
    typeMismatchAny: (selected: string, required: readonly string[]): string =>
      `選択されているのは${selected}です。この操作には${required.join("か")}が必要です。`,
    useVideoAudioInstead: "動画の音声を使うには『🎬 この動画の音声でa2v』を選んでください。",
    useVideoAudioLongInstead:
      "動画の音声を使うには『🎬 この動画の音声でlong a2v』を選んでください。",
    videoTooShort: (requiredSeconds: number): string =>
      `この動画は短すぎます。約${Math.ceil(requiredSeconds)}秒以上の動画が必要です。`,
    rangeNotSelected:
      "先にタイムライン上で撮り直したい範囲を選択してから、もう一度実行してください。",
    rangeTooShort: (requiredFrames: number): string =>
      `選択された範囲が短すぎます。${requiredFrames}フレーム以上を選択してください。`,
    reservationBusy:
      "前の生成が完了するまで、新しい生成予約はできません。完了後にもう一度お試しください。",
    modeUnsupportedByBaseModel:
      "この操作は選択中のベースモデルでは実行できません。上のベースモデルを切り替えてから、もう一度お試しください。",
    loadedFromRightClick: (fileName: string): string => `右クリックから読み込みました: ${fileName}`,
    sourceFileMissing: "素材ファイルが見つかりません（移動または削除された可能性があります）。",
    capturedFrameToKeyframe: "現在フレームをキーフレームに取り込みました。",
    captureFrameFailed: "現在フレームの取り込みに失敗しました。",
    extractingAudio: "音声を抽出しています…（タイムラインで実際に鳴っている音を抽出します）",
    loadedVideoAudio: "動画の音声を読み込みました。",
    extractingObjectAudio: "音声を切り出しています…（選択した音声オブジェクトの範囲を切り出します）",
    loadedObjectAudio: "音声オブジェクトの音を読み込みました。",
    audioExtractSilent:
      "選択範囲の音声が無音のため抽出できませんでした（ミュートやレイヤー無効化を確認してください）。",
    audioExtractTimeout: "音声の抽出に時間がかかりすぎたため中断しました。",
    audioExtractFailed: "音声の抽出に失敗しました。",
    promptTextEmpty: "テキストが空です。",
    appendedToPrompt: (head: string, truncated: boolean): string =>
      `プロンプトに追記しました: ${head}${truncated ? "…" : ""}`,
    keyframeLimitReached: "キーフレームの上限に達しました。",
    appendedImageToKeyframe: (fileName: string): string =>
      `画像をキーフレームに追記しました: ${fileName}`,
  },
  /** タイムラインの仮オブジェクトに焼き込む文言は、UI言語設定に関わらず常に英語
   * （ASCII）で固定する（オーナー決定 2026-07-19）。AviUtl2の既定フォントには絵文字
   * グリフが無く ⏳✅❌ が豆腐になるため、フォント・環境に依存しないASCII英語へ統一
   * した。したがって ja でも en とまったく同じ英文を用いる（パネルのノート欄
   * `notes.*` は従来どおり日本語のまま）。個々のキーの意図は `en.provisional`
   * のコメントを参照。 */
  provisional: {
    reservedPrefix: "AI video will be ",
    reservedBody: "placed here",
    generatingPrefix: "Generating: ",
    done: "Done",
    failed: (reason: string): string => `Failed: ${reason}`,
    failedGeneric: "unknown error",
    cancelled: "canceled",
  },
  prefill: {
    fpsSnappedToast: (from: string, to: number): string =>
      `フレームレート ${from} を ${to} fps に丸めました。生成は整数のフレームレートでのみ動きます。`,
  },
  dialogs: {
    chainDiscard: {
      title: "Chainの編集内容を破棄しますか？",
      body: "Chain画面の編集中の内容を破棄して読み込みますか？",
      confirmButton: "OK",
      cancelButton: "キャンセル",
    },
  },
  common: {
    loadingPreview: "プレビューを読み込み中…",
    promptLengthError: "プロンプトは1〜2000文字で入力してください。",
    dismissNotification: "通知を閉じる",
    modeAriaLabel: "モード",
    loraTagsAriaLabel: "LoRAタグ",
    decreaseStrength: (name: string): string => `${name} の強さを下げる`,
    increaseStrength: (name: string): string => `${name} の強さを上げる`,
    removeLora: (name: string): string => `${name} を削除`,
    muteLoraAudio: (name: string): string => `${name} の音声をミュート`,
    unmuteLoraAudio: (name: string): string => `${name} の音声のミュートを解除`,
  },
  dnd: {
    unsupportedVideo: "対応していないファイル形式です。動画（.mp4, .mov, .webm, .mkv）を選択してください。",
    unsupportedAudio: "対応していないファイル形式です。音声ファイル（.wav, .mp3, .m4a, .aac, .flac, .ogg）を選択してください。",
    unsupportedSource:
      "対応していないファイル形式です。画像（.png, .jpg, .jpeg, .webp）または動画（.mp4, .mov, .webm, .mkv）を選択してください。",
    unsupportedImage: "対応していないファイル形式です。画像ファイル（.png、.jpg、.jpeg、.webp）をドロップしてください。",
    resolveFailed: "ドロップされたファイルを解決できませんでした。上のボタンから選択し直してください。",
  },
  settings: {
    title: "設定",
    closeDialogAriaLabel: "設定を閉じる",
    languageLabel: "言語",
    themeLabel: "テーマ",
    themeDark: "ダーク",
    themeLight: "ライト",
    rightClickMenuHeading: "右クリックからの動画生成時:",
    prefillSizePolicyLabel: "サイズはどれに合わせる？",
    prefillFpsPolicyLabel: "fpsはどれに合わせる？",
    prefillPolicyDefaults: "開発用",
    prefillPolicyProject: "プロジェクト",
    prefillPolicyMaterial: "素材",
    prefillPolicyDevTooltip: "開発用の既定値",
    accelerationHeading: "生成の高速化:",
    accelFusedGgufKernelLabel: "GGUF脱量子化カーネルの1本化",
    accelFusedGgufKernelOn: "ON",
    accelFusedGgufKernelOff: "OFF",
    accelFusedGgufKernelNote:
      "圧縮されたモデルの重みを展開する処理をまとめて実行し、生成を高速化します。生成結果は変わりません（同じシードなら完全に同じ絵になります）。1回の生成あたり約20秒短縮します。この環境で動かせない場合は自動的に従来の方法に切り替わり、生成そのものには影響しません。",
    accelAttentionLabel: "Attention機構の実装",
    accelVaeLabel: "VAE（映像の復元処理）",
    accelVaeDefault: "標準",
    accelVaePruneVaed: "PrunaVAED",
    accelVaeNote:
      "枝刈りした復元処理で映像の書き出しを高速化します。出力品質がわずかに低下する可能性があります。サーバー側に枝刈り版が入っていない場合は自動的に通常の処理に切り替わり、生成そのものには影響しません。",
    accelSageUnavailableTooltip: "サーバーにSageAttentionが導入されていないため選択できません",
    accelSageNote: "同じシードでも生成結果の細部が変わります（数値精度が異なるため）。速度は約1.2〜1.6倍",
    accelPrefetchLabel: "Block swapの先読み",
    accelPrefetchOn: "ON",
    accelPrefetchOff: "OFF",
    accelPrefetchUnavailableTooltip: "サーバーがブロック入れ替えを使わない設定のため、効果がありません",
    accelPrefetchNote:
      "ブロック転送を計算と並行して先読みし、生成を高速化します。生成結果は変わりません（同じシードなら完全に同じ絵になります）。速度は約10〜13%短縮",
    accelKeepResidentLabel: "モデル骨格の常駐（ジョブ間キャッシュ）",
    accelKeepResidentOn: "ON",
    accelKeepResidentOff: "OFF",
    accelKeepResidentPrefetchOffTooltip: "先読みblock swapが有効なときだけ使えます",
    accelKeepResidentNote:
      "メモリ64GB以上を推奨（LTX 2.3では約20GB、LTX 2.5では約8GBをメインメモリに常駐で使用します。2.5で常駐するのは文章を読み取る部分だけです）。モデルのCPU側骨格をジョブ間で保持し、2回目以降の生成の前処理を大幅に短縮します。生成結果は変わりません。",
    backendUrlLabel: "接続先URL",
    backendUrlPlaceholder: "http://127.0.0.1:18620",
    loadingCurrent: "現在の設定を読み込み中…",
    loadError: "現在の接続先URLを取得できませんでした。",
    save: "保存",
    saving: "保存中…",
    invalidUrl: "接続先URLが無効です。http://127.0.0.1:18620 のような形式で入力してください。",
    close: "閉じる",
    configLoading: "設定を読み込み中…",
    spillFreeSectionTitle: "快適フレーム上限",
    spillFreeHint: "各解像度における、VRAM溢れが起きずに生成が2～4倍遅くならないフレーム数です。（暫定版）",
    spillFreeResolutionHeader: "解像度",
    spillFreeFramesHeader: "フレーム数",
    rawConfigSectionTitle: "生の /config",
    rawConfigFallbackNote: "サーバーに接続できなかったため、内蔵のフォールバック設定を表示しています。",
    apiKeySectionTitle: "APIキー",
    apiKeyHint: "サーバーにAPIキーが設定されているかどうかを表示します。キーの値そのものはここには表示されません。",
    apiKeySet: "設定済み",
    apiKeyUnset: "未設定",
    apiKeyUnknown: "不明",
    dangerZoneSectionTitle: "危険な操作",
    dangerZoneHint: "これらの操作は即座に実行され、元に戻せません。",
    unloadButton: "パイプラインを解放",
    unloadConfirm: "読み込み済みのエンジンのVRAMを解放します。今すぐ解放しますか？",
    unloadConfirmButton: "今すぐ解放",
    dangerCancelButton: "取消",
    unloadDone: "パイプラインを解放しました。",
    unloadBusy: "生成ジョブの実行中はパイプラインを解放できません。",
    unloadError: "パイプラインの解放に失敗しました。",
    purgeButton: "終了したジョブを一括削除",
    purgeConfirm: "終了したジョブ（完了・失敗・キャンセル済み）をすべて完全に削除します。今すぐ削除しますか？",
    purgeConfirmButton: "今すぐ削除",
    purgeDone: (count: number): string => `終了したジョブを${count}件削除しました。`,
    purgeNone: "削除対象の終了したジョブはありません。",
    purgePartial: (deleted: number, attempted: number): string =>
      `終了したジョブ${attempted}件中${deleted}件を削除しました（残りは削除できませんでした）。`,
    purgeError: "ジョブ一覧の取得に失敗しました。",
  },
  models: {
    sectionTitle: "モデル",
    hint: "選択は「読込」ボタンで反映されます。default は標準構成です。",
    categories: {
      transformer: "動画モデル (checkpoint)",
      text_encoder: "テキストエンコーダ",
      video_vae: "動画VAE",
      audio: "音声モデル (音声VAE+ボコーダ)",
    },
    refreshButton: "モデル一覧を更新",
    loadButton: "選択したモデルを読込",
    loadingNotice: "モデルを読込中… 切替はエンジンの再構築を伴うため数分かかることがあります。",
    fetchError: "モデル一覧の取得に失敗しました。",
    loadSuccess: (summary: string): string => `モデルを読み込みました: ${summary}`,
    loadFailed: (detail: string): string => `モデルの読込に失敗しました: ${detail}`,
    missingSuffix: "ファイルなし",
    jobBusy: "生成ジョブの実行中はモデルを切り替えられません。",
    errorHints: {
      MODEL_NOT_FOUND: "不明なモデル名です。モデル一覧を更新して選び直してください。",
      MODEL_FILE_MISSING: "モデルファイルがディスク上に見つかりません。再ダウンロードするか別のモデルを選んでください。",
      MODEL_INCOMPATIBLE: "選択したファイルはこの用途のモデルとして不正です。別のモデルを選んでください。",
      PIPELINE_LOAD_FAILED: "モデルパイプラインの読み込みに失敗しました。サーバのVRAM/ログを確認して再試行してください。",
    },
  },
};

export type Lang = "en" | "ja";

export const DICTIONARIES: Record<Lang, Strings> = { en, ja };

export const DEFAULT_LANG: Lang = "en";

/** `localStorage` key the active language is persisted under (task brief: "選択はlocalStorageに保存"). */
export const LANG_STORAGE_KEY = "nzvideomni.lang";
