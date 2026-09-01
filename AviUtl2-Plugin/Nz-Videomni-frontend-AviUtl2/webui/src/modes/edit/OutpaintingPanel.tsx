import type { NativeBridge } from "../../bridge";
import { useStrings } from "../../i18n/LanguageContext";
import { useJobsContext } from "../../jobs/JobsContext";
import { latestJobSeed } from "../../jobs/seedUtils";
import { VIDEO_PLACEHOLDER_DATA_URL } from "../../shell/thumbnailPlaceholders";
import { useFileDrop } from "../../shell/useFileDrop";
import { FALLBACK_APP_CONFIG, MIN_NUM_FRAMES } from "../single/defaultConfig";
import { formatDurationHint, isStepEvent } from "../single/paramUtils";
import { resolveSpillFreeFrames } from "../single/spillUtils";
import {
  BLEND_DILATION_MAX,
  BLEND_DILATION_MIN,
  featherWidthPx,
  MAX_PAD,
  PAD_SLIDER_MAX,
} from "./outpaintGeometry";
import { OutpaintPreview } from "./OutpaintPreview";
import { OUTPAINT_VIDEO_EXTENSIONS } from "./useOutpaintForm";
import type { PadSide, UseOutpaintFormResult } from "./useOutpaintForm";

/** The duration slider's spill tick, as a `<datalist>` id. Distinct from
 * Create's own `"spill-tick"` (`modes/single/CommonGenerationFields.tsx`) on
 * purpose: `AppShell` keeps every mode screen mounted at once, so both
 * datalists are in the document simultaneously and a shared id would be a
 * duplicate. */
const OUTPAINT_SPILL_TICK_ID = "outpaint-spill-tick";

export interface OutpaintingPanelProps {
  /** The whole Outpainting form, owned by `EditScreen` (which also renders the
   * Generate button this form gates, over in the generation column). Same
   * arrangement as Create's `GenerationForm`, which takes `useGenerationForm`'s
   * result from `SingleScreen`. */
  form: UseOutpaintFormResult;
  /** `true` while a submit is in flight — freezes every control, exactly as
   * `GenerationForm`'s own `disabled` does. */
  disabled: boolean;
  /** Test/integration seam threaded down from `EditScreen`. Production omits
   * it, and the drop target then binds to the app-wide singleton bridge. */
  nativeBridge?: NativeBridge | undefined;
}

/**
 * Outpainting（アウトペインティング：画面の描き足し）のパネル。正本は
 * `Docs/OUTPAINTING_DESIGN_NOTES.md`、台帳は `Docs/PENDING_TASKS_CLOSED.md`
 * §3-70（起票当時は§1-13）。
 *
 * The user's path through it, top to bottom: attach the video → look at the
 * picture → dial in how much to add on each side (0 起点・1px 刻み、スライダー
 * でも数値入力でも) → set the mask blur → set length/frame rate/seed → press
 * Generate (which lives in `EditScreen`'s generation column, to the right — the
 * Create/Chain placement rule). The four pad sliders and the mask blur only
 * appear once the source has actually been measured; before that there is
 * nothing honest to draw, and the reason note beside the Generate button says
 * so. The preview itself is always mounted (it draws an empty frame until a
 * source lands), so the panel does not jump when one arrives.
 *
 * 2026-08-11: 初期配置（アライメント）のラジオ 2 系統は撤去した。パッドは自動で
 * 足されず、128 の格子から外れたキャンバスは右カラムのブロック理由（幅・高さ別々
 * の 2 行）が止める。スライダー直下の `gridNote` は常設の説明として残す。
 *
 * Three things this panel deliberately does NOT have:
 *  - a prompt field. Outpainting reads the SHARED `PromptBar` above the tabs,
 *    exactly like Create/Chain — one prompt in the app, no second box to keep
 *    in sync (2026-08-09; it briefly had a local one).
 *  - a LoRA picker. The server requires exactly one control adapter alongside
 *    an `outpaint` request and `in-outpainting` is the only one that does this
 *    job, so it is pinned (`lora/controlLoras.OUTPAINT_LORA_NAME`) and its
 *    absence from `GET /loras` becomes a block reason instead of an empty
 *    dropdown. The same name is also excluded from Create/Chain's OWN control
 *    LoRA dropdowns (`UI_HIDDEN_CONTROL_LORA_NAMES`) — this panel is its one
 *    and only place in the UI.
 *  - form state of its own. `EditScreen` owns the hook and hands the result in,
 *    because the Generate button it gates sits in that screen's generation
 *    column — the same split as `SingleScreen` / `GenerationForm`.
 */
export function OutpaintingPanel({ form, disabled, nativeBridge }: OutpaintingPanelProps) {
  const strings = useStrings();
  const t = strings.edit.outpainting;

  const known = form.mediaInfo.width > 0 && form.mediaInfo.height > 0;

  // なじみ幅の実寸。膨張段数はキャンバスに対する相対量なので、ピクセルは
  // 「今のキャンバス」から毎回導く（保存しない）。プレビューの破線・凡例・
  // 説明文はすべてこの 1 つの値を見る。
  const canvasLongSide = Math.max(form.canvas.width, form.canvas.height);
  const featherPx = featherWidthPx(form.blendDilation, canvasLongSide);
  // 上げすぎ警告（ブロックはしない）: 向かい合う 2 本の帯が出会うと、元動画に
  // 混ぜ合わせ外の芯が残らなくなる。
  const featherCoversSource = known && featherPx > 0 && 2 * featherPx >= Math.min(form.mediaInfo.width, form.mediaInfo.height);

  // 快適上限（spill）の目盛りと警告 —— Create の DURATION と同じ仕組み
  // (`modes/single/CommonGenerationFields.tsx` の `spillThresholdFrames` /
  // `isOverSpillThreshold`)。違いは解像度の出どころだけで、Create が生成
  // width/height を渡すところに、こちらは **拡張後キャンバス** を渡す —— 実際に
  // 生成されるのはその寸法だからで、元動画の寸法で引くと必ず甘い値になる。
  // 表は `FALLBACK_APP_CONFIG` から取る: このパネルは尺の上限もフレームレート
  // の初期値も同じ組み込み設定から導いており（`useOutpaintForm`）、ここだけ
  // `GET /config` を追加で叩くと出どころが2つに割れるため。
  // Gated on `known`: with no source measured the canvas is 0x0, and the
  // nearest-area fallback would answer with the SMALLEST entry in the table —
  // a tick and a warning about a resolution nobody has chosen yet.
  const spillThresholdFrames = known
    ? resolveSpillFreeFrames(FALLBACK_APP_CONFIG.limits.spill_free_frames, form.canvas.width, form.canvas.height)
    : null;
  const isOverSpillThreshold = spillThresholdFrames !== null && form.numFrames > spillThresholdFrames;

  return (
    <section className="edit-panel outpaint-panel">
      <h2>{t.heading}</h2>

      <SourceCard form={form} disabled={disabled} nativeBridge={nativeBridge} />

      <div className="field">
        <OutpaintPreview
          sourceWidth={form.mediaInfo.width}
          sourceHeight={form.mediaInfo.height}
          pads={form.pads}
          blendBandPx={featherPx}
        />
        {known && <p className="field-hint">{t.canvasReadout(form.canvas.width, form.canvas.height)}</p>}
      </div>

      {known && (
        <>
          <div className="field">
            <span className="field-label">{t.padsHeading}</span>
            {/* センタリング（§5-D、2026-08-12）。下のスライダー群の「書き込まれ方」
                を変えるスイッチなので、見出しの直下・スライダーの上に置く。説明文は
                付けない（§5-C-6 の「画面の文字を増やさない」を維持）。`<label>` の
                暗黙ラベルで中のチェックボックスに名前が付くため `aria-label` は不要。 */}
            <label className="field field-inline">
              <input
                type="checkbox"
                checked={form.centering}
                disabled={disabled}
                onChange={(e) => form.setCentering(e.target.checked)}
              />
              <span className="field-label">{t.centeringLabel}</span>
            </label>
            <div className="outpaint-pads">
              <PadSlider form={form} side="top" label={t.padTop} disabled={disabled} />
              <PadSlider form={form} side="bottom" label={t.padBottom} disabled={disabled} />
              <PadSlider form={form} side="left" label={t.padLeft} disabled={disabled} />
              <PadSlider form={form} side="right" label={t.padRight} disabled={disabled} />
            </div>
            {/* 必須説明文。アライメント撤去後も残す（オーナー決定 2026-08-11）
                —— ブロック理由は右カラムに出るので、スライダーの手元にも常設の
                説明が要る。 */}
            <p className="field-hint">{t.gridNote}</p>
          </div>

          {/* マスクブラー。内部は膨張段数 (0-15、サーバーの範囲そのまま)、
              目に見える数字はすべて実寸ピクセル。 */}
          <label className="field">
            <span className="field-label">{t.blurLabel}</span>
            <input
              type="range"
              aria-label={t.blurLabel}
              min={BLEND_DILATION_MIN}
              max={BLEND_DILATION_MAX}
              step={1}
              value={form.blendDilation}
              disabled={disabled}
              onChange={(e) => form.setBlendDilation(Number(e.target.value))}
            />
            <span className="field-hint">{form.blendDilation === 0 ? t.blurZero : featherPx}</span>
            {/* ブレンド帯の必須説明文（このキャンバスでの実寸）。「緑が残ったら
                この値を上げる」という助言は 2026-08-09 にこの一文へ畳み込んだ
                （旧 `blurGreenNote` は廃止）—— 同じスライダーの話が2段落に
                分かれて見えていたため。 */}
            <p className="field-hint">{t.blendNote(featherPx)}</p>
            {featherCoversSource && <p className="warning-banner warning-banner-mild">{t.blurWarning}</p>}
          </label>
        </>
      )}

      {/* 尺・フレーム数・フレームレート・シード。共有の `DurationField` /
          `FrameRateSeedFields` は使わない —— このパネルだけ「スライダー」と
          「数値ボックス＋フレームレート横並び」を別の行に分けるため。挙動
          （8n+1 きざみ、自由入力はそのまま通す、上限は元動画長から毎描画
          導出）は共有側と同じにしてある。 */}
      <label className="field">
        <span className="field-label">{strings.single.duration.label}</span>
        <input
          type="range"
          min={MIN_NUM_FRAMES}
          max={form.numFramesCeiling}
          step={8}
          value={form.numFrames}
          disabled={disabled}
          list={spillThresholdFrames != null ? OUTPAINT_SPILL_TICK_ID : undefined}
          onChange={(e) => form.setNumFrames(Number(e.target.value), true)}
        />
        {/* 快適上限の目盛り。Create と同じく `<datalist>` 1 点で描く。 */}
        {spillThresholdFrames != null && (
          <datalist id={OUTPAINT_SPILL_TICK_ID}>
            <option value={spillThresholdFrames} />
          </datalist>
        )}
        <span className="field-hint">{formatDurationHint(form.numFrames, form.frameRate)}</span>
        {/* Edit専用文言（修正1, 2026-08-09）。Single側の spillWarning とは別キー
            —— Single の文言は変えない方針のため、Edit だけ独立させた。警告だけで、
            生成は止めない。 */}
        {isOverSpillThreshold && (
          <p className="warning-banner warning-banner-mild">{t.spillWarning}</p>
        )}
      </label>

      <div className="field-row">
        <label className="field field-inline">
          <span className="field-label">{t.numFramesLabel}</span>
          <input
            type="number"
            min={MIN_NUM_FRAMES}
            max={form.numFramesCeiling}
            step={8}
            value={form.numFrames}
            disabled={disabled}
            onChange={(e) => form.setNumFrames(Number(e.target.value), isStepEvent(e.nativeEvent))}
          />
        </label>
        <label className="field field-inline">
          <span className="field-label">{strings.single.duration.fps}</span>
          <input
            type="number"
            min={1}
            max={60}
            value={form.frameRate}
            disabled={disabled}
            onChange={(e) => form.setFrameRate(Number(e.target.value))}
          />
        </label>
      </div>

      <SeedField seed={form.seed} disabled={disabled} onChange={form.setSeed} />

      {/* 快適上限 (§4-5): 警告のみ。生成は止めない。尺スライダーの spill 目盛り
          とは別物 —— あちらは「この解像度での尺の目安」、こちらは解像度と尺を
          掛け合わせた総処理量の目安で、片方だけが出る組み合わせもある。 */}
      {form.isOverComfortBudget && (
        <p className="warning-banner warning-banner-mild">{t.comfortWarning(form.comfortTokens)}</p>
      )}
    </section>
  );
}

/** One pad: a slider and a number box showing the SAME value, both starting at
 * 0 and moving one pixel at a time (オーナー決定 2026-08-11). The pair is laid
 * out exactly like Create's width/height fields (`CommonGenerationFields.tsx`'s
 * `SizeFields`), which is the app's existing shape for one value with two ways
 * to set it.
 *
 * ## 二つの入力部品の役割分担（第2弾・2026-08-11）
 *
 * The two carry DIFFERENT ceilings on purpose.
 *
 *  - **スライダーは入力補助**。`PAD_SLIDER_MAX` (220) までしか受け持たない。第1弾
 *    では「4096 − 元動画の辺 − 対辺のパッド」を上限にしていたが、可動域が約 2800
 *    に達するとトラック上のマウス 1px が値 13 相当になり、マウスでは決して指定
 *    できない値が生まれていた。
 *  - **数値ボックスが正確な入力口**。上限は `MAX_PAD` (4096) 固定で、元動画の
 *    寸法にも対辺の値にも依存しない。
 *
 * したがって 220 を超える値はボックスからしか入らず、そのときスライダーのつまみは
 * **右端に貼りつく**（`value > max` の range 入力を表示上 max へ丸める、HTML 本来の
 * 挙動をそのまま使っている。state の値は 220 に変わらず、ボックスは実値を出し
 * 続ける）。
 *
 * **貼りつき中にスライダーを操作すると値は 220 以下へ落ちる**（ドラッグ、トラック
 * 上のクリック、矢印キー左、フォーカス中のホイール）。厳密には「つまみが今いる
 * 右端ちょうど」を狙ったクリックのように DOM 値が 220 のまま変わらない操作では
 * `change` が出ないので値も動かないが、それ以外はどこを触っても落ちると考えて
 * よい。これは仕様である（オーナー決定 2026-08-11）—— 「スライダーを操作した＝
 * スライダーの範囲で指定し直す意思」とみなし、`max(220, 現在値)` のような動的
 * 上限は入れない。
 *
 * Nothing here snaps: whatever arrives goes through `useOutpaintForm.setPad` ->
 * `clampPad`, which only brings the value into `[0, MAX_PAD]`. A canvas that
 * misses the 128 grid or passes 4096 is stopped by the block reasons instead. */
function PadSlider({
  form,
  side,
  label,
  disabled,
}: {
  form: UseOutpaintFormResult;
  side: PadSide;
  label: string;
  disabled: boolean;
}) {
  const value = form.pads[side];
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {/* Explicit `aria-label`s rather than relying on the wrapping label: an
          implicit label binds to the FIRST control inside it, so the number box
          would otherwise have no accessible name at all. Both carry the same
          name on purpose — they are two views of one value. */}
      <input
        type="range"
        aria-label={label}
        min={0}
        max={PAD_SLIDER_MAX}
        step={1}
        value={value}
        disabled={disabled}
        onChange={(e) => form.setPad(side, Number(e.target.value))}
      />
      <input
        type="number"
        aria-label={label}
        min={0}
        max={MAX_PAD}
        step={1}
        value={value}
        disabled={disabled}
        onChange={(e) => form.setPad(side, Number(e.target.value))}
      />
    </label>
  );
}

/** The seed box + its 🎲/♻ pair. A local copy of `FrameRateSeedFields`' seed
 * half (that component pairs seed WITH the frame rate, and this panel puts the
 * frame rate next to the frame count instead) — ♻ reads the very same job
 * ledger this screen now shows, so it offers the most recent job's seed. */
function SeedField({
  seed,
  disabled,
  onChange,
}: {
  seed: number;
  disabled: boolean;
  onChange: (value: number) => void;
}) {
  const strings = useStrings();
  const { jobs } = useJobsContext();
  const lastSeed = latestJobSeed(jobs);
  return (
    <label className="field field-inline">
      <span className="field-label">{strings.single.seed.label}</span>
      <input
        type="number"
        value={seed}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <span className="seed-actions">
        <button
          type="button"
          className="icon-action-button"
          title={strings.single.seed.randomTooltip}
          aria-label={strings.single.seed.randomTooltip}
          disabled={disabled}
          onClick={() => onChange(-1)}
        >
          <span aria-hidden="true">🎲</span>
        </button>
        <button
          type="button"
          className="icon-action-button"
          title={strings.single.seed.reuseTooltip}
          aria-label={strings.single.seed.reuseTooltip}
          disabled={disabled || lastSeed === null}
          onClick={() => {
            if (lastSeed !== null) onChange(lastSeed);
          }}
        >
          <span aria-hidden="true">♻</span>
        </button>
      </span>
      <span className="field-hint">{strings.single.seed.hint}</span>
    </label>
  );
}

/** The 素材 (material) card: the whole card is the drop target (the same recipe
 * as Create's `ReferenceVideoSection` and Chain's `SourceInputPanel`), with the
 * choose/clear pair collapsed to two emoji icon buttons whose text lives on as
 * `aria-label`/`title`, and a 64x64 thumbnail slot that shows the shared play
 * glyph once a video is attached.
 *
 * Written out here rather than reusing `modes/chained/SourceInputPanel`: that
 * component is bound to `UseChainFormResult` and carries Chain's strength /
 * context-frames controls, none of which exist here. What IS borrowed is the
 * look — see `EditScreen.css` for which rules came from where. */
function SourceCard({
  form,
  disabled,
  nativeBridge,
}: {
  form: UseOutpaintFormResult;
  disabled: boolean;
  nativeBridge?: NativeBridge | undefined;
}) {
  const strings = useStrings();
  const t = strings.edit.outpainting;
  const { state, pick, clear } = form.source;
  const uploading = state.status === "uploading";
  const ready = state.status === "ready";
  const chooseLabel = uploading ? t.uploadingButton : ready ? t.changeButton : t.chooseButton;

  const {
    isDragOver,
    error: dropError,
    handlers,
  } = useFileDrop({
    accept: OUTPAINT_VIDEO_EXTENSIONS,
    onFile: form.attachByPath,
    disabled: disabled || uploading,
    nativeBridge,
  });

  const className = ["field", "source-section", isDragOver ? "source-section-dragover" : ""].filter(Boolean).join(" ");

  return (
    <div
      className={className}
      onDragEnter={handlers.onDragEnter}
      onDragOver={handlers.onDragOver}
      onDragLeave={handlers.onDragLeave}
      onDrop={handlers.onDrop}
    >
      <span className="field-label">{t.sourceHeading}</span>
      <div className="source-input-actions">
        <button
          type="button"
          className="icon-action-button"
          title={t.clearButton}
          aria-label={t.clearButton}
          disabled={disabled || uploading || !ready}
          onClick={clear}
        >
          <span aria-hidden="true">❌</span>
        </button>
        <button
          type="button"
          className="icon-action-button"
          title={chooseLabel}
          aria-label={chooseLabel}
          disabled={disabled || uploading}
          onClick={() => void pick()}
        >
          <span aria-hidden="true">📁</span>
        </button>
      </div>

      <div className="source-input-body">
        <div className="keyframe-thumb">
          {uploading ? (
            <span className="keyframe-spinner" role="status" aria-label={t.uploadingButton} />
          ) : ready ? (
            <img src={VIDEO_PLACEHOLDER_DATA_URL} alt={t.sourceThumbAlt} />
          ) : (
            <span className="keyframe-thumb-filename">—</span>
          )}
        </div>

        <div className="source-input-controls">
          <p className="field-hint">{t.sourceHint}</p>
          {state.fileName && <p className="field-hint">{state.fileName}</p>}
          {ready && form.mediaInfo.width > 0 && (
            <p className="field-hint">
              {t.sourceReadout(form.mediaInfo.width, form.mediaInfo.height, form.mediaInfo.durationSec.toFixed(1))}
            </p>
          )}
          {dropError && (
            <p className="field-hint field-hint-error">
              {dropError === "unsupported" ? strings.dnd.unsupportedVideo : strings.dnd.resolveFailed}
            </p>
          )}
          {state.status === "idle" && <p className="field-hint">{t.none}</p>}
          {state.status === "error" && <p className="field-hint field-hint-error">{state.errorCode}</p>}
        </div>
      </div>
    </div>
  );
}
