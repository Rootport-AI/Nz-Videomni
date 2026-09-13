import { useStrings } from "../../i18n/LanguageContext";
import { useJobsContext } from "../../jobs/JobsContext";
import { latestJobSeed } from "../../jobs/seedUtils";
import { VIDEO_PLACEHOLDER_DATA_URL } from "../../shell/thumbnailPlaceholders";
import { DurationField } from "../single/CommonGenerationFields";
import { formatDurationHint } from "../single/paramUtils";
import type { UseInpaintFormResult } from "./useInpaintForm";

export interface InpaintingPanelProps {
  /** Inpainting フォーム一式。所有者は `EditScreen`（Generate ボタンが生成列に
   * あるため）—— `RetakePanel`/`OutpaintingPanel` と同じ分担。 */
  form: UseInpaintFormResult;
  /** 送信中・マスク描画中は全部の操作を凍らせる。 */
  disabled: boolean;
}

/**
 * Inpainting（マスクによる部分再生成）のパネル。台帳は
 * `Docs/PENDING_TASKS.md` §3-55、設計正本は `Docs/INPAINTING_DESIGN.md`。
 *
 * **表示専用**（状態は `useInpaintForm` が持ち、`EditScreen` が所有する）。
 * 上から順に: 見出し → 対象動画カード → 部分フィルタカード → フレーム数 →
 * 窓の読み出し → シード → のりしろ（モック）→ 進捗 → 生成前の注意
 * （カード 2 枚の並びは設計正本 §3.1「上が対象動画、下が部分フィルタ」）。
 *
 * ## 無いもの（すべて裁定の結果で、作り忘れではない）
 *
 * - **マスク動画がどこにも出ない**（D7）。作るのはプラグインで、置き場所も
 *   中身も利用者には見せない。部分フィルタカードに出るのはレイヤーと
 *   フレーム範囲と中間点の枚数だけ。
 * - **幅・高さのつまみが無い**（D5）。出力解像度は対象動画の実寸で決まる。
 *   一致しない素材は生成させる前に止める（伸縮しない）。
 * - **マスクを作るボタンが無い**（D8）。Generate 押下時に描画→アップロード→
 *   送信を一続きで行う。
 * - **マスクの膨張つまみが無い**。注意文（「部分フィルタは対象より広めに」）と
 *   画角拡張と同じ既定値で足りる、という監督の既定。
 * - **プロンプト欄が無い**。他タブと同じ共有バーを読む。空欄のままでも生成できる。
 */
export function InpaintingPanel({ form, disabled }: InpaintingPanelProps) {
  const strings = useStrings();
  const t = strings.edit.inpainting;
  const slots = form.slots;
  const busy = disabled || form.maskPhase !== "idle";

  /** 1 始まりへ直した窓の読み出し（AviUtl2 本体 UI の表記に合わせる規約
   * ——`Docs/RIGHTCLICK_REDESIGN_SPEC.md` §6-3）。 */
  const windowLine = form.window
    ? t.windowReadout(form.window.windowStart + 1, form.window.windowEnd + 1)
    : null;

  /** native の 3 コードを 1 行の文言へ。`useObjectTracking` と同じ分担で、
   * フックはコードを運ぶだけ・写像はここ。知らないコードは総称の 1 行へ落ちる
   * （黙って消さない）。 */
  const maskNote =
    form.maskErrorCode === ""
      ? null
      : form.maskErrorCode === "MASK_SEED_INVALID"
        ? t.seedGone
        : form.maskErrorCode === "MASK_BUSY"
          ? t.maskBusy
          : t.maskFailed;

  return (
    <section className="edit-panel inpaint-panel">
      <h2>{t.heading}</h2>
      <p className="hint">{t.summary}</p>
      {!slots.partialFilter && !slots.target && <p className="hint">{t.idle}</p>}

      {/* 対象動画カード。骨格は `RetakePanel` の素材カードの写しで、CSS の追加は
          していない（同じ見た目を 2 か所で持たないため）。差し替えボタンは無く、
          あるのは ❌（この Inpainting を片付ける）だけ —— 窓は「そのオブジェクトの
          その場所」に対して測った量なので、素材だけ入れ替えると意味が変わる。 */}
      <div className="field source-section">
        <span className="field-label">{t.targetHeading}</span>
        <div className="source-input-actions">
          <button
            type="button"
            className="icon-action-button"
            title={t.clearButton}
            aria-label={t.clearButton}
            disabled={busy}
            onClick={form.clearAll}
          >
            <span aria-hidden="true">❌</span>
          </button>
        </div>
        <div className="source-input-body">
          <div className="keyframe-thumb">
            {form.uploadStatus === "uploading" ? (
              <span className="keyframe-spinner" role="status" aria-label={t.uploading} />
            ) : form.uploadStatus === "ready" ? (
              <img src={VIDEO_PLACEHOLDER_DATA_URL} alt={t.targetThumbAlt} />
            ) : (
              <span className="keyframe-thumb-filename">—</span>
            )}
          </div>
          <div className="source-input-controls">
            {slots.target ? (
              <>
                {form.targetFileName && <p className="field-hint">{form.targetFileName}</p>}
                {slots.target.item.mediaWidth > 0 && (
                  <p className="field-hint">
                    {t.targetReadout(
                      slots.target.item.mediaWidth,
                      slots.target.item.mediaHeight,
                      slots.target.item.mediaDurationSec.toFixed(1),
                    )}
                  </p>
                )}
              </>
            ) : (
              <p className="field-hint">{t.targetNone}</p>
            )}
            {form.uploadStatus === "error" && <p className="field-hint field-hint-error">{t.loadFailed}</p>}
            {form.trimFailed && <p className="field-hint field-hint-error">{t.trimFailed}</p>}
          </div>
        </div>
      </div>

      {/* 部分フィルタカード（設計正本 §3.1 の並びで**下**）。サムネイルも ❌ も
          無い —— 素材ではないので捨てる単位は「この Inpainting 全体」であり、
          その ❌ は上の対象動画カードに 1 つだけ置く。出すのはレイヤー・フレーム
          範囲・中間点の枚数の 3 つだけで、マスクの絵は出さない（D7）。 */}
      <div className="field">
        <span className="field-label">{t.partialFilterHeading}</span>
        {slots.partialFilter ? (
          <p className="field-hint">
            {t.partialFilterReadout(
              slots.partialFilter.layer + 1,
              slots.partialFilter.frameStart + 1,
              slots.partialFilter.frameEnd + 1,
              slots.partialFilter.midpoints,
            )}
          </p>
        ) : (
          <p className="field-hint">{t.partialFilterNone}</p>
        )}
      </div>

      {/* フレーム数。Single と同じ `DurationField`（スライダー＋数値欄）で、
          9〜481・8n+1 刻み。快適上限マーカーは**助言だけ**で、生成は止めない
          （D12）。値の確定時に格子へ乗せるのは `commitNumFrames` の仕事。 */}
      <DurationField
        label={t.framesLabel}
        value={form.numFrames}
        min={form.minFrames}
        max={form.maxFrames}
        disabled={busy}
        onChange={form.setNumFrames}
        onCommit={(value) => form.commitNumFrames(value)}
        hint={formatDurationHint(form.numFrames, form.frameRate)}
        {...(form.comfortFrames !== null ? { spillThresholdFrames: form.comfortFrames } : {})}
      />
      {form.isOverComfortBudget && (
        <p className="warning-banner warning-banner-mild">{t.comfortWarning(form.comfortTokens)}</p>
      )}

      {/* 窓の読み出し。ずらしたときと、部分フィルタを覆いきれていないときの
          注意文もここに並べる（どちらも生成は止めない）。 */}
      <div className="field">
        {windowLine && <p className="field-hint">{windowLine}</p>}
        {form.window?.shiftedHead && <p className="field-hint">{t.windowShifted}</p>}
        {form.window && !form.window.coversFilter && (
          <p className="warning-banner warning-banner-mild">{t.windowNotCoveredNote}</p>
        )}
      </div>

      <SeedField seed={form.seed} disabled={busy} onChange={form.setSeed} />

      {/* のりしろ（モック・D4）。`<fieldset disabled>` なので中の入力は**全部**
          押せない —— 個々の `disabled` を並べるより、「この塊はまだ動かない」が
          1 箇所で言えて、消し忘れも起きない。既定は「なし」。 */}
      <fieldset className="outpaint-align-group" disabled>
        <legend className="field-label">{t.glueHeading}</legend>
        <label className="field field-inline">
          <input type="radio" name="inpaint-glue" value="yes" defaultChecked={false} />
          <span className="field-label">{t.glueYes}</span>
        </label>
        <label className="field field-inline">
          <input type="radio" name="inpaint-glue" value="no" defaultChecked />
          <span className="field-label">{t.glueNo}</span>
        </label>
        <label className="field field-inline">
          <span className="field-label">{t.glueFramesLabel}</span>
          <input type="number" defaultValue={0} />
        </label>
        <p className="field-hint">{t.glueNote}</p>
      </fieldset>

      {/* 進捗 1 行。Generate 押下中にしか出ない。 */}
      {form.maskPhase === "rendering" && (
        <p className="field-hint" role="status">
          {form.maskProgress
            ? t.maskRenderingProgress(form.maskProgress.index, form.maskProgress.total)
            : t.maskRendering}
        </p>
      )}
      {form.maskPhase === "uploading" && (
        <p className="field-hint" role="status">
          {t.maskUploading}
        </p>
      )}
      {maskNote && <p className="warning-banner warning-banner-mild">{maskNote}</p>}

      {/* 生成前の注意。4 本とも常に出す —— どれも「押したあとで気づく」類の話
          （`RetakePanel` と同じ判断・同じ並びの位置）。上 2 本は段0のスパイクで
          分かった事実で、残り 2 本は元からの注意。 */}
      <div className="field">
        <span className="field-label">{t.noticesHeading}</span>
        <ul className="retake-notices">
          <li className="field-hint">{t.promptNotes.coverWholeSubject}</li>
          <li className="field-hint">{t.promptNotes.writePositively}</li>
          <li className="field-hint">{t.promptNotes.duplication}</li>
          <li className="field-hint">{t.promptNotes.blurIgnored}</li>
        </ul>
      </div>
    </section>
  );
}

/** シード欄＋🎲/♻。`RetakePanel`/`OutpaintingPanel` の同名ローカル部品と
 * 同じ作り（3 つ目の写しだが、共有部品化は本作業の範囲外——あの 2 つは
 * 触らない方針）。 */
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
      <input type="number" value={seed} disabled={disabled} onChange={(e) => onChange(Number(e.target.value))} />
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
