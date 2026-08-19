import { useStrings } from "../../i18n/LanguageContext";
import { useJobsContext } from "../../jobs/JobsContext";
import { latestJobSeed } from "../../jobs/seedUtils";
import { VIDEO_PLACEHOLDER_DATA_URL } from "../../shell/thumbnailPlaceholders";
import type { Stage2Window } from "../../shell/tokenBudget";
import { SizeFields } from "../single/CommonGenerationFields";
import { RangeBand } from "./RangeBand";
import type { UseRetakeFormResult } from "./useRetakeForm";

export interface RetakePanelProps {
  /** Retake フォーム一式。所有者は `EditScreen`（Generate ボタンが生成列に
   * あるため）——`SingleScreen`/`GenerationForm` と同じ分担。 */
  form: UseRetakeFormResult;
  /** 送信中は全部の操作を凍らせる。 */
  disabled: boolean;
}

/**
 * Retake（リテイク：撮り直し）のパネル。正本は実装計画 §4-F5、台帳は
 * `Docs/PENDING_TASKS.md` §1-17。
 *
 * **表示専用**（状態は `useRetakeForm` が持ち、`EditScreen` が所有する）。
 * 上から順に: 見出し → 素材カード → 区間バー → 撮り直す対象（音声トグル）→
 * 幅・高さ → フレームレート → シード → Stage-2 のクリップ長 → 生成前の注意 →
 * タイムラインとのずれ注意。
 *
 * ## 3 つの状態
 *
 * `snapshot ? 通常 : awaitingSource ? 素材待ち : 案内` の順で分岐する。
 * 「素材待ち」は 🔁 で素材と区間だけを捨てたあとで、設定欄は出したまま次の
 * 右クリックを待つ状態（生成ボタン群は `EditScreen` 側の `snapshot !== null`
 * ゲートで自然に消える）。
 *
 * ## 無いもの
 *
 * - **DURATION（尺）欄が無い**。撮り直す長さは区間バーが決めるので、別に数字を
 *   置くと 2 つの正が生まれる。
 * - **「AviUtl2から取得」ボタンが無い**。幅・高さは Create/Chain と同じ
 *   `SizeFields` を借りているが、取得ボタンの 3 props は渡していない
 *   （`onGetSize` を省くと描かれない）。取ってくるのはプロジェクトの解像度で、
 *   撮り直す素材の寸法とは無関係だから。
 * - **素材を差し替えるボタンが無い**。右クリックしたオブジェクトに対して測った
 *   区間なので、素材だけ入れ替えると意味が変わる。素材カードにあるのは ❌（この
 *   撮り直しを片付ける）と 🔁（素材と区間だけ捨てて、設定はそのまま次の右クリック
 *   を待つ）の 2 つだけで、どちらも**捨てる**方向にしか動かない。
 * - **プロンプト欄が無い**。Create/Chain/Outpainting と同じ共有バーを読む。
 * - **のりしろのつまみが無い**。実測で較正された既定値をサーバ側に追随させる方針
 *   （実装計画 §2）で、ここは 1 行の説明と区間バーの縞だけ。
 */
export function RetakePanel({ form, disabled }: RetakePanelProps) {
  const strings = useStrings();
  const t = strings.edit.retake;

  /** ❌（片付け）。3 状態のうち 2 つに出るので、ここで 1 回だけ作る。
   * `OutpaintingPanel` の同種ボタンと同じ規約（`.icon-action-button` ＋ title ＋
   * aria-label ＋ 絵文字は `aria-hidden`）。 */
  const clearButton = (
    <button
      type="button"
      className="icon-action-button"
      title={t.clearButton}
      aria-label={t.clearButton}
      disabled={disabled}
      onClick={form.clearAll}
    >
      <span aria-hidden="true">❌</span>
    </button>
  );

  if (!form.snapshot) {
    // 素材待ち（🔁 の直後）: 空の素材カード ＋ 案内文 ＋ 設定欄。🔁 は出さない
    // —— 捨てる素材がもう無いので、押しても何も起きないボタンになる。
    if (form.awaitingSource) {
      return (
        <section className="edit-panel retake-panel">
          <h2>{t.heading}</h2>
          <p className="hint">{t.summary}</p>
          <div className="field source-section">
            <span className="field-label">{t.sourceHeading}</span>
            <div className="source-input-actions">{clearButton}</div>
            <div className="source-input-body">
              <div className="keyframe-thumb">
                <span className="keyframe-thumb-filename">—</span>
              </div>
              <div className="source-input-controls">
                <p className="field-hint">{t.awaitingSource}</p>
              </div>
            </div>
          </div>
          <RetakeSettings form={form} disabled={disabled} />
        </section>
      );
    }
    return (
      <section className="edit-panel retake-panel">
        <h2>{t.heading}</h2>
        <p className="hint">{t.summary}</p>
        <p className="hint">{t.idle}</p>
      </section>
    );
  }

  const status = form.source.state.status;

  /**
   * 読み出し行の主表示。**RangeBand へは文字列ではなく関数を渡す** ——
   * あちらはドラッグ中 `onChange` を上げないので、ここで確定値
   * （`form.placement`）を読んで文字列を作ると、ドラッグ中だけ数字が固まる。
   *
   * 写像そのものは `form.toProjectRange`（`placement` と同じ 1 本）で、ここが
   * やるのは **+1（1 始まりへ）と文字列化だけ**。1 始まりは AviUtl2 本体 UI の
   * 表記に合わせる規約（`Docs/RIGHTCLICK_REDESIGN_SPEC.md` §6-3）。
   */
  const formatPlacement = (startFrame: number, frames: number): string => {
    const range = form.toProjectRange(startFrame, frames);
    if (!range) return "";
    return t.rangeBand.placementReadout(range.frameStart + 1, range.frameEnd + 1);
  };

  return (
    <section className="edit-panel retake-panel">
      <h2>{t.heading}</h2>
      <p className="hint">{t.summary}</p>

      {/* 素材カード。差し替えボタンは意図的に無く、あるのは ❌ と 🔁 だけ
          （doc 参照）。位置決めは Chain の SOURCE カードと同じ
          `.source-input-actions`（`ChainedScreen.css`）の借用で、CSS の追加は
          していない —— 同じ見た目を 2 か所で持たないため。 */}
      <div className="field source-section">
        <span className="field-label">{t.sourceHeading}</span>
        <div className="source-input-actions">
          {clearButton}
          <button
            type="button"
            className="icon-action-button"
            title={t.resetSourceButton}
            aria-label={t.resetSourceButton}
            disabled={disabled}
            onClick={form.resetSource}
          >
            <span aria-hidden="true">🔁</span>
          </button>
        </div>
        <div className="source-input-body">
          <div className="keyframe-thumb">
            {status === "uploading" ? (
              <span className="keyframe-spinner" role="status" aria-label={t.uploading} />
            ) : status === "ready" ? (
              <img src={VIDEO_PLACEHOLDER_DATA_URL} alt={t.sourceThumbAlt} />
            ) : (
              <span className="keyframe-thumb-filename">—</span>
            )}
          </div>
          <div className="source-input-controls">
            {/* 「◯秒 ◯フレーム（◯fps換算）／解像度」。秒は**実際に再生されて
                いる区間の長さ**（ファイル全長ではない）で、区間バーの帯の全幅と
                同じ量。フレーム数はそれを下の FPS 欄の値で換算したもの
                —— 素材の真の fps は AviUtl2 から取れないので「換算」と書く。 */}
            <p className="field-hint">
              {t.sourceReadout(
                form.snapshot.fileName,
                form.materialSeconds.toFixed(2),
                form.materialFrames,
                form.frameRate,
                form.snapshot.mediaWidth,
                form.snapshot.mediaHeight,
              )}
            </p>
            <p className="field-hint">{t.sourceFixedNote}</p>
            {status === "error" && <p className="field-hint field-hint-error">{t.loadFailed}</p>}
            {form.source.state.trimFailed && (
              <p className="field-hint field-hint-error">{t.trimFailed}</p>
            )}
          </div>
        </div>
      </div>

      {/* 区間バー。8n+1 の格子も上下限のクランプも `useRetakeForm` 側
          （`resolveRetakeWindow`）が持っていて、ここは controlled な子。 */}
      <div className="field">
        <RangeBand
          materialFrames={form.materialFrames}
          startFrame={form.windowStartFrame}
          frames={form.windowFrames}
          genFps={form.frameRate}
          glueHeadFrames={form.glueHeadFrames}
          glueTailFrames={form.glueTailFrames}
          minFrames={form.minWindowFrames}
          maxFrames={form.maxWindowFrames}
          formatPlacement={formatPlacement}
          disabled={disabled}
          onChange={form.setWindow}
        />
        <p className="field-hint">{t.glueHint(form.glueHeadFrames, form.glueTailFrames)}</p>
      </div>

      <RetakeSettings form={form} disabled={disabled} />

      {/* 生成前の注意。3 本とも常に出す —— どれも「押したあとで気づく」類の話で、
          条件付きで隠すと、隠れた回だけ知らないまま進むことになる。設定欄の
          **後ろ**に置くのは、読むのは 1 回・触るのは何度もという頻度の差から
          （2026-08-10 オーナー指示の並び）。 */}
      <div className="field">
        <span className="field-label">{t.noticesHeading}</span>
        <ul className="retake-notices">
          <li className="field-hint">{t.noticeDifferent}</li>
          <li className="field-hint">{t.noticeGlueQuality}</li>
          <li className="field-hint">{t.noticeSilentAudio}</li>
        </ul>
      </div>

      {/* タイムラインとのずれ。黄色の注意だけで、生成は止めない。 */}
      {form.stale && <p className="warning-banner warning-banner-mild">{t.staleWarning}</p>}
    </section>
  );
}

/**
 * 設定欄（対象 → 幅・高さ → フレームレート → シード → Stage-2 のクリップ長）。
 *
 * 通常状態と「素材待ち」の**両方**から呼ぶためにローカル部品へ切り出してある
 * —— 🔁 の主眼は「設定は残したまま次の素材を待つ」ことなので、2 つの状態で
 * 同じ並びの同じ欄が出ていなければならない（同じ JSX を 2 か所に書くと、
 * 片方だけ直る事故になる）。ここは `form.snapshot` に一切触らない
 * （幅・高さの実効値も、シードも、上下限も、すべてフック側で解決済み）。
 */
function RetakeSettings({ form, disabled }: RetakePanelProps) {
  const strings = useStrings();
  const t = strings.edit.retake;
  return (
    <>
      {/* 撮り直す対象。既定は「映像と音声」。 */}
      <fieldset className="outpaint-align-group">
        <legend className="field-label">{t.audioHeading}</legend>
        {(
          [
            [true, t.audioBoth],
            [false, t.audioVideoOnly],
          ] as Array<[boolean, string]>
        ).map(([value, label]) => (
          <label key={String(value)} className="field field-inline">
            <input
              type="radio"
              name="retake-audio"
              checked={form.regenerateAudio === value}
              disabled={disabled}
              onChange={() => form.setRegenerateAudio(value)}
            />
            <span className="field-label">{label}</span>
          </label>
        ))}
      </fieldset>

      {/* 幅・高さ。Create/Chain と同じ `SizeFields` をそのまま借りる（取得ボタンの
          3 props は渡さないので、あのボタンは描かれない）。初期値は素材の実寸から
          導いた値で、ユーザーが動かすまではそれに追随する。 */}
      <SizeFields
        width={form.width}
        height={form.height}
        minWidth={form.limits.minWidth}
        maxWidth={form.limits.maxWidth}
        minHeight={form.limits.minHeight}
        maxHeight={form.limits.maxHeight}
        disabled={disabled}
        onWidthChange={form.setWidth}
        onHeightChange={form.setHeight}
      />

      <div className="field-row">
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

      {/* Stage-2 のクリップ長。**Chained と同じ選択肢・同じ解説**（オーナー指示）
          なので、i18n も `strings.chained.stage2Window` を直接読む —— 同じ文言を
          2 箇所で持つと、片方だけ直る事故になる。JSX は `ChainedScreen` の同じ
          ブロックの写しで、あちらは本作業の立入禁止領域なので触っていない。
          Retake ではこの選択が**窓長の上限**にも効く（潜在19フレーム = 145）。 */}
      <label className="field">
        <span className="field-label">{strings.chained.stage2Window.label}</span>
        <select
          value={form.stage2Window}
          disabled={disabled}
          onChange={(e) => form.setStage2Window(e.target.value as Stage2Window)}
        >
          <option value="standard">{strings.chained.stage2Window.standardOption}</option>
          <option value="high_resolution">{strings.chained.stage2Window.highResolutionOption}</option>
        </select>
        {strings.chained.stage2Window.hint.split("\n\n").map((paragraph, i) => (
          <p className="field-hint" key={i}>
            {paragraph}
          </p>
        ))}
      </label>
    </>
  );
}

/** シード欄＋🎲/♻。`OutpaintingPanel` の同名ローカル部品と同じ作り（あちらは
 * フレームレートを尺の隣に置いている都合で共有の `FrameRateSeedFields` を
 * 使っておらず、このパネルも同じ理由でこちら側に持っている）。 */
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
