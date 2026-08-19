import { useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent as ReactPointerEvent } from "react";
import { useStrings } from "../../i18n/LanguageContext";
import { RETAKE_WINDOW_MAX_PX, RETAKE_WINDOW_MIN_PX } from "../../timeline/retakeWindow";
import "./RangeBand.css";

/**
 * §1-17 Retake の「撮り直す区間」バー。**v1 は 4 層**（実装計画 §4-F2）:
 *
 *  1. 素材全長帯 —— 薄地の下敷き。素材ファイル全体の長さがバーの全幅。
 *  2. 窓 —— 塗りつぶし。ドラッグの本体（つかんで左右へ動かすと区間ごと平行移動）。
 *  3. 糊代帯 —— 窓の**内側**の両端に描く半透明の縞。**非対話**（`pointer-events:
 *     none` ＋ `aria-hidden`）で、見せるだけ。前後の映像となめらかにつなぐために
 *     元の映像がそのまま残る部分で、ユーザーが個別に動かすものではない。
 *  4. ハンドル 2 つ —— 開始側・終了側。それぞれ `role="slider"`。
 *
 * 「選択範囲のスナップショット層」（右クリックした瞬間の範囲を薄く残す 5 層目）は
 * v1.1 送り。パネル側の checkStale 注意文が同じ役目を先に果たすため。
 *
 * 帯の下の読み出しは **2 段**（2026-08-10）: 主表示「タイムラインの何フレーム目を
 * 撮り直すか」と、その下に括弧書きで窓の長さ。主表示は親から受け取った
 * {@link RangeBandProps.formatPlacement} が作る —— 理由はその props の doc を参照。
 *
 * ## props は純粋（値 ＋ onChange）
 *
 * フォームにも bridge にも依存しない。値は**すべて生成 fps 上のフレーム数**で
 * 受け渡す（秒はこのバーが表示のために作るだけ）。フレームで持つのは、ドラッグ中に
 * 秒へ往復させると丸めのごみが溜まるため。
 *
 * ## 8n+1 スナップはここでは**しない**
 *
 * 窓長の 8n+1 スナップは `timeline/retakeWindow.ts` の `resolveRetakeWindow` が
 * 唯一の持ち主。ここで同じ規則を二重に実装すると、格子の定義が 2 箇所へ散る。
 * このバーは「下限・上限へのクランプ」だけを行い（ハンドルが端で止まるという
 * 触覚はこのクランプが作る）、格子への吸着は制御された値として親から戻ってくる
 * —— React の controlled input と同じ形。だから `minFrames`/`maxFrames` は
 * props で受け、既定値だけを `retakeWindow.ts` の定数から借りている
 * （将来 `AppConfig.limits` で上書きされたら親が渡すだけで済む）。
 *
 * ## ポインタ作法は `modes/single/KeyframeTimeline.tsx` 踏襲
 *
 * `setPointerCapture` でつかみ、`pointercancel` と `lostpointercapture` の
 * **両方**を購読して取りこぼしを潰す。矢印キーは 1 フレーム、Shift を足すと
 * 8 フレーム。
 */
export interface RangeBandProps {
  /** 素材の全長（生成 fps 上のフレーム数）。バーの全幅にあたる。 */
  materialFrames: number;
  /** 窓の開始（素材頭からのフレーム数）。 */
  startFrame: number;
  /** 窓の長さ（フレーム数）。 */
  frames: number;
  /** 生成 fps。秒の表示にだけ使う。 */
  genFps: number;
  /** 窓の内側・頭側の糊代（フレーム数）。0 なら描かない。 */
  glueHeadFrames?: number;
  /** 窓の内側・尾側の糊代（フレーム数）。0 なら描かない。 */
  glueTailFrames?: number;
  /** 窓長の下限。既定は `RETAKE_WINDOW_MIN_PX`。 */
  minFrames?: number;
  /** 窓長の上限。既定は `RETAKE_WINDOW_MAX_PX`。 */
  maxFrames?: number;
  /**
   * 読み出しの**主表示**を作る整形関数（省略なら主表示は出ない）。
   *
   * このバーは「素材頭からのフレーム数」しか知らない —— タイムライン上の
   * 何フレーム目かは、素材秒 ↔ プロジェクトフレームの写像を持つ親
   * （`useRetakeForm`）にしか出せない。そこで**文字列を作る関数を親から受け取り、
   * ここが毎描画呼ぶ**形にしてある。
   *
   * `useRetakeForm.placement` を親側で読んで文字列を作り、それを props で渡す形は
   * **不可**: このバーはドラッグ中 `onChange` を上げず `pointerup` で初めて確定を
   * 伝えるので、その作りだと**ドラッグ中だけ数字が固まる**。引数はどちらも
   * {@link view}（＝ドラッグ中はプレビュー値）を渡す。
   */
  formatPlacement?: ((startFrame: number, frames: number) => string) | undefined;
  disabled?: boolean;
  /** 操作の結果（すでにクランプ済み）。値が変わらないときは呼ばない。 */
  onChange: (next: { startFrame: number; frames: number }) => void;
}

/** ドラッグ中の状態。`grabOffset` は窓本体をつかんだ位置（窓頭からの
 * フレーム差）で、これが無いとつかんだ瞬間に窓が指へ飛びつく。 */
interface DragState {
  kind: "move" | "start" | "end";
  pointerId: number;
  grabOffset: number;
  preview: { startFrame: number; frames: number };
}

/** 測れないとき（初回描画・jsdom）の仮の幅。0 で割らないためだけの値。 */
const FALLBACK_TRACK_WIDTH = 320;

function clamp(value: number, min: number, max: number): number {
  if (max < min) return min;
  return Math.min(Math.max(value, min), max);
}

/** 秒の表示。fps が 0/NaN の間（設定読み込み中）でも `NaN` を出さない。 */
function formatSeconds(frames: number, genFps: number): string {
  if (!Number.isFinite(genFps) || genFps <= 0) return "0.00";
  return (frames / genFps).toFixed(2);
}

export function RangeBand({
  materialFrames,
  startFrame,
  frames,
  genFps,
  glueHeadFrames = 0,
  glueTailFrames = 0,
  minFrames = RETAKE_WINDOW_MIN_PX,
  maxFrames = RETAKE_WINDOW_MAX_PX,
  formatPlacement,
  disabled = false,
  onChange,
}: RangeBandProps) {
  const strings = useStrings();
  const t = strings.edit.retake.rangeBand;
  const trackRef = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<DragState | null>(null);

  const total = Number.isFinite(materialFrames) && materialFrames > 0 ? materialFrames : 0;
  /** 窓長として許される最大値。素材そのものより長い窓は作れない。 */
  const maxAllowed = total > 0 ? Math.min(maxFrames, total) : maxFrames;

  // 描画に使う値: ドラッグ中はプレビュー、そうでなければ props。親は
  // pointerup まで何も知らない（KeyframeTimeline と同じ「途中を上げない」方針）。
  const view = drag?.preview ?? { startFrame, frames };
  const viewEnd = view.startFrame + view.frames;

  /** フレーム数を帯の中の位置（%）へ。小数 4 桁で丸めるのは、`245/400*100` が
   * `61.25000000000001` になるような二進の端数をスタイル文字列へ持ち込まない
   * ため（1e-4 % は 4096px 幅でも 0.004px 未満で、見た目には効かない）。 */
  const pct = (value: number): number =>
    total > 0 ? Math.round((value / total) * 1e6) / 1e4 : 0;

  /** 窓の値をまとめて正当化する: 長さを `[minFrames, maxAllowed]` へ、
   * 開始を `[0, total - 長さ]` へ。ハンドルが端で「止まる」触覚はここが作る。 */
  const clampWindow = (nextStart: number, nextFrames: number): { startFrame: number; frames: number } => {
    const f = clamp(Math.round(nextFrames), minFrames, maxAllowed);
    const s = clamp(Math.round(nextStart), 0, Math.max(0, total - f));
    return { startFrame: s, frames: f };
  };

  /** 開始側ハンドル: 終端を固定したまま頭を動かす（＝長さが変わる）。長さの
   * 上限に `end` を混ぜてあるのは、頭を素材の外（負のフレーム）へ引っぱったときに
   * 「頭が 0 で止まる代わりに尻尾が右へ伸びる」という裏返りを防ぐため。 */
  const windowFromStartEdge = (desiredStart: number): { startFrame: number; frames: number } => {
    const end = viewEnd;
    const f = clamp(Math.round(end - desiredStart), minFrames, Math.min(maxAllowed, Math.max(minFrames, end)));
    return clampWindow(end - f, f);
  };

  /** 終了側ハンドル: 頭を固定したまま尻尾を動かす。上限に「頭から素材末尾までの
   * 残り」を混ぜてあるのは、尻尾を素材の外へ引っぱったときに `clampWindow` が
   * 代わりに**頭**を手前へずらしてしまうのを防ぐため（尻尾は素材の端で止まる）。 */
  const windowFromEndEdge = (desiredEnd: number): { startFrame: number; frames: number } => {
    const s = view.startFrame;
    const room = total > 0 ? total - s : maxAllowed;
    const f = clamp(Math.round(desiredEnd) - s, minFrames, Math.min(maxAllowed, Math.max(minFrames, room)));
    return clampWindow(s, f);
  };

  /** ポインタの clientX を素材上のフレーム番号へ。トラックが測れないうちは
   * 仮幅で計算する（比率しか使わないので、実測が入れば自然に正しくなる）。 */
  const frameFromClientX = (clientX: number): number => {
    const track = trackRef.current;
    if (!track || total <= 0) return 0;
    const rect = track.getBoundingClientRect();
    const width = rect.width > 0 ? rect.width : FALLBACK_TRACK_WIDTH;
    return Math.round(((clientX - rect.left) / width) * total);
  };

  const beginDrag = (e: ReactPointerEvent, kind: DragState["kind"]): void => {
    if (disabled) return;
    e.currentTarget.setPointerCapture?.(e.pointerId);
    const anchor = kind === "end" ? viewEnd : view.startFrame;
    setDrag({
      kind,
      pointerId: e.pointerId,
      grabOffset: frameFromClientX(e.clientX) - anchor,
      preview: { startFrame: view.startFrame, frames: view.frames },
    });
  };

  const handlePointerMove = (e: ReactPointerEvent): void => {
    if (!drag || drag.pointerId !== e.pointerId) return;
    const pointed = frameFromClientX(e.clientX) - drag.grabOffset;
    const next =
      drag.kind === "move"
        ? clampWindow(pointed, drag.preview.frames)
        : drag.kind === "start"
          ? windowFromStartEdge(pointed)
          : windowFromEndEdge(pointed);
    if (next.startFrame === drag.preview.startFrame && next.frames === drag.preview.frames) return;
    setDrag({ ...drag, preview: next });
  };

  const handlePointerUp = (e: ReactPointerEvent): void => {
    if (!drag || drag.pointerId !== e.pointerId) return;
    const { preview } = drag;
    e.currentTarget.releasePointerCapture?.(e.pointerId);
    setDrag(null);
    if (preview.startFrame !== startFrame || preview.frames !== frames) onChange(preview);
  };

  /** 取りこぼしたドラッグを、確定させずに畳む。`pointercancel`（OS/ジェスチャ
   * による中断）と `lostpointercapture`（キャプチャを横取りされた）の両方から
   * 呼ぶ —— `pointerup` が来ないままドラッグが死ぬ経路はこの 2 つ。 */
  const abortDrag = (e: ReactPointerEvent): void => {
    if (!drag || drag.pointerId !== e.pointerId) return;
    e.currentTarget.releasePointerCapture?.(e.pointerId);
    setDrag(null);
  };

  const handleKeyDown = (e: KeyboardEvent, kind: "start" | "end"): void => {
    if (disabled) return;
    let direction: -1 | 1;
    if (e.key === "ArrowRight" || e.key === "ArrowUp") direction = 1;
    else if (e.key === "ArrowLeft" || e.key === "ArrowDown") direction = -1;
    else return;
    e.preventDefault();
    const step = (e.shiftKey ? 8 : 1) * direction;
    const next =
      kind === "start" ? windowFromStartEdge(view.startFrame + step) : windowFromEndEdge(viewEnd + step);
    // 端に当たっているときは値が動かない = 何も上げない（ハンドルが止まる）。
    if (next.startFrame === startFrame && next.frames === frames) return;
    onChange(next);
  };

  const glueHead = clamp(glueHeadFrames, 0, view.frames);
  const glueTail = clamp(glueTailFrames, 0, Math.max(0, view.frames - glueHead));

  const handleProps = (kind: "start" | "end") => ({
    className: `range-band-handle range-band-handle-${kind}`,
    role: "slider" as const,
    tabIndex: disabled ? -1 : 0,
    "aria-disabled": disabled || undefined,
    onPointerDown: (e: ReactPointerEvent) => beginDrag(e, kind),
    onPointerMove: handlePointerMove,
    onPointerUp: handlePointerUp,
    onPointerCancel: abortDrag,
    onLostPointerCapture: abortDrag,
    onKeyDown: (e: KeyboardEvent) => handleKeyDown(e, kind),
  });

  return (
    <div className={`range-band${disabled ? " range-band-disabled" : ""}`} role="group" aria-label={t.label}>
      {/* 1. 素材全長帯。トラックそのものが素材の全長で、以下はすべてこの上の重ね。 */}
      <div className="range-band-track" ref={trackRef} data-testid="range-band-track">
        {/* 2. 窓（塗り・ドラッグ本体） */}
        <div
          className={`range-band-window${drag ? " range-band-window-dragging" : ""}`}
          data-testid="range-band-window"
          style={{ left: `${pct(view.startFrame)}%`, width: `${pct(view.frames)}%` }}
          onPointerDown={(e) => beginDrag(e, "move")}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={abortDrag}
          onLostPointerCapture={abortDrag}
        />
        {/* 3. 糊代帯。窓の子ではなく**兄弟**として重ねてある: CSS の
            `pointer-events: none` が効かない環境（テストの jsdom）でも、
            ここへのポインタ操作が窓のドラッグへ紛れ込まないようにするため。 */}
        {glueHead > 0 && (
          <div
            className="range-band-glue range-band-glue-head"
            data-testid="range-band-glue-head"
            aria-hidden="true"
            style={{ left: `${pct(view.startFrame)}%`, width: `${pct(glueHead)}%` }}
          />
        )}
        {glueTail > 0 && (
          <div
            className="range-band-glue range-band-glue-tail"
            data-testid="range-band-glue-tail"
            aria-hidden="true"
            style={{ left: `${pct(viewEnd - glueTail)}%`, width: `${pct(glueTail)}%` }}
          />
        )}
        {/* 4. ハンドル 2 つ */}
        <div
          {...handleProps("start")}
          data-testid="range-band-handle-start"
          aria-label={t.startHandle}
          aria-valuemin={Math.max(0, viewEnd - maxAllowed)}
          aria-valuemax={Math.max(0, viewEnd - minFrames)}
          aria-valuenow={view.startFrame}
          aria-valuetext={t.startValueText(view.startFrame, formatSeconds(view.startFrame, genFps))}
          style={{ left: `${pct(view.startFrame)}%` }}
        />
        <div
          {...handleProps("end")}
          data-testid="range-band-handle-end"
          aria-label={t.endHandle}
          aria-valuemin={view.startFrame + minFrames}
          aria-valuemax={Math.min(total, view.startFrame + maxAllowed)}
          aria-valuenow={viewEnd}
          aria-valuetext={t.endValueText(viewEnd, formatSeconds(viewEnd, genFps))}
          style={{ left: `${pct(viewEnd)}%` }}
        />
      </div>
      {/* 読み出しは 2 段。主表示（タイムラインの何フレーム目を撮り直すか）は
          親が作り、副表示（窓の長さ）は括弧書きでその下に添える。ユーザーが
          知りたいのは「どこを」であって「何フレームぶんか」は補足、という
          オーナー目視フィードバック（2026-08-10 ③）に沿った並び。 */}
      {formatPlacement && (
        <p className="hint range-band-placement" data-testid="range-band-placement">
          {formatPlacement(view.startFrame, view.frames)}
        </p>
      )}
      <p className="hint range-band-readout" data-testid="range-band-readout">
        {t.readout(view.frames, formatSeconds(view.frames, genFps))}
      </p>
      {view.frames >= maxAllowed && (
        <p className="hint range-band-note" data-testid="range-band-max-note">
          {t.maxNote(maxAllowed)}
        </p>
      )}
      {glueHead + glueTail > 0 && <p className="hint range-band-note">{t.glueNote}</p>}
    </div>
  );
}
