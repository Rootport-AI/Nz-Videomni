import { useStrings } from "../../i18n/LanguageContext";
import type { Pads } from "./outpaintGeometry";

export interface OutpaintPreviewProps {
  /** 元動画の寸法（保持領域）。 */
  sourceWidth: number;
  sourceHeight: number;
  /** 上下左右に描き足す量。利用者が入れた現在値そのもの（0 起点・1px 刻み）で、
   * 自動で足される分はもう無い（2026-08-11 アライメント撤去）。 */
  pads: Pads;
  /**
   * なじみ帯（元の映像と描き足した映像が混ぜ合わされる幅）の実寸ピクセル。
   * 呼び出し側が `outpaintGeometry.featherWidthPx(段数, キャンバス長辺)` で
   * 出した値をそのまま渡す。
   *
   * ここが以前は 150 の固定値だった（マスクブラー実装前）。その 150 は
   * 「1920px のキャンバス・既定の 5 段」でしか正しくない数字で、それ以外の
   * 解像度では破線の位置も凡例の文言もずれていた —— キャンバスに依存する量を
   * 定数で持っていたのが原因なので、計算そのものを純関数へ出し、ここは
   * 受け取るだけにしてある。0 なら帯は描かない。
   */
  blendBandPx: number;
}

/**
 * 拡張後キャンバスの見取り図。動画のサムネイルは出さない（設計方針書 §4-3）—
 * 塗った長方形が元動画、線だけの長方形が広げたあとのキャンバス。
 *
 * A plain inline `<svg>`: no external library, no image decoding, and the
 * `viewBox` is the canvas itself, so the two rectangles are drawn in real pixel
 * coordinates and the aspect ratio comes out right for free. Colours come from
 * the existing CSS custom properties (see `EditScreen.css`), so light and dark
 * themes both work without a second code path.
 *
 * Accessibility: the whole figure is one `role="img"` with an `aria-label`
 * naming both sizes — a screen reader gets the numbers, which is the entire
 * information content of the picture.
 */
export function OutpaintPreview({ sourceWidth, sourceHeight, pads, blendBandPx }: OutpaintPreviewProps) {
  const strings = useStrings();
  const t = strings.edit.outpainting;

  if (!(sourceWidth > 0 && sourceHeight > 0)) {
    // 素材が未読込のあいだは **文言を出さず、枠だけ** を置く（オーナー指示、
    // 2026-08-09）。以前ここには「元になる動画を読み込むとプレビューが出ます」
    // の一文があったが、素材カードのすぐ下という位置そのものが同じことを言って
    // いるため削除した（`strings.edit.outpainting.previewEmpty` ごと廃止）。
    // 空でも同じ場所に同じ高さの箱が残るので、素材が届いた瞬間にパネルが
    // 飛び跳ねないという元の一文の役割は、この空枠がそのまま引き継ぐ。
    return <div className="outpaint-preview-empty" />;
  }

  const canvasWidth = sourceWidth + pads.left + pads.right;
  const canvasHeight = sourceHeight + pads.top + pads.bottom;
  // The blend band is drawn INSIDE the kept region, on the sides that actually
  // gained padding — a side with no padding has no seam to blend. It is skipped
  // entirely when the source is too small for the band to be meaningful (the
  // inset rectangle would collapse or invert).
  const band = blendBandPx > 0 ? blendBandPx : 0;
  const bandLeft = pads.left > 0 ? band : 0;
  const bandRight = pads.right > 0 ? band : 0;
  const bandTop = pads.top > 0 ? band : 0;
  const bandBottom = pads.bottom > 0 ? band : 0;
  const bandWidth = sourceWidth - bandLeft - bandRight;
  const bandHeight = sourceHeight - bandTop - bandBottom;
  const showBand = bandWidth > 0 && bandHeight > 0 && bandLeft + bandRight + bandTop + bandBottom > 0;

  // Stroke widths are in viewBox units, so they must scale with the canvas or a
  // 4096-wide preview would draw hairlines and a 384-wide one would draw slabs.
  const stroke = Math.max(2, Math.round(Math.max(canvasWidth, canvasHeight) / 200));

  return (
    <div className="outpaint-preview">
      <svg
        className="outpaint-preview-svg"
        viewBox={`0 0 ${canvasWidth} ${canvasHeight}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={t.previewAlt(canvasWidth, canvasHeight, sourceWidth, sourceHeight)}
      >
        {/* 広げたあとのキャンバス: 線だけ。inset by half the stroke so the
            outline is not clipped by the viewBox edge. */}
        <rect
          className="outpaint-preview-canvas"
          x={stroke / 2}
          y={stroke / 2}
          width={Math.max(0, canvasWidth - stroke)}
          height={Math.max(0, canvasHeight - stroke)}
          strokeWidth={stroke}
        />
        {/* 元動画: 塗り。 */}
        <rect
          className="outpaint-preview-source"
          x={pads.left}
          y={pads.top}
          width={sourceWidth}
          height={sourceHeight}
        />
        {showBand && (
          <rect
            className="outpaint-preview-band"
            x={pads.left + bandLeft}
            y={pads.top + bandTop}
            width={bandWidth}
            height={bandHeight}
            strokeWidth={stroke}
            strokeDasharray={`${stroke * 4} ${stroke * 3}`}
          />
        )}
      </svg>
      <ul className="outpaint-preview-legend">
        <li className="outpaint-preview-legend-source">{t.previewLegendSource}</li>
        <li className="outpaint-preview-legend-canvas">{t.previewLegendCanvas}</li>
        {showBand && <li className="outpaint-preview-legend-band">{t.previewLegendBlend(band)}</li>}
      </ul>
    </div>
  );
}
