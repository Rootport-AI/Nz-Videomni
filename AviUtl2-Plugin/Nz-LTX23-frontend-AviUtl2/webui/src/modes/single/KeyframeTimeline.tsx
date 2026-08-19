import { useLayoutEffect, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent as ReactPointerEvent } from "react";
import { useStrings } from "../../i18n/LanguageContext";
import {
  maxPlaceablePosition,
  nearestGridPosition,
  positionToRatio,
  ratioToFrame,
  resolveLanding,
  stepPin,
} from "./keyframeGrid";
import type { KeyframeItem } from "./keyframeUtils";
import "./KeyframeTimeline.css";

export interface KeyframeTimelineProps {
  /** Keyframe cards, already sorted ascending by `frameIdx` by the caller. */
  items: readonly KeyframeItem[];
  /** The video's total frame count. May transiently be an off-grid value
   * (not `8n+1`) while the duration field is being typed into. */
  numFrames: number;
  /** Frames-per-second, used only to render the seconds half of each label. */
  frameRate: number;
  /** Grid step (`conditioning_frame_idx_multiple`, 8 in practice). */
  multiple: number;
  /** Grid origin (`conditioning_keyframe_grid_offset`, 1 in practice). */
  offset: number;
  disabled: boolean;
  /** Called with an already-landing-resolved frame index — the parent should
   * store it verbatim (no further snapping). */
  onCommitFrameIdx: (id: string, frameIdx: number) => void;
}

/** Minimum on-screen gap (px) to a pin's nearest neighbour for its label to
 * stay permanently visible. Below this the pin joins a "crowded cluster" and
 * its label degrades to hover/focus/drag-only so neighbours don't overlap. */
const LABEL_MIN_GAP_PX = 64;

/** Assumed track width (px) when the real one can't be measured yet (initial
 * render, or a headless/jsdom environment with no layout). Keeps the crowding
 * heuristic deterministic instead of collapsing every pin into "crowded". */
const FALLBACK_TRACK_WIDTH = 320;

/** Floor (px) for the left-edge snap-to-zero zone, used when position `1`'s
 * pixel position is so close to the left edge that the natural 0↔1 midpoint
 * boundary would be a hair-thin, unusable target. `max` (not `min`) with the
 * natural half-way distance is deliberate: the natural boundary alone already
 * matches `nearestGridPosition`, so it wouldn't widen anything — we want the
 * zero zone to be *at least* this wide so dragging near the start reliably
 * lands on `0`. The trade-off is intentional and owner-approved: on a dense
 * grid this swallows position `1`, which is then reached via the arrow keys or
 * the card's FRAME field instead of the mouse. */
const SNAP_ZERO_MIN_PX = 14;

/** Formats the seconds half of a pin/interval label to one decimal. Guards a
 * zero/NaN frame rate (transient while config loads) so labels never read
 * `NaN`. */
function formatSeconds(frame: number, frameRate: number): string {
  if (!Number.isFinite(frameRate) || frameRate <= 0) return "0.0";
  return (frame / frameRate).toFixed(1);
}

/**
 * The Create screen's keyframe timeline bar (keyframe rework 2026-07-18): every
 * pin (keyframe thumbnail) sits on a single track at its `frameIdx`, including
 * position `0` — there is no separate start-frame slot. The position-`0` pin is
 * distinguished by shape/colour (square, accent-framed) and drawn above the
 * position-`1` pin so it stays grabbable when the two nearly overlap. Pins drag
 * along the bar (snapping live to the server grid via {@link keyframeGrid}, with
 * a widened snap zone at the left edge for `0`) and step with the arrow keys;
 * only landing-resolved values are ever committed upward. All grid math lives in
 * `keyframeGrid.ts` — this component only maps pixels ⇄ frames and owns the
 * transient drag state.
 */
export function KeyframeTimeline({
  items,
  numFrames,
  frameRate,
  multiple,
  offset,
  disabled,
  onCommitFrameIdx,
}: KeyframeTimelineProps) {
  const strings = useStrings();
  const t = strings.single.keyframes;
  const trackRef = useRef<HTMLDivElement>(null);
  // The pin currently being dragged and its live, landing-resolved preview
  // position. Only this local state moves during a drag — the parent's items
  // (and therefore every other pin) stay put until `onPointerUp` commits.
  const [drag, setDrag] = useState<{ id: string; preview: number } | null>(null);
  // Measured track width, driving the label-crowding heuristic. Falls back to
  // FALLBACK_TRACK_WIDTH until a real layout measurement lands.
  const [trackWidth, setTrackWidth] = useState(0);

  useLayoutEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    const measure = () => setTrackWidth(el.getBoundingClientRect().width);
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const effectiveWidth = trackWidth > 0 ? trackWidth : FALLBACK_TRACK_WIDTH;
  const maxPlaceable = maxPlaceablePosition(numFrames, multiple, offset);

  /** The frame a pin is drawn/labelled at right now: its live preview while
   * dragging, otherwise its committed `frameIdx`. */
  const displayFrameOf = (item: KeyframeItem): number =>
    drag?.id === item.id ? drag.preview : item.frameIdx;

  /** Converts a pointer's `clientX` into a desired grid position. The left edge
   * carries a widened snap-to-zero zone (see {@link SNAP_ZERO_MIN_PX}) so
   * dragging near the start reliably lands on `0`; past that zone the raw frame
   * snaps to the nearest grid tick. Returns `null` only if the track isn't
   * measurable yet. */
  const desiredFromClientX = (clientX: number): number | null => {
    const track = trackRef.current;
    if (!track) return null;
    const rect = track.getBoundingClientRect();
    // Half the pixel distance to position `1` is the natural 0↔1 boundary;
    // widen it to at least SNAP_ZERO_MIN_PX so `0` is always easy to hit.
    const offsetPx = positionToRatio(offset, numFrames) * rect.width;
    const snapZeroPx = Math.max(offsetPx / 2, SNAP_ZERO_MIN_PX);
    if (clientX <= rect.left + snapZeroPx) return 0;
    const ratio = rect.width > 0 ? (clientX - rect.left) / rect.width : 0;
    const rawFrame = ratioToFrame(ratio, numFrames);
    return nearestGridPosition(rawFrame, numFrames, multiple, offset);
  };

  /** Resolves where `desired` actually lands for pin `id`, sliding past any
   * pins other than `id` that already occupy grid ticks. */
  const resolveFor = (id: string, desired: number): number => {
    const occupied = new Set(items.filter((i) => i.id !== id).map((i) => i.frameIdx));
    const origin = items.find((i) => i.id === id)?.frameIdx ?? 0;
    return resolveLanding({ desired, occupied, numFrames, multiple, offset, origin });
  };

  const handlePointerDown = (e: ReactPointerEvent, item: KeyframeItem): void => {
    if (disabled) return;
    e.currentTarget.setPointerCapture?.(e.pointerId);
    setDrag({ id: item.id, preview: item.frameIdx });
  };

  const handlePointerMove = (e: ReactPointerEvent, item: KeyframeItem): void => {
    if (disabled || drag?.id !== item.id) return;
    const desired = desiredFromClientX(e.clientX);
    if (desired === null) return;
    setDrag({ id: item.id, preview: resolveFor(item.id, desired) });
  };

  const handlePointerUp = (e: ReactPointerEvent, item: KeyframeItem): void => {
    if (drag?.id !== item.id) return;
    const preview = drag.preview;
    e.currentTarget.releasePointerCapture?.(e.pointerId);
    setDrag(null);
    onCommitFrameIdx(item.id, preview);
  };

  /** Aborts an in-flight drag without committing — the pin snaps back to its
   * pre-drag position. Wired to both `pointercancel` (OS/gesture interruption)
   * and `lostpointercapture` (capture stolen mid-drag), the two ways a drag can
   * die without a `pointerup`. */
  const handleDragAbort = (e: ReactPointerEvent, item: KeyframeItem): void => {
    if (drag?.id !== item.id) return;
    e.currentTarget.releasePointerCapture?.(e.pointerId);
    setDrag(null);
  };

  const handleKeyDown = (e: KeyboardEvent, item: KeyframeItem): void => {
    if (disabled) return;
    let direction: -1 | 1;
    if (e.key === "ArrowRight" || e.key === "ArrowUp") direction = 1;
    else if (e.key === "ArrowLeft" || e.key === "ArrowDown") direction = -1;
    else return;
    e.preventDefault();
    const occupied = new Set(items.filter((i) => i.id !== item.id).map((i) => i.frameIdx));
    const next = stepPin(item.frameIdx, direction, occupied, numFrames, multiple, offset);
    if (next !== item.frameIdx) onCommitFrameIdx(item.id, next);
  };

  // Which pins keep their label permanently visible vs. degrade to
  // hover/focus/drag: a pin whose nearest neighbour is at least
  // LABEL_MIN_GAP_PX away (or which has no neighbour) stays labelled. Every
  // pin — including position `0` — participates.
  const labelledIds = new Set<string>();
  const barSorted = items
    .map((item) => ({ id: item.id, ratio: positionToRatio(displayFrameOf(item), numFrames) }))
    .sort((a, b) => a.ratio - b.ratio);
  barSorted.forEach((pin, idx) => {
    const prev = barSorted[idx - 1];
    const next = barSorted[idx + 1];
    let nearestPx = Infinity;
    if (prev) nearestPx = Math.min(nearestPx, Math.abs(pin.ratio - prev.ratio) * effectiveWidth);
    if (next) nearestPx = Math.min(nearestPx, Math.abs(next.ratio - pin.ratio) * effectiveWidth);
    if (nearestPx >= LABEL_MIN_GAP_PX) labelledIds.add(pin.id);
  });

  const renderPin = (item: KeyframeItem, index: number) => {
    const frame = displayFrameOf(item);
    const isDragging = drag?.id === item.id;
    const isZero = frame === 0;
    const seconds = formatSeconds(frame, frameRate);
    const valueText = t.pinValueText(frame, seconds);
    // A dragging pin always shows its label; the rest follow the crowding
    // heuristic computed above.
    const labelled = isDragging || labelledIds.has(item.id);
    const classNames = [
      "kf-pin",
      "kf-pin-bar",
      // Position 0: square, accent-framed, and raised above the near-overlapping
      // position-1 pin so it stays grabbable/visible.
      isZero ? "kf-pin-zero" : "",
      isDragging ? "kf-pin-dragging" : "",
      labelled ? "kf-pin-labelled" : "",
      item.thumbnailDataUrl ? "" : "kf-pin-empty",
    ]
      .filter(Boolean)
      .join(" ");

    return (
      <div
        key={item.id}
        data-kf-pin-id={item.id}
        className={classNames}
        role="slider"
        tabIndex={disabled ? -1 : 0}
        aria-label={t.pinAriaLabel(index + 1)}
        aria-valuemin={0}
        aria-valuemax={numFrames - 1}
        aria-valuenow={frame}
        aria-valuetext={valueText}
        aria-disabled={disabled || undefined}
        style={{ left: `${positionToRatio(frame, numFrames) * 100}%` }}
        onPointerDown={(e) => handlePointerDown(e, item)}
        onPointerMove={(e) => handlePointerMove(e, item)}
        onPointerUp={(e) => handlePointerUp(e, item)}
        onPointerCancel={(e) => handleDragAbort(e, item)}
        onLostPointerCapture={(e) => handleDragAbort(e, item)}
        onKeyDown={(e) => handleKeyDown(e, item)}
      >
        <span className="kf-pin-hit" aria-hidden="true" />
        <span className="kf-pin-thumb" aria-hidden="true">
          {item.thumbnailDataUrl ? (
            <img src={item.thumbnailDataUrl} alt="" draggable={false} />
          ) : (
            <span className="kf-pin-placeholder">{index + 1}</span>
          )}
        </span>
        <span className="kf-pin-label" aria-hidden="true">
          {valueText}
        </span>
      </div>
    );
  };

  // Interval (gap) markers between consecutive occupied positions. Decorative /
  // aria-hidden; recomputed from live preview positions so gaps update as a pin
  // drags.
  const presentFrames = [...items].map(displayFrameOf).sort((a, b) => a - b);
  const intervals: { key: string; left: number; text: string }[] = [];
  for (let i = 1; i < presentFrames.length; i++) {
    const prev = presentFrames[i - 1];
    const curr = presentFrames[i];
    if (prev === undefined || curr === undefined) continue;
    const gap = curr - prev;
    const left = ((positionToRatio(prev, numFrames) + positionToRatio(curr, numFrames)) / 2) * 100;
    intervals.push({ key: `${prev}-${curr}-${i}`, left, text: t.intervalLabel(gap, formatSeconds(gap, frameRate)) });
  }

  // The non-placeable tail: everything past the last valid grid tick.
  const deadZoneStart = positionToRatio(maxPlaceable, numFrames) * 100;
  const showDeadZone = deadZoneStart < 100;

  return (
    <div
      className={`kf-timeline${disabled ? " kf-timeline-disabled" : ""}`}
      role="group"
      aria-label={t.timelineLabel}
    >
      <div className="kf-track" ref={trackRef}>
        {showDeadZone && (
          <div className="kf-deadzone" aria-hidden="true" style={{ left: `${deadZoneStart}%`, right: 0 }} />
        )}
        {intervals.map((interval) => (
          <span key={interval.key} className="kf-interval" aria-hidden="true" style={{ left: `${interval.left}%` }}>
            {interval.text}
          </span>
        ))}
        {items.map((item, index) => renderPin(item, index))}
      </div>
    </div>
  );
}
