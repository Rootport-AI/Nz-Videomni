import "@testing-library/jest-dom/vitest";

// jsdom doesn't implement `Element.scrollIntoView` (real browsers/WebView2 do).
// The job ledger (`jobs/JobLedger.tsx`) scrolls a freshly submitted job into
// view via it, so stub it to a no-op instead of letting it throw during tests.
if (typeof Element !== "undefined" && !Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {};
}

// jsdom doesn't implement the Pointer Capture API (real browsers/WebView2 do).
// The keyframe timeline's pin-dragging (`modes/single/KeyframeTimeline.tsx`)
// captures the pointer on drag-start so the drag keeps tracking outside the
// pin's own bounds, so stub these to no-ops instead of letting them throw.
if (typeof Element !== "undefined" && !Element.prototype.setPointerCapture) {
  Element.prototype.setPointerCapture = () => {};
}
if (typeof Element !== "undefined" && !Element.prototype.releasePointerCapture) {
  Element.prototype.releasePointerCapture = () => {};
}
if (typeof Element !== "undefined" && !Element.prototype.hasPointerCapture) {
  Element.prototype.hasPointerCapture = () => false;
}

// jsdom doesn't implement `PointerEvent` either — it only knows `MouseEvent`.
// The keyframe timeline listens for pointerdown/move/up to drag pins, so
// tests need a minimal `PointerEvent` shim (extends `MouseEvent`, which
// already carries clientX/clientY/button through) when jsdom hasn't supplied
// a real one. Deliberately NOT `implements PointerEvent`: the DOM lib's
// `PointerEvent` interface gains/renames required properties across
// TypeScript versions (e.g. `persistentDeviceId`), which makes a full
// `implements` pledge brittle — this shim only carries what the timeline's
// tests actually touch (pointerId/pointerType; clientX etc. via MouseEvent).
if (typeof window !== "undefined" && typeof window.PointerEvent === "undefined") {
  class PointerEventPolyfill extends MouseEvent {
    readonly pointerId: number;
    readonly pointerType: string;

    constructor(type: string, params: PointerEventInit = {}) {
      super(type, params);
      this.pointerId = params.pointerId ?? 0;
      this.pointerType = params.pointerType ?? "mouse";
    }
  }

  // Cast needed: the shim intentionally implements only the subset of
  // `PointerEvent` the tests use, not the full (version-dependent) DOM
  // interface.
  window.PointerEvent = PointerEventPolyfill as unknown as typeof PointerEvent;
}
