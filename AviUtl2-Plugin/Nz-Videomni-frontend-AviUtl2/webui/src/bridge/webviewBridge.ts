// `webview2-global.d.ts` augments the global `Window` type ambiently; it does
// not need (and, being a .d.ts with no JS output, cannot have) a runtime
// import — TypeScript picks it up automatically because it's inside `src/`,
// which `tsconfig.app.json` includes.
import { RequestDispatcher } from "./requestDispatcher";
import type { NativeBridge } from "./types";

/**
 * Production bridge implementation: talks to the real AviUtl2 host through
 * `window.chrome.webview` (WebView2's postMessage channel). Throws if that
 * object is not present — callers should check availability (see
 * `bridge/index.ts`) before constructing this.
 */
export function createWebviewBridge(): NativeBridge {
  const webview = window.chrome?.webview;
  if (!webview) {
    throw new Error(
      "window.chrome.webview is not available; cannot create the WebView2 bridge.",
    );
  }

  const dispatcher = new RequestDispatcher({
    // Contract v7: when `extra.additionalObjects` is present (a
    // `requestWithFiles` call), use `postMessageWithAdditionalObjects` so
    // native's `WebMessageReceived` handler can resolve them via
    // `get_AdditionalObjects()`; otherwise the plain, pre-v7 `postMessage`.
    send: (request, extra) =>
      extra?.additionalObjects
        ? webview.postMessageWithAdditionalObjects(request, extra.additionalObjects)
        : webview.postMessage(request),
  });

  webview.addEventListener("message", (event) => {
    dispatcher.handleIncoming(event.data);
  });

  return {
    request: (method, params) => dispatcher.request(method, params),
    requestWithFiles: (method, params, files) => dispatcher.request(method, params, { additionalObjects: files }),
    on: (event, handler) => dispatcher.on(event, handler),
  };
}

/** True when running inside a host that exposes the WebView2 message channel. */
export function isWebviewHostAvailable(): boolean {
  return typeof window !== "undefined" && Boolean(window.chrome?.webview);
}
