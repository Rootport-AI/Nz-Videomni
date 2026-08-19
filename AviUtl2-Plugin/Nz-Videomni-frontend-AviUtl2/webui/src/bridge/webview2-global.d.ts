export {};

/**
 * Minimal ambient typing for the WebView2 host object injected into
 * `window.chrome.webview` when the page runs inside the AviUtl2 plugin's
 * WebView2 control. Only the subset used by this bridge is declared.
 */
declare global {
  interface Window {
    chrome?: {
      webview?: WebView2Host;
    };
  }

  interface WebView2Host {
    /** Posts a plain object (not a JSON string) to the native host. */
    postMessage(message: unknown): void;
    /** Contract v7: posts `message` (a plain object, not a JSON string) plus
     * `additionalObjects` (e.g. dropped `File`s) — native receives these via
     * `ICoreWebView2WebMessageReceivedEventArgs2::get_AdditionalObjects()`,
     * resolving each `File` to its real local path (`webview_host.cpp`). */
    postMessageWithAdditionalObjects(message: unknown, additionalObjects: unknown[]): void;
    addEventListener(
      type: "message",
      listener: (event: MessageEvent<unknown>) => void,
    ): void;
    removeEventListener(
      type: "message",
      listener: (event: MessageEvent<unknown>) => void,
    ): void;
  }
}
