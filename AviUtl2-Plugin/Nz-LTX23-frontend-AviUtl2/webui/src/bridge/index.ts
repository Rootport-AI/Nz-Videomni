/**
 * Bridge entry point: selects the production WebView2 bridge when the app is
 * hosted inside AviUtl2 (`window.chrome.webview` present), otherwise falls
 * back to the dev mock bridge (plain browser / `npm run dev` / tests that
 * import this module directly).
 */
import { createMockBridge } from "./mockBridge";
import { createWebviewBridge, isWebviewHostAvailable } from "./webviewBridge";
import type { NativeBridge } from "./types";

export * from "./types";
export { createMockBridge } from "./mockBridge";
export type { MockBridgeOptions, MockBridge } from "./mockBridge";
export { createWebviewBridge, isWebviewHostAvailable } from "./webviewBridge";
export { RequestDispatcher } from "./requestDispatcher";

/** Creates the appropriate bridge for the current environment. */
export function createBridge(): NativeBridge {
  return isWebviewHostAvailable() ? createWebviewBridge() : createMockBridge();
}

/** Singleton bridge instance for app-wide use. */
export const bridge: NativeBridge = createBridge();
