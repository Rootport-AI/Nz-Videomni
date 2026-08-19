import {
  BridgeError,
  DEFAULT_TIMEOUT_MS,
  NO_LOCAL_TIMEOUT_METHODS,
  isBridgeEvent,
  isBridgeResponse,
  type BridgeEvent,
  type BridgeMethod,
  type BridgeRequest,
  type BridgeResponse,
  type ParamsOf,
  type ResultOf,
} from "./types";

type EventHandler = (data: unknown) => void;

interface PendingEntry {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
  /** Absent for methods in `NO_LOCAL_TIMEOUT_METHODS` (e.g. `ui.pickFile`),
   * which never get a local wait ceiling — see that set's doc comment. */
  timer: ReturnType<typeof setTimeout> | undefined;
}

/** Contract v7: extra transport-level data alongside the request itself.
 * `additionalObjects` carries the raw `File`(-like) objects a caller wants
 * WebView2 to attach via `postMessageWithAdditionalObjects` (see
 * `webviewBridge.ts`'s `send`) — unused by the mock transport, which has no
 * such channel. */
export interface RequestTransportExtra {
  additionalObjects?: unknown[];
}

export interface RequestDispatcherOptions {
  /** Called with the outgoing request; must hand it to the transport
   * (e.g. `window.chrome.webview.postMessage`). `extra`, when given, is
   * transport-level data alongside the request (contract v7:
   * `additionalObjects` for `postMessageWithAdditionalObjects`) — plain
   * `request()` calls never pass it. */
  send: (request: BridgeRequest, extra?: RequestTransportExtra) => void;
  /** Overrides DEFAULT_TIMEOUT_MS, mainly for tests. */
  timeoutMs?: number;
}

/**
 * Transport-agnostic core of the bridge protocol: request id assignment,
 * pending-request bookkeeping, timeout handling, response matching and
 * event fan-out. `webviewBridge.ts` wires this to the real WebView2 message
 * channel; it is also exercised directly in tests with a fake `send`.
 */
export class RequestDispatcher {
  private nextId = 1;
  private readonly pending = new Map<number, PendingEntry>();
  private readonly eventHandlers = new Map<string, Set<EventHandler>>();
  private readonly send: (request: BridgeRequest, extra?: RequestTransportExtra) => void;
  private readonly timeoutMs: number;

  constructor(options: RequestDispatcherOptions) {
    this.send = options.send;
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  }

  request<M extends BridgeMethod>(
    method: M,
    params: ParamsOf<M>,
    /** Contract v7: transport-level extras (e.g. `additionalObjects` for a
     * drag-and-drop file) passed straight through to `send` alongside the
     * request. Omitted by every caller except `requestWithFiles`. */
    transportExtra?: RequestTransportExtra,
  ): Promise<ResultOf<M>> {
    const id = this.nextId++;
    // v4.1: a per-call `timeoutMs` in `params` (currently only
    // `backend.request` declares one, e.g. `joinJob`'s 120s override — see
    // `bridge/types.ts`) overrides this dispatcher's default local wait for
    // *this* request only, so a deliberately slow call isn't rejected
    // client-side before native's own (longer) timeout has a chance to fire.
    const perCallTimeoutMs = (params as { timeoutMs?: unknown } | null)?.timeoutMs;
    const timeoutMs = typeof perCallTimeoutMs === "number" ? perCallTimeoutMs : this.timeoutMs;
    // ui.pickFile/ui.pickFolder open a synchronous native dialog with no
    // notion of "too long" — the user may sit on it indefinitely. Don't arm
    // a local timer for these; only dispose() may still settle them early.
    const noLocalTimeout = NO_LOCAL_TIMEOUT_METHODS.has(method);

    return new Promise<ResultOf<M>>((resolve, reject) => {
      const timer = noLocalTimeout
        ? undefined
        : setTimeout(() => {
            this.pending.delete(id);
            reject(
              new BridgeError(
                "TIMEOUT",
                `Bridge request "${method}" (id ${id}) timed out after ${timeoutMs}ms`,
              ),
            );
          }, timeoutMs);

      this.pending.set(id, {
        resolve: resolve as (value: unknown) => void,
        reject,
        timer,
      });

      this.send({ id, method, params }, transportExtra);
    });
  }

  /** Subscribe to a native-initiated event; returns an unsubscribe function. */
  on(event: string, handler: EventHandler): () => void {
    let handlers = this.eventHandlers.get(event);
    if (!handlers) {
      handlers = new Set();
      this.eventHandlers.set(event, handlers);
    }
    handlers.add(handler);

    return () => {
      handlers?.delete(handler);
    };
  }

  /** Feed a raw incoming message (from `addEventListener('message', ...)`)
   * into the dispatcher. Messages that are neither a recognized response nor
   * a recognized event are silently ignored. */
  handleIncoming(message: unknown): void {
    if (isBridgeResponse(message)) {
      this.handleResponse(message);
      return;
    }
    if (isBridgeEvent(message)) {
      this.handleEvent(message);
    }
  }

  /** Reject all still-pending requests and drop all event subscriptions.
   * Call when the underlying transport is torn down. */
  dispose(): void {
    for (const entry of this.pending.values()) {
      if (entry.timer !== undefined) clearTimeout(entry.timer);
      entry.reject(new BridgeError("DISPOSED", "Bridge was disposed"));
    }
    this.pending.clear();
    this.eventHandlers.clear();
  }

  private handleResponse(message: BridgeResponse): void {
    const entry = this.pending.get(message.id);
    if (!entry) return; // Unknown or already-settled (e.g. timed out) id.

    this.pending.delete(message.id);
    if (entry.timer !== undefined) clearTimeout(entry.timer);

    if (message.ok) {
      entry.resolve(message.result);
    } else {
      entry.reject(new BridgeError(message.error.code, message.error.message));
    }
  }

  private handleEvent(message: BridgeEvent): void {
    const handlers = this.eventHandlers.get(message.event);
    if (!handlers) return;
    for (const handler of handlers) {
      handler(message.data);
    }
  }
}
