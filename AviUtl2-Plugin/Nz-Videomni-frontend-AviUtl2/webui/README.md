# Nz-Videomni WebUI

React + TypeScript + Vite frontend for the Nz-Videomni AviUtl2 plugin. It's a
single-page app with four modes — **Single** (T2V/I2V generation, including
IC-LoRA reference-video conditioning and an audio attach section for
single-shot audio-to-video, plus keyframes and style/control LoRA selection),
**Chained** (multi-clip chains with presets, per-clip LoRAs, and optional
chunked upsampling), **Edit** (video-editing tools, with its own sub-tab row —
**Retake** (regenerates a selected span of an existing video; shipped
2026-08-10, see `../../../Docs/PENDING_TASKS.md` §1-17), **Outpainting** (shipped
2026-08-09, `../../../Docs/PENDING_TASKS_CLOSED.md` §3-70) and a disabled
**Inpainting** placeholder (`../../../Docs/PENDING_TASKS.md` §3-55)), and **Inventory**
(job history, downloads, and
a LoRA browser — model management lives in the settings panel's `ModelsPanel`,
not here) — plus a batch-A2V section (stateless folder-scan-driven bulk
generation, no CSV manifest) and a job ledger (a table of all jobs shown
next to the Generate button; there
is no separate job lane or reservation queue — the backend runs one job at a
time, so the Generate button just disables itself while one is running). The
tab bar also has a **Toolbox** tab, the one remaining disabled placeholder
with no panel behind it yet. It
runs inside AviUtl2's WebView2 control and
talks to the AviUtl2 host and the Nz-Videomni backend entirely through the
native JSON-RPC bridge described below (no direct network calls from the page).

It runs both:

- **In the browser** (`npm run dev` / `npm run build` + any static server) —
  talks to an in-process **mock bridge**, no native host required.
- **Inside AviUtl2's WebView2 control** — talks to the real native plugin
  through `window.chrome.webview`.

Which one is used is decided automatically at startup (see
[Bridge architecture](#bridge-architecture) below).

Current generation defaults are 1280×768, 361 frames (raised from 257 on
2026-08-19, see `Docs/DEVLOG.md` §85), `crop_output`
1280×720 (matches the backend's `standard_720p` preset). The generation
size cap is 4096; resolutions above 1440p are not covered by the
spill-free-frames OOM warning (see `Docs/DEVLOG.md` §24.8/§22.3).

## Requirements

- Node.js v22.14.0+ and npm (matches the version used for this project).

## Getting started

All commands run from this `webui/` directory.

```bash
npm install       # first time only
npm run dev       # start the Vite dev server (mock bridge, browser only)
npm run build     # type-check (tsc -b) + production build -> dist/
npm run preview   # serve the dist/ build locally, for a quick sanity check
npm test          # run the vitest suite once (CI mode)
npm run test:watch  # run vitest in watch mode
npm run typecheck # tsc -b --noEmit, no build output
npm run lint      # oxlint
```

`npm run build` type-checks with `tsc -b` before invoking `vite build`, so a
type error fails the build. `npm run dev` opens a normal browser-only preview
at `http://localhost:5173/` — the connection badge will show
"Native bridge not available" only if the mock bridge itself fails; by
default the mock always answers `ping` successfully after a short simulated
delay, so in `dev` you should see it flip to "Connected (0.0.0-mock)".

### Deployment target: WebView2 virtual host

`vite.config.ts` sets `base: './'` so every asset reference in the built
`index.html` is relative (`./assets/...`) rather than root-absolute
(`/assets/...`). This matters because the native host does not serve
`dist/` from an HTTP root — it maps the folder to a **virtual host name**
via WebView2's `SetVirtualHostNameToFolderMapping`
(`https://app.nzvideomni.local/` → `webui/dist/`) and navigates to
`https://app.nzvideomni.local/index.html`. Root-absolute asset paths would still
resolve correctly against that virtual host, but relative paths are used so
the exact same `dist/` output also works if it's ever opened directly from
disk or served from a sub-path.

`dist/` and `node_modules/` are git-ignored (`webui/.gitignore`, inherited
from the Vite scaffold) — they are build artifacts, not source.

## Project layout

```
webui/
  src/
    bridge/            # RPC contract + both bridge implementations (see below)
    api/               # Backend REST client built on top of the bridge
    i18n/
      strings.ts       # All UI copy, en/ja dictionaries (see below)
      strings.test.ts  # Enforces en/ja key parity
      LanguageContext.tsx
    jobs/              # Job ledger: polling, JobsContext, JobLedger/JobCard, join/insert
    lora/              # LoRA selection helpers shared by Single/Chained
    modes/
      single/          # Single screen: T2V/I2V, keyframes, audio attach, LoRAs
      chained/          # Chained screen: multi-clip chains, presets, per-clip LoRAs
      batch/            # Batch A2V: folder scan, in-memory row list (stateless), batch runner
      inventory/        # Inventory screen: job history, LoRA browser
                        #   (model management lives in shell/ModelsPanel, not here)
    shell/             # AppShell, mode tabs, prompt bar, settings, models panel
    timeline/          # Right-click-from-timeline routing and prefill logic
    test/
      setupTests.ts    # vitest + @testing-library/jest-dom wiring
    App.tsx            # Root component, renders AppShell
    App.css / index.css
    App.test.tsx, App.chained.test.tsx, App.prefill.test.tsx
    main.tsx
  index.html
  vite.config.ts
```

## Bridge architecture

`src/bridge/types.ts` is **the single source of truth** for the RPC contract
between the WebUI and the native `.aux2` plugin (currently contract v7 — see
[`Docs/BRIDGE_CONTRACT.md`](../Docs/BRIDGE_CONTRACT.md) for the authoritative,
implementation-cross-checked reference). v7 adds `ui.resolveDroppedFiles`
for drag-and-drop (see below). If the contract changes, this file
changes first, and the native side is updated to match — everything else in
`src/bridge/` and in the UI is typed against it.

### Transport

- **WebUI → Native**: `window.chrome.webview.postMessage(request)` with a
  plain object — **not** `JSON.stringify`'d; WebView2 handles serialization.
  ```ts
  { id: number, method: string, params: object }
  ```
- **Native → WebUI**: delivered via
  `window.chrome.webview.addEventListener('message', e => ...)`. `e.data` is
  one of:
  ```ts
  // success
  { id: number, ok: true, result: object }
  // failure
  { id: number, ok: false, error: { code: string, message: string } }
  // native-initiated push (no id)
  { event: string, data: object }
  ```
  Events are actively used: `timeline.menuInvoked` is pushed by the native
  side when a timeline right-click menu action is chosen, and
  `src/timeline/useMenuRouter.ts` subscribes to it to route into the
  appropriate mode with a prefill. See `Docs/BRIDGE_CONTRACT.md` §1.4/§4.14
  for the full picture (including which other events exist and are not yet
  emitted by native).

### Methods

The full, versioned list of RPC methods, their params/results, error codes,
timeouts, and thread model lives in
[`Docs/BRIDGE_CONTRACT.md`](../Docs/BRIDGE_CONTRACT.md) (contract v7 as of
this writing) — that document is the source of truth and is kept in sync
with `src/bridge/types.ts` and the native implementation, so it isn't
duplicated here. At a glance, methods span plugin/edit-info queries
(`ping`, `getEditInfo`), backend proxying (`backend.request`,
`backend.downloadVideo`, `backend.uploadFile`, `backend.getBaseUrl`),
timeline interaction (`timeline.insertMedia`, `timeline.captureFrame`,
`timeline.getSelection`), native file/UI pickers (`ui.pickFile`,
`ui.pickFolder`, `ui.makeThumbnail`), settings (`settings.get`/`set`), and
the filesystem bridge used by batch A2V (`fs.listFiles`,
`fs.probeAudioDuration`).

Extra fields on a result are ignored by the WebUI. The WebUI also raises two
purely local error codes that never come from native: `TIMEOUT` (no response
within 10s by default) and `DISPOSED` (bridge torn down with requests still
in flight). All other error codes are native-originated and are enumerated
in `Docs/BRIDGE_CONTRACT.md` §5.

### Files

- **`types.ts`** — contract types (`BridgeRequest`, `BridgeResponse`,
  `BridgeEvent`, the `Method -> params/result` type maps, `BridgeError`) and
  the `NativeBridge` interface every implementation exposes:
  `request<M>(method, params): Promise<Result<M>>` and
  `on(event, handler): () => void`.
- **`requestDispatcher.ts`** — transport-agnostic core: id assignment,
  pending-request bookkeeping, the 10s default timeout, response matching,
  and event fan-out. Both the real and any test transport drive it via
  `handleIncoming(rawMessage)`.
- **`webviewBridge.ts`** — production implementation; wires
  `RequestDispatcher` to the real `window.chrome.webview` message channel.
  Throws if that object isn't present.
- **`mockBridge.ts`** — dev implementation used whenever there's no
  `window.chrome.webview` (browser dev/build, unit tests). `ping` resolves
  after a configurable simulated delay (default 200ms);
  `getEditInfo` resolves with
  `{ width: 1920, height: 1080, rate: 30, scale: 1, sampleRate: 44100, frame: 120 }`.
  Options exist to simulate failures (`failPing`, `failEditInfo`) and to
  override the returned edit info, for exercising error UI during
  development.
- **`index.ts`** — picks `webviewBridge` when `window.chrome.webview` exists,
  otherwise `mockBridge`, and exports a ready-to-use singleton `bridge`. Also
  re-exports everything from `types.ts`.
- **`webview2-global.d.ts`** — ambient typing for `window.chrome.webview`.
  Not imported anywhere at runtime (it's a `.d.ts`, no JS output); TypeScript
  picks it up automatically because it's inside `src/`.

## Testing

`npm test` runs the full suite once (vitest, jsdom environment); `npm run
test:watch` runs it in watch mode. Tests live next to the code they cover
(`*.test.ts` / `*.test.tsx` throughout `src/`, ~48 files at the time of
writing) rather than in a separate top-level directory. The bridge layer's
tests remain the most load-bearing ones since every mode ultimately talks to
native through it:

- `src/bridge/requestDispatcher.test.ts` — id assignment, id-based response
  matching (including out-of-order responses), error propagation, timeout,
  late-response-after-timeout is ignored, unknown-id responses are ignored,
  event fan-out only reaches subscribers of the right event name,
  unsubscribe works, malformed incoming messages are ignored, and
  `dispose()` rejects any requests still in flight.
- `src/bridge/mockBridge.test.ts` — simulated delay before resolving,
  default `getEditInfo` payload, override merging, `failPing`/`failEditInfo`
  flags, and the no-op `on()`.
- `src/App.test.tsx`, `App.chained.test.tsx`, `App.editTab.test.tsx`,
  `App.prefill.test.tsx` —
  app-level smoke tests wired to the real mock bridge (through
  `bridge/index.ts`'s automatic selection, since jsdom has no
  `window.chrome.webview`), covering the Single/Chained/Edit/Inventory shell
  (Toolbox is a disabled placeholder tab, not covered here) and
  right-click-menu prefill routing end to end. Note that these files scope
  ambiguous queries with a *singular* `getByRole("tabpanel")` — the one
  visible mode screen — so Edit's own sub-panels deliberately carry no
  `role="tabpanel"` (see `src/modes/edit/EditSubTabs.tsx`).
- `src/i18n/strings.test.ts` — asserts the `en` and `ja` dictionaries expose
  exactly the same set of keys and agree on which leaves are plain strings
  vs. template functions.

Beyond that, each mode (`src/modes/single/`, `src/modes/chained/`,
`src/modes/batch/`, `src/modes/edit/`, `src/modes/inventory/`), the shell
(`src/shell/`), and the
timeline right-click routing layer (`src/timeline/`) carry their own unit
tests for hooks and pure logic (form state, param derivation, batch
row-list scan/re-judge logic, menu routing, etc.) alongside the components
they back.
