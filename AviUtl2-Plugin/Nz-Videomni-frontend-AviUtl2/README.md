# Nz-Videomni (AviUtl2 frontend)

A general-purpose AviUtl2 plugin (`.aux2`) that acts as a thin frontend for the
`Nz-Videomni` video-generation backend, which switches between several base
models (LTX 2.3, LTX 2.5 and, as an alpha, MiniMax H3) — the plugin header
carries a base-model dropdown that selects the active one (selecting a model
makes the server load it). The backend README lists what each base model can
and cannot do ([`../../README.md`](../../README.md) §7.1 and §7.6).

This directory is the frontend half of the **Nz-Videomni monorepo**: the backend
lives at the repository root (two levels up), and this tree — plus the built
`AviUtl2-Plugin\NzVideomni.aux2` next to it — is the plugin. Backend changes are
made at the repository root, not here; the two halves ship together. (The old
standalone frontend repository `Nz-LTX23-frontend-AviUtl2` is frozen.)

`NzVideomni.aux2` registers a dockable window titled **Nz-Videomni** and hosts a
**WebView2**-based React/TypeScript Web UI over a JSON-RPC bridge
(`window.chrome.webview.postMessage` / `PostWebMessageAsJson`).

Current release: **1.0.0**. The authoritative version string lives in two
places that must be kept in sync: `native/src/bridge_core.h`'s
`kPluginVersion` constant (also what the `ping` RPC method returns) and the
`-Version` default in `scripts/package.ps1` (used for the `.au2pkg.zip` file
name and `package.ini`).

## Features

### Screens

The tab bar shows five tabs, left to right: **Toolbox**, **Single**,
**Chained**, **Edit** and **Inventory**. Every tab stays mounted, so switching
tabs never discards what you entered; a tab or sub-tab the loaded base model
cannot run is greyed out. One line per screen here — the details are in
[`webui/README.md`](webui/README.md), and the backend README's tab overview is
[`../../README.md`](../../README.md) §3 ("AviUtl2 操作パネルのタブ").

- **Toolbox** — tools that do not generate video, in two sub-tabs: **Tracking**
  (makes a timeline partial filter's box follow a moving subject; design canon
  [`../../Docs/OBJECT_TRACKING_DESIGN.md`](../../Docs/OBJECT_TRACKING_DESIGN.md))
  and **mp4 info** (shows the generation conditions embedded in a generated
  mp4; [`Docs/API_REFERENCE.md`](Docs/API_REFERENCE.md) §3.23).
- **Single** — one video at a time: T2V / I2V with keyframes, the IC-LoRA
  reference video, single-shot A2V, and the batch panel (a2v from an audio
  folder, or i2v from an image folder alone;
  [`Docs/BATCH_A2V_I2V_MODE.md`](Docs/BATCH_A2V_I2V_MODE.md)).
- **Chained** — a long video from chained clips, including V2V continuation
  from a start source and arriving at an end source
  ([`Docs/API_REFERENCE.md`](Docs/API_REFERENCE.md) §5.2), its own A2V and
  IC-LoRA reference video (long A2V and long IC-LoRA across the clips), plus
  the batch i2v-long panel.
- **Edit** — reworks a finished video, in four sub-tabs: **Retake** (regenerates
  a selected span), **Outpainting** (widens the frame), **Inpainting** (repaints
  only the inside of a partial filter's box; design canon
  [`../../Docs/INPAINTING_DESIGN.md`](../../Docs/INPAINTING_DESIGN.md)) and
  **Alpha Gen** (makes a black-and-white matte video for background
  transparency from a video, LTX 2.5 only; the finished job's card then makes
  the transparent MOV or a color-filled video; design canon
  [`../../Docs/ALPHAGEN_DESIGN.md`](../../Docs/ALPHAGEN_DESIGN.md)).
- **Inventory** — the LoRA shelf of the selected base model (a card puts its
  tag into the prompt; the shelf follows the base model, and a LoRA the server
  marks unsupported carries a badge) and the generated-job history.

### Cross-cutting

- **Right-click generation**: an item on the AviUtl2 timeline's right-click
  menu opens the matching tab (and sub-tab) with the material and values
  pre-filled; you still review and submit from the panel. Which item lands
  where: [`Docs/RIGHTCLICK_REDESIGN_SPEC.md`](Docs/RIGHTCLICK_REDESIGN_SPEC.md).
- **Repeat count** (below the Generate button on Single and Chained) runs
  Generate N times in a row, each run submitted after the previous one
  completes.
- **Job ledger**: a table of all jobs (`JobLedger`, built from `JobCard`)
  sitting directly under the full-width Generate button and its estimate hint,
  in the same right-hand panel. It shows per-job progress, a completed-job
  preview (auto-expands only on the transition to completed, then a toggle),
  and per-job actions (timeline insert, V2V join, cancel, delete); polled at a
  2 s interval (`useJobsPoll.ts`'s `DEFAULT_INTERVAL_MS`) with no client-side
  deadline. There is no separate job lane or
  reservation queue: because the backend runs one job at a time and rejects a
  second with HTTP 409, the Generate button simply disables itself ("Busy…")
  while a job is running rather than queueing/reserving. (With Repeat count > 1
  it instead shows an enabled "Stop (N left)" until the last run; each repeat is
  submitted only after the previous job completes — still no queue.) A
  completion toast can
  be clicked to scroll to its job (switching from Inventory to Single if needed).
- **NAG (Negative Prompt) accordion**: a single collapsible "Negative Prompt"
  section directly under the prompt bar, outside the mode tabs — Single,
  Chained, and the batch panels all read the same shared state (`AppShell`-owned,
  no separate copy per screen). NAG (Normalized Attention Guidance) is a
  CFG-free way to make a negative prompt actually affect generation (backend
  commit `2ae497b`, 2026-07-28); the accordion's heading shows nothing while
  disabled and "Negative Prompt 【🔴ON】" once the checkbox is checked. The
  text body (seeded with a default 5-word prompt) is the only piece that
  persists across restarts (`localStorage`, `nzvideomni.nagNegativeText`); the
  enable checkbox always starts unchecked on a fresh session, and the
  scale/tau/alpha sliders are session-local (soft-disable: unchecking keeps
  every value intact, it just stops them from being sent). `nag_scale` is
  exposed as a top-level slider with tau/alpha tucked into a nested "Advanced"
  sub-accordion (a 🔄 button resets those three numeric values only, never the
  text), matching NAG's own recommendation to leave tau/alpha fixed and tune
  scale alone. **The batch panels silently borrow the shared NAG settings** —
  there is no separate batch UI or on-screen note about it. Generation is
  blocked on Single, Chained and the batch panels (with a reason shown)
  whenever NAG is enabled
  but the negative-prompt body is empty.
- **Settings panel**: houses model management (`ModelsPanel`) — 4 model
  categories (`transformer` / `text_encoder` / `video_vae` / `audio`, the
  backend's own `ModelCategory` classification — LoRAs are a separate
  concern, browsed via `GET /loras` on the Inventory screen above, not part of
  this category set) with refresh/load, backed by `GET /models` and `POST
  /pipeline/load`. Load requests use a 10-minute (`timeoutMs: 600000`) budget
  and surface the backend's HTTP 409 / `JOB_BUSY` busy state with a dedicated
  message. The same panel also hosts language/theme toggles, connection
  settings, and the danger zone (see below).
- **i18n**: English and Japanese, switchable in the settings panel.
- **Theme**: dark (default) and light, switchable next to the language
  toggle in the settings panel; persisted independently in `localStorage`
  (not the native settings bridge). The video-preview black background and
  button white text are deliberately kept the same in both themes.
- **Connection settings**: the backend base URL is a persisted runtime
  setting (`settings.get` / `settings.set`, `%LOCALAPPDATA%\NzVideomni\
  settings.json`), not compiled in.
- **Native bridge contract v16**: beyond the request/response envelope and
  `ping` / `getEditInfo` / `backend.request` / `backend.downloadVideo` /
  `backend.uploadFile` / `timeline.*`, v6 added `ui.pickFolder`,
  `fs.listFiles`, `fs.probeAudioDuration`, and a `destDir` / `fileName` /
  `noClobber` extension to `backend.downloadVideo`; v7 adds
  `ui.resolveDroppedFiles` and the reserved `__droppedPaths` key (always
  overwritten by native) for drag-and-drop support on the reference-video,
  A2V audio, and Chained start-source / end-source drop targets. A
  `reuseIfPresent` flag was
  later added to `backend.downloadVideo` (plain-video-only opt-in: skips the
  re-download when the destination file already exists with a non-zero size,
  used for the job-card re-insert cycle; a joined V2V video always re-fetches
  since a crossfade change alters its content). `fs.readTextFile`,
  `fs.writeTextFileAtomic`, and the `WRITE_LOCKED` error code were part of the
  v6 batch-A2V CSV manifest and have since been removed along with that
  manifest (2026-07-18, batch A2V went stateless — see `Docs/DEVLOG.md` §26).
  v8 adds the right-click-redesign RPCs (`timeline.getSelection` extensions,
  `timeline.insertProvisional`, `timeline.insertMediaForJob`, etc.) — see
  `Docs/BRIDGE_CONTRACT.md` §4.14.1. v9–v16: see `Docs/BRIDGE_CONTRACT.md` §8
  (v14 adds `usedFallback` to `timeline.insertMedia` and the one `layer_max+1`
  retreat; v15 adds `fps` to the `fs.probeMediaInfo` result, which is 0 by
  rule for mkv / webm; v16 adds `kind` / `color` to `backend.downloadVideo`
  (the Alpha Gen exits: `<jobId>_rgba.mov` / `<jobId>_chroma_<color>.mp4`)).
  See `Docs/BRIDGE_CONTRACT.md` for the full method/error reference.
- **`.au2pkg.zip` packaging**: `scripts/package.ps1` bundles the
  single-file-embedded `NzVideomni.aux2` plus `Language/*.aul2` (and
  `package.ini` / `package.txt`) into `dist/NzVideomni-<version>.au2pkg.zip` in
  the layout AviUtl2 expects (`Plugin/NzVideomni/`, `Language/`).
- **Generation fps is always a whole number**: whichever of the three fps
  sources feeds a form (material, project, or the backend defaults) and
  however it gets there (right-click prefill, manual edit, or lazy init),
  the value sent to the backend is an integer in `[1, 60]` — every entry
  point rounds through `snapFrameRate` (`webui/src/modes/single/paramUtils.ts`,
  §3-71/§3-72 in `Docs/PENDING_TASKS.md`). Only a right-click prefill that
  actually changed the value surfaces a one-shot toast; manual edits and
  mount overwrites round silently.

## Directory layout

Paths below are relative to this directory
(`AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\` in the monorepo).

```
CMakeLists.txt           Root build (plugin + tests)
CMakePresets.json         Ninja Release/Debug presets (+ a VS preset stub)
native/
  src/                    plugin.cpp, webview_host.{h,cpp}, bridge.{h,cpp},
                          bridge_core.{h,cpp} (RPC dispatch, kPluginVersion),
                          settings.{h,cpp}, fs_util.{h,cpp}, wav_probe.{h,cpp},
                          alias_util.{h,cpp}, provisional.{h,cpp},
                          log.{h,cpp}, strconv.{h,cpp}
                          (plus http_client / wic_png / mf_mp4_writer /
                          timeline_math / timeline_select, each {h,cpp})
  third_party/            Vendored deps (see third_party/README.md for versions):
                            webview2/ (WebView2 SDK headers + static loader),
                            wil/, nlohmann/ (json.hpp), doctest/
  res/                    webui.rc.in (CMake-configured RCDATA embed script)
  tests/                  doctest test suite (bridge_core, settings, fs_util,
                          wav_probe, strconv, http_client, wic_png, ...)
scripts/                  build.ps1, deploy.ps1, package.ps1, run-aviutl.ps1
Language/                 *.NzVideomni.aul2 (native UI strings, English/Japanese)
webui/                    Web UI (React/TypeScript, its own npm project;
                          built to webui/dist or the single-file
                          webui/dist-single/index.html)
Docs/                     BRIDGE_CONTRACT.md, API_REFERENCE.md, DEVLOG.md, ...
aviutl2_sdk/              Vendored AviUtl2 SDK headers/samples (read-only)
```

Elsewhere in the monorepo:

```
..\NzVideomni.aux2        The built plugin, tracked in git and shipped to users
                          (deploy.ps1 refreshes it; see Deploy below)
..\..\                    Backend (main.py, api/, services/, engine/, ...)
..\..\Docs\               Project-wide docs and the task ledger
                          (PENDING_TASKS.md)
```

## Prerequisites

- Windows x64.
- Visual Studio with the MSVC C++ toolset (this repo was validated against
  **VS 18 / MSVC 19.51**). `scripts\build.ps1` sources `vcvars64.bat`
  automatically.
- CMake + Ninja. If they are not already available, `build.ps1` downloads a
  pinned portable copy into `%LOCALAPPDATA%\NzVideomni\buildtools` on first use
  (per-user cache, **not** a global/system install, PATH untouched).

> Toolchain note: the installed Visual Studio (v18 / 2026 preview) does not ship
> the "C++ CMake tools" component, and stock CMake generators do not yet target
> a "Visual Studio 18" generator. The working configuration is therefore
> **Ninja + MSVC (vcvars)**, which is what the presets and scripts use. The
> `vs2022` preset is kept as a stub for environments that have a matching VS.

## Build

```powershell
# Release (default), embeds the Web UI (npm run build:single) by default.
# Add -RunTests to run the test suite, -Clean for a fresh build dir.
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Config Release -RunTests
```

> **`-RunTests` excludes the `"integration"` suite** (it runs
> `NzVideomni_tests.exe --test-suite-exclude=integration` directly, not `ctest`),
> so it passes safely without a mock backend running. See the Test section
> below for how to run the full suite, `"integration"` included.

Output: `build\ninja-release\NzVideomni.aux2` and `build\ninja-release\NzVideomni_tests.exe`.
The Web UI (`webui\` `npm run build:single` -> `webui\dist-single\index.html`)
is built automatically and compiled into the DLL as an RCDATA resource
(`-DNZVIDEOMNI_EMBED_WEBUI=ON`), so `NzVideomni.aux2` serves its UI from memory and
needs no on-disk `webui\` folder. Pass `-NoEmbedWebui` to skip this and get a
native-only build (no npm required; not deployable via `scripts\deploy.ps1`,
which is embedded-only — see its header comment). The older `-EmbedWebui`
switch is still accepted as a redundant, explicit way to request the (now
default) embedded build; `-EmbedWebui -NoEmbedWebui` together is an error.

To point the build at specific tools, set `NZVIDEOMNI_CMAKE`, `NZVIDEOMNI_NINJA`, or
`NZVIDEOMNI_VCVARS` to the full path of the respective executable/batch file.

> **webui-only rebuild, fixed.** Earlier, CMake's `OBJECT_DEPENDS` for the
> embedded Web UI resource was not wired up to track
> `webui/dist-single/index.html`'s contents, so rebuilding after a `webui/`-only
> change could leave a stale embedded UI in the `.aux2` (previously worked
> around by manually deleting
> `build\ninja-release\CMakeFiles\NzVideomni.dir\webui_embedded.rc.res`). This is
> fixed in `CMakeLists.txt` (`OBJECT_DEPENDS` now tracks the single-file HTML),
> so a normal `build.ps1` run picks up `webui/` changes without any manual
> workaround.

## Test

```powershell
# Built by build.ps1; run directly or via ctest:
build\ninja-release\NzVideomni_tests.exe
# or
cd build\ninja-release; ctest --output-on-failure
```

> **Integration tests need a mock backend running.** Both invocations above
> (the plain exe **and** `ctest`) include the `"integration"` doctest suite
> (`TEST_SUITE("integration")`), which assumes a mock backend is already
> listening on `127.0.0.1:18620` (`Nz-Videomni`'s `run.ps1` with
> `config.yaml`'s `model.backend: mock`) and will fail if none is running. In
> an environment without that backend up, exclude it by running the test exe
> directly with a filter (`ctest` has no equivalent doctest-suite filter, so
> use the exe form when the mock backend isn't up):
> `build\ninja-release\NzVideomni_tests.exe --test-suite-exclude=integration`.
> **`build.ps1 -RunTests` already does exactly this exclusion for you** (runs
> the exe directly with `--test-suite-exclude=integration`, not `ctest`), so
> it's safe to use without a mock backend up; the two invocations above are
> for running the full suite (`"integration"` included) or driving `ctest`
> directly.

Web UI tests (vitest) and the strict type-check gate live under `webui/`; run
`npm run typecheck` there (NOT `npx tsc --noEmit -p .`, which passes even with
real type errors because of the project-references build layout) and `npm
test` / `npm run test:run` as appropriate.

## Deploy

**Embedded-only.** `deploy.ps1` requires an embedded build (the `build.ps1`
default) and refuses to deploy a `-NoEmbedWebui` build, stopping with an error
and touching nothing at the deploy target. There is no supported way to deploy
a non-embedded build; the on-disk `Plugin\NzVideomni\webui\` folder layout from
early development is gone (see `Docs/DEVLOG.md` for the embedded-only-operation
switch).

**`deploy.ps1` writes to two places**, and both matter:

1. **The live AviUtl2 install** (`-PluginDir`), for testing the build by hand.
   AviUtl2 currently runs from a **portable install** (kept off the C: drive for
   disk-space reasons), so its targets sit under a `data\` folder next to
   `aviutl2.exe`. `-PluginDir` defaults to
   `D:\For_Videos\AviUtl2\aviutl2_v2.0.54\data\Plugin`.
2. **The monorepo's distribution copy** (`-DistDir`), i.e.
   `AviUtl2-Plugin\NzVideomni.aux2` — the git-tracked file end users install.
   The default is derived from the repo's own location at run time (the
   monorepo's `AviUtl2-Plugin\` folder, one level above this frontend repo),
   so it works on any clone path without an override; pass `-DistDir`
   explicitly only to send the copy somewhere else.
   **Forgetting this copy means users keep getting the old plugin**, which is
   why deploy.ps1 does it automatically rather than leaving it to a later
   manual step. (Skipped automatically when the build carries a `[PROBE]`
   marker, so instrumented builds never leak into the distribution copy.)

Pass an empty string to either flag to skip that destination.

```powershell
# Deploys the embedded .aux2 (build.ps1's default output) to:
#   <install>\data\Plugin\NzVideomni\NzVideomni.aux2
#   <install>\data\Language\*.NzVideomni.aul2 (from Language\)
#   ..\NzVideomni.aux2  (the monorepo distribution copy)
# Also removes the old single-file <install>\data\Plugin\NzVideomni.aux2
# (avoids a double load), and any stale <install>\data\Plugin\NzVideomni\webui\
# folder left over from an earlier non-embedded deploy.
# Refuses to run while AviUtl2 is open, and refuses a non-embedded .aux2.
powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1 -Config Release
```

(Both defaults are built into `scripts\deploy.ps1`, so the flags are only needed
to override them — e.g. `-PluginDir` for a different AviUtl2 install.)

## Release packaging (embedded single-file build)

```powershell
# Build (embeds the Web UI by default; -EmbedWebui is accepted as a
# redundant, explicit way to say the same thing):
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Config Release
#   -SkipWebuiBuild            embed the existing webui\dist-single\index.html
#   -EmbeddedHtml <path>       embed an alternate single-file HTML

# Bundle the embedded .aux2 + Language files into dist\NzVideomni-<version>.au2pkg.zip:
powershell -ExecutionPolicy Bypass -File scripts\package.ps1 -Config Release
```

The embedded build serves its UI from the DLL, so the `.au2pkg.zip` ships no
`webui\` folder. **Embedded-only**: `package.ps1` stops with an error if the
`.aux2` was not built with the Web UI embedded, since packaging a
non-embedded build would ship a release with no UI (same rationale and check
as `scripts\deploy.ps1`; see its Deploy section above). `-Version` defaults to
the current `kPluginVersion` (1.0.0); pass it explicitly if you have
bumped one without the other.

## Run AviUtl2

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-aviutl.ps1
```

(`scripts\run-aviutl.ps1`'s built-in `-ExePath` default already points at the
v2.0.54 portable `aviutl2.exe`, so pass `-ExePath` only to launch a different
install.)

After launching, the **Nz-Videomni** panel can be shown/docked from the AviUtl2
window menu. Plugin log lines appear in AviUtl2's in-app Log view **and** in
`%LOCALAPPDATA%\NzVideomni\logs\plugin.log`.

> **First-launch trust prompt.** AviUtl2 blocks any newly added native plugin
> until it is approved: on first launch a dialog ("スクリプト・プラグインの追加")
> appears and you must click **このプラグイン・スクリプトを信頼して使用する**.
> The approval is persisted in the portable install's
> `data\module.ini` (e.g.
> `D:\For_Videos\AviUtl2\aviutl2_v2.0.54\data\module.ini`) as
> `[NzVideomni\NzVideomni.aux2] trust=1`. Until approved, `RegisterPlugin` never runs
> and the plugin does not load (no log is written).

> **Right-click menu language.** The timeline's right-click menu (Nz-Videomni's
> "🎬 Video: …" / "🖼 Image: …" / "✨ Insert AI generation here" /
> "📷 Generate a video from the current frame (i2v)" items etc.) is
> translated using **AviUtl2's own 設定 → 言語の設定 (Settings → Language)
> setting**, not the Web UI panel's `LANGUAGE` toggle inside Nz-Videomni — the two
> are independent. Translation happens once, when the plugin registers its
> menu items at AviUtl2 startup, by looking up the matching
> `Language\*.NzVideomni.aul2` file for whichever language AviUtl2 itself is set
> to. If AviUtl2's language setting is left at **Default** (its built-in
> Japanese strings; this does *not* trigger loading a plugin language file),
> the menu falls back to its English keys even on a Japanese-language build of
> AviUtl2. To see the menu in Japanese, go to 設定 → 言語の設定, explicitly
> select **Japanese**, and restart AviUtl2 — the change only takes effect on
> the next startup.

## Further reading

- `Docs/DEVLOG.md` — chronological development log, verification results, and
  known issues/discoveries.
- `Docs/BRIDGE_CONTRACT.md` — full native/Web UI JSON-RPC contract reference.
- `Docs/API_REFERENCE.md` / `Docs/SDK_REFERENCE.md` — backend HTTP API and
  AviUtl2 SDK notes.

Elsewhere in the monorepo:

- `..\..\README.md` — backend setup, startup, `models/` layout, API overview.
- `..\..\Videomni_Backend_Specification.md` — the frozen API contract (§6) and
  the backend's own specification.
- `..\..\Docs\PENDING_TASKS.md` — the project-wide task ledger (backend and
  frontend alike); it is the single entry point for a session.
- `..\..\Docs\HANDOFF_ARCHIVE.md` — append-only archive of past handoff
  entries (the `NEXT_SESSION_HANDOFF.md` note it archived was retired on
  2026-09-01).
