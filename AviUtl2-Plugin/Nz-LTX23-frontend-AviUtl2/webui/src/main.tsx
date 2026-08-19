import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { applyThemeToDocument, readStoredTheme } from './shell/ThemeContext'

// N8 FOUC avoidance: stamp the persisted theme onto <html> synchronously,
// before the first paint, so a stored "light" preference doesn't flash dark
// for one frame while `ThemeProvider`'s own effect (shell/ThemeContext.tsx)
// catches up on mount.
applyThemeToDocument(readStoredTheme())

// Contract v7 (drag-and-drop): WebView2's AllowExternalDrop defaults to true
// (unmodified by this project), so a plain DOM drop over the page already
// fires without any extra native drop-target plumbing. Without this pair of
// listeners, though, a drop that misses every `DropZone` (there are only
// three: Create's reference-video/source-audio slots and Chain's source
// slot) falls through to Chromium's own default handling, which navigates
// the whole WebView to the dropped file and discards the running app. These
// are registered once, at module load, above React entirely, so they can
// never be missed regardless of which mode/screen is mounted.
//
// MINOR fix (post-review): guarded against double-registration across a Vite
// HMR reload of this module (`npm run dev`; never happens in the packaged
// production build, which never hot-reloads) — a naive unconditional
// `addEventListener` pair here would leak one more pair of listeners every
// time this module re-executes. The flag lives on `window` (survives this
// module's own re-execution, unlike a module-scope variable) so a second
// execution skips re-installing; `import.meta.hot.dispose` additionally
// tears the listeners down (and clears the flag) right before a hot reload
// swaps this module out, so a genuinely fresh execution starts clean rather
// than relying on the flag alone.
const DND_LISTENERS_INSTALLED_FLAG = '__nzltx23DndListenersInstalled'
type WindowWithDndGuard = Window & { [DND_LISTENERS_INSTALLED_FLAG]?: boolean }
const windowWithDndGuard = window as WindowWithDndGuard

function preventDefaultDrop(e: DragEvent): void {
  e.preventDefault()
}

if (!windowWithDndGuard[DND_LISTENERS_INSTALLED_FLAG]) {
  windowWithDndGuard[DND_LISTENERS_INSTALLED_FLAG] = true
  window.addEventListener('dragover', preventDefaultDrop)
  window.addEventListener('drop', preventDefaultDrop)
}

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    window.removeEventListener('dragover', preventDefaultDrop)
    window.removeEventListener('drop', preventDefaultDrop)
    delete windowWithDndGuard[DND_LISTENERS_INSTALLED_FLAG]
  })
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
