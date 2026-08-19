import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
//
// `base: './'` keeps every emitted asset reference relative so the build
// output works when served from a WebView2 virtual host mapping
// (https://app.nzvideomni.local/index.html -> webui/dist), which has no
// concept of an absolute site root the way a normal HTTP server does.
export default defineConfig({
  base: './',
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    css: true,
    setupFiles: ['./src/test/setupTests.ts'],
  },
})
