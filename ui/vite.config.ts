/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  // strictPort: if 5173 is taken, fail rather than move to 5174, because the server's CORS list names 5173 and 4173.
  server: { port: 5173, strictPort: true },
  preview: { port: 4173, strictPort: true },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
    // `vitest --coverage` (make test-ui): the map layer files and the layer panel keep the repo's 90% line floor.
    coverage: {
      provider: 'v8',
      include: ['src/map/**/*.{ts,tsx}', 'src/components/LayerPanel.tsx', 'src/components/Legend.tsx'],
      exclude: ['**/*.test.*', 'src/map/types.ts'],
      reporter: ['text'],
      thresholds: { lines: 90, perFile: true },
    },
  },
})
