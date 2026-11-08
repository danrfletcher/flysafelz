import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { setupServer } from 'msw/node'
import { afterAll, afterEach, beforeAll, beforeEach, expect, vi } from 'vitest'

// No handlers by default: a test that makes a request must say which (src/test/handlers.ts has the map routes). onUnhandledRequest
// 'error' makes MSW refuse any other request, and the list below makes the test that caused it fail, even when the code swallows the refusal.
export const server = setupServer()
let unhandled: string[] = []

// jsdom does not load stylesheets (css: false), so the theme's variables are put on the page from theme.css itself: the map reads
// its outline colours from them, and a test sees the same values the browser does.
const THEME = readFileSync(resolve(__dirname, '../styles/theme.css'), 'utf8')
if (typeof document !== 'undefined') {
  for (const [, name, value] of THEME.matchAll(/^\s*(--[\w-]+):\s*(#[0-9a-fA-F]{3,8})\s*;/gm)) {
    document.documentElement.style.setProperty(name, value)
  }
}

// jsdom has no ResizeObserver; Radix (Popover, Switch) measures with one.
globalThis.ResizeObserver ??= class {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeAll(() => {
  server.events.on('request:unhandled', ({ request }) => {
    unhandled.push(`${request.method} ${request.url}`)
  })
  server.listen({ onUnhandledRequest: 'error' })
})
beforeEach(() => {
  unhandled = []
  // F4: tests need no CARTO key. A key in a developer's ui/.env must not reach them either, so each test starts keyless (a test that wants a key stubs it).
  vi.stubEnv('VITE_CARTO_KEY', '')
})
afterEach(() => {
  cleanup()
  server.resetHandlers()
  expect(unhandled, 'requests made with no MSW handler').toEqual([])
})
afterAll(() => server.close())
