import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { setupServer } from 'msw/node'
import { afterAll, afterEach, beforeAll, beforeEach, expect } from 'vitest'

// No handlers: nothing in this PR may make a request. onUnhandledRequest 'error' makes MSW refuse the request, and the
// list below makes the test that caused it fail, even when the code under test swallows the refusal.
export const server = setupServer()
let unhandled: string[] = []

beforeAll(() => {
  server.events.on('request:unhandled', ({ request }) => {
    unhandled.push(`${request.method} ${request.url}`)
  })
  server.listen({ onUnhandledRequest: 'error' })
})
beforeEach(() => {
  unhandled = []
})
afterEach(() => {
  cleanup()
  server.resetHandlers()
  expect(unhandled, 'requests made with no MSW handler').toEqual([])
})
afterAll(() => server.close())
