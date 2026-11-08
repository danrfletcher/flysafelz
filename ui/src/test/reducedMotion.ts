import { act } from '@testing-library/react'
import { vi } from 'vitest'

/** A matchMedia that answers `prefers-reduced-motion: reduce` as told (and `min-width` from the window width), and fires `change` to every listener when `set` is called. */
export function mockReducedMotion(initial: boolean) {
  const listeners = new Set<(event: MediaQueryListEvent) => void>()
  let reduced = initial
  vi.stubGlobal('matchMedia', (query: string) => ({
    get matches() {
      if (query.includes('prefers-reduced-motion')) return reduced
      // the chat's width query (useIsDesktop) is answered from the window width, as a browser would
      const min = /min-width: (\d+)px/.exec(query)
      return min ? window.innerWidth >= Number(min[1]) : false
    },
    media: query,
    addEventListener: (_type: string, listener: (event: MediaQueryListEvent) => void) => listeners.add(listener),
    removeEventListener: (_type: string, listener: (event: MediaQueryListEvent) => void) => listeners.delete(listener),
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
    onchange: null,
  }))
  return {
    set(value: boolean) {
      reduced = value
      act(() => listeners.forEach((listener) => listener({ matches: value } as MediaQueryListEvent)))
    },
    listeners: () => listeners.size,
  }
}
