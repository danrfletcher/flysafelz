import { useSyncExternalStore } from 'react'

/** The width from which the chat is a side sheet; below it, a bottom drawer. */
export const DESKTOP_MIN_WIDTH = 768
const QUERY = `(min-width: ${DESKTOP_MIN_WIDTH}px)`

/** The media query, or null where there is none: no matchMedia (jsdom, an old browser) or one that throws. */
function query(): MediaQueryList | null {
  try {
    return window.matchMedia?.(QUERY) ?? null
  } catch {
    return null
  }
}

function subscribe(onChange: () => void): () => void {
  const list = query()
  list?.addEventListener?.('change', onChange)
  window.addEventListener('resize', onChange)
  return () => {
    list?.removeEventListener?.('change', onChange)
    window.removeEventListener('resize', onChange)
  }
}

// where the query cannot be read, the window width gives the same answer
const isDesktop = () => query()?.matches ?? window.innerWidth >= DESKTOP_MIN_WIDTH

/** True at 768 px and wider. */
export function useIsDesktop(): boolean {
  return useSyncExternalStore(subscribe, isDesktop, () => true)
}
