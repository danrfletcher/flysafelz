import { useSyncExternalStore } from 'react'

const QUERY = '(prefers-reduced-motion: reduce)'

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
  return () => list?.removeEventListener?.('change', onChange)
}

const prefersReduced = () => query()?.matches ?? false

/**
 * The one place the app reads the system's reduced-motion setting: the wind layer, the rain layer and every enter animation (src/lib/motion.ts)
 * take it from here, and App hands it to `MotionConfig`. It follows a change of the setting live, with no reload. Where the setting cannot be
 * read it is false: full motion.
 */
export function useReducedMotion(): boolean {
  return useSyncExternalStore(subscribe, prefersReduced, () => false)
}
