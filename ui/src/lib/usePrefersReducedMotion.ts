import { useSyncExternalStore } from 'react'

const QUERY = '(prefers-reduced-motion: reduce)'

function subscribe(onChange: () => void): () => void {
  const query = window.matchMedia?.(QUERY)
  query?.addEventListener('change', onChange)
  return () => query?.removeEventListener('change', onChange)
}

const prefersReduced = () => window.matchMedia?.(QUERY).matches ?? false

/** True while the visitor's system asks for reduced motion. It follows a change of that setting live, with no reload. A browser with no matchMedia reads as false. */
export function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(subscribe, prefersReduced, () => false)
}
