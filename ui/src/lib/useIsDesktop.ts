import { useSyncExternalStore } from 'react'

/** The width from which the chat is a side sheet; below it, a bottom drawer. */
export const DESKTOP_MIN_WIDTH = 768
const QUERY = `(min-width: ${DESKTOP_MIN_WIDTH}px)`

function subscribe(onChange: () => void): () => void {
  const query = window.matchMedia?.(QUERY)
  query?.addEventListener('change', onChange)
  window.addEventListener('resize', onChange)
  return () => {
    query?.removeEventListener('change', onChange)
    window.removeEventListener('resize', onChange)
  }
}

const isDesktop = () => (window.matchMedia ? window.matchMedia(QUERY).matches : window.innerWidth >= DESKTOP_MIN_WIDTH)

/** True at 768 px and wider. */
export function useIsDesktop(): boolean {
  return useSyncExternalStore(subscribe, isDesktop, () => true)
}
