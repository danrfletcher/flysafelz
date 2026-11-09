import { useSyncExternalStore } from 'react'

import { formatUpdated } from '@/map/useMapLayers'

/**
 * What the map tells the rest of the page about the Rain layer while it is on: the frame time on screen (unix seconds; null when the layer is off or
 * has no frame) and whether the zoom has faded it out. The layer panel's row and the timestamp on the map read it. It is a small store of its own, so
 * a frame change every second or so re-renders those two and not the whole app.
 */
export interface RainViewState {
  frameTime: number | null
  hidden: boolean
}

export interface RainView {
  get: () => RainViewState
  /** Changes the given fields; listeners hear of it only when something differs. */
  set: (change: Partial<RainViewState>) => void
  subscribe: (listener: () => void) => () => void
}

export function createRainView(): RainView {
  let state: RainViewState = { frameTime: null, hidden: false }
  const listeners = new Set<() => void>()
  return {
    get: () => state,
    set: (change) => {
      const next = { ...state, ...change }
      if (next.frameTime === state.frameTime && next.hidden === state.hidden) return
      state = next
      listeners.forEach((listener) => listener())
    },
    subscribe: (listener) => {
      listeners.add(listener)
      return () => void listeners.delete(listener)
    },
  }
}

export function useRainView(view: RainView): RainViewState {
  return useSyncExternalStore(view.subscribe, view.get)
}

/** "HH:MM" of a frame time (unix seconds) in the browser's time zone. */
export function formatFrameTime(seconds: number): string {
  return formatUpdated(new Date(seconds * 1000).toISOString())
}
