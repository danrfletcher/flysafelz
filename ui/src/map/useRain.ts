import { useEffect, useState } from 'react'

import { fetchRoute, REFETCH_MS } from '@/map/useMapLayers'
import { fetchRainIndex, type RainFrame } from '@/map/rainIndex'
import { RAIN_GRID_PATH, readRainGridPayload, type RainCell } from '@/map/rainGrid'
import { framesInWindow, mergeFrames, TileGate } from '@/map/rainScheduler'
import { createRainView, type RainView } from '@/map/rainView'

/**
 * Where the Rain layer's data comes from (G5, E3):
 *   loading      nothing has come back yet
 *   radar        RainViewer's index gave frames in the 2 hour window
 *   fallback     the index failed and /map/rain-grid gave Open-Meteo's cells (`stale` is the route's own flag, or true when a later refresh failed)
 *   unavailable  both failed: the layer is hidden and the panel says so
 */
export type RainState =
  | { mode: 'loading' }
  | { mode: 'radar'; host: string; frames: RainFrame[]; fetchedAt: number }
  | { mode: 'fallback'; cells: RainCell[]; updatedAt: string; stale: boolean }
  | { mode: 'unavailable' }

export interface Rain {
  state: RainState
  /** Every radar tile request goes through this: throttled, and made once per URL for the page's life. */
  gate: TileGate
  /** What the map says of the layer while it is on: the frame on screen and whether the zoom has hidden it. */
  view: RainView
}

const sameFrames = (a: readonly RainFrame[], b: readonly RainFrame[]) => a.length === b.length && a.every((frame, i) => frame.time === b[i].time)

/**
 * The Rain layer's data. On load and every 15 minutes while the page is open it fetches RainViewer's index; new frames are appended and frames
 * older than 2 hours dropped. If the index fails and no frame in the window is held, it asks `/map/rain-grid` (the only caller of that route); if that fails
 * too, the state is `unavailable`. A later failure of the index keeps the radar already held while its frames last. The browser never calls Open-Meteo.
 * The fetch runs whether or not the Rain switch is on, so switching it on draws what is held.
 */
export function useRain(): Rain {
  const [state, setState] = useState<RainState>({ mode: 'loading' })
  const [gate] = useState(() => new TileGate())
  const [view] = useState(createRainView)

  useEffect(() => {
    let held: RainState = { mode: 'loading' }
    let latest = 0
    let controller: AbortController | null = null
    const commit = (next: RainState) => {
      held = next
      setState(next)
    }

    const load = async () => {
      controller?.abort() // a slow earlier fetch must not overwrite a newer answer
      const mine = new AbortController()
      controller = mine
      const run = ++latest
      const current = () => run === latest && !mine.signal.aborted

      const index = await fetchRainIndex(mine.signal)
      if (!current()) return
      const now = Date.now()
      const heldFrames = held.mode === 'radar' ? held.frames : []
      if (index.status === 'ok') {
        const { frames } = mergeFrames(heldFrames, index.frames, now)
        if (frames.length > 0) {
          commit({ mode: 'radar', host: index.host, frames: sameFrames(frames, heldFrames) ? heldFrames : frames, fetchedAt: now })
          return
        }
      }
      // The index failed (or has nothing inside the window). Radar already held stays while any of its frames is in the window.
      if (held.mode === 'radar') {
        const kept = framesInWindow(held.frames, now)
        if (kept.length > 0) {
          commit({ ...held, frames: sameFrames(kept, held.frames) ? held.frames : kept })
          return
        }
      }
      const grid = await fetchRoute(RAIN_GRID_PATH, readRainGridPayload, mine.signal)
      if (!current()) return
      if (grid) commit({ mode: 'fallback', cells: grid.grid, updatedAt: grid.updated_at, stale: grid.stale })
      else if (held.mode === 'fallback') commit({ ...held, stale: true })
      else commit({ mode: 'unavailable' })
    }

    void load()
    const timer = setInterval(() => void load(), REFETCH_MS)
    return () => {
      clearInterval(timer)
      controller?.abort()
      latest += 1
    }
  }, [])

  return { state, gate, view }
}
