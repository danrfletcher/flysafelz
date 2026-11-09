import { useEffect, useRef } from 'react'
import { useMap } from 'react-leaflet'

import { useReducedMotion } from '@/hooks/useReducedMotion'
import type { RainFrame } from '@/map/rainIndex'
import { useRainPane } from '@/map/rainPane'
import { RainRadar } from '@/map/rainRadar'
import type { TileGate } from '@/map/rainScheduler'
import type { RainView } from '@/map/rainView'

export interface RainLayerProps {
  host: string
  frames: readonly RainFrame[]
  /** Changes with each index fetch that worked, so tiles that failed are tried again. */
  fetchedAt: number
  gate: TileGate
  view: RainView
}

/**
 * The radar inside a MapContainer: the frames' tiles in the rain pane, looped with the frame time published to `view`. With reduced motion only the latest
 * frame is made and nothing else is fetched. Mount it only while the Rain switch is on and the index gave frames: unmounting stops the loop timer, drops the
 * queued tile requests and removes every layer, so switching off and on leaves one layer set and one timer. Tiles already loaded are not fetched again
 * (the gate remembers them). The loop stops while the layer is faded out at zoom 9 and over.
 */
export function RainLayer({ host, frames, fetchedAt, gate, view }: RainLayerProps) {
  const map = useMap()
  const reduced = useReducedMotion()
  const hidden = useRainPane(view)
  const radar = useRef<RainRadar | null>(null)

  useEffect(() => {
    const made = new RainRadar(map, gate, (frameTime) => view.set({ frameTime }))
    radar.current = made
    return () => {
      made.destroy()
      radar.current = null
    }
  }, [map, gate, view])

  useEffect(() => {
    radar.current?.update(host, frames, reduced)
  }, [map, gate, view, host, frames, fetchedAt, reduced])

  useEffect(() => {
    radar.current?.setPaused(hidden)
  }, [map, gate, view, hidden])

  return null
}
