import type L from 'leaflet'
import { useEffect, useState } from 'react'
import { TileLayer, useMap } from 'react-leaflet'

import { ATTRIBUTION, basemapStatus, INITIAL_BASEMAP, tileUrl, type BasemapEvent, type BasemapStatus } from '@/map/basemap'

export interface BasemapLayerProps {
  /** The status the app holds, which this layer reports to through onStatus. */
  status: BasemapStatus
  onStatus: (status: BasemapStatus) => void
}

/**
 * The CARTO Dark Matter tile layer. It feeds Leaflet's tile events to the status rule and reports only a change of status, so the
 * hundreds of tile loads do not each re-render the app. While the tiles are unavailable the layer is hidden (opacity 0) but stays on
 * the map, so the attribution stays and the next tile that loads brings it back, and the map container gets `basemap-unavailable`
 * (react-leaflet reads a MapContainer's className once, so the class is set on the container element here). Nothing is retried and no other provider is tried:
 * Leaflet's own tile requests for each view are all the browser makes.
 */
export function BasemapLayer({ status, onStatus }: BasemapLayerProps) {
  const map = useMap()
  const [layer, setLayer] = useState<L.TileLayer | null>(null)

  useEffect(() => {
    map.getContainer().classList.toggle('basemap-unavailable', status === 'unavailable')
  }, [map, status])

  // The batch counts live in this closure, not in React state: only a change of status leaves it.
  useEffect(() => {
    if (!layer) return
    let state = INITIAL_BASEMAP
    const feed = (event: BasemapEvent) => () => {
      const before = state.status
      state = basemapStatus(state, event)
      if (state.status !== before) onStatus(state.status)
    }
    const handlers = { tileload: feed('tileload'), tileerror: feed('tileerror'), load: feed('load') }
    layer.on(handlers)
    return () => void layer.off(handlers)
  }, [layer, onStatus])

  return <TileLayer ref={setLayer} url={tileUrl()} attribution={ATTRIBUTION} subdomains="abcd" maxZoom={20} opacity={status === 'unavailable' ? 0 : 1} />
}
