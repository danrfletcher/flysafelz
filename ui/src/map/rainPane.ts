import type L from 'leaflet'
import { useEffect, useState } from 'react'
import { useMap, useMapEvents } from 'react-leaflet'

import { rainHidden, rainOpacity } from '@/map/rainFade'
import type { RainView } from '@/map/rainView'

export const RAIN_PANE = 'rain'

// Leaflet's tile pane is 200 and its overlay pane (every vector shape: airspace, NOTAMs, an answer's highlights) 400. Rain sits between them: just over the
// basemap, under every outline.
const RAIN_Z_INDEX = '250'

/** The rain pane, made on first use. It takes no pointer events, so a click or hover goes through to the map and the shapes. */
export function ensureRainPane(map: L.Map): HTMLElement {
  const pane = map.getPane(RAIN_PANE) ?? map.createPane(RAIN_PANE)
  pane.style.zIndex = RAIN_Z_INDEX
  pane.style.pointerEvents = 'none'
  return pane
}

/**
 * Makes the rain pane and keeps its opacity on the zoom fade (60% to zoom 7, down to nothing at zoom 9; fractional zooms in between), the one place the
 * fade is applied, so the radar and the fallback cells share it and never stack. It tells `view` whether the layer is faded out, and returns that.
 * `model` marks the pane as the soft Open-Meteo cells. Call it before adding anything to the pane.
 */
export function useRainPane(view: RainView, model = false): boolean {
  const map = useMap()
  const [zoom, setZoom] = useState(() => map.getZoom())
  useMapEvents({ zoom: () => setZoom(map.getZoom()), zoomend: () => setZoom(map.getZoom()) })
  const hidden = rainHidden(zoom)

  useEffect(() => {
    ensureRainPane(map).style.opacity = String(rainOpacity(zoom))
  }, [map, zoom])

  useEffect(() => {
    view.set({ hidden })
    return () => view.set({ hidden: false })
  }, [view, hidden])

  useEffect(() => {
    const pane = ensureRainPane(map)
    pane.classList.toggle('rain-pane--model', model)
    return () => pane.classList.remove('rain-pane--model')
  }, [map, model])

  return hidden
}
