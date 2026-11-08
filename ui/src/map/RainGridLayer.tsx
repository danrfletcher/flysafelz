import L from 'leaflet'
import { useEffect } from 'react'
import { useMap } from 'react-leaflet'

import { themeColor } from '@/map/airspaceStyle'
import { CELL_DEGREES, cellOpacity, DRY_MM, type RainCell } from '@/map/rainGrid'
import { RAIN_PANE, useRainPane } from '@/map/rainPane'
import type { RainView } from '@/map/rainView'

/**
 * The fallback: Open-Meteo's precipitation as soft squares, one per 1 degree cell that has rain, in the rain pane with the same zoom fade as the radar. Display only: a square takes no
 * pointer events and has no tooltip, so nothing suggests it is radar or gives a value. The pane is blurred (src/styles/index.css) to make the squares soft.
 * Mount it only while the Rain switch is on and the grid is held; unmounting removes the squares.
 */
export function RainGridLayer({ cells, view }: { cells: readonly RainCell[]; view: RainView }) {
  const map = useMap()
  useRainPane(view, true)

  useEffect(() => {
    const renderer = L.svg({ pane: RAIN_PANE, padding: 0.5 })
    const color = themeColor('--rain')
    const half = CELL_DEGREES / 2
    const group = L.layerGroup(
      cells
        .filter((cell) => cell.precipitation >= DRY_MM)
        .map((cell) =>
          L.rectangle(
            [
              [cell.lat - half, cell.lon - half],
              [cell.lat + half, cell.lon + half],
            ],
            { renderer, stroke: false, fill: true, fillColor: color, fillOpacity: cellOpacity(cell.precipitation), interactive: false },
          ),
        ),
    )
    group.addTo(map)
    return () => {
      group.remove()
      renderer.remove()
    }
  }, [map, cells])

  return null
}
