import { useEffect } from 'react'
import { useMap } from 'react-leaflet'

import { fitPadding, lzBounds, type ChatLayout } from '@/map/fitLz'
import type { LatLon } from '@/pick/whatsHere'

export interface FitToLzProps {
  /** The LZ the chat is open for, or null when it is closed (the map is then left as it is). */
  point: LatLon | null
  layout: ChatLayout
}

/** Inside the map: fits the 2 NM circle round the LZ when the chat opens, when it moves to another LZ and when the layout changes. Renders nothing. */
export function FitToLz({ point, layout }: FitToLzProps) {
  const map = useMap()
  const lat = point?.lat
  const lon = point?.lon

  useEffect(() => {
    if (lat === undefined || lon === undefined) return
    map.fitBounds(lzBounds({ lat, lon }), fitPadding(layout, map.getSize()))
  }, [map, lat, lon, layout])

  return null
}
