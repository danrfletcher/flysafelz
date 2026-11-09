import 'leaflet/dist/leaflet.css'

import type { ReactNode } from 'react'
import { MapContainer } from 'react-leaflet'

import { UK_BOUNDS } from '@/lib/ukBounds'
import { BasemapLayer } from '@/map/BasemapLayer'
import type { BasemapStatus } from '@/map/basemap'

export interface UkMapProps {
  basemap: BasemapStatus
  onBasemapStatus: (status: BasemapStatus) => void
  children?: ReactNode
}

/**
 * The full-screen map, opened on the whole UK. It fills whatever space its parent gives it; Leaflet's own window-resize
 * handling (trackResize, on by default) calls invalidateSize when that space changes. Layers go in as children.
 * The basemap layer reports its status up and, while the tiles are unavailable, marks the container so the theme's map colour shows.
 */
export function UkMap({ basemap, onBasemapStatus, children }: UkMapProps) {
  return (
    <MapContainer bounds={UK_BOUNDS} zoomSnap={0.25} className="size-full" aria-label="Map of the United Kingdom">
      <BasemapLayer status={basemap} onStatus={onBasemapStatus} />
      {children}
    </MapContainer>
  )
}
