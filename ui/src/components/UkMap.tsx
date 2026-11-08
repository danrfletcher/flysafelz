import 'leaflet/dist/leaflet.css'

import { MapContainer, TileLayer } from 'react-leaflet'

import { UK_BOUNDS } from '@/lib/ukBounds'

// Keyless CARTO Dark Matter tiles. {r} asks for retina tiles on high-density screens.
export const TILE_URL = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
export const ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'

/**
 * The full-screen map, opened on the whole UK. It fills whatever space its parent gives it; Leaflet's own window-resize
 * handling (trackResize, on by default) calls invalidateSize when that space changes.
 */
export function UkMap() {
  return (
    <MapContainer bounds={UK_BOUNDS} zoomSnap={0.25} className="size-full" aria-label="Map of the United Kingdom">
      <TileLayer url={TILE_URL} attribution={ATTRIBUTION} subdomains="abcd" maxZoom={20} />
    </MapContainer>
  )
}
