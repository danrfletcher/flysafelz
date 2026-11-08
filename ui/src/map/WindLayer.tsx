import L from 'leaflet'
import { useEffect } from 'react'
import { useMap } from 'react-leaflet'

import { useReducedMotion } from '@/hooks/useReducedMotion'
import { createVelocityLayer } from '@/map/velocity'
import { arrowLength, CALM_MS, toArrows, windBand, type WindArrow, type WindGrid } from '@/map/wind'

// An arrow, drawn pointing up (north) and turned to its bearing. Stroke and fill take the band's colour from the CSS class (src/styles/index.css).
function arrowHtml(arrow: WindArrow): string {
  const length = arrowLength(arrow.speed)
  const half = length / 2
  return (
    `<svg width="${length}" height="${length}" viewBox="0 0 ${length} ${length}" style="transform: rotate(${Math.round(arrow.bearing)}deg)" aria-hidden="true">` +
    `<line x1="${half}" y1="${length - 2}" x2="${half}" y2="3" />` +
    `<path d="M${half - 4} 9 L${half} 2 L${half + 4} 9 Z" />` +
    `</svg>`
  )
}

const CALM_HTML = '<svg width="8" height="8" viewBox="0 0 8 8" aria-hidden="true"><circle cx="4" cy="4" r="2.5" /></svg>'

/**
 * One non-interactive marker per grid point, pointing downwind, longer and in the band's colour for a faster wind; calm air is a dot.
 * Static: nothing animates, and a marker takes no pointer events, so a click or hover goes through to the map.
 */
export function buildArrowGroup(grid: WindGrid): L.LayerGroup {
  const group = L.layerGroup()
  for (const arrow of toArrows({ grid })) {
    const calm = arrow.speed < CALM_MS
    const size = calm ? 8 : arrowLength(arrow.speed)
    const icon = L.divIcon({
      className: `wind-arrow wind-arrow--${windBand(arrow.speed)}`,
      html: calm ? CALM_HTML : arrowHtml(arrow),
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2],
    })
    group.addLayer(L.marker([arrow.lat, arrow.lon], { icon, interactive: false, keyboard: false }))
  }
  return group
}

/** Adds the layer `make` builds from the grid to the map, and removes it (with its canvas and animation loop) when the grid changes or the component unmounts. */
function useMapLayer(make: (grid: WindGrid) => L.Layer, grid: WindGrid) {
  const map = useMap()
  useEffect(() => {
    const added = make(grid)
    added.addTo(map)
    return () => {
      added.remove()
    }
  }, [map, make, grid])
}

function Particles({ grid }: { grid: WindGrid }) {
  useMapLayer(createVelocityLayer, grid)
  return null
}

function Arrows({ grid }: { grid: WindGrid }) {
  useMapLayer(buildArrowGroup, grid)
  return null
}

/**
 * The Wind layer inside a MapContainer: leaflet-velocity particles from the grid, or, while the visitor's system asks for reduced motion, static
 * arrows from the same grid with no particles and no animation loop. The two swap live when that setting changes. Mount it only while the
 * switch is on and a grid is held: unmounting removes the layer, so switching on and off leaves one layer on the map, or none.
 */
export function WindLayer({ grid }: { grid: WindGrid }) {
  const reduced = useReducedMotion()
  return reduced ? <Arrows grid={grid} /> : <Particles grid={grid} />
}
