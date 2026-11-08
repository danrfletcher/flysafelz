import L from 'leaflet'
import 'leaflet-velocity'

import { themeColor } from '@/map/airspaceStyle'
import { windBand, type WindBand, type WindGrid } from '@/map/wind'

// leaflet-velocity adds `L.velocityLayer` to Leaflet's global `L` when it loads. This is the one file that imports it.

/** The speed (m/s) at which the particle colours reach their last step. */
export const MAX_VELOCITY = 12

interface VelocityOptions {
  data: WindGrid
  displayValues: boolean
  maxVelocity: number
  colorScale: string[]
  lineWidth: number
  particleMultiplier: number
}

// The parts of leaflet-velocity this file reaches into. They are private to the library, so each use is on one line below, and package.json pins
// the library to the exact version these were read from (src/fence.test.ts checks there is no range), so an upgrade is a deliberate step: read this file against the new code first.
interface CanvasLayerPrototype {
  drawLayer: (this: { _map: L.Map | null; _frame: number | null }) => void
  _onLayerDidMove: (this: { _map: L.Map | null }) => void
}

interface VelocityInternals {
  _windy: { stop: () => void } | null
  _clearAndRestart: () => void
  _clearWind: () => void
}

interface VelocityClass {
  prototype: { onRemove: (this: unknown, map: L.Map) => void }
  extend: (members: object) => new (options: VelocityOptions) => L.Layer
}

const library = L as unknown as { VelocityLayer: VelocityClass; CanvasLayer: { prototype: CanvasLayerPrototype } }

// leaflet-velocity's canvas layer asks for an animation frame and a zero-delay timer when it is added, and never cancels either. Remove the layer
// in that moment (switch Wind off at once, or swap to arrows) and they still run, on a map the layer has left: "Cannot read properties of null
// (reading 'getSize')" in the console. A frame cannot be cancelled by the library's own handle (its first draw clears that handle while the frame
// is still waiting), so the two callbacks themselves do nothing for a layer that is on no map. Done once, here, where the library is loaded.
const canvasLayer = library.CanvasLayer.prototype
const draw = canvasLayer.drawLayer
const move = canvasLayer._onLayerDidMove
canvasLayer.drawLayer = function guardedDraw() {
  if (this._map) draw.call(this)
  else this._frame = null
}
canvasLayer._onLayerDidMove = function guardedMove() {
  if (this._map) move.call(this)
}

/**
 * leaflet-velocity's layer with a clean removal. Its own `onRemove` stops the particle loop, clears its timer and takes the canvas away, but
 * leaves on the map the five listeners it added when it first drew (stopping and restarting the particles on a drag, a zoom or a resize).
 * This removal takes them off, so after `remove()` nothing of the layer is left on the map.
 */
const CleanVelocityLayer = library.VelocityLayer.extend({
  onRemove(this: L.Layer & VelocityInternals, map: L.Map) {
    const windy = this._windy
    library.VelocityLayer.prototype.onRemove.call(this, map)
    if (windy) {
      map.off('dragstart', windy.stop)
      map.off('zoomstart', windy.stop)
    }
    map.off('dragend', this._clearAndRestart)
    map.off('zoomend', this._clearAndRestart)
    map.off('resize', this._clearWind)
  },
})

const TOKEN: Record<WindBand, '--wind-light' | '--wind-moderate' | '--wind-strong' | '--wind-severe'> = {
  light: '--wind-light',
  moderate: '--wind-moderate',
  strong: '--wind-strong',
  severe: '--wind-severe',
}

/** One colour for each whole m/s from 0 up to MAX_VELOCITY, from the theme's wind bands, so a particle is coloured like the legend's band for its speed. */
export function particleColours(): string[] {
  return Array.from({ length: MAX_VELOCITY + 1 }, (_, speed) => themeColor(TOKEN[windBand(speed)]))
}

/**
 * A particle layer for the grid: animated by leaflet-velocity on its own canvas. The caller adds it to the map and removes it; removing it
 * stops the animation loop and takes the canvas away. It shows no speed read-out under the pointer (displayValues off), so it adds no control
 * to the page. The canvas takes no pointer events (src/styles/index.css), so clicks and hovers reach the map and the shapes under it.
 */
export function createVelocityLayer(grid: WindGrid): L.Layer {
  return new CleanVelocityLayer({
    data: grid,
    displayValues: false,
    maxVelocity: MAX_VELOCITY,
    colorScale: particleColours(),
    lineWidth: 1.25,
    particleMultiplier: 1 / 400,
  })
}
