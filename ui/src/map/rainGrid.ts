// GET /map/rain-grid: Open-Meteo's current precipitation on the wind route's 1 degree UK grid, the Rain layer's fallback when RainViewer's index fails
// (src/flysafelz/mapdata/rain_grid.py). The cells are a model's guess, not radar, and are drawn as soft squares with no values shown.

export const RAIN_GRID_PATH = '/map/rain-grid'

/** One 1 degree cell, centred on lat/lon, with the precipitation in mm. Cells the server had no value for are not in the list. */
export interface RainCell {
  lat: number
  lon: number
  precipitation: number
}

export interface RainGridPayload {
  grid: RainCell[]
  updated_at: string
  stale: boolean
}

/** The cell edge in degrees. */
export const CELL_DEGREES = 1

/** Precipitation under this (mm) is dry: no cell is drawn for it. */
export const DRY_MM = 0.05

const MAX_CELLS = 10_000

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)

/** The payload in a response body, or null when it is not one. A cell that is not {lat, lon, precipitation >= 0} is left out, never drawn as zero. */
export function readRainGridPayload(body: unknown): RainGridPayload | null {
  if (!isObject(body) || !Array.isArray(body.grid) || body.grid.length > MAX_CELLS) return null
  if (typeof body.updated_at !== 'string' || Number.isNaN(Date.parse(body.updated_at)) || typeof body.stale !== 'boolean') return null
  const grid = body.grid.filter(
    (cell): cell is RainCell =>
      isObject(cell) && isFiniteNumber(cell.lat) && isFiniteNumber(cell.lon) && isFiniteNumber(cell.precipitation) && cell.precipitation >= 0,
  )
  return { grid, updated_at: body.updated_at, stale: body.stale }
}

/** How strongly a cell is filled: from 0.25 for a trace to 0.75 from 4 mm up. Before the pane's zoom fade. */
export function cellOpacity(precipitation: number): number {
  return 0.25 + (Math.min(precipitation, 4) / 4) * 0.5
}
