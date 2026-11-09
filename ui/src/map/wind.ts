// GET /map/wind: the shape of the route's answer, the reader that checks it, and the arrows drawn from it when motion is reduced (G4, E12).
//
// The grid is leaflet-velocity's own: two records, U (parameterCategory 2, parameterNumber 2) then V (2, 3), each with a header and `data`
// in rows from the north, each row from the west (src/flysafelz/mapdata/wind.py). u is the wind towards the east and v towards the north, in m/s.

export const WIND_PATH = '/map/wind'

export interface WindHeader {
  parameterCategory: number
  parameterNumber: number
  /** Columns and rows. */
  nx: number
  ny: number
  /** The first column's longitude and the first row's latitude (the north-west corner), then the last (the south-east corner). */
  lo1: number
  la1: number
  lo2: number
  la2: number
  /** The step between columns and between rows, in degrees. */
  dx: number
  dy: number
  refTime: string
}

export interface WindRecord {
  header: WindHeader
  data: number[]
}

/** The U record, then the V record. */
export type WindGrid = [WindRecord, WindRecord]

export interface WindPayload {
  grid: WindGrid
  updated_at: string
  stale: boolean
}

const HEADER_NUMBERS = ['parameterCategory', 'parameterNumber', 'nx', 'ny', 'lo1', 'la1', 'lo2', 'la2', 'dx', 'dy'] as const
const MAX_POINTS = 10_000 // the route sends 143; a body claiming far more is not this route's

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)

/** One record, or null: a header with its numbers and a refTime, and `data` holding exactly nx × ny finite numbers. */
function readRecord(value: unknown, parameterNumber: number): WindRecord | null {
  if (!isObject(value) || !isObject(value.header) || !Array.isArray(value.data)) return null
  const header = value.header
  if (!HEADER_NUMBERS.every((name) => isFiniteNumber(header[name])) || typeof header.refTime !== 'string') return null
  const { nx, ny, dx, dy } = header as unknown as WindHeader
  if (header.parameterCategory !== 2 || header.parameterNumber !== parameterNumber) return null
  if (!Number.isInteger(nx) || !Number.isInteger(ny) || nx < 1 || ny < 1 || nx * ny > MAX_POINTS || dx <= 0 || dy <= 0) return null
  if (value.data.length !== nx * ny || !value.data.every(isFiniteNumber)) return null
  return { header: header as unknown as WindHeader, data: value.data as number[] }
}

/**
 * The payload in a response body, or null when it is not one: the grid must be a U record then a V record of the same size, and `updated_at`
 * and `stale` must be there as the other routes have them. A grid that is short or has a non-number in it is refused whole, never drawn in part.
 */
export function readWindPayload(body: unknown): WindPayload | null {
  if (!isObject(body) || !Array.isArray(body.grid) || body.grid.length !== 2) return null
  if (typeof body.updated_at !== 'string' || Number.isNaN(Date.parse(body.updated_at)) || typeof body.stale !== 'boolean') return null
  const u = readRecord(body.grid[0], 2)
  const v = readRecord(body.grid[1], 3)
  if (!u || !v) return null
  const same = HEADER_NUMBERS.slice(2).every((name) => u.header[name] === v.header[name])
  return same ? { grid: [u, v], updated_at: body.updated_at, stale: body.stale } : null
}

/** One wind arrow: where it is, how fast the air moves (m/s), and the compass bearing it moves towards (degrees clockwise from north). */
export interface WindArrow {
  lat: number
  lon: number
  speed: number
  bearing: number
}

/**
 * One arrow per grid point, rows from the north and each row from the west, pointing downwind: the bearing is the way the air goes
 * (atan2 of u over v), which is 180 degrees from the direction the wind is reported to blow from. Calm air has speed 0 and bearing 0.
 */
export function toArrows({ grid: [u, v] }: Pick<WindPayload, 'grid'>): WindArrow[] {
  const { nx, ny, lo1, la1, dx, dy } = u.header
  const arrows: WindArrow[] = []
  for (let row = 0; row < ny; row += 1) {
    for (let column = 0; column < nx; column += 1) {
      const east = u.data[row * nx + column]
      const north = v.data[row * nx + column]
      const speed = Math.hypot(east, north)
      const bearing = speed === 0 ? 0 : (((Math.atan2(east, north) * 180) / Math.PI) + 360) % 360
      arrows.push({ lat: la1 - row * dy, lon: lo1 + column * dx, speed, bearing })
    }
  }
  return arrows
}

/** Speed bands in m/s, lowest first. Each has a theme colour (--wind-<name>) and a legend line, so the colour is never the only signal. */
export const WIND_BANDS = [
  { name: 'light', below: 4, label: 'Wind under 4 m/s' },
  { name: 'moderate', below: 8, label: 'Wind 4 to 8 m/s' },
  { name: 'strong', below: 12, label: 'Wind 8 to 12 m/s' },
  { name: 'severe', below: Infinity, label: 'Wind over 12 m/s' },
] as const

export type WindBand = (typeof WIND_BANDS)[number]['name']

export function windBand(speed: number): WindBand {
  return WIND_BANDS.find((band) => speed < band.below)?.name ?? 'severe'
}

/** Air slower than this is drawn as a dot, not an arrow: its direction is not worth showing. */
export const CALM_MS = 0.3

/** The arrow's length in px: longer for faster air, between 14 and 44. */
export function arrowLength(speed: number): number {
  return Math.round(Math.min(44, 14 + speed * 2.5))
}
