import { describe, expect, it } from 'vitest'

import { arrowLength, CALM_MS, readWindPayload, toArrows, WIND_BANDS, windBand, type WindGrid, type WindPayload } from '@/map/wind'
import { WIND_OK } from '@/mocks/wind'

// A hand-built 2 × 2 grid, north-west corner (51 N, 1 W), 1° apart. Rows run north to south and each row west to east:
//   (51, -1)  (51, 0)
//   (50, -1)  (50, 0)
// u is the wind towards the east and v towards the north, in m/s.
const header = (parameterNumber: number) => ({ parameterCategory: 2, parameterNumber, nx: 2, ny: 2, lo1: -1, la1: 51, lo2: 0, la2: 50, dx: 1, dy: 1, refTime: '2026-11-08T12:00:00Z' })
const GRID: WindGrid = [
  { header: header(2), data: [0, -10, 0, 3] }, // u
  { header: header(3), data: [-10, 0, 10, 4] }, // v
]
const payload = (over: Partial<WindPayload> = {}): unknown => ({ grid: GRID, updated_at: '2026-11-08T12:00:00Z', stale: false, ...over })

describe('toArrows', () => {
  it('gives one arrow per grid point, rows from the north and each row from the west, with the right position', () => {
    const arrows = toArrows({ grid: GRID })
    expect(arrows).toHaveLength(4)
    expect(arrows.map((a) => [a.lat, a.lon])).toEqual([
      [51, -1],
      [51, 0],
      [50, -1],
      [50, 0],
    ])
  })

  it('points each arrow downwind: the bearing is the way the air goes, not the way it comes from', () => {
    const [toSouth, toWest, toNorth, toNorthEast] = toArrows({ grid: GRID })
    expect(toSouth.bearing).toBeCloseTo(180) // u 0, v -10: a north wind blows south
    expect(toWest.bearing).toBeCloseTo(270) // u -10, v 0: an east wind blows west
    expect(toNorth.bearing).toBeCloseTo(0) // u 0, v 10: a south wind blows north
    expect(toNorthEast.bearing).toBeCloseTo(36.87, 1) // u 3, v 4
  })

  it('gives the speed as the magnitude of u and v', () => {
    expect(toArrows({ grid: GRID }).map((a) => a.speed)).toEqual([10, 10, 10, 5])
  })

  it('gives calm air speed 0 and bearing 0, never NaN', () => {
    const calm: WindGrid = [
      { header: GRID[0].header, data: [0, 0, 0, 0] },
      { header: GRID[1].header, data: [0, 0, 0, 0] },
    ]
    for (const arrow of toArrows({ grid: calm })) expect(arrow).toMatchObject({ speed: 0, bearing: 0 })
  })

  it('lays out the recorded 11 × 13 grid from 61 N, 8 W to 49 N, 2 E', () => {
    const arrows = toArrows(WIND_OK)
    expect(arrows).toHaveLength(143)
    expect([arrows[0].lat, arrows[0].lon]).toEqual([61, -8])
    expect([arrows[10].lat, arrows[10].lon]).toEqual([61, 2])
    expect([arrows[142].lat, arrows[142].lon]).toEqual([49, 2])
    expect(arrows.every((a) => Number.isFinite(a.speed) && a.bearing >= 0 && a.bearing < 360)).toBe(true)
  })
})

describe('speed bands, arrow length and the calm dot', () => {
  it.each([
    [0, 'light'],
    [3.99, 'light'],
    [4, 'moderate'],
    [7.99, 'moderate'],
    [8, 'strong'],
    [11.99, 'strong'],
    [12, 'severe'],
    [40, 'severe'],
  ])('puts %s m/s in the %s band', (speed, band) => {
    expect(windBand(speed)).toBe(band)
  })

  it('has a legend line for each band, in order', () => {
    expect(WIND_BANDS.map((band) => band.name)).toEqual(['light', 'moderate', 'strong', 'severe'])
    expect(new Set(WIND_BANDS.map((band) => band.label)).size).toBe(4)
  })

  it('makes a faster wind a longer arrow, between 14 and 44 px', () => {
    expect(arrowLength(0.5)).toBeLessThan(arrowLength(5))
    expect(arrowLength(5)).toBeLessThan(arrowLength(10))
    expect(arrowLength(0)).toBeGreaterThanOrEqual(14)
    expect(arrowLength(100)).toBe(44)
    expect(CALM_MS).toBeGreaterThan(0)
  })
})

describe('readWindPayload', () => {
  it('reads a whole payload', () => {
    expect(readWindPayload(payload())).toEqual(payload())
    expect(readWindPayload(WIND_OK)).not.toBeNull()
  })

  it.each([
    ['null', null],
    ['text', 'oops'],
    ['an array', []],
    ['an error body', { reason: 'down' }],
    ['features instead of a grid', { type: 'FeatureCollection', features: [], updated_at: '2026-11-08T12:00:00Z', stale: false }],
    ['a grid of one record', payload({ grid: [GRID[0]] as unknown as WindGrid })],
    ['the records the wrong way round', payload({ grid: [GRID[1], GRID[0]] })],
    ['no updated_at', { grid: GRID, stale: false }],
    ['a date that is not one', payload({ updated_at: 'yesterday' })],
    ['stale not a boolean', payload({ stale: 'no' as unknown as boolean })],
    ['a short data list', payload({ grid: [{ header: GRID[0].header, data: [1, 2, 3] }, GRID[1]] })],
    ['a null in the data', payload({ grid: [{ header: GRID[0].header, data: [0, null, 0, 0] as unknown as number[] }, GRID[1]] })],
    ['a NaN-like string in the data', payload({ grid: [GRID[0], { header: GRID[1].header, data: [0, 'NaN', 0, 0] as unknown as number[] }] })],
    ['two grids of different sizes', payload({ grid: [GRID[0], { header: { ...GRID[1].header, nx: 4, ny: 1 }, data: [0, 0, 0, 0] }] })],
    ['a header without a refTime', payload({ grid: [{ header: { ...GRID[0].header, refTime: undefined as unknown as string }, data: GRID[0].data }, GRID[1]] })],
    ['a zero step', payload({ grid: [{ header: { ...GRID[0].header, dx: 0 }, data: GRID[0].data }, { header: { ...GRID[1].header, dx: 0 }, data: GRID[1].data }] })],
    ['a grid that claims a million points', payload({ grid: [{ header: { ...GRID[0].header, nx: 1000, ny: 1000 }, data: [] }, { header: { ...GRID[1].header, nx: 1000, ny: 1000 }, data: [] }] })],
  ])('refuses %s whole', (_label, body) => {
    expect(readWindPayload(body)).toBeNull()
  })
})
