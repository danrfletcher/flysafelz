import { statSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

import { LAND_MAX_BYTES, isUkLand } from '@/pick/land'

const LAND: [string, number, number][] = [
  ['Blue Mill', 53.802797, -0.086644],
  ['Belfast', 54.5973, -5.9301],
  ['Lerwick', 60.1545, -1.1494],
  ['Stornoway', 58.2105, -6.3875],
  ['Douglas (Isle of Man)', 54.1503, -4.4824],
  ['London', 51.5074, -0.1278],
  ['Edinburgh', 55.9533, -3.1883],
  ['Land’s End', 50.0705, -5.6838],
  ['Kirkwall (Orkney)', 58.9809, -2.9605],
]

const NOT_LAND: [string, number, number][] = [
  ['the North Sea', 54.0, 1.5],
  ['mid Irish Sea', 53.7, -4.5],
  ['Paris', 48.8566, 2.3522],
  ['St Helier (Jersey)', 49.1858, -2.1103],
  ['Dublin', 53.3498, -6.2603],
  ['the Atlantic west of Lewis', 58.3, -8.0],
  ['the Channel', 50.2, -1.0],
]

describe('isUkLand against the bundled polygon', () => {
  it.each(LAND)('%s is land', (_name, lat, lon) => {
    expect(isUkLand(lat, lon)).toBe(true)
  })

  it.each(NOT_LAND)('%s is not land', (_name, lat, lon) => {
    expect(isUkLand(lat, lon)).toBe(false)
  })

  it.each([
    ['NaN latitude', Number.NaN, -0.08],
    ['NaN longitude', 53.8, Number.NaN],
    ['infinite latitude', Infinity, -0.08],
    ['latitude above 90', 91, -0.08],
    ['latitude below -90', -91, -0.08],
    ['longitude above 180', 53.8, 181],
    ['longitude below -180', 53.8, -181],
  ])('returns false for %s', (_name, lat, lon) => {
    expect(isUkLand(lat, lon)).toBe(false)
  })

  it('is under the size cap (300 KB) on disk', () => {
    const size = statSync(resolve(__dirname, '../data/uk-land.geojson')).size
    expect(LAND_MAX_BYTES).toBe(300 * 1024)
    expect(size).toBeLessThan(LAND_MAX_BYTES)
    expect(size).toBeGreaterThan(10_000) // a real outline, not a stub
  })
})
