import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { AirspaceProps, MapFeature, NotamProps } from '@/map/types'
import { resetSkipped } from '@/pick/skipped'
import { whatsHere, type LoadedFeature } from '@/pick/whatsHere'
import { BLUE_MILL, BLUE_MILL_AIRSPACE, BLUE_MILL_NOTAMS } from '@/test/handlers'

const blueMill = { airspace: BLUE_MILL_AIRSPACE.features, notams: BLUE_MILL_NOTAMS.features }
const names = (rows: { name: string }[]) => rows.map((row) => row.name)

let warn: ReturnType<typeof vi.spyOn>
beforeEach(() => {
  resetSkipped()
  warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
})
afterEach(() => {
  vi.restoreAllMocks()
})

function airspace(id: string, geometry: unknown, over: Partial<AirspaceProps> = {}): LoadedFeature<AirspaceProps> {
  return {
    geometry,
    properties: { id, name: `AREA ${id}`, type: 'tma', class: null, lower: 'SFC', upper: '2000 ft AMSL', lower_ft: 0, upper_ft: 2000, hours: null, action: null, ...over },
  }
}

function notam(id: string, geometry: unknown, over: Partial<NotamProps> = {}): LoadedFeature<NotamProps> {
  return {
    geometry,
    properties: {
      id, q_code: 'QWULW', lower_fl: 0, upper_fl: 10, item_f: 'SFC', item_g: '1000FT AMSL', valid_from: '2026-11-08T10:00:00Z', valid_to: '2026-11-08T18:00:00Z',
      permanent: false, est: false, item_d: null, text: 'RAW TEXT', approx: false, ...over,
    },
  }
}

const square = (w: number, s: number, e: number, n: number) => [[[w, s], [e, s], [e, n], [w, n], [w, s]]]

describe('whatsHere at Blue Mill (GP3)', () => {
  const rows = whatsHere(BLUE_MILL, blueMill)

  it('lists the airspace and NOTAMs whose shape contains the point, one of each kind at least, and not the far ones', () => {
    expect(rows.filter((row) => row.kind === 'airspace').length).toBeGreaterThan(0)
    expect(rows.filter((row) => row.kind === 'notam').length).toBeGreaterThan(0)
    expect(names(rows)).not.toContain('TESTVILLE RESTRICTED AREA')
    expect(names(rows)).not.toContain('H2005/26')
  })

  it('sorts by floor, lowest first, with ties by name or id', () => {
    expect(rows.map((row) => `${row.floorFt} ${row.name}`)).toEqual([
      '0 BLUE MILL TEST ATZ',
      '0 H2001/26',
      '0 H2002/26',
      '0 H2004/26',
      '1500 BLUE MILL TEST CTA',
      '10000 BLUE MILL FL100 FLOOR CTA',
      '10500 BLUE MILL TEST TMA ABOVE FL100',
      '10500 H2003/26',
    ])
  })

  it('carries the chip, limits, times and raw text each row needs', () => {
    const atz = rows.find((row) => row.name === 'BLUE MILL TEST ATZ')
    expect(atz).toMatchObject({ kind: 'airspace', chip: 'ATZ', token: '--type-controlled', limits: 'SFC to 2000 ft AMSL', times: 'MON-FRI 0800-1700', text: null, approx: false })
    const built = rows.find((row) => row.name === 'H2001/26')
    expect(built).toMatchObject({ kind: 'notam', chip: 'NOTAM', token: '--type-notam', limits: 'SFC to 1200FT AMSL', approx: false })
    expect(built?.text).toContain('CTC 01632 960001')
    expect(built?.times).toMatch(/to/)
  })

  it('marks the approx Q-circle NOTAM, and only that one', () => {
    expect(rows.filter((row) => row.approx).map((row) => row.name)).toEqual(['H2002/26'])
  })

  it('counts a NOTAM with no Q-line lower limit as floor 0', () => {
    expect(rows.find((row) => row.name === 'H2004/26')?.floorFt).toBe(0)
  })

  it('does not know about layer switches: switched-off layers are listed because it reads the loaded payloads', () => {
    // the same call, the only input being the loaded payloads, finds both kinds; there is no switch to pass
    expect(whatsHere(BLUE_MILL, blueMill)).toEqual(rows)
    expect(whatsHere(BLUE_MILL, { airspace: blueMill.airspace, notams: [] }).every((row) => row.kind === 'airspace')).toBe(true)
    expect(whatsHere(BLUE_MILL, { airspace: [], notams: blueMill.notams }).every((row) => row.kind === 'notam')).toBe(true)
  })

  it('finds above-FL100 items too (any height)', () => {
    expect(rows.filter((row) => row.floorFt >= 10000)).toHaveLength(3)
  })
})

describe('E5: edges, holes and MultiPolygons', () => {
  const built = BLUE_MILL_NOTAMS.features.find((f: MapFeature<NotamProps>) => f.properties.id === 'H2001/26')!
  const only = (lat: number, lon: number) => names(whatsHere({ lat, lon }, { airspace: [], notams: [built] }))

  it('includes a point exactly on the outer edge', () => {
    expect(only(53.805, -0.1)).toEqual(['H2001/26']) // west edge, lon -0.10
    expect(only(53.79, -0.085)).toEqual(['H2001/26']) // south edge
    expect(only(53.79, -0.1)).toEqual(['H2001/26']) // a corner
  })

  it('excludes a point just outside the edge', () => {
    expect(only(53.805, -0.1001)).toEqual([])
  })

  it('excludes a point in a hole, includes one just outside it, and counts the hole’s own edge as inside the shape', () => {
    expect(only(53.796, -0.085)).toEqual([])
    expect(only(53.798, -0.085)).toEqual(['H2001/26'])
    expect(only(53.796, -0.09)).toEqual(['H2001/26']) // on the hole's west edge: not in the hole
  })

  it('works for a MultiPolygon, in either part and in neither, and with a hole in a part', () => {
    const parts = {
      type: 'MultiPolygon',
      coordinates: [square(-1, 52, -0.5, 52.5), [square(0, 53, 1, 54)[0], [[0.4, 53.4], [0.4, 53.6], [0.6, 53.6], [0.6, 53.4], [0.4, 53.4]]]],
    }
    const layers = { airspace: [airspace('mp', parts)], notams: [] }
    expect(whatsHere({ lat: 52.2, lon: -0.7 }, layers)).toHaveLength(1)
    expect(whatsHere({ lat: 53.2, lon: 0.2 }, layers)).toHaveLength(1)
    expect(whatsHere({ lat: 52.2, lon: 0.2 }, layers)).toHaveLength(0)
    expect(whatsHere({ lat: 53.5, lon: 0.5 }, layers)).toHaveLength(0) // in the hole of the second part
    expect(whatsHere({ lat: 52.5, lon: -0.7 }, layers)).toHaveLength(1) // on an edge
  })
})

describe('bad data', () => {
  const good = airspace('good', { type: 'Polygon', coordinates: square(-1, 52, 1, 54) })
  const at = { lat: 53, lon: 0 }

  it.each([
    ['null geometry', null],
    ['an empty geometry', {}],
    ['a Polygon with no rings', { type: 'Polygon', coordinates: [] }],
    ['an empty MultiPolygon part', { type: 'MultiPolygon', coordinates: [[]] }],
    ['a Point', { type: 'Point', coordinates: [0, 53] }],
    ['a LineString', { type: 'LineString', coordinates: [[0, 53], [1, 53]] }],
    ['a ring that is too short', { type: 'Polygon', coordinates: [[[0, 53], [1, 53], [0, 53]]] }],
    ['coordinates that are not numbers', { type: 'Polygon', coordinates: [[['a', 'b'], [1, 53], [1, 54], [0, 54], [0, 53]]] }],
    ['a string', 'Polygon'],
  ])('skips %s in either layer, keeps the good ones and does not throw', (_name, geometry) => {
    const rows = whatsHere(at, {
      airspace: [airspace('bad-a', geometry), good],
      notams: [notam('H9/26', geometry), notam('H8/26', { type: 'Polygon', coordinates: square(-1, 52, 1, 54) })],
    })
    expect(names(rows).sort()).toEqual(['AREA good', 'H8/26'])
    expect(warn).toHaveBeenCalledTimes(2)
  })

  it('names each skipped feature once, by layer and id, however many times it is looked at, with no text content', () => {
    const bad = notam('H7/26', null, { text: 'SECRET NOTAM WORDS 01632 960001' })
    const layers = { airspace: [airspace('bad-a', null, { name: 'SECRET AREA NAME' })], notams: [bad] }
    for (let i = 0; i < 3; i += 1) whatsHere(at, layers)
    expect(warn.mock.calls.map((call: unknown[]) => call[0])).toEqual([
      'Skipped airspace feature bad-a: no usable shape',
      'Skipped notams feature H7/26: no usable shape',
    ])
    expect(JSON.stringify(warn.mock.calls)).not.toMatch(/SECRET/)
  })

  it('names a skipped feature without an id by its place in the list, and skips one with no properties', () => {
    const rows = whatsHere(at, {
      airspace: [{ geometry: null, properties: { ...good.properties, id: undefined as unknown as string } }, { geometry: good.geometry, properties: null as unknown as AirspaceProps }, good],
      notams: [],
    })
    expect(names(rows)).toEqual(['AREA good'])
    expect(warn.mock.calls.map((call: unknown[]) => call[0])).toEqual([
      'Skipped airspace feature #0: no usable shape',
      'Skipped airspace feature #1: no usable shape',
    ])
  })

  it('returns an empty list for empty layers', () => {
    expect(whatsHere(at, { airspace: [], notams: [] })).toEqual([])
  })

  it('does not warn for a good payload', () => {
    whatsHere(BLUE_MILL, blueMill)
    expect(warn).not.toHaveBeenCalled()
  })
})
