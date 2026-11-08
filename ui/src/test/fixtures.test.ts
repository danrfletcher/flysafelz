import { describe, expect, it } from 'vitest'

import { ALL_HEIGHTS_AIRSPACE, ALL_HEIGHTS_NOTAMS, AIRSPACE_FIXTURE, BLUE_MILL_AIRSPACE, BLUE_MILL_NOTAMS, NOTAMS_FIXTURE } from '@/test/handlers'

// The airspace ids are 24-character openAIP ids made of digits, so the id is left out of the scan.
// F5: a phone number in a UI fixture is in a range Ofcom keeps for drama, never a real one. The numbers are found by a blunt rule
// (every run of 10 or 11 digits once whitespace is removed), the same idea as tests/unit/phone_leaks.py, and each must start with one of these.
const FICTION = [
  '01632960', '07700900', '02079460', '0113496', '0114496', '0115496', '0116496', '0117496', '0118496', '0121496', '0131496',
  '0141496', '0151496', '0161496', '0191498', '02896496', '02920180', '03069990', '08081570', '09098790',
]

function runs(text: string): string[] {
  return text.replace(/\s+/g, '').match(/\d{10,11}/g) ?? []
}

const strings = (value: unknown): string[] =>
  typeof value === 'string' ? [value] : Array.isArray(value) ? value.flatMap(strings) : value && typeof value === 'object' ? Object.values(value).flatMap(strings) : []

describe('the recorded fixtures', () => {
  it('F5: every phone number in them is in the fiction range, and there are some', () => {
    const found = [AIRSPACE_FIXTURE, NOTAMS_FIXTURE].flatMap((payload) =>
      payload.features.flatMap((feature) => strings({ ...feature.properties, id: '' }).flatMap(runs)),
    )
    expect(found.length).toBeGreaterThanOrEqual(3)
    for (const number of found) expect(FICTION.some((prefix) => number.startsWith(prefix)), number).toBe(true)
  })

  it('cover every coloured type, an unknown type, and a floor at, above and below FL100', () => {
    const types = AIRSPACE_FIXTURE.features.map((feature) => feature.properties.type)
    for (const type of ['restricted', 'prohibited', 'danger', 'ctr', 'cta', 'atz', 'tma', 'matz', 'rmz', 'tmz', 'other']) {
      expect(types, type).toContain(type)
    }
    expect(types).toContain('balloon_zone') // not in openAIP's list
    const floors = AIRSPACE_FIXTURE.features.map((feature) => feature.properties.lower_ft)
    expect(floors).toContain(10000) // exactly FL100
    expect(floors.some((ft) => ft !== null && ft > 10000)).toBe(true)
    expect(floors.some((ft) => ft !== null && ft < 10000)).toBe(true)
    const kinds = new Set(AIRSPACE_FIXTURE.features.map((feature) => feature.geometry.type))
    expect(kinds).toEqual(new Set(['Polygon', 'MultiPolygon']))
    expect(AIRSPACE_FIXTURE.features.some((feature) => feature.geometry.coordinates.length === 2 && feature.geometry.type === 'Polygon')).toBe(true) // a hole
  })

  it('cover an approx NOTAM, an exact one, and Q-line floors of 99, 100 and above', () => {
    const properties = NOTAMS_FIXTURE.features.map((feature) => feature.properties)
    expect(properties.some((p) => p.approx)).toBe(true)
    expect(properties.some((p) => !p.approx)).toBe(true)
    const floors = properties.map((p) => p.lower_fl)
    expect(floors).toEqual(expect.arrayContaining([0, 99, 100]))
    expect(floors.some((fl) => fl !== null && fl > 100)).toBe(true)
  })

  it('are 10 to 20 features each, and in the route’s envelope (not stale, with a time)', () => {
    for (const payload of [AIRSPACE_FIXTURE, NOTAMS_FIXTURE]) {
      expect(Object.keys(payload).sort()).toEqual(['features', 'stale', 'type', 'updated_at'])
      expect(payload.stale).toBe(false)
      expect(Number.isNaN(Date.parse(payload.updated_at))).toBe(false)
    }
    expect(AIRSPACE_FIXTURE.features.length).toBeGreaterThanOrEqual(10)
    expect(AIRSPACE_FIXTURE.features.length).toBeLessThanOrEqual(20)
    expect(NOTAMS_FIXTURE.features.length).toBeLessThanOrEqual(20)
  })
})

describe('the Blue Mill fixtures (PR-4.F1)', () => {
  it('F5: every phone number in them is in the fiction range, and there are some', () => {
    const found = [BLUE_MILL_AIRSPACE, BLUE_MILL_NOTAMS].flatMap((payload) =>
      payload.features.flatMap((feature) => strings({ ...feature.properties, id: '' }).flatMap(runs)),
    )
    expect(found.length).toBeGreaterThanOrEqual(3)
    for (const number of found) expect(FICTION.some((prefix) => number.startsWith(prefix)), number).toBe(true)
  })

  it('hold an ATZ over Blue Mill, a NOTAM with a hole, an approx Q-circle NOTAM, an item at FL100 and one at FL105, in the route’s envelope', () => {
    for (const payload of [BLUE_MILL_AIRSPACE, BLUE_MILL_NOTAMS]) {
      expect(Object.keys(payload).sort()).toEqual(['features', 'stale', 'type', 'updated_at'])
      expect(payload.stale).toBe(false)
    }
    expect(BLUE_MILL_AIRSPACE.features.some((f) => f.properties.type === 'atz' && f.properties.lower_ft === 0)).toBe(true)
    expect(BLUE_MILL_AIRSPACE.features.map((f) => f.properties.lower_ft)).toEqual(expect.arrayContaining([10000, 10500]))
    expect(BLUE_MILL_NOTAMS.features.some((f) => f.geometry.type === 'Polygon' && f.geometry.coordinates.length === 2 && !f.properties.approx)).toBe(true)
    expect(BLUE_MILL_NOTAMS.features.some((f) => f.properties.approx)).toBe(true)
    expect(BLUE_MILL_NOTAMS.features.map((f) => f.properties.lower_fl)).toEqual(expect.arrayContaining([null, 105]))
  })
})

describe('the all-heights fixtures (PR-11.F1)', () => {
  it('F5: every phone number in them is in the fiction range, and there are some', () => {
    const found = [ALL_HEIGHTS_AIRSPACE, ALL_HEIGHTS_NOTAMS].flatMap((payload) =>
      payload.features.flatMap((feature) => strings({ ...feature.properties, id: '' }).flatMap(runs)),
    )
    expect(found.length).toBeGreaterThanOrEqual(3)
    for (const number of found) expect(FICTION.some((prefix) => number.startsWith(prefix)), number).toBe(true)
  })

  it('hold at least 200 airspaces, an unknown type, every limit form, and NOTAMs with Q-line floors on both sides of FL100, in the route’s envelope', () => {
    for (const payload of [ALL_HEIGHTS_AIRSPACE, ALL_HEIGHTS_NOTAMS]) {
      expect(Object.keys(payload).sort()).toEqual(['features', 'stale', 'type', 'updated_at'])
      expect(payload.stale).toBe(false)
    }
    expect(ALL_HEIGHTS_AIRSPACE.features.length).toBeGreaterThanOrEqual(200)
    const lowers = new Set(ALL_HEIGHTS_AIRSPACE.features.map((f) => f.properties.lower))
    for (const form of ['SFC', 'GND', '', 'FL65', 'FL100', 'FL105', 'FL195', '9999 ft AMSL', '10000 ft AMSL', '12000 ft AMSL', '984 ft AGL']) expect(lowers, form).toContain(form)
    expect(ALL_HEIGHTS_AIRSPACE.features.map((f) => f.properties.type)).toContain('balloon_zone')
    const floors = ALL_HEIGHTS_NOTAMS.features.map((f) => f.properties.lower_fl)
    expect(floors).toEqual(expect.arrayContaining([0, 99, 100, 150, 195, 245, null]))
    expect(ALL_HEIGHTS_NOTAMS.features.some((f) => f.properties.approx)).toBe(true)
  })

  it('F5: nothing is at the old West Sleekburn point (55.17 N, 1.58 W) or at Blue Mill', () => {
    const near = (lat: number, lon: number, point: [number, number]) => Math.abs(lat - point[0]) < 0.5 && Math.abs(lon - point[1]) < 0.5
    for (const payload of [ALL_HEIGHTS_AIRSPACE, ALL_HEIGHTS_NOTAMS]) {
      for (const feature of payload.features) {
        const ring = (feature.geometry.type === 'Polygon' ? feature.geometry.coordinates[0] : feature.geometry.coordinates[0][0]) as [number, number][]
        for (const [lon, lat] of ring) {
          expect(near(lat, lon, [55.17, -1.58])).toBe(false)
          expect(near(lat, lon, [53.8028, -0.0866])).toBe(false)
        }
      }
    }
  })
})
