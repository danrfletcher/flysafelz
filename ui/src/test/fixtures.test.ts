import { describe, expect, it } from 'vitest'

import { AIRSPACE_FIXTURE, NOTAMS_FIXTURE } from '@/test/handlers'

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
