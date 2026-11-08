import { describe, expect, it } from 'vitest'

import { airspaceFloorFt, notamFloorFt, sortByFloor, splitAtFl100, type Sortable } from '@/pick/floor'

describe('airspaceFloorFt (C2)', () => {
  it.each([
    ['SFC', 'SFC', 0, 0],
    ['GND', 'GND', 0, 0],
    ['SFC with no feet', 'SFC', null, 0],
    ['a missing lower limit (empty text)', '', null, 0],
    ['a height above ground (984 ft AGL)', '984 ft AGL', 984, 984],
    ['feet AMSL', '1500 ft AMSL', 1500, 1500],
    ['FL95', 'FL95', 9500, 9500],
    ['FL100', 'FL100', 10000, 10000],
    ['FL245', 'FL245', 24500, 24500],
    ['FL65 sent with no feet', 'FL65', null, 6500],
    ['text it cannot read, with no feet', 'NOT A LIMIT', null, 0],
    ['an above-ground limit with no feet', '300 m AGL', null, 0],
  ])('%s -> %i ft', (_name, lower, lower_ft, expected) => {
    expect(airspaceFloorFt({ lower, lower_ft })).toBe(expected)
  })
})

describe('notamFloorFt (C2)', () => {
  it.each([
    [0, 0],
    [4, 400],
    [95, 9500],
    [99, 9900],
    [100, 10000],
    [245, 24500],
    [null, 0],
  ])('Q-line lower FL %s -> %i ft', (lower_fl, expected) => {
    expect(notamFloorFt({ lower_fl })).toBe(expected)
  })
})

describe('the FL100 split', () => {
  // Which group a row is in is the rule of belowFl100.ts (tested there and in whatsHere.test.ts); here the flag is given, as the card's rows give it.
  const row = (name: string, floorFt: number, key = name): Sortable => ({ name, floorFt, key, aboveFl100: floorFt >= 10000 })

  it('sorts by floor, lowest first, then by name or id, then by key', () => {
    const sorted = sortByFloor([row('B', 1500), row('C', 0), row('A', 1500), row('Z', 0, 'z2'), row('Z', 0, 'z1'), row('D', 24500)])
    expect(sorted.map((r) => `${r.name}${r.floorFt}${r.key.length > 1 ? r.key : ''}`)).toEqual(['C0', 'Z0z1', 'Z0z2', 'A1500', 'B1500', 'D24500'])
  })

  it('is stable: the same rows in any input order give one order, and the input is not changed', () => {
    const rows = [row('B', 500), row('A', 500), row('A', 100), row('C', 10000)]
    const reversed = [...rows].reverse()
    expect(sortByFloor(rows)).toEqual(sortByFloor(reversed))
    expect(rows.map((r) => r.name)).toEqual(['B', 'A', 'A', 'C'])
  })

  it('splits sorted rows into below FL100 and the group above, each in order', () => {
    const { below, above } = splitAtFl100([
      row('TMA', 10500),
      row('CTA FL100', 10000),
      row('ATZ', 0),
      row('FL95', 9500),
      row('CTA', 1500),
      row('HIGH', 24500),
    ])
    expect(below.map((r) => r.name)).toEqual(['ATZ', 'CTA', 'FL95'])
    expect(above.map((r) => r.name)).toEqual(['CTA FL100', 'TMA', 'HIGH'])
  })

  it('gives an empty group when nothing is at or above FL100, and empty lists for no rows', () => {
    expect(splitAtFl100([row('ATZ', 0)]).above).toEqual([])
    expect(splitAtFl100([])).toEqual({ below: [], above: [] })
  })
})
