import { describe, expect, it } from 'vitest'

import { FL100_FEET, airspaceBelowFl100, notamBelowFl100 } from '@/map/belowFl100'

// (label, lower text as the route serves it, lower_ft, below FL100?)
const AIRSPACE: [string, string, number | null, boolean][] = [
  ['FL65, below', 'FL65', 6500, true],
  ['FL99, just below', 'FL99', 9900, true],
  ['FL100, exactly at: not below', 'FL100', 10000, false],
  ['FL101, just above', 'FL101', 10100, false],
  ['FL105, above', 'FL105', 10500, false],
  ['FL195, well above', 'FL195', 19500, false],
  ['4500 ft AMSL, below', '4500 ft AMSL', 4500, true],
  ['9999 ft AMSL, just below', '9999 ft AMSL', 9999, true],
  ['10000 ft AMSL, exactly at: not below', '10000 ft AMSL', 10000, false],
  ['10001 ft AMSL, above', '10001 ft AMSL', 10001, false],
  ['12000 ft AMSL, above', '12000 ft AMSL', 12000, false],
  ['1500 ft AGL, below', '1500 ft AGL', 1500, true],
  ['984 ft AGL (300 m on the airspace), below', '984 ft AGL', 984, true],
  ['300 m AGL as text, below', '300 m AGL', 984, true],
  ['an AGL floor with no feet, below', '2000 ft AGL', null, true],
  ['an AGL floor too high to be on a map of the low sky, still below (cannot be compared with a level)', '15000 ft AGL', 15000, true],
  ['SFC, below', 'SFC', 0, true],
  ['GND, below', 'GND', null, true],
  ['sfc in lower case, below', 'sfc', 0, true],
  ['missing lower text, below', '', null, true],
  ['no feet, but a text flight level under 100, below', 'FL65', null, true],
  ['no feet, but a text flight level at 100, not below', 'FL100', null, false],
  ['no feet and a limit that cannot be read, below (nothing low is hidden)', 'whatever', null, true],
]

describe('airspaceBelowFl100', () => {
  it.each(AIRSPACE)('%s', (_label, lower, lowerFt, below) => {
    expect(airspaceBelowFl100({ lower, lower_ft: lowerFt })).toBe(below)
  })

  it('has FL100 as 10,000 ft', () => {
    expect(FL100_FEET).toBe(10_000)
  })

  it('tolerates a lower limit that is not a string or a number', () => {
    expect(airspaceBelowFl100({ lower: undefined as unknown as string, lower_ft: undefined as unknown as number })).toBe(true)
    expect(airspaceBelowFl100({ lower: 'FL65', lower_ft: Number.NaN })).toBe(true)
  })
})

// (label, Q-line lower FL, below FL100?)
const NOTAMS: [string, number | null, boolean][] = [
  ['lower_fl 0: below', 0, true],
  ['lower_fl 99: below', 99, true],
  ['lower_fl 100: exactly at, not below', 100, false],
  ['lower_fl 101: above', 101, false],
  ['lower_fl 150: above', 150, false],
  ['lower_fl 999: above', 999, false],
  ['missing lower_fl: below', null, true],
]

describe('notamBelowFl100', () => {
  it.each(NOTAMS)('%s', (_label, lowerFl, below) => {
    expect(notamBelowFl100({ lower_fl: lowerFl })).toBe(below)
  })

  it('treats a lower_fl that is not a number as missing', () => {
    expect(notamBelowFl100({ lower_fl: undefined as unknown as number })).toBe(true)
    expect(notamBelowFl100({ lower_fl: Number.NaN })).toBe(true)
  })
})
