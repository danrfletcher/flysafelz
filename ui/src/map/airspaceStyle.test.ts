import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  AIRSPACE_TYPES,
  airspaceStyle,
  notamStyle,
  pathOptions,
  resetWarnedTypes,
  themeColor,
  type LayerStyle,
} from '@/map/airspaceStyle'

// Outline weight 2 and no dash for every airspace type; toEqual reads a missing dashArray and an undefined one as the same.
const RED: LayerStyle = { token: '--type-restricted', weight: 2, fillOpacity: 0.15 }
const ORANGE: LayerStyle = { token: '--type-danger', weight: 2, fillOpacity: 0 }
const SKY: LayerStyle = { token: '--type-controlled', weight: 2, fillOpacity: 0 }
const SLATE: LayerStyle = { token: '--type-other', weight: 2, fillOpacity: 0 }

// Every name in openAIP's CODES["type"] list (src/flysafelz/clients/openaip.py), and what the rule gives it.
const EXPECTED: Record<string, LayerStyle> = Object.fromEntries(AIRSPACE_TYPES.map((name) => [name, SLATE]))
Object.assign(EXPECTED, {
  restricted: RED,
  prohibited: RED,
  danger: ORANGE,
  ctr: SKY,
  cta: SKY,
  atz: SKY,
})

const TABLE = Object.entries(EXPECTED)

let warn: ReturnType<typeof vi.spyOn>

beforeEach(() => {
  resetWarnedTypes()
  warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
})

afterEach(() => {
  warn.mockRestore()
})

describe('airspaceStyle: type to outline, fill and dash', () => {
  it('covers all 37 names of the openAIP type list', () => {
    expect(AIRSPACE_TYPES).toHaveLength(37)
    expect(new Set(AIRSPACE_TYPES).size).toBe(37)
  })

  it.each(TABLE)('%s', (type, want) => {
    const style = airspaceStyle(type)
    expect(style).toEqual(want)
    expect(style.dashArray).toBeUndefined()
    expect(warn).not.toHaveBeenCalled() // a known name never warns
  })

  it('gives only restricted and prohibited a fill', () => {
    const filled = AIRSPACE_TYPES.filter((type) => airspaceStyle(type).fillOpacity > 0)
    expect(filled).toEqual(['restricted', 'prohibited'])
  })

  it('makes tma, matz, rmz and tmz slate, as Q10 says', () => {
    for (const type of ['tma', 'matz', 'rmz', 'tmz', 'other']) expect(airspaceStyle(type).token).toBe('--type-other')
  })
})

describe('E11: an unknown type', () => {
  it('is slate and warns once per distinct value, with the value only', () => {
    expect(airspaceStyle('balloon_zone')).toEqual(SLATE)
    expect(airspaceStyle('balloon_zone')).toEqual(SLATE)
    expect(airspaceStyle('balloon_zone')).toEqual(SLATE)
    expect(warn).toHaveBeenCalledTimes(1)
    expect(warn.mock.calls[0]).toEqual(['Unknown airspace type: "balloon_zone"'])

    airspaceStyle('other_new_thing')
    airspaceStyle('balloon_zone')
    expect(warn).toHaveBeenCalledTimes(2)
    expect(warn.mock.calls[1]).toEqual(['Unknown airspace type: "other_new_thing"'])
  })

  it('treats names that exist on every object (constructor, __proto__) as unknown, not as a rule', () => {
    expect(airspaceStyle('constructor')).toEqual(SLATE)
    expect(airspaceStyle('__proto__')).toEqual(SLATE)
    expect(warn).toHaveBeenCalledTimes(2)
  })

  it('takes a type string and nothing else, so an action cannot change the colour', () => {
    expect(airspaceStyle.length).toBe(1)
    expect(airspaceStyle('ctr')).toEqual(SKY)
  })
})

describe('notamStyle', () => {
  it('is violet and dashed with no fill, and approx has its own finer dash', () => {
    expect(notamStyle(false)).toEqual({ token: '--type-notam', weight: 2, fillOpacity: 0, dashArray: '8 6' })
    expect(notamStyle(true)).toEqual({ token: '--type-notam', weight: 2, fillOpacity: 0, dashArray: '3 7' })
  })
})

describe('pathOptions: the theme variable, resolved', () => {
  it('reads the colour from the theme variable on the page', () => {
    expect(themeColor('--type-restricted')).toBe('#ef4444')
    expect(themeColor('--type-notam')).toBe('#a78bfa')
    const red = pathOptions(airspaceStyle('restricted'))
    expect(red).toMatchObject({ color: '#ef4444', fillColor: '#ef4444', fillOpacity: 0.15, weight: 2, opacity: 1, fill: true })
    const orange = pathOptions(airspaceStyle('danger'))
    expect(orange).toMatchObject({ color: '#f97316', fillOpacity: 0, fill: true }) // fill stays on so a hover inside still counts
    const violet = pathOptions(notamStyle(false))
    expect(violet).toMatchObject({ color: '#a78bfa', dashArray: '8 6' })
  })

  it('follows a changed variable', () => {
    const before = document.documentElement.style.getPropertyValue('--type-other')
    document.documentElement.style.setProperty('--type-other', '#123456')
    try {
      expect(pathOptions(airspaceStyle('tma')).color).toBe('#123456')
    } finally {
      document.documentElement.style.setProperty('--type-other', before)
    }
  })
})
