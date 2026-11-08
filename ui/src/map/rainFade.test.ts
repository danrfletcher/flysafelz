import { describe, expect, it } from 'vitest'

import { FADE_START_ZOOM, HIDDEN_ZOOM, RAIN_OPACITY, rainHidden, rainOpacity } from '@/map/rainFade'

// G5 / Q4: 60% up to zoom 7, falling in a straight line to nothing at zoom 9, nothing above.

describe('rainOpacity', () => {
  it('names the rule it implements', () => {
    expect([RAIN_OPACITY, FADE_START_ZOOM, HIDDEN_ZOOM]).toEqual([0.6, 7, 9])
  })

  it.each<[number, number]>([
    [0, 0.6],
    [3, 0.6],
    [5.25, 0.6],
    [6.75, 0.6],
    [7, 0.6],
    [7.25, 0.525],
    [7.5, 0.45],
    [7.75, 0.375],
    [8, 0.3],
    [8.25, 0.225],
    [8.5, 0.15],
    [8.75, 0.075],
    [9, 0],
    [9.25, 0],
    [12, 0],
    [20, 0],
  ])('at zoom %s the opacity is %s', (zoom, expected) => {
    expect(rainOpacity(zoom)).toBeCloseTo(expected, 10)
  })

  it('falls in a straight line between 7 and 9, never rising', () => {
    let previous = rainOpacity(7)
    for (let zoom = 7.05; zoom <= 9.5; zoom += 0.05) {
      const opacity = rainOpacity(zoom)
      expect(opacity).toBeLessThanOrEqual(previous + 1e-12)
      previous = opacity
    }
    const step = rainOpacity(7.5) - rainOpacity(7.75)
    expect(rainOpacity(7.25) - rainOpacity(7.5)).toBeCloseTo(step, 10)
    expect(rainOpacity(8.5) - rainOpacity(8.75)).toBeCloseTo(step, 10)
  })

  it.each([NaN, Infinity, -Infinity])('gives the full opacity for a zoom that is not a number (%s)', (zoom) => {
    expect(rainOpacity(zoom)).toBe(RAIN_OPACITY)
  })
})

describe('rainHidden (the flag behind "radar: regional view only")', () => {
  it.each<[number, boolean]>([
    [0, false],
    [7, false],
    [8, false],
    [8.75, false],
    [8.999, false],
    [9, true],
    [9.25, true],
    [18, true],
  ])('at zoom %s hidden is %s', (zoom, hidden) => {
    expect(rainHidden(zoom)).toBe(hidden)
  })

  it('is true exactly when the opacity is 0', () => {
    for (let zoom = 0; zoom <= 20; zoom += 0.25) expect(rainHidden(zoom)).toBe(rainOpacity(zoom) === 0)
  })
})
