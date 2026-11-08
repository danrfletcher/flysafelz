import { describe, expect, it } from 'vitest'

import { cellOpacity, DRY_MM, readRainGridPayload } from '@/map/rainGrid'
import { RAIN_GRID_OK } from '@/mocks/rain'

const body = (over: object = {}) => ({ grid: [{ lat: 54, lon: -2, precipitation: 1.2 }], updated_at: '2026-11-08T12:00:00Z', stale: false, ...over })

describe('readRainGridPayload', () => {
  it('reads the recorded answer whole', () => {
    const payload = readRainGridPayload(RAIN_GRID_OK)
    expect(payload?.grid).toHaveLength(RAIN_GRID_OK.grid.length)
    expect(payload?.stale).toBe(false)
  })

  it('keeps the route’s stale flag', () => {
    expect(readRainGridPayload(body({ stale: true }))?.stale).toBe(true)
  })

  it('accepts an empty grid (a dry day is no failure)', () => {
    expect(readRainGridPayload(body({ grid: [] }))?.grid).toEqual([])
  })

  it('leaves out a cell that is not a number triple or has negative rain, and never reads it as zero', () => {
    const payload = readRainGridPayload(
      body({
        grid: [
          { lat: 54, lon: -2, precipitation: 0.4 },
          { lat: 54, lon: -1, precipitation: null },
          { lat: 54, lon: 0, precipitation: -1 },
          { lat: 54, lon: 1, precipitation: '2' },
          { lat: 'x', lon: 1, precipitation: 2 },
          { lat: 54, lon: 2 },
          null,
          'cell',
        ],
      }),
    )
    expect(payload?.grid).toEqual([{ lat: 54, lon: -2, precipitation: 0.4 }])
  })

  it.each<[string, unknown]>([
    ['a list', []],
    ['text', 'x'],
    ['null', null],
    ['no grid', { updated_at: '2026-11-08T12:00:00Z', stale: false }],
    ['grid not a list', body({ grid: {} })],
    ['a reason (the 503 body)', { reason: 'Open-Meteo: not reachable' }],
    ['no time', body({ updated_at: undefined })],
    ['a time that is not a time', body({ updated_at: 'tomorrow' })],
    ['stale not a boolean', body({ stale: 'no' })],
    ['an absurd number of cells', body({ grid: Array.from({ length: 10_001 }, () => ({ lat: 1, lon: 1, precipitation: 1 })) })],
  ])('refuses %s', (_name, value) => {
    expect(readRainGridPayload(value)).toBeNull()
  })
})

describe('cellOpacity', () => {
  it('rises from a faint trace to 0.75 at 4 mm and stays there', () => {
    expect(cellOpacity(0)).toBeCloseTo(0.25)
    expect(cellOpacity(2)).toBeCloseTo(0.5)
    expect(cellOpacity(4)).toBeCloseTo(0.75)
    expect(cellOpacity(40)).toBeCloseTo(0.75)
    expect(DRY_MM).toBeGreaterThan(0)
  })
})
