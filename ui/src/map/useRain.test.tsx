import { act, renderHook } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { RAINVIEWER_INDEX_URL } from '@/map/rainIndex'
import { REFETCH_MS } from '@/map/useMapLayers'
import { useRain } from '@/map/useRain'
import { RAIN_GRID_OK, RAIN_GRID_URL, rainIndexBody } from '@/mocks/rain'
import { server } from '@/test/setup'

// The data side of the Rain layer: which source it is on, and what a refresh every 15 minutes does. The clock (Date and setInterval only) is fake; the
// requests are real fetches answered by MSW.

const START = new Date('2026-11-08T12:00:00Z')
const secs = (minutesAfterStart: number) => Math.floor(START.getTime() / 1000) + minutesAfterStart * 60

let times: number[] | 'fail'
let grid: 'ok' | 'stale' | 'fail'
let asked: { index: number; grid: number }

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'], now: START })
  times = [secs(-20), secs(-10), secs(-5)]
  grid = 'fail'
  asked = { index: 0, grid: 0 }
  server.use(
    http.get(RAINVIEWER_INDEX_URL, () => {
      asked.index += 1
      return times === 'fail' ? new HttpResponse('down', { status: 500 }) : HttpResponse.json(rainIndexBody(times))
    }),
    http.get(RAIN_GRID_URL(), () => {
      asked.grid += 1
      if (grid === 'fail') return HttpResponse.json({ reason: 'x' }, { status: 503 })
      return HttpResponse.json(grid === 'stale' ? { ...RAIN_GRID_OK, stale: true } : RAIN_GRID_OK)
    }),
  )
})
afterEach(() => vi.useRealTimers())

/** Polls like Testing Library's waitFor, which cannot run here (it polls with the setInterval that is faked): lets real time pass between looks, up to 3 s. */
async function waitFor(check: () => void) {
  let failure: unknown
  for (let i = 0; i < 60; i += 1) {
    try {
      return check()
    } catch (error) {
      failure = error
    }
    await act(async () => {
      await new Promise((done) => setTimeout(done, 50))
    })
  }
  throw failure
}

/** One refresh: the clock moves 15 minutes and the interval fires. */
const refresh = async () => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(REFETCH_MS)
  })
}
const frameMinutes = (state: ReturnType<typeof useRain>['state']) =>
  state.mode === 'radar' ? state.frames.map((frame) => Math.round((frame.time - START.getTime() / 1000) / 60)) : null

describe('useRain', () => {
  it('starts loading, then holds the radar frames oldest first, without touching the rain-grid route', async () => {
    const { result } = renderHook(() => useRain())
    expect(result.current.state).toEqual({ mode: 'loading' })
    await waitFor(() => expect(result.current.state.mode).toBe('radar'))
    expect(frameMinutes(result.current.state)).toEqual([-20, -10, -5])
    expect(asked).toEqual({ index: 1, grid: 0 })
  })

  it('every 15 minutes appends the new frames and drops those older than 2 hours', async () => {
    times = [secs(-115), secs(-105), secs(-5)]
    const { result } = renderHook(() => useRain())
    await waitFor(() => expect(frameMinutes(result.current.state)).toEqual([-115, -105, -5]))
    times = [secs(-105), secs(-5), secs(10)] // 15 minutes later: the -115 frame is now 130 minutes old, the -105 one 120
    await refresh()
    await waitFor(() => expect(frameMinutes(result.current.state)).toEqual([-105, -5, 10]))
    expect(asked.index).toBe(2)
    times = [secs(-5), secs(10), secs(25)] // and 15 more: -105 is 135 old
    await refresh()
    await waitFor(() => expect(frameMinutes(result.current.state)).toEqual([-5, 10, 25]))
    expect(asked.grid).toBe(0)
  })

  it('keeps the frame objects across a refresh that adds nothing, so the layer redraws nothing', async () => {
    const { result } = renderHook(() => useRain())
    await waitFor(() => expect(result.current.state.mode).toBe('radar'))
    const before = result.current.state.mode === 'radar' ? result.current.state.frames : []
    await refresh()
    await waitFor(() => expect(asked.index).toBe(2))
    const after = result.current.state.mode === 'radar' ? result.current.state.frames : null
    expect(after).toBe(before)
  })

  it('keeps the radar it has when a later refresh of the index fails, while its frames are in the window', async () => {
    const { result } = renderHook(() => useRain())
    await waitFor(() => expect(result.current.state.mode).toBe('radar'))
    times = 'fail'
    await refresh()
    await waitFor(() => expect(asked.index).toBe(2))
    expect(frameMinutes(result.current.state)).toEqual([-20, -10, -5])
    expect(asked.grid).toBe(0)
  })

  it('falls back to the grid when the index fails at the start, with the route’s own stale flag', async () => {
    times = 'fail'
    grid = 'stale'
    const { result } = renderHook(() => useRain())
    await waitFor(() => expect(result.current.state.mode).toBe('fallback'))
    expect(result.current.state).toMatchObject({ mode: 'fallback', updatedAt: RAIN_GRID_OK.updated_at, stale: true })
    expect(asked).toEqual({ index: 1, grid: 1 })
  })

  it('goes from the fallback to the radar when the index comes back', async () => {
    times = 'fail'
    grid = 'ok'
    const { result } = renderHook(() => useRain())
    await waitFor(() => expect(result.current.state.mode).toBe('fallback'))
    times = [secs(10)]
    await refresh()
    await waitFor(() => expect(result.current.state.mode).toBe('radar'))
  })

  it('says unavailable when the index fails and the grid does too', async () => {
    times = 'fail'
    const { result } = renderHook(() => useRain())
    await waitFor(() => expect(result.current.state.mode).toBe('unavailable'))
    expect(asked).toEqual({ index: 1, grid: 1 })
  })

  it('marks the grid it has stale when its refresh fails, and recovers on the next good one', async () => {
    times = 'fail'
    grid = 'ok'
    const { result } = renderHook(() => useRain())
    await waitFor(() => expect(result.current.state).toMatchObject({ mode: 'fallback', stale: false }))
    grid = 'fail'
    await refresh()
    await waitFor(() => expect(result.current.state).toMatchObject({ mode: 'fallback', stale: true }))
    grid = 'ok'
    await refresh()
    await waitFor(() => expect(result.current.state).toMatchObject({ mode: 'fallback', stale: false }))
  })

  it('falls back to the grid when the radar it holds has aged out of the window and the index still fails', async () => {
    times = [secs(-100)]
    grid = 'ok'
    const { result } = renderHook(() => useRain())
    await waitFor(() => expect(result.current.state.mode).toBe('radar'))
    times = 'fail'
    await refresh() // 15 minutes on: the frame is 115 minutes old, still in
    await waitFor(() => expect(asked.index).toBe(2))
    expect(result.current.state.mode).toBe('radar')
    await refresh() // 130 minutes old: out
    await waitFor(() => expect(result.current.state.mode).toBe('fallback'))
  })

  it('stops refreshing when it is unmounted', async () => {
    const { result, unmount } = renderHook(() => useRain())
    await waitFor(() => expect(result.current.state.mode).toBe('radar'))
    unmount()
    await refresh()
    await refresh()
    expect(asked.index).toBe(1)
  })
})
