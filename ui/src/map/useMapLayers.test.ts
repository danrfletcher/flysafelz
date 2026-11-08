import { act, renderHook, waitFor } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { REFETCH_MS, formatUpdated, nextState, readPayload, useMapLayers, type LayerState } from '@/map/useMapLayers'
import { server } from '@/test/setup'
import {
  AIRSPACE_FIXTURE,
  AIRSPACE_URL,
  NOTAMS_FIXTURE,
  NOTAMS_URL,
  airspaceRoute,
  mapRoutes,
  notamsRoute,
  unavailable,
} from '@/test/handlers'

afterEach(() => {
  vi.useRealTimers()
})

describe('fetch states', () => {
  it('starts as loading, then ok with the features and the route’s updated_at', async () => {
    server.use(...mapRoutes())
    const { result } = renderHook(() => useMapLayers())
    expect(result.current.airspace).toMatchObject({ status: 'loading', features: [], updatedAt: null })
    expect(result.current.notams.status).toBe('loading')
    await waitFor(() => expect(result.current.airspace.status).toBe('ok'))
    await waitFor(() => expect(result.current.notams.status).toBe('ok'))
    expect(result.current.airspace.features).toHaveLength(AIRSPACE_FIXTURE.features.length)
    expect(result.current.notams.features).toHaveLength(NOTAMS_FIXTURE.features.length)
    expect(result.current.airspace).toMatchObject({ updatedAt: AIRSPACE_FIXTURE.updated_at, stale: false })
  })

  it('carries stale: true from the route and still has the data', async () => {
    server.use(airspaceRoute(), notamsRoute({ ...NOTAMS_FIXTURE, stale: true }))
    const { result } = renderHook(() => useMapLayers())
    await waitFor(() => expect(result.current.notams.status).toBe('ok'))
    expect(result.current.notams.stale).toBe(true)
    expect(result.current.notams.features.length).toBeGreaterThan(0)
    expect(result.current.airspace.stale).toBe(false)
  })

  it('treats a 503 as unavailable for that layer only', async () => {
    server.use(unavailable(AIRSPACE_URL()), notamsRoute())
    const { result } = renderHook(() => useMapLayers())
    await waitFor(() => expect(result.current.airspace.status).toBe('unavailable'))
    await waitFor(() => expect(result.current.notams.status).toBe('ok'))
    expect(result.current.airspace.features).toEqual([])
  })

  it('treats a network error as unavailable, with no unhandled rejection', async () => {
    const rejections: unknown[] = []
    const onRejection = (reason: unknown) => rejections.push(reason)
    process.on('unhandledRejection', onRejection)
    server.use(http.get(AIRSPACE_URL(), () => HttpResponse.error()), notamsRoute())
    const { result } = renderHook(() => useMapLayers())
    await waitFor(() => expect(result.current.airspace.status).toBe('unavailable'))
    await new Promise((done) => setTimeout(done, 20))
    process.off('unhandledRejection', onRejection)
    expect(rejections).toEqual([])
  })

  it.each([
    ['text that is not JSON', () => new HttpResponse('<html>Access denied</html>', { headers: { 'content-type': 'text/html' } })],
    ['JSON that is not a payload', () => HttpResponse.json({ hello: 'world' })],
    ['a features list that is not a list', () => HttpResponse.json({ features: 'no', updated_at: '2026-11-08T12:00:00Z', stale: false })],
    ['an updated_at that is not a time', () => HttpResponse.json({ features: [], updated_at: 'yesterday', stale: false })],
    ['a stale flag that is not a boolean', () => HttpResponse.json({ features: [], updated_at: '2026-11-08T12:00:00Z', stale: 'no' })],
    ['null', () => HttpResponse.json(null)],
    ['a list', () => HttpResponse.json([])],
  ])('treats %s as unavailable', async (_label, answer) => {
    server.use(http.get(NOTAMS_URL(), answer), airspaceRoute())
    const { result } = renderHook(() => useMapLayers())
    await waitFor(() => expect(result.current.notams.status).toBe('unavailable'))
    await waitFor(() => expect(result.current.airspace.status).toBe('ok'))
  })

  it('shows an empty payload as ok with no features (E15)', async () => {
    server.use(notamsRoute({ features: [], updated_at: '2026-11-08T12:00:00Z', stale: false }), airspaceRoute())
    const { result } = renderHook(() => useMapLayers())
    await waitFor(() => expect(result.current.notams.status).toBe('ok'))
    expect(result.current.notams).toEqual({ status: 'ok', features: [], updatedAt: '2026-11-08T12:00:00Z', stale: false })
  })

  it('leaves out a feature that cannot be drawn and keeps the rest', async () => {
    const good = AIRSPACE_FIXTURE.features[0]
    const broken = [
      { ...good, geometry: { type: 'Polygon', coordinates: [[[0, 0], [1, 1]]] } }, // a ring with two points
      { ...good, geometry: { type: 'Point', coordinates: [0, 0] } },
      { ...good, geometry: { type: 'Polygon', coordinates: [[['x', 1], [1, 1], [1, 2], ['x', 1]]] } },
      { ...good, geometry: null },
      { ...good, properties: null },
      { ...good, properties: { ...good.properties, type: 7 } },
      null,
      'feature',
    ]
    server.use(airspaceRoute({ ...AIRSPACE_FIXTURE, features: [good, ...broken] }), notamsRoute())
    const { result } = renderHook(() => useMapLayers())
    await waitFor(() => expect(result.current.airspace.status).toBe('ok'))
    expect(result.current.airspace.features).toEqual([good])
  })

  it('draws a MultiPolygon and a polygon with a hole', async () => {
    server.use(...mapRoutes())
    const { result } = renderHook(() => useMapLayers())
    await waitFor(() => expect(result.current.airspace.status).toBe('ok'))
    const kinds = result.current.airspace.features.map((feature) => feature.geometry.type)
    expect(kinds).toContain('MultiPolygon')
    expect(result.current.airspace.features.some((feature) => feature.geometry.type === 'Polygon' && feature.geometry.coordinates.length === 2)).toBe(true)
  })

  it('renders an empty airspace list as ok', async () => {
    server.use(airspaceRoute({ ...AIRSPACE_FIXTURE, features: [] }), notamsRoute())
    const { result } = renderHook(() => useMapLayers())
    await waitFor(() => expect(result.current.airspace.status).toBe('ok'))
    expect(result.current.airspace.features).toEqual([])
  })
})

// Only setInterval and clearInterval are faked, so fetch and MSW run on real time. Neither Testing Library's waitFor (it polls with
// setInterval) nor vi.waitFor (it advances fake time on every poll) can wait here, so `until` polls on the real setTimeout.
async function until(check: () => void): Promise<void> {
  for (let tries = 0; tries < 200; tries += 1) {
    try {
      return check()
    } catch {
      await act(() => new Promise((done) => setTimeout(done, 10)))
    }
  }
  check()
}

describe('refetch every 15 minutes', () => {
  let airspaceCalls: number
  let notamCalls: number

  beforeEach(() => {
    airspaceCalls = 0
    notamCalls = 0
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] })
  })

  const counting = (kind: 'airspace' | 'notams', answer: () => Response) =>
    http.get(kind === 'airspace' ? AIRSPACE_URL() : NOTAMS_URL(), () => {
      if (kind === 'airspace') airspaceCalls += 1
      else notamCalls += 1
      return answer()
    })

  it('is 15 minutes', () => {
    expect(REFETCH_MS).toBe(15 * 60 * 1000)
  })

  it('fetches both routes on load and again after each 15 minutes, until unmounted', async () => {
    server.use(
      counting('airspace', () => HttpResponse.json(AIRSPACE_FIXTURE)),
      counting('notams', () => HttpResponse.json(NOTAMS_FIXTURE)),
    )
    const { result, unmount } = renderHook(() => useMapLayers())
    await until(() => expect(result.current.notams.status).toBe('ok'))
    expect([airspaceCalls, notamCalls]).toEqual([1, 1])

    await act(() => vi.advanceTimersByTimeAsync(REFETCH_MS - 1))
    expect([airspaceCalls, notamCalls]).toEqual([1, 1])
    await act(() => vi.advanceTimersByTimeAsync(1))
    await until(() => expect([airspaceCalls, notamCalls]).toEqual([2, 2]))

    await act(() => vi.advanceTimersByTimeAsync(REFETCH_MS))
    await until(() => expect([airspaceCalls, notamCalls]).toEqual([3, 3]))

    unmount()
    await act(() => vi.advanceTimersByTimeAsync(3 * REFETCH_MS))
    expect([airspaceCalls, notamCalls]).toEqual([3, 3])
  })

  it('takes the new data and the new updated_at from a later fetch', async () => {
    let body = AIRSPACE_FIXTURE
    server.use(http.get(AIRSPACE_URL(), () => HttpResponse.json(body)), notamsRoute())
    const { result } = renderHook(() => useMapLayers())
    await until(() => expect(result.current.airspace.status).toBe('ok'))
    body = { ...AIRSPACE_FIXTURE, features: AIRSPACE_FIXTURE.features.slice(0, 2), updated_at: '2026-11-08T12:15:00Z' }
    await act(() => vi.advanceTimersByTimeAsync(REFETCH_MS))
    await until(() => expect(result.current.airspace.features).toHaveLength(2))
    expect(result.current.airspace.updatedAt).toBe('2026-11-08T12:15:00Z')
  })

  it('keeps the last drawn data and marks it stale when a refetch fails, then recovers', async () => {
    let answer: () => Response = () => HttpResponse.json(NOTAMS_FIXTURE)
    server.use(airspaceRoute(), http.get(NOTAMS_URL(), () => answer()))
    const { result } = renderHook(() => useMapLayers())
    await until(() => expect(result.current.notams.status).toBe('ok'))
    const drawn = result.current.notams.features

    answer = () => HttpResponse.json({ reason: 'PIB down' }, { status: 503 })
    await act(() => vi.advanceTimersByTimeAsync(REFETCH_MS))
    await until(() => expect(result.current.notams.stale).toBe(true))
    expect(result.current.notams).toMatchObject({ status: 'ok', updatedAt: NOTAMS_FIXTURE.updated_at })
    expect(result.current.notams.features).toBe(drawn)
    expect(result.current.airspace.stale).toBe(false)

    answer = () => HttpResponse.error()
    await act(() => vi.advanceTimersByTimeAsync(REFETCH_MS))
    expect(result.current.notams).toMatchObject({ status: 'ok', stale: true })

    answer = () => HttpResponse.json(NOTAMS_FIXTURE)
    await act(() => vi.advanceTimersByTimeAsync(REFETCH_MS))
    await until(() => expect(result.current.notams.stale).toBe(false))
  })

  it('recovers from unavailable when a later fetch works', async () => {
    let answer: () => Response = () => HttpResponse.json({ reason: 'down' }, { status: 503 })
    server.use(airspaceRoute(), http.get(NOTAMS_URL(), () => answer()))
    const { result } = renderHook(() => useMapLayers())
    await until(() => expect(result.current.notams.status).toBe('unavailable'))
    answer = () => HttpResponse.json(NOTAMS_FIXTURE)
    await act(() => vi.advanceTimersByTimeAsync(REFETCH_MS))
    await until(() => expect(result.current.notams.status).toBe('ok'))
  })
})

describe('nextState and readPayload', () => {
  const ok: LayerState<unknown> = { status: 'ok', features: [], updatedAt: 'x', stale: false }

  it('keeps ok data as stale on a failure and says unavailable when there is none', () => {
    expect(nextState(ok, null)).toEqual({ ...ok, stale: true })
    expect(nextState({ status: 'loading', features: [], updatedAt: null, stale: false }, null).status).toBe('unavailable')
    expect(nextState({ status: 'unavailable', features: [], updatedAt: null, stale: false }, null).status).toBe('unavailable')
  })

  it('reads the fixtures whole', () => {
    expect(readPayload(NOTAMS_FIXTURE, () => true)?.features).toHaveLength(NOTAMS_FIXTURE.features.length)
  })
})

describe('formatUpdated: HH:MM in the browser’s time zone', () => {
  const original = process.env.TZ
  afterEach(() => {
    if (original === undefined) delete process.env.TZ
    else process.env.TZ = original
  })

  it.each([
    ['Europe/London', '2026-11-08T12:00:00Z', '12:00'], // GMT
    ['Europe/London', '2026-07-08T12:00:00Z', '13:00'], // BST
    ['America/New_York', '2026-11-08T12:00:00Z', '07:00'],
    ['Asia/Kolkata', '2026-11-08T12:00:00Z', '17:30'], // a half-hour zone
    ['Pacific/Auckland', '2026-11-08T12:00:00Z', '01:00'], // next day, past midnight
    ['UTC', '2026-11-08T00:05:00Z', '00:05'], // midnight is 00, not 24
    ['UTC', '2026-11-08T23:59:59Z', '23:59'],
  ])('%s: %s is %s', (zone, iso, want) => {
    process.env.TZ = zone
    expect(formatUpdated(iso)).toBe(want)
  })

  // The clocks go back in the UK at 01:00 UTC on 25 Oct 2026 (02:00 BST becomes 01:00 GMT): 00:30Z is 01:30 BST, and 01:30Z is 01:30 GMT.
  it.each([
    ['2026-10-25T00:30:00Z', '01:30'], // still BST
    ['2026-10-25T00:59:00Z', '01:59'], // the last minute of BST
    ['2026-10-25T01:00:00Z', '01:00'], // the first minute of GMT
    ['2026-10-25T01:30:00Z', '01:30'], // GMT: the same wall clock again
    ['2026-10-24T23:59:00Z', '00:59'], // the day before, BST
  ])('at the BST to GMT change in Europe/London, %s is %s', (iso, want) => {
    process.env.TZ = 'Europe/London'
    expect(formatUpdated(iso)).toBe(want)
  })

  it('at the GMT to BST change on 29 Mar 2026 (01:00 UTC), skips the missing hour', () => {
    process.env.TZ = 'Europe/London'
    expect(formatUpdated('2026-03-29T00:59:00Z')).toBe('00:59')
    expect(formatUpdated('2026-03-29T01:00:00Z')).toBe('02:00')
  })

  it('gives an empty string for a value that is not a time', () => {
    expect(formatUpdated('not a time')).toBe('')
  })
})
