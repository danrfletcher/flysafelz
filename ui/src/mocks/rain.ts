import { http, HttpResponse } from 'msw'

import { apiUrl } from '@/config'
import { RAINVIEWER_INDEX_URL } from '@/map/rainIndex'
import type { RainGridPayload } from '@/map/rainGrid'
import okGrid from '@/mocks/rain-grid-ok.json'

// What the Rain layer's two sources answer (PR-10.F1). RainViewer: the index (https://api.rainviewer.com/public/weather-maps.json) in the shape its docs show,
// with frames at 10-minute steps ending at `nowMs`; it holds no real radar. The fallback: GET /map/rain-grid (src/flysafelz/mapdata/app.py), a synthetic
// 11 × 13 grid of made-up precipitation: not a forecast. Radar tiles are images: jsdom does not load them, so no handler is needed for them.

export const TILE_HOST = 'https://tilecache.rainviewer.com'
export const RAIN_GRID_URL = () => `${apiUrl()}/map/rain-grid`
export const RAIN_GRID_OK = okGrid as unknown as RainGridPayload
export const RAIN_GRID_STALE: RainGridPayload = { ...RAIN_GRID_OK, stale: true }
export const RAIN_GRID_UNAVAILABLE = { reason: 'Open-Meteo: not reachable' }

/** The frame times (unix seconds) of `count` frames 10 minutes apart, the last one `agoMinutes` before `nowMs`, oldest first. */
export function frameTimes(nowMs: number, count = 3, agoMinutes = 5): number[] {
  const last = Math.floor(nowMs / 1000) - agoMinutes * 60
  return Array.from({ length: count }, (_, i) => last - (count - 1 - i) * 600)
}

/** An index body, as RainViewer sends it. */
export function rainIndexBody(times: readonly number[], host = TILE_HOST) {
  return {
    version: '2.0',
    generated: times.length ? times[times.length - 1] : 0,
    host,
    radar: { past: times.map((time) => ({ time, path: `/v2/radar/${time.toString(16)}` })), nowcast: [] },
  }
}

export type RainIndexAnswer =
  | { frames: number } // `frames` frames ending 5 minutes ago, 10 minutes apart
  | { times: number[] }
  | 'empty'
  | 'server-error'
  | 'not-json'
  | 'network-error'
  | 'not-an-index'

/** The MSW handler for RainViewer's index, answering as told (three current frames by default). Each request is answered from the clock at that moment. */
export function rainIndexRoute(answer: RainIndexAnswer = { frames: 3 }) {
  return http.get(RAINVIEWER_INDEX_URL, () => {
    if (answer === 'server-error') return new HttpResponse('upstream down', { status: 500 })
    if (answer === 'network-error') return HttpResponse.error()
    if (answer === 'not-json') return new HttpResponse('<html>no</html>', { status: 200, headers: { 'content-type': 'text/html' } })
    if (answer === 'not-an-index') return HttpResponse.json({ hello: 'world' })
    if (answer === 'empty') return HttpResponse.json(rainIndexBody([]))
    return HttpResponse.json(rainIndexBody('times' in answer ? answer.times : frameTimes(Date.now(), answer.frames)))
  })
}

export type RainGridAnswer = 'ok' | 'stale' | 'unavailable'

/** The MSW handler for the rain-grid route, answering as told. */
export function rainGridRoute(answer: RainGridAnswer = 'ok') {
  return http.get(RAIN_GRID_URL(), () => {
    if (answer === 'unavailable') return HttpResponse.json(RAIN_GRID_UNAVAILABLE, { status: 503 })
    return HttpResponse.json(answer === 'stale' ? RAIN_GRID_STALE : RAIN_GRID_OK)
  })
}
