import { http, HttpResponse } from 'msw'

import { apiUrl } from '@/config'
import type { WindPayload } from '@/map/wind'
import okPayload from '@/mocks/wind-ok.json'

// The three answers GET /map/wind gives (src/flysafelz/mapdata/app.py): a fresh grid, the last cached grid with `stale: true`, and a 503 with a reason when
// there is no cache. The grid is synthetic (11 × 13 at 1° over 49–61 N, 8 W–2 E): made-up winds, not a forecast.

export const WIND_URL = () => `${apiUrl()}/map/wind`

export const WIND_OK = okPayload as unknown as WindPayload
export const WIND_STALE: WindPayload = { ...WIND_OK, stale: true }
export const WIND_UNAVAILABLE = { reason: 'wind refresh failed (ReadTimeout)' }

export type WindAnswer = 'ok' | 'stale' | 'unavailable'

/** The MSW handler for the wind route, answering as told. */
export function windRoute(answer: WindAnswer = 'ok') {
  return http.get(WIND_URL(), () => {
    if (answer === 'unavailable') return HttpResponse.json(WIND_UNAVAILABLE, { status: 503 })
    return HttpResponse.json(answer === 'stale' ? WIND_STALE : WIND_OK)
  })
}
