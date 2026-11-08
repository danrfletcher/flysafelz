import { http, HttpResponse } from 'msw'

import { apiUrl } from '@/config'
import type { AirspaceProps, LayerPayload, NotamProps } from '@/map/types'
import airspaceFixture from '@/test/fixtures/map-airspace.json'
import notamsFixture from '@/test/fixtures/map-notams.json'

export const AIRSPACE_FIXTURE = airspaceFixture as unknown as LayerPayload<AirspaceProps>
export const NOTAMS_FIXTURE = notamsFixture as unknown as LayerPayload<NotamProps>

export const AIRSPACE_URL = () => `${apiUrl()}/map/airspace`
export const NOTAMS_URL = () => `${apiUrl()}/map/notams`

/** The recorded payloads for both routes, with any part replaced. */
export function airspaceRoute(body: object = AIRSPACE_FIXTURE) {
  return http.get(AIRSPACE_URL(), () => HttpResponse.json(body))
}

export function notamsRoute(body: object = NOTAMS_FIXTURE) {
  return http.get(NOTAMS_URL(), () => HttpResponse.json(body))
}

/** What the server says when a route has no cache and its refresh failed (src/flysafelz/mapdata/cache.py). */
export const unavailable = (url: string) => http.get(url, () => HttpResponse.json({ reason: 'openAIP: not reachable' }, { status: 503 }))

export function mapRoutes() {
  return [airspaceRoute(), notamsRoute()]
}
