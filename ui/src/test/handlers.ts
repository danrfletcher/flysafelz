import { http, HttpResponse } from 'msw'

import { apiUrl } from '@/config'
import { windRoute } from '@/mocks/wind'
import type { AirspaceProps, LayerPayload, NotamProps } from '@/map/types'
import airspaceFixture from '@/test/fixtures/map-airspace.json'
import notamsFixture from '@/test/fixtures/map-notams.json'
import blueMillAirspace from '@/test/fixtures/map-airspace.blue-mill.json'
import blueMillNotams from '@/test/fixtures/map-notams.blue-mill.json'

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

/** All three map routes with their recorded answers. */
export function mapRoutes() {
  return [airspaceRoute(), notamsRoute(), windRoute()]
}

/** The Blue Mill payloads (53.802797, -0.086644): an ATZ and a CTA over it, a CTA floored at FL100 and a TMA at FL105, a NOTAM with a hole, an approx Q-circle NOTAM, a NOTAM with no Q-line lower limit, an FL105 NOTAM, and one far away. */
export const BLUE_MILL_AIRSPACE = blueMillAirspace as unknown as LayerPayload<AirspaceProps>
export const BLUE_MILL_NOTAMS = blueMillNotams as unknown as LayerPayload<NotamProps>
export const BLUE_MILL = { lat: 53.802797, lon: -0.086644 }

export function blueMillRoutes() {
  return [airspaceRoute(BLUE_MILL_AIRSPACE), notamsRoute(BLUE_MILL_NOTAMS), windRoute()]
}
