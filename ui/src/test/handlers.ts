import { http, HttpResponse } from 'msw'

import { apiUrl } from '@/config'
import { rainIndexRoute } from '@/mocks/rain'
import { windRoute } from '@/mocks/wind'
import type { AirspaceProps, LayerPayload, NotamProps } from '@/map/types'
import airspaceFixture from '@/test/fixtures/map-airspace.json'
import notamsFixture from '@/test/fixtures/map-notams.json'
import allHeightsAirspace from '@/test/fixtures/map-airspace-all-heights.json'
import allHeightsNotams from '@/test/fixtures/map-notams-all-heights.json'
import blueMillAirspace from '@/test/fixtures/map-airspace.blue-mill.json'
import blueMillNotams from '@/test/fixtures/map-notams.blue-mill.json'

export const AIRSPACE_FIXTURE = airspaceFixture as unknown as LayerPayload<AirspaceProps>
export const NOTAMS_FIXTURE = notamsFixture as unknown as LayerPayload<NotamProps>

export const AIRSPACE_URL = () => `${apiUrl()}/map/airspace`
export const NOTAMS_URL = () => `${apiUrl()}/map/notams`

/** The made-up payloads for both routes (shaped as the routes send them), with any part replaced. */
export function airspaceRoute(body: object = AIRSPACE_FIXTURE) {
  return http.get(AIRSPACE_URL(), () => HttpResponse.json(body))
}

export function notamsRoute(body: object = NOTAMS_FIXTURE) {
  return http.get(NOTAMS_URL(), () => HttpResponse.json(body))
}

/** What the server says when a route has no cache and its refresh failed (src/flysafelz/mapdata/cache.py). */
export const unavailable = (url: string) => http.get(url, () => HttpResponse.json({ reason: 'openAIP: not reachable' }, { status: 503 }))

/** All the map routes with their made-up answers, and RainViewer's index with three current frames (a test of the fallback adds `rainGridRoute`). */
export function mapRoutes() {
  return [airspaceRoute(), notamsRoute(), windRoute(), rainIndexRoute()]
}

/** The Blue Mill payloads (53.802797, -0.086644): an ATZ and a CTA over it, a CTA floored at FL100 and a TMA at FL105, a NOTAM with a hole, an approx Q-circle NOTAM, a NOTAM with no Q-line lower limit, an FL105 NOTAM, and one far away. */
export const BLUE_MILL_AIRSPACE = blueMillAirspace as unknown as LayerPayload<AirspaceProps>
export const BLUE_MILL_NOTAMS = blueMillNotams as unknown as LayerPayload<NotamProps>
export const BLUE_MILL = { lat: 53.802797, lon: -0.086644 }

export function blueMillRoutes() {
  return [airspaceRoute(BLUE_MILL_AIRSPACE), notamsRoute(BLUE_MILL_NOTAMS), windRoute(), rainIndexRoute()]
}

/**
 * The "all heights" payloads (PR-11.F1), synthetic and generated, not recorded from the routes: 240 stacked airspaces (each cell has a low and a high shape over it; floors from SFC to FL195) and 60 NOTAMs
 * (Q-line lower 0, 99, 100, 150, 195, 245 and none; every 5th is an approx circle), with fiction-range phone numbers. Four routes' answers: fresh, stale, and 503.
 */
export const ALL_HEIGHTS_AIRSPACE = allHeightsAirspace as unknown as LayerPayload<AirspaceProps>
export const ALL_HEIGHTS_NOTAMS = allHeightsNotams as unknown as LayerPayload<NotamProps>
export const ALL_HEIGHTS_AIRSPACE_STALE = { ...ALL_HEIGHTS_AIRSPACE, stale: true }
export const ALL_HEIGHTS_NOTAMS_STALE = { ...ALL_HEIGHTS_NOTAMS, stale: true }

export function allHeightsRoutes() {
  return [airspaceRoute(ALL_HEIGHTS_AIRSPACE), notamsRoute(ALL_HEIGHTS_NOTAMS), windRoute(), rainIndexRoute()]
}

export function allHeightsStaleRoutes() {
  return [airspaceRoute(ALL_HEIGHTS_AIRSPACE_STALE), notamsRoute(ALL_HEIGHTS_NOTAMS_STALE), windRoute(), rainIndexRoute()]
}

/** `/map/notams` answers 503 with no cache; the other routes answer as made up above. */
export function allHeightsNotams503Routes() {
  return [airspaceRoute(ALL_HEIGHTS_AIRSPACE), unavailable(NOTAMS_URL()), windRoute(), rainIndexRoute()]
}
