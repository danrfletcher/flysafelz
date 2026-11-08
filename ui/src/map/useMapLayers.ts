import { useEffect, useState } from 'react'

import { apiUrl } from '@/config'
import type { AirspaceProps, LayerPayload, MapFeature, MapGeometry, NotamProps, Position } from '@/map/types'

/** Both routes are fetched on load and then this often while the page is open. There is no refresh button. */
export const REFETCH_MS = 15 * 60 * 1000

export const AIRSPACE_PATH = '/map/airspace'
export const NOTAMS_PATH = '/map/notams'

/**
 * What the panel and the map know about one route.
 *   loading      nothing has come back yet
 *   ok           `features` are drawn; `stale` is the route's own flag, or true when a later refetch failed and the old data is kept
 *   unavailable  the route answered 503, the network failed, or the body was not a payload, and there is no earlier data to keep
 */
export interface LayerState<P> {
  status: 'loading' | 'ok' | 'unavailable'
  features: MapFeature<P>[]
  /** The route's `updated_at` (ISO UTC) of the data on the map, or null. */
  updatedAt: string | null
  stale: boolean
}

const LOADING: LayerState<never> = { status: 'loading', features: [], updatedAt: null, stale: false }
const UNAVAILABLE: LayerState<never> = { status: 'unavailable', features: [], updatedAt: null, stale: false }

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isPosition(value: unknown): value is Position {
  return Array.isArray(value) && value.length >= 2 && Number.isFinite(value[0]) && Number.isFinite(value[1])
}

function isRing(value: unknown): boolean {
  return Array.isArray(value) && value.length >= 4 && value.every(isPosition)
}

function isGeometry(value: unknown): value is MapGeometry {
  if (!isObject(value) || !Array.isArray(value.coordinates)) return false
  if (value.type === 'Polygon') return value.coordinates.length > 0 && value.coordinates.every(isRing)
  if (value.type === 'MultiPolygon') {
    return value.coordinates.every((polygon) => Array.isArray(polygon) && polygon.length > 0 && polygon.every(isRing))
  }
  return false
}

const isText = (value: unknown) => typeof value === 'string'
const isNumberOrNull = (value: unknown) => value === null || (typeof value === 'number' && Number.isFinite(value))

function hasAirspaceProps(p: Record<string, unknown>): boolean {
  return isText(p.name) && isText(p.type) && isText(p.lower) && isText(p.upper) && isNumberOrNull(p.lower_ft)
}

function hasNotamProps(p: Record<string, unknown>): boolean {
  return isText(p.id) && isNumberOrNull(p.lower_fl) && isText(p.valid_from) && typeof p.approx === 'boolean'
}

/** The payload in a response body, or null when it is not one. A feature that cannot be drawn is left out; the rest stay. */
export function readPayload<P>(body: unknown, hasProps: (p: Record<string, unknown>) => boolean): LayerPayload<P> | null {
  if (!isObject(body) || !Array.isArray(body.features)) return null
  if (typeof body.updated_at !== 'string' || Number.isNaN(Date.parse(body.updated_at))) return null
  if (typeof body.stale !== 'boolean') return null
  const features = body.features.filter(
    (feature): feature is MapFeature<P> =>
      isObject(feature) && isGeometry(feature.geometry) && isObject(feature.properties) && hasProps(feature.properties),
  )
  return { features, updated_at: body.updated_at, stale: body.stale }
}

/** One fetch of one route: the payload, or null for a 503, any other bad status, a network error or a body that is not a payload. Never rejects. */
async function fetchLayer<P>(path: string, hasProps: (p: Record<string, unknown>) => boolean, signal: AbortSignal) {
  try {
    const response = await fetch(`${apiUrl()}${path}`, { signal })
    if (!response.ok) return null
    return readPayload<P>(await response.json(), hasProps)
  } catch {
    return null
  }
}

/** The state after a fetch: new data replaces the old; a failure keeps what is drawn and marks it stale, or says unavailable when nothing is. */
export function nextState<P>(previous: LayerState<P>, payload: LayerPayload<P> | null): LayerState<P> {
  if (payload) {
    return { status: 'ok', features: payload.features, updatedAt: payload.updated_at, stale: payload.stale }
  }
  if (previous.status === 'ok') return { ...previous, stale: true }
  return UNAVAILABLE
}

function useRoute<P>(path: string, hasProps: (p: Record<string, unknown>) => boolean): LayerState<P> {
  const [state, setState] = useState<LayerState<P>>(LOADING)

  useEffect(() => {
    let latest = 0
    let controller: AbortController | null = null

    const load = async () => {
      controller?.abort() // a slow earlier request must not overwrite a newer answer
      const mine = new AbortController()
      controller = mine
      const run = ++latest
      const payload = await fetchLayer<P>(path, hasProps, mine.signal)
      if (run === latest && !mine.signal.aborted) setState((previous) => nextState(previous, payload))
    }

    void load()
    const timer = setInterval(() => void load(), REFETCH_MS)
    return () => {
      clearInterval(timer)
      controller?.abort()
      latest += 1
    }
  }, [path, hasProps])

  return state
}

/** The two base layers' data: fetched on load and every 15 minutes. A failure of one never touches the other. */
export function useMapLayers(): { airspace: LayerState<AirspaceProps>; notams: LayerState<NotamProps> } {
  const airspace = useRoute<AirspaceProps>(AIRSPACE_PATH, hasAirspaceProps)
  const notams = useRoute<NotamProps>(NOTAMS_PATH, hasNotamProps)
  return { airspace, notams }
}

/** "HH:MM" of an ISO time in the browser's time zone (24 hour clock), or "" when it is not a time. */
export function formatUpdated(iso: string): string {
  const when = new Date(iso)
  if (Number.isNaN(when.getTime())) return ''
  return new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }).format(when)
}
